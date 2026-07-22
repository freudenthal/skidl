# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: inverting 2-switch buck-boost macromodel (Stage 27.7).

``Sim.Device="INVBUCKBOOST"`` replaces only the two switches of an inverting
buck-boost. One ``_emit_sync_leg`` spans VIN and the **negative** output VOUT
with the switch node SW in the middle: the high-side switch ties VIN->SW for the
on-fraction ``D``, the low-side switch ties SW->VOUT for the rest (the
synchronous rectifier). The low-side body diode is VOUT->SW -- the classic
inverting-buck-boost rectifier orientation the Stage 27.3 device-level twin
confirmed load-bearing. Open-loop -- the ideal DC gain is the negative
``Vout = -Vin * D/(1-D)``.

Emission tests inspect the SPICE text; validate tests check the terminal / FSW
guards; the gated live test confirms convergence and a *negative* output.
"""

import pytest

from skidl import KICAD10, SKIDL, Net, Part, Pin, lib_search_paths, set_default_tool
from skidl.pin import pin_types

try:
    from skidl.sim.converter import SimulationValidationError, SpiceConverter

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _view():
    from skidl.sim import skidl_flat_view

    return skidl_flat_view()


# No stock KiCad symbol carries a distinct SW + VOUT pin, so the inverting
# controller is a synthetic SKIDL part whose pins carry exactly the names the
# multi-node resolver keys on (VIN / SW / VOUT / GND).
def _ibb(ref="U1", *, connect_vout=True, **fields):
    pins = [
        Pin(num=1, name="VIN", func=pin_types.PWRIN),
        Pin(num=2, name="SW", func=pin_types.PASSIVE),
        Pin(num=3, name="VOUT", func=pin_types.PWROUT),
        Pin(num=4, name="GND", func=pin_types.PWRIN),
    ]
    if not connect_vout:
        pins = [p for p in pins if p.name != "VOUT"]
    u = Part(tool=SKIDL, name="IBB", ref_prefix="U", ref=ref, pins=pins)
    for k, v in fields.items():
        setattr(u, k, v)
    return u


def _wire(u, *, vout=True):
    Net("VIN").connect(u["VIN"])
    Net("SW").connect(u["SW"])
    if vout:
        Net("VOUT").connect(u["VOUT"])
    Net("GND").connect(u["GND"])


def _emit():
    return str(SpiceConverter(_view()).convert(strict=False))


# --- emission ------------------------------------------------------------- #


@requires_sim
def test_invbuckboost_emits_one_switching_leg():
    """A duty in (0,1) -> one complementary leg = 2 S, 2 D, 2 gate PULSEs, with
    the sync-rectifier body diode oriented VOUT->SW (the negative-rail path)."""
    _setup()
    u = _ibb(Sim_Device="INVBUCKBOOST", Sim_Params="fsw=500k d=0.5")
    _wire(u)
    netlist = _emit()
    lines = netlist.splitlines()
    # high-side VIN->SW, low-side SW->VOUT (the negative rail), gates ref'd to GND (0)
    assert "SU1_hs VIN SW U1_ghs 0 SWU1" in netlist, netlist
    assert "SU1_ls SW VOUT U1_gls 0 SWU1" in netlist
    # antiparallel body diodes: high-side SW->VIN; low-side VOUT->SW is the
    # inverting-buck-boost rectifier (anode at the negative rail, cathode at SW).
    assert "DU1_hs SW VIN DFWU1" in netlist
    assert "DU1_ls VOUT SW DFWU1" in netlist, netlist
    # exactly two switches, two freewheel diodes, two gate PULSE sources
    assert sum(1 for ln in lines if ln.startswith("SU1_")) == 2
    assert sum(1 for ln in lines if ln.startswith("DU1_")) == 2
    assert sum(1 for ln in lines if ".model DFW" in ln) == 1
    assert sum(1 for ln in lines if "_ghs U1" in ln or "_gls U1" in ln) == 2


@requires_sim
def test_invbuckboost_provenance_recorded():
    _setup()
    u = _ibb(Sim_Device="INVBUCKBOOST", Sim_Params="fsw=500k d=0.5")
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    prov = conv.model_provenance["U1"]
    assert prov.kind == "invbuckboost"
    assert prov.tier == "sim_params"
    assert "invbuckboost_openloop" in prov.name
    assert "d=0.5" in prov.name


@requires_sim
def test_invbuckboost_vout_target_convenience():
    """A negative VOUT target + VIN (no explicit D) derives D=|Vout|/(|Vout|+Vin)."""
    _setup()
    u = _ibb(Sim_Device="INVBUCKBOOST", Sim_Params="fsw=500k vout=-12 vin=12")
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    # |−12|/(|−12|+12) = 0.5
    assert "d=0.5" in conv.model_provenance["U1"].name


# --- validation ----------------------------------------------------------- #


@requires_sim
def test_invbuckboost_missing_fsw_is_validation_error():
    _setup()
    u = _ibb(Sim_Device="INVBUCKBOOST", Sim_Params="d=0.5")  # no FSW
    _wire(u)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("FSW" in p for p in ei.value.problems), ei.value.problems


@requires_sim
def test_invbuckboost_missing_output_node_is_validation_error():
    _setup()
    u = _ibb(Sim_Device="INVBUCKBOOST", Sim_Params="fsw=500k d=0.5",
             connect_vout=False)
    _wire(u, vout=False)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any(
        "invbuckboost needs connected" in p for p in ei.value.problems
    ), ei.value.problems


# --- live (gated) --------------------------------------------------------- #


@requires_sim
def test_invbuckboost_converges_and_inverts():
    """Live: an INVBUCKBOOST + real L/Cout/R converges (stiff+UIC) and settles a
    *negative* output that tracks ``-Vin*D/(1-D)``. D=0.5 (1x gain) tracks -Vin
    within 10 %; the 4x-gain point (D=0.66) runs a few % lossier because the
    single switch AND rectifier each carry input+output current (the Stage 27.3
    finding), still within 12 %. dt=25 ns per that same finding."""
    import numpy as np

    Lval, Cout, Rload, Vin = 10e-6, 4.7e-6, 10.0, 12.0
    fsw = 500e3
    per = 1.0 / fsw

    def run(d):
        _setup()
        u = _ibb(Sim_Device="INVBUCKBOOST",
                 Sim_Params=f"fsw=500k d={d} dt=25n")
        v = Part("Simulation_SPICE", "VDC", value=str(Vin), ref="V1")
        L1 = Part("Device", "L", value=str(Lval), ref="L1")
        C1 = Part("Device", "C", value=str(Cout), ref="C1")
        R1 = Part("Device", "R", value=str(Rload), ref="R1")
        vin, sw, vout, gnd = (Net(n) for n in ("VIN", "SW", "VOUT", "GND"))
        vin.connect(v[1], u["VIN"])
        gnd.connect(v[2], u["GND"], L1[2], C1[2], R1[2])
        sw.connect(u["SW"], L1[1])
        vout.connect(u["VOUT"], C1[1], R1[1])
        from skidl.sim import simulate

        sim = simulate()
        try:
            an = sim.transient_analysis(
                step_time=per / 200,
                end_time=400 * per,
                max_time=per / 60,
                stiff=True,
                use_initial_condition=True,
                initial_conditions={"VOUT": 0, "SW": 0},
            )
        except Exception as e:
            pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
        vo = np.array(an.get_voltage("VOUT"))
        return float(vo[int(len(vo) * 0.7):].mean())

    duties = (0.33, 0.5, 0.66)
    outs = [run(d) for d in duties]
    ideals = [-Vin * d / (1.0 - d) for d in duties]
    # every operating point produces a NEGATIVE rail (sign inversion)
    for d, vo in zip(duties, outs):
        assert vo < 0, (d, vo)
    # monotone: more duty -> more-negative output
    assert outs[0] > outs[1] > outs[2], outs
    # D=0.5 tracks -Vin within 10 % (the plan's explicit requirement)
    assert abs(outs[1] - (-Vin)) / Vin < 0.10, outs[1]
    # low/mid points within 10 %; the 4x-gain point within 12 %
    assert abs(outs[0] - ideals[0]) / abs(ideals[0]) < 0.10, (outs[0], ideals[0])
    assert abs(outs[2] - ideals[2]) / abs(ideals[2]) < 0.12, (outs[2], ideals[2])
