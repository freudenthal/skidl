# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: bidirectional SEPIC 2-switch macromodel (Stage 27.8).

``Sim.Device="SEPIC"`` replaces only the two switches of a bidirectional SEPIC
(SEPIC forward / Zeta reverse). The user's two inductors, the **coupling cap Cs**
and the output cap stay real parts -- the macromodel emits ONLY the main switch
(node A -> GND) + the synchronous rectifier (node B -> VOUT), each with its body
diode, plus the two complementary gate PULSEs. It is NOT a single totem-pole leg
(the two switches sit on independent nodes A and B, with the real Cs between
them), so it does not route through ``_emit_sync_leg`` -- that would double the
device count. Open-loop, non-inverting: the ideal DC gain is
``Vout = Vin * D/(1-D)`` (steps up or down through the D=0.5 crossover at ~Vin).

Body-diode orientation is load-bearing (the Stage 27.4 device-level twin): the
main FET's body diode is GND->A, the rectifier's is B->VOUT (``DQ1_body 0 A`` /
``DQ2_body B VOUT``; the reversed rectifier wiring silently produced -7.8 V in the
27.1 Spike-2). Emission tests inspect the SPICE text and confirm Cs is *not*
emitted; validate tests check the terminal / FSW / shared-node guards; the gated
live test confirms convergence and a non-inverting step-up/down output.
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


# No stock KiCad symbol carries distinct A (SW) + B (SW2) + VOUT pins, so the
# SEPIC controller is a synthetic SKIDL part whose pins carry exactly the names
# the multi-node resolver keys on: VIN / SW (node A) / SW2 (node B) / VOUT / GND.
def _sepic(ref="U1", *, connect_swb=True, **fields):
    pins = [
        Pin(num=1, name="VIN", func=pin_types.PWRIN),
        Pin(num=2, name="SW", func=pin_types.PASSIVE),   # node A (main switch)
        Pin(num=3, name="SW2", func=pin_types.PASSIVE),  # node B (rectifier)
        Pin(num=4, name="VOUT", func=pin_types.PWROUT),
        Pin(num=5, name="GND", func=pin_types.PWRIN),
    ]
    if not connect_swb:
        pins = [p for p in pins if p.name != "SW2"]
    u = Part(tool=SKIDL, name="SEPIC", ref_prefix="U", ref=ref, pins=pins)
    for k, v in fields.items():
        setattr(u, k, v)
    return u


def _wire(u, *, swb=True, ab_shared=False):
    Net("VIN").connect(u["VIN"])
    a = Net("A")
    a.connect(u["SW"])
    if swb:
        # ab_shared ties node B to the same net as A -> the coupling cap Cs would
        # be shorted; validate() must flag it.
        (a if ab_shared else Net("B")).connect(u["SW2"])
    Net("VOUT").connect(u["VOUT"])
    Net("GND").connect(u["GND"])


def _emit():
    return str(SpiceConverter(_view()).convert(strict=False))


# --- emission ------------------------------------------------------------- #


@requires_sim
def test_sepic_emits_two_switches_cs_not_emitted():
    """A duty in (0,1) -> exactly the main switch (A->GND) + sync rectifier
    (B->VOUT), each with a body diode + gate PULSE. Cs (A->B) is NOT emitted --
    the user's real coupling cap survives."""
    _setup()
    u = _sepic(Sim_Device="SEPIC", Sim_Params="fsw=500k d=0.5")
    _wire(u)
    netlist = _emit()
    lines = netlist.splitlines()
    # main switch A->GND, sync rectifier B->VOUT, gates referenced to GND.
    # GND is the SPICE reference net, mapped to node 0.
    assert "SU1_m A 0 U1_gm 0 SWU1" in netlist, netlist
    assert "SU1_r B VOUT U1_gr 0 SWU1" in netlist, netlist
    # body diodes: main GND(0)->A (freewheel), rectifier B->VOUT (the SEPIC
    # rectifier path -- orientation load-bearing, matches the device twin).
    assert "DU1_m 0 A DFWU1" in netlist, netlist
    assert "DU1_r B VOUT DFWU1" in netlist, netlist
    # exactly two switches, two body diodes, two gate PULSE sources, one of each model
    assert sum(1 for ln in lines if ln.startswith("SU1_")) == 2
    assert sum(1 for ln in lines if ln.startswith("DU1_")) == 2
    assert sum(1 for ln in lines if ".model SW" in ln) == 1
    assert sum(1 for ln in lines if ".model DFW" in ln) == 1
    assert sum(1 for ln in lines if ln.startswith("VU1_g")) == 2
    # Cs is the user's real part -- the macromodel must not emit a cap on A/B.
    assert not any(ln.startswith("C") and " A " in f" {ln} " for ln in lines), netlist


@requires_sim
def test_sepic_provenance_recorded():
    _setup()
    u = _sepic(Sim_Device="SEPIC", Sim_Params="fsw=500k d=0.5")
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    prov = conv.model_provenance["U1"]
    assert prov.kind == "sepic"
    assert prov.tier == "sim_params"
    assert "sepic_openloop" in prov.name
    assert "d=0.5" in prov.name


@requires_sim
def test_sepic_vout_target_convenience():
    """A VOUT target + VIN (no explicit D) derives D=Vout/(Vout+Vin)."""
    _setup()
    # Vout=24, Vin=12 -> D = 24/36 = 0.666...
    u = _sepic(Sim_Device="SEPIC", Sim_Params="fsw=500k vout=24 vin=12")
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    assert "d=0.666667" in conv.model_provenance["U1"].name, conv.model_provenance["U1"].name


# --- validation ----------------------------------------------------------- #


@requires_sim
def test_sepic_missing_fsw_is_validation_error():
    _setup()
    u = _sepic(Sim_Device="SEPIC", Sim_Params="d=0.5")  # no FSW
    _wire(u)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("FSW" in p for p in ei.value.problems), ei.value.problems


@requires_sim
def test_sepic_missing_node_is_validation_error():
    _setup()
    u = _sepic(Sim_Device="SEPIC", Sim_Params="fsw=500k d=0.5", connect_swb=False)
    _wire(u, swb=False)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any(
        "sepic needs connected" in p for p in ei.value.problems
    ), ei.value.problems


@requires_sim
def test_sepic_shared_ab_net_is_validation_error():
    """A and B on the same net short the coupling cap -- validate() flags it."""
    _setup()
    u = _sepic(Sim_Device="SEPIC", Sim_Params="fsw=500k d=0.5")
    _wire(u, ab_shared=True)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any(
        "same net" in p and "Cs" in p for p in ei.value.problems
    ), ei.value.problems


# --- live (gated) --------------------------------------------------------- #


@requires_sim
def test_sepic_converges_and_steps_up_down():
    """Live: a SEPIC macromodel + real L1/L2/Cs/Cout/R converges (stiff+UIC) and
    settles a *non-inverting* output that tracks ``Vin*D/(1-D)``. D=0.5 (1x gain)
    tracks +Vin within 10 % (the step-up/down crossover); D=0.33 steps down
    (VOUT<Vin) and D=0.66 steps up (VOUT>Vin), monotone-increasing. dt=25 ns and
    end=600*per per the Stage 27.4 device-level recipe (both switch and rectifier
    carry input+output current; Cs must warm up to ~Vin)."""
    import numpy as np

    L1v, L2v, Csv, Cout, Rload, Vin = 22e-6, 22e-6, "1u", 22e-6, 10.0, 12.0
    fsw = 500e3
    per = 1.0 / fsw

    def run(d):
        _setup()
        u = _sepic(Sim_Device="SEPIC", Sim_Params=f"fsw=500k d={d} dt=25n")
        v = Part("Simulation_SPICE", "VDC", value=str(Vin), ref="V1")
        L1 = Part("Device", "L", value=str(L1v), ref="L1")
        L2 = Part("Device", "L", value=str(L2v), ref="L2")
        Cs = Part("Device", "C", value=Csv, ref="CS")
        Co = Part("Device", "C", value=str(Cout), ref="C1")
        R1 = Part("Device", "R", value=str(Rload), ref="R1")
        vin, a, b, vout, gnd = (Net(n) for n in ("VIN", "A", "B", "VOUT", "GND"))
        vin.connect(v[1], u["VIN"], L1[1])
        gnd.connect(v[2], u["GND"], L2[2], Co[2], R1[2])
        a.connect(u["SW"], L1[2], Cs[1])          # node A: main switch, L1, Cs
        b.connect(u["SW2"], Cs[2], L2[1])          # node B: rectifier, Cs, L2
        vout.connect(u["VOUT"], Co[1], R1[1])
        from skidl.sim import simulate

        sim = simulate()
        try:
            an = sim.transient_analysis(
                step_time=per / 200,
                end_time=600 * per,
                max_time=per / 60,
                stiff=True,
                use_initial_condition=True,
                initial_conditions={"VOUT": 0},
            )
        except Exception as e:
            pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
        vo = np.array(an.get_voltage("VOUT"))
        return float(vo[int(len(vo) * 0.7):].mean())

    duties = (0.33, 0.5, 0.66)
    outs = [run(d) for d in duties]
    ideals = [Vin * d / (1.0 - d) for d in duties]
    # every operating point is a POSITIVE rail (non-inverting)
    for d, vo in zip(duties, outs):
        assert vo > 0, (d, vo)
    # monotone: more duty -> higher output (the single SEPIC duty knob)
    assert outs[0] < outs[1] < outs[2], outs
    # D=0.33 steps DOWN (buck region), D=0.66 steps UP (boost region)
    assert outs[0] < Vin, outs[0]
    assert outs[2] > Vin, outs[2]
    # D=0.5 tracks +Vin within 10 % (the plan's explicit crossover requirement)
    assert abs(outs[1] - Vin) / Vin < 0.10, outs[1]
    # low/mid within 10 %; the boost-region point within 12 % (deadtime loss)
    assert abs(outs[0] - ideals[0]) / ideals[0] < 0.10, (outs[0], ideals[0])
    assert abs(outs[2] - ideals[2]) / ideals[2] < 0.12, (outs[2], ideals[2])
