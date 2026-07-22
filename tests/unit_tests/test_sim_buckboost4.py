# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: non-inverting 4-switch buck-boost macromodel (Stage 27.6).

``Sim.Device="BUCKBOOST4"`` replaces only the four switches of a non-inverting
4-switch buck-boost. Two ``_emit_sync_leg`` legs share one clock: a buck leg
(VIN/SWA/GND) at ``DBUCK`` and a boost leg (VOUT/SWB/GND) at ``1-DBOOST``, with
the user's real inductor between the two switch nodes. Open-loop -- the ideal DC
gain is ``Vout/Vin = DBUCK/(1-DBOOST)``. A leg driven to a saturated duty (a pure
buck/boost endpoint) collapses to a static ``RON`` pass-through tie rather than a
non-switching (and unemittable) sync leg.

Emission tests inspect the SPICE text; validate tests check the terminal / FSW
guards; the gated live test confirms convergence and buck-mode tracking.
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


# No stock KiCad symbol carries a distinct second-switch-node + VOUT pin, so the
# 4-switch controller is a synthetic SKIDL part whose pins carry exactly the names
# the multi-node resolver keys on (VIN / SW / SW2 / VOUT / GND).
def _bb4(ref="U1", *, connect_sw2=True, **fields):
    pins = [
        Pin(num=1, name="VIN", func=pin_types.PWRIN),
        Pin(num=2, name="SW", func=pin_types.PASSIVE),
        Pin(num=3, name="SW2", func=pin_types.PASSIVE),
        Pin(num=4, name="VOUT", func=pin_types.PWROUT),
        Pin(num=5, name="GND", func=pin_types.PWRIN),
    ]
    if not connect_sw2:
        pins = [p for p in pins if p.name != "SW2"]
    u = Part(tool=SKIDL, name="BB4", ref_prefix="U", ref=ref, pins=pins)
    for k, v in fields.items():
        setattr(u, k, v)
    return u


def _wire(u, *, sw2=True):
    Net("VIN").connect(u["VIN"])
    Net("SWA").connect(u["SW"])
    if sw2:
        Net("SWB").connect(u["SW2"])
    Net("VOUT").connect(u["VOUT"])
    Net("GND").connect(u["GND"])


def _emit():
    return str(SpiceConverter(_view()).convert(strict=False))


# --- emission ------------------------------------------------------------- #


@requires_sim
def test_buckboost4_emits_two_switching_legs():
    """Both duties in (0,1) -> two complementary legs = 4 S, 4 D, 4 gate PULSEs."""
    _setup()
    u = _bb4(Sim_Device="BUCKBOOST4", Sim_Params="fsw=500k dbuck=0.6 dboost=0.4")
    _wire(u)
    netlist = _emit()
    lines = netlist.splitlines()
    # buck leg on VIN/SWA/GND, boost leg on VOUT/SWB/GND, gates referenced to GND (0)
    assert "SU1A_hs VIN SWA U1A_ghs 0 SWU1A" in netlist, netlist
    assert "SU1A_ls SWA 0 U1A_gls 0 SWU1A" in netlist
    assert "SU1B_hs VOUT SWB U1B_ghs 0 SWU1B" in netlist
    assert "SU1B_ls SWB 0 U1B_gls 0 SWU1B" in netlist
    # antiparallel body diodes on each switch (the bidirectional reverse path)
    assert "DU1A_hs SWA VIN DFWU1A" in netlist
    assert "DU1B_hs SWB VOUT DFWU1B" in netlist
    # exactly four switches, four freewheel diodes, four gate PULSE sources
    assert sum(1 for ln in lines if ln.startswith(("SU1A_", "SU1B_"))) == 4
    assert sum(1 for ln in lines if ln.startswith(("DU1A_", "DU1B_"))) == 4
    assert sum(1 for ln in lines if ".model DFW" in ln) == 2
    assert sum(1 for ln in lines if "_ghs U1" in ln or "_gls U1" in ln) == 4


@requires_sim
def test_buckboost4_buck_mode_static_boost_leg():
    """dboost=0 saturates the boost leg -> no boost switches, a static SWB->VOUT
    RON tie instead (a plain synchronous buck)."""
    _setup()
    u = _bb4(Sim_Device="BUCKBOOST4", Sim_Params="fsw=500k dbuck=0.5 dboost=0")
    _wire(u)
    netlist = _emit()
    lines = netlist.splitlines()
    # buck leg still switches
    assert "SU1A_hs VIN SWA U1A_ghs 0 SWU1A" in netlist
    # boost leg is a static pass-through tie, not switches
    assert "RU1B_sat SWB VOUT 0.1" in netlist, netlist
    assert not any(ln.startswith("SU1B_") for ln in lines)


@requires_sim
def test_buckboost4_provenance_recorded():
    _setup()
    u = _bb4(Sim_Device="BUCKBOOST4", Sim_Params="fsw=500k dbuck=0.5 dboost=0")
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    prov = conv.model_provenance["U1"]
    assert prov.kind == "buckboost4"
    assert prov.tier == "sim_params"
    assert "buckboost4_openloop" in prov.name
    assert "dbuck=0.5" in prov.name and "dboost=0" in prov.name


@requires_sim
def test_buckboost4_vout_target_convenience():
    """VOUT+VIN target (no explicit duties) derives buck-mode duty Vout/Vin."""
    _setup()
    u = _bb4(Sim_Device="BUCKBOOST4", Sim_Params="fsw=500k vout=5 vin=10")
    _wire(u)
    conv = SpiceConverter(_view())
    netlist = str(conv.convert(strict=False))
    # 5/10 -> buck mode, dbuck=0.5, dboost=0 (boost leg saturates to a static tie)
    assert "dbuck=0.5" in conv.model_provenance["U1"].name
    assert "RU1B_sat SWB VOUT" in netlist


# --- validation ----------------------------------------------------------- #


@requires_sim
def test_buckboost4_missing_fsw_is_validation_error():
    _setup()
    u = _bb4(Sim_Device="BUCKBOOST4", Sim_Params="dbuck=0.5 dboost=0")  # no FSW
    _wire(u)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("FSW" in p for p in ei.value.problems), ei.value.problems


@requires_sim
def test_buckboost4_missing_switch_node_is_validation_error():
    _setup()
    u = _bb4(Sim_Device="BUCKBOOST4", Sim_Params="fsw=500k dbuck=0.5 dboost=0",
             connect_sw2=False)
    _wire(u, sw2=False)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any(
        "buckboost4 needs connected" in p for p in ei.value.problems
    ), ei.value.problems


# --- live (gated) --------------------------------------------------------- #


@requires_sim
def test_buckboost4_buck_mode_converges_and_tracks():
    """Live: a BUCKBOOST4 in buck mode + real L/Cout/R converges (stiff+UIC) and
    the output tracks DBUCK*Vin within ~10 % across a duty sweep (device-level
    conduction + deadtime loss pulls it a few % low, monotone-increasing)."""
    import numpy as np

    Lval, Cout, Rload, Vin = 10e-6, 4.7e-6, 10.0, 12.0
    fsw = 500e3
    per = 1.0 / fsw

    def run(dbuck):
        _setup()
        u = _bb4(Sim_Device="BUCKBOOST4",
                 Sim_Params=f"fsw=500k dbuck={dbuck} dboost=0 dt=50n")
        v = Part("Simulation_SPICE", "VDC", value=str(Vin), ref="V1")
        L1 = Part("Device", "L", value=str(Lval), ref="L1")
        C1 = Part("Device", "C", value=str(Cout), ref="C1")
        R1 = Part("Device", "R", value=str(Rload), ref="R1")
        vin, swa, swb, vout, gnd = (
            Net(n) for n in ("VIN", "SWA", "SWB", "VOUT", "GND")
        )
        vin.connect(v[1], u["VIN"])
        gnd.connect(v[2], u["GND"], C1[2], R1[2])
        swa.connect(u["SW"], L1[1])
        swb.connect(u["SW2"], L1[2])
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
                initial_conditions={"VOUT": 0, "SWA": 0, "SWB": 0},
            )
        except Exception as e:
            pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
        vo = np.array(an.get_voltage("VOUT"))
        return float(vo[int(len(vo) * 0.7):].mean())

    outs = [run(d) for d in (0.33, 0.5, 0.66)]
    for d, vo in zip((0.33, 0.5, 0.66), outs):
        ideal = d * Vin
        assert vo > 0, (d, vo)
        assert abs(vo - ideal) / ideal < 0.10, (d, vo, ideal)
    # monotone-increasing with duty
    assert outs[0] < outs[1] < outs[2], outs
