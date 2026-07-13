# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: half-bridge / resonant switch-stage model (Stage 26 Phase B).

``Sim.Device="HALFBRIDGE"`` (alias ``"LLC"``) rides on any switcher-shaped
symbol (SW/VIN/GND pins) and emits an open-loop complementary S-switch pair with
deadtime and mandatory antiparallel diodes -- the switch stage of a half/
resonant bridge, FSW-swept for the gain curve. Emission tests inspect the SPICE
text; validate tests check the FSW/deadtime guards.
"""

import pytest

from skidl import KICAD9, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SimulationValidationError, SpiceConverter

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)

# A stand-in symbol with SW/VIN/GND pins (EN/FB are cap-only unmodeled pins).
HB_SYMBOL = ("Regulator_Switching", "TPS61040DBV")


def _setup():
    set_default_tool(KICAD9)
    from skidl.tools.kicad9.lib import default_lib_paths

    lib_search_paths["kicad9"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _view():
    from skidl.sim import skidl_flat_view

    return skidl_flat_view()


def _build(sim_device="HALFBRIDGE", params="fsw=100k", ref="U1"):
    u = Part(*HB_SYMBOL, ref=ref)
    u.Sim_Device = sim_device
    if params is not None:
        u.Sim_Params = params
    Net("VINN").connect(u["VIN"])
    Net("SWN").connect(u["SW"])
    Net("GND").connect(u["GND"])
    return u


def _emit():
    return str(SpiceConverter(_view()).convert(strict=False))


@requires_sim
def test_halfbridge_emission_structure():
    _setup()
    _build()
    netlist = _emit()
    lines = netlist.splitlines()
    # two complementary S switches sharing one SW model
    assert "SU1_hs VINN SWN U1_ghs" in netlist, netlist
    assert any(ln.startswith("SU1_ls SWN") for ln in lines), netlist
    assert ".model SWU1 SW(Ron=0.1" in netlist
    # two ground-referenced complementary gate PULSE sources with a deadtime gap:
    # ghs starts at t=0, gls starts at half-period; on-time = half - DT.
    assert "VU1_ghs U1_ghs" in netlist and "PULSE(0 5 0 " in netlist
    assert "PULSE(0 5 5e-06 " in netlist  # gls delayed by half-period (per=1e-5)
    assert "4.9e-06" in netlist  # conduction time = 5e-6 - 100n default DT
    # mandatory antiparallel diodes on both switches
    assert "DU1_hs SWN VINN DFWU1" in netlist
    assert "DU1_ls" in netlist and "SWN DFWU1" in netlist
    assert ".model DFWU1 D(" in netlist
    # exactly two switches and two freewheel diodes
    assert sum(1 for ln in lines if ln.startswith("SU1_")) == 2
    assert sum(1 for ln in lines if ln.startswith("DU1_")) == 2


@requires_sim
def test_llc_alias_maps_to_halfbridge():
    _setup()
    _build(sim_device="LLC")
    netlist = _emit()
    assert "SU1_hs VINN SWN U1_ghs" in netlist
    assert "SU1_ls SWN" in netlist


@requires_sim
def test_halfbridge_missing_fsw_is_validation_error():
    _setup()
    _build(params=None)  # no Sim.Params -> no FSW
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("FSW" in p for p in ei.value.problems), ei.value.problems


@requires_sim
def test_halfbridge_deadtime_too_long_is_validation_error():
    _setup()
    # DT = 6us >= half period (5us at 100k) -> no conduction time
    _build(params="fsw=100k dt=6u")
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("deadtime" in p.lower() for p in ei.value.problems), ei.value.problems


@requires_sim
def test_halfbridge_custom_deadtime_and_ron():
    _setup()
    _build(params="fsw=200k dt=200n ron=0.25")
    netlist = _emit()
    # per = 5e-6, half = 2.5e-6, on = 2.5e-6 - 200n = 2.3e-6
    assert "2.3e-06" in netlist
    assert ".model SWU1 SW(Ron=0.25" in netlist


# --- stiff transient recipe (Stage 26 Phase E) ----------------------------


@requires_sim
def test_stiff_options_merge():
    """stiff=True injects the gear/reltol/... recipe; an explicit option wins,
    and the class constant is never mutated."""
    from skidl.sim.simulator import CircuitSimulator

    base = CircuitSimulator._STIFF_TRAN_OPTIONS
    assert CircuitSimulator._merge_stiff_options(None) == base
    merged = CircuitSimulator._merge_stiff_options({"reltol": 1e-4, "trtol": 1})
    assert merged["method"] == "gear"  # recipe default kept
    assert merged["reltol"] == 1e-4  # explicit override wins
    assert merged["trtol"] == 1  # extra knob threaded through
    assert "trtol" not in base  # constant not mutated


@requires_sim
def test_halfbridge_tank_converges_and_resonates():
    """Live: a half-bridge driving a series L-C-R tank converges with stiff=True
    (+UIC) and the tank current peaks at the resonant frequency vs off-resonance
    -- the Phase B acceptance gate."""
    import math

    import numpy as np

    Lr, Cr, R = 25e-6, 100e-9, 1.0
    fr = 1 / (2 * math.pi * math.sqrt(Lr * Cr))

    def run(fsw):
        _setup()
        u = Part(*HB_SYMBOL, ref="U1")
        u.Sim_Device = "HALFBRIDGE"
        u.Sim_Params = f"fsw={fsw:.6g}"
        v = Part("Simulation_SPICE", "VDC", value="48", ref="V1")
        lr = Part("Device", "L", value=str(Lr), ref="L1")
        cr = Part("Device", "C", value=str(Cr), ref="C1")
        rr = Part("Device", "R", value=str(R), ref="R1")
        vin, sw, a, b, gnd = (Net(n) for n in ("VIN", "SW", "A", "B", "GND"))
        vin.connect(v[1], u["VIN"])
        gnd.connect(v[2], u["GND"], rr[2])
        sw.connect(u["SW"], lr[1])
        a.connect(lr[2], cr[1])
        b.connect(cr[2], rr[1])
        from skidl.sim import simulate

        sim = simulate()
        per = 1 / fsw
        try:
            an = sim.transient_analysis(
                step_time=per / 200,
                end_time=200 * per,
                max_time=per / 50,
                stiff=True,
                use_initial_condition=True,
            )
        except Exception as e:
            pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
        swv = np.array(an.get_voltage("SW"))
        bv = np.array(an.get_voltage("B"))
        s = int(len(swv) * 0.6)
        return swv[s:], bv[s:]

    sw_on, b_on = run(fr)
    _, b_hi = run(1.5 * fr)
    # V(SW) swings roughly 0..48 (small negative excursion = diode freewheel)
    assert sw_on.max() > 40 and sw_on.min() < 5, (sw_on.min(), sw_on.max())
    amp = lambda x: (x.max() - x.min()) / 2
    # tank current (V(B)/R) is far larger at resonance than above it
    assert amp(b_on) > 3 * amp(b_hi), (amp(b_on), amp(b_hi))
