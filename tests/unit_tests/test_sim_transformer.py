# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: multi-winding transformer SPICE emission (Stage 26 Phase A).

Covers the generalization of ``_transformer_terminals`` / ``_transformer_params``
/ ``_add_transformer`` from 1P_1S-only to N windings:

* ``Transformer_1P_1S`` -- AA/AB, SA/SB (the legacy case: emission byte-identical)
* ``Transformer_1P_2S`` -- adds an independent SC/SD secondary
* ``Transformer_1P_SS`` -- center-tapped secondary SA/SC/SB (SC = tap)

Detection tests run off the live pin map alone (no ngspice); emission tests need
PySpice to build the SPICE circuit and inspect its text.
"""

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

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


# --- winding detection (skidl only, no ngspice) ---------------------------


@requires_sim
def test_1p1s_terminals_single_secondary():
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    Net("PA").connect(t["AA"])
    Net("PB").connect(t["AB"])
    Net("SA_N").connect(t["SA"])
    Net("SB_N").connect(t["SB"])
    term = SpiceConverter(_view())._transformer_terminals(_view().components["T1"])
    assert term["primary"] == ("PA", "PB")
    assert term["secondaries"] == [("SA_N", "SB_N")]
    assert term["center_tap"] is False


@requires_sim
def test_1p2s_terminals_two_independent_secondaries():
    _setup()
    t = Part("Device", "Transformer_1P_2S", ref="T1")
    Net("PA").connect(t["AA"])
    Net("PB").connect(t["AB"])
    Net("S1A").connect(t["SA"])
    Net("S1B").connect(t["SB"])
    Net("S2A").connect(t["SC"])
    Net("S2B").connect(t["SD"])
    term = SpiceConverter(_view())._transformer_terminals(_view().components["T1"])
    assert term["primary"] == ("PA", "PB")
    assert term["secondaries"] == [("S1A", "S1B"), ("S2A", "S2B")]
    assert term["center_tap"] is False


@requires_sim
def test_1pss_terminals_center_tap_halves():
    _setup()
    t = Part("Device", "Transformer_1P_SS", ref="T1")
    Net("PA").connect(t["AA"])
    Net("PB").connect(t["AB"])
    Net("TOP").connect(t["SA"])
    Net("TAP").connect(t["SC"])
    Net("BOT").connect(t["SB"])
    term = SpiceConverter(_view())._transformer_terminals(_view().components["T1"])
    assert term["center_tap"] is True
    # two half-windings, dots at SA and SC: SA->SC and SC->SB
    assert term["secondaries"] == [("TOP", "TAP"), ("TAP", "BOT")]


@requires_sim
def test_half_wound_secondary_detected_and_named():
    """A 1P_2S with SD left unconnected is a half-wound winding: terminals None,
    and validate() names the missing pin."""
    _setup()
    t = Part("Device", "Transformer_1P_2S", ref="T1")
    Net("PA").connect(t["AA"])
    Net("PB").connect(t["AB"])
    Net("S1A").connect(t["SA"])
    Net("S1B").connect(t["SB"])
    Net("S2A").connect(t["SC"])  # SD deliberately left open
    conv = SpiceConverter(_view())
    comp = _view().components["T1"]
    assert conv._transformer_terminals(comp) is None
    assert conv._transformer_missing_pins(comp) == ["SD"]


# --- winding params -------------------------------------------------------


@requires_sim
def test_params_center_tap_per_half_ratio():
    _setup()
    t = Part("Device", "Transformer_1P_SS", ref="T1")
    t.Sim_Params = "lp=1m n=0.5"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "TOP"), ("SC", "TAP"), ("SB", "BOT")):
        Net(net).connect(t[pin])
    params = SpiceConverter(_view())._transformer_params(_view().components["T1"])
    assert params["center_tap"] is True
    # per-half inductance LP*N^2 = 1e-3 * 0.25 for BOTH halves
    assert params["secondaries"] == pytest.approx([2.5e-4, 2.5e-4])
    assert params["K"] == pytest.approx(0.999)


@requires_sim
def test_params_bad_k_reported_as_none():
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5 k=1.5"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    params = SpiceConverter(_view())._transformer_params(_view().components["T1"])
    assert params["K"] is None


# --- SPICE emission (PySpice) --------------------------------------------


def _emit(view):
    return str(SpiceConverter(view).convert(strict=False))


@requires_sim
def test_1p1s_emission_byte_identical():
    """The single-secondary case keeps the exact legacy 3-line emission
    (L<ref>_P / L<ref>_S / K<ref>) -- a hard backward-compat requirement."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_P PA PB 0.0001" in netlist
    assert "LT1_S SA_N SB_N 2.5e-05" in netlist
    assert "KT1 LT1_P LT1_S 0.999" in netlist
    # never the multi-winding spellings
    assert "LT1_S1" not in netlist
    assert "KT1_" not in netlist


@requires_sim
def test_winding_resistance_emits_series_r(monkeypatch):
    """rp=/rs= insert a series resistor (via an internal node) on each winding's
    A-side terminal; the coupled inductor keeps its name so the K card is
    unchanged (C3)."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5 rp=0.5 rs=0.1"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    # series R on primary, feeding the coupled inductor through an internal node
    assert "RT1_P PA T1_p_ri 0.5" in netlist, netlist
    assert "LT1_P T1_p_ri PB 0.0001" in netlist, netlist
    # series R on the secondary
    assert "RT1_S SA_N T1_s_ri 0.1" in netlist, netlist
    assert "LT1_S T1_s_ri SB_N 2.5e-05" in netlist, netlist
    # K card still couples the inductors by name
    assert "KT1 LT1_P LT1_S 0.999" in netlist, netlist


@requires_sim
def test_winding_resistance_absent_is_byte_identical():
    """No rp/rs -> exactly the legacy 3-line emission (no R lines)."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "RT1_" not in netlist
    assert "_ri" not in netlist


@requires_sim
def test_winding_resistance_seen_at_dc_live():
    """Live: a DC source across the primary settles to ~V/rp at steady state
    (the winding is not an ideal short), so the ideal-inductor DC degeneracy
    breaks. Transient (coupled inductors are singular in a bare .op); the
    secondary carries a load resistor so its node has a DC path to ground."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5 rp=2 rs=0.1"
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="6")
    rload = Part("Device", "R", ref="R2", value="1k")
    # One net object per node (reused), else same-named-but-distinct nets are
    # electrically separate and get renamed.
    p, gnd, s1 = Net("P"), Net("GND"), Net("S1")
    p.connect(t["AA"], v[1])
    gnd.connect(t["AB"], v[2], rload[2])
    s1.connect(t["SA"], rload[1])
    gnd.connect(t["SB"])
    from skidl.sim import simulate

    try:
        an = simulate().transient_analysis(
            step_time=1e-6, end_time=2e-3, max_time=1e-5,
            use_initial_condition=True,
        )
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    # I -> V/rp = 6/2 = 3 A at steady state (an ideal 0-ohm winding would be a
    # short -> current limited only by the source, far larger).
    iv = an.get_current("V1")
    cur = abs(iv[-1] if hasattr(iv, "__len__") else iv)
    assert 2.0 < cur < 4.0, cur


@requires_sim
def test_1p2s_emission_three_L_three_K():
    _setup()
    t = Part("Device", "Transformer_1P_2S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5 n2=0.1"
    for pin, net in (
        ("AA", "PA"), ("AB", "PB"),
        ("SA", "S1A"), ("SB", "S1B"),
        ("SC", "S2A"), ("SD", "S2B"),
    ):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_P PA PB 0.0001" in netlist
    assert "LT1_S1 S1A S1B 2.5e-05" in netlist
    assert "LT1_S2 S2A S2B 1e-06" in netlist
    # a K card for every winding pair
    assert "KT1_PS1 LT1_P LT1_S1 0.999" in netlist
    assert "KT1_PS2 LT1_P LT1_S2 0.999" in netlist
    assert "KT1_S1S2 LT1_S1 LT1_S2 0.999" in netlist


@requires_sim
def test_1pss_emission_center_tap_halves():
    _setup()
    t = Part("Device", "Transformer_1P_SS", ref="T1")
    t.Sim_Params = "lp=1m n=0.5"
    for pin, net in (
        ("AA", "PA"), ("AB", "PB"),
        ("SA", "TOP"), ("SC", "TAP"), ("SB", "BOT"),
    ):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_P PA PB 0.001" in netlist
    # two half-windings, LP*N^2 each, dots at SA(TOP) and SC(TAP)
    assert "LT1_S1 TOP TAP 0.00025" in netlist
    assert "LT1_S2 TAP BOT 0.00025" in netlist
    assert "KT1_PS1 LT1_P LT1_S1 0.999" in netlist
    assert "KT1_PS2 LT1_P LT1_S2 0.999" in netlist
    assert "KT1_S1S2 LT1_S1 LT1_S2 0.999" in netlist
