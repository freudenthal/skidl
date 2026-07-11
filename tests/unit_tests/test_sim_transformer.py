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
