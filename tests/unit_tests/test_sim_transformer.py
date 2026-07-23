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


# --- Stage 30.1: Lm / n / Llk flyback-friendly parameterization -----------
#
# New opt-in Sim.Params keys LM (magnetizing inductance) and LLK (primary
# leakage, henries) map internally to the existing LP/K coupled-inductor form
# via the T-model (LP=Lm+Llk, K=sqrt(Lm/LP), LS=LP*n^2), so leakage is a real
# tunable value. Absent LM, the LP/K path stays byte-identical.


def _emit_conv(view):
    """Emit and return (netlist_text, converter) so provenance is inspectable."""
    conv = SpiceConverter(view)
    return str(conv.convert(strict=False)), conv


@requires_sim
def test_transformer_lm_llk_maps_to_lp_k():
    """lm=100u llk=2u n=0.2 -> LP=102u, K=sqrt(100/102), LS=LP*0.04."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u llk=2u n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_P PA PB 0.000102" in netlist, netlist
    assert "KT1 LT1_P LT1_S 0.990148" in netlist, netlist
    assert "LT1_S SA_N SB_N 4.08e-06" in netlist, netlist


@requires_sim
def test_transformer_lm_no_llk_is_ideal_k1():
    """lm=100u n=0.2 (no llk) -> ideal coupling K=1, LP=Lm, LS=LP*0.04."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_P PA PB 0.0001" in netlist, netlist
    assert "KT1 LT1_P LT1_S 1" in netlist, netlist
    assert "LT1_S SA_N SB_N 4e-06" in netlist, netlist


@requires_sim
def test_transformer_lp_path_byte_identical():
    """The existing lp/n/k spelling emits byte-identically (hard gate)."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_P PA PB 0.0001" in netlist
    assert "LT1_S SA_N SB_N 2.5e-05" in netlist
    assert "KT1 LT1_P LT1_S 0.999" in netlist


@requires_sim
def test_transformer_over_constrained_rejected(caplog):
    """Both lp and lm is over-constrained -> params None + a logged warning."""
    import logging

    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u lm=98u n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    conv = SpiceConverter(_view())
    with caplog.at_level(logging.WARNING):
        params = conv._transformer_params(_view().components["T1"])
    assert params is None
    assert any("over-constrained" in r.message for r in caplog.records), caplog.text


@requires_sim
def test_transformer_provenance_records_llk():
    """Provenance records lm/llk iff LM drove the mapping."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u llk=2u n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    _netlist, conv = _emit_conv(_view())
    prov = conv.model_provenance["T1"].name
    assert "lm=" in prov and "llk=" in prov, prov


@requires_sim
def test_transformer_provenance_unchanged_without_lm():
    """The lp/n/k spelling records the legacy xfmr(lp=.., ls=.., k=..) string."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    _netlist, conv = _emit_conv(_view())
    prov = conv.model_provenance["T1"].name
    assert prov == "xfmr(lp=0.0001, ls=2.5e-05, k=0.999)", prov
    assert "lm=" not in prov and "llk=" not in prov


@requires_sim
def test_transformer_llk_is_short_circuit_inductance_live():
    """Live proof that LLK is a real element: with the secondary shorted, the
    primary short-circuit inductance is the leakage (magnetizing is shorted out
    by the reflected short), so a DC step ramps primary current at ~V/Llk. Recover
    Lsc = V*t/I from the measured slope and check it matches the set LLK (10u), not
    the magnetizing-limited LP (110u, ~11x slower)."""
    _setup()
    LLK = 10e-6
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u llk=10u n=1"
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
    rsc = Part("Device", "R", ref="R1", value="0.001")  # secondary short
    p, gnd, s1 = Net("P"), Net("GND"), Net("S1")
    p.connect(t["AA"], v[1])
    gnd.connect(t["AB"], v[2])
    s1.connect(t["SA"], rsc[1])
    gnd.connect(t["SB"], rsc[2])
    from skidl.sim import simulate

    try:
        an = simulate().transient_analysis(
            step_time=2e-7, end_time=2e-5, max_time=2e-7,
            use_initial_condition=True,
        )
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    iv = an.get_current("V1")
    tv = an.time_array()
    i_end = abs(iv[-1] if hasattr(iv, "__len__") else iv)
    t_end = float(tv[-1])
    lsc = 5.0 * t_end / i_end  # V*t/I from the leakage-limited ramp
    # leakage-limited, not magnetizing-limited: Lsc ~ LLK (10u), far below LP=110u
    assert 0.5 * LLK < lsc < 2.0 * LLK, f"Lsc={lsc:g} (I={i_end:g} at t={t_end:g})"


# --- Stage 30.2: nonlinear (saturable) magnetizing core -------------------
#
# An opt-in ISAT/BSAT selects a nonlinear magnetizing inductance: below the
# knee (|phi| < Lm*Isat) the incremental inductance is Lm; past it the core
# saturates to a small residual Lsat, so the magnetizing current RUNS AWAY
# under volt-second overload. The linear (no-sat) emission stays byte-identical.


@requires_sim
def test_transformer_saturation_off_is_byte_identical():
    """No isat/bsat -> the exact linear coupled-inductor emission (no flux node,
    no magnetizing B-source). Hard gate."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u n=0.5"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_P PA PB 0.0001" in netlist
    assert "LT1_S SA_N SB_N 2.5e-05" in netlist
    assert "KT1 LT1_P LT1_S 0.999" in netlist
    # nonlinear-mode artefacts must be absent
    assert "flux" not in netlist
    assert "T1_mag" not in netlist
    assert "ET1_" not in netlist


@requires_sim
def test_transformer_saturation_emits_flux_node():
    """lm=100u isat=2 n=0.2 -> the behavioral T-model: a flux-integrator cap, the
    saturating magnetizing B-source, and an ideal n:1 E/F secondary coupling.
    Knee flux phis=Lm*Isat=0.0002; residual Lsat=0.05*Lm -> coefw=1/Lsat-1/Lm."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u isat=2 n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    # flux integrator: a 1 F cap whose voltage is the flux, IC=0, with a DC path
    assert "BT1_fluxdrv 0 T1_flux I = V(PA)-V(PB)" in netlist, netlist
    assert "CT1_flux T1_flux 0 1 IC=0" in netlist, netlist
    assert "RT1_fluxlk T1_flux 0 1e12" in netlist, netlist
    # saturating magnetizing branch (magp=PA since llk=0), with the linear slope
    # 1/Lm=10000, the extra saturated-slope coefficient 190000 and the knee flux
    assert "BT1_mag PA PB I = V(T1_flux)*10000+190000*(" in netlist, netlist
    assert "0.0002" in netlist  # phis = Lm*Isat (knee flux)
    # ideal n:1 secondary: open-circuit voltage n*Vm + reflected current
    assert "ET1_S1 T1_s1v SB_N PA PB 0.2" in netlist, netlist
    assert "VT1_S1 T1_s1v SA_N 0" in netlist, netlist
    assert "FT1_S1 PA PB VT1_S1 0.2" in netlist, netlist
    # never the linear coupled form in nonlinear mode
    assert "KT1 " not in netlist


@requires_sim
def test_transformer_saturation_llk_emits_leakage_inductor():
    """With an explicit llk the magnetizing node is separated by a real series
    leakage inductor (magp = T1_magp), not tied to the primary terminal."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u llk=10u isat=2 n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    netlist = _emit(_view())
    assert "LT1_LKP PA T1_magp 1e-05" in netlist, netlist
    assert "BT1_mag T1_magp PB I = " in netlist, netlist
    assert "ET1_S1 T1_s1v SB_N T1_magp PB 0.2" in netlist, netlist


@requires_sim
def test_transformer_saturation_provenance():
    """The sat(isat=.., lsat=..) suffix is present iff nonlinear mode is active."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u isat=2 n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    _netlist, conv = _emit_conv(_view())
    prov = conv.model_provenance["T1"].name
    assert "sat(isat=" in prov and "lsat=" in prov, prov


@requires_sim
def test_transformer_isat_nonpositive_rejected(caplog):
    """isat<=0 -> params None + a logged warning (validate() names it)."""
    import logging

    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u isat=0 n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    conv = SpiceConverter(_view())
    with caplog.at_level(logging.WARNING):
        params = conv._transformer_params(_view().components["T1"])
    assert params is None
    assert any("isat" in r.message for r in caplog.records), caplog.text


@requires_sim
def test_transformer_saturation_needs_lm(caplog):
    """A saturation key with the lp/k spelling (no lm) is rejected -- saturation
    acts on the magnetizing inductance, which must be separated out via lm=."""
    import logging

    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lp=100u isat=2 n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    conv = SpiceConverter(_view())
    with caplog.at_level(logging.WARNING):
        params = conv._transformer_params(_view().components["T1"])
    assert params is None
    assert any("magnetizing spelling" in r.message for r in caplog.records), caplog.text


@requires_sim
def test_transformer_lsat_out_of_range_rejected(caplog):
    """lsat must be a residual inductance in (0, lm); lsat >= lm is rejected."""
    import logging

    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = "lm=100u isat=2 lsat=200u n=0.2"
    for pin, net in (("AA", "PA"), ("AB", "PB"), ("SA", "SA_N"), ("SB", "SB_N")):
        Net(net).connect(t[pin])
    conv = SpiceConverter(_view())
    with caplog.at_level(logging.WARNING):
        params = conv._transformer_params(_view().components["T1"])
    assert params is None
    assert any("lsat" in r.message for r in caplog.records), caplog.text


def _sat_ramp_current(isat_key, end_time, vdrive=1.0):
    """Build a saturable transformer, drive the primary with a DC source (secondary
    lightly loaded to the shared sim ground), run a transient and return
    (i_end, t_end) of the primary current -- i.e. the magnetizing ramp. Skips if
    ngspice is unavailable."""
    _setup()
    t = Part("Device", "Transformer_1P_1S", ref="T1")
    t.Sim_Params = f"lm=100u {isat_key} n=1"
    v = Part("Simulation_SPICE", "VDC", ref="V1", value=str(vdrive))
    rload = Part("Device", "R", ref="R1", value="1Meg")  # DC path, negligible load
    p, gnd, s1 = Net("P"), Net("GND"), Net("S1")
    p.connect(t["AA"], v[1])
    gnd.connect(t["AB"], v[2])
    s1.connect(t["SA"], rload[1])
    gnd.connect(t["SB"], rload[2])
    from skidl.sim import simulate

    an = simulate().transient_analysis(
        step_time=end_time / 300, end_time=end_time, max_time=end_time / 300,
        use_initial_condition=True,
    )
    iv = an.get_current("V1")
    tv = an.time_array()
    return abs(iv[-1] if hasattr(iv, "__len__") else iv), float(tv[-1])


@requires_sim
def test_transformer_saturates_above_knee():
    """Live: drive the primary so the magnetizing flux passes the knee. Past the
    knee the magnetizing current RUNS AWAY -- far above the linear V*t/Lm
    extrapolation. lm=100u isat=0.5 -> phis=Lm*Isat=5e-5 V-s; at V=1 the knee is
    at t=5e-5 s, so run to t=1e-4 s (phi ~ 2*phis)."""
    LM = 100e-6
    try:
        i_end, t_end = _sat_ramp_current("isat=0.5", end_time=1e-4, vdrive=1.0)
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    i_lin = 1.0 * t_end / LM  # linear-inductor extrapolation V*t/Lm
    ratio = i_end / i_lin if i_lin else float("inf")
    print(f"\n[sat] i_end={i_end:g} A, linear-extrap={i_lin:g} A, "
          f"runaway ratio={ratio:.2f} (phi~2*knee)")
    assert ratio > 3.0, f"core did not run away past the knee (ratio={ratio:.2f})"


@requires_sim
def test_transformer_linear_region_matches_linear_model():
    """Live: well below the knee the nonlinear magnetizing current tracks the
    linear V*t/Lm ramp within a few %. lm=100u isat=2 -> phis=2e-4; drive to
    phi ~ 0.15*phis so the saturating term is negligible."""
    LM = 100e-6
    try:
        # phis = Lm*Isat = 2e-4; run to phi = 0.15*phis -> t = 3e-5 s at V=1.
        i_end, t_end = _sat_ramp_current("isat=2", end_time=3e-5, vdrive=1.0)
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    i_lin = 1.0 * t_end / LM
    err = abs(i_end - i_lin) / i_lin if i_lin else float("inf")
    print(f"\n[lin] i_end={i_end:g} A, linear={i_lin:g} A, error={err*100:.2f}%")
    assert err < 0.08, f"below-knee current deviates from linear by {err*100:.1f}%"
