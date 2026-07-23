# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: behavioral CLOSED-LOOP peak-current-mode controller (Stage 29.1).

``Sim.Device="CMCONTROLLER"`` + ``Sim.Params="topology=buck fsw=... vref=..."``
selects a **cycle-accurate, large-signal, closed-loop** controller that generates
the real gate from feedback and regulates a live rail in ``.tran`` -- a NEW regime,
distinct from both the open-loop switch macromodels (duty is swept) and the averaged
28.D loop model (small-signal ``.ac``). It is built entirely from ngspice
``B``-sources + the switch-held-memory latch pattern of ``_emit_ms_dff`` (no XSPICE,
no corpus model):

* an oscillator (SET clock + slope-comp sawtooth + max-duty force-off pulse);
* a gm error amp into the user's real external VC net (reuses 28.D's cell);
* a current sense (0 V source in the switch branch) + slope-compensated signal;
* a reset = (current comparator OR max-duty), RC-slowed for convergence;
* an SR latch (set=clk, RESET-DOMINANT) as a switch-held memory cap;
* a latch-gated buck switch stage + freewheel diode.

Emission tests inspect the SPICE text (the whole PWM engine + the buck stage, the
reset-dominant set gating, the optional soft reference); validate tests check the
terminal / VREF / FSW / buck-only guards; a gated live test confirms the closed loop
starts from 0 V and regulates. Full acceptance (regulation, settled switching, the
28.D regime cross-check, D>0.5 slope-comp stability) lives in
``skidl-eda/canaries/cmcontroller/drive_buck_cmcontroller.py``.
"""

import re

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


# A CMCONTROLLER stand-in whose pins carry exactly the names the controller
# terminal resolver keys on: VIN / SW / VOUT / FB / VC / GND.
def _cmc_part(ref="U1", *, drop=(), **fields):
    pins = [
        Pin(num=1, name="VIN", func=pin_types.PWRIN),
        Pin(num=2, name="SW", func=pin_types.PASSIVE),
        Pin(num=3, name="VOUT", func=pin_types.PWROUT),
        Pin(num=4, name="FB", func=pin_types.PASSIVE),
        Pin(num=5, name="VC", func=pin_types.PASSIVE),
        Pin(num=6, name="GND", func=pin_types.PWRIN),
    ]
    pins = [p for p in pins if p.name not in drop]
    u = Part(tool=SKIDL, name="CMCONTROLLER", ref_prefix="U", ref=ref, pins=pins)
    for k, v in fields.items():
        setattr(u, k, v)
    return u


def _wire(u):
    for name in ("VIN", "SW", "VOUT", "FB", "VC", "GND"):
        if name in [p.name for p in u.pins]:
            Net(name).connect(u[name])


def _emit(u):
    _wire(u)
    return str(SpiceConverter(_view()).convert(strict=False))


_BUCK_PARAMS = "topology=buck fsw=500k vout=3.3 vin=12 vref=0.8 ri=0.1 mcslope=0.1"


# --- emission ------------------------------------------------------------- #


@requires_sim
def test_cmcontroller_emits_pwm_engine_and_buck_stage():
    """The buck CMCONTROLLER emits the whole PWM engine (oscillator, gm error amp
    into the real VC net, current sense, slope-comp signal, reset OR, SR latch,
    gate) plus a latch-gated buck switch stage and a freewheel diode."""
    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS)
    net = _emit(u)
    # oscillator: SET clock + slope sawtooth + max-duty force-off pulse
    assert "VU1_clk U1_clk 0 PULSE(0 5 0 " in net, net
    assert "VU1_ramp U1_ramp 0 PULSE(0 1 0 " in net, net
    assert "VU1_duty U1_rstd 0 PULSE(0 5 " in net, net
    # gm error amp into the real external VC net (28.D cell), now source/sink-limited
    # (min/max on the current) with soft VHIGH/VLOW output-swing clamps (Stage 29.2)
    assert ("BU1_ea 0 VC I = min(max(0.00025*(0.8 - V(FB)), -0.001), 0.001)"
            in net), net
    assert "RU1_ea VC 0 1e+06" in net, net
    assert "VU1_vh U1_vh 0 2.4" in net and "DU1_hi VC U1_vh DCLU1" in net, net
    assert "VU1_vl U1_vl 0 0" in net and "DU1_lo U1_vl VC DCLU1" in net, net
    assert ".model DCLU1 D(RS=10 N=0.01)" in net, net
    # current sense (0 V in the switch branch) + slope-compensated signal
    assert "VU1_isns U1_swhi SW 0" in net, net
    assert "BU1_isig U1_isig 0 V = 0.1*I(VU1_isns) + 0.1*V(U1_ramp)" in net, net
    # reset = current comparator OR max-duty, then RC-slowed
    assert ("BU1_rst U1_rstr 0 V = V(U1_isig) > V(VC) ? 5 : "
            "(V(U1_rstd) > 2.5 ? 5 : 0)") in net, net
    assert "RU1_rd U1_rstr U1_rstf 1000" in net and "CU1_rd U1_rstf 0" in net, net
    # SR latch: switch-held memory cap with IC=0, latched gate
    assert "SU1_set U1_hi U1_qm U1_sg 0 SWLU1" in net, net
    assert "SU1_rst U1_qm 0 U1_rstf 0 SWLU1" in net, net
    assert re.search(r"CU1_qm U1_qm 0 [\d.eE+-]+ IC=0", net), net
    assert ".model SWLU1 SW(Ron=50 Roff=1e9 Vt=2.5 Vh=0.2)" in net, net
    assert "BU1_gate U1_gate 0 V = V(U1_qm) > 2.5 ? 5 : 0" in net, net
    # latch-gated buck switch stage + freewheel diode
    assert "SU1_hs VIN U1_swhi U1_gate 0 SWMU1" in net, net
    assert ".model SWMU1 SW(Ron=0.1 Roff=1e6 Vt=2.5 Vh=0.2)" in net, net
    assert "DU1_fw 0 SW DFWU1" in net, net
    assert ".model DFWU1 D(IS=1e-9 N=1.05 CJO=100p)" in net, net


@requires_sim
def test_reset_dominant_set_gating():
    """The SET is gated OFF while the latch is resetting (nested ternary) so the
    comparator / current-limit path always wins -- the safety-correct reset-dominant
    latch."""
    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS)
    net = _emit(u)
    assert ("BU1_sg U1_sg 0 V = V(U1_clk) > 2.5 ? "
            "(V(U1_rstf) > 2.5 ? 0 : 5) : 0") in net, net


@requires_sim
def test_soft_reference_ramp_emitted_only_when_tss_positive():
    """A nonzero TSS ramps the internal reference from 0 (a B-source using ``time``)
    and the error amp references that node; TSS=0 (default) inlines the literal."""
    _setup()
    u_hard = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS)
    net_hard = _emit(u_hard)
    assert "U1_vref" not in net_hard, net_hard
    assert ("BU1_ea 0 VC I = min(max(0.00025*(0.8 - V(FB)), -0.001), 0.001)"
            in net_hard)

    _setup()
    u_soft = _cmc_part(
        Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " tss=60u"
    )
    net_soft = _emit(u_soft)
    assert re.search(
        r"BU1_vref U1_vref 0 V = 0\.8\*\(time > [\d.eE-]+ \? 1 : time/[\d.eE-]+\)",
        net_soft,
    ), net_soft
    # the error amp now regulates FB to the RAMPED reference node
    assert ("BU1_ea 0 VC I = min(max(0.00025*(V(U1_vref) - V(FB)), -0.001), 0.001)"
            in net_soft), net_soft


@requires_sim
def test_negative_vref_accepted():
    """VREF may be negative (the LT3757 dual-reference / inverting-FBX
    configuration), unlike the voltage-mode buck which requires a positive VREF.
    The sign flows straight through the gm subtraction: FB is regulated to the
    negative reference. Same source/sink limit wraps it."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=buck fsw=500k vref=-0.8 ri=0.1",
    )
    net = _emit(u)
    assert ("BU1_ea 0 VC I = min(max(0.00025*(-0.8 - V(FB)), -0.001), 0.001)"
            in net), net


@requires_sim
def test_slope_factor_scales_ramp_into_current_signal():
    """MCSLOPE (aliased MC_SLOPE) sets the volts of slope-comp ramp added into the
    current signal -- the knob that damps D>0.5 subharmonic oscillation."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=buck fsw=500k vref=0.8 ri=0.1 mc_slope=0.3",
    )
    net = _emit(u)
    assert "BU1_isig U1_isig 0 V = 0.1*I(VU1_isns) + 0.3*V(U1_ramp)" in net, net


@requires_sim
def test_error_amp_clamps_and_limits_configurable():
    """VHIGH/VLOW (output-swing rails) and ISOURCE/ISINK (source/sink current
    limits) are datasheet knobs (Stage 29.2). Given values flow into the emitted
    diode-clamp rails and the min/max on the gm current."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params=_BUCK_PARAMS + " vhigh=2 vlow=0.2 isource=50u isink=30u",
    )
    net = _emit(u)
    # current source/sink limit on the gm cell (min +ISOURCE, max -ISINK)
    assert ("BU1_ea 0 VC I = min(max(0.00025*(0.8 - V(FB)), -3e-05), 5e-05)"
            in net), net
    # soft output-swing rails at the given VHIGH / VLOW
    assert "VU1_vh U1_vh 0 2" in net and "VU1_vl U1_vl 0 0.2" in net, net


@requires_sim
def test_gain_db_sets_error_amp_dc_gain_resistor():
    """GAIN (dB) sets the error-amp DC open-loop gain by deriving REA = 10^(GAIN/20)/GM
    (Stage 29.2); an explicit REA still wins over GAIN."""
    _setup()
    # GAIN=80 dB, GM=250u -> REA = 1e4 / 250e-6 = 4e7.
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=buck fsw=500k vref=0.8 gm=250u gain=80",
    )
    net = _emit(u)
    assert "RU1_ea VC 0 4e+07" in net, net

    _setup()
    # explicit REA overrides GAIN.
    u2 = _cmc_part(
        ref="U2",
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=buck fsw=500k vref=0.8 gm=250u gain=80 rea=1e6",
    )
    net2 = _emit(u2)
    assert "RU2_ea VC 0 1e+06" in net2, net2


@requires_sim
def test_provenance_recorded():
    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS)
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    prov = conv.model_provenance["U1"]
    assert prov.kind == "cmcontroller"
    assert prov.tier == "sim_params"
    assert prov.name == "buck_cmcontroller(vref=0.8, fsw=500k)", prov.name


# --- supervisory features (Stage 29.3) ------------------------------------ #


@requires_sim
def test_supervisory_off_by_default_is_byte_identical():
    """With no supervisory param set, the emission is byte-identical to 29.1/29.2:
    the reset is the bare current-comparator-OR-max-duty line, the set gate samples
    the plain FSW clock, the gate is the bare latched Q, and none of the
    current-limit / foldback / UVLO nodes appear. This is the additive-only gate."""
    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS)
    net = _emit(u)
    # the exact 29.1 reset / set-gate / gate lines (no leading blank, no extra OR)
    assert ("BU1_rst U1_rstr 0 V = V(U1_isig) > V(VC) ? 5 : "
            "(V(U1_rstd) > 2.5 ? 5 : 0)") in net, net
    assert ("BU1_sg U1_sg 0 V = V(U1_clk) > 2.5 ? "
            "(V(U1_rstf) > 2.5 ? 0 : 5) : 0") in net, net
    assert "BU1_gate U1_gate 0 V = V(U1_qm) > 2.5 ? 5 : 0" in net, net
    # no supervisory nodes emitted
    for tok in ("U1_clkf", "U1_clksel", "U1_uvset", "U1_uvrst", "U1_run", "I(VU1_isns) >"):
        assert tok not in net, (tok, net)


@requires_sim
def test_current_limit_ors_into_reset():
    """VSENSE_MAX adds a cycle-by-cycle current-limit term to the reset: the latch
    also resets when the RAW sensed current RI*I(isns) (slope comp excluded) exceeds
    VSENSE_MAX, OR'd in after the PWM comparator and max-duty."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " vsense_max=0.11"
    )
    net = _emit(u)
    assert ("BU1_rst U1_rstr 0 V = V(U1_isig) > V(VC) ? 5 : "
            "(V(U1_rstd) > 2.5 ? 5 : (0.1*I(VU1_isns) > 0.11 ? 5 : 0))") in net, net


@requires_sim
def test_min_on_time_blanks_the_reset():
    """TON_MIN blanks the reset for the leading TON_MIN of each cycle by gating the
    whole reset off while V(ramp) < TON_MIN*FSW (here 100 ns / 2 us = 0.05)."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " ton_min=100n"
    )
    net = _emit(u)
    assert ("BU1_rst U1_rstr 0 V = V(U1_ramp) < 0.05 ? 0 : "
            "(V(U1_isig) > V(VC) ? 5 : (V(U1_rstd) > 2.5 ? 5 : 0))") in net, net


@requires_sim
def test_frequency_foldback_selects_slower_clock():
    """FB_FOLD emits a second (folded) clock at FSW*FOLD_RATIO and a selector that
    clocks the latch with it while V(FB) < FB_FOLD -- the two-state foldback. The set
    gate then samples the SELECTED clock node, not the plain FSW clock."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params=_BUCK_PARAMS + " fb_fold=1.2 fold_ratio=0.25",
    )
    net = _emit(u)
    # folded clock at 500k*0.25 = 125 kHz -> period 8 us
    assert "VU1_clkf U1_clkf 0 PULSE(0 5 0 " in net and " 8e-06)" in net, net
    assert ("BU1_clksel U1_clksel 0 V = V(FB) < 1.2 ? V(U1_clkf) : V(U1_clk)"
            in net), net
    assert ("BU1_sg U1_sg 0 V = V(U1_clksel) > 2.5 ? "
            "(V(U1_rstf) > 2.5 ? 0 : 5) : 0") in net, net


@requires_sim
def test_uvlo_gates_gate_with_hysteretic_run_latch():
    """UVLO_RISE/UVLO_FALL emit a hysteretic run latch (the switch-held-memory SR
    pattern) that ANDs into the gate: the gate is Q only while the run latch is set,
    and the run latch sets above UVLO_RISE / resets below UVLO_FALL."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params=_BUCK_PARAMS + " uvlo_rise=8 uvlo_fall=7",
    )
    net = _emit(u)
    assert "BU1_uvset U1_uvset 0 V = V(VIN) > 8 ? 5 : 0" in net, net
    assert "BU1_uvrst U1_uvrst 0 V = V(VIN) < 7 ? 5 : 0" in net, net
    assert "SU1_uvs U1_hi U1_runm U1_uvset 0 SWLU1" in net, net
    assert "SU1_uvr U1_runm 0 U1_uvrst 0 SWLU1" in net, net
    assert re.search(r"CU1_uvr U1_runm 0 [\d.eE+-]+ IC=0", net), net
    assert "BU1_run U1_run 0 V = V(U1_runm) > 2.5 ? 5 : 0" in net, net
    assert ("BU1_gate U1_gate 0 V = V(U1_qm) > 2.5 ? "
            "(V(U1_run) > 2.5 ? 5 : 0) : 0") in net, net


@requires_sim
def test_uvlo_fall_defaults_below_rise():
    """UVLO_FALL defaults to 0.9*UVLO_RISE (hysteresis) when only UVLO_RISE is set."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " uvlo_rise=10"
    )
    net = _emit(u)
    assert "BU1_uvset U1_uvset 0 V = V(VIN) > 10 ? 5 : 0" in net, net
    assert "BU1_uvrst U1_uvrst 0 V = V(VIN) < 9 ? 5 : 0" in net, net


@requires_sim
def test_supervisory_features_recorded_in_provenance():
    """The active NEW supervisory features are recorded in the provenance name (so the
    netlist documents which protections it emulates); TSS (pre-existing) is not, so a
    soft-start-only controller keeps the 29.1 provenance string byte-identical."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params=_BUCK_PARAMS + " vsense_max=0.11 uvlo_rise=8",
    )
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    name = conv.model_provenance["U1"].name
    assert "ilim=0.11" in name and "uvlo=8/7.2" in name, name

    # TSS-only keeps the bare 29.1 provenance string
    _setup()
    u2 = _cmc_part(
        ref="U2", Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " tss=40u"
    )
    _wire(u2)
    conv2 = SpiceConverter(_view())
    conv2.convert(strict=False)
    assert conv2.model_provenance["U2"].name == "buck_cmcontroller(vref=0.8, fsw=500k)"


# --- topology generalization (Stage 29.4) --------------------------------- #


# A SEPIC/Ćuk stand-in adds the SWB (node-B, coupling-cap junction) pin the two
# coupling-cap topologies need on top of the six buck terminals.
def _cmc_part_swb(ref="U1", **fields):
    pins = [
        Pin(num=1, name="VIN", func=pin_types.PWRIN),
        Pin(num=2, name="SW", func=pin_types.PASSIVE),
        Pin(num=3, name="SWB", func=pin_types.PASSIVE),
        Pin(num=4, name="VOUT", func=pin_types.PWROUT),
        Pin(num=5, name="FB", func=pin_types.PASSIVE),
        Pin(num=6, name="VC", func=pin_types.PASSIVE),
        Pin(num=7, name="GND", func=pin_types.PWRIN),
    ]
    u = Part(tool=SKIDL, name="CMCONTROLLER", ref_prefix="U", ref=ref, pins=pins)
    for k, v in fields.items():
        setattr(u, k, v)
    return u


def _wire_swb(u):
    for name in ("VIN", "SW", "SWB", "VOUT", "FB", "VC", "GND"):
        Net(name).connect(u[name])


def _emit_swb(u):
    _wire_swb(u)
    return str(SpiceConverter(_view()).convert(strict=False))


@requires_sim
def test_boost_emits_lowside_switch_and_rectifier():
    """A boost CMCONTROLLER reuses the whole core but emits a LOW-side switch
    (SW->GND, latch-gated, sensed in the switch branch) plus a rectifier SW->VOUT --
    not the buck's high-side switch + freewheel. The error amp keeps the non-inverting
    (VREF - FB) sense, and the current sense reads the switch on-time current."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=boost fsw=500k vout=12 vin=5 vref=1.2 ri=0.1",
    )
    net = _emit(u)
    # non-inverting error amp, sense in the low-side branch
    assert "BU1_ea 0 VC I = min(max(0.00025*(1.2 - V(FB)), -0.001), 0.001)" in net, net
    assert "VU1_isns U1_swlo 0 0" in net, net
    assert "BU1_isig U1_isig 0 V = 0.1*I(VU1_isns) + 0.1*V(U1_ramp)" in net, net
    # low-side switch + rectifier to VOUT; NO buck high-side switch / freewheel
    assert "SU1_ls SW U1_swlo U1_gate 0 SWMU1" in net, net
    assert ".model SWMU1 SW(Ron=0.1 Roff=1e6 Vt=2.5 Vh=0.2)" in net, net
    assert "DU1_rect SW VOUT DFWU1" in net, net
    assert "SU1_hs" not in net and "DU1_fw" not in net, net
    assert conv_name(u) == "boost_cmcontroller(vref=1.2, fsw=500k)"


@requires_sim
def test_sepic_emits_main_switch_and_rectifier_to_vout():
    """A SEPIC CMCONTROLLER emits the main switch A(SW)->GND (sensed) with a freewheel
    GND->SW and a rectifier from node B (SWB) to VOUT -- the load-bearing anode-at-B
    orientation. Non-inverting. Needs the SWB pin."""
    _setup()
    u = _cmc_part_swb(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=sepic fsw=500k vout=12 vin=12 vref=1.2 ri=0.1",
    )
    net = _emit_swb(u)
    assert "BU1_ea 0 VC I = min(max(0.00025*(1.2 - V(FB)), -0.001), 0.001)" in net, net
    assert "VU1_isns U1_swlo 0 0" in net, net
    assert "SU1_main SW U1_swlo U1_gate 0 SWMU1" in net, net
    assert "DU1_mfw 0 SW DFWU1" in net, net
    assert "DU1_rect SWB VOUT DFWU1" in net, net


@requires_sim
def test_cuk_emits_inverted_error_amp_and_rectifier_to_gnd():
    """The inverting Ćuk flips the error-amp sense to (FB - VREF) (with a negative VREF)
    so the loop stays negative-feedback on the negative output, and its rectifier ties
    node B (SWB) to GND (anode at B) -- the output is the negative rail behind the
    user's L2 (SWB->VOUT). Needs the SWB pin."""
    _setup()
    u = _cmc_part_swb(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=cuk fsw=500k vout=-5 vin=12 vref=-0.8 ri=0.1",
    )
    net = _emit_swb(u)
    # inverted sense: (FB - VREF), VREF negative
    assert "BU1_ea 0 VC I = min(max(0.00025*(V(FB) - -0.8), -0.001), 0.001)" in net, net
    assert "SU1_main SW U1_swlo U1_gate 0 SWMU1" in net, net
    assert "DU1_mfw 0 SW DFWU1" in net, net
    assert "DU1_rect SWB 0 DFWU1" in net, net
    assert conv_name(u) == "cuk_cmcontroller(vref=-0.8, fsw=500k)"


@requires_sim
def test_flyback_emits_primary_switch_only():
    """A flyback CMCONTROLLER emits only the primary LS switch SW->GND (sensed); the
    user's transformer + secondary rectifier + output cap form the isolated output, so
    no rectifier is emitted. Non-inverting, no SWB needed."""
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=flyback fsw=250k vout=12 vin=24 vref=1.2 ri=0.2",
    )
    net = _emit(u)
    assert "SU1_ls SW U1_swlo U1_gate 0 SWMU1" in net, net
    assert "VU1_isns U1_swlo 0 0" in net, net
    assert "DU1_rect" not in net and "DU1_fw" not in net, net


@requires_sim
def test_sepic_cuk_current_limit_reads_main_switch_current():
    """VSENSE_MAX still works on the non-buck topologies: the cycle-by-cycle limit
    reads the same V{ref}_isns (the main switch on-time current), OR'd into the reset,
    for boost/sepic/cuk exactly as for buck."""
    _setup()
    u = _cmc_part_swb(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=cuk fsw=500k vout=-5 vin=12 vref=-0.8 ri=0.1 "
        "vsense_max=0.3",
    )
    net = _emit_swb(u)
    assert "0.1*I(VU1_isns) > 0.3 ? 5 : 0" in net, net


@requires_sim
def test_sepic_without_swb_skips_with_warning(caplog):
    """SEPIC/Ćuk need the node-B (SWB) coupling-cap-junction pin; without it the
    emitter warns and emits nothing (honest skip), and validate() flags it."""
    import logging

    _setup()
    # a plain 6-pin part (no SWB) asked to be a sepic
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=sepic fsw=500k vout=12 vin=12 vref=1.2 ri=0.1",
    )
    with caplog.at_level(logging.WARNING):
        net = _emit(u)
    assert "SU1_main" not in net and "BU1_gate" not in net, net
    assert any("needs a connected SWB" in r.message for r in caplog.records)

    _setup()
    u2 = _cmc_part(
        ref="U2",
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=sepic fsw=500k vout=12 vin=12 vref=1.2 ri=0.1",
    )
    _wire(u2)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("needs a connected SWB" in p for p in ei.value.problems), (
        ei.value.problems
    )


def conv_name(u):
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    return conv.model_provenance[u.ref].name


# --- validation ----------------------------------------------------------- #


@requires_sim
def test_missing_vref_is_validation_error():
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER", Sim_Params="topology=buck fsw=500k"  # no vref
    )
    _wire(u)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("CMCONTROLLER needs Sim.Params VREF" in p for p in ei.value.problems), (
        ei.value.problems
    )


@requires_sim
def test_missing_vc_pin_is_validation_error():
    """No VC (compensation) pin -> the loop cannot close through the real comp
    network; validate() flags the missing terminal."""
    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS, drop=("VC",))
    _wire(u)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any(
        "CMCONTROLLER needs connected VIN, SW, VOUT, FB, VC" in p
        for p in ei.value.problems
    ), ei.value.problems


@requires_sim
def test_missing_fb_skips_with_warning(caplog):
    """No FB pin -> the emitter cannot resolve terminals; it emits nothing and warns
    (honest skip) rather than a wrong netlist."""
    import logging

    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS, drop=("FB",))
    _wire(u)
    with caplog.at_level(logging.WARNING):
        net = str(SpiceConverter(_view()).convert(strict=False))
    assert "BU1_ea" not in net, net
    assert any(
        "needs connected VIN, SW, VOUT, FB, VC" in r.message for r in caplog.records
    )


@requires_sim
def test_unknown_topology_skips_with_warning(caplog):
    """An unrecognised topology (not buck/boost/sepic/cuk/flyback) is a config error:
    the emitter warns and emits nothing, and validate() flags it."""
    import logging

    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=zeta fsw=500k vref=1.6 ri=0.1",
    )
    with caplog.at_level(logging.WARNING):
        net = _emit(u)
    assert "SU1_hs" not in net and "BU1_gate" not in net, net
    assert any("is not a supported topology" in r.message for r in caplog.records)

    _setup()
    u2 = _cmc_part(
        ref="U2",
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=zeta fsw=500k vref=1.6 ri=0.1",
    )
    _wire(u2)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("is not a supported topology" in p for p in ei.value.problems), (
        ei.value.problems
    )


# --- live (gated) --------------------------------------------------------- #


@requires_sim
def test_cmcontroller_buck_regulates_closed_loop():
    """Live closed-loop .tran: the CMCONTROLLER buck starts from VOUT=0 and regulates
    the rail to VREF*(Rtop+Rbot)/Rbot (~3.31 V) with the stiff + UIC recipe. This is
    the large-signal, closed-loop capability the open-loop macromodels lack."""
    import numpy as np

    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " tss=40u")
    v1 = Part("Simulation_SPICE", "VDC", value="12", ref="V1")
    L1 = Part("Device", "L", value="22u", ref="L1")
    Co = Part("Device", "C", value="47u", ref="C1")
    Rl = Part("Device", "R", value="3.3", ref="RL")
    Rt = Part("Device", "R", value="43k", ref="RT")
    Rb = Part("Device", "R", value="13.7k", ref="RB")
    Rc = Part("Device", "R", value="22k", ref="RC")
    Cc = Part("Device", "C", value="2.2n", ref="CC")

    vin, sw, vout, vc, ncc = (Net(n) for n in ("VIN", "SW", "VOUT", "VC", "NCC"))
    fb, gnd = Net("FB"), Net("GND")
    vin.connect(v1[1], u["VIN"])
    gnd.connect(v1[2], u["GND"], Co[2], Rl[2], Rb[2], Cc[2])
    sw.connect(u["SW"], L1[1])
    vout.connect(L1[2], u["VOUT"], Co[1], Rl[1], Rt[1])
    fb.connect(Rt[2], Rb[1], u["FB"])
    vc.connect(u["VC"], Rc[1])
    ncc.connect(Rc[2], Cc[1])

    from skidl.sim import simulate

    per = 1.0 / 500e3
    try:
        res = simulate().transient_analysis(
            step_time=per / 100, end_time=400e-6, max_time=per / 50, stiff=True,
            use_initial_condition=True, initial_conditions={"VOUT": 0},
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")

    t = np.asarray(res.analysis.time, dtype=float)
    vo = np.asarray(res.analysis["VOUT"], dtype=float)
    tail = vo[t > (t[-1] - 40e-6)]
    vreg = float(tail.mean())
    target = 0.8 * (43.0 + 13.7) / 13.7
    assert np.isfinite(vo).all(), "non-finite VOUT (diverged)"
    assert abs(vreg - target) / target <= 0.05, (vreg, target)
    # the gate actually switches (closed-loop, not a stuck rail)
    g = np.asarray(res.analysis["U1_gate"], dtype=float)
    rises = int(np.sum((g[:-1] < 2.5) & (g[1:] >= 2.5)))
    assert rises > 50, ("gate not switching", rises)


@requires_sim
def test_error_amp_vc_clamps_at_vhigh_on_overdrive():
    """Live: force a sustained over-drive the loop cannot satisfy -- a heavy 0.5 ohm
    load whose demand at the 3.3 V target exceeds the peak current an explicit
    VHIGH=0.6 ceiling permits (peak iL ~ VHIGH/RI ~ 6 A). The output sags below
    target, so V(FB) < VREF forever and the error amp keeps sourcing; the soft VHIGH
    clamp (Stage 29.2) holds VC at the error-amp output ceiling instead of winding up
    unbounded (anti-windup -- the ceiling soft-start (29.3) ramps up to and the
    current limit clamps below). Everything stays finite and converges."""
    import numpy as np

    _setup()
    vhigh = 0.6
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=buck fsw=500k vout=3.3 vin=12 vref=0.8 ri=0.1 "
        "mcslope=0.1 vhigh=0.6 tss=40u",
    )
    v1 = Part("Simulation_SPICE", "VDC", value="12", ref="V1")
    L1 = Part("Device", "L", value="22u", ref="L1")
    Co = Part("Device", "C", value="47u", ref="C1")
    Rl = Part("Device", "R", value="0.5", ref="RL")     # heavy: >VHIGH-limited current
    Rt = Part("Device", "R", value="43k", ref="RT")
    Rb = Part("Device", "R", value="13.7k", ref="RB")
    Rc = Part("Device", "R", value="22k", ref="RC")
    Cc = Part("Device", "C", value="2.2n", ref="CC")

    vin, sw, vout, vc, ncc = (Net(nm) for nm in ("VIN", "SW", "VOUT", "VC", "NCC"))
    fb, gnd = Net("FB"), Net("GND")
    vin.connect(v1[1], u["VIN"])
    gnd.connect(v1[2], u["GND"], Co[2], Rl[2], Rb[2], Cc[2])
    sw.connect(u["SW"], L1[1])
    vout.connect(L1[2], u["VOUT"], Co[1], Rl[1], Rt[1])
    fb.connect(Rt[2], Rb[1], u["FB"])
    vc.connect(u["VC"], Rc[1])
    ncc.connect(Rc[2], Cc[1])

    from skidl.sim import simulate

    per = 1.0 / 500e3
    try:
        res = simulate().transient_analysis(
            step_time=per / 100, end_time=400e-6, max_time=per / 50, stiff=True,
            use_initial_condition=True, initial_conditions={"VOUT": 0},
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")

    t = np.asarray(res.analysis.time, dtype=float)
    vo = np.asarray(res.analysis["VOUT"], dtype=float)
    vcv = np.asarray(res.analysis["VC"], dtype=float)
    target = 0.8 * (43.0 + 13.7) / 13.7
    assert np.isfinite(vo).all() and np.isfinite(vcv).all(), "diverged"
    # genuine over-drive: the heavy load holds VOUT well below target.
    vo_tail = float(vo[t > (t[-1] - 40e-6)].mean())
    assert vo_tail < 0.9 * target, ("load not heavy enough to over-drive", vo_tail)
    # VC saturates at the VHIGH ceiling instead of winding up unbounded.
    vc_tail = float(vcv[t > (t[-1] - 40e-6)].mean())
    assert abs(vc_tail - vhigh) <= 0.1, ("VC did not clamp at VHIGH", vc_tail)
    # the clamp is a ceiling, not a wall the integrator blew through.
    assert float(vcv.max()) <= vhigh + 0.25, ("VC overshot the clamp", float(vcv.max()))


# --- live supervisory features (Stage 29.3, gated) ------------------------ #

_TARGET = 0.8 * (43.0 + 13.7) / 13.7  # ~3.31 V regulated rail (0.8 V ref divider)


def _build_buck(u, *, rload="3.3", vin_src=None):
    """Wire the standard 12 V -> 3.3 V buck power stage around a CMCONTROLLER part.

    ``vin_src`` overrides the default 12 V VDC (e.g. a ramping VPULSE for UVLO). All
    real external parts (L/Cout/divider/comp) are the same as the 29.1 canary."""
    if vin_src is None:
        vin_src = Part("Simulation_SPICE", "VDC", value="12", ref="V1")
    L1 = Part("Device", "L", value="22u", ref="L1")
    Co = Part("Device", "C", value="47u", ref="C1")
    Rl = Part("Device", "R", value=rload, ref="RL")
    Rt = Part("Device", "R", value="43k", ref="RT")
    Rb = Part("Device", "R", value="13.7k", ref="RB")
    Rc = Part("Device", "R", value="22k", ref="RC")
    Cc = Part("Device", "C", value="2.2n", ref="CC")
    vin, sw, vout, vc, ncc = (Net(n) for n in ("VIN", "SW", "VOUT", "VC", "NCC"))
    fb, gnd = Net("FB"), Net("GND")
    vin.connect(vin_src[1], u["VIN"])
    gnd.connect(vin_src[2], u["GND"], Co[2], Rl[2], Rb[2], Cc[2])
    sw.connect(u["SW"], L1[1])
    vout.connect(L1[2], u["VOUT"], Co[1], Rl[1], Rt[1])
    fb.connect(Rt[2], Rb[1], u["FB"])
    vc.connect(u["VC"], Rc[1])
    ncc.connect(Rc[2], Cc[1])


def _tran(end_time, **kw):
    from skidl.sim import simulate

    per = 1.0 / 500e3
    return simulate().transient_analysis(
        step_time=per / 100, end_time=end_time, max_time=per / 50, stiff=True,
        use_initial_condition=True, initial_conditions={"VOUT": 0}, **kw,
    )


def _iload(res):
    """Inductor branch current I(L1) as a numpy array (ngspice branch 'll1')."""
    import numpy as np

    return np.asarray(res.analysis.branches["ll1"], dtype=float)


@requires_sim
def test_soft_start_bounds_inrush_no_overshoot():
    """Live soft-start (TSS): the rail rises monotonically to target with the rise
    time set by TSS and NO overshoot, and the startup inductor current is bounded --
    a longer TSS lowers the peak inrush current vs a near-instant reference. This is
    the Stage 29.3 soft-start behavior (a monotone reference ramp)."""
    import numpy as np

    # long soft-start
    _setup()
    u = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " tss=120u")
    _build_buck(u)
    try:
        res = _tran(300e-6)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    t = np.asarray(res.analysis.time, dtype=float)
    vo = np.asarray(res.analysis["VOUT"], dtype=float)
    assert np.isfinite(vo).all(), "diverged"
    # reaches target, no overshoot beyond 5 %
    assert abs(float(vo[t > t[-1] - 30e-6].mean()) - _TARGET) / _TARGET <= 0.05
    assert float(vo.max()) <= _TARGET * 1.05, ("overshoot", float(vo.max()))
    # rise time tracks TSS: VOUT crosses 90 % target near TSS, not long before it
    cross = t[np.argmax(vo >= 0.9 * _TARGET)]
    assert 0.6 * 120e-6 <= cross <= 2.2 * 120e-6, ("rise time not ~TSS", cross)
    ipk_slow = float(np.max(np.abs(_iload(res))))

    # near-instant reference draws a larger inrush peak
    _setup()
    u2 = _cmc_part(Sim_Device="CMCONTROLLER", Sim_Params=_BUCK_PARAMS + " tss=2u")
    _build_buck(u2)
    res2 = _tran(300e-6)
    ipk_fast = float(np.max(np.abs(_iload(res2))))
    assert ipk_slow < ipk_fast, ("soft-start did not lower inrush", ipk_slow, ipk_fast)


@requires_sim
def test_peak_current_limit_clamps_into_short():
    """Live cycle-by-cycle current limit (VSENSE_MAX): into a hard short the peak
    inductor current clamps at ~VSENSE_MAX/RI and the rail collapses (constant
    current), instead of the ideal switch running the current away. With the limit
    OFF the same short draws a materially larger peak current."""
    import numpy as np

    ri, vsmax = 0.1, 0.3            # -> ~3 A peak current limit
    ilim = vsmax / ri
    # limit ON, hard short
    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params=f"topology=buck fsw=500k vout=3.3 vin=12 vref=0.8 ri={ri} "
        f"mcslope=0.1 tss=20u vsense_max={vsmax}",
    )
    _build_buck(u, rload="0.4")     # hard short: demands >> ilim at 3.3 V
    try:
        res = _tran(200e-6)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    vo = np.asarray(res.analysis["VOUT"], dtype=float)
    il = _iload(res)
    assert np.isfinite(vo).all() and np.isfinite(il).all(), "diverged"
    ipk_lim = float(np.max(np.abs(il)))
    # peak inductor current clamps near the datasheet limit (cycle-by-cycle),
    # allowing ripple/overshoot above the average trip point.
    assert ipk_lim <= ilim * 1.6, ("current not limited", ipk_lim, ilim)
    # the rail collapses under the short (constant-current, not regulating)
    assert float(vo[-1]) < 0.7 * _TARGET, ("rail did not collapse", float(vo[-1]))

    # limit OFF: the same short draws a larger peak
    _setup()
    u2 = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params=f"topology=buck fsw=500k vout=3.3 vin=12 vref=0.8 ri={ri} "
        f"mcslope=0.1 tss=20u",
    )
    _build_buck(u2, rload="0.4")
    res2 = _tran(200e-6)
    ipk_free = float(np.max(np.abs(_iload(res2))))
    assert ipk_lim < ipk_free, ("limit did not reduce peak", ipk_lim, ipk_free)


@requires_sim
def test_uvlo_holds_gate_off_below_threshold():
    """Live UVLO: with VIN ramped 0 -> 12 V, the gate stays quiet while VIN is below
    UVLO_RISE and only starts switching after VIN crosses it (hysteretic run latch)."""
    import numpy as np

    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params=_BUCK_PARAMS + " tss=20u uvlo_rise=8 uvlo_fall=7",
    )
    # VIN ramps linearly 0 -> 12 V over 200 us (crosses 8 V at ~133 us), then holds.
    vramp = Part("Simulation_SPICE", "VPULSE", value="0", ref="V1")
    vramp.Sim_Params = "v1=0 v2=12 td=0 tr=200u tf=1u pw=1 per=2"
    _build_buck(u, vin_src=vramp)
    try:
        res = _tran(360e-6)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    t = np.asarray(res.analysis.time, dtype=float)
    g = np.asarray(res.analysis["U1_gate"], dtype=float)
    t_cross = 8.0 / 12.0 * 200e-6   # VIN = UVLO_RISE
    # essentially no switching before the threshold (allow a stray edge from ramp)
    pre = (t[:-1] < t_cross - 10e-6)
    rises_pre = int(np.sum((g[:-1] < 2.5) & (g[1:] >= 2.5) & pre))
    assert rises_pre <= 1, ("gate switched below UVLO", rises_pre)
    # switching after the threshold
    post = (t[:-1] > t_cross + 15e-6)
    rises_post = int(np.sum((g[:-1] < 2.5) & (g[1:] >= 2.5) & post))
    assert rises_post > 20, ("gate did not start above UVLO", rises_post)


@requires_sim
def test_frequency_foldback_slows_switching_under_low_fb():
    """Live two-state frequency foldback: held into a sustained short (V(FB) far below
    FB_FOLD), the latch is clocked by the folded clock, so the measured switching rate
    is ~FSW*FOLD_RATIO instead of FSW. A current limit keeps the short current bounded."""
    import numpy as np

    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=buck fsw=500k vout=3.3 vin=12 vref=0.8 ri=0.1 "
        "mcslope=0.1 tss=10u vsense_max=0.3 fb_fold=0.4 fold_ratio=0.25",
    )
    _build_buck(u, rload="0.4")     # hard short holds FB well below 0.4
    try:
        res = _tran(240e-6)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    t = np.asarray(res.analysis.time, dtype=float)
    g = np.asarray(res.analysis["U1_gate"], dtype=float)
    fb = np.asarray(res.analysis["FB"], dtype=float)
    # confirm we are genuinely in foldback (FB below the threshold in the tail)
    m = t > (t[-1] - 120e-6)
    assert float(fb[m].mean()) < 0.4, ("not in foldback", float(fb[m].mean()))
    tt, gt = t[m], g[m]
    rises = int(np.sum((gt[:-1] < 2.5) & (gt[1:] >= 2.5)))
    fsw_meas = rises / (tt[-1] - tt[0])
    folded = 500e3 * 0.25           # 125 kHz
    # the switching rate is the folded rate, clearly below the nominal FSW
    assert abs(fsw_meas - folded) / folded <= 0.35, ("not folded", fsw_meas)
    assert fsw_meas < 0.5 * 500e3, ("rate not reduced", fsw_meas)


# --- live topology generalization (Stage 29.4, gated) --------------------- #


@requires_sim
def test_cmcontroller_boost_regulates_closed_loop():
    """Live closed-loop .tran: the CMCONTROLLER BOOST steps 5 V -> 12 V and regulates
    to VREF*(Rtop+Rbot)/Rbot with the same core as the buck (only the switch stage is
    a low-side switch + rectifier). The rectifier pre-charges VOUT to ~VIN and the loop
    boosts it to target; the compensation crosses over below the RHP zero."""
    import numpy as np

    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=boost fsw=500k vout=12 vin=5 vref=1.2 ri=0.1 "
        "mcslope=0.1 tss=100u",
    )
    v1 = Part("Simulation_SPICE", "VDC", value="5", ref="V1")
    L1 = Part("Device", "L", value="10u", ref="L1")
    Co = Part("Device", "C", value="100u", ref="C1")
    Rl = Part("Device", "R", value="24", ref="RL")     # ~0.5 A at 12 V
    Rt = Part("Device", "R", value="90k", ref="RT")
    Rb = Part("Device", "R", value="10k", ref="RB")
    Rc = Part("Device", "R", value="10k", ref="RC")
    Cc = Part("Device", "C", value="22n", ref="CC")

    vin, sw, vout, vc, ncc = (Net(nm) for nm in ("VIN", "SW", "VOUT", "VC", "NCC"))
    fb, gnd = Net("FB"), Net("GND")
    vin.connect(v1[1], u["VIN"], L1[1])        # boost: user's inductor VIN->SW
    gnd.connect(v1[2], u["GND"], Co[2], Rl[2], Rb[2], Cc[2])
    sw.connect(u["SW"], L1[2])
    vout.connect(u["VOUT"], Co[1], Rl[1], Rt[1])
    fb.connect(Rt[2], Rb[1], u["FB"])
    vc.connect(u["VC"], Rc[1])
    ncc.connect(Rc[2], Cc[1])

    from skidl.sim import simulate

    per = 1.0 / 500e3
    try:
        res = simulate().transient_analysis(
            step_time=per / 100, end_time=700e-6, max_time=per / 50, stiff=True,
            use_initial_condition=True, initial_conditions={"VOUT": 0},
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")

    t = np.asarray(res.analysis.time, dtype=float)
    vo = np.asarray(res.analysis["VOUT"], dtype=float)
    target = 1.2 * (90.0 + 10.0) / 10.0        # 12 V
    assert np.isfinite(vo).all(), "non-finite VOUT (diverged)"
    vreg = float(vo[t > (t[-1] - 60e-6)].mean())
    assert abs(vreg - target) / target <= 0.08, (vreg, target)
    g = np.asarray(res.analysis["U1_gate"], dtype=float)
    rises = int(np.sum((g[:-1] < 2.5) & (g[1:] >= 2.5)))
    assert rises > 50, ("gate not switching", rises)


@requires_sim
def test_cmcontroller_cuk_regulates_negative_rail():
    """Live closed-loop .tran: the inverting Ćuk CMCONTROLLER regulates a NEGATIVE rail
    (12 V -> -5 V). This is the true negative-OUTPUT inverting-FBX converter deferred
    from Stage 29.2: VOUT itself is negative, VREF is negative (-0.8 V), the error amp
    senses (FB - VREF), and the simple VOUT->FB->GND divider makes FB negative. The
    rectifier ties node B to GND; the output is behind the user's L2."""
    import numpy as np

    _setup()
    u = _cmc_part_swb(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=cuk fsw=500k vout=-5 vin=12 vref=-0.8 ri=0.1 "
        "mcslope=0.15 tss=120u",
    )
    v1 = Part("Simulation_SPICE", "VDC", value="12", ref="V1")
    L1 = Part("Device", "L", value="22u", ref="L1")   # VIN->SW (node A)
    Cs = Part("Device", "C", value="1u", ref="CS")    # coupling cap SW->SWB
    L2 = Part("Device", "L", value="22u", ref="L2")   # SWB->VOUT (negative rail)
    Co = Part("Device", "C", value="22u", ref="C1")
    Rl = Part("Device", "R", value="10", ref="RL")    # ~0.5 A at -5 V
    Rt = Part("Device", "R", value="42k", ref="RT")
    Rb = Part("Device", "R", value="8k", ref="RB")    # tap = -5*8/50 = -0.8 V
    Rc = Part("Device", "R", value="22k", ref="RC")
    Cc = Part("Device", "C", value="2.2n", ref="CC")

    vin, sw, swb, vout = (Net(nm) for nm in ("VIN", "SW", "SWB", "VOUT"))
    vc, ncc, fb, gnd = (Net(nm) for nm in ("VC", "NCC", "FB", "GND"))
    vin.connect(v1[1], u["VIN"], L1[1])
    gnd.connect(v1[2], u["GND"], Co[2], Rl[2], Rb[2], Cc[2])
    sw.connect(u["SW"], L1[2], Cs[1])              # node A
    swb.connect(u["SWB"], Cs[2], L2[1])            # node B
    vout.connect(u["VOUT"], L2[2], Co[1], Rl[1], Rt[1])
    fb.connect(Rt[2], Rb[1], u["FB"])              # negative tap
    vc.connect(u["VC"], Rc[1])
    ncc.connect(Rc[2], Cc[1])

    from skidl.sim import simulate

    per = 1.0 / 500e3
    try:
        res = simulate().transient_analysis(
            step_time=per / 100, end_time=900e-6, max_time=per / 50, stiff=True,
            use_initial_condition=True, initial_conditions={"VOUT": 0},
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")

    t = np.asarray(res.analysis.time, dtype=float)
    vo = np.asarray(res.analysis["VOUT"], dtype=float)
    fbv = np.asarray(res.analysis["FB"], dtype=float)
    assert np.isfinite(vo).all(), "non-finite VOUT (diverged)"
    vreg = float(vo[t > (t[-1] - 80e-6)].mean())
    fbreg = float(fbv[t > (t[-1] - 80e-6)].mean())
    assert vreg < 0, ("output not negative", vreg)         # a real negative rail
    assert abs(vreg - (-5.0)) / 5.0 <= 0.12, (vreg,)       # regulates near -5 V
    assert fbreg < 0, ("FB not negative", fbreg)           # negative-ref path
    g = np.asarray(res.analysis["U1_gate"], dtype=float)
    rises = int(np.sum((g[:-1] < 2.5) & (g[1:] >= 2.5)))
    assert rises > 50, ("gate not switching", rises)
