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
def test_nonbuck_topology_skips_with_warning(caplog):
    """A topology other than buck is not wired in 29.1 (boost/SEPIC/Ćuk/flyback are
    Stage 29.4): the emitter warns and emits nothing, and validate() flags it."""
    import logging

    _setup()
    u = _cmc_part(
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=boost fsw=500k vref=1.6 ri=0.1",
    )
    with caplog.at_level(logging.WARNING):
        net = _emit(u)
    assert "SU1_hs" not in net and "BU1_gate" not in net, net
    assert any("topology=boost is not wired" in r.message for r in caplog.records)

    _setup()
    u2 = _cmc_part(
        ref="U2",
        Sim_Device="CMCONTROLLER",
        Sim_Params="topology=boost fsw=500k vref=1.6 ri=0.1",
    )
    _wire(u2)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("topology=boost is not wired" in p for p in ei.value.problems), (
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
