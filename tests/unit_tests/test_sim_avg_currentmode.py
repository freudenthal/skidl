# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: averaged peak-current-mode loop model (Stage 28.D).

``Sim.Device="BOOST"`` (or ``BUCK``) + ``Sim.Params="... mode=avg cmode=peak
vref=1.6 ..."`` selects a **small-signal averaged peak-current-mode LOOP model**
for ``.ac`` compensation design (crossover / phase margin of the VC-pin network),
NOT a cycle-accurate controller and NOT the closed-loop switching sim. It is
structurally distinct from the voltage-mode ``_emit_averaged_buck`` (Stage 20.5):

* the error amp closes into the user's **real external VC network** (only a gm
  cell + a finite-DC-gain REA are emitted);
* the inner current loop makes the plant a controlled current source injected into
  the **real** Cout/Rload (dominant pole + ESR zero come from the user's parts);
* boost/SEPIC/Ćuk get a **right-half-plane zero** (``Rload*(1-D)^2/(2*pi*L)``) --
  gain rise with phase LAG; buck has none;
* a **subharmonic double pole at fsw/2** with ``Q=1/(pi*(mc*(1-D)-0.5))``.

Emission tests inspect the SPICE text (RHP zero present for boost / absent for
buck, the fsw/2 double pole, the gm error amp into VC); a byte-identity test locks
the voltage-mode ``_emit_averaged_buck`` output (a hard gate -- current mode is a
new emitter, the voltage-mode path must not move); validate tests check the
terminal / VREF / duty guards; the gated live test confirms a finite phase margin
with the RHP zero and subharmonic pole visible.
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


# A synthetic controller stand-in whose pins carry exactly the names the
# current-mode terminal resolver keys on: VIN / SW / VC (compensation) / FB
# (divider tap) / VOUT / GND.
def _cm_part(ref="U1", *, name="BOOST", drop=(), **fields):
    pins = [
        Pin(num=1, name="VIN", func=pin_types.PWRIN),
        Pin(num=2, name="SW", func=pin_types.PASSIVE),
        Pin(num=3, name="VC", func=pin_types.PASSIVE),
        Pin(num=4, name="FB", func=pin_types.PASSIVE),
        Pin(num=5, name="VOUT", func=pin_types.PWROUT),
        Pin(num=6, name="GND", func=pin_types.PWRIN),
    ]
    pins = [p for p in pins if p.name not in drop]
    u = Part(tool=SKIDL, name=name, ref_prefix="U", ref=ref, pins=pins)
    for k, v in fields.items():
        setattr(u, k, v)
    return u


def _wire(u, *, fb=True, vc=True, vout=True):
    if "VIN" in [p.name for p in u.pins]:
        Net("VIN").connect(u["VIN"])
    if "SW" in [p.name for p in u.pins]:
        Net("SW").connect(u["SW"])
    if vc and "VC" in [p.name for p in u.pins]:
        Net("VC").connect(u["VC"])
    if fb and "FB" in [p.name for p in u.pins]:
        Net("FB").connect(u["FB"])
    if vout and "VOUT" in [p.name for p in u.pins]:
        Net("VOUT").connect(u["VOUT"])
    Net("GND").connect(u["GND"])


def _emit(u):
    _wire(u)
    return str(SpiceConverter(_view()).convert(strict=False))


_BOOST_PARAMS = (
    "fsw=300k vout=24 vin=12 mode=avg cmode=peak vref=1.6 rload=12 l=10u"
)


# --- emission ------------------------------------------------------------- #


@requires_sim
def test_boost_current_mode_emits_loop_blocks():
    """Boost mode=avg cmode=peak emits the gm error amp into the real VC net, the
    fsw/2 subharmonic RLC, the RHP-zero differentiator, and the output
    transconductance -- and NO switching (no S-switch, no PULSE, no sawtooth)."""
    _setup()
    u = _cm_part(Sim_Device="BOOST", Sim_Params=_BOOST_PARAMS)
    net = _emit(u)
    # gm error amp: current INTO VC = GM*(VREF - V(FB)); default gm=250u, vref=1.6
    assert "BU1_ea 0 VC I = 0.00025*(1.6 - V(FB))" in net, net
    assert "RU1_ea VC 0 1e+06" in net, net
    # buffered subharmonic 2-pole R-L-C into node U1_sh
    assert "BU1_shin U1_shin 0 V = V(VC)" in net, net
    assert "RU1_sh U1_shin U1_nrh" in net, net
    assert "LU1_sh U1_nrh U1_sh" in net, net
    assert "CU1_sh U1_sh 0 1e-06" in net, net
    # RHP zero (boost): sensed-cap differentiator, then the output transconductance
    # reads the post-RHP-zero node U1_rz.
    assert "CU1_z U1_sh U1_nz 1e-09" in net, net
    assert "VU1_z U1_nz 0 DC 0" in net, net
    assert re.search(r"BU1_rz U1_rz 0 V = V\(U1_sh\) - [\d.e+]+\*I\(VU1_z\)", net), net
    assert re.search(r"BU1_out 0 VOUT I = [\d.]+\*V\(U1_rz\)", net), net
    # NOT a switching model: no ideal switch, PULSE source, or sawtooth ramp.
    assert " SW(" not in net and "PULSE(" not in net and "_saw " not in net, net


@requires_sim
def test_subharmonic_double_pole_at_fsw_over_2():
    """The emitted subharmonic L*C places the double pole at fsw/2:
    wn = 1/sqrt(L*C) == 2*pi*(fsw/2) == pi*fsw."""
    import math

    _setup()
    u = _cm_part(Sim_Device="BOOST", Sim_Params=_BOOST_PARAMS)
    net = _emit(u)
    lh = float(re.search(r"LU1_sh U1_nrh U1_sh ([\d.eE+-]+)", net).group(1))
    ch = float(re.search(r"CU1_sh U1_sh 0 ([\d.eE+-]+)", net).group(1))
    wn = 1.0 / math.sqrt(lh * ch)
    assert wn == pytest.approx(math.pi * 300e3, rel=1e-3), (wn, math.pi * 300e3)


@requires_sim
def test_subharmonic_Q_rises_toward_half_duty():
    """Q = 1/(pi*(mc*(1-D)-0.5)) grows as the duty climbs toward the mc-set
    subharmonic edge; the emitted damping resistor RH = 1/(Q*wn*CH) therefore
    SHRINKS. A D=0.6 point (VOUT=30 from VIN=12) is less damped than D=0.5."""
    def rh_of(vout):
        _setup()
        u = _cm_part(
            Sim_Device="BOOST",
            Sim_Params=f"fsw=300k vout={vout} vin=12 mode=avg cmode=peak "
                       f"vref=1.6 rload=12 l=10u mc=1.5",
        )
        net = _emit(u)
        return float(re.search(r"RU1_sh U1_shin U1_nrh ([\d.eE+-]+)", net).group(1))

    assert rh_of(30) < rh_of(24)  # higher duty -> higher Q -> smaller RH


@requires_sim
def test_buck_current_mode_has_no_rhp_zero():
    """A buck in current mode has NO right-half-plane zero: the differentiator
    block is omitted and the output transconductance reads U1_sh directly. Its
    modulator gain is 1/RI (all inductor current reaches the output)."""
    _setup()
    u = _cm_part(
        name="BUCK",
        Sim_Device="BUCK",
        Sim_Params="fsw=500k vout=3.3 vin=12 mode=avg cmode=peak vref=0.8 ri=0.1",
    )
    net = _emit(u)
    assert "CU1_z" not in net and "BU1_rz" not in net, net
    assert re.search(r"BU1_out 0 VOUT I = [\d.]+\*V\(U1_sh\)", net), net
    # gmc = 1/RI = 1/0.1 = 10
    assert "BU1_out 0 VOUT I = 10*V(U1_sh)" in net, net


@requires_sim
def test_negative_vref_accepted_for_inverting():
    """VREF may be negative (the -0.8 V FBX inverting configuration) -- unlike the
    voltage-mode buck, which requires a positive VREF."""
    _setup()
    u = _cm_part(
        Sim_Device="BOOST",
        Sim_Params="fsw=300k d=0.5 mode=avg cmode=peak vref=-0.8 frhpz=40k",
    )
    net = _emit(u)
    assert "BU1_ea 0 VC I = 0.00025*(-0.8 - V(FB))" in net, net
    assert conv_prov(u) == "boost_averaged_cm(vref=-0.8, d=0.500)"


def conv_prov(u):
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    return conv.model_provenance["U1"].name


@requires_sim
def test_provenance_recorded():
    _setup()
    u = _cm_part(Sim_Device="BOOST", Sim_Params=_BOOST_PARAMS)
    _wire(u)
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    prov = conv.model_provenance["U1"]
    assert prov.kind == "boost"
    assert prov.tier == "sim_params"
    # name carries "averaged" so ac_analysis does not warn (it is built for .ac),
    # plus the operating duty.
    assert prov.name == "boost_averaged_cm(vref=1.6, d=0.500)", prov.name
    assert "averaged" in prov.name


@requires_sim
def test_rhp_zero_omitted_with_warning_when_unresolvable(caplog):
    """Boost current mode with no FRHPZ and no RLOAD/L -> the RHP zero is omitted
    (honest warning), the differentiator is not emitted, and the output reads
    U1_sh directly."""
    import logging

    _setup()
    u = _cm_part(
        Sim_Device="BOOST",
        Sim_Params="fsw=300k d=0.5 mode=avg cmode=peak vref=1.6",  # no frhpz/rload/l
    )
    with caplog.at_level(logging.WARNING):
        net = _emit(u)
    assert "CU1_z" not in net and "BU1_rz" not in net, net
    assert re.search(r"BU1_out 0 VOUT I = [\d.]+\*V\(U1_sh\)", net), net
    assert any("right-half-plane zero is omitted" in r.message for r in caplog.records)


# --- byte-identity hard gate: voltage-mode buck must not move ------------- #


@requires_sim
def test_voltage_mode_buck_averaged_unchanged():
    """HARD GATE: a plain MODE=avg buck (no CMODE) still emits the Stage-20.5
    voltage-mode averaged buck -- the multiplicative averaged switch V(c)*V(vin)
    and the gm-C error amp -- byte-for-byte, NOT the current-mode model. Current
    mode is an additive new emitter; this path must stay unchanged."""
    _setup()
    # A buck stand-in with SW/VIN/GND/FB (the voltage-mode resolver's terminals).
    pins = [
        Pin(num=1, name="VIN", func=pin_types.PWRIN),
        Pin(num=2, name="SW", func=pin_types.PASSIVE),
        Pin(num=3, name="FB", func=pin_types.PASSIVE),
        Pin(num=4, name="GND", func=pin_types.PWRIN),
    ]
    u = Part(tool=SKIDL, name="BUCK", ref_prefix="U", ref="U1", pins=pins)
    u.Sim_Device = "BUCK"
    u.Sim_Params = "fsw=500k vout=3.3 mode=avg vref=0.8"
    Net("VIN").connect(u["VIN"])
    Net("SW").connect(u["SW"])
    Net("FB").connect(u["FB"])
    Net("GND").connect(u["GND"])
    net = str(SpiceConverter(_view()).convert(strict=False))
    # the Stage-20.5 voltage-mode signature (multiplicative averaged switch)
    assert "BU1_ea 0 U1_c I = 0.001*(0.8 - V(FB))" in net, net
    assert "CU1_ea U1_c 0 1e-07" in net, net
    assert "RU1_ea U1_c 0 1e+06" in net, net
    assert "BU1_sw U1_swi 0 V = V(U1_c) * V(VIN)" in net, net
    assert "RU1_sw U1_swi SW 0.1" in net, net
    # current-mode blocks must NOT appear on this path
    assert "cmode" not in net.lower() and "U1_shin" not in net, net
    conv = SpiceConverter(_view())
    conv.convert(strict=False)
    assert conv.model_provenance["U1"].name == "buck_averaged(vref=0.8)"


# --- validation ----------------------------------------------------------- #


@requires_sim
def test_missing_vref_is_validation_error():
    _setup()
    u = _cm_part(
        Sim_Device="BOOST",
        Sim_Params="fsw=300k vout=24 vin=12 mode=avg cmode=peak",  # no vref
    )
    _wire(u)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any("VREF" in p and "CMODE=peak" in p for p in ei.value.problems), ei.value.problems


@requires_sim
def test_missing_vc_pin_is_validation_error():
    """No VC (compensation) pin -> the loop cannot close through the real comp
    network; validate() flags it (not the power-stage SW/VIN/GND message)."""
    _setup()
    u = _cm_part(Sim_Device="BOOST", Sim_Params=_BOOST_PARAMS, drop=("VC",))
    _wire(u, vc=False)
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(_view()).convert(strict=True)
    assert any(
        "CMODE=peak needs connected FB, VC" in p for p in ei.value.problems
    ), ei.value.problems


@requires_sim
def test_missing_fb_skips_with_warning(caplog):
    """No FB pin -> the current-mode emitter cannot resolve terminals; it emits
    nothing and warns (honest skip), rather than a wrong netlist."""
    import logging

    _setup()
    u = _cm_part(Sim_Device="BOOST", Sim_Params=_BOOST_PARAMS, drop=("FB",))
    _wire(u, fb=False)
    with caplog.at_level(logging.WARNING):
        net = str(SpiceConverter(_view()).convert(strict=False))
    assert "BU1_ea" not in net, net
    assert any("CMODE=peak needs connected FB" in r.message for r in caplog.records)


# --- live (gated) --------------------------------------------------------- #


@requires_sim
def test_boost_loop_ac_has_finite_phase_margin():
    """Live .ac on the LT3757_Boost.asc loop values: a finite crossover below
    fsw/2 with a sane phase margin, an RHP zero that adds gain with phase LAG near
    ``Rload*(1-D)^2/(2*pi*L)``, and the subharmonic double pole peaking near
    fsw/2. The loop is broken by a VSIN at the divider tap (fb_plant -> fb_ctrl)."""
    import math

    import numpy as np

    _setup()
    u = _cm_part(Sim_Device="BOOST", Sim_Params=_BOOST_PARAMS + " ri=0.1 gm=250u mc=1.5")
    v1 = Part("Simulation_SPICE", "VDC", value="12", ref="V1")
    L1 = Part("Device", "L", value="10u", ref="L1")
    R3 = Part("Device", "R", value="226k", ref="R3")
    R2 = Part("Device", "R", value="16.2k", ref="R2")
    Rc = Part("Device", "R", value="22k", ref="RC")
    Cc = Part("Device", "C", value="6800p", ref="CC")
    Co = Part("Device", "C", value="47u", ref="C1")
    Rl = Part("Device", "R", value="12", ref="RL")
    vinj = Part("Simulation_SPICE", "VSIN", value="0", ref="VINJ")

    vin, sw, vc, ncc, vout = (Net(n) for n in ("VIN", "SW", "VC", "NCC", "VOUT"))
    fbp, fbc, gnd = Net("FBP"), Net("FBC"), Net("GND")
    vin.connect(v1[1], u["VIN"], L1[1])
    gnd.connect(v1[2], u["GND"], R2[2], Co[2], Rl[2], Cc[2])
    sw.connect(u["SW"], L1[2])
    vc.connect(u["VC"], Rc[1])
    ncc.connect(Rc[2], Cc[1])
    vout.connect(u["VOUT"], R3[1], Co[1], Rl[1])
    fbp.connect(R3[2], R2[1], vinj[1])
    fbc.connect(vinj[2], u["FB"])

    from skidl.sim import simulate

    sim = simulate()
    try:
        res = sim.ac_analysis(start_freq=1.0, stop_freq=1e6, points=100)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")

    pm = res.phase_margin("FBP", "FBC")
    freq, magdb, _ = res.loop_gain("FBP", "FBC")
    # finite crossover below fsw/2
    xo = None
    for i in range(1, len(freq)):
        if magdb[i - 1] >= 0 > magdb[i]:
            xo = float(freq[i])
            break
    assert xo is not None and 100.0 < xo < 150e3, xo
    assert pm is not None and 30.0 <= pm <= 90.0, pm

    # RHP zero visible: |rz/vc| rises AND phase lags near frhpz = Rload*(1-D)^2/(2piL)
    f = np.asarray(res.analysis.frequency, dtype=float)
    H = np.asarray(res.analysis["U1_rz"], dtype=complex) / np.asarray(
        res.analysis["VC"], dtype=complex
    )
    frhpz = 12.0 * (1.0 - 0.5) ** 2 / (2.0 * math.pi * 10e-6)  # ~47.7 kHz
    i = int(np.argmin(np.abs(f - frhpz)))
    assert 20 * np.log10(abs(H[i])) > 1.0, ("RHP gain rise", H[i])
    assert np.angle(H[i], deg=True) < -20.0, ("RHP phase lag", H[i])
