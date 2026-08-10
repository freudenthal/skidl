# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: the tunable op-amp macromodel, ``Sim.Device="OPAMP"`` (Stage 9.6).

The pre-existing op-amp paths -- the ideal VCVS and the one-pole ``Sim.Gbw``
model -- give no slew rate, no input offset, no output clamp and no Rin/Rout.
``Sim.Device="OPAMP"`` opts a part into a Boyle-style behavioral macromodel that
adds all of them, parameterized from ``Sim.Params`` or from a datasheet row in
``OPAMP_PROFILES``.

Each test below drives ONE parameter to a value with a closed-form consequence
and measures that consequence, so a passing bench means the emitted netlist
realizes the requested numbers -- not merely that it simulates:

===  ===========================================================
B1   open-loop Bode: DC gain = aol, unity crossing = gbw, -20 dB/dec
B2   follower closed-loop -3 dB = gbw
B3   large-step slew rate = sr; a small step is exponential, not a ramp
B4   follower, input grounded -> output = vos
B5   output clamps at V(rail) -/+ headroom; unwired rails -> no clamp
B6   fp2 adds the expected extra phase lag at gbw
B7   profile resolution, and an explicit param overriding one key
B8   no regression: the ideal and Sim.Gbw paths are untouched
===  ===========================================================

Net names avoid every rail token the converter's supply heuristic injects on
(VIN/VSUPPLY/VCC_5V/+5V/3V3): the op-amp rails here are ``VSPOS``/``VSNEG``,
driven by explicit VDC parts.
"""

import math

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import OPAMP_PROFILES, SpiceConverter

    HAS_SIM = True
except Exception:
    HAS_SIM = False
    OPAMP_PROFILES = {}

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)

OPAMP_LIB, OPAMP_SYM = "Amplifier_Operational", "LM358"
VS = 15.0


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _rails(u, gnd, vs=VS):
    """Drive the op-amp's supply pins from explicit VDC sources."""
    vpos, vneg = Net("VSPOS"), Net("VSNEG")
    vp = Part("Simulation_SPICE", "VDC", ref="VP", value=f"{vs}V")
    vn = Part("Simulation_SPICE", "VDC", ref="VN", value=f"{vs}V")
    vpos.connect(vp[1], u["V+"])
    gnd.connect(vp[2], vn[1])
    vneg.connect(vn[2], u["V-"])


def _opamp(ref="U1", **fields):
    """A macromodel op-amp with NO profile in play.

    ``value`` is set to a name absent from OPAMP_PROFILES on purpose: skidl
    defaults a Part's value to its symbol name, so leaving it unset would make
    every ``Amplifier_Operational:LM358`` here silently pick up the LM358
    datasheet row (2 mV offset and all) instead of the parameters the test
    asked for. Profile resolution is exercised deliberately in B7.
    """
    fields.setdefault("value", "BENCH_GENERIC")
    return Part(OPAMP_LIB, OPAMP_SYM, ref=ref, Sim_Device="OPAMP", **fields)


def _ac(start, stop, points=200):
    from skidl.sim import simulate

    try:
        return simulate().ac_analysis(start_freq=start, stop_freq=stop, points=points)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")


def _tran(step, end, **kw):
    from skidl.sim import simulate

    try:
        return simulate().transient_analysis(step_time=step, end_time=end, **kw)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")


def _follower(params, ac=True, step_v=None, tr=1e-9):
    """Unity-gain follower around the macromodel. Input SIG, output OUT."""
    _setup()
    u = _opamp(Sim_Params=params)
    sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
    if ac:
        src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    else:
        src = Part(
            "Simulation_SPICE", "VPULSE", ref="V1", value=f"{step_v}V",
            Sim_Params=(
                f"v1=0 v2={step_v} td=0 tr={tr} tf={tr} pw=1 per=2"
            ),
        )
    sig.connect(src[1], u[3])   # unit-A "+"
    gnd.connect(src[2])
    out.connect(u[1], u[2])     # OUT tied back to "-"
    _rails(u, gnd)
    return u


# --- B1: open-loop Bode ---------------------------------------------------- #


@requires_sim
def test_b1_open_loop_bode_matches_aol_and_gbw():
    """Bare macromodel, no feedback: DC gain is ``aol`` and |A| crosses unity
    at ``gbw``, with a single-pole -20 dB/decade slope in between."""
    _setup()
    aol, gbw = 200e3, 3e6
    # vos=0 is required OPEN loop: with any offset, aol*vos drives the output
    # into the clamp at DC (200k x 3 mV = 600 V demanded), the AC linearization
    # sits on a saturated clamp, and |A| is identically zero. That is correct
    # physics, not a model defect -- a real op-amp open-loop is railed too.
    u = _opamp(Sim_Params=f"aol={aol:g} gbw={gbw:g} vos=0 rout=1")
    sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
    src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    sig.connect(src[1], u[3])
    gnd.connect(src[2], u[2])   # "-" grounded: open loop
    out.connect(u[1])
    _rails(u, gnd)

    res = _ac(0.1, 1e8, points=100)
    freq, mag_db, _ = res.bode("OUT")

    assert abs(mag_db[0] - 20 * math.log10(aol)) < 0.5, mag_db[0]

    # unity (0 dB) crossing == gbw
    xo = None
    for i in range(1, len(freq)):
        if mag_db[i - 1] >= 0.0 > mag_db[i]:
            t = (0.0 - mag_db[i - 1]) / (mag_db[i] - mag_db[i - 1])
            xo = 10 ** (
                math.log10(freq[i - 1])
                + t * (math.log10(freq[i]) - math.log10(freq[i - 1]))
            )
            break
    assert xo is not None, "|A| never crosses 0 dB"
    assert abs(xo - gbw) / gbw < 0.05, (xo, gbw)

    # -20 dB/decade across the decade centred on gbw/10
    def _at(f):
        i = min(range(len(freq)), key=lambda k: abs(freq[k] - f))
        return mag_db[i]

    slope = _at(gbw / 3.0) - _at(gbw / 30.0)
    assert abs(slope + 20.0) < 1.0, slope


# --- B2: closed-loop bandwidth --------------------------------------------- #


@requires_sim
def test_b2_follower_bandwidth_equals_gbw():
    """A unity-gain follower's -3 dB corner is the gain-bandwidth product."""
    gbw = 3e6
    _follower(f"aol=200k gbw={gbw:g} rout=1")
    fc = _ac(1e3, 1e9, points=200).cutoff_frequency("OUT")
    assert fc is not None
    assert abs(fc - gbw) / gbw < 0.10, (fc, gbw)


# --- B3: slew rate ---------------------------------------------------------- #


@requires_sim
def test_b3_large_step_slews_at_sr_and_small_step_does_not():
    """Large step: the output is a straight ramp of slope ``sr``. Small step:
    the response is the closed-loop exponential, well under the slew limit."""
    import numpy as np

    sr, gbw = 13e6, 3e6  # 13 V/us
    _follower(f"aol=200k gbw={gbw:g} sr={sr:g} rout=1", ac=False, step_v=10.0)
    res = _tran(1e-9, 4e-6)
    t, v = res._node_series("OUT")

    # Slope over the middle of the transition (20 %..80 % of the 10 V step).
    lo = np.where(v >= 2.0)[0]
    hi = np.where(v >= 8.0)[0]
    assert len(lo) and len(hi), (v.min(), v.max())
    slope = (v[hi[0]] - v[lo[0]]) / (t[hi[0]] - t[lo[0]])
    assert abs(slope - sr) / sr < 0.05, (slope, sr)

    # A small step (10 mV) is nowhere near the slew limit: its 10-90 rise time
    # is set by the closed-loop pole (0.35/gbw), not by sr.
    _follower(f"aol=200k gbw={gbw:g} sr={sr:g} rout=1", ac=False, step_v=0.010)
    m = _tran(1e-10, 2e-6).step_metrics("OUT")
    expected_tr = 0.35 / gbw
    assert m["rise_time"] is not None
    assert abs(m["rise_time"] - expected_tr) / expected_tr < 0.25, (
        m["rise_time"], expected_tr
    )
    slew_limited_tr = 0.8 * 0.010 / sr
    assert m["rise_time"] > 10 * slew_limited_tr, (m["rise_time"], slew_limited_tr)


# --- B4: input offset ------------------------------------------------------- #


@requires_sim
def test_b4_offset_appears_at_the_follower_output():
    """Follower with its non-inverting input grounded settles at ``+vos``."""
    from skidl.sim import simulate

    _setup()
    vos = 3e-3
    u = _opamp(Sim_Params=f"aol=200k gbw=3Meg vos={vos:g} rout=1")
    out, gnd = Net("OUT"), Net("GND")
    gnd.connect(u[3])
    out.connect(u[1], u[2])
    _rails(u, gnd)

    try:
        res = simulate().operating_point()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    v = res.get_voltage("OUT")
    assert abs(v - vos) / vos < 0.02, (v, vos)


# --- B5: output clamp ------------------------------------------------------- #


@requires_sim
def test_b5_output_clamps_at_the_rails_with_headroom():
    """Gain of 10 driven past the rails saturates at ``V(rail) -/+ headroom``."""
    from skidl.sim import simulate

    _setup()
    voh, vol = 1.5, 1.5
    u = _opamp(Sim_Params=f"aol=200k gbw=3Meg voh={voh:g} vol={vol:g} rout=1")
    sig, ninv, out, gnd = Net("SIG"), Net("NINV"), Net("OUT"), Net("GND")
    # Non-inverting gain of 10 from a 5 V input -> 50 V demanded, rails at +-15.
    src = Part("Simulation_SPICE", "VDC", ref="V1", value="5V")
    rg = Part("Device", "R", ref="RG", value="1k")
    rf = Part("Device", "R", ref="RF", value="9k")
    sig.connect(src[1], u[3])
    gnd.connect(src[2], rg[1])
    ninv.connect(rg[2], rf[1], u[2])
    out.connect(rf[2], u[1])
    _rails(u, gnd)

    try:
        res = simulate().operating_point()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    v = res.get_voltage("OUT")
    assert abs(v - (VS - voh)) < 0.05, (v, VS - voh)


@requires_sim
def test_b5_no_supply_pins_means_no_clamp_and_a_provenance_note():
    """With the supply pins unwired and no ``vclamp_*``, the model must NOT
    invent a rail: it emits unclamped and says so in its provenance."""
    from skidl.sim import simulate

    _setup()
    u = _opamp(Sim_Params="aol=200k gbw=3Meg rout=1")
    sig, ninv, out, gnd = Net("SIG"), Net("NINV"), Net("OUT"), Net("GND")
    src = Part("Simulation_SPICE", "VDC", ref="V1", value="5V")
    rg = Part("Device", "R", ref="RG", value="1k")
    rf = Part("Device", "R", ref="RF", value="9k")
    sig.connect(src[1], u[3])
    gnd.connect(src[2], rg[1])
    ninv.connect(rg[2], rf[1], u[2])
    out.connect(rf[2], u[1])
    # deliberately no _rails(): V+ / V- left unconnected

    sim = simulate()
    prov = sim.model_provenance["U1"]
    assert "UNCLAMPED" in prov.name, prov.name
    try:
        res = sim.operating_point()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    # 5 V x 10 with nothing to stop it
    assert abs(res.get_voltage("OUT") - 50.0) < 0.5, res.get_voltage("OUT")


# --- B6: second pole -------------------------------------------------------- #


@requires_sim
def test_b6_second_pole_adds_the_expected_phase_lag():
    """``fp2 = 3*gbw`` costs ``atan(gbw/fp2) = 18.4 deg`` of extra lag at gbw."""
    _setup()
    gbw = 3e6
    fp2 = 3 * gbw

    def _phase_at(params, f):
        _setup()
        u = _opamp(Sim_Params=params)
        sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
        src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
        sig.connect(src[1], u[3])
        gnd.connect(src[2], u[2])
        out.connect(u[1])
        _rails(u, gnd)
        freq, _mag, ph = _ac(1e4, 1e8, points=200).bode("OUT")
        i = min(range(len(freq)), key=lambda k: abs(freq[k] - f))
        return ph[i]

    single = _phase_at(f"aol=200k gbw={gbw:g} vos=0 rout=1", gbw)
    double = _phase_at(f"aol=200k gbw={gbw:g} vos=0 fp2={fp2:g} rout=1", gbw)
    extra = single - double
    expected = math.degrees(math.atan(gbw / fp2))  # 18.43 deg
    assert abs(extra - expected) < 3.0, (extra, expected, single, double)


# --- B7: profile resolution ------------------------------------------------- #


@requires_sim
def test_b7_profile_resolves_from_value_and_params_override_one_key():
    """``value="TL072"`` picks the profile row (tier ``datasheet_fit``); an
    explicit ``Sim.Params`` key overrides only that key and moves the tier to
    ``sim_params``."""
    conv = SpiceConverter.__new__(SpiceConverter)

    class _C:
        _extra_fields = {}
        ref = "U1"

    params, tier, name = conv._opamp_macro_params(_C(), "TL072")
    assert tier == "datasheet_fit", tier
    assert name == "TL072", name
    assert params["gbw"] == OPAMP_PROFILES["TL072"]["gbw"]
    assert params["sr"] == OPAMP_PROFILES["TL072"]["sr"]

    class _D:
        _extra_fields = {"Sim_Params": "gbw=1Meg"}
        ref = "U1"

    params2, tier2, name2 = conv._opamp_macro_params(_D(), "TL072")
    assert tier2 == "sim_params", tier2
    assert name2 == "TL072"
    assert params2["gbw"] == 1e6, params2["gbw"]
    # every other key still comes from the profile
    assert params2["sr"] == OPAMP_PROFILES["TL072"]["sr"]
    assert params2["aol"] == OPAMP_PROFILES["TL072"]["aol"]


@requires_sim
def test_b7_unknown_chip_falls_back_to_defaults_not_a_wrong_part():
    conv = SpiceConverter.__new__(SpiceConverter)

    class _C:
        _extra_fields = {"Sim_Model": "NOT_A_REAL_PART_XYZ"}
        ref = "U1"

    params, tier, name = conv._opamp_macro_params(_C(), "")
    assert tier == "generic", tier
    assert name is None
    assert params["gbw"] == SpiceConverter._OPAMP_PARAM_DEFAULTS["gbw"]


@requires_sim
def test_b7_manufacturer_models_rows_are_reachable():
    """Stage 9.6's dangling wire: the ``ManufacturerModels`` op-amp rows carried
    GBW/SR/AOL/VOS/RIN/ROUT but nothing could reach them. A chip present there
    and absent from ``OPAMP_PROFILES`` must now resolve."""
    conv = SpiceConverter.__new__(SpiceConverter)
    row = conv._opamp_manufacturer_row("TL072")
    assert row, "TL072_TI row not reachable from ManufacturerModels"
    assert row["gbw"] == 3e6 and row["sr"] == 13e6, row

    # A row that is not an op-amp contributes nothing rather than an
    # all-default pseudo-profile. (TL431 is a shunt reference: it does carry a
    # ROUT, which is why the test is "no GBW -> not an op-amp" and not "carries
    # none of our parameter names".)
    assert conv._opamp_manufacturer_row("TL431") == {}


@requires_sim
def test_b7_si_suffixes_reach_giga_and_tera():
    """``sr=1G`` and ``rin=1T`` must parse -- the op-amp parameter set genuinely
    reaches that far."""
    assert SpiceConverter._parse_si_number("1G") == 1e9
    assert SpiceConverter._parse_si_number("1T") == 1e12
    assert SpiceConverter._parse_si_number("13Meg") == 13e6
    assert SpiceConverter._parse_si_number("3m") == 3e-3


@requires_sim
def test_nonpositive_parameters_do_not_emit_a_dead_amplifier():
    """``aol``/``gbw`` of zero would divide by zero or emit an amplifier with no
    gain -- one that simulates happily and measures nothing. Both fall back to
    the defaults; ``sr=0`` means "no slew limit", not "frozen output"."""
    from skidl.sim import simulate

    def _text(params):
        _setup()
        u = _opamp(Sim_Params=params)
        sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
        src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
        sig.connect(src[1], u[3])
        gnd.connect(src[2])
        out.connect(u[1], u[2])
        _rails(u, gnd)
        return simulate().get_netlist()

    # sr=0 -> the gain cell has no min/max clamp at all, but still drives
    good = _text("aol=200k gbw=3Meg sr=0 rout=1")
    gm_line = next(l for l in good.splitlines() if l.startswith("BU1_gm"))
    assert "min(" not in gm_line, gm_line
    assert "0.001*V(" in gm_line, gm_line

    # aol=0 -> defaults, and the emitted RP1 is aol_default/GM = 2e8
    zero_aol = _text("aol=0 gbw=3Meg rout=1")
    assert "RU1_p1 U1_p1 0 200000000.0" in zero_aol, [
        l for l in zero_aol.splitlines() if l.startswith("RU1_p1")
    ]


# --- B8: no regression on the pre-existing paths ---------------------------- #


@requires_sim
def test_b8_ideal_and_gbw_paths_are_unchanged():
    """Without ``Sim.Device="OPAMP"`` a part still gets the ideal VCVS (or the
    1-pole model with ``Sim.Gbw``) -- byte-identical netlist text either way."""
    from skidl.sim import simulate

    def _netlist(**fields):
        _setup()
        u = Part(OPAMP_LIB, OPAMP_SYM, ref="U1", **fields)
        sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
        src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
        sig.connect(src[1], u[3])
        gnd.connect(src[2])
        out.connect(u[1], u[2])
        _rails(u, gnd)
        s = simulate()
        return s.get_netlist(), s.model_provenance["U1"]

    ideal_text, ideal_prov = _netlist()
    assert ideal_prov.name == "ideal_vcvs", ideal_prov.name
    assert ideal_prov.tier == "generic"
    assert "eu1 " in ideal_text.lower(), ideal_text

    gbw_text, gbw_prov = _netlist(Sim_Gbw="3Meg")
    assert gbw_prov.name.startswith("gbw_1pole("), gbw_prov.name
    assert gbw_prov.tier == "sim_params"
    # the 1-pole model's two internal nodes and unity buffer
    assert "u1_p1" in gbw_text.lower() and "u1_p2" in gbw_text.lower()

    # and the macromodel is a genuinely different emission
    macro_text, macro_prov = _netlist(Sim_Device="OPAMP", Sim_Params="gbw=3Meg")
    assert macro_prov.name.startswith("opamp_macro("), macro_prov.name
    assert "bu1_gm" in macro_text.lower(), macro_text
    assert macro_text != ideal_text and macro_text != gbw_text


# =========================================================================== #
# Stage 9.7 -- input-referred noise (``en`` / ``inoise``)                     #
# =========================================================================== #
#
# Everything below is graded against a closed-form noise budget:
#
# ===  =====================================================================
# N1   en/inoise unset -> the emitted deck is byte-identical to pre-9.7
# N2   the residual floor with en unset, and where it actually comes from
# N3   a follower reads sqrt(en^2 + floor^2); N5 doubling en scales it
# N4   a non-inverting x10 reads the full budget x noise gain; N8 adds inoise
# N6   profile / ManufacturerModels / explicit-param resolution of en
# N7   a closed-loop .op still solves with the generators emitted
# ===  =====================================================================

_K_B = 1.380649e-23
_T_REF = 298.15

#: Fast, high-loop-gain macromodel so the residual floor sits far below ``en``.
_NOISE_BASE = "aol=1e6 gbw=10Meg sr=100Meg vos=0 rout=100"


def _noise(output="OUT", source="V1", start=1e3, stop=1e5, points=10):
    from skidl.sim import simulate

    try:
        return simulate().noise_analysis(output, source, start, stop, points=points)
    except (RuntimeError, OSError, ImportError) as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")


def _noise_follower(params):
    from skidl.sim import simulate

    _setup()
    u = _opamp(Sim_Params=params)
    sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
    src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    sig.connect(src[1], u[3])
    gnd.connect(src[2])
    out.connect(u[1], u[2])
    _rails(u, gnd)
    return simulate()


def _noninv(params, rf=9e3, rg=1e3):
    """Non-inverting amplifier, noise gain ``1 + rf/rg``."""
    from skidl.sim import simulate

    _setup()
    u = _opamp(Sim_Params=params)
    sig, out, ninv, gnd = Net("SIG"), Net("OUT"), Net("NINV"), Net("GND")
    src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    r_f = Part("Device", "R", ref="RF", value=f"{rf:.10g}",
               footprint="Resistor_SMD:R_0603_1608Metric")
    r_g = Part("Device", "R", ref="RG", value=f"{rg:.10g}",
               footprint="Resistor_SMD:R_0603_1608Metric")
    sig.connect(src[1], u[3])
    gnd.connect(src[2])
    out.connect(u[1], r_f[1])
    ninv.connect(r_f[2], r_g[1], u[2])
    gnd.connect(r_g[2])
    _rails(u, gnd)
    return simulate()


# --- N1: byte-identity when the noise keys are unset ----------------------- #


@requires_sim
def test_n1_unset_noise_keys_emit_a_byte_identical_deck():
    """The frozen-behavior gate: adding ``en``/``inoise`` to the parameter
    table must not change one character of a deck that does not use them."""
    baseline = _noise_follower(_NOISE_BASE).get_netlist()

    # the exact pre-9.7 emission, node for node -- an accidental extra element
    # or a renamed node fails here rather than three benches downstream
    assert "U1_nv" not in baseline, baseline
    assert "U1_ni" not in baseline, baseline
    assert "VU1_os U1_inp SIG" in baseline, baseline
    assert baseline == _noise_follower(_NOISE_BASE + " en=0").get_netlist()

    with_en = _noise_follower(_NOISE_BASE + " en=18n").get_netlist()
    assert with_en != baseline
    assert "RU1_nv U1_nv 0 19677" in with_en, with_en
    assert "EU1_nv U1_nvi SIG U1_nv 0 1.0" in with_en, with_en
    assert "VU1_os U1_inp U1_nvi" in with_en, with_en

    # removing the two added lines gets the baseline back: the generator is
    # purely additive, it rewires nothing else
    stripped = [
        line.replace("U1_nvi", "SIG")
        for line in with_en.splitlines()
        if not line.startswith(("RU1_nv ", "EU1_nv "))
    ]
    assert stripped == baseline.splitlines(), stripped


# --- N2: the residual floor, and what it is made of ------------------------ #


@requires_sim
def test_n2_floor_without_en_is_the_loop_suppressed_rp1_and_rout():
    """With ``en`` unset the macromodel is not silent -- but it is close.

    The survey expected rout's own ``sqrt(4kT*rout)`` = 1.3 nV/rtHz. Measured,
    the follower's floor is ~300x SMALLER, because both noisy resistors sit
    INSIDE the feedback loop and are divided by the loop gain, and because the
    dominant term is not rout at all but ``RP1`` (the dominant-pole resistor,
    aol/gm) shunted by ``CP1``::

        Re(Z_p1)(f) = RP1 / (1 + (2*pi*f*RP1*CP1)^2)
        floor(f)    = sqrt(4kT*Re(Z_p1) + 4kT*rout) / (1 + gbw/f)

    RP1 is a *modeling artifact* -- its value is chosen to realize aol from a
    fixed gm -- so this floor is not physical. At ~4 pV/rtHz against a typical
    18 nV/rtHz ``en`` it is 2e-4 of the signal, which is why it is recorded
    rather than engineered away.
    """
    _noise_follower(_NOISE_BASE)
    res = _noise()
    f = 1e4
    got = res.spot(f)

    gm, aol, gbw, rout = 1e-3, 1e6, 10e6, 100.0
    rp1 = aol / gm
    cp1 = gm / (2 * math.pi * gbw)
    re_z = rp1 / (1 + (2 * math.pi * f * rp1 * cp1) ** 2)
    loop = 1.0 + gbw / f
    predicted = math.sqrt(4 * _K_B * _T_REF * (re_z + rout)) / loop

    assert abs(got - predicted) / predicted < 0.05, (got, predicted)
    assert got < 1e-11, got                 # and it is negligible either way


# --- N3/N5: the follower anchor and its scaling ---------------------------- #


@requires_sim
def test_n3_follower_reads_en_and_n5_scales_with_it():
    _noise_follower(_NOISE_BASE)
    floor = _noise().spot(1e4)

    previous = previous_predicted = None
    for en in (18e-9, 36e-9):
        _noise_follower(_NOISE_BASE + f" en={en:g}")
        res = _noise()
        got = res.spot(1e4)
        predicted = math.sqrt(en ** 2 + floor ** 2)
        assert abs(got - predicted) / predicted < 0.05, (en, got, predicted)
        # unity gain, so the input-referred density is the same number
        assert abs(res.spot(1e4, which="input") - got) / got < 0.01
        if previous is not None:
            # the ratio is COMPUTED from the budget, not assumed to be 2
            expected = predicted / previous_predicted
            assert abs(got / previous - expected) < 0.05 * expected
        previous, previous_predicted = got, predicted


# --- N4/N8: the non-inverting budget, with and without inoise -------------- #


@requires_sim
def test_n4_noise_gain_budget_and_n8_inoise_across_the_feedback_network():
    """``e_out = noise_gain * sqrt(en^2 + 4kT*(Rf||Rg) + (inoise*(Rf||Rg))^2)``.

    Three independent contributions, each dominant in a different arm, so a
    wrong sign or a missing term cannot hide behind the others.
    """
    _noise_follower(_NOISE_BASE)
    floor = _noise().spot(1e4)

    en, rf, rg = 18e-9, 9e3, 1e3
    r_par = rf * rg / (rf + rg)
    noise_gain = 1.0 + rf / rg

    measured = {}
    for inoise in (None, 5e-12):
        params = _NOISE_BASE + f" en={en:g}"
        if inoise is not None:
            params += f" inoise={inoise:g}"
        _noninv(params, rf, rg)
        got = _noise().spot(1e4)
        measured[inoise] = got

        referred_sq = en ** 2 + 4 * _K_B * _T_REF * r_par
        if inoise is not None:
            referred_sq += (inoise * r_par) ** 2
        predicted = math.sqrt(referred_sq * noise_gain ** 2 + floor ** 2)
        assert abs(got - predicted) / predicted < 0.05, (inoise, got, predicted)

    # and the current-noise term is big enough here to be a real discriminator,
    # not a difference lost in the gate band
    assert measured[5e-12] / measured[None] > 1.02, measured


# --- N6: where en comes from ----------------------------------------------- #


@requires_sim
def test_n6_en_resolves_from_profile_manufacturer_row_or_explicit_param():
    from skidl.sim import simulate

    def _prov(**fields):
        _setup()
        u = Part(OPAMP_LIB, OPAMP_SYM, ref="U1", Sim_Device="OPAMP", **fields)
        sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
        src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
        sig.connect(src[1], u[3])
        gnd.connect(src[2])
        out.connect(u[1], u[2])
        _rails(u, gnd)
        s = simulate()
        return s.get_netlist(), s.model_provenance["U1"]

    def _rn_line(text):
        return next(l for l in text.splitlines() if l.startswith("RU1_nv "))

    # every profile row carries a datasheet en, and it reaches the deck
    for chip, row in OPAMP_PROFILES.items():
        en = row.get("en")
        assert en is not None, f"{chip} has no datasheet en"
        text, prov = _prov(value=chip)
        assert prov.tier == "datasheet_fit", (chip, prov.tier)
        assert "en=" in prov.name, (chip, prov.name)
        rn = (en ** 2) / (4 * _K_B * _T_REF)
        assert abs(float(_rn_line(text).split()[-1]) - rn) / rn < 1e-9, (chip, text)

    # an explicit Sim.Params en OVERRIDES the profile row
    text, prov = _prov(value="TL072", Sim_Params="en=1n")
    assert prov.tier == "sim_params"
    rn = (1e-9 ** 2) / (4 * _K_B * _T_REF)
    assert abs(float(_rn_line(text).split()[-1]) - rn) / rn < 1e-9, text

    # the Stage-9 ManufacturerModels VNOISE rows are now reachable: AD8605 has
    # no OPAMP_PROFILES row, so it can only resolve through that table
    assert "AD8605" not in OPAMP_PROFILES
    text, prov = _prov(value="AD8605")
    assert "ManufacturerModels" in (prov.name or ""), prov.name
    rn = (8e-9 ** 2) / (4 * _K_B * _T_REF)
    assert abs(float(_rn_line(text).split()[-1]) - rn) / rn < 1e-9, text


@requires_sim
def test_n6b_ldo_vnoise_row_is_not_mistaken_for_an_opamp():
    """``LT1117_ADI`` also carries a ``VNOISE``, but it means output noise as a
    FRACTION of Vout. The ``GBW`` guard must keep it out of the op-amp path --
    otherwise a regulator's 0.3 % would be read as 0.003 V/rtHz."""
    from skidl.sim.converter import SpiceConverter

    assert SpiceConverter._opamp_manufacturer_row("LT1117") == {}
    assert SpiceConverter._opamp_manufacturer_row("TL072")["en"] == 18e-9


# --- N7: the DC solve still converges -------------------------------------- #


@requires_sim
def test_n7_closed_loop_op_still_solves_with_the_generators_emitted():
    """The Stage 9.6 negative -- a discontinuous B-source reading another
    node's voltage kills the closed-loop ``.op`` -- is why the generators are
    a resistor plus a LINEAR controlled source. Prove the loop still solves,
    with the output clamp (itself a min/max B-source) also in play."""
    from skidl.sim import simulate

    _noise_follower(_NOISE_BASE + " en=18n inoise=5p vos=1m")
    try:
        op = simulate().operating_point()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    # vos=1 mV through a follower: the generators must not shift the DC point
    assert abs(op.get_voltage("OUT") - 1e-3) < 1e-5, op.get_voltage("OUT")
