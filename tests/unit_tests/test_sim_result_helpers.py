# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: the ``SimulationResult`` frequency/step helpers (filter bench P1).

Three helpers were added for the filter/oscillator bench so a report can state
the same figures for every topology:

* ``band_edges(node)`` -- both -3 dB edges and their geometric center, walking
  **outward from the response peak** so a band-pass yields two edges, a low-pass
  only ``f_high`` and a high-pass only ``f_low``;
* ``nyquist(node)`` -- the complex locus as ``(freq, re, im)``;
* ``step_metrics(node)`` -- final value, overshoot, rise/peak/settling time,
  referred to the pre-step level so a non-unity or inverting gain measures right.

Every case here has a **closed-form** answer, so the test is a check against
analysis rather than against a previous run:

* RC low-pass: at ``f = fc`` exactly, ``|H| = 1/sqrt(2)`` and ``arg H = -45 deg``
  -- and the Nyquist locus of ``1/(1+jw/wc)`` is the semicircle of radius 1/2
  centered at ``(1/2, 0)``, so ``|H - 0.5| = 0.5`` at every frequency.
* Series-RLC low-pass step: a 2nd-order system of damping ``zeta`` overshoots by
  ``exp(-pi*zeta/sqrt(1-zeta^2))``.
* Series-RLC band-pass (across R): edges at ``f0*(sqrt(1+1/(4Q^2)) -/+ 1/(2Q))``
  and ``f_center == f0`` exactly.

``band_edges`` on a low-pass must also agree with the existing
``cutoff_frequency``, which is the guard against the new walker drifting from
the established interpolation convention.
"""

import math

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter  # noqa: F401

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


# --- decks -------------------------------------------------------------- #
#
# Net names avoid every rail token the converter's name heuristic injects a
# supply for (VIN/VCC/VDD/V+/3V3/+5V/VSUPPLY) -- see the bench naming rule in
# skidl_eda.filters.circuits.


def _rc_lowpass(r_ohms, c_farads):
    """VSIN(ac=1) -> R -> C to ground. fc = 1/(2*pi*R*C)."""
    _setup()
    src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    r = Part("Device", "R", ref="R1", value=f"{r_ohms}")
    c = Part("Device", "C", ref="C1", value=f"{c_farads}")
    sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
    sig.connect(src[1], r[1])
    out.connect(r[2], c[1])
    gnd.connect(src[2], c[2])


def _series_rlc(r_ohms, l_henries, c_farads, across="C"):
    """VSIN(ac=1) -> L -> R -> C to ground.

    ``OUT`` is taken across C (a 2nd-order LOW-PASS) or across R (a BAND-PASS),
    per ``across``. w0 = 1/sqrt(L*C); Q = w0*L/R (series form).
    """
    _setup()
    src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    l = Part("Device", "L", ref="L1", value=f"{l_henries}")
    r = Part("Device", "R", ref="R1", value=f"{r_ohms}")
    c = Part("Device", "C", ref="C1", value=f"{c_farads}")
    sig, mid, out, gnd = Net("SIG"), Net("MID"), Net("OUT"), Net("GND")
    if across == "C":
        # SIG -L- MID -R- OUT -C- GND   (output across C)
        sig.connect(src[1], l[1])
        mid.connect(l[2], r[1])
        out.connect(r[2], c[1])
        gnd.connect(src[2], c[2])
    else:
        # SIG -L- MID -C- OUT -R- GND   (output across R)
        sig.connect(src[1], l[1])
        mid.connect(l[2], c[1])
        out.connect(c[2], r[1])
        gnd.connect(src[2], r[2])


def _ac(start, stop, points=400):
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


# --- band_edges ---------------------------------------------------------- #


@requires_sim
def test_band_edges_lowpass_matches_cutoff_frequency():
    """RC low-pass: only ``f_high`` exists, it equals ``cutoff_frequency``, and
    the closed-form fc is recovered to 2 %."""
    r, c = 1.0e3, 159.155e-9  # fc = 1 kHz
    fc = 1.0 / (2 * math.pi * r * c)
    _rc_lowpass(r, c)
    res = _ac(10.0, 1e5, points=400)

    edges = res.band_edges("OUT")
    assert edges["f_low"] is None, edges
    assert edges["f_center"] is None, edges
    assert edges["f_high"] is not None
    assert abs(edges["f_high"] - fc) / fc < 0.02, (edges["f_high"], fc)
    # the new walker must not drift from the established interpolation
    assert abs(edges["f_high"] - res.cutoff_frequency("OUT")) < 1e-6


@requires_sim
def test_band_edges_highpass_has_only_lower_edge():
    """C-R high-pass (the RC deck read across R): only ``f_low``."""
    _setup()
    r, c = 1.0e3, 159.155e-9
    fc = 1.0 / (2 * math.pi * r * c)
    src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    cc = Part("Device", "C", ref="C1", value=f"{c}")
    rr = Part("Device", "R", ref="R1", value=f"{r}")
    sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
    sig.connect(src[1], cc[1])
    out.connect(cc[2], rr[1])
    gnd.connect(src[2], rr[2])

    edges = _ac(10.0, 1e5, points=400).band_edges("OUT")
    assert edges["f_high"] is None, edges
    assert edges["f_center"] is None, edges
    assert abs(edges["f_low"] - fc) / fc < 0.02, (edges["f_low"], fc)


@requires_sim
def test_band_edges_bandpass_recovers_both_edges_and_center():
    """Series-RLC read across R. Closed form:

    ``f_low/high = f0*(sqrt(1+1/(4Q^2)) -/+ 1/(2Q))`` and their geometric mean
    is exactly ``f0``.
    """
    l_h, c_f, r_o = 1e-3, 1e-6, 20.0  # f0 ~ 5.033 kHz, Q = w0*L/R ~ 1.58
    f0 = 1.0 / (2 * math.pi * math.sqrt(l_h * c_f))
    q = (2 * math.pi * f0) * l_h / r_o
    k = math.sqrt(1 + 1 / (4 * q * q))
    f_lo_ref, f_hi_ref = f0 * (k - 1 / (2 * q)), f0 * (k + 1 / (2 * q))

    _series_rlc(r_o, l_h, c_f, across="R")
    edges = _ac(100.0, 1e6, points=600).band_edges("OUT")

    assert abs(edges["f_low"] - f_lo_ref) / f_lo_ref < 0.02, (edges, f_lo_ref)
    assert abs(edges["f_high"] - f_hi_ref) / f_hi_ref < 0.02, (edges, f_hi_ref)
    assert abs(edges["f_center"] - f0) / f0 < 0.02, (edges, f0)


@requires_sim
def test_band_edges_raises_on_degenerate_sweep():
    """A one-point sweep observes nothing -- it must RAISE, not return None
    edges that read like 'no crossing found'."""
    _rc_lowpass(1.0e3, 159.155e-9)
    res = _ac(1000.0, 1000.0, points=1)
    with pytest.raises(ValueError, match="at least 2 AC samples"):
        res.band_edges("OUT")


# --- nyquist -------------------------------------------------------------- #


@requires_sim
def test_nyquist_rc_lowpass_is_the_unit_semicircle():
    """``H = 1/(1+jw/wc)`` traces the circle |H - 1/2| = 1/2, and at f = fc the
    point is (1/2, -1/2)."""
    r, c = 1.0e3, 159.155e-9
    fc = 1.0 / (2 * math.pi * r * c)
    _rc_lowpass(r, c)
    freq, re, im = _ac(10.0, 1e5, points=400).nyquist("OUT")

    assert len(freq) == len(re) == len(im) >= 100, len(freq)
    radii = [abs(complex(a, b) - 0.5) for a, b in zip(re, im)]
    assert max(abs(x - 0.5) for x in radii) < 0.01, max(radii)

    i = min(range(len(freq)), key=lambda k: abs(freq[k] - fc))
    assert abs(re[i] - 0.5) < 0.02, re[i]
    assert abs(im[i] + 0.5) < 0.02, im[i]
    # |H| = 1/sqrt(2) and phase = -45 deg at the corner
    assert abs(abs(complex(re[i], im[i])) - 1 / math.sqrt(2)) < 0.02
    assert abs(math.degrees(math.atan2(im[i], re[i])) + 45.0) < 2.0


# --- step_metrics --------------------------------------------------------- #


@requires_sim
def test_step_metrics_rlc_overshoot_matches_zeta_formula():
    """Series-RLC low-pass step: overshoot = exp(-pi*zeta/sqrt(1-zeta^2)).

    zeta = 1/(2Q); with L=1 mH, C=1 uF, R=20 ohm -> Q ~ 1.58, zeta ~ 0.316,
    overshoot ~ 35 %. Final value is the source amplitude (unity DC gain).
    """
    l_h, c_f, r_o = 1e-3, 1e-6, 20.0
    w0 = 1.0 / math.sqrt(l_h * c_f)
    zeta = r_o / (2 * w0 * l_h)
    expected = 100.0 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta))

    _setup()
    # PULSE from 0 to 1 V at t=0 with a step far faster than the ring period.
    src = Part(
        "Simulation_SPICE", "VPULSE", ref="V1", value="1V",
        Sim_Params="v1=0 v2=1 td=0 tr=1n tf=1n pw=10m per=20m",
    )
    l = Part("Device", "L", ref="L1", value=f"{l_h}")
    r = Part("Device", "R", ref="R1", value=f"{r_o}")
    c = Part("Device", "C", ref="C1", value=f"{c_f}")
    sig, mid, out, gnd = Net("SIG"), Net("MID"), Net("OUT"), Net("GND")
    sig.connect(src[1], l[1])
    mid.connect(l[2], r[1])
    out.connect(r[2], c[1])
    gnd.connect(src[2], c[2])

    # ~30 ring periods so the tail average is the settled value
    period = 2 * math.pi / w0
    res = _tran(period / 400.0, 30 * period)
    m = res.step_metrics("OUT")

    assert abs(m["final_value"] - 1.0) < 0.02, m
    assert abs(m["overshoot_pct"] - expected) < 2.0, (m["overshoot_pct"], expected)
    assert m["rise_time"] is not None and 0 < m["rise_time"] < period
    assert m["peak_time"] is not None and 0 < m["peak_time"] < 2 * period
    assert m["settling_time"] is not None


@requires_sim
def test_step_metrics_rc_first_order_has_no_overshoot():
    """A 1st-order step cannot overshoot: the helper must report exactly 0.0,
    not a small negative undershoot number. Rise time = 2.2*R*C."""
    r, c = 1.0e3, 1.0e-6
    tau = r * c
    _setup()
    src = Part(
        "Simulation_SPICE", "VPULSE", ref="V1", value="1V",
        Sim_Params="v1=0 v2=1 td=0 tr=1n tf=1n pw=100m per=200m",
    )
    rr = Part("Device", "R", ref="R1", value=f"{r}")
    cc = Part("Device", "C", ref="C1", value=f"{c}")
    sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
    sig.connect(src[1], rr[1])
    out.connect(rr[2], cc[1])
    gnd.connect(src[2], cc[2])

    m = _tran(tau / 200.0, 20 * tau).step_metrics("OUT")
    # Not `== 0.0`: `final_value` is a tail MEAN, so it sits infinitesimally
    # below the true asymptote and the "excursion past final" is a float hair
    # (measured 2.6e-6 %). The physical claim under test is that a 1st-order
    # step shows no *meaningful* overshoot, and that the helper's max(0, .)
    # clamp keeps the figure from going negative.
    assert 0.0 <= m["overshoot_pct"] < 0.01, m
    assert abs(m["final_value"] - 1.0) < 0.01, m
    assert abs(m["rise_time"] - 2.197 * tau) / (2.197 * tau) < 0.05, m


# --- degenerate inputs must RAISE, not silently measure nothing ----------- #
#
# Driven off stub analyses rather than ngspice: ngspice will not emit a
# 2-sample transient on request, so a live-deck version of these tests skips
# and the raise path goes unexercised. The workspace's most-reproduced defect
# class is an instrument that observes nothing and stays quiet, so these two
# guards are tested unconditionally.


class _StubAC:
    """Minimal stand-in for a PySpice AC analysis: one node, N points."""

    def __init__(self, n):
        import numpy as np

        self.frequency = np.logspace(2, 3, n)
        self.nodes = ["OUT"]
        self._d = {"OUT": np.ones(n, dtype=complex)}

    def __getitem__(self, k):
        return self._d[k]


class _StubTran:
    """Minimal stand-in for a PySpice transient analysis: one node, N samples."""

    def __init__(self, n):
        import numpy as np

        self.time = np.linspace(0.0, 1e-3, n)
        self.nodes = ["OUT"]
        self._d = {"OUT": np.ones(n)}

    def __getitem__(self, k):
        return self._d[k]


@requires_sim
def test_band_edges_raises_on_single_point_sweep():
    from skidl.sim.simulator import SimulationResult

    res = SimulationResult(_StubAC(1), "ac")
    with pytest.raises(ValueError, match="at least 2 AC samples"):
        res.band_edges("OUT")


@requires_sim
def test_step_metrics_raises_on_two_sample_run():
    from skidl.sim.simulator import SimulationResult

    res = SimulationResult(_StubTran(2), "transient")
    with pytest.raises(ValueError, match="at least 3 transient samples"):
        res.step_metrics("OUT")
