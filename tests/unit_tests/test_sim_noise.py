# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: ngspice ``.noise`` through ``skidl.sim`` (Stage 9.7).

``CircuitSimulator.noise_analysis`` exposes ngspice's small-signal noise
analysis and returns a :class:`~skidl.sim.simulator.NoiseResult`. The shape
differs from every other analysis -- one run produces **two** ngspice plots
(``noise1``: the spectra vs frequency; ``noise2``: the band-integrated totals)
and PySpice's own result path reads only the last of them.

Every measurement below is graded against a **closed form** that shares no code
with this repo:

===  ===========================================================
N1   a bare 1 kohm reads ``sqrt(4kTR)`` -- and that also pins the
     spectrum's UNITS as V/sqrt(Hz) rather than V^2/Hz
N2   thermal noise follows the RUN temperature, not TNOM
N3   an RC low-pass integrates to ``sqrt(kT/C)``, independent of R
N4   ``NoiseResult.integrated`` reproduces ngspice's own total
N5   ``spot()`` interpolates in log-log on a known 1/f-shaped grid,
     and refuses to extrapolate
N6   a bad source name raises and names the sources it did find
N7   a missing plot raises (the denominator gate)
===  ===========================================================
"""

import math

import pytest

from skidl import KICAD10, Circuit, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.simulator import NoiseResult  # noqa: F401

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)

K_B = 1.380649e-23


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _r(ref, value):
    return Part("Device", "R", ref=ref, value=f"{value:.10g}",
                footprint="Resistor_SMD:R_0603_1608Metric")


def _series_rc(r_ohm=1000.0, c_farad=None, r_load=1e9):
    """``SIG -R1- OUT``, OUT shunted by ``c_farad`` or by ``r_load``.

    With an ideal source at SIG, the output sees ``R1`` to ground, so the
    output noise is that resistor's -- shunted by ``r_load``, which is chosen
    six decades larger so its own contribution is below every gate band.
    """
    _setup()
    sig, out, gnd = Net("SIG"), Net("OUT"), Net("GND")
    src = Part("Simulation_SPICE", "VSIN", ref="V1", value="1V")
    r1 = _r("R1", r_ohm)
    sig.connect(src[1], r1[1])
    gnd.connect(src[2])
    out.connect(r1[2])
    if c_farad is None:
        rl = _r("R2", r_load)
        out.connect(rl[1])
        gnd.connect(rl[2])
    else:
        c1 = Part("Device", "C", ref="C1", value=f"{c_farad:.10g}",
                  footprint="Capacitor_SMD:C_0603_1608Metric")
        out.connect(c1[1])
        gnd.connect(c1[2])


def _noise(output="OUT", source="V1", start=1e3, stop=1e6, points=10, **kw):
    from skidl.sim import simulate

    try:
        return simulate().noise_analysis(
            output, source, start, stop, points=points, **kw
        )
    except (RuntimeError, OSError, ImportError) as e:  # noqa: BLE001
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")


# --- N1: the 4kTR anchor, which also pins the units ------------------------ #


@requires_sim
def test_n1_bare_resistor_matches_4ktr_in_volts_per_root_hz():
    """A 1 kohm reads ``sqrt(4kTR)``, flat across the band.

    This is also the UNITS gate. ngspice documents ``onoise_spectrum`` as a
    density, but "documented" is not "measured": at 25 C a 1 kohm reads
    4.058e-9 if the vector is V/sqrt(Hz) and 1.647e-17 if it is V^2/Hz. The
    accessor is built on the measured convention.
    """
    _series_rc(1000.0)
    res = _noise()
    expected = math.sqrt(4 * K_B * 298.15 * 1000.0)

    assert len(res) == 31, len(res)          # dec 10 over 3 decades, inclusive
    assert res.onoise_spectrum.min() > 0
    for value in (res.onoise_spectrum[0], res.onoise_spectrum[-1]):
        assert abs(value - expected) / expected < 0.02, (value, expected)
    # flat: a thermal spectrum has no frequency dependence
    spread = res.onoise_spectrum.max() / res.onoise_spectrum.min() - 1.0
    assert spread < 1e-6, spread
    # unity gain source->output (R2 >> R1), so the referred noise matches
    assert abs(res.inoise_spectrum[0] - res.onoise_spectrum[0]) / expected < 0.02


@requires_sim
def test_n1b_noise_scales_as_sqrt_r():
    """Doubling R multiplies the density by sqrt(2) -- the 4kTR shape itself."""
    _series_rc(1000.0)
    low = _noise().onoise_spectrum[0]
    _series_rc(4000.0)
    high = _noise().onoise_spectrum[0]
    assert abs(high / low - 2.0) < 0.02, (low, high)


# --- N2: the run temperature, not TNOM ------------------------------------- #


@requires_sim
def test_n2_thermal_noise_follows_run_temperature():
    """``4*k*T_run*R``. Measured, because the alternative (TNOM sets it) is a
    plausible reading of the ngspice manual and would put a 15 % error into
    every ``en`` value computed at emission time."""
    for celsius in (25.0, 125.0):
        _series_rc(1000.0)
        res = _noise(temperature=celsius)
        expected = math.sqrt(4 * K_B * (273.15 + celsius) * 1000.0)
        got = res.onoise_spectrum[0]
        assert abs(got - expected) / expected < 0.01, (celsius, got, expected)
        assert res.temperature == celsius


# --- N3/N4: the kT/C anchor and the integrator ----------------------------- #


@requires_sim
def test_n3_rc_lowpass_integrates_to_sqrt_kt_over_c():
    """Total output noise of an RC low-pass is ``sqrt(kT/C)`` -- independent of
    R, which is the point of the anchor: the resistor sets both the density and
    the bandwidth, and R cancels.

    The band runs to >= 630*fc, which captures (2/pi)*atan(630) = 99.9 % of the
    Lorentzian's power.
    """
    c = 1e-9
    expected = math.sqrt(K_B * 298.15 / c)
    for r_ohm in (1000.0, 4000.0):
        fc = 1.0 / (2 * math.pi * r_ohm * c)
        _series_rc(r_ohm, c_farad=c)
        res = _noise(start=10.0, stop=max(200e6, 700 * fc), points=100)
        assert abs(res.onoise_total_v - expected) / expected < 0.02, (
            r_ohm, res.onoise_total_v, expected)


@requires_sim
def test_n4_integrated_matches_the_closed_form():
    """``NoiseResult.integrated`` over a FLAT spectrum == ``density*sqrt(BW)``.

    The input-referred spectrum of the RC bench is flat (``onoise/|H|`` puts
    the resistor's density straight back at the source), so its band integral
    has a closed form with no simulator in it.
    """
    _series_rc(1000.0, c_farad=1e-9)
    res = _noise(start=10.0, stop=200e6, points=100)

    density = math.sqrt(4 * K_B * 298.15 * 1000.0)
    assert res.inoise_spectrum.max() / res.inoise_spectrum.min() - 1 < 1e-6
    bandwidth = float(res.frequency[-1] - res.frequency[0])
    expected = density * math.sqrt(bandwidth)
    ours = res.integrated(res.frequency[0], res.frequency[-1], which="input")
    assert abs(ours - expected) / expected < 1e-4, (ours, expected)

    # the output side, whose energy sits far from the top of the grid, also
    # matches ngspice's own total
    ours_out = res.integrated(res.frequency[0], res.frequency[-1])
    assert abs(ours_out - res.onoise_total_v) / res.onoise_total_v < 1e-3

    with pytest.raises(ValueError):
        res.integrated(1.0, 1e3)              # below the swept band
    with pytest.raises(ValueError):
        res.integrated(1e3, 1e3)              # empty band


@requires_sim
def test_n4b_ngspice_inoise_total_is_grid_dependent():
    """``inoise_total`` is not a substitute for :meth:`integrated`.

    Measured 2026-08-09 across 24 (deck, band, points-per-decade) combinations.
    Two facts, both gated here:

    * on a deck whose OUTPUT spectrum is flat (a bare resistor into a very
      light load) ngspice's total equals the trapezoid to 1e-6 at 10, 20, 100
      and 400 points/decade -- so there is no constant offset to correct for;
    * on an RC low-pass -- flat input-referred spectrum, rolled-off output --
      it reads HIGH, and the excess shrinks ~4x when the grid is refined 4x,
      i.e. it is a convergence error in ngspice's own integrator.

    A quantity that moves with the sweep grid is not one to grade a datasheet
    number against, which is why ``integrated()`` exists.
    """
    # flat output spectrum: no disagreement at any grid density
    for points in (10, 400):
        _series_rc(1000.0)
        res = _noise(start=1e3, stop=1e6, points=points)
        ours = res.integrated(res.frequency[0], res.frequency[-1], which="input")
        assert abs(res.inoise_total_v - ours) / ours < 1e-5, (points, ours)

    # rolled-off output spectrum: high, and convergent
    excess = {}
    for points in (100, 400):
        _series_rc(1000.0, c_farad=1e-9)
        res = _noise(start=10.0, stop=200e6, points=points)
        ours = res.integrated(res.frequency[0], res.frequency[-1], which="input")
        excess[points] = res.inoise_total_v / ours - 1.0
        assert excess[points] > 0, (points, excess)
    assert excess[400] < excess[100] / 3.0, excess


# --- N5: spot() ------------------------------------------------------------ #


@requires_sim
def test_n5_spot_interpolates_in_log_log_and_refuses_to_extrapolate():
    """On a sloped spectrum, the log-log interpolation must beat a linear one.

    An RC low-pass past its corner rolls off as 1/f, so the true value halfway
    (in log f) between two decade-spaced grid points is the geometric mean,
    not the arithmetic one -- a linear interpolation is ~6 % high there.
    """
    _series_rc(1000.0, c_farad=1e-9)
    res = _noise(start=1e6, stop=1e8, points=1)   # decade grid: 1e6, 1e7, 1e8
    assert len(res) == 3, res.frequency

    f_lo, f_hi = float(res.frequency[1]), float(res.frequency[2])
    s_lo, s_hi = float(res.onoise_spectrum[1]), float(res.onoise_spectrum[2])
    f_mid = math.sqrt(f_lo * f_hi)

    geometric = math.sqrt(s_lo * s_hi)
    arithmetic = 0.5 * (s_lo + s_hi)
    got = res.spot(f_mid)
    assert abs(got - geometric) / geometric < 1e-9, (got, geometric)
    assert abs(got - arithmetic) / arithmetic > 0.05, (got, arithmetic)

    # exact on a grid point, and both spectra are reachable
    assert abs(res.spot(f_lo) - s_lo) / s_lo < 1e-9
    assert res.spot(f_lo, which="input") > 0
    with pytest.raises(ValueError):
        res.spot(f_lo, which="sideways")

    # extrapolation is a fabricated number, not a measurement
    with pytest.raises(ValueError):
        res.spot(1e3)
    with pytest.raises(ValueError):
        res.spot(1e9)


# --- N6/N7: the failure paths ---------------------------------------------- #


@requires_sim
def test_n6_source_name_accepts_ref_or_element_and_raises_on_neither():
    """A part ref ``V1`` is emitted as element ``VV1``; both must resolve, and
    an unknown name must name the sources it did find."""
    from skidl.sim import simulate

    _series_rc(1000.0)
    sim = simulate()
    sim._convert_to_spice()
    assert sim._resolve_noise_source("V1") == "VV1"
    assert sim._resolve_noise_source("VV1") == "VV1"
    assert sim._resolve_noise_source("v1") == "VV1"
    with pytest.raises(ValueError) as exc:
        sim._resolve_noise_source("V9")
    assert "VV1" in str(exc.value)


@requires_sim
def test_n7_missing_noise_plot_raises():
    """A run that leaves no spectrum plot must RAISE, not return a half result.

    An analysis whose spectra were silently dropped looks exactly like one that
    measured a flat zero -- the defect class this workspace has hit seven times.
    """
    from skidl.sim.simulator import CircuitSimulator

    class _Plot(dict):
        plot_name = "noise2"

    class _Shared:
        plot_names = ["noise2", "const"]

        def plot(self, simulation, name):
            p = _Plot()
            p["onoise_total"] = object()
            return p

    class _Sim:
        ngspice = _Shared()

    with pytest.raises(RuntimeError) as exc:
        CircuitSimulator._fetch_noise_plots(_Sim())
    assert "spectra" in str(exc.value)

    class _NoShared:
        pass

    with pytest.raises(RuntimeError) as exc:
        CircuitSimulator._fetch_noise_plots(_NoShared())
    assert "ngspice-shared" in str(exc.value)


@requires_sim
def test_n7b_empty_frequency_axis_raises():
    """The denominator gate on the result object itself."""
    from skidl.sim.simulator import NoiseResult

    with pytest.raises(ValueError) as exc:
        NoiseResult([], [], [], 0.0, 0.0, output="OUT", source="VV1",
                    temperature=25)
    assert "EMPTY" in str(exc.value)

    with pytest.raises(ValueError):
        NoiseResult([1.0, 2.0], [1e-9], [1e-9], 0.0, 0.0, output="OUT",
                    source="VV1", temperature=25)
