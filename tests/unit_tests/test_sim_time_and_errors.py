# -*- coding: utf-8 -*-
"""SI-suffix transient times (B3) and surfaced ngspice failure reasons (B1).

The SI-parse and error-surfacing helpers are pure (no PySpice); one live test
confirms an SI-string transient actually runs on the vendored ngspice stack.
"""

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter  # noqa: F401  (needs PySpice)

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(not HAS_SIM, reason="PySpice not installed")

from skidl.sim.simulator import (  # noqa: E402
    _augment_ngspice_error,
    _parse_si_time,
    _salvage_completed_transient,
    _to_seconds,
    _uic_collapse_hint,
)


# --- B3: SI-suffix time parsing (pure) ------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (5e-6, 5e-6), (2.5, 2.5), ("2.5", 2.5), ("5u", 5e-6), ("5us", 5e-6),
        ("10m", 10e-3), ("10ms", 10e-3), ("1n", 1e-9), ("1ns", 1e-9),
        ("1k", 1e3), ("1e-6", 1e-6), ("0.5m", 0.5e-3),
    ],
)
def test_parse_si_time(raw, expected):
    assert _parse_si_time(raw) == pytest.approx(expected)


def test_parse_si_time_garbage_is_none():
    assert _parse_si_time("u") is None
    assert _parse_si_time("garbage") is None
    assert _parse_si_time("") is None


def test_to_seconds_raises_on_garbage():
    with pytest.raises(ValueError):
        _to_seconds("u", "step_time")


# --- B1: ngspice failure reason surfaced (pure, via a fake shared) ----------


class _FakeShared:
    def __init__(self, stdout="", stderr=""):
        self.stdout = stdout
        self.stderr = stderr


class _FakeSim:
    def __init__(self, shared):
        self.ngspice = shared


def test_augment_appends_ngspice_tail():
    exc = RuntimeError("Command 'run' failed")
    shared = _FakeShared(
        stdout="doing transient analysis\nTimestep too small; "
        "trouble with xq1:dmos-instance\nrun aborted\n")
    aug = _augment_ngspice_error(_FakeSim(shared), exc)
    assert aug is not exc
    assert isinstance(aug, RuntimeError)  # type preserved for existing except clauses
    assert "Timestep too small" in str(aug)
    assert "dmos-instance" in str(aug)


def test_augment_no_output_returns_same():
    exc = RuntimeError("Command 'run' failed")
    aug = _augment_ngspice_error(_FakeSim(_FakeShared()), exc)
    assert aug is exc


# --- S4: driven-subckt UIC-collapse HINT (pure) ----------------------------

_COLLAPSE = ('doing transient analysis\nTimestep too small; time=2.0e-10, '
             'timestep=1.25e-19: trouble with node "xu1.md1_5"\nrun aborted\n')


def test_uic_collapse_hint_present_when_uic():
    hint = _uic_collapse_hint(_COLLAPSE, uic=True)
    assert "HINT" in hint
    assert "xu1" not in hint and "'u1'" in hint  # ref surfaced without the x prefix
    assert "without use_initial_condition" in hint.lower()


def test_uic_collapse_hint_absent_when_uic_off():
    assert _uic_collapse_hint(_COLLAPSE, uic=False) == ""


def test_uic_collapse_hint_absent_for_external_node():
    # An *external* node ("sw", no x<ref>. prefix) is not the subckt-internal
    # collapse signature -> no hint.
    txt = ('Timestep too small; time=2.0e-10: trouble with node "sw"\n'
           'run aborted\n')
    assert _uic_collapse_hint(txt, uic=True) == ""


def test_uic_collapse_hint_absent_when_time_large():
    # A late-time timestep failure is a real switching-transient issue, not a
    # t~=0 UIC-initialization collapse.
    txt = ('Timestep too small; time=3.0e-3: trouble with node "xu1.md1_5"\n'
           'run aborted\n')
    assert _uic_collapse_hint(txt, uic=True) == ""


def test_augment_includes_uic_hint():
    exc = RuntimeError("Command 'run' failed")
    aug = _augment_ngspice_error(_FakeSim(_FakeShared(stdout=_COLLAPSE)), exc,
                                 uic=True)
    assert "HINT" in str(aug) and "'u1'" in str(aug)
    # without uic, only the tail is appended, no hint
    aug2 = _augment_ngspice_error(_FakeSim(_FakeShared(stdout=_COLLAPSE)), exc,
                                  uic=False)
    assert "HINT" not in str(aug2) and "Timestep too small" in str(aug2)


# --- F1: salvage a completed transient past a benign end-of-run status ------
#
# Pure tests: a fake shared ngspice whose last plot rebuilds to a
# TransientAnalysis-like object with a .time axis. No ngspice needed.


class _FakeAnalysis:
    def __init__(self, time):
        self.time = time


class _FakePlot:
    def __init__(self, analysis):
        self._analysis = analysis

    def to_analysis(self):
        return self._analysis


class _FakeSharedPlot:
    def __init__(self, last_plot, analysis):
        self.last_plot = last_plot
        self._plot = _FakePlot(analysis)

    def plot(self, simulation, plot_name):
        return self._plot


class _FakeSimPlot:
    def __init__(self, shared):
        self.ngspice = shared


def _fake_sim(last_plot, time):
    import numpy as np

    an = _FakeAnalysis(None if time is None else np.asarray(time, dtype=float))
    return _FakeSimPlot(_FakeSharedPlot(last_plot, an))


def test_salvage_returns_completed_transient():
    import numpy as np

    t = np.linspace(0.0, 24e-6, 480049)  # full run to the requested 24 us
    res = _salvage_completed_transient(_fake_sim("tran1", t), 24e-6)
    assert res is not None
    analysis, note = res
    assert analysis.time is not None and analysis.time.size == 480049
    assert "salvaged" in note and "480049" in note


def test_salvage_rejects_short_rowcount():
    import numpy as np

    # A genuine early collapse leaves far fewer rows than a completed run.
    t = np.linspace(0.0, 24e-6, 50)
    assert _salvage_completed_transient(_fake_sim("tran1", t), 24e-6) is None


def test_salvage_rejects_run_that_stopped_short():
    import numpy as np

    # Many rows, but the time axis only reached 5 us of a 24 us request -> the
    # transient did NOT complete; this must re-raise, not be salvaged.
    t = np.linspace(0.0, 5e-6, 480049)
    assert _salvage_completed_transient(_fake_sim("tran1", t), 24e-6) is None


def test_salvage_none_without_last_plot():
    import numpy as np

    t = np.linspace(0.0, 24e-6, 480049)
    assert _salvage_completed_transient(_fake_sim(None, t), 24e-6) is None


def test_salvage_none_for_non_transient_plot():
    # An op/dc/ac last plot rebuilds without a .time axis -> not salvageable here.
    assert _salvage_completed_transient(_fake_sim("op1", None), 24e-6) is None


@requires_sim
def test_run_analysis_salvages_on_benign_command_error():
    """_run_analysis returns the salvaged transient (and tags a warning) when the
    thunk raises NgSpiceCommandError but the last plot is a completed run."""
    import numpy as np

    from skidl.sim.simulator import CircuitSimulator
    from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError

    t = np.linspace(0.0, 24e-6, 200_000)
    fake_simulator = _fake_sim("tran1", t)

    cs = CircuitSimulator.__new__(CircuitSimulator)  # skip real __init__/convert
    cs.model_provenance = {}

    def thunk():
        raise NgSpiceCommandError("Command 'run' failed")

    analysis = cs._run_analysis(fake_simulator, thunk, salvage_end_time=24e-6)
    assert getattr(analysis, "_skidl_salvage_warning", None)
    assert analysis.time.size == 200_000


@requires_sim
def test_run_analysis_reraises_when_not_salvageable():
    """A short/absent last plot is a real failure -> _run_analysis re-raises."""
    import numpy as np

    from skidl.sim.simulator import CircuitSimulator
    from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError

    t = np.linspace(0.0, 1e-6, 10)  # collapsed early
    fake_simulator = _fake_sim("tran1", t)

    cs = CircuitSimulator.__new__(CircuitSimulator)
    cs.model_provenance = {}

    def thunk():
        raise NgSpiceCommandError("Command 'run' failed")

    with pytest.raises(NgSpiceCommandError):
        cs._run_analysis(fake_simulator, thunk, salvage_end_time=24e-6)


# --- B3 live: an SI-string transient actually runs -------------------------


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


@requires_sim
def test_transient_accepts_si_string_times():
    """A trivial RC transient driven with SI-suffix strings runs identically to
    the float form (B3 -- the one API that used to reject '5u')."""
    import numpy as np

    def run(step, end):
        _setup()
        v = Part("Simulation_SPICE", "VDC", value="5", ref="V1")
        r = Part("Device", "R", value="1k", ref="R1")
        c = Part("Device", "C", value="1u", ref="C1")
        vin, out, gnd = Net("VIN"), Net("OUT"), Net("0")
        vin.connect(v[1], r[1])
        out.connect(r[2], c[1])
        gnd.connect(v[2], c[2])
        from skidl.sim import simulate

        sim = simulate()
        an = sim.transient_analysis(step_time=step, end_time=end)
        return float(np.array(an.get_voltage("OUT"))[-1])

    v_str = run("50u", "5m")
    v_flt = run(50e-6, 5e-3)
    assert v_str == pytest.approx(v_flt, rel=0.05)
    assert v_flt > 4.0  # RC (tau=1ms) is well charged by 5 ms
