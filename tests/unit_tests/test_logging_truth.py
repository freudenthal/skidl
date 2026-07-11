# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: logging truth & noise (LLC E2E findings R4/R6/R7).

R4 -- the per-phase summary must be self-describing, silent at zero, WARNING
when errors were logged, and must not re-report a previous phase's counts
(bare_* counters used to leak across phases). Sim-only parts (Simulation_SPICE)
with no footprint must not count as errors.

R6 -- the KICADn_SYMBOL_DIR warnings must not fire at import time for backends
that are never used; they re-emit lazily when a backend actually fails to load
a library.

R7 -- ``convert()`` prints at most one INFO line (the tier summary); per-device
build lines are DEBUG.
"""

import logging

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool
from skidl.logger import active_logger

try:
    from skidl.sim import skidl_flat_view
    from skidl.sim.converter import SpiceConverter  # noqa: F401  (needs PySpice)

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


class _Capture(logging.Handler):
    """Handler attached directly to a logger (the 'skidl' logger has
    propagate=False, so pytest's root-logger caplog can miss its records)."""

    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def skidl_log():
    cap = _Capture()
    raw = logging.getLogger("skidl")
    raw.addHandler(cap)
    active_logger.reset_counters()
    yield cap
    raw.removeHandler(cap)
    active_logger.reset_counters()


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


# --- R4: summary truth ------------------------------------------------------


def test_report_summary_silent_at_zero(skidl_log):
    active_logger.report_summary("doing nothing")
    assert skidl_log.records == []


def test_report_summary_warns_on_errors_and_names_itself(skidl_log):
    active_logger.bare_error("synthetic problem")
    skidl_log.records.clear()
    active_logger.report_summary("generating netlist")
    warns = [r for r in skidl_log.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "1 SKiDL log errors" in msg
    assert "independent of any external ERC gate" in msg
    # The summary itself must not bump the counters it reports.
    assert active_logger.warning.count + active_logger.bare_warning.count == 0


def test_reset_counters_clears_bare_counters_too(skidl_log):
    """The R4 double-count: bare_* counts leaked across phases because the
    per-phase resets cleared only error/warning."""
    active_logger.bare_error("phase-1 problem")
    active_logger.bare_warning("phase-1 warning")
    active_logger.reset_counters()
    assert active_logger.error.count == 0
    assert active_logger.bare_error.count == 0
    assert active_logger.warning.count == 0
    assert active_logger.bare_warning.count == 0
    skidl_log.records.clear()
    active_logger.report_summary("generating schematic")
    assert skidl_log.records == []  # nothing re-reported from the prior phase


def test_sim_only_part_missing_footprint_is_not_an_error(tmp_path, skidl_log):
    """A Simulation_SPICE source with no footprint logs a warning, not an error,
    during netlist generation -- the exact 'N errors found' phantom class."""
    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value="5")  # sim-only: no footprint
    r1 = Part("Device", "R", value="1k", footprint="Resistor_SMD:R_0603_1608Metric")
    Net("VIN").connect(v1[1], r1[1])
    Net("0").connect(v1[2], r1[2])
    import builtins

    builtins.default_circuit.generate_netlist(
        tool=KICAD10, file_=str(tmp_path / "t.net")
    )
    assert active_logger.error.count + active_logger.bare_error.count == 0
    warns = [r.getMessage() for r in skidl_log.records if r.levelno == logging.WARNING]
    assert any("sim-only" in m and "VDC" in m for m in warns), warns


# --- R6: lazy symbol-dir warning ---------------------------------------------


def test_default_lib_paths_quiet_when_env_unset(monkeypatch, skidl_log):
    """Import-time path: default_lib_paths with the env var unset must not WARN
    (all five kicad5-9 backends run it on import via config_)."""
    from skidl.tools.kicad9.lib import default_lib_paths

    monkeypatch.delenv("KICAD9_SYMBOL_DIR", raising=False)
    skidl_log.records.clear()
    default_lib_paths()
    assert not [r for r in skidl_log.records if r.levelno >= logging.WARNING]


def test_symbol_dir_warning_fires_on_failed_lib_use(monkeypatch, tmp_path, skidl_log):
    """Lazy path: actually asking the backend for a library it can't find, with
    the env var still unset, warns loudly (and still raises)."""
    _setup()
    from skidl.tools.kicad9.lib import load_sch_lib

    monkeypatch.delenv("KICAD9_SYMBOL_DIR", raising=False)
    skidl_log.records.clear()
    with pytest.raises(FileNotFoundError):
        load_sch_lib(
            lib=None,
            filename="no_such_library_xyz",
            lib_search_paths_=[str(tmp_path)],
        )
    warns = [r.getMessage() for r in skidl_log.records if r.levelno == logging.WARNING]
    assert any("KICAD9_SYMBOL_DIR" in m for m in warns), warns


# --- R7: convert() INFO budget ----------------------------------------------


@requires_sim
def test_convert_prints_at_most_one_info_line():
    """Per-device build lines are DEBUG; convert() emits a single tier-summary
    INFO (so an FSW sweep doesn't wall the log)."""
    _setup()
    cap = _Capture(level=logging.INFO)
    conv_logger = logging.getLogger("skidl.sim.converter")
    conv_logger.addHandler(cap)
    try:
        d1 = Part("Device", "D", value="1N4148", ref="D1")
        d2 = Part("Device", "D", value="SS3H10", ref="D2")
        v1 = Part("Simulation_SPICE", "VDC", value="5")
        r1 = Part("Device", "R", value="1k")
        Net("A").connect(v1[1], d1["A"], d2["A"])
        Net("K").connect(d1["K"], d2["K"], r1[1])
        Net("0").connect(v1[2], r1[2])
        conv = SpiceConverter(skidl_flat_view())
        conv.convert(strict=True)
    finally:
        conv_logger.removeHandler(cap)
    infos = [r for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) <= 1, [r.getMessage() for r in infos]
    assert infos, "the tier-summary INFO must still exist"
    msg = infos[0].getMessage()
    assert "Converted" in msg and "datasheet_fit" in msg, msg
