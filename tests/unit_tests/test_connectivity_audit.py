# -*- coding: utf-8 -*-

"""Unit tests for the stage-19 emission connectivity audit.

``_audit_sheet_connectivity`` is the render-space tripwire that logs a WARNING
when two nets end up sharing a coordinate (the fusion mechanism Blocker B fixed).
These tests drive it directly with synthetic elements so they need no KiCad
libraries: one asserts it fires on a deliberate cross-net cell, the other that a
same-net coincidence stays silent.
"""

from simp_sexp import Sexp

import skidl.logger as _logger
from skidl.tools.kicad9 import sexp_schematic as _ksch


class _FakeNode:
    parts = []
    sheet_filename = "fake_sheet.kicad_sch"


def _label(name, x, y):
    return Sexp(["global_label", name, ["at", float(x), float(y), 0]])


def _capture_warnings(monkeypatch):
    calls = []

    def _rec(msg, *args, **kwargs):
        try:
            calls.append(msg % args if args else msg)
        except Exception:  # noqa: BLE001
            calls.append(str(msg))

    monkeypatch.setattr(_logger.active_logger, "warning", _rec)
    return calls


def test_audit_warns_on_cross_net_cell(monkeypatch):
    calls = _capture_warnings(monkeypatch)
    elements = [_label("SN", 100.0, 100.0), _label("GND", 100.0, 100.0)]
    _ksch._audit_sheet_connectivity(_FakeNode(), elements, backend=None, sheet_tx=None)
    assert any("connectivity audit" in c for c in calls), calls
    assert any("SN" in c and "GND" in c for c in calls), calls


def test_audit_silent_on_same_net(monkeypatch):
    calls = _capture_warnings(monkeypatch)
    elements = [_label("SN", 100.0, 100.0), _label("SN", 100.0, 100.0)]
    _ksch._audit_sheet_connectivity(_FakeNode(), elements, backend=None, sheet_tx=None)
    assert not calls, calls


def test_audit_respects_disable_flag(monkeypatch):
    calls = _capture_warnings(monkeypatch)
    monkeypatch.setattr(_ksch, "_EMIT_CONNECTIVITY_AUDIT", False)
    elements = [_label("SN", 100.0, 100.0), _label("GND", 100.0, 100.0)]
    _ksch._audit_sheet_connectivity(_FakeNode(), elements, backend=None, sheet_tx=None)
    assert not calls, calls
