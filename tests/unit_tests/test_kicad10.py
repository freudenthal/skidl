# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests for the KiCad 10 backend (``skidl.tools.kicad10``).

Covers the three regressions called out in the KiCad-10 port plan:

1. Two-digit tool-version derivation (``kicad10`` -> ``"10"``, not ``"0"``).
2. The ``{kicad_version}`` fp-lib-table path interpolation bug (literal,
   un-interpolated ``{kicad_version}`` leaking into fallback paths).
3. Parsing real KiCad-10 ``.kicad_sym`` grammar: ``extends`` (derived symbols)
   and ``body_style`` (De Morgan alternates).
"""

import os
from pathlib import Path

import pytest

import skidl
from skidl import KICAD10, SchLib, lib_search_paths, set_default_tool

FIXTURE_DIR = str(Path(__file__).parent.parent / "test_data" / "kicad10")


def test_kicad10_tool_registered():
    """The kicad10 package auto-registers as a tool with the .kicad_sym suffix."""
    from skidl.tools import ALL_TOOLS, lib_suffixes

    assert "kicad10" in ALL_TOOLS
    assert skidl.KICAD10 == "kicad10"
    assert lib_suffixes["kicad10"] == [".kicad_sym"]


def test_version_derivation_two_digits():
    """Version is derived from the module name and must survive two digits.

    A naive ``module_name[-1]`` slice would yield ``"0"`` for ``kicad10``; the
    real extraction slices off the ``"kicad"`` prefix and must yield ``"10"``.
    """
    import skidl.tools.kicad10.lib as k10
    import skidl.tools.kicad9.lib as k9

    assert k10.__name__.split(".")[-2][len("kicad"):] == "10"
    assert k9.__name__.split(".")[-2][len("kicad"):] == "9"


def test_fp_lib_tbl_dir_no_literal_interpolation(monkeypatch):
    """fp-lib-table fallback paths must interpolate the version, not leak the
    literal ``{kicad_version}`` placeholder (the missing-f-string bug)."""
    import skidl.tools.kicad10.lib as k10

    captured = {}

    def spy(name, paths=None, **kwargs):
        captured["paths"] = list(paths or [])
        return ""

    monkeypatch.setattr(k10, "get_abs_filename", spy)
    k10.get_fp_lib_tbl_dir()

    assert captured["paths"], "no candidate paths were built"
    for p in captured["paths"]:
        assert "{kicad_version}" not in p, f"literal placeholder leaked: {p}"
    # And the version was actually substituted somewhere.
    assert any("kicad/10.0" in p for p in captured["paths"])


def test_default_symbol_discovery_prefers_matching_version(monkeypatch, tmp_path):
    """With ``KICAD10_SYMBOL_DIR`` unset, discovery finds a stock install and
    prefers the install whose version matches the tool (newest otherwise)."""
    import skidl.tools.kicad10.lib as k10

    # Build a fake multi-version install root under PROGRAMFILES.
    root = tmp_path / "KiCad"
    for ver in ("9.0", "10.0", "11.0"):
        (root / ver / "share" / "kicad" / "symbols").mkdir(parents=True)
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
    monkeypatch.delenv("KICAD10_SYMBOL_DIR", raising=False)

    dirs = k10._discover_default_symbol_dirs("10")
    assert dirs, "nothing discovered"
    # The 10.0 install must sort first (matches the tool version).
    assert f"{os.sep}10.0{os.sep}" in dirs[0], dirs


def test_default_lib_paths_uses_discovery(monkeypatch):
    """default_lib_paths falls back to discovery instead of only warning."""
    import skidl.tools.kicad10.lib as k10

    monkeypatch.delenv("KICAD10_SYMBOL_DIR", raising=False)
    monkeypatch.setattr(
        k10, "_discover_default_symbol_dirs", lambda ver: ["/fake/symbols"]
    )
    paths = k10.default_lib_paths()
    assert "/fake/symbols" in paths


def _load_fixture():
    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = [FIXTURE_DIR]
    SchLib.reset()
    return SchLib("kicad10_grammar")


def test_kicad10_fixture_parses_extends_and_body_style():
    """The vendored KiCad-10 fixture parses: a base symbol, a derived symbol
    (``extends``), and a De Morgan (``body_style``) symbol."""
    lib = _load_fixture()
    names = {p.name for p in lib.parts}
    assert names == {"C_Feedthrough", "Filter_EMI_C", "4001"}


def test_kicad10_extends_inherits_pins():
    """A symbol that ``extends`` its parent inherits the parent's pins."""
    lib = _load_fixture()
    child = lib["Filter_EMI_C"]  # extends "C_Feedthrough"
    parent = lib["C_Feedthrough"]
    assert len(child.pins) == len(parent.pins) == 3
    assert child.ref_prefix == "C"


def test_kicad10_body_style_symbol_parses():
    """A De Morgan (``body_style``) symbol parses to its real pin count and its
    alternate-style units are not double-counted."""
    lib = _load_fixture()
    gate = lib["4001"]  # quad 2-input NOR, has body_style alternates
    assert len(gate.pins) == 14
    assert gate.ref_prefix == "U"
