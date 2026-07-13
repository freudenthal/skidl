# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: the 'forgot setup_kicad10()' library-lookup hint (HV LLC E2E N2).

When a KiCad backend is asked for a symbol library with no real symbol paths
bound, the raw ``Can't open file: <lib>`` names a random lib rather than the
cause. ``_unbound_kicad_lib_hint`` appends a pointed hint, but only for a KiCad
tool with defaulted paths -- a genuine missing file (real paths bound) is left
alone so it isn't masked.
"""

from skidl.schlib import _unbound_kicad_lib_hint


def test_hint_for_unbound_kicad10():
    hint = _unbound_kicad_lib_hint("kicad10", ["."])
    assert "setup_kicad10()" in hint


def test_hint_for_unbound_kicad10_empty_paths():
    assert "setup_kicad10()" in _unbound_kicad_lib_hint("kicad10", [])
    assert "setup_kicad10()" in _unbound_kicad_lib_hint("kicad10", None)


def test_no_hint_when_real_paths_bound():
    # A genuine missing file with real library dirs bound must not be masked.
    assert _unbound_kicad_lib_hint("kicad10", ["C:/Program Files/KiCad/10.0/"
                                               "share/kicad/symbols"]) == ""


def test_no_hint_for_non_kicad_tool():
    assert _unbound_kicad_lib_hint("spice", ["."]) == ""
    assert _unbound_kicad_lib_hint("skidl", ["."]) == ""
