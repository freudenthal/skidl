# -*- coding: utf-8 -*-

"""Regression test for the mis-quoted ``private`` symbol-property flag.

A KiCad symbol property may carry a bare ``private`` KEYWORD flag before its
name -- ``(property private "KLC_S3.3" "note")`` -- which ships on the KLC
graphical-note properties of library symbols such as ``Device:Crystal_GND23``
(and the other 4-pin crystals). The schematic writer's ``add_quotes`` quotes
every string of a ``property`` list from index 1, so it emitted
``(property "private" ...)``. That is invalid: KiCad then FAILS TO LOAD the whole
sheet and silently drops every component on it from the exported netlist -- the
root cause of ``stm32_bluepill`` / ``feather_rp2040`` rendering with their
crystal + load caps (and, cascading, their MCU) missing.

``_restore_private_property_flags`` must put the bare flag back after quoting.
"""

import os
import re
import tempfile

from simp_sexp import Sexp

from skidl.tools.kicad10.sexp_schematic import (
    _restore_private_property_flags,
    _write_sexp_schematic,
)


def test_restore_private_flag_unquotes_only_the_flag():
    """Only a leading ``"private"`` in a property is unquoted; the name/value stay
    quoted, and an ordinary property is untouched."""
    sexp = Sexp(
        [
            "symbol",
            ["property", '"private"', '"KLC_S3.3"', '"a graphical note"'],
            ["property", '"Reference"', '"Y1"'],
        ]
    )
    _restore_private_property_flags(sexp)
    # the flag is now a bare keyword...
    assert sexp[1][1] == "private"
    # ...but the real name and value keep their quotes...
    assert sexp[1][2] == '"KLC_S3.3"'
    assert sexp[1][3] == '"a graphical note"'
    # ...and an ordinary property (name at index 1) is left alone.
    assert sexp[2][1] == '"Reference"'


def test_written_schematic_emits_bare_private_flag():
    """End to end: a schematic Sexp carrying a private-flag property is written
    with the flag BARE, so KiCad can load it (KiCad rejects the quoted form)."""
    schematic = Sexp(
        [
            "kicad_sch",
            ["version", 20230409],
            ["generator", "skidl"],
            [
                "symbol",
                ["lib_id", "Device:Crystal_GND23"],
                # a KLC note property with the leading private keyword flag
                ["property", "private", "KLC_S3.3", "The rectangle is a note"],
                # an ordinary property whose NAME must stay quoted
                ["property", "Reference", "Y1", ["at", 0, 0, 0]],
            ],
        ]
    )
    fd, path = tempfile.mkstemp(suffix=".kicad_sch")
    os.close(fd)
    try:
        _write_sexp_schematic(schematic, path)
        text = open(path, encoding="utf-8").read()
    finally:
        os.remove(path)

    # The flag must be bare; the quoted form is what breaks KiCad's loader.
    assert re.search(r'\(property\s+private\s+"KLC_S3\.3"', text), text
    assert '(property "private"' not in text
    # The ordinary property keeps a quoted name.
    assert re.search(r'\(property\s+"Reference"\s+"Y1"', text), text
