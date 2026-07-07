# -*- coding: utf-8 -*-

"""Regression tests for the lib_symbols nickname bug.

The KiCad-9 schematic writer used ``os.path.splitext(part.lib.filename)[0]`` to
derive a symbol's library nickname. ``splitext`` strips the extension but NOT
the directory, so when a symbol was resolved by scanning an absolute
``KICAD9_SYMBOL_DIR`` the emitted lib_id became the full path, e.g.
``(symbol "C:\\Program Files\\...\\Connector_Generic:Conn_01x02"`` — which KiCad
refuses to load ("Failed to load schematic"). ``_lib_nickname`` must return just
the nickname regardless of how the library filename was recorded.
"""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from skidl.tools.kicad9.sexp_schematic import _lib_nickname


class _FakeLib:
    def __init__(self, filename):
        self.filename = filename


class _FakePart:
    def __init__(self, filename):
        self.lib = _FakeLib(filename)


@pytest.mark.parametrize(
    "filename, expected",
    [
        (r"C:\Program Files\KiCad\10.0\share\kicad\symbols\Connector_Generic.kicad_sym",
         "Connector_Generic"),
        ("/usr/share/kicad/symbols/Device.kicad_sym", "Device"),
        ("Connector_Generic.kicad_sym", "Connector_Generic"),
        ("Connector_Generic", "Connector_Generic"),
        (None, "Device"),
        ("", "Device"),
    ],
)
def test_lib_nickname_strips_path_and_extension(filename, expected):
    assert _lib_nickname(_FakePart(filename)) == expected


# --------------------------------------------------------------------------- #
# End-to-end: render a part that previously triggered the bug and load it.
# --------------------------------------------------------------------------- #

_HAS_LIBS = (
    bool(os.environ.get("KICAD9_SYMBOL_DIR"))
    or os.path.exists("/usr/share/kicad/symbols")
    or os.path.exists(os.path.expanduser("~/.local/share/kicad/9.0/symbols"))
)
_KICAD_CLI = shutil.which("kicad-cli") or r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"
_HAS_CLI = os.path.exists(_KICAD_CLI) or shutil.which("kicad-cli") is not None


@pytest.mark.skipif(not _HAS_LIBS, reason="KiCad symbol libraries not available")
def test_rendered_lib_symbols_have_clean_nicknames():
    from skidl import Circuit, Net, Part

    d = tempfile.mkdtemp(prefix="skidl_nick_")
    try:
        c = Circuit(name="nick")
        with c:
            # Conn_01x02 is one of the symbols that reproduced the full-path bug.
            j = Part("Connector_Generic", "Conn_01x02")
            r = Part("Device", "R", value="1k")
            vin, gnd = Net("VIN"), Net("GND")
            j[1] += vin
            j[2] += gnd
            r[1] += vin
            r[2] += gnd
            c.generate_schematic(
                filepath=d, top_name="nick", auto_stub=True,
                auto_stub_fallback="labels",
            )
        sch = os.path.join(d, "nick.kicad_sch")
        text = open(sch, encoding="utf-8").read()

        # Every lib_symbols name must be a bare "Nickname:Part" — no path
        # separators, no drive letter, exactly one colon.
        for name in re.findall(r'\(symbol "([^"]+)"', text):
            if "_" in name and ":" not in name:
                continue  # sub-unit like "R_0_1"
            assert "\\" not in name and "/" not in name, name
            assert not re.match(r"^[A-Za-z]:", name) or name.count(":") == 1, name

        # And kicad-cli must be able to load it.
        if _HAS_CLI:
            out = os.path.join(d, "nick.net")
            proc = subprocess.run(
                [_KICAD_CLI, "sch", "export", "netlist", "--output", out, sch],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, (
                f"kicad-cli failed to load: {proc.stdout}{proc.stderr}"
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)
