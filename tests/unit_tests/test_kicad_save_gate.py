# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Unit tests for the save-crash gate helper ``assert_kicad_save_ok``.

Pins the gate's semantics WITHOUT needing KiCad installed, by handing the helper
a fake ``kicad-cli`` that reproduces each failure mode:

  * nonzero rc (segfault proxy)            -> gate must fail
  * rc 0 but 0-byte output (truncation)    -> gate must fail
  * rc 0, size > 0, reload ok              -> gate must pass

This is the guard that "exit != 139" alone is not a gate.
"""

import stat
import sys
import textwrap
from pathlib import Path

import pytest

from tests.utils.kicad_gate import assert_kicad_save_ok

# argv after the fake cli path is e.g.
#   sch upgrade --force <copy.kicad_sch>
#   sch erc -o <rpt> <copy.kicad_sch>
# The <copy> is the target; for upgrade it is the last arg, for erc it follows -o.
_STUB_TEMPLATE = textwrap.dedent("""\
    import sys
    args = sys.argv[1:]
    if "upgrade" in args:
        target = args[-1]
        mode = {mode!r}
        if mode == "crash":
            sys.exit(139)
        if mode == "zerobyte":
            open(target, "w").close()  # truncate to 0 bytes, rc 0
            sys.exit(0)
        sys.exit(0)  # ok: leave the copy's content intact
    if "erc" in args:
        sys.exit(0)
    sys.exit(0)
    """)


def _make_fake_cli(tmp_path, mode):
    """Write a fake kicad-cli (launcher + python stub); return its path."""
    stub = tmp_path / f"fake_stub_{mode}.py"
    stub.write_text(_STUB_TEMPLATE.format(mode=mode), encoding="utf-8")
    py = sys.executable
    if sys.platform.startswith("win"):
        launcher = tmp_path / f"fake_kicad_{mode}.cmd"
        launcher.write_text(f'@echo off\r\n"{py}" "{stub}" %*\r\n', encoding="utf-8")
    else:
        launcher = tmp_path / f"fake_kicad_{mode}.sh"
        launcher.write_text(f'#!/bin/sh\nexec "{py}" "{stub}" "$@"\n', encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
    return str(launcher)


@pytest.fixture
def sch(tmp_path):
    p = tmp_path / "design.kicad_sch"
    p.write_text("(kicad_sch (version 20230409))\n", encoding="utf-8")
    return p


def test_gate_passes_when_rc0_size_positive_and_reloads(sch, tmp_path):
    cli = _make_fake_cli(tmp_path, "ok")
    assert_kicad_save_ok(sch, kicad_cli=cli)  # must not raise


def test_gate_fails_on_nonzero_rc(sch, tmp_path):
    cli = _make_fake_cli(tmp_path, "crash")
    with pytest.raises(AssertionError, match="upgrade returned 139"):
        assert_kicad_save_ok(sch, kicad_cli=cli)


def test_gate_fails_on_zero_byte_output(sch, tmp_path):
    cli = _make_fake_cli(tmp_path, "zerobyte")
    with pytest.raises(AssertionError, match="0 bytes"):
        assert_kicad_save_ok(sch, kicad_cli=cli)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
