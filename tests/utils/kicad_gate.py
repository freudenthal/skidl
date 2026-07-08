# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""KiCad save-crash gate helper (ported from circuit-synth ``kicad_gate_utils``).

KiCad can *load* a ``.kicad_sch`` fine yet **segfault on save**, truncating the
file, on several defects that ``kicad-cli`` ERC/netlist/pdf all tolerate -- a
GUI-only failure mode. This gate reproduces that headlessly: copy the file, run
``kicad-cli sch upgrade --force`` (KiCad's own writer) on the copy, and assert
the write round-trips.

Three traps make the naive "exit != 139" check a false gate (each pinned by a
fake-cli unit test in ``test_kicad_save_gate.py``):

  1. **Pipe rc trap** -- ``kicad-cli ... | tail; echo $?`` reports *tail's* rc,
     so a 139 segfault reads as 0. We read ``returncode`` directly (list argv,
     no shell).
  2. **0-byte truncation with rc=0** -- a crash mid-write can leave a 0-byte
     file, so assert ``size > 0`` too.
  3. **MSYS path trap** (Git-Bash) -- handing the Windows ``kicad-cli.exe`` an
     MSYS ``/tmp/...`` path silently writes 0 bytes with rc=0. Pass native
     paths only; the size assertion catches it regardless.

The gate is: **rc == 0 AND filesize > 0 AND the upgraded file reloads**
(``kicad-cli sch erc`` rc in {0, 5} -- 0 clean / 5 violations, both prove a
loadable file; any other code is a crash/parse failure).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path


def find_kicad_cli(explicit=None):
    """Return a path to ``kicad-cli``, or None if not found.

    Prefers an explicit path, then a version-numbered Windows install
    (newest first), then ``PATH``. Version numbers are globbed, never
    hardcoded, so a future KiCad release is discovered automatically.
    """
    if explicit and Path(explicit).exists():
        return str(explicit)

    program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    kicad_root = Path(program_files) / "KiCad"
    if kicad_root.is_dir():
        versioned = []
        for child in kicad_root.iterdir():
            if child.is_dir() and re.fullmatch(r"\d+(?:\.\d+)*", child.name):
                key = tuple(int(p) for p in child.name.split("."))
                versioned.append((key, child))
        versioned.sort(key=lambda t: t[0], reverse=True)
        for _, vdir in versioned:
            cand = vdir / "bin" / "kicad-cli.exe"
            if cand.exists():
                return str(cand)

    return shutil.which("kicad-cli")


class KicadCliUnavailable(RuntimeError):
    """Raised when no kicad-cli can be located for the gate."""


def assert_kicad_save_ok(sch_path, kicad_cli=None):
    """Assert KiCad can re-save ``sch_path`` without crashing.

    Copies the file, runs ``kicad-cli sch upgrade --force`` on the copy, and
    asserts ``rc == 0`` AND the copy is non-empty AND ``kicad-cli sch erc``
    reloads it. Raises ``AssertionError`` on a save-crash class, or
    ``KicadCliUnavailable`` if kicad-cli is not installed (callers decide
    whether to skip).
    """
    sch_path = Path(sch_path)
    cli = find_kicad_cli(kicad_cli)
    if not cli:
        raise KicadCliUnavailable("kicad-cli (KiCad 10) not available")

    copy = sch_path.with_name(sch_path.stem + "_savecopy.kicad_sch")
    shutil.copyfile(sch_path, copy)

    up = subprocess.run(
        [cli, "sch", "upgrade", "--force", str(copy)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert up.returncode == 0, (
        f"kicad-cli sch upgrade returned {up.returncode} on {copy.name} "
        f"(139 = segfault -> KiCad save crash). stderr: {up.stderr.strip()}"
    )
    size = copy.stat().st_size
    assert size > 0, (
        f"{copy.name} is 0 bytes after upgrade (a crash truncated it, or an MSYS "
        f"path trap); rc was {up.returncode}"
    )
    erc = subprocess.run(
        [cli, "sch", "erc", "-o", str(copy.with_suffix(".rpt")), str(copy)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert erc.returncode in (0, 5), (
        f"upgraded {copy.name} failed to reload in kicad-cli sch erc "
        f"(rc={erc.returncode}). stderr: {erc.stderr.strip()}"
    )
