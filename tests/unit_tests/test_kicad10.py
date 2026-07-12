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
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import skidl
from skidl import KICAD10, SchLib, lib_search_paths, set_default_tool

FIXTURE_DIR = str(Path(__file__).parent.parent / "test_data" / "kicad10")

# Real KiCad-10 install discovery (Windows-first; skip integration tests if absent).
_KICAD_CLI_CANDIDATES = [
    r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
    "/usr/bin/kicad-cli",
    "/usr/local/bin/kicad-cli",
    shutil.which("kicad-cli") or "",
]
KICAD_CLI = next((c for c in _KICAD_CLI_CANDIDATES if c and Path(c).exists()), None)


def _kicad10_symbols_available():
    import skidl.tools.kicad10.lib as k10

    return bool(k10._discover_default_symbol_dirs("10")) or bool(
        os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    )


requires_kicad10 = pytest.mark.skipif(
    not (KICAD_CLI and _kicad10_symbols_available()),
    reason="requires a real KiCad 10 install (kicad-cli + stock symbol libraries)",
)


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

    assert k10.__name__.split(".")[-2][len("kicad") :] == "10"
    assert k9.__name__.split(".")[-2][len("kicad") :] == "9"


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


# --------------------------------------------------------------------------
# Integration: generation vs a real KiCad 10 install (Phase 1.5 / 1.6).
# --------------------------------------------------------------------------


def _save_gate_ok(sch_path: Path):
    """Hardened save gate: rc==0 AND non-empty AND reloads (see Phase 2.1).

    Reproduces the GUI save-crash headlessly via ``kicad-cli sch upgrade`` on a
    copy, then confirms the upgraded file reloads (erc rc in {0, 5}). Uses
    native paths only (never an MSYS /tmp path) and reads the rc directly.
    """
    work = sch_path.parent / (sch_path.stem + "_gate.kicad_sch")
    shutil.copy(sch_path, work)
    up = subprocess.run(
        [KICAD_CLI, "sch", "upgrade", str(work)], capture_output=True, text=True
    )
    if up.returncode != 0:
        return False, f"upgrade rc={up.returncode}"
    if work.stat().st_size == 0:
        return False, "upgraded file is 0 bytes"
    erc = subprocess.run(
        [KICAD_CLI, "sch", "erc", str(work)], capture_output=True, text=True
    )
    if erc.returncode not in (0, 5):
        return False, f"reload erc rc={erc.returncode}"
    return True, f"rc=0 size={sch_path.stat().st_size}"


def _build_divider():
    from skidl import Net, Part

    r1 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0805_2012Metric")
    r2 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0805_2012Metric")
    gnd = Part("power", "GND")
    vcc = Part("power", "VCC")
    Net("VCC").connect(vcc[1], r1[1])
    Net("VMID").connect(r1[2], r2[1])
    Net("GND").connect(r2[2], gnd[1])


@requires_kicad10
def test_kicad10_netlist_generation(tmp_path):
    """A KICAD10 netlist generates with a version stamp and the real parts."""
    from skidl import generate_netlist

    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    _build_divider()

    net_path = tmp_path / "divider.net"
    generate_netlist(tool=KICAD10, file_=str(net_path))
    text = net_path.read_text(encoding="utf-8")
    assert re.search(r"\(version\s+\S+\)", text)
    assert len(re.findall(r"\(comp\b", text)) >= 2


@requires_kicad10
def test_kicad10_multiunit_shares_base_reference(tmp_path):
    """A multi-unit part (incl. its dedicated power unit) renders as ONE reference
    with distinct (unit N), not the compound "U1.uA"/"U1.uE" per unit -- KiCad reads
    a compound ref as separate components, which lost each amp its shared power unit
    (missing_power_pin) and diverged the schematic from the netlist (B1/B2)."""
    from skidl import Net, Part, generate_schematic

    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()
    # Pick a real multi-unit op-amp with a dedicated power unit.
    part = None
    for name in ("ADA4807-4ARUZ", "LM2902", "LM324", "TL074"):
        try:
            part = Part("Amplifier_Operational", name, ref="U1")
            break
        except Exception:
            builtins.default_circuit.mini_reset()
    if part is None:
        pytest.skip("no multi-unit op-amp with a power unit in the installed libs")
    if len(part.unit) < 2:
        pytest.skip(f"{part.name} did not resolve as multi-unit")

    vp, vn = Net("V+"), Net("V-")
    # Connect the dedicated power unit's pins (V+/V-) so it is placed.
    for pin in part.pins:
        nm = (getattr(pin, "name", "") or "").upper()
        if nm in ("V+", "VCC", "VDD"):
            pin += vp
        elif nm in ("V-", "VEE", "VSS"):
            pin += vn

    out = tmp_path / "mu"
    out.mkdir()
    generate_schematic(tool=KICAD10, filepath=str(out), top_name="mu")
    text = sorted(out.glob("*.kicad_sch"))[0].read_text(encoding="utf-8")
    # No compound unit ref leaked into the schematic.
    assert "U1.u" not in text, "compound PartUnit reference leaked into the schematic"
    # Every unit instance references the base "U1".
    assert '(reference "U1")' in text
    # More than one distinct (unit N) present (multi-unit really rendered).
    units = set(re.findall(r"\(unit (\d+)\)", text))
    assert len(units) >= 2, units


@requires_kicad10
def test_kicad10_in_bom_false_renders_no(tmp_path):
    """Part(in_bom=False) renders (in_bom "no") on the placed symbol instance so
    a model-only element can be kept out of the exported BOM (C9); a default
    part stays (in_bom "yes")."""
    from skidl import Net, Part, generate_schematic

    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()
    r1 = Part("Device", "R", ref="R1", value="10k",
              footprint="Resistor_SMD:R_0805_2012Metric")
    c1 = Part("Device", "C", ref="CHV1", value="12p", in_bom=False,
              footprint="Capacitor_SMD:C_0805_2012Metric")
    Net("A").connect(r1[1], c1[1])
    Net("0").connect(r1[2], c1[2])

    out = tmp_path / "bom"
    out.mkdir()
    generate_schematic(tool=KICAD10, filepath=str(out), top_name="bom")
    text = sorted(out.glob("*.kicad_sch"))[0].read_text(encoding="utf-8")
    # Placed symbol blocks: the one holding CHV1's Reference must carry
    # (in_bom no); the R1 block keeps (in_bom yes). (KiCad serializes the flag
    # as an unquoted token.)
    blocks = text.split("(symbol")
    chv = [b for b in blocks if '"CHV1"' in b and "lib_id" in b]
    assert chv, "CHV1 symbol instance not found"
    assert any("(in_bom no)" in b for b in chv), "CHV1 not marked in_bom no"
    r_blocks = [b for b in blocks if '"R1"' in b and "lib_id" in b]
    assert r_blocks and all("(in_bom yes)" in b for b in r_blocks), "R1 not in_bom yes"


def test_kicad10_sourcing_kwargs_render_as_properties(tmp_path):
    """MPN/Manufacturer bare kwargs (and a fields={} dict) pass through to
    schematic properties so they reach the BOM (E2E A5)."""
    from skidl import Net, Part, generate_schematic

    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()
    q1 = Part("Device", "R", ref="Q1", value="IRF740",
              Manufacturer="Vishay", MPN="IRF740PBF",
              fields={"Distributor": "DigiKey"},
              footprint="Package_TO_SOT_THT:TO-220-3_Vertical")
    r1 = Part("Device", "R", ref="R1", value="10k")  # no sourcing kwargs
    Net("A").connect(q1[1], r1[1])
    Net("0").connect(q1[2], r1[2])

    out = tmp_path / "src"
    out.mkdir()
    generate_schematic(tool=KICAD10, filepath=str(out), top_name="src")
    text = sorted(out.glob("*.kicad_sch"))[0].read_text(encoding="utf-8")
    blocks = text.split("(symbol")
    q = [b for b in blocks if '"Q1"' in b and "lib_id" in b]
    assert q, "Q1 symbol instance not found"
    qb = "".join(q)
    assert '(property "MPN" "IRF740PBF"' in qb, qb
    assert '(property "Manufacturer" "Vishay"' in qb
    assert '(property "Distributor" "DigiKey"' in qb
    # a part without sourcing kwargs emits none of them
    r_blocks = "".join(b for b in blocks if '"R1"' in b and "lib_id" in b)
    assert '(property "MPN"' not in r_blocks


@requires_kicad10
def test_kicad10_schematic_passes_save_gate(tmp_path):
    """A KICAD10-generated schematic (stamp 20230409) passes the hardened save
    gate — KiCad 10 upgrades the file on load without a save-crash."""
    from skidl import generate_schematic

    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    _build_divider()

    out = tmp_path / "out"
    out.mkdir()
    generate_schematic(tool=KICAD10, filepath=str(out), top_name="divider")
    sch_files = sorted(out.glob("*.kicad_sch"))
    assert sch_files, "no schematic generated"
    for f in sch_files:
        ok, detail = _save_gate_ok(f)
        assert ok, f"{f.name} failed save gate: {detail}"
