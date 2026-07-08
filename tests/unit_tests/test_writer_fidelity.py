# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Phase 2 writer-fidelity / save-crash hardening tests.

These prove skidl's ``.kicad_sch`` writer does not emit a file that KiCad opens
but segfaults on *save* (a GUI-only failure mode kicad-cli ERC/netlist/pdf all
tolerate). The three defect classes root-caused in the circuit-synth/ksa effort:

  1. Dangling ``(path "/")`` instance paths (ordinary, power, and unit>=2
     symbols) -- the null-deref-on-save class.
  2. Zero-length wires (coincident endpoints).
  3. Inconsistent ``(project "...")`` names across instances.

Gate acceptance is exercised over real generated fixtures (a plain divider, a
power-symbol design, and a dual op-amp LM358 -- the multi-unit case that is
skidl issue #318's defect class) plus a deliberately-broken fixture that MUST
fail the gate (proves the gate detects the class, not just that our output is
clean).
"""

import re
import shutil

import pytest

from skidl import KICAD10, Net, Part, generate_schematic, lib_search_paths, set_default_tool
from tests.unit_tests.test_kicad10 import requires_kicad10
from tests.utils.kicad_gate import KicadCliUnavailable, assert_kicad_save_ok


def _setup_kicad10():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()


def _build_divider():
    r1 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0805_2012Metric")
    r2 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0805_2012Metric")
    Net("A").connect(r1[1])
    Net("MID").connect(r1[2], r2[1])
    Net("B").connect(r2[2])


def _build_power_design():
    r1 = Part("Device", "R", value="1k", footprint="Resistor_SMD:R_0805_2012Metric")
    gnd = Part("power", "GND")
    vcc = Part("power", "VCC")
    Net("VCC").connect(vcc[1], r1[1])
    Net("GND").connect(r1[2], gnd[1])


def _build_dual_opamp():
    """LM358 with BOTH units placed (the multi-unit save-crash / #318 case)."""
    op = Part("Amplifier_Operational", "LM358",
              footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    gnd = Part("power", "GND")
    vcc = Part("power", "VCC")
    ua, ub = op.unit["uA"], op.unit["uB"]
    Net("INA").connect(ua[3])
    Net("OUTA").connect(ua[1], ua[2])
    Net("INB").connect(ub[5])
    Net("OUTB").connect(ub[7], ub[6])
    Net("VCC").connect(vcc[1], op[8])
    Net("GND").connect(gnd[1], op[4])


def _generate(tmp_path, name, builder):
    _setup_kicad10()
    builder()
    out = tmp_path / name
    out.mkdir()
    generate_schematic(tool=KICAD10, filepath=str(out), top_name=name)
    return sorted(out.glob("*.kicad_sch"))


# ---- Static (no KiCad needed) defect-class checks on generated output ----


@pytest.mark.parametrize(
    "name,builder",
    [("divider", _build_divider), ("power", _build_power_design),
     ("dual_opamp", _build_dual_opamp)],
)
def test_no_save_crash_defect_classes_in_output(tmp_path, name, builder):
    """Generated output is free of all three save-crash defect classes."""
    sch_files = _generate(tmp_path, name, builder)
    assert sch_files, "no schematic generated"

    projects = set()
    for f in sch_files:
        txt = f.read_text(encoding="utf-8")
        # 1. No bare "/" or empty instance paths.
        for m in re.finditer(r'\(path\s+"([^"]*)"', txt):
            assert m.group(1) not in ("", "/"), f"dangling instance path in {f.name}"
        # 2. No zero-length wires.
        for wm in re.finditer(
            r'\(wire\b.*?\(pts\s*\(xy\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)\s*'
            r'\(xy\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)',
            txt, re.S,
        ):
            x1, y1, x2, y2 = map(float, wm.groups())
            assert not (x1 == x2 and y1 == y2), f"zero-length wire in {f.name}"
        # 3. Collect project names.
        projects.update(re.findall(r'\(project\s+"([^"]*)"', txt))
    # Uniform project name across all instances/sheets.
    assert len(projects) <= 1, f"inconsistent project names: {projects}"


def test_multi_unit_shares_root_instance_path(tmp_path):
    """Every placed unit of a multi-unit part shares one valid root path
    (the cs Stage-23 #B / skidl #318 save-crash class)."""
    sch_files = _generate(tmp_path, "dual_opamp", _build_dual_opamp)
    txt = sch_files[0].read_text(encoding="utf-8")
    # The sheet's own uuid is the first (uuid ...) in the file.
    sheet_uuid = re.search(r"\(uuid\s+([0-9a-f-]+)\)", txt).group(1)
    lm358_paths = [
        m.group(1)
        for m in re.finditer(
            r'\(symbol\b.*?\(lib_id "Amplifier_Operational:LM358".*?\(path "([^"]+)"',
            txt, re.S,
        )
    ]
    assert len(lm358_paths) >= 2, "expected multiple LM358 units placed"
    assert all(p == f"/{sheet_uuid}" for p in lm358_paths), lm358_paths


# ---- Live save gate (requires a real KiCad 10 install) ----


@requires_kicad10
@pytest.mark.parametrize(
    "name,builder",
    [("divider", _build_divider), ("power", _build_power_design),
     ("dual_opamp", _build_dual_opamp)],
)
def test_generated_fixture_passes_save_gate(tmp_path, name, builder):
    sch_files = _generate(tmp_path, name, builder)
    for f in sch_files:
        assert_kicad_save_ok(f)  # raises on any save-crash class


@requires_kicad10
def test_gate_detects_deliberately_dangling_path(tmp_path):
    """A fixture with bare ``(path "/")`` MUST fail the gate -- proving the gate
    detects the dangling-path save-crash class, not just that our output is clean.
    On KiCad 10 this trips a writer access violation (Windows 0xC0000005 /
    Linux SIGSEGV 139)."""
    sch_files = _generate(tmp_path, "good", _build_dual_opamp)
    good = sch_files[0]
    broken = good.with_name("broken_dangling.kicad_sch")
    broken_txt = re.sub(r'\(path "/[0-9a-f-]+"', '(path "/"', good.read_text(encoding="utf-8"))
    assert '(path "/"' in broken_txt, "failed to construct a dangling-path fixture"
    broken.write_text(broken_txt, encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_kicad_save_ok(broken)
