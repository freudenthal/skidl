# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: BOM / reference cosmetics (LLC E2E finding R8).

- Bare float part values render in engineering notation (``2.2e-05`` -> ``22u``);
  exact user strings pass through verbatim.
- Refs without a trailing number are finalized (``CO`` -> ``CO1``) before output,
  so KiCad never shows an unannotated ``CO?``; netlist and schematic stay
  ref-equivalent.
- Power flags use one ref format (``#FLG01``-style) matching the skidl-eda
  ERC-autofix path.
"""

import os
import re

import pytest

from skidl import KICAD10, Net, Part, generate_netlist, generate_schematic
from skidl import lib_search_paths, set_default_tool
from skidl.utilities import eng_value_str


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


# --- eng_value_str unit tests ----------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # float-shaped strings reformat
        ("2.2e-05", "22u"),
        ("1e-07", "100n"),
        ("0.00022", "220u"),
        ("0.1", "100m"),
        ("2.2", "2.2"),
        ("4700.0", "4.7k"),
        ("-2.5e-06", "-2.5u"),
        # exact strings pass through verbatim
        ("22uH", "22uH"),
        ("SS3H10", "SS3H10"),
        ("100n", "100n"),
        ("10", "10"),  # integer string: exact, not a float repr
        ("4700", "4700"),
        ("PWR_FLAG", "PWR_FLAG"),
        ("", ""),
    ],
)
def test_eng_value_str_strings(raw, expected):
    assert eng_value_str(raw) == expected


def test_eng_value_str_python_floats():
    assert eng_value_str(2.2e-05) == "22u"
    assert eng_value_str(1e-07) == "100n"
    assert eng_value_str(10.0) == "10"
    assert eng_value_str(0.0) == "0.0"  # zero passes through as-is
    assert eng_value_str(4700) == "4700"  # ints are exact values


# --- render-level checks -----------------------------------------------------


def _render(tmp_path, top):
    generate_netlist(tool=KICAD10, file_=str(tmp_path / f"{top}.net"))
    generate_schematic(tool=KICAD10, filepath=str(tmp_path), top_name=top)
    net = (tmp_path / f"{top}.net").read_text(encoding="utf-8", errors="replace")
    sch = (tmp_path / f"{top}.kicad_sch").read_text(encoding="utf-8", errors="replace")
    return net, sch


def test_render_finalizes_refs_and_formats_values(tmp_path):
    """The LLC canary pattern: ref='CO' + value=str(float) must come out as an
    annotated ref (CO1) and an engineering-notation value (220u) in BOTH the
    netlist and the schematic (ref-equivalent), with no '?'-style refs."""
    _setup()
    co = Part(
        "Device", "C", ref="CO", value=str(220e-6),
        footprint="Capacitor_SMD:C_1210_3225Metric",
    )
    rl = Part(
        "Device", "R", ref="RL", value="10",
        footprint="Resistor_SMD:R_1206_3216Metric",
    )
    Net("A").connect(co[1], rl[1])
    Net("B").connect(co[2], rl[2])
    net, sch = _render(tmp_path, "cosm")

    sch_refs = set(re.findall(r'\(property "Reference" "([^"]+)"', sch))
    net_refs = set(re.findall(r'\(comp\s*\(ref "([^"]+)"', net))
    # Both instance refs finalized, identically, in both outputs.
    assert "CO1" in sch_refs and "RL1" in sch_refs, sch_refs
    assert "CO1" in net_refs and "RL1" in net_refs, net_refs
    assert not any(r.endswith("?") for r in sch_refs | net_refs)
    # No unannotated instance refs remain (lib_symbols templates aside).
    assert "CO" not in net_refs and "RL" not in net_refs

    sch_vals = set(re.findall(r'\(property "Value" "([^"]+)"', sch))
    assert "220u" in sch_vals, sch_vals
    assert "0.00022" not in sch_vals and "2.2e-05" not in sch_vals
    assert '(value "220u")' in net
    # The exact string value is untouched.
    assert "10" in sch_vals


def test_power_flags_use_two_digit_refs(tmp_path):
    """Structural PWR_FLAGs use the #FLG01 format (matches the ERC autofix)."""
    from skidl import POWER

    _setup()
    r1 = Part(
        "Device", "R", ref="R1", value="1k",
        footprint="Resistor_SMD:R_0603_1608Metric",
    )
    vcc = Net("VCC")
    vcc.drive = POWER
    gnd = Net("GND")
    gnd.drive = POWER
    r1[1] += vcc
    r1[2] += gnd
    _, sch = _render(tmp_path, "flg")
    flgs = re.findall(r'"(#FLG\d+)"', sch)
    assert flgs, "expected structural PWR_FLAG refs"
    assert all(re.fullmatch(r"#FLG\d{2}", f) for f in set(flgs)), set(flgs)
