# -*- coding: utf-8 -*-

"""Power-symbol-first rendering (wired-render default).

Power nets must render as KiCad ``power:*`` symbols on EVERY render path (not
only ``auto_stub``): classified by ``drive == POWER`` or a power-name pattern,
excluded from the A* router, one symbol per pin, a cloned in-file ``(power)``
symbol for non-stock rail names, and exactly one project-wide ``PWR_FLAG`` per
undriven rail. See the wired-render-default plan, Phase 1.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from skidl import KICAD10, POWER, lib_search_paths, set_default_tool

# ---- real-KiCad-10 discovery (mirrors test_kicad10.py) --------------------
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


def _use_kicad10():
    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


# ---------------------------------------------------------------------------
# Unit: classification (no libs needed)
# ---------------------------------------------------------------------------


def test_mark_power_nets_classifies_drive_and_pattern():
    from skidl import Circuit, Net, Part
    from skidl.tools.kicad10.gen_schematic import mark_power_nets

    ckt = Circuit(name="cls")
    with ckt:
        rail = Net("VBIAS_28V")
        rail.drive = POWER  # non-stock name, classified via drive
        gnd = Net("GND")  # pattern-classified
        sig = Net("SIG")  # neither -> a signal net
        r1 = Part("Device", "R", value="1k")
        r2 = Part("Device", "R", value="1k")
        r1[1] += rail
        r1[2] += sig
        r2[1] += sig
        r2[2] += gnd

    mark_power_nets(ckt)
    assert getattr(rail, "_is_power_net", False) is True
    assert getattr(gnd, "_is_power_net", False) is True
    assert getattr(sig, "_is_power_net", False) is False
    # Power pins get stubbed (excluded from the A* router); signal pins do not.
    assert all(p.stub for p in rail.get_pins())
    assert all(not p.stub for p in sig.get_pins())


def test_mark_power_nets_flags_undriven_only():
    """A rail with no ``power_output`` pin needs a PWR_FLAG; a rail carrying a
    PWROUT pin (e.g. an explicit PWR_FLAG part / regulator output) does not."""
    from skidl import Circuit, Net, Part
    from skidl.tools.kicad10.gen_schematic import mark_power_nets

    ckt = Circuit(name="flg")
    with ckt:
        undriven = Net("VCC")
        driven = Net("VDD")
        r1 = Part("Device", "R", value="1k")
        r1[1] += undriven
        r1[2] += driven
        # PWR_FLAG's single pin is power_out -> marks `driven` as driven.
        flg = Part("power", "PWR_FLAG")
        flg[1] += driven

    mark_power_nets(ckt)
    assert getattr(undriven, "_needs_pwr_flag", None) is True
    assert getattr(driven, "_needs_pwr_flag", None) is False


# ---------------------------------------------------------------------------
# Render integration (needs real libs + kicad-cli)
# ---------------------------------------------------------------------------


def _render(build, top, tmp_path, **opts):
    from skidl import Circuit

    _use_kicad10()
    ckt = Circuit(name=top)
    build(ckt)
    out = tmp_path / top
    out.mkdir()
    render_opts = {"seed_placement": True, "auto_stub": False}
    render_opts.update(opts)
    ckt.generate_schematic(tool=KICAD10, filepath=str(out), top_name=top, **render_opts)
    text = "".join(
        p.read_text(encoding="utf-8") for p in sorted(out.glob("*.kicad_sch"))
    )
    return out, text


def _erc_error_types(out, top):
    sch = out / f"{top}.kicad_sch"
    rpt = out / f"{top}-erc.rpt"
    subprocess.run(
        [KICAD_CLI, "sch", "erc", "--output", str(rpt), "--severity-error", str(sch)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    txt = rpt.read_text(encoding="utf-8") if rpt.exists() else ""
    types = {}
    for m in re.finditer(r"\[(\w+)\]", txt):
        types[m.group(1)] = types.get(m.group(1), 0) + 1
    return types


def _erc_all_types(out, top):
    """ERC violation-type histogram including WARNINGS (lib_symbol_issues et al.
    are warnings, invisible to the --severity-error run above)."""
    sch = out / f"{top}.kicad_sch"
    rpt = out / f"{top}-erc-all.rpt"
    subprocess.run(
        [KICAD_CLI, "sch", "erc", "--output", str(rpt), str(sch)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    txt = rpt.read_text(encoding="utf-8") if rpt.exists() else ""
    types = {}
    for m in re.finditer(r"\[(\w+)\]:", txt):
        types[m.group(1)] = types.get(m.group(1), 0) + 1
    return types


def _divider(ckt):
    from skidl import Net, Part

    with ckt:
        vcc = Net("VCC"); vcc.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        mid = Net("MID")
        r1 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r2 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r1[1] += vcc; r1[2] += mid
        r2[1] += mid; r2[2] += gnd


def _custom_rail(ckt):
    from skidl import Net, Part

    with ckt:
        vbias = Net("VBIAS_28V"); vbias.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        mid = Net("MID")
        r1 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r2 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r1[1] += vbias; r1[2] += mid
        r2[1] += mid; r2[2] += gnd


@requires_kicad10
def test_wired_render_power_nets_not_routed(tmp_path):
    """On the wired path, power nets render as power symbols with NO routed
    wires (they are excluded from A*), and ERC has zero power_pin_not_driven."""
    out, text = _render(_divider, "pwrdiv", tmp_path)
    # Power symbols for both rails present.
    assert 'lib_id "power:VCC"' in text
    assert 'lib_id "power:GND"' in text
    # Exactly one PWR_FLAG per undriven rail (2 rails here).
    assert len(re.findall(r'lib_id "power:PWR_FLAG"', text)) == 2
    types = _erc_error_types(out, "pwrdiv")
    assert types.get("power_pin_not_driven", 0) == 0, types


@requires_kicad10
def test_power_stubs_option_offsets_symbol_onto_stub_wire(tmp_path):
    """With power_stubs=True every power symbol is pulled one grid step off its
    pin onto a short stub WIRE (the classic pin -> wire -> power-symbol look),
    and the render stays ERC-clean. Default (OFF) keeps symbols on the pin."""
    # Default: power symbols coincident with pins (no dedicated stub wire).
    out0, text0 = _render(_divider, "pwr0", tmp_path)
    # power_stubs ON: symbols offset, each sitting on a wire endpoint.
    out1, text1 = _render(_divider, "pwr1", tmp_path, power_stubs=True)

    def _sym_pts(txt):
        return [
            (round(float(m.group(1)), 2), round(float(m.group(2)), 2))
            for m in re.finditer(
                r'\(lib_id "power:(?!PWR_FLAG)[^"]+"\)\s*\(at ([\d.-]+) ([\d.-]+)', txt
            )
        ]

    def _wire_ends(txt):
        pts = set()
        for m in re.finditer(
            r"\(wire\s*\(pts\s*\(xy ([\d.-]+) ([\d.-]+)\)\s*\(xy ([\d.-]+) ([\d.-]+)\)",
            txt,
        ):
            x1, y1, x2, y2 = map(float, m.groups())
            pts.add((round(x1, 2), round(y1, 2)))
            pts.add((round(x2, 2), round(y2, 2)))
        return pts

    syms1 = _sym_pts(text1)
    assert syms1, "no power symbols emitted"
    ends1 = _wire_ends(text1)
    # Every power symbol sits at a wire endpoint (its stub), and on-grid.
    for sx, sy in syms1:
        assert (sx, sy) in ends1, f"power symbol {sx},{sy} not on a stub wire end"
        assert abs(sx / _GRID_MM - round(sx / _GRID_MM)) < 1e-3, (sx, sy)
        assert abs(sy / _GRID_MM - round(sy / _GRID_MM)) < 1e-3, (sx, sy)
    # The stub path produced MORE wires than the on-pin default.
    assert text1.count("(wire") > text0.count("(wire"), "no stub wires added"
    # Still ERC-clean.
    assert not _erc_error_types(out1, "pwr1"), _erc_error_types(out1, "pwr1")
    # DIRECTION: the stubs must push the power symbols AWAY from the part bodies,
    # not into them. In this divider the two resistors are stacked vertically and
    # VCC/GND are on the outermost pins, so the power symbols must be the vertical
    # EXTREMES -- beyond both resistor origins. A toward-body offset (the bug this
    # guards) would drop them between the resistors and fail this.
    res_y = [
        float(m.group(1))
        for m in re.finditer(r'\(lib_id "Device:R"\)\s*\(at [\d.-]+ ([\d.-]+)', text1)
    ]
    pw_y = [y for _x, y in syms1]
    assert res_y, "no resistors found"
    assert min(pw_y) < min(res_y), f"a power stub points INTO the body: {pw_y} vs {res_y}"
    assert max(pw_y) > max(res_y), f"a power stub points INTO the body: {pw_y} vs {res_y}"


@requires_kicad10
def test_custom_rail_gets_infile_power_symbol(tmp_path):
    """A non-stock rail name (drive=POWER) gets a cloned in-file power symbol whose
    Value is the rail name -- not a plain global label. It carries the project-local
    ``SKiDL_rails:`` nickname (not stock ``power:``) and is backed by a written
    ``SKiDL_rails.kicad_sym`` + ``sym-lib-table`` so ERC resolves it (0
    lib_symbol_issues) instead of complaining against the stock ``power`` lib."""
    out, text = _render(_custom_rail, "cust", tmp_path)
    assert 'symbol "SKiDL_rails:VBIAS_28V"' in text  # cloned lib definition
    assert 'lib_id "SKiDL_rails:VBIAS_28V"' in text  # instance
    assert 'property "Value" "VBIAS_28V"' in text  # value = rail name
    # The clone must NOT masquerade as a stock power-lib symbol (that is finding C).
    assert 'lib_id "power:VBIAS_28V"' not in text
    # The cloned definition carries the (power ...) flag, so KiCad treats it as a
    # real power symbol (global-by-name), not a plain global label.
    defn = text.split('symbol "SKiDL_rails:VBIAS_28V"', 1)[1].split("(symbol", 1)[0]
    assert "(power" in defn
    assert 'global_label "VBIAS_28V"' not in text
    # Project-local library + table backing the nickname.
    lib = out / "SKiDL_rails.kicad_sym"
    tbl = out / "sym-lib-table"
    assert lib.exists() and tbl.exists()
    lib_text = lib.read_text(encoding="utf-8")
    assert '(symbol "VBIAS_28V"' in lib_text  # bare name in the standalone lib
    assert '"SKiDL_rails"' in tbl.read_text(encoding="utf-8")
    types = _erc_error_types(out, "cust")
    assert types.get("power_pin_not_driven", 0) == 0, types
    # The whole point of finding C: no lib_symbol_issues for the custom rail. The
    # table's ${KIPRJMOD} only resolves when a project file is present (the harness
    # scaffold always writes one); stub a minimal .kicad_pro so ERC has that context.
    (out / "cust.kicad_pro").write_text("{}", encoding="utf-8")
    all_types = _erc_all_types(out, "cust")
    assert all_types.get("lib_symbol_issues", 0) == 0, all_types


@requires_kicad10
def test_one_pwr_flag_per_rail_project_wide(tmp_path):
    """Exactly ONE PWR_FLAG per undriven rail regardless of how many pins the
    rail has (power symbols connect globally by name, so one flag suffices), and
    every rail is still represented by at least one power symbol -- even when
    identical floating parts stack exactly (the fully-coincident cluster must not
    suppress its last representative)."""

    def build(ckt):
        from skidl import Net, Part

        with ckt:
            vcc = Net("VCC"); vcc.drive = POWER
            gnd = Net("GND"); gnd.drive = POWER
            for i in range(4):
                c = Part("Device", "C", value="100nF",
                         footprint="Capacitor_SMD:C_0603_1608Metric")
                c[1] += vcc
                c[2] += gnd

    out, text = _render(build, "manypin", tmp_path)
    # Each rail keeps at least one power symbol (never fully suppressed)...
    assert len(re.findall(r'lib_id "power:VCC"', text)) >= 1
    assert len(re.findall(r'lib_id "power:GND"', text)) >= 1
    # ...and exactly one PWR_FLAG per undriven rail (2 rails), project-wide.
    assert len(re.findall(r'lib_id "power:PWR_FLAG"', text)) == 2
    # And ERC sees both rails as driven.
    types = _erc_error_types(out, "manypin")
    assert types.get("power_pin_not_driven", 0) == 0, types


# ---------------------------------------------------------------------------
# hierarchical_sheet_pins option (KiCad hierarchical-interconnect surface)
# ---------------------------------------------------------------------------


def _hier_build(ckt):
    from skidl import Net, Part, subcircuit

    @subcircuit
    def stage(vin, vout, vpos, gnd):
        u = Part("Amplifier_Operational", "OPA340NA")
        r = Part("Device", "R", value="1k",
                 footprint="Resistor_SMD:R_0603_1608Metric")
        u["5"] += vpos; u["2"] += gnd; u["3"] += vin; u["4"] += vout; u["1"] += vout
        r[1] += vout; r[2] += gnd

    with ckt:
        vpos = Net("+5V"); vpos.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        a, b, c = Net("A"), Net("B"), Net("C")
        stage(a, b, vpos, gnd, tag="s1")
        stage(b, c, vpos, gnd, tag="s2")


@requires_kicad10
def test_hier_sheet_pins_default_off_is_clean(tmp_path):
    """By default boundary nets connect by global_label name -- no hierarchical
    labels or sheet pins are emitted, and the render is ERC-clean."""
    out, text = _render(_hier_build, "hspoff", tmp_path)
    assert "(hierarchical_label" not in text
    # No sheet pins (a sheet pin is `(pin NAME bidirectional ...)`; the shape of a
    # global_label is `(shape bidirectional)`, which must NOT be mistaken for one).
    assert re.search(r"\(pin \S+ bidirectional", text) is None
    # Boundary net B (s1->s2) connects by global label.
    assert 'global_label "B"' in text
    types = _erc_error_types(out, "hspoff")
    assert types.get("label_dangling", 0) == 0, types
    assert types.get("pin_not_connected", 0) == 0, types


def _hier_labels(text):
    """(name, x, y) for every hierarchical_label in a schematic text blob."""
    out = []
    for m in re.finditer(
        r'\(hierarchical_label "([^"]+)".*?\(at ([\d.-]+) ([\d.-]+)', text, re.DOTALL
    ):
        out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out


@requires_kicad10
def test_hier_sheet_pins_option_emits_hierarchical_interconnect(tmp_path):
    """With hierarchical_sheet_pins=True the KiCad hierarchical interconnect is
    COMPLETE (Tiers 1+2): each boundary net's on-sheet label is a
    ``hierarchical_label`` emitted ON the net (not a dangling fixed slot), the
    redundant ``global_label`` is gone, the parent's sheet symbols carry the
    paired sheet pins each WIRED out to a name label, the root carries no hier
    label, and ERC is error-free (label_dangling 0, pin_not_connected 0)."""
    out, text = _render(_hier_build, "hspon", tmp_path, hierarchical_sheet_pins=True)
    labels = _hier_labels(text)
    assert labels, "child hierarchical labels not emitted"
    # Boundary net B (s1<->s2) now rides a hierarchical_label, NOT a global_label.
    names = {n for n, _x, _y in labels}
    assert "B" in names, names
    assert 'global_label "B"' not in text, "redundant global_label survived for B"
    # Every hierarchical_label sits on the 1.27 mm grid (so it lands on wires/pins).
    for n, x, y in labels:
        assert abs(x / _GRID_MM - round(x / _GRID_MM)) < 1e-3, (n, x, y)
        assert abs(y / _GRID_MM - round(y / _GRID_MM)) < 1e-3, (n, x, y)
    # Parent sheet symbol carries sheet pins for boundary nets: `(pin NAME
    # bidirectional (at ...))` inside a `(sheet ...)` (the net name is unquoted).
    top = (out / "hspon.kicad_sch").read_text(encoding="utf-8")
    assert re.search(r"\(pin \S+ bidirectional", top), "no sheet pins on parent"
    # Each sheet pin is wired out to a same-named LOCAL label on the root sheet.
    assert '(label "B"' in top, "no parent-side name label for B"
    assert re.search(r"\(wire", top), "no parent-side stub wire"
    # Root sheet (parent) uses LOCAL labels for stubs, not hier labels.
    assert "(hierarchical_label" not in top
    # Interconnect complete: no dangling labels, no unconnected sheet pins.
    types = _erc_error_types(out, "hspon")
    assert types.get("label_dangling", 0) == 0, types
    assert types.get("pin_not_connected", 0) == 0, types


def _three_level(ckt):
    """root -> child (NO own parts) -> grandchild, with signal net SIGX passing
    through the intermediate child. The intermediate has no parts of its own, so
    ``get_boundary_nets()`` (which scans node.parts) misses SIGX -- only the
    descendant-closure classification gives the child's box a SIGX sheet pin and
    the child's sheet a SIGX export label. Exercises the transit-net path."""
    from skidl import Net, Part, subcircuit

    @subcircuit
    def grandchild(sig, gnd, vpos):
        u = Part("Amplifier_Operational", "OPA340NA")
        r = Part("Device", "R", value="1k",
                 footprint="Resistor_SMD:R_0603_1608Metric")
        u["5"] += vpos; u["2"] += gnd; u["3"] += sig
        u["4"] += r[1]; u["1"] += r[1]; r[2] += gnd

    @subcircuit
    def child(sig, gnd, vpos):
        # No own parts -- only the grandchild. SIGX transits this level.
        grandchild(sig, gnd, vpos)

    with ckt:
        vpos = Net("+5V"); vpos.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        sig = Net("SIGX")
        # A root-level part on SIGX makes it boundary at EVERY level.
        rt = Part("Device", "R", value="10k",
                  footprint="Resistor_SMD:R_0603_1608Metric")
        rt[1] += sig; rt[2] += gnd
        child(sig, gnd, vpos)


@requires_kicad10
def test_hier_sheet_pins_three_level_transit(tmp_path):
    """A net transiting an intermediate sheet with no own parts still wires end
    to end (descendant-closure boundary classification): ERC error-free, and the
    intermediate sheet carries BOTH the grandchild's box and a hierarchical_label
    for the transit net (its upward export)."""
    out, _text = _render(_three_level, "tri", tmp_path, hierarchical_sheet_pins=True)
    types = _erc_error_types(out, "tri")
    assert not types, f"ERC errors on 3-level transit: {types}"
    # The intermediate sheet embeds the grandchild box AND exports SIGX upward.
    intermediate = None
    for f in Path(out).glob("*.kicad_sch"):
        if f.name == "tri.kicad_sch":
            continue  # root
        txt = f.read_text(encoding="utf-8")
        if "(sheet" in txt and 'hierarchical_label "SIGX"' in txt:
            intermediate = f.name
            break
    assert intermediate, "no intermediate sheet exporting the transit net SIGX"


def _contained_partless(ckt):
    """root -> container (NO own parts) -> two grandchildren sharing net MID.

    MID is created in the container and used ONLY by its two grandchildren, so it
    is fully CONTAINED in the container's subtree -- it must stay a local label
    (pairing the two grandchild boxes' sheet pins on the container page), NOT a
    ``hierarchical_label`` exported upward (which would have no matching sheet pin
    in the root -> ``hier_label_mismatch``). INP genuinely transits root->
    container->grandchild, so it MUST still export."""
    from skidl import Net, Part, subcircuit

    @subcircuit
    def gchild(vin, vout, vpos, gnd):
        u = Part("Amplifier_Operational", "OPA340NA")
        r = Part("Device", "R", value="1k",
                 footprint="Resistor_SMD:R_0603_1608Metric")
        u["5"] += vpos; u["2"] += gnd; u["3"] += vin; u["4"] += vout; u["1"] += vout
        r[1] += vout; r[2] += gnd

    @subcircuit
    def container(inp, vpos, gnd):
        mid = Net("MID")  # shared only between the two grandchildren below
        gchild(inp, mid, vpos, gnd, tag="g1")   # produces MID
        gchild(mid, Net("DEEP"), vpos, gnd, tag="g2")  # consumes MID

    with ckt:
        vpos = Net("+5V"); vpos.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        inp = Net("INP")
        rt = Part("Device", "R", value="10k",
                  footprint="Resistor_SMD:R_0603_1608Metric")
        rt[1] += inp; rt[2] += gnd  # root part -> INP transits into the container
        container(inp, vpos, gnd)


def _contained_mixed(ckt):
    """root -> section (OWN part + a child sheet) with contained net LOC.

    LOC is produced by the child sheet and consumed by the section's OWN part, so
    it never leaves the section's subtree -> must stay a local label, NOT an
    exported ``hierarchical_label`` (the mixed-sheet variant of the bug). INP
    still transits root->section->child and must export."""
    from skidl import Net, Part, subcircuit

    @subcircuit
    def gchild(vin, vout, vpos, gnd):
        u = Part("Amplifier_Operational", "OPA340NA")
        u["5"] += vpos; u["2"] += gnd; u["3"] += vin; u["4"] += vout; u["1"] += vout

    @subcircuit
    def section(inp, vpos, gnd):
        loc = Net("LOC")
        gchild(inp, loc, vpos, gnd)  # child sheet produces LOC
        # section's OWN part consumes LOC -> LOC is contained in section's subtree
        r = Part("Device", "R", value="2k",
                 footprint="Resistor_SMD:R_0603_1608Metric")
        r[1] += loc; r[2] += gnd

    with ckt:
        vpos = Net("+5V"); vpos.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        inp = Net("INP")
        rt = Part("Device", "R", value="10k",
                  footprint="Resistor_SMD:R_0603_1608Metric")
        rt[1] += inp; rt[2] += gnd
        section(inp, vpos, gnd)


def _intermediate_sheet_text(out, top):
    """Text of the non-root sheet that itself embeds a child sheet box (the
    intermediate container/section in these tests)."""
    for f in sorted(Path(out).glob("*.kicad_sch")):
        if f.name == f"{top}.kicad_sch":
            continue  # root
        txt = f.read_text(encoding="utf-8")
        if "(sheet" in txt:  # embeds at least one child sheet box
            return txt
    return None


@requires_kicad10
def test_hier_sheet_pins_contained_net_not_exported_partless(tmp_path):
    """A net contained within a parts-less intermediate's subtree (shared only
    among its descendant sheets) is NOT exported ABOVE that intermediate: no
    ``hier_label_mismatch`` ERC error, the container sheet does not carry a
    hierarchical_label for the contained net MID, and the genuine transit net INP
    still exports. Regression for the nesting-breaks bug.

    (MID *is* a legit hierarchical_label inside the grandchild sheets -- it
    escapes each grandchild UP TO the container, pairing with the container's
    sheet pin. The bug was the container re-exporting it to the root.)"""
    out, text = _render(
        _contained_partless, "cpl", tmp_path, hierarchical_sheet_pins=True
    )
    types = _erc_error_types(out, "cpl")
    assert types.get("hier_label_mismatch", 0) == 0, types
    assert not types, f"ERC errors on contained-net nesting: {types}"
    # The container (intermediate) sheet must NOT re-export the contained net MID.
    inter = _intermediate_sheet_text(out, "cpl")
    assert inter is not None, "no intermediate sheet found"
    assert 'hierarchical_label "MID"' not in inter, "contained net MID over-exported"
    # But the genuine transit net INP must still export upward (somewhere).
    assert 'hierarchical_label "INP"' in text, "real transit net INP no longer exports"


@requires_kicad10
def test_hier_sheet_pins_contained_net_not_exported_mixed(tmp_path):
    """A net linking an intermediate sheet's OWN part to one of its CHILD sheets
    stays within the subtree -> the intermediate does not export it upward
    (mixed-sheet variant): no ``hier_label_mismatch``, the section sheet carries no
    hierarchical_label for LOC, INP still exports."""
    out, text = _render(
        _contained_mixed, "cmx", tmp_path, hierarchical_sheet_pins=True
    )
    types = _erc_error_types(out, "cmx")
    assert types.get("hier_label_mismatch", 0) == 0, types
    assert not types, f"ERC errors on mixed contained-net nesting: {types}"
    inter = _intermediate_sheet_text(out, "cmx")
    assert inter is not None, "no intermediate sheet found"
    assert 'hierarchical_label "LOC"' not in inter, "contained net LOC over-exported"
    assert 'hierarchical_label "INP"' in text, "real transit net INP no longer exports"


# ---------------------------------------------------------------------------
# On-grid wire endpoints (finding B: A* corridors quantized to the grid)
# ---------------------------------------------------------------------------

_GRID_MM = 1.27  # KiCad 50-mil grid


def _off_grid_endpoints(out):
    """Every wire endpoint across all sheets whose x or y is off the 1.27 grid."""
    bad = []
    wire_re = re.compile(
        r"\(wire\s*\(pts\s*\(xy ([\d.-]+) ([\d.-]+)\)\s*\(xy ([\d.-]+) ([\d.-]+)\)"
    )
    for f in sorted(Path(out).glob("*.kicad_sch")):
        txt = f.read_text(encoding="utf-8")
        for m in wire_re.finditer(txt):
            x1, y1, x2, y2 = map(float, m.groups())
            for x, y in ((x1, y1), (x2, y2)):
                if (
                    abs(x / _GRID_MM - round(x / _GRID_MM)) > 1e-3
                    or abs(y / _GRID_MM - round(y / _GRID_MM)) > 1e-3
                ):
                    bad.append((f.name, x, y))
    return bad


def _dense_wired(ckt):
    """A routing-dense sheet: three op-amps chained with feedback + input/load
    resistors, so the A* router must run wires through corridors between part
    bodies (where finding B's off-grid Hanan lines surfaced)."""
    from skidl import Net, Part

    with ckt:
        vpos = Net("+5V"); vpos.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        prev = Net("IN")
        for i in range(3):
            u = Part("Amplifier_Operational", "OPA340NA")
            rf = Part("Device", "R", value="100k",
                      footprint="Resistor_SMD:R_0603_1608Metric")
            rin = Part("Device", "R", value="10k",
                       footprint="Resistor_SMD:R_0603_1608Metric")
            out_net = Net(f"N{i}")
            u["5"] += vpos; u["2"] += gnd
            u["3"] += prev
            rin[1] += prev; rin[2] += gnd
            u["4"] += out_net; u["1"] += out_net
            rf[1] += out_net; rf[2] += u["3"]
            prev = out_net


@requires_kicad10
def test_wired_render_endpoints_on_grid(tmp_path):
    """Finding B guard rail: on the wired (seed_placement) path every emitted
    wire endpoint lands on the 1.27 mm grid -- the A* corridor tracks derived
    from part bbox edges are snapped to the grid, so KiCad ERC reports zero
    endpoint_off_grid."""
    out, _text = _render(_dense_wired, "grid3", tmp_path)
    bad = _off_grid_endpoints(out)
    assert not bad, f"off-grid wire endpoints: {bad[:8]}"
    # Cross-check against ERC's own detector (all severities).
    all_types = _erc_all_types(out, "grid3")
    assert all_types.get("endpoint_off_grid", 0) == 0, all_types


# ---------------------------------------------------------------------------
# Verbatim lib_symbols (finding F: embed library symbols, kill lib_symbol_mismatch)
# ---------------------------------------------------------------------------


def _multiunit(ckt):
    """A circuit with a NON-extends multi-unit op-amp (ADA4807-2ACP: two amp units
    + a power unit), a multi-graphic IC (OPA340NA) and passives -- exercises
    verbatim embedding of a body with several unit sub-symbols. A symbol that
    ``extends`` a parent (e.g. LM358 -> LM2904) deliberately keeps the regenerated
    path -- its raw subtree is the parent's -- so it is not used here."""
    from skidl import Net, Part

    with ckt:
        vpos = Net("+5V"); vpos.drive = POWER
        vneg = Net("-5V"); vneg.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        a, b, c = Net("A"), Net("B"), Net("C")
        u = Part("Amplifier_Operational", "ADA4807-2ACP")  # dual, non-extends
        u2 = Part("Amplifier_Operational", "OPA340NA")
        r1 = Part("Device", "R", value="1k",
                  footprint="Resistor_SMD:R_0603_1608Metric")
        r2 = Part("Device", "R", value="1k",
                  footprint="Resistor_SMD:R_0603_1608Metric")
        # Amp unit A, amp unit B (chained), power unit uC; disables tied high.
        u[3] += a; u[2] += b; u[1] += b; u[5] += vpos
        u[7] += b; u[8] += c; u[9] += c; u[6] += vpos
        u[10] += vpos; u[4] += vneg; u[11] += gnd
        u2["3"] += c; u2["4"] += a; u2["1"] += a; u2["5"] += vpos; u2["2"] += gnd
        r1[1] += a; r1[2] += gnd
        r2[1] += c; r2[2] += gnd


@requires_kicad10
def test_lib_symbols_embedded_verbatim(tmp_path):
    """Finding F guard rail: non-extends library symbols are embedded VERBATIM from
    the parsed library subtree (not regenerated from draw_cmds), including a
    multi-unit body (the dual ADA4807-2ACP).

    This asserts the invariant ENV-ROBUSTLY (the deliverable's live
    lib_symbol_mismatch -> 0 is shown on the SiPM hier repro; here the resolved
    library may be a bundled test_data copy, so we do NOT rely on KiCad's
    cross-install mismatch check):
      * a library-fidelity marker the regeneration path never emits proves the raw
        subtree was spliced (regeneration hardcodes ``(pin_names (offset 0))`` and
        emits no ``ki_fp_filters``; the op-amp libraries carry ``(offset 0.127)``);
      * the verbatim body -- multi-unit included -- round-trips through KiCad's own
        writer (the extends-inheritance regression produced a body that failed
        exactly this save gate)."""
    from utils.kicad_gate import KicadCliUnavailable, assert_kicad_save_ok

    out, text = _render(_multiunit, "verbatim", tmp_path)
    # Library parts embedded under their LIB:NAME id, inner units keep NAME_u_s.
    assert 'symbol "Amplifier_Operational:ADA4807-2ACP"' in text
    assert 'symbol "Amplifier_Operational:OPA340NA"' in text
    assert 'symbol "Device:R"' in text
    # Verbatim markers absent from the regenerated path (offset 0 + no fp filters).
    assert "(offset 0.127)" in text, "pin_names offset not verbatim (regenerated?)"
    assert "ki_fp_filters" in text, "library-only property missing (regenerated?)"
    # The verbatim multi-unit body must round-trip through KiCad's writer.
    try:
        for f in sorted(out.glob("*.kicad_sch")):
            assert_kicad_save_ok(f)
    except KicadCliUnavailable:
        pytest.skip("kicad-cli unavailable for save gate")
