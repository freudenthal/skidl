# -*- coding: utf-8 -*-

"""Render occupancy / constructive-relaxation plan -- Phase 0 repro + canary.

Locks the 2026-07-12 layout-regression investigation as permanent fixtures:

  * ``sheet_cell_owners(sch_path)`` -- the cell-scan assertion primitive shared by
    every later phase's gate. It reads an emitted ``.kicad_sch`` and returns
    ``{(x, y) -> set(net names)}`` faithful to KiCad's fusion semantics (net-name
    anchors = label/power-symbol positions; wires electrically merge the cells
    they touch). A cell whose wire-connected blob carries >1 distinct net name is
    a cross-net fusion -- exactly what ``_audit_sheet_connectivity`` warns about
    in-memory.

  * a small *packed* canary modelled on the HV supply's ``linear_postreg`` sheet
    (op-amp + pot + reference + pass FET + dividers), which is what produced the
    ``cell ... shared by nets ['GND', 'VREF10']`` fusion.

The canary is rendered two ways: with the force-directed refiner ON (today's
default -- must stay fusion-free, a regression guard for every later phase) and
with the refiner monkeypatched OFF (constructive seed only). Refiner-OFF fuses
GND vs VREF10 today, so that assertion is ``xfail`` -- Phase 2 (power/stub cells
in the unified occupancy registry) closes the hole and flips it to PASS
independent of spacing.

Needs a real KiCad-10 install (kicad-cli + stock symbols); skips otherwise.
"""

import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from skidl import KICAD10, POWER, lib_search_paths, set_default_tool

# ---- real-KiCad-10 discovery (mirrors test_power_symbol_rendering.py) ------
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


# ===========================================================================
# sheet_cell_owners -- the cell-scan assertion primitive (file reader)
# ===========================================================================
def _tokenize(text):
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in "()":
            out.append(c)
            i += 1
        elif c == '"':
            j, buf = i + 1, []
            while j < n:
                if text[j] == "\\":
                    buf.append(text[j + 1])
                    j += 2
                    continue
                if text[j] == '"':
                    break
                buf.append(text[j])
                j += 1
            out.append(("str", "".join(buf)))
            i = j + 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '()"':
                j += 1
            out.append(("atom", text[i:j]))
            i = j
    return out


def _parse(tokens):
    pos = 0

    def rd():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst = []
            while tokens[pos] != ")":
                lst.append(rd())
            pos += 1
            return lst
        return tok

    forms = []
    while pos < len(tokens):
        forms.append(rd())
    return forms


def _val(node):
    return node[1] if isinstance(node, tuple) else node


def _find(lst, key):
    for e in lst:
        if isinstance(e, list) and e and _val(e[0]) == key:
            return e
    return None


def sheet_cell_owners(sch_path):
    """Map every connection cell -> the set of net names electrically on it.

    Anchors (label / power-symbol positions) name nets; wires merge the cells
    they touch (KiCad's fusion rule). Returns ``{(x, y) -> set(net names)}``; a
    cell whose value has >1 name is a cross-net fusion.
    """
    text = Path(sch_path).read_text(encoding="utf-8")
    root = _parse(_tokenize(text))[0]

    def cell(x, y):
        return (round(float(x), 2), round(float(y), 2))

    anchors = defaultdict(set)  # cell -> {net names}
    edges = []  # (cellA, cellB) wire segments

    for e in root:
        if not isinstance(e, list) or not e:
            continue
        tag = _val(e[0])
        if tag in ("label", "global_label", "hierarchical_label"):
            name = _val(e[1])
            at = _find(e, "at")
            if at and isinstance(name, str):
                anchors[cell(_val(at[1]), _val(at[2]))].add(name)
        elif tag == "symbol":
            libid = _find(e, "lib_id")
            if not libid:
                continue
            lid = str(_val(libid[1]))
            # power:PWR_FLAG is deliberately coincident with a rail (that is how
            # it drives the net); its value is not a net name -> exclude it.
            if not lid.startswith("power:") or lid == "power:PWR_FLAG":
                continue
            net = lid.split(":", 1)[1]
            at = _find(e, "at")
            if at:
                anchors[cell(_val(at[1]), _val(at[2]))].add(net)
        elif tag == "wire":
            pts = _find(e, "pts")
            if not pts:
                continue
            xys = [s for s in pts if isinstance(s, list) and s and _val(s[0]) == "xy"]
            for a, b in zip(xys, xys[1:]):
                edges.append(
                    (cell(_val(a[1]), _val(a[2])), cell(_val(b[1]), _val(b[2])))
                )

    # Union-find over all cells (anchors + wire endpoints).
    parent = {}

    def find(c):
        parent.setdefault(c, c)
        r = c
        while parent[r] != r:
            r = parent[r]
        while parent[c] != r:
            parent[c], c = r, parent[c]
        return r

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in anchors:
        find(c)
    for a, b in edges:
        find(a)
        find(b)
        union(a, b)

    comp_nets = defaultdict(set)
    for c, nets in anchors.items():
        comp_nets[find(c)] |= nets
    owners = {}
    for c in parent:
        owners[c] = set(comp_nets.get(find(c), set()))
    return owners


def sheet_fusions(sch_dir):
    """All cross-net fusions in every ``.kicad_sch`` under ``sch_dir``."""
    out = {}
    for p in Path(sch_dir).rglob("*.kicad_sch"):
        fused = {c: sorted(ns) for c, ns in sheet_cell_owners(p).items() if len(ns) > 1}
        if fused:
            out[p.name] = fused
    return out


# ===========================================================================
# Packed canary + render harness
# ===========================================================================
FP_R = "Resistor_SMD:R_0805_2012Metric"
FP_C = "Capacitor_SMD:C_0805_2012Metric"


def _use_kicad10():
    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _build_packed_canary():
    """Dense mixed power+signal sheet modelled on the HV ``linear_postreg`` stage.

    A 10 V pot-set reference feeding an op-amp error amp that drives an IRF740
    source-follower pass FET, with an output sense divider and filter caps. The
    GND (power) and VREF10 (signal) nets sit on adjacent parts -- the pair that
    fuses when the constructive seed's arrangement is not spaced by the refiner.
    """
    from skidl import Circuit, Net, Part

    ckt = Circuit(name="postreg_canary")
    with ckt:
        vin = Net("VIN")
        vin.drive = POWER
        rail = Net("RAIL")
        rail.drive = POWER
        gnd = Net("GND")
        gnd.drive = POWER
        vout = Net("VOUT")
        VREF = Net("VREF10")
        VSET = Net("VSET")
        GATE = Net("GATE_P")
        FB = Net("FB_OUT")

        rref = Part("Device", "R", ref="R7", value="470", footprint=FP_R)
        vin += rref[1]
        VREF += rref[2]
        dref = Part("Reference_Voltage", "LM4040DBZ-10", ref="D2")
        VREF += dref[1]
        gnd += dref[2]
        cref = Part("Device", "C", ref="C5", value="100nF", footprint=FP_C)
        VREF += cref[1]
        gnd += cref[2]
        pot = Part("Device", "R_Potentiometer", ref="RV1", value="10k")
        VREF += pot[1]
        VSET += pot[2]
        gnd += pot[3]
        u1 = Part("Amplifier_Operational", "MCP6001R", ref="U1")
        u1[1] += GATE
        u1[3] += VSET
        u1[4] += FB
        u1[2] += rail
        u1[5] += gnd
        rg = Part("Device", "R", ref="R8", value="100", footprint=FP_R)
        u1[1] += rg[1]
        rg[2] += GATE
        q2 = Part("Transistor_FET", "IRF740", ref="Q3", value="IRF740")
        q2[2] += rail
        q2[1] += GATE
        q2[3] += vout
        rd1 = Part("Device", "R", ref="R9", value="190k", footprint=FP_R)
        rd2 = Part("Device", "R", ref="R10", value="10k", footprint=FP_R)
        vout += rd1[1]
        FB += rd1[2], rd2[1]
        gnd += rd2[2]
        cout1 = Part("Device", "C", ref="C6", value="10uF", footprint=FP_C)
        vout += cout1[1]
        gnd += cout1[2]
        cout2 = Part("Device", "C", ref="C7", value="100nF", footprint=FP_C)
        vout += cout2[1]
        gnd += cout2[2]
    return ckt


# The deliverable's default render_opts (skidl-eda project.py): constructive
# seed + deconflict-stub wiring + power symbols one grid off the pin.
_RENDER_OPTS = dict(
    seed_placement=True,
    auto_stub=False,
    deconflict_stubs=True,
    power_stubs=True,
)


def _render_and_scan(refiner_on, monkeypatch=None):
    """Render the packed canary and return its cross-net fusions.

    ``refiner_on=False`` monkeypatches ``push_and_pull`` to a no-op (the
    constructive seed alone; the canonical refiner-off probe). Uses a
    ``monkeypatch`` fixture when given, else patches/restores by hand.
    """
    import skidl.schematics.place as place_mod

    _use_kicad10()
    ckt = _build_packed_canary()

    saved = place_mod.push_and_pull
    if not refiner_on:
        if monkeypatch is not None:
            monkeypatch.setattr(place_mod, "push_and_pull", lambda *a, **k: None)
        else:
            place_mod.push_and_pull = lambda *a, **k: None
    d = tempfile.mkdtemp(prefix="skidl_occ_")
    try:
        ckt.generate_schematic(filepath=d, top_name="postreg_canary", **_RENDER_OPTS)
        return sheet_fusions(d)
    finally:
        if monkeypatch is None and not refiner_on:
            place_mod.push_and_pull = saved
        shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# Tests
# ===========================================================================
@requires_kicad10
def test_cell_scan_primitive_reads_a_render():
    """sheet_cell_owners returns a non-empty cell map for a real render (the
    later phases assert against it, so a broken reader must fail loudly here)."""
    _use_kicad10()
    ckt = _build_packed_canary()
    d = tempfile.mkdtemp(prefix="skidl_occ_")
    try:
        ckt.generate_schematic(filepath=d, top_name="postreg_canary", **_RENDER_OPTS)
        p = next(Path(d).rglob("*.kicad_sch"))
        owners = sheet_cell_owners(p)
        assert owners, "cell-scan primitive found no connection cells"
        # Every named cell carries at least one net; GND (power) must appear.
        all_nets = set().union(*owners.values())
        assert "GND" in all_nets
    finally:
        shutil.rmtree(d, ignore_errors=True)


@requires_kicad10
def test_refiner_on_no_fusion():
    """Today's PASSING behavior: with the force-directed refiner ON, the
    constructive+deconflict render is fusion-free. This is the regression guard
    every later phase must keep green."""
    fusions = _render_and_scan(refiner_on=True)
    assert fusions == {}, f"refiner-ON render fused nets: {fusions}"


@requires_kicad10
@pytest.mark.xfail(
    strict=True,
    reason="Structural hole (root causes #1-#3): power-symbol/stub cells are "
    "decided at EMIT time and never enter the router's deconflict occupancy, so "
    "the constructive seed alone lets a GND power symbol land on a VREF10 stub -> "
    "KiCad fuses them. The force-directed refiner only accidentally masks this by "
    "spreading parts out. Phase 2 (unified SheetOccupancy: power + stub segments) "
    "closes it independent of spacing, at which point this flips to PASS.",
)
def test_refiner_off_no_fusion():
    """With the refiner OFF (constructive seed only) the render must NOT fuse
    nets. It does today (GND/VREF10), so this is xfail until Phase 2."""
    fusions = _render_and_scan(refiner_on=False)
    assert fusions == {}, f"refiner-OFF render fused nets: {fusions}"
