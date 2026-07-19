# -*- coding: utf-8 -*-

"""Tests for the opt-in ``place_score_refine`` swap-polish pass (round-2 WS1).

After a connected group is placed (default pass OR the round-1
``place_score_select`` bake-off), ``place_score_refine`` optionally runs a
deterministic accept-if-better SAME-CLASS pairwise swap loop gated by
``sch_score`` (ported from skidl-layout's ``refine_placement``). It defaults OFF,
so the load-bearing safety property is byte-identity when the flag is unset.

The pure-geometry tests (fake parts, no KiCad) prove the swap mechanism and its
overlap-safety invariant; the two end-to-end tests need real KiCad-10 symbols and
skip otherwise.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from skidl.geometry import Tx, Point, BBox
from skidl.schematics import sch_score
from skidl.schematics.place import _score_swap_refine, _score_move_refine
from skidl.schematics.route import GRID  # stable 50 fallback == the injected value


# ---------------------------------------------------------------------------
# Pure-geometry fakes (no tool backend needed): a same-class part carries a
# shared place_bbox, an identity tx orientation submatrix (dx/dy translate it),
# and one pin at the local origin so sch_score's representative point is the tx
# origin (dx, dy).
# ---------------------------------------------------------------------------
class _FakePin:
    def __init__(self, part):
        self.part = part
        self.place_pt = Point(0.0, 0.0)


class _FakePart:
    def __init__(self, ref, x, y, w=2.0, h=2.0):
        self.ref = ref
        self.num = 0
        # Default: shared local bbox + identity orientation => all such parts are
        # one swap class; only the tx translation (dx, dy) differs. Passing a
        # distinct (w, h) makes a part its own swap class (used to force a
        # move-only, swap-impossible scenario).
        self.place_bbox = BBox(Point(0.0, 0.0), Point(w, h))
        self.tx = Tx(dx=x, dy=y)
        self.pins = [_FakePin(self)]


class _FakeNet:
    def __init__(self, name, parts):
        self.name = name
        self.pins = [p.pins[0] for p in parts]


def _known_X():
    """Four same-class parts at unit-square corners; two nets whose star
    segments are the crossing diagonals (1 crossing)."""
    a = _FakePart("A", 0.0, 0.0)
    b = _FakePart("B", 1.0, 1.0)
    c = _FakePart("C", 0.0, 1.0)
    d = _FakePart("D", 1.0, 0.0)
    nets = [_FakeNet("n1", [a, b]), _FakeNet("n2", [c, d])]
    return [a, b, c, d], nets


def _world_bboxes(parts):
    out = []
    for p in parts:
        wb = p.place_bbox * p.tx
        out.append((wb.min.x, wb.min.y, wb.max.x, wb.max.y))
    return sorted(out)


def test_refine_swap_uncrosses_known_X():
    """A same-class swap uncrosses the known X: crossings 1 -> 0, >=1 accepted."""
    parts, nets = _known_X()
    before = sch_score.score_parts(parts, nets)
    assert before["crossings"] == 1
    accepted = _score_swap_refine(parts, nets)
    assert accepted >= 1
    after = sch_score.score_parts(parts, nets)
    assert after["crossings"] == 0


def test_refine_preserves_world_bboxes():
    """Overlap-safety witness: the MULTISET of world bounding boxes is unchanged
    by the swap pass (a same-class swap only exchanges translations)."""
    parts, nets = _known_X()
    before = _world_bboxes(parts)
    _score_swap_refine(parts, nets)
    after = _world_bboxes(parts)
    assert before == after


def test_refine_never_worse():
    """The score after the pass is <= the score before (lexicographic)."""
    parts, nets = _known_X()
    before = sch_score.score_parts(parts, nets)
    _score_swap_refine(parts, nets)
    after = sch_score.score_parts(parts, nets)
    assert (after["crossings"], after["hpwl"]) <= (
        before["crossings"],
        before["hpwl"],
    )


def test_refine_no_swap_when_already_optimal():
    """Already-uncrossed layout => 0 accepted swaps, positions bit-unchanged."""
    a = _FakePart("A", 0.0, 0.0)
    b = _FakePart("B", 0.0, 1.0)
    c = _FakePart("C", 5.0, 0.0)
    d = _FakePart("D", 5.0, 1.0)
    nets = [_FakeNet("n1", [a, b]), _FakeNet("n2", [c, d])]
    parts = [a, b, c, d]
    before = _world_bboxes(parts)
    accepted = _score_swap_refine(parts, nets)
    assert accepted == 0
    assert _world_bboxes(parts) == before


def test_refine_different_classes_never_swapped():
    """Parts of different swap classes (different place_bbox) are never
    exchanged even when a swap would help geometrically."""
    parts, nets = _known_X()
    # Make C a different class: distinct place_bbox.
    parts[2].place_bbox = BBox(Point(0.0, 0.0), Point(3.0, 3.0))
    before = _world_bboxes(parts)
    _score_swap_refine(parts, nets)
    # C's world bbox (unique dims) must still be present unchanged; and since C
    # can pair with no one, any accepted swap only exchanges A/B/D translations.
    after = _world_bboxes(parts)
    assert (0.0, 1.0, 3.0, 4.0) in after  # C's box at its original (0,1) origin
    assert sorted(before) == sorted(after)


# ===========================================================================
# Single-part move trials (round-3 WS2): overlap-primary accept key
# ===========================================================================
def _move_fixture():
    """Two nets crossing at (50, 0); moving A or B one grid onto the other net's
    line uncrosses them with NO new overlap (parts are >=1 grid apart). Parts
    carry DIFFERENT bbox sizes => distinct swap classes, so this is genuinely
    move territory (a same-class swap can never arise)."""
    a = _FakePart("A", 0.0, 0.0, w=2.0)
    b = _FakePart("B", 2 * GRID, 0.0, w=3.0)     # (100, 0)
    c = _FakePart("C", GRID, -GRID, w=4.0)       # (50, -50)
    d = _FakePart("D", GRID, GRID, w=5.0)        # (50, 50)
    nets = [_FakeNet("n1", [a, b]), _FakeNet("n2", [c, d])]
    return [a, b, c, d], nets


def test_move_refine_fixes_offset_crossing():
    """A single grid-multiple move uncrosses two nets no swap could fix:
    crossings 1 -> 0, >=1 accepted, overlap never increased."""
    parts, nets = _move_fixture()
    before = sch_score.score_parts(parts, nets)
    assert before["crossings"] == 1
    accepted = _score_move_refine(parts, nets)
    assert accepted >= 1
    after = sch_score.score_parts(parts, nets)
    assert after["crossings"] == 0
    assert after["overlap"] <= before["overlap"]


def test_move_refine_never_increases_overlap():
    """Start with two connected parts fully overlapping; the overlap-primary
    accept key means the pass can only keep or reduce total overlap area."""
    a = _FakePart("A", 0.0, 0.0)
    b = _FakePart("B", 0.0, 0.0)          # exactly overlaps A
    c = _FakePart("C", 2 * GRID, 0.0)
    parts = [a, b, c]
    nets = [_FakeNet("n1", [a, b]), _FakeNet("n2", [b, c])]
    before = sch_score.score_parts(parts, nets)
    assert before["overlap"] > 0.0
    _score_move_refine(parts, nets)
    after = sch_score.score_parts(parts, nets)
    assert after["overlap"] <= before["overlap"]


def test_move_refine_stays_on_grid():
    """Every part's net displacement is an integer multiple of GRID."""
    parts, nets = _move_fixture()
    orig = {p.ref: (p.tx.dx, p.tx.dy) for p in parts}
    _score_move_refine(parts, nets)
    for p in parts:
        assert (p.tx.dx - orig[p.ref][0]) % GRID == 0
        assert (p.tx.dy - orig[p.ref][1]) % GRID == 0


def test_move_refine_deterministic():
    """Two runs on identical fresh fixtures land every part identically."""
    p1, n1 = _move_fixture()
    _score_move_refine(p1, n1)
    p2, n2 = _move_fixture()
    _score_move_refine(p2, n2)
    assert [(p.tx.dx, p.tx.dy) for p in p1] == [(p.tx.dx, p.tx.dy) for p in p2]


# ===========================================================================
# End-to-end: flag OFF byte-identity + flag ON determinism (need real KiCad-10)
# ===========================================================================
def _kicad10_symbols_available():
    import skidl.tools.kicad10.lib as k10

    return bool(k10._discover_default_symbol_dirs("10")) or bool(
        os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    )


requires_kicad10 = pytest.mark.skipif(
    not _kicad10_symbols_available(),
    reason="requires real KiCad 10 stock symbol libraries",
)


def _build_flat(circuit):
    """Op-amp TIA + bias divider (7 real parts), below _ROW_PLACE_THRESHOLD so
    it takes the force path where the refine hook lives."""
    from skidl import Net, Part, POWER

    with circuit:
        u1 = Part("Amplifier_Operational", "OPA340NA")
        rf = Part("Device", "R", value="1M"); cf = Part("Device", "C", value="2p")
        rin = Part("Device", "R", value="50"); rl = Part("Device", "R", value="1k")
        rb1 = Part("Device", "R", value="10k"); rb2 = Part("Device", "R", value="10k")
        cb = Part("Device", "C", value="100n")
        gnd, vplus, sig, bias = Net("GND"), Net("+5V"), Net("SIG"), Net("BIAS")
        gnd.drive = POWER; vplus.drive = POWER
        u1["4"] += rf[1], cf[1], rin[2]
        u1["1"] += rf[2], cf[2], rl[1]
        u1["3"] += bias; u1["2"] += gnd; u1["5"] += vplus
        rb1[1] += vplus; rb1[2] += bias; rb2[1] += bias; rb2[2] += gnd
        cb[1] += bias; cb[2] += gnd; rin[1] += sig; rl[2] += gnd


def _gen_sheets(top, **opts):
    from skidl import Circuit

    d = tempfile.mkdtemp(prefix="skidl_psr_")
    try:
        c = Circuit(name=top)
        _build_flat(c)
        c.generate_schematic(
            filepath=d, top_name=top, seed_placement=True, auto_stub=False, **opts
        )
        out = {}
        for f in sorted(Path(d).glob("*.kicad_sch")):
            txt = f.read_text(encoding="utf-8")
            txt = re.sub(r'\(date "[^"]*"\)', '(date "X")', txt)
            out[f.name] = txt
        assert out, "no schematic produced"
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)


@requires_kicad10
def test_refine_flag_off_is_byte_identical():
    """Flag absent vs explicitly False => byte-identical (default path
    untouched by the presence of the feature). Same top_name both runs."""
    absent = _gen_sheets("psr_off")
    false = _gen_sheets("psr_off", place_score_refine=False)
    assert set(absent) == set(false)
    for name in absent:
        assert absent[name] == false[name], f"{name} differs with flag off"


@requires_kicad10
def test_refine_flag_on_is_deterministic():
    """Two runs with place_score_refine=True are identical, and BOTH flags True
    together are also deterministic (no trial state leaks)."""
    a = _gen_sheets("psr_det", place_score_refine=True)
    b = _gen_sheets("psr_det", place_score_refine=True)
    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], f"{name} differs between two refine-on runs"

    ab1 = _gen_sheets("psr_both", place_score_select=True, place_score_refine=True)
    ab2 = _gen_sheets("psr_both", place_score_select=True, place_score_refine=True)
    for name in ab1:
        assert ab1[name] == ab2[name], f"{name} differs with both flags on"
