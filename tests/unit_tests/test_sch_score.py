# -*- coding: utf-8 -*-

"""Unit tests for the pure schematic placement scorer (``sch_score``).

WS1 of the placement-scoring cross-pollination plan: the crossing/HPWL scorer
ported from ``skidl-layout``. These tests exercise the pure geometry (literal
segment lists + fake part/net objects, no KiCad needed) and add one end-to-end
determinism check that needs real KiCad-10 symbols (skips otherwise).
"""

import os
import random

import pytest

from skidl.schematics import sch_score
from skidl.geometry import Tx


# ---- real-KiCad-10 discovery (mirrors test_render_determinism.py) ----------
def _kicad10_symbols_available():
    import skidl.tools.kicad10.lib as k10

    return bool(k10._discover_default_symbol_dirs("10")) or bool(
        os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    )


requires_kicad10 = pytest.mark.skipif(
    not _kicad10_symbols_available(),
    reason="requires real KiCad 10 stock symbol libraries",
)


# ---- fake structural objects (no tool backend needed) ----------------------
class _FakePart:
    """Minimal part-like: a ref and a tx origin (no placed pins => the scorer
    falls back to the tx origin for the representative point)."""

    def __init__(self, ref, x, y):
        self.ref = ref
        self.num = 0
        self.pins = []
        self.tx = Tx(dx=x, dy=y)


class _FakePin:
    def __init__(self, part):
        self.part = part


class _FakeNet:
    def __init__(self, name, parts):
        self.name = name
        self.pins = [_FakePin(p) for p in parts]


class _FakeNode:
    def __init__(self, parts, nets):
        self.parts = parts
        self._nets = nets
        self.children = {}

    def get_internal_nets(self):
        return self._nets


# ===========================================================================
# Pure crossing geometry
# ===========================================================================
def test_crossings_zero_on_trivial():
    """Two parts on one net => a single star segment => 0 crossings."""
    a = _FakePart("A", 0.0, 0.0)
    b = _FakePart("B", 10.0, 0.0)
    node = _FakeNode([a, b], [_FakeNet("N", [a, b])])
    assert sch_score.estimate_crossings(node) == 0


def test_crossings_detects_a_known_X():
    """Four parts at unit-square corners; two nets whose star segments are the
    two diagonals => exactly one crossing."""
    a = _FakePart("A", 0.0, 0.0)
    b = _FakePart("B", 1.0, 1.0)
    c = _FakePart("C", 0.0, 1.0)
    d = _FakePart("D", 1.0, 0.0)
    node = _FakeNode(
        [a, b, c, d],
        [_FakeNet("diag1", [a, b]), _FakeNet("diag2", [c, d])],
    )
    assert sch_score.estimate_crossings(node) == 1


def test_count_segment_crossings_literal():
    """Direct check on literal segment lists (ref-sharing pairs are skipped)."""
    # Clean proper crossing.
    segs = [
        ("A", "B", (0.0, 0.0), (2.0, 2.0)),
        ("C", "D", (0.0, 2.0), (2.0, 0.0)),
    ]
    assert sch_score._count_segment_crossings(segs) == 1
    # Shared ref => skipped even though geometry crosses.
    segs_shared = [
        ("A", "B", (0.0, 0.0), (2.0, 2.0)),
        ("A", "C", (0.0, 2.0), (2.0, 0.0)),
    ]
    assert sch_score._count_segment_crossings(segs_shared) == 0
    # Non-crossing (parallel).
    segs_par = [
        ("A", "B", (0.0, 0.0), (2.0, 0.0)),
        ("C", "D", (0.0, 1.0), (2.0, 1.0)),
    ]
    assert sch_score._count_segment_crossings(segs_par) == 0


def test_numpy_matches_loop():
    """The vectorized crossing count is bit-exact vs the scalar loop across many
    random segment sets that include shared endpoints, collinear/touching, and
    zero-length degenerate cases (ported from layout's
    test_vectorized_crossings_matches_reference)."""
    rng = random.Random(42)
    refs = ["U1", "R1", "R2", "C1", "C2", "J1", "Q1", "D1"]
    coords = [0.0, 1.0, 2.0, 3.0]
    for trial in range(200):
        # At least 50 so the numpy path is the one production would take.
        n = rng.randint(50, 80)
        segments = []
        for _ in range(n):
            a_ref = rng.choice(refs)
            b_ref = rng.choice(refs)
            p1 = (rng.choice(coords), rng.choice(coords))
            p2 = p1 if rng.random() < 0.1 else (rng.choice(coords), rng.choice(coords))
            segments.append((a_ref, b_ref, p1, p2))
        loop = sch_score._count_segment_crossings_loop(segments)
        vec = sch_score._count_segment_crossings_numpy(segments)
        assert vec is not None, "numpy expected in test env"
        assert vec == loop, (trial, n, loop, vec)


# ===========================================================================
# score_node aggregation + purity
# ===========================================================================
def test_score_node_dict_shape_and_hierarchy():
    """score_node returns the documented dict and SUMS child scores."""
    a = _FakePart("A", 0.0, 0.0)
    b = _FakePart("B", 1.0, 1.0)
    c = _FakePart("C", 0.0, 1.0)
    d = _FakePart("D", 1.0, 0.0)
    child = _FakeNode(
        [a, b, c, d],
        [_FakeNet("diag1", [a, b]), _FakeNet("diag2", [c, d])],
    )
    e = _FakePart("E", 5.0, 5.0)
    f = _FakePart("F", 6.0, 5.0)
    root = _FakeNode([e, f], [_FakeNet("NN", [e, f])])
    root.children = {"child": child}

    score = sch_score.score_node(root)
    assert set(score) == {"crossings", "hpwl", "parts", "nets", "skipped"}
    # child contributes 1 crossing + 2 scored nets + 4 parts; root adds 0/1/2.
    assert score["crossings"] == 1
    assert score["parts"] == 6
    assert score["nets"] == 3
    assert score["hpwl"] > 0.0


def test_pure_no_tool_imports():
    """sch_score must be importable without any skidl.tools backend loaded."""
    import sys

    tool_mods = [m for m in sys.modules if m.startswith("skidl.tools")]
    # It is fine if tools are already imported by the session; the real property
    # is that importing sch_score does not REQUIRE a tool backend. Re-import it
    # in a way that would fail if it reached into skidl.tools at module scope.
    import importlib

    mod = importlib.reload(sch_score)
    assert hasattr(mod, "score_node")
    assert hasattr(mod, "estimate_crossings")
    assert hasattr(mod, "total_hpwl")
    # Sanity: the module's own globals name no skidl.tools symbol.
    assert not any(
        getattr(v, "__module__", "").startswith("skidl.tools")
        for v in vars(mod).values()
    )
    del tool_mods


# ===========================================================================
# End-to-end determinism on a generated schematic (needs real KiCad-10)
# ===========================================================================
def _build_divider(circuit):
    from skidl import Net, Part

    with circuit:
        r1 = Part("Device", "R", value="10K")
        r2 = Part("Device", "R", value="10K")
        r3 = Part("Device", "R", value="1K")
        vin, mid, vout, gnd = Net("VIN"), Net("MID"), Net("VOUT"), Net("GND")
        vin += r1[1]
        r1[2] += mid
        mid += r2[1]
        r2[2] += vout
        vout += r3[1]
        r3[2] += gnd


def _placed_node(top):
    from skidl import Circuit, get_default_tool
    from skidl.tools import tool_modules
    from skidl.tools.kicad10.gen_schematic import preprocess_circuit
    from skidl.schematics.sch_node import SchNode

    circuit = Circuit(name=top)
    _build_divider(circuit)
    opts = dict(seed_placement=True, auto_stub=False, expansion_factor=1.0)
    preprocess_circuit(circuit, **opts)
    node = SchNode(circuit, tool_modules[get_default_tool()], ".", top, top, 0.0)
    node.place(**opts)
    return node


@requires_kicad10
def test_score_node_deterministic():
    """Scoring the same placed node twice is identical, and two independent
    builds of the same circuit produce the same crossings/HPWL (placement is
    deterministic, so the score must be too)."""
    from skidl import KICAD10, set_default_tool

    set_default_tool(KICAD10)
    node_a = _placed_node("scoreA")
    s1 = sch_score.score_node(node_a)
    s2 = sch_score.score_node(node_a)
    assert s1 == s2

    node_b = _placed_node("scoreB")
    s3 = sch_score.score_node(node_b)
    assert s1["crossings"] == s3["crossings"]
    assert s1["hpwl"] == s3["hpwl"]
    assert s1["parts"] == s3["parts"] == 3
