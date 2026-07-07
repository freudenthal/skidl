# -*- coding: utf-8 -*-

"""Unit tests for the constructive seed-placement module (stage 19, Phase B).

These exercise ``skidl.schematics.seed_place`` with plain fake objects
implementing its structural interface — no KiCad symbol libraries required.
Each test maps to a row of the failure-mode table in the Stage-19 plan overview.
"""

import random

import pytest

from skidl import POWER
from skidl.geometry import BBox, Point, Tx, Vector, tx_rot_90
from skidl.schematics import seed_place as sp


# --------------------------------------------------------------------------- #
# Fakes implementing the structural interface
# --------------------------------------------------------------------------- #

class FakePin:
    def __init__(self, pt, orientation, stub=False):
        self.pt = pt
        self.orientation = orientation  # letter R/U/L/D, INWARD
        self.stub = stub
        self.part = None
        self.net = None


class FakePart:
    def __init__(self, ref, pins, bbox=None, locked=False, tx=None):
        self.ref = ref
        self.pins = pins
        for p in pins:
            p.part = self
        self.place_bbox = bbox or BBox(Point(-100, -100), Point(100, 100))
        self.orientation_locked = locked
        self.tx = tx or Tx()


class FakeNet:
    def __init__(self, name, pins, drive=None, stub=False):
        self.name = name
        self.pins = pins
        self.drive = drive
        self.stub = stub
        for p in pins:
            p.net = self


def _r_pins():
    """Two-pin passive: pin1 top (letter D, outward up), pin2 bottom (U, down)."""
    return [FakePin(Point(0, 381), "D"), FakePin(Point(0, -381), "U")]


def make_part(ref, n_pins=2, **kw):
    if n_pins == 2:
        return FakePart(ref, _r_pins(), **kw)
    pins = [FakePin(Point(0, 100 * i), "R") for i in range(n_pins)]
    return FakePart(ref, pins, **kw)


# --------------------------------------------------------------------------- #
# 1. Graph filter
# --------------------------------------------------------------------------- #

def test_graph_filter_excludes_power_and_high_fanout():
    a = make_part("R1")
    b = make_part("R2")
    c = make_part("R3")
    # Signal net between a and b (2 pins) -> included.
    sig = FakeNet("SIG", [a.pins[0], b.pins[0]])
    # GND by drive=POWER -> excluded even though name wouldn't match here.
    gnd = FakeNet("MYGND_alias", [a.pins[1], b.pins[1]], drive=POWER)
    # +3V3 by name regex -> excluded.
    rail = FakeNet("+3V3", [a.pins[0], c.pins[0]])
    # 5-pin net -> excluded with max_fanout=3.
    big_pins = [make_part(f"X{i}").pins[0] for i in range(5)]
    big = FakeNet("BUS", big_pins)
    # stubbed net -> excluded.
    stub_net = FakeNet("S", [b.pins[0], c.pins[0]], stub=True)

    adj = sp.build_wired_graph([a, b, c], [sig, gnd, rail, big, stub_net], max_fanout=3)
    assert b in adj[a] and len(adj[a][b]) == 1  # only SIG survives
    assert c not in adj[a]  # +3V3 excluded
    # GND (POWER) excluded -> a-b edge count stays 1, not 2.
    assert len(adj[a].get(b, [])) == 1


def test_graph_filter_stub_pin_excludes_net():
    a = make_part("R1")
    b = make_part("R2")
    a.pins[0].stub = True
    net = FakeNet("SIG", [a.pins[0], b.pins[0]])
    adj = sp.build_wired_graph([a, b], [net])
    assert b not in adj[a]  # a pin is stubbed -> net dropped


def test_graph_parallel_nets_yield_two_pin_pairs():
    # Rf and Cf both bridge nodes N1 and N2 -> edge Rf-Cf has 2 pin-pairs.
    p1 = make_part("U1")
    rf = make_part("RF")
    cf = make_part("CF")
    n1 = FakeNet("N1", [p1.pins[0], rf.pins[0], cf.pins[0]])
    n2 = FakeNet("N2", [p1.pins[1], rf.pins[1], cf.pins[1]])
    adj = sp.build_wired_graph([p1, rf, cf], [n1, n2], max_fanout=3)
    assert len(adj[rf][cf]) == 2
    assert len(adj[p1][rf]) == 2  # p1-rf via both nets too


# --------------------------------------------------------------------------- #
# 2. k-core
# --------------------------------------------------------------------------- #

def test_k_core_triangle_plus_tail():
    a, b, c, d = (make_part(r) for r in ("A", "B", "C", "D"))
    # Triangle A-B-C + tail C-D.
    nets = [
        FakeNet("ab", [a.pins[0], b.pins[0]]),
        FakeNet("bc", [b.pins[1], c.pins[0]]),
        FakeNet("ca", [c.pins[1], a.pins[1]]),
        FakeNet("cd", [c.pins[0], d.pins[0]]),
    ]
    adj = sp.build_wired_graph([a, b, c, d], nets)
    cores = sp.k_core(adj)
    assert cores[a] == 2 and cores[b] == 2 and cores[c] == 2
    assert cores[d] == 1


def test_k_core_chain_all_one():
    parts = [make_part(f"R{i}") for i in range(5)]
    nets = [
        FakeNet(f"n{i}", [parts[i].pins[1], parts[i + 1].pins[0]])
        for i in range(4)
    ]
    adj = sp.build_wired_graph(parts, nets)
    cores = sp.k_core(adj)
    assert set(cores.values()) == {1}


# --------------------------------------------------------------------------- #
# 3. pick_center
# --------------------------------------------------------------------------- #

def test_pick_center_hub_wins_over_passives():
    hub = make_part("U1", n_pins=6)
    sats = [make_part(f"R{i}") for i in range(6)]
    nets = [FakeNet(f"n{i}", [hub.pins[i], sats[i].pins[0]]) for i in range(6)]
    adj = sp.build_wired_graph([hub] + sats, nets)
    cores = sp.k_core(adj)
    pin_counts = {p: len(p.pins) for p in adj}
    assert sp.pick_center(adj, cores, pin_counts) is hub


def test_pick_center_deterministic_under_shuffle():
    parts = [make_part(f"R{i}") for i in range(5)]
    nets = [
        FakeNet("ab", [parts[0].pins[0], parts[1].pins[0]]),
        FakeNet("bc", [parts[1].pins[1], parts[2].pins[0]]),
        FakeNet("ca", [parts[2].pins[1], parts[0].pins[1]]),
        FakeNet("cd", [parts[2].pins[0], parts[3].pins[0]]),
        FakeNet("ce", [parts[2].pins[1], parts[4].pins[0]]),
    ]
    results = set()
    for _ in range(5):
        order = parts[:]
        random.shuffle(order)
        adj = sp.build_wired_graph(order, nets)
        cores = sp.k_core(adj)
        pin_counts = {p: len(p.pins) for p in adj}
        results.add(sp.pick_center(adj, cores, pin_counts).ref)
    assert results == {"R2"}  # the triangle+2-tail hub, deterministic


def test_pick_center_chain_picks_endpoint():
    parts = [make_part(f"R{i}") for i in range(5)]
    nets = [
        FakeNet(f"n{i}", [parts[i].pins[1], parts[i + 1].pins[0]])
        for i in range(4)
    ]
    adj = sp.build_wired_graph(parts, nets)
    cores = sp.k_core(adj)
    pin_counts = {p: len(p.pins) for p in adj}
    center = sp.pick_center(adj, cores, pin_counts)
    assert len(adj[center]) == 1  # an endpoint (degree 1)
    assert center.ref == "R0"  # smallest-ref endpoint


# --------------------------------------------------------------------------- #
# 4. outward_face — the ADA4817 table (the +180 trap)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "letter, expected",
    [
        ("R", (-1, 0)),  # -in / +in : letter R -> outward LEFT
        ("L", (1, 0)),   # OUT / FB  : letter L -> outward RIGHT
        ("D", (0, 1)),   # +Vs       : letter D -> outward UP
        ("U", (0, -1)),  # -Vs       : letter U -> outward DOWN
    ],
)
def test_outward_face_ada4817_identity(letter, expected):
    pin = FakePin(Point(0, 0), letter)
    v = sp.outward_face(pin, Tx())
    assert (v.x, v.y) == expected


def test_outward_face_rotated_90():
    # A left-facing pin (letter R) on a part rotated 90deg CCW faces down.
    pin = FakePin(Point(0, 0), "R")
    v = sp.outward_face(pin, tx_rot_90)
    assert (v.x, v.y) == (0, -1)


# --------------------------------------------------------------------------- #
# 5. grow_order — feedback pulls a bridging part forward
# --------------------------------------------------------------------------- #

def test_grow_order_bridging_part_before_leaf():
    # Triangle A-B-C (feedback) + leaf A-D. After A,B placed, C bridges 2 placed
    # parts (key -2) and must precede D which bridges only 1 (key -1).
    a, b, c, d = (make_part(r) for r in ("A", "B", "C", "D"))
    nets = [
        FakeNet("ab", [a.pins[0], b.pins[0]]),
        FakeNet("bc", [b.pins[1], c.pins[0]]),
        FakeNet("ca", [c.pins[1], a.pins[1]]),
        FakeNet("ad", [a.pins[0], d.pins[0]]),
    ]
    adj = sp.build_wired_graph([a, b, c, d], nets)
    order = [p.ref for p in sp.grow_order(adj, a)]
    assert order[0] == "A"
    assert order.index("C") < order.index("D")


# --------------------------------------------------------------------------- #
# 6. choose_slot / fan-out — same-face satellites don't stack
# --------------------------------------------------------------------------- #

def test_fanout_satellites_do_not_overlap():
    # Hub with 4 pins all facing right (letter L -> outward right); 4 satellites.
    hub_pins = [FakePin(Point(200, 150 - 100 * i), "L") for i in range(4)]
    hub = FakePart("U1", hub_pins, bbox=BBox(Point(-200, -200), Point(200, 200)))
    sats = [make_part(f"R{i}") for i in range(4)]
    nets = [FakeNet(f"n{i}", [hub.pins[i], sats[i].pins[0]]) for i in range(4)]

    sp.seed_placement([hub] + sats, nets, gap=100, grid=50)

    origins = [(s.tx.dx, s.tx.dy) for s in sats]
    assert len(set(origins)) == 4  # all distinct

    # No two satellite world bboxes overlap.
    def wbbox(s):
        return s.place_bbox * s.tx
    for i in range(len(sats)):
        for j in range(i + 1, len(sats)):
            assert not wbbox(sats[i]).intersects(wbbox(sats[j])), (i, j)


# --------------------------------------------------------------------------- #
# 7. Rotation anti-parallel
# --------------------------------------------------------------------------- #

def test_rotation_makes_pin_face_back():
    # Driver's connecting pin faces right; resistor placed to its right must be
    # rotated so its connecting pin faces LEFT (back toward the driver).
    # Name the driver so it sorts first and is chosen as the seed center.
    driver_pin_sig = FakePin(Point(200, 0), "L")  # outward right
    driver_pin_other = FakePin(Point(-200, 0), "R")
    driver = FakePart("A1", [driver_pin_sig, driver_pin_other])
    r = make_part("R1")
    net = FakeNet("SIG", [driver.pins[0], r.pins[0]])

    sp.seed_placement([driver, r], [net], gap=100, grid=50)

    # r's connecting pin (r.pins[0]) outward face in its placed frame == LEFT.
    face = sp.outward_face(r.pins[0], r.tx)
    assert (face.x, face.y) == (-1, 0)
    # And r sits to the right of the driver.
    assert r.tx.dx > driver.tx.dx


# --------------------------------------------------------------------------- #
# 8. orientation_locked -> translation only
# --------------------------------------------------------------------------- #

def test_orientation_locked_translation_only():
    # Driver named to sort first -> it's the center; the locked R1 is placed
    # via choose_slot, exercising the locked-translation-only path.
    driver = FakePart("A1", [FakePin(Point(200, 0), "L"), FakePin(Point(-200, 0), "R")])
    # Locked part starts rotated 90deg; placement must keep that linear part.
    locked = FakePart("R1", _r_pins(), locked=True, tx=tx_rot_90)
    net = FakeNet("SIG", [driver.pins[0], locked.pins[0]])

    sp.seed_placement([driver, locked], [net], gap=100, grid=50)

    assert (locked.tx.a, locked.tx.b, locked.tx.c, locked.tx.d) == (
        tx_rot_90.a, tx_rot_90.b, tx_rot_90.c, tx_rot_90.d
    )


# --------------------------------------------------------------------------- #
# 9. Determinism + RNG untouched
# --------------------------------------------------------------------------- #

def _build_tia_like():
    u1 = FakePart("U1", [FakePin(Point(-200, 50), "R"), FakePin(Point(200, 0), "L")])
    rf = make_part("RF")
    cf = make_part("CF")
    rin = make_part("RIN")
    inv = FakeNet("INV", [u1.pins[0], rf.pins[0], cf.pins[0]])
    out = FakeNet("OUT", [u1.pins[1], rf.pins[1], cf.pins[1]])
    src = FakeNet("SIG", [rin.pins[0], u1.pins[0]])  # rin also into inv? keep 2-pin
    # keep INV at 3 pins; source is a separate 2-pin net to rin
    src2 = FakeNet("SRC", [rin.pins[1], cf.pins[0]])
    return [u1, rf, cf, rin], [inv, out, src2]


def test_determinism_identical_placements():
    parts1, nets1 = _build_tia_like()
    sp.seed_placement(parts1, nets1)
    tx1 = {p.ref: (p.tx.a, p.tx.b, p.tx.c, p.tx.d, p.tx.dx, p.tx.dy) for p in parts1}

    parts2, nets2 = _build_tia_like()
    sp.seed_placement(parts2, nets2)
    tx2 = {p.ref: (p.tx.a, p.tx.b, p.tx.c, p.tx.d, p.tx.dx, p.tx.dy) for p in parts2}

    assert tx1 == tx2


def test_rng_state_untouched():
    random.seed(12345)
    before = random.getstate()
    parts, nets = _build_tia_like()
    sp.seed_placement(parts, nets)
    after = random.getstate()
    assert before == after


# --------------------------------------------------------------------------- #
# 10. Centroid — a part between two placed neighbours
# --------------------------------------------------------------------------- #

def test_centroid_between_two_placed():
    # A placed left, B placed right, C connects to both -> lands between them.
    a = make_part("A")
    b = make_part("B")
    c = make_part("C")
    na = FakeNet("na", [a.pins[0], c.pins[0]])
    nb = FakeNet("nb", [b.pins[0], c.pins[1]])
    adj = sp.build_wired_graph([a, b, c], [na, nb])

    placed_info = {
        a: {"origin": Point(-1000, 0), "tx": Tx().move(Point(-1000, 0)),
            "wbbox": a.place_bbox * Tx().move(Point(-1000, 0))},
        b: {"origin": Point(1000, 0), "tx": Tx().move(Point(1000, 0)),
            "wbbox": b.place_bbox * Tx().move(Point(1000, 0))},
    }
    origin, _rot = sp.choose_slot(c, placed_info, adj, gap=100, grid=50)
    assert -1000 <= origin.x <= 1000
    assert abs(origin.x) < 1000  # strictly between, pulled in-ish
