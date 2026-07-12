# -*- coding: utf-8 -*-

"""Unit tests for constructive relaxation placement (render-occupancy Phase 3).

``relax_placement`` is the deterministic spacing pass that replaces the
force-directed refiner in deconflict mode. It wraps ``seed_placement`` with a
wider relaxation ``gap`` so parts get more breathing room WHILE the directional
slot choices stay identical (arrangement preserved). These tests exercise it with
plain fakes (no KiCad libraries), mirroring ``test_seed_place.py``.
"""

from skidl.geometry import BBox, Point, Tx
from skidl.schematics import relax_place as rp


# Fakes implementing the structural interface (mirrors test_seed_place.py).
class FakePin:
    def __init__(self, pt, orientation, stub=False):
        self.pt = pt
        self.orientation = orientation
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


def make_part(ref, **kw):
    pins = [FakePin(Point(0, 381), "D"), FakePin(Point(0, -381), "U")]
    return FakePart(ref, pins, **kw)


def _origin(part):
    """Translation (origin) of a part's tx."""
    return (round(part.tx.dx, 3), round(part.tx.dy, 3))


def _chain():
    """A 4-part chain R1-R2-R3-R4 wired pin0->pin0 in sequence."""
    parts = [make_part(f"R{i+1}") for i in range(4)]
    nets = [
        FakeNet(f"N{i}", [parts[i].pins[0], parts[i + 1].pins[0]])
        for i in range(3)
    ]
    return parts, nets


def test_relax_places_all_parts_without_overlap():
    parts, nets = _chain()
    rp.relax_placement(parts, nets, grid=50)
    boxes = [p.place_bbox * p.tx for p in parts]
    # No two part bboxes overlap after relaxation (seed's collision search).
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].intersects(boxes[j]), f"R{i+1}/R{j+1} overlap"


def test_relax_is_deterministic():
    p1, n1 = _chain()
    rp.relax_placement(p1, n1, grid=50)
    o1 = [_origin(p) for p in p1]
    p2, n2 = _chain()
    rp.relax_placement(p2, n2, grid=50)
    o2 = [_origin(p) for p in p2]
    assert o1 == o2, "relax_placement is not deterministic"


def test_relax_preserves_arrangement_vs_wider_gap():
    """Arrangement preservation: the RELATIVE direction from the seed center to
    each part is unchanged when the gap widens -- only magnitudes grow. This is
    the property force-directed placement destroys and relaxation must keep."""
    import math

    from skidl.schematics.seed_place import seed_placement

    def dirs(gap):
        parts = [make_part(f"R{i+1}") for i in range(4)]
        nets = [
            FakeNet(f"N{i}", [parts[i].pins[0], parts[i + 1].pins[0]])
            for i in range(3)
        ]
        seed_placement(parts, nets, gap=gap, grid=50)
        # center = first part origin (seed places center at (0,0))
        cx, cy = parts[0].tx.dx, parts[0].tx.dy
        out = {}
        for p in parts[1:]:
            dx, dy = p.tx.dx - cx, p.tx.dy - cy
            out[p.ref] = (
                0 if abs(dx) < 1e-6 else (1 if dx > 0 else -1),
                0 if abs(dy) < 1e-6 else (1 if dy > 0 else -1),
            )
        return out

    narrow = dirs(100)   # 2*grid
    wide = dirs(200)     # 4*grid
    assert narrow == wide, f"arrangement changed with gap: {narrow} vs {wide}"


def test_relax_gap_wider_than_plain_seed():
    """The relaxation default gap spreads parts at least as far as the plain
    seed's 2*grid default (more breathing room), given the same arrangement."""
    from skidl.schematics.seed_place import seed_placement

    def span(seeder, **kw):
        parts = [make_part(f"R{i+1}") for i in range(4)]
        nets = [
            FakeNet(f"N{i}", [parts[i].pins[0], parts[i + 1].pins[0]])
            for i in range(3)
        ]
        seeder(parts, nets, grid=50, **kw)
        xs = [p.tx.dx for p in parts]
        ys = [p.tx.dy for p in parts]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    plain = span(seed_placement)
    relaxed = span(rp.relax_placement)
    assert relaxed >= plain, f"relax span {relaxed} < plain seed span {plain}"
