# -*- coding: utf-8 -*-

"""Tests for the optional placement-animation debug tool (schematics.debug_anim).

Covers the plan's acceptance: the observer hook is None by default (production
path untouched), it fires exactly once per placed part, placements are segmented
per sheet, the recorder itself needs no Pillow, and render_gifs writes one GIF per
sheet (skipped when Pillow is absent).
"""

import re

import pytest

from skidl.geometry import BBox, Point, Tx, tx_rot_90
from skidl.schematics import debug_anim
from skidl.schematics import seed_place as sp


class FakePin:
    def __init__(self, pt, orientation):
        self.pt = pt
        self.orientation = orientation
        self.stub = False
        self.part = None
        self.net = None


class FakePart:
    def __init__(self, ref, pins, hiertuple=None, bbox=None, tx=None):
        self.ref = ref
        self.pins = pins
        for p in pins:
            p.part = self
        self.place_bbox = bbox or BBox(Point(-100, -100), Point(100, 100))
        self.orientation_locked = False
        self.tx = tx or Tx()
        if hiertuple is not None:
            self.hiertuple = hiertuple


class FakeNet:
    def __init__(self, name, pins, drive=None):
        self.name = name
        self.pins = pins
        self.drive = drive
        self.stub = False
        for p in pins:
            p.net = self


def _rpins():
    return [FakePin(Point(0, 381), "D"), FakePin(Point(0, -381), "U")]


def _chain(n, hiertuple=None):
    parts = [FakePart(f"R{i}", _rpins(), hiertuple=hiertuple) for i in range(n)]
    nets = [
        FakeNet(f"N{i}", [parts[i].pins[0], parts[i + 1].pins[1]])
        for i in range(n - 1)
    ]
    return parts, nets


def test_observer_none_by_default():
    """The production placement path must carry no observer."""
    assert sp._PLACEMENT_OBSERVER is None


def test_records_one_snapshot_per_placement():
    parts, nets = _chain(5)
    with debug_anim.record_placements() as rec:
        sp.seed_placement(parts, nets, grid=50)
    # observer restored on exit
    assert sp._PLACEMENT_OBSERVER is None

    total = sum(len(v) for v in rec.sheets.values())
    assert total == len(parts), f"expected one frame per part, got {total}"

    # No hiertuple on the fakes -> single 'top' sheet, orders 0..n-1 contiguous.
    (key,) = list(rec.sheets)
    recs = rec.sheets[key]
    assert [r["order"] for r in recs] == list(range(len(parts)))
    for r in recs:
        assert len(r["box"]) == 4
        assert r["pins"], "pins should be captured"
        ux, uy = r["up"]
        assert abs((ux * ux + uy * uy) ** 0.5 - 1.0) < 1e-6  # unit up vector


def test_up_vector_tracks_rotation():
    """The recorded 'up' vector must follow the part's tx rotation.

    (Snapshot the recorder directly, since seed_placement itself chooses each
    part's rotation and would overwrite a hand-set tx.)"""
    rec = debug_anim.PlacementRecorder()
    a = FakePart("U1", _rpins())               # identity tx -> up along -y
    b = FakePart("U2", _rpins(), tx=tx_rot_90)  # rotated -> up gains an x-component
    rec(a)
    rec(b)
    ups = {r["ref"]: r["up"] for r in next(iter(rec.sheets.values()))}
    assert abs(ups["U1"][0]) < 1e-6            # identity: up stays on the y axis
    assert abs(ups["U2"][0]) > 0.5             # rot90: up swings onto the x axis


def test_segments_per_sheet_by_hiertuple():
    pa, na = _chain(3, hiertuple=("top", "sheetA"))
    pb, nb = _chain(2, hiertuple=("top", "sheetB"))
    with debug_anim.record_placements() as rec:
        sp.seed_placement(pa, na, grid=50)
        sp.seed_placement(pb, nb, grid=50)
    names = {debug_anim._sheet_name(k): len(v) for k, v in rec.sheets.items()}
    assert names == {"sheetA": 3, "sheetB": 2}


def test_render_gifs_writes_one_per_sheet(tmp_path):
    pytest.importorskip("PIL")
    pa, na = _chain(3, hiertuple=("top", "sheetA"))
    pb, nb = _chain(2, hiertuple=("top", "sheetB"))
    with debug_anim.record_placements() as rec:
        sp.seed_placement(pa, na, grid=50)
        sp.seed_placement(pb, nb, grid=50)
    paths = debug_anim.render_gifs(
        rec, out_dir=str(tmp_path), prefix="t", ms_per_frame=100
    )
    assert len(paths) == 2
    names = sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in paths)
    assert re.match(r"t_sheet00_sheetA\.gif$", names[0])
    assert re.match(r"t_sheet01_sheetB\.gif$", names[1])
    for p in paths:
        import os

        assert os.path.getsize(p) > 0
