# -*- coding: utf-8 -*-

"""Unit tests for the SheetOccupancy registry (render-occupancy plan Phase 1).

Pure data-structure tests -- no KiCad libraries needed. They pin the claim/deny,
segment-claim, bbox-blocking, and determinism semantics the router + emitter
build on in Phases 2-4.
"""

from types import SimpleNamespace

from skidl.schematics.occupancy import SheetOccupancy

GRID = 50.0


def _pt(x, y):
    return (x, y)


def test_cell_snaps_to_grid():
    occ = SheetOccupancy(GRID)
    assert occ.cell(0, 0) == (0.0, 0.0)
    assert occ.cell(24, -24) == (0.0, -0.0)
    assert occ.cell(26, 74) == (50.0, 50.0)
    assert occ.cell(51, 100) == (50.0, 100.0)


def test_claim_and_owner():
    occ = SheetOccupancy(GRID)
    netA, netB = object(), object()
    c = occ.cell(100, 100)
    assert occ.owner(c) is None
    assert occ.is_free_for(c, netA)
    assert occ.claim(c, netA) is True
    assert occ.owner(c) is netA
    # same net may re-claim; a foreign net may not.
    assert occ.claim(c, netA) is True
    assert occ.is_free_for(c, netB) is False
    assert occ.claim(c, netB) is False
    assert occ.owner(c) is netA


def test_seed_is_first_writer_wins():
    occ = SheetOccupancy(GRID)
    netA, netB = object(), object()
    c = occ.cell(0, 0)
    occ.seed(c, netA)
    occ.seed(c, netB)  # ignored -- already owned
    assert occ.owner(c) is netA
    # set() overwrites unconditionally
    occ.set(c, netB)
    assert occ.owner(c) is netB


def test_claim_segment_registers_interior():
    occ = SheetOccupancy(GRID)
    net = object()
    conflicts = occ.claim_segment(_pt(0, 0), _pt(200, 0), net)
    assert conflicts == []
    for x in (0.0, 50.0, 100.0, 150.0, 200.0):
        assert occ.owner((x, 0.0)) is net
    # a foreign net cannot land on the interior
    other = object()
    assert occ.is_free_for((100.0, 0.0), other) is False


def test_claim_segment_reports_foreign_conflicts():
    occ = SheetOccupancy(GRID)
    netA, netB = object(), object()
    occ.set(occ.cell(100, 0), netA)
    conflicts = occ.claim_segment(_pt(0, 0), _pt(200, 0), netB)
    assert conflicts == [(100.0, 0.0)]
    # the conflicting cell keeps its original owner; the rest belong to netB
    assert occ.owner((100.0, 0.0)) is netA
    assert occ.owner((50.0, 0.0)) is netB


def test_claim_segment_vertical_and_single_cell():
    occ = SheetOccupancy(GRID)
    net = object()
    occ.claim_segment(_pt(0, 0), _pt(0, 100), net)
    assert occ.owner((0.0, 50.0)) is net
    # zero-length segment claims one cell
    net2 = object()
    occ.claim_segment(_pt(500, 500), _pt(500, 500), net2)
    assert occ.owner((500.0, 500.0)) is net2


def test_block_bbox_strict_interior():
    occ = SheetOccupancy(GRID)
    bbox = SimpleNamespace(min=SimpleNamespace(x=0, y=0), max=SimpleNamespace(x=100, y=100))
    occ.block_bbox(bbox, strict=True)
    # interior cell blocked; edge cells (pins live there) not blocked
    assert occ.is_blocked((50.0, 50.0)) is True
    assert occ.is_blocked((0.0, 50.0)) is False
    assert occ.is_blocked((100.0, 50.0)) is False
    # is_free_for only rejects blocked when asked to
    net = object()
    assert occ.is_free_for((50.0, 50.0), net, respect_blocked=False) is True
    assert occ.is_free_for((50.0, 50.0), net, respect_blocked=True) is False


def test_published_drops_none_and_is_deterministic():
    occ = SheetOccupancy(GRID)
    netA = object()
    occ.set(occ.cell(0, 0), netA)
    occ.set(occ.cell(50, 0), None)  # unowned marker
    pub = occ.published()
    assert (0.0, 0.0) in pub and (50.0, 0.0) not in pub
    # owners_sorted is stable
    keys = [c for c, _ in occ.owners_sorted()]
    assert keys == sorted(keys)


def test_rescaled_converts_frame():
    occ = SheetOccupancy(GRID)
    net = object()
    occ.set(occ.cell(100, 200), net)
    r = occ.rescaled(0.0254 / 50.0 * 50.0)  # arbitrary scale
    # scale factor 1 -> identical
    r1 = occ.rescaled(1.0)
    assert r1.owner((100.0, 200.0)) is net
