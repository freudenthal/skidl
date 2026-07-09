# -*- coding: utf-8 -*-

"""Unit tests for the stage-24 per-net Hanan-grid A* router primitives.

Exercises the module-level routing helpers in ``skidl.schematics.route`` that
replaced the face-capacity switchbox stack:

  a. two pins with an obstacle between them  -> routed around, all H/V;
  b. two nets crossing                        -> both route, crossing is
     perpendicular (legal, no colinear overlap);
  c. colinear-overlap forbidden               -> a second net sharing a
     corridor detours instead of overlapping a foreign segment;
  d. enclosed pin                             -> A* returns None (the router's
     per-net stub fallback path);
  + geometry regression on the REAL placed buck_5v geometry -> 8/8 nets routed
    (incl. SW3, the net skidl's capacity router could not route).
"""

import json
from pathlib import Path

import pytest

from skidl.schematics.route import (
    _astar_pair,
    _colinear_foreign,
    _mst_edges,
    _seg_hits_interior,
)

# Optional geometry-regression asset. Drop ``buck5v_geom.json`` (a real placed
# board geometry dumped from the schematic placer) next to this test to enable
# the regression below; it is skipped when the asset is absent.
REPRO = Path(__file__).resolve().parent / "data" / "buck5v_geom.json"


def _is_axis_aligned(path):
    """Every consecutive pair in a path is horizontal or vertical."""
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if not (x0 == x1 or y0 == y1):
            return False
    return True


def _segments(path):
    segs = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if (x0, y0) != (x1, y1):
            segs.append((x0, y0, x1, y1))
    return segs


def test_route_around_obstacle():
    """(a) A pin-to-pin route steps around a blocking part body, staying H/V."""
    # Pins at (0,0) and (400,0); a part body straddles the straight line.
    obstacle = (100, -100, 300, 100)
    path = _astar_pair(
        (0, 0), (400, 0), [obstacle], {}, {}, {}, net="A", margin=100, turn=50.0
    )
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (400, 0)
    assert _is_axis_aligned(path)
    # No segment may pierce the obstacle interior.
    for x0, y0, x1, y1 in _segments(path):
        assert not _seg_hits_interior(x0, y0, x1, y1, [obstacle])


def test_two_nets_cross_perpendicular():
    """(b) Two nets whose routes cross do so perpendicularly (legal, no overlap)."""
    foreign_h, foreign_v = {}, {}
    # Net A: horizontal across the middle.
    pa = _astar_pair(
        (0, 0), (400, 0), [], {}, foreign_h, foreign_v, net="A", margin=100, turn=50.0
    )
    for x0, y0, x1, y1 in _segments(pa):
        if y0 == y1:
            foreign_h.setdefault(y0, []).append((min(x0, x1), max(x0, x1), "A"))
        else:
            foreign_v.setdefault(x0, []).append((min(y0, y1), max(y0, y1), "A"))
    # Net B: vertical through the same region -> must cross A, not overlap it.
    pb = _astar_pair(
        (200, -200),
        (200, 200),
        [],
        {},
        foreign_h,
        foreign_v,
        net="B",
        margin=100,
        turn=50.0,
    )
    assert pa is not None and pb is not None
    # B's segments never run colinear over A's.
    for x0, y0, x1, y1 in _segments(pb):
        assert not _colinear_foreign(x0, y0, x1, y1, "B", foreign_h, foreign_v)


def test_colinear_overlap_forbidden():
    """(c) A net may not run colinear-overlapping a foreign net's segment."""
    # A foreign horizontal segment owned by net A along y=0, x in [0,400].
    foreign_h = {0: [(0, 400, "A")]}
    foreign_v = {}
    # Directly stepping along y=0 from (100,0)->(300,0) would overlap A.
    assert _colinear_foreign(100, 0, 300, 0, "B", foreign_h, foreign_v) is True
    # The SAME net is free to share its own line.
    assert _colinear_foreign(100, 0, 300, 0, "A", foreign_h, foreign_v) is False
    # A perpendicular crossing of A is legal (not colinear).
    assert _colinear_foreign(200, -50, 200, 50, "B", foreign_h, foreign_v) is False
    # And the router detours net B around the occupied corridor rather than
    # overlapping it: route B from (0,0) to (400,0) with A owning y=0.
    path = _astar_pair(
        (0, 0), (400, 0), [], {}, foreign_h, foreign_v, net="B", margin=100, turn=50.0
    )
    assert path is not None
    for x0, y0, x1, y1 in _segments(path):
        assert not _colinear_foreign(x0, y0, x1, y1, "B", foreign_h, foreign_v)


def test_enclosed_pin_returns_none():
    """(d) A fully enclosed endpoint yields no path (per-net stub fallback)."""
    # A solid obstacle ring whose four rects OVERLAP at the corners seals the
    # hole x(180,220) y(-20,20) around the target (200,0): every approach to the
    # inner boundary crosses a wall interior (strict-interior obstacles can be
    # circumnavigated along edges only when the walls merely touch -- overlap
    # turns the shared inner-edge extensions into interior). Start outside.
    walls = [
        (100, -100, 300, -20),  # bottom (full width)
        (100, 20, 300, 100),  # top    (full width)
        (100, -100, 180, 100),  # left   (full height, overlaps top & bottom)
        (220, -100, 300, 100),  # right  (full height, overlaps top & bottom)
    ]
    path = _astar_pair(
        (0, 0), (200, 0), walls, {}, {}, {}, net="A", margin=100, turn=50.0
    )
    assert path is None


def test_mst_edges_deterministic_and_spanning():
    pts = [(0, 0), (100, 0), (0, 100), (100, 100)]
    e1 = _mst_edges(pts)
    e2 = _mst_edges(pts)
    assert e1 == e2  # deterministic
    assert len(e1) == len(pts) - 1  # spanning tree
    # single / empty point sets have no edges
    assert _mst_edges([(5, 5)]) == []
    assert _mst_edges([]) == []


@pytest.mark.skipif(not REPRO.exists(), reason="buck5v_geom.json repro asset missing")
def test_buck5v_geometry_regression():
    """The real placed buck_5v geometry routes 8/8 nets, 0 body violations."""
    data = json.loads(REPRO.read_text())
    obstacles, pin_world = [], {}
    for part in data["parts"]:
        for pin in part["pins"]:
            if "world" in pin:
                pin_world[f'{part["ref"]}.{pin["num"]}'] = tuple(pin["world"])
        if part["ref"].startswith("NT"):
            continue
        (xmn, ymn), (xmx, ymx) = part["body_bbox"]
        obstacles.append((xmn, ymn, xmx, ymx))

    occupied, foreign_h, foreign_v, net_pts = {}, {}, {}, {}
    for net in data["internal_nets"]:
        uniq = list(dict.fromkeys(pin_world[p] for p in net["pins"] if p in pin_world))
        net_pts[net["name"]] = uniq
        for p in uniq:
            occupied[p] = net["name"]
    occ_xs = {p[0] for p in occupied}
    occ_ys = {p[1] for p in occupied}

    def register(net, x0, y0, x1, y1):
        if y0 == y1:
            foreign_h.setdefault(y0, []).append((min(x0, x1), max(x0, x1), net))
        elif x0 == x1:
            foreign_v.setdefault(x0, []).append((min(y0, y1), max(y0, y1), net))

    routed = total = 0
    for name in sorted(net_pts):
        uniq = net_pts[name]
        if len(uniq) < 2:
            continue
        total += 1
        ok, segs = True, []
        for i, j in _mst_edges(uniq):
            path = _astar_pair(
                uniq[i],
                uniq[j],
                obstacles,
                occupied,
                foreign_h,
                foreign_v,
                name,
                extra_xs=occ_xs,
                extra_ys=occ_ys,
                margin=100,
                turn=50.0,
            )
            if path is None:
                ok = False
                break
            segs.extend(_segments(path))
        if ok:
            routed += 1
            for s in segs:
                # no routed segment may pierce a part body
                assert not _seg_hits_interior(*s, obstacles)
                register(name, *s)
    assert routed == total == 8, f"routed {routed}/{total}"
    # SW3 specifically -- the net skidl's capacity router could not route.
    assert "SW3" in net_pts and len(net_pts["SW3"]) >= 2
