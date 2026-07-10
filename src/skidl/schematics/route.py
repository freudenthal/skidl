# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""
Autorouter for generating wiring between symbols in a schematic.
"""

import copy
import heapq
import math
import random
import sys
from collections import Counter, defaultdict
from enum import Enum
from itertools import chain, count, zip_longest

from skidl import Part
from skidl.utilities import export_to_all, rmv_attr
from .debug_draw import (
    draw_end,
    draw_endpoint,
    draw_routing,
    draw_seg,
    draw_start,
    draw_text,
)
from skidl.geometry import BBox, Point, Segment, Tx, Vector, tx_rot_90

__all__ = ["RoutingFailure", "GlobalRoutingFailure", "SwitchboxRoutingFailure"]


###################################################################
#
# OVERVIEW OF SCHEMATIC AUTOROUTER
#
# The input is a Node containing child nodes and parts, each with a
# bounding box and an assigned (x,y) position. The following operations
# are done for each child node, and then for the parts within this node.
#
# The edges of each part bbox are extended to form tracks that divide the
# routing area into a set of four-sided, non-overlapping switchboxes. Each
# side of a switchbox is a Face, and each Face is a member of two adjoining
# switchboxes (except those Faces on the boundary of the total
# routing area.) Each face is adjacent to the six other faces of
# the two switchboxes it is part of.
#
# Each face has a capacity that indicates the number of wires that can
# cross through it. The capacity is the length of the face divided by the
# routing grid. (Faces on a part boundary have zero capacity to prevent
# routing from entering a part.)
#
# Each face on a part bbox is assigned terminals associated with the I/O
# pins of that symbol.
#
# After creating the faces and terminals, the global routing phase creates
# wires that connect the part pins on the nets. Each wire passes from
# a face of a switchbox to one of the other three faces, either directly
# across the switchbox to the opposite face or changing direction to
# either of the right-angle faces. The global router is basically a maze
# router that uses the switchboxes as high-level grid squares.
#
# After global routing, each net has a sequence of switchbox faces
# through which it will transit. The exact coordinate that each net
# enters a face is then assigned to create a Terminal.
#
# At this point there are a set of switchboxes which have fixed terminals located
# along their four faces. A greedy switchbox router
# (https://doi.org/10.1016/0167-9260(85)90029-X)
# does the detailed routing within each switchbox.
#
# The detailed wiring within all the switchboxes is combined and output
# as the total wiring for the parts in the Node.
#
###################################################################


# Orientations and directions.
class Orientation(Enum):
    HORZ = 1
    VERT = 2


class Direction(Enum):
    LEFT = 3
    RIGHT = 4


# Put the orientation/direction enums in global space to make using them easier.
for orientation in Orientation:
    globals()[orientation.name] = orientation.value
for direction in Direction:
    globals()[direction.name] = direction.value


class RoutingFailure(Exception):
    """Exception raised when a net connecting pins cannot be routed."""

    pass


class GlobalRoutingFailure(RoutingFailure):
    """Failure during global routing phase."""

    pass


class SwitchboxRoutingFailure(RoutingFailure):
    """Failure during switchbox routing phase."""

    pass


def _stub_node_subtree_nets(node):
    """Stub every non-explicit net owned by *node* and its descendants to labels.

    Used by the per-sheet routing-failure isolation in ``Router.route``: when one
    sheet can't be routed, its nets become labels (no wires) so the sheet still
    emits and the rest of the design keeps its routed wires. Backend-agnostic —
    touches only skidl core ``net._stub`` / ``pin.stub`` flags. Explicit user
    stubs are left as-is. (stage 19 router robustness)
    """
    for net in node.get_internal_nets():
        if getattr(net, "_stub_explicit", False):
            continue
        net._stub = True
        for pin in net.get_pins():
            pin.stub = True
    for child in node.children.values():
        _stub_node_subtree_nets(child)


class Boundary:
    """Class for indicating a boundary.

    When a Boundary object is placed in the part attribute of a Face, it
    indicates the Face is on the outer boundary of the Node routing area
    and no routes can pass through it.
    """

    pass


# Boundary object for placing in the bounding Faces of the Node routing area.
boundary = Boundary()

# Absolute coords of all part pins. Used when trimming stub nets.
pin_pts = []


# ---------------------------------------------------------------------------
# Per-net Hanan-grid A* routing primitives (stage 24).
#
# Schematic wires may cross freely (a crossing without a junction dot is "no
# connection"), so there is NO face-capacity / congestion model -- only:
#   * part BODY bboxes are obstacles (routes never enter a part interior),
#   * two DIFFERENT nets must never run COLINEAR-overlapping (a silent merge),
#     and a wire must not pass through / end on another net's pin,
#   * perpendicular crossings of other nets are unconstrained.
# These module-level helpers are pure (no Node state) so they are directly
# unit-testable; ``Router.route_internal_nets_astar`` composes them.
# ---------------------------------------------------------------------------


def _seg_hits_interior(x0, y0, x1, y1, obstacles):
    """True if an axis-aligned segment crosses a part-body (STRICT) interior.

    Route-points sit ON obstacle edges and wires may run along a boundary, so
    the test is strict-interior: only a positive-length overlap with the OPEN
    rectangle counts. ``obstacles`` is a list of (xmin, ymin, xmax, ymax).
    """
    if x0 == x1:  # vertical
        x = x0
        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
        for xmn, ymn, xmx, ymx in obstacles:
            if xmn < x < xmx and min(hi, ymx) - max(lo, ymn) > 0:
                return True
    else:  # horizontal
        y = y0
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        for xmn, ymn, xmx, ymx in obstacles:
            if ymn < y < ymx and min(hi, xmx) - max(lo, xmn) > 0:
                return True
    return False


def _colinear_foreign(x0, y0, x1, y1, net, foreign_h, foreign_v):
    """True if the step runs COLINEAR-overlapping a DIFFERENT net's segment.

    ``foreign_h`` maps y -> list of (xlo, xhi, net); ``foreign_v`` maps
    x -> list of (ylo, yhi, net). Perpendicular crossings are never reported
    here (they are legal); only same-line positive-length overlap with a
    segment owned by another net counts.
    """
    if y0 == y1:
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        for xlo, xhi, onet in foreign_h.get(y0, ()):
            if onet is not net and min(hi, xhi) - max(lo, xlo) > 0:
                return True
    else:
        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
        for ylo, yhi, onet in foreign_v.get(x0, ()):
            if onet is not net and min(hi, yhi) - max(lo, ylo) > 0:
                return True
    return False


def _mst_edges(points):
    """Prim MST over world points; deterministic (dist, endpoints) tie-break.

    Returns a list of (i, j) index pairs into ``points``.
    """
    n = len(points)
    if n <= 1:
        return []
    used = {0}
    edges = []
    while len(used) < n:
        best = None
        for i in used:
            for j in range(n):
                if j in used:
                    continue
                d = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                cand = (d, points[i], points[j], i, j)
                if best is None or cand < best:
                    best = cand
        used.add(best[4])
        edges.append((best[3], best[4]))
    return edges


def _astar_pair(
    a,
    b,
    obstacles,
    occupied,
    foreign_h,
    foreign_v,
    net,
    extra_xs=(),
    extra_ys=(),
    margin=100,
    turn=50.0,
):
    """A* on a Hanan grid from world point ``a`` to ``b`` for ``net``.

    Candidate grid lines pass through the two endpoints, every obstacle edge,
    each endpoint +/- ``margin`` (detour clearance), and ``extra_xs``/``extra_ys``
    (fold every foreign pin coordinate in so a perpendicular pass-through cannot
    skip over a pin un-vertexed). Returns the path as a list of world points, or
    ``None`` if no obstacle/overlap-free path exists (only when an endpoint is
    fully enclosed). Deterministic: the open set is ordered by a monotone
    counter, never by comparing nodes or paths.
    """
    ax, ay = a
    bx, by = b
    # Corridor tracks come from obstacle edges. Endpoints, endpoint+/-margin
    # (margin is a whole-grid multiple) and the folded-in pin coords (extra_*)
    # are all on-grid, so the ONLY off-grid Hanan lines are the raw bbox edges.
    # Snap those OUTWARD to the grid (low edges down, high edges up): the
    # candidate corridor moves into free space (never toward the raw obstacle,
    # which _seg_hits_interior still tests against), and every routed corner
    # lands on-grid -> no endpoint_off_grid. Idempotent for the deconflict path,
    # whose obstacles are already grid-grown.
    _g = float(GRID)

    def _lo(v):
        return math.floor(v / _g) * _g

    def _hi(v):
        return math.ceil(v / _g) * _g

    xs = sorted(
        {
            ax,
            bx,
            ax - margin,
            ax + margin,
            bx - margin,
            bx + margin,
            *(_lo(o[0]) for o in obstacles),
            *(_hi(o[2]) for o in obstacles),
            *extra_xs,
        }
    )
    ys = sorted(
        {
            ay,
            by,
            ay - margin,
            ay + margin,
            by - margin,
            by + margin,
            *(_lo(o[1]) for o in obstacles),
            *(_hi(o[3]) for o in obstacles),
            *extra_ys,
        }
    )
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    start = (xi[ax], yi[ay])
    goal = (xi[bx], yi[by])

    def h(nd):
        return abs(xs[nd[0]] - bx) + abs(ys[nd[1]] - by)

    ctr = count()
    openq = [(h(start), 0.0, next(ctr), start, 0, (a,))]
    best = {}
    while openq:
        f, g, _, nd, indir, path = heapq.heappop(openq)
        if nd == goal:
            return path
        key = (nd, indir)
        if key in best and best[key] <= g:
            continue
        best[key] = g
        cx, cy = xs[nd[0]], ys[nd[1]]
        for dxn, dyn, axis in ((1, 0, 1), (-1, 0, 1), (0, 1, 2), (0, -1, 2)):
            nx, ny = nd[0] + dxn, nd[1] + dyn
            if not (0 <= nx < len(xs) and 0 <= ny < len(ys)):
                continue
            wx, wy = xs[nx], ys[ny]
            o = occupied.get((wx, wy))
            if o is not None and o is not net and (wx, wy) != b:
                continue  # don't route through/onto another net's pin
            if _seg_hits_interior(cx, cy, wx, wy, obstacles):
                continue
            if _colinear_foreign(cx, cy, wx, wy, net, foreign_h, foreign_v):
                continue
            cost = (
                abs(wx - cx) + abs(wy - cy) + (turn if indir and indir != axis else 0.0)
            )
            heapq.heappush(
                openq,
                (
                    g + cost + h((nx, ny)),
                    g + cost,
                    next(ctr),
                    (nx, ny),
                    axis,
                    path + ((wx, wy),),
                ),
            )
    return None


# ---------------------------------------------------------------------------
# Deconflicted-stub geometry (stage 25).
#
# Retires snap: instead of cramming 2-pin parts onto IC pins (which caused both
# the off-grid warnings and the cross-net false-merges), every non-power pin
# gets a short, on-grid, world-unique stub wire projecting straight out of the
# part body. The A* router then wires the stub ENDS. Because no two nets ever
# share a stub-end cell, the connectivity audit is silent by construction and
# closure labels (stage 25 phase 2) can sit on the ends without fusing nets.
# ---------------------------------------------------------------------------


def _invert_dihedral(pt, tx):
    """Map a WORLD point back to a part's LOCAL frame for a dihedral ``tx``.

    ``Point * Tx`` computes ``wx = lx*a + ly*c + dx`` and
    ``wy = lx*b + ly*d + dy``. Part transforms are dihedral (90-degree rotation
    and/or mirror + translation) so ``det = a*d - b*c = +/-1`` and the inverse is
    exact. Returns the local ``Point`` (so ``local * tx == pt``).
    """
    det = tx.a * tx.d - tx.b * tx.c
    if det == 0:
        return Point(pt.x, pt.y)  # degenerate (shouldn't happen for a placed part)
    ux = pt.x - tx.dx
    uy = pt.y - tx.dy
    lx = (ux * tx.d - uy * tx.c) / det
    ly = (uy * tx.a - ux * tx.b) / det
    return Point(lx, ly)


def _snap_away(v, ref, grid):
    """Snap ``v`` to the ``grid``, rounding AWAY from ``ref`` (never toward it).

    Used to push a stub end out to a grid line without ever pulling it back
    inside the part body.
    """
    import math

    if v >= ref:
        return math.ceil(v / grid - 1e-9) * grid
    return math.floor(v / grid + 1e-9) * grid


# Direction a pin POINTS (toward the part body), local frame. The stub extends
# the OPPOSITE way (outward, away from the body).
_ORIENT_VEC = {
    "U": Point(0, 1),
    "D": Point(0, -1),
    "L": Point(-1, 0),
    "R": Point(1, 0),
}


def _world_outward_dir(pin):
    """Unit world-outward direction of a pin's stub as an integer ``(dx, dy)``.

    The pin's local ``orientation`` folded through the rotation/mirror part of
    ``pin.part.tx`` (translation dropped) and NEGATED — outward means away from
    the part body. This is the same source the emitters' ``calc_pin_dir`` uses
    (so stub geometry and its closure label agree under any rotation/mirror),
    just negated. Dihedral transforms only, so the result is exactly axial.
    """
    tx = pin.part.tx
    rot = Tx(a=tx.a, b=tx.b, c=tx.c, d=tx.d)  # rotation/mirror only, no shift
    ov = _ORIENT_VEC[pin.orientation] * rot
    return -int(round(ov.x)), -int(round(ov.y))


@export_to_all
class Router:
    """Mixin to add routing function to Node class."""

    def add_routing_points(node, nets):
        """Add routing points by extending wires from pins out to the edge of the part bounding box.

        Args:
            nets (list): List of nets to be routed.
        """

        def add_routing_pt(pin):
            """Add the point for a pin on the boundary of a part."""

            bbox = pin.part.lbl_bbox
            pin.route_pt = copy.copy(pin.pt)
            if pin.orientation == "U":
                # Pin points up, so extend downward to the bottom of the bounding box.
                pin.route_pt.y = bbox.min.y
            elif pin.orientation == "D":
                # Pin points down, so extend upward to the top of the bounding box.
                pin.route_pt.y = bbox.max.y
            elif pin.orientation == "L":
                # Pin points left, so extend rightward to the right-edge of the bounding box.
                pin.route_pt.x = bbox.max.x
            elif pin.orientation == "R":
                # Pin points right, so extend leftward to the left-edge of the bounding box.
                pin.route_pt.x = bbox.min.x
            else:
                raise RuntimeError("Unknown pin orientation.")

        # Global set of part pin (x,y) points may have stuff from processing previous nodes, so clear it.
        del pin_pts[:]  # Clear the list. Works for Python 2 and 3.

        for net in nets:
            # Add routing points for all pins on the net that are inside this node.
            for pin in node.get_internal_pins(net):
                # Store the point where the pin is. (This is used after routing to trim wire stubs.)
                pin_pts.append((pin.pt * pin.part.tx).round())

                # Add the point to which the wiring should be extended.
                add_routing_pt(pin)

                # Add a wire to connect the part pin to the routing point on the bounding box periphery.
                if pin.route_pt != pin.pt:
                    seg = Segment(pin.pt, pin.route_pt) * pin.part.tx
                    node.wires[pin.net].append(seg)

    def add_deconflicted_stubs(node, internal_nets, **options):
        """Stage-25: give every non-power, non-NC pin an on-grid, world-unique
        stub wire projecting out of the part body, then route between the ends.

        Replaces ``add_routing_points`` when ``deconflict_stubs`` is set. For
        EVERY connected non-power, non-NC pin on a real part in this node
        (routed nets AND label-only stubbed nets) it:

          * pushes the pin out to its labeled-bbox edge (same outward
            axis/direction as ``add_routing_points``),
          * snaps that end OUTWARD to the 50-mil grid and guarantees it is at
            least one grid unit past the pin,
          * DECONFLICTS the end against every other pin and every stub end on
            the node (stepping one grid further out until the cell is free), so
            no two different nets ever share a cell -> the connectivity audit is
            silent by construction and off-grid endpoints are impossible,
          * records the world end in ``node._stub_ends[id(pin)]`` (the closure
            labeller's anchor in phase 2).

        For ROUTED (non-stub) nets it also appends the pin->end stub ``Segment``
        to ``node.wires`` (as ``add_routing_points`` does) and sets ``route_pt``
        so the A* router wires the ends. For label-only (stubbed) nets it only
        records the end (they carry no wire until the phase-2 emitter draws the
        stub from the recorded end). The full occupancy is published on
        ``node._deconflict_occupied`` so the A* router avoids every pin and stub
        end of every net, not just the routed ones.

        Deterministic: pins processed in ``(part.ref, str(pin.num))`` order.
        """
        from skidl.net import NCNet
        from skidl.schematics.net_terminal import NetTerminal

        grid = float(GRID)

        del pin_pts[:]  # world pin coords protect stub roots during trimming
        node._stub_ends = {}
        node._stub_wire_nets = defaultdict(list)
        node._stub_terminal_pins = set()  # NetTerminal pins (own label already)
        node._deconflict_stubs = True  # signals the emitter's closure labeller

        internal_ids = {id(n) for n in internal_nets}

        # Collect every stub-able pin in a deterministic order. NetTerminal pins
        # ARE included (they need a routing point so A* wires the net to the
        # terminal), but they carry their own label so phase 2 must not add a
        # closure label at their end -- flag them.
        pin_recs = []  # (sort_key, pin, net, is_routed)
        for part in node.parts:
            is_terminal = isinstance(part, NetTerminal)
            for pin in part:
                if not pin.is_connected():
                    continue
                net = pin.net
                if isinstance(net, NCNet):
                    continue
                if getattr(net, "_is_power_net", False):
                    continue
                if is_terminal:
                    node._stub_terminal_pins.add(id(pin))
                ref = str(getattr(part, "ref", "") or "")
                sort_key = (ref, str(pin.num), id(pin))
                pin_recs.append((sort_key, pin, net, id(net) in internal_ids))
        pin_recs.sort(key=lambda r: r[0])

        # Occupancy: grid cell -> owning net. Seed with EVERY part pin (world,
        # grid-rounded) of EVERY net so a stub end never lands on a pin.
        def cell(x, y):
            return (round(x / grid) * grid, round(y / grid) * grid)

        occupied = {}
        for part in node.parts:
            for pin in part:
                wp = (pin.pt * part.tx).round()
                occupied.setdefault(cell(wp.x, wp.y), getattr(pin, "net", None))

        from skidl.logger import active_logger

        for _key, pin, net, is_routed in pin_recs:
            part = pin.part
            pin_w = (pin.pt * part.tx).round()
            pin_pts.append(pin_w)

            # Outward point at the labeled-bbox edge (local frame), same
            # axis/direction rule as add_routing_points.
            bbox = part.lbl_bbox
            edge = copy.copy(pin.pt)
            if pin.orientation == "U":
                edge.y = bbox.min.y
            elif pin.orientation == "D":
                edge.y = bbox.max.y
            elif pin.orientation == "L":
                edge.x = bbox.max.x
            elif pin.orientation == "R":
                edge.x = bbox.min.x
            else:
                raise RoutingFailure("Unknown pin orientation.")
            edge_w = (edge * part.tx).round()

            # The world stub is axial and points along the pin's TRUE world
            # outward direction (orientation folded through part.tx), NOT
            # inferred from lbl_bbox deltas: when the bbox edge coincides with
            # the pin (common -- pins sit on the body edge) those deltas are
            # zero and a magnitude compare mis-picks the axis, flattening every
            # stub to left/right. The bbox edge is kept only to set the stub
            # LENGTH along that direction.
            dx, dy = _world_outward_dir(pin)
            if abs(dx) + abs(dy) != 1:
                raise RoutingFailure(
                    "Non-dihedral part transform for pin %s of %s"
                    % (getattr(pin, "num", "?"), getattr(part, "ref", "?"))
                )
            if dx != 0:
                axis = "x"
                sign = float(dx)
                fixed, moving_pin, moving_edge = pin_w.y, pin_w.x, edge_w.x
            else:
                axis = "y"
                sign = float(dy)
                fixed, moving_pin, moving_edge = pin_w.x, pin_w.y, edge_w.y
            if (moving_edge - moving_pin) * sign <= 0:
                # bbox edge coincides with the pin (or lies on the wrong side):
                # still project a real stub, one grid out along the outward dir.
                moving_edge = moving_pin + sign * grid

            # Snap outward to grid and guarantee >= 1 grid of stub.
            end_v = _snap_away(moving_edge, moving_pin, grid)
            if abs(end_v - moving_pin) < grid:
                end_v = moving_pin + sign * grid

            # Deconflict: step one grid further out until the cell is free (or
            # already owned by this same net). Bounded; on exhaustion fall back
            # to the pin (no stub) with a warning -- the audit remains the net.
            end_x = fixed if axis == "y" else end_v
            end_y = fixed if axis == "x" else end_v
            tries = 0
            while tries < 8:
                c = cell(end_x, end_y)
                owner = occupied.get(c)
                if owner is None or owner is net:
                    break
                end_v += sign * grid
                end_x = fixed if axis == "y" else end_v
                end_y = fixed if axis == "x" else end_v
                tries += 1
            else:
                active_logger.warning(
                    "deconflict_stubs: could not place a clear stub end for "
                    "pin %s of net %r; labelling on the pin instead",
                    getattr(pin, "num", "?"),
                    getattr(net, "name", "?"),
                )
                node._stub_ends[id(pin)] = Point(pin_w.x, pin_w.y)
                pin.route_pt = copy.copy(pin.pt)
                continue

            end_w = Point(round(end_x), round(end_y))
            occupied[cell(end_w.x, end_w.y)] = net
            node._stub_ends[id(pin)] = end_w
            # Protect the stub END from cleanup's trim_stubs: when A* routes
            # collinearly with a stub, split_segments cuts at the pin and the
            # outer piece (pin->end) becomes a leaf. Only pin_pts endpoints are
            # spared, so register the end too — else the deconfliction is undone
            # and the closure label at the end would dangle.
            pin_pts.append(end_w)

            # route_pt (local) maps back to the world end so the A* router wires
            # the deconflicted end. The emitted stub segment uses exact world
            # coords so its endpoints are guaranteed on-grid.
            pin.route_pt = _invert_dihedral(end_w, part.tx)
            # NetTerminal pins have no emitted symbol (channel-edge markers), so
            # a pin->end stub would leave the pin-side endpoint bare -> a
            # dangling wire end. Skip the stub: A* still routes to route_pt (the
            # deconflicted end), which carries the terminal's label, so the wire
            # ends cleanly at the labelled end.
            if is_routed and not isinstance(part, NetTerminal):
                seg = Segment(Point(pin_w.x, pin_w.y), Point(end_w.x, end_w.y))
                node.wires[net].append(seg)
                node._stub_wire_nets[id(net)].append(seg)

        # Publish the full occupancy so the A* router avoids every pin + stub
        # end of every net (routed and label-only alike).
        node._deconflict_occupied = {k: v for k, v in occupied.items() if v is not None}

    def cleanup_wires(node):
        """Try to make wire segments look prettier."""

        # Deconflict-stub mode (stage 25): the A* output is already a clean,
        # connected, axis-aligned tree, and every pin has an intentional stub to
        # a deconflicted on-grid end (registered in pin_pts so trim_stubs spares
        # it). remove_jogs (cosmetic, and nondeterministic via random.shuffle)
        # reshapes routing OFF those ends -> lost pin connections, so it is
        # skipped in this mode; the other passes run normally.
        _deconflict = getattr(node, "_deconflict_stubs", False)

        def _pt_key(pt):
            """Stable geometric sort key for a Point (determinism)."""
            return (pt.x, pt.y)

        def _seg_key(seg):
            """Stable geometric sort key for a Segment (determinism)."""
            return (seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y)

        def order_seg_points(segments):
            """Order endpoints in a horizontal or vertical segment."""
            for seg in segments:
                if seg.p2 < seg.p1:
                    seg.p1, seg.p2 = seg.p2, seg.p1

        def segments_bbox(segments):
            """Return bounding box containing the given list of segments."""
            seg_pts = list(chain(*((s.p1, s.p2) for s in segments)))
            return BBox(*seg_pts)

        def extract_horz_vert_segs(segments):
            """Separate segments and return lists of horizontal & vertical segments."""
            horz_segs = [seg for seg in segments if seg.p1.y == seg.p2.y]
            vert_segs = [seg for seg in segments if seg.p1.x == seg.p2.x]
            assert len(horz_segs) + len(vert_segs) == len(segments)
            return horz_segs, vert_segs

        def split_segments(segments, net_pin_pts):
            """Return list of net segments split into the smallest intervals without intersections with other segments."""

            # Check each horizontal segment against each vertical segment and split each one if they intersect.
            # (This clunky iteration is used so the horz/vert lists can be updated within the loop.)
            horz_segs, vert_segs = extract_horz_vert_segs(segments)
            i = 0
            while i < len(horz_segs):
                hseg = horz_segs[i]
                hseg_y = hseg.p1.y
                j = 0
                while j < len(vert_segs):
                    vseg = vert_segs[j]
                    vseg_x = vseg.p1.x
                    if (
                        hseg.p1.x <= vseg_x <= hseg.p2.x
                        and vseg.p1.y <= hseg_y <= vseg.p2.y
                    ):
                        int_pt = Point(vseg_x, hseg_y)
                        if int_pt != hseg.p1 and int_pt != hseg.p2:
                            horz_segs.append(
                                Segment(copy.copy(int_pt), copy.copy(hseg.p2))
                            )
                            hseg.p2 = copy.copy(int_pt)
                        if int_pt != vseg.p1 and int_pt != vseg.p2:
                            vert_segs.append(
                                Segment(copy.copy(int_pt), copy.copy(vseg.p2))
                            )
                            vseg.p2 = copy.copy(int_pt)
                    j += 1
                i += 1

            i = 0
            while i < len(horz_segs):
                hseg = horz_segs[i]
                hseg_y = hseg.p1.y
                for pt in net_pin_pts:
                    if pt.y == hseg_y and hseg.p1.x < pt.x < hseg.p2.x:
                        horz_segs.append(Segment(copy.copy(pt), copy.copy(hseg.p2)))
                        hseg.p2 = copy.copy(pt)
                i += 1

            j = 0
            while j < len(vert_segs):
                vseg = vert_segs[j]
                vseg_x = vseg.p1.x
                for pt in net_pin_pts:
                    if pt.x == vseg_x and vseg.p1.y < pt.y < vseg.p2.y:
                        vert_segs.append(Segment(copy.copy(pt), copy.copy(vseg.p2)))
                        vseg.p2 = copy.copy(pt)
                j += 1

            return horz_segs + vert_segs

        def merge_segments(segments):
            """Return segments after merging those that run the same direction and overlap."""

            # Preprocess the segments.
            horz_segs, vert_segs = extract_horz_vert_segs(segments)

            merged_segs = []

            # Separate horizontal segments having the same Y coord.
            horz_segs_v = defaultdict(list)
            for seg in horz_segs:
                horz_segs_v[seg.p1.y].append(seg)

            # Merge overlapping segments having the same Y coord.
            for segs in horz_segs_v.values():
                # Order segments by their starting X coord.
                segs.sort(key=lambda s: s.p1.x)
                # Append first segment to list of merged segments.
                merged_segs.append(segs[0])
                # Go thru the remaining segments looking for overlaps with the last entry on the merge list.
                for seg in segs[1:]:
                    if seg.p1.x <= merged_segs[-1].p2.x:
                        # Segments overlap, so update the extent of the last entry.
                        merged_segs[-1].p2.x = max(seg.p2.x, merged_segs[-1].p2.x)
                    else:
                        # No overlap, so append the current segment to the merge list and use it for
                        # further checks of intersection with remaining segments.
                        merged_segs.append(seg)

            # Separate vertical segments having the same X coord.
            vert_segs_h = defaultdict(list)
            for seg in vert_segs:
                vert_segs_h[seg.p1.x].append(seg)

            # Merge overlapping segments having the same X coord.
            for segs in vert_segs_h.values():
                # Order segments by their starting Y coord.
                segs.sort(key=lambda s: s.p1.y)
                # Append first segment to list of merged segments.
                merged_segs.append(segs[0])
                # Go thru the remaining segments looking for overlaps with the last entry on the merge list.
                for seg in segs[1:]:
                    if seg.p1.y <= merged_segs[-1].p2.y:
                        # Segments overlap, so update the extent of the last entry.
                        merged_segs[-1].p2.y = max(seg.p2.y, merged_segs[-1].p2.y)
                    else:
                        # No overlap, so append the current segment to the merge list and use it for
                        # further checks of intersection with remaining segments.
                        merged_segs.append(seg)

            return merged_segs

        def break_cycles(segments):
            """Remove segments to break any cycles of a net's segments."""

            # Create a dict storing set of segments adjacent to each endpoint.
            adj_segs = defaultdict(set)
            for seg in segments:
                # Add segment to set for each endpoint.
                adj_segs[seg.p1].add(seg)
                adj_segs[seg.p2].add(seg)

            # Create a dict storing the list of endpoints adjacent to each endpoint.
            adj_pts = dict()
            for pt, segs in adj_segs.items():
                # Store endpoints of all segments adjacent to endpoint, then remove the endpoint.
                adj_pts[pt] = list({p for seg in segs for p in (seg.p1, seg.p2)})
                adj_pts[pt].remove(pt)

            # Start at any endpoint and visit adjacent endpoints until all have been visited.
            # If an endpoint is seen more than once, then a cycle exists. Remove the segment forming the cycle.
            visited_pts = []  # List of visited endpoints.
            frontier_pts = list(adj_pts.keys())[:1]  # Arbitrary starting point.
            while frontier_pts:
                # Visit a point on the frontier.
                frontier_pt = frontier_pts.pop()
                visited_pts.append(frontier_pt)

                # Check each adjacent endpoint for cycles.
                for adj_pt in adj_pts[frontier_pt][:]:
                    if adj_pt in visited_pts + frontier_pts:
                        # This point was already reached by another path so there is a cycle.
                        # Break it by removing segment between frontier_pt and adj_pt.
                        loop_seg = (adj_segs[frontier_pt] & adj_segs[adj_pt]).pop()
                        segments.remove(loop_seg)
                        adj_segs[frontier_pt].remove(loop_seg)
                        adj_segs[adj_pt].remove(loop_seg)
                        adj_pts[frontier_pt].remove(adj_pt)
                        adj_pts[adj_pt].remove(frontier_pt)
                    else:
                        # First time adjacent point has been reached, so add it to frontier.
                        frontier_pts.append(adj_pt)
                        # Keep this new frontier point from backtracking to the current frontier point later.
                        adj_pts[adj_pt].remove(frontier_pt)

            return segments

        def is_pin_pt(pt):
            """Return True if the point is on one of the part pins."""
            return pt in pin_pts

        def contains_pt(seg, pt):
            """Return True if the point is contained within the horz/vert segment."""
            return seg.p1.x <= pt.x <= seg.p2.x and seg.p1.y <= pt.y <= seg.p2.y

        def trim_stubs(segments):
            """Return segments after removing stubs that have an unconnected endpoint."""

            def get_stubs(segments):
                """Return set of stub segments."""

                # For end point, the dict entry contains a list of the segments that meet there.
                stubs = defaultdict(list)

                # Process the segments looking for points that are on only a single segment.
                for seg in segments:
                    # Add the segment to the segment list of each end point.
                    stubs[seg.p1].append(seg)
                    stubs[seg.p2].append(seg)

                # Keep only the segments with an unconnected endpoint that is not on a part pin.
                stubs = {
                    segs[0]
                    for endpt, segs in stubs.items()
                    if len(segs) == 1 and not is_pin_pt(endpt)
                }
                return stubs

            trimmed_segments = set(segments[:])
            stubs = get_stubs(trimmed_segments)
            while stubs:
                trimmed_segments -= stubs
                stubs = get_stubs(trimmed_segments)
            # Return in a stable coordinate order: a set-to-list conversion is
            # id-ordered (nondeterministic across runs), which would perturb the
            # emitted wire order. (stage 24 routing determinism)
            return sorted(
                trimmed_segments,
                key=lambda s: (s.p1.x, s.p1.y, s.p2.x, s.p2.y),
            )

        def remove_jogs(net, segments, wires, net_bboxes, part_bboxes):
            """Remove jogs and staircases in wiring segments.

            Args:
                net (Net): Net whose wire segments will be modified.
                segments (list): List of wire segments for the given net.
                wires (dict): Dict of lists of wire segments indexed by nets.
                net_bboxes (dict): Dict of BBoxes for wire segments indexed by nets.
                part_bboxes (list): List of BBoxes for the placed parts.
            """

            def obstructed(segment):
                """Return true if segment obstructed by parts or segments of other nets."""

                # Obstructed if segment bbox intersects one of the part bboxes.
                segment_bbox = BBox(segment.p1, segment.p2)
                for part_bbox in part_bboxes:
                    if part_bbox.intersects(segment_bbox):
                        return True

                # BBoxes don't intersect if they line up exactly edge-to-edge.
                # So expand the segment bbox slightly so intersections with bboxes of
                # other segments will be detected.
                segment_bbox = segment_bbox.resize(Vector(2, 2))

                # Look for an overlay intersection with a segment of another net.
                for nt, nt_bbox in net_bboxes.items():
                    if nt is net:
                        # Don't check this segment with other segments of its own net.
                        continue

                    if not segment_bbox.intersects(nt_bbox):
                        # Don't check this segment against segments of another net whose
                        # bbox doesn't even intersect this segment.
                        continue

                    # Check for overlay intersectionss between this segment and the
                    # parallel segments of the other net.
                    for seg in wires[nt]:
                        if segment.p1.x == segment.p2.x == seg.p1.x == seg.p2.x:
                            # Segments are both aligned vertically on the same track X coord.
                            if segment.p1.y <= seg.p2.y and segment.p2.y >= seg.p1.y:
                                # Segments overlap so segment is obstructed.
                                return True
                        elif segment.p1.y == segment.p2.y == seg.p1.y == seg.p2.y:
                            # Segments are both aligned horizontally on the same track Y coord.
                            if segment.p1.x <= seg.p2.x and segment.p2.x >= seg.p1.x:
                                # Segments overlap so segment is obstructed.
                                return True

                # No obstructions found, so return False.
                return False

            def get_corners(segments):
                """Return dictionary of right-angle corner points and lists of associated segments."""

                # For each corner point, the dict entry contains a list of the segments that meet there.
                corners = defaultdict(list)

                # Process the segments so that any potential right-angle corner has the horizontal
                # segment followed by the vertical segment.
                horz_segs, vert_segs = extract_horz_vert_segs(segments)
                for seg in horz_segs + vert_segs:
                    # Add the segment to the segment list of each end point.
                    corners[seg.p1].append(seg)
                    corners[seg.p2].append(seg)

                # Keep only the corner points where two segments meet at right angles at a point not on a part pin.
                corners = {
                    corner: segs
                    for corner, segs in corners.items()
                    if len(segs) == 2
                    and not is_pin_pt(corner)
                    and segs[0] in horz_segs
                    and segs[1] in vert_segs
                }
                return corners

            def get_jogs(segments):
                """Yield the three segments and starting and end points of a staircase or tophat jog."""

                # Get dict of right-angle corners formed by segments.
                corners = get_corners(segments)

                # Look for segments with both endpoints on right-angle corners, indicating this segment
                # is in the middle of a three-segment staircase or tophat jog.
                for segment in segments:
                    if segment.p1 in corners and segment.p2 in corners:
                        # Get the three segments in the jog.
                        jog_segs = set()
                        jog_segs.add(corners[segment.p1][0])
                        jog_segs.add(corners[segment.p1][1])
                        jog_segs.add(corners[segment.p2][0])
                        jog_segs.add(corners[segment.p2][1])

                        # Get the points where the three-segment jog starts and stops.
                        start_stop_pts = set()
                        for seg in jog_segs:
                            start_stop_pts.add(seg.p1)
                            start_stop_pts.add(seg.p2)
                        start_stop_pts.discard(segment.p1)
                        start_stop_pts.discard(segment.p2)

                        # Send the jog that was found. Order the set-derived
                        # lists on stable geometric keys: iterating a set of
                        # Segment/Point OBJECTS is id()-ordered (varies per
                        # process), which leaked into which jog got corrected and
                        # made the rendered wires -- and, via the sheet bbox that
                        # centers the page, the PART positions -- non-reproducible.
                        yield (
                            sorted(jog_segs, key=_seg_key),
                            sorted(start_stop_pts, key=_pt_key),
                        )

            # Detect jogs in a stable order (was random.shuffle -> a seeded
            # render still diverged per process). Sorting on the geometric key
            # makes jog correction deterministic without changing its effect.
            segments.sort(key=_seg_key)

            # Get iterator for jogs.
            jogs = get_jogs(segments)

            # Search for jogs and break from the loop if a correctable jog is found or we run out of jogs.
            while True:
                # Get the segments and start-stop points for the next jog.
                try:
                    jog_segs, start_stop_pts = next(jogs)
                except StopIteration:
                    # No more jogs and no corrections made, so return segments and stop flag is true.
                    return segments, True

                # Get the start-stop points and order them so p1 < p3.
                p1, p3 = sorted(start_stop_pts)

                # These are the potential routing points for correcting the jog.
                # Either start at p1 and move vertically and then horizontally to p3, or
                # move horizontally from p1 and then vertically to p3.
                # Deterministic order (was random.shuffle): the first VALID
                # correction is applied, so a fixed order makes the reshaped wire
                # reproducible across processes.
                p2s = [Point(p1.x, p3.y), Point(p3.x, p1.y)]

                # Check each routing point to see if it leads to a valid routing.
                for p2 in p2s:
                    # Replace the three-segment jog with these two right-angle segments.
                    new_segs = [
                        Segment(copy.copy(pa), copy.copy(pb))
                        for pa, pb in ((p1, p2), (p2, p3))
                        if pa != pb
                    ]
                    order_seg_points(new_segs)

                    # Check the new segments to see if they run into parts or segments of other nets.
                    if not any((obstructed(new_seg) for new_seg in new_segs)):
                        # OK, segments are good so replace the old segments in the jog with them.
                        for seg in jog_segs:
                            segments.remove(seg)
                        segments.extend(new_segs)

                        # Return updated segments and set stop flag to false because segments were modified.
                        return segments, False

        # Get part bounding boxes so parts can be avoided when modifying net segments.
        part_bboxes = [p.bbox * p.tx for p in node.parts]

        # Get dict of bounding boxes for the nets in this node.
        net_bboxes = {net: segments_bbox(segs) for net, segs in node.wires.items()}

        # Get locations for part pins of each net. (For use when splitting net segments.)
        net_pin_pts = dict()
        for net in node.wires.keys():
            net_pin_pts[net] = [
                (pin.pt * pin.part.tx).round() for pin in node.get_internal_pins(net)
            ]

        # Do a generalized cleanup of the wire segments of each net.
        for net, segments in node.wires.items():
            # Round the wire segment endpoints to integers.
            segments = [seg.round() for seg in segments]

            # Keep only non zero-length segments.
            segments = [seg for seg in segments if seg.p1 != seg.p2]

            # Make sure the segment endpoints are in the right order.
            order_seg_points(segments)

            # Merge colinear, overlapping segments. Also removes any duplicated segments.
            segments = merge_segments(segments)

            # Split intersecting segments.
            segments = split_segments(segments, net_pin_pts[net])

            # Break loops of segments.
            segments = break_cycles(segments)

            # Keep only non zero-length segments.
            segments = [seg for seg in segments if seg.p1 != seg.p2]

            # Trim genuinely-dangling wire stubs. In deconflict mode the
            # intended pin->end stubs are spared because their ends are
            # registered in pin_pts (is_pin_pt), so only true A*/merge artifacts
            # are removed.
            segments = trim_stubs(segments)

            node.wires[net] = segments

        # Remove jogs in the wire segments of each net (skipped in deconflict
        # mode -- cosmetic + nondeterministic, and it disconnects stub ends).
        keep_cleaning = not _deconflict
        while keep_cleaning:
            keep_cleaning = False

            for net, segments in node.wires.items():
                while True:
                    # Split intersecting segments.
                    segments = split_segments(segments, net_pin_pts[net])

                    # Remove unnecessary wire jogs.
                    segments, stop = remove_jogs(
                        net, segments, node.wires, net_bboxes, part_bboxes
                    )

                    # Keep only non zero-length segments.
                    segments = [seg for seg in segments if seg.p1 != seg.p2]

                    # Merge segments made colinear by removing jogs.
                    segments = merge_segments(segments)

                    # Split intersecting segments.
                    segments = split_segments(segments, net_pin_pts[net])

                    # Keep only non zero-length segments.
                    segments = [seg for seg in segments if seg.p1 != seg.p2]

                    # Trim wire stubs caused by removing jogs.
                    segments = trim_stubs(segments)

                    if stop:
                        # Break from loop once net segments can no longer be improved.
                        break

                    # Recalculate the net bounding box after modifying its segments.
                    net_bboxes[net] = segments_bbox(segments)

                    keep_cleaning = True

                # Merge segments made colinear by removing jogs.
                segments = merge_segments(segments)

                # Update the node net's wire with the cleaned version.
                node.wires[net] = segments

    def add_junctions(node):
        """Add X & T-junctions where wire segments in the same net meet."""

        def find_junctions(route):
            """Find junctions where segments of a net intersect.

            Args:
                route (List): List of Segment objects.

            Returns:
                List: List of Points, one for each junction.

            Notes:
                You must run merge_segments() before finding junctions
                or else the segment endpoints might not be ordered
                correctly with p1 < p2.
            """

            # Separate route into vertical and horizontal segments.
            horz_segs = [seg for seg in route if seg.p1.y == seg.p2.y]
            vert_segs = [seg for seg in route if seg.p1.x == seg.p2.x]

            junctions = []

            # Check each pair of horz/vert segments for an intersection, except
            # where they form a right-angle turn.
            for hseg in horz_segs:
                hseg_y = hseg.p1.y  # Horz seg Y coord.
                for vseg in vert_segs:
                    vseg_x = vseg.p1.x  # Vert seg X coord.
                    if (hseg.p1.x < vseg_x < hseg.p2.x) and (
                        vseg.p1.y <= hseg_y <= vseg.p2.y
                    ):
                        # The vert segment intersects the interior of the horz seg.
                        junctions.append(Point(vseg_x, hseg_y))
                    elif (vseg.p1.y < hseg_y < vseg.p2.y) and (
                        hseg.p1.x <= vseg_x <= hseg.p2.x
                    ):
                        # The horz segment intersects the interior of the vert seg.
                        junctions.append(Point(vseg_x, hseg_y))

            return junctions

        for net, segments in node.wires.items():
            # Add X & T-junctions between segments in the same net.
            junctions = find_junctions(segments)
            node.junctions[net].extend(junctions)

    def rmv_routing_stuff(node):
        """Remove attributes added to parts/pins during routing."""

        rmv_attr(node.parts, ("left_track", "right_track", "top_track", "bottom_track"))
        for part in node.parts:
            rmv_attr(part.pins, ("route_pt", "face"))

    def route_internal_nets_astar(node, internal_nets, **options):
        """Per-net Hanan-grid A* router (stage 24).

        Replaces the face-capacity switchbox stack. Schematic wires may cross
        freely (a crossing without a junction dot is "no connection" -- KiCad's
        wire-hop glyphs are cosmetic), so there is NO capacity/congestion
        constraint. The only hard rules:

          * part BODY bboxes (``lbl_bbox``) are obstacles -- routes never enter
            a part interior;
          * two DIFFERENT nets must never run COLINEAR-overlapping (that would be
            a silent Blocker-B-class merge), and a wire must not pass through or
            end on another net's pin route-point;
          * perpendicular crossings of other nets are unconstrained.

        Routes between the pin route-points already pushed to the part bbox edges
        by ``add_routing_points`` (whose pin->edge stub wires are already in
        ``node.wires``). Appends world-coordinate, axis-aligned ``Segment``s to
        ``node.wires[net]``; ``cleanup_wires`` + ``add_junctions`` finish the job.
        A net whose points cannot be connected (only possible for a fully
        enclosed route-point) is stubbed to labels on its own -- a per-net
        fallback finer-grained than the per-sheet isolation in ``route``.

        Deterministic by construction: nets sorted by name, MST/A* tie-breaks
        keyed, A* heap ordered by a monotone counter (no RNG anywhere).
        """
        from skidl.logger import active_logger

        MARGIN = 2 * GRID  # extra Hanan lines for detour clearance around corners
        TURN = float(GRID)  # bend penalty (favors straighter routes)
        deconflict = options.get("deconflict_stubs", False)

        # Process nets in a stable name order for determinism.
        def _net_key(net):
            return (getattr(net, "name", "") or "", id(net))

        nets = sorted(internal_nets, key=_net_key)

        # World obstacles: each part's labeled bbox. Route-points sit ON these
        # edges, so a STRICT-interior test keeps them (and boundary-hugging
        # wires) legal. In deconflict mode grow each obstacle OUTWARD to the
        # grid so every Hanan line is on-grid -> every routed corner is on-grid
        # (kills endpoint_off_grid).
        obstacles = []
        for part in node.parts:
            b = (part.lbl_bbox * part.tx).round()
            if deconflict:
                g = float(GRID)
                import math

                lo_x = math.floor(b.min.x / g) * g
                lo_y = math.floor(b.min.y / g) * g
                hi_x = math.ceil(b.max.x / g) * g
                hi_y = math.ceil(b.max.y / g) * g
                obstacles.append((lo_x, lo_y, hi_x, hi_y))
            else:
                obstacles.append((b.min.x, b.min.y, b.max.x, b.max.y))

        # Route-points (world) per net + an occupancy map so no wire routes
        # through / ends on ANOTHER net's pin route-point. In deconflict mode
        # seed it with EVERY pin + stub end of EVERY net (published by
        # add_deconflicted_stubs) so routed wires also avoid label-only nets'
        # stub ends.
        occupied = {}  # (x, y) -> the net that owns this pin route-point
        if deconflict:
            for (x, y), onet in getattr(node, "_deconflict_occupied", {}).items():
                occupied[(round(x), round(y))] = onet
        net_points = {}
        for net in nets:
            pts = []
            for pin in node.get_internal_pins(net):
                wp = (pin.route_pt * pin.part.tx).round()
                key = (wp.x, wp.y)
                pts.append(key)
                occupied.setdefault(key, net)
            net_points[net] = list(dict.fromkeys(pts))  # de-dup, keep order

        # Registry of already-routed segments (incl. the pin->edge stubs added by
        # add_routing_points) keyed by axis+coord, for colinear-overlap veto.
        foreign_h = defaultdict(list)  # y -> list of (xlo, xhi, net)
        foreign_v = defaultdict(list)  # x -> list of (ylo, yhi, net)

        def register(net, x0, y0, x1, y1):
            if y0 == y1:
                foreign_h[y0].append((min(x0, x1), max(x0, x1), net))
            elif x0 == x1:
                foreign_v[x0].append((min(y0, y1), max(y0, y1), net))

        for net, segs in node.wires.items():
            for seg in segs:
                p1, p2 = seg.p1.round(), seg.p2.round()
                register(net, p1.x, p1.y, p2.x, p2.y)

        # Fold every foreign pin coordinate into the Hanan grid so a
        # perpendicular pass-through can't skip over a pin un-vertexed.
        occ_xs = {p[0] for p in occupied}
        occ_ys = {p[1] for p in occupied}

        for net in nets:
            pts = net_points[net]
            if len(pts) < 2:
                continue  # single distinct point: nothing to route (label/stub)
            ok = True
            new_segs = []
            for i, j in _mst_edges(pts):
                path = _astar_pair(
                    pts[i],
                    pts[j],
                    obstacles,
                    occupied,
                    foreign_h,
                    foreign_v,
                    net,
                    extra_xs=occ_xs,
                    extra_ys=occ_ys,
                    margin=MARGIN,
                    turn=TURN,
                )
                if path is None:
                    ok = False
                    break
                for k in range(len(path) - 1):
                    (x0, y0), (x1, y1) = path[k], path[k + 1]
                    if (x0, y0) != (x1, y1):
                        new_segs.append((x0, y0, x1, y1))
            if not ok:
                # Per-net fallback: stub THIS net to labels; siblings stay wired.
                active_logger.warning(
                    f"A* could not route net {getattr(net, 'name', '?')!r} "
                    f"(enclosed route-point); stubbing it to labels while the "
                    f"other nets on this sheet stay wired"
                )
                net._stub = True
                for pin in net.get_pins():
                    pin.stub = True
                if deconflict:
                    # Keep the deconflicted pin->end stubs (and node._stub_ends)
                    # so the phase-2 closure labeller still anchors a label per
                    # pin island; only the ROUTED segments are discarded.
                    node.wires[net] = list(node._stub_wire_nets.get(id(net), []))
                else:
                    node.wires[net] = []  # drop its pin->edge stub wires too
                continue
            for x0, y0, x1, y1 in new_segs:
                node.wires[net].append(Segment(Point(x0, y0), Point(x1, y1)))
                register(net, x0, y0, x1, y1)

    def route(node, tool=None, **options):
        """Route the wires between part pins in this node and its children.

        Routing strategy (stage 24): per-net Hanan-grid A* with free crossings
        (``route_internal_nets_astar``) -- the face-capacity switchbox stack was
        removed. Steps: extend pin routing points to part-bbox edges, A*-route
        every internal net, then clean up wires and add intra-net junctions.

        Args:
            node (Node): Hierarchical node containing the parts to be connected.
            tool (str): Backend tool for schematics.
            options (dict, optional): Dictionary of options and values:
                "allow_routing_failure", "draw", "draw_all_terminals", "show_capacities",
                "draw_switchbox", "draw_routing", "draw_channels"
        """

        # Inject the constants for the backend tool into this module.
        import skidl
        from skidl.tools import tool_modules

        tool = tool or skidl.config.tool
        this_module = sys.modules[__name__]
        this_module.__dict__.update(tool_modules[tool].constants.__dict__)

        # Default the seed so an unset seed is REPRODUCIBLE across processes (see
        # place.route). seed=None stays the explicit randomized-exploration opt-in.
        random.seed(options.get("seed", 42))

        # Remove any stuff leftover from a previous place & route run.
        node.rmv_routing_stuff()

        # First, recursively route any children of this node.
        # TODO: Child nodes are independent so could they be processed in parallel?
        #
        # Per-sheet routing-failure ISOLATION (stage 19 router robustness): child
        # sheets are independent, so a routing failure on ONE dense sheet should
        # NOT collapse the whole design to labels-only. If a child fails to route,
        # stub just that child's (subtree's) nets to labels and re-route it (which
        # then trivially succeeds with no wires), keeping every other sheet wired.
        # Opt out with isolate_sheet_routing_failure=False. This is only reached
        # under auto_stub (labels are the intended fallback there).
        isolate = options.get("isolate_sheet_routing_failure", True) and options.get(
            "auto_stub", False
        )
        for child in node.children.values():
            try:
                child.route(tool=tool, **options)
            except RoutingFailure as e:
                if not isolate:
                    raise
                from skidl.logger import active_logger

                active_logger.warning(
                    f"routing failed on sheet {getattr(child, 'name', '?')!r} "
                    f"({type(e).__name__}: {e}); stubbing that sheet's nets to "
                    f"labels and keeping the other sheets wired"
                )
                _stub_node_subtree_nets(child)
                child.route(tool=tool, **options)

        # Exit if no parts to route in this node.
        if not node.parts:
            return

        # Get all the nets that have one or more pins within this node.
        internal_nets = node.get_internal_nets()

        deconflict = options.get("deconflict_stubs", False)

        # Exit if no nets to route. In deconflict mode, a sheet whose only nets
        # are cross-sheet / label-only still needs its pins stubbed out to
        # deconflicted labelled ends -- get_internal_nets() skips stub pins, so
        # such a sheet reports no internal nets, but bailing here would leave its
        # pins rendered as bare labels ON the part body (a connector/decoupling
        # sheet reads as "unstubbed"). Only bail in the classic path; let the
        # deconflict stub pass run so those pins get proper stubs (the A* router
        # then has nothing to route, which is fine).
        if not internal_nets and not deconflict:
            return

        try:
            if deconflict:
                # Stage-25: on-grid, world-unique stub end per non-power pin
                # (snap retired). Publishes node._deconflict_occupied for the
                # router and node._stub_ends for the phase-2 closure labeller.
                node.add_deconflicted_stubs(internal_nets, **options)
            else:
                # Extend routing points of part pins to the edges of their
                # bounding boxes (adds the pin->edge stub wires to node.wires).
                node.add_routing_points(internal_nets)

            # Route every internal net with the per-net Hanan-grid A* router.
            node.route_internal_nets_astar(internal_nets, **options)

            # Now clean-up the wires and add intra-net junctions.
            node.cleanup_wires()
            node.add_junctions()

            # Remove any stuff leftover from this place & route run.
            node.rmv_routing_stuff()

        except RoutingFailure:
            # Remove any stuff leftover from this place & route run.
            node.rmv_routing_stuff()
            # Re-raise the ORIGINAL failure (bare `raise`) so its message and the
            # underlying GlobalRoutingFailure/SwitchboxRoutingFailure survive for
            # diagnosis and per-sheet isolation. (stage 19 router robustness)
            raise
