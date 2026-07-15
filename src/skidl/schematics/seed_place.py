# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Constructive seed placement for the force-directed schematic placer (stage 19).

This module replaces ``random_placement`` as the INITIAL state fed to the
force-directed placer. Instead of scattering parts randomly, it builds a
part-adjacency graph from the *wired* nets only, picks a graph-central part as
the seed, and grows the placement outward best-first — placing each part in the
direction its connecting pin *faces* (KiCad pin geometry already encodes which
way inputs/outputs/rails point). The force-directed refiner still runs after;
we only change where it starts.

Design constraints (Stage-19 plan, Phase B):

* **Pure functions, no I/O.** Imports are limited to ``skidl.geometry`` and the
  stdlib — NO ``skidl.tools.*`` (a ``schematics`` module must not depend on a
  specific tool backend) and no KiCad symbol libraries. Everything is typed
  against a *structural* interface so the unit tests exercise it with plain fake
  objects:

    - part-like: ``.ref``, ``.pins`` (iterable), ``.tx`` (Tx),
      ``.place_bbox`` (BBox, local frame), ``.orientation_locked`` (bool)
    - pin-like:  ``.pt`` (Point, local frame, mils), ``.orientation``
      (letter R/U/L/D, pointing INWARD — see below), ``.stub`` (bool),
      ``.net`` (net-like or None)
    - net-like:  ``.name``, ``.pins`` (iterable), ``.stub``
      (read via ``getattr(net,"stub",False)``), ``.drive``

* **Determinism.** Every tie-break is keyed on the part ``ref`` string, so two
  runs on the same input produce bit-identical placements. This module never
  touches the ``random`` module.

THE +180 / OUTWARD-FACE CONVENTION (the #1 correctness risk — see reference §3):
after ``preprocess_circuit`` a pin's ``.orientation`` letter is derived from the
raw KiCad angle, which points INWARD (from the wire-attach point toward the
body). So the letter points INWARD and the OUTWARD wire face (the direction a
wire leaves, hence the direction to place a neighbor) is the OPPOSITE of the
letter's vector::

    R -> outward (-1, 0)   (letter R => wire attaches on the LEFT => faces left)
    L -> outward (+1, 0)
    U -> outward (0, -1)
    D -> outward (0, +1)

Locked by the ADA4817 table in ``test_seed_place.py``.
"""

import heapq
import re

from skidl.geometry import (
    BBox,
    Point,
    Tx,
    Vector,
    tx_rot_0,
    tx_rot_90,
    tx_rot_180,
    tx_rot_270,
)

# Copied verbatim from skidl/tools/kicad9/gen_schematic.py:56 (do NOT import it —
# that would make schematics depend on a tool backend, the wrong layering
# direction). Keep in sync if the source pattern changes.
POWER_NET_RE = re.compile(
    r"^(\+\d[\d.]*V[\d]*|GND|AGND|DGND|PGND|VCC|VDD|VSS|VEE|VBUS|VBAT|AVCC|AVDD|DVCC|DVDD)$",
    re.IGNORECASE,
)

# Pin-orientation letter (INWARD) -> LOCAL outward unit vector (see module doc).
_LOCAL_OUTWARD = {
    "R": Vector(-1, 0),
    "L": Vector(1, 0),
    "U": Vector(0, -1),
    "D": Vector(0, 1),
}

# Rotation quadrant (degrees CCW) -> Tx.
_ROT_TX = {0: tx_rot_0, 90: tx_rot_90, 180: tx_rot_180, 270: tx_rot_270}

_DEFAULT_GRID = 50  # mils (kicad9 GRID constant); overridable via seed_placement.

# Debug-only hook. When set (see skidl.schematics.debug_anim.record_placements),
# it is called once per part right after its tx is finalized, in placement order.
# None in all normal use, so the placer stays pure/no-I/O (the only cost on the
# production path is one `is not None` check per placed part). Single-threaded
# debug use only -- placement is not run concurrently.
_PLACEMENT_OBSERVER = None


def _notify_placed(part):
    """Fire the optional placement observer for ``part`` (no-op when unset)."""
    if _PLACEMENT_OBSERVER is not None:
        _PLACEMENT_OBSERVER(part)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _ref(part):
    """Deterministic sort key for a part: its ref string ('' if missing)."""
    return getattr(part, "ref", "") or ""


def _pins(obj):
    """Return a list of an object's pins (parts and nets both expose .pins)."""
    return list(getattr(obj, "pins", []) or [])


def _is_power_drive(drive):
    """True if a net's ``drive`` is the POWER level.

    Duck-typed on the enum member name so this module needs no import from
    skidl core (skidl.pin_drives.POWER.name == 'POWER'). Fakes in the tests set
    the real enum or any object exposing ``.name == 'POWER'``.
    """
    return getattr(drive, "name", None) == "POWER"


def _net_is_power(net):
    name = getattr(net, "name", "") or ""
    return bool(POWER_NET_RE.match(name)) or _is_power_drive(
        getattr(net, "drive", None)
    )


def _net_stubbed(net):
    if getattr(net, "stub", False):
        return True
    return any(getattr(p, "stub", False) for p in _pins(net))


def _unit(vec):
    """Snap a near-axis-aligned vector to a clean unit vector."""
    n = vec.norm
    return Vector(round(n.x), round(n.y))


def _half_extent(bbox, direction):
    """Half the bbox size along an axis-aligned unit ``direction``."""
    if abs(direction.x) >= abs(direction.y):
        return bbox.w / 2.0
    return bbox.h / 2.0


def _rotated_bbox(local_bbox, rot_tx):
    """Local bbox transformed by rotation only (no translation)."""
    return local_bbox * rot_tx.no_translate()


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #


def build_wired_graph(parts, nets, max_fanout=3):
    """Build a part-adjacency graph from WIRED nets only.

    Returns ``adj``: ``{part: {neighbor_part: [(my_pin, their_pin), ...]}}``.

    A net contributes edges only if ALL of:
      * it is not stubbed (net-level or any pin-level ``.stub``),
      * its ``drive`` is not POWER and its name does not match ``POWER_NET_RE``,
      * ``2 <= (# of its non-stub pins on these parts) <= max_fanout``
        (GND cliques and high-fanout buses are excluded — they make the graph
        center meaningless).

    Each qualifying net becomes a clique over its pins, so two parallel nets
    between the same pair of parts (Rf ∥ Cf) produce two pin-pairs on that edge.
    """
    part_set = set(parts)
    adj = {p: {} for p in parts}

    for net in nets:
        if _net_stubbed(net) or _net_is_power(net):
            continue
        real_pins = [
            pin
            for pin in _pins(net)
            if getattr(pin, "part", None) in part_set
            and not getattr(pin, "stub", False)
        ]
        if not (2 <= len(real_pins) <= max_fanout):
            continue
        for i in range(len(real_pins)):
            for j in range(i + 1, len(real_pins)):
                pi, pj = real_pins[i], real_pins[j]
                if pi.part is pj.part:
                    continue
                adj[pi.part].setdefault(pj.part, []).append((pi, pj))
                adj[pj.part].setdefault(pi.part, []).append((pj, pi))
    return adj


def k_core(adj):
    """Core number per part (iterative min-degree peeling). Stdlib only.

    Returns ``{part: core_number}``. Tie-broken by ref for a deterministic
    peeling order (the core numbers themselves are order-independent).
    """
    nbr = {p: set(neighbors.keys()) for p, neighbors in adj.items()}
    deg = {p: len(nbr[p]) for p in nbr}
    remaining = set(nbr.keys())
    core = {}
    k = 0
    while remaining:
        v = min(remaining, key=lambda p: (deg[p], _ref(p)))
        k = max(k, deg[v])
        core[v] = k
        remaining.discard(v)
        for u in nbr[v]:
            if u in remaining:
                deg[u] -= 1
    return core


def pick_center(adj, cores, pin_counts):
    """Pick the seed part: argmax by (core, pin_count, graph degree), smallest ref.

    The plan's key is (core, pin_count, ref); graph **degree** is added as a
    finer tiebreak BEFORE ref so the actual hub wins when several parts share
    the same core and pin count (e.g. equal-pin passives around one busy node) —
    strictly better centering, still fully deterministic (Phase-B finding).

    Special case for chain/path graphs (no 2-core AND max degree <= 2): pick an
    ENDPOINT (a degree-1 leaf) so the chain lays out linearly. A star (a hub
    with degree >= 3) is NOT a chain — the hub wins there.
    """
    if not adj:
        return None
    max_core = max(cores.values())
    max_degree = max(len(adj[p]) for p in adj)
    if max_core <= 1 and max_degree <= 2:
        leaves = [p for p in adj if len(adj[p]) == 1]
        candidates = leaves or list(adj)
    else:
        candidates = list(adj)
    return min(
        candidates,
        key=lambda p: (
            -cores.get(p, 0),
            -pin_counts.get(p, 0),
            -len(adj[p]),
            _ref(p),
        ),
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def outward_face(pin, part_tx):
    """Unit Vector (PLACED frame) pointing the way a wire leaves ``pin``.

    Letter -> local outward vector (OPPOSITE of the inward-pointing letter),
    then rotated through ``part_tx``'s linear part (no translation).
    """
    local = _LOCAL_OUTWARD.get(getattr(pin, "orientation", None))
    if local is None:
        return Vector(1, 0)  # unknown orientation: default right, force fixes it
    return _unit(local * part_tx.no_translate())


def _best_rotation(part, pin, target_dir):
    """Quadrant (0/90/180/270) whose ``outward_face(pin)`` best matches target.

    Returns the quadrant maximizing the dot product with ``target_dir`` (exact
    match for axis-aligned targets), smallest quadrant on ties.
    """
    best_q, best_dot = 0, None
    for q in (0, 90, 180, 270):
        face = outward_face(pin, _ROT_TX[q])
        dot = face.x * target_dir.x + face.y * target_dir.y
        if best_dot is None or dot > best_dot:
            best_dot, best_q = dot, q
    return best_q


def _part_world_bbox(part, tx):
    """The part's placement bbox transformed to the world frame by ``tx``."""
    return part.place_bbox * tx


def choose_slot(part, placed_info, adj, gap, grid):
    """Choose (origin Point, rotation quadrant) for ``part`` given placed nbrs.

    Places ``part`` outward from each placed neighbor along that neighbor's
    connecting-pin face, at the centroid of the per-neighbor slots; rotates so
    ``part``'s own connecting pin faces back toward the primary neighbor. On a
    bbox overlap, fans out along the axis perpendicular to the primary
    direction. Returns ``None`` if no neighbor is placed yet.
    """
    neighbors = [(n, pairs) for n, pairs in adj[part].items() if n in placed_info]
    if not neighbors:
        return None

    # Primary neighbor: most pin-pairs, smallest ref on ties.
    primary_n, primary_pairs = min(
        neighbors, key=lambda np_: (-len(np_[1]), _ref(np_[0]))
    )
    p_pin, n_pin = primary_pairs[0]
    dir_primary = outward_face(n_pin, placed_info[primary_n]["tx"])

    # Rotation: make part's own pin face back toward the primary neighbor.
    if getattr(part, "orientation_locked", False):
        rot = None
        rot_tx = part.tx.no_translate()
    else:
        rot = _best_rotation(part, p_pin, -dir_primary)
        rot_tx = _ROT_TX[rot]

    he_part = _half_extent(_rotated_bbox(part.place_bbox, rot_tx), dir_primary)

    # Per-neighbor slot -> centroid.
    slots = []
    for n, pairs in neighbors:
        pp, nn = pairs[0]
        d = outward_face(nn, placed_info[n]["tx"])
        he_n = _half_extent(placed_info[n]["wbbox"], d)
        he_p_d = _half_extent(_rotated_bbox(part.place_bbox, rot_tx), d)
        slots.append(placed_info[n]["origin"] + d * (he_n + gap + he_p_d))
    origin = _centroid(slots)

    # Collision resolution along the axis perpendicular to the primary direction.
    # Try offsets 0, +1, -1, +2, -2, ... times (part size + gap).
    perp = Vector(-dir_primary.y, dir_primary.x)
    step = 2 * he_part + gap
    for i in range(50):
        if i == 0:
            offset = Vector(0, 0)
        else:
            mult = (i + 1) // 2
            sign = 1 if (i % 2 == 1) else -1
            offset = perp * (sign * step * mult)
        cand = (origin + offset).snap(grid)
        cand_tx = rot_tx.move(cand)
        wbbox = _part_world_bbox(part, cand_tx)
        if not _overlaps_any(wbbox, placed_info):
            return cand, rot
    # Give up after the bounded search: stack at the centroid; force-directed
    # will separate the residual overlap.
    return origin.snap(grid), rot


def _centroid(points):
    if not points:
        return Point(0, 0)
    sx = sum(p.x for p in points) / len(points)
    sy = sum(p.y for p in points) / len(points)
    return Point(sx, sy)


def _overlaps_any(wbbox, placed_info):
    for info in placed_info.values():
        if wbbox.intersects(info["wbbox"]):
            return True
    return False


# --------------------------------------------------------------------------- #
# Growth order
# --------------------------------------------------------------------------- #


def _bfs_depths(adj, center):
    depth = {center: 0}
    frontier = [center]
    while frontier:
        nxt = []
        for p in frontier:
            for nb in adj[p]:
                if nb not in depth:
                    depth[nb] = depth[p] + 1
                    nxt.append(nb)
        frontier = nxt
    return depth


def grow_order(adj, center):
    """Yield parts in placement order, best-first from ``center``.

    Priority key = (-#distinct placed neighbors, bfs_depth_from_center, ref).
    A part bridging two already-placed parts is pulled forward (feedback/loop
    handling) rather than following a pure BFS tree. Lazy re-push keeps keys
    current as the placed set grows. Any parts unreachable from ``center``
    (disconnected within the group) are yielded last, by ref.
    """
    if center is None:
        for p in sorted(adj, key=_ref):
            yield p
        return

    depth = _bfs_depths(adj, center)
    placed = set()

    def placed_neighbor_count(p):
        return sum(1 for nb in adj[p] if nb in placed)

    def key(p):
        return (-placed_neighbor_count(p), depth.get(p, 1 << 30), _ref(p))

    yield center
    placed.add(center)

    heap = []
    for nb in adj[center]:
        heapq.heappush(heap, (key(nb), _ref(nb), nb))

    while heap:
        k, _r, p = heapq.heappop(heap)
        if p in placed:
            continue
        cur = key(p)
        if cur != k:  # stale key: re-push with the refreshed priority
            heapq.heappush(heap, (cur, _ref(p), p))
            continue
        yield p
        placed.add(p)
        for nb in adj[p]:
            if nb not in placed:
                heapq.heappush(heap, (key(nb), _ref(nb), nb))

    # Parts not reachable from center (isolated or in a disjoint sub-part).
    for p in sorted(adj, key=_ref):
        if p not in placed:
            yield p
            placed.add(p)


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def seed_placement(
    parts, nets, skip=None, max_fanout=3, gap=None, grid=None, **options
):
    """Deterministically seed ``part.tx`` for a connected group of parts.

    Mirrors ``random_placement``'s contract: mutates ``part.tx`` only, never
    consumes the ``random`` module. ``skip(part) -> bool`` filters out parts the
    caller places separately (e.g. NetTerminals). Parts with no wired edges are
    laid out in a deterministic non-overlapping row (never randomly).
    """
    if skip is None:
        skip = lambda p: False  # noqa: E731
    if grid is None:
        grid = _DEFAULT_GRID
    if gap is None:
        gap = 2 * grid  # 100..? default 2*GRID mils; tuned in Phase D.

    targets = [p for p in parts if not skip(p)]
    if not targets:
        return

    adj = build_wired_graph(targets, nets, max_fanout=max_fanout)
    has_edges = any(neighbors for neighbors in adj.values())
    if not has_edges:
        _fallback_row(targets, gap, grid)
        return

    cores = k_core(adj)
    pin_counts = {p: len(_pins(p)) for p in adj}
    center = pick_center(adj, cores, pin_counts)

    placed_info = {}
    _place(center, None, _ROT_TX[0], Point(0, 0), placed_info)

    row_isolated = []
    for p in grow_order(adj, center):
        if p is center or p in placed_info:
            continue
        result = choose_slot(p, placed_info, adj, gap, grid)
        if result is None:
            # No placed neighbor (isolated within the group): defer to a row.
            row_isolated.append(p)
            continue
        origin, rot = result
        rot_tx = _ROT_TX[rot] if rot is not None else p.tx.no_translate()
        _place(p, rot, rot_tx, origin, placed_info)

    if row_isolated:
        _fallback_row(row_isolated, gap, grid, start_y=_below(placed_info, gap))


def _place(part, rot, rot_tx, origin, placed_info):
    """Set part.tx to put local origin at ``origin`` with rotation, and record it."""
    if getattr(part, "orientation_locked", False):
        tx = part.tx.no_translate().move(origin)
    else:
        tx = rot_tx.move(origin)
    part.tx = tx
    placed_info[part] = {
        "origin": Point(origin.x, origin.y),
        "tx": tx,
        "wbbox": part.place_bbox * tx,
    }
    _notify_placed(part)


def _fallback_row(parts, gap, grid, start_y=0):
    """Deterministic left-to-right non-overlapping row, ordered by ref."""
    x = 0
    for part in sorted(parts, key=_ref):
        origin = Point(x, start_y).snap(grid)
        if getattr(part, "orientation_locked", False):
            tx = part.tx.no_translate().move(origin)
        else:
            tx = _ROT_TX[0].move(origin)
        part.tx = tx
        _notify_placed(part)
        w = part.place_bbox.w if part.place_bbox.w else grid
        x += w + gap


def _below(placed_info, gap):
    """A y-coordinate below all currently-placed bboxes."""
    if not placed_info:
        return 0
    return min(info["wbbox"].min.y for info in placed_info.values()) - gap
