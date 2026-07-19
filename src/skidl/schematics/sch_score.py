# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Pure crossing / HPWL scorer for schematic placement.

Ported from ``skidl-layout``'s PCB placement scorer
(``skidl_layout/scoring.py``): the schematic placer historically had *no*
placement-time cost function at all (it ran one deterministic constructive pass
and left crossings to the A* router, which cannot move parts). This module gives
the schematic placer the same crossing-count / half-perimeter-wire-length (HPWL)
measurement the layout engine uses, so candidate seed strategies can be scored
and the best kept (see ``place.place_connected_parts``'s ``place_score_select``
bake-off).

Design constraints (mirrors ``seed_place.py`` / ``relax_place.py``):

* **Pure functions, no I/O, no ``skidl.tools.*`` imports.** A ``schematics``
  module must not depend on a specific tool backend. Everything is typed against
  a *structural* interface so unit tests exercise it with plain fake objects:

    - part-like: ``.ref`` (str), ``.pins`` (iterable), ``.tx`` (Tx with
      ``.dx``/``.dy``); pins may carry ``.place_pt`` (Point) during placement.
    - pin-like:  ``.place_pt`` (Point, local frame) when placed, ``.part``.
    - net-like:  ``.name``, ``.pins`` (iterable of pin-like).
    - node-like: ``.parts`` (iterable), ``.get_internal_nets()``, ``.children``
      (mapping to child node-likes).

* **Determinism.** Every dict build and every tie-break is keyed on the part
  ``ref`` string (never ``id()`` or unsorted set iteration), so two runs on the
  same input produce identical scores. The crossing geometry below is copied
  verbatim from the layout engine so the numpy path stays bit-identical to the
  scalar loop (the layout tests depend on that; the ported tests here check it
  too).

The coordinate model matches the placer's own: a pin's placement point is
``pin.place_pt * pin.part.tx`` (see ``place.net_force_dist``); a part's
representative point is the mean of its placed pin points, falling back to the
part's ``tx`` origin ``(tx.dx, tx.dy)`` when no pin carries a ``place_pt`` (e.g.
after ``rmv_placement_stuff`` strips them at reporting time).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Crossing geometry -- copied VERBATIM from skidl_layout/scoring.py so the
# numpy path stays bit-identical to the scalar loop. Do NOT alter the
# orientation-product operand order.
# ---------------------------------------------------------------------------


def _segment_intersects(a1, a2, b1, b2) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    return o1 * o2 < 0 and o3 * o4 < 0


def _count_segment_crossings_loop(segments) -> int:
    crossings = 0
    for idx, (a_ref, b_ref, a1, a2) in enumerate(segments):
        for c_ref, d_ref, b1, b2 in segments[idx + 1:]:
            if {a_ref, b_ref}.intersection({c_ref, d_ref}):
                continue
            if _segment_intersects(a1, a2, b1, b2):
                crossings += 1
    return crossings


# Vectorize only when the O(S^2) loop actually costs something.
_VECTORIZED_CROSSINGS_MIN = 40


def _count_segment_crossings_numpy(segments):
    """Vectorized exact equivalent of _count_segment_crossings_loop.

    Returns the crossing count, or None if numpy is unavailable so the caller
    falls back to the loop. Mirrors _segment_intersects exactly: a pair counts
    iff o1*o2 < 0 and o3*o4 < 0 (strict -- collinear/touching never counts). The
    orientation products are computed in the same operand order as the scalar
    predicate, so the float64 result is bit-identical.
    """
    try:
        import numpy as np
    except Exception:
        return None

    n = len(segments)
    # Endpoints and integer ref ids per segment.
    a1x = np.empty(n); a1y = np.empty(n)
    a2x = np.empty(n); a2y = np.empty(n)
    ref_ids: dict[str, int] = {}
    sa = np.empty(n, dtype=np.int64)
    sb = np.empty(n, dtype=np.int64)
    for i, (a_ref, b_ref, p1, p2) in enumerate(segments):
        a1x[i], a1y[i] = p1
        a2x[i], a2y[i] = p2
        sa[i] = ref_ids.setdefault(a_ref, len(ref_ids))
        sb[i] = ref_ids.setdefault(b_ref, len(ref_ids))

    dx = a2x - a1x            # (S,) segment direction x
    dy = a2y - a1y            # (S,) segment direction y

    # [i,j] deltas. dA1{x,y}[i,j] = A1[j] - A1[i]; b{x,y}[i,j] = A2[i] - A1[j].
    dA1x = a1x[None, :] - a1x[:, None]
    dA1y = a1y[None, :] - a1y[:, None]
    dA2x1 = a2x[None, :] - a1x[:, None]
    dA2y1 = a2y[None, :] - a1y[:, None]
    bx = a2x[:, None] - a1x[None, :]
    by = a2y[:, None] - a1y[None, :]

    o1 = dx[:, None] * dA1y - dy[:, None] * dA1x   # orient(a1_i,a2_i, a1_j)
    o2 = dx[:, None] * dA2y1 - dy[:, None] * dA2x1  # orient(a1_i,a2_i, a2_j)
    o3 = -dx[None, :] * dA1y + dy[None, :] * dA1x   # orient(a1_j,a2_j, a1_i)
    o4 = dx[None, :] * by - dy[None, :] * bx        # orient(a1_j,a2_j, a2_i)

    cross = (o1 * o2 < 0) & (o3 * o4 < 0)
    shared = (
        (sa[:, None] == sa[None, :])
        | (sa[:, None] == sb[None, :])
        | (sb[:, None] == sa[None, :])
        | (sb[:, None] == sb[None, :])
    )
    hits = cross & ~shared
    return int(np.triu(hits, k=1).sum())


def _count_segment_crossings(segments) -> int:
    """Exact star-topology crossing count. Pairs sharing a ref are skipped."""
    if len(segments) >= _VECTORIZED_CROSSINGS_MIN:
        vectorized = _count_segment_crossings_numpy(segments)
        if vectorized is not None:
            return vectorized
    return _count_segment_crossings_loop(segments)


# ---------------------------------------------------------------------------
# Schematic-specific adapters: positions and net membership from placed parts.
# ---------------------------------------------------------------------------


def _is_net_terminal(part) -> bool:
    """True if ``part`` is a NetTerminal (lazy import mirrors
    ``place.is_net_terminal`` so this module imports nothing at load time)."""
    from skidl.schematics.net_terminal import NetTerminal

    return isinstance(part, NetTerminal)


def _part_ref(part) -> str:
    return str(getattr(part, "ref", "") or "")


def _part_sort_key(part):
    """Stable, id-independent sort key for a Part (mirrors
    ``place._part_order_key``): ``num`` disambiguates multi-unit parts."""
    return (_part_ref(part), getattr(part, "num", 0) or 0)


def _part_positions(parts) -> tuple[dict[str, tuple[float, float]], int]:
    """Map each real part's ref -> a representative (x, y), plus a skipped tally.

    The representative point is the mean of the part's placed pin points in the
    placer's coordinate model (``pin.place_pt * pin.part.tx``). When no pin
    carries a ``place_pt`` (reporting time, after ``rmv_placement_stuff``), fall
    back to the part's ``tx`` origin ``(tx.dx, tx.dy)``; if even that is
    unavailable, skip the part and count it. Parts are iterated in ``(ref, num)``
    order so the dict build (and last-write-wins on a shared ref) is
    deterministic. NetTerminals are excluded.
    """
    positions: dict[str, tuple[float, float]] = {}
    skipped = 0
    for part in sorted(parts, key=_part_sort_key):
        if _is_net_terminal(part):
            continue
        xs: list[float] = []
        ys: list[float] = []
        for pin in getattr(part, "pins", []) or []:
            place_pt = getattr(pin, "place_pt", None)
            if place_pt is None:
                continue
            tx = getattr(getattr(pin, "part", None), "tx", None)
            if tx is None:
                continue
            pt = place_pt * tx
            xs.append(pt.x)
            ys.append(pt.y)
        if xs:
            positions[_part_ref(part)] = (sum(xs) / len(xs), sum(ys) / len(ys))
            continue
        tx = getattr(part, "tx", None)
        if tx is not None and hasattr(tx, "dx") and hasattr(tx, "dy"):
            positions[_part_ref(part)] = (tx.dx, tx.dy)
        else:
            skipped += 1
    return positions, skipped


def _net_members(nets, present_refs) -> list[tuple[str, list[str]]]:
    """``(net_name, sorted [ref, ...])`` for nets touching >= 2 present refs.

    Refs are filtered to ``present_refs`` (the parts with a known position),
    de-duplicated, and both the ref list and the outer list are sorted so the
    result is order-independent of net/pin object iteration. NetTerminal pins
    are excluded.
    """
    result: list[tuple[str, list[str]]] = []
    for net in nets:
        name = str(getattr(net, "name", "") or "")
        refs: list[str] = []
        for pin in getattr(net, "pins", []) or []:
            part = getattr(pin, "part", None)
            if part is None or _is_net_terminal(part):
                continue
            ref = _part_ref(part)
            if ref in present_refs and ref not in refs:
                refs.append(ref)
        if len(refs) >= 2:
            result.append((name, sorted(refs)))
    result.sort(key=lambda item: (item[0], item[1]))
    return result


def _crossings_from(members, positions) -> int:
    """Star-topology crossing count over ``members`` given ``positions``.

    Mirrors ``skidl_layout.scoring._estimate_crossings``: per net, the anchor is
    the position-minimum ref (ref breaks ties), and one segment runs from the
    anchor to every other member.
    """
    segments = []
    for _name, refs in members:
        if len(refs) < 2:
            continue
        anchor = min(
            refs, key=lambda ref: (positions[ref][0], positions[ref][1], ref)
        )
        for ref in refs:
            if ref != anchor:
                segments.append((anchor, ref, positions[anchor], positions[ref]))
    return _count_segment_crossings(segments)


def _hpwl_from(members, positions) -> float:
    """Sum of bounding-box half-perimeters over ``members`` (mirrors
    ``skidl_layout.scoring._total_hpwl``)."""
    total = 0.0
    for _name, refs in members:
        xs = [positions[ref][0] for ref in refs]
        ys = [positions[ref][1] for ref in refs]
        if len(xs) >= 2:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def _score_parts_core(parts, nets) -> dict:
    """Score a single placement frame given its parts and nets."""
    positions, skipped = _part_positions(parts)
    members = _net_members(nets, set(positions))
    real_parts = sum(1 for p in parts if not _is_net_terminal(p))
    return {
        "crossings": _crossings_from(members, positions),
        "hpwl": _hpwl_from(members, positions),
        "parts": real_parts,
        "nets": len(members),
        "skipped": skipped,
    }


def _node_nets(node):
    getter = getattr(node, "get_internal_nets", None)
    if getter is None:
        return []
    return getter()


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def score_node(node) -> dict:
    """Score a placed ``SchNode`` and all its children.

    Returns ``{"crossings", "hpwl", "parts", "nets", "skipped"}``. Positions
    across a hierarchy boundary live in per-node frames, so each node is scored
    independently and the counts / lengths are SUMMED (a per-node score is the
    right granularity for a per-node placement bake-off).
    """
    agg = dict(_score_parts_core(list(getattr(node, "parts", []) or []), _node_nets(node)))
    children = getattr(node, "children", None)
    if children:
        for child in children.values():
            child_score = score_node(child)
            agg["crossings"] += child_score["crossings"]
            agg["hpwl"] += child_score["hpwl"]
            agg["parts"] += child_score["parts"]
            agg["nets"] += child_score["nets"]
            agg["skipped"] += child_score["skipped"]
    return agg


def score_parts(parts, nets) -> dict:
    """Score a single placement frame (one connected group) directly.

    Same crossing/HPWL math as :func:`score_node`, but scoped to a caller-
    supplied ``(parts, nets)`` list instead of a node walk, so the placer's
    opt-in candidate bake-off (``place.place_connected_parts``) can score a group
    in its own frame while placing it. Returns
    ``{"crossings", "hpwl", "parts", "nets", "skipped"}``.
    """
    return _score_parts_core(list(parts), list(nets))


def estimate_crossings(node) -> int:
    """Total estimated net crossings for a placed node and its children."""
    return score_node(node)["crossings"]


def total_hpwl(node) -> float:
    """Total half-perimeter wire length for a placed node and its children."""
    return score_node(node)["hpwl"]
