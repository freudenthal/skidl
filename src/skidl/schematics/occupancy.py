# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Unified per-sheet grid-cell occupancy registry.

Single source of truth for "which grid cell belongs to which net, and what is
blocked", built once at routing time and carried through to emission. It
replaces the several ad-hoc ``occupied`` dicts that ``add_deconflicted_stubs``,
the A* router, and the label deconflicter each derived independently -- the
second-derived occupancy is exactly the root cause of the cross-net fusion the
render-occupancy plan closes.

Pure module: no tool imports (same layering rule as ``seed_place.py``). The
caller injects the grid pitch, so ONE registry serves whatever coordinate frame
it was built in (the router/world frame at 50-mil ``GRID``); a frame conversion
is an explicit, separate step (:meth:`rescaled`) rather than a second registry.

Cells are ``(x, y)`` tuples snapped to the grid. Iteration order is deterministic
(sorted) so anything derived from the registry is reproducible.
"""

from __future__ import annotations


class SheetOccupancy:
    """Grid-cell ownership + blocking registry for a single schematic sheet.

    * ``owner`` maps a snapped cell to the net object that owns it (a pin, a
      stub end, a routed segment, or a power symbol/stub).
    * ``blocked`` is the set of cells inside a part body -- no stub end or label
      anchor may land there (consulted via ``respect_blocked=True``).
    """

    __slots__ = ("_grid", "_owner", "_blocked")

    def __init__(self, grid):
        self._grid = float(grid)
        self._owner = {}  # cell -> net object
        self._blocked = set()  # cells inside part bodies

    # -- grid ---------------------------------------------------------------
    @property
    def grid(self):
        return self._grid

    def cell(self, x, y):
        """Snap a world point to its grid cell (deterministic key)."""
        g = self._grid
        return (round(x / g) * g, round(y / g) * g)

    # -- reads --------------------------------------------------------------
    def owner(self, cell):
        """The net owning ``cell`` (already snapped), or ``None``."""
        return self._owner.get(cell)

    def is_blocked(self, cell):
        return cell in self._blocked

    def is_free_for(self, cell, net, respect_blocked=False):
        """True if ``net`` may occupy ``cell``.

        A cell is free for ``net`` when it is unowned or already owned by the
        SAME net (same-net coincidence is legal -- that is how a net connects).
        With ``respect_blocked`` also reject cells inside a part body.
        """
        o = self._owner.get(cell)
        if o is not None and o is not net:
            return False
        if respect_blocked and cell in self._blocked:
            return False
        return True

    # -- writes -------------------------------------------------------------
    def seed(self, cell, net):
        """Record ``net`` at ``cell`` only if the cell is unowned.

        First-writer-wins, matching ``dict.setdefault`` (the seeding order in
        ``add_deconflicted_stubs`` relies on this)."""
        if cell not in self._owner:
            self._owner[cell] = net

    def set(self, cell, net):
        """Unconditionally record ``net`` as the owner of ``cell``."""
        self._owner[cell] = net

    def claim(self, cell, net, respect_blocked=False):
        """Take ``cell`` for ``net`` if free; return whether it was taken."""
        if not self.is_free_for(cell, net, respect_blocked=respect_blocked):
            return False
        self._owner[cell] = net
        return True

    def _axis_cells(self, p1, p2):
        """Yield every grid cell along the axial segment ``p1``->``p2`` (incl.
        both ends). Raises for a non-axial (diagonal) segment."""
        c1 = self.cell(p1[0], p1[1])
        c2 = self.cell(p2[0], p2[1])
        g = self._grid
        if c1 == c2:
            yield c1
            return
        if abs(c1[1] - c2[1]) < g / 2:  # horizontal
            y = c1[1]
            x0, x1 = sorted((c1[0], c2[0]))
            n = int(round((x1 - x0) / g))
            for i in range(n + 1):
                yield (x0 + i * g, y)
        elif abs(c1[0] - c2[0]) < g / 2:  # vertical
            x = c1[0]
            y0, y1 = sorted((c1[1], c2[1]))
            n = int(round((y1 - y0) / g))
            for i in range(n + 1):
                yield (x, y0 + i * g)
        else:
            raise ValueError("claim_segment: non-axial segment %r -> %r" % (p1, p2))

    def claim_segment(self, p1, p2, net):
        """Register every cell of an axial segment as owned by ``net``.

        Records the pin->end stub (or a routed run) so a foreign net's endpoint
        cannot land on its interior (the KiCad T-junction fusion). Returns the
        list of cells that were already owned by a DIFFERENT net (empty = clean);
        free/same-net cells are set to ``net`` regardless so the segment is fully
        registered for the A* obstacle set."""
        conflicts = []
        for c in self._axis_cells(p1, p2):
            o = self._owner.get(c)
            if o is not None and o is not net:
                conflicts.append(c)
            else:
                self._owner[c] = net
        return conflicts

    def block_bbox(self, bbox, strict=True):
        """Mark the grid cells covered by ``bbox`` as blocked (part body).

        ``bbox`` is anything with ``.min``/``.max`` points in this registry's
        frame. With ``strict`` only the interior is blocked (cells on the bbox
        edge stay claimable -- pins sit on body edges)."""
        g = self._grid
        import math

        lo_x = math.floor(bbox.min.x / g) * g
        lo_y = math.floor(bbox.min.y / g) * g
        hi_x = math.ceil(bbox.max.x / g) * g
        hi_y = math.ceil(bbox.max.y / g) * g
        nx = int(round((hi_x - lo_x) / g))
        ny = int(round((hi_y - lo_y) / g))
        for i in range(nx + 1):
            cx = lo_x + i * g
            for j in range(ny + 1):
                cy = lo_y + j * g
                if strict and (
                    cx <= bbox.min.x
                    or cx >= bbox.max.x
                    or cy <= bbox.min.y
                    or cy >= bbox.max.y
                ):
                    continue
                self._blocked.add((cx, cy))

    # -- outputs ------------------------------------------------------------
    def published(self):
        """The ``{cell -> net}`` map with unowned cells dropped (the payload the
        A* router consumes as ``node._deconflict_occupied``)."""
        return {c: n for c, n in self._owner.items() if n is not None}

    def owners_sorted(self):
        """Deterministic ``(cell, net)`` iteration for tests/audits."""
        return sorted(self._owner.items(), key=lambda kv: kv[0])

    def rescaled(self, scale, dx=0.0, dy=0.0):
        """A copy of the ownership map in another frame: ``cell*scale + (dx,dy)``.

        Used to hand the router-frame registry to the render-frame emitter
        without maintaining a second registry. Blocked cells convert too."""
        out = SheetOccupancy(self._grid * scale)
        for (x, y), net in self._owner.items():
            out._owner[out.cell(x * scale + dx, y * scale + dy)] = net
        for (x, y) in self._blocked:
            out._blocked.add(out.cell(x * scale + dx, y * scale + dy))
        return out
