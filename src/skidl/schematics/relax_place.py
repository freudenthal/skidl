# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Constructive relaxation placement -- deterministic spacing that PRESERVES the
constructive seed's directional arrangement, replacing the force-directed refiner
in deconflict mode.

The force-directed pass (``place.push_and_pull``) historically provided two
things: (1) rescue ``random_placement`` by separating overlaps, and (2) incidental
inter-part breathing room. (1) is obsolete once the constructive seed runs -- the
seed (``seed_place.py``) is already overlap-aware (``choose_slot``'s 50-step
collision search on the inflated ``place_bbox``). (2) is what this module
reproduces deterministically, via the seed's own ``gap`` parameter, WITHOUT the
force pass's re-mixing that scrambles the seed's pin-face directional layout
(and, in deconflict mode, was only accidentally load-bearing for net-fusion
avoidance -- now closed structurally by the occupancy registry).

Pure/structural like ``seed_place.py``: no tool imports. Operates on the same
part-like interface (``.tx``, ``.place_bbox``, ``.pins``, ``.ref``).
"""

from skidl.schematics.seed_place import _DEFAULT_GRID, seed_placement


def relax_placement(
    parts, nets, skip=None, max_fanout=3, gap=None, grid=None, **options
):
    """Constructively (re)seed ``parts`` with clearance-relaxed spacing.

    A thin wrapper over :func:`seed_placement` that defaults ``gap`` to a
    relaxation spacing (``relax_gap``, a touch wider than the plain seed's
    ``2*grid``) so ``choose_slot``'s existing collision search spreads parts
    apart while making the IDENTICAL directional slot choices. Deterministic and
    arrangement-preserving by construction; never consumes ``random``.

    Mirrors ``seed_placement``'s signature so ``place_connected_parts`` can call
    it in the same spot.
    """
    if grid is None:
        grid = _DEFAULT_GRID
    if gap is None:
        gap = options.get("relax_gap", 3 * grid)
    seed_placement(
        parts,
        nets,
        skip=skip,
        max_fanout=max_fanout,
        gap=gap,
        grid=grid,
        **{
            k: v
            for k, v in options.items()
            if k not in ("skip", "grid", "max_fanout", "gap")
        },
    )
