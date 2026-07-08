# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""
Backend-agnostic snap geometry for schematic generation.

After placement+routing, two-pin parts (resistors, caps, LEDs, ...) are
"snapped" so one of their pins lands exactly on the pin of a connected
multi-pin part (an IC) or on the free pin of an already-snapped two-pin
part.  This turns a sea of labelled stubs into directly-touching pins and
short wires, which the KiCad emitter then renders without redundant labels.

The geometry here is independent of any particular EDA tool: it operates on
SKiDL Part/Pin/Net objects and SchNode trees, mutating ``part.tx`` and
recording wire/suppression hints on the node:

    node._tjunction_wires            list[(x1,y1,x2,y2)]  (mils, pre-sheet-tx)
    node._tjunction_suppressed_pins  set[id(pin)]
    node._power_cap_wires            list[(x1,y1,x2,y2)]  (mils, pre-sheet-tx)
    node._power_cap_junctions        list[Point]          (mils, pre-sheet-tx)
    node._power_cap_suppressed_pins  set[id(pin)]

The tool-specific emitter (e.g. kicad9/sexp_schematic.py) reads these
attributes to draw wires and suppress the corresponding pin labels.
"""

import logging
import re
from collections import defaultdict

from skidl.geometry import Point, Tx
from skidl.schematics.net_terminal import NetTerminal

logger = logging.getLogger(__name__)


# PROPOSAL FLAG (default OFF): when True, _stagger_tjunctions drops a junction
# dot at each fan's shared point and suppresses the IC-pin net label, relying on
# pure pin-to-pin/wire connectivity for the fan net.  This removes the redundant
# SW_n label but MUST be validated with real IC-fan ERC (kicad-cli sch erc) on a
# board that actually triggers staggered T-junction fans before being enabled,
# because the IC-pin label is currently load-bearing (see finding #7).
_ENABLE_IC_FAN_PIN_REWIRE = False


# Pattern matching common power net names (kept in sync with the gen wrapper).
_POWER_NET_RE = re.compile(
    r"^(\+\d[\d.]*V[\d]*|GND|AGND|DGND|PGND|VCC|VDD|VSS|VEE|VBUS|VBAT|AVCC|AVDD|DVCC|DVDD)$",
    re.IGNORECASE,
)


def _is_two_pin_part(part):
    """Return True if part is a simple 2-pin component (LED, R, C, etc.)."""
    return not isinstance(part, NetTerminal) and len(part.pins) == 2


def _stub_snapped_part(part):
    """Mark a snapped part's pins (and their nets) as stubbed.

    Snapping moves a part AFTER routing, so any wire the router drew to it is
    now stale. Converting its connections to labels (stubs) makes them
    re-emit at the part's new position; the emitter then suppresses the
    redundant ones where pins physically overlap. Explicit user stubs are
    left untouched.
    """
    for pin in part.pins:
        net = getattr(pin, "net", None)
        if net is None or getattr(net, "_stub_explicit", False):
            continue
        pin.stub = True
        try:
            net._stub = True
            for p in net.get_pins():
                p.stub = True
        except AttributeError:
            pass


def _is_power_net(net):
    """Return True if net is a power rail (GND, VCC, +3.3V, etc.)."""
    name = getattr(net, "name", "")
    return name.startswith("+") or bool(_POWER_NET_RE.match(name))


_GND_NET_RE = re.compile(r"^(A|D|P)?GND$|^(VSS|VEE|GROUND)$", re.I)


def _is_gnd_net(net):
    """Return True if net is a ground rail (GND, AGND, VSS, etc.)."""
    name = getattr(net, "name", "")
    return bool(_GND_NET_RE.match(name))


def _pin_world_orient(pin, part):
    """Get the world-space outward direction from a pin after part rotation.

    Transforms the pin's stub direction vector through the part's full
    transform (including mirrors/flips), then returns the opposite direction.
    """
    orient_to_vec = {"R": (1, 0), "L": (-1, 0), "U": (0, -1), "D": (0, 1)}
    outward = {"L": "R", "R": "L", "U": "D", "D": "U"}

    raw_orient = getattr(pin, "orientation", "R")
    vx, vy = orient_to_vec.get(raw_orient, (1, 0))
    tx = part.tx
    wx = tx.a * vx + tx.b * vy
    wy = tx.c * vx + tx.d * vy
    if abs(wx) >= abs(wy):
        world_orient = "R" if wx > 0 else "L"
    else:
        world_orient = "D" if wy > 0 else "U"
    return outward.get(world_orient, "R")


def _compute_snap_tx(my_pin, other_pin, target_world, extend_dir):
    """Compute the transform to snap a 2-pin part onto a target pin position.

    Orients the part so `other_pin` extends in `extend_dir` from the target,
    and places `my_pin` exactly at `target_world`.

    Returns:
        Tx: The new transform for the 2-pin part.
    """
    dx_local = other_pin.pt.x - my_pin.pt.x
    dy_local = other_pin.pt.y - my_pin.pt.y

    if extend_dir == "R":
        if abs(dx_local) >= abs(dy_local):
            symtx = "" if dx_local > 0 else "H"
        else:
            symtx = "R" if dy_local > 0 else "L"
    elif extend_dir == "L":
        if abs(dx_local) >= abs(dy_local):
            symtx = "" if dx_local < 0 else "H"
        else:
            symtx = "L" if dy_local > 0 else "R"
    elif extend_dir == "U":
        if abs(dy_local) >= abs(dx_local):
            symtx = "" if dy_local > 0 else "V"
        else:
            symtx = "L" if dx_local > 0 else "R"
    elif extend_dir == "D":
        if abs(dy_local) >= abs(dx_local):
            symtx = "" if dy_local < 0 else "V"
        else:
            symtx = "R" if dx_local > 0 else "L"
    else:
        symtx = ""

    new_tx = Tx.from_symtx(symtx)
    my_pin_placed = my_pin.pt * new_tx
    offset = Point(
        target_world.x - my_pin_placed.x,
        target_world.y - my_pin_placed.y,
    )
    return new_tx.move(offset)


# --------------------------------------------------------------------------- #
# Cross-net collision guard (Blocker B, stage 19)
# --------------------------------------------------------------------------- #
#
# The snap passes place one pin of a 2-pin part exactly on a target pin and let
# the OTHER pin fall wherever the part geometry puts it, with no check on where
# that far pin lands. When a far pin (or the snapped pin) coincides with a pin of
# a DIFFERENT net, the kicad emitter draws a power symbol / label / wire at that
# shared point and KiCad's connectivity engine fuses the two nets (measured:
# ldo_bias R7/2(VBIAS_SENSE) landing on C7/2(GND); sipm_tia C15/2(GND) on
# U1/3(SN)). These helpers let each snap site veto a colliding transform and a
# final sweep guarantees no cross-net coincidence survives.

_COLLISION_TOL = 1.0  # mils; pins sit on a 100-mil grid, so this is exact-ish.


def _net_ident(pin):
    """A hashable identity for a pin's net: its name if named, else object id.

    Two Net objects can represent the same electrical net; comparing by name
    treats same-named nets as the same net (correct for coincidence purposes).
    """
    net = getattr(pin, "net", None)
    if net is None:
        return None
    return getattr(net, "name", None) or id(net)


def _would_collide(part, cand_tx, node, ignore_parts=()):
    """True if any pin of *part* at *cand_tx* lands on a foreign-net pin.

    Scans every other real part in *node* (at its current ``tx``) for a pin
    within ``_COLLISION_TOL`` of one of *part*'s pins that belongs to a
    DIFFERENT net. Everything is in mils, pre-sheet-tx (snap's working space);
    the per-node sheet transform is affine + uniform, so a pre-sheet-tx
    coincidence maps 1:1 to an emitted coincidence. NetTerminals are labels
    relocated at emission, so they are skipped.
    """
    mine = []
    for p in part.pins:
        w = p.pt * cand_tx
        mine.append((w.x, w.y, _net_ident(p)))

    for other in node.parts:
        if other is part or other in ignore_parts:
            continue
        if isinstance(other, NetTerminal):
            continue
        for op in other.pins:
            ow = op.pt * other.tx
            onet = _net_ident(op)
            for (mx, my, mnet) in mine:
                if abs(mx - ow.x) <= _COLLISION_TOL and abs(my - ow.y) <= _COLLISION_TOL:
                    if onet != mnet:
                        return True
    return False


def _revert_cross_net_snaps(node, snapped, presnap_tx):
    """Post-snap invariant sweep: revert any snap that fused two nets.

    Runs after every snap stage. For any snapped 2-pin part whose current
    placement leaves a pin coincident with a foreign-net pin, restore its
    pre-snap transform (where it kept its labels — always electrically safe).
    Repeats until clean because reverting one part can clear a collision that
    was masking another. A cross-net coincidence between two NON-snapped parts
    is a genuine placement bug we cannot fix here, so it is logged loudly.
    """
    for _sweep in range(8):
        reverted = False
        for part in node.parts:
            if id(part) not in snapped:
                continue
            if _would_collide(part, part.tx, node):
                old = presnap_tx.get(id(part))
                if old is not None and old is not part.tx:
                    part.tx = old
                    snapped.discard(id(part))
                    reverted = True
                    logger.info(
                        "snap: reverted %s to pre-snap position (would have fused "
                        "nets)",
                        getattr(part, "ref", part),
                    )
        if not reverted:
            break

    # Anything still coincident across nets now involves an unsnapped part; that
    # is a placement-level collision this pass cannot repair — surface it.
    for part in node.parts:
        if isinstance(part, NetTerminal):
            continue
        if _would_collide(part, part.tx, node):
            logger.warning(
                "snap: residual cross-net pin coincidence at %s after sweep "
                "(non-snap placement collision)",
                getattr(part, "ref", part),
            )


def _resolve_cluster(decl, part, node):
    """Resolve a declared ``cluster="REF.PIN"`` / ``"REF"`` snap target (stage 24).

    Lets the designer state design intent -- "this 2-pin part belongs on that IC
    pin" -- so snap honors the declaration instead of guessing by the pin-count
    heuristic (the Blocker-B root). Returns ``(my_pin, target_pin, target_part)``
    on success, or ``None`` (after a warning) so the caller falls back to the
    heuristic. A declaration says WHERE to try, not permission to collide -- every
    Blocker-B guard (``_would_collide`` veto, ``_revert_cross_net_snaps`` sweep)
    still applies at the call site. Never fails the render on a bad hint.
    """
    decl = str(decl).strip()
    if not decl:
        return None
    ref, _, pin_id = decl.partition(".")
    ref, pin_id = ref.strip(), pin_id.strip()
    who = getattr(part, "ref", part)

    # Target part must live on THIS node/sheet (snap works in one node's frame).
    target_part = next(
        (p for p in node.parts
         if not isinstance(p, NetTerminal) and getattr(p, "ref", None) == ref),
        None,
    )
    if target_part is None:
        logger.warning(
            "cluster: %s target ref %r not on this sheet; using heuristic", who, ref)
        return None

    # Candidate target pins: the named/numbered pin, or all pins for a bare ref.
    if pin_id:
        cand_pins = [
            tp for tp in target_part.pins
            if str(getattr(tp, "num", "")) == pin_id
            or getattr(tp, "name", None) == pin_id
        ]
        if not cand_pins:
            logger.warning(
                "cluster: %s pin %r not on %r; using heuristic", who, pin_id, ref)
            return None
    else:
        cand_pins = list(target_part.pins)

    # The target pin must share a net with one of our pins (that shared net's
    # pin on OUR side becomes my_pin).
    my_by_net = {
        id(getattr(p, "net", None)): p
        for p in part.pins if getattr(p, "net", None) is not None
    }
    matches = [
        (my_by_net[id(tp.net)], tp)
        for tp in cand_pins
        if getattr(tp, "net", None) is not None and id(tp.net) in my_by_net
    ]
    if not matches:
        logger.warning(
            "cluster: %s shares no net with target %r; using heuristic", who, decl)
        return None
    if len(matches) > 1 and not pin_id:
        logger.warning(
            "cluster: %s bare ref %r ambiguous (%d shared pins); using heuristic",
            who, ref, len(matches))
        return None
    my_pin, target_pin = matches[0]
    return my_pin, target_pin, target_part


def snap_two_pin_parts(node, stub=True):
    """Snap 2-pin parts onto their connected IC or already-snapped part pins.

    Pass 1: Snap onto IC pins (parts with >2 pins). Each IC pin only accepts
    one snapped part; extras keep their labels.

    Pass 2+: Iteratively snap remaining 2-pin parts onto the free pins of
    already-snapped 2-pin parts, building chains (e.g. IC <- R <- LED).

    Pass 3: Stack remaining 2-pin parts onto already-occupied IC pins,
    extending perpendicular to the first snapped part. Handles nets shared
    between multiple 2-pin parts (e.g. switch + pull-down on the same IC input).

    Recurses into child nodes first.

    Args:
        stub: When True (default, the classic route->snap order) a snapped
            part's nets are stubbed to labels because snapping moves parts
            AFTER routing, invalidating the routed wires. When False (the
            stage-24 snap->route order) snap only MOVES parts; the per-net A*
            router then wires every net on the final geometry, so pin-touched
            pairs route to nothing (coincident) and spread nets get real wires.
    """
    for child in node.children.values():
        snap_two_pin_parts(child, stub=stub)

    node_part_ids = {id(p) for p in node.parts}
    # Remember every part's placement before any snap runs, so the final
    # invariant sweep can revert a snap that fused two nets (Blocker B).
    presnap_tx = {id(p): p.tx for p in node.parts}
    snapped = set()
    occupied_pins = set()
    # How many decoupling caps have already snapped onto each +supply pin, so
    # additional caps on the same pin fan out instead of stacking.
    power_pin_cap_idx = {}
    # Per +supply pin: the snapped cap +ve pin positions, so the fan can be
    # routed as a single right-angle bus instead of a star of diagonals.
    power_cap_groups = {}

    # Declared-adjacency (cluster=) parts snap FIRST so they claim their declared
    # pin before an undeclared part's heuristic can steal it. (stage 24)
    ordered_parts = sorted(
        list(node.parts), key=lambda p: 0 if getattr(p, "cluster", None) else 1
    )
    for part in ordered_parts:
        if not _is_two_pin_part(part):
            continue

        p1, p2 = part.pins[0], part.pins[1]
        net1 = getattr(p1, "net", None)
        net2 = getattr(p2, "net", None)
        if not net1 or not net2:
            continue

        target_pin = None
        target_part = None
        my_pin = None

        both_power = _is_power_net(net1) and _is_power_net(net2)
        min_target_pins = 8 if both_power else 2

        # Honor an explicit cluster="REF.PIN" declaration before the pin-count
        # heuristic; on any resolution failure it returns None and we fall
        # through. The declaration bypasses the size heuristic but keeps every
        # Blocker-B guard (the _would_collide veto + revert sweep below).
        decl = getattr(part, "cluster", None)
        if decl:
            resolved = _resolve_cluster(decl, part, node)
            if resolved:
                my_pin, target_pin, target_part = resolved

        if target_pin is None:
            for my_p, other_net in [(p1, net1), (p2, net2)]:
                if _is_power_net(other_net) and not both_power:
                    continue
                # Decoupling caps (both pins on power) anchor only to the +supply
                # pin, never GND — this clusters them at the IC's VIN/VDD pin.
                if both_power and _is_gnd_net(other_net):
                    continue
                for net_pin in other_net.pins:
                    other_part = net_pin.part
                    if (
                        other_part is not part
                        and id(other_part) in node_part_ids
                        and not isinstance(other_part, NetTerminal)
                        and len(other_part.pins) > min_target_pins
                        and id(net_pin) not in occupied_pins
                    ):
                        target_pin = net_pin
                        target_part = other_part
                        my_pin = my_p
                        break
                if target_pin:
                    break

        if not target_pin:
            continue

        target_world = target_pin.pt * target_part.tx
        extend_dir = _pin_world_orient(target_pin, target_part)
        other_pin = p2 if my_pin is p1 else p1

        if both_power:
            # Multiple decoupling caps share one +supply pin: push each out by a
            # base offset along the pin, then fan successive caps perpendicular
            # so they sit in a readable row instead of stacking on top of
            # each other.
            n = power_pin_cap_idx.get(id(target_pin), 0)
            _offset_dir = {"R": (200, 0), "L": (-200, 0), "U": (0, 200), "D": (0, -200)}
            ax, ay = _offset_dir.get(extend_dir, (200, 0))
            # Unit perpendicular to the outward pin direction (rotate 90°).
            perp_x, perp_y = (-ay // 200, ax // 200)
            FAN_STEP = 300

            def _fan_tx(fan_n):
                base = _compute_snap_tx(my_pin, other_pin, target_world, extend_dir)
                dx = ax + perp_x * FAN_STEP * fan_n
                dy = ay + perp_y * FAN_STEP * fan_n
                return base.move(Point(dx, dy))

            # On collision advance the fan index (walk further out along the
            # perpendicular row) rather than flipping direction, so the fan stays
            # a straight bus. Give up after a few steps and leave labels.
            cand = _fan_tx(n)
            tries = 0
            while _would_collide(part, cand, node) and tries < 4:
                n += 1
                tries += 1
                cand = _fan_tx(n)
            if _would_collide(part, cand, node):
                continue  # skip the snap; part keeps its labels (safe)
            power_pin_cap_idx[id(target_pin)] = n + 1
            part.tx = cand
            # Emit a wire from the IC's power pin back to the now-offset cap +ve pin
            # so the connection is visually drawn rather than relying on two power
            # labels. Suppress the cap +ve pin's label since the wire makes it
            # redundant.
            cap_pin_world = my_pin.pt * part.tx
            grp = power_cap_groups.setdefault(
                id(target_pin), {"target": target_world, "caps": []}
            )
            grp["caps"].append(cap_pin_world)
            power_cap_suppressed = getattr(node, "_power_cap_suppressed_pins", set())
            power_cap_suppressed.add(id(my_pin))
            node._power_cap_suppressed_pins = power_cap_suppressed
        else:
            cand = _compute_snap_tx(my_pin, other_pin, target_world, extend_dir)
            if _would_collide(part, cand, node):
                # Try the other extend directions before giving up so a good
                # layout is preserved where possible.
                for alt in [d for d in "RLUD" if d != extend_dir]:
                    alt_cand = _compute_snap_tx(my_pin, other_pin, target_world, alt)
                    if not _would_collide(part, alt_cand, node):
                        cand = alt_cand
                        break
                else:
                    continue  # skip the snap entirely; part keeps labels (safe)
            part.tx = cand
        if stub:
            _stub_snapped_part(part)
        snapped.add(id(part))
        # Decoupling caps fan out off a shared +supply pin, so don't mark it
        # occupied — let later caps cluster on the same pin. Single-target snaps
        # still claim their pin so the next part finds a fresh one.
        if not both_power:
            occupied_pins.add(id(target_pin))

    # Route each +supply pin's decoupling caps as a right-angle bus rather than
    # a star of diagonals: a single trunk along the (collinear) cap +ve pins,
    # plus a short orthogonal connector from the IC pin onto the trunk, with a
    # junction dot at every tap. The caps were placed at
    # target + base*along + n*FAN_STEP*perp, so trunk and connector are
    # axis-aligned by construction.
    if power_cap_groups:
        power_cap_wires = getattr(node, "_power_cap_wires", [])
        power_cap_junctions = getattr(node, "_power_cap_junctions", [])
        for grp in power_cap_groups.values():
            t = grp["target"]
            caps = grp["caps"]
            # connector: IC pin -> first cap pin, along the pin's outward axis
            power_cap_wires.append((t.x, t.y, caps[0].x, caps[0].y))
            # trunk: consecutive cap +ve pins (the perpendicular fan row)
            for a, b in zip(caps, caps[1:]):
                power_cap_wires.append((a.x, a.y, b.x, b.y))
            # junction at every tap where 3 segments/pins meet (all taps except
            # the trunk's far endpoint, which is a plain wire-end-on-pin)
            for cap in caps[:-1]:
                power_cap_junctions.append(cap)
        node._power_cap_wires = power_cap_wires
        node._power_cap_junctions = power_cap_junctions

    for _iteration in range(5):
        newly_snapped = set()

        for part in list(node.parts):
            if id(part) in snapped or not _is_two_pin_part(part):
                continue

            p1, p2 = part.pins[0], part.pins[1]
            net1 = getattr(p1, "net", None)
            net2 = getattr(p2, "net", None)
            if not net1 or not net2:
                continue

            target_pin = None
            target_part = None
            my_pin = None

            both_power = _is_power_net(net1) and _is_power_net(net2)

            for my_p, other_net in [(p1, net1), (p2, net2)]:
                if _is_power_net(other_net) and not both_power:
                    continue
                for net_pin in other_net.pins:
                    other_part = net_pin.part
                    if (
                        other_part is not part
                        and id(other_part) in snapped
                        and id(net_pin) not in occupied_pins
                    ):
                        target_pin = net_pin
                        target_part = other_part
                        my_pin = my_p
                        break
                if target_pin:
                    break

            if not target_pin:
                continue

            target_world = target_pin.pt * target_part.tx
            extend_dir = _pin_world_orient(target_pin, target_part)
            other_pin = p2 if my_pin is p1 else p1

            cand = _compute_snap_tx(my_pin, other_pin, target_world, extend_dir)
            if _would_collide(part, cand, node):
                for alt in [d for d in "RLUD" if d != extend_dir]:
                    alt_cand = _compute_snap_tx(my_pin, other_pin, target_world, alt)
                    if not _would_collide(part, alt_cand, node):
                        cand = alt_cand
                        break
                else:
                    continue  # skip the snap entirely; part keeps labels (safe)
            part.tx = cand
            if stub:
                _stub_snapped_part(part)
            newly_snapped.add(id(part))
            occupied_pins.add(id(target_pin))

        if not newly_snapped:
            break
        snapped |= newly_snapped

    perp_map = {"R": "D", "L": "U", "U": "R", "D": "L"}
    for part in list(node.parts):
        if id(part) in snapped or not _is_two_pin_part(part):
            continue

        p1, p2 = part.pins[0], part.pins[1]
        net1 = getattr(p1, "net", None)
        net2 = getattr(p2, "net", None)
        if not net1 or not net2:
            continue

        target_pin = None
        target_part = None
        my_pin = None

        for my_p, other_net in [(p1, net1), (p2, net2)]:
            for net_pin in other_net.pins:
                other_part = net_pin.part
                if (
                    other_part is not part
                    and id(other_part) in node_part_ids
                    and not isinstance(other_part, NetTerminal)
                    and len(other_part.pins) > 2
                    and id(net_pin) in occupied_pins
                ):
                    target_pin = net_pin
                    target_part = other_part
                    my_pin = my_p
                    break
            if target_pin:
                break

        if not target_pin:
            continue

        target_world = target_pin.pt * target_part.tx
        ic_dir = _pin_world_orient(target_pin, target_part)
        extend_dir = perp_map.get(ic_dir, ic_dir)
        other_pin = p2 if my_pin is p1 else p1

        cand = _compute_snap_tx(my_pin, other_pin, target_world, extend_dir)
        if _would_collide(part, cand, node):
            for alt in [d for d in "RLUD" if d != extend_dir]:
                alt_cand = _compute_snap_tx(my_pin, other_pin, target_world, alt)
                if not _would_collide(part, alt_cand, node):
                    cand = alt_cand
                    break
            else:
                continue  # skip the snap entirely; part keeps labels (safe)
        part.tx = cand
        if stub:
            _stub_snapped_part(part)
        snapped.add(id(part))

    _stagger_tjunctions(node, node_part_ids, snapped, occupied_pins, stub=stub)

    # Guarantee: no snapped 2-pin part may leave a pin coincident with a
    # foreign-net pin. Any veto above is best-effort per-site; this sweep is the
    # backstop that reverts any residual cross-net fusion to the pre-snap
    # placement (where the part kept its labels — always electrically safe).
    _revert_cross_net_snaps(node, snapped, presnap_tx)


def _stagger_tjunctions(node, node_part_ids, snapped, occupied_pins, min_group=2, stub=True):
    """Detect repeating T-junction patterns and stagger parts outward from IC.

    Phase 1: identify stagger groups, compute how much space each needs,
    and shift ICs apart vertically so fans won't overlap.
    Phase 2: place the staggered parts at the (now separated) IC positions.
    """
    perp_map = {"R": "D", "L": "U", "U": "R", "D": "L"}
    anti_perp = {"U": "D", "D": "U", "L": "R", "R": "L"}
    _dir_vec = {"R": (1, 0), "L": (-1, 0), "U": (0, -1), "D": (0, 1)}

    ic_pin_to_parts = defaultdict(list)

    for part in node.parts:
        if not _is_two_pin_part(part):
            continue

        p1, p2 = part.pins[0], part.pins[1]
        net1 = getattr(p1, "net", None)
        net2 = getattr(p2, "net", None)
        if not net1 or not net2:
            continue

        for my_p, other_net in [(p1, net1), (p2, net2)]:
            if _is_power_net(other_net):
                continue
            for net_pin in other_net.pins:
                ic = net_pin.part
                if (
                    ic is not part
                    and id(ic) in node_part_ids
                    and not isinstance(ic, NetTerminal)
                    and len(ic.pins) > 2
                    and id(net_pin) in occupied_pins
                ):
                    other_pin = p2 if my_p is p1 else p1
                    ic_pin_to_parts[id(net_pin)].append(
                        (part, my_p, other_pin, net_pin, ic)
                    )
                    break
            else:
                continue
            break

    ic_groups = defaultdict(list)
    for ic_pin_id, parts_list in ic_pin_to_parts.items():
        if not parts_list:
            continue
        ic = parts_list[0][4]
        ic_groups[id(ic)].append((parts_list[0][3], parts_list))

    # Phase 1: identify qualifying groups and pre-shift ICs.
    MM_TO_MILS = 1 / 0.0254
    stagger_plans = []

    for ic_id, pin_entries in ic_groups.items():
        fanout_counts = [len(pl) for _, pl in pin_entries]
        dominant = max(set(fanout_counts), key=fanout_counts.count)
        if dominant < 2:
            continue
        matching = [(ip, pl) for ip, pl in pin_entries if len(pl) == dominant]

        # Stagger a fan when EITHER the pattern repeats across >= min_group IC
        # pins (e.g. a row of 74HC165 inputs, each switch + pull-down), OR a
        # single IC pin carries >= 2 two-pin parts (e.g. an MCU EN pin with a
        # pull-up to VCC + a filter cap to GND). The single-pin case is the
        # body-overlap bug: without staggering, the first part tees off the pin
        # but the second chains colinearly back across the IC body.
        single_pin_fan = dominant >= 2 and len(matching) >= 1
        if len(matching) < min_group and not single_pin_fan:
            continue

        ic_part = matching[0][1][0][4]
        ic_dir = _pin_world_orient(matching[0][0], ic_part)
        step_dx, step_dy = _dir_vec.get(ic_dir, (1, 0))

        max_span = 0
        for _, parts_list_scan in matching:
            for (scan_part, _, _, _, _) in parts_list_scan:
                pts = [getattr(p, "pt", Point(p.x * MM_TO_MILS, p.y * MM_TO_MILS)) for p in scan_part.pins]
                if pts:
                    span = max(
                        max(p.x for p in pts) - min(p.x for p in pts),
                        max(p.y for p in pts) - min(p.y for p in pts),
                    )
                    max_span = max(max_span, span)
        step_size = max(100, int(max_span) + 50)

        n_pins = len(matching)
        stagger_extent = step_size * n_pins + max_span

        stagger_plans.append({
            "ic_part": ic_part,
            "matching": matching,
            "ic_dir": ic_dir,
            "step_dx": step_dx,
            "step_dy": step_dy,
            "step_size": step_size,
            "stagger_extent": stagger_extent,
            "dominant": dominant,
        })

    if len(stagger_plans) > 1:
        _pre_shift_ics(stagger_plans, node, snapped)

    # Phase 2: place staggered parts at final IC positions.
    junction_wires = getattr(node, "_tjunction_wires", [])
    junction_dots = getattr(node, "_tjunction_junctions", [])
    suppressed_pins = set()

    for plan in stagger_plans:
        ic_part = plan["ic_part"]
        matching = plan["matching"]
        ic_dir = plan["ic_dir"]
        step_dx = plan["step_dx"]
        step_dy = plan["step_dy"]
        step_size = plan["step_size"]
        perp_dir = perp_map.get(ic_dir, ic_dir)

        # Net labels sit on the IC's stubbed pins and extend outward along the
        # same direction the fan staggers. Shift the whole fan out past the
        # widest label so the first staggered part clears it instead of landing
        # on top of it. Label width ~= (len(name)+1) * label-font (~50 mils).
        # Capped so a long net name can't push the fan absurdly far.
        _LABEL_CHAR_W = 50  # mils, ~KiCad PIN_LABEL_FONT_SIZE
        label_clearance = 0
        for _ic_pin, _ in matching:
            _net = getattr(_ic_pin, "net", None)
            if _net is not None and getattr(_ic_pin, "stub", False):
                label_clearance = max(
                    label_clearance, (len(_net.name) + 1) * _LABEL_CHAR_W
                )
        label_clearance = min(label_clearance, 800)

        def _pin_sort_key(entry, _ic_part=ic_part, _ic_dir=ic_dir):
            ic_pin = entry[0]
            w = ic_pin.pt * _ic_part.tx
            if _ic_dir in ("L", "R"):
                return w.y
            return w.x

        matching.sort(key=_pin_sort_key)

        parts_per_pin = plan["dominant"]
        anti = anti_perp.get(perp_dir, perp_dir)
        extend_dirs = [perp_dir, anti] if parts_per_pin >= 2 else [perp_dir]

        for pin_idx, (ic_pin, parts_list) in enumerate(matching):
            ic_pin_world = ic_pin.pt * ic_part.tx

            parts_list.sort(key=lambda t: getattr(t[0], "ref", ""))

            # Walk the fan's junction point outward (advance offset_n) until no
            # staggered part lands a pin on a foreign net; keep the fan geometry
            # consistent by moving the shared junction rather than flipping
            # individual parts. Give up after a few steps and skip this fan
            # (parts keep their labels — electrically safe).
            def _fan_candidates(offset_n):
                ox = ic_pin_world.x + step_dx * (label_clearance + step_size * offset_n)
                oy = ic_pin_world.y + step_dy * (label_clearance + step_size * offset_n)
                jp = Point(ox, oy)
                cands = []
                for part_idx, (part, my_pin, other_pin, _, _) in enumerate(parts_list):
                    ext_dir = extend_dirs[part_idx % len(extend_dirs)]
                    cands.append(
                        (part, my_pin, _compute_snap_tx(my_pin, other_pin, jp, ext_dir))
                    )
                return ox, oy, cands

            offset_n = pin_idx + 1
            ox, oy, cands = _fan_candidates(offset_n)
            tries = 0
            while (
                any(_would_collide(part, tx, node) for part, _, tx in cands)
                and tries < 3
            ):
                offset_n += 1
                tries += 1
                ox, oy, cands = _fan_candidates(offset_n)
            if any(_would_collide(part, tx, node) for part, _, tx in cands):
                continue  # skip this fan; parts keep their labels (safe)

            for part, my_pin, tx in cands:
                part.tx = tx
                if stub:
                    _stub_snapped_part(part)
                snapped.add(id(part))
                suppressed_pins.add(id(my_pin))
            junction_wires.append(
                (ic_pin_world.x, ic_pin_world.y, ox, oy)
            )

            # PROPOSAL (needs real IC-fan ERC before enabling — see note below):
            # The fan's pin-to-pin geometry is already a fully-connected chain:
            # a wire IC-pin -> junction_pt and >= 2 fan `my_pin`s coincident at
            # junction_pt.  With a junction dot dropped at the shared point the
            # net is resolvable by connectivity alone, so the SW_n label that
            # currently sits on the IC pin (load-bearing today, finding #7) can
            # be suppressed without dangling the wire.  Three or more pins meet
            # at junction_pt (>=2 fan pins + 1 wire end) so a dot is required.
            if _ENABLE_IC_FAN_PIN_REWIRE and len(parts_list) >= 2:
                junction_dots.append(Point(ox, oy))
                suppressed_pins.add(id(ic_pin))

    node._tjunction_wires = junction_wires
    node._tjunction_junctions = junction_dots
    node._tjunction_suppressed_pins = suppressed_pins


def _pre_shift_ics(plans, node, snapped):
    """Shift ICs vertically BEFORE stagger placement so fans won't overlap.

    Collects all parts already snapped to each IC and moves them together.
    The stagger parts haven't been placed yet, so they'll naturally land
    at the shifted IC positions in phase 2.
    """
    for plan in plans:
        ic = plan["ic_part"]
        ic_deps = set()
        ic_id = id(ic)

        for part in node.parts:
            if id(part) == ic_id or id(part) not in snapped:
                continue
            if not _is_two_pin_part(part):
                continue
            for pin in part.pins:
                net = getattr(pin, "net", None)
                if not net:
                    continue
                for net_pin in net.pins:
                    if net_pin.part is ic:
                        ic_deps.add(id(part))
                        break
                if id(part) in ic_deps:
                    break

        plan["_deps"] = [p for p in node.parts if id(p) in ic_deps]

    def _ic_bbox(plan):
        ic = plan["ic_part"]
        all_parts = [ic] + plan["_deps"]
        min_y = float("inf")
        max_y = float("-inf")
        for part in all_parts:
            for pin in part.pins:
                w = pin.pt * part.tx
                min_y = min(min_y, w.y)
                max_y = max(max_y, w.y)
        return min_y, max_y

    plans.sort(key=lambda p: _ic_bbox(p)[0])

    margin = 200
    prev_max_y = None

    for plan in plans:
        ic_min_y, ic_max_y = _ic_bbox(plan)
        needed_height = plan["stagger_extent"]
        group_max_y = max(ic_max_y, ic_min_y + needed_height)

        if prev_max_y is not None and ic_min_y < prev_max_y + margin:
            shift = (prev_max_y + margin) - ic_min_y
            vec = Point(0, shift)
            shifted = set()
            for part in [plan["ic_part"]] + plan["_deps"]:
                if id(part) not in shifted:
                    part.tx = part.tx.move(vec)
                    shifted.add(id(part))
            ic_min_y += shift
            group_max_y += shift

        prev_max_y = group_max_y
