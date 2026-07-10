# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""
Shared S-expression schematic generation for KiCad 6/8/9.

Converts placed+routed SchNode trees into .kicad_sch files.
Used by kicad6, kicad8, and kicad9 gen_schematic thin wrappers.

Sources:
  - part_to_sexp / wire_to_sexp: upstream sexp_schematics branch (devbisme)
  - Hierarchy / custom fields / lib_symbols: feature/kicad8-gen-schematic (PR #281)
  - Net label logic: feature/inject-net-labels (PR #280)
  - Original kicad5 hierarchy walker: node_to_eeschema (kicad5/gen_schematic.py)
  - Credit: cyberhuman (PR #270) for initial KiCad 8 schematic work
"""

import copy
import datetime
import os
import uuid
from collections import Counter, OrderedDict

from simp_sexp import Sexp

from skidl.geometry import Point, Segment, Tx
from skidl.net import NCNet
from skidl.pckg_info import __version__
from skidl.schematics.net_terminal import NetTerminal
from skidl.schlib import SchLib
from skidl.utilities import export_to_all

# UUID namespace — same as gen_netlist.py so UUIDs are cross-referenceable.
_NAMESPACE_UUID = uuid.UUID("7026fcc6-e1a0-409e-aaf4-6a17ea82654f")

# Shared label/symbol orientation table, keyed on calc_pin_dir(). The angle is
# chosen so a net label's text — and a power symbol's body — always extends
# AWAY from the pin (clear of the part body) for every pin direction, including
# mirrored parts. Both net_label_to_sexp and _power_symbol_to_sexp derive their
# angle from this single table so they can never drift apart again. The vertical
# values (U:270, D:90) are the corrected ones; an earlier table had them swapped,
# which overprinted vertical net labels and mis-oriented vertical GND/rail
# symbols. Do not change these without re-rendering the vertical + mirrored cases.
_PIN_LABEL_ANGLE = {"R": 180, "L": 0, "U": 270, "D": 90}


def _lib_nickname(part):
    """Return the library NICKNAME for a part's lib_id (e.g. "Connector_Generic").

    ``part.lib.filename`` may be a bare nickname, a basename with extension, or a
    full absolute path (when the symbol was resolved by scanning an absolute
    ``KICAD*_SYMBOL_DIR``). Only the final path component without its extension is
    a valid KiCad lib nickname; emitting the full path yields a lib_id with
    backslashes and an extra colon that KiCad refuses to load. Handle both path
    separators so this is correct regardless of the OS that produced the path.
    """
    filename = getattr(getattr(part, "lib", None), "filename", None)
    if not filename:
        return "Device"
    base = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    return os.path.splitext(base)[0]


# ---------------------------------------------------------------------------
# Power symbol support
# ---------------------------------------------------------------------------


def init_power_symbol_data():
    """Initialize power symbol state at the start of schematic generation."""

    global pwr_symbol_sexp_dict, pwr_symbol_names, _used_power_symbols, _pwr_counter
    global _pwr_flag_net_names, _pwr_flagged, _flg_counter, _custom_power_symbols

    _used_power_symbols = set()
    _pwr_counter = [0]
    _flg_counter = [0]
    # Net names that need exactly one project-wide PWR_FLAG (undriven rails);
    # populated by write_top_schematic from the circuit. _pwr_flagged tracks the
    # names already flagged so only ONE flag is emitted per rail across all
    # sheets (power symbols connect globally by name).
    _pwr_flag_net_names = set()
    _pwr_flagged = set()
    # Custom in-file (power) symbol definitions cloned for non-stock rail names
    # (e.g. VBIAS_28V): lib_id "power:<name>" -> Sexp definition.
    _custom_power_symbols = {}

    # Just read in the power symbols at the start.
    pwr_lib = SchLib("power")
    with open(pwr_lib.filepath, "r") as f:
        pwr_lib_text = f.read()
    pwr_lib_sexp = Sexp(pwr_lib_text)
    pwr_symbol_sexps = pwr_lib_sexp.search("/kicad_symbol_lib/symbol")
    pwr_symbol_sexp_dict = {sym[1]: sym for sym in pwr_symbol_sexps}
    pwr_symbol_names = set([p.name for p in pwr_lib])


# Nickname for non-stock rail symbols cloned in-file (e.g. V5P, VBIAS_28V).
# Stock rail names (GND, VCC, +5V ... present in the KiCad ``power`` library) and
# PWR_FLAG keep the ``power:`` lib nickname, which resolves against KiCad's global
# sym-lib-table. A non-stock rail name has no ``power`` entry, so KiCad ERC reports
# ``lib_symbol_issues`` ("Symbol 'V5P' not found in symbol library 'power'"); such
# clones instead carry ``SKiDL_rails:`` and are backed by a project-local
# ``SKiDL_rails.kicad_sym`` + ``sym-lib-table`` (written by write_top_schematic) so
# ERC resolves them cleanly.
_CUSTOM_RAIL_LIB = "SKiDL_rails"
_POWER_LIB_PREFIXES = ("power:", _CUSTOM_RAIL_LIB + ":")


def _is_power_libid(lib_id):
    """True if *lib_id* is a rendered power/rail symbol (stock or custom clone)."""
    return isinstance(lib_id, str) and lib_id.startswith(_POWER_LIB_PREFIXES)


def _power_net_from_libid(lib_id):
    """Recover the net/rail name from a power-symbol lib_id (either prefix)."""
    if isinstance(lib_id, str) and ":" in lib_id:
        return lib_id.split(":", 1)[1]
    return lib_id


def _power_libid_for(name):
    """The lib_id a power symbol for rail *name* should carry.

    Stock names present in the KiCad ``power`` library (and ``PWR_FLAG``) keep the
    ``power:`` nickname; any other rail name is a non-stock in-file clone and gets
    the project-local ``SKiDL_rails:`` nickname.
    """
    if name == "PWR_FLAG" or name in pwr_symbol_names:
        return f"power:{name}"
    return f"{_CUSTOM_RAIL_LIB}:{name}"


def _net_wants_power_symbol(net):
    """True if *net* should render as a KiCad ``power:*`` symbol at each pin.

    Either the name is a stock power-lib symbol (``GND``, ``VCC``, ``+3V3`` ...)
    OR the net was classified as a power net upstream (``_is_power_net``, set by
    ``mark_power_nets`` for ``drive == POWER`` / pattern-matched rails). The
    second arm is what lets a non-stock rail name like ``VBIAS_28V`` get a
    (cloned, in-file) power symbol instead of a plain global label.
    """
    if net is None:
        return False
    name = getattr(net, "name", None)
    if name in pwr_symbol_names:
        return True
    return bool(getattr(net, "_is_power_net", False))


def _gnd_style_name(name):
    """True if a rail name reads as a ground (bar-down GND-style symbol)."""
    n = (name or "").upper()
    return n == "GND" or n.endswith("GND") or "VSS" in n or "VEE" in n


def _power_template_name(name):
    """Pick a stock power symbol to clone for a non-stock rail *name*.

    GND-like names clone ``GND`` (bar points down); everything else clones
    ``VCC`` (rail bar points up). Both are ``(power global)`` symbols, so the
    clone connects globally by name exactly like a stock rail.
    """
    if _gnd_style_name(name) and "GND" in pwr_symbol_sexp_dict:
        return "GND"
    return "VCC" if "VCC" in pwr_symbol_sexp_dict else "GND"


def _extract_power_lib_symbol(name):
    """Extract and parse the lib_symbol definition for a power symbol.

    The returned Sexp has its top-level symbol name changed to "power:NAME"
    so it matches the lib_id used in symbol instances.

    For a NON-stock rail name (no ``power`` lib symbol, e.g. ``VBIAS_28V``) a
    template (``VCC`` for rails, ``GND`` for grounds) is cloned: the symbol +
    inner unit-symbol names are renamed and the ``Value`` property is set to the
    rail name. KiCad treats any ``(power global)`` in-file symbol as a power
    symbol connecting globally by name, so no library install is needed. Returns
    None only if even the template is unavailable (caller falls back to a label).

    Args:
        name: Power symbol / rail name (e.g., "GND", "+3V3", "VBIAS_28V").

    Returns:
        Sexp: Parsed symbol definition, or None if not clonable.
    """
    from copy import deepcopy

    src = pwr_symbol_sexp_dict.get(name, None)
    if src is not None:
        pwr_sym_sexp = deepcopy(src)
        # Change the symbol name from "NAME" to "power:NAME" for lib_id matching.
        pwr_sym_sexp[1] = f"power:{name}"
        return pwr_sym_sexp

    # Non-stock rail name: clone a template and rename it to the rail name.
    template = _power_template_name(name)
    src = pwr_symbol_sexp_dict.get(template, None)
    if src is None:
        return None
    clone = deepcopy(src)
    clone[1] = f"{_CUSTOM_RAIL_LIB}:{name}"
    prefix = template + "_"  # inner unit symbols are "<template>_<unit>_<style>"
    for sub in clone:
        if not (hasattr(sub, "__getitem__") and len(sub) >= 2):
            continue
        tag = sub[0]
        # Rename inner unit sub-symbols so they carry the rail name, not the
        # template's (keeps two clones of the same template from colliding).
        if tag == "symbol" and isinstance(sub[1], str) and sub[1].startswith(prefix):
            sub[1] = name + "_" + sub[1][len(prefix):]
        # Set the visible Value to the rail name (this is what KiCad shows and
        # what the global-by-name connection keys on).
        elif tag == "property" and len(sub) >= 3 and sub[1] == "Value":
            sub[2] = name
    # Register an UNQUOTED snapshot so write_top_schematic can emit a project-local
    # SKiDL_rails.kicad_sym + sym-lib-table (keyed by full lib_id). Snapshot now
    # because the returned ``clone`` is embedded per-sheet and mutated in place by
    # that sheet's add_quotes (which is not idempotent) -- storing the live object
    # would double-quote the library file.
    _custom_power_symbols[clone[1]] = deepcopy(clone)
    return clone


def _power_symbol_pin_angle(net_name):
    """Return the intrinsic pin angle (deg) of a power symbol's connection pin.

    KiCad power symbols carry a single pin whose ``(at x y ANGLE)`` describes
    the direction the symbol's body extends at instance-angle 0 (ground
    symbols point down -> 270, voltage rails point up -> 90).  We need this so
    we can rotate the *instance* to align the symbol body with the schematic
    pin's outward stub direction.  Returns 270 (ground-style) if the symbol or
    its pin can't be found, which leaves the historical angle=0 behaviour for
    the common ground case.
    """
    try:
        sym = pwr_symbol_sexp_dict.get(net_name)
        if sym is None:
            # Non-stock rail: use the template we would clone (VCC rails point
            # up -> pin angle 90; GND-style point down -> 270).
            return 90 if not _gnd_style_name(net_name) else 270
        pins = sym.search("/symbol/symbol/pin") or sym.search("/symbol/pin")
        for p in pins:
            at = p.search("/pin/at")
            if at:
                return float(at[0][3]) % 360
    except Exception:
        pass
    return 270


def _power_symbol_to_sexp(pin, net_name, tx, uuid_path=None):
    """Generate a power symbol instance S-expression.

    Args:
        pin: The pin where the power symbol should be placed.
        net_name: The power net name (e.g., "GND", "+3V3").
        tx: Sheet-level transformation matrix.
        uuid_path: Hierarchical UUID path of the sheet this symbol is placed on
            (``"/<root_uuid>"`` for a flat sheet, ``"/<root>/<child>"`` in a
            hierarchy) -- the SAME path ordinary parts on the sheet use. When
            None, falls back to a constant root path for backward compatibility;
            passing the real path keeps auto power symbols on the correct sheet
            (across hierarchical sheets a constant would collapse them all onto
            one nonexistent sheet).

    Returns:
        Sexp: Power symbol instance, or None on failure.
    """
    _used_power_symbols.add(net_name)

    _pwr_counter[0] += 1
    pwr_ref = f"#PWR{_pwr_counter[0]:03d}"

    # Position at pin location.
    part_tx = getattr(pin.part, "tx", Tx())
    combined_tx = part_tx * tx
    pin_pt = getattr(pin, "pt", Point(pin.x, pin.y))
    pt = pin_pt * combined_tx

    px = _round_mm(pt.x)  # the real IC pin position
    py = _round_mm(pt.y)

    # Symbol position: coincident with the pin by default. With the
    # ``power_stubs`` option ON, pull the symbol one grid step OUTWARD along the
    # pin's stub direction and draw a pin->symbol stub wire (the classic KiCad
    # "pin -> short wire -> power symbol" look, so nothing sits crammed on the
    # body). calc_pin_dir gives the outward direction in SKiDL space; map it into
    # rendered (mm, Y-down) space via the linear part of combined_tx.
    x, y = px, py
    if _EMIT_POWER_STUBS:
        _dvec = {"U": Point(0, 1), "D": Point(0, -1), "L": Point(-1, 0), "R": Point(1, 0)}[
            calc_pin_dir(pin)
        ]
        _p0 = Point(0, 0) * combined_tx
        _p1 = _dvec * combined_tx
        _dx, _dy = _p1.x - _p0.x, _p1.y - _p0.y
        _dlen = (_dx * _dx + _dy * _dy) ** 0.5 or 1.0
        x = _snap_grid(px + _POWER_STUB_LEN * _dx / _dlen)
        y = _snap_grid(py + _POWER_STUB_LEN * _dy / _dlen)
        if (x, y) != (px, py):
            _power_stub_wires.append(
                _sheet_stub_wire_sexp(px, py, x, y, f"pwrstub:{net_name}:{px}:{py}:{x}:{y}")
            )

    # Power symbol angle: align the symbol body with the schematic pin's
    # outward stub direction.  ``calc_pin_dir`` gives the world-space
    # direction the pin's stub extends (where the symbol sits); _PIN_LABEL_ANGLE
    # is the same label-orientation table used by net_label_to_sexp.  The
    # symbol's intrinsic pin angle (270 for GND, 90 for voltage rails) is
    # subtracted so e.g. a GND on a pin that exits to the right gets rotated
    # to face left into the pin instead of always hanging straight down.
    intrinsic = _power_symbol_pin_angle(net_name)
    angle = (_PIN_LABEL_ANGLE[calc_pin_dir(pin)] - intrinsic) % 360

    lib_id = _power_libid_for(net_name)
    inst_uuid = _gen_uuid(f"pwr:{net_name}:{x}:{y}:{_pwr_counter[0]}")

    symbol = Sexp(
        [
            "symbol",
            ["lib_id", lib_id],
            ["at", x, y, angle],
            ["unit", 1],
            ["exclude_from_sim", "yes"],
            ["in_bom", "no"],
            ["on_board", "yes"],
            ["dnp", "no"],
            ["fields_autoplaced", "yes"],
            ["uuid", inst_uuid],
        ]
    )

    # Reference property.
    symbol.append(
        Sexp(
            [
                "property",
                "Reference",
                pwr_ref,
                ["at", x, y - 1.27, 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )
    )

    # Value property.
    symbol.append(
        Sexp(
            [
                "property",
                "Value",
                net_name,
                ["at", x, y - 3.81, 0],
                ["effects", ["font", ["size", 1.27, 1.27]]],
            ]
        )
    )

    # Footprint property.
    symbol.append(
        Sexp(
            [
                "property",
                "Footprint",
                "",
                ["at", x, y, 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )
    )

    # Datasheet property.
    symbol.append(
        Sexp(
            [
                "property",
                "Datasheet",
                "",
                ["at", x, y, 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )
    )

    # Pin entry (power symbols have a single pin "1").
    pin_uuid = _gen_uuid(f"pwr_pin:{net_name}:{x}:{y}:{_pwr_counter[0]}")
    symbol.append(Sexp(["pin", '"1"', ["uuid", pin_uuid]]))

    # Instances section.
    symbol.append(
        Sexp(
            [
                "instances",
                [
                    "project",
                    "SKiDL-Generated",
                    [
                        "path",
                        (
                            uuid_path
                            if uuid_path is not None
                            else f"/{_gen_uuid('root_schematic')}"
                        ),
                        ["reference", pwr_ref],
                        ["unit", 1],
                    ],
                ],
            ]
        )
    )

    return symbol


def _pwr_flag_to_sexp(x, y, net_name, uuid_path=None):
    """Generate a ``power:PWR_FLAG`` instance coincident with (x, y).

    PWR_FLAG is a special power symbol whose single pin is ``power_out``; placed
    coincident with a rail's power-symbol pin it tells ERC the rail is driven
    (from off-board), clearing ``power_pin_not_driven`` WITHOUT a real source.
    It connects by physical coincidence (not by name -- every PWR_FLAG shares the
    value "PWR_FLAG"), so it must sit exactly on a point already on the net.

    Emitted at instance-angle 0 so its pin lands at (x, y); one flag per rail is
    enough because the coincident point is on the globally-named power net.
    """
    _flg_counter[0] += 1
    flg_ref = f"#FLG{_flg_counter[0]:03d}"
    x = _round_mm(x)
    y = _round_mm(y)
    inst_uuid = _gen_uuid(f"flg:{net_name}:{x}:{y}:{_flg_counter[0]}")

    symbol = Sexp(
        [
            "symbol",
            ["lib_id", "power:PWR_FLAG"],
            ["at", x, y, 0],
            ["unit", 1],
            ["exclude_from_sim", "yes"],
            ["in_bom", "no"],
            ["on_board", "yes"],
            ["dnp", "no"],
            ["fields_autoplaced", "yes"],
            ["uuid", inst_uuid],
        ]
    )
    symbol.append(
        Sexp(
            [
                "property",
                "Reference",
                flg_ref,
                ["at", x, y + 3.81, 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )
    )
    symbol.append(
        Sexp(
            [
                "property",
                "Value",
                "PWR_FLAG",
                ["at", x, y + 2.54, 0],
                ["effects", ["font", ["size", 1.27, 1.27]]],
            ]
        )
    )
    for prop in ("Footprint", "Datasheet"):
        symbol.append(
            Sexp(
                [
                    "property",
                    prop,
                    "",
                    ["at", x, y, 0],
                    ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
                ]
            )
        )
    pin_uuid = _gen_uuid(f"flg_pin:{net_name}:{x}:{y}:{_flg_counter[0]}")
    symbol.append(Sexp(["pin", '"1"', ["uuid", pin_uuid]]))
    symbol.append(
        Sexp(
            [
                "instances",
                [
                    "project",
                    "SKiDL-Generated",
                    [
                        "path",
                        (
                            uuid_path
                            if uuid_path is not None
                            else f"/{_gen_uuid('root_schematic')}"
                        ),
                        ["reference", flg_ref],
                        ["unit", 1],
                    ],
                ],
            ]
        )
    )
    return symbol


def _append_pwr_flags(elements, uuid_path):
    """Append one ``power:PWR_FLAG`` per undriven rail that has a power-symbol
    instance in *elements* (and has not been flagged on an earlier sheet).

    Scans the already-built ``elements`` for ``power:<name>`` instances (skipping
    PWR_FLAG itself), and for each name in ``_pwr_flag_net_names`` not yet in
    ``_pwr_flagged`` drops a coincident flag on the first such instance's pin.
    ``_pwr_flagged`` is module-global so exactly ONE flag ships per rail across
    every sheet.
    """
    if not _pwr_flag_net_names:
        return
    # Deterministic: first instance in element order per rail name.
    seen = OrderedDict()
    for el in elements:
        if not (hasattr(el, "__getitem__") and len(el) and el[0] == "symbol"):
            continue
        lib_id = None
        at = None
        for sub in el:
            if not (hasattr(sub, "__getitem__") and len(sub) >= 2):
                continue
            if sub[0] == "lib_id" and isinstance(sub[1], str):
                lib_id = sub[1]
            elif sub[0] == "at" and len(sub) >= 3:
                at = (sub[1], sub[2])
        if not _is_power_libid(lib_id) or lib_id == "power:PWR_FLAG":
            continue
        name = _power_net_from_libid(lib_id)
        if name in _pwr_flag_net_names and name not in _pwr_flagged and at is not None:
            seen.setdefault(name, at)
    for name, (x, y) in seen.items():
        elements.append(_pwr_flag_to_sexp(x, y, name, uuid_path=uuid_path))
        _pwr_flagged.add(name)


def _gen_uuid(name=""):
    """Generate a deterministic UUID from *name*, or a random one if empty."""
    if not name:
        return str(uuid.uuid4())
    return str(uuid.uuid5(_NAMESPACE_UUID, name))


def _round_mm(val, ndigits=2):
    """Round a value to *ndigits* decimal places for mm output.

    KiCad 9 uses mm with decimal precision.  The old integer .round()
    was fine for kicad5 (mils) but destroys sub-mm precision needed
    for pin-to-wire alignment in KiCad 9.
    """
    return round(val, ndigits)


# ---------------------------------------------------------------------------
# Paper sizes
# ---------------------------------------------------------------------------

A_SIZES = OrderedDict(
    [
        ("A4", (297, 210)),
        ("A3", (420, 297)),
        ("A2", (594, 420)),
        ("A1", (841, 594)),
        ("A0", (1189, 841)),
    ]
)


def _pick_paper_size(bbox):
    """Choose the smallest A-size paper that fits *bbox* (in mils)."""
    import math

    w = abs(bbox.w) if bbox.w and not math.isinf(bbox.w) else 0
    h = abs(bbox.h) if bbox.h and not math.isinf(bbox.h) else 0

    # Convert bbox dimensions from mils to mm.
    w_mm = w * 0.0254 if w else 0
    h_mm = h * 0.0254 if h else 0

    for name, (pw, ph) in A_SIZES.items():
        if w_mm <= pw and h_mm <= ph:
            return name
    return "A0"


# ---------------------------------------------------------------------------
# Part → S-expression
# ---------------------------------------------------------------------------


def part_to_sexp(part, uuid_path, tx=Tx()):
    """Create S-expression for a symbol instance.

    Applies part transform and sheet transform (Y-flip is in sheet_tx).
    Adds ``(mirror y)`` because the sheet transform's Y-flip negates pin
    Y-offsets, and KiCad must be told to mirror pin positions to match.

    Args:
        part: SKiDL Part object (placed).
        uuid_path: Hierarchical UUID path to node containing this part.
        tx: Sheet-level transformation matrix.

    Returns:
        Sexp: Symbol S-expression.
    """
    part_tx = getattr(part, "tx", Tx())
    angle, mx, my = part_tx.analyze_transform()
    if mx:
        mirror = ["mirror", "x"]
    elif my:
        mirror = ["mirror", "y"]
    else:
        mirror = []
    tx = part_tx * tx
    origin = Point(_round_mm(tx.origin.x), _round_mm(tx.origin.y))
    unit_num = getattr(part, "num", 1)

    lib_name = _lib_nickname(part)
    part_name = part.name or "Unknown"
    lib_id = f"{lib_name}:{part_name}"

    # KiCad reference for the symbol instance. All units of a multi-unit part share
    # ONE reference ("U1") and are distinguished only by (unit N); skidl's PartUnit
    # ref is the compound "U1.uA", which KiCad reads as a DISTINCT component -- so
    # each amp unit lost its shared power unit's pins (missing_power_pin), the
    # netlist-from-schematic diverged from the logical netlist, and the compound ref
    # tripped kicad-sch-api's validator (B1/B2). Emit the parent's base ref instead.
    base_ref = getattr(getattr(part, "parent", None), "ref", None) or part.ref

    symbol_list = [
        "symbol",
        ["lib_id", lib_id],
        ["at", origin.x, origin.y, angle],
        mirror,
        ["unit", unit_num],
        ["exclude_from_sim", "no"],
        ["in_bom", "yes"],
        ["on_board", "yes"],
        ["dnp", "no"],
        ["fields_autoplaced", "yes"],
        ["uuid", _gen_uuid(part.hiername)],
    ]
    if not mirror:
        symbol_list.remove([])
    symbol = Sexp(symbol_list)

    # Reference
    symbol.append(
        Sexp(
            [
                "property",
                "Reference",
                base_ref,
                ["at", origin.x, origin.y - 2.54, angle],
                ["effects", ["font", ["size", 1.27, 1.27]], ["justify", "left"]],
            ]
        )
    )

    # Value
    symbol.append(
        Sexp(
            [
                "property",
                "Value",
                str(part.value),
                ["at", origin.x, origin.y + 2.54, angle],
                ["effects", ["font", ["size", 1.27, 1.27]], ["justify", "left"]],
            ]
        )
    )

    # Footprint
    symbol.append(
        Sexp(
            [
                "property",
                "Footprint",
                getattr(part, "footprint", ""),
                ["at", origin.x, origin.y, angle],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )
    )

    # Datasheet
    symbol.append(
        Sexp(
            [
                "property",
                "Datasheet",
                getattr(part, "datasheet", "~") or "~",
                ["at", origin.x, origin.y, angle],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )
    )

    # Description
    symbol.append(
        Sexp(
            [
                "property",
                "Description",
                getattr(part, "description", "") or "",
                ["at", origin.x, origin.y, angle],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )
    )

    # Custom fields from part.fields dict.
    y_offset = 5.08
    if hasattr(part, "fields") and part.fields:
        for field_name, field_value in part.fields.items():
            if field_name.lower() in (
                "reference",
                "value",
                "footprint",
                "datasheet",
                "description",
            ):
                continue
            if field_value and str(field_value).strip():
                symbol.append(
                    Sexp(
                        [
                            "property",
                            field_name,
                            str(field_value),
                            ["at", origin.x, origin.y + y_offset, angle],
                            [
                                "effects",
                                ["font", ["size", 1.27, 1.27]],
                                ["hide", "yes"],
                            ],
                        ]
                    )
                )
                y_offset += 1.27

    # Pin entries (required by KiCad 8/9 for connectivity tracking).
    for pin in part.pins:
        pin_num = str(pin.num)
        pin_uuid = _gen_uuid(f"{part.hiername}_pin_{pin_num}")
        symbol.append(Sexp(["pin", f'"{pin_num}"', ["uuid", pin_uuid]]))

    # Instances section (required by KiCad 8/9 for correct reference display).
    symbol.append(
        Sexp(
            [
                "instances",
                [
                    "project",
                    "SKiDL-Generated",
                    [
                        "path",
                        f"{uuid_path}",
                        ["reference", base_ref],
                        ["unit", unit_num],
                    ],
                ],
            ]
        )
    )

    return symbol


# ---------------------------------------------------------------------------
# Library symbol definition
# ---------------------------------------------------------------------------


def part_to_lib_symbol_definition(part):
    """Extract library symbol definition from a part's draw_cmds.

    Args:
        part: SKiDL Part object.

    Returns:
        list: Nested list for the lib_symbols section.
    """
    lib_name = _lib_nickname(part)
    part_name = part.name or "Unknown"
    lib_id = f"{lib_name}:{part_name}"

    # Prefer embedding the raw library symbol VERBATIM (retained at parse time as
    # part._raw_lib_sexp). KiCad's lib_symbol_mismatch check is structural, so a
    # body regenerated from draw_cmds -- which omits fields the library carries
    # (pin_names hide, ki_keywords/ki_fp_filters, per-pin metadata ...) -- always
    # mismatches its library copy. Only the top-level name needs the LIB:NAME id
    # (the library file names it bare); inner unit symbols keep NAME_u_s. Parts
    # without a retained subtree (extends children, tool=SKIDL custom symbols,
    # in-file power clones) fall through to the generated path below.
    raw = getattr(part, "_raw_lib_sexp", None)
    if raw is not None:
        from copy import deepcopy

        sym = deepcopy(raw)  # fresh copy per sheet: add_quotes is in-place
        sym[1] = lib_id
        return sym

    symbol_def = [
        "symbol",
        lib_id,
        ["pin_numbers", ["hide", "yes"]],
        ["pin_names", ["offset", 0]],
        ["exclude_from_sim", "no"],
        ["in_bom", "yes"],
        ["on_board", "yes"],
    ]

    # Standard properties.
    symbol_def.extend(
        [
            [
                "property",
                "Reference",
                part.ref_prefix or "U",
                ["at", 2.032, 0, 90],
                ["effects", ["font", ["size", 1.27, 1.27]]],
            ],
            [
                "property",
                "Value",
                part_name,
                ["at", 0, 0, 90],
                ["effects", ["font", ["size", 1.27, 1.27]]],
            ],
            [
                "property",
                "Footprint",
                "",
                ["at", 0, 0, 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ],
            [
                "property",
                "Datasheet",
                getattr(part, "datasheet", "~") or "~",
                ["at", 0, 0, 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ],
        ]
    )

    if hasattr(part, "description") and part.description:
        symbol_def.append(
            [
                "property",
                "Description",
                part.description,
                ["at", 0, 0, 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]],
            ]
        )

    # Process draw_cmds into sub-symbols.
    if hasattr(part, "draw_cmds") and part.draw_cmds:
        # Common graphics (unit 0).
        if 0 in part.draw_cmds:
            graphics = [
                copy.deepcopy(cmd) for cmd in part.draw_cmds[0] if cmd[0] != "pin"
            ]
            if graphics:
                symbol_def.append(["symbol", f"{part_name}_0_1"] + graphics)

        # Per-unit graphics and pins.
        for unit_num, draw_cmds in part.draw_cmds.items():
            if unit_num == 0:
                continue
            pin_cmds = [copy.deepcopy(cmd) for cmd in draw_cmds if cmd[0] == "pin"]
            graphics = [
                copy.deepcopy(cmd)
                for cmd in draw_cmds
                if cmd[0] not in ("pin", "property")
            ]
            if pin_cmds or graphics:
                unit_sym = ["symbol", f"{part_name}_{unit_num}_{unit_num}"]
                unit_sym.extend(graphics)
                unit_sym.extend(pin_cmds)
                symbol_def.append(unit_sym)

    symbol_def.append(["embedded_fonts", "no"])

    return symbol_def


# ---------------------------------------------------------------------------
# Wires, junctions, net labels
# ---------------------------------------------------------------------------


def wire_to_sexp(net, wire, tx=Tx(), junctions=None):
    """Create S-expression for wire segments.

    Splits segments at junction points so KiCad properly connects
    pins at wire endpoints.  Without this, a junction in the middle
    of a wire does **not** create separate connectivity segments.

    Args:
        net: Net associated with the wire.
        wire: List of Segments.
        tx: Transformation matrix.
        junctions: Optional list of junction Points (pre-transform).

    Returns:
        list[Sexp]: Wire S-expression objects.
    """

    # Build set of junction coordinates in mm (post-transform).
    junc_pts = set()
    if junctions:
        for j in junctions:
            jt = j * tx
            junc_pts.add((_round_mm(jt.x), _round_mm(jt.y)))

    def _make_wire(x1, y1, x2, y2):
        return Sexp(
            [
                "wire",
                ["pts", ["xy", x1, y1], ["xy", x2, y2]],
                ["stroke", ["width", 0], ["type", "default"]],
                ["uuid", _gen_uuid(f"wire:{net.name}:{x1}:{y1}:{x2}:{y2}")],
            ]
        )

    wires = []
    for segment in wire:
        w = segment * tx
        x1, y1 = _round_mm(w.p1.x), _round_mm(w.p1.y)
        x2, y2 = _round_mm(w.p2.x), _round_mm(w.p2.y)

        # Collect junction points that lie strictly between endpoints.
        splits = []
        for jx, jy in junc_pts:
            if x1 == x2 == jx:  # Vertical wire.
                lo, hi = min(y1, y2), max(y1, y2)
                if lo < jy < hi:
                    splits.append(jy)
            elif y1 == y2 == jy:  # Horizontal wire.
                lo, hi = min(x1, x2), max(x1, x2)
                if lo < jx < hi:
                    splits.append(jx)

        if not splits:
            wires.append(_make_wire(x1, y1, x2, y2))
        else:
            # Split into ordered sub-segments.
            if x1 == x2:  # Vertical – split by Y.
                pts = sorted({y1, y2, *splits})
                if y1 > y2:
                    pts.reverse()
                for a, b in zip(pts, pts[1:]):
                    wires.append(_make_wire(x1, a, x2, b))
            else:  # Horizontal – split by X.
                pts = sorted({x1, x2, *splits})
                if x1 > x2:
                    pts.reverse()
                for a, b in zip(pts, pts[1:]):
                    wires.append(_make_wire(a, y1, b, y2))

    return wires


def junction_to_sexp(net, junctions, tx=Tx()):
    """Create S-expression for junction points.

    Args:
        net: Net associated with the junctions.
        junctions: List of junction Points.
        tx: Transformation matrix.

    Returns:
        list[Sexp]: Junction S-expression objects.
    """
    result = []
    for junction in junctions:
        pt = junction * tx
        x, y = _round_mm(pt.x), _round_mm(pt.y)
        result.append(
            Sexp(
                [
                    "junction",
                    ["at", x, y],
                    ["diameter", 0],
                    ["color", 0, 0, 0, 0],
                    ["uuid", _gen_uuid(f"junction:{x}:{y}")],
                ]
            )
        )
    return result


def calc_pin_dir(pin):
    """Calculate pin direction accounting for part transformation matrix."""

    # Copy the part trans. matrix, but remove the translation vector, leaving only scaling/rotation stuff.
    tx = pin.part.tx
    tx = Tx(a=tx.a, b=tx.b, c=tx.c, d=tx.d)

    # Use the pin orientation to compute the pin direction vector.
    pin_vector = {
        "U": Point(0, 1),
        "D": Point(0, -1),
        "L": Point(-1, 0),
        "R": Point(1, 0),
    }[pin.orientation]

    # Rotate the direction vector using the part rotation matrix.
    pin_vector = pin_vector * tx

    # Create an integer tuple from the rotated direction vector.
    pin_vector = (int(round(pin_vector.x)), int(round(pin_vector.y)))

    # Return the pin orientation based on its rotated direction vector.
    return {
        (0, 1): "U",
        (0, -1): "D",
        (-1, 0): "L",
        (1, 0): "R",
    }[pin_vector]


def net_label_to_sexp(
    pin, tx=Tx(), force=False, local=False, at_world=None, uuid_path=None, kind=None
):
    """Create S-expression for a net label at a pin stub.

    Generates a power symbol if the net name matches a known KiCad power
    symbol; otherwise a label of the requested ``kind``.

    Args:
        pin: Pin with net connection.
        tx: Transformation matrix.
        force: If True, skip the stub check (used for NetTerminal pins
            which always need a label regardless of stub state).
        local: Back-compat boolean used only when ``kind`` is None: True ->
            ``kind="local"``, False -> ``kind="global"``.
        kind: Label kind, one of ``"local"`` / ``"global"`` / ``"hier"``. When
            given it wins over ``local``.

            * ``"local"`` -- a SHEET-LOCAL ``label`` (stage 24 scoping fix): a
              ``global_label`` connects by name across every sheet in the
              project, so a sheet-INTERNAL net named e.g. ``SW3`` on two sheets
              would silently merge -- a Blocker-B-class leak. A local ``label``
              is confined to its sheet.
            * ``"global"`` -- a project-wide ``global_label`` (the default
              cross-sheet connection: boundary nets connect by name).
            * ``"hier"`` -- a ``hierarchical_label`` (shape bidirectional): the
              child half of the KiCad hierarchical interconnect, paired by name
              with the sheet pin on the parent's sheet symbol. Used for
              boundary nets when the ``hierarchical_sheet_pins`` option is ON.
              KiCad merges a same-named ``label`` on the sheet into it, so the
              emission audit can still add a local unifier.

            All three sit ERC-clean on a pin or a wire end (only the benign
            isolated_pin_label warning). ``global``/``hier`` carry
            ``(shape bidirectional)``; a plain ``label`` does not.

    Returns:
        Sexp or None: Label/power symbol S-expression, or None if no label needed.
    """
    if not force and (not pin.stub or not pin.is_connected()):
        return None

    if isinstance(getattr(pin, "net", None), NCNet):
        return None

    # Check if this net is a power net (stock power-lib name OR classified via
    # drive==POWER / pattern). If so, emit a power symbol instance instead of a
    # label -- the standard KiCad idiom, and what removes power nets from the
    # A* router. Non-stock rail names get a cloned in-file (power) symbol.
    if pin.is_connected() and _net_wants_power_symbol(pin.net):
        pwr = _power_symbol_to_sexp(pin, pin.net.name, tx, uuid_path=uuid_path)
        if pwr:
            return pwr

    # Sheet-INTERNAL nets get a local ``label`` (sheet-scoped); cross-sheet /
    # boundary nets keep ``global_label`` for project-wide name-based
    # connectivity, or a ``hierarchical_label`` when the hierarchical_sheet_pins
    # option routes them through sheet pins. A local label is a global_label
    # Sexp minus the ``shape`` field.
    if kind is None:
        kind = "local" if local else "global"
    label_type = {
        "local": "label",
        "global": "global_label",
        "hier": "hierarchical_label",
    }[kind]

    # Position at pin location (Y-flip is already in sheet_tx), unless the
    # caller overrides with an explicit placement-space point (deconflict-stub
    # closure labels sit at the pin's deconflicted STUB END, not on the pin).
    if at_world is not None:
        pt = at_world * tx
    else:
        pin_pt = getattr(pin, "pt", Point(pin.x, pin.y))
        part_tx = getattr(pin.part, "tx", Tx())
        pt = pin_pt * part_tx * tx

    # Angle + justification chosen so the label text always extends AWAY from
    # the pin (and therefore clear of the part body), for every pin direction
    # including mirrored parts -- calc_pin_dir() already folds the part's full
    # transform (rotation + mirror) into the reported direction.
    #
    # Verified by rendering a resistor rotated/mirrored into all four
    # orientations: horizontal labels reach out to the side, vertical labels
    # run clear above/below the body. The angle comes from the shared
    # _PIN_LABEL_ANGLE table (so it can't drift from the power-symbol emitter);
    # justify is "left" for the 0/90 angles and "right" otherwise, which exactly
    # reproduces R:(180,right) L:(0,left) U:(270,right) D:(90,left).
    angle = _PIN_LABEL_ANGLE[calc_pin_dir(pin)]
    justify = "left" if angle in (0, 90) else "right"

    fields = [label_type, pin.net.name]
    if kind in ("global", "hier"):
        # global_label / hierarchical_label carry a shape; a plain label does not.
        fields.append(["shape", "bidirectional"])
    fields.extend(
        [
            ["at", _round_mm(pt.x), _round_mm(pt.y), angle],
            ["effects", ["font", ["size", 1.27, 1.27]], ["justify", justify]],
            ["uuid", _gen_uuid(f"label:{pin.net.name}:{pt.x}:{pt.y}")],
        ]
    )
    return Sexp(fields)


# ---------------------------------------------------------------------------
# Snap-aware emitters: label suppression, power buses, no-connect flags
# ---------------------------------------------------------------------------


def _kicad_pin_pos(pin, part_tx, sheet_tx):
    """Compute pin position as KiCad renders it from symbol placement.

    KiCad's pin transform order: Y-flip, rotate(-angle), then mirror.
    The angle from analyze_transform() is the visual angle in SKiDL's Y-up
    space; KiCad uses its negative because the sheet Y-flip reverses rotation.

    NOTE (stage 19): two coordinate formulas coexist. Emission elsewhere uses
    ``pin.pt * part.tx * sheet_tx`` (Point.__mul__); this path re-derives the
    rotation via ``analyze_transform``. They were verified to agree for every
    dihedral ``part.tx`` (all rotations/mirrors snap produces), but would
    silently diverge if a non-dihedral (shear/non-uniform) tx ever appeared. The
    connectivity audit (_audit_sheet_connectivity) would catch any real desync.
    """
    import math

    angle_deg, mx, my = part_tx.analyze_transform()
    composed = part_tx * sheet_tx
    ox = _round_mm(composed.origin.x)
    oy = _round_mm(composed.origin.y)

    px, py = pin.x, -pin.y

    theta = math.radians(-angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rx = px * cos_t - py * sin_t
    ry = px * sin_t + py * cos_t

    if mx:
        ry = -ry
    if my:
        rx = -rx

    return _round_mm(ox + rx), _round_mm(oy + ry)


def _render_xy(lx, ly, part_tx, sheet_tx):
    """Transform a part-local point to KiCad render-mm (same convention as
    _kicad_pin_pos, but for arbitrary points like bbox corners)."""
    import math

    angle_deg, mx, my = part_tx.analyze_transform()
    composed = part_tx * sheet_tx
    ox, oy = _round_mm(composed.origin.x), _round_mm(composed.origin.y)
    px, py = lx, -ly
    theta = math.radians(-angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    rx, ry = px * c - py * s, px * s + py * c
    if mx:
        ry = -ry
    if my:
        rx = -rx
    return _round_mm(ox + rx), _round_mm(oy + ry)


# ---------------------------------------------------------------------------
# Title block
# ---------------------------------------------------------------------------


def create_title_block_sexp(title):
    """Create a title block S-expression."""
    return [
        "title_block",
        ["title", title],
        ["date", datetime.date.today().isoformat()],
        ["company", ""],
        ["comment", 1, "Generated with SKiDL"],
        ["comment", 2, ""],
        ["comment", 3, ""],
        ["comment", 4, ""],
    ]


# ---------------------------------------------------------------------------
# Hierarchical sheet reference
# ---------------------------------------------------------------------------

_HGRID_MM = 1.27  # KiCad 50-mil grid; sheet pins/stubs/labels must land on it.


def _snap_grid(v):
    """Quantize a mm coordinate to the 1.27 mm grid (2-dp rounded)."""
    return _round_mm(round(v / _HGRID_MM) * _HGRID_MM)


def _descendant_parts(node):
    """Every real part in this node's subtree (its own + all descendants')."""
    parts = list(getattr(node, "parts", []))
    for child in getattr(node, "children", {}).values():
        parts.extend(_descendant_parts(child))
    return parts


def _hier_boundary_nets(node):
    """Boundary nets of the hierarchical sheet BOX for ``node``.

    A sheet box represents the node's WHOLE subtree, so a net is a boundary of
    the box iff it has a pin on a part anywhere in the subtree AND a pin on a
    part outside it. This is deliberately broader than
    ``SchNode.get_boundary_nets()`` (which scans only ``node.parts`` -- correct
    for a leaf): it also catches a TRANSIT net that passes through an
    intermediate sheet holding only child sheets (no own parts). Without the
    descendant closure such a net gets no sheet pin on the intermediate's box
    and the transit connection breaks (verified by the 3-level canary).

    Returns nets in a name-sorted, deterministic order.
    """
    sub_parts = _descendant_parts(node)
    sub_ids = {id(p) for p in sub_parts}
    boundary = []
    seen = set()
    for part in sub_parts:
        for pin in part:
            if not pin.is_connected():
                continue
            net = pin.net
            if id(net) in seen:
                continue
            seen.add(id(net))
            if any(id(p.part) not in sub_ids for p in net.pins):
                boundary.append(net)
    boundary.sort(key=lambda n: str(getattr(n, "name", "")))
    return boundary


def _sheet_stub_wire_sexp(x1, y1, x2, y2, uuid_seed):
    """A short horizontal wire from a sheet pin out to its name label."""
    return Sexp(
        [
            "wire",
            ["pts", ["xy", _round_mm(x1), _round_mm(y1)], ["xy", _round_mm(x2), _round_mm(y2)]],
            ["stroke", ["width", 0], ["type", "default"]],
            ["uuid", _gen_uuid(uuid_seed)],
        ]
    )


def _name_label_sexp(name, x, y, kind, uuid_seed, angle=0, justify="right"):
    """A bare name label at (x, y): local ``label`` or ``hierarchical_label``.

    Used for parent-side sheet-pin stubs, where there is no Pin object to feed
    ``net_label_to_sexp``. ``kind`` is "local" (root parent: connects sibling
    boxes by name) or "hier" (non-root/intermediate parent: also exports the
    transit net upward through the intermediate's own sheet pin).
    """
    if kind == "hier":
        fields = ["hierarchical_label", name, ["shape", "bidirectional"]]
    else:
        fields = ["label", name]
    fields.extend(
        [
            ["at", _round_mm(x), _round_mm(y), angle],
            ["effects", ["font", ["size", 1.27, 1.27]], ["justify", justify]],
            ["uuid", _gen_uuid(uuid_seed)],
        ]
    )
    return Sexp(fields)


def create_hierarchical_sheet_sexp(node, sheet_uuid, sheet_tx, parent_is_root=True):
    """Create a hierarchical sheet S-expression for insertion into a parent sheet.

    With the ``hierarchical_sheet_pins`` option ON this emits the parent half of
    the KiCad hierarchical interconnect: a sheet pin per boundary net on the
    box's left edge, each wired out to a short stub carrying a name label. The
    label kind depends on whether the PARENT sheet is the root: on the root a
    sheet-local ``label`` joins same-named pins across sibling boxes; on a
    non-root (intermediate) sheet a ``hierarchical_label`` also EXPORTS the net
    upward through that intermediate's own sheet pin (so a transit net threads
    through every level). Same-named labels/pins connect by name -- no inter-box
    routing.

    Args:
        node: SchNode for the child sheet.
        sheet_uuid: UUID of this sheet (for the "uuid" property).
        sheet_tx: Transformation matrix of the parent sheet.
        parent_is_root: True if the sheet this box is drawn on is the root sheet.

    Returns:
        (Sexp, list[Sexp]): the ``sheet`` S-expression, and a list of sibling
        elements (stub wires + name labels) to append alongside it in the parent
        sheet's element list.
    """
    bbox = node.bbox * node.tx * sheet_tx
    # With the option ON, snap the box origin to the grid so pins/stubs/labels on
    # the left edge land on-grid (the box size may keep 0.01 mm rounding -- only
    # connection points must be grid-true; finding B cleared the off-grid class).
    # With the option OFF the box has no pins, so keep the prior 0.01 mm rounding
    # verbatim (default-path output stays byte-identical).
    if _EMIT_HIER_SHEET_PINS:
        bx = _snap_grid(bbox.ll.x)
        by = _snap_grid(bbox.ll.y)
    else:
        bx = _round_mm(bbox.ll.x)
        by = _round_mm(bbox.ll.y)
    bw = _round_mm(bbox.w)
    bh = _round_mm(bbox.h)

    extras = []
    pins = []
    if _EMIT_HIER_SHEET_PINS:
        pin_spacing = 2.54  # mm between pins (a grid multiple -> stays on-grid)
        stub_len = 2.54  # outward stub length
        boundary_nets = [
            net
            for net in _hier_boundary_nets(node)
            if not _net_wants_power_symbol(net)
            and not (getattr(net, "stub", False) or getattr(net, "_stub", False))
        ]
        # Grow the box so all pins fit on the left edge (visual only).
        needed_h = pin_spacing * (len(boundary_nets) + 1)
        if needed_h > bh:
            bh = needed_h
        label_kind = "local" if parent_is_root else "hier"
        for i, net in enumerate(boundary_nets):
            pin_y = _snap_grid(by + pin_spacing * (i + 1))
            pins.append(
                Sexp(
                    [
                        "pin",
                        net.name,
                        "bidirectional",
                        ["at", bx, pin_y, 180],
                        ["effects", ["font", ["size", 1.27, 1.27]], ["justify", "left"]],
                        ["uuid", _gen_uuid(f"sheet_pin:{node.sheet_filename}:{net.name}")],
                    ]
                )
            )
            # Stub wire out to the left + a name label at its far end.
            far_x = _snap_grid(bx - stub_len)
            extras.append(
                _sheet_stub_wire_sexp(
                    bx, pin_y, far_x, pin_y,
                    f"sheet_stub_wire:{node.sheet_filename}:{net.name}",
                )
            )
            extras.append(
                _name_label_sexp(
                    net.name, far_x, pin_y, label_kind,
                    f"sheet_stub_label:{node.sheet_filename}:{net.name}",
                )
            )

    sheet = Sexp(
        [
            "sheet",
            ["at", bx, by],
            ["size", bw, bh],
            ["exclude_from_sim", "no"],
            ["in_bom", "yes"],
            ["on_board", "yes"],
            ["dnp", "no"],
            ["fields_autoplaced", "yes"],
            ["stroke", ["width", 0.1524], ["type", "solid"]],
            ["fill", ["color", 0, 0, 0, 0.0]],
            ["uuid", sheet_uuid],
            [
                "property",
                "Sheetname",
                node.name,
                ["at", bx, _round_mm(by - 0.7116), 0],
                [
                    "effects",
                    ["font", ["size", 1.27, 1.27]],
                    ["justify", "left", "bottom"],
                ],
            ],
            [
                "property",
                "Sheetfile",
                node.sheet_filename,
                ["at", bx, _round_mm(by + bh + 0.5846), 0],
                ["effects", ["font", ["size", 1.27, 1.27]], ["justify", "left", "top"]],
            ],
        ]
    )
    for pin in pins:
        sheet.append(pin)

    return sheet, extras


def hierarchical_label_to_sexp(net_name, pt_x, pt_y, angle=180):
    """Create a hierarchical_label S-expression for a boundary net in a child sheet.

    Args:
        net_name: Name of the boundary net.
        pt_x: X coordinate in mm.
        pt_y: Y coordinate in mm.
        angle: Label angle (degrees).

    Returns:
        Sexp: Hierarchical label S-expression.
    """
    return Sexp(
        [
            "hierarchical_label",
            net_name,
            ["shape", "bidirectional"],
            ["at", _round_mm(pt_x), _round_mm(pt_y), angle],
            ["effects", ["font", ["size", 1.27, 1.27]], ["justify", "left"]],
            ["uuid", _gen_uuid(f"hlabel:{net_name}:{pt_x}:{pt_y}")],
        ]
    )


# ---------------------------------------------------------------------------
# Sheet-level transform calculation (mirrors kicad5 calc_sheet_tx)
# ---------------------------------------------------------------------------

MILS_TO_MM = 0.0254


def _calc_sheet_tx(bbox):
    """Calculate transformation matrix for placing circuitry in a sheet.

    Mirrors the kicad5 calc_sheet_tx pattern:
      1. Y-flip via d=-1 (placement engine is Y-up, KiCad is Y-down)
      2. Mils-to-mm conversion via a/d scaling (KiCad 9 uses mm)
      3. Center content on the chosen paper size

    The Y-flip is built into this transform so callers must NOT apply
    tx_flip_y separately (that would double-flip and cancel it out).
    """
    paper = _pick_paper_size(bbox)
    pw, ph = A_SIZES[paper]  # mm

    # Apply Y-flip + mils→mm in one transform, then center on page.
    page_bbox = bbox * Tx(a=MILS_TO_MM, d=-MILS_TO_MM)
    page_ctr = Point(pw / 2, ph / 2)
    content_ctr = Point(
        (page_bbox.ll.x + page_bbox.ur.x) / 2,
        (page_bbox.ll.y + page_bbox.ur.y) / 2,
    )
    move = page_ctr - content_ctr

    # Snap centering offset to KiCad's 1.27mm grid (50 mils) so that
    # grid-aligned placement coordinates stay on-grid after the move.
    GRID_MM = 1.27
    move = Point(
        round(move.x / GRID_MM) * GRID_MM,
        round(move.y / GRID_MM) * GRID_MM,
    )

    tx = Tx(a=MILS_TO_MM, d=-MILS_TO_MM).move(move)

    return tx, paper


# ---------------------------------------------------------------------------
# Recursive hierarchy walker — node_to_sexp_schematic
# ---------------------------------------------------------------------------


def _power_lib_ids_in_elements(elements):
    """Return the set of power-symbol lib_ids (``power:*`` and ``SKiDL_rails:*``)
    referenced by symbol instances in ``elements``. Used to emit exactly the
    power-symbol definitions a sheet needs, so every emitted power instance has a
    matching lib_symbols definition."""
    found = set()
    for el in elements:
        if not (hasattr(el, "__getitem__") and len(el) and el[0] == "symbol"):
            continue
        for sub in el:
            if (
                hasattr(sub, "__getitem__")
                and len(sub) >= 2
                and sub[0] == "lib_id"
                and _is_power_libid(sub[1])
            ):
                found.add(sub[1])
    return found


# When True, audit each finished sheet for cross-net pin/label/power-symbol
# coincidences and log a loud WARNING for any it finds. This is a DIAGNOSTIC
# tripwire (stage 19) — the equivalence gate + native fallback downstream remain
# the enforcement; the audit just localizes a fusion to sheet/nets/coords. Cheap
# at these sheet sizes; flip off if it is ever noisy.
_EMIT_CONNECTIVITY_AUDIT = True

# When strict, a shared-coordinate fusion RAISES instead of only warning, so the
# skidl render aborts and the equivalence gate installs the native render
# instead (correctness stays safe). Off by default (warn-only); the circuit-synth
# skidl-render path sets SKIDL_AUDIT_STRICT=1. (stage 24)
_AUDIT_STRICT = os.environ.get("SKIDL_AUDIT_STRICT", "0") not in (
    "0",
    "",
    "false",
    "False",
)

# Hierarchical-sheet-pin interconnect mode. When True, a child sheet emits a
# ``hierarchical_label`` per boundary net and the parent's sheet symbol gets a
# matching ``pin`` (sheet pin) -- the KiCad hierarchical-sheet machinery the
# upstream fork is building out. When False (the default), boundary nets connect
# across sheets by NAME through the ``global_label`` each of their pins already
# carries, which is ERC-clean today; the hierarchical machinery is still
# INCOMPLETE (labels/pins are placed at sheet-edge slots, not yet wired to the
# net inside the child nor to the parent net), so turning it on currently
# produces label_dangling / pin_not_connected until that wiring lands. Set via
# the ``hierarchical_sheet_pins`` render option (write_top_schematic), so the
# feature is preserved and re-enableable without churn. See
# create_hierarchical_sheet_sexp + node_to_sexp_schematic.
_EMIT_HIER_SHEET_PINS = False

# When True (the ``power_stubs`` render option), each power pin renders its
# ``power:*`` symbol at the end of a short outward STUB WIRE (the classic KiCad
# look) instead of coincident with the pin. _power_symbol_to_sexp offsets the
# symbol one grid step along the pin's outward direction and records the pin->
# symbol wire in _power_stub_wires, which node_to_sexp_schematic drains into the
# sheet's elements. Default OFF -- symbols sit on the pin exactly as before.
_EMIT_POWER_STUBS = False
_POWER_STUB_LEN = 2.54  # mm outward stub (2 grid units)
_power_stub_wires = []  # per-sheet sink: pin->symbol stub wires to emit


class SheetConnectivityError(Exception):
    """A rendered sheet has a coordinate owned by more than one net (a fusion)."""


def _audit_sheet_connectivity(node, elements, backend, sheet_tx):
    """Log a WARNING for any coordinate owned by more than one net name.

    Builds ``{rounded (x,y) -> set(net names)}`` over every emitted connection
    anchor with a known net — component pins (via ``backend.pin_render_pos``),
    ``global_label`` / ``hierarchical_label`` / ``label`` text, and ``power:``
    symbol instances. A cell owned by >1 net is exactly the fusion mechanism
    Blocker B fixed and the phases-3/4 hardening must keep closed. Render-space
    version of snap's occupancy checker; catches all coincidence-class fusers.
    """
    if not _EMIT_CONNECTIVITY_AUDIT:
        return

    from collections import defaultdict

    def _key(x, y):
        return (round(float(x), 2), round(float(y), 2))

    cell_nets = defaultdict(set)

    # Component pins -> net name.
    for part in node.parts:
        if isinstance(part, NetTerminal):
            continue
        for pin in part:
            net = getattr(pin, "net", None)
            name = getattr(net, "name", None)
            if not name:
                continue
            try:
                x, y = backend.pin_render_pos(pin, sheet_tx)
            except Exception:  # noqa: BLE001 - audit must never break emission
                continue
            cell_nets[_key(x, y)].add(name)

    # Emitted labels + power symbols -> net name.
    for elem in elements:
        if not (hasattr(elem, "__getitem__") and len(elem) >= 1):
            continue
        tag = elem[0]
        if tag in ("global_label", "hierarchical_label", "label"):
            at = next(
                (
                    s
                    for s in elem
                    if hasattr(s, "__getitem__") and len(s) >= 3 and s[0] == "at"
                ),
                None,
            )
            if at and isinstance(elem[1], str):
                cell_nets[_key(at[1], at[2])].add(elem[1])
        elif tag == "symbol":
            lib_id = next(
                (
                    s
                    for s in elem
                    if hasattr(s, "__getitem__") and len(s) >= 2 and s[0] == "lib_id"
                ),
                None,
            )
            # PWR_FLAG is deliberately placed COINCIDENT with a rail's power
            # symbol (that is how it drives the net); its value "PWR_FLAG" is not
            # a net name, so excluding it keeps the audit from false-flagging the
            # intended coincidence as a cross-net fusion.
            if (
                lib_id
                and _is_power_libid(str(lib_id[1]))
                and str(lib_id[1]) != "power:PWR_FLAG"
            ):
                at = next(
                    (
                        s
                        for s in elem
                        if hasattr(s, "__getitem__") and len(s) >= 3 and s[0] == "at"
                    ),
                    None,
                )
                if at:
                    cell_nets[_key(at[1], at[2])].add(
                        _power_net_from_libid(str(lib_id[1]))
                    )

    from skidl.logger import active_logger

    sheet = getattr(node, "sheet_filename", None) or getattr(node, "name", "?")
    fusions = []
    for (x, y), names in sorted(cell_nets.items()):
        if len(names) > 1:
            active_logger.warning(
                "connectivity audit: sheet %s cell (%s, %s) shared by nets %s "
                "(cross-net coincidence -> KiCad would fuse them)",
                sheet,
                x,
                y,
                sorted(names),
            )
            fusions.append(((x, y), sorted(names)))
    if fusions and _AUDIT_STRICT:
        # Hard-fail so the caller (equivalence gate) falls back to the native
        # render rather than installing a schematic with a silent net fusion.
        raise SheetConnectivityError(
            f"sheet {sheet}: {len(fusions)} cross-net coordinate collision(s); "
            f"first: cell {fusions[0][0]} shared by {fusions[0][1]}"
        )


def _audit_and_force_pin_labels(node, elements, tx, uuid_path, label_kind):
    """Guarantee every net's on-sheet pins are CONNECTED in the emitted drawing.

    Per-pin "is it covered?" is not enough: the per-net A* fallback and the single
    backstop label can leave a net SPLIT -- most pins wired into one island and a
    boxed-in pin stranded on a dangling stub (that pin looks "covered" by its own
    stub-wire endpoint yet the net's drawing diverges from the netlist). This runs
    a union-find over the emitted wires AND same-name labels, finds each net whose
    on-sheet pins fall into more than one component, and drops a name label on any
    component that lacks one -- unifying the net by name. The label kind comes
    from ``label_kind(net)`` (local for internal, global for a boundary net, or
    local when the hierarchical_sheet_pins option carries the boundary net by a
    hier label -- a same-named local label merges into it, whereas a global there
    would silently bridge project-wide names the hier path deliberately scopes).
    This is the renderer-side mirror of the harness ``drawing_connectivity`` gate,
    closing the hole before the file ships. Returns the number of labels forced.
    Mutates ``elements`` in place.
    """
    from collections import OrderedDict, defaultdict

    from skidl.net import NCNet

    def _pt(x, y):
        return (_round_mm(x), _round_mm(y))

    # Union-find over coordinate keys (wire endpoints, pin points, label points).
    parent = {}

    def find(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Collect wire segments; union their endpoints (multi-point wires share
    # endpoints transitively). Collect label points keyed by name.
    segments = []
    label_pts = []  # (name, key)
    for el in elements:
        if not (hasattr(el, "__getitem__") and len(el)):
            continue
        tag = el[0]
        if tag == "wire":
            for s in el:
                if hasattr(s, "__getitem__") and len(s) and s[0] == "pts":
                    pts = [
                        _pt(xy[1], xy[2])
                        for xy in s[1:]
                        if hasattr(xy, "__getitem__") and len(xy) >= 3 and xy[0] == "xy"
                    ]
                    for i in range(1, len(pts)):
                        union(pts[0], pts[i])
                    if len(pts) >= 2:
                        segments.append((pts[0], pts[-1]))
                    break
        elif tag in ("global_label", "label", "hierarchical_label") and len(el) >= 2:
            name = el[1]
            for s in el:
                if hasattr(s, "__getitem__") and len(s) >= 3 and s[0] == "at":
                    key = _pt(s[1], s[2])
                    label_pts.append((name, key))
                    break

    # Labels connect by NAME across the sheet: union all like-named label points,
    # and remember which names sit at each component root.
    by_name = defaultdict(list)
    for name, key in label_pts:
        by_name[name].append(key)
    for name, keys in by_name.items():
        for k in keys[1:]:
            union(keys[0], k)

    # Pin records (on-sheet, non-power, non-NC). A pin's key auto-shares a
    # component with any wire ENDPOINT / coincident pin / label at the same point
    # (same tuple -> same union node). We deliberately do NOT union a pin that
    # merely lies on the MIDDLE of a wire: that is exactly where our geometry
    # model and kicad-cli's netlist can disagree (rounding, or a mid-span touch
    # KiCad won't fuse without a junction), and an over-optimistic union would
    # hide a real split. Being conservative can only ADD a redundant same-name
    # label (harmless), never miss a genuine disconnection.
    pin_recs = []  # (net, pin, key)
    for part in node.parts:
        if isinstance(part, NetTerminal):
            continue
        for pin in part:
            if not pin.is_connected():
                continue
            net = getattr(pin, "net", None)
            if net is None or isinstance(net, NCNet):
                continue
            if _net_wants_power_symbol(net):
                continue  # power pins are covered by their power symbol
            pp = getattr(pin, "pt", Point(pin.x, pin.y))
            w = pp * getattr(pin.part, "tx", Tx()) * tx
            pin_recs.append((net, pin, _pt(w.x, w.y)))

    # Names present at each component root (after unioning), to avoid a redundant
    # label on a component that already carries the net's name.
    root_names = defaultdict(set)
    for name, key in label_pts:
        root_names[find(key)].add(name)

    # Group pins per net; a net split across >1 component needs unifying labels.
    net_pins = OrderedDict()
    for net, pin, key in pin_recs:
        net_pins.setdefault(id(net), (net, []))[1].append((pin, key))

    def _pin_key(pk):
        p = pk[0]
        return (str(getattr(p.part, "ref", "") or ""), str(getattr(p, "num", "")))

    forced = 0
    for _nid, (net, pins) in net_pins.items():
        if len(pins) < 2:
            continue  # a lone on-sheet pin is closed by its own stub/label
        comps = defaultdict(list)
        for pin, key in pins:
            comps[find(key)].append((pin, key))
        if len(comps) <= 1:
            continue  # fully connected by wires / coincidence / shared labels
        name = getattr(net, "name", None)
        for root in sorted(
            comps, key=lambda r: min(_pin_key(pk) for pk in comps[r])
        ):
            if name and name in root_names.get(root, ()):
                continue  # this component already carries the net's name
            anchor = min(comps[root], key=_pin_key)[0]
            label = net_label_to_sexp(
                anchor, tx=tx, force=True, kind=label_kind(net), uuid_path=uuid_path
            )
            if label:
                elements.append(label)
                root_names[root].add(name)
                forced += 1
    if forced:
        from skidl.logger import active_logger

        active_logger.warning(
            "emission audit: forced %d unifying label(s) on split net(s) on sheet "
            "%s (a routed net left pins in separate islands; closed by name)",
            forced,
            getattr(node, "sheet_filename", None) or getattr(node, "name", "?"),
        )
    return forced


@export_to_all
def node_to_sexp_schematic(node, uuid_path, sheet_tx=Tx(), version=20230409):
    """Convert a SchNode tree to S-expression schematic(s).

    Follows the same recursive pattern as kicad5's node_to_eeschema():
    - Flattened nodes: return elements for inclusion in the parent sheet.
    - Unflattened nodes: write a separate .kicad_sch file and return a
      sheet reference for the parent.

    Args:
        node: SchNode to convert.
        uuid_path: Hierarchical UUID path to this node.
        sheet_tx: Parent sheet transformation matrix.
        version: S-expression version number (20240108 for kicad6, 20230409 for kicad8/9).

    Returns:
        list[Sexp]: S-expression for elements in this node (parts, wires, labels, sheet refs).
        dict: Dict of symbols in this node including in any flattened children.
        dict: Dict of power symbols used in this node including in any flattened children.
        str: For unflattened nodes, the filename of the .kicad_sch file written for this sheet.
    """
    # Fix filename extension for KiCad 6+ S-expression format.

    """Ensure node.sheet_filename uses .kicad_sch extension (SchNode defaults to .sch)."""
    node.sheet_filename = node.sheet_filename or "no_sheet_filename"
    node.sheet_filename = os.path.splitext(node.sheet_filename)[0] + ".kicad_sch"

    if node.flattened:
        # Flattened node doesn't get its own sheet, so remove its UUID.
        uuid_path = "/".join(uuid_path.split("/")[:-1])
        # Flattened node shares the parent sheet, so apply the parent's sheet_tx.
        tx = node.tx * sheet_tx
    else:
        # Compute the transform and sheet paper size for this node's sheet.
        tx, paper = _calc_sheet_tx(node.internal_bbox())

    # Storage for S-expression elements of schematic.
    elements = []
    # Reset the per-sheet power-stub-wire sink (power_stubs option). This node's
    # power symbols are emitted AFTER the child recursion below, so clearing here
    # is safe: each child recursion clears + drains its own sink before returning.
    _power_stub_wires.clear()

    # Collect lib_symbols needed for this node's parts.
    lib_symbols = {}
    for part in node.parts:
        if not isinstance(part, NetTerminal):
            lib_id = f"{_lib_nickname(part)}:{part.name or 'Unknown'}"
            lib_symbols[lib_id] = part

    # Power-symbol definitions are emitted later from the instances actually
    # placed on this sheet (see `_power_lib_ids_in_elements`), so no wire-based
    # pre-scan is needed here. `pwr_symbols` is kept only to satisfy the
    # flattened-node return contract.
    pwr_symbols = {}

    # Recurse into children.
    for i, child in enumerate(node.children.values()):
        # Skip phantom children (materialised by the children defaultdict but
        # never given parts): they have no .name / .sheet_filename and carry no
        # circuitry, so there is nothing to emit.
        if not child.parts and not child.children:
            continue
        # Give each child a unique UUID path based on its name and index.
        child.uuid = _gen_uuid(f"{child.name}_{i}")
        child_uuid_path = f"{uuid_path}/{child.uuid}"
        # Get elements for child sheet (or for inclusion in this sheet if child is flattened).
        sexp_list, part_dict, pwr_dict, _ = node_to_sexp_schematic(
            child, child_uuid_path, sheet_tx=tx, version=version
        )
        elements.extend(sexp_list)
        lib_symbols.update(part_dict)
        pwr_symbols.update(pwr_dict)

    # Collect net names that have real (non-NetTerminal) stubbed pins on this
    # sheet.  NetTerminal labels are redundant for these nets since the pins
    # will generate their own labels.
    nets_with_real_pins = set()
    for part in node.parts:
        if isinstance(part, NetTerminal):
            continue
        for pin in part:
            if pin.stub and pin.is_connected():
                nets_with_real_pins.add(pin.net.name)

    # ON-PIN LABELS (single-real-pin routed nets, incl. cross-sheet):
    # A routed net with exactly ONE real component pin ON THIS SHEET gets its
    # net label emitted AT that pin instead of at the NetTerminal pin (which
    # sits at the routing-channel edge, off the component body — the off-pin
    # step-out the user is chasing). Cross-sheet connectivity is preserved
    # because the relocated label keeps the same global_label NAME, which KiCad
    # resolves across all sheets regardless of position; the NetTerminal was
    # only ever the carrier for that name. Done purely at emit time: the net is
    # NOT marked stubbed and no place/route state is mutated.
    # The route wire (pin -> channel-edge NetTerminal pin) is suppressed for
    # this net: it can't be left in, because the dangling-wire purge below
    # treats the NetTerminal pin as an anchor (it iterates node.parts, which
    # includes NetTerminals), so the wire would survive with its far end on the
    # now-unlabelled terminal -> wire_dangling. With exactly one real pin there
    # is nothing else on-sheet for that wire to connect, so the on-pin label is
    # the net's sole connectivity marker — the stubbed-single-pin shape KiCad
    # accepts. Nets with >=2 real pins on-sheet keep the NetTerminal + their
    # wires (the wire carries real intra-sheet connectivity).
    node_part_ids = {id(p) for p in node.parts}
    _onpin_enabled = os.environ.get("SKIDL_ONPIN_LABELS", "1") != "0"

    # Sheet-INTERNAL vs cross-sheet classification for label scoping (stage 24).
    # A net with a pin on a part OUTSIDE this node is a boundary/cross-sheet net
    # and keeps its project-wide global_label; everything else is sheet-internal
    # and gets a sheet-local ``label`` so two sheets can reuse a net name (SW3,
    # FB3, ...) without a silent project-wide merge (the latent Blocker-B leak).
    if hasattr(node, "get_boundary_nets"):
        _boundary_net_ids = {id(n) for n in node.get_boundary_nets()}
    else:
        _boundary_net_ids = set()

    # KiCad rejects a hierarchical_label on the root sheet (no parent to pair
    # its sheet pin with), so the root's boundary nets fall back to a local
    # label. Same root guard the sheet-pin emitter uses.
    _is_root_sheet = uuid_path.count("/") <= 1

    def _is_internal(net):
        return net is not None and id(net) not in _boundary_net_ids

    def _label_kind(net):
        """Label kind for ``net`` on this sheet: 'local' | 'global' | 'hier'.

        Internal nets are always sheet-local. A boundary (cross-sheet) net
        connects by project-wide ``global_label`` by default; with the
        ``hierarchical_sheet_pins`` option ON it uses the KiCad hierarchical
        interconnect -- a ``hierarchical_label`` in the child paired by name to
        the parent's sheet pin -- except on the ROOT sheet, where a hierarchical
        label is illegal so it falls back to a local label (joined to the
        children through the sheet pins + Tier-2 parent stubs' local labels).
        """
        if _is_internal(net):
            return "local"
        if _EMIT_HIER_SHEET_PINS:
            return "local" if _is_root_sheet else "hier"
        return "global"

    def _onpin_real_pin(nt_net):
        """Return the lone on-sheet real pin for an on-pin-eligible net, else None.

        Eligible iff exactly one of the net's pins is a non-NetTerminal part on
        THIS node (off-sheet pins are ignored — they reconnect by label name),
        and that pin is routed (not already self-labelling via a stub).
        """
        if not _onpin_enabled:
            return None
        real_pins = [
            np
            for np in nt_net.pins
            if id(np.part) in node_part_ids and not isinstance(np.part, NetTerminal)
        ]
        if len(real_pins) != 1:
            return None
        rp = real_pins[0]
        if not rp.is_connected() or getattr(rp, "stub", False):
            return None  # stubbed pins already self-label; only handle routed pins
        return rp

    onpin_net_ids = set()  # id(net) of nets relocated on-pin (wire/junction suppressed)

    # Deconflict-stub mode (stage 25): every pin has a deconflicted on-grid stub
    # end (node._stub_ends); closure labels sit at those ends, one per connected
    # component, replacing the on-pin / one-backstop label paths below.
    deconflict = getattr(node, "_deconflict_stubs", False)
    stub_ends = getattr(node, "_stub_ends", {})
    terminal_pin_ids = getattr(node, "_stub_terminal_pins", set())

    # Deconflict mode: net ids that have a real (non-terminal) pin on THIS sheet
    # which the closure labeller will name. For a STUB (label-only) net such a
    # net's NetTerminal is redundant -- its stub wire is not drawn (bare pin
    # would dangle), so its label has nothing to attach to and would report
    # label_dangling. The real pin's same-named global_label already exports the
    # net, so suppress the terminal entirely. Routed nets keep their terminal
    # (its label sits on the A*-routed wire at route_pt, so it does not dangle).
    deconflict_real_pin_net_ids = set()
    if deconflict:
        for part in node.parts:
            if isinstance(part, NetTerminal):
                continue
            for pin in part:
                if id(pin) in stub_ends:
                    net = getattr(pin, "net", None)
                    if net is not None:
                        deconflict_real_pin_net_ids.add(id(net))

    # Generate part S-expressions.
    for part in node.parts:
        if isinstance(part, NetTerminal):
            # NetTerminals become net labels (unless a real pin already labels the net).
            pin = part.pins[0]
            if not pin.is_connected():
                continue
            # In deconflict mode the terminal emits its label at its stub end,
            # EXCEPT for a stub net whose net a real on-sheet pin already labels
            # (see deconflict_real_pin_net_ids) -- there the terminal is
            # redundant and its label would dangle, so skip it.
            if (
                deconflict
                and getattr(pin.net, "_stub", False)
                and id(pin.net) in deconflict_real_pin_net_ids
            ):
                continue
            if not deconflict and pin.net.name in nets_with_real_pins:
                continue
            real_pin = None if deconflict else _onpin_real_pin(pin.net)
            if real_pin is not None:
                # Emit the label at the real component pin, suppress the
                # NetTerminal (channel-edge) label + route wire for this net.
                label = net_label_to_sexp(
                    real_pin,
                    tx=tx,
                    force=True,
                    kind=_label_kind(pin.net),
                    uuid_path=uuid_path,
                )
                if label:
                    elements.append(label)
                    onpin_net_ids.add(id(pin.net))
                continue
            label = net_label_to_sexp(
                pin,
                tx=tx,
                force=True,
                kind=_label_kind(pin.net),
                at_world=stub_ends.get(id(pin)) if deconflict else None,
                uuid_path=uuid_path,
            )
            if label:
                elements.append(label)
        else:
            elements.append(part_to_sexp(part, uuid_path, tx=tx))

    # Generate wire S-expressions (split at junction points).
    # Skip nets that snap converted to stubs after routing — their routed
    # geometry is stale now that the parts moved, so they use labels instead.
    # Also skip nets relocated to an on-pin label above: their only route wire
    # ran to the now-unlabelled NetTerminal (channel-edge) pin, which the
    # dangling-wire purge cannot remove (that pin is itself an anchor). The
    # on-pin label is the net's sole connectivity marker. Keyed by id(net) so a
    # name-collision can't suppress a different net's wires.
    for net, wire in node.wires.items():
        if getattr(net, "_stub", False):
            continue
        if id(net) in onpin_net_ids:
            continue
        net_junctions = node.junctions.get(net, [])
        elements.extend(wire_to_sexp(net, wire, tx=tx, junctions=net_junctions))

    # Generate junction S-expressions.
    for net, junctions in node.junctions.items():
        if getattr(net, "_stub", False):
            continue
        if id(net) in onpin_net_ids:
            continue
        elements.extend(junction_to_sexp(net, junctions, tx=tx))

    # Tool-agnostic decision layer reaches kicad geometry/emission through the
    # backend adapter (see schematics/decisions.py + tools/kicad9/backend.py).
    # Wrap in a RenderContext so repeated pin_render_pos/dir queries within this
    # sheet are memoized. Created here, AFTER snap has finalized all part.tx in
    # gen_schematic, so no pre-snap stale positions can be cached (doc S7.6).
    from skidl.schematics import decisions as _decisions
    from skidl.schematics.backend import RenderContext
    from .backend import Kicad9Backend

    _backend = RenderContext(Kicad9Backend())

    # Suppress labels for snap-overlapping pins (one label per connected cluster).
    wired_pin_ids = _decisions.find_overlapping_pins(node, _backend, tx)

    # Connect co-linear power net pins with bus wires.
    bus_segments, bus_pin_ids = _decisions.find_power_bus_runs(node, _backend, tx)
    for x1, y1, x2, y2, net_name in bus_segments:
        elements.append(
            _backend.emit_wire(
                x1,
                y1,
                x2,
                y2,
                net_name=net_name,
                uuid_seed=f"pbus:{net_name}:{x1}:{y1}:{x2}:{y2}",
            )
        )
    wired_pin_ids.update(bus_pin_ids)

    # Generate T-junction wires from staggered snap placement.
    for tjw in getattr(node, "_tjunction_wires", []):
        x1_mil, y1_mil, x2_mil, y2_mil = tjw
        p1 = Point(x1_mil, y1_mil) * tx
        p2 = Point(x2_mil, y2_mil) * tx
        x1, y1 = _round_mm(p1.x), _round_mm(p1.y)
        x2, y2 = _round_mm(p2.x), _round_mm(p2.y)
        elements.append(
            Sexp(
                [
                    "wire",
                    ["pts", ["xy", x1, y1], ["xy", x2, y2]],
                    ["stroke", ["width", 0], ["type", "default"]],
                    ["uuid", _gen_uuid(f"tjwire:{x1}:{y1}:{x2}:{y2}")],
                ]
            )
        )
    # PROPOSAL (off unless snap._ENABLE_IC_FAN_PIN_REWIRE): drop a junction dot
    # at each fan's shared point so the IC pin, its wire, and the >=2 coincident
    # fan pins resolve as one net by connectivity, letting the redundant IC-pin
    # SW_n label be suppressed without dangling the wire.
    for jpt in getattr(node, "_tjunction_junctions", []):
        jp = jpt * tx
        jx, jy = _round_mm(jp.x), _round_mm(jp.y)
        elements.append(
            Sexp(
                [
                    "junction",
                    ["at", jx, jy],
                    ["diameter", 0],
                    ["color", 0, 0, 0, 0],
                    ["uuid", _gen_uuid(f"tjjunction:{jx}:{jy}")],
                ]
            )
        )
    # Suppress labels for staggered T-junction signal pins (connected by wire).
    wired_pin_ids.update(getattr(node, "_tjunction_suppressed_pins", set()))

    # Emit wires for decoupling caps offset-snapped to IC power pins, and
    # suppress their pin labels (the wire makes the redundant label noise).
    for pcw in getattr(node, "_power_cap_wires", []):
        x1_mil, y1_mil, x2_mil, y2_mil = pcw
        p1 = Point(x1_mil, y1_mil) * tx
        p2 = Point(x2_mil, y2_mil) * tx
        x1, y1 = _round_mm(p1.x), _round_mm(p1.y)
        x2, y2 = _round_mm(p2.x), _round_mm(p2.y)
        elements.append(
            Sexp(
                [
                    "wire",
                    ["pts", ["xy", x1, y1], ["xy", x2, y2]],
                    ["stroke", ["width", 0], ["type", "default"]],
                    ["uuid", _gen_uuid(f"pcwire:{x1}:{y1}:{x2}:{y2}")],
                ]
            )
        )
    # Junction dots where the power-cap trunk taps each cap +ve pin / the IC
    # connector, so the right-angle bus resolves as one connected net.
    for jpt in getattr(node, "_power_cap_junctions", []):
        jp = jpt * tx
        jx, jy = _round_mm(jp.x), _round_mm(jp.y)
        elements.append(
            Sexp(
                [
                    "junction",
                    ["at", jx, jy],
                    ["diameter", 0],
                    ["color", 0, 0, 0, 0],
                    ["uuid", _gen_uuid(f"pcjunction:{jx}:{jy}")],
                ]
            )
        )
    wired_pin_ids.update(getattr(node, "_power_cap_suppressed_pins", set()))

    # Generate net labels for stubbed pins (skip pins that got direct wires).
    # In deconflict mode the closure labeller below replaces this per-pin path
    # for SIGNAL nets (one label per island at a deconflicted stub end, not one
    # per pin on the pin); only POWER pins are still handled here so they emit
    # their power symbol at the pin (power nets are excluded from stubbing).
    for part in node.parts:
        if isinstance(part, NetTerminal):
            continue
        for pin in part:
            if id(pin) in wired_pin_ids:
                continue
            net = getattr(pin, "net", None)
            is_power = pin.is_connected() and _net_wants_power_symbol(net)
            if deconflict and not is_power:
                continue  # closure labeller handles non-power pins in this mode
            label = net_label_to_sexp(
                pin, tx=tx, kind=_label_kind(net), uuid_path=uuid_path
            )
            if label:
                elements.append(label)
            elif (
                len(part.pins) == 2
                and not pin.stub
                and pin.is_connected()
                and _net_wants_power_symbol(pin.net)
            ):
                label = net_label_to_sexp(pin, tx=tx, force=True, uuid_path=uuid_path)
                if label:
                    elements.append(label)

    if deconflict:
        # Stage-25 closure labels: ONE label per connected component of each
        # non-power net's (wires ∪ pins), anchored at a deconflicted stub end.
        # A fully wired net -> 1 label; a split net -> one label per island,
        # closing it by shared name WITHOUT a false merge (ends are net-unique,
        # on-grid cells). Stubbed (label-only) nets also get their pin->end stub
        # wires drawn here. Half-grid coincidence tolerance (ends are on the
        # 50-mil grid; nothing lands closer than one grid across nets).
        _CLOSE_TOL = 25.0
        net_pins = OrderedDict()
        for part in node.parts:
            for pin in part:
                if not pin.is_connected():
                    continue
                net = pin.net
                if isinstance(net, NCNet) or getattr(net, "_is_power_net", False):
                    continue
                if id(pin) not in stub_ends:
                    continue
                net_pins.setdefault(id(net), (net, []))[1].append(pin)
        for _nid, (net, pins) in net_pins.items():
            nm = getattr(net, "name", None)
            if not nm or _net_wants_power_symbol(net):
                continue
            is_stub_net = getattr(net, "_stub", False) or id(net) in onpin_net_ids
            if is_stub_net:
                # Draw the pin->end stub wires (the wire block skipped this net)
                # and use them as the island geometry. NetTerminal pins have no
                # emitted symbol, so drawing their pin->end stub would leave the
                # bare terminal pin dangling; skip it -- the net is closed by
                # name via the real pins' closure labels (or, if the net has no
                # real on-sheet pin, the terminal's own label above).
                segs = []
                for pin in pins:
                    if id(pin) in terminal_pin_ids:
                        continue
                    end = stub_ends[id(pin)]
                    pin_w = (pin.pt * pin.part.tx).round()
                    if (pin_w.x, pin_w.y) != (end.x, end.y):
                        elements.extend(
                            wire_to_sexp(
                                net,
                                [Segment(Point(pin_w.x, pin_w.y), Point(end.x, end.y))],
                                tx=tx,
                            )
                        )
                    segs.append((pin_w.x, pin_w.y, end.x, end.y))
            else:
                segs = [
                    (s.p1.x, s.p1.y, s.p2.x, s.p2.y) for s in node.wires.get(net, [])
                ]
            pin_by_id = {id(pin): pin for pin in pins}
            pin_pts = [
                (id(pin), stub_ends[id(pin)].x, stub_ends[id(pin)].y) for pin in pins
            ]
            islands = _decisions.net_islands(pin_pts, segs, tol=_CLOSE_TOL)
            net_kind = _label_kind(net)
            # Endpoint degree over this net's segments: a stub end that is a
            # degree-1 leaf and carries no label reads as an
            # ``unconnected_wire_endpoint`` in KiCad. trim_stubs already spared
            # only pin/stub-end leaves, so every surviving leaf is a real stub
            # end -- label the ones the single island anchor doesn't cover.
            deg = Counter()
            for x1, y1, x2, y2 in segs:
                deg[(round(x1), round(y1))] += 1
                deg[(round(x2), round(y2))] += 1
            for island in islands:
                # Prefer a real (non-terminal) pin as the anchor; skip an island
                # that is only a NetTerminal (it already carries the label).
                real = [pid for pid in island if pid not in terminal_pin_ids]
                if not real:
                    continue
                anchor_pid = min(
                    real,
                    key=lambda pid: (
                        str(getattr(pin_by_id[pid].part, "ref", "") or ""),
                        str(pin_by_id[pid].num),
                        pid,
                    ),
                )
                label = net_label_to_sexp(
                    pin_by_id[anchor_pid],
                    tx=tx,
                    force=True,
                    kind=net_kind,
                    at_world=stub_ends[anchor_pid],
                    uuid_path=uuid_path,
                )
                if label:
                    elements.append(label)
                # Label every OTHER leaf stub end in this island so no wire end
                # dangles. Deterministic (ref, pin.num, id) order; skip the
                # anchor, terminals (already labelled), and fallback no-stub
                # pins (end on the pin -> the pin itself terminates the wire).
                for pid in sorted(
                    real,
                    key=lambda p: (
                        str(getattr(pin_by_id[p].part, "ref", "") or ""),
                        str(pin_by_id[p].num),
                        p,
                    ),
                ):
                    if pid == anchor_pid:
                        continue
                    end = stub_ends[pid]
                    pin = pin_by_id[pid]
                    pin_w = (pin.pt * pin.part.tx).round()
                    if (round(pin_w.x), round(pin_w.y)) == (round(end.x), round(end.y)):
                        continue  # no stub -> the pin ends the wire, not a bare end
                    if deg[(round(end.x), round(end.y))] != 1:
                        continue  # interior of the routing tree, already connected
                    leaf_label = net_label_to_sexp(
                        pin,
                        tx=tx,
                        force=True,
                        kind=net_kind,
                        at_world=end,
                        uuid_path=uuid_path,
                    )
                    if leaf_label:
                        elements.append(leaf_label)
    else:
        # Backstop label on every ROUTED sheet-internal net (stage 24): a wired
        # net otherwise carries no on-sheet name. Emit exactly ONE local
        # ``label`` at a deterministic pin so (a) the net name is visible, (b) if
        # any wire segment is imperfect the shared local name still closes the
        # net, and (c) the ERC gate has an anchor. Power nets (power symbols) and
        # cross-sheet nets (global/hierarchical labels) are excluded; stubbed
        # nets already self-label above; on-pin-relocated nets keep their single
        # label. Deconfliction (apply_label_deconfliction) resolves collisions.
        _labeled_backstop = set()
        for net, wire in node.wires.items():
            if not wire or getattr(net, "_stub", False) or id(net) in onpin_net_ids:
                continue
            if not _is_internal(net):
                continue
            nm = getattr(net, "name", None)
            if not nm or _net_wants_power_symbol(net) or id(net) in _labeled_backstop:
                continue
            # Deterministic anchor pin: internal pin with the smallest world coord.
            cand = [
                p for p in node.get_internal_pins(net) if id(p) not in wired_pin_ids
            ] or list(node.get_internal_pins(net))
            if not cand:
                continue

            def _pin_world(p):
                pp = getattr(p, "pt", Point(p.x, p.y))
                w = pp * getattr(p.part, "tx", Tx()) * tx
                return (_round_mm(w.x), _round_mm(w.y))

            anchor = min(cand, key=_pin_world)
            label = net_label_to_sexp(
                anchor, tx=tx, force=True, local=True, uuid_path=uuid_path
            )
            if label:
                elements.append(label)
                _labeled_backstop.add(id(net))

    # No-connect flags for NCNet pins.
    for nc_x, nc_y, part_ref, pin_num in _decisions.find_no_connect_pins(
        node, _backend, tx
    ):
        elements.append(
            _backend.emit_no_connect(
                nc_x,
                nc_y,
                uuid_seed=f"nc:{part_ref}:{pin_num}:{nc_x}:{nc_y}",
            )
        )

    # One PWR_FLAG per undriven rail present on this sheet (coincident with a
    # power-symbol pin). Done before the dangling-wire purge so the flag's point
    # counts as an anchor. Module-global bookkeeping keeps it to ONE flag/rail
    # across the whole project (power symbols connect globally by name).
    _append_pwr_flags(elements, uuid_path)

    # Drain the power-stub wires (pin -> offset power symbol) collected while
    # emitting power symbols on this sheet (power_stubs option). Done before the
    # purge so it sees these wires + their power-symbol anchors.
    if _power_stub_wires:
        elements.extend(_power_stub_wires)
        _power_stub_wires.clear()

    # Purge dangling wire remnants. Snap moves parts by reassigning part.tx,
    # but a router wire to the old position can survive as a short stub whose
    # far end touches nothing (KiCad flags these as wire_dangling). Drop any
    # wire segment with an endpoint anchored to nothing — not a pin, label,
    # junction, no-connect, or another wire endpoint. Iterate, since removing
    # one stub can expose the next.
    from collections import Counter as _Counter

    _anchor_pts = set()
    for _part in node.parts:
        for _pin in _part:
            _pp = getattr(_pin, "pt", Point(_pin.x, _pin.y))
            _ptx = getattr(_pin.part, "tx", Tx())
            _w = _pp * _ptx * tx
            _anchor_pts.add((_round_mm(_w.x), _round_mm(_w.y)))
    for _el in elements:
        if not (isinstance(_el, (list, Sexp)) and len(_el)):
            continue
        if _el[0] in (
            "global_label",
            "label",
            "hierarchical_label",
            "junction",
            "no_connect",
        ):
            for _s in _el:
                if isinstance(_s, (list, Sexp)) and len(_s) >= 3 and _s[0] == "at":
                    _anchor_pts.add((_s[1], _s[2]))
                    break
        elif _el[0] == "sheet":
            # A hierarchical sheet's PINs are connection anchors on the parent
            # sheet: the Tier-2 sheet-pin stub wires run from a pin out to a name
            # label, so the pin end must count as anchored or the purge below
            # (which does not know sheet pins) would drop the whole stub.
            for _sub in _el:
                if isinstance(_sub, (list, Sexp)) and len(_sub) and _sub[0] == "pin":
                    for _s in _sub:
                        if (
                            isinstance(_s, (list, Sexp))
                            and len(_s) >= 3
                            and _s[0] == "at"
                        ):
                            _anchor_pts.add((_s[1], _s[2]))
                            break
        elif _el[0] == "symbol":
            # A power symbol's pin (= its `at`) anchors the power_stubs stub wire
            # whose far end sits on the offset symbol; without this the purge
            # would drop that stub (its symbol end looks unanchored).
            _lib = next(
                (s for s in _el if isinstance(s, (list, Sexp)) and len(s) >= 2 and s[0] == "lib_id"),
                None,
            )
            if _lib and _is_power_libid(str(_lib[1])):
                _at = next(
                    (s for s in _el if isinstance(s, (list, Sexp)) and len(s) >= 3 and s[0] == "at"),
                    None,
                )
                if _at:
                    _anchor_pts.add((_at[1], _at[2]))

    def _wire_endpoints_of(_el):
        for _s in _el:
            if isinstance(_s, (list, Sexp)) and len(_s) and _s[0] == "pts":
                return [
                    (_xy[1], _xy[2])
                    for _xy in _s[1:]
                    if isinstance(_xy, (list, Sexp))
                    and len(_xy) >= 3
                    and _xy[0] == "xy"
                ]
        return []

    for _ in range(8):  # iterate; cap as a safety backstop
        _counts = _Counter()
        _wires = []
        for _i, _el in enumerate(elements):
            if isinstance(_el, (list, Sexp)) and len(_el) and _el[0] == "wire":
                _pts = _wire_endpoints_of(_el)
                _wires.append((_i, _pts))
                for _p in _pts:
                    _counts[_p] += 1
        _drop = set()
        for _i, _pts in _wires:
            if any(_p not in _anchor_pts and _counts.get(_p, 0) < 2 for _p in _pts):
                _drop.add(_i)
        if not _drop:
            break
        elements = [_el for _i, _el in enumerate(elements) if _i not in _drop]

    # Belt-and-braces emission audit (wired-render plan, Phase 3): every
    # connected pin MUST be covered by a wire (endpoint or pass-through), a
    # label, a power symbol, or a no-connect. The per-net A* fallback and the
    # single backstop label do not guarantee this -- a net can route MOST of its
    # pins yet leave one uncovered (that pin reads as pin_not_connected, and the
    # drawing diverges from the netlist). Force a name label on any uncovered pin
    # so it reconnects to its net by name; this is the renderer-side mirror of
    # the harness drawing_connectivity gate, catching the hole before the file
    # ships. Power/NC pins are covered by their symbol; NetTerminals self-label.
    _forced = _audit_and_force_pin_labels(node, elements, tx, uuid_path, _label_kind)

    if node.flattened:
        # This node is flattened, so return elements for inclusion in the parent sheet.
        return elements, lib_symbols, pwr_symbols, ""

    # --- Unflattened node: write a separate .kicad_sch file for this sheet. ---

    schematic = Sexp(
        [
            "kicad_sch",
            ["version", version],
            ["generator", "skidl"],
            ["generator_version", __version__],
            ["uuid", node.uuid],
            ["paper", paper],
        ]
    )

    # Add title block to schematic sheet.
    schematic.append(Sexp(create_title_block_sexp(node.title)))

    # Build lib_symbols section for this sheet.
    lib_symbols_sexp = Sexp(["lib_symbols"])
    for part in lib_symbols.values():
        lib_symbols_sexp.append(Sexp(part_to_lib_symbol_definition(part)))

    # Add power-symbol definitions. Derive these from the power-symbol INSTANCES
    # actually emitted on this sheet (`elements`), NOT from `node.wires`: with
    # auto_stub, power nets are stubbed (labels, not wires), so a wire-based scan
    # misses them and the emitted instance has no definition -> KiCad reports an
    # "unknown component". Scanning emitted instances guarantees every instance
    # has a matching definition, and also covers power symbols contributed by
    # flattened children (whose instances are already in `elements`).
    for pwr_lib_id in sorted(_power_lib_ids_in_elements(elements)):
        if pwr_lib_id not in lib_symbols:
            pwr_sexp = _extract_power_lib_symbol(pwr_lib_id.split(":", 1)[1])
            if pwr_sexp:
                lib_symbols_sexp.append(pwr_sexp)

    # Add lib_symbols section to schematic.
    schematic.append(lib_symbols_sexp)

    # Child-sheet half of the KiCad hierarchical interconnect: with the
    # ``hierarchical_sheet_pins`` option ON, each boundary net's on-sheet label
    # is emitted as a ``hierarchical_label`` (kind="hier") AT the net's routed
    # position (NetTerminal pin / on-pin relocation point) by the label-emission
    # loops above -- see _label_kind() -- so it lands ON the net (no
    # label_dangling) and pairs by name with the sheet pin on the parent's sheet
    # symbol (create_hierarchical_sheet_sexp). The old fixed sheet-edge-slot loop
    # that emitted a redundant, disconnected hierarchical_label here (the
    # label_dangling source) has been removed. Default OFF: boundary nets connect
    # by ``global_label`` name exactly as before.

    # Spread net labels off component bodies (connectivity-preserving).
    # Decision (overlap + nudge target) lives in schematics/decisions.py; the
    # backend reads/mutates the label Sexps and appends connecting wires.
    _backend.apply_label_deconfliction(elements, node, tx)

    # Diagnostic: warn if any coordinate ended up shared by two nets (the
    # fusion mechanism Blocker B fixed). Enforcement is the downstream
    # equivalence gate + native fallback; this just localizes a regression.
    _audit_sheet_connectivity(node, elements, _backend, tx)

    # Add all the collected elements of the schematic.
    for elem in elements:
        schematic.append(elem)

    # Write schematic file.
    filepath = os.path.join(node.filepath, node.sheet_filename)
    _write_sexp_schematic(schematic, filepath)

    # Return a hierarchical sheet reference for this node to be included in the parent sheet.
    sheet_uuid = uuid_path.split("/")[
        -1
    ]  # Use the last UUID in the path for the sheet UUID.
    # This box is drawn on the PARENT's sheet. The parent is the root iff this
    # node is a direct child of root: node uuid_path "/root/thisnode" -> 2 "/".
    # (The parent-side stub label kind hinges on this -- see
    # create_hierarchical_sheet_sexp: root parent -> local label, intermediate
    # parent -> hierarchical_label that also exports the transit net upward.)
    _parent_is_root = uuid_path.count("/") <= 2
    _sheet_sexp, _sheet_extras = create_hierarchical_sheet_sexp(
        node, sheet_uuid, sheet_tx, parent_is_root=_parent_is_root
    )
    return (
        [_sheet_sexp] + _sheet_extras,
        {},
        {},
        filepath,
    )


# ---------------------------------------------------------------------------
# Top-level schematic assembly + write
# ---------------------------------------------------------------------------


@export_to_all
def write_top_schematic(
    circuit,
    node,
    filepath,
    top_name,
    title,
    version=20230409,
    hierarchical_sheet_pins=False,
    power_stubs=False,
):
    """Generate and write the complete schematic from a placed+routed node tree.

    This is the main entry point called by each tool's gen_schematic().

    Args:
        circuit: The Circuit object.
        node: Root SchNode (placed and routed).
        filepath: Output directory.
        top_name: Base filename (without extension).
        title: Schematic title.
        version: S-expression version number.
        hierarchical_sheet_pins: When True, emit KiCad hierarchical sheet pins +
            hierarchical labels for boundary nets (the in-progress
            hierarchical-interconnect surface). Default False -- boundary nets
            connect by ``global_label`` name, which is ERC-clean; the sheet-pin
            path is not yet fully wired (see _EMIT_HIER_SHEET_PINS).
    """

    global _EMIT_HIER_SHEET_PINS, _EMIT_POWER_STUBS
    _EMIT_HIER_SHEET_PINS = bool(hierarchical_sheet_pins)
    _EMIT_POWER_STUBS = bool(power_stubs)
    _power_stub_wires.clear()

    init_power_symbol_data()

    # Undriven power rails (marked by mark_power_nets) each get exactly one
    # project-wide PWR_FLAG. Collect their names now; the per-sheet emitter drops
    # the flag on the first sheet carrying that rail's power symbol.
    for net in getattr(circuit, "nets", []):
        if getattr(net, "_is_power_net", False) and getattr(
            net, "_needs_pwr_flag", False
        ):
            nm = getattr(net, "name", None)
            if nm:
                _pwr_flag_net_names.add(nm)

    node.title = title
    node.sheet_filename = top_name or "schematic"
    node.filepath = filepath

    # Top node is never flattened because it has no parent to accept its contents, so it must always generate a sheet.
    node.flattened = False

    # Generate a deterministic UUID for the top node based on its name. A top
    # node built from a circuit whose parts carry no hierarchy level never had
    # .name assigned by add_part; fall back to top_name.
    node.uuid = _gen_uuid(getattr(node, "name", None) or top_name)

    # UUID paths start from this root node. Used for hierarchical sheet references.
    uuid_path = f"/{node.uuid}"

    # Write root schematic. Ignore returned items except name of top-level sheet file.
    _, _, _, output_file = node_to_sexp_schematic(
        node, uuid_path=uuid_path, version=version
    )

    # Non-stock rail clones (SKiDL_rails:*) need a project-local library +
    # sym-lib-table so KiCad ERC can resolve them (else lib_symbol_issues against
    # the stock ``power`` lib). Written after all sheets, once the clone set is known.
    _write_custom_rail_library(filepath)

    # Optional: validate with kicad-cli if available.
    _validate_with_kicad_cli(output_file)

    return output_file


def _write_custom_rail_library(directory):
    """Emit ``SKiDL_rails.kicad_sym`` + a ``sym-lib-table`` entry for the non-stock
    rail symbols cloned in-file this generation.

    Stock power symbols resolve against KiCad's global ``power`` library; the
    in-file clones for non-stock rails (V5P, VBIAS_28V ...) carry a
    ``SKiDL_rails:`` nickname that only resolves via a project-local table. Without
    this, ERC reports ``lib_symbol_issues`` for every custom-rail instance even
    though the symbol renders fine from the embedded ``lib_symbols`` definition.
    """
    if not _custom_power_symbols:
        return

    from copy import deepcopy

    lib = Sexp(
        [
            "kicad_symbol_lib",
            ["version", 20241209],
            ["generator", "skidl"],
            ["generator_version", __version__],
        ]
    )
    for lib_id, definition in sorted(_custom_power_symbols.items()):
        sym = deepcopy(definition)
        # In a standalone .kicad_sym the symbol name is the bare rail name; the
        # nickname (SKiDL_rails) comes from the sym-lib-table, not the symbol.
        sym[1] = _power_net_from_libid(lib_id)
        lib.append(sym)

    def need_quote(x):
        tag = x[0]
        if tag == "symbol" and len(x) > 1 and isinstance(x[1], str):
            return True
        return tag in (
            "property",
            "name",
            "number",
            "generator",
            "generator_version",
        )

    def need_quote_alternate(x):
        return x[0] == "alternate"

    lib.add_quotes(need_quote)
    lib.add_quotes(need_quote_alternate, stop_idx=2)

    sym_path = os.path.join(directory, f"{_CUSTOM_RAIL_LIB}.kicad_sym")
    with open(sym_path, "w") as f:
        f.write(lib.to_str())

    _ensure_sym_lib_table_entry(directory)


def _ensure_sym_lib_table_entry(directory):
    """Ensure the project ``sym-lib-table`` maps the ``SKiDL_rails`` nickname to the
    local ``SKiDL_rails.kicad_sym`` (KIPRJMOD-relative). Creates the table if absent;
    otherwise inserts the entry only when missing (preserving other libraries)."""
    table_path = os.path.join(directory, "sym-lib-table")
    entry = (
        f'  (lib (name "{_CUSTOM_RAIL_LIB}")(type "KiCad")'
        f'(uri "${{KIPRJMOD}}/{_CUSTOM_RAIL_LIB}.kicad_sym")(options "")(descr ""))'
    )
    if not os.path.exists(table_path):
        with open(table_path, "w") as f:
            f.write(f"(sym_lib_table\n  (version 7)\n{entry}\n)\n")
        return

    with open(table_path, "r") as f:
        text = f.read()
    if f'(name "{_CUSTOM_RAIL_LIB}")' in text:
        return  # already present -- leave the user's table untouched.
    idx = text.rfind(")")
    if idx == -1:
        return
    with open(table_path, "w") as f:
        f.write(text[:idx] + entry + "\n" + text[idx:])


# ---------------------------------------------------------------------------
# Optional KiCad CLI validation
# ---------------------------------------------------------------------------


def _validate_with_kicad_cli(filepath):
    """Run kicad-cli ERC on generated schematic if available."""
    import shutil
    import subprocess

    kicad_cli = shutil.which("kicad-cli")
    if not kicad_cli:
        return  # Silent skip if not installed.
    try:
        result = subprocess.run(
            [kicad_cli, "sch", "erc", "--exit-code-violations", filepath],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            from skidl.logger import active_logger

            active_logger.warning(
                f"KiCad ERC found issues in {filepath}:\n{result.stderr}"
            )
    except (subprocess.TimeoutExpired, OSError):
        pass  # Don't fail generation if CLI has issues.


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------


def _write_sexp_schematic(schematic, filepath):
    """Write an Sexp schematic object to a file with proper quoting.

    Args:
        schematic: Sexp object.
        filepath: Output file path.
    """

    def need_quote(x):
        tag = x[0]
        if tag == "symbol" and len(x) > 1 and isinstance(x[1], str):
            # Quote lib_symbol names like "Device:R", "power:GND", "R_0_1"
            return True
        return tag in (
            "title",
            "date",
            "company",
            "comment",
            "path",
            "project",
            "property",
            "name",
            "number",
            "lib_id",
            "reference",
            "label",
            "global_label",
            "hierarchical_label",
            "generator",
            "generator_version",
            "paper",
        )

    def need_quote_alternate(x):
        return x[0] == "alternate"

    schematic.add_quotes(need_quote)
    schematic.add_quotes(need_quote_alternate, stop_idx=2)

    with open(filepath, "w") as f:
        f.write(schematic.to_str())
