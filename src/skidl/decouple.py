# -*- coding: utf-8 -*-
"""Explicit decoupling-cap declaration (Arc A).

Skidl-friendly: no new class, no registry -- a cap declares which IC / pin it
decouples via a ``decouples=`` keyword attribute (or plain assignment), which
falls out of skidl's existing arbitrary-attribute behaviour::

    vreg = Part("Regulator_Linear", "AMS1117-3.3")
    cbyp = Part("Device", "C", value="1u", decouples=vreg["VI"])  # a Pin
    cbyp.decouples = vreg           # a Part  -> parent ref, no pin
    cbyp.decouples = "U1"           # a ref string
    cbyp.decouples = "U1.3"         # ref.pin string (the exported form)
    cbyp.decouples = ("U1", "3")    # (ref, pin) tuple

The declaration is normalised **lazily at read time** (a part may get its ref
assigned late) into a ``(ref, pin | None)`` target. Consumers:

- ``skidl_layout`` reads the live part attribute (``decouples_target``) to place
  the cap against the declared parent's supply pad (PCB side -- the priority).
- the KiCad-10 schematic writer exports it as a ``Decouples=U1.3`` property
  (``decouples_field_value``) so it survives HITL round-trips + skidl-codegen.

Absent the attribute, every consumer is a no-op -> outputs byte-identical.
"""

from __future__ import annotations


def _pin_id(pin) -> str | None:
    """A KiCad pin number/name string for a skidl Pin, or None."""
    num = getattr(pin, "num", None)
    if num is not None and str(num).strip():
        return str(num).strip()
    name = getattr(pin, "name", None)
    if name is not None and str(name).strip():
        return str(name).strip()
    return None


def normalize_decouples(value):
    """Normalise a ``decouples=`` value to ``(ref, pin | None)``.

    Accepts a skidl ``Part`` (-> ``(ref, None)``), ``Pin`` (-> ``(ref, num)``),
    a ``"REF"`` / ``"REF.PIN"`` string, or a ``(ref, pin)`` tuple/list. ``ref``
    is ``None`` when the target part has no reference assigned yet (caller
    defers). Raises ``ValueError`` on any other type.
    """
    if value is None:
        return None

    # (ref, pin) tuple / list
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"decouples= tuple must be (ref, pin); got {len(value)} items"
            )
        ref, pin = value
        ref_s = str(ref).strip() if ref not in (None, "") else None
        pin_s = str(pin).strip() if pin not in (None, "") else None
        return (ref_s, pin_s)

    # "REF" or "REF.PIN" string
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("decouples= string is empty")
        if "." in s:
            ref, pin = s.split(".", 1)
            return (ref.strip() or None, pin.strip() or None)
        return (s, None)

    # skidl Pin: carries a pin number ``.num`` and a parent ``.part`` (a Part has
    # neither -- and a Pin exposes a ``.ref`` property + ``.pins``, so ``.num`` is
    # the reliable discriminator, checked first).
    if hasattr(value, "num") and hasattr(value, "part"):
        parent = getattr(value, "part", None)
        ref = getattr(parent, "ref", None)
        ref_s = str(ref).strip() if ref not in (None, "") else None
        return (ref_s, _pin_id(value))

    # skidl Part / PartUnit: has a .ref and a .pins collection.
    if hasattr(value, "ref") and hasattr(value, "pins"):
        ref = getattr(value, "ref", None)
        ref_s = str(ref).strip() if ref not in (None, "") else None
        return (ref_s, None)

    raise ValueError(
        "decouples= expects a Part, Pin, 'REF', 'REF.PIN', or (ref, pin); "
        f"got {type(value).__name__}"
    )


def decouples_target(part):
    """Resolved ``(ref, pin | None)`` a part decouples, or ``None``.

    ``None`` when the part carries no ``decouples`` attribute, or when the
    declared target has no reference assigned yet (not resolvable). Reads
    lazily, so it is correct to call after refs are finalised.
    """
    value = getattr(part, "decouples", None)
    if value is None:
        return None
    target = normalize_decouples(value)
    if target is None or target[0] is None:
        return None
    return target


def decouples_field_value(part) -> str | None:
    """The ``Decouples`` schematic-property value (``"U1.3"`` / ``"U1"``)."""
    target = decouples_target(part)
    if target is None:
        return None
    ref, pin = target
    return f"{ref}.{pin}" if pin else ref
