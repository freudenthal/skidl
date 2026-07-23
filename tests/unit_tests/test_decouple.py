# -*- coding: utf-8 -*-
"""Tests for the explicit decoupling-cap declaration (skidl.decouple)."""

import pytest

from skidl import Part, TEMPLATE
from skidl.decouple import (
    normalize_decouples,
    decouples_target,
    decouples_field_value,
)


# --- lightweight duck-typed fakes matching the Pin / Part discriminators -----

class _FakePart:
    def __init__(self, ref):
        self.ref = ref
        self.pins = []          # Part has a .pins collection


class _FakePin:
    def __init__(self, part, num):
        self.part = part        # Pin has a parent .part and no .pins
        self.num = num


# --- normalize_decouples matrix ---------------------------------------------

def test_normalize_ref_string():
    assert normalize_decouples("U1") == ("U1", None)


def test_normalize_ref_pin_string():
    assert normalize_decouples("U1.3") == ("U1", "3")
    assert normalize_decouples(" U2 . VI ") == ("U2", "VI")


def test_normalize_tuple():
    assert normalize_decouples(("U1", "3")) == ("U1", "3")
    assert normalize_decouples(("U1", None)) == ("U1", None)
    assert normalize_decouples(["U1", 3]) == ("U1", "3")


def test_normalize_part():
    assert normalize_decouples(_FakePart("U7")) == ("U7", None)


def test_normalize_pin():
    part = _FakePart("U7")
    assert normalize_decouples(_FakePin(part, "3")) == ("U7", "3")


def test_normalize_none():
    assert normalize_decouples(None) is None


@pytest.mark.parametrize("bad", [42, 3.14, object(), ("U1",), ("U1", "3", "x")])
def test_normalize_garbage_raises(bad):
    with pytest.raises(ValueError):
        normalize_decouples(bad)


def test_normalize_empty_string_raises():
    with pytest.raises(ValueError):
        normalize_decouples("   ")


# --- decouples_target: late-ref binding + absence ---------------------------

def test_target_absent_attr_is_none():
    assert decouples_target(_FakePart("C1")) is None  # no .decouples


def test_target_late_ref_binding():
    parent = _FakePart(None)             # ref not yet assigned
    cap = _FakePart("C1")
    cap.decouples = parent
    assert decouples_target(cap) is None  # unresolved -> deferred
    parent.ref = "U3"                     # ref assigned late
    assert decouples_target(cap) == ("U3", None)


def test_field_value_formats():
    cap = _FakePart("C1")
    cap.decouples = "U1.3"
    assert decouples_field_value(cap) == "U1.3"
    cap.decouples = "U1"
    assert decouples_field_value(cap) == "U1"
    cap2 = _FakePart("C2")
    assert decouples_field_value(cap2) is None


# --- real skidl Part / Pin round-trip ---------------------------------------

def test_real_part_and_pin_normalize():
    vreg = Part("Device", "R")          # NETLIST -> auto-assigned ref
    ref = str(vreg.ref)
    # Part target
    assert normalize_decouples(vreg) == (ref, None)
    # Pin target (first pin, by its own number)
    p0 = vreg.pins[0]
    tref, pin = normalize_decouples(p0)
    assert tref == ref
    assert pin == str(p0.num)


def test_real_part_decouples_attr_kwarg():
    # decouples= as a construction kwarg persists as a plain attribute.
    cap = Part("Device", "C", decouples="U1.2")
    assert decouples_target(cap) == ("U1", "2")
    assert decouples_field_value(cap) == "U1.2"
