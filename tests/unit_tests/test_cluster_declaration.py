# -*- coding: utf-8 -*-

"""Stage-24 declared-adjacency (``cluster=``) resolution in snap.

A 2-pin part may declare ``cluster="REF.PIN"`` / ``"REF"`` to state which
multi-pin part pin it belongs on, so snap honors the declaration instead of
guessing by the pin-count heuristic (the Blocker-B root). A bad hint must fall
back to the heuristic (warn, return None), never fail the render.
"""

from types import SimpleNamespace

import pytest

from skidl import Net, Part, KICAD9
from skidl.schematics.snap import _resolve_cluster


def _node(*parts):
    return SimpleNamespace(parts=list(parts), children={})


def _mk():
    """A TL072 (U3) + two caps sharing/not-sharing the VCC net at pin 8."""
    u = Part("Amplifier_Operational", "TL072", tool=KICAD9)
    u.ref = "U3"
    c = Part("Device", "C", tool=KICAD9)
    c.ref = "C2"
    other = Part("Device", "C", tool=KICAD9)
    other.ref = "C9"
    vcc = Net("VCC")
    gnd = Net("GND")
    u[8] += vcc  # TL072 pin 8 = V+
    c[1] += vcc  # C2 shares VCC with U3.8
    c[2] += gnd
    other[1] += gnd  # C9 shares NO net with U3
    other[2] += Net("N1")
    return u, c, other


def test_cluster_resolves_to_declared_pin():
    u, c, other = _mk()
    node = _node(u, c, other)
    res = _resolve_cluster("U3.8", c, node)
    assert res is not None
    my_pin, target_pin, target_part = res
    assert target_part is u
    assert str(target_pin.num) == "8"
    # my_pin is C2's pin on the shared VCC net (pin 1).
    assert my_pin in c.pins and my_pin.net.name == "VCC"


def test_cluster_bare_ref_resolves_when_unambiguous():
    u, c, other = _mk()
    node = _node(u, c, other)
    res = _resolve_cluster("U3", c, node)  # bare ref, one shared net
    assert res is not None
    _my, target_pin, target_part = res
    assert target_part is u and str(target_pin.num) == "8"


def test_cluster_unknown_ref_falls_back():
    u, c, other = _mk()
    node = _node(u, c, other)
    assert _resolve_cluster("U99.8", c, node) is None  # no such ref -> heuristic


def test_cluster_no_shared_net_falls_back():
    u, c, other = _mk()
    node = _node(u, c, other)
    # C9 shares no net with U3 -> declaration cannot resolve.
    assert _resolve_cluster("U3.8", other, node) is None


def test_cluster_unknown_pin_falls_back():
    u, c, other = _mk()
    node = _node(u, c, other)
    assert _resolve_cluster("U3.999", c, node) is None  # no such pin -> heuristic
