# -*- coding: utf-8 -*-

"""Floating-node conditioning: DC-stranded nodes are tied to ground (finding F4).

Excluding a part (``Sim.Enable=0``) can leave a neighbor node with no DC path to
ground -- a lone bypass cap, or a whole R-C compensation island whose only tie to
ground is through a capacitor. ngspice then throws ``singular matrix: check node
<n>``. The converter now detects such nodes on the final device graph and ties
each to ground with a 1 G bleed (electrically invisible), off via
``SKIDL_SIM_TIE_FLOATING=0``.
"""

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter  # noqa: F401 (needs PySpice)

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _conv():
    from skidl.sim import skidl_flat_view

    return SpiceConverter(skidl_flat_view())


# --- _conducting_nodes: DC-path topology per element type (pure) -------------


@requires_sim
def test_conducting_nodes_topology():
    _setup()
    conv = _conv()

    class Mosfet:
        node_names = ["D", "G", "S", "B"]

    class Capacitor:
        node_names = ["x", "0"]

    class CurrentSource:
        node_names = ["a", "0"]

    class Resistor:
        node_names = ["a", "b"]

    # MOSFET gate (index 1) is DC-isolated; drain/source/bulk conduct
    assert conv._conducting_nodes(Mosfet()) == ["D", "S", "B"]
    # capacitor / current source are DC-opens -> no conducting nodes
    assert conv._conducting_nodes(Capacitor()) == []
    assert conv._conducting_nodes(CurrentSource()) == []
    # everything else anchors all its nodes
    assert conv._conducting_nodes(Resistor()) == ["a", "b"]


# --- end-to-end through convert() -------------------------------------------


def _driven_rail():
    """A 5 V source + load resistor so VIN/GND are anchored (returns nets)."""
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
    r = Part("Device", "R", ref="R1", value="1k")
    vin, gnd = Net("VIN"), Net("GND")
    v[1] += vin
    v[2] += gnd
    r[1] += vin
    r[2] += gnd
    return vin, gnd


@requires_sim
def test_cap_only_node_is_tied(monkeypatch):
    monkeypatch.delenv("SKIDL_SIM_TIE_FLOATING", raising=False)
    _setup()
    _vin, gnd = _driven_rail()
    ciso = Part("Device", "C", ref="C1", value="1u")
    iso = Net("ISO")
    ciso[1] += iso
    ciso[2] += gnd
    net = str(_conv().convert(strict=False))
    # ISO (bypass-cap only) is tied to ground with a 1 G bleed
    assert "float_tie_ISO ISO 0 1000000000.0" in net, net


@requires_sim
def test_resistor_capacitor_island_is_tied(monkeypatch):
    """VC -R- MID -C- GND: the whole island floats at DC; both nodes tied.

    The local 'cap-only' heuristic misses this (each node touches a resistor);
    only a global DC-reachability check catches it -- the LT3757 comp network."""
    monkeypatch.delenv("SKIDL_SIM_TIE_FLOATING", raising=False)
    _setup()
    _vin, gnd = _driven_rail()
    r = Part("Device", "R", ref="R2", value="10k")
    c = Part("Device", "C", ref="C2", value="1n")
    vc, mid = Net("VC"), Net("MID")
    r[1] += vc
    r[2] += mid
    c[1] += mid
    c[2] += gnd
    net = str(_conv().convert(strict=False))
    assert "float_tie_VC VC 0 1000000000.0" in net, net
    assert "float_tie_MID MID 0 1000000000.0" in net, net


@requires_sim
def test_fully_connected_circuit_is_byte_identical(monkeypatch):
    """No floating node -> no ties, and identical to the kill-switch emission."""
    _setup()
    _vin, gnd = _driven_rail()
    # a second resistor to ground: every node has a DC path
    r2 = Part("Device", "R", ref="R2", value="2k")
    out = Net("OUT")
    r2[1] += out
    r2[2] += gnd
    # OUT also to VIN so it isn't dangling
    r3 = Part("Device", "R", ref="R3", value="3k")
    r3[1] += Net("VIN")  # same-named net merges
    r3[2] += out

    monkeypatch.setenv("SKIDL_SIM_TIE_FLOATING", "1")
    on = str(_conv().convert(strict=False))
    monkeypatch.setenv("SKIDL_SIM_TIE_FLOATING", "0")
    off = str(_conv().convert(strict=False))
    assert "float_tie" not in on
    assert on == off


@requires_sim
def test_kill_switch_suppresses_ties(monkeypatch):
    monkeypatch.setenv("SKIDL_SIM_TIE_FLOATING", "0")
    _setup()
    _vin, gnd = _driven_rail()
    ciso = Part("Device", "C", ref="C1", value="1u")
    iso = Net("ISO")
    ciso[1] += iso
    ciso[2] += gnd
    net = str(_conv().convert(strict=False))
    assert "float_tie" not in net, net
