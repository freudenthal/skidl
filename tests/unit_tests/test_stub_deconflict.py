# -*- coding: utf-8 -*-

"""Stage-25 phase 1: deconflicted-stub geometry (snap retired).

Verifies the ``deconflict_stubs`` mode of ``generate_schematic``:
  * every non-power pin gets a stub end >= 1 grid unit from the pin;
  * no two different nets ever share a stub-end grid cell (kills the
    Blocker-B-class false merge);
  * stub ends (and hence routed corners) land on the 50-mil grid;
  * the mode is deterministic and refuses to combine with snap_before_route.

Geometry invariants are checked at the NODE level (before emission) so the
assertions read the real placement/routing state directly; smoke +
determinism are checked on the emitted .kicad_sch. Needs real KiCad symbol
libraries, so the module skips unless they are available.
"""

import os
import re
import shutil
import tempfile

import pytest

_HAS_LIBS = (
    bool(os.environ.get("KICAD9_SYMBOL_DIR"))
    or os.path.exists("/usr/share/kicad/symbols")
    or os.path.exists(os.path.expanduser("~/.local/share/kicad/9.0/symbols"))
)
requires_libs = pytest.mark.skipif(
    not _HAS_LIBS, reason="KiCad symbol libraries not available"
)
pytestmark = requires_libs

GRID = 50.0  # mils (kicad9 constants.GRID)


# --------------------------------------------------------------------------
# Circuit builders
# --------------------------------------------------------------------------
def _build_divider(circuit):
    from skidl import Net, Part

    with circuit:
        r1 = Part("Device", "R", value="10K")
        r2 = Part("Device", "R", value="10K")
        vin, vout, gnd = Net("VIN"), Net("VOUT"), Net("GND")
        vin += r1[1]
        r1[2] += vout
        vout += r2[1]
        r2[2] += gnd


def _build_tia(circuit):
    from skidl import Net, Part

    with circuit:
        u1 = Part("Amplifier_Operational", "OPA340NA")
        rf = Part("Device", "R", value="1M")
        cf = Part("Device", "C", value="2p")
        rin = Part("Device", "R", value="50")
        rl = Part("Device", "R", value="1k")
        gnd, vplus, sig = Net("GND"), Net("+5V"), Net("SIG")
        u1["4"] += rf[1], cf[1], rin[2]
        u1["1"] += rf[2], cf[2], rl[1]
        u1["3"] += gnd
        u1["2"] += gnd
        u1["5"] += vplus
        rin[1] += sig
        rl[2] += gnd


# --------------------------------------------------------------------------
# Pipeline harness: run place + deconflicted-stub routing, return the node.
# --------------------------------------------------------------------------
def _routed_node(circuit, **opts):
    from skidl import get_default_tool
    from skidl.tools import tool_modules
    from skidl.tools.kicad9.gen_schematic import (
        auto_stub_nets,
        preprocess_circuit,
        _place_and_classify,
    )
    from skidl.schematics.sch_node import SchNode

    tool_module = tool_modules[get_default_tool()]
    options = dict(
        auto_stub=True,
        deconflict_stubs=True,
        use_push_pull=True,
        rotate_parts=True,
        pt_to_pt_mult=5,
        pin_normalize=True,
        **opts,
    )
    auto_stub_nets(circuit, **options)
    preprocess_circuit(circuit, **options)
    node = SchNode(circuit, tool_module, ".", "top", "t", 0.0)
    _place_and_classify(node, circuit, 1.0, **options)
    node.route(**options)
    return node


def _iter_nodes(node):
    yield node
    for child in node.children.values():
        yield from _iter_nodes(child)


def _cell(x, y):
    return (round(x / GRID) * GRID, round(y / GRID) * GRID)


def _on_grid(v):
    return abs(v - round(v / GRID) * GRID) < 1e-6


# --------------------------------------------------------------------------
# Node-level geometry invariants
# --------------------------------------------------------------------------
@requires_libs
def test_every_signal_pin_gets_a_stub_end():
    from skidl import Circuit
    from skidl.net import NCNet

    c = Circuit(name="div_stub")
    _build_divider(c)
    node = _routed_node(c)

    checked = 0
    for n in _iter_nodes(node):
        stub_ends = getattr(n, "_stub_ends", {})
        for part in n.parts:
            for pin in part:
                if not pin.is_connected():
                    continue
                net = pin.net
                if isinstance(net, NCNet) or getattr(net, "_is_power_net", False):
                    continue
                assert id(pin) in stub_ends, (
                    f"pin {getattr(part, 'ref', '?')}.{pin.num} has no stub end"
                )
                end = stub_ends[id(pin)]
                pin_w = (pin.pt * part.tx).round()
                dist = abs(end.x - pin_w.x) + abs(end.y - pin_w.y)
                assert dist >= GRID - 1e-6, (
                    f"stub for {getattr(part, 'ref', '?')}.{pin.num} shorter "
                    f"than one grid ({dist})"
                )
                checked += 1
    assert checked >= 4  # VIN, VOUT(x2), GND on the divider


@requires_libs
def test_no_two_nets_share_a_stub_cell():
    """The core anti-false-merge invariant (Experiment B)."""
    from skidl import Circuit
    from skidl.net import NCNet

    c = Circuit(name="tia_stub")
    _build_tia(c)
    node = _routed_node(c)

    for n in _iter_nodes(node):
        stub_ends = getattr(n, "_stub_ends", {})
        cell_owner = {}
        pin_by_id = {}
        for part in n.parts:
            for pin in part:
                pin_by_id[id(pin)] = (part, pin)
        for pid, end in stub_ends.items():
            part, pin = pin_by_id[pid]
            name = getattr(pin.net, "name", None)
            c_key = _cell(end.x, end.y)
            if c_key in cell_owner and cell_owner[c_key] != name:
                pytest.fail(
                    f"stub cell {c_key} shared by nets {cell_owner[c_key]} "
                    f"and {name} (false-merge)"
                )
            cell_owner[c_key] = name


@requires_libs
def test_stub_ends_and_wire_endpoints_on_grid():
    from skidl import Circuit

    c = Circuit(name="tia_grid")
    _build_tia(c)
    node = _routed_node(c)

    for n in _iter_nodes(node):
        for end in getattr(n, "_stub_ends", {}).values():
            assert _on_grid(end.x) and _on_grid(end.y), f"stub end off-grid: {end}"
        for net, segs in n.wires.items():
            if getattr(net, "_stub", False):
                continue
            for seg in segs:
                for p in (seg.p1, seg.p2):
                    assert _on_grid(p.x) and _on_grid(p.y), (
                        f"routed wire endpoint off-grid: {p}"
                    )


# --------------------------------------------------------------------------
# Mode guard + emitted-file smoke/determinism
# --------------------------------------------------------------------------
@requires_libs
def test_deconflict_and_snap_before_route_mutually_exclusive():
    from skidl import Circuit

    c = Circuit(name="conflict")
    _build_divider(c)
    d = tempfile.mkdtemp(prefix="skidl_stub_")
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            c.generate_schematic(
                filepath=d, top_name="conflict", auto_stub=True,
                deconflict_stubs=True, snap_before_route=True,
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _gen_file(circuit, out, top, **opts):
    circuit.generate_schematic(
        filepath=out, top_name=top, auto_stub=True,
        auto_stub_fallback="labels", deconflict_stubs=True, **opts,
    )
    path = os.path.join(out, f"{top}.kicad_sch")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@requires_libs
def test_deconflict_divider_produces_wires():
    from skidl import Circuit

    d = tempfile.mkdtemp(prefix="skidl_stub_")
    try:
        c = Circuit(name="div_wires")
        _build_divider(c)
        text = _gen_file(c, d, "div_wires", seed=1)
        assert len(re.findall(r"\(wire\b", text)) >= 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


@requires_libs
def test_wired_nets_survive_the_full_path():
    """Regression: the ERC correction loop must NOT stub the routed nets in
    deconflict mode (it would discard the deconflicted wires). The TIA's 4-pin
    feedback nets stay wired end-to-end through generate_schematic."""
    from skidl import Circuit

    d = tempfile.mkdtemp(prefix="skidl_stub_")
    try:
        c = Circuit(name="tia_full")
        _build_tia(c)
        text = _gen_file(
            c, d, "tia_full", seed=1,
            auto_stub_max_wire_pins=5, auto_stub_max_wire_dist=4000,
        )
        assert len(re.findall(r"\(wire\b", text)) >= 4
    finally:
        shutil.rmtree(d, ignore_errors=True)


@requires_libs
def test_deconflict_routing_deterministic():
    from skidl import Circuit

    def run():
        d = tempfile.mkdtemp(prefix="skidl_stub_")
        try:
            c = Circuit(name="det")
            _build_tia(c)
            text = _gen_file(c, d, "det", seed=1)
            return sorted(re.findall(r"\(xy [-\d. ]+\)", text))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    assert run() == run()
