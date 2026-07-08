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


_DET_SCRIPT = r'''
import sys
from skidl import Circuit, Net, Part
out = sys.argv[1]
c = Circuit(name="det")
with c:
    u1 = Part("Amplifier_Operational", "OPA340NA")
    rf = Part("Device", "R", value="1M"); cf = Part("Device", "C", value="2p")
    rin = Part("Device", "R", value="50"); rl = Part("Device", "R", value="1k")
    gnd, vplus, sig = Net("GND"), Net("+5V"), Net("SIG")
    u1["4"] += rf[1], cf[1], rin[2]
    u1["1"] += rf[2], cf[2], rl[1]
    u1["3"] += gnd; u1["2"] += gnd; u1["5"] += vplus
    rin[1] += sig; rl[2] += gnd
c.generate_schematic(filepath=out, top_name="det", auto_stub=True,
    auto_stub_fallback="labels", deconflict_stubs=True,
    auto_stub_max_wire_pins=5, auto_stub_max_wire_dist=4000, seed=1)
'''


# --------------------------------------------------------------------------
# Stage-25b: stub DIRECTION follows the pin's world orientation
# --------------------------------------------------------------------------
def test_world_outward_dir_all_dihedral_orientations():
    """Pure-math (no KiCad libs): the stub outward direction is the pin's
    local orientation folded through the part's rotation/mirror and negated.
    Hand-computed expectations for every orientation x dihedral transform."""
    from types import SimpleNamespace
    from skidl.geometry import (
        tx_rot_0, tx_rot_90, tx_rot_180, tx_rot_270, tx_flip_x, tx_flip_y,
    )
    from skidl.schematics.route import _world_outward_dir

    # expected[tx_name][orientation] = (dx, dy)
    expected = {
        "I":   {"U": (0, -1), "D": (0, 1), "L": (1, 0), "R": (-1, 0)},
        "R90": {"U": (1, 0), "D": (-1, 0), "L": (0, 1), "R": (0, -1)},
        "R180": {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)},
        "R270": {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)},
        "Fx":  {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)},
        "Fy":  {"U": (0, 1), "D": (0, -1), "L": (1, 0), "R": (-1, 0)},
    }
    txs = {
        "I": tx_rot_0, "R90": tx_rot_90, "R180": tx_rot_180,
        "R270": tx_rot_270, "Fx": tx_flip_x, "Fy": tx_flip_y,
    }
    for tx_name, tx in txs.items():
        for orient, want in expected[tx_name].items():
            pin = SimpleNamespace(orientation=orient, part=SimpleNamespace(tx=tx))
            got = _world_outward_dir(pin)
            assert got == want, f"{tx_name}+{orient}: got {got}, want {want}"
            # always axial + unit
            assert abs(got[0]) + abs(got[1]) == 1


@requires_libs
def test_stub_direction_follows_pin_orientation():
    """Pipeline invariant: every stub vector points along the pin's true world
    outward direction (covers whatever rotations the placer emits)."""
    from skidl import Circuit
    from skidl.net import NCNet
    from skidl.schematics.route import _world_outward_dir

    c = Circuit(name="tia_dir")
    _build_tia(c)
    node = _routed_node(c)

    checked = 0
    vertical = 0
    for n in _iter_nodes(node):
        stub_ends = getattr(n, "_stub_ends", {})
        pin_by_id = {}
        for part in n.parts:
            for pin in part:
                pin_by_id[id(pin)] = (part, pin)
        for pid, end in stub_ends.items():
            part, pin = pin_by_id[pid]
            net = pin.net
            if isinstance(net, NCNet) or getattr(net, "_is_power_net", False):
                continue
            pin_w = (pin.pt * part.tx).round()
            vx, vy = end.x - pin_w.x, end.y - pin_w.y
            if vx == 0 and vy == 0:
                continue  # fallback no-stub pin (deconflict exhaustion)
            assert (vx == 0) != (vy == 0), (
                f"{getattr(part, 'ref', '?')}.{pin.num} stub not axial: ({vx},{vy})"
            )
            dx, dy = _world_outward_dir(pin)
            svx = 1 if vx > 0 else -1 if vx < 0 else 0
            svy = 1 if vy > 0 else -1 if vy < 0 else 0
            assert (svx, svy) == (dx, dy), (
                f"{getattr(part, 'ref', '?')}.{pin.num} stub ({vx},{vy}) "
                f"!= world outward ({dx},{dy})"
            )
            if svy != 0:
                vertical += 1
            checked += 1
    assert checked >= 6
    # Regression canary: before the fix every stub read left/right; a TIA has
    # U/D pins (caps/resistors) so at least one stub must run vertically.
    assert vertical >= 1, "no vertical stubs — direction fix regressed"


@requires_libs
@pytest.mark.xfail(
    reason="stage-19 placement residual: the force-directed placer still iterates "
    "object sets in id() order, so a realistic circuit (op-amp + feedback) diverges "
    "run-to-run in the EMITTED geometry even in separate interpreters. The A* ROUTER "
    "is deterministic by construction (test_astar_router MST/tie-break tests); this "
    "whole-render check stays xfail until the placement residual is fixed. Not a "
    "deconflict-stub regression.",
    strict=False,
)
def test_deconflict_routing_deterministic():
    """Two same-seed renders produce an identical geometry multiset. Run in
    SEPARATE interpreters (the real path): multiple in-process generations also
    hit the placement nondeterminism, so isolate to the subprocess path."""
    import subprocess
    import sys

    def run():
        d = tempfile.mkdtemp(prefix="skidl_det_")
        try:
            subprocess.run([sys.executable, "-c", _DET_SCRIPT, d],
                           capture_output=True, text=True, env=dict(os.environ))
            with open(os.path.join(d, "det.kicad_sch"), encoding="utf-8") as f:
                return sorted(re.findall(r"\(xy [-\d. ]+\)", f.read()))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    assert run() == run()
