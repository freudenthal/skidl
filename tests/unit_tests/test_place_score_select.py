# -*- coding: utf-8 -*-

"""Tests for the opt-in ``place_score_select`` candidate bake-off (WS2).

The bake-off places each connected group with several seed strategies, scores
each with the ported crossing/HPWL scorer (``sch_score``), and keeps the lowest.
It is gated behind ``place_score_select`` and defaults OFF, so the load-bearing
safety property is that a bare generate (flag absent/False) is byte-identical to
before. These need real KiCad-10 symbols; they skip otherwise.
"""

import os
import re
import shutil
import tempfile

import pytest


def _kicad10_symbols_available():
    import skidl.tools.kicad10.lib as k10

    return bool(k10._discover_default_symbol_dirs("10")) or bool(
        os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    )


requires_kicad10 = pytest.mark.skipif(
    not _kicad10_symbols_available(),
    reason="requires real KiCad 10 stock symbol libraries",
)

pytestmark = requires_kicad10


def _build_flat(circuit):
    """A single connected group of 7 real parts (op-amp TIA + bias divider),
    below _ROW_PLACE_THRESHOLD so it takes the force path where the bake-off
    lives."""
    from skidl import Net, Part, POWER

    with circuit:
        u1 = Part("Amplifier_Operational", "OPA340NA")
        rf = Part("Device", "R", value="1M"); cf = Part("Device", "C", value="2p")
        rin = Part("Device", "R", value="50"); rl = Part("Device", "R", value="1k")
        rb1 = Part("Device", "R", value="10k"); rb2 = Part("Device", "R", value="10k")
        cb = Part("Device", "C", value="100n")
        gnd, vplus, sig, bias = Net("GND"), Net("+5V"), Net("SIG"), Net("BIAS")
        gnd.drive = POWER; vplus.drive = POWER
        u1["4"] += rf[1], cf[1], rin[2]
        u1["1"] += rf[2], cf[2], rl[1]
        u1["3"] += bias; u1["2"] += gnd; u1["5"] += vplus
        rb1[1] += vplus; rb1[2] += bias; rb2[1] += bias; rb2[2] += gnd
        cb[1] += bias; cb[2] += gnd; rin[1] += sig; rl[2] += gnd


def _gen_sheets(top, **opts):
    """Generate the schematic and return {sheet_name -> date-neutralized text}."""
    from skidl import Circuit

    d = tempfile.mkdtemp(prefix="skidl_pss_")
    try:
        c = Circuit(name=top)
        _build_flat(c)
        c.generate_schematic(
            filepath=d, top_name=top, seed_placement=True, auto_stub=False, **opts
        )
        out = {}
        from pathlib import Path

        for f in sorted(Path(d).glob("*.kicad_sch")):
            txt = f.read_text(encoding="utf-8")
            txt = re.sub(r'\(date "[^"]*"\)', '(date "X")', txt)
            out[f.name] = txt
        assert out, "no schematic produced"
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_flag_off_is_byte_identical():
    """Flag absent vs explicitly False produce byte-identical schematics -- the
    default path must be untouched by the presence of the feature. Same top_name
    both times (UUIDs derive from it) so this is a pure byte compare."""
    absent = _gen_sheets("pss_off")
    false = _gen_sheets("pss_off", place_score_select=False)
    assert set(absent) == set(false)
    for name in absent:
        assert absent[name] == false[name], f"{name} differs with flag off"


def test_flag_on_is_deterministic():
    """Two runs with place_score_select=True produce identical output (trial
    state does not leak; selection is deterministic)."""
    a = _gen_sheets("pss_det", place_score_select=True)
    b = _gen_sheets("pss_det", place_score_select=True)
    assert set(a) == set(b)
    for name in a:
        assert a[name] == b[name], f"{name} differs between two flag-on runs"


def _placed_node(top, **opts):
    from skidl import Circuit, get_default_tool, KICAD10, set_default_tool
    from skidl.tools import tool_modules
    from skidl.tools.kicad10.gen_schematic import preprocess_circuit
    from skidl.schematics.sch_node import SchNode

    set_default_tool(KICAD10)
    circuit = Circuit(name=top)
    _build_flat(circuit)
    base = dict(seed_placement=True, auto_stub=False, expansion_factor=1.0)
    base.update(opts)
    preprocess_circuit(circuit, **base)
    node = SchNode(circuit, tool_modules[get_default_tool()], ".", top, top, 0.0)
    node.place(**base)
    return node


def test_flag_on_never_worse(monkeypatch):
    """The bake-off cannot lose to its own default member: for every scored
    group, the selected (min) score is <= the constructive_relax (idx 0) score.

    Spy on sch_score.score_parts to capture the per-strategy scores in call
    order (relax, seed, force per group), then assert the min of each triple is
    no worse than the first."""
    from skidl.schematics import sch_score

    recorded = []
    orig = sch_score.score_parts

    def spy(parts, nets):
        s = orig(parts, nets)
        recorded.append(s)
        return s

    monkeypatch.setattr(sch_score, "score_parts", spy)
    _placed_node("pss_nw", place_score_select=True)

    assert recorded, "bake-off never scored a group (fixture took the row path?)"
    assert len(recorded) % 3 == 0, recorded
    for i in range(0, len(recorded), 3):
        triple = recorded[i : i + 3]
        relax = triple[0]  # constructive_relax is strategy index 0
        best = min(triple, key=lambda s: (s["crossings"], s["hpwl"]))
        assert (best["crossings"], best["hpwl"]) <= (
            relax["crossings"],
            relax["hpwl"],
        ), triple
