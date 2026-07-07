# -*- coding: utf-8 -*-

"""Committed determinism harness for the render pipeline (stage 19).

These render the same circuit twice, in one process, with an explicit RNG
``seed`` and compare the PART-PLACEMENT fingerprint.

STATUS (stage 19, measured 2026-07-07): the render pipeline is NOT yet
reproducible even with ``seed=1``. Threading the seed removed the OS-entropy
source (``random.seed(None)``), and stage-19 added stable part/pin/Face sort
keys, but the force-directed placer AND the switchbox maze router still iterate
Python sets of *objects* in ``id()`` order at many sites — so which object
receives which random draw varies per build. A trivial 2-part divider happens to
be stable, but a realistic ~13-part circuit diverges essentially every run
(large layout shifts). Making the whole pipeline reproducible is a substantial
follow-up (systematically replacing id-ordered iteration with stable keys across
place.py + route.py); it is NOT required for Blocker B correctness, whose snap
fix guarantees no cross-net coincidence via a post-snap invariant sweep
regardless of layout.

These tests are therefore marked ``xfail(strict=False)``: they document the
target and serve as the committed measurement that will flip to passing once the
placer/router are made deterministic. Do NOT "fix" them by weakening the
fingerprint — fix the pipeline.

Like test_seed_integration, these need real KiCad symbol libraries and skip
otherwise.
"""

import glob
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


def _symbol_placements(text):
    """Sorted multiset of the SYMBOL ``(at ...)`` tokens = deterministic part placement."""
    return sorted(
        re.findall(r"\(symbol\b[^\n]*\n\s*\(lib_id[^\n]*\n\s*\(at ([-\d. ]+)\)", text)
    )


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_tia(circuit):
    """~13-part flat TIA (op-amp + passives), matching the seed smoke fixture."""
    from skidl import Net, Part

    with circuit:
        u1 = Part("Amplifier_Operational", "OPA340NA")
        rf = Part("Device", "R", value="1M")
        cf = Part("Device", "C", value="2p")
        rin = Part("Device", "R", value="50")
        rl = Part("Device", "R", value="1k")
        rb1 = Part("Device", "R", value="10k")
        rb2 = Part("Device", "R", value="10k")
        cb = Part("Device", "C", value="100n")
        gnd, vplus, sig = Net("GND"), Net("+5V"), Net("SIG")
        bias = Net("BIAS")
        u1["4"] += rf[1], cf[1], rin[2]
        u1["1"] += rf[2], cf[2], rl[1]
        u1["3"] += bias
        u1["2"] += gnd
        u1["5"] += vplus
        rb1[1] += vplus
        rb1[2] += bias
        rb2[1] += bias
        rb2[2] += gnd
        cb[1] += bias
        cb[2] += gnd
        rin[1] += sig
        rl[2] += gnd


def _build_hier(circuit):
    """Top with one @subcircuit child owning an internal wireable net."""
    from skidl import Net, Part, subcircuit

    @subcircuit
    def rc(vin, gnd):
        mid = Net()
        r1 = Part("Device", "R", value="10k")
        c1 = Part("Device", "C", value="100n")
        vin += r1[1]
        r1[2] += mid
        mid += c1[1]
        c1[2] += gnd

    with circuit:
        vin, gnd = Net("VIN"), Net("GND")
        rc(vin, gnd, tag="b1")


def _gen_symbol_placements(builder, top):
    from skidl import Circuit

    d = tempfile.mkdtemp(prefix="skidl_det_")
    try:
        c = Circuit(name=top)
        builder(c)
        c.generate_schematic(
            filepath=d,
            top_name=top,
            auto_stub=True,
            auto_stub_fallback="labels",
            seed=1,
        )
        path = os.path.join(d, f"{top}.kicad_sch")
        assert os.path.exists(path), f"schematic not generated at {path}"
        # Read every generated sheet (top + hierarchical children) so the
        # fingerprint covers child-owned parts too — each child is its own file.
        text = "".join(
            _read(p) for p in sorted(glob.glob(os.path.join(d, "*.kicad_sch")))
        )
        return _symbol_placements(text)
    finally:
        shutil.rmtree(d, ignore_errors=True)


_XFAIL_DET = pytest.mark.xfail(
    reason="stage-19 follow-up: force-directed placer + maze router still iterate "
    "object sets in id() order, so render placement is not reproducible even with "
    "seed=1 (see module docstring). Not required for Blocker B correctness.",
    strict=False,
)


@requires_libs
@_XFAIL_DET
def test_flat_placement_deterministic():
    # Two in-process renders with the same seed -> identical part placement.
    a = _gen_symbol_placements(_build_tia, "det_tia")
    b = _gen_symbol_placements(_build_tia, "det_tia")
    assert a, "expected some symbol placements in the fingerprint"
    assert a == b


@requires_libs
@_XFAIL_DET
def test_hierarchical_placement_deterministic():
    # Covers the per-node reseed path (one @subcircuit child).
    a = _gen_symbol_placements(_build_hier, "det_hier")
    b = _gen_symbol_placements(_build_hier, "det_hier")
    assert a, "expected some symbol placements in the fingerprint"
    assert a == b
