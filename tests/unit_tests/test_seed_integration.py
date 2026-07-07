# -*- coding: utf-8 -*-

"""End-to-end smoke tests for seed_placement wired into the placer (stage 19.3).

These need real KiCad symbol libraries, so they skip unless libraries are
available (KICAD9_SYMBOL_DIR set, or the standard Linux install paths exist).
They verify the integration works and is deterministic — quality evaluation is
Phase D (the benchmark), not here.
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


@pytest.fixture
def out_dir():
    d = tempfile.mkdtemp(prefix="skidl_seed_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


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
        # -in='4', OUT='1', +in='3', V-='2', V+='5'
        u1["4"] += rf[1], cf[1], rin[2]
        u1["1"] += rf[2], cf[2], rl[1]
        u1["3"] += gnd
        u1["2"] += gnd
        u1["5"] += vplus
        rin[1] += sig
        rl[2] += gnd


def _gen(circuit, out, top, **opts):
    circuit.generate_schematic(
        filepath=out,
        top_name=top,
        auto_stub=True,
        auto_stub_fallback="labels",
        **opts,
    )
    path = os.path.join(out, f"{top}.kicad_sch")
    assert os.path.exists(path), f"schematic not generated at {path}"
    return path


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _placements(text):
    """Sorted multiset of all ``(at X Y ...)`` tokens = the placement fingerprint.

    A whole-file byte compare is NOT usable for determinism: skidl iterates over
    Python sets of parts/nets when emitting, so the ELEMENT ORDER (and the
    generated uuids) varies run-to-run even when every coordinate is identical.
    The sorted position multiset is order- and uuid-independent, so it isolates
    the thing we actually care about — where parts and labels landed.
    """
    return sorted(re.findall(r"\(at [-\d. ]+\)", text))


@requires_libs
def test_seed_divider_produces_wires(out_dir):
    from skidl import Circuit

    c = Circuit(name="div_seed")
    _build_divider(c)
    path = _gen(c, out_dir, "div_seed", seed_placement=True)
    text = _read(path)
    assert len(re.findall(r"\(wire\b", text)) >= 1


def _gen_placements(top, seed):
    from skidl import Circuit

    d = tempfile.mkdtemp(prefix="skidl_seed_det_")
    try:
        c = Circuit(name=top)
        _build_divider(c)
        return _placements(_read(_gen(c, d, top, seed_placement=True, seed=seed)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


@requires_libs
def test_seed_deterministic_placement():
    # Same inputs -> identical placement fingerprint (reproducible pipeline).
    # NOTE: a whole-file compare is not usable (skidl emits set-ordered elements
    # + fresh uuids); and RNG-seed-independence of the *part* placement is
    # measured rigorously in Phase D (benchmark HPWL variance), not here — the
    # (at) fingerprint below also includes label/terminal positions, which the
    # net-terminal placer still derives with the RNG.
    assert _gen_placements("det1", seed=1) == _gen_placements("det1", seed=1)


@requires_libs
def test_seed_tia_completes_without_routing_failure(out_dir):
    from skidl import Circuit

    c = Circuit(name="tia_seed")
    _build_tia(c)
    # The run must complete (fallback may stub nets, but no unhandled failure).
    path = _gen(c, out_dir, "tia_seed", seed_placement=True)
    assert os.path.getsize(path) > 0


def _build_hier_rc(circuit):
    """One small @subcircuit (RC with an internal wireable ``mid`` net)."""
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


def _child_wire_count(out, top, **opts):
    from skidl import Circuit

    os.makedirs(out, exist_ok=True)
    c = Circuit(name=top)
    _build_hier_rc(c)
    _gen(c, out, top, flatness=0.0, **opts)
    child = os.path.join(out, f"{top}_b1.kicad_sch")
    assert os.path.exists(child), f"child sheet not generated at {child}"
    return len(re.findall(r"\(wire\b", _read(child)))


@requires_libs
def test_small_subcircuit_max_zero_keeps_child_wires(out_dir):
    # Default (skidl blanket-stubs <=6-net subcircuits to labels): the small RC
    # child sheet routes no wires. Setting the knob to 0 keeps its local wire.
    default_wires = _child_wire_count(
        os.path.join(out_dir, "def"), "hier_def"
    )
    kept_wires = _child_wire_count(
        os.path.join(out_dir, "keep"), "hier_keep", auto_stub_small_subcircuit_max=0
    )
    assert default_wires == 0
    assert kept_wires >= 1
