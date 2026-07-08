# -*- coding: utf-8 -*-

"""Stage-25 phase 2: per-connected-component closure labels (union-find).

Pure ``net_islands`` unit tests (no libraries) plus an end-to-end check that
the closure labeller closes every net island (no pin_not_connected), keeps
power pins on power symbols, and never fuses two nets (strict audit).
"""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from skidl.schematics.decisions import net_islands


# --------------------------------------------------------------------------
# Pure union-find tests (no KiCad libraries needed)
# --------------------------------------------------------------------------
def test_islands_all_wired_one_component():
    # 3 pins on a straight wire -> one island.
    pins = [("a", 0, 0), ("b", 100, 0), ("c", 200, 0)]
    segs = [(0, 0, 100, 0), (100, 0, 200, 0)]
    islands = net_islands(pins, segs, tol=1.0)
    assert len(islands) == 1
    assert set(islands[0]) == {"a", "b", "c"}


def test_islands_split_when_a_pin_is_unwired():
    # a-b wired; c has only its own (disconnected) stub -> two islands.
    pins = [("a", 0, 0), ("b", 100, 0), ("c", 500, 500)]
    segs = [(0, 0, 100, 0), (500, 500, 550, 500)]  # c's stub goes nowhere
    islands = net_islands(pins, segs, tol=1.0)
    sets = sorted((sorted(g) for g in islands), key=len)
    assert sets == [["c"], ["a", "b"]]


def test_islands_t_junction_pin_on_segment_interior():
    # A pin end sitting on a segment interior (T-join) is connected.
    pins = [("a", 0, 0), ("b", 200, 0), ("c", 100, 0)]  # c on the a-b segment
    segs = [(0, 0, 200, 0)]
    islands = net_islands(pins, segs, tol=1.0)
    assert len(islands) == 1


def test_islands_deterministic():
    pins = [("a", 0, 0), ("b", 100, 0), ("c", 300, 0)]
    segs = [(0, 0, 100, 0)]
    assert net_islands(pins, segs, 1.0) == net_islands(pins, segs, 1.0)


# --------------------------------------------------------------------------
# End-to-end closure behaviour (needs KiCad libraries + kicad-cli).
#
# Generation runs in a SUBPROCESS -- the same way circuit-synth invokes skidl,
# and the only way to get deterministic, uncorrupted placement: running many
# generate_schematic() calls in ONE process hits the known stage-19 placement
# nondeterminism (id-ordered set iteration; see test_render_determinism xfail),
# which yields unstable/degenerate nodes. One generation per fresh interpreter
# is the real usage path.
# --------------------------------------------------------------------------
import sys

_HAS_LIBS = (
    bool(os.environ.get("KICAD9_SYMBOL_DIR"))
    or os.path.exists("/usr/share/kicad/symbols")
    or os.path.exists(os.path.expanduser("~/.local/share/kicad/9.0/symbols"))
)
_HAS_CLI = shutil.which("kicad-cli") is not None
requires_e2e = pytest.mark.skipif(
    not (_HAS_LIBS and _HAS_CLI), reason="KiCad libraries / kicad-cli not available"
)

_GEN_SCRIPT = r'''
import sys
from skidl import Circuit, Net, Part
out = sys.argv[1]
c = Circuit(name="tia")
with c:
    u1 = Part("Amplifier_Operational", "OPA340NA")
    rf = Part("Device", "R", value="1M"); cf = Part("Device", "C", value="2p")
    rin = Part("Device", "R", value="50"); rl = Part("Device", "R", value="1k")
    gnd, vplus, sig = Net("GND"), Net("+5V"), Net("SIG")
    u1["4"] += rf[1], cf[1], rin[2]
    u1["1"] += rf[2], cf[2], rl[1]
    u1["3"] += gnd; u1["2"] += gnd; u1["5"] += vplus
    rin[1] += sig; rl[2] += gnd
c.generate_schematic(filepath=out, top_name="tia", auto_stub=True,
    auto_stub_fallback="labels", deconflict_stubs=True,
    auto_stub_max_wire_pins=5, auto_stub_max_wire_dist=4000, seed=1)
print("GEN_OK")
'''


def _subprocess_gen(out, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([sys.executable, "-c", _GEN_SCRIPT, out],
                       capture_output=True, text=True, env=env)
    path = os.path.join(out, "tia.kicad_sch")
    return r, path


def _erc_text(path):
    rpt = path + ".rpt"
    subprocess.run(["kicad-cli", "sch", "erc", path, "-o", rpt], capture_output=True)
    with open(rpt, encoding="utf-8") as f:
        return f.read()


@requires_e2e
def test_closure_no_pin_not_connected_and_power_symbols():
    d = tempfile.mkdtemp(prefix="skidl_closure_")
    try:
        r, path = _subprocess_gen(d)
        assert os.path.exists(path), r.stderr[-2000:]
        with open(path, encoding="utf-8") as f:
            sch = f.read()
        # Power pins land on power symbols, not orphaned.
        assert sch.count('lib_id "power:') >= 3
        # Signal nets carry closure labels.
        assert len(re.findall(r"\(label ", sch)) >= 1
        # No signal pin is left unconnected (the split the closure labels fix)
        # and off-grid stays zero.
        erc = _erc_text(path)
        assert "[pin_not_connected]" not in erc
        assert "off_grid" not in erc.lower()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@requires_e2e
def test_closure_strict_audit_no_fusion():
    """Two distinct nets must never share a coordinate: generation completing
    under SKIDL_AUDIT_STRICT (which raises SheetConnectivityError on a fusion)
    proves the deconflicted ends + closure labels never merge nets."""
    d = tempfile.mkdtemp(prefix="skidl_closure_")
    try:
        r, path = _subprocess_gen(d, extra_env={"SKIDL_AUDIT_STRICT": "1"})
        assert os.path.exists(path) and "GEN_OK" in r.stdout, r.stderr[-2000:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
