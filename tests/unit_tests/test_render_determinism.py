# -*- coding: utf-8 -*-

"""Committed determinism harness for the render pipeline.

The render pipeline must be REPRODUCIBLE: rendering the same circuit twice must
produce byte-identical ``.kicad_sch`` files (modulo the title-block date), even
across processes with a different ``PYTHONHASHSEED``. Two nondeterminism sources
were closed for this (wired-render-default plan, Phase 2):

* the ``seed`` now defaults to 42 (was ``random.seed(None)`` = wall-clock);
* the two surviving order-dependent RNG/object-set sources — ``overlap_force``'s
  symmetry-breaker (now a deterministic per-part-pair jitter) and the cosmetic
  ``remove_jogs`` wire pass (``random.shuffle`` + ``list(set(...))`` → sorted on
  stable geometric keys). The latter reshaped wires per process AND, via the
  sheet bbox that centers the page, shifted every part.

These tests render in SEPARATE subprocesses with DIFFERENT ``PYTHONHASHSEED``s
and assert byte-identity (date excluded). Do NOT "fix" a regression here by
weakening the comparison — fix the pipeline.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# --- real-KiCad-10 discovery (mirrors test_kicad10.py) ---------------------
_KICAD_CLI_CANDIDATES = [
    r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
    "/usr/bin/kicad-cli",
    "/usr/local/bin/kicad-cli",
]


def _kicad10_symbols_available():
    import skidl.tools.kicad10.lib as k10

    return bool(k10._discover_default_symbol_dirs("10")) or bool(
        os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    )


requires_kicad10 = pytest.mark.skipif(
    not _kicad10_symbols_available(),
    reason="requires real KiCad 10 stock symbol libraries",
)


# Render script run in a child process. Prints nothing; writes <top>.kicad_sch
# (+ child sheets) into argv[1]. Kept dependency-free of skidl_eda so it is a
# pure-fork test.
_RENDER = r"""
import os
import sys
from skidl import (Circuit, Net, Part, POWER, KICAD10, subcircuit,
                   lib_search_paths, set_default_tool)
set_default_tool(KICAD10)
lib_search_paths["kicad10"] = ["."] + __import__(
    "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]).default_lib_paths()

which, outdir = sys.argv[1], sys.argv[2]

def flat(ckt):
    with ckt:
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

def hier(ckt):
    @subcircuit
    def stage(vin, vout, vpos, gnd):
        u = Part("Amplifier_Operational", "OPA340NA")
        r1 = Part("Device", "R", value="1k"); r2 = Part("Device", "R", value="1k")
        u["5"] += vpos; u["2"] += gnd; u["3"] += vin; u["4"] += vout; u["1"] += vout
        r1[1] += vin; r1[2] += gnd; r2[1] += vout; r2[2] += gnd
        c = Part("Device", "C", value="100n"); c[1] += vpos; c[2] += gnd
    with ckt:
        vpos = Net("+5V"); vpos.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        a, b, c = Net("A"), Net("B"), Net("C")
        stage(a, b, vpos, gnd, tag="s1"); stage(b, c, vpos, gnd, tag="s2")

top = {"flat": "detf", "hier": "deth", "hier_netfirst": "detn"}[which]
ckt = Circuit(name=top)
{"flat": flat, "hier": hier, "hier_netfirst": hier}[which](ckt)
if which == "hier_netfirst":
    # Reproduce the harness order: generate_netlist (which runs check_tags and
    # assigns fallback tags to every un-tagged part) BEFORE generate_schematic.
    # This is the order skidl_eda.generate() uses; a RANDOM fallback tag makes
    # every symbol/pin UUID drift run-to-run, which schematic-only rendering
    # (the flat/hier cases) never exercises because tag stays None -> ref.
    ckt.generate_netlist(tool=KICAD10, file_=os.path.join(outdir, top + ".net"))
ckt.generate_schematic(tool=KICAD10, filepath=outdir, top_name=top,
                       seed_placement=True, auto_stub=False)
"""


def _render_in_subprocess(which, outdir, hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    env["PYTHONUTF8"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _RENDER, which, str(outdir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, f"render failed ({which}, seed {hashseed}):\n{r.stderr}"


def _normalized_sheets(outdir):
    """Map {filename -> content with the title-block date neutralized}."""
    out = {}
    for f in sorted(Path(outdir).glob("*.kicad_sch")):
        txt = f.read_text(encoding="utf-8")
        txt = re.sub(r'\(date "[^"]*"\)', '(date "X")', txt)
        out[f.name] = txt
    return out


@requires_kicad10
@pytest.mark.parametrize("which", ["flat", "hier", "hier_netfirst"])
def test_render_byte_identical_across_hashseed(which):
    """Same circuit, two processes, different PYTHONHASHSEED -> byte-identical
    schematic (date excluded), for the wired (seed_placement) default path.

    The ``hier_netfirst`` case runs the netlist-then-schematic order the real
    harness uses. It is RED before the deterministic-tag fix (random fallback
    tags drift the symbol/pin UUIDs) and GREEN after."""
    a = tempfile.mkdtemp(prefix=f"skidl_det_{which}_a_")
    b = tempfile.mkdtemp(prefix=f"skidl_det_{which}_b_")
    _render_in_subprocess(which, a, hashseed=0)
    _render_in_subprocess(which, b, hashseed=12345)
    sa, sb = _normalized_sheets(a), _normalized_sheets(b)
    assert sa, "no schematic produced"
    assert set(sa) == set(sb), f"sheet set differs: {set(sa)} vs {set(sb)}"
    for name in sa:
        assert sa[name] == sb[name], f"{name} differs across PYTHONHASHSEED"


# Focused, render-free check: the fallback tag check_tags() assigns must be a
# deterministic function of (hierpath, ref), NOT random. Two processes with a
# different PYTHONHASHSEED must produce identical tag-per-ref maps.
_TAG_DUMP = r"""
import json, sys
from skidl import (Circuit, Net, Part, KICAD10, lib_search_paths, set_default_tool)
set_default_tool(KICAD10)
lib_search_paths["kicad10"] = ["."] + __import__(
    "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]).default_lib_paths()
ckt = Circuit(name="tagd")
with ckt:
    r1 = Part("Device", "R", value="1k"); r2 = Part("Device", "R", value="2k")
    c1 = Part("Device", "C", value="1u")
    n = Net("N"); r1[1] += n; r2[1] += n; c1[1] += n
ckt.check_tags()
print(json.dumps({p.ref: p.tag for p in ckt.parts}))
"""


def _dump_tags(hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    env["PYTHONUTF8"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _TAG_DUMP],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"tag dump failed (seed {hashseed}):\n{r.stderr}"
    return __import__("json").loads(r.stdout.strip().splitlines()[-1])


@requires_kicad10
def test_check_tags_deterministic_across_hashseed():
    """Fallback tags derived by check_tags() are identical across processes."""
    ta = _dump_tags(0)
    tb = _dump_tags(9999)
    assert ta and all(v for v in ta.values()), f"missing tags: {ta}"
    assert ta == tb, f"tag maps differ across PYTHONHASHSEED:\n{ta}\n{tb}"
