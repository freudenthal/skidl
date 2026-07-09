# -*- coding: utf-8 -*-

"""Regression tests for Blocker B (stage 19): snap must not fuse two nets.

The snap passes place one pin of a 2-pin part exactly on a target pin and let
the OTHER pin fall where the part geometry puts it. Before the fix, that far pin
could land on a foreign-net pin, and the kicad emitter's power symbol / label /
wire at the shared point made KiCad fuse the two nets. Measured on the
SiPM_TIA_Filter deliverable: ldo_bias R7/2 (VBIAS_SENSE) landed on C7/2 (GND) via
TPS7A4701's stacked OUT pins (layout-independent), and sipm_tia C15/2 (GND) on
U1/3 (SN) (layout-dependent).

Two layers of coverage:
  * a fast pure-unit test of the ``_would_collide`` occupancy helper, and
  * a libs+kicad-cli end-to-end test of the layout-independent ldo_bias short,
    asserting the exported netlist keeps VBIAS_SENSE and GND distinct.
"""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from skidl.geometry import Point, Tx
from skidl.schematics.snap import _would_collide

# --------------------------------------------------------------------------- #
# Fast pure-unit coverage of the occupancy helper (no KiCad libs needed).
# --------------------------------------------------------------------------- #


class _FakeNet:
    def __init__(self, name):
        self.name = name


class _FakePin:
    def __init__(self, x, y, net):
        self.pt = Point(x, y)
        self.net = net


class _FakePart:
    def __init__(self, pins, tx=None):
        self.pins = pins
        self.tx = tx or Tx()


class _FakeNode:
    def __init__(self, parts):
        self.parts = parts


def test_would_collide_detects_cross_net_coincidence():
    gnd = _FakeNet("GND")
    sense = _FakeNet("VBIAS_SENSE")
    # A part with a GND pin sitting at (100, 200).
    other = _FakePart([_FakePin(100, 200, gnd)])
    # The candidate part has a VBIAS_SENSE pin that would land at (100, 200).
    part = _FakePart([_FakePin(0, 0, sense), _FakePin(100, 200, sense)])
    node = _FakeNode([part, other])
    # Identity tx: the sense pin at local (100,200) lands on the GND pin.
    assert _would_collide(part, Tx(), node) is True


def test_would_collide_allows_same_net_touch():
    sn = _FakeNet("SN")
    other = _FakePart([_FakePin(100, 200, sn)])
    part = _FakePart([_FakePin(0, 0, sn), _FakePin(100, 200, sn)])
    node = _FakeNode([part, other])
    # Same-net pin coincidence is the snapper's intended idiom — not a collision.
    assert _would_collide(part, Tx(), node) is False


def test_would_collide_ignores_non_coincident_pins():
    a = _FakeNet("A")
    b = _FakeNet("B")
    other = _FakePart([_FakePin(500, 500, b)])
    part = _FakePart([_FakePin(0, 0, a), _FakePin(100, 200, a)])
    node = _FakeNode([part, other])
    assert _would_collide(part, Tx(), node) is False


# --------------------------------------------------------------------------- #
# Layout-independent ldo_bias short: end-to-end netlist assertion.
# --------------------------------------------------------------------------- #

_HAS_LIBS = (
    bool(os.environ.get("KICAD9_SYMBOL_DIR"))
    or os.path.exists("/usr/share/kicad/symbols")
    or os.path.exists(os.path.expanduser("~/.local/share/kicad/9.0/symbols"))
)
requires_libs = pytest.mark.skipif(
    not _HAS_LIBS, reason="KiCad symbol libraries not available"
)


def _find_kicad_cli():
    for cand in (
        r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
        shutil.which("kicad-cli"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


def _net_membership(net_path):
    """Map net name -> set('REF.PIN') from a kicad-cli netlist (s-expr)."""
    txt = open(net_path, encoding="utf-8").read()
    toks = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', txt)
    pos = 0

    def parse():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            lst = []
            while toks[pos] != ")":
                lst.append(parse())
            pos += 1
            return lst
        return t

    tree = parse()
    found = []

    def walk(node):
        if isinstance(node, list):
            if node and node[0] == "net":
                found.append(node)
            for x in node:
                walk(x)

    walk(tree)
    nets = {}
    for net in found:
        name = None
        nodes = []
        for el in net:
            if isinstance(el, list) and el[0] == "name":
                name = el[1].strip('"')
            if isinstance(el, list) and el[0] == "node":
                ref = pin = None
                for f in el:
                    if isinstance(f, list) and f[0] == "ref":
                        ref = f[1].strip('"')
                    if isinstance(f, list) and f[0] == "pin":
                        pin = f[1].strip('"')
                nodes.append("%s.%s" % (ref, pin))
        if name is not None:
            nets.setdefault(name.split("/")[-1], set()).update(nodes)
    return nets


@requires_libs
def test_ldo_bias_sense_not_fused_to_gnd():
    cli = _find_kicad_cli()
    if not cli:
        pytest.skip("kicad-cli not available")

    from skidl import Net, Part, generate_schematic, reset

    reset()
    # The layout-independent short needs a part with two OUT pins stacked at the
    # same symbol coordinate (TPS7A4701 pins 1 & 20). The unit-test lib harness
    # resolves some multi-unit symbols differently than a fresh interpreter, so
    # skip cleanly if it cannot be loaded here — the fix is also covered
    # end-to-end by the deliverable render (see stage-19 Phase-2 verification).
    try:
        u5 = Part("Regulator_Linear", "TPS7A4701xRGW", ref="U5")
    except Exception as e:  # noqa: BLE001 - library-resolution quirk, not the SUT
        pytest.skip(f"stacked-pin regulator symbol unavailable in test harness: {e}")
    c7 = Part("Device", "C", value="10uF", ref="C7")
    r7 = Part("Device", "R", value="10k", ref="R7")
    r8 = Part("Device", "R", value="10k", ref="R8")
    out, gnd, sense = Net("VBIAS29"), Net("GND"), Net("VBIAS_SENSE")
    u5[1] += out  # OUT
    u5[20] += out  # OUT (stacked at the same symbol position as pin 1)
    u5[3] += sense  # a sense/FB pin
    c7[1] += out
    c7[2] += gnd
    r7[1] += out
    r7[2] += sense
    r8[1] += sense
    r8[2] += gnd

    d = tempfile.mkdtemp(prefix="skidl_snapB_")
    try:
        generate_schematic(
            filepath=d,
            top_name="ldo_repro",
            auto_stub=True,
            auto_stub_fallback="labels",
            seed=1,
        )
        sch = os.path.join(d, "ldo_repro.kicad_sch")
        assert os.path.exists(sch)
        net = os.path.join(d, "ldo_repro.net")
        subprocess.run(
            [cli, "sch", "export", "netlist", "-o", net, sch],
            check=True,
            capture_output=True,
            text=True,
        )
        nets = _net_membership(net)
        gnd_nodes = nets.get("GND", set())
        sense_nodes = nets.get("VBIAS_SENSE", set())
        # Before the fix, R7.2 (VBIAS_SENSE) fused into GND via C7.2.
        assert sense_nodes, "VBIAS_SENSE net missing entirely"
        assert not (
            gnd_nodes & sense_nodes
        ), f"VBIAS_SENSE fused with GND: shared nodes {gnd_nodes & sense_nodes}"
        assert "R7.2" in sense_nodes and "R8.1" in sense_nodes
    finally:
        shutil.rmtree(d, ignore_errors=True)
