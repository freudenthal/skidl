# -*- coding: utf-8 -*-

"""Stage-24 label-scoping regression: sheet-INTERNAL nets must emit a local
``label`` (sheet-scoped), never a project-wide ``global_label``.

A ``global_label`` connects by NAME across every sheet in the project, so an
internal net named e.g. ``SW`` reused on two sheets would silently merge -- a
Blocker-B-class leak. The stage-24 emitter classifies a net as sheet-internal
(no pin on a part outside the node) and emits a local ``label`` for it, keeping
``global_label`` only for genuine cross-sheet (boundary) nets.
"""

import glob
import os
import sys

import pytest

from skidl import Net, Part, TEMPLATE, generate_schematic
from skidl.schematics.place import PlacementFailure
from skidl.schematics.route import RoutingFailure


def _render(top_name):
    out_root = "./test_data/schematic_output"
    py = ".".join(str(n) for n in sys.version_info[0:3])
    out_dir = os.path.join(out_root, py)
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, top_name) + "*.kicad_sch"):
        os.remove(f)
    generate_schematic(filepath=out_dir, top_name=top_name, flatness=1.0, retries=3)
    return out_dir


@pytest.mark.xfail(raises=(PlacementFailure, RoutingFailure))
def test_internal_signal_net_is_local_label():
    """A routed internal signal net emits a local ``label``, no ``global_label``."""
    r = Part("Device", "R", footprint="Resistor_SMD:R_0805_2012Metric",
             dest=TEMPLATE, value="10K")
    vcc = Net("VCC")   # power -> power symbol (not a label)
    gnd = Net("GND")   # power -> power symbol
    sig = Net("SIG")   # internal 2-pin signal -> routed + local backstop label
    vcc & r() & sig & r() & gnd

    out_dir = _render("test_scope_flat")
    files = glob.glob(os.path.join(out_dir, "test_scope_flat*.kicad_sch"))
    assert files, "no schematic written"
    text = "".join(open(f, encoding="utf-8").read() for f in files)

    # SIG must appear as a sheet-local label, never as a project-wide global_label.
    assert '(label "SIG"' in text, "internal SIG net has no local label backstop"
    assert 'global_label "SIG"' not in text, "internal SIG net leaked as global_label"
    # No sheet-internal net should be a global_label at all in a flat design.
    assert "global_label" not in text, "flat design emitted a global_label"
