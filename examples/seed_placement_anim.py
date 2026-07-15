#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Debug example: animate the constructive seed placer building a schematic.

Renders one slow GIF per hierarchical sheet showing the placer add parts one at a
time -- each a bounding box with L/R/U/D edge pin dots, an "up" arrow marking the
part's post-rotation orientation, a new colour per part, and a centred ref +
placement-order label.

Run (needs real KiCad-10 symbol libraries and Pillow -- `pip install .[debug]`)::

    python examples/seed_placement_anim.py --out .

See ``skidl.schematics.debug_anim`` for the reusable recorder/renderer, and
``kicadprojects/dual_phase_sqgen/make_placement_anim.py`` for a multi-sheet run.
"""

import argparse
import tempfile

from skidl import (
    Circuit,
    Net,
    Part,
    POWER,
    KICAD10,
    lib_search_paths,
    set_default_tool,
)
from skidl.schematics import debug_anim


def _setup_kicad10():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()


def build():
    """A small single-sheet amplifier: op-amp + feedback R/C + bias divider."""
    ckt = Circuit(name="seedanim")
    with ckt:
        u = Part("Amplifier_Operational", "OPA340NA")
        rf = Part("Device", "R", value="1M")
        cf = Part("Device", "C", value="2p")
        rin = Part("Device", "R", value="50")
        rl = Part("Device", "R", value="1k")
        rb1 = Part("Device", "R", value="10k")
        rb2 = Part("Device", "R", value="10k")
        gnd, vp, sig, bias = Net("GND"), Net("+5V"), Net("SIG"), Net("BIAS")
        gnd.drive = POWER
        vp.drive = POWER
        u["4"] += rf[1], cf[1], rin[2]
        u["1"] += rf[2], cf[2], rl[1]
        u["3"] += bias
        u["2"] += gnd
        u["5"] += vp
        rb1[1] += vp
        rb1[2] += bias
        rb2[1] += bias
        rb2[2] += gnd
        rin[1] += sig
        rl[2] += gnd
    return ckt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".", help="directory for the GIF(s)")
    ap.add_argument("--prefix", default="seedanim", help="GIF filename prefix")
    ap.add_argument("--ms-per-frame", type=int, default=600)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    _setup_kicad10()
    ckt = build()
    render_dir = tempfile.mkdtemp(prefix="seedanim_")
    with debug_anim.record_placements() as rec:
        ckt.generate_schematic(
            tool=KICAD10,
            filepath=render_dir,
            top_name="seedanim",
            seed_placement=True,
            auto_stub=False,
            deconflict_stubs=True,
        )
    paths = debug_anim.render_gifs(
        rec,
        out_dir=args.out,
        prefix=args.prefix,
        resolution=(args.width, args.height),
        ms_per_frame=args.ms_per_frame,
    )
    if paths:
        print("wrote:")
        for p in paths:
            print("  " + p)
    else:
        print("no placements recorded (nothing to animate)")


if __name__ == "__main__":
    main()
