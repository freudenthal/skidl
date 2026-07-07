# -*- coding: utf-8 -*-

"""Seed-placement benchmark runner (stage 19, Phase A).

White-box replication of the ``kicad9`` schematic orchestrator
(``gen_schematic``) so we can measure placement/routing quality per fixture and
seed WITHOUT writing a full project. Metrics per Stage-19 reference doc section 5:

    hpwl        half-perimeter wirelength over wired internal nets
                (pre-route: bbox of ``pin.pt * part.tx`` per net)
    routed_wl   node.collect_stats() after node.route()  (total Manhattan length)
    wires       count of ``(wire`` tokens in the emitted .kicad_sch
    crossings   intersecting wire-segment pairs not sharing an endpoint (proxy)
    time_s      wall time for place+route
    place_fail  1 if PlacementFailure raised
    route_fail  1 if RoutingFailure raised
    group_sizes real-part counts of each connected group that reached
                place_connected_parts (measured via instrumentation)
    placement_path  per-group "force" | "rowbased" (vs _ROW_PLACE_THRESHOLD)

Usage (run with the .venv-skidl interpreter, KICAD9_SYMBOL_DIR set)::

    python tests/benchmarks/bench_seed_placement.py \
        --mode random --seeds 1 2 3 4 5 --out baseline_random.json

``--mode seed`` is accepted but raises NotImplementedError until Phase C wires it
to ``options["seed_placement"]=True``. Diagnostic overrides ``--row-threshold``
and ``--max-group`` let Phase A time a forced force-directed run on the big
scale fixture (do NOT commit results produced with overrides as the baseline).

This module has no ``test_`` prefix so pytest does not collect it, and it imports
nothing that does not yet exist (no dependency on the Phase-B seed module).
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

# Ensure sibling fixture module is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default KiCad symbol dir (skidl's kicad9 reader consumes KiCad-10 symbols).
os.environ.setdefault(
    "KICAD9_SYMBOL_DIR", r"C:\Program Files\KiCad\10.0\share\kicad\symbols"
)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from seed_fixtures import FIXTURES  # noqa: E402

from skidl import get_default_tool  # noqa: E402
from skidl.geometry import BBox  # noqa: E402
from skidl.schematics.place import (  # noqa: E402
    PlacementFailure,
    is_net_terminal,
)
from skidl.schematics.route import RoutingFailure  # noqa: E402
from skidl.schematics.sch_node import SchNode  # noqa: E402
from skidl.tools import tool_modules  # noqa: E402
from skidl.tools.kicad9.gen_schematic import (  # noqa: E402
    _classify_and_stub_complex_nets,
    auto_stub_nets,
    preprocess_circuit,
)
from skidl.tools.kicad9.sexp_schematic import write_top_schematic  # noqa: E402

# Records (real_count, path) for each connected group placed in the current run.
_GROUP_RECORDS = []


class InstrumentedSchNode(SchNode):
    """SchNode that records the size + placement path of each connected group.

    Overriding only ``place_connected_parts`` captures the force/rowbased
    decision exactly as the placer makes it (after all auto-stub fragmentation),
    which is what answers the ``_ROW_PLACE_THRESHOLD`` scale question.
    """

    def place_connected_parts(node, parts, nets, **options):
        real_count = sum(1 for p in parts if not is_net_terminal(p))
        path = "rowbased" if real_count > node._ROW_PLACE_THRESHOLD else "force"
        if real_count:
            _GROUP_RECORDS.append((real_count, path))
        return super().place_connected_parts(parts, nets, **options)


def _base_options(seed):
    """The forced options gen_schematic sets, plus benchmark controls."""
    return dict(
        use_push_pull=True,
        rotate_parts=True,
        pt_to_pt_mult=5,
        pin_normalize=True,
        auto_stub=True,
        auto_stub_fallback="labels",
        seed=seed,
    )


def _hpwl(node):
    """Sum half-perimeter wirelength over every wired internal net in the tree."""
    total = 0.0
    for child in node.children.values():
        total += _hpwl(child)
    for net in node.get_internal_nets():
        pins = node.get_internal_pins(net)
        if len(pins) < 2:
            continue
        bbox = BBox()
        for pin in pins:
            bbox.add(pin.pt * pin.part.tx)
        total += bbox.w + bbox.h
    return total


def _parse_wire_segments(sch_text):
    """Extract wire segments as ((x1,y1),(x2,y2)) from a .kicad_sch string."""
    import re

    segs = []
    # (wire (pts (xy X Y) (xy X Y)) ...)
    wire_re = re.compile(
        r"\(wire\b.*?\(pts\s*\(xy\s+([-\d.]+)\s+([-\d.]+)\)\s*"
        r"\(xy\s+([-\d.]+)\s+([-\d.]+)\)",
        re.DOTALL,
    )
    for m in wire_re.finditer(sch_text):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        segs.append(((x1, y1), (x2, y2)))
    return segs


def _segments_cross(a, b):
    """True if segments a,b properly intersect and share no endpoint."""
    (ax1, ay1), (ax2, ay2) = a
    (bx1, by1), (bx2, by2) = b
    pts = {(ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2)}
    if len(pts) < 4:
        return False  # shared endpoint

    def ccw(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    d1 = ccw(bx1, by1, bx2, by2, ax1, ay1)
    d2 = ccw(bx1, by1, bx2, by2, ax2, ay2)
    d3 = ccw(ax1, ay1, ax2, ay2, bx1, by1)
    d4 = ccw(ax1, ay1, ax2, ay2, bx2, by2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _count_crossings(segs):
    n = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if _segments_cross(segs[i], segs[j]):
                n += 1
    return n


def run_one(
    fixture_name, mode, seed, row_threshold=None, max_group=None,
    seed_max_fanout=None, seed_row_threshold=None,
):
    """Build + place + route one fixture and return its metrics dict."""
    _GROUP_RECORDS.clear()
    circuit = FIXTURES[fixture_name]()
    options = _base_options(seed)
    if mode == "seed":
        options["seed_placement"] = True
        if seed_max_fanout is not None:
            options["seed_max_fanout"] = seed_max_fanout
        if seed_row_threshold is not None:
            options["seed_row_threshold"] = seed_row_threshold
    if max_group is not None:
        options["auto_stub_max_group"] = max_group

    tool_module = tool_modules[get_default_tool()]

    metrics = dict(
        hpwl=None,
        routed_wl=None,
        wires=None,
        crossings=None,
        time_s=None,
        place_fail=0,
        route_fail=0,
        group_sizes=[],
        placement_path=[],
    )

    # Pre-placement heuristic stubbing (mirrors gen_schematic phase 1).
    auto_stub_nets(circuit, **options)
    preprocess_circuit(circuit, **options)

    node_cls = InstrumentedSchNode
    if row_threshold is not None:
        # Diagnostic: force force-directed on large groups by raising the cutoff.
        node_cls = type(
            "OverrideNode",
            (InstrumentedSchNode,),
            {"_ROW_PLACE_THRESHOLD": row_threshold},
        )

    with tempfile.TemporaryDirectory() as tmp:
        node = node_cls(circuit, tool_module, tmp, fixture_name, fixture_name, 0.0)

        t0 = time.perf_counter()
        try:
            node.place(expansion_factor=1.0, **options)
        except PlacementFailure:
            metrics["place_fail"] = 1
            metrics["time_s"] = round(time.perf_counter() - t0, 4)
            metrics["group_sizes"] = [c for c, _ in _GROUP_RECORDS]
            metrics["placement_path"] = [p for _, p in _GROUP_RECORDS]
            return metrics

        metrics["hpwl"] = round(_hpwl(node), 2)

        # Post-placement complex-net stubbing (mirrors gen_schematic).
        _classify_and_stub_complex_nets(circuit, node, **options)

        try:
            node.route(**options)
        except RoutingFailure:
            metrics["route_fail"] = 1

        metrics["time_s"] = round(time.perf_counter() - t0, 4)

        try:
            metrics["routed_wl"] = float(node.collect_stats().strip())
        except (ValueError, AttributeError):
            metrics["routed_wl"] = None

        # Emit the schematic to count wires + crossings.
        try:
            out_file = write_top_schematic(
                circuit, node, tmp, fixture_name, fixture_name, version=20230409
            )
            with open(out_file, "r", encoding="utf-8") as f:
                text = f.read()
            import re

            metrics["wires"] = len(re.findall(r"\(wire\b", text))
            metrics["crossings"] = _count_crossings(_parse_wire_segments(text))
        except Exception as e:  # noqa: BLE001 - record but don't abort the run
            metrics["wires"] = None
            metrics["crossings"] = None
            metrics["emit_error"] = f"{type(e).__name__}: {e}"

    metrics["group_sizes"] = [c for c, _ in _GROUP_RECORDS]
    metrics["placement_path"] = [p for _, p in _GROUP_RECORDS]
    return metrics


def _aggregate(seed_results):
    """Per-metric median/min/max/spread across seeds for one fixture."""
    agg = {}
    for key in ("hpwl", "routed_wl", "wires", "crossings", "time_s"):
        vals = [
            r[key] for r in seed_results.values() if isinstance(r[key], (int, float))
        ]
        if not vals:
            agg[key] = None
            continue
        agg[key] = dict(
            median=round(statistics.median(vals), 3),
            min=round(min(vals), 3),
            max=round(max(vals), 3),
            spread=round(max(vals) - min(vals), 3),
        )
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("random", "seed"), default="random")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument(
        "--fixtures", nargs="+", default=list(FIXTURES.keys()),
        help="subset of fixtures (default all)",
    )
    ap.add_argument("--out", default=None, help="write JSON results to this path")
    ap.add_argument(
        "--row-threshold", type=int, default=None,
        help="diagnostic: override _ROW_PLACE_THRESHOLD (force force-directed)",
    )
    ap.add_argument(
        "--max-group", type=int, default=None,
        help="diagnostic: override auto_stub_max_group (keep big groups whole)",
    )
    ap.add_argument(
        "--seed-max-fanout", type=int, default=None,
        help="seed mode: max net fanout to keep in the wired graph (default 3)",
    )
    ap.add_argument(
        "--seed-row-threshold", type=int, default=None,
        help="seed mode: force-directed cutoff for seeded groups",
    )
    args = ap.parse_args(argv)

    results = {}
    for fx in args.fixtures:
        results[fx] = {"seeds": {}}
        for seed in args.seeds:
            m = run_one(
                fx, args.mode, seed,
                row_threshold=args.row_threshold, max_group=args.max_group,
                seed_max_fanout=args.seed_max_fanout,
                seed_row_threshold=args.seed_row_threshold,
            )
            results[fx]["seeds"][str(seed)] = m
            print(
                f"{fx:14s} seed={seed} hpwl={m['hpwl']} routed_wl={m['routed_wl']} "
                f"wires={m['wires']} cross={m['crossings']} t={m['time_s']}s "
                f"groups={m['group_sizes']} path={m['placement_path']} "
                f"pf={m['place_fail']} rf={m['route_fail']}"
            )
        results[fx]["summary"] = _aggregate(results[fx]["seeds"])

    payload = {
        "mode": args.mode,
        "seeds": args.seeds,
        "row_threshold_override": args.row_threshold,
        "max_group_override": args.max_group,
        "results": results,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
