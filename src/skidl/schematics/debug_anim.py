# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Debug animation of the constructive seed placer (optional, Pillow-only).

Renders a slow GIF that shows the schematic placer building a layout ONE PART PER
FRAME: each part is drawn as its placement bounding box with dots on the L/R/U/D
edges for pins, an arrow marking the part's local "up" direction after rotation, a
distinct light fill (a new colour per part), and a centred ``ref`` + placement-order
label, on a white background. One GIF is produced per hierarchical sheet.

This is a DEBUG tool and is kept strictly optional:

* the core placer (:mod:`skidl.schematics.seed_place`) only gains a None-by-default
  observer hook -- importing skidl / running a normal render never touches this
  module;
* Pillow (``PIL``) is imported LAZILY inside :func:`render_gifs` only, so recording
  placements needs no third-party dependency; only the GIF render does. Install it
  with ``pip install pillow`` or ``pip install .[debug]``.

Typical use::

    from skidl.schematics import debug_anim
    with debug_anim.record_placements() as rec:
        ...run a schematic generation (seed/constructive placement)...
    debug_anim.render_gifs(rec, out_dir=".", prefix="dpsg")

The placer sets each ``part.tx`` exactly once under the constructive/relax path and
never moves it again, so the cumulative reveal is faithful to the real layout.
"""

import contextlib

from skidl.geometry import Vector

from . import seed_place

__all__ = ["PlacementRecorder", "record_placements", "render_gifs"]


# Light, high-lightness fills that stay readable (black text/border) on white and
# are visually distinct. Cycled by placement order -> "a new part gets a new colour".
_PALETTE = [
    (255, 179, 186), (255, 223, 186), (255, 255, 186), (186, 255, 201),
    (186, 225, 255), (215, 189, 255), (255, 198, 236), (200, 240, 240),
    (230, 210, 180), (210, 230, 190), (190, 210, 235), (235, 205, 205),
]


def _sheet_key(part):
    """Hierarchy tuple identifying the part's sheet (subcircuit), robust to setup."""
    try:
        ht = tuple(part.hiertuple)
    except Exception:  # noqa: BLE001 - part.node may be unset for bare/fake parts
        ht = ()
    return ht or ("top",)


def _sheet_name(sheet_key):
    """Human sheet name = deepest hierarchy level ('top' for the root sheet)."""
    if len(sheet_key) <= 1:
        return "top"
    return str(sheet_key[-1])


def _project_to_rect_edge(px, py, x0, y0, x1, y1):
    """Snap a point onto the nearest edge of the rect so a pin dot sits ON the box."""
    # Clamp inside first, then push to whichever of the 4 edges is closest.
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    d_left, d_right = cx - x0, x1 - cx
    d_top, d_bot = cy - y0, y1 - cy
    m = min(d_left, d_right, d_top, d_bot)
    if m == d_left:
        return x0, cy
    if m == d_right:
        return x1, cy
    if m == d_top:
        return cx, y0
    return cx, y1


class PlacementRecorder:
    """Placement observer: captures a frozen geometry snapshot per placed part.

    Grouped by sheet (hierarchy tuple). Each sheet keeps parts in placement order;
    a part is recorded once per sheet (first placement wins, so a re-placed sheet
    does not double its frames). Holds only plain numbers -- no live skidl objects
    and no Pillow dependency.
    """

    def __init__(self):
        # sheet_key -> list of record dicts (in placement order)
        self.sheets = {}
        # sheet_key -> set of already-seen part ids (dedup within a sheet)
        self._seen = {}
        # first-appearance order of sheets, for stable sheet numbering
        self.sheet_order = []

    def __call__(self, part):
        key = _sheet_key(part)
        if key not in self.sheets:
            self.sheets[key] = []
            self._seen[key] = set()
            self.sheet_order.append(key)
        seen = self._seen[key]
        pid = id(part)
        if pid in seen:
            return  # re-placed part: keep the first frame for this sheet
        seen.add(pid)
        self.sheets[key].append(self._snapshot(part, order=len(self.sheets[key])))

    @staticmethod
    def _snapshot(part, order):
        tx = part.tx
        wb = part.place_bbox * tx
        x0, y0 = float(wb.min.x), float(wb.min.y)
        x1, y1 = float(wb.max.x), float(wb.max.y)

        pins = []
        for pin in getattr(part, "pins", []) or []:
            pt = getattr(pin, "pt", None)
            if pt is None:
                continue
            wp = pt * tx
            ex, ey = _project_to_rect_edge(float(wp.x), float(wp.y), x0, y0, x1, y1)
            pins.append((ex, ey))

        # Local "up" (KiCad Y-down: up = -y) rotated into the world frame -> the
        # direction the part's top points after its placement rotation.
        up = Vector(0, -1) * tx.no_translate()
        umag = (up.x ** 2 + up.y ** 2) ** 0.5 or 1.0
        up_world = (up.x / umag, up.y / umag)

        return {
            "ref": str(getattr(part, "ref", "?") or "?"),
            "order": order,
            "box": (x0, y0, x1, y1),
            "pins": pins,
            "up": up_world,
        }


@contextlib.contextmanager
def record_placements():
    """Install a :class:`PlacementRecorder` on the seed placer for the block's body.

    Restores the previous observer on exit. Yields the recorder so the caller can
    pass it to :func:`render_gifs`.
    """
    rec = PlacementRecorder()
    prev = seed_place._PLACEMENT_OBSERVER
    seed_place._PLACEMENT_OBSERVER = rec
    try:
        yield rec
    finally:
        seed_place._PLACEMENT_OBSERVER = prev


# --------------------------------------------------------------------------- #
# Rendering (Pillow only, lazy import)
# --------------------------------------------------------------------------- #


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401

        return Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "debug_anim.render_gifs needs Pillow. Install it with "
            "`pip install pillow` (or `pip install .[debug]`)."
        ) from exc


def _fit_transform(records, resolution, pad_frac):
    """Uniform world->pixel transform letterboxing the records' bbox into the canvas."""
    W, H = resolution
    xs0 = min(r["box"][0] for r in records)
    ys0 = min(r["box"][1] for r in records)
    xs1 = max(r["box"][2] for r in records)
    ys1 = max(r["box"][3] for r in records)
    cw = (xs1 - xs0) or 1.0
    ch = (ys1 - ys0) or 1.0
    avail_w = W * (1 - 2 * pad_frac)
    avail_h = H * (1 - 2 * pad_frac)
    s = min(avail_w / cw, avail_h / ch)
    ox = (W - cw * s) / 2 - xs0 * s
    oy = (H - ch * s) / 2 - ys0 * s

    def to_px(wx, wy):
        return (wx * s + ox, wy * s + oy)

    return to_px, s


def _draw_frame(Image, ImageDraw, ImageFont, records, k, resolution, pad_frac, font):
    """Render frame k (parts 0..k of one sheet) to an RGB PIL image."""
    W, H = resolution
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    shown = records[: k + 1]
    # Running fit: the view grows to hold the parts placed so far (fixed canvas).
    to_px, _s = _fit_transform(shown, resolution, pad_frac)

    for rec in shown:
        color = _PALETTE[rec["order"] % len(_PALETTE)]
        x0, y0, x1, y1 = rec["box"]
        p0 = to_px(x0, y0)
        p1 = to_px(x1, y1)
        rx0, ry0 = min(p0[0], p1[0]), min(p0[1], p1[1])
        rx1, ry1 = max(p0[0], p1[0]), max(p0[1], p1[1])
        d.rectangle([rx0, ry0, rx1, ry1], fill=color, outline=(40, 40, 40), width=2)

        # Pin dots on the box edges.
        for (ex, ey) in rec["pins"]:
            px, py = to_px(ex, ey)
            r = 4
            d.ellipse([px - r, py - r, px + r, py + r],
                      fill=(30, 30, 30), outline=(30, 30, 30))

        # "Up" arrow, drawn in the part's UP half (offset from centre) so it marks
        # the post-rotation "up" direction without overlapping the centred label.
        # world "up" is (ux,uy); pixels are y-down like the world map, so use it
        # directly. half-extent along the up axis (arrow lives between center and
        # the up edge).
        cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        ux, uy = rec["up"]
        half_up = abs(ux) * (rx1 - rx0) / 2 + abs(uy) * (ry1 - ry0) / 2
        tail = (cx + ux * half_up * 0.30, cy + uy * half_up * 0.30)
        tip = (cx + ux * max(half_up * 0.90, 14), cy + uy * max(half_up * 0.90, 14))
        d.line([tail[0], tail[1], tip[0], tip[1]], fill=(0, 0, 160), width=3)
        _arrow_head(d, tail[0], tail[1], tip[0], tip[1], (0, 0, 160))

        # Centred ref + placement order.
        label = "{}\n#{}".format(rec["ref"], rec["order"])
        try:
            d.multiline_text((cx, cy), label, fill=(0, 0, 0), font=font,
                             anchor="mm", align="center", spacing=2)
        except TypeError:  # very old Pillow without anchor= support
            d.multiline_text((cx - 12, cy - 10), label, fill=(0, 0, 0),
                             font=font, align="center", spacing=2)

    return img


def _arrow_head(d, x0, y0, x1, y1, color, size=9):
    """Draw a small triangular arrowhead at (x1,y1) pointing away from (x0,y0)."""
    import math

    ang = math.atan2(y1 - y0, x1 - x0)
    for da in (math.radians(150), math.radians(-150)):
        hx = x1 + size * math.cos(ang + da)
        hy = y1 + size * math.sin(ang + da)
        d.line([x1, y1, hx, hy], fill=color, width=3)


def render_gifs(
    recorder,
    out_dir=".",
    prefix="placement",
    *,
    resolution=(1280, 720),
    ms_per_frame=600,
    hold_last_ms=2500,
    pad_frac=0.07,
):
    """Render one GIF per recorded sheet into ``out_dir``.

    Filenames: ``{prefix}_sheet{NN}_{sheetname}.gif`` (NN = 2-digit first-appearance
    sheet number). Returns the list of written paths. Requires Pillow.
    """
    import os

    Image, ImageDraw, ImageFont = _require_pillow()
    try:
        font = ImageFont.load_default(size=15)  # Pillow >= 10.1 sizes the default
    except TypeError:  # older Pillow: fixed-size bitmap default
        font = ImageFont.load_default()
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for idx, key in enumerate(recorder.sheet_order):
        records = recorder.sheets[key]
        if not records:
            continue
        name = _sheet_name(key)
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
        fname = "{}_sheet{:02d}_{}.gif".format(prefix, idx, safe)
        path = os.path.join(out_dir, fname)

        frames = [
            _draw_frame(Image, ImageDraw, ImageFont, records, k, resolution,
                        pad_frac, font)
            for k in range(len(records))
        ]
        durations = [ms_per_frame] * (len(frames) - 1) + [hold_last_ms]
        frames[0].save(
            path, save_all=True, append_images=frames[1:],
            duration=durations, loop=0, optimize=False,
        )
        written.append(path)
    return written
