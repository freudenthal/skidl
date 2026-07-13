# -*- coding: utf-8 -*-

"""Regression: the render must never emit two elements with the same UUID.

Element UUIDs are deterministic uuid5(kind:net:coords) hashes, so two coincident
same-net stub wires (or a stub end labelled as both an island anchor and a leaf)
produce byte-identical duplicate elements sharing a UUID. KiCad tolerates the
visual redundancy, but any consumer that keys a unique index on uuid
(kicad-sch-api, and thus the skidl-eda ERC autofix) raises "Duplicate key ... in
unique index 'uuid'". ``_dedupe_elements_by_uuid`` collapses them at write time.
"""

from skidl.tools.kicad10.sexp_schematic import (
    Sexp,
    _dedupe_elements_by_uuid,
    _top_level_uuid,
)


def _wire(x1, y1, x2, y2, u):
    return Sexp(
        [
            "wire",
            ["pts", ["xy", x1, y1], ["xy", x2, y2]],
            ["stroke", ["width", 0], ["type", "default"]],
            ["uuid", u],
        ]
    )


def _label(name, x, y, u):
    return Sexp(["label", f'"{name}"', ["at", x, y, 0], ["uuid", u]])


def test_top_level_uuid_reads_direct_child_only():
    assert _top_level_uuid(_wire(0, 0, 1, 1, "abc")) == "abc"
    # No uuid child -> None.
    assert _top_level_uuid(Sexp(["junction", ["at", 1, 2]])) is None
    # Non-element inputs are tolerated.
    assert _top_level_uuid("not-an-element") is None
    assert _top_level_uuid([]) is None


def test_dedupe_drops_only_repeated_uuids():
    a = _wire(1, 1, 1, 2, "u-dup")
    b = _wire(1, 1, 1, 2, "u-dup")  # coincident same-net stub drawn twice
    c = _label("NET", 3, 4, "u-lbl")
    d = _label("NET", 3, 4, "u-lbl")  # anchor + leaf label on the same end
    e = _wire(9, 9, 9, 8, "u-uniq")
    out = _dedupe_elements_by_uuid([a, b, c, d, e])
    # Exactly one of each duplicate survives; the unique one is untouched.
    assert out == [a, c, e]
    uuids = [_top_level_uuid(x) for x in out]
    assert len(uuids) == len(set(uuids)), uuids


def test_dedupe_keeps_all_uuidless_elements():
    j1 = Sexp(["junction", ["at", 1, 2]])
    j2 = Sexp(["junction", ["at", 3, 4]])
    w = _wire(0, 0, 1, 0, "only")
    out = _dedupe_elements_by_uuid([j1, j2, w, j1])
    # uuid-less elements are always kept (even the repeated j1 object); only
    # uuid-bearing repeats are collapsed.
    assert out == [j1, j2, w, j1]
