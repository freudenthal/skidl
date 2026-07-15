# -*- coding: utf-8 -*-

"""Regression tests for the leaked-placeholder-pin-alias / pickle-cache render
nondeterminism (render-route-preexisting-failures plan, Issue 2).

Background: ``Pin.__init__`` assigns a random placeholder pin number so freshly
created pins are distinct under ``__eq__``. That assignment used to go through the
``num`` setter, which added an alias ``f"p{num}"`` -> ``p<hugerandom>``. The ``num``
deleter then discarded the WRONG key (the bare ``self._num`` instead of
``f"p{self._num}"``), so the random placeholder alias was never removed and leaked
permanently onto every parsed pin -- and got frozen into the ``SchLib`` pickle
cache. A pickle-loaded part therefore carried DIFFERENT random aliases than a
freshly-parsed one, which shifted placement -> wires -> broke render byte
determinism whenever the cache state was mixed (cold vs warm).

The pure-``Pin`` tests below lock the mechanism (no libraries needed); the
real-part tests confirm parsed parts are clean and cold==warm across the pickle
cache. See also ``test_render_determinism.py::test_render_cold_equals_warm`` for
the end-to-end render invariant this subsumes.
"""

import os
import re

import pytest

from skidl import KICAD10, Part, lib_search_paths, set_default_tool
from skidl.pin import Pin

# Matches a leaked placeholder alias: "p" followed by a large integer (the random
# placeholder num is drawn from (MAX_PIN_NUM+1 .. sys.maxsize), so 7+ digits).
_PLACEHOLDER_ALIAS = re.compile(r"p\d{7,}$")


def _placeholder_aliases(pin):
    return [str(a) for a in pin.aliases if _PLACEHOLDER_ALIAS.fullmatch(str(a))]


def _kicad10_symbols_available():
    import skidl.tools.kicad10.lib as k10

    return bool(k10._discover_default_symbol_dirs("10")) or bool(
        os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    )


requires_kicad10 = pytest.mark.skipif(
    not _kicad10_symbols_available(),
    reason="requires real KiCad 10 stock symbol libraries",
)


def _setup_kicad10():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()


# --- pure-Pin mechanism (library-free) -------------------------------------


def test_bare_pin_has_no_placeholder_alias():
    """A freshly constructed Pin must not expose a p<hugerandom> alias (P1)."""
    p = Pin()
    assert _placeholder_aliases(p) == [], (
        f"placeholder alias leaked onto a bare pin: {list(p.aliases)}"
    )
    # The random placeholder num itself is still assigned (for __eq__ distinctness);
    # only its alias side effect is gone.
    assert Pin() != p  # distinct placeholder nums keep bare pins unequal


def test_assigning_num_clears_placeholder_and_prior_aliases():
    """Assigning a real num leaves only p<num>; reassigning removes the old
    p<num> (P2 -- the deleter must discard f"p{num}", not the bare num)."""
    p = Pin()
    p.num = 2
    aliases = {str(a) for a in p.aliases}
    assert "p2" in aliases
    assert _placeholder_aliases(p) == []

    p.num = 3
    aliases = {str(a) for a in p.aliases}
    assert "p3" in aliases
    assert "p2" not in aliases, f"stale p2 alias survived renumber: {aliases}"
    assert _placeholder_aliases(p) == []


# --- net.pins traversal order (build-independent) --------------------------


@requires_kicad10
def test_net_pins_sorted_by_ref_num_name():
    """Net.pins must come out in a stable (ref, num, name) order, not the
    id()-based order of the internal set traversal -- the schematic router
    consumes net.pins IN ORDER, so an id-ordered list makes a cold render (fresh
    part parse) diverge from a warm one (pickle-cache load)."""
    from skidl import Circuit, Net, Part

    _setup_kicad10()
    ckt = Circuit(name="netorder")
    with ckt:
        # Refs out of alphabetical order and pins connected out of order, so an
        # insertion/id ordering would almost certainly NOT already be sorted.
        r_b = Part("Device", "R", value="1k", ref="RB")
        r_a = Part("Device", "R", value="1k", ref="RA")
        c_a = Part("Device", "C", value="1u", ref="CA")
        n = Net("N")
        n += r_b[2], c_a[1], r_a[1]
        n += r_b[1]

    keys = [
        (str(getattr(p.part, "ref", "") or ""), str(p.num), str(p.name))
        for p in n.pins
    ]
    assert keys == sorted(keys), f"net.pins not in stable sorted order: {keys}"


# --- real parsed parts + pickle cache --------------------------------------


def _load_part_with_pickle_dir(pickle_dir):
    """Load Device:R with a controlled pickle_dir and a cleared SchLib cache, so
    an empty dir forces a COLD parse (and writes the pickle) and a populated dir
    forces a WARM pickle load."""
    import skidl
    from skidl.schlib import SchLib

    skidl.config.pickle_dir = str(pickle_dir)
    SchLib._cache.clear()
    return Part("Device", "R")


@requires_kicad10
def test_parsed_part_pins_have_no_placeholder_alias(tmp_path):
    _setup_kicad10()
    part = _load_part_with_pickle_dir(tmp_path / "cold")
    leaked = {
        str(part.ref) + "." + str(pin.num): _placeholder_aliases(pin)
        for pin in part.pins
        if _placeholder_aliases(pin)
    }
    assert leaked == {}, f"placeholder aliases leaked onto parsed pins: {leaked}"


@requires_kicad10
def test_cold_and_warm_part_have_equal_pin_aliases(tmp_path):
    """A part parsed cold (fresh .kicad_sym) and one loaded from the lib pickle
    must carry identical pin alias sets -- the property whose failure drove the
    render nondeterminism."""
    _setup_kicad10()
    pkl = tmp_path / "pkl"  # one shared pickle dir: first load cold, second warm

    cold = _load_part_with_pickle_dir(pkl)
    warm = _load_part_with_pickle_dir(pkl)

    cold_by_num = {str(p.num): {str(a) for a in p.aliases} for p in cold.pins}
    warm_by_num = {str(p.num): {str(a) for a in p.aliases} for p in warm.pins}
    assert cold_by_num == warm_by_num, (
        "cold vs warm pin aliases differ (pickle froze leaked aliases):\n"
        f"cold={cold_by_num}\nwarm={warm_by_num}"
    )
