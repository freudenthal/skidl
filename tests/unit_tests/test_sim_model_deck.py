# -*- coding: utf-8 -*-
"""Unit tests for skidl.sim.model_deck (pure; no ngspice, no corpus)."""

import os
import time

import pytest

from skidl.sim.model_deck import (
    DECK_VERSION,
    extract_minimal_deck,
    stage_minimal_deck,
)

# A miniature vendor library with the shape that motivates the feature:
# two top-level parts sharing a helper subckt, a top-level .param, and a
# POISON block whose malformed line would condemn the whole file if included.
LIB_TEXT = """\
* miniature vendor library
.param vtemp=27

.subckt SHARED_HELPER a b
Rh a b 1k
.ends

.subckt PARTA 1 2
Xa 1 2 SHARED_HELPER
.ends

.subckt PARTB 1 2
Xb 1 2 SHARED_HELPER
.ends

.model MYZ D (IS=1e-14 BV=5.1)

.subckt POISON 1 2
i source 1 2 bogus-line-that-kills-the-file
.ends
"""


@pytest.fixture
def lib(tmp_path):
    p = tmp_path / "vendor.lib"
    p.write_text(LIB_TEXT, encoding="utf-8")
    return str(p)


def test_single_name_pulls_target_dependency_and_param(lib):
    deck = extract_minimal_deck(lib, ["PARTA"])
    assert deck is not None
    assert ".subckt PARTA" in deck
    assert ".subckt SHARED_HELPER" in deck  # transitive dependency
    assert ".param vtemp=27" in deck


def test_absent_name_returns_none(lib):
    # None means "fall back to the whole-file .include" -- never a silent deck.
    assert extract_minimal_deck(lib, ["NO_SUCH_MODEL"]) is None
    # one bad name in a set invalidates the whole request
    assert extract_minimal_deck(lib, ["PARTA", "NO_SUCH_MODEL"]) is None


def test_multi_name_union_emits_shared_helper_once(lib):
    deck = extract_minimal_deck(lib, ["PARTA", "PARTB"])
    assert ".subckt PARTA" in deck and ".subckt PARTB" in deck
    # a redefined subckt is an ngspice error -> exactly one definition
    assert deck.count(".subckt SHARED_HELPER") == 1


def test_poisoned_sibling_is_not_in_the_deck(lib):
    deck = extract_minimal_deck(lib, ["MYZ"])
    assert ".model MYZ" in deck
    assert "POISON" not in deck
    assert "bogus-line-that-kills-the-file" not in deck


def test_non_ascii_is_sanitized(tmp_path):
    p = tmp_path / "accents.lib"
    p.write_text(".model MÜD D (IS=1e-14)\n* résistance\n", encoding="utf-8")
    deck = extract_minimal_deck(str(p), ["MÜD"])
    assert deck is not None
    deck.encode("ascii")  # would raise if a non-ASCII byte survived


def test_stage_is_deterministic_and_named_for_the_source(lib, tmp_path):
    cache = str(tmp_path / "cache")
    a = stage_minimal_deck(lib, ["PARTA"], cache)
    b = stage_minimal_deck(lib, ["PARTA"], cache)
    assert a == b and a.endswith("_mindeck.lib")
    assert "vendor" in os.path.basename(a)
    with open(a, "r", encoding="ascii") as fh:
        assert ".subckt PARTA" in fh.read()
    # name order must not change the staged identity
    assert (stage_minimal_deck(lib, ["PARTA", "PARTB"], cache)
            == stage_minimal_deck(lib, ["PARTB", "PARTA"], cache))
    # different requests get different files
    assert stage_minimal_deck(lib, ["PARTB"], cache) != a


def test_stage_restages_when_the_source_is_newer(lib, tmp_path):
    cache = str(tmp_path / "cache")
    staged = stage_minimal_deck(lib, ["PARTA"], cache)
    assert "RENAMED" not in open(staged, encoding="ascii").read()
    with open(lib, "w", encoding="utf-8") as fh:
        fh.write(LIB_TEXT.replace("Rh a b 1k", "Rh a b 2k ; RENAMED"))
    os.utime(lib, (time.time() + 10, time.time() + 10))
    again = stage_minimal_deck(lib, ["PARTA"], cache)
    assert again == staged
    assert "RENAMED" in open(staged, encoding="ascii").read()


def test_stage_returns_none_for_an_absent_name(lib, tmp_path):
    assert stage_minimal_deck(lib, ["NO_SUCH_MODEL"], str(tmp_path / "c")) is None
    assert stage_minimal_deck(lib, [], str(tmp_path / "c")) is None


def test_deck_version_participates_in_the_digest(lib, tmp_path, monkeypatch):
    """A future extractor fix must not reuse decks staged by the old one."""
    cache = str(tmp_path / "cache")
    before = stage_minimal_deck(lib, ["PARTA"], cache)
    monkeypatch.setattr("skidl.sim.model_deck.DECK_VERSION", DECK_VERSION + 1)
    assert stage_minimal_deck(lib, ["PARTA"], cache) != before
