# -*- coding: utf-8 -*-
"""Unit tests for skidl.sim.library_index (no ngspice / corpus needed)."""

import json
import os

import pytest

from skidl.sim.library_index import SpiceLibraryIndex, get_library_index

FIXTURES = os.path.join(os.path.dirname(__file__), "spice_lib_fixtures")


@pytest.fixture
def index():
    # unique cache per run so tests don't collide with a real ~/.skidl cache
    cache = os.path.join(FIXTURES, "_test_index_cache.json")
    if os.path.exists(cache):
        os.remove(cache)
    idx = SpiceLibraryIndex([FIXTURES], cache_path=cache).build(force=True)
    yield idx
    if os.path.exists(cache):
        os.remove(cache)


def test_resolves_model_and_type(index):
    hit = index.resolve("MYD")
    assert hit is not None
    assert hit.kind == "model"
    assert hit.device_type.upper() == "D"


def test_case_insensitive(index):
    assert index.resolve("myd") is not None
    assert index.resolve("MyD").name.upper() == "MYD"


def test_subckt_node_order_recovered(index):
    hit = index.resolve("OPA5")
    assert hit.kind == "subckt"
    # node list stops before PARAMS: and continuation is honored (3 + 2 = 5)
    assert hit.nodes == ["1", "2", "3", "4", "5"]


def test_nested_model_not_indexed(index):
    # DX is defined INSIDE the OPA5 subckt -> must not be a top-level name.
    assert index.resolve("DX") is None


def test_cir_files_ignored(index):
    assert index.resolve("SHOULD_NOT_APPEAR") is None


def test_precedence_manufacturer_wins(index):
    # DUP is defined in both Manufacturer/ (prec 100) and uncategorized/ (prec 20).
    hit = index.resolve("DUP")
    assert "manufacturer" in hit.path.replace("\\", "/").lower()
    assert hit.prec == 100
    # both definitions are retained as alternates
    assert len(index.alternates("DUP")) == 2


def test_header_hint_captured(index):
    hit = index.resolve("DUP")
    assert "connections" in hit.header.lower()


def test_search_and_type_filter(index):
    names = {h.name for h in index.search("my")}
    assert {"MYD", "MYLED"} <= names
    diodes = index.search("", device_types=["D"])
    # A device-type filter classifies only bare .model entries; a subckt carries
    # no device_type and must NOT be excluded (E2E A2 -- real HV MOSFETs ship as
    # subckts). So: every .model hit is type D, and subckts pass through.
    assert all(h.device_type.upper() == "D" for h in diodes if h.kind == "model")
    assert any(h.kind == "model" for h in diodes)


def test_device_type_filter_keeps_subckts(index):
    # A subckt must survive a device-type filter that cannot classify it.
    hits = index.search("opa", device_types=["NMOS", "VDMOS"])
    assert any(h.kind == "subckt" and h.name == "OPA5" for h in hits)


def test_cache_roundtrip_and_determinism(index):
    # Cache written; a second index with the same cache loads it verbatim.
    assert os.path.exists(index.cache_path)
    idx2 = SpiceLibraryIndex([FIXTURES], cache_path=index.cache_path).build()
    assert idx2.resolve("MYD").name == index.resolve("MYD").name
    # Deterministic serialization: rebuild twice -> identical JSON bytes.
    c1 = index.cache_path + ".a"
    c2 = index.cache_path + ".b"
    SpiceLibraryIndex([FIXTURES], cache_path=c1).build(force=True)
    SpiceLibraryIndex([FIXTURES], cache_path=c2).build(force=True)
    with open(c1, "rb") as f1, open(c2, "rb") as f2:
        assert f1.read() == f2.read()
    os.remove(c1)
    os.remove(c2)


def test_inert_without_roots():
    # get_library_index([]) -> None (feature off, no behavior change).
    assert get_library_index([]) is None


# --------------------------------------------------------------------------- #
# Converter integration: the auto-resolve tier (no ngspice run; netlist only)  #
# --------------------------------------------------------------------------- #

try:
    from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool
    from skidl.sim.converter import SpiceConverter

    _HAS_SIM = True
except Exception:
    _HAS_SIM = False

requires_sim = pytest.mark.skipif(not _HAS_SIM, reason="skidl.sim not available")


def _setup_kicad():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _conv():
    from skidl.sim import skidl_flat_view

    return SpiceConverter(skidl_flat_view())


@requires_sim
def test_converter_auto_resolves_model_from_index(monkeypatch):
    monkeypatch.setenv("SKIDL_SPICE_LIB_PATH", FIXTURES)
    monkeypatch.setenv(
        "SKIDL_SPICE_LIB_CACHE", os.path.join(FIXTURES, "_conv_cache.json"))
    try:
        _setup_kicad()
        v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
        r = Part("Device", "R", ref="R1", value="1k")
        d = Part("Device", "D", ref="D1", value="MYD")  # fixture diode, no Sim_Library
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    Net("VIN").connect(v[1], r[1])
    Net("VD").connect(r[2], d["A"])
    Net("0").connect(v[2], d["K"])
    conv = _conv()
    conv.convert(strict=False)  # builds PySpice circuit; does not run ngspice
    prov = conv.model_provenance.get("D1")
    assert prov is not None
    assert prov.tier == "vendor_lib"
    assert prov.source == "library_index"
    cache = os.path.join(FIXTURES, "_conv_cache.json")
    if os.path.exists(cache):
        os.remove(cache)


@requires_sim
def test_subckt_index_hit_without_usable_sim_pins_falls_through(monkeypatch):
    """A same-named corpus .subckt must NOT hijack a generic part out of the box.

    A KiCad diode symbol carries a default ``Sim.Pins="1=K 2=A"`` for the ngspice
    diode PRIMITIVE. If a part's value happens to match a corpus ``.subckt`` (here
    the 3-node ``DUP`` fixture), that symbol-default map names none of the subckt's
    nodes -- so we must fall through to the built-in/generic model rather than emit
    an invalid X-line and hard-fail (env-reinit ISSUE-3). Sim.Prefer="library" is
    the explicit opt-in for the corpus subckt; absent it, generic wins.
    """
    monkeypatch.setenv("SKIDL_SPICE_LIB_PATH", FIXTURES)
    monkeypatch.setenv(
        "SKIDL_SPICE_LIB_CACHE", os.path.join(FIXTURES, "_conv_cache2.json"))
    try:
        _setup_kicad()
        v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
        r = Part("Device", "R", ref="R1", value="1k")
        # value matches the DUP .subckt (3 nodes); only the symbol-default Sim.Pins
        d = Part("Device", "D", ref="D1", value="DUP", Sim_Params="BV=12")
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    Net("VIN").connect(v[1], r[1])
    Net("VD").connect(r[2], d["A"])
    Net("0").connect(v[2], d["K"])
    conv = _conv()
    conv.convert(strict=False)  # must NOT raise SimulationValidationError
    prov = conv.model_provenance.get("D1")
    # Fell through to the built-in generic diode, not the corpus subckt.
    assert prov is None or prov.source != "library_index"
    if prov is not None:
        assert prov.tier == "generic"
    cache = os.path.join(FIXTURES, "_conv_cache2.json")
    if os.path.exists(cache):
        os.remove(cache)


# --------------------------------------------------------------------------- #
# Minimal-deck includes for the auto-resolve tier (netlist strings only)       #
# --------------------------------------------------------------------------- #

def _includes(netlist):
    return [l.strip() for l in str(netlist).splitlines()
            if l.strip().lower().startswith(".include")]


def _index_env(monkeypatch, tmp_path):
    """Point the index at the fixtures, with a throwaway cache + deck dir."""
    monkeypatch.setenv("SKIDL_SPICE_LIB_PATH", FIXTURES)
    monkeypatch.setenv(
        "SKIDL_SPICE_LIB_CACHE", os.path.join(str(tmp_path), "index.json"))
    monkeypatch.setattr(
        SpiceConverter, "_include_cache_dir", staticmethod(lambda: str(tmp_path)))


def _diode_circuit(*values):
    """V1 -- R1 -- diode(s) in parallel to 0; returns the built converter."""
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
    r = Part("Device", "R", ref="R1", value="1k")
    Net("VIN").connect(v[1], r[1])
    vd, gnd = Net("VD"), Net("0")
    gnd.connect(v[2])
    for i, val in enumerate(values, start=1):
        d = Part("Device", "D", ref=f"D{i}", value=val)
        vd.connect(r[2], d["A"])
        gnd.connect(d["K"])
    return _conv()


@requires_sim
def test_index_hit_includes_a_minimal_deck_not_the_whole_file(monkeypatch, tmp_path):
    """The motivating fix: a corpus-resolved model no longer drags its file in."""
    _index_env(monkeypatch, tmp_path)
    try:
        _setup_kicad()
        conv = _diode_circuit("MYD")
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    netlist = str(conv.convert(strict=False))
    incs = _includes(netlist)
    assert len(incs) == 1 and "_mindeck.lib" in incs[0], incs
    deck = open(incs[0].split(None, 1)[1].strip().strip('"'), encoding="ascii").read()
    assert "MYD" in deck
    # the sibling definition in the same file stays out -- if it were malformed,
    # including it would condemn MYD too
    assert "MYLED" not in deck


@requires_sim
def test_two_models_from_one_file_share_one_deck(monkeypatch, tmp_path):
    _index_env(monkeypatch, tmp_path)
    try:
        _setup_kicad()
        conv = _diode_circuit("MYD", "MYLED")
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    incs = _includes(conv.convert(strict=False))
    # one deck, both models -- separate decks would redefine shared helpers
    assert len(incs) == 1, incs
    deck = open(incs[0].split(None, 1)[1].strip().strip('"'), encoding="ascii").read()
    assert "MYD" in deck and "MYLED" in deck


@requires_sim
def test_kill_switch_restores_the_whole_file_include(monkeypatch, tmp_path):
    _index_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SKIDL_SIM_MINIMAL_DECK", "0")
    try:
        _setup_kicad()
        conv = _diode_circuit("MYD")
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    incs = _includes(conv.convert(strict=False))
    assert len(incs) == 1 and "_mindeck" not in incs[0], incs
    assert incs[0].replace("\\", "/").lower().endswith("d.lib")


@requires_sim
def test_explicit_sim_library_still_includes_the_whole_file(monkeypatch, tmp_path):
    """User intent is the escape hatch by construction -- do not second-guess it."""
    _index_env(monkeypatch, tmp_path)
    lib = tmp_path / "explicit.lib"
    lib.write_text(".model EXPD D (IS=1e-14)\n.model OTHER D (IS=2e-14)\n")
    try:
        _setup_kicad()
        v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
        d = Part("Device", "D", ref="D1")
        d.Sim_Library = str(lib)
        d.Sim_Name = "EXPD"
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    Net("VD").connect(v[1], d["A"])
    Net("0").connect(v[2], d["K"])
    incs = _includes(_conv().convert(strict=False))
    assert len(incs) == 1 and "_mindeck" not in incs[0], incs
    assert "explicit.lib" in incs[0]


@requires_sim
def test_explicit_and_index_on_one_file_include_it_only_once(monkeypatch, tmp_path):
    """A deck on top of a whole-file include would redefine the same models."""
    _index_env(monkeypatch, tmp_path)
    try:
        _setup_kicad()
        v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
        d1 = Part("Device", "D", ref="D1", value="MYD")  # auto-resolved
        d2 = Part("Device", "D", ref="D2")               # explicit, same file
        d2.Sim_Library = os.path.join(FIXTURES, "Diode", "d.lib")
        d2.Sim_Name = "MYLED"
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    vd, gnd = Net("VD"), Net("0")
    vd.connect(v[1], d1["A"], d2["A"])
    gnd.connect(v[2], d1["K"], d2["K"])
    incs = _includes(_conv().convert(strict=False))
    assert len(incs) == 1 and "_mindeck" not in incs[0], incs
    assert incs[0].replace("\\", "/").lower().endswith("d.lib")


@requires_sim
def test_extraction_failure_falls_back_to_the_whole_file(monkeypatch, tmp_path):
    """Degrade to today's behavior, never to a silently missing model."""
    _index_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "skidl.sim.model_deck.stage_minimal_deck", lambda *a, **k: None)
    try:
        _setup_kicad()
        conv = _diode_circuit("MYD")
    except Exception:
        pytest.skip("KiCad symbol libs not available on this host")
    incs = _includes(conv.convert(strict=False))
    assert len(incs) == 1 and "_mindeck" not in incs[0], incs
    assert incs[0].replace("\\", "/").lower().endswith("d.lib")
