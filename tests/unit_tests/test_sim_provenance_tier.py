# -*- coding: utf-8 -*-

"""Provenance tier for external models: user_lib vs vendor_lib (finding F2).

A hand-authored macromodel attached via an explicit ``Sim.Library`` path outside
the configured corpus must report tier ``user_lib``, distinct from a catalogued
vendor model (``vendor_lib``). Otherwise a behavioral guess is presented with the
same authority as real silicon -- the exact honesty defeat the avalanche E2E hit.
"""

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter  # noqa: F401 (needs PySpice)

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


def _setup():
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    lib_search_paths["kicad10"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _conv():
    from skidl.sim import skidl_flat_view

    return SpiceConverter(skidl_flat_view())


def _write_subckt(dir_path):
    p = dir_path / "avalanche_models.lib"
    p.write_text(
        ".subckt FMMT417_AVAL c b e\n"
        "Bsw c e I = V(c,e)*0.83/(1+exp(-(V(b)-2.5)/0.05))\n"
        ".ends\n"
    )
    return str(p)


# --- pure tier decision (no index needed) ----------------------------------


@requires_sim
def test_external_tier_source_matrix(monkeypatch):
    """Only an out-of-corpus Sim.Library is user_lib; store/index stay vendor_lib."""
    monkeypatch.delenv("SKIDL_SPICE_LIB_PATH", raising=False)
    _setup()
    conv = _conv()
    # sim_library, no corpus configured -> the path can't be under a corpus root
    assert conv._external_tier("sim_library", r"/some/where/mymodels.lib") == "user_lib"
    # store + index resolutions are catalogued vendor models
    assert conv._external_tier("local_store", r"/store/BAT54.lib") == "vendor_lib"
    assert conv._external_tier("library_index", r"/corpus/DIODE2.lib") == "vendor_lib"


# --- end-to-end through convert() ------------------------------------------


@requires_sim
def test_authored_sim_library_reports_user_lib(tmp_path, monkeypatch):
    """A subckt from an explicit Sim.Library outside the corpus -> user_lib."""
    monkeypatch.delenv("SKIDL_SPICE_LIB_PATH", raising=False)
    _setup()
    lib = _write_subckt(tmp_path)
    q = Part("Transistor_BJT", "Q_NPN_BCE", ref="Q2")
    q.Sim_Library = lib
    q.Sim_Name = "FMMT417_AVAL"
    q.Sim_Pins = "2=c 1=b 3=e"
    Net("COL").connect(q[2])
    Net("BAS").connect(q[1])
    Net("EMI").connect(q[3])
    conv = _conv()
    str(conv.convert(strict=False))
    prov = conv.model_provenance["Q2"]
    assert prov.tier == "user_lib", prov
    assert prov.source == "sim_library"


@requires_sim
def test_sim_library_inside_corpus_reports_vendor_lib(tmp_path, monkeypatch):
    """A Sim.Library file that happens to live UNDER the corpus root can't be told
    from a vendor model -> vendor_lib (the documented limitation)."""
    corpus = tmp_path / "KiCad-Spice-Library"
    corpus.mkdir()
    lib = _write_subckt(corpus)
    monkeypatch.setenv("SKIDL_SPICE_LIB_PATH", str(corpus))
    # Force the library index to rebuild against this root.
    import skidl.sim.library_index as li

    li._INDEX_SINGLETON = None
    _setup()
    q = Part("Transistor_BJT", "Q_NPN_BCE", ref="Q2")
    q.Sim_Library = lib
    q.Sim_Name = "FMMT417_AVAL"
    q.Sim_Pins = "2=c 1=b 3=e"
    Net("COL").connect(q[2])
    Net("BAS").connect(q[1])
    Net("EMI").connect(q[3])
    conv = _conv()
    str(conv.convert(strict=False))
    assert conv.model_provenance["Q2"].tier == "vendor_lib"
    li._INDEX_SINGLETON = None  # don't leak the tmp-root index into other tests
