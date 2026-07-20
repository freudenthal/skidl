# -*- coding: utf-8 -*-

"""Subckt terminal-mapping confidence in provenance (finding F3).

A 3-node transistor subckt with numeric nodes (``10 20 30``) has no self-evident
D/G/S identity -- a wrong Sim.Pins map gives a converged-but-wrong result with no
error. The converter now records ``pin_map_confidence="heuristic"`` on such an
attach and warns, pointing at ``--verify-terminals``; a named-node subckt is
``"named"``.
"""

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter

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


@requires_sim
@pytest.mark.parametrize(
    "kind,nodes,expected",
    [
        ("mosfet", ["10", "20", "30"], "heuristic"),
        ("bjt", ["1", "2", "3"], "heuristic"),
        ("mosfet", ["D", "G", "S"], "named"),
        ("mosfet", ["10", "20", "30", "40"], ""),   # 4-node: not a 3-terminal case
        ("opamp", ["1", "2", "3"], ""),             # not a transistor
        ("diode", ["1", "2"], ""),
    ],
)
def test_pin_map_confidence_classification(kind, nodes, expected):
    assert SpiceConverter._pin_map_confidence(kind, nodes) == expected


@requires_sim
def test_numeric_mosfet_subckt_marked_heuristic(tmp_path):
    _setup()
    lib = tmp_path / "q.lib"
    lib.write_text(
        ".subckt IRFTEST 10 20 30\n"
        "M1 10 20 30 30 nmosmod\n"
        ".model nmosmod NMOS (VTO=3.3 KP=4)\n"
        ".ends\n"
    )
    q = Part("Transistor_FET", "IRF540N", ref="Q1")
    q.Sim_Library = str(lib)
    q.Sim_Name = "IRFTEST"
    q.Sim_Pins = "1=20 2=10 3=30"  # symbol pin -> subckt node
    Net("DRN").connect(q["D"])
    Net("GATE").connect(q["G"])
    Net("SRC").connect(q["S"])
    conv = _conv()
    str(conv.convert(strict=False))
    assert conv.model_provenance["Q1"].pin_map_confidence == "heuristic"


@requires_sim
def test_named_mosfet_subckt_marked_named(tmp_path):
    _setup()
    lib = tmp_path / "qn.lib"
    lib.write_text(
        ".subckt NAMEDFET d g s\n"
        "M1 d g s s nmosmod\n"
        ".model nmosmod NMOS (VTO=3.3 KP=4)\n"
        ".ends\n"
    )
    q = Part("Transistor_FET", "IRF540N", ref="Q1")
    q.Sim_Library = str(lib)
    q.Sim_Name = "NAMEDFET"
    q.Sim_Pins = "1=g 2=d 3=s"
    Net("DRN").connect(q["D"])
    Net("GATE").connect(q["G"])
    Net("SRC").connect(q["S"])
    conv = _conv()
    str(conv.convert(strict=False))
    assert conv.model_provenance["Q1"].pin_map_confidence == "named"
