# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: a mismatched external-subckt Sim.Pins raises a clear error (M2).

A wrong/short ``Sim.Pins`` mapping on a ``.subckt`` device previously emitted an
``X`` line with the wrong node count, then ngspice died later with a cryptic
"Too few parameters for subcircuit" and "there aren't any circuits loaded". The
converter now fails at build time with a message naming the symbol's pins and
the subckt's node order (E2E finding M2).
"""

import os

import pytest

from skidl import KICAD9, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter, SimulationValidationError

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


def _setup():
    set_default_tool(KICAD9)
    from skidl.tools.kicad9.lib import default_lib_paths

    lib_search_paths["kicad9"] = ["."] + default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


def _conv():
    from skidl.sim import skidl_flat_view

    return SpiceConverter(skidl_flat_view())


def _write_subckt(tmp_path):
    p = tmp_path / "q.lib"
    p.write_text(
        ".subckt IRFP260 10 20 40\n"
        "M1 10 20 40 40 nmosmod\n"
        ".model nmosmod NMOS\n"
        ".ends\n"
    )
    return str(p)


@requires_sim
def test_wrong_sim_pins_raises_with_pin_and_node_lists(tmp_path):
    _setup()
    lib = _write_subckt(tmp_path)
    q = Part("Transistor_FET", "IRF540N", ref="Q1")
    q.Sim_Library = lib
    q.Sim_Name = "IRFP260"
    q.Sim_Pins = "<pin1>=1 <pin2>=2 <pin3>=3"  # placeholder keys, unmatched
    Net("DRN").connect(q["D"])
    Net("GATE").connect(q["G"])
    Net("SRC").connect(q["S"])
    with pytest.raises(SimulationValidationError) as exc:
        str(_conv().convert(strict=False))
    msg = str(exc.value)
    # names the symbol's real pins (number/name) ...
    assert "1/G" in msg and "2/D" in msg and "3/S" in msg, msg
    # ... and the subckt's node order
    assert "10 20 40" in msg, msg


@requires_sim
def test_correct_sim_pins_emits_x_line(tmp_path):
    """A correct Sim.Pins mapping emits an X line with all subckt nodes."""
    _setup()
    lib = _write_subckt(tmp_path)
    q = Part("Transistor_FET", "IRF540N", ref="Q1")
    q.Sim_Library = lib
    q.Sim_Name = "IRFP260"
    q.Sim_Pins = "1=10 2=20 3=40"  # symbol pin NUMBER -> subckt node
    Net("DRN").connect(q["D"])
    Net("GATE").connect(q["G"])
    Net("SRC").connect(q["S"])
    netlist = str(_conv().convert(strict=False))
    # all three subckt nodes present -> a full X line, no raise
    assert "XQ1 GATE DRN SRC IRFP260" in netlist, netlist
