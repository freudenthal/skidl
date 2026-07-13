# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: power-MOSFET body-diode / Coss companion emission (Stage 26 Phase D).

A Level-1 ``.model`` card cannot express a MOSFET's intrinsic body diode or
output capacitance, so a curated power part (or a ``Sim.Params`` COSS=/BODY=1
override) emits an antiparallel diode + a drain-source cap alongside the ``M``
device. Small-signal / generic MOSFETs get neither.
"""

import pytest

from skidl import KICAD9, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter

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


def _build(symbol=("Transistor_FET", "2N7000"), value=None, sim_params=None, ref="Q1"):
    q = Part(*symbol, ref=ref, value=value)
    if sim_params is not None:
        q.Sim_Params = sim_params
    Net("DRN").connect(q["D"])
    Net("GATE").connect(q["G"])
    Net("SRC").connect(q["S"])
    return q


def _conv():
    from skidl.sim import skidl_flat_view

    return SpiceConverter(skidl_flat_view())


@requires_sim
def test_power_mosfet_emits_body_diode_and_coss():
    _setup()
    _build(symbol=("Transistor_FET", "IRF540N"), value="IRF540N")
    conv = _conv()
    netlist = str(conv.convert(strict=False))
    # body diode: NMOS anode at source, cathode at drain
    assert "DQ1_body SRC DRN DBODYQ1" in netlist, netlist
    assert ".model DBODYQ1 D(IS=1e-09 RS=0.02 CJO=8e-10 BV=100)" in netlist
    # output capacitance across drain-source
    assert "CQ1_oss DRN SRC 2.5e-10" in netlist
    # provenance annotated so the composite is never silent
    assert "+body+coss" in conv.model_provenance["Q1"].name


@requires_sim
def test_small_signal_mosfet_has_no_companions():
    """A 2N7000 (curated but no body-diode/Coss metadata) emits the M device
    only -- no antiparallel diode, no Coss."""
    _setup()
    _build(symbol=("Transistor_FET", "2N7000"), value="2N7000")
    netlist = str(_conv().convert(strict=False))
    assert "_body" not in netlist
    assert "_oss" not in netlist


@requires_sim
def test_default_power_nmos_generic_no_companions():
    """value='powernmos' selects the DefaultPowerNMOS generic (higher VTO/KP)
    but, being a bare generic, carries no body-diode/Coss metadata."""
    _setup()
    _build(value="powernmos")
    conv = _conv()
    netlist = str(conv.convert(strict=False))
    assert "DefaultPowerNMOS" in netlist  # the .model card is emitted
    assert "_body" not in netlist and "_oss" not in netlist


@requires_sim
def test_sim_params_force_body_and_coss():
    """Sim.Params COSS=<F> / BODY=1 add companions to an otherwise plain part."""
    _setup()
    _build(value="nmos", sim_params="COSS=470p BODY=1")
    netlist = str(_conv().convert(strict=False))
    assert "CQ1_oss DRN SRC 4.7e-10" in netlist
    assert "DQ1_body SRC DRN DBODYQ1" in netlist
    assert ".model DBODYQ1 D(" in netlist


@requires_sim
def test_sim_params_coss_overrides_curated_value():
    """An explicit COSS in Sim.Params overrides the curated part's Coss."""
    _setup()
    _build(symbol=("Transistor_FET", "IRLZ44N"), value="IRLZ44N",
           sim_params="COSS=1n")
    netlist = str(_conv().convert(strict=False))
    assert "CQ1_oss DRN SRC 1e-09" in netlist
    # body diode still comes from the curated entry (BV=55 for IRLZ44N)
    assert "BV=55" in netlist
