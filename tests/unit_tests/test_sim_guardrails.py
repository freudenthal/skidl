# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: simulation guardrails / HINTs (DPSG WS5).

* B2 -- the rail-name supply heuristic must NOT re-supply a net that is already
  driven from an explicit source through a series R/L (it would short the series
  element); a genuinely bare rail net still gets its heuristic supply.
* B3 -- an operating-point failure with no DC solution (self-oscillator) gets a
  fix HINT pointing at a seeded stiff transient.
"""

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool
from skidl.sim import skidl_flat_view

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


@requires_sim
def test_series_driven_net_not_re_supplied():
    """A net named VIN behind a series resistor from a VDC gets NO phantom rail."""
    _setup()
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
    r1 = Part("Device", "R", ref="R1", value="10")
    rl = Part("Device", "R", ref="R2", value="1k")
    raw, vin, gnd = Net("RAWIN"), Net("VIN"), Net("GND")
    v[1] += raw
    v[2] += gnd
    r1[1] += raw
    r1[2] += vin
    rl[1] += vin
    rl[2] += gnd
    netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
    # The only voltage source is the explicit VDC; no injected V_supply on VIN.
    assert "V_supply" not in netlist, netlist


@requires_sim
def test_bare_rail_net_still_injects():
    """A genuinely undriven VIN rail (no source, no series path) is still
    supplied -- the guard is targeted, not a blanket disable."""
    _setup()
    r1 = Part("Device", "R", ref="R1", value="1k")
    r2 = Part("Device", "R", ref="R2", value="1k")
    vin, mid, gnd = Net("VIN"), Net("MID"), Net("GND")
    r1[1] += vin
    r1[2] += mid
    r2[1] += mid
    r2[2] += gnd
    netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
    # No explicit source anywhere -> VIN's name-heuristic supply must appear.
    assert "V_supply" in netlist, netlist


def test_no_dc_solution_hint_fires_on_dc_failure():
    from skidl.sim.simulator import _no_dc_solution_hint

    tail = "doAnalyses: No convergence in dc analysis\nrun simulation(s) aborted"
    hint = _no_dc_solution_hint(tail)
    assert "HINT" in hint and "use_initial_condition" in hint
    # A benign tail (no DC-failure signature) yields no hint.
    assert _no_dc_solution_hint("Warning: unrecognized parameter (iave)") == ""
