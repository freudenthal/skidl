# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: the Sim.Pins CROSSED-mapping heuristic guard (HV LLC E2E S1).

``find_spice_model`` used to emit a *positional* Sim_Pins suggestion that is
silently wrong when a symbol's pin numbering differs from the subckt's node
order (the IR2104 case: LO<->VB, HO<->VS swapped). The CLI fix keys named-node
templates by name; this converter-side guard is the backstop -- when an applied
Sim.Pins mapping looks like a positional-paste swap for a self-descriptive
subckt, it logs a WARNING (never errors -- vendor node names can legitimately
differ from a symbol's pin names).
"""

import logging

import pytest

try:
    from skidl.sim.converter import SpiceConverter

    HAS_SIM = True
except Exception:  # noqa: BLE001
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


class _Pin:
    def __init__(self, name):
        self.name = name


class _Comp:
    """Minimal component with a ``_pins`` {num: pin} map (what the guard reads)."""

    def __init__(self, names_by_num):
        self._pins = {num: _Pin(nm) for num, nm in names_by_num.items()}


def _warn(sim_pins, nodes, names_by_num, ref="U1"):
    conv = SpiceConverter.__new__(SpiceConverter)  # skip __init__; method is self-contained
    conv._warn_crossed_sim_pins(ref, _Comp(names_by_num), sim_pins, nodes)


@requires_sim
def test_crossed_named_mapping_warns(caplog):
    names = {"1": "A", "2": "B", "3": "C", "4": "D"}
    nodes = ["A", "B", "C", "D"]
    with caplog.at_level(logging.WARNING, logger="skidl.sim.converter"):
        _warn("1=B 2=A 3=C 4=D", nodes, names)  # A<->B swapped
    assert any("CROSSED" in r.message for r in caplog.records)


@requires_sim
def test_identity_mapping_no_warning(caplog):
    names = {"1": "A", "2": "B", "3": "C", "4": "D"}
    nodes = ["A", "B", "C", "D"]
    with caplog.at_level(logging.WARNING, logger="skidl.sim.converter"):
        _warn("1=A 2=B 3=C 4=D", nodes, names)
    assert not any("CROSSED" in r.message for r in caplog.records)


@requires_sim
def test_numeric_node_names_exempt(caplog):
    # An all-numeric subckt node list has no names to match -> never a "cross".
    names = {"1": "A", "2": "B", "3": "C", "4": "D"}
    nodes = ["1", "2", "3", "4"]
    with caplog.at_level(logging.WARNING, logger="skidl.sim.converter"):
        _warn("1=2 2=1 3=3 4=4", nodes, names)
    assert not any("CROSSED" in r.message for r in caplog.records)


@requires_sim
def test_single_rename_not_flagged(caplog):
    # One pin whose name != its node is a legitimate rename, not a swap: no warn
    # unless >=2 pins are crossed.
    names = {"1": "VCC", "2": "IN", "3": "SD", "4": "GNDPIN"}
    nodes = ["VCC", "IN", "SD", "com"]  # pin4 name GNDPIN != node com, but com
    with caplog.at_level(logging.WARNING, logger="skidl.sim.converter"):
        _warn("1=VCC 2=IN 3=SD 4=com", nodes, names)
    assert not any("CROSSED" in r.message for r in caplog.records)
