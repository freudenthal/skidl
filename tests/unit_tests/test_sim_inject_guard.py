# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: the net-name supply-injection heuristic's driven-net guard (E2E D1).

``_add_power_sources`` injects a heuristic supply on a net whose *name* reads as
a rail (``VIN``/``VCC``/``+5V``...). E2E finding D1: a net named ``VIN_SS`` was
already driven by an op-amp OUTPUT, and stacking a phantom 5 V rail on it
corrupted the whole VCO. The guard skips injection when the net already carries
an ``OUTPUT``/``PWROUT`` pin, while still injecting on genuinely undriven rails
and still respecting the whole-token match (``VINT_RAW`` is not a ``VIN`` rail).
"""

import pytest

from skidl import Circuit, Net, Part, lib_search_paths, set_default_tool, KICAD10

try:
    from skidl.sim.converter import SpiceConverter  # noqa: F401

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


def _setup():
    from skidl_eda import setup_kicad10

    try:
        setup_kicad10()
    except Exception:  # noqa: BLE001
        set_default_tool(KICAD10)
        from skidl.tools.kicad10.lib import default_lib_paths

        lib_search_paths["kicad10"] = ["."] + default_lib_paths()


def _emit(ckt):
    from skidl.sim.adapter import skidl_flat_view

    return str(SpiceConverter(skidl_flat_view(ckt)).convert(strict=False))


def _supply_lines(txt):
    return [ln.strip() for ln in txt.splitlines() if "supply" in ln.lower()]


@requires_sim
def test_no_injection_on_opamp_driven_named_rail():
    """A ``VIN_SS`` net driven by an op-amp OUTPUT gets NO heuristic supply."""
    _setup()
    ckt = Circuit(name="driven")
    with ckt:
        op = Part("Amplifier_Operational", "TL071", ref="U1")
        vss = Net("VIN_SS"); inn = Net("INN"); gnd = Net("GND")
        vp = Net("VP"); vn = Net("VN")
        op[6] += vss; op[2] += inn; op[3] += gnd; op[7] += vp; op[4] += vn
        Part("Device", "R", ref="R1", value="10k")[1, 2] += vss, inn
    assert _supply_lines(_emit(ckt)) == []


@requires_sim
def test_injection_still_fires_on_bare_undriven_rail():
    """A bare ``VIN`` divider with no driver still gets the 5 V supply."""
    _setup()
    ckt = Circuit(name="bare")
    with ckt:
        vin = Net("VIN"); gnd = Net("GND"); mid = Net("MID")
        Part("Device", "R", ref="R1", value="10k")[1, 2] += vin, mid
        Part("Device", "R", ref="R2", value="10k")[1, 2] += mid, gnd
    lines = _supply_lines(_emit(ckt))
    assert any("VIN" in ln and "5.0" in ln for ln in lines)


@requires_sim
def test_vint_raw_token_still_not_injected():
    """``VINT_RAW`` tokenizes to {VINT, RAW}: not a ``VIN`` rail (bug #13)."""
    _setup()
    ckt = Circuit(name="token")
    with ckt:
        vraw = Net("VINT_RAW"); gnd = Net("GND"); mid = Net("MID")
        Part("Device", "R", ref="R1", value="10k")[1, 2] += vraw, mid
        Part("Device", "R", ref="R2", value="10k")[1, 2] += mid, gnd
    assert _supply_lines(_emit(ckt)) == []
