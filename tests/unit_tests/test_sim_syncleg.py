# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: duty+phase synchronous-leg primitive + multi-node resolver (Stage 27.5).

``_emit_sync_leg`` generalizes ``_add_halfbridge``'s fixed-50 %/zero-phase
machinery to an arbitrary per-leg duty and inter-leg phase; at duty=0.5/phase=0
between VIN and GND it must be **byte-identical** to the legacy half-bridge.
``_multiswitch_terminals`` resolves the two-switch-node / negative-output pin
maps the Stage-27 bidirectional families need. Both are pure (no ngspice); the
tests drive the helpers directly on a bare converter.
"""

from types import SimpleNamespace

import pytest

try:
    from skidl.sim.converter import SpiceConverter

    HAS_SIM = True
except Exception:
    HAS_SIM = False

requires_sim = pytest.mark.skipif(
    not HAS_SIM, reason="PySpice (skidl.sim SPICE stack) not installed"
)


def _conv():
    """A bare converter with just the state ``_emit_sync_leg`` writes to."""
    conv = SpiceConverter(None)
    conv.spice_circuit = SimpleNamespace(raw_spice="")
    conv.node_map = {}
    return conv


def _pin(name, netname):
    return SimpleNamespace(name=name, net=SimpleNamespace(name=netname))


def _component(*name_net_pairs):
    return SimpleNamespace(
        _pins={i: _pin(nm, nt) for i, (nm, nt) in enumerate(name_net_pairs)}
    )


# --- _emit_sync_leg -------------------------------------------------------- #

# The exact half-bridge block for fsw=100k, duty=0.5, phase=0, dt=100n, ron=0.1
# between VINN and GND (gates referenced to GND). This is the byte-for-byte gate:
# it must match what _add_halfbridge emits (test_sim_halfbridge.py asserts the
# same lines from the full build path).
HALFBRIDGE_GOLDEN = (
    "VU1_ghs U1_ghs GND PULSE(0 5 0 5e-08 5e-08 4.9e-06 1e-05)\n"
    "VU1_gls U1_gls GND PULSE(0 5 5e-06 5e-08 5e-08 4.9e-06 1e-05)\n"
    "SU1_hs VINN SWN U1_ghs GND SWU1\n"
    "SU1_ls SWN GND U1_gls GND SWU1\n"
    ".model SWU1 SW(Ron=0.1 Roff=1e6 Vt=2.5 Vh=0.2)\n"
    "DU1_hs SWN VINN DFWU1\n"
    "DU1_ls GND SWN DFWU1\n"
    ".model DFWU1 D(IS=1e-9 N=1.05 CJO=100p)"
)


@requires_sim
def test_sync_leg_reproduces_halfbridge_byte_for_byte():
    conv = _conv()
    ok = conv._emit_sync_leg(
        "VINN",
        "SWN",
        "GND",
        "GND",
        fsw=100e3,
        duty=0.5,
        phase=0.0,
        dt=100e-9,
        ron=0.1,
        suffix="U1",
    )
    assert ok is True
    # raw_spice is seeded with a leading "\n" + the joined lines.
    assert conv.spice_circuit.raw_spice == "\n" + HALFBRIDGE_GOLDEN


@requires_sim
def test_sync_leg_duty_and_phase_change_pulse_numbers():
    conv = _conv()
    conv._emit_sync_leg(
        "TOP",
        "SW",
        "BOT",
        "GND",
        fsw=100e3,
        duty=0.25,
        phase=0.5,
        dt=100e-9,
        ron=0.1,
        suffix="U1",
    )
    txt = conv.spice_circuit.raw_spice
    lines = txt.strip().splitlines()
    ghs = next(ln for ln in lines if ln.startswith("VU1_ghs"))
    gls = next(ln for ln in lines if ln.startswith("VU1_gls"))
    # per = 1e-5. high-side: td = phase*per = 5e-6, on = duty*per - dt = 2.4e-6.
    assert ghs == "VU1_ghs U1_ghs GND PULSE(0 5 5e-06 5e-08 5e-08 2.4e-06 1e-05)", ghs
    # low-side: td = (phase+duty)*per = 7.5e-6, on = (1-duty)*per - dt = 7.4e-6.
    assert gls == "VU1_gls U1_gls GND PULSE(0 5 7.5e-06 5e-08 5e-08 7.4e-06 1e-05)", gls
    # switches straddle top/sw and sw/bottom, gates referenced to gnd.
    assert "SU1_hs TOP SW U1_ghs GND SWU1" in txt
    assert "SU1_ls SW BOT U1_gls GND SWU1" in txt
    # antiparallel diodes sw->top and bottom->sw.
    assert "DU1_hs SW TOP DFWU1" in txt
    assert "DU1_ls BOT SW DFWU1" in txt


@requires_sim
def test_sync_leg_custom_ron_formats():
    conv = _conv()
    conv._emit_sync_leg(
        "T", "S", "B", "GND",
        fsw=200e3, duty=0.5, phase=0.0, dt=200e-9, ron=0.25, suffix="U1",
    )
    assert ".model SWU1 SW(Ron=0.25 Roff=1e6 Vt=2.5 Vh=0.2)" in conv.spice_circuit.raw_spice


@requires_sim
def test_sync_leg_deadtime_too_large_emits_nothing():
    conv = _conv()
    # duty=0.5, fsw=100k -> min(duty,1-duty)/fsw = 5e-6; dt=6us leaves no window.
    ok = conv._emit_sync_leg(
        "T", "S", "B", "GND",
        fsw=100e3, duty=0.5, phase=0.0, dt=6e-6, ron=0.1, suffix="U1",
    )
    assert ok is False
    assert conv.spice_circuit.raw_spice == ""


@requires_sim
def test_sync_leg_deadtime_too_large_on_skewed_duty():
    conv = _conv()
    # duty=0.1 -> min window = 0.1*per = 1e-6; dt=1.5us kills the high-side leg.
    ok = conv._emit_sync_leg(
        "T", "S", "B", "GND",
        fsw=100e3, duty=0.1, phase=0.0, dt=1.5e-6, ron=0.1, suffix="U1",
    )
    assert ok is False
    assert conv.spice_circuit.raw_spice == ""


# --- _multiswitch_terminals ------------------------------------------------ #


@requires_sim
def test_multiswitch_buckboost4_resolves_five_nodes():
    conv = _conv()
    comp = _component(
        ("VIN", "vin_net"),
        ("SW", "swa_net"),
        ("SW2", "swb_net"),
        ("VOUT", "vout_net"),
        ("GND", "gnd_net"),
    )
    got = conv._multiswitch_terminals(comp, "buckboost4")
    assert got == {
        "vin": "vin_net",
        "swa": "swa_net",
        "swb": "swb_net",
        "vout": "vout_net",
        "gnd": "gnd_net",
    }


@requires_sim
def test_multiswitch_invbuckboost_maps_single_sw():
    conv = _conv()
    comp = _component(
        ("VIN", "vin_net"),
        ("LX", "sw_net"),  # SW resolved from the SW-name set (LX alias)
        ("VOUT", "vout_net"),
        ("GND", "gnd_net"),
    )
    got = conv._multiswitch_terminals(comp, "invbuckboost")
    assert got == {
        "vin": "vin_net",
        "sw": "sw_net",
        "vout": "vout_net",
        "gnd": "gnd_net",
    }


@requires_sim
def test_multiswitch_sepic_resolves_node_a_and_b():
    conv = _conv()
    comp = _component(
        ("VIN", "vin_net"),
        ("SW", "A"),
        ("SWB", "B"),
        ("VO", "vout_net"),  # VO alias for VOUT
        ("PGND", "gnd_net"),  # PGND alias for GND
    )
    got = conv._multiswitch_terminals(comp, "sepic")
    assert got == {
        "vin": "vin_net",
        "swa": "A",
        "swb": "B",
        "vout": "vout_net",
        "gnd": "gnd_net",
    }


@requires_sim
def test_multiswitch_missing_node_returns_none():
    conv = _conv()
    # buckboost4 needs a second switch node; omit SW2.
    comp = _component(
        ("VIN", "vin_net"),
        ("SW", "swa_net"),
        ("VOUT", "vout_net"),
        ("GND", "gnd_net"),
    )
    assert conv._multiswitch_terminals(comp, "buckboost4") is None


@requires_sim
def test_multiswitch_uses_node_map():
    conv = _conv()
    conv.node_map = {"vin_net": "0remapped"}
    comp = _component(
        ("VIN", "vin_net"),
        ("SW", "swa_net"),
        ("VOUT", "vout_net"),
        ("GND", "gnd_net"),
    )
    got = conv._multiswitch_terminals(comp, "invbuckboost")
    assert got["vin"] == "0remapped"


@requires_sim
def test_multiswitch_unknown_kind_returns_none():
    conv = _conv()
    comp = _component(("VIN", "vin_net"), ("SW", "swa_net"), ("GND", "gnd_net"))
    assert conv._multiswitch_terminals(comp, "buck") is None


@requires_sim
def test_multiswitch_no_pin_map_returns_none():
    conv = _conv()
    assert conv._multiswitch_terminals(SimpleNamespace(_pins=None), "sepic") is None
