# -*- coding: utf-8 -*-

"""Behavioral triggered-breakdown primitive Sim.Device="TRIGSW" (finding F5).

One parameterized smooth-conductance switch for the whole triggered-breakdown /
negative-resistance class (avalanche transistor, spark gap, SCR/DIAC) ngspice
can't model natively. A 3-terminal device is externally triggered on a
ground-referenced control node; a 2-terminal device self-triggers on the voltage
across it. No ideal ``sw`` (timestep collapse) and no latched state (bistable
op-point) -- both documented avalanche-E2E failure modes.
"""

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool

try:
    from skidl.sim.converter import SpiceConverter, SimulationValidationError

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


def _trigsw(pins, params=None):
    """A Q_NPN_BCE symbol driven as a TRIGSW, with the given Sim.Pins role map."""
    q = Part("Transistor_BJT", "Q_NPN_BCE", ref="Q1")
    q.Sim_Device = "TRIGSW"
    q.Sim_Pins = pins
    if params:
        q.Sim_Params = params
    return q


@requires_sim
def test_trigsw_external_trigger_emission():
    """3-terminal: ground-referenced control V(G); a B-source + leak, tier sim_params."""
    _setup()
    col, trig, em = Net("COL"), Net("TRIG"), Net("EM")
    q = _trigsw("2=P 1=G 3=N", "vt=2.5 ron=1.2")
    q[2] += col  # P (collector)
    q[1] += trig  # G (base / trigger)
    q[3] += em  # N (emitter)
    conv = _conv()
    net = str(conv.convert(strict=False))
    # a behavioral current source across the power terminals, gated on V(TRIG)
    assert "BQ1_sw COL EM I =" in net, net
    assert "V(COL,EM)" in net, net  # driving voltage across the switch
    assert "V(TRIG)" in net, net  # ground-referenced trigger, NOT V(P,N)
    assert "RQ1_leak COL EM" in net, net  # op-point / off-state leak
    prov = conv.model_provenance["Q1"]
    assert prov.tier == "sim_params" and prov.kind == "trigsw"
    assert "ext-trig" in prov.name


@requires_sim
def test_trigsw_self_trigger_when_two_terminal():
    """2-terminal (no control pin): self-triggered on the voltage across it."""
    _setup()
    p, n = Net("P"), Net("N")
    q = _trigsw("2=P 3=N", "vt=90")  # only power terminals mapped
    q[2] += p
    q[3] += n
    conv = _conv()
    net = str(conv.convert(strict=False))
    # the sigmoid trigger references the across-switch voltage, not a control node
    assert "exp(-(V(P,N) - 90)" in net, net
    assert "self-trig" in conv.model_provenance["Q1"].name


@requires_sim
def test_trigsw_params_and_quench():
    """VT / RLEAK / optional VHOLD quench thread into the emitted expression."""
    _setup()
    col, trig, em = Net("COL"), Net("TRIG"), Net("EM")
    q = _trigsw("2=P 1=G 3=N", "vt=200 ron=1.2 width=0.1 rleak=1meg vhold=60")
    q[2] += col
    q[1] += trig
    q[3] += em
    net = str(_conv().convert(strict=False))
    assert "(V(TRIG) - 200)/0.1" in net, net  # threshold + sharpness
    # optional quench: a second sigmoid falling off below VHOLD across the switch
    assert "exp(-(V(COL,EM) - 60)" in net, net


@requires_sim
def test_trigsw_missing_power_terminal_is_validation_error():
    """Only one power terminal resolvable -> a clear validation error, not a crash."""
    _setup()
    col = Net("COL")
    q = _trigsw("2=P")  # no N terminal
    q[2] += col
    with pytest.raises(SimulationValidationError) as exc:
        _conv().convert(strict=True)
    assert "TRIGSW" in str(exc.value)
