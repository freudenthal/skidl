# -*- coding: utf-8 -*-

"""Behavioral optical-coupling primitive Sim.Device="OPTOCOUPLER".

Models an emitter (a driven LED) -> detector (a photodiode) light path as a
linear current scale: the photodiode delivers a photocurrent proportional to the
emitter's branch current, ``i_pd = K * I(<sense>)``. This lets a simulation
couple a real, driven LED's current to a photodiode front-end (e.g. a TIA)
without modeling an optical channel. Realized as an ngspice-native behavioral
``B`` current source across the two photodiode terminals, exactly like the
TRIGSW / LDO / gate emitters -- no XSPICE, no external symbol support needed.

The controlling current is read from a named 0 V series sense source: PySpice
names a ``Simulation_SPICE:VDC`` ref ``VLED`` as deck element ``VVLED``, so a
``sense=VLED`` param is emitted as ``I(VVLED)``.
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


def _opto(params, pins=None):
    """A Device:D photodiode symbol driven as an OPTOCOUPLER."""
    d = Part("Device", "D", ref="U1")
    d.Sim_Device = "OPTOCOUPLER"
    d.Sim_Params = params
    if pins:
        d.Sim_Pins = pins
    return d


@requires_sim
def test_optocoupler_emission_by_pin_name():
    """A/K resolve by pin name; a behavioral B-source injects k*I(V<sense>)."""
    _setup()
    ninv, gnd = Net("NINV"), Net("GND")
    d = _opto("k=5e-6 sense=VLED")
    d["A"] += ninv  # photodiode output -> TIA summing node
    d["K"] += gnd
    conv = _conv()
    net = str(conv.convert(strict=False))
    # photocurrent A->K, proportional to the LED sense-source branch current.
    assert "BU1_pd NINV" in net, net
    assert "5e-06*I(VVLED)" in net, net  # V-prefix prepended to the skidl ref
    prov = conv.model_provenance["U1"]
    assert prov.tier == "sim_params" and prov.kind == "optocoupler"


@requires_sim
def test_optocoupler_gain_alias_and_default():
    """GAIN aliases K; an absent coefficient defaults to 1e-3."""
    _setup()
    a, gnd = Net("A"), Net("GND")
    d = _opto("gain=2e-4 sense=VLED")
    d["A"] += a
    d["K"] += gnd
    net = str(_conv().convert(strict=False))
    assert "2e-04*I(VVLED)" in net or "0.0002*I(VVLED)" in net, net

    _setup()
    a, gnd = Net("A"), Net("GND")
    d = _opto("sense=VLED")  # no coefficient -> default 1e-3
    d["A"] += a
    d["K"] += gnd
    net = str(_conv().convert(strict=False))
    assert "I(VVLED)" in net and "BU1_pd" in net, net


@requires_sim
def test_optocoupler_already_prefixed_sense_is_idempotent():
    """A sense already given as the deck name (VVx) is not double-prefixed."""
    _setup()
    a, gnd = Net("A"), Net("GND")
    d = _opto("k=1e-3 sense=VVLED")
    d["A"] += a
    d["K"] += gnd
    net = str(_conv().convert(strict=False))
    assert "I(VVLED)" in net and "I(VVVLED)" not in net, net


@requires_sim
def test_optocoupler_voltage_sense_mode():
    """gm + vp/vn -> a VCCS keyed on a real sense-resistor voltage (board-fabricable)."""
    _setup()
    ninv, gnd = Net("NINV"), Net("GND")
    d = _opto("gm=1e-4 vp=NDRV vn=LED_A")
    d["A"] += ninv
    d["K"] += gnd
    conv = _conv()
    net = str(conv.convert(strict=False))
    assert "BU1_pd NINV" in net, net
    assert "V(NDRV,LED_A)" in net, net  # senses the shunt voltage, no 0V source
    assert "I(" not in net.split("BU1_pd")[1].split("\n")[0], net  # not current-sense
    assert conv.model_provenance["U1"].kind == "optocoupler"


@requires_sim
def test_optocoupler_voltage_sense_maps_gnd_to_node0():
    """A GND sense terminal maps to ngspice node 0."""
    _setup()
    a, gnd = Net("A"), Net("GND")
    d = _opto("gm=1e-4 vp=SHUNT vn=GND")
    d["A"] += a
    d["K"] += gnd
    net = str(_conv().convert(strict=False))
    assert "V(SHUNT,0)" in net, net


@requires_sim
def test_optocoupler_missing_sense_is_skipped_not_crash():
    """No sense= -> the device is skipped with a warning, not a crash."""
    _setup()
    a, gnd = Net("A"), Net("GND")
    d = _opto("k=1e-3")  # no sense source named
    d["A"] += a
    d["K"] += gnd
    conv = _conv()
    net = str(conv.convert(strict=False))
    assert "BU1_pd" not in net, net
    assert "U1" not in conv.model_provenance


@requires_sim
def test_optocoupler_coupling_regulates_tia_output():
    """End-to-end op-point: VOUT = k * I_LED * Rf on an ideal inverting TIA."""
    _setup()
    from skidl.sim import simulate
    from skidl import Circuit

    ckt = Circuit(name="opto_e2e")
    with ckt:
        v24, a, k, gnd = Net("V24"), Net("LED_A"), Net("LED_K"), Net("GND")
        ninv, vout, vp, vn = Net("NINV"), Net("VOUT"), Net("VP"), Net("VN")
        Part("Simulation_SPICE", "VDC", ref="V1", value="24")[1, 2] += v24, gnd
        rset = Part("Device", "R", ref="R1", value="11")
        rset[1] += v24
        rset[2] += a
        led = Part("Device", "LED", ref="D1", value="1N4148")  # a diode drop
        led["A"] += a
        led["K"] += k
        Part("Simulation_SPICE", "VDC", ref="VLED", value="0")[1, 2] += k, gnd
        u1 = Part("Amplifier_Operational", "TL071", ref="U1")
        u1[3] += gnd
        u1[2] += ninv
        u1[6] += vout
        u1[7] += vp
        u1[4] += vn
        rf = Part("Device", "R", ref="RF", value="100k")
        rf[1] += ninv
        rf[2] += vout
        Part("Simulation_SPICE", "VDC", ref="VP", value="12")[1, 2] += vp, gnd
        Part("Simulation_SPICE", "VDC", ref="VN", value="-12")[1, 2] += vn, gnd
        opt = Part("Device", "D", ref="U2", Sim_Device="OPTOCOUPLER",
                   Sim_Params="k=5e-6 sense=VLED")
        opt["A"] += ninv
        opt["K"] += gnd
    res = simulate(ckt).operating_point()
    iled = abs(res.get_current("VLED"))
    vout = res.get_voltage("VOUT")
    assert iled > 1.0, iled  # LED is conducting a real current
    # inverting TIA: |Vout| = k * I_LED * Rf, and the sign is positive here.
    assert vout == pytest.approx(5e-6 * iled * 100e3, rel=0.02), (vout, iled)
