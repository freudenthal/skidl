# -*- coding: utf-8 -*-

"""Electro-thermal LED primitive Sim.Device="LEDTHERMAL".

A high-current LED as a coupled electrical + thermal network: a parasitic series
resistance, a real junction diode whose forward voltage falls with temperature, a
thermal RC network driven by the dissipated power (probeable junction-temperature
node <ref>_TJ), and an optical-output node <ref>_LOPT with a temperature droop.
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
def test_led_thermal_emission():
    """Emits RS, a real diode + .model, the thermal Bpwr/Rth/Cth, and Bopt."""
    _setup()
    a, k = Net("A"), Net("GND")
    d = Part("Device", "LED", ref="D2", Sim_Device="LEDTHERMAL",
             Sim_Params="RS=0.12 N=3 RTH=15 CTH=1m KVF=-4m KDROOP=-4m")
    d["A"] += a
    d["K"] += k
    conv = _conv()
    net = str(conv.convert(strict=False))
    assert "RD2_rs D2_A2 D2_NJ 0.12" in net, net       # parasitic series R
    assert "DD2_d D2_NJT" in net and ".model D2_dmod D(" in net, net  # real diode
    assert "BD2_tsh D2_NJ D2_NJT V = (-0.004)" in net, net  # Vf(T) shift
    assert "BD2_pwr 0 D2_TJ I = V(A," in net, net       # power -> junction temp
    assert "RD2_th D2_TJ D2_AMB 15" in net, net
    assert "BD2_opt 0 D2_LOPT" in net and "-0.004" in net, net  # optical droop
    prov = conv.model_provenance["D2"]
    assert prov.tier == "sim_params" and prov.kind == "led_thermal"


@requires_sim
def test_led_thermal_selfheating_droops_output():
    """A 2 A DC drive heats the junction: Tj rises, Vf falls, optical output droops
    -- the electro-thermal behaviour a plain diode cannot show."""
    _setup()
    import numpy as np
    from skidl import Circuit
    from skidl.sim import simulate

    ckt = Circuit(name="ledet")
    with ckt:
        a, gnd = Net("A"), Net("GND")
        Part("Simulation_SPICE", "IDC", ref="I1", value="2")[1, 2] += gnd, a
        led = Part("Device", "LED", ref="D2", Sim_Device="LEDTHERMAL",
                   Sim_Params="RS=0.1 N=3 IS=1e-17 RTH=12 CTH=400u "
                              "TAMB=25 KVF=-4m KDROOP=-4m")
        led["A"] += a
        led["K"] += gnd
    res = simulate(ckt).transient_analysis(
        "50u", "12m", use_initial_condition=True, initial_conditions={"A": 3.2})
    t = np.asarray(res.time_array())
    vf = np.asarray(res.get_voltage("A"))
    tj = np.asarray(res.get_voltage("D2_TJ"))
    lopt = np.asarray(res.get_voltage("D2_LOPT"))
    i0, i1 = int(np.argmin(np.abs(t - 0.1e-3))), -1
    assert 3.0 < vf[i0] < 3.6                    # cold Vf incl. RS drop
    assert tj[i1] > tj[i0] + 30                  # junction heated tens of degC
    assert vf[i1] < vf[i0] - 0.1                 # Vf fell with temperature
    assert lopt[i1] < 0.9 * lopt[i0]             # optical output drooped
    assert lopt[i0] == pytest.approx(2.0, abs=0.05)  # cold light ~ drive current
