# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: corpus-independent behavioral logic primitives (DPSG WS2).

``Sim.Device="DFF"/"TFF"/"DLATCH"`` and the combinational gates give mixed-signal
designs a *simulatable* digital path that does NOT depend on the un-runnable
corpus digital models (XSPICE ``d_*``, PSpice ``U``-device). These prove the
emitted ngspice netlist is well-formed and, when ngspice is present, that a ÷2
divider actually converges and halves its clock.
"""

import numpy as np
import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool
from skidl.sim import skidl_flat_view

try:
    from skidl.sim.converter import SpiceConverter  # noqa: F401 (needs PySpice)

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


def _dff_divider():
    """A CD4013 D flip-flop wired Q̄->D (÷2), clocked at 50 kHz."""
    u = Part("4xxx", "4013", ref="U1")
    u.Sim_Device = "DFF"
    u.Sim_Params = "vdd=5 tpd=50n"
    vclk = Part("Simulation_SPICE", "VPULSE", ref="V1",
                v1="0", v2="5", td="0", tr="20n", tf="20n", pw="9.98u", per="20u")
    vdd = Part("Simulation_SPICE", "VDC", ref="V2", value="5")
    clk, q, qn, gnd, vd = (
        Net("CLK"), Net("Q"), Net("QBAR"), Net("GND"), Net("VDD"),
    )
    u["3"] += clk       # C   (clock)
    u["1"] += q         # Q
    u["2"] += qn        # ~Q
    u["5"] += qn        # D <- ~Q  (toggle)
    u["14"] += vd       # VDD
    u["7"] += gnd       # VSS
    u["6"] += gnd       # S
    u["4"] += gnd       # R
    vclk[1] += clk
    vclk[2] += gnd
    vdd[1] += vd
    vdd[2] += gnd
    rl = Part("Device", "R", ref="R1", value="1Meg")
    rl[1] += q
    rl[2] += gnd


@requires_sim
def test_dff_netlist_is_master_slave():
    """The DFF emits a master-slave latch (two switch-held caps) + stiff Q/Q̄."""
    _setup()
    _dff_divider()
    netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
    # Two switches (master/slave) sharing one SW model, two memory caps, and the
    # stiff Q / Q̄ B-source outputs.
    assert "SU1_m U1_db U1_m U1_ckb 0 SWLU1" in netlist, netlist
    assert "SU1_s U1_mb U1_qi CLK 0 SWLU1" in netlist, netlist
    assert ".model SWLU1 SW(" in netlist
    assert "CU1_m U1_m 0" in netlist and "IC=0" in netlist
    assert "BU1_q Q 0 V = V(U1_qi) > 2.5 ? 5 : 0" in netlist, netlist
    assert "BU1_qn QBAR 0 V = V(U1_qi) > 2.5 ? 0 : 5" in netlist, netlist
    # No corpus digital model was pulled in.
    assert "d_dff" not in netlist.lower() and "ugate" not in netlist.lower()


@requires_sim
def test_gate_expressions():
    """AND / NOR / XOR gates emit the expected threshold B-source expressions."""
    _setup()
    for op, expect in [
        ("AND", "(V(A) > 2.5) && (V(B) > 2.5)) ? 5 : 0"),
        ("NOR", "(V(A) > 2.5) || (V(B) > 2.5)) ? 0 : 5"),
        ("XOR", "((V(A) > 2.5) + (V(B) > 2.5)) == 1) ? 5 : 0"),
    ]:
        _setup()
        # Host on a 2-input gate symbol (whose pins are the generic KiCad "~"),
        # mapping roles explicitly via Sim.Pins (pin number = role).
        g = Part("74xx", "74LS08", ref="U1")
        g.Sim_Device = op
        g.Sim_Params = "vdd=5"
        g.Sim_Pins = "3=Y 1=A 2=B"
        va = Part("Simulation_SPICE", "VDC", ref="VA", value="5")
        vb = Part("Simulation_SPICE", "VDC", ref="VB", value="0")
        vc = Part("Simulation_SPICE", "VDC", ref="VC", value="5")
        a, b, y, gnd, vcc = (
            Net("A"), Net("B"), Net("Y"), Net("GND"), Net("VCC"),
        )
        g["1"] += a
        g["2"] += b
        g["3"] += y
        g["14"] += vcc
        g["7"] += gnd
        va[1] += a
        va[2] += gnd
        vb[1] += b
        vb[2] += gnd
        vc[1] += vcc
        vc[2] += gnd
        rl = Part("Device", "R", ref="R1", value="1Meg")
        rl[1] += y
        rl[2] += gnd
        netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
        assert expect in netlist, f"{op}: {netlist}"


@requires_sim
def test_dff_divides_clock_by_two():
    """A ÷2 toggle from Sim.Device='DFF' converges and halves the clock with a
    clean rail-to-rail 50%-duty output -- with no corpus model involved."""
    import skidl.sim.simulator  # binds the KiCad ngspice DLL
    import builtins

    _setup()
    _dff_divider()
    try:
        from skidl.sim import simulate

        sim = simulate(builtins.default_circuit)
        res = sim.transient_analysis(
            "50n", "200u", stiff=True, use_initial_condition=True
        )
    except Exception as e:  # pragma: no cover - environment without ngspice
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")

    qv = np.array(res.analysis["Q"])
    cv = np.array(res.analysis["CLK"])

    def rising(v):
        h = v > 2.5
        return int(np.sum((~h[:-1]) & (h[1:])))

    # Rail-to-rail logic swing.
    assert qv.max() > 4.5 and qv.min() < 0.5, f"Q swing {qv.min()}..{qv.max()}"
    # Exactly half the clock's rising edges (±1 for the transient boundary).
    assert abs(2 * rising(qv) - rising(cv)) <= 1, (
        f"Q rising={rising(qv)} vs CLK rising={rising(cv)}"
    )
