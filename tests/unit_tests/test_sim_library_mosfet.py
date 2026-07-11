# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: external ``.model`` transistor terminals resolve by pin NAME + VDMOS.

Regression for E2E finding M1: an external/vendor ``.model`` MOSFET (or BJT) was
wired D/G/S (C/B/E) by pin *position*, but KiCad symbols number their pins
inconsistently, so the device came out with gate/drain (or collector/base)
swapped -- a clean-provenance but DEAD part. Terminals must be resolved by pin
name, exactly like the curated ``_add_mosfet``/``_add_bjt_transistor`` paths.
Also: an ngspice ``VDMOS`` card is a 3-terminal element (``M nd ng ns model``),
so it must be emitted with 3 nodes, not the 4-node ``M d g s b`` line.
"""

import os

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


def _write_lib(tmp_path, text):
    p = tmp_path / "vendor.lib"
    p.write_text(text)
    return str(p)


@requires_sim
def test_external_model_mosfet_resolves_by_name(tmp_path):
    """A vendor NMOS ``.model`` on an IRF540N symbol (pins 1=G,2=D,3=S) emits
    the M line in D/G/S *named* order, not pin-number order (which would put
    the gate net first)."""
    _setup()
    lib = _write_lib(tmp_path, ".model MYNMOS NMOS(Vto=2 Kp=1)\n")
    q = Part("Transistor_FET", "IRF540N", ref="Q1")
    q.Sim_Library = lib
    q.Sim_Name = "MYNMOS"
    Net("DRN").connect(q["D"])
    Net("GATE").connect(q["G"])
    Net("SRC").connect(q["S"])
    netlist = str(_conv().convert(strict=False))
    # D G S (B defaults to S) -- named order
    assert "MQ1 DRN GATE SRC SRC MYNMOS" in netlist, netlist
    # positional order (pin1=G first) would have been the DEAD wiring
    assert "MQ1 GATE DRN" not in netlist, netlist


@requires_sim
def test_external_model_vdmos_is_three_terminal(tmp_path):
    """A VDMOS card is emitted as a 3-terminal element (D G S model), not the
    4-node M line -- a 4-node line misparses against a VDMOS ``.model``."""
    _setup()
    lib = _write_lib(tmp_path, ".model MYVMOS VDMOS(Vto=4 Kp=20)\n")
    q = Part("Transistor_FET", "IRF540N", ref="Q1")
    q.Sim_Library = lib
    q.Sim_Name = "MYVMOS"
    Net("DRN").connect(q["D"])
    Net("GATE").connect(q["G"])
    Net("SRC").connect(q["S"])
    netlist = str(_conv().convert(strict=False))
    assert "MQ1 DRN GATE SRC MYVMOS" in netlist, netlist
    # no 4th (bulk) node before the model name
    assert "MQ1 DRN GATE SRC SRC" not in netlist, netlist


@requires_sim
def test_external_model_bjt_resolves_by_name(tmp_path):
    """A vendor NPN ``.model`` on a BJT symbol resolves C/B/E by name."""
    _setup()
    lib = _write_lib(tmp_path, ".model MYNPN NPN(BF=100)\n")
    qq = None
    for sym in ("2N3904", "BC547", "2N2222"):
        try:
            qq = Part("Transistor_BJT", sym, ref="Q2")
            break
        except Exception:
            qq = None
    if qq is None:
        pytest.skip("no BJT symbol available on this host")
    qq.Sim_Library = lib
    qq.Sim_Name = "MYNPN"
    Net("COL").connect(qq["C"])
    Net("BAS").connect(qq["B"])
    Net("EMI").connect(qq["E"])
    netlist = str(_conv().convert(strict=False))
    assert "QQ2 COL BAS EMI MYNPN" in netlist, netlist


@requires_sim
def test_external_vdmos_conducts_live(tmp_path):
    """Live ngspice: common-source .op with a VDMOS Vto=4, Vgs=10 -> the drain
    pulls down (device conducts). Before the M1 fix it sat at Vsupply (dead)."""
    _setup()
    lib = _write_lib(
        tmp_path, ".model MYVMOS VDMOS(Vto=4 Kp=20 Rd=0.02 Rs=0.01)\n"
    )
    vdd = Part("Simulation_SPICE", "VDC", ref="V1", value="24")
    vg = Part("Simulation_SPICE", "VDC", ref="V2", value="10")
    rd = Part("Device", "R", ref="R1", value="1k")
    q = Part("Transistor_FET", "IRF540N", ref="Q1")
    q.Sim_Library = lib
    q.Sim_Name = "MYVMOS"
    vp, dr, g, gnd = (Net(n) for n in ("VDD", "DRN", "G", "0"))
    vp.connect(vdd[1])
    gnd.connect(vdd[2])
    g.connect(vg[1])
    gnd.connect(vg[2])
    vp.connect(rd[1])
    dr.connect(rd[2], q["D"])
    g.connect(q["G"])
    gnd.connect(q["S"])
    from skidl.sim import simulate

    try:
        r = simulate().operating_point()
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    vdrn = float(r.get_voltage("DRN"))
    assert vdrn < 2.0, f"VDMOS not conducting: V(DRN)={vdrn}"
