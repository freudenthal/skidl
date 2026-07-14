# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: part-agnostic model-simulatability classification (DPSG WS1) and the
stale-absolute-Sim.Library auto-resolve fallback (DPSG WS3).

The SPICE corpus mixes dialects; some classes ngspice-in-KiCad cannot run
(XSPICE ``d_*`` digital, PSpice ``U``-device digital, encrypted). The classifier
flags them by structural signature (never by part name), and the sim pre-flight
raises a clear, class-named error instead of dying on a dead node.
"""

import textwrap

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool
from skidl.sim import skidl_flat_view
from skidl.sim.simulatability import classify_model_file, classify_spice_model

try:
    from skidl.sim.converter import SimulationValidationError, SpiceConverter

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


# --- classifier (no PySpice needed) ---------------------------------------


def test_classify_xspice_digital():
    body = textwrap.dedent(
        """\
        .subckt FF74 clrbar d clk prebar q qbar
        adff d clk prebar clrbar q qbar ls_dff
        .model ls_dff d_dff(clk_delay=1e-9)
        .ends
        """
    )
    mc = classify_spice_model(body)
    assert mc.simulatable == "no"
    assert mc.dialect == "xspice-digital"


def test_classify_pspice_udevice():
    body = textwrap.dedent(
        """\
        .subckt CD4013 q qbar clk d
        U1 dff(1) VDD VSS d clk q qbar DLY_4000 IO_4000UB
        .model DLY_4000 ugate(tplhty=90ns)
        .ends
        """
    )
    mc = classify_spice_model(body)
    assert mc.simulatable == "no"
    assert mc.dialect == "pspice-digital"


def test_classify_analog_yes():
    assert classify_spice_model(".model MYD D(is=1e-14)").simulatable == "yes"
    sub = ".subckt OP 1 2 3\nR1 1 2 1k\n.ends"
    assert classify_spice_model(sub).simulatable == "yes"


def test_classify_encrypted():
    mc = classify_spice_model("*ENCRYPTED\n" + "\x01\x02\x03" * 50)
    assert mc.simulatable == "no" and mc.dialect == "encrypted"


def test_classify_file_slices_named_block(tmp_path):
    lib = tmp_path / "mix.lib"
    lib.write_text(
        ".model GOODD D(is=1e-14)\n"
        ".subckt BADFF c q\n a1 c q dm\n .model dm d_dff()\n.ends\n"
    )
    assert classify_model_file(str(lib), "GOODD").simulatable == "yes"
    assert classify_model_file(str(lib), "BADFF").simulatable == "no"


# --- pre-flight raises a class-named error --------------------------------


@requires_sim
def test_preflight_rejects_xspice_digital(tmp_path):
    """A Part resolving to an XSPICE-digital subckt raises a named error, not a
    dead run."""
    _setup()
    lib = tmp_path / "dig.lib"
    lib.write_text(
        ".subckt MYFF clk q\n adff clk q dm\n .model dm d_dff()\n.ends\n"
    )
    d = Part("Device", "D", ref="D1")
    d.Sim_Library = str(lib)
    d.Sim_Name = "MYFF"
    n1, gnd = Net("N1"), Net("GND")
    d[1] += n1
    d[2] += gnd
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
    v[1] += n1
    v[2] += gnd
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(skidl_flat_view()).convert(strict=True)
    msg = str(ei.value)
    assert "non-simulatable" in msg and "xspice-digital" in msg


@requires_sim
def test_preflight_rejects_pspice_udevice(tmp_path):
    _setup()
    lib = tmp_path / "u.lib"
    lib.write_text(
        ".subckt MYU a b\n U1 buf(1) VDD VSS a b IO_4000UB\n"
        " .model IO_4000UB uio()\n.ends\n"
    )
    d = Part("Device", "D", ref="D1")
    d.Sim_Library = str(lib)
    d.Sim_Name = "MYU"
    n1, gnd = Net("N1"), Net("GND")
    d[1] += n1
    d[2] += gnd
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
    v[1] += n1
    v[2] += gnd
    with pytest.raises(SimulationValidationError) as ei:
        SpiceConverter(skidl_flat_view()).convert(strict=True)
    assert "pspice-digital" in str(ei.value)


@requires_sim
def test_preflight_allows_analog_subckt(tmp_path):
    """A normal analog subckt is NOT flagged (0 FAILED)."""
    _setup()
    lib = tmp_path / "an.lib"
    lib.write_text(".subckt MYR a b\nR1 a b 1k\n.ends\n")
    d = Part("Device", "D", ref="D1")
    d.Sim_Library = str(lib)
    d.Sim_Name = "MYR"
    d.Sim_Pins = "1=a 2=b"
    n1, gnd = Net("N1"), Net("GND")
    d[1] += n1
    d[2] += gnd
    v = Part("Simulation_SPICE", "VDC", ref="V1", value="5")
    v[1] += n1
    v[2] += gnd
    # Should convert without a simulatability problem (may still succeed fully).
    netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
    assert "XD1" in netlist or "xd1" in netlist.lower()
