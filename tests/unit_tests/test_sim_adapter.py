# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Phase 3 tests: skidl -> SPICE sim via the ``skidl.sim`` adapter.

Exercises the VENDORED, standalone ``skidl.sim`` stack (no circuit_synth
dependency). Layered so most run with no extra deps:

* Adapter-shape tests need only skidl (assert the ``_FlatCircuit`` view exposes
  exactly what the converter reads).
* Conversion tests need PySpice (skidl-authored circuit -> correct SPICE netlist
  through ``skidl.sim.converter.SpiceConverter``).
* Live tests additionally need ngspice (skidl-authored circuit actually
  simulates: DC op point, a diode with correct pin-name polarity, an LDO
  macromodel).

The port is an adapter, not a rewrite: these prove skidl parts satisfy the
converter's duck-typed contract unchanged.
"""

import subprocess
import sys
import textwrap

import pytest

from skidl import KICAD10, Net, Part, lib_search_paths, set_default_tool
from skidl.sim import skidl_flat_view

# --- optional-dependency probes -------------------------------------------

# Exercise the VENDORED, standalone stack (skidl.sim.*), NOT circuit_synth.
try:
    from skidl.sim.converter import SpiceConverter  # noqa: F401  (needs PySpice)

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


# --- adapter shape (skidl only) -------------------------------------------


def test_adapter_view_shape():
    _setup()
    r1 = Part("Device", "R", value="10k")
    r2 = Part("Device", "R", value="20k")
    r1.Sim_Device = "R"
    Net("VIN").connect(r1[1])
    Net("MID").connect(r1[2], r2[1])
    Net("0").connect(r2[2])

    view = skidl_flat_view()
    assert set(view.components) == {"R1", "R2"}
    r1v = view.components["R1"]
    assert r1v.symbol == "Device:R"
    assert r1v.value == "10k"
    # Sim.* fields land in _extra_fields (converter reads them there).
    assert r1v._extra_fields.get("Sim_Device") == "R"
    # Pins carry num/name and resolve to nets by name.
    assert {p.net.name for p in r1v._pins.values()} == {"VIN", "MID"}
    assert "VIN" in view.nets and "MID" in view.nets


def test_adapter_unconnected_pin_has_no_net():
    _setup()
    r = Part("Device", "R", value="1k")
    Net("A").connect(r[1])  # pin 2 left unconnected
    view = skidl_flat_view()
    pins = view.components["R1"]._pins
    assert pins["1"].net is not None and pins["1"].net.name == "A"
    # skidl auto-nets a lone pin; the adapter must not invent a shared node.
    assert pins["2"].net is None or pins["2"].net.name != "A"


def test_adapter_supplies_multi_unit_symbol_data():
    """For a multi-unit part the adapter attaches ``_symbol_data`` (unit_count +
    per-pin unit/function/name) so the converter's op-amp section resolution
    works standalone -- no circuit_synth SymbolLibCache needed."""
    _setup()
    op = Part("Amplifier_Operational", "LM358")
    ua, ub = op.unit["uA"], op.unit["uB"]
    Net("INA").connect(ua[3])
    Net("OUTA").connect(ua[1], ua[2])
    Net("INB").connect(ub[5])
    Net("OUTB").connect(ub[7], ub[6])

    sd = skidl_flat_view().components["U1"]._symbol_data
    assert sd is not None and sd["unit_count"] >= 2
    funcs = {p["function"] for p in sd["pins"]}
    assert "output" in funcs and "input" in funcs
    # A single-unit part gets no symbol_data (converter keeps whole-component).
    _setup()
    Part("Device", "R", value="1k")
    assert skidl_flat_view().components["R1"]._symbol_data is None


# --- conversion through the real cs converter (PySpice) --------------------


@requires_sim
def test_divider_converts_to_correct_spice():
    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value="5")
    r1 = Part("Device", "R", value="10k")
    r2 = Part("Device", "R", value="20k")
    Net("VIN").connect(v1[1], r1[1])
    Net("MID").connect(r1[2], r2[1])
    Net("0").connect(v1[2], r2[2])

    spice = SpiceConverter(skidl_flat_view()).convert(strict=True)
    netlist = str(spice)
    assert "RR1 VIN MID 10000" in netlist
    assert "RR2 MID 0 20000" in netlist
    assert "VV1 VIN 0 5" in netlist


@requires_sim
def test_diode_polarity_resolved_by_pin_name():
    """A skidl Device:D places anode-then-cathode by KiCad pin NAME (A/K), not
    pin number (pin 1 is K, pin 2 is A) -- the run-3 silent-wrong-polarity class."""
    _setup()
    d = Part("Device", "D", value="D")
    Net("ANODE").connect(d["A"])
    Net("CATH").connect(d["K"])

    spice = SpiceConverter(skidl_flat_view()).convert(strict=False)
    netlist = str(spice)
    # PySpice emits diodes as "D<ref> <anode> <cathode> <model>".
    diode_lines = [ln for ln in netlist.splitlines() if ln.strip().startswith("D")]
    assert diode_lines, f"no diode in netlist:\n{netlist}"
    parts = diode_lines[0].split()
    assert parts[1] == "ANODE" and parts[2] == "CATH", diode_lines[0]


# --- live simulation (ngspice) --------------------------------------------


@requires_sim
def test_divider_simulates_dc_operating_point():
    """Full seam, live: skidl divider -> adapter -> converter -> ngspice DC op."""
    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value="5")
    r1 = Part("Device", "R", value="10k")
    r2 = Part("Device", "R", value="20k")
    Net("VIN").connect(v1[1], r1[1])
    Net("MID").connect(r1[2], r2[1])
    Net("0").connect(v1[2], r2[2])

    spice = SpiceConverter(skidl_flat_view()).convert(strict=True)
    try:
        import skidl.sim.simulator  # noqa: F401  configures ngspice

        sim = spice.simulator()
        an = sim.operating_point()
    except Exception as e:  # ngspice shared lib not loadable in this env
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    mid = float(an["MID"][0])
    assert abs(mid - 5.0 * 20.0 / 30.0) < 0.01, f"V(MID)={mid}"


@requires_sim
def test_ldo_macromodel_regulates(tmp_path):
    """A skidl-authored LDO (Sim_Device=LDO + Sim_Params) regulates -- proves the
    Sim.* macromodel path (a crown-jewel behavioral model) reaches skidl parts."""
    _setup()
    u1 = Part("Regulator_Linear", "AMS1117-3.3", ref="U1")
    u1.Sim_Device = "LDO"
    u1.Sim_Params = "vout=3.3 vdrop=0.3 rser=0.1 iq=2m"
    v1 = Part("Simulation_SPICE", "VDC", value="5")
    rl = Part("Device", "R", value="33")
    vin, vout, gnd = Net("VIN"), Net("VOUT"), Net("GND")
    v1[1] += vin
    v1[2] += gnd
    u1[3] += vin  # VI
    u1[1] += gnd  # GND
    u1[2] += vout  # VO
    rl[1] += vout
    rl[2] += gnd

    spice = SpiceConverter(skidl_flat_view()).convert(strict=True)
    try:
        import skidl.sim.simulator  # noqa: F401

        sim = spice.simulator()
        an = sim.operating_point()
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    vo = float(an["VOUT"][0])
    # 3.3 V minus the RSER * I_load drop; well clear of an unregulated pass-through.
    assert 3.15 < vo < 3.31, f"V(VOUT)={vo}"


# --- standalone (no circuit_synth) ----------------------------------------


@requires_sim
def test_sim_stack_imports_without_circuit_synth():
    """The vendored sim stack must import with circuit_synth unavailable -- the
    whole point of vendoring it for an upstream PR. Runs in a subprocess with a
    meta-path finder that hard-blocks circuit_synth, then imports the heavy
    modules (converter/simulator) and builds a view."""
    code = textwrap.dedent("""
        import sys, importlib.abc
        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "circuit_synth" or name.startswith("circuit_synth."):
                    raise ImportError("blocked for standalone test: " + name)
        sys.meta_path.insert(0, _Block())
        try:
            import circuit_synth  # must fail
            print("FAIL: circuit_synth importable"); sys.exit(1)
        except ImportError:
            pass
        import skidl.sim.converter, skidl.sim.simulator, skidl.sim.models
        from skidl.sim import simulate, skidl_flat_view, SpiceConverter
        print("OK")
        """)
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert (
        r.returncode == 0 and "OK" in r.stdout
    ), f"standalone import failed:\nstdout={r.stdout}\nstderr={r.stderr[-800:]}"
