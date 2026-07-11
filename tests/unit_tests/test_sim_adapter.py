# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Tests: skidl -> SPICE sim via the ``skidl.sim`` adapter.

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


def _setup_real_libs():
    """Setup that binds to the REAL KiCad-10 symbol dirs only (no ``"."``, so the
    bundled ``test_data/kicad6`` libs don't shadow KiCad-10-only parts like
    ADA4817). Skips the test if a real KiCad-10 install isn't present."""
    set_default_tool(KICAD10)
    from skidl.tools.kicad10.lib import default_lib_paths

    real = [
        p
        for p in default_lib_paths()
        if p not in (".", "") and "test_data" not in str(p).replace("\\", "/")
    ]
    if not real:
        pytest.skip("no real KiCad-10 symbol library on this host")
    lib_search_paths["kicad10"] = real
    import builtins

    builtins.default_circuit.mini_reset()
    try:
        Part("Amplifier_Operational", "ADA4817-1ACP")
    except Exception:
        pytest.skip("ADA4817-1ACP not in the installed KiCad-10 libraries")
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


def test_adapter_pins_carry_electrical_func():
    """Every AdaptedPin exposes its electrical type as ``.func`` (lowercased
    ``"output"``/``"input"``/``"power_in"``/...). Without it a single-unit op-amp's
    output pin is invisible to the converter's terminal resolver, which then falls
    back to a positional guess and scrambles (out, in+, in-) -> singular matrix."""
    _setup_real_libs()
    op = Part("Amplifier_Operational", "ADA4817-1ACP")
    Net("VOUT").connect(op[7], op[2])  # OUT + FB
    Net("INN").connect(op[3])  # "-"
    Net("INP").connect(op[4])  # "+"
    pins = skidl_flat_view().components["U1"]._pins
    assert pins["7"].func == "output"
    assert pins["3"].func == "input" and pins["4"].func == "input"
    assert "output" not in pins["3"].func


@requires_sim
def test_single_unit_opamp_terminals_resolve_by_func():
    """A single-unit op-amp (ADA4817) resolves (out, in+, in-) from pin func/name,
    NOT position -- the SiPM-TIA canary regression. Before the ``.func`` fix the
    VCVS drove the wrong node (a supply rail), shorting two voltage sources."""
    _setup_real_libs()
    op = Part("Amplifier_Operational", "ADA4817-1ACP", ref="U1")
    op.Sim_Gbw = "1.4G"
    r = Part("Device", "R", value="100k")
    vout, ninv, gnd = Net("VOUT"), Net("NINV"), Net("GND")
    op[7] += vout  # OUT
    op[2] += vout  # FB
    op[3] += ninv  # -
    op[4] += gnd  # +
    op[8] += Net("VP")
    op[5] += Net("VN")
    r[1] += ninv
    r[2] += vout

    # Resolve directly off the view component (node_map empty pre-convert is fine).
    view = skidl_flat_view()
    out, inp, inn = SpiceConverter(view)._opamp_terminals(view.components["U1"])
    assert (out, inp, inn) == ("VOUT", "GND", "NINV"), (out, inp, inn)


# --- conversion through the SPICE converter (PySpice) --------------------


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


@requires_sim
def test_negative_dc_source_keeps_sign():
    """A negative VDC ``value`` must reach the netlist with its sign intact --
    the run-B4 silent sign-drop turned "-1" into +1 V (the DiffAmp repro)."""
    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value="-1")
    r1 = Part("Device", "R", value="1k")
    Net("N").connect(v1[1], r1[1])
    Net("0").connect(v1[2], r1[2])

    netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
    assert "VV1 N 0 -1" in netlist, netlist


@requires_sim
@pytest.mark.parametrize(
    "value,expected",
    [("-2.5", "-2.5"), ("+1", "1"), ("1.0", "1"), ("3.3V", "3.3"), ("-1.0", "-1")],
)
def test_dc_source_value_forms(value, expected):
    """Signed / decimal / suffixed DC source values all parse to the right float."""
    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value=value)
    r1 = Part("Device", "R", value="1k")
    Net("N").connect(v1[1], r1[1])
    Net("0").connect(v1[2], r1[2])
    netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
    vline = next(ln for ln in netlist.splitlines() if ln.startswith("VV1"))
    assert float(vline.split()[-1]) == float(expected), vline


@requires_sim
def test_vpulse_negative_levels_survive():
    """A VPULSE swinging negative keeps its negative level in the spec (the
    waveform parsers must not drop signs either)."""
    _setup()
    v1 = Part("Simulation_SPICE", "VPULSE", value="1")
    v1.Sim_Params = "v1=-2 v2=2"
    r1 = Part("Device", "R", value="1k")
    Net("N").connect(v1[1], r1[1])
    Net("0").connect(v1[2], r1[2])
    netlist = str(SpiceConverter(skidl_flat_view()).convert(strict=True))
    pulse = next(ln for ln in netlist.splitlines() if "PULSE(" in ln.upper())
    assert "-2" in pulse and "2" in pulse, pulse


@requires_sim
def test_unparseable_source_value_raises():
    """An unparseable *source* value is a correctness trap -- it must raise, not
    silently substitute 1.0 (same defect class as the sign drop)."""
    from skidl.sim.converter import SimulationValidationError

    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value="garbage")
    r1 = Part("Device", "R", value="1k")
    Net("N").connect(v1[1], r1[1])
    Net("0").connect(v1[2], r1[2])
    with pytest.raises(SimulationValidationError):
        SpiceConverter(skidl_flat_view()).convert(strict=True)


# --- model aliasing across package suffixes (B5) --------------------------


@requires_sim
def test_alias_table_resolves_die_model():
    """ModelLibrary.resolve_model maps a package variant to its die model."""
    from skidl.sim.models import get_model_library

    lib = get_model_library()
    model, canonical = lib.resolve_model("MMBT3904")
    assert canonical == "2N3904" and model is not None
    model, canonical = lib.resolve_model("1N4148W")
    assert canonical == "1N4148" and model is not None
    # An exact hit is unchanged; a truly unknown part does not resolve.
    assert lib.resolve_model("1N4148") == (lib.get_model("1N4148"), "1N4148")
    assert lib.resolve_model("XYZ999") == (None, "XYZ999")


@requires_sim
def test_lookup_model_spec_alias_and_suffix_strip():
    """_lookup_model_spec resolves via the alias table and the diode/BJT-only
    package-suffix strip, and reports the canonical name; unknown stays
    unresolved; MOSFETs are not suffix-stripped."""
    _setup()
    conv = SpiceConverter(skidl_flat_view())
    # explicit alias
    spec, tier, resolved = conv._lookup_model_spec("1N4148W", "diode")
    assert spec is not None and tier == "datasheet_fit" and resolved == "1N4148"
    # generic suffix strip (not in ALIASES): 1N4007W -> 1N4007
    spec, tier, resolved = conv._lookup_model_spec("1N4007W", "diode")
    assert spec is not None and resolved == "1N4007"
    # truly unknown stays unresolved
    assert conv._lookup_model_spec("XYZ999", "diode") == (None, "unresolved", "XYZ999")
    # MOSFETs are not suffix-stripped (avoid guessing across FET families)
    assert conv._lookup_model_spec("2N7000W", "mosfet")[0] is None


@requires_sim
def test_aliased_diode_records_provenance():
    """A diode with value='1N4148W' converts with datasheet_fit provenance whose
    name records the alias mapping '1N4148W->1N4148' (never silent)."""
    _setup()
    d = Part("Device", "D", value="1N4148W", ref="D1")
    Net("A").connect(d["A"])
    Net("K").connect(d["K"])
    conv = SpiceConverter(skidl_flat_view())
    conv.convert(strict=False)
    prov = conv.model_provenance["D1"]
    assert prov.tier == "datasheet_fit"
    assert prov.name == "1N4148W->1N4148", prov.name


@requires_sim
def test_alias_sim_params_override_still_wins():
    """Sim.Params on an aliased part still produces a per-device derived card."""
    _setup()
    d = Part("Device", "D", value="1N4148W", ref="D1")
    d.Sim_Params = "IS=2e-9"
    Net("A").connect(d["A"])
    Net("K").connect(d["K"])
    conv = SpiceConverter(skidl_flat_view())
    conv.convert(strict=False)
    assert conv.model_provenance["D1"].overridden
    assert any(name.startswith("1N4148W_") for name in conv.derived_models)


# --- HV Schottky models + Schottky-aware fallback (LLC E2E R1/R2) ---------


@requires_sim
def test_hv_schottky_models_resolve_datasheet_fit():
    """SS3H10 / STPS3150 / SS310 resolve as datasheet_fit with BV >= 100 -- the
    R1 gap (no >=100 V Schottky meant LLC rectifiers silently reverse-broke)."""
    _setup()
    for name, bv in (("SS3H10", 100), ("STPS3150", 150), ("SS310", 100)):
        conv = SpiceConverter(skidl_flat_view())
        spec, tier, resolved = conv._lookup_model_spec(name, "diode")
        assert spec is not None and tier == "datasheet_fit", name
        device_type, params = spec
        assert device_type == "D" and params["BV"] >= bv, (name, params)
        assert params.get("EG") == 0.69, name  # a real Schottky fit


@requires_sim
def test_ss3h10_diode_converts_with_provenance():
    """A diode with plain value='SS3H10' (no Sim.Params) converts datasheet_fit
    and its .model card carries BV=100 -- the R1 fix's user-facing contract."""
    _setup()
    d = Part("Device", "D", value="SS3H10", ref="D1")
    Net("A").connect(d["A"])
    Net("K").connect(d["K"])
    conv = SpiceConverter(skidl_flat_view())
    netlist = str(conv.convert(strict=False))
    prov = conv.model_provenance["D1"]
    assert prov.tier == "datasheet_fit" and prov.name == "SS3H10"
    assert "SS3H10" in conv.library_models
    assert conv.library_models["SS3H10"][1]["BV"] == 100
    assert "SS3H10" in netlist


@requires_sim
def test_schottky_keyword_selects_generic():
    """value='schottky' is a type hint -> DefaultSchottky (mirrors 'diode')."""
    _setup()
    d = Part("Device", "D", value="schottky", ref="D1")
    Net("A").connect(d["A"])
    Net("K").connect(d["K"])
    conv = SpiceConverter(skidl_flat_view())
    conv.convert(strict=False)
    assert conv.model_provenance["D1"].name == "DefaultSchottky"
    assert conv.model_provenance["D1"].tier == "generic"


@requires_sim
def test_sim_device_schottky_hint_seeds_fallback(caplog):
    """Unknown diode name + Sim.Params + Sim.Device='SCHOTTKY' seeds the derived
    card from DefaultSchottky (EG=0.69 present), provenance recorded, and the
    fallback warning names the seed + effective BV (R2: never silently silicon)."""
    import logging as _logging

    _setup()
    d = Part("Device", "D", value="XYZ999", ref="D1")
    d.Sim_Device = "SCHOTTKY"
    d.Sim_Params = "BV=100 IBV=1e-4"
    Net("A").connect(d["A"])
    Net("K").connect(d["K"])
    conv = SpiceConverter(skidl_flat_view())
    with caplog.at_level(_logging.WARNING, logger="skidl.sim.converter"):
        conv.convert(strict=False)
    prov = conv.model_provenance["D1"]
    assert prov.tier == "generic" and prov.name == "XYZ999->DefaultSchottky"
    card = conv.derived_models["DefaultSchottky_D1"]
    assert card[0] == "D" and card[1]["EG"] == 0.69 and card[1]["BV"] == 100
    warn = next(r for r in caplog.records if "not in library" in r.message)
    assert "DefaultSchottky" in warn.message and "BV=100" in warn.message


@requires_sim
def test_schottky_prefix_table_seeds_fallback():
    """Unknown 'PMEG10020' + Sim.Params (no hint) -> the family-prefix table
    seeds DefaultSchottky; a no-prefix unknown ('XYZ999') stays silicon."""
    _setup()
    d1 = Part("Device", "D", value="PMEG10020", ref="D1")
    d1.Sim_Params = "BV=100"
    d2 = Part("Device", "D", value="XYZ999", ref="D2")
    d2.Sim_Params = "IS=1e-12"
    Net("A").connect(d1["A"], d2["A"])
    Net("K").connect(d1["K"], d2["K"])
    conv = SpiceConverter(skidl_flat_view())
    conv.convert(strict=False)
    assert conv.model_provenance["D1"].name == "PMEG10020->DefaultSchottky"
    assert conv.model_provenance["D2"].name == "XYZ999->DefaultDiode"


@requires_sim
def test_unknown_diode_without_overrides_still_hard_error():
    """No Sim.Params -> the unknown name stays a loud validation error; the
    Schottky-aware fallback must not have widened the silent path."""
    from skidl.sim.converter import SimulationValidationError

    _setup()
    d = Part("Device", "D", value="SS999", ref="D1")
    d.Sim_Device = "SCHOTTKY"
    Net("A").connect(d["A"])
    Net("K").connect(d["K"])
    with pytest.raises(SimulationValidationError):
        SpiceConverter(skidl_flat_view()).convert(strict=True)


def _reverse_bias_node(diode_value, volts):
    """Live helper: <volts> V reverse across a diode behind 100 ohm; returns the
    cathode-node voltage. ~volts = blocking; ~BV = breakdown conduction."""
    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value=str(volts))
    r1 = Part("Device", "R", value="100")
    d1 = Part("Device", "D", value=diode_value, ref="D1")
    Net("VS").connect(v1[1], r1[1])
    Net("KN").connect(r1[2], d1["K"])  # reverse: cathode to the + side
    Net("0").connect(v1[2], d1["A"])
    spice = SpiceConverter(skidl_flat_view()).convert(strict=True)
    import skidl.sim.simulator  # noqa: F401

    an = spice.simulator().operating_point()
    return float(an["KN"][0])


@requires_sim
def test_hv_schottky_blocks_llc_piv_where_ss14_breaks_down():
    """Live (the R1 repro, both directions): at the LLC's ~82 V PIV an SS3H10
    (BV=100) blocks -- node holds ~82 V, leakage-only -- while an SS14 (BV=40)
    conducts in reverse breakdown (node clamps toward ~40 V). The >40 V misuse
    must at least SIMULATE the breakdown; physics honesty, not a lint."""
    try:
        kn_hv = _reverse_bias_node("SS3H10", 82)
        kn_lv = _reverse_bias_node("SS14", 82)
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    assert kn_hv > 81.0, f"SS3H10 should block 82 V, node={kn_hv}"
    assert kn_lv < 50.0, f"SS14 must show breakdown at 82 V PIV, node={kn_lv}"


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
def test_negative_dc_source_simulates_negative():
    """Live: a VDC value="-1" across a resistor to GND holds the node at -1.0 V
    (the exact DiffAmp B4 repro), and get_current resolves the source branch by
    plain ref despite PySpice's v-prefixed lowercased branch name (R2)."""
    _setup()
    v1 = Part("Simulation_SPICE", "VDC", value="-1", ref="V1")
    r1 = Part("Device", "R", value="1k")
    Net("N").connect(v1[1], r1[1])
    Net("0").connect(v1[2], r1[2])

    spice = SpiceConverter(skidl_flat_view()).convert(strict=True)
    try:
        import skidl.sim.simulator  # noqa: F401
        from skidl.sim.simulator import SimulationResult

        sim = spice.simulator()
        an = sim.operating_point()
    except Exception as e:
        pytest.skip(f"ngspice not available: {type(e).__name__}: {str(e)[:80]}")
    n = float(an["N"][0])
    assert abs(n - (-1.0)) < 0.01, f"V(N)={n}"
    # R2: get_current by plain ref must resolve the source's branch current even
    # though PySpice names the branch "vv1" (lowercased, v-prefixed).
    result = SimulationResult(an, "dc_op")
    i = result.get_current("V1")
    i = i[0] if hasattr(i, "__len__") else i
    assert abs(abs(float(i)) - 1.0 / 1000.0) < 1e-6, f"I(V1)={i}"


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
