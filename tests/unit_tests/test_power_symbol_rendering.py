# -*- coding: utf-8 -*-

"""Power-symbol-first rendering (wired-render default).

Power nets must render as KiCad ``power:*`` symbols on EVERY render path (not
only ``auto_stub``): classified by ``drive == POWER`` or a power-name pattern,
excluded from the A* router, one symbol per pin, a cloned in-file ``(power)``
symbol for non-stock rail names, and exactly one project-wide ``PWR_FLAG`` per
undriven rail. See the wired-render-default plan, Phase 1.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from skidl import KICAD10, POWER, lib_search_paths, set_default_tool

# ---- real-KiCad-10 discovery (mirrors test_kicad10.py) --------------------
_KICAD_CLI_CANDIDATES = [
    r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
    "/usr/bin/kicad-cli",
    "/usr/local/bin/kicad-cli",
    shutil.which("kicad-cli") or "",
]
KICAD_CLI = next((c for c in _KICAD_CLI_CANDIDATES if c and Path(c).exists()), None)


def _kicad10_symbols_available():
    import skidl.tools.kicad10.lib as k10

    return bool(k10._discover_default_symbol_dirs("10")) or bool(
        os.environ.get("KICAD10_SYMBOL_DIR") or os.environ.get("KICAD_SYMBOL_DIR")
    )


requires_kicad10 = pytest.mark.skipif(
    not (KICAD_CLI and _kicad10_symbols_available()),
    reason="requires a real KiCad 10 install (kicad-cli + stock symbol libraries)",
)


def _use_kicad10():
    set_default_tool(KICAD10)
    lib_search_paths["kicad10"] = ["."] + __import__(
        "skidl.tools.kicad10.lib", fromlist=["default_lib_paths"]
    ).default_lib_paths()
    import builtins

    builtins.default_circuit.mini_reset()


# ---------------------------------------------------------------------------
# Unit: classification (no libs needed)
# ---------------------------------------------------------------------------


def test_mark_power_nets_classifies_drive_and_pattern():
    from skidl import Circuit, Net, Part
    from skidl.tools.kicad10.gen_schematic import mark_power_nets

    ckt = Circuit(name="cls")
    with ckt:
        rail = Net("VBIAS_28V")
        rail.drive = POWER  # non-stock name, classified via drive
        gnd = Net("GND")  # pattern-classified
        sig = Net("SIG")  # neither -> a signal net
        r1 = Part("Device", "R", value="1k")
        r2 = Part("Device", "R", value="1k")
        r1[1] += rail
        r1[2] += sig
        r2[1] += sig
        r2[2] += gnd

    mark_power_nets(ckt)
    assert getattr(rail, "_is_power_net", False) is True
    assert getattr(gnd, "_is_power_net", False) is True
    assert getattr(sig, "_is_power_net", False) is False
    # Power pins get stubbed (excluded from the A* router); signal pins do not.
    assert all(p.stub for p in rail.get_pins())
    assert all(not p.stub for p in sig.get_pins())


def test_mark_power_nets_flags_undriven_only():
    """A rail with no ``power_output`` pin needs a PWR_FLAG; a rail carrying a
    PWROUT pin (e.g. an explicit PWR_FLAG part / regulator output) does not."""
    from skidl import Circuit, Net, Part
    from skidl.tools.kicad10.gen_schematic import mark_power_nets

    ckt = Circuit(name="flg")
    with ckt:
        undriven = Net("VCC")
        driven = Net("VDD")
        r1 = Part("Device", "R", value="1k")
        r1[1] += undriven
        r1[2] += driven
        # PWR_FLAG's single pin is power_out -> marks `driven` as driven.
        flg = Part("power", "PWR_FLAG")
        flg[1] += driven

    mark_power_nets(ckt)
    assert getattr(undriven, "_needs_pwr_flag", None) is True
    assert getattr(driven, "_needs_pwr_flag", None) is False


# ---------------------------------------------------------------------------
# Render integration (needs real libs + kicad-cli)
# ---------------------------------------------------------------------------


def _render(build, top, tmp_path, **opts):
    from skidl import Circuit

    _use_kicad10()
    ckt = Circuit(name=top)
    build(ckt)
    out = tmp_path / top
    out.mkdir()
    render_opts = {"seed_placement": True, "auto_stub": False}
    render_opts.update(opts)
    ckt.generate_schematic(tool=KICAD10, filepath=str(out), top_name=top, **render_opts)
    text = "".join(
        p.read_text(encoding="utf-8") for p in sorted(out.glob("*.kicad_sch"))
    )
    return out, text


def _erc_error_types(out, top):
    sch = out / f"{top}.kicad_sch"
    rpt = out / f"{top}-erc.rpt"
    subprocess.run(
        [KICAD_CLI, "sch", "erc", "--output", str(rpt), "--severity-error", str(sch)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    txt = rpt.read_text(encoding="utf-8") if rpt.exists() else ""
    types = {}
    for m in re.finditer(r"\[(\w+)\]", txt):
        types[m.group(1)] = types.get(m.group(1), 0) + 1
    return types


def _divider(ckt):
    from skidl import Net, Part

    with ckt:
        vcc = Net("VCC"); vcc.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        mid = Net("MID")
        r1 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r2 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r1[1] += vcc; r1[2] += mid
        r2[1] += mid; r2[2] += gnd


def _custom_rail(ckt):
    from skidl import Net, Part

    with ckt:
        vbias = Net("VBIAS_28V"); vbias.drive = POWER
        gnd = Net("GND"); gnd.drive = POWER
        mid = Net("MID")
        r1 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r2 = Part("Device", "R", value="10k", footprint="Resistor_SMD:R_0603_1608Metric")
        r1[1] += vbias; r1[2] += mid
        r2[1] += mid; r2[2] += gnd


@requires_kicad10
def test_wired_render_power_nets_not_routed(tmp_path):
    """On the wired path, power nets render as power symbols with NO routed
    wires (they are excluded from A*), and ERC has zero power_pin_not_driven."""
    out, text = _render(_divider, "pwrdiv", tmp_path)
    # Power symbols for both rails present.
    assert 'lib_id "power:VCC"' in text
    assert 'lib_id "power:GND"' in text
    # Exactly one PWR_FLAG per undriven rail (2 rails here).
    assert len(re.findall(r'lib_id "power:PWR_FLAG"', text)) == 2
    types = _erc_error_types(out, "pwrdiv")
    assert types.get("power_pin_not_driven", 0) == 0, types


@requires_kicad10
def test_custom_rail_gets_infile_power_symbol(tmp_path):
    """A non-stock rail name (drive=POWER) gets a cloned in-file (power) symbol
    whose Value is the rail name -- not a plain global label."""
    out, text = _render(_custom_rail, "cust", tmp_path)
    assert 'symbol "power:VBIAS_28V"' in text  # cloned lib definition
    assert 'lib_id "power:VBIAS_28V"' in text  # instance
    assert 'property "Value" "VBIAS_28V"' in text  # value = rail name
    # The cloned definition carries the (power ...) flag, so KiCad treats it as a
    # real power symbol (global-by-name), not a plain global label.
    defn = text.split('symbol "power:VBIAS_28V"', 1)[1].split("(symbol", 1)[0]
    assert "(power" in defn
    assert 'global_label "VBIAS_28V"' not in text
    types = _erc_error_types(out, "cust")
    assert types.get("power_pin_not_driven", 0) == 0, types


@requires_kicad10
def test_one_pwr_flag_per_rail_project_wide(tmp_path):
    """Exactly ONE PWR_FLAG per undriven rail regardless of how many pins the
    rail has (power symbols connect globally by name, so one flag suffices), and
    every rail is still represented by at least one power symbol -- even when
    identical floating parts stack exactly (the fully-coincident cluster must not
    suppress its last representative)."""

    def build(ckt):
        from skidl import Net, Part

        with ckt:
            vcc = Net("VCC"); vcc.drive = POWER
            gnd = Net("GND"); gnd.drive = POWER
            for i in range(4):
                c = Part("Device", "C", value="100nF",
                         footprint="Capacitor_SMD:C_0603_1608Metric")
                c[1] += vcc
                c[2] += gnd

    out, text = _render(build, "manypin", tmp_path)
    # Each rail keeps at least one power symbol (never fully suppressed)...
    assert len(re.findall(r'lib_id "power:VCC"', text)) >= 1
    assert len(re.findall(r'lib_id "power:GND"', text)) >= 1
    # ...and exactly one PWR_FLAG per undriven rail (2 rails), project-wide.
    assert len(re.findall(r'lib_id "power:PWR_FLAG"', text)) == 2
    # And ERC sees both rails as driven.
    types = _erc_error_types(out, "manypin")
    assert types.get("power_pin_not_driven", 0) == 0, types
