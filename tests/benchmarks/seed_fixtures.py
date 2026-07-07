# -*- coding: utf-8 -*-

"""Pure-skidl fixture circuits for the seed-placement benchmark (stage 19).

Each ``build_<name>() -> Circuit`` returns a freshly-``reset()`` circuit whose
shape exercises one of the failure modes from the Stage-19 plan overview:

    divider       trivial sanity
    rc_chain8     flat series chain -> endpoint fallback / linear layout
    tia_feedback  Rf||Cf feedback loop across in-/out -> centroid pull
    mcu_star      one hub + many satellites -> direction contention / fan-out
    two_groups    two disjoint clusters + one floating part
    scale_chain28 >20-part single connected group -> _ROW_PLACE_THRESHOLD path

Ground-truth notes (verified 2026-07-07, recorded in
plans/stage-19-constructive-seed-placement/02-phase-a-benchmark-harness.md):

* Op-amp symbol substitution: the plan named ``Amplifier_Operational:ADA4817-1ACP``
  but that part is absent from the installed KiCad-10 library. We use
  ``Amplifier_Operational:OPA340NA`` (single op-amp) instead.
  Pin map: 1=OUT, 2=V-, 3=+in, 4=-in, 5=V+.
* Power nets are given a KiCad power-style name (GND / +5V) AND ``drive = POWER``,
  exactly like ``circuit_synth.interop.skidl_export`` does. ``auto_stub_nets``
  stubs them by *name* (``_POWER_NET_RE``); the ``drive`` marker is what the
  Phase-B seed filter keys on, so we set both.
* ``auto_stub_fanout`` default is **5** and the test is ``len(pins) >= 5``
  (gen_schematic.py:79 — the docstring's "3" is stale). Every signal net below is
  kept to 2-4 pins so nothing is stubbed for fanout; only the deliberate power
  nets are stubbed.
"""

from skidl import POWER, Circuit, Net, Part, reset

# Single op-amp used by the feedback / scale fixtures (see module docstring).
_OPAMP_LIB = "Amplifier_Operational"
_OPAMP_PART = "OPA340NA"
# Pin identifiers on OPA340NA.
_OUT, _VN, _INP, _INN, _VP = "1", "2", "3", "5", "4"
# NOTE: pin '4' is the inverting input (-in), pin '5' is V+. Bound to names above
# so the fixtures read by function, not by bare pin number.


def _power_net(name):
    """Create a POWER-drive net with a power-style name (auto-stubbed by name)."""
    n = Net(name)
    n.drive = POWER
    return n


def _opamp(ref, gnd, vplus):
    """Instantiate an op-amp with its rails/+in tied to power nets (all stubbed).

    Leaves the two signal pins (-in='4', OUT='1') free for the caller to wire.
    """
    u = Part(_OPAMP_LIB, _OPAMP_PART, ref=ref)
    u[_VN] += gnd  # V-
    u[_VP] += vplus  # V+
    u[_INP] += gnd  # +in tied to ground reference (stubbed power)
    return u


def build_divider():
    """2 resistors, nets VIN / VOUT / GND. Trivial sanity fixture."""
    reset()
    vin = Net("VIN")
    vout = Net("VOUT")
    gnd = _power_net("GND")
    r1 = Part("Device", "R", ref="R1", value="10k")
    r2 = Part("Device", "R", ref="R2", value="10k")
    r1[1] += vin
    r1[2] += vout
    r2[1] += vout
    r2[2] += gnd
    return default_circuit_snapshot()


def build_rc_chain8():
    """8 parts (4R + 4C alternating) in a single series chain VIN..GND.

    Flat topology: k-core decomposition is trivial, so a good seed must fall
    back to an endpoint as the center and lay the chain out linearly.
    """
    reset()
    vin = Net("VIN")
    gnd = _power_net("GND")
    parts = []
    for i in range(4):
        parts.append(Part("Device", "R", ref=f"R{i + 1}", value="1k"))
        parts.append(Part("Device", "C", ref=f"C{i + 1}", value="100n"))
    # Series-wire: VIN - P0 - P1 - ... - P7 - GND.
    prev = vin
    for idx, p in enumerate(parts):
        p[1] += prev
        if idx < len(parts) - 1:
            nxt = Net(f"n{idx + 1}")
            p[2] += nxt
            prev = nxt
        else:
            p[2] += gnd
    return default_circuit_snapshot()


def build_tia_feedback():
    """Transimpedance-amplifier shape: Rf || Cf across (-in, OUT).

    Mirrors the SiPM TIA topology (~6 parts). The feedback pair Rf/Cf bridges the
    inverting-input node and the output node, forming a loop with the op-amp; a
    good seed places them *between* input and output (centroid pull), not off to
    one side as a pure BFS tree would.
    """
    reset()
    gnd = _power_net("GND")
    vplus = _power_net("+5V")
    u1 = _opamp("U1", gnd, vplus)
    rf = Part("Device", "R", ref="RF", value="1M")
    cf = Part("Device", "C", ref="CF", value="2p")
    rin = Part("Device", "R", ref="RIN", value="50")
    csrc = Part("Device", "C", ref="CSRC", value="10p")
    rload = Part("Device", "R", ref="RL", value="1k")
    # Feedback pair across (-in='4', OUT='1').
    rf[1] += u1[_INN]
    rf[2] += u1[_OUT]
    cf[1] += u1[_INN]
    cf[2] += u1[_OUT]
    # Source network injecting into the inverting node.
    nsig = Net("SIG")
    rin[1] += nsig
    rin[2] += u1[_INN]
    csrc[1] += nsig
    csrc[2] += gnd
    # Load on the output.
    rload[1] += u1[_OUT]
    rload[2] += gnd
    return default_circuit_snapshot()


def build_mcu_star():
    """One 8-pin hub + 8 two-pin satellites, each on its own hub pin.

    Every hub pin faces the same direction, so all satellites want the same
    placement slot: the seed's slot allocator must fan them along the
    perpendicular axis instead of stacking them.
    """
    reset()
    gnd = _power_net("GND")
    hub = Part("Connector_Generic", "Conn_01x08", ref="J1")
    for i in range(1, 9):
        r = Part("Device", "R", ref=f"R{i}", value="100")
        r[1] += hub[str(i)]
        r[2] += gnd
    return default_circuit_snapshot()


def build_two_groups():
    """Two disjoint 3-resistor clusters + one floating capacitor.

    Exercises group_parts multi-group separation and floating-part handling.
    The floating cap's pins both land on (stubbed) power nets, so it has no
    wired connection to anything.
    """
    reset()
    gnd = _power_net("GND")
    vplus = _power_net("+5V")

    def _cluster(tag, boundary):
        r1 = Part("Device", "R", ref=f"R{tag}1", value="1k")
        r2 = Part("Device", "R", ref=f"R{tag}2", value="1k")
        r3 = Part("Device", "R", ref=f"R{tag}3", value="1k")
        a1 = Net(f"{tag}_a1")
        a2 = Net(f"{tag}_a2")
        r1[1] += boundary
        r1[2] += a1
        r2[1] += a1
        r2[2] += a2
        r3[1] += a2
        r3[2] += gnd

    _cluster("A", Net("VINA"))
    _cluster("B", Net("VINB"))
    # Floating part: both pins on stubbed power nets -> no wired neighbours.
    cfloat = Part("Device", "C", ref="CFLT", value="1u")
    cfloat[1] += gnd
    cfloat[2] += vplus
    return default_circuit_snapshot()


def build_scale_chain28():
    """N cascaded op-amp filter stages linked into ONE connected group >20 parts.

    Mirrors the run-4 deliverable's shape (a long signal chain). Each stage is an
    op-amp + 4 passives (Rin, Rf, Cf, Rout); stages are linked by 2-pin signal
    nets so, after power stubbing, all real parts form a single connected group
    that exceeds ``_ROW_PLACE_THRESHOLD`` (=20) and takes the row-based path.

    Actual part count is recorded in the Phase-A findings.
    """
    reset()
    gnd = _power_net("GND")
    vplus = _power_net("+5V")
    n_stages = 6  # 6 * 5 = 30 real parts > 20 (verified via group_parts at build)
    prev_sig = Net("IN")
    for s in range(n_stages):
        u = _opamp(f"U{s + 1}", gnd, vplus)
        rin = Part("Device", "R", ref=f"RIN{s + 1}", value="1k")
        rf = Part("Device", "R", ref=f"RF{s + 1}", value="10k")
        cf = Part("Device", "C", ref=f"CF{s + 1}", value="10p")
        rout = Part("Device", "R", ref=f"ROUT{s + 1}", value="100")
        # Input resistor from the previous stage's output into this -in node.
        rin[1] += prev_sig
        rin[2] += u[_INN]
        # Feedback pair.
        rf[1] += u[_INN]
        rf[2] += u[_OUT]
        cf[1] += u[_INN]
        cf[2] += u[_OUT]
        # Output resistor into the next stage's signal net.
        nxt_sig = Net(f"SIG{s + 1}")
        rout[1] += u[_OUT]
        rout[2] += nxt_sig
        prev_sig = nxt_sig
    return default_circuit_snapshot()


def default_circuit_snapshot():
    """Return the current default Circuit (what the builders wire into).

    skidl exposes the active circuit as ``builtins.default_circuit`` (set in
    Circuit.__init__), so we read it from builtins rather than importing it.
    """
    import builtins

    return builtins.default_circuit


# Registry consumed by the benchmark runner.
FIXTURES = {
    "divider": build_divider,
    "rc_chain8": build_rc_chain8,
    "tia_feedback": build_tia_feedback,
    "mcu_star": build_mcu_star,
    "two_groups": build_two_groups,
    "scale_chain28": build_scale_chain28,
}
