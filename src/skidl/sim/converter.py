"""
SpiceConverter: Converts circuit-synth designs to PySpice format.

This module handles the translation from circuit-synth components and nets
to SPICE netlists that can be simulated with PySpice/ngspice.

External vendor models are attached in tiers: an explicit ``Sim.Library``, the
local MPN store, then automatic resolution through the corpus library index.
That last tier does NOT ``.include`` the model's whole file -- one malformed
line anywhere in a vendor library makes ngspice reject every model defined in
it, so ``_emit_minimal_decks`` includes an extracted deck holding only the
models the netlist needs (see ``model_deck``). Explicit ``Sim.Library`` paths
keep whole-file includes, and ``SKIDL_SIM_MINIMAL_DECK=0`` reverts everything.
"""

import hashlib
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """Which model tier a device's SPICE ``.model`` was resolved from.

    Recorded per device so callers can see -- and log -- that a part simulated
    with datasheet-fit params, a textbook generic, or an external vendor model,
    rather than a generic being silently passed off as the real part.

    ``vendor_lib`` covers a model resolved from the corpus library index or the
    local MPN store (real silicon someone catalogued). ``user_lib`` is distinct
    (F2): a model attached from an explicit ``Sim.Library`` path OUTSIDE the
    configured corpus -- i.e. an author's own hand-written macromodel. The two are
    kept separate so a *behavioral guess* is never presented with the same
    authority as a genuine vendor model. (A user file dropped INSIDE the corpus
    tree cannot be distinguished and reports ``vendor_lib`` -- a known limit.)
    """

    ref: str
    kind: str  # diode | bjt | mosfet
    tier: str  # datasheet_fit | generic | vendor_lib | user_lib | unresolved
    name: str  # the resolved model/base name
    overridden: bool = False  # True if Sim.Params overlaid a derived card
    source: str = ""  # provenance of a vendor_lib model: sim_library | local_store
    # For a subckt attach, how much confidence the terminal (Sim.Pins) mapping
    # carries (F3): "" not-applicable; "named" the subckt's nodes are self-
    # descriptive; "heuristic" the nodes are numeric so D/G/S identity is a guess
    # (verify with `find_spice_model <name> --verify-terminals`).
    pin_map_confidence: str = ""


try:
    from PySpice.Spice.Netlist import Circuit as SpiceCircuit
    from PySpice.Unit import *

    PYSPICE_AVAILABLE = True
except ImportError:
    PYSPICE_AVAILABLE = False


class SimulationValidationError(ValueError):
    """Raised when a circuit cannot be safely simulated.

    Carries the full list of problems (``.problems``) so the caller sees every
    issue at once, rather than discovering them one ngspice crash at a time. This
    replaces the old behaviour where unknown components were silently skipped,
    yielding a wrong-but-"successful" simulation.
    """

    def __init__(self, problems):
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"circuit is not valid for simulation:\n{body}")


class SpiceConverter:
    """Converts circuit-synth circuits to PySpice format."""

    # Built-in generic SPICE model cards, keyed by model name. Each entry is
    # (ngspice device type, params dict). These back the ``Default*`` names the
    # device handlers assign, so any diode/BJT/MOSFET is simulatable out of the box
    # without a vendor model; a device's ``value=`` may also name one of these
    # built-ins directly. Only models a circuit actually uses are emitted (see
    # ``_emit_models``). Params are deliberately generic (textbook silicon values).
    GENERIC_MODELS = {
        "DefaultDiode": ("D", {"IS": 1e-14, "RS": 0.1, "N": 1.0, "CJO": 2e-12}),
        # Generic Schottky: BV=40 V (a generic, NOT a rating -- above ~40 V PIV
        # use a curated HV entry like SS3H10/STPS3150 or override BV). EG/XTI
        # make derived cards seeded from it temperature-honest Schottkys.
        "DefaultSchottky": (
            "D",
            {
                "IS": 1e-6,
                "RS": 0.05,
                "N": 1.05,
                "CJO": 100e-12,
                "BV": 40,
                "IBV": 1e-3,
                "EG": 0.69,
                "XTI": 2,
            },
        ),
        "DefaultNPN": ("NPN", {"BF": 100, "IS": 1e-14, "VAF": 100}),
        "DefaultPNP": ("PNP", {"BF": 100, "IS": 1e-14, "VAF": 100}),
        "DefaultNMOS": ("NMOS", {"VTO": 1.0, "KP": 2e-5, "LAMBDA": 0.02}),
        "DefaultPMOS": ("PMOS", {"VTO": -1.0, "KP": 2e-5, "LAMBDA": 0.02}),
        # A generic power NMOS: higher VTO/KP than the small-signal DefaultNMOS,
        # for a switch that must actually pass amps. No body-diode/Coss metadata
        # (those live on the curated datasheet-fit entries, or Sim.Params
        # COSS=/BODY=1) -- this is just the conduction model.
        "DefaultPowerNMOS": ("NMOS", {"VTO": 3.5, "KP": 20.0, "LAMBDA": 0.01}),
    }

    # Per-kind generic fallback: an unresolved model NAME carrying Sim.Params
    # degrades to this generic + the overrides (rather than a hard error), so an
    # unknown part can be param-fitted into a usable model (bug #11).
    _KIND_GENERIC = {
        "diode": "DefaultDiode",
        "bjt": "DefaultNPN",
        "mosfet": "DefaultNMOS",
    }

    # A bare device-type keyword in ``value`` selects the matching generic model,
    # so ``value="pnp"`` is treated as a type hint rather than a literal model name.
    _TYPE_KEYWORD_MODELS = {
        "npn": "DefaultNPN",
        "pnp": "DefaultPNP",
        "nmos": "DefaultNMOS",
        "pmos": "DefaultPMOS",
        "powernmos": "DefaultPowerNMOS",
        "diode": "DefaultDiode",
        "schottky": "DefaultSchottky",
    }

    # Well-known Schottky part-number family prefixes (prefix + digit). Used ONLY
    # to pick which generic seeds the Sim.Params fallback for an unknown diode
    # name -- never to invent a datasheet fit. A silicon seed on a Schottky is
    # silently wrong (Vf ~0.7 V vs ~0.4 V; LLC E2E finding R2).
    _SCHOTTKY_PREFIX_RE = re.compile(
        r"^(?:SS|STPS|PMEG|MBR|SK|SB|B[23])\d", re.IGNORECASE
    )

    def __init__(self, circuit_synth_circuit):
        self.circuit = circuit_synth_circuit
        self.spice_circuit = None
        self.voltage_sources = []
        self.node_map = {}
        # Nets driven by an explicit source component (Device:V, Simulation_SPICE:V*,
        # ...). _add_power_sources skips these so the net-name heuristic never adds a
        # second, conflicting supply on a net an explicit source already drives.
        self.driven_nets = set()
        # Model names referenced by diode/BJT/MOSFET devices; a matching .model card
        # is emitted for each that resolves to a built-in generic (see _emit_models).
        self.used_models = set()
        # Per-device model cards synthesized from Sim.Params overrides, keyed by a
        # derived name ({base}_{ref}) so two parts overriding the same base model
        # do not collide. Each value is (device_type, params). Emitted alongside
        # the built-in generics in _emit_models.
        self.derived_models = {}
        # Datasheet-fit model cards pulled from the built-in ModelLibrary and
        # actually referenced by a device, keyed by model name -> (device_type,
        # params). Emitted in _emit_models (GENERIC_MODELS covers only generics).
        self.library_models = {}
        # ref -> ResolvedModel: which tier each active device's model came from.
        self.model_provenance = {}
        # Absolute paths of external .lib/.sub files already `.include`d (dedup).
        self.included_libs = set()
        # Corpus-resolved models whose include is deferred to _emit_minimal_decks:
        # source file path -> set of model names needed from it. See that method
        # for why auto-resolved hits get a minimal deck instead of the whole file.
        self._mindeck_needs = {}
        # First non-empty Sim.Compat across components (e.g. "psa" for a vendor
        # PSpice lib) -> the ngspice dialect the simulator should select. Resolved
        # in convert(); a disagreement between components is a validate() error.
        self.compat_hint = None
        # After _flatten on a hierarchical circuit: flat_ref -> (subcircuit_name,
        # original_ref) for any ref that had to be uniquified. Empty otherwise.
        # Keeps model provenance / error messages traceable back to the source.
        self.flattened_ref_map = {}
        # node name -> set of excluded (Sim.Enable=0) refs that touched it. Lets
        # the floating-node tie (F4) name the excluded part that stranded a node.
        self._excluded_node_refs = {}

    class _FlatCircuit:
        """A read-only, flattened view of a hierarchical circuit.

        The converter only reads ``.components`` (dict ref->component),
        ``.nets`` (dict name->net) and ``.name`` off its circuit, so this thin
        view is all conversion/validation need after flattening.
        """

        def __init__(self, name, components, nets):
            self.name = name
            self.components = components
            self.nets = nets

    @staticmethod
    def _iter_values(container):
        """Values of a dict-or-iterable container (components/nets come as either)."""
        if hasattr(container, "values"):
            return list(container.values())
        if hasattr(container, "__iter__"):
            return list(container)
        return []

    def _flatten(self, circuit):
        """Merge a hierarchical circuit's components and nets into one view.

        Returns ``circuit`` unchanged when it has no subcircuits. Otherwise walks
        the subcircuit tree depth-first and returns a :class:`_FlatCircuit` with:

        - **components** merged into one dict. Refs are normally already unique
          across the hierarchy (the reference manager uniquifies at construction),
          but a genuine collision (two *distinct* components sharing a ref) is
          resolved by renaming the later one -- which propagates to its pins'
          string form (read live), so node extraction stays correct.
        - **nets** merged by name. A net shared across sheets is the *same object*
          (same name) so it merges cleanly; two *distinct* nets sharing a name
          would be conflated into one node, so that case is logged as a warning
          (a name-scoping limitation, not introduced by flattening).
        """
        if not getattr(circuit, "_subcircuits", None):
            return circuit  # already flat -- leave the existing path untouched

        components = {}
        nets = {}
        name_collisions = []
        self.flattened_ref_map = {}

        def visit(circ, path):
            for comp in self._iter_values(getattr(circ, "components", {})):
                ref = getattr(comp, "ref", None)
                if ref is None:
                    continue
                if ref in components and components[ref] is not comp:
                    new_ref = f"{ref}_{len(components)}"
                    while new_ref in components:
                        new_ref = f"{new_ref}_x"
                    self.flattened_ref_map[new_ref] = (circ.name, ref)
                    comp.ref = new_ref  # live -> updates str(pin) used for matching
                    ref = new_ref
                components[ref] = comp
            for net in self._iter_values(getattr(circ, "nets", {})):
                name = getattr(net, "name", str(net))
                existing = nets.get(name)
                if existing is None:
                    nets[name] = net
                elif existing is not net:
                    name_collisions.append(name)
            for sub in getattr(circ, "_subcircuits", []) or []:
                visit(sub, path + [getattr(sub, "name", "?")])

        visit(circuit, [getattr(circuit, "name", "Circuit")])

        if name_collisions:
            logger.warning(
                "Flattening hierarchy: distinct nets share name(s) %s and will "
                "be treated as one SPICE node -- rename them to disambiguate.",
                sorted(set(name_collisions)),
            )
        logger.debug(
            "Flattened hierarchy for simulation: %d component(s), %d net(s)",
            len(components),
            len(nets),
        )
        return self._FlatCircuit(getattr(circuit, "name", "Circuit"), components, nets)

    def convert(self, strict: bool = True) -> "SpiceCircuit":
        """Convert circuit-synth circuit to PySpice circuit.

        Args:
            strict: When True (default), validate the circuit first and raise
                ``SimulationValidationError`` if anything would produce a
                wrong-but-"successful" simulation (unknown component, floating
                node, no source, under-connected op-amp). When False, fall back to
                the lenient path that warns and skips unknown components -- useful
                for exploratory conversion of partial circuits.
        """
        if not PYSPICE_AVAILABLE:
            raise ImportError("PySpice not available")

        # Hierarchical designs place their components/nets in subcircuits; the
        # converter iterates only self.circuit, so flatten first. No-op for a
        # flat circuit. (See _flatten for the identity/collision handling.)
        self.circuit = self._flatten(self.circuit)

        if strict:
            self.validate()

        # Resolve the requested ngspice dialect (Sim.Compat) for the simulator.
        self.compat_hint = self._resolve_compat_hint()

        # Create PySpice circuit
        circuit_name = getattr(self.circuit, "name", "Circuit")
        self.spice_circuit = SpiceCircuit(circuit_name)

        # Map circuit-synth nets to SPICE nodes
        self._map_nodes()

        # Add components to SPICE circuit
        self._add_components()

        # Turn the corpus-resolved models deferred by _emit_external into minimal
        # `.include`d decks (one per source file). Must run after _add_components
        # so every model needed from a file is known before its deck is built.
        self._emit_minimal_decks()

        # Emit .model cards for the semiconductor models the components referenced.
        self._emit_models()

        # Add power sources (voltage/current sources)
        self._add_power_sources()

        # Netlist conditioning: tie any DC-floating node (e.g. a node stranded by
        # an excluded Sim.Enable=0 part) so the op-point isn't singular (F4). Runs
        # last, on the final device graph.
        self._tie_floating_nodes()

        return self.spice_circuit

    def _map_nodes(self):
        """Create mapping from circuit-synth nets to SPICE node names."""
        self.node_map = {}

        # Handle both dict and list formats for nets
        if hasattr(self.circuit.nets, "values"):
            # Dict format: {name: net_object}
            nets_to_process = self.circuit.nets.values()
        elif hasattr(self.circuit.nets, "__iter__"):
            # List format: [net_object, ...]
            nets_to_process = self.circuit.nets
        else:
            logger.error("Unknown nets format")
            return

        # Map GND net to SPICE ground
        for net in nets_to_process:
            net_name = getattr(net, "name", str(net))
            if net_name.upper() in ["GND", "GROUND", "VSS"]:
                self.node_map[net_name] = self.spice_circuit.gnd
            else:
                self.node_map[net_name] = net_name

    def _add_components(self):
        """Add circuit-synth components to SPICE circuit."""
        # circuit.components is a dict {ref: component}; iterating it directly
        # yields refs (strings), so pull the component objects out explicitly.
        components = self.circuit.components
        if hasattr(components, "values"):
            components = components.values()
        for component in components:
            self._add_component(component)

    @staticmethod
    def _classify(symbol: str) -> Optional[str]:
        """Map a KiCad symbol to a SPICE primitive kind, or None if unrecognized.

        Single source of truth shared by ``_add_component`` (dispatch) and
        ``validate`` (so validation and conversion never disagree about what is
        simulatable). Source checks come before the op-amp heuristic so a source
        symbol is never misclassified. Explicit SPICE sources use KiCad's real
        Simulation_SPICE library (VDC/VSIN/IDC/...); "Device:V"/"Device:I" are
        exact aliases (not real KiCad symbols but sometimes referenced by docs) --
        matched exactly so they do not also swallow "Device:Varistor".
        """
        if not symbol:
            return None
        if "Device:R" in symbol:
            return "resistor"
        if "Device:C" in symbol:
            return "capacitor"
        if "Device:L" in symbol:
            return "inductor"
        if "Device:D" in symbol or "Diode:" in symbol:
            return "diode"
        if (
            symbol.startswith("Simulation_SPICE:V")
            or "Reference_Voltage:" in symbol
            or symbol == "Device:V"
        ):
            return "voltage_source"
        if (
            symbol.startswith("Simulation_SPICE:I")
            or "Reference_Current:" in symbol
            or symbol == "Device:I"
        ):
            return "current_source"
        # Linear regulators before the op-amp heuristic: many regulator names
        # contain "lm" (LM317/LM1117) and would otherwise be misread as op-amps.
        if "Regulator_Linear:" in symbol:
            return "ldo"
        # Switching regulators: topology (buck vs boost) can't be read from the
        # symbol reliably, so classify to a pseudo-kind validate() turns into an
        # actionable "set Sim.Device=BUCK/BOOST" error (explicit beats guessing).
        if "Regulator_Switching:" in symbol:
            return "switcher_unknown"
        # Transformers before the op-amp heuristic (defensive); matches every
        # Device:Transformer_* variant, though only 1P_1S pin names (AA/AB/SA/SB)
        # resolve -- multi-winding parts fail terminal resolution with a clear error.
        if "Device:Transformer" in symbol:
            return "transformer"
        if any(x in symbol.lower() for x in ["op", "amp", "lm", "tl"]):
            return "opamp"
        if "Transistor_BJT:" in symbol or "Device:Q" in symbol:
            return "bjt"
        if "Transistor_FET:" in symbol or "Device:M" in symbol:
            return "mosfet"
        return None

    # KiCad ``Sim.Device`` device tokens -> our SPICE primitive kinds. Lets a
    # simulation-only stand-in ride on any symbol (the schematic keeps its real
    # symbol/footprint; only the simulation model is redirected). SUBCKT is left
    # unmapped here (external-model attach is Stage 9.3).
    _SIM_DEVICE_KINDS = {
        "R": "resistor",
        "C": "capacitor",
        "L": "inductor",
        "D": "diode",
        "SCHOTTKY": "diode",
        "NPN": "bjt",
        "PNP": "bjt",
        "NMOS": "mosfet",
        "PMOS": "mosfet",
        "V": "voltage_source",
        "I": "current_source",
        "LDO": "ldo",
        "BUCK": "buck",
        "BOOST": "boost",
        "FLYBACK": "flyback",
        "HALFBRIDGE": "halfbridge",
        "LLC": "halfbridge",
        # Multi-switch synchronous bidirectional families (Stage 27, Route B).
        # Each replaces only the switch stage; the user's L/caps/divider stay real.
        "BUCKBOOST4": "buckboost4",
        "INVBUCKBOOST": "invbuckboost",
        "SEPIC": "sepic",
        "CUK": "cuk",
        # Behavioral CLOSED-LOOP peak-current-mode controller (Stage 29.1). Unlike
        # the open-loop switch macromodels above (duty is swept) and the averaged
        # 28.D loop model (small-signal .ac), this generates the real gate from
        # feedback and regulates a live rail cycle-accurately in .tran.
        "CMCONTROLLER": "cmcontroller",
        "TRANSFORMER": "transformer",
        "XFMR": "transformer",
        # Corpus-independent behavioral logic primitives (Stage: DPSG WS2). These
        # give mixed-signal designs a simulatable digital path without depending
        # on the un-runnable corpus digital models (XSPICE d_*, PSpice U-device);
        # realized as ngspice-native B-sources + switch-held memory, no bridges.
        "DFF": "dff",
        "TFF": "tff",
        "DLATCH": "dlatch",
        # Triggered-breakdown / negative-resistance primitive (F5): one behavioral
        # smooth-conductance switch covering avalanche transistors, spark gaps and
        # SCR/DIAC-like discharge devices ngspice can't model natively. Aliases map
        # to the same emitter; terminal count picks externally- vs self-triggered.
        "TRIGSW": "trigsw",
        "SPARKGAP": "trigsw",
        "DIAC": "trigsw",
        "AVSW": "trigsw",
        "NOT": "gate",
        "INV": "gate",
        "BUF": "gate",
        "AND": "gate",
        "NAND": "gate",
        "OR": "gate",
        "NOR": "gate",
        "XOR": "gate",
        "XNOR": "gate",
    }

    # ``Sim.Device`` tokens that map to the combinational-gate emitter.
    _GATE_OPS = {"NOT", "INV", "BUF", "AND", "NAND", "OR", "NOR", "XOR", "XNOR"}

    @staticmethod
    def _sim_props(component) -> dict:
        """KiCad ``Sim.*`` fields as ``{lowercased suffix: value}`` (empty if none).

        Reads them off ``_extra_fields`` like the waveform params. Accepts both the
        native dotted spelling (``Sim.Enable``) and an underscore fallback
        (``Sim_Enable``) in case a flow sanitizes the dot away.
        """
        extra = getattr(component, "_extra_fields", None)
        if not isinstance(extra, dict):
            return {}
        out = {}
        for k, v in extra.items():
            low = str(k).lower()
            if low.startswith("sim.") or low.startswith("sim_"):
                out[low[4:]] = v
        return out

    def _sim_excluded(self, component) -> bool:
        """True if ``Sim.Enable`` marks the component out of simulation.

        KiCad uses ``Sim.Enable="0"`` to keep a part on the schematic but exclude
        it from simulation. Excluded parts are skipped by both conversion and
        validation (and their pins do not count toward net connectivity).
        """
        val = self._sim_props(component).get("enable", None)
        if val is None:
            return False
        return str(val).strip().lower() in ("0", "false", "no", "off")

    def _distinct_compat_values(self) -> list:
        """Distinct non-empty ``Sim.Compat`` values across components (sorted)."""
        values = set()
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            val = self._sim_props(component).get("compat")
            if val is not None and str(val).strip():
                values.add(str(val).strip())
        return sorted(values)

    def _resolve_compat_hint(self) -> Optional[str]:
        """The single ``Sim.Compat`` dialect requested by the schematic, or None.

        A disagreement between components is caught in ``validate()``; here we just
        take the first value so conversion still proceeds in the lenient path.
        """
        values = self._distinct_compat_values()
        return values[0] if values else None

    def _kind(self, component) -> Optional[str]:
        """SPICE primitive kind for a component: ``Sim.Device`` wins over the symbol.

        Returns None for an unrecognized ``Sim.Device`` token or an unmapped symbol
        (validation reports it, naming the offending token/symbol).
        """
        device = self._sim_props(component).get("device", None)
        if device:
            return self._SIM_DEVICE_KINDS.get(str(device).strip().upper())
        return self._classify(self._attr(component, "symbol", ""))

    def _add_component(self, component):
        """Add a single component to the SPICE circuit."""
        symbol = getattr(component, "symbol", "")
        ref = getattr(component, "ref", "X")
        value = getattr(component, "value", None)

        if self._sim_excluded(component):
            # Remember which nodes an excluded part touched, so if excluding it
            # strands a neighbor node (F4) the tie warning can name the culprit.
            for node in self._symbol_pin_nodes(component).values():
                self._excluded_node_refs.setdefault(str(node), set()).add(str(ref))
            logger.debug(f"{ref}: excluded from simulation (Sim.Enable=0)")
            return

        # An external vendor model (Sim.Library) supersedes the built-in handlers:
        # attach the .lib/.subckt directly (Stage 9.3).
        if self._sim_props(component).get("library"):
            lib = self._sim_props(component).get("library")
            path = self._resolve_lib_path(lib)
            if not path or not os.path.exists(path):
                # A hardcoded (often absolute, cross-checkout) Sim.Library path
                # that no longer exists must NOT hard-fail when the corpus can
                # cover it (DPSG A4): warn and auto-resolve by value/Sim.Name.
                hit = self._library_index_hit(component)
                if hit is not None:
                    logger.warning(
                        f"{ref}: Sim.Library path not found ({lib}); "
                        f"auto-resolved '{hit.name}' from the corpus instead. "
                        f"Prefer value=\"{hit.name}\" (drop the hardcoded path)."
                    )
                    self._add_index_model(component, ref, hit)
                    return
            self._add_external_model(component, ref)
            return

        # A model in the local MPN store is attached like an implicit Sim.Library
        # (Stage 9.4), above the datasheet-fit/generic tiers.
        store_path = self._store_lib_for(component)
        if store_path:
            self._add_store_model(component, ref, store_path)
            return

        # A model discovered in a configured external library index (e.g. the
        # KiCad-Spice-Library) is attached like an implicit Sim.Library. It sits
        # BELOW curated datasheet_fit (the corpus is broad but unvetted) and only
        # fires for real named parts we'd otherwise model with a generic/ideal;
        # Sim.Prefer="library" flips it above the built-ins. Inert when no
        # SKIDL_SPICE_LIB_PATH is configured.
        index_hit = self._library_index_hit(component)
        if index_hit is not None:
            self._add_index_model(component, ref, index_hit)
            return

        handlers = {
            "resistor": self._add_resistor,
            "capacitor": self._add_capacitor,
            "inductor": self._add_inductor,
            "diode": self._add_diode,
            "voltage_source": self._add_voltage_source,
            "current_source": self._add_current_source,
            "opamp": self._add_opamp,
            "ldo": self._add_ldo,
            "buck": self._add_buck,
            "boost": self._add_boost,
            "flyback": self._add_flyback,
            "halfbridge": self._add_halfbridge,
            "buckboost4": self._add_buckboost4,
            "invbuckboost": self._add_invbuckboost,
            "sepic": self._add_sepic,
            "cuk": self._add_cuk,
            "cmcontroller": self._add_cmcontroller,
            "transformer": self._add_transformer,
            "bjt": self._add_bjt_transistor,
            "mosfet": self._add_mosfet,
            "dff": self._add_dff,
            "tff": self._add_tff,
            "dlatch": self._add_dlatch,
            "trigsw": self._add_trigsw,
            "gate": self._add_gate,
        }
        handler = handlers.get(self._kind(component))
        if handler is None:
            logger.warning(f"Unknown component type: {symbol} - skipping")
            return
        handler(component, ref, value)

    def _add_resistor(self, component, ref: str, value: str):
        """Add resistor to SPICE circuit."""
        # Get connected nodes
        nodes = self._get_component_nodes(component)
        if len(nodes) < 2:
            logger.warning(f"Resistor {ref} needs 2 connections, got {len(nodes)}")
            return

        # Convert value to SPICE format
        spice_value = self._convert_value_to_spice(value, "R")

        # Add to SPICE circuit
        self.spice_circuit.R(ref, nodes[0], nodes[1], spice_value)
        logger.debug(f"Added resistor {ref}: {nodes[0]} -> {nodes[1]} = {spice_value}")

    def _add_capacitor(self, component, ref: str, value: str):
        """Add capacitor to SPICE circuit."""
        nodes = self._get_component_nodes(component)
        if len(nodes) < 2:
            logger.warning(f"Capacitor {ref} needs 2 connections, got {len(nodes)}")
            return

        spice_value = self._convert_value_to_spice(value, "C")
        self.spice_circuit.C(ref, nodes[0], nodes[1], spice_value)
        logger.debug(f"Added capacitor {ref}: {nodes[0]} -> {nodes[1]} = {spice_value}")

    def _add_inductor(self, component, ref: str, value: str):
        """Add inductor to SPICE circuit."""
        nodes = self._get_component_nodes(component)
        if len(nodes) < 2:
            logger.warning(f"Inductor {ref} needs 2 connections, got {len(nodes)}")
            return

        spice_value = self._convert_value_to_spice(value, "L")
        self.spice_circuit.L(ref, nodes[0], nodes[1], spice_value)
        logger.debug(f"Added inductor {ref}: {nodes[0]} -> {nodes[1]} = {spice_value}")

    def _device_model_name(self, component) -> Optional[str]:
        """Model name a diode/BJT/MOSFET references (shared by handlers + validate).

        Returns ``None`` if the component is not a modelled semiconductor. A bare
        type keyword in ``value`` (``'npn'``/``'pmos'``/...) selects the matching
        generic; an empty ``value`` falls back to the ``Default*`` generic implied
        by the symbol's polarity; any other ``value`` is used verbatim as the model
        name (which ``validate`` then checks resolves to a built-in).
        """
        kind = self._kind(component)
        if kind not in ("diode", "bjt", "mosfet"):
            return None
        symbol = str(self._attr(component, "symbol", "")).lower()
        # A Sim.Device token (NPN/PNP/NMOS/PMOS) sets polarity when the symbol
        # doesn't (e.g. a sim-only stand-in on a generic symbol).
        device = str(self._sim_props(component).get("device", "")).strip().lower()
        value = self._attr(component, "value", None)
        v = str(value).strip().lower() if value else ""
        if v in self._TYPE_KEYWORD_MODELS:
            return self._TYPE_KEYWORD_MODELS[v]
        if value:
            return str(value)
        if kind == "diode":
            return "DefaultSchottky" if device == "schottky" else "DefaultDiode"
        if kind == "bjt":
            return (
                "DefaultPNP" if ("pnp" in symbol or device == "pnp") else "DefaultNPN"
            )
        return (
            "DefaultPMOS" if ("pmos" in symbol or device == "pmos") else "DefaultNMOS"
        )

    def _diode_fallback_generic(self, component, base) -> str:
        """Which generic seeds an unknown diode's Sim.Params fallback.

        ``DefaultSchottky`` when the part is identified as a Schottky -- an
        explicit ``Sim.Device="SCHOTTKY"`` hint, or a well-known Schottky family
        prefix on the model name (classification only; the datasheet fit still
        comes from the user's overrides). Silicon ``DefaultDiode`` otherwise.
        """
        device = str(self._sim_props(component).get("device", "")).strip().lower()
        if device == "schottky":
            return "DefaultSchottky"
        if base and self._SCHOTTKY_PREFIX_RE.match(str(base).strip()):
            return "DefaultSchottky"
        return "DefaultDiode"

    @staticmethod
    def _parse_sim_params(spec) -> dict:
        """Parse a KiCad ``Sim.Params`` string (``"bf=200 is=1e-14"``) to ``{K: v}``.

        Keys are upper-cased (ngspice model params are case-insensitive; upper-case
        matches the built-in generics). Accepts space- or comma-separated pairs.
        """
        out = {}
        if not spec:
            return out
        for token in str(spec).replace(",", " ").split():
            key, sep, val = token.partition("=")
            if sep and key.strip():
                out[key.strip().upper()] = val.strip()
        return out

    @staticmethod
    def _coerce_param(value):
        """Coerce a model-param value to float when it looks numeric, else keep it."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    # SPICE model_type (from the built-in ModelLibrary) -> our device kind, used to
    # reject attaching e.g. an NPN model card to a diode.
    _MODEL_TYPE_TO_KIND = {
        "D": "diode",
        "NPN": "bjt",
        "PNP": "bjt",
        "NMOS": "mosfet",
        "PMOS": "mosfet",
    }

    # Trailing package/reel tokens stripped (case-insensitive) when a model name
    # doesn't resolve, to retry against its bare die name -- only for diodes/BJTs,
    # and only if the stripped base actually exists in the library (never a guess).
    _PKG_SUFFIX_RE = re.compile(
        r"(?:-7-F|T/R|WS|WT|TR|W|S|A|B)$", re.IGNORECASE
    )

    def _lookup_model_spec(self, name, kind):
        """Resolve a model name through the ladder -> ((device_type, params), tier,
        resolved_name).

        Tier is ``generic`` (built-in ``Default*``), ``datasheet_fit`` (a matching
        entry in the built-in ``ModelLibrary``, possibly via a package-suffix alias
        -- ``1N4148W`` -> ``1N4148``), or ``unresolved`` (unknown, or a library
        entry whose device type is wrong for ``kind``). ``resolved_name`` is the
        canonical library name actually matched (== ``name`` on an exact hit), so
        the caller can record an ``asked->base`` provenance for an alias. Returns
        ``(None, "unresolved", name)`` when nothing resolves.
        """
        if name in self.GENERIC_MODELS:
            device_type, params = self.GENERIC_MODELS[name]
            return (device_type, dict(params)), "generic", name
        entry, resolved = self._resolve_library_model(name, kind)
        if entry is not None:
            mtype = str(entry.model_type).upper()
            if self._MODEL_TYPE_TO_KIND.get(mtype) == kind:
                return (entry.model_type, dict(entry.parameters)), "datasheet_fit", resolved
        return None, "unresolved", name

    def _resolve_library_model(self, name, kind):
        """Find a ModelLibrary entry for ``name`` -> ``(SpiceModel|None, canonical)``.

        Exact match, then the explicit ALIASES table (``models.resolve_model``),
        then -- for diodes/BJTs only -- a conservative package-suffix strip that is
        accepted only when the stripped base already exists in the library.
        """
        try:
            from .models import get_model_library
        except Exception:  # pragma: no cover - library import/init failure
            return None, name
        lib = get_model_library()
        entry, resolved = lib.resolve_model(name)
        if entry is not None:
            return entry, resolved
        if kind in ("diode", "bjt") and name:
            base = self._PKG_SUFFIX_RE.sub("", str(name))
            if base and base != name:
                stripped = lib.get_model(base)
                if stripped is not None:
                    return stripped, base
        return None, name

    def _resolve_device_model(self, component, ref) -> Optional[str]:
        """Model name a device instance should reference; resolve tier + Sim.Params.

        Resolves ``_device_model_name`` through the tiered ladder
        (datasheet_fit -> generic), records provenance for the device, and applies
        any ``Sim.Params`` override as a per-device derived card (``{base}_{ref}``)
        so it can't collide with another part's overrides. Returns None for
        non-semiconductors. An unresolved base is referenced verbatim (validate()
        reports it) and recorded as tier ``unresolved``.
        """
        kind = self._kind(component)
        if kind not in ("diode", "bjt", "mosfet"):
            return None
        base = self._device_model_name(component)
        spec, tier, resolved = self._lookup_model_spec(base, kind)
        # Provenance name records a package-suffix alias as "asked->die" so an
        # aliased model is never silent (e.g. "1N4148W->1N4148"); an exact hit
        # keeps the plain name.
        prov_name = base if resolved == base else f"{base}->{resolved}"
        overrides = self._parse_sim_params(self._sim_props(component).get("params"))

        if spec is None:
            # An unknown model name carrying Sim.Params is NOT a dead end: fall
            # back to the kind's generic and overlay the overrides as a derived
            # card, so an unlisted part can be param-fitted (bug #11). With NO
            # overrides this stays a hard, loud validation error (below).
            if overrides and kind in self._KIND_GENERIC:
                generic = self._KIND_GENERIC[kind]
                if kind == "diode":
                    generic = self._diode_fallback_generic(component, base)
                device_type, gparams = self.GENERIC_MODELS[generic]
                merged = dict(gparams)
                for key, val in overrides.items():
                    merged[key] = self._coerce_param(val)
                derived = f"{generic}_{ref}"
                self.derived_models[derived] = (device_type, merged)
                self.model_provenance[ref] = ResolvedModel(
                    ref, kind, "generic", f"{base}->{generic}", overridden=True
                )
                # The seed choice (silicon vs Schottky) and the effective reverse
                # rating must never be silent -- a silicon seed on a Schottky, or
                # a BV=40 generic at PSU PIV, is the silently-wrong class the LLC
                # E2E hit (R1/R2).
                extra = ""
                if device_type == "D":
                    bv = merged.get("BV")
                    extra = (
                        f", effective BV={self._coerce_param(bv)} V"
                        if bv is not None
                        else ", no BV (reverse breakdown not modeled)"
                    )
                logger.warning(
                    f"{ref}: model '{base}' not in library; simulating as "
                    f"{generic} + Sim.Params overrides (tier=generic{extra})"
                )
                return derived
            self.model_provenance[ref] = ResolvedModel(ref, kind, "unresolved", base)
            return base

        device_type, params = spec
        self.model_provenance[ref] = ResolvedModel(
            ref, kind, tier, prov_name, overridden=bool(overrides)
        )
        if resolved != base:
            logger.debug(
                f"{ref}: model '{base}' aliased to library die '{resolved}' "
                f"(package-suffix), tier={tier}"
            )
        logger.debug(
            f"{ref} ({prov_name}): model tier={tier}"
            + (" (+Sim.Params override)" if overrides else "")
        )

        if not overrides:
            self._register_model_card(base, device_type, params, tier)
            return base

        merged = dict(params)
        for key, val in overrides.items():
            merged[key] = self._coerce_param(val)
        derived = f"{base}_{ref}"
        self.derived_models[derived] = (device_type, merged)
        return derived

    def _register_model_card(self, name, device_type, params, tier):
        """Mark a model for emission by tier (generic -> built-in, else explicit)."""
        if tier == "generic":
            self.used_models.add(name)
        else:  # datasheet_fit: emit the library's params explicitly
            self.library_models[name] = (device_type, params)

    # ------------------------------------------------------------------ #
    # External vendor models: Sim.Library / Sim.Name / Sim.Pins (9.3)    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spice_model_store_dir() -> str:
        """Local MPN-keyed model store (~/.skidl/spice_models/models)."""
        try:
            from .model_store import get_model_store

            return get_model_store().models_dir
        except Exception:  # pragma: no cover
            return os.path.join(
                os.path.expanduser("~"), ".skidl", "spice_models", "models"
            )

    def _lib_search_dirs(self) -> List[str]:
        """Directories a relative Sim.Library is resolved against, in order."""
        dirs = [os.getcwd(), self._spice_model_store_dir()]
        # Fall back near the circuit's own source file when we can tell where it is.
        src = getattr(self.circuit, "source_file", None) or getattr(
            self.circuit, "_source_file", None
        )
        if src:
            dirs.insert(0, os.path.dirname(os.path.abspath(str(src))))
        return dirs

    def _resolve_lib_path(self, lib) -> Optional[str]:
        """Resolve a Sim.Library reference to an absolute path (existing if found)."""
        if not lib:
            return None
        p = str(lib)
        if os.path.isabs(p):
            return p
        for base in self._lib_search_dirs():
            cand = os.path.join(base, p)
            if os.path.exists(cand):
                return os.path.abspath(cand)
        return os.path.abspath(p)  # may not exist; validate() reports it

    @staticmethod
    def _model_simulatability(path, name):
        """Classify the ``name`` model in ``path`` (dialect + simulatable verdict).

        Part-agnostic: inspects the model *body* for structural signatures of a
        class ngspice-in-KiCad cannot run (XSPICE digital, PSpice U-device,
        encrypted). Returns a ``ModelClass`` or None if classification is
        unavailable (import/read failure -> never blocks).
        """
        try:
            from .simulatability import classify_model_file

            return classify_model_file(path, name)
        except Exception:  # pragma: no cover - classification never blocks
            return None

    @staticmethod
    def _scan_lib(path, name):
        """Find ``name`` in a .lib/.sub file.

        -> ``('subckt', [nodes], "")`` | ``('model', None, DTYPE)`` |
        ``(None, None, "")``. ``DTYPE`` is the ``.model``'s declared device type
        (``VDMOS`` / ``NMOS`` / ``NPN`` / ``D`` ...), needed to emit a 3-terminal
        VDMOS line correctly. Returns ``(None, None, "")`` if the file is
        unreadable or defines neither a ``.subckt`` nor a ``.model`` by that name.
        Subckt node names are read from the definition line (params like
        ``PARAM=1`` end the node list).
        """
        if not name or not path:
            return None, None, ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return None, None, ""
        sub = re.search(
            rf"^\s*\.subckt\s+{re.escape(str(name))}\b(.*)$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if sub:
            nodes = []
            for tok in sub.group(1).split():
                if tok.lower() == "params:" or "=" in tok:
                    break  # PSpice 'PARAMS:' keyword or the first param -> nodes end
                nodes.append(tok)
            return "subckt", nodes, ""
        mod = re.search(
            rf"^\s*\.model\s+{re.escape(str(name))}\s+(\S+)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if mod:
            return "model", None, mod.group(1).split("(")[0]
        # A .model with the type on a continuation line (rare) still matches kind.
        mod2 = re.search(
            rf"^\s*\.model\s+{re.escape(str(name))}\b",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if mod2:
            return "model", None, ""
        return None, None, ""

    @staticmethod
    def _scan_lib_first(path):
        """First model in a file.

        -> ``('subckt', name, [nodes], "")`` | ``('model', name, None, DTYPE)`` |
        ``(None, None, None, "")``. ``DTYPE`` is the ``.model``'s declared device
        type (for the VDMOS 3-terminal case). Used for store files keyed only by
        MPN, where the internal model/subckt name isn't known ahead of time.
        """
        if not path:
            return None, None, None, ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return None, None, None, ""
        sub = re.search(
            r"^\s*\.subckt\s+(\S+)(.*)$", text, re.IGNORECASE | re.MULTILINE
        )
        mod = re.search(
            r"^\s*\.model\s+(\S+)\s*(\S+)?", text, re.IGNORECASE | re.MULTILINE
        )
        # Honor whichever appears first in the file.
        if sub and (not mod or sub.start() < mod.start()):
            nodes = []
            for tok in sub.group(2).split():
                if tok.lower() == "params:" or "=" in tok:
                    break
                nodes.append(tok)
            return "subckt", sub.group(1), nodes, ""
        if mod:
            dtype = (mod.group(2) or "").split("(")[0]
            return "model", mod.group(1), None, dtype
        return None, None, None, ""

    @staticmethod
    def _norm_node_name(s) -> str:
        """Normalize a subckt node / symbol pin name for name-matching (S1):
        strip the KiCad overbar wrapper ``~{...}``, a leading ``/``, underscores
        and whitespace, then casefold. ``~{SD}``/``SD``/``sd`` compare equal."""
        s = str(s or "").strip()
        s = s.replace("~{", "").replace("}", "").replace("~", "")
        return s.lstrip("/").replace("_", "").strip().casefold()

    def _warn_crossed_sim_pins(self, ref, component, pins_spec, subckt_nodes):
        """Heuristic guard: warn when a ``Sim.Pins`` mapping looks *swapped* (S1).

        For a driver/IC subckt whose node names are self-descriptive (``VCC IN SD
        com VB HO VS LO``), a user who pasted a *positional* mapping (assuming the
        symbol's pin numbering matches the subckt's node order) silently
        cross-wires pairs. We can't error -- vendor node names may legitimately
        differ from a symbol's pin names -- but a clean swap (>=2 pins whose own
        NAME equals some node X, yet are mapped to a different node Y, with X
        assigned to another pin) is almost always a mistake. Numeric node names
        are exempt (no name to match). One warning per (ref) per run.
        """
        mapping = self._parse_sim_pins(pins_spec)
        if not mapping or not subckt_nodes:
            return
        warned = getattr(self, "_crossed_pins_warned", None)
        if warned is None:
            warned = self._crossed_pins_warned = set()
        if ref in warned:
            return
        # symbol pin number -> pin name
        pin_names = {}
        pin_map = getattr(component, "_pins", None)
        if isinstance(pin_map, dict):
            for num, pin in pin_map.items():
                pin_names[str(num)] = (getattr(pin, "name", "") or "").strip()
        # nodes that carry a matchable (alphabetic) name -> normalized -> node
        norm_node = {}
        for n in subckt_nodes:
            if any(c.isalpha() for c in n):
                norm_node[self._norm_node_name(n)] = n
        # the node name each symbol pin is actually mapped to
        assigned = {}
        for sym_pin, target in mapping.items():
            if target in subckt_nodes:
                assigned[str(sym_pin)] = target
            else:
                try:
                    idx = int(target)
                except (TypeError, ValueError):
                    continue
                if 1 <= idx <= len(subckt_nodes):
                    assigned[str(sym_pin)] = subckt_nodes[idx - 1]
        assigned_norms = {self._norm_node_name(v) for v in assigned.values()}
        crossed = []
        for num, name in pin_names.items():
            if not name or not any(c.isalpha() for c in name):
                continue
            nn = self._norm_node_name(name)
            if nn not in norm_node:
                continue  # this pin's name isn't one of the subckt's node names
            got = assigned.get(num)
            if got is None or self._norm_node_name(got) == nn:
                continue  # not assigned, or assigned to its own-named node (fine)
            # pin's name matches node `nn`, but it's mapped elsewhere -- and `nn`
            # is claimed by some other pin => a genuine cross, not just a rename.
            if nn in assigned_norms:
                crossed.append((num, name, got, norm_node[nn]))
        if len(crossed) >= 2:
            warned.add(ref)
            details = "; ".join(
                f"pin {num} (name {name!r}) -> node '{got}' but its name matches "
                f"node '{exp}'"
                for num, name, got, exp in crossed
            )
            logger.warning(
                f"{ref}: Sim.Pins may be CROSSED -- {len(crossed)} pins are mapped "
                f"to a node other than the one matching their own name "
                f"({details}). If you pasted a positional mapping, note the "
                f"subckt's node ORDER need not match your symbol's pin numbering; "
                f"map each pin to the node whose NAME is that pin's role."
            )

    @staticmethod
    def _parse_sim_pins(spec) -> dict:
        """Parse KiCad ``Sim.Pins`` (``"1=out 2=inp"``) -> {symbol_pin: target}."""
        out = {}
        if not spec:
            return out
        for tok in str(spec).replace(",", " ").split():
            key, sep, val = tok.partition("=")
            if sep and key.strip():
                out[key.strip()] = val.strip()
        return out

    def _symbol_pin_nodes(self, component) -> dict:
        """{symbol pin number (str): spice node} for a component's connected pins."""
        out = {}
        pin_map = getattr(component, "_pins", None)
        if isinstance(pin_map, dict):
            for num, pin in pin_map.items():
                net = getattr(pin, "net", None)
                name = getattr(net, "name", None)
                if name:
                    out[str(num)] = self.node_map.get(name, name)
        return out

    def _subckt_pin_mismatch_message(
        self, ref, name, component, subckt_nodes, mapped_nodes
    ) -> str:
        """A clear Sim.Pins-mismatch error (subckt nodes vs the symbol's pins)."""
        pins = []
        nums = []
        pin_map = getattr(component, "_pins", None)
        if isinstance(pin_map, dict):
            for num, pin in sorted(
                pin_map.items(), key=lambda kv: self._pin_sort_key(kv[0])
            ):
                pn = (getattr(pin, "name", "") or "").strip()
                pins.append(f"{num}/{pn}" if pn else str(num))
                nums.append(str(num))
        pin_list = ", ".join(pins) if pins else "(none connected)"
        example = f"{nums[0]}={subckt_nodes[0]}" if nums else "1=<node>"
        return (
            f"{ref}: Sim.Pins maps {len(mapped_nodes)} of subckt '{name}''s "
            f"{len(subckt_nodes)} nodes -- ngspice would fail with 'Too few "
            f"parameters'. This symbol's pins are: {pin_list} (number/name). "
            f"The subckt's nodes (in order) are: {' '.join(subckt_nodes)}. Set "
            f'Sim.Pins="<symbol pin NUMBER>=<subckt node>" for each node, e.g. '
            f'Sim.Pins="{example} ...".'
        )

    def _external_nodes(self, component, pins_spec, subckt_nodes) -> List[str]:
        """Order a component's nets to match an external subckt's node order.

        With ``Sim.Pins`` each symbol pin maps to a subckt node (by name or 1-based
        position); nodes are emitted in subckt-definition order. Without it, the
        symbol's connected pins are used in pin-number order.
        """
        sym_node = self._symbol_pin_nodes(component)
        mapping = self._parse_sim_pins(pins_spec)
        if not mapping:
            return [sym_node[k] for k in sorted(sym_node, key=self._pin_sort_key)]
        pos_node = {}
        for sym_pin, target in mapping.items():
            node = sym_node.get(str(sym_pin))
            if node is None:
                continue
            if subckt_nodes and target in subckt_nodes:
                idx = subckt_nodes.index(target) + 1
            else:
                try:
                    idx = int(target)
                except (TypeError, ValueError):
                    continue
            pos_node[idx] = node
        return [pos_node[i] for i in sorted(pos_node)]

    @staticmethod
    def _include_cache_dir() -> str:
        """Space-free cache for staged external libs. ngspice's ``.include`` (as
        emitted unquoted by PySpice) truncates a path at the first space and
        mishandles non-ASCII, so a lib living under e.g. ``Operational
        Amplifier/`` must be staged to a clean path first."""
        return os.path.join(
            os.path.expanduser("~"), ".skidl", "spice_models", "_include_cache"
        )

    @staticmethod
    def _needs_staging(path) -> bool:
        p = str(path)
        return any(c.isspace() for c in p) or any(ord(c) > 127 for c in p)

    def _safe_lib_path(self, path) -> str:
        """An ngspice-safe (space-free, ASCII) path for ``.include``.

        When the source path has spaces / non-ASCII, copy it into the include
        cache under a deterministic ``<stem>_<hash8><ext>`` name (same source ->
        same staged file, so runs stay reproducible) and return that. Clean
        paths are returned unchanged -> byte-identical emission to before.
        Note: a staged copy breaks any *relative* ``.include``/``.lib`` inside
        the model file; corpus files are overwhelmingly self-contained.
        """
        p = os.path.abspath(str(path))
        if not self._needs_staging(p) or not os.path.exists(p):
            return p
        cache = self._include_cache_dir()
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(p))
        root, ext = os.path.splitext(stem)
        digest = hashlib.sha1(p.encode("utf-8", "replace")).hexdigest()[:8]
        staged = os.path.join(cache, f"{root}_{digest}{ext or '.lib'}")
        try:
            if not os.path.exists(staged) or (
                os.path.getmtime(staged) < os.path.getmtime(p)
            ):
                os.makedirs(cache, exist_ok=True)
                shutil.copyfile(p, staged)
            return staged
        except OSError as exc:  # pragma: no cover - filesystem specifics
            logger.warning(f"Could not stage SPICE lib {p} to space-free cache: {exc}")
            return p

    def _include_lib(self, path) -> None:
        """`.include` an external file once (idempotent per converter)."""
        if not path or path in self.included_libs:
            return
        self.included_libs.add(path)
        safe = self._safe_lib_path(path)
        try:
            self.spice_circuit.include(safe)
            logger.debug(f"Included SPICE library {safe}")
        except Exception as exc:  # pragma: no cover - PySpice/ngspice specifics
            logger.warning(f"Failed to include SPICE library {safe}: {exc}")

    @staticmethod
    def _minimal_deck_enabled() -> bool:
        """Minimal-deck includes for corpus-resolved models (on by default).

        ``SKIDL_SIM_MINIMAL_DECK=0`` is the kill switch: every include reverts to
        the whole file, byte-identical to the pre-feature emission.
        """
        return os.environ.get("SKIDL_SIM_MINIMAL_DECK", "1") != "0"

    def _emit_minimal_decks(self) -> None:
        """`.include` an extracted deck per corpus-resolved library file.

        One malformed line anywhere in a vendor library makes ngspice reject
        every model in it -- measured: 2,101 corpus load failures from 102 files,
        70 of them 100% dead (``Zener_DiodesInc.lib`` alone defines 842 zeners). So
        for models the library *index* resolved automatically we include only the
        blocks the netlist actually needs, extracted by ``model_deck``.

        Only the auto-resolve tier changes. An explicit ``Sim.Library`` is user
        intent -- and the escape hatch by construction -- so it keeps its
        whole-file include; if such a part already included this same file, the
        deck is skipped too (adding it would redefine those subckts). Extraction
        failure falls back to the whole file: degrade to today, never to silence.
        """
        from .model_deck import stage_minimal_deck

        def _same_file(p):
            return os.path.normcase(os.path.abspath(str(p)))

        already = {_same_file(p) for p in self.included_libs}
        for path, names in sorted(self._mindeck_needs.items()):
            if _same_file(path) in already:
                logger.debug(
                    f"minimal-deck include skipped for {os.path.basename(path)}: "
                    f"already included whole-file (explicit Sim.Library)"
                )
                continue
            base = os.path.basename(path)
            staged = stage_minimal_deck(
                path, sorted(names), self._include_cache_dir()
            )
            if staged:
                self.included_libs.add(path)  # dedup against a later whole-file ask
                self._include_lib(staged)
                logger.info(
                    f"minimal-deck include: {base} ({len(names)} model(s))"
                )
            else:
                logger.warning(
                    f"Could not extract a minimal deck from {base} for "
                    f"{sorted(names)} - including the whole file (a malformed "
                    f"line anywhere in it will fail the simulation)"
                )
                self._include_lib(path)

    def _add_external_model(self, component, ref) -> None:
        """Attach a device's external vendor model (Sim.Library + Sim.Name)."""
        sim = self._sim_props(component)
        name = sim.get("name")
        path = self._resolve_lib_path(sim.get("library"))
        kind_in_file, subckt_nodes, model_type = self._scan_lib(path, name)
        self._emit_external(
            component, ref, path, name, kind_in_file, subckt_nodes,
            "sim_library", model_type,
        )

    def _add_store_model(self, component, ref, path) -> None:
        """Attach a device's model from the local MPN store (name discovered)."""
        kind_in_file, name, subckt_nodes, model_type = self._scan_lib_first(path)
        self._emit_external(
            component, ref, path, name, kind_in_file, subckt_nodes,
            "local_store", model_type,
        )

    def _path_under_corpus_root(self, path) -> bool:
        """True if ``path`` lives under a configured SPICE-library (corpus) root.

        Path-class test only (F2): compares absolute, normcased prefixes against
        the library index's configured roots. Used to tell an author's own
        ``Sim.Library`` file (outside the corpus -> ``user_lib``) from a catalogued
        vendor library. Returns False when no corpus is configured (no roots to be
        under) or on any resolution error -- degrade to the ``vendor_lib`` label,
        never crash.
        """
        if not path:
            return False
        try:
            from .library_index import get_library_index

            index = get_library_index()
        except Exception:  # pragma: no cover - index import/init failure
            index = None
        if index is None:
            return False
        try:
            ap = os.path.normcase(os.path.abspath(str(path)))
        except Exception:  # pragma: no cover
            return False
        for root in getattr(index, "roots", []) or []:
            try:
                r = os.path.normcase(os.path.abspath(str(root)))
            except Exception:  # pragma: no cover
                continue
            if ap == r or ap.startswith(r + os.sep):
                return True
        return False

    def _external_tier(self, source, path) -> str:
        """Provenance tier for an external model (F2).

        An explicit ``Sim.Library`` path outside the corpus is a user-authored
        artifact -> ``user_lib``; a corpus/index or MPN-store model is
        ``vendor_lib``. Keeping them distinct is the whole point of provenance:
        a hand-written behavioral macromodel must never wear a real vendor
        model's label.
        """
        if source == "sim_library" and not self._path_under_corpus_root(path):
            return "user_lib"
        return "vendor_lib"

    @staticmethod
    def _pin_map_confidence(dev_kind, subckt_nodes) -> str:
        """Confidence in a subckt's terminal (Sim.Pins) mapping (F3).

        ``"heuristic"`` for a 3-node transistor subckt whose nodes are all-numeric
        (``10 20 30``) -- the D/G/S (C/B/E) identity is a convention-based guess a
        wrong map turns into a converged-but-WRONG result with no error;
        ``"named"`` when the nodes carry self-descriptive names; ``""`` otherwise.
        """
        if dev_kind not in ("mosfet", "bjt") or not subckt_nodes:
            return ""
        if len(subckt_nodes) != 3:
            return ""
        if all(not any(c.isalpha() for c in str(n)) for n in subckt_nodes):
            return "heuristic"
        return "named"

    def _warn_heuristic_pin_map(self, ref, name, subckt_nodes) -> None:
        """One WARNING per (ref) when a heuristic-graded transistor pin map is used."""
        warned = getattr(self, "_heuristic_pin_warned", None)
        if warned is None:
            warned = self._heuristic_pin_warned = set()
        if ref in warned:
            return
        warned.add(ref)
        logger.warning(
            f"{ref}: subckt '{name}' terminal identity (which of nodes "
            f"{' '.join(str(n) for n in subckt_nodes)} is Drain/Gate/Source or "
            f"Collector/Base/Emitter) is a HEURISTIC, not verified -- a wrong "
            f"Sim.Pins map gives a converged-but-wrong result with no error. "
            f"Verify with: python -m skidl_eda.sourcing.find_spice_model "
            f"{name} --verify-terminals"
        )

    def _emit_external(
        self, component, ref, path, name, kind_in_file, subckt_nodes, source,
        model_type="",
    ) -> None:
        """Shared emit for an external model (Sim.Library or store), by file kind."""
        tier = self._external_tier(source, path)
        if source == "library_index" and self._minimal_deck_enabled() and name:
            # Defer: _emit_minimal_decks includes an extracted deck instead of the
            # whole file once every needed name from it is known.
            self._mindeck_needs.setdefault(str(path), set()).add(str(name))
        else:
            self._include_lib(path)
        dev_kind = self._kind(component)
        base = os.path.basename(str(path))

        if kind_in_file == "subckt":
            pins_spec = self._sim_props(component).get("pins")
            nodes = self._external_nodes(component, pins_spec, subckt_nodes)
            # Warn (never error) if the mapping looks like a positional-paste swap
            # for a named-node subckt (S1).
            self._warn_crossed_sim_pins(ref, component, pins_spec, subckt_nodes)
            if subckt_nodes and len(nodes) != len(subckt_nodes):
                # A wrong/short Sim.Pins mapping would emit an X line with the
                # wrong node count -> a cryptic ngspice "Too few parameters for
                # subcircuit" crash. Fail here with the actual pin/node lists so
                # the user can fix the mapping (E2E finding M2).
                raise SimulationValidationError(
                    [self._subckt_pin_mismatch_message(
                        ref, name, component, subckt_nodes, nodes)]
                )
            # Pass Sim.Params through as subckt parameters (X ... NAME p=v). A
            # Sim.Library part has no other Sim.Params consumer (the derived-model
            # path is only reached for built-in primitives), so this is safe to
            # reuse. Empty -> no kwargs -> byte-identical to the pre-20.2 emission.
            xparams = {
                k.lower(): v
                for k, v in self._parse_sim_params(
                    self._sim_props(component).get("params")
                ).items()
            }
            self.spice_circuit.X(ref, name, *nodes, **xparams)
            confidence = self._pin_map_confidence(dev_kind, subckt_nodes)
            self.model_provenance[ref] = ResolvedModel(
                ref, dev_kind or "subckt", tier, name, source=source,
                pin_map_confidence=confidence,
            )
            if confidence == "heuristic":
                self._warn_heuristic_pin_map(ref, name, subckt_nodes)
            logger.debug(
                f"{ref}: external subckt {name} from {base} "
                f"(tier={tier}, source={source}, pin_map={confidence or 'n/a'})"
            )
        elif kind_in_file == "model":
            nodes = self._get_component_nodes(component)
            self._emit_primitive_with_external_model(
                dev_kind, ref, name, nodes, component, model_type
            )
            self.model_provenance[ref] = ResolvedModel(
                ref, dev_kind or "?", tier, name, source=source
            )
            logger.debug(
                f"{ref}: external .model {name} from {base} "
                f"(tier={tier}, source={source})"
            )
        else:
            # validate() reports this in strict mode; lenient mode just skips.
            logger.warning(f"{ref}: no usable model found in {path} - skipping")

    def _emit_primitive_with_external_model(
        self, kind, ref, name, nodes, component=None, model_type=""
    ) -> None:
        """Emit a D/Q/M instance referencing an external ``.model`` name (no card).

        Transistor terminals are resolved by pin *name* (C/B/E, D/G/S) exactly
        like the curated ``_add_bjt_transistor``/``_add_mosfet`` paths -- KiCad
        symbols number their pins inconsistently, so the pre-fix positional order
        silently swapped drain/gate (or collector/base) for a vendor-lib device,
        producing a clean-provenance but DEAD part (bug M1). ``model_type`` is the
        ``.model``'s declared type; an ngspice ``VDMOS`` card is a *3-terminal*
        element (``M nd ng ns model``), so a 4-node M line misparses against it.
        """
        mtype = (model_type or "").strip().upper()

        # Defensive: MOS-family model type on a non-MOSFET device (or vice versa).
        if mtype in ("NMOS", "PMOS", "VDMOS") and kind != "mosfet":
            logger.warning(
                f"{ref}: external model '{name}' is a {mtype} card but the device "
                f"kind is '{kind}' - terminal mapping may be wrong"
            )

        if kind == "diode" and len(nodes) >= 2:
            # Resolve A/K by pin name so a vendor-lib diode isn't reversed either
            # (bug #12). component is None only for legacy callers -> positional.
            if component is not None:
                self._emit_diode(component, ref, name)
            else:
                self.spice_circuit.D(ref, nodes[0], nodes[1], model=name)
        elif kind == "bjt" and len(nodes) >= 3:
            named = self._named_terminal_nodes(component) if component is not None else {}
            if all(k in named for k in ("C", "B", "E")):
                c_node, b_node, e_node = named["C"], named["B"], named["E"]
            else:
                c_node, b_node, e_node = nodes[0], nodes[1], nodes[2]
            self.spice_circuit.Q(ref, c_node, b_node, e_node, model=name)
        elif kind == "mosfet" and len(nodes) >= 3:
            named = self._named_terminal_nodes(component) if component is not None else {}
            if all(k in named for k in ("D", "G", "S")):
                d_node, g_node, s_node = named["D"], named["G"], named["S"]
                b_node = named.get("B", s_node)
            else:
                d_node, g_node, s_node = nodes[0], nodes[1], nodes[2]
                b_node = nodes[3] if len(nodes) >= 4 else nodes[2]
            if mtype == "VDMOS":
                # ngspice VDMOS is 3-terminal (D G S). PySpice's M() forces 4
                # nodes, so emit the element raw. Prefix matches PySpice's M
                # naming (M<ref>) so get_current()/probes stay consistent.
                self.spice_circuit.raw_spice += (
                    f"\nM{ref} {d_node} {g_node} {s_node} {name}"
                )
            else:
                self.spice_circuit.M(
                    ref, d_node, g_node, s_node, b_node, model=name
                )
        else:
            logger.warning(
                f"{ref}: external .model '{name}' needs a diode/BJT/MOSFET device "
                f"(kind={kind}, {len(nodes)} nodes) - skipping"
            )

    def _has_external_lib(self, component) -> bool:
        return bool(self._sim_props(component).get("library"))

    def _component_mpn(self, component) -> Optional[str]:
        """The MPN a component names, from an ``mpn`` field or its ``value``."""
        extra = getattr(component, "_extra_fields", None)
        if isinstance(extra, dict):
            for key in ("mpn", "MPN", "Mpn"):
                if extra.get(key):
                    return str(extra[key])
        value = self._attr(component, "value", None)
        return str(value) if value else None

    def _store_lib_for(self, component) -> Optional[str]:
        """Path to a local-store model file matching this device's MPN, or None.

        Only active devices (diode/BJT/MOSFET/op-amp) are matched, so a passive
        R/C/L is never hijacked by a coincidentally-named file. An explicit
        ``Sim.Library`` always takes precedence (handled before this is consulted).
        """
        if self._kind(component) not in ("diode", "bjt", "mosfet", "opamp"):
            return None
        mpn = self._component_mpn(component)
        if not mpn:
            return None
        try:
            from .model_store import get_model_store

            return get_model_store().lookup(mpn)
        except Exception:  # pragma: no cover
            return None

    def _library_index_hit(self, component):
        """A ModelHit for this device from the configured library index, or None.

        Gated so it never silently overrides better/intended behavior:
          * active devices only (diode/BJT/MOSFET/op-amp);
          * needs a real model name (Sim.Name, else mpn/value) that isn't a bare
            type keyword;
          * defers to a curated ``datasheet_fit`` card unless Sim.Prefer=library;
          * a ``.subckt`` hit auto-resolves only with Sim.Pins present (so node
            order is explicit) or Sim.Prefer=library -- otherwise fall through,
            since guessing subckt node order from symbol pins is unsafe. Bare
            ``.model`` hits (the common diode/BJT/MOSFET case) need no mapping.
        """
        kind = self._kind(component)
        if kind not in ("diode", "bjt", "mosfet", "opamp"):
            return None
        sim = self._sim_props(component)
        name = sim.get("name") or self._component_mpn(component)
        if not name:
            return None
        name = str(name).strip()
        if not name or name.lower() in self._TYPE_KEYWORD_MODELS:
            return None
        prefer = str(sim.get("prefer", "")).strip().lower()
        if prefer != "library" and kind in ("diode", "bjt", "mosfet"):
            _spec, tier, _res = self._lookup_model_spec(name, kind)
            if tier == "datasheet_fit":
                return None  # curated card wins over the unvetted corpus
        try:
            from .library_index import get_library_index

            index = get_library_index()
            if index is None:
                return None
            hit = index.resolve(name)
        except Exception:  # pragma: no cover - index/init failure is non-fatal
            return None
        if hit is None:
            return None
        if hit.kind == "subckt" and prefer != "library":
            # A subckt auto-resolves only when Sim.Pins actually pins ITS nodes.
            # An absent map -- or a symbol-default map that names none of the
            # subckt's nodes -- is not an explicit node order, so fall through to
            # the built-in/generic model. This matters out of the box: a KiCad
            # symbol carries a default Sim.Pins for the ngspice PRIMITIVE (a diode
            # is "1=K 2=A"), which is meaningless to a same-named corpus .subckt
            # whose nodes are e.g. "1 2". Treating that default as an explicit
            # subckt map used to emit an invalid X-line and hard-fail; instead we
            # keep the intended generic model. A partial-but-nonzero map still
            # falls through to _emit_external's mismatch error (a real, loud
            # signal that a user's explicit Sim.Pins is wrong). Sim.Prefer="library"
            # opts a part into the corpus subckt regardless.
            pins_spec = sim.get("pins")
            mapped = (
                self._external_nodes(component, pins_spec, hit.nodes)
                if pins_spec else []
            )
            if not mapped:
                logger.debug(
                    f"{getattr(component, 'ref', '?')}: library index has subckt "
                    f"'{hit.name}' but Sim.Pins does not map its nodes "
                    f"({hit.nodes}); using the built-in model (set Sim.Pins to the "
                    f"subckt's nodes, or Sim.Prefer=library, to use the corpus model)"
                )
                return None
        return hit

    def _add_index_model(self, component, ref, hit) -> None:
        """Attach a model resolved from the external library index."""
        self._emit_external(
            component, ref, hit.path, hit.name, hit.kind, hit.nodes,
            "library_index", getattr(hit, "device_type", ""),
        )

    def _emit_models(self):
        """Emit a ``.model`` card for each referenced built-in and derived model.

        Only models actually used by a component are emitted (PySpice would
        otherwise serialize every registered model). Unresolved custom model names
        are left to ``validate`` to report; in the lenient path they simply have no
        card (ngspice then errors on them, as before)."""
        for name in sorted(self.used_models):
            spec = self.GENERIC_MODELS.get(name)
            if spec is None:
                continue
            device_type, params = spec
            self.spice_circuit.model(name, device_type, **params)
            logger.debug(f"Emitted .model {name} {device_type}")
        for name in sorted(self.library_models):
            device_type, params = self.library_models[name]
            self.spice_circuit.model(name, device_type, **params)
            logger.debug(f"Emitted datasheet-fit .model {name} {device_type}")
        for name in sorted(self.derived_models):
            device_type, params = self.derived_models[name]
            self.spice_circuit.model(name, device_type, **params)
            logger.debug(f"Emitted derived .model {name} {device_type}")

        # One honest summary line per convert() (LLC E2E R7: the per-device build
        # lines are DEBUG; this is the single INFO): device count + tier counts,
        # so a textbook generic is never silently mistaken for the real part.
        # Full per-ref detail stays at DEBUG and on ``model_provenance``.
        if self.model_provenance:
            from collections import Counter as _Counter

            tiers = _Counter(p.tier for p in self.model_provenance.values())
            tier_str = ", ".join(f"{n} {t}" for t, n in sorted(tiers.items()))
            logger.info(
                f"Converted {len(self.model_provenance)} modelled device(s); "
                f"tiers: {tier_str}"
            )
            summary = ", ".join(
                f"{r}={p.tier}" for r, p in sorted(self.model_provenance.items())
            )
            logger.debug(f"Model provenance: {summary}")
            generics = [
                p.name for p in self.model_provenance.values() if p.tier == "generic"
            ]
            if generics:
                logger.warning(
                    "Simulating with textbook-generic models (not datasheet-fit) "
                    f"for: {', '.join(sorted(set(generics)))}"
                )

    # Diode terminal pin names. KiCad Device:D/D_Schottky/D_Zener/LED all put
    # K (cathode) on pin 1 and A (anode) on pin 2, so pin-number order is the
    # reverse of SPICE's (anode, cathode) -- resolve by name instead (bug #12).
    _DIODE_ANODE_NAMES = {"A", "A1", "AA", "+"}
    _DIODE_CATHODE_NAMES = {"K", "K1", "KK", "-"}

    def _diode_terminals(self, component):
        """(anode_node, cathode_node) resolved by pin NAME, or None.

        Returns None -- so ``_add_diode`` falls back to positional order -- when
        there is no live pin map (dict/JSON circuits) or the connected pins don't
        yield exactly one anode-named and one cathode-named terminal (unnamed or
        multi-unit symbols). Only connected pins (``pin.net`` set) are considered.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        anode = cathode = None
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            name = (getattr(pin, "name", "") or "").strip().upper()
            node = self.node_map.get(net.name, net.name)
            if name in self._DIODE_ANODE_NAMES:
                if anode is not None:
                    return None  # ambiguous
                anode = node
            elif name in self._DIODE_CATHODE_NAMES:
                if cathode is not None:
                    return None  # ambiguous
                cathode = node
        if anode is None or cathode is None:
            return None
        return anode, cathode

    def _emit_diode(self, component, ref: str, model_name: str) -> bool:
        """Emit one SPICE ``D`` card, terminals by A/K pin name, positional fallback.

        SPICE ``D`` is (anode, cathode). Resolving by name means a
        schematically-correct diode (KiCad pin 1 = K, pin 2 = A) simulates in the
        right direction (bug #12). Falls back to pin-number order only when the
        names can't be resolved -- and warns, so the fallback is never silent.
        Shared by the built-in path (``_add_diode``) and the external-model path
        (``_emit_primitive_with_external_model``). Returns False if the diode had
        too few connections to emit.
        """
        terminals = self._diode_terminals(component)
        if terminals is not None:
            anode, cathode = terminals
            self.spice_circuit.D(ref, anode, cathode, model=model_name)
            logger.debug(
                f"Added diode {ref}: A={anode} K={cathode} model={model_name} (by name)"
            )
            return True

        nodes = self._get_component_nodes(component)
        if len(nodes) < 2:
            logger.warning(f"Diode {ref} needs 2 connections, got {len(nodes)}")
            return False
        logger.warning(
            f"Diode {ref}: pins not resolvable by A/K name; using pin-number "
            f"order (pin1=anode). Verify polarity."
        )
        self.spice_circuit.D(ref, nodes[0], nodes[1], model=model_name)
        logger.debug(
            f"Added diode {ref}: {nodes[0]} -> {nodes[1]} model={model_name} (positional)"
        )
        return True

    def _add_diode(self, component, ref: str, value: str):
        """Add a built-in-model diode to the SPICE circuit (terminals by A/K name)."""
        model_name = self._resolve_device_model(component, ref) or "DefaultDiode"
        self._emit_diode(component, ref, model_name)

    # Ideal op-amp open-loop gain. Large enough that closed-loop behaviour is set
    # by the feedback network, frequency-independent (infinite GBW) so an active
    # filter's response is the RC network alone.
    OPAMP_OPEN_LOOP_GAIN = 1e6

    # Output-typed pins that aren't the real signal output on some dual-output
    # op-amp symbols (e.g. ADA4817 pin 2 is FB). Prefer a pin not named these.
    _OPAMP_NON_OUTPUT_PIN_NAMES = {"FB", "COMP"}

    def _opamp_terminals(self, component, pin_nums=None):
        """Resolve an op-amp's (out, in+, in-) SPICE nodes by pin function/name.

        Op-amp pins must be mapped semantically, not by position: KiCad pinouts
        vary (an LM358 unit is out=1, in-=2, in+=3), so pin-number order would
        swap the inputs. Uses the live pin map and considers only *connected*
        pins, so the unused unit of a dual op-amp (pins with no net) is skipped.
        Power pins (V+/V-) are not needed by the ideal model. Returns
        (out, in_plus, in_minus) spice nodes, or None if a signal terminal is
        missing or no live pin map is available.

        ``pin_nums`` (a set of pin-number strings) restricts resolution to one
        unit of a multi-unit symbol, so each amplifier section of a dual/quad
        gets its own terminals instead of all units collapsing together.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        outputs = []  # (pin_name, node) for every connected output-func pin
        in_plus = in_minus = None
        for num, pin in pin_map.items():
            if pin_nums is not None and str(num) not in pin_nums:
                continue  # a different unit's pin
            net = getattr(pin, "net", None)
            if net is None:
                continue  # unconnected (e.g. the unused unit of a dual op-amp)
            func = str(getattr(pin, "func", "")).lower()
            name = (getattr(pin, "name", "") or "").strip()
            node = self.node_map.get(net.name, net.name)
            if "output" in func:
                outputs.append((name, node))
            elif name == "+":
                in_plus = node
            elif name == "-":
                in_minus = node
        out = self._choose_opamp_output(component, outputs)
        if out is None or in_plus is None or in_minus is None:
            return None
        return out, in_plus, in_minus

    def _choose_opamp_output(self, component, outputs):
        """Pick the real output node from an op-amp's output-func pins.

        Some symbols expose more than one output-typed pin -- e.g. ADA4817 has pin 2
        ``FB`` (feedback, output-typed) and pin 7 ``OUT``. The old code kept whichever
        was iterated last (dict order), so a symbol with FB and OUT on *different*
        nets resolved nondeterministically and could open the feedback loop silently
        (report F4). The ideal VCVS has no separate feedback concept, so drive the
        true OUT: prefer a pin whose name isn't ``FB``/``COMP``. When output pins land
        on more than one distinct net, warn -- the single-output model can't represent
        that, so the user should know which pin was driven.
        """
        if not outputs:
            return None
        preferred = [
            (n, node)
            for (n, node) in outputs
            if n.upper() not in self._OPAMP_NON_OUTPUT_PIN_NAMES
        ]
        chosen_name, chosen_node = (preferred or outputs)[0]
        if len({node for (_, node) in outputs}) > 1:
            ref = getattr(component, "ref", None) or "?"
            ignored = ", ".join(
                sorted(f"{n}->{node}" for (n, node) in outputs if node != chosen_node)
            )
            logger.warning(
                f"Op-amp {ref} has output pins on multiple nets; driving "
                f"'{chosen_name}'->{chosen_node}, ignoring {ignored}. The ideal VCVS "
                f"models a single output -- verify this matches intent."
            )
        return chosen_node

    def _opamp_amp_units(self, component):
        """``{unit_number: {pin_numbers}}`` for the amplifier units of a
        multi-unit op-amp symbol -- units with at least one input pin AND one
        output pin. Power-only units (e.g. LM358 unit 3 = V+/V-) are excluded;
        the ideal/GBW model ignores supply pins.

        Returns ``{}`` when the symbol resolves to a single unit or can't be
        resolved, so the caller keeps the whole-component path. The pin->unit
        map is read from the KiCad symbol (Component.Pin.unit is not populated),
        the same source the schematic writer uses to place per-unit bodies.
        """
        symbol = self._attr(component, "symbol", None) or getattr(
            component, "symbol", None
        )
        if not symbol:
            return {}
        # Multi-unit pin->unit map. The skidl adapter attaches ``_symbol_data``
        # built from the parsed symbol (unit_count + per-pin unit/function/name);
        # fall back to circuit-synth's SymbolLibCache when present (keeps this
        # module usable inside circuit-synth too), else keep the whole-component
        # path. This is the only place the sim layer needs symbol-level data.
        data = getattr(component, "_symbol_data", None)
        if data is None:
            try:
                from circuit_synth.kicad.kicad_symbol_cache import SymbolLibCache

                data = SymbolLibCache.get_symbol_data(symbol)
            except Exception:
                return {}
        if not data or data.get("unit_count", 1) <= 1:
            return {}
        unit_pins = {}  # unit -> {pin number strings}
        unit_has = {}  # unit -> {"in": bool, "out": bool}
        for p in data.get("pins", []):
            num = str(p.get("number", "")).strip()
            unit = p.get("unit", 0)
            if not num or not unit:  # unit 0 = common-to-all; not an amp section
                continue
            func = str(p.get("function", "")).lower()
            name = (p.get("name", "") or "").strip()
            unit_pins.setdefault(unit, set()).add(num)
            flags = unit_has.setdefault(unit, {"in": False, "out": False})
            if "output" in func:
                flags["out"] = True
            elif "input" in func or name in ("+", "-"):
                flags["in"] = True
        return {
            u: pins
            for u, pins in unit_pins.items()
            if unit_has.get(u, {}).get("in") and unit_has.get(u, {}).get("out")
        }

    def _add_opamp(self, component, ref: str, value: str):
        """Add an op-amp: ideal VCVS by default, 1-pole GBW macromodel when opted in.

        Terminals are resolved by pin function/name so the model is correct
        regardless of the symbol's pin numbering. Falls back to a positional guess
        only when no live pin map is available (dict/JSON-shaped circuits).

        A **multi-unit** symbol (dual/quad) with more than one wired amplifier
        section emits **one model per wired unit** -- otherwise both halves of a
        dual collapse into a single VCVS, leaving the second section's output net
        undriven (singular matrix, hard sim failure; bug #A). The first
        amplifier unit keeps the plain ``ref`` element name for netlist
        backward-compatibility; later units get ``{ref}u{unit}``. The die's GBW
        (if any) applies to every unit.

        Without a gain-bandwidth product the op-amp is an ideal VCVS with
        frequency-independent gain ``OPAMP_OPEN_LOOP_GAIN`` (exactly as before). With
        a GBW -- from an explicit ``Sim.Gbw`` field, or a ModelLibrary OPAMP entry
        whose ``value``/name carries a ``GBW`` param -- it becomes a single-pole
        macromodel so source/feedback capacitance limits bandwidth and can peak. Slew
        rate is out of scope (nonlinear); this models only the small-signal pole.
        """
        gbw_hz, tier = self._opamp_gbw(component, value)

        amp_units = self._opamp_amp_units(component)
        pin_map = getattr(component, "_pins", None)
        # Per-unit emission only when there are 2+ amplifier units AND a live pin
        # map to resolve each unit's terminals; otherwise the whole-component
        # path (incl. the dict/JSON positional fallback) runs unchanged.
        if len(amp_units) > 1 and isinstance(pin_map, dict):
            emitted = 0
            for unit in sorted(amp_units):
                unit_pins = amp_units[unit]
                wired = any(
                    getattr(pin, "net", None) is not None
                    for num, pin in pin_map.items()
                    if str(num) in unit_pins
                )
                if not wired:
                    continue  # unused spare section -- fine to leave unmodeled
                terminals = self._opamp_terminals(component, pin_nums=unit_pins)
                if terminals is None:
                    logger.warning(
                        f"Op-amp {ref} unit {unit} is partially wired (missing an "
                        f"input or output terminal); skipping this section"
                    )
                    continue
                out, in_plus, in_minus = terminals
                elem_ref = ref if emitted == 0 else f"{ref}u{unit}"
                self._emit_opamp_model(elem_ref, out, in_plus, in_minus, gbw_hz)
                emitted += 1
            if emitted:
                self._record_opamp_provenance(ref, tier, gbw_hz, units=emitted)
                return
            # Nothing wired -> fall through to the single-model path (which will
            # warn about too-few connections), preserving prior behavior.

        terminals = self._opamp_terminals(component)
        if terminals is None:
            nodes = self._get_component_nodes(component)
            if len(nodes) < 3:
                logger.warning(
                    f"Op-amp {ref} needs at least 3 connections, got {len(nodes)}"
                )
                return
            # Legacy positional fallback: [out, in+, in-].
            out, in_plus, in_minus = nodes[0], nodes[1], nodes[2]
        else:
            out, in_plus, in_minus = terminals

        self._emit_opamp_model(ref, out, in_plus, in_minus, gbw_hz)
        self._record_opamp_provenance(ref, tier, gbw_hz, units=1)

    def _emit_opamp_model(self, elem_ref, out, in_plus, in_minus, gbw_hz):
        """Emit one op-amp section: ideal VCVS, or the 1-pole GBW macromodel."""
        if gbw_hz is None:
            self._add_ideal_opamp(elem_ref, out, in_plus, in_minus)
        else:
            self._add_gbw_opamp(elem_ref, out, in_plus, in_minus, gbw_hz)

    def _record_opamp_provenance(self, ref, tier, gbw_hz, units):
        """One provenance entry per op-amp ref (not per unit). Notes the wired
        unit count when more than one section was modeled."""
        if gbw_hz is None:
            name = "ideal_vcvs"
            tier = "generic"
        else:
            name = f"gbw_1pole({self._fmt_hz(gbw_hz)})"
        if units > 1:
            name = f"{name} x{units} units"
        self.model_provenance[ref] = ResolvedModel(
            ref=ref, kind="opamp", tier=tier, name=name
        )

    def _add_ideal_opamp(self, ref, out, in_plus, in_minus):
        """Ideal op-amp: a single high-gain VCVS (unchanged legacy behaviour)."""
        gain = self.OPAMP_OPEN_LOOP_GAIN
        self.spice_circuit.VCVS(
            ref, out, self.spice_circuit.gnd, in_plus, in_minus, gain
        )
        logger.debug(
            f"Added op-amp {ref} (ideal VCVS, gain={gain}): "
            f"out={out}, in+={in_plus}, in-={in_minus}"
        )

    def _add_gbw_opamp(self, ref, out, in_plus, in_minus, gbw_hz):
        """Single-pole GBW-limited op-amp macromodel.

        ``Aol(s) = Aol0 / (1 + s/wp)`` with ``wp = 2*pi*GBW/Aol0``. Realized as a
        high-gain VCVS (Aol0) into an R-C low-pass (``R*C = 1/wp``), then a unity
        VCVS buffer driving the output node so the pole is unloaded by the feedback
        network. Two internal nodes per op-amp, named from ``ref``.
        """
        gnd = self.spice_circuit.gnd
        aol0 = self.OPAMP_OPEN_LOOP_GAIN
        wp = 2.0 * math.pi * gbw_hz / aol0  # dominant-pole angular frequency
        # Fix C, solve R for R*C = 1/wp (both stay in a numerically sane range).
        cap = 1e-9
        res = 1.0 / (wp * cap)

        p1 = f"{ref}_p1"  # gain-stage output (before the pole)
        p2 = f"{ref}_p2"  # after the R-C pole -> buffered to `out`
        self.spice_circuit.VCVS(f"{ref}_a", p1, gnd, in_plus, in_minus, aol0)
        self.spice_circuit.R(f"{ref}_p", p1, p2, res)
        self.spice_circuit.C(f"{ref}_p", p2, gnd, cap)
        self.spice_circuit.VCVS(f"{ref}_b", out, gnd, p2, gnd, 1.0)
        logger.debug(
            f"Added op-amp {ref} (1-pole GBW macromodel, GBW={self._fmt_hz(gbw_hz)}, "
            f"Aol0={aol0}, fp={wp / (2 * math.pi):.3g} Hz): out={out}, "
            f"in+={in_plus}, in-={in_minus}"
        )

    def _opamp_gbw(self, component, value):
        """Resolve an op-amp's gain-bandwidth product (Hz) and provenance tier.

        Explicit ``Sim.Gbw`` wins; otherwise a ``value``/name hit in the ModelLibrary
        on an OPAMP-type model carrying a ``GBW`` param. Returns ``(gbw_hz, tier)``
        with tier ``sim_params`` (explicit) or ``datasheet_fit`` (library), or
        ``(None, "generic")`` when neither is present -> ideal VCVS.
        """
        gbw_field = self._sim_props(component).get("gbw")
        if gbw_field is not None and str(gbw_field).strip():
            gbw = self._parse_frequency(gbw_field)
            if gbw and gbw > 0:
                return gbw, "sim_params"
            logger.warning(
                f"Op-amp {getattr(component, 'ref', '?')}: could not parse "
                f"Sim.Gbw '{gbw_field}'; falling back to ideal op-amp model"
            )

        name = (value or "").strip()
        if name:
            try:
                from .models import get_model_library

                model = get_model_library().models.get(name)
            except Exception:
                model = None
            if (
                model is not None
                and str(getattr(model, "model_type", "")).upper() == "OPAMP"
            ):
                gbw = getattr(model, "parameters", {}).get("GBW")
                if gbw and float(gbw) > 0:
                    return float(gbw), "datasheet_fit"
        return None, "generic"

    # ------------------------------------------------------------------ #
    # Linear regulators / LDOs: Tier-A behavioral macromodel (Stage 20.1) #
    # ------------------------------------------------------------------ #

    # Pin names (upper-cased) that identify an LDO's three terminals, so the model
    # is correct regardless of the symbol's pin numbering. First connected match
    # wins. A ground pin named ADJ is accepted (adjustable parts) but warned about.
    _LDO_IN_NAMES = {"VI", "VIN", "IN", "VIN+", "IN+", "VCC"}
    _LDO_OUT_NAMES = {"VO", "VOUT", "OUT"}
    _LDO_GND_NAMES = {"GND", "ADJ", "GND/ADJ"}

    # Macromodel param defaults (VOUT has no default -- it must be resolved).
    _LDO_PARAM_DEFAULTS = {"VDROP": 0.3, "RSER": 0.05, "IQ": 1e-3}

    @staticmethod
    def _parse_si_number(raw) -> Optional[float]:
        """Parse an LDO param value ('3.3', '0.3', '100m', '2m', '1u') to float.

        For LDO params (volts / ohms / amps) a bare ``m`` means **milli** and ``u``
        micro -- unlike the resistor-value parser, mega/kilo-scale regulator params
        are nonsensical here, so this small dedicated parser keeps ``m`` unambiguous.
        Plain decimals pass straight through. Returns None if unparseable.
        """
        if raw is None:
            return None
        s = str(raw).strip().lower().replace(" ", "")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        m = re.match(r"^(-?[0-9.]+)([a-z]+)$", s)
        if not m:
            return None
        mult = {
            "k": 1e3,
            "meg": 1e6,
            "m": 1e-3,
            "u": 1e-6,
            "n": 1e-9,
            "p": 1e-12,
        }.get(m.group(2))
        if mult is None:
            return None
        try:
            return float(m.group(1)) * mult
        except ValueError:
            return None

    def _ldo_terminals(self, component):
        """Resolve an LDO's (in, out, gnd) SPICE nodes by pin NAME.

        Uses the live pin map and considers only connected pins, so an unconnected
        EN/SENSE pin is ignored. Returns ``(nin, nout, ngnd)`` or None when a signal
        terminal is missing or no live pin map is available (dict/JSON circuits).
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        nin = nout = ngnd = None
        adj_used = False
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            name = (getattr(pin, "name", "") or "").strip().upper()
            node = self.node_map.get(net.name, net.name)
            if nin is None and name in self._LDO_IN_NAMES:
                nin = node
            elif nout is None and name in self._LDO_OUT_NAMES:
                nout = node
            elif ngnd is None and name in self._LDO_GND_NAMES:
                ngnd = node
                adj_used = name == "ADJ"
        if nin is None or nout is None or ngnd is None:
            return None
        if adj_used:
            logger.warning(
                f"LDO {getattr(component, 'ref', '?')}: ADJ pin used as the "
                f"reference/ground node; adjustable output modeled as a fixed VOUT "
                f"(the external divider is not read)"
            )
        return nin, nout, ngnd

    def _ldo_lib_params(self, value) -> Optional[dict]:
        """LDO macromodel params from a ModelLibrary entry named by ``value``.

        A SUBCKT-type entry carrying a ``VOUT`` param maps VOUT/VDROPOUT/IQ onto the
        macromodel (RSER falls back to its default). Returns None if no such entry.
        """
        name = (value or "").strip()
        if not name:
            return None
        try:
            from .models import get_model_library

            model = get_model_library().models.get(name)
        except Exception:  # pragma: no cover - library import/init failure
            model = None
        params = getattr(model, "parameters", None) if model is not None else None
        if not isinstance(params, dict) or "VOUT" not in params:
            return None
        return {
            "VOUT": float(params["VOUT"]),
            "VDROP": float(params.get("VDROPOUT", self._LDO_PARAM_DEFAULTS["VDROP"])),
            "RSER": float(params.get("RSER", self._LDO_PARAM_DEFAULTS["RSER"])),
            "IQ": float(params.get("IQ", self._LDO_PARAM_DEFAULTS["IQ"])),
        }

    def _ldo_params(self, component, value):
        """(params, tier) for an LDO, or ``(None, "sim_params")`` if VOUT unresolved.

        ``Sim.Params`` (with VOUT) wins at tier ``sim_params``; otherwise a
        ModelLibrary entry named by ``value`` gives tier ``datasheet_fit``.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        if "VOUT" in raw:
            vout = self._parse_si_number(raw["VOUT"])
            if vout is not None:
                params = {"VOUT": vout}
                for key, default in self._LDO_PARAM_DEFAULTS.items():
                    parsed = self._parse_si_number(raw[key]) if key in raw else None
                    params[key] = parsed if parsed is not None else default
                return params, "sim_params"
        lib = self._ldo_lib_params(value)
        if lib is not None:
            return lib, "datasheet_fit"
        return None, "sim_params"

    @staticmethod
    def _fmt_num(x: float) -> str:
        """Compact numeric literal for a netlist ('3.3', '0.002', not '0.0020000')."""
        return f"{x:g}"

    def _stub_unmodeled_pins(self, component, ref, modeled_nodes, gnd) -> None:
        """Tie each connected-but-unmodeled pin's node to ground through 1 GOhm.

        A macromodel drives only its resolved terminals; a real part's other pins
        (NR/SS/BYP/COMP/EN...) remain plain nodes. If such a node's net is cap-only
        (the usual decoupling), it has NO DC path, so ngspice's op-point solve hits
        a ``singular matrix`` on it (bug #16). A 1 GOhm resistor to ground gives the
        node a DC path at ~nA leakage -- harmless whether the net is cap-only,
        driven (an EN header), or resistor-loaded.

        ``modeled_nodes`` are the SPICE nodes the macromodel already uses; those and
        the ground node are skipped, and each remaining node is stubbed once (a pin
        group all tapping one node -- e.g. several pads on OUT -- yields no stub). No
        live pin map (dict/JSON circuits) -> no stubs.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return
        skip = {str(n) for n in modeled_nodes if n is not None} | {str(gnd)}
        stubbed = set()
        lines = []
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            node = str(self.node_map.get(net.name, net.name))
            if node in skip or node in stubbed:
                continue
            stubbed.add(node)
            lines.append(f"R{ref}_stub{len(stubbed)} {node} {gnd} 1e9")
            logger.debug(f"{ref}: 1G stub on unmodeled pin net {node}")
        if lines:
            self.spice_circuit.raw_spice += "\n" + "\n".join(lines)

    def _add_ldo(self, component, ref: str, value: str):
        """Add a linear regulator as a datasheet-parameterized behavioral macromodel.

        Emits (VOUT/VDROP/RSER/IQ from Sim.Params or a ModelLibrary entry)::

            B<ref>_reg <reg> <gnd> V = min(VOUT, V(<in>,<gnd>)-VDROP)
            R<ref>_ser <reg> <out> RSER
            B<ref>_iq  <in> <gnd> I = IQ

        so the output regulates to VOUT, tracks (VIN-VDROP) in dropout, and draws a
        quiescent current from the input. The B-sources go through ``raw_spice``
        (PySpice has no first-class behavioral-source polarity we need); the series
        resistor is a normal element. Limitation: no current limit / thermal
        foldback -- the output is a voltage-behavioral source and will source
        unlimited current into a short.
        """
        terminals = self._ldo_terminals(component)
        if terminals is None:
            nodes = self._get_component_nodes(component)
            if len(nodes) < 3:
                logger.warning(
                    f"LDO {ref} needs at least 3 connections, got {len(nodes)}"
                )
                return
            # Positional fallback (no live pin names): assume 78xx-style IN, GND, OUT.
            nin, ngnd, nout = nodes[0], nodes[1], nodes[2]
            logger.warning(
                f"LDO {ref}: no live pin map; assuming pin order IN, GND, OUT"
            )
        else:
            nin, nout, ngnd = terminals

        params, tier = self._ldo_params(component, value)
        if params is None:
            # validate() reports this in strict mode; the lenient path just skips.
            logger.warning(
                f"LDO {ref}: no VOUT resolved -- skipping "
                f'(set Sim.Params="vout=3.3 vdrop=0.3")'
            )
            return

        vout = self._fmt_num(params["VOUT"])
        vdrop = self._fmt_num(params["VDROP"])
        rser = self._fmt_num(params["RSER"])
        iq = self._fmt_num(params["IQ"])
        gnd = str(ngnd)
        inn = str(nin)
        reg = f"{ref}_reg"

        self.spice_circuit.raw_spice += (
            f"\nB{ref}_reg {reg} {gnd} V = min({vout}, V({inn},{gnd})-{vdrop})"
        )
        self.spice_circuit.R(f"{ref}_ser", reg, nout, rser)
        self.spice_circuit.raw_spice += f"\nB{ref}_iq {inn} {gnd} I = {iq}"

        # Give cap-only unmodeled pins (NR/BYP/EN...) a DC path so the op-point solves.
        self._stub_unmodeled_pins(component, ref, {nin, nout, ngnd}, gnd)

        self.model_provenance[ref] = ResolvedModel(
            ref=ref,
            kind="ldo",
            tier=tier,
            name=f"ldo_macro(vout={vout},vdrop={vdrop})",
        )
        logger.debug(
            f"Added LDO {ref} (behavioral macromodel, tier={tier}): in={inn}, "
            f"out={nout}, gnd={gnd}, VOUT={vout}, VDROP={vdrop}, RSER={rser}, IQ={iq}"
        )

    # ------------------------------------------------------------------ #
    # Behavioral logic primitives (DPSG WS2): DFF / TFF / DLATCH / gates. #
    # ------------------------------------------------------------------ #
    #
    # Corpus digital models don't run in ngspice (XSPICE d_* need adc/dac
    # bridges; PSpice U-devices aren't implemented), so a design needing a
    # flip-flop or gate had NO simulatable path (DPSG A1). These give it one the
    # same way Sim.Device="LDO"/"BUCK" give a behavioral power block: an ngspice-
    # native model attached to an ordinary symbol, with real analog I/O nodes.
    #
    # Logic levels are analog voltages: a node is HIGH above the switching
    # threshold VM (default VDD/2, or (VIH+VIL)/2 if both given). Outputs are
    # stiff B-sources (0 / VDD). A flip-flop's memory is a capacitor held by a
    # voltage-controlled switch (transparent one clock phase, Roff-isolated the
    # other) -- a behavioral master-slave latch, edge-triggered with no XSPICE.

    # Terminal name-sets. KiCad 10 symbols write a complemented pin as ``~{Q}``
    # and a clock as ``C``; ``_logic_pin_nodes`` normalizes the overbar wrapper so
    # ``~{Q}``/``/Q`` both match the QN set.
    _LOGIC_GND_NAMES = {"GND", "VSS", "VEE", "0"}
    _LOGIC_CLK_NAMES = {"CLK", "CK", "CP", "CLOCK", "C", ">"}
    _LOGIC_D_NAMES = {"D", "DATA", "DIN", "T"}
    _LOGIC_EN_NAMES = {"EN", "E", "G", "LE", "GATE", "ENABLE"}
    _LOGIC_Q_NAMES = {"Q", "QOUT"}
    _LOGIC_QN_NAMES = {"QN", "QB", "NQ", "QBAR", "Q_BAR", "~Q", "/Q", "QNOT"}
    _GATE_OUT_NAMES = {"Y", "Z", "O", "OUT", "OUTPUT", "Q"}

    # Behavioral-logic param defaults. VDD is the logic swing; VM the switching
    # threshold; TPD the propagation delay (folded into the memory/edge RC).
    _LOGIC_PARAM_DEFAULTS = {"VDD": 5.0, "TPD": 10e-9}

    def _logic_pin_nodes(self, component):
        """``{ROLE(upper): node}`` for a component's connected pins, or None.

        Behavioral logic resolves its terminals by pin NAME (like the LDO/switcher
        macromodels), so a live ``_pins`` map is required; dict/JSON circuits
        (no names) return None and are skipped by validate().

        Many gate symbols carry meaningless pin names (KiCad writes ``~`` for a
        generic gate pin), so an explicit ``Sim.Pins="3=Y 1=A 2=B"`` (pin NUMBER =
        ROLE) overrides the name map -- letting a behavioral gate ride on any
        symbol, exactly as a subckt's Sim.Pins pins its nodes."""
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        out = {}
        # Explicit Sim.Pins role map (pin number -> role) wins when present.
        spec = self._sim_props(component).get("pins")
        if spec:
            by_num = {str(num): pin for num, pin in pin_map.items()}
            for pin_num, role in self._parse_sim_pins(spec).items():
                pin = by_num.get(str(pin_num))
                net = getattr(pin, "net", None) if pin is not None else None
                if net is None:
                    continue
                out[str(role).strip().upper()] = self.node_map.get(net.name, net.name)
            if out:
                return out
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            name = self._norm_logic_pin_name(getattr(pin, "name", ""))
            if not name:
                continue
            out.setdefault(name, self.node_map.get(net.name, net.name))
        return out

    @staticmethod
    def _norm_logic_pin_name(name) -> str:
        """Normalize a KiCad pin name for logic-role matching.

        Unwraps the overbar spelling ``~{Q}`` -> ``~Q`` (KiCad 10 writes a
        complemented pin that way) so it matches the QN name-set, and upper-cases."""
        s = (name or "").strip().upper()
        m = re.fullmatch(r"~\{(.+)\}", s)
        if m:
            return "~" + m.group(1)
        return s

    def _logic_gnd(self, nodes):
        """The logic reference node: a connected GND-class pin, else ngspice ``0``."""
        for name, node in nodes.items():
            if name in self._LOGIC_GND_NAMES:
                return str(node)
        return "0"

    def _logic_params(self, component) -> dict:
        """Behavioral-logic params (VDD, VM threshold, TPD) with defaults.

        ``Sim.Params`` may set ``vdd``, ``tpd``, and either ``vih``/``vil`` (the
        threshold is their midpoint) or nothing (threshold = VDD/2)."""
        raw = self._parse_sim_params(self._sim_props(component).get("params"))

        def g(key, default):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        vdd = g("VDD", self._LOGIC_PARAM_DEFAULTS["VDD"])
        vih = self._parse_si_number(raw["VIH"]) if "VIH" in raw else None
        vil = self._parse_si_number(raw["VIL"]) if "VIL" in raw else None
        if vih is not None and vil is not None:
            vm = (vih + vil) / 2.0
        else:
            vm = vdd / 2.0
        tpd = g("TPD", self._LOGIC_PARAM_DEFAULTS["TPD"])
        if tpd <= 0:
            tpd = self._LOGIC_PARAM_DEFAULTS["TPD"]
        return {"VDD": vdd, "VM": vm, "TPD": tpd}

    def _emit_ms_dff(self, ref, d_expr, clk, q, qn, gnd, params) -> None:
        """Emit a behavioral rising-edge master-slave D flip-flop.

        ``d_expr`` is the ngspice expression sampled onto the master (an external
        D node for a DFF, the internal Q̄ for a toggle FF). Two switch-held caps
        form the master (transparent while CLK low) and slave (transparent while
        CLK high) latches; on CLK's rising edge the master freezes the last D and
        the slave passes it through, so Q updates once per rising edge. Q/Q̄ are
        stiff B-sources. The memory caps carry ``IC=0`` so a divider starts from a
        defined state under ``use_initial_condition`` (self-oscillating start)."""
        vdd = self._fmt_num(params["VDD"])
        vm = self._fmt_num(params["VM"])
        ron = 100.0
        cmem = max(params["TPD"] / ron, 1e-12)
        cm = self._fmt_num(cmem)
        clkbar = f"{ref}_ckb"
        dbuf, mnode, mbuf, qint = (
            f"{ref}_db", f"{ref}_m", f"{ref}_mb", f"{ref}_qi",
        )
        model = f"SWL{ref}"
        lines = [
            f"B{ref}_ckb {clkbar} {gnd} V = V({clk}) > {vm} ? 0 : {vdd}",
            f"B{ref}_db {dbuf} {gnd} V = {d_expr}",
            f"S{ref}_m {dbuf} {mnode} {clkbar} {gnd} {model}",
            f"C{ref}_m {mnode} {gnd} {cm} IC=0",
            f"B{ref}_mb {mbuf} {gnd} V = V({mnode}) > {vm} ? {vdd} : 0",
            f"S{ref}_s {mbuf} {qint} {clk} {gnd} {model}",
            f"C{ref}_s {qint} {gnd} {cm} IC=0",
            f".model {model} SW(Ron={ron:g} Roff=1e9 Vt={vm} Vh={self._fmt_num(params['VM'] * 0.1 or 0.05)})",
            f"B{ref}_q {q} {gnd} V = V({qint}) > {vm} ? {vdd} : 0",
        ]
        if qn is not None:
            lines.append(f"B{ref}_qn {qn} {gnd} V = V({qint}) > {vm} ? 0 : {vdd}")
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)

    def _add_dff(self, component, ref: str, value: str):
        """Behavioral rising-edge D flip-flop (Sim.Device="DFF").

        Resolves D / CLK / Q (and optional Q̄) by pin name; a ÷2 divider wires Q̄
        back to D externally. No corpus model, no XSPICE bridge."""
        nodes = self._logic_pin_nodes(component)
        if nodes is None:
            logger.warning(f"DFF {ref}: no live pin map; skipping")
            return
        gnd = self._logic_gnd(nodes)
        clk = self._first_named(nodes, self._LOGIC_CLK_NAMES)
        d = self._first_named(nodes, self._LOGIC_D_NAMES)
        q = self._first_named(nodes, self._LOGIC_Q_NAMES)
        qn = self._first_named(nodes, self._LOGIC_QN_NAMES)
        if clk is None or d is None or (q is None and qn is None):
            logger.warning(
                f"DFF {ref}: needs connected D, CLK and Q (or QN) pins - skipping"
            )
            return
        params = self._logic_params(component)
        vm = self._fmt_num(params["VM"])
        vdd = self._fmt_num(params["VDD"])
        q_out = q if q is not None else f"{ref}_q"
        self._emit_ms_dff(
            ref, f"V({d}) > {vm} ? {vdd} : 0", clk, q_out, qn, gnd, params
        )
        self.model_provenance[ref] = ResolvedModel(
            ref, "dff", "sim_params", "dff_behavioral"
        )
        logger.debug(f"Added behavioral DFF {ref}: d={d} clk={clk} q={q} qn={qn}")

    def _add_tff(self, component, ref: str, value: str):
        """Behavioral toggle flip-flop (Sim.Device="TFF"): ÷2 on each CLK edge.

        Same master-slave core as the DFF, but D is the internal Q̄ (no external D
        pin), so a single TFF divides its clock by two."""
        nodes = self._logic_pin_nodes(component)
        if nodes is None:
            logger.warning(f"TFF {ref}: no live pin map; skipping")
            return
        gnd = self._logic_gnd(nodes)
        clk = self._first_named(nodes, self._LOGIC_CLK_NAMES)
        q = self._first_named(nodes, self._LOGIC_Q_NAMES)
        qn = self._first_named(nodes, self._LOGIC_QN_NAMES)
        if clk is None or (q is None and qn is None):
            logger.warning(f"TFF {ref}: needs connected CLK and Q (or QN) - skipping")
            return
        params = self._logic_params(component)
        vm = self._fmt_num(params["VM"])
        vdd = self._fmt_num(params["VDD"])
        q_out = q if q is not None else f"{ref}_q"
        # D = internal Q̄: sample the complement of the held state each edge.
        self._emit_ms_dff(
            ref, f"V({ref}_qi) > {vm} ? 0 : {vdd}", clk, q_out, qn, gnd, params
        )
        self.model_provenance[ref] = ResolvedModel(
            ref, "tff", "sim_params", "tff_behavioral"
        )
        logger.debug(f"Added behavioral TFF {ref}: clk={clk} q={q} qn={qn}")

    def _add_dlatch(self, component, ref: str, value: str):
        """Behavioral level-sensitive D latch (Sim.Device="DLATCH").

        Q follows D while EN is high, holds when EN is low (one switch-held cap)."""
        nodes = self._logic_pin_nodes(component)
        if nodes is None:
            logger.warning(f"DLATCH {ref}: no live pin map; skipping")
            return
        gnd = self._logic_gnd(nodes)
        en = self._first_named(nodes, self._LOGIC_EN_NAMES | self._LOGIC_CLK_NAMES)
        d = self._first_named(nodes, self._LOGIC_D_NAMES)
        q = self._first_named(nodes, self._LOGIC_Q_NAMES)
        qn = self._first_named(nodes, self._LOGIC_QN_NAMES)
        if en is None or d is None or (q is None and qn is None):
            logger.warning(
                f"DLATCH {ref}: needs connected D, EN and Q (or QN) - skipping"
            )
            return
        params = self._logic_params(component)
        vdd = self._fmt_num(params["VDD"])
        vm = self._fmt_num(params["VM"])
        ron = 100.0
        cm = self._fmt_num(max(params["TPD"] / ron, 1e-12))
        dbuf, qint, model = f"{ref}_db", f"{ref}_qi", f"SWL{ref}"
        lines = [
            f"B{ref}_db {dbuf} {gnd} V = V({d}) > {vm} ? {vdd} : 0",
            f"S{ref}_l {dbuf} {qint} {en} {gnd} {model}",
            f"C{ref}_l {qint} {gnd} {cm} IC=0",
            f".model {model} SW(Ron={ron:g} Roff=1e9 Vt={vm} Vh={self._fmt_num(params['VM'] * 0.1 or 0.05)})",
        ]
        if q is not None:
            lines.append(f"B{ref}_q {q} {gnd} V = V({qint}) > {vm} ? {vdd} : 0")
        if qn is not None:
            lines.append(f"B{ref}_qn {qn} {gnd} V = V({qint}) > {vm} ? 0 : {vdd}")
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        self.model_provenance[ref] = ResolvedModel(
            ref, "dlatch", "sim_params", "dlatch_behavioral"
        )
        logger.debug(f"Added behavioral DLATCH {ref}: d={d} en={en} q={q} qn={qn}")

    # Triggered-breakdown primitive terminal roles (F5). Power terminals accept
    # switch/anode/collector spellings; the control is the ground-referenced
    # trigger (base/gate/trigger). Matched via _logic_pin_nodes (pin NAME, or an
    # explicit Sim.Pins="<num>=P <num>=G <num>=N" override for a bare symbol).
    _TRIGSW_P_NAMES = {"P", "A", "ANODE", "AN", "C", "COLLECTOR", "MT2", "H", "+"}
    _TRIGSW_N_NAMES = {"N", "K", "CATHODE", "KA", "E", "EMITTER", "MT1", "L", "-"}
    _TRIGSW_G_NAMES = {"G", "GATE", "TRIG", "TRIGGER", "B", "BASE", "CTL", "CONTROL"}

    def _trigsw_params(self, component) -> dict:
        """Parameters for the triggered-breakdown primitive (F5), with defaults.

        ``VT`` trigger threshold (V, default 2.5); ``RON`` on-resistance (ohm,
        default 1.2) -> on-conductance ``GON`` (a direct ``GON=`` wins); ``WIDTH``
        turn-on sharpness of the sigmoid (V, default 0.05); ``RLEAK`` off-state /
        op-point leak (ohm, default 100 Meg); optional ``VHOLD`` quench threshold
        -- when set, the conductance also falls off as the voltage across the
        switch drops below it (a smooth self-terminating quench, NOT a latch: a
        latch makes the DC op-point bistable, a documented failure mode)."""
        raw = self._parse_sim_params(self._sim_props(component).get("params"))

        def g(key, default):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        ron = g("RON", 1.2)
        gon = g("GON", None)
        if gon is None or gon <= 0:
            gon = (1.0 / ron) if (ron and ron > 0) else 0.83
        width = g("WIDTH", 0.05)
        if not width or width <= 0:
            width = 0.05
        rleak = g("RLEAK", 100e6)
        if not rleak or rleak <= 0:
            rleak = 100e6
        return {
            "GON": gon,
            "VT": g("VT", 2.5),
            "WIDTH": width,
            "RLEAK": rleak,
            "VHOLD": g("VHOLD", None),
        }

    def _add_trigsw(self, component, ref: str, value: str):
        """Behavioral triggered-breakdown switch (Sim.Device=TRIGSW/SPARKGAP/DIAC).

        One parameterized primitive for the whole triggered-breakdown /
        negative-resistance class -- avalanche transistors, spark gaps, SCR/DIAC-
        like discharge devices -- none of which ngspice models natively. It is a
        smooth (sigmoid) gated conductance across the two power terminals, NOT an
        ideal ``sw`` (which collapses the timestep in a resonant loop) and NOT a
        latched state node (whose DC op-point is bistable) -- both are documented
        failure modes from the avalanche E2E.

        Terminals (by pin NAME, or Sim.Pins override): ``P`` power+ (A/anode/C/
        collector), ``N`` power- (K/cathode/E/emitter). With a third control
        terminal ``G`` (B/base/gate/trig) the device is EXTERNALLY triggered and the
        gate references ``V(G)`` -- GROUND-referenced, the key avalanche lesson (an
        emitter/terminal-referenced trigger fails when the power- terminal floats).
        With only two terminals it is SELF-triggered on the voltage across it
        (``V(P,N)`` > VT), a spark-gap/DIAC breakover.

        Emission (params via ``_trigsw_params``)::

            B<ref>_sw   P N I = V(P,N) * GON / (1 + exp(-(<trig> - VT)/WIDTH)) [*quench]
            R<ref>_leak P N RLEAK

        Untriggered the sigmoid is ~0 (a single, unambiguous DC op-point via the
        leak); triggered it opens to GON (=1/RON). The current ``GON*V(P,N)`` self-
        terminates as the driving voltage collapses, so the ns pulse SHAPE is set by
        the EXTERNAL L-C loop, not this model. Provenance tier ``sim_params``.
        """
        nodes = self._logic_pin_nodes(component)
        if nodes is None:
            logger.warning(f"TRIGSW {ref}: no live pin map; skipping")
            return
        p = self._first_named(nodes, self._TRIGSW_P_NAMES)
        n_ = self._first_named(nodes, self._TRIGSW_N_NAMES)
        ctrl = self._first_named(nodes, self._TRIGSW_G_NAMES)
        if p is None or n_ is None:
            logger.warning(
                f"TRIGSW {ref}: needs two power terminals -- P/A/anode/C(+) and "
                f"N/K/cathode/E(-) -- resolved by pin name or Sim.Pins; skipping "
                f"(got P={p}, N={n_})"
            )
            return
        prm = self._trigsw_params(component)
        fn = self._fmt_num
        gon, vt, width, rleak = (
            fn(prm["GON"]), fn(prm["VT"]), fn(prm["WIDTH"]), fn(prm["RLEAK"])
        )
        # externally triggered on the ground-referenced control node, else self-
        # triggered on the voltage across the switch (spark-gap breakover).
        trig = f"V({ctrl})" if ctrl is not None else f"V({p},{n_})"
        gate = f"1/(1+exp(-({trig} - {vt})/{width}))"
        if prm["VHOLD"] is not None:
            vhold = fn(prm["VHOLD"])
            gate = f"({gate})*(1/(1+exp(-(V({p},{n_}) - {vhold})/{width})))"
        lines = [
            f"B{ref}_sw {p} {n_} I = V({p},{n_}) * {gon} * {gate}",
            f"R{ref}_leak {p} {n_} {rleak}",
        ]
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        mode = "ext-trig" if ctrl is not None else "self-trig"
        self.model_provenance[ref] = ResolvedModel(
            ref, "trigsw", "sim_params", f"trigsw_{mode}(vt={vt}, gon={gon})"
        )
        logger.debug(
            f"Added behavioral TRIGSW {ref} ({mode}): P={p} N={n_} "
            f"G={ctrl} vt={vt} gon={gon} rleak={rleak}"
        )

    def _gate_expr(self, op, his, vdd):
        """ngspice B-source expression for a logic gate over input predicates ``his``.

        Each entry of ``his`` is a ``(V(n) > VM)`` predicate evaluating to 1/0."""
        if op in ("BUF",):
            return f"{his[0]} ? {vdd} : 0"
        if op in ("NOT", "INV"):
            return f"{his[0]} ? 0 : {vdd}"
        if op == "AND":
            return f"({' && '.join(his)}) ? {vdd} : 0"
        if op == "NAND":
            return f"({' && '.join(his)}) ? 0 : {vdd}"
        if op == "OR":
            return f"({' || '.join(his)}) ? {vdd} : 0"
        if op == "NOR":
            return f"({' || '.join(his)}) ? 0 : {vdd}"
        if op == "XOR":  # exactly-one-high (2-input); ngspice has no reliable %
            return f"(({' + '.join(his)}) == 1) ? {vdd} : 0"
        if op == "XNOR":
            return f"(({' + '.join(his)}) == 1) ? 0 : {vdd}"
        return None

    def _add_gate(self, component, ref: str, value: str):
        """Behavioral combinational gate (Sim.Device in AND/OR/XOR/NOT/... ).

        Output resolves by name (Y/Z/OUT/Q); every other connected non-ground pin
        is an input. A TPD RC low-pass gives the output a finite edge."""
        op = str(self._sim_props(component).get("device", "")).strip().upper()
        nodes = self._logic_pin_nodes(component)
        if nodes is None:
            logger.warning(f"gate {ref} ({op}): no live pin map; skipping")
            return
        gnd = self._logic_gnd(nodes)
        out = self._first_named(nodes, self._GATE_OUT_NAMES)
        # Inputs = connected pins that are neither the output nor a ground pin.
        skip = self._LOGIC_GND_NAMES | self._GATE_OUT_NAMES
        ins = [node for name, node in nodes.items() if name not in skip]
        if out is None or not ins:
            logger.warning(
                f"gate {ref} ({op}): needs an output (Y/OUT/Q) and >=1 input pin "
                f"- skipping"
            )
            return
        if op in ("NOT", "INV", "BUF") and len(ins) > 1:
            ins = ins[:1]
        if op in ("XOR", "XNOR") and len(ins) != 2:
            if len(ins) < 2:
                logger.warning(f"gate {ref} ({op}): needs 2 inputs - skipping")
                return
            logger.warning(
                f"gate {ref} ({op}): behavioral XOR is 2-input; using the first two"
            )
            ins = ins[:2]
        params = self._logic_params(component)
        vdd = self._fmt_num(params["VDD"])
        vm = self._fmt_num(params["VM"])
        his = [f"(V({n}) > {vm})" for n in ins]
        expr = self._gate_expr(op, his, vdd)
        if expr is None:
            logger.warning(f"gate {ref}: unsupported op '{op}' - skipping")
            return
        raw = f"{ref}_raw"
        ron = 1e3
        ctp = self._fmt_num(max(params["TPD"] / ron, 1e-13))
        lines = [
            f"B{ref}_g {raw} {gnd} V = {expr}",
            f"R{ref}_tp {raw} {out} {ron:g}",
            f"C{ref}_tp {out} {gnd} {ctp}",
        ]
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        self.model_provenance[ref] = ResolvedModel(
            ref, "gate", "sim_params", f"{op.lower()}_behavioral"
        )
        logger.debug(f"Added behavioral {op} gate {ref}: out={out} ins={ins}")

    @staticmethod
    def _first_named(nodes, names):
        """First node whose (upper-cased) pin name is in ``names``, else None."""
        for name, node in nodes.items():
            if name in names:
                return str(node)
        return None

    # ------------------------------------------------------------------ #
    # Transformers: coupled inductors via a K element (Stage 21.1)        #
    # ------------------------------------------------------------------ #

    # Secondary winding letter pairs, in symbol order: SA/SB (1st secondary),
    # SC/SD (2nd), SE/SF (3rd), ... A KiCad Transformer_1P_2S adds SC/SD; wider
    # multi-secondary symbols would continue the sequence.
    _XFMR_SEC_PAIRS = (("SA", "SB"), ("SC", "SD"), ("SE", "SF"), ("SG", "SH"))

    def _transformer_pin_scan(self, component):
        """Scan a transformer's winding pins -> ``(present, nodes)`` or None.

        ``present`` is the set of winding pin *names the symbol has* (AA/AB and
        SA..SH), regardless of connection; ``nodes`` maps the *connected* ones to
        their SPICE node. Returns None when no live pin map is available (a
        dict/JSON circuit). The present/connected split lets callers tell a
        center-tapped 1P_SS (symbol has SA/SC/SB, no SD) from an independent-
        secondary 1P_2S (SA/SB/SC/SD) by pin existence, and name a specific
        unconnected winding end.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        present = set()
        nodes = {}
        for pin in pin_map.values():
            name = (getattr(pin, "name", "") or "").strip().upper()
            if not name:
                continue
            if name in ("AA", "AB") or (
                len(name) == 2 and name[0] == "S" and name[1] in "ABCDEFGH"
            ):
                present.add(name)
                net = getattr(pin, "net", None)
                if net is not None and name not in nodes:
                    nodes[name] = self.node_map.get(net.name, net.name)
        return present, nodes

    def _transformer_shape(self, component):
        """Winding topology from the symbol's pins -> ``(center_tap, n_sec)``.

        ``center_tap`` True for a 1P_SS (two half-windings about the tap SC);
        ``n_sec`` is the number of secondary windings the symbol carries (2 for a
        center-tap, one per present SA/SB, SC/SD, ... pair otherwise). Returns None
        with no live pin map (the caller then assumes a single secondary, the
        1P_1S back-compat default).
        """
        scan = self._transformer_pin_scan(component)
        if scan is None:
            return None
        present, _ = scan
        if "SC" in present and "SB" in present and "SD" not in present:
            return (True, 2)  # center-tapped secondary -> two half-windings
        n = sum(1 for a, b in self._XFMR_SEC_PAIRS if a in present or b in present)
        return (False, max(n, 1))

    def _transformer_terminals(self, component):
        """Resolve a transformer's winding SPICE nodes by pin name.

        Supports ``Transformer_1P_1S`` (AA/AB, SA/SB), ``Transformer_1P_2S``
        (adds an independent SC/SD secondary) and the center-tapped
        ``Transformer_1P_SS`` (SA/SC/SB, SC = tap -> two half-windings SA->SC and
        SC->SB). Returns
        ``{"primary": (nAA, nAB), "secondaries": [(a, b), ...], "center_tap": bool}``
        (dots at the first-named pin of each winding: AA, SA, and -- for the
        center-tap second half -- SC), or None when the primary is incomplete, a
        detected secondary winding is only half-connected (no half-wound windings),
        or no live pin map is available.
        """
        scan = self._transformer_pin_scan(component)
        if scan is None:
            return None
        present, nodes = scan
        if "AA" not in nodes or "AB" not in nodes:
            return None
        primary = (nodes["AA"], nodes["AB"])
        center_tap = "SC" in present and "SB" in present and "SD" not in present
        if center_tap:
            if not all(p in nodes for p in ("SA", "SB", "SC")):
                return None
            return {
                "primary": primary,
                "secondaries": [
                    (nodes["SA"], nodes["SC"]),
                    (nodes["SC"], nodes["SB"]),
                ],
                "center_tap": True,
            }
        secondaries = []
        for a, b in self._XFMR_SEC_PAIRS:
            if a not in present and b not in present:
                continue
            if a not in nodes or b not in nodes:
                return None  # half-wound secondary
            secondaries.append((nodes[a], nodes[b]))
        if not secondaries:
            return None
        return {"primary": primary, "secondaries": secondaries, "center_tap": False}

    def _transformer_missing_pins(self, component):
        """Winding pins the symbol has but leaves unconnected (for validate()).

        ``[]`` when every required winding end is connected or there is no live
        pin map. Names the exact SA/SC/... a half-wound transformer is missing.
        """
        scan = self._transformer_pin_scan(component)
        if scan is None:
            return []
        present, nodes = scan
        required = set()
        if "AA" in present or "AB" in present:
            required |= {"AA", "AB"}
        if "SC" in present and "SB" in present and "SD" not in present:
            required |= {"SA", "SB", "SC"}  # center-tap
        else:
            for a, b in self._XFMR_SEC_PAIRS:
                if a in present or b in present:
                    required |= {a, b}
        return sorted(p for p in required if p not in nodes)

    def _transformer_params(self, component) -> Optional[dict]:
        """Winding params for a transformer, or None if LP / a ratio is missing.

        ``LP`` (primary inductance) is required. Each secondary's inductance is an
        explicit ``LS``/``LS2``/``LS3``... or is derived as ``LP*Ni^2`` from that
        winding's turns ratio ``N``/``N2``/``N3`` (Ns/Np); the first secondary uses
        the unsuffixed ``LS``/``N``. **Center-tap semantics:** for a 1P_SS ``N`` is
        the *per-half* turns ratio -- each half winding is ``LP*N^2`` -- and both
        halves share ``N`` unless ``N2`` is given for the second half. ``K`` is the
        coupling coefficient applied to every winding pair, default 0.999; a value
        outside (0, 1] is rejected (ngspice requires it) and reported as ``K=None``.
        Returns ``{"LP", "secondaries": [ls, ...], "K", "center_tap"}``.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        lp = self._parse_si_number(raw["LP"]) if "LP" in raw else None
        if not lp or lp <= 0:
            return None
        shape = self._transformer_shape(component)
        center_tap, n_sec = shape if shape is not None else (False, 1)

        def ratio_ind(ls_key, n_key, fallback_n_key=None):
            ls = self._parse_si_number(raw[ls_key]) if ls_key in raw else None
            if ls is not None:
                return ls if ls > 0 else None
            n_ratio = self._parse_si_number(raw[n_key]) if n_key in raw else None
            if n_ratio is None and fallback_n_key and fallback_n_key in raw:
                n_ratio = self._parse_si_number(raw[fallback_n_key])
            if not n_ratio or n_ratio <= 0:
                return None
            return lp * n_ratio * n_ratio

        def winding_res(key):
            """Optional series winding resistance in ohms (>0), else 0.0."""
            if key not in raw:
                return 0.0
            r = self._parse_si_number(raw[key])
            return r if (r is not None and r > 0) else 0.0

        secondaries = []
        sec_res = []
        for i in range(1, n_sec + 1):
            if center_tap:
                # Per-half: half 1 uses N (LS), half 2 uses N2 (LS2) or falls
                # back to N (both halves the same ratio unless N2 is given).
                ls_key = "LS" if i == 1 else f"LS{i}"
                n_key = "N" if i == 1 else f"N{i}"
                ls = ratio_ind(ls_key, n_key, fallback_n_key="N")
            else:
                suffix = "" if i == 1 else str(i)
                ls = ratio_ind(f"LS{suffix}", f"N{suffix}")
            if ls is None:
                return None
            secondaries.append(ls)
            sec_res.append(winding_res("RS" if i == 1 else f"RS{i}"))

        # Optional winding DCR (C3): a real winding's series resistance breaks the
        # ideal-inductor DC degeneracy (a Royer tap settling below Vin). Absent ->
        # 0.0 -> byte-identical emission to the pre-DCR output.
        rp = winding_res("RP")

        k = self._parse_si_number(raw["K"]) if "K" in raw else 0.999
        if k is None or not (0 < k <= 1):
            # bad k -> validate() names it
            return {"LP": lp, "secondaries": secondaries, "K": None,
                    "center_tap": center_tap, "RP": rp, "sec_res": sec_res}
        return {"LP": lp, "secondaries": secondaries, "K": k,
                "center_tap": center_tap, "RP": rp, "sec_res": sec_res}

    def _add_transformer(self, component, ref: str, value: str):
        """Emit a transformer as N coupled inductors + pairwise ``K`` cards.

        One ``L`` per winding and a ``K`` card for *every* winding pair
        (primary<->secondary and secondary<->secondary -- the sec<->sec coupling
        is what makes a center-tapped/full-wave secondary behave correctly). Node
        order follows the KiCad symbol's dots (AA first, SA first, and SC first for
        the center-tap second half), so the SPICE dot convention matches the
        printed dots and winding polarity is set entirely by user wiring.

        Naming: the single-secondary (1P_1S) case emits ``L<ref>_P`` / ``L<ref>_S``
        / ``K<ref>`` -- **byte-identical to the pre-multi-winding output** (a hard
        backward-compat requirement). Multi-winding emits ``L<ref>_S1``,
        ``L<ref>_S2``, ... and ``K<ref>_PS1`` / ``K<ref>_S1S2`` / ... Winding
        currents are readable via ``branch_current("<ref>_P")`` etc.

        Simulation caveat (SPICE, not the design): every node needs a DC path to
        ground, so a galvanically isolated secondary must share the sim's GND net
        (or bridge to it through a large resistor); a center-tap grounds via the
        tap.
        """
        term = self._transformer_terminals(component)
        if term is None:
            logger.warning(
                f"transformer {ref}: winding ends unconnected - skipping "
                f"(need AA/AB + each secondary pair, or SA/SC/SB for a center-tap)"
            )
            return
        params = self._transformer_params(component)
        if params is None or params["K"] is None:
            logger.warning(
                f"transformer {ref}: unresolved winding params - skipping "
                f'(set Sim.Params="lp=100u n=0.5")'
            )
            return
        n = self._fmt_num
        lp, k = n(params["LP"]), n(params["K"])
        sec_nodes = term["secondaries"]
        sec_ind = params["secondaries"]
        rp = params.get("RP", 0.0) or 0.0
        sec_res = params.get("sec_res") or [0.0] * len(sec_ind)
        if len(sec_nodes) != len(sec_ind):
            # Winding count from terminals vs. params disagree (a param supplied
            # for a winding the symbol lacks, or vice versa) -- refuse to guess.
            logger.warning(
                f"transformer {ref}: {len(sec_nodes)} secondary winding(s) but "
                f"{len(sec_ind)} inductance(s) resolved - skipping"
            )
            return

        def winding_lines(lname, tag, node_a, node_b, ind, res):
            """The L line for a winding, prefixed by a series R when res>0.

            A series winding resistance (C3) is inserted on the A-side terminal
            through an internal node; the coupled inductor keeps its name (so the
            K cards are unchanged). res==0 -> a single L line, byte-identical to
            the pre-DCR emission.
            """
            if res and res > 0:
                internal = f"{ref}_{tag.lower()}_ri"
                return [
                    f"R{ref}_{tag} {node_a} {internal} {n(res)}",
                    f"{lname} {internal} {node_b} {n(ind)}",
                ]
            return [f"{lname} {node_a} {node_b} {n(ind)}"]

        if len(sec_nodes) == 1:
            # Single-secondary: preserve the exact legacy emission (byte-identical
            # when rp/rs are absent).
            ls = n(sec_ind[0])
            lines = (
                winding_lines(f"L{ref}_P", "P", term["primary"][0],
                              term["primary"][1], params["LP"], rp)
                + winding_lines(f"L{ref}_S", "S", sec_nodes[0][0],
                                sec_nodes[0][1], sec_ind[0], sec_res[0])
                + [f"K{ref} L{ref}_P L{ref}_S {k}"]
            )
            self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
            self.model_provenance[ref] = ResolvedModel(
                ref, "transformer", "sim_params", f"xfmr(lp={lp}, ls={ls}, k={k})"
            )
            logger.debug(
                f"{ref}: transformer as coupled inductors (lp={lp}, ls={ls}, "
                f"k={k}); dots at AA/SA per the KiCad symbol"
            )
            return

        # Multi-winding: L<ref>_P + L<ref>_S1.. and a K card per winding pair.
        lnames = [f"L{ref}_P"]
        tags = ["P"]
        lines = list(winding_lines(f"L{ref}_P", "P", term["primary"][0],
                                   term["primary"][1], params["LP"], rp))
        for i, (pair, ls) in enumerate(zip(sec_nodes, sec_ind), start=1):
            lname = f"L{ref}_S{i}"
            lnames.append(lname)
            tags.append(f"S{i}")
            lines += winding_lines(lname, f"S{i}", pair[0], pair[1], ls,
                                   sec_res[i - 1])
        for (ia, la), (ib, lb) in combinations(list(enumerate(lnames)), 2):
            lines.append(f"K{ref}_{tags[ia]}{tags[ib]} {la} {lb} {k}")
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        ls_str = ", ".join(n(x) for x in sec_ind)
        shape = "center-tap" if term.get("center_tap") else f"{len(sec_nodes)}-sec"
        self.model_provenance[ref] = ResolvedModel(
            ref, "transformer", "sim_params",
            f"xfmr_{shape}(lp={lp}, ls=[{ls_str}], k={k})",
        )
        logger.debug(
            f"{ref}: {shape} transformer as {len(lnames)} coupled inductors "
            f"(lp={lp}, ls=[{ls_str}], k={k}); pairwise K; dots at first-named "
            f"pin of each winding"
        )

    # ------------------------------------------------------------------ #
    # Switching regulators: behavioral buck/boost macromodel (Stage 20.3) #
    # ------------------------------------------------------------------ #

    # Pin names (upper-cased) identifying a switcher's terminals. SW/VIN/GND are
    # required by the model; FB is the user's divider tap (not read by the
    # open-loop model, resolved only for completeness). First connected match wins.
    _SWITCH_SW_NAMES = {"SW", "LX", "SWITCH", "PH", "L1", "LX1"}
    _SWITCH_FB_NAMES = {"FB", "VFB", "FEEDBACK", "VSENSE", "SENSE"}
    _SWITCH_VIN_NAMES = {"VIN", "IN", "PVIN", "AVIN", "VCC"}
    _SWITCH_GND_NAMES = {"GND", "PGND", "AGND", "EP"}

    def _switcher_terminals(self, component):
        """Resolve a switcher's SW/VIN/GND (and optional FB) SPICE nodes by pin name.

        Returns a dict ``{"sw","vin","gnd","fb"}`` (fb may be None) or None when a
        required terminal is missing or no live pin map is available.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        sw = vin = gnd = fb = None
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            name = (getattr(pin, "name", "") or "").strip().upper()
            node = self.node_map.get(net.name, net.name)
            if sw is None and name in self._SWITCH_SW_NAMES:
                sw = node
            elif vin is None and name in self._SWITCH_VIN_NAMES:
                vin = node
            elif gnd is None and name in self._SWITCH_GND_NAMES:
                gnd = node
            elif fb is None and name in self._SWITCH_FB_NAMES:
                fb = node
        if sw is None or vin is None or gnd is None:
            return None
        return {"sw": sw, "vin": vin, "gnd": gnd, "fb": fb}

    # Second switch node (boost-leg / SEPIC node B) and the explicit output node,
    # for the multi-switch bidirectional families (Stage 27). The 3-node
    # _switcher_terminals resolver can express neither a second SW node nor a
    # distinct (possibly negative) VOUT.
    _SWITCH_SW2_NAMES = {"SW2", "SWB", "LX2", "PH2"}
    _SWITCH_VOUT_NAMES = {"VOUT", "OUT", "VO"}
    # Error-amp compensation (VC/ITH/COMP) pin, needed only by the averaged
    # peak-current-mode loop model (Stage 28.D): the loop closes through the
    # user's real external VC network, so this pin must be resolvable by name.
    _SWITCH_VC_NAMES = {"VC", "COMP", "ITH", "COMPENSATION"}

    def _multiswitch_terminals(self, component, kind):
        """Resolve the terminals a multi-switch converter ``kind`` needs, by name.

        Generalizes ``_switcher_terminals`` for the Stage-27 bidirectional
        families. Resolves each pin by its (upper-cased) NAME -- not position --
        for the same reason the transistor mapping does (KiCad symbols number
        pins inconsistently). First connected match per role wins.

        Returns a dict of SPICE nodes shaped for ``kind``:

        - ``buckboost4`` -> ``{vin, swa, swb, vout, gnd}`` (swa = buck-leg node
          from SW/LX..., swb = boost-leg node from SW2/SWB/LX2/PH2);
        - ``invbuckboost`` -> ``{vin, sw, vout, gnd}`` (vout is the negative
          output node; the single switch node comes from SW/LX...);
        - ``sepic`` -> ``{vin, swa, swb, vout, gnd}`` (swa = node A, swb = node B,
          with the coupling cap Cs a real external part between them);
        - ``cuk`` -> ``{vin, swa, swb, vout, gnd}`` (same five terminals as sepic;
          vout is the NEGATIVE output behind the real output inductor L2, not a
          switch node -- swb->vout is the L2 the emitter leaves in place).

        Returns None when any node ``kind`` requires is unresolved (or no live
        pin map is available), so the handler can log an actionable warning and
        skip.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        found = {"vin": None, "swa": None, "swb": None, "vout": None, "gnd": None}
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            name = (getattr(pin, "name", "") or "").strip().upper()
            node = self.node_map.get(net.name, net.name)
            if found["swa"] is None and name in self._SWITCH_SW_NAMES:
                found["swa"] = node
            elif found["swb"] is None and name in self._SWITCH_SW2_NAMES:
                found["swb"] = node
            elif found["vin"] is None and name in self._SWITCH_VIN_NAMES:
                found["vin"] = node
            elif found["vout"] is None and name in self._SWITCH_VOUT_NAMES:
                found["vout"] = node
            elif found["gnd"] is None and name in self._SWITCH_GND_NAMES:
                found["gnd"] = node
        required = {
            "buckboost4": ("vin", "swa", "swb", "vout", "gnd"),
            "invbuckboost": ("vin", "swa", "vout", "gnd"),
            "sepic": ("vin", "swa", "swb", "vout", "gnd"),
            "cuk": ("vin", "swa", "swb", "vout", "gnd"),
        }.get(kind)
        if required is None:
            return None
        if any(found[k] is None for k in required):
            return None
        if kind == "invbuckboost":
            return {
                "vin": found["vin"],
                "sw": found["swa"],
                "vout": found["vout"],
                "gnd": found["gnd"],
            }
        return {k: found[k] for k in required}

    def _switcher_params(self, component, value, topology) -> Optional[dict]:
        """Macromodel params for a switcher, or None if a required one is missing.

        VOUT (target output) and FSW (switching frequency) are required; a flyback
        additionally requires N (the transformer turns ratio Ns/Np -- the CCM duty
        needs it, and it cannot be read from the separate transformer component).
        VF (diode drop used for the first-order duty correction), DMAX, RON_HS,
        VRAMP and VCLAMP (the flyback's emitted drain-clamp breakdown) take
        defaults. All parsed with the milli-aware SI parser.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        vout = self._parse_si_number(raw["VOUT"]) if "VOUT" in raw else None
        fsw = self._parse_si_number(raw["FSW"]) if "FSW" in raw else None
        if not vout or vout <= 0 or not fsw or fsw <= 0:
            return None

        def g(key, default):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        params = {
            "VOUT": vout,
            "FSW": fsw,
            "VF": g("VF", 0.45),
            "DMAX": g("DMAX", 0.95 if topology == "buck" else 0.9),
            "RON_HS": g("RON_HS", 0.1),
            "VRAMP": g("VRAMP", 1.0),
        }
        if topology == "flyback":
            n_ratio = self._parse_si_number(raw["N"]) if "N" in raw else None
            if not n_ratio or n_ratio <= 0:
                return None
            params["N"] = n_ratio
            params["VCLAMP"] = g("VCLAMP", 150.0)
        return params

    def _switcher_mode(self, component) -> str:
        """Model variant for a switcher: ``'avg'`` (averaged, for loop stability)
        or ``'cycle'`` (the default cycle-accurate 20.3 model).

        Read from ``Sim.Params`` ``MODE=`` (case-insensitive); anything other than
        ``avg`` -- including absent -- is ``cycle``.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        return "avg" if str(raw.get("MODE", "")).strip().lower() == "avg" else "cycle"

    def _switcher_cmode(self, component) -> str:
        """Control-mode variant of the averaged model: ``'peak'`` (peak
        current-mode, Stage 28.D) or ``''`` (voltage-mode, today's default).

        Read from ``Sim.Params`` ``CMODE=`` (case-insensitive); anything other than
        ``peak`` -- including absent -- selects the voltage-mode averaged buck (so a
        plain ``MODE=avg`` stays byte-identical to Stage 20.5). Only consulted when
        ``_switcher_mode`` is already ``'avg'``.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        return "peak" if str(raw.get("CMODE", "")).strip().lower() == "peak" else ""

    def _currentmode_terminals(self, component):
        """Resolve the terminals the averaged peak-current-mode loop needs, by name.

        Returns ``{"fb","vc","vout","gnd","vin"}`` (``vin`` may be None -- it is not
        used by the small-signal loop, only resolved for completeness) or None when
        a required node (FB, VC, VOUT, GND) is missing or no live pin map exists.

        Unlike ``_switcher_terminals`` this resolves the **VC** (compensation) and
        **VOUT** nodes -- the current-mode loop closes the error amp into the real
        external VC network and injects the averaged output current into the real
        VOUT/Cout/Rload, so both must be pins. FB is the divider tap (error-amp
        input); ``FBX`` (the LT3757 single-pin dual-reference feedback) is accepted
        alongside the usual FB names. First connected match per role wins.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        fb_names = self._SWITCH_FB_NAMES | {"FBX"}
        fb = vc = vout = gnd = vin = None
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            name = (getattr(pin, "name", "") or "").strip().upper()
            node = self.node_map.get(net.name, net.name)
            if fb is None and name in fb_names:
                fb = node
            elif vc is None and name in self._SWITCH_VC_NAMES:
                vc = node
            elif vout is None and name in self._SWITCH_VOUT_NAMES:
                vout = node
            elif gnd is None and name in self._SWITCH_GND_NAMES:
                gnd = node
            elif vin is None and name in self._SWITCH_VIN_NAMES:
                vin = node
        if fb is None or vc is None or vout is None or gnd is None:
            return None
        return {"fb": fb, "vc": vc, "vout": vout, "gnd": gnd, "vin": vin}

    def _averaged_current_mode_params(self, component, topology) -> Optional[dict]:
        """Params for the averaged peak-current-mode loop model, or None if unusable.

        Extends ``_averaged_params`` with the current-mode knobs. Requires a nonzero
        ``VREF`` (the FBX reference the divider tap regulates to -- **may be negative**
        for the -0.8 V inverting configuration, unlike the voltage-mode buck which
        requires a positive VREF) and ``FSW`` (sets the fsw/2 subharmonic double
        pole). The steady-state duty ``D`` (for the RHP-zero frequency and the
        subharmonic Q) is derived from ``VOUT``/``VIN`` per topology, or given
        directly as ``D``.

        Current-mode knobs (all with datasheet-anchored defaults, documented):
          * ``GM``   error-amp transconductance (default 250 uS, the LT3757 value);
          * ``RI``   effective current-sense gain in ohms = Rsense * sense-amp-gain
                     (default 0.1 = 0.01 R sense x 10 V/V), sets the modulator gain
                     ``gmc = (1-D)/RI`` (boost/sepic/cuk) or ``1/RI`` (buck);
          * ``MC``   slope-compensation factor mc = 1 + Se/Sn (default 1.5), sets the
                     subharmonic Q ``Qp = 1/(pi*(mc*(1-D) - 0.5))``;
          * ``REA``  finite error-amp DC-gain resistor (default 1e6) -- a DC-path
                     safety so the op point solves; the AC zero/pole come from the
                     user's real Rc/Cc, not this;
          * RHP zero (boost/sepic/cuk only): ``FRHPZ`` in Hz directly, else derived
                     ``Rload*(1-D)^2/(2*pi*L)`` from ``RLOAD`` and ``L`` params.
                     Buck has no RHP zero (omitted).

        Returns None when VREF is zero/absent, FSW is missing/invalid, or D cannot
        be resolved.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))

        def g(key, default=None):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        vref = self._parse_si_number(raw["VREF"]) if "VREF" in raw else None
        if not vref or vref == 0:
            return None
        fsw = g("FSW")
        if not fsw or fsw <= 0:
            return None

        d = g("D")
        if d is None:
            vout, vin = g("VOUT"), g("VIN")
            if vout and vin and vin > 0 and abs(vout) > 0:
                av = abs(vout)
                if topology == "buck":
                    d = av / vin if vin > 0 else None
                elif topology == "boost":
                    d = 1.0 - vin / av if av > vin else None
                else:  # sepic / cuk: |Vout|/(|Vout|+Vin)
                    d = av / (av + vin)
            if d is None:
                return None
        d = min(max(d, 0.01), 0.99)

        mc = g("MC", 1.5)
        ri = g("RI", 0.1)
        gm = g("GM", 250e-6)
        rea = g("REA", 1e6)

        # RHP zero (boost/sepic/cuk). FRHPZ wins; else derive from RLOAD and L.
        frhpz = None
        if topology != "buck":
            frhpz = g("FRHPZ")
            if frhpz is None:
                rload, lval = g("RLOAD"), g("L")
                if rload and rload > 0 and lval and lval > 0:
                    frhpz = rload * (1.0 - d) ** 2 / (2.0 * math.pi * lval)

        return {
            "VREF": vref,
            "FSW": fsw,
            "D": d,
            "MC": mc,
            "RI": ri,
            "GM": gm,
            "REA": rea,
            "FRHPZ": frhpz,
        }

    def _cmcontroller_terminals(self, component):
        """Resolve the CMCONTROLLER's VIN/SW/VOUT/FB/VC/GND SPICE nodes by pin name.

        The behavioral closed-loop controller (Stage 29.1) emits its own switch stage
        between VIN and SW and closes the loop through the user's real divider
        (VOUT->FB->GND) and VC compensation network, so all six terminals must be
        real pins. Returns ``{"vin","sw","vout","fb","vc","gnd"}`` or None when any is
        missing (or no live pin map is available), so the handler logs an actionable
        skip. FB accepts the usual divider-tap names plus ``FBX`` (the LT3757
        dual-reference feedback), same as the averaged current-mode resolver. First
        connected match per role wins.
        """
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        fb_names = self._SWITCH_FB_NAMES | {"FBX"}
        vin = sw = vout = fb = vc = gnd = None
        for pin in pin_map.values():
            net = getattr(pin, "net", None)
            if net is None:
                continue
            name = (getattr(pin, "name", "") or "").strip().upper()
            node = self.node_map.get(net.name, net.name)
            if sw is None and name in self._SWITCH_SW_NAMES:
                sw = node
            elif vin is None and name in self._SWITCH_VIN_NAMES:
                vin = node
            elif vout is None and name in self._SWITCH_VOUT_NAMES:
                vout = node
            elif vc is None and name in self._SWITCH_VC_NAMES:
                vc = node
            elif fb is None and name in fb_names:
                fb = node
            elif gnd is None and name in self._SWITCH_GND_NAMES:
                gnd = node
        if None in (vin, sw, vout, fb, vc, gnd):
            return None
        return {"vin": vin, "sw": sw, "vout": vout, "fb": fb, "vc": vc, "gnd": gnd}

    def _cmcontroller_params(self, component) -> Optional[dict]:
        """Params for the behavioral closed-loop CMCONTROLLER (Stage 29.1), or None.

        Requires a nonzero VREF (the FB reference; **may be negative**, the 28.D
        convention) and a positive FSW. Everything else is a datasheet-anchored
        default matching the averaged current-mode model's knobs, so the two regimes
        cross-check:

          * ``GM``   error-amp transconductance (default 250 uS, the LT3757 value);
          * ``RI``   effective current-sense gain (ohms) = Rsense * sense-amp gain
                     (default 0.1); scales the sensed inductor current into the
                     comparator;
          * ``MCSLOPE``  volts of slope-compensation ramp added across one period
                     into the current signal (default 0.1). Raise it if a D>0.5 run
                     subharmonic-oscillates (Basso Fig. 5c/d); aliased ``MC_SLOPE``;
          * ``REA``  finite error-amp DC-gain resistor on VC (default 1e6), a DC-path
                     safety (the AC shaping is the user's real Rth/Cth). May instead
                     be set implicitly via ``GAIN`` (below);
          * ``GAIN`` error-amp DC open-loop gain in dB (Stage 29.2). When given (and
                     ``REA`` is not), REA is derived as ``10^(GAIN/20)/GM`` so the
                     gm-into-REA DC gain equals GAIN; an explicit REA always wins;
          * ``VHIGH``/``VLOW``  error-amp output-swing clamps on VC (default 2.4 / 0.0
                     V; Stage 29.2). The VC/ITH pin cannot exceed these internal rails
                     -- the ceiling soft-start ramps up to and the anti-windup floor.
                     Realized as soft diode clamps (never a hard voltage clamp, which
                     would zero the integrator Jacobian);
          * ``ISOURCE``/``ISINK``  error-amp source/sink current limits in amps
                     (default 1e-3 each; Stage 29.2). The gm cell can only push/pull a
                     finite current into the ITH pin -- clamped on the CURRENT
                     (``min``/``max``), which slew-limits VC without floating the node;
          * ``RON``  emitted switch on-resistance in ohms (default 0.1);
          * ``DMAX`` max duty (default 0.9); a max-duty pulse force-resets the latch
                     past it, bounding inrush even before soft-start;
          * ``TSS``  soft-start / soft-reference ramp time in seconds (default 0 = a
                     hard DC reference). A nonzero TSS ramps the internal reference
                     from 0 to VREF over TSS, so the rail rises monotonically and the
                     startup inrush stays bounded -- the Stage 29.3 soft-start. Also
                     eases the UIC ``.tran`` op point;
          * ``TOPOLOGY``  the switch stage to emit (Stage 29.1: ``buck`` only).

        Supervisory features (Stage 29.3) -- each behind its own param, ABSENT = the
        feature is off and the emission stays byte-identical to 29.1/29.2:

          * ``VSENSE_MAX``  cycle-by-cycle peak current limit as a sense voltage
                     (e.g. 0.11 for 110 mV). When set, the latch also resets when the
                     RAW sensed current ``RI*I(isns)`` exceeds it (slope comp excluded),
                     so the switch current can never command past the datasheet limit --
                     into a short the rail collapses at constant current instead of
                     running away (Basso Fig. 6 ``ILIMIT``, OR'd into the reset);
          * ``TON_MIN``  minimum on-time / leading-edge blanking in seconds. When set,
                     the reset comparator is ignored for the first ``TON_MIN`` of each
                     cycle (``V(ramp) < TON_MIN*FSW``), enforcing a floor on the on-time
                     and blanking current-sense noise at the switch edge;
          * ``FB_FOLD``  frequency-foldback FB threshold. When ``V(FB)`` is below it
                     (startup into a heavy load / a short), the latch is clocked by a
                     slower folded clock at ``FSW*FOLD_RATIO`` instead of ``FSW``, so
                     the inductor current stays controllable at the forced short on-time.
                     This is the documented TWO-STATE foldback (a dual fixed-frequency
                     clock select), not a continuously variable oscillator -- robust and
                     convergence-safe;
          * ``FOLD_RATIO``  folded-clock frequency as a fraction of FSW (default 0.25);
                     only used when ``FB_FOLD`` is set;
          * ``UVLO_RISE``/``UVLO_FALL``  under-voltage lockout on VIN with hysteresis.
                     When ``UVLO_RISE`` is set, the gate is held off until ``V(VIN)``
                     rises past ``UVLO_RISE`` and turns off again below ``UVLO_FALL``
                     (default ``0.9*UVLO_RISE``); realized with a hysteretic run latch
                     (the switch-held-memory pattern) gating the gate output.

        Returns None when VREF is zero/absent or FSW is missing/invalid.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))

        def g(key, default=None):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        vref = self._parse_si_number(raw["VREF"]) if "VREF" in raw else None
        if not vref or vref == 0:
            return None
        fsw = g("FSW")
        if not fsw or fsw <= 0:
            return None
        topology = str(raw.get("TOPOLOGY", "buck")).strip().lower() or "buck"
        gm = g("GM", 250e-6)
        # Finite error-amp DC gain: an explicit REA wins; else derive from GAIN (dB)
        # as 10^(GAIN/20)/GM so the gm-into-REA DC gain equals GAIN; else the default.
        if "REA" in raw:
            rea = g("REA", 1e6)
        elif "GAIN" in raw:
            gain_db = self._parse_si_number(raw["GAIN"])
            rea = (10.0 ** (gain_db / 20.0)) / gm if (gain_db is not None and gm) else 1e6
        else:
            rea = 1e6
        # Supervisory features (Stage 29.3): each defaults to None (= off) so the
        # emission is byte-identical to 29.1/29.2 when the param is absent.
        uvlo_rise = g("UVLO_RISE", None)
        uvlo_fall = g("UVLO_FALL", None)
        if uvlo_rise is not None and uvlo_fall is None:
            uvlo_fall = 0.9 * uvlo_rise
        return {
            "VREF": vref,
            "FSW": fsw,
            "GM": gm,
            "RI": g("RI", 0.1),
            "MCSLOPE": g("MCSLOPE", g("MC_SLOPE", 0.1)),
            "REA": rea,
            "VHIGH": g("VHIGH", 2.4),
            "VLOW": g("VLOW", 0.0),
            "ISOURCE": g("ISOURCE", 1e-3),
            "ISINK": g("ISINK", 1e-3),
            "RON": g("RON", 0.1),
            "DMAX": g("DMAX", 0.9),
            "TSS": g("TSS", 0.0),
            "TOPOLOGY": topology,
            # supervisory (None = off, byte-identical emission)
            "VSENSE_MAX": g("VSENSE_MAX", None),
            "TON_MIN": g("TON_MIN", None),
            "FB_FOLD": g("FB_FOLD", None),
            "FOLD_RATIO": g("FOLD_RATIO", 0.25),
            "UVLO_RISE": uvlo_rise,
            "UVLO_FALL": uvlo_fall,
        }

    def _averaged_params(self, component) -> Optional[dict]:
        """Error-amp params for the averaged model, or None if VREF is missing.

        VREF (the controller's internal reference the divider tap regulates to) is
        required -- the averaged loop closes through the user's real divider, so the
        output is set by VREF and that ratio, not by VOUT. GM (error-amp
        transconductance), CEA (compensation cap) and REA (finite-DC-gain resistor)
        default to a plain gm-C integrator and are the knobs the user tunes for
        phase margin.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        vref = self._parse_si_number(raw["VREF"]) if "VREF" in raw else None
        if not vref or vref <= 0:
            return None

        def g(key, default):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        return {
            "VREF": vref,
            "GM": g("GM", 1e-3),
            "CEA": g("CEA", 1e-7),
            "REA": g("REA", 1e6),
        }

    def _add_buck(self, component, ref: str, value: str):
        self._add_switching_regulator(component, ref, value, "buck")

    def _add_boost(self, component, ref: str, value: str):
        self._add_switching_regulator(component, ref, value, "boost")

    def _add_flyback(self, component, ref: str, value: str):
        self._add_switching_regulator(component, ref, value, "flyback")

    def _add_switching_regulator(self, component, ref, value, topology):
        """Emit a behavioral buck/boost/flyback macromodel replacing only the IC.

        v1 is a **computed-duty open-loop** model (the closed loop was found to
        limit-cycle in voltage mode; open-loop is robust and steady-state-accurate):

          * a sawtooth PWM ramp at FSW,
          * a duty from VOUT/VIN with a first-order diode-drop correction (buck:
            ``D=(VOUT+VF)/(VIN+VF)``; boost: ``D=1-VIN/(VOUT+VF)``; flyback CCM:
            ``D=(VOUT+VF)/((VOUT+VF)+N*VIN)``) -- tracks line, not load,
          * a comparator gating an ``S`` switch (buck: high-side + emitted freewheel
            diode; boost/flyback: low-side, relying on the user's external rectifier
            diode). A flyback additionally gets a **drain avalanche clamp** diode
            (``BV=VCLAMP``, default 150 V): without it, transformer leakage dumped
            into the ideal open switch rings to kV and ~30x-inflates the timepoint
            count -- a real integrated flyback switch avalanches at its rating.

        The inductor/transformer, output cap and feedback divider stay the user's
        real parts. Emitted via ``raw_spice`` (behavioral sources + a switch +
        ``.model`` cards). Limitations (documented, not silently hidden): no active
        load-step recovery (open loop), non-synchronous (diode Vf, so sync-rectifier
        efficiency is underestimated), no current limit; the flyback duty is the CCM
        formula (light-load/DCM reads high) and clamp dissipation is modeled but not
        reported as loss. Boost and flyback need UIC to converge (boost: start
        V(out) at V(in); flyback: start V(out) at 0); buck converges without it.
        """
        # MODE=avg CMODE=peak selects the averaged peak-current-mode LOOP model
        # (Stage 28.D), for .ac crossover/phase-margin of the VC-pin compensation
        # network. It resolves its OWN terminals (FB/VC/VOUT, not the SW node the
        # power-stage models need), so branch before the SW/VIN/GND gate. Only
        # buck/boost are wired (flyback current-mode is out of scope); a plain
        # MODE=avg (no CMODE) still takes the voltage-mode buck path below,
        # byte-identical to Stage 20.5.
        if (
            topology in ("buck", "boost")
            and self._switcher_mode(component) == "avg"
            and self._switcher_cmode(component) == "peak"
        ):
            self._emit_averaged_current_mode(component, ref, value, topology)
            return

        term = self._switcher_terminals(component)
        if term is None:
            logger.warning(
                f"{topology} {ref}: could not resolve SW/VIN/GND terminals - skipping"
            )
            return
        params = self._switcher_params(component, value, topology)
        if params is None:
            need = "VOUT/FSW/N" if topology == "flyback" else "VOUT/FSW"
            example = (
                "fsw=100k vout=5 n=0.5"
                if topology == "flyback"
                else "fsw=500k vout=3.3"
            )
            logger.warning(
                f"{topology} {ref}: no {need} resolved - skipping "
                f'(set Sim.Params="{example}")'
            )
            return

        # Give cap-only unmodeled pins (SS/BYP/EN/PG...) a DC path so the op-point
        # solves. Done before the MODE branch so both models get the stubs.
        self._stub_unmodeled_pins(
            component,
            ref,
            {term["sw"], term["vin"], term["gnd"], term["fb"]},
            term["gnd"],
        )

        # MODE=avg selects the averaged (non-switching) model for loop-stability
        # analysis. v1 is voltage-mode buck only; boost falls back to the cycle
        # model with a clear warning (honest, not silent).
        if self._switcher_mode(component) == "avg":
            if topology == "buck":
                self._emit_averaged_buck(component, ref, term, params)
                return
            logger.warning(
                f"{topology} {ref}: MODE=avg is voltage-mode-buck-only in v1; "
                f"using the cycle-accurate model instead"
            )

        sw, vin, gnd = str(term["sw"]), str(term["vin"]), str(term["gnd"])
        n = self._fmt_num
        vout = n(params["VOUT"])
        vf = n(params["VF"])
        dmax = n(params["DMAX"])
        ron = n(params["RON_HS"])
        vramp = n(params["VRAMP"])
        per = 1.0 / params["FSW"]
        saw, dd, gg = f"{ref}_saw", f"{ref}_d", f"{ref}_g"

        if topology == "buck":
            duty = f"({vout}+{vf})/(V({vin})+{vf})"
        elif topology == "boost":
            duty = f"1 - V({vin})/({vout}+{vf})"
        else:  # flyback CCM: VOUT = VIN*N*D/(1-D), first-order diode correction
            duty = f"({vout}+{vf})/(({vout}+{vf}) + {n(params['N'])}*V({vin}))"

        lines = [
            f"V{ref}_saw {saw} {gnd} PULSE(0 {vramp} 0 "
            f"{per * 0.99:.6g} {per * 0.005:.6g} {per * 0.005:.6g} {per:.6g})",
            f"B{ref}_d {dd} {gnd} V = min(max({duty}, 0.0), {dmax})",
            f"B{ref}_g {gg} {gnd} V = V({dd}) > V({saw}) ? 5 : 0",
        ]
        if topology == "buck":
            lines += [
                f"S{ref}_hs {vin} {sw} {gg} {gnd} SW{ref}",
                f".model SW{ref} SW(Ron={ron} Roff=1e6 Vt=2.5 Vh=0.2)",
                f"D{ref}_fw {gnd} {sw} DFW{ref}",
                f".model DFW{ref} D(IS=1e-9 N=1.05 CJO=100p)",
            ]
        else:  # boost/flyback: low-side switch; the rectifier diode is the user's
            lines += [
                f"S{ref}_ls {sw} {gnd} {gg} {gnd} SW{ref}",
                f".model SW{ref} SW(Ron={ron} Roff=1e6 Vt=2.5 Vh=0.2)",
            ]
        if topology == "flyback":
            # Drain avalanche clamp: bounds the leakage spike at the IC's rated
            # breakdown, exactly as an integrated flyback switch does.
            vclamp = n(params["VCLAMP"])
            lines += [
                f"D{ref}_cl {gnd} {sw} DCL{ref}",
                f".model DCL{ref} D(IS=1e-12 BV={vclamp} IBV=1m)",
            ]

        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        name = f"{topology}_openloop(vout={vout})"
        if topology == "flyback":
            name = f"flyback_openloop(vout={vout}, n={n(params['N'])})"
        self.model_provenance[ref] = ResolvedModel(ref, topology, "sim_params", name)
        extra = (
            f"; drain clamp at BV={n(params['VCLAMP'])} V (set vclamp= to the IC's "
            f"rating); CCM duty formula (light-load/DCM reads high)"
            if topology == "flyback"
            else ""
        )
        logger.debug(
            f"{ref}: {topology} behavioral macromodel (open-loop, vout={vout}, "
            f"fsw={self._fmt_hz(params['FSW'])}); active load-step recovery is not "
            f"modeled (open loop){extra}"
        )

    def _emit_averaged_buck(self, component, ref, term, params):
        """Emit the averaged (non-switching) voltage-mode buck model (Stage 20.5).

        Replaces the sawtooth/comparator/switch/diode of the cycle model with a
        continuous averaged PWM switch and a linear gm-C error amplifier, so ``.ac``
        linearizes it and produces a real loop gain (crossover / phase margin) --
        the question the cycle-accurate model structurally cannot answer::

            B<ref>_ea  <gnd> <ref>_c  I = GM*(VREF - V(<fb>))   ; error amp (integrator)
            C<ref>_ea  <ref>_c <gnd> CEA                         ; compensation cap
            R<ref>_ea  <ref>_c <gnd> REA                         ; finite DC gain
            B<ref>_sw  <ref>_swi <gnd> V = V(<ref>_c) * V(<vin>) ; averaged switch (d=V(c))
            R<ref>_sw  <ref>_swi <sw> RON_HS                     ; conduction loss

        The control voltage V(c) *is* the duty (in 0..1 at the operating point of a
        well-specified buck). The multiplicative averaged switch linearizes to a
        d->vout gain of ~VIN around the DC operating point, so the user's real L /
        Cout / divider supply the plant dynamics and the loop closes through the
        divider tap. Negative feedback: Vout low -> V(fb) low -> error>0 -> V(c) up
        -> duty up -> Vout up.

        No duty clamp is emitted: it is irrelevant to the small-signal ``.ac`` this
        model exists for, and a hard ``min/max`` saturation zeroes the Jacobian and
        breaks DC operating-point convergence (the clamp hides the feedback path
        from Newton during startup). Validity (documented, logged): CCM voltage-mode
        only; results above ~FSW/2 are meaningless (averaging breaks); no current
        limit.
        """
        avg = self._averaged_params(component)
        if avg is None:
            logger.warning(
                f"buck {ref}: MODE=avg needs Sim.Params VREF (the reference the "
                f'divider tap regulates to), e.g. "vref=0.8" - skipping'
            )
            return
        fb = term.get("fb")
        if fb is None:
            logger.warning(
                f"buck {ref}: MODE=avg needs a connected FB pin (the divider tap) "
                f"to close the loop - skipping"
            )
            return

        sw, vin, gnd = str(term["sw"]), str(term["vin"]), str(term["gnd"])
        fb = str(fb)
        n = self._fmt_num
        vref, gm, cea, rea = (
            n(avg["VREF"]),
            n(avg["GM"]),
            n(avg["CEA"]),
            n(avg["REA"]),
        )
        ron = n(params["RON_HS"])
        cc, swi = f"{ref}_c", f"{ref}_swi"

        lines = [
            f"B{ref}_ea {gnd} {cc} I = {gm}*({vref} - V({fb}))",
            f"C{ref}_ea {cc} {gnd} {cea}",
            f"R{ref}_ea {cc} {gnd} {rea}",
            f"B{ref}_sw {swi} {gnd} V = V({cc}) * V({vin})",
            f"R{ref}_sw {swi} {sw} {ron}",
        ]
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        self.model_provenance[ref] = ResolvedModel(
            ref, "buck", "sim_params", f"buck_averaged(vref={vref})"
        )
        fsw = params["FSW"]
        logger.debug(
            f"{ref}: buck averaged macromodel (voltage-mode, CCM, vref={vref}, "
            f"gm={gm}, cea={cea}); for loop-gain/phase-margin via .ac. Results above "
            f"~{self._fmt_hz(fsw / 2)} (FSW/2) are not physical (averaging breaks)."
        )

    def _emit_averaged_current_mode(self, component, ref, value, topology):
        """Emit the averaged **peak-current-mode** loop model (Stage 28.D).

        A small-signal, ``.ac``-linearizable macromodel for **compensation design**
        -- crossover / phase margin / gain margin of the VC-pin network on a
        current-mode boost/buck (SEPIC/Ćuk share the boost RHP-zero shape). It is
        NOT a cycle-accurate controller and NOT a closed-loop switching sim: no
        soft-start, no current-limit/foldback, no burst mode, no SYNC, no
        large-signal startup (those are the deferred ``CMCONTROLLER`` step).

        Structurally different from the voltage-mode ``_emit_averaged_buck``: the
        inner current loop makes the inductor a *controlled current source*, so the
        control-to-output plant is (to first order) single-pole ``1/(Rload*Cout)``
        plus the ESR zero -- both supplied by the user's **real** Cout/Rload -- and
        the model injects the averaged output current directly into VOUT. Emitted,
        per ref (all behavioral ``B`` sources + linear ``R``/``L``/``C`` so ``.ac``
        linearizes cleanly, no switching)::

            B<ref>_ea   0 <vc>  I = GM*(VREF - V(<fb>))   ; gm error amp into the
            R<ref>_ea  <vc> <gnd> REA                     ;   real external VC network
            B<ref>_shin <shin> <gnd> V = V(<vc>)          ; buffer (no load on VC net)
            R<ref>_sh  <shin> <nrh> RH                    ; subharmonic double pole at
            L<ref>_sh  <nrh> <sh>  LH                     ;   fsw/2, Q=1/(pi*(mc*D'-.5))
            C<ref>_sh  <sh>  <gnd> CH                     ;   (unity DC, 2-pole LP)
            C<ref>_z   <sh>  <nz>  CZ                     ; RHP zero (boost/sepic/cuk):
            V<ref>_z   <nz>  <gnd> DC 0                   ;   V(rz)=V(sh)*(1 - s/wz)
            B<ref>_rz  <rz>  <gnd> V = V(<sh>) - KZ*I(V<ref>_z)   ;  via sensed dV/dt
            B<ref>_out 0 <vout> I = GMC*V(<rz>)           ; controlled output current

        Negative feedback: VOUT up -> V(fb) up -> (VREF-fb) down -> less current into
        VC -> V(vc) down -> less GMC*V(rz) injected -> VOUT down. The real Cout/Rload
        integrate the injected current into the dominant pole (+ ESR zero if Cout has
        ESR); the real Rc/Cc on VC set the type-II compensation zero/pole; REA is a
        DC-path safety (finite error-amp DC gain), not the AC shaping.

        The physical boost inductor L (VIN->SW) is **not** in the small-signal path
        (the inner current loop subsumes its pole); its value enters only through the
        RHP-zero frequency (``L``/``FRHPZ`` param). Any SW/VIN pins are DC-tied via
        the 1G unmodeled-pin stubs. Validity (documented, logged): CCM, small-signal,
        results above ~fsw/2 are meaningless; no current limit, no slope-comp
        large-signal instability. A *design-margin* tool, not a validated reference.

        No hard duty clamp (same reason as the buck: a min/max saturation zeroes the
        DC Jacobian). Emits nothing and warns (honest skip) when the terminals or the
        required params (VREF/FSW/D) don't resolve.
        """
        term = self._currentmode_terminals(component)
        if term is None:
            logger.warning(
                f"{topology} {ref}: MODE=avg CMODE=peak needs connected FB, VC "
                f"(compensation) and VOUT pins (resolved by pin name) - skipping"
            )
            return
        cm = self._averaged_current_mode_params(component, topology)
        if cm is None:
            logger.warning(
                f"{topology} {ref}: MODE=avg CMODE=peak needs Sim.Params VREF (the "
                f"FBX reference, may be negative), FSW, and a resolvable duty "
                f'(D, or VOUT+VIN), e.g. "fsw=300k vout=24 vin=12 vref=1.6 '
                f'mode=avg cmode=peak" - skipping'
            )
            return

        fb, vc, vout, gnd = (
            str(term["fb"]), str(term["vc"]), str(term["vout"]), str(term["gnd"])
        )
        # Give any other pins (SW/VIN/SS/RT/INTVCC...) a DC path; the loop model
        # uses only FB/VC/VOUT/GND. VIN (if present) is a driven rail; a 1G stub on
        # it is harmless.
        self._stub_unmodeled_pins(component, ref, {vc, vout, gnd, fb}, gnd)

        n = self._fmt_num
        vref, gm, rea, ri, mc, d = (
            n(cm["VREF"]), n(cm["GM"]), n(cm["REA"]),
            cm["RI"], cm["MC"], cm["D"],
        )
        dprime = 1.0 - d
        # Modulator gain: control voltage commands the inductor current (iL=V(vc)/Ri);
        # the delivered average output current is iL for a buck, iL*(1-D) for the
        # step-up families.
        gmc = (1.0 / ri) if topology == "buck" else (dprime / ri)

        # Subharmonic double pole at fsw/2 realized as a unity-DC 2-pole LP (R-L-C):
        # wn=2*pi*(fsw/2), Q from the slope factor. Q blows up as mc*D' -> 0.5, so
        # clamp it (and warn) rather than emit a negative/None damping.
        wn = math.pi * cm["FSW"]  # = 2*pi*(fsw/2)
        q_den = math.pi * (mc * dprime - 0.5)
        if q_den <= 0.05:
            qp = 20.0
            logger.warning(
                f"{topology} {ref}: slope factor mc={mc:g} at D={d:.3f} gives a "
                f"subharmonic Q near/over instability (mc*(1-D)={mc * dprime:.3f} "
                f"<= 0.5); clamped to Q=20. Increase MC (more slope comp)."
            )
        else:
            qp = 1.0 / q_den
        ch = 1e-6
        lh = 1.0 / (wn * wn * ch)
        rh = 1.0 / (qp * wn * ch)

        shin, nrh, sh = f"{ref}_shin", f"{ref}_nrh", f"{ref}_sh"
        lines = [
            f"B{ref}_ea 0 {vc} I = {gm}*({vref} - V({fb}))",
            f"R{ref}_ea {vc} {gnd} {rea}",
            f"B{ref}_shin {shin} {gnd} V = V({vc})",
            f"R{ref}_sh {shin} {nrh} {n(rh)}",
            f"L{ref}_sh {nrh} {sh} {n(lh)}",
            f"C{ref}_sh {sh} {gnd} {n(ch)}",
        ]

        # RHP zero (boost/sepic/cuk). Realize (1 - s/wz) exactly via a sensed
        # capacitor current (no extra pole): I(Vz)=Cz*dV(sh)/dt, so
        # V(sh) - (1/(wz*Cz))*I(Vz) = V(sh)*(1 - s/wz).
        frhpz = cm["FRHPZ"]
        src = sh  # node feeding the output transconductance
        if topology != "buck":
            if frhpz and frhpz > 0:
                wz = 2.0 * math.pi * frhpz
                cz = 1e-9
                kz = 1.0 / (wz * cz)
                nz, rz = f"{ref}_nz", f"{ref}_rz"
                lines += [
                    f"C{ref}_z {sh} {nz} {n(cz)}",
                    f"V{ref}_z {nz} {gnd} DC 0",
                    f"B{ref}_rz {rz} {gnd} V = V({sh}) - {n(kz)}*I(V{ref}_z)",
                ]
                src = rz
            else:
                logger.warning(
                    f"{topology} {ref}: no RHP-zero frequency resolved (give FRHPZ, "
                    f"or RLOAD and L) - the boost/SEPIC/Ćuk right-half-plane zero is "
                    f"omitted; the loop model is optimistic above ~fsw/10"
                )

        lines.append(f"B{ref}_out 0 {vout} I = {n(gmc)}*V({src})")

        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        self.model_provenance[ref] = ResolvedModel(
            ref, topology, "sim_params",
            f"{topology}_averaged_cm(vref={vref}, d={d:.3f})",
        )
        fsw = cm["FSW"]
        rhpz_txt = (
            f"RHP zero {self._fmt_hz(frhpz)}" if (topology != "buck" and frhpz)
            else "no RHP zero" if topology == "buck" else "RHP zero omitted"
        )
        logger.debug(
            f"{ref}: {topology} averaged peak-current-mode loop model (vref={vref}, "
            f"d={d:.3f}, gm={gm}, ri={ri:g}, gmc={gmc:g}, mc={mc:g}, Q={qp:.2f} at "
            f"fsw/2={self._fmt_hz(fsw / 2)}, {rhpz_txt}); small-signal compensation "
            f"model for .ac loop gain -- NOT the closed-loop switching controller. "
            f"Results above ~{self._fmt_hz(fsw / 2)} (fsw/2) are not physical."
        )

    def _add_cmcontroller(self, component, ref: str, value: str):
        """Behavioral CLOSED-LOOP peak-current-mode controller (Sim.Device="CMCONTROLLER").

        Stage 29.1 -- the cycle-accurate, large-signal, closed-loop regime neither
        the open-loop switch macromodels (duty is swept) nor the averaged 28.D loop
        model (small-signal ``.ac``) can reach. It **generates the real gate from
        feedback** and regulates a live rail in ``.tran``. Built entirely from
        ngspice ``B``-sources + the switch-held-memory latch pattern of
        ``_emit_ms_dff`` (no XSPICE, no corpus model). The controller emits its OWN
        buck switch stage between VIN and SW, gated by the latch; the user supplies
        the real inductor (SW->VOUT), Cout, feedback divider (VOUT->FB->GND) and the
        VC/ITH compensation network -- exactly the parts the averaged 28.D model uses,
        so the two regimes cross-check.

        Emitted per ref (``per`` = 1/FSW; nodes prefixed ``<ref>_``)::

            oscillator   VCLK narrow SET pulse @ FSW; VRAMP 0->1 slope sawtooth;
                         VDUTY max-duty force-off pulse past DMAX
            error amp    B_ea: I into VC = clamp(GM*(VREF - V(FB)), -ISINK, ISOURCE)
                         (28.D's gm cell, now source/sink-limited; Stage 29.2)
                         R_ea VC->GND REA (finite-DC-gain / DC-path safety)
                         soft VHIGH/VLOW diode clamps bound the VC output swing
            sense+slope  V_isns (0 V) in the switch branch -> I(V_isns) = iL(on);
                         B_isig = RI*I(V_isns) + MCSLOPE*V(ramp)
            reset        B_rst = (isig > VC) OR (past DMAX) [OR raw sensed I >
                         VSENSE_MAX, the cycle-by-cycle current limit], RC-slowed
                         (Basso conv.); optionally blanked for TON_MIN after each set
            SR latch     switch-held memory cap; set=clk, RESET-DOMINANT (set is
                         gated off while resetting) so the comparator/current-limit
                         always wins; B_gate = latched Q [AND UVLO run]
            switch stage S_hs (VIN->SW, latch-gated) + freewheel diode GND->SW

        Supervisory features (Stage 29.3, each behind a param, absent = off + emission
        byte-identical to 29.1/29.2): TSS soft-start (a monotone reference ramp bounding
        inrush), VSENSE_MAX cycle-by-cycle peak current limit, TON_MIN leading-edge
        blanking / min-on-time, FB_FOLD/FOLD_RATIO two-state frequency foldback (a slow
        folded clock selected while V(FB) is far below target), and UVLO_RISE/UVLO_FALL
        under-voltage lockout with hysteresis. Load-transient recovery needs no extra
        modeling -- it is the closed loop's own response to a load step.

        Negative feedback (buck): VOUT up -> V(FB) up -> (VREF-FB) down -> less
        current into VC -> VC down -> the peak-current command drops -> the switch
        turns off earlier -> VOUT down. The peak comparator + max-duty bound the
        per-cycle current, so inrush is limited even without soft-start; a nonzero
        ``TSS`` ramps VREF from 0 to ease startup convergence (the full soft-start is
        Stage 29.3). All memory caps carry ``IC=0``: the closed-loop ``.tran`` must
        use ``use_initial_condition=True`` (seed ``{VOUT:0}``), exactly as the DFF
        divider does.

        HONEST BOUNDARY (logged + recorded in provenance): behavioral emulation of a
        current-mode controller's headline datasheet specs, NOT the encrypted silicon.
        CCM; no thermal / gate-charge / protection corner cases beyond the
        parameterized ones. GM/RI/MCSLOPE are datasheet-anchored design inputs.
        Emits nothing and warns (honest skip) when the terminals or required params
        (VREF/FSW) do not resolve, or the topology is not yet wired (29.1: buck only).
        """
        term = self._cmcontroller_terminals(component)
        if term is None:
            logger.warning(
                f"CMCONTROLLER {ref}: needs connected VIN, SW, VOUT, FB, VC "
                f"(compensation) and GND pins (resolved by pin name) - skipping"
            )
            return
        cp = self._cmcontroller_params(component)
        if cp is None:
            logger.warning(
                f"CMCONTROLLER {ref}: needs Sim.Params VREF (the FB reference, may be "
                f"negative) and FSW, e.g. "
                f'"topology=buck fsw=500k vout=3.3 vin=12 vref=0.8 ri=0.1" - skipping'
            )
            return
        topology = cp["TOPOLOGY"]
        if topology != "buck":
            logger.warning(
                f"CMCONTROLLER {ref}: topology={topology} is not wired in Stage 29.1 "
                f"(buck only); the boost/SEPIC/Ćuk/flyback stages are Stage 29.4 - "
                f"skipping"
            )
            return

        vin, sw, vout, fb, vc, gnd = (
            str(term["vin"]), str(term["sw"]), str(term["vout"]),
            str(term["fb"]), str(term["vc"]), str(term["gnd"]),
        )
        # Any other pins (SS/RT/EN/SYNC/INTVCC...) get a 1G DC path; 29.1 models only
        # the six resolved terminals (VIN is a driven rail, a stub on it is harmless).
        self._stub_unmodeled_pins(component, ref, {vin, sw, vout, fb, vc, gnd}, gnd)

        n = self._fmt_num
        vref, gm, rea, ri, mcslope, ron, dmax, tss = (
            cp["VREF"], n(cp["GM"]), n(cp["REA"]), n(cp["RI"]),
            n(cp["MCSLOPE"]), n(cp["RON"]), cp["DMAX"], cp["TSS"],
        )
        vhigh, vlow, isource, isink = (
            n(cp["VHIGH"]), n(cp["VLOW"]), n(cp["ISOURCE"]), n(-cp["ISINK"]),
        )
        # Supervisory features (Stage 29.3); None = off (byte-identical to 29.1/29.2).
        vsense_max, ton_min, fb_fold = (
            cp["VSENSE_MAX"], cp["TON_MIN"], cp["FB_FOLD"],
        )
        fold_ratio, uvlo_rise, uvlo_fall = (
            cp["FOLD_RATIO"], cp["UVLO_RISE"], cp["UVLO_FALL"],
        )
        per = 1.0 / cp["FSW"]
        tr = per * 1e-3            # narrow switch edge
        thi = per * 0.02           # narrow SET pulse
        rd = 1000.0                # reset RC-slow: tau = rd*cd = per/1000
        cd = per / (rd * 1000.0)
        cq = per / 5000.0          # latch memory: tau = 50*cq ~ per/100
        dmax_on = max(per * dmax, tr)
        dmax_off = max(per * (1.0 - dmax), tr)

        clk, ramp, rstd = f"{ref}_clk", f"{ref}_ramp", f"{ref}_rstd"
        isig, rstr, rstf = f"{ref}_isig", f"{ref}_rstr", f"{ref}_rstf"
        hi, sg, qm, gate, swhi = (
            f"{ref}_hi", f"{ref}_sg", f"{ref}_qm", f"{ref}_gate", f"{ref}_swhi",
        )

        # Soft reference: a nonzero TSS ramps V(vref) from 0 -> VREF over TSS so the
        # loop starts gently (helps the UIC .tran converge); TSS=0 uses a hard literal.
        pre = []
        if tss and tss > 0:
            vrefn = f"{ref}_vref"
            pre.append(
                f"B{ref}_vref {vrefn} {gnd} V = {n(vref)}*(time > {n(tss)} ? "
                f"1 : time/{n(tss)})"
            )
            vref_expr = f"V({vrefn})"
        else:
            vref_expr = n(vref)

        # Frequency foldback (Stage 29.3): a slower folded clock at FSW*FOLD_RATIO is
        # SELECTED to set the latch while V(FB) is far below target (two-state, not a
        # continuous VCO -- robust). Absent FB_FOLD, the set clock is the FSW clock.
        fold = []
        clk_set = clk               # node the SR latch's set gate samples
        if fb_fold is not None:
            clkf, clksel = f"{ref}_clkf", f"{ref}_clksel"
            perf = per / fold_ratio if fold_ratio else per
            fold = [
                f"V{ref}_clkf {clkf} {gnd} PULSE(0 5 0 {tr:g} {tr:g} {thi:g} "
                f"{perf:g})",
                f"B{ref}_clksel {clksel} {gnd} V = V({fb}) < {n(fb_fold)} ? "
                f"V({clkf}) : V({clk})",
            ]
            clk_set = clksel

        # Reset expression (Stage 29.3): the 29.1 core (current comparator OR max-duty),
        # optionally OR'd with the cycle-by-cycle current limit (RAW sensed current past
        # VSENSE_MAX, slope comp excluded), optionally blanked for TON_MIN after the set
        # (V(ramp) < TON_MIN*FSW == the leading TON_MIN of the cycle). Built so that
        # both-off reproduces the exact 29.1 line byte-for-byte.
        reset_core = (
            f"V({isig}) > V({vc}) ? 5 : (V({rstd}) > 2.5 ? 5 : 0)"
        )
        if vsense_max is not None:
            reset_core = (
                f"V({isig}) > V({vc}) ? 5 : (V({rstd}) > 2.5 ? 5 : "
                f"({ri}*I(V{ref}_isns) > {n(vsense_max)} ? 5 : 0))"
            )
        reset_expr = reset_core
        if ton_min is not None:
            reset_expr = f"V({ramp}) < {n(ton_min / per)} ? 0 : ({reset_core})"

        # UVLO / EN (Stage 29.3): a hysteretic run latch (switch-held memory, the SR
        # pattern) gates the gate output off until V(VIN) rises past UVLO_RISE, back off
        # below UVLO_FALL. Absent UVLO_RISE, the gate is the bare latched Q (29.1).
        uvlo = []
        gate_expr = f"V({qm}) > 2.5 ? 5 : 0"
        if uvlo_rise is not None:
            runm, run = f"{ref}_runm", f"{ref}_run"
            uvset, uvrst = f"{ref}_uvset", f"{ref}_uvrst"
            uvlo = [
                f"B{ref}_uvset {uvset} {gnd} V = V({vin}) > {n(uvlo_rise)} ? 5 : 0",
                f"B{ref}_uvrst {uvrst} {gnd} V = V({vin}) < {n(uvlo_fall)} ? 5 : 0",
                f"S{ref}_uvs {hi} {runm} {uvset} {gnd} SWL{ref}",
                f"S{ref}_uvr {runm} {gnd} {uvrst} {gnd} SWL{ref}",
                f"C{ref}_uvr {runm} {gnd} {n(cq)} IC=0",
                f"B{ref}_run {run} {gnd} V = V({runm}) > 2.5 ? 5 : 0",
            ]
            gate_expr = f"V({qm}) > 2.5 ? (V({run}) > 2.5 ? 5 : 0) : 0"

        lines = pre + fold + [
            # ---- oscillator: SET clock + slope-comp ramp + max-duty force-off ----
            f"V{ref}_clk {clk} {gnd} PULSE(0 5 0 {tr:g} {tr:g} {thi:g} {per:g})",
            f"V{ref}_ramp {ramp} {gnd} PULSE(0 1 0 {per * 0.98:g} {per * 0.01:g} "
            f"0 {per:g})",
            f"V{ref}_duty {rstd} {gnd} PULSE(0 5 {per * dmax:g} {tr:g} {tr:g} "
            f"{dmax_off:g} {per:g})",
            # ---- error amp: gm cell into the real external VC net (28.D cell),
            #      source/sink-limited (min/max on the CURRENT -- slew-limits VC
            #      without floating the node) with soft VHIGH/VLOW output-swing
            #      clamps (Basso Fig. 4; Stage 29.2). REA keeps the DC path. ----
            f"B{ref}_ea {gnd} {vc} I = min(max({gm}*({vref_expr} - V({fb})), "
            f"{isink}), {isource})",
            f"R{ref}_ea {vc} {gnd} {rea}",
            f"V{ref}_vh {ref}_vh {gnd} {vhigh}",
            f"D{ref}_hi {vc} {ref}_vh DCL{ref}",
            f"V{ref}_vl {ref}_vl {gnd} {vlow}",
            f"D{ref}_lo {ref}_vl {vc} DCL{ref}",
            f".model DCL{ref} D(RS=10 N=0.01)",
            # ---- current sense (0 V in the switch branch) + slope-comp signal ----
            f"V{ref}_isns {swhi} {sw} 0",
            f"B{ref}_isig {isig} {gnd} V = {ri}*I(V{ref}_isns) + "
            f"{mcslope}*V({ramp})",
            # ---- reset = current comparator OR max-duty [OR current limit],
            #      optionally leading-edge-blanked for TON_MIN, RC-slowed ----
            f"B{ref}_rst {rstr} {gnd} V = {reset_expr}",
            f"R{ref}_rd {rstr} {rstf} {n(rd)}",
            f"C{ref}_rd {rstf} {gnd} {n(cd)}",
            # ---- SR latch (set=clk, RESET-DOMINANT), switch-held memory cap ----
            f"V{ref}_hi {hi} {gnd} 5",
            f"B{ref}_sg {sg} {gnd} V = V({clk_set}) > 2.5 ? "
            f"(V({rstf}) > 2.5 ? 0 : 5) : 0",
            f"S{ref}_set {hi} {qm} {sg} {gnd} SWL{ref}",
            f"S{ref}_rst {qm} {gnd} {rstf} {gnd} SWL{ref}",
            f"C{ref}_qm {qm} {gnd} {n(cq)} IC=0",
            f".model SWL{ref} SW(Ron=50 Roff=1e9 Vt=2.5 Vh=0.2)",
        ] + uvlo + [
            # ---- gate = latched Q [AND the UVLO run signal] ----
            f"B{ref}_gate {gate} {gnd} V = {gate_expr}",
            # ---- buck switch stage gated by the latch (HS switch + freewheel) ----
            f"S{ref}_hs {vin} {swhi} {gate} {gnd} SWM{ref}",
            f".model SWM{ref} SW(Ron={ron} Roff=1e6 Vt=2.5 Vh=0.2)",
            f"D{ref}_fw {gnd} {sw} DFW{ref}",
            f".model DFW{ref} D(IS=1e-9 N=1.05 CJO=100p)",
        ]
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        # Note any active NEW supervisory features (Stage 29.3) in the provenance so
        # the netlist records which nonlinear protections the model is emulating. TSS
        # (pre-existing since 29.1) is deliberately NOT added here, keeping the
        # provenance string byte-identical for circuits that only use soft-start.
        sup = []
        if vsense_max is not None:
            sup.append(f"ilim={n(vsense_max)}")
        if ton_min is not None:
            sup.append(f"tonmin={self._fmt_num(ton_min)}s")
        if fb_fold is not None:
            sup.append(f"foldback<{n(fb_fold)}@{n(fold_ratio)}x")
        if uvlo_rise is not None:
            sup.append(f"uvlo={n(uvlo_rise)}/{n(uvlo_fall)}")
        sup_str = (", " + ", ".join(sup)) if sup else ""
        self.model_provenance[ref] = ResolvedModel(
            ref, "cmcontroller", "sim_params",
            f"{topology}_cmcontroller(vref={n(vref)}, "
            f"fsw={self._fmt_hz(cp['FSW'])}{sup_str})",
        )
        logger.debug(
            f"{ref}: {topology} behavioral CLOSED-LOOP peak-current-mode controller "
            f"(vref={n(vref)}, fsw={self._fmt_hz(cp['FSW'])}, ri={ri}, mcslope="
            f"{mcslope}, dmax={dmax:g}"
            f"{f', tss={self._fmt_num(tss)}s' if tss else ''}{sup_str}); "
            f"cycle-accurate .tran needs use_initial_condition=True + "
            f"initial_conditions={{{vout}:0}}. "
            f"Behavioral emulation of datasheet specs -- NOT the encrypted silicon; "
            f"CCM; no thermal / gate-charge / protection corners beyond the "
            f"parameterized ones."
        )

    # ------------------------------------------------------------------ #
    # Half-bridge / resonant switch stage (Stage 26 Phase B)              #
    # ------------------------------------------------------------------ #

    def _halfbridge_params(self, component) -> Optional[dict]:
        """Params for a half-bridge switch stage, or None if unusable.

        ``FSW`` (switching frequency) is required -- it is the control variable
        for an open-loop resonant converter (the gain is swept by FSW, not set by
        a duty computation). ``DT`` (deadtime, default 100 ns) and ``RON`` (switch
        on-resistance, default 0.1 Ohm) take defaults. Returns None when FSW is
        missing/invalid or the deadtime is >= half the period (no conduction time
        left) -- validate() reports which.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        fsw = self._parse_si_number(raw["FSW"]) if "FSW" in raw else None
        if not fsw or fsw <= 0:
            return None

        def g(key, default):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        dt = g("DT", 100e-9)
        ron = g("RON", 0.1)
        if dt < 0 or dt >= 0.5 / fsw:
            return None
        return {"FSW": fsw, "DT": dt, "RON": ron}

    def _emit_sync_leg(
        self,
        top: str,
        sw: str,
        bottom: str,
        gnd: str,
        *,
        fsw: float,
        duty: float,
        phase: float = 0.0,
        dt: float,
        ron: float,
        suffix: str,
    ) -> bool:
        """Emit one duty+phase-parameterized synchronous switch leg.

        A complementary voltage-controlled ``S`` switch pair between ``top`` and
        ``bottom`` with the switch node ``sw`` in the middle, each device
        carrying a **mandatory antiparallel body diode** so it conducts in
        reverse -- that reverse path is what makes bidirectionality essentially
        free at the switch level (the Stage-27 insight). Generalizes
        ``_add_halfbridge``'s fixed-50 %/zero-phase machinery:

        - the high-side device (``top`` -> ``sw``) conducts for ``duty/fsw - dt``;
        - the low-side device (``sw`` -> ``bottom``) conducts for
          ``(1-duty)/fsw - dt``, delayed by ``duty/fsw`` (so it fills the rest of
          the period, minus a deadtime at each edge);
        - the whole leg is delayed by ``phase/fsw`` (the inter-leg phase used by
          multi-leg topologies).

        Gate PULSEs are referenced to ``gnd`` (the control-voltage reference), so
        a high-side device floating on ``top`` still switches correctly. All card
        names are suffixed by ``suffix`` (e.g. ``f"{ref}A"``) so several legs may
        coexist on one converter.

        Returns True on success; False -- emitting **nothing** -- when the
        deadtime leaves no conduction window (``dt >= min(duty, 1-duty)/fsw``) so
        the caller can log a warning and skip, mirroring ``_halfbridge_params``.

        At ``duty=0.5, phase=0.0, top=vin, bottom=gnd, gnd=gnd, suffix=ref`` the
        emitted lines are **byte-for-byte identical** to ``_add_halfbridge`` -- a
        hard gate (the HALFBRIDGE output must not change; see
        ``test_sim_halfbridge.py`` and ``test_sim_syncleg.py``).
        """
        per = 1.0 / fsw
        on_hs = duty * per - dt  # high-side conduction time
        on_ls = (1.0 - duty) * per - dt  # low-side conduction time
        if on_hs <= 0 or on_ls <= 0:
            return False
        td_hs = phase * per  # high-side turn-on delay
        td_ls = phase * per + duty * per  # low-side turn-on delay
        edge = per / 200.0
        ron = self._fmt_num(ron)
        ghs, gls = f"{suffix}_ghs", f"{suffix}_gls"

        lines = [
            # Complementary gate drives with a deadtime gap: high-side on first
            # (for duty*per - DT), low-side after (for (1-duty)*per - DT); both
            # off during the two DT windows. The whole leg is shifted by phase*per.
            f"V{suffix}_ghs {ghs} {gnd} PULSE(0 5 {td_hs:.6g} "
            f"{edge:.6g} {edge:.6g} {on_hs:.6g} {per:.6g})",
            f"V{suffix}_gls {gls} {gnd} PULSE(0 5 {td_ls:.6g} "
            f"{edge:.6g} {edge:.6g} {on_ls:.6g} {per:.6g})",
            f"S{suffix}_hs {top} {sw} {ghs} {gnd} SW{suffix}",
            f"S{suffix}_ls {sw} {bottom} {gls} {gnd} SW{suffix}",
            f".model SW{suffix} SW(Ron={ron} Roff=1e6 Vt=2.5 Vh=0.2)",
            # Antiparallel body diodes (high-side sw->top, low-side bottom->sw):
            # the deadtime freewheel path and the source of reverse conduction.
            f"D{suffix}_hs {sw} {top} DFW{suffix}",
            f"D{suffix}_ls {bottom} {sw} DFW{suffix}",
            f".model DFW{suffix} D(IS=1e-9 N=1.05 CJO=100p)",
        ]
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        return True

    def _add_halfbridge(self, component, ref: str, value: str):
        """Emit an open-loop half-bridge switch stage (Sim.Device=HALFBRIDGE/LLC).

        Replaces ONLY the two switches of a half/resonant bridge -- the user's
        Cr/Lr/transformer/rectifier stay real parts. A complementary S-switch pair
        runs at a fixed 50 % duty (minus deadtime) driven from FSW; the gain curve
        of a resonant tank is obtained by sweeping FSW across `.tran` runs (there
        is no VOUT/duty computation, which is what makes the model robust).

        Each switch carries a **mandatory antiparallel diode**: during the
        deadtime the resonant-tank current must have a path or the switch node
        rings to kV and the transient fails to converge -- and the diode is what
        lets a ZVS-shaped V(sw) emerge naturally (the device-level twin in Phase C
        checks this). Built from the buck's proven S-switch + PULSE machinery
        (real current paths through the off state), not a stiff behavioral source.
        """
        term = self._switcher_terminals(component)
        if term is None:
            logger.warning(
                f"halfbridge {ref}: could not resolve SW/VIN/GND terminals - skipping"
            )
            return
        params = self._halfbridge_params(component)
        if params is None:
            logger.warning(
                f"halfbridge {ref}: no usable FSW/DT resolved - skipping "
                f'(set Sim.Params="fsw=100k")'
            )
            return

        # Give cap-only unmodeled pins (BOOT/EN/VDD...) a DC path so the op-point
        # solves, exactly like the switching-regulator handler.
        self._stub_unmodeled_pins(
            component,
            ref,
            {term["sw"], term["vin"], term["gnd"], term["fb"]},
            term["gnd"],
        )

        sw, vin, gnd = str(term["sw"]), str(term["vin"]), str(term["gnd"])
        n = self._fmt_num
        fsw, dt, ron = params["FSW"], params["DT"], n(params["RON"])
        # A half-bridge is exactly a synchronous leg at 50 % duty / zero phase
        # between VIN and GND (gates referenced to GND). Emitting through the
        # shared primitive keeps the two in lockstep; params guaranteed dt < half
        # the period, so the leg always emits (byte-identical to the legacy
        # output -- a hard gate; see test_sim_halfbridge.py).
        self._emit_sync_leg(
            vin,
            sw,
            gnd,
            gnd,
            fsw=fsw,
            duty=0.5,
            phase=0.0,
            dt=dt,
            ron=params["RON"],
            suffix=ref,
        )
        self.model_provenance[ref] = ResolvedModel(
            ref, "halfbridge", "sim_params",
            f"halfbridge_openloop(fsw={self._fmt_hz(fsw)}, dt={n(dt)})",
        )
        logger.debug(
            f"{ref}: half-bridge switch stage (open-loop, fsw={self._fmt_hz(fsw)}, "
            f"dt={n(dt)}, ron={ron}); FSW-swept for the resonant gain curve; "
            f"antiparallel diodes give the tank a deadtime freewheel path (ZVS)"
        )

    # ------------------------------------------------------------------ #
    # Multi-switch synchronous bidirectional converters (Stage 27 Route B) #
    # ------------------------------------------------------------------ #

    def _emit_static_or_sync_leg(
        self, top, sw, bottom, gnd, *, fsw, duty, dt, ron, suffix
    ) -> bool:
        """Emit a switching sync leg, or a static pass-through when saturated.

        ``duty`` is the high-side on-fraction. A leg that never switches (a pure
        buck/boost operating point saturates the *other* leg) has no conduction
        window for ``_emit_sync_leg`` -- the same problem the device-level twins
        solved with a static gate tie (Stage 27.2's "static-leg gating pattern").
        Here the static case is a small ``RON`` tie of the switch node to the rail
        the permanently-on device connects it to:

        - ``duty >= 1`` -> high-side always on -> ``sw`` tied to ``top``;
        - ``duty <= 0`` -> low-side always on -> ``sw`` tied to ``bottom``.

        Otherwise delegates to ``_emit_sync_leg`` (complementary switches + body
        diodes). Returns True if anything was emitted; False only when a genuine
        switching leg was requested but the deadtime leaves no window -- callers
        pre-check with ``_leg_has_window`` so this never partially emits.
        """
        if duty >= 1.0:
            self.spice_circuit.raw_spice += (
                f"\nR{suffix}_sat {sw} {top} {self._fmt_num(ron)}"
            )
            return True
        if duty <= 0.0:
            self.spice_circuit.raw_spice += (
                f"\nR{suffix}_sat {sw} {bottom} {self._fmt_num(ron)}"
            )
            return True
        return self._emit_sync_leg(
            top, sw, bottom, gnd,
            fsw=fsw, duty=duty, phase=0.0, dt=dt, ron=ron, suffix=suffix,
        )

    @staticmethod
    def _leg_has_window(duty, fsw, dt) -> bool:
        """True if a leg at ``duty`` is emittable: saturated (static) or the
        deadtime leaves a conduction window (``dt < min(duty,1-duty)/fsw``)."""
        if duty >= 1.0 or duty <= 0.0:
            return True
        return dt < min(duty, 1.0 - duty) / fsw

    def _buckboost4_params(self, component) -> Optional[dict]:
        """Params for a 4-switch buck-boost macromodel, or None if unusable.

        ``FSW`` is required. The control variables are the two leg duties
        (open-loop, matching every other switch macromodel's honesty limit):

        - ``DBUCK`` -- buck-leg high-side on-fraction (1 = pass-through);
        - ``DBOOST`` -- boost-leg low-side on-fraction (0 = pass-through).

        An unspecified leg saturates to pass-through (``DBUCK`` default 1,
        ``DBOOST`` default 0), so ``dbuck=0.5`` alone is a plain synchronous buck.
        Convenience: if neither duty is given but a target ``VOUT`` **and** the
        input ``VIN`` are, one leg's duty is derived (``Vout<Vin`` -> buck mode
        ``dbuck=Vout/Vin, dboost=0``; else boost mode ``dbuck=1,
        dboost=1-Vin/Vout``) with the documented first-order caveat. ``DT``
        (deadtime, default 100 ns) and ``RON`` (default 0.1 Ohm) as HALFBRIDGE.

        Returns None when FSW is missing/invalid, the deadtime is >= half the
        period, or no duties can be resolved.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        fsw = self._parse_si_number(raw["FSW"]) if "FSW" in raw else None
        if not fsw or fsw <= 0:
            return None

        def g(key, default=None):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        dt = g("DT", 100e-9)
        ron = g("RON", 0.1)
        if dt < 0 or dt >= 0.5 / fsw:
            return None

        dbuck = g("DBUCK")
        dboost = g("DBOOST")
        if dbuck is None and dboost is None:
            vout, vin = g("VOUT"), g("VIN")
            if vout and vout > 0 and vin and vin > 0:
                if vout < vin:
                    dbuck, dboost = vout / vin, 0.0
                else:
                    dbuck, dboost = 1.0, 1.0 - vin / vout
            else:
                return None
        else:
            if dbuck is None:
                dbuck = 1.0  # buck leg pass-through (boost-only operation)
            if dboost is None:
                dboost = 0.0  # boost leg pass-through (buck-only operation)

        dbuck = min(max(dbuck, 0.0), 1.0)
        dboost = min(max(dboost, 0.0), 1.0)
        return {"FSW": fsw, "DT": dt, "RON": ron, "DBUCK": dbuck, "DBOOST": dboost}

    def _add_buckboost4(self, component, ref: str, value: str):
        """Emit a non-inverting 4-switch buck-boost switch stage (Sim.Device=BUCKBOOST4).

        Replaces ONLY the four switches -- the user's shared inductor (between the
        two switch nodes), output cap and divider stay real parts. Two synchronous
        legs share one clock (phase 0): a buck leg (VIN/SWA/GND) at ``DBUCK`` and a
        boost leg (VOUT/SWB/GND) at ``1-DBOOST``. Open-loop: the duties are the
        control variables, so the ideal DC gain is ``Vout/Vin = DBUCK/(1-DBOOST)``
        (swept, not regulated). Bidirectional at the switch level for free -- the
        antiparallel body diodes in each ``_emit_sync_leg`` conduct either way, so
        power direction is set by which port the user attaches source vs. load
        (Stage 27.2's device-level twin proves the reverse-boost case).
        """
        term = self._multiswitch_terminals(component, "buckboost4")
        if term is None:
            logger.warning(
                f"buckboost4 {ref}: could not resolve VIN/SWA/SWB/VOUT/GND "
                f"terminals (by pin name) - skipping"
            )
            return
        params = self._buckboost4_params(component)
        if params is None:
            logger.warning(
                f"buckboost4 {ref}: no usable FSW / duties resolved - skipping "
                f'(set Sim.Params="fsw=500k dbuck=0.5 dboost=0")'
            )
            return

        # Cap-only unmodeled pins (BOOT/EN/VDD...) get a DC path, as HALFBRIDGE does.
        self._stub_unmodeled_pins(component, ref, set(term.values()), term["gnd"])

        vin, swa, swb, vout, gnd = (
            str(term[k]) for k in ("vin", "swa", "swb", "vout", "gnd")
        )
        fsw, dt, ron = params["FSW"], params["DT"], params["RON"]
        dbuck, dboost = params["DBUCK"], params["DBOOST"]
        boost_hs = 1.0 - dboost  # boost-leg high-side on-fraction

        # Pre-check both legs so a too-large deadtime never leaves half a stage.
        if not (
            self._leg_has_window(dbuck, fsw, dt)
            and self._leg_has_window(boost_hs, fsw, dt)
        ):
            logger.warning(
                f"buckboost4 {ref}: deadtime DT ({self._fmt_num(dt)}s) leaves no "
                f"conduction window for a switching leg at these duties - skipping"
            )
            return

        self._emit_static_or_sync_leg(
            vin, swa, gnd, gnd,
            fsw=fsw, duty=dbuck, dt=dt, ron=ron, suffix=f"{ref}A",
        )
        self._emit_static_or_sync_leg(
            vout, swb, gnd, gnd,
            fsw=fsw, duty=boost_hs, dt=dt, ron=ron, suffix=f"{ref}B",
        )

        n = self._fmt_num
        self.model_provenance[ref] = ResolvedModel(
            ref, "buckboost4", "sim_params",
            f"buckboost4_openloop(dbuck={n(dbuck)}, dboost={n(dboost)}, "
            f"fsw={self._fmt_hz(fsw)})",
        )
        logger.debug(
            f"{ref}: 4-switch buck-boost switch stage (open-loop, dbuck={n(dbuck)}, "
            f"dboost={n(dboost)}, fsw={self._fmt_hz(fsw)}, dt={n(dt)}); ideal DC gain "
            f"Vout/Vin = dbuck/(1-dboost) = {n(dbuck / (1.0 - dboost)) if dboost < 1.0 else 'inf'}; "
            f"antiparallel body diodes make it bidirectional (source/load placement "
            f"sets direction)"
        )

    def _invbuckboost_params(self, component) -> Optional[dict]:
        """Params for a 2-switch inverting buck-boost macromodel, or None if unusable.

        ``FSW`` is required. ``D`` (the single high-side on-fraction) is the
        open-loop control variable and sets the ideal (negative) DC gain
        ``Vout = -Vin * D/(1-D)`` (swept, not regulated -- matching every other
        switch macromodel's honesty limit). Convenience: if ``D`` is absent but a
        target ``VOUT`` (the negative output, either sign accepted) **and** the
        input ``VIN`` are given, ``D = |Vout| / (|Vout| + Vin)`` with the
        documented first-order caveat (ignores conduction/deadtime loss). ``DT``
        (deadtime, default 100 ns) and ``RON`` (default 0.1 Ohm) as HALFBRIDGE.

        Returns None when FSW is missing/invalid, the deadtime is >= half the
        period, or no duty can be resolved.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        fsw = self._parse_si_number(raw["FSW"]) if "FSW" in raw else None
        if not fsw or fsw <= 0:
            return None

        def g(key, default=None):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        dt = g("DT", 100e-9)
        ron = g("RON", 0.1)
        if dt < 0 or dt >= 0.5 / fsw:
            return None

        d = g("D")
        if d is None:
            vout, vin = g("VOUT"), g("VIN")
            avout = abs(vout) if vout is not None else 0.0
            if avout > 0 and vin and vin > 0:
                d = avout / (avout + vin)  # |Vout|/Vin = D/(1-D)
            else:
                return None

        d = min(max(d, 0.0), 1.0)
        return {"FSW": fsw, "DT": dt, "RON": ron, "D": d}

    def _add_invbuckboost(self, component, ref: str, value: str):
        """Emit a 2-switch inverting buck-boost switch stage (Sim.Device=INVBUCKBOOST).

        Replaces ONLY the two switches -- the user's inductor (SW->GND), output
        cap and load stay real parts. One synchronous leg spans VIN and the
        **negative** output VOUT with the switch node SW in the middle: the
        high-side switch ties VIN->SW for the on-fraction ``D`` (charging the
        inductor from VIN), the low-side switch ties SW->VOUT for the rest (the
        synchronous rectifier delivering the inverted output). Open-loop -- ``D``
        is the control variable, so the ideal DC gain is the negative
        ``Vout = -Vin * D/(1-D)``.

        The low-side device's antiparallel body diode is VOUT->SW (anode at the
        negative rail, cathode at the switch node) -- the classic inverting
        buck-boost rectifier orientation, which the Stage 27.3 device-level twin
        confirmed is load-bearing (the reversed wiring silently mis-regulated to
        the wrong sign). Bidirectional at the switch level for free: a negative
        drive on the VOUT port makes VIN regulate to a positive rail through the
        same synchronous FETs (Stage 27.3's reverse-boost proof). Emission is
        sign-agnostic -- no positive-only clamp or seed is applied, so the
        negative rail forms naturally (the 27.1/27.3 spikes verified the F4
        floating-node tie needs no special handling: the load-tied output is not
        DC-floating).
        """
        term = self._multiswitch_terminals(component, "invbuckboost")
        if term is None:
            logger.warning(
                f"invbuckboost {ref}: could not resolve VIN/SW/VOUT/GND terminals "
                f"(by pin name) - skipping"
            )
            return
        params = self._invbuckboost_params(component)
        if params is None:
            logger.warning(
                f"invbuckboost {ref}: no usable FSW / duty resolved - skipping "
                f'(set Sim.Params="fsw=500k d=0.5")'
            )
            return

        # Cap-only unmodeled pins (BOOT/EN/VDD...) get a DC path, as HALFBRIDGE does.
        self._stub_unmodeled_pins(component, ref, set(term.values()), term["gnd"])

        vin, sw, vout, gnd = (str(term[k]) for k in ("vin", "sw", "vout", "gnd"))
        fsw, dt, ron, d = params["FSW"], params["DT"], params["RON"], params["D"]

        # A single switching leg -- unlike the 4-switch model there is no *other*
        # leg to saturate this one, so a saturated duty (0 or 1) is not a valid
        # inverting buck-boost operating point (it is a dead short / open, not a
        # pass-through). This leg must genuinely switch, so reject a deadtime that
        # leaves no conduction window rather than fall back to a static tie.
        if d <= 0.0 or d >= 1.0 or dt >= min(d, 1.0 - d) / fsw:
            logger.warning(
                f"invbuckboost {ref}: duty D ({self._fmt_num(d)}) with deadtime DT "
                f"({self._fmt_num(dt)}s) leaves no conduction window - skipping"
            )
            return

        self._emit_sync_leg(
            vin, sw, vout, gnd,
            fsw=fsw, duty=d, phase=0.0, dt=dt, ron=ron, suffix=ref,
        )

        n = self._fmt_num
        self.model_provenance[ref] = ResolvedModel(
            ref, "invbuckboost", "sim_params",
            f"invbuckboost_openloop(d={n(d)}, fsw={self._fmt_hz(fsw)})",
        )
        logger.debug(
            f"{ref}: 2-switch inverting buck-boost switch stage (open-loop, "
            f"d={n(d)}, fsw={self._fmt_hz(fsw)}, dt={n(dt)}); ideal DC gain "
            f"Vout/Vin = -d/(1-d) = {n(-d / (1.0 - d))} (negative rail); "
            f"antiparallel body diodes make it bidirectional (source/load "
            f"placement sets direction)"
        )

    def _emit_sepic_switches(
        self, swa, swb, vout, gnd, *, fsw, duty, dt, ron, suffix
    ) -> bool:
        """Emit the SEPIC's two switches (main + sync rectifier) directly.

        The SEPIC's two switches do NOT share a switch node -- the main switch is
        ``swa->gnd`` and the sync rectifier is ``swb->vout`` -- so they are not a
        top/bottom totem-pole pair and cannot be a single ``_emit_sync_leg`` leg
        (which emits a complementary pair on ONE middle node). Emitting two full
        legs would double the device count, so this follows the same
        PULSE / S-switch / body-diode / ``.model`` machinery as ``_emit_sync_leg``
        but places the two switches on independent nodes with complementary gate
        timing:

        - the main switch (``swa->gnd``) conducts for ``duty/fsw - dt`` at phase 0,
          body diode ``gnd->swa`` (the ground-referenced main FET's freewheel path,
          Stage 27.4's ``DQ1_body 0 A``);
        - the sync rectifier (``swb->vout``) conducts for ``(1-duty)/fsw - dt``,
          delayed ``duty/fsw`` (complementary, one deadtime at each edge), body
          diode ``swb->vout`` -- the load-bearing SEPIC rectifier orientation the
          Stage 27.4 device-level twin locked (``DQ2_body B VOUT``; the reversed
          wiring silently produced -7.8 V in the 27.1 Spike-2).

        Gate PULSEs are referenced to ``gnd``. The user's real coupling cap Cs
        (``swa->swb``) and both inductors survive -- this emits ONLY the switch
        stage. Returns True on success; False -- emitting **nothing** -- when the
        deadtime leaves no conduction window (``dt >= min(duty, 1-duty)/fsw``), so
        the caller can log a warning and skip.
        """
        per = 1.0 / fsw
        on_m = duty * per - dt          # main-switch conduction time
        on_r = (1.0 - duty) * per - dt  # sync-rectifier conduction time
        if on_m <= 0 or on_r <= 0:
            return False
        td_r = duty * per               # rectifier turn-on delay (complementary)
        edge = per / 200.0
        ron = self._fmt_num(ron)
        gm, gr = f"{suffix}_gm", f"{suffix}_gr"
        lines = [
            # Complementary gate drives with a deadtime gap, referenced to GND:
            # main on first (duty*per - DT), rectifier after (delayed duty*per),
            # both off during the two DT windows.
            f"V{suffix}_gm {gm} {gnd} PULSE(0 5 0 "
            f"{edge:.6g} {edge:.6g} {on_m:.6g} {per:.6g})",
            f"V{suffix}_gr {gr} {gnd} PULSE(0 5 {td_r:.6g} "
            f"{edge:.6g} {edge:.6g} {on_r:.6g} {per:.6g})",
            f"S{suffix}_m {swa} {gnd} {gm} {gnd} SW{suffix}",
            f"S{suffix}_r {swb} {vout} {gr} {gnd} SW{suffix}",
            f".model SW{suffix} SW(Ron={ron} Roff=1e6 Vt=2.5 Vh=0.2)",
            # Body diodes: main gnd->swa (freewheel), rectifier swb->vout (the
            # SEPIC rectifier path -- orientation load-bearing; matches the twin).
            f"D{suffix}_m {gnd} {swa} DFW{suffix}",
            f"D{suffix}_r {swb} {vout} DFW{suffix}",
            f".model DFW{suffix} D(IS=1e-9 N=1.05 CJO=100p)",
        ]
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        return True

    def _sepic_params(self, component) -> Optional[dict]:
        """Params for a bidirectional SEPIC macromodel, or None if unusable.

        ``FSW`` is required. ``D`` (the main-switch on-fraction) is the open-loop
        control variable and sets the ideal (non-inverting) DC gain
        ``Vout = Vin * D/(1-D)`` -- swept, not regulated, matching every other
        switch macromodel's honesty limit. Convenience: if ``D`` is absent but a
        target ``VOUT`` **and** the input ``VIN`` are given, ``D = Vout/(Vout+Vin)``
        with the documented first-order caveat (ignores conduction/deadtime loss).
        ``DT`` (deadtime, default 100 ns) and ``RON`` (default 0.1 Ohm) as
        HALFBRIDGE.

        Returns None when FSW is missing/invalid, the deadtime is >= half the
        period, or no duty can be resolved.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        fsw = self._parse_si_number(raw["FSW"]) if "FSW" in raw else None
        if not fsw or fsw <= 0:
            return None

        def g(key, default=None):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        dt = g("DT", 100e-9)
        ron = g("RON", 0.1)
        if dt < 0 or dt >= 0.5 / fsw:
            return None

        d = g("D")
        if d is None:
            vout, vin = g("VOUT"), g("VIN")
            if vout and vout > 0 and vin and vin > 0:
                d = vout / (vout + vin)  # Vout/Vin = D/(1-D)  (non-inverting)
            else:
                return None

        d = min(max(d, 0.0), 1.0)
        return {"FSW": fsw, "DT": dt, "RON": ron, "D": d}

    def _add_sepic(self, component, ref: str, value: str):
        """Emit a bidirectional SEPIC switch stage (Sim.Device=SEPIC).

        Replaces ONLY the two switches -- the user's two inductors (VIN->A,
        B->GND), the **coupling cap Cs (A->B)** and the output cap stay real
        parts. The main switch ties node A (SWA) to GND for the on-fraction ``D``
        (charging L1 from VIN and driving L2 through Cs); the synchronous
        rectifier ties node B (SWB) to VOUT for the rest of the period (delivering
        the non-inverting output). Open-loop -- ``D`` is the control variable, so
        the ideal DC gain is ``Vout = Vin * D/(1-D)`` (steps up or down through the
        crossover at D=0.5, where Vout=Vin).

        The two switches sit on **independent** nodes (A and B, with the real Cs
        between them), so this is not a totem-pole leg: it emits the pair directly
        through ``_emit_sepic_switches`` rather than ``_emit_sync_leg`` (which would
        double the device count). The rectifier body diode is B->VOUT -- the
        load-bearing SEPIC orientation the Stage 27.4 device-level twin confirmed
        (``DQ2_body B VOUT``; reversed it silently mis-regulates). Bidirectional at
        the switch level for free: driving the VOUT port (Zeta) makes VIN regulate
        through the same synchronous switches (Stage 27.4's reverse-flow proof).
        The coupling cap self-biases to ~Vin (the defining SEPIC invariant) -- the
        SKILL guidance seeds Cs and uses ``stiff=True`` for convergence.
        """
        term = self._multiswitch_terminals(component, "sepic")
        if term is None:
            logger.warning(
                f"sepic {ref}: could not resolve VIN/SWA/SWB/VOUT/GND terminals "
                f"(by pin name) - skipping"
            )
            return
        params = self._sepic_params(component)
        if params is None:
            logger.warning(
                f"sepic {ref}: no usable FSW / duty resolved - skipping "
                f'(set Sim.Params="fsw=500k d=0.5")'
            )
            return

        # Cap-only unmodeled pins (BOOT/EN/VDD...) get a DC path, as HALFBRIDGE does.
        self._stub_unmodeled_pins(component, ref, set(term.values()), term["gnd"])

        swa, swb, vout, gnd = (str(term[k]) for k in ("swa", "swb", "vout", "gnd"))
        fsw, dt, ron, d = params["FSW"], params["DT"], params["RON"], params["D"]

        # A single switching pair -- like the inverting model (and unlike the
        # 4-switch one) there is no *other* leg to saturate this one, so a
        # saturated duty (0 or 1) is a dead short / open, not a valid SEPIC
        # operating point. This pair must genuinely switch, so reject a deadtime
        # that leaves no conduction window rather than fall back to a static tie.
        if d <= 0.0 or d >= 1.0 or dt >= min(d, 1.0 - d) / fsw:
            logger.warning(
                f"sepic {ref}: duty D ({self._fmt_num(d)}) with deadtime DT "
                f"({self._fmt_num(dt)}s) leaves no conduction window - skipping"
            )
            return

        self._emit_sepic_switches(
            swa, swb, vout, gnd,
            fsw=fsw, duty=d, dt=dt, ron=ron, suffix=ref,
        )

        n = self._fmt_num
        self.model_provenance[ref] = ResolvedModel(
            ref, "sepic", "sim_params",
            f"sepic_openloop(d={n(d)}, fsw={self._fmt_hz(fsw)})",
        )
        logger.debug(
            f"{ref}: 2-switch bidirectional SEPIC switch stage (open-loop, "
            f"d={n(d)}, fsw={self._fmt_hz(fsw)}, dt={n(dt)}); ideal DC gain "
            f"Vout/Vin = d/(1-d) = {n(d / (1.0 - d))} (non-inverting, steps "
            f"up/down through d=0.5); coupling cap Cs stays the user's real part "
            f"and self-biases to ~Vin; antiparallel body diodes make it "
            f"bidirectional (source/load placement sets direction)"
        )

    def _emit_cuk_switches(
        self, swa, swb, gnd, *, fsw, duty, dt, ron, suffix
    ) -> bool:
        """Emit the inverting Cuk's two switches (main + sync rectifier) directly.

        Modeled on ``_emit_sepic_switches`` -- two switches on independent nodes A
        and B (with the real series coupling cap Cs between them), complementary
        gate timing referenced to GND -- but for the **inverting Cuk** the two
        differences from the SEPIC are:

        - the sync rectifier spans ``swb->gnd`` (NOT ``swb->vout``): the negative
          output is reached through the user's real output inductor L2 (``B->VOUT``),
          so VOUT is not a switch node and is not passed to this emitter;
        - the rectifier body diode is ``swb->gnd`` (NOT ``swb->vout``) -- anode at
          node B (the same anode-at-B orientation as the SEPIC rectifier's
          ``swb->vout``, just returned to GND). This is load-bearing: node B swings
          to ``-(Vin+|Vout|)`` during the main-switch on-phase, so a reversed diode
          (``gnd->swb``) would forward-bias and clamp B, collapsing the inversion
          (measured: it turns the negative rail POSITIVE). The device-level twin's
          ``DQ2_body B 0`` locks the same orientation.

        The main switch is unchanged from the SEPIC: ``swa->gnd`` conducting for
        ``duty/fsw - dt`` at phase 0, body diode ``gnd->swa`` (the ground-referenced
        main FET's freewheel, the twin's ``DQ1_body 0 A``). The rectifier conducts
        for ``(1-duty)/fsw - dt``, delayed ``duty/fsw`` (complementary, one deadtime
        at each edge).

        Gate PULSEs are referenced to ``gnd``. The user's real coupling cap Cs
        (``swa->swb``), both inductors, Cout and Rload survive -- this emits ONLY the
        switch stage. Returns True on success; False -- emitting **nothing** -- when
        the deadtime leaves no conduction window (``dt >= min(duty, 1-duty)/fsw``),
        so the caller can log a warning and skip.
        """
        per = 1.0 / fsw
        on_m = duty * per - dt          # main-switch conduction time
        on_r = (1.0 - duty) * per - dt  # sync-rectifier conduction time
        if on_m <= 0 or on_r <= 0:
            return False
        td_r = duty * per               # rectifier turn-on delay (complementary)
        edge = per / 200.0
        ron = self._fmt_num(ron)
        gm, gr = f"{suffix}_gm", f"{suffix}_gr"
        lines = [
            # Complementary gate drives with a deadtime gap, referenced to GND:
            # main on first (duty*per - DT), rectifier after (delayed duty*per),
            # both off during the two DT windows.
            f"V{suffix}_gm {gm} {gnd} PULSE(0 5 0 "
            f"{edge:.6g} {edge:.6g} {on_m:.6g} {per:.6g})",
            f"V{suffix}_gr {gr} {gnd} PULSE(0 5 {td_r:.6g} "
            f"{edge:.6g} {edge:.6g} {on_r:.6g} {per:.6g})",
            f"S{suffix}_m {swa} {gnd} {gm} {gnd} SW{suffix}",
            # Cuk sync rectifier ties node B to GND (the output is behind L2, not
            # through the rectifier) -- the one switch-node difference from SEPIC.
            f"S{suffix}_r {swb} {gnd} {gr} {gnd} SW{suffix}",
            f".model SW{suffix} SW(Ron={ron} Roff=1e6 Vt=2.5 Vh=0.2)",
            # Body diodes: main gnd->swa (freewheel), rectifier swb->gnd (anode at
            # B, as the SEPIC rectifier -- load-bearing: B swings to -(Vin+|Vout|)
            # on the main-switch on-phase, so the reversed gnd->swb would clamp B
            # and turn the negative rail positive; the twin's DQ2_body B 0).
            f"D{suffix}_m {gnd} {swa} DFW{suffix}",
            f"D{suffix}_r {swb} {gnd} DFW{suffix}",
            f".model DFW{suffix} D(IS=1e-9 N=1.05 CJO=100p)",
        ]
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        return True

    def _cuk_params(self, component) -> Optional[dict]:
        """Params for an inverting Cuk macromodel, or None if unusable.

        ``FSW`` is required. ``D`` (the main-switch on-fraction) is the open-loop
        control variable and sets the ideal (inverting) DC gain
        ``Vout = -Vin * D/(1-D)`` -- swept, not regulated, matching every other
        switch macromodel's honesty limit. Convenience: if ``D`` is absent but a
        target ``VOUT`` (the negative output, either sign accepted) **and** the
        input ``VIN`` are given, ``D = |Vout| / (|Vout| + Vin)`` (abs-based, as the
        inverting buck-boost does) with the documented first-order caveat (ignores
        conduction/deadtime loss). ``DT`` (deadtime, default 100 ns) and ``RON``
        (default 0.1 Ohm) as HALFBRIDGE.

        Returns None when FSW is missing/invalid, the deadtime is >= half the
        period, or no duty can be resolved.
        """
        raw = self._parse_sim_params(self._sim_props(component).get("params"))
        fsw = self._parse_si_number(raw["FSW"]) if "FSW" in raw else None
        if not fsw or fsw <= 0:
            return None

        def g(key, default=None):
            v = self._parse_si_number(raw[key]) if key in raw else None
            return v if v is not None else default

        dt = g("DT", 100e-9)
        ron = g("RON", 0.1)
        if dt < 0 or dt >= 0.5 / fsw:
            return None

        d = g("D")
        if d is None:
            vout, vin = g("VOUT"), g("VIN")
            avout = abs(vout) if vout is not None else 0.0
            if avout > 0 and vin and vin > 0:
                d = avout / (avout + vin)  # |Vout|/Vin = D/(1-D)  (inverting)
            else:
                return None

        d = min(max(d, 0.0), 1.0)
        return {"FSW": fsw, "DT": dt, "RON": ron, "D": d}

    def _add_cuk(self, component, ref: str, value: str):
        """Emit an inverting Cuk switch stage (Sim.Device=CUK).

        Replaces ONLY the two switches -- the user's two inductors (VIN->A via L1,
        B->VOUT via L2), the **series coupling cap Cs (A->B)** and the output cap
        stay real parts. The main switch ties node A (SWA) to GND for the
        on-fraction ``D`` (charging L1 from VIN); the synchronous rectifier ties
        node B (SWB) to GND for the rest of the period. Unlike the SEPIC the output
        is **not** a switch node -- it is the **negative** rail reached through the
        user's real output inductor L2 (B->VOUT). Open-loop -- ``D`` is the control
        variable, so the ideal (inverting) DC gain is ``Vout = -Vin * D/(1-D)``.

        The two switches sit on **independent** nodes (A and B, with the real Cs
        between them), so this is not a totem-pole leg: it emits the pair directly
        through ``_emit_cuk_switches`` rather than ``_emit_sync_leg`` (which would
        double the device count). The rectifier body diode is B->GND (anode at node
        B, the same anode-at-B orientation as the SEPIC rectifier) -- load-bearing:
        B swings to ``-(Vin+|Vout|)`` on the main-switch on-phase, so the reversed
        GND->B would forward-bias and clamp B, collapsing the inversion (it turns
        the negative rail positive); the twin's ``DQ2_body B 0``. The main switch is
        ground-referenced (A->GND, body diode GND->A, the same as the SEPIC main).
        The coupling cap self-biases to ``Vin - Vout = Vin + |Vout|`` (the Cuk
        invariant -- a larger bias than the SEPIC's ~Vin, since L2 balances V(B) to
        the negative output) -- the SKILL guidance seeds VOUT and uses ``stiff=True``
        on the negative rail for convergence. Bidirectional at the switch level for
        free: the antiparallel body diodes let power flow either way (source/load
        placement sets direction).
        """
        term = self._multiswitch_terminals(component, "cuk")
        if term is None:
            logger.warning(
                f"cuk {ref}: could not resolve VIN/SWA/SWB/VOUT/GND terminals "
                f"(by pin name) - skipping"
            )
            return
        params = self._cuk_params(component)
        if params is None:
            logger.warning(
                f"cuk {ref}: no usable FSW / duty resolved - skipping "
                f'(set Sim.Params="fsw=500k d=0.5")'
            )
            return

        # Cap-only unmodeled pins (BOOT/EN/VDD...) get a DC path, as HALFBRIDGE does.
        self._stub_unmodeled_pins(component, ref, set(term.values()), term["gnd"])

        swa, swb, gnd = (str(term[k]) for k in ("swa", "swb", "gnd"))
        fsw, dt, ron, d = params["FSW"], params["DT"], params["RON"], params["D"]

        # A single switching pair -- like the SEPIC and inverting models there is
        # no *other* leg to saturate this one, so a saturated duty (0 or 1) is a
        # dead short / open, not a valid Cuk operating point. This pair must
        # genuinely switch, so reject a deadtime that leaves no conduction window
        # rather than fall back to a static tie.
        if d <= 0.0 or d >= 1.0 or dt >= min(d, 1.0 - d) / fsw:
            logger.warning(
                f"cuk {ref}: duty D ({self._fmt_num(d)}) with deadtime DT "
                f"({self._fmt_num(dt)}s) leaves no conduction window - skipping"
            )
            return

        self._emit_cuk_switches(
            swa, swb, gnd,
            fsw=fsw, duty=d, dt=dt, ron=ron, suffix=ref,
        )

        n = self._fmt_num
        self.model_provenance[ref] = ResolvedModel(
            ref, "cuk", "sim_params",
            f"cuk_openloop(d={n(d)}, fsw={self._fmt_hz(fsw)})",
        )
        logger.debug(
            f"{ref}: 2-switch inverting Cuk switch stage (open-loop, "
            f"d={n(d)}, fsw={self._fmt_hz(fsw)}, dt={n(dt)}); ideal DC gain "
            f"Vout/Vin = -d/(1-d) = {n(-d / (1.0 - d))} (negative rail, reached "
            f"through the real output inductor L2); coupling cap Cs stays the "
            f"user's real part and self-biases to ~Vin+|Vout|; antiparallel body "
            f"diodes make it bidirectional (source/load placement sets direction)"
        )

    @staticmethod
    def _parse_frequency(value) -> Optional[float]:
        """Parse a GBW/frequency string ('1.4G', '10MEG', '1k', '5e5', '2MHz') to Hz.

        SI suffixes G/MEG/M/k (``M`` means mega here -- milli-Hz is meaningless for a
        GBW), optional trailing ``Hz``. Returns None if unparseable.
        """
        if value is None:
            return None
        s = str(value).strip().lower().replace(" ", "")
        if s.endswith("hz"):
            s = s[:-2]
        if not s:
            return None
        mult = 1.0
        for suf, m in (("meg", 1e6), ("g", 1e9), ("m", 1e6), ("k", 1e3)):
            if s.endswith(suf):
                mult = m
                s = s[: -len(suf)]
                break
        try:
            return float(s) * mult
        except ValueError:
            return None

    @staticmethod
    def _fmt_hz(hz: float) -> str:
        """Compact human label for a frequency ('1.4G', '10M', '5k', '200')."""
        for suf, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
            if hz >= scale:
                return f"{hz / scale:g}{suf}"
        return f"{hz:g}"

    def _named_terminal_nodes(self, component) -> dict:
        """``{uppercased pin name: spice node}`` for a component's connected pins.

        Transistor terminals must be mapped by pin *name* (C/B/E, D/G/S), not pin
        number: KiCad symbols number them inconsistently (2N7000 is pin1=S,2=G,3=D;
        BC547 is 1=C,2=B,3=E), so a positional mapping silently swaps
        drain/source (or collector/emitter) for many real parts.
        """
        out = {}
        pin_map = getattr(component, "_pins", None)
        if isinstance(pin_map, dict):
            for pin in pin_map.values():
                net = getattr(pin, "net", None)
                name = (getattr(pin, "name", "") or "").strip().upper()
                if net is not None and name:
                    out[name] = self.node_map.get(net.name, net.name)
        return out

    def _add_bjt_transistor(self, component, ref: str, value: str):
        """Add BJT transistor to SPICE circuit.

        Terminals are resolved by pin name (C/B/E); falls back to pin-number order
        only when the symbol doesn't name all three (rare).
        """
        named = self._named_terminal_nodes(component)
        if all(k in named for k in ("C", "B", "E")):
            c_node, b_node, e_node = named["C"], named["B"], named["E"]
        else:
            nodes = self._get_component_nodes(component)
            if len(nodes) < 3:
                logger.warning(
                    f"BJT {ref} needs 3 connections (C,B,E), got {len(nodes)}"
                )
                return
            c_node, b_node, e_node = nodes[0], nodes[1], nodes[2]

        # Determine model (NPN/PNP) from value keyword or symbol polarity, applying
        # any Sim.Params override.
        model_name = self._resolve_device_model(component, ref) or "DefaultNPN"

        self.spice_circuit.Q(ref, c_node, b_node, e_node, model=model_name)
        logger.debug(
            f"Added BJT {ref}: C={c_node}, B={b_node}, E={e_node}, model={model_name}"
        )

    def _add_mosfet(self, component, ref: str, value: str):
        """Add MOSFET to SPICE circuit.

        Terminals are resolved by pin name (D/G/S, plus an optional bulk pin B);
        falls back to pin-number order only when the symbol doesn't name D/G/S.
        Bulk defaults to the source when the symbol has no separate bulk pin
        (3-terminal parts like 2N7000/BSS84).
        """
        named = self._named_terminal_nodes(component)
        if all(k in named for k in ("D", "G", "S")):
            d_node, g_node, s_node = named["D"], named["G"], named["S"]
            b_node = named.get("B", s_node)
        else:
            nodes = self._get_component_nodes(component)
            if len(nodes) < 3:
                logger.warning(
                    f"MOSFET {ref} needs at least 3 connections (D,G,S), got {len(nodes)}"
                )
                return
            d_node, g_node, s_node = nodes[0], nodes[1], nodes[2]
            b_node = nodes[3] if len(nodes) >= 4 else nodes[2]

        # Determine model (NMOS/PMOS) from value keyword or symbol polarity, applying
        # any Sim.Params override.
        model_name = self._resolve_device_model(component, ref) or "DefaultNMOS"

        self.spice_circuit.M(ref, d_node, g_node, s_node, b_node, model=model_name)
        # Emit the intrinsic body diode + Coss for a curated power part (or when
        # Sim.Params forces them) -- a Level-1 .model can't carry either.
        self._emit_mosfet_companions(component, ref, d_node, s_node)
        logger.debug(
            f"Added MOSFET {ref}: D={d_node}, G={g_node}, S={s_node}, B={b_node}, "
            f"model={model_name}"
        )

    def _emit_mosfet_companions(self, component, ref, d_node, s_node) -> None:
        """Emit an antiparallel body diode + a drain-source Coss for a power MOSFET.

        Fires when the resolved model carries ``body_diode``/``coss`` metadata (a
        curated ModelLibrary power part), or when ``Sim.Params`` supplies
        ``COSS=<F>`` / ``BODY=1``. A Level-1 ``.model`` card cannot express either,
        so the composite (M + D + C) is what gives a switch its reverse-conduction
        and switch-node capacitance -- required for ZVS / hard-switching fidelity.

        Body-diode polarity: for an NMOS the intrinsic diode's anode is at the
        source (conducts S->D on a negative Vds); for a PMOS it is reversed.
        Provenance name is annotated ``+body``/``+coss`` so the composite is never
        silent.
        """
        base = self._device_model_name(component)
        if base is None:
            return
        entry, _ = self._resolve_library_model(base, "mosfet")
        body = (
            dict(entry.body_diode)
            if entry is not None and getattr(entry, "body_diode", None)
            else None
        )
        coss = getattr(entry, "coss", None) if entry is not None else None

        # Sim.Params can override Coss or force a default body diode onto any part.
        overrides = self._parse_sim_params(self._sim_props(component).get("params"))
        if "COSS" in overrides:
            c = self._parse_si_number(overrides["COSS"])
            if c is not None and c > 0:
                coss = c
        if (
            str(overrides.get("BODY", "")).strip().lower() in ("1", "true", "yes", "on")
            and body is None
        ):
            body = {"IS": 1e-9, "RS": 0.02, "CJO": 500e-12, "BV": 60}

        if body is None and coss is None:
            return  # a plain small-signal/generic MOSFET -- no companions

        spec, _tier, _resolved = self._lookup_model_spec(base, "mosfet")
        mtype = spec[0].upper() if spec else "NMOS"
        n = self._fmt_num
        lines = []
        notes = []
        if body is not None:
            dmodel = f"DBODY{ref}"
            pstr = " ".join(f"{k}={n(v)}" for k, v in body.items())
            anode, cathode = (
                (d_node, s_node) if mtype == "PMOS" else (s_node, d_node)
            )
            lines.append(f"D{ref}_body {anode} {cathode} {dmodel}")
            lines.append(f".model {dmodel} D({pstr})")
            notes.append("+body")
        if coss is not None:
            lines.append(f"C{ref}_oss {d_node} {s_node} {n(coss)}")
            notes.append("+coss")
        self.spice_circuit.raw_spice += "\n" + "\n".join(lines)
        prov = self.model_provenance.get(ref)
        if prov is not None and notes:
            prov.name = f"{prov.name} {''.join(notes)}"

    def _add_voltage_source(self, component, ref: str, value: str):
        """Add voltage source to SPICE circuit.

        The emitted SPICE spec depends on the KiCad symbol:

        * ``VDC`` -> a DC value (from ``value``, default 5 V).
        * ``VAC`` -> ``DC 0 AC <mag>`` (small-signal only; ``value`` is the AC mag).
        * ``VSIN`` -> ``DC <off> AC <acmag> SIN(<off> <ampl> <freq> <td> <theta>)``
          -- both AC-analysis magnitude *and* a transient sinusoid, so one source
          works for both `.ac` and `.tran`. ``value`` sets ampl+acmag by default.
        * ``VPULSE`` -> ``PULSE(<v1> <v2> <td> <tr> <tf> <pw> <per>)``.
        * ``VPWL`` -> ``PWL(<t1> <v1> <t2> <v2> ...)`` (from a ``points`` field).

        Waveform parameters are read from the component's extra fields (any kwarg
        passed to ``Component`` lands in ``_extra_fields``), e.g.
        ``Component(symbol="Simulation_SPICE:VSIN", amplitude="1", frequency="1k")``.
        Numeric values keep their SI suffix (``1k``/``1m``/``1u``/``1n``) -- ngspice
        parses those directly.
        """
        nodes = self._get_component_nodes(component)
        if len(nodes) < 2:
            logger.warning(
                f"Voltage source {ref} needs 2 connections, got {len(nodes)}"
            )
            return

        # Add to list of voltage sources for tracking
        self.voltage_sources.append(ref)
        # Mark this source's nets so the net-name heuristic won't double-drive them.
        self.driven_nets.update(self._component_net_names(component))

        symbol = (getattr(component, "symbol", "") or "").upper()
        params = self._source_params(component)
        spec = self._voltage_source_spec(symbol, value, params)

        # nodes[] is in pin-number order, so nodes[0] is pin 1 (KiCad Sim.Pins
        # "1=+") and nodes[1] is pin 2 ("2=-"): V(name, +, -, spec).
        self.spice_circuit.V(ref, nodes[0], nodes[1], spec)
        logger.debug(
            f"Added voltage source {ref}: {nodes[0]}(+) -> {nodes[1]}(-) = {spec}"
        )

    def _source_params(self, component) -> dict:
        """Lowercased waveform-param map for a source (empty if none).

        Merges KiCad ``Sim.Params`` (as a base) with the component's own extra
        fields, where an explicit extra field wins over a ``Sim.Params`` value.
        The ``Sim.*`` fields themselves are dropped so they don't masquerade as
        waveform params.
        """
        extra = getattr(component, "_extra_fields", None)
        if not isinstance(extra, dict):
            return {}
        merged = {
            k.lower(): v
            for k, v in self._parse_sim_params(
                self._sim_props(component).get("params")
            ).items()
        }
        for k, v in extra.items():
            low = str(k).lower()
            if low.startswith("sim.") or low.startswith("sim_"):
                continue
            merged[low] = v
        return merged

    @staticmethod
    def _wave_num(value, default: str) -> str:
        """Normalize a waveform number, keeping its SI suffix for ngspice.

        Strips a trailing unit (V/A/Hz/s/F/H) so ``"1kHz"`` -> ``"1k"`` and
        ``"5V"`` -> ``"5"`` while leaving the SI prefix (k/m/u/n/p) intact.
        Returns ``default`` when ``value`` is None/empty.
        """
        if value is None:
            return default
        s = str(value).strip()
        if not s:
            return default
        for unit in ("hz", "v", "a", "s", "f", "h"):
            if len(s) > len(unit) and s.lower().endswith(unit):
                s = s[: -len(unit)]
                break
        return s

    def _pick(self, params: dict, keys, default: str) -> str:
        """First present param among ``keys`` (normalized), else ``default``."""
        for k in keys:
            if k in params and params[k] is not None and str(params[k]).strip():
                return self._wave_num(params[k], default)
        return default

    def _voltage_source_spec(self, symbol: str, value, params: dict):
        """Build the SPICE source spec (string or float) from symbol + params."""
        if "VPULSE" in symbol:
            v1 = self._pick(params, ("v1", "initial", "low"), "0")
            v2 = self._pick(
                params,
                ("v2", "pulsed", "high", "amplitude"),
                self._wave_num(value, "1"),
            )
            td = self._pick(params, ("td", "delay"), "0")
            tr = self._pick(params, ("tr", "rise"), "1n")
            tf = self._pick(params, ("tf", "fall"), "1n")
            pw = self._pick(params, ("pw", "width"), "0.5m")
            per = self._pick(params, ("per", "period"), "1m")
            return f"PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})"

        if "VPWL" in symbol:
            pts = self._pwl_points(params)
            if pts:
                return f"PWL({pts})"
            # No points given -> degrade to a DC source at `value` (or 0).
            return self._convert_value_to_spice(value or "0V", "V")

        if "VSIN" in symbol:
            base = self._wave_num(value, "1")  # value sets ampl + AC mag by default
            offset = self._pick(params, ("offset", "dc", "voffset"), "0")
            ampl = self._pick(params, ("amplitude", "ampl", "amp"), base)
            freq = self._pick(params, ("frequency", "freq", "f"), "1k")
            td = self._pick(params, ("td", "delay"), "0")
            theta = self._pick(params, ("theta", "damping"), "0")
            ac_mag = self._pick(params, ("ac", "ac_mag", "acmag"), base)
            return f"DC {offset} AC {ac_mag} SIN({offset} {ampl} {freq} {td} {theta})"

        if "VAC" in symbol:
            ac_mag = self._convert_value_to_spice(value, "V") if value else 1.0
            return f"DC 0 AC {ac_mag}"

        # VDC / default: a plain DC value.
        return self._convert_value_to_spice(value or "5V", "V")

    def _pwl_points(self, params: dict) -> str:
        """Render VPWL points ('t1 v1 t2 v2 ...') from a ``points`` field.

        Accepts a list/tuple of (t, v) pairs, a flat list, or a whitespace/
        comma-separated string. Returns '' if no usable points are present.
        """
        pts = params.get("points")
        if pts is None:
            return ""
        if isinstance(pts, str):
            toks = [t for t in pts.replace(",", " ").split() if t]
            return " ".join(self._wave_num(t, "0") for t in toks)
        if isinstance(pts, (list, tuple)):
            flat = []
            for item in pts:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            return " ".join(self._wave_num(t, "0") for t in flat)
        return ""

    def _add_current_source(self, component, ref: str, value: str):
        """Add current source to SPICE circuit.

        Symbol-aware, mirroring ``_add_voltage_source``: ``ISIN`` carries an AC
        magnitude (for ``.ac``) plus a transient sinusoid, ``IPULSE``/``IPWL`` get
        their waveforms, and ``IDC``/default stays a plain DC current. Without this
        an ``ISIN`` source injected only DC, so AC analysis saw zero drive.
        """
        nodes = self._get_component_nodes(component)
        if len(nodes) < 2:
            logger.warning(
                f"Current source {ref} needs 2 connections, got {len(nodes)}"
            )
            return

        # Mark this source's nets so the net-name heuristic won't double-drive them.
        self.driven_nets.update(self._component_net_names(component))

        symbol = (getattr(component, "symbol", "") or "").upper()
        params = self._source_params(component)
        spec = self._current_source_spec(symbol, value, params)

        # nodes[] is pin-number ordered (pin 1 = +, pin 2 = -); ngspice current
        # flows from the + node, through the source, to the - node.
        self.spice_circuit.I(ref, nodes[0], nodes[1], spec)
        logger.debug(
            f"Added current source {ref}: {nodes[0]}(+) -> {nodes[1]}(-) = {spec}"
        )

    def _current_source_spec(self, symbol: str, value, params: dict):
        """Build the SPICE current-source spec (string or float) from symbol+params.

        Current-source analogue of ``_voltage_source_spec`` (units 'A'):

        * ``ISIN``  -> ``DC <off> AC <acmag> SIN(<off> <ampl> <freq> <td> <theta>)``
          -- one source serves both ``.ac`` (via AC magnitude) and ``.tran``.
          ``value`` sets ampl+acmag by default.
        * ``IPULSE`` -> ``PULSE(<i1> <i2> <td> <tr> <tf> <pw> <per>)``.
        * ``IPWL``  -> ``PWL(<t1> <i1> ...)`` (from a ``points`` field).
        * ``IDC`` / default -> a plain DC current (from ``value``, default 1 mA).
        """
        if "IPULSE" in symbol:
            i1 = self._pick(params, ("i1", "initial", "low"), "0")
            i2 = self._pick(
                params,
                ("i2", "pulsed", "high", "amplitude"),
                self._wave_num(value, "1"),
            )
            td = self._pick(params, ("td", "delay"), "0")
            tr = self._pick(params, ("tr", "rise"), "1n")
            tf = self._pick(params, ("tf", "fall"), "1n")
            pw = self._pick(params, ("pw", "width"), "0.5m")
            per = self._pick(params, ("per", "period"), "1m")
            return f"PULSE({i1} {i2} {td} {tr} {tf} {pw} {per})"

        if "IPWL" in symbol:
            pts = self._pwl_points(params)
            if pts:
                return f"PWL({pts})"
            return self._convert_value_to_spice(value or "0A", "I")

        if "ISIN" in symbol:
            base = self._wave_num(value, "1")  # value sets ampl + AC mag by default
            offset = self._pick(params, ("offset", "dc", "ioffset"), "0")
            ampl = self._pick(params, ("amplitude", "ampl", "amp"), base)
            freq = self._pick(params, ("frequency", "freq", "f"), "1k")
            td = self._pick(params, ("td", "delay"), "0")
            theta = self._pick(params, ("theta", "damping"), "0")
            ac_mag = self._pick(params, ("ac", "ac_mag", "acmag"), base)
            return f"DC {offset} AC {ac_mag} SIN({offset} {ampl} {freq} {td} {theta})"

        # IDC / default: a plain DC current.
        return self._convert_value_to_spice(value or "1mA", "I")

    @staticmethod
    def _pin_sort_key(num):
        """Sort key that orders pin numbers numerically ('2' < '10') when possible."""
        s = str(num)
        return (0, int(s)) if s.isdigit() else (1, s)

    def _component_net_names(self, component) -> set:
        """Original (unmapped) net names connected to a component's pins.

        Best effort: only available for live Component objects that carry a
        ``_pins`` map. Returns an empty set for dict/JSON-shaped components.
        """
        names = set()
        pin_map = getattr(component, "_pins", None)
        if isinstance(pin_map, dict):
            for pin in pin_map.values():
                net = getattr(pin, "net", None)
                name = getattr(net, "name", None)
                if name:
                    names.add(name)
        return names

    def _get_component_nodes(self, component) -> List[str]:
        """Get the SPICE nodes connected to a component, in pin-number order.

        Pin order is significant: a voltage/current source's pin 1 is + and pin 2
        is - (KiCad ``Sim.Pins "1=+ 2=-"``), and a transistor's pins are C/B/E or
        D/G/S. A live Component exposes a ``{pin_num: Pin}`` map, so we read the
        node for each pin in ascending pin-number order. Unlike the legacy net
        scan this does *not* dedupe or alphabetically sort, so polarity and
        terminal order are preserved.

        Falls back to the legacy net scan for dict/JSON-shaped circuits whose
        components have no live ``_pins`` map.
        """
        pin_map = getattr(component, "_pins", None)
        if isinstance(pin_map, dict) and pin_map:
            ordered = []
            for _num, pin in sorted(
                pin_map.items(), key=lambda kv: self._pin_sort_key(kv[0])
            ):
                net = getattr(pin, "net", None)
                net_name = getattr(net, "name", None)
                if net_name is None:
                    continue  # unconnected pin
                ordered.append(self.node_map.get(net_name, net_name))
            if ordered:
                return ordered
            # else: fall through to the net scan (e.g. pins present but netless)

        return self._get_component_nodes_by_net_scan(component)

    def _get_component_nodes_by_net_scan(self, component) -> List[str]:
        """Legacy fallback: recover a component's nodes by scanning nets.

        Used for dict/JSON-shaped circuits without live Pin objects. Nodes are
        alphabetically sorted (pin order is not recoverable here), which is fine
        for symmetric R/C/L but does not guarantee polarity for sources.
        """
        nodes = []

        # Get component connections from the circuit
        # circuit.nets is a dict, iterate over values
        if hasattr(self.circuit.nets, "values"):
            nets_to_check = self.circuit.nets.values()
        else:
            nets_to_check = self.circuit.nets

        for net in nets_to_check:
            net_name = getattr(net, "name", str(net))

            # Check if this net has pins connected to our component
            if hasattr(net, "pins"):
                for pin in net.pins:
                    # Each pin has a reference back to its component
                    # The pin string format is like "Pin(~ of R1, net=VIN)"
                    pin_str = str(pin)
                    component_ref = getattr(component, "ref", "")

                    # Check if this pin belongs to our component
                    if f" of {component_ref}," in pin_str:
                        # Map to SPICE node name
                        spice_node = self.node_map.get(net_name, net_name)
                        if spice_node not in nodes:
                            nodes.append(spice_node)
                        break

        # Sort nodes to ensure consistent pin ordering (important for SPICE)
        # Convert all to strings first to avoid type comparison issues
        nodes.sort(key=str)

        # If we didn't find connections, log for debugging
        if not nodes:
            logger.warning(
                f"No connections found for component {getattr(component, 'ref', 'unknown')}"
            )

        return nodes

    def _convert_value_to_spice(self, value: str, component_type: str) -> float:
        """Convert circuit-synth component value to SPICE format.

        ``component_type`` is the SPICE element letter ("R"/"C"/"L" for passives,
        "V"/"I" for sources). For a **source** an unparseable ``value`` is a
        correctness trap -- silently substituting 1.0 changes the circuit -- so it
        raises :class:`SimulationValidationError` rather than warning. Passives keep
        the lenient warn-and-default behaviour (a symmetric R/C/L defaulting is far
        less dangerous), but log at error level so it is never truly silent.
        """
        is_source = component_type in ("V", "I")
        if not value:
            # Default values
            defaults = {"R": 1000, "C": 1e-6, "L": 1e-3}
            return defaults.get(component_type, 1.0)

        # Parse value string (e.g., "10k", "100nF", "1mH", "-1.0", "2.5e-3").
        # The numeric group carries an optional leading sign and exponent so a
        # negative DC source value ("-1") keeps its sign instead of failing the
        # match and hitting the fallback -- which used to silently emit +1.0 (B4).
        value = str(value).strip().replace(" ", "")

        # Extract numeric part and suffix
        match = re.match(
            r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)([a-zA-Z]*)$", value
        )
        if not match:
            if is_source:
                raise SimulationValidationError(
                    [f"could not parse source value '{value}' (element {component_type})"]
                )
            logger.error(
                f"Could not parse value '{value}' (element {component_type}); "
                f"using 1.0 -- fix the value, this default is almost certainly wrong"
            )
            return 1.0

        numeric_part = float(match.group(1))
        suffix = match.group(2).lower()

        # Convert suffixes to multipliers
        multipliers = {
            # Resistance
            "r": 1,
            "ohm": 1,
            "ohms": 1,
            "k": 1e3,
            "kohm": 1e3,
            "kohms": 1e3,
            "m": 1e6,
            "meg": 1e6,
            "mohm": 1e6,
            "mohms": 1e6,
            # Capacitance
            "f": 1,
            "pf": 1e-12,
            "nf": 1e-9,
            "uf": 1e-6,
            "mf": 1e-3,
            "p": 1e-12,
            "n": 1e-9,
            "u": 1e-6,
            # Inductance
            "h": 1,
            "nh": 1e-9,
            "uh": 1e-6,
            "mh": 1e-3,
            # Voltage
            "v": 1,
            "mv": 1e-3,
            "kv": 1e3,
            # Current
            "a": 1,
            "ma": 1e-3,
            "ua": 1e-6,
            "na": 1e-9,
        }

        multiplier = multipliers.get(suffix, 1.0)
        return numeric_part * multiplier

    def _active_driver_net_names(self) -> set:
        """Net names already driven by a real ``OUTPUT``/``PWROUT`` pin.

        The net-name supply heuristic must skip these: an op-amp/regulator output
        already sources the net, and a phantom rail stacked on top would fight the
        real driver (E2E D1 -- a net named ``VIN_SS`` driven by an op-amp output
        silently got a 5 V supply, corrupting the whole VCO). Derived from the
        *component* side because the flat-view nets don't carry pin funcs; each
        live component exposes a ``{pin_num: Pin}`` map whose pins carry both
        ``.net`` and ``.func``. A ``Sim.Enable=0`` part drives nothing in sim, so
        it is skipped. Best-effort: pins without a usable ``func`` are ignored, so
        a genuine undriven rail is never mis-skipped.
        """
        # Live Pin objects carry ``func`` as a ``pin_types`` IntEnum; the flat
        # view's AdaptedComponent stores it as a lowercase string ("output"/
        # "pwrout"). Accept both forms.
        driving_names = {"output", "pwrout"}
        try:
            from skidl.pin import pin_types

            driving_vals = {int(pin_types.OUTPUT), int(pin_types.PWROUT)}
        except Exception:  # noqa: BLE001 - pin metadata unavailable
            driving_vals = set()

        def _is_driver(func) -> bool:
            if func is None:
                return False
            if isinstance(func, str):
                return func.strip().lower() in driving_names
            try:
                if int(func) in driving_vals:
                    return True
            except (TypeError, ValueError):
                pass
            nm = getattr(func, "name", None)
            return isinstance(nm, str) and nm.lower() in driving_names

        driven = set()
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            pin_map = getattr(component, "_pins", None)
            if not isinstance(pin_map, dict):
                continue
            for pin in pin_map.values():
                if _is_driver(getattr(pin, "func", None)):
                    name = getattr(getattr(pin, "net", None), "name", None)
                    if name:
                        driven.add(name)
        return driven

    def _series_reachable_net_names(self) -> set:
        """Nets DC-reachable from an explicit source through series R/L passives.

        A net downstream of a ``Simulation_SPICE`` source via a resistor/inductor
        is ALREADY driven (the op-point sees the source through the passive), so
        the rail-name heuristic must not stack a phantom ideal supply on it and
        short the series element (E2E B2 -- a net named ``VIN`` behind a 10 Ohm
        input resistor got a second 5 V supply across the resistor). Capacitors
        are excluded: they carry no DC, so a net behind a series cap is NOT
        source-driven at the operating point. Returns only the ADDITIONAL nets
        reached (the directly-driven ones are already handled by driven_nets)."""
        adj = {}
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) not in ("resistor", "inductor"):
                continue
            names = list(self._component_net_names(component))
            if len(names) != 2:
                continue  # only a plain 2-terminal series element bridges DC
            a, b = names
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        seed = set(self.driven_nets)
        seen = set(seed)
        stack = list(seed)
        while stack:
            cur = stack.pop()
            for nbr in adj.get(cur, ()):
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        return seen - seed

    def _add_power_sources(self):
        """Add power sources needed for simulation."""
        # Check if we need to add power sources based on net names
        power_nets = []
        # Nets driven from an explicit source through a series R/L (E2E B2): the
        # rail heuristic must not re-supply these and short the series element.
        series_reachable = self._series_reachable_net_names()

        # Handle both dict and list formats for nets
        if hasattr(self.circuit.nets, "values"):
            nets_to_process = self.circuit.nets.values()
        elif hasattr(self.circuit.nets, "__iter__"):
            nets_to_process = self.circuit.nets
        else:
            return

        active_drivers = self._active_driver_net_names()

        for net in nets_to_process:
            orig_name = getattr(net, "name", str(net))
            # An explicit source component already drives this net; don't add a
            # second heuristic supply on top of it (that would over-constrain the
            # node / fight the real source).
            if orig_name in self.driven_nets:
                continue
            voltage = self._heuristic_source_voltage(orig_name)
            if voltage is None:
                continue
            # A net already driven by a real OUTPUT/PWROUT pin (op-amp/regulator
            # output) must not get a heuristic supply stacked on it (E2E D1).
            if orig_name in active_drivers:
                logger.warning(
                    f"Net '{orig_name}': name matches the supply heuristic "
                    f"({voltage} V) but it is already driven by an OUTPUT/PWROUT "
                    f"pin -- NOT injecting a supply (a phantom rail would fight the "
                    f"real driver). Rename the net if you did want a bare rail here."
                )
                continue
            # A net driven from an explicit source THROUGH a series R/L is already
            # supplied; a phantom ideal rail would short the series element (E2E
            # B2 -- an input-protection resistor silently bypassed).
            if orig_name in series_reachable:
                logger.warning(
                    f"Net '{orig_name}': name matches the supply heuristic "
                    f"({voltage} V) but it is driven from an explicit source "
                    f"through a series R/L -- NOT injecting (a phantom rail would "
                    f"short the series element). Rename the net if you did want a "
                    f"bare rail here."
                )
                continue
            power_nets.append((orig_name, voltage))

        # Add voltage sources
        for i, (net_name, voltage) in enumerate(power_nets):
            source_name = f"V_supply_{i+1}"
            spice_node = self.node_map.get(net_name, net_name)
            self.spice_circuit.V(
                source_name, spice_node, self.spice_circuit.gnd, voltage @ u_V
            )
            logger.warning(
                f"Net '{net_name}': injecting heuristic {voltage} V supply "
                f"{source_name} from its NAME. If unintended, rename the net or "
                f"drive it with an explicit Simulation_SPICE:VDC (explicit "
                f"sources suppress this)."
            )

    # ------------------------------------------------------------------ #
    # Floating-node conditioning (F4)                                     #
    # ------------------------------------------------------------------ #

    # PySpice element classes that are an OPEN at DC: a node reachable from ground
    # only through these has no operating point (ngspice: 'singular matrix: check
    # node <n>'). A capacitor is open at DC; an (independent) current source pins a
    # branch current, not a node voltage.
    _DC_OPEN_ELEMENT_TYPES = ("Capacitor", "CurrentSource")

    @staticmethod
    def _tie_floating_enabled() -> bool:
        """Gmin-tie of DC-floating nodes (on by default).

        ``SKIDL_SIM_TIE_FLOATING=0`` is the kill switch: no tie resistors are
        added, so a circuit's deck is byte-identical to the pre-feature emission.
        """
        return os.environ.get("SKIDL_SIM_TIE_FLOATING", "1") != "0"

    def _conducting_nodes(self, element) -> List[str]:
        """The nodes an element gives a DC path to (F4 helper).

        Empty for a DC-open element (capacitor / current source). For a MOSFET the
        gate is DC-isolated from the channel (a capacitive terminal), so a
        gate-only node still floats -- drain/source/bulk conduct, gate does not.
        Every other element (R/L/sources/diode/BJT/subckt/behavioral) anchors all
        of its nodes. Node-topology only, no per-part tables.
        """
        names = [str(n) for n in (getattr(element, "node_names", None) or [])]
        tname = type(element).__name__
        if tname in self._DC_OPEN_ELEMENT_TYPES:
            return []
        if tname == "Mosfet" and len(names) >= 3:
            # PySpice Mosfet node order is d g s [b]; index 1 (gate) is isolated.
            return [names[0]] + names[2:]
        return names

    def _union_raw_mosfet_nodes(self, all_nodes, union, find) -> None:
        """Union drain/source/bulk (NOT gate) of any raw-spice ``M`` line.

        A VDMOS is emitted via ``raw_spice`` (PySpice's ``M`` forces 4 nodes), so it
        is not a structured element -- account for its DC path here so its
        drain/source aren't mistaken for floating. Best-effort; a missed line only
        risks a harmless extra 1G tie, never a wrong answer."""
        raw = getattr(self.spice_circuit, "raw_spice", "") or ""
        for line in str(raw).splitlines():
            s = line.strip()
            if not s or s[0] in "*.":
                continue
            toks = s.split()
            if toks[0][:1].upper() != "M" or len(toks) < 5:
                continue
            nodes = toks[1:-1]  # drop the ref and the trailing model name
            # index 1 is the gate (isolated); union drain/source/bulk together
            channel = [n for i, n in enumerate(nodes) if i != 1]
            for n in nodes:
                all_nodes.add(n)
                find(n)
            for n in channel[1:]:
                union(channel[0], n)

    def _tie_floating_nodes(self) -> None:
        """Tie every DC-floating node to ground with a 1 G bleed (F4).

        Excluding a part (``Sim.Enable=0``) -- or any emission path -- can leave a
        node with no DC path to ground: its only connections are capacitors /
        current sources (open at DC), possibly across a chain of resistors (a
        stranded R-C compensation island is the common case when the controller IC
        is excluded). The operating point then has no equation pinning those node
        voltages, so ngspice reports ``singular matrix: check node <n>`` and limps
        through gmin/source stepping -- or fails outright on a less patient circuit.

        Connectivity is computed on the FINAL device graph with a union-find over
        nodes joined by DC-conducting elements (everything but capacitors, current
        sources, and a MOSFET gate). Any node not in ground's component is tied to
        ground with a resistor high enough to be electrically invisible (1 G) yet
        enough to anchor the op-point, warning once per node (naming the excluded
        part(s) that stranded it, when known).

        Byte-identical to before when nothing floats (a fully connected circuit
        gets no ties). Kill switch: ``SKIDL_SIM_TIE_FLOATING=0``.
        """
        if not self._tie_floating_enabled() or self.spice_circuit is None:
            return
        gnd = str(self.spice_circuit.gnd)
        try:
            elements = list(self.spice_circuit.elements)
        except Exception:  # pragma: no cover - PySpice internals
            return

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a, b):
            parent[find(a)] = find(b)

        all_nodes = {gnd}
        find(gnd)
        for el in elements:
            names = [str(n) for n in (getattr(el, "node_names", None) or [])]
            for n in names:
                all_nodes.add(n)
                find(n)
            # Nodes this element gives a DC path between are mutually connected;
            # a capacitor/current source yields none, a MOSFET omits its gate.
            conducting = self._conducting_nodes(el)
            for n in conducting[1:]:
                union(conducting[0], n)
        self._union_raw_mosfet_nodes(all_nodes, union, find)

        groot = find(gnd)
        floating = sorted(n for n in all_nodes if n != gnd and find(n) != groot)
        for node in floating:
            safe = re.sub(r"[^A-Za-z0-9]", "_", node) or "n"
            try:
                self.spice_circuit.R(f"float_tie_{safe}", node, gnd, 1e9)
            except Exception as exc:  # pragma: no cover - duplicate/name edge cases
                logger.debug(f"could not tie floating node '{node}': {exc}")
                continue
            culprits = sorted(self._excluded_node_refs.get(node, ()))
            cause = (
                f"floated by excluded {', '.join(culprits)}"
                if culprits
                else "has no DC path to ground"
            )
            logger.warning(
                f"sim node '{node}' {cause}; tied to ground with 1G to anchor the "
                f"op-point (electrically invisible). Set SKIDL_SIM_TIE_FLOATING=0 "
                f"to disable this conditioning."
            )

    def _extract_voltage_from_net_name(self, net_name: str) -> Optional[float]:
        """Extract voltage value from net name (e.g., '+5V' -> 5.0)."""
        upper = net_name.upper()

        # KiCad "decimal-in-V" notation where the V is the decimal point:
        # 3V3 -> 3.3, 1V8 -> 1.8. Checked first so "5V" (no trailing digit)
        # still falls through to the plain patterns below (-> 5.0, not 5.x).
        m = re.search(r"\+?([0-9]+)V([0-9]+)\b", upper)
        if m:
            try:
                return float(f"{m.group(1)}.{m.group(2)}")
            except ValueError:
                pass

        # Look for voltage patterns
        patterns = [
            r"\+?([0-9.]+)V",  # +5V, 3.3V, etc.
            r"VCC_?([0-9.]+)",  # VCC_5, VCC5, etc.
            r"VDD_?([0-9.]+)",  # VDD_3, VDD3, etc.
        ]

        for pattern in patterns:
            match = re.search(pattern, upper)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _heuristic_source_voltage(self, net_name: str) -> Optional[float]:
        """Voltage the net-name heuristic would assign to this net, or None.

        Single source of truth shared by ``_add_power_sources`` (which actually
        injects the supply) and ``validate`` (which must know whether a
        single-connection net will nonetheless be driven). A bare ``VCC`` with no
        embedded number is *not* driven (no voltage to assign); ``VIN``/``VSUPPLY``
        default to 5 V.

        Rail keywords are matched as **whole tokens**, not substrings, so an
        intermediate net like ``VINT_RAW`` (tokens ``{VINT, RAW}``) is NOT treated
        as a ``VIN`` rail -- that substring trap injected a phantom 5 V supply and
        clamped a 32.5 V boost output (bug #13). Known accepted limitation: a name
        like ``MAIN_VIN_SENSE`` still matches on the whole token ``VIN`` -- token
        matching cannot tell it is a sense line. Two backstops make an unintended
        match harmless: ``_add_power_sources`` skips the injection entirely when
        the net is already driven by an ``OUTPUT``/``PWROUT`` pin (E2E D1), and it
        logs any injection it does perform at WARNING (a circuit-mutating action).
        """
        upper = net_name.upper()
        tokens = {t for t in re.split(r"[^A-Z0-9+]+", upper) if t}
        if tokens & {"VCC", "VDD", "V+"}:
            return self._extract_voltage_from_net_name(upper)
        if tokens & {"VIN", "VSUPPLY"}:
            return self._extract_voltage_from_net_name(upper) or 5.0
        if any(re.fullmatch(r"\+?[0-9]+V[0-9]*", t) for t in tokens):
            return self._extract_voltage_from_net_name(upper)
        return None

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _attr(obj, name, default=None):
        """Read an attribute from either a live object or a dict-shaped one."""
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _iter_components(self):
        comps = getattr(self.circuit, "components", None)
        if comps is None:
            return []
        return list(comps.values()) if hasattr(comps, "values") else list(comps)

    def _iter_nets(self):
        nets = getattr(self.circuit, "nets", None)
        if nets is None:
            return []
        return list(nets.values()) if hasattr(nets, "values") else list(nets)

    @staticmethod
    def _is_ground_name(name: str) -> bool:
        return str(name).upper() in ("GND", "GROUND", "VSS", "0")

    @staticmethod
    def _net_pin_count(net) -> Optional[int]:
        """Number of pins on a net, or None if the net can't report it."""
        pins = getattr(net, "pins", None)
        if pins is None:
            return None
        try:
            return len(pins)
        except TypeError:
            return None

    def _connected_pin_count(self, component) -> Optional[int]:
        """Number of a component's pins attached to a net, or None if unknown."""
        pin_map = getattr(component, "_pins", None)
        if not isinstance(pin_map, dict):
            return None
        return sum(
            1 for pin in pin_map.values() if getattr(pin, "net", None) is not None
        )

    def _net_live_pin_count(self, net, excluded_refs) -> Optional[int]:
        """Pins on a net whose owning component isn't excluded, or None if unknown.

        A ``Sim.Enable=0`` part's pins don't hold a net up, so a net left with
        fewer than two *live* pins is still floating.
        """
        pins = getattr(net, "pins", None)
        if pins is None:
            return None
        try:
            count = 0
            for pin in pins:
                comp = getattr(pin, "_component", None)
                cref = getattr(comp, "ref", None)
                if cref is not None and cref in excluded_refs:
                    continue
                count += 1
            return count
        except TypeError:
            return None

    def _all_driven_net_names(self) -> set:
        """Nets that will carry a source: explicit source components + heuristic rails.

        An excluded (``Sim.Enable=0``) source doesn't drive anything, so it isn't
        counted as excitation.
        """
        driven = set()
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) in (
                "voltage_source",
                "current_source",
            ):
                driven |= self._component_net_names(component)
        for net in self._iter_nets():
            name = self._attr(net, "name", None) or str(net)
            if self._heuristic_source_voltage(name) is not None:
                driven.add(name)
        return driven

    def validate(self) -> None:
        """Check the circuit is safe to simulate; raise with every problem found.

        Catches the failure modes that otherwise produce a wrong-but-"successful"
        simulation or an opaque ngspice error:

        * a component whose symbol has no SPICE mapping (was silently skipped);
        * a net with a single connection that no source drives (floating node ->
          singular matrix);
        * no source at all (nothing to excite the circuit);
        * an op-amp with fewer than three connected pins (needs in+, in-, out).

        Checks that need data a dict/JSON-shaped circuit can't provide (live pin
        maps, net pin counts) are skipped for those inputs rather than raising
        false positives.
        """
        problems = []

        # Parts opted out via Sim.Enable=0 are ignored by every check below and
        # don't count toward net connectivity.
        excluded_refs = set()
        for component in self._iter_components():
            if self._sim_excluded(component):
                excluded_refs.add(self._attr(component, "ref", None) or "?")

        # 1. Every component must map to a known SPICE primitive (Sim.Device wins).
        #    Parts carrying an external model (Sim.Library) are exempt -- the
        #    external .subckt/.model defines their behaviour (checked in #6).
        for component in self._iter_components():
            if self._sim_excluded(component) or self._has_external_lib(component):
                continue
            symbol = self._attr(component, "symbol", "")
            ref = self._attr(component, "ref", None) or "?"
            if self._kind(component) is None:
                device = self._sim_props(component).get("device", None)
                if device:
                    problems.append(
                        f"{ref}: Sim.Device '{device}' is not a supported device "
                        f"(known: {', '.join(sorted(self._SIM_DEVICE_KINDS))})"
                    )
                else:
                    problems.append(
                        f"{ref}: unrecognized symbol '{symbol}' has no SPICE mapping "
                        f"(would be silently skipped)"
                    )

        driven = self._all_driven_net_names()

        # 2. There must be some excitation.
        if not driven:
            problems.append(
                "no voltage or current source found (declare a "
                "Simulation_SPICE:VDC/VAC, or use a rail-named net); simulation "
                "would have no excitation"
            )

        # 3. No floating nodes: every net needs >=2 connections, ground, or a source.
        for net in self._iter_nets():
            name = self._attr(net, "name", None) or str(net)
            if self._is_ground_name(name) or name in driven:
                continue
            if excluded_refs:
                # A net whose *only* pins belong to Sim.Enable=0 parts never enters
                # the SPICE netlist, so it isn't floating -- it's absent. Drop it
                # instead of flagging (report F6): otherwise a bias/output rail on a
                # sim-disabled sensor/connector aborts the whole simulation.
                live = self._net_live_pin_count(net, excluded_refs)
                total = self._net_pin_count(net)
                if live is not None and total is not None and live == 0 and total > 0:
                    logger.debug(
                        f"net '{name}' is private to Sim.Enable=0 part(s) "
                        f"({total} pin(s), all excluded); dropped from validation "
                        f"(not floating)"
                    )
                    continue
                count = live
            else:
                count = self._net_pin_count(net)
            if count is None:
                continue  # can't tell (dict/JSON net) -> don't block
            if count < 2:
                problems.append(
                    f"net '{name}' has {count} connection(s); needs >=2 or a source "
                    f"(floating node)"
                )

        # 4. Op-amps need at least their three signal pins connected.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "opamp":
                continue
            count = self._connected_pin_count(component)
            if count is None:
                continue
            if count < 3:
                ref = self._attr(component, "ref", None) or "?"
                problems.append(
                    f"{ref}: op-amp needs >=3 connected pins (in+, in-, out), "
                    f"found {count}"
                )

        # 4b. LDOs need their three terminals (in, out, gnd) connected, and a VOUT
        #     that resolves -- a regulator's output voltage cannot be guessed.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "ldo":
                continue
            ref = self._attr(component, "ref", None) or "?"
            count = self._connected_pin_count(component)
            if count is not None and count < 3:
                problems.append(
                    f"{ref}: LDO needs >=3 connected pins (in, out, gnd), "
                    f"found {count}"
                )
            params, _tier = self._ldo_params(
                component, self._attr(component, "value", None)
            )
            if params is None:
                problems.append(
                    f"{ref}: LDO has no resolvable output voltage; set "
                    f'Sim.Params="vout=3.3 vdrop=0.3" (or name a ModelLibrary '
                    f"entry with a VOUT param via value=)"
                )

        # 4c. Switching regulators: explicit topology + resolvable terminals/params.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            kind = self._kind(component)
            ref = self._attr(component, "ref", None) or "?"
            if kind == "switcher_unknown":
                problems.append(
                    f"{ref}: switching-regulator topology is ambiguous; set "
                    f"Sim.Device=BUCK or Sim.Device=BOOST"
                )
                continue
            if kind not in ("buck", "boost", "flyback"):
                continue
            # 4c-avgcm. Averaged peak-current-mode LOOP model (MODE=avg CMODE=peak,
            #           buck/boost only): checks FB/VC/VOUT terminals + VREF/FSW/duty
            #           instead of the power-stage SW/VIN/GND, since it resolves its
            #           own terminals and never emits a switch node.
            if (
                kind in ("buck", "boost")
                and self._switcher_mode(component) == "avg"
                and self._switcher_cmode(component) == "peak"
            ):
                if (
                    getattr(component, "_pins", None) is not None
                    and self._currentmode_terminals(component) is None
                ):
                    problems.append(
                        f"{ref}: {kind} MODE=avg CMODE=peak needs connected FB, VC "
                        f"(compensation) and VOUT pins (resolved by pin name)"
                    )
                if self._averaged_current_mode_params(component, kind) is None:
                    problems.append(
                        f"{ref}: {kind} MODE=avg CMODE=peak needs Sim.Params VREF "
                        f"(may be negative), FSW and a resolvable duty (D, or "
                        f'VOUT+VIN), e.g. Sim.Params="fsw=300k vout=24 vin=12 '
                        f'vref=1.6 mode=avg cmode=peak"'
                    )
                continue
            if (
                getattr(component, "_pins", None) is not None
                and self._switcher_terminals(component) is None
            ):
                problems.append(
                    f"{ref}: {kind} needs connected SW, VIN and GND pins "
                    f"(resolved by pin name)"
                )
            if (
                self._switcher_params(
                    component, self._attr(component, "value", None), kind
                )
                is None
            ):
                if kind == "flyback":
                    problems.append(
                        f"{ref}: flyback needs Sim.Params with VOUT, FSW and N "
                        f"(the transformer turns ratio Ns/Np), e.g. "
                        f'Sim.Params="fsw=100k vout=5 n=0.5"'
                    )
                else:
                    problems.append(
                        f"{ref}: {kind} needs Sim.Params with VOUT and FSW, e.g. "
                        f'Sim.Params="fsw=500k vout=3.3"'
                    )

        # 4c-hb. Half-bridge / resonant switch stage: SW/VIN/GND + FSW required,
        #        deadtime must leave conduction time.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "halfbridge":
                continue
            ref = self._attr(component, "ref", None) or "?"
            if (
                getattr(component, "_pins", None) is not None
                and self._switcher_terminals(component) is None
            ):
                problems.append(
                    f"{ref}: halfbridge needs connected SW, VIN and GND pins "
                    f"(resolved by pin name)"
                )
            raw = self._parse_sim_params(self._sim_props(component).get("params"))
            fsw = self._parse_si_number(raw["FSW"]) if "FSW" in raw else None
            if not fsw or fsw <= 0:
                problems.append(
                    f"{ref}: halfbridge needs Sim.Params with FSW, e.g. "
                    f'Sim.Params="fsw=100k"'
                )
            else:
                dt = self._parse_si_number(raw["DT"]) if "DT" in raw else 100e-9
                if dt is not None and dt >= 0.5 / fsw:
                    problems.append(
                        f"{ref}: halfbridge deadtime DT ({self._fmt_num(dt)}s) must "
                        f"be < 1/(2*FSW) = {0.5 / fsw:.3g}s"
                    )

        # 4c-bb4. 4-switch buck-boost: all five terminals + FSW + resolvable
        #         duties; deadtime must leave a conduction window on each leg.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "buckboost4":
                continue
            ref = self._attr(component, "ref", None) or "?"
            if (
                getattr(component, "_pins", None) is not None
                and self._multiswitch_terminals(component, "buckboost4") is None
            ):
                problems.append(
                    f"{ref}: buckboost4 needs connected VIN, SW (buck node), SW2 "
                    f"(boost node), VOUT and GND pins (resolved by pin name)"
                )
            if self._buckboost4_params(component) is None:
                problems.append(
                    f"{ref}: buckboost4 needs Sim.Params with FSW and duties, e.g. "
                    f'Sim.Params="fsw=500k dbuck=0.5 dboost=0" (or a VOUT+VIN target)'
                )

        # 4c-ibb. Inverting buck-boost: VIN/SW/VOUT/GND + FSW + a resolvable duty.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "invbuckboost":
                continue
            ref = self._attr(component, "ref", None) or "?"
            if (
                getattr(component, "_pins", None) is not None
                and self._multiswitch_terminals(component, "invbuckboost") is None
            ):
                problems.append(
                    f"{ref}: invbuckboost needs connected VIN, SW (switch node), "
                    f"VOUT (negative output) and GND pins (resolved by pin name)"
                )
            if self._invbuckboost_params(component) is None:
                problems.append(
                    f"{ref}: invbuckboost needs Sim.Params with FSW and a duty, e.g. "
                    f'Sim.Params="fsw=500k d=0.5" (or a VOUT+VIN target)'
                )

        # 4c-sepic. Bidirectional SEPIC: VIN/SWA/SWB/VOUT/GND + FSW + a resolvable
        #           duty; the coupling cap Cs must be a real external part between
        #           the two switch nodes (warn if A and B collapse to one net).
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "sepic":
                continue
            ref = self._attr(component, "ref", None) or "?"
            if getattr(component, "_pins", None) is not None:
                term = self._multiswitch_terminals(component, "sepic")
                if term is None:
                    problems.append(
                        f"{ref}: sepic needs connected VIN, SW (node A), SW2 "
                        f"(node B), VOUT and GND pins (resolved by pin name)"
                    )
                elif term["swa"] == term["swb"]:
                    problems.append(
                        f"{ref}: sepic nodes A (SW) and B (SW2) are the same net -- "
                        f"the coupling cap Cs must be a real external part between them"
                    )
            if self._sepic_params(component) is None:
                problems.append(
                    f"{ref}: sepic needs Sim.Params with FSW and a duty, e.g. "
                    f'Sim.Params="fsw=500k d=0.5" (or a VOUT+VIN target)'
                )

        # 4c-cuk. Inverting Cuk: VIN/SWA/SWB/VOUT/GND + FSW + a resolvable duty; the
        #         series coupling cap Cs must be a real external part between the two
        #         switch nodes (warn if A and B collapse to one net). VOUT is the
        #         negative output behind the real output inductor L2, not a switch node.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "cuk":
                continue
            ref = self._attr(component, "ref", None) or "?"
            if getattr(component, "_pins", None) is not None:
                term = self._multiswitch_terminals(component, "cuk")
                if term is None:
                    problems.append(
                        f"{ref}: cuk needs connected VIN, SW (node A), SW2 "
                        f"(node B), VOUT (negative output) and GND pins (resolved by pin name)"
                    )
                elif term["swa"] == term["swb"]:
                    problems.append(
                        f"{ref}: cuk nodes A (SW) and B (SW2) are the same net -- "
                        f"the coupling cap Cs must be a real external part between them"
                    )
            if self._cuk_params(component) is None:
                problems.append(
                    f"{ref}: cuk needs Sim.Params with FSW and a duty, e.g. "
                    f'Sim.Params="fsw=500k d=0.5" (or a VOUT+VIN target)'
                )

        # 4c-cmc. Behavioral closed-loop peak-current-mode controller (Stage 29.1):
        #         all six terminals (VIN/SW/VOUT/FB/VC/GND) + VREF/FSW; buck only.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "cmcontroller":
                continue
            ref = self._attr(component, "ref", None) or "?"
            if (
                getattr(component, "_pins", None) is not None
                and self._cmcontroller_terminals(component) is None
            ):
                problems.append(
                    f"{ref}: CMCONTROLLER needs connected VIN, SW, VOUT, FB, VC "
                    f"(compensation) and GND pins (resolved by pin name)"
                )
            cp = self._cmcontroller_params(component)
            if cp is None:
                problems.append(
                    f"{ref}: CMCONTROLLER needs Sim.Params VREF (may be negative) and "
                    f'FSW, e.g. Sim.Params="topology=buck fsw=500k vout=3.3 vin=12 '
                    f'vref=0.8"'
                )
            elif cp["TOPOLOGY"] != "buck":
                problems.append(
                    f"{ref}: CMCONTROLLER topology={cp['TOPOLOGY']} is not wired in "
                    f"Stage 29.1 (buck only; boost/SEPIC/Ćuk/flyback are Stage 29.4)"
                )

        # 4d. Transformers: all four winding ends connected + resolvable params.
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            if self._kind(component) != "transformer":
                continue
            ref = self._attr(component, "ref", None) or "?"
            if (
                getattr(component, "_pins", None) is not None
                and self._transformer_terminals(component) is None
            ):
                missing = self._transformer_missing_pins(component)
                where = (
                    f" (unconnected: {', '.join(missing)})" if missing else ""
                )
                problems.append(
                    f"{ref}: transformer winding ends must all be connected -- "
                    f"AA/AB + each secondary pair (SA/SB, SC/SD, ...), or SA/SC/SB "
                    f"for a center-tapped 1P_SS{where}"
                )
            params = self._transformer_params(component)
            if params is None:
                problems.append(
                    f"{ref}: transformer needs Sim.Params with LP and a turns "
                    f"ratio N (or LS) per winding, e.g. Sim.Params=\"lp=100u n=0.5\" "
                    f'(two secondaries: "lp=25u n=0.5 n2=0.1")'
                )
            elif params["K"] is None:
                problems.append(f"{ref}: transformer coupling k must be in (0, 1]")

        # 4d. Behavioral logic primitives need their terminals resolvable by pin
        #     name (skipped for dict/JSON circuits that carry no live pin names).
        for component in self._iter_components():
            if self._sim_excluded(component):
                continue
            kind = self._kind(component)
            if kind not in ("dff", "tff", "dlatch", "gate", "trigsw"):
                continue
            ref = self._attr(component, "ref", None) or "?"
            nodes = self._logic_pin_nodes(component)
            if nodes is None:
                continue  # no live pin map -> can't check (lenient path skips it)
            if kind == "trigsw":
                p = self._first_named(nodes, self._TRIGSW_P_NAMES)
                n_ = self._first_named(nodes, self._TRIGSW_N_NAMES)
                if p is None or n_ is None:
                    problems.append(
                        f"{ref}: TRIGSW needs two power terminals -- P/A/anode/C(+) "
                        f"and N/K/cathode/E(-) -- resolved by pin name or Sim.Pins"
                    )
            elif kind == "gate":
                op = str(self._sim_props(component).get("device", "")).strip().upper()
                out = self._first_named(nodes, self._GATE_OUT_NAMES)
                ins = [
                    n for nm, n in nodes.items()
                    if nm not in (self._LOGIC_GND_NAMES | self._GATE_OUT_NAMES)
                ]
                need = 1 if op in ("NOT", "INV", "BUF") else 2
                if out is None or len(ins) < need:
                    problems.append(
                        f"{ref}: {op} gate needs an output (Y/OUT/Q) and >={need} "
                        f"input pin(s), resolved by pin name"
                    )
            elif kind == "dlatch":
                if (
                    self._first_named(nodes, self._LOGIC_EN_NAMES | self._LOGIC_CLK_NAMES)
                    is None
                    or self._first_named(nodes, self._LOGIC_D_NAMES) is None
                    or (
                        self._first_named(nodes, self._LOGIC_Q_NAMES) is None
                        and self._first_named(nodes, self._LOGIC_QN_NAMES) is None
                    )
                ):
                    problems.append(
                        f"{ref}: D latch needs connected D, EN and Q (or QN) pins"
                    )
            else:  # dff / tff
                clk = self._first_named(nodes, self._LOGIC_CLK_NAMES)
                has_q = (
                    self._first_named(nodes, self._LOGIC_Q_NAMES) is not None
                    or self._first_named(nodes, self._LOGIC_QN_NAMES) is not None
                )
                need_d = kind == "dff"
                d = self._first_named(nodes, self._LOGIC_D_NAMES)
                if clk is None or not has_q or (need_d and d is None):
                    terms = "D, CLK and Q (or QN)" if need_d else "CLK and Q (or QN)"
                    problems.append(
                        f"{ref}: {kind.upper()} needs connected {terms} pins"
                    )

        # 5. Every diode/BJT/MOSFET must reference a model that resolves to a
        #    built-in generic (otherwise ngspice errors on an undefined model).
        for component in self._iter_components():
            if self._sim_excluded(component) or self._has_external_lib(component):
                continue
            if self._store_lib_for(component):
                continue  # resolved from the local MPN store (tier vendor_lib)
            if self._library_index_hit(component) is not None:
                continue  # resolved from the external library index (tier vendor_lib)
            model = self._device_model_name(component)
            if model is None:
                continue
            kind = self._kind(component)
            spec, _tier, _resolved = self._lookup_model_spec(model, kind)
            if spec is not None:
                continue
            # An unresolved base carrying Sim.Params degrades to the kind's generic
            # + overrides (see _resolve_device_model), so it is simulatable -- not an
            # error. With no overrides it stays a hard error below.
            overrides = self._parse_sim_params(self._sim_props(component).get("params"))
            if overrides and kind in self._KIND_GENERIC:
                continue
            ref = self._attr(component, "ref", None) or "?"
            # Distinguish a device-type mismatch from an entirely unknown name.
            try:
                from .models import get_model_library

                entry = get_model_library().get_model(model)
            except Exception:  # pragma: no cover
                entry = None
            if entry is not None:
                problems.append(
                    f"{ref}: model '{model}' is a {entry.model_type} but device is "
                    f"{kind} (wrong model type)"
                )
            else:
                known = ", ".join(sorted(self.GENERIC_MODELS))
                problems.append(
                    f"{ref}: references SPICE model '{model}' with no .model card "
                    f"(not in the model library; known generics: {known})"
                )

        # 6. External vendor models (Sim.Library) must be coherent and locatable.
        for component in self._iter_components():
            if self._sim_excluded(component) or not self._has_external_lib(component):
                continue
            sim = self._sim_props(component)
            ref = self._attr(component, "ref", None) or "?"
            name = sim.get("name")
            if not name:
                problems.append(
                    f"{ref}: Sim.Library set without Sim.Name (which subckt/model "
                    f"to use is ambiguous)"
                )
                continue
            path = self._resolve_lib_path(sim.get("library"))
            if not path or not os.path.exists(path):
                # A missing hardcoded path that the corpus can auto-resolve is a
                # convert-time WARNING (see _add_component), not a hard failure
                # (DPSG A4): only flag it when there is no corpus fallback.
                if self._library_index_hit(component) is not None:
                    continue
                problems.append(
                    f"{ref}: Sim.Library file not found: {sim.get('library')}"
                )
                continue
            kind_in_file, _nodes, _dtype = self._scan_lib(path, name)
            if kind_in_file is None:
                problems.append(
                    f"{ref}: Sim.Name '{name}' is neither a .subckt nor a .model in "
                    f"{os.path.basename(str(path))}"
                )
                continue
            # The named model may resolve to a class ngspice cannot run (XSPICE
            # digital, PSpice U-device, encrypted). Flag it by class here, naming
            # the reason, instead of dying on a dead node / opaque run later (A1).
            mc = self._model_simulatability(path, name)
            if mc is not None and mc.simulatable == "no":
                problems.append(
                    f"{ref}: Sim.Name '{name}' resolves to a non-simulatable model "
                    f"class ({mc.dialect}) -- {mc.reason}"
                )

        # 6b. A corpus/index-resolved model may likewise be a non-simulatable
        #     class. Classify auto-resolved subckt hits (the digital corpus lands
        #     here when Sim.Prefer=library or Sim.Pins name the subckt's nodes).
        for component in self._iter_components():
            if self._sim_excluded(component) or self._has_external_lib(component):
                continue
            hit = self._library_index_hit(component)
            if hit is None or not getattr(hit, "path", None):
                continue
            mc = self._model_simulatability(hit.path, getattr(hit, "name", None))
            if mc is not None and mc.simulatable == "no":
                ref = self._attr(component, "ref", None) or "?"
                problems.append(
                    f"{ref}: auto-resolved model '{getattr(hit, 'name', '?')}' is a "
                    f"non-simulatable class ({mc.dialect}) -- {mc.reason}"
                )

        # 7. Sim.Compat must be unambiguous: one ngspice dialect per simulation.
        compat_values = self._distinct_compat_values()
        if len(compat_values) > 1:
            problems.append(
                f"conflicting Sim.Compat values across components: "
                f"{', '.join(compat_values)} (one dialect per simulation)"
            )

        if problems:
            raise SimulationValidationError(problems)
