# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""Adapter: skidl ``Circuit`` -> the circuit-synth ``_FlatCircuit`` view.

The circuit-synth ``SpiceConverter`` was deliberately made frontend-agnostic
(Stage-16 loop-boundary contract): it reads ONLY ``.components`` (dict
ref->component), ``.nets`` (dict name->net) and ``.name`` off its circuit, and
touches components/nets only through duck-typed attributes:

  * component: ``.ref``, ``.value``, ``.symbol``, ``._pins`` (dict of pin
    objects), ``._extra_fields`` (dict, for ``Sim.*`` fields + waveform params);
  * pin: ``.net`` (net object or None), ``.num``, ``.name``;
  * net: ``.name``.

This module reconstructs exactly that shape from a skidl ``Circuit`` so the whole
converter/validator/macromodel/measurement stack works unchanged. skidl's
hierarchy is already flat at the ``circuit.parts`` / ``circuit.nets`` level, so
the view is presented flat and the converter's ``_flatten`` is a no-op (the view
carries no ``_subcircuits``).
"""

import os


def _lib_nickname(part):
    """Best-effort KiCad library nickname for a skidl part (e.g. ``"Device"``).

    The circuit-synth converter classifies a component by its ``symbol`` string
    in ``"{lib}:{name}"`` form (``"Device:R"``, ``"Amplifier_Operational:LM358"``,
    ...), so we need the short library name. skidl stores the source library on
    ``part.lib`` (a ``SchLib`` whose ``.filename`` is the library file); fall
    back to any explicit ``lib_nickname``/``lib_id`` the part carries.
    """
    for attr in ("lib_nickname", "libname"):
        val = getattr(part, attr, None)
        if val:
            return str(val)
    lib = getattr(part, "lib", None)
    filename = getattr(lib, "filename", None) if lib is not None else None
    if filename:
        base = os.path.basename(str(filename))
        return os.path.splitext(base)[0]
    # Last resort: a lib_id like "Device:R" already carries the nickname.
    lib_id = getattr(part, "lib_id", None)
    if lib_id and ":" in str(lib_id):
        return str(lib_id).split(":", 1)[0]
    return ""


def _part_symbol(part):
    """Reconstruct the KiCad ``lib_id`` (``"{lib}:{name}"``) for a skidl part."""
    lib_id = getattr(part, "lib_id", None)
    if lib_id and ":" in str(lib_id):
        return str(lib_id)
    nickname = _lib_nickname(part)
    name = getattr(part, "name", None) or ""
    return f"{nickname}:{name}" if nickname else str(name)


def _pin_func_str(pin):
    """Lowercased electrical-type string for a skidl pin (``"output"``, ``"input"``,
    ``"power_in"``, ...), or ``""`` if unknown.

    skidl ``Pin.func`` is a ``pin_types`` enum whose ``.name`` is ``OUTPUT`` /
    ``INPUT`` / ``PWRIN`` / ``PASSIVE`` / ... The converter's op-amp terminal
    resolver (:meth:`SpiceConverter._opamp_terminals`) substring-matches
    ``"output"`` / ``"input"`` on this string to classify signal pins, so it MUST
    be threaded through -- without it every op-amp falls back to the positional
    guess and scrambles its (out, in+, in-) nodes (singular matrix, hard fail).
    """
    func = getattr(pin, "func", None)
    if func is None:
        return ""
    return str(getattr(func, "name", func)).lower()


def _pin_net(pin):
    """Return the skidl net a pin is on, or None if unconnected.

    skidl exposes ``pin.net`` (single net) and ``pin.nets`` (list). Prefer the
    singular; fall back to the first of the list. Unconnected pins have no net.
    """
    net = getattr(pin, "net", None)
    if net is not None:
        # ``pin.net`` can be a Net or a 1-list depending on skidl internals.
        if isinstance(net, (list, tuple)):
            return net[0] if net else None
        return net
    nets = getattr(pin, "nets", None)
    if nets:
        return nets[0]
    return None


# Waveform-param kwargs the converter's source specs read via ``_pick``
# (SpiceConverter._voltage_source_spec / _current_source_spec). Bare kwargs with
# these names on a ``Simulation_SPICE`` source part must reach ``_extra_fields``
# or ngspice gets symbol defaults (E2E finding B2: a kwargs VPULSE emitted
# ``pulse(0 vpulse ...)`` and failed to parse). Kept in sync with converter.py
# (the union of every key list passed to ``_pick`` across V*/I* specs).
_WAVEFORM_KWARGS = frozenset({
    # VPULSE / IPULSE
    "v1", "initial", "low", "v2", "pulsed", "high", "amplitude",
    "td", "delay", "tr", "rise", "tf", "fall", "pw", "width", "per", "period",
    "i1", "i2",
    # VSIN / ISIN
    "offset", "dc", "voffset", "ioffset", "ampl", "amp",
    "frequency", "freq", "f", "theta", "damping", "ac", "ac_mag", "acmag",
    # VPWL / IPWL
    "points",
})


def _is_spice_source(part) -> bool:
    """True if ``part`` is a ``Simulation_SPICE`` library part (a stimulus source).

    Scoping the bare-kwargs waveform sweep to these parts means arbitrary attrs
    on ordinary parts (a stray ``f=`` on a resistor) are never collected.
    """
    if _lib_nickname(part).upper() == "SIMULATION_SPICE":
        return True
    return _part_symbol(part).upper().startswith("SIMULATION_SPICE:")


def _extra_fields(part):
    """Collect the ``Sim.*`` fields (and any ``fields`` dict) off a skidl part.

    skidl ``Part(...)`` accepts arbitrary kwargs as attributes (that is how the
    ``cluster=`` render hint already flows through), so simulation intent rides on
    ``Sim_Device`` / ``Sim_Params`` / ``Sim_Gbw`` / ``Sim_Library`` / ``Sim_Name``
    / ``Sim_Pins`` / ``Sim_Compat`` / ``Sim_Enable`` attributes. The converter's
    ``_sim_props`` lowercases the key and strips a ``sim.``/``sim_`` prefix, so
    the underscore spelling is picked up unchanged. A ``fields`` dict (as produced
    by ``netlist_to_skidl``) is merged in too.

    Additionally, on a ``Simulation_SPICE`` source part, bare waveform kwargs
    (``v1=``/``v2=``/``tr=``/``frequency=`` ...) are collected from the instance
    dict so the SKILL-documented ``Part("Simulation_SPICE","VPULSE", v1=..., ...)``
    form actually reaches the source spec (E2E finding B2). ``Sim_Params="..."``
    still works and is the base; explicit kwargs win over it (as "explicit fields"
    in ``_source_params``).
    """
    extra = {}

    fields = getattr(part, "fields", None)
    if isinstance(fields, dict):
        extra.update(fields)

    is_source = _is_spice_source(part)

    # Instance attributes whose name looks like a Sim.* field, plus -- for a
    # Simulation_SPICE source only -- allowlisted bare waveform kwargs. Use the
    # instance __dict__ so we do not sweep in class-level machinery.
    for key, val in list(vars(part).items()):
        low = str(key).lower()
        if low.startswith("sim_") or low.startswith("sim."):
            extra[key] = val
        elif is_source and low in _WAVEFORM_KWARGS:
            extra[key] = val

    return extra


def _symbol_data_for(part):
    """Build the multi-unit symbol map the converter's ``_amp_units`` reads.

    circuit-synth reads this from its ``SymbolLibCache``; skidl already has it on
    the parsed Part (each Pin carries ``.unit`` and a ``pin_types`` ``.func``), so
    the adapter supplies it directly -- the one place the sim layer needs
    symbol-level data, not just connectivity. Shape:
    ``{"unit_count": N, "pins": [{"number","unit","function","name"}, ...]}``.
    Returns None for a single-unit part (converter then keeps the whole-component
    path).
    """
    pins = list(getattr(part, "pins", []) or [])
    units = {getattr(p, "unit", None) for p in pins}
    units.discard(None)
    if len(units) <= 1:
        return None
    out_pins = []
    for p in pins:
        func = getattr(p, "func", None)
        # skidl pin.func is a pin_types enum; its .name ("OUTPUT"/"INPUT"/...) is
        # what the converter substring-matches ("output"/"input").
        func_str = getattr(func, "name", str(func) if func is not None else "").lower()
        out_pins.append(
            {
                "number": getattr(p, "num", None),
                "unit": getattr(p, "unit", 0),
                "function": func_str,
                "name": getattr(p, "name", "") or "",
            }
        )
    return {"unit_count": len(units), "pins": out_pins}


class AdaptedNet:
    """Minimal net view: the converter reads only ``.name``."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


class AdaptedPin:
    """Minimal pin view: ``.net`` (AdaptedNet|None), ``.num``, ``.name``, ``.func``.

    ``.func`` is the lowercased electrical-type string (``"output"``/``"input"``/
    ``"power_in"``/...) the converter reads to resolve op-amp signal terminals.
    """

    __slots__ = ("net", "num", "name", "func")

    def __init__(self, net, num, name, func=""):
        self.net = net
        self.num = num
        self.name = name
        self.func = func


class AdaptedComponent:
    """Minimal component view matching the converter's duck-typed contract."""

    __slots__ = ("ref", "value", "symbol", "_pins", "_extra_fields", "_symbol_data")

    def __init__(self, ref, value, symbol, pins, extra_fields, symbol_data=None):
        self.ref = ref
        self.value = value
        self.symbol = symbol
        self._pins = pins
        self._extra_fields = extra_fields
        # Multi-unit pin->unit map for op-amp section resolution (None if single
        # unit); read by SpiceConverter._amp_units.
        self._symbol_data = symbol_data


class SkidlFlatView:
    """The three-attribute view the converter consumes.

    Deliberately carries no ``_subcircuits`` so ``SpiceConverter._flatten``
    treats it as already flat (which it is -- skidl flattens at ``.parts``).
    """

    def __init__(self, name, components, nets):
        self.name = name
        self.components = components
        self.nets = nets


def skidl_flat_view(circuit=None):
    """Build a converter-ready flat view from a skidl ``Circuit``.

    Args:
        circuit: a skidl ``Circuit``. Defaults to skidl's active
            ``default_circuit``.

    Returns:
        SkidlFlatView with ``.name``, ``.components`` (dict ref->AdaptedComponent)
        and ``.nets`` (dict name->AdaptedNet).
    """
    if circuit is None:
        import builtins

        circuit = getattr(builtins, "default_circuit", None)
        if circuit is None:
            raise ValueError("no circuit given and no active default_circuit")

    # One AdaptedNet per distinct net name (shared across pins by identity).
    nets_by_name = {}

    def net_view(skidl_net):
        if skidl_net is None:
            return None
        name = getattr(skidl_net, "name", None) or str(skidl_net)
        av = nets_by_name.get(name)
        if av is None:
            av = AdaptedNet(name)
            nets_by_name[name] = av
        return av

    components = {}
    for part in getattr(circuit, "parts", []) or []:
        ref = getattr(part, "ref", None)
        if ref is None:
            continue
        pins = {}
        for pin in getattr(part, "pins", []) or []:
            num = getattr(pin, "num", None)
            if num is None:
                continue
            pins[str(num)] = AdaptedPin(
                net=net_view(_pin_net(pin)),
                num=num,
                name=getattr(pin, "name", "") or "",
                func=_pin_func_str(pin),
            )
        components[ref] = AdaptedComponent(
            ref=ref,
            value=getattr(part, "value", None),
            symbol=_part_symbol(part),
            pins=pins,
            extra_fields=_extra_fields(part),
            symbol_data=_symbol_data_for(part),
        )

    # Register every skidl net (even ones with no pins reached above) by name.
    for net in getattr(circuit, "nets", []) or []:
        net_view(net)

    name = getattr(circuit, "name", None) or "Circuit"
    return SkidlFlatView(name, components, nets_by_name)
