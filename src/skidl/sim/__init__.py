# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""skidl.sim -- simulate a skidl-authored KiCad design through a macromodel SPICE
stack (ported from the circuit-synth simulation layer).

This subpackage is optional and self-contained: importing ``skidl`` does not
import it, and it has NO ``circuit_synth`` dependency (the SPICE stack is
vendored here). Importing ``skidl.sim`` itself only pulls in the lightweight
adapter; the converter/simulator (which need PySpice) load lazily on first use.

Typical use::

    from skidl import *
    from skidl.sim import simulate
    # ... build a circuit with Part/Net, set Sim_* attrs for macromodels ...
    sim = simulate()                 # uses the active default_circuit
    v = sim.operating_point()

The heavy lifting is an *adapter* (:func:`skidl_flat_view`) from a skidl
``Circuit`` to the frontend-agnostic read-only view the converter consumes
(``.components``/``.nets``/``.name``). The port is an adapter, not a rewrite.
"""

from .adapter import (  # noqa: F401  (lightweight, no PySpice)
    AdaptedComponent,
    AdaptedNet,
    AdaptedPin,
    SkidlFlatView,
    skidl_flat_view,
)

__all__ = [
    "skidl_flat_view",
    "SkidlFlatView",
    "AdaptedComponent",
    "AdaptedNet",
    "AdaptedPin",
    "simulate",
    "SpiceConverter",
    "CircuitSimulator",
    "SimulationResult",
    "SimulationValidationError",
]


def simulate(circuit=None, compat=None):
    """Simulate a skidl ``Circuit`` through the vendored SPICE stack.

    Args:
        circuit: a skidl ``Circuit``; defaults to the active ``default_circuit``.
        compat: optional ngspice dialect (``ngbehavior``) for vendor
            PSpice/LTspice ``.lib`` files (e.g. ``"psa"``). Overrides any
            ``Sim_Compat`` on the parts.

    Returns:
        A ``CircuitSimulator`` exposing ``operating_point`` / ``dc_analysis`` /
        ``ac_analysis`` / ``transient_analysis`` and the ``SimulationResult``
        measurement helpers. Requires PySpice + a loadable ngspice.
    """
    from .simulator import CircuitSimulator

    return CircuitSimulator(skidl_flat_view(circuit), compat=compat)


# Lazily surface the heavy classes (they import PySpice) without forcing PySpice
# at ``import skidl.sim`` time.
def __getattr__(name):
    if name in ("SpiceConverter", "ResolvedModel", "SimulationValidationError"):
        from . import converter

        return getattr(converter, name)
    if name in ("CircuitSimulator", "SimulationResult"):
        from . import simulator

        return getattr(simulator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
