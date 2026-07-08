# -*- coding: utf-8 -*-

# The MIT License (MIT) - Copyright (c) Dave Vandenbout.

"""skidl.sim -- simulate a skidl-authored KiCad design through a macromodel SPICE
stack (ported from the circuit-synth simulation layer).

This subpackage is optional: importing ``skidl`` does not import it, so it adds
no PySpice/ngspice dependency to the base package. It exposes an *adapter* from a
skidl ``Circuit`` to the frontend-agnostic read-only view the circuit-synth
``SpiceConverter`` consumes (its documented ``_FlatCircuit`` seam: only
``.components``, ``.nets``, ``.name``). The port is therefore an adapter, not a
rewrite.

Packaging note: implemented in-tree under ``skidl.sim`` for the fork (least
friction to test). The adapter has NO circuit_synth dependency, so it can be
lifted verbatim into a standalone companion package (``skidl-sim``) for
upstreaming; only the converter/simulator it feeds need vendoring alongside.
"""

from .adapter import (  # noqa: F401
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
]
