# -*- coding: utf-8 -*-

"""Symbol-classification regressions in ``SpiceConverter._classify``.

Two substring traps fixed together (found building an LED pulser):
  * ``Device:LED`` contains ``Device:L`` -> was misread as an *inductor* (and it
    is not caught by the ``Device:D`` diode branch), so every LED simulated as a
    1 H inductor.
  * ``Transistor_FET:IRLML0030`` contains ``lm`` -> the ``["op","amp","lm","tl"]``
    op-amp heuristic grabbed it as an *op-amp* before the ``Transistor_FET:``
    check, so the FET simulated as a VCVS. Same trap the regulator checks already
    guard against (LM317/LM1117).
"""

import pytest

from skidl.sim.converter import SpiceConverter


@pytest.mark.parametrize(
    "symbol,expected",
    [
        # LEDs are diodes, not inductors.
        ("Device:LED", "diode"),
        ("Device:LED_Small", "diode"),
        ("Device:LED_ALT", "diode"),
        # Real inductors still classify as inductors.
        ("Device:L", "inductor"),
        ("Device:L_Small", "inductor"),
        ("Device:L_Core_Ferrite", "inductor"),
        # Transistors with op-amp-like substrings ("lm"/"tl"/"op") are transistors.
        ("Transistor_FET:IRLML0030", "mosfet"),
        ("Transistor_FET:IRLZ44N", "mosfet"),
        ("Transistor_BJT:BC547", "bjt"),
        # Genuine op-amps still classify as op-amps.
        ("Amplifier_Operational:OPA365xxD", "opamp"),
        ("Amplifier_Operational:TL071", "opamp"),
        ("Amplifier_Operational:LM358", "opamp"),
        # Regulators unaffected.
        ("Regulator_Linear:LM317_TO-220", "ldo"),
        # Other diodes unaffected.
        ("Device:D_Photo", "diode"),
        ("Device:D_Schottky", "diode"),
    ],
)
def test_classify(symbol, expected):
    assert SpiceConverter._classify(symbol) == expected
