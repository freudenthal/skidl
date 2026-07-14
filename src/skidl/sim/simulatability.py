"""Part-agnostic SPICE model-class classification.

The SPICE corpus is, and will remain, unreliable: ~50k models contributed by
many vendors in mixed dialects (ngspice ``.model``, PSpice/LTspice ``.subckt``,
XSPICE digital primitives, PSpice ``U``/``ugate`` digital devices, encrypted
blobs). ngspice-in-KiCad can only run the analog ones. A design that *depends*
on a class ngspice cannot run (a corpus flip-flop, say) currently discovers that
only after being wired in and getting a dead/opaque run.

This module inspects a resolved model's **body text** -- never its name -- and
labels its dialect plus a ``simulatable`` verdict, so the sourcing CLI and the
sim pre-flight can flag an un-runnable class *before* a design commits to it.

Governing principle: **no part is named here.** Detection is by structural
signature only, so a verdict holds for the next unknown part exactly as it holds
for the ones seen so far.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ``simulatable`` verdict values.
YES = "yes"
NO = "no"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelClass:
    """A model's dialect + whether ngspice-in-KiCad can run it.

    ``dialect`` is a short structural label (``"ngspice-model"``,
    ``"spice-subckt"``, ``"xspice-digital"``, ``"pspice-digital"``,
    ``"xspice-codemodel"``, ``"encrypted"``, ``"unresolved"``, ``"unknown"``).
    ``simulatable`` is one of ``yes`` / ``no`` / ``unknown`` and ``reason`` is a
    human-readable one-liner naming the *class* (not the part).
    """

    dialect: str
    simulatable: str
    reason: str

    @property
    def ok(self) -> bool:
        """True only for a positively-simulatable class (``yes``)."""
        return self.simulatable == YES

    @property
    def runnable(self) -> bool:
        """True unless the class is known un-runnable (``yes`` or ``unknown``)."""
        return self.simulatable != NO


# -- Structural signatures (by dialect, never by part name) ------------------- #

# XSPICE: any ``A`` device is a code-model instance; the digital ones bind a
# ``.model <name> d_<type>`` card (d_dff, d_xor, d_and, d_source, ...). Those are
# pure event nodes that need adc/dac bridges the sim layer cannot inject.
_XSPICE_ADEV_RE = re.compile(r"^[ \t]*a\w+\s", re.IGNORECASE | re.MULTILINE)
_XSPICE_DMODEL_RE = re.compile(
    r"^[ \t]*\.model\s+\S+\s+d_\w+", re.IGNORECASE | re.MULTILINE
)

# PSpice digital ``U`` devices: ``U<name> <primitive>(<n>...) ...`` bound to
# ``ugate``/``utgate``/``uio``/``ueff``/``uadc``/``udac``/``udly`` timing models
# and ``IO_*`` I/O models. ngspice does not implement the PSpice digital family.
_PSPICE_UDEV_RE = re.compile(r"^[ \t]*u\w+\s+\w+\s*\(", re.IGNORECASE | re.MULTILINE)
_PSPICE_UMODEL_RE = re.compile(
    r"^[ \t]*\.model\s+\S+\s+u(gate|tgate|io|eff|adc|dac|dly)\b",
    re.IGNORECASE | re.MULTILINE,
)
_PSPICE_IO_RE = re.compile(r"\bIO_\w+\b")

# Encrypted / binary body: an explicit marker, an include of a ``.enc`` blob, or
# a body whose bytes are mostly non-printable (LTspice/PSpice encrypted subckts).
_ENC_MARKER_RE = re.compile(r"encrypted|\*\s*encoded", re.IGNORECASE)
_ENC_INCLUDE_RE = re.compile(
    r"^[ \t]*\.(inc|include|lib)\b.*\.enc\b", re.IGNORECASE | re.MULTILINE
)

_MODEL_RE = re.compile(r"^[ \t]*\.model\b", re.IGNORECASE | re.MULTILINE)
_SUBCKT_RE = re.compile(r"^[ \t]*\.subckt\b", re.IGNORECASE | re.MULTILINE)


def _looks_encrypted(text: str) -> bool:
    """True if the body is mostly non-printable (an encrypted/binary blob)."""
    sample = text[:4000]
    if not sample:
        return False
    nonprint = sum(
        1 for c in sample if ord(c) < 9 or 13 < ord(c) < 32 or ord(c) == 127
    )
    return nonprint > len(sample) * 0.02


def classify_spice_model(block: str, full_text: str | None = None) -> ModelClass:
    """Classify a model's dialect + simulatability from its body text.

    ``block`` is the specific ``.subckt``/``.model`` body (from
    :func:`classify_model_file`); ``full_text`` is the whole file, consulted for
    the decisive ``.model d_*`` / ``ugate`` card when a subckt references it but
    defines it at file scope. Pass only ``block`` when the whole model is
    self-contained.
    """
    if not block or not block.strip():
        return ModelClass("empty", UNKNOWN, "empty model body")

    scope = block if full_text is None else block + "\n" + full_text

    # Encrypted first: a blob can't be parsed for anything else.
    if _ENC_MARKER_RE.search(block) or _ENC_INCLUDE_RE.search(block) or _looks_encrypted(
        block
    ):
        return ModelClass(
            "encrypted", NO, "encrypted/binary model body -- ngspice cannot read it"
        )

    # XSPICE ``A`` device: digital (d_* model) is un-runnable; a bare code-model
    # instance needs .cm codemodels loaded (verdict unknown, not a hard no).
    if _XSPICE_ADEV_RE.search(block):
        if _XSPICE_DMODEL_RE.search(scope):
            return ModelClass(
                "xspice-digital",
                NO,
                "XSPICE digital primitive (d_* model); ngspice cannot run it in "
                "the analog netlist without adc/dac bridges the sim layer cannot "
                "inject",
            )
        return ModelClass(
            "xspice-codemodel",
            UNKNOWN,
            "XSPICE A-device code model; needs the matching .cm codemodel loaded",
        )

    # PSpice ``U`` digital device.
    if _PSPICE_UDEV_RE.search(block) and (
        _PSPICE_UMODEL_RE.search(scope) or _PSPICE_IO_RE.search(scope)
    ):
        return ModelClass(
            "pspice-digital",
            NO,
            "PSpice U-device digital primitive (ugate/IO_*); ngspice does not "
            "implement the PSpice digital device family",
        )

    # Analog: an analog .subckt (checked first -- its body legitimately contains
    # internal .model cards) or a plain .model card.
    if _SUBCKT_RE.search(block):
        return ModelClass("spice-subckt", YES, "analog .subckt")
    if _MODEL_RE.search(block):
        return ModelClass("ngspice-model", YES, "analog .model card")

    return ModelClass("unknown", UNKNOWN, "no recognizable .model or .subckt")


def _read(path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _slice_subckt(text: str, name: str) -> str | None:
    pat = re.compile(
        r"^[ \t]*\.subckt\s+" + re.escape(str(name)) + r"\b",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return None
    end = re.compile(r"^[ \t]*\.ends\b", re.IGNORECASE | re.MULTILINE)
    e = end.search(text, m.end())
    return text[m.start() : (e.end() if e else len(text))]


def _slice_model(text: str, name: str) -> str | None:
    pat = re.compile(
        r"^[ \t]*\.model\s+" + re.escape(str(name)) + r"\b",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return None
    lines = text[m.start() :].splitlines()
    out = [lines[0]]
    for ln in lines[1:]:  # gather '+' continuation lines
        if ln.lstrip().startswith("+"):
            out.append(ln)
        else:
            break
    return "\n".join(out)


def classify_model_file(path, name=None) -> ModelClass:
    """Read ``path``, isolate the ``name`` block, and classify it.

    Best-effort: if the named block can't be isolated the whole file is
    classified. Returns an ``unresolved`` verdict when the file is unreadable.
    """
    full = _read(path)
    if not full.strip():
        return ModelClass("unresolved", UNKNOWN, "model file unreadable or missing")
    block = None
    if name:
        block = _slice_subckt(full, name) or _slice_model(full, name)
    if block is None:
        # No named block found: classify the whole file (may be a single model).
        return classify_spice_model(full)
    return classify_spice_model(block, full_text=full)
