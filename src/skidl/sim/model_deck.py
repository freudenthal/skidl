"""Minimal self-contained SPICE decks extracted from vendor library files.

Vendor libraries ship hundreds of models per file, and ngspice condemns the
*whole file* when it trips over one malformed line. Measured across the
KiCad-Spice-Library corpus: 2,101 load failures came from just 102 files, 70 of
which failed at 100% -- e.g. one bad ``i source`` line poisons every one of the 842
zeners defined in ``Zener_DiodesInc.lib``. Those are good models nobody can simulate as long as
the sim path ``.include``s the entire file.

This module extracts only what a netlist actually needs: the target
``.subckt``/``.model`` blocks, their transitive same-file dependencies, and the
top-level ``.param``/``.func`` definitions -- ASCII-sanitized. Everything else,
poison included, stays out of the deck.

Extraction never speaks for the model: when a requested name is not in the file
the functions return ``None`` and the caller falls back to the whole-file
``.include`` (today's behavior). Degrade to today, never to silence.

NOTE: this is a deliberate port of ``skidl_eda.sourcing.corpus_eval``'s
extractor, not an import -- skidl-eda depends on skidl, never the reverse, and
that module's function *source text* is hashed into every stored corpus_eval
record (editing it invalidates the measured store wholesale). If either copy is
ever changed, keep both behaviorally aligned.
"""

import hashlib
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bump when the extraction logic changes: it is part of the staged deck's
# digest, so stale decks from an older extractor are never reused.
DECK_VERSION = 1

_DEF_CACHE: Dict[Any, Any] = {}

_TOKEN_RE = re.compile(r"[A-Za-z_][\w./+-]*")


def _to_ascii(text: str) -> str:
    """Drop non-ASCII bytes -- ngspice rejects the whole deck on a UTF-8 error."""
    return text.encode("ascii", "replace").decode("ascii")


def _parse_definitions(path) -> Tuple[Dict[str, str], List[str]]:
    """``({name.lower(): block_text}, [top-level .param/.func blocks])`` for a file.

    Memoized on (path, mtime, size) so a 500-part library is parsed once.
    """
    try:
        st = os.stat(path)
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return {}, []
    hit = _DEF_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return {}, []

    defs: Dict[str, str] = {}
    params: List[str] = []
    i, n, depth = 0, len(lines), 0
    while i < n:
        s = lines[i].strip()
        low = s.lower()
        if low.startswith(".subckt"):
            if depth != 0:  # nested helper -- consumed by its parent block
                depth += 1
                i += 1
                continue
            start, d, j = i, 1, i + 1
            while j < n and d > 0:
                l2 = lines[j].strip().lower()
                if l2.startswith(".subckt"):
                    d += 1
                elif l2.startswith(".ends") or l2 == ".end":
                    d -= 1
                j += 1
            toks = s.split()[1:]
            # a '+' continued header still names the subckt on the first line
            if toks:
                defs.setdefault(toks[0].lower(), "\n".join(lines[start:j]))
            i = j
            continue
        if low.startswith(".ends"):
            depth = max(0, depth - 1)
        elif low.startswith(".model") and depth == 0:
            start, j = i, i + 1
            while j < n and lines[j].lstrip().startswith("+"):
                j += 1
            toks = s.split()
            if len(toks) >= 2:
                defs.setdefault(toks[1].lower(), "\n".join(lines[start:j]))
            i = j
            continue
        elif (low.startswith(".param") or low.startswith(".func")) and depth == 0:
            start, j = i, i + 1
            while j < n and lines[j].lstrip().startswith("+"):
                j += 1
            params.append("\n".join(lines[start:j]))
            i = j
            continue
        i += 1

    _DEF_CACHE[key] = (defs, params)
    return defs, params


def extract_minimal_deck(path, names: Iterable, max_defs: int = 400) -> Optional[str]:
    """Smallest self-contained ASCII deck defining every name in ``names``.

    Returns ``None`` when any requested name is not found in the file (the
    caller then falls back to including the whole file).

    Each definition is emitted exactly ONCE across the union of per-name picks:
    one netlist may use several models from the same library file, and separate
    per-name decks would redefine the helper subckts they share -- which ngspice
    treats as a redefinition error.

    Dependencies are found by a deliberately loose token scan of each picked
    block: over-including a definition is harmless, missing one is not.
    """
    defs, params = _parse_definitions(path)
    keys = [str(n).strip().lower() for n in names]
    keys = [k for k in keys if k]
    if not keys or any(k not in defs for k in keys):
        return None
    picked: List[str] = []
    seen = set()
    stack = list(reversed(keys))
    while stack and len(seen) < max_defs:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k)
        block = defs.get(k)
        if block is None:
            continue
        picked.append(block)
        for tok in _TOKEN_RE.findall(block):
            t = tok.lower()
            if t not in seen and t in defs:
                stack.append(t)
    return _to_ascii("\n".join(params + picked))


def stage_minimal_deck(path, names: Iterable, cache_dir) -> Optional[str]:
    """Write ``extract_minimal_deck``'s text to ``cache_dir``; return its path.

    The file name ``<stem>_<digest8>_mindeck.lib`` is deterministic over
    (absolute source path, sorted names, ``DECK_VERSION``), so the same request
    always maps to the same file and runs stay reproducible. Restaged when
    missing or older than the source -- the same staleness rule
    ``SpiceConverter._safe_lib_path`` uses. ``None`` -> caller falls back to the
    whole-file include.
    """
    src = os.path.abspath(str(path))
    wanted = sorted({str(n).strip().lower() for n in names if str(n).strip()})
    if not wanted:
        return None
    try:
        deck = extract_minimal_deck(src, wanted)
    except Exception as exc:  # noqa: BLE001 - extraction must never break a sim
        logger.warning(f"Minimal-deck extraction failed for {src}: {exc}")
        return None
    if not deck:
        return None

    stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(src))
    root, _ext = os.path.splitext(stem)
    digest = hashlib.sha1(
        "\x00".join([src, str(DECK_VERSION)] + wanted).encode("utf-8", "replace")
    ).hexdigest()[:8]
    staged = os.path.join(cache_dir, f"{root}_{digest}_mindeck.lib")
    try:
        if not os.path.exists(staged) or (
            os.path.getmtime(staged) < os.path.getmtime(src)
        ):
            os.makedirs(cache_dir, exist_ok=True)
            with open(staged, "w", encoding="ascii", errors="replace", newline="\n") as fh:
                fh.write(deck)
                if not deck.endswith("\n"):
                    fh.write("\n")
        return staged
    except OSError as exc:  # pragma: no cover - filesystem specifics
        logger.warning(f"Could not stage minimal deck for {src}: {exc}")
        return None
