"""Index a directory tree of external SPICE model files by internal name.

The local MPN store (:mod:`skidl.sim.model_store`) resolves a model only when a
file is *named* ``<MPN>.lib``. A bulk corpus like the KiCad-Spice-Library has
tens of thousands of arbitrarily-named files, each defining one or more
``.model`` / ``.subckt`` blocks whose *internal* names are what a netlist
references. This module walks such trees once, parses every ``.model`` and
``.subckt`` definition, and resolves a model name -> the file that defines it
(plus its kind, device type, and subckt node order).

Dependency-free and OFF by default: with no roots configured (env
``SKIDL_SPICE_LIB_PATH`` unset) :func:`get_library_index` returns ``None`` and
the converter behaves exactly as before. The parse regexes deliberately mirror
``SpiceConverter._scan_lib`` so the index and the emitter never disagree.

Not indexed: ``.cir`` files (example netlists, not reusable models).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Env: os.pathsep-separated model-library roots to index; a JSON cache location.
LIB_PATH_ENV_VAR = "SKIDL_SPICE_LIB_PATH"
CACHE_ENV_VAR = "SKIDL_SPICE_LIB_CACHE"

# File extensions that hold reusable models. ``.cir`` is intentionally excluded
# (whole example netlists, not libraries). Compared case-insensitively.
_MODEL_EXTS = (".lib", ".sub", ".spi", ".mod", ".fam")

_CACHE_VERSION = 2


@dataclass
class ModelHit:
    """One resolved model: which file defines it, and how to instantiate it."""

    name: str  # internal .model/.subckt name, original casing
    path: str  # absolute path to the defining file
    kind: str  # "subckt" | "model"
    device_type: str = ""  # for .model: D / NPN / PNP / NJF / NMOS ...; "" for subckt
    nodes: List[str] = field(default_factory=list)  # subckt node order (contract!)
    header: str = ""  # comment lines above a subckt (pin-role hints for the user)
    root: str = ""  # the configured root this file was found under
    prec: int = 0  # precedence score (higher wins on duplicate names)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelHit":
        return cls(**d)


def _precedence(path: str) -> int:
    """Rank duplicate definitions. README precedence: Manufacturer > curated
    device dirs > spice_complete > uncategorized. Higher wins."""
    p = path.replace("\\", "/").lower()
    if "/manufacturer/" in p:
        return 100
    for d in (
        "/diode/",
        "/transistor/",
        "/operational amplifier/",
        "/digital logic/",
        "/optocoupler/",
    ):
        if d in p:
            return 80
    if "/spice_complete/" in p:
        return 40
    if "/uncategorized/" in p:
        return 20
    return 50


def _parse_file(path: str) -> List[ModelHit]:
    """Extract every ``.model`` / ``.subckt`` definition from one file.

    Line-based (robust + fast over tens of thousands of files) and tolerant of
    ``+`` continuation lines in a subckt node list. Never raises on a bad file.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return []
    hits: List[ModelHit] = []
    prec = _precedence(path)
    n = len(lines)
    depth = 0  # nesting inside .subckt/.ends -- only depth-0 defs are usable
    for i, raw in enumerate(lines):
        s = raw.strip()
        low = s.lower()
        if low.startswith(".ends") or low == ".end":
            depth = max(0, depth - 1)
            continue
        if low.startswith(".subckt"):
            top_level = depth == 0
            depth += 1
            if not top_level:
                continue  # an internal helper subckt -- not instantiable directly
            toks = s.split()[1:]
            j = i + 1
            while j < n and lines[j].lstrip().startswith("+"):
                toks += lines[j].lstrip()[1:].split()
                j += 1
            if not toks:
                continue
            name = toks[0]
            nodes: List[str] = []
            for t in toks[1:]:
                if t.lower() == "params:" or "=" in t:
                    break
                nodes.append(t)
            # Gather the immediately-preceding comment block as pin-role hints.
            header_lines = []
            k = i - 1
            while k >= 0 and lines[k].lstrip().startswith("*") and len(header_lines) < 10:
                header_lines.append(lines[k].rstrip())
                k -= 1
            header = "\n".join(reversed(header_lines))
            hits.append(ModelHit(name, path, "subckt", "", nodes, header, prec=prec))
        elif low.startswith(".model") and depth == 0:
            toks = s.split()
            if len(toks) < 2:
                continue
            name = toks[1]
            dtype = toks[2].split("(")[0] if len(toks) > 2 else ""
            hits.append(ModelHit(name, path, "model", dtype, prec=prec))
    return hits


class SpiceLibraryIndex:
    """Name -> defining-file index over one or more model-library roots."""

    def __init__(self, roots: List[str], cache_path: Optional[str] = None):
        self.roots = [os.path.abspath(r) for r in roots if r]
        self.cache_path = cache_path or _default_cache_path()
        # name.lower() -> [ModelHit, ...] ranked best-first
        self._by_name: Dict[str, List[ModelHit]] = {}
        self._built = False

    # -- building / caching ------------------------------------------------- #

    def _signature(self) -> str:
        """Cheap identity of the configured roots (paths only) -> cache key."""
        h = hashlib.sha1()
        h.update(str(_CACHE_VERSION).encode())
        for r in sorted(self.roots):
            h.update(r.encode("utf-8", "replace"))
            h.update(b"\0")
        return h.hexdigest()

    def build(self, force: bool = False) -> "SpiceLibraryIndex":
        """Populate the index. Loads a matching JSON cache unless ``force``.

        Note: staleness is keyed on the *set of roots*, not file mtimes -- adding
        files to a corpus requires ``build(force=True)`` (the CLI exposes a
        ``--rebuild`` flag). This keeps every-run resolve O(load-cache), not
        O(walk 50k files).
        """
        if self._built and not force:
            return self
        if not force and self._load_cache():
            self._built = True
            return self
        self._by_name = {}
        total_files = 0
        for root in self.roots:
            if not os.path.isdir(root):
                logger.warning(f"SPICE library root not found: {root}")
                continue
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() not in _MODEL_EXTS:
                        continue
                    total_files += 1
                    fpath = os.path.join(dirpath, fn)
                    for hit in _parse_file(fpath):
                        hit.root = root
                        self._by_name.setdefault(hit.name.lower(), []).append(hit)
        # Rank each name's definitions deterministically: precedence desc, then
        # shortest path, then lexicographic.
        for name, hits in self._by_name.items():
            hits.sort(key=lambda h: (-h.prec, len(h.path), h.path))
        self._built = True
        logger.info(
            f"Indexed {len(self._by_name)} SPICE models from {total_files} files "
            f"across {len(self.roots)} root(s)"
        )
        self._save_cache()
        return self

    def _load_cache(self) -> bool:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return False
        if data.get("version") != _CACHE_VERSION or data.get("signature") != self._signature():
            return False
        self._by_name = {
            name: [ModelHit.from_dict(h) for h in hits]
            for name, hits in data.get("models", {}).items()
        }
        logger.debug(f"Loaded SPICE library index cache: {self.cache_path}")
        return True

    def _save_cache(self) -> None:
        data = {
            "version": _CACHE_VERSION,
            "signature": self._signature(),
            "roots": self.roots,
            "models": {
                name: [h.to_dict() for h in hits]
                for name, hits in sorted(self._by_name.items())
            },
        }
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=0, sort_keys=True)
        except OSError as exc:  # pragma: no cover
            logger.debug(f"Could not write SPICE library index cache: {exc}")

    # -- querying ----------------------------------------------------------- #

    def resolve(self, name: str) -> Optional[ModelHit]:
        """Best-ranked definition of ``name`` (case-insensitive), or None."""
        if not name:
            return None
        if not self._built:
            self.build()
        hits = self._by_name.get(str(name).strip().lower())
        return hits[0] if hits else None

    def alternates(self, name: str) -> List[ModelHit]:
        """All definitions of ``name``, best-first (for reporting/fallback)."""
        if not name:
            return []
        if not self._built:
            self.build()
        return list(self._by_name.get(str(name).strip().lower(), []))

    def search(self, query: str, kind: Optional[str] = None,
               device_types: Optional[List[str]] = None,
               limit: int = 25) -> List[ModelHit]:
        """Substring search over model names, ranked. Optional kind
        ("subckt"/"model") and device-type filters (e.g. ["D"] for diodes)."""
        if not self._built:
            self.build()
        q = (query or "").strip().lower()
        dts = {d.upper() for d in (device_types or [])}
        out: List[ModelHit] = []
        for name_l, hits in self._by_name.items():
            if q and q not in name_l:
                continue
            best = hits[0]
            if kind and best.kind != kind:
                continue
            if dts and best.device_type.upper() not in dts:
                continue
            out.append(best)
        # exact match first, then prefix, then precedence, then name
        def rank(h: ModelHit):
            nl = h.name.lower()
            return (nl != q, not nl.startswith(q), -h.prec, nl)
        out.sort(key=rank)
        return out[:limit]

    def stats(self) -> dict:
        if not self._built:
            self.build()
        subckts = sum(1 for hs in self._by_name.values() if hs[0].kind == "subckt")
        return {
            "names": len(self._by_name),
            "subckts": subckts,
            "models": len(self._by_name) - subckts,
            "roots": list(self.roots),
        }


def _default_cache_path() -> str:
    env = os.environ.get(CACHE_ENV_VAR)
    if env:
        return env
    return os.path.join(
        os.path.expanduser("~"), ".skidl", "spice_models", "library_index.json"
    )


def _configured_roots() -> List[str]:
    env = os.environ.get(LIB_PATH_ENV_VAR, "")
    return [p for p in env.split(os.pathsep) if p.strip()]


# Process-global singleton so the cache is loaded at most once per run.
_INDEX_SINGLETON: Optional[SpiceLibraryIndex] = None
_INDEX_ROOTS_KEY: Optional[str] = None


def get_library_index(roots: Optional[List[str]] = None) -> Optional[SpiceLibraryIndex]:
    """The configured library index, or ``None`` when none is configured.

    ``roots`` overrides the ``SKIDL_SPICE_LIB_PATH`` env var (mainly for tests).
    Returns ``None`` (feature inert) when there are no roots -- the converter
    then behaves exactly as before.
    """
    global _INDEX_SINGLETON, _INDEX_ROOTS_KEY
    use_roots = roots if roots is not None else _configured_roots()
    if not use_roots:
        return None
    key = os.pathsep.join(os.path.abspath(r) for r in use_roots)
    if _INDEX_SINGLETON is None or _INDEX_ROOTS_KEY != key:
        _INDEX_SINGLETON = SpiceLibraryIndex(use_roots)
        _INDEX_ROOTS_KEY = key
    return _INDEX_SINGLETON
