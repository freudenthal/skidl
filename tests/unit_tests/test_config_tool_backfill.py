# -*- coding: utf-8 -*-

"""Regression tests for the stored-config tool backfill (adopted from
devbisme/skidl development, 2026-07-21).

Background: ``SkidlConfig.__init__`` seeds ``lib_search_paths`` and
``footprint_search_paths`` with an entry per tool in ``ALL_TOOLS``, but ONLY when
the key is absent entirely -- i.e. only when no ``.skidlcfg`` was found. Once a
config file exists on disk, both dicts are taken verbatim from it.

That is a silent trap whenever a NEW tool is added after a user's config was
written. A ``.skidlcfg`` stored while ``kicad9`` was the newest backend has no
``kicad10`` key, so after upgrading, ``lib_search_paths["kicad10"]`` raises
KeyError (or resolves to nothing) and the KiCad 10 libraries cannot be located --
with no diagnostic pointing at the stale config.

The fix backfills any tool missing from a previously-stored config with that
tool's own defaults, while leaving every key the config DID declare untouched.
The tests below lock both halves: the backfill happens, and it does not overwrite.

Note: ``skidl_eda.env`` sets ``lib_search_paths["kicad10"]`` explicitly at its own
call site, so this defect is masked for the E2E harness; it bites a bootstrapped
project venv that imports the fork directly.

⛔ The backfill branches are UNREACHABLE unless a ``.skidlcfg`` exists on disk --
without one, the ``if <key> not in self`` branch above them always runs. On a dev
box with no stored config the whole change is provably inert (the resolved
config's sha256 is identical with and without it), so the rest of the suite
cannot exercise it and these tests are the only coverage. Each one therefore
writes its OWN config into a tmp dir with the storage dir monkeypatched, rather
than relying on whatever the machine happens to have.
"""

import json
import os

import pytest

from skidl import KICAD10
from skidl.config_ import SkidlConfig
from skidl.tools import ALL_TOOLS


CFG_NAME = ".skidlcfg"


@pytest.fixture
def isolated_cfg(tmp_path, monkeypatch):
    """Point SkidlConfig at empty tmp dirs so the real user config cannot leak in.

    Returns the storage dir; write a ``.skidlcfg`` into it to simulate a
    previously-stored config.
    """
    storage = tmp_path / "storage"
    cwd = tmp_path / "cwd"
    storage.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(
        "skidl.config_._get_default_skidl_storage_dir", lambda: str(storage)
    )
    monkeypatch.chdir(cwd)
    return storage


def _write_cfg(storage, cfg):
    with open(os.path.join(str(storage), CFG_NAME), "w") as fp:
        json.dump(cfg, fp, indent=4)


def test_missing_tool_is_backfilled_in_lib_search_paths(isolated_cfg):
    """A stored config written before kicad10 existed still resolves kicad10 libs."""
    stale = {t: ["/stale/lib/" + t] for t in ALL_TOOLS if t != KICAD10}
    assert KICAD10 not in stale, "fixture precondition: kicad10 absent from the stored config"
    _write_cfg(isolated_cfg, {"tool": KICAD10, "lib_search_paths": stale})

    cfg = SkidlConfig(KICAD10)

    assert KICAD10 in cfg["lib_search_paths"]
    assert cfg["lib_search_paths"][KICAD10], "backfilled entry must not be empty"


def test_missing_tool_is_backfilled_in_footprint_search_paths(isolated_cfg):
    """Same for footprints -- the two dicts are seeded by separate branches."""
    stale = {t: ["/stale/fp/" + t] for t in ALL_TOOLS if t != KICAD10}
    _write_cfg(isolated_cfg, {"tool": KICAD10, "footprint_search_paths": stale})

    cfg = SkidlConfig(KICAD10)

    assert KICAD10 in cfg["footprint_search_paths"]
    assert cfg["footprint_search_paths"][KICAD10], "backfilled entry must not be empty"


def test_backfill_does_not_overwrite_declared_paths(isolated_cfg):
    """The user's own entries win; only ABSENT tools are filled in."""
    declared = {t: ["/mine/" + t] for t in ALL_TOOLS}
    _write_cfg(
        isolated_cfg,
        {
            "tool": KICAD10,
            "lib_search_paths": dict(declared),
            "footprint_search_paths": dict(declared),
        },
    )

    cfg = SkidlConfig(KICAD10)

    for t in ALL_TOOLS:
        assert cfg["lib_search_paths"][t] == ["/mine/" + t]
        assert cfg["footprint_search_paths"][t] == ["/mine/" + t]


def test_every_tool_present_when_no_config_file_exists(isolated_cfg):
    """The pre-existing no-config path is unchanged: every tool gets a default."""
    cfg = SkidlConfig(KICAD10)

    for t in ALL_TOOLS:
        assert t in cfg["lib_search_paths"]
        assert t in cfg["footprint_search_paths"]


def test_backfill_loop_does_not_clobber_the_tool_argument(isolated_cfg):
    """The backfill iterates ALL_TOOLS; it must not rebind the ``tool`` parameter.

    Upstream's version writes ``for tool in ALL_TOOLS``, which shadows the
    constructor argument of the same name. Harmless where it stands today only
    because ``self.tool`` is assigned earlier -- this test pins the property so a
    later edit cannot reintroduce the trap silently.
    """
    stale = {t: ["/stale/" + t] for t in ALL_TOOLS if t != KICAD10}
    _write_cfg(isolated_cfg, {"lib_search_paths": stale, "footprint_search_paths": stale})

    cfg = SkidlConfig(KICAD10)

    assert cfg.tool == KICAD10
