"""Load the RSI loop (scripts/rsi/loop.py and the modules it is split into) fresh, for a test.

The loop used to be one module that tests patched as `lp.NAME`. It is now `state` (the run's paths, caps and limits),
`ledger`, `audit`, `roles`, `drafts`, `cascade`, `stop`, `plugin` and `loop`, plus the scene plug-in's `loop_scene` for a
family without a backend. `LoopNS` is the old `lp`: a name read from it comes from the module that defines it, a name set
on it (monkeypatch.setattr, mock.patch.object) is set there, so a test that patches `lp.transcript` or `lp.STATE` still
patches what the other modules call.

`fresh()` takes the loop's modules out of `sys.modules` while a test loads its own copy and puts the old ones back, so a
test's copy never mixes with another's."""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
SCENE = HERE / "scene"
CORE = ("state", "ledger", "audit", "roles", "drafts", "cascade", "stop", "plugin", "loop")
# every module of the scene plug-in a loop test may load (they import the core by name too)
SCENE_MODULES = tuple(sorted(p.stem for p in SCENE.glob("*.py"))) if SCENE.is_dir() else ()
CLASH = (
    "gate",
    "batch",
    "backend",
    "verdict",
    "merge",
    "segments",
    "rules_build",
    "sealed",
    "try",
    "hit_rate",
    "sandbox",
)
OWNERS = ("state", "cascade", "drafts", "roles", "audit", "ledger", "stop", "loop")
MODULE_ALIASES = {"S", "LG", "AU", "RL", "DR", "CS", "ST", "SC", "BK", "B", "V", "MG", "RB", "SG"}


@contextlib.contextmanager
def fresh(*extra: str, scene: bool = True):
    """The loop's modules (and `extra`) removed from sys.modules and sys.path set; the old ones back afterwards."""
    names = [*CORE, *CLASH, *extra, *SCENE_MODULES]
    saved = {k: sys.modules.pop(k, None) for k in names}
    old_path = list(sys.path)
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    if scene:
        sys.path.insert(0, str(SCENE))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


class _View(Mapping):
    """What `lp.__dict__` shows (pytest's monkeypatch reads it to remember a name's old value)."""

    def __init__(self, ns):
        self._ns = ns

    def __getitem__(self, name):
        m = self._ns._owner(name)
        if m is None:
            raise KeyError(name)
        return vars(m)[name]

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class LoopNS:
    """`lp`: names of the split loop resolved to the module that defines them."""

    __slots__ = ("_m",)

    def __init__(self, modules: dict):
        object.__setattr__(self, "_m", modules)

    @property
    def __dict__(self):  # noqa: A003
        return _View(self)

    def _owner(self, name: str):
        mods = object.__getattribute__(self, "_m")
        for key in (*OWNERS, "loop_scene"):
            m = mods.get(key)
            if m is not None and name in vars(m) and name not in MODULE_ALIASES:
                return m
        return None

    def __getattr__(self, name):
        m = self._owner(name)
        if m is None:  # a module alias the loop's code uses (B, BK, V ...): the first module that has it
            for m in object.__getattribute__(self, "_m").values():
                if name in vars(m):
                    return getattr(m, name)
            raise AttributeError(name)
        return getattr(m, name)

    def __setattr__(self, name, value):
        m = self._owner(name)
        if m is None:
            raise AttributeError(f"{name} is defined in none of the loop's modules")
        setattr(m, name, value)

    def __delattr__(self, name):
        m = self._owner(name)
        if m is None:
            raise AttributeError(name)
        delattr(m, name)

    def module(self, key: str):
        return object.__getattribute__(self, "_m")[key]


def load(*extra: str, scene: bool = True) -> LoopNS:
    """A fresh loop. The caller keeps `fresh()` open while the loop is used (the modules import each other lazily);
    use `with fresh(): lp = load()` only for what runs inside the block."""
    mods = {k: importlib.import_module(k) for k in CORE}
    if scene:
        try:
            mods["loop_scene"] = importlib.import_module("loop_scene")
        except ModuleNotFoundError as e:  # numpy-only environments: the scene runtime is optional for most tests
            if "skfmm" not in str(e):
                raise
    if scene and "loop_scene" in mods:
        mods["state"].H = mods["loop_scene"]  # the scene family's hooks, before any configure()
        mods["state"].CLASS_SUGGEST = mods["loop_scene"].CLASS_SUGGEST
    plug = mods["plugin"]
    plug.load = lambda scene: mods["loop_scene"] if scene else plug.NULL  # this copy's plug-in, not another test's
    return LoopNS(mods)


def fresh_loop(scene: bool = True) -> LoopNS:
    """A fresh loop for a test that does not need to keep `fresh()` open (what the old per-test loaders returned)."""
    with fresh(scene=scene):
        return load(scene=scene)
