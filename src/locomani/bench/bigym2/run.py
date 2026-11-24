"""One BiGym 2.0 episode: the skill package on the facade, the judge on the server's record (TODO 49.5).

The skill package (--skills: a folder with __init__.py and skill.py) is loaded by path under a unique module name (`load_package`);
nothing is put on sys.path. It gets `client.PackageEnv`, not the link. Its parameters are the whole setting space (`pkg.apply_set`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import itertools
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

from locomani.bench.bigym2 import client as C
from locomani.bench.bigym2 import episodes as EP
from locomani.bench.bigym2 import judge, record
from locomani.bench.bigym2.config import SEED_DIR, BiGym2Config
from locomani.bench.bigym2.privileged import BiGymPrivileged
from locomani.bench.bigym2.truth import TruthView

PACKAGE_FILES = ("__init__.py", "skill.py")
_COUNTER = itertools.count()


def load_package(path) -> ModuleType:
    """Load the skill package in folder `path` under a unique module name (two copies side by side do not clash)."""
    path = Path(path).resolve()
    missing = [f for f in PACKAGE_FILES if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: skill package files missing: {missing}")
    name = f"_bigym2_skill_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        importlib.import_module(f"{name}.skill")
    except BaseException:
        for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[k]
        raise
    mod.path = path
    return mod


def package_hash(pkg: ModuleType) -> str:
    h = hashlib.sha256()
    for f in PACKAGE_FILES:
        h.update((Path(pkg.path) / f).read_bytes())
    return h.hexdigest()[:12]


@dataclasses.dataclass
class Bed:
    """One process's run context: the configs, the skill package, the environment server link."""

    cfg: BiGym2Config
    pkg: ModuleType
    extra: dict = dataclasses.field(default_factory=dict)  # the fields every row carries (bench, bigym2_config, ...)
    link: C.ServerLink = None

    def __post_init__(self):
        if self.link is None:
            self.link = C.ServerLink(self.cfg)


def run(bed: Bed, eid: str, setting: dict) -> dict:
    """One episode -> its row. A failure of the package or the server is a row with `error`, not an exception."""
    cfg = bed.cfg
    task, pool, _ = EP.parse_id(eid)
    seed = EP.check_id(cfg, eid)
    base = record.row_fields(eid, task, pool, seed)
    t0 = time.time()
    try:
        K = bed.pkg.apply_set(setting, task)
        info = bed.link.ask(*EP.reset_msg(cfg, eid))
        priv = BiGymPrivileged(bed.link, task, info["instruction"], info["names"], info["movable"])
        ep = C.Episode(
            task=task, link=bed.link, info=info, K=K, state=info["state"], max_steps=cfg.max_steps, view=TruthView(priv)
        )
        flow = {}
        try:
            flow = bed.pkg.run(ep.facade()) or {}
        except C.Done:
            pass
        outcome = bed.link.ask("outcome")
        verdict = judge.verdict(task, outcome)
        return record.make_row(base, outcome, verdict, flow, ep.marks, time.time() - t0,
                               dict(bed.extra, package=package_hash(bed.pkg), setting=setting))  # fmt: skip
    except Exception:  # noqa: BLE001 - a row says what happened
        if bed.link.proc is not None and bed.link.proc.poll() is not None:
            bed.link.close()  # the server died: the next episode starts a new one
        return record.error_row(base, traceback.format_exc(), time.time() - t0, dict(bed.extra, setting=setting))


__all__ = ["Bed", "load_package", "run", "SEED_DIR"]
