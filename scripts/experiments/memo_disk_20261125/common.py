"""Shared by the TODO 52 experiments: services of one slot up, a Bed on them, the skill package and the loop's settings."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from locomani.bench import options
from locomani.bench.simple import __main__ as M
from locomani.bench.simple import run as R
from locomani.bench.simple.config import SimpleConfig
from locomani.bench.simple.session import Session, take_slot

POOL = "/workspace/code/rsi-scene/simple_art_r2/outputs/scene_rsi_20261016/simple_art/pool_diag"
PKG = "/workspace/code/rsi-scene/simple_art_r2/outputs/scene_rsi_20261016/simple_art/incumbents/fda7af7386a1/skills"
TASKS = ("OpenOvenTeleop", "CloseDoorTeleop", "OpenFaucetTeleop", "OpenTrashCanTeleop")
SETTING = {"feed_torque": True}


def pool_ids(task: str) -> list[str]:
    ids = [json.load(open(f))["episode_id"] for f in sorted(Path(POOL).glob(f"{task}_s*.json"))]
    return sorted(ids, key=lambda i: int(i.split("/")[-1]))


@contextlib.contextmanager
def bed_on_slot(task: str, out: Path, slots: str = "1,2,3", overrides: tuple = ()):
    """Take a slot, start the depth service and the Isaac server of it, yield (bed, session); everything is stopped at the end."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SimpleConfig().with_overrides([f"run_dir={out}/run", f"slots={slots}", *overrides])
    idf = out / "ids.txt"
    idf.write_text("")
    sess = Session(cfg, task, idf, out / "rows.jsonl")
    sess.slot, lock = take_slot(cfg)
    try:
        assert sess.start_depth() and sess.start_server(), "services did not start"
        args = M.build_parser().parse_args(["batch", "--slot", str(sess.slot)])
        resolved = options.from_args(args, allow_planned=True)
        bed = R.Bed(cfg, R.load_package(PKG), sess.slot, dict(), resolved.options)
        yield bed, sess
    finally:
        sess.stop_all()
        lock.close()


def timed(fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    return r, time.time() - t
