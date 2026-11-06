#!/usr/bin/env python3
"""One episode of a task of a logistics_room_1030 family with the main skill library -- the pack line's smoke test
(user 2026-11-05: code and a smoke test before any RSI run). Not an RSI round: no sandbox, no pool, no gate, no record
in the loop's output tree.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src $VENV_SIM_PY scripts/gso_pack_check.py \\
        --family gso_pack_s0 [--task pack_gso_0266 | first] [--seed 0] [--tries 10] [--out outputs/gso_pack_check]

Loads the family (scripts/rsi/family.py), builds the world as a round does (scene_round.make_world: the family's motor
block, the ScaleBFM controller), draws the task's start as its pools do (sample_start with the family's spawn and the
row's placement; a seed the surface cannot place is skipped, up to --tries), runs `plan_for(task)` of the library in
src/ with the family's hold_s / judge_s (episode.run_episode) and writes <out>/<family>/<task>_s<seed>.json (ok, why,
the step log, the judging window) and .npz (the trace). Prints one line per logged step and the verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "src"), str(HERE / "rsi")]

import family as F  # noqa: E402
import scene_round  # noqa: E402

from locomani.tasks.scene import skills  # noqa: E402
from locomani.tasks.scene.episode import HOLD_S, run_episode  # noqa: E402
from locomani.tasks.scene.start import sample_start  # noqa: E402

OUT = REPO / "outputs" / "gso_pack_check"


def draw_start(fam: dict, world, task, task_id: str, seed: int, tries: int):
    """The first seed from `seed` whose start the surface can place (the pools skip the others the same way)."""
    for s in range(seed, seed + tries):
        try:
            return sample_start(fam["meta"], world.occ, world.nav, list(task.objects), np.random.default_rng(s), seed=s,
                                room=task.room, spawn=fam.get("spawn"), placement=fam["rows"][task_id].get("placement"))  # fmt: skip
        except RuntimeError as e:
            print(f"seed {s}: skipped ({e})", flush=True)
    raise SystemExit(f"no start for {task_id} in seeds {seed}..{seed + tries - 1}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="a family under scripts/rsi/families, e.g. gso_pack_s0")
    ap.add_argument("--task", default="first", help="a task id of the family, or `first`")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=10, help="seeds tried for a start the surface can place")
    ap.add_argument("--controller", default="scalebfm")
    ap.add_argument("--limit-s", type=float, default=None, help="simulated seconds the episode may take")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    fam = F.load(a.family)
    task_id = fam["tasks"][0] if a.task == "first" else a.task
    task = F.task(fam, task_id)
    plan = skills.plan_for(task)
    if plan is None:
        raise SystemExit(f"no skill in the library covers {list(task.goal)}")
    t0 = time.time()
    world = scene_round.make_world(fam, a.controller)
    start = draw_start(fam, world, task, task_id, a.seed, a.tries)
    print(f"{a.family} {task_id}: goal {list(task.goal)}; start seed {start.seed}, robot at "
          f"{np.round(start.robot_xy, 2).tolist()}, objects "
          f"{ {k: np.round(p[0][:2], 2).tolist() for k, (p) in start.placements.items()} } ({time.time() - t0:.0f} s to load)",
          flush=True)  # fmt: skip
    kw = {} if a.limit_s is None else dict(limit_s=a.limit_s)
    res, arrays = run_episode(world, task, start, plan, hold_s=float(fam.get("hold_s", HOLD_S)),
                              judge_s=fam.get("judge_s"), **kw)  # fmt: skip
    ex = res.execution
    for s in ex["steps"]:
        extra = {k: v for k, v in s.items() if k not in ("step", "ok", "why", "t0", "t1")}
        print(f"  {s['t0']:7.2f}-{s['t1']:7.2f} {s['step']:<14} {'ok ' if s['ok'] else 'FAIL'} {s['why'][:70]} "
              f"{json.dumps(extra, default=float)[:160]}", flush=True)  # fmt: skip
    out = a.out / a.family
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(family=a.family, task=task_id, goal=list(task.goal), seed=start.seed, start=start.to_dict(), ok=res.ok,
               why=res.why, fell=ex["fell"], plan_ok=ex["plan_ok"], plan_why=ex["plan_why"], error=ex["error"],
               steps=ex["steps"], t_end=ex["t_end"], judge=ex.get("judge"), wall_s=round(time.time() - t0, 1),
               skills=scene_round.skills_hash(REPO / "src" / "locomani" / "tasks" / "scene" / "skills"))  # fmt: skip
    (out / f"{task_id}_s{start.seed}.json").write_text(json.dumps(rec, indent=1, default=float))
    np.savez_compressed(out / f"{task_id}_s{start.seed}.npz", **arrays)
    print(f"{'OK' if res.ok else 'FAIL'} {task_id}: {res.why!r} t_end={ex['t_end']} wall={rec['wall_s']}s "
          f"judge={ex.get('judge')} -> {out / f'{task_id}_s{start.seed}.json'}", flush=True)  # fmt: skip
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
