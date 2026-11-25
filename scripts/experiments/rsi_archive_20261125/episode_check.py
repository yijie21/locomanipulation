"""TODO 27.2.3 (plan docs/handoff/scene-rsi-plan-20261016.md §7): the scene-task runner end to end on the ADT
apartment -- a Demo 1 task (default `canister_pickup`: held(CoffeeCanisterLarge)), random starts (task object moved on
its surface, robot spawned 1-3 m away), the seed skills' plan, the goal judged on the privileged state.

    CUDA_VISIBLE_DEVICES=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/rsi/episode_check.py [--task canister_pickup] [--rounds 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src")]

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.tasks.scene.episode import run_episode  # noqa: E402
from locomani.tasks.scene.skills import plan_for  # noqa: E402
from locomani.tasks.scene.skills.pick import pick_simple  # noqa: E402
from locomani.tasks.scene.start import sample_start  # noqa: E402
from locomani.tasks.scene.world import SceneWorld, Task  # noqa: E402

ADT = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"
TASKS = REPO / "outputs" / "scene_discovery" / "adt_apartment" / "discovery" / "opus_20261005b" / "tasks.jsonl"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "episode_check"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="canister_pickup")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick", choices=("program", "simple"), default="program", help="held(X): the seed pick variant")
    a = ap.parse_args(argv)
    rows = {json.loads(x)["id"]: json.loads(x) for x in TASKS.read_text().splitlines() if x.strip()}
    policy = R.Policy("cuda", ScaleBfmBackendConfig())
    world = SceneWorld(ADT, policy)
    task = Task.from_row(rows[a.task], world.meta)
    plan = plan_for(task)
    if a.pick == "simple" and plan is not None:
        key = task.objects[0]
        plan = lambda ctx: pick_simple(ctx, key)  # noqa: E731
    if plan is None:
        raise SystemExit(f"no seed plan covers {task.goal}")
    out = OUT / f"{a.task}_{a.pick}"
    out.mkdir(parents=True, exist_ok=True)
    wins = 0
    for i in range(a.rounds):
        seed = a.seed + i
        start = sample_start(
            world.meta, world.occ, world.nav, list(task.objects), np.random.default_rng(seed), seed=seed, room=task.room
        )
        res, arrays = run_episode(world, task, start, plan)
        wins += res.ok
        (out / f"round_{seed:04d}.json").write_text(
            json.dumps(dict(ok=res.ok, why=res.why, execution=res.execution), indent=1, default=float)
        )
        np.savez_compressed(out / f"round_{seed:04d}.npz", **arrays)
        steps = [(s["step"], s["ok"], s["why"][:40]) for s in res.execution["steps"]]
        print(
            f"seed {seed}: ok={res.ok} why={res.why!r} t={res.execution['t_end']} wall={res.execution['wall_s']}s",
            steps,
            flush=True,
        )
    print(f"{wins}/{a.rounds} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
