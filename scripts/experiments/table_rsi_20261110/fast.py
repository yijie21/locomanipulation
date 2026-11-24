"""TODO 46.1: run many variants of a skills package on the same starts, sharing the simulation of what they have in
common (memo.py). One process per GPU slot; a process takes whole starts, so every variant of a start meets the states
its earlier variants stored.

A job file is JSON: dict(family, pool, memo=bool, starts=[label...], variants=[dict(name, skills, set, out)]).
Round records are scene_round's own, written to <out>/<label>.json; a record that exists is not run again.

    $VENV_SIM_PY scripts/experiments/table_rsi_20261110/fast.py <job.json> --procs 6 --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspace/code/locomanipulation")
HERE = Path(__file__).resolve().parent


def worker(job_file: Path, shard: int, shards: int) -> None:
    sys.path[:0] = [str(REPO / "scripts/rsi"), str(REPO / "src"), str(HERE)]
    os.chdir(REPO)
    import family as F
    import memo as M
    import scene_round as SR

    job = json.loads(job_file.read_text())
    fam = F.load(job["family"])
    world = SR.make_world(fam, "scalebfm")
    memo = M.install(world) if job.get("memo", True) else None
    pool = Path(job["pool"])
    t0 = time.time()
    for label in job["starts"][shard::shards]:
        if memo:
            memo.clear()
            memo.label = label
        for v in job["variants"]:
            if v.get("starts") is not None and label not in v["starts"]:
                continue
            out = Path(v["out"])
            if (out / f"{label}.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            skills = SR.load_skills(Path(v["skills"]))
            skills.apply_set(skills, v.get("set") or {})
            SR.run_round(fam, world, skills, SR.skills_hash(Path(v["skills"])), pool, out, label, v.get("set") or {},
                         False, "scalebfm")  # fmt: skip
    if memo:
        print(f"shard {shard}: {time.time() - t0:.0f} s, hits {memo.hits}, misses {memo.misses}, "
              f"simulated seconds skipped {memo.saved_ticks * 0.02:.0f}", flush=True)  # fmt: skip


def run(job_file: Path, procs: int, gpus: str, log: Path | None = None) -> int:
    devices = gpus.split(",")
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0")
    ps = []
    for i in range(procs):
        f = open(log or job_file.with_suffix(".log"), "a")
        ps.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(job_file), "--shard", str(i),
                                    "--shards", str(procs)], env=dict(env, CUDA_VISIBLE_DEVICES=devices[i % len(devices)]),
                                   stdout=f, stderr=subprocess.STDOUT, cwd=REPO))  # fmt: skip
    return max(p.wait() for p in ps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    if a.shard is not None:
        worker(a.job, a.shard, a.shards)
    else:
        raise SystemExit(run(a.job, a.procs, a.gpus))
