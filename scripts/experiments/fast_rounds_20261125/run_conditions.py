"""TODO 52 mechanism 1, step 3: the loop's 60 diagnostic starts (15 per articulated task) under several render conditions, as rounds of
`python -m locomani.bench.simple run` (the loop's own entry: session, services, slot locks), at most --parallel at once.

    python run_conditions.py --out DIR --conds every:1,every:2,lazy48:1,lazy48:2,lazy0:1 [--parallel 3] [--tasks T1,T2]

Condition names: every (lazy_render=false), lazyN (lazy_render=true, lazy_warm=N), subN_M (lazy, warm N, rt_subframes M).
Rows land in DIR/<cond>_r<rep>/<Task>.jsonl; a job whose file already has 15 rows is skipped (resumable).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common as X

SIM = "/workspace/code/locomanipulation/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
SRC = str(Path(__file__).resolve().parents[3] / "src")


def cond_sets(name: str) -> list[str]:
    if name == "every":
        return ["lazy_render=false"]
    m = re.fullmatch(r"lazy(\d+)", name)
    if m:
        return ["lazy_render=true", f"lazy_warm={m.group(1)}"]
    m = re.fullmatch(r"sub(\d+)_(\d+)", name)
    if m:
        return ["lazy_render=true", f"lazy_warm={m.group(1)}", f"lazy_subframes={m.group(2)}"]
    raise SystemExit(f"unknown condition {name}")


def job(args):
    out, cond, rep, task, extra = args
    d = Path(out) / f"{cond}_r{rep}"
    d.mkdir(parents=True, exist_ok=True)
    rows = d / f"{task}.jsonl"
    ids = X.pool_ids(task)
    if rows.exists() and len(rows.read_text().splitlines()) >= len(ids):
        return task, cond, rep, "skipped", 0
    idf = d / f"{task}.ids"
    idf.write_text("".join(i + "\n" for i in ids))
    cmd = [
        SIM,
        "-m",
        "locomani.bench.simple",
        "run",
        task,
        "--ids",
        str(idf),
        "--setting",
        json.dumps(X.SETTING),
        "--out",
        str(rows),
        "--skills",
        X.PKG,
        "--set",
        f"run_dir={d}/run_{task}",
        *sum((["--set", s] for s in cond_sets(cond) + extra), []),
    ]
    t = time.time()
    env = dict(os.environ, PYTHONPATH=SRC)
    with open(d / f"{task}.log", "ab") as lf:
        rc = subprocess.run(
            cmd, env=env, cwd=SRC + "/..", stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        ).returncode
    return task, cond, rep, rc, round(time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--tasks", default=",".join(X.TASKS))
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    jobs = []
    conds = [c.split(":") for c in a.conds.split(",")]
    for rep in sorted(
        {int(r) for _, r in conds}
    ):  # repeat-major, conditions interleaved: slow drift hits every condition alike
        for c, r in conds:
            if int(r) == rep:
                jobs += [(a.out, c, rep, t, a.set) for t in a.tasks.split(",")]
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for r in ex.map(job, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
