"""Round batches for the loop (probes, diagnostic re-runs) and the gate: every job -- one spec, optionally forked, on
some starts -- is cut into shards of SHARD_ROUNDS starts and all shards share one pool of `procs` processes, so a slot
takes the next shard as soon as it frees instead of idling until the slowest shard of a fixed split ends. Rounds are
shard-invariant (phase 0), so the split changes wall time only. A round process starts in ~3 s against 35-50 s per
round, and OMP_NUM_THREADS stays 2: OMP 1 changed the outcome of 2 of 2 rounds (TODO 26.5.5, 2026-10-15)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SHARD_ROUNDS = 2


@dataclass(frozen=True)
class Job:
    spec: Path
    out: Path
    starts: tuple[str, ...] | None = None  # start labels (sonic_...); None = the whole pool
    fork: str | None = None  # bfm_round --fork PHASE:DELTA


def round_file(out: Path, label: str) -> Path:
    return out / f"{label.replace('sonic', 'bfm', 1)}.json"


def drop_torn(jobs: list[Job]) -> None:
    """A shard killed while saving can leave a half-written round JSON, which would count as done: remove it."""
    for job in jobs:
        for f in job.out.glob("bfm_*.json"):
            try:
                json.loads(f.read_text())
            except ValueError:
                f.unlink()
                f.with_suffix(".npz").unlink(missing_ok=True)


def run_jobs(
    jobs: list[Job],
    pool: Path,
    procs: int,
    log: Path,
    gpus: str = "1",
    script: Path | None = None,
    stop: threading.Event | None = None,
) -> dict:
    """Run the jobs' missing rounds on `pool` with at most `procs` processes spread over `gpus` ("0,1": each shard goes
    to the device running the fewest of this batch's processes). Setting `stop` kills the running shards and drops
    the rest (a side batch nobody needs any more). Returns {job index: labels still missing}."""
    devices = gpus.split(",")
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0")
    env.setdefault("TMPDIR", "/workspace/tmp")
    every = sorted(p.stem for p in pool.glob("*.npz"))
    todo = {k: [x for x in (job.starts or every) if not round_file(job.out, x).exists()] for k, job in enumerate(jobs)}
    queue = []
    for k, job in enumerate(jobs):
        n = math.ceil(len(todo[k]) / SHARD_ROUNDS)
        for i in range(n):
            cmd = [str(SIM_PY), str(script or HERE / "bfm_round.py"), str(pool), "--spec", str(job.spec),
                   "--out", str(job.out), "--only", ",".join(todo[k]), "--shard", f"{i}/{n}"]  # fmt: skip
            if job.fork:
                cmd += ["--fork", job.fork]
            queue.append(cmd)
    running: list[tuple[subprocess.Popen, str]] = []
    with log.open("a") as fh:
        while queue or running:
            if stop is not None and stop.is_set():
                for p, _ in running:
                    p.kill()
                    p.wait()
                drop_torn(jobs)
                break
            while queue and len(running) < procs:
                d = min(devices, key=lambda x: sum(g == x for _, g in running))
                p = subprocess.Popen(queue.pop(0), cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": d}, stdout=fh,
                                     stderr=subprocess.STDOUT)  # fmt: skip
                running.append((p, d))
            time.sleep(0.5)
            running = [(p, d) for p, d in running if p.poll() is None]
    return {k: [x for x in todo[k] if not round_file(jobs[k].out, x).exists()] for k in todo}
