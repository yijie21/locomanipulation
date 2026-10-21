"""TODO 27.2.4: round batches for the scene loop and its gate (the warehouse loop's batch.py, for scene_round.py): every
job -- one skills directory with its overrides, on some starts of a pool -- puts its missing rounds into one queue, and
`procs` worker processes spread over the GPUs take rounds from it one at a time until it is empty (TODO 32.2: a worker
loads the scene and the policy once, about 13 s, where a process per two rounds paid it every 80 s of simulation).
Rounds already written under the current success rule are skipped (a batch resumes); a round judged under an older
rule is set aside as `_rule<n>_<label>.*` and run again.

Every worker also holds one of SIM_SLOTS machine-wide slots (a locked file under SLOTS_DIR) while it runs, so batches
from several sandboxes at once -- two families' loops side by side (TODO 27.10.7, plan §12.6) -- share the machine
instead of each taking `procs` (12 processes of OMP 2 already fill the container's 15.36 cores). The lock goes with
the process holding it. A round's outcome does not depend on the schedule: the 2026-10-19 regression re-ran every
family's sealed pool with 2 processes against records made with 12, identical; TODO 32.2 re-ran grasp_atomic's sealed
pool from queue workers against records made two rounds per process, identical."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SIM_SLOTS = int(os.environ.get("RSI_SIM_SLOTS", "12"))  # simulation processes across all sandboxes at once; 0: no cap
SLOTS_DIR = Path(os.environ.get("RSI_SIM_SLOTS_DIR", "/workspace/tmp/rsi_sim_slots"))
# 1: held(X) = a finger touches it, off its surface (to 2026-10-16); 2: both fingers of one hand, or both hands (TODO 27.9)
SUCCESS_RULE = 2


@dataclass(frozen=True)
class Job:
    skills: Path
    out: Path
    overrides: dict | None = None
    labels: tuple[str, ...] | None = None  # None = the whole pool
    npz: bool = False
    controller: str = "scalebfm"  # scene_round --controller (TODO 28: "amo")


def done(f: Path) -> bool:
    """A round file judged under the current success rule."""
    if not f.exists():
        return False
    try:
        return json.loads(f.read_text()).get("rule", 1) == SUCCESS_RULE
    except (OSError, ValueError):
        return False


def retire(f: Path) -> None:
    """Keep an older rule's round beside the new one, out of `outcomes` (underscore names are skipped)."""
    if not f.exists():
        return
    rule = json.loads(f.read_text()).get("rule", 1)
    for p in (f, f.with_suffix(".npz")):
        if p.exists():
            p.rename(p.with_name(f"_rule{rule}_{p.name}"))


def take_slot():
    """A free machine-wide slot (its locked file, held open), or None when all SIM_SLOTS are taken."""
    SLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(SIM_SLOTS):
        fh = open(SLOTS_DIR / f"slot_{i:02d}", "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            fh.close()
    return None


def pool_labels(pool: Path) -> list[str]:
    return sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))


def run_jobs(
    jobs: list[Job],
    family: str,
    pool: Path,
    procs: int,
    log: Path,
    gpus: str = "1",
    stop: threading.Event | None = None,
) -> dict:
    """Run the jobs' missing rounds; returns {job index: labels still missing}. Setting `stop` kills the running
    shards and drops the rest."""
    devices = gpus.split(",")
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0")
    env.setdefault("TMPDIR", "/workspace/tmp")
    every = pool_labels(pool)
    todo = {k: [x for x in (job.labels or every) if not done(job.out / f"{x}.json")] for k, job in enumerate(jobs)}
    for k, job in enumerate(jobs):
        for x in todo[k]:
            retire(job.out / f"{x}.json")
    q = make_queue(jobs, todo, Path(env["TMPDIR"]))
    cmd = [str(SIM_PY), str(HERE / "scene_round.py"), family, "--pool", str(pool), "--queue", str(q)]
    running: list[tuple[subprocess.Popen, str, object]] = []  # (worker, GPU, its slot)
    failed = 0  # workers that died: a worker that cannot start would otherwise be relaunched for ever
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("a") as fh:
            while True:
                if stop is not None and stop.is_set():
                    for p, _, slot in running:
                        p.kill()
                        p.wait()
                        if slot is not None:
                            slot.close()
                    break
                left = queued(q) if failed < 2 * procs else 0
                while left and len(running) < min(procs, left):
                    slot = take_slot() if SIM_SLOTS else None
                    if SIM_SLOTS and slot is None:  # another sandbox's batch holds every slot
                        break
                    d = min(devices, key=lambda x: sum(g == x for _, g, _ in running))
                    p = subprocess.Popen(cmd, cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": d}, stdout=fh,
                                         stderr=subprocess.STDOUT)  # fmt: skip
                    running.append((p, d, slot))
                if not running and not left:
                    break
                time.sleep(0.5)
                for p, _, slot in running:
                    if p.poll() is not None:
                        failed += p.returncode != 0
                        if slot is not None:
                            slot.close()
                running = [(p, d, slot) for p, d, slot in running if p.poll() is None]
    finally:
        shutil.rmtree(q, ignore_errors=True)
    return {k: [x for x in todo[k] if not done(jobs[k].out / f"{x}.json")] for k in todo}


def make_queue(jobs: list[Job], todo: dict, tmp: Path) -> Path:
    """A queue directory for scene_round.py's workers: jobs.json, and one file per missing round under todo/ (job
    order, then label order); a worker takes a round by renaming its file into taken/ (claim_round)."""
    tmp.mkdir(parents=True, exist_ok=True)
    q = Path(tempfile.mkdtemp(prefix="rsi_queue_", dir=tmp))
    (q / "todo").mkdir()
    (q / "taken").mkdir()
    spec = [dict(skills=str(j.skills), out=str(j.out), overrides=j.overrides or {}, npz=j.npz, controller=j.controller)
            for j in jobs]  # fmt: skip
    (q / "jobs.json").write_text(json.dumps(spec))
    n = 0
    for k in sorted(todo):
        for x in todo[k]:
            (q / "todo" / f"{n:06d}.json").write_text(json.dumps(dict(job=k, label=x)))
            n += 1
    return q


def queued(q: Path) -> int:
    return sum(1 for _ in (q / "todo").iterdir())


def claim_round(q: Path) -> dict | None:
    """The next round in the queue, taken (renamed into taken/, which only one process can do), or None when empty."""
    for f in sorted((q / "todo").iterdir()):
        try:
            os.rename(f, q / "taken" / f"{f.stem}_{os.getpid()}.json")
        except FileNotFoundError:  # another worker took it
            continue
        return json.loads((q / "taken" / f"{f.stem}_{os.getpid()}.json").read_text())
    return None


def outcomes(out: Path) -> dict:
    """{label: round record} of a job's directory."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if not p.name.startswith("_")}


def main(argv=None) -> int:
    """`batch.py <family> <pool> <skills> <out> [--set JSON] [--procs N] [--gpus 0,1] [--npz]`: one job's missing rounds,
    then `<out>/_done.json` (missing labels). Long batches run detached (nohup setsid): a foreground tool call that
    times out takes the running shards with it."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("family")
    ap.add_argument("pool", type=Path)
    ap.add_argument("skills", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--set", default="{}")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--npz", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "_done.json").unlink(missing_ok=True)
    job = Job(a.skills.resolve(), a.out.resolve(), json.loads(a.set), None, a.npz)
    missing = run_jobs([job], a.family, a.pool.resolve(), a.procs, a.out / "_batch.log", a.gpus)
    (a.out / "_done.json").write_text(json.dumps(dict(missing=missing[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
