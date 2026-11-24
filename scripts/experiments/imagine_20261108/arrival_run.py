"""Run drafts on the imagined Motor with arrival errors, K times per start (TODO 45.8).

    arrival_run.py train <out> --k 4 [--no-noise] [--no-fit] [--limit N]    # the 41 packages recorded before shadow mode
    arrival_run.py test  <out> --k 8 [--no-noise] [--no-fit]                # the 17 shadow drafts and their incumbent

Writes <out>/manifest.json and <out>/<id>/k<i>/<label>.json + .kin.json. The rounds already written are kept.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import re
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imagine as IM  # noqa: E402

FAMILY = "grasp_floor"
BASE = IM.ROOT / "outputs" / "scene_rsi_20261016" / "imagine_20261108"
POOL = BASE / "arrival_v1" / "pool.npz"
CHUNKS = 2


def train_jobs() -> list[dict]:
    out = []
    for j in json.loads((BASE / "v7_kin_gf" / "manifest.json").read_text()):
        m = re.search(r"/([^/]+)/outputs/.*/loop/(iter_\d+)/", j["pkg"])
        group = None if j["incumbent"] or m is None else f"{m.group(1)} {m.group(2)}"
        out.append(dict(id=j["id"], pkg=j["pkg"], set=j["set"], pool=j["pool"], labels=j["labels"], real=j["real"], group=group, incumbent=j["incumbent"]))  # fmt: skip
    return out


def test_jobs() -> list[dict]:
    out, seen = [], set()
    for sb in sorted(glob.glob("/workspace/code/rsi-scene/sh-single-gf?-r?")):
        fam = Path(sb) / "outputs" / "scene_rsi_20261016" / FAMILY
        for f in sorted(
            (Path(sb) / "outputs" / "scene_rsi_20261016_hidden" / FAMILY / "shadow").glob("iter_*/candidates.json")
        ):
            labels = json.loads((f.parent / "shadow.json").read_text())["labels"]
            for c in json.loads(f.read_text()):
                if c["incumbent"] and c["hash"] in seen:
                    continue
                seen.add(c["hash"])
                kind = "incumbent" if c["incumbent"] else "candidate"
                out.append(dict(id=f"{Path(sb).name}-{f.parent.name}-{c['number']}", pkg=str(Path(c["dir"]) / "skills"), set=c["set"],
                                pool=str((fam / "pool_diag").resolve()), labels=labels, real=str(fam / "diag" / f"{kind}_{c['hash']}"),
                                group=None if c["incumbent"] else f"{Path(sb).name} {f.parent.name}", incumbent=bool(c["incumbent"])))  # fmt: skip
    return out


def work(a: tuple) -> str:
    job, labels, root, k, noise, fit = a
    IM.setup_paths()
    import arrival_motor as AM
    import family as F
    import scene_round as SR

    fam = F.load(FAMILY)
    skills = SR.load_skills(Path(job["pkg"]))
    skills.apply_set(skills, job["set"])
    h = SR.skills_hash(Path(job["pkg"]))
    world = AM.make_world(fam, str(POOL) if noise else None, noise, fit)
    for label in labels:
        for i in range(k):
            out = Path(root) / job["id"] / f"k{i}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / f"{label}.json").exists():
                continue
            AM.ArrivalMotor.round_seed = zlib.crc32(f"{label}|{i}".encode())  # the same draws for every draft
            t0 = time.time()
            SR.run_round(fam, world, skills, h, Path(job["pool"]), out, label, job["set"], False, "scalebfm")
            log = dict(wall_s=round(time.time() - t0, 2), events=world.motor.kin_log, grip_pen=world.motor.grip_pen)
            (out / f"{label}.kin.json").write_text(json.dumps(log))
    return job["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_", choices=["train", "test"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N packages and 4 starts each (a smoke run)")
    a = ap.parse_args()
    jobs = train_jobs() if a.set_ == "train" else test_jobs()
    if a.limit:
        jobs = [dict(j, labels=j["labels"][:4]) for j in jobs[: a.limit]]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manifest.json").write_text(
        json.dumps(dict(jobs=jobs, k=a.k, noise=not a.no_noise, fit=not a.no_fit), indent=1)
    )
    k = 1 if a.no_noise else a.k
    tasks = [
        (j, j["labels"][c::CHUNKS], str(a.out), k, not a.no_noise, not a.no_fit)
        for j in jobs
        for c in range(CHUNKS)
        if j["labels"][c::CHUNKS]
    ]
    print(len(jobs), "packages,", len(tasks), "tasks, k =", k, flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as p:
        for i, jid in enumerate(p.imap_unordered(work, tasks, chunksize=1)):
            if i % 10 == 0:
                print(i + 1, jid, round(time.time() - t0), flush=True)
    print("done", round(time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
