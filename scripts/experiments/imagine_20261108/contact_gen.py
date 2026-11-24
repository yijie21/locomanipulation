"""Contact truth for the imagined rounds (TODO 45.3): what a real grip does on the grasps skills actually make.

`plan` reads an imagined batch: for every grip a round reached, the stance, every free body's pose, and the reaches
between the last walk and the grip (kin_motor's `grips` in `<label>.kin.json`, the round's `calls`). A grip driven by a
hand program has no reach to replay and is counted, not sampled. Distinct grasps are kept (same start and target to
5 mm), each with the object shifted by OFFSETS.

`run` replays each on the real Motor: reset at the stance, the free bodies where they were, the reaches, grip, the lift
the skill made (or 10 cm up), a 2 s hold. The replay starts with the waist and arms as they were when the walk ended.

    PYTHONPATH=src $VENV_SIM_PY contact_gen.py plan --batch <batch dir> --family grasp_floor --out <dir>
    PYTHONPATH=src $VENV_SIM_PY contact_gen.py run --jobs <dir>/jobs.json --out <dir> --procs 8 --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFFSETS = ((0.0, 0.0), (0.015, 0.0), (-0.015, 0.0), (0.0, 0.015), (0.0, -0.015))
LIFT_M = 0.10
HELD_RISE_M = 0.03
KNOCKED_M = 0.02
MOVES = {"walk_to", "turn", "sidestep"}


def setup_paths() -> None:
    for p in (HERE, ROOT / "scripts" / "rsi", ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def plan(a) -> int:
    setup_paths()
    import family as F

    fam = F.load(a.family)
    jobs, seen, why = [], set(), {}
    manifest = json.loads(Path(a.batch, "manifest.json").read_text())
    scene = json.loads((Path(fam["scene_dir"]) / "scene_meta.json").read_text())["objects"]

    def skip(k: str) -> None:
        why[k] = why.get(k, 0) + 1

    for j in manifest:
        for label in j["labels"]:
            kin_f, rec_f = Path(j["out"], f"{label}.kin.json"), Path(j["out"], f"{label}.json")
            if not kin_f.exists() or not rec_f.exists():
                skip("no record")
                continue
            kin, rec = json.loads(kin_f.read_text()), json.loads(rec_f.read_text())
            row = json.loads(Path(j["pool"], f"{label}.json").read_text())
            key = F.task(fam, row["task"]).objects[0]
            body = scene[key]["body"]
            for g in kin.get("grips", []):
                calls = rec["calls"]
                n = g["n_calls"]
                last = max([i for i in range(n) if calls[i]["name"] in MOVES], default=-1)
                reaches = [c for c in calls[last + 1 : n] if c["name"] in ("reach", "posture")]
                if not [c for c in reaches if c["name"] == "reach"]:
                    skip("hand program")
                    continue
                if body not in g["free"]:
                    skip("object not free")
                    continue
                target = reaches[-1]["args"][1] if reaches[-1]["name"] == "reach" else None
                if target is None:
                    skip("last call not a reach")
                    continue
                if isinstance(target[0], list):
                    skip("two hands")
                    continue
                pos = g["free"][body][:3]
                rel = [round(float(t - p), 3) for t, p in zip(target, pos)]
                ident = (a.family, row["task"], label, tuple(round(v * 200) / 200 for v in rel), g["side"])
                if ident in seen:
                    skip("duplicate")
                    continue
                seen.add(ident)
                after = next((c for c in calls[n + 1 :] if c["name"] == "reach"), None)
                for off in OFFSETS:
                    jobs.append(dict(id=f"{len(jobs):06d}", family=a.family, label=label, key=key, body=body, side=g["side"],
                                     pelvis=g["pelvis"], yaw=g["yaw"], free=g["free"], reaches=reaches, lift=after, joints=g["stance_joints"],
                                     offset=list(off), rel=rel, candidate=j["id"]))  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps(jobs))
    print(f"{len(jobs)} samples of {len(jobs) // len(OFFSETS)} distinct grasps; skipped {why}")
    return 0


def one(world, job: dict) -> dict:
    import mujoco
    import numpy as np

    m = world.motor
    t0 = time.time()
    body = job["body"]
    m.track = {body: m.sim.m.body(body).id}
    m.reset(job["pelvis"][:2], job["yaw"])
    # the waist and arms as they were when the walk ended (skills raise the hand before walking in): from the hanging
    # pose the hand would sweep through the table's edge
    q = m.sim.d.qpos[m.kp.qadr].copy()
    q[12:] = job["joints"][12:]
    m.sim.d.qpos[m.kp.qadr] = q
    mujoco.mj_forward(m.sim.m, m.sim.d)
    m.pol.reset(m.sim.state())
    m._stand()
    for name, pose in job["free"].items():
        pos = np.array(pose[:3], float)
        if name == body:
            pos[:2] += job["offset"]
        m.place(name, pos, np.array(pose[3:], float))
    obj = m.sim.m.body(body).id
    m.hold(1.0)
    start = m.sim.d.xpos[obj].copy()
    res = dict(id=job["id"], family=job["family"], label=job["label"], key=job["key"], side=job["side"], offset=job["offset"], rel=job["rel"],
               candidate=job["candidate"], reached=True)  # fmt: skip
    for c in job["reaches"]:
        out = getattr(m, c["name"])(*c["args"], **c["kwargs"])
        if m.fell_at is not None:
            break
    res["reach_err_cm"] = out.info.get("jaw_err_cm") if m.fell_at is None else None
    before = m.sim.d.xpos[obj].copy()
    res["knocked_cm"] = round(100 * float(np.linalg.norm(before[:2] - start[:2])), 2)
    if m.fell_at is None:
        g = m.grip(job["side"])
        res["grip_ok"], res["opening_cm"] = bool(g.ok), g.info.get("opening_cm")
    if m.fell_at is None:
        if job["lift"]:
            m.reach(*job["lift"]["args"], **job["lift"]["kwargs"])
        else:
            jaw, rot = m.jaw_centre(job["side"])
            m.reach(job["side"], jaw + np.array([0.0, 0.0, LIFT_M]), rot)
    if m.fell_at is None:
        m.hold(2.0)
    rise = float(m.sim.d.xpos[obj][2] - start[2])
    res.update(fell=m.fell_at is not None, rise_cm=round(100 * rise, 2), gripped=bool(m.gripping(obj)),
               held=bool(m.gripping(obj) and rise >= HELD_RISE_M and m.fell_at is None), knocked=res["knocked_cm"] >= 100 * KNOCKED_M,
               wall_s=round(time.time() - t0, 1))  # fmt: skip
    return res


def worker(args: tuple) -> int:
    gpu, jobs, out = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    setup_paths()
    import family as F
    import scene_round as SR

    out = Path(out)
    done = 0
    by_family: dict = {}
    for j in jobs:
        by_family.setdefault(j["family"], []).append(j)
    for family, js in by_family.items():
        world = SR.make_world(F.load(family), "scalebfm")
        for j in js:
            f = out / "rows" / f"{j['id']}.json"
            if f.exists():
                continue
            try:
                res = one(world, j)
            except Exception as e:  # noqa: BLE001 -- one replay that fails must not end the worker
                res = dict(id=j["id"], error=f"{type(e).__name__}: {e}"[:300])
            f.write_text(json.dumps(res))
            done += 1
    return done


def run(a) -> int:
    out = Path(a.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    jobs = [j for f in a.jobs.split(",") for j in json.loads(Path(f).read_text())]
    jobs = [j for j in jobs if not (out / "rows" / f"{j['id']}.json").exists()]
    if a.max:
        jobs = jobs[: a.max]
    gpus = a.gpus.split(",")
    tasks = [(gpus[i % len(gpus)], jobs[i :: a.procs], str(out)) for i in range(a.procs)]
    print(f"{len(jobs)} samples on {a.procs} processes", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as p:
        for i, n in enumerate(p.imap_unordered(worker, tasks, chunksize=1)):
            print(f"worker {i + 1} done {n} at {round(time.time() - t0)} s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--batch", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--jobs", required=True, help="comma-separated jobs.json files")
    r.add_argument("--out", required=True)
    r.add_argument("--procs", type=int, default=8)
    r.add_argument("--gpus", default="0,1")
    r.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    return dict(plan=plan, run=run)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
