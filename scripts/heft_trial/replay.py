"""TODO 51 stage A diagnostic: is the HEFT port or the Motor's reference the reason HEFT fails the walk and the large reaches?

ScaleBFM runs a clip; its measured whole-body trajectory (pelvis pose + 29 joints per 20 ms tick, stepping legs included) is then
given to HEFT as the reference of an identical start. A tracker that is ported right follows a whole-body reference that a robot
of this model has itself performed. Run with the sim venv: PYTHONPATH=src <sim_python> scripts/heft_trial/replay.py --out DIR."""

import argparse
import dataclasses
import json
import sys
from multiprocessing.connection import Client
from pathlib import Path

import clips as CL  # noqa: E402  (this folder)
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Pose
from locomani.bench.simple.session import Session, take_slot


def replay_ref(arr, t0):
    t, robot, joints = arr["t"], arr["robot"].astype(float), arr["joints"].astype(float)

    def ref(now):
        x = float(np.clip(now - t0, t[0], t[-1]))
        i = int(np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2))
        u = (x - t[i]) / max(t[i + 1] - t[i], 1e-9)
        pos = robot[i, :3] * (1 - u) + robot[i + 1, :3] * u
        q0, q1 = robot[i, 3:7], robot[i + 1, 3:7]
        q1 = q1 if q0 @ q1 >= 0 else -q1
        q = q0 * (1 - u) + q1 * u
        return Pose(pos, q / np.linalg.norm(q), joints[i] * (1 - u) + joints[i + 1] * u, np.zeros(29))

    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode", default="OpenOvenTeleop/Q/0")
    ap.add_argument("--clips", default="walk,reach_low,reach_both")
    ap.add_argument("--slots", default="3")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(
        CL.SimpleConfig(), run_dir=str(out / "run"), slots=tuple(int(s) for s in a.slots.split(",")), lazy_render=True
    )
    ids = out / "ids.txt"
    ids.write_text(a.episode + "\n")
    ses = Session(cfg, "OpenOvenTeleop", ids, out / "rows.jsonl", stereo=False, look=False)
    ses.slot, lock = take_slot(cfg, 600)
    ses._locks.append(lock)
    try:
        if not ses.start_server():
            return 2
        bed = CL.Bed(cfg, ses.slot, Client(str(ses.sock("isaac")), family="AF_UNIX"))
        for name in a.clips.split(","):
            # 1. ScaleBFM performs the clip; keep its tick trace from the clip's start
            sim, mo, info = bed.episode("scalebfm", a.episode)
            n0 = len(mo.trace)
            rec = CL.Rec(sim, mo)
            p0, q0 = mo.pelvis()
            CL.CLIPS[name](mo, rec, p0, R.yaw_of(q0), {})
            arr = {k: np.array([r[k] for r in mo.trace[n0:]], dtype=np.float32) for k in ("t", "robot", "joints")}
            t_start = arr["t"][0]
            # 2. HEFT is given that trajectory as its reference from an identical start
            sim2, mo2, _ = bed.episode("heft", a.episode)
            rec2 = CL.Rec(sim2, mo2)
            ref = replay_ref(arr, t_start - mo2.t)  # arr["t"] is ScaleBFM's clock: shift to this Motor's
            mo2.ref = lambda t, f=ref: f(t)
            dur = float(arr["t"][-1] - arr["t"][0])
            o = mo2.hold(dur)
            n = len(rec2.rows)
            p = np.array([r["p"] for r in rec2.rows])
            want = np.array([ref(r["t"]).pos for r in rec2.rows])
            res = dict(clip=name, seconds=round(dur, 1), ok=o.ok, why=o.why, fell=mo2.fell_at is not None,
                       pelvis_err_cm_mean=round(float(100 * np.linalg.norm(p[:, :2] - want[:, :2], axis=1).mean()), 1),
                       pelvis_err_cm_end=round(float(100 * np.linalg.norm(p[-1, :2] - want[-1, :2])), 1),
                       pelvis_z_err_cm_mean=round(float(100 * np.abs(p[:, 2] - want[:, 2]).mean()), 1), ticks=n,
                       scalebfm_travel_cm=round(float(100 * np.linalg.norm(arr["robot"][-1, :2] - arr["robot"][0, :2])), 1))  # fmt: skip
            print(json.dumps(res), flush=True)
            with open(out / "replay.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
    finally:
        ses.stop_all()
        for f in ses._locks:
            f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
