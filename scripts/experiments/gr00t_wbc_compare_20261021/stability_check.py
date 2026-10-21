"""TODO 34.3: how steady and how smooth each controller is through our motor layer, on the flat walk_mars floor -- the
question behind rhoyn's walk-under-disturbance board (user 2026-10-21: "会比 scalebfm 更稳定更流畅吗"), asked of our
robot (mars G1 + Dex1) and our primitives rather than the board's harness.

Cases, each from a fresh reset with a jittered start (xy +-5 cm, yaw +-3 deg), N seeds:
  stand30    hold 30 s: how far the feet slide apart, how far the pelvis drifts
  walk3      walk_to 3 m ahead (the motor layer's 0.5 m/s): time, stop error, and while walking the pelvis's vertical
             acceleration RMS, the torso's angular-rate RMS, the pelvis's lateral sway about the path, and how far a
             foot slides while it is on the floor (foot below 4 cm and slower than 0.3 m/s vertically)
  push_stand_F / push_walk_F
             a horizontal push of F N (100 / 200 / 300) on the torso for 0.08 s in a random direction, 1.5 s into a
             hold or 1.5 s into a 3 m walk; fell = a body other than the feet on the floor or the pelvis below 0.3 m
             within 4 s of the push
Writes outputs/gr00t_wbc_compare_20261021/stability/<controller>__<case>.jsonl (one row per run; one process per
case can run beside the others) and, with --summary, summary.json over every file.

    PYTHONPATH=src $PY scripts/experiments/gr00t_wbc_compare_20261021/stability_check.py --controllers scalebfm gr00t_wbc \\
        gr00t_wbc_h066_p012 [--seeds 10] [--cases stand30 walk3 push_stand_200 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts/rsi"), str(REPO / "src")]

import motor_check as MC  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

OUT = REPO / "outputs/gr00t_wbc_compare_20261021/stability"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
CASES = ["stand30", "walk3"] + [f"push_{w}_{f}" for w in ("stand", "walk") for f in (100, 200, 300)]


def policy_for(ctrl: str):
    if ctrl.startswith("gr00t_wbc"):
        from locomani.backends.gr00t_wbc.runtime import Gr00tPolicy

        return Gr00tPolicy()
    return R.Policy("cuda", ScaleBfmBackendConfig())


class Recorder:
    """Wraps sim.step: logs feet, pelvis and torso every tick, applies a scheduled push."""

    def __init__(self, m):
        self.m, sim = m, m.sim
        self.feet = [sim.m.body(n).id for n in FEET]
        self.torso = sim.m.body("torso_link").id
        self.rows, self.push = [], None  # push: (t0, t1, force xyz)
        self._step = sim.step
        sim.step = self.step

    def step(self, targets):
        sim, t = self.m.sim, self.m.t
        if self.push and self.push[0] <= t < self.push[1]:
            sim.d.xfrc_applied[self.torso, :3] = self.push[2]
        else:
            sim.d.xfrc_applied[self.torso, :3] = 0.0
        self._step(targets)
        d = sim.d
        self.rows.append(np.r_[t, d.qpos[:3], d.qvel[:3], d.cvel[self.torso][:3], d.xpos[self.feet[0]], d.xpos[self.feet[1]],
                               d.cvel[self.feet[0]][3:], d.cvel[self.feet[1]][3:]])  # fmt: skip

    def array(self):
        return np.array(self.rows)


def fallen(m) -> bool:
    return bool(m.sim.fallen()) or m.sim.d.qpos[2] < 0.3


def walk_metrics(a: np.ndarray, t0: float, t1: float, y_path: float) -> dict:
    w = a[(a[:, 0] >= t0) & (a[:, 0] < t1)]
    if len(w) < 5:
        return {}
    dt = np.diff(w[:, 0]).mean()
    vz = w[:, 6]
    az = np.diff(vz) / dt
    slip = 0.0
    for f in (0, 1):
        pos = w[:, 10 + 3 * f : 13 + 3 * f]
        vel = w[:, 16 + 3 * f : 19 + 3 * f]
        on = (pos[:, 2] < 0.04) & (np.abs(vel[:, 2]) < 0.3)
        step = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
        slip += float(step[on[1:] & on[:-1]].sum())
    return dict(acc_z_rms=round(float(np.sqrt(np.mean(az**2))), 3),
                ang_rate_rms=round(float(np.sqrt(np.mean(np.sum(w[:, 7:10] ** 2, axis=1)))), 3),
                sway_cm=round(100 * float(np.std(w[:, 2] - y_path)), 2), foot_slide_cm=round(100 * slip, 1))  # fmt: skip


def run(ctrl: str, case: str, seed: int, pol) -> dict:
    rng = np.random.default_rng(seed)
    MC.CONTROLLER = ctrl
    m = MC.make(pol, MC.WALK)
    xy = rng.uniform(-0.05, 0.05, 2)
    yaw = float(np.radians(rng.uniform(-3, 3)))
    m.reset(xy, yaw)
    rec = Recorder(m)
    out = dict(controller=ctrl, case=case, seed=seed)
    feet0 = np.linalg.norm(m.sim.d.xpos[rec.feet[0]][:2] - m.sim.d.xpos[rec.feet[1]][:2])
    p0 = m.pelvis()[0].copy()
    if case == "stand30":
        o = m.hold(30.0)
        feet1 = np.linalg.norm(m.sim.d.xpos[rec.feet[0]][:2] - m.sim.d.xpos[rec.feet[1]][:2])
        out.update(ok=o.ok, fell=fallen(m) or (not o.ok and o.why.startswith("fell")), feet_spread_cm=round(100 * float(feet1 - feet0), 1),
                   pelvis_drift_cm=round(100 * float(np.linalg.norm(m.pelvis()[0][:2] - p0[:2])), 1))  # fmt: skip
        return out
    goal = p0[:2] + R.rot_z(yaw)[:2, :2] @ np.array([3.0, 0.0])
    if case == "walk3":
        t0 = m.t
        o = m.walk_to(goal, yaw)
        a = rec.array()
        # sideways from the straight line start -> goal, in world y after rotating by -yaw
        c, s = np.cos(-yaw), np.sin(-yaw)
        lat = np.array([[c, -s], [s, c]]) @ (a[:, 1:3] - p0[:2]).T
        a2 = a.copy()
        a2[:, 2] = lat[1]
        out.update(ok=o.ok, fell=fallen(m) or (not o.ok and o.why.startswith("fell")), seconds=round(o.t1 - o.t0, 2),
                   stop_err_cm=o.info.get("stop_err_cm"), yaw_err_deg=o.info.get("yaw_err_deg"),
                   **walk_metrics(a2, t0 + 0.5, o.t1, 0.0))  # fmt: skip
        return out
    _, where, force = case.split("_")
    ang = float(rng.uniform(-np.pi, np.pi))
    F = float(force) * np.array([np.cos(ang), np.sin(ang), 0.0])
    t_push = m.t + 1.5
    rec.push = (t_push, t_push + 0.08, F)
    o = m.hold(5.5) if where == "stand" else m.walk_to(goal, yaw)
    if o.ok and where == "walk":
        o = m.hold(max(0.0, t_push + 4.0 - m.t))
    out.update(ok=o.ok, fell=fallen(m) or (not o.ok and o.why.startswith("fell")), push_deg=round(float(np.degrees(ang)), 1),
               pelvis_moved_cm=round(100 * float(np.linalg.norm(m.pelvis()[0][:2] - p0[:2])), 1), why=o.why)  # fmt: skip
    return out


def summarise(rows: list[dict]) -> dict:
    out = {}
    for case in sorted({r["case"] for r in rows}):
        rs = [r for r in rows if r["case"] == case]
        s = dict(n=len(rs), falls=sum(bool(r["fell"]) for r in rs))
        for k in ("feet_spread_cm", "pelvis_drift_cm", "seconds", "stop_err_cm", "acc_z_rms", "ang_rate_rms", "sway_cm",
                  "foot_slide_cm", "pelvis_moved_cm"):  # fmt: skip
            v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            if v:
                s[k] = dict(median=round(float(np.median(v)), 3), max=round(float(np.max(v)), 3))
        out[case] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controllers", nargs="+", default=["scalebfm", "gr00t_wbc", "gr00t_wbc_h066_p012"])
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--summary", action="store_true", help="only summarise what the files hold")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ctrl in a.controllers:
        pol = policy_for(ctrl) if not a.summary else None
        t0 = time.time()
        for case in a.cases:
            if a.summary:
                break
            f = OUT / f"{ctrl}__{case}.jsonl"
            done = {r["seed"] for r in map(json.loads, f.read_text().splitlines())} if f.exists() else set()
            for seed in range(a.seeds):
                if seed in done:
                    continue
                r = run(ctrl, case, seed, pol)
                with f.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
        rows = [json.loads(x) for f in sorted(OUT.glob(f"{ctrl}__*.jsonl")) for x in f.read_text().splitlines()]
        summary[ctrl] = summarise([r for r in rows if r["case"] in a.cases])
        print(ctrl, round(time.time() - t0), "s", json.dumps(summary[ctrl]), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
