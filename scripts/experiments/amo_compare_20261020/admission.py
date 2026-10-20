"""TODO 28.1: AMO admission on the mars G1 (the ScaleBFM line's W0, docs/handoff/controller-compare-20261013.md §3).

Ten jittered runs (start yaw +-3 deg, xy +-5 cm) of each item on the flat walk_mars floor, from AMO's default pose
(every joint 0: legs straight, arms hanging), arms held there:
  stand      30 s standing
  walk       0.3 m/s forward for 5 s (about 1.5 m), then stop and stand 3 s
  turn       target heading +90 deg, stepping in place, 6 s
  squat_H    base height command H (0.65 / 0.55 / 0.45 m) for 6 s, then back to 0.75 for 3 s
  lean_P     torso pitch command P (0.5 / 1.0 rad) at height 0.65 for 6 s, then upright 3 s
A run falls when a robot body other than the feet touches the floor, or the pelvis drops below 0.2 m.
Writes outputs/amo_compare_20261020/admission/summary.json (+ one trace npz per item, run 0).

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/amo_compare_20261020/admission.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from locomani.backends.amo import runtime as A
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.motor import lowest_foot_point
from locomani.paths import REPO as _REPO

REPO = Path(_REPO)

WALK = REPO / "outputs/scene/walk_mars"
OUT = REPO / "outputs/amo_compare_20261020/admission"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def robot_bodies(m) -> set[int]:
    root = m.body("pelvis").id
    return {b for b in range(m.nbody) if b == root or _ancestor(m, b, root)}


def _ancestor(m, b, root) -> bool:
    while b > 0:
        b = int(m.body_parentid[b])
        if b == root:
            return True
    return False


def floor_contact(m, d, robot, feet) -> str | None:
    for i in range(d.ncon):
        b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
        for a, b in ((b1, b2), (b2, b1)):
            if a in robot and a not in feet and b not in robot and b == 0:
                return m.body(a).name
    return None


def plan(item: str):
    """t -> Command for the item, and its duration."""
    if item == "stand":
        return (lambda t, y0: A.Command(yaw=y0)), 30.0
    if item == "walk":
        return (lambda t, y0: A.Command(vx=0.3 if 1.0 <= t < 6.0 else 0.0, yaw=y0)), 9.0
    if item == "turn":
        return (lambda t, y0: A.Command(yaw=y0 + (np.pi / 2 if t >= 1.0 else 0.0), turning=1.0 <= t < 7.0)), 8.0
    kind, val = item.split("_")
    v = float(val)
    if kind == "squat":
        return (lambda t, y0: A.Command(yaw=y0, height=v if 1.0 <= t < 7.0 else A.HEIGHT0)), 10.0
    if kind == "lean":
        return (lambda t, y0: A.Command(yaw=y0, height=0.65 if t >= 1.0 else A.HEIGHT0,
                                        torso_pitch=v if 1.0 <= t < 7.0 else 0.0)), 10.0  # fmt: skip
    raise ValueError(item)


def run(item: str, seed: int, sim: A.AmoSim, pol: A.AmoPolicy, robot, feet):
    rng = np.random.default_rng(seed)
    y0 = float(np.deg2rad(rng.uniform(-3, 3)))
    xy = rng.uniform(-0.05, 0.05, 2)
    sim.reset(xy, y0, 0.0)
    sim.d.qpos[sim.qadr] = 0.0  # AMO's default pose; from ScaleBFM's bent-knee pose it walks off (2026-10-20)
    mujoco.mj_forward(sim.m, sim.d)
    sim.d.qpos[2] += 0.005 - lowest_foot_point(sim.m, sim.d, feet)  # Motor.reset: feet 5 mm above the floor
    mujoco.mj_forward(sim.m, sim.d)
    arms = sim.d.qpos[sim.qadr][A.ARMS].copy()
    pol.reset()
    cmd_of, T = plan(item)
    torso = sim.m.body("torso_link").id
    rows, fell = [], None
    for k in range(int(T / A.DT)):
        t = k * A.DT
        st = sim.state()
        sim.step(pol(st, cmd_of(t, y0), arms))
        who = floor_contact(sim.m, sim.d, robot, feet)
        if who or sim.d.qpos[2] < 0.2:
            fell = dict(t=round(t, 2), body=who or "pelvis low")
            break
        rows.append(np.r_[t, sim.d.qpos[:7], sim.d.xquat[torso], A.euler(sim.d.xquat[torso])])
    tr = np.array(rows)
    pos = tr[:, 1:4]
    out = dict(item=item, seed=seed, fell=fell, t_end=float(tr[-1, 0]))
    out["xy_travel_m"] = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
    out["yaw_end_deg"] = float(np.rad2deg(R.wrap(R.yaw_of(tr[-1, 4:8]) - y0)))
    out["pelvis_z_min"] = float(pos[:, 2].min())
    out["torso_pitch_max_deg"] = float(np.rad2deg(tr[:, -2].max()))
    still = tr[tr[:, 0] >= tr[-1, 0] - 2.0]  # last 2 s
    out["drift_last2s_m"] = float(np.linalg.norm(still[-1, 1:3] - still[0, 1:3]))
    return out, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", default=["stand", "walk", "turn", "squat_0.65", "squat_0.55", "squat_0.45",
                                                   "lean_0.5", "lean_1.0"])  # fmt: skip
    ap.add_argument("--runs", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sim = A.AmoSim(ScaleBfmBackendConfig(scene_dir=str(WALK), object_name=""))
    pol = A.AmoPolicy("cuda")
    robot = robot_bodies(sim.m)
    feet = {sim.m.body(n).id for n in FEET}
    summary = {}
    for item in a.items:
        res = []
        for s in range(a.runs):
            r, tr = run(item, s, sim, pol, robot, feet)
            res.append(r)
            if s == 0:
                np.save(OUT / f"{item}_trace.npy", tr)
        falls = sum(r["fell"] is not None for r in res)
        keys = [k for k in res[0] if isinstance(res[0][k], float)]
        summary[item] = dict(falls=falls, runs=a.runs, **{k: [round(r[k], 3) for r in res] for k in keys},
                             fell=[r["fell"] for r in res if r["fell"]])  # fmt: skip
        print(item, "falls", falls, {k: round(float(np.median(summary[item][k])), 3) for k in keys}, flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
