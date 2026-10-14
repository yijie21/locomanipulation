"""Walk to a finish line on the flat walk scene and stop -- ScaleBFM's side of the brake comparison (TODO 24.7).

Same protocol as the SONIC diagnostic (`locomani.backends.sonic.diag_brake`, 2026-09-23): reset at the scene's
robot_start facing +x, settle standing, walk straight at a commanded pelvis speed, and read where the robot stands
--settle simulated seconds after the stop. Two stop programs, one per run:

* `marker` -- the analogue of SONIC's "IDLE at the marker": the pelvis reference walks at the speed, and the tick the
  measured pelvis has advanced `marker` metres the reference is frozen at the measured pelvis (stay where you are).
  `carry` = settled pelvis minus the pelvis at that tick, as on the SONIC side.
* `plan` -- ScaleBFM's native stop, the analogue of SONIC's guard brake (never cross the line): the pelvis reference
  follows a trapezoid (accelerate at --acc, cruise at the speed, decelerate at --acc) that comes to rest with the
  body's frontmost low point --margin before the line, i.e. the pelvis target is line - toe reach - margin, the toe
  reach measured on the settled stance of the same trial. `brake` is the tick the reference starts decelerating,
  and the stop is read --settle seconds after the reference has come to rest.

ScaleBFM's sim is deterministic, so each repeat jitters the run-up (--jitter, uniform, seeded): the line moves a few
centimetres and the gait phase at the stop changes. results.json and trace/<label>_state_trace.npz (qpos at 50 Hz)
follow diag_brake's layout, so `scripts/experiments/render_brake_marker_20260923.py` renders both sides.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/wbc_compare_20261012/bfm_brake.py --program plan --out outputs/wbc_compare_20261012/brake/bfm_plan
Touch <out>/STOP to end after the current trial; results.json is rewritten after every trial.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.primitives.foot_watch import OBSTACLE_TOP

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WALK_SCENE = REPO / "outputs" / "scene" / "walk"
HZ = int(round(1 / R.DT))  # 50 policy steps per simulated second = the SONIC sim's privileged-state rate
# TODO 24.7.1: how far the plan program settled short of its reference end per speed (bfm_plan, 10-13, 10 trials
# each: 1.6 / 3.6 / 4.5 cm); --ff speed aims this far past the target, interpolated in speed
FF_SPEEDS, FF_SHORT = (0.25, 0.35, 0.55), (0.045, 0.036, 0.016)
STILL_V = 0.03  # m/s over 0.5 s: the pelvis counts as still (diag_brake's waypoint rule)
PHASE_LABELS = {
    "marker": dict(walk="行走 参考前进", brake="刹车 参考冻结在骨盆处", slow="减速"),
    "plan": dict(walk="行走 参考前进", brake="减速 参考停到线前", slow="减速"),
}


def frontmost_low(m, d, robot, origin, h, top=OBSTACLE_TOP) -> float:
    """Farthest point along h (from origin) of any robot geom whose world AABB reaches below `top`
    (primitives.foot_watch.FootWatch.frontmost_low on the sim's own data)."""
    best = -np.inf
    for g in robot:
        rot = d.geom_xmat[g].reshape(3, 3)
        centre = d.geom_xpos[g] + rot @ m.geom_aabb[g, :3]
        half = np.abs(rot) @ m.geom_aabb[g, 3:]
        if centre[2] - half[2] < top:
            best = max(best, float((centre[:2] - origin) @ h + np.abs(h) @ half[:2]))
    return best


class Profile:
    """Distance along the heading vs time since the walk started: accelerate at `acc` to `v`, cruise, and (when
    `dist` is given) decelerate at `acc` to rest exactly at `dist`; `dist` None = cruise forever."""

    def __init__(self, v: float, acc: float, dist: float | None):
        self.v, self.acc, self.dist = v, acc, dist
        ta = v / acc
        da = 0.5 * acc * ta * ta
        if dist is None:
            self.vp, self.ta, self.tc = v, ta, np.inf
        elif dist <= 2 * da:  # too short to reach v: a triangle
            self.vp = float(np.sqrt(acc * max(dist, 0.0)))
            self.ta, self.tc = self.vp / acc, 0.0
        else:
            self.vp, self.ta, self.tc = v, ta, (dist - 2 * da) / v
        self.t_dec = self.ta + self.tc  # the deceleration starts
        self.t_end = self.t_dec + self.vp / acc  # at rest

    def s(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if t <= self.ta:
            return 0.5 * self.acc * t * t
        da = 0.5 * self.acc * self.ta * self.ta
        if t <= self.t_dec:
            return da + self.vp * (t - self.ta)
        if t >= self.t_end:
            return float(self.dist)
        u = self.t_end - t
        return float(self.dist) - 0.5 * self.acc * u * u


def trial(
    pol, sim, robot, speed, marker, program, acc, margin, settle, timeout_s, label, trace_dir, ff=0.0, odo=None
) -> dict:
    start = sim.smeta.get("robot_start", {})
    xy0 = np.asarray(start.get("xy", [0.0, 0.0]), float)
    yaw0 = float(np.radians(start.get("yaw_deg", 0.0)))
    h = np.array([np.cos(yaw0), np.sin(yaw0)])
    n = np.array([-h[1], h[0]])
    sim.reset(xy0, yaw0, 0.0)
    if odo is not None:
        mujoco.mj_forward(sim.m, sim.d)
        odo.reset(sim.d.qpos[:3].copy(), sim.d.qpos[odo.qadr].copy(), sim.d.qpos[3:7].copy())
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    pel_q = np.array([np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)])
    st8 = dict(prof=None, hold=None)  # the walk profile (set when the walk starts) / the frozen pelvis xy

    def pelvis_ref(t):
        if st8["hold"] is not None:
            return st8["hold"]
        if st8["prof"] is None:
            return xy0
        return xy0 + h * st8["prof"].s(t - R.SETTLE_S)

    def ref(t):
        p = pelvis_ref(t)
        pel = np.array([p[0], p[1], R.PELVIS_Z])
        out = {"pelvis": (pel, pel_q)}
        out.update(refs.hands_at_side(pel, yaw0))
        return out

    def ankles():
        return [sim.d.xpos[sim.bid[f"{s}_ankle_roll_link"]][:2].copy() for s in ("left", "right")]

    trace, recent = [], deque(maxlen=HZ // 2)  # qpos per policy step; (k, fwd) over the last 0.5 s
    brake, end_k, still_k, reach, target = None, None, None, None, None
    frontmost = -np.inf
    k = 0
    while True:
        t = k * R.DT
        mujoco.mj_kinematics(sim.m, sim.d)
        trace.append(sim.d.qpos.copy())
        pxy = sim.d.qpos[:2].copy()
        fwd = float((pxy - xy0) @ h)
        recent.append((k, fwd))
        (k0, f0), (k1, f1) = recent[0], recent[-1]
        v = (f1 - f0) * HZ / max(k1 - k0, 1)
        if sim.fallen():
            return dict(fell="walk" if brake is None else "brake", fwd=fwd, sim_s=round(t, 2))
        if st8["prof"] is None and t >= R.SETTLE_S:
            # the walk starts: the standing toe reach (frontmost low point ahead of the pelvis) fixes plan's target
            reach = frontmost_low(sim.m, sim.d, robot, xy0, h) - fwd
            target = None if program == "marker" else marker - reach - margin
            st8["prof"] = Profile(speed, acc, None if target is None else target + ff)
        if st8["prof"] is not None:
            frontmost = max(frontmost, frontmost_low(sim.m, sim.d, robot, xy0, h) - marker)
        if brake is None and st8["prof"] is not None:
            if program == "marker":
                trigger = fwd >= marker
            else:
                trigger = t - R.SETTLE_S >= st8["prof"].t_dec
            if trigger:
                brake = dict(
                    k=k, fwd=fwd, lat=float((pxy - xy0) @ n), speed=v, stagger=abs(float(np.subtract(*ankles()) @ h))
                )
                if program == "marker":
                    st8["hold"] = pxy.copy()
                    end_k = k + int(settle * HZ)
                else:
                    end_k = k + int((st8["prof"].t_end - st8["prof"].t_dec + settle) * HZ)
            elif t > timeout_s:
                return dict(fell=None, timeout=True, fwd=fwd, sim_s=round(t, 2))
        elif brake is not None:
            if still_k is None and k - brake["k"] >= HZ // 2 and abs(v) < STILL_V:
                still_k = k
            if k >= end_k:
                break
        sim.step(pol(sim.state(), refs.futures(ref, t)))
        k += 1
    np.savez(trace_dir / f"{label}_state_trace.npz", seq=np.arange(len(trace)), qpos=np.array(trace))
    d = pxy - xy0
    yaw = float(np.arctan2(sim.d.xmat[sim.bid["pelvis"]][3], sim.d.xmat[sim.bid["pelvis"]][0]))
    return dict(
        fell=None,
        origin_xy=xy0.tolist(),
        heading=yaw0,
        marker=round(float(marker), 4),
        brake_fwd=brake["fwd"],
        brake_lat=brake["lat"],
        brake_speed=brake["speed"],
        brake_seq=brake["k"],
        stop_fwd=float(d @ h),
        stop_lat=float(d @ n),
        carry=float(d @ h) - brake["fwd"],
        stagger=abs(float(np.subtract(*ankles()) @ h)),
        stagger_settled=brake["stagger"],
        still_after_brake_s=None if still_k is None else (still_k - brake["k"]) / HZ,
        frontmost_max_past_marker=float(frontmost),
        frontmost_final_past_marker=float(frontmost_low(sim.m, sim.d, robot, xy0, h) - marker),
        toe_reach=float(reach),
        target_fwd=target,
        ff=ff,
        target_err=None if target is None else float(d @ h) - target,
        yaw_deg=float(np.degrees((yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi)),
        sim_s=round(k * R.DT, 2),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", choices=("marker", "plan"), required=True)
    ap.add_argument("--speeds", default="0.55,0.35,0.25", help="pelvis reference speeds, m/s")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--marker", type=float, default=1.5, help="metres along the heading to the line (before jitter)")
    ap.add_argument("--jitter", type=float, default=0.10, help="uniform +- jitter of the line per trial (m)")
    ap.add_argument("--acc", type=float, default=R.WALK_ACC, help="reference acceleration / deceleration (m/s^2)")
    ap.add_argument("--margin", type=float, default=0.03, help="plan: clearance kept before the line (m)")
    ap.add_argument(
        "--ff", default="0",
        help="plan: feedforward past the target, metres, or 'speed' = the measured shortfall at that speed (TODO 24.7.1)",
    )  # fmt: skip
    ap.add_argument("--settle", type=float, default=3.0, help="simulated seconds after the stop before the read")
    ap.add_argument("--timeout", type=float, default=40.0, help="simulated seconds to reach the line")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--perturb", default="", help="sim/perturb.py spec (JSON), TODO 25.5")
    ap.add_argument(
        "--odometry", action="store_true",
        help="ScaleBFM sees the leg-kinematics + IMU pelvis estimate (backends/scalebfm/odometry.py), not the truth (25.5)",
    )  # fmt: skip
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "trace").mkdir(exist_ok=True)
    cfg = ScaleBfmBackendConfig(scene_dir=str(WALK_SCENE))
    pol = R.Policy(device=a.device, backend=cfg)
    sim = R.Sim(pol.meta, backend=cfg, perturb=json.loads(a.perturb) if a.perturb else None)
    odo = None
    if a.odometry:
        from locomani.backends.scalebfm.odometry import attach

        odo = attach(sim, str(WALK_SCENE / "scene_locomani.xml"), pol.meta["joint_names"])
    robot = [g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] != 0]
    rng = np.random.default_rng(a.seed)
    rows, falls = [], 0
    print(
        f"{'speed':>6} {'i':>3} {'brake v':>8} {'carry cm':>9} {'stop cm':>8} {'toes cm':>8} {'lat cm':>7} {'yaw':>6}"
    )
    for speed in [float(x) for x in a.speeds.split(",")]:
        for i in range(a.repeats):
            if (a.out / "STOP").exists() or falls >= 3:
                break
            marker = a.marker + float(rng.uniform(-a.jitter, a.jitter))
            label = f"brake_{speed:.2f}_{i:02d}"
            r = dict(speed=speed, final_speed=None, final_distance=0.0, i=i, label=label)
            t0 = time.time()
            r.update(
                trial(
                    pol,
                    sim,
                    robot,
                    speed,
                    marker,
                    a.program,
                    a.acc,
                    a.margin,
                    a.settle,
                    a.timeout,
                    label,
                    a.out / "trace",
                    float(np.interp(speed, FF_SPEEDS, FF_SHORT)) if a.ff == "speed" else float(a.ff),
                    odo,
                )
            )
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            falls = falls + 1 if r.get("fell") else 0
            if r.get("fell") is None and "carry" in r:
                print(
                    f"{speed:>6.2f} {i:>3d} {r['brake_speed']:>8.2f} {100 * r['carry']:>+9.1f} "
                    f"{100 * (r['stop_fwd'] - r['marker']):>+8.1f} {100 * r['frontmost_final_past_marker']:>+8.1f} "
                    f"{100 * r['stop_lat']:>+7.1f} {r['yaw_deg']:>+6.1f}  still {r['still_after_brake_s']} s",
                    flush=True,
                )
            else:
                print(f"{speed:>6.2f} {i:>3d}  {r}", flush=True)
            (a.out / "results.json").write_text(
                json.dumps(
                    dict(
                        controller="scalebfm",
                        program=a.program,
                        checkpoint=cfg.checkpoint,
                        scene=str(WALK_SCENE.relative_to(REPO)),
                        marker=a.marker,
                        jitter=a.jitter,
                        acc=a.acc,
                        margin=a.margin,
                        ff=a.ff,
                        perturb=a.perturb or None,
                        odometry=bool(a.odometry),
                        settle_s=a.settle,
                        phase_labels=PHASE_LABELS[a.program],
                        rows=rows,
                    ),
                    indent=1,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
