"""W0 admission (TODO 25.1, docs/handoff/controller-compare-20261013.md §3): can ScaleBFM stand, walk and bend
on the mars robot at all. Three tests in the mars warehouse scenes (waist 50 / ankle 35 N m come from the
scene's joint limits, ScaleBFM's own clip is 50), each repeated with jittered initial conditions because the
sim is deterministic:

* `stand` -- 30 s at the grasp stance, mode 2, hands hanging.
* `walk`  -- 1.5 m (+-10 cm) straight to the grasp stance on a trapezoid pelvis reference (acc 0.6 m/s^2) at
  each `--speeds`, then 3 s still. The room meshes do not collide; only the floor, platform and box do.
* `bend`  -- from standing at the stance, blend (smoothstep, 2.5 s) to SONIC's bimanual grasp posture read from
  its warehouse traces (sonic_grasp_posture.py: close-phase pelvis height + tilt, both wrists in the pelvis-yaw
  frame), hold 4 s, blend back, hold 2 s. `--mode 2` gives pelvis + wrists, `--mode 1` wrists only (the policy
  picks the pelvis). Grippers stay open; the box is in the scene.

Veto (plan §2 a / §3.1): any test with >= 3 falls in 10. results.json is rewritten after every trial.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/controller_compare_20261013/w0_admission.py --test bend --box 33 --mode 2 \\
        --out outputs/controller_compare_20261013/w0/bend_33_m2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "wbc_compare_20261012"))

from bfm_brake import Profile  # noqa: E402 (the brake line's trapezoid pelvis profile)

HZ = int(round(1 / R.DT))
SCENES = {33: "warehouse_box_45x32x33_mars", 18: "warehouse_box_45x32x18_mars"}
POSTURES = {33: "posture_33_mars.json", 18: "posture_18_stock.json"}  # 18: no mars run exists, stock pilot
STILL_V = 0.03  # m/s over 0.5 s (bfm_brake's rule)


def slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    c = float(q0 @ q1)
    if c < 0:
        q1, c = -q1, -c
    if c > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    a = np.arccos(c)
    return (np.sin((1 - u) * a) * q0 + np.sin(u * a) * q1) / np.sin(a)


def quat_angle_deg(q0, q1):
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.asarray(q0) @ np.asarray(q1)))))))


def seat(sim, rng=None, jitter_q=0.0):
    """Lower / raise the freshly reset robot so its lowest foot geom sits 5 mm above the floor: Sim.reset puts the
    pelvis at ScaleBFM's 0.78 m, which is not the mars legs' length. jitter_q (rad, uniform +-) perturbs the 29 body
    joints first: on the flat floor a shifted or turned start is the same trial (the policy sees the base frame), so
    the joint state is what makes repeats differ."""
    if rng is not None and jitter_q > 0:
        sim.d.qpos[sim.qadr] += rng.uniform(-jitter_q, jitter_q, sim.qadr.size)
    mujoco.mj_forward(sim.m, sim.d)
    low = min(
        sim.d.geom_xpos[g][2] - sim.m.geom_rbound[g]
        for g in range(sim.m.ngeom)
        if sim.m.geom_bodyid[g] in (sim.bid["left_ankle_roll_link"], sim.bid["right_ankle_roll_link"])
        and sim.m.geom_contype[g]
    )
    sim.d.qpos[2] += 0.005 - low
    mujoco.mj_forward(sim.m, sim.d)


class Stance:
    """The grasp stance of SONIC's close phase: pelvis at box - R(yaw) @ box_rel, facing the box (scene ym edge)."""

    def __init__(self, sim, posture, rng, jitter_xy, jitter_yaw_deg):
        box = np.asarray(sim.smeta["objects"]["nor_cal_box"]["init_pos"], float)
        self.yaw = np.radians(sim.smeta["robot_start"]["yaw_deg"]) + np.radians(rng.uniform(-1, 1) * jitter_yaw_deg)
        Rz = R.rot_z(self.yaw)
        rel = np.asarray(posture["box"], float)
        self.xy = box[:2] - (Rz @ rel)[:2] + rng.uniform(-1, 1, 2) * jitter_xy
        self.h = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        self.q = np.array([np.cos(self.yaw / 2), 0, 0, np.sin(self.yaw / 2)])


def run(pol, sim, refs, ref, t_end, trace, watch=None):
    """Step until t_end (simulated s) or a fall; watch(k, t) is called every tick before the step."""
    k = 0
    while k * R.DT < t_end:
        t = k * R.DT
        mujoco.mj_kinematics(sim.m, sim.d)
        trace.append(sim.d.qpos.copy())
        if sim.fallen():
            return t
        if watch:
            watch(k, t)
        sim.step(pol(sim.state(), refs.futures(ref, t)))
        k += 1
    return None


def feet(sim):
    return np.array([sim.d.xpos[sim.bid[f"{s}_ankle_roll_link"]][:2] for s in ("left", "right")])


class FootWatch:
    """Per foot: net displacement since start() (`foot_move`, includes deliberate steps) and slip in stance
    (`foot_slip`): over each stance span (floor contact every tick) the horizontal travel of the span's first contact
    centroid, a point fixed in the foot, from the span's first tick to its last, summed -- bottle_measure.foot_slip's
    rule, so contact jitter does not accumulate and a heel/toe roll is not slip."""

    def __init__(self, sim):
        self.sim = sim
        self.body = [sim.bid[f"{s}_ankle_roll_link"] for s in ("left", "right")]
        self.floor = {g for g in range(sim.m.ngeom) if sim.m.geom_bodyid[g] == 0}
        self.p0 = None
        self.span = [None, None]
        self.slip = np.zeros(2)

    def start(self):
        self.p0 = feet(self.sim)

    def tick(self):
        if self.p0 is None:
            return
        d, m = self.sim.d, self.sim.m
        pts = [[], []]
        for c in d.contact[: d.ncon]:
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if o in self.floor and m.geom_bodyid[g] in self.body:
                    pts[self.body.index(m.geom_bodyid[g])].append(c.pos.copy())
        for k, b in enumerate(self.body):
            R_, x = d.xmat[b].reshape(3, 3), d.xpos[b]
            if not pts[k]:
                self._close(k)
                continue
            if self.span[k] is None:
                local = R_.T @ (np.mean(pts[k], axis=0) - x)
                self.span[k] = [local, (R_ @ local + x)[:2].copy(), None]
            self.span[k][2] = (R_ @ self.span[k][0] + x)[:2].copy()

    def _close(self, k):
        if self.span[k] is not None and self.span[k][2] is not None:
            self.slip[k] += np.linalg.norm(self.span[k][2] - self.span[k][1])
        self.span[k] = None

    def result(self):
        for k in range(2):
            self._close(k)
        return dict(
            foot_move_cm=100 * float(np.max(np.linalg.norm(feet(self.sim) - self.p0, axis=1))),
            foot_slip_cm=100 * float(self.slip.max()),
        )


def pelvis_yaw(sim):
    return R.yaw_of(sim.d.xquat[sim.bid["pelvis"]])


def stand_trial(pol, sim, st, a, rng, trace):
    sim.reset(st.xy, st.yaw, 0.0)
    seat(sim, rng, a.jitter_q)
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    pel = np.array([st.xy[0], st.xy[1], R.PELVIS_Z])

    def ref(t):
        out = {"pelvis": (pel, st.q)}
        out.update(refs.hands_at_side(pel, st.yaw))
        return out

    fw = FootWatch(sim)

    def watch(k, t):
        if fw.p0 is None and t >= R.SETTLE_S:
            fw.start()
        fw.tick()

    fell = run(pol, sim, refs, ref, R.SETTLE_S + a.hold, trace, watch)
    if fell is not None:
        return dict(fell=round(fell, 2))
    d = sim.d.qpos[:2] - st.xy
    return dict(
        fell=None,
        pelvis_drift_cm=100 * float(np.linalg.norm(d)),
        yaw_drift_deg=float(np.degrees(R.wrap(pelvis_yaw(sim) - st.yaw))),
        pelvis_z=float(sim.d.qpos[2]),
        **fw.result(),
    )


def walk_trial(pol, sim, st, a, speed, rng, trace):
    dist = a.distance + float(rng.uniform(-a.jitter_dist, a.jitter_dist))
    xy0 = st.xy - st.h * dist
    sim.reset(xy0, st.yaw, 0.0)
    seat(sim, rng, a.jitter_q)
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    prof = Profile(speed, R.WALK_ACC, dist)

    def ref(t):
        p = xy0 + st.h * prof.s(t - R.SETTLE_S)
        pel = np.array([p[0], p[1], R.PELVIS_Z])
        out = {"pelvis": (pel, st.q)}
        out.update(refs.hands_at_side(pel, st.yaw))
        return out

    fell = run(pol, sim, refs, ref, R.SETTLE_S + prof.t_end + a.settle, trace)
    if fell is not None:
        return dict(fell=round(fell, 2), speed=speed, distance=dist, fwd=float((sim.d.qpos[:2] - xy0) @ st.h))
    d = sim.d.qpos[:2] - st.xy
    n = np.array([-st.h[1], st.h[0]])
    return dict(
        fell=None,
        speed=speed,
        distance=dist,
        stop_err_cm=100 * float(d @ st.h),
        lat_cm=100 * float(d @ n),
        yaw_deg=float(np.degrees(R.wrap(pelvis_yaw(sim) - st.yaw))),
        pelvis_z=float(sim.d.qpos[2]),
    )


def bend_trial(pol, sim, st, a, posture, rng, trace):
    sim.reset(st.xy, st.yaw, 0.0)
    seat(sim, rng, a.jitter_q)
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    Rz = R.rot_z(st.yaw)
    pel0 = np.array([st.xy[0], st.xy[1], R.PELVIS_Z])
    start = {"pelvis": (pel0, st.q)}
    start.update(refs.hands_at_side(pel0, st.yaw))
    pel1 = np.array([st.xy[0], st.xy[1], posture["pelvis_z"]])
    goal = {"pelvis": (pel1, R.quat_wxyz(Rz @ np.asarray(posture["pelvis_R"])))}
    for s in ("left", "right"):
        goal[f"{s}_wrist_yaw_link"] = (
            pel1 + Rz @ np.asarray(posture[f"{s}_wrist"]),
            R.quat_wxyz(Rz @ np.asarray(posture[f"{s}_wrist_R"])),
        )
    t1 = R.SETTLE_S
    t2 = t1 + a.blend
    t3 = t2 + a.hold
    t4 = t3 + a.blend
    t5 = t4 + a.settle

    def u_of(t):
        if t < t1:
            return 0.0
        if t < t2:
            return R.smoothstep((t - t1) / a.blend)
        if t < t3:
            return 1.0
        return 1.0 - R.smoothstep((t - t3) / a.blend)

    def ref(t):
        u = u_of(t)
        return {n: (start[n][0] + u * (goal[n][0] - start[n][0]), slerp(start[n][1], goal[n][1], u)) for n in goal}

    wpitch = sim.m.jnt_dofadr[mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")]
    box_b = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.smeta["objects"]["nor_cal_box"]["body"])
    box0 = sim.d.xpos[box_b].copy()
    held, tau, fw = [], [], FootWatch(sim)

    def watch(k, t):
        if fw.p0 is None and t >= R.SETTLE_S:
            fw.start()
        fw.tick()
        tau.append(abs(float(sim.d.qfrc_actuator[wpitch])))
        if t3 - 1.0 <= t < t3:  # the last second of the hold
            row = []
            for s in ("left", "right"):
                p, q = sim.body(f"{s}_wrist_yaw_link")
                gp, gq = goal[f"{s}_wrist_yaw_link"]
                row += [np.linalg.norm(p - gp), quat_angle_deg(q, gq)]
            tor = sim.d.xmat[sim.bid["torso_link"]].reshape(3, 3)
            row += [sim.d.qpos[2], np.degrees(np.arcsin(-tor[2, 0]))]
            held.append(row)

    fell = run(pol, sim, refs, ref, t5, trace, watch)
    out = dict(
        fell=None if fell is None else round(fell, 2),
        fell_phase=None if fell is None else ("down" if fell < t2 else "hold" if fell < t3 else "up"),
        waist_pitch_tau_p95=float(np.percentile(tau, 95)) if tau else None,
        box_moved_cm=100 * float(np.linalg.norm(sim.d.xpos[box_b] - box0)),
    )
    if held:
        h = np.array(held).mean(0)
        out.update(
            left_err_cm=100 * h[0],
            left_err_deg=h[1],
            right_err_cm=100 * h[2],
            right_err_deg=h[3],
            pelvis_z_held=h[4],
            torso_pitch_held=h[5],
            pelvis_z_goal=posture["pelvis_z"],
            torso_pitch_goal=posture["torso_pitch_deg"],
        )
    if fell is None:
        out.update(
            **fw.result(),
            pelvis_drift_cm=100 * float(np.linalg.norm(sim.d.qpos[:2] - st.xy)),
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=("stand", "walk", "bend"), required=True)
    ap.add_argument("--box", type=int, choices=sorted(SCENES), default=33, help="box height (cm): scene + posture")
    ap.add_argument("--mode", type=int, default=2, help="ScaleBFM control mode (assets/scalebfm/mode_table.pt)")
    ap.add_argument("--speeds", default="0.25,0.35,0.55", help="walk: pelvis reference speeds, m/s")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--distance", type=float, default=1.5, help="walk: metres to the stance (before jitter)")
    ap.add_argument("--jitter-dist", type=float, default=0.10)
    ap.add_argument("--jitter-xy", type=float, default=0.02, help="stance xy jitter (m, uniform +-)")
    ap.add_argument("--jitter-yaw", type=float, default=3.0, help="stance yaw jitter (deg, uniform +-)")
    ap.add_argument("--jitter-q", type=float, default=0.03, help="initial joint jitter (rad, uniform +-; see seat())")
    ap.add_argument("--hold", type=float, default=None, help="stand: 30 s; bend: 4 s")
    ap.add_argument("--blend", type=float, default=2.5, help="bend: seconds down / up")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.hold is None:
        a.hold = 30.0 if a.test == "stand" else 4.0
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "trace").mkdir(exist_ok=True)
    posture_path = REPO / "outputs" / "controller_compare_20261013" / "w0" / POSTURES[a.box]
    posture = json.loads(posture_path.read_text())["phases"]["close"]
    cfg = ScaleBfmBackendConfig(scene_dir=str(REPO / "outputs" / "scene" / SCENES[a.box]), object_name="nor_cal_box")
    pol = R.Policy(device=a.device, backend=cfg)
    pol.mode = torch.tensor([a.mode], dtype=torch.long, device=a.device)
    sim = R.Sim(pol.meta, backend=cfg)
    rng = np.random.default_rng(a.seed)
    speeds = [float(x) for x in a.speeds.split(",")] if a.test == "walk" else [None]
    rows = []
    for speed in speeds:
        for i in range(a.repeats):
            if (a.out / "STOP").exists():
                break
            st = Stance(sim, posture, rng, a.jitter_xy, a.jitter_yaw)
            label = f"{a.test}_{'' if speed is None else f'{speed:.2f}_'}{i:02d}"
            trace, t0 = [], time.time()
            if a.test == "stand":
                r = stand_trial(pol, sim, st, a, rng, trace)
            elif a.test == "walk":
                r = walk_trial(pol, sim, st, a, speed, rng, trace)
            else:
                r = bend_trial(pol, sim, st, a, posture, rng, trace)
            r.update(label=label, i=i, wall_s=round(time.time() - t0, 1), stance_xy=st.xy.tolist(), yaw=st.yaw)
            np.savez(a.out / "trace" / f"{label}_state_trace.npz", seq=np.arange(len(trace)), qpos=np.array(trace))
            rows.append(r)
            print(
                json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "stance_xy"}),
                flush=True,
            )
            (a.out / "results.json").write_text(
                json.dumps(
                    dict(
                        controller="scalebfm",
                        test=a.test,
                        mode=a.mode,
                        box_cm=a.box,
                        scene=SCENES[a.box],
                        posture=str(posture_path.relative_to(REPO)),
                        checkpoint=cfg.checkpoint,
                        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                        rows=rows,
                    ),
                    indent=1,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
