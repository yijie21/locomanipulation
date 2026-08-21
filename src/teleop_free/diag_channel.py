"""M1+M2: does the PLANNER channel deliver the arm better than the streamed one?

The whole grasp currently runs on the streamed 29-DoF channel with a
sim-side actuator pin holding the arm on the IK solution. The 432-sample
delivery experiment measured what happens without that pin: 8.6 cm and
36 deg of orientation error while standing. That is the number the
dataset's latent would actually reproduce, so it has to come down before
the pin can go.

Two candidate explanations are tested here, as a channel factor:

    streamed     what the pipeline does today (StreamedArm)
    planner      PLANNER_FROZEN_UPPER_BODY: no mode switch, we write only
                 the 17 upper-body joints and the legs stay with the
                 deploy's k-planner animation
    planner_vel  the same, plus upper_body_velocity by finite difference
                 (every channel so far has sent zeros into an encoder that
                 observes motion_joint_velocities_*)

plus one control arm, `planner` with the sim pin ON, which should land near
the 2.2 cm the streamed channel reaches with its pin -- if it does not, the
rig is wrong, not the channel.

The robot stands still throughout. A cylinder is pinned gravity-free at a
stratified sample of the right hand's core; nothing is ever grasped, so
every sample is a clean pose measurement.

    python src/teleop_free/diag_channel.py [--dry] [--strata 27]
Output: outputs/diag/channel.npz  (one row per sample)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_grasp  # noqa: E402
import grasping  # noqa: E402
import reach_map as rmm  # noqa: E402
import stream_reach  # noqa: E402
from arm_ik import ArmIK  # noqa: E402
from g1_joints import ARM_MJ  # noqa: E402
from primitives import (  # noqa: E402
    GRASP_CENTRE_OFFSET,
    LocomotionMode,
    Primitives,
    StreamMode,
)
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"
OUT = "/workspace/code/locomanipulation/outputs/diag/channel.npz"
SIDE = "right"
SETTLE_OPEN = 1.5
STAND_DRIFT_MAX = 0.25
BLOCK = 9                        # strata per channel block, to spread drift

# (name, channel, pin) -- the four arms of the experiment.
CONFIGS = [
    ("streamed",    "streamed",    False),
    ("planner",     "planner",     False),
    ("planner_vel", "planner_vel", False),
    ("planner_pin", "planner",     True),
]


def rot_of(quat) -> np.ndarray:
    return Primitives._rot(quat)


def quat_of(rot: np.ndarray) -> np.ndarray:
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=float).flatten())
    return q


def can_quat(tilt: float, yaw: float) -> np.ndarray:
    ct, st = np.cos(tilt / 2), np.sin(tilt / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qt = np.array([ct, st, 0.0, 0.0])
    qy = np.array([cy, 0.0, 0.0, sy])
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qt
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def strata(rm, seed: int = 0) -> list:
    """One core cell per (radius, azimuth, height) stratum, 3x3x3.

    `seed` picks a different cell inside each stratum, so repeats sample the
    stratum rather than re-measuring one point of it."""
    grid = rm._grid(SIDE, core=True)
    pts = rm.origin + (np.argwhere(grid) + 0.5) * rm.vox
    r = np.hypot(pts[:, 0], pts[:, 1])
    az = np.arctan2(pts[:, 1], pts[:, 0])
    z = pts[:, 2]
    out, rng = [], np.random.default_rng(seed)

    def band(v, k):
        qs = np.quantile(v, [0.0, 1 / 3, 2 / 3, 1.0])
        return (v >= qs[k]) & (v <= qs[k + 1])

    for i in range(3):
        for j in range(3):
            for k in range(3):
                m = band(r, i) & band(az, j) & band(z, k)
                if not m.any():
                    continue
                cand = pts[m]
                out.append((i, j, k, cand[rng.integers(len(cand))]))
    return out


def make_arm(p, kind: str):
    if kind == "streamed":
        return stream_reach.StreamedArm(p, SIDE)
    return stream_reach.PlannerArm(p, SIDE, send_vel=(kind == "planner_vel"))


def restand(p, reset, label: str):
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(None)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    stream_reach.reanchor(p, side=SIDE)
    return p.obs()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--strata", type=int, default=27)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    obj_key = meta["target_object"]
    can = meta["objects"][obj_key]
    surf_top = meta["surfaces"]["A"]["top"]
    rm = rmm.ReachMap()

    blocks = []                                  # (config, [cells])
    for rep in range(args.repeats):
        cells = strata(rm, seed=rep)[: args.strata]
        for c0 in range(0, len(cells), BLOCK):
            chunk = cells[c0:c0 + BLOCK]
            for cfg in CONFIGS:
                blocks.append((cfg, chunk))
    n_total = sum(len(c) for _, c in blocks)
    print(f"[chan] {args.strata} strata x {len(CONFIGS)} configs x "
          f"{args.repeats} repeats = {n_total} samples in {len(blocks)} blocks")
    for name, kind, pin in CONFIGS:
        print(f"        {name:12} channel={kind:12} pin={'on' if pin else 'off'}")
    if args.dry:
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    p = Primitives(meta=meta, verbose=False)
    reset = EpisodeControlClient()
    ik = ArmIK()
    s0 = restand(p, reset, "channel")
    if s0["fallen"]:
        sys.exit("robot did not stand")
    home_xy = p.pelvis_xy(s0).copy()
    yaw0 = float(p.pelvis_yaw(s0))
    print(f"[chan] standing at {np.round(home_xy, 3)} yaw {np.degrees(yaw0):.0f}")

    rows, t0, n = [], time.time(), 0
    for (cname, kind, pin_on), chunk in blocks:
        s = p.obs()
        if s["fallen"] or float(np.linalg.norm(p.pelvis_xy(s) - home_xy)) > STAND_DRIFT_MAX:
            s = restand(p, reset, "restand")
            print(f"[chan] re-stood before block {cname}")
        reset.pin_arms()                     # release before any arm starts
        arm = make_arm(p, kind)
        arm.start()
        for (i, j, k, cell_pf) in chunk:
            n += 1
            s = p.obs()
            P = np.array(s["pelvis"]["pos"]); R = rot_of(s["pelvis"]["quat"])
            grip_w = P + R @ cell_pf
            q_can = can_quat(0.0, yaw0)
            axis_w = rot_of(q_can) @ np.array([0.0, 0.0, 1.0])
            centre = grip_w - axis_w * (can["height"] / 2.0 - grasping.GRIP_BELOW_RIM)
            reset.pin_object(obj_key, centre, q_can)
            time.sleep(0.25)

            s = p.obs()
            palm_R = rot_of(s["palms"][SIDE]["quat"])
            d_h = grip_w[:2] - p.pelvis_xy(s)
            d_h = d_h / max(float(np.linalg.norm(d_h)), 1e-6)
            v_start = grip_w - 0.25 * np.array([d_h[0], d_h[1], 0.0]) + [0, 0, 0.08]
            lat = -0.31 if SIDE == "right" else 0.31
            c2, s2 = np.cos(p.pelvis_yaw(s)), np.sin(p.pelvis_yaw(s))
            pz = s["pelvis"]["pos"]
            shoulder = np.array([pz[0] - s2 * lat, pz[1] + c2 * lat, pz[2] + 0.44])
            plan = adaptive_grasp.plan(
                grip_w, v_start, palm_R, surf_top, box=can["box"],
                obj_axis=axis_w, obj_rot=rot_of(s["objects"][obj_key]["quat"]),
                azimuth_from=shoulder,
                avoid_dir=np.array([d_h[0], d_h[1], 0.0]),
                avoid_sign=1.0 if SIDE == "right" else -1.0)
            if not plan.ok:
                rows.append(dict(cfg=cname, i=i, j=j, k=k, cell=cell_pf,
                                 ok=False, why=plan.why))
                reset.pin_object(obj_key)
                continue
            want_w = np.asarray(plan.wrist, float)
            want_R = np.asarray(plan.rot, float)

            # -- OPEN LOOP -------------------------------------------------
            reset.pin_arms(**({SIDE: [float(v) for v in arm.q[ARM_MJ[SIDE]]]}
                              if pin_on else {}))
            entry = want_R[:, 0]
            q_cmd, res = None, 0.0
            for tgt in (want_w - entry * stream_reach.SAFE_GAP + [0, 0, 0.10],
                        want_w - entry * stream_reach.SAFE_GAP, want_w):
                s = p.obs()
                aim_pf = p.world_to_root_full(tgt, s)
                rot_pf = rot_of(s["pelvis"]["quat"]).T @ want_R
                q_cmd, res, _ = ik.solve(SIDE, aim_pf, rot_pf, q_mj29=arm.q,
                                         use_waist=True)
                arm.move_to(q_cmd, speed=0.8)
                if pin_on:
                    reset.pin_arms(**{SIDE: [float(v) for v in q_cmd[ARM_MJ[SIDE]]]})
            arm.hold(SETTLE_OPEN)
            s = p.obs()
            open_pos = np.array(s["palms"][SIDE]["pos"])
            open_R = rot_of(s["palms"][SIDE]["quat"])
            open_q = np.array(s["joints"])

            # -- CLOSED LOOP -----------------------------------------------
            cl_err = arm.goto(want_w, want_R, refine=3, tol=0.015,
                              speed=stream_reach.FINE_SPEED)
            if pin_on:
                reset.pin_arms(**{SIDE: [float(v) for v in arm.q[ARM_MJ[SIDE]]]})
            arm.hold(0.6)
            s = p.obs()
            cl_pos = np.array(s["palms"][SIDE]["pos"])
            cl_R = rot_of(s["palms"][SIDE]["quat"])
            rows.append(dict(
                cfg=cname, i=i, j=j, k=k, cell=cell_pf, ok=True, why="",
                ik_res=float(res),
                cl_err=(-1.0 if cl_err is None else float(cl_err)),
                want_pos=want_w, want_quat=quat_of(want_R),
                open_pos=open_pos, open_quat=quat_of(open_R),
                closed_pos=cl_pos, closed_quat=quat_of(cl_R),
                cmd_joints=np.asarray(q_cmd, float),
                open_joints=open_q, closed_joints=np.array(s["joints"]),
                pelvis=np.concatenate([s["pelvis"]["pos"], s["pelvis"]["quat"]]),
                grip_w=grip_w))
            reset.pin_object(obj_key)
            e = float(np.linalg.norm(open_pos + open_R @ GRASP_CENTRE_OFFSET - grip_w))
            e2 = float(np.linalg.norm(cl_pos + cl_R @ GRASP_CENTRE_OFFSET - grip_w))
            wp = f"{q_cmd[14]:+.2f}->{open_q[14]:+.2f}"
            print(f"[chan] {n}/{n_total} {cname:11} r{i}a{j}z{k}  "
                  f"open {e * 100:5.1f}cm  closed {e2 * 100:5.1f}cm  "
                  f"waist {wp}  ({(time.time() - t0) / 60:.0f} min)")
            if n % 10 == 0:
                np.savez_compressed(OUT, rows=np.array(rows, dtype=object))
        arm.stop()
    np.savez_compressed(OUT, rows=np.array(rows, dtype=object))
    print(f"[chan] wrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
