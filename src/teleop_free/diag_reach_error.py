"""How far is SONIC's delivered gripper pose from the one IK asked for?

The robot stands still. A cylinder is pinned (sim-side, every step, so
gravity and contact cannot move it) at a stratified sample of the right
hand's sweet core, at four orientations. For each sample the arm is driven
to the planned grasp pose twice — open loop (stream the IK solution and
settle) and then closed loop (let the servo's trim/replan converge) — and
the DELIVERED palm pose is compared with the COMMANDED one.

Four factors, full factorial:
    position    3x3x3 strata over (radius, azimuth, height) of the core
    orientation upright yaw 0 / upright yaw 90 / tilt 45 / lying 90
    squat       standing / the planned crouch
    arm pin     on  (the live pipeline: arm joints forced onto the IK
                     solution in position mode, so the error is torso only)
                off (the policy tracks the 29-DoF pose stream itself)

Pin-on minus pin-off separates the delivery error into an arm-tracking
part and a body-pose part. Nothing is grasped: the gripper never closes,
the object is never released, so every sample is a clean pose measurement.

    python src/teleop_free/diag_reach_error.py --shard 0/4 [--dry]
Output: outputs/diag/reach_error_<shard>.npz  (one row per sample)
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
import paths
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

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"
OUT_DIR = f"{paths.REPO}/outputs/diag"
SIDE = "right"
ORIENTS = {  # (name, axis-angle applied to the upright can)
    "up_yaw0": (0.0, 0.0),
    "up_yaw90": (0.0, np.pi / 2),
    "tilt45": (np.pi / 4, 0.0),
    "lying90": (np.pi / 2, 0.0),
}
SETTLE_OPEN = 1.5  # s of holding the open-loop command
CLOSED_S = 8.0  # s cap on the closed-loop convergence
STAND_DRIFT_MAX = 0.25  # m of pelvis drift before we re-stand. The
# object is placed from the CURRENT pelvis
# pose every sample, so drift does not
# invalidate a sample — only a fall or a
# walk-away does.


def quat_of(rot: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=float).flatten())
    return q


def rot_of(quat) -> np.ndarray:
    return Primitives._rot(quat)


def can_quat(tilt: float, yaw: float) -> np.ndarray:
    """Upright can rotated by `tilt` about x, then `yaw` about world z."""
    ct, st = np.cos(tilt / 2), np.sin(tilt / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qt = np.array([ct, st, 0.0, 0.0])  # tilt about +x
    qy = np.array([cy, 0.0, 0.0, sy])  # yaw about +z
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qt
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def strata(rm) -> list:
    """One core cell per (radius, azimuth, height) stratum, 3x3x3."""
    grid = rm._grid(SIDE, core=True)
    pts = rm.origin + (np.argwhere(grid) + 0.5) * rm.vox
    r = np.hypot(pts[:, 0], pts[:, 1])
    az = np.arctan2(pts[:, 1], pts[:, 0])
    z = pts[:, 2]
    out, rng = [], np.random.default_rng(0)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--dry", action="store_true", help="list samples only")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))

    meta = json.load(open(SCENE_META))
    obj_key = meta["target_object"]
    can = meta["objects"][obj_key]
    surf_top = meta["surfaces"]["A"]["top"]
    rm = rmm.ReachMap()
    cells = strata(rm)

    plan_cs = rm.plan_core_stand(0.0)  # only for the crouch magnitude
    squat_cmd = (
        0.09 if plan_cs is None else float(np.interp(plan_cs[1], [0.0, 0.044, 0.12, 0.195], [0.0, 0.05, 0.09, 0.13]))
    )

    samples = [(c, o, sq, pin) for c in cells for o in ORIENTS for sq in (0.0, squat_cmd) for pin in (True, False)]
    mine = samples[si::sn]
    print(
        f"[err] {len(cells)} strata x {len(ORIENTS)} orientations x 2 squats"
        f" x 2 pins = {len(samples)} samples; shard {si}/{sn} -> {len(mine)}"
    )
    if args.dry:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    p = Primitives(meta=meta, verbose=False)
    reset = EpisodeControlClient()
    ik = ArmIK()

    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="reach_error")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    stream_reach.reanchor(p, side=SIDE)
    s0 = p.obs()
    if s0["fallen"]:
        sys.exit("robot did not stand")
    home_xy = p.pelvis_xy(s0).copy()
    yaw0 = float(p.pelvis_yaw(s0))
    fv = (float(np.cos(yaw0)), float(np.sin(yaw0)), 0.0)
    print(f"[err] standing at {np.round(home_xy, 3)} yaw {np.degrees(yaw0):.0f}")

    rows, t_start = [], time.time()
    arm = None
    for n, (cell, oname, sq, pin_on) in enumerate(mine):
        i, j, k, cell_pf = cell
        s = p.obs()
        # -- keep the robot on its stand ---------------------------------
        if s["fallen"] or float(np.linalg.norm(p.pelvis_xy(s) - home_xy)) > STAND_DRIFT_MAX:
            if arm is not None:
                arm.stop()
                arm = None
            p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
            p.cmd.upper_body = None
            p.cmd.reset_pose()
            reset.pin_arms()
            reset.reset(label="restand")
            time.sleep(0.3)
            p.cmd.start_control()
            p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
            stream_reach.reanchor(p, side=SIDE)
            s = p.obs()
            print(f"[err] re-stood at sample {n}")
        if arm is None:
            arm = stream_reach.StreamedArm(p, SIDE)
            arm.start()
            if sq > 0.03:
                arm.squat(sq, track=False)
        # -- place and PIN the object ------------------------------------
        P = np.array(s["pelvis"]["pos"])
        R = rot_of(s["pelvis"]["quat"])
        grip_w = P + R @ cell_pf
        tilt, yaw = ORIENTS[oname]
        q_can = can_quat(tilt, yaw + yaw0)
        axis_w = rot_of(q_can) @ np.array([0.0, 0.0, 1.0])
        centre = grip_w - axis_w * (can["height"] / 2.0 - grasping.GRIP_BELOW_RIM)
        reset.pin_object(obj_key, centre, q_can)
        time.sleep(0.25)
        # -- plan, exactly like the pipeline ------------------------------
        s = p.obs()
        palm_w = np.array(s["palms"][SIDE]["pos"])
        palm_R = rot_of(s["palms"][SIDE]["quat"])
        d_h = grip_w[:2] - p.pelvis_xy(s)
        d_h = d_h / max(float(np.linalg.norm(d_h)), 1e-6)
        v_start = grip_w - 0.25 * np.array([d_h[0], d_h[1], 0.0]) + [0, 0, 0.08]
        lat = -0.31 if SIDE == "right" else 0.31
        c2, s2 = np.cos(p.pelvis_yaw(s)), np.sin(p.pelvis_yaw(s))
        pz = s["pelvis"]["pos"]
        shoulder = np.array([pz[0] - s2 * lat, pz[1] + c2 * lat, pz[2] + 0.44])
        plan = adaptive_grasp.plan(
            grip_w,
            v_start,
            palm_R,
            surf_top,
            box=can["box"],
            obj_axis=axis_w,
            obj_rot=rot_of(s["objects"][obj_key]["quat"]),
            azimuth_from=shoulder,
            avoid_dir=np.array([d_h[0], d_h[1], 0.0]),
            avoid_sign=1.0 if SIDE == "right" else -1.0,
        )
        if not plan.ok:
            rows.append(dict(i=i, j=j, k=k, cell=cell_pf, orient=oname, squat=sq, pin=pin_on, ok=False, why=plan.why))
            reset.pin_object(obj_key)
            continue
        want_w = np.asarray(plan.wrist, float)
        want_R = np.asarray(plan.rot, float)
        # -- OPEN LOOP: stream the IK solution, settle, measure ------------
        reset.pin_arms(**({SIDE: [float(v) for v in arm.q[ARM_MJ[SIDE]]]} if pin_on else {}))
        entry = want_R[:, 0]
        for tgt in (
            want_w - entry * stream_reach.SAFE_GAP + [0, 0, 0.10],
            want_w - entry * stream_reach.SAFE_GAP,
            want_w,
        ):
            s = p.obs()
            aim_pf = p.world_to_root_full(tgt, s)
            rot_pf = rot_of(s["pelvis"]["quat"]).T @ want_R
            q_sol, res, _ = ik.solve(SIDE, aim_pf, rot_pf, q_mj29=arm.q, use_waist=True)
            arm.move_to(q_sol, speed=0.8)
            if pin_on:
                reset.pin_arms(**{SIDE: [float(v) for v in q_sol[ARM_MJ[SIDE]]]})
        arm.hold(SETTLE_OPEN)
        s = p.obs()
        open_pos = np.array(s["palms"][SIDE]["pos"])
        open_R = rot_of(s["palms"][SIDE]["quat"])
        open_pelvis = np.concatenate([s["pelvis"]["pos"], s["pelvis"]["quat"]])
        open_q = np.array(s["joints"])
        # -- CLOSED LOOP: the pipeline's own correction, without grasping.
        # StreamedArm.goto re-aims at (target + measured residual) and
        # re-solves, which is exactly the correction _servo_continuous
        # applies; calling the servo itself would close the gripper and
        # lift, and the measured palm would then be a carry pose, not a
        # converged reach (first smoke: 21-231 cm "residuals").
        cl_err = arm.goto(want_w, want_R, refine=3, tol=0.015, speed=stream_reach.FINE_SPEED)
        if pin_on:
            reset.pin_arms(**{SIDE: [float(v) for v in arm.q[ARM_MJ[SIDE]]]})
        arm.hold(0.6)
        cl = (cl_err, "")
        s = p.obs()
        cl_pos = np.array(s["palms"][SIDE]["pos"])
        cl_R = rot_of(s["palms"][SIDE]["quat"])
        rows.append(
            dict(
                i=i,
                j=j,
                k=k,
                cell=cell_pf,
                orient=oname,
                squat=sq,
                pin=pin_on,
                ok=True,
                why="",
                ik_res=float(res),
                want_pos=want_w,
                want_quat=quat_of(want_R),
                open_pos=open_pos,
                open_quat=quat_of(open_R),
                closed_pos=cl_pos,
                closed_quat=quat_of(cl_R),
                pelvis=open_pelvis,
                joints=open_q,
                grip_w=grip_w,
                obj_axis=axis_w,
            )
        )
        reset.pin_object(obj_key)
        e = float(np.linalg.norm(open_pos + open_R @ GRASP_CENTRE_OFFSET - grip_w))
        e2 = float(np.linalg.norm(cl_pos + cl_R @ GRASP_CENTRE_OFFSET - grip_w))
        print(
            f"[err] {n + 1}/{len(mine)} r{i}a{j}z{k} {oname:8} "
            f"sq{sq * 100:.0f} pin{'+' if pin_on else '-'}  "
            f"open {e * 100:5.1f}cm  closed {e2 * 100:5.1f}cm  "
            f"({(time.time() - t_start) / 60:.0f} min)"
        )
        if (n + 1) % 10 == 0:
            np.savez_compressed(f"{OUT_DIR}/reach_error_{si}.npz", rows=np.array(rows, dtype=object))
    if arm is not None:
        arm.stop()
    np.savez_compressed(f"{OUT_DIR}/reach_error_{si}.npz", rows=np.array(rows, dtype=object))
    print(f"[err] wrote {OUT_DIR}/reach_error_{si}.npz  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
