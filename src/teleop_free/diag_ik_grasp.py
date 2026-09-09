"""Pure-kinematics IK grasp bench: is the PLANNED hand pose any good?

Every failure in the live pipeline is ambiguous between "the planned pose is
wrong" and "SONIC could not deliver it". This bench removes the second half:
no policy, no deploy, no physics step. The robot stands in the nominal stance
with the READY arms, a can is pinned (no gravity) at a random cell of the
RIGHT hand's sweet core with a random yaw, adaptive_grasp plans the grasp, the
magenta ghost shows the planned wrist pose, and the arm is driven onto it by
the SAME ArmIK the servo uses. What the video shows is therefore the raw IK
answer — flight, entry and closure — with nothing downstream to blame.

Per trial it reports: IK residual, delivered jaw-centre error, palm rotation
error, the worst joint-limit margin (how "twisted" the posture is), and the
finger/can contacts at the closed pose.

    python src/teleop_free/diag_ik_grasp.py [--trials 12] [--seed 0]
Output: outputs/videos/ik_grasp.mp4  (+ a per-trial table on stdout)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_grasp  # noqa: E402
import grasping  # noqa: E402
import paths
import reach_map as rmm  # noqa: E402
from arm_ik import ArmIK  # noqa: E402
from g1_joints import ARM_MJ, MUJOCO_JOINT_NAMES  # noqa: E402
from primitives import (  # noqa: E402
    GRASP_CENTRE_OFFSET,
    HAND_CLOSED_RIGHT,
    HAND_OPEN,
    READY_ARM,
)
from stream_reach import SAFE_GAP  # noqa: E402

SCENE_DIR = f"{paths.REPO}/outputs/scene/sim"
SCENE = f"{SCENE_DIR}/scene_locomani_zone.xml"
META = f"{SCENE_DIR}/scene_meta.json"
OUT = f"{paths.REPO}/outputs/videos/ik_grasp.mp4"
TMP = "/tmp/claude-1002/ik_grasp_frames"
W, H, FPS = 1280, 720, 30
SIDE = "right"
PELVIS_Z = rmm.PELVIS_MAX  # 0.793, the standing pelvis height


def stance_qpos(model, data, arm_q=None, squat=0.0):
    """Nominal standing pose: the controller's stance legs + READY arms.

    `squat` is a metres-of-drop request mapped through the same gain the
    streamed squat uses (SQUAT_GAIN 6.8 rad/m), so --live reproduces the
    body the servo actually grasps from."""
    mujoco.mj_resetData(model, data)
    q = np.zeros(29)
    a = float(np.clip(squat * 6.8, 0.0, 0.65))
    q[[0, 6]] = -0.10 - a  # hip pitch
    q[[3, 9]] = 0.30 + 2 * a  # knee
    q[[4, 10]] = -0.20 - a  # ankle pitch
    q[ARM_MJ["left"]] = READY_ARM["left"]
    q[ARM_MJ["right"]] = arm_q if arm_q is not None else READY_ARM["right"]
    return q


def apply_q(model, data, q29, hand_alpha=0.0):
    """Write a 29-vector plus the gripper opening into the scene qpos."""
    for name, v in zip(MUJOCO_JOINT_NAMES, q29):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = float(v)
    grip = HAND_OPEN + hand_alpha * (HAND_CLOSED_RIGHT - HAND_OPEN)
    for sd in ("left", "right"):
        for i in (1, 2):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{sd}_hand_gripper_{i}_joint")
            if j >= 0:
                data.qpos[model.jnt_qposadr[j]] = float(grip[i - 1] if sd == SIDE else HAND_OPEN[i - 1])
    mujoco.mj_forward(model, data)


def seat_on_floor(model, data, root_id):
    mujoco.mj_forward(model, data)
    low = min(
        (
            float(data.geom_xpos[g][2] - model.geom_rbound[g])
            for g in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[g]] == root_id
        ),
        default=0.0,
    )
    data.qpos[2] -= low
    mujoco.mj_forward(model, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--live",
        action="store_true",
        help="reproduce the EXECUTED pose instead of sampling the "
        "whole core: squat to the planned depth, stand the "
        "can upright at the planned rel, randomize yaw only",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(TMP, exist_ok=True)
    for f in os.listdir(TMP):
        os.remove(f"{TMP}/{f}")

    meta = json.load(open(META))
    can = meta["objects"]["can"]
    surf_top = meta["surfaces"]["A"]["top"]
    grip_z_w = float(can["init_pos"][2] + can["height"] / 2.0 - grasping.GRIP_BELOW_RIM)
    grip_z_pf = grip_z_w - PELVIS_Z
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    ik = ArmIK()
    rm = rmm.ReachMap()

    pelvis_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    palm_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{SIDE}_wrist_yaw_link")
    can_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, can["body"])
    can_adr = int(can["qpos_adr"])
    ghost_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost_hand")
    ghost_m = int(model.body_mocapid[ghost_b]) if ghost_b >= 0 else -1
    root_id = pelvis_b

    # Core cells of the RIGHT hand, pelvis frame, as (x, y, z) centres.
    core = rm._grid(SIDE, core=True)
    cells = np.argwhere(core)
    pts = rm.origin + (cells + 0.5) * rm.vox
    live_plan = None
    if args.live:
        # The executed geometry: plan_core_stand's aim cell, the crouch the
        # gym commands for it, and an upright can at grip height.
        res = rm.plan_core_stand(float(grip_z_pf))
        if res is None:
            sys.exit("no core slice at the grip height")
        aim, want_drop, marg = res
        squat_cmd = float(np.interp(want_drop, [0.0, 0.044, 0.12, 0.195], [0.0, 0.05, 0.09, 0.13]))
        live_plan = (aim, want_drop, squat_cmd)
        print(
            f"[ik] LIVE mode: {len(aim)} aim cells, drop {want_drop:.3f} m "
            f"(cmd {squat_cmd:.3f}), scatter margin {marg:.2f} m"
        )
    else:
        print(f"[ik] {len(pts)} core cells to sample from")

    lo, hi = ik.lo[ARM_MJ[SIDE]], ik.hi[ARM_MJ[SIDE]]
    rows, frames = [], 0
    cam = mujoco.MjvCamera()
    with mujoco.Renderer(model, height=H, width=W) as r:
        for t in range(args.trials):
            # -- pose the robot, pin the can at a random core cell ----------
            sq = live_plan[2] if live_plan else 0.0
            q = stance_qpos(model, data, squat=sq)
            apply_q(model, data, q)
            seat_on_floor(model, data, root_id)
            P = data.xpos[pelvis_b].copy()
            R = data.xmat[pelvis_b].reshape(3, 3).copy()
            if live_plan:
                aim = live_plan[0]
                xy = aim[rng.integers(len(aim))]
                # grip height in the SQUATTED pelvis frame, plus the +-4 cm
                # of arrival scatter the stand loop accepts
                grip_pf = np.array(
                    [xy[0] + rng.uniform(-0.04, 0.04), xy[1] + rng.uniform(-0.04, 0.04), grip_z_w - float(P[2])]
                )
            else:
                grip_pf = pts[rng.integers(len(pts))]
            grip_w = P + R @ grip_pf
            # FULL random orientation (uniform on SO(3), Shoemake), not just
            # yaw: a tilted can is what tests the planner's jaw-tilt gate and
            # the entry geometry. The live scene only ever spawns cans
            # upright, so this is strictly harder than the real task.
            if live_plan:  # upright, yaw only: the scene
                yaw = float(rng.uniform(-np.pi, np.pi))
                quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            else:
                u1, u2, u3 = rng.random(3)
                quat = np.array(
                    [
                        np.sqrt(u1) * np.cos(2 * np.pi * u3),
                        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                        np.sqrt(u1) * np.sin(2 * np.pi * u3),
                    ]
                )
            cR = np.empty(9)
            mujoco.mju_quat2Mat(cR, quat)
            axis_w = cR.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            tilt = float(np.degrees(np.arccos(np.clip(abs(axis_w[2]), -1, 1))))
            # The grip point sits GRIP_BELOW_RIM under the rim ALONG THE CAN'S
            # OWN AXIS — with a tilted can that is no longer a z offset.
            centre = grip_w - axis_w * (can["height"] / 2.0 - grasping.GRIP_BELOW_RIM)
            data.qpos[can_adr : can_adr + 3] = centre
            data.qpos[can_adr + 3 : can_adr + 7] = quat
            mujoco.mj_forward(model, data)

            # -- plan, exactly like the live pipeline -----------------------
            palm_w = data.xpos[palm_b].copy()
            palm_R = data.xmat[palm_b].reshape(3, 3).copy()
            d_h = grip_w[:2] - P[:2]
            d_h = d_h / max(float(np.linalg.norm(d_h)), 1e-6)
            v_start = grip_w - 0.25 * np.array([d_h[0], d_h[1], 0.0]) + [0, 0, 0.08]
            shoulder = P + R @ np.array([0.0, -0.31, 0.44])
            plan = adaptive_grasp.plan(
                grip_w,
                v_start,
                palm_R,
                surf_top,
                box=can["box"],
                obj_axis=axis_w,
                obj_rot=data.xmat[can_b].reshape(3, 3).copy(),
                azimuth_from=shoulder,
                avoid_dir=np.array([d_h[0], d_h[1], 0.0]),
                avoid_sign=1.0,
            )
            if not plan.ok:
                rows.append((t, grip_pf, tilt, None, None, None, None, f"PLAN REJECTED: {plan.why}"))
                continue
            if ghost_m >= 0:
                gq = np.empty(4)
                mujoco.mju_mat2Quat(gq, np.asarray(plan.rot, float).flatten())
                data.mocap_pos[ghost_m] = plan.wrist
                data.mocap_quat[ghost_m] = gq
            entry = np.asarray(plan.rot)[:, 0]  # dex1 enters along palm +x
            pre_w = np.asarray(plan.wrist) - entry * SAFE_GAP
            above_w = pre_w + np.array([0.0, 0.0, 0.10])

            # -- drive the arm with the SAME IK the servo uses --------------
            way, q_way, res_last = [above_w, pre_w, np.asarray(plan.wrist)], [], 0.0
            q_cur = q.copy()
            for w in way:
                aim_pf = R.T @ (np.asarray(w) - P)
                rot_pf = R.T @ np.asarray(plan.rot)
                q_sol, res, rerr = ik.solve(SIDE, aim_pf, rot_pf, q_mj29=q_cur, use_waist=True)
                q_way.append(q_sol.copy())
                q_cur, res_last = q_sol, res

            # -- measure the DELIVERED pose --------------------------------
            apply_q(model, data, q_cur)
            palm_w2 = data.xpos[palm_b].copy()
            palm_R2 = data.xmat[palm_b].reshape(3, 3).copy()
            jaw = palm_w2 + palm_R2 @ GRASP_CENTRE_OFFSET
            e_jaw = float(np.linalg.norm(jaw - grip_w))
            e_rot = float(np.degrees(np.arccos(np.clip((np.trace(palm_R2.T @ np.asarray(plan.rot)) - 1) / 2, -1, 1))))
            arm_now = q_cur[ARM_MJ[SIDE]]
            marg = float(np.degrees(np.minimum(arm_now - lo, hi - arm_now).min()))
            # can inside the jaw? (palm-frame box test against the pads)
            q_obj = palm_R2.T @ (data.xpos[can_b] - palm_w2) - GRASP_CENTRE_OFFSET
            inside = bool(abs(q_obj[0]) < 0.032 and abs(q_obj[1]) < 0.030 and abs(q_obj[2]) < 0.030)

            # -- render: READY -> above -> pre -> grip -> close ------------
            segs = [
                ("flight", q, q_way[0], 12, 0.0),
                ("flight", q_way[0], q_way[1], 14, 0.0),
                ("entry", q_way[1], q_way[2], 16, 0.0),
                ("close", q_way[2], q_way[2], 18, 1.0),
            ]
            # Contacts BEFORE the fingers close are the interesting ones: a
            # can touched during the flight or the slide-in is a can the
            # live robot would knock over. Contacts during the close are
            # the grasp itself.
            touch = {"flight": set(), "entry": set(), "close": set()}
            for ph, a, b, n, close_to in segs:
                for k in range(n):
                    u = (k + 1) / n
                    sm = 3 * u * u - 2 * u**3
                    apply_q(model, data, a + sm * (b - a), hand_alpha=close_to * sm)
                    mujoco.mj_collision(model, data)
                    for ci in range(data.ncon):
                        g1, g2 = data.contact.geom1[ci], data.contact.geom2[ci]
                        for ga, gb in ((g1, g2), (g2, g1)):
                            if model.geom_bodyid[ga] == can_b:
                                nm = model.body(model.geom_bodyid[gb]).name
                                if "dex1" in nm or "wrist" in nm:
                                    touch[ph].add(nm.replace("right_", "R.").replace("_link", ""))
                    cam.lookat[:] = 0.5 * (P + grip_w) + [0, 0, 0.15]
                    cam.azimuth = float(np.degrees(np.arctan2(*(grip_w[:2] - P[:2])[::-1])) - 55.0)
                    cam.elevation, cam.distance = -14.0, 1.55
                    r.update_scene(data, camera=cam)
                    img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
                    lines = [
                        f"trial {t + 1}/{args.trials}   core cell (pelvis) "
                        f"[{grip_pf[0]:+.2f} {grip_pf[1]:+.2f} {grip_pf[2]:+.2f}]"
                        f"   can tilt {tilt:.0f}deg",
                        f"IK res {res_last * 100:.1f}cm   jaw-centre err "
                        f"{e_jaw * 100:.1f}cm   palm rot err {e_rot:.0f}deg"
                        f"   worst joint margin {marg:.0f}deg"
                        f"   {'IN JAW' if inside else 'NOT in jaw'}"
                        + (
                            f"   PRE-CONTACT {sorted(touch['flight'] | touch['entry'])}"
                            if (touch["flight"] | touch["entry"])
                            else ""
                        ),
                    ]
                    for i, ln in enumerate(lines):
                        cv2.putText(
                            img, ln, (16, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
                        )
                    cv2.imwrite(f"{TMP}/f_{frames:05d}.png", img)
                    frames += 1
            pre_c = sorted(touch["flight"] | touch["entry"])
            rows.append(
                (
                    t,
                    grip_pf,
                    tilt,
                    res_last,
                    e_jaw,
                    e_rot,
                    marg,
                    ("in jaw" if inside else "NOT IN JAW")
                    + (f"; pre-contact {pre_c}" if pre_c else "")
                    + ("" if touch["close"] else "; CLOSED ON NOTHING"),
                )
            )

    print(
        f"\n{'#':>3} {'core cell (pelvis)':>26} {'tilt':>5} {'IKres':>7} "
        f"{'jawErr':>7} {'rotErr':>7} {'margin':>7}  verdict"
    )
    for t, c, ti, res, ej, er, mg, note in rows:
        cell = f"[{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}]"
        if res is None:
            print(f"{t + 1:>3} {cell:>26} {ti:>4.0f}d {'-':>7} {'-':>7} {'-':>7} {'-':>7}  {note}")
        else:
            print(
                f"{t + 1:>3} {cell:>26} {ti:>4.0f}d {res * 100:>6.1f}c "
                f"{ej * 100:>6.1f}c {er:>6.0f}d {mg:>6.0f}d  {note}"
            )
    rej = [r for r in rows if r[3] is None]
    if rej:
        print(f"\n规划直接拒绝 {len(rej)}/{len(rows)}  倾角 {sorted(round(r[2]) for r in rej)} deg")
    ok = [r for r in rows if r[3] is not None]
    if ok:
        print(
            f"\n在爪内 {sum('NOT IN JAW' not in r[7] for r in ok)}/{len(ok)}   "
            f"进入途中撞到罐子 {sum('pre-contact' in r[7] for r in ok)}/{len(ok)}   "
            f"闭合无接触 {sum('NOTHING' in r[7] for r in ok)}/{len(ok)}\n"
            f"IK 残差中位 {np.median([r[3] for r in ok]) * 100:.2f} cm   "
            f"jaw 误差中位 {np.median([r[4] for r in ok]) * 100:.2f} cm   "
            f"姿态误差中位 {np.median([r[5] for r in ok]):.1f} deg   "
            f"最差关节裕量中位 {np.median([r[6] for r in ok]):.0f} deg"
        )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            f"{TMP}/f_%05d.png",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            OUT,
        ],
        check=True,
    )
    print(f"\nwrote {OUT}  ({frames} frames)")


if __name__ == "__main__":
    main()
