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

SCENE_DIR = "/workspace/code/locomanipulation/outputs/scene/sim"
SCENE = f"{SCENE_DIR}/scene_locomani_zone.xml"
META = f"{SCENE_DIR}/scene_meta.json"
OUT = "/workspace/code/locomanipulation/outputs/videos/ik_grasp.mp4"
TMP = "/tmp/claude-1002/ik_grasp_frames"
W, H, FPS = 1280, 720, 30
SIDE = "right"
PELVIS_Z = rmm.PELVIS_MAX          # 0.793, the standing pelvis height


def stance_qpos(model, data, arm_q=None):
    """Nominal standing pose: the controller's stance legs + READY arms."""
    mujoco.mj_resetData(model, data)
    q = np.zeros(29)
    q[[0, 6]] = -0.10             # hip pitch
    q[[3, 9]] = 0.30              # knee
    q[[4, 10]] = -0.20            # ankle pitch
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
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{sd}_hand_gripper_{i}_joint")
            if j >= 0:
                data.qpos[model.jnt_qposadr[j]] = float(
                    grip[i - 1] if sd == SIDE else HAND_OPEN[i - 1])
    mujoco.mj_forward(model, data)


def seat_on_floor(model, data, root_id):
    mujoco.mj_forward(model, data)
    low = min((float(data.geom_xpos[g][2] - model.geom_rbound[g])
               for g in range(model.ngeom)
               if model.body_rootid[model.geom_bodyid[g]] == root_id),
              default=0.0)
    data.qpos[2] -= low
    mujoco.mj_forward(model, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(TMP, exist_ok=True)
    for f in os.listdir(TMP):
        os.remove(f"{TMP}/{f}")

    meta = json.load(open(META))
    can = meta["objects"]["can"]
    surf_top = meta["surfaces"]["A"]["top"]
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    ik = ArmIK()
    rm = rmm.ReachMap()

    pelvis_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    palm_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                               f"{SIDE}_wrist_yaw_link")
    can_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, can["body"])
    can_adr = int(can["qpos_adr"])
    ghost_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost_hand")
    ghost_m = int(model.body_mocapid[ghost_b]) if ghost_b >= 0 else -1
    root_id = pelvis_b

    # Core cells of the RIGHT hand, pelvis frame, as (x, y, z) centres.
    core = rm._grid(SIDE, core=True)
    cells = np.argwhere(core)
    pts = rm.origin + (cells + 0.5) * rm.vox
    print(f"[ik] {len(pts)} core cells to sample from")

    lo, hi = ik.lo[ARM_MJ[SIDE]], ik.hi[ARM_MJ[SIDE]]
    rows, frames = [], 0
    cam = mujoco.MjvCamera()
    with mujoco.Renderer(model, height=H, width=W) as r:
        for t in range(args.trials):
            # -- pose the robot, pin the can at a random core cell ----------
            q = stance_qpos(model, data)
            apply_q(model, data, q)
            seat_on_floor(model, data, root_id)
            P = data.xpos[pelvis_b].copy()
            R = data.xmat[pelvis_b].reshape(3, 3).copy()
            grip_pf = pts[rng.integers(len(pts))]
            grip_w = P + R @ grip_pf
            yaw = float(rng.uniform(-np.pi, np.pi))
            # the can's grip point sits GRIP_BELOW_RIM under its rim
            c_z = grip_w[2] - (can["height"] / 2.0 - grasping.GRIP_BELOW_RIM)
            data.qpos[can_adr:can_adr + 3] = [grip_w[0], grip_w[1], c_z]
            data.qpos[can_adr + 3:can_adr + 7] = [np.cos(yaw / 2), 0, 0,
                                                  np.sin(yaw / 2)]
            mujoco.mj_forward(model, data)

            # -- plan, exactly like the live pipeline -----------------------
            palm_w = data.xpos[palm_b].copy()
            palm_R = data.xmat[palm_b].reshape(3, 3).copy()
            d_h = grip_w[:2] - P[:2]
            d_h = d_h / max(float(np.linalg.norm(d_h)), 1e-6)
            v_start = grip_w - 0.25 * np.array([d_h[0], d_h[1], 0.0]) + [0, 0, 0.08]
            shoulder = P + R @ np.array([0.0, -0.31, 0.44])
            plan = adaptive_grasp.plan(
                grip_w, v_start, palm_R, surf_top, box=can["box"],
                obj_rot=data.xmat[can_b].reshape(3, 3).copy(),
                azimuth_from=shoulder,
                avoid_dir=np.array([d_h[0], d_h[1], 0.0]), avoid_sign=1.0)
            if not plan.ok:
                rows.append((t, grip_pf, None, None, None, None, f"PLAN {plan.why}"))
                continue
            if ghost_m >= 0:
                gq = np.empty(4)
                mujoco.mju_mat2Quat(gq, np.asarray(plan.rot, float).flatten())
                data.mocap_pos[ghost_m] = plan.wrist
                data.mocap_quat[ghost_m] = gq
            entry = np.asarray(plan.rot)[:, 0]        # dex1 enters along palm +x
            pre_w = np.asarray(plan.wrist) - entry * SAFE_GAP
            above_w = pre_w + np.array([0.0, 0.0, 0.10])

            # -- drive the arm with the SAME IK the servo uses --------------
            way, q_way, res_last = [above_w, pre_w, np.asarray(plan.wrist)], [], 0.0
            q_cur = q.copy()
            for w in way:
                aim_pf = R.T @ (np.asarray(w) - P)
                rot_pf = R.T @ np.asarray(plan.rot)
                q_sol, res, rerr = ik.solve(SIDE, aim_pf, rot_pf,
                                            q_mj29=q_cur, use_waist=True)
                q_way.append(q_sol.copy())
                q_cur, res_last = q_sol, res

            # -- measure the DELIVERED pose --------------------------------
            apply_q(model, data, q_cur)
            palm_w2 = data.xpos[palm_b].copy()
            palm_R2 = data.xmat[palm_b].reshape(3, 3).copy()
            jaw = palm_w2 + palm_R2 @ GRASP_CENTRE_OFFSET
            e_jaw = float(np.linalg.norm(jaw - grip_w))
            e_rot = float(np.degrees(np.arccos(np.clip(
                (np.trace(palm_R2.T @ np.asarray(plan.rot)) - 1) / 2, -1, 1))))
            arm_now = q_cur[ARM_MJ[SIDE]]
            marg = float(np.degrees(np.minimum(arm_now - lo, hi - arm_now).min()))
            # can inside the jaw? (palm-frame box test against the pads)
            q_obj = palm_R2.T @ (data.xpos[can_b] - palm_w2) - GRASP_CENTRE_OFFSET
            inside = bool(abs(q_obj[0]) < 0.032 and abs(q_obj[1]) < 0.030
                          and abs(q_obj[2]) < 0.030)

            # -- render: READY -> above -> pre -> grip -> close ------------
            segs = [("flight", q, q_way[0], 12, 0.0),
                    ("flight", q_way[0], q_way[1], 14, 0.0),
                    ("entry", q_way[1], q_way[2], 16, 0.0),
                    ("close", q_way[2], q_way[2], 18, 1.0)]
            # Contacts BEFORE the fingers close are the interesting ones: a
            # can touched during the flight or the slide-in is a can the
            # live robot would knock over. Contacts during the close are
            # the grasp itself.
            touch = {"flight": set(), "entry": set(), "close": set()}
            for ph, a, b, n, close_to in segs:
                for k in range(n):
                    u = (k + 1) / n
                    sm = 3 * u * u - 2 * u ** 3
                    apply_q(model, data, a + sm * (b - a), hand_alpha=close_to * sm)
                    mujoco.mj_collision(model, data)
                    for ci in range(data.ncon):
                        g1, g2 = data.contact.geom1[ci], data.contact.geom2[ci]
                        for ga, gb in ((g1, g2), (g2, g1)):
                            if model.geom_bodyid[ga] == can_b:
                                nm = model.body(model.geom_bodyid[gb]).name
                                if "dex1" in nm or "wrist" in nm:
                                    touch[ph].add(nm.replace("right_", "R.")
                                                  .replace("_link", ""))
                    cam.lookat[:] = 0.5 * (P + grip_w) + [0, 0, 0.15]
                    cam.azimuth = float(np.degrees(np.arctan2(
                        *(grip_w[:2] - P[:2])[::-1])) - 55.0)
                    cam.elevation, cam.distance = -14.0, 1.55
                    r.update_scene(data, camera=cam)
                    img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
                    lines = [f"trial {t + 1}/{args.trials}   core cell (pelvis) "
                             f"[{grip_pf[0]:+.2f} {grip_pf[1]:+.2f} {grip_pf[2]:+.2f}]"
                             f"   can yaw {np.degrees(yaw):+.0f}deg",
                             f"IK res {res_last * 100:.1f}cm   jaw-centre err "
                             f"{e_jaw * 100:.1f}cm   palm rot err {e_rot:.0f}deg"
                             f"   worst joint margin {marg:.0f}deg"
                             f"   {'IN JAW' if inside else 'NOT in jaw'}"
                             + (f"   PRE-CONTACT {sorted(touch['flight'] | touch['entry'])}"
                                if (touch["flight"] | touch["entry"]) else "")]
                    for i, ln in enumerate(lines):
                        cv2.putText(img, ln, (16, 34 + 30 * i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imwrite(f"{TMP}/f_{frames:05d}.png", img)
                    frames += 1
            pre_c = sorted(touch["flight"] | touch["entry"])
            rows.append((t, grip_pf, res_last, e_jaw, e_rot, marg,
                         ("in jaw" if inside else "NOT IN JAW")
                         + (f"; pre-contact {pre_c}" if pre_c else "")
                         + ("" if touch["close"] else "; CLOSED ON NOTHING")))

    print(f"\n{'#':>3} {'core cell (pelvis)':>26} {'IKres':>7} {'jawErr':>7} "
          f"{'rotErr':>7} {'margin':>7}  verdict")
    for t, c, res, ej, er, mg, note in rows:
        cell = f"[{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}]"
        if res is None:
            print(f"{t + 1:>3} {cell:>26} {'-':>7} {'-':>7} {'-':>7} {'-':>7}  {note}")
        else:
            print(f"{t + 1:>3} {cell:>26} {res * 100:>6.1f}c {ej * 100:>6.1f}c "
                  f"{er:>6.0f}d {mg:>6.0f}d  {note}")
    ok = [r for r in rows if r[2] is not None]
    if ok:
        print(f"\n在爪内 {sum('NOT IN JAW' not in r[6] for r in ok)}/{len(ok)}   "
              f"进入途中撞到罐子 {sum('pre-contact' in r[6] for r in ok)}/{len(ok)}   "
              f"闭合无接触 {sum('NOTHING' in r[6] for r in ok)}/{len(ok)}\n"
              f"IK 残差中位 {np.median([r[2] for r in ok]) * 100:.2f} cm   "
              f"jaw 误差中位 {np.median([r[3] for r in ok]) * 100:.2f} cm   "
              f"姿态误差中位 {np.median([r[4] for r in ok]):.1f} deg   "
              f"最差关节裕量中位 {np.median([r[5] for r in ok]):.0f} deg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{TMP}/f_%05d.png", "-vf", "format=yuv420p",
                    "-c:v", "libx264", "-crf", "23", OUT], check=True)
    print(f"\nwrote {OUT}  ({frames} frames)")


if __name__ == "__main__":
    main()
