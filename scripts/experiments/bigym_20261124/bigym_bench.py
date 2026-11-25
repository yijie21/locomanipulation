"""BiGym 2.0's reach and top-drawer tasks with one command per reach: A (straight reach) and J (cuRobo's joint trajectory
as the controller's reference), on BiGym's own controller (bigym_20261124; the BiGym part of the main table).

The controller is BiGym's native one (bigym.loco.make with the task's official config): GR00T-WBC Balance/Walk ONNX on
the legs and the waist, the arms' absolute joint targets straight into PD (no gravity compensation: they sag and
overshoot). Only the right arm moves; the base, the legs and the left arm hold env.raw_hold_action() taken at the reset,
and both grippers are commanded open (0) except in the drawer's fixed sequence. One command per reach: nothing measured
of the hand after the start corrects anything.

--stance (opt-in, not the brief's design; see "why" below): before the reach the base walks (GR00T-WBC's own velocity
command, vx WALK_VX, closed on the pelvis's floating-base readings along its heading at the look) until the perceived
goal sits STAND_X ahead of the pelvis, settles WALK_SETTLE_S, then steps sideways (vy, likewise) until the goal sits
STAND_Y to the pelvis's left (-0.12: in front of the right shoulder, so the right arm does not cross the torso; without
it, goals near the midline drove the arm into the torso), settles again (each skipped under WALK_MIN); for the drawer
the height command then drops to SQUAT_H for SQUAT_S (the handle is below the right arm's grasp workspace standing). The look stays the
one at the reset: the goal is carried into the new pelvis frame by the pelvis's own readings (F0 -> now). For the drawer
the pull is a walk back (vx PULL_VX, PULL_WALK_M, the arm holding its targets) instead of the arm line. Why: with the
base held, 5 of 10 dev reach targets are beyond the right arm (the elbow stops at -1.047 rad, the arm cannot straighten),
the drawer handle is out of the grasp workspace on 10 of 10 dev seeds (7-11 cm short) and a 36 cm arm pull is
impossible.

--method servo (diagnostic, not a table method): A's targets plus an integral of (target - measured joints) per joint
(SERVO_KI, capped SERVO_MAX): the sag taken out by joint feedback, an upper bound for a sag-free open-loop reach.

An episode:
  1. reset(seed); the look: one 640x360 RGB-D frame from the head camera (inner_env.cameras_map["head"], fovy 60, fx = fy
     = 311.77, cx = 319.5, cy = 179.5, MuJoCo's linear depth along the optical axis) by a renderer of our own on the
     env's model and data (multisampling off, so edge pixels are one geom's), turned into points in the pelvis frame by
     the camera's pose relative to the pelvis from the robot's own joint state (a scratch MjData of the same model with
     the robot's joint readings: the waist angles carry the torso-mounted camera). The robot's own pixels (the robot's
     geoms in a segmentation pass, dilated SELF_DIL px) are dropped. No object's simulated state enters any decision;
     the true target / drawer state is read only after the episode, for the row's scoring columns.
  2. perception (rules below) -> the goal pose of right_end_effector (the Dex1 pinch centre, 0.148 m beyond
     right_wrist_yaw_link) in the pelvis frame at the look.
  3. the reach, its timing that of Motor.reach_joints (locomani.backends.scalebfm.motor): min-jerk in the hand point's arc
     length, lasting max(REACH_MIN_S, length / REACH_SPEED), the start gap (the arm's current PD targets minus its
     measured joints: the sag) faded out linearly along the arc, one arm target per 20 ms control step.
       A  right_end_effector on a straight line from its measured pose to the goal pose (orientation slerped with the
          same min-jerk fraction), resolved every control step by damped least-squares IK on the 7 right-arm joints in
          BiGym's own MuJoCo model (a scratch MjData; the IK warm-starts from the previous step's solution).
       J  cuRobo plans the 7 right-arm joints from the measured start to the same goal pose, the look's voxels as
          obstacles (plan_server.py in curobo/, plan2.py's way: voxels within `free` of the start's spheres dropped, the
          target's / handle's own points dropped). The waist is locked: the server's robot has the waist at 0 and gets
          everything in the frame of the "virtual pelvis" that carries the real torso with the waist at 0 (exact for the
          arm, whose chain starts at the torso). The path is retimed as reach_joints does. No plan: A, recorded.
     The goal pose both methods aim at: the goal position with an orientation from our IK. Reach: DLS IK from the
     measured joints to the target centre, the start's hand orientation kept at a small weight (GOAL_ROT_W), the goal
     pose then FK of that solution (the nearest reachable pose: a target beyond the right arm's reach is aimed at as
     close as the arm gets; the row's goal_ik_cm says how far that is). Drawer: the rule's pregrasp pose, IK at full
     weight, then FK of the solution likewise.
  4. drawer only, after the reach, a fixed sequence identical for every method (straight lines by the same DLS IK and
     timing rule, starting from the reach's last target, gap 0): hold SETTLE_S, go straight in IN_M along the drawer
     front's normal (to the grasp), hold IN_HOLD_S, close the gripper (1) and hold CLOSE_S, pull back PULL_M along the
     normal (--stance: the walk back), open the gripper (0) over CLOSE_S (--stance: still walking back), hold to the
     episode's end. Letting go: the drawer (damped, frictionloss, no spring) stays where it was pulled, while the base's
     step forward when it stops walking back dragged it from 0.93 to 0.79 while held (dev seed 720002). An arm line
     whose IK leaves the hand more than IK_GIVE_UP from the line stops at the last point it reached (no flinging).
  5. the episode runs to its official end (the task's budget, or the env ending it on held success or on a physics
     error); success is bigym.loco.eval.is_success (held success, success_hold_seconds = 1 s, and never fell).

Perception rules (the look only, and the robot's own state):
  reach   the target sphere by colour: pixels with R >= RED_MIN and G, B < RED_RATIO * R (the scene's only red; the floor
          is blue, the robot white and grey), not the robot's, the largest 8-connected blob; its points (pelvis frame);
          the centre by an algebraic least-squares sphere fit (the fitted radius is recorded; a fit outside 2-10 cm or
          fewer than MIN_PTS points fails the look).
  drawer  SAM 3 (tmux sam3, port 18832, scripts/rsi/sam3_server.py's protocol) on the RGB frame with the prompt "drawer
          handle"; of its instances with score >= SAM_MIN and >= MIN_PTS depth points (the robot's pixels dropped), those
          whose points' principal axis is horizontal (|axis z| < 0.5) and 5-35 cm long are drawer handles; the top
          drawer's is the one whose points' centroid is highest. The drawer front: a ring of pixels around the handle's
          mask (dilated RING_OUT px less dilated RING_IN px, not the robot's), a least-squares plane refitted once
          without points > 1 cm off it; its normal n points to the camera. The bar's axis b: the handle points'
          principal axis made orthogonal to n. The grasp: approach along -n (the hand's x, the direction the fingers
          point), the fingers (the hand's y, their sliding axis) across the bar, y = +-(x cross b): of the two, the
          one whose pregrasp and grasp IK solutions keep the right arm off the robot's own body in our model (deepest
          arm-body penetration <= 1 mm; mj_collision on the scratch data, the robot's geoms only), then the smaller
          joint motion from the measured joints (an earlier "nearer the hand's y" rule picked an arm-through-torso
          solution on 3 of 5 dev seeds); the grasp point is the handle points' centroid moved out along n until the finger tips
          (TIP_M beyond the pinch centre, our own gripper's geometry) stop MARGIN_M short of the front plane:
          g = c + n * max(0, TIP_M + MARGIN_M - standoff), standoff = (c - front) . n. The pregrasp is PRE_M out along
          n from the grasp (the reach's goal); the fixed sequence goes IN_M = PRE_M in, back to the grasp.
          PULL_M = 0.36 m: 95 % of the drawer's 0.38 m slide (the cabinet asset's travel, the same for every episode;
          not the episode's state): a closed drawer ends at 0.95 > the task's 0.9; one that starts open (0-0.057 m by
          the g1_id_v1 reset) meets its stop first.
  J's obstacles: every depth point of the frame that is not the robot's and not the target's / handle's (their mask
          dilated OBJ_DIL px, points within OBJ_CLEAR of their points and, for the reach, within the fitted radius +
          OBJ_CLEAR of the fitted centre), inside BOX (pelvis frame), as VOX cubes.

Row columns beyond the brief's: err_end_cm (right_end_effector, the sim's site, to the aimed pose at the reach's last
target), err_settled_cm (the same 1 s after it, SETTLE_S for the drawer), err_grasp_cm (to the grasp point at the
in-move's last target), goal.goal_ik_cm (how far the arm's best pose is from the perceived goal), twin (the finger
rule's two candidates), contacts (the right arm's contacts with other bodies at each phase's end, scored only), truth
(scored only: the target's true centre and percept error, the drawer's start, max and end states).

Output: one JSON row per episode appended to --out, a "film" npz per episode in --film (the self model's inputs: per
control step the arm's joint targets sent, the measured arm joints and velocities, the waist joints, the pelvis pose and
right_end_effector's pose, world, and the whole robot's readings: root (the pelvis's position and wxyz quaternion),
root_v (its velocity: linear in the world, angular in the pelvis frame), qj / dqj (every other hinge and slide joint,
named in jnt); the reset state first), and the look in the same folder. The eval seed block (620000-629999,
bigym.loco.eval.EVAL_SEED_RESERVED) is refused without --allow-eval-seeds.

    export BIGYM_EGL_GPU=0; source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh
    cd <repo>; python scripts/experiments/bigym_20261124/bigym_bench.py --task drawer_top_open --method J \
        --seeds 720000:720010 --out outputs/bigym_20261124/smoke/rows.jsonl --film outputs/bigym_20261124/smoke/film
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing.connection import Client
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]
TASKS = ("reach_target_multi_modal", "drawer_top_open")
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
SITE, PELVIS, TORSO = "right_end_effector", "pelvis", "torso_link"
EVAL_LO, EVAL_HI = 620000, 630000  # bigym.loco.eval.EVAL_SEED_RESERVED

# the look
W, H, FX, CX, CY = 640, 360, 311.7691453623979, 319.5, 179.5
SELF_DIL, OBJ_DIL, OBJ_CLEAR, MIN_PTS = 2, 4, 0.02, 30
RED_MIN, RED_RATIO = 40, 0.4
SAM_ADDRESS, SAM_AUTHKEY, SAM_PROMPT, SAM_MIN = ("127.0.0.1", 18832), b"locomani-vision-rsi", "drawer handle", 0.5
RING_IN, RING_OUT = 5, 20  # px
VOX = 0.02  # m
BOX = ((-0.10, 0.90), (-0.60, 0.60), (-0.45, 0.75))  # m, pelvis frame: ahead, left, up
# the reach (Motor.reach_joints' timing)
DT, REACH_SPEED, REACH_MIN_S = 0.02, 0.15, 0.6
GOAL_ROT_W, PATH_ROT_W, LAM = 0.02, 0.5, 0.05
# the drawer's grasp rule and fixed sequence
TIP_M, MARGIN_M, PRE_M, IN_M, PULL_M = 0.037, 0.01, 0.08, 0.08, 0.36
SETTLE_S, IN_HOLD_S, CLOSE_S, IK_GIVE_UP = 0.5, 0.3, 0.6, 0.02
PULL_WALK_M = 0.42  # --stance: the walk back, the drawer's 0.38 m travel and ~4 cm of the arm's give (dev seeds)
# --stance (not the default: the brief holds the base): the walk to a stance, and the drawer's pull by walking back
STAND_X = {"reach_target_multi_modal": 0.38, "drawer_top_open": 0.45}  # m ahead of the pelvis: the target, the grasp
STAND_Y = {"reach_target_multi_modal": -0.12, "drawer_top_open": -0.12}  # m to its left: in front of the right shoulder
SQUAT_H, SQUAT_S = 0.62, 3.0  # drawer: the height command after the walk (0.74 standing) and its settle
WALK_MIN, WALK_VX, WALK_LEAD, WALK_SETTLE_S, PULL_VX = 0.02, 0.2, 0.01, 2.5, -0.3
SERVO_KI, SERVO_MAX = 0.05, 0.3  # servo (a check, not a method): integral of (target - reading) per step, rad cap
PLAN_AUTHKEY = b"bigym-jref"


# ----------------------------------------------------------------------------------------------------------- geometry
def minjerk(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def quat_of(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).ravel())
    return q


def mat_of(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def rotvec(R: np.ndarray) -> np.ndarray:
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, quat_of(R), 1.0)
    return v


def rot_of(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(v, float) / th, th)
    return mat_of(q)


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(rotvec(Ra.T @ Rb))))


# ------------------------------------------------------------------------------------------ the robot's own kinematics
def arm_bodies(m, ns: str) -> set:
    """The right arm's and gripper's bodies: right_shoulder_pitch_link and everything under it."""
    top = m.body(ns + "right_shoulder_pitch_link").id
    out = set()
    for b in range(m.nbody):
        c = b
        while c > 0 and c != top:
            c = m.body_parentid[c]
        if c == top:
            out.add(b)
    return out


class Kin:
    """BiGym's own MuJoCo model on a scratch MjData: the robot's joints copied from the env, nothing else; poses in the
    pelvis frame. FK and DLS IK of right_end_effector on the 7 right-arm joints, the head camera's and the torso's pose
    relative to the pelvis."""

    def __init__(self, env):
        inner = env.inner_env
        m, ns = inner.model, inner.robot.namespace
        self.m, self.d = m, mujoco.MjData(m)
        self.pelvis = m.body(ns + PELVIS).id
        self.torso = m.body(ns + TORSO).id
        self.site = m.site(ns + SITE).id
        self.cam = int(inner.cameras_map["head"][0])
        root = m.body_rootid[self.pelvis]
        self.robot_q = np.array([m.jnt_qposadr[j] for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == root])
        self.robot_bodies = np.array([m.body_rootid[b] == root for b in range(m.nbody)])
        # the film's whole-body readings: every hinge and slide joint of the robot but the floating base's (BiGym's
        # floating base is 3 slides and 3 hinges on the pelvis; the film keeps the pelvis's pose and velocity instead)
        jr = [j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == root and m.jnt_bodyid[j] != self.pelvis]
        hs = [j for j in jr if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        self.v6 = np.zeros(6)
        self.jnt = [m.joint(j).name[len(ns) :] for j in hs]
        self.jq, self.jv = np.array([m.jnt_qposadr[j] for j in hs]), np.array([m.jnt_dofadr[j] for j in hs])
        jid = [m.joint(ns + n).id for n in ARM]
        self.arm_q = np.array([m.jnt_qposadr[j] for j in jid])
        self.arm_v = np.array([m.jnt_dofadr[j] for j in jid])
        self.lo, self.hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        self.waist_q = np.array([m.jnt_qposadr[m.joint(ns + n).id] for n in WAIST])
        self.right = arm_bodies(m, ns)
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def load(self, d_env) -> None:
        """The robot's joint readings (floating base, waist, arms, fingers) into the scratch data."""
        self.d.qpos[self.robot_q] = d_env.qpos[self.robot_q]
        mujoco.mj_kinematics(self.m, self.d)

    def _rel(self, p, R):
        Rp, pp = self.d.xmat[self.pelvis].reshape(3, 3), self.d.xpos[self.pelvis]
        return Rp.T @ (p - pp), Rp.T @ R

    def _set_arm(self, q) -> None:
        self.d.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.m, self.d)

    def site_pose(self, q=None):
        if q is not None:
            self._set_arm(q)
        return self._rel(self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy())

    def cam_pose(self):
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)  # camera poses are not part of mj_kinematics
        return self._rel(self.d.cam_xpos[self.cam].copy(), self.d.cam_xmat[self.cam].reshape(3, 3).copy())

    def torso_pose(self, waist=None):
        if waist is not None:
            keep = self.d.qpos[self.waist_q].copy()
            self.d.qpos[self.waist_q] = waist
            mujoco.mj_kinematics(self.m, self.d)
        out = self._rel(self.d.xpos[self.torso].copy(), self.d.xmat[self.torso].reshape(3, 3).copy())
        if waist is not None:
            self.d.qpos[self.waist_q] = keep
            mujoco.mj_kinematics(self.m, self.d)
        return out

    def _jac(self, q):
        self._set_arm(q)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
        Rp = self.d.xmat[self.pelvis].reshape(3, 3)
        p, R = self._rel(self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy())
        return p, R, Rp.T @ self.jacp[:, self.arm_v], Rp.T @ self.jacr[:, self.arm_v]

    def self_contact(self, q) -> float:
        """The robot's own model with the right arm at q (the rest of the body at its readings): the deepest
        penetration (mm) between a right-arm/gripper body and another robot body, 0 if none. Scene geoms are ignored
        (the scratch data's objects are not where the scene's are)."""
        self._set_arm(q)
        mujoco.mj_collision(self.m, self.d)
        worst = 0.0
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if not (self.robot_bodies[b1] and self.robot_bodies[b2]):
                continue
            if (b1 in self.right) != (b2 in self.right):
                worst = max(worst, -1000.0 * float(c.dist))
        return worst

    def ik(self, p_t, R_t, q0, rot_w: float, iters: int = 30, tol_p: float = 2e-4, tol_r: float = 2e-3):
        """Damped least squares on the 7 arm joints, clamped to their ranges. -> q, position error (m), angle (rad)."""
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        for _ in range(iters):
            p, R, Jp, Jr = self._jac(q)
            ep, er = p_t - p, rotvec(R_t @ R.T)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                break
            J, e = np.vstack([Jp, rot_w * Jr]), np.r_[ep, rot_w * er]
            dq = J.T @ np.linalg.solve(J @ J.T + LAM**2 * np.eye(6), e)
            n = float(np.linalg.norm(dq))
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.site_pose(q)
        return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(rotvec(R_t @ R.T)))


# --------------------------------------------------------------------------------------------------------- the look
class Look:
    """One RGB-D frame of the head camera and its points in the pelvis frame."""

    def __init__(self, env, kin: Kin, renderer):
        inner = env.inner_env
        m, d = inner.model, inner.data
        renderer.update_scene(d, camera=kin.cam)
        self.rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera=kin.cam)
        self.depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera=kin.cam)
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        geom = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        robot_geom = kin.robot_bodies[m.geom_bodyid]
        own = np.zeros(geom.shape, bool)
        own[geom] = robot_geom[seg[..., 0][geom]]
        self.own = ndimage.binary_dilation(own, iterations=SELF_DIL)  # the robot's own pixels
        self.valid = np.isfinite(self.depth) & (self.depth > 0.05) & (self.depth < 3.0)
        # the camera relative to the pelvis, from the robot's joint state (kin holds the readings)
        self.cam_p, self.cam_R = kin.cam_pose()
        v, u = np.mgrid[0:H, 0:W]
        z = self.depth
        pc = np.stack([(u - CX) / FX * z, -(v - CY) / FX * z, -z], axis=-1)  # MuJoCo camera: x right, y up, looks -z
        self.P = pc @ self.cam_R.T + self.cam_p  # (H, W, 3) pelvis frame

    def points(self, mask: np.ndarray) -> np.ndarray:
        return self.P[mask & self.valid & ~self.own]


def sphere_fit(X: np.ndarray):
    A = np.c_[2 * X, np.ones(len(X))]
    sol = np.linalg.lstsq(A, (X**2).sum(1), rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 0.0)))


def find_target(look: Look) -> dict:
    rgb = look.rgb.astype(float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (r >= RED_MIN) & (g < RED_RATIO * r) & (b < RED_RATIO * r) & ~look.own
    lab, n = ndimage.label(red, structure=np.ones((3, 3)))
    if n == 0:
        return dict(ok=False, why="no red pixels")
    sizes = ndimage.sum(red, lab, range(1, n + 1))
    mask = lab == 1 + int(np.argmax(sizes))
    X = look.points(mask)
    out = dict(px=int(mask.sum()), n_pts=len(X), blobs=n, mask=mask)
    if len(X) < MIN_PTS:
        return dict(out, ok=False, why=f"{len(X)} points")
    c, rad = sphere_fit(X)
    out.update(centre=c, radius_cm=round(100 * rad, 2), cloud=X)
    if not 0.02 <= rad <= 0.10:
        return dict(out, ok=False, why=f"sphere fit radius {100 * rad:.1f} cm")
    return dict(out, ok=True)


def sam3(rgb: np.ndarray, prompt: str) -> list:
    c = Client(SAM_ADDRESS, authkey=SAM_AUTHKEY)
    try:
        c.send(dict(image=rgb, prompts=[prompt]))
        rep = c.recv()
    finally:
        c.close()
    if "error" in rep:
        raise RuntimeError(f"SAM 3: {rep['error']}")
    out = []
    for s, pk in rep["results"][prompt]:
        m = np.unpackbits(pk["bits"], count=int(np.prod(pk["shape"]))).astype(bool).reshape(pk["shape"])
        out.append((float(s), m))
    return out


def plane_fit(X: np.ndarray):
    c = X.mean(0)
    n = np.linalg.svd(X - c, full_matrices=False)[2][-1]
    return c, n


def find_handle(look: Look, R_hand: np.ndarray) -> dict:
    t0 = time.time()
    inst = sam3(look.rgb, SAM_PROMPT)
    t_sam = time.time() - t0
    cands = []
    for i, (s, m) in enumerate(inst):
        X = look.points(m)
        rec = dict(i=i, score=round(s, 3), px=int(m.sum()), n_pts=len(X))
        if s >= SAM_MIN and len(X) >= MIN_PTS:
            c = X.mean(0)
            w, V = np.linalg.eigh(np.cov((X - c).T))
            a = V[:, -1]
            proj = (X - c) @ a
            rec.update(
                c=c.round(4).tolist(), axis=a.round(3).tolist(), len_cm=round(100 * float(proj.max() - proj.min()), 1)
            )
            rec["handle"] = bool(abs(a[2]) < 0.5 and 0.05 <= proj.max() - proj.min() <= 0.35)
        cands.append(rec)
    handles = [r for r in cands if r.get("handle")]
    out = dict(t_sam_s=round(t_sam, 3), cands=cands)
    if not handles:
        return dict(out, ok=False, why="no horizontal handle among SAM 3's instances")
    top = max(handles, key=lambda r: r["c"][2])
    mask = inst[top["i"]][1]
    X = look.points(mask)
    c = X.mean(0)
    ring = ndimage.binary_dilation(mask, iterations=RING_OUT) & ~ndimage.binary_dilation(mask, iterations=RING_IN)
    F = look.points(ring)
    if len(F) < MIN_PTS:
        return dict(out, ok=False, why=f"{len(F)} drawer-front points")
    fc, n = plane_fit(F)
    keep = np.abs((F - fc) @ n) < 0.01
    fc, n = plane_fit(F[keep])
    if n @ (look.cam_p - fc) < 0:
        n = -n
    w, V = np.linalg.eigh(np.cov((X - c).T))
    b = V[:, -1] - (V[:, -1] @ n) * n
    b /= np.linalg.norm(b)
    x = -n
    y = np.cross(x, b)
    y /= np.linalg.norm(y)
    Rg = np.column_stack([x, y, np.cross(x, y)])  # and its twin turned 180 deg about x (the caller picks one)
    standoff = float((c - fc) @ n)
    off = max(0.0, TIP_M + MARGIN_M - standoff)
    grasp = c + off * n
    pre = grasp + PRE_M * n
    out.update(ok=True, top=top["i"], score=top["score"], n_pts=len(X), n_front=int(keep.sum()), centroid=c, normal=n,
               bar_axis=b, standoff_cm=round(100 * standoff, 2), out_cm=round(100 * off, 2), grasp=grasp, pre=pre,
               R=Rg, R_twin=Rg @ np.diag([1.0, -1.0, -1.0]),
               tilt_deg=round(float(np.degrees(np.arcsin(abs(n[2])))), 2), mask=mask, cloud=X)  # fmt: skip
    return out


def voxels(look: Look, obj_mask, obj_cloud, T_vp, T_F, sphere=None) -> tuple[np.ndarray, int]:
    """J's obstacles: the frame's points less the robot's and the object's, moved from the look's pelvis frame into the
    current one (T_F), cropped to BOX, in the virtual pelvis's frame as VOX cubes (centres). sphere = (centre, radius):
    the reach target's fit; every point within radius + OBJ_CLEAR of its centre is the target's too (its shaded rim
    fails the colour rule and was left as an obstacle the hand's goal sits inside)."""
    keep = look.valid & ~look.own
    if obj_mask is not None:
        keep &= ~ndimage.binary_dilation(obj_mask, iterations=OBJ_DIL)
    X = look.P[keep]
    n_obj = 0
    if obj_cloud is not None and len(obj_cloud):
        from scipy.spatial import cKDTree

        near = cKDTree(obj_cloud).query(X)[0] < OBJ_CLEAR
        if sphere is not None:
            near |= np.linalg.norm(X - sphere[0], axis=1) < sphere[1] + OBJ_CLEAR
        n_obj = int(near.sum())
        X = X[~near]
    X = X @ T_F[0].T + T_F[1]
    (x0, x1), (y0, y1), (z0, z1) = BOX
    X = X[(X[:, 0] > x0) & (X[:, 0] < x1) & (X[:, 1] > y0) & (X[:, 1] < y1) & (X[:, 2] > z0) & (X[:, 2] < z1)]
    Rv, tv = T_vp
    Xv = (X - tv) @ Rv  # pelvis -> virtual pelvis
    V = np.unique(np.floor(Xv / VOX).astype(np.int64), axis=0)
    return ((V + 0.5) * VOX).astype(np.float32), n_obj


# ------------------------------------------------------------------------------------------------- reference paths
def line_refs(kin: Kin, p0, R0, p1, R1, q_ref0, q_seed, rot_w: float = PATH_ROT_W):
    """A's straight line: the hand point p0 -> p1 (orientation slerped by the same fraction), min-jerk over
    max(REACH_MIN_S, |p1 - p0| / REACH_SPEED), DLS IK each control step warm-started from the last; the start gap
    q_ref0 - IK(p0) faded out along the line."""
    L = float(np.linalg.norm(p1 - p0))
    total = max(REACH_MIN_S, L / REACH_SPEED)
    N = int(math.ceil(total / DT - 1e-9))
    dR = rotvec(R0.T @ R1)
    q, ep0, _ = kin.ik(p0, R0, q_seed, rot_w, iters=60)
    gap = np.asarray(q_ref0, float) - q
    refs, worst, good, stuck = [], 0.0, q.copy(), None
    for k in range(N):
        u = minjerk(min((k + 1) * DT, total) / total)
        q, ep, er = kin.ik(p0 + u * (p1 - p0), R0 @ rot_of(u * dR), good, rot_w, iters=20)
        worst = max(worst, ep)
        if ep <= IK_GIVE_UP and stuck is None:
            good = q
        elif stuck is None:  # beyond the arm's reach: the rest of the line holds the last point it reached
            stuck = k
        refs.append(good + (1.0 - u) * gap)
    info = dict(path_cm=round(100 * L, 2), dur_s=round(total, 2), gap_deg=round(float(np.degrees(np.abs(gap).max())), 2),
                ik_worst_mm=round(1000 * worst, 2), ik_end_mm=round(1000 * ep, 2), ik_end_deg=round(math.degrees(er), 2),
                stuck_at=None if stuck is None else round(stuck / N, 3))  # fmt: skip
    return np.array(refs), info


def joint_refs(kin: Kin, Q: np.ndarray, q_ref0):
    """J: Motor.reach_joints' retiming of a joint path: min-jerk in the hand point's arc length (FK of the path in
    BiGym's model), max(REACH_MIN_S, length / REACH_SPEED), the start gap q_ref0 - Q[0] faded out along the arc."""
    P = np.array([kin.site_pose(q)[0] for q in Q])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    L = float(arc[-1])
    total = max(REACH_MIN_S, L / REACH_SPEED)
    frac = arc / L if L > 1e-6 else np.linspace(0.0, 1.0, len(Q))
    gap = np.asarray(q_ref0, float) - Q[0]
    N = int(math.ceil(total / DT - 1e-9))
    refs = []
    for k in range(N):
        u = minjerk(min((k + 1) * DT, total) / total)
        refs.append(np.array([np.interp(u, frac, Q[:, i]) for i in range(Q.shape[1])]) + (1.0 - u) * gap)
    info = dict(path_cm=round(100 * L, 2), dur_s=round(total, 2), gap_deg=round(float(np.degrees(np.abs(gap).max())), 2),
                plan_T=len(Q))  # fmt: skip
    return np.array(refs), info


def plan_j(sock: str, kin: Kin, q_meas, vox, goal_p, goal_R, T_vp, timeout: float):
    """cuRobo's plan (curobo/plan_server.py) in the virtual pelvis's frame. -> Q (T, 7) in ARM order or None, info"""
    Rv, tv = T_vp
    req = dict(q={n: float(v) for n, v in zip(ARM, q_meas)}, vox=vox, vox_size=VOX, goal_p=(goal_p - tv) @ Rv,
               goal_q=quat_of(Rv.T @ goal_R), attempts=4, timeout=timeout, free=0.01)  # fmt: skip
    c = Client(sock, "AF_UNIX", authkey=PLAN_AUTHKEY)
    try:
        c.send(req)
        rep = c.recv()
    finally:
        c.close()
    info = {k: rep.get(k) for k in ("ok", "status", "t_plan", "n_vox", "n_vox_dropped", "start_clear_cm", "end_err_mm",
                                    "end_err_deg")}  # fmt: skip
    info = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in info.items()}
    fk_p = rep.get("fk_p")
    if fk_p is not None:  # the start's hand by cuRobo against BiGym's (virtual pelvis frame)
        p0 = (kin.site_pose(q_meas)[0] - tv) @ Rv
        info["fk_start_mm"] = round(1000 * float(np.linalg.norm(np.asarray(fk_p, float) - p0)), 2)
    if not rep.get("ok") or rep.get("Q") is None:
        return None, info
    names = list(rep["names"])
    Q = np.asarray(rep["Q"], float)[:, [names.index(n) for n in ARM]]
    return Q, info


# --------------------------------------------------------------------------------------------------------- episode
def r3(x, k=4):
    return np.round(np.asarray(x, float), k).tolist()


class Run:
    """The episode's control steps: one raw action per step (the hold action with the right arm's targets, the right
    gripper and the base's forward speed replaced), the film, the official success clock."""

    def __init__(self, env, kin: Kin, task: str, servo: bool = False):
        self.env, self.kin, self.task, self.servo = env, kin, task, servo
        self.d = env.inner_env.data
        lay = env.wholebody_action_layout()
        base_dim, names = lay["base_dim"], list(lay["limb_names"])
        self.slots = np.array([base_dim + names.index(n) for n in ARM])
        n_act = int(env.action_spec().shape[0])
        self.g_right = n_act - 1
        self.hold = env.raw_hold_action().astype(np.float64)
        self.hold[[n_act - 2, n_act - 1]] = 0.0  # both grippers open
        self.ref, self.grip, self.k, self.done = self.hold[self.slots].copy(), 0.0, 0, False
        self.integ = np.zeros(len(ARM))
        self.t_success, self.drawer_max, self.marks, self.mark_at = None, 0.0, {}, {}
        self.film = {k: [] for k in ("t", "phase", "ref", "sent", "grip", "vx", "vy", "q", "dq", "waist", "pelvis_p",
                                     "pelvis_q", "ee_p", "ee_q", "root", "root_v", "qj", "dqj")}  # fmt: skip
        self.phases, self.contacts = ["reset"], {}
        self.right = kin.right
        self.record("reset", 0.0)

    def record(self, phase: str, vx: float, vy: float = 0.0) -> None:
        d, kin, f = self.d, self.kin, self.film
        f["t"].append(self.k * DT)
        f["phase"].append(phase)
        f["ref"].append(self.ref.copy())
        f["sent"].append(self.ref + self.integ)
        f["grip"].append(self.grip)
        f["vx"].append(vx)
        f["vy"].append(vy)
        f["q"].append(d.qpos[kin.arm_q].copy())
        f["dq"].append(d.qvel[kin.arm_v].copy())
        f["waist"].append(d.qpos[kin.waist_q].copy())
        f["pelvis_p"].append(d.xpos[kin.pelvis].copy())
        f["pelvis_q"].append(d.xquat[kin.pelvis].copy())
        f["ee_p"].append(d.site_xpos[kin.site].copy())
        f["ee_q"].append(quat_of(d.site_xmat[kin.site].reshape(3, 3)))
        f["root"].append(np.r_[d.xpos[kin.pelvis], d.xquat[kin.pelvis]])
        mujoco.mj_objectVelocity(kin.m, d, mujoco.mjtObj.mjOBJ_BODY, kin.pelvis, kin.v6, 0)  # world axes: [rot, lin]
        lin = kin.v6[3:].copy()
        mujoco.mj_objectVelocity(kin.m, d, mujoco.mjtObj.mjOBJ_BODY, kin.pelvis, kin.v6, 1)  # the pelvis's axes
        f["root_v"].append(np.r_[lin, kin.v6[:3]])
        f["qj"].append(d.qpos[kin.jq].astype(np.float32))
        f["dqj"].append(d.qvel[kin.jv].astype(np.float32))

    def pelvis(self):
        """The pelvis's pose (world), the robot's floating-base joint readings (BiGym's proprioception_floating_base
        and its passive tilt)."""
        return self.d.xpos[self.kin.pelvis].copy(), self.d.xmat[self.kin.pelvis].reshape(3, 3).copy()

    def hand(self) -> np.ndarray:
        """right_end_effector in the current pelvis frame (the simulator's site: scored only)."""
        p, R = self.pelvis()
        return R.T @ (self.d.site_xpos[self.kin.site] - p)

    def step(self, phase: str, ref=None, grip=None, vx: float = 0.0, vy: float = 0.0) -> bool:
        if self.done:
            return True
        if self.phases[-1] != phase:
            self.contacts[self.phases[-1]] = self.arm_contacts()
            self.phases.append(phase)
        if ref is not None:
            self.ref = np.asarray(ref, float)
        if grip is not None:
            self.grip = float(grip)
        raw = self.hold.copy()
        if self.servo:  # joint readings only: the sag integrated out
            err = self.ref - self.d.qpos[self.kin.arm_q]
            self.integ = np.clip(self.integ + SERVO_KI * err, -SERVO_MAX, SERVO_MAX)
        raw[self.slots] = self.ref + self.integ
        raw[self.g_right] = self.grip
        raw[0], raw[1] = vx, vy
        act = np.clip(self.env.normalize_action(raw.astype(np.float32)), -1.0, 1.0).astype(np.float32)
        ts = self.env.step(act)
        self.k += 1
        self.record(phase, vx, vy)
        if self.t_success is None and self.env.episode_succeeded():
            self.t_success = round(self.k * DT, 2)
        if self.task == "drawer_top_open":  # scored only
            self.drawer_max = max(self.drawer_max, float(self.env.inner_env.cabinet_drawers.get_state()[-1]))
        for name, k in list(self.mark_at.items()):
            if self.k == k:
                self.marks[name] = self.hand()
        self.done = bool(ts.last())
        return self.done

    def path(self, phase: str, refs, grip=None) -> None:
        for r in refs:
            if self.step(phase, r, grip):
                return
        self.marks[phase] = self.hand()

    def wait(self, phase: str, seconds: float, grip=None) -> None:
        for _ in range(int(round(seconds / DT))):
            if self.step(phase, None, grip):
                return

    def walk(self, phase: str, dist: float, vx: float, cap_s: float, axis: int = 0) -> float:
        """Base speed vx along the pelvis's heading at the start (axis 0; axis 1: sideways, its y, the command's vy)
        until the pelvis (its floating-base readings) has moved `dist` along it less WALK_LEAD (what it drifts on after
        the command stops), at most cap_s; the arm holds its targets. -> the distance moved."""
        p0, R0 = self.pelvis()
        h = np.array([R0[0, axis], R0[1, axis], 0.0])
        h /= np.linalg.norm(h)
        sign = 1.0 if vx > 0 else -1.0
        moved = 0.0
        for _ in range(int(round(cap_s / DT))):
            moved = float(sign * (self.pelvis()[0] - p0) @ h)
            if moved >= dist - WALK_LEAD or self.step(phase, None, None, *((vx, 0.0) if axis == 0 else (0.0, vx))):
                break
        return float(sign * (self.pelvis()[0] - p0) @ h)

    def arm_contacts(self) -> list:
        """Contacts of the right arm and gripper at a phase's end: [other body, deepest penetration mm] (the
        simulator's, scored only)."""
        m, d, out = self.env.inner_env.model, self.d, {}
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self.right) == (b2 in self.right):
                continue
            other = b2 if b1 in self.right else b1
            name = m.body(other).name or f"body{other}"
            out[name] = max(out.get(name, 0.0), round(-1000 * float(c.dist), 1))
        return sorted(out.items())

    def to_end(self, phase: str = "hold") -> None:
        while not self.step(phase):
            pass


def episode(env, kin: Kin, renderer, task: str, seed: int, method: str, a, index: int) -> dict:
    from bigym.loco.eval import is_success

    t_wall = time.time()
    inner = env.inner_env
    d = inner.data
    env.reset(seed=seed)
    # scored only, never read by a decision: the drawer's state at the reset, for the row's truth columns
    drawer0 = float(inner.cabinet_drawers.get_state()[-1]) if task == "drawer_top_open" else None
    run = Run(env, kin, task, servo=method == "servo")
    row = dict(task=task, seed=seed, index=index, method=method, stance=bool(a.stance), bigym_commit="35ec409")

    # the look (frame F0: the pelvis at the look)
    t0 = time.time()
    kin.load(d)
    look = Look(env, kin, renderer)
    _, R_hand0 = kin.site_pose()
    if task.startswith("reach"):
        per = find_target(look)
        row["look"] = {k: v for k, v in per.items() if k in ("ok", "why", "px", "n_pts", "blobs", "radius_cm")}
        if per["ok"]:
            row["look"]["centre"] = r3(per["centre"])
    else:
        per = find_handle(look, R_hand0)
        row["look"] = {k: (r3(v) if isinstance(v, np.ndarray) else v) for k, v in per.items()
                       if k not in ("mask", "cloud", "R", "R_twin")}  # fmt: skip
    obj_mask, obj_cloud = per.get("mask"), per.get("cloud")
    row["look_s"] = round(time.time() - t0, 3)
    if a.film:
        os.makedirs(a.film, exist_ok=True)
        np.savez_compressed(f"{a.film}/{task}_{seed}_{method}_look.npz", rgb=look.rgb, depth=look.depth.astype(np.float32),
                            own=look.own, obj=obj_mask if obj_mask is not None else np.zeros((H, W), bool),
                            cam_p=look.cam_p, cam_R=look.cam_R)  # fmt: skip
    pF0, RF0 = run.pelvis()

    # --stance: walk straight ahead until the goal lies STAND_X ahead (the reach's target centre, the drawer's grasp),
    # then sideways until it lies STAND_Y to the left (in front of the right shoulder: the right arm does not cross the
    # torso), each skipped under WALK_MIN
    if a.stance and per["ok"]:
        g0 = per["centre"] if task.startswith("reach") else per["grasp"]
        dist, side = float(g0[0]) - STAND_X[task], float(g0[1]) - STAND_Y[task]
        moved = moved_y = 0.0
        if dist > WALK_MIN:
            moved = run.walk("walk", dist, WALK_VX, dist / WALK_VX + 2.0)
            run.wait("walk_settle", WALK_SETTLE_S)
        if abs(side) > WALK_MIN:
            moved_y = math.copysign(
                run.walk("side", abs(side), math.copysign(WALK_VX, side), abs(side) / WALK_VX + 2.0, 1), side
            )
            run.wait("side_settle", WALK_SETTLE_S)
        row["walk"] = dict(want_cm=round(100 * dist, 1), moved_cm=round(100 * moved, 1), side_want_cm=round(100 * side, 1),
                           side_moved_cm=round(100 * moved_y, 1))  # fmt: skip
        if task == "drawer_top_open":  # squat: the handle comes up to the shoulder's reach (base height command)
            run.hold[2] = SQUAT_H
            run.wait("squat", SQUAT_S)
            row["walk"]["pelvis_dz_cm"] = round(100 * float(run.pelvis()[0][2] - pF0[2]), 1)

    # the reach's start: everything from F0 into the pelvis frame now (the floating-base readings at both times)
    kin.load(d)
    q_meas, ctrl0 = d.qpos[kin.arm_q].copy(), run.ref.copy()
    p_start, R_start = kin.site_pose(q_meas)
    pN, RN = run.pelvis()
    RF, tF = RN.T @ RF0, RN.T @ (pF0 - pN)  # F0 -> now
    tw, Rw = kin.torso_pose()
    tw0, Rw0 = kin.torso_pose(np.zeros(3))
    T_vp = (Rw @ Rw0.T, tw - (Rw @ Rw0.T) @ tw0)  # the virtual pelvis in the pelvis frame
    row["waist"] = r3(d.qpos[kin.waist_q])
    if per["ok"]:
        if task.startswith("reach"):
            goal_p, R_goal, goal_rot_w = RF @ per["centre"] + tF, R_start, GOAL_ROT_W
        else:
            goal_p, R_goal, goal_rot_w = RF @ per["pre"] + tF, RF @ per["R"], 1.0
            normal, grasp = RF @ per["normal"], RF @ per["grasp"] + tF
        q_goal, e_p, e_r = kin.ik(goal_p, R_goal, q_meas, goal_rot_w, iters=300, tol_p=1e-4)
        if task == "drawer_top_open":
            # the fingers' two ways across the bar (turned 180 deg about the approach): the one whose pregrasp and
            # grasp configurations keep the arm off the robot's own body (our model), then the smaller joint motion
            R_twin = RF @ per["R_twin"]
            q2, e_p2, e_r2 = kin.ik(goal_p, R_twin, q_meas, goal_rot_w, iters=300, tol_p=1e-4)
            score = []
            for q_, e_, R_ in ((q_goal, e_p, R_goal), (q2, e_p2, R_twin)):
                qg = kin.ik(goal_p - IN_M * normal, R_, q_, 1.0, iters=200)[0]
                pen = max(kin.self_contact(q_), kin.self_contact(qg))
                score.append(
                    (e_ >= 0.01, round(pen) if pen > 1.0 else 0, float(np.abs(q_ - q_meas).sum()), round(pen, 1))
                )
            if score[1][:3] < score[0][:3]:
                q_goal, e_p, e_r, R_goal = q2, e_p2, e_r2, R_twin
            row["twin"] = dict(self_mm=[score[0][3], score[1][3]], joint_sum_deg=[round(math.degrees(x[2]), 1) for x in score],
                               took=int(R_goal is R_twin))  # fmt: skip
        gp, gR = kin.site_pose(q_goal)  # the goal pose both methods aim at
        row["goal"] = dict(p=r3(goal_p), aim_p=r3(gp), aim_q=r3(quat_of(gR)), goal_ik_cm=round(100 * e_p, 2),
                           goal_ik_deg=round(math.degrees(e_r), 2))  # fmt: skip
        refs, info = None, {}
        if method == "J":
            t0 = time.time()
            sph = (per["centre"], per["radius_cm"] / 100.0) if task.startswith("reach") else None
            vox, n_obj = voxels(look, obj_mask, obj_cloud, T_vp, (RF, tF), sph)
            try:
                Q, pinfo = plan_j(a.plan_socket, kin, q_meas, vox, gp, gR, T_vp, a.plan_timeout)
            except (OSError, EOFError) as e:
                Q, pinfo = None, dict(ok=False, status=f"no plan server: {type(e).__name__}: {e}")
            row["plan"] = dict(pinfo, vox=len(vox), vox_obj_dropped=n_obj, wall_s=round(time.time() - t0, 3))
            if Q is not None:
                refs, info = joint_refs(kin, Q, ctrl0)
            row["plan_ok"], row["fallback"] = Q is not None, Q is None
        if refs is None:
            refs, info = line_refs(kin, p_start, R_start, gp, gR, ctrl0, q_meas)
        row["reach"] = info
        settle_s = SETTLE_S if task == "drawer_top_open" else 1.0  # the drawer's sequence moves on after SETTLE_S
        run.mark_at["reach_settled"] = run.k + len(refs) + int(round(settle_s / DT))
        row["settled_after_s"] = settle_s
        run.path("reach", refs, 0.0)
        if task == "drawer_top_open":
            q_end = refs[-1]
            p_pre, R_pre = kin.site_pose(q_end)  # the reach's last target's hand pose
            r_in, i_in = line_refs(kin, p_pre, R_pre, p_pre - IN_M * normal, R_pre, q_end, q_end)
            run.wait("settle", SETTLE_S)
            run.path("in", r_in)
            run.wait("in_hold", IN_HOLD_S)
            run.wait("close", CLOSE_S, 1.0)
            row["seq"] = dict(in_=i_in)
            if a.stance:  # the pull by walking back, the arm holding its targets
                row["seq"]["pull_walk_cm"] = round(
                    100 * run.walk("pull", PULL_WALK_M, PULL_VX, PULL_WALK_M / -PULL_VX + 3.0), 1
                )
            else:
                p_g, R_g = kin.site_pose(r_in[-1])
                r_pull, i_pull = line_refs(kin, p_g, R_g, p_g + PULL_M * normal, R_g, r_in[-1], r_in[-1])
                row["seq"]["pull"] = i_pull
                run.path("pull", r_pull)
            # let go: the drawer (damped, no spring) stays where it was pulled to, and the base's step forward when it
            # stops walking back no longer drags it shut; with --stance the walk back goes on while the fingers open
            for _ in range(int(round(CLOSE_S / DT))):
                if run.step("release", None, 0.0, PULL_VX if a.stance else 0.0):
                    break
    else:
        row["plan_ok"], row["fallback"] = (False, False) if method == "J" else (None, None)
    row.setdefault("plan_ok", None)
    row.setdefault("fallback", None)
    t0 = time.time()
    run.to_end()
    run.contacts[run.phases[-1]] = run.arm_contacts()
    row["contacts"] = {k: v for k, v in run.contacts.items() if v}
    fell = bool(env.episode_fell())
    row.update(success=bool(is_success(env)), t_success=run.t_success, fell=fell, termination=env.episode_termination(),
               steps=run.k, budget_steps=int(env._time_limit))  # fmt: skip
    if per["ok"]:
        for key, mk in (("err_end_cm", "reach"), ("err_settled_cm", "reach_settled")):
            if mk in run.marks:
                e = run.marks[mk] - gp
                row[key], row[key.replace("_cm", "_vec_cm")] = round(100 * float(np.linalg.norm(e)), 2), r3(100 * e, 2)
        if "in" in run.marks:
            row["err_grasp_cm"] = round(100 * float(np.linalg.norm(run.marks["in"] - grasp)), 2)
    # scoring columns only (the simulator's truth, read after the episode)
    if task.startswith("reach"):
        tgt = inner.targets[0].get_position()
        row["truth"] = dict(target_F0=r3(RF0.T @ (tgt - pF0)))
        if per["ok"]:
            row["truth"]["percept_err_cm"] = round(100 * float(np.linalg.norm(RF0.T @ (tgt - pF0) - per["centre"])), 2)
        ee = np.array(run.film["ee_p"])
        row["truth"]["min_dist_cm"] = round(100 * float(np.linalg.norm(ee - tgt, axis=1).min()), 2)
    else:
        row["truth"] = dict(drawer_start=round(drawer0, 3), drawer_max=round(run.drawer_max, 3),
                            drawer_end=round(float(inner.cabinet_drawers.get_state()[-1]), 3))  # fmt: skip
    row["wall_s"] = round(time.time() - t_wall, 2)
    row["sim_wall_s"] = round(time.time() - t0, 2)
    if a.film:
        meta = {k_: row.get(k_) for k_ in ("task", "seed", "index", "method", "stance", "success", "t_success", "fell",
                                           "plan_ok", "fallback")}  # fmt: skip
        np.savez_compressed(f"{a.film}/{task}_{seed}_{method}{'_stance' if a.stance else ''}.npz", meta=json.dumps(meta),
                            arm=np.array(ARM), jnt=np.array(kin.jnt), phases=np.array(run.phases), dt=DT,
                            **{k_: np.array(v) for k_, v in run.film.items()})  # fmt: skip
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument(
        "--method",
        required=True,
        choices=("A", "J", "servo"),
        help="servo: A plus a joint-space integral"
        " on the arm's readings, a check that the sequence works when the hand arrives (not a method)",
    )
    ap.add_argument("--seeds", required=True, help="lo:hi (hi exclusive) or a comma list")
    ap.add_argument("--out", required=True, help="jsonl, one row per episode appended")
    ap.add_argument("--film", default="", help="a folder for the per-episode film and look npz")
    ap.add_argument("--plan-socket", default=str(REPO / "outputs/bigym_20261124/plan.sock"))
    ap.add_argument("--plan-timeout", type=float, default=10.0)
    ap.add_argument("--stance", action="store_true", help="walk to a stance first and pull the drawer by walking back")
    ap.add_argument("--allow-eval-seeds", action="store_true", help="run seeds in the eval block 620000-629999")
    a = ap.parse_args()
    if ":" in a.seeds:
        lo, hi = map(int, a.seeds.split(":"))
        seeds = list(range(lo, hi))
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
    bad = [s for s in seeds if EVAL_LO <= s < EVAL_HI]
    if bad and not a.allow_eval_seeds:
        raise SystemExit(
            f"seeds {bad[:3]}... are in the eval block {EVAL_LO}-{EVAL_HI - 1}; --allow-eval-seeds to run them"
        )
    from bigym.loco import make
    from bigym.loco.eval import protocol_violations

    env = make(a.task)
    viol = protocol_violations(env)
    if viol:
        raise SystemExit(f"not the official config: {viol}")
    kin = Kin(env)
    m = env.inner_env.model
    keep = int(m.vis.quality.offsamples)
    m.vis.quality.offsamples = 0  # our renderer only: one geom per pixel at the edges
    renderer = mujoco.Renderer(m, H, W)
    m.vis.quality.offsamples = keep
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ok = 0
    for s in seeds:
        row = episode(env, kin, renderer, a.task, s, a.method, a, s % 10000)
        ok += row["success"]
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in ("task", "seed", "method", "success", "t_success", "fell", "plan_ok",
                                                  "fallback", "err_end_cm", "err_settled_cm", "wall_s")}), flush=True)  # fmt: skip
    print(json.dumps(dict(summary=True, task=a.task, method=a.method, n=len(seeds), success=ok)), flush=True)
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
