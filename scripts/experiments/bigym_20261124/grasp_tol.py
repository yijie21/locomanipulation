"""bigym_grasptol_20261125: how much hand-position error the grasp-and-lift step of BiGym 2.0's move_plate (MovePlateG1)
and flip_cup (FlipCupG1) tolerates. The registration is docs/handoff/notebook-20261120/bigym_grasptol_20261125.md.

This measures the TASK's tolerance, not a method: the nominal grasp is computed from the object's simulated pose, and the
hand is servoed onto the commanded point with the simulator's right_end_effector reading as feedback. Both are
instruments of this probe only. Nothing else of the simulator's state enters a decision; the object's pose is otherwise
read for the row's scoring and failure-analysis columns only.

The controller is BiGym's native one (bigym.loco.make with the task's official config; protocol_violations refused but
for camera_keys): GR00T-WBC on the legs and the waist, the arms' absolute joint targets straight into PD. The env is made
without cameras (no observation is read). The task's own episode end on held success and its time limit are switched
off on the instance (nothing in this sequence can meet either task's success), so an episode ends only when the
sequence does, or on a fall or a physics error.

An episode (task, seed, condition):
  1. reset(seed) (BiGym's official reset; resets are bit-identical by seed, so the conditions of one seed differ only
     in the offset; the row's prefix_hash, md5 of qpos/qvel after the stance, checks that).
  2. the ready pose (the resting hands sit at the counter / rack height and would hit them on the walk in): both arms'
     joint targets min-jerk over READY_S from the reset's to the ready joints, then READY_HOLD_S. Right arm: DLS IK
     (from FAMILY) for right_end_effector at READY_P (pelvis frame) with the attitude ready_R(TILT): the fingers down,
     turned TILT forward, the closing axis along the pelvis's -y. Left arm: LEFT_READY.
  3. the walk to the stance: WALK_ROUNDS rounds of [sideways, then ahead], each bigym_bench --stance's walk (WALK_VX on
     the pelvis's readings, skipped under WALK_MIN, WALK_SETTLE_S after it) with a heading hold added (wz = -YAW_K x the
     pelvis's yaw change since the ready pose, clipped to YAW_MAX), until the nominal grasp point (rule below, at the
     reset) sits STAND_X ahead of and STAND_Y to the left of the pelvis. The height command stays the hold's 0.74
     (HEIGHT None: no squat).
  4. the nominal grasp, from the object's pose after the stance (it has not moved: the row's walk_disp_mm says so):
     a point g and a frame R_g = [feed, close, third] (feed: the way the fingers point and go in; close: the fingers'
     sliding axis; third = feed x close). feed is "from above" turned TILT (45 deg) about close toward the robot's
     heading at the reset (fingertips down and forward; tilted() below): a straight-down approach puts the right arm in
     high-shoulder configurations where the native WBC pitches the waist to its 30 deg limit (DESIGN RECORD).
     The hand's attitude: R_g or its twin R_g diag(1, -1, -1) (the same fingers turned 180 deg about the feed), the one
     whose pregrasp and grasp IK solutions are within 1 cm, keep the right arm off the robot's own body
     (bigym_bench's self-contact check, 1 mm) and then move the arm least from its measured joints (bigym_bench's
     drawer rule); chosen on the nominal grasp, the same for every condition of the seed.
     The pinch point's coordinate along `close` is always the middle of the object's extent along `close` inside the
     pads' footprint (rays along +-close at 7 x 3 points over the pads' face: site x in PAD_X, z in +-PAD_Z), so both
     fingers meet the object together.
       plate  close: the plate's normal (to its concave side); "down": world -z made orthogonal to the normal (u its
              opposite, the plate's in-plane up); the rim's top point by rays along -u; g = PLATE_DEPTH below it along
              u, in the plate's mid-plane through its centre; feed = "down" tilted.
       cup    the mug's frame (BiGym's mug asset: z its axis from the bottom to the opening, x toward the handle); the
              mug starts upside down, so z points down. "down" = z, close = the mug's y (across the handle), feed = z
              tilted (toward the mug's axis: the handle faces the robot); g = the point (CUP_R, 0, CUP_DEPTH) of the
              mug's frame for the chosen grasp CUP_GRASP:
                body    CUP_R 0: across the body's diameter (92 mm; the Dex1's widest opening is 89.4 mm between the
                        pads, so the fingers cannot pass the body);
                edge    CUP_R 30 mm toward the handle: a chord of the wall near the upturned bottom's edge (70 mm);
                handle  CUP_R 68 mm: across the handle (15 mm thick);
              chosen on the design seeds by the registration's rule (DESIGN RECORD).
  5. the offset d (mm, grasp frame: one of feed/close/third, +-2/5/10/20, or 0): the commanded grasp g + R_g d and the
     pregrasp PRE_M back along feed from it (both moved together).
  6. the reach: the arm's joint targets min-jerk (bigym_bench's REACH_SPEED, at least REACH_MIN_S) to the IK solution of
     the hover point HOVER_M above the pregrasp (world z), a straight line down to the pregrasp (IK each step), the
     servo at the pregrasp, a straight line in along feed (IN_S, min-jerk, the pregrasp servo's correction kept) and
     the servo at the grasp. The servo: every control step the measured right_end_effector's error to the commanded
     point (world) is integrated into the IK's target point (SERVO_K a step, its norm capped at SERVO_CAP; the attitude
     target fixed: position feedback only), DLS IK on the 7 right-arm joints in BiGym's model warm-started from the
     last solution, until the error is under SERVO_TOL for SERVO_HOLD_N steps running, at most SERVO_MAX_S.
     Blocked (the registration's 预测修订 of 2026-11-25): a servo (at the pregrasp or the grasp) stops at once, the last
     joint targets kept, when the gripper (BLOCK_BODIES of GRIPPER) touches the target object while the error is
     SERVO_TOL or more; the episode goes on to the close and is scored as any other (blocked_by_object, filed under
     its commanded offset, in the tolerance curve; blocked at the pregrasp skips the line in and the grasp servo).
     Arrival: the achieved offset (measured right_end_effector minus g, grasp frame) and the error to the commanded
     point are recorded; a servo that was not blocked and ends over PLACE_FAIL is a placement failure (joint limits,
     the rack or counter, the 3 s limit: counted and reported apart, left out of the tolerance curve).
  7. close (gripper 1) and hold CLOSE_S with the arm's targets frozen; the hand up LIFT_M along world z over LIFT_S,
     a straight line from the held joint targets' own point (IK each step, no correction), so a blocked hand is not
     pushed on toward g; hold HOLD_S.
  8. scoring (the simulator's): success = the object's centre RISE_MIN above its centre at the reset and
     robot.is_gripper_holding_object(object, right) at the end.

Failure modes (reported, not scored; the rule frozen before the formal run):
  placement failed   not blocked and the servo's error at the grasp > PLACE_FAIL (whatever the lift did: apart);
  blocked by object  else, a failure whose servo was stopped by the gripper touching the object (description only:
                     scored as any other; blocked episodes that lifted count as successes);
  knocked away       else, a failure where the object moved >= KNOCK_MM or turned >= KNOCK_DEG between the nominal
                     grasp (after the stance) and the start of the close (the hand's approach moved it);
  closed on nothing  else, the pads hold nothing at the end of the close hold;
  slipped after grasp  else (held after the close, not risen / not held at the end).

Row columns: the condition, the achieved offset (mm, grasp frame) and the servo's errors and times, the object's
disturbance before and after the close, the rise, the hold, contacts of the right arm at the phases' ends, and a trace
every TRACE_EVERY control steps (time, phase, right_end_effector position/wxyz, object position/wxyz, finger joints,
pelvis position, waist yaw/roll/pitch deg, pelvis yaw deg) for the failure analysis.

    source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh; export OMP_NUM_THREADS=1; cd <repo>
    python scripts/experiments/bigym_20261124/grasp_tol.py run --task flip_cup --seeds 731000:731005 --conds 0 \
        --out /tmp/x.jsonl                                                    # design seeds only without --formal
    python scripts/experiments/bigym_20261124/grasp_tol.py run --formal --task move_plate --shard 3/15 --out <rows>.jsonl
    python scripts/experiments/bigym_20261124/grasp_tol.py analyze outputs/bigym_20261124/grasptol/rows/*.jsonl

DESIGN RECORD (design seeds 731000-731004 only; "k/5" = 0-offset successes with the servo's error <= 2 mm)
  Instruments. Both resting hands hit the counter / rack on the walk in -> the ready pose first. The servo's integrator
  wound up when the target sat at the arm's edge (elbow at its -1.05 limit, the WBC's waist sagging forward under
  the arm, the arm rising to make up for it, to the waist's 30 deg limit) -> its correction capped at 5 cm. A joint-
  space reach straight to the pregrasp swept the open fingers through the plate (731002, plate pushed 47 mm) -> the
  reach ends 6 cm above the pregrasp and comes straight down. The walk turned the pelvis up to -32 deg on a 27 cm
  sidestep -> sideways first, two rounds, and the heading hold (the pelvis's yaw at the stance -7.3 to +1.5 deg).
  The servo stopped after 5 steps under 1 mm with 0.55-0.7 mm left (the integrator still closing) -> 15 steps.
  Approach (both tasks). Straight down (TILT 0) is out of reach of a usable arm configuration: the right arm needs
  shoulder pitch about -2.3 already at the ready pose, where the WBC pitches the waist to its 30 deg limit (from the
  reach to the grasp the waist reached 30 deg in 13 of 15 plate and 7 of 10 cup episodes at TILT 0; TILT 45: 7.8-9.6
  deg). Plate TILT 0: 2/5 (STAND_X 0.32), 1/5 (0.36), 1/5 (0.40); cup handle TILT 0 (STAND_Y -0.18): 0/5 (0.37), 0/5
  (0.40). TILT 15: plate 4/5 (0.36), 2/5 (0.40), cup handle 3/5 (0.37). Tilting
  the fingertips toward the robot instead (TILT_DIR -1) has no IK solution at the ready pose for TILT 45 (0 of 24 ready
  points) and only joint-limit ones for TILT 30.
  Plate (STAND_Y -0.12, PLATE_DEPTH 15 mm, both set before any run). With an earlier ready point (0.20, -0.15, 0.24),
  24 mm out of reach at TILT 45: TILT 45 with STAND_X 0.32: 2/5 (3 placement failures: the elbow limit at the
  pregrasp, see above), 0.34: 4/5, 0.36: 5/5, 0.38: 5/5; TILT 30: 0.34 3/5, 0.36 4/5; TILT 60: 0/5 at 0.34 and 0.36
  (all placement failures). With READY_P (reachable, IK margin 0.40 rad): TILT 45, 0.36 5/5, 0.37 5/5, 0.38 5/5 ->
  STAND_X 0.37 (the middle of the 5/5 range). Per seed at the frozen rule: see the servo check.
  Cup (the registration's rule: the most 0-offset successes of 5). Each grasp on the same 8 stances / tilts (TILT 30, 45
  x STAND_X 0.34, 0.37 x STAND_Y -0.12, -0.18), the grasp's best and its total over the 8:
    body    best 0/5, total 0/40 (every episode a placement failure: the open fingers meet the mug and push it,
            14-944 mm before the close at TILT 45 / STAND_X 0.37 / STAND_Y -0.18);
    edge    best 4/5 (TILT 30, STAND_X 0.37, STAND_Y -0.12 or -0.18), total 20/40;
    handle  best 5/5 (TILT 45, STAND_X 0.37, STAND_Y -0.18), total 26/40
  -> handle, at TILT 45, STAND_X 0.37, STAND_Y -0.18 (no tie). Per seed at the frozen rule: see the servo check.
  Not tuned on successes: READY_P / FAMILY (picked by IK margin for the ready attitude), LEFT_READY, CUP_R /
  CUP_DEPTH (from the mug's geometry, before any run), KNOCK_MM / KNOCK_DEG 5 mm / 5 deg (before any run; the clean
  approaches move the object under 0.1 mm).
  Servo check at the frozen rule (design seeds, 0 and +-2/5 on every axis for the plate plus feed+10/20, third+10/20,
  -20; 0 and +-5/20 on every axis for the cup; 160 episodes): the prefix hash is one per seed across conditions;
  placed episodes |achieved - commanded offset| median 0.28 mm (plate) / 0.26 mm (cup), max 0.69 / 0.88 mm; under
  1 mm after median 0.66 s, max 1.70 / 2.32 s. 0 offset: 5/5 on both tasks. Placement failures (before the blocked
  rule): plate feed+5 2/5, feed+10 5/5, feed+20 5/5, third+20 5/5 (the gripper's base, right_dex1_base_link, meets the
  rim: the pinch point cannot go more than about 4 mm deeper than PLATE_DEPTH at TILT 45); none elsewhere.
  The blocked rule (预测修订 2026-11-25), checked on the design seeds before the formal run (plate 0, feed+5/10/20,
  third+20; cup 0): at 0 offset no part of the gripper touched the object during either servo on either task, so
  the fingers' brush does not trip it -> BLOCK_BODIES "gripper" (the base and both fingers); 0 offset 5/5 on both
  tasks. The 20 deep plate episodes were all blocked at the grasp servo's first step (the base touching the rim,
  reached on the line in, which had pushed the plate 1.0-8.9 mm), all 20 lifted; achieved feed +2.8 to +3.0 mm
  (feed+5), +4.5 to +4.7 (feed+10), +8.2 to +8.6 (feed+20); third+20: third +23.9 to +24.6, feed -2.5 to -2.6.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bigym_bench as B  # noqa: E402

TASKS = ("move_plate", "flip_cup")
OBJ_BODY = {"move_plate": "plate/plate", "flip_cup": "mug/mug"}
DESIGN = (731000, 731005)  # the design seeds (hi exclusive)
FORMAL = (730000, 730060)  # 0 offset on all 60
PAIRED = (730000, 730030)  # every non-zero offset
AXES = ("feed", "close", "third")
MAGS_MM = (2, 5, 10, 20)
CONDS = ["0"] + [f"{ax}{s}{mm}" for ax in AXES for s in "+-" for mm in MAGS_MM]  # 25

# ------------------------------------------------------------------------------------- the frozen design (see the end)
STAND_X = {"move_plate": 0.37, "flip_cup": 0.37}  # m ahead of the pelvis: the nominal grasp point at the stance
STAND_Y = {"move_plate": -0.12, "flip_cup": -0.18}  # m to its left (in front of the right shoulder)
HEIGHT = {"move_plate": None, "flip_cup": None}  # the base height command after the walk (None: the hold's 0.74)
HEIGHT_S = 2.0
WALK_ROUNDS = 2
READY_P = {"move_plate": (0.26, -0.18, 0.28), "flip_cup": (0.26, -0.18, 0.28)}  # right_end_effector, pelvis frame
FAMILY = (-1.0, -1.2, 1.0, -0.6, -0.8, -0.3, 1.3)  # the IK's start for the ready pose: the grasps' arm family
TILT = {"move_plate": 45.0, "flip_cup": 45.0}  # deg: the feed turned from straight down toward forward
TILT_DIR = {"move_plate": 1, "flip_cup": 1}  # +1: the fingertips turned away from the robot, -1: toward it
LEFT_READY = (0.3, 0.2, 0.0, -1.0, 0.0, 0.0, 0.0)  # left arm joint targets: forearm up, hand above the counter
READY_S, READY_HOLD_S = 2.0, 0.5
YAW_K, YAW_MAX = 1.5, 0.3  # the walk's heading hold: wz = -YAW_K * (yaw - yaw after the ready pose), rad/s, clipped
PLATE_DEPTH = 0.015  # m below the rim's top, along the plate's in-plane up
CUP_GRASP = "handle"
CUP_R = {"body": 0.0, "edge": 0.030, "handle": 0.068}  # m from the mug's axis toward the handle
CUP_DEPTH = {"body": 0.030, "edge": 0.030, "handle": 0.030}  # m below the upturned bottom (the mug frame's z)
# the instruments (the registration's numbers where it gives them)
PRE_M, IN_S = 0.06, 1.0
HOVER_M = 0.06  # the reach ends this far above the pregrasp (world z), then a straight line down to it
SERVO_TOL, SERVO_HOLD_N, SERVO_MAX_S, SERVO_K, SERVO_CAP = 0.001, 15, 3.0, 0.1, 0.05
PLACE_FAIL = 0.002
# the 预测修订 of 2026-11-25: a servo whose gripper touches the object before it is within SERVO_TOL stops there
GRIPPER = ("right_dex1_base_link", "right_dex1_finger_link_1", "right_dex1_finger_link_2")
BLOCK_BODIES = "gripper"  # "gripper": any of GRIPPER; "base": right_dex1_base_link only (if the fingers' brush of a
#                           0-offset approach would trip it)
CLOSE_S, LIFT_M, LIFT_S, HOLD_S, RISE_MIN = 0.5, 0.10, 1.0, 1.0, 0.05
KNOCK_MM, KNOCK_DEG = 5.0, 5.0
PAD_X, PAD_Z = (-0.029, 0.037), 0.013  # the Dex1 pads' face in the right_end_effector frame (pad geoms, measured)
TRACE_EVERY = 10
DT = B.DT
LEFT = [n.replace("right_", "left_") for n in B.ARM]


def ready_R(tilt_deg: float) -> np.ndarray:
    """The ready attitude (pelvis frame): fingers down turned tilt_deg forward, the closing axis along -y (its twin,
    along +y, is out of the right arm's reach over these objects)."""
    t = math.radians(tilt_deg)
    x = np.array([math.sin(t), 0.0, -math.cos(t)])
    y = np.array([0.0, -1.0, 0.0])
    return np.column_stack([x, y, np.cross(x, y)])

DESIGN_KEYS = ("STAND_X", "STAND_Y", "WALK_ROUNDS", "HEIGHT", "HEIGHT_S", "READY_P", "FAMILY", "TILT", "TILT_DIR", "HOVER_M", "LEFT_READY", "YAW_K", "PLATE_DEPTH", "CUP_GRASP", "CUP_R",
               "CUP_DEPTH", "SERVO_K", "BLOCK_BODIES")  # fmt: skip


def offset_of(cond: str) -> np.ndarray:
    d = np.zeros(3)
    if cond != "0":
        ax = next(a for a in AXES if cond.startswith(a))
        d[AXES.index(ax)] = float(cond[len(ax) :]) / 1000.0
    return d


def jobs_formal(task: str) -> list:
    out = [(task, s, "0") for s in range(*FORMAL)]
    out += [(task, s, c) for s in range(*PAIRED) for c in CONDS[1:]]
    return out  # 60 + 24 x 30 = 780


# ------------------------------------------------------------------------------------------------- the object's truth
class Obj:
    """The object's body, its collision geoms and rays against them only (the nominal grasp's instrument)."""

    def __init__(self, env, task: str):
        inner = env.inner_env
        self.m, self.d = inner.model, inner.data
        self.b = self.m.body(OBJ_BODY[task]).id
        self.geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.b
                      and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]  # fmt: skip
        self.prop = inner.plates[0] if task == "move_plate" else inner.cup

    def pose(self):
        return self.d.xpos[self.b].copy(), self.d.xmat[self.b].reshape(3, 3).copy()

    def ray(self, p, v) -> float:
        """Distance along the unit vector v from p to the object's first surface, -1 if none."""
        best = -1.0
        p, v = np.asarray(p, float), np.asarray(v, float)
        for g in self.geoms:
            t = mujoco.mj_rayMesh(self.m, self.d, g, p, v)
            if t >= 0 and (best < 0 or t < best):
                best = t
        return best


def centre_on_close(obj: Obj, p, feed, close, third):
    """p moved along close to the middle of the object's extent along close inside the pads' footprint."""
    hi, lo = [], []
    for a in np.linspace(PAD_X[0], PAD_X[1], 7):
        for b in (-PAD_Z, 0.0, PAD_Z):
            q = p + a * feed + b * third
            t = obj.ray(q + 0.1 * close, -close)
            if t >= 0:
                hi.append(0.1 - t)
            t = obj.ray(q - 0.1 * close, close)
            if t >= 0:
                lo.append(t - 0.1)
    if not hi or not lo:
        return p, dict(width_mm=None)
    top, bot = max(hi), min(lo)
    return p + 0.5 * (top + bot) * close, dict(width_mm=round(1000 * (top - bot), 2), shift_mm=round(500 * (top + bot), 2))


def tilted(down: np.ndarray, close: np.ndarray, heading: np.ndarray, tilt_deg: float, sign: int) -> np.ndarray:
    """The feed: `down` (orthogonal to close) turned about close by tilt_deg toward the horizontal direction orthogonal
    to close that points away from the robot (positive along its heading; sign -1: toward it)."""
    w = np.cross(close, np.array([0.0, 0.0, 1.0]))
    w /= np.linalg.norm(w)
    if w @ heading < 0:
        w = -w
    w = w * sign
    w -= (w @ down) * down
    w /= np.linalg.norm(w)
    t = math.radians(tilt_deg)
    return math.cos(t) * down + math.sin(t) * w


def nominal_grasp(obj: Obj, task: str, heading: np.ndarray) -> dict:
    """The nominal grasp point g (world) and frame R_g = [feed, close, third] from the object's simulated pose."""
    c, R = obj.pose()
    if task == "move_plate":
        n = R[:, 2]  # the plate's normal, to its concave side (BiGym's plate asset: the dish along +z)
        u = np.array([0.0, 0.0, 1.0]) - n[2] * n
        u /= np.linalg.norm(u)
        close = n
        feed = tilted(-u, close, heading, TILT[task], TILT_DIR[task])
        third = np.cross(feed, close)
        tops = []
        for s in np.linspace(0.0, 0.0192, 9):  # across the plate's depth (0 to 19.2 mm along its normal)
            t = obj.ray(c + 0.25 * u + s * n, -u)
            if t >= 0:
                tops.append(0.25 - t)
        if not tops:
            return dict(ok=False, why="no rim top")
        top = max(tops)
        p = c + (top - PLATE_DEPTH) * u + 0.0096 * n
        info = dict(rim_top_mm=round(1000 * top, 2))
    else:
        close = R[:, 1]
        feed = tilted(R[:, 2], close, heading, TILT[task], TILT_DIR[task])
        third = np.cross(feed, close)
        p = c + R @ np.array([CUP_R[CUP_GRASP], 0.0, CUP_DEPTH[CUP_GRASP]])
        info = dict(grasp=CUP_GRASP, handle_deg=round(math.degrees(math.atan2(-R[1, 0], -R[0, 0])), 1))
    g, ci = centre_on_close(obj, p, feed, close, third)
    info.update(ci)
    return dict(ok=True, g=g, R=np.column_stack([feed, close, third]), **info)


# --------------------------------------------------------------------------------------------------------- an episode
class Episode:
    def __init__(self, env, kin, obj: Obj, task: str):
        self.env, self.kin, self.obj, self.task = env, kin, obj, task
        self.d = env.inner_env.data
        m, ns = env.inner_env.model, env.inner_env.robot.namespace
        self.grip_b = {m.body(ns + n).id: n for n in GRIPPER}
        self.obj_g = set(obj.geoms)
        self.trace = []
        self.render_dir, self.renderer, self.tag = "", None, ""

    def shot(self, name: str, at) -> None:
        """Design only (--render): two free-camera views of the grasp point."""
        if not self.render_dir:
            return
        from PIL import Image

        m = self.env.inner_env.model
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, 360, 480)
        imgs = []
        for az, el in ((180.0, -25.0), (90.0, -15.0), (180.0, -80.0)):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = at
            cam.distance, cam.azimuth, cam.elevation = 0.45, az, el
            self.renderer.update_scene(self.d, cam)
            imgs.append(self.renderer.render().copy())
        os.makedirs(self.render_dir, exist_ok=True)
        Image.fromarray(np.concatenate(imgs, axis=1)).save(f"{self.render_dir}/{self.tag}_{name}.png")

    # -- control --------------------------------------------------------------------------------------------------
    def step(self, phase, ref=None, grip=None, vx=0.0, vy=0.0) -> bool:
        return self.run.step(phase, ref, grip, vx, vy)  # traced (go() wraps Run.step, the walk's steps included)

    def snap(self, phase) -> None:
        d, kin = self.d, self.kin
        op, oR = self.obj.pose()
        fq = [float(d.qpos[a]) for a in self.finger_q]
        self.trace.append([round(self.run.k * DT, 2), phase, B.r3(d.site_xpos[kin.site]),
                           B.r3(B.quat_of(d.site_xmat[kin.site].reshape(3, 3))), B.r3(op), B.r3(B.quat_of(oR)),
                           B.r3(fq, 5), B.r3(d.xpos[kin.pelvis]), B.r3(np.degrees(d.qpos[kin.waist_q]), 1),
                           round(math.degrees(self.yaw()), 1)])  # fmt: skip

    def site_w(self) -> np.ndarray:
        return self.d.site_xpos[self.kin.site].copy()

    def site_R(self) -> np.ndarray:
        return self.d.site_xmat[self.kin.site].reshape(3, 3).copy()

    def ik_w(self, p_w, R_w, q_seed, iters=20):
        """DLS IK (bigym_bench.Kin.ik, full attitude weight) of a world target in the current pelvis frame."""
        kin = self.kin
        kin.load(self.d)
        pp, Rp = self.run.pelvis()
        q, ep, er = kin.ik(Rp.T @ (p_w - pp), Rp.T @ R_w, q_seed, 1.0, iters=iters)
        return q, ep, er

    def touching(self) -> list:
        """The gripper's bodies (GRIPPER) in contact with the target object now."""
        m, d, out = self.env.inner_env.model, self.d, set()
        for c in d.contact[: d.ncon]:
            for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                b = m.geom_bodyid[ga]
                if b in self.grip_b and gb in self.obj_g:
                    out.add(self.grip_b[b])
        return sorted(out)

    def servo(self, phase, p_w, R_w, corr, q, grip=0.0):
        """Integrate the measured right_end_effector's error into the IK's target point until it is within SERVO_TOL
        for SERVO_HOLD_N steps running, at most SERVO_MAX_S; stop at once (the last joint targets kept) when the gripper
        (BLOCK_BODIES) touches the object while the error is SERVO_TOL or more (blocked)."""
        k0, n_ok, blocked, first_touch = self.run.k, 0, False, None
        trip = set(GRIPPER) if BLOCK_BODIES == "gripper" else {"right_dex1_base_link"}
        for _ in range(int(round(SERVO_MAX_S / DT))):
            e = p_w - self.site_w()
            err = float(np.linalg.norm(e))
            touch = self.touching()
            if touch and first_touch is None:
                first_touch = dict(t_s=round((self.run.k - k0) * DT, 2), err_mm=round(1000 * err, 3), bodies=touch)
            if err >= SERVO_TOL and trip & set(touch):
                blocked = True
                break
            n_ok = n_ok + 1 if err < SERVO_TOL else 0
            if n_ok >= SERVO_HOLD_N:
                break
            corr = corr + SERVO_K * e
            n = float(np.linalg.norm(corr))
            if n > SERVO_CAP:  # anti-windup: the correction never exceeds SERVO_CAP
                corr *= SERVO_CAP / n
            q, _, _ = self.ik_w(p_w + corr, R_w, q)
            if self.step(phase, q, grip):
                break
        e = p_w - self.site_w()
        return corr, q, dict(err_mm=round(1000 * float(np.linalg.norm(e)), 3), t_s=round((self.run.k - k0) * DT, 2),
                             arrived=n_ok >= SERVO_HOLD_N, blocked=blocked, first_touch=first_touch,
                             corr_mm=B.r3(1000 * corr, 2), deg=round(B.angle_deg(self.site_R(), R_w), 2))  # fmt: skip

    def line_w(self, phase, p0, p1, R_w, corr, q, dur, grip=0.0):
        """A straight line p0 -> p1 (world, min-jerk over dur), the correction corr held, IK each step."""
        N = int(math.ceil(dur / DT - 1e-9))
        worst = 0.0
        for k in range(N):
            u = B.minjerk((k + 1) / N)
            q, ep, _ = self.ik_w(p0 + u * (p1 - p0) + corr, R_w, q)
            worst = max(worst, ep)
            if self.step(phase, q, grip):
                break
        return q, round(1000 * worst, 2)

    # -- the episode ----------------------------------------------------------------------------------------------
    def go(self, seed: int, cond: str) -> dict:
        from bigym.const import HandSide
        from bigym.loco.eval import protocol_violations  # noqa: F401  (checked at make)

        env, kin, obj, task = self.env, self.kin, self.obj, self.task
        inner, d, m = env.inner_env, self.d, env.inner_env.model
        t_wall = time.time()
        env.reset(seed=seed)
        inner._success_hold_steps = 10**9  # no episode end on success (the instance only)
        env.bigym._time_limit = 10**9  # nor on the time limit
        ns = inner.robot.namespace
        self.finger_q = [m.jnt_qposadr[m.joint(ns + f"right_dex1_finger_joint_{i}").id] for i in (1, 2)]
        self.run = run = B.Run(env, kin, task)
        self.trace = []
        run_step = run.step

        def traced(phase, *args, **kw):
            done = run_step(phase, *args, **kw)
            if run.k % TRACE_EVERY == 0:
                self.snap(phase)
            return done

        run.step = traced
        lay = env.wholebody_action_layout()
        lslots = np.array([lay["base_dim"] + list(lay["limb_names"]).index(n) for n in LEFT])
        row = dict(task=task, seed=seed, cond=cond, offset_cmd_mm=B.r3(1000 * offset_of(cond), 1), bigym_commit="35ec409")
        o0p, o0R = obj.pose()
        row["obj0"] = dict(p=B.r3(o0p), q=B.r3(B.quat_of(o0R)))

        # 2. the ready pose: both arms' joint targets min-jerk to their ready joints over READY_S
        kin.load(d)
        pp, Rp = run.pelvis()
        self.heading = Rp[:, 0].copy()
        q_ready, e_r, _ = kin.ik(np.array(READY_P[task]), ready_R(TILT[task]), np.array(FAMILY), 1.0, iters=400, tol_p=1e-4)
        a0, l0, l1 = run.ref.copy(), run.hold[lslots].copy(), np.array(LEFT_READY)
        N = int(round(READY_S / DT))
        for k in range(N):
            u = B.minjerk((k + 1) / N)
            run.hold[lslots] = l0 + u * (l1 - l0)
            self.step("ready", a0 + u * (q_ready - a0), 0.0)
        for _ in range(int(round(READY_HOLD_S / DT))):
            self.step("ready_hold")
        self.yaw_ref = self.yaw()
        row["ready"] = dict(q=B.r3(q_ready, 3), ik_mm=round(1000 * e_r, 2), self_mm=round(kin.self_contact(q_ready), 1),
                            end_mm=round(1000 * float(np.linalg.norm(run.hand() - np.array(READY_P[task]))), 1))  # fmt: skip

        # 3. the stance: the nominal grasp point STAND_X ahead, STAND_Y left of the pelvis
        ng0 = nominal_grasp(obj, task, self.heading)
        if not ng0["ok"]:
            return dict(row, error=ng0["why"])
        walks = []
        for rnd in range(WALK_ROUNDS):  # sideways, then ahead; a second round corrects what the first left over
            for axis, stand in ((1, STAND_Y[task]), (0, STAND_X[task])):
                pp, Rp = run.pelvis()
                err = float((Rp.T @ (ng0["g"] - pp))[axis]) - stand
                if abs(err) <= B.WALK_MIN:
                    continue
                moved = self.walk(f"walk{rnd}{'yx'[1 - axis]}", abs(err), math.copysign(B.WALK_VX, err),
                                  abs(err) / B.WALK_VX + 2.0, axis)  # fmt: skip
                self.wait(f"settle{rnd}{'yx'[1 - axis]}", B.WALK_SETTLE_S)
                walks.append(["yx"[1 - axis], round(100 * err, 1), round(100 * math.copysign(moved, err), 1)])
        if HEIGHT[task] is not None:
            run.hold[2] = HEIGHT[task]
            self.wait("height", HEIGHT_S)
        pp, Rp = run.pelvis()
        g_p = Rp.T @ (ng0["g"] - pp)
        yaw = math.degrees(math.atan2(Rp[1, 0], Rp[0, 0]))
        row["walk"] = dict(walks=walks, grasp_pelvis=B.r3(g_p), pelvis=B.r3(pp), yaw_deg=round(yaw, 2),
                           height_cmd=float(run.hold[2]), body_contacts=self.body_contacts())  # fmt: skip
        h = hashlib.md5()
        h.update(np.round(d.qpos, 9).tobytes())
        h.update(np.round(d.qvel, 6).tobytes())
        row["prefix_hash"] = h.hexdigest()[:12]
        row["prefix_steps"] = run.k

        # 4. the nominal grasp after the stance
        sp, sR = obj.pose()
        row["walk_disp_mm"] = round(1000 * float(np.linalg.norm(sp - o0p)), 2)
        ng = nominal_grasp(obj, task, self.heading)
        if not ng["ok"]:
            return dict(row, error=ng["why"])
        g, Rg = ng["g"], ng["R"]
        feed = Rg[:, 0]
        row["nominal"] = {k: (B.r3(v) if isinstance(v, np.ndarray) else v) for k, v in ng.items() if k != "ok"}
        kin.load(d)
        q_meas = d.qpos[kin.arm_q].copy()
        pp, Rp = run.pelvis()
        score, sols, sols_q = [], [], []
        for Rh in (Rg, Rg @ np.diag([1.0, -1.0, -1.0])):
            qp, ep, _ = kin.ik(Rp.T @ (g - PRE_M * feed - pp), Rp.T @ Rh, q_meas, 1.0, iters=300, tol_p=1e-4)
            qg, eg, _ = kin.ik(Rp.T @ (g - pp), Rp.T @ Rh, qp, 1.0, iters=300, tol_p=1e-4)
            pen = max(kin.self_contact(qp), kin.self_contact(qg))
            score.append((max(ep, eg) >= 0.01, round(pen) if pen > 1.0 else 0, float(np.abs(qp - q_meas).sum()),
                          round(1000 * max(ep, eg), 1), round(pen, 1)))  # fmt: skip
            sols.append(Rh)
            sols_q.append(qp)
        took = int(score[1][:3] < score[0][:3])
        R_h = sols[took]
        row["twin"] = dict(took=took, ik_mm=[score[0][3], score[1][3]], self_mm=[score[0][4], score[1][4]],
                           joint_sum_deg=[round(math.degrees(s[2]), 1) for s in score])  # fmt: skip

        # 5. the offset
        g_cmd = g + Rg @ offset_of(cond)
        pre_cmd = g_cmd - PRE_M * feed

        # 6. the reach (the arm's joint targets min-jerk to the pregrasp's IK solution), the servo at the pregrasp, the
        # line in, the servo at the grasp
        kin.load(d)
        pp, Rp = run.pelvis()
        hover = pre_cmd + np.array([0.0, 0.0, HOVER_M])
        q_hov, e_hov, _ = kin.ik(Rp.T @ (hover - pp), Rp.T @ R_h, sols_q[took], 1.0, iters=300, tol_p=1e-4)
        a0 = run.ref.copy()
        L = float(np.linalg.norm(kin.site_pose(q_hov)[0] - kin.site_pose(a0)[0]))
        N = int(math.ceil(max(B.REACH_MIN_S, L / B.REACH_SPEED) / DT))
        for k in range(N):
            self.step("reach", a0 + B.minjerk((k + 1) / N) * (q_hov - a0), 0.0)
        corr, q = np.zeros(3), run.ref.copy()
        q, w_desc = self.line_w("descend", hover, pre_cmd, R_h, corr, q, max(B.REACH_MIN_S, HOVER_M / B.REACH_SPEED))
        row["reach"] = dict(path_cm=round(100 * L, 2), dur_s=round(N * DT, 2), ik_mm=round(1000 * e_hov, 2),
                            descend_ik_worst_mm=w_desc)
        corr, q, row["servo_pre"] = self.servo("servo_pre", pre_cmd, R_h, corr, q)
        if row["servo_pre"]["blocked"]:  # stopped by the object already at the pregrasp: no line in, no grasp servo
            e = g_cmd - self.site_w()
            sv = dict(row["servo_pre"], err_mm=round(1000 * float(np.linalg.norm(e)), 3), t_s=0.0, arrived=False)
            row["in_ik_worst_mm"] = None
        else:
            q, row["in_ik_worst_mm"] = self.line_w("in", pre_cmd, g_cmd, R_h, corr, q, IN_S)
            corr, q, sv = self.servo("servo", g_cmd, R_h, corr, q)
        row["servo"] = sv
        row["blocked_by_object"] = bool(sv["blocked"])
        row["blocked_phase"] = ("servo_pre" if row["servo_pre"]["blocked"] else "servo") if sv["blocked"] else None
        ach = Rg.T @ (self.site_w() - g)
        row["offset_ach_mm"] = B.r3(1000 * ach, 3)
        row["placement_failed"] = (not sv["blocked"]) and sv["err_mm"] > 1000 * PLACE_FAIL
        row["contacts_arrival"] = self.arm_pairs()
        self.tag = f"{task}_{seed}_{cond}"
        self.shot("1arrival", g)
        bp, bR = obj.pose()
        row["pre_close_disp"] = dict(mm=round(1000 * float(np.linalg.norm(bp - sp)), 2), deg=round(B.angle_deg(sR, bR), 2))

        # 7. close, hold; lift; hold
        for _ in range(int(round(CLOSE_S / DT))):
            self.step("close", q, 1.0)
        cp, cR = obj.pose()
        row["after_close"] = dict(mm=round(1000 * float(np.linalg.norm(cp - sp)), 2), deg=round(B.angle_deg(sR, cR), 2),
                                  holding=bool(inner.robot.is_gripper_holding_object(obj.prop, HandSide.RIGHT)),
                                  fingers_mm=[round(1000 * float(d.qpos[a]), 2) for a in self.finger_q],
                                  contacts=self.arm_pairs())  # fmt: skip
        self.shot("2closed", g)
        # the lift starts at the held joint targets' own point (forward kinematics now), so its first target is the
        # held command whether the servo arrived or was blocked short of g_cmd
        kin.load(d)
        pp, Rp = run.pelvis()
        p_held = pp + Rp @ kin.site_pose(q)[0]
        q, row["lift_ik_worst_mm"] = self.line_w("lift", p_held, p_held + np.array([0.0, 0.0, LIFT_M]), R_h, np.zeros(3), q,
                                                 LIFT_S, 1.0)  # fmt: skip
        for _ in range(int(round(HOLD_S / DT))):
            self.step("hold", q, 1.0)
        self.snap("end")
        self.shot("3end", g + np.array([0.0, 0.0, LIFT_M]))

        # 8. scoring
        ep_, eR = obj.pose()
        holding = bool(inner.robot.is_gripper_holding_object(obj.prop, HandSide.RIGHT))
        rise = float(ep_[2] - o0p[2])
        row.update(rise_cm=round(100 * rise, 2), holding=holding, success=bool(rise >= RISE_MIN and holding),
                   end_disp_mm=round(1000 * float(np.linalg.norm(ep_ - o0p)), 1), end_deg=round(B.angle_deg(o0R, eR), 1),
                   contacts_end=self.arm_pairs(), fell=bool(env.episode_fell()), termination=env.episode_termination(),
                   steps=run.k, budget_steps=1700 if task == "move_plate" else 1850)  # fmt: skip
        row["mode"] = failure_mode(row)
        row["trace"] = self.trace
        row["wall_s"] = round(time.time() - t_wall, 2)
        return row

    def yaw(self) -> float:
        R = self.run.pelvis()[1]
        return math.atan2(R[1, 0], R[0, 0])

    def walk(self, phase: str, dist: float, vx: float, cap_s: float, axis: int = 0) -> float:
        """bigym_bench.Run.walk (the base's speed vx along the pelvis's heading axis at the start, closed on its
        floating-base readings, WALK_LEAD short, at most cap_s; the arm holds its targets) with the heading held at
        the reset's: wz = -YAW_K * (yaw - yaw_ref) (GR00T-WBC's Walk policy otherwise turns the robot up to 30 deg on a
        25 cm sideways walk). -> the distance moved."""
        run = self.run
        p0, R0 = run.pelvis()
        h = np.array([R0[0, axis], R0[1, axis], 0.0])
        h /= np.linalg.norm(h)
        sign = 1.0 if vx > 0 else -1.0
        for _ in range(int(round(cap_s / DT))):
            if float(sign * (run.pelvis()[0] - p0) @ h) >= dist - B.WALK_LEAD:
                break
            err = (self.yaw() - self.yaw_ref + math.pi) % (2 * math.pi) - math.pi
            run.hold[3] = float(np.clip(-YAW_K * err, -YAW_MAX, YAW_MAX))
            if run.step(phase, None, None, *((vx, 0.0) if axis == 0 else (0.0, vx))):
                break
        run.hold[3] = 0.0
        return float(sign * (run.pelvis()[0] - p0) @ h)

    def arm_pairs(self) -> list:
        """The right arm's and gripper's contacts with any other body: [arm body, other body, deepest mm]."""
        m, d, kin, out = self.env.inner_env.model, self.d, self.kin, {}
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in kin.right) == (b2 in kin.right):
                continue
            mine, other = (b1, b2) if b1 in kin.right else (b2, b1)
            k = (m.body(mine).name.split("/")[-1], m.body(other).name.split("/")[-1])
            out[k] = max(out.get(k, 0.0), round(-1000 * float(c.dist), 1))
        return [[a, b, v] for (a, b), v in sorted(out.items())]

    def body_contacts(self) -> list:
        """The robot's contacts with the scene (not the floor) but the right arm's: [robot body, scene body, mm]."""
        m, d, kin, out = self.env.inner_env.model, self.d, self.kin, {}
        for c in d.contact[: d.ncon]:
            g1, g2 = c.geom1, c.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            r1, r2 = kin.robot_bodies[b1], kin.robot_bodies[b2]
            if r1 == r2:
                continue
            mine, other, og = (b1, b2, g2) if r1 else (b2, b1, g1)
            if m.geom_type[og] == mujoco.mjtGeom.mjGEOM_PLANE or mine in kin.right:
                continue
            k = (m.body(mine).name.split("/")[-1], m.body(other).name)
            out[k] = max(out.get(k, 0.0), round(-1000 * float(c.dist), 1))
        return [[a, b, v] for (a, b), v in sorted(out.items())]

    def wait(self, phase, seconds, grip=None) -> None:
        for _ in range(int(round(seconds / DT))):
            if self.step(phase, None, grip):
                return


def failure_mode(row: dict) -> str:
    """placement_failed first: those episodes are counted apart whatever the lift did (the registration). A blocked
    episode is scored as any other; blocked_by_object describes a blocked one that failed (the 预测修订)."""
    if row.get("placement_failed"):
        return "placement_failed"
    if row.get("success"):
        return "success"
    if row.get("blocked_by_object"):
        return "blocked_by_object"
    pc = row["pre_close_disp"]
    if pc["mm"] >= KNOCK_MM or pc["deg"] >= KNOCK_DEG:
        return "knocked_away"
    if not row["after_close"]["holding"]:
        return "closed_on_nothing"
    return "slipped_after_grasp"


# ------------------------------------------------------------------------------------------------------------- run
def make_env(task: str):
    from bigym.loco import make
    from bigym.loco.eval import protocol_violations

    env = make(task, camera_keys=())  # no observation is read: the only departure allowed from the official config
    viol = [v for v in protocol_violations(env) if not v.startswith("camera_keys:")]
    if viol:
        raise SystemExit(f"not the official config: {viol}")
    return env


def cmd_run(a) -> None:
    for kv in a.set or []:  # design trials only
        k, v = kv.split("=", 1)
        if k not in DESIGN_KEYS:
            raise SystemExit(f"--set {k}: not a design key")
        cur, val = globals()[k], json.loads(v)
        if isinstance(cur, dict) and not isinstance(val, dict):
            val = {**cur, a.task: val} if k not in ("CUP_R", "CUP_DEPTH") else {**cur, CUP_GRASP: val}
        elif isinstance(cur, dict):
            val = {**cur, **val}
        globals()[k] = val
    if a.formal:
        if a.set:
            raise SystemExit("--formal takes no --set")
        jobs = jobs_formal(a.task)
        k, n = (int(x) for x in a.shard.split("/"))
        jobs = jobs[k::n]
    else:
        lo, hi = (int(x) for x in a.seeds.split(":"))
        if not (DESIGN[0] <= lo and hi <= DESIGN[1]):
            raise SystemExit(f"seeds {lo}:{hi} outside the design block {DESIGN}; the formal seeds need --formal")
        conds = CONDS if a.conds == "all" else a.conds.split(",")
        jobs = [(a.task, s, c) for s in range(lo, hi) for c in conds]
    env = make_env(a.task)
    kin = B.Kin(env)
    obj = Obj(env, a.task)
    ep = Episode(env, kin, obj, a.task)
    ep.render_dir = a.render
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):  # resume
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["task"], r["seed"], r["cond"]))
    for job in jobs:
        if job in done:
            continue
        t, s, c = job
        try:
            row = ep.go(s, c)
        except Exception as e:  # a row says so; the shard goes on
            import traceback

            row = dict(task=t, seed=s, cond=c, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in ("task", "seed", "cond", "success", "mode", "rise_cm", "holding",
                                                  "offset_ach_mm", "error", "wall_s")}), flush=True)  # fmt: skip
    print(json.dumps(dict(done=True, task=a.task, n=len(jobs))), flush=True)
    env.close()


# -------------------------------------------------------------------------------------------------------- analysis
def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)


def cmd_analyze(a) -> None:
    rows = []
    for pat in a.rows:
        for fn in sorted(glob.glob(pat)):
            rows += [json.loads(x) for x in open(fn)]
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"rows {len(rows)}, error rows {len(errs)}")
    for r in errs[:5]:
        print("  error:", r["task"], r["seed"], r["cond"], r["error"])
    for task in TASKS:
        R = {(r["seed"], r["cond"]): r for r in rows if r["task"] == task}
        if not R:
            continue
        print(f"\n=== {task} ===")
        hashes = {}
        for (s, c), r in R.items():
            hashes.setdefault(s, set()).add(r.get("prefix_hash"))
        bad = [s for s, h in hashes.items() if len(h) > 1]
        print(f"seeds {len(hashes)}; seeds whose prefix (reset..stance) differs between conditions: {len(bad)} {bad[:5]}")
        print(f"{'cond':>9} {'n':>3} {'place':>5} {'succ':>4} {'rate':>5} {'wilson95':>13} {'med_ach_mm':>10}"
              f" {'med_|other|':>11} {'med_servo_mm':>12} {'med_t_s':>7} {'blk':>3} {'blk_med_ach':>11}  modes")  # fmt: skip
        for c in CONDS:
            rs = [r for (s, cc), r in sorted(R.items()) if cc == c]
            if not rs:
                continue
            ok = [r for r in rs if not r["placement_failed"]]
            k = sum(r["success"] for r in ok)
            lo, hi = wilson(k, len(ok))
            ax = next((i for i, x in enumerate(AXES) if c.startswith(x)), None)
            ach = np.array([r["offset_ach_mm"] for r in rs])
            med = float(np.median(ach[:, ax])) if ax is not None else float(np.median(np.linalg.norm(ach, axis=1)))
            oth = [i for i in range(3) if i != ax] if ax is not None else []
            medo = float(np.median(np.abs(ach[:, oth]).max(axis=1))) if oth else float("nan")
            sv = np.median([r["servo"]["err_mm"] for r in rs])
            ts = np.median([r["servo"]["t_s"] for r in rs])
            modes = {}
            for r in rs:
                modes[r["mode"]] = modes.get(r["mode"], 0) + 1
            flag = " FLAG>10%place" if (len(rs) - len(ok)) > 0.1 * len(rs) else ""
            mtxt = " ".join(f"{k_}:{v}" for k_, v in sorted(modes.items()) if k_ != "success")
            medtxt = f"{med:+10.2f}" if ax is not None else f"|{med:8.2f}|"
            blk = [r for r in rs if r.get("blocked_by_object")]
            if blk:
                ab = np.array([r["offset_ach_mm"] for r in blk])
                bm = float(np.median(ab[:, ax])) if ax is not None else float(np.median(np.linalg.norm(ab, axis=1)))
                blktxt = f"{len(blk):>3} {bm:+11.2f}" if ax is not None else f"{len(blk):>3} |{bm:9.2f}|"
            else:
                blktxt = f"{0:>3} {'-':>11}"
            rate = f"{k / len(ok):5.2f} [{lo:4.2f},{hi:4.2f}]" if ok else f"{'n/a':>5} {'':13}"
            print(f"{c:>9} {len(rs):>3} {len(rs) - len(ok):>5} {k:>4} {rate}"
                  f" {medtxt} {medo:11.2f} {sv:12.3f} {ts:7.2f} {blktxt}  {mtxt}{flag}")  # fmt: skip
        print(f"\npaired on seeds {PAIRED[0]}-{PAIRED[1] - 1} (pairs with a placement failure on either side dropped;"
              " blocked episodes kept):")
        print(f"{'axis':>6} {'sign':>4} {'mm':>3} {'pairs':>5} {'rate0':>5} {'rate':>5} {'drop_pp':>7} {'b':>3} {'c':>3}"
              f" {'p_mcnemar':>9}  meets")  # fmt: skip
        tol, thin = {}, {}
        for ax in AXES:
            for sg in "+-":
                tol[(ax, sg)], thin[(ax, sg)] = None, []
                for mm in MAGS_MM:
                    c = f"{ax}{sg}{mm}"
                    both = [(R.get((s, "0")), R.get((s, c))) for s in range(*PAIRED)]
                    both = [(x, y) for x, y in both if x and y]
                    pairs = [(x, y) for x, y in both if not x["placement_failed"] and not y["placement_failed"]]
                    if len(pairs) < 0.9 * len(both):  # over 10 % of the pairs lost to placement failures
                        thin[(ax, sg)].append(f"{mm} mm: {len(pairs)} of {len(both)}")
                    if not pairs:
                        print(f"{ax:>6} {sg:>4} {mm:>3} {0:>5}  (no placed pair of {len(both)})")
                        continue
                    r0 = sum(x["success"] for x, _ in pairs) / len(pairs)
                    r1 = sum(y["success"] for _, y in pairs) / len(pairs)
                    b = sum(x["success"] and not y["success"] for x, y in pairs)
                    cc = sum(y["success"] and not x["success"] for x, y in pairs)
                    p = mcnemar_exact(b, cc)
                    meets = (r0 - r1) >= 0.20 - 1e-12 and p < 0.05
                    if meets and tol[(ax, sg)] is None:
                        tol[(ax, sg)] = mm
                    print(f"{ax:>6} {sg:>4} {mm:>3} {len(pairs):>5} {r0:5.2f} {r1:5.2f} {100 * (r0 - r1):7.1f} {b:>3} {cc:>3}"
                          f" {p:9.4f}  {'yes' if meets else ''}")  # fmt: skip
        print("\ntolerance (the smallest |offset| whose rate is >= 20 pp under 0's on the same seeds with McNemar p < 0.05):")
        for ax in AXES:
            note = "".join(f"  [{ax}{sg} pairs left after placement failures: {', '.join(thin[(ax, sg)])}]" for sg in "+-"
                           if thin[(ax, sg)])  # fmt: skip
            print(f"  {ax:>5}: +side {tol[(ax, '+')] or '> 20'} mm, -side {tol[(ax, '-')] or '> 20'} mm{note}")
        vals = [v for v in tol.values() if v is not None]
        print(f"  task tolerance: {min(vals) if vals else '> 20'} mm"
              + (f" ({', '.join(f'{ax}{sg}' for (ax, sg), v in tol.items() if v == min(vals))})" if vals else ""))
        allm = {}
        for r in R.values():
            allm[r["mode"]] = allm.get(r["mode"], 0) + 1
        print("  failure modes, all conditions:", " ".join(f"{k}:{v}" for k, v in sorted(allm.items())))
        nb = sum(bool(r.get("blocked_by_object")) for r in R.values())
        nbs = sum(bool(r.get("blocked_by_object")) and r["success"] for r in R.values())
        print(f"  blocked by the object (scored, in the curve): {nb} episodes, {nbs} of them lifted")
        print(f"  rule: placement_failed = not blocked and the servo's error at the grasp > {1000 * PLACE_FAIL:.0f} mm "
              f"(left out); blocked = the gripper ({BLOCK_BODIES}: {', '.join(GRIPPER) if BLOCK_BODIES == 'gripper' else 'right_dex1_base_link'}) "
              f"touched the object while the servo's error was >= {1000 * SERVO_TOL:.0f} mm (the servo stopped there, scored); "
              f"for a failure: blocked_by_object = blocked; knocked_away = else the object moved >= {KNOCK_MM} mm or turned "
              f">= {KNOCK_DEG} deg before the close; closed_on_nothing = else no pad contact with the object at the end of "
              "the close hold; slipped_after_grasp = the rest")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, choices=TASKS)
    r.add_argument("--seeds", default=f"{DESIGN[0]}:{DESIGN[1]}")
    r.add_argument("--conds", default="0", help="comma list or 'all'")
    r.add_argument("--formal", action="store_true", help="the registration's jobs for --task, --shard k/n of them")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--out", required=True)
    r.add_argument("--set", action="append", help="KEY=json (design trials only)")
    r.add_argument("--render", default="")
    z = sub.add_parser("analyze")
    z.add_argument("rows", nargs="+")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_analyze(a)


if __name__ == "__main__":
    main()
