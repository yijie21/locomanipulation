"""`Primitives`'s tuning constants and kinematic helper functions, split out
of primitives.py (refactor step 4b family 4, pure extraction; decision 9's
file-size cap). Every statement here is unchanged from the original inline
body -- only the packaging moved.

This is the leaf of the mixin split: `primitives_observation.py`,
`primitives_walk.py`, `primitives_arms.py`, `primitives_crouch.py` (the four
mixins `Primitives` is composed from) and `primitives.py` itself all import
from here, and nothing here imports any of them, so there is no circular
import to manage -- unlike families 2/3/5's monkeypatchable module globals,
none of these names is ever overridden by assignment anywhere in the repo
(this is pure numeric tuning + math, not a runtime knob), so a plain
top-level import is correct and sufficient in every consumer.
"""

from __future__ import annotations

import numpy as np

from locomani.backends.sonic.config import DEFAULT_BACKEND
from locomani.backends.sonic.primitives_const import GRASP_CENTRE_OFFSET_CONST

CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

# Hand close/open poses per hand type (SonicBackendConfig.hand, default dex1).
# dex3: G1 three-finger hand, joint order per privileged_pub.HAND_JOINTS;
# signs mirrored between hands (left middle_0 range [-1.5708,0], right
# [0,1.5708]). dex1: two prismatic fingers, both commanded identically —
# +0.0245 = 9.5 cm open, -0.02 = 0.6 cm closed; NOT mirrored.
#
# Module-level default, not an env read (decision 8): never actually overridden
# in any script today, so this constant IS that default.
HAND_TYPE = DEFAULT_BACKEND.hand
if HAND_TYPE == "dex1":
    HAND_CLOSED_LEFT = np.array([-0.02, -0.02, 0, 0, 0, 0, 0], dtype=np.float32)
    HAND_CLOSED_RIGHT = HAND_CLOSED_LEFT.copy()
else:
    HAND_CLOSED_LEFT = np.array([0.5, 0.6, 1.2, -1.2, -1.3, -1.2, -1.3], dtype=np.float32)
    HAND_CLOSED_RIGHT = -HAND_CLOSED_LEFT

# Pre-grasp geometry, LEGS-style: a fixed offset from the object's known pose.
# Approach the object from above, in three steps. A single reach straight to a
# point just over the orange sweeps the open hand in laterally at table height
# and knocks it away — measured, the orange ends up 0.9 to 1.8 m from the hand,
# which then reads as "the fingers closed on nothing". Coming down onto it
# instead keeps the hand clear of the table until it is over the target.
PREGRASP_ABOVE = 0.22  # height of the staging point above the object
PREGRASP_BACK = 0.16  # how far short of the object that staging point sits
GRASP_ABOVE = 0.02  # palm this far above the object centre when closing
LIFT_HEIGHT = 0.10  # how far to raise it after the fingers close

# Where the hand actually grips, in the wrist frame the privileged stream
# reports as "palm". Driving the wrist itself onto the object closes the
# fingers ~11 cm past it, which is what "no contact after closing" meant in the
# second batch: the reach was landing correctly and the hand was gripping empty
# air in front of the object.
#
# Calibrated by closing the hand on the object at every candidate position and
# keeping the ones where the thumb and the fingers end up on OPPOSITE sides
# (locomani/tasks/bottle/diag/diag_grasp_pose.py). The region's centre is
#
#     (0.112, +0.049, +0.001)   half-widths  x +-35, y +-45, z +-50 mm
#
# Two earlier values were both wrong in the same way. Reading the link origins
# out of the model gave (0.11, 0.02, 0); measuring the closest approach between
# the thumb's surface and the fingers' with the hand OPEN gave
# (0.107, -0.001, 0.014). Both put y at roughly zero — and the pinch region
# starts at y = +0.004, so the object sat just outside the jaw and the fingers
# closed beside it every time. An open hand's narrowest gap is at the knuckles;
# where it *holds* something is decided by how it closes.
#
# The tolerances are the encouraging part: +-35 mm is far looser than the
# 0-6 mm the alignment servo already achieves.
GRASP_CENTRE_OFFSET = np.array(GRASP_CENTRE_OFFSET_CONST)

# Where to stand, read off the reachability map (locomani/tasks/bottle/diag/diag_reach_map.py)
# rather than guessed. Solving the arm's IK for both the grip point and the
# staging point, over the controller's measured envelope, leaves a feasible
# region that is closer and further to the right than anything tried so far:
#
#     0.32 m   object bearing -40 to +20 deg
#     0.36 m   object bearing -40 to -10 deg
#     0.40 m   object bearing -40, -30 deg
#     0.44 m+  staging reachable, grip not
#
# Every batch to date stood at 0.45-0.52 m, i.e. outside it, which is why the
# grip reach stalled however the staging offset was tuned.
STAND_OFF = 0.36

# Integrating-reach gains. The loop closes on the privileged palm pose, so the
# gain only sets how fast, not how accurately, it converges.
# Integrating at the 50 Hz control rate oscillates: the arm needs about a
# second to respond, so a fast loop keeps winding the command up through the
# whole lag and then swings it back — measured, that leaves the palm cycling
# between 3 cm and 10 cm of error and never settling. Correct slowly instead,
# and start from an inverse of the measured mapping so there is little to
# correct.
REACH_GAIN = 0.45
REACH_UPDATE_HZ = 3.0
# Hold the seeded command still before integrating at all. The whole-body
# controller squats to follow a low wrist target, and that takes a couple of
# seconds — diag_reach.py only saw the palm reach down to table height because
# it held each command static for 2.5 s. Integrate through that transient
# instead and the loop reads the not-yet-settled palm as failure to descend,
# winds the command into the box limits, and pulls the arm sideways.
REACH_SEED_HOLD = 2.5
# Smooth-move parameters. The command is glided from wherever the arm is
# holding to the seed at NAV_HAND_SPEED, never stepped there.
NAV_HAND_SPEED = 0.30  # metres per second of commanded wrist travel
REACH_GLIDE_MIN = 0.8  # even a 2 cm correction gets this much ramp
REACH_SETTLE_AFTER_GLIDE = 1.2  # let the body finish squatting before servoing
# servo_grasp closes on the palm-frame error, so it can afford a firmer gain
# than reach: the quantity it measures is the one that decides the grasp.
SERVO_GAIN = 0.6
SERVO_MAX_STEP = 0.04
# How far out along the ENTRY axis to hold the object while lining the jaw up
# on the other two. The entry axis is the palm's y — the thumb side — because
# that is the only direction with a clear run at the grip centre: sliding the
# object in along each palm axis with the hand open, +x is blocked 0.105 m out
# by the middle fingertip and +z 0.085 m out by the wrist, while +y is clear to
# within 0.015 m (locomani/tasks/bottle/diag/diag_grasp_pose.py).
# Held just clear of that 0.015 m block, not three times clear of it. The
# align phase pulls the hand back OUT to this depth before correcting, so a
# staging gap smaller than this is wasted travel — grasping.PREGRASP_GAP is
# kept just above it for that reason. Every centimetre here is a centimetre of
# sideways reach, which is the axis the arm runs out of first.
SERVO_SAFE_GAP = 0.10
SERVO_DEPTH_TOL = 0.03
# The advance is throttled by how well the jaw is lined up. This has to be a
# looser bound than align_tol or the throttle is shut the instant alignment is
# declared; at 2.5x it is wide open when aligned and closed by the time the
# drift would matter.
SERVO_GATE_TOL = 0.015
# How long the wrist orientation needs before the palm has actually turned.
# Measured in diag_pitch.py, which uses 3.5 s per case and sees a settled pose;
# the reach only needs the rotation to be done, not the position.
REACH_QUAT_SETTLE = 2.5

# --- track(): stream a pose reference instead of searching for a setpoint ----
# A hand's pace. Slower than the arm can move, on purpose: the reference is
# what the whole body follows, and a reference that outruns the controller is
# the same lurch the seed-and-hold scheme produced, just continuous.
TRACK_SPEED = 0.12  # metres per second of reference travel
TRACK_MIN_S = 1.5  # even a short move gets this much ramp
# How far the arm lags the command. The bias integrator compares the palm
# against the reference from this long ago, so the lag is not mistaken for
# error and integrated away (which overshoots by exactly the lag distance).
TRACK_LAG = 1.0
TRACK_BIAS_GAIN = 1.0  # per second
# Proportional gain on the pelvis-height regulator, in wrist-metres per
# pelvis-metre. Same lever and same 1.5 the crouch has always used.
PELVIS_HOLD_GAIN = 1.5
TRACK_BIAS_MAX = 0.60  # metres; the mapping needs ~0.3, this is a rail
# Pre-scale on the wrist orientation command. 4.5 was the inverse of the
# release model's 22% delivery (105 deg asked, 23 achieved). SONIC v1.1 is a
# different animal: it was trained with wrist-pose augmentation, and the same
# sweep now returns 35-47 deg of nose-down whatever is asked, with two
# identical 0 deg commands landing 25 deg apart. The palm's tilt there is set
# by the arm configuration that reaches the wrist point, not by the quaternion.
# So on v1.1 boosting is meaningless. But v1.1 was reverted (see
# start_stack.sh), and dropping the boost to 1.0 on the RELEASE model is a
# regression, measured: the palm never reaches plan.rot, the whole-body
# controller settles on a different arm solution for the same wrist point, the
# position response changes underneath the bias seed, and the integrator enters
# a limit cycle — one trace converged to 4.1 cm at u=0.68 and then diverged to
# 27 cm with the palm swinging half a metre. 4.5 is the release model's number
# and stays until the checkpoint changes.
TRACK_ROT_BOOST = 4.5
TRACK_ROT_GAIN = 0.5  # per second, on the orientation trim
# Stop this far short of the grip point, along the approach line. The plan ends
# with the grasp centre ON the object, so tracking all the way there puts the
# fingers into it before the jaw has been checked — which is exactly what the
# object guard then reports as "moved 2 cm while the hand was on its way in
# (touched R:index_0, R:middle_0)". That contact is the intended one arriving
# early. Hand the last few centimetres to servo_grasp, which closes on the
# palm-frame error and knows how to stop.
TRACK_STOP_SHORT = 0.06
REACH_MAX_STEP = 0.06  # metres of command change per update
REACH_STALL_S = 4.0  # no improvement for this long => out of reach
REACH_PROGRESS = 0.005  # what counts as improvement, in metres
# Commanded wrist targets stay inside a plausible workspace box (pelvis frame),
# so a saturating loop cannot wind the command out to absurd values.
REACH_BOX_LO = np.array([-0.10, -0.70, -0.60])
REACH_BOX_HI = np.array([1.20, 0.70, 0.70])

# Inverse of the wrist-target -> palm mapping, measured by diag_reach.py over a
# grid at table height: palm_x ~ 0.55 * cmd_x + 0.02 and palm_z ~ 0.5 * cmd_z +
# 0.07, both in pelvis frame, while y tracks 1:1. Only a seed for the
# integrator, which removes whatever this gets wrong.
#
# The same sweep settled two other questions. `vr_3pt_compliance` is inert —
# 0.5 and 1.0 give identical palm positions to the millimetre. And the
# planner's `height` field does not squat the robot: pelvis height stayed at
# 0.78 across -1, 0.70, 0.60 and 0.50. What actually lowers the body is a low
# *wrist* target — the whole-body controller squats to follow the arm, down to
# a pelvis at 0.65 m. So reaching a 0.75 m table needs no stance command at
# all, just a target the controller will bend for.
#
# The reachable envelope is the real constraint: the palm gets about 0.50 m in
# front of the pelvis, which is why STAND_OFF has to be well under that.
REACH_X_GAIN, REACH_X_OFFSET = 0.55, -0.02
REACH_Z_GAIN, REACH_Z_OFFSET = 0.50, -0.07

# How far to the side of straight-ahead the grasp target should sit. The right
# hand hangs to the right, so turning this much *past* the object leaves it in
# front of that hand instead of on the body's centreline.
# Back to 32 deg — but for the ARM, no longer for the palm. The reachability
# map puts the arm's feasible cells at object bearings -40 to -20 deg; facing
# the object square-on (bearing 0) stalled 5 of 10 staging reaches. The palm,
# meanwhile, must stay at the identity orientation, because commanding wrist
# yaw pitches it ~25 deg down (locomani/diag_wrist.py) — which is what had
# the fingertips sweeping the can off the table at 5 mm of reported error. The
# resolution is that the palm does not need to face the object at all: the
# plan takes the pelvis-forward direction as the palm's facing
# (grasping.grasp_pose(approach_dir=...)) and the object slides into the jaw
# from the side.
GRASP_BEARING = np.deg2rad(32.0)

# Where the arm actually starts, and how far it goes. Both measured off a probe
# compile of SONIC_SCENE in the nominal pose: the right shoulder-roll joint sits
# at this offset in the pelvis frame, and the shoulder->elbow->wrist links sum
# to ARM_REACH when straight. Mirror y for the left arm.
#
# This replaces "distance from the pelvis" as the test of whether a grasp is
# in range, and the difference is not a refinement. Pelvis distance is a
# rotationally symmetric measure, so it says a point dead ahead is easier than
# the same point off to the hand's side — the exact opposite of the truth for
# a one-armed reach, and following it would aim the robot into a cross-body
# reach every time. Measured from the shoulder, the four episodes of the last
# batch separate cleanly: the one that grasped needed 0.277 m, the one that
# converged and then failed to lift needed 0.378, and the two that stalled
# needed 0.412 and 0.441.
SHOULDER_OFFSET = np.array([0.0, -0.141, 0.289])
ARM_REACH = 0.373
# What the pelvis sits at with nothing commanded, and how small a squat is not
# worth having. The crouch costs a held arm and several seconds, and it is the
# mechanism that collapsed under every reach; below a few centimetres it buys
# less than it risks.
STANDING_PELVIS_Z = 0.793
CROUCH_SKIP = 0.03
# What the palm actually reaches when grasping.PALM_PITCH_CMD is commanded.
# The command is 105 deg; this is the 23 deg it delivers (diag_pitch.py). Used
# wherever the REAL orientation matters rather than the request — drawing the
# ghost, and any offline check of where the fingers will land.
PALM_PITCH_ACHIEVED = np.deg2rad(23.0)
# What to actually plan against. Above the rigid-link sum because the shoulder
# itself travels when the torso leans: 0.378 m converged live.
ARM_REACH_LIMIT = 0.39

# How far off the grasping hand's side the object may sit once the robot has
# stopped. Coarse now: the grasp stage re-aims (turn_to) and walks itself the
# rest of the way in, so this only rejects stands with the object nearly
# beside the robot. At 0.32 it rejected d=0.60 stands whose heading the aim
# stage would have fixed in two seconds.
APPROACH_LATERAL_MAX = 0.55

# Floor on the corrective stand-off. The table's footprint reaches 0.617 m and
# the body needs 0.30, so the standing ring sits 0.92 m from its centre; with
# the can 0.58 m out that is 0.34 m of separation, and the floor only has to
# stay under it.
MIN_STAND_OFF = 0.30

# -- navigation -------------------------------------------------------------
# How far ahead the follower looks along the distance field. Shorter than this
# and the bearing chatters at the grid resolution; longer and the robot cuts
# corners the field was routing it around.
NAV_STEP = 0.45
# Three bands, because a humanoid walking sideways is a humanoid about to fall.
# Beyond TURN_IN_PLACE, stop and turn; inside TURN_STRAIGHT, ignore the error
# and walk; in between, walk slower the worse the alignment.
NAV_TURN_IN_PLACE = np.deg2rad(60.0)
NAV_TURN_STRAIGHT = np.deg2rad(5.0)
# Geodesic metres from the goal set. Every centimetre here is a centimetre the
# arm has to make up later, and the arm is the scarce resource: measured
# against the map, the nearest legal pelvis cell to the can is 0.355 m away,
# while approach kept plateauing at 0.43 — five attempts landing on 0.433,
# 0.430, 0.431, 0.434, none of them blocked by anything. That plateau was this
# tolerance, not the geometry.
# 0.07 was too tight: it is finer than one step of the gait, so approach began
# retrying instead of arriving — five attempts landing on 0.536 m unchanged in
# one episode, and a 2.13 m walk-away in another. 0.10 keeps most of the ground
# the tighter value won (the plateau was at 0.43) without asking the walk to
# resolve better than it steps.
NAV_ARRIVE = 0.10
NAV_EASE = 0.60  # start easing off this far out
NAV_REPLAN_S = 0.5  # how often to re-march with a fresh corridor
# Half-width of the acceptable standing annulus around a manipulation target.
NAV_RING_BAND = 0.07
# Give up rather than walk on forever. Both guards exist because the deploy's
# odometry does not recover from a fall: the robot then walks briskly in a
# direction unrelated to the command, and the only symptom the field shows is
# that the distance to the goal stops falling.
NAV_STALL_S = 8.0
NAV_PROGRESS = 0.05  # metres of geodesic progress that counts
# Commanded pace. The mapping is steeply non-linear (measured net travel:
# 0.30 -> 0.05 m/s, 0.50 -> 0.42 m/s), so this sits just under the knee: brisk
# enough to cross the room, slow enough that the gait is not fighting itself.
NAV_SPEED = 0.45
# Re-measured on open floor with a reset per leg (locomani/diag_shuffle.py,
# 3 repeats): 0.40 -> 0.29 m/s (25 cm/step), 0.30 -> 0.247 (17), 0.26 -> 0.155
# (11), 0.215 -> 0.04 (5), 0.185-0.205 -> ~0.013 (1-2 cm/step, still stepping
# at 1.1 Hz — the shuffle), and a zero movement vector marches in place. The
# OLD note here claimed 0.30 travelled 0.05 m/s "i.e. nothing"; that reading
# came from a leg that had walked into the table, and believing it is what
# made the gym's final crawl run at a 17 cm stride. Easing toward zero on a
# long walk still stalls the approach (7 of 10 episodes halted 0.2 m short),
# so NAV_SPEED_MIN stays — the shuffle band is for the last few centimetres,
# where the loop closes on the measured object instead of on arrival.
NAV_SPEED_MIN = 0.35

# Where the wrists sit while walking. Not cosmetic: an episode analysed frame
# by frame (outputs/videos/low_004.mp4) fell with the pelvis 0.26 m clear of
# the table — it was the *hand*, held out forward at rest, that caught the
# corner. Arms out give the robot an effective radius near 0.42 m, which leaves
# almost no legal standing ring at a table; tucked, it is about 0.25 m, and a
# 0.30 m inflation then has room to work.
TUCK_WRISTS = {"left": np.array([0.16, 0.16, -0.02]), "right": np.array([0.16, -0.16, -0.02])}

# The standing/walking READY pose: upper arm hanging, forearm LEVEL with the
# ground, wrists just above pelvis height — the posture a waiter walks with.
# Carried from the first step of the episode so the hand never has to come up
# from a tuck at the table (the raise-at-the-table step swept the can once
# and clipped the tabletop before that). The z puts the hands above every
# surface-A tabletop, so the extra forward radius (~0.35 m vs 0.25 tucked)
# lives entirely ABOVE table height and the corner-catch that felled
# low_004 cannot recur.
READY_WRISTS = {"left": np.array([0.24, 0.18, 0.05]), "right": np.array([0.24, -0.18, 0.05])}

# The same READY pose in JOINT space, for the planner's FROZEN_UPPER_BODY
# mode — the only mode that actually holds the arms through a walk (VR-3pt
# wrist targets are tracked while standing and ignored the moment the gait
# starts; watched: arms swinging straight down mid-walk with READY_WRISTS
# set). Shoulder hanging, elbow folded to 1.5 rad = forearm level with the
# ground — the joint pose the elbow-raise era verified keeps the torso
# upright.
# Elbow ZERO is the level forearm: the G1's joint zero is the L-pose and
# POSITIVE elbow EXTENDS the arm downward. (The earlier 2.09 values here
# were the inverted-sign era — commanded hanging arms while believing they
# commanded a fold.) Held exactly by the sim-side arm pin during walking;
# the frozen-upper-body reference carries the same values so the policy's
# balance model agrees with what the motors do.
READY_ARM = {"left": [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], "right": [0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]}

# The raised walking variant: shoulder pulled back, elbow compensating so
# the forearm stays level, wrist pitched up. Lifts the hand's LOWEST point
# (FK + measured hand point cloud) from pelvis-frame +0.055 to +0.113 —
# the hand-sweep clearance pose for walking near tables with objects on
# them, where the level-forearm posture's larger effective radius would
# otherwise sweep through anything taller than ~0.80 m.
READY_ARM_NEAR = {"left": [-0.35, 0.2, 0.0, 0.30, 0.0, -0.40, 0.0], "right": [-0.35, -0.2, 0.0, 0.30, 0.0, -0.40, 0.0]}
# Lowest hand point, pelvis frame, for each walking pose (FK, hand_points).
HAND_LOW_PF = {"ready": 0.055, "near": 0.113}
GAIT_PELVIS_DIP = 0.04  # how far the pelvis dips below standing mid-gait


def walking_arm_pose(obstacle_top_z: float, margin: float = 0.08):
    """Pick the walking arm pose whose swept hand volume clears
    `obstacle_top_z` (world) by `margin` even at the gait's lowest pelvis.
    Returns (name, {side: 7 joints}). Falls back to the raised pose."""
    for name, pose in (("ready", READY_ARM), ("near", READY_ARM_NEAR)):
        low = STANDING_PELVIS_Z - GAIT_PELVIS_DIP + HAND_LOW_PF[name]
        if low >= obstacle_top_z + margin:
            return name, pose
    return "near", READY_ARM_NEAR


def upper_body_17(left7, right7, waist=(0.0, 0.0, 0.0)) -> np.ndarray:
    """The planner message's 17-value upper-body order: waist y/r/p, then
    left/right-interleaved shoulder p, r, y, elbow, wrist r, p, y."""
    v = list(waist)
    for i in range(7):
        v += [left7[i], right7[i]]

    return np.array(v, dtype=np.float32)


# Names for that order, so a diagnostic can say WHICH joint it means.
UPPER_BODY_NAMES = ["waist_y", "waist_r", "waist_p"] + [
    f"{side}_{j}" for j in ("sh_p", "sh_r", "sh_y", "elbow", "wr_r", "wr_p", "wr_y") for side in ("L", "R")
]


def upper_body_from_mj(q29) -> np.ndarray:
    """Snapshot a MuJoCo-ordered 29-joint state into the 17-value target."""
    q = np.asarray(q29, dtype=np.float32)

    return upper_body_17(q[15:22], q[22:29], waist=q[12:15])


READY_UPPER = upper_body_17(READY_ARM["left"], READY_ARM["right"])

# Crouch. The holding arm's target sits beside the body, not in front of it,
# so a deep squat does not swing that hand into the table.
CROUCH_HOLD_X, CROUCH_HOLD_Y = 0.12, 0.26
CROUCH_WRIST_MIN, CROUCH_WRIST_MAX = -0.60, 0.10
CROUCH_STEP = 0.03  # metres of wrist command per control tick
CROUCH_TOL = 0.025  # how close the pelvis has to get, in metres
# How far above the pelvis to put the grip point. The reachable band starts at
# the pelvis and runs 0.20 m up, so a few centimetres of margin costs nothing
# and absorbs the bob.
CROUCH_MARGIN = 0.05


def _opposed(state: dict, obj_key: str, side: str) -> bool:
    """Is the object held between the thumb and the fingers, or just touched?

    The privileged stream reports contacting link names, so this is a direct
    read rather than an inference. One-sided contact is what pushing an object
    across a table looks like; opposed contact is what holding it looks like.
    """
    tag = "R:" if side == "right" else "L:"
    links = state["objects"][obj_key].get("touch_links", [])
    mine = [l for l in links if l.startswith(tag)]

    return any("thumb" in l for l in mine) and any(("index" in l or "middle" in l) for l in mine)


def yaw_of(quat) -> float:
    w, x, y, z = quat

    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _axis_angle(rot: np.ndarray):
    """Rotation matrix -> (unit axis, angle). Identity returns +z, 0."""
    angle = float(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if angle > np.pi - 1e-6:
        # Near 180 deg the antisymmetric part vanishes; take the axis from the
        # symmetric part instead, which is where it survives.
        w, v = np.linalg.eigh((rot + np.eye(3)) / 2.0)
        axis = v[:, int(np.argmax(w))]

        return axis / max(float(np.linalg.norm(axis)), 1e-9), angle
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2.0 * np.sin(angle))

    return axis / max(float(np.linalg.norm(axis)), 1e-9), angle


def _from_axis_angle(axis, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])

    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rot_scale(rot: np.ndarray, k: float) -> np.ndarray:
    """Same axis, angle multiplied by k and clamped short of a half turn.

    Used to pre-compensate the attenuated wrist channel: asking for k times the
    turn gets roughly the turn wanted. Clamped because past 180 deg the command
    wraps and starts asking for rotation the other way round, which is how the
    120 deg pitch command produced less tilt than 105.
    """
    axis, angle = _axis_angle(rot)

    return _from_axis_angle(axis, float(np.clip(angle * k, -2.9, 2.9)))


def _rot_lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Move `a` a fraction `t` of the way to `b`, along their relative axis."""
    axis, angle = _axis_angle(b @ a.T)

    return _from_axis_angle(axis, angle * float(np.clip(t, 0.0, 1.0))) @ a


def _quat_of(rot: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion of a rotation matrix, without mujoco."""
    m = np.asarray(rot, dtype=float)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]

    return np.array(q, dtype=np.float32)


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
