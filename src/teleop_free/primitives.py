"""Motion primitives for the scripted collector (P1.2).

LEGS's insight is that a whole-body controller already solves the hard part:
given a heading, a walk command and a wrist target, SONIC produces plausible
whole-body motion. What the data engine has to supply is only *when* to issue
which command — and privileged simulator state is enough to decide that. So
there is no grasp generator here, just a fixed pre-grasp offset expressed
relative to the object's known pose, exactly as LEGS does.

Five primitives, each a closed loop on the privileged state:

    walk_to(xy)     drive the pelvis to a standing spot, in a straight line
    navigate()      follow the occupancy map's distance field around obstacles
    turn_to(yaw)    rotate in place
    reach(pos)      put a palm at a world point, via a root-frame wrist target
    grasp(side)     close the hand and confirm contact
    release(side)   open the hand

Frames. Wrist targets go out in the *root* (pelvis) frame — see the wire-format
note in agent_server.py — so every world-frame target is rotated by the pelvis
yaw before it is sent. `movement` and `facing` go out in the world frame, which
is what pico_manager_thread_server sends (its YawAccumulator integrates a
world-frame heading vector, planner loop line 1826-1830).
"""

from __future__ import annotations

import collections
import time

import numpy as np

import grasping
import navigator as nav
import wrist_map
from agent_server import (
    AgentCommander,
    HAND_OPEN,
    LocomotionMode,
    StreamMode,
)
from primitives_const import GRASP_CENTRE_OFFSET_CONST
from privileged_pub import PrivilegedSubscriber

CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

# The G1 three-finger hand, in the joint order privileged_pub.HAND_JOINTS lists.
# Signs are mirrored between hands because the joint axes are (see the ranges in
# g1_29dof_with_hand.xml: left middle_0 is [-1.5708, 0], right is [0, 1.5708]).
HAND_CLOSED_LEFT = np.array([0.5, 0.6, 1.2, -1.2, -1.3, -1.2, -1.3], dtype=np.float32)
HAND_CLOSED_RIGHT = -HAND_CLOSED_LEFT

# Pre-grasp geometry, LEGS-style: a fixed offset from the object's known pose.
# Approach the object from above, in three steps. A single reach straight to a
# point just over the orange sweeps the open hand in laterally at table height
# and knocks it away — measured, the orange ends up 0.9 to 1.8 m from the hand,
# which then reads as "the fingers closed on nothing". Coming down onto it
# instead keeps the hand clear of the table until it is over the target.
PREGRASP_ABOVE = 0.22      # height of the staging point above the object
PREGRASP_BACK = 0.16       # how far short of the object that staging point sits
GRASP_ABOVE = 0.02         # palm this far above the object centre when closing
LIFT_HEIGHT = 0.20         # how far to raise it after the fingers close

# Where the hand actually grips, in the wrist frame the privileged stream
# reports as "palm". Driving the wrist itself onto the object closes the
# fingers ~11 cm past it, which is what "no contact after closing" meant in the
# second batch: the reach was landing correctly and the hand was gripping empty
# air in front of the object.
#
# Calibrated by closing the hand on the object at every candidate position and
# keeping the ones where the thumb and the fingers end up on OPPOSITE sides
# (teleop_free/diag_grasp_pose.py). The region's centre is
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

# Where to stand, read off the reachability map (teleop_free/diag_reach_map.py)
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
NAV_HAND_SPEED = 0.30          # metres per second of commanded wrist travel
REACH_GLIDE_MIN = 0.8          # even a 2 cm correction gets this much ramp
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
# within 0.015 m (teleop_free/diag_grasp_pose.py).
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
TRACK_SPEED = 0.12             # metres per second of reference travel
TRACK_MIN_S = 1.5              # even a short move gets this much ramp
# How far the arm lags the command. The bias integrator compares the palm
# against the reference from this long ago, so the lag is not mistaken for
# error and integrated away (which overshoots by exactly the lag distance).
TRACK_LAG = 1.0
TRACK_BIAS_GAIN = 1.0          # per second
# Proportional gain on the pelvis-height regulator, in wrist-metres per
# pelvis-metre. Same lever and same 1.5 the crouch has always used.
PELVIS_HOLD_GAIN = 1.5
TRACK_BIAS_MAX = 0.60          # metres; the mapping needs ~0.3, this is a rail
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
TRACK_ROT_GAIN = 0.5           # per second, on the orientation trim
# Stop this far short of the grip point, along the approach line. The plan ends
# with the grasp centre ON the object, so tracking all the way there puts the
# fingers into it before the jaw has been checked — which is exactly what the
# object guard then reports as "moved 2 cm while the hand was on its way in
# (touched R:index_0, R:middle_0)". That contact is the intended one arriving
# early. Hand the last few centimetres to servo_grasp, which closes on the
# palm-frame error and knows how to stop.
TRACK_STOP_SHORT = 0.06
REACH_MAX_STEP = 0.06          # metres of command change per update
REACH_STALL_S = 4.0            # no improvement for this long => out of reach
REACH_PROGRESS = 0.005         # what counts as improvement, in metres
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
# yaw pitches it ~25 deg down (teleop_free/diag_wrist.py) — which is what had
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
NAV_EASE = 0.60                # start easing off this far out
NAV_REPLAN_S = 0.5             # how often to re-march with a fresh corridor
# Half-width of the acceptable standing annulus around a manipulation target.
NAV_RING_BAND = 0.07
# Give up rather than walk on forever. Both guards exist because the deploy's
# odometry does not recover from a fall: the robot then walks briskly in a
# direction unrelated to the command, and the only symptom the field shows is
# that the distance to the goal stops falling.
NAV_STALL_S = 8.0
NAV_PROGRESS = 0.05            # metres of geodesic progress that counts
# Commanded pace. The mapping is steeply non-linear (measured net travel:
# 0.30 -> 0.05 m/s, 0.50 -> 0.42 m/s), so this sits just under the knee: brisk
# enough to cross the room, slow enough that the gait is not fighting itself.
NAV_SPEED = 0.45
# Below about 0.35 the gait has a deadband — commanded 0.30 travels 0.05 m/s
# and 0.20 travels 0.02, i.e. nothing (teleop_free/diag_speed.py). Easing the
# speed toward zero on approach therefore does not slow the robot down, it
# stops it: 7 of 10 episodes halted 0.2 m short reporting "no progress". Stop
# by arriving, not by fading out.
NAV_SPEED_MIN = 0.35

# Where the wrists sit while walking. Not cosmetic: an episode analysed frame
# by frame (outputs/videos/low_004.mp4) fell with the pelvis 0.26 m clear of
# the table — it was the *hand*, held out forward at rest, that caught the
# corner. Arms out give the robot an effective radius near 0.42 m, which leaves
# almost no legal standing ring at a table; tucked, it is about 0.25 m, and a
# 0.30 m inflation then has room to work.
TUCK_WRISTS = {"left": np.array([0.16, 0.16, -0.02]),
               "right": np.array([0.16, -0.16, -0.02])}

# The standing/walking READY pose: upper arm hanging, forearm LEVEL with the
# ground, wrists just above pelvis height — the posture a waiter walks with.
# Carried from the first step of the episode so the hand never has to come up
# from a tuck at the table (the raise-at-the-table step swept the can once
# and clipped the tabletop before that). The z puts the hands above every
# surface-A tabletop, so the extra forward radius (~0.35 m vs 0.25 tucked)
# lives entirely ABOVE table height and the corner-catch that felled
# low_004 cannot recur.
READY_WRISTS = {"left": np.array([0.24, 0.18, 0.05]),
                "right": np.array([0.24, -0.18, 0.05])}

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
READY_ARM = {"left": [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
             "right": [0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]}

# The raised walking variant: shoulder pulled back, elbow compensating so
# the forearm stays level, wrist pitched up. Lifts the hand's LOWEST point
# (FK + measured hand point cloud) from pelvis-frame +0.055 to +0.113 —
# the hand-sweep clearance pose for walking near tables with objects on
# them, where the level-forearm posture's larger effective radius would
# otherwise sweep through anything taller than ~0.80 m.
READY_ARM_NEAR = {"left": [-0.35, 0.2, 0.0, 0.30, 0.0, -0.40, 0.0],
                  "right": [-0.35, -0.2, 0.0, 0.30, 0.0, -0.40, 0.0]}
# Lowest hand point, pelvis frame, for each walking pose (FK, hand_points).
HAND_LOW_PF = {"ready": 0.055, "near": 0.113}
GAIT_PELVIS_DIP = 0.04        # how far the pelvis dips below standing mid-gait


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


def upper_body_from_mj(q29) -> np.ndarray:
    """Snapshot a MuJoCo-ordered 29-joint state into the 17-value target."""
    q = np.asarray(q29, dtype=np.float32)

    return upper_body_17(q[15:22], q[22:29], waist=q[12:15])


READY_UPPER = upper_body_17(READY_ARM["left"], READY_ARM["right"])

# Crouch. The holding arm's target sits beside the body, not in front of it,
# so a deep squat does not swing that hand into the table.
CROUCH_HOLD_X, CROUCH_HOLD_Y = 0.12, 0.26
CROUCH_WRIST_MIN, CROUCH_WRIST_MAX = -0.60, 0.10
CROUCH_STEP = 0.03             # metres of wrist command per control tick
CROUCH_TOL = 0.025             # how close the pelvis has to get, in metres
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

    return (any("thumb" in l for l in mine)
            and any(("index" in l or "middle" in l) for l in mine))


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
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0],
                     rot[1, 0] - rot[0, 1]]) / (2.0 * np.sin(angle))

    return axis / max(float(np.linalg.norm(axis)), 1e-9), angle


def _from_axis_angle(axis, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])

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
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]

    return np.array(q, dtype=np.float32)


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class Primitives:
    """One episode's worth of scripted control over the live stack."""

    def __init__(self, commander: AgentCommander | None = None,
                 state: PrivilegedSubscriber | None = None, verbose: bool = True,
                 meta: dict | None = None):
        self.cmd = commander or AgentCommander()
        self.state = state or PrivilegedSubscriber()
        self.verbose = verbose
        self.aborted: str | None = None
        self.last_cmd: np.ndarray | None = None      # for reach(resume=True)
        self.crouch_cmd: np.ndarray | None = None    # the arm holding a squat
        # Yaw anchor. Every loop that sends facing=(current yaw) is a random
        # walk — "hold whatever heading I have now", re-read each tick, drifts
        # freely. Measured: 50 deg of rotation between planning a grasp and
        # reaching for it, which put the pregrasp behind the shoulder. When
        # set, every stationary loop faces THIS heading instead.
        self.hold_yaw: float | None = None
        # Active crouch: (side, target pelvis z, cmd at convergence). A static
        # pelvis-frame wrist command cannot hold a squat — when the pelvis
        # rises, the command rises with it and the controller happily stands
        # back up (measured: 0.65 -> 0.78 during the staging reach). The hold
        # is closed-loop: each tick of any loop, the holding arm's z is biased
        # by how far the pelvis has risen above its target.
        self.crouch_hold: tuple[str, float, np.ndarray] | None = None
        # The map is static — the surfaces never move — so it is built once per
        # process and the per-episode cost is one fast-march per leg.
        self.occ = self.planner = None
        if meta is not None:
            self.occ, self.planner = nav.build(meta)

    # -- state helpers -----------------------------------------------------

    @staticmethod
    def _legacy_seed(local) -> np.ndarray:
        """The hand-fitted inverse, kept only as a fallback.

        Measured against 64 samples it is wrong by 24.6 cm on average and 47.4
        at worst — z has the wrong sign and y is not modelled at all. It stands
        in only until diag_map.py has been run on a given deploy, and anything
        it produces should be treated as a starting guess, not a calibration.
        """
        seed = np.asarray(local, dtype=float).copy()
        seed[0] = (seed[0] + REACH_X_OFFSET) / REACH_X_GAIN
        seed[2] = (seed[2] + REACH_Z_OFFSET) / REACH_Z_GAIN

        return seed

    def obs(self) -> dict:
        return self.state.wait()

    def pelvis_xy(self, s=None):
        s = s or self.obs()

        return np.array(s["pelvis"]["pos"][:2])

    def pelvis_yaw(self, s=None) -> float:
        s = s or self.obs()

        return float(s["pelvis"]["yaw"])

    def object_pos(self, key: str, s=None):
        s = s or self.obs()

        return np.array(s["objects"][key]["pos"])

    def world_to_root(self, p_world, s=None):
        """World point -> pelvis frame, yaw-only (the pelvis stays near upright)."""
        s = s or self.obs()
        root = np.array(s["pelvis"]["pos"])
        yaw = float(s["pelvis"]["yaw"])
        d = np.asarray(p_world, dtype=float) - root
        c, sn = np.cos(-yaw), np.sin(-yaw)

        return np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1], d[2]])

    @staticmethod
    def _rot(quat) -> np.ndarray:
        w, x, y, z = quat

        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def world_to_root_full(self, p_world, s=None):
        """World point -> pelvis frame using the pelvis's *full* orientation.

        The yaw-only version is fine for walking, where the pelvis stays near
        upright, but not for reaching: the robot pitches noticeably as it
        extends, and a yaw-only frame then disagrees with the frame the
        controller interprets the wrist target in. Feeding an error measured in
        one frame into a command expressed in another leaves a standing offset
        that no amount of integrating removes.
        """
        s = s or self.obs()
        d = np.asarray(p_world, dtype=float) - np.array(s["pelvis"]["pos"])

        return self._rot(s["pelvis"]["quat"]).T @ d

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    # -- primitives --------------------------------------------------------

    def walk_to(self, target_xy, stop_dist: float = 0.05, timeout: float = 30.0,
                mode: LocomotionMode = LocomotionMode.SLOW_WALK,
                face_target: bool = True) -> bool:
        """Drive the pelvis to within `stop_dist` of `target_xy`.

        `movement` is a world-frame direction with magnitude in [0, 1] acting as
        a speed scale, so it is recomputed every tick from the live position
        error — which is what makes this robust to the perturbed start poses
        P2 samples.
        """
        target = np.asarray(target_xy, dtype=float)
        end = time.time() + timeout
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                # Where it fell is the whole question — the orbit, the radial
                # leg in, and the corrective re-aim all use walk_to, and they
                # fail for different reasons.
                self.aborted = (f"fell while walking at "
                                f"{np.round(self.pelvis_xy(s), 2)} "
                                f"heading for {target.round(2)}")
                return False
            err = target - self.pelvis_xy(s)
            dist = float(np.linalg.norm(err))
            if dist < stop_dist:
                self.cmd.frame(mode=LocomotionMode.IDLE)
                self.log(f"walk_to {target.round(2)} done ({dist:.3f} m)")
                return True

            direction = err / max(dist, 1e-6)
            # Ease off over the last 0.5 m so it does not overshoot the stop.
            gain = float(np.clip(dist / 0.5, 0.25, 1.0))
            facing = direction if face_target else np.array(
                [np.cos(self.pelvis_yaw(s)), np.sin(self.pelvis_yaw(s))])
            # Same as navigate: the deploy normalises the movement vector, so
            # the gain has to go into `speed` to have any effect at all.
            self.cmd.frame(
                mode=mode,
                speed=float(np.clip(NAV_SPEED * gain, NAV_SPEED_MIN, NAV_SPEED)),
                movement=(direction[0], direction[1], 0.0),
                facing=(facing[0], facing[1], 0.0),
            )
            time.sleep(DT)

        self.aborted = f"walk_to {target.round(2)} timed out"
        return False

    def turn_to(self, yaw: float, tol: float = 0.15, timeout: float = 15.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while turning"
                return False
            err = wrap(yaw - self.pelvis_yaw(s))
            if abs(err) < tol:
                self.cmd.frame(mode=LocomotionMode.IDLE,
                               facing=(np.cos(yaw), np.sin(yaw), 0.0))
                self.log(f"turn_to {np.rad2deg(yaw):.0f}deg done")
                return True
            # Walk the facing command toward the goal instead of jumping it
            # there: an IDLE facing more than ~60 deg away stalls the
            # controller (measured: 68 deg turns timing out while 30 deg turns
            # finish in a second), so keep the commanded heading within a step
            # the policy actually takes.
            goal = self.pelvis_yaw(s) + np.clip(err, -0.5, 0.5)
            self.cmd.frame(mode=LocomotionMode.IDLE,
                           facing=(np.cos(goal), np.sin(goal), 0.0))
            time.sleep(DT)

        self.aborted = f"turn_to {np.rad2deg(yaw):.0f}deg timed out"
        return False

    def reach(self, world_pos, side: str = "right", tol: float = 0.05,
              settle: float = 0.8, timeout: float = 10.0,
              height: float = -1.0, palm_offset=None,
              resume: bool = False, quat=None, glide: float | None = None,
              speed: float = NAV_HAND_SPEED) -> bool:
        """Drive the palm to `world_pos` by integrating the measured error.

        Commanding the target directly does not work. Measured on the live
        stack (teleop_free/diag_reach.py), the wrist target's y and z track to
        within ~3 cm but x lands roughly `0.75 * x - 0.09`: ask for 0.65 m
        ahead and the palm stops at 0.40 m. Raising `vr_3pt_compliance` changes
        nothing at all — 0.5 and 1.0 give identical results to the millimetre,
        so it is not a tracking gain.

        Rather than fit that mapping and re-fit it whenever the pose changes,
        integrate: nudge the commanded point by the error between where the
        palm *is* and where it should be. The privileged stream already carries
        the true palm pose, so the loop closes on the real quantity and absorbs
        whatever the mapping happens to be.

        Two phases, not one. The integrator alone produces the motion the
        videos showed — the commanded point teleports to the seed and the arm
        lunges after it, then holds for REACH_SEED_HOLD, then creeps. The seed
        is computed by dividing by the tracking gains (0.55 in x, 0.50 in z),
        so a 0.20 m change of target becomes a 0.40 m step of command in a
        single tick. So: glide the command from wherever the arm is already
        holding to the seed at a bounded speed, then let the integrator take
        over. The arm follows a trajectory instead of chasing a jump.

        That is not cosmetic. These trajectories are training data, and a
        policy finetuned on lunges will lunge.
        """
        end = time.time() + timeout
        ok_since = None
        # `resume` continues from the command the previous reach left the arm
        # holding, instead of re-seeding from the inverse model. A refinement
        # pass must resume: re-seeding jumps the command discontinuously and
        # REACH_SEED_HOLD then pins the arm at that jump for 2.5 s, which is a
        # swipe, not a correction. Measured, adding a naive second descent pass
        # took "orange knocked away" from 1/10 to 6/10 of episodes.
        cmd_local = self.last_cmd.copy() if (resume and self.last_cmd is not None) else None
        palm_rot = None
        err = float("inf")
        best = float("inf")
        stalled_since = time.time()
        next_log = time.time()
        next_update = time.time() + (0.0 if cmd_local is not None else REACH_SEED_HOLD)
        # Glide phase state. `start` is what the arm is being told to hold
        # right now, which is where a smooth move has to begin.
        start = np.array(self.cmd.right_wrist_pos if side == "right"
                         else self.cmd.left_wrist_pos, dtype=float)
        seed = None
        glide_t0 = glide_s = None
        if quat is not None:
            # Command the orientation and let it ARRIVE before anything else
            # reads the palm. The loop below freezes the palm rotation on its
            # first tick and uses it to turn the grip point into a wrist point;
            # freeze it while the hand is still rotating and the offset is
            # taken from the old orientation, which for the tilted grasp is a
            # 23 deg error on a 0.12 m lever — 5 cm of target, silently.
            self.cmd.set_wrists(**{f"{side}_quat": np.asarray(quat, np.float32)})
            settle_end = time.time() + REACH_QUAT_SETTLE
            while time.time() < settle_end:
                st = self.obs()
                self._maintain_crouch(st)
                self.cmd.frame(mode=LocomotionMode.IDLE, height=height,
                               facing=self._facing(st))
                time.sleep(DT)
            end += REACH_QUAT_SETTLE

        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while reaching"
                return False

            # `world_pos` is where the *grasp centre* should end up, but what
            # the loop can steer is the wrist, 11 cm behind it. Converting one
            # into the other needs the palm's orientation — and that
            # orientation must be FROZEN at the start of the leg, not tracked.
            #
            # Tracking it live is a positive feedback loop, and it is what has
            # been failing every grasp. Measured on the descent: the palm came
            # down 0.167 m while the target it was chasing rose 0.117 m and
            # moved 0.218 m forward, because descending rotates the wrist and
            # rotating the wrist swings the offset. The error sat at 0.217 m
            # for the whole stall window and then reported "out of the arm's
            # workspace", which it never was.
            #
            # Frozen, the target is a fixed point the integrator can converge
            # on. The cost is that any subsequent wrist rotation shows up as a
            # position error of at most the offset length; that is bounded and
            # measurable, whereas a moving target is neither.
            goal = np.asarray(world_pos, dtype=float)
            if palm_offset is not None:
                if palm_rot is None:
                    palm_rot = self._rot(s["palms"][side]["quat"])
                goal = goal - palm_rot @ palm_offset

            desired_local = self.world_to_root_full(goal, s)
            palm_local = self.world_to_root_full(s["palms"][side]["pos"], s)
            if seed is None:
                seed = (wrist_map.command_for(desired_local)
                        if wrist_map.available()
                        else self._legacy_seed(desired_local))
                seed = np.clip(seed, REACH_BOX_LO, REACH_BOX_HI)
                # Time the glide by distance, so a short correction is quick
                # and a long move is not a lunge. Floored, because even a tiny
                # step wants a moment of ramp at each end.
                span = float(np.linalg.norm(seed - start))
                glide_s = (glide if glide is not None
                           else max(REACH_GLIDE_MIN, span / max(speed, 1e-3)))
                glide_t0 = time.time()
                if cmd_local is None:
                    cmd_local = start.copy()

            if glide_t0 is not None:
                u = (time.time() - glide_t0) / max(glide_s, 1e-3)
                if u < 1.0:
                    # smoothstep: zero velocity at both ends, so the arm eases
                    # out of rest and eases into the target
                    e = u * u * (3.0 - 2.0 * u)
                    cmd_local = start + (seed - start) * e
                else:
                    cmd_local = seed.copy()
                    glide_t0 = None
                    # The whole-body controller squats to follow a low wrist
                    # target and that takes a couple of seconds; hold before
                    # integrating, or the loop reads the transient as failure.
                    next_update = time.time() + REACH_SETTLE_AFTER_GLIDE

            delta = desired_local - palm_local
            err = float(np.linalg.norm(delta))
            if glide_t0 is None and time.time() >= next_update:
                step = np.clip(REACH_GAIN * delta, -REACH_MAX_STEP, REACH_MAX_STEP)
                cmd_local = np.clip(cmd_local + step, REACH_BOX_LO, REACH_BOX_HI)
                next_update = time.time() + 1.0 / REACH_UPDATE_HZ

            if side == "right":
                self.cmd.set_wrists(right_pos=cmd_local)
            else:
                self.cmd.set_wrists(left_pos=cmd_local)
            self._maintain_crouch(s)
            self.cmd.frame(mode=LocomotionMode.IDLE, height=height,
                           facing=self._facing(s))

            if err < tol and glide_t0 is None:
                ok_since = ok_since or time.time()
                if time.time() - ok_since > settle:
                    # How much the palm turned since the offset was frozen.
                    # This is the exact size of the error the freeze buys, and
                    # it is the only unexplained term left between "the reach
                    # converged" and "the object is 13 cm off the pinch axis".
                    now = self._rot(s["palms"][side]["quat"])
                    drift = (np.linalg.norm(
                        (now - palm_rot) @ palm_offset) if palm_offset is not None
                        else 0.0)
                    self.log(f"reach {side} done (err {err:.3f} m, "
                             f"cmd {np.round(cmd_local, 3)}, "
                             f"offset drift since freeze {drift:.3f} m)")
                    self.last_cmd = cmd_local
                    return True
            else:
                ok_since = None

            # Saturation: the command keeps growing but the palm has stopped
            # improving, i.e. the point is outside the arm's workspace. Say so
            # instead of burning the whole timeout.
            if self.verbose and time.time() >= next_log:
                # `want` and the pelvis height are what separate the two ways
                # this stalls. If want_z stays far below the palm the arm is
                # genuinely out of travel; if want_z climbs towards the palm as
                # the body squats, the target is being reached by the legs and
                # the arm only has to hold station. Without both numbers the
                # trace cannot tell them apart, which is how "out of the arm's
                # workspace" got printed for both.
                self.log(f"    reach err {err:.3f} cmd {np.round(cmd_local, 3)} "
                         f"palm {np.round(palm_local, 3)} "
                         f"want {np.round(desired_local, 3)} "
                         f"pelvis_z {s['pelvis']['pos'][2]:.3f}")
                next_log = time.time() + 1.0

            if err < best - REACH_PROGRESS or glide_t0 is not None:
                best, stalled_since = min(best, err), time.time()
            elif time.time() - stalled_since > REACH_STALL_S:
                self.aborted = (f"reach {side} stalled at {err:.3f} m — target "
                                f"looks out of the arm's workspace")
                self.last_cmd = cmd_local
                return False

            time.sleep(DT)

        self.aborted = f"reach {side} timed out (err {err:.3f} m)"
        self.last_cmd = cmd_local
        return False

    def servo_grasp(self, obj_key: str, side: str = "right",
                    offset=GRASP_CENTRE_OFFSET, tol: float = 0.018,
                    align_tol: float = 0.006, settle: float = 0.6,
                    timeout: float = 16.0, nudge_max: float = 0.05,
                    entry_axis: int = 1) -> bool:
        """Close the last few centimetres by servoing on the palm-frame error.

        The reach loop puts the *wrist* at a world point derived from a frozen
        palm rotation. Two separate errors then land on top of each other: the
        loop's own residual (3.7 cm measured) and whatever the wrist turned
        since the freeze (4 cm). Eight centimetres, into a hand that opens 6.1
        — so the fingers close beside the object, which is exactly what the
        pre-close logs showed.

        Neither error is interesting. The quantity that decides the grasp is
        where the object sits *in the palm's own frame*, and the privileged
        stream reports both poses, so servo on that directly. Rotation drift
        cancels: it moves the palm frame and the measurement together.

        If the hand moves by d (world), the object's palm coordinates shift by
        -d, so to take the object from q to the grip centre g the hand moves by
        R_palm @ (q - g).

        Two phases, and the split is the whole trick. Correcting across the jaw
        *while* advancing spends the clearance before the alignment converges,
        and the fingertip hits the object head-on. So: line up first, holding
        the object SERVO_SAFE_GAP out along the entry axis, and only advance
        once it is centred.

        `entry_axis` is the palm's y by default, not its x. Along x the middle
        fingertip fouls the object 10.5 cm before the grip centre reaches it,
        so a straight push shoves the object for the last tenth of a metre;
        the thumb side is clear the whole way in.
        """
        offset = np.asarray(offset, dtype=float)
        start = self.object_pos(obj_key)
        end = time.time() + timeout
        ok_since = None
        next_update = time.time()
        next_log = time.time()
        cmd_local = np.array(self.cmd.right_wrist_pos if side == "right"
                             else self.cmd.left_wrist_pos, dtype=float)
        err = float("inf")
        phase = "align"

        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while closing on the object"
                return False
            moved = float(np.linalg.norm(self.object_pos(obj_key, s) - start))
            if moved > nudge_max:
                # Name the link. "Shoved 5 cm" is the same message whether a
                # fingertip caught the object on the way in or the palm did,
                # and the two want opposite fixes.
                hit = s["objects"][obj_key].get("touch_links") or ["nothing"]
                # Toppled or shoved? A tall object that has been knocked over
                # reports a large displacement with nothing touching it, since
                # the contact ended long before it finished falling. Reading
                # that as "the hand pushed it sideways" points the fix in the
                # wrong direction, so check the object's own up-axis.
                up = self._rot(s["objects"][obj_key]["quat"])[:, 2]
                how = "toppled" if abs(float(up[2])) < 0.7 else "shoved"
                self.aborted = (f"{obj_key} {how} {moved * 100:.0f} cm while "
                                f"closing in ({phase}, touched {', '.join(hit)})")
                return False

            rot = self._rot(s["palms"][side]["quat"])
            q = rot.T @ (self.object_pos(obj_key, s) - np.array(s["palms"][side]["pos"]))
            full = q - offset
            others = [i for i in range(3) if i != entry_axis]
            lat = float(np.linalg.norm(full[others]))
            if phase == "align":
                # Hold the object at a depth where the fingertips are clear of
                # it, and correct across the jaw there. Leaving the depth to
                # whatever the approach reach happened to leave is what broke
                # the previous version: that reach lands 4-8 cm out, and the
                # fingertips lead the grip centre by 11 cm, so the hand could
                # arrive already resting on the object — and then even a purely
                # sideways correction shoves it.
                delta_palm = full.copy()
                delta_palm[entry_axis] = full[entry_axis] - SERVO_SAFE_GAP
                if lat < align_tol and abs(delta_palm[entry_axis]) < SERVO_DEPTH_TOL:
                    phase = "advance"
                    self.log(f"servo_grasp aligned ({np.round(full, 3)} in the "
                             f"palm frame); sliding in "
                             f"{abs(full[entry_axis]) * 100:.0f} cm")
            if phase == "advance":
                # Gate the advance on how well the jaw is lined up, rather than
                # committing to it once. The clearance either side of the
                # object is (aperture - width) / 2 — 5.5 mm for a 5 cm can in a
                # 6.1 cm hand — so a centimetre of drift acquired *during* the
                # 15 cm advance is enough to catch a fingertip on it, which a
                # one-shot alignment check cannot see coming. Scaling the depth
                # step by the alignment makes the hand stop and re-centre by
                # itself whenever it starts to wander.
                # SERVO_GATE_TOL, not align_tol. Gating on the same threshold
                # the alignment phase had to cross means the gate is exactly
                # shut at the moment alignment completes — the advance never
                # starts and the whole thing times out, which is what every
                # "timed out in the advance phase" was.
                gate = float(np.clip(1.0 - lat / SERVO_GATE_TOL, 0.0, 1.0))
                delta_palm = full.copy()
                delta_palm[entry_axis] = full[entry_axis] * gate
            err = float(np.linalg.norm(full if phase == "advance"
                                       else delta_palm))

            if phase == "advance" and err < tol:
                ok_since = ok_since or time.time()
                if time.time() - ok_since > settle:
                    self.log(f"servo_grasp {obj_key}: object at "
                             f"{np.round(q, 3)} in the palm frame "
                             f"(want {np.round(offset, 3)}, err {err:.3f})")
                    self.last_cmd = cmd_local
                    return True
            else:
                ok_since = None

            if time.time() >= next_update:
                # Same cadence as reach: the arm lags about a second, so a
                # faster loop winds the command up through the lag.
                # palm frame -> world -> pelvis frame, which is the frame the
                # wrist command is interpreted in
                in_root = (self._rot(s["pelvis"]["quat"]).T @ rot) @ delta_palm
                step = np.clip(SERVO_GAIN * in_root,
                               -SERVO_MAX_STEP, SERVO_MAX_STEP)
                cmd_local = np.clip(cmd_local + step, REACH_BOX_LO, REACH_BOX_HI)
                next_update = time.time() + 1.0 / REACH_UPDATE_HZ

            self.cmd.set_wrists(**{f"{side}_pos": cmd_local})
            self._maintain_crouch(s)
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))
            if self.verbose and time.time() >= next_log:
                self.log(f"    servo[{phase}] err {err:.3f} object@palm "
                         f"{np.round(q, 3)} cmd {np.round(cmd_local, 3)} "
                         f"nudged {moved * 100:.0f} cm")
                next_log = time.time() + 1.0
            time.sleep(DT)

        self.aborted = (f"servo_grasp {obj_key} timed out in the {phase} phase "
                        f"(err {err:.3f} m)")
        self.last_cmd = cmd_local
        return False

    def grasp(self, obj_key: str, side: str = "right", hold: float = 1.5,
              squeeze: float = 0.08, ramp: float = 1.6) -> bool:
        """Close until the fingers touch, then a little more. Then require it.

        Not a single jump to a fixed closed pose. That pose is sized for
        nothing in particular, and against a 4 cm can it buries the fingers
        27 mm inside it — a position-controlled hand really does drive there,
        so the object is squeezed out and lands on the floor, which the episode
        then reports as "no contact after closing". Closing progressively and
        stopping just past first contact needs no model of the object's width:
        the same loop grips a can, a bottle or a fork.

        Measured offline (teleop_free/diag_grasp_pose.py), this hand first
        opposes a 4 cm cylinder at 70% of the closed pose, with 6 mm of
        penetration; at 100% it is 27 mm.

        The stopping test is OPPOSED contact — a thumb link and a finger link
        at once — not merely "the hand is touching it". Stopping at any contact
        stops far too early: two episodes latched at 34% and 38%, where only
        the middle finger was on the object, squeezed 8% further, pushed it off
        the table, and reported a confirmed grasp on the way down.
        """
        closed = HAND_CLOSED_RIGHT if side == "right" else HAND_CLOSED_LEFT
        touched_at = None
        start = time.time()
        while True:
            t = time.time() - start
            alpha = min(t / ramp, 1.0)
            if touched_at is not None:
                alpha = min(touched_at + squeeze, 1.0)
            self.cmd.set_hands(**{side: (closed * alpha).astype(np.float32)})
            s = self.obs()
            self._maintain_crouch(s)
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))
            if touched_at is None and _opposed(s, obj_key, side):
                touched_at = alpha
                self.log(f"grasp {obj_key}: opposed contact at {alpha:.2f} of "
                         f"the closed pose; squeezing to "
                         f"{min(alpha + squeeze, 1.0):.2f}")
                start = time.time() - ramp        # freeze the ramp, start the hold
            if touched_at is not None and time.time() - start > hold:
                break
            if touched_at is None and alpha >= 1.0 and t > ramp + hold:
                break
            time.sleep(DT)

        s = self.obs()
        links = s["objects"][obj_key].get("touch_links", [])
        if not _opposed(s, obj_key, side):
            self.aborted = (f"grasp {obj_key}: closed to {alpha:.2f} without "
                            f"opposed contact (touching {', '.join(links) or 'nothing'})")
            return False
        self.log(f"grasp {obj_key} with {side} confirmed by opposed contact "
                 f"({', '.join(links)})")

        return True

    def release(self, side: str = "right", hold: float = 1.0) -> None:
        self.cmd.set_hands(**{side: HAND_OPEN})
        deadline = time.time() + hold
        while time.time() < deadline:
            self.cmd.frame(mode=LocomotionMode.IDLE)
            time.sleep(DT)
        self.log(f"release {side}")

    # -- composites --------------------------------------------------------

    def crouch(self, target_pelvis_z: float, hold_side: str = "left",
               timeout: float = 12.0, settle: float = 0.8) -> bool:
        """Lower the body until the pelvis sits at `target_pelvis_z`.

        The grasp needs the grip point *above* the pelvis — with the palm held
        horizontal, which is the only orientation this controller offers, the
        arm cannot reach below it (teleop_free/diag_reach_map.py). At a waist
        height table that is impossible standing up, and no amount of arm
        tuning changes it. Squatting moves the pelvis instead of the hand.

        The lever is the wrist target, not the planner: `height` is inert
        (measured again in teleop_free/diag_squat.py — 0.70, 0.60 and 0.50 all
        leave the pelvis at 0.536), while a low wrist target takes it from
        0.782 down to 0.536, 25 cm.

        Which is why the crouch is held with the OTHER arm. Driving it with the
        grasping arm is a positive feedback loop: once that arm rises to a grip
        point now above the pelvis, the body stands back up and takes the grip
        point out of range again. The free arm holds a low target beside the
        body and the working arm is left free to do the task — which is also
        roughly what a person does.
        """
        pos = np.array([CROUCH_HOLD_X,
                        CROUCH_HOLD_Y if hold_side == "left" else -CROUCH_HOLD_Y,
                        -0.10])
        end = time.time() + timeout
        ok_since = None
        if self.hold_yaw is None:
            self.hold_yaw = self.pelvis_yaw()
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while crouching"
                return False
            z = float(s["pelvis"]["pos"][2])
            err = z - target_pelvis_z
            if abs(err) < CROUCH_TOL:
                ok_since = ok_since or time.time()
                if time.time() - ok_since > settle:
                    self.crouch_cmd = pos.copy()
                    self.crouch_hold = (hold_side, target_pelvis_z, pos.copy())
                    self.log(f"crouch to {z:.3f} m "
                             f"(wanted {target_pelvis_z:.3f}, wrist z "
                             f"{pos[2]:+.2f})")
                    return True
            else:
                ok_since = None
                # The mapping from wrist target to pelvis height is steeply
                # non-linear — flat to -0.25, then 0.10 m of pelvis per 0.15 of
                # command — so close the loop on the pelvis instead of fitting
                # it.
                pos[2] = float(np.clip(pos[2] - np.sign(err) * CROUCH_STEP,
                                       CROUCH_WRIST_MIN, CROUCH_WRIST_MAX))
            self.cmd.set_wrists(**{f"{hold_side}_pos": pos})
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))
            time.sleep(DT)

        self.aborted = (f"crouch timed out at {z:.3f} m, wanted "
                        f"{target_pelvis_z:.3f}")
        return False

    def stand_up(self, hold_side: str = "left") -> None:
        """Release the crouch by putting the holding arm back at rest."""
        self.crouch_cmd = None
        self.crouch_hold = None
        self.hold_yaw = None
        self.cmd.set_wrists(**{f"{hold_side}_pos": TUCK_WRISTS[hold_side]})

    def _facing(self, s) -> tuple:
        yaw = self.hold_yaw if self.hold_yaw is not None else self.pelvis_yaw(s)

        return (float(np.cos(yaw)), float(np.sin(yaw)), 0.0)

    def track(self, target_pos, target_rot=None, side: str = "right",
              speed: float = TRACK_SPEED, tol: float = 0.03,
              settle: float = 0.5, extra: float = 6.0,
              obj_key: str | None = None, nudge_max: float = 0.10) -> bool:
        """Stream a wrist pose along a straight line, and let the body follow.

        This replaces reach()'s search with a trajectory. reach() treated the
        interface as a black box to be probed: seed the command from an inverse
        model, hold it still for 2.5 s so the transient dies, then nudge at 3 Hz
        and watch what happens. Eight and a half seconds of that is commanded
        waiting before the first correction, and it is most of why the videos
        show a robot freezing and then lurching.

        But SONIC is a motion TRACKER. Its training data is VR teleoperation —
        continuous, human, whole-body references — and a reference that stands
        still for two seconds and then jumps is the one thing it never saw. So
        give it what it expects: a smooth pose reference, updated every control
        tick, moving at a hand's pace from where the hand is to where it has to
        be. The body then follows a trajectory instead of chasing a setpoint.

        Two corrections make that work on this deploy:

        * Position does not track 1:1 — ask for 0.65 m ahead and the palm stops
          at 0.40. So the command carries a bias, seeded from the measured
          mapping and then trimmed continuously rather than in 3 Hz steps.
        * The bias is integrated against a DELAYED copy of the reference. The
          arm lags the command by about a second; comparing the palm to where
          the reference is *now* would read that lag as error, wind the bias up
          through the whole move and overshoot at the end. Comparing it to
          where the reference was a second ago leaves only the real error.

        Orientation gets the same treatment. The wrist channel is attenuated
        ~4.5x (diag_pitch.py), so the commanded rotation is the wanted one
        scaled up about its own axis, and then trimmed by the same lagged loop.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        s = self.obs()
        start = np.array(s["palms"][side]["pos"], dtype=float)
        span = float(np.linalg.norm(target_pos - start))
        duration = max(TRACK_MIN_S, span / max(speed, 1e-3))

        # Bias seed: what the command has to be for the palm to end up at the
        # target, according to the measured mapping. The integrator only has to
        # trim this, so it starts near zero error instead of metres away.
        end_local = self.world_to_root_full(target_pos, s)
        seed = (wrist_map.command_for(end_local) if wrist_map.available()
                else self._legacy_seed(end_local))
        bias = np.clip(seed - end_local, -TRACK_BIAS_MAX, TRACK_BIAS_MAX)

        rot_bias = np.eye(3)
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=float).reshape(3, 3)

        obj_start = self.object_pos(obj_key, s) if obj_key else None
        free_side = "left" if side == "right" else "right"
        pelvis_z0 = float(s["pelvis"]["pos"][2])
        lag = collections.deque()          # (t, reference position)
        t0 = time.time()
        end_t = t0 + duration + extra
        ok_since = None
        err = float("inf")
        next_log = t0

        while time.time() < end_t:
            now = time.time()
            s = self.obs()
            if s["fallen"]:
                self.aborted = "fell while tracking"
                return False
            if obj_key is not None:
                moved = float(np.linalg.norm(self.object_pos(obj_key, s) - obj_start))
                if moved > nudge_max:
                    hit = s["objects"][obj_key].get("touch_links") or ["nothing"]
                    self.aborted = (f"{obj_key} moved {moved * 100:.0f} cm while "
                                    f"the hand was on its way in (touched "
                                    f"{', '.join(hit)})")
                    return False

            # --- reference: smoothstep along the straight line --------------
            u = min(1.0, (now - t0) / duration)
            e = u * u * (3.0 - 2.0 * u)
            ref = start + (target_pos - start) * e

            lag.append((now, ref.copy()))
            while len(lag) > 1 and now - lag[0][0] > TRACK_LAG:
                lag.popleft()
            ref_then = lag[0][1]

            palm = np.array(s["palms"][side]["pos"], dtype=float)
            err_vec = ref_then - palm
            err = float(np.linalg.norm(target_pos - palm))
            bias = np.clip(bias + TRACK_BIAS_GAIN * DT * err_vec,
                           -TRACK_BIAS_MAX, TRACK_BIAS_MAX)

            cmd = np.clip(self.world_to_root_full(ref, s) + bias,
                          REACH_BOX_LO, REACH_BOX_HI)

            kw = {f"{side}_pos": cmd}
            if target_rot is not None:
                measured = self._rot(s["palms"][side]["quat"])
                # Trim the orientation bias against the same lagged idea of
                # where the palm should be pointing by now.
                slip = target_rot @ measured.T
                rot_bias = _rot_lerp(rot_bias, slip @ rot_bias, TRACK_ROT_GAIN * DT)
                want = rot_bias @ target_rot
                # Pre-scale about the axis of the remaining turn: the channel
                # only delivers a fraction of what it is asked for.
                kw[f"{side}_quat"] = _quat_of(
                    _rot_scale(want @ measured.T, TRACK_ROT_BOOST) @ measured)

            self.cmd.set_wrists(**kw)
            if self.crouch_hold is not None:
                self._maintain_crouch(s)
            else:
                self._hold_pelvis(s, pelvis_z0, free_side)
            self.cmd.frame(mode=LocomotionMode.IDLE, facing=self._facing(s))

            if self.verbose and now >= next_log:
                self.log(f"    track u={u:.2f} err {err:.3f} palm "
                         f"{np.round(self.world_to_root_full(palm, s), 3)} "
                         f"cmd {np.round(cmd, 3)} bias {np.round(bias, 3)} "
                         f"pelvis_z {s['pelvis']['pos'][2]:.3f}")
                next_log = now + 1.0

            if u >= 1.0 and err < tol:
                ok_since = ok_since or now
                if now - ok_since > settle:
                    self.log(f"track {side} done (err {err:.3f} m after "
                             f"{now - t0:.1f} s, span {span:.3f} m)")
                    self.last_cmd = cmd
                    return True
            else:
                ok_since = None

            time.sleep(DT)

        self.aborted = (f"track {side} did not arrive: {err:.3f} m short of "
                        f"{np.round(target_pos, 3)} after {duration + extra:.1f} s")
        self.last_cmd = cmd

        return False

    def crouch_needed(self, plan, side: str = "right", state=None):
        """Highest pelvis that still puts the wrist target inside the arm.

        `plan` carries the wrist pose outright now, so there is nothing to
        derive: the adaptive planner already turned the grip point into the
        place the wrist has to be. Walk the pelvis down from standing and stop
        at the first height where the shoulder can make it.

        Returns None when no height works, which is a stand-pose problem rather
        than a crouch problem: squatting moves the shoulder down but also back,
        so past a point it stops helping and the answer is to stand closer.
        """
        s = state or self.obs()
        here = self.pelvis_xy(s)
        yaw = self.pelvis_yaw(s)
        rot = self._rot([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        shoulder = SHOULDER_OFFSET * np.array(
            [1.0, 1.0 if side == "right" else -1.0, 1.0])
        wrist = np.asarray(plan.wrist, dtype=float)

        for pelvis_z in np.arange(STANDING_PELVIS_Z, 0.60, -0.01):
            local = rot.T @ np.array([wrist[0] - here[0], wrist[1] - here[1],
                                      wrist[2] - pelvis_z])
            if float(np.linalg.norm(local - shoulder)) <= ARM_REACH_LIMIT:
                return float(pelvis_z)

        return None

    def _hold_pelvis(self, s, target_z: float, free_side: str) -> None:
        """Keep the pelvis at `target_z` using the free hand as the lever.

        _maintain_crouch only runs when a crouch was commanded, so during a
        reach from a standing pose NOTHING regulated the pelvis — and the
        controller squats on its own whenever the arm is asked for something
        demanding. Measured on tracks with no crouch at all: 0.779 -> 0.641 m,
        14 cm of sink nobody asked for. That is not cosmetic. The wrist command
        is in the pelvis frame, so a sinking pelvis moves the target away from
        the hand, the bias integrator winds up chasing it, and the sink gets
        worse. It is the same runaway the crouch used to have, minus the
        crouch.

        The lever is the same one crouch uses: the free hand's height. Pushing
        it down makes squatting cheap for the optimizer, lifting it makes
        standing cheap, and a proportional term on the deviation holds station.
        """
        base = np.asarray(TUCK_WRISTS[free_side], dtype=float)
        rise = float(s["pelvis"]["pos"][2]) - target_z
        cmd = base.copy()
        cmd[2] = float(np.clip(base[2] - PELVIS_HOLD_GAIN * rise,
                               CROUCH_WRIST_MIN, CROUCH_WRIST_MAX))
        self.cmd.set_wrists(**{f"{free_side}_pos": cmd})

    def _maintain_crouch(self, s) -> None:
        """Keep the squat held while another loop runs. Call once per tick."""
        if self.crouch_hold is None:
            return
        side, target_z, base = self.crouch_hold
        rise = float(s["pelvis"]["pos"][2]) - target_z
        cmd = base.copy()
        # Correct BOTH directions. Pushing the holding hand down when the
        # pelvis rises makes standing up a losing trade for the optimizer;
        # clamping that correction at zero, as this did, left the other
        # direction completely uncontrolled — and the other direction is the
        # one that was breaking every grasp.
        #
        # Measured over a whole batch: the crouch lands on target (0.657 m for
        # a 0.651 m ask) and then, once the reaching arm is asked for a point
        # it cannot make, the controller squats to lengthen the arm — pelvis
        # 0.658 -> 0.439 m in one reach, 5 episodes out of 5. That sinking
        # moves the shoulder down and back, so the fixed world target recedes
        # in the pelvis frame (want_x 0.36 -> 0.64), so the command grows, so
        # it squats further. The arm's error converged to 0.14 m and then
        # climbed back to 0.31 m purely from the body falling out from under
        # it. A symmetric term turns that runaway into a regulator.
        cmd[2] = float(np.clip(base[2] - 1.5 * rise,
                               CROUCH_WRIST_MIN, CROUCH_WRIST_MAX))
        self.cmd.set_wrists(**{f"{side}_pos": cmd})

    def tuck_arms(self, hold: str | None = None) -> None:
        """Bring the wrists in close before walking anywhere.

        The default rest pose leaves the hands out where they catch table
        corners the pelvis clears by a comfortable margin — see TUCK_WRISTS.
        This is issued once; the wrist target persists until something else
        sets it.

        `hold` names an arm that is already carrying something and has its own
        pose (carry_pose): re-commanding it mid-carry would swing the load.

        The walking pose is READY_WRISTS — forearms level, hands above
        tabletop height — so the grasp never has to raise the hand at the
        table; the arms travel the whole episode in the posture the reach
        starts from.
        """
        kw = {f"{s}_pos": READY_WRISTS[s] for s in ("left", "right")
              if s != hold}
        # Never overwrite an arm that is holding a crouch — that arm's low
        # target is the only thing keeping the body down.
        if self.crouch_cmd is not None:
            kw.pop("left_pos", None)
        self.cmd.set_wrists(**kw)
        if self.cmd.stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
            if hold is None:
                self.cmd.set_upper_body(READY_UPPER)
            else:
                # Freeze the holding arm where it is (it has the object);
                # only the free arm goes to READY.
                v = upper_body_from_mj(self.obs()["joints"])
                free = "left" if hold == "right" else "right"
                col = 0 if free == "left" else 1
                for k in range(7):
                    v[3 + 2 * k + col] = READY_ARM[free][k]
                self.cmd.set_upper_body(v)

    def navigate(self, timeout: float = 60.0, stop_dist: float = NAV_ARRIVE,
                 tuck: bool = True, hold: str | None = None) -> bool:
        """Follow the planner's distance field to whichever goal it holds.

        The field is the plan: `waypoint` returns the best point one stride
        away, and steering at it is the whole controller. What is *not*
        automatic is the humanoid part — a walk command whose direction is 90
        degrees off the robot's heading makes it crab sideways, which is how
        most of the falls happened. Hence the three bands: turn in place when
        badly misaligned, walk slower while correcting, walk freely when
        aligned.
        """
        if self.planner is None or self.planner.dist is None:
            self.aborted = "navigate: no plan (set a goal first)"
            return False
        if tuck:
            self.tuck_arms(hold)

        here = self.pelvis_xy()
        if not self.planner.reachable(here):
            # One attempt at re-marching from where the robot actually is; the
            # goal set may simply have no route to it, which is a real answer
            # and much better than walking into a wall for 60 s.
            self.planner.replan(here)
        if not self.planner.reachable(here):
            self.aborted = (f"navigate: no route from {np.round(here, 2)} to "
                            f"the goal set")
            return False

        # A goal that had to be snapped is not where it was asked for, so
        # arriving at it means arriving `snap` metres short of the request.
        tol = stop_dist + self.planner.snap
        end = time.time() + timeout
        next_replan = time.time() + NAV_REPLAN_S
        next_log = time.time()
        best_d, since_progress = float("inf"), time.time()
        while time.time() < end:
            s = self.obs()
            if s["fallen"]:
                self.aborted = (f"fell while navigating at "
                                f"{np.round(self.pelvis_xy(s), 2)}")
                return False

            here = self.pelvis_xy(s)
            # Two ways the deploy's odometry can put the robot somewhere the
            # plan cannot help with. After a fall its state estimator never
            # relearns, so `facing` is interpreted in a rotated frame and the
            # robot walks confidently in the wrong direction; one run ended at
            # x = -6.97, five metres outside the map, and only then reported
            # "boxed in". Name it where it happens.
            if not self.occ.in_bounds(self.occ.cell_of(here)):
                self.aborted = (f"navigate: pelvis left the map at "
                                f"{np.round(here, 2)} — odom has probably "
                                f"diverged, restart the stack")
                return False

            d = self.planner.distance_at(here)
            if d < best_d - NAV_PROGRESS:
                best_d, since_progress = d, time.time()
            elif time.time() - since_progress > NAV_STALL_S:
                self.aborted = (f"navigate: no progress for {NAV_STALL_S:.0f}s "
                                f"at {np.round(here, 2)}, {d:.2f} m of field left")
                return False
            # (re-set below while arcing to align: the steering arc trades
            # field distance for heading on purpose, and counting that as a
            # stall killed every episode that spawned facing away)
            if d < tol:
                self.cmd.frame(mode=LocomotionMode.IDLE)
                self.log(f"navigate done ({d:.2f} m of field left, tol {tol:.2f})")
                return True
            # Re-march on a timer, and immediately whenever the field has gone
            # flat underfoot. The goal ring at a table sits right on the edge
            # of the inflation, so a robot arriving at it routinely lands one
            # cell inside — where the last wavefront never reached, because its
            # escape corridor was carved somewhere else entirely. That is a
            # stale plan, not an unreachable goal, and the first batch failed
            # every episode by treating it as the latter.
            flat = not self.planner.reachable(here)
            if flat or time.time() >= next_replan:
                self.planner.replan(here)
                next_replan = time.time() + NAV_REPLAN_S
            if flat and not self.planner.reachable(here):
                self.aborted = f"navigate: field went flat at {np.round(here, 2)}"
                return False
            if flat:
                d = self.planner.distance_at(here)
                if d < tol:
                    self.cmd.frame(mode=LocomotionMode.IDLE)
                    self.log(f"navigate done after replan ({d:.2f} m left)")
                    return True

            wp = self.planner.waypoint(here, min(NAV_STEP, max(d, 0.15)))
            if wp is None:
                self.aborted = f"navigate: boxed in at {np.round(here, 2)}"
                return False

            bearing = float(np.arctan2(*(wp - here)[::-1]))
            yaw = self.pelvis_yaw(s)
            dth = abs(wrap(bearing - yaw))
            if dth > NAV_TURN_IN_PLACE:
                gain = 0.0
            elif dth > NAV_TURN_STRAIGHT:
                gain = float(np.clip(1.0 - dth / NAV_TURN_IN_PLACE, 0.3, 1.0))
            else:
                gain = 1.0
            gain *= float(np.clip(d / NAV_EASE, 0.3, 1.0))
            # A facing goal more than ~60 deg away stalls the controller
            # outright, and a pure in-place turn barely rotates at all
            # (~6 deg/s, drifting the while) — SONIC turns properly only
            # while WALKING. So: carrot the heading half a radian ahead, and
            # when badly misaligned walk a slow steering arc along the
            # robot's own heading instead of stopping to pirouette.
            face = yaw + float(np.clip(wrap(bearing - yaw), -0.5, 0.5))
            if gain <= 0.0:
                step = np.array([np.cos(face), np.sin(face)])
                since_progress = time.time()
                best_d = min(best_d, d)
            else:
                step = (wp - here) / max(np.linalg.norm(wp - here), 1e-6)
            self.cmd.frame(
                # SLOW_WALK throughout, never WALK. Measured on a straight
                # line with nothing in the way (teleop_free/diag_speed.py),
                # WALK at the default speed falls over; SLOW_WALK covers
                # 0.166 m/s and stays upright. The follower used to switch to
                # WALK whenever it was well aligned and still far from the
                # goal, which is most of every approach.
                #
                # `gain` scales the *speed* field, not the movement vector: the
                # deploy normalises that vector's length away, so scaling it
                # was controlling nothing at all. Pace has never been under
                # this loop's control until now.
                mode=LocomotionMode.SLOW_WALK,
                speed=(NAV_SPEED_MIN if gain <= 0.0 else
                       float(np.clip(NAV_SPEED * gain, NAV_SPEED_MIN, NAV_SPEED))),
                movement=(step[0], step[1], 0.0),
                facing=(np.cos(face), np.sin(face), 0.0),
            )

            if self.verbose and time.time() >= next_log:
                self.log(f"    nav d={d:.2f} at {np.round(here, 2)} -> "
                         f"{np.round(wp, 2)} (turn {np.rad2deg(dth):.0f}deg, "
                         f"gain {gain:.2f})")
                next_log = time.time() + 2.0
            time.sleep(DT)

        self.aborted = f"navigate timed out with {d:.2f} m of field left"
        return False

    def goto_object(self, target_xy, want_dist: float = STAND_OFF,
                    timeout: float = 60.0, hold: str | None = None) -> bool:
        """Plan and walk to a standing pose the given distance from a target."""
        if self.planner is None:
            self.aborted = "goto_object: no map (Primitives built without meta)"
            return False
        here = self.pelvis_xy()
        if not self.planner.set_goal_ring(target_xy, want_dist - NAV_RING_BAND,
                                          want_dist + NAV_RING_BAND,
                                          here_xy=here):
            self.aborted = (f"goto_object: nowhere legal to stand "
                            f"{want_dist:.2f} m from {np.round(target_xy, 2)}")
            return False
        # A widened goal is a geometry problem, not a plan: it means nowhere on
        # the requested ring is legal, so the arrival tolerance absorbs the
        # widening and the robot stops short. Say what the achievable distance
        # actually is instead of retrying the same request five times.
        if self.planner.snap > 0.05:
            self.log(f"WARNING: nothing legal {want_dist:.2f} m from "
                     f"{np.round(target_xy, 2)} — ring widened by "
                     f"{self.planner.snap:.2f} m, so the closest the pelvis can "
                     f"stand is about {want_dist + self.planner.snap:.2f} m. The "
                     f"object is too far inside the surface's footprint.")
        self.log(f"plan to stand {want_dist:.2f} m from "
                 f"{np.round(target_xy, 2)}: {self.planner.distance_at(here):.2f} m "
                 f"of path, goal widened by {self.planner.snap:.2f} m")

        return self.navigate(timeout=timeout, hold=hold)

    def face_for_grasp(self, target_xy, side: str = "right") -> bool:
        """Turn so the target sits in front of the grasping hand, not the nose.

        Facing the object dead-on puts it on the pelvis's centreline, and the
        arm then has to reach across the body — which is where the integrating
        reach ran out of workspace. Offsetting the heading by GRASP_BEARING
        puts the object roughly where the hand naturally hangs.
        """
        target = np.asarray(target_xy, dtype=float)
        bearing = float(np.arctan2(*(target - self.pelvis_xy())[::-1]))
        offset = GRASP_BEARING if side == "right" else -GRASP_BEARING

        return self.turn_to(wrap(bearing + offset))

    def approach(self, target_xy, surface_xy, want_dist: float,
                 attempts: int = 3, side: str | None = None,
                 circumradius: float = 0.0, half_extent=None,
                 hold: str | None = None) -> bool:
        """Get within `want_dist` of a target on a surface, facing it.

        `surface_xy`, `circumradius` and `half_extent` are accepted for
        compatibility and no longer used: the obstacle geometry now lives in
        the map, which knows about *both* surfaces and the walls rather than
        the one surface the caller happened to name.

        The retry loop stays, though, because it was never about the path. The
        robot keeps drifting while it turns — the deploy is still finishing the
        step it had already committed to — so it can end a clean walk 0.5 m
        from the object and be 0.9 m away by the time it faces it. Re-planning
        to a tighter ring and walking the last of it is what converges.
        """
        target = np.asarray(target_xy, dtype=float)
        offset = min(STAND_OFF, want_dist * 0.9)
        d = float("inf")
        for i in range(attempts):
            if not self.goto_object(target, offset, hold=hold):
                return False
            # A turn that times out used to be ignored, which left the robot
            # sideways-on with the object 0.5 m off its centreline and the arm
            # unable to reach it — the failure surfaced two stages later as a
            # workspace error. Treat it as the approach failing, here.
            turned = (self.face_for_grasp(target, side) if side
                      else self.turn_to(
                          float(np.arctan2(*(target - self.pelvis_xy())[::-1]))))
            if not turned:
                self.log(f"approach attempt {i + 1}: {self.aborted}")
                continue

            s = self.obs()
            d = float(np.linalg.norm(target - self.pelvis_xy(s)))
            local = self.world_to_root(np.append(target, s["pelvis"]["pos"][2]), s)
            self.log(f"approach attempt {i + 1}: {d:.3f} m from target, "
                     f"pelvis-frame xy {np.round(local[:2], 3)}")
            # Distance alone is not enough. The robot drifts sideways through
            # the turn as much as forwards, so it can end up the right distance
            # away with the object nearly beside it, which no arm extension can
            # fix. Bound the lateral offset too.
            if d < want_dist and (side is None or abs(local[1]) < APPROACH_LATERAL_MAX):
                return True

            # Aim closer next time, floored so the ring never collapses inside
            # the table's own inflation — where it would have nowhere legal to
            # stand and goto_object would fail outright rather than get closer.
            offset = max(MIN_STAND_OFF, offset - (d - want_dist * 0.92))
            self.log(f"  re-planning for a stand-off of {offset:.2f} m")

        self.aborted = f"approach did not converge: {d:.2f} m > {want_dist}"
        return False

    def carry_pose(self, side: str = "right") -> None:
        """Tuck the loaded arm in front of the chest before walking.

        An arm left extended is both a collision risk and, at walking speed, a
        large disturbance torque on a balancing humanoid.
        """
        pos = np.array([0.28, -0.12 if side == "right" else 0.12, 0.05])
        if side == "right":
            self.cmd.set_wrists(right_pos=pos)
        else:
            self.cmd.set_wrists(left_pos=pos)

    def close(self) -> None:
        self.cmd.close()
        self.state.close()


__all__ = [
    "Primitives", "CONTROL_HZ", "DT", "STAND_OFF", "PREGRASP_ABOVE",
    "GRASP_ABOVE", "LIFT_HEIGHT", "HAND_CLOSED_LEFT", "HAND_CLOSED_RIGHT",
    "LocomotionMode", "StreamMode", "yaw_of", "wrap",
]
