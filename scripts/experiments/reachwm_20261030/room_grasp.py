"""Whole-robot grasps in the grasp room (2026-11-09): realistic objects (simready_objects.py) on the six stations of
grasp_room.py, ScaleBFM on leg odometry, one look at the object before the walk (sim truth of the object pose), then nothing
about the pelvis from the simulator.

An episode: start 0.45-0.90 m back from the stance, facing the station within 8 deg; a validated grasp of the object's
grasp_cache.npz inside the wrist range the self model was trained on (GRASP RANGE below); the open walk to the stance; a
waypoint CLEAR_OUT outside the furniture front; the pre-grasp PRE_BACK behind the grasp along the jaw's own axis; the entry at ENTRY_SPEED; close; lift LIFT_M; hold HOLD_S. Success:
both fingers on the object at the end of the hold and the object risen more than 2 cm. HOLD_S is the 1.5 s of grasp_cache.py's
validation; held_3s also records 3 s of hold, where MuJoCo's friction creep (about 0.4 cm/s on round objects) has started to tell.

  direct  ScaleBFM alone: the goal as the command, the walk taken as exact
  feedback (2026-11-10) no learned model: geometric leg odometry (leg_odo.py) gives where the robot stopped, the goal is the
          command, and the PRE_FIX / ENTRY_FIX joint-reading corrections (and GRIP_CHECK) apply; PICK needs the model, so the rule's grasp
  model   the walk model on the body after the walk (post_wkp) gives where the robot really stopped, the reach model's
          inversion gives the command of the pre-grasp and of the entry; the lift is the plain command

Result 2026-11-09 (tag _v2, 147 placements, 128 objects, same action for both): direct 31 of 147 (21 %), model 67 of 147 (46 %), the
model alone 44, direct alone 8. Jaw error at the grip, median: 4.99 and 2.75 cm. Success of model grips by that error: under 1 cm
100 %, 1-2 78 %, 2-3 57 %, 3-5 19 %, 5 and more 0 %. The stance rules and the lift height were tuned on layout 0.

With the room's own reaches in the reach model's data (tag _v3, gen_reach.py REACH_GRASP_ROOM, APPROACH_WM_POST=$I/invert/wrist_graspr_odo/wm_ref.pt):
model 73 of 147 (50 %), jaw error at the grip 1.91 cm (p90 7.63, was 8.09), paired -0.57 cm against _v2 (95 % -0.85..-0.45). The success count is
inside the noise (22 against 16, p 0.42). The tail is commands ScaleBFM cannot deliver: entry correction 6 cm and more succeeds 2 of 22, and the
ensemble spread at the entry ranks the jaw error with Spearman 0.69, so it is a choice-of-grasp problem, not a data problem.

Switches added 11-09 night, all off by default (off = bit-identical to _v3; with RETRY the first try is bit-identical to the run without it):

  ROOM_GRASP_PICK=8 ROOM_GRASP_PICK_ATT=1.0  at the waypoint, invert up to 8 distinct grasps, score spread + predicted wrist miss,
                                             switch when the rule's grasp scores above PICK_GATE 1.5 and the best saves PICK_MARGIN 0.3
  ROOM_GRASP_PRE_FIX=2 ROOM_GRASP_ENTRY_FIX=2 add the jaw's measured miss to the command, at the pre-grasp and at the end of the entry
  ROOM_GRASP_RETRY=2 ROOM_GRASP_RETRY_MIN_Z=0.5  the jaw reading under RETRY_OPEN (3 mm) after the grip or the hold = nothing held:
                                             open, back out RETREAT_M along the jaw, waypoint, look at the object again, try again
                                             (the failed grasp is banned if the object moved under 2 cm); only on surfaces >= MIN_Z
  rejected (kept for the record): ATT_FIX, ENTRY_MID, ENTRY_DIRECT, PICK_ENTRY, PICK_WIDTH, PICK_ALL, REPICK (moving to another
  pre-grasp when the entry's spread there is high: 108 -> 99 first tries, 8 falls), GRIP_SETTLE 0.3 + LIFT_SPEED 0.05 (109 vs 108:
  the drops are the wrist attitude, not the lift), a model retrained on the candidates' commands

Switches added 11-10 (found from per-tick traces of the 97 held-out failures, scratch trace.py):

  ROOM_GRASP_CONT=1 ROOM_GRASP_LIFT_CONT=1   command continuity: a correction and the lift start at the previous command (position
                                             and attitude), the lift also plans from the reference pose. Motor.reach starts a path
                                             at the measured jaw, and ScaleBFM holds the jaw a few cm and 10-15 deg off its command,
                                             so each new reach pulled the hand back first (recoil up to 5 cm beside the object; the
                                             lift's jaw peaked at 45-380 cm/s, now 16-20). Development, two start seeds: 206 -> 218
                                             of 294 (+20 -8 paired). Held out: first try 197 -> 205 (+21 -13), with RETRY above
                                             0.5 m 205 -> 212, falls 3 -> 2. CONT=2/3 (the entry too) add nothing.
  rejected 11-10: wrist attitude by joint-reading feedback (ATT_FIX with ENTRY_CARRY, ATT_FIX_MIN 12, ATT_BACK 0.08): it rescues
  10 of 15 drops (the drops are a pitch miss about the closing axis, median -11 deg against -3 on held grasps), but a turned wrist
  command moves the jaw 3-10 cm and its offset does not carry 8 cm (arrival 4.9 cm), so as many grasps break: 108, 109, 98 against
  112. ENTRY_CARRY alone 108. ROOM_GRASP_STANCE_BACK_LOW 0.47 / 0.52: 102 / worse (low surfaces 23 -> 13 -> 7 of 35).

  ROOM_GRASP_GRIP_CHECK=1 (11-10, later)     before the jaw closes, grasp_cache.py's hand-only model closes and lifts at the pose the
                                             jaw has reached; if it loses the object, the nearest jaw shift at which it holds is
                                             commanded. Development (two seeds, partial): +7 -0. Held out with CONT, LIFT_CONT and
                                             RETRY: first try 205 -> 206 (+3 -2), with retry 212 -> 215 (+5 -2), falls 2.
  rejected 11-10, later: choosing grasps by grasp_robust.py's tolerance score (ROBUST_MIN 0.67, ROBUST 0.5, PICK_ROB 2). End pose
  only (grasp_robust.npz): 111 and 105 against 112 and 106, drops down, knock-aways up. With the measured entry replayed
  (grasp_robust2.npz): 113 against 112. The score separates outcomes offline (success 48 / 63 / 81 / 85 % by quarter) but
  switching grasps by it does not pay.

Feedback controls, 11-10 evening (the "is the learned model needed" check; tags fb*, fs*, ms*, mnp*):

  --methods feedback                         no learned model: leg_odo.py's geometric odometry over the walk, plain commands, PRE_FIX /
                                             ENTRY_FIX corrections after arrival. Held out first try 149, with retry 160 (7 falls, 73 knocks).
  ROOM_GRASP_FB_SERVO=1 (feedback arm)       Motor.reach's continuous jaw-centre + attitude integral servo instead (PRE_FIX=ENTRY_FIX=0). It
                                             reads the true jaw pose in the world; ROOM_GRASP_SERVO_SRC=odo below is the robot's version.
                                             Held out first try 220, with retry 233 (5 falls, 15 knocks, 21 drops), attitude 1.1 deg median.
                                             Development 104 and 99. So it matches or beats the reach inversion (205 / 212, dev 112 / 106).
  ROOM_GRASP_SERVO=1 (model arm)             walk model + PICK, reaches servoed to the goal, no reach inversion. Development 118 and 114,
                                             held out first try 234, with retry 239 (7 falls), at grip 0.92 cm, attitude 0.69 deg. Against the
                                             servo alone: dev +51 -22, held out first try +35 -21 (p 0.08), with retry +25 -19.
  model arm with PICK=0 (inversion only): development 108 and 99.
  ROOM_GRASP_SERVO_SRC=odo (11-10 night)     the servo measures the jaw from joint readings + leg odometry (leg_odo.JawFeed: pelvis anchored
                                             where the goals are after the walk, load-weighted feet, heading rate of the forward axis, the
                                             wrist by forward kinematics). Jaw estimate off by 0.5 cm / 0.3 deg median. Held out with retry:
                                             servo alone 220 (true pose 233; paired +9 -22), walk model + PICK + servo 235 (239; +10 -14).
                                             Development 199 and 225 of 294 (203 and 232). The arm IK still reads the true pelvis.

Held out, two sets of 7 new layouts (GRASP_ROOM_SEED 9200000 and 9300000), 294 episodes, first try: direct 56 (19 %), _v3 161 (55 %),
PICK+FIXes 197 (67 %), jaw error 1.12 cm (p90 3.47). With RETRY=2: 205 above 0.5 m only (3 falls, as without), 210 everywhere (10 falls,
the 7 new ones at the tray and the low cabinet). Direct with the same retry: 65.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python; I=$PWD/outputs/reachwm_20261030
    export BFM_ROOT=odo OMP_NUM_THREADS=1 GRASP_ROOM_OBJECTS=$I/simready GRASP_ROOM_OUT=$I/grasp_room_sr GRASP_ROOM_MIN_HELD=10
    export APPROACH_WM_POST=$I/invert/wrist_odo/wm_ref.pt WALK_DATA=approach_odo,approach0_odo WALK_SUFFIX=_odo
    $PY scripts/experiments/reachwm_20261030/room_grasp.py h --procs 6 --layouts 0,1,2,3,4,5,6 --tag _v1
    $PY scripts/experiments/reachwm_20261030/room_grasp.py d --tag _v1
"""

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("REACHWM_APPROACH", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import approach_eval as AE  # noqa: E402
import grasp_room as ROOM  # noqa: E402
import numpy as np  # noqa: E402

E, IE, PI, WI = AE.E, AE.IE, AE.PI, AE.WI
OUT = ROOM.OUT / "grasp"
SEED = int(os.environ.get("ROOM_GRASP_SEED", 6200000))  # another value draws other start points on the same placements
# Off by default (2026-11-09, after _v3). ROOM_GRASP_PICK=k: after the waypoint the reach model compares up to k grasps of the object by its
# spread on the pre-grasp and leaves the rule's grasp when that spread is above ROOM_GRASP_PICK_GATE cm. ROOM_GRASP_PRE_FIX=n: up to n
# corrections at the pre-grasp, the jaw's miss from its own joints added to the command, in free space, before the entry.
PICK_K, PICK_GATE = int(os.environ.get("ROOM_GRASP_PICK", 0)), float(os.environ.get("ROOM_GRASP_PICK_GATE", 1.5))
PICK_MARGIN = float(os.environ.get("ROOM_GRASP_PICK_MARGIN", 0.3))  # cm of spread a new grasp has to save
PRE_FIX, FIX_MIN = int(os.environ.get("ROOM_GRASP_PRE_FIX", 0)), float(os.environ.get("ROOM_GRASP_FIX_MIN", 0.008))
ENTRY_FIX = int(
    os.environ.get("ROOM_GRASP_ENTRY_FIX", 0)
)  # up to n such corrections at the end of the entry, at the entry's speed
ATT_FIX = (
    int(os.environ.get("ROOM_GRASP_ATT_FIX", 0)) == 1
)  # the pre-grasp's corrections also turn the wrist command by the jaw's measured rotation miss
ATT_FIX_MAX = math.radians(
    float(os.environ.get("ROOM_GRASP_ATT_FIX_MAX", 30.0))
)  # the command stays within this of the wanted rotation
# ROOM_GRASP_ATT_FIX_MIN (deg, 2026-11-10): the wrist command is turned only while the jaw's rotation miss is above this; the
# drops sit above 12 deg, and turning every wrist also moved grasps that already held.
ATT_FIX_MIN = math.radians(float(os.environ.get("ROOM_GRASP_ATT_FIX_MIN", 5.0)))
# ROOM_GRASP_ATT_BACK=d (m, 2026-11-10): the wrist command is turned d further back along the approach, not at the pre-grasp:
# a turned wrist command moves the jaw 3-10 cm before the position corrections catch up (pre_fix 0.7 -> 3.3 -> 1.3 cm), and at
# the pre-grasp the fingertips are a few cm from the object. The hand backs off, turns and corrects there, comes back by the
# same command step, then the pre-grasp's corrections are position only.
ATT_BACK = float(os.environ.get("ROOM_GRASP_ATT_BACK", 0.0))
ATT_N = int(os.environ.get("ROOM_GRASP_ATT_N", 3))  # corrections back there (they turn while the miss is above 5 deg)
ENTRY_MID = float(os.environ.get("ROOM_GRASP_ENTRY_MID", 0.0))
# 2026-11-10, grasp_robust.py: rob = the share of the poses ScaleBFM really reaches (development runs' errors) at which the
# hand-only validator still closes and lifts this grasp. ROOM_GRASP_ROBUST=w adds w * (rob - 1) to the rule's score,
# ROOM_GRASP_ROBUST_MIN=r takes grasps under r only when no other is in range, ROOM_GRASP_PICK_ROB=c adds c * (1 - rob) cm to
# the pick's spread score.
ROBUST_W, ROBUST_MIN = (
    float(os.environ.get("ROOM_GRASP_ROBUST", 0.0)),
    float(os.environ.get("ROOM_GRASP_ROBUST_MIN", 0.0)),
)
PICK_ROB = float(os.environ.get("ROOM_GRASP_PICK_ROB", 0.0))
ROB_OF: dict = {}  # id(cache tool array) -> rob (n, 2), set by load_cache
# ROOM_GRASP_GRIP_CHECK=1 (2026-11-10): before the jaw closes, the hand-only model of grasp_cache.py (grasp_robust.run_path)
# closes and lifts at the pose the jaw has really reached, by its own joints: its miss from the commanded grasp in position and
# rotation, on the object as it was seen. If that loses the object, the nearest jaw shift within GRIP_GRID (jaw frame, the
# attitude as it is) at which the model holds is commanded as one more correction. Development traces: the model lifts 100 of
# 112 held grasps and loses 10 of 15 dropped ones at the reached pose, and finds a 1 cm shift for 7 of those 10 (mostly 1 cm
# lower: the wrist arrives pitched up); 1 held grasp would be shifted.
GRIP_CHECK = int(os.environ.get("ROOM_GRASP_GRIP_CHECK", 0))
HANDS: dict = {}  # object id -> grasp_cache.HandModel
GRIP_GRID_N = int(
    os.environ.get("ROOM_GRASP_GRIP_GRID_N", 32)
)  # the nearest shifts tried (the 74 of the full grid took minutes on some objects)


def grip_check(slot: dict, tool_b, U, ep, Re):
    """(the model holds at the reached pose, the shift to command or None). ep, Re: the jaw's miss in the wanted tool frame."""
    import itertools

    import grasp_robust as GRB

    if slot["id"] not in HANDS:
        HANDS.clear()  # one hand model at a time: a worker that kept every object's model was killed for memory
        HANDS[slot["id"]] = GRB.GC.HandModel(str(ROOM.GN / "scenes" / f"obj_{slot['id']:03d}" / "scene_locomani.xml"))
    hm = HANDS[slot["id"]]

    def holds(d):
        return GRB.run_path(hm, tool_b, U, np.array([ep, ep + Re @ d]), [Re, Re])["ok"]

    if holds(np.zeros(3)):
        return True, None
    grid = sorted(
        (
            np.array(d)
            for d in itertools.product((-0.02, -0.01, 0, 0.01, 0.02), (-0.01, 0, 0.01), (-0.02, -0.01, 0, 0.01, 0.02))
            if any(d)
        ),
        key=np.linalg.norm,
    )
    return False, next((d for d in grid[:GRIP_GRID_N] if holds(d)), None)


def tiers():
    """(strict pitch band, least rob) in the order the choice falls back."""
    return (
        ((True, ROBUST_MIN), (True, 0.0), (False, ROBUST_MIN), (False, 0.0))
        if ROBUST_MIN > 0
        else ((True, 0.0), (False, 0.0))
    )


REPICK, REPICK_GATE = (
    int(os.environ.get("ROOM_GRASP_REPICK", 0)),
    float(os.environ.get("ROOM_GRASP_REPICK_GATE", 1.5)),
)  # at the pre-grasp: moves to another candidate while the entry's spread is above the gate
# ROOM_GRASP_CONT (2026-11-10): Motor.reach starts a path at the measured jaw pose, but ScaleBFM holds the jaw a few cm and 10-15 deg off
# its command, so each new reach first pulls the command back to the measured pose and the hand recoils beside the object.
# 1: a correction's path starts at the previous command (position and attitude). 2: also the entry's attitude starts at the
# commanded one. 3: also the entry's position starts at the previous command (not half-way).
CONT = int(os.environ.get("ROOM_GRASP_CONT", 0))
# ROOM_GRASP_LIFT_CONT=1: the lift is the last command raised by the lift height, its path from that command and its attitude, its
# whole-body IK from the reference pose (the lift was the one reach left on the measured start and the waist-forward seed: the
# jaw peaked at 45-380 cm/s in its first 0.6 s and threw light objects).
LIFT_CONT = int(os.environ.get("ROOM_GRASP_LIFT_CONT", 0))
# ROOM_GRASP_ENTRY_CARRY=g (off below 0): the entry's command is the grasp plus the offset the pre-grasp's command ended with
# (its inversion and its corrections), not a new inversion, when the entry's spread is above g cm (0: always; -2: only when a
# correction has turned the wrist command, which the reach model was not trained on). The entry is a
# 10 cm move from a corrected pose; the model's own entry shifts reach 12-23 cm at the low cabinet, where it extrapolates.
ENTRY_CARRY = float(os.environ.get("ROOM_GRASP_ENTRY_CARRY", -1.0))
RETRY = int(os.environ.get("ROOM_GRASP_RETRY", 0))  # further tries after the jaw reading says nothing is held
RETRY_OPEN, RETREAT_M = (
    float(os.environ.get("ROOM_GRASP_RETRY_OPEN", 0.003)),
    float(os.environ.get("ROOM_GRASP_RETREAT", 0.13)),
)
RETRY_MIN_Z = float(
    os.environ.get("ROOM_GRASP_RETRY_MIN_Z", 0.0)
)  # 0.5: no retry at the tray and the low cabinet, where retries fell
GRIP_SETTLE, LIFT_SPEED = (
    float(os.environ.get("ROOM_GRASP_GRIP_SETTLE", 0.0)),
    float(os.environ.get("ROOM_GRASP_LIFT_SPEED", 0.0)),
)  # m before the grasp: stop there, correct (ROOM_GRASP_ENTRY_FIX), then go on
PICK_WIDTH = float(
    os.environ.get("ROOM_GRASP_PICK_WIDTH", 0.0)
)  # cm of score per cm of grasp width above 7 cm (Dex1 opens 9.5 cm)
PICK_ALL = int(os.environ.get("ROOM_GRASP_PICK_ALL", 0)) == 1  # candidates also from outside the height's pitch band
PICK_ENTRY = float(
    os.environ.get("ROOM_GRASP_PICK_ENTRY", 0.0)
)  # weight of the entry's own spread in the score (evaluated from the waypoint)
ENTRY_DIRECT = float(
    os.environ.get("ROOM_GRASP_ENTRY_DIRECT", 0.0)
)  # cm: above this spread the entry takes the plain command, not the inverted one
PICK_ATT = float(
    os.environ.get("ROOM_GRASP_PICK_ATT", 0.0)
)  # cm of score per 10 deg of predicted wrist miss above 8 deg
STANCE_BACK, HAND_Y, ENTRY_SPEED, PRE_BACK, LIFT_M, RISE_M, HOLD_S = 0.30, 0.18, 0.08, 0.10, 0.10, 0.02, 1.5
ENTRY_SPEED = float(os.environ.get("ROOM_GRASP_ENTRY_SPEED", ENTRY_SPEED))
# the wrist commands the reach model has seen: heading offset within 60 deg (or nearly straight down), down 0-90 deg, no roll
MAX_AZ, MAX_ROLL = 60.0, 15.0
STANCE_BACK_LOW, STANCE_LOW_Z = 0.42, 0.50  # surfaces lower than this: stand this far from the front
STANCE_BACK_LOW = float(
    os.environ.get("ROOM_GRASP_STANCE_BACK_LOW", STANCE_BACK_LOW)
)  # 2026-11-10: knees met the low cabinet in the entry
STANCE_BACK_TRAY = 0.22  # the tray is 8 cm high, knees clear it; the hand reaches the floor band only 0.3-0.5 m ahead (select_eval map)
LIFT_CABINET = 0.06  # a 10 cm lift meets the board above (grasp_pilot v1 too)
CLEAR_OUT = 0.10  # the first waypoint this far outside the furniture front, m
WRIST_CEILING = 1.22  # ScaleBFM's wrist tops out near 1.26 m at the cabinet
METHODS = ("direct", "model", "feedback")
SERVO = int(
    os.environ.get("ROOM_GRASP_SERVO", 0)
)  # the model arm without the reach inversion: walk model + grasp choice, reaches servoed to the goal
FB_SERVO = int(os.environ.get("ROOM_GRASP_FB_SERVO", 0))
SERVO_SRC = os.environ.get(
    "ROOM_GRASP_SERVO_SRC", "true"
)  # "odo": the servo measures the jaw from joints + leg odometry  # the feedback arm's reaches with Motor.reach's continuous jaw-centre + attitude integral servo (it reads the true jaw in the world, a little more than joints give)
FB = (
    "model",
    "feedback",
)  # the arms that use the joint-reading corrections (2026-11-10: "feedback" has no learned model)


def quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def station_frame(name: str):
    st = ROOM.by_name(name)
    c, a = ROOM.station_pose(st)
    return st, c, a, np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def surfaces_above(st: dict, surf: float) -> float:
    tops = sorted(st["tops"])
    return next((t for t in tops if t > surf + 1e-6), 1.22 if st["name"] == "cabinet" else 9.0)


def pitch_band(z: float) -> tuple[float, float]:
    """The wrist pitch (deg, down) by the grasp's height, from the reach data's landing errors and the wrist's height limit:
    level high up, pointing down near the floor."""
    return (-5.0, 35.0) if z >= 0.80 else (-5.0, 65.0) if z >= 0.55 else (-5.0, 95.0) if z >= 0.30 else (40.0, 95.0)


def grasp_options(
    slot: dict, tool_b, R_b, score, held, p_o: np.ndarray, R_o: np.ndarray, strict: bool, rmin: float = 0.0
):
    """Every validated grasp inside the trained wrist commands and clear of the station: (index, info, tool position, tool rotation)."""
    st, _, a0, _, _ = station_frame(slot["station"])
    Rs = E.rot_z(a0).T
    above = surfaces_above(st, slot["surf"]) if st["name"] == "cabinet" else 9.0
    rob_ = ROB_OF.get(id(tool_b))
    for i in np.flatnonzero(held):
        Rt = R_o @ R_b[i]
        Rr = Rs @ Rt
        x, y = Rr[:, 0], Rr[:, 1]
        yaw = math.degrees(math.atan2(-y[0], y[1]))
        flip = abs(yaw) > 90
        if flip:
            yaw = yaw - math.copysign(180.0, yaw)
        pitch, roll = math.degrees(math.asin(np.clip(-x[2], -1, 1))), math.degrees(math.asin(min(1.0, abs(y[2]))))
        if roll > MAX_ROLL or abs(yaw) > MAX_AZ or not -5 <= pitch <= 95:
            continue
        rob = float(rob_[i, int(flip)]) if rob_ is not None else 1.0
        if rob < rmin:
            continue
        tool = p_o + R_o @ tool_b[i]
        if strict and not pitch_band(float(tool[2]))[0] <= pitch <= pitch_band(float(tool[2]))[1]:
            continue
        if x[0] < -0.2:  # the jaw would come from behind the object, over the furniture
            continue
        wrist, wrist_pre = tool - Rt[:, 0] * 0.1515, tool - Rt[:, 0] * (0.1515 + PRE_BACK)
        lo, hi = min(wrist[2], wrist_pre[2]), max(wrist[2], wrist_pre[2])
        if lo < slot["surf"] + 0.035 or hi > WRIST_CEILING or hi > above - 0.07:
            continue
        # the reach model corrects the position only; the wrist's attitude error grows with the pitch (30-38 deg near 90)
        s = float(score[i]) - 0.2 * abs(yaw) / MAX_AZ - 0.2 * roll / MAX_ROLL - 0.3 * max(0.0, pitch - 65.0) / 30.0
        s += ROBUST_W * (rob - 1.0)
        yield (
            int(i),
            dict(s=s, az=yaw, pitch=pitch, roll=roll, flip=flip, strict=strict, rob=rob),
            tool,
            Rt @ np.diag([1.0, -1.0, -1.0]) if flip else Rt,
        )


def choose_grasp(slot: dict, tool_b, R_b, score, held, p_o: np.ndarray, R_o: np.ndarray):
    """The best-scored validated grasp inside the trained wrist commands and clear of the station; (index, flip, angles) or None.

    The reach data commands R = Rz(a) Ry(b), a within 60 deg, b 0-90: the closing axis y stays level at heading a, for any b,
    so the yaw is read from y (the jaw is symmetric: y and -y are the same grasp, the sign nearer the heading is used)."""
    for strict, rmin in tiers():
        best, info = None, None
        for i, inf, _, _ in grasp_options(slot, tool_b, R_b, score, held, p_o, R_o, strict, rmin):
            if best is None or inf["s"] > info["s"]:
                best, info = i, inf
        if best is not None:
            return best, info
    return None, None


def grasp_candidates(slot: dict, g: dict, p_o: np.ndarray, R_o: np.ndarray, k: int) -> list:
    """The k best-scored grasps of choose_grasp's tier that differ by 1 cm or 15 deg: near-duplicates would only repeat the same spread."""
    for strict, rmin in tiers() if not PICK_ALL else ((False, 0.0),):
        out = sorted(
            grasp_options(slot, g["tool"], g["R"], g["score"], g["held"], p_o, R_o, strict, rmin),
            key=lambda c: -c[1]["s"],
        )
        if out:
            keep: list = []
            for c in out:
                if all(
                    np.linalg.norm(c[2] - d[2]) > 0.01 or np.trace(c[3].T @ d[3]) < 1 + 2 * math.cos(math.radians(15))
                    for d in keep
                ):
                    keep.append(c)
                if len(keep) == k:
                    break
            return keep
    return []


def x0_now(rec, sd: float, walked: bool, start: np.ndarray, chain: int) -> np.ndarray:
    """AE.x0_here with the chain flag (the entry is the second reach of a chain)."""
    import data as DT

    seg = dict(S=np.array(rec.S), W=np.array(rec.W), J=np.array(rec.J), C=np.array(rec.C), k0=len(rec.S) - 1, side=sd,
               target=np.zeros(3, np.float32), dyaw=0.0, pitch=0.0, rot6=np.zeros(6, np.float32), pelvis_z=-1.0, start=start,
               walked=walked, chain=chain, fell=False, ok=True, jaw_err_cm=-1.0, att_err_deg=-1.0, ik_err_cm=-1.0, reach_s=0.0, reset=0)  # fmt: skip
    D = IE.pack([seg])
    D.update(ref_q=np.zeros((1, 29), np.float32), ref_pel=np.r_[0, 0, 0, 1, 0, 0, 0][None].astype(np.float32),
             ref_s=np.zeros(1, np.float32), walk=np.zeros((1, 3), np.float32), target_b=D["target"])  # fmt: skip
    DT.derive(D)
    return D["x0"][0, :206]


def state_now(env, rec, side: str, goal_w: np.ndarray, rot_w: np.ndarray, chain: int) -> dict:
    pv, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    Rh = E.rot_z(yaw)
    g = Rh.T @ np.r_[goal_w[:2] - pv[:2], 0.0]
    goal = np.r_[g[:2], goal_w[2]].astype(np.float32).astype(np.float64)
    rot_h = Rh.T @ rot_w
    sd = 1.0 if side == "right" else -1.0
    start = np.r_[pv, yaw].astype(np.float32)
    return dict(sd=sd, p=pv, yaw=yaw, Rh=Rh, goal=goal, rot_w=rot_w, rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32),
                c2=dict(side=side, dyaw=float(math.atan2(rot_h[1, 0], rot_h[0, 0])), pitch=0.0), start=start, walked=True,
                x0b=x0_now(rec, sd, True, start, chain))  # fmt: skip


class Contacts:
    """Ticks with a robot-furniture contact, per phase."""

    def __init__(self, env):
        m, sm = env.m, env.sim.m
        self.env, self.phase, self.n, self.parts = env, "walk", {}, {}
        root = sm.body_rootid[sm.body("pelvis").id]
        self.robot = np.array([sm.body_rootid[b] == root for b in range(sm.nbody)])
        self.furn = np.array([sm.body(int(sm.geom_bodyid[g])).name.startswith("st_") for g in range(sm.ngeom)])
        orig = m.tick

        def tick():
            ok = orig()
            d, s = env.sim.d, env.sim.m
            for i in range(d.ncon):
                g1, g2 = d.contact[i].geom1, d.contact[i].geom2
                ga = g2 if self.furn[g1] else g1 if self.furn[g2] else None
                if ga is not None and self.robot[s.geom_bodyid[ga]]:
                    self.n[self.phase] = self.n.get(self.phase, 0) + 1
                    key = s.body(int(s.geom_bodyid[ga])).name
                    self.parts[key] = self.parts.get(key, 0) + 1
                    break
            return ok

        m.tick = tick


def brief(r: dict) -> dict:
    """What a try leaves in the record; the top level of the episode's record is the last try."""
    keys = "success grip_held rise_cm rise_15_cm at_grip_err att_deg pushed_cm opening_post_cm jaw_end_cm fell grasp".split()
    return {k_: r.get(k_) for k_ in keys if k_ in r}


def episode(
    env, rec, con, models: dict, layout: dict, k: int, j: int, method: str, cache: dict, odo=None, jf=None
) -> dict:
    import gen_reach as GR
    import walk_model as WK

    m, sim = env.m, env.sim
    slot = layout["slots"][j]
    res = dict(
        layout=k,
        slot=j,
        method=method,
        object=slot["object"],
        station=slot["station"],
        si=slot["si"],
        surf=slot["surf"],
    )
    rng = np.random.default_rng(SEED + 100 * k + j)
    st, c, a, ax, perp = station_frame(slot["station"])
    p_slot = np.array(slot["pos"])
    lat = float(perp @ (p_slot[:2] - c))
    side, sd = ("right", 1.0) if lat < 0 else ("left", -1.0)
    # a crouch brings the knees and torso forward: 0.30 m put them into the solid low cabinet and the cabinet's plinth
    back = STANCE_BACK_TRAY if st["name"] == "tray" else STANCE_BACK_LOW if slot["surf"] < STANCE_LOW_Z else STANCE_BACK
    stance = c - ax * back + perp * (lat + sd * HAND_Y)
    p_start = stance - ax * float(rng.uniform(0.45, 0.90)) + perp * float(rng.uniform(-0.10, 0.10))
    yaw_start = a + math.radians(float(rng.uniform(-8.0, 8.0)))
    con.n, con.phase, con.parts = {}, "walk", {}
    m.reset((float(p_start[0]), float(p_start[1])), yaw_start)
    for s in layout["slots"]:
        m.place(s["object"], np.array(s["pos"]), np.array(s["quat"]))
    rec.start(side)
    m.hold(1.0)
    if m.fell_at is not None:
        return dict(res, fell="settle", success=False)
    b = sim.m.body(slot["object"]).id
    p_o, R_o = sim.d.xpos[b].copy(), quat_mat(sim.d.xquat[b])
    z0_others = {s["object"]: sim.d.xpos[sim.m.body(s["object"]).id].copy() for s in layout["slots"]}
    g = cache[slot["object"]]
    i, info = choose_grasp(slot, g["tool"], g["R"], g["score"], g["held"], p_o, R_o)
    if i is None:
        return dict(res, declined="no grasp in range", success=False)
    tool_w, R_w = p_o + R_o @ g["tool"][i], R_o @ g["R"][i]
    if info["flip"]:
        R_w = R_w @ np.diag([1.0, -1.0, -1.0])  # the same grasp with the fingers swapped
    res.update(grasp=int(i), side=side, **info)
    p0, Rp0 = env.pelvis()
    yaw0 = E.yaw_of(Rp0)
    R0 = E.rot_z(yaw0)
    start = np.r_[p0, yaw0].astype(np.float32)
    lo, hi = GR.APPROACH_BOX
    w = R0.T @ np.r_[stance - p0[:2], 0.0]
    walk = np.r_[np.clip(w[0], lo[0], hi[0]), np.clip(w[1], lo[1], hi[1]), 0.0]
    k_pre = len(rec.S) - 1
    x0_pre = AE.x0_here(rec, sd, False, start)
    gs = R0.T @ (tool_w - p0)
    gs[2] = tool_w[2]
    Rs = R0.T @ R_w
    if method == "feedback":
        odo.begin()
    GR.open_walk(m, float(walk[0]), float(walk[1]), 0.0)
    res["walk"] = walk.tolist()
    if m.fell_at is not None:
        return dict(res, fell="walk", success=False)
    S_walk = np.array(rec.S[k_pre:])
    rec.start(side)
    m.hold(GR.PRE_S)
    p1, Rp1 = env.pelvis()
    yaw1 = E.yaw_of(Rp1)
    Rh1 = E.rot_z(yaw1)
    if jf is not None:  # the servo's jaw measurement: joints + leg odometry from here, anchored where the goals are
        from locomani.backends.scalebfm import motor as MM

        jf.anchor(p1, yaw1, MM.WRIST, MM.R.OFF)
        m.jaw_source = jf.jaw
        res["jaw_est"] = []
    d1 = R0.T @ np.r_[p1[:2] - p0[:2], 0.0]
    res["walk_err_cm"] = [100 * float(d1[0] - walk[0]), 100 * float(d1[1] - walk[1])]
    if method == "model":
        S_post = np.array(rec.S)
        x = WK.features(x0_pre, walk, WK.post_feats(S_post[-3:]), WK.seq_feats(np.r_[S_walk, S_post[1:]]))
        mu, _ = models["post_wkp"].predict_np(x)
        e = mu[0]
        res["walk_pred_cm"] = [float(e[0]), float(e[1])]
    elif method == "feedback":  # geometric leg odometry over the walk (leg_odo.py), as approach_eval's post_odo
        odo.on = False
        pz_ = odo.odo.pose()
        e = np.r_[100 * (pz_[0] - walk[0]), 100 * (pz_[1] - walk[1]), math.degrees(pz_[2])]
        res["walk_odo_cm"] = e.tolist()
    else:
        e = np.zeros(3)
    # where the robot believes it stands, then the goal in that frame: the command is relative to the body
    pb, yb = walk[:2] + e[:2] / 100, math.radians(float(e[2]))
    Rb = E.rot_z(yb)

    def to_cmd(tw: np.ndarray, Rw: np.ndarray):
        """A grasp pose (world, as seen before the walk) as the commands from where the robot believes it stands."""
        gs_ = R0.T @ (tw - p0)
        gs_[2] = tw[2]
        gb = Rb.T @ np.r_[gs_[:2] - pb, 0.0]
        tb = np.r_[gb[:2], gs_[2]]
        gw = np.r_[p1[:2] + (Rh1 @ np.r_[tb[:2], 0.0])[:2], tb[2]]
        rw = Rh1 @ (Rb.T @ (R0.T @ Rw))
        return gw, rw, gw - rw[:, 0] * PRE_BACK, AE.pelvis_for(float(gw[2]))

    grasp_w, rot_w, pre_w, pz = to_cmd(tool_w, R_w)
    res["pz"] = pz
    last_cmd = None
    turned_cmd = [False]  # a correction has turned the wrist command off the wanted rotation

    def reach(stage: str, goal_w: np.ndarray, chain: int, invert: bool, **kw):
        nonlocal last_cmd
        con.phase = stage
        rec.start(side)
        m.hold(GR.PRE_S)
        rec.on = False
        info_ = dict(goal_w=goal_w.tolist())
        target = goal_w
        if invert and method == "model" and not SERVO:
            st_ = state_now(env, rec, side, goal_w, rot_w, chain)
            plan0 = PI.plan_of
            PI.plan_of = lambda *a_: plan0(*a_, **kw)
            try:
                evs = WI.fixed_point(models["post"], m, st_, st_["goal"], pz, None)
            finally:
                PI.plan_of = plan0
            best = min(evs, key=lambda ev: ev["l_pos"])
            target = np.r_[st_["p"][:2] + (st_["Rh"] @ np.r_[best["t"][:2], 0.0])[:2], best["t"][2]]
            info_.update(
                pred_err=float(best["pred_err"]),
                pred_std=float(best["pred_std"]),
                shift_cm=100 * float(np.linalg.norm(target - goal_w)),
            )
            carry = (ENTRY_CARRY >= 0 and float(best["pred_std"]) >= ENTRY_CARRY) or (
                ENTRY_CARRY == -2 and turned_cmd[0]
            )
            if stage == "entry" and carry and last_cmd is not None:
                target = goal_w + (last_cmd - pre_w)
                info_.update(carry=True, shift_cm=100 * float(np.linalg.norm(target - goal_w)))
            if stage == "entry" and ENTRY_DIRECT > 0 and float(best["pred_std"]) > ENTRY_DIRECT:
                target = goal_w
                info_["plain"] = True
        out = m.reach(
            side,
            np.asarray(target, float),
            rot_w,
            pelvis_z=None if pz < 0 else float(pz),
            servo=bool((FB_SERVO and method == "feedback") or (SERVO and method == "model")),
            **kw,
        )
        last_cmd = np.asarray(target, float)
        if m.fell_at is None:
            m.hold(GR.POST_S)
        info_.update(ok=bool(out.ok), err_cm=100 * float(np.linalg.norm(m.jaw_centre(side)[0] - goal_w)))
        if jf is not None and m.fell_at is None:  # estimate against the simulator's jaw: cm, deg
            (pe, Re), (pt, Rt) = jf.jaw(side), m.jaw_centre(side)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Re @ Rt.T) - 1) / 2))))
            (pp, Rpe), (pq, Rq) = jf.pelvis(), env.pelvis()
            dR = Rpe @ Rq.T
            res["jaw_est"].append([stage, round(100 * float(np.linalg.norm(pe - pt)), 3), round(ang, 3)]
                                  + (100 * (pp - pq)).round(3).tolist()
                                  + [round(math.degrees(math.atan2(dR[1, 0], dR[0, 0])), 3),
                                     round(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))), 3)])  # fmt: skip
        return info_

    # the first move lifts the hand clear: outside the front face, at or above the pre-grasp height (grasp_pilot's lesson: a
    # hand going straight from where it hangs passes under the board's edge)
    z_clear = max(float(pre_w[2]), slot["surf"] + 0.10)
    if st["name"] == "cabinet":
        z_clear = min(z_clear, surfaces_above(st, slot["surf"]) - 0.10)
    clear_w = np.r_[c + perp * float(perp @ (tool_w[:2] - c)) - ax * CLEAR_OUT, z_clear]
    res["clear"] = reach("clear", clear_w, 0, False)
    if m.fell_at is not None:
        return dict(res, fell="clear", contacts=con.n, success=False)

    def attempt(ban: frozenset = frozenset(), retry: bool = False) -> dict:
        """Pick, reach, grip, lift, hold from the waypoint: one try. Nothing here reads more than the first try did."""
        nonlocal i, info, tool_w, R_w, grasp_w, rot_w, pre_w, pz, last_cmd
        evald: list = []
        turned_cmd[0] = False
        if PICK_K and method == "model":
            rec.start(side)
            m.hold(GR.PRE_S)
            rec.on = False
            for c in grasp_candidates(slot, g, p_o, R_o, PICK_K):
                if c[0] in ban:
                    continue
                gw_, rw_, pw_, pz_ = to_cmd(c[2], c[3])
                st_ = state_now(env, rec, side, pw_, rw_, 1)
                ev_ = min(WI.fixed_point(models["post"], m, st_, st_["goal"], pz_, None), key=lambda ev: ev["l_pos"])
                w_cm = 100 * float(g["width"][c[0]])
                entry_std = 0.0
                if PICK_ENTRY > 0:
                    st2_ = state_now(env, rec, side, gw_, rw_, 1)
                    plan0_ = PI.plan_of
                    PI.plan_of = lambda *a_: plan0_(*a_, speed=ENTRY_SPEED, ik_from_ref=True, start=pw_)
                    try:
                        entry_std = float(
                            min(
                                WI.fixed_point(models["post"], m, st2_, st2_["goal"], pz_, None),
                                key=lambda ev: ev["l_pos"],
                            )["pred_std"]
                        )
                    finally:
                        PI.plan_of = plan0_
                evald.append(
                    (
                        float(ev_["pred_std"])
                        + PICK_ATT * max(0.0, float(ev_["pred_att"]) - 8.0) / 10.0
                        + PICK_WIDTH * max(0.0, w_cm - 7.0)
                        + PICK_ENTRY * entry_std
                        + PICK_ROB * (1.0 - c[1]["rob"]),
                        c,
                        gw_,
                        rw_,
                        pw_,
                        pz_,
                    )
                )
            now = next((x for x in evald if x[1][0] == i), None)
            std0 = now[0] if now else float("inf")
            pick = dict(n=len(evald), std0=std0, std=std0, moved=False)
            if evald and (std0 > PICK_GATE or retry):
                best_ = min(evald, key=lambda x: x[0])
                if best_[0] < std0 - PICK_MARGIN:
                    i, info = best_[1][0], best_[1][1]
                    tool_w, R_w = best_[1][2], best_[1][3]
                    grasp_w, rot_w, pre_w, pz = best_[2:6]
                    res.update(grasp=int(i), pz=pz, **info)
                    pick.update(std=best_[0], moved=True)
            res["pick"] = pick
        res["pregrasp"] = reach("pregrasp", pre_w, 1, True)
        if m.fell_at is not None:
            return dict(res, fell="pregrasp", contacts=con.n, success=False)
        rot_goal = rot_w.copy()

        def fix_at(goal: np.ndarray, n: int, att: bool = False, att_min: float = ATT_FIX_MIN, **kw) -> list:
            """Up to n corrections: the jaw's miss by its own joints added to the command, optionally its rotation miss to the wrist command."""
            nonlocal last_cmd, rot_w
            out = []
            for _ in range(n):
                jp, Rj = m.jaw_centre(side)
                miss = goal - jp
                out.append(100 * float(np.linalg.norm(miss)))
                Re = rot_goal @ Rj.T
                ang = math.acos(float(np.clip((np.trace(Re) - 1) / 2, -1, 1))) if att else 0.0
                if np.linalg.norm(miss) < FIX_MIN and ang < att_min:
                    break
                prev_cmd, prev_rot = last_cmd.copy(), rot_w.copy()
                last_cmd = last_cmd + miss
                if att and ang >= att_min:
                    turned_cmd[0] = True
                    turned = Re @ rot_w
                    v = WI.rotvec_of(rot_goal.T @ turned)
                    if np.linalg.norm(v) > ATT_FIX_MAX:
                        turned = rot_goal @ GR.rotvec(v * ATT_FIX_MAX / np.linalg.norm(v))
                    rot_w = turned
                kw_ = dict(kw, start=prev_cmd, start_rot=prev_rot) if CONT else kw
                m.reach(side, last_cmd, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=False, **kw_)
                if m.fell_at is None:
                    m.hold(GR.POST_S)
            return out + [100 * float(np.linalg.norm(goal - m.jaw_centre(side)[0]))]

        if ATT_FIX and ATT_BACK > 0 and method == "model":
            Re0 = rot_goal @ m.jaw_centre(side)[1].T
            ang0 = math.acos(float(np.clip((np.trace(Re0) - 1) / 2, -1, 1)))
            res["att_pre_deg"] = math.degrees(ang0)
            if ang0 >= ATT_FIX_MIN:
                step, bkw = rot_goal[:, 0] * ATT_BACK, dict(speed=2 * ENTRY_SPEED, ik_from_ref=True)
                for sgn in (-1.0, 1.0):
                    prev_cmd = last_cmd.copy()
                    last_cmd = last_cmd + sgn * step
                    m.reach(side, last_cmd, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=False, start=prev_cmd, start_rot=rot_w.copy(), **bkw)  # fmt: skip
                    if m.fell_at is None:
                        m.hold(GR.POST_S)
                    if m.fell_at is None and sgn < 0:
                        res["att_fix"] = fix_at(pre_w - step, ATT_N, True, math.radians(5.0), **bkw)
                        Re1 = rot_goal @ m.jaw_centre(side)[1].T
                        res["att_far_deg"] = math.degrees(math.acos(float(np.clip((np.trace(Re1) - 1) / 2, -1, 1))))
                    if m.fell_at is not None:
                        return dict(res, fell="att_back", contacts=con.n, success=False)
        if PRE_FIX and method in FB:
            res["pre_fix"] = fix_at(pre_w, PRE_FIX, ATT_FIX and ATT_BACK <= 0)
            if m.fell_at is not None:
                return dict(res, fell="pre_fix", contacts=con.n, success=False)
        if REPICK and method == "model" and evald:

            def entry_spread() -> float:
                """The entry's ensemble spread from the state the hand is really in, before committing to it."""
                rec.start(side)
                m.hold(GR.PRE_S)
                rec.on = False
                st_ = state_now(env, rec, side, grasp_w, rot_w, 1)
                p_, _ = m.jaw_centre(side)
                plan0 = PI.plan_of
                PI.plan_of = lambda *a_: plan0(
                    *a_, speed=ENTRY_SPEED, ik_from_ref=True, start=p_ + 0.5 * (last_cmd - p_)
                )
                try:
                    return float(
                        min(WI.fixed_point(models["post"], m, st_, st_["goal"], pz, None), key=lambda ev: ev["l_pos"])[
                            "pred_std"
                        ]
                    )
                finally:
                    PI.plan_of = plan0

            seen, rp = {int(i)}, []
            res["repick"] = rp
            for _ in range(REPICK):
                std_ = entry_spread()
                rp.append(std_)
                rest = sorted((x for x in evald if x[1][0] not in seen), key=lambda x: x[0])
                if std_ <= REPICK_GATE or not rest:
                    break
                x_ = rest[0]
                seen.add(x_[1][0])
                i, info = x_[1][0], x_[1][1]
                tool_w, R_w = x_[1][2], x_[1][3]
                grasp_w, rot_w, pre_w, pz = x_[2:6]
                res.update(grasp=int(i), pz=pz, **info)
                rot_goal = rot_w.copy()
                res["pregrasp"] = reach("pregrasp", pre_w, 1, True)
                if m.fell_at is not None:
                    return dict(res, fell="repick", contacts=con.n, success=False)
                if PRE_FIX:
                    res["pre_fix"] = fix_at(pre_w, PRE_FIX, ATT_FIX)
                    if m.fell_at is not None:
                        return dict(res, fell="repick_fix", contacts=con.n, success=False)
        if ENTRY_MID > 0 and method == "model":
            mid_w = grasp_w - rot_goal[:, 0] * ENTRY_MID
            p_, _ = m.jaw_centre(side)
            res["entry_mid"] = reach(
                "entry", mid_w, 1, True, speed=ENTRY_SPEED, ik_from_ref=True, start=p_ + 0.5 * (last_cmd - p_)
            )
            if m.fell_at is not None:
                return dict(res, fell="entry_mid", contacts=con.n, success=False)
            res["mid_fix"] = fix_at(mid_w, max(ENTRY_FIX, 1), False, speed=ENTRY_SPEED, ik_from_ref=True)
            if m.fell_at is not None:
                return dict(res, fell="mid_fix", contacts=con.n, success=False)
        p_, _ = m.jaw_centre(side)
        ekw = dict(start=last_cmd.copy() if CONT >= 3 else p_ + 0.5 * (last_cmd - p_))
        if CONT >= 2:
            ekw["start_rot"] = rot_w.copy()
        res["entry"] = reach("entry", grasp_w, 1, True, speed=ENTRY_SPEED, ik_from_ref=True, **ekw)
        if m.fell_at is not None:
            return dict(res, fell="entry", contacts=con.n, success=False)
        if ENTRY_FIX and method in FB:
            res["entry_fix"] = fix_at(grasp_w, ENTRY_FIX, False, speed=ENTRY_SPEED, ik_from_ref=True)
            if m.fell_at is not None:
                return dict(res, fell="entry_fix", contacts=con.n, success=False)
        if GRIP_CHECK and method in FB:
            jp_, Rj_ = m.jaw_centre(side)
            U_ = g["R"][i] @ (np.diag([1.0, -1.0, -1.0]) if info["flip"] else np.eye(3))
            hold_, d_ = grip_check(slot, g["tool"][i], U_, rot_goal.T @ (jp_ - grasp_w), rot_goal.T @ Rj_)
            res["grip_check"] = dict(hold=bool(hold_), shift_cm=None if d_ is None else (100 * d_).tolist())
            if d_ is not None:
                goal2 = grasp_w + Rj_ @ d_
                prev_cmd = last_cmd.copy()
                last_cmd = last_cmd + Rj_ @ d_
                m.reach(side, last_cmd, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=False, speed=ENTRY_SPEED, ik_from_ref=True, start=prev_cmd, start_rot=rot_w.copy())  # fmt: skip
                if m.fell_at is None:
                    m.hold(GR.POST_S)
                if m.fell_at is None:
                    res["grip_check"]["fix"] = fix_at(goal2, 2, False, speed=ENTRY_SPEED, ik_from_ref=True)
                if m.fell_at is not None:
                    return dict(res, fell="grip_check", contacts=con.n, success=False)
        jaw = m.jaw_centre(side)[0]
        res["at_grip_cm"] = (100 * (jaw - tool_w)).tolist()
        res["at_grip_err"] = 100 * float(np.linalg.norm(jaw - tool_w))
        Rj = m.jaw_centre(side)[1]
        res["att_deg"] = math.degrees(float(np.arccos(np.clip((np.trace(R_w.T @ Rj) - 1) / 2, -1, 1))))
        res["opening_pre_cm"] = 100 * m.jaw_opening(side)
        res["obj_in_tool_cm"] = (
            100 * (R_w.T @ (sim.d.xpos[b] - tool_w))
        ).tolist()  # where the object centre should be, in the planned tool frame
        res["obj_in_jaw_cm"] = (100 * (Rj.T @ (sim.d.xpos[b] - jaw))).tolist()  # where it is, in the real jaw frame
        res["grasp_width_cm"] = 100 * float(g["width"][i])
        res["cmd_rot_err_deg"] = math.degrees(float(np.arccos(np.clip((np.trace(rot_w.T @ Rj) - 1) / 2, -1, 1))))
        res["pushed_cm"] = 100 * float(np.linalg.norm(sim.d.xpos[b][:2] - p_o[:2]))
        con.phase = "grip"
        gr = m.grip(side)
        res["grip_held"], res["grip_info"] = bool(gr.ok), dict(gr.info or {})
        res["opening_post_cm"] = 100 * m.jaw_opening(side)
        if RETRY and m.jaw_opening(side) < RETRY_OPEN:
            return dict(res, fell=None if m.fell_at is None else "grip", gripped=False, success=False, held_3s=False, rise_cm=100 * float(sim.d.xpos[b][2] - p_o[2]), rise_15_cm=0.0, contacts=dict(con.n), parts=dict(con.parts), others_moved_cm=0.0, jaw_end_cm=100 * m.jaw_opening(side))  # fmt: skip
        trace = []
        if os.environ.get("ROOM_GRASP_TRACE"):
            orig_tick = m.tick
            ct = [0]

            def traced():
                ok = orig_tick()
                ct[0] += 1
                if ct[0] % 10 == 0:
                    trace.append(
                        [
                            round(float(m.t), 2),
                            round(100 * float(sim.d.xpos[b][2] - p_o[2]), 2),
                            round(100 * float(m.jaw_centre(side)[0][2] - jaw[2]), 2),
                            round(100 * m.jaw_opening(side), 2),
                            int(side in m.gripping(b)),
                            [round(float(v), 3) for v in m.jaw],
                        ]
                    )
                return ok

            m.tick = traced
        con.phase = "lift"
        lift = m.jaw_centre(side)[0] + np.array([0.0, 0.0, LIFT_CABINET if st["name"] == "cabinet" else LIFT_M])
        if GRIP_SETTLE > 0:
            m.hold(GRIP_SETTLE)
        lkw: dict = {"speed": LIFT_SPEED} if LIFT_SPEED > 0 else {}
        if LIFT_CONT:
            lift = last_cmd + np.array([0.0, 0.0, LIFT_CABINET if st["name"] == "cabinet" else LIFT_M])
            lkw.update(start=last_cmd.copy(), start_rot=rot_w.copy(), ik_from_ref=True)
        m.reach(side, lift, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=False, **lkw)
        held_15 = False
        rise_15 = 0.0
        if m.fell_at is None:
            con.phase = "hold"
            m.hold(
                HOLD_S
            )  # the validator's hold: MuJoCo's friction creeps about 0.4 cm/s under a held load (memory mujoco-grip-creep)
            held_15, rise_15 = side in m.gripping(b), 100 * float(sim.d.xpos[b][2] - p_o[2])
            m.hold(1.5)
        if trace:
            m.tick = orig_tick
            res["trace"] = trace
        obj = sim.d.xpos[b]
        others = [
            100 * float(np.linalg.norm(sim.d.xpos[sim.m.body(n).id] - p0_))
            for n, p0_ in z0_others.items()
            if n != slot["object"]
        ]
        gripped = side in m.gripping(b)
        return dict(res, fell=None if m.fell_at is None else "grip/lift", gripped=bool(gripped), rise_cm=100 * float(obj[2] - p_o[2]), jaw_end_cm=100 * m.jaw_opening(side),
                    success=bool(held_15 and rise_15 > 100 * RISE_M and m.fell_at is None), held_3s=bool(gripped and obj[2] > p_o[2] + RISE_M), rise_15_cm=rise_15, contacts=dict(con.n), parts=dict(con.parts), others_moved_cm=max(others) if others else 0.0)  # fmt: skip

    out = attempt()
    if RETRY:
        res["tries"] = tries = [brief(out)]
        ban: set = set()
        while (
            slot["surf"] >= RETRY_MIN_Z
            and len(tries) <= RETRY
            and out.get("fell") is None
            and not out.get("declined")
            and out.get("jaw_end_cm", 99.0) < 100 * RETRY_OPEN
        ):
            # the jaw reading says nothing is held: open, back out along the jaw, go to the waypoint, look at the object again
            con.phase = "retreat"
            m.release(side)
            jp_, Rj_ = m.jaw_centre(side)
            kw_ = dict(speed=2 * ENTRY_SPEED, ik_from_ref=True)
            if m.fell_at is None:
                m.reach(
                    side, jp_ - Rj_[:, 0] * RETREAT_M, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=False, **kw_
                )
            if m.fell_at is None:
                m.hold(0.5)
            if m.fell_at is not None:
                out = dict(out, fell="retreat", success=False)
                break
            p2, R2 = sim.d.xpos[b].copy(), quat_mat(sim.d.xquat[b])
            moved = 100 * float(np.linalg.norm(p2 - p_o))
            if (
                p2[2] < slot["surf"] - 0.02
            ):  # off its surface: on the floor or in the footwell, nothing to reach from here
                tries.append(dict(stop="off the surface", moved_cm=moved))
                break
            if moved < 2.0:
                ban.add(int(i))  # same object, same grasp would repeat the same arm
            else:
                ban.clear()
            i2, info2 = choose_grasp(slot, g["tool"], g["R"], g["score"], g["held"], p2, R2)
            if i2 is not None and i2 in ban:
                alt = [c_ for c_ in grasp_candidates(slot, g, p2, R2, max(PICK_K, 8)) if c_[0] not in ban]
                i2, info2 = (alt[0][0], alt[0][1]) if alt else (None, None)
            if i2 is None:
                tries.append(dict(stop="no grasp", moved_cm=moved))
                break
            p_o, R_o, i, info = p2, R2, i2, info2
            tool_w, R_w = p_o + R_o @ g["tool"][i], R_o @ g["R"][i]
            if info["flip"]:
                R_w = R_w @ np.diag([1.0, -1.0, -1.0])
            res.update(grasp=int(i), **info)
            grasp_w, rot_w, pre_w, pz = to_cmd(tool_w, R_w)
            res["pz"] = pz
            z2 = max(float(pre_w[2]), slot["surf"] + 0.10)
            if st["name"] == "cabinet":
                z2 = min(z2, surfaces_above(st, slot["surf"]) - 0.10)
            res["clear"] = reach(
                "clear", np.r_[c + perp * float(perp @ (tool_w[:2] - c)) - ax * CLEAR_OUT, z2], 0, False
            )
            if m.fell_at is not None:
                out = dict(out, fell="clear2", success=False)
                break
            out = attempt(frozenset(ban), True)
            tries.append(dict(brief(out), moved_cm=moved))
        out = dict(out, tries=tries, success_first=bool(tries[0].get("success")))
    return out


def load_cache(layout: dict) -> dict:
    root = ROOM.GN / "scenes"
    out = {}
    for s in layout["slots"]:
        z = np.load(root / f"obj_{s['id']:03d}" / "grasp_cache.npz", allow_pickle=True)
        out[s["object"]] = dict(
            tool=z["tool"], R=z["R"], score=z["score"], held=z["held"].astype(bool), width=z["width"]
        )
        rf = root / f"obj_{s['id']:03d}" / os.environ.get("ROOM_GRASP_ROBUST_FILE", "grasp_robust.npz")
        if (ROBUST_W or ROBUST_MIN or PICK_ROB) and rf.exists():
            ROB_OF[id(out[s["object"]]["tool"])] = np.load(rf)["rob"]
    return out


def worker(wid: int, jobs: list, methods, tag: str) -> None:
    import gen_reach as GR
    import torch
    import walk_model as WK

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    by: dict[int, list] = {}
    for k, j in jobs:
        by.setdefault(k, []).append(j)
    models = {"post": torch.load(AE.WM_POST, map_location="cpu", weights_only=False).eval(),
              "post_wkp": torch.load(WK.WALK_MODEL_POST, map_location="cpu", weights_only=False).eval()}  # fmt: skip
    for k, slots in by.items():
        env = ROOM.make_env(k)
        layout = json.load(open(ROOM.OUT / f"layout_{k}" / "layout.json"))
        cache = load_cache(layout)
        rec = GR.Recorder(env)
        con = Contacts(env)
        odo = None
        if "feedback" in methods:
            import leg_odo as LO

            odo = LO.TickFeed(env)
        jf = None
        if SERVO_SRC == "odo":
            import leg_odo as LO

            jf = LO.JawFeed(env)
        env.m.reset(
            (0.0, 0.0), 0.0
        )  # the first call after start-up differs (scalebfm-first-call-nondeterminism): warm up
        env.m.hold(1.0)
        for j in slots:
            for meth in methods:
                tic = time.time()
                env.m.jaw_source = None
                r = episode(env, rec, con, models, layout, k, j, meth, cache, odo, jf)
                if jf is not None:
                    jf.on, env.m.jaw_source = False, None
                    if getattr(jf, "raw", None):
                        np.save(OUT / f"raw{tag}_L{k}_s{j}.npy", np.array(jf.raw))
                out.append(r)
                print(f"w{wid} L{k} s{j} {r['station']} {r['object']} {meth}: success {r.get('success')} grip err {r.get('at_grip_err', float('nan')):.2f} "
                      f"rise {r.get('rise_cm', float('nan')):.1f} pushed {r.get('pushed_cm', float('nan')):.1f} fell {r.get('fell')} declined {r.get('declined')} contacts {r.get('contacts')} | {time.time() - tic:.0f} s", flush=True)  # fmt: skip
            json.dump(
                out,
                open(OUT / f"h_w{wid}{tag}.json", "w"),
                default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
            )


def launch(procs: int, gpus: str, layouts: list[int], slots_per: int, methods: str, tag: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in layouts:
        n = len(json.load(open(ROOM.OUT / f"layout_{k}" / "layout.json"))["slots"])
        jobs += [(k, j) for j in range(n if slots_per <= 0 else min(n, slots_per))]
    cards, ps = gpus.split(","), []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)])
        log = open(OUT / f"h_w{w}{tag}.log", "w")
        spec = ";".join(f"{k},{j}" for k, j in jobs[w::procs])
        cmd = [str(E.SIM_PY), __file__, "_w", "--wid", str(w), "--jobs", spec, "--methods", methods, "--tag", tag]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def report(tag: str) -> None:
    import glob

    R = []
    for f in sorted(glob.glob(str(OUT / f"h_w?{tag}.json"))):
        R += json.load(open(f))
    methods = [m for m in METHODS if any(r["method"] == m for r in R)]
    print(len(R), "episodes")
    for m in methods:
        d = [r for r in R if r["method"] == m]
        ok = sum(1 for r in d if r.get("success"))
        e = [r["at_grip_err"] for r in d if "at_grip_err" in r]
        print(f"{m:8s} {ok}/{len(d)} = {100 * ok / len(d):.0f}%  grip err med {np.median(e) if e else float('nan'):.2f}  fell {sum(1 for r in d if r.get('fell'))}  "
              f"declined {sum(1 for r in d if r.get('declined'))}  furniture contact {sum(1 for r in d if r.get('contacts'))}")  # fmt: skip
    for m in methods:
        print(
            f"  held 3 s {m}: {sum(1 for r in R if r['method'] == m and r.get('held_3s'))}/{sum(1 for r in R if r['method'] == m)}"
        )
    for st in ("tray", "low", "table", "counter", "high", "cabinet"):
        print(
            f"  {st:8s}",
            "  ".join(
                f"{m} {sum(1 for r in R if r['method'] == m and r['station'] == st and r.get('success'))}/{sum(1 for r in R if r['method'] == m and r['station'] == st)}"
                for m in methods
            ),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["h", "_w", "d"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--layouts", default="0")
    ap.add_argument("--slots", type=int, default=0, help="first N slots of each layout; 0: all")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--jobs", default="")
    a = ap.parse_args()
    if a.phase == "h":
        launch(a.procs, a.gpus, [int(x) for x in a.layouts.split(",")], a.slots, a.methods, a.tag)
    elif a.phase == "_w":
        worker(a.wid, [tuple(int(v) for v in s.split(",")) for s in a.jobs.split(";")], a.methods.split(","), a.tag)
    else:
        report(a.tag)
