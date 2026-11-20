"""Reach world-model dataset (2026-10-30): what frozen ScaleBFM actually delivers when told to put a jaw centre somewhere.

One reset = stand at the origin (35 % first walk 0.3-1.2 m to a random pose, so the reach starts in the post-walk sway),
then CHAIN reaches in a row, each starting where the last one left the body. A reach is one task-space command held to
the end, raw controller (servo off, so the model learns the controller, not our servo on top of it):

  command  side, jaw-centre target (x ahead, y left in the heading frame at command time, z above the floor), wrist yaw
           offset (-60..60 deg) and pitch (0 level .. 90 down), pelvis height (stand, or 0.52-0.76 m)
  targets  70 % core box (x 0.15-0.70, y -0.40..0.15, z 0.15-1.35 m), 30 % wide box (x 0.05-1.00, y -0.60..0.35,
           z 0.02-1.50 m: much of it out of reach on purpose); y mirrored for the left hand

Recorded at 10 Hz from 0.3 s before the command to 1 s after the reach returns: self state S (65, env.py), world row W
(13, env.world2), acting jaw rotation J (6, world, first two columns), the commanded jaw centre in the pelvis frame and
the time since the command C (4). Open floor, nothing to touch.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 12000 --gpus 0,1
    $PY scripts/experiments/reachwm_20261030/gen_reach.py merge
    # short side steps before the reach (2026-11-01, see REACH_SIDESTEP below)
    REACH_SIDESTEP=1 REACH_CHAIN=1 REACH_ID0=100000 REACH_OUT=outputs/reachwm_20261030/data_side \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 12000 --seed 5000000
    # wide wrist commands on top of the short steps (2026-11-02, see REACH_WRIST below)
    REACH_WRIST=1 REACH_SIDESTEP=1 REACH_CHAIN=1 REACH_ID0=200000 REACH_OUT=outputs/reachwm_20261030/data_wrist \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 18000 --seed 7000000
    # far turns only (2026-11-02)
    REACH_WRIST=1 REACH_WRIST_MIN=45 REACH_SIDESTEP=1 REACH_CHAIN=1 REACH_ID0=300000 REACH_OUT=outputs/reachwm_20261030/data_wrist2 \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 18000 --seed 9000000
    # the grasp's reaches with the entry as it now runs: 0.08 m/s, half-way start, IK from the reference (2026-11-08; 74 min at 6 procs, 0 falls)
    REACH_GRASP=1 REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 REACH_GRASP_IKREF=1 REACH_CHAIN=2 REACH_ID0=900000 REACH_OUT=outputs/reachwm_20261030/data_grasp2 \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 9000 --seed 17000000 --gpus 0,1
    # the grasp room's reaches under leg odometry (2026-11-09, see REACH_GRASP_ROOM below)
    BFM_ROOT=odo REACH_GRASP=1 REACH_GRASP_ROOM=1 REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 REACH_GRASP_IKREF=1 REACH_ID0=1300000 REACH_OUT=outputs/reachwm_20261030/data_graspr_odo \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 7500 --seed 27000000 --gpus 0,1
    # walk then reach, recorded from before the walk (2026-11-03, see REACH_APPROACH below)
    REACH_APPROACH=1 REACH_ID0=400000 REACH_OUT=outputs/reachwm_20261030/data_approach \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 18000 --seed 11000000
    REACH_APPROACH=1 REACH_APPROACH_NOTURN=1 REACH_ID0=430000 REACH_OUT=outputs/reachwm_20261030/data_approach0 \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 6000 --seed 13000000
    # one command through several via points, on SIMPLE's robot (2026-11-20, see REACH_VIA and REACH_VIA_REPLAY below)
    X="BFM_ROOT=odo REACH_HAND_OFF=0.0415,0,0 REACH_SCENE_DIR=outputs/scene/simple_g1_sonic REACH_GRASP=1 REACH_GRASP_ROOM=1"
    X="$X REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 REACH_GRASP_IKREF=1 REACH_WRIST=1"
    env $X REACH_VIA=1 REACH_ID0=2000000 REACH_OUT=outputs/reachwm_20261030/data_via_trial \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 300 --seed 31000000 --gpus 1
    env $X REACH_VIA_REPLAY=<via dir>:<prep dir>,<via dir>:<prep dir> REACH_ID0=2100000 \
        REACH_OUT=outputs/reachwm_20261030/data_via_replay \
        $PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs 6 --resets 35 --seed 33000000 --gpus 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))
import env as E  # noqa: E402
import via_gen as VG  # noqa: E402

OUT = Path(os.environ.get("REACH_OUT", E.REPO / "outputs" / "reachwm_20261030" / "data"))
# sensitivity probe (2026-10-30): move every reach target by this many metres in a direction drawn from a separate
# generator, so the episode's own random draws stay identical (0 = exact replay)
PERTURB_M = float(os.environ.get("REACH_PERTURB_M", "0"))
# REACH_HOME=1 (2026-10-31, canonical starts): before every reach, blend the whole body back to the reset's standing pose
# (legs, waist and arms) at the current pelvis xy and yaw over HOME_S, then stand HOME_SETTLE_S. REACH_CHAIN reaches per
# reset.
HOME = os.environ.get("REACH_HOME", "0") == "1"
HOME_S, HOME_SETTLE_S = 1.0, 1.0
CHAIN = int(os.environ.get("REACH_CHAIN", "2"))
# BFM_ROOT=odo (2026-11-09, no pelvis truth at run time, selfwm_20261029/env.py): ScaleBFM reads leg odometry. data_ref_odo 10000 resets
# (seed 21000000, ID0 1000000, 78 min at 6 procs), data_approach_odo 12000 (23000000, 1100000), data_approach0_odo 4000 (25000000, 1200000).
# REACH_SIDESTEP=1 (2026-11-01, step then reach): after the reset (and, in SIDESTEP_WALK of the resets, the 0.3-1.2 m walk)
# hold PRE_S, then step with the heading kept (Motor.sidestep) by a displacement drawn uniformly from SIDESTEP_BOX
# (forward, left; a second step in SIDESTEP_TWICE of the resets), so the reaches start in the stance a short step leaves.
# restance.py's step-then-reach starts there; the walks above are 0.3-1.2 m and mostly turn. These segments count as walked.
SIDESTEP = os.environ.get("REACH_SIDESTEP", "0") == "1"
SIDESTEP_BOX = (np.array([-0.30, -0.20]), np.array([0.20, 0.20]))
SIDESTEP_TWICE, SIDESTEP_WALK = 0.25, 0.5
# REACH_WRIST=1 (2026-11-02, wide wrist commands): turn the drawn wrist command further, about a uniformly random axis of
# the jaw frame by an angle uniform in 0..WRIST_MAX_DEG (stored as wrist_dev, a rotation vector in the jaw frame). The
# controller follows about half of a commanded wrist turn, so pre-compensating a wanted rotation needs commands about
# twice as far off it, roll included; the dyaw/pitch family alone has no roll and stops at the wanted range.
WRIST = os.environ.get("REACH_WRIST", "0") == "1"
WRIST_MAX_DEG = 90.0
# REACH_WRIST_MIN (2026-11-02, far turns): the turn angle's lower bound. data_wrist2 draws 45..90 deg only: the learning
# curve (lc_eval.py) is saturated below 30 deg and still falling at >= 60 deg.
WRIST_MIN_DEG = float(os.environ.get("REACH_WRIST_MIN", "0"))
# REACH_APPROACH=1 (2026-11-03, walk then reach, eyes closed): the segment starts BEFORE an approach walk. After the reset
# (and, in APPROACH_DIVERSE of the resets, the usual 0.3-1.2 m walk to a random pose, stop correction on) the robot holds
# PRE_S; that state is the segment's start (k0, start). Both commands are drawn there: the walk (forward, left from
# APPROACH_BOX, heading kept while walking, then a turn of APPROACH_YAW_DEG at most) and the reach (sample_command, in the
# frame the robot will believe it stands in once the walk went as commanded). The walk runs with the motor's stop
# correction off (it reads the true pelvis position), so after the sighting nothing reads the simulator's truth: the
# reach is then commanded in the body frame as drawn, wherever the walk really ended. Saved: target / rot6 = the reach
# command expressed in the start frame through the commanded walk (where the hand lands if walk and reach both go
# exactly as commanded), target_b = the reach command in the believed frame, walk = (forward, left, turn) asked,
# walk_err = (forward, left, turn) the walk really ended off it (start frame; truth, for analysis only), ref_* = the
# plan the reach would get if the body stood where it believes it will, with its pre-walk posture (computed before the
# walk, available to a model at the start), ref_*_post = the plan the reach really got.
APPROACH = os.environ.get("REACH_APPROACH", "0") == "1"
APPROACH_BOX = (np.array([-0.40, -0.30]), np.array([1.00, 0.30]))
APPROACH_YAW_DEG, APPROACH_DIVERSE = 30.0, 0.5
# REACH_APPROACH_NOTURN (2026-11-03, after approach_eval v1): the fraction of approach walks with no turn at all. Any turn,
# even under 1 deg, adds a turn-in-place phase that moves the feet (from the same state a straight 0.9 m walk ended 5.1 cm
# off, the same walk with a 9 deg turn 9.6 cm), so no turn is its own regime, and the uniform turn draw never hit it.
APPROACH_NOTURN = float(os.environ.get("REACH_APPROACH_NOTURN", "0"))
# REACH_GRASP=1 (2026-11-06, the grasp's own reaches): every reset is the grasp experiment's approach (reset 1.5-2.5 m back,
# +-0.2 m to the side, heading +-15 deg, walk to the stance, heading 0, hold), then the two reaches of a grasp, level jaw:
# the pre-grasp in front of the body and the entry straight ahead from it, both commands jittered like inverted ones.
# Open floor: the model stays a free-space model. The tracks of the entries are what a model of the path would train on.
GRASP = os.environ.get("REACH_GRASP", "0") == "1"
# REACH_GRASP_ROOM=1 (2026-11-09, the grasp room's own reaches, room_grasp.py, leg odometry): with REACH_GRASP and its entry switches.
# Each reset is a short open walk to the stance, then the three reaches of a grasp: the waypoint outside the furniture front, the
# pre-grasp 10 cm back along the jaw's axis, the slow entry. The grasp's height 0.10-1.12 m, the pelvis-to-front stance the room
# uses at that height, the wrist at the pitch room_grasp.pitch_band allows and a heading offset within 60 deg, the commands
# jittered like inverted ones. Open floor: the model stays a free-space model (body-furniture contact did not explain the room's
# 2.75 cm: 2.78 with a contact, 2.71 without). The entry is recorded as chain 1, as room_grasp.py uses it.
# Result 2026-11-09: 9000 resets, 27000 segments, 8 falls; room_grasp.py _v3 jaw error at the grip 2.75 -> 1.91 cm, see its docstring.
GRASP_ROOM = os.environ.get("REACH_GRASP_ROOM", "0") == "1"
# REACH_GRASP_ROOM_EMP=<json> (2026-11-09): draw the grasp from the rows of that file (the candidates room_grasp.py's pick compares on the development layouts,
# stance frame) with some noise, instead of the uniform box: more data where the selection actually asks for commands.
EMP_ROWS = json.load(open(os.environ["REACH_GRASP_ROOM_EMP"])) if os.environ.get("REACH_GRASP_ROOM_EMP") else None
GRASP_WALK, GRASP_X, GRASP_Y, GRASP_Z, GRASP_FWD = (1.5, 2.5), (0.18, 0.30), (-0.30, 0.0), (0.25, 1.30), (0.10, 0.42)
GRASP_JITTER_M = 0.03
# REACH_GRASP_SPEED, REACH_GRASP_CONT, REACH_GRASP_IKREF (2026-11-08, the entry as the grasp runs it): the entry at this speed
# (m/s; 0: MotorConfig.reach_speed), its path from this fraction of the way from the measured jaw to the pre-grasp command
# (grasp_exp.py's Opt.cont), its whole-body IK from the reference pose (Motor.reach's ik_from_ref). The pre-grasp is unchanged.
# Any of them on also saves swing_cm, how far the plan's first blend takes the reference jaw off the straight segment.
GRASP_SPEED = float(os.environ.get("REACH_GRASP_SPEED", "0"))
GRASP_CONT = float(os.environ.get("REACH_GRASP_CONT", "0"))
GRASP_IKREF = os.environ.get("REACH_GRASP_IKREF", "0") == "1"
GRASP_ENTRY = GRASP and bool(GRASP_SPEED or GRASP_CONT or GRASP_IKREF)
# REACH_VIA=1 (2026-11-20, selfmodel_via_20261120; with REACH_GRASP=1 REACH_GRASP_ROOM=1 and the entry switches): the grasp
# room's reset, but the hand goes from where the walk left it to the pre-grasp in ONE command through several via points
# (Motor.reach's via / via_rot, as simple_bench.py --avoid sends a planned path), along a random path drawn at the command
# (via_gen.random_path: bends, lead-in, wrist turns, speed); then the slow entry as before. Two segments a reset: chain 0
# the path, chain 1 the entry; the waypoint outside the furniture front is not run. REACH_VIA_SIDE=right|left: one hand
# only (default: either, as room_commands draws it). Open floor, servo off.
# Saved on top of the usual fields (heading frame at the command, like target): via_p (16, 3) and via_r6 (16, 6) the via
# points and the wrist rotations there, n_via, via_speed (m/s), via_gen (via_gen.GEN_KEYS), via_tries and via_tol (the
# path drawn at that try, simplified to within that many m), via_route (-1), via_start_cm (-1), hand_q (the acting
# hand's finger joints at the command) and the whole reference plan of every segment: RQ (joints), RP (pelvis position
# and quaternion), RT (s from the command) of each whole-body IK pose, n_ref poses a segment (the first is where the
# reference stood at the command) -- "the arm goes where the reference says" is the baseline a model has to beat.
# REACH_VIA_REPLAY=<via dir>:<prep dir>[,...] (with the same switches; REACH_VIA not needed): reset i runs path
# i % (number of paths) of via_gen.load_routes -- the cuRobo paths simple_bench.py --avoid ran -- on the open floor, the
# right hand, the bench's speed, no entry: the first pass from the reset's stand, later passes after the room's short
# walk. Each path is placed by its own heading frame onto the robot's. via_route: the path's index in <REACH_OUT>/
# routes.json; via_start_cm: how far the hand starts from where it started in the bench.
VIA_REPLAY = os.environ.get("REACH_VIA_REPLAY", "")
VIA = os.environ.get("REACH_VIA", "0") == "1" or bool(VIA_REPLAY)
VIA_SIDE = os.environ.get("REACH_VIA_SIDE", "")
ROUTES = VG.load_routes(VIA_REPLAY) if VIA_REPLAY else []
assert not VIA or (GRASP and GRASP_ROOM), "REACH_VIA / REACH_VIA_REPLAY run with REACH_GRASP=1 REACH_GRASP_ROOM=1"
assert not VIA_REPLAY or ROUTES, f"REACH_VIA_REPLAY: no via_*.npz with its prep_*.npz under {VIA_REPLAY}"
ENTRY_CHAIN = 2 if GRASP_ROOM and not VIA else 1  # which of a reset's reaches is the grasp's entry
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
ID0 = int(os.environ.get("REACH_ID0", "0"))  # added to the reset ids, so a second dataset keeps its own split buckets
# REACH_STUDENT=<student .pt> (2026-11-04, the relabelled controller of relabel_train.py): from the command to the end of
# the POST_S hold ScaleBFM runs the student's weights, copied into the policy's own parameters in place so the captured
# CUDA graph runs them; the walk, the steps and the PRE_S hold run the frozen ones (as relabel_eval.py deploys it).
# Default off. A file holding the frozen weights replays the frozen dataset bit for bit.
STUDENT = os.environ.get("REACH_STUDENT", "")
FLUSH = 500  # segments per saved chunk
PRE_S = 0.3
# REACH_POST_S (2026-11-16): seconds recorded after the reach returns, 1.0 by default (every dataset so far); a longer
# tail is a diagnostic for controllers that settle more slowly (SONIC: 5-7 % of segments settled at 1.0 s)
POST_S = float(os.environ.get("REACH_POST_S", "1.0"))
CORE = (np.array([0.15, -0.40, 0.15]), np.array([0.70, 0.15, 1.35]))
WIDE = (np.array([0.05, -0.60, 0.02]), np.array([1.00, 0.35, 1.50]))


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotvec(v) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues)."""
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


def sample_command(rng) -> dict:
    side = "right" if rng.random() < 0.5 else "left"
    lo, hi = CORE if rng.random() < 0.7 else WIDE
    t = rng.uniform(lo, hi)
    if side == "left":
        t[1] = -t[1]
    if PERTURB_M > 0:
        d = np.random.default_rng(abs(hash(tuple(np.round(t, 6)))) % (1 << 32)).normal(size=3)
        t = t + PERTURB_M * d / np.linalg.norm(d)
    dyaw = float(np.radians(rng.uniform(-60.0, 60.0)))
    pitch = 0.0 if rng.random() < 0.35 else float(np.radians(rng.uniform(0.0, 90.0)))
    pel = None if rng.random() < 0.45 else float(rng.uniform(0.52, 0.76))
    dev = np.zeros(3)
    if WRIST:
        a = rng.normal(size=3)
        dev = a / np.linalg.norm(a) * math.radians(rng.uniform(WRIST_MIN_DEG, WRIST_MAX_DEG))
    return dict(side=side, target=t, dyaw=dyaw, pitch=pitch, pelvis_z=pel, wrist_dev=dev)


def room_commands(rng) -> list[dict]:
    """REACH_GRASP_ROOM: the waypoint, the pre-grasp and the entry of one grasp in the room, heading frame at the stance."""
    side = "right" if rng.random() < 0.5 else "left"
    side = VIA_SIDE or side
    sd = 1.0 if side == "right" else -1.0
    z = float(rng.uniform(0.10, 1.12))
    back = 0.22 if z < 0.25 else 0.42 if z < 0.50 else 0.30
    grasp = np.array([back + float(rng.uniform(0.12, 0.30)), -sd * float(rng.uniform(0.06, 0.30)), z])
    lo, hi = (0.0, 35.0) if z >= 0.80 else (0.0, 65.0) if z >= 0.55 else (0.0, 70.0) if z >= 0.30 else (40.0, 70.0)
    pitch = math.radians(float(rng.uniform(lo, hi)))
    dyaw = math.radians(
        float(np.clip(rng.normal(0.0, 10.0), -25, 25)) if rng.random() < 0.6 else float(rng.uniform(-60.0, 60.0))
    )
    emp_surf = None
    if EMP_ROWS is not None:
        row = EMP_ROWS[int(rng.integers(len(EMP_ROWS)))]
        side, sd, back, emp_surf = row["side"], 1.0 if row["side"] == "right" else -1.0, row["back"], row["surf"]
        z = float(row["z"] + rng.normal(0.0, 0.015))
        grasp = np.array([row["x"] + float(rng.normal(0.0, 0.02)), row["y"] + float(rng.normal(0.0, 0.02)), z])
        pitch = math.radians(float(np.clip(row["pitch"] + rng.normal(0.0, 4.0), -5.0, 90.0)))
        dyaw = math.radians(float(np.clip(row["az"] + rng.normal(0.0, 4.0), -60.0, 60.0)))
    axis = E.rot_z(dyaw) @ rot_y(pitch)[:, 0]
    rule = None if z >= 0.60 else float(np.clip(z + 0.05, 0.52, 0.76))
    pel = rule if rng.random() < 0.9 else float(rng.uniform(0.52, 0.76))
    pre = grasp - 0.10 * axis + rng.normal(0.0, GRASP_JITTER_M, 3)
    entry = grasp + rng.normal(0.0, GRASP_JITTER_M, 3)
    surf = z - float(rng.uniform(0.04, 0.12)) if emp_surf is None else emp_surf
    clear = np.array(
        [back - 0.10 + float(rng.normal(0.0, 0.02)), grasp[1], max(float(pre[2]), surf + 0.10)]
    ) + rng.normal(0.0, 0.01, 3)
    dev = np.zeros(3)
    if WRIST:  # 2026-11-19 (simple_bench): one turn for the grasp's three reaches, as sample_command draws it; the
        # dyaw/pitch family keeps the palm's normal level, and SIMPLE's grasps tilt it 10-90 deg down
        a = rng.normal(size=3)
        dev = a / np.linalg.norm(a) * math.radians(rng.uniform(WRIST_MIN_DEG, WRIST_MAX_DEG))
    cmds = [dict(side=side, target=t, dyaw=dyaw, pitch=pitch, pelvis_z=pel, wrist_dev=dev) for t in (clear, pre, entry)]
    if VIA:  # the pre-grasp along a random path, then the entry
        return [dict(cmds[1], via="random"), cmds[2]]
    return cmds


def replay_command(route: dict) -> dict:
    """REACH_VIA_REPLAY: one path of via_gen.load_routes as a reach command (heading frame, standing, right hand)."""
    return dict(side="right", target=route["goal"].copy(), dyaw=0.0, pitch=0.0, pelvis_z=None, wrist_dev=np.zeros(3),
                rot_h=route["goal_R"], via=route)  # fmt: skip


def body_pos(m, name: str) -> np.ndarray:
    import mujoco

    return m.sim.d.xpos[mujoco.mj_name2id(m.sim.m, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def hand_q(m, side: str) -> np.ndarray:
    """`side`'s finger joints as the simulator has them (zeros for a hand without these joints)."""
    import mujoco

    q = np.zeros(len(FINGERS), np.float32)
    for k, f in enumerate(FINGERS):
        j = mujoco.mj_name2id(m.sim.m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{f}_joint")
        if j >= 0:
            q[k] = m.sim.d.qpos[m.sim.m.jnt_qposadr[j]]
    return q


def grasp_commands(rng) -> list[dict]:
    """REACH_GRASP: the pre-grasp and the entry of one grasp, heading frame, right-hand side mirrored for the left."""
    if GRASP_ROOM:
        return room_commands(rng)
    side = "right" if rng.random() < 0.5 else "left"
    sd = 1.0 if side == "right" else -1.0
    x, y, z = rng.uniform(*GRASP_X), sd * rng.uniform(*GRASP_Y), rng.uniform(*GRASP_Z)
    rule = None if z >= 0.60 else float(np.clip(z + 0.05, 0.52, 0.76))
    pel = rule if rng.random() < 0.5 else (None if z >= 0.5 and rng.random() < 0.3 else float(rng.uniform(0.52, 0.76)))
    pre = np.array([x, y, z]) + rng.normal(0.0, GRASP_JITTER_M, 3)
    entry = np.array([x + rng.uniform(*GRASP_FWD), y, z]) + rng.normal(0.0, GRASP_JITTER_M, 3)
    pel2 = pel if pel is None or rng.random() < 0.7 else float(np.clip(pel + rng.choice([-0.06, 0.06]), 0.52, 0.76))
    return [
        dict(side=side, target=t, dyaw=0.0, pitch=0.0, pelvis_z=p, wrist_dev=np.zeros(3))
        for t, p in ((pre, pel), (entry, pel2))
    ]


class Recorder:
    """Samples the robot every TICKS motor ticks while `on`, by wrapping the motor's tick (Motor._run calls self.tick)."""

    def __init__(self, env: E.Env):
        self.env, self.on, self.k, self.side, self.t_cmd = env, False, 0, "right", None
        self.S, self.W, self.J, self.C = [], [], [], []
        m = env.m
        orig = m.tick

        def tick():
            ok = orig()
            if ok and self.on:
                self.k += 1
                if self.k % E.TICKS == 0:
                    self.sample()
            return ok

        m.tick = tick
        # the reference plan of the last _blend call (2026-10-31, improvement 2): reach's whole-body IK poses and leg
        # durations, so a segment can record the final reference pose the controller is asked to track
        self.plan = None
        blend = m._blend

        def plan(poses, durations):
            self.plan = (poses, durations)
            return blend(poses, durations)

        m._blend = plan

    def start(self, side: str) -> None:
        self.S, self.W, self.J, self.C = [], [], [], []
        self.side, self.k, self.t_cmd, self.on = side, 0, None, True
        self.sample()

    def sample(self) -> None:
        env, m = self.env, self.env.m
        p, Rp = env.pelvis()
        jaw, Rj = m.jaw_centre(self.side)
        cmd = Rp.T @ (m.commanded_jaw(self.side) - p) if m.last_fut else Rp.T @ (jaw - p)
        tc = -1.0 if self.t_cmd is None else float(m.t - self.t_cmd)
        self.S.append(env.self_state())
        self.W.append(env.world2())
        self.J.append(np.r_[Rj[:, 0], Rj[:, 1]].astype(np.float32))
        self.C.append(np.r_[cmd, tc].astype(np.float32))


def home(m) -> None:
    """Motor._stand_up for the whole body: the reference blends from where it is to m.stand_pose's joints (arms
    included) at the reset's pelvis height, the current pelvis xy and yaw, ankles pinned where they are."""
    from locomani.backends.scalebfm.motor import Outcome, Pose, yaw_quat

    start = m.ref(m.t)
    goal = Pose(
        np.r_[m.pelvis()[0][:2], m.stand_pose.pos[2]], yaw_quat(m.yaw()), m.stand_pose.joints.copy(), np.zeros(29)
    )
    at, total = m._blend([start, goal], [HOME_S])
    m._stand(start)
    m.ref = at
    t0 = m.t
    m._run("home", lambda t: Outcome(True) if t - t0 >= total + HOME_SETTLE_S else None, total + HOME_SETTLE_S + 1.0)


_WEIGHTS: dict = {}


def use_weights(m, which: str) -> None:
    """REACH_STUDENT: put the "student" or "frozen" weights into the policy (the frozen ones are kept at the first call,
    before any swap)."""
    import torch

    params = list(m.pol.w.named_parameters())
    if not _WEIGHTS:
        dev = params[0][1].device
        _WEIGHTS["frozen"] = {n: p.detach().clone() for n, p in params}
        sd = torch.load(STUDENT, map_location="cpu", weights_only=False)["params"]
        assert set(sd) == set(_WEIGHTS["frozen"])
        _WEIGHTS["student"] = {n: v.to(dev) for n, v in sd.items()}
    with torch.inference_mode():
        for n, p in params:
            p.copy_(_WEIGHTS[which][n])


def blend_swing(m, side: str, a, b) -> float:
    """How far the min-jerk joint blend from pose a to pose b takes `side`'s jaw off the straight segment between its ends, m."""
    from locomani.backends.scalebfm import motor as MO
    from locomani.backends.scalebfm import runtime as RT

    def jaw(pose):
        p_, q_ = m.kp(pose.pos, pose.quat, pose.joints)[f"{side}_wrist_yaw_link"]
        return p_ + RT.mat(q_) @ RT.OFF

    ja, jb = jaw(a), jaw(b)
    qb = b.quat if float(np.dot(a.quat, b.quat)) >= 0 else -b.quat
    dev = 0.0
    for u in np.linspace(0.1, 0.9, 9):
        s = MO.minjerk(u)
        q = (1 - s) * a.quat + s * qb
        mid = MO.Pose(
            a.pos + s * (b.pos - a.pos), q / np.linalg.norm(q), a.joints + s * (b.joints - a.joints), np.zeros(29)
        )
        dev = max(dev, float(np.linalg.norm(jaw(mid) - (ja + s * (jb - ja)))))
    return dev


def segment(
    env: E.Env, rec: Recorder, rng, walked: bool, chain: int, c: dict | None = None, last_cmd=None
) -> dict | None:
    m = env.m
    c = c or sample_command(rng)
    if HOME:
        home(m)
        if m.fell_at is not None:
            return None
    side = c["side"]
    rec.start(side)
    m.hold(PRE_S)
    if m.fell_at is not None:
        rec.on = False
        return None
    p, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    Rh = E.rot_z(yaw)
    target_w = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
    rot_w = E.rot_z(yaw + c["dyaw"]) @ rot_y(c["pitch"])
    if WRIST:
        rot_w = rot_w @ rotvec(c["wrist_dev"])
    if "rot_h" in c:  # a replayed path's own goal attitude
        rot_w = Rh @ c["rot_h"]
    k0 = len(rec.S) - 1
    rec.t_cmd = m.t
    rec.plan = None
    if STUDENT:
        use_weights(m, "student")
    kw: dict = {}
    if GRASP_ENTRY and chain == ENTRY_CHAIN:
        if GRASP_SPEED:
            kw["speed"] = GRASP_SPEED
        if GRASP_CONT and last_cmd is not None:
            j = m.jaw_centre(side)[0]
            kw["start"] = j + GRASP_CONT * (last_cmd - j)
        if GRASP_IKREF:
            kw["ik_from_ref"] = True
    extra: dict = {}
    if VIA:
        j0, R0 = m.jaw_centre(side)
        v = c.get("via")
        via = dict(p=np.zeros((0, 3)), R=np.zeros((0, 3, 3)), speed=None, gen=np.zeros(len(VG.GEN_KEYS), np.float32),
                   tries=0, tol=0.0)  # fmt: skip
        route, start_cm = -1, -1.0
        if isinstance(v, str):  # "random"
            sd = 1.0 if side == "right" else -1.0
            sh = body_pos(m, f"{side}_shoulder_pitch_link")
            via = VG.random_path(rng, j0, target_w, R0, rot_w, p, Rh, sd, sh, c["pelvis_z"])
        elif v is not None:  # a path of REACH_VIA_REPLAY, from its heading frame onto this one
            via = dict(via, p=VG.from_h(v["p"], p, Rh), R=np.array([Rh @ r for r in v["R"]]).reshape(-1, 3, 3),
                       speed=v["speed"])  # fmt: skip
            route = c["route"]
            start_cm = 100 * float(np.linalg.norm(VG.to_h(j0, p, Rh)[0] - v["start"]))
        if len(via["p"]):
            kw.update(via=list(via["p"]), via_rot=list(via["R"]))
        if via["speed"] is not None and v is not None:
            kw["speed"] = via["speed"]
        n = len(via["p"])
        vp, vr = np.zeros((VG.VIA_MAX, 3), np.float32), np.zeros((VG.VIA_MAX, 6), np.float32)
        if n:
            vp[:n] = VG.to_h(via["p"], p, Rh)
            vr[:n] = [np.r_[(Rh.T @ r)[:, 0], (Rh.T @ r)[:, 1]] for r in via["R"]]
        extra = dict(via_p=vp, via_r6=vr, n_via=n, via_speed=float(kw.get("speed") or m.cfg.reach_speed),
                     via_gen=via["gen"], via_tries=int(via["tries"]), via_tol=float(via["tol"]), via_route=route,
                     via_start_cm=start_cm, hand_q=hand_q(m, side))  # fmt: skip
    out = m.reach(side, target_w, rot_w, pelvis_z=c["pelvis_z"], servo=False, **kw)
    goal = rec.plan[0][-1]
    if VIA:
        poses, durations = rec.plan
        extra.update(
            RQ=np.array([q.joints for q in poses], np.float32), RP=np.array([np.r_[q.pos, q.quat] for q in poses], np.float32),
            RT=np.r_[0.0, np.cumsum(durations)].astype(np.float32), n_ref=len(poses),
        )  # fmt: skip
    swing = 100 * blend_swing(m, side, rec.plan[0][0], rec.plan[0][1]) if GRASP_ENTRY else -1.0
    reach_s = float(m.t - rec.t_cmd)
    if m.fell_at is None:
        m.hold(POST_S)
    if STUDENT:
        use_weights(m, "frozen")
    rec.on = False
    info = out.info or {}
    rot_h = Rh.T @ rot_w
    return dict(
        S=np.array(rec.S), W=np.array(rec.W), J=np.array(rec.J), C=np.array(rec.C), k0=k0,
        side=1 if side == "right" else -1, target=c["target"].astype(np.float32), dyaw=c["dyaw"], pitch=c["pitch"],
        rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32), pelvis_z=-1.0 if c["pelvis_z"] is None else c["pelvis_z"],
        start=np.r_[p, yaw].astype(np.float32), walked=walked, chain=min(chain, 1) if GRASP_ROOM else chain, homed=HOME, fell=m.fell_at is not None,
        ok=bool(out.ok), jaw_err_cm=float(info.get("jaw_err_cm", -1.0)), att_err_deg=float(info.get("attitude_err_deg", -1.0)),
        ik_err_cm=float(info.get("ik_err_cm", -1.0)), reach_s=reach_s,
        ref_q=goal.joints.astype(np.float32), ref_pel=np.r_[goal.pos, goal.quat].astype(np.float32),
        ref_s=float(sum(rec.plan[1])), wrist_dev=c["wrist_dev"].astype(np.float32), swing_cm=swing, cmd_w=target_w,
        **extra,
    )  # fmt: skip


def open_walk(m, forward: float, left: float, turn: float):
    """Motor.walk_to with the heading kept, then a turn, and the stop correction off: the reference ends where the walk
    profile ends and the motor stands after settle_s, without reading the true pelvis position."""
    import dataclasses

    yaw = m.yaw()
    c, s = math.cos(yaw), math.sin(yaw)
    goal = m.sim.d.qpos[:2] + np.array([c * forward - s * left, s * forward + c * left])
    cfg = m.cfg
    m.cfg = dataclasses.replace(cfg, stop_ki=0.0, stop_timeout=cfg.settle_s)
    try:
        return m.walk_to(goal, yaw + turn, face="fixed")
    finally:
        m.cfg = cfg


class _PlanOnly(Exception):
    pass


def plan_only(m, side: str, target_w, rot_w, pelvis_z):
    """plan_invert.plan_of without its imports: reach()'s whole-body IK plan from the current state, without moving."""
    blend, got = m._blend, {}

    def stop(poses, durations):
        got["plan"] = (poses, durations)
        raise _PlanOnly

    m._blend = stop
    try:
        m.reach(side, target_w, rot_w, pelvis_z=pelvis_z, servo=False)
    except _PlanOnly:
        pass
    finally:
        m._blend = blend
    poses, durations = got["plan"]
    g = poses[-1]
    return dict(
        ref_q=g.joints.astype(np.float32), ref_pel=np.r_[g.pos, g.quat].astype(np.float32), ref_s=float(sum(durations))
    )


def approach_segment(env: E.Env, rec: Recorder, rng, walked: bool) -> dict | None:
    """REACH_APPROACH: one walk-then-reach segment recorded from before the walk (see the flag)."""
    m = env.m
    c = sample_command(rng)
    wd = rng.uniform(*APPROACH_BOX)
    turn = float(np.radians(rng.uniform(-APPROACH_YAW_DEG, APPROACH_YAW_DEG)))
    if APPROACH_NOTURN > 0 and rng.random() < APPROACH_NOTURN:
        turn = 0.0
    side = c["side"]
    rec.start(side)
    m.hold(PRE_S)
    if m.fell_at is not None:
        rec.on = False
        return None
    p0, Rp0 = env.pelvis()
    yaw0 = E.yaw_of(Rp0)
    R0 = E.rot_z(yaw0)
    k0 = len(rec.S) - 1
    rot_b = rot_y(c["pitch"]) if not WRIST else rot_y(c["pitch"]) @ rotvec(c["wrist_dev"])  # wrist, believed frame
    # the plan this reach would get from the pre-walk posture, the body moved to where it believes it will stand
    tb = c["target"]
    pr = plan_only(
        m, side, np.r_[p0[:2] + (R0 @ np.r_[tb[:2], 0.0])[:2], tb[2]], E.rot_z(yaw0 + c["dyaw"]) @ rot_b, c["pelvis_z"]
    )
    wout = open_walk(m, float(wd[0]), float(wd[1]), turn)
    if m.fell_at is not None:
        rec.on = False
        return None
    m.hold(PRE_S)
    if m.fell_at is not None:
        rec.on = False
        return None
    p1, Rp1 = env.pelvis()
    yaw1 = E.yaw_of(Rp1)
    Rh1 = E.rot_z(yaw1)
    target_w = np.r_[p1[:2] + (Rh1 @ np.r_[tb[:2], 0.0])[:2], tb[2]]  # the body-frame command, from where it really is
    rot_w = E.rot_z(yaw1 + c["dyaw"]) @ rot_b
    k_reach = len(rec.S) - 1
    rec.t_cmd = m.t
    rec.plan = None
    out = m.reach(side, target_w, rot_w, pelvis_z=c["pelvis_z"], servo=False)
    goal = rec.plan[0][-1]
    reach_s = float(m.t - rec.t_cmd)
    if m.fell_at is None:
        m.hold(POST_S)
    rec.on = False
    info = out.info or {}
    # the reach command in the start frame through the commanded walk, and how far the walk really ended off it
    Rt = E.rot_z(turn)
    t0 = np.r_[wd + (Rt @ np.r_[tb[:2], 0.0])[:2], tb[2]]
    rot_h = Rt @ E.rot_z(c["dyaw"]) @ rot_b
    d1 = R0.T @ np.r_[p1[:2] - p0[:2], 0.0]
    werr = np.r_[d1[:2] - wd, (yaw1 - yaw0 - turn + math.pi) % (2 * math.pi) - math.pi]
    return dict(
        S=np.array(rec.S), W=np.array(rec.W), J=np.array(rec.J), C=np.array(rec.C), k0=k0,
        side=1 if side == "right" else -1, target=t0.astype(np.float32), dyaw=c["dyaw"], pitch=c["pitch"],
        rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32), pelvis_z=-1.0 if c["pelvis_z"] is None else c["pelvis_z"],
        start=np.r_[p0, yaw0].astype(np.float32), walked=walked, chain=0, homed=False, fell=m.fell_at is not None,
        ok=bool(out.ok), jaw_err_cm=float(info.get("jaw_err_cm", -1.0)), att_err_deg=float(info.get("attitude_err_deg", -1.0)),
        ik_err_cm=float(info.get("ik_err_cm", -1.0)), reach_s=reach_s,
        ref_q=pr["ref_q"], ref_pel=pr["ref_pel"], ref_s=pr["ref_s"], wrist_dev=c["wrist_dev"].astype(np.float32),
        ref_q_post=goal.joints.astype(np.float32), ref_pel_post=np.r_[goal.pos, goal.quat].astype(np.float32),
        ref_s_post=float(sum(rec.plan[1])), target_b=tb.astype(np.float32), walk=np.r_[wd, turn].astype(np.float32),
        walk_err=werr.astype(np.float32), k_reach=k_reach, walk_ok=bool(wout.ok),
    )  # fmt: skip


def worker(wid: int, ids: list[int], seed: int) -> None:
    """Every reset of `ids`. A reset whose whole-body IK raises a numerical error (scipy's least-squares SVD not
    converging; seen once in 12000 approach resets under SONIC, 2026-11-16, never under ScaleBFM) is dropped whole --
    its segments removed, its id printed and counted in skipped.json -- and the worker goes on, so one degenerate pose
    does not cost the other resets. The resets that raise nothing are recorded exactly as before."""
    env = E.Env("open")
    rec = Recorder(env)
    segs, chunk, t0 = [], 0, time.time()
    skipped: list[int] = []
    for n, i in enumerate(ids):
        n_before = len(segs)
        try:
            segs = _reset(env, rec, i, seed, segs)
        except (np.linalg.LinAlgError, ValueError) as e:
            del segs[n_before:]
            rec.on = False
            skipped.append(ID0 + i)
            print(f"w{wid} reset {ID0 + i} skipped: {type(e).__name__}: {e}", flush=True)
        if (n + 1) % 25 == 0:
            print(f"w{wid} {n + 1}/{len(ids)} resets, {len(segs)} segments, {time.time() - t0:.0f}s", flush=True)
        if len(segs) >= FLUSH:
            save(segs, OUT / f"w{wid}_{chunk:03d}.npz")
            segs, chunk = [], chunk + 1
    if segs:
        save(segs, OUT / f"w{wid}_{chunk:03d}.npz")
    json.dump(dict(wid=wid, skipped=skipped), open(OUT / f"w{wid}_skipped.json", "w"))
    print(f"w{wid} done {len(ids)} resets {time.time() - t0:.0f}s", flush=True)


def _reset(env, rec, i: int, seed: int, segs: list) -> list:
    """One reset of `worker`: its segments appended to `segs` (returned). It writes nothing to disk, so a reset that
    raises leaves no trace once `worker` drops what it appended."""
    rng = np.random.default_rng(seed + i)
    env.reset(None)
    m = env.m
    walked = bool(rng.random() < (APPROACH_DIVERSE if APPROACH else SIDESTEP_WALK if SIDESTEP else 0.35))
    cmds = [None] * CHAIN
    if VIA_REPLAY:  # the first pass from the reset's stand, later passes after the room's short walk
        r = i % len(ROUTES)
        cmds, walked = [dict(replay_command(ROUTES[r]), route=r)], False
        if i >= len(ROUTES):
            d, y0 = float(rng.uniform(0.45, 0.90)), float(rng.uniform(-0.10, 0.10))
            m.reset((-d, y0), float(math.radians(rng.uniform(-8, 8))))
            m.hold(1.0)
            open_walk(m, d, float(rng.uniform(-0.06, 0.06)), 0.0)
            walked = True
        if m.fell_at is not None:
            return segs
        m.hold(1.0)
    elif GRASP:
        walked = False  # the walk below is this flag's own
        cmds = grasp_commands(rng)
        if GRASP_ROOM:  # as room_grasp.episode: a short open walk, the heading off by what it was off at the start
            d, y0 = float(rng.uniform(0.45, 0.90)), float(rng.uniform(-0.10, 0.10))
            m.reset((-d, y0), float(math.radians(rng.uniform(-8, 8))))
            m.hold(1.0)
            open_walk(m, d, float(rng.uniform(-0.06, 0.06)), 0.0)
        else:
            m.reset(
                (-float(rng.uniform(*GRASP_WALK)), float(rng.uniform(-0.2, 0.2))),
                float(math.radians(rng.uniform(-15, 15))),
            )
            m.hold(1.0)
            m.walk_to(np.zeros(2), yaw=0.0)
        if m.fell_at is not None:
            return segs
        m.hold(0.5)
        walked = True
    if walked and not GRASP and not VIA_REPLAY:
        ang, d = rng.uniform(-math.pi, math.pi), rng.uniform(0.3, 1.2)
        m.walk_to(np.array([d * math.cos(ang), d * math.sin(ang)]), yaw=float(rng.uniform(-math.pi, math.pi)))
        if m.fell_at is not None:
            return segs
    asked, n_side = np.zeros(2, np.float32), 0
    if APPROACH:
        s = approach_segment(env, rec, rng, walked)
        if s is not None:
            s["reset"], s["sidestep"], s["n_sidesteps"] = ID0 + i, asked, n_side
            segs.append(s)
        return segs
    if SIDESTEP:
        for _ in range(2 if rng.random() < SIDESTEP_TWICE else 1):
            m.hold(PRE_S)
            d = rng.uniform(*SIDESTEP_BOX)
            if m.fell_at is None:
                m.sidestep(left=float(d[1]), forward=float(d[0]))
            asked, n_side = asked + d.astype(np.float32), n_side + 1
        if m.fell_at is not None:
            return segs
        walked = True
    last_cmd = None
    for k in range(len(cmds)):
        s = segment(env, rec, rng, walked, k, cmds[k], last_cmd)
        if s is None:
            break
        last_cmd = s["cmd_w"]
        s["reset"], s["sidestep"], s["n_sidesteps"] = ID0 + i, asked, n_side
        segs.append(s)
        if s["fell"]:
            break
    return segs


def save(segs: list[dict], path: Path) -> None:
    pack = {k: np.concatenate([s[k] for s in segs]) for k in ("S", "W", "J", "C")}
    pack["n_steps"] = np.array([len(s["S"]) for s in segs])
    for k in ("k0", "side", "dyaw", "pitch", "pelvis_z", "walked", "chain", "homed", "fell", "ok", "jaw_err_cm",
              "att_err_deg", "ik_err_cm", "reach_s", "reset", "ref_s", "n_sidesteps"):  # fmt: skip
        pack[k] = np.array([s[k] for s in segs])
    for k in ("target", "rot6", "start", "ref_q", "ref_pel", "sidestep", "wrist_dev"):
        pack[k] = np.stack([s[k] for s in segs])
    if GRASP_ENTRY:
        pack["swing_cm"] = np.array([s["swing_cm"] for s in segs], np.float32)
    if "via_p" in segs[0]:  # REACH_VIA, REACH_VIA_REPLAY
        for k in ("n_via", "via_speed", "via_tries", "via_tol", "via_route", "via_start_cm", "n_ref"):
            pack[k] = np.array([s[k] for s in segs])
        for k in ("via_p", "via_r6", "via_gen", "hand_q"):
            pack[k] = np.stack([s[k] for s in segs])
        for k in ("RQ", "RP", "RT"):
            pack[k] = np.concatenate([s[k] for s in segs])
    if "walk" in segs[0]:  # REACH_APPROACH
        for k in ("ref_s_post", "k_reach", "walk_ok"):
            pack[k] = np.array([s[k] for s in segs])
        for k in ("ref_q_post", "ref_pel_post", "target_b", "walk", "walk_err"):
            pack[k] = np.stack([s[k] for s in segs])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack)


def run(procs: int, resets: int, seed: int, gpus: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if VIA_REPLAY:
        json.dump(dict(spec=VIA_REPLAY, names=[r["name"] for r in ROUTES]), open(OUT / "routes.json", "w"))
    cards = gpus.split(",")
    jobs = []
    for w in range(procs):
        envv = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[w % len(cards)], OMP_NUM_THREADS="1")
        ids = list(range(w, resets, procs))
        log = open(OUT / f"w{w}.log", "w")
        cmd = [
            str(E.SIM_PY),
            __file__,
            "worker",
            "--wid",
            str(w),
            "--seed",
            str(seed),
            "--ids",
            ",".join(map(str, ids)),
        ]
        jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    codes = [j.wait() for j in jobs]
    json.dump(dict(procs=procs, resets=resets, seed=seed, codes=codes), open(OUT / "run.json", "w"))
    print("exit codes", codes)


def merge() -> None:
    zs = [dict(np.load(p)) for p in sorted(OUT.glob("w[0-9]*_[0-9]*.npz"))]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
    np.savez_compressed(OUT / "all.npz", **out)
    print(f"{len(out['n_steps'])} segments, {int(out['n_steps'].sum())} steps, fell {int(out['fell'].sum())}, "
          f"walked {int(out['walked'].sum())}, ok {int(out['ok'].sum())}")  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "merge"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--resets", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=3000000)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args(argv)
    if a.what == "run":
        run(a.procs, a.resets, a.seed, a.gpus)
    elif a.what == "worker":
        worker(a.wid, [int(x) for x in a.ids.split(",") if x], a.seed)
    else:
        merge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
