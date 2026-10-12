"""TODO 21.3.4 / 23.2: `grasp_planned`, a single-arm grasp primitive that takes its grasp from one of two sources and its
arm path from one of two planners, executed the same way on the SONIC pose stream.

Grasp source (`GraspPlanned(source=...)`, run_grasp.py --grasp-source, or a plan step's `source` argument):
- `cache` (default since 2026-10-12): the scene's offline grasp cache (grasp_cache.py: antipodal grasps on the
  object's mesh, each closed, lifted and shaken in a hand-only MuJoCo model), carried to the measured object pose;
  only the validated ones, ordered by tier, then validation score. Needs <scene>/grasp_cache.npz (cache_all.sh).
- `graspnet`: graspnet-baseline on the ego depth image (steps 1-2 below). Needs graspnet_server.py listening
  (install_graspnet.sh, servers.sh start graspnet).
Why the cache is the default, with the numbers: docs/object_grasp_design.md.

One call, from where the round's spawn left the robot (at the table edge, arms hanging):
1. perception (graspnet source): the ego (head) camera's depth image of this exact state, rendered by MuJoCo from the
   privileged qpos, cropped to the table around the target and sent to graspnet_server.py (graspnet-baseline,
   RealSense weights);
2. candidates (graspnet source): graspnet's grasps (approach = Dex1 palm x, closing = palm y) whose width fits the jaw, with the Dex1
   pushed in along the approach as deep as graspnet's depth allows while its open hand (hand_cloud.py) clears the table top and the target's points, and the jaw has target points between its fingers; both 180-degree
   rolls about the approach; graspnet's score order;
3. planner, the only thing that differs between the two arms of the comparison:
   - `ik` (ours): the simple expert's staged hand path -- up beside the body, over to above the pre-grasp, down to the
     pre-grasp 10 cm behind the grasp, straight in -- solved by ArmIK (right arm + waist, the expert's joint limits)
     every centimetre, the first candidate whose path solves within 1.5 cm / 10 deg and whose hand clears the table
     and the target up to the pre-grasp;
   - `curobo` (SIMPLE's): curobo_server.py, one batched MotionGen plan to all candidates' pre-grasps with the table and
     the target as obstacles, the shortest success, then the same straight entry and lift as CuRobo IK steps;
The arm is taken over (StreamedArm, as the simple expert takes it) BEFORE perception and planning, and the pose stream
holds the robot while they run: under the planner's IDLE the robot drifted 7 cm during a 20 s CuRobo plan and the
executed grasp landed 8 cm off (object 002, 2026-10-08).
4. execution (shared): the planner's joint path to the pre-grasp, retimed to 0.6 rad/s and streamed open loop on the
   physics clock through StreamedArm as the simple expert streams (PosePreview, reference velocities); from there one
   closed-loop tool servo for both -- settle on the pre-grasp, straight entry, seat, close the jaw, lift 10 cm, hold
   1.5 s (ArmIK against the measured pelvis each tick plus an integrated tracking offset: open loop the tool landed
   4-8 cm off the grasp and the jaw closed on air, object 002, 2026-10-08). The planners' own entry and lift paths are
   recorded, not executed. Held when the target rose >= 5 cm and touches the right hand.
The chosen grasp is shown as the ghost hand in the round's video. Everything is written to the round's directory.

Fixed grasp (`fixed=True`, run_grasp.py --fixed-grasp; user 2026-10-09): above, each round perceives again and each
planner picks its own grasp, so the two rounds of an object never aimed at the same grasp (0 of 52 objects, median 1.9
cm and 61 deg apart) and the comparison was not of the planners alone. Here round 0 perceives once, plans both
planners to each of the top MAX_CANDIDATES candidates (ours one by one, CuRobo as one batch) and fixes the first
candidate in rank order that both reach -- else the first that either reaches, `common` false -- in
<out_dir>/target.json. Round 0 executes ours to it; round 1 skips perception and plans CuRobo to the same grasp,
carried with the target's pose since round 0. Both rounds stand by the same rule (STAND_FORWARD behind and
STAND_RIGHT beside the target along the robot's own heading), so the target sits at the same place in the pelvis frame;
only the heading differs, by the spawn's settle (2-5.5 deg on the first objects), recorded as align.yaw. Not the same
world spot: with the heading off by d, that moves the target ~0.40 m x d in the pelvis frame (3.8 cm at 5.5 deg).
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[3]
# the right Dex1 with the jaw fully open (hand_cloud.py), tool frame; outputs/scene/hand_points_dex1.npy is the ghost's
# half-open jaw and rejected every grasp wider than 4.6 cm
HAND_POINTS = np.load(REPO / "outputs" / "graspnet_bottle_20261008" / "hand_open_points.npy")
JAW_MAX = 0.085  # graspnet width accepted: the Dex1 opens 9.5 cm
PREGRASP_BACK = 0.10
TABLE_CLEAR, OBJECT_CLEAR = 0.005, 0.004
CROP_XY, CROP_TOP = 0.22, 0.40
MAX_CANDIDATES = 40
JOINT_SPEED, MIN_SEGMENT_S = 0.6, 0.4
LIFT_M, HOLD_S, LIFT_S = 0.10, 1.5, 2.0
SERVO_GAIN, SERVO_CAP = 3.0, 0.12  # /s and m: the integrated tracking offset (the reach-envelope study's cap 0.10-0.15)
SETTLE_S, SEAT_S, ENTRY_SPEED = 1.2, 0.6, 0.06
# The entry advance: half the expert's dash (0.12 m/s, tuned on a 4.5 cm bottle with 2.5 cm of jaw clearance a side);
# these targets are wider and the fingers knocked them over at the full dash (objects 002 and 010, 2026-10-08).
DASH_SPEED = 0.06
SERVO_STEP_RAD = 0.02  # per physics tick (1 rad/s): an IK branch jump flung a held can off the table (2026-10-08)
# Waist limits of both planners and the servo: StreamedArm's own (yaw +-0.35, roll +-0.15, pitch -0.05..), not the
# model's +-2.6 rad yaw, which let the solver swing the torso.
# Narrowed again after CuRobo's first plans parked the waist on all three of those limits (yaw 0.35, roll 0.15, pitch
# 0.20) and SONIC delivered the hand 13 cm off (object 002, 2026-10-08): roll is held near zero, yaw within 0.25.
WAIST_LIMITS = {12: (-0.25, 0.25), 13: (-0.05, 0.05), 14: (-0.05, None)}
# Where the robot stands for the grasp: the target this far ahead of the pelvis and to its right. Spawn's 4-5 s of
# planner IDLE drifted the robot 0.11-0.30 m back (object 0.46-0.54 m ahead, out of reach), and the planner kept
# drifting it ~0.1 m in the 1-2 s before the takeover (object 002, 2026-10-08); so once the pose stream holds the
# robot, the sim's align_robot_xy puts the settled pelvis here (the crate wall line does the same).
STAND_FORWARD, STAND_RIGHT = 0.38, 0.12
IK_POS_TOL, IK_ROT_TOL_DEG = 0.015, 10.0
# The IK planner's pull toward the start posture (ArmIK's nullspace prior), as CuRobo regularises toward its retract
# configuration. Without it the staged path ended with the wrist roll on its limit (-1.96 of -1.97) or the shoulder
# rolled out to -2.0, SONIC lagged 6-14 cm behind the entry and the fingers knocked the targets over (objects 000-003,
# 2026-10-08).
# 0.1 toward the hanging start held the hand 4-5 cm short of the raise on every candidate (objects 002 and 005), so
# the pull is toward a neutral arm (READY_ARM: wrist and shoulder yaw at zero, waist straight) and much weaker.
POSTURE_WEIGHT = 0.01
PLANNER_ORDER = ("ik", "curobo")  # round r runs PLANNER_ORDER[r % 2]; set by run_grasp.py
ROUND = {"n": 0}
# Where the candidates come from (module docstring): the default of GraspPlanned(source=...), which a plan step's
# `source` argument overrides.
GRASP_SOURCES = ("cache", "graspnet")
CACHE_FILE = "grasp_cache.npz"  # in the scene directory, next to scene_locomani.xml (grasp_cache.py writes it)
# TODO 23.2.6: before the staged IK path, order the candidates by one direct IK solve each at the pre-grasp and the
# grasp (from the neutral posture): the ones that reach both within the planner's tolerance first, by joint-limit
# margin, then the rest by error; the tier stays the primary key. Each candidate the staged path rejects costs ~1.8 s
# (median 8 rejected per round, plan median 20 s in the cache acceptance, 2026-10-10); the direct solves cost ~20 ms.
# key: feasible ones in the source order ("score") or by joint-limit margin ("margin"; replay: no gain).
# staged: also solve the raise point, the point above the pre-grasp and the descent from it in DESCENT_STEPS, as the
# path does -- the direct solve at the pre-grasp from the neutral posture finds a branch the descent from above does
# not reach (41/168 replayed rounds still took > 20 s with the direct presort, their rejections 534 at leg 2).
# tier_first: the tier stays the primary key (False: feasibility first, so a feasible tier-1 grasp precedes an
# infeasible tier-0 one).
# Replay of the 168 cache rounds (replay_plan.py, 2026-10-10): off 9.6 s median / 43.6 p90; direct 4.0 / 40.3; staged
# 4.5 / 41.3; staged + feasibility first 3.6 / 5.6 (max 24 s) -- the same grasp chosen in every round.
PRESORT = {"on": True, "key": "score", "staged": True, "tier_first": False}
# TODO 23.4.5 (user 2026-10-10): the decoupled right arm. The expert's joint targets go to SONIC as the streamed
# reference as before (so its balance sees the arm move), AND every tick to the sim-side arm pin
# (EpisodeControlClient.pin_arms -> run_sim_locomani's compute_body_torques hook, kp 120 / kd 3 / |tau| <= 30), so
# the right arm (and with "waist" the waist trio) follows the IK solution like motors in position mode instead of
# the tracking policy's rendition of it. Four batches showed the hand drifting 3-5 cm off its own command during
# the entry on low, tilted grasps (handoff section 9). Released when the primitive finishes.
# waist: False (policy), "yawroll" (TODO 23.4.6.2: yaw + roll pinned with the controller's gains, pitch left to the
# policy -- the full waist pin delivered the IK's lean and tripped the support gate), True (all three).
DECOUPLE = {"on": True, "waist": False}  # default on (user 2026-10-10, after the two-batch pair: knocks 35 -> 22)
# TODO 23.4.6.1: freeze the closed-loop IK's virtual base (WholeBodyIK base_range) so the joints alone place the
# hand; the decomposition put ~3.2 cm of the entry's lateral error on base motion SONIC did not deliver.
# Alone it diverged: with the base frozen and the waist capped the low targets are unreachable, the tracking
# feedback and the entry's lateral shift wind up (013: target ran 108 cm away, 2026-10-10). Off unless asked.
BASE_FREEZE = {"on": False, "range": (0.01, 0.01, 0.05)}
# With the waist pinned the IK's lean is delivered in full: at the expert's waist_pitch_max 0.52 rad the torso went
# to 32 deg and SONIC's support gate tripped (the policy had been delivering about a third). Cap it near that.
WAIST_PITCH_MAX_PINNED = 0.2
# TODO 23.4.4 (user 2026-10-10): the two entry-side levers left after the three control-side ones (handoff section 12).
# ENTRY_GATE: the advance along the approach waits while the measured grip offset across the palm (jaw axis, pad
# height) exceeds lateral_m, so the lateral shift correction (3 cm/s, cap 4 cm) gets its time before the pads reach
# the object; at most hold_s of waiting per round, then the advance resumes regardless (entry_timeout_s stays 6 s).
ENTRY_GATE = {"on": True, "lateral_m": 0.015, "hold_s": 2.0}  # default on (user 2026-10-12: 65/129 vs 100/260)
# PREFER_HIGH: the cached candidates are ordered tier, then tool height above the table in bin_m bins (higher first),
# then validation score -- the low grasps are the ones whose reach needs the base lean SONIC under-delivers.
PREFER_HIGH = {"on": False, "bin_m": 0.02}
# The IK has no self-collision: with the arm pinned its commands put the right shoulder roll at +0.3..+0.9 rad (arm
# swung into the torso: elbow 8 cm, shoulder-yaw link 9 cm inside it at the close, object 000, 2026-10-10). The policy
# never delivered more than ~0.1 and the servo lived with the hand that produced. Cap the inward roll.
RIGHT_SHOULDER_ROLL_MAX = 0.05
DESCENT_STEPS = 4
PRESORT_ITERS = 60


def cached_candidates(scene_dir, obj_pos, obj_quat, table_top, heading):
    """The scene's grasp cache at the measured object pose -> the candidates list of `candidates`, and the object's
    surface points in the world (the IK planner's collision tree)."""
    z = np.load(Path(scene_dir) / CACHE_FILE)
    R_now = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
    obj_pos = np.asarray(obj_pos, dtype=float)
    surface = z["surface"].astype(float) @ R_now.T + obj_pos
    out = []
    for i in np.nonzero(z["held"])[0]:
        tool = R_now @ z["tool"][i] + obj_pos
        R = R_now @ z["R"][i]
        hand = tool + HAND_POINTS @ R.T
        if hand[:, 2].min() < table_top + TABLE_CLEAR:
            continue  # the cache's table check was at the rest attitude; the spawn may have tilted the object
        out.append(
            dict(
                score=float(z["score"][i]), width=float(z["width"][i]), tool=tool, R=R, row=int(i),
                depth_used=float(z["depth"][i]), slip=float(z["slip"][i]), force=z["force"][i].tolist(),
            )
        )  # fmt: skip
    for c in out:
        c["tier"] = _tier(c, heading)
        c["bin"] = int((c["tool"][2] - table_top) // PREFER_HIGH["bin_m"]) if PREFER_HIGH["on"] else 0
    out.sort(key=lambda c: (c["tier"], -c["bin"], -c["score"]))
    return out[: 2 * MAX_CANDIDATES], surface, dict(cached=int(len(z["held"])), held=int(z["held"].sum()), fit=len(out))


def pose7(p, R) -> list:
    x, y, z, w = Rotation.from_matrix(np.asarray(R)).as_quat()
    return [*map(float, p), float(w), float(x), float(y), float(z)]


class DepthCamera:
    """The ego camera's depth of a given qpos, from the run's own scene model."""

    def __init__(self, scene_xml: str, camera: str = "head_camera", width: int = 640, height: int = 480):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.cam = self.model.camera(camera).id
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.enable_depth_rendering()
        self.camera = camera
        f = 0.5 * height / np.tan(np.radians(self.model.cam_fovy[self.cam]) / 2)
        self.K = (f, f, (width - 1) / 2, (height - 1) / 2)

    def capture(self, qpos):
        """(points in the OpenCV camera frame, R world<-cv, t world<-cv)."""
        self.data.qpos[:] = qpos
        self.mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        depth = self.renderer.render()
        fx, fy, cx, cy = self.K
        v, u = np.nonzero((depth > 0.05) & (depth < 2.0))
        z = depth[v, u]
        pts = np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z]
        R = self.data.cam_xmat[self.cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # MuJoCo cam: -z forward, y up
        return pts.astype(np.float32), R, self.data.cam_xpos[self.cam].copy()


def graspnet_alive(port: int, timeout_s: float = 3.0) -> bool:
    """Whether graspnet_server.py answers on `port` (its `ping` request; the network is not run)."""
    import zmq

    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://127.0.0.1:{port}")
    buf = io.BytesIO()
    np.savez(buf, ping=np.array(1))
    try:
        sock.send(buf.getvalue())
        return "pong" in np.load(io.BytesIO(sock.recv())).files
    except zmq.Again:
        return False
    finally:
        sock.close()


def query_graspnet(port: int, cloud_cv: np.ndarray) -> np.ndarray:
    import zmq

    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://127.0.0.1:{port}")
    buf = io.BytesIO()
    np.savez(buf, cloud=cloud_cv)
    sock.send(buf.getvalue())
    rep = np.load(io.BytesIO(sock.recv()))
    sock.close()
    if str(rep["error"]):
        raise RuntimeError(f"graspnet: {rep['error']}")
    return np.asarray(rep["grasps"])


def query_curobo(port: int, req: dict, timeout_s: float = 240.0) -> dict:
    import zmq

    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://127.0.0.1:{port}")
    sock.send_string(json.dumps(req))
    rep = json.loads(sock.recv())
    sock.close()
    return rep


SIDE_ELEVATION_DEG, SIDE_YAW_DEG, LEVEL_DEG = 25.0, 75.0, 30.0
# the open Dex1's pad faces in the tool frame (hand_cloud.py): x -0.02..0.033 along the approach, z +-0.014, faces at
# y +-0.0475
PAD_X, PAD_Z, PAD_Y, PAD_OVERLAP = (-0.02, 0.033), 0.014, 0.0475, 0.015


def candidates(grasps, R_wc, t_wc, world_pts, table_top, obj_pos, heading):
    """Graspnet rows -> Dex1 tool poses (world): [(score, width, tool_pos, R_palm, row index)], score order."""
    obj_pts = world_pts[
        (world_pts[:, 2] > table_top + 0.004) & (np.linalg.norm(world_pts[:, :2] - obj_pos[:2], axis=1) < 0.15)
    ]
    tree = cKDTree(obj_pts) if len(obj_pts) else None
    out = []
    for i, g in enumerate(grasps):
        score, width, depth = float(g[0]), float(g[1]), float(g[3])
        if width > JAW_MAX or tree is None:
            continue
        Rg = R_wc @ g[4:13].reshape(3, 3)
        tg = R_wc @ g[13:16] + t_wc
        if np.linalg.norm(tg[:2] - obj_pos[:2]) > 0.12:
            continue
        for flip in (np.eye(3), np.diag([1.0, -1.0, -1.0])):
            R = Rg @ flip
            approach = R[:, 0]
            for a in np.arange(depth / 2, -0.0201, -0.005):
                tool = tg + a * approach
                hand = tool + HAND_POINTS @ R.T
                if hand[:, 2].min() < table_top + TABLE_CLEAR:
                    continue
                if tree.query(hand, k=1)[0].min() < OBJECT_CLEAR:
                    continue
                local = (obj_pts - tool) @ R  # palm frame: what the pads will close on
                inside = (
                    (np.abs(local[:, 1]) < PAD_Y)
                    & (np.abs(local[:, 2]) < PAD_Z)
                    & (local[:, 0] > PAD_X[0])
                    & (local[:, 0] < PAD_X[1])
                )
                # enough of the target between the pads, and along at least PAD_OVERLAP of their length: an edge
                # pinched by the fingertips (0.5 cm of a lying box's rim) let go at the lift (object 000, 2026-10-08)
                if inside.sum() < 20 or np.ptp(local[inside, 0]) < PAD_OVERLAP:
                    continue
                # the whole straight entry from the pre-grasp must keep the open hand off the target and the table
                # (graspnet checks its own, smaller gripper 5 cm back; a Dex1 palm swept through a target's top)
                swept = (
                    tool[None, None, :]
                    - np.linspace(PREGRASP_BACK, 0.0, 11)[:, None, None] * approach
                    + (HAND_POINTS @ R.T)[None]
                )
                swept = swept.reshape(-1, 3)
                if swept[:, 2].min() < table_top + TABLE_CLEAR or tree.query(swept, k=1)[0].min() < OBJECT_CLEAR:
                    continue  # a shallower grasp along the same line may still clear
                out.append(dict(score=score, width=width, tool=tool, R=R, row=i, depth_used=float(a)))
                break

    # The grasp module's ranking, the same for both planners. Tier 0 is the attitude the simple expert's wrap grasp is
    # calibrated in and SONIC delivers: approach at most SIDE_ELEVATION_DEG below horizontal and within SIDE_YAW_DEG of
    # the robot's heading, fingers level (closing axis within LEVEL_DEG of horizontal). Tilted and top-down grasps
    # twisted the wrist near its limits and the hand drifted 5-20 cm off the held command while the jaw closed
    # (objects 002 and 040, 2026-10-08), so they come only after (tier 1: approach within 45 deg, fingers level;
    # tier 2: the rest), each tier by graspnet score.
    for c in out:
        c["tier"] = _tier(c, heading)
    out.sort(key=lambda c: (c["tier"], -c["score"]))
    return out[: 2 * MAX_CANDIDATES], obj_pts


def _tier(c, heading):
    a, closing = c["R"][:, 0], c["R"][:, 1]
    elev = np.degrees(np.arcsin(np.clip(-a[2], -1, 1)))
    yaw_dev = np.degrees(np.arccos(np.clip(a[:2] @ heading / max(np.linalg.norm(a[:2]), 1e-9), -1, 1)))
    level = np.degrees(np.arcsin(min(1.0, abs(closing[2])))) <= LEVEL_DEG
    if level and elev <= SIDE_ELEVATION_DEG and yaw_dev <= SIDE_YAW_DEG:
        return 0
    return 1 if level and elev <= 45.0 and yaw_dev <= SIDE_YAW_DEG else 2


def retime(path: np.ndarray, speed: float = JOINT_SPEED) -> tuple[np.ndarray, np.ndarray]:
    """Waypoint times for a joint path at `speed` rad/s on the largest joint, smoothstep over the whole path."""
    path = np.asarray(path, dtype=float)
    seg = np.r_[0.0, np.cumsum(np.abs(np.diff(path, axis=0)).max(axis=1))]
    total = max(MIN_SEGMENT_S, 1.5 * seg[-1] / speed)  # smoothstep peaks at 1.5x the mean speed
    return path, seg / max(seg[-1], 1e-9), total


def sample(path, s_of_wp, total, t):
    u = min(1.0, max(0.0, t / total))
    s = 3 * u * u - 2 * u**3
    ds = (6 * u - 6 * u * u) / total
    k = int(np.clip(np.searchsorted(s_of_wp, s, side="right") - 1, 0, len(path) - 2))
    span = max(s_of_wp[k + 1] - s_of_wp[k], 1e-9)
    w = (s - s_of_wp[k]) / span
    q = path[k] + w * (path[k + 1] - path[k])
    v = (path[k + 1] - path[k]) / span * ds
    return q, v


class GraspPlanned:
    name = "grasp_planned"

    def __init__(
        self, profiles, graspnet_port: int, curobo_port: int, out_dir: Path, fixed: bool = False, source: str = "cache"
    ):
        from locomani.skills.session import ArmOwner

        if source not in GRASP_SOURCES:
            raise ValueError(f"grasp source {source!r}: one of {GRASP_SOURCES}")
        self._profiles, self.gn_port, self.cu_port, self.out_dir = profiles, graspnet_port, curobo_port, out_dir
        self.fixed = fixed
        self.source = source
        self.requires = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})
        self.yields = ArmOwner.POSE_STREAM
        self._camera = None

    def preconditions(
        self, session, *, target: str = "water_bottle", profile: str = "bottle_simple", planner=None, source=None
    ):
        """The grasp source is ready: the scene has its cache, or graspnet_server.py answers."""
        from locomani import paths
        from locomani.skills.contract import Violation

        source = source or self.source
        if source not in GRASP_SOURCES:
            return [Violation("grasp_source", measured=source, note=f"one of {', '.join(GRASP_SOURCES)}")]
        if source == "cache":
            cache = Path(paths.SCENE_XML).parent / CACHE_FILE
            if not cache.exists():
                return [
                    Violation(
                        "grasp_cache",
                        measured=str(cache),
                        note="no grasp cache for this scene: build it with grasp_cache.py <object id> "
                        "(cache_all.sh for every object), or use the graspnet source",
                    )
                ]
        elif not graspnet_alive(self.gn_port):
            return [
                Violation(
                    "graspnet_server",
                    measured=f"127.0.0.1:{self.gn_port}",
                    note="graspnet_server.py does not answer: servers.sh start graspnet (install_graspnet.sh once)",
                )
            ]
        return []

    # ------------------------------------------------------------------ helpers
    def _keepalive(self, arm, work):
        """Run `work()` in a thread while the pose stream holds the robot (the deploy drops a silent channel in 1 s)."""
        box = {}

        def run():
            try:
                box["value"] = work()
            except Exception as exc:  # noqa: BLE001 -- re-raised in the caller's thread
                box["error"] = exc

        th = threading.Thread(target=run, daemon=True)
        th.start()
        while th.is_alive():
            arm.hold(0.2)
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _ik_setup(self, cfg):
        from locomani import paths
        from locomani.primitives.arm_ik import ArmIK
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        ik = ArmIK(paths.SCENE_XML)
        physical = ik.model.jnt_range[ik.jids]
        ik.lo, ik.hi = physical[:, 0].copy(), physical[:, 1].copy()
        ik.hi[14] = min(ik.hi[14], cfg.waist_pitch_max)
        if DECOUPLE["on"] and DECOUPLE["waist"] is True:
            ik.hi[14] = min(ik.hi[14], WAIST_PITCH_MAX_PINNED)
        if DECOUPLE["on"]:
            ik.hi[23] = min(ik.hi[23], RIGHT_SHOULDER_ROLL_MAX)
        for j, (lo, hi) in WAIST_LIMITS.items():
            ik.lo[j] = max(ik.lo[j], lo) if lo is not None else ik.lo[j]
            ik.hi[j] = min(ik.hi[j], hi) if hi is not None else ik.hi[j]
        self.active = [12, 13, 14, 22, 23, 24, 25, 26, 27, 28]
        self.active_names = [MUJOCO_JOINT_NAMES[i] for i in self.active]
        return ik

    def _plan_ik(self, ik, q0, pel_R, pel_t, cands, table_top, obj_pts, offset, log):
        """Ours: the staged hand path, solved every centimetre; the first candidate that solves and clears."""
        from locomani.backends.sonic.primitives_tuning import READY_ARM

        tree = cKDTree(obj_pts)
        tool0, R0 = self._tool_fk(ik, q0, offset)
        neutral = q0.copy()
        neutral[12:15] = 0.0
        neutral[22:29] = READY_ARM["right"]
        tried = []
        order = list(range(len(cands[: 2 * MAX_CANDIDATES])))
        presort = None
        if PRESORT["on"]:
            order, presort = self._presort(ik, q0, pel_R, pel_t, [cands[i] for i in order], offset, neutral, table_top)
        for ci in order:
            c = cands[ci]
            Rw, gw = c["R"], c["tool"]
            pre_w = gw - PREGRASP_BACK * Rw[:, 0]
            # pelvis frame
            to_p = lambda x: pel_R.T @ (x - pel_t)  # noqa: E731
            R = pel_R.T @ Rw
            grasp, pre = to_p(gw), to_p(pre_w)
            top_p = to_p(np.array([pel_t[0], pel_t[1], table_top]))[2]
            height = max(pre[2], top_p + 0.22)
            up = tool0.copy()
            up[0] += 0.05
            up[2] = max(height, tool0[2])
            over = pre.copy()
            over[2] = height
            legs = [(tool0, up), (up, over), (over, pre), (pre, grasp)]
            q, path, ok, why = q0.copy(), [q0[self.active].copy()], True, ""
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R0, R]))
            for li, (a, b) in enumerate(legs):
                n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.01)))
                for k in range(1, n + 1):
                    u = k / n
                    pos = a + u * (b - a)
                    # the hand turns to the grasp attitude while it rises and travels over (half each), then holds it
                    rot = slerp(0.5 * u) if li == 0 else slerp(0.5 + 0.5 * u) if li == 1 else slerp(1.0)
                    qn, perr, rerr = ik.solve(
                        "right", pos, rot.as_matrix(), q, iters=80, rot_weight=1.0, use_waist=True, point_offset=offset,
                        posture=neutral, posture_weight=POSTURE_WEIGHT,
                    )  # fmt: skip
                    final_leg = li >= 2
                    if final_leg and (perr > IK_POS_TOL or np.degrees(rerr) > IK_ROT_TOL_DEG):
                        ok, why = False, f"leg {li} step {k}: {perr * 100:.1f} cm {np.degrees(rerr):.0f} deg"
                        break
                    if not final_leg and perr > 0.04:
                        ok, why = False, f"leg {li} step {k}: {perr * 100:.1f} cm"
                        break
                    if np.abs(qn[self.active] - q[self.active]).max() > 0.35:
                        ok, why = False, f"leg {li} step {k}: joint jump"
                        break
                    if li < 3:  # up to the pre-grasp the hand must clear the table and the target
                        Rw_now = pel_R @ rot.as_matrix()
                        hand = pel_R @ pos + pel_t + HAND_POINTS @ Rw_now.T
                        if hand[:, 2].min() < table_top + TABLE_CLEAR and li >= 1:
                            ok, why = False, f"leg {li} step {k}: hand under the table top"
                            break
                        if tree.query(hand, k=1)[0].min() < OBJECT_CLEAR:
                            ok, why = False, f"leg {li} step {k}: hand into the target"
                            break
                    q = qn
                    path.append(q[self.active].copy())
                if not ok:
                    break
            if not ok:
                tried.append(dict(candidate=ci, why=why))
                continue
            entry_start = len(path) - int(np.ceil(np.linalg.norm(grasp - pre) / 0.01))
            lift = []
            qa = q.copy()
            for k in range(1, int(round(LIFT_M / 0.01)) + 1):
                qa, perr, rerr = ik.solve(
                    "right", grasp + [0, 0, 0.01 * k], R, qa, iters=60, rot_weight=1.0, use_waist=True, point_offset=offset,
                    posture=neutral, posture_weight=POSTURE_WEIGHT,
                )  # fmt: skip
                lift.append(qa[self.active].copy())
            log(f"[grasp_planned] ik: candidate {ci} after {len(tried)} rejected")
            return dict(
                ok=True,
                index=ci,
                approach=np.array(path[:entry_start]),
                entry=np.array(path[entry_start - 1 :]),
                lift=np.array(lift),
                rejected=tried,
                presort=presort,
            )
        return dict(ok=False, rejected=tried, presort=presort)

    def _presort(self, ik, q0, pel_R, pel_t, cands, offset, neutral, table_top):
        """Candidate indices in the order the staged path should try them (PRESORT above), and what it cost."""
        t0 = time.time()
        rot_tol = np.radians(IK_ROT_TOL_DEG)
        lo, hi = ik.lo[self.active], ik.hi[self.active]
        rows = []
        tool0, R0 = self._tool_fk(ik, q0, offset)  # the path starts from the hanging hand, not the neutral posture
        top_p = (pel_R.T @ (np.array([pel_t[0], pel_t[1], table_top]) - pel_t))[2]

        def solve(pos, rot, q):
            return ik.solve(
                "right", pos, rot, q, iters=PRESORT_ITERS, rot_weight=1.0, use_waist=True, point_offset=offset,
                posture=neutral, posture_weight=POSTURE_WEIGHT,
            )  # fmt: skip

        for ci, c in enumerate(cands):
            R = pel_R.T @ c["R"]
            grasp = pel_R.T @ (c["tool"] - pel_t)
            pre = grasp - PREGRASP_BACK * R[:, 0]
            q, feasible, err = q0, True, 0.0
            if PRESORT["staged"]:
                height = max(pre[2], top_p + 0.22)
                up = tool0.copy()
                up[0] += 0.05
                up[2] = max(height, tool0[2])
                over = pre.copy()
                over[2] = height
                slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R0, R]))
                q, pe, re = solve(up, slerp(0.5).as_matrix(), q)
                feasible &= pe <= 0.04
                q, pe, re = solve(over, R, q)
                feasible &= pe <= 0.04
                err = pe
                for k in range(1, DESCENT_STEPS + 1):
                    q, pe, re = solve(over + (pre - over) * k / DESCENT_STEPS, R, q)
                    err = max(err, pe)
                    feasible &= pe <= IK_POS_TOL and re <= rot_tol
                qa = q
            else:
                qa, pe1, re1 = solve(pre, R, q0)
                err = pe1
                feasible = pe1 <= IK_POS_TOL and re1 <= rot_tol
            qb, pe2, re2 = solve(grasp, R, qa)
            err = max(err, pe2)
            feasible = feasible and pe2 <= IK_POS_TOL and re2 <= rot_tol
            margin = min(
                float(np.min(np.minimum(q[self.active] - lo, hi - q[self.active]) / np.maximum(hi - lo, 1e-6)))
                for q in (qa, qb)
            )
            within = (-margin if PRESORT["key"] == "margin" else ci) if feasible else err
            tier = c.get("tier", 0)
            rows.append((tier, not feasible, within, ci) if PRESORT["tier_first"] else (not feasible, tier, within, ci))
        rows.sort()
        order = [r[-1] for r in rows]
        return order, dict(
            seconds=time.time() - t0, candidates=len(cands), feasible=sum(1 for r in rows if not r[1]), first=order[:10]
        )

    @staticmethod
    def _pin(session, q):
        """DECOUPLE: the right arm (MuJoCo joints 22-28) and optionally the waist (12-14) to the sim-side pin."""
        if not DECOUPLE["on"]:
            return
        q = np.asarray(q, dtype=float)
        waist = None
        if DECOUPLE["waist"] == "yawroll":
            waist = [float(q[12]), float(q[13]), None]
        elif DECOUPLE["waist"]:
            waist = q[12:15]
        session.reset.pin_arms(right=q[22:29], waist=waist)

    @staticmethod
    def _tool_fk(ik, q, offset):
        p, R = ik.fk(q, "right")
        return p + R @ offset, R

    def _plan_curobo(self, ik, q0, pel_R, pel_t, cands, table_top, obj_mesh, offset, log):
        """SIMPLE's batch over the grasp set, one tier at a time in the module's order (the shortest success of the
        best tier that has one); `index` is into `cands`."""
        tried = []
        for t in sorted({c["tier"] for c in cands}):
            idx = [i for i, c in enumerate(cands) if c["tier"] == t][:MAX_CANDIDATES]
            plan = self._plan_curobo_batch(q0, pel_R, pel_t, [cands[i] for i in idx], table_top, obj_mesh, offset, log)
            tried.append(
                dict(tier=t, candidates=len(idx), successes=plan.get("successes"), seconds=plan.get("seconds"))
            )
            if plan["ok"]:
                plan["index"] = idx[plan["index"]]
                plan["tiers"] = tried
                return plan
        return dict(ok=False, tiers=tried, rejected=[])

    def _plan_curobo_batch(self, q0, pel_R, pel_t, top, table_top, obj_mesh, offset, log):
        pre, gra = [], []
        for c in top:
            R = pel_R.T @ c["R"]
            tool = pel_R.T @ (c["tool"] - pel_t)
            wrist = tool - R @ offset
            pre.append(pose7(wrist - PREGRASP_BACK * R[:, 0], R))
            gra.append(pose7(wrist, R))
        verts, faces = obj_mesh
        req = dict(
            start=q0[self.active].tolist(),
            names=self.active_names,
            lower=self.ik.lo[self.active].tolist(),
            upper=self.ik.hi[self.active].tolist(),
            pregrasp=pre,
            grasp=gra,
            table=dict(
                center=self._table_center(pel_R, pel_t, table_top).tolist(),
                half=self._table_half,
                quat=Rotation.from_matrix(pel_R.T)
                .as_quat()[[3, 0, 1, 2]]
                .tolist(),  # world-aligned table, pelvis frame
            ),
            object=dict(vertices=((verts - pel_t) @ pel_R).tolist(), faces=faces.tolist()),
            lift_m=LIFT_M,
        )
        (self._round_out / "curobo_request.json").write_text(json.dumps(req))
        rep = query_curobo(self.cu_port, req)
        log(
            f"[grasp_planned] curobo: ok {rep['ok']} index {rep.get('index')} successes {rep.get('successes')} "
            f"{rep.get('seconds', 0):.1f} s"
        )
        if not rep["ok"]:
            return dict(
                ok=False,
                rejected=rep.get("rejected", []),
                status=rep.get("status"),
                seconds=rep.get("seconds"),
                success=rep.get("success"),
            )
        return dict(
            ok=True,
            index=rep["index"],
            approach=np.vstack([q0[self.active], np.asarray(rep["pregrasp_path"])]),
            entry=np.asarray(rep["entry_path"]),
            lift=np.asarray(rep["lift_path"]),
            entry_status=rep.get("entry_status"),
            seconds=rep.get("seconds"),
            successes=rep.get("successes"),
            success=rep.get("success"),
            rejected=rep.get("rejected", []),
        )

    def _table_center(self, pel_R, pel_t, table_top):
        s = self._surface
        c_world = np.array([s["center"][0], s["center"][1], table_top - 0.025])
        self._table_half = [float(s["half"][0]), float(s["half"][1]), 0.025]
        return pel_R.T @ (c_world - pel_t)

    # ------------------------------------------------------------------ the call
    def run(self, session, *, target: str = "water_bottle", profile: str = "bottle_simple", planner=None, source=None):
        from locomani.backends.sonic.pose_preview import PosePreview
        from locomani.backends.sonic.primitives import HAND_CLOSED_RIGHT, HAND_OPEN, Primitives
        from locomani.backends.sonic.streamed_arm import StreamedArm
        from locomani.skills.contract import Outcome
        from locomani.skills.session import ArmOwner, HeldObject
        from locomani.tasks.bottle.ready_grasp_policy import OFFSET

        planner = planner or PLANNER_ORDER[ROUND["n"] % len(PLANNER_ORDER)]
        rnd = ROUND["n"]
        ROUND["n"] += 1
        t0 = session.clock()
        p, meta = session.stack, session.scene.meta
        cfg = self._profiles[profile].config.reach.whole_body
        log = p.log
        source = source or self.source
        record = dict(planner=planner, round=rnd, source=source)
        out = self.out_dir / f"round_{rnd:02d}_{planner}"
        out.mkdir(parents=True, exist_ok=True)
        self._round_out = out

        self._trace = []

        def finish(ok, why, produced=None):
            if DECOUPLE["on"]:
                session.reset.pin_arms()  # release the sim-side pin, whatever happened
            record["decoupled_arm"] = dict(DECOUPLE)
            record["entry_gate"] = dict(ENTRY_GATE)
            record["prefer_high"] = dict(PREFER_HIGH)
            record["base_freeze"] = dict(BASE_FREEZE)
            record.update(ok=ok, why=why, seconds=session.clock() - t0)
            if self._trace:  # stamp, phase, tool target xyz, measured tool xyz, lean, object xyz, closure, bilateral,
                # finger 1 / 2 touching, right arm joints
                np.save(out / "expert_trace.npy", np.asarray(self._trace))
            (out / "record.json").write_text(json.dumps(record, default=lambda o: np.asarray(o).tolist(), indent=1))
            return Outcome(
                ok,
                "" if ok else f"[grasp_planned] {why}",
                duration_s=record["seconds"],
                metrics=record,
                produced=produced,
            )

        self.ik = ik = self._ik_setup(cfg)
        obj_meta = meta["objects"][target]
        surf = meta["surfaces"][obj_meta.get("surface", "A")]
        self._surface = dict(center=surf["xy"], half=surf["half_extent"])
        table_top = float(surf["top"])
        # takeover, as the simple expert takes it
        arm = StreamedArm(p, "right", ik=ik, left_guard=False)
        arm.preview = PosePreview(
            acceleration=cfg.joint_acceleration, joint_limits=(ik.lo, ik.hi), continuous_handoff=True
        )
        arm.hands = {"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()}
        p.cmd.set_hands(left=HAND_OPEN.copy(), right=HAND_OPEN.copy())
        arm.start(hold_planner_pose=True, inherit_planner_velocity=True)
        arm.brake_to_stop()
        session.arm, session.arm_owner = arm, ArmOwner.POSE_STREAM
        target_file = self.out_dir / "target.json"
        shared = None
        if self.fixed and rnd == 0:
            target_file.unlink(missing_ok=True)
        elif self.fixed:
            if not target_file.exists():
                return finish(False, "no shared grasp: round 0 did not fix one")
            shared = json.loads(target_file.read_text())
        state = p.obs()
        yaw_now = float(p.pelvis_yaw(state))
        f = np.array([np.cos(yaw_now), np.sin(yaw_now)])
        want = (
            np.asarray(state["objects"][target]["pos"][:2]) - STAND_FORWARD * f - STAND_RIGHT * np.array([f[1], -f[0]])
        )
        before = np.asarray(p.pelvis_xy(state))
        session.reset.align_robot_xy(want)
        arm.hold(0.6)
        state = p.obs()
        yaw = float(p.pelvis_yaw(state))
        f = np.array([np.cos(yaw), np.sin(yaw)])
        rel = np.asarray(state["objects"][target]["pos"][:2]) - np.asarray(p.pelvis_xy(state))
        record["align"] = dict(
            before=before,
            wanted=want,
            after=np.asarray(p.pelvis_xy(state)),
            yaw=yaw,
            rule="own_heading",
            target_in_pelvis=[float(rel @ f), float(rel @ np.array([-f[1], f[0]]))],
        )
        log(f"[grasp_planned] aligned pelvis {np.round(before, 3)} -> {np.round(record['align']['after'], 3)}")
        qpos = np.asarray(state["qpos"])
        q0 = np.asarray(arm.q, dtype=float).copy()  # the reference the stream continues from
        pel_t = np.asarray(state["pelvis"]["pos"], dtype=float)
        pel_R = Primitives._rot(state["pelvis"]["quat"])
        obj_pos = np.asarray(state["objects"][target]["pos"], dtype=float)
        obj_quat = state["objects"][target]["quat"]
        record.update(object_pos=obj_pos, object_quat=obj_quat, pelvis=pel_t, table_top=table_top)

        def perceive():
            heading = pel_R[:2, 0] / np.linalg.norm(pel_R[:2, 0])
            if source == "cache":
                from locomani import paths

                cands, obj_pts, info = cached_candidates(
                    Path(paths.SCENE_XML).parent, obj_pos, obj_quat, table_top, heading
                )
                record["cache"] = info
                record["side_candidates"] = sum(1 for c in cands if c["tier"] == 0)
                record["candidates"] = len(cands)
                log(f"[grasp_planned] cache: {info['held']} validated of {info['cached']}, {len(cands)} fit here")
                return cands, obj_pts
            if self._camera is None:
                from locomani import paths

                self._camera = DepthCamera(paths.SCENE_XML)
            cloud, R_wc, t_wc = self._camera.capture(qpos)
            world = cloud @ R_wc.T + t_wc
            keep = (
                (np.linalg.norm(world[:, :2] - obj_pos[:2], axis=1) < CROP_XY)
                & (world[:, 2] > table_top - 0.01)
                & (world[:, 2] < table_top + CROP_TOP)
            )
            np.savez_compressed(out / "cloud.npz", cloud_cv=cloud[keep], R_wc=R_wc, t_wc=t_wc)
            tq = time.time()
            grasps = query_graspnet(self.gn_port, cloud[keep])
            record.update(graspnet=dict(points=int(keep.sum()), grasps=len(grasps), seconds=time.time() - tq))
            cands, obj_pts = candidates(grasps, R_wc, t_wc, world[keep], table_top, obj_pos, heading)
            record["side_candidates"] = sum(1 for c in cands if c["tier"] == 0)
            record["candidates"] = len(cands)
            np.save(out / "grasps.npy", grasps)
            return cands, obj_pts

        def perceive_and_plan():
            cands, obj_pts = perceive()
            if not cands:
                return None, None
            tp = time.time()
            if planner == "ik":
                plan = self._plan_ik(ik, q0, pel_R, pel_t, cands, table_top, obj_pts, OFFSET, log)
            else:
                plan = self._plan_curobo(
                    ik,
                    q0,
                    pel_R,
                    pel_t,
                    cands,
                    table_top,
                    self._object_mesh(meta, target, obj_pos, obj_quat),
                    OFFSET,
                    log,
                )
            plan["plan_seconds"] = time.time() - tp
            return cands, plan

        def perceive_and_fix():
            """Round 0 of a fixed-grasp object: both planners over the top candidates, the shared grasp fixed."""
            cands, obj_pts = perceive()
            if not cands:
                return None, None
            if planner != "ik":
                raise ValueError("a fixed-grasp object's round 0 runs our planner (PLANNER_ORDER[0] == 'ik')")
            top = cands[:MAX_CANDIDATES]
            tp = time.time()
            ours = [self._plan_ik(ik, q0, pel_R, pel_t, [c], table_top, obj_pts, OFFSET, lambda *_a: None) for c in top]
            ik_s = time.time() - tp
            mesh = self._object_mesh(meta, target, obj_pos, obj_quat)
            cu = self._plan_curobo_batch(q0, pel_R, pel_t, top, table_top, mesh, OFFSET, log)
            ik_ok = [bool(pl["ok"]) for pl in ours]
            cu_ok = [bool(v) for v in (cu.get("success") or [False] * len(top))]
            both = [i for i in range(len(top)) if ik_ok[i] and cu_ok[i]]
            either = [i for i in range(len(top)) if ik_ok[i] or cu_ok[i]]
            pick = both[0] if both else either[0] if either else None
            selection = dict(
                candidates=len(top), ik_ok=ik_ok, curobo_ok=cu_ok, both=len(both), either=len(either), index=pick,
                common=bool(both), ik_seconds=ik_s, curobo_seconds=cu.get("seconds"),
            )  # fmt: skip
            record["shared"] = selection
            log(
                f"[grasp_planned] shared grasp: rank {pick} of {len(top)}, ours reach {sum(ik_ok)}, CuRobo "
                f"{sum(cu_ok)}, both {len(both)}"
            )
            if pick is None:
                return cands, dict(ok=False, why=f"neither planner reaches any of {len(top)} candidates")
            c = top[pick]
            target_file.write_text(
                json.dumps(
                    dict(
                        selection,
                        grasp=dict(
                            score=c["score"],
                            width=c["width"],
                            tool=c["tool"],
                            R=c["R"],
                            depth_used=c["depth_used"],
                            tier=c["tier"],
                        ),
                        object_pos=obj_pos,
                        object_quat=obj_quat,
                        align_wanted=want,
                        align_after=record["align"]["after"],
                        yaw=record["align"]["yaw"],
                    ),  # fmt: skip
                    default=lambda o: np.asarray(o).tolist(),
                    indent=1,
                )
            )
            if not ik_ok[pick]:
                return cands, dict(
                    ok=False, why=f"ik planner found no path to the shared grasp (rank {pick}, CuRobo's only)"
                )
            plan = dict(ours[pick], index=pick)
            plan["plan_seconds"] = time.time() - tp
            return cands, plan

        def plan_to_shared():
            """A later round of a fixed-grasp object: no perception, round 0's grasp moved with the target since."""
            if planner != "curobo":
                raise ValueError("a fixed-grasp object's later rounds run CuRobo (PLANNER_ORDER[1] == 'curobo')")
            g = shared["grasp"]
            q_then, q_now = np.asarray(shared["object_quat"], dtype=float), np.asarray(obj_quat, dtype=float)
            turn = Rotation.from_quat(q_now[[1, 2, 3, 0]]) * Rotation.from_quat(q_then[[1, 2, 3, 0]]).inv()
            c = dict(
                g,
                tool=turn.apply(np.asarray(g["tool"]) - shared["object_pos"]) + obj_pos,
                R=turn.as_matrix() @ np.asarray(g["R"]),
            )
            record["shared"] = {k: shared[k] for k in ("index", "common", "candidates", "both", "either")}
            if not shared["curobo_ok"][shared["index"]]:
                return [c], dict(
                    ok=False,
                    why=f"curobo planner found no path to the shared grasp (rank {shared['index']}, ours only)",
                )
            tp = time.time()
            mesh = self._object_mesh(meta, target, obj_pos, obj_quat)
            plan = self._plan_curobo_batch(q0, pel_R, pel_t, [c], table_top, mesh, OFFSET, log)
            plan["plan_seconds"] = time.time() - tp
            if not plan["ok"]:
                plan["why"] = f"curobo planner found no path to the shared grasp (rank {shared['index']}) this round"
            return [c], plan

        work = perceive_and_plan if not self.fixed else perceive_and_fix if shared is None else plan_to_shared
        try:
            cands, plan = self._keepalive(arm, work)
        except Exception as exc:  # noqa: BLE001
            return finish(False, f"perception/planning error: {exc!r}")
        if cands is None:
            if "graspnet" in record:
                return finish(False, f"no graspnet candidate fits the Dex1 ({record['graspnet']['grasps']} grasps)")
            return finish(False, f"no cached grasp fits here ({record.get('cache')})")
        record["plan"] = {k: v for k, v in plan.items() if k not in ("approach", "entry", "lift")}
        if not plan["ok"]:
            return finish(
                False,
                plan.get("why")
                or f"{planner} planner found no path over {len(cands[: (2 if planner == 'ik' else 1) * MAX_CANDIDATES])} candidates",
            )
        drift = float(np.linalg.norm(np.asarray(p.obs()["pelvis"]["pos"])[:2] - pel_t[:2]))
        record["pelvis_drift_while_planning_m"] = drift
        np.savez_compressed(out / "plan.npz", **{k: np.asarray(plan[k]) for k in ("approach", "entry", "lift")})
        c = cands[plan["index"]]
        record["grasp"] = dict(
            score=c["score"], width=c["width"], tool=c["tool"], R=c["R"], depth_used=c["depth_used"], tier=c["tier"],
            **{k: c[k] for k in ("row", "slip", "force") if k in c},
        )  # fmt: skip
        record["start_joints"] = q0
        wrist = c["tool"] - c["R"] @ OFFSET
        session.reset.set_ghost(pos=wrist.tolist(), quat=Rotation.from_matrix(c["R"]).as_quat()[[3, 0, 1, 2]].tolist())

        obj0 = np.asarray(p.obs()["objects"][target]["pos"], dtype=float)

        def stream(path10, phase, check_knock=True):
            base = np.asarray(arm.q, dtype=float).copy()
            full = np.repeat(base[None], len(path10) + 1, axis=0)
            full[0] = base
            full[1:, self.active] = path10
            P, s_wp, total = retime(full)
            began = previous = None
            wall = time.monotonic()
            while True:
                st = p.obs()
                stamp = float(st["t"])
                if previous is not None and stamp == previous:
                    if time.monotonic() - wall > 2.0:
                        return f"physical clock stalled in {phase}"
                    time.sleep(0.004)
                    continue
                wall = time.monotonic()
                began = stamp if began is None else began
                previous = stamp
                if st.get("fallen"):
                    return f"fell in {phase}"
                moved = float(np.linalg.norm(np.asarray(st["objects"][target]["pos"])[:2] - obj0[:2]))
                if check_knock and moved > 0.05:
                    return f"target knocked {moved * 100:.0f} cm in {phase}"
                q, v = sample(P, s_wp, total, stamp - began)
                arm.q = q
                arm._push(q, reference_velocity=v, reference_time=stamp)
                self._pin(session, q)
                if stamp - began >= total:
                    return None

        record["timeline"] = []
        tp = session.clock()
        why = stream(np.asarray(plan["approach"]), "approach", True)
        record["timeline"].append(dict(phase="approach", seconds=session.clock() - tp, waypoints=len(plan["approach"])))
        if why:
            return finish(False, why)
        st = p.obs()
        tool_now = np.asarray(st["palms"]["right"]["pos"]) + Primitives._rot(st["palms"]["right"]["quat"]) @ OFFSET
        record["pregrasp_error_open_loop_m"] = float(
            np.linalg.norm(tool_now - (c["tool"] - PREGRASP_BACK * c["R"][:, 0]))
        )
        rise_or_fail = self._grasp_like_the_expert(session, arm, ik, cfg, target, c, obj0, record, out)
        if isinstance(rise_or_fail, str):
            return finish(False, rise_or_fail)
        z_before = rise_or_fail
        held = HeldObject(
            key=target, sides=("right",), grip_in_palm={"right": np.zeros(3)}, hold_offset=None, start_z=z_before
        )
        session.held, session.hands["right"] = held, 1.0
        return finish(True, "", produced=held)

    def _grasp_like_the_expert(self, session, arm, ik, cfg, target, c, obj0, record, out):
        """From the pre-grasp on, the simple expert's own closed loop (tasks/bottle/simple_policy.py `_run`, phases
        settle -> entry -> close, then lift and hold) on SupportedMotion, for the grasp the planner chose: the right tool
        to the pre-grasp 10 cm back along the grasp's approach in the grasp attitude, one dash along the approach with
        the target riding the measured object, close on arrival or bilateral pad contact holding the measured tools;
        then (not the expert's) lift 10 cm straight up with the tracking feedback on and hold. Same gates as the expert
        (foot error, COM shift, knocked target, contact timeout, rise). Returns the target's height before the lift, or the failure reason."""
        from locomani.backends.sonic.primitives import HAND_CLOSED_RIGHT, HAND_OPEN, Primitives
        from locomani.primitives.contacts import bilateral_dex1_contact
        from locomani.primitives.supported_motion import SupportedMotion
        from locomani.tasks.bottle.ready_grasp_policy import OFFSET
        from locomani.tasks.bottle.simple_policy import (
            entry_captured,
            entry_command,
            lean_angle,
            minjerk,
            object_in_palm,
        )

        p = session.stack
        simple = self._profiles["bottle_simple"].config.reach.simple
        state = p.obs()

        def tools(st):
            return np.array(
                [
                    np.asarray(st["palms"][s]["pos"]) + Primitives._rot(st["palms"][s]["quat"]) @ OFFSET
                    for s in ("left", "right")
                ]
            )

        actual = tools(state)
        targets = actual.copy()
        posture = np.asarray(arm.q, dtype=float).copy()
        posture[12:14] = 0.0
        weights = np.full(29, 0.04)
        weights[12:14] = cfg.comfort_waist_weight
        motion = SupportedMotion(
            ik, state, cfg, arm.q, arm.root_quat, tool_offset=OFFSET, hand_weights=(0.35, 1.0),
            posture_reference=posture, posture_weights=weights,
            # the closed loop's whole-body IK takes its joint ranges from the model, not from `ik`: with the arm
            # pinned it must honour the planner's caps (waist, the inward shoulder roll), else the pin drives the
            # arm into the torso (2026-10-10)
            joint_limits=(ik.lo, ik.hi) if DECOUPLE["on"] else None,
            base_range=BASE_FREEZE["range"] if BASE_FREEZE["on"] else None,
        )  # fmt: skip
        heading = motion.body.heading
        root_stream = motion.root  # the root reference SONIC keeps receiving: re-anchoring the solver (BASE_FREEZE)
        # must not feed the measured heading back as the reference (yaw ran away 33 deg in a round, 2026-10-10)
        forward = heading[:, 0]
        direction = c["R"][:, 0]
        grip_offset = c["tool"] - obj0
        rotations = np.array([Primitives._rot(state["palms"]["left"]["quat"]), c["R"]])
        phase, began = "settle", float(state["t"])
        closure, lean, entry_cmd, lift_from = 0.0, 0.0, None, None
        retreat_from = retreat_to = None
        shift = np.zeros(3)
        gate_held = 0.0
        z_before = None
        previous, wall = None, time.monotonic()
        trace = self._trace = []
        record["expert_phases"] = phases = []

        def enter(name, stamp):
            nonlocal phase, began
            phases.append(dict(phase=name, t=stamp))
            phase, began = name, stamp

        while True:
            state = p.obs()
            stamp = float(state["t"])
            if previous is not None and stamp == previous:
                if time.monotonic() - wall > 2.0:
                    return "physical clock stalled"
                time.sleep(0.004)
                continue
            wall = time.monotonic()
            dt = 0.02 if previous is None else stamp - previous
            previous = stamp
            if state.get("fallen"):
                return f"fell in {phase}"
            obj = state["objects"][target]
            actual = tools(state)
            bilateral = bilateral_dex1_contact(obj, "right")
            elapsed = stamp - began
            grip = np.asarray(obj["pos"], dtype=float) + grip_offset
            pregrasp = grip - PREGRASP_BACK * direction
            if phase in ("settle", "entry", "retreat"):
                knocked = float(np.linalg.norm(np.asarray(obj["pos"])[:2] - obj0[:2]))
                if knocked > simple.disturb_limit_m:
                    return f"target knocked {knocked * 100:.0f} cm in {phase}"
            if phase == "settle":
                targets[1] = pregrasp
                lean = lean_angle(state["pelvis"]["pos"], grip, simple.lean_max_rad, forward)
                err = float(np.linalg.norm(actual[1] - pregrasp))
                if (err <= simple.settle_tolerance_m and elapsed > 0.3) or elapsed >= simple.settle_timeout_s:
                    record["settle"] = dict(error_m=err, seconds=elapsed)
                    entry_cmd = targets[1].copy()
                    enter("entry", stamp)
            elif phase == "entry":
                palm = Primitives._rot(state["palms"]["right"]["quat"])
                off = object_in_palm(palm, actual[1], grip)
                # ENTRY_GATE: the advance waits (bounded) while the measured offset across the palm is still large
                if (
                    ENTRY_GATE["on"]
                    and max(abs(off[1]), abs(off[2])) > ENTRY_GATE["lateral_m"]
                    and gate_held < ENTRY_GATE["hold_s"]
                ):
                    gate_held += dt
                else:
                    entry_cmd = entry_command(entry_cmd, grip, direction, DASH_SPEED, dt, simple.entry_overshoot_m)
                # the expert's jaw-axis correction riding the advance (lateral_correction), here on both axes across
                # the approach: a top-down grasp's palm z is horizontal and a few centimetres there miss a small target
                wanted = shift + palm[:, 1] * off[1] + palm[:, 2] * off[2]
                delta = wanted - shift
                norm = float(np.linalg.norm(delta))
                if norm > 1e-9:
                    shift = shift + delta * min(1.0, simple.lateral_correction_speed_m_s * dt / norm)
                total = float(np.linalg.norm(shift))
                if total > simple.lateral_correction_max_m:
                    shift *= simple.lateral_correction_max_m / total
                targets[1] = entry_cmd + shift
                arrived = entry_captured(actual[1], grip, direction, simple.entry_capture_depth_m)
                if arrived or bilateral:
                    off = object_in_palm(palm, actual[1], grip)
                    # the expert's alignment gate (entry_aligned), with the jaw-axis allowance this grasp's width leaves
                    lateral_tol = max(0.005, (2 * PAD_Y - c["width"]) / 2 + 0.005)
                    aligned = abs(off[1]) <= lateral_tol and abs(off[2]) <= PAD_Z
                    entries = record.setdefault("entries", [])
                    entries.append(
                        dict(
                            arrived=bool(arrived), bilateral=bool(bilateral), offset_in_palm=off, aligned=bool(aligned)
                        )
                    )
                    if aligned or bilateral or len(entries) >= max(2, int(simple.entry_attempts)):
                        record["entry"] = dict(
                            arrived=bool(arrived),
                            bilateral=bool(bilateral),
                            error_m=float(np.linalg.norm(actual[1] - grip)),
                            offset_in_palm=off,
                            shift_m=float(np.linalg.norm(shift)),
                            attempts=len(entries),
                            gate_held_s=gate_held,
                        )
                        targets[:] = actual
                        motion.freeze_rotation_feedback()
                        enter("close", stamp)
                    else:
                        # the expert's last resort: straight back off and re-enter with the measured offset as the
                        # starting correction (retry_target)
                        retreat_from = targets[1].copy()
                        retreat_to = retreat_from - simple.retreat_m * direction
                        shift = shift + palm[:, 1] * off[1] + np.array([0.0, 0.0, off[2]])  # as retry_target
                        enter("retreat", stamp)
                elif elapsed > simple.entry_timeout_s:
                    return f"entry timed out {np.linalg.norm(actual[1] - grip) * 100:.1f} cm from the grasp"
            elif phase == "retreat":
                targets[1] = retreat_from + minjerk(elapsed / simple.retreat_seconds) * (retreat_to - retreat_from)
                if elapsed >= simple.retreat_seconds:
                    entry_cmd = targets[1].copy() - shift  # the shift rides on top of the advance again
                    enter("entry", stamp)
            elif phase == "close":
                closure = min(1.0, elapsed / simple.close_seconds)
                if elapsed >= simple.close_seconds:
                    if bilateral:
                        lift_from = actual[1].copy()
                        motion.resume_rotation_feedback()
                        z_before = float(obj["pos"][2])
                        record["closed_touching"] = True
                        enter("lift", stamp)
                    elif elapsed >= simple.close_seconds + simple.contact_timeout_s:
                        record["closed_touching"] = False
                        return "no bilateral pad contact after the close"
            elif phase == "lift":
                # Straight up from where the jaw closed, translation feedback still on, the lean kept. The expert's
                # lift (frozen feedback, torso-held offset, then the lean back upright) left the hand 12-18 cm under
                # its command on these grasps and lowered a lifted bottle back to the table (object 040, 2026-10-08).
                targets[1] = lift_from + [0.0, 0.0, LIFT_M * minjerk(elapsed / simple.lift_seconds)]
                if elapsed >= simple.lift_seconds:
                    enter("recover", stamp)
            elif phase == "recover":
                targets[1] = lift_from + [0.0, 0.0, LIFT_M]
                if elapsed >= HOLD_S:
                    rise = float(obj["pos"][2]) - z_before
                    record.update(rise_m=rise, touching_end=bool(bilateral))
                    if not bilateral:
                        return f"lost pad contact during the lift (rise {rise * 100:.1f} cm)"
                    if rise < 0.05:
                        return f"not lifted (rise {rise * 100:.1f} cm)"
                    return z_before
            torso = heading @ Rotation.from_euler("y", lean).as_matrix()
            if BASE_FREEZE["on"]:
                motion.anchor_base(state["pelvis"]["pos"], state["pelvis"]["quat"], state["joints"])
            pose = motion.step(
                targets, rotations, actual, dt, torso_rotation=torso,
                measured_rotations=np.array([Primitives._rot(state["palms"][s]["quat"]) for s in ("left", "right")]),
            )  # fmt: skip
            if max(pose.foot_errors) > 0.025 or float(np.linalg.norm(pose.com_shift)) > 0.04:
                return f"support reference outside tolerance in {phase}"
            arm.q, arm.root_quat = pose.joints, (root_stream if BASE_FREEZE["on"] else motion.root)
            arm.hands["right"] = HAND_OPEN + closure * (HAND_CLOSED_RIGHT - HAND_OPEN)
            arm._push(pose.joints, reference_velocity=motion.v[4:], reference_time=stamp)
            self._pin(session, pose.joints)
            touch = set(obj.get("touch_links") or ())
            trace.append(
                [
                    stamp,
                    ["settle", "entry", "close", "lift", "recover", "retreat"].index(phase),
                    *targets[1],
                    *actual[1],
                    lean,
                    *obj["pos"],
                    closure,
                    float(bilateral),
                    float(any("finger_1" in t for t in touch)),
                    float(any("finger_2" in t for t in touch)),
                    *np.asarray(state["joints"])[22:29],
                    *np.asarray(pose.joints)[22:29],
                    *np.asarray(state["joints"])[12:15],
                    *np.asarray(pose.joints)[12:15],
                    float(pose.hand_errors[1]),
                    float(max(pose.foot_errors)),
                    float(np.linalg.norm(pose.com_shift)),
                    *np.asarray(state["pelvis"]["pos"], dtype=float),
                    *np.asarray(state["pelvis"]["quat"], dtype=float),
                    *np.asarray(motion.feedback[1] if motion.feedback is not None else np.zeros(3), dtype=float),
                ]
            )  # fmt: skip  (23 columns as documented + commanded right arm 7, measured waist 3, commanded waist 3,
            # the whole-body IK's right-hand error, foot error and COM shift, the measured pelvis pos 3 + quat wxyz 4
            # (the pose this tick's IK solved against), the tracking feedback added to the right tool target 3: 49 columns)

    def _object_mesh(self, meta, target, pos, quat):
        """The target's CoACD hull (coacd_merge.obj) in the world, as the scene placed it (make_scenes.py)."""
        cache = getattr(self, "_mesh_cache", None)
        if cache is None:
            import mujoco

            from locomani import paths

            m = mujoco.MjModel.from_xml_path(paths.SCENE_XML)
            b = m.body(meta["objects"][target]["body"]).id
            verts, faces = [], []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                Rg = Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
                faces.append(f + sum(len(x) for x in verts))
                verts.append(v @ Rg.T + m.geom_pos[g])
            cache = self._mesh_cache = (np.vstack(verts), np.vstack(faces))
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        return cache[0] @ R.T + np.asarray(pos), cache[1]
