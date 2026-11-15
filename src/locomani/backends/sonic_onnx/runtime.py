"""SONIC v1.1 in this process: the deploy's teleop mode re-implemented around the shipped ONNX files.

What runs, per 50 Hz tick, in the order of gear_sonic_deploy's CONTROL tick (g1_deploy_onnx_ref.cpp:4063-4263; file and
line references below are to third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/, the
2026-11-15 checkout on vastgpu, whose built binary carries the +-0.2 shoulder-roll defaults):

  1. log the state (state_logger.cpp:197-242): gyro, q - default and dq in IsaacLab order, the previous tick's raw action,
     the base quaternion; the decoder reads the last 10 entries oldest first, zero entries in the oldest slots at start-up
     (a zero quaternion's gravity is (0, 0, +1), math_utils.hpp:132-150);
  2. the planner, a 10 Hz thread in the deploy (cpp:3818-3899): on every 5th tick the replan rule runs on the latest
     movement command; a new plan is generated from the playing motion 2 frames ahead (localmotion_kplanner.hpp:533-678)
     and merged one tick later -- the thread's latency, made deterministic here;
  3. the heading anchor (cpp:572-619): captured at the first plan, `apply` = heading(robot) * heading(plan root)^-1 maps
     the planner's yaw-zero world onto the robot's;
  4. the encoder in teleop mode (mode 1, observation_config.yaml): the anchor orientation, 10 future frames (step 5) of
     the plan's 12 leg joints and their velocities, and the VR 3-point targets (left wrist, right wrist, head in the
     reference pelvis frame); every other block zero (cpp:2028, 2085-2111);
  5. the decoder: token + the 10-entry histories -> 29 raw actions, IsaacLab order;
  6. PD targets = default + action[isaaclab_to_mujoco] * action_scale (cpp:3319-3327), no clip or filter;
  7. frame advancement (cpp:3347-3567): merge a pending plan (8-frame crossfade), advance one frame, and at the end of
     the motion in IDLE re-adapt the held leg targets toward the measured legs.

The C++ deploy drives Dex3 hands; here the Dex1 jaws stay with `scalebfm.runtime.Sim`'s own PD, outside the policy (the
hands are not in SONIC's observations or actions).

`SonicPolicy` takes the sim's state dict and returns 29 PD targets, both in the sim's joint order (`joint_names`, the
ScaleBFM metadata order `scalebfm.runtime.Sim` uses), so the leg odometry and the self-model features read the same
columns as under ScaleBFM. `SonicSim` is that Sim with SONIC's gains, effort limits and default pose (the deploy's
policy_parameters.hpp; effort limits of NVIDIA's own MuJoCo config, gear_sonic/utils/mujoco_sim/wbc_configs/
g1_29dof_sonic_model12.yaml), on the same 5 ms x 4 step the policy was trained with (config.yaml sim_dt, decimation).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import yaml

from locomani import paths
from locomani.backends.scalebfm import runtime as R
from locomani.backends.sonic.sonic_decoder import ACTION_SCALE, DEFAULT_ANGLES, ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

DT = R.DT  # 50 Hz control
PLANNER_EVERY = 5  # control ticks per planner tick (10 Hz, cpp:2183-2187)
LOOK_AHEAD = 2  # motion_look_ahead_steps (localmotion_kplanner.hpp:215)
BLEND_FRAMES = 8  # crossfade of a new plan into the playing motion (cpp:3403)
HISTORY = 10  # decoder history entries (observation_config.yaml *_10frame_step1)
FUTURE = 10  # encoder future frames
FUTURE_STEP = 5  # ... every 5th 50 Hz frame
PLANNER_FPS = 30.0
INIT_ROOT_Z = 0.788740  # the planner's initial context pelvis height (cpp:3765-3781)
SEED = 1234  # planner random_seed, set once at Initialize (localmotion_kplanner.hpp:217, 351)
ALLOWED_TOKENS = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64)  # 36, 40 or 44 frames (tensorrt.hpp:96-102)

# locomotion modes (localmotion_kplanner.hpp:78-115); the static ones never replan on speed or direction
IDLE, SLOW_WALK, WALK, RUN, IDEL_SQUAT = 0, 1, 2, 3, 4
STATIC_MODES = frozenset({0, 4, 5, 6, 7, 8})  # IDLE, IDEL_SQUAT, IDEL_KNEEL_TWO_LEGS, IDEL_KNEEL, ... (kpl:108-115)
REPLAN_INTERVAL = {RUN: 0.1}  # s; crawling 0.2, punches 1.0, everything else 1.0 (cpp:232-235)
N_MODES = 27

# idle re-adaptation at the end of a motion (cpp:245-247, 3503-3564)
ADAPT_TRIGGER, RECOVER_TRIGGER, ADAPT_STOP = 0.10, 0.045, 0.05
LEGS = np.arange(12)  # MuJoCo order: left hip pitch/roll/yaw, knee, ankle pitch/roll, then right

# gains (policy_parameters.hpp:40-65, 143-207): kp = armature * w^2, kd = 4 * armature * w, w = 2 pi 10 Hz
_W = 2.0 * math.pi * 10.0
_A5020, _A7520_14, _A7520_22, _A4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
_LEG = [_A7520_22, _A7520_22, _A7520_14, _A7520_22, 2 * _A5020, 2 * _A5020]
_ARM = [_A5020] * 5 + [_A4010] * 2
ARMATURE = np.array(_LEG + _LEG + [_A7520_14, 2 * _A5020, 2 * _A5020] + _ARM + _ARM)
KP = ARMATURE * _W**2
KD = 4.0 * ARMATURE * _W
# effort limits of NVIDIA's MuJoCo config (g1_29dof_sonic_model12.yaml:241-249; training uses 139 at the hips'
# pitch and roll); the scene's own actuator ranges clamp on top of these, as they do for ScaleBFM
_LEG_EFFORT = [88.0, 88.0, 88.0, 139.0, 50.0, 50.0]
_ARM_EFFORT = [25.0] * 5 + [5.0] * 2
EFFORT = np.array(_LEG_EFFORT + _LEG_EFFORT + [88.0, 50.0, 50.0] + _ARM_EFFORT + _ARM_EFFORT)

# encoder / decoder input layouts (observation_config.yaml order; cpp:1868-1891, 1807-1811)
ENCODER_BLOCKS = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_anchor_orientation_heading_10frame_step5", 60),
    ("motion_anchor_orientation_heading", 6),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_heading_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
)
DECODER_BLOCKS = (
    ("token_state", 64),
    ("his_base_angular_velocity_10frame_step1", 30),
    ("his_body_joint_positions_10frame_step1", 290),
    ("his_body_joint_velocities_10frame_step1", 290),
    ("his_last_actions_10frame_step1", 290),
    ("his_gravity_dir_10frame_step1", 30),
)
TELEOP_MODE = 1


def _offsets(blocks) -> dict:
    out, k = {}, 0
    for name, n in blocks:
        out[name] = slice(k, k + n)
        k += n
    return out


ENC = _offsets(ENCODER_BLOCKS)
DEC = _offsets(DECODER_BLOCKS)
ENC_DIM, DEC_DIM = ENC[ENCODER_BLOCKS[-1][0]].stop, DEC[DECODER_BLOCKS[-1][0]].stop

# the VR 3-point targets, training's definition (gear_sonic mdp/observations.py:1348-1363, commands.py:2215-2266):
# order left wrist, right wrist, head; points on the links, orientations the links' own
VR_LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_OFFSETS = (np.array([0.18, -0.025, 0.0]), np.array([0.18, 0.025, 0.0]), np.array([0.0, 0.0, 0.35]))


# -- quaternions (w, x, y, z) ----------------------------------------------------------------------------------------
def qmul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])  # fmt: skip


def qconj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], float)


def qmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def heading(q) -> np.ndarray:
    """The z rotation by the yaw of q's rotated +x axis (math_utils.hpp:206-215, 300-304)."""
    x = qmat(q)[:, 0]
    yaw = math.atan2(x[1], x[0])
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def slerp(q0, q1, t: float) -> np.ndarray:
    """quat_slerp_d (math_utils.hpp:343): shortest arc, linear when nearly parallel."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    s = math.sin(th)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / s


def gravity(q) -> np.ndarray:
    """rotate(conj(base_quat), (0, 0, -1)) (cpp:1637-1645)."""
    return qmat(q).T @ np.array([0.0, 0.0, -1.0])


def rot6(q) -> np.ndarray:
    """[R00, R01, R10, R11, R20, R21] (cpp:629-708)."""
    m = qmat(q)
    return np.array([m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[2, 0], m[2, 1]])


# -- the playing motion ----------------------------------------------------------------------------------------------
@dataclass
class Motion:
    """A 50 Hz reference: root position and quaternion (planner world), 29 joints and their velocities, MuJoCo order."""

    pos: np.ndarray
    quat: np.ndarray
    q: np.ndarray
    dq: np.ndarray

    @property
    def T(self) -> int:
        return len(self.q)

    def copy(self) -> "Motion":
        return Motion(self.pos.copy(), self.quat.copy(), self.q.copy(), self.dq.copy())

    def sample(self, t: float) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz, 29 joints] at time t (s, frame 0 at 0): linear between frames, slerp for the root
        quaternion (localmotion_kplanner.hpp:628-678)."""
        f = t / DT
        f0 = min(int(math.floor(f)), self.T - 1)
        f1 = min(f0 + 1, self.T - 1)
        w = f - f0 if f1 > f0 else 0.0
        pos = (1 - w) * self.pos[f0] + w * self.pos[f1]
        q = (1 - w) * self.q[f0] + w * self.q[f1]
        return np.r_[pos, slerp(self.quat[f0], self.quat[f1], w), q]


def from_planner(qpos: np.ndarray) -> Motion:
    """The planner's 30 Hz frames [x, y, z, qw..qz, 29 MuJoCo joints] resampled to 50 Hz (localmotion_kplanner.hpp:
    395-519): T50 = floor(T30 / 30 * 50), linear position and joints, slerped root, velocity a forward difference with
    the last frame copying the one before."""
    n30 = len(qpos)
    n50 = int(math.floor(n30 / PLANNER_FPS * 50.0))
    pos, quat, q = np.zeros((n50, 3)), np.zeros((n50, 4)), np.zeros((n50, 29))
    for f in range(n50):
        f30 = f * PLANNER_FPS / 50.0
        f0 = min(int(math.floor(f30)), n30 - 1)
        f1 = min(f0 + 1, n30 - 1)
        w0 = 1.0 - (f30 - f0)
        pos[f] = w0 * qpos[f0, :3] + (1 - w0) * qpos[f1, :3]
        quat[f] = slerp(qpos[f0, 3:7], qpos[f1, 3:7], f30 - f0)
        q[f] = w0 * qpos[f0, 7:] + (1 - w0) * qpos[f1, 7:]
    dq = np.zeros_like(q)
    dq[:-1] = (q[1:] - q[:-1]) * 50.0
    if n50 > 1:
        dq[-1] = dq[-2]
    return Motion(pos, quat, q, dq)


def merge(old: Motion, new: Motion, current: int, fgen: int) -> Motion | None:
    """A new plan generated at frame `fgen` of `old` crossfaded in from `current` on (cpp:3392-3432); None when it
    would be empty."""
    length = fgen - current + new.T
    if length <= 0:
        return None
    start = max(0, fgen - current)
    out = Motion(np.zeros((length, 3)), np.zeros((length, 4)), np.zeros((length, 29)), np.zeros((length, 29)))
    for f in range(length):
        f_old = f + current
        f_new = f_old - fgen
        f_old = min(max(f_old, 0), old.T - 1)
        f_new = min(max(f_new, 0), new.T - 1)
        w = min(max((f - start) / BLEND_FRAMES, 0.0), 1.0)
        out.q[f] = (1 - w) * old.q[f_old] + w * new.q[f_new]
        out.dq[f] = (1 - w) * old.dq[f_old] + w * new.dq[f_new]
        out.pos[f] = (1 - w) * old.pos[f_old] + w * new.pos[f_new]
        out.quat[f] = slerp(old.quat[f_old], new.quat[f_new], w)
    return out


# -- commands --------------------------------------------------------------------------------------------------------
@dataclass
class Movement:
    """The planner's movement state: mode, speed (-1: the mode's default), directions in the ROBOT's world frame (the
    policy turns them into the planner's), pelvis height (-1: none; only the squat modes honour it)."""

    mode: int = IDLE
    speed: float = -1.0
    movement: np.ndarray = field(default_factory=lambda: np.zeros(3))
    facing: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    height: float = -1.0


@dataclass
class SonicCommand:
    """One tick's input: the planner's movement state and the VR 3-point targets (left wrist, right wrist, head; points
    in the reference pelvis frame, quaternions w, x, y, z of the links relative to the reference pelvis)."""

    movement: Movement
    vr_pos: np.ndarray  # (9,)
    vr_quat: np.ndarray  # (12,)


def vr_targets(links: dict, pelvis_pos, pelvis_quat) -> tuple[np.ndarray, np.ndarray]:
    """The 9 + 12 VR numbers from world link poses {name: (pos, quat)} and the reference pelvis pose."""
    Rp = qmat(pelvis_quat)
    inv = qconj(pelvis_quat)
    pos, quat = [], []
    for name, off in zip(VR_LINKS, VR_OFFSETS):
        p, q = links[name]
        pos.append(Rp.T @ (np.asarray(p) + qmat(q) @ off - np.asarray(pelvis_pos)))
        quat.append(qmul(inv, q))
    return np.concatenate(pos), np.concatenate(quat)


# -- the policy ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SonicConfig:
    device_id: int = 0  # CUDA device of the planner (and of the encoder / decoder unless nets_on_cpu); -1: all on CPU
    nets_on_cpu: bool = False  # encoder and decoder on the CPU: many processes then do not queue on one GPU every tick
    policy_dir: str = paths.SONIC_POLICY_DIR
    planner: str = f"{paths.SONIC_DEPLOY}/planner/target_vel/V2/planner_sonic.onnx"
    threads: int = 1  # onnxruntime CPU threads per session


class SonicPolicy:
    """SONIC v1.1 in teleop mode. `joint_names`: the sim's joint order, in which state and targets are exchanged."""

    # keypoint bodies Motor's FK computes (Keypoints(sim.m, policy.sel))
    sel = ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link",
           "right_ankle_roll_link")  # fmt: skip

    def __init__(self, joint_names, cfg: SonicConfig | None = None):
        self.cfg = cfg or SonicConfig()
        self.to_mj = np.array(
            [list(joint_names).index(n) for n in MUJOCO_JOINT_NAMES]
        )  # sim index of each MuJoCo joint
        self.from_mj = np.array([MUJOCO_JOINT_NAMES.index(n) for n in joint_names])  # MuJoCo index of each sim joint
        self._check_layout()
        nets = -1 if self.cfg.nets_on_cpu else self.cfg.device_id
        self.enc = self._session(f"{self.cfg.policy_dir}/model_encoder.onnx", nets)
        self.dec = self._session(f"{self.cfg.policy_dir}/model_decoder.onnx", nets)
        self.plan = self._session(self.cfg.planner, self.cfg.device_id)
        self.root_source = None  # leg odometry (reachwm leg_odo.OdoRoot) sets this; called once a tick
        self.root_est = None
        self.reset_state()

    def _check_layout(self) -> None:
        conf = yaml.safe_load(open(f"{self.cfg.policy_dir}/observation_config.yaml"))
        enc = [o["name"] for o in conf["encoder"]["encoder_observations"] if o.get("enabled", True)]
        dec = [o["name"] for o in conf["observations"] if o.get("enabled", True)]
        assert enc == [n for n, _ in ENCODER_BLOCKS], f"encoder layout changed: {enc}"
        assert dec == [n for n, _ in DECODER_BLOCKS], f"decoder layout changed: {dec}"
        modes = {m["name"]: m["mode_id"] for m in conf["encoder"]["encoder_modes"]}
        assert modes.get("teleop") == TELEOP_MODE, modes

    def _session(self, path: str, device: int):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = self.cfg.threads
        so.log_severity_level = 3
        if device < 0:
            return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        import torch  # noqa: F401 -- loads the CUDA 12 / cuDNN 9 libraries onnxruntime's CUDA provider needs

        ort.preload_dlls()
        s = ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", {"device_id": device})])
        assert "CUDAExecutionProvider" in s.get_providers(), f"no CUDA provider for {path}"
        return s

    # -- state ------------------------------------------------------------------------------------------------------
    def reset_state(self) -> None:
        self.hist: deque = deque(maxlen=HISTORY)
        self.last_action = np.zeros(29)  # raw, IsaacLab order
        self.motion: Motion | None = None
        self.frame = 0
        self.k = 0
        self.last_move: Movement | None = None
        self.timer = 0.0
        self.pending: tuple | None = None  # (motion, fgen, merge at tick)
        self.reinit_heading = False
        self.apply = np.array([1.0, 0.0, 0.0, 0.0])
        self.init_base = self.init_ref = None
        self.readapt_stored = False
        self.readapt_state = "idle"
        self.readapt_orig = np.zeros(12)
        self.token = np.zeros(64)
        self.move = Movement()

    def reset(self, st: dict) -> None:
        """Fresh histories and the planner's Initialize from the measured joints (cpp:3765-3781): four identical
        context frames at the origin, identity root, IDLE; the first plan is the playing motion from frame 0."""
        self.reset_state()
        q = np.asarray(st["dof_pos"], float)[self.to_mj]
        ctx = np.r_[0.0, 0.0, INIT_ROOT_Z, 1.0, 0.0, 0.0, 0.0, q]
        move = Movement(IDLE, -1.0, np.zeros(3), np.array([1.0, 0.0, 0.0]), -1.0)
        self.motion = from_planner(self._run_planner(np.repeat(ctx[None], 4, 0), move, planner_frame=True))
        self.frame = 0
        # the anchor the deploy captures on the first tick after this plan (cpp:572-598; the robot has not moved)
        self.init_base, self.init_ref = np.asarray(st["root_quat"], float).copy(), self.motion.quat[0].copy()
        self.apply = qmul(heading(self.init_base), qconj(heading(self.init_ref)))
        # the movement state Initialize used, in the robot's world (commands are compared there)
        self.last_move = Movement(IDLE, -1.0, np.zeros(3), qmat(self.apply) @ move.facing, -1.0)

    # -- planner ----------------------------------------------------------------------------------------------------
    def to_planner(self, v) -> np.ndarray:
        """A robot-world direction in the planner's world (the inverse of the heading anchor)."""
        return qmat(self.apply).T @ np.asarray(v, float)

    def _run_planner(self, ctx: np.ndarray, move: Movement, planner_frame: bool = False) -> np.ndarray:
        mv = move.movement if planner_frame else self.to_planner(move.movement)
        fc = move.facing if planner_frame else self.to_planner(move.facing)

        def unit(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else np.zeros(3)  # zmq_manager.hpp:644-645

        mode = move.mode if 0 <= move.mode < N_MODES else IDLE  # tensorrt.hpp:384-387
        feed = dict(
            context_mujoco_qpos=ctx[None].astype(np.float32),
            target_vel=np.array([move.speed], np.float32),
            mode=np.array([mode], np.int64),
            movement_direction=unit(mv)[None].astype(np.float32),
            facing_direction=unit(fc)[None].astype(np.float32),
            random_seed=np.array([SEED], np.int64),
            has_specific_target=np.zeros((1, 1), np.int64),
            specific_target_positions=np.zeros((1, 4, 3), np.float32),
            specific_target_headings=np.zeros((1, 4), np.float32),
            allowed_pred_num_tokens=ALLOWED_TOKENS,
            height=np.array([move.height], np.float32),
        )
        out = self.plan.run(None, feed)
        n = int(np.asarray(out[1]).ravel()[0])
        assert 0 < n <= 64 and np.isfinite(out[0][0, :n]).all(), "planner returned a bad motion"
        return np.asarray(out[0][0, :n], float)

    @staticmethod
    def _same(a: Movement, b: Movement) -> dict:
        """Exact comparisons, as the deploy's (the anchor is fixed after start-up, so robot-world vectors compare the
        same as the planner-world ones it stores)."""
        return dict(
            mode=a.mode == b.mode,
            speed=a.speed == b.speed,
            movement=np.array_equal(a.movement, b.movement),
            facing=np.array_equal(a.facing, b.facing),
            height=a.height == b.height,
        )

    def _planner_tick(self, move: Movement) -> None:
        """The deploy's 10 Hz replan rule (cpp:3818-3899) on the latest movement state."""
        self.timer += PLANNER_EVERY * DT
        same = self._same(move, self.last_move)
        replan = not (same["mode"] and same["facing"] and same["height"])
        if not replan and move.mode not in STATIC_MODES:
            interval = REPLAN_INTERVAL.get(move.mode, 1.0)
            replan = not same["speed"] or not same["movement"] or (self.timer >= interval - 1e-9 and move.speed != 0)
        if not replan:
            return
        self.timer = 0.0
        self.last_move = move
        fgen = self.frame + LOOK_AHEAD
        ctx = np.stack([self.motion.sample(fgen * DT + n / PLANNER_FPS) for n in range(4)])
        self.pending = (from_planner(self._run_planner(ctx, move)), fgen, self.k + 1)

    # -- one tick ---------------------------------------------------------------------------------------------------
    def __call__(self, st: dict, cmd: SonicCommand) -> np.ndarray:
        if self.root_source is not None:
            self.root_est = np.asarray(self.root_source(st, None), float)
        self.move = cmd.movement
        q = np.asarray(st["dof_pos"], float)[self.to_mj]
        dq = np.asarray(st["dof_vel"], float)[self.to_mj]
        quat = np.asarray(st["root_quat"], float)
        # 1. log
        self.hist.append(
            (
                np.asarray(st["ang_vel"], float),
                (q - DEFAULT_ANGLES)[MUJOCO_TO_ISAACLAB],
                dq[MUJOCO_TO_ISAACLAB],
                self.last_action.copy(),
                gravity(quat),
            )  # fmt: skip
        )
        # 2. planner
        if self.k % PLANNER_EVERY == 0:
            self._planner_tick(cmd.movement)
        # 3. heading anchor
        if self.reinit_heading:
            self.init_base, self.init_ref, self.reinit_heading = quat.copy(), self.motion.quat[self.frame].copy(), False
        elif self.frame == 0:
            self.init_ref = self.motion.quat[0].copy()
        self.apply = qmul(heading(self.init_base), qconj(heading(self.init_ref)))
        # 4-5. encoder, decoder
        self.token = self.enc.run(None, {"obs_dict": self.encoder_input(quat, cmd)[None].astype(np.float32)})[0][0]
        action = self.dec.run(None, {"obs_dict": self.decoder_input(self.token)[None].astype(np.float32)})[0][0]
        action = np.asarray(action, float)
        # 6. targets
        target = DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        self.last_action = action
        # 7. frame advancement
        self._advance(q)
        self.k += 1
        return target[self.from_mj]

    def encoder_input(self, base_quat, cmd: SonicCommand) -> np.ndarray:
        x = np.zeros(ENC_DIM)
        x[ENC["encoder_mode_4"]] = [TELEOP_MODE, 0.0, 0.0, 0.0]
        m, f = self.motion, self.frame
        rel = qmul(qconj(heading(base_quat)), qmul(self.apply, m.quat[f]))
        x[ENC["motion_anchor_orientation_heading"]] = rot6(rel)
        frames = np.minimum(f + FUTURE_STEP * np.arange(FUTURE), m.T - 1)
        x[ENC["motion_joint_positions_lowerbody_10frame_step5"]] = m.q[frames][:, LEGS].ravel()
        x[ENC["motion_joint_velocities_lowerbody_10frame_step5"]] = m.dq[frames][:, LEGS].ravel()
        x[ENC["vr_3point_local_target"]] = cmd.vr_pos
        x[ENC["vr_3point_local_orn_target"]] = cmd.vr_quat
        return x

    def decoder_input(self, token) -> np.ndarray:
        x = np.zeros(DEC_DIM)
        x[DEC["token_state"]] = token
        pad = HISTORY - len(self.hist)
        rows = [(np.zeros(3), np.zeros(29), np.zeros(29), np.zeros(29), np.array([0.0, 0.0, 1.0]))] * pad + list(
            self.hist
        )
        cols = list(zip(*rows))
        for name, c in zip(DECODER_BLOCKS[1:], (0, 1, 2, 3, 4)):
            x[DEC[name[0]]] = np.concatenate(cols[c])
        return x

    def _advance(self, q_meas: np.ndarray) -> None:
        if self.pending is not None and self.k >= self.pending[2]:
            new, fgen, _ = self.pending
            self.pending = None
            merged = merge(self.motion, new, self.frame, fgen)
            if merged is not None:
                self.motion, self.frame, self.readapt_stored = merged, 0, False
        nf = self.frame + 1
        if nf >= self.motion.T:
            nf = self.motion.T - 1
            if self.move.mode == IDLE:
                self._readapt(nf, q_meas)
        self.frame = nf

    def _readapt(self, f: int, q_meas: np.ndarray) -> None:
        """Hold the motion's last frame and let its legs follow the measured legs (cpp:3503-3564)."""
        legs = self.motion.q[f, LEGS]
        if not self.readapt_stored:
            self.readapt_orig, self.readapt_stored, self.readapt_state = legs.copy(), True, "idle"
        err = float(np.mean(np.abs(legs - q_meas[LEGS])))
        s = self.readapt_state
        if s == "idle":
            s = "adapting" if err > ADAPT_TRIGGER else "recovering" if err < RECOVER_TRIGGER else "idle"
        elif s == "adapting" and err < ADAPT_STOP:
            s = "idle"
        elif s == "recovering" and err > ADAPT_TRIGGER:
            s = "adapting"
        self.readapt_state = s
        if s == "adapting":
            self.motion.q[f, LEGS] = 0.98 * legs + 0.02 * q_meas[LEGS]
        elif s == "recovering":
            self.motion.q[f, LEGS] = 0.98 * legs + 0.02 * self.readapt_orig

    # -- the plan's own pose, for start-up and inspection -------------------------------------------------------------
    def plan_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(root position, root quaternion, 29 MuJoCo joints) of the playing frame, in the planner's world."""
        m, f = self.motion, self.frame
        return m.pos[f].copy(), m.quat[f].copy(), m.q[f].copy()


# -- the sim ---------------------------------------------------------------------------------------------------------
def sonic_meta(meta: dict) -> dict:
    """ScaleBFM's metadata (joint order kept) with SONIC's gains, effort limits and default pose."""
    idx = [MUJOCO_JOINT_NAMES.index(n) for n in meta["joint_names"]]
    out = dict(meta)
    out["stiffness"] = KP[idx].tolist()
    out["damping"] = KD[idx].tolist()
    out["torque_limit"] = EFFORT[idx].tolist()
    out["default_dof_pos"] = DEFAULT_ANGLES[idx].tolist()
    return out


class SonicSim(R.Sim):
    """`scalebfm.runtime.Sim` -- same scene, robot, joint order, 5 ms x 4 step and Dex1 PD -- with SONIC's joint PD."""

    def __init__(self, meta: dict, video=False, backend=None, perturb: dict | None = None):
        super().__init__(sonic_meta(meta), video=video, backend=backend, perturb=perturb)


__all__ = [
    "DEFAULT_ANGLES", "EFFORT", "IDEL_SQUAT", "IDLE", "KD", "KP", "Motion", "Movement", "SLOW_WALK", "SonicCommand",
    "SonicConfig", "SonicPolicy", "SonicSim", "WALK", "from_planner", "merge", "sonic_meta", "vr_targets",
]  # fmt: skip
