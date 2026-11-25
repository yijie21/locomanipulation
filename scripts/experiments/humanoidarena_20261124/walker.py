"""Walking references for HumanoidArena's SONIC: SONIC's own velocity planner (gear_sonic_deploy planner_sonic.onnx,
target_vel V2) run as the deploy runs it, its frames sent through the 40-D frame.

Why: HumanoidArena's SONIC path (joint29 / "g1" encoder mode) tracks a whole-body reference; nothing in its encoder
input carries the root's horizontal motion (2026-11-24 dev probe: 0.6 m of root xy deltas with standing legs moved the
robot 3 mm; a yaw ramp turned it 89 deg). It walks when the reference's legs walk, as the VLA's teleop-derived frames
do. The planner is the deploy's kinematic motion generator for exactly this (velocity commands -> a 30 Hz full-body
motion); it is a command source like the VLA, not part of the controller: SONIC's encoder/decoder and PD are untouched.

Ported from locomani.backends.sonic_onnx.runtime (SonicPolicy's planner part: Initialize from the measured joints, the
10 Hz replan rule, 2-frame look-ahead, the 8-frame crossfade merge, the idle re-adaptation of the held legs, the heading
anchor). Per 50 Hz tick `tick(move, q_meas)` gives the playing frame: root position (planner world), root quaternion
relative to the plan's own start heading (= relative to the robot's heading at reset: what the 40-D frame's rot6d
wants), root height and 29 joints (IsaacLab order). The caller splices its own arms (and waist) in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PLANNER = "/wuji-vefps-D/wuji-il/yijie/ws/code/loco/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
DT = 0.02
PLANNER_EVERY, LOOK_AHEAD, BLEND_FRAMES, PLANNER_FPS = 5, 2, 8, 30.0
INIT_ROOT_Z, SEED = 0.788740, 1234
ALLOWED_TOKENS = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64)
IDLE, SLOW_WALK, WALK, RUN = 0, 1, 2, 3
STATIC_MODES = frozenset({0, 4, 5, 6, 7, 8})
REPLAN_INTERVAL = {RUN: 0.1}
N_MODES = 27
ADAPT_TRIGGER, RECOVER_TRIGGER, ADAPT_STOP = 0.10, 0.045, 0.05
LEGS = np.arange(12)  # MuJoCo order
MJ = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip


def qmul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])  # fmt: skip


def qconj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], float)


def qmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def heading(q) -> np.ndarray:
    x = qmat(q)[:, 0]
    yaw = math.atan2(x[1], x[0])
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def slerp(q0, q1, t: float) -> np.ndarray:
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


@dataclass
class Motion:
    pos: np.ndarray
    quat: np.ndarray
    q: np.ndarray

    @property
    def T(self) -> int:
        return len(self.q)

    def sample(self, t: float) -> np.ndarray:
        f = t / DT
        f0 = min(int(math.floor(f)), self.T - 1)
        f1 = min(f0 + 1, self.T - 1)
        w = f - f0 if f1 > f0 else 0.0
        return np.r_[(1 - w) * self.pos[f0] + w * self.pos[f1], slerp(self.quat[f0], self.quat[f1], w),
                     (1 - w) * self.q[f0] + w * self.q[f1]]  # fmt: skip


def from_planner(qpos: np.ndarray) -> Motion:
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
    return Motion(pos, quat, q)


def merge(old: Motion, new: Motion, current: int, fgen: int) -> Motion | None:
    length = fgen - current + new.T
    if length <= 0:
        return None
    start = max(0, fgen - current)
    out = Motion(np.zeros((length, 3)), np.zeros((length, 4)), np.zeros((length, 29)))
    for f in range(length):
        f_old = min(max(f + current, 0), old.T - 1)
        f_new = min(max(f + current - fgen, 0), new.T - 1)
        w = min(max((f - start) / BLEND_FRAMES, 0.0), 1.0)
        out.q[f] = (1 - w) * old.q[f_old] + w * new.q[f_new]
        out.pos[f] = (1 - w) * old.pos[f_old] + w * new.pos[f_new]
        out.quat[f] = slerp(old.quat[f_old], new.quat[f_new], w)
    return out


@dataclass
class Movement:
    """mode, speed (-1: the mode's default), directions in the ROBOT's world (unit or zero), height (-1: none)."""

    mode: int = IDLE
    speed: float = -1.0
    movement: np.ndarray = field(default_factory=lambda: np.zeros(3))
    facing: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    height: float = -1.0


class Walker:
    def __init__(self, isaac_order, device_id: int = 0, path: str = PLANNER, lead: int = 0):
        import onnxruntime as ort
        import torch  # noqa: F401  (CUDA libraries for onnxruntime's CUDA provider)

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", {"device_id": int(device_id)})])
        self.lead = int(lead)  # frames: the reference sent is the plan this far ahead (SONIC tracks ~160 ms late)
        self.to_isaac = np.array([MJ.index(n) for n in isaac_order])  # MuJoCo index of each IsaacLab joint
        self.from_isaac = np.array([list(isaac_order).index(n) for n in MJ])  # IsaacLab index of each MuJoCo joint

    def _run(self, ctx, move: Movement, planner_frame=False) -> np.ndarray:
        def unit(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else np.zeros(3)

        mv = move.movement if planner_frame else qmat(self.apply).T @ np.asarray(move.movement, float)
        fc = move.facing if planner_frame else qmat(self.apply).T @ np.asarray(move.facing, float)
        mode = move.mode if 0 <= move.mode < N_MODES else IDLE
        feed = dict(
            context_mujoco_qpos=ctx[None].astype(np.float32), target_vel=np.array([move.speed], np.float32),
            mode=np.array([mode], np.int64), movement_direction=unit(mv)[None].astype(np.float32),
            facing_direction=unit(fc)[None].astype(np.float32), random_seed=np.array([SEED], np.int64),
            has_specific_target=np.zeros((1, 1), np.int64), specific_target_positions=np.zeros((1, 4, 3), np.float32),
            specific_target_headings=np.zeros((1, 4), np.float32), allowed_pred_num_tokens=ALLOWED_TOKENS,
            height=np.array([move.height], np.float32),
        )  # fmt: skip
        out = self.sess.run(None, feed)
        n = int(np.asarray(out[1]).ravel()[0])
        assert 0 < n <= 64 and np.isfinite(out[0][0, :n]).all(), "planner returned a bad motion"
        return np.asarray(out[0][0, :n], float)

    def reset(self, q_isaac, root_quat) -> None:
        """The deploy's Initialize from the measured joints: four identical context frames at the origin, identity
        root, IDLE; the heading anchor maps the plan's yaw-zero world onto the robot's."""
        q = np.asarray(q_isaac, float)[self.from_isaac]
        ctx = np.r_[0.0, 0.0, INIT_ROOT_Z, 1.0, 0.0, 0.0, 0.0, q]
        move = Movement()
        self.apply = np.array([1.0, 0.0, 0.0, 0.0])
        self.motion = from_planner(self._run(np.repeat(ctx[None], 4, 0), move, planner_frame=True))
        self.frame, self.k, self.timer, self.pending = 0, 0, 0.0, None
        self.readapt_stored, self.readapt_state, self.readapt_orig = False, "idle", np.zeros(12)
        self.init_ref = self.motion.quat[0].copy()
        self.apply = qmul(heading(np.asarray(root_quat, float)), qconj(heading(self.init_ref)))
        self.last_move = Movement(IDLE, -1.0, np.zeros(3), qmat(self.apply) @ move.facing, -1.0)
        self.move = self.last_move
        self.replans = 0

    def _planner_tick(self, move: Movement) -> None:
        self.timer += PLANNER_EVERY * DT
        a, b = move, self.last_move
        same = dict(mode=a.mode == b.mode, speed=a.speed == b.speed, movement=np.array_equal(a.movement, b.movement),
                    facing=np.array_equal(a.facing, b.facing), height=a.height == b.height)  # fmt: skip
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
        self.pending = (from_planner(self._run(ctx, move)), fgen, self.k + 1)
        self.replans += 1

    def tick(self, move: Movement, q_meas_isaac) -> dict:
        """This tick's reference (then the frame advances, as the deploy does after its encoder)."""
        self.move = move
        if self.k % PLANNER_EVERY == 0:
            self._planner_tick(move)
        m = self.motion
        f = min(self.frame + self.lead, m.T - 1)
        rel = qmul(qconj(heading(self.init_ref)), m.quat[f])  # relative to the plan's start heading
        nxt = min(f + 1, m.T - 1)
        d_world = m.pos[nxt] - m.pos[f]
        d_local = qmat(heading(m.quat[f])).T @ d_world
        out = dict(pos=m.pos[f].copy(), quat_rel=rel, z=float(m.pos[f, 2]), q=m.q[f][self.to_isaac].copy(),
                   dxy=d_local[:2].copy())  # fmt: skip
        self._advance(np.asarray(q_meas_isaac, float)[self.from_isaac])
        self.k += 1
        return out

    def _advance(self, q_meas) -> None:
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

    def _readapt(self, f: int, q_meas) -> None:
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


def rot6d_of(q) -> np.ndarray:
    """vla_smpl_runtime.quat_to_rot6d_wxyz: the rotation's first two columns, row-major."""
    m = qmat(q)
    return np.array([m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[2, 0], m[2, 1]], np.float32)
