"""HEFT's policy input, vendored from LocoManiLab's g1_backend (`external/g1_backend` at 25d262a, "LocoManiLab
modifications to EGalahad/sim2real"): `sim2real/rl_policy/observations/heft.py` class `heft_policy_obs` (the PMG variant;
`heft_component_obs` only splits its `_build_obs_parts` into the four ONNX inputs), and the four quaternion helpers it
imports from `sim2real/utils/math.py`.

What is theirs, unchanged: the history buffers and their order, `update`, `_build_obs_parts` (every line of the
command, target and proprioception math), the float32 casts. What is ours: the class no longer derives from their
`motion_obs` / `Observation`, which only resolve names and hand over views of their RealtimeMotionBuffer. Here the
caller hands over the same three views itself (`set_reference`: the reference's joints, root position and root
quaternion at the policy's 13 reference steps) and the robot state their StateProcessor holds (`set_state`). The
reference root is the pelvis frame of the qpos frame: their buffer FKs each pushed qpos frame on robot_fk.xml, whose
`pelvis` body carries the free joint at its origin, so its pelvis pose is the frame's root (pos, quat) exactly.
Their StateProcessor stores the robot state cast to float32 (RobotState) in float64 arrays; `set_state` does the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np


# -- sim2real/utils/math.py (25d262a): the code unchanged, docstrings shortened ---------------------------------------
def quat_rotate_inverse_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    shape = v.shape
    assert q.shape[:-1] == v.shape[:-1], "q and v must have the same batch size"
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return (a - b + c).reshape(shape)


def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """Quaternions (w, x, y, z), shape (..., 4) -> rotation matrices (..., 3, 3) (pytorch3d's formula)."""
    original_shape = quaternions.shape[:-1]
    quaternions = quaternions.reshape(-1, 4)
    r, i, j, k = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return o.reshape(original_shape + (3, 3))


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """q1 * q2, both (w, x, y, z) of the same shape (..., 4)."""
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=-1).reshape(shape)


# -- observations/heft.py heft_policy_obs (25d262a) ------------------------------------------------------------------
class HeftObs:
    """HEFT G1 PMG policy input (their docstring: mirrors upstream ``TrackingPolicyRaw._build_obs_modules()`` order:
    boot indicator, command, target joints/root/gravity, robot histories, and previous raw actions).

    `joint_names`: the policy's joint order (policy.yaml `policy_joint_names`, the observation's `joint_names`);
    `state_joint_names`: the order of the state and reference arrays handed in (their RobotCfg / motion source order,
    MuJoCo's). The other arguments are the observation block of policy.yaml."""

    OBS_DIM = 1729
    INCLUDE_COMPLIANCE_FLAG = False
    COMPLIANCE_FLAG = False
    COMPLIANCE_FLAG_THRESHOLD = 10.0
    PARTS = ("context", "motion_command", "target_motion", "proprioception")  # policy.json in_keys

    def __init__(
        self,
        future_steps: Sequence[int],
        root_angvel_history_steps: Sequence[int],
        projected_gravity_history_steps: Sequence[int],
        joint_pos_history_steps: Sequence[int],
        joint_vel_history_steps: Sequence[int],
        prev_action_steps: int,
        joint_names: Sequence[str],
        state_joint_names: Sequence[str],
        root_body_name: str = "pelvis",
        boot_indicator_max: int = 25,
        **_component,  # `component` / `policy_variant` of heft_component_obs: the parts are all built here
    ) -> None:
        self.joint_names = list(joint_names)
        self.future_steps = np.asarray([int(step) for step in future_steps], dtype=np.int64)
        if self.future_steps.ndim != 1 or self.future_steps.size == 0:
            raise ValueError("future_steps must be a non-empty 1D sequence")
        if int(self.future_steps[0]) != 0:
            raise ValueError(f"future_steps[0] must be 0, got {self.future_steps.tolist()}")

        seen_negative = False
        for step in self.future_steps[1:].tolist():
            if int(step) < 0:
                seen_negative = True
            elif seen_negative:
                raise ValueError(
                    "future_steps format must be [0, ...positive/non-negative, ...negative], "
                    f"got {self.future_steps.tolist()}"
                )

        self.root_angvel_history_steps = [int(step) for step in root_angvel_history_steps]
        self.projected_gravity_history_steps = [int(step) for step in projected_gravity_history_steps]
        self.joint_pos_history_steps = [int(step) for step in joint_pos_history_steps]
        self.joint_vel_history_steps = [int(step) for step in joint_vel_history_steps]
        self.prev_action_steps = int(prev_action_steps)
        if self.prev_action_steps <= 0:
            raise ValueError("prev_action_steps must be positive")

        self.boot_indicator_max = int(boot_indicator_max)
        if self.boot_indicator_max <= 0:
            raise ValueError(f"boot_indicator_max must be positive, got {self.boot_indicator_max}")
        self._boot_indicator_value = self.boot_indicator_max

        if len(self.joint_names) != 29:
            raise ValueError(f"HEFT G1 tracking expects 29 joints, got {len(self.joint_names)}")

        # their StateProcessor: qpos (root 7 + joints) / qvel (root 6 + joints) arrays and views into them
        n = len(state_joint_names)
        qpos, qvel = np.zeros(7 + n), np.zeros(6 + n)
        self.state_processor = SimpleNamespace(
            joint_names=list(state_joint_names), qpos=qpos, qvel=qvel, root_quat_w=qpos[3:7], root_ang_vel_b=qvel[3:6],
            joint_pos=qpos[7:], joint_vel=qvel[6:],
        )  # fmt: skip
        self._state_joint_indices = [self.state_processor.joint_names.index(name) for name in self.joint_names]
        self._ref_joint_indices = self._state_joint_indices  # motion_obs._joint_indices: the same source order here
        self.future_step_indices = np.arange(self.future_steps.size)  # the reference is handed over in this order
        self.ref_joint_pos_future = np.zeros((1, self.future_steps.size, 29), np.float32)
        self.ref_root_pos_future_w = np.zeros((1, self.future_steps.size, 3), np.float32)
        self.ref_root_quat_future_w = np.tile(np.array([1.0, 0, 0, 0], np.float32), (1, self.future_steps.size, 1))

        self._root_angvel_history = np.zeros(
            (max(self.root_angvel_history_steps) + 1, 3),
            dtype=np.float32,
        )
        self._projected_gravity_history = np.zeros(
            (max(self.projected_gravity_history_steps) + 1, 3),
            dtype=np.float32,
        )
        self._joint_pos_history = np.zeros(
            (max(self.joint_pos_history_steps) + 1, len(self.joint_names)),
            dtype=np.float32,
        )
        self._joint_vel_history = np.zeros(
            (max(self.joint_vel_history_steps) + 1, len(self.joint_names)),
            dtype=np.float32,
        )
        self._prev_actions = np.zeros(
            (self.prev_action_steps, len(self.joint_names)),
            dtype=np.float32,
        )
        self._obs = np.zeros((1, self.OBS_DIM), dtype=np.float32)

    # -- ours: what motion_obs and the StateProcessor provided ------------------------------------------------------
    def set_state(self, joint_pos, joint_vel, root_quat_w, root_ang_vel_b) -> None:
        """The robot this tick: joints and joint velocities (state_joint_names order), the pelvis quaternion (wxyz,
        world) and the pelvis angular velocity in its own frame (MuJoCo's free-joint qvel[3:6], as their host reads it)."""
        sp = self.state_processor
        sp.qpos[3:7] = np.asarray(root_quat_w, np.float32)
        sp.qpos[7:] = np.asarray(joint_pos, np.float32)
        sp.qvel[3:6] = np.asarray(root_ang_vel_b, np.float32)
        sp.qvel[6:] = np.asarray(joint_vel, np.float32)

    def set_reference(self, joint_pos, root_pos_w, root_quat_w) -> None:
        """The reference at the policy's steps (`future_steps` order, here 0..6 then -1..-16 frames): joints
        (steps, 29) in state_joint_names order, the reference pelvis position (steps, 3) and quaternion (steps, 4)."""
        jp = np.asarray(joint_pos, np.float32)
        self.ref_joint_pos_future = jp[None][:, :, self._ref_joint_indices]
        self.ref_root_pos_future_w = np.asarray(root_pos_w, np.float32)[None]
        self.ref_root_quat_future_w = np.asarray(root_quat_w, np.float32)[None]

    def _select(self, x: np.ndarray) -> np.ndarray:
        return np.take(x, self.future_step_indices, axis=1)

    # -- theirs ----------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._boot_indicator_value = self.boot_indicator_max
        self._root_angvel_history[:] = 0.0
        self._projected_gravity_history[:] = 0.0
        self._joint_pos_history[:] = 0.0
        self._joint_vel_history[:] = 0.0
        self._prev_actions[:] = 0.0
        self._obs[:] = 0.0

    def _current_joint_pos(self) -> np.ndarray:
        return np.asarray(
            self.state_processor.joint_pos[self._state_joint_indices],
            dtype=np.float32,
        )

    def _current_joint_vel(self) -> np.ndarray:
        return np.asarray(
            self.state_processor.joint_vel[self._state_joint_indices],
            dtype=np.float32,
        )

    def _current_projected_gravity(self) -> np.ndarray:
        gravity = quat_rotate_inverse_numpy(
            np.asarray(self.state_processor.root_quat_w, dtype=np.float32).reshape(1, 4),
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        )[0]
        return (gravity / (np.linalg.norm(gravity) + 1.0e-8)).astype(np.float32)

    @staticmethod
    def _append_history(history: np.ndarray, value: np.ndarray) -> None:
        history[:] = np.roll(history, 1, axis=0)
        history[0] = value

    def _compliance_obs(self) -> np.ndarray:
        value = 1.0 if self.COMPLIANCE_FLAG else 0.0
        threshold = float(self.COMPLIANCE_FLAG_THRESHOLD)
        kp = threshold / 0.05
        return np.asarray([value, value * threshold, value * kp], dtype=np.float32)

    def update(self, data: dict) -> None:
        """One policy step: `data["action"]` is the previous step's raw action (policy order), as their state_dict."""
        self._boot_indicator_value = max(self._boot_indicator_value - 1, 0)
        self._append_history(
            self._root_angvel_history,
            np.asarray(self.state_processor.root_ang_vel_b, dtype=np.float32),
        )
        self._append_history(self._projected_gravity_history, self._current_projected_gravity())
        self._append_history(self._joint_pos_history, self._current_joint_pos())
        self._append_history(self._joint_vel_history, self._current_joint_vel())

        prev_action = np.asarray(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if prev_action.shape[0] != len(self.joint_names):
            raise ValueError(
                f"Previous action dim mismatch: expected {len(self.joint_names)}, got {prev_action.shape[0]}"
            )
        self._prev_actions[:] = np.roll(self._prev_actions, 1, axis=0)
        self._prev_actions[0, :] = prev_action

        self._obs[0, :] = self._build_obs()

    def _build_obs_parts(self) -> dict[str, np.ndarray]:
        ref_joint_pos = np.asarray(self._select(self.ref_joint_pos_future)[0], dtype=np.float32)
        ref_root_pos_w = np.asarray(self._select(self.ref_root_pos_future_w)[0], dtype=np.float32)
        ref_root_quat_w = np.asarray(self._select(self.ref_root_quat_future_w)[0], dtype=np.float32)

        pos_diff_w = ref_root_pos_w[1:] - ref_root_pos_w[0:1]
        base_ref_quat = np.broadcast_to(ref_root_quat_w[0:1], (pos_diff_w.shape[0], 4))
        pos_diff_b = quat_rotate_inverse_numpy(base_ref_quat, pos_diff_w)

        robot_root_quat_w = np.asarray(self.state_processor.root_quat_w, dtype=np.float32)
        robot_root_quat_w = np.broadcast_to(robot_root_quat_w.reshape(1, 4), ref_root_quat_w.shape)
        rel_quat = quat_mul(quat_conjugate(robot_root_quat_w), ref_root_quat_w)
        rel_rot = matrix_from_quat(rel_quat)
        rot6d = rel_rot[:, :, :2].transpose(0, 2, 1).reshape(-1).astype(np.float32)
        command = np.concatenate([pos_diff_b.reshape(-1), rot6d], axis=0)

        cur_joint_pos = self._current_joint_pos().reshape(1, -1)
        target_joint = np.concatenate(
            [
                ref_joint_pos.reshape(-1),
                (ref_joint_pos - cur_joint_pos).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)

        down = np.broadcast_to(
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
            (ref_root_quat_w.shape[0], 3),
        )
        target_projected_gravity = quat_rotate_inverse_numpy(ref_root_quat_w, down).reshape(-1)

        parts = {
            "context": np.asarray(
                [self._boot_indicator_value / self.boot_indicator_max],
                dtype=np.float32,
            ),
            "motion_command": command.astype(np.float32),
        }
        if self.INCLUDE_COMPLIANCE_FLAG:
            parts["compliance_context"] = self._compliance_obs()
        parts.update(
            {
                "target_motion": np.concatenate(
                    [
                        target_joint,
                        ref_root_pos_w[:, 2].astype(np.float32),
                        target_projected_gravity.astype(np.float32),
                    ],
                    axis=0,
                ).astype(np.float32),
                "proprioception": np.concatenate(
                    [
                        self._root_angvel_history[self.root_angvel_history_steps].reshape(-1),
                        self._projected_gravity_history[self.projected_gravity_history_steps].reshape(-1),
                        self._joint_pos_history[self.joint_pos_history_steps].reshape(-1),
                        self._joint_vel_history[self.joint_vel_history_steps].reshape(-1),
                        self._prev_actions.reshape(-1),
                    ],
                    axis=0,
                ).astype(np.float32),
            }
        )
        return parts

    def _build_obs(self) -> np.ndarray:
        obs = np.concatenate(
            list(self._build_obs_parts().values()),
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self.OBS_DIM:
            raise ValueError(f"HEFT G1 tracking obs dim mismatch: {obs.shape[0]} != {self.OBS_DIM}")
        return obs

    def compute(self) -> np.ndarray:
        return self._obs

    # -- ours: heft_component_obs.compute for each ONNX input --------------------------------------------------------
    def inputs(self) -> dict[str, np.ndarray]:
        """The four ONNX inputs, each (1, n), as heft_component_obs gives them (after `update`)."""
        parts = self._build_obs_parts()
        return {k: parts[k][None, :] for k in self.PARTS}


__all__ = ["HeftObs", "matrix_from_quat", "quat_conjugate", "quat_mul", "quat_rotate_inverse_numpy"]
