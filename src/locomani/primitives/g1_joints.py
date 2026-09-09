"""G1 29-DoF joint orderings and the MuJoCo <-> IsaacLab permutation.

Two orderings coexist in the SONIC stack (policy_parameters.hpp:6-15):

- MuJoCo order: URDF tree order — left leg, right leg, waist, left arm,
  right arm. Used by the simulator and by reference-motion *files*.
- IsaacLab order: breadth-first with left/right interleaved. Used inside the
  policy, and — the part that matters here — by the `pose` topic's
  `joint_pos` field. pico_manager_thread_server.py writes wrist angles at
  indices 23/25/27 (left r/p/y) and 24/26/28 (right), which is IsaacLab
  order; localmotion_kplanner.hpp:675 confirms streamed JointPositions are
  indexed through mujoco_to_isaaclab when written into MuJoCo qpos.

The arrays below are copied from policy_parameters.hpp:100-104. Their names
are kept even though they read backwards: MUJOCO_TO_ISAACLAB[isaac] = mujoco
index (verified: element 23 -> 19, left wrist roll in both conventions).
"""

from __future__ import annotations

import numpy as np

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# policy_parameters.hpp:103 — indexed by IsaacLab index, yields MuJoCo index.
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)

# policy_parameters.hpp:100 — indexed by MuJoCo index, yields IsaacLab index.
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)

assert (MUJOCO_TO_ISAACLAB[ISAACLAB_TO_MUJOCO] == np.arange(29)).all()

# MuJoCo indices of each arm's 7 joints (shoulder p/r/y, elbow, wrist r/p/y).
ARM_MJ = {"left": list(range(15, 22)), "right": list(range(22, 29))}


def mujoco_to_isaac(q_mj: np.ndarray) -> np.ndarray:
    """Reorder a 29-vector (or [N,29]) from MuJoCo order to IsaacLab order."""
    q_mj = np.asarray(q_mj)

    return q_mj[..., MUJOCO_TO_ISAACLAB]


def isaac_to_mujoco(q_isaac: np.ndarray) -> np.ndarray:
    q_isaac = np.asarray(q_isaac)

    return q_isaac[..., ISAACLAB_TO_MUJOCO]
