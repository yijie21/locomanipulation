"""Low reach geometry with planted feet, COM support and a shallow pelvis posture.

The result is a geometric reference, not a guarantee that SONIC will track it.
All robot joint limits come from the loaded model; no object pose is altered.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK


@dataclass(frozen=True)
class WholeBodyPose:
    joints: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    hand_errors: np.ndarray
    foot_errors: np.ndarray
    com_shift: np.ndarray

    @property
    def reachable(self):
        return max(self.hand_errors) < 0.02 and max(self.foot_errors) < 0.005 and np.linalg.norm(self.com_shift) < 0.015


def plan_low_reach(ik: ArmIK, state, positions, rotations, config, *, tool_offset):
    """Solve from observed feet, hands and pelvis; minimize lowering and posture change."""
    model, data, mj = ik.model, ik.data, ik._mj
    q0 = np.asarray(state["joints"], dtype=float)
    base0 = np.asarray(state["pelvis"]["pos"], dtype=float)
    rot0 = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]])
    yaw = rot0.as_euler("xyz")[2]
    heading = Rotation.from_euler("z", yaw).as_matrix()
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    hands = [ik.palm_bid[side] for side in ("left", "right")]
    pelvis = model.body("pelvis").id
    data.qpos[:] = model.qpos0
    data.qpos[ik.qadr] = q0
    data.qpos[ik.base_qadr : ik.base_qadr + 3] = base0
    data.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = state["pelvis"]["quat"]
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    feet_pos, feet_rot = data.xpos[feet].copy(), data.xmat[feet].reshape(2, 3, 3).copy()
    com0 = data.subtree_com[pelvis, :2].copy()
    ranges = model.jnt_range[ik.jids].copy()
    ranges[14, 1] = min(ranges[14, 1], config.waist_pitch_max)
    ranges[[3, 9], 0] = 0.05
    ranges[[3, 9], 1] = np.minimum(ranges[[3, 9], 1], 0.85)
    ranges[[18, 25], 0] = np.maximum(ranges[[18, 25], 0], np.radians(5))
    if config.waist_dofs == "pitch":
        # least_squares needs strictly separated bounds; the FK still fixes these axes below.
        ranges[12:14] = [-1e-9, 1e-9]
    lo = np.r_[base0[:2] - [0.3, 0.15], config.pelvis_height_min, 0, ranges[:, 0]]
    hi = np.r_[
        base0[:2] + [0.2, 0.15],
        max(base0[2] + 0.02, config.pelvis_height_min + 0.01),
        config.pelvis_pitch_max,
        ranges[:, 1],
    ]
    x0 = np.r_[base0, 0.3, q0]
    x0[4 + np.array([0, 6])] -= 0.3
    x0[18] = min(0.45, config.waist_pitch_max)
    x0 = np.clip(x0, lo, hi)
    positions, rotations, offset = np.asarray(positions), np.asarray(rotations), np.asarray(tool_offset)

    def put(x):
        base = heading @ Rotation.from_euler("y", x[3]).as_matrix()
        data.qpos[ik.base_qadr : ik.base_qadr + 3] = x[:3]
        data.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = Rotation.from_matrix(base).as_quat()[[3, 0, 1, 2]]
        data.qpos[ik.qadr] = x[4:]
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        return base

    def hand_positions():
        return np.array([data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ offset for bid in hands])

    def residual(x):
        put(x)
        foot_rot = np.concatenate(
            [Rotation.from_matrix(data.xmat[b].reshape(3, 3) @ r.T).as_rotvec() for b, r in zip(feet, feet_rot)]
        )
        hand_rot = np.concatenate(
            [Rotation.from_matrix(data.xmat[b].reshape(3, 3) @ r.T).as_rotvec() for b, r in zip(hands, rotations)]
        )
        return np.r_[
            20 * (data.xpos[feet] - feet_pos).ravel(),
            3 * foot_rot,
            10 * (hand_positions() - positions).ravel(),
            0.6 * hand_rot,
            5 * (data.subtree_com[pelvis, :2] - com0),
            0.008 * (x[4:] - q0),
            0.3 * (x[2] - base0[2]),
        ]

    solution = least_squares(residual, x0, bounds=(lo, hi), max_nfev=120, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    rotation = put(solution.x)
    return WholeBodyPose(
        solution.x[4:].copy(),
        solution.x[:3].copy(),
        rotation,
        np.linalg.norm(hand_positions() - positions, axis=1),
        np.linalg.norm(data.xpos[feet] - feet_pos, axis=1),
        data.subtree_com[pelvis, :2].copy() - com0,
    )
