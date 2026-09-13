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


class WholeBodyIK:
    """A fixed support problem, reusable for admission and continuous paired motion."""

    def __init__(self, ik: ArmIK, state, config, *, tool_offset, joint_limits=None):
        self.ik = ik
        self.q0 = np.asarray(state["joints"], dtype=float)
        self.base0 = np.asarray(state["pelvis"]["pos"], dtype=float)
        rot0 = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
        pitch0 = Rotation.from_matrix(rot0).as_euler("xyz")[1]
        self.heading = rot0 @ Rotation.from_euler("y", -pitch0).as_matrix()
        m, d, mj = ik.model, ik.data, ik._mj
        self.feet = [m.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self.hands = [ik.palm_bid[side] for side in ("left", "right")]
        self.pelvis = m.body("pelvis").id
        self.offset = np.asarray(tool_offset)
        d.qpos[:] = m.qpos0
        self.initial = np.r_[self.base0, pitch0, self.q0]
        self.put(self.initial)
        self.feet_pos = d.xpos[self.feet].copy()
        self.feet_rot = d.xmat[self.feet].reshape(2, 3, 3).copy()
        self.com0 = d.subtree_com[self.pelvis, :2].copy()
        ranges = m.jnt_range[ik.jids].copy()
        ranges[14, 1] = min(ranges[14, 1], config.waist_pitch_max)
        ranges[[3, 9], 0] = 0.05
        ranges[[3, 9], 1] = np.minimum(ranges[[3, 9], 1], 0.85)
        ranges[[18, 25], 0] = np.maximum(ranges[[18, 25], 0], np.radians(5))
        if joint_limits is not None:
            ranges[:, 0] = np.maximum(ranges[:, 0], joint_limits[0])
            ranges[:, 1] = np.minimum(ranges[:, 1], joint_limits[1])
        if config.waist_dofs == "pitch":
            ranges[12:14] = [-1e-9, 1e-9]
        self.lo = np.r_[self.base0[:2] - [0.3, 0.15], config.pelvis_height_min, min(0, pitch0), ranges[:, 0]]
        self.hi = np.r_[
            self.base0[:2] + [0.2, 0.15],
            max(self.base0[2] + 0.02, config.pelvis_height_min + 0.01),
            config.pelvis_pitch_max,
            ranges[:, 1],
        ]
        self.seed = self.initial.copy()
        self.seed[3] = 0.3
        self.seed[4 + np.array([0, 6])] -= 0.3
        self.seed[18] = min(0.45, config.waist_pitch_max)
        self.seed = np.clip(self.seed, self.lo, self.hi)

    def put(self, x):
        ik = self.ik
        base = self.heading @ Rotation.from_euler("y", x[3]).as_matrix()
        ik.data.qpos[ik.base_qadr : ik.base_qadr + 3] = x[:3]
        ik.data.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = Rotation.from_matrix(base).as_quat()[[3, 0, 1, 2]]
        ik.data.qpos[ik.qadr] = x[4:]
        ik._mj.mj_kinematics(ik.model, ik.data)
        ik._mj.mj_comPos(ik.model, ik.data)
        return base

    def tools(self, x):
        self.put(x)
        d = self.ik.data
        return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.offset for b in self.hands])

    def solve(self, positions, rotations, *, seed=None, bounds=None, posture=None, iterations=120):
        d = self.ik.data
        positions, rotations = np.asarray(positions), np.asarray(rotations)
        prior = self.q0 if posture is None else posture

        def residual(x):
            points = self.tools(x)
            foot_rot = np.concatenate(
                [
                    Rotation.from_matrix(d.xmat[b].reshape(3, 3) @ r.T).as_rotvec()
                    for b, r in zip(self.feet, self.feet_rot)
                ]
            )
            hand_rot = np.concatenate(
                [Rotation.from_matrix(d.xmat[b].reshape(3, 3) @ r.T).as_rotvec() for b, r in zip(self.hands, rotations)]
            )
            return np.r_[
                20 * (d.xpos[self.feet] - self.feet_pos).ravel(),
                3 * foot_rot,
                10 * (points - positions).ravel(),
                0.6 * hand_rot,
                5 * (d.subtree_com[self.pelvis, :2] - self.com0),
                (0.008 if posture is None else 0.04) * (x[4:] - prior),
                0.3 * (x[2] - self.base0[2]),
            ]

        lo, hi = (self.lo, self.hi) if bounds is None else bounds
        x0 = np.clip(self.seed if seed is None else seed, lo, hi)
        result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=iterations, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        self.seed = result.x.copy()
        rotation = self.put(result.x)
        return WholeBodyPose(
            result.x[4:].copy(),
            result.x[:3].copy(),
            rotation,
            np.linalg.norm(self.tools(result.x) - positions, axis=1),
            np.linalg.norm(d.xpos[self.feet] - self.feet_pos, axis=1),
            d.subtree_com[self.pelvis, :2].copy() - self.com0,
        )


def plan_low_reach(ik: ArmIK, state, positions, rotations, config, *, tool_offset):
    """Solve from observed feet, hands and pelvis; minimize lowering and posture change."""
    return WholeBodyIK(ik, state, config, tool_offset=tool_offset).solve(positions, rotations)
