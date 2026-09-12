"""Simultaneous tool-point IK for two arms and one shared waist pitch."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import ARM_MJ


@dataclass(frozen=True)
class BimanualSolution:
    joints: np.ndarray
    errors: np.ndarray


class BimanualIK:
    """Targets and results are pelvis-relative; no object state or world pose is read."""

    sides = ("left", "right")

    def __init__(self, ik: ArmIK, *, waist_pitch_max: float, tool_offset=(0.1515, 0.0, 0.0)):
        self.ik = ik
        self.offset = np.asarray(tool_offset, dtype=float)
        self.indices = np.array([14, *ARM_MJ["left"], *ARM_MJ["right"]])
        self.cols = ik.dadr[self.indices]
        # Read physical ranges, never the single-arm solver's historical 1.6x waist bounds.
        ranges = ik.model.jnt_range[ik.jids].copy()
        self.lo, self.hi = ranges[:, 0], ranges[:, 1]
        self.lo[12:15] = 0.0
        self.hi[12:14] = 0.0
        self.hi[14] = min(float(waist_pitch_max), self.hi[14])
        for side in self.sides:
            arm = ARM_MJ[side]
            self.lo[arm[3]] = max(self.lo[arm[3]], np.radians(5))
            for index in (arm[2], arm[4]):
                self.lo[index] = max(self.lo[index], -np.radians(50))
                self.hi[index] = min(self.hi[index], np.radians(50))

    def tool_poses(self, joints):
        poses = [self.ik.fk(joints, side) for side in self.sides]
        return np.array([p + r @ self.offset for p, r in poses]), np.array([r for _, r in poses])

    def solve(self, joints, positions, rotations, *, iterations=25, posture=None) -> BimanualSolution:
        q = np.asarray(joints, dtype=float).copy()
        positions, rotations = np.asarray(positions), np.asarray(rotations)
        if q.shape != (29,) or positions.shape != (2, 3) or rotations.shape != (2, 3, 3):
            raise ValueError("expected 29 joints, two tool positions and two rotations")
        if not all(np.isfinite(x).all() for x in (q, positions, rotations)):
            raise ValueError("IK inputs must be finite")
        q[12:15] = np.clip(q[12:15], self.lo[12:15], self.hi[12:15])
        prior = q.copy() if posture is None else np.asarray(posture, dtype=float)
        model, data, mj = self.ik.model, self.ik.data, self.ik._mj
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(iterations):
            data.qpos[:] = 0
            data.qpos[self.ik.base_qadr + 3] = 1.0
            data.qpos[self.ik.qadr] = q
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            jacobians, residuals = [], []
            for index, side in enumerate(self.sides):
                bid = self.ik.palm_bid[side]
                rot = data.xmat[bid].reshape(3, 3)
                point = data.xpos[bid] + rot @ self.offset
                mj.mj_jac(model, data, jp, jr, point, bid)
                jacobians.append(np.vstack((jp[:, self.cols], 0.15 * jr[:, self.cols])))
                residuals.append(
                    np.r_[positions[index] - point, 0.15 * Rotation.from_matrix(rotations[index] @ rot.T).as_rotvec()]
                )
            J, error = np.vstack(jacobians), np.concatenate(residuals)
            regularizer = 0.008
            dq = np.linalg.solve(
                J.T @ J + regularizer * np.eye(len(self.cols)),
                J.T @ error + 0.0001 * (prior[self.indices] - q[self.indices]),
            )
            dq *= min(1.0, 0.15 / max(np.linalg.norm(dq), 1e-9))
            q[self.indices] = np.clip(q[self.indices] + dq, self.lo[self.indices], self.hi[self.indices])
            if np.max(np.abs(dq)) < 1e-5:
                break
        actual, _ = self.tool_poses(q)
        return BimanualSolution(q, np.linalg.norm(actual - positions, axis=1))
