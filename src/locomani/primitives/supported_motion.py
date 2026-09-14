"""A continuous supported reference with physical-time rate and acceleration bounds."""

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.whole_body_ik import WholeBodyIK


class SupportedMotion:
    def __init__(self, ik, state, config, reference_joints, reference_root, *, tool_offset, hand_weights=(1, 1)):
        self.body = WholeBodyIK(ik, state, config, tool_offset=tool_offset, hand_weights=hand_weights)
        root = Rotation.from_quat(np.asarray(reference_root)[[1, 2, 3, 0]]).as_matrix()
        pitch = Rotation.from_matrix(root).as_euler("xyz")[1]
        self.body.heading = root @ Rotation.from_euler("y", -pitch).as_matrix()
        self.x = np.r_[state["pelvis"]["pos"], pitch, reference_joints]
        self.body.put(self.x)
        self.x[:3] += np.mean(self.body.feet_pos - ik.data.xpos[self.body.feet], axis=0)
        self.measured_feet = self.body.feet_pos.copy()
        self.body.put(self.x)
        # A planner reference has an existing decoder tracking error. Preserve
        # its support frame at takeover; check actual planted feet separately.
        self.support_offset = ik.data.xpos[self.body.feet].copy() - self.measured_feet
        self.body.feet_pos = ik.data.xpos[self.body.feet].copy()
        self.body.feet_rot = ik.data.xmat[self.body.feet].reshape(2, 3, 3).copy()
        self.body.com0 = ik.data.subtree_com[self.body.pelvis, :2].copy()
        self.lo, self.hi = np.minimum(self.body.lo, self.x), np.maximum(self.body.hi, self.x)
        self.v = np.zeros(33)
        self.feedback = np.zeros((2, 3))
        self.rotation_feedback = np.zeros((2, 3))
        self.config = config
        self.posture = np.asarray(reference_joints).copy()
        self.previous_dt = None
        self.diagnostics = {}

    @property
    def joints(self):
        return self.x[4:].copy()

    def measured_foot_drift(self, state):
        ik, d = self.body.ik, self.body.ik.data
        d.qpos[ik.qadr] = state["joints"]
        d.qpos[ik.base_qadr : ik.base_qadr + 7] = [*state["pelvis"]["pos"], *state["pelvis"]["quat"]]
        ik._mj.mj_kinematics(ik.model, d)
        return np.linalg.norm(d.xpos[self.body.feet] - self.measured_feet, axis=1)

    @property
    def root(self):
        return Rotation.from_matrix(self.body.put(self.x)).as_quat()[[3, 0, 1, 2]]

    def step(self, positions, rotations, measured, dt, *, torso_rotation=None, measured_rotations=None):
        if not np.isfinite(dt) or dt <= 0 or dt > 0.5:
            raise ValueError("supported motion needs a fresh advancing physical clock")
        cfg = self.config
        error = np.clip(self.body.tools(self.x) - measured, -0.08, 0.08)
        self.feedback += (1 - np.exp(-dt / 0.4)) * (error - self.feedback)
        if measured_rotations is not None:
            reference_rotations = self.body.ik.data.xmat[self.body.hands].reshape(2, 3, 3)
            error_rot = Rotation.from_matrix(
                reference_rotations @ np.asarray(measured_rotations).transpose(0, 2, 1)
            ).as_rotvec()
            error_rot *= np.minimum(1, np.radians(20) / np.maximum(np.linalg.norm(error_rot, axis=1), 1e-9))[:, None]
            self.rotation_feedback += (1 - np.exp(-dt / 0.4)) * (error_rot - self.rotation_feedback)
        corrected_rotations = Rotation.from_rotvec(cfg.tracking_gain * self.rotation_feedback).as_matrix() @ rotations
        corrected_positions = np.asarray(positions) + cfg.tracking_gain * self.feedback
        speed = np.r_[[0.04] * 3, 0.6, [cfg.joint_speed] * 29]
        accel = np.r_[[0.2] * 3, cfg.joint_acceleration, [cfg.joint_acceleration] * 29]
        interval = dt if self.previous_dt is None else (self.previous_dt + dt) / 2
        self.previous_dt = dt
        vlo, vhi = np.maximum(-speed, self.v - accel * interval), np.minimum(speed, self.v + accel * interval)
        brake = accel / 2
        vlo = np.maximum(
            vlo, -np.maximum(0, np.sqrt(2 * brake * np.maximum(self.x - self.lo, 0)) - brake * max(dt, 0.1))
        )
        vhi = np.minimum(
            vhi, np.maximum(0, np.sqrt(2 * brake * np.maximum(self.hi - self.x, 0)) - brake * max(dt, 0.1))
        )
        lo = np.maximum(self.lo, self.x + vlo * dt)
        hi = np.minimum(self.hi, self.x + vhi * dt)
        lo = np.minimum(lo, self.hi - 1e-10)
        hi = np.maximum(hi, self.lo + 1e-10)
        lo = np.minimum(lo, hi - 1e-10)
        pose = self.body.solve(
            corrected_positions,
            corrected_rotations,
            seed=self.x + np.clip(self.v, vlo, vhi) * dt,
            bounds=(lo, hi),
            posture=self.posture,
            torso_rotation=torso_rotation,
            iterations=3,
        )
        following = self.body.seed.copy()
        self.v = (following - self.x) / dt
        self.x = following
        self.diagnostics = dict(foot_error=pose.foot_errors, com_shift=pose.com_shift, hand_error=pose.hand_errors)
        return pose
