"""A continuous supported reference with physical-time rate and acceleration bounds."""

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.whole_body_ik import WholeBodyIK


def _clip_norm(vector, limit):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector) * min(1.0, limit / max(norm, 1e-12))


def residual_deadband(residual, deadband):
    """Apply an isotropic deadband independently to each hand residual."""
    residual = np.asarray(residual, dtype=float)
    result = np.zeros_like(residual)
    for index in np.ndindex(residual.shape[:-1]):
        norm = float(np.linalg.norm(residual[index]))
        if norm > deadband:
            result[index] = residual[index] * (norm - deadband) / norm
    return result


def bounded_correction_step(value, rate, target, dt, speed_limit, acceleration_limit):
    """Approach vector targets with physical-time speed, acceleration and stopping limits."""
    value = np.asarray(value, dtype=float).copy()
    rate = np.asarray(rate, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    for index in np.ndindex(value.shape[:-1]):
        delta = target[index] - value[index]
        # A critically damped target acceleration avoids the biased ratcheting
        # that bang-bang acceleration develops under alternating tick noise.
        natural_frequency = acceleration_limit / speed_limit
        target_acceleration = natural_frequency**2 * delta - 2 * natural_frequency * rate[index]
        rate[index] += _clip_norm(target_acceleration, acceleration_limit) * dt
        rate[index] = _clip_norm(rate[index], speed_limit)
        value[index] += rate[index] * dt
    return value, rate


class SupportedMotion:
    def __init__(
        self,
        ik,
        state,
        config,
        reference_joints,
        reference_root,
        *,
        tool_offset,
        hand_weights=(1, 1),
        reference_velocity=None,
        root_velocity=None,
        root_angular_velocity=None,
        posture_reference=None,
        posture_weights=None,
        continuity_weight=0.0,
        joint_limits=None,  # (lo, hi) over the 29 joints, intersected with the model's (the planner's caps)
        base_range=None,  # WholeBodyIK base_range: freeze the virtual base near the measured pelvis
    ):
        self.body = WholeBodyIK(
            ik, state, config, tool_offset=tool_offset, hand_weights=hand_weights, com_weight=20.0,
            joint_limits=joint_limits,
            base_range=base_range,
        )  # fmt: skip
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
        if reference_velocity is not None:
            reference_velocity = np.asarray(reference_velocity, dtype=float)
            if reference_velocity.shape != (29,) or not np.isfinite(reference_velocity).all():
                raise ValueError("supported reference_velocity must contain 29 finite values")
            self.v[4:] = reference_velocity
        if root_velocity is not None:
            root_velocity = np.asarray(root_velocity, dtype=float)
            if root_velocity.shape != (3,) or not np.isfinite(root_velocity).all():
                raise ValueError("supported root_velocity must contain three finite values")
            self.v[:3] = root_velocity
        if root_angular_velocity is not None:
            root_angular_velocity = np.asarray(root_angular_velocity, dtype=float)
            if root_angular_velocity.shape != (3,) or not np.isfinite(root_angular_velocity).all():
                raise ValueError("supported root_angular_velocity must contain three finite values")
            # x[3] is pitch about the heading-rotated local y axis. RefTap's
            # quaternion difference is a world-frame angular velocity.
            self.v[3] = float(root[:, 1] @ root_angular_velocity)
        self.feedback = np.zeros((2, 3))
        self.rotation_feedback = np.zeros((2, 3))
        self.feedback_rate = np.zeros((2, 3))
        self.rotation_feedback_rate = np.zeros((2, 3))
        self.feedback_origin = self.rotation_origin = None
        self.translation_feedback_frozen = False
        self.rotation_feedback_frozen = False
        self.config = config
        self.posture = np.asarray(reference_joints, dtype=float).copy()
        self.posture_start = self.posture.copy()
        self.posture_target = None if posture_reference is None else np.asarray(posture_reference, dtype=float).copy()
        self.posture_weights_target = (
            None if posture_weights is None else np.asarray(posture_weights, dtype=float).copy()
        )
        if self.posture_target is not None and self.posture_target.shape != (29,):
            raise ValueError("supported posture_reference must contain 29 values")
        if self.posture_weights_target is not None and (
            self.posture_weights_target.shape != (29,)
            or not np.isfinite(self.posture_weights_target).all()
            or np.any(self.posture_weights_target < 0)
        ):
            raise ValueError("supported posture_weights must contain 29 finite nonnegative values")
        if (self.posture_target is None) != (self.posture_weights_target is None):
            raise ValueError("supported posture reference and weights must be provided together")
        if not np.isfinite(continuity_weight) or continuity_weight < 0:
            raise ValueError("supported continuity_weight must be finite and nonnegative")
        self.posture_weights = None if self.posture_target is None else np.full(29, 0.04)
        self.continuity_weight = float(continuity_weight)
        self.previous_dt = None
        self.diagnostics = dict(
            inherited_joint_speed=float(np.max(np.abs(self.v[4:]))),
            inherited_root_velocity=self.v[:3].copy(),
            inherited_root_pitch_velocity=float(self.v[3]),
        )

    def blend_posture(self, amount):
        """Blend the optional task posture prior without a phase-boundary jump."""
        if self.posture_target is None:
            return
        amount = float(np.clip(amount, 0.0, 1.0))
        amount = amount * amount * (3 - 2 * amount)
        self.posture = self.posture_start + amount * (self.posture_target - self.posture_start)
        self.posture_weights = 0.04 + amount * (self.posture_weights_target - 0.04)
        self.diagnostics["posture_blend"] = amount

    def retarget_posture(self, posture_reference, posture_weights):
        """Start a continuous posture blend from the current supported reference."""
        target = np.asarray(posture_reference, dtype=float)
        weights = np.asarray(posture_weights, dtype=float)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("supported posture_reference must contain 29 finite values")
        if weights.shape != (29,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("supported posture_weights must contain 29 finite nonnegative values")
        self.posture_start = self.joints
        self.posture = self.posture_start.copy()
        self.posture_target = target.copy()
        self.posture_weights_target = weights.copy()
        self.posture_weights = np.full(29, 0.04)
        self.diagnostics["posture_retargeted"] = True

    def freeze_posture(self):
        """Hold the current reference as the loaded posture prior."""
        self.posture = self.joints
        self.posture_start = self.posture.copy()
        self.posture_target = self.posture.copy()
        self.diagnostics["posture_frozen"] = True

    def freeze_translation_at_measured_tools(self, measured):
        """Freeze tracking feedback and brake toward the measured tool poses."""
        reference = self.body.tools(self.x).copy()
        measured = np.asarray(measured, dtype=float)
        if measured.shape != reference.shape or not np.isfinite(measured).all():
            raise ValueError("measured tools must match the finite active reference")
        if self.feedback_origin is None:
            self.feedback_origin = reference - measured
            self.feedback = self.feedback_origin.copy()
        self.translation_feedback_frozen = True
        self.feedback_rate.fill(0)
        self.diagnostics["translation_feedback_frozen"] = True
        return measured.copy(), reference

    def freeze_translation_feedback(self):
        """Hold the current translation compensation without moving the reference.

        Unlike `freeze_translation_at_measured_tools` this keeps the active
        reference where it is; it stops the bounded correction from chasing the
        tracking bias. At the loaded hold that chase moved the hand 7-17 mm per
        0.2 s at 4-8 cm/s, the compensation rate itself (stage3_7_governor8),
        so the measured quiet window could never form.
        """
        self.translation_feedback_frozen = True
        self.feedback_rate.fill(0)
        self.diagnostics["translation_feedback_frozen"] = True

    def resume_translation_feedback(self):
        """Resume bounded measured tracking correction without a rate jump."""
        self.translation_feedback_frozen = False
        self.diagnostics["translation_feedback_frozen"] = False

    def freeze_rotation_feedback(self):
        """Hold rotation compensation and discard a stale pre-close derivative."""
        self.rotation_feedback_frozen = True
        self.rotation_feedback_rate.fill(0.0)
        self.diagnostics["rotation_feedback_frozen"] = True

    def resume_rotation_feedback(self):
        """Resume bounded rotation correction from the stationary close hold."""
        self.rotation_feedback_frozen = False
        self.diagnostics["rotation_feedback_frozen"] = False

    @property
    def joints(self):
        return self.x[4:].copy()

    def measured_foot_positions(self, state):
        """Return measured world foot positions for support-snapshot comparisons."""
        ik, d = self.body.ik, self.body.ik.data
        d.qpos[ik.qadr] = state["joints"]
        d.qpos[ik.base_qadr : ik.base_qadr + 7] = [*state["pelvis"]["pos"], *state["pelvis"]["quat"]]
        ik._mj.mj_kinematics(ik.model, d)
        return d.xpos[self.body.feet].copy()

    def measured_foot_drift(self, state):
        return np.linalg.norm(self.measured_foot_positions(state) - self.measured_feet, axis=1)

    @property
    def root(self):
        return Rotation.from_matrix(self.body.put(self.x)).as_quat()[[3, 0, 1, 2]]

    def anchor_base(self, pelvis_pos, pelvis_quat_wxyz, measured_joints):
        """Put the virtual base on the measured pelvis and the legs on the measured leg joints (with a frozen
        base_range the base then stays within it); the arm and waist keep the commanded state."""
        x = self.body.anchor(pelvis_pos, pelvis_quat_wxyz, np.r_[np.asarray(measured_joints)[:12], self.x[16:]])
        self.x[:16] = x[:16]
        self.lo[:4], self.hi[:4] = self.body.lo[:4], self.body.hi[:4]

    def step(self, positions, rotations, measured, dt, *, torso_rotation=None, measured_rotations=None):
        if not np.isfinite(dt) or dt <= 0 or dt > 0.5:
            raise ValueError("supported motion needs a fresh advancing physical clock")
        cfg = self.config
        error = self.body.tools(self.x) - measured
        # "first" is the first step ever, for both loops: a frozen translation
        # loop must not leave the rotation loop without it (the simple expert
        # resumes rotation feedback at the lift with translation still frozen).
        first = self.feedback_origin is None
        if not self.translation_feedback_frozen:
            if first:
                self.feedback_origin = error.copy()
            # Preserve the decoder's existing reference/measurement offset at
            # takeover. Starting compensation at zero would move an otherwise held
            # hand just to erase that offset, then reverse as feedback warmed up.
            error = self.feedback_origin + np.clip(error - self.feedback_origin, -0.08, 0.08)
            alpha = 1 if first else 1 - np.exp(-dt / 0.4)
            if first:
                self.feedback = error.copy()
            elif cfg.tracking_correction_dynamics:
                target_feedback = self.feedback_origin + residual_deadband(
                    error - self.feedback_origin,
                    cfg.tracking_translation_deadband,
                )
                self.feedback, self.feedback_rate = bounded_correction_step(
                    self.feedback,
                    self.feedback_rate,
                    target_feedback,
                    dt,
                    cfg.tracking_translation_speed,
                    cfg.tracking_translation_acceleration,
                )
            else:
                self.feedback += alpha * (error - self.feedback)
        if measured_rotations is not None and not self.rotation_feedback_frozen:
            reference_rotations = self.body.ik.data.xmat[self.body.hands].reshape(2, 3, 3)
            error_rot = Rotation.from_matrix(
                reference_rotations @ np.asarray(measured_rotations).transpose(0, 2, 1)
            ).as_rotvec()
            if self.rotation_origin is None:
                self.rotation_origin = error_rot.copy()
                self.rotation_feedback = error_rot.copy()
            residual = error_rot - self.rotation_origin
            residual *= np.minimum(1, np.radians(20) / np.maximum(np.linalg.norm(residual, axis=1), 1e-9))[:, None]
            target_rotation_feedback = self.rotation_origin + residual
            if first:
                self.rotation_feedback = target_rotation_feedback.copy()
            elif cfg.tracking_correction_dynamics:
                target_rotation_feedback = self.rotation_origin + residual_deadband(
                    residual,
                    cfg.tracking_rotation_deadband,
                )
                self.rotation_feedback, self.rotation_feedback_rate = bounded_correction_step(
                    self.rotation_feedback,
                    self.rotation_feedback_rate,
                    target_rotation_feedback,
                    dt,
                    cfg.tracking_rotation_speed,
                    cfg.tracking_rotation_acceleration,
                )
            else:
                self.rotation_feedback += alpha * (target_rotation_feedback - self.rotation_feedback)
        corrected_rotations = Rotation.from_rotvec(cfg.tracking_gain * self.rotation_feedback).as_matrix() @ rotations
        corrected_positions = np.asarray(positions) + cfg.tracking_gain * self.feedback
        speed = np.r_[[0.04] * 3, 0.6, [cfg.joint_speed] * 29]
        accel = np.r_[[0.2] * 3, cfg.joint_acceleration, [cfg.joint_acceleration] * 29]
        interval = dt if self.previous_dt is None else (self.previous_dt + dt) / 2
        self.previous_dt = dt
        # An inherited reference can be just above the steady-state cap. Keep
        # the current speed inside the first acceleration cone and brake it;
        # clipping directly to ``speed`` would recreate the handoff reset.
        effective_speed = np.maximum(speed, np.abs(self.v))
        vlo = np.maximum(-effective_speed, self.v - accel * interval)
        vhi = np.minimum(effective_speed, self.v + accel * interval)
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
        predicted = self.x + np.clip(self.v, vlo, vhi) * dt
        pose = self.body.solve(
            corrected_positions,
            corrected_rotations,
            seed=predicted,
            bounds=(lo, hi),
            posture=self.posture,
            posture_weights=self.posture_weights,
            continuity_reference=predicted,
            continuity_dt=dt,
            continuity_weight=self.continuity_weight,
            torso_rotation=torso_rotation,
            iterations=3,
        )
        following = self.body.seed.copy()
        self.v = (following - self.x) / dt
        self.x = following
        self.diagnostics.update(
            foot_error=pose.foot_errors,
            com_shift=pose.com_shift,
            hand_error=pose.hand_errors,
            feedback=self.feedback.copy(),
            feedback_rate=self.feedback_rate.copy(),
            rotation_feedback=self.rotation_feedback.copy(),
            rotation_feedback_rate=self.rotation_feedback_rate.copy(),
            tracking_correction_dynamics=bool(cfg.tracking_correction_dynamics),
            translation_feedback_frozen=bool(self.translation_feedback_frozen),
            rotation_feedback_frozen=bool(self.rotation_feedback_frozen),
            **self.body.last_diagnostics,
        )
        return pose
