"""Continuous SE(3) commands shared by expert and deployed execution.

This module has no simulator, object pose, phase timer or policy dependency.
Future sampling uses copies: only the first 50 Hz tick commits controller state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def bounded(vector: np.ndarray, limit: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-12))


@dataclass(frozen=True)
class MotionLimits:
    linear_speed: float = 0.6
    linear_acceleration: float = 2.0
    angular_speed: float = 2.5
    angular_acceleration: float = 12.0
    gain: float = 6.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Motion limits must be finite and positive")


class PoseReference:
    """Rate/acceleration-limited pose servo, with angular velocity in world axes.

    Initialize from the previous executed command when available, or the measured
    wrist at takeover. Quaternion arrays use MuJoCo/ScaleBFM wxyz throughout.
    """

    def __init__(self, position, quaternion, limits: MotionLimits | None = None):
        self.position = np.asarray(position, dtype=float).copy()
        q = np.asarray(quaternion, dtype=float)
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("Position must be a finite 3-vector")
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
            raise ValueError("Quaternion must be a finite nonzero wxyz 4-vector")
        self.rotation = Rotation.from_quat(q, scalar_first=True)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.limits = limits or MotionLimits()

    def copy(self):
        out = PoseReference(self.position, self.quaternion, self.limits)
        out.velocity = self.velocity.copy()
        out.angular_velocity = self.angular_velocity.copy()
        return out

    @property
    def quaternion(self):
        return self.rotation.as_quat(scalar_first=True)

    def advance(self, position, quaternion, dt: float):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        goal = PoseReference(position, quaternion, self.limits)
        cfg = self.limits
        desired_v = bounded(cfg.gain * (goal.position - self.position), cfg.linear_speed)
        self.velocity += bounded(desired_v - self.velocity, cfg.linear_acceleration * dt)
        self.position += self.velocity * dt
        error = (goal.rotation * self.rotation.inv()).as_rotvec()
        desired_w = bounded(cfg.gain * error, cfg.angular_speed)
        self.angular_velocity += bounded(desired_w - self.angular_velocity, cfg.angular_acceleration * dt)
        self.rotation = Rotation.from_rotvec(self.angular_velocity * dt) * self.rotation
        return self.position.copy(), self.quaternion

    def futures(self, positions, quaternions, dt: float):
        """Commit one step, preview remaining uniformly spaced targets without mutation."""
        positions, quaternions = np.asarray(positions), np.asarray(quaternions)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("positions must be nonempty [F,3]")
        if quaternions.shape != (len(positions), 4):
            raise ValueError("quaternions must be [F,4]")
        # Validate the entire chunk before committing any state.
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
            raise ValueError("Nonfinite future target")
        if np.any(np.linalg.norm(quaternions, axis=1) < 1e-12):
            raise ValueError("Zero future quaternion")
        first = self.advance(positions[0], quaternions[0], dt)
        preview = self.copy()
        values = [first]
        for p, q in zip(positions[1:], quaternions[1:]):
            values.append(preview.advance(p, q, dt))
        return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])


class PoseTransition:
    """Join a newly generated pad path to the previously executed wrist command.

    The planner provides elapsed transition progress; this is not an extra
    deployed filter. Record its OUTPUT as the demonstration action. Corrections
    decay to zero, avoiding a persistent low-pass lag on the intended trajectory.
    """

    def __init__(self, previous, raw_start, duration: float, pad_offset=(0.1515, 0, 0)):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Transition duration must be finite and positive")
        self.duration = duration
        self.offset = np.asarray(pad_offset, dtype=float)
        self.start = PoseReference(*previous)
        raw = PoseReference(*raw_start)
        self.pad_correction = (self.start.position + self.start.rotation.apply(self.offset)
                               - raw.position - raw.rotation.apply(self.offset))

    def sample(self, position, quaternion, elapsed: float):
        raw = PoseReference(position, quaternion)
        u = float(np.clip(elapsed / self.duration, 0, 1))
        blend = float(np.clip(10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5, 0, 1))
        delta = (raw.rotation * self.start.rotation.inv()).as_rotvec()
        rotation = Rotation.from_rotvec(delta * blend) * self.start.rotation
        pad = raw.position + raw.rotation.apply(self.offset) + (1 - blend) * self.pad_correction
        return pad - rotation.apply(self.offset), rotation.as_quat(scalar_first=True)
