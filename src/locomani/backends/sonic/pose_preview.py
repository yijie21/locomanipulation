"""Physical-time POSE samples with the future window required by SONIC.

The encoder consumes frames through ``current + 45``. Every packet therefore
contains a short reference history followed by a bounded trajectory that
decelerates to rest. The velocity stored for a predicted frame is the finite
difference velocity that produced it.
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PosePreview:
    dt = 0.02
    future_frames = 50
    history_frames = 5

    def __init__(self, *, acceleration, joint_limits, continuous_handoff=False):
        self.acceleration = float(acceleration)
        self.lo, self.hi = (np.asarray(value, dtype=float).copy() for value in joint_limits)
        if (
            not np.isfinite(self.acceleration)
            or self.acceleration <= 0
            or self.lo.shape != (29,)
            or self.hi.shape != (29,)
            or not np.isfinite(self.lo).all()
            or not np.isfinite(self.hi).all()
            or np.any(self.lo > self.hi)
        ):
            raise ValueError("preview requires positive acceleration and 29-joint bounds")
        self.history = deque(maxlen=self.history_frames + 1)
        self.origin = self.previous = None
        self.continuous_handoff = bool(continuous_handoff)
        self.diagnostics = {}

    @staticmethod
    def _rotation(value):
        value = np.asarray(value, dtype=float)
        if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) < 0.5:
            raise ValueError("preview root must be a finite wxyz quaternion")
        return Rotation.from_quat(value[[1, 2, 3, 0]])

    def sample(self, stamp, joints, velocity, root, *, root_angular_velocity=None):
        q, v = np.asarray(joints, dtype=float).copy(), np.asarray(velocity, dtype=float).copy()
        if q.shape != (29,) or v.shape != (29,) or not np.isfinite([q, v]).all():
            raise ValueError("preview joints and velocity must contain 29 finite values")
        root = self._rotation(root)
        supplied_angular = root_angular_velocity is not None
        angular = (
            np.zeros(3) if root_angular_velocity is None else np.asarray(root_angular_velocity, dtype=float).copy()
        )
        if angular.shape != (3,) or not np.isfinite(angular).all():
            raise ValueError("root_angular_velocity must contain three finite values")
        stamp = float(stamp)
        if not np.isfinite(stamp):
            raise ValueError("preview reference time must be finite")
        if self.origin is None:
            self.origin = stamp
        tick = round((stamp - self.origin) / self.dt)

        if self.previous is None:
            self.history.append((tick, q.copy(), v.copy(), root))
        else:
            old_tick, old_q, old_v, old_root = self.previous
            if tick < old_tick:
                raise ValueError("POSE reference time moved backwards")
            if tick > old_tick:
                duration = (tick - old_tick) * self.dt
                if not supplied_angular:
                    angular = (root * old_root.inv()).as_rotvec() / duration
                rotations = Slerp([0, 1], Rotation.concatenate([old_root, root]))
                for index in range(max(old_tick + 1, tick - self.history_frames), tick + 1):
                    u = (index - old_tick) / (tick - old_tick)
                    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
                    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
                    qi = (
                        h00 * old_q + h10 * duration * old_v + h01 * q + h11 * duration * v
                        if self.continuous_handoff
                        else old_q + u * (q - old_q)
                    )
                    if self.continuous_handoff:
                        qi = np.clip(qi, np.minimum(old_q, q), np.maximum(old_q, q))
                    self.history.append((index, qi, v.copy(), rotations(u)))
            else:
                self.history[-1] = (tick, q.copy(), v.copy(), root)
        self.previous = tick, q.copy(), v.copy(), root

        samples = list(self.history)
        current_index = len(samples) - 1
        speed0 = max(float(np.max(np.abs(v))), float(np.linalg.norm(angular)))
        endpoint_v, endpoint_angular = v.copy(), angular.copy()
        effective_lo, effective_hi = np.minimum(self.lo, q), np.maximum(self.hi, q)
        clipped_axes = set()
        for offset in range(1, self.future_frames + 1):
            if self.continuous_handoff:
                # One braking progress preserves the whole-body tangent: limb,
                # waist and root do not stop in different proportions.
                scale = max(0.0, 1 - offset * self.dt / (speed0 / self.acceleration)) if speed0 else 0.0
                next_v = v * scale
                interval_v = 0.5 * (endpoint_v + next_v)
            else:
                scale = max(0.0, 1 - offset * self.dt / (speed0 / self.acceleration)) if speed0 else 0.0
                next_v = v * scale
                interval_v = next_v
            following = q + interval_v * self.dt
            clipped = (following < effective_lo) | (following > effective_hi)
            if np.any(clipped):
                clipped_axes.update(np.flatnonzero(clipped).tolist())
                following = np.clip(following, effective_lo, effective_hi)
                interval_v = (following - q) / self.dt
                next_v[clipped] = 0.0

            next_angular = angular * scale
            interval_angular = 0.5 * (endpoint_angular + next_angular) if self.continuous_handoff else next_angular
            root = Rotation.from_rotvec(interval_angular * self.dt) * root
            q, endpoint_v, endpoint_angular = following, next_v, next_angular
            samples.append((tick + offset, q.copy(), interval_v.copy(), root))

        positions = np.array([sample[1] for sample in samples])
        velocities = np.array([sample[2] for sample in samples])
        if self.continuous_handoff and len(samples) > 1:
            velocities[1:] = np.diff(positions, axis=0) / self.dt
        roots = np.array([sample[3].as_quat()[[3, 0, 1, 2]] for sample in samples])
        self.diagnostics = dict(
            current_offset=current_index,
            future_frames=self.future_frames,
            braking_time=float(speed0 / self.acceleration),
            max_input_speed=float(np.max(np.abs(v))),
            root_angular_speed=float(np.linalg.norm(angular)),
            root_angular_velocity=angular.copy(),
            clipped_axes=sorted(clipped_axes),
            continuous_handoff=self.continuous_handoff,
            max_packet_acceleration=float(np.max(np.abs(np.diff(velocities, axis=0))) / self.dt)
            if len(velocities) > 1
            else 0.0,
        )
        return positions, velocities, roots, samples[0][0]
