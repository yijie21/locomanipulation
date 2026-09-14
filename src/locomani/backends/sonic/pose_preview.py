"""Physical-time POSE samples with the future window required by the SONIC encoder.

v1.1 observes 10 poses at a stride of 5 ticks and reserves 46 frames before
advancing playback. Past-only packets consequently delay feedback by 46 sends.
This sender supplies 50 future ticks: a bounded tangent continuation that brakes
to rest, replaced on the next observation. It never extrapolates indefinitely.
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PosePreview:
    dt = 0.02
    future_frames = 50
    history_frames = 5

    def __init__(self, *, acceleration, joint_limits):
        self.acceleration = float(acceleration)
        self.lo, self.hi = (np.asarray(v).copy() for v in joint_limits)
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

    def sample(self, stamp, joints, velocity, root):
        q, v = np.asarray(joints).copy(), np.asarray(velocity).copy()
        root = Rotation.from_quat(np.asarray(root)[[1, 2, 3, 0]])
        if self.origin is None:
            self.origin = float(stamp)
        tick = round((stamp - self.origin) / self.dt)
        angular = np.zeros(3)
        if self.previous is None:
            self.history.append((tick, q, v, root))
        else:
            old_tick, old_q, _, old_root = self.previous
            if tick < old_tick:
                raise ValueError("POSE reference time moved backwards")
            if tick > old_tick:
                duration = (tick - old_tick) * self.dt
                angular = (root * old_root.inv()).as_rotvec() / duration
                rotations = Slerp([0, 1], Rotation.concatenate([old_root, root]))
                for index in range(max(old_tick + 1, tick - self.history_frames), tick + 1):
                    u = (index - old_tick) / (tick - old_tick)
                    self.history.append((index, old_q + u * (q - old_q), v, rotations(u)))
            else:
                # Priming can observe the same simulation tick twice.
                self.history[-1] = (tick, q, v, root)
        self.previous = tick, q.copy(), v.copy(), root
        samples = list(self.history)
        # All axes share the same braking progress, preserving the tangent of
        # the supported body trajectory instead of stopping each limb separately.
        braking_time = max(np.max(np.abs(v)), np.linalg.norm(angular)) / self.acceleration
        for offset in range(1, self.future_frames + 1):
            scale = max(0.0, 1 - offset * self.dt / braking_time) if braking_time else 0.0
            following = np.clip(q + v * scale * self.dt, np.minimum(self.lo, q), np.maximum(self.hi, q))
            step_velocity = (following - q) / self.dt
            q = following
            root = Rotation.from_rotvec(angular * scale * self.dt) * root
            samples.append((tick + offset, q.copy(), step_velocity, root))
        return (
            np.array([s[1] for s in samples]),
            np.array([s[2] for s in samples]),
            np.array([s[3].as_quat()[[3, 0, 1, 2]] for s in samples]),
            samples[0][0],
        )
