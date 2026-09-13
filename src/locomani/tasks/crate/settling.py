"""Stance readiness from fresh, timestamped pose measurements."""

import numpy as np
from scipy.spatial.transform import Rotation


class SettlingWindow:
    def __init__(self, seconds):
        self.seconds = seconds
        self.previous_time = self.quiet_since = None
        self.position = self.rotation = None
        self.speed = self.angular_speed = float("inf")

    def step(self, now, position, quaternion, *, reference_ready):
        if self.previous_time is not None and now <= self.previous_time:
            return False
        position = np.asarray(position, dtype=float)
        rotation = Rotation.from_quat(np.roll(quaternion, -1))
        if self.previous_time is not None:
            dt = now - self.previous_time
            self.speed = float(np.linalg.norm(position - self.position) / dt)
            self.angular_speed = float((rotation * self.rotation.inv()).magnitude() / dt)
            # A gap cannot certify continuous stability.
            if dt > 0.1:
                self.quiet_since = None
        self.previous_time, self.position, self.rotation = now, position.copy(), rotation
        quiet = reference_ready and self.speed < 0.06 and self.angular_speed < 0.2
        self.quiet_since = (now if self.quiet_since is None else self.quiet_since) if quiet else None
        return self.quiet_since is not None and now - self.quiet_since >= self.seconds
