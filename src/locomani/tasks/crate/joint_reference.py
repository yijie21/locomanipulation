"""Continuous upper-body references with bounded speed and acceleration."""

import numpy as np


class JointReference:
    def __init__(self, joints, solver, *, speed, acceleration):
        self.joints = np.asarray(joints, dtype=float).copy()
        self.velocity = np.zeros_like(self.joints)
        self.indices, self.lo, self.hi = solver.indices, solver.lo, solver.hi
        self.speed, self.acceleration = speed, acceleration
        self.last_dt = None

    def step(self, target, dt):
        dt = max(float(dt), 0.0)
        if dt == 0:
            return self.joints.copy()
        indices = self.indices
        target = np.clip(np.asarray(target)[indices], self.lo[indices], self.hi[indices])
        error = target - self.joints[indices]
        braking_speed = np.maximum(0, np.sqrt(2 * self.acceleration * np.abs(error)) - self.acceleration * dt)
        wanted = np.sign(error) * np.minimum(self.speed, braking_speed)
        old = self.velocity[indices]
        interval = dt if self.last_dt is None else (dt + self.last_dt) / 2
        self.last_dt = dt
        velocity = old + np.clip(wanted - old, -self.acceleration * interval, self.acceleration * interval)
        candidate = self.joints[indices] + velocity * dt
        bounded = np.clip(candidate, self.lo[indices], self.hi[indices])
        self.velocity[indices] = (bounded - self.joints[indices]) / dt
        self.joints[indices] = bounded
        self.joints[12:14] = 0
        return self.joints.copy()
