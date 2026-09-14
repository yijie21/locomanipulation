"""Slow visual preparation during gait, with continuously bounded arm commands.

The admitted world point is converted using proprioception, filtered BEFORE
planning is decimated, and projected into a conservative arm-only workspace.
Legs remain owned by SONIC; the waist returns smoothly to its neutral walking reference. This is assisted geometry,
not an RGB pose estimator or a high-frequency image-centering servo.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.primitives import Primitives, upper_body_from_mj
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ready_q


class WalkingReach:
    def __init__(self, ik, config, upper_reference):
        self.ik, self.config = ik, config
        self.q = np.asarray(upper_reference, dtype=float).copy()
        self.v = np.zeros(17)
        self.wanted = self.q.copy()
        self.seed = ready_q()
        self.relative = self.root = self.rotation = None
        self.base_velocity = np.zeros(3)
        self.last = self.planned = None
        self.diagnostics = {}

    def step(self, state, target):
        stamp = float(state["t"])
        dt = 0 if self.last is None else stamp - self.last
        if dt < 0 or dt > 0.5:
            raise ValueError("walking reach physical state clock gap")
        if self.last == stamp:
            return self.q.copy(), self.v.copy()
        self.last = stamp
        updated = False
        if target is not None:
            root = Primitives._rot(state["pelvis"]["quat"])
            relative = root.T @ (np.asarray(target) - state["pelvis"]["pos"])
            if self.relative is None:
                self.relative, self.root = relative.copy(), root.copy()
            alpha = 1 - np.exp(-dt / self.config.walk_filter_seconds)
            velocity = np.asarray(state["pelvis"].get("vel", [0, 0, 0]), dtype=float).copy()
            velocity[2] = 0  # vertical gait bounce is not an approach command
            self.base_velocity += (1 - np.exp(-dt / (self.config.walk_filter_seconds / 3))) * (
                velocity - self.base_velocity
            )
            # Predict the slow approach before correcting the filtered signal.
            # A plain local-position low-pass lags forward travel by v*tau and
            # can put the hand beyond the bottle while the base is braking.
            self.relative -= self.root.T @ self.base_velocity * dt
            self.relative += alpha * (relative - self.relative)
            self.root = (
                Rotation.from_rotvec(alpha * Rotation.from_matrix(root @ self.root.T).as_rotvec()).as_matrix()
                @ self.root
            )
            if self.planned is None or stamp - self.planned >= 1 / self.config.walk_plan_hz:
                self.planned, updated = stamp, True
                self.rotation = Rotation.from_euler("z", np.arctan2(self.root[1, 0], self.root[0, 0])).as_matrix()
                # Leave clearance during gait. The same bottle/entry axis is
                # used for the final short reach after braking; no raised waypoint.
                advance_speed = max(0, float(self.base_velocity @ self.rotation[:, 0]))
                gap = 0.12 + advance_speed / self.config.walk_plan_hz
                goal = self.relative - self.root.T @ self.rotation[:, 0] * gap
                goal[0] = np.clip(goal[0], 0.16, 0.26)
                goal[1] = np.clip(goal[1], -0.26, -0.14)
                goal[2] = max(goal[2] + 0.06, 0.05)
                seed = self.seed.copy()
                solved, error, _ = self.ik.solve(
                    "right",
                    goal,
                    self.root.T @ self.rotation,
                    q_mj29=seed,
                    point_offset=OFFSET,
                    use_waist=False,
                    iters=30,
                    rot_weight=0.2,
                    posture=seed,
                    posture_weight=0.005,
                )
                self.seed = solved
                self.wanted = upper_body_from_mj(solved)
                self.diagnostics.update(goal=goal.copy(), ik_error=float(error))
        else:
            # An expired observation must not continue pursuing hidden truth.
            # Decelerate smoothly; a later admitted image can resume preparation.
            self.wanted = self.q.copy()
        speed = min(0.45, self.config.joint_speed)
        acceleration = min(1.0, self.config.joint_acceleration)
        error = self.wanted - self.q
        braking_speed = np.sqrt(2 * acceleration * np.abs(error)) * 0.8
        wanted_v = np.sign(error) * np.minimum(np.minimum(4 * np.abs(error), braking_speed), speed)
        self.v += np.clip(wanted_v - self.v, -acceleration * dt, acceleration * dt)
        self.q += self.v * dt
        self.diagnostics.update(
            t=stamp,
            plan_updated=updated,
            relative=None if self.relative is None else self.relative.copy(),
            reference=self.q.copy(),
            velocity=self.v.copy(),
            target_available=target is not None,
        )
        return self.q.copy(), self.v.copy()
