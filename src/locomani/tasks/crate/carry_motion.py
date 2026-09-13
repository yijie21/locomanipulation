"""One reference from walk-in through lift; IK and compensation supply targets."""

import numpy as np

from locomani.backends.sonic.primitives import DT, upper_body_from_mj
from locomani.tasks.crate.joint_reference import JointReference


class CarryMotion:
    def __init__(self, arm, solver, config, now):
        self.arm = arm
        self.reference = JointReference(
            arm.q,
            solver,
            speed=config.joint_speed,
            acceleration=config.joint_acceleration,
            joint_limits=(arm.ik.lo, arm.ik.hi),
        )
        self.last_time = now - DT

    @property
    def q(self):
        return upper_body_from_mj(self.reference.joints)

    @property
    def quiet(self):
        return bool(np.max(np.abs(self.reference.velocity)) < 0.02)

    def step(self, target, now):
        dt = max(now - self.last_time, 0.0)
        self.last_time = max(now, self.last_time)
        self.arm.q = self.reference.step(target, dt)
        return self.arm.q

    def push(self, upper_target, now):
        """Publish during navigation; the navigation loop owns frame pacing."""
        target = self.reference.joints.copy()
        target[12:15] = upper_target[:3]
        target[15:22], target[22:29] = upper_target[3::2], upper_target[4::2]
        self.step(target, now)
        self.arm.p.cmd.set_upper_body(self.q)
        self.arm.p.cmd.set_upper_body_vel(upper_body_from_mj(self.reference.velocity))
        return float(np.max(np.abs(upper_target - self.q)))
