"""Shared 25 Hz action / 50 Hz ScaleBFM executor, using robot state only.

The teacher and deployed VLA submit the same float32, pelvis-yaw-frame rows.
Each row is held for two physics-control ticks and decoded on the live pelvis.
No object estimate, task phase, scene access or hidden servo is accepted here.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.scalebfm.action import ACTION_DIM, decode_action
from locomani.backends.scalebfm.reference import bounded


class ActionExecutor:
    def __init__(self, hand_rel, action_hz=25, control_hz=50):
        if control_hz <= 0 or action_hz <= 0 or control_hz % action_hz:
            raise ValueError("Action rate must divide the control rate")
        self.hand_rel = hand_rel
        self.dt = 1.0 / control_hz
        self.stride = control_hz // action_hz
        self.tick = 0
        self.row = None
        self.previous_wrist = None

    @property
    def needs_action(self):
        return self.tick % self.stride == 0

    def step(self, root_position, root_quaternion, action=None):
        if self.needs_action != (action is not None):
            raise ValueError("Submit one action exactly at each action-rate boundary")
        row = self.row if action is None else np.asarray(action, dtype=np.float32)
        if row is None or row.shape != (ACTION_DIM,):
            raise ValueError("A 29-D initial action is required")
        targets, grip = decode_action(row, root_position, root_quaternion, self.hand_rel, dt=self.dt)
        positions, quaternions = targets["right_wrist_yaw_link"]
        if self.previous_wrist is not None:
            previous_p, previous_q = self.previous_wrist
            p0 = previous_p + bounded(positions[0] - previous_p, 0.02)
            positions = positions + (p0 - positions[0])
            r0 = Rotation.from_quat(quaternions[0], scalar_first=True)
            previous_r = Rotation.from_quat(previous_q, scalar_first=True)
            delta = (r0 * previous_r.inv()).as_rotvec()
            limited = Rotation.from_rotvec(bounded(delta, np.radians(5))) * previous_r
            correction = limited * r0.inv()
            quaternions = (correction * Rotation.from_quat(quaternions, scalar_first=True)).as_quat(scalar_first=True)
            targets["right_wrist_yaw_link"] = positions, quaternions
        self.previous_wrist = positions[0].copy(), quaternions[0].copy()
        self.row = row.copy()
        self.tick += 1
        return targets, grip
