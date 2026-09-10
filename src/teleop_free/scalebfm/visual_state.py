"""VLA inputs: joint/FK/IMU state, robot velocities and the previous action.

No absolute world position, task phase, object pose or success/contact flag is
returned. Images are paired with their actual 0.4-second-old observation.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from visual_action import heading

STATE_DIMS = {"left_leg": 6, "right_leg": 6, "waist": 3, "left_arm": 7,
              "right_arm": 7, "left_hand": 7, "right_hand": 7,
              "projected_gravity": 3, "right_wrist_pos": 3,
              "right_wrist_abs_quat": 4, "robot_velocity": 9, "previous_action": 29}


def state_features(q29, gripper4, root_position, root_quaternion, wrist_position,
                   wrist_quaternion, root_velocity, root_angular_velocity,
                   wrist_velocity, previous_action):
    """Velocities are world-frame measurements transformed into the live yaw frame.

    The previous action is the last submitted 25 Hz row, including when a new
    inference request arrives. It must never be filled with the current label.
    """
    q, g = np.asarray(q29), np.asarray(gripper4)
    root = Rotation.from_quat(root_quaternion, scalar_first=True)
    inv_yaw = heading(root_quaternion).inv()
    lh, rh = np.zeros(7), np.zeros(7)
    lh[4:6], rh[4:6] = g[:2], g[2:]
    out = dict(left_leg=q[:6], right_leg=q[6:12], waist=q[12:15],
               left_arm=q[15:22], right_arm=q[22:29], left_hand=lh, right_hand=rh,
               projected_gravity=root.inv().apply([0., 0., -1.]),
               right_wrist_pos=root.inv().apply(np.asarray(wrist_position) - root_position),
               right_wrist_abs_quat=(root.inv() * Rotation.from_quat(wrist_quaternion, scalar_first=True)).as_quat(scalar_first=True),
               robot_velocity=np.concatenate([inv_yaw.apply(v) for v in (root_velocity, root_angular_velocity, wrist_velocity)]),
               previous_action=previous_action)
    out = {key: np.asarray(value, dtype=np.float32).copy() for key, value in out.items()}
    for key, width in STATE_DIMS.items():
        if out[key].shape != (width,) or not np.isfinite(out[key]).all():
            raise ValueError(f"Invalid proprioception field {key}")
    return out


class ImageHistory:
    def __init__(self, history_s=.4):
        if not np.isfinite(history_s) or history_s <= 0:
            raise ValueError("History interval must be positive")
        self.history_s = history_s
        self.frames = deque()

    def push(self, t, images):
        if not np.isfinite(t) or (self.frames and t <= self.frames[-1][0]):
            raise ValueError("Image timestamps must increase")
        self.frames.append((t, {key: np.asarray(frame).copy() for key, frame in images.items()}))
        target = t - self.history_s
        while len(self.frames) > 1 and self.frames[1][0] <= target + 1e-8:
            self.frames.popleft()
        old, current = self.frames[0][1], self.frames[-1][1]
        # Leading history pads with the first actual frame, matching dataset
        # clamped delta indices; never invent an unseen preceding observation.
        return {key: np.stack([old[key], current[key]]) for key in current}
