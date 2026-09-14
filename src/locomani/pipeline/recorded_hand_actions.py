"""Record the hand command consumed alongside the recorded SONIC token.

The upstream SMPL subscriber may not decode joint-only POSE packets. Falling
back to its old planner message then labels a real closure as an open hand.
The C++ state packet already carries both applied 7-D hand command buffers.
"""

import numpy as np


def consumed_hands(proprio):
    result = {}
    for side in ("left", "right"):
        key = f"last_{side}_hand_action"
        if key not in proprio:
            raise ValueError(f"missing consumed hand command: {key}")
        value = np.asarray(proprio[key], dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain seven finite command values")
        result[f"teleop.{side}_hand_joints"] = value.copy()
    return result


def install(collector_class):
    original = collector_class._add_sonic_pose_features

    def observed(self, frame_data):
        latency = original(self, frame_data)
        frame_data.update(consumed_hands(self.latest_proprio_msg))
        return latency

    collector_class._add_sonic_pose_features = observed
