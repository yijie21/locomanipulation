"""Minimal fakes shared across control-flow tests.

Moved verbatim out of the retired tests/test_grasp_transition.py (refactor
step 2, family 2, 2026-09-09) -- test_reach_config.py, test_execution_record.py
and test_ready_grasp_policy.py all import Clock/Robot from here.
"""

from types import SimpleNamespace

import numpy as np


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class Robot:
    hold_yaw = None

    def __init__(self, x=0.6, fallen=False):
        self.x, self.fallen = x, fallen
        self.frames = []
        self.cmd = SimpleNamespace(frame=lambda **kw: self.frames.append(kw))

    def obs(self):
        return {
            "fallen": self.fallen,
            "pelvis": {"vel": [0.0, 0.0, 0.0], "pos": [0.0, 0.0, 0.8], "yaw": 0.0},
            "palms": {"right": {"pos": [0.0, 0.0, 0.9], "quat": [1.0, 0.0, 0.0, 0.0]}},
        }

    def world_to_root_full(self, point, state):
        return np.array([self.x, 0.0, -0.07])

    world_to_root = world_to_root_full

    def pelvis_yaw(self, state):
        return 0.0

    def pelvis_xy(self, state):
        return np.zeros(2)

    def log(self, text):
        pass
