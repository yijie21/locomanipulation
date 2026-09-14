"""Limits for continuous supported bottle reach; geometry stays in the task."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WholeBodyConfig:
    waist_pitch_max: float = 0.52
    pelvis_pitch_max: float = 0.70
    pelvis_height_min: float = 0.65
    joint_speed: float = 0.70
    joint_acceleration: float = 2.0
    hand_speed: float = 0.16
    close_seconds: float = 1.2
    tracking_gain: float = 1.0
    vision_timeout: float = 1.0

    @property
    def waist_dofs(self):
        return "all"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"whole_body.{name} must be positive and finite")
        if not (self.waist_pitch_max <= 0.52 and self.pelvis_pitch_max <= 0.7):
            raise ValueError("whole-body posture exceeds supported limits")
        if not (0.55 <= self.pelvis_height_min <= 0.8 and self.joint_speed <= 1.0 and self.hand_speed <= 0.25):
            raise ValueError("whole-body height or speed outside supported range")
        if not (self.tracking_gain <= 1 and self.vision_timeout <= 1 and self.close_seconds >= 0.5):
            raise ValueError("whole-body feedback, freshness or closure outside supported range")
