"""Walk to the visible basket, brake and observe a quiet stance before reaching."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ApproachCommand:
    phase: str
    direction: np.ndarray
    speed: float = 0.0
    ready: bool = False
    reason: str = ""


def initial_robot_xy(meta, distance):
    surface = meta["surfaces"][meta["objects"]["blue_crate"]["surface"]]
    edge_x = surface["xy"][0] - surface["flat_half_extent"][0]
    return np.array([edge_x - distance, surface["xy"][1]])


class VisualApproach:
    def __init__(self, config):
        self.config = config
        self.phase = "walk"
        self.started = self.last_time = self.last_seen = self.quiet_since = None
        self.position = self.orientation = None
        self.relative_speed = self.angular_speed = 0.0

    def step(self, now, target, base_quaternion, *, grasp_pose_available):
        cfg = self.config
        if self.started is None:
            self.started = self.last_time = now
        dt = float(np.clip(now - self.last_time, 0.001, 0.1))
        self.last_time = now
        orientation = Rotation.from_quat(np.roll(base_quaternion, -1))
        if self.orientation is not None:
            rate = (orientation * self.orientation.inv()).magnitude() / dt
            self.angular_speed += min(1, dt / 0.3) * (rate - self.angular_speed)
        self.orientation = orientation
        reason = "approach_timeout" if now - self.started > cfg.walk_timeout else ""
        if target is None or now - target.timestamp > cfg.vision_timeout:
            self.quiet_since = None
            if now - (self.last_seen if self.last_seen is not None else self.started) > cfg.vision_timeout:
                reason = reason or "approach_vision_lost"
            return ApproachCommand(self.phase, np.zeros(3), reason=reason)
        self.last_seen = target.timestamp
        point = target.position
        if self.position is not None:
            speed = np.linalg.norm(point[:2] - self.position[:2]) / dt
            self.relative_speed += min(1, dt / 0.4) * (speed - self.relative_speed)
        self.position = point.copy()
        if self.phase == "walk" and point[0] <= cfg.stand_off + cfg.walk_brake_margin and abs(point[1]) < 0.04:
            self.phase = "settle"
        if self.phase == "settle":
            quiet = grasp_pose_available and self.relative_speed < 0.06 and self.angular_speed < 0.2
            self.quiet_since = (now if self.quiet_since is None else self.quiet_since) if quiet else None
            ready = self.quiet_since is not None and now - self.quiet_since >= cfg.settle_seconds
            return ApproachCommand(self.phase, np.zeros(3), ready=ready, reason=reason)
        error = np.array([max(0, point[0] - cfg.stand_off), point[1], 0.0])
        direction = error / max(np.linalg.norm(error), 1e-6)
        speed = float(np.clip(0.7 * np.linalg.norm(error), 0.10, cfg.walk_speed))
        return ApproachCommand(self.phase, direction, speed, reason=reason)
