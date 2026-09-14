"""Walk to the visible basket, brake and observe a quiet stance before reaching."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.tasks.crate.grasp_region import GRASP_POINTS
from locomani.tasks.crate.joint_reference import JointReference
from locomani.tasks.crate.visual_low_reach import blend


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
        self.relative_velocity = np.zeros(2)
        self.stop_distance = config.stand_off + config.walk_brake_margin

    def retry_closer(self):
        """Use remaining braking allowance after unsupported reach admission; never pass stand_off."""
        allowance = self.stop_distance - self.config.stand_off
        if allowance <= 0:
            return False
        self.stop_distance = self.config.stand_off + (allowance / 2 if allowance > 0.01 else 0)
        self.phase, self.quiet_since = "walk", None
        return True

    def step(self, wall_time, target, base_quaternion, *, grasp_pose_available, motion_time=None, base_velocity=None):
        """Use the motion clock for movement/settling, wall time for camera freshness."""
        cfg = self.config
        now = wall_time if motion_time is None else float(motion_time)
        if self.started is None:
            self.started = self.last_time = now
            self.started_wall = wall_time
        dt = float(np.clip(now - self.last_time, 0.001, 0.1))
        self.last_time = now
        orientation = Rotation.from_quat(np.roll(base_quaternion, -1))
        rate = 0.0
        if self.orientation is not None:
            rate = (orientation * self.orientation.inv()).magnitude() / dt
            self.angular_speed += min(1, dt / 0.3) * (rate - self.angular_speed)
        self.orientation = orientation
        reason = "approach_timeout" if now - self.started > cfg.walk_timeout else ""
        if target is None or wall_time - target.timestamp > cfg.vision_timeout:
            self.quiet_since = None
            if wall_time - (self.last_seen if self.last_seen is not None else self.started_wall) > cfg.vision_timeout:
                reason = reason or "approach_vision_lost"
            return ApproachCommand(self.phase, np.zeros(3), reason=reason)
        self.last_seen = target.timestamp
        point = target.position
        if self.position is not None:
            # Filter signed displacement before taking its magnitude. Filtering
            # magnitudes interprets zero-mean pixel-fit jitter as persistent motion.
            velocity = (point[:2] - self.position[:2]) / dt
            self.relative_velocity += min(1, dt / 0.4) * (velocity - self.relative_velocity)
            self.relative_speed = float(np.linalg.norm(self.relative_velocity))
        self.position = point.copy()
        if self.phase == "walk" and point[0] <= self.stop_distance and abs(point[1]) < cfg.stand_lateral_tolerance:
            self.phase = "settle"
        if self.phase == "settle":
            # Assisted execution has measured base velocity. Use that current
            # physical signal instead of waiting for the visual derivative's
            # 0.4 s filter to forget a walk that has already ended.
            speed = self.relative_speed if base_velocity is None else float(np.linalg.norm(base_velocity))
            angular = self.angular_speed if base_velocity is None else rate
            quiet = grasp_pose_available and speed < 0.06 and angular < 0.2
            self.quiet_since = (now if self.quiet_since is None else self.quiet_since) if quiet else None
            ready = self.quiet_since is not None and now - self.quiet_since >= cfg.settle_seconds
            return ApproachCommand(self.phase, np.zeros(3), ready=ready, reason=reason)
        error = np.array([max(0, point[0] - cfg.stand_off), point[1], 0.0])
        direction = error / max(np.linalg.norm(error), 1e-6)
        speed = float(np.clip(0.7 * np.linalg.norm(error), 0.10, cfg.walk_speed))
        return ApproachCommand(self.phase, direction, speed, reason=reason)


class ApproachPreparation:
    """Prepare the visible rim approach while the walking planner still owns the feet."""

    def __init__(self, solver, config, joints):
        self.solver, self.config = solver, config
        self.initial = np.asarray(joints).copy()
        self.points, self.attitudes = solver.tool_poses(self.initial)
        self.reference = JointReference(
            joints, solver, speed=config.joint_speed, acceleration=config.joint_acceleration
        )
        self.last_time = None

    def step(self, now, estimate):
        dt = 0 if self.last_time is None else max(0, now - self.last_time)
        self.last_time = now
        if estimate is None:
            target = self.reference.joints  # decelerate toward the current pose
        else:
            cfg = self.config
            grips = estimate.transform(GRASP_POINTS)
            lead = cfg.walk_speed * 3
            progress = np.clip((cfg.stand_off + lead - estimate.position[0]) / lead, 0, 1)
            u, _ = blend(progress, 1)
            prepared = self.points.copy()
            # Stay above the low rim and short of the table while the feet move.
            prepared[:, 0] = np.minimum(prepared[:, 0], cfg.stand_off - cfg.walk_brake_margin)
            prepared[:, 2] -= min(0.12, max(0, self.points[:, 2].mean() - grips[:, 2].mean()) * 0.3)
            points = self.points + u * (prepared - self.points)
            attitude = Rotation.from_euler("y", cfg.palm_pitch_deg, degrees=True).as_matrix()
            rotations = np.array(
                [Slerp([0, 1], Rotation.from_matrix([r, attitude]))(u).as_matrix() for r in self.attitudes]
            )
            lo, hi = self.solver.lo.copy(), self.solver.hi.copy()
            # Walking preparation has a smaller lean envelope; all axes remain
            # available in the supported reach after the feet stop.
            lo[12:15], hi[12:15] = np.maximum(lo[12:15], [-0.1, -0.1, 0]), np.minimum(hi[12:15], [0.1, 0.1, 0.12])
            target = self.solver.solve(
                self.reference.joints, points, rotations, iterations=8, joint_limits=(lo, hi)
            ).joints
        return self.reference.step(target, dt), self.reference.velocity.copy()
