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
    facing_offset: float = 0.0  # radians to add to the held heading (turn phase only; + = left)


def approach_heading(meta):
    """World yaw of the walk-in, radians: the scene's `robot_start` heading.

    0 in the synthetic table scenes (the robot walks +x), pi/2 in the logistics
    room (it walks +y toward the platform). Every placement below is expressed
    along this axis so one expert runs in both.
    """
    return float(np.radians((meta.get("robot_start") or {}).get("yaw_deg", 0.0)))


def near_edge(meta):
    """Centre of the target surface's edge facing the walk-in, and the unit walk-in axis."""
    surface = meta["surfaces"][meta["objects"][meta.get("target_object") or "blue_crate"]["surface"]]
    heading = approach_heading(meta)
    axis = np.array([np.cos(heading), np.sin(heading)])
    reach = abs(axis[0]) * surface["flat_half_extent"][0] + abs(axis[1]) * surface["flat_half_extent"][1]
    return np.asarray(surface["xy"][:2], dtype=float) - axis * reach, axis


def initial_robot_xy(meta, distance):
    """Start point `distance` metres before the near table edge, on the walk-in axis."""
    edge, axis = near_edge(meta)
    return edge - axis * distance


def edge_distance(meta, xy):
    """Signed distance from `xy` to the near table edge along the walk-in axis (positive = short of it)."""
    edge, axis = near_edge(meta)
    return float(axis @ (edge - np.asarray(xy[:2], dtype=float)))


def facing_correction(mode, estimate, grasp_points_local):
    """Yaw to add to the current heading (radians, + = left) for visual.face_after_brake, from a pelvis-frame estimate.

    centre: aim the body at the target's centre. box_yaw: aim it along the normal of the grasp line (the two
    grasp points, i.e. the gripped walls), whichever sense is nearest to straight ahead, so the box is square
    to the body. none: 0.
    """
    if mode == "none" or estimate is None:
        return 0.0
    if mode == "centre":
        return float(np.arctan2(estimate.position[1], estimate.position[0]))
    across = np.asarray(estimate.transform(grasp_points_local), dtype=float)
    across = across[1, :2] - across[0, :2]
    normal = float(np.arctan2(across[1], across[0])) - np.pi / 2
    normal = (normal + np.pi) % (2 * np.pi) - np.pi
    if abs(normal) > np.pi / 2:
        normal = (normal + 2 * np.pi) % (2 * np.pi) - np.pi
    return float(normal)


def scene_object_yaw(meta):
    """World yaw of the target object as the scene places it (from `init_quat`, w-x-y-z)."""
    obj = meta["objects"][meta.get("target_object") or "blue_crate"]
    w, x, y, z = obj.get("init_quat") or (1.0, 0.0, 0.0, 0.0)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class VisualApproach:
    def __init__(self, config):
        self.config = config
        self.phase = "walk"
        self.started = self.last_time = self.last_seen = self.quiet_since = None
        self.position = self.orientation = None
        self.relative_speed = self.angular_speed = 0.0
        self.relative_velocity = np.zeros(2)
        self.stop_distance = config.stand_off + config.walk_brake_margin
        # Every quiet stance the walk delivered, whether or not it was accepted.
        # The gait scatters the achieved depth by centimetres around the planned
        # stop, and a far stance is what the reach later cannot support.
        self.stand_attempts = []
        self.depth_retries = 0
        self.quiet_points = []
        self.back_until = None  # guard re-walk: the target depth (m) at which backing off ends

    def _tighten_stop(self):
        """Halve the remaining braking allowance; never command a stop past stand_off."""
        allowance = self.stop_distance - self.config.stand_off
        if allowance <= 0:
            return False
        self.stop_distance = self.config.stand_off + (allowance / 2 if allowance > 0.01 else 0)
        return True

    def _walk_again(self):
        self.phase, self.quiet_since, self.quiet_points = "walk", None, []

    def retry_closer(self):
        """Use remaining braking allowance after unsupported reach admission; never pass stand_off."""
        if self.config.approach_stop == "guard":
            return False  # the guard stance is the nearest one that keeps the body behind the line
        if not self._tighten_stop():
            return False
        self._walk_again()
        return True

    def _stance_converged(self):
        """Depth-axis convergence: a quiet stance farther than the tolerance re-walks while retries remain."""
        cfg = self.config
        if cfg.approach_stop == "guard":
            point = np.mean(self.quiet_points, axis=0)
            attempt = dict(depth_m=float(point[0]), lateral_m=float(point[1]), accepted=True)
            # The guard stance cannot be walked closer (the line ends the walk), but each walk-in lands at an
            # independent depth: back off and walk in again (TODO 2.4.10.23.2).
            if (
                cfg.guard_rewalk_depth > 0
                and round(float(point[0]), 3) > cfg.guard_rewalk_depth
                and self.depth_retries < cfg.stand_depth_retries
            ):
                self.depth_retries += 1
                attempt["accepted"] = False
                self.stand_attempts.append(attempt)
                self.phase, self.quiet_since, self.quiet_points = "back", None, []
                self.back_until = float(point[0]) + cfg.guard_rewalk_back
                return False
            self.stand_attempts.append(attempt)
            return True
        # Judge the stance on the quiet interval's mean, not one pose sample:
        # millimetre pose jitter must not decide a re-walk any more than it
        # decides quietness.
        point = np.mean(self.quiet_points, axis=0)
        depth = float(point[0] - cfg.stand_off)
        attempt = dict(depth_m=float(point[0]), lateral_m=float(point[1]), depth_error_m=depth)
        # Millimetre resolution: the visible pose is not better than that, and
        # float rounding must not turn a stance on the tolerance into a re-walk.
        if round(depth, 3) > cfg.stand_depth_tolerance and self.depth_retries < cfg.stand_depth_retries:
            self.depth_retries += 1
            self._tighten_stop()
            attempt["accepted"] = False
            self.stand_attempts.append(attempt)
            self._walk_again()
            return False
        attempt["accepted"] = True
        self.stand_attempts.append(attempt)
        return True

    def step(
        self,
        wall_time,
        target,
        base_quaternion,
        *,
        grasp_pose_available,
        motion_time=None,
        base_velocity=None,
        guard_brake=False,
    ):
        """Use the motion clock for movement/settling, wall time for camera freshness.

        `guard_brake` (approach_stop=guard): the caller's foot-landing decision that one more stride would cross
        the line; it replaces the stop_distance test.
        """
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
        if self.phase == "back":
            if point[0] < self.back_until:
                return self._back_command(reason)
            self._walk_again()
            self.started = now  # each guard walk-in gets the whole walk_timeout
        aligned = abs(point[1]) < cfg.stand_lateral_tolerance
        depth_only = cfg.stand_lateral_gate == "turn" or cfg.walk_steer == "depth"
        if cfg.approach_stop == "guard":
            if self.phase == "walk" and guard_brake:
                self.phase = "settle"
        elif self.phase == "walk" and point[0] <= self.stop_distance and (aligned or depth_only):
            self.phase = "settle"
        if self.phase == "settle" and not aligned and cfg.stand_lateral_gate == "turn":
            # Depth is reached; the lateral residual is a heading error (a
            # walk holds its entry heading and drifts 3-16 deg in 5 s, ledger
            # crate.md 2026-09-22). Turn on the spot to face the target instead
            # of walking on: walking on for lateral is what overshoots depth.
            self.quiet_since, self.quiet_points = None, []
            return ApproachCommand(
                "turn", np.zeros(3), reason=reason, facing_offset=float(np.arctan2(point[1], max(point[0], 1e-3)))
            )
        if self.phase == "settle":
            # Assisted execution has measured base velocity. Use that current
            # physical signal instead of waiting for the visual derivative's
            # 0.4 s filter to forget a walk that has already ended.
            speed = self.relative_speed if base_velocity is None else float(np.linalg.norm(base_velocity))
            angular = self.angular_speed if base_velocity is None else rate
            quiet = grasp_pose_available and speed < 0.06 and angular < 0.2
            self.quiet_since = (now if self.quiet_since is None else self.quiet_since) if quiet else None
            self.quiet_points = self.quiet_points + [point[:2].copy()] if quiet else []
            ready = self.quiet_since is not None and now - self.quiet_since >= cfg.settle_seconds
            if ready and not self._stance_converged():
                if self.phase == "back":
                    self.started = now
                    return self._back_command(reason)
                return self._walk_command(point, reason)
            return ApproachCommand(self.phase, np.zeros(3), ready=ready, reason=reason)
        return self._walk_command(point, reason)

    def _back_command(self, reason):
        """Guard re-walk: straight back along the body heading at the guard's constant speed (`phase` back)."""
        return ApproachCommand("back", np.array([-1.0, 0.0, 0.0]), self.config.walk_speed, reason=reason)

    def _walk_command(self, point, reason):
        cfg = self.config
        error = np.array([max(0, point[0] - cfg.stand_off), point[1], 0.0])
        if cfg.walk_steer == "depth" or cfg.approach_stop == "guard":
            # Straight along the body heading: the braking carry measured in the walk scene is a
            # straight-line carry, and a pixel-noisy lateral term only swings the step direction.
            error[1] = 0.0
        if cfg.approach_stop == "guard":
            error = np.array([1.0, 0.0, 0.0])  # the line, not stand_off, ends the walk
        direction = error / max(np.linalg.norm(error), 1e-6)
        # The planner's speed field is not m/s and its map belongs to the deployed
        # stack (ledger crate.md 2026-09-22): 0.10 delivered 0.011 m/s over 940
        # samples of two crate batches, while on 2026-09-22 0.20 delivered 0.05 m/s
        # and 0.35 0.25 m/s with a 10 cm carry. A floor below the map's foot is a
        # walk that never starts, which is how a depth re-walk stood still for 20 s;
        # re-measure with backends/sonic/diag_shuffle.py before moving the floor.
        speed = (
            cfg.walk_speed
            if cfg.approach_stop == "guard"  # constant: the guard's calibration is a constant-speed gait
            else float(np.clip(0.7 * np.linalg.norm(error), cfg.walk_speed_min, cfg.walk_speed))
        )
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
