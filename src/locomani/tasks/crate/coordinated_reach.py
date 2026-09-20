"""Visible-target reach with simulation-assisted, continuously coordinated body feedback."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.tasks.crate.grasp_region import GRASP_POINTS, RimRegion
from locomani.tasks.crate.visual_low_reach import LowReachCommand, blend, quaternion, rotation
from locomani.tasks.crate.whole_body_ik import WholeBodyIK


@dataclass(frozen=True)
class CoordinatedTarget:
    """Object-local grasp geometry consumed by :class:`CoordinatedReach`.

    ``capture_rotation_local`` has capture across, tangent and up axes as
    columns. The default exactly describes the historical blue-crate target.
    """

    object_key: str = "blue_crate"
    grasp_points_local: tuple[tuple[float, float, float], ...] = GRASP_POINTS
    capture_rotation_local: np.ndarray = field(default_factory=lambda: np.eye(3))
    region: RimRegion = field(default_factory=RimRegion)
    information_source: str = "simulation_assisted"
    preserve_point_order: bool = False
    reach_only: bool = False

    def __post_init__(self):
        points = np.asarray(self.grasp_points_local, dtype=float)
        frame = np.asarray(self.capture_rotation_local, dtype=float)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("grasp_points_local must contain two finite 3D points")
        if frame.shape != (3, 3) or not np.isfinite(frame).all():
            raise ValueError("capture_rotation_local must be a finite 3x3 rotation")
        if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-6) or np.linalg.det(frame) < 0.999999:
            raise ValueError("capture_rotation_local must be right-handed and orthonormal")
        if not self.object_key:
            raise ValueError("object_key must be non-empty")
        object.__setattr__(self, "grasp_points_local", tuple(map(tuple, points)))
        object.__setattr__(self, "capture_rotation_local", frame.copy())


class CoordinatedReach:
    """One supported body reference through descent, capture and paired lifting.

    Simulation supplies measured world poses; reference motion remains bounded,
    feet stay anchored, and unseen targets cannot advance the trajectory.
    """

    def __init__(
        self,
        solver,
        config,
        state,
        *,
        reference_joints,
        reference_quaternion,
        target: CoordinatedTarget | None = None,
    ):
        self.solver, self.config = solver, config
        self.target = CoordinatedTarget() if target is None else target
        self.region = self.target.region
        self.body = WholeBodyIK(
            solver.ik, state, config.carry, tool_offset=solver.offset, joint_limits=(solver.lo, solver.hi)
        )
        self.q = np.asarray(reference_joints).copy()
        self.root = np.asarray(reference_quaternion).copy()
        pitch = Rotation.from_matrix(rotation(self.root)).as_euler("xyz")[1]
        self.body.heading = rotation(self.root) @ Rotation.from_euler("y", -pitch).as_matrix()
        # The consumed planner reference is continuous; infer its translation from the same feet.
        self.x = np.r_[state["pelvis"]["pos"], pitch, self.q]
        self.body.put(self.x)
        self.x[:3] += np.mean(self.body.feet_pos - solver.ik.data.xpos[self.body.feet], axis=0)
        self.travel_lo, self.travel_hi = np.minimum(self.body.lo, self.x), np.maximum(self.body.hi, self.x)
        self.last_dt = None
        self.v = np.zeros(33)
        self.velocity = np.zeros(29)
        self.phase, self.closure = "observe", 0.0
        self.started = self.last_time = self.last_seen = None
        self.was_visible = False
        self.elapsed = 0.0
        self.ready_since = None
        self.feedback = np.zeros((2, 3))
        self.best_error, self.progress_time = np.inf, None
        self.diagnostics = {}
        self.torso_target = None
        self.previous_load = None
        self.hold_window = deque()

    def result(self, reason="", finished=False):
        return LowReachCommand(
            self.q.copy(), self.velocity.copy(), self.root.copy(), self.closure, self.phase, reason, finished
        )

    def enter(self, phase):
        self.phase, self.elapsed, self.ready_since = phase, 0.0, None

    def advance(self, targets, attitudes, measured, dt):
        """Rate/acceleration limits belong inside the supported body solve, including hips."""
        if dt <= 0:
            return
        cfg = self.config.visual
        predicted = self.body.tools(self.x)
        observed_error = np.clip(predicted - measured, -cfg.tracking_cap, cfg.tracking_cap)
        self.feedback += (1 - np.exp(-dt / 0.4)) * (observed_error - self.feedback)
        speed = np.r_[[self.config.carry.height_speed] * 3, cfg.palm_rotation_speed, [cfg.joint_speed] * 29]
        acceleration = np.r_[[0.2] * 3, cfg.joint_acceleration, [cfg.joint_acceleration] * 29]
        interval = dt if self.last_dt is None else (dt + self.last_dt) / 2
        self.last_dt = dt
        vlo = np.maximum(-speed, self.v - acceleration * interval)
        vhi = np.minimum(speed, self.v + acceleration * interval)
        braking_acceleration = acceleration / 2  # reserve braking margin for the next irregular interval
        vlo = np.maximum(
            vlo,
            -np.maximum(
                0,
                np.sqrt(2 * braking_acceleration * np.maximum(self.x - self.travel_lo, 0))
                - braking_acceleration * max(dt, 0.1),
            ),
        )
        vhi = np.minimum(
            vhi,
            np.maximum(
                0,
                np.sqrt(2 * braking_acceleration * np.maximum(self.travel_hi - self.x, 0))
                - braking_acceleration * max(dt, 0.1),
            ),
        )
        lo = np.maximum(self.travel_lo, self.x + vlo * dt)
        hi = np.minimum(self.travel_hi, self.x + vhi * dt)
        # Brake before physical limits. Tiny open bounds also cover an initially saturated axis.
        lo = np.minimum(lo, self.travel_hi - 1e-10)
        hi = np.maximum(hi, self.travel_lo + 1e-10)
        lo = np.minimum(lo, hi - 1e-10)
        pose = self.body.solve(
            targets + cfg.tracking_gain * self.feedback,
            attitudes,
            seed=self.x + np.clip(self.v, vlo, vhi) * dt,
            bounds=(lo, hi),
            posture=self.posture,
            torso_rotation=self.torso_target,
            iterations=6,
        )
        next_x = self.body.seed.copy()
        self.v = (next_x - self.x) / dt
        self.x, self.q, self.root = next_x, pose.joints, quaternion(pose.rotation)
        self.velocity = self.v[4:].copy()
        self.diagnostics.update(
            reference_position=pose.position,
            foot_error=pose.foot_errors,
            com_shift=pose.com_shift,
            tracking_correction=self.feedback,
            reference_hand_error=pose.hand_errors,
        )

    def brake(self, dt):
        """A lost visual target stops progress and brakes the last physical reference."""
        if dt <= 0:
            return
        acceleration = np.r_[
            [0.2] * 3, self.config.visual.joint_acceleration, [self.config.visual.joint_acceleration] * 29
        ]
        interval = dt if self.last_dt is None else (dt + self.last_dt) / 2
        velocity = self.v + np.clip(-self.v, -acceleration * interval, acceleration * interval)
        x = np.clip(self.x + velocity * dt, self.travel_lo, self.travel_hi)
        self.v = (x - self.x) / dt
        self.x, self.last_dt = x, dt
        self.q, self.velocity = x[4:].copy(), self.v[4:].copy()
        self.root = quaternion(self.body.put(x))

    def step(self, wall_time, estimate, state, hand_openings, *, motion_time=None):
        """Advance by physical time; camera timestamps and freshness use wall time."""
        cfg = self.config.visual
        now = wall_time if motion_time is None else float(motion_time)
        if self.started is None:
            self.started = self.last_time = now
            self.started_wall = wall_time
        control_dt = max(float(now - self.last_time), 0.0)
        dt = min(control_dt, 0.1)  # cap trajectory progress, not the physical rate interval
        self.last_time = now
        if now - self.started > cfg.timeout_seconds:
            return self.result("timeout", True)
        if estimate is None or not 0 <= wall_time - estimate.timestamp <= cfg.vision_timeout:
            self.was_visible, self.ready_since = False, None
            self.brake(control_dt)
            stale = (
                wall_time - (self.last_seen if self.last_seen is not None else self.started_wall) > cfg.vision_timeout
            )
            return self.result("vision_lost" if stale else "waiting_for_vision", stale)
        dt = dt if self.was_visible else 0.0
        self.was_visible, self.last_seen = True, estimate.timestamp
        self.elapsed += dt
        base, origin = rotation(state["pelvis"]["quat"]), np.asarray(state["pelvis"]["pos"])
        tools, tool_rotations = self.solver.tool_poses(np.asarray(state["joints"]))
        tools, tool_rotations = tools @ base.T + origin, base @ tool_rotations
        local_centres = estimate.transform(self.target.grasp_points_local)
        if not self.target.preserve_point_order:
            local_centres = local_centres[np.argsort(-local_centres[:, 1])]
        centres = local_centres @ base.T + origin
        obj_rotation, obj_position = (
            base @ estimate.rotation @ self.target.capture_rotation_local,
            origin + base @ estimate.position,
        )
        # A reachability probe owns exact world capture centres.  Ordinary
        # crate execution still permits the historical slide along the rim.
        targets = centres if self.target.reach_only else self.region.nearest(centres, obj_rotation[:, 1], tools)
        # The cached camera-admitted pose owns reach targets. Current simulation
        # state independently checks displacement, attachment and load safety.
        if self.target.object_key in state.get("objects", {}):
            observed_object = state["objects"][self.target.object_key]
            obj_position = np.asarray(observed_object["pos"])
            obj_rotation = rotation(observed_object["quat"]) @ self.target.capture_rotation_local
        self.diagnostics = dict(
            observed_grips=centres, measured_tools=tools, information_source=self.target.information_source
        )
        if self.phase == "observe":
            axis = centres[0] - centres[1]
            yaw = np.arctan2(-axis[0], axis[1])
            attitude = (
                Rotation.from_euler("z", yaw).as_matrix()
                @ Rotation.from_euler("y", cfg.palm_pitch_deg, degrees=True).as_matrix()
            )
            self.attitudes = np.array([attitude] * 2)
            admission = self.body.solve(targets, self.attitudes)
            self.diagnostics.update(
                admission_hand_error=admission.hand_errors,
                admission_foot_error=admission.foot_errors,
                admission_com_shift=admission.com_shift,
            )
            if not admission.reachable:
                return self.result("unsupported_assisted_reach", True)
            self.posture = admission.joints.copy()
            final_torso = self.solver.ik.data.xmat[self.body.torso].reshape(3, 3).copy()
            self.body.put(self.x)
            initial_torso = self.solver.ik.data.xmat[self.body.torso].reshape(3, 3).copy()
            self.reach_rotation = Slerp([0, 1], Rotation.from_matrix([initial_torso, final_torso]))
            self.start_tools = self.body.tools(self.x)
            self.body.put(self.x)
            initial_attitudes = self.solver.ik.data.xmat[self.body.hands].reshape(2, 3, 3)
            self.rotations = [Slerp([0, 1], Rotation.from_matrix([a, attitude])) for a in initial_attitudes]
            distance = np.max(np.linalg.norm(targets - self.start_tools, axis=1))
            self.duration = max(3.0, 1.875 * distance / cfg.approach_speed)
            self.reach_object = obj_position.copy()
            self.enter("reach")
        self.diagnostics.update(
            reference_elapsed=self.elapsed, planned_reach_seconds=self.duration, control_interval=control_dt
        )
        if self.phase == "reach":
            if np.linalg.norm(obj_position - self.reach_object) > self.region.half_length:
                return self.result("basket_displaced_before_capture", True)
            error = float(np.max(np.linalg.norm(tools - targets, axis=1)))
            if self.progress_time is None or error < self.best_error - 0.005:
                self.best_error, self.progress_time = error, now
            if not self.target.reach_only and self.elapsed > self.duration and now - self.progress_time > 3.0:
                return self.result("reach_stalled", True)
            if self.target.reach_only:
                self.diagnostics.update(capture_error=np.linalg.norm(tools - targets, axis=1), capture_ready=False)
            else:
                ready, errors = self.region.contains(centres, obj_rotation, tools, tool_rotations)
                self.diagnostics.update(capture_error=errors, capture_ready=ready)
                self.ready_since = (now if self.ready_since is None else self.ready_since) if ready else None
                if self.ready_since is not None and now - self.ready_since >= 0.1:
                    self.closed_targets = self.body.tools(self.x)
                    self.enter("close")
            if self.phase == "reach":
                u, _ = blend(self.elapsed, self.duration)
                self.torso_target = self.reach_rotation(u).as_matrix()
                commanded = self.start_tools + u * (targets - self.start_tools)
                attitudes = np.array([r(u).as_matrix() for r in self.rotations])
                self.advance(commanded, attitudes, tools, control_dt)
        if self.phase == "close":
            # Maintain the paired world reference while smoothly braking any residual body motion.
            self.advance(self.closed_targets - cfg.tracking_gain * self.feedback, self.attitudes, tools, control_dt)
            self.closure = min(1.0, self.elapsed / cfg.close_seconds)
            if self.elapsed >= cfg.close_seconds + 0.2:
                if min(hand_openings) < 0.003:
                    return self.result("empty_gripper", True)
                self.closed_tools = tools.copy()
                self.closed_relative = (tools - obj_position) @ obj_rotation
                self.closed_object = obj_position.copy()
                # The held FK already includes feedback; remove it before adding
                # the same correction to the new trajectory, preserving its start.
                self.lift_targets = self.body.tools(self.x) - cfg.tracking_gain * self.feedback
                self.stand_targets = self.lift_targets + self.start_tools.mean(axis=0) - self.lift_targets.mean(axis=0)
                self.stand_targets[:, 2] = np.maximum(
                    self.stand_targets[:, 2], self.lift_targets[:, 2] + cfg.lift_height
                )
                self.body.put(self.x)
                planned_torso = self.solver.ik.data.xmat[self.body.torso].reshape(3, 3).copy()
                measured_torso = base @ self.solver.ik.body_pose(np.asarray(state["joints"]), "torso_link")[1]
                # Do not ask a lagging torso to bend farther at lift-off.
                initial_torso = min((planned_torso, measured_torso), key=lambda r: np.arccos(np.clip(r[2, 2], -1, 1)))
                self.torso_target = initial_torso.copy()
                self.stand_rotation = Slerp([0, 1], Rotation.from_matrix([initial_torso, self.body.heading]))
                self.stand_posture = self.posture.copy()
                self.recovery_started = None
                self.enter("lift")
        elif self.phase in ("lift", "stand", "hold"):
            attachment = float(
                np.max(np.linalg.norm(tools - (self.closed_relative @ obj_rotation.T + obj_position), axis=1))
            )
            rise = float(obj_position[2] - self.closed_object[2])
            tilt = float(np.degrees(np.arccos(np.clip(obj_rotation[2, 2], -1, 1))))
            self.diagnostics.update(attachment_error=attachment, observed_rise=rise, observed_tilt=tilt)
            torso = base @ self.solver.ik.body_pose(np.asarray(state["joints"]), "torso_link")[1]
            lean = float(np.degrees(np.arccos(np.clip(torso[2, 2], -1, 1))))
            self.diagnostics["observed_torso_lean_deg"] = lean
            if self.previous_load is not None and control_dt > 0:
                previous_position, previous_torso = self.previous_load
                basket_speed = float(np.linalg.norm(obj_position - previous_position) / control_dt)
                torso_speed = float(Rotation.from_matrix(torso @ previous_torso.T).magnitude() / control_dt)
                self.diagnostics.update(observed_basket_speed=basket_speed, observed_torso_speed=torso_speed)
            self.previous_load = obj_position.copy(), torso.copy()
            if min(hand_openings) < 0.003 or (self.phase != "lift" and attachment > 0.06):
                return self.result("lost_attachment", True)
            if tilt > cfg.max_tilt_deg:
                return self.result("basket_tilt", True)
            if self.phase in ("lift", "stand"):
                delivered_probe = np.mean(tools[:, 2] - self.closed_tools[:, 2]) >= cfg.probe_height
                if self.recovery_started is None:
                    if delivered_probe:
                        if rise < cfg.probe_height * 0.35 or attachment > 0.03:
                            return self.result("probe_not_attached", True)
                        # A phase label changes, not the motion clock or reference.
                        self.recovery_started = self.elapsed
                        self.phase = "stand"
                    elif self.elapsed >= cfg.lift_seconds + 2:
                        return self.result("probe_not_attached", True)
                lift_u, _ = blend(self.elapsed, cfg.lift_seconds)
                recovery_time = 0 if self.recovery_started is None else self.elapsed - self.recovery_started
                stand_u, _ = blend(recovery_time, cfg.stand_seconds)
                lifted = self.lift_targets + [0, 0, cfg.lift_height]
                targets = self.lift_targets + [0, 0, lift_u * cfg.lift_height]
                targets += stand_u * (self.stand_targets - lifted)
                self.torso_target = self.stand_rotation(stand_u).as_matrix()
                self.posture = self.stand_posture + stand_u * (self.body.q0 - self.stand_posture)
                self.diagnostics.update(lift_progress=lift_u, recovery_progress=stand_u, loaded_targets=targets.copy())
                self.advance(targets, self.attitudes, tools, control_dt)
                if (
                    self.recovery_started is not None
                    and recovery_time >= cfg.stand_seconds
                    and self.elapsed >= cfg.lift_seconds
                    and lean <= cfg.upright_tolerance_deg
                    and attachment < 0.03
                    and rise >= cfg.lift_height * 0.6
                ):
                    self.enter("hold")
                elif recovery_time >= cfg.stand_seconds + 4:
                    return self.result("stand_not_delivered", True)
            else:
                self.advance(self.stand_targets, self.attitudes, tools, control_dt)
                if self.elapsed >= cfg.hold_seconds + 2.0:
                    return self.result("unstable_hold", True)
                stable = (
                    attachment < 0.03
                    and rise >= 0.6 * cfg.lift_height
                    and min(hand_openings) >= 0.003
                    and lean <= cfg.upright_tolerance_deg
                )
                if not stable:
                    self.hold_window.clear()
                    self.ready_since = None
                else:
                    self.hold_window.append((now, obj_position.copy()))
                    while len(self.hold_window) > 1 and self.hold_window[1][0] <= now - cfg.hold_seconds:
                        self.hold_window.popleft()
                    positions = np.array([p for _, p in self.hold_window])
                    radius = float(np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
                    self.diagnostics["hold_position_radius"] = radius
                    self.ready_since = self.hold_window[0][0] if radius <= cfg.hold_position_tolerance else None
                if self.ready_since is not None and now - self.ready_since >= cfg.hold_seconds:
                    return self.result("completed", True)
        return self.result()
