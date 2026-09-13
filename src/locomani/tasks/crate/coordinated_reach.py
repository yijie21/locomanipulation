"""Visible-target reach with simulation-assisted, continuously coordinated body feedback."""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.tasks.crate.grasp_region import GRASP_POINTS, RimRegion
from locomani.tasks.crate.visual_low_reach import LowReachCommand, blend, quaternion, rotation
from locomani.tasks.crate.whole_body_ik import WholeBodyIK


class CoordinatedReach:
    """One supported body reference through descent, capture and paired lifting.

    Simulation supplies measured world poses; reference motion remains bounded,
    feet stay anchored, and unseen targets cannot advance the trajectory.
    """

    def __init__(self, solver, config, state, *, reference_joints, reference_quaternion):
        self.solver, self.config = solver, config
        self.region = RimRegion()
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
            self.last_dt = None
            self.v[:] = 0
            self.velocity[:] = 0
            stale = (
                wall_time - (self.last_seen if self.last_seen is not None else self.started_wall) > cfg.vision_timeout
            )
            return self.result("vision_lost" if stale else "waiting_for_vision", stale)
        dt = dt if self.was_visible else 0.0
        control_dt = control_dt if self.was_visible else 0.0
        self.was_visible, self.last_seen = True, estimate.timestamp
        self.elapsed += dt
        base, origin = rotation(state["pelvis"]["quat"]), np.asarray(state["pelvis"]["pos"])
        tools, tool_rotations = self.solver.tool_poses(np.asarray(state["joints"]))
        tools, tool_rotations = tools @ base.T + origin, base @ tool_rotations
        local_centres = estimate.transform(GRASP_POINTS)
        centres = local_centres[np.argsort(-local_centres[:, 1])] @ base.T + origin
        obj_rotation, obj_position = base @ estimate.rotation, origin + base @ estimate.position
        targets = self.region.nearest(centres, obj_rotation[:, 1], tools)
        self.diagnostics = dict(observed_grips=centres, measured_tools=tools, information_source="simulation_assisted")
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
            if self.elapsed > self.duration and now - self.progress_time > 3.0:
                return self.result("reach_stalled", True)
            ready, errors = self.region.contains(centres, obj_rotation, tools, tool_rotations)
            self.diagnostics.update(capture_error=errors, capture_ready=ready)
            self.ready_since = (now if self.ready_since is None else self.ready_since) if ready else None
            if self.ready_since is not None and now - self.ready_since >= 0.1:
                self.closed_targets = self.body.tools(self.x)
                self.enter("close")
            else:
                u, _ = blend(self.elapsed, self.duration)
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
            if self.phase == "lift":
                u, _ = blend(self.elapsed, cfg.lift_seconds)
                self.advance(self.lift_targets + [0, 0, u * cfg.lift_height], self.attitudes, tools, control_dt)
                delivered_probe = np.mean(tools[:, 2] - self.closed_tools[:, 2]) >= cfg.probe_height
                if delivered_probe and (rise < cfg.probe_height * 0.35 or attachment > 0.03):
                    return self.result("probe_not_attached", True)
                if self.elapsed >= cfg.lift_seconds and rise >= cfg.lift_height * 0.6 and attachment < 0.03:
                    self.stand_start = self.body.tools(self.x) - cfg.tracking_gain * self.feedback
                    # Move the attached pair back to the upright ready-pose centre.
                    # Preserve its separation and attitude; never command the object.
                    self.stand_targets = (
                        self.stand_start + self.start_tools.mean(axis=0) - self.stand_start.mean(axis=0)
                    )
                    self.body.put(self.x)
                    initial_torso = self.solver.ik.data.xmat[self.body.torso].reshape(3, 3).copy()
                    self.stand_rotation = Slerp([0, 1], Rotation.from_matrix([initial_torso, self.body.heading]))
                    self.stand_posture = self.posture.copy()
                    self.enter("stand")
                elif self.elapsed >= cfg.lift_seconds + 2:
                    return self.result("probe_not_attached", True)
            elif self.phase == "stand":
                u, _ = blend(self.elapsed, cfg.stand_seconds)
                self.torso_target = self.stand_rotation(u).as_matrix()
                self.posture = self.stand_posture + u * (self.body.q0 - self.stand_posture)
                self.advance(
                    self.stand_start + u * (self.stand_targets - self.stand_start), self.attitudes, tools, control_dt
                )
                if self.elapsed >= cfg.stand_seconds and lean <= cfg.upright_tolerance_deg and attachment < 0.03:
                    self.enter("hold")
                elif self.elapsed >= cfg.stand_seconds + 4:
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
