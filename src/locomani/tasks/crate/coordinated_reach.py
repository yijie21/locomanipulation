"""Visible-target reach with simulation-assisted, continuously coordinated body feedback."""

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
        self.diagnostics = {}

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
        vlo = np.maximum(
            vlo, -np.maximum(0, np.sqrt(2 * acceleration * np.maximum(self.x - self.travel_lo, 0)) - acceleration * dt)
        )
        vhi = np.minimum(
            vhi, np.maximum(0, np.sqrt(2 * acceleration * np.maximum(self.travel_hi - self.x, 0)) - acceleration * dt)
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

    def step(self, now, estimate, state, hand_openings):
        cfg = self.config.visual
        if self.started is None:
            self.started = self.last_time = now
        dt = float(np.clip(now - self.last_time, 0, 0.1))
        self.last_time = now
        if now - self.started > cfg.timeout_seconds:
            return self.result("timeout", True)
        if estimate is None or not 0 <= now - estimate.timestamp <= cfg.vision_timeout:
            self.was_visible, self.ready_since = False, None
            self.last_dt = None
            self.v[:] = 0
            self.velocity[:] = 0
            stale = now - (self.last_seen if self.last_seen is not None else self.started) > cfg.vision_timeout
            return self.result("vision_lost" if stale else "waiting_for_vision", stale)
        dt = dt if self.was_visible else 0.0
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
            if not admission.reachable:
                return self.result("unsupported_assisted_reach", True)
            self.posture = admission.joints.copy()
            self.start_tools = self.body.tools(self.x)
            self.body.put(self.x)
            initial_attitudes = self.solver.ik.data.xmat[self.body.hands].reshape(2, 3, 3)
            self.rotations = [Slerp([0, 1], Rotation.from_matrix([a, attitude])) for a in initial_attitudes]
            distance = np.max(np.linalg.norm(targets - self.start_tools, axis=1))
            self.duration = max(3.0, 1.875 * distance / cfg.approach_speed)
            self.enter("reach")
        self.diagnostics.update(reference_elapsed=self.elapsed, planned_reach_seconds=self.duration)
        if self.phase == "reach":
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
                self.advance(commanded, attitudes, tools, dt)
        if self.phase == "close":
            # Maintain the paired world reference while smoothly braking any residual body motion.
            self.advance(self.closed_targets - cfg.tracking_gain * self.feedback, self.attitudes, tools, dt)
            self.closure = min(1.0, self.elapsed / cfg.close_seconds)
            if self.elapsed >= cfg.close_seconds + 0.2:
                if min(hand_openings) < 0.003:
                    return self.result("empty_gripper", True)
                self.closed_tools = tools.copy()
                self.closed_relative = (tools - obj_position) @ obj_rotation
                self.closed_object = obj_position.copy()
                self.lift_targets = self.body.tools(self.x)
                self.enter("lift")
        elif self.phase in ("lift", "hold"):
            attachment = float(
                np.max(np.linalg.norm(tools - (self.closed_relative @ obj_rotation.T + obj_position), axis=1))
            )
            rise = float(obj_position[2] - self.closed_object[2])
            tilt = float(np.degrees(np.arccos(np.clip(obj_rotation[2, 2], -1, 1))))
            self.diagnostics.update(attachment_error=attachment, observed_rise=rise, observed_tilt=tilt)
            if tilt > cfg.max_tilt_deg:
                return self.result("basket_tilt", True)
            u, _ = blend(self.elapsed, cfg.lift_seconds) if self.phase == "lift" else (1, 0)
            self.advance(self.lift_targets + [0, 0, u * cfg.lift_height], self.attitudes, tools, dt)
            if self.phase == "lift":
                delivered_probe = np.mean(tools[:, 2] - self.closed_tools[:, 2]) >= cfg.probe_height
                if delivered_probe and (rise < cfg.probe_height * 0.35 or attachment > 0.03):
                    return self.result("probe_not_attached", True)
                if self.elapsed >= cfg.lift_seconds and rise >= cfg.lift_height * 0.6 and attachment < 0.03:
                    self.enter("hold")
                elif self.elapsed >= cfg.lift_seconds + 2:
                    return self.result("probe_not_attached", True)
            else:
                stable = attachment < 0.03 and rise >= 0.6 * cfg.lift_height and min(hand_openings) >= 0.003
                self.ready_since = (now if self.ready_since is None else self.ready_since) if stable else None
                if self.ready_since is not None and now - self.ready_since >= cfg.hold_seconds:
                    return self.result("completed", True)
        return self.result()
