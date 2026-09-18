"""Low-rim commands from RGB pose, encoders and IMU; no object truth interface.

The planted-foot frame is reconstructed from encoder FK during the stationary
reach. It is a local geometric reference, not simulator odometry. Loss of sight
pauses the reference clock, including bending, closure and loaded motion.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.tasks.crate.grasp_region import RimRegion
from locomani.tasks.crate.joint_reference import JointReference
from locomani.tasks.crate.whole_body_ik import plan_low_reach


def rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()


def quaternion(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def blend(t, duration):
    """Smootherstep and its derivative, both clipped -- clipping `u` is not enough.

    `u**3 * (10 - 15u + 6u**2)` is 1 at u == 1 in exact arithmetic but evaluates to
    1.0000000000000016 for u just below it, and `coordinated_reach`'s recovery feeds
    this straight to a `Slerp` built on [0, 1], which rejects anything above 1.0.
    Measured 2026-09-18: 171 of 250001 samples across the boundary came back high,
    and one did it in a real round -- a grasp that had already succeeded (rise
    0.218 m, both pads touching) was recorded as `runtime_error`, and because
    `expert.py` re-raises, it took the rest of that batch with it: 3 rounds of 10.
    """
    u = np.clip(t / duration, 0, 1)
    return np.clip(u**3 * (10 - 15 * u + 6 * u * u), 0.0, 1.0), 30 * u * u * (1 - u) ** 2 / duration


class SupportFrame:
    """Stationary double support: estimate base translation from both foot FK poses."""

    def __init__(self, ik, joints, base_quaternion):
        self.ik = ik
        base = rotation(base_quaternion)
        feet = self.feet(joints) @ base.T
        self.origin = np.array([0, 0, 0.04 - np.mean(feet[:, 2])])
        self.anchors = feet + self.origin

    def feet(self, joints):
        return np.array([self.ik.body_pose(joints, f"{side}_ankle_roll_link")[0] for side in ("left", "right")])

    def observe(self, joints, base_quaternion):
        base = rotation(base_quaternion)
        return np.mean(self.anchors - self.feet(joints) @ base.T, axis=0), base


@dataclass(frozen=True)
class LowReachCommand:
    joints: np.ndarray
    velocity: np.ndarray
    root_quaternion: np.ndarray
    closure: float
    phase: str
    reason: str = ""
    finished: bool = False


class VisualLowReach:
    def __init__(
        self, solver, geometry, config, joints, base_quaternion, *, reference_joints=None, reference_quaternion=None
    ):
        self.solver, self.config = solver, config
        self.region = RimRegion()
        self.local_grips = np.asarray(geometry["grasp_points"])
        self.frame = SupportFrame(solver.ik, joints, base_quaternion)
        self.q = np.asarray(joints if reference_joints is None else reference_joints).copy()
        self.velocity = np.zeros(29)
        self.root = np.asarray(base_quaternion if reference_quaternion is None else reference_quaternion).copy()
        self.phase, self.closure = "observe", 0.0
        self.started = self.last_time = self.last_seen = None
        self.was_visible = False
        self.elapsed = 0.0
        self.ready_since = None
        self.reference = None
        self.diagnostics = {}

    def result(self, reason="", finished=False):
        return LowReachCommand(
            self.q.copy(), self.velocity.copy(), self.root.copy(), self.closure, self.phase, reason, finished
        )

    def enter(self, phase):
        self.phase, self.elapsed, self.ready_since = phase, 0.0, None

    def step(self, now, estimate, joints, base_quaternion, hand_openings):
        cfg = self.config.visual
        if self.started is None:
            self.started = self.last_time = now
        dt = float(np.clip(now - self.last_time, 0, 0.1))
        self.last_time = now
        if now - self.started > cfg.timeout_seconds:
            return self.result("timeout", True)
        if estimate is None or not 0 <= now - estimate.timestamp <= cfg.vision_timeout:
            self.was_visible, self.ready_since = False, None
            self.velocity[:] = 0
            if self.reference is not None:
                self.reference.pause()
            stale = now - (self.last_seen if self.last_seen is not None else self.started) > cfg.vision_timeout
            return self.result("vision_lost" if stale else "waiting_for_vision", stale)
        dt = dt if self.was_visible else 0.0
        self.was_visible, self.last_seen = True, estimate.timestamp
        self.elapsed += dt
        origin, base = self.frame.observe(joints, base_quaternion)
        tools, tool_rotations = self.solver.tool_poses(joints)
        tools = tools @ base.T + origin
        tool_rotations = base @ tool_rotations
        centres = estimate.transform(self.local_grips) @ base.T + origin
        centres = centres[np.argsort(-estimate.transform(self.local_grips)[:, 1])]
        object_rotation = base @ estimate.rotation
        object_position = origin + base @ estimate.position
        self.diagnostics = dict(support_origin=origin, observed_grips=centres)
        if self.phase == "observe":
            targets = self.region.nearest(centres, object_rotation[:, 1], tools)
            axis = centres[0] - centres[1]
            yaw = np.arctan2(-axis[0], axis[1])
            attitude = (
                Rotation.from_euler("z", yaw).as_matrix()
                @ Rotation.from_euler("y", cfg.palm_pitch_deg, degrees=True).as_matrix()
            )
            state = dict(joints=joints, pelvis=dict(pos=origin, quat=base_quaternion))
            plan = plan_low_reach(
                self.solver.ik, state, targets, [attitude] * 2, self.config.carry, tool_offset=self.solver.offset
            )
            if not plan.reachable:
                return self.result("unsupported_visual_reach", True)
            self.posture = plan.joints.copy()
            self.attitudes = np.array([attitude] * 2)
            normal = Rotation.from_euler("z", yaw).apply([1, 0, 0])
            staging = targets - 0.04 * normal + [0, 0, cfg.pregrasp_above]
            self.staging_reference = staging.copy()
            goal = self.solver.solve(
                plan.joints,
                (staging - plan.position) @ plan.rotation,
                plan.rotation.T @ self.attitudes,
                iterations=180,
                posture=self.posture,
                rot_weight=0.25,
            )
            if max(goal.errors) > 0.02:
                return self.result("unreachable_visual_staging", True)
            self.q0, self.q1 = self.q.copy(), goal.joints
            self.roots = Slerp([0, 1], Rotation.from_matrix([rotation(self.root), plan.rotation]))
            distance = np.max(np.abs(self.q1 - self.q0))
            self.duration = max(
                4.0, 1.875 * distance / cfg.joint_speed, np.sqrt(5.8 * distance / cfg.joint_acceleration)
            )
            self.enter("bend")
        if self.phase == "bend":
            u, speed = blend(self.elapsed, self.duration)
            self.q = self.q0 + u * (self.q1 - self.q0)
            self.velocity = (speed if dt > 0 else 0) * (self.q1 - self.q0)
            self.root = quaternion(self.roots(u).as_matrix())
            if self.elapsed >= self.duration:
                self.reference = JointReference(
                    self.q, self.solver, speed=cfg.joint_speed, acceleration=cfg.joint_acceleration
                )
                self.tool_reference = self.staging_reference.copy()
                self.enter("align")
        elif self.phase == "align":
            ready, errors = self.region.contains(centres, object_rotation, tools, tool_rotations)
            self.diagnostics.update(capture_error=errors, capture_ready=ready)
            self.ready_since = (now if self.ready_since is None else self.ready_since) if ready else None
            if self.ready_since is not None and now - self.ready_since >= 0.1:
                self.enter("close")
                self.velocity[:] = 0
            else:
                targets = self.region.nearest(centres, object_rotation[:, 1], tools)
                delta = targets - self.tool_reference
                delta *= min(1.0, cfg.approach_speed * dt / max(np.max(np.linalg.norm(delta, axis=1)), 1e-9))
                self.tool_reference += delta
                lead = self.tool_reference - tools
                lead *= np.minimum(1.0, 0.06 / np.maximum(np.linalg.norm(lead, axis=1), 1e-9))[:, None]
                self.tool_reference = tools + lead
                correction = np.clip(cfg.tracking_gain * lead, -cfg.tracking_cap, cfg.tracking_cap)
                target = self.solver.solve(
                    self.q,
                    (self.tool_reference + correction - origin) @ base,
                    base.T @ self.attitudes,
                    iterations=12,
                    posture=self.posture,
                    rot_weight=0.25,
                )
                self.q = self.reference.step(target.joints, dt)
                self.velocity = self.reference.velocity.copy()
        elif self.phase == "close":
            self.closure = min(1.0, self.elapsed / cfg.close_seconds)
            if self.elapsed >= cfg.close_seconds + 0.2:
                if min(hand_openings) < 0.003:
                    return self.result("empty_gripper", True)
                self.closed_tools = tools.copy()
                self.closed_relative = (tools - object_position) @ object_rotation
                self.closed_object = object_position.copy()
                self.lift_start = self.q.copy()
                positions, attitudes = self.solver.tool_poses(self.q)
                goal = self.solver.solve(
                    self.q,
                    positions + base.T @ np.array([0, 0, cfg.lift_height]),
                    attitudes,
                    iterations=180,
                    rot_weight=0.5,
                )
                if max(goal.errors) > 0.02:
                    return self.result("unreachable_visual_lift", True)
                self.lift_target = goal.joints
                distance = np.max(np.abs(self.lift_target - self.lift_start))
                self.duration = max(
                    cfg.lift_seconds,
                    1.875 * distance / cfg.joint_speed,
                    np.sqrt(5.8 * distance / cfg.joint_acceleration),
                )
                self.enter("lift")
        elif self.phase in ("lift", "hold"):
            attachment = float(
                np.max(np.linalg.norm(tools - (self.closed_relative @ object_rotation.T + object_position), axis=1))
            )
            rise = float(object_position[2] - self.closed_object[2])
            tilt = np.degrees(np.arccos(np.clip(object_rotation[2, 2], -1, 1)))
            self.diagnostics.update(attachment_error=attachment, visual_rise=rise, visual_tilt=tilt)
            if tilt > cfg.max_tilt_deg:
                return self.result("visual_tilt", True)
            if self.phase == "lift":
                delivered_probe = np.mean(tools[:, 2] - self.closed_tools[:, 2]) >= cfg.probe_height
                if delivered_probe and (rise < cfg.probe_height * 0.35 or attachment > 0.03):
                    return self.result("probe_not_attached", True)
                u, speed = blend(self.elapsed, self.duration)
                self.q = self.lift_start + u * (self.lift_target - self.lift_start)
                self.velocity = (speed if dt > 0 else 0) * (self.lift_target - self.lift_start)
                if self.elapsed >= self.duration and rise >= cfg.lift_height * 0.6 and attachment < 0.03:
                    self.enter("hold")
                elif self.elapsed >= self.duration + 2.0:
                    return self.result("probe_not_attached", True)
            else:
                stable = attachment < 0.03 and rise >= 0.6 * cfg.lift_height and min(hand_openings) >= 0.003
                self.ready_since = (now if self.ready_since is None else self.ready_since) if stable else None
                if self.ready_since is not None and now - self.ready_since >= cfg.hold_seconds:
                    return self.result("completed", True)
        return self.result()
