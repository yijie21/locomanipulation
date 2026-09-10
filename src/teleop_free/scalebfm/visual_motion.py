"""Generate continuous ScaleBFM references from held task intent and robot state.

All feedback here is proprioceptive. Object/scene truth is not accessible.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from visual_reference import PoseTransition, bounded


class GraspMotion:
    def __init__(self, robot, hand_rel, dt=0.02, height=0.76, previous_wrist=None, integral_gain=0.0, reference_style='legacy'):
        self.dt, self.height, self.hand_rel = dt, height, hand_rel
        self.xy = robot.pelvis_position[:2].copy()
        self.velocity = np.zeros(2)
        self.yaw = float(Rotation.from_quat(robot.pelvis_quaternion, scalar_first=True).as_euler("xyz")[2])
        self.grip = robot.gripper_command
        self.phase = None
        self.transition = None
        self.elapsed = 0.0
        self.offset = np.array([0.1515, 0, 0])
        self.integral_gain = integral_gain
        if reference_style not in ('legacy', 'direct'):
            raise ValueError('Unknown reference style')
        self.bias = np.array([.100, -.030, .028]) if reference_style == 'legacy' else np.zeros(3)
        self.correction = np.zeros(3)
        self.previous_wrist = previous_wrist if previous_wrist is not None else (
            robot.right_pad - Rotation.from_quat(robot.right_quaternion, scalar_first=True).apply(self.offset), robot.right_quaternion.copy())

    def _hand_raw(self, intent, robot, xy, yaw, side):
        r = Rotation.from_euler("z", yaw)
        if side == "left" or intent.pad_target is None:
            p, matrix = self.hand_rel[side]
            return np.array([*xy, self.height]) + r.apply(p), (r * Rotation.from_matrix(matrix)).as_quat(scalar_first=True)
        stand_r = Rotation.from_euler("z", intent.yaw)
        error = stand_r.inv().apply(intent.pad_target - robot.right_pad)
        pad = intent.pad_target + stand_r.apply(self.bias + np.clip(error, -0.05, 0.05)) + self.correction
        q = intent.wrist_quaternion
        return pad - Rotation.from_quat(q, scalar_first=True).apply(self.offset), q

    def targets(self, intent, robot, future_count=6):
        dt = self.dt
        # Correct steady tracking bias only after entry motion has settled.
        # During closure the correction stays fixed, avoiding integration of
        # contact forces. The resulting command, including this correction, is
        # the recorded action; deployment does not apply a second integrator.
        if (self.phase in ("reach", "entry") and intent.phase == self.phase and not intent.pause
                and self.transition is not None and self.elapsed >= self.transition.duration
                and np.linalg.norm(robot.right_velocity) < 0.05):
            error = intent.pad_target - robot.right_pad
            self.correction = bounded(self.correction + self.integral_gain * dt * bounded(error, .05), .10)
        error = intent.stand_xy - self.xy
        distance = float(np.linalg.norm(error))
        speed = min(0.5, np.sqrt(2 * 0.6 * distance))
        desired = np.zeros(2) if intent.pause else error / max(distance, 1e-9) * speed
        # A reference reaching the stand does not mean the robot arrived. Keep
        # native moving futures while measured distance is outside the planner's
        #10cm arrival corridor; otherwise delayed gait can become a static
        # attractor with zero speed. Do not force motion inside that corridor or
        # during precision phases/visual pause.
        measured_error = intent.stand_xy - robot.pelvis_position[:2]
        measured_distance = float(np.linalg.norm(measured_error))
        delayed_approach = (not intent.pause and intent.phase in ('walk', 'reach')
                            and measured_distance > .10 and distance < .04)
        if delayed_approach:
            desired = measured_error / measured_distance * min(.12, .6 * (measured_distance - .08))
        self.velocity += bounded(desired - self.velocity, 0.6 * dt)
        next_xy = self.xy + self.velocity * dt
        if not intent.pause and np.dot(intent.stand_xy - next_xy, error) <= 0:
            next_xy = intent.stand_xy.copy()
            if not delayed_approach:
                self.velocity[:] = 0
        # Bound reference lead if the plant is delayed, preserving moving futures.
        lead = next_xy - robot.pelvis_position[:2]
        self.xy = robot.pelvis_position[:2] + bounded(lead, 0.25)
        yaw_error = (intent.yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = 0.0 if intent.pause else float(np.clip(yaw_error / dt, -0.8, 0.8))
        self.yaw += yaw_rate * dt
        self.grip += float(np.clip(intent.gripper - self.grip, -0.08 * dt, 0.08 * dt))
        raw = self._hand_raw(intent, robot, self.xy, self.yaw, "right")
        if self.phase != intent.phase and not intent.pause:
            duration = {"walk": 0.5, "reach": 2.5, "entry": 1.5, "close": 0.5, "lift": 1.5, "carry": 1.2}[intent.phase]
            self.transition = PoseTransition(self.previous_wrist, raw, duration)
            self.elapsed = 0.0; self.phase = intent.phase
        out = {name: ([], []) for name in ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")}
        previous = self.previous_wrist
        for j in range(future_count):
            tau = j * dt
            xy, yaw = self.xy + self.velocity * tau, self.yaw + yaw_rate * tau
            poses = {"pelvis": (np.array([*xy, self.height]), Rotation.from_euler("z", yaw).as_quat(scalar_first=True)),
                     "left_wrist_yaw_link": self._hand_raw(intent, robot, xy, yaw, "left")}
            raw = self._hand_raw(intent, robot, xy, yaw, "right")
            if intent.pause or self.transition is None:
                wrist = self.previous_wrist
            else:
                wrist = self.transition.sample(*raw, self.elapsed + tau)
            # Last-resort continuity bound for a newly observed object displacement.
            p = previous[0] + bounded(wrist[0] - previous[0], 0.02)
            r0 = Rotation.from_quat(previous[1], scalar_first=True)
            delta = (Rotation.from_quat(wrist[1], scalar_first=True) * r0.inv()).as_rotvec()
            q = (Rotation.from_rotvec(bounded(delta, np.radians(5))) * r0).as_quat(scalar_first=True)
            previous = p, q
            poses["right_wrist_yaw_link"] = previous
            for name, (p, q) in poses.items():
                out[name][0].append(p); out[name][1].append(q)
        out = {name: (np.asarray(p), np.asarray(q)) for name, (p, q) in out.items()}
        self.previous_wrist = out["right_wrist_yaw_link"][0][0].copy(), out["right_wrist_yaw_link"][1][0].copy()
        if not intent.pause:
            self.elapsed += dt
        return out, self.grip
