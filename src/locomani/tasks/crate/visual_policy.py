"""Observable crate decisions and coordinated references; no simulator state interface."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.perception import CrateEstimate
from locomani.tasks.crate.visual_config import CrateVisualConfig


@dataclass(frozen=True)
class ExpertCommand:
    positions: np.ndarray
    rotations: np.ndarray
    closure: float
    phase: str
    reason: str = ""
    finished: bool = False


class VisualCratePolicy:
    """Finite history consists of the previous reference and observable transition events."""

    def __init__(self, grasp_points, config: CrateVisualConfig):
        self.local_grips = np.asarray(grasp_points, dtype=float)
        self.config = config
        self.phase = "observe"
        self.started = None
        self.phase_started = None
        self.last_seen = None
        self.reference = None
        self.rotations = None
        self.closed_grips = None
        self.closed_object = None
        self.closed_relative_grips = None
        self.closure = 0.0
        self.ready_since = None
        self.last_time = None
        self.was_visible = False

    def _enter(self, phase, now):
        self.phase, self.phase_started, self.ready_since = phase, now, None

    def _result(self, *, reason="", finished=False):
        return ExpertCommand(self.reference.copy(), self.rotations.copy(), self.closure, self.phase, reason, finished)

    def step(self, now, estimate: CrateEstimate | None, measured_tools, measured_rotations, hand_openings):
        cfg = self.config
        tools = np.asarray(measured_tools)
        if self.started is None:
            self.started = self.phase_started = self.last_time = now
            self.reference = tools.copy()
            self.rotations = np.asarray(measured_rotations).copy()
        dt = float(np.clip(now - self.last_time, 0.001, 0.1))
        self.last_time = now
        if now - self.started > cfg.timeout_seconds:
            return self._result(reason="timeout", finished=True)
        if estimate is None or now - estimate.timestamp > cfg.vision_timeout:
            self.was_visible = False
            self.ready_since = None
            if now - (self.last_seen if self.last_seen is not None else self.started) > cfg.vision_timeout:
                return self._result(reason="vision_lost", finished=True)
            return self._result(reason="waiting_for_vision")
        # Reacquisition does not make up motion skipped during occlusion.
        dt = dt if self.was_visible else 0.0
        self.was_visible = True
        self.last_seen = estimate.timestamp
        grips = estimate.transform(self.local_grips)
        grips = grips[np.argsort(-grips[:, 1])]
        axis = grips[0] - grips[1]
        yaw = np.arctan2(-axis[0], axis[1])
        base_rotation = Rotation.from_euler("z", yaw).as_matrix()
        if self.phase == "observe":
            self._enter("pregrasp", now)

        if self.phase in ("pregrasp", "align"):
            target = grips.copy()
            if self.phase == "pregrasp":
                target += base_rotation @ np.array([-0.05, 0, cfg.pregrasp_above])
                pitch = 0.0
            else:
                pitch = cfg.palm_pitch_deg * np.clip(
                    1 - np.mean(self.reference[:, 2] - grips[:, 2]) / cfg.pregrasp_above, 0, 1
                )
            self.rotations[:] = base_rotation @ Rotation.from_euler("y", pitch, degrees=True).as_matrix()
            # Shared progress: neither hand gets ahead while its partner is still tracking.
            if np.max(np.linalg.norm(self.reference - tools, axis=1)) < 0.065:
                delta = target - self.reference
                delta *= min(1.0, cfg.approach_speed * dt / max(np.max(np.linalg.norm(delta, axis=1)), 1e-9))
                self.reference += delta
            errors = np.linalg.norm(target - tools, axis=1)
            tolerance = 0.028 if self.phase == "pregrasp" else cfg.align_tolerance
            if np.max(errors) < tolerance:
                self.ready_since = now if self.ready_since is None else self.ready_since
                if now - self.ready_since > 0.3:
                    self._enter("align" if self.phase == "pregrasp" else "close", now)
            else:
                self.ready_since = None
        elif self.phase == "close":
            self.closure = min(1.0, self.closure + dt / cfg.close_seconds)
            if self.closure >= 1.0:
                self.ready_since = now if self.ready_since is None else self.ready_since
            if self.ready_since is not None and now - self.ready_since > 0.4:
                if np.any(np.asarray(hand_openings) < 0.003):
                    return self._result(reason="empty_gripper", finished=True)
                self.closed_grips = tools.copy()
                self.closed_object = estimate.position.copy()
                self.closed_relative_grips = (tools - estimate.position) @ estimate.rotation
                self._enter("probe", now)
        elif self.phase in ("probe", "lift", "hold"):
            # One rigid translation supplies both hand references.
            height = cfg.probe_height if self.phase == "probe" else cfg.lift_height
            target = self.closed_grips + np.array([0.0, 0.0, height])
            tracking = np.max(np.linalg.norm(self.reference - tools, axis=1))
            attachment_error = np.max(np.linalg.norm(tools - estimate.transform(self.closed_relative_grips), axis=1))
            if tracking < 0.065 and (self.phase == "probe" or attachment_error < 0.025):
                delta = target - self.reference
                delta *= min(1.0, cfg.approach_speed * dt / max(np.max(np.linalg.norm(delta, axis=1)), 1e-9))
                self.reference += delta
            tilt = np.degrees(np.arccos(np.clip(estimate.rotation[2, 2], -1, 1)))
            if tilt > cfg.max_tilt_deg:
                return self._result(reason="visual_tilt", finished=True)
            if self.phase == "probe" and now - self.phase_started > 1.5:
                object_motion = estimate.position - self.closed_object
                if object_motion[2] < cfg.probe_height * 0.35 or attachment_error > 0.025:
                    return self._result(reason="probe_not_attached", finished=True)
                self._enter("lift", now)
            elif self.phase == "lift" and np.max(np.linalg.norm(target - tools, axis=1)) < 0.025:
                self._enter("hold", now)
            elif self.phase == "hold":
                stable = (
                    np.max(np.linalg.norm(target - tools, axis=1)) < 0.025
                    and attachment_error < 0.025
                    and estimate.position[2] - self.closed_object[2] >= cfg.lift_height * 0.6
                    and np.min(hand_openings) >= 0.003
                )
                self.ready_since = (now if self.ready_since is None else self.ready_since) if stable else None
                if self.ready_since is not None and now - self.ready_since >= cfg.hold_seconds:
                    return self._result(reason="completed", finished=True)
        return self._result()
