"""Task decisions from current visual grasp error, with no stage or clock input."""

from dataclasses import dataclass

import numpy as np

from locomani.tasks.bottle.image_servo_config import ServoLimits
from locomani.tasks.bottle.ready_grasp_policy import OFFSET


@dataclass(frozen=True)
class ServoObservation:
    target: np.ndarray  # RGB-derived grasp point, in the pelvis frame.
    palm: np.ndarray
    rotation: np.ndarray
    grasp_rotation: np.ndarray
    up: np.ndarray


DEFAULT_LIMITS = ServoLimits()


@dataclass(frozen=True)
class ServoCommand:
    aim: np.ndarray
    rotation: np.ndarray
    intent: str
    close: bool = False


def decide(obs: ServoObservation, limits=DEFAULT_LIMITS):
    """Align outside the bottle, then enter; a parked input needs no warm-up."""
    measured_error = obs.rotation.T @ (obs.target - obs.palm) - OFFSET
    height_error = float((obs.target - obs.palm - obs.rotation @ OFFSET) @ obs.up)
    if (
        abs(measured_error[0]) < limits.close_depth
        and abs(measured_error[1]) < limits.close_lateral
        and abs(height_error) < limits.close_height
        and abs(float(obs.up @ obs.rotation[:, 1])) < 0.25
    ):
        return ServoCommand(obs.palm.copy(), obs.rotation.copy(), "capture", True)
    desired = obs.target - obs.grasp_rotation @ OFFSET
    entry = obs.grasp_rotation[:, 0]
    error = desired - obs.palm
    depth = float(error @ entry)
    cross = error - depth * entry
    angle = float(np.arccos(np.clip((np.trace(obs.rotation.T @ obs.grasp_rotation) - 1) / 2, -1.0, 1.0)))
    aligned = np.linalg.norm(cross) < limits.alignment_tolerance and angle < 0.16
    if not aligned and depth < limits.entry_gap - 0.015:
        # Dex1 fingers extend past the grasp centre: clear them before descending.
        aim = obs.palm + (depth - limits.entry_gap) * entry
        return ServoCommand(aim, obs.rotation.copy(), "clear_entry")
    if not aligned:
        # Hold the current depth while correcting height/sideways error;
        # moving forward to a pregrasp waypoint would consume finger clearance.
        aim = obs.palm + cross + min(depth - limits.entry_gap, 0.0) * entry
        return ServoCommand(aim, obs.grasp_rotation.copy(), "align")
    return ServoCommand(desired, obs.grasp_rotation.copy(), "enter")
