"""Event-driven bottle task decisions from scene estimates and proprioception.

No simulator access. Time is used by the separate trajectory generator, not to
decide whether the robot has arrived, aligned or lifted the object.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from visual_observation import SceneEstimate


@dataclass(frozen=True)
class RobotObservation:
    pelvis_position: np.ndarray
    pelvis_quaternion: np.ndarray
    right_pad: np.ndarray
    right_quaternion: np.ndarray
    right_velocity: np.ndarray
    gripper_command: float


@dataclass(frozen=True)
class GraspIntent:
    phase: str
    stand_xy: np.ndarray
    yaw: float
    pad_target: np.ndarray | None
    wrist_quaternion: np.ndarray | None
    gripper: float
    pause: bool = False


@dataclass(frozen=True)
class GraspConfig:
    stand_distance: float = 0.375
    stand_backstep: float = 0.06
    stand_left: float = 0.15
    reach_start: float = 0.5
    pre_back: float = 0.10
    carry_back: float = 0.07
    pre_tolerance: float = 0.035
    close_along: float = 0.02
    close_across: float = 0.018
    close_speed: float = 0.10
    pitch_feedforward_deg: float = -13.0
    lift_height: float = 0.10
    carry_clearance: float | None = None
    edge_clearance: float | None = None
    high_table_front: float | None = None
    facing: str = 'bottle'


class VisualGraspPlanner:
    def __init__(self, config=None):
        self.config = config or GraspConfig()
        self.phase = "walk"
        self.hold_xy = None
        self.hold_yaw = None
        self.close_anchor = None
        self.lift_anchor = None
        self.lift_reference_z = None
        self.carry_anchor = None

    def decide(self, scene: SceneEstimate, robot: RobotObservation) -> GraspIntent:
        cfg = self.config
        yaw_now = float(Rotation.from_quat(robot.pelvis_quaternion, scalar_first=True).as_euler("xyz")[2])
        if scene.bottle_position is None or scene.confidence <= 0:
            return GraspIntent(self.phase, robot.pelvis_position[:2].copy(), yaw_now,
                               None, None, robot.gripper_command, pause=True)
        bottle = np.asarray(scene.bottle_position)
        inward = np.asarray(scene.inward_normal)
        left = np.array([-inward[1], inward[0], 0.0])
        front = cfg.stand_distance - cfg.stand_backstep
        height_weight = float(np.clip((scene.table_height - .6211385812222395) / .14, 0, 1))
        if cfg.high_table_front is not None:
            front = (1 - height_weight) * front + height_weight * cfg.high_table_front
        if cfg.edge_clearance is not None:
            depth = float((bottle - np.asarray(scene.edge_point)) @ inward)
            clearance = cfg.edge_clearance
            if cfg.high_table_front is not None:
                clearance = (1 - height_weight) * clearance + height_weight * .15
            front = max(front, depth + clearance)
        stand = bottle[:2] - inward[:2] * front + left[:2] * cfg.stand_left
        delta = bottle[:2] - stand
        if cfg.facing not in ('bottle', 'edge'):
            raise ValueError('Unknown facing rule')
        facing = inward[:2] if cfg.facing == 'edge' else delta
        yaw = float(np.arctan2(facing[1], facing[0]))
        if self.hold_xy is not None:
            stand, yaw = self.hold_xy, self.hold_yaw
        approach = np.array([np.cos(yaw), np.sin(yaw), 0])
        quaternion = (Rotation.from_euler("z", yaw) * Rotation.from_euler("y", cfg.pitch_feedforward_deg, degrees=True)).as_quat(scalar_first=True)
        grasp = bottle + np.array([0, 0, -0.02])
        pre = grasp - approach * cfg.pre_back
        remaining = float(np.linalg.norm(stand - robot.pelvis_position[:2]))
        if self.phase == "walk" and remaining <= cfg.reach_start and scene.precision_allowed:
            self.phase = "reach"
        if self.phase == "reach" and remaining <= 0.10 and np.linalg.norm(robot.right_pad - pre) <= cfg.pre_tolerance and scene.precision_allowed:
            self.phase = "entry"
            self.hold_xy, self.hold_yaw = robot.pelvis_position[:2].copy(), yaw_now
            stand, yaw = self.hold_xy, self.hold_yaw
            approach = np.array([np.cos(yaw), np.sin(yaw), 0])
            quaternion = (Rotation.from_euler("z", yaw) * Rotation.from_euler("y", cfg.pitch_feedforward_deg, degrees=True)).as_quat(scalar_first=True)
        if self.phase == "entry" and scene.precision_allowed:
            error = robot.right_pad - grasp
            along = float(error @ approach)
            across = np.linalg.norm(error - along * approach)
            if abs(along) <= cfg.close_along and across <= cfg.close_across and np.linalg.norm(robot.right_velocity) <= cfg.close_speed:
                self.phase = "close"
                self.close_anchor = grasp.copy()
                self.lift_reference_z = float(bottle[2])
        if self.phase == "close" and robot.gripper_command <= -0.0225:
            self.phase = "lift"
            self.lift_anchor = robot.right_pad + np.array([0, 0, cfg.lift_height])
        if self.phase == "lift" and scene.precision_allowed and self.lift_reference_z is not None:
            # A visible lift outcome, not a contact/attached flag or an elapsed timer.
            rise = bottle[2] - self.lift_reference_z
            if rise >= 0.08:
                self.phase = "carry"
                self.carry_anchor = robot.right_pad - approach * cfg.carry_back
        if self.phase == "carry" and scene.precision_allowed and cfg.carry_clearance is not None and self.lift_reference_z is not None:
            # Use visible bottle height to maintain clearance during retraction.
            # This also compensates slow slip; a fixed wrist height cannot.
            rise = bottle[2] - self.lift_reference_z
            self.carry_anchor[2] = robot.right_pad[2] + np.clip(cfg.carry_clearance - rise, -0.04, 0.04)
        pad = {"walk": None, "reach": pre, "entry": grasp,
               "close": self.close_anchor, "lift": self.lift_anchor,
               "carry": self.carry_anchor}[self.phase]
        grip = 0.0245 if self.phase in ("walk", "reach", "entry") else -0.023
        # Permit short-memory walking, but no precision motion without a current
        # visual observation. Robot-only grip execution is not a new object cue.
        pause = not scene.precision_allowed and self.phase in ("reach", "entry", "lift", "carry")
        return GraspIntent(self.phase, np.asarray(stand).copy(), yaw,
                           None if pad is None else np.asarray(pad).copy(), quaternion, grip, pause)
