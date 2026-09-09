"""Tabletop grasp geometry: initial candidate and live post-crouch replanning.

This retains the current upright-object strategy. It does not certify a swept
collision-free path and does not generalize to handles, drawers, or fallen objects.
"""

from dataclasses import dataclass

import adaptive_grasp
import numpy as np
from primitives import Primitives


@dataclass(frozen=True)
class TabletopTarget:
    key: str
    side: str
    grip_z: float
    surface_top: float
    box: dict
    body: str | None = None
    surface_bodies: tuple[str, ...] = ()
    below_center: float | None = None

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError(f"Unsupported grasp side: {self.side}")

        if self.below_center is not None:
            half_z = float(self.box["half_extent"][2])
            if not np.isfinite(self.below_center) or self.below_center < 0 or self.below_center >= half_z:
                raise ValueError("Grasp offset must remain inside the object")


class TabletopGrasp:
    def __init__(self, robot, target: TabletopTarget):
        self.robot, self.target = robot, target

    def grip(self, state):
        obj = self.robot.object_pos(self.target.key, state)
        if self.target.below_center is None:
            return np.array([obj[0], obj[1], self.target.grip_z])
        local = np.array(self.target.box.get("center", [0.0, 0.0, 0.0]), dtype=float)
        local[2] -= self.target.below_center
        return np.asarray(obj) + Primitives._rot(state["objects"][self.target.key]["quat"]) @ local

    def _plan(self, state, wrist, rotation):
        p, t = self.robot, self.target
        grip = self.grip(state)
        yaw = p.pelvis_yaw(state)
        lat = -0.31 if t.side == "right" else 0.31
        pelvis = state["pelvis"]["pos"]
        anchor = np.array([pelvis[0] - np.sin(yaw) * lat, pelvis[1] + np.cos(yaw) * lat, pelvis[2] + 0.44])
        delta = grip[:2] - p.pelvis_xy(state)
        bearing = delta / max(float(np.linalg.norm(delta)), 1e-6)
        return adaptive_grasp.plan(
            grip,
            wrist,
            rotation,
            t.surface_top,
            box=t.box,
            obj_rot=Primitives._rot(state["objects"][t.key]["quat"]),
            azimuth_from=anchor,
            avoid_dir=np.array([*bearing, 0.0]),
            avoid_sign=1.0 if t.side == "right" else -1.0,
        )

    def initial(self, state):
        grip = self.grip(state)
        delta = grip[:2] - self.robot.pelvis_xy(state)
        bearing = delta / max(float(np.linalg.norm(delta)), 1e-6)
        virtual = grip - 0.25 * np.array([*bearing, 0.0]) + [0.0, 0.0, 0.08]
        return self._plan(state, virtual, Primitives._rot(state["palms"][self.target.side]["quat"]))

    def replan(self, wrist, rotation):
        return self._plan(self.robot.obs(), wrist, rotation)

    def transport_guard(self, ik, config):
        from transport_guard import TransportGuard

        if self.target.body is None:
            raise ValueError("Collision screening needs the target body name")
        return TransportGuard(
            ik,
            self.target.key,
            self.target.body,
            self.target.surface_bodies,
            margin=config.transport_margin,
            joint_step=config.sweep_joint_step,
            include_body=config.ready_grasp_recovery,
            finger_only_target_contact=getattr(config, "tool_point_grasp", False),
        )
