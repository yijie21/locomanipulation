"""Planted-foot bookkeeping and the "never cross a line" brake decision, from the robot's measured state.

`FootWatch` runs forward kinematics of the scene model on the privileged state each tick. A foot is planted
while its ankle is within 1 cm of its standing height; a landing is the tick a foot becomes planted. After
IDLE is commanded the robot takes exactly one more step and its frontmost low-body point settles ~16-18 cm
(worst 20.3) past the front planted ankle (walk scene, 2026-09-23, docs/walk.md). `GuardBrake` turns that into
a decision: at each landing, brake if one more stride would carry that point past `line - margin`.

Used by the crate expert's approach (`visual.approach_stop=guard`) and by `backends/sonic/diag_brake.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
OBSTACLE_TOP = 0.37  # m: geoms whose bottom is below this count as the body that must stay behind a line


class FootWatch:
    def __init__(self, scene_xml: str):
        import mujoco

        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.qadr = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in MUJOCO_JOINT_NAMES]
        self.feet = [self.model.body(n).id for n in FEET]
        self.robot_geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] != 0 and self._is_robot(g)]
        self.stand_z = None
        self.planted = [True, True]
        self.landings = []
        self.last_strides = []

    def _is_robot(self, geom) -> bool:
        """Robot geoms hang off the pelvis; scene objects are other root children."""
        body = int(self.model.geom_bodyid[geom])
        while self.model.body_parentid[body] != 0:
            body = int(self.model.body_parentid[body])
        return self.model.body(body).name == "pelvis"

    def put(self, s) -> None:
        q = self.data.qpos
        q[:3] = s["pelvis"]["pos"]
        q[3:7] = s["pelvis"]["quat"]
        q[self.qadr] = s["joints"]
        self.mujoco.mj_kinematics(self.model, self.data)

    def feet_now(self, s):
        self.put(s)
        return [self.data.xpos[b].copy() for b in self.feet]

    def frontmost_low(self, s, origin, h, top=OBSTACLE_TOP) -> float:
        """Farthest point along h (from origin) of any robot geom reaching below `top` (world AABB)."""
        self.put(s)
        best = -np.inf
        for g in self.robot_geoms:
            rot = self.data.geom_xmat[g].reshape(3, 3)
            centre = self.data.geom_xpos[g] + rot @ self.model.geom_aabb[g, :3]
            half = np.abs(rot) @ self.model.geom_aabb[g, 3:]
            if centre[2] - half[2] < top:
                best = max(best, float((centre[:2] - origin) @ h + np.abs(h) @ half[:2]))
        return best

    def update(self, s, x0, h):
        """Returns (landed this tick, front planted ankle along h from x0, last stride or None)."""
        feet = self.feet_now(s)
        if self.stand_z is None:
            self.stand_z = [f[2] for f in feet]
        planted = [f[2] < z + 0.01 for f, z in zip(feet, self.stand_z)]
        landed = any(p and not was for p, was in zip(planted, self.planted))
        self.planted = planted
        along = [float((f[:2] - x0) @ h) for f in feet]
        if landed:
            self.landings.append(max(a for a, p in zip(along, planted) if p))
        front = max((a for a, p in zip(along, planted) if p), default=float("nan"))
        stride = self.landings[-1] - self.landings[-2] if len(self.landings) >= 2 else None
        self.last_strides = list(np.diff(self.landings[-3:]))  # the most recent (up to) two strides
        return landed, front, stride


@dataclass(frozen=True)
class GuardBrake:
    """Brake decision at a foot landing. `extra` covers the post-IDLE step (worst measured 20.3 cm at a 0.20
    final segment, 18.2 at a constant 0.25); `stride0` floors the predicted stride."""

    extra: float = 0.20
    margin: float = 0.03
    stride0: float = 0.14
    stride_mode: str = "floor"  # floor: max(last stride, stride0); recent2: max of the last two strides

    def should_brake(self, landed: bool, front: float, stride, last_strides, line: float) -> tuple[bool, float]:
        """(brake now, the stride used). `front` and `line` are distances along the same axis."""
        if self.stride_mode == "recent2":
            used = max(last_strides) if len(last_strides) >= 2 else self.stride0
        else:
            used = self.stride0 if stride is None else max(stride, self.stride0)
        return bool(landed and front + used + self.extra >= line - self.margin), float(used)
