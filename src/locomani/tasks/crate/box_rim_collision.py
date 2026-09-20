"""MuJoCo collider validation and static contact classification for rim plans."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
from locomani.tasks.crate.box_rim_geometry import OpenBoxGeometry

SupportContactPolicy = Literal["bounded_growth", "diagnostic_floor"]


@dataclass(frozen=True)
class ContactDiagnostic:
    classification: str
    geom_1: str
    geom_2: str
    body_1: str
    body_2: str
    distance_m: float
    baseline_distance_m: float | None = None
    support_penetration_growth_m: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class StaticCollisionChecker:
    """Check one matched primitive target and active robot/environment colliders."""

    def __init__(self, ik, target_body: str, box: OpenBoxGeometry, *, geometry_tolerance_m=0.002):
        import mujoco

        self.mj = mujoco
        self.model = copy.deepcopy(ik.model)
        self.data = mujoco.MjData(self.model)
        self.target_body_name = target_body
        self.target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        if self.target_body < 0:
            raise ValueError(f"target body {target_body!r} is not in the scene")
        target_joint = int(self.model.body_jntadr[self.target_body])
        if target_joint < 0 or self.model.jnt_type[target_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("target body must own a free joint")
        self.target_qadr = int(self.model.jnt_qposadr[target_joint])
        self.base_qadr = int(
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")]
        )
        self.joint_qadr = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in MUJOCO_JOINT_NAMES
            ]
        )
        self.gripper_qadr = self._gripper_addresses()
        pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot_root = int(self.model.body_rootid[pelvis])
        self.robot_bodies = {
            body for body in range(self.model.nbody) if int(self.model.body_rootid[body]) == robot_root
        }
        self.target_bodies = self._descendants(self.target_body)
        self.target_geoms = {
            geom for geom in range(self.model.ngeom) if int(self.model.geom_bodyid[geom]) in self.target_bodies
        }
        self.floor_geoms = self._structural_floor_geoms()
        self.finger_geoms = {
            geom
            for geom in range(self.model.ngeom)
            if self._body_name(int(self.model.geom_bodyid[geom])).startswith(
                ("left_dex1_finger_", "right_dex1_finger_")
            )
            and (self.model.geom_contype[geom] != 0 or self.model.geom_conaffinity[geom] != 0)
        }
        self.side_finger_geoms = {
            side: tuple(
                sorted(
                    geom
                    for geom in self.finger_geoms
                    if self._body_name(int(self.model.geom_bodyid[geom])).startswith(f"{side}_dex1_finger_")
                )
            )
            for side in ("left", "right")
        }
        self.collider_diagnostics = self._validate_primitive_box(box, geometry_tolerance_m)
        self.gripper_q = self._gripper_q_for_wall(box.wall_thickness_m)
        self.collider_diagnostics["derived_gripper_q"] = self.gripper_q

    def _descendants(self, root: int) -> set[int]:
        result = {root}
        for body in range(1, self.model.nbody):
            parent = int(self.model.body_parentid[body])
            if parent in result:
                result.add(body)
        return result

    def _body_name(self, body: int) -> str:
        return self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, body) or ""

    def _geom_name(self, geom: int) -> str:
        return self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, geom) or f"geom_{geom}"

    def _structural_floor_geoms(self) -> frozenset[int]:
        """Return the lowest active horizontal planes attached to the world body."""
        candidates = []
        for geom in range(self.model.ngeom):
            active = self.model.geom_contype[geom] != 0 or self.model.geom_conaffinity[geom] != 0
            if (
                not active
                or int(self.model.geom_bodyid[geom]) != 0
                or self.model.geom_type[geom] != self.mj.mjtGeom.mjGEOM_PLANE
            ):
                continue
            quaternion = np.asarray(self.model.geom_quat[geom], dtype=float)
            w, x, y, z = quaternion
            normal = np.array(
                [
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ]
            )
            if float(normal @ np.array([0.0, 0.0, 1.0])) >= 1.0 - 1e-8:
                candidates.append((float(self.model.geom_pos[geom, 2]), geom))
        if not candidates:
            return frozenset()
        lowest = min(height for height, _ in candidates)
        return frozenset(geom for height, geom in candidates if abs(height - lowest) <= 1e-8)

    def _gripper_addresses(self) -> np.ndarray:
        values = []
        for side in ("left", "right"):
            for finger in (1, 2):
                joint = self.mj.mj_name2id(
                    self.model, self.mj.mjtObj.mjOBJ_JOINT, f"{side}_hand_gripper_{finger}_joint"
                )
                if joint >= 0:
                    values.append(int(self.model.jnt_qposadr[joint]))
        return np.asarray(values, dtype=int)

    def _validate_primitive_box(self, box: OpenBoxGeometry, tolerance: float) -> dict:
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("geometry_tolerance_m must be positive and finite")
        active_boxes = []
        active_target = []
        for geom in sorted(self.target_geoms):
            active = self.model.geom_contype[geom] != 0 or self.model.geom_conaffinity[geom] != 0
            if active:
                active_target.append(geom)
            if int(self.model.geom_bodyid[geom]) != self.target_body:
                continue
            if active and self.model.geom_type[geom] == self.mj.mjtGeom.mjGEOM_BOX:
                quaternion = self.model.geom_quat[geom]
                if not np.allclose(np.abs(quaternion[0]), 1.0, atol=1e-8) or not np.allclose(
                    quaternion[1:], 0.0, atol=1e-8
                ):
                    raise ValueError("target open-box colliders must be axis-aligned in the target body")
                active_boxes.append((self.model.geom_pos[geom].copy(), self.model.geom_size[geom].copy()))
        if len(active_target) != 5 or len(active_boxes) != 5:
            raise ValueError("target must have exactly five active primitive box colliders and no lid")
        sx, sy, sz = box.size_m
        t = box.wall_thickness_m
        expected = [
            ([0, 0, t / 2], [sx / 2, sy / 2, t / 2]),
            ([sx / 2 - t / 2, 0, sz / 2], [t / 2, sy / 2, sz / 2]),
            ([-sx / 2 + t / 2, 0, sz / 2], [t / 2, sy / 2, sz / 2]),
            ([0, sy / 2 - t / 2, sz / 2], [sx / 2, t / 2, sz / 2]),
            ([0, -sy / 2 + t / 2, sz / 2], [sx / 2, t / 2, sz / 2]),
        ]
        remaining = list(active_boxes)
        for position, size in expected:
            match = next(
                (
                    index
                    for index, actual in enumerate(remaining)
                    if np.allclose(actual[0], position, atol=tolerance, rtol=0)
                    and np.allclose(actual[1], size, atol=tolerance, rtol=0)
                ),
                None,
            )
            if match is None:
                raise ValueError("target active colliders do not match the declared open-box geometry")
            remaining.pop(match)
        return {
            "mode": "matched_target_primitive",
            "target_body": self.target_body_name,
            "matched_box_geoms": 5,
            "coverage": "active MuJoCo colliders only",
        }

    def _gripper_q_for_wall(self, thickness_m: float) -> float:
        left = []
        for finger in (1, 2):
            body = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, f"left_dex1_finger_{finger}_link")
            geoms = [geom for geom in self.finger_geoms if int(self.model.geom_bodyid[geom]) == body]
            if len(geoms) != 1 or self.model.geom_type[geoms[0]] != self.mj.mjtGeom.mjGEOM_MESH:
                raise ValueError("wall aperture derivation requires one active mesh pad per Dex1 finger")
            left.append(geoms[0])
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.gripper_qadr] = 0.0
        self.mj.mj_forward(self.model, self.data)
        palm = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        axis = self.data.xmat[palm].reshape(3, 3)[:, 1]
        intervals = []
        for geom in left:
            mesh = int(self.model.geom_dataid[geom])
            start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
            vertices = self.model.mesh_vert[start : start + count]
            world = vertices @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]
            projection = world @ axis
            intervals.append((float(projection.min()), float(projection.max())))
        intervals.sort()
        zero_gap = intervals[1][0] - intervals[0][1]
        q = float((thickness_m - zero_gap) / 2)
        limits = []
        for address in self.gripper_qadr:
            joint = int(np.flatnonzero(self.model.jnt_qposadr == address)[0])
            limits.append(self.model.jnt_range[joint])
        lo = max(float(limit[0]) for limit in limits)
        hi = min(float(limit[1]) for limit in limits)
        if q < lo or q > hi:
            raise ValueError("box wall thickness is outside the loaded gripper aperture capability")
        return q

    def _place(self, state, joints, position, quaternion_wxyz, box: OpenBoxGeometry, gripper_q):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_qadr : self.base_qadr + 3] = position
        self.data.qpos[self.base_qadr + 3 : self.base_qadr + 7] = quaternion_wxyz
        self.data.qpos[self.joint_qadr] = joints
        if len(self.gripper_qadr):
            self.data.qpos[self.gripper_qadr] = gripper_q
        self.data.qpos[self.target_qadr : self.target_qadr + 3] = box.pose.position
        self.data.qpos[self.target_qadr + 3 : self.target_qadr + 7] = box.pose.quaternion_wxyz
        self.mj.mj_forward(self.model, self.data)

    def _vertices_world(self, geom: int) -> np.ndarray:
        mesh = int(self.model.geom_dataid[geom])
        start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
        vertices = self.model.mesh_vert[start : start + count]
        return vertices @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]

    def capture_diagnostics(self, targets_world, rotations_world, thickness_m, tolerance_m):
        hands = []
        for side, target, rotation in zip(("left", "right"), targets_world, rotations_world):
            projections = []
            for geom in self.side_finger_geoms[side]:
                delta = self._vertices_world(geom) - target
                projections.append(delta @ rotation)
            projections.sort(key=lambda values: float(values[:, 1].mean()))
            if len(projections) != 2:
                return {"capture_valid": False, "reason": "missing_finger_pad_geometry", "hands": hands}
            lower, upper = projections
            normal_errors = [
                abs(float(lower[:, 1].max()) + thickness_m / 2),
                abs(float(upper[:, 1].min()) - thickness_m / 2),
            ]
            tangent_contains = [
                bool(
                    values[:, 0].min() - tolerance_m <= 0 <= values[:, 0].max() + tolerance_m
                    and values[:, 2].min() - tolerance_m <= 0 <= values[:, 2].max() + tolerance_m
                )
                for values in projections
            ]
            hands.append(
                {
                    "side": side,
                    "pad_wall_normal_errors_m": normal_errors,
                    "target_inside_pad_tangent_bounds": tangent_contains,
                }
            )
        valid = all(
            max(hand["pad_wall_normal_errors_m"]) <= tolerance_m and all(hand["target_inside_pad_tangent_bounds"])
            for hand in hands
        )
        return {"capture_valid": valid, "reason": None if valid else "pad_wall_straddling", "hands": hands}

    def _intended_wall_geoms(self, targets_world, box):
        inverse = box.pose.rotation.T
        result = []
        active = [
            geom
            for geom in self.target_geoms
            if self.model.geom_contype[geom] != 0 or self.model.geom_conaffinity[geom] != 0
        ]
        for target in targets_world:
            local = inverse @ (target - box.pose.position)
            distances = [np.linalg.norm((local - self.model.geom_pos[geom])[:2]) for geom in active]
            result.append(active[int(np.argmin(distances))])
        return result

    def _is_foot_support(self, robot_body: int, normal_on_robot) -> bool:
        name = self._body_name(robot_body)
        return (
            name.startswith(("left_foot", "right_foot", "left_ankle", "right_ankle"))
            and float(np.asarray(normal_on_robot, dtype=float) @ np.array([0.0, 0.0, 1.0])) >= 0.8
        )

    def _is_floor_support(self, robot_body: int, other_geom: int, normal_on_robot) -> bool:
        return other_geom in self.floor_geoms and self._is_foot_support(robot_body, normal_on_robot)

    def _baseline_support_contacts(self, state, box, gripper_q) -> dict[tuple[int, int], float]:
        """Measured loaded support keyed by the exact robot/environment geom pair."""
        self._place(
            state,
            state.joints,
            state.pelvis_position,
            state.pelvis_quaternion_wxyz,
            box,
            gripper_q,
        )
        baseline: dict[tuple[int, int], float] = {}
        for contact in self.data.contact[: self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 == r2:
                continue
            robot_geom = g1 if r1 else g2
            other_geom = g2 if r1 else g1
            if other_geom in self.target_geoms:
                continue
            robot_body = int(self.model.geom_bodyid[robot_geom])
            normal_on_robot = contact.frame[:3] if r2 else -contact.frame[:3]
            if not self._is_foot_support(robot_body, normal_on_robot):
                continue
            key = (robot_geom, other_geom)
            baseline[key] = min(baseline.get(key, float("inf")), float(contact.dist))
        return baseline

    def check(
        self,
        state,
        joints,
        position,
        quaternion_wxyz,
        box,
        targets_world,
        *,
        target_contact_tolerance_m=0.04,
        max_target_penetration_m=0.005,
        support_baseline_max_penetration_m=0.006,
        support_penetration_growth_tolerance_m=0.0015,
        gripper_q=None,
        support_contact_policy: SupportContactPolicy = "bounded_growth",
    ):
        if support_contact_policy not in ("bounded_growth", "diagnostic_floor"):
            raise ValueError("support_contact_policy must be 'bounded_growth' or 'diagnostic_floor'")
        if (
            not np.isfinite([support_baseline_max_penetration_m, support_penetration_growth_tolerance_m]).all()
            or support_baseline_max_penetration_m <= 0
            or support_penetration_growth_tolerance_m < 0
        ):
            raise ValueError("support penetration limits must be finite with positive baseline and nonnegative growth")
        if gripper_q is None:
            gripper_q = self.gripper_q
        else:
            gripper_q = np.asarray(gripper_q, dtype=float)
            if gripper_q.shape != self.gripper_qadr.shape or not np.isfinite(gripper_q).all():
                raise ValueError(f"gripper_q must contain {len(self.gripper_qadr)} finite values")
        baseline_support = self._baseline_support_contacts(state, box, gripper_q)
        self._place(state, joints, position, quaternion_wxyz, box, gripper_q)
        intended_walls = self._intended_wall_geoms(targets_world, box)
        allowed, rejected = [], []
        for contact in self.data.contact[: self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            t1, t2 = g1 in self.target_geoms, g2 in self.target_geoms
            if not (r1 or r2):
                continue
            robot_geom = g1 if r1 else g2
            other_geom = g2 if r1 else g1
            other_target = t2 if r1 else t1
            robot_body = int(self.model.geom_bodyid[robot_geom])
            support_normal_on_robot = contact.frame[:3] if r2 else -contact.frame[:3]
            hand_index = 0 if self._body_name(robot_body).startswith("left_") else 1
            near_target = np.linalg.norm(contact.pos - targets_world[hand_index])
            intended_target_contact = (
                other_target
                and robot_geom in self.finger_geoms
                and other_geom == intended_walls[hand_index]
                and near_target <= target_contact_tolerance_m
                and contact.dist >= -max_target_penetration_m
            )
            if intended_target_contact:
                classification = "allowed_finger_pad_target_contact"
                destination = allowed
            elif r1 and r2:
                classification = "robot_self_collision"
                destination = rejected
            elif other_target:
                classification = "non_pad_target_collision"
                destination = rejected
            elif support_contact_policy == "diagnostic_floor" and self._is_floor_support(
                robot_body, other_geom, support_normal_on_robot
            ):
                classification = "diagnostic_floor_support_contact"
                destination = allowed
            elif support_contact_policy == "bounded_growth" and (
                self._is_foot_support(robot_body, support_normal_on_robot)
                and (
                    contact.dist >= -0.002
                    or (
                        (robot_geom, other_geom) in baseline_support
                        and baseline_support[(robot_geom, other_geom)] >= -support_baseline_max_penetration_m
                        and contact.dist
                        >= baseline_support[(robot_geom, other_geom)] - support_penetration_growth_tolerance_m
                    )
                )
            ):
                classification = "allowed_foot_support_contact"
                destination = allowed
            else:
                classification = "robot_environment_collision"
                destination = rejected
            baseline_distance = baseline_support.get((robot_geom, other_geom))
            destination.append(
                ContactDiagnostic(
                    classification,
                    self._geom_name(g1),
                    self._geom_name(g2),
                    self._body_name(b1),
                    self._body_name(b2),
                    float(contact.dist),
                    baseline_distance_m=baseline_distance,
                    support_penetration_growth_m=(
                        max(0.0, baseline_distance - float(contact.dist)) if baseline_distance is not None else None
                    ),
                )
            )
        return tuple(allowed), tuple(rejected)
