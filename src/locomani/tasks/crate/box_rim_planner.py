"""Static, collision-checked bimanual planner for rectangular open-box rims."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.whole_body_ik import WholeBodyIK
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.box_rim_collision import StaticCollisionChecker, SupportContactPolicy
from locomani.tasks.crate.box_rim_geometry import OpenBoxGeometry, RimPairCandidate, sample_opposite_rims


def _array(values, length: int, name: str) -> np.ndarray:
    value = np.array(values, dtype=float, copy=True)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {length} finite values")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class RobotState:
    joints: np.ndarray
    pelvis_position: np.ndarray
    pelvis_quaternion_wxyz: np.ndarray

    def __init__(self, joints, pelvis_position, pelvis_quaternion_wxyz):
        joints = _array(joints, 29, "robot_state.joints")
        position = _array(pelvis_position, 3, "robot_state.pelvis_position")
        quaternion = _array(pelvis_quaternion_wxyz, 4, "robot_state.pelvis_quaternion_wxyz")
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-6:
            raise ValueError("robot_state.pelvis_quaternion_wxyz must be unit length")
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "pelvis_position", position)
        object.__setattr__(self, "pelvis_quaternion_wxyz", quaternion)

    @property
    def rotation(self) -> np.ndarray:
        return Rotation.from_quat(self.pelvis_quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()

    def whole_body_state(self) -> dict:
        return {
            "joints": self.joints.copy(),
            "pelvis": {"pos": self.pelvis_position.copy(), "quat": self.pelvis_quaternion_wxyz.copy()},
        }


@dataclass(frozen=True)
class PlannerOptions:
    samples_per_rim: int = 5  # Number of evenly spaced centres on each opposite rim pair.
    side_offsets_m: tuple[float, ...] = (-0.025, 0.0, 0.025)  # Per-hand offsets along a rim, m.
    depths_m: tuple[float, ...] = (0.006, 0.012)  # Tool depths below the rim top, m.
    palm_pitch_deg: tuple[float, ...] = (35.0, 45.0, 55.0)  # Palm pitches about the wall normal, deg.
    contact_half_width_m: float = 0.012  # Half-width reserved around each sampled contact, m.
    top_k: int = 8  # Coarse IK candidates admitted to whole-body refinement.
    coarse_iterations: int = 160  # Maximum BimanualIK iterations per geometry candidate.
    whole_body_iterations: int = 140  # Maximum WholeBodyIK function evaluations per finalist.
    coarse_position_tolerance_m: float = 0.035  # Coarse per-hand tool error gate, m.
    coarse_orientation_tolerance_rad: float = 0.35  # Coarse per-hand rotation error gate, rad.
    hand_position_tolerance_m: float = 0.015  # Refined per-hand tool error gate, m.
    hand_orientation_tolerance_rad: float = 0.20  # Refined per-hand rotation error gate, rad.
    foot_anchor_tolerance_m: float = 0.005  # Refined foot-position drift gate, m.
    com_shift_tolerance_m: float = 0.015  # Robot COM shift from measured pose gate, m.
    min_joint_margin_rad: float = 0.002  # Minimum distance from a physical joint limit, rad.
    waist_pitch_max_rad: float = 0.52  # Maximum physical forward waist pitch, rad.
    waist_dofs: str = "pitch"  # Waist axes available to both IK stages: pitch or all.
    pelvis_height_min_m: float = 0.60  # Minimum refined pelvis height, m.
    pelvis_pitch_max_rad: float = 0.70  # Maximum refined pelvis pitch, rad.
    max_pelvis_xy_change_m: float = 0.10  # Pelvis motion over planted feet from measured state, m.
    max_pelvis_height_change_m: float = 0.16  # Pelvis-height change from measured state, m.
    capture_tolerance_m: float = 0.003  # Pad-to-wall straddling error allowed per side, m.
    target_contact_tolerance_m: float = 0.04  # Pad-contact distance from intended target, m.
    max_target_penetration_m: float = 0.0015  # Allowed pad-target penetration at ghost pose, m.
    support_baseline_max_penetration_m: float = 0.006  # Maximum measured loaded-foot compression, m.
    support_penetration_growth_tolerance_m: float = 0.0015  # Additional candidate compression, m.
    movement_weight: float = 1.0  # Ranking weight for joint/base movement.
    joint_limit_weight: float = 0.05  # Ranking weight for inverse joint-limit margin.
    com_midpoint_weight: float = 0.2  # Heuristic weight for grasp midpoint versus declared COM.

    def __post_init__(self):
        if self.samples_per_rim < 1 or self.top_k < 1 or self.coarse_iterations < 1 or self.whole_body_iterations < 1:
            raise ValueError("sample counts, top_k and iterations must be positive")
        if self.waist_dofs not in ("pitch", "all"):
            raise ValueError("waist_dofs must be pitch or all")
        sequences = (self.side_offsets_m, self.depths_m, self.palm_pitch_deg)
        if (
            any(not values for values in sequences)
            or not np.isfinite([v for values in sequences for v in values]).all()
        ):
            raise ValueError("offset, depth and pitch candidates must be nonempty and finite")
        positive = (
            self.coarse_position_tolerance_m,
            self.coarse_orientation_tolerance_rad,
            self.hand_position_tolerance_m,
            self.hand_orientation_tolerance_rad,
            self.foot_anchor_tolerance_m,
            self.com_shift_tolerance_m,
            self.waist_pitch_max_rad,
            self.pelvis_pitch_max_rad,
            self.max_pelvis_xy_change_m,
            self.max_pelvis_height_change_m,
            self.capture_tolerance_m,
            self.target_contact_tolerance_m,
            self.max_target_penetration_m,
            self.support_baseline_max_penetration_m,
            self.support_penetration_growth_tolerance_m,
        )
        nonnegative = (
            self.min_joint_margin_rad,
            self.contact_half_width_m,
            self.movement_weight,
            self.joint_limit_weight,
            self.com_midpoint_weight,
        )
        if not np.isfinite([*positive, *nonnegative, self.pelvis_height_min_m]).all():
            raise ValueError("planner numeric options must be finite")
        if any(value <= 0 for value in positive) or any(value < 0 for value in nonnegative):
            raise ValueError("planner tolerances must be positive and weights nonnegative")

    @property
    def waist_pitch_max(self):
        return self.waist_pitch_max_rad

    @property
    def pelvis_height_min(self):
        return self.pelvis_height_min_m

    @property
    def pelvis_pitch_max(self):
        return self.pelvis_pitch_max_rad


@dataclass(frozen=True)
class CandidateReport:
    identifier: str
    rim_pair: str
    stage: str
    static_valid: bool
    reasons: tuple[str, ...]
    targets_world_m: tuple
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BoxRimPlan:
    static_valid: bool
    reason: str
    selected_candidate: str | None
    joints: np.ndarray | None
    pelvis_position: np.ndarray | None
    pelvis_quaternion_wxyz: np.ndarray | None
    targets_world_m: np.ndarray | None
    selection: dict | None
    candidate_reports: tuple[CandidateReport, ...]
    diagnostics: dict

    def to_dict(self) -> dict:
        def values(array):
            return None if array is None else np.asarray(array).tolist()

        return {
            "static_valid": self.static_valid,
            "reason": self.reason,
            "selected_candidate": self.selected_candidate,
            "qpos": {
                "pelvis_position": values(self.pelvis_position),
                "pelvis_quaternion_wxyz": values(self.pelvis_quaternion_wxyz),
                "joints": values(self.joints),
            },
            "targets_world_m": values(self.targets_world_m),
            "selection": self.selection,
            "candidate_reports": [report.to_dict() for report in self.candidate_reports],
            "diagnostics": self.diagnostics,
            "claim_scope": "static pose only; no force-closure, loaded equilibrium, path, or dynamics certification",
        }


@dataclass
class _CoarseCandidate:
    geometry: RimPairCandidate
    targets: np.ndarray
    rotations: np.ndarray
    joints: np.ndarray
    diagnostics: dict
    score: float


class BoxRimPlanner:
    """Select a static pose from measured state; pelvis motion must preserve foot anchors."""

    def __init__(self, ik, target_body: str, box: OpenBoxGeometry, options: PlannerOptions | None = None):
        options = PlannerOptions() if options is None else options
        self.ik = ik
        self.box = box
        self.options = options
        self.bimanual = BimanualIK(
            ik,
            waist_pitch_max=options.waist_pitch_max_rad,
            waist_dofs=options.waist_dofs,
        )
        self.collision = StaticCollisionChecker(ik, target_body, box)
        self.joint_ranges = ik.model.jnt_range[ik.jids].copy()

    def plan(
        self,
        state: RobotState,
        *,
        deadline_monotonic: float | None = None,
        support_contact_policy: SupportContactPolicy = "bounded_growth",
    ) -> BoxRimPlan:
        if support_contact_policy not in ("bounded_growth", "diagnostic_floor"):
            raise ValueError("support_contact_policy must be 'bounded_growth' or 'diagnostic_floor'")
        geometries = sample_opposite_rims(
            self.box,
            samples_per_rim=self.options.samples_per_rim,
            side_offsets_m=self.options.side_offsets_m,
            depths_m=self.options.depths_m,
            palm_pitch_deg=self.options.palm_pitch_deg,
            contact_half_width_m=self.options.contact_half_width_m,
        )
        if not geometries:
            return self._failure("no_usable_opposite_rim_samples", (), 0)
        reports, coarse, deadline_expired = self._coarse_candidates(
            state, geometries, deadline_monotonic=deadline_monotonic
        )
        if deadline_expired:
            return self._failure("planning_deadline_exceeded", reports, len(geometries))
        if not coarse:
            return self._failure("no_coarse_ik_candidate", reports, len(geometries))
        coarse.sort(key=lambda item: (item.score, item.geometry.identifier))
        refine_ids = {item.geometry.identifier for item in coarse[: self.options.top_k]}
        for item in coarse[self.options.top_k :]:
            reports.append(self._report(item, "coarse", False, ("not_refined_top_k",), item.diagnostics))
        refined = []
        for item in coarse:
            if self._deadline_expired(deadline_monotonic):
                return self._failure("planning_deadline_exceeded", reports, len(geometries))
            if item.geometry.identifier not in refine_ids:
                continue
            report, solution = self._refine(state, item, support_contact_policy)
            reports.append(report)
            if self._deadline_expired(deadline_monotonic):
                return self._failure("planning_deadline_exceeded", reports, len(geometries))
            if solution is not None:
                refined.append(solution)
        if not refined:
            return self._failure("no_static_supported_collision_free_candidate", reports, len(geometries))
        selected = min(refined, key=lambda item: (item[0], item[1].geometry.identifier))
        return self._success(selected, reports, len(geometries))

    @staticmethod
    def _deadline_expired(deadline_monotonic) -> bool:
        return deadline_monotonic is not None and time.monotonic() >= deadline_monotonic

    def _coarse_candidates(self, state, geometries, *, deadline_monotonic=None):
        reports, accepted = [], []
        for geometry in geometries:
            if self._deadline_expired(deadline_monotonic):
                return reports, accepted, True
            assignment = self._assign_and_orient(state, geometry)
            if assignment is None:
                reports.append(self._geometry_rejection(geometry, "ambiguous_hand_assignment"))
                continue
            targets, rotations = assignment
            positions_local = (targets - state.pelvis_position) @ state.rotation
            rotations_local = np.array([state.rotation.T @ rotation for rotation in rotations])
            solved = self.bimanual.solve(
                state.joints,
                positions_local,
                rotations_local,
                iterations=self.options.coarse_iterations,
                posture=state.joints,
                posture_weight=0.002,
            )
            actual_positions, actual_rotations = self.bimanual.tool_poses(solved.joints)
            orientation_errors = self._rotation_errors(actual_rotations, rotations_local)
            diagnostics = {
                "position_errors_m": solved.errors.tolist(),
                "orientation_errors_rad": orientation_errors.tolist(),
                "joint_motion_l2_rad": float(np.linalg.norm(solved.joints - state.joints)),
            }
            reasons = []
            if max(solved.errors) > self.options.coarse_position_tolerance_m:
                reasons.append("coarse_position_error")
            if max(orientation_errors) > self.options.coarse_orientation_tolerance_rad:
                reasons.append("coarse_orientation_error")
            item = _CoarseCandidate(geometry, targets, rotations, solved.joints, diagnostics, 0.0)
            if reasons:
                reports.append(self._report(item, "coarse", False, tuple(reasons), diagnostics))
                continue
            item.score = diagnostics["joint_motion_l2_rad"]
            accepted.append(item)
        return reports, accepted, False

    def _assign_and_orient(self, state, geometry):
        points = np.array(geometry.points_world, copy=True)
        lateral = state.rotation[:, 1]
        projections = points @ lateral
        if abs(projections[0] - projections[1]) < 1e-5:
            return None
        order = np.argsort(projections)[::-1]
        points = points[order]
        up = self.box.pose.rotation[:, 2]
        wall_normal = self.box.pose.rotation[:, 0 if geometry.rim_pair == "x" else 1]
        midpoint_direction = points.mean(axis=0) - state.pelvis_position
        forward = np.cross(wall_normal, up)
        if float(forward @ midpoint_direction) < 0:
            wall_normal = -wall_normal
            forward = -forward
        if float(forward @ midpoint_direction) <= 1e-8:
            return None
        base = np.column_stack((forward, wall_normal, up))
        rotation = base @ Rotation.from_euler("y", geometry.pitch_deg, degrees=True).as_matrix()
        return points, np.repeat(rotation[None], 2, axis=0)

    @staticmethod
    def _rotation_errors(actual, target):
        return np.array([Rotation.from_matrix(wanted @ found.T).magnitude() for found, wanted in zip(actual, target)])

    def _refine(self, state, item, support_contact_policy: SupportContactPolicy):
        whole = WholeBodyIK(
            self.ik,
            state.whole_body_state(),
            self.options,
            tool_offset=self.bimanual.offset,
            joint_limits=(self.bimanual.lo, self.bimanual.hi),
        )
        seed = whole.initial.copy()
        seed[4:] = item.joints
        lo, hi = whole.lo.copy(), whole.hi.copy()
        shift = self.options.max_pelvis_xy_change_m
        lo[:2] = np.maximum(lo[:2], state.pelvis_position[:2] - shift)
        hi[:2] = np.minimum(hi[:2], state.pelvis_position[:2] + shift)
        lo[2] = max(lo[2], state.pelvis_position[2] - self.options.max_pelvis_height_change_m)
        hi[2] = min(hi[2], state.pelvis_position[2] + self.options.max_pelvis_height_change_m)
        pose = whole.solve(
            item.targets,
            item.rotations,
            seed=seed,
            bounds=(lo, hi),
            posture=state.joints,
            iterations=self.options.whole_body_iterations,
        )
        actual_rotations = np.array([self.ik.data.xmat[body].reshape(3, 3).copy() for body in whole.hands])
        orientation_errors = self._rotation_errors(actual_rotations, item.rotations)
        margins = np.minimum(pose.joints - self.joint_ranges[:, 0], self.joint_ranges[:, 1] - pose.joints)
        quaternion = Rotation.from_matrix(pose.rotation).as_quat()[[3, 0, 1, 2]]
        allowed, rejected = self.collision.check(
            state,
            pose.joints,
            pose.position,
            quaternion,
            self.box,
            item.targets,
            target_contact_tolerance_m=self.options.target_contact_tolerance_m,
            max_target_penetration_m=min(self.options.max_target_penetration_m, self.box.wall_thickness_m / 4),
            support_baseline_max_penetration_m=self.options.support_baseline_max_penetration_m,
            support_penetration_growth_tolerance_m=self.options.support_penetration_growth_tolerance_m,
            support_contact_policy=support_contact_policy,
        )
        capture = self.collision.capture_diagnostics(
            item.targets,
            item.rotations,
            self.box.wall_thickness_m,
            self.options.capture_tolerance_m,
        )
        reasons = self._support_reasons(pose, orientation_errors, margins, capture, rejected)
        diagnostics = {
            **item.diagnostics,
            "refined_hand_position_errors_m": pose.hand_errors.tolist(),
            "refined_hand_orientation_errors_rad": orientation_errors.tolist(),
            "foot_anchor_errors_m": pose.foot_errors.tolist(),
            "robot_com_shift_m": pose.com_shift.tolist(),
            "minimum_joint_margin_rad": float(margins.min()),
            "capture": capture,
            "pelvis_translation_from_measured_m": (pose.position - state.pelvis_position).tolist(),
            "allowed_contacts": [contact.to_dict() for contact in allowed],
            "rejected_contacts": [contact.to_dict() for contact in rejected],
        }
        report = self._report(item, "refined", not reasons, reasons, diagnostics)
        if reasons:
            return report, None
        score = self._score(state, item, pose, margins)
        diagnostics["ranking_score"] = score
        return report, (score, item, pose, quaternion, diagnostics)

    def _support_reasons(self, pose, rotation_errors, margins, capture, rejected):
        reasons = []
        if max(pose.hand_errors) > self.options.hand_position_tolerance_m:
            reasons.append("hand_position_error")
        if max(rotation_errors) > self.options.hand_orientation_tolerance_rad:
            reasons.append("hand_orientation_error")
        if max(pose.foot_errors) > self.options.foot_anchor_tolerance_m:
            reasons.append("foot_anchor_error")
        if np.linalg.norm(pose.com_shift) > self.options.com_shift_tolerance_m:
            reasons.append("robot_com_shift")
        if min(margins) < self.options.min_joint_margin_rad:
            reasons.append("joint_limit_margin")
        if not capture["capture_valid"]:
            reasons.append("pad_wall_straddling")
        if rejected:
            reasons.append("collision")
        return tuple(reasons)

    def _score(self, state, item, pose, margins):
        movement = np.linalg.norm(pose.joints - state.joints) + np.linalg.norm(pose.position - state.pelvis_position)
        limit = float(np.mean(1.0 / np.maximum(margins, 1e-4)))
        midpoint = np.linalg.norm(item.targets.mean(axis=0) - self.box.com_world)
        return float(
            self.options.movement_weight * movement
            + self.options.joint_limit_weight * limit
            + self.options.com_midpoint_weight * midpoint
        )

    def _geometry_rejection(self, geometry, reason):
        return CandidateReport(
            geometry.identifier,
            geometry.rim_pair,
            "geometry",
            False,
            (reason,),
            tuple(tuple(float(v) for v in point) for point in geometry.points_world),
        )

    def _report(self, item, stage, valid, reasons, diagnostics):
        return CandidateReport(
            item.geometry.identifier,
            item.geometry.rim_pair,
            stage,
            valid,
            tuple(reasons),
            tuple(tuple(float(v) for v in point) for point in item.targets),
            diagnostics,
        )

    def _failure(self, reason, reports, generated):
        return BoxRimPlan(
            False,
            reason,
            None,
            None,
            None,
            None,
            None,
            None,
            tuple(reports),
            {"geometry_candidates": generated, "collider": self.collision.collider_diagnostics},
        )

    def _success(self, selected, reports, generated):
        _, item, pose, quaternion, diagnostics = selected
        local = (item.targets - self.box.pose.position) @ self.box.pose.rotation
        axis = 1 if item.geometry.rim_pair == "x" else 0
        normal_axis = 0 if item.geometry.rim_pair == "x" else 1
        rim_names = [item.geometry.rim_pair + ("+" if point[normal_axis] > 0 else "-") for point in local]
        target_quaternions = [
            Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist() for rotation in item.rotations
        ]
        selection = {
            "rim_pair": item.geometry.rim_pair,
            "hand_rims": {"left": rim_names[0], "right": rim_names[1]},
            "sample_center_coordinate_m": item.geometry.coordinate_m,
            "hand_along_coordinates_m": {"left": float(local[0, axis]), "right": float(local[1, axis])},
            "hand_offsets_m": {
                "left": float(local[0, axis] - item.geometry.coordinate_m),
                "right": float(local[1, axis] - item.geometry.coordinate_m),
            },
            "depth_m": item.geometry.depth_m,
            "palm_pitch_deg": item.geometry.pitch_deg,
            "target_rotations_world": item.rotations.tolist(),
            "target_quaternions_wxyz": target_quaternions,
        }
        return BoxRimPlan(
            True,
            "static_pose_found",
            item.geometry.identifier,
            pose.joints.copy(),
            pose.position.copy(),
            quaternion.copy(),
            item.targets.copy(),
            selection,
            tuple(reports),
            {"geometry_candidates": generated, "collider": self.collision.collider_diagnostics, **diagnostics},
        )
