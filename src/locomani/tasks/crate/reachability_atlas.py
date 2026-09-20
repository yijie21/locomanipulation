"""Conditional bimanual reachability slices for one measured standing state.

The atlas is an offline IK candidate map.  A candidate has passed the same
whole-body support geometry and joint bounds used by the crate expert plus a
static MuJoCo contact check.  It is not evidence that the dynamic controller
will reach the point.  Dynamic evidence is therefore stored separately and is
``untested`` unless an exact original-expert trial is attached by a caller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
from locomani.primitives.whole_body_ik import WholeBodyIK
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.carry_config import CrateCarryConfig
from locomani.tasks.crate.ghost_reach_execution import MeasuredCollisionChecker

TOOL_OFFSET_M = (0.1515, 0.0, 0.0)
DEFAULT_TILT_CANDIDATES_DEG = (30.0, 15.0, 45.0, 0.0)


@dataclass(frozen=True)
class ConditionalSlice:
    slice_id: str
    height_m: float
    separation_m: float
    yaw_deg: float

    def __post_init__(self) -> None:
        values = np.asarray((self.height_m, self.separation_m, self.yaw_deg), dtype=float)
        if not self.slice_id or not np.isfinite(values).all() or self.height_m <= 0 or self.separation_m <= 0:
            raise ValueError("slice needs a name, positive height/separation, and finite yaw")

    def to_dict(self) -> dict:
        return {
            "height_m": self.height_m,
            "separation_m": self.separation_m,
            "yaw_deg": self.yaw_deg,
        }


def default_slices() -> tuple[ConditionalSlice, ...]:
    """Seven one-factor slices about h=.65, d=.40, yaw=0."""
    return (
        ConditionalSlice("height_055", 0.55, 0.40, 0.0),
        ConditionalSlice("center", 0.65, 0.40, 0.0),
        ConditionalSlice("height_075", 0.75, 0.40, 0.0),
        ConditionalSlice("separation_030", 0.65, 0.30, 0.0),
        ConditionalSlice("separation_050", 0.65, 0.50, 0.0),
        ConditionalSlice("yaw_m15", 0.65, 0.40, -15.0),
        ConditionalSlice("yaw_p15", 0.65, 0.40, 15.0),
    )


@dataclass(frozen=True)
class AtlasConfig:
    x_bounds_m: tuple[float, float] = (0.15, 0.75)
    y_bounds_m: tuple[float, float] = (-0.24, 0.24)
    coarse_x_count: int = 7
    coarse_y_count: int = 5
    boundary_resolution_m: float = 0.025
    max_refinement_depth: int = 2
    sample_budget_per_slice: int = 160
    solver_iterations: int = 60
    max_attempts_per_sample: int = 5
    tilt_candidates_deg: tuple[float, ...] = DEFAULT_TILT_CANDIDATES_DEG
    hand_position_tolerance_m: float = 0.020
    hand_orientation_tolerance_deg: float = 15.0
    foot_position_tolerance_m: float = 0.005
    foot_orientation_tolerance_deg: float = 5.0
    com_shift_tolerance_m: float = 0.015
    waist_pitch_max_rad: float = 0.52

    def __post_init__(self) -> None:
        if not (self.x_bounds_m[0] < self.x_bounds_m[1] and self.y_bounds_m[0] < self.y_bounds_m[1]):
            raise ValueError("atlas x/y bounds must increase")
        if self.coarse_x_count < 2 or self.coarse_y_count < 2:
            raise ValueError("coarse grid needs at least two samples per axis")
        baseline_count = self.coarse_x_count * self.coarse_y_count + (self.coarse_x_count - 1) * (
            self.coarse_y_count - 1
        )
        if self.max_refinement_depth < 0 or self.sample_budget_per_slice < baseline_count:
            raise ValueError("invalid atlas depth or sample budget")
        if self.solver_iterations <= 0 or self.max_attempts_per_sample <= 0:
            raise ValueError("solver iterations and attempts must be positive")
        positive = (
            self.boundary_resolution_m,
            self.hand_position_tolerance_m,
            self.hand_orientation_tolerance_deg,
            self.foot_position_tolerance_m,
            self.foot_orientation_tolerance_deg,
            self.com_shift_tolerance_m,
            self.waist_pitch_max_rad,
        )
        if not np.isfinite(positive).all() or any(value <= 0 for value in positive):
            raise ValueError("atlas tolerances must be positive and finite")
        tilts = np.asarray(self.tilt_candidates_deg, dtype=float)
        if not len(tilts) or not np.isfinite(tilts).all() or np.any((tilts < 0) | (tilts > 45)):
            raise ValueError("tilt candidates must be finite degrees in [0, 45]")
        if len(set(map(float, tilts))) != len(tilts):
            raise ValueError("tilt candidates must be unique")

    def to_dict(self) -> dict:
        return asdict(self)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_standing_state(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(("joints", "pelvis")) - set(raw):
        raise ValueError("standing state must contain joints and pelvis")
    joints = np.asarray(raw["joints"], dtype=float)
    position = np.asarray(raw["pelvis"].get("pos"), dtype=float)
    quaternion = np.asarray(raw["pelvis"].get("quat"), dtype=float)
    if joints.shape != (29,) or position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("standing state must contain 29 joints, pelvis xyz, and pelvis wxyz quaternion")
    if not np.isfinite(np.r_[joints, position, quaternion]).all() or np.linalg.norm(quaternion) < 1e-9:
        raise ValueError("standing state values must be finite with a nonzero pelvis quaternion")
    result = copy.deepcopy(raw)
    result["joints"] = joints.tolist()
    result["pelvis"]["pos"] = position.tolist()
    result["pelvis"]["quat"] = (quaternion / np.linalg.norm(quaternion)).tolist()
    return result


def target_centres_robot(condition: ConditionalSlice, x_m: float, y_m: float) -> np.ndarray:
    yaw = math.radians(condition.yaw_deg)
    line = np.array((-math.sin(yaw), math.cos(yaw), 0.0))
    midpoint = np.array((x_m, y_m, condition.height_m))
    return np.array((midpoint + 0.5 * condition.separation_m * line, midpoint - 0.5 * condition.separation_m * line))


def _robot_frame(state: dict) -> tuple[np.ndarray, float, np.ndarray]:
    origin = np.asarray(state["pelvis"]["pos"], dtype=float).copy()
    origin[2] = 0.0
    pelvis_rotation = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]])
    yaw = float(state["pelvis"].get("yaw", pelvis_rotation.as_euler("xyz")[2]))
    rotation = Rotation.from_euler("z", yaw).as_matrix()
    return origin, yaw, rotation


def _targets_world(state: dict, condition: ConditionalSlice, x_m: float, y_m: float, tilt_deg: float):
    origin, yaw, rotation = _robot_frame(state)
    robot_targets = target_centres_robot(condition, x_m, y_m)
    world_targets = robot_targets @ rotation.T + origin
    target_yaw = yaw + math.radians(condition.yaw_deg)
    wrist_rotation = (
        Rotation.from_euler("z", target_yaw).as_matrix()
        @ Rotation.from_euler("y", 90.0 - tilt_deg, degrees=True).as_matrix()
    )
    return robot_targets, world_targets, np.asarray((wrist_rotation, wrist_rotation))


def _sample_id(condition: ConditionalSlice, x_m: float, y_m: float) -> str:
    return f"{condition.slice_id}:x{round(x_m * 1_000_000):+07d}:y{round(y_m * 1_000_000):+07d}"


def _point_key(x_m: float, y_m: float) -> tuple[int, int]:
    return round(x_m * 1_000_000_000), round(y_m * 1_000_000_000)


def _pose_vector(pose) -> np.ndarray:
    pitch = Rotation.from_matrix(pose.rotation).as_euler("xyz")[1]
    return np.r_[pose.position, pitch, pose.joints]


class ReachabilityAtlas:
    """Map one or more fixed h/separation/yaw slices for a measured stance."""

    def __init__(self, scene_xml: Path, standing_state: dict, config: AtlasConfig | None = None):
        self.scene_xml = Path(scene_xml).resolve()
        if not self.scene_xml.is_file():
            raise ValueError(f"scene XML does not exist: {self.scene_xml}")
        self.state = copy.deepcopy(standing_state)
        joints = np.asarray(self.state.get("joints"), dtype=float)
        if joints.shape != (29,):
            raise ValueError("standing state joints must contain all 29 model joints")
        self.config = config or AtlasConfig()
        self.ik = ArmIK(str(self.scene_xml))
        self.bimanual = BimanualIK(
            self.ik,
            waist_pitch_max=self.config.waist_pitch_max_rad,
            tool_offset=TOOL_OFFSET_M,
            waist_dofs="all",
        )
        carry = CrateCarryConfig(
            reach_control="whole_body",
            waist_dofs="all",
            waist_pitch_max=self.config.waist_pitch_max_rad,
        )
        self.body = WholeBodyIK(
            self.ik,
            self.state,
            carry,
            tool_offset=self.bimanual.offset,
            joint_limits=(self.bimanual.lo, self.bimanual.hi),
        )
        self.default_seed = self.body.seed.copy()
        self.standing_seed = self.body.initial.copy()
        self.collision_checker = MeasuredCollisionChecker(ArmIK(str(self.scene_xml)))
        self.identity = {
            "scene_xml": str(self.scene_xml),
            "scene_xml_sha256": _sha256(self.scene_xml),
            "standing_state_sha256": _canonical_sha256(_json_safe(self.state)),
            "config": self.config.to_dict(),
            "resolved_carry_config": asdict(carry),
            "solver": "WholeBodyIK+BimanualIK+MeasuredCollisionChecker",
            "atlas_implementation_sha256": _sha256(Path(__file__).resolve()),
        }
        self.identity["sha256"] = _canonical_sha256(_json_safe(self.identity))

    def _solution_seed(self, sample: dict) -> np.ndarray | None:
        selected = sample.get("selected_attempt")
        if not selected:
            return None
        solution = selected.get("solution")
        if not solution:
            return None
        position = np.asarray(solution["pelvis_position_m"], dtype=float)
        quaternion = np.asarray(solution["pelvis_quaternion_wxyz"], dtype=float)
        joints = np.asarray(solution["joints"], dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,) or joints.shape != (29,):
            return None
        pitch = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_euler("xyz")[1]
        return np.r_[position, pitch, joints]

    @staticmethod
    def _nearest_candidate(samples: Iterable[dict], x_m: float, y_m: float) -> dict | None:
        candidates = [sample for sample in samples if sample.get("ik_classification") == "candidate"]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda sample: (
                (sample["xy_m"][0] - x_m) ** 2 + (sample["xy_m"][1] - y_m) ** 2,
                sample["sample_id"],
            ),
        )

    def _attempt(
        self,
        condition: ConditionalSlice,
        x_m: float,
        y_m: float,
        tilt_deg: float,
        seed: np.ndarray,
        seed_kind: str,
    ) -> tuple[dict, np.ndarray]:
        robot_targets, world_targets, rotations = _targets_world(self.state, condition, x_m, y_m, tilt_deg)
        pose = self.body.solve(world_targets, rotations, seed=seed, iterations=self.config.solver_iterations)
        vector = _pose_vector(pose)
        self.body.put(vector)
        data = self.body.ik.data
        actual_rotations = data.xmat[self.body.hands].reshape(2, 3, 3).copy()
        orientation_errors = np.degrees(
            np.asarray(
                [
                    Rotation.from_matrix(target @ actual.T).magnitude()
                    for target, actual in zip(rotations, actual_rotations)
                ]
            )
        )
        foot_rotations = data.xmat[self.body.feet].reshape(2, 3, 3).copy()
        foot_orientation_errors = np.degrees(
            np.asarray(
                [
                    Rotation.from_matrix(actual @ initial.T).magnitude()
                    for actual, initial in zip(foot_rotations, self.body.feet_rot)
                ]
            )
        )
        quaternion = Rotation.from_matrix(pose.rotation).as_quat()[[3, 0, 1, 2]]
        allowed, rejected = self.collision_checker.check(
            {
                "pelvis": {"pos": pose.position, "quat": quaternion},
                "joints": pose.joints,
                "palms": {},
            }
        )
        joint_lo, joint_hi = self.body.lo[4:], self.body.hi[4:]
        joint_margins = np.minimum(pose.joints - joint_lo, joint_hi - pose.joints)
        checks = {
            "hand_position": bool(np.max(pose.hand_errors) < self.config.hand_position_tolerance_m),
            "hand_orientation": bool(np.max(orientation_errors) < self.config.hand_orientation_tolerance_deg),
            "feet_position": bool(np.max(pose.foot_errors) < self.config.foot_position_tolerance_m),
            "feet_orientation": bool(np.max(foot_orientation_errors) < self.config.foot_orientation_tolerance_deg),
            "com_support": bool(np.linalg.norm(pose.com_shift) < self.config.com_shift_tolerance_m),
            "joint_limits": bool(np.min(joint_margins) >= -1e-8),
            "self_collision_free": not rejected,
        }
        endpoint_admissible = all(checks.values())
        attempt = {
            "tilt_deg": float(tilt_deg),
            "seed_kind": seed_kind,
            "endpoint_admissible": endpoint_admissible,
            "solver_attempt_completed": True,
            "solver_status": _json_safe(self.body.last_diagnostics),
            "target_centres_world_m": world_targets,
            "target_tool_rotation": rotations[0],
            "residuals": {
                "hand_position_m": pose.hand_errors,
                "hand_orientation_deg": orientation_errors,
                "feet_position_m": pose.foot_errors,
                "feet_orientation_deg": foot_orientation_errors,
                "com_shift_m": pose.com_shift,
            },
            "constraints": {
                "checks": checks,
                "minimum_joint_limit_margin_rad": float(np.min(joint_margins)),
                "joint_limit_margin_rad": {
                    name: float(margin) for name, margin in zip(MUJOCO_JOINT_NAMES, joint_margins)
                },
                "allowed_contacts": allowed,
                "rejected_contacts": rejected,
            },
            "solution": {
                "joints": pose.joints,
                "pelvis_position_m": pose.position,
                "pelvis_quaternion_wxyz": quaternion,
            },
        }
        return _json_safe(attempt), vector

    def solve_sample(
        self,
        condition: ConditionalSlice,
        x_m: float,
        y_m: float,
        existing_samples: Iterable[dict] = (),
    ) -> dict:
        nearest = self._nearest_candidate(existing_samples, x_m, y_m)
        warm_seed = self._solution_seed(nearest) if nearest else None
        primary_seed = self.default_seed if warm_seed is None else warm_seed
        primary_kind = "default_crouch" if warm_seed is None else f"nearest_candidate:{nearest['sample_id']}"
        schedule: list[tuple[float, np.ndarray, str]] = [
            (float(tilt), primary_seed.copy(), primary_kind) for tilt in self.config.tilt_candidates_deg
        ]
        alternate_seed = self.default_seed if warm_seed is not None else self.standing_seed
        alternate_kind = "default_crouch" if warm_seed is not None else "measured_standing"
        if not np.allclose(alternate_seed, primary_seed):
            schedule.append((float(self.config.tilt_candidates_deg[0]), alternate_seed.copy(), alternate_kind))
        schedule = schedule[: self.config.max_attempts_per_sample]

        attempts: list[dict] = []
        selected = None
        for tilt, seed, seed_kind in schedule:
            attempt, _vector = self._attempt(condition, x_m, y_m, tilt, seed, seed_kind)
            attempts.append(attempt)
            if attempt["endpoint_admissible"]:
                selected = attempt
                break
        robot_targets = target_centres_robot(condition, x_m, y_m)
        return {
            "sample_id": _sample_id(condition, x_m, y_m),
            "xy_m": [float(x_m), float(y_m)],
            "target_centres_robot_m": robot_targets.tolist(),
            "ik_classification": "candidate" if selected else "solver_not_proven",
            "candidate_checks": selected["constraints"]["checks"] if selected else None,
            "selected_attempt": selected,
            "attempts": attempts,
            "untested_tilts_deg": [
                float(tilt)
                for tilt in self.config.tilt_candidates_deg
                if not any(math.isclose(float(tilt), attempt["tilt_deg"], abs_tol=1e-12) for attempt in attempts)
            ],
            "warm_start_provenance": primary_kind,
            "cache_provenance": "fresh",
            "dynamic_evidence": {"status": "untested"},
        }

    def map_slice(self, condition: ConditionalSlice, cached_slice: dict | None = None) -> dict:
        cached_by_key: dict[tuple[int, int], dict] = {}
        if cached_slice:
            if cached_slice.get("condition") != condition.to_dict():
                raise ValueError(f"cache condition mismatch for {condition.slice_id}")
            for raw in cached_slice.get("samples", []):
                copied = copy.deepcopy(raw)
                copied["cache_provenance"] = "cache"
                cached_by_key[_point_key(*copied["xy_m"])] = copied

        samples: list[dict] = []
        by_key: dict[tuple[int, int], dict] = {}

        def sample(x_m: float, y_m: float) -> dict | None:
            key = _point_key(x_m, y_m)
            if key in by_key:
                return by_key[key]
            if len(samples) >= self.config.sample_budget_per_slice:
                return None
            if key in cached_by_key:
                result = cached_by_key[key]
            else:
                result = self.solve_sample(condition, x_m, y_m, samples)
            by_key[key] = result
            samples.append(result)
            return result

        xs = np.linspace(*self.config.x_bounds_m, self.config.coarse_x_count)
        ys = np.linspace(*self.config.y_bounds_m, self.config.coarse_y_count)
        for y_m in ys:
            for x_m in xs:
                sample(float(x_m), float(y_m))

        initial_cells = [
            (float(x0), float(x1), float(y0), float(y1), 0)
            for y0, y1 in zip(ys[:-1], ys[1:])
            for x0, x1 in zip(xs[:-1], xs[1:])
        ]
        # Complete baseline coverage before spending any budget on a boundary.
        for x0, x1, y0, y1, _depth in initial_cells:
            sample(0.5 * (x0 + x1), 0.5 * (y0 + y1))

        leaves: list[dict] = []
        pending = list(initial_cells)
        cursor = 0
        while cursor < len(pending):
            x0, x1, y0, y1, depth = pending[cursor]
            cursor += 1
            xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            support = {
                "corners": [sample(x0, y0), sample(x1, y0), sample(x0, y1), sample(x1, y1)],
                "center": sample(xc, yc),
            }
            all_support = [*support["corners"], support["center"]]
            missing = any(value is None for value in all_support)
            classes = [] if missing else [value["ik_classification"] for value in all_support]
            mixed = bool(classes) and len(set(classes)) > 1
            width, height = x1 - x0, y1 - y0
            resolution_reached = (
                width <= self.config.boundary_resolution_m and height <= self.config.boundary_resolution_m
            )
            can_refine = (
                mixed
                and depth < self.config.max_refinement_depth
                and not resolution_reached
                and len(samples) < self.config.sample_budget_per_slice
            )
            if can_refine:
                pending.extend(
                    (
                        (x0, xc, y0, yc, depth + 1),
                        (xc, x1, y0, yc, depth + 1),
                        (x0, xc, yc, y1, depth + 1),
                        (xc, x1, yc, y1, depth + 1),
                    )
                )
                continue
            if missing:
                classification = "unresolved"
                reason = "sample_budget_exhausted"
            elif all(value == "candidate" for value in classes):
                classification = "candidate_interpolated"
                reason = None
            elif all(value == "solver_not_proven" for value in classes):
                classification = "solver_not_proven"
                reason = "all_support_samples_solver_not_proven"
            else:
                classification = "unresolved"
                if resolution_reached:
                    reason = "mixed_support_at_resolution"
                elif depth >= self.config.max_refinement_depth:
                    reason = "mixed_support_at_depth_limit"
                else:
                    reason = "sample_budget_exhausted"
            selected_tilts = sorted(
                {
                    float(value["selected_attempt"]["tilt_deg"])
                    for value in all_support
                    if value is not None and value.get("selected_attempt")
                }
            )
            leaves.append(
                {
                    "bounds_xy_m": {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1},
                    "support_sample_ids": {
                        "corners": [value["sample_id"] if value else None for value in support["corners"]],
                        "center": support["center"]["sample_id"] if support["center"] else None,
                    },
                    "classification": classification,
                    "refinement_depth": depth,
                    "uncertainty_reason": reason,
                    "selected_support_tilts_deg": selected_tilts,
                    "tilt_interpolation_status": (
                        "constant_at_support_samples"
                        if len(selected_tilts) == 1
                        else "varying_or_missing_support_tilts_unverified_between_samples"
                    ),
                }
            )

        candidate_count = sum(sample["ik_classification"] == "candidate" for sample in samples)
        classifications = {
            name: sum(cell["classification"] == name for cell in leaves)
            for name in (
                "candidate_interpolated",
                "unresolved",
                "solver_not_proven",
            )
        }
        edge_values = {
            "x_min": (0, self.config.x_bounds_m[0]),
            "x_max": (0, self.config.x_bounds_m[1]),
            "y_min": (1, self.config.y_bounds_m[0]),
            "y_max": (1, self.config.y_bounds_m[1]),
        }
        clipped_edges = {
            name: any(
                sample["ik_classification"] == "candidate" and math.isclose(sample["xy_m"][axis], value, abs_tol=1e-9)
                for sample in samples
            )
            for name, (axis, value) in edge_values.items()
        }
        return {
            "slice_id": condition.slice_id,
            "condition": condition.to_dict(),
            "samples": samples,
            "cells": leaves,
            "summary": {
                "sample_count": len(samples),
                "baseline_sample_count": self.config.coarse_x_count * self.config.coarse_y_count
                + (self.config.coarse_x_count - 1) * (self.config.coarse_y_count - 1),
                "baseline_coverage_complete": True,
                "ik_candidate_sample_count": candidate_count,
                "solver_not_proven_sample_count": len(samples) - candidate_count,
                "cell_classification_counts": classifications,
                "dynamic_success_count": sum(
                    sample["dynamic_evidence"].get("status") == "success" for sample in samples
                ),
                "dynamic_failure_count": sum(
                    sample["dynamic_evidence"].get("status") == "failure" for sample in samples
                ),
            },
            "clipped_edges": clipped_edges,
            "domain_expandable": True,
            "budget_exhausted": len(samples) >= self.config.sample_budget_per_slice,
        }

    def result(self, slices: Iterable[dict]) -> dict:
        slices = list(slices)
        result = {
            "schema_version": 1,
            "kind": "conditional_bimanual_reachability_atlas",
            "claim_scope": {
                "ik_candidates": "whole-body IK and static-contact candidates for sampled conditional x/y slices",
                "continuous_reliability": "not established",
                "five_dimensional_certification": "not claimed",
                "solver_failures": "solver_not_proven; not physical-impossibility evidence",
                "dynamic_successes": "only exact original-expert trials in dynamic_evidence",
                "tool_tilt": "fixed per endpoint solve; interpolation between differing selected tilts is unverified",
            },
            "identity": self.identity,
            "standing_state": _json_safe(self.state),
            "slices": slices,
        }
        result["representative_bank"] = representative_validation_bank(result)
        return result


def _representative_sets(mapped_slice: dict) -> tuple[set[str], set[str], tuple[float, float, float, float]]:
    by_id = {sample["sample_id"]: sample for sample in mapped_slice["samples"]}
    candidate_centres: set[str] = set()
    boundary_failures: set[str] = set()
    for cell in mapped_slice["cells"]:
        identifiers = [*cell["support_sample_ids"]["corners"], cell["support_sample_ids"]["center"]]
        support = [by_id[identifier] for identifier in identifiers if identifier in by_id]
        classes = {sample["ik_classification"] for sample in support}
        if cell["classification"] == "candidate_interpolated":
            candidate_centres.add(cell["support_sample_ids"]["center"])
        elif cell["classification"] == "unresolved" and classes == {"candidate", "solver_not_proven"}:
            boundary_failures.update(
                sample["sample_id"] for sample in support if sample["ik_classification"] == "solver_not_proven"
            )
    bounds = [cell["bounds_xy_m"] for cell in mapped_slice["cells"]]
    domain = (
        min(value["x_min"] for value in bounds),
        max(value["x_max"] for value in bounds),
        min(value["y_min"] for value in bounds),
        max(value["y_max"] for value in bounds),
    )
    return candidate_centres, boundary_failures, domain


def require_exact_slice(atlas: dict, *, height_m: float, separation_m: float, yaw_deg: float) -> dict:
    """Return an exact conditional slice; never substitute a nearby condition."""
    wanted = (float(height_m), float(separation_m), float(yaw_deg))
    matches = [
        mapped_slice
        for mapped_slice in atlas.get("slices", [])
        if np.allclose(
            (
                mapped_slice["condition"]["height_m"],
                mapped_slice["condition"]["separation_m"],
                mapped_slice["condition"]["yaw_deg"],
            ),
            wanted,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise ValueError("condition is not an exactly sampled atlas slice; run fresh IK")
    return matches[0]


def query_slice_region(mapped_slice: dict, x_m: float, y_m: float) -> dict:
    """Return the most refined leaf containing one x/y point."""
    if not np.isfinite((x_m, y_m)).all():
        raise ValueError("query point must be finite")
    containing = []
    for cell in mapped_slice.get("cells", []):
        bounds = cell["bounds_xy_m"]
        if bounds["x_min"] <= x_m <= bounds["x_max"] and bounds["y_min"] <= y_m <= bounds["y_max"]:
            containing.append(cell)
    if not containing:
        raise ValueError("query point is outside the sampled x/y domain")
    cell = min(
        containing,
        key=lambda value: (
            (value["bounds_xy_m"]["x_max"] - value["bounds_xy_m"]["x_min"])
            * (value["bounds_xy_m"]["y_max"] - value["bounds_xy_m"]["y_min"]),
            -value["refinement_depth"],
            value["bounds_xy_m"]["x_min"],
            value["bounds_xy_m"]["y_min"],
        ),
    )
    return {
        "slice_id": mapped_slice["slice_id"],
        "condition": copy.deepcopy(mapped_slice["condition"]),
        "query_xy_m": [float(x_m), float(y_m)],
        "classification": cell["classification"],
        "uncertainty_reason": cell["uncertainty_reason"],
        "cell": copy.deepcopy(cell),
    }


def _line_rectangle_interval(start: np.ndarray, delta: np.ndarray, bounds: dict) -> tuple[float, float] | None:
    lower, upper = 0.0, 1.0
    for origin, direction, minimum, maximum in (
        (start[0], delta[0], bounds["x_min"], bounds["x_max"]),
        (start[1], delta[1], bounds["y_min"], bounds["y_max"]),
    ):
        if abs(direction) < 1e-15:
            if origin < minimum or origin > maximum:
                return None
            continue
        first, second = (minimum - origin) / direction, (maximum - origin) / direction
        enter, leave = min(first, second), max(first, second)
        lower, upper = max(lower, enter), min(upper, leave)
        if lower > upper:
            return None
    return lower, upper


def candidate_intervals_along_line(
    mapped_slice: dict, start_xy: Iterable[float], end_xy: Iterable[float]
) -> list[dict]:
    """Intersect a line segment with candidate cells, preserving disjoint gaps."""
    start = np.asarray(tuple(start_xy), dtype=float)
    end = np.asarray(tuple(end_xy), dtype=float)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite(np.r_[start, end]).all():
        raise ValueError("line endpoints must be finite xy pairs")
    delta = end - start
    if np.linalg.norm(delta) < 1e-12:
        raise ValueError("line endpoints must be distinct")
    raw = []
    for cell in mapped_slice.get("cells", []):
        if cell["classification"] != "candidate_interpolated":
            continue
        interval = _line_rectangle_interval(start, delta, cell["bounds_xy_m"])
        if interval is not None and interval[1] - interval[0] > 1e-12:
            raw.append(interval)
    raw.sort()
    merged: list[list[float]] = []
    for lower, upper in raw:
        if not merged or lower > merged[-1][1] + 1e-12:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return [
        {
            "t_start": lower,
            "t_end": upper,
            "start_xy_m": (start + lower * delta).tolist(),
            "end_xy_m": (start + upper * delta).tolist(),
        }
        for lower, upper in merged
    ]


def representative_validation_bank(atlas: dict) -> dict:
    """Choose two distinct deterministic points from each of seven slices."""
    chosen: list[tuple[dict, dict, str]] = []
    for mapped_slice in atlas["slices"]:
        samples = mapped_slice["samples"]
        candidate_centres, boundary_failures, (x_min, x_max, y_min, y_max) = _representative_sets(mapped_slice)
        domain_centre = np.array((0.5 * (x_min + x_max), 0.5 * (y_min + y_max)))

        def margin(sample: dict) -> float:
            x_m, y_m = sample["xy_m"]
            return min(x_m - x_min, x_max - x_m, y_m - y_min, y_max - y_m)

        candidates = [sample for sample in samples if sample["ik_classification"] == "candidate"]
        if not candidates:
            raise ValueError(f"slice {mapped_slice['slice_id']} has no IK candidate for validation")
        first = min(
            candidates,
            key=lambda sample: (
                0 if sample["sample_id"] in candidate_centres and margin(sample) > 1e-9 else 1,
                -margin(sample),
                float(np.linalg.norm(np.asarray(sample["xy_m"]) - domain_centre)),
                sample["sample_id"],
            ),
        )
        first_role = (
            "interior" if first["sample_id"] in candidate_centres and margin(first) > 1e-9 else "sampled_candidate"
        )
        failures = [
            sample
            for sample in samples
            if sample["ik_classification"] == "solver_not_proven" and sample["xy_m"] != first["xy_m"]
        ]
        frontier = [sample for sample in failures if sample["sample_id"] in boundary_failures]
        fallback = failures or [sample for sample in candidates if sample["xy_m"] != first["xy_m"]]
        pool = frontier or fallback
        if not pool:
            raise ValueError(f"slice {mapped_slice['slice_id']} does not contain two distinct validation points")
        second = min(
            pool,
            key=lambda sample: (
                -sample["xy_m"][0],
                abs(sample["xy_m"][1]),
                sample["sample_id"],
            ),
        )
        if second["sample_id"] in boundary_failures:
            second_role = "boundary_ik_failed"
        elif second["ik_classification"] == "solver_not_proven":
            second_role = "ik_failed"
        else:
            second_role = "sampled_candidate"
        chosen.extend(((mapped_slice, first, first_role), (mapped_slice, second, second_role)))

    if len(chosen) != 14:
        raise ValueError("representative bank requires exactly seven slices")
    cases = []
    for index, (mapped_slice, sample, role) in enumerate(chosen, 1):
        condition = mapped_slice["condition"]
        x_m, y_m = sample["xy_m"]
        tilt = 30.0
        if sample.get("selected_attempt"):
            tilt = float(sample["selected_attempt"]["tilt_deg"])
        cases.append(
            {
                "index": index,
                "label_zh": f"{mapped_slice['slice_id']}·{role}",
                "parameters": {
                    "pair_midpoint_x_m": x_m,
                    "pair_midpoint_y_m": y_m,
                    "ground_height_z_m": condition["height_m"],
                    "pair_separation_m": condition["separation_m"],
                    "pair_line_yaw_deg": condition["yaw_deg"],
                },
                "capture_centres_xyz": {
                    "left": sample["target_centres_robot_m"][0],
                    "right": sample["target_centres_robot_m"][1],
                },
                "bodyward_tilt_deg": tilt,
                "orientation_endpoints": {"selected_bodyward_tilt_deg": tilt},
                "atlas_sample_id": sample["sample_id"],
                "atlas_ik_classification": sample["ik_classification"],
                "validation_role": role,
            }
        )
    return {
        "schema_version": 1,
        "seed": 0,
        "proposal_status": "IK-selected validation bank; dynamic reachability untested",
        "frame": {"origin": "pelvis ground projection", "+x": "robot forward", "+y": "robot left", "+z": "up"},
        "atlas_identity_sha256": atlas["identity"]["sha256"],
        "samples": cases,
    }


def attach_original_expert_evidence(atlas: dict, result_path: Path) -> None:
    """Attach exact point observations from a ghost original-expert result.

    This deliberately updates samples, never cells.  A successful point is not
    evidence for its surrounding interpolated IK cell.
    """
    result_path = Path(result_path).resolve()
    result = json.loads(result_path.read_text())
    request = result.get("request", {})
    source_identity = request.get("source_targets", {})
    source_path = Path(source_identity.get("path", ""))
    if not source_path.is_file() or _sha256(source_path) != source_identity.get("sha256"):
        raise ValueError(f"dynamic result source identity is unavailable or changed: {result_path}")
    source = json.loads(source_path.read_text())
    if source.get("atlas_identity_sha256") != atlas["identity"]["sha256"]:
        raise ValueError("dynamic result source was not produced by this atlas identity")
    source_by_index = {int(sample["index"]): sample for sample in source.get("samples", [])}
    atlas_samples = {
        sample["sample_id"]: sample for mapped_slice in atlas["slices"] for sample in mapped_slice["samples"]
    }
    offline_joints = np.asarray(atlas["standing_state"]["joints"], dtype=float)
    offline_pelvis = np.asarray(atlas["standing_state"]["pelvis"]["pos"], dtype=float)
    for case in result.get("cases", []):
        source_case = source_by_index.get(int(case.get("index", -1)))
        if source_case is None or "atlas_sample_id" not in source_case:
            raise ValueError("dynamic result case does not map to an atlas sample")
        sample = atlas_samples.get(source_case["atlas_sample_id"])
        if sample is None:
            raise ValueError(f"dynamic result references unknown sample {source_case['atlas_sample_id']}")
        expected = np.asarray(sample["target_centres_robot_m"], dtype=float)
        supplied = np.asarray(
            [source_case["capture_centres_xyz"]["left"], source_case["capture_centres_xyz"]["right"]], dtype=float
        )
        expected_tilt = float(sample["selected_attempt"]["tilt_deg"] if sample.get("selected_attempt") else 30.0)
        supplied_tilt = float(source_case.get("bodyward_tilt_deg", 30.0))
        if not np.allclose(expected, supplied, atol=1e-9) or not math.isclose(
            expected_tilt, supplied_tilt, abs_tol=1e-9
        ):
            raise ValueError("dynamic result target parameters or tilt do not match the atlas sample")

        traces = case.get("traces") or []
        initial = traces[0] if traces else None
        stance_delta = None
        if initial:
            measured_joints = np.asarray(initial.get("measured_joints"), dtype=float)
            measured_pelvis = np.asarray(initial.get("pelvis_position_m"), dtype=float)
            if measured_joints.shape != (29,) or measured_pelvis.shape != (3,):
                raise ValueError("dynamic evidence first trace lacks full measured stance")
            stance_delta = {
                "pelvis_position_m": (measured_pelvis - offline_pelvis).tolist(),
                "pelvis_height_m": float(measured_pelvis[2] - offline_pelvis[2]),
                "measured_joint_max_abs_rad": float(np.max(np.abs(measured_joints - offline_joints))),
                "first_trace_seq": initial.get("seq"),
                "first_trace_sim_time_s": initial.get("sim_time_s"),
            }
            feet = np.asarray(initial.get("feet_world_m", []), dtype=float)
            quaternion = np.asarray(initial.get("pelvis_quaternion_wxyz", []), dtype=float)
            if feet.shape == (2, 3) and quaternion.shape == (4,):
                yaw = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_euler("xyz")[2]
                inverse_yaw = Rotation.from_euler("z", -yaw).as_matrix()
                stance_delta["feet_in_measured_body_frame_m"] = ((feet - measured_pelvis) @ inverse_yaw.T).tolist()
        task_observed = bool(traces)
        if task_observed:
            evidence_status = "success" if case.get("status") == "success" else "failure"
        else:
            evidence_status = "not_run"
        observation = {
            "status": evidence_status,
            "controller": "original_expert_CoordinatedReach/WholeBodyIK",
            "case_status": case.get("status"),
            "reason": case.get("reason"),
            "result_path": str(result_path),
            "result_sha256": _sha256(result_path),
            "request_identity": request.get("request_identity"),
            "request_sha256": request.get("request_sha256"),
            "source_targets_sha256": source_identity.get("sha256"),
            "initial_stance_delta_from_offline_snapshot": stance_delta,
            "scope": "this reset and original-expert trial at this exact point only",
        }
        prior = sample.get("dynamic_evidence", {"status": "untested"})
        observations = [] if prior.get("status") == "untested" else list(prior.get("observations", [prior]))
        observations.append(observation)
        statuses = {value["status"] for value in observations}
        aggregate = "mixed" if len(statuses) > 1 else next(iter(statuses))
        sample["dynamic_evidence"] = {"status": aggregate, "observations": observations}

    for mapped_slice in atlas["slices"]:
        mapped_slice["summary"]["dynamic_success_count"] = sum(
            sample["dynamic_evidence"].get("status") == "success" for sample in mapped_slice["samples"]
        )
        mapped_slice["summary"]["dynamic_failure_count"] = sum(
            sample["dynamic_evidence"].get("status") == "failure" for sample in mapped_slice["samples"]
        )


def load_cache(path: Path | None, identity_sha256: str) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1 or raw.get("identity", {}).get("sha256") != identity_sha256:
        raise ValueError("atlas cache identity does not match scene, stance, and configuration")
    return {mapped["slice_id"]: mapped for mapped in raw.get("slices", [])}


__all__ = [
    "AtlasConfig",
    "attach_original_expert_evidence",
    "candidate_intervals_along_line",
    "ConditionalSlice",
    "ReachabilityAtlas",
    "default_slices",
    "load_cache",
    "load_standing_state",
    "query_slice_region",
    "representative_validation_bank",
    "require_exact_slice",
    "target_centres_robot",
]
