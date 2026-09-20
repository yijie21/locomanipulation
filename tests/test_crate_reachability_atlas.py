from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from locomani.tasks.crate import reachability_atlas as atlas
from locomani.tasks.crate.ghost_reach_execution import (
    DEFAULT_BODYWARD_TILT_DEG,
    GhostReachCase,
    original_reach_control_config,
    world_target,
)

FIXTURE_SCENE = Path(__file__).parent / "fixtures" / "geometry" / "scene_locomani.xml"


def _standing_state():
    joints = np.zeros(29)
    joints[[0, 6]] = -0.1
    joints[[3, 9]] = 0.2
    joints[[4, 10]] = -0.1
    joints[15:22] = [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
    joints[22:29] = [-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
    return {"joints": joints.tolist(), "pelvis": {"pos": [-0.69, 0.0, 0.78], "quat": [1.0, 0.0, 0.0, 0.0]}}


def _small_real_config(**overrides):
    values = {
        "coarse_x_count": 2,
        "coarse_y_count": 2,
        "sample_budget_per_slice": 5,
        "max_refinement_depth": 0,
        "solver_iterations": 60,
        "max_attempts_per_sample": 1,
    }
    values.update(overrides)
    return atlas.AtlasConfig(**values)


def test_real_fixture_endpoint_records_actual_support_orientation_and_whole_body_limits():
    mapper = atlas.ReachabilityAtlas(FIXTURE_SCENE, _standing_state(), _small_real_config())
    sample = mapper.solve_sample(atlas.ConditionalSlice("fixture", 0.65, 0.30, 0.0), 0.40, 0.0)

    assert sample["ik_classification"] == "candidate"
    selected = sample["selected_attempt"]
    assert selected["tilt_deg"] == 30.0
    assert selected["endpoint_admissible"]
    assert selected["constraints"]["checks"]["feet_orientation"]
    assert max(selected["residuals"]["feet_orientation_deg"]) < 5.0
    assert len(selected["constraints"]["joint_limit_margin_rad"]) == 29
    assert selected["constraints"]["rejected_contacts"] == []


def test_real_fixture_environment_collision_is_solver_not_proven_not_impossible():
    mapper = atlas.ReachabilityAtlas(FIXTURE_SCENE, _standing_state(), _small_real_config())
    sample = mapper.solve_sample(atlas.ConditionalSlice("fixture", 0.445, 0.38, 0.0), 0.34, 0.02)

    assert sample["ik_classification"] == "solver_not_proven"
    assert sample["selected_attempt"] is None
    attempt = sample["attempts"][0]
    assert attempt["constraints"]["checks"]["hand_position"]
    assert not attempt["constraints"]["checks"]["self_collision_free"]
    assert attempt["constraints"]["rejected_contacts"]


class _SyntheticMapper:
    def __init__(self):
        self.config = atlas.AtlasConfig(
            x_bounds_m=(0.0, 1.0),
            y_bounds_m=(0.0, 1.0),
            coarse_x_count=3,
            coarse_y_count=3,
            sample_budget_per_slice=13,
            max_refinement_depth=2,
        )

    def solve_sample(self, condition, x_m, y_m, _samples):
        classification = "candidate" if x_m <= 0.5 else "solver_not_proven"
        return {
            "sample_id": f"{condition.slice_id}:{x_m:.3f}:{y_m:.3f}",
            "xy_m": [x_m, y_m],
            "target_centres_robot_m": atlas.target_centres_robot(condition, x_m, y_m).tolist(),
            "ik_classification": classification,
            "candidate_checks": {} if classification == "candidate" else None,
            "selected_attempt": {"tilt_deg": 30.0} if classification == "candidate" else None,
            "attempts": [],
            "warm_start_provenance": "synthetic",
            "cache_provenance": "fresh",
            "dynamic_evidence": {"status": "untested"},
        }


def test_budget_cannot_starve_coarse_nodes_or_coarse_cell_centres():
    mapper = _SyntheticMapper()
    condition = atlas.ConditionalSlice("synthetic", 0.65, 0.4, 0.0)
    mapped = atlas.ReachabilityAtlas.map_slice(mapper, condition)

    assert mapped["summary"]["sample_count"] == 13
    assert mapped["summary"]["baseline_sample_count"] == 13
    assert mapped["summary"]["baseline_coverage_complete"]
    assert {tuple(sample["xy_m"]) for sample in mapped["samples"]} == {
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    }
    assert len(mapped["cells"]) == 4
    assert all(
        None not in [*cell["support_sample_ids"]["corners"], cell["support_sample_ids"]["center"]]
        for cell in mapped["cells"]
    )


def _query_slice():
    def cell(x0, x1, classification):
        return {
            "bounds_xy_m": {"x_min": x0, "x_max": x1, "y_min": 0.0, "y_max": 1.0},
            "support_sample_ids": {"corners": [], "center": None},
            "classification": classification,
            "refinement_depth": 0,
            "uncertainty_reason": None if classification == "candidate_interpolated" else "gap",
        }

    return {
        "slice_id": "line",
        "condition": {"height_m": 0.65, "separation_m": 0.4, "yaw_deg": 0.0},
        "cells": [
            cell(0.0, 0.3, "candidate_interpolated"),
            cell(0.3, 0.6, "unresolved"),
            cell(0.6, 1.0, "candidate_interpolated"),
        ],
    }


def test_line_query_keeps_candidate_holes_disjoint_and_region_query_reports_uncertainty():
    mapped = _query_slice()
    intervals = atlas.candidate_intervals_along_line(mapped, (0.0, 0.5), (1.0, 0.5))

    assert [(value["t_start"], value["t_end"]) for value in intervals] == [(0.0, 0.3), (0.6, 1.0)]
    query = atlas.query_slice_region(mapped, 0.45, 0.5)
    assert query["classification"] == "unresolved"
    assert query["uncertainty_reason"] == "gap"


def test_condition_lookup_requires_exact_slice_instead_of_nearest_substitution():
    mapped = _query_slice()
    result = {"slices": [mapped]}

    assert atlas.require_exact_slice(result, height_m=0.65, separation_m=0.4, yaw_deg=0.0) is mapped
    with pytest.raises(ValueError, match="exactly sampled"):
        atlas.require_exact_slice(result, height_m=0.66, separation_m=0.4, yaw_deg=0.0)


def test_ghost_case_tilt_is_optional_and_fixed_for_the_whole_case():
    raw = {
        "index": 1,
        "label_zh": "fixture",
        "parameters": {
            "pair_midpoint_x_m": 0.4,
            "pair_midpoint_y_m": 0.0,
            "ground_height_z_m": 0.65,
            "pair_separation_m": 0.4,
            "pair_line_yaw_deg": 0.0,
        },
        "capture_centres_xyz": {"left": [0.4, 0.2, 0.65], "right": [0.4, -0.2, 0.65]},
        "orientation_endpoints": {},
    }
    assert GhostReachCase.from_source(raw).bodyward_tilt_deg == DEFAULT_BODYWARD_TILT_DEG

    raw["bodyward_tilt_deg"] = 15.0
    case = GhostReachCase.from_source(raw)
    target = world_target(case, _standing_state())
    expected = Rotation.from_euler("y", 75.0, degrees=True).as_matrix()
    assert np.allclose(target.permitted_tool_rotation, expected)
    assert original_reach_control_config(15.0).visual.palm_pitch_deg == 75.0
    with pytest.raises(ValueError, match=r"\[0, 45\]"):
        original_reach_control_config(46.0)


def test_dynamic_success_attaches_to_exact_point_with_reset_stance_delta_not_cell(tmp_path):
    standing = _standing_state()
    sample = {
        "sample_id": "center:x+400000:y+000000",
        "target_centres_robot_m": [[0.4, 0.2, 0.65], [0.4, -0.2, 0.65]],
        "selected_attempt": {"tilt_deg": 30.0},
        "dynamic_evidence": {"status": "untested"},
    }
    document = {
        "identity": {"sha256": "atlas-sha"},
        "standing_state": standing,
        "slices": [
            {
                "samples": [sample],
                "cells": [{"classification": "candidate_interpolated"}],
                "summary": {"dynamic_success_count": 0, "dynamic_failure_count": 0},
            }
        ],
    }
    source = {
        "atlas_identity_sha256": "atlas-sha",
        "samples": [
            {
                "index": 1,
                "atlas_sample_id": sample["sample_id"],
                "capture_centres_xyz": {
                    "left": sample["target_centres_robot_m"][0],
                    "right": sample["target_centres_robot_m"][1],
                },
                "bodyward_tilt_deg": 30.0,
            }
        ],
    }
    source_path = tmp_path / "bank.json"
    source_path.write_text(json.dumps(source))
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    measured_joints = np.asarray(standing["joints"]) + 0.02
    result = {
        "request": {
            "request_identity": "exact-bank-run",
            "request_sha256": "request-sha",
            "source_targets": {"path": str(source_path), "sha256": source_hash},
        },
        "cases": [
            {
                "index": 1,
                "status": "success",
                "reason": "measured_reach_held",
                "traces": [
                    {
                        "seq": 10,
                        "sim_time_s": 0.0,
                        "measured_joints": measured_joints.tolist(),
                        "pelvis_position_m": [-0.69, 0.0, 0.79],
                        "pelvis_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "feet_world_m": [[-0.69, 0.1, 0.0], [-0.69, -0.1, 0.0]],
                    }
                ],
            }
        ],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))

    atlas.attach_original_expert_evidence(document, result_path)

    evidence = sample["dynamic_evidence"]
    assert evidence["status"] == "success"
    delta = evidence["observations"][0]["initial_stance_delta_from_offline_snapshot"]
    assert delta["pelvis_height_m"] == pytest.approx(0.01)
    assert delta["measured_joint_max_abs_rad"] == pytest.approx(0.02)
    assert document["slices"][0]["cells"][0]["classification"] == "candidate_interpolated"


def test_validation_bank_avoids_clipped_near_edge_and_chooses_forward_failed_boundary():
    slices = []
    for number in range(7):
        condition = atlas.ConditionalSlice(f"slice_{number}", 0.55 + number * 0.01, 0.4, 0.0)

        def sample(name, x_m, y_m, classification):
            return {
                "sample_id": f"{condition.slice_id}:{name}",
                "xy_m": [x_m, y_m],
                "target_centres_robot_m": atlas.target_centres_robot(condition, x_m, y_m).tolist(),
                "ik_classification": classification,
                "selected_attempt": {"tilt_deg": 30.0} if classification == "candidate" else None,
            }

        edge = sample("edge", 0.15, 0.0, "candidate")
        interior = sample("interior", 0.45, 0.0, "candidate")
        rear_failure = sample("rear_failure", 0.30, 0.0, "solver_not_proven")
        forward_failure = sample("forward_failure", 0.70, 0.04, "solver_not_proven")
        cells = [
            {
                "bounds_xy_m": {"x_min": 0.15, "x_max": 0.75, "y_min": -0.24, "y_max": 0.24},
                "support_sample_ids": {
                    "corners": [edge["sample_id"]] * 4,
                    "center": interior["sample_id"],
                },
                "classification": "candidate_interpolated",
            },
            {
                "bounds_xy_m": {"x_min": 0.45, "x_max": 0.75, "y_min": -0.12, "y_max": 0.12},
                "support_sample_ids": {
                    "corners": [
                        interior["sample_id"],
                        rear_failure["sample_id"],
                        forward_failure["sample_id"],
                        edge["sample_id"],
                    ],
                    "center": forward_failure["sample_id"],
                },
                "classification": "unresolved",
            },
        ]
        slices.append(
            {
                "slice_id": condition.slice_id,
                "condition": condition.to_dict(),
                "samples": [edge, interior, rear_failure, forward_failure],
                "cells": cells,
            }
        )
    document = {"identity": {"sha256": "atlas-sha"}, "slices": slices}

    bank = atlas.representative_validation_bank(document)

    for first, second in zip(bank["samples"][::2], bank["samples"][1::2]):
        assert first["validation_role"] == "interior"
        assert first["parameters"]["pair_midpoint_x_m"] == 0.45
        assert second["validation_role"] == "boundary_ik_failed"
        assert second["parameters"]["pair_midpoint_x_m"] == 0.70
        assert second["atlas_ik_classification"] == "solver_not_proven"
