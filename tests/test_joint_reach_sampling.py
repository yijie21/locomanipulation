from __future__ import annotations

import json

import numpy as np
import pytest

from locomani.tasks.crate.ghost_reach_execution import load_source_cases
from locomani.tasks.crate.joint_reach_sampling import DIMENSIONS, classify_reach_attempt, generate_joint_bank


def _bounds():
    return {
        "x_m": [0.15, 0.75],
        "y_m": [-0.24, 0.24],
        "height_m": [0.50, 0.80],
        "separation_m": [0.25, 0.55],
        "yaw_deg": [-30.0, 30.0],
    }


def test_sobol_bank_is_deterministic_joint_5d_and_source_compatible(tmp_path):
    config = {
        "schema_version": 1,
        "method": "sobol",
        "seed": 20260920,
        "bounds": _bounds(),
        "count": 16,
        "bodyward_tilt_deg": 30,
        "anchors": [[0.45, 0.0, 0.65, 0.40, 0.0]],
        "anchor_grid": {
            "x_m": [0.25, 0.65],
            "y_m": [-0.12, 0.12],
            "height_m": [0.701],
            "separation_m": [0.44, 0.45],
            "yaw_deg": [-15.0, 15.0],
        },
    }

    first = generate_joint_bank(config)
    second = generate_joint_bank(config)

    assert first == second
    assert first["generation"]["sample_count"] == 33
    assert first["generation"]["dimensions"] == list(DIMENSIONS)
    sobol = np.asarray([[sample["joint_coordinates"][name] for name in DIMENSIONS] for sample in first["samples"][:16]])
    lower = np.asarray([_bounds()[name][0] for name in DIMENSIONS])
    upper = np.asarray([_bounds()[name][1] for name in DIMENSIONS])
    normalized_span = np.ptp((sobol - lower) / (upper - lower), axis=0)
    assert np.all(normalized_span > 0.75)
    assert all(len(np.unique(sobol[:, dimension])) == 16 for dimension in range(5))

    path = tmp_path / "bank.json"
    path.write_text(json.dumps(first))
    cases, identity = load_source_cases(path)
    assert len(cases) == 33
    assert [case.index for case in cases] == list(range(1, 34))
    assert identity["sample_count"] == 33
    assert all(case.bodyward_tilt_deg == 30.0 for case in cases)


def test_boundary_refinement_uses_only_executed_mixed_neighbors():
    config = {
        "schema_version": 1,
        "method": "boundary_refinement",
        "seed": 17,
        "bounds": _bounds(),
        "count": 8,
        "neighbor_count": 2,
        "min_normalized_distance": 0.005,
    }
    observations = {
        "schema_version": 1,
        "observations": [
            {
                "sample_id": 1,
                "coordinates": [0.30, -0.10, 0.60, 0.35, -10.0],
                "status": "success",
                "reason": "measured_reach_held",
                "execution_started": True,
                "reach_commanded": True,
            },
            {
                "sample_id": 2,
                "coordinates": [0.50, 0.10, 0.70, 0.45, 10.0],
                "status": "failed",
                "reason": "target_timeout",
                "execution_started": True,
                "reach_commanded": True,
            },
            {
                "sample_id": 3,
                "coordinates": [0.35, 0.0, 0.65, 0.40, 0.0],
                "status": "failed",
                "reason": "initial_stance_unstable",
                "execution_started": False,
                "reach_commanded": False,
            },
            {
                "sample_id": 4,
                "coordinates": [0.42, 0.03, 0.63, 0.39, 3.0],
                "status": "failed",
                "reason": "runtime_error",
                "execution_started": True,
                "reach_commanded": True,
            },
        ],
    }

    first = generate_joint_bank(config, observations=observations)
    second = generate_joint_bank(config, observations=observations)

    assert first == second
    assert len(first["samples"]) == 8
    first_point = first["samples"][0]
    np.testing.assert_allclose(
        [first_point["joint_coordinates"][name] for name in DIMENSIONS],
        [0.40, 0.0, 0.65, 0.40, 0.0],
        atol=1e-14,
    )
    assert all(sample["sampling_provenance"]["success_sample_id"] == 1 for sample in first["samples"])
    assert all(sample["sampling_provenance"]["failure_sample_id"] == 2 for sample in first["samples"])
    assert first["generation"]["claim_scope"].endswith("no unsampled cell is classified")


def test_boundary_refinement_does_not_treat_initialization_failure_as_negative_evidence():
    config = {
        "schema_version": 1,
        "method": "boundary_refinement",
        "seed": 1,
        "bounds": _bounds(),
        "count": 1,
    }
    observations = {
        "schema_version": 1,
        "observations": [
            {
                "sample_id": 1,
                "coordinates": [0.30, 0.0, 0.60, 0.35, 0.0],
                "status": "success",
                "execution_started": True,
                "reach_commanded": True,
            },
            {
                "sample_id": 2,
                "coordinates": [0.50, 0.0, 0.70, 0.45, 0.0],
                "status": "initialization_failed",
                "execution_started": False,
                "reach_commanded": False,
            },
        ],
    }

    with pytest.raises(ValueError, match="executed success and failure"):
        generate_joint_bank(config, observations=observations)


def test_boundary_refinement_excludes_explicit_dispute_and_accepts_legacy_rows():
    config = {
        "schema_version": 1,
        "method": "boundary_refinement",
        "seed": 1,
        "bounds": _bounds(),
        "count": 1,
        "neighbor_count": 2,
    }
    observations = {
        "schema_version": 1,
        "observations": [
            {
                "sample_id": "native-disputed-success",
                "coordinates": [0.20, 0.0, 0.60, 0.35, 0.0],
                "status": "success",
                "reach_commanded": True,
                "refinement_eligible": False,
                "refinement_evidence": "excluded_native_runtime_disagreement",
            },
            {
                "sample_id": "legacy-success",
                "coordinates": [0.30, 0.0, 0.60, 0.35, 0.0],
                "status": "success",
                "reach_commanded": True,
            },
            {
                "sample_id": "legacy-failure",
                "coordinates": [0.50, 0.0, 0.70, 0.45, 0.0],
                "status": "failed",
                "reason": "target_timeout",
                "reach_commanded": True,
            },
        ],
    }

    bank = generate_joint_bank(config, observations=observations)

    provenance = bank["samples"][0]["sampling_provenance"]
    assert provenance["success_sample_id"] == "legacy-success"
    assert provenance["failure_sample_id"] == "legacy-failure"


def test_attempt_classifier_rejects_observe_only_controller_admission():
    rejected = classify_reach_attempt(
        {
            "status": "failed",
            "reason": "controller_unsupported_assisted_reach",
            "execution_started": True,
            "max_measured_tool_displacement_m": 0.018,
            "traces": [
                {
                    "seq": 2307,
                    "sim_time_s": 1.56,
                    "controller_phase": "observe",
                    "controller_reason": "unsupported_assisted_reach",
                }
            ],
        }
    )
    attempted = classify_reach_attempt(
        {
            "status": "failed",
            "reason": "target_timeout",
            "execution_started": True,
            "traces": [{"seq": 42, "sim_time_s": 0.1, "controller_phase": "reach", "controller_reason": ""}],
        }
    )

    assert rejected["executor_invoked"] is True
    assert rejected["reach_commanded"] is False
    assert rejected["attempt_classification"] == "controller_admission_rejection"
    assert attempted["reach_commanded"] is True
    assert attempted["attempt_classification"] == "executed_reach"
    assert attempted["first_reach_seq"] == 42
