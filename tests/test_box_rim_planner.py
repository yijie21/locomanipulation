import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest
from geometry_fixture import DIRECTORY, open_box_scene
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK
from locomani.tasks.crate.box_rim_cli import load_request, run
from locomani.tasks.crate.box_rim_geometry import OpenBoxGeometry, RigidPose, sample_opposite_rims
from locomani.tasks.crate.box_rim_planner import BoxRimPlanner, PlannerOptions, RobotState


def box(pose=None, thickness=0.01):
    return OpenBoxGeometry(
        [0.32, 0.28, 0.18],
        thickness,
        RigidPose([0, 0, 0.62], [1, 0, 0, 0]) if pose is None else pose,
    )


def options(**changes):
    values = dict(
        samples_per_rim=5,
        side_offsets_m=(0.0,),
        depths_m=(0.006,),
        palm_pitch_deg=(45.0,),
        top_k=4,
        coarse_iterations=350,
        whole_body_iterations=160,
        min_joint_margin_rad=0.0,
    )
    values.update(changes)
    return PlannerOptions(**values)


def state(x):
    return RobotState(np.zeros(29), [x, 0, 0.793], [1, 0, 0, 0])


def planner(scene, geometry, **option_changes):
    return BoxRimPlanner(
        ArmIK(str(scene)),
        "water_bottle_object",
        geometry,
        options(**option_changes),
    )


def test_loaded_foot_support_uses_bounded_pair_matched_baseline():
    geometry = box()
    with open_box_scene(geometry) as scene:
        checker = planner(scene, geometry).collision
        targets = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        # Measured depth is about 2.56 mm.  The same ankle/floor geom pairs may
        # compress one additional millimetre, despite exceeding the legacy 2 mm.
        measured = RobotState(np.zeros(29), [-0.6, 0.0, 0.7893], [1, 0, 0, 0])
        allowed, rejected = checker.check(
            measured,
            measured.joints,
            [-0.6, 0.0, 0.7883],
            [1, 0, 0, 0],
            geometry,
            targets,
        )
        assert not rejected
        support = [item for item in allowed if item.classification == "allowed_foot_support_contact"]
        assert support
        assert all(item.baseline_distance_m is not None for item in support)
        assert max(item.support_penetration_growth_m for item in support) == pytest.approx(0.001)

        # Extra sinking beyond the growth budget stays a collision.
        _, growth_rejected = checker.check(
            measured,
            measured.joints,
            [-0.6, 0.0, 0.7860],
            [1, 0, 0, 0],
            geometry,
            targets,
        )
        assert any(item.support_penetration_growth_m > 0.003 for item in growth_rejected)

        # A deeply sunk baseline and a candidate whose pair had no measured
        # contact cannot legitimize penetration.
        deep = RobotState(np.zeros(29), [-0.6, 0.0, 0.7850], [1, 0, 0, 0])
        _, deep_rejected = checker.check(
            deep, deep.joints, deep.pelvis_position, deep.pelvis_quaternion_wxyz, geometry, targets
        )
        assert deep_rejected
        unsupported = RobotState(np.zeros(29), [-0.6, 0.0, 0.7930], [1, 0, 0, 0])
        _, unsupported_rejected = checker.check(
            unsupported,
            unsupported.joints,
            [-0.6, 0.0, 0.7890],
            [1, 0, 0, 0],
            geometry,
            targets,
        )
        assert unsupported_rejected

        robot_geom, _ = next(iter(checker._baseline_support_contacts(measured, geometry, checker.gripper_q)))
        support_body = int(checker.model.geom_bodyid[robot_geom])
        assert checker._is_foot_support(support_body, [0, 0, 1])
        assert not checker._is_foot_support(support_body, [1, 0, 0])
        assert not checker._is_foot_support(checker.target_body, [0, 0, 1])


def test_expired_planning_deadline_returns_typed_refusal_before_search():
    geometry = box()
    with open_box_scene(geometry) as scene:
        result = planner(scene, geometry).plan(state(-0.6), deadline_monotonic=time.monotonic() - 1.0)
    assert not result.static_valid
    assert result.reason == "planning_deadline_exceeded"
    assert result.candidate_reports == ()


def test_real_robot_fixture_selects_different_rim_points_from_stopped_stance():
    geometry = box()
    with open_box_scene(geometry) as scene:
        farther = planner(scene, geometry).plan(state(-0.6))
        nearer = planner(scene, geometry).plan(state(-0.5))
    assert farther.static_valid and nearer.static_valid
    assert farther.selection["rim_pair"] == nearer.selection["rim_pair"] == "y"
    assert farther.selection["sample_center_coordinate_m"] != nearer.selection["sample_center_coordinate_m"]
    assert farther.selection["hand_rims"] == {"left": "y+", "right": "y-"}


def test_static_valid_pose_passes_capture_joint_foot_com_and_collision_gates():
    geometry = box()
    source_joints = np.zeros(29)
    measured = RobotState(source_joints, [-0.6, 0, 0.793], [1, 0, 0, 0])
    before = measured.joints.copy()
    with open_box_scene(geometry) as scene:
        result = planner(scene, geometry, min_joint_margin_rad=0.002).plan(measured)
    assert result.static_valid
    selected = next(report for report in result.candidate_reports if report.identifier == result.selected_candidate)
    assert selected.diagnostics["capture"]["capture_valid"]
    assert selected.diagnostics["rejected_contacts"] == []
    assert max(selected.diagnostics["foot_anchor_errors_m"]) <= 0.005
    assert np.linalg.norm(selected.diagnostics["robot_com_shift_m"]) <= 0.015
    assert selected.diagnostics["minimum_joint_margin_rad"] >= 0.002
    np.testing.assert_array_equal(measured.joints, before)
    np.testing.assert_array_equal(source_joints, np.zeros(29))


def test_collision_check_uses_optional_measured_open_gripper_configuration():
    geometry = box()
    measured = state(-0.6)
    with open_box_scene(geometry) as scene:
        rim_planner = planner(scene, geometry, min_joint_margin_rad=0.0)
        plan = rim_planner.plan(measured)
        assert plan.static_valid
        closed_contacts, _ = rim_planner.collision.check(
            measured,
            plan.joints,
            plan.pelvis_position,
            plan.pelvis_quaternion_wxyz,
            geometry,
            plan.targets_world_m,
        )
        open_contacts, open_rejected = rim_planner.collision.check(
            measured,
            plan.joints,
            plan.pelvis_position,
            plan.pelvis_quaternion_wxyz,
            geometry,
            plan.targets_world_m,
            gripper_q=np.full(4, 0.025),
        )

    assert any(item.classification == "allowed_finger_pad_target_contact" for item in closed_contacts)
    assert not any(item.classification == "allowed_finger_pad_target_contact" for item in open_contacts)
    assert not open_rejected


def test_far_stance_and_colliding_top_rank_are_honest_refusals():
    geometry = box()
    with open_box_scene(geometry) as scene:
        unreachable = planner(scene, geometry).plan(state(-1.1))
        colliding = planner(scene, geometry, top_k=1).plan(state(-0.6))
    assert not unreachable.static_valid
    assert unreachable.reason == "no_coarse_ik_candidate"
    assert not colliding.static_valid
    assert colliding.reason == "no_static_supported_collision_free_candidate"
    refined = [report for report in colliding.candidate_reports if report.stage == "refined"]
    assert refined and refined[0].reasons == ("collision",)
    assert refined[0].diagnostics["capture"]["capture_valid"]


def test_forearm_obstacle_is_reported_as_environment_collision():
    geometry = box()
    obstacle = ([-0.34, 0.14, 0.91], [0.05, 0.05, 0.05])
    with open_box_scene(geometry, obstacle=obstacle) as scene:
        result = planner(scene, geometry).plan(state(-0.6))
    assert not result.static_valid
    collisions = [
        contact
        for report in result.candidate_reports
        for contact in report.diagnostics.get("rejected_contacts", [])
        if contact["classification"] == "robot_environment_collision"
    ]
    assert collisions
    assert any("elbow" in contact["body_1"] + contact["body_2"] for contact in collisions)


def test_thick_wall_outside_loaded_gripper_aperture_is_rejected():
    geometry = box(thickness=0.12)
    with open_box_scene(geometry) as scene:
        with pytest.raises(ValueError, match="gripper aperture"):
            planner(scene, geometry)


def test_rotated_offset_box_keeps_jaw_axis_on_selected_wall_normal():
    quaternion = Rotation.from_euler("xyz", [0, 0, 38], degrees=True).as_quat()[[3, 0, 1, 2]]
    geometry = box(RigidPose([0.15, -0.06, 0.62], quaternion))
    with open_box_scene(geometry) as scene:
        rim_planner = planner(scene, geometry, samples_per_rim=1)
        candidates = sample_opposite_rims(
            geometry,
            samples_per_rim=1,
            side_offsets_m=(0,),
            depths_m=(0.006,),
            palm_pitch_deg=(45,),
            contact_half_width_m=0.012,
        )
        candidate = next(candidate for candidate in candidates if candidate.rim_pair == "y")
        assigned = rim_planner._assign_and_orient(state(-0.6), candidate)
    assert assigned is not None
    _, rotations = assigned
    wall_normal = geometry.pose.rotation[:, 1]
    assert abs(float(rotations[0][:, 1] @ wall_normal)) == pytest.approx(1.0, abs=1e-12)


def test_planning_is_deterministic_for_same_explicit_state():
    geometry = box()
    measured = state(-0.6)
    source_xml = DIRECTORY / "scene_locomani.xml"
    original_xml = source_xml.read_bytes()
    with open_box_scene(geometry) as scene:
        ik = ArmIK(str(scene))
        collision_types = ik.model.geom_contype.copy()
        rim_planner = BoxRimPlanner(ik, "water_bottle_object", geometry, options())
        first = rim_planner.plan(measured)
        second = rim_planner.plan(measured)
    assert first.selected_candidate == second.selected_candidate
    np.testing.assert_allclose(first.joints, second.joints, atol=1e-10)
    assert first.to_dict()["selection"] == second.to_dict()["selection"]
    np.testing.assert_array_equal(ik.model.geom_contype, collision_types)
    assert source_xml.read_bytes() == original_xml


def test_complete_json_entry_rejects_unknown_fields_and_runs(tmp_path):
    request = {
        "schema_version": 1,
        "scene_xml": str((DIRECTORY / "scene_locomani.xml").resolve()),
        "target": {"body": "water_bottle_object", "collider_mode": "replace_with_open_box_proxy"},
        "box": {
            "size_m": [0.32, 0.28, 0.18],
            "wall_thickness_m": 0.01,
            "pose": {"position": [0, 0, 0.62], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "robot_state": {
            "source": "explicit",
            "joints": [0] * 29,
            "pelvis": {"position": [-0.6, 0, 0.793], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "planner": {
            "samples_per_rim": 5,
            "side_offsets_m": [0],
            "depths_m": [0.006],
            "palm_pitch_deg": [45],
            "top_k": 4,
            "coarse_iterations": 350,
            "whole_body_iterations": 160,
            "min_joint_margin_rad": 0,
        },
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    parsed = load_request(path)
    _, _, result = run(parsed)
    assert result.static_valid
    assert len(result.to_dict()["selection"]["target_quaternions_wxyz"]) == 2
    invalid = copy.deepcopy(request)
    invalid["planner"]["mystery"] = 1
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="unknown fields: mystery"):
        load_request(path)


def test_diagnostic_floor_support_allows_growth_but_rejects_non_floor_obstacle():
    geometry = box()
    targets = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    measured = RobotState(np.zeros(29), [-0.6, 0.0, 0.7893], [1, 0, 0, 0])
    with open_box_scene(geometry) as scene:
        checker = planner(scene, geometry).collision
        allowed, rejected = checker.check(
            measured,
            measured.joints,
            [-0.6, 0.0, 0.7860],
            [1, 0, 0, 0],
            geometry,
            targets,
            support_contact_policy="diagnostic_floor",
        )

        assert not rejected
        support = [item for item in allowed if item.classification == "diagnostic_floor_support_contact"]
        assert support
        assert checker.floor_geoms == frozenset({0})
        assert max(item.support_penetration_growth_m for item in support) > 0.003

    obstacle = ([-0.56, 0.1185, 0.02], [0.1, 0.05, 0.02])
    elevated = RobotState(np.zeros(29), [-0.6, 0.0, 0.83], [1, 0, 0, 0])
    with open_box_scene(geometry, obstacle=obstacle) as scene:
        checker = planner(scene, geometry).collision
        _, rejected = checker.check(
            elevated,
            elevated.joints,
            elevated.pelvis_position,
            elevated.pelvis_quaternion_wxyz,
            geometry,
            targets,
            support_contact_policy="diagnostic_floor",
        )

        assert any(
            item.classification == "robot_environment_collision"
            and "box_rim_test_obstacle" in (item.geom_1, item.geom_2)
            for item in rejected
        )


def test_planner_propagates_diagnostic_floor_policy_and_keeps_support_admission():
    geometry = box()
    measured = state(-0.6)
    with open_box_scene(geometry) as scene:
        rim_planner = planner(scene, geometry, min_joint_margin_rad=0.002)
        original_check = rim_planner.collision.check
        policies = []

        def record_policy(*args, **kwargs):
            policies.append(kwargs["support_contact_policy"])
            return original_check(*args, **kwargs)

        rim_planner.collision.check = record_policy
        result = rim_planner.plan(measured, support_contact_policy="diagnostic_floor")

    assert result.static_valid
    assert policies and set(policies) == {"diagnostic_floor"}
    selected = next(report for report in result.candidate_reports if report.identifier == result.selected_candidate)
    assert max(selected.diagnostics["foot_anchor_errors_m"]) <= 0.005
    assert np.linalg.norm(selected.diagnostics["robot_com_shift_m"]) <= 0.015
