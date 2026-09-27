import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation
from test_crate_visual_perception import observation

from locomani.tasks.crate.perception import ApproachTarget, CratePerception
from locomani.tasks.crate.visual_approach import (
    VisualApproach,
    approach_heading,
    edge_distance,
    initial_robot_xy,
    scene_object_yaw,
)
from locomani.tasks.crate.visual_config import CrateVisualConfig
from locomani.tasks.crate.visual_scene import MARKERS


def test_unsupported_reach_can_use_braking_allowance_without_crossing_stand_off():
    cfg = CrateVisualConfig(stand_off=0.39, walk_brake_margin=0.10)
    approach = VisualApproach(cfg)
    target = ApproachTarget(np.array([0.48, 0, 0]), 0.0, "visible_simulation_pose")
    assert approach.step(0, target, [1, 0, 0, 0], grasp_pose_available=True).phase == "settle"
    assert approach.retry_closer()
    command = approach.step(0.01, target, [1, 0, 0, 0], grasp_pose_available=True)
    assert command.phase == "walk" and command.speed > 0 and not command.ready
    for _ in range(10):
        if not approach.retry_closer():
            break
    assert approach.stop_distance == cfg.stand_off
    assert not approach.retry_closer()


def test_quiet_stance_is_measured_in_physics_time_when_the_host_slows():
    fast, slow = VisualApproach(CrateVisualConfig()), VisualApproach(CrateVisualConfig())
    for simulation_time in np.arange(0, 2, 0.05):
        target = np.array([0.35, 0, 0])
        a = fast.step(
            simulation_time,
            ApproachTarget(target, simulation_time, "visible"),
            [1, 0, 0, 0],
            grasp_pose_available=True,
            motion_time=simulation_time,
        )
        wall = 10 + 4 * simulation_time
        b = slow.step(
            wall,
            ApproachTarget(target, wall, "visible"),
            [1, 0, 0, 0],
            grasp_pose_available=True,
            motion_time=simulation_time,
        )
        assert a.ready == b.ready and a.phase == b.phase


def test_far_visible_pair_guides_approach_without_filling_occluded_markers():
    perception = CratePerception(scene_ik(), {"markers": MARKERS})
    position = [1.35, 0.025, -0.05]
    images, joints = observation(perception, position)
    # Only the top pair is observable from the low, distant view.
    for image in images.values():
        green_or_yellow = (image[..., 1] > 200) & (image[..., 2] < 50)
        image[green_or_yellow] = 0
    assert perception.update(images, joints, 1.0) is None
    target = perception.approach_target(images, joints, 1.0)
    assert target is not None and target.source == "visible_top_pair"
    np.testing.assert_allclose(target.position, position, atol=0.03)
    for image in images.values():
        image[:] = 0
    assert perception.approach_target(images, joints, 2.0) is None


def test_initial_distance_uses_the_table_edge_not_its_centre():
    meta = {
        "objects": {"blue_crate": {"surface": "A"}},
        "surfaces": {"A": {"xy": [2, 1], "flat_half_extent": [0.5, 0.5]}},
    }
    np.testing.assert_allclose(initial_robot_xy(meta, 1.3), [0.2, 1])
    with pytest.raises(ValueError, match="start_edge"):
        CrateVisualConfig(start_edge_min=0.3)


def test_stop_and_continuous_observed_settle_precede_grasp():
    approach = VisualApproach(CrateVisualConfig())
    far = approach.step(0, ApproachTarget(np.array([1.5, 0, 0]), 0, "pair"), [1, 0, 0, 0], grasp_pose_available=False)
    assert far.speed > 0 and not far.ready
    ready_at = None
    for now in np.arange(0.05, 6, 0.05):
        command = approach.step(
            now, ApproachTarget(np.array([0.40, 0, 0]), now, "pose"), [1, 0, 0, 0], grasp_pose_available=True
        )
        assert command.speed == 0
        if command.ready:
            ready_at = now
            break
    assert ready_at is not None and ready_at > 1.0


def test_missing_visual_target_cannot_keep_walking():
    approach = VisualApproach(CrateVisualConfig())
    approach.step(0, ApproachTarget(np.array([1.5, 0, 0]), 0, "pair"), [1, 0, 0, 0], grasp_pose_available=False)
    missing = approach.step(0.2, None, [1, 0, 0, 0], grasp_pose_available=False)
    assert missing.speed == 0 and not missing.ready
    assert approach.step(0.8, None, [1, 0, 0, 0], grasp_pose_available=False).reason == "approach_vision_lost"


def test_millimetre_pixel_fit_jitter_does_not_look_like_continuous_body_motion():
    approach = VisualApproach(CrateVisualConfig(settle_seconds=0.3))
    ready = False
    for index, now in enumerate(np.arange(0, 1.5, 0.03)):
        position = np.array([0.40 + (-1) ** index * 0.003, 0, 0])
        command = approach.step(now, ApproachTarget(position, now, "pose"), [1, 0, 0, 0], grasp_pose_available=True)
        ready |= command.ready
    assert ready, "zero-mean 3 mm pose noise prevented a quiet-stance decision"


def test_sustained_visual_motion_still_prevents_stance_admission():
    approach = VisualApproach(CrateVisualConfig(settle_seconds=0.3))
    for now in np.arange(0, 1.5, 0.03):
        command = approach.step(
            now,
            ApproachTarget(np.array([0.40 - 0.12 * now, 0, 0]), now, "pose"),
            [1, 0, 0, 0],
            grasp_pose_available=True,
        )
        assert not command.ready


def test_approach_prepares_both_hands_before_stopping_with_bounded_joint_motion():
    from test_crate_coordinated_reach import setup

    from locomani.tasks.crate.perception import CrateEstimate
    from locomani.tasks.crate.visual_approach import ApproachPreparation

    policy, state = setup()
    prep = ApproachPreparation(policy.solver, policy.config.visual, state["joints"])
    previous, velocity = state["joints"].copy(), np.zeros(29)
    initial_tools = policy.solver.tool_poses(previous)[0]
    for now in np.arange(0, 4.01, 0.05):
        seen = CrateEstimate(
            np.array([1.4 - 0.25 * now, 0, -0.395]),
            Rotation.from_euler("z", 90, degrees=True).as_matrix(),
            now,
            None,
            {},
        )
        q, v = prep.step(now, seen)
        if now:
            np.testing.assert_allclose((q - previous) / 0.05, v, atol=1e-8)
            assert np.max(np.abs(v - velocity)) <= prep.config.joint_acceleration * 0.05 + 1e-6
        if 2.99 < now < 3.01:
            # Still outside the stop distance: both hands already move down.
            assert seen.position[0] > prep.config.stand_off + prep.config.walk_brake_margin
            assert np.max(policy.solver.tool_poses(q)[0][:, 2] - initial_tools[:, 2]) < -0.03
        previous, velocity = q.copy(), v.copy()
    np.testing.assert_allclose(q[:12], state["joints"][:12])
    assert 0.01 < q[14] <= 0.12 + 1e-3


def test_assisted_stance_uses_current_proprioception_and_still_requires_a_quiet_interval():
    cfg = CrateVisualConfig(settle_seconds=0.3)
    approach = VisualApproach(cfg)
    # The target stops moving before the visual velocity filter has decayed.
    for now in np.arange(0, 1.21, 0.05):
        moving = now < 0.8
        point = np.array([max(0.39, 0.9 - 0.8 * now), 0, 0])
        command = approach.step(
            now,
            ApproachTarget(point, now, "visible"),
            [1, 0, 0, 0],
            grasp_pose_available=True,
            base_velocity=[0.3 if moving else 0.01, 0, 0],
        )
        if now < 1.1:
            assert not command.ready
    assert command.ready


def _settle(approach, depth, start, grasp_pose_available=True):
    """Hold a quiet stance at `depth` from `start` until the approach decides; return the deciding command."""
    for now in np.arange(start, start + 3, 0.05):
        command = approach.step(
            now,
            ApproachTarget(np.array([depth, 0, 0]), now, "pose"),
            [1, 0, 0, 0],
            grasp_pose_available=grasp_pose_available,
        )
        if command.ready or command.phase == "walk":
            return command, now
    raise AssertionError("no decision within 3 s")


def test_far_quiet_stance_re_walks_before_it_is_admitted():
    cfg = CrateVisualConfig(stand_off=0.39, walk_brake_margin=0.07, stand_depth_tolerance=0.04)
    approach = VisualApproach(cfg)
    # The brake margin lets the walk stop at 0.46; a stance that rests there is 7 cm too far.
    command, now = _settle(approach, 0.46, 0.0)
    assert command.phase == "walk" and command.speed > 0 and not command.ready
    assert approach.stand_attempts == [
        dict(depth_m=pytest.approx(0.46), lateral_m=0.0, depth_error_m=pytest.approx(0.07), accepted=False)
    ]
    assert approach.stop_distance == pytest.approx(0.425)
    # The second walk-in lands inside the tolerance and is admitted after a fresh quiet interval.
    command, later = _settle(approach, 0.42, now + 0.05)
    assert command.ready and later - now >= cfg.settle_seconds
    assert approach.stand_attempts[-1]["accepted"] and len(approach.stand_attempts) == 2


def test_depth_retries_are_bounded_and_the_last_stance_is_recorded():
    cfg = CrateVisualConfig(stand_off=0.39, walk_brake_margin=0.07, stand_depth_tolerance=0.04, stand_depth_retries=0)
    approach = VisualApproach(cfg)
    command, _ = _settle(approach, 0.46, 0.0)
    assert command.ready, "with no retries the far stance is handed to reach admission, not re-walked"
    assert approach.stand_attempts[-1] == dict(
        depth_m=pytest.approx(0.46), lateral_m=0.0, depth_error_m=pytest.approx(0.07), accepted=True
    )


def test_near_stance_is_not_re_walked():
    approach = VisualApproach(CrateVisualConfig(stand_off=0.39, stand_depth_tolerance=0.04))
    command, _ = _settle(approach, 0.33, 0.0)
    assert command.ready and approach.stand_attempts[-1]["accepted"]


def test_depth_convergence_fields_are_validated():
    with pytest.raises(ValueError, match="stand_depth_retries"):
        CrateVisualConfig(stand_depth_retries=1.5)
    with pytest.raises(ValueError, match="stand_depth_tolerance"):
        CrateVisualConfig(stand_depth_tolerance=0)


def test_walk_commands_never_fall_below_the_stepping_floor():
    cfg = CrateVisualConfig(stand_off=0.39, walk_brake_margin=0.07, walk_speed_min=0.35)
    approach = VisualApproach(cfg)
    # 8 cm of depth error used to command 0.10, which the deploy does not walk on.
    near = approach.step(0, ApproachTarget(np.array([0.47, 0, 0]), 0, "pose"), [1, 0, 0, 0], grasp_pose_available=True)
    assert near.phase == "walk" and near.speed == pytest.approx(0.35)
    with pytest.raises(ValueError, match="walk_speed_min"):
        CrateVisualConfig(walk_speed=0.3, walk_speed_min=0.35)


def test_walk_gate_keeps_walking_while_lateral_is_out_of_tolerance():
    cfg = CrateVisualConfig(stand_off=0.36, walk_brake_margin=0.07, stand_lateral_tolerance=0.14)
    approach = VisualApproach(cfg)
    command = approach.step(
        0, ApproachTarget(np.array([0.40, 0.20, 0]), 0, "pose"), [1, 0, 0, 0], grasp_pose_available=True
    )
    assert command.phase == "walk" and command.speed > 0 and command.facing_offset == 0


def test_turn_gate_stops_on_depth_and_turns_toward_the_lateral_residual():
    cfg = CrateVisualConfig(
        stand_off=0.36,
        walk_brake_margin=0.07,
        stand_lateral_tolerance=0.14,
        stand_lateral_gate="turn",
        settle_seconds=0.3,
    )
    approach = VisualApproach(cfg)
    target = ApproachTarget(np.array([0.40, 0.20, 0]), 0, "pose")
    command = approach.step(0, target, [1, 0, 0, 0], grasp_pose_available=True)
    assert command.phase == "turn" and command.speed == 0 and not command.ready
    assert command.facing_offset == pytest.approx(np.arctan2(0.20, 0.40))
    # Once the target sits inside the lateral tolerance the ordinary quiet settle follows.
    command, _ = _settle(approach, 0.40, 0.05)
    assert command.ready
    assert approach.stand_attempts[-1]["accepted"]


def test_turn_gate_field_is_validated():
    with pytest.raises(ValueError):
        CrateVisualConfig(stand_lateral_gate="spin")


def _room_meta():
    """The logistics room: robot_start faces +y, the platform sits ahead, the box lies with its long axis across."""
    return {
        "target_object": "nor_cal_box",
        "robot_start": {"xy": [0.0, -1.205], "yaw_deg": 90.0},
        "objects": {"nor_cal_box": {"surface": "A", "init_quat": [1.0, 0.0, 0.0, 0.0]}},
        "surfaces": {"A": {"xy": [0.0, -0.105], "flat_half_extent": [0.6, 0.45]}},
    }


def test_room_placement_walks_in_along_the_scene_heading():
    meta = _room_meta()
    assert approach_heading(meta) == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(initial_robot_xy(meta, 1.3), [0.0, -0.555 - 1.3], atol=1e-9)
    assert edge_distance(meta, [0.0, -0.98, 0.79]) == pytest.approx(0.425)
    assert scene_object_yaw(meta) == pytest.approx(0.0)


def test_synthetic_table_placement_is_unchanged_by_the_heading_generalisation():
    meta = {
        "target_object": "nor_cal_box",
        "robot_start": {"xy": [-1.5, 0.0], "yaw_deg": 0.0},
        "objects": {"nor_cal_box": {"surface": "A", "init_quat": [0.7071067811865476, 0.0, 0.0, 0.7071067811865475]}},
        "surfaces": {"A": {"xy": [0.0, 0.0], "flat_half_extent": [0.301, 0.301]}},
    }
    np.testing.assert_allclose(initial_robot_xy(meta, 1.3), [-0.301 - 1.3, 0.0])
    assert edge_distance(meta, [-1.5, 0.0]) == pytest.approx(1.199)
    assert scene_object_yaw(meta) == pytest.approx(np.pi / 2)


def test_start_pose_field_is_validated():
    with pytest.raises(ValueError):
        CrateVisualConfig(start_pose="video")
    assert CrateVisualConfig(start_pose="scene", start_advance=0.225).start_advance == 0.225


def test_depth_steering_walks_straight_and_stops_on_depth_despite_lateral_offset():
    """walk_steer=depth: no sideways movement, and a lateral residual beyond tolerance does not keep it walking."""
    cfg = CrateVisualConfig(stand_off=0.40, walk_brake_margin=0.05, stand_lateral_tolerance=0.04, walk_steer="depth")
    approach = VisualApproach(cfg)
    far = approach.step(
        0, ApproachTarget(np.array([0.60, 0.10, 0]), 0.0, "visible"), [1, 0, 0, 0], grasp_pose_available=True
    )
    assert far.phase == "walk" and far.speed > 0 and np.allclose(far.direction, [1, 0, 0])
    near = approach.step(
        0.02, ApproachTarget(np.array([0.44, 0.10, 0]), 0.02, "visible"), [1, 0, 0, 0], grasp_pose_available=True
    )
    assert near.phase == "settle" and near.speed == 0


def test_target_steering_still_chases_the_lateral_offset():
    approach = VisualApproach(CrateVisualConfig(stand_off=0.40, walk_brake_margin=0.05, stand_lateral_tolerance=0.04))
    command = approach.step(
        0, ApproachTarget(np.array([0.44, 0.10, 0]), 0.0, "visible"), [1, 0, 0, 0], grasp_pose_available=True
    )
    assert command.phase == "walk" and command.direction[1] > 0.5


def test_walk_steer_rejects_unknown_choice():
    with pytest.raises(ValueError, match="walk_steer"):
        CrateVisualConfig(walk_steer="sideways")


def test_facing_correction_aims_at_the_centre_or_squares_to_the_grasp_line():
    from locomani.tasks.crate.perception import CrateEstimate
    from locomani.tasks.crate.visual_approach import facing_correction

    points = [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]  # object-local grasp points across the walk-in (left, right)
    seen = CrateEstimate(np.array([0.5, 0.1, 0.0]), np.eye(3), 0.0, None, {})
    assert facing_correction("none", seen, points) == 0.0
    assert facing_correction("centre", seen, points) == pytest.approx(np.arctan2(0.1, 0.5))
    assert facing_correction("box_yaw", seen, points) == pytest.approx(0.0, abs=1e-9)  # square already
    turned = CrateEstimate(
        np.array([0.5, 0.0, 0.0]), Rotation.from_euler("z", 10, degrees=True).as_matrix(), 0.0, None, {}
    )
    assert facing_correction("box_yaw", turned, points) == pytest.approx(np.radians(10))
    # the grasp line's normal has two senses; the one nearest to straight ahead is taken
    flipped = CrateEstimate(
        np.array([0.5, 0.0, 0.0]), Rotation.from_euler("z", 190, degrees=True).as_matrix(), 0.0, None, {}
    )
    assert facing_correction("box_yaw", flipped, points) == pytest.approx(np.radians(10))
    assert facing_correction("centre", None, points) == 0.0


def test_ready_after_brake_choices():
    CrateVisualConfig(ready_after_brake="quick")
    with pytest.raises(ValueError, match="ready_after_brake"):
        CrateVisualConfig(ready_after_brake="fast")
