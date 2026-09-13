import numpy as np
import pytest
from geometry_fixture import scene_ik
from test_crate_visual_perception import observation

from locomani.tasks.crate.perception import ApproachTarget, CratePerception
from locomani.tasks.crate.visual_approach import VisualApproach, initial_robot_xy
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
