import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.box_rim_geometry import (
    ForbiddenRimInterval,
    OpenBoxGeometry,
    RigidPose,
    sample_opposite_rims,
)


def pose(position=(0, 0, 0), euler_deg=(0, 0, 0)):
    quaternion = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()[[3, 0, 1, 2]]
    return RigidPose(position, quaternion)


def sample(box, **changes):
    options = dict(
        samples_per_rim=1,
        side_offsets_m=(0.0,),
        depths_m=(0.01,),
        palm_pitch_deg=(45.0,),
        contact_half_width_m=0.01,
    )
    options.update(changes)
    return sample_opposite_rims(box, **options)


def test_rims_follow_full_pose_translation_yaw_and_tilt():
    base = OpenBoxGeometry([0.4, 0.3, 0.2], 0.01, pose())
    transformed = OpenBoxGeometry(
        [0.4, 0.3, 0.2],
        0.01,
        pose((1.2, -0.7, 0.4), (18, -11, 67)),
    )
    local_candidates = sample(base)
    world_candidates = sample(transformed)
    assert [candidate.identifier for candidate in local_candidates] == [
        candidate.identifier for candidate in world_candidates
    ]
    for local, world in zip(local_candidates, world_candidates):
        expected = transformed.pose.transform_points(local.points_world)
        np.testing.assert_allclose(world.points_world, expected, atol=1e-12)


def test_inputs_are_copied_and_results_are_read_only():
    size = np.array([0.4, 0.3, 0.2])
    position = np.array([0.1, 0.2, 0.3])
    box = OpenBoxGeometry(size, 0.01, RigidPose(position, [1, 0, 0, 0]))
    candidate = sample(box)[0]
    size[:] = 9
    position[:] = 9
    np.testing.assert_array_equal(box.size_m, [0.4, 0.3, 0.2])
    np.testing.assert_array_equal(box.pose.position, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        candidate.points_world[0, 0] = 3


def test_one_sample_is_rim_midpoint_and_forbidden_intervals_cover_contact_span():
    box = OpenBoxGeometry(
        [0.4, 0.3, 0.2],
        0.01,
        pose(),
        forbidden_intervals=(ForbiddenRimInterval("x-", -0.005, 0.005),),
    )
    candidates = sample(box)
    assert {candidate.rim_pair for candidate in candidates} == {"y"}
    assert all(candidate.coordinate_m == pytest.approx(0.0) for candidate in candidates)


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: RigidPose([0, 0, 0], [2, 0, 0, 0]), "unit length"),
        (lambda: OpenBoxGeometry([0.4, 0.3, 0.005], 0.01, pose()), "wall height"),
        (lambda: OpenBoxGeometry([0.03, 0.3, 0.2], 0.02, pose()), "positive box interior"),
        (
            lambda: OpenBoxGeometry([0.4, 0.3, 0.2], 0.01, pose(), corner_clearance_m=0.16),
            "no usable rim",
        ),
    ],
)
def test_invalid_geometry_is_rejected(constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor()


def test_all_forbidden_rims_produce_no_candidates():
    restrictions = tuple(ForbiddenRimInterval(rim, -1, 1) for rim in ("x-", "x+", "y-", "y+"))
    box = OpenBoxGeometry([0.4, 0.3, 0.2], 0.01, pose(), forbidden_intervals=restrictions)
    assert sample(box) == ()
