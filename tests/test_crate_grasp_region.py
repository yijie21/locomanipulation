import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.grasp_region import RimRegion


def test_valid_off_centre_rim_grasp_does_not_require_returning_to_the_slot_centre():
    region = RimRegion()
    centres = np.array([[0.1933, 0, 0.060], [-0.1933, 0, 0.060]])
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    centres = centres @ rotation.T
    wrists = [Rotation.from_euler("y", 55, degrees=True).as_matrix()] * 2
    tools = centres + [0.070, 0, 0.02]
    ready, _ = region.contains(centres, rotation, tools, wrists)
    assert ready
    target = region.nearest(centres, rotation[:, 1], tools)
    np.testing.assert_allclose(target, centres + [0.070, 0, 0])
    np.testing.assert_allclose(target[0] - target[1], centres[0] - centres[1], atol=1e-12)


def test_end_wall_wrong_jaw_direction_and_high_fingers_cannot_authorize_closure():
    region = RimRegion()
    centres = np.array([[0, 0.1933, 0.060], [0, -0.1933, 0.060]])
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    wrists = [Rotation.from_euler("y", 60, degrees=True).as_matrix()] * 2
    for displacement in ([0.105, 0, 0], [0, 0.020, 0], [0, 0, 0.045], [0, 0, -0.015]):
        assert not region.contains(centres, rotation, centres + displacement, wrists)[0]
    assert not region.contains(centres, rotation, centres, [np.eye(3)] * 2)[0]
