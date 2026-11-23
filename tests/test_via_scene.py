"""selfmodel_via_20261120/via_scene.py: the clearance of a sweep of spheres to a route's boxes, which every touch share and
every "clearance lost" number rests on."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "selfmodel_via_20261120"))
import via_scene as VS  # noqa: E402

BOX = np.array([[0.0, 0.0, 0.0, 0.1, 0.2, 0.3]])  # centre, half sizes


def test_clearance_is_the_distance_to_the_box_minus_the_radius_and_negative_inside():
    P = np.array([[[0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.0]]])  # one pose, three spheres
    got = VS.clearance(P, BOX, np.array([0.05, 0.0, 0.0]))
    assert np.allclose(got, [0.2 - 0.05, -0.1, np.hypot(0.1, 0.1)])  # beside a face; at the centre; off an edge


def test_clearance_takes_the_smallest_over_poses_and_over_boxes():
    P = np.array([[[1.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[0.8, 0.0, 0.0]]])  # three poses of one sphere
    far = np.array([[5.0, 0.0, 0.0, 0.1, 0.1, 0.1]])
    assert np.allclose(VS.clearance(P, BOX, np.zeros(1)), [0.4])
    assert np.allclose(VS.clearance(P, np.concatenate([far, BOX]), np.zeros(1)), [0.4])


def test_lost_and_good_read_the_worst_sphere_of_each_reach():
    o = dict(
        wanted=np.array([[0.03, 0.02], [0.03, 0.02]]),
        actual=np.array([[0.01, 0.04], [0.05, -0.01]]),
        end=np.array([0.01, 0.01]),
    )
    assert np.allclose(VS.lost(o), [0.02 - 0.01, 0.02 + 0.01])
    assert VS.good(o).tolist() == [True, False]  # both arrive; the second has a sphere inside a box
    o["end"] = np.array([0.03, 0.01])
    assert VS.good(o).tolist() == [False, False]
