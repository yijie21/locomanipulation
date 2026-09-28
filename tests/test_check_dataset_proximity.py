"""The audit's "wrist near the object at close" check, measured to the object's box when the scene gives one."""

import numpy as np

from locomani.pipeline.check_dataset import box_surface_distance

IDENTITY = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)  # pos3 + the first two rotation columns
# nor_cal_box: the origin is the bottom face, the box centre 0.165 m above it
CARTON = {"half_extent": [0.225, 0.16, 0.165], "centre": [0.0, 0.0, 0.165]}


def test_a_hand_on_the_rim_is_near_a_box_whose_origin_is_its_bottom_face():
    wrists = np.array([0.0, 0.0, 0.43])  # both wrists' midpoint, 10 cm above the rim
    assert np.linalg.norm(wrists - IDENTITY[:3]) > 0.30  # what the audit measured before
    assert abs(box_surface_distance(IDENTITY, wrists, CARTON) - 0.10) < 1e-9


def test_inside_is_zero_and_the_rotation_is_applied():
    assert box_surface_distance(IDENTITY, np.array([0.1, 0.1, 0.1]), CARTON) == 0.0
    yawed = IDENTITY.copy()
    yawed[3:9] = [0, 1, 0, -1, 0, 0]  # 90 deg about z: the long side now runs along world y
    point = np.array([0.0, 0.2, 0.165])  # inside the long side (0.225), outside the short one (0.16)
    assert box_surface_distance(yawed, point, CARTON) == 0.0
    assert abs(box_surface_distance(IDENTITY, point, CARTON) - 0.04) < 1e-9


def test_video_alignment_judges_a_smoke_sized_set_on_its_median():
    """2026-09-28: a two-episode smoke set failed the majority test on one episode's one-frame jitter."""
    from locomani.pipeline.check_dataset import MAJORITY_MIN, video_aligned

    assert video_aligned([-1, 0, 0, 1])  # crate_smoke's four measurements
    assert video_aligned([0])
    assert not video_aligned([3, 3, 2, 3])  # a 3-frame delay, as the injection experiment made it
    assert not video_aligned([1, 1, -1, 1])  # nothing at lag 0
    assert not video_aligned([])
    clean = [0] * 37 + [1, -1, 2]
    assert len(clean) >= MAJORITY_MIN and video_aligned(clean)
    assert not video_aligned([0] * 5 + [1] * 5)  # the full-size rule is unchanged: a strict majority at 0
