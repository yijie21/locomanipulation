import cv2
import numpy as np
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.perception import CratePerception
from locomani.tasks.crate.visual_scene import MARKERS


def observation(perception, position, markers=MARKERS):
    images = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in perception.streams}
    joints = np.zeros(29)
    cameras = perception.cameras(joints, images)
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    for stream, camera in cameras.items():
        for marker in markers.values():
            xyz = rotation @ np.array(marker["position"]) + position
            pixel, depth = camera.project([xyz])
            x, y = np.round(pixel[0]).astype(int)
            if depth[0] > 0 and 8 <= x < 632 and 8 <= y < 472:
                cv2.circle(images[stream], (x, y), 4, tuple(int(c * 255) for c in marker["rgba"][:3]), -1)
    return images, joints


def test_rgb_pose_follows_visible_object_motion_using_encoder_camera_extrinsics():
    perception = CratePerception(scene_ik(), {"markers": MARKERS})
    for position in ([0.4, 0, -0.05], [0.44, 0.025, -0.03]):
        images, joints = observation(perception, position)
        estimate = perception.update(images, joints, 1.0)
        assert estimate is not None
        np.testing.assert_allclose(estimate.position, position, atol=0.008)
        assert estimate.reprojection_px < 1


def test_occlusion_returns_no_new_pose_even_with_a_previous_valid_estimate():
    perception = CratePerception(scene_ik(), {"markers": MARKERS})
    images, joints = observation(perception, [0.4, 0, -0.05])
    assert perception.update(images, joints, 1.0) is not None
    hidden = {key: np.zeros_like(value) for key, value in images.items()}
    assert perception.update(hidden, joints, 2.0) is None


def test_coloured_background_fragment_in_one_wrist_does_not_override_agreeing_views():
    perception = CratePerception(scene_ik(), {"markers": MARKERS})
    position = [0.4, 0, -0.05]
    images, joints = observation(perception, position)
    cv2.circle(images["right_wrist"], (50, 50), 12, (0, 255, 255), -1)
    estimate = perception.update(images, joints, 1.0)
    assert estimate is not None
    np.testing.assert_allclose(estimate.position, position, atol=0.008)
    assert "cyan" not in estimate.detections["right_wrist"]


def test_visible_shadowed_labels_still_supply_a_pose_but_black_occlusion_does_not():
    perception = CratePerception(scene_ik(), {"markers": MARKERS})
    images, joints = observation(perception, [0.4, 0, -0.05])
    shadowed = {name: (image.astype(float) * 0.25).astype(np.uint8) for name, image in images.items()}
    estimate = perception.update(shadowed, joints, 1.0)
    assert estimate is not None
    np.testing.assert_allclose(estimate.position, [0.4, 0, -0.05], atol=0.008)
    assert perception.update({name: image * 0 for name, image in images.items()}, joints, 1.1) is None


def test_floor_labels_recover_pose_when_the_far_wall_has_left_every_image():
    from locomani.tasks.crate.visual_scene import FLOOR_MARKERS

    perception = CratePerception(scene_ik(), {"markers": {**MARKERS, **FLOOR_MARKERS}})
    images, joints = observation(perception, [0.4, 0, -0.05], FLOOR_MARKERS)
    estimate = perception.update(images, joints, 1.0)
    assert estimate is not None
    np.testing.assert_allclose(estimate.position, [0.4, 0, -0.05], atol=0.008)
    assert all(set(points) <= set(FLOOR_MARKERS) for points in estimate.detections.values())


def test_floor_colour_identity_survives_render_gamma_without_becoming_wall_magenta():
    from locomani.tasks.crate.perception import detect_markers
    from locomani.tasks.crate.visual_scene import FLOOR_MARKERS

    image = np.zeros((80, 240, 3), dtype=np.uint8)
    for index, marker in enumerate(FLOOR_MARKERS.values()):
        colour = tuple(int(c ** (1 / 2.2) * 255) for c in marker["rgba"][:3])
        cv2.circle(image, (30 + index * 60, 40), 8, colour, -1)
    assert set(detect_markers(image)) == set(FLOOR_MARKERS)


def test_visible_floor_markers_are_not_discarded_for_larger_background_components():
    from locomani.tasks.crate.visual_scene import FLOOR_MARKERS

    perception = CratePerception(scene_ik(), {"markers": FLOOR_MARKERS})
    images, joints = observation(perception, [0.4, 0, -0.05], FLOOR_MARKERS)
    assert perception.update(images, joints, 1.0) is not None
    for image in images.values():
        cv2.circle(image, (30, 30), 10, (255, 7, 7), -1)
        cv2.circle(image, (90, 30), 10, (255, 102, 7), -1)
    estimate = perception.update(images, joints, 1.04)
    assert estimate is not None
    np.testing.assert_allclose(estimate.position, [0.4, 0, -0.05], atol=0.008)
