import cv2
import numpy as np
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.perception import CratePerception
from locomani.tasks.crate.visual_scene import MARKERS


def observation(perception, position):
    images = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in perception.streams}
    joints = np.zeros(29)
    cameras = perception.cameras(joints, images)
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    for stream, camera in cameras.items():
        for marker in MARKERS.values():
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
