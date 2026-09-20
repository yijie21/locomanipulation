import mujoco
import pytest

from locomani.pipeline.sim_video import recording_camera


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis"><geom size="0.1"/></body>'
        '<camera name="reverse_room" pos="-1.4 1.33 1.71"/></worldbody></mujoco>'
    )


def test_explicit_scene_camera_replaces_pelvis_tracking(model):
    camera = recording_camera(model, camera_name="reverse_room")
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert camera.fixedcamid == model.camera("reverse_room").id


def test_default_preserves_tracking_camera(model):
    camera = recording_camera(model)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert camera.trackbodyid == model.body("pelvis").id
    assert (camera.distance, camera.azimuth, camera.elevation) == (2.5, 135.0, -20.0)


@pytest.mark.parametrize("name", ["missing", "", False])
def test_invalid_camera_refuses_silent_fallback(model, name):
    with pytest.raises(ValueError, match="video_camera"):
        recording_camera(model, camera_name=name)
