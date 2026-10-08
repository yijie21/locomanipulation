"""third_person_shot: the one camera every renderer films a round with, and its pose MuJoCo computes for them."""

import math

import mujoco
import numpy as np
import pytest

from locomani.pipeline import sim_video
from locomani.sim import third_person_shot as shot

SCENE = """<mujoco><visual><global fovy="45"/></visual><worldbody>
  <body name="pelvis" pos="0 0 0.8"><freejoint/><geom size="0.1"/></body>
  <camera name="edge_ym" pos="-1.4 1.3 1.7" xyaxes="1 -1 0 0.3 0.3 1" fovy="{fovy_ym}"/>
  <camera name="edge_xp" pos="1.4 -1.5 1.7" xyaxes="-1 1 0 -0.3 -0.3 1" fovy="73.74"/>
  <camera name="reverse_room" pos="-1.4 1.3 1.7" xyaxes="1 -1 0 0.3 0.3 1" fovy="60"/>
</worldbody></mujoco>"""
EDGES = {
    "ym": {"heading_deg": 90.0, "video_camera": "edge_ym"},
    "xp": {"heading_deg": 180.0, "video_camera": "edge_xp"},
}
FOUR_EDGE = {"target_object": "box", "objects": {"box": {"surface": "A"}}, "surfaces": {"A": {"edges": EDGES}}}


def model(fovy_ym=73.74):
    return mujoco.MjModel.from_xml_string(SCENE.format(fovy_ym=fovy_ym))


def qpos_facing(yaw_deg):
    return [0, 0, 0.8, math.cos(math.radians(yaw_deg) / 2), 0, 0, math.sin(math.radians(yaw_deg) / 2)]


def test_film_rows_are_the_ones_the_recorder_writes():
    written = [i for i in range(23) if (i + 1) % sim_video.EVERY_NTH == 0]
    assert list(shot.film_rows(23)) == written == [3, 7, 11, 15, 19]


def test_a_four_edge_round_is_filmed_from_its_edge_camera_else_the_scene_camera_else_tracking():
    m = model()
    cam = shot.round_camera(m, FOUR_EDGE, qpos_facing(175))
    assert (cam.type, cam.fixedcamid) == (mujoco.mjtCamera.mjCAMERA_FIXED, m.camera("edge_xp").id)
    cam = shot.round_camera(m, {"video_camera": "reverse_room"}, qpos_facing(0))
    assert cam.fixedcamid == m.camera("reverse_room").id
    assert shot.round_camera(m, {}, qpos_facing(0)).type == mujoco.mjtCamera.mjCAMERA_TRACKING


def test_scene_fovy_is_the_shared_edge_fovy_and_refuses_edges_that_differ():
    assert shot.scene_fovy(model(), FOUR_EDGE) == 73.74
    assert shot.scene_fovy(model(), {"video_camera": "reverse_room"}) == 60.0
    assert shot.scene_fovy(model(), {}) == 45.0  # the tracking camera's: the model's free-camera fovy
    with pytest.raises(ValueError, match="differ in fovy"):
        shot.scene_fovy(model(fovy_ym=60), FOUR_EDGE)


def test_a_fixed_camera_pose_is_the_scene_camera_itself_in_the_minus_z_forward_convention():
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = shot.round_camera(m, FOUR_EDGE, qpos_facing(90))  # edge_ym
    pos, quat = shot.camera_poses(m, [qpos_facing(90), qpos_facing(95)], cam)
    want = np.zeros(4)
    mujoco.mju_mat2Quat(want, d.cam_xmat[m.camera("edge_ym").id])  # MuJoCo cameras look down -Z, +Y up
    assert np.allclose(pos, d.cam_xpos[m.camera("edge_ym").id], atol=1e-6)
    assert all(abs(abs(float(q @ want)) - 1) < 1e-6 for q in quat)


def test_the_tracking_camera_follows_the_pelvis():
    m = model()
    cam = shot.round_camera(m, {}, qpos_facing(0))
    moved = qpos_facing(0)
    moved[0] = 1.0
    pos, _quat = shot.camera_poses(m, [qpos_facing(0), moved], cam)
    assert np.allclose(pos[1] - pos[0], [1.0, 0, 0], atol=1e-3)
