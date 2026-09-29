"""view.IsaacView: the dataset streams it renders, and that it never hands back a
partial or stale set.

The offline dataset re-render and the live mirror both call this object, so a
stream mapped to the wrong MJCF camera, or a stale image slipping through, would
reach training and evaluation alike. Kit is not needed: the view's stage-facing
parts are replaced by fakes, the same way test_isaac_capture drives
stage.step_and_grab_all.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomani.sim.dataset_cameras import CAMERA_CONFIGS  # noqa: E402
from locomani.sim.isaac import view as V  # noqa: E402

XML = (
    "<mujoco><worldbody>"
    '<body name="torso_link"><camera name="head_camera" pos="0.06 0 0.45" quat="1 0 0 0" fovy="70"/>'
    '<geom size="0.1"/></body>'
    '<body name="right_wrist_yaw_link"><camera name="right_wrist" pos="0.1 0 0" quat="0 1 0 0" fovy="90"/>'
    '<geom size="0.1"/></body>'
    "</worldbody></mujoco>"
)


def _xml():
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(XML)
    return f.name


def test_streams_map_to_the_cameras_and_resolution_mujoco_publishes():
    specs = V.camera_specs(_xml(), ("ego_view", "right_wrist"))
    assert set(specs) == {"head_camera", "right_wrist"}
    for key in ("ego_view", "right_wrist"):
        c = CAMERA_CONFIGS[key]
        _pos, _quat, _fovy, w, h = specs[c["mjcf_name"]]
        assert (w, h) == (c["width"], c["height"])
    assert specs["right_wrist"][2] == 90.0


def test_unknown_stream_is_refused():
    with pytest.raises(SystemExit):
        V.camera_specs(_xml(), ("ego_view", "head_cam"))


class _World:
    tick = 0

    def step(self, render=True):
        self.tick += 1


class _Sensor:
    """Returns a fresh frame stamped with the world tick, unless `dead(tick)`."""

    def __init__(self, world, dead=lambda tick: False):
        self.world, self.dead = world, dead

    def get_current_frame(self):
        if self.dead(self.world.tick):
            return {"rendering_frame": 0, "rgba": None}
        return {"rendering_frame": self.world.tick, "rgba": np.full((2, 3, 4), self.world.tick, np.uint8)}


def _view(dead_wrist=lambda tick: False, supersample=1):
    view = V.IsaacView.__new__(V.IsaacView)
    view.world = _World()
    view.log = lambda *_: None
    view.supersample = supersample
    view.cameras = {
        "ego_view": ("head_camera", _Sensor(view.world)),
        "right_wrist": ("right_wrist", _Sensor(view.world, dead_wrist)),
    }
    view.push = lambda qpos, qvel=None: None
    return view


def test_render_returns_every_stream_as_rgb_from_the_last_settle_step():
    images = _view().render(np.zeros(3))
    assert set(images) == {"ego_view", "right_wrist"}
    assert images["ego_view"].shape == (2, 3, 3)
    assert images["ego_view"][0, 0, 0] == 2  # SETTLE_STEPS = 2: the second step's frame


def test_supersampled_streams_come_back_box_averaged_to_the_dataset_size():
    view = _view(supersample=2)
    tile = np.array([[0, 10], [20, 31]], np.uint8)[..., None].repeat(4, axis=2)
    for _name, sensor in view.cameras.values():
        sensor.get_current_frame = lambda w=view.world: {"rendering_frame": w.tick, "rgba": np.tile(tile, (2, 3, 1))}
    images = view.render(np.zeros(3))
    assert images["ego_view"].shape == (2, 3, 3)
    assert (images["ego_view"] == 15).all()  # (0 + 10 + 20 + 31) / 4 = 15.25


def test_downsample_refuses_a_size_that_is_not_a_multiple():
    with pytest.raises(ValueError):
        V.S.downsample(np.zeros((5, 4, 3), np.uint8), 2)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert V.S.downsample(image, 1) is image


def test_render_raises_instead_of_returning_a_partial_set():
    with pytest.raises(RuntimeError):
        _view(dead_wrist=lambda tick: True).render(np.zeros(3))


def test_warm_up_waits_for_a_late_stream_then_for_a_still_picture():
    view = _view(dead_wrist=lambda tick: tick < 6)
    steps = view.warm_up(np.zeros(3))
    # 3 steps until the wrist answers, then the picture must repeat 3 times --
    # it never does here (each frame carries its tick), so all 80 are spent.
    assert steps == 3 + 80


def test_warm_up_gives_up_on_a_stream_that_never_answers():
    with pytest.raises(RuntimeError):
        _view(dead_wrist=lambda tick: True).warm_up(np.zeros(3))
