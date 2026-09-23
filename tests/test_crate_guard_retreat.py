"""Guard brake decision, guard approach, box grasp depth and the loaded retreat."""

from __future__ import annotations

import numpy as np
import pytest

from locomani.primitives.foot_watch import GuardBrake
from locomani.tasks.crate import retreat as retreat_module
from locomani.tasks.crate.grasp_region import grasp_points_for
from locomani.tasks.crate.perception import ApproachTarget
from locomani.tasks.crate.visual_approach import VisualApproach
from locomani.tasks.crate.visual_config import CrateVisualConfig


def test_guard_brakes_only_on_a_landing_that_one_more_stride_would_carry_past_the_line():
    rule = GuardBrake(extra=0.20, margin=0.03, stride0=0.14)
    line = 1.0
    # front ankle 0.60: 0.60 + 0.14 + 0.20 = 0.94 < 0.97 -> keep walking
    assert rule.should_brake(True, 0.60, 0.10, [], line) == (False, 0.14)
    # front ankle 0.64: 0.98 >= 0.97 -> brake, but only on a landing
    assert rule.should_brake(True, 0.64, 0.10, [], line)[0]
    assert not rule.should_brake(False, 0.64, 0.10, [], line)[0]
    # a longer observed stride is used over the floor
    assert rule.should_brake(True, 0.60, 0.20, [], line) == (True, 0.20)
    recent = GuardBrake(stride_mode="recent2")
    assert recent.should_brake(True, 0.50, 0.30, [0.05, 0.08], line)[1] == 0.08


def test_guard_approach_walks_straight_at_constant_speed_and_settles_on_the_brake_only():
    cfg = CrateVisualConfig(approach_stop="guard", walk_speed=0.25, walk_speed_min=0.25, stand_off=0.50)
    approach = VisualApproach(cfg)

    def near(t):  # already inside stand_off and off-centre; a fresh sighting each tick
        return ApproachTarget(np.array([0.30, 0.08, 0]), t, "visible")

    first = approach.step(0, near(0), [1, 0, 0, 0], grasp_pose_available=True)
    assert first.phase == "walk" and first.speed == 0.25 and np.allclose(first.direction, [1, 0, 0])
    braked = approach.step(0.02, near(0.02), [1, 0, 0, 0], grasp_pose_available=True, guard_brake=True)
    assert braked.phase == "settle"
    assert not approach.retry_closer()
    ready = None
    for k in range(3, 80):
        ready = approach.step(
            k * 0.02, near(k * 0.02), [1, 0, 0, 0], grasp_pose_available=True, base_velocity=[0, 0, 0]
        )
        if ready.ready:
            break
    assert ready.ready and approach.stand_attempts[-1]["accepted"]


def test_box_grasp_depth_is_configurable_and_the_crate_bar_ignores_it():
    box = {"box": {"half_extent": [0.225, 0.16, 0.165], "centre": [0, 0, 0.165]}}
    assert grasp_points_for(box, 0.01)[0][2] == pytest.approx(0.33 - 0.01)
    assert grasp_points_for(box)[0][2] == pytest.approx(0.33 - 0.025)
    assert grasp_points_for({}, 0.01) == grasp_points_for({})
    with pytest.raises(ValueError, match="approach_stop"):
        CrateVisualConfig(approach_stop="sometimes")


class _Cmd:
    def __init__(self):
        self.frames, self.hands = [], None

    def set_hands(self, left, right):
        self.hands = (left, right)

    def frame(self, **kw):
        self.frames.append(kw)


class _World:
    """Pelvis moves backward 1 cm per frame while SLOW_WALK is commanded; the load follows or drops."""

    def __init__(self, drop_after=None):
        self.cmd = _Cmd()
        self.t, self.y, self.frame_count, self.drop_after = 0.0, 0.0, 0, drop_after

    def obs(self):
        if self.cmd.frames and self.cmd.frames[-1]["mode"].name == "SLOW_WALK":
            self.y -= 0.01
        self.t += 0.02
        self.frame_count += 1
        dropped = self.drop_after is not None and self.frame_count > self.drop_after
        return {
            "t": self.t,
            "fallen": False,
            "pelvis": {"pos": [0.0, self.y, 0.78]},
            "objects": {
                "box": {"pos": [0, 0, 0.371 if dropped else 0.47], "touching": [] if dropped else ["left", "right"]}
            },
        }


class _Arm:
    hands = {"left": np.zeros(7), "right": np.zeros(7)}

    def __init__(self):
        self.stopped = None

    def stop(self, *, preserve_reference=False):
        self.stopped = preserve_reference


def test_retreat_completes_when_the_load_stays_held(monkeypatch):
    monkeypatch.setattr(retreat_module.time, "sleep", lambda _s: None)
    cfg = CrateVisualConfig(retreat_distance=0.3, lift_height=0.1)
    world, arm = _World(), _Arm()
    reason, diag, _state = retreat_module.retreat(world, arm, cfg, "box", np.pi / 2, 0.371)
    assert reason == "completed" and diag["walked"] >= 0.3 and arm.stopped is True
    assert world.cmd.hands is not None  # both hands re-sent before the handback
    backward = [f for f in world.cmd.frames if f["mode"].name == "SLOW_WALK"][0]
    assert np.allclose(backward["movement"][:2], [0, -1], atol=1e-9) and np.allclose(backward["facing"][:2], [0, 1])


def test_retreat_reports_a_drop(monkeypatch):
    monkeypatch.setattr(retreat_module.time, "sleep", lambda _s: None)
    cfg = CrateVisualConfig(retreat_distance=0.3, lift_height=0.1)
    reason, _diag, _state = retreat_module.retreat(_World(drop_after=5), _Arm(), cfg, "box", np.pi / 2, 0.371)
    assert reason == "retreat_dropped"


def test_box_front_line_is_the_near_face_along_the_walk_in_axis():
    from locomani.tasks.crate.expert import box_front_line

    box = {"half_extent": [0.225, 0.16, 0.165], "centre": [0, 0, 0.165]}
    axis = np.array([0.0, 1.0])  # the warehouse walks +y
    state = {"objects": {"b": {"pos": [0.0, -0.39, 0.371], "quat": [1.0, 0.0, 0.0, 0.0]}}}
    assert box_front_line(state, "b", box, axis) == pytest.approx(-0.55)
    # yawed 90 deg the long side faces the robot: the face moves 0.225 - 0.16 nearer
    c = np.cos(np.pi / 4)
    state["objects"]["b"]["quat"] = [c, 0.0, 0.0, c]
    assert box_front_line(state, "b", box, axis) == pytest.approx(-0.39 - 0.225)


def test_start_advance_may_step_back_from_the_scene_start():
    assert CrateVisualConfig(start_advance=-0.25).start_advance == -0.25
    with pytest.raises(ValueError):
        CrateVisualConfig(start_advance=float("nan"))


def test_hanging_walk_sends_plain_planner_frames():
    from locomani.backends.sonic.agent_server import LocomotionMode
    from locomani.tasks.crate.expert import HangingWalk

    sent = []

    class Cmd:
        def frame(self, **kw):
            sent.append(kw)

    class P:
        cmd = Cmd()

        def obs(self):
            return {"joints": np.zeros(29)}

    walk = HangingWalk(P(), np.pi / 2)
    walk.mode = LocomotionMode.SLOW_WALK
    walk._push(np.ones(29), reference_velocity=np.zeros(29), movement=np.array([0.0, 1.0, 0.0]), speed=0.25)
    (frame,) = sent
    assert frame["mode"] == LocomotionMode.SLOW_WALK and frame["speed"] == 0.25
    assert np.allclose(frame["facing"], [0.0, 1.0, 0.0], atol=1e-12)
    assert set(frame) == {"mode", "facing", "movement", "speed"}  # no upper-body reference rides along


def test_offline_render_drops_the_sim_cameras_and_its_video():
    from locomani.stack.manager import spec_from_config

    cfg = {"task": {"prompt": "x"}, "collect": {}, "backend": {"render": "offline"}}
    spec = spec_from_config(cfg, run_dir="/tmp/r", gpu="0", video_dir="/tmp/r/videos")
    assert not spec.image_publish and spec.video_dir is None
    live = spec_from_config({**cfg, "backend": {}}, run_dir="/tmp/r", gpu="0", video_dir="/tmp/r/videos")
    assert live.image_publish and live.video_dir == "/tmp/r/videos"


def test_offline_guidance_admits_placeholder_frames_on_geometry():
    from locomani.tasks.crate.expert import PlaceholderCamera

    frames = PlaceholderCamera().read()
    assert set(frames["images"]) == {"ego_view", "left_wrist", "right_wrist"}
    assert all(img.shape == (480, 640, 3) and not img.any() for img in frames["images"].values())


def test_placeholder_frames_keep_their_stamp_between_ticks():
    from locomani.tasks.crate.expert import PlaceholderCamera

    camera = PlaceholderCamera()
    first = camera.read()["timestamps"]["ego_view"]
    assert camera.read()["timestamps"]["ego_view"] == first  # same frame within 1/30 s, like the live stream
