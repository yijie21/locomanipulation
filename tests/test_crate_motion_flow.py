from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.backends.sonic import streamed_arm
from locomani.tasks.crate import bimanual_carry as carry
from locomani.tasks.crate import carry_phases as phases
from locomani.tasks.crate.carry_config import DEFAULT_CARRY
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.settling import SettlingWindow


class Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


def controller(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(phases, "time", clock)
    monkeypatch.setattr(streamed_arm, "time", clock)
    monkeypatch.setattr(carry, "time", clock)
    ik = scene_ik()
    state = {"joints": np.zeros(29), "pelvis": {"pos": [0, 0, 0.79], "quat": [1, 0, 0, 0]}}
    command = Mock(left_hand=[0, 0], right_hand=[0, 0])
    p = SimpleNamespace(
        cmd=command,
        obs=lambda: dict(state, wall=clock.now),
        pelvis_xy=lambda s: np.asarray(s["pelvis"]["pos"][:2]),
        _facing=lambda s: (1, 0, 0),
    )
    monkeypatch.setattr(carry, "PlannerArm", lambda *a, **kw: streamed_arm.PlannerArm(*a, **kw, ik=ik))
    return carry.BimanualArms(p, config=DEFAULT_CARRY), p, clock


def test_stable_brake_does_not_add_a_fixed_one_second_wait(monkeypatch):
    _, p, clock = controller(monkeypatch)
    ctx = SimpleNamespace(
        p=p,
        ramp=SimpleNamespace(push=lambda q, now: 0.0, quiet=True),
        q_pre17=np.zeros(17),
        psi=0.0,
        m={},
        motion_recording=None,
        config=DEFAULT_CARRY,
    )
    assert phases.brake(ctx)
    assert clock.now <= 0.5, f"already stable but brake waited {clock.now:.2f}s"


def test_settling_requires_fresh_continuous_quiet_pose_and_completed_reference():
    window = SettlingWindow(0.3)

    def step(t, angle=0, ready=True):
        quat = np.roll(Rotation.from_euler("y", angle).as_quat(), 1)
        return window.step(t, [0, 0], quat, reference_ready=ready)

    for t in np.arange(0, 0.5, 0.02):
        assert not step(t, t)  # body rotates at 1 rad/s, even with stationary feet
    for _ in range(30):
        assert not step(0.48, 0.48)  # a repeated frame cannot count as quiet time
    for t in np.arange(0.5, 0.9, 0.02):
        assert not step(t, 0.48, ready=False)
    for t in np.arange(0.9, 1.16, 0.02):
        assert not step(t, 0.48)
    assert not step(1.4, 0.48)  # missing frames break the continuous interval
    result = False
    for t in np.arange(1.42, 1.76, 0.02):
        result = step(t, 0.48)
    assert result


@pytest.mark.parametrize("field", ["settle_seconds", "joint_speed", "joint_acceleration"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_carry_setting_is_rejected(field, value):
    with pytest.raises(ValueError):
        CrateTaskConfig.from_dict({"carry": {field: value}})


def test_carry_cli_override_reaches_round_context(monkeypatch, tmp_path):
    cfg = CrateTaskConfig.from_dict({"carry": {"settle_seconds": 0.5}})
    path = tmp_path / "config.json"
    cfg.dump(path)
    assert CrateTaskConfig.load(path) == cfg
    assert CrateTaskConfig.from_dict({}).carry == DEFAULT_CARRY
    meta = tmp_path / "scene.json"
    meta.write_text('{"target_object": "blue_crate"}')
    monkeypatch.setattr(carry.paths, "SCENE_META", meta)
    monkeypatch.setattr(carry.sys, "argv", ["carry", "--config", str(path), "--set", "carry.settle_seconds=0.4"])
    monkeypatch.setattr(carry, "Primitives", Mock())
    monkeypatch.setattr(carry, "EpisodeControlClient", Mock())
    monkeypatch.setattr(carry.round_runner, "prime", Mock())
    seen = []

    def reset(ctx):
        seen.append(ctx.config.settle_seconds)
        return ctx.m

    def rounds(p, reset, meta, *, run_round, **kwargs):
        run_round(p, reset, meta, 0, None, None, "fixture")

    monkeypatch.setattr(carry, "_phase_reset", reset)
    monkeypatch.setattr(carry.round_runner, "run_rounds", rounds)
    carry.main()
    assert seen == [0.4]


def test_carry_sent_reference_acceleration_is_bounded_on_reversal(monkeypatch):
    arms, p, clock = controller(monkeypatch)
    arms.start(q17=np.zeros(17))
    p.cmd.reset_mock()
    previous = np.zeros(17)
    velocity = np.zeros(17)
    for tick in range(40):
        target = arms.arm.q.copy()
        target[15] = 0.5 if tick < 20 else -0.5
        arms.push(target)
        sent = np.asarray(p.cmd.set_upper_body.call_args.args[0])
        actual_velocity = (sent - previous) / 0.02
        assert abs(actual_velocity[3]) <= 0.7 + 5e-6, "unbounded shoulder speed on the actual wire"
        assert abs(actual_velocity[3] - velocity[3]) <= 2.0 * 0.02 + 5e-6
        previous, velocity = sent.copy(), actual_velocity


def test_walk_to_reach_preserves_velocity_and_uses_elapsed_time(monkeypatch):
    arms, p, clock = controller(monkeypatch)
    arms.start(q17=np.zeros(17), settle_seconds=0)
    target = np.zeros(17)
    target[[2, 3, 4]] = [0.3, 0.5, 0.5]
    for _ in range(10):
        arms.motion.push(target, clock.now)
        clock.sleep(0.02)
    reference = arms.motion.reference
    previous = arms.motion.q.astype(float)
    velocity = np.asarray(p.cmd.set_upper_body_vel.call_args.args[0], dtype=float)
    assert np.max(velocity) > 0.2
    p.cmd.reset_mock()
    arms.start(q17=previous, settle_seconds=0)
    assert arms.motion.reference is reference
    p.cmd.set_upper_body.assert_not_called()  # no second ownership/zero-velocity packet
    old_dt = 0.02
    for dt in (0.02, 0.05, 0.03, 0.08) * 10:
        # Backend sleeps until the next 20 ms tick; supply the remaining delay.
        clock.now = arms.motion.last_time + dt
        q = reference.joints.copy()
        q[[14, 15, 22]] = [-0.1, -0.5, -0.5]
        arms.push(q)
        sent = np.asarray(p.cmd.set_upper_body.call_args.args[0], dtype=float)
        rate = (sent - previous) / dt
        np.testing.assert_allclose(p.cmd.set_upper_body_vel.call_args.args[0], rate, atol=5e-6)
        assert np.max(np.abs(rate)) <= DEFAULT_CARRY.joint_speed + 5e-6
        assert np.max(np.abs(rate - velocity)) <= DEFAULT_CARRY.joint_acceleration * (dt + old_dt) / 2 + 5e-6
        previous, velocity, old_dt = sent, rate, dt


@pytest.mark.parametrize("distance,valid", [(1.19, False), (1.51, False), (1.3, True)])
@pytest.mark.parametrize("reach_control", ["planner", "whole_body"])
def test_reset_drift_is_checked_before_recording_or_walking(monkeypatch, distance, valid, reach_control):
    from dataclasses import replace

    from locomani.backends.sonic import streamed_arm

    _, p, _ = controller(monkeypatch)
    state = p.obs()
    state["fallen"] = False
    state["pelvis"]["pos"][0] = -distance
    p.obs = lambda: state
    p.pelvis_yaw = lambda state: 0
    monkeypatch.setattr(phases.upper_ref, "shared", Mock())
    monkeypatch.setattr(carry, "hand_txt", lambda *args: "fixture")
    reset = Mock()
    events = []
    reset.reset.side_effect = lambda **kw: events.append("physics_reset")
    p._facing = lambda state: [1, 0, 0]

    def reanchor(p):
        events.append("reanchor")
        p.cmd.heading_offset = 0.37

    anchor = Mock(side_effect=reanchor)
    monkeypatch.setattr(streamed_arm, "reanchor", anchor)
    reset.start_video.side_effect = lambda *args: events.append("video")
    ctx = phases.CarryRoundContext(
        p=p,
        reset=reset,
        meta={
            "surfaces": {"A": {"xy": [0.5, 0], "flat_half_extent": [0.5, 0.5]}},
            "objects": {"blue_crate": {"init_pos": [0.2, 0, 0.7], "init_quat": [1, 0, 0, 0], "surface": "A"}},
            "robot_start": {"xy": [-1.3, 0]},
        },
        rnd=0,
        rng=np.random.default_rng(11),
        log=lambda msg: None,
        video_label="fixture",
        m={},
        config=replace(DEFAULT_CARRY, reach_control=reach_control),
    )
    result = phases._phase_reset(ctx)
    assert (result is None) == valid
    assert reset.start_video.called == valid
    assert ctx.m["initial_edge_distance"] == distance
    if reach_control == "whole_body":
        assert events[:2] == ["physics_reset", "reanchor"]
        assert p.cmd.heading_offset == 0.37
    else:
        anchor.assert_not_called()
    if not valid:
        assert "initial table-edge distance" in result["why"]
