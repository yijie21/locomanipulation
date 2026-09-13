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
    return carry.BimanualArms(p), p, clock


def test_stable_brake_does_not_add_a_fixed_one_second_wait(monkeypatch):
    _, p, clock = controller(monkeypatch)
    ctx = SimpleNamespace(
        p=p,
        ramp=SimpleNamespace(push=lambda q: 0.0),
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


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_carry_setting_is_rejected(value):
    with pytest.raises(ValueError):
        CrateTaskConfig.from_dict({"carry": {"settle_seconds": value}})


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
