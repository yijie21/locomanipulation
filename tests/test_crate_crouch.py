from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_fixture import scene_ik

from locomani.backends.sonic import streamed_arm
from locomani.backends.sonic.agent_server import LocomotionMode
from locomani.tasks.crate import bimanual_carry as carry
from locomani.tasks.crate import carry_crouch
from locomani.tasks.crate.carry_config import CrateCarryConfig
from locomani.tasks.crate.carry_crouch import HeightReference
from locomani.tasks.crate.config import CrateTaskConfig


def test_height_feedback_corrects_an_offset_without_an_instant_height_jump():
    height = HeightReference(0.53, 0.04, 0.78, 0)
    measured = 0.78
    previous = height.command
    for i in range(1, 701):
        command = height.step(i * 0.02, measured)
        assert abs(command - previous) <= 0.04 * 0.02 + 1e-12
        # A lagging plant delivers 5 cm above the planner command.
        measured += 0.02 / 0.3 * (command + 0.05 - measured)
        previous = command
    assert abs(measured - 0.53) < 0.003
    assert abs(command - 0.48) < 0.003
    assert height.step(14, measured) == command  # duplicate observation
    assert height.step(20, measured) == command  # a gap must not jump the height


@pytest.mark.parametrize(
    "field,value", [("pelvis_height", 0.3), ("pelvis_height", True), ("waist_pitch_max", 0.6), ("height_speed", 0)]
)
def test_invalid_low_grasp_profile_is_rejected(field, value):
    with pytest.raises(ValueError):
        CrateTaskConfig.from_dict({"carry": {field: value}})


def test_profile_and_explicit_override_deliver_low_height(tmp_path):
    config = CrateTaskConfig.from_dict({"carry": {"pelvis_height": 0.53, "waist_pitch_max": 0.52}})
    path = tmp_path / "low.json"
    config.dump(path)
    restored = CrateTaskConfig.load(path).with_overrides(["carry.pelvis_height=null", "carry.waist_pitch_max=0.4"])
    assert restored.carry == CrateCarryConfig()
    assert CrateTaskConfig().with_overrides(["carry.pelvis_height=0.53"]).carry.pelvis_height == 0.53


def test_actual_planner_packet_keeps_both_arms_and_squat_through_grasp_and_end(monkeypatch):
    ik = scene_ik()
    state = {"wall": 0.0, "joints": np.zeros(29), "pelvis": {"pos": [0, 0, 0.78], "quat": [1, 0, 0, 0]}}
    command = Mock(left_hand=np.zeros(7), right_hand=np.zeros(7))
    p = SimpleNamespace(cmd=command, obs=lambda: state, _facing=lambda s: (1, 0, 0))
    monkeypatch.setattr(carry, "PlannerArm", lambda *a, **kw: streamed_arm.PlannerArm(*a, **kw, ik=ik))
    monkeypatch.setattr(streamed_arm.time, "sleep", lambda seconds: None)
    config = CrateCarryConfig(pelvis_height=0.53, waist_pitch_max=0.52)
    arms = carry.BimanualArms(p, config=config)
    arms.start(q17=np.zeros(17), settle_seconds=0)
    assert arms.solver.hi[14] == min(0.52, ik.model.jnt_range[ik.jids[14], 1])
    arms.height_control = HeightReference(config.pelvis_height, config.height_speed, 0.78, 0)
    for tick in range(1, 21):
        state["wall"] = tick * 0.02
        target = arms.arm.q.copy()
        target[[15, 22]] = [0.2, 0.3]
        arms.push(target)
    packet = command.frame.call_args.kwargs
    assert packet["mode"] == LocomotionMode.IDEL_SQUAT
    assert 0.53 < packet["height"] < 0.78
    assert len(command.set_upper_body.call_args.args[0]) == 17
    assert len(command.set_upper_body_vel.call_args.args[0]) == 17
    assert set(command.set_hands.call_args.kwargs) == {"left", "right"}
    arms.arm.stand = Mock()
    arms.stop()
    arms.arm.stand.assert_not_called()  # no unmeasured stand-up with a held crate


def test_crouch_cannot_admit_grasp_when_height_is_not_delivered(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(carry_crouch.time, "time", lambda: now[0])
    state = dict(
        wall=0.0,
        fallen=False,
        pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]),
        objects={"blue_crate": {"pos": [0, 0, 0.385]}},
    )

    def observe():
        return dict(state, wall=now[0])

    p = SimpleNamespace(obs=observe)
    arm = SimpleNamespace(q=np.zeros(29), height=-1)
    arms = SimpleNamespace(
        arm=arm,
        solver=SimpleNamespace(sides=("left", "right")),
        motion=SimpleNamespace(quiet=True),
        grip_point=lambda s, side: (np.array([0, 0.2 if side == "left" else -0.2, 0.8]), np.eye(3)),
        solve=lambda *args, **kwargs: np.zeros(29),
    )

    def push(q):
        now[0] += 0.02
        arm.height = arms.height_control.step(now[0], 0.78)

    arms.push = push
    monkeypatch.setattr(
        carry,
        "crate_frame",
        lambda *args: (
            np.array([0, 0, 0.385]),
            dict(left=np.array([0, 0.2, 0.445]), right=np.array([0, -0.2, 0.445])),
            np.array([1, 0]),
            np.array([0, 1]),
            0,
        ),
    )
    ctx = SimpleNamespace(
        p=p, arms=arms, config=CrateCarryConfig(pelvis_height=0.53), m={"why": ""}, log=lambda msg: None
    )
    result = carry_crouch._phase_crouch(ctx)
    assert not result["crouch_ready"]
    assert result["pelvis_height_measured"] == 0.78
    assert result["planner_height_command"] == 0.4
    assert result["why"] == "[crouch] height or pose did not settle"
