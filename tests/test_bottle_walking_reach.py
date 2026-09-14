from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.primitives import upper_body_from_mj
from locomani.tasks.bottle.reach_config import ReachConfig
from locomani.tasks.bottle.ready_grasp_policy import ready_q
from locomani.tasks.bottle.walking_reach import WalkingReach
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig


def state(t, pitch=0):
    return dict(
        t=t,
        joints=ready_q(),
        pelvis=dict(
            pos=[0, 0, 0.79],
            quat=Rotation.from_euler("y", pitch).as_quat()[[3, 0, 1, 2]],
        ),
    )


def test_gait_is_filtered_before_slow_planning_but_real_target_changes_are_followed():
    cfg = WholeBodyConfig()
    controller = WalkingReach(scene_ik(), cfg, upper_body_from_mj(ready_q()))
    raw, filtered, updates = [], [], []
    previous_q, previous_v = controller.q.copy(), controller.v.copy()
    for t in np.arange(0, 8, 0.02):
        sample = state(t, 0.10 * np.sin(2 * np.pi * 2 * t))
        target = np.array([0.40, -0.18 if t < 5 else -0.26, 0.70])
        q, velocity = controller.step(sample, target)
        if t > 0:
            np.testing.assert_allclose((q - previous_q) / 0.02, velocity, atol=1e-9)
            assert max(abs(velocity - previous_v)) <= 0.0200001
        assert max(abs(velocity)) <= 0.4500001
        np.testing.assert_array_equal(q[:4], upper_body_from_mj(ready_q())[:4])
        if controller.diagnostics["plan_updated"]:
            updates.append(t)
        if 2 < t < 4:
            raw.append(
                (Rotation.from_quat(sample["pelvis"]["quat"][[1, 2, 3, 0]]).inv().apply(target - [0, 0, 0.79]))[2]
            )
            filtered.append(controller.relative[2])
        previous_q, previous_v = q, velocity
    assert np.std(filtered) < np.std(raw) * 0.25
    assert min(np.diff(updates)) >= 1 / cfg.walk_plan_hz - 1e-8
    assert abs(controller.relative[1] + 0.26) < 0.001
    assert np.linalg.norm(controller.q - upper_body_from_mj(ready_q())) > 0.1


def test_cold_camera_preserves_consumed_reference_and_stale_images_stop_preparation():
    controller = WalkingReach(scene_ik(), WholeBodyConfig(), upper_body_from_mj(ready_q()))
    initial = controller.q.copy()
    for t in np.arange(0, 0.3, 0.02):
        q, v = controller.step(state(t), None)
        np.testing.assert_array_equal(q, initial)
        np.testing.assert_array_equal(v, np.zeros(17))
    controller.step(state(0.3), [0.4, -0.18, 0.7])
    planned = controller.planned
    for t in np.arange(0.32, 1.0, 0.02):
        controller.step(state(t), None)
    assert controller.planned == planned
    assert max(abs(controller.v)) < 1e-6
    q = controller.q.copy()
    controller.step(state(t), [99, 99, 99])
    np.testing.assert_array_equal(controller.q, q)
    with pytest.raises(ValueError, match="clock gap"):
        controller.step(state(t + 0.6), [0.4, -0.18, 0.7])


@pytest.mark.parametrize(
    "strategy,expected", [("whole_body", ["prepare", "walk", "arm"]), ("coupled", ["walk", "prepare"])]
)
def test_actual_navigation_attaches_continuous_controller_before_first_walk(strategy, expected):
    from locomani.tasks.bottle.round_phases import _phase_navigate

    calls = []
    controller = SimpleNamespace(walk_tick=lambda _: calls.append("arm"))

    def prepare(ctx, **_):
        calls.append("prepare")
        ctx.coordinated = controller

    def walk(*_, on_tick=None, **__):
        calls.append("walk")
        if on_tick is not None:
            on_tick({}, 1)
        return True

    ctx = SimpleNamespace(
        p=SimpleNamespace(obs=lambda: {}, pelvis_xy=lambda _: [-3, 0], walk_to=walk),
        execution=SimpleNamespace(enter=lambda _: None),
        reach_config=ReachConfig(strategy=strategy),
        rel=[0.3, -0.18],
        spawn_xy=[0, 0],
        raise_lead=1,
        fv=[1, 0, 0],
        spawn_off=1.2,
        psi=0,
    )
    with patch("locomani.tasks.bottle.round_phases._prepare_grasp_policy", side_effect=prepare):
        assert _phase_navigate(ctx) is None
    assert calls == expected


def test_forward_travel_prediction_avoids_filter_delay_at_the_table():
    controller = WalkingReach(scene_ik(), WholeBodyConfig(), upper_body_from_mj(ready_q()))
    for t in np.arange(0, 4, 0.02):
        sample = state(t)
        sample["pelvis"]["pos"] = [0.3 * t, 0, 0.79]
        sample["pelvis"]["vel"] = [0.3, 0, 0]
        controller.step(sample, [1.6, -0.18, 0.7])
    # A plain 0.45-second low-pass would lag by 13.5 cm at this speed.
    assert abs(controller.relative[0] - (1.6 - 0.3 * t)) < 0.01


def test_arbitrary_takeover_reference_is_preserved_then_changes_without_a_step():
    initial = upper_body_from_mj(ready_q()) + np.linspace(-0.15, 0.15, 17)
    controller = WalkingReach(scene_ik(), WholeBodyConfig(), initial)
    q, v = controller.step(state(0), [0.4, -0.18, 0.7])
    np.testing.assert_array_equal(q, initial)
    np.testing.assert_array_equal(v, np.zeros(17))
    q, v = controller.step(state(0.02), [0.4, -0.18, 0.7])
    assert max(abs(q - initial)) <= 0.000400001


def test_walk_callback_rejection_stops_before_another_forward_command():
    from unittest.mock import Mock

    from locomani.backends.sonic.agent_server import LocomotionMode
    from locomani.backends.sonic.primitives_walk import PrimitivesWalkMixin

    cmd = Mock()
    robot = SimpleNamespace(obs=lambda: {"fallen": False}, pelvis_xy=lambda _: np.zeros(2), cmd=cmd)
    assert not PrimitivesWalkMixin.walk_to(robot, [1, 0], on_tick=lambda *_: False)
    cmd.frame.assert_called_once_with(mode=LocomotionMode.IDLE, speed=0.0, movement=(0, 0, 0))
