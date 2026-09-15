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
    cfg = WholeBodyConfig(raise_trigger_px=0)
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
        assert max(abs(velocity)) <= cfg.walk_joint_speed + 1e-6
        assert np.all(
            np.abs(q[:3]) <= np.array([0.1, 0.1, 0.12]) + 1e-9
        )  # the waist moves only inside the walking lean envelope
        np.testing.assert_array_equal(q[3], upper_body_from_mj(ready_q())[3])  # the left arm keeps READY
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


def test_wrist_turns_at_once_and_the_hand_waits_beside_the_hip_for_the_bottle():
    """Crate-style preparation: fingers forward from the first plan with the tool point held
    `walk_clearance` above the grip beside the hip (pelvis-frame x constant, like the crate's
    stand_off clamp); the hand follows the bottle laterally, never advances past HAND_X, and
    stays the entry gap behind a bottle the base has overrun; the waist stays in the envelope."""
    from locomani.tasks.bottle.walking_reach import HAND_X, WAIST_ENVELOPE

    cfg = WholeBodyConfig(raise_trigger_px=0)
    ik = scene_ik()
    hanging = upper_body_from_mj(ready_q())
    hanging[4::2] = [0.0, -0.27, 0.64, 1.02, -0.07, 0.22, -0.15]  # the right arm as the walk starts (recorded)
    far = WalkingReach(ik, cfg, hanging, brake_depth=0.31)
    for t in np.arange(0, 4.0, 0.02):
        far.step(state(t), [1.5, -0.20, 0.70])
    np.testing.assert_allclose(far.diagnostics["goal"], [HAND_X, -0.20, 0.70 - 0.79 + cfg.walk_clearance], atol=1e-9)
    assert far.diagnostics["ik_error"] < 0.01 and far.diagnostics["rot_error"] < 0.2
    pos, rot = ik.fk(far.seed, "right")
    assert rot[2, 0] > -0.2  # the solved palm x axis (finger direction) is no longer pointing down
    assert np.all(far.seed[12:15] >= WAIST_ENVELOPE[0] - 1e-9) and np.all(far.seed[12:15] <= WAIST_ENVELOPE[1] + 1e-9)
    overrun = WalkingReach(ik, cfg, hanging, brake_depth=0.31)
    for t in np.arange(0, 4.0, 0.02):
        overrun.step(state(t), [0.15, -0.20, 0.70])
    assert overrun.diagnostics["goal"][0] <= 0.15 - 0.12 + 1e-9  # behind the bottle by the entry gap, not at HAND_X


def test_cold_camera_preserves_consumed_reference_and_stale_images_stop_preparation():
    controller = WalkingReach(scene_ik(), WholeBodyConfig(raise_trigger_px=0), upper_body_from_mj(ready_q()))
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
    controller = WalkingReach(scene_ik(), WholeBodyConfig(raise_trigger_px=0), upper_body_from_mj(ready_q()))
    for t in np.arange(0, 4, 0.02):
        sample = state(t)
        sample["pelvis"]["pos"] = [0.3 * t, 0, 0.79]
        sample["pelvis"]["vel"] = [0.3, 0, 0]
        controller.step(sample, [1.6, -0.18, 0.7])
    # A plain 0.45-second low-pass would lag by 13.5 cm at this speed.
    assert abs(controller.relative[0] - (1.6 - 0.3 * t)) < 0.01


def test_arbitrary_takeover_reference_is_preserved_then_changes_without_a_step():
    initial = upper_body_from_mj(ready_q()) + np.linspace(-0.15, 0.15, 17)
    controller = WalkingReach(scene_ik(), WholeBodyConfig(raise_trigger_px=0), initial)
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


def test_arm_waits_until_the_bottle_subtends_the_trigger_in_the_ego_view():
    """The raise trigger is a visible quantity: below raise_trigger_px the reference is held, at or
    above it preparation starts and latches; the waist envelope carries no roll."""
    from locomani.tasks.bottle.walking_reach import WAIST_ENVELOPE

    assert WAIST_ENVELOPE[0][1] == 0 and WAIST_ENVELOPE[1][1] == 0
    cfg = WholeBodyConfig(raise_trigger_px=36.0)
    hanging = upper_body_from_mj(ready_q())
    hanging[4::2] = [0.0, -0.27, 0.64, 1.02, -0.07, 0.22, -0.15]
    c = WalkingReach(scene_ik(), cfg, hanging, brake_depth=0.31)
    initial = c.q.copy()
    for t in np.arange(0, 1.0, 0.02):
        c.step(state(t), [1.5, -0.20, 0.70], ego_px=30.0)
    assert not c.raised and np.array_equal(c.q, initial)
    for t in np.arange(1.0, 3.0, 0.02):
        c.step(state(t), [1.5, -0.20, 0.70], ego_px=40.0 if t < 1.1 else 30.0)  # fires once, then latches
    assert c.raised and max(abs(c.q - initial)) > 0.1


def test_natural_preparation_selects_a_feasible_stop_and_pregrasp_with_real_ik():
    cfg = WholeBodyConfig(raise_trigger_px=0, natural_walk_prep=True)
    ik = scene_ik()
    checked = []

    def clearance(_state, joints, candidate_base):
        position, _ = ik.fk(joints, "right")
        checked.append((joints.copy(), candidate_base.copy(), position.copy()))
        return dict(clear=True, positive_clearance=True, required_margin=0.004, checks=[])

    controller = WalkingReach(
        ik,
        cfg,
        upper_body_from_mj(ready_q()),
        brake_depth=0.31,
        brake_edge=0.27,
        feasibility=clearance,
    )
    controller.step(state(0.0), [1.20, -0.20, 0.70], edge_distance=1.10)
    geometry = controller.diagnostics["chosen_geometry"]

    assert geometry["feasible"]
    assert geometry["fallback_reason"] is None
    assert geometry["goal"][0] != pytest.approx(0.08)
    assert geometry["stop_depth"] >= 0.31 - cfg.stance_search_span
    assert geometry["stop_edge"] >= 0.27 - 1e-9
    assert geometry["ik_error"] <= cfg.stance_ik_tolerance
    assert 1 <= len(checked) <= cfg.stance_candidates
    assert checked[0][1].shape == (7,)
    selected_edge = controller.chosen_stop_edge
    controller.step(state(0.2), [1.20, -0.20, 0.70], edge_distance=None)
    assert controller.chosen_stop_edge == pytest.approx(selected_edge)


def test_infeasible_natural_search_holds_the_incoming_arm_and_reports_fallback():
    cfg = WholeBodyConfig(raise_trigger_px=0, natural_walk_prep=True)
    initial = upper_body_from_mj(ready_q())
    controller = WalkingReach(
        scene_ik(),
        cfg,
        initial,
        feasibility=lambda *_: dict(clear=False, reason="body:table", checks=[]),
    )
    for t in np.arange(0, 0.5, 0.02):
        q, velocity = controller.step(state(t), [1.0, -0.20, 0.70], edge_distance=0.9)

    np.testing.assert_allclose(q, initial)
    np.testing.assert_allclose(velocity, 0)
    geometry = controller.diagnostics["chosen_geometry"]
    assert not geometry["feasible"]
    assert geometry["fallback_reason"] == "no_candidate_passed_ik_and_swept_clearance"


def test_wrist_roi_requires_consecutive_frames_and_uses_projected_camera_basis():
    cfg = WholeBodyConfig(
        natural_walk_prep=True,
        wrist_roi_supervisor=True,
        raise_trigger_px=36,
        raise_release_px=28,
        vision_confirm_frames=2,
    )
    controller = WalkingReach(
        scene_ik(),
        cfg,
        upper_body_from_mj(ready_q()),
        feasibility=lambda *_: dict(clear=True, checks=[]),
    )

    def camera(stamp, *, roi=True):
        result = dict(image_timestamps={"ego_view": stamp, "right_wrist": stamp}, visible={"ego_view": stamp})
        if roi:
            result["visible"]["right_wrist"] = stamp
            result["wrist_roi"] = dict(
                timestamp=stamp,
                centroid=[0.8, 0.5],
                extent=[0.2, 0.3],
                quality=0.5,
                correction_basis_pelvis=np.array([[0, 0], [1, 0], [0, 1]]),
            )
        return result

    controller.step(state(0.0), [1.2, -0.2, 0.7], camera=camera(1.0), ego_px=40, edge_distance=1.1)
    assert not controller.raised and not controller.wrist_acquired
    controller.step(state(0.1), [1.2, -0.2, 0.7], camera=camera(1.1, roi=False), ego_px=40, edge_distance=1.1)
    assert controller.diagnostics["wrist_confirm_frames"] == 0
    for t, stamp in ((0.2, 1.2), (0.3, 1.3)):
        controller.step(state(t), [1.2, -0.2, 0.7], camera=camera(stamp), ego_px=40, edge_distance=1.1)

    assert controller.raised and controller.wrist_acquired
    assert controller.diagnostics["raise_reason"] == "ego_size_stable"
    assert controller.diagnostics["roi_correction"][1] > 0
    assert controller.diagnostics["roi_correction"][2] == pytest.approx(0)
    stopped_raise = controller.diagnostics["raise_offset"]
    controller.step(state(0.4), [1.2, -0.2, 0.7], camera=camera(1.4), ego_px=40, edge_distance=1.1)
    assert controller.diagnostics["raise_offset"] == pytest.approx(stopped_raise)


def test_natural_stop_lead_latches_measured_speed_and_keeps_raw_entry_gap():
    cfg = WholeBodyConfig(raise_trigger_px=0, natural_walk_prep=True)
    controller = WalkingReach(
        scene_ik(),
        cfg,
        upper_body_from_mj(ready_q()),
        brake_depth=0.31,
        brake_edge=0.27,
        feasibility=lambda *_: dict(clear=True, checks=[]),
    )
    fast = state(0.0)
    fast["pelvis"]["vel"] = [0.45, 0, 0]
    controller.step(fast, [1.2, -0.2, 0.7], edge_distance=1.1)
    geometry = controller.diagnostics["chosen_geometry"]
    lead = controller.diagnostics["stop_lead"]

    assert lead == pytest.approx(cfg.walk_stop_lead_max)
    assert controller.diagnostics["chosen_stop_depth"] == pytest.approx(geometry["stop_depth"] + lead)
    assert geometry["command_gap"] == pytest.approx(cfg.pregrasp_gap_max + cfg.walk_tracking_gap)
    assert geometry["goal"][0] == pytest.approx(geometry["stop_depth"] - geometry["command_gap"])

    slow = state(0.2)
    slow["pelvis"]["vel"] = [0.02, 0, 0]
    controller.step(slow, [0.5, -0.2, 0.7], edge_distance=0.4)
    assert controller.diagnostics["stop_lead"] == pytest.approx(lead)
    heading = controller.root.T @ controller.rotation[:, 0]
    commanded = float(np.asarray(controller.diagnostics["goal"]) @ heading)
    raw = float(np.asarray(controller.diagnostics["raw_relative"]) @ heading)
    assert raw - commanded >= geometry["command_gap"] + lead - 1e-8
