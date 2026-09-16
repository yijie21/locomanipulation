"""Focused contracts for bottle tracking dynamics and whole-body posture costs."""

from dataclasses import replace

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.primitives.supported_motion import (
    SupportedMotion,
    bounded_correction_step,
    residual_deadband,
)
from locomani.primitives.whole_body_ik import WholeBodyIK
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig
from locomani.tasks.bottle.whole_body_policy import (
    CaptureSettleGate,
    begin_live_translation_hold,
    entry_step,
    freeze_rotation_reference,
    terminal_entry_speed,
)


def supported_state():
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    return dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))


def _capture_gate():
    return CaptureSettleGate(0.10, 6, 0.004, 0.03, 0.035, 0.20)


def test_capture_window_admits_bounded_high_frequency_motion_and_arbitrary_yaw():
    gate = _capture_gate()
    statuses = []
    for index, stamp in enumerate((0.0, 0.019, 0.041, 0.060, 0.082, 0.101)):
        sign = 1 if index % 2 == 0 else -1
        position = [0.008 + sign * 0.001, 0.0, 0.0]
        rotation = Rotation.from_euler("z", 1.2).as_matrix() @ Rotation.from_rotvec([0.0, sign * 0.01, 0.0]).as_matrix()
        statuses.append(gate.step(stamp, True, position, rotation))

    assert not any(status["ready"] for status in statuses[:-1])
    admitted = statuses[-1]
    assert admitted["ready"]
    assert admitted["sample_count"] == 6
    assert admitted["held_seconds"] == pytest.approx(0.101)
    assert admitted["position_diameter_m"] == pytest.approx(0.002)
    assert admitted["relative_drift_m_s"] <= 0.03
    assert admitted["rotation_diameter_rad"] == pytest.approx(0.02)
    assert admitted["angular_drift_rad_s"] <= 0.20


def test_capture_window_rejects_fast_crossing_and_large_zero_drift_oscillation():
    crossing = _capture_gate()
    for index, stamp in enumerate(np.linspace(0.0, 0.10, 6)):
        crossed = crossing.step(stamp, True, [0.012 - 0.0018 * index, 0.0, 0.0], np.eye(3))
    assert not crossed["ready"]
    assert crossed["position_diameter_m"] == pytest.approx(0.009)
    assert crossed["relative_drift_m_s"] == pytest.approx(0.09)

    oscillating = _capture_gate()
    positions = (0.003, -0.003, 0.003, -0.003, 0.003, -0.003)
    angles = (0.02, -0.02, 0.02, -0.02, 0.02, -0.02)
    for stamp, position, angle in zip(np.linspace(0.0, 0.10, 6), positions, angles, strict=True):
        rejected = oscillating.step(
            stamp,
            True,
            [position, 0.0, 0.0],
            Rotation.from_rotvec([0.0, angle, 0.0]).as_matrix(),
        )
    assert not rejected["ready"]
    assert rejected["position_diameter_m"] == pytest.approx(0.006)
    assert rejected["rotation_diameter_rad"] == pytest.approx(0.04)


def test_capture_window_clears_on_geometry_exit_and_ignores_repeated_time():
    gate = _capture_gate()
    identity = np.eye(3)
    gate.step(0.00, True, [0.008, 0.0, 0.0], identity)
    repeated = gate.step(0.00, True, [0.009, 0.0, 0.0], identity)
    assert repeated["sample_count"] == 1
    assert repeated["held_seconds"] == 0.0

    gate.step(0.02, True, [0.008, 0.0, 0.0], identity)
    cleared = gate.step(0.04, False, [0.008, 0.0, 0.0], identity)
    assert cleared["sample_count"] == 0
    assert not cleared["ready"]

    statuses = [
        gate.step(stamp, True, [0.008, 0.0, 0.0], identity) for stamp in (0.06, 0.081, 0.099, 0.122, 0.141, 0.161)
    ]
    assert not any(status["ready"] for status in statuses[:-1])
    assert statuses[-1]["ready"]
    with pytest.raises(ValueError, match="clock moved backwards"):
        gate.step(0.15, True, [0.008, 0.0, 0.0], identity)


def test_capture_window_keeps_six_samples_across_slower_irregular_control_ticks():
    gate = _capture_gate()
    statuses = [
        gate.step(stamp, True, [0.008, 0.0, 0.0], np.eye(3)) for stamp in (0.0, 0.037, 0.081, 0.119, 0.162, 0.201)
    ]

    assert not any(status["ready"] for status in statuses[:-1])
    assert statuses[-1]["ready"]
    assert statuses[-1]["sample_count"] == 6
    assert statuses[-1]["held_seconds"] == pytest.approx(0.201)


def test_terminal_entry_decelerates_at_active_goal_and_keeps_corrections_continuous():
    assert terminal_entry_speed(0.10, 0.07, 0.20) == pytest.approx(0.07)
    assert terminal_entry_speed(0.0001, 0.024, 0.20) == pytest.approx(np.sqrt(0.00004))
    assert terminal_entry_speed(0.0, 0.024, 0.20) == 0.0

    axis = np.array([1.0, 0.0, 0.0])
    target = np.zeros(3)
    centred = np.array([-0.0012, 0.0, 0.0])
    held, _ = entry_step(
        centred,
        centred,
        target,
        axis,
        0.02,
        0.16,
        0.0012,
        brake_acceleration=0.20,
    )
    np.testing.assert_array_equal(held, centred)

    outside = np.array([-0.0092, 0.0, 0.0])
    resumed, _ = entry_step(
        outside,
        outside,
        target,
        axis,
        0.02,
        0.16,
        0.0012,
        brake_acceleration=0.20,
    )
    assert resumed[0] > outside[0]
    assert resumed[0] < -0.0012

    centred_with_lateral_error = np.array([-0.0012, 0.01, 0.0])
    corrected, _ = entry_step(
        centred_with_lateral_error,
        centred_with_lateral_error,
        target,
        axis,
        0.02,
        0.16,
        0.0012,
        brake_acceleration=0.20,
    )
    assert corrected[1] < centred_with_lateral_error[1]

    too_deep = np.array([0.003, 0.0, 0.0])
    retreated, _ = entry_step(
        too_deep,
        too_deep,
        target,
        axis,
        0.02,
        0.16,
        0.0012,
        brake_acceleration=0.20,
    )
    assert retreated[0] < too_deep[0]
    assert np.linalg.norm(retreated - too_deep) <= 0.16 * 0.02 + 1e-12


def test_bounded_correction_attenuates_tick_oscillation_and_obeys_physical_rate_limits():
    dt = 0.02
    value = np.zeros((1, 3))
    rate = np.zeros_like(value)
    values, rates = [], []
    raw = []
    for index in range(100):
        disturbance = np.array([[0.008 if index % 2 else -0.008, 0, 0]])
        target = residual_deadband(disturbance, 0.002)
        value, rate = bounded_correction_step(value, rate, target, dt, 0.04, 0.20)
        raw.append(target.copy())
        values.append(value.copy())
        rates.append(rate.copy())

    values, rates, raw = np.asarray(values), np.asarray(rates), np.asarray(raw)
    assert np.std(values[:, 0, 0]) < 0.08 * np.std(raw[:, 0, 0])
    assert np.max(np.linalg.norm(rates, axis=-1)) <= 0.04 + 1e-12
    accelerations = np.diff(rates[:, 0], axis=0) / dt
    assert np.max(np.linalg.norm(accelerations, axis=-1)) <= 0.20 + 1e-12


def test_bounded_correction_has_similar_state_at_equal_physical_time_for_changed_clock_rate():
    def run(dt):
        value = np.zeros((1, 3))
        rate = np.zeros_like(value)
        for _ in range(round(1.0 / dt)):
            value, rate = bounded_correction_step(value, rate, [[0.03, 0, 0]], dt, 0.04, 0.20)
        return value, rate

    fine_value, fine_rate = run(0.01)
    coarse_value, coarse_rate = run(0.04)
    np.testing.assert_allclose(fine_value, coarse_value, atol=0.0015)
    np.testing.assert_allclose(fine_rate, coarse_rate, atol=0.006)


def test_supported_motion_keeps_takeover_bias_and_constant_offset_with_dynamics_enabled():
    ik = scene_ik()
    state = supported_state()
    cfg = replace(WholeBodyConfig(), tracking_correction_dynamics=True)
    motion = SupportedMotion(
        ik,
        state,
        cfg,
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
        hand_weights=(0.35, 1),
    )
    rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    bias = np.array([[0.012, -0.004, 0.003], [-0.015, 0.002, -0.005]])

    for _ in range(6):
        tools = motion.body.tools(motion.x).copy()
        motion.step(tools, rotations, tools - bias, 0.02, measured_rotations=rotations)
        np.testing.assert_allclose(motion.feedback, bias, atol=1e-12)
    np.testing.assert_allclose(motion.feedback_rate, 0, atol=1e-12)
    assert motion.diagnostics["tracking_correction_dynamics"] is True


def test_capture_brakes_translation_at_measured_tools_and_lift_resumes_continuously():
    ik = scene_ik()
    state = supported_state()
    cfg = replace(WholeBodyConfig(), tracking_correction_dynamics=True)
    motion = SupportedMotion(
        ik,
        state,
        cfg,
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
        hand_weights=(0.35, 1),
    )
    rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    bias = np.array([[0.012, -0.004, 0.003], [-0.015, 0.002, -0.005]])
    tools = motion.body.tools(motion.x).copy()
    motion.step(tools, rotations, tools - bias, 0.02, measured_rotations=rotations)
    tools = motion.body.tools(motion.x).copy()
    changing_bias = bias + [[0.01, 0, 0], [-0.01, 0, 0]]
    motion.step(
        tools,
        rotations,
        tools - changing_bias,
        0.02,
        measured_rotations=rotations,
    )
    assert np.max(np.linalg.norm(motion.feedback_rate, axis=1)) > 0

    active_reference = motion.body.tools(motion.x).copy()
    state_before = motion.x.copy()
    velocity_before = motion.v.copy()
    measured_at_capture = active_reference - changing_bias
    close_target, previous_reference = motion.freeze_translation_at_measured_tools(measured_at_capture)

    np.testing.assert_allclose(previous_reference, active_reference, atol=1e-12)
    np.testing.assert_array_equal(close_target, measured_at_capture)
    corrected_brake_target = measured_at_capture + cfg.tracking_gain * motion.feedback
    assert np.max(np.linalg.norm(corrected_brake_target - active_reference, axis=1)) > 0
    np.testing.assert_array_equal(motion.x, state_before)
    np.testing.assert_array_equal(motion.v, velocity_before)
    np.testing.assert_array_equal(motion.feedback_rate, np.zeros((2, 3)))

    frozen_feedback = motion.feedback.copy()
    motion.step(
        close_target,
        rotations,
        active_reference - 0.05,
        0.02,
    )
    np.testing.assert_array_equal(motion.feedback, frozen_feedback)
    np.testing.assert_allclose(
        close_target + cfg.tracking_gain * motion.feedback,
        corrected_brake_target,
        atol=1e-12,
    )

    close_reference = motion.body.tools(motion.x).copy()
    lift_target = close_reference - cfg.tracking_gain * motion.feedback
    motion.resume_translation_feedback()
    corrected_before = lift_target + cfg.tracking_gain * motion.feedback
    np.testing.assert_allclose(corrected_before, close_reference, atol=1e-12)

    motion.step(lift_target, rotations, close_reference - 0.05, 0.02)
    corrected_after = lift_target + cfg.tracking_gain * motion.feedback
    correction_step = np.linalg.norm(corrected_after - corrected_before, axis=1)
    assert np.max(correction_step) <= cfg.tracking_translation_acceleration * 0.02**2 + 1e-12

    fresh = SupportedMotion(
        scene_ik(),
        state,
        cfg,
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
        hand_weights=(0.35, 1),
    )
    fresh_reference = fresh.body.tools(fresh.x).copy()
    first_measured = fresh_reference - bias
    first_target, first_reference = fresh.freeze_translation_at_measured_tools(first_measured)
    np.testing.assert_allclose(fresh.feedback_origin, bias, atol=1e-12)
    np.testing.assert_array_equal(first_target, first_measured)
    np.testing.assert_allclose(
        first_target + cfg.tracking_gain * fresh.feedback,
        first_reference,
        atol=1e-12,
    )

    fresh.resume_translation_feedback()
    corrected_before = first_target + cfg.tracking_gain * fresh.feedback
    fresh_rotations = fresh.body.ik.data.xmat[fresh.body.hands].reshape(2, 3, 3).copy()
    fresh.step(
        first_target,
        fresh_rotations,
        fresh_reference - changing_bias,
        0.02,
    )
    corrected_after = first_target + cfg.tracking_gain * fresh.feedback
    correction_step = np.linalg.norm(corrected_after - corrected_before, axis=1)
    assert np.max(correction_step) <= cfg.tracking_translation_acceleration * 0.02**2 + 1e-12


def test_rotation_feedback_freezes_close_hold_and_restarts_lift_from_zero_rate():
    ik = scene_ik()
    state = supported_state()
    cfg = replace(WholeBodyConfig(), tracking_correction_dynamics=True)
    motion = SupportedMotion(
        ik,
        state,
        cfg,
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
        hand_weights=(0.35, 1),
    )
    dt = 0.02

    tools = motion.body.tools(motion.x).copy()
    reference_rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    motion.step(tools, reference_rotations, tools, dt, measured_rotations=reference_rotations)

    tools = motion.body.tools(motion.x).copy()
    reference_rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    reach_error = np.array([[0.0, 0.06, 0.0], [0.0, -0.08, 0.0]])
    measured_rotations = Rotation.from_rotvec(-reach_error).as_matrix() @ reference_rotations
    motion.step(tools, reference_rotations, tools, dt, measured_rotations=measured_rotations)
    assert np.max(np.linalg.norm(motion.rotation_feedback_rate, axis=1)) > 0

    reference_rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    frozen_targets = freeze_rotation_reference(
        reference_rotations,
        motion.rotation_feedback,
        cfg.tracking_gain,
    )
    corrected_close = Rotation.from_rotvec(cfg.tracking_gain * motion.rotation_feedback).as_matrix() @ frozen_targets
    np.testing.assert_allclose(corrected_close, reference_rotations, atol=1e-12)

    feedback_at_close = motion.rotation_feedback.copy()
    motion.freeze_rotation_feedback()
    np.testing.assert_array_equal(motion.rotation_feedback_rate, np.zeros((2, 3)))
    tools = motion.body.tools(motion.x).copy()
    noisy_hold = Rotation.from_rotvec(-2 * reach_error).as_matrix() @ reference_rotations
    motion.step(tools, frozen_targets, tools, dt, measured_rotations=noisy_hold)
    np.testing.assert_array_equal(motion.rotation_feedback, feedback_at_close)
    np.testing.assert_array_equal(motion.rotation_feedback_rate, np.zeros((2, 3)))

    motion.resume_rotation_feedback()
    tools = motion.body.tools(motion.x).copy()
    live_reference = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    lift_error = np.array([[0.0, 0.12, 0.0], [0.0, -0.14, 0.0]])
    measured_lift = Rotation.from_rotvec(-lift_error).as_matrix() @ live_reference
    motion.step(tools, frozen_targets, tools, dt, measured_rotations=measured_lift)

    max_rate = cfg.tracking_rotation_acceleration * dt
    assert np.max(np.linalg.norm(motion.rotation_feedback_rate, axis=1)) <= max_rate + 1e-12
    feedback_step = np.linalg.norm(motion.rotation_feedback - feedback_at_close, axis=1)
    assert np.max(feedback_step) <= cfg.tracking_rotation_acceleration * dt**2 + 1e-12
    assert not motion.diagnostics["rotation_feedback_frozen"]


def test_translation_feedback_freeze_holds_value_without_moving_the_reference():
    ik = scene_ik()
    state = supported_state()
    cfg = replace(WholeBodyConfig(), tracking_correction_dynamics=True)
    motion = SupportedMotion(
        ik,
        state,
        cfg,
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
        hand_weights=(0.35, 1),
    )
    dt = 0.02
    tools = motion.body.tools(motion.x).copy()
    rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    motion.step(tools, rotations, tools, dt)
    # A steady tracking error makes the bounded compensation move.
    measured = tools + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.02]])
    for _ in range(5):
        tools = motion.body.tools(motion.x).copy()
        motion.step(tools, rotations, measured, dt)
    assert np.linalg.norm(motion.feedback_rate[1]) > 0
    held = motion.feedback.copy()
    reference_before = motion.body.tools(motion.x).copy()
    motion.freeze_translation_feedback()
    np.testing.assert_array_equal(motion.feedback_rate, np.zeros((2, 3)))
    assert motion.diagnostics["translation_feedback_frozen"]
    # The reference is not moved by the freeze and the value stays put under more error.
    np.testing.assert_allclose(motion.body.tools(motion.x), reference_before)
    tools = motion.body.tools(motion.x).copy()
    motion.step(tools, rotations, measured + [[0, 0, 0], [0, 0, -0.03]], dt)
    np.testing.assert_array_equal(motion.feedback, held)
    motion.resume_translation_feedback()
    tools = motion.body.tools(motion.x).copy()
    motion.step(tools, rotations, measured, dt)
    assert (
        np.max(np.linalg.norm(motion.feedback - held, axis=1)) <= cfg.tracking_translation_acceleration * dt**2 + 1e-12
    )


def test_rotation_feedback_runs_while_translation_feedback_is_frozen():
    ik = scene_ik()
    state = supported_state()
    cfg = replace(WholeBodyConfig(), tracking_correction_dynamics=True)
    motion = SupportedMotion(
        ik,
        state,
        cfg,
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
        hand_weights=(0.35, 1),
    )
    dt = 0.02
    tools = motion.body.tools(motion.x).copy()
    rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    motion.step(tools, rotations, tools, dt, measured_rotations=rotations)
    motion.freeze_translation_feedback()
    motion.freeze_rotation_feedback()
    motion.resume_rotation_feedback()
    held = motion.feedback.copy()
    tools = motion.body.tools(motion.x).copy()
    measured = Rotation.from_rotvec([[0, 0.05, 0], [0, -0.05, 0]]).as_matrix() @ rotations
    # Regression: this raised UnboundLocalError('first') on 2026-09-16.
    motion.step(tools, rotations, tools, dt, measured_rotations=measured)
    np.testing.assert_array_equal(motion.feedback, held)
    assert (
        np.max(np.linalg.norm(motion.rotation_feedback_rate, axis=1)) <= cfg.tracking_rotation_acceleration * dt + 1e-12
    )


def test_stationary_close_hold_unwinds_takeover_bias_with_bounded_feedback():
    state = supported_state()
    cfg = replace(WholeBodyConfig(), tracking_correction_dynamics=True)
    bias = np.array([[0.012, -0.004, 0.003], [-0.015, 0.002, -0.005]])

    def prepared_motion():
        ik = scene_ik()
        motion = SupportedMotion(
            ik,
            state,
            cfg,
            state["joints"],
            [1, 0, 0, 0],
            tool_offset=[0.1515, 0, 0],
            hand_weights=(0.35, 1),
        )
        rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
        reference = motion.body.tools(motion.x).copy()
        raw_target = reference - bias
        motion.step(raw_target, rotations, reference - bias, 0.02)
        reference = motion.body.tools(motion.x).copy()
        motion.step(raw_target, rotations, reference - 0.5 * bias, 0.02)
        assert np.max(np.linalg.norm(motion.feedback_rate, axis=1)) > 0
        return motion, rotations

    # A frozen correction retains the takeover bias even after the live
    # reference/measurement error has fallen.
    frozen, frozen_rotations = prepared_motion()
    frozen_reference = frozen.body.tools(frozen.x).copy()
    frozen_target, _ = frozen.freeze_translation_at_measured_tools(frozen_reference - 0.5 * bias)
    frozen_feedback = frozen.feedback.copy()
    for scale in (0.4, 0.3, 0.2, 0.1):
        frozen_reference = frozen.body.tools(frozen.x).copy()
        frozen.step(frozen_target, frozen_rotations, frozen_reference - scale * bias, 0.02)
        np.testing.assert_array_equal(frozen.feedback, frozen_feedback)
        np.testing.assert_array_equal(frozen.feedback_rate, np.zeros((2, 3)))

    # The close transition keeps the measured raw hold fixed and leaves the
    # already active bounded correction value and rate intact.
    motion, rotations = prepared_motion()
    reference = motion.body.tools(motion.x).copy()
    measured_at_capture = reference - 0.5 * bias
    feedback_at_capture = motion.feedback.copy()
    rate_at_capture = motion.feedback_rate.copy()
    state_at_capture = motion.x.copy()
    velocity_at_capture = motion.v.copy()
    close_target, captured_reference = begin_live_translation_hold(motion, measured_at_capture)
    raw_hold = close_target.copy()
    np.testing.assert_array_equal(close_target, measured_at_capture)
    np.testing.assert_allclose(captured_reference, reference, atol=1e-12)
    np.testing.assert_array_equal(motion.feedback, feedback_at_capture)
    np.testing.assert_array_equal(motion.feedback_rate, rate_at_capture)
    np.testing.assert_array_equal(motion.x, state_at_capture)
    np.testing.assert_array_equal(motion.v, velocity_at_capture)

    corrected_targets = [close_target + cfg.tracking_gain * feedback_at_capture]
    feedback_rates = [rate_at_capture]
    measured_error_norms = []
    for scale in np.linspace(0.45, 0.1, 25):
        reference = motion.body.tools(motion.x).copy()
        measured = reference - scale * bias
        measured_error_norms.append(np.linalg.norm(reference - measured, axis=1))
        motion.step(close_target, rotations, measured, 0.02)
        corrected_targets.append(close_target + cfg.tracking_gain * motion.feedback)
        feedback_rates.append(motion.feedback_rate.copy())

    np.testing.assert_array_equal(close_target, raw_hold)
    assert np.all(np.diff(np.asarray(measured_error_norms), axis=0) < 0)
    assert np.all(np.linalg.norm(motion.feedback, axis=1) < np.linalg.norm(feedback_at_capture, axis=1))

    corrected_steps = np.linalg.norm(np.diff(corrected_targets, axis=0), axis=2)
    assert np.max(corrected_steps) <= cfg.tracking_translation_speed * 0.02 + 1e-12
    feedback_rates = np.asarray(feedback_rates)
    assert np.max(np.linalg.norm(feedback_rates, axis=2)) <= cfg.tracking_translation_speed + 1e-12
    feedback_accelerations = np.diff(feedback_rates, axis=0) / 0.02
    assert np.max(np.linalg.norm(feedback_accelerations, axis=2)) <= cfg.tracking_translation_acceleration + 1e-12


def test_optional_posture_blend_is_continuous_and_capture_freeze_has_no_reference_jump():
    ik = scene_ik()
    state = supported_state()
    target = state["joints"].copy()
    target[12:14] = 0
    target[25:29] = [1.02, 0, 0.12, 0]
    weights = np.full(29, 0.04)
    weights[12:14], weights[25:29] = 0.12, 0.08
    motion = SupportedMotion(
        ik,
        state,
        WholeBodyConfig(),
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
        posture_reference=target,
        posture_weights=weights,
    )

    motion.blend_posture(0)
    np.testing.assert_array_equal(motion.posture, state["joints"])
    motion.blend_posture(0.5)
    np.testing.assert_allclose(motion.posture, 0.5 * (state["joints"] + target))
    np.testing.assert_allclose(motion.posture_weights[[12, 13, 25, 26, 27, 28]], [0.08] * 2 + [0.06] * 4)
    before = motion.joints
    before_state = motion.x.copy()
    before_velocity = motion.v.copy()
    motion.freeze_posture()
    np.testing.assert_array_equal(motion.posture, before)
    np.testing.assert_array_equal(motion.x, before_state)
    np.testing.assert_array_equal(motion.v, before_velocity)
    motion.blend_posture(1)
    np.testing.assert_array_equal(motion.posture, before)

    legacy = SupportedMotion(
        scene_ik(),
        state,
        WholeBodyConfig(),
        state["joints"],
        [1, 0, 0, 0],
        tool_offset=[0.1515, 0, 0],
    )
    legacy_prior = legacy.posture.copy()
    legacy.blend_posture(1)
    np.testing.assert_array_equal(legacy.posture, legacy_prior)
    assert legacy.posture_weights is None and legacy.continuity_weight == 0


def test_weighted_comfort_prior_reduces_redundant_extremes_without_losing_hand_or_feet():
    ik = scene_ik()
    state = supported_state()
    cfg = WholeBodyConfig()
    template = WholeBodyIK(ik, state, cfg, tool_offset=[0.1515, 0, 0], hand_weights=(0.35, 1), com_weight=20)
    positions = template.tools(template.initial).copy()
    rotations = ik.data.xmat[template.hands].reshape(2, 3, 3).copy()
    positions[1] += [0.06, -0.015, -0.06]

    extreme = state["joints"].copy()
    extreme[12:14] = [0.25, -0.12]
    extreme[25:29] = [1.7, 0.5, -0.4, 0.4]
    comfort = state["joints"].copy()
    comfort[12:14] = 0
    comfort[25:29] = [1.02, 0, 0.12, 0]
    weights = np.full(29, 0.04)
    weights[12:14], weights[25:29] = 0.12, 0.08

    def solve(prior):
        body = WholeBodyIK(
            scene_ik(),
            state,
            cfg,
            tool_offset=[0.1515, 0, 0],
            hand_weights=(0.35, 1),
            com_weight=20,
        )
        return body.solve(positions, rotations, posture=prior, posture_weights=weights, iterations=200)

    extreme_pose, comfort_pose = solve(extreme), solve(comfort)
    selected = np.r_[12:14, 25:29]
    assert np.linalg.norm((comfort_pose.joints - comfort)[selected]) < 0.35
    assert np.linalg.norm((comfort_pose.joints - comfort)[selected]) < 0.4 * np.linalg.norm(
        (extreme_pose.joints - comfort)[selected]
    )
    assert max(comfort_pose.hand_errors) < 0.001
    assert max(comfort_pose.foot_errors) < 0.0001


def test_velocity_continuity_is_opt_in_and_scaled_as_a_velocity_residual():
    ik = scene_ik()
    state = supported_state()
    cfg = WholeBodyConfig()
    template = WholeBodyIK(ik, state, cfg, tool_offset=[0.1515, 0, 0], hand_weights=(0.35, 1), com_weight=20)
    positions = template.tools(template.initial).copy()
    rotations = ik.data.xmat[template.hands].reshape(2, 3, 3).copy()
    positions[1] += [0.04, 0, -0.03]
    predicted = template.initial.copy()

    legacy = WholeBodyIK(scene_ik(), state, cfg, tool_offset=[0.1515, 0, 0], hand_weights=(0.35, 1), com_weight=20)
    explicit_zero = WholeBodyIK(
        scene_ik(), state, cfg, tool_offset=[0.1515, 0, 0], hand_weights=(0.35, 1), com_weight=20
    )
    legacy_pose = legacy.solve(positions, rotations, seed=predicted, posture=state["joints"], iterations=80)
    zero_pose = explicit_zero.solve(
        positions,
        rotations,
        seed=predicted,
        posture=state["joints"],
        continuity_weight=0.0,
        iterations=80,
    )
    np.testing.assert_allclose(zero_pose.joints, legacy_pose.joints, atol=1e-12)
    np.testing.assert_allclose(zero_pose.position, legacy_pose.position, atol=1e-12)

    smooth = WholeBodyIK(scene_ik(), state, cfg, tool_offset=[0.1515, 0, 0], hand_weights=(0.35, 1), com_weight=20)
    smooth_pose = smooth.solve(
        positions,
        rotations,
        seed=predicted,
        posture=state["joints"],
        continuity_reference=predicted,
        continuity_dt=0.05,
        continuity_weight=0.03,
        iterations=80,
    )
    legacy_state = np.r_[
        legacy_pose.position, Rotation.from_matrix(legacy_pose.rotation).as_euler("xyz")[1], legacy_pose.joints
    ]
    smooth_state = np.r_[
        smooth_pose.position, Rotation.from_matrix(smooth_pose.rotation).as_euler("xyz")[1], smooth_pose.joints
    ]
    assert np.linalg.norm(smooth_state - predicted) < np.linalg.norm(legacy_state - predicted)
    assert max(smooth_pose.hand_errors) < 0.01
    assert max(smooth_pose.foot_errors) < 0.001
    assert smooth.last_diagnostics["continuity_weight"] == 0.03
