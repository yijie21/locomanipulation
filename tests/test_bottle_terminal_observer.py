import numpy as np
import pytest

from locomani.tasks.bottle.whole_body_policy import (
    TerminalMotionObserver,
    TerminalRetreatCommand,
    predictive_capture_geometry,
    terminal_retreat_speed_limit,
)

AXIS = np.array([1.0, 0.0, 0.0])
PREFERRED_DEPTH = 0.0012
DEPTH_TOLERANCE = 0.005


def _observer():
    return TerminalMotionObserver(
        0.20,
        0.10,
        window_seconds=0.20,
        min_samples=6,
        max_gap_seconds=0.05,
    )


def _geometry(depth=PREFERRED_DEPTH):
    return {
        "ready": True,
        "rejection_reasons": [],
        "depth_m": depth,
        "preferred_depth_m": PREFERRED_DEPTH,
        "depth_tolerance_m": DEPTH_TOLERANCE,
    }


def test_zero_mean_alternating_motion_is_a_bounded_displacement_not_forward_drift():
    observer = _observer()
    status = None
    for index, stamp in enumerate(np.linspace(0.0, 0.20, 11)):
        position = 0.0015 if index % 2 == 0 else -0.0015
        status = observer.step(stamp, [position, 0.0, 0.0], AXIS, PREFERRED_DEPTH)

    assert status["ready"]
    assert status["raw_velocity_m_s"] == pytest.approx(0.15)
    assert status["trend_velocity_m_s"] == pytest.approx(0.0, abs=1e-12)
    assert status["oscillation_envelope_m"] == pytest.approx(0.003)
    assert status["stopping_displacement_m"] == pytest.approx(0.0, abs=1e-12)
    assert status["predicted_depth_interval_m"] == pytest.approx([-0.0018, 0.0042])

    geometry = predictive_capture_geometry(_geometry(), status)
    assert geometry["ready"]
    assert geometry["stopping"]["accepted_depth_interval_m"] == pytest.approx([-0.0038, 0.0062])

    # Feed the same alternating measurement through the real goal-retreat
    # state. Rectifying each frame would ratchet toward the 10 cm cap.
    observer = _observer()
    command = TerminalRetreatCommand(0.16, 0.20)
    retreat = []
    stamp = 10_000.0
    for index in range(401):
        position = -0.0015 if index % 2 == 0 else 0.0015
        status = observer.step(stamp, [position, 0.0, 0.0], AXIS, PREFERRED_DEPTH)
        stop = status["stopping_displacement_m"]
        value, _ = command.step(
            0.0 if stop is None else max(0.0, stop),
            0.02,
            hold=not status["ready"],
        )
        retreat.append(value)
        stamp += 0.02
    assert status["ready"]
    assert max(retreat) < 1e-12


def test_sustained_forward_motion_brakes_and_places_interval_before_capture_window():
    observer = _observer()
    status = None
    for stamp in np.linspace(0.0, 0.20, 11):
        status = observer.step(stamp, [0.080 * stamp, 0.0, 0.0], AXIS, 0.0057)

    assert status["ready"]
    assert status["raw_velocity_m_s"] == pytest.approx(0.080)
    assert status["trend_velocity_m_s"] == pytest.approx(0.080)
    assert status["oscillation_envelope_m"] == pytest.approx(0.0, abs=1e-12)
    assert status["stopping_displacement_m"] == pytest.approx(0.024)
    assert status["predicted_depth_interval_m"] == pytest.approx([-0.0183, -0.0183])
    assert not predictive_capture_geometry(_geometry(0.0057), status)["ready"]


def test_velocity_reversal_remains_signed_and_blocks_capture_while_moving_away():
    observer = _observer()
    for stamp in np.linspace(0.0, 0.20, 11):
        observer.step(stamp, [0.060 * stamp, 0.0, 0.0], AXIS, PREFERRED_DEPTH)

    turn_position = 0.060 * 0.20
    status = None
    for stamp in np.linspace(0.22, 0.42, 11):
        position = turn_position - 0.060 * (stamp - 0.20)
        status = observer.step(stamp, [position, 0.0, 0.0], AXIS, PREFERRED_DEPTH)

    assert status["ready"]
    assert status["raw_velocity_m_s"] == pytest.approx(-0.060)
    assert status["trend_velocity_m_s"] == pytest.approx(-0.060)
    assert status["stopping_displacement_m"] == pytest.approx(-0.015)
    assert status["predicted_depth_interval_m"] == pytest.approx([0.0162, 0.0162])
    rejected = predictive_capture_geometry(_geometry(), status)
    assert not rejected["ready"]
    assert "predicted_depth_interval" in rejected["rejection_reasons"]


def test_short_real_displacement_pulse_expands_interval_without_becoming_drift():
    observer = _observer()
    status = None
    for index, stamp in enumerate(np.linspace(0.0, 0.20, 11)):
        position = 0.006 if index == 5 else 0.0
        status = observer.step(stamp, [position, 0.0, 0.0], AXIS, PREFERRED_DEPTH)

    assert status["ready"]
    assert status["trend_velocity_m_s"] == pytest.approx(0.0, abs=1e-12)
    assert status["oscillation_envelope_m"] == pytest.approx(0.006)
    assert status["predicted_depth_interval_m"] == pytest.approx([-0.0048, 0.0072])
    assert not predictive_capture_geometry(_geometry(), status)["ready"]


def test_clock_gap_and_phase_boundary_reset_history_and_block_capture():
    observer = _observer()
    for stamp in np.linspace(0.0, 0.20, 11):
        ready = observer.step(stamp, [0.0, 0.0, 0.0], AXIS, PREFERRED_DEPTH)
    assert ready["ready"]

    gap = observer.step(0.28, [0.0, 0.0, 0.0], AXIS, PREFERRED_DEPTH)
    assert not gap["ready"]
    assert gap["reset_reason"] == "clock_gap"
    assert gap["sample_count"] == 1
    assert not predictive_capture_geometry(_geometry(), gap)["ready"]

    phase = observer.step(
        0.30,
        [0.0, 0.0, 0.0],
        AXIS,
        PREFERRED_DEPTH,
        phase="close",
    )
    assert not phase["ready"]
    assert phase["reset_reason"] == "phase_boundary"
    assert phase["sample_count"] == 1


def test_retreat_command_has_no_jump_or_ratchet_and_brakes_smoothly_at_zero():
    command = TerminalRetreatCommand(0.16, 0.20)
    dt = 0.02
    values = []
    rates = []
    targets = [0.024] * 50 + [0.0] * 600
    for target in targets:
        value, rate = command.step(target, dt)
        values.append(value)
        rates.append(rate)

    values = np.asarray(values)
    rates = np.asarray(rates)
    assert np.min(values) >= 0.0
    assert np.max(np.abs(rates)) <= 0.16 + 1e-12
    assert np.max(np.abs(np.diff(rates))) <= 0.20 * dt + 1e-12
    assert np.max(np.abs(np.diff(values))) <= 0.16 * dt + 1e-12
    assert values[-1] < 1e-6
    assert abs(rates[-1]) < 1e-6

    held = TerminalRetreatCommand(0.16, 0.20)
    for _ in range(20):
        before_value, before_rate = held.step(0.024, dt)
    held_value, held_rate = held.step(0.0, dt, hold=True)
    assert held_value > before_value
    assert abs(held_rate - before_rate) <= 0.20 * dt + 1e-12

    alternating = TerminalRetreatCommand(0.16, 0.20)
    alternating_values = []
    for index in range(600):
        target = 0.001 if index % 2 else 0.0
        value, _ = alternating.step(target, dt)
        alternating_values.append(value)
    assert max(alternating_values) < 0.001
    assert max(alternating_values[300:]) - min(alternating_values[300:]) < 0.0001


def test_runtime_retreat_bandwidth_reaches_ninety_percent_without_hand_speed_coupling():
    speed_limit = terminal_retreat_speed_limit(0.16, 0.20, 0.20)
    assert speed_limit == pytest.approx(0.04)

    command = TerminalRetreatCommand(speed_limit, 0.20)
    values = [command.step(0.024, 0.02)[0] for _ in range(45)]
    assert values[-1] >= 0.9 * 0.024
    assert np.max(np.abs(np.diff(values))) <= speed_limit * 0.02 + 1e-12
