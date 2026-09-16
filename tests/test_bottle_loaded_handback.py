import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from locomani.tasks.bottle.loaded_handback import (
    AsyncLoadedEndpointPlanner,
    LoadedHandbackStabilityGate,
    commanded_palm_rotation,
    loaded_lift_height,
    loaded_liftoff_event,
    loaded_orientation_metrics,
    loaded_orientation_supervision,
    loaded_progress_governor,
    loaded_tracking_scale,
    loaded_world_path,
)
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig


def _gate():
    return LoadedHandbackStabilityGate(
        quiet_seconds=0.10,
        min_samples=3,
        max_gap_seconds=0.06,
        hand_position_diameter_max=0.004,
        hand_speed_max=0.03,
        hand_rotation_diameter_max=0.035,
        hand_angular_speed_max=0.20,
        object_relative_speed_max=0.08,
    )


def _gate_step(gate, stamp, *, hand=None, root=None, legs=None, relative=None, bilateral=True):
    return gate.step(
        stamp,
        np.zeros(3) if hand is None else hand,
        np.eye(3),
        np.zeros(3) if root is None else root,
        np.eye(3),
        np.zeros(12) if legs is None else legs,
        np.array([0.15, 0.0, 0.0]) if relative is None else relative,
        bilateral,
    )


def _swing_metrics(swing_rad, tilt_rad=0.0):
    return {
        "translation_error_m": 0.0,
        "rotation_error_rad": 0.0,
        "object_tilt_rad": tilt_rad,
        "grip_axis_swing_rad": swing_rad,
    }


_GOVERNOR_LIMITS = {
    "translation_error_m": (0.015, 0.050),
    "rotation_error_rad": (0.060, 0.180),
    "object_tilt_rad": (0.105, 0.262),
    "grip_axis_swing_rad": (0.035, 0.120),
}


def test_grip_swing_does_not_govern_or_stop_before_liftoff():
    # stage3_3_switch8 30700:8: swing 0.151 rad within 0.28 s of lift entry with the
    # bottle still on the table froze progress at 0.018 and timed out; the table
    # pins the bottle, so palm rotation is not a slip signal until liftoff.
    metrics = _swing_metrics(0.151)
    on_table = loaded_progress_governor(metrics, _GOVERNOR_LIMITS, lifted=False)
    assert on_table["scale"] == 1.0 and not on_table["lifted"]
    assert on_table["components"]["grip_axis_swing_rad"] == 1.0
    carried = loaded_progress_governor(metrics, _GOVERNOR_LIMITS, lifted=True)
    assert carried["scale"] == 0.0 and carried["lifted"]

    stop = {"object_tilt_rad": 0.262, "grip_axis_swing_rad": 0.120}
    fail = {"object_tilt_rad": 0.349, "grip_axis_swing_rad": 0.200}
    calm = loaded_orientation_supervision(metrics, stop, fail, 1.0, None, 0.4, lifted=False)
    assert not calm["stopped"] and not calm["failed"]
    stopped = loaded_orientation_supervision(metrics, stop, fail, 1.0, None, 0.4, lifted=True)
    assert stopped["stopped"] and not stopped["failed"]
    # Bottle tilt on the table (30800:0, 0.139 rad rising past the stop band) still stops.
    tipping = loaded_orientation_supervision(_swing_metrics(0.0, 0.30), stop, fail, 1.0, None, 0.4, lifted=False)
    assert tipping["stopped"]
    over = loaded_orientation_supervision(_swing_metrics(0.0, 0.36), stop, fail, 1.0, None, 0.4, lifted=False)
    assert over["failed"] and over["reason"] == "object_tilt_rad"


def test_loaded_rotation_error_is_measured_against_the_corrected_command():
    desired = np.eye(3)
    feedback = np.array([0.0, 0.25, 0.0])
    commanded = commanded_palm_rotation(desired, feedback, 1.0)
    # A palm that follows the corrected command exactly has zero loaded error.
    latched = commanded @ commanded.T
    current = commanded @ commanded.T
    assert Rotation.from_matrix(current @ latched.T).magnitude() == pytest.approx(0.0)
    # Measured against the bare desired rotation the same palm reads the whole compensation.
    assert Rotation.from_matrix(commanded @ desired.T).magnitude() == pytest.approx(0.25)
    assert np.allclose(commanded_palm_rotation(desired, feedback, 0.0), desired)


def test_rotation_tracking_error_is_diagnostic_not_governing():
    metrics = {
        "translation_error_m": 0.0,
        "rotation_error_rad": 0.25,
        "object_tilt_rad": 0.0,
        "grip_axis_swing_rad": 0.0,
    }
    governor = loaded_progress_governor(metrics, _GOVERNOR_LIMITS, lifted=True)
    assert governor["scale"] == 1.0
    assert governor["components"]["rotation_error_rad"] == 0.0
    assert "rotation_error_rad" not in governor["governing"]
    tilted = loaded_progress_governor({**metrics, "object_tilt_rad": 0.30}, _GOVERNOR_LIMITS, lifted=True)
    assert tilted["scale"] == 0.0


def test_liftoff_event_fires_once_on_bilateral_rise():
    assert not loaded_liftoff_event(0.0, 0.01, True, False)
    assert not loaded_liftoff_event(0.009, 0.01, True, False)
    assert not loaded_liftoff_event(0.02, 0.01, False, False)
    assert loaded_liftoff_event(0.01, 0.01, True, False)
    assert not loaded_liftoff_event(0.05, 0.01, True, True)
    with pytest.raises(ValueError):
        loaded_liftoff_event(0.01, 0.0, True, False)


def test_relatched_grip_relation_zeroes_on_table_rotation():
    # The relation carried after liftoff is measured at liftoff, so the palm
    # rotation absorbed while the table held the bottle no longer counts as swing.
    palm0, obj = np.eye(3), np.eye(3)
    captured = palm0.T @ obj
    palm1 = Rotation.from_euler("x", 0.15).as_matrix()
    before = loaded_orientation_metrics(palm1, obj, captured)
    assert before["grip_axis_swing_rad"] == pytest.approx(0.15)
    relatched = before["object_in_palm"]
    after = loaded_orientation_metrics(palm1, obj, relatched)
    assert after["grip_axis_swing_rad"] == pytest.approx(0.0)
    later = loaded_orientation_metrics(Rotation.from_euler("x", 0.20).as_matrix(), obj, relatched)
    assert later["grip_axis_swing_rad"] == pytest.approx(0.05)


def test_loaded_world_path_clears_before_smooth_inward_carry():
    start = np.array([0.50, -0.20, 0.70])
    axis = np.array([1.0, 0.0, 0.0])
    clearance = 0.75

    for progress in np.linspace(0.0, clearance, 13):
        point, status = loaded_world_path(start, axis, 0.08, progress, clearance_fraction=clearance)
        assert point[0] == pytest.approx(start[0])
        assert status["phase"] == "clear_table"

    after, status = loaded_world_path(start, axis, 0.08, clearance + 0.05, clearance_fraction=clearance)
    assert after[0] < start[0]
    assert status["phase"] == "carry_inward"

    # The carry polynomial starts with zero velocity and acceleration at the
    # clearance boundary; vertical rise remains one global quintic.
    epsilon = 1e-5
    at_boundary, _ = loaded_world_path(start, axis, 0.08, clearance)
    first, _ = loaded_world_path(start, axis, 0.08, clearance + epsilon)
    second, _ = loaded_world_path(start, axis, 0.08, clearance + 2 * epsilon)
    inward_velocity = (first[0] - at_boundary[0]) / epsilon
    inward_acceleration = (second[0] - 2 * first[0] + at_boundary[0]) / epsilon**2
    assert abs(inward_velocity) < 1e-6
    assert abs(inward_acceleration) < 0.01


def test_loaded_endpoint_keeps_measured_rise_margin_under_tracking_residuals():
    required_rise = 0.06
    tracking_margin = 0.025
    commanded_rise = loaded_lift_height(required_rise, tracking_margin)
    start = np.array([0.50, -0.20, 0.70])
    endpoint, status = loaded_world_path(
        start,
        [1.0, 0.0, 0.0],
        commanded_rise,
        1.0,
        carry_distance=required_rise,
    )

    assert endpoint[2] - start[2] == pytest.approx(0.085)
    assert status["carry_distance_m"] == pytest.approx(required_rise)
    for terminal_residual in (-0.002, -0.024):
        assert endpoint[2] + terminal_residual - start[2] >= required_rise

    dynamic_residual = np.linspace(0.0, -0.024, 11)
    measured_rise = []
    for progress, residual in zip(np.linspace(0.0, 1.0, 11), dynamic_residual):
        target, _ = loaded_world_path(
            start,
            [1.0, 0.0, 0.0],
            commanded_rise,
            progress,
            carry_distance=required_rise,
        )
        measured_rise.append(target[2] + residual - start[2])
    assert measured_rise[-1] >= required_rise


def test_loaded_progress_scales_continuously_with_tracking_load():
    from locomani.tasks.bottle.whole_body_policy import advance_loaded_progress

    assert advance_loaded_progress(0.40, 0.05, 2.0, 1.0, 0.0) == pytest.approx(0.40)
    assert advance_loaded_progress(0.40, 0.05, 2.0, 1.0, 0.5) == pytest.approx(0.4125)
    assert advance_loaded_progress(0.40, 0.05, 2.0, 1.0, 1.0) == pytest.approx(0.425)


def test_recoverable_angular_tracking_residual_advances_at_reduced_speed():
    residual = np.radians(5.5)
    scale = loaded_tracking_scale(residual, 0.060, 0.180)

    assert 0.0 < scale < 1.0
    assert scale == pytest.approx((0.180 - residual) / 0.120)


def test_loaded_orientation_governor_slows_then_stops_as_tilt_grows():
    limits = {
        "translation_error_m": (0.015, 0.050),
        "rotation_error_rad": (0.060, 0.180),
        "object_tilt_rad": (0.105, 0.262),
        "grip_axis_swing_rad": (0.035, 0.120),
    }
    nominal = {
        "translation_error_m": 0.0,
        "rotation_error_rad": 0.0,
        "object_tilt_rad": 0.0,
        "grip_axis_swing_rad": 0.0,
    }

    reduced = loaded_progress_governor({**nominal, "object_tilt_rad": 0.18}, limits)
    stopped = loaded_progress_governor({**nominal, "object_tilt_rad": 0.262}, limits)

    assert 0.0 < reduced["scale"] < 1.0
    assert stopped["scale"] == 0.0
    assert reduced["components"]["object_tilt_rad"] == reduced["scale"]


def test_loaded_orientation_stop_has_timed_and_immediate_failure_reasons():
    stop = {"object_tilt_rad": 0.262, "grip_axis_swing_rad": 0.120}
    fail = {"object_tilt_rad": 0.349, "grip_axis_swing_rad": 0.200}
    stopped_metrics = {"object_tilt_rad": 0.28, "grip_axis_swing_rad": 0.02}

    first = loaded_orientation_supervision(stopped_metrics, stop, fail, 1.0, None, 0.40)
    recovering = loaded_orientation_supervision(stopped_metrics, stop, fail, 1.39, first["bad_since"], 0.40)
    timed_out = loaded_orientation_supervision(stopped_metrics, stop, fail, 1.41, first["bad_since"], 0.40)
    immediate = loaded_orientation_supervision(
        {"object_tilt_rad": 0.36, "grip_axis_swing_rad": 0.02},
        stop,
        fail,
        2.0,
        None,
        0.40,
    )
    recovered = loaded_orientation_supervision(
        {"object_tilt_rad": 0.10, "grip_axis_swing_rad": 0.02},
        stop,
        fail,
        1.2,
        first["bad_since"],
        0.40,
    )

    assert first["stopped"] and not first["failed"]
    assert recovering["stopped"] and not recovering["failed"]
    assert timed_out["failed"] and timed_out["timed_out"]
    assert timed_out["reason"] == "object_tilt_rad"
    assert immediate["failed"] and not immediate["timed_out"]
    assert recovered["bad_since"] is None


def test_loaded_relative_orientation_ignores_stable_bias_and_detects_new_slip():
    palm = Rotation.from_euler("y", 0.20).as_matrix()
    stable_bias = Rotation.from_euler("z", -0.12).as_matrix()
    captured = palm.T @ (palm @ stable_bias)

    initial = loaded_orientation_metrics(palm, palm @ stable_bias, captured)
    slipped = loaded_orientation_metrics(
        palm,
        palm @ stable_bias @ Rotation.from_euler("x", 0.08).as_matrix(),
        captured,
    )

    assert initial["grip_rotation_error_rad"] == pytest.approx(0.0, abs=1e-12)
    assert slipped["grip_rotation_error_rad"] == pytest.approx(0.08)


def test_loaded_grip_axial_twist_is_diagnostic_but_not_axis_swing():
    palm = Rotation.from_euler("xyz", [0.15, -0.20, 0.30]).as_matrix()
    captured = np.eye(3)
    twist = Rotation.from_euler("z", 0.40).as_matrix()

    metrics = loaded_orientation_metrics(palm, palm @ twist, captured)

    assert metrics["grip_rotation_error_rad"] == pytest.approx(0.40)
    assert metrics["grip_axis_swing_rad"] == pytest.approx(0.0, abs=1e-12)


def test_loaded_grip_axis_deflection_is_measured_as_swing():
    palm = Rotation.from_euler("xyz", [0.15, -0.20, 0.30]).as_matrix()
    captured = np.eye(3)
    deflection = Rotation.from_euler("x", 0.15).as_matrix()

    metrics = loaded_orientation_metrics(palm, palm @ deflection, captured)

    assert metrics["grip_rotation_error_rad"] == pytest.approx(0.15)
    assert metrics["grip_axis_swing_rad"] == pytest.approx(0.15)


def test_async_loaded_endpoint_planner_is_nonblocking_and_copies_payload():
    started = threading.Event()
    release = threading.Event()

    class Evaluator:
        def evaluate(self, payload):
            started.set()
            assert release.wait(timeout=1.0)
            return {"admitted": True, "value": payload["values"][0]}

    planner = AsyncLoadedEndpointPlanner(Evaluator())
    payload = {"values": np.array([1.0])}
    before = time.perf_counter()
    response = planner.request("request-1", payload)
    elapsed = time.perf_counter() - before

    assert response["status"] == "pending"
    assert elapsed < 0.05
    assert started.wait(timeout=1.0)
    payload["values"][0] = 9.0
    release.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not planner.active["future"].done():
        time.sleep(0.001)
    assert planner.active["future"].done()
    time.sleep(0.03)
    observed_wall = time.monotonic()
    response = planner.poll("request-1")
    planner.close()

    assert response["status"] == "ready"
    assert response["result"]["value"] == pytest.approx(1.0)
    assert response["result"]["completed_wall"] >= response["result"]["submitted_wall"]
    assert observed_wall - response["result"]["completed_wall"] >= 0.02


def test_pending_endpoint_keeps_pose_references_advancing_across_clock_gaps():
    from locomani.tasks.bottle.whole_body_policy import push_loaded_reference

    arm = Mock()
    events = []
    stamps = (1.00, 1.02, 1.20)
    for previous, stamp in zip((0.98, *stamps[:-1]), stamps):
        event = push_loaded_reference(
            arm,
            np.zeros(29),
            np.ones(29) * 0.1,
            np.zeros(3),
            {"seq": round(stamp * 100), "t": stamp, "wall": 1000.0 + stamp},
            stamp - previous,
            endpoint_pending=True,
            request_count=1,
        )
        events.append(event)

    assert arm._push.call_count == len(stamps)
    assert [call.kwargs["reference_time"] for call in arm._push.call_args_list] == list(stamps)
    assert [event["dt"] for event in events] == pytest.approx([0.02, 0.02, 0.18])
    assert all(event["request_count"] == 1 for event in events)


def test_loaded_stability_quiet_window_admits_and_contact_loss_resets():
    gate = _gate()
    statuses = [_gate_step(gate, stamp) for stamp in (0.0, 0.05, 0.10)]
    assert not statuses[0]["ready"]
    assert statuses[-1]["ready"]
    assert statuses[-1]["held_seconds"] == pytest.approx(0.10)

    lost = _gate_step(gate, 0.15, bilateral=False)
    assert not lost["ready"]
    assert lost["sample_count"] == 0
    assert lost["reset_reason"] == "bilateral_contact_lost"


def test_policy_endpoint_gate_admits_quiet_measurements_despite_constant_decoder_residual():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = WholeBodyConfig()
    gate = policy._loaded_stability_gate()
    decoder_translation_residual = 0.08
    decoder_rotation_residual = 0.10
    assert decoder_translation_residual > policy.cfg.stance_ik_tolerance
    assert decoder_rotation_residual > policy.cfg.capture_rotation_diameter_max

    statuses = [
        _gate_step(gate, stamp, hand=np.array([decoder_translation_residual, 0.0, 0.0]))
        for stamp in np.linspace(0.0, policy.cfg.capture_terminal_window_seconds, 11)
    ]

    assert statuses[-1]["ready"]


@pytest.mark.parametrize(
    "changing",
    [
        {"hand": np.array([0.01, 0.0, 0.0])},
        {"root": np.array([0.01, 0.0, 0.0])},
        {"legs": np.r_[0.05, np.zeros(11)]},
    ],
)
def test_loaded_stability_rejects_measured_hand_root_and_leg_motion(changing):
    gate = _gate()
    _gate_step(gate, 0.0)
    moved = _gate_step(gate, 0.05, **changing)
    assert not moved["ready"]
    assert moved["reset_reason"] == "motion"
    assert moved["sample_count"] == 1


def test_loaded_stability_rejects_zero_trend_symmetric_oscillation():
    gate = _gate()
    statuses = []
    for stamp, offset in zip((0.0, 0.025, 0.05, 0.075, 0.10), (0.0, 0.0035, 0.0, -0.0035, 0.0)):
        statuses.append(_gate_step(gate, stamp, hand=np.array([offset, 0.0, 0.0])))
    assert not any(status["ready"] for status in statuses)
    assert any(status["reset_reason"] == "motion" for status in statuses)


def test_loaded_stability_resets_non_increasing_time_and_clock_gaps():
    gate = _gate()
    _gate_step(gate, 0.0)
    repeated = _gate_step(gate, 0.0)
    assert repeated["reset_reason"] == "non_increasing_time"
    assert repeated["sample_count"] == 1

    gap = _gate_step(gate, 0.20)
    assert gap["reset_reason"] == "clock_gap"
    assert gap["sample_count"] == 1
    assert not gap["ready"]


def test_loaded_stability_allows_one_declared_boundary_gap_and_keeps_measuring():
    gate = _gate()
    for stamp in (0.0, 0.03, 0.06):
        _gate_step(gate, stamp)
    gate.expect_boundary(0.20)
    crossed = _gate_step(gate, 0.12)
    assert crossed["reset_reason"] is None
    assert crossed["sample_count"] == 4
    assert crossed["boundary_gap_seconds"] == pytest.approx(0.06)
    # Motion across the boundary is still measured against the retained samples.
    gate2 = _gate()
    for stamp in (0.0, 0.03, 0.06):
        _gate_step(gate2, stamp)
    gate2.expect_boundary(0.20)
    moved = _gate_step(gate2, 0.12, hand=np.array([0.01, 0.0, 0.0]))
    assert moved["reset_reason"] == "motion"
    # The allowance is single-use and bounded.
    later = _gate_step(gate, 0.20)
    assert later["reset_reason"] == "clock_gap"
    gate.expect_boundary(0.20)
    too_far = _gate_step(gate, 0.50)
    assert too_far["reset_reason"] == "clock_gap"
    with pytest.raises(ValueError):
        gate.expect_boundary(0.01)


def _handback_state(stamp, *, hand_x=0.0, contact=True, stage="pre"):
    return {
        "seq": int(round((stamp + 1.0) * 100)),
        "t": float(stamp),
        "wall": 1000.0 + float(stamp),
        "stage": stage,
        "contact": contact,
        "hand": np.array([hand_x, 0.0, 0.0]),
        "joints": np.zeros(29),
        "pelvis": {"pos": np.zeros(3), "quat": np.array([1.0, 0.0, 0.0, 0.0])},
    }


def _handback_policy(states):
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    reference = np.linspace(-0.1, 0.1, 29)
    stop = reference + 0.02
    admission = _handback_state(-0.01)
    admission["joints"] = reference - 0.01
    prediction = {
        "joints": stop,
        "root": np.array([1.0, 0.0, 0.0, 0.0]),
        "duration": 0.2,
        "frames": 10,
        "preview": {},
    }
    arm = Mock(
        q=reference.copy(),
        root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        reference_velocity=np.full(29, 0.4),
        root_angular_velocity=np.array([0.0, 0.2, 0.0]),
        hands={"right": np.zeros(7)},
        _stream_released=False,
    )
    arm.stopping_reference.return_value = prediction
    events = []

    def brake(admitted):
        assert events == ["clear"]
        events.append("brake")
        arm.q = np.asarray(admitted["joints"], dtype=float).copy()
        arm.reference_velocity = np.zeros(29)
        arm.root_angular_velocity = np.zeros(3)
        return {"packet": {"max_joint_acceleration": 2.0}, "observed": {"max_joint_acceleration": 3.0}}

    arm.brake_to_stop.side_effect = brake
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.arm = arm
    policy.stats = {}
    policy.cfg = replace(
        WholeBodyConfig(),
        capture_terminal_window_seconds=0.10,
        capture_min_samples=3,
        capture_terminal_max_gap_seconds=0.05,
    )
    policy._stream_released = False
    policy.p = SimpleNamespace(
        obs=Mock(side_effect=[admission, *states]),
        cmd=SimpleNamespace(reference_observer=None, exit_streamed=Mock()),
    )
    policy._clear = Mock(side_effect=lambda *_args, **_kwargs: events.append("clear"))
    policy._handback_test_events = events
    evidence = []

    def stability(gate, state):
        evidence.append((id(gate), state["stage"], len(gate.samples)))
        return _gate_step(gate, state["t"], hand=state["hand"], bilateral=state["contact"])

    policy._loaded_stability_step = stability
    return policy, evidence


def _quiet_handback_states():
    pre = [_handback_state(stamp, stage="pre") for stamp in (0.0, 0.04, 0.08, 0.12)]
    post = [_handback_state(stamp, stage="post") for stamp in (0.16, 0.20, 0.24)]
    return pre, post


def test_handback_retains_evidence_across_switch_and_observes_idle_tail():
    from locomani.backends.sonic.primitives import HAND_CLOSED_RIGHT

    pre, post = _quiet_handback_states()
    policy, evidence = _handback_policy([*pre, *post])
    with patch("locomani.tasks.bottle.whole_body_policy.time.sleep"):
        ok, reason = policy._brake_handoff()

    assert ok and "stable handback" in reason
    assert len({gate_id for gate_id, _stage, _count in evidence}) == 1
    assert next(count for _gate_id, stage, count in evidence if stage == "post") > 0
    assert policy.stats["handback"]["phase"] == "complete"
    assert policy.stats["handback"]["completed"]
    assert policy.stats["handback"]["post_switch"]["ready"]
    assert policy.stats["handback"]["switch"]["requested_mode"] == "planner_frozen_upper_body_idle"
    assert "no controller consumption" in policy.stats["handback"]["switch"]["observation_scope"]
    assert (
        policy.stats["handback"]["switch"]["release_returned_wall"]
        >= policy.stats["handback"]["switch"]["release_requested_wall"]
    )
    assert policy.stats["handback"]["post_switch_window_end"]["elapsed_physical_seconds"] >= 0.10
    policy.arm.stop.assert_called_once_with(preserve_reference=True)
    assert policy._handback_test_events == ["clear", "brake"]
    assert policy.stats["handback"]["braking"]["packet"]["max_joint_acceleration"] == 2.0
    assert all(np.allclose(call.kwargs["reference_velocity"], 0) for call in policy.arm._push.call_args_list)
    np.testing.assert_array_equal(policy.arm.hands["right"], HAND_CLOSED_RIGHT)


def test_handback_tolerates_switch_latency_gap_and_records_first_post_switch_sample():
    pre, _post = _quiet_handback_states()
    # stage3_2_aligned8: the first post-switch state arrived 0.060 s after the
    # switch sample, above the 0.05 s gap limit, while the stream stayed at 0.02 s.
    post = [_handback_state(stamp, stage="post") for stamp in (0.18, 0.22, 0.26)]
    policy, _evidence = _handback_policy([*pre, *post])
    with patch("locomani.tasks.bottle.whole_body_policy.time.sleep"):
        ok, reason = policy._brake_handoff()

    assert ok and "stable handback" in reason
    first = policy.stats["handback"]["switch"]["first_post_switch"]
    assert first["t"] == pytest.approx(0.18)
    assert first["gap_from_switch_sample_s"] == pytest.approx(0.06)
    assert first["boundary_tolerance_s"] == pytest.approx(0.10)
    assert policy.stats["handback"]["post_switch"]["boundary_gap_seconds"] == pytest.approx(0.06)


def test_handback_still_fails_when_switch_gap_exceeds_quiet_window():
    pre, _post = _quiet_handback_states()
    policy, _evidence = _handback_policy([*pre, _handback_state(0.40, stage="post")])
    with patch("locomani.tasks.bottle.whole_body_policy.time.sleep"):
        ok, reason = policy._brake_handoff()

    assert not ok and "clock_gap" in reason
    assert policy.stats["handback"]["phase"] == "post_switch_failed"


@pytest.mark.parametrize(
    "first_post, expected_reason",
    [
        (_handback_state(0.16, hand_x=0.02, stage="post"), "motion"),
        (_handback_state(0.16, contact=False, stage="post"), "bilateral_contact_lost"),
    ],
)
def test_handback_rejects_post_switch_motion_and_contact_loss(first_post, expected_reason):
    pre, _post = _quiet_handback_states()
    policy, _evidence = _handback_policy([*pre, first_post])
    with patch("locomani.tasks.bottle.whole_body_policy.time.sleep"):
        ok, reason = policy._brake_handoff()

    assert not ok and expected_reason in reason
    assert policy.stats["handback"]["phase"] == "post_switch_failed"
    assert policy.stats["handback"]["reason"] == expected_reason
    policy.arm.stop.assert_called_once_with(preserve_reference=True)


def test_failed_execute_cleanup_releases_stream_once():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.arm = Mock(q=np.zeros(29), _stream_released=False)
    policy.stats = {}
    policy._stream_released = False
    policy.settle_for_admission = Mock(return_value=True)
    policy._reach = Mock(return_value=(False, "loaded reach failed"))
    policy.completed = False
    policy.previous_observer = object()
    policy.p = SimpleNamespace(
        obs=Mock(return_value={}),
        cmd=SimpleNamespace(reference_observer=None, exit_streamed=Mock()),
    )

    assert policy.execute(grip_world=None, squat_hint=None, ghost=None) == (False, "loaded reach failed")
    policy.arm.stop.assert_called_once_with(preserve_reference=True)
    policy._release_streamed()
    policy.arm.stop.assert_called_once_with(preserve_reference=True)


def _loaded_endpoint_fixture(failure=None):
    from locomani.tasks.bottle.whole_body_policy import LoadedEndpointEvaluator

    reference_joints = np.linspace(-0.20, 0.20, 29)
    measured_joints = reference_joints + np.linspace(0.01, 0.03, 29)
    nominal_joints = reference_joints + np.linspace(-0.04, 0.05, 29)
    reference_position = np.array([0.10, 0.20, 0.80])
    solved_position = np.array([0.15, 0.18, 0.82])
    reference_rotation = Rotation.from_euler("z", 0.20).as_matrix()
    solved_rotation = Rotation.from_euler("z", 0.30).as_matrix()
    measured_rotation = Rotation.from_euler("z", -0.10).as_matrix()
    pose = SimpleNamespace(
        joints=nominal_joints.copy(),
        rotation=solved_rotation,
        position=solved_position,
        hand_errors=np.array([0.001, 0.001]),
        foot_errors=np.array([0.001, 0.001]),
        com_shift=np.array([0.001, 0.0]),
    )
    if failure == "hand":
        pose.hand_errors[1] = 0.10
    elif failure == "foot":
        pose.foot_errors[0] = 0.10
    elif failure == "com":
        pose.com_shift[0] = 0.10

    body = SimpleNamespace(torso=0, solve=Mock(return_value=pose))
    ik = SimpleNamespace(data=SimpleNamespace(xmat=np.eye(3).reshape(1, 9)))
    hit = SimpleNamespace(
        clear=failure != "collision",
        link="robot/link",
        obstacle="table",
    )
    guard = Mock()
    guard.check.return_value = hit
    support = {
        "heading": Rotation.from_euler("z", 0.17).as_matrix(),
        "feet_pos": np.array([[0.1, 0.2, 0.0], [0.1, -0.2, 0.0]]),
        "feet_rot": np.array([np.eye(3), Rotation.from_euler("z", 0.02).as_matrix()]),
        "com0": np.array([0.03, -0.01]),
        "lo": np.linspace(-1.0, -0.5, 33),
        "hi": np.linspace(0.5, 1.0, 33),
        "hand_weights": np.array([0.8, 1.2]),
        "com_weight": 3.5,
    }
    endpoint = np.array([0.30, -0.20, 0.85])
    reference_root = Rotation.from_matrix(reference_rotation).as_quat()[[3, 0, 1, 2]]
    state = {
        "joints": measured_joints.copy(),
        "pelvis": {
            "pos": np.array([1.0, 2.0, 0.70]),
            "quat": Rotation.from_matrix(measured_rotation).as_quat()[[3, 0, 1, 2]],
        },
    }
    payload = {
        "state": state,
        "targets": np.array([[0.20, 0.20, 0.80], [0.30, -0.20, 0.75]]),
        "rotations": np.array([np.eye(3), np.eye(3)]),
        "endpoint": endpoint,
        "reference_x": np.r_[reference_position, 0.0, reference_joints],
        "reference_root": reference_root,
        "standing_reference": np.zeros(29),
        "standing_weights": np.ones(29) * 0.04,
        "support_problem": support,
        "source": {"seq": 7, "t": 1.2, "wall": 100.0},
    }
    evaluator = LoadedEndpointEvaluator(ik, guard, WholeBodyConfig())
    with patch("locomani.tasks.bottle.whole_body_policy.WholeBodyIK", return_value=body) as constructor:
        result = evaluator.evaluate(payload)
    return SimpleNamespace(
        result=result,
        pose=pose,
        body=body,
        ik=ik,
        guard=guard,
        support=support,
        payload=payload,
        constructor=constructor,
        reference_joints=reference_joints,
        measured_joints=measured_joints,
        measured_rotation=measured_rotation,
        reference_rotation=reference_rotation,
        solved_rotation=solved_rotation,
    )


@pytest.mark.parametrize("failure", ["collision", "hand", "foot", "com"])
def test_loaded_endpoint_requires_collision_support_and_grip_constraints(failure):
    fixture = _loaded_endpoint_fixture(failure)

    assert not fixture.result["admitted"]
    assert fixture.result["reason"] is not None
    np.testing.assert_array_equal(fixture.body.solve.call_args.args[0][1], fixture.payload["endpoint"])


def test_loaded_endpoint_uses_measured_tracking_offsets_for_collision_candidate():
    fixture = _loaded_endpoint_fixture()
    expected_joints = fixture.measured_joints + fixture.pose.joints - fixture.reference_joints
    expected_position = (
        fixture.payload["state"]["pelvis"]["pos"] + fixture.pose.position - fixture.payload["reference_x"][:3]
    )
    expected_rotation = fixture.solved_rotation @ fixture.reference_rotation.T @ fixture.measured_rotation

    assert fixture.result["admitted"]
    np.testing.assert_allclose(fixture.result["nominal_candidate_joints"], fixture.pose.joints)
    np.testing.assert_allclose(fixture.result["candidate_joints"], expected_joints)
    np.testing.assert_allclose(fixture.result["candidate_base"][:3], expected_position)
    np.testing.assert_allclose(
        Rotation.from_quat(fixture.result["candidate_base"][3:][[1, 2, 3, 0]]).as_matrix(),
        expected_rotation,
        atol=1e-12,
    )
    for call in fixture.guard.check.call_args_list:
        np.testing.assert_allclose(call.args[1], expected_joints)
        np.testing.assert_allclose(call.kwargs["candidate_base"], fixture.result["candidate_base"])


def test_loaded_endpoint_evaluator_copies_live_support_problem_and_reference_state():
    fixture = _loaded_endpoint_fixture()

    for name in ("heading", "feet_pos", "feet_rot", "com0", "lo", "hi"):
        np.testing.assert_allclose(getattr(fixture.body, name), fixture.support[name])
    np.testing.assert_allclose(fixture.result["reference_x"], fixture.payload["reference_x"])
    np.testing.assert_allclose(fixture.result["reference_root"], fixture.payload["reference_root"])
    np.testing.assert_allclose(
        fixture.result["support_problem"]["hand_weights"],
        fixture.support["hand_weights"],
    )
    assert fixture.result["support_com_weight"] == pytest.approx(fixture.support["com_weight"])
    assert fixture.constructor.call_args.kwargs["com_weight"] == pytest.approx(fixture.support["com_weight"])
    np.testing.assert_allclose(fixture.constructor.call_args.kwargs["hand_weights"], fixture.support["hand_weights"])


def test_loaded_support_snapshot_isolated_from_live_problem_mutation():
    from locomani.tasks.bottle.whole_body_policy import loaded_support_problem_snapshot

    body = SimpleNamespace(
        heading=Rotation.from_euler("z", 0.2).as_matrix(),
        feet_pos=np.ones((2, 3)),
        feet_rot=np.array([np.eye(3), np.eye(3)]),
        com0=np.array([0.1, -0.2]),
        lo=np.full(33, -1.0),
        hi=np.full(33, 1.0),
        hand_weights=np.array([0.4, 1.0]),
        com_weight=2.5,
    )
    snapshot = loaded_support_problem_snapshot(SimpleNamespace(body=body))
    body.heading[:] = 0.0
    body.feet_pos[:] = 9.0
    body.com0[:] = 8.0

    assert snapshot["com_weight"] == pytest.approx(2.5)
    np.testing.assert_allclose(snapshot["heading"], Rotation.from_euler("z", 0.2).as_matrix())
    np.testing.assert_allclose(snapshot["feet_pos"], np.ones((2, 3)))
    np.testing.assert_allclose(snapshot["com0"], [0.1, -0.2])


def _loaded_contract_fixture():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = WholeBodyConfig()
    policy.task = SimpleNamespace(target=SimpleNamespace(key="bottle"))
    tools = np.array([[0.20, 0.20, 0.80], [0.30, -0.20, 0.75]])
    feet = {"value": np.array([[0.10, 0.20, 0.0], [0.10, -0.20, 0.0]])}
    motion = SimpleNamespace(
        body=SimpleNamespace(tools=Mock(return_value=tools.copy())),
        x=np.zeros(33),
        feedback=np.zeros((2, 3)),
        measured_foot_positions=Mock(side_effect=lambda state: feet["value"].copy()),
        measured_foot_drift=Mock(return_value=np.array([0.03, 0.03])),
    )
    state = {
        "seq": 11,
        "t": 2.0,
        "wall": 1002.0,
        "joints": np.linspace(-0.3, 0.3, 29),
        "pelvis": {"pos": np.array([0.0, 0.0, 0.75]), "quat": np.array([1.0, 0.0, 0.0, 0.0])},
        "palms": {
            "right": {
                "pos": np.array([0.30, -0.20, 0.75]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }
        },
        "objects": {
            "bottle": {
                "pos": np.array([0.31, -0.20, 0.75]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "touch_links": ["R.dex1_finger_1", "R.dex1_finger_2"],
            }
        },
    }
    targets = tools.copy()
    rotations = np.array([np.eye(3), np.eye(3)])
    contract = {
        "targets": targets.copy(),
        "rotations": rotations.copy(),
        "loaded_start": tools[1].copy(),
        "root_position": state["pelvis"]["pos"].copy(),
        "root_rotation": np.eye(3),
        "measured_foot_positions": feet["value"].copy(),
        "palm_relative": np.array([0.01, 0.0, 0.0]),
        "source": {"seq": 10, "t": 1.9, "wall": 1001.9},
    }
    return policy, motion, state, targets, rotations, {"contract": contract}, feet


def test_loaded_ready_contract_tolerates_leg_churn_and_cumulative_foot_bias():
    policy, motion, state, targets, rotations, request, _ = _loaded_contract_fixture()
    state["joints"] += 0.4

    result = policy._loaded_endpoint_contract_difference(request, state, motion, targets, rotations)

    assert result["matches"]
    assert result["max_cumulative_foot_drift_m"] == pytest.approx(0.03)
    assert result["max_request_foot_motion_m"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "change",
    ["target", "target_rotation", "root", "root_yaw", "foot", "contact"],
)
def test_loaded_ready_contract_rejects_task_geometry_changes(change):
    policy, motion, state, targets, rotations, request, feet = _loaded_contract_fixture()
    if change == "target":
        targets[1, 0] += 0.04
    elif change == "target_rotation":
        rotations[1] = Rotation.from_euler("z", 0.4).as_matrix()
    elif change == "root":
        state["pelvis"]["pos"][0] += 0.04
    elif change == "root_yaw":
        state["pelvis"]["quat"] = Rotation.from_euler("z", 0.4).as_quat()[[3, 0, 1, 2]]
    elif change == "foot":
        feet["value"][0, 0] += 0.04
    else:
        state["objects"]["bottle"]["touch_links"] = []

    result = policy._loaded_endpoint_contract_difference(request, state, motion, targets, rotations)

    assert not result["matches"]
    assert (
        result["reason"]
        == {
            "target": "target",
            "target_rotation": "rotation_target",
            "root": "root_xy",
            "root_yaw": "root_yaw",
            "foot": "fixed_feet",
            "contact": "closed_grip",
        }[change]
    )
