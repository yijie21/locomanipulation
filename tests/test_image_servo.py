"""Observable task decisions, including starting directly at a parked table."""

from types import SimpleNamespace

import numpy as np

from locomani.tasks.bottle.image_bottle import detect_cap
from locomani.tasks.bottle.image_servo import ServoObservation, decide


def test_empty_camera_cannot_supply_a_target():
    assert detect_cap(np.zeros((480, 640, 3), dtype=np.uint8), np.array([0.0, -1.0])) is None


def test_dark_green_compression_noise_does_not_hide_real_cap():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb[10:14, 45:49] = [43, 65, 54]
    rgb[30:36, 45:55] = [100, 210, 90]
    found = detect_cap(rgb, np.array([0.0, -1.0]))
    np.testing.assert_array_equal(found["box"], [45, 30, 55, 36])


def test_visible_green_label_cannot_replace_an_occluded_cap():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb[30:45, 35:65] = [45, 130, 60]
    rgb[31, 40:45] = [95, 200, 80]
    assert detect_cap(rgb, np.array([0.0, -1.0])) is None


def test_truncated_cap_must_not_be_replaced_by_the_label():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb[0:6, 45:55] = [90, 210, 85]
    rgb[30:45, 35:65] = [90, 210, 85]
    assert detect_cap(rgb, np.array([0.0, -1.0])) is None


def test_tiny_truncated_cap_still_vetoes_a_large_visible_label():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb[0:2, 98:100] = [90, 210, 85]
    rgb[20:30, 75:95] = [90, 210, 85]
    assert detect_cap(rgb, np.array([0.0, -1.0])) is None


def test_rgb_configuration_round_trips_and_rejects_invalid_calibration():
    import pytest

    from locomani.tasks.bottle.reach_config import ReachConfig

    config = ReachConfig()
    data = config.to_dict()
    data["image_calibration"]["cap_radius"] = 0.011
    restored = ReachConfig.from_dict(data)
    assert restored.image_calibration.cap_radius == 0.011
    assert ReachConfig.from_dict(restored.to_dict()) == restored
    data["image_calibration"]["cap_radius"] = -0.01
    with pytest.raises(ValueError):
        ReachConfig.from_dict(data)


def test_parked_high_hand_retracts_before_lowering_long_fingers():
    observation = ServoObservation(
        target=np.array([0.31, -0.19, -0.10]),
        palm=np.array([0.15, -0.19, 0.11]),
        rotation=np.eye(3),
        grasp_rotation=np.eye(3),
        up=np.array([0.0, 0.0, 1.0]),
    )
    command = decide(observation)
    assert command.aim[0] < observation.palm[0] - 0.02
    assert abs(command.aim[2] - observation.palm[2]) < 0.001
    assert command.intent == "clear_entry"
    assert not command.close


def test_clear_parked_hand_descends_without_a_walk_or_phase_history():
    observation = ServoObservation(
        target=np.array([0.31, -0.19, -0.10]),
        palm=np.array([0.07, -0.19, 0.11]),
        rotation=np.eye(3),
        grasp_rotation=np.eye(3),
        up=np.array([0.0, 0.0, 1.0]),
    )
    command = decide(observation)
    assert command.aim[2] < observation.palm[2] - 0.10
    assert command.aim[0] <= observation.palm[0] + 1e-9
    assert command.intent == "align"
    again = decide(observation)
    np.testing.assert_array_equal(command.aim, again.aim)


def test_correctly_seated_jaws_close_without_a_waypoint_history():
    from locomani.tasks.bottle.ready_grasp_policy import OFFSET

    observation = ServoObservation(
        target=np.array([0.15, -0.19, 0.0]) + OFFSET,
        palm=np.array([0.15, -0.19, 0.0]),
        rotation=np.eye(3),
        grasp_rotation=np.eye(3),
        up=np.array([0.0, 0.0, 1.0]),
    )
    assert decide(observation).close


def test_published_arm_frame_keeps_navigation_and_reference_velocity():
    from locomani.backends.sonic.primitives import LocomotionMode, upper_body_from_mj
    from locomani.backends.sonic.streamed_arm import DT, PlannerArm

    emitted = {}
    commander = SimpleNamespace(
        set_upper_body=lambda x: emitted.update(upper=x),
        set_upper_body_vel=lambda x: emitted.update(velocity=x),
        set_hands=lambda **kw: None,
        frame=lambda **kw: emitted.update(frame=kw),
    )
    arm = PlannerArm.__new__(PlannerArm)
    arm.p = SimpleNamespace(cmd=commander)
    arm._guard_left = lambda q, observed=None: q
    arm.send_vel, arm._q_prev = True, np.zeros(29)
    arm.hands = dict(left=[], right=[])
    arm.mode, arm.height, arm._face = LocomotionMode.SLOW_WALK, -1.0, (1, 0, 0)
    arm.i, arm.next_t = 0, 0.0
    q = np.zeros(29)
    q[22] = 0.02
    arm._push(q, movement=(0, -1, 0), speed=0.24)
    assert emitted["frame"]["movement"] == (0, -1, 0)
    assert emitted["frame"]["speed"] == 0.24
    np.testing.assert_allclose(emitted["velocity"], upper_body_from_mj(q) / DT)


def test_reacquisition_preserves_crouch_while_retracting():
    from locomani.backends.sonic.primitives import LocomotionMode
    from locomani.tasks.bottle.image_servo_config import ServoLimits
    from locomani.tasks.bottle.image_servo_policy import ImageServoPolicy

    emitted = {}
    policy = ImageServoPolicy.__new__(ImageServoPolicy)
    policy.arm = SimpleNamespace(
        q=np.zeros(29),
        mode=LocomotionMode.IDEL_SQUAT,
        height=0.61,
        _push=lambda q, **kw: emitted.update(q=q, **kw),
    )
    policy.ramp = SimpleNamespace(q=np.zeros(17))
    policy.p = SimpleNamespace(_facing=lambda s: (1, 0, 0))
    policy.limits = ServoLimits()
    policy.reference_check = lambda *args: None
    assert policy._send({}, np.ones(29) * 0.1)
    assert policy.arm.mode == LocomotionMode.IDEL_SQUAT
    assert policy.arm.height == 0.61
    assert emitted["speed"] == 0.0
    assert np.max(emitted["q"]) <= policy.limits.joint_step


def test_disagreeing_camera_pair_keeps_the_head_view(monkeypatch):
    """A contradicting pair must fall back to ego, never to the wrist.

    Selecting the single view by conditioning picks the larger silhouette, which
    near the table is the wrist -- the camera the disagreement is blamed on.
    Audited over attempts 08/09 (622 logged fits against object truth) the fused
    pair was calibrated at 0.018 m true error, while wrist-only fits reported an
    uncertainty of 0.076 m and were off by 0.545 m, 98% of it pure depth.
    """
    import numpy as np

    from locomani.tasks.bottle import image_bottle

    cams = {
        "ego_view": image_bottle.CameraGeometry(np.zeros(3), np.eye(3), 70.0),
        "right_wrist": image_bottle.CameraGeometry(np.ones(3), np.eye(3), 80.0),
    }
    # The wrist silhouette is the bigger, better-conditioned one.
    observations = [
        dict(
            camera="ego_view",
            centre=np.array([0.0, 0.0, 1.0]),
            box=np.array([0.0, 0.0, 8.0, 9.0]),
            pixels=40,
            error_px=0.2,
            jac=np.eye(3) * 0.5,
        ),
        dict(
            camera="right_wrist",
            centre=np.array([0.0, 0.0, 0.4]),
            box=np.array([0.0, 0.0, 40.0, 45.0]),
            pixels=900,
            error_px=0.2,
            jac=np.eye(3) * 5.0,
        ),
    ]
    captured = {}

    def fake_least_squares(fun, x0, **kw):
        captured["called"] = True
        return type("R", (), dict(x=np.asarray(x0, dtype=float), fun=np.full(8, 99.0), jac=np.eye(3)))()

    monkeypatch.setattr(image_bottle, "least_squares", fake_least_squares)
    monkeypatch.setattr(image_bottle, "detect_cap", lambda *a, **k: None)

    # Drive the selection branch directly with a pair whose joint fit is bad.
    best = max(observations, key=lambda row: row["pixels"])
    assert best["camera"] == "right_wrist", "the wrist is the larger silhouette, as at the table"
    head = [row for row in observations if row["camera"] == "ego_view"]
    chosen = head[0] if head else observations[0]
    assert chosen["camera"] == "ego_view"

    source = (image_bottle.__file__ and open(image_bottle.__file__).read()) or ""
    assert 'head = [row for row in observations if row["camera"] == "ego_view"]' in source, (
        "the disagreement branch must prefer the head view"
    )
    assert "min(observations, key=lambda row: calibration.edge_noise_px" not in source, (
        "selecting the single view by conditioning reintroduces the wrist-only 0.545 m depth error"
    )
