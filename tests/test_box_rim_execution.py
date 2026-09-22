import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_fixture import DIRECTORY
from scipy.spatial.transform import Rotation

from locomani.tasks.crate import box_rim_execution as execution
from locomani.tasks.crate import box_rim_execution_cli as execution_cli
from locomani.tasks.crate.box_rim_execution import (
    HAND_OPEN,
    HAND_READY,
    BoxRimExecutionConfig,
    BoxRimExecutor,
)
from locomani.tasks.crate.box_rim_execution_cli import load_execution_request
from locomani.tasks.crate.box_rim_geometry import OpenBoxGeometry, RigidPose


class _Collision:
    def __init__(self):
        self.last_kwargs = None

    def check(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return (), ()


class _ProbeRuntime:
    def __init__(self, *, rise=True, lose_contact=False, walk_jump=0.0):
        self.rise = rise
        self.lose_contact = lose_contact
        self.walk_jump = walk_jump
        self.steps = 0
        self.stopped = False
        self.reference_velocities = []
        self.commanded_joints = []
        self.commanded_roots = []
        self.state = {
            "seq": 0,
            "t": 0.0,
            "wall": time.time(),
            "joints": np.zeros(29),
            "pelvis": {
                "pos": [0.0, -0.6, 0.793],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "yaw": 0.0,
                "vel": [0.0, 0.0, 0.0],
            },
            "palms": {
                "left": {
                    "pos": [0.0, 0.1, 0.8],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                    "hand_q": [0.025, 0.025],
                },
                "right": {
                    "pos": [0.0, -0.1, 0.8],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                    "hand_q": [0.025, 0.025],
                },
            },
            "objects": {
                "box": {
                    "pos": [0.0, 0.0, 0.62],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                    "touching": ["left", "right"],
                    "touch_links": ["L.dex1_finger_1", "R.dex1_finger_1"],
                }
            },
            "fallen": False,
            "furniture_touch": [],
        }

    def _advance(self):
        self.steps += 1
        self.state["seq"] += 1
        self.state["t"] += 0.05
        self.state["wall"] = time.time()

    def observe(self):
        return self.state

    def pose_step(self, joints, root_quaternion_wxyz, hands, *, reference_velocity=None):
        self.commanded_joints.append(np.asarray(joints).copy())
        self.commanded_roots.append(np.asarray(root_quaternion_wxyz).copy())
        self.reference_velocities.append(None if reference_velocity is None else np.asarray(reference_velocity).copy())
        self.state["joints"] = np.asarray(joints).copy()
        if self.rise and self.steps >= 2:
            dz = 0.020
            self.state["objects"]["box"]["pos"][2] = 0.62 + dz
            for palm in self.state["palms"].values():
                palm["pos"][2] = 0.8 + dz
        if self.lose_contact and self.steps >= 2:
            self.state["objects"]["box"]["touching"] = ["left"]
        self._advance()

    def walk_step(self, target_xy, target_yaw, speed):
        self.state["pelvis"]["pos"][0] += self.walk_jump
        self._advance()

    def stop_walk(self):
        self.stopped = True
        self._advance()

    def safe_stop(self):
        self.stopped = True

    def hold_ready(self):
        self._advance()

    def ready_step(self, joints, hands):
        self.pose_step(joints, self.state["pelvis"]["quat"], hands)

    def enter_pose(self, acceleration, joint_limits):
        return self.state["joints"].copy(), self.state["pelvis"]["quat"].copy()

    def close(self):
        return None


def _executor(runtime, **config):
    box = OpenBoxGeometry([0.32, 0.28, 0.18], 0.01, RigidPose([0, 0, 0.62], [1, 0, 0, 0]))
    planner = SimpleNamespace(
        collision=_Collision(),
        options=SimpleNamespace(
            target_contact_tolerance_m=0.04,
            max_target_penetration_m=0.001,
            support_baseline_max_penetration_m=0.006,
            support_penetration_growth_tolerance_m=0.0015,
        ),
        bimanual=SimpleNamespace(lo=np.full(29, -3.0), hi=np.full(29, 3.0)),
    )
    defaults = dict(probe_timeout_s=0.8, contact_hold_s=0.20)
    defaults.update(config)
    result = BoxRimExecutor(planner, box, runtime, "box", BoxRimExecutionConfig(**defaults))
    result._solve_at_observed_root = lambda state, targets, rotations, seed: np.asarray(seed).copy()
    return result


def test_probe_success_requires_sustained_measured_rise_and_bilateral_contacts():
    runtime = _ProbeRuntime(rise=True)
    controller = _executor(runtime)
    state = controller._observe(require_advance=False)
    targets = np.array([[0.0, 0.1, 0.8], [0.0, -0.1, 0.8]])
    controller._probe(state, targets, np.repeat(np.eye(3)[None], 2, axis=0))
    event = controller.journal.events[-1]
    assert event["kind"] == "measured_rise"
    assert event["rise_m"] >= controller.config.minimum_object_rise_m
    assert event["touching"] == ["left", "right"]
    assert runtime.state["t"] >= 0.25  # rise was maintained, not accepted on its first sample


def test_probe_rejects_lost_contact_even_when_object_briefly_rises():
    runtime = _ProbeRuntime(rise=True, lose_contact=True)
    controller = _executor(runtime)
    state = controller._observe(require_advance=False)
    with pytest.raises(execution._Failure, match="lost_contact_during_probe"):
        controller._probe(
            state,
            np.array([[0.0, 0.1, 0.8], [0.0, -0.1, 0.8]]),
            np.repeat(np.eye(3)[None], 2, axis=0),
        )


def test_stale_observation_and_reposition_overshoot_stop_with_typed_reasons():
    stale = _ProbeRuntime()
    stale.state["wall"] = time.time() - 5.0
    with pytest.raises(execution._Failure, match="stale_observation"):
        _executor(stale)._observe(require_advance=False)

    overshoot = _ProbeRuntime(walk_jump=0.25)
    controller = _executor(overshoot, maximum_total_reposition_m=0.22)
    with pytest.raises(execution._Failure, match="reposition_travel_budget_exceeded"):
        controller._walk_to(np.array([0.1, -0.6]), 0.0)
    assert overshoot.stopped


def test_reposition_stops_at_target_plane_and_records_achieved_pose():
    runtime = _ProbeRuntime(walk_jump=0.11)
    controller = _executor(runtime, maximum_total_reposition_m=0.22)
    state = controller._walk_to(np.array([0.1, -0.6]), 0.3)
    assert runtime.stopped
    assert state["pelvis"]["pos"][0] == pytest.approx(0.11)
    event = controller.journal.events[-1]
    assert event["kind"] == "brake_command"
    assert event["brake_xy"] == pytest.approx([0.11, -0.6])
    assert event["remaining_forward_m"] == pytest.approx(-0.01)
    assert event["yaw_error_rad"] == pytest.approx(0.3)


def test_observation_watchdog_ignores_sequence_progress_without_simulation_time_progress():
    runtime = _ProbeRuntime()
    controller = _executor(runtime)
    controller._observe(require_advance=False)
    runtime.state["seq"] += 1
    runtime.state["wall"] = time.time()
    controller._last_wall_progress = time.time() - controller.config.watchdog_wall_s - 0.01

    with pytest.raises(execution._Failure, match="observation_watchdog_timeout"):
        controller._observe()


def test_reposition_brakes_early_from_measured_forward_speed():
    runtime = _ProbeRuntime(walk_jump=0.04)
    runtime.state["pelvis"]["vel"] = [0.1, 0.0, 0.0]
    controller = _executor(runtime, maximum_total_reposition_m=0.22)
    controller._walk_to(np.array([0.1, -0.6]), 0.0)
    assert runtime.state["pelvis"]["pos"][0] == pytest.approx(0.04)
    event = controller.journal.events[-1]
    assert event["kind"] == "brake_command"
    assert event["remaining_forward_m"] == pytest.approx(0.06)
    assert event["braking_threshold_m"] == pytest.approx(0.075)


def test_failure_diagnostic_named_phase_does_not_destroy_journal():
    runtime = _ProbeRuntime()
    controller = _executor(runtime)

    def fail(_phase):
        raise execution._Failure("settle_timeout", phase="admission")

    controller._settle = fail
    result = controller.execute()
    assert not result.ok
    assert result.why == "settle_timeout"
    assert result.execution["events"][-1]["reported_phase"] == "admission"
    assert runtime.stopped


def test_reposition_previews_nearest_box_directed_stance_with_bounded_yaw():
    runtime = _ProbeRuntime()
    runtime.state["pelvis"]["yaw"] = 2.1
    runtime.state["pelvis"]["quat"] = Rotation.from_euler("z", 2.1).as_quat()[[3, 0, 1, 2]].tolist()
    controller = _executor(runtime)
    previews = []

    def plan(state, **_kwargs):
        previews.append(state)
        return SimpleNamespace(
            static_valid=True,
            reason="static_pose_found",
            selected_candidate="box-facing",
            to_dict=lambda: {"static_valid": True, "selected_candidate": "box-facing"},
        )

    controller.planner.plan = plan
    state = controller._observe(require_advance=False)
    target_xy, target_yaw, _ = controller._choose_reposition(state)
    np.testing.assert_allclose(target_xy, [0.0, -0.5], atol=1e-12)
    assert target_yaw == pytest.approx(2.1 - controller.config.maximum_reposition_yaw_change_rad)
    np.testing.assert_allclose(previews[0].pelvis_position[:2], target_xy)
    assert controller.journal.planning_attempts[0]["strategy"] == "toward_box"


def test_runtime_collision_checks_forward_all_four_measured_jaw_positions():
    runtime = _ProbeRuntime()
    runtime.state["palms"]["left"]["hand_q"] = [0.011, 0.012]
    runtime.state["palms"]["right"]["hand_q"] = [0.021, 0.022]
    controller = _executor(runtime)
    state = controller._observe(require_advance=False)

    controller._collision_reasons(state, state["joints"], np.zeros((2, 3)))

    np.testing.assert_allclose(
        controller.planner.collision.last_kwargs["gripper_q"],
        [0.011, 0.012, 0.021, 0.022],
    )


def test_supported_segment_refuses_com_violation_before_collision_admission():
    runtime = _ProbeRuntime()
    controller = _executor(runtime)
    state = controller._observe(require_advance=False)
    controller._support_diagnostics = lambda *_args: (
        {
            "foot_anchor_errors_m": [0.0, 0.0],
            "robot_com_shift_m": [0.0, 0.02],
            "minimum_joint_margin_rad": 0.1,
        },
        ["robot_com_shift"],
    )
    waypoint = {
        "segment": "above_rim",
        "step": 1,
        "joints": np.zeros(29),
        "pelvis_position": np.asarray(state["pelvis"]["pos"]),
        "pelvis_quaternion_wxyz": np.asarray(state["pelvis"]["quat"]),
    }

    with pytest.raises(execution._Failure, match="unsupported_swept_reference") as caught:
        controller._admit_supported_segment(
            state,
            np.zeros(29),
            np.asarray(state["pelvis"]["pos"]),
            np.asarray(state["pelvis"]["quat"]),
            waypoint,
            np.zeros((2, 3)),
            controller._commanded_gripper_q(HAND_OPEN),
        )

    assert caught.value.diagnostics["support_reasons"] == ["robot_com_shift"]


def _escape_contact(depth, *, geom_2="wrist"):
    return {
        "classification": "robot_self_collision",
        "geom_1": "hip",
        "geom_2": geom_2,
        "body_1": "right_hip_roll_link",
        "body_2": f"right_{geom_2}_link",
        "distance_m": -depth,
        "baseline_distance_m": None,
        "support_penetration_growth_m": None,
    }


def _escape_waypoint(state):
    joints = np.asarray(state["joints"], dtype=float).copy()
    joints[0] = 0.1
    return {
        "segment": "raised_front",
        "step": 1,
        "joints": joints,
        "pelvis_position": np.asarray(state["pelvis"]["pos"], dtype=float),
        "pelvis_quaternion_wxyz": np.asarray(state["pelvis"]["quat"], dtype=float),
    }


def _clear_support(*_args, **_kwargs):
    return (
        {
            "foot_anchor_errors_m": [0.0, 0.0],
            "robot_com_shift_m": [0.0, 0.0],
            "minimum_joint_margin_rad": 0.1,
        },
        [],
    )


def test_initial_escape_admits_only_monotonic_resolution_and_forbids_recontact():
    controller = _executor(_ProbeRuntime(), initial_escape_samples=5)
    state = controller._observe(require_advance=False)
    waypoint = _escape_waypoint(state)
    targets = np.zeros((2, 3))
    controller._support_diagnostics = _clear_support

    def clears(_state, joints, _targets, **_kwargs):
        contacts = [_escape_contact(0.0001)] if joints[0] < 0.01 else []
        return [], contacts

    controller._collision_reasons = clears
    contacts = controller._admit_initial_escape(state, waypoint, targets)
    assert contacts == {("hip", "wrist"): pytest.approx(0.0001)}
    assert controller.journal.events[-1]["cleared_fraction"] == pytest.approx(0.25)

    def recontacts(_state, joints, _targets, **_kwargs):
        if joints[0] < 0.01:
            contacts = [_escape_contact(0.0001)]
        elif joints[0] >= 0.05:
            contacts = [_escape_contact(0.00001)]
        else:
            contacts = []
        return [], contacts

    controller._collision_reasons = recontacts
    with pytest.raises(execution._Failure, match="initial_escape_penetration_increased"):
        controller._admit_initial_escape(state, waypoint, targets)


def test_initial_escape_tracks_a_graze_on_other_links_but_rejects_a_deep_new_pair():
    """A new self-contact within the escape depth envelope is tracked like the initial ones (it may
    not deepen, and must clear); a new pair beyond the envelope is a collision."""
    runtime = _ProbeRuntime(rise=False)
    controller = _executor(runtime, initial_escape_samples=5)
    state = controller._observe(require_advance=False)
    waypoint = _escape_waypoint(state)
    targets = np.zeros((2, 3))
    controller._support_diagnostics = _clear_support

    def graze_then_clear(_state, joints, _targets, **_kwargs):
        if 0.05 <= joints[0] < 0.08:
            return [], [_escape_contact(0.00001, geom_2="finger")]
        return [], [_escape_contact(0.0001)] if joints[0] < 0.05 else []

    controller._collision_reasons = graze_then_clear
    controller._admit_initial_escape(state, waypoint, targets)

    def graze_deepens(_state, joints, _targets, **_kwargs):
        if joints[0] >= 0.05:
            return [], [_escape_contact(0.00001 + 0.01 * (float(joints[0]) - 0.05), geom_2="finger")]
        return [], [_escape_contact(0.0001)]

    controller._collision_reasons = graze_deepens
    with pytest.raises(execution._Failure, match="initial_escape_penetration_increased"):
        controller._admit_initial_escape(state, waypoint, targets)

    def deep_new_pair(_state, joints, _targets, **_kwargs):
        if joints[0] >= 0.05:
            return [], [_escape_contact(0.002, geom_2="finger")]
        return [], [_escape_contact(0.0001)]

    controller._collision_reasons = deep_new_pair
    with pytest.raises(execution._Failure, match="initial_escape_new_collision"):
        controller._admit_initial_escape(state, waypoint, targets)


def test_initial_escape_rejects_new_pair_and_tracks_measured_clearance():
    runtime = _ProbeRuntime(rise=False)
    controller = _executor(runtime, initial_escape_samples=5)
    state = controller._observe(require_advance=False)
    waypoint = _escape_waypoint(state)
    targets = np.zeros((2, 3))
    controller._support_diagnostics = _clear_support

    def adds_pair(_state, joints, _targets, **_kwargs):
        contacts = [_escape_contact(0.0001)]
        if joints[0] >= 0.05:
            contacts = [_escape_contact(0.002, geom_2="finger")]  # beyond the escape depth envelope
        return [], contacts

    controller._collision_reasons = adds_pair
    with pytest.raises(execution._Failure, match="initial_escape_new_collision"):
        controller._admit_initial_escape(state, waypoint, targets)

    def measured_clearance(_state, joints, _targets, **_kwargs):
        penetration = max(0.0, 0.0001 - 0.1 * float(joints[0]))
        return [], [_escape_contact(penetration)] if penetration > 0.0 else []

    controller._collision_reasons = measured_clearance
    final = controller._move_initial_escape(
        state,
        waypoint,
        targets,
        {("hip", "wrist"): 0.0001},
    )
    assert final["joints"][0] == pytest.approx(0.1)
    assert controller.journal.events[-1]["kind"] == "measured_clear"
    assert any("initial_escape_contacts" in trace for trace in controller.journal.traces)


def test_escape_tracking_envelope_is_bounded_and_reference_remains_strict():
    controller = _executor(_ProbeRuntime())
    initial = {("hip", "wrist"): 0.0001}
    cleared = {("hip", "wrist"): 0.0}
    contacts = [_escape_contact(0.0002)]
    with pytest.raises(execution._Failure, match="initial_escape_penetration_increased"):
        controller._resolving_contact_depths(contacts, initial, cleared, phase="reference", progress=0.01)
    measured = controller._resolving_contact_depths(
        contacts, initial, cleared, phase="observed", progress=0.01, tracking=True
    )
    assert measured[("hip", "wrist")] == pytest.approx(0.0002)
    with pytest.raises(execution._Failure, match="initial_escape_penetration_increased"):
        controller._resolving_contact_depths(
            [_escape_contact(0.00036)], initial, measured, phase="observed", progress=0.02, tracking=True
        )
    # A graze on other links inside the envelope is tracked from its own depth; a deep new pair is not.
    grazed = controller._resolving_contact_depths(
        [_escape_contact(0.00001, geom_2="finger")], initial, measured, phase="observed", progress=0.02, tracking=True
    )
    assert grazed[("finger", "hip")] == pytest.approx(0.00001) and ("finger", "hip") in initial
    with pytest.raises(execution._Failure, match="initial_escape_new_collision"):
        controller._resolving_contact_depths(
            [_escape_contact(0.002, geom_2="finger_b")],
            initial,
            measured,
            phase="observed",
            progress=0.02,
            tracking=True,
        )


def test_planner_approach_keeps_commands_on_planner_channel():
    runtime = execution.SonicBoxRimRuntime.__new__(execution.SonicBoxRimRuntime)
    runtime.approach_mode = "planner_arms"
    calls = []
    runtime.stop_walk = lambda: calls.append("stop_walk")
    runtime.ready_step = lambda joints, hands: calls.append((joints.copy(), hands.copy()))
    runtime.enter_pose(0.7, (np.zeros(29), np.ones(29)))
    runtime.pose_step(np.zeros(29), np.array([1, 0, 0, 0]), HAND_OPEN)
    assert calls[0] == "stop_walk"
    np.testing.assert_array_equal(calls[1][1], HAND_OPEN)
    assert not hasattr(runtime, "pose_arm")
    with pytest.raises(ValueError, match="approach_mode"):
        BoxRimExecutionConfig(approach_mode="unknown")


def test_initial_ready_reference_survives_planner_takeover_and_tracks_later_hand_commands(
    monkeypatch,
):
    starts = []
    pushes = []

    class FakePlannerArm:
        def __init__(self, *_args, **_kwargs):
            self.q = np.zeros(29)
            self.hands = {}
            self.mode = None

        def start(self, **kwargs):
            starts.append(kwargs)

        def _push(self, q, **kwargs):
            pushes.append((np.asarray(q).copy(), kwargs))

    monkeypatch.setattr(execution, "PlannerArm", FakePlannerArm)
    reference = np.linspace(-0.2, 0.2, 29)
    primitives = SimpleNamespace(obs=lambda: {"joints": reference.copy()})
    runtime = execution.SonicBoxRimRuntime(
        primitives,
        ik=object(),
        approach_mode="legacy_low_reach",
        initial_reference=reference,
        initial_hands=HAND_OPEN,
    )

    assert starts[0]["settle_seconds"] == 0.0
    np.testing.assert_array_equal(runtime.planner_arm.q, reference)
    np.testing.assert_array_equal(runtime.ready_hands, HAND_OPEN)
    np.testing.assert_array_equal(runtime.planner_arm.hands["left"], HAND_OPEN)
    runtime.hold_ready()
    np.testing.assert_array_equal(runtime.planner_arm.hands["right"], HAND_OPEN)

    runtime.ready_step(reference, HAND_READY)
    runtime.hold_ready()
    np.testing.assert_array_equal(runtime.ready_hands, HAND_READY)
    np.testing.assert_array_equal(runtime.planner_arm.hands["left"], HAND_READY)
    assert pushes


def test_planner_arm_inherited_hold_uses_legacy_push_contract():
    frames = []
    command = SimpleNamespace(
        set_upper_body=lambda value: frames.append(np.asarray(value).copy()),
        set_upper_body_vel=lambda _value: None,
        set_hands=lambda **_kwargs: None,
        frame=lambda **_kwargs: None,
    )
    arm = execution.PlannerArm.__new__(execution.PlannerArm)
    arm.p = SimpleNamespace(cmd=command)
    arm.q = np.linspace(-0.2, 0.2, 29)
    arm.send_vel = False
    arm.guard = None
    arm.left_guard = False
    arm.hands = {"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()}
    arm.mode = execution.LocomotionMode.IDLE
    arm._face = (0.0, 1.0, 0.0)
    arm.height = -1.0
    arm.i = 0
    arm.next_t = 0.0

    arm.hold(0.02)

    assert len(frames) == 1


def test_initial_ready_reset_applies_only_upper_dofs_and_keeps_open_hands(monkeypatch):
    ready_pose = {
        "waist": [0.0, 0.0, 0.0],
        "left_arm": [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "right_arm": [-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "hands": HAND_OPEN.tolist(),
    }
    events = []

    class FakeCommand:
        def reset_pose(self):
            events.append(("reset_pose",))

        def set_hands(self, **kwargs):
            events.append(("hands", kwargs))

        def exit_streamed(self):
            events.append(("exit_streamed",))

        def hold(self, seconds, **kwargs):
            events.append(("hold", seconds, kwargs))

        def start_control(self):
            events.append(("start_control",))

    class FakePrimitives:
        def __init__(self):
            self.cmd = FakeCommand()
            self.hold_yaw = None
            self.state = {
                "joints": np.linspace(-0.1, 0.1, 29),
                "pelvis": {"yaw": np.pi / 2},
            }

        def obs(self):
            return self.state

        def _facing(self, _state):
            return (0.0, 1.0, 0.0)

    class FakeReset:
        def pin_arms(self, **kwargs):
            events.append(("pin", kwargs))

        def reset(self, **kwargs):
            events.append(("reset", kwargs))

    monkeypatch.setattr(execution_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        execution_cli.upper_ref,
        "shared",
        lambda: SimpleNamespace(wait=lambda _seconds: events.append(("upper_wait",))),
    )
    monkeypatch.setattr(execution_cli, "reanchor", lambda _primitives: events.append(("reanchor",)))
    request = {
        "runtime": {
            "reset_before_episode": True,
            "initial_ready_pose": ready_pose,
            "robot_dxy": [0.0, 0.375],
            "robot_yaw": np.pi / 2,
            "video_label": "square_ready",
        }
    }
    initial = execution_cli._initialise(request, FakePrimitives(), FakeReset())

    hand_event = next(event for event in events if event[0] == "hands")
    np.testing.assert_array_equal(hand_event[1]["left"], HAND_OPEN)
    pin_event = next(event for event in events if event[0] == "pin")
    assert pin_event[1] == {
        "waist": ready_pose["waist"],
        "left": ready_pose["left_arm"],
        "right": ready_pose["right_arm"],
    }
    reset_event = next(event for event in events if event[0] == "reset")
    robot = reset_event[1]["robot"]
    assert robot["dof_from"] == 12
    assert len(robot["dof"]) == 29
    np.testing.assert_array_equal(robot["dof"][:12], np.zeros(12))
    np.testing.assert_array_equal(
        robot["dof"][12:],
        ready_pose["waist"] + ready_pose["left_arm"] + ready_pose["right_arm"],
    )
    np.testing.assert_array_equal(initial["reference"][:12], np.linspace(-0.1, 0.1, 29)[:12])
    np.testing.assert_array_equal(initial["hands"], HAND_OPEN)
    start_index = next(i for i, event in enumerate(events) if event[0] == "start_control")
    wait_index = next(i for i, event in enumerate(events) if event[0] == "upper_wait")
    reanchor_index = next(i for i, event in enumerate(events) if event[0] == "reanchor")
    holds = [i for i, event in enumerate(events) if event[0] == "hold"]
    final_hold_index, calibration_hold_index = holds[-2], holds[-1]
    assert start_index < wait_index < reanchor_index < final_hold_index < calibration_hold_index
    assert events[final_hold_index][1] == pytest.approx(4.5)
    assert events[calibration_hold_index][1] == pytest.approx(1.5)  # heading-offset calibration hold
    np.testing.assert_allclose(
        events[final_hold_index][2]["facing"],
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_low_reach_bend_starts_from_handoff_and_sends_smooth_reference_velocity():
    runtime = _ProbeRuntime()
    controller = _executor(
        runtime,
        approach_mode="legacy_low_reach",
        low_reach_minimum_bend_s=0.4,
        maximum_joint_speed_rps=1.0,
        maximum_joint_acceleration_rps2=2.0,
    )
    state = controller._observe(require_advance=False)
    controller.journal.selected_plan = {"targets_world_m": np.zeros((2, 3)).tolist()}
    controller._support_diagnostics = _clear_support
    controller._collision_reasons = lambda *_args, **_kwargs: ([], [])
    controller._check_box_not_pushed = lambda _state: None
    q0 = state["joints"].copy()
    q0[0] = 0.02
    q1 = q0.copy()
    q1[0] = 0.12
    waypoint = {
        "joints": q1,
        "pelvis_quaternion_wxyz": state["pelvis"]["quat"].copy(),
    }

    final = controller._move_low_reach_bend(
        state,
        (q0, state["pelvis"]["quat"].copy()),
        waypoint,
    )

    assert final["joints"][0] == pytest.approx(0.12)
    assert any(velocity is not None and np.max(np.abs(velocity)) > 0 for velocity in runtime.reference_velocities)
    np.testing.assert_array_equal(runtime.reference_velocities[-1], np.zeros(29))
    assert controller.journal.events[-1]["duration_s"] >= 0.4


def test_streamed_runtime_forwards_low_reach_reference_velocity():
    runtime = execution.SonicBoxRimRuntime.__new__(execution.SonicBoxRimRuntime)
    runtime.approach_mode = "legacy_low_reach"
    pushed = []
    runtime.pose_arm = SimpleNamespace(
        q=np.zeros(29),
        root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        hands={},
        _push=lambda q, **kwargs: pushed.append((q.copy(), kwargs)),
    )
    velocity = np.linspace(0.0, 0.28, 29)
    runtime.pose_step(np.ones(29), np.array([1, 0, 0, 0]), HAND_OPEN, reference_velocity=velocity)

    np.testing.assert_array_equal(pushed[0][0], np.ones(29))
    np.testing.assert_array_equal(pushed[0][1]["reference_velocity"], velocity)


def test_low_reach_arm_tracking_retains_commanded_legs_and_root_when_measurement_lags():
    runtime = _ProbeRuntime()
    controller = _executor(
        runtime,
        approach_mode="legacy_low_reach",
        maximum_joint_speed_rps=1.0,
        maximum_joint_acceleration_rps2=2.0,
        waypoint_settle_s=0.1,
    )
    state = controller._observe(require_advance=False)
    controller.journal.selected_plan = {"targets_world_m": np.zeros((2, 3)).tolist()}
    controller._collision_reasons = lambda *_args, **_kwargs: ([], [])
    controller._support_diagnostics = _clear_support
    controller._check_box_not_pushed = lambda _state: None
    reference = state["joints"].copy()
    reference[0] = 0.25
    target = reference.copy()
    target[15] = 0.1
    root = Rotation.from_euler("y", 0.2).as_quat()[[3, 0, 1, 2]]
    controller._low_reach_reference_joints = reference.copy()
    controller._low_reach_root_quaternion = root.copy()
    state["joints"][0] = -0.25
    state["pelvis"]["quat"] = np.array([1.0, 0.0, 0.0, 0.0])

    controller._move_reference("align", state, target, HAND_OPEN, target_root_quat=root)

    assert runtime.commanded_joints
    assert all(command[0] == pytest.approx(0.25) for command in runtime.commanded_joints)
    for commanded_root in runtime.commanded_roots:
        np.testing.assert_allclose(commanded_root, root, atol=1e-8)


def test_low_reach_reference_ignores_only_intermediate_foot_floor_interpolation_contact():
    controller = _executor(_ProbeRuntime(), approach_mode="legacy_low_reach", swept_samples=3)
    state = controller._observe(require_advance=False)
    controller._support_diagnostics = _clear_support
    waypoint = _escape_waypoint(state)
    waypoint["segment"] = "low_reach_bend"

    def collision(_state, joints, _targets, **_kwargs):
        if 0 < joints[0] < waypoint["joints"][0]:
            return [], [
                {
                    "classification": "robot_environment_collision",
                    "geom_1": "floor",
                    "geom_2": "foot_geom",
                    "body_1": "world",
                    "body_2": "left_ankle_roll_link",
                    "distance_m": -0.003,
                }
            ]
        return [], []

    controller._collision_reasons = collision
    controller._admit_low_reach_bend(
        state,
        state["joints"],
        state["pelvis"]["pos"],
        state["pelvis"]["quat"],
        waypoint,
        np.zeros((2, 3)),
        np.zeros(4),
    )

    target_contact = {
        "classification": "non_pad_target_collision",
        "geom_1": "box_wall",
        "geom_2": "finger",
        "body_1": "box",
        "body_2": "right_dex1_finger_2_link",
        "distance_m": -0.002,
    }
    controller._collision_reasons = lambda *_args, **_kwargs: ([], [target_contact])
    with pytest.raises(execution._Failure, match="unsafe_swept_reference"):
        controller._admit_low_reach_bend(
            state,
            state["joints"],
            state["pelvis"]["pos"],
            state["pelvis"]["quat"],
            waypoint,
            np.zeros((2, 3)),
            np.zeros(4),
        )


def test_low_reach_observed_bend_does_not_exempt_foot_floor_collision():
    runtime = _ProbeRuntime()
    controller = _executor(
        runtime,
        approach_mode="legacy_low_reach",
        low_reach_minimum_bend_s=0.2,
        maximum_joint_speed_rps=1.0,
        maximum_joint_acceleration_rps2=2.0,
    )
    state = controller._observe(require_advance=False)
    controller.journal.selected_plan = {"targets_world_m": np.zeros((2, 3)).tolist()}
    controller._check_box_not_pushed = lambda _state: None
    controller._support_diagnostics = _clear_support
    controller._collision_reasons = lambda *_args, **_kwargs: (
        [],
        [
            {
                "classification": "robot_environment_collision",
                "geom_1": "floor",
                "geom_2": "foot_geom",
                "body_1": "world",
                "body_2": "left_ankle_roll_link",
                "distance_m": -0.003,
            }
        ],
    )
    waypoint = {
        "joints": state["joints"].copy(),
        "pelvis_quaternion_wxyz": state["pelvis"]["quat"].copy(),
    }
    waypoint["joints"][0] = 0.1

    with pytest.raises(execution._Failure, match="observed_low_reach_bend_rejected"):
        controller._move_low_reach_bend(
            state,
            (state["joints"], state["pelvis"]["quat"]),
            waypoint,
        )


def test_wrapper_dry_run_validates_request_and_writes_nothing(tmp_path):
    request = {
        "schema_version": 1,
        "scene_xml": str((DIRECTORY / "scene_locomani.xml").resolve()),
        "target": {"body": "water_bottle_object", "collider_mode": "matched_target"},
        "box": {
            "size_m": [0.32, 0.28, 0.18],
            "wall_thickness_m": 0.01,
            "pose": {"position": [0, 0, 0.62], "quaternion_wxyz": [1, 0, 0, 0]},
        },
        "robot_state": {"source": "scene_default"},
        "target_key": "water_bottle",
        "execution": {},
        "runtime": {
            "backend": {"schema_version": 1, "no_encoder": False},
            "gpu": "0",
            "instance": 2,
            "scene_dir": str(DIRECTORY.resolve()),
        },
    }
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "must_not_exist.json"
    request_path.write_text(json.dumps(request))
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "probe_box_grasp.py"),
            str(request_path),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not output_path.exists()
    payload = json.loads(completed.stdout)
    assert payload["instance"] == 2
    assert payload["scene_dir"] == str(DIRECTORY.resolve())

    ready_pose = {
        "waist": [0.0, 0.0, 0.0],
        "left_arm": [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "right_arm": [-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "hands": HAND_OPEN.tolist(),
    }
    with_ready = copy.deepcopy(request)
    with_ready["runtime"]["initial_ready_pose"] = ready_pose
    request_path.write_text(json.dumps(with_ready))
    parsed = load_execution_request(request_path)
    assert parsed["runtime"]["initial_ready_pose"] == ready_pose
    np.testing.assert_array_equal(
        execution_cli._ready_reference(np.arange(29, dtype=float), ready_pose)[12:],
        np.asarray(ready_pose["waist"] + ready_pose["left_arm"] + ready_pose["right_arm"]),
    )

    invalid = copy.deepcopy(with_ready)
    invalid["runtime"]["initial_ready_pose"]["hands"] = [0.0, 0.0]
    request_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="initial_ready_pose.hands must contain 7"):
        load_execution_request(request_path)

    invalid = copy.deepcopy(with_ready)
    invalid["runtime"]["reset_before_episode"] = False
    request_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="initial_ready_pose requires reset_before_episode"):
        load_execution_request(request_path)

    invalid = copy.deepcopy(request)
    invalid["runtime"]["backend"]["no_encoder"] = True
    request_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="no_encoder must be false"):
        load_execution_request(request_path)

    invalid = copy.deepcopy(request)
    invalid["runtime"]["reset_before_episode"] = "false"
    request_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="reset_before_episode must be boolean"):
        load_execution_request(request_path)

    invalid = copy.deepcopy(request)
    invalid["runtime"]["robot_dxy"] = [float("inf"), 0.0]
    request_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="robot_dxy and robot_yaw must be finite"):
        load_execution_request(request_path)

    invalid = copy.deepcopy(request)
    invalid["target_key"] = "missing"
    request_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="target_key must name an object"):
        load_execution_request(request_path)

    invalid = copy.deepcopy(request)
    invalid["target"]["body"] = "another_body"
    request_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="target.body must match"):
        load_execution_request(request_path)


def test_execution_config_requires_real_boolean_and_finite_braking_horizon():
    with pytest.raises(ValueError, match="enable_reposition must be boolean"):
        BoxRimExecutionConfig(enable_reposition="false")
    with pytest.raises(ValueError, match="braking_horizon_s must be finite nonnegative"):
        BoxRimExecutionConfig(reposition_braking_horizon_s=float("inf"))
    assert BoxRimExecutionConfig(reposition_braking_horizon_s=0).reposition_braking_horizon_s == 0


def test_initial_ready_pins_release_if_runtime_takeover_fails(tmp_path, monkeypatch):
    (tmp_path / "scene_meta.json").write_text(json.dumps({"robot_start": {"xy": [0, 0]}}))
    pin_calls = []

    class FakeReset:
        def pin_arms(self, *args, **kwargs):
            pin_calls.append((args, kwargs))

        def close(self):
            pass

    class FakePrimitives:
        def close(self):
            pass

    ready_pose = {
        "waist": [0.0, 0.0, 0.0],
        "left_arm": [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "right_arm": [-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "hands": HAND_OPEN.tolist(),
    }
    request = {
        "runtime": {
            "scene_dir": tmp_path,
            "initial_ready_pose": ready_pose,
            "video_label": None,
        },
        "scene_xml": tmp_path / "scene.xml",
        "robot_raw": {},
        "target": {"body": "box"},
        "box": object(),
        "planner": object(),
        "execution": SimpleNamespace(approach_mode="legacy_low_reach"),
        "target_key": "box",
    }
    initial = {
        "reference": np.zeros(29),
        "hands": HAND_OPEN.copy(),
        "requested": ready_pose,
    }
    monkeypatch.setattr(execution_cli, "require_pose_ownership", lambda: None)
    monkeypatch.setattr(execution_cli, "ArmIK", lambda _path: object())
    monkeypatch.setattr(execution_cli.box_rim_cli, "_parse_robot", lambda *_args: None)
    monkeypatch.setattr(execution_cli, "BoxRimPlanner", lambda *_args: object())
    monkeypatch.setattr(execution_cli, "Primitives", lambda **_kwargs: FakePrimitives())
    monkeypatch.setattr(execution_cli, "EpisodeControlClient", FakeReset)
    monkeypatch.setattr(execution_cli, "_initialise", lambda *_args: initial)

    class FailedTakeover:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("takeover failed")

    monkeypatch.setattr(execution_cli, "SonicBoxRimRuntime", FailedTakeover)
    with pytest.raises(RuntimeError, match="takeover failed"):
        execution_cli.run_live(request)
    assert pin_calls == [((), {})]


def test_coordinated_runtime_collision_checks_select_diagnostic_floor_support_policy():
    controller = _executor(_ProbeRuntime(), approach_mode="coordinated")
    state = controller._observe(require_advance=False)

    controller._collision_reasons(state, state["joints"], np.zeros((2, 3)))

    assert controller.planner.collision.last_kwargs["support_contact_policy"] == "diagnostic_floor"
