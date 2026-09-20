import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.tasks.crate import box_rim_execution as execution
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.box_rim_execution import HAND_OPEN, BoxRimExecutionConfig, BoxRimExecutor
from locomani.tasks.crate.box_rim_geometry import OpenBoxGeometry, RigidPose
from locomani.tasks.crate.coordinated_reach import CoordinatedReach, CoordinatedTarget
from locomani.tasks.crate.perception import ApproachTarget, CrateEstimate
from locomani.tasks.crate.visual_approach import VisualApproach
from locomani.tasks.crate.visual_low_reach import LowReachCommand


def _box(yaw=0.0, forbidden_intervals=()):
    quat = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
    return OpenBoxGeometry(
        [0.45, 0.32, 0.33],
        0.01,
        RigidPose([0.3, -0.2, 0.4], quat),
        local_com_m=[0, 0, 0.165],
        forbidden_intervals=forbidden_intervals,
    )


@pytest.mark.parametrize(
    ("rim_pair", "local_points", "expected_frame"),
    [
        (
            "x",
            np.array([[0.225, 0.02, 0.325], [-0.225, -0.01, 0.325]]),
            np.eye(3),
        ),
        (
            "y",
            np.array([[0.03, 0.16, 0.325], [-0.02, -0.16, 0.325]]),
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
    ],
)
def test_coordinated_target_uses_bottom_origin_rotated_box_and_selected_pair(rim_pair, local_points, expected_frame):
    controller = BoxRimExecutor.__new__(BoxRimExecutor)
    controller.box = _box(yaw=0.63)
    controller.target_key = "box"
    plan = SimpleNamespace(
        targets_world_m=controller.box.pose.transform_points(local_points),
        selection={"rim_pair": rim_pair},
    )

    target = controller._coordinated_target(plan)

    np.testing.assert_allclose(target.grasp_points_local, local_points, atol=1e-12)
    np.testing.assert_allclose(target.capture_rotation_local, expected_frame, atol=1e-12)
    assert target.object_key == "box"
    assert target.information_source == "simulation_truth"
    assert target.preserve_point_order


def test_coordinated_target_rejects_forbidden_intervals_instead_of_crossing_them():
    from locomani.tasks.crate.box_rim_geometry import ForbiddenRimInterval

    controller = BoxRimExecutor.__new__(BoxRimExecutor)
    controller.box = _box(forbidden_intervals=(ForbiddenRimInterval("x+", -0.02, 0.02),))
    controller.target_key = "box"
    plan = SimpleNamespace(
        targets_world_m=controller.box.pose.transform_points([[0.225, 0, 0.325], [-0.225, 0, 0.325]]),
        selection={"rim_pair": "x"},
    )

    with pytest.raises(RuntimeError, match="coordinated_forbidden_rim_intervals_unsupported"):
        controller._coordinated_target(plan)


def test_coordinated_runtime_enters_pose_with_open_hands(monkeypatch):
    started_hands = []

    class FakeArm:
        def __init__(self, *_args, **_kwargs):
            self.q = np.arange(29, dtype=float)
            self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.hands = {}

        def start(self, **_kwargs):
            started_hands.append({side: value.copy() for side, value in self.hands.items()})

    runtime = execution.SonicBoxRimRuntime.__new__(execution.SonicBoxRimRuntime)
    runtime.approach_mode = "coordinated"
    runtime.ready_hands = HAND_OPEN.copy()
    runtime.pose_arm = None
    runtime.p = object()
    runtime.ik = object()
    runtime.stop_walk = lambda: None
    monkeypatch.setattr(execution, "PosePreview", lambda **_kwargs: object())
    monkeypatch.setattr(execution, "StreamedArm", FakeArm)

    reference = runtime.enter_pose(2.0, (np.full(29, -1.0), np.full(29, 1.0)))

    np.testing.assert_array_equal(started_hands[0]["left"], HAND_OPEN)
    np.testing.assert_array_equal(started_hands[0]["right"], HAND_OPEN)
    np.testing.assert_array_equal(reference[0], np.arange(29, dtype=float))
    np.testing.assert_array_equal(reference[1], [1.0, 0.0, 0.0, 0.0])


class _Runtime:
    def __init__(self, state):
        self.state = state
        self.commands = []

    def enter_pose(self, *_args):
        return np.full(29, 0.4), np.array([0.98, 0.0, 0.2, 0.0])

    def pose_step(self, joints, root, hands, *, reference_velocity=None):
        self.commands.append(
            (
                np.asarray(joints).copy(),
                np.asarray(root).copy(),
                np.asarray(hands).copy(),
                np.asarray(reference_velocity).copy(),
            )
        )

    def observe(self):
        self.state["seq"] += 1
        self.state["t"] += 0.1
        self.state["wall"] = time.time()
        if self.commands:
            self.state["joints"] = self.commands[-1][0].copy()
            self.state["pelvis"]["quat"] = self.commands[-1][1].copy()
        if len(self.commands) >= 1:
            self.state["objects"]["box"]["touching"] = ["left", "right"]
        return self.state


class _FakeCoordinatedReach:
    instances = []

    def __init__(self, _solver, _config, _state, *, reference_joints, reference_quaternion, target):
        self.index = len(self.instances)
        self.instances.append(self)
        self.step_index = 0
        self.x = np.r_[np.array([0.0, 0.0, 0.8]), 0.0, np.asarray(reference_joints)]
        self.diagnostics = {
            "observed_grips": np.array([[0.3, 0.1, 0.72], [0.3, -0.1, 0.72]]),
            "admission_hand_error": [0.0, 0.0],
            "admission_foot_error": [0.0, 0.0],
            "admission_com_shift": [0.0, 0.0],
        }

    def step(self, *_args, **_kwargs):
        if self.index == 0:
            return LowReachCommand(np.zeros(29), np.zeros(29), np.array([1.0, 0.0, 0.0, 0.0]), 0.0, "reach")
        phases = [
            ("reach", 0.0, False, "", False),
            ("close", 0.5, True, "", False),
            ("lift", 1.0, False, "", False),
            ("hold", 1.0, False, "completed", True),
        ]
        phase, closure, ready, reason, finished = phases[self.step_index]
        self.step_index += 1
        q = np.full(29, 1.0 + self.step_index)
        root = np.array([1.0, 0.01 * self.step_index, 0.0, 0.0])
        self.x[:3] = [0.01 * self.step_index, 0.0, 0.8]
        self.diagnostics = {
            "observed_grips": np.array([[0.3, 0.1, 0.72], [0.3, -0.1, 0.72]]),
            "measured_tools": np.array([[0.3, 0.1, 0.72], [0.3, -0.1, 0.72]]),
            "capture_ready": ready,
            "capture_error": np.zeros((2, 3)),
            "foot_error": np.array([0.007, 0.009]),
        }
        if phase in ("lift", "hold"):
            self.diagnostics["attachment_error"] = 0.0
        return LowReachCommand(q, np.full(29, 0.2), root, closure, phase, reason, finished)


def test_coordinated_execution_forwards_each_reference_pair_and_waits_for_capture(monkeypatch):
    state = {
        "seq": 0,
        "t": 1.0,
        "wall": time.time(),
        "joints": np.zeros(29),
        "pelvis": {
            "pos": [0.0, 0.0, 0.8],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "yaw": 0.0,
            "vel": [0.0, 0.0, 0.0],
        },
        "palms": {
            "left": {
                "pos": [0.3, 0.1, 0.72],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "hand_q": [0.0245, 0.0245],
            },
            "right": {
                "pos": [0.3, -0.1, 0.72],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "hand_q": [0.0245, 0.0245],
            },
        },
        "objects": {
            "box": {
                "pos": [0.3, 0.0, 0.4],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "touching": [],
                "touch_links": [],
            }
        },
        "fallen": False,
        "furniture_touch": [],
    }
    runtime = _Runtime(state)
    controller = BoxRimExecutor.__new__(BoxRimExecutor)
    controller.runtime = runtime
    controller.target_key = "box"
    controller.box = _box()
    controller.config = BoxRimExecutionConfig(
        approach_mode="coordinated", contact_hold_s=0.05, observed_foot_anchor_tolerance_m=0.005
    )
    controller.journal = execution._Journal()
    controller.planner = SimpleNamespace(
        bimanual=SimpleNamespace(lo=np.full(29, -3.0), hi=np.full(29, 3.0)),
    )
    approach = SimpleNamespace(retry_closer=lambda: False)
    plan = SimpleNamespace(
        static_valid=True,
        reason="",
        selection={"rim_pair": "x"},
        to_dict=lambda: {"static_valid": True},
    )
    target = CoordinatedTarget(
        object_key="box",
        grasp_points_local=((0.225, 0.0, 0.325), (-0.225, 0.0, 0.325)),
        information_source="simulation_truth",
        preserve_point_order=True,
    )
    controller._coordinated_stance = lambda current, _approach=None: (current, approach)
    controller._coordinated_control_config = lambda: SimpleNamespace()
    controller._refresh_box_pose = lambda _state: None
    controller._plan_with_keepalive = lambda current: (plan, current)
    controller._coordinated_target = lambda _plan: target
    controller._coordinated_estimate = lambda *_args: object()
    controller._observe = lambda **_kwargs: runtime.observe()
    foot_observation_count = 0

    def observed_feet(_state):
        nonlocal foot_observation_count
        foot_observation_count += 1
        return np.zeros((2, 3)) if foot_observation_count == 1 else np.full((2, 3), 0.01)

    support_contact = {
        "classification": "diagnostic_floor_support_contact",
        "geom_1": "floor",
        "geom_2": "left_foot",
        "body_1": "world",
        "body_2": "left_ankle_roll_link",
        "distance_m": -0.0025931,
        "baseline_distance_m": -0.0007671,
        "support_penetration_growth_m": 0.001826,
    }
    controller._observed_foot_positions = observed_feet
    controller._collision_reasons = lambda *_args, **_kwargs: ([support_contact], [])
    controller._tool_geometry = lambda *_args: (np.zeros((2, 3)), np.repeat(np.eye(3)[None], 2, axis=0))
    controller._observed_gripper_q = lambda _state: np.zeros(4)
    monkeypatch.setattr(execution, "CoordinatedReach", _FakeCoordinatedReach)
    _FakeCoordinatedReach.instances = []

    controller._execute_coordinated(state)

    assert len(runtime.commands) == 4
    np.testing.assert_array_equal(runtime.commands[0][2], HAND_OPEN)
    assert np.any(runtime.commands[1][2] != HAND_OPEN)
    for index, (joints, root, _hands, _velocity) in enumerate(runtime.commands, start=1):
        np.testing.assert_array_equal(joints, np.full(29, 1.0 + index))
        np.testing.assert_array_equal(root, [1.0, 0.01 * index, 0.0, 0.0])
    traces = controller.journal.traces
    assert traces[0]["capture_ready"] is False
    assert traces[1]["capture_ready"] is True
    assert traces[0]["reference_hands"] == HAND_OPEN.tolist()
    assert max(traces[0]["observed_foot_error_m"]) > 0.005
    assert max(traces[0]["reference_foot_error_m"]) > 0.005
    assert traces[0]["measured_allowed_contacts"] == [support_contact]
    assert traces[0]["reference_allowed_contacts"] == [support_contact]
    assert traces[-1]["phase"] == "hold"


def test_execution_config_accepts_only_explicit_coordinated_opt_in():
    assert BoxRimExecutionConfig(approach_mode="coordinated").approach_mode == "coordinated"
    assert BoxRimExecutionConfig().approach_mode == "supported_pose"


def test_coordinated_visual_approach_accepts_the_approved_near_stance_without_retreat():
    executor = BoxRimExecutor.__new__(BoxRimExecutor)
    executor.config = BoxRimExecutionConfig(approach_mode="coordinated")
    executor.planner = SimpleNamespace(
        options=SimpleNamespace(waist_dofs="all", waist_pitch_max_rad=0.52),
        bimanual=SimpleNamespace(lo=np.full(29, -1.0), hi=np.full(29, 1.0)),
    )
    config = executor._coordinated_control_config().visual
    approach = VisualApproach(config)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    last = None
    for now in (0.0, 0.15, 0.31):
        last = approach.step(
            now,
            ApproachTarget(np.array([config.stand_off, 0.0, 0.0]), now, "simulation_truth"),
            quaternion,
            grasp_pose_available=True,
            motion_time=now,
            base_velocity=np.zeros(3),
        )
        assert last.speed == 0.0
        np.testing.assert_array_equal(last.direction, np.zeros(3))
    assert last.ready


def test_coordinated_failure_result_with_controller_arrays_is_json_serializable():
    controller = BoxRimExecutor.__new__(BoxRimExecutor)
    controller.config = BoxRimExecutionConfig(approach_mode="coordinated")
    controller.journal = execution._Journal()
    controller.runtime = SimpleNamespace(safe_stop=lambda: None)
    controller._settle = lambda _phase: {}

    def fail(_state):
        raise execution._Failure(
            "controller_failed",
            controller_diagnostics={"capture_error": np.zeros((2, 3)), "ready": np.bool_(False)},
        )

    controller._execute_coordinated = fail

    result = controller.execute()

    assert not result.ok
    assert result.why == "controller_failed"
    json.dumps(result.execution)


@pytest.mark.parametrize("waist_dofs", ["pitch", "all"])
def test_real_coordinated_reach_consumes_planner_waist_mode_with_strict_bounds(waist_dofs):
    ik = scene_ik()
    solver = BimanualIK(ik, waist_pitch_max=0.52, waist_dofs=waist_dofs)
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    state = {"joints": q, "pelvis": {"pos": [0, 0, 0.78], "quat": [1, 0, 0, 0]}}
    executor = BoxRimExecutor.__new__(BoxRimExecutor)
    executor.config = BoxRimExecutionConfig(
        approach_mode="coordinated", maximum_joint_speed_rps=0.7, maximum_joint_acceleration_rps2=2.0
    )
    executor.planner = SimpleNamespace(
        options=SimpleNamespace(waist_dofs=waist_dofs, waist_pitch_max_rad=0.52),
        bimanual=solver,
    )
    control = executor._coordinated_control_config()
    policy = CoordinatedReach(
        solver,
        control,
        state,
        reference_joints=q,
        reference_quaternion=state["pelvis"]["quat"],
    )
    estimate = CrateEstimate(
        np.array([0.36, 0.02, 0.385]),
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        0.0,
        None,
        {},
    )

    command = policy.step(0.0, estimate, state, [0.08, 0.08])

    assert control.carry.waist_dofs == waist_dofs
    assert np.all(policy.body.lo < policy.body.hi)
    assert command.reason in ("", "unsupported_assisted_reach")
