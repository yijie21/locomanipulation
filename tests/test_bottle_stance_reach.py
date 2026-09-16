import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.backends.sonic import agent_server
from locomani.backends.sonic.agent_server import AgentCommander, StreamMode
from locomani.backends.sonic.primitives import STANDING_PELVIS_Z, upper_body_from_mj
from locomani.tasks.bottle.ready_grasp_policy import ready_q
from locomani.tasks.bottle.stance_reach import (
    STANDING_SUPPORT_SOURCE,
    AsyncStanceReachPlanner,
    StanceReachEvaluator,
    cap_clearing_target_contact_allowed,
    shifted_stance_state,
    standing_support_state,
)
from locomani.tasks.bottle.walking_reach import WalkingReach
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig
from locomani.tasks.bottle.whole_body_policy import (
    WholeBodyGraspPolicy,
    bounded_takeover_brake_step,
)


def state(*, yaw=0.0):
    quat = Rotation.from_euler("xyz", [0.2, -0.1, yaw]).as_quat()[[3, 0, 1, 2]]
    return {
        "t": 1.0,
        "joints": ready_q() + np.linspace(-0.1, 0.1, 29),
        "pelvis": {"pos": np.array([1.0, 2.0, 0.74]), "quat": quat, "vel": np.zeros(3)},
    }


def candidate(yaw=0.4, score=0.0):
    q = ready_q()
    q[12:] += np.linspace(-0.05, 0.05, 17)
    quat = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
    return {"q": q, "candidate_base": np.r_[[1.2, 2.1, 0.79], quat], "score": score}


def result(score=0.0, feasible=True, reason=None):
    return {
        "feasible": feasible,
        "reason": reason,
        "selection_score": [score, score, score, score, score, score, score],
    }


def wait_ready(planner, kind, key, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = planner.poll(kind, key)
        if response["status"] == "ready":
            return response
        time.sleep(0.005)
    raise AssertionError(f"{kind} plan did not finish")


def test_standing_prior_uses_neutral_double_support_and_yaw_only_candidate_root():
    original = state(yaw=-0.7)
    proposal = candidate(yaw=0.4)
    standing = standing_support_state(original, proposal["q"], proposal["candidate_base"])

    assert np.allclose(standing["joints"][:12], ready_q()[:12])
    assert np.allclose(standing["joints"][12:], proposal["q"][12:])
    assert standing["pelvis"]["pos"] == pytest.approx([1.2, 2.1, STANDING_PELVIS_Z])
    rotation = Rotation.from_quat(np.asarray(standing["pelvis"]["quat"])[[1, 2, 3, 0]])
    assert rotation.as_euler("xyz") == pytest.approx([0.0, 0.0, 0.4], abs=1e-9)
    assert np.allclose(original["joints"], state(yaw=-0.7)["joints"])


def test_shifted_prior_covers_joint_forward_and_positive_or_negative_yaw_uncertainty():
    standing = standing_support_state(state(), candidate()["q"], candidate()["candidate_base"])
    for yaw in (np.radians(16), -np.radians(16)):
        shifted, rotation = shifted_stance_state(standing, 0.18, yaw)
        expected = np.asarray(standing["pelvis"]["pos"]) + 0.18 * rotation[:, 0]
        assert shifted["pelvis"]["pos"] == pytest.approx(expected)
        assert np.linalg.norm(expected[:2] - np.asarray(standing["pelvis"]["pos"])[:2]) == pytest.approx(0.18)


def test_walking_batch_requires_every_uncertainty_scenario_and_scores_natural_margin():
    evaluator = StanceReachEvaluator.__new__(StanceReachEvaluator)
    evaluator.config = WholeBodyConfig(natural_walk_prep=True)
    evaluator.feasibility = None
    calls = []

    def evaluate(
        support,
        target,
        rotation,
        posture,
        *,
        allow_target_contact,
        verify_path=True,
        cancel_event=None,
    ):
        calls.append((deepcopy(support), rotation.copy(), posture.copy(), verify_path))
        candidate_number = 0 if posture[22] < ready_q()[22] + 0.01 else 1
        yaw = Rotation.from_matrix(rotation).as_euler("xyz")[2]
        failed_uncertainty = candidate_number == 0 and yaw > 0.45
        answer = result(
            -0.2 if candidate_number == 0 else 0.3,
            feasible=not failed_uncertainty,
            reason="support_com_shift" if failed_uncertainty else None,
        )
        answer["reach_path"] = {"complete": verify_path}
        answer["feasibility_scope"] = "cartesian_path" if verify_path else "endpoint_only"
        return answer

    evaluator.evaluate = evaluate
    first = candidate(score=-1.0)
    first["q"][22] = ready_q()[22]
    second = candidate(score=2.0)
    second["q"][22] = ready_q()[22] + 0.02
    batch = evaluator.evaluate_walking(
        {"state": state(), "target": np.array([1.5, 1.8, 0.7]), "candidates": [first, second]}
    )

    assert batch["support_prior_source"] == STANDING_SUPPORT_SOURCE
    assert set(batch["checked"][0]["scenarios"]) == {
        "expected",
        "forward_yaw_left",
        "forward_yaw_right",
    }
    assert batch["checked"][0]["reason"] == "support_com_shift"
    assert batch["chosen_index"] == 1
    assert batch["path_verification"] == "chosen_expected_cartesian_path; uncertainty_endpoints_only"
    assert batch["checked"][1]["scenarios"]["expected"]["reach_path"]["complete"]
    assert not batch["checked"][1]["scenarios"]["forward_yaw_left"]["reach_path"]["complete"]
    assert len(calls) == 7

    calls.clear()
    endpoint_only = evaluator.evaluate_walking(
        {
            "state": state(),
            "target": np.array([1.5, 1.8, 0.7]),
            "candidates": [first, second],
            "verify_path": False,
        }
    )
    assert endpoint_only["feasible"]
    assert endpoint_only["path_verification"] == "endpoint_only; live_per_tick_cartesian_guard"
    assert calls and all(not verify_path for *_inputs, verify_path in calls)


def test_cartesian_rollout_checks_cancellation_between_samples():
    class CancelAfterOneSample:
        def __init__(self):
            self.checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks >= 2

    measured = state(yaw=0.0)
    evaluator = StanceReachEvaluator(scene_ik(), WholeBodyConfig(natural_walk_prep=True))
    path, diagnostics = evaluator._cartesian_reach_path(
        measured,
        np.array([1.35, 1.8, 0.70]),
        np.eye(3),
        ready_q(),
        np.eye(3),
        cancel_event=CancelAfterOneSample(),
    )

    assert len(path) == 1
    assert diagnostics["source"] == "supported_cap_clearing_cartesian_reference"
    assert not diagnostics["complete"]
    assert diagnostics["cancelled"]
    assert diagnostics["samples"] == 1
    assert diagnostics["ik_step_seconds"] > 0


def test_settled_live_refinement_can_defer_nominal_rollout():
    evaluator = StanceReachEvaluator.__new__(StanceReachEvaluator)
    received = {}

    def evaluate(*args, **kwargs):
        received.update(kwargs)
        return {"feasible": True, "solve_seconds": 0.1}

    evaluator.evaluate = evaluate
    answer = evaluator.evaluate_settled(
        {
            "state": state(),
            "target": np.array([1.4, 1.8, 0.7]),
            "desired_rotation": np.eye(3),
            "posture": ready_q(),
            "verify_path": False,
        }
    )

    assert answer["feasible"]
    assert received["verify_path"] is False
    assert answer["support_prior_source"] == "measured_post_stop_state"


def test_stance_path_uses_live_height_gate_for_target_contact():
    target = np.array([1.2, -0.2, 0.70])
    assert not cap_clearing_target_contact_allowed([1.1, -0.2, 0.736], target)
    assert cap_clearing_target_contact_allowed([1.1, -0.2, 0.734], target)

    contacts = []

    class ClearResult:
        clear = True

        @staticmethod
        def to_dict():
            return {"clear": True, "link": None, "obstacle": None}

    class Guard:
        margin = 0.004

        def check(self, *_args, allow_target_contact, **_kwargs):
            contacts.append(allow_target_contact)
            return ClearResult()

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.stance_guard = Guard()
    measured = state()
    base = np.r_[measured["pelvis"]["pos"], measured["pelvis"]["quat"]]
    path = [
        {"joints": measured["joints"], "candidate_base": base, "allow_target_contact": False},
        {"joints": measured["joints"], "candidate_base": base, "allow_target_contact": True},
    ]

    clearance = policy._stance_feasibility(
        measured,
        measured["joints"],
        base,
        allow_target_contact=True,
        path=path,
    )

    assert clearance["clear"]
    assert contacts == [False, False, False, True, True, True]


def test_async_poll_never_submits_and_slow_solve_does_not_block_control_tick():
    class SlowEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate_walking(self, payload):
            self.calls.append(("walking", payload["value"]))
            time.sleep(0.10)
            return {"feasible": True, "chosen_index": 0}

        def evaluate_settled(self, payload):
            self.calls.append(("settled", payload["value"]))
            return {"feasible": True}

    evaluator = SlowEvaluator()
    planner = AsyncStanceReachPlanner(evaluator)
    try:
        started = time.perf_counter()
        assert planner.poll("walking", 1)["status"] == "missing"
        assert time.perf_counter() - started < 0.05
        assert evaluator.calls == []

        started = time.perf_counter()
        assert planner.request("walking", 1, {"value": "first"})["status"] == "pending"
        assert planner.request("walking", 1, {"value": "replacement"})["status"] == "pending"
        assert time.perf_counter() - started < 0.05
        response = wait_ready(planner, "walking", 1)
        assert response["result"]["feasible"]
        assert evaluator.calls == [("walking", "first")]
    finally:
        planner.close()


def test_settled_priority_discards_running_walking_result_before_it_can_publish():
    walking_started = threading.Event()

    class SlowEvaluator:
        def evaluate_walking(self, payload):
            walking_started.set()
            time.sleep(0.08)
            return {"feasible": True, "chosen_index": 0}

        def evaluate_settled(self, payload):
            return {"feasible": True, "marker": payload["marker"]}

    planner = AsyncStanceReachPlanner(SlowEvaluator())
    try:
        planner.request("walking", 4, {"unused": True})
        assert walking_started.wait(0.2)
        assert planner.request("settled", 7, {"marker": "actual"}, priority=True)["status"] == "pending"
        response = wait_ready(planner, "settled", 7)
        assert response["result"]["marker"] == "actual"
        assert planner.poll("walking", 4)["status"] == "missing"
    finally:
        planner.close()


def test_settled_priority_cooperatively_stops_walking_between_scenario_solves():
    first_scenario = threading.Event()

    class CooperativeEvaluator(StanceReachEvaluator):
        def __init__(self):
            self.config = WholeBodyConfig(natural_walk_prep=True)
            self.feasibility = None
            self.scenario_calls = 0

        def evaluate(self, *args, **kwargs):
            self.scenario_calls += 1
            first_scenario.set()
            time.sleep(0.06)
            return result(0.0)

        def evaluate_settled(self, payload):
            return {"feasible": True, "marker": payload["marker"]}

    evaluator = CooperativeEvaluator()
    planner = AsyncStanceReachPlanner(evaluator)
    try:
        planner.request(
            "walking",
            1,
            {"state": state(), "target": [1.4, 1.8, 0.7], "candidates": [candidate()]},
        )
        assert first_scenario.wait(0.2)
        planner.request("settled", 2, {"marker": "priority"}, priority=True)
        response = wait_ready(planner, "settled", 2)
        assert response["result"]["marker"] == "priority"
        assert evaluator.scenario_calls == 1
    finally:
        planner.close()


def test_settled_validity_uses_target_root_endpoint_and_joint_state():
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = WholeBodyConfig(natural_walk_prep=True)
    measured = state(yaw=0.2)
    request = {
        "target": np.array([1.4, 1.8, 0.7]),
        "pelvis_position": np.asarray(measured["pelvis"]["pos"]).copy(),
        "root_rotation": Rotation.from_euler("z", 0.2).as_matrix(),
        "joints": np.asarray(measured["joints"]).copy(),
    }
    assert policy._settled_support_matches(request, measured, request["target"])
    moved = deepcopy(measured)
    moved["pelvis"]["pos"] = np.asarray(moved["pelvis"]["pos"]) + [0.02, 0, 0]
    assert not policy._settled_support_matches(request, moved, request["target"])
    assert not policy._settled_support_matches(request, measured, request["target"] + [0, 0.02, 0])
    moved = deepcopy(measured)
    moved["joints"][3] += 0.10
    assert not policy._settled_support_matches(request, moved, request["target"])


def test_settled_invalidation_diagnostics_retain_each_exact_snapshot_delta():
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = WholeBodyConfig(natural_walk_prep=True)
    policy.stats = {}
    requested = state(yaw=0.2)
    requested["seq"] = 42
    requested["t"] = 7.5
    request = {
        "key": 3,
        "target": np.array([1.4, 1.8, 0.7]),
        "pelvis_position": np.asarray(requested["pelvis"]["pos"]).copy(),
        "root_rotation": Rotation.from_euler("z", 0.2).as_matrix(),
        "joints": np.asarray(requested["joints"]).copy(),
        "state": requested,
        "reference_source": "pose_owned_reference",
    }
    moved = state(yaw=0.25)
    moved["t"] = 7.7
    moved["pelvis"]["pos"] = np.asarray(moved["pelvis"]["pos"]) + [0.02, 0, 0]
    moved["joints"][3] += 0.10
    target = request["target"] + [0, 0.02, 0]

    policy._record_settled_invalidation("support_or_target_changed", request, moved, target, "pose_owned_reference")
    moved["t"] = 7.8
    policy._record_settled_invalidation(
        "reference_source_changed", request, moved, target, "walking_owned_upper_reference"
    )

    first, second = policy.stats["settled_stance_invalidations"]
    assert first["request_key"] == 3
    assert first["request_seq"] == 42
    assert first["request_physical_time"] == pytest.approx(7.5)
    assert first["physical_time"] == pytest.approx(7.7)
    assert first["target_error_m"] == pytest.approx(0.02)
    assert first["root_translation_error_m"] == pytest.approx(0.02)
    assert first["root_yaw_error_rad"] == pytest.approx(0.05)
    assert first["max_joint_error_rad"] == pytest.approx(0.10)
    assert first["max_joint_error_index"] == 3
    assert second["physical_time"] == pytest.approx(7.8)
    assert second["reason"] == "reference_source_changed"
    assert second["current_reference_source"] == "walking_owned_upper_reference"


def test_walking_reach_prepares_at_conservative_stop_while_full_body_plan_is_pending():
    class Planner:
        def __init__(self):
            self.requests = []
            self.ready = False

        def request(self, kind, key, payload, *, priority=False):
            self.requests.append((kind, key, deepcopy(payload)))
            return {"status": "pending", "poll_seconds": 0.0001}

        def poll(self, kind, key):
            if not self.ready:
                return {"status": "pending", "poll_seconds": 0.0001, "plan_age_seconds": 0.1}
            return {
                "status": "ready",
                "poll_seconds": 0.0001,
                "result": {
                    "feasible": True,
                    "chosen_index": 0,
                    "checked": [],
                    "support_prior_source": STANDING_SUPPORT_SOURCE,
                    "solve_seconds": 0.4,
                },
            }

    planner = Planner()
    cfg = WholeBodyConfig(
        raise_trigger_px=0,
        natural_walk_prep=True,
        wrist_roi_supervisor=True,
        vision_confirm_frames=2,
    )
    hanging = upper_body_from_mj(ready_q())
    hanging[4::2] = [0.0, -0.27, 0.64, 1.02, -0.07, 0.22, -0.15]
    controller = WalkingReach(
        scene_ik(),
        cfg,
        hanging,
        brake_depth=0.31,
        brake_edge=0.27,
        stance_planner=planner,
    )
    measured = state()
    measured["pelvis"]["pos"] = np.array([0.0, 0.0, STANDING_PELVIS_Z])
    measured["pelvis"]["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
    measured["pelvis"]["vel"] = np.zeros(3)
    incoming = controller.q.copy()
    target = np.array([1.2, -0.2, 0.70])

    def camera(stamp):
        return {
            "image_timestamps": {"ego_view": stamp, "right_wrist": stamp},
            "visible": {"ego_view": stamp, "right_wrist": stamp},
            "wrist_roi": {
                "timestamp": stamp,
                "centroid": [0.5, 0.5],
                "extent": [0.2, 0.3],
                "quality": 0.5,
                "correction_basis_pelvis": np.zeros((3, 2)),
            },
        }

    controller.step(measured, target, camera=camera(1.0), edge_distance=1.1)
    assert controller.q == pytest.approx(incoming)
    assert controller.diagnostics["stance_plan_status"] == "pending"
    assert len(planner.requests) == 1
    assert planner.requests[0][2]["verify_path"] is False

    measured = deepcopy(measured)
    measured["t"] = 1.10
    controller.step(measured, target, camera=camera(1.1), edge_distance=1.1)
    assert np.max(np.abs(controller.q - incoming)) > 0
    assert controller.wrist_acquired
    assert controller.diagnostics["chosen_geometry"]["preparation_feasible"]
    assert controller.diagnostics["chosen_geometry"]["stop_depth"] == pytest.approx(0.31 + cfg.stance_search_span)

    planner.ready = True
    measured = deepcopy(measured)
    measured["t"] = 1.21
    controller.step(measured, target, edge_distance=1.1)
    assert controller.diagnostics["chosen_geometry"]["feasible"]
    assert controller.diagnostics["stance_plan_solve_seconds"] == pytest.approx(0.4)
    assert np.max(np.abs(controller.q - incoming)) < 0.05

    for stamp in (1.42, 1.63, 1.84):
        measured = deepcopy(measured)
        measured["t"] = stamp
        controller.step(measured, target, edge_distance=1.1)
    assert len(planner.requests) == 1


def test_settled_path_request_waits_for_measured_local_preparation_and_records_exact_input():
    class Planner:
        def __init__(self):
            self.requests = []

        def request(self, kind, key, payload, *, priority=False):
            self.requests.append((kind, key, deepcopy(payload), priority))
            return {"status": "pending", "poll_seconds": 0.0001}

        def poll(self, kind, key):
            return {"status": "pending", "poll_seconds": 0.0001}

    measured = state()
    measured["seq"] = 42
    upper = upper_body_from_mj(measured["joints"])
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = WholeBodyConfig(natural_walk_prep=True)
    policy.stance_planner = Planner()
    policy.walking = SimpleNamespace(q=upper.copy(), v=np.full(17, 0.2))
    policy.target = SimpleNamespace(observe=lambda _: np.array([1.2, 1.8, 0.70]))
    policy.stats = {}
    policy._settled_request = None
    policy._settled_plan = None
    policy._settled_request_count = 0

    def sample(stamp, joint_shift, hand_shift):
        current = deepcopy(measured)
        current["t"] = stamp
        current["joints"][12] += joint_shift
        current["palms"] = {
            "right": {
                "pos": np.array([1.0 + hand_shift, 2.0, 0.75]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }
        }
        return current

    preparing = policy._settled_stance_tick(sample(7.25, 0.0, 0.0))
    assert preparing["status"] == "preparing"
    assert preparing["reason"] == "reference_moving"
    assert policy.stance_planner.requests == []

    policy.walking.v[:] = 0
    policy.walking.q[0] += 0.20
    for stamp, joint_shift, hand_shift in (
        (7.30, 0.00, 0.00),
        (7.35, 0.03, 0.01),
        (7.40, 0.06, 0.02),
        (7.45, 0.06, 0.02),
        (7.50, 0.06, 0.02),
    ):
        tracking = policy._settled_stance_tick(sample(stamp, joint_shift, hand_shift))
        assert tracking["status"] == "preparing"
        assert policy.stance_planner.requests == []

    submitted_state = sample(7.55, 0.06, 0.02)
    pending = policy._settled_stance_tick(submitted_state)
    assert pending["status"] == "pending"
    assert len(policy.stance_planner.requests) == 1
    request = policy.stats["settled_stance_plan"]["request"]
    assert request["seq"] == 42 and request["t"] == pytest.approx(7.55)
    np.testing.assert_array_equal(request["state"]["joints"], submitted_state["joints"])
    expected_reference = np.asarray(measured["joints"]).copy()
    expected_reference[12] += 0.20
    np.testing.assert_allclose(request["reference_joints"], expected_reference, atol=1e-7)
    assert request["reference_measurement_offset_linf_rad"] > 0.08
    np.testing.assert_array_equal(request["reference_root"], submitted_state["pelvis"]["quat"])
    np.testing.assert_array_equal(request["target"], [1.2, 1.8, 0.70])
    planner_payload = policy.stance_planner.requests[0][2]
    assert planner_payload["verify_path"] is False
    np.testing.assert_array_equal(request["grasp_rotation"], planner_payload["desired_rotation"])
    np.testing.assert_array_equal(request["reference_joints"], planner_payload["reference_joints"])
    np.testing.assert_array_equal(request["reference_root"], planner_payload["reference_root"])

    retained_request = policy._settled_request
    moving = policy._settled_stance_tick(sample(7.60, 0.08, 0.03), support_quiet=False, allow_submission=False)
    assert moving["reason"] == "support_moving"
    assert policy._settled_request is retained_request

    for stamp, hand_shift in ((7.65, 0.04), (7.70, 0.02), (7.75, 0.02)):
        preparing = policy._settled_stance_tick(sample(stamp, 0.06, hand_shift), allow_submission=False)
        assert preparing["status"] == "preparing"
        assert policy._settled_request is retained_request
    assert len(policy.stance_planner.requests) == 1

    resumed = policy._settled_stance_tick(sample(7.81, 0.06, 0.02))
    assert resumed["status"] == "pending"
    assert policy._settled_request is retained_request
    assert len(policy.stance_planner.requests) == 1


def test_brake_pending_settled_request_times_out_during_transient_support_motion(monkeypatch):
    clock = SimpleNamespace(now=0.0, stamp=0.0)
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = SimpleNamespace(stance_settled_timeout=0.03)
    policy.stats = {}
    policy.error = None
    request = {"submitted_wall": 0.0}
    policy._settled_request = request
    policy._settled_plan = None
    policy._settled_wait_started = 0.0

    def obs():
        clock.stamp += 0.01
        return {
            "t": clock.stamp,
            "fallen": False,
            "pelvis": {"vel": np.array([0.10, 0.0, 0.0])},
        }

    policy.p = SimpleNamespace(obs=obs)
    policy._hold_planner = lambda _state: None
    policy._observe_approach_settlement = lambda *_args, **_kwargs: None
    policy._preparation_hold_tick = lambda _state: True
    policy._settled_stance_tick = lambda *_args, **_kwargs: {
        "status": "preparing",
        "reason": "support_moving",
    }

    monkeypatch.setattr("locomani.tasks.bottle.whole_body_policy.time.monotonic", lambda: clock.now)
    monkeypatch.setattr(
        "locomani.tasks.bottle.whole_body_policy.time.sleep",
        lambda duration: setattr(clock, "now", clock.now + duration),
    )

    assert not policy._brake()
    assert policy.error == "whole-body settled stance planning timed out"
    assert policy._settled_request is request
    assert clock.now > policy.cfg.stance_settled_timeout


def test_live_reach_uses_grasp_rotation_from_verified_settled_plan():
    settled_rotation = Rotation.from_euler("z", 0.25).as_matrix()
    stale_walking_rotation = Rotation.from_euler("z", -0.40).as_matrix()
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy._settled_plan = {"grasp_rotation": settled_rotation}
    policy.walking = SimpleNamespace(rotation=stale_walking_rotation)

    chosen, source = policy._verified_grasp_rotation(state(yaw=0.7))

    np.testing.assert_allclose(chosen, settled_rotation)
    assert source == "settled_stance_plan"


def test_live_reach_preserves_rollout_control_goals_separately_from_admitted_endpoint():
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    admitted_joints = np.zeros(29)
    control_joints = np.linspace(-0.2, 0.2, 29)
    admitted_torso = Rotation.from_euler("y", 0.15).as_matrix()
    control_torso = Rotation.from_euler("y", 0.25).as_matrix()
    policy._settled_plan = {
        "joints": admitted_joints,
        "torso_rotation": admitted_torso,
        "control_posture_joints": control_joints,
        "control_torso_rotation": control_torso,
    }

    posture, torso = policy._settled_control_targets()

    np.testing.assert_array_equal(posture, control_joints)
    np.testing.assert_array_equal(torso, control_torso)
    assert not np.shares_memory(posture, control_joints)
    assert not np.shares_memory(torso, control_torso)


def test_pose_hold_heartbeat_overrides_stale_commander_stream_mode(monkeypatch):
    manager_states = []
    planner_frames = []
    commander = AgentCommander.__new__(AgentCommander)
    commander.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    commander.heading_offset = 0.0
    commander.upper_body = np.zeros(17)
    commander.upper_body_vel = np.zeros(17)
    commander.vr_targets = False
    commander.left_hand = np.zeros(7)
    commander.right_hand = np.zeros(7)
    commander.reference_observer = None
    commander.manager_state = manager_states.append
    commander.command = lambda **_kwargs: None
    commander.sonic = SimpleNamespace(send=planner_frames.append)
    monkeypatch.setattr(agent_server, "build_planner_message", lambda **data: data)

    commander.enter_streamed()
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.p = SimpleNamespace(cmd=commander, _facing=lambda _state: (1.0, 0.0, 0.0))
    policy._hold_planner({"pelvis": {}}, stream_mode=StreamMode.POSE)

    assert commander.stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
    assert manager_states == [StreamMode.POSE, StreamMode.POSE]
    assert "upper_body_position" not in planner_frames[-1]


def test_post_takeover_settled_timeout_starts_at_first_submission(monkeypatch):
    clock = SimpleNamespace(now=0.0, plan_delay_applied=False)
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.cfg = SimpleNamespace(stance_settled_timeout=0.30)
    policy.stats = {}
    policy.stance_planner = object()
    policy._settled_request = None
    policy._settled_plan = None
    physical = SimpleNamespace(stamp=0.0)

    def obs():
        physical.stamp += 0.10
        return {
            "t": physical.stamp,
            "fallen": False,
            "pelvis": {"vel": np.zeros(3)},
        }

    def sleep(_seconds):
        if policy._settled_request is not None and not clock.plan_delay_applied:
            clock.now += 0.20
            clock.plan_delay_applied = True
        else:
            clock.now += 0.05

    policy.p = SimpleNamespace(obs=obs)
    policy._admit_takeover_stop = lambda _state: ({"admitted": True}, None)
    heartbeat_modes = []
    policy._hold_planner = lambda _state, *, stream_mode=None: heartbeat_modes.append(stream_mode)
    policy.arm = SimpleNamespace(
        q=ready_q(),
        root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        reference_velocity=np.zeros(29),
        root_angular_velocity=np.zeros(3),
        preview=SimpleNamespace(
            acceleration=2.0,
            lo=np.full(29, -10.0),
            hi=np.full(29, 10.0),
        ),
        _push=lambda *_args, **_kwargs: None,
    )
    planner_polls = 0

    def settled_tick(_state):
        nonlocal planner_polls
        planner_polls += 1
        if policy._settled_request is None:
            policy._settled_request = {"submitted_wall": clock.now}
            return {"status": "pending"}
        policy._settled_plan = {"feasible": True}
        return {
            "status": "ready",
            "result": {"batch_solve_seconds": 0.15},
        }

    policy._settled_stance_tick = settled_tick
    monkeypatch.setattr("locomani.tasks.bottle.whole_body_policy.time.monotonic", lambda: clock.now)
    monkeypatch.setattr("locomani.tasks.bottle.whole_body_policy.time.sleep", sleep)

    settled_state, error = policy._post_takeover_stance()

    assert error is None
    assert settled_state is not None
    assert planner_polls == 2
    assert heartbeat_modes and set(heartbeat_modes) == {StreamMode.POSE}
    assert policy.stats["post_takeover_stance"]["status"] == "ready"
    assert policy.stats["post_takeover_stance"]["planner_elapsed_seconds"] == pytest.approx(0.20)
    assert policy.stats["post_takeover_stance"]["planner_worker_seconds"] == pytest.approx(0.15)
    assert policy.stats["post_takeover_stance"]["total_wall_seconds"] > 0.30


def test_nonzero_joint_and_root_angular_takeover_velocity_brakes_continuously():
    q = ready_q()
    root = Rotation.from_euler("z", 0.2).as_quat()[[3, 0, 1, 2]]
    velocity = np.zeros(29)
    velocity[14] = 0.6
    angular = np.array([0.0, 0.0, 0.4])
    following, next_root, next_velocity, next_angular, stopped = bounded_takeover_brake_step(
        q,
        root,
        velocity,
        angular,
        0.02,
        2.0,
        (np.full(29, -10.0), np.full(29, 10.0)),
    )

    assert not stopped
    assert next_velocity[14] == pytest.approx(0.56)
    assert next_angular[2] == pytest.approx(0.4 * 0.56 / 0.6)
    assert following[14] - q[14] == pytest.approx(0.5 * (0.6 + 0.56) * 0.02)
    yaw_change = Rotation.from_quat(next_root[[1, 2, 3, 0]]) * Rotation.from_quat(root[[1, 2, 3, 0]]).inv()
    assert yaw_change.as_rotvec()[2] > 0
