"""visual.reach_route planned (user, 2026-10-14): the coordinated reach follows a planned joint path from where the hands
are to its own pregrasp solve, and `ReachRoutePlanner` maps that request onto the CuRobo route service's 17 joints.
The service itself (CuRobo, SIMPLE's venv) is exercised live, not here."""

import numpy as np
import pytest
from test_crate_coordinated_reach import estimate, setup

from locomani.primitives.arm_route import Route
from locomani.tasks.crate import reach_route
from locomani.tasks.crate.reach_route import PLANNED, PLANNED_INDEX, ReachRoutePlanner
from locomani.tasks.crate.visual_config import CrateVisualConfig


def test_reach_route_config():
    assert CrateVisualConfig().reach_route == "straight"
    assert CrateVisualConfig(reach_route="planned", reach_path="staged").reach_route == "planned"
    with pytest.raises(ValueError, match="needs reach_path staged"):
        CrateVisualConfig(reach_route="planned", reach_path="direct")
    with pytest.raises(ValueError, match="straight or planned"):
        CrateVisualConfig(reach_route="curobo")
    with pytest.raises(ValueError, match="reach_route_speed"):
        CrateVisualConfig(reach_route_speed=3.0)
    with pytest.raises(ValueError, match="reach_route_clearance"):
        CrateVisualConfig(reach_route_clearance=-0.01)


class FakeClient:
    def __init__(self, ok=True, waypoints=5, statuses=()):
        self.ok, self.waypoints, self.configured, self.requests = ok, waypoints, [], []
        self.statuses = list(statuses)  # per request: (ok, status) before falling back to self.ok

    def configure(self, names, lower, upper, retract):
        self.configured.append((list(names), np.array(lower), np.array(upper), np.array(retract)))

    def plan_to_joints(self, start, goal, obstacles, check_start_validity=True):
        self.requests.append((np.array(start), np.array(goal), obstacles, check_start_validity))
        ok, status = self.statuses.pop(0) if self.statuses else (self.ok, "")
        path = np.linspace(start, goal, self.waypoints) if ok else np.zeros((0, len(start)))
        return Route(ok=ok, index=0 if ok else None, path=path, success=(ok,), seconds=0.1, status=status)


def test_planner_plans_waist_and_arms_and_keeps_the_legs(monkeypatch):
    monkeypatch.setattr(reach_route, "scene_obstacles", lambda *a, **k: ["box"])
    lower, upper = -np.ones(29), np.ones(29)
    client = FakeClient()
    planner = ReachRoutePlanner(client, None, lower, upper, {"box_body": "box"})
    start, goal = np.full(29, 0.2), np.full(29, 0.5)
    start[PLANNED_INDEX[0]] = 1.3  # a measured joint a hair past its limit is clipped, not refused
    state = dict(
        pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]), objects={"box": dict(pos=[1, 0, 0], quat=[1, 0, 0, 0])}
    )
    path = planner(start, goal, state)
    planner(start, goal, state)
    assert len(client.configured) == 1 and client.configured[0][0] == PLANNED and len(PLANNED) == 17
    sent_start, sent_goal, obstacles, check = client.requests[0]
    assert sent_start[0] == 1.0 and np.allclose(sent_goal, 0.5) and obstacles == ["box"] and check
    legs = np.setdiff1d(np.arange(29), PLANNED_INDEX)
    assert path.shape == (5, 29) and np.allclose(path[:, legs], 0.2)  # legs are not planned
    assert np.allclose(path[-1, PLANNED_INDEX], 0.5)
    assert planner.last["ok"] and planner.last["waypoints"] == 5
    assert ReachRoutePlanner(FakeClient(ok=False), None, lower, upper, {})(start, goal, state) is None


def test_planner_grows_boxes_by_the_clearance(monkeypatch):
    from locomani.primitives.arm_route import Box, MeshObstacle

    mesh = MeshObstacle("m", (0, 0, 0, 1, 0, 0, 0), np.zeros((3, 3)), np.array([[0, 1, 2]]))
    monkeypatch.setattr(
        reach_route, "scene_obstacles", lambda *a, **k: [Box("wall", (0, 0, 0, 1, 0, 0, 0), (0.01, 0.3, 0.3)), mesh]
    )
    planner = ReachRoutePlanner(FakeClient(), None, -np.ones(29), np.ones(29), {}, clearance=0.015)
    state = dict(pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]), objects={})
    wall, same = planner.obstacles(state)
    assert np.allclose(wall.dims, (0.04, 0.33, 0.33)) and same is mesh
    with pytest.raises(ValueError, match="clearance"):
        ReachRoutePlanner(FakeClient(), None, -np.ones(29), np.ones(29), {}, clearance=0.2)


def test_a_start_inside_the_clearance_is_planned_again_without_the_start_check(monkeypatch):
    monkeypatch.setattr(reach_route, "scene_obstacles", lambda *a, **k: [])
    client = FakeClient(statuses=[(False, "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"), (True, "")])
    planner = ReachRoutePlanner(client, None, -np.ones(29), np.ones(29), {})
    state = dict(pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]), objects={})
    assert planner(np.zeros(29), np.full(29, 0.1), state) is not None
    assert [r[3] for r in client.requests] == [True, False]
    assert planner.last["start_in_collision"] and planner.last["ok"]
    other = FakeClient(statuses=[(False, "MotionGenStatus.TRAJOPT_FAIL")])  # any other failure is not retried
    planner = ReachRoutePlanner(other, None, -np.ones(29), np.ones(29), {})
    assert planner(np.zeros(29), np.full(29, 0.1), state) is None and len(other.requests) == 1


def planned_reach(planner):
    policy, state = setup()
    policy.config = policy.config.with_overrides(["visual.reach_path=staged", "visual.reach_route=planned"])
    policy.route_planner = planner
    return policy, state


def test_reach_follows_the_planned_route_to_the_pregrasp_solve():
    calls = []

    def planner(start, goal, state):
        calls.append((start.copy(), goal.copy()))
        return np.linspace(start, goal, 30)

    policy, state = planned_reach(planner)
    start_tools = policy.body.tools(policy.x).copy()
    command = policy.step(0, estimate(0, state), state, [0.08] * 2)
    assert not command.finished, command.reason
    assert len(calls) == 1 and policy.route is not None
    route = policy.route
    assert np.isclose(policy.duration, route["total"])
    assert np.allclose(route["positions"][0], start_tools, atol=1e-6)  # no jump at the start
    positions, _, joints, s = policy.route_sample(0.0)
    assert s == 0.0 and np.allclose(joints, calls[0][0])
    positions, attitudes, joints, s = policy.route_sample(route["total"])
    assert s == 1.0 and np.allclose(joints, calls[0][1]) and np.allclose(positions, route["positions"][-1])
    assert np.allclose(attitudes @ attitudes.transpose(0, 2, 1), np.eye(3), atol=1e-6)
    phases = []
    for now in np.arange(0.05, route["total"] + 2.0, 0.05):
        command = policy.step(now, estimate(now, state), state, [0.015] * 2)
        assert not command.finished, command.reason
        phases.append(policy.phase)
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3].copy(), quat=command.root_quaternion))
    # the reach phase lasts the route's own duration; the staged reach goes on as before after it (align or descend,
    # by the profile's descend_align_tolerance)
    after = [p for p in phases if p != "reach"]
    assert phases[0] == "reach" and after and after[0] in ("align", "descend")
    assert abs(0.05 * phases.index(after[0]) - route["total"]) <= 0.1


def test_reach_ends_the_round_when_no_route_exists():
    policy, state = planned_reach(lambda start, goal, state: None)
    command = policy.step(0, estimate(0, state), state, [0.08] * 2)
    assert command.finished and command.reason == "route_not_found"


def test_planned_reach_without_a_planner_is_an_error():
    policy, state = planned_reach(None)
    with pytest.raises(RuntimeError, match="route_planner"):
        policy.step(0, estimate(0, state), state, [0.08] * 2)
