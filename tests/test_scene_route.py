"""TODO 40 (user 2026-10-27): `SceneRoute`, the scene line's CuRobo pregrasp route -- the obstacles it builds from the
live scene and how it maps a request onto the route service's 17 joints. The service itself (CuRobo, SIMPLE's venv)
is exercised live, not here."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from test_crate_reach_route import FakeClient
from test_scalebfm_motor import motor

from locomani.primitives.arm_route import Box, Route
from locomani.tasks.crate.reach_route import PLANNED, PLANNED_INDEX
from locomani.tasks.scene.route import SceneRoute


def _route(client=None, **kw):
    m = motor()
    return m, SceneRoute(m.sim, client or FakeClient(), **kw)


def test_obstacles_are_the_scene_boxes_near_the_pelvis_grown_and_in_its_frame():
    m, route = _route(radius=5.0)
    sim = m.sim
    pos, quat = m.pelvis()
    boxes = {o.name: o for o in route.obstacles(pos, quat, 0.015) if isinstance(o, Box)}
    assert boxes
    frame = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    robot = {sim.m.body("pelvis").id}
    for b in range(sim.m.body("pelvis").id + 1, sim.m.nbody):
        if int(sim.m.body_parentid[b]) in robot:
            robot.add(b)
    for g in range(sim.m.ngeom):
        if sim.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or not (sim.m.geom_contype[g] or sim.m.geom_conaffinity[g]):
            continue
        name = sim.m.geom(g).name or f"geom{g}"
        if int(sim.m.geom_bodyid[g]) in robot:
            assert name not in boxes  # the robot is not its own obstacle
            continue
        box = boxes[name]
        np.testing.assert_allclose(box.dims, 2 * sim.m.geom_size[g] + 0.03)
        np.testing.assert_allclose(box.pose[:3], frame.T @ (sim.d.geom_xpos[g] - pos), atol=1e-9)
    far = SceneRoute(sim, FakeClient(), radius=-10.0).obstacles(pos, quat, 0.015)
    assert far == []  # nothing within a negative radius


def test_a_failed_plan_is_tried_again_with_the_planners_own_margin_and_the_legs_move_linearly():
    client = FakeClient(statuses=[(False, "MotionGenStatus.TRAJOPT_FAIL"), (True, "")])
    m, route = _route(client, radius=5.0)
    pos, quat = m.pelvis()
    start, goal = np.zeros(29), np.full(29, 0.1)
    path = route(start, goal, pos, quat)
    assert len(client.configured) == 1 and client.configured[0][0] == PLANNED
    (_, _, grown, _), (_, _, bare, _) = client.requests
    first = {o.name: o for o in grown if isinstance(o, Box)}
    for o in bare:
        if isinstance(o, Box):
            np.testing.assert_allclose(np.array(first[o.name].dims) - o.dims, 0.03)
    assert route.last["ok"] and route.last["clearance_cm"] == 0.0
    legs = np.setdiff1d(np.arange(29), PLANNED_INDEX)
    np.testing.assert_allclose(path[:, legs], np.linspace(0.0, 0.1, len(path))[:, None] * np.ones(len(legs)))
    np.testing.assert_allclose(path[-1, PLANNED_INDEX], np.clip(0.1, route.lower, route.upper))


def test_a_start_in_collision_is_planned_once_more_without_the_start_check():
    client = FakeClient(statuses=[(False, "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"), (True, "")])
    m, route = _route(client)
    route(np.zeros(29), np.zeros(29), *m.pelvis())
    assert [r[3] for r in client.requests] == [True, False]
    assert route.last["start_in_collision"] and route.last["clearance_cm"] == 1.5


def test_no_plan_at_any_margin_is_none():
    client = FakeClient(ok=False)
    m, route = _route(client)
    assert route(np.zeros(29), np.zeros(29), *m.pelvis()) is None
    assert len(client.requests) == 2 and not route.last["ok"]


class PoseClient(FakeClient):
    """FakeClient that also plans to wrist poses (TODO 40.4); its joint plans fail."""

    def __init__(self):
        super().__init__(ok=False)
        self.poses = []

    def plan(self, start, goals, obstacles, check_start_validity=True):
        self.poses.append((np.array(start), [tuple(g) for g in goals], obstacles))
        path = np.linspace(start, np.asarray(start) + 0.05, 4)
        return Route(ok=True, index=0, path=path, success=(True,), seconds=0.1, status="")


def test_a_joint_goal_that_cannot_be_planned_is_tried_as_the_wrist_pose():
    from locomani.backends.scalebfm import runtime as R

    client = PoseClient()
    m, route = _route(client, pose_goal=True, tool_offset=R.OFF)
    pos, quat = m.pelvis()
    jaw_p, jaw_r = np.array([0.4, -0.2, 0.9]), np.eye(3)
    path = route(np.zeros(29), np.zeros(29), pos, quat, jaw=(jaw_p, jaw_r), pose_ok=True)
    assert path is not None and route.last["goal"] == "pose" and len(client.poses) == 1
    frame = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    np.testing.assert_allclose(client.poses[0][1][0][:3], frame.T @ (jaw_p - np.asarray(R.OFF) - pos), atol=1e-9)
    np.testing.assert_allclose(path[-1, PLANNED_INDEX], 0.05)
    client.poses.clear()
    assert route(np.zeros(29), np.zeros(29), pos, quat, jaw=(jaw_p, jaw_r), pose_ok=False) is None
    assert client.poses == []  # only a route to the reach's goal may end in CuRobo's own arm


def test_an_approach_leaves_its_target_out_of_the_obstacles():
    m, route = _route(radius=5.0)
    assert len(route.free)
    body = int(route.free[0])
    pos, quat = m.pelvis()
    names = {o.name for o in route.obstacles(pos, quat, 0.0)}
    kept = {o.name for o in route.obstacles(pos, quat, 0.0, exclude=body)}
    gone = names - kept
    assert gone and all(
        name == m.sim.m.body(body).name or name in {m.sim.m.geom(g).name for g in range(m.sim.m.ngeom)
                                                   if int(m.sim.m.geom_bodyid[g]) == body}
        for name in gone
    )  # fmt: skip
    route(np.zeros(29), np.zeros(29), pos, quat, jaw=(m.sim.d.xpos[body], np.eye(3)), approach=True)
    assert route.last["excluded"] == m.sim.m.body(body).name
    route(np.zeros(29), np.zeros(29), pos, quat, jaw=(m.sim.d.xpos[body] + 1.0, np.eye(3)), approach=True)
    assert route.last["excluded"] is None  # nothing within TARGET_M
