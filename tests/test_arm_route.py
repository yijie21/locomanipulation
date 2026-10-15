"""The CuRobo arm-route service's pure parts (curobo_service) and the episode-side client (arm_route).

CuRobo itself lives in SIMPLE's venv and is exercised by the live replay check of the route plan, not here.
"""

import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from locomani.primitives import arm_route, curobo_service
from locomani.primitives.arm_route import (
    ArmRouteClient,
    Box,
    MeshObstacle,
    RouteServiceError,
    retime_path,
    sample_path,
    scene_obstacles,
)
from locomani.sim import ports

SCENE = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.04 0 0  0 0.04 0  0 0 0.06"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="table" pos="0.6 0 0.7" euler="0 0 30">
      <geom name="top" type="box" size="0.3 0.5 0.02"/>
      <geom name="decor" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0"/>
    </body>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom name="torso" type="box" size="0.1 0.1 0.2"/>
      <body name="hand" pos="0.3 0 0"><geom type="sphere" size="0.03"/></body>
    </body>
    <body name="can" pos="0.5 0 0.75">
      <freejoint/>
      <geom name="can_a" type="mesh" mesh="tet"/>
      <geom name="can_b" type="mesh" mesh="tet" pos="0 0 0.06" euler="0 0 90"/>
    </body>
  </worldbody>
</mujoco>
"""


class PaddingAndSelectionTests(unittest.TestCase):
    def test_goals_pad_to_the_fixed_batch_with_copies_of_the_first(self):
        goals = [[0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0]]
        padded, n = curobo_service.pad_goals(goals, 4)
        self.assertEqual(n, 2)
        self.assertEqual(len(padded), 4)
        self.assertEqual(padded[2], padded[0])
        self.assertEqual(padded[3], padded[0])

    def test_more_goals_than_the_batch_or_none_is_refused(self):
        with self.assertRaises(ValueError):
            curobo_service.pad_goals([[0, 0, 0, 1, 0, 0, 0]] * 3, 2)
        with self.assertRaises(ValueError):
            curobo_service.pad_goals([], 2)
        with self.assertRaises(ValueError):
            curobo_service.pad_goals([[0, 0, 0]], 2)

    def test_the_shortest_real_success_wins_and_padding_never_does(self):
        # goal 3 is a padding copy: even if it succeeded shortest it must not be chosen
        self.assertEqual(curobo_service.shortest_success([True, True, False, True], [30, 20, 5, 1], 3), 1)
        self.assertIsNone(curobo_service.shortest_success([False, False, True], [1, 1, 1], 2))
        # ties go to the earlier goal
        self.assertEqual(curobo_service.shortest_success([True, True], [10, 10], 2), 0)

    def test_no_success_needs_no_lengths(self):
        # CuRobo leaves path_buffer_last_tstep None when every goal failed (first live replay, 2026-10-09)
        self.assertIsNone(curobo_service.shortest_success([False], None, 1))
        with self.assertRaises(ValueError):
            curobo_service.shortest_success([True], None, 1)

    def test_limited_urdf_rewrites_only_the_named_joints(self):
        urdf = (
            '<robot name="r"><joint name="a" type="revolute"><limit lower="-1" upper="1"/></joint>'
            '<joint name="b" type="revolute"><limit lower="-2" upper="2"/></joint></robot>'
        )
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.urdf"), os.path.join(d, "out.urdf")
            with open(src, "w") as f:
                f.write(urdf)
            curobo_service.limited_urdf(src, dst, {"a": (-0.5, 0.25)})
            text = open(dst).read()
        self.assertIn('lower="-0.5" upper="0.25"', text)
        self.assertIn('lower="-2" upper="2"', text)

    def test_the_client_and_the_service_agree_on_the_protocol(self):
        self.assertEqual(arm_route.PROTOCOL, curobo_service.PROTOCOL)

    def test_the_service_port_is_shared_and_outside_every_instance_block(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ports.curobo_route(), 5795)
        for i in range(5):
            self.assertNotIn(5795, ports.instance_ports(i).values())
        with patch.dict(os.environ, {"TF_CUROBO_PORT": "6001"}, clear=True):
            self.assertEqual(ports.curobo_route(), 6001)


class SceneObstacleTests(unittest.TestCase):
    def setUp(self):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(SCENE)
        self.can_pose = (np.array([0.45, 0.1, 0.72]), np.array([np.cos(0.3), 0, 0, np.sin(0.3)]))  # wxyz, 0.6 rad yaw

    def world(self, frame_p=None, frame_r=None):
        frame_p = np.zeros(3) if frame_p is None else frame_p
        frame_r = np.eye(3) if frame_r is None else frame_r
        return scene_obstacles(self.model, {"can": self.can_pose}, frame_p, frame_r)

    def test_the_robot_planes_and_non_colliding_geoms_are_left_out(self):
        names = [o.name for o in self.world()]
        self.assertEqual(sorted(names), ["can", "top"])

    def test_mesh_vertices_land_where_mujoco_puts_them(self):
        # independent check: put the can at the same pose in MuJoCo and transform its mesh with the solver's geom frames
        mj = self.mujoco
        data = mj.MjData(self.model)
        adr = self.model.jnt_qposadr[self.model.body("can").jntadr[0]]
        data.qpos[adr : adr + 3], data.qpos[adr + 3 : adr + 7] = self.can_pose
        mj.mj_kinematics(self.model, data)
        expected = []
        for name in ("can_a", "can_b"):
            g = self.model.geom(name).id
            mid = self.model.geom_dataid[g]
            v = self.model.mesh_vert[
                self.model.mesh_vertadr[mid] : self.model.mesh_vertadr[mid] + self.model.mesh_vertnum[mid]
            ]
            expected.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        expected = np.vstack(expected)
        frame_p, frame_r = np.array([0.1, -0.2, 0.8]), Rotation.from_euler("z", 0.4).as_matrix()
        can = next(o for o in self.world(frame_p, frame_r) if o.name == "can")
        self.assertEqual(can.pose, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(can.vertices @ frame_r.T + frame_p, expected, atol=1e-9)
        self.assertEqual(len(can.faces), 2 * self.model.mesh_facenum[0])
        self.assertTrue(can.faces.max() < len(can.vertices))

    def test_a_static_box_is_posed_in_the_planning_frame(self):
        mj = self.mujoco
        data = mj.MjData(self.model)
        mj.mj_kinematics(self.model, data)
        g = self.model.geom("top").id
        frame_p, frame_r = np.array([0.1, -0.2, 0.8]), Rotation.from_euler("z", -0.7).as_matrix()
        top = next(o for o in self.world(frame_p, frame_r) if o.name == "top")
        np.testing.assert_allclose(np.array(top.pose[:3]) @ frame_r.T + frame_p, data.geom_xpos[g], atol=1e-9)
        q = top.pose[3:]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        np.testing.assert_allclose(frame_r @ r, data.geom_xmat[g].reshape(3, 3), atol=1e-9)
        np.testing.assert_allclose(top.dims, (0.6, 1.0, 0.04))

    def test_a_free_body_without_a_pose_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "can"):
            scene_obstacles(self.model, {}, np.zeros(3), np.eye(3))

    def test_the_bottle_scene_is_two_boxes_and_one_bottle(self):
        from pathlib import Path

        from locomani import paths

        xml = Path(paths.REPO) / "outputs/scene/sim_lowcom/scene_locomani.xml"
        if not xml.is_file():
            self.skipTest(f"{xml} not built")
        model = self.mujoco.MjModel.from_xml_path(str(xml))
        world = scene_obstacles(
            model, {"water_bottle_object": (np.array([0, 0, 0.8]), np.array([1, 0, 0, 0]))}, np.zeros(3), np.eye(3)
        )
        self.assertEqual(sorted(type(o).__name__ for o in world), ["Box", "Box", "MeshObstacle"])


class RetimeTests(unittest.TestCase):
    def test_the_retimed_path_starts_and_ends_at_rest_on_its_ends(self):
        path = np.array([[0.0, 0.0], [0.3, -0.1], [0.6, 0.1]])
        p, frac, total = retime_path(path, speed=0.6)
        self.assertAlmostEqual(total, 1.5 * 0.6 / 0.6)
        q0, v0 = sample_path(p, frac, total, 0.0)
        q1, v1 = sample_path(p, frac, total, total + 1.0)
        np.testing.assert_allclose(q0, path[0])
        np.testing.assert_allclose(q1, path[-1])
        np.testing.assert_allclose(v0, 0.0, atol=1e-12)
        np.testing.assert_allclose(v1, 0.0, atol=1e-12)

    def test_the_most_moved_joint_peaks_at_the_given_speed(self):
        path = np.linspace([0.0, 0.0], [0.9, 0.3], 30)
        p, frac, total = retime_path(path, speed=0.6)
        self.assertAlmostEqual(total, 1.5 * 0.9 / 0.6)
        peak = max(np.abs(sample_path(p, frac, total, t)[1]).max() for t in np.linspace(0, total, 401))
        self.assertAlmostEqual(peak, 0.6, delta=0.005)
        # the sampled positions follow the path: finite-difference velocity matches the reported one mid-way
        t, h = total / 3, 1e-5
        fd = (sample_path(p, frac, total, t + h)[0] - sample_path(p, frac, total, t - h)[0]) / (2 * h)
        np.testing.assert_allclose(fd, sample_path(p, frac, total, t)[1], atol=1e-4)

    def test_a_too_short_path_is_refused_and_a_tiny_one_takes_the_minimum_time(self):
        with self.assertRaises(ValueError):
            retime_path(np.zeros((1, 3)), 0.6)
        self.assertEqual(retime_path(np.zeros((2, 3)), 0.6, min_seconds=0.4)[2], 0.4)


class FakeService:
    """A REP socket on a free port answering each request with the next canned reply (or not at all)."""

    def __init__(self, replies):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.replies, self.requests = list(replies), []
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        for reply in self.replies:
            if not self.sock.poll(5000):
                return
            self.requests.append(json.loads(self.sock.recv()))
            if reply is None:  # swallow the request: the client must time out
                return
            self.sock.send_string(json.dumps(reply))

    def close(self):
        self.thread.join(timeout=6)
        self.sock.close()


class ClientTests(unittest.TestCase):
    def test_configure_then_plan_sends_the_world_and_returns_the_path(self):
        fake = FakeService(
            [
                dict(ok=True, built=True, seconds=1.0, protocol=arm_route.PROTOCOL),
                dict(ok=True, index=0, path=[[0.0, 0.1], [0.2, 0.3]], success=[True], status="ok", seconds=0.5),
            ]
        )
        client = ArmRouteClient(fake.port, plan_timeout_s=5, configure_timeout_s=5)
        try:
            client.configure(["j1", "j2"], [-1, -1], [1, 1], [0, 0])
            route = client.plan(
                [0, 0],
                [[0.3, -0.2, 0.1, 1, 0, 0, 0]],
                [
                    Box("table", (0.5, 0, -0.1, 1, 0, 0, 0), (0.6, 1.2, 0.05)),
                    MeshObstacle("bottle", (0.4, -0.2, 0.0, 1, 0, 0, 0), np.zeros((3, 3)), np.array([[0, 1, 2]])),
                ],
            )
        finally:
            client.close()
            fake.close()
        self.assertTrue(route.ok)
        self.assertEqual(route.path.shape, (2, 2))
        sent = fake.requests[1]
        self.assertEqual(sent["cmd"], "plan")
        self.assertEqual([c["name"] for c in sent["world"]["cuboids"]], ["table"])
        self.assertEqual([m["name"] for m in sent["world"]["meshes"]], ["bottle"])
        self.assertEqual(sent["world"]["meshes"][0]["faces"], [[0, 1, 2]])

    def test_a_joint_goal_is_sent_as_goal_joints_and_checked_for_length(self):
        fake = FakeService(
            [
                dict(ok=True, built=False, seconds=0, protocol=arm_route.PROTOCOL),
                dict(ok=True, index=0, path=[[0.0, 0.0], [0.1, 0.2]], success=[True], status="ok", seconds=0.2),
            ]
        )
        client = ArmRouteClient(fake.port, plan_timeout_s=5, configure_timeout_s=5)
        try:
            client.configure(["j1", "j2"], [-1, -1], [1, 1], [0, 0])
            with self.assertRaises(ValueError):
                client.plan_to_joints([0, 0], [0.1, 0.2, 0.3])
            route = client.plan_to_joints([0, 0], [0.1, 0.2])
        finally:
            client.close()
            fake.close()
        sent = fake.requests[1]
        self.assertEqual(sent["goal_joints"], [0.1, 0.2])
        self.assertNotIn("goals", sent)
        self.assertTrue(route.ok)
        np.testing.assert_allclose(route.path[-1], [0.1, 0.2])

    def test_the_service_refuses_a_joint_goal_off_its_limits(self):
        np.testing.assert_allclose(curobo_service.check_goal_joints([0.0, 0.5], [-1, -1], [1, 1]), [0.0, 0.5])
        with self.assertRaisesRegex(ValueError, "outside"):
            curobo_service.check_goal_joints([0.0, 1.2], [-1, -1], [1, 1])
        with self.assertRaises(ValueError):
            curobo_service.check_goal_joints([0.0], [-1, -1], [1, 1])
        with self.assertRaises(ValueError):
            curobo_service.check_goal_joints([0.0, np.nan], [-1, -1], [1, 1])

    def test_a_service_error_raises_instead_of_returning_a_failed_route(self):
        fake = FakeService(
            [
                dict(ok=True, built=False, seconds=0, protocol=arm_route.PROTOCOL),
                dict(ok=False, status="error", error="boom"),
            ]
        )
        client = ArmRouteClient(fake.port, plan_timeout_s=5, configure_timeout_s=5)
        try:
            client.configure(["j"], [-1], [1], [0])
            with self.assertRaisesRegex(RouteServiceError, "boom"):
                client.plan([0], [[0, 0, 0, 1, 0, 0, 0]])
        finally:
            client.close()
            fake.close()

    def test_a_silent_service_times_out_and_the_client_can_send_again(self):
        fake = FakeService([None])
        client = ArmRouteClient(fake.port, configure_timeout_s=0.3)
        try:
            with self.assertRaisesRegex(RouteServiceError, "no reply"):
                client.configure(["j"], [-1], [1], [0])
            self.assertIsNone(client.sock)  # the stuck REQ socket was dropped
        finally:
            client.close()
            fake.close()

    def test_a_protocol_mismatch_is_refused(self):
        fake = FakeService([dict(ok=True, built=True, seconds=0, protocol=99)])
        client = ArmRouteClient(fake.port, configure_timeout_s=5)
        try:
            with self.assertRaisesRegex(RouteServiceError, "protocol"):
                client.configure(["j"], [-1], [1], [0])
        finally:
            client.close()
            fake.close()

    def test_plan_before_configure_and_a_wrong_start_length_are_refused(self):
        client = ArmRouteClient(1)
        with self.assertRaises(RouteServiceError):
            client.plan([0], [[0, 0, 0, 1, 0, 0, 0]])
        client.names = ["a", "b"]
        with self.assertRaises(ValueError):
            client.plan([0], [[0, 0, 0, 1, 0, 0, 0]])


class RouteCacheTests(unittest.TestCase):
    PLANNER = dict(
        settings=dict(cuda_graph=True, checker="primitive"), batch=2, robot="abc", protocol=arm_route.PROTOCOL
    )

    def configured(self, replies, cache, planner=PLANNER):
        fake = FakeService(
            [dict(ok=True, built=True, seconds=0, protocol=arm_route.PROTOCOL, planner=planner), *replies]
        )
        client = ArmRouteClient(fake.port, plan_timeout_s=5, configure_timeout_s=5, cache_dir=cache)
        client.configure(["j1", "j2"], [-1, -1], [1, 1], [0, 0])
        return fake, client

    def test_a_repeated_request_is_answered_from_the_cache_with_the_first_path(self):
        first = dict(ok=True, index=0, path=[[0.0, 0.0], [0.1, 0.2]], success=[True], status="ok", seconds=6.0)
        with tempfile.TemporaryDirectory() as cache:
            fake, client = self.configured([first], cache)
            try:
                a = client.plan_to_joints([0, 0], [0.1, 0.2], [Box("wall", (0.5, 0, 0, 1, 0, 0, 0), (0.1, 1, 1))])
                b = client.plan_to_joints([0, 0], [0.1, 0.2], [Box("wall", (0.5, 0, 0, 1, 0, 0, 0), (0.1, 1, 1))])
            finally:
                client.close()
                fake.close()
            self.assertEqual(len(fake.requests), 2)  # configure + one plan: the repeat never reached the service
            self.assertFalse(a.cached)
            self.assertTrue(b.cached)
            np.testing.assert_array_equal(a.path, b.path)
            self.assertEqual(b.seconds, 6.0)

    def test_another_start_obstacle_or_planner_is_a_new_request(self):
        reply = dict(ok=True, index=0, path=[[0.0, 0.0], [0.1, 0.2]], success=[True], status="ok", seconds=1.0)
        with tempfile.TemporaryDirectory() as cache:
            fake, client = self.configured([reply, reply, reply], cache)
            try:
                client.plan_to_joints([0, 0], [0.1, 0.2])
                client.plan_to_joints([0, 1e-12], [0.1, 0.2])
                moved = client.plan_to_joints([0, 0], [0.1, 0.2], [Box("wall", (0.5, 0, 0, 1, 0, 0, 0), (0.1, 1, 1))])
            finally:
                client.close()
                fake.close()
            self.assertFalse(moved.cached)
            self.assertEqual(len(fake.requests), 4)
            other = dict(self.PLANNER, settings=dict(cuda_graph=False, checker="mesh"))
            fake, client = self.configured([reply], cache, planner=other)
            try:
                self.assertFalse(client.plan_to_joints([0, 0], [0.1, 0.2]).cached)
            finally:
                client.close()
                fake.close()

    def test_a_failed_plan_is_cached_too_but_a_service_error_is_not(self):
        failed = dict(ok=False, index=None, path=[], success=[False], status="INVALID_START_STATE_WORLD_COLLISION",
                      seconds=2.0)  # fmt: skip
        with tempfile.TemporaryDirectory() as cache:
            fake, client = self.configured([dict(ok=False, status="error", error="boom"), failed], cache)
            try:
                with self.assertRaisesRegex(RouteServiceError, "boom"):
                    client.plan_to_joints([0, 0], [0.1, 0.2])
                self.assertFalse(client.plan_to_joints([0, 0], [0.1, 0.2]).cached)
                again = client.plan_to_joints([0, 0], [0.1, 0.2])
            finally:
                client.close()
                fake.close()
            self.assertTrue(again.cached)
            self.assertFalse(again.ok)

    def test_caching_needs_the_services_planner_fingerprint(self):
        with tempfile.TemporaryDirectory() as cache:
            fake = FakeService([dict(ok=True, built=True, seconds=0, protocol=arm_route.PROTOCOL)])
            client = ArmRouteClient(fake.port, configure_timeout_s=5, cache_dir=cache)
            try:
                with self.assertRaisesRegex(RouteServiceError, "fingerprint"):
                    client.configure(["j"], [-1], [1], [0])
            finally:
                client.close()
                fake.close()


class ServiceSettingsTests(unittest.TestCase):
    def test_the_defaults_are_the_graspnet_settings_and_bad_values_are_refused(self):
        s = curobo_service.Settings()
        self.assertEqual((s.cuda_graph, s.checker, s.ik_iters, s.trajopt_iters), (False, "mesh", 200, 200))
        with self.assertRaises(ValueError):
            curobo_service.Settings(checker="sdf")
        with self.assertRaises(ValueError):
            curobo_service.Settings(trajopt_iters=0)

    def test_several_service_ports_come_from_tf_curobo_ports(self):
        with patch.dict(os.environ, {"TF_CUROBO_PORTS": "5795, 5796,5797"}, clear=True):
            self.assertEqual(ports.curobo_routes(), [5795, 5796, 5797])
        with patch.dict(os.environ, {"TF_CUROBO_PORT": "6001"}, clear=True):
            self.assertEqual(ports.curobo_routes(), [6001])


if __name__ == "__main__":
    unittest.main()
