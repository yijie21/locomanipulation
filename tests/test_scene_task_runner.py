"""The scene-task runner's pure parts (TODO 27.2.3, src/locomani/tasks/scene/): object placement stays on its surface and
clear of what rests there (oriented footprints), the robot spawns free, in the spawn region, at the asked distance and
with the object reachable; a walk is cut into straight legs that stay on traversable cells; a task names its free
objects; the seed pick chooses an edge from which a side of the object fits the jaw. Live: scripts/experiments/
scene_rsi_20261016/episode_check.py."""

import unittest

import numpy as np

from locomani.primitives.navigator import build
from locomani.tasks.scene import start as S
from locomani.tasks.scene.episode import goal_keys
from locomani.tasks.scene.navigate import line_clear, path
from locomani.tasks.scene.skills.pick import JAW_SPAN_M, facings, grasp_yaw
from locomani.tasks.scene.world import Task


def yaw_quat(y):
    return [float(np.cos(y / 2)), 0.0, 0.0, float(np.sin(y / 2))]


def meta():
    """A 6 x 6 m room, a 0.8 x 0.6 table in the middle, a tray across most of it, a can and a box on it."""
    table = dict(xy=[0.0, 0.0], half_extent=[0.4, 0.3], flat_half_extent=[0.4, 0.3], yaw=0.0, top=0.7,
                 approach_edges=["+x", "-y"], room="r", of_object="Table")  # fmt: skip
    objects = {
        "Tray": dict(
            surface="T",
            free=False,
            init_pos=[0.0, -0.08, 0.72],
            init_quat=yaw_quat(0.0),
            box=dict(half_extent=[0.35, 0.18, 0.02]),
        ),  # fmt: skip
        "Can": dict(
            surface="T",
            free=True,
            body="Can_object",
            init_pos=[0.2, 0.2, 0.79],
            init_quat=yaw_quat(0.0),
            box=dict(half_extent=[0.04, 0.06, 0.09]),
        ),  # fmt: skip
        "Box": dict(
            surface="T",
            free=True,
            body="Box_object",
            init_pos=[-0.3, 0.2, 0.75],
            init_quat=yaw_quat(0.0),
            box=dict(half_extent=[0.05, 0.05, 0.05]),
        ),  # fmt: skip
    }
    room = dict(bounds=dict(x=[-3.0, 3.0], y=[-3.0, 3.0]), walls=[], obstacles=[],
                spawn={"r": [[[-2.8, -2.8], [2.8, -2.8], [2.8, 2.8], [-2.8, 2.8]]]})  # fmt: skip
    return dict(objects=objects, surfaces={"T": table}, room=room)


class Placement(unittest.TestCase):
    def test_object_lands_on_its_surface_clear_of_the_tray_and_the_box(self):
        m = meta()
        taken = [
            S.footprint(m["objects"][k], m["objects"][k]["init_pos"], m["objects"][k]["init_quat"])
            for k in ("Tray", "Box")
        ]
        top = (np.zeros(2), np.array([0.4, 0.3]) - S.SURFACE_MARGIN_M, 0.0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            pos, quat = S.place_on_surface(m, "Can", rng, taken)
            fp = S.footprint(m["objects"]["Can"], pos, quat)
            self.assertTrue(S.inside_top(fp, top))
            self.assertFalse(any(S.overlap(fp, t, S.OBJECT_GAP_M) for t in taken))
            self.assertAlmostEqual(pos[2], 0.79)

    def test_overlap_uses_orientation(self):
        long = (np.zeros(2), np.array([0.3, 0.02]), 0.0)
        # a square 0.1 m above the long thin box's centre: clear when it lies along x, hit when it is turned upright
        sq = (np.array([0.0, 0.1]), np.array([0.03, 0.03]), 0.0)
        self.assertFalse(S.overlap(long, sq, 0.01))
        self.assertTrue(S.overlap((np.zeros(2), np.array([0.3, 0.02]), np.pi / 2), sq, 0.01))

    def test_spawn_is_free_in_region_at_distance_and_reaches_the_object(self):
        m = meta()
        occ, nav = build(m)
        st = S.sample_start(m, occ, nav, ["Can"], np.random.default_rng(3), seed=3)
        xy = np.asarray(st.robot_xy)
        can = st.placements["Can"][0][:2]
        self.assertTrue(occ.is_free(xy))
        self.assertTrue(S.SPAWN_DISTANCE_M[0] <= np.linalg.norm(xy - can) <= S.SPAWN_DISTANCE_M[1])
        nav.set_goal_ring(can, *S.APPROACH_RING_M)
        self.assertTrue(nav.reachable(xy))
        again = S.Start.from_dict(st.to_dict())
        np.testing.assert_allclose(again.placements["Can"][0], st.placements["Can"][0])


class Paths(unittest.TestCase):
    def test_path_goes_around_the_table_in_straight_clear_legs(self):
        m = meta()
        occ, nav = build(m)
        corners = path(occ, nav, (-2.0, 0.0), (2.0, 0.0))
        self.assertIsNotNone(corners)
        self.assertGreater(len(corners), 2)  # not straight through the table
        for a, b in zip(corners[:-2], corners[1:-1]):
            self.assertTrue(line_clear(nav.trav, occ, a, b))
        np.testing.assert_allclose(corners[-1], [2.0, 0.0])


class TasksAndPick(unittest.TestCase):
    def test_task_names_its_free_objects(self):
        t = Task.from_row({"id": "x", "goal": ["on(Can, T)", "near(Box, Tray, 0.2)", "upright(Can)"]}, meta())
        self.assertEqual(t.objects, ("Can", "Box"))
        self.assertEqual(goal_keys(t.goal), ["Can", "T", "Box", "Tray"])

    def test_pick_faces_an_edge_whose_approach_closes_across_a_fitting_side(self):
        m = meta()
        f = facings(m["surfaces"]["T"], [0.3, 0.2])
        self.assertEqual(len(f), 2)
        self.assertLessEqual(f[0][1], f[1][1])
        # 8 x 12 cm can lying along x: only an approach along x (from the +x edge, facing -x) closes across 8 cm
        half = [0.06, 0.04, 0.09]
        self.assertIsNone(grasp_yaw(yaw_quat(0.0), half, -np.pi / 2))  # from the -y edge, facing +y
        self.assertAlmostEqual(abs(grasp_yaw(yaw_quat(0.0), half, np.pi)), np.pi)
        self.assertEqual(grasp_yaw(yaw_quat(0.3), [JAW_SPAN_M / 2] * 3, 1.0), 1.0)  # both sides fit: face it


if __name__ == "__main__":
    unittest.main()
