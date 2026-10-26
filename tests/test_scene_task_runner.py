"""The scene-task runner's pure parts (TODO 27.2.3, src/locomani/tasks/scene/): object placement stays on its surface and
clear of what rests there (oriented footprints), the robot spawns free, in the spawn region, at the asked distance and
with the object reachable; a walk is cut into straight legs that stay on traversable cells; a task names its free
objects; a family's spawn option widens the spawn to every room and any distance. The skills catalog's own tests are
in test_scene_skills_catalog.py (a sandbox with another seed drops them). Live: scripts/rsi/episode_check.py."""

import unittest

import numpy as np

from locomani.primitives.navigator import build
from locomani.tasks.scene import start as S
from locomani.tasks.scene.episode import goal_keys, others_moved
from locomani.tasks.scene.navigate import line_clear, path
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


class OthersMoved(unittest.TestCase):
    """TODO 36.9: a knocked neighbour is in the round record; the task's own objects and small drift are not."""

    def test_moved_and_tilted_neighbours_only(self):
        up, tipped = [1.0, 0, 0, 0], [np.cos(np.radians(10)), np.sin(np.radians(10)), 0, 0]  # 20 deg about x
        rest = {"vase": ([0, 0, 0.8], up), "cup": ([1, 0, 0.8], up), "jar": ([2, 0, 0.8], up),
                "spoon": ([3, 0, 0.8], up)}  # fmt: skip
        now = {"vase": ([0, 0.05, 0.8], up), "cup": ([1, 0.005, 0.8], up), "jar": ([2, 0, 0.8], tipped),
               "spoon": ([3, 0.3, 1.0], up)}  # fmt: skip
        out = others_moved(rest, now, skip=("spoon",))
        self.assertEqual(sorted(out), ["jar", "vase"])
        self.assertEqual(out["vase"]["moved_cm"], 5.0)
        self.assertAlmostEqual(out["jar"]["tilt_deg"], 20.0, places=1)
        self.assertEqual(others_moved({}, now), {})


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

    def test_spawn_option_takes_every_room_and_any_distance(self):
        m = meta()
        m["room"]["spawn"]["q"] = [[[3.0, 2.0], [3.0, 2.8], [2.0, 2.8], [2.0, 2.0]]]
        m["surfaces"]["T"]["room"] = "q"  # the object's own room is the small corner: the default spawns only there
        occ, nav = build(m)
        can = None
        for seed in range(10):
            st = S.sample_start(m, occ, nav, ["Can"], np.random.default_rng(seed), seed=seed,
                                spawn={"rooms": "all", "distance_m": [0.5, None]})  # fmt: skip
            xy, can = np.asarray(st.robot_xy), st.placements["Can"][0][:2]
            self.assertGreaterEqual(np.linalg.norm(xy - can), 0.5)
            nav.set_goal_ring(can, *S.APPROACH_RING_M)
            self.assertTrue(nav.reachable(xy))
            if not (2.0 <= xy[0] <= 3.0 and 2.0 <= xy[1] <= 2.8):
                break
        else:
            self.fail("every spawn fell in the object's own room")
        with self.assertRaises(KeyError):
            S.sample_start(m, occ, nav, ["Can"], np.random.default_rng(0), seed=0, spawn={"room": "all"})


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


class Tasks(unittest.TestCase):
    def test_task_names_its_free_objects(self):
        t = Task.from_row({"id": "x", "goal": ["on(Can, T)", "near(Box, Tray, 0.2)", "upright(Can)"]}, meta())
        self.assertEqual(t.objects, ("Can", "Box"))
        self.assertEqual(goal_keys(t.goal), ["Can", "T", "Box", "Tray"])


if __name__ == "__main__":
    unittest.main()
