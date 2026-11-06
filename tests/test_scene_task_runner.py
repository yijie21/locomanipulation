"""The scene-task runner's pure parts (TODO 27.2.3, src/locomani/tasks/scene/): object placement stays on its surface and
clear of what rests there (oriented footprints), the robot spawns free, in the spawn region, at the asked distance and
with the object reachable; a walk is cut into straight legs that stay on traversable cells; a task names its free
objects; a family's spawn option widens the spawn to every room and any distance; the physics copy of a scene points
relative asset files at the scene directory. The skills catalog's own tests are
in test_scene_skills_catalog.py (a sandbox with another seed drops them). Live: scripts/rsi/episode_check.py."""

import unittest
from pathlib import Path

import numpy as np

from locomani.primitives.navigator import build
from locomani.tasks.scene import start as S
from locomani.tasks.scene.episode import goal_keys
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

    def test_an_object_takes_its_own_placement_option_in_place_of_the_shared_one(self):
        """`placement.objects` (the pack tasks, 2026-11-05): the carton on its own band while the object keeps the
        shared reach band; an option for an object outside the task, or an unknown key in it, is refused."""
        m = meta()
        occ, nav = build(m)
        own = {"depth_m": [0.2, 0.3], "edge": "+x", "yaw_deg": [85, 90]}
        want = {"depth_m": [0.0, 0.1], "objects": {"Box": own}}
        for seed in range(5):
            st = S.sample_start(m, occ, nav, ["Can", "Box"], np.random.default_rng(seed), seed=seed, placement=want)
            can, box = st.placements["Can"], st.placements["Box"]
            self.assertLessEqual(min(S.edge_depths(m["surfaces"]["T"], can[0][:2]).values()), 0.1 + 1e-9)
            yaw = 2 * np.arctan2(box[1][3], box[1][0])
            self.assertTrue(S.placement_ok(m["surfaces"]["T"], box[0][:2], yaw, own))
            self.assertFalse(S.placement_ok(m["surfaces"]["T"], box[0][:2], yaw, {"depth_m": [0.0, 0.1]}))
        self.assertEqual(want["objects"], {"Box": own})  # the row's option is not consumed
        with self.assertRaisesRegex(KeyError, "not objects of the task"):
            S.sample_start(m, occ, nav, ["Can"], np.random.default_rng(0), seed=0, placement=want)
        with self.assertRaisesRegex(KeyError, "unknown"):
            S.sample_start(m, occ, nav, ["Can", "Box"], np.random.default_rng(0), seed=0,
                           placement={"objects": {"Box": {"blocker": {"key": "Tray"}}}})  # fmt: skip
        with self.assertRaises(RuntimeError):  # a band the table cannot give: the +x edge is 0.4 m from the centre
            S.sample_start(m, occ, nav, ["Can", "Box"], np.random.default_rng(0), seed=0,
                           placement={"objects": {"Box": {"depth_m": [0.9, 1.0], "edge": "+x"}}})  # fmt: skip

    def test_two_hand_corridors_lie_beside_the_squeezed_walls_and_in_front_from_the_nearest_edge(self):
        """`hand_corridors` (user 2026-11-06): a box 0.15 m in from the -y edge, its y axis the one closest to the
        inward normal: a strip `clear` wide beside each x wall from the edge to the far face, and one in front."""
        surf = meta()["surfaces"]["T"]
        fp = (np.array([0.0, -0.15]), np.array([0.05, 0.08]), 0.0)
        left, right, front = S.hand_corridors(surf, fp, 0.10)
        for (c, h, y), x in ((left, -0.10), (right, 0.10)):
            np.testing.assert_allclose(c, [x, -0.185], atol=1e-9)
            np.testing.assert_allclose(h, [0.115, 0.05], atol=1e-9)
            self.assertAlmostEqual(y, np.pi / 2)
        np.testing.assert_allclose(front[0], [0.0, -0.265], atol=1e-9)
        np.testing.assert_allclose(front[1], [0.035, 0.15], atol=1e-9)
        # turned a quarter, the box's x axis points at the edge: the hands squeeze its y walls instead
        turned = S.hand_corridors(surf, (fp[0], fp[1], np.pi / 2), 0.10)
        np.testing.assert_allclose(turned[0][0], [-(0.08 + 0.05), -0.15 + (0.05 - 0.15) / 2], atol=1e-9)
        # named edge: from +x, 0.4 m away
        c, h, _ = S.hand_corridors(surf, fp, 0.10, "+x")[0]
        self.assertAlmostEqual(h[0], (0.4 + 0.05) / 2)

    def test_hands_clear_keeps_the_hug_corridors_free_of_what_stays_and_of_later_task_objects(self):
        """`hands_clear_m`: the corridors of the hugged object miss every resting object, task objects placed after it
        keep out of them, and a task object placed before it does not count (it is moved by the time of the hug)."""
        m = meta()
        del m["objects"]["Tray"]
        m["surfaces"]["T"].update(half_extent=[0.6, 0.45], flat_half_extent=[0.6, 0.45], approach_edges=None)
        m["objects"]["Wall"] = dict(surface="T", free=True, init_pos=[0.0, -0.32, 0.75], init_quat=yaw_quat(0.0),
                                    box=dict(half_extent=[0.05, 0.05, 0.05]))  # fmt: skip
        m["objects"]["Big"] = dict(surface="T", free=True, init_pos=[0.3, 0.2, 0.8], init_quat=yaw_quat(0.0),
                                   box=dict(half_extent=[0.1, 0.08, 0.1]))  # fmt: skip
        occ, nav = build(m)
        surf = m["surfaces"]["T"]

        def fp(st, k):
            return S.footprint(m["objects"][k], st.placements[k][0], st.placements[k][1])

        def wall():
            o = m["objects"]["Wall"]
            return S.footprint(o, o["init_pos"], o["init_quat"])

        band = {"depth_m": [0.0, 0.3]}
        hit_without = 0
        for seed in range(30):
            st = S.sample_start(m, occ, nav, ["Big"], np.random.default_rng(seed), seed=seed, placement=band)
            hit_without += any(S.overlap(c, wall(), 0.0) for c in S.hand_corridors(surf, fp(st, "Big"), 0.15))
            st = S.sample_start(m, occ, nav, ["Big"], np.random.default_rng(seed), seed=seed,
                                placement={**band, "hands_clear_m": 0.15})  # fmt: skip
            self.assertFalse(any(S.overlap(c, wall(), 0.0) for c in S.hand_corridors(surf, fp(st, "Big"), 0.15)))
            # Big first: Can, placed after it, keeps out of its corridors
            st = S.sample_start(m, occ, nav, ["Big", "Can"], np.random.default_rng(seed), seed=seed,
                                placement={**band, "objects": {"Big": {**band, "hands_clear_m": 0.15}}})  # fmt: skip
            self.assertFalse(any(S.overlap(fp(st, "Can"), c, 0.0) for c in S.hand_corridors(surf, fp(st, "Big"), 0.15)))
        self.assertGreater(hit_without, 0)  # the condition removes starts that happen without it
        # Can first, then Big: Can may stay in Big's corridors (it is gone when Big is hugged), the wall may not
        in_corridor = 0
        for seed in range(40):
            st = S.sample_start(m, occ, nav, ["Can", "Big"], np.random.default_rng(seed), seed=seed,
                                placement={**band, "objects": {"Big": {**band, "hands_clear_m": 0.15}}})  # fmt: skip
            corridors = S.hand_corridors(surf, fp(st, "Big"), 0.15)
            self.assertFalse(any(S.overlap(c, wall(), 0.0) for c in corridors))
            in_corridor += any(S.overlap(fp(st, "Can"), c, 0.0) for c in corridors)
        self.assertGreater(in_corridor, 0)

    def test_shuffle_others_moves_the_other_objects_clear_of_the_task_objects_and_their_corridors(self):
        """`shuffle_others` (user 2026-11-06): with a resting object in every corridor the hug finds no start; moving
        the others puts each one on its surface, clear of the task objects, of one another and of the corridors."""
        m = meta()
        del m["objects"]["Tray"]
        m["surfaces"]["T"].update(half_extent=[0.6, 0.45], flat_half_extent=[0.6, 0.45], approach_edges=None)
        for i, (x, y) in enumerate([(-0.45, -0.3), (0.0, -0.3), (0.45, -0.3), (-0.45, 0.3), (0.0, 0.3), (0.45, 0.3),
                                    (-0.5, 0.0), (0.5, 0.0)]):  # fmt: skip
            m["objects"][f"O{i}"] = dict(surface="T", free=True, init_pos=[x, y, 0.75], init_quat=yaw_quat(0.0),
                                         box=dict(half_extent=[0.06, 0.06, 0.05]))  # fmt: skip
        m["objects"]["Big"] = dict(surface="T", free=True, init_pos=[0.0, 0.0, 0.8], init_quat=yaw_quat(0.0),
                                   box=dict(half_extent=[0.1, 0.08, 0.1]))  # fmt: skip
        occ, nav = build(m)
        surf = m["surfaces"]["T"]
        want = {"depth_m": [0.0, 0.3], "hands_clear_m": 0.15}
        with self.assertRaises(RuntimeError):
            S.sample_start(m, occ, nav, ["Big"], np.random.default_rng(0), seed=0, placement=want)
        moved = 0
        for seed in range(10):
            st = S.sample_start(m, occ, nav, ["Big"], np.random.default_rng(seed), seed=seed,
                                placement={**want, "shuffle_others": True})  # fmt: skip
            self.assertEqual(set(st.placements), {k for k, o in m["objects"].items() if o.get("surface") == "T"})
            fps = {k: S.footprint(m["objects"][k], *st.placements[k]) for k in st.placements}
            corridors = S.hand_corridors(surf, fps["Big"], 0.15)
            top = (np.zeros(2), np.array([0.6, 0.45]) - S.SURFACE_MARGIN_M, 0.0)
            for k, f in fps.items():
                self.assertTrue(S.inside_top(f, top))
                self.assertFalse(any(S.overlap(f, g, S.OBJECT_GAP_M) for j, g in fps.items() if j != k))
                if k != "Big":
                    self.assertFalse(any(S.overlap(f, c, 0.0) for c in corridors))
            moved += any(not np.allclose(st.placements[k][0], m["objects"][k]["init_pos"]) for k in st.placements)
        self.assertEqual(moved, 10)

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


class PhysicsCopy(unittest.TestCase):
    def test_the_physics_copy_points_relative_asset_files_at_the_scene_directory(self):
        """A logistics_room_1030 instance names its scanned meshes `gso/<key>/...` relative to its directory; the
        physics copy is written next to it, so a relative file there would not be found (the pack smoke test's first
        crash, 2026-11-05). Absolute files and the compiler's meshdir are honoured."""
        import tempfile
        import xml.etree.ElementTree as ET

        import mujoco

        from locomani.tasks.scene.world import absolute_asset_files, physics_scene

        tet = "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
        with tempfile.TemporaryDirectory() as tmp:
            sc = Path(tmp) / "scene"
            (sc / "gso" / "gso_0001").mkdir(parents=True)
            (sc / "gso" / "gso_0001" / "model.obj").write_text(tet)
            (sc / "meshes").mkdir()
            (sc / "meshes" / "part.obj").write_text(tet)
            absolute = str((sc / "meshes" / "part.obj").resolve())
            (sc / "scene_locomani.xml").write_text(
                '<mujoco><compiler angle="radian"/><asset>'
                '<mesh name="scan" file="gso/gso_0001/model.obj"/><mesh name="part" file="meshes/part.obj"/>'
                f'<mesh name="abs" file="{absolute}"/></asset><worldbody>'
                '<body name="pelvis"><freejoint/><geom type="mesh" mesh="abs" mass="1"/></body>'
                '<body name="gso_0001" pos="0 0 1"><freejoint/><geom type="mesh" mesh="scan" mass="0.1"/>'
                '<geom type="mesh" mesh="part" mass="0.1" contype="0" conaffinity="0"/></body></worldbody></mujoco>'
            )
            (sc / "scene_meta.json").write_text("{}")
            out = physics_scene(sc)
            self.assertEqual(out, sc.parent / "scene_physics")
            copy = ET.parse(out / "scene_locomani.xml").getroot()
            files = {m.get("name"): m.get("file") for m in copy.find("asset")}
            # the visual-only geom's mesh is pruned; the others are absolute under the scene directory
            self.assertEqual(files, {"scan": str((sc / "gso" / "gso_0001" / "model.obj").resolve()), "abs": absolute})
            m = mujoco.MjModel.from_xml_path(str(out / "scene_locomani.xml"))
            self.assertEqual(m.nmesh, 2)
            self.assertEqual(mujoco.MjModel.from_xml_path(str(out / "robot.xml")).nbody, 2)
            # the helper alone: meshdir / texturedir are folded into the files and dropped, two rewritten, one kept
            root = ET.fromstring('<mujoco><compiler meshdir="m" texturedir="t"/><asset><mesh file="a.obj"/>'
                                 f'<texture file="t.png"/><mesh file="{absolute}"/></asset></mujoco>')  # fmt: skip
            self.assertEqual(absolute_asset_files(root, sc), 2)
            self.assertEqual([e.get("file") for e in root.find("asset")],
                             [str(sc / "m" / "a.obj"), str(sc / "t" / "t.png"), absolute])  # fmt: skip
            self.assertEqual(root.find("compiler").attrib, {})
