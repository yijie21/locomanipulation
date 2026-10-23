"""TODO 33.5 (user 2026-10-23 "先做 33.5"): the preparatory-action layer -- goals (turn, pull, clear) the selector proposes
when the chooser has no stance, judged by re-choosing on the moved object; the `placement` start option that draws the
situations that need them."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.navigator import build as build_navigation  # noqa: E402
from locomani.tasks.scene import body  # noqa: E402
from locomani.tasks.scene import choose as C  # noqa: E402
from locomani.tasks.scene import graspability as G  # noqa: E402
from locomani.tasks.scene import prep as P  # noqa: E402
from locomani.tasks.scene import start as S  # noqa: E402

TOP = 0.70


def meta_for(table_xy, half, objects: dict, edges=("-y",)) -> dict:
    """A room 6 m square whose table stands against the +y wall with walls either side of it: only its -y edge is open."""
    x0, y0 = table_xy
    hx, hy = half
    return dict(
        room=dict(bounds=dict(x=[-3.0, 3.0], y=[-3.0, 3.0]),
                  obstacles=[[[x0 - hx - 2.0, y0 - hy], [x0 - hx - 0.01, y0 - hy], [x0 - hx - 0.01, 3.0], [x0 - hx - 2.0, 3.0]],
                             [[x0 + hx + 0.01, y0 - hy], [x0 + hx + 2.0, y0 - hy], [x0 + hx + 2.0, 3.0], [x0 + hx + 0.01, 3.0]]],
                  spawn={"r": [[[-2.5, -2.5], [2.5, -2.5], [2.5, -1.0], [-2.5, -1.0]]]}),
        surfaces={"table": dict(xy=[x0, y0], top=TOP, half_extent=[hx, hy, 0.01], yaw=0.0, approach_edges=list(edges))},
        objects=objects,
    )  # fmt: skip


def scene(bodies: str, meta: dict) -> G.StaticScene:
    m = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")
    d = Path(tempfile.mkdtemp())
    (d / "scene_meta.json").write_text(json.dumps(meta))
    return G.StaticScene(d, model=m)


def box_body(name: str, half, pos) -> str:
    return (f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint/>'
            f'<geom type="box" size="{half[0]} {half[1]} {half[2]}"/></body>')  # fmt: skip


def obj(name: str, half, pos, quat=(1, 0, 0, 0)) -> dict:
    return dict(body=name, surface="table", init_pos=list(pos), init_quat=list(quat), free=True,
                box=dict(half_extent=list(half)))  # fmt: skip


def table_file(path: Path) -> None:
    """A synthetic envelope: everything within 0.65 m lands, beyond misses; the straight-down wrist misses."""
    rows = [[x, y, z, att, pel, hand] for x in np.arange(0.2, 0.95, 0.1) for y in (-0.05, 0.05, 0.15)
            for z in np.arange(0.4, 1.2, 0.1) for att in (0.0, 1.0, 2.0) for pel in (0.78, 0.68, 0.62) for hand in (0.0, 1.0)]  # fmt: skip
    X = np.array(rows)
    miss = ((X[:, 0] > 0.65) | (X[:, 3] == 1.0)).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), y_miss=miss.astype(np.float32),
             y_err=np.where(miss > 0, 10.0, 1.0).astype(np.float32), mu=mu, sd=sd,
             features=np.array(["x", "y_out", "z", "att", "pelvis", "hand"]), k=np.array(15), miss_cm=np.array(3.0),
             stand_z=np.array(0.78))  # fmt: skip


QUAT_90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


class Prep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = str(Path(self.tmp.name) / "table.npz")
        table_file(Path(self.table))
        body.table.cache_clear()
        P.ACTIONS.clear()

    def tearDown(self):
        self.tmp.cleanup()
        P.ACTIONS.clear()

    def test_a_bar_whose_narrow_side_lies_along_the_open_edge_wants_a_turn(self):
        # a 14 x 6 cm bar at a table open on -y only, its long axis along x (along the open edge): from -y the jaw
        # would close across 14 cm; the approaches along x that fit have no stance (walls either side)
        pos = (0.0, 2.6, TOP + 0.05)
        m = meta_for((0.0, 2.7), (0.3, 0.3), {"bar": obj("bar", (0.07, 0.03, 0.05), pos)})
        sc = scene(box_body("bar", (0.07, 0.03, 0.05), pos), m)
        sc.place({"bar": (np.array(pos), np.array([1.0, 0.0, 0.0, 0.0]))})
        before = C.choose(sc, "bar", (0.0, -1.5), table=self.table)
        self.assertTrue(before is None or before.needs_prep)
        plans = P.propose(sc, "bar", (0.0, -1.5), table=self.table)
        self.assertTrue(plans)
        self.assertEqual(plans[0].goal.kind, "turn")
        self.assertGreaterEqual(abs(plans[0].goal.turn_deg), 60.0)  # 60 deg already lets an oblique approach fit
        self.assertLessEqual(plans[0].after["p_miss"], C.P_MISS_OK)
        self.assertEqual(plans[0].after["kind"], "side")
        self.assertFalse(plans[0].implemented)
        self.assertEqual(P.wanted(plans)[0], "turn")
        self.assertIsNone(P.select(sc, "bar", (0.0, -1.5), table=self.table))

        @P.register("turn")
        def turn(ctx, plan):
            return "turned"

        plan = P.select(sc, "bar", (0.0, -1.5), table=self.table)
        self.assertEqual(plan.goal.kind, "turn")
        self.assertTrue(plan.implemented)
        self.assertEqual(P.run(None, plan), "turned")
        # the goal is met within tolerance, and not when the object is still where it was
        self.assertTrue(P.met(plan.goal, plan.goal.pos, plan.goal.quat))
        self.assertFalse(P.met(plan.goal, pos, (1.0, 0.0, 0.0, 0.0)))
        self.assertTrue(P.met(plan.goal, pos, P.turned((1.0, 0.0, 0.0, 0.0), plan.goal.turn_deg)))
        self.assertIn("turn", P.explain(plans[:1]))

    def test_a_cup_deep_on_a_long_table_wants_a_pull_toward_the_open_edge(self):
        pos = (0.0, 1.9, TOP + 0.05)  # 0.8 m in from the -y edge at y = 1.1: no stance within 0.70 m
        m = meta_for((0.0, 2.0), (0.3, 0.9), {"cup": obj("cup", (0.03, 0.03, 0.05), pos)})
        sc = scene(box_body("cup", (0.03, 0.03, 0.05), pos), m)
        sc.place({"cup": (np.array(pos), np.array([1.0, 0, 0, 0]))})
        self.assertEqual(C.options(sc, "cup", (0.0, -1.5), table=self.table), [])
        plans = P.propose(sc, "cup", (0.0, -1.5), table=self.table)
        self.assertTrue(plans)
        self.assertEqual({x.goal.kind for x in plans}, {"pull"})
        self.assertGreaterEqual(plans[0].goal.move_m, 0.3)
        self.assertAlmostEqual(plans[0].goal.direction_deg, -90.0)  # toward -y
        self.assertAlmostEqual(plans[0].goal.pos[1], pos[1] - plans[0].goal.move_m, places=6)
        self.assertTrue(all(x.goal.pos[1] >= 1.1 + P.EDGE_IN_M - 1e-6 for x in plans))  # never past the edge margin

    def test_nothing_to_prepare_when_the_chooser_already_has_a_stance(self):
        pos = (0.0, 2.55, TOP + 0.05)  # 0.15 m in from the open edge
        m = meta_for((0.0, 2.7), (0.3, 0.3), {"cup": obj("cup", (0.03, 0.03, 0.05), pos)})
        sc = scene(box_body("cup", (0.03, 0.03, 0.05), pos), m)
        sc.place({"cup": (np.array(pos), np.array([1.0, 0, 0, 0]))})
        self.assertFalse(C.choose(sc, "cup", (0.0, -1.5), table=self.table).needs_prep)
        self.assertEqual(P.propose(sc, "cup", (0.0, -1.5), table=self.table), [])

    def test_edges_and_on_support(self):
        surf = dict(xy=[1.0, 2.0], half_extent=[0.5, 0.3, 0.01], yaw=0.0, approach_edges=["-y", "+x"])
        e = {name: (n.round(3).tolist(), pt.round(3).tolist()) for name, n, pt in P.edges(surf)}
        self.assertEqual(e["-y"], ([0.0, -1.0], [1.0, 1.7]))
        self.assertEqual(e["+x"], ([1.0, 0.0], [1.5, 2.0]))
        pos = (0.0, 2.6, TOP + 0.05)
        m = meta_for((0.0, 2.7), (0.3, 0.3), {"cup": obj("cup", (0.03, 0.03, 0.05), pos)})
        sc = scene(box_body("cup", (0.03, 0.03, 0.05), pos), m)
        self.assertTrue(P.on_support(sc, "cup", pos, (1, 0, 0, 0)))
        self.assertFalse(P.on_support(sc, "cup", (0.0, 2.3, pos[2]), (1, 0, 0, 0)))  # past the -y edge

    def test_a_neighbour_in_the_corridor_asks_for_a_clear(self):
        # a bar lies across the hand's way between the cup and the open -y edge: no approach from that side passes it
        cup, can = (0.0, 2.55, TOP + 0.05), (0.0, 2.43, TOP + 0.08)
        m = meta_for(
            (0.0, 2.7),
            (0.3, 0.3),
            {"cup": obj("cup", (0.03, 0.03, 0.05), cup), "can": obj("can", (0.08, 0.03, 0.08), can)},
        )
        sc = scene(box_body("cup", (0.03, 0.03, 0.05), cup) + box_body("can", (0.08, 0.03, 0.08), can), m)
        sc.place({"cup": (np.array(cup), np.array([1.0, 0, 0, 0])), "can": (np.array(can), np.array([1.0, 0, 0, 0]))})
        before = C.choose(sc, "cup", (0.0, -1.5), table=self.table)
        self.assertIsNotNone(before)
        self.assertTrue(before.needs_prep)
        self.assertEqual(before.corridor, "can")
        plans = P.propose(sc, "cup", (0.0, -1.5), table=self.table)
        kinds = [x.goal.kind for x in plans]
        self.assertIn("clear", kinds)
        clear = next(x for x in plans if x.goal.kind == "clear")
        self.assertEqual(clear.goal.key, "can")
        self.assertGreater(float(np.hypot(clear.goal.pos[0] - can[0], clear.goal.pos[1] - can[1])), 0.09)  # moved
        self.assertIsNone(clear.after["corridor"])
        self.assertIsNone(clear.remaining)
        self.assertIn("clear", P.wanted(plans))


class Placement(unittest.TestCase):
    def _meta(self):
        cup, can = (0.0, 2.6, TOP + 0.05), (0.2, 2.6, TOP + 0.08)
        return meta_for((0.0, 2.0), (0.3, 0.9), {"cup": obj("cup", (0.03, 0.03, 0.05), cup),
                                                "can": obj("can", (0.03, 0.03, 0.08), can),
                                                "bar": obj("bar", (0.07, 0.03, 0.05), (0.0, 1.5, TOP + 0.05))})  # fmt: skip

    def test_depth_yaw_and_blocker_options(self):
        meta = self._meta()
        occ, nav = build_navigation(meta, body_radius=0.35)
        surf = meta["surfaces"]["table"]
        for seed in range(5):
            st = S.sample_start(meta, occ, nav, ["cup"], np.random.default_rng(seed), seed=seed,
                                spawn={"rooms": "all", "distance_m": [0.5, None]}, placement={"depth_m": [0.4, 0.6]})  # fmt: skip
            d = S.edge_depths(surf, st.placements["cup"][0][:2])["-y"]
            self.assertTrue(0.4 - 1e-6 <= d <= 0.6 + 1e-6, d)
        for seed in range(5):
            st = S.sample_start(meta, occ, nav, ["bar"], np.random.default_rng(seed), seed=seed,
                                spawn={"rooms": "all", "distance_m": [0.5, None]},
                                placement={"depth_m": [0.2, 0.5], "yaw_deg": [80, 90]})  # fmt: skip
            fp = S.footprint(meta["objects"]["bar"], st.placements["bar"][0][:2], st.placements["bar"][1])
            self.assertTrue(S.placement_ok(surf, fp[0], fp[2], {"yaw_deg": [80, 90]}))
            rel = abs((fp[2] - math.atan2(-1.0, 0.0) + math.pi / 2) % math.pi - math.pi / 2)
            self.assertGreaterEqual(math.degrees(rel), 80 - 1e-6)
        st = S.sample_start(meta, occ, nav, ["cup"], np.random.default_rng(3), seed=3,
                            spawn={"rooms": "all", "distance_m": [0.5, None]},
                            placement={"depth_m": [0.3, 0.6], "blocker": {"key": "can", "gap_m": 0.03}})  # fmt: skip
        self.assertIn("can", st.placements)
        cup_xy, can_xy = st.placements["cup"][0][:2], st.placements["can"][0][:2]
        self.assertLess(can_xy[1], cup_xy[1])  # the can sits between the cup and the open -y edge
        self.assertAlmostEqual(can_xy[0], cup_xy[0], places=6)
        gap = (
            float(np.linalg.norm(can_xy - cup_xy))
            - S.footprint_radius(meta["objects"]["cup"])
            - S.footprint_radius(meta["objects"]["can"])
        )
        self.assertAlmostEqual(gap, 0.03, places=6)
        with self.assertRaises(KeyError):
            S.sample_start(meta, occ, nav, ["cup"], np.random.default_rng(0), seed=0, placement={"depth": 1})
        with self.assertRaises(KeyError):
            S.sample_start(meta, occ, nav, ["cup"], np.random.default_rng(0), seed=0,
                           placement={"blocker": {"key": "cup"}})  # fmt: skip
        with self.assertRaises(RuntimeError):  # the table is 1.8 m deep: no spot 5 m in
            S.sample_start(meta, occ, nav, ["cup"], np.random.default_rng(0), seed=0,
                           spawn={"rooms": "all", "distance_m": [0.5, None]}, placement={"depth_m": [5.0, 6.0]})  # fmt: skip


if __name__ == "__main__":
    unittest.main()
