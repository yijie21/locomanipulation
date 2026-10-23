"""TODO 33.12 (user 2026-10-23 "接着做 33.12"): the stance-and-grasp chooser -- grasps from the section, stances for both
hands kept clear of the furniture, scored by the body model, the fallback order (standing, lowered pelvis, a
preparatory action)."""

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

from locomani.tasks.scene import body  # noqa: E402
from locomani.tasks.scene import choose as C  # noqa: E402
from locomani.tasks.scene import graspability as G  # noqa: E402

TOP = 0.70
TABLE_HALF = 0.30  # a small square table at the origin, the robot can stand on every side


def scene(bodies: str, objects: dict) -> G.StaticScene:
    xml = f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    m = mujoco.MjModel.from_xml_string(xml)
    d = Path(tempfile.mkdtemp())
    meta = dict(room=dict(bounds=dict(x=[-3.0, 3.0], y=[-3.0, 3.0])),
                surfaces={"table": dict(xy=[0.0, 0.0], top=TOP, half_extent=[TABLE_HALF, TABLE_HALF, 0.01], yaw=0.0)},
                objects=objects)  # fmt: skip
    (d / "scene_meta.json").write_text(json.dumps(meta))
    return G.StaticScene(d, model=m)


def box_body(name: str, half, z: float) -> str:
    return (f'<body name="{name}" pos="0 0 {z}"><freejoint/>'
            f'<geom type="box" size="{half[0]} {half[1]} {half[2]}"/></body>')  # fmt: skip


def obj(name: str, half, z: float, **extra) -> dict:
    return dict(body=name, surface="table", init_pos=[0, 0, z], init_quat=[1, 0, 0, 0], free=True,
                box=dict(half_extent=list(half)), **extra)  # fmt: skip


def table(path: Path) -> None:
    """A synthetic envelope: the straight-down wrist misses, everything within 0.65 m lands, beyond misses."""
    rows = []
    for x in np.arange(0.2, 0.95, 0.1):
        for y in (-0.05, 0.05, 0.15):
            for z in np.arange(0.4, 1.2, 0.1):
                for att in (0.0, 1.0, 2.0):
                    for pel in (0.78, 0.68, 0.62):
                        for hand in (0.0, 1.0):
                            rows.append([x, y, z, att, pel, hand])
    X = np.array(rows)
    miss = ((X[:, 0] > 0.65) | (X[:, 3] == 1.0)).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), y_miss=miss.astype(np.float32),
             y_err=np.where(miss > 0, 10.0, 1.0).astype(np.float32), mu=mu, sd=sd,
             features=np.array(["x", "y_out", "z", "att", "pelvis", "hand"]), k=np.array(15), miss_cm=np.array(3.0),
             stand_z=np.array(0.78))  # fmt: skip


class Chooser(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = str(Path(self.tmp.name) / "table.npz")
        table(Path(self.table))
        body.table.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_wide_box_is_taken_across_its_narrow_side_from_either_hand(self):
        # 14 cm along x, 6 cm along y: only the approaches along y (closing across x... no: closing across y) fit
        sc = scene(box_body("bar", (0.07, 0.03, 0.05), TOP + 0.05), {"bar": obj("bar", (0.07, 0.03, 0.05), TOP + 0.05)})
        gs = C.grasps(sc, "bar")
        self.assertTrue(gs)
        self.assertTrue(all(g["kind"] == "side" for g in gs))
        # the approach runs along x (the long axis) so the jaw closes across y (the 6 cm side)
        self.assertTrue(
            all(min(abs(C._wrap(g["approach_deg"])), abs(C._wrap(g["approach_deg"] - 180))) <= 30 for g in gs)
        )
        opts = C.options(sc, "bar", start_xy=(-2.0, 0.0), table=self.table)
        self.assertTrue(opts)
        self.assertEqual({o.hand for o in opts}, {"left", "right"})
        self.assertTrue(all(o.clear_m >= C.STANCE_CLEAR_M for o in opts))
        self.assertTrue(all(o.distance_m <= 0.65 for o in opts if o.p_miss < 0.5))
        best = C.choose(sc, "bar", start_xy=(-2.0, 0.0), table=self.table)
        self.assertFalse(best.needs_prep)
        self.assertIsNone(best.pelvis_z)  # standing works: no lowered pelvis
        self.assertLess(best.walk_m, 3.0)  # from the -x side it stands on the -x side, not around the table
        self.assertLess(abs(C._wrap(best.approach_deg)), 30)

    def test_a_flat_object_gets_a_tilted_pinch_not_a_straight_down_one(self):
        sc = scene(
            box_body("card", (0.05, 0.02, 0.008), TOP + 0.008), {"card": obj("card", (0.05, 0.02, 0.008), TOP + 0.008)}
        )
        gs = C.grasps(sc, "card")
        self.assertEqual({g["kind"] for g in gs}, {"top"})
        self.assertEqual({g["att"] for g in gs}, {"tilt45", "down"})
        best = C.choose(sc, "card", start_xy=(0.0, -2.0), table=self.table)
        self.assertEqual((best.kind, best.att), ("top", "tilt45"))  # the synthetic envelope says "down" misses
        self.assertFalse(best.needs_prep)

    def test_nothing_fits_asks_for_a_preparatory_action(self):
        big = (0.09, 0.09, 0.06)  # 18 cm every way: no side fits, not flat, not a container
        sc = scene(box_body("block", big, TOP + 0.06), {"block": obj("block", big, TOP + 0.06)})
        self.assertEqual(C.grasps(sc, "block"), [])
        self.assertEqual(C.options(sc, "block", start_xy=(0.0, -2.0), table=self.table), [])
        self.assertIsNone(C.choose(sc, "block", start_xy=(0.0, -2.0), table=self.table))

    def test_a_far_reach_falls_back_to_the_relaxed_listing_and_says_so(self):
        # the object sits 0.9 m in from every edge of a big table: no clear stance within 0.70 m
        sc = scene(box_body("cup", (0.03, 0.03, 0.05), TOP + 0.05), {"cup": obj("cup", (0.03, 0.03, 0.05), TOP + 0.05)})
        sc.meta["surfaces"]["table"]["half_extent"] = [0.9, 0.9, 0.01]
        sc.occ = sc.nav = None
        self.assertEqual(C.options(sc, "cup", start_xy=(0.0, -2.5), table=self.table), [])
        best = C.choose(sc, "cup", start_xy=(0.0, -2.5), table=self.table)
        self.assertIsNone(best)  # even relaxed (0.80 m, 0.12 m clear) the hand does not get there

    def test_explain_and_as_dict(self):
        sc = scene(box_body("bar", (0.07, 0.03, 0.05), TOP + 0.05), {"bar": obj("bar", (0.07, 0.03, 0.05), TOP + 0.05)})
        opts = C.options(sc, "bar", table=self.table)[:2]
        text = C.explain(opts)
        self.assertIn("side grasp, ", text)
        self.assertIn("miss odds", text)
        d = opts[0].as_dict()
        self.assertEqual(set(d) >= {"kind", "hand", "att", "stance_xy", "p_miss", "score", "why"}, True)
        self.assertIsNone(d["walk_m"])  # no start given
        self.assertAlmostEqual(math.degrees(opts[0].stance_yaw), opts[0].approach_deg, places=6)


if __name__ == "__main__":
    unittest.main()
