"""TODO 33.18: the jaw fit check (src/locomani/tasks/scene/graspability.py) on synthetic objects."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from locomani.tasks.scene import graspability as G

TOP = 0.70


def scene(bodies: str, objects: dict) -> G.StaticScene:
    """A StaticScene over a tiny model: free bodies on a 0.70 m support."""
    xml = f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    m = mujoco.MjModel.from_xml_string(xml)
    d = Path(tempfile.mkdtemp())
    meta = dict(surfaces={"table": dict(top=TOP, half_extent=[1, 1, 0.01], yaw=0.0)}, objects=objects)
    (d / "scene_meta.json").write_text(json.dumps(meta))
    return G.StaticScene(d, model=m)


def box_body(name: str, half, z: float) -> str:
    return (f'<body name="{name}" pos="0 0 {z}"><freejoint/>'
            f'<geom type="box" size="{half[0]} {half[1]} {half[2]}"/></body>')  # fmt: skip


def obj(name: str, z: float, **extra) -> dict:
    return dict(body=name, surface="table", init_pos=[0, 0, z], init_quat=[1, 0, 0, 0], free=True, **extra)


class Graspability(unittest.TestCase):
    def test_jaw_constant_matches_the_runtime(self):
        from locomani.backends.scalebfm import runtime as R

        self.assertAlmostEqual(G.JAW_OPEN_M, 2 * (R.GRIP_OPEN - R.GRIP_CLOSED), places=6)

    def test_box_widths_follow_the_closing_axis(self):
        z = TOP + 0.10
        sc = scene(box_body("b", (0.03, 0.06, 0.10), z), {"B": obj("b", z)})
        w = G.section_widths(sc, "B", step_deg=45)
        self.assertAlmostEqual(w[0.0], 0.12, places=4)  # approach along x: the jaw closes along y
        self.assertAlmostEqual(w[90.0], 0.06, places=4)
        self.assertAlmostEqual(w[45.0], (0.06 + 0.12) * math.sqrt(0.5), places=4)
        f0, f90 = G.fit(sc, "B", 0.0), G.fit(sc, "B", math.pi / 2)
        self.assertEqual((f0["verdict"], f0["fits"]), ("too_wide", False))
        self.assertEqual((f90["verdict"], f90["fits"]), ("fits", True))
        self.assertAlmostEqual(f90["margin_m"], (G.JAW_OPEN_M - 0.06) / 2, places=4)

    def test_a_turned_object_turns_its_widths(self):
        z = TOP + 0.10
        sc = scene(box_body("b", (0.03, 0.06, 0.10), z), {"B": obj("b", z)})
        sc.place({"B": ([0, 0, z], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])})  # 90 deg about z
        self.assertAlmostEqual(G.fit(sc, "B", 0.0)["width_m"], 0.06, places=4)

    def test_tight_is_inside_the_insertion_error(self):
        z = TOP + 0.10
        sc = scene(box_body("b", (0.045, 0.045, 0.10), z), {"B": obj("b", z)})  # 9.0 cm: 0.25 cm a side
        self.assertEqual(G.fit(sc, "B", 0.0)["verdict"], "tight")

    def test_assess_rotates_the_approach_to_a_fit(self):
        z = TOP + 0.10
        sc = scene(box_body("b", (0.005, 0.0475, 0.10), z), {"B": obj("b", z)})  # a 1 x 9.5 cm board: fits from 34 deg
        a = G.assess(sc, "B", 0.0)
        self.assertFalse(a["side"]["fits"])
        self.assertIsNotNone(a["rotate"]["smallest_deg"])
        self.assertLessEqual(abs(a["rotate"]["smallest_deg"]), G.ROTATE_MAX_DEG)
        self.assertTrue(a["recommend"][0].startswith("rotate the approach"), a["recommend"])

    def test_nothing_fits_asks_for_a_prep_action(self):
        z = TOP + 0.10
        sc = scene(box_body("b", (0.06, 0.07, 0.10), z), {"B": obj("b", z)})
        a = G.assess(sc, "B", 0.0)
        self.assertTrue(any("nothing fits" in r for r in a["recommend"]), a["recommend"])

    def test_a_thin_walled_container_offers_a_rim_pinch(self):
        z = TOP + 0.046
        walls = "".join(
            f'<geom type="box" size="{sx} {sy} 0.045" pos="{px} {py} 0"/>'
            for sx, sy, px, py in ((0.005, 0.058, 0.04, 0), (0.005, 0.058, -0.04, 0), (0.035, 0.005, 0, 0.053),
                                   (0.035, 0.005, 0, -0.053))
        )  # fmt: skip
        body = f'<body name="mug" pos="0 0 {z}"><freejoint/>{walls}</body>'
        sc = scene(body, {"Mug": obj("mug", z, container=True)})
        a = G.assess(sc, "Mug", 0.0)
        self.assertTrue(a["rim"]["available"])
        self.assertAlmostEqual(a["rim"]["wall_m"], 0.01, places=4)
        self.assertTrue(any(r.startswith("rim pinch across the 1.0 cm wall") for r in a["recommend"]), a["recommend"])

    def test_a_flat_object_is_below_the_side_jaw_and_gets_a_top_pinch(self):
        z = TOP + 0.009
        sc = scene(box_body("s", (0.155, 0.034, 0.009), z), {"S": obj("s", z)})
        self.assertAlmostEqual(G.grasp_height(sc, "S"), TOP + G.JAW_ABOVE_TOP_M, places=6)
        self.assertEqual(G.fit(sc, "S", 0.0)["verdict"], "below_jaw")
        a = G.assess(sc, "S", 0.0)
        self.assertTrue(a["flat"])
        self.assertAlmostEqual(a["top_pinch"]["width_m"], 0.068, places=3)
        self.assertTrue(a["recommend"][0].startswith("top pinch"), a["recommend"])

    def test_grasp_height_is_the_centre_above_the_jaw_floor(self):
        z = TOP + 0.14
        sc = scene(box_body("b", (0.04, 0.04, 0.14), z), {"B": obj("b", z)})
        self.assertAlmostEqual(G.grasp_height(sc, "B"), z, places=6)
        self.assertGreater(len(G.section(sc, "B")), 0)
        self.assertEqual(len(G.section(sc, "B", z=TOP + 0.5)), 0)


if __name__ == "__main__":
    unittest.main()
