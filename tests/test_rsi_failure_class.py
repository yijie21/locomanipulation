"""TODO 33.15: the round failure classifier (scripts/rsi/failure_class.py) on synthetic traces and step records."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsi"))

import failure_class as FC  # noqa: E402
from test_rsi_graspability import TOP, box_body, obj, scene  # noqa: E402

DT = 0.02
Z = TOP + 0.10


def box_scene():
    return scene(box_body("b", (0.03, 0.06, 0.10), Z), {"B": obj("b", Z)})


def trace(n_walk=150, n_reach=150, n_after=150, grip=True, opening=0.0, move_cm=0.0, move_at=None):
    """hold 1 s, walk, reach, (grip, hold): 50 Hz arrays as Motor.arrays writes them."""
    n0 = 50
    T = n0 + n_walk + n_reach + n_after
    t = np.arange(T) * DT
    prims = np.array(["grip", "hold", "reach", "walk_to"])
    p = np.full(T, 1)
    p[n0 : n0 + n_walk] = 3
    p[n0 + n_walk : n0 + n_walk + n_reach] = 2
    gi = n0 + n_walk + n_reach
    if grip:
        p[gi : gi + 50] = 0
    jaw = np.full((T, 2), FC.GRIP_OPEN)
    fingers = np.full((T, 4), FC.GRIP_OPEN)
    if grip:
        jaw[gi:, 1] = -0.03
        fingers[gi:, 2:] = opening / 2 - 0.023
    obj = np.zeros((T, 7))
    obj[:, :3] = [0.5, 0.0, Z]
    obj[:, 3] = 1.0
    if move_cm:
        k = move_at if move_at is not None else gi - 40
        obj[k:, 0] += move_cm / 100
    wr = np.zeros((T, 2, 3))
    wr[:, 1] = [0.2, 0.0, Z]
    wr[n0 + n_walk : gi, 1, 0] = np.linspace(0.2, 0.40, n_reach)
    wr[gi:, 1, 0] = 0.40
    ref = wr.copy()
    robot = np.zeros((T, 7))
    robot[:, 3] = 1.0
    return dict(t=t, primitive=p.astype(np.int16), primitives=prims, jaw=jaw, fingers=fingers, wrists=wr,
                ref_wrists=ref, robot=robot, joints=np.zeros((T, 29)), mode=np.zeros(T), ref_pelvis=robot[:, :3],
                obj_b=obj)  # fmt: skip


def rec(ok=False, fell=False, why="[goal] held(B): no hand touches it", steps=None, **kw):
    out = dict(ok=ok, fell=fell, why=why, plan_why="", error="", steps=steps or [], t_end=10.0)
    out.update(kw)
    return out


WALK = dict(step="walk", ok=True, t0=1.0, t1=4.0, stop_err_cm=1.5)


def reach(err, t0=4.0):
    return dict(step="pregrasp", ok=err <= 5, t0=t0, t1=t0 + 3.0, jaw_err_cm=err, attitude_err_deg=2.0)


class FailureClass(unittest.TestCase):
    def test_ok_no_plan_time_limit(self):
        self.assertEqual(FC.classify(rec(ok=True), None, None, None)[0], "ok")
        self.assertEqual(FC.classify(rec(why="no skill covers this goal"), None, None, None)[0], "no_plan")
        self.assertEqual(FC.classify(rec(why="episode time limit"), None, None, None)[0], "time_limit")

    def test_falls_split_by_the_first_reach(self):
        walk_fall = rec(fell=True, steps=[dict(step="settle", t0=0, t1=1), dict(step="walk", ok=False, t0=1, t1=6)])
        self.assertEqual(FC.classify(walk_fall, None, None, None)[0], "fell_walk")
        late = rec(fell=True, steps=[WALK, reach(1.0)], t_end=20.0)
        self.assertEqual(FC.classify(late, None, None, None)[0], "fell")

    def test_walk_off_and_reach_misses(self):
        off = rec(steps=[dict(WALK, stop_err_cm=12.0), reach(1.0)])
        self.assertEqual(FC.classify(off, None, None, None)[0], "walk_off")
        side = rec(steps=[WALK, reach(15.0), dict(step="grip", ok=False, t0=7.0, t1=8.0, opening_cm=0.01, held=[])])
        self.assertEqual(FC.classify(side, None, None, None)[0], "side_reach")
        pinch = rec(steps=[dict(step="stance", flat=True, t0=1, t1=1), WALK, reach(12.0),
                           dict(step="grip", ok=False, t0=7.0, t1=8.0, opening_cm=0.01, held=[])])  # fmt: skip
        self.assertEqual(FC.classify(pinch, None, None, None)[0], "pinch_reach")

    def test_too_wide_from_the_approach_used(self):
        sc = box_scene()
        steps = [WALK, dict(step="geom", t0=4, t1=4, approach_deg=0.0), reach(0.8),
                 dict(step="grip", ok=False, t0=7.0, t1=8.0, opening_cm=0.01, held=[])]  # fmt: skip
        cls, ev = FC.classify(rec(steps=steps), None, sc, "B")
        self.assertEqual(cls, "too_wide")
        self.assertEqual(ev["fit"]["verdict"], "too_wide")
        steps[1] = dict(
            step="geom", t0=4, t1=4, approach_deg=90.0
        )  # the 6 cm side: fits, so the empty jaw is a near miss
        self.assertEqual(FC.classify(rec(steps=steps), None, sc, "B")[0], "miss_near")

    def test_trace_events(self):
        z = trace(opening=0.0)
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0, t0=4.0)]), z, box_scene(), "B")
        self.assertEqual(ev["source"], "trace")
        self.assertAlmostEqual(ev["grip_t"], 350 * DT, places=6)
        self.assertAlmostEqual(ev["walk_end_t"], 199 * DT, places=6)
        self.assertAlmostEqual(ev["last_reach_end_t"], 349 * DT, places=6)
        self.assertEqual(ev["grip_side"], "right")
        self.assertLess(ev["opening_at_grip_cm"], FC.EMPTY_CM)
        self.assertAlmostEqual(ev["reach_track_ratio"], 1.0, places=3)
        self.assertIsNone(ev["obj_first_move_t"])
        self.assertEqual(cls, "miss_near")

    def test_knocked_before_the_grip(self):
        z = trace(opening=0.0, move_cm=5.0)
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0)]), z, box_scene(), "B")
        self.assertEqual(cls, "knocked")
        self.assertGreater(ev["obj_moved_before_grip_cm"], 4.0)
        self.assertEqual(ev["obj_first_move_by"], "right_hand")

    def test_slip_after_a_grip(self):
        z = trace(opening=0.06)
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0)]), z, box_scene(), "B")
        self.assertEqual(cls, "slip")
        self.assertAlmostEqual(ev["opening_at_grip_cm"], 6.0, places=2)

    def test_unknown_object_still_classifies(self):
        self.assertEqual(FC.classify(rec(steps=[WALK, reach(20.0)]), None, None, None)[0], "side_reach")


if __name__ == "__main__":
    unittest.main()
