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


def bibox_scene():
    """The logistics carton: 0.45 x 0.32 x 0.33 m, origin at its bottom centre, wider than a jaw either way."""
    body = (
        f'<body name="b" pos="0 0 {TOP}"><freejoint/><geom type="box" size="0.225 0.16 0.165" pos="0 0 0.165"/></body>'
    )
    return scene(
        body, {"B": obj("b", TOP, bimanual=True, box=dict(half_extent=[0.225, 0.16, 0.165], centre=[0, 0, 0.165]))}
    )


def bitrace(left_on=True, right_on=True, squeeze=True, rise_cm=0.0, drop=False, push_cm=0.0):
    """hold, walk, a both-hand reach, squeeze, posture (the lift): the hands end on the box's x faces (or short)."""
    n0, n_walk, n_reach, n_sq, n_after = 50, 150, 150, 50, 200
    T = n0 + n_walk + n_reach + n_sq + n_after
    t = np.arange(T) * DT
    prims = np.array(["hold", "posture", "reach", "squeeze", "walk_to"])
    p = np.zeros(T, int)
    p[n0 : n0 + n_walk] = 4
    p[n0 + n_walk : n0 + n_walk + n_reach] = 2
    hi = n0 + n_walk + n_reach
    if squeeze:
        p[hi : hi + n_sq] = 3
    p[hi + n_sq :] = 1
    obj = np.zeros((T, 7))
    obj[:, :3] = [0.5, 0.0, TOP]
    obj[:, 3] = 1.0
    if push_cm:
        obj[hi - 30 :, 1] += push_cm / 100
    if rise_cm:
        obj[hi + n_sq :, 2] += rise_cm / 100
        if drop:
            obj[hi + n_sq + 100 :, 2] = TOP
    wr = np.zeros((T, 2, 3))
    z = TOP + 0.2
    # the jaw centre is OFF ahead of the wrist towards the object: a wrist 0.1515 m outside a face puts the jaw on it
    gap = {"left": 0.0 if left_on else 0.10, "right": 0.0 if right_on else 0.10}
    for j, (s, sign) in enumerate((("left", -1), ("right", 1))):
        wr[:, j] = [0.5 + sign * (0.225 + 0.1515 + gap[s] + 0.30), 0.0, z]
        wr[n0 + n_walk : hi, j, 0] = np.linspace(wr[0, j, 0], 0.5 + sign * (0.225 + 0.1515 + gap[s]), n_reach)
        wr[hi:, j, 0] = 0.5 + sign * (0.225 + 0.1515 + gap[s])
    if rise_cm:
        wr[hi + n_sq :, :, 2] += rise_cm / 100
    jaw = np.full((T, 2), FC.GRIP_OPEN)
    fingers = np.full((T, 4), FC.GRIP_OPEN)
    robot = np.zeros((T, 7))
    robot[:, 3] = 1.0
    return dict(t=t, primitive=p.astype(np.int16), primitives=prims, jaw=jaw, fingers=fingers, wrists=wr,
                ref_wrists=wr.copy(), robot=robot, joints=np.zeros((T, 29)), mode=np.zeros(T), ref_pelvis=robot[:, :3],
                obj_b=obj)  # fmt: skip


class Bimanual(unittest.TestCase):
    """TODO 38.11: the two-hand classes on a bimanual object."""

    def test_a_bimanual_object_is_recognised_and_a_single_hand_one_is_not(self):
        self.assertTrue(FC.bimanual(bibox_scene(), "B"))
        self.assertFalse(FC.bimanual(box_scene(), "B"))
        self.assertFalse(FC.bimanual(None, "B"))

    def test_both_hands_on_no_rise_is_no_lift(self):
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0)]), bitrace(), bibox_scene(), "B")
        self.assertEqual(cls, "no_lift")
        self.assertEqual(ev["hold_t"], 7.0)
        self.assertLess(max(ev["hand_box_cm"].values()), FC.HAND_ON_CM)
        self.assertEqual(ev["obj_rise_max_cm"], 0.0)

    def test_one_hand_short_and_both_short(self):
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0)]), bitrace(left_on=False), bibox_scene(), "B")
        self.assertEqual(cls, "one_hand_short")
        self.assertGreater(ev["hand_box_cm"]["left"], FC.HAND_ON_CM)
        self.assertLess(ev["hand_box_cm"]["right"], FC.HAND_ON_CM)
        cls, _ = FC.classify(rec(steps=[WALK, reach(1.0)]), bitrace(left_on=False, right_on=False), bibox_scene(), "B")
        self.assertEqual(cls, "side_reach")

    def test_rose_then_lost_is_slip_and_pushed_before_the_hold_is_knocked(self):
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0)]), bitrace(rise_cm=8.0, drop=True), bibox_scene(), "B")
        self.assertEqual(cls, "slip")
        self.assertEqual(ev["obj_rise_max_cm"], 8.0)
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0)]), bitrace(push_cm=5.0), bibox_scene(), "B")
        self.assertEqual(cls, "knocked")
        self.assertEqual(ev["obj_moved_before_hold_cm"], 5.0)

    def test_the_strict_goal_classes(self):
        # TODO 38.19: rose 8 cm and stayed (bitrace rise without drop): too low for lifted(.., 0.20, ..)
        low = rec(why="[goal] lifted(B, 0.2, 30.0): lowest corner 8.0 cm above table (need 20)")
        self.assertEqual(FC.classify(low, bitrace(rise_cm=8.0), bibox_scene(), "B")[0], "low_lift")
        tilt = rec(why="[goal] lifted(B, 0.2, 30.0): tilted 41 deg (limit 30)")
        self.assertEqual(FC.classify(tilt, bitrace(rise_cm=8.0), bibox_scene(), "B")[0], "tilted")
        judge = dict(ok=False, ever_ok=True, first_fail_t=12.0, first_fail_why="[goal] held(B): no hand touches it")
        dropped = rec(why="[judge] ...", judge=judge)
        cls, ev = FC.classify(dropped, bitrace(rise_cm=8.0), bibox_scene(), "B")
        self.assertEqual(cls, "dropped_hold")
        self.assertEqual(ev["judge"]["first_fail_t"], 12.0)

    def test_the_common_classes_come_first(self):
        self.assertEqual(FC.classify(rec(ok=True), bitrace(), bibox_scene(), "B")[0], "ok")
        self.assertEqual(FC.classify(rec(fell=True, steps=[WALK]), bitrace(), bibox_scene(), "B")[0], "fell_walk")
        self.assertEqual(
            FC.classify(rec(fell=True, steps=[WALK, reach(1.0)]), bitrace(), bibox_scene(), "B")[0], "fell"
        )
        walk_off = dict(WALK, stop_err_cm=9.0)
        self.assertEqual(FC.classify(rec(steps=[walk_off, reach(1.0)]), bitrace(), bibox_scene(), "B")[0], "walk_off")
        # no trace: the steps alone say whether the reach got there
        self.assertEqual(FC.classify(rec(steps=[WALK, reach(20.0)]), None, bibox_scene(), "B")[0], "side_reach")


RIM = TOP + 0.33
CARTON = dict(half_extent=[0.225, 0.16, 0.165], centre=[0, 0, 0.165], wall=0.01)
PACK_GOAL = ["in(X, C)", "held(C)", "lifted(C, 0.2, 60)"]


def pack_scene():
    """The carton C at (0.5, 0) on the support and a 6 x 6 x 10 cm object X beside it (box centre at its origin)."""
    bodies = (f'<body name="c" pos="0.5 0 {TOP}"><freejoint/><geom type="box" size="0.225 0.16 0.165" pos="0 0 0.165"/>'
              f"</body>" + box_body("x", (0.03, 0.03, 0.05), TOP + 0.05))  # fmt: skip
    objs = {"C": obj("c", TOP, bimanual=True, container=True, box=CARTON),
            "X": obj("x", TOP + 0.05, box=dict(half_extent=[0.03, 0.03, 0.05], centre=[0, 0, 0]))}  # fmt: skip
    return scene(bodies, objs)


def packtrace(keys):
    """20 s of hold / walk / reach with X's box centre moving through `keys` [(t, x, y, z)]; C stays put."""
    T = 1000
    t = np.arange(T) * DT
    prims = np.array(["hold", "reach", "walk_to"])
    p = np.zeros(T, int)
    p[50:200] = 2
    p[200:300] = 1
    k = np.asarray([(0.0, 0.5, 0.5, TOP + 0.05), *keys], float)
    x = np.zeros((T, 7))
    x[:, 3] = 1.0
    for j in range(3):
        x[:, j] = np.interp(t, k[:, 0], k[:, j + 1])
    c = np.zeros((T, 7))
    c[:, :3] = [0.5, 0.0, TOP]
    c[:, 3] = 1.0
    wr = np.zeros((T, 2, 3))
    wr[:, :, 2] = TOP + 0.2
    robot = np.zeros((T, 7))
    robot[:, 3] = 1.0
    return dict(t=t, primitive=p.astype(np.int16), primitives=prims, jaw=np.full((T, 2), FC.GRIP_OPEN),
                fingers=np.full((T, 4), FC.GRIP_OPEN), wrists=wr, ref_wrists=wr.copy(), robot=robot,
                joints=np.zeros((T, 29)), mode=np.zeros(T), ref_pelvis=robot[:, :3], obj_x=x, obj_c=c)  # fmt: skip


class Pack(unittest.TestCase):
    """gso_pack.md D14: the put-in's classes on a pack goal, from X's and the carton's trace poses."""

    def classify(self, keys, **kw):
        return FC.classify(rec(steps=[WALK, reach(1.0)], **kw), packtrace(keys), pack_scene(), "X", goal=PACK_GOAL)

    def test_the_container_comes_from_an_in_goal_naming_the_object(self):
        sc = pack_scene()
        self.assertEqual(FC.container_of(PACK_GOAL, "X", sc), "C")
        self.assertIsNone(FC.container_of(PACK_GOAL, "C", sc))
        self.assertIsNone(FC.container_of(["held(X)"], "X", sc))
        self.assertIsNone(FC.container_of(None, "X", sc))

    def test_never_picked_keeps_the_picks_class(self):
        cls, ev = self.classify(
            [(5.0, 0.5, 0.5, TOP + 0.05), (8.0, 0.5, 0.5, TOP + 0.09), (10.0, 0.5, 0.5, TOP + 0.05)]
        )
        self.assertEqual(cls, "pick_failed")
        self.assertEqual(ev["x_rise_max_cm"], 4.0)
        self.assertIn("pick_class", ev)

    def test_lost_below_the_rim_or_far_from_the_opening_is_carry_dropped(self):
        low = [(5.0, 0.5, 0.5, TOP + 0.05), (8.0, 0.5, 0.5, TOP + 0.20), (10.0, 0.5, 0.5, TOP + 0.05)]
        cls, ev = self.classify(low)
        self.assertEqual(cls, "carry_dropped")
        self.assertIsNone(ev["x_high_t"])
        far = [(5.0, 0.5, 0.5, TOP + 0.05), (8.0, 0.5, 0.9, TOP + 0.45), (10.0, 0.5, 0.9, 0.05)]
        cls, ev = self.classify(far)
        self.assertEqual(cls, "carry_dropped")
        self.assertGreater(ev["x_over_opening_cm"], FC.NEAR_OPENING_CM)

    def test_let_go_beside_the_opening_is_missed_opening(self):
        cls, ev = self.classify(
            [(5.0, 0.5, 0.5, TOP + 0.05), (8.0, 0.5, 0.25, TOP + 0.45), (10.0, 0.5, 0.25, TOP + 0.05)]
        )
        self.assertEqual(cls, "missed_opening")
        self.assertAlmostEqual(ev["x_over_opening_cm"], 10.0, delta=0.5)
        self.assertFalse(ev["x_in_end"])

    def test_resting_on_the_rim_over_the_opening_is_stuck_on_rim(self):
        cls, ev = self.classify(
            [(5.0, 0.5, 0.5, TOP + 0.05), (8.0, 0.5, 0.0, TOP + 0.45), (10.0, 0.5, 0.0, RIM + 0.05)]
        )
        self.assertEqual(cls, "stuck_on_rim")
        self.assertEqual(ev["x_over_opening_cm"], 0.0)
        self.assertTrue(ev["x_high_end"])

    def test_in_the_carton_and_still_failing_is_in_but_not_lifted_with_the_lifts_class(self):
        keys = [(5.0, 0.5, 0.5, TOP + 0.05), (8.0, 0.5, 0.0, TOP + 0.45), (10.0, 0.5, 0.0, TOP + 0.06)]
        cls, ev = self.classify(keys, why="[goal] held(C): no hand touches it")
        self.assertEqual(cls, "in_but_not_lifted")
        self.assertTrue(ev["x_in_end"])
        self.assertIn(ev["lift_class"], ("side_reach", "one_hand_short", "knocked", "no_lift", "slip", "other"))
        self.assertAlmostEqual(ev["x_high_t"], 8.4, delta=0.1)

    def test_without_a_pack_goal_the_objects_own_classes_stay(self):
        keys = [(5.0, 0.5, 0.5, TOP + 0.05), (8.0, 0.5, 0.5, TOP + 0.09), (10.0, 0.5, 0.5, TOP + 0.05)]
        cls, ev = FC.classify(rec(steps=[WALK, reach(1.0)]), packtrace(keys), pack_scene(), "X")
        self.assertNotIn(cls, ("pick_failed", "carry_dropped", "missed_opening", "stuck_on_rim", "in_but_not_lifted"))
        self.assertNotIn("pick_class", ev)
        self.assertEqual(FC.classify(rec(ok=True), packtrace(keys), pack_scene(), "X", goal=PACK_GOAL)[0], "ok")


if __name__ == "__main__":
    unittest.main()


class WhyZh(unittest.TestCase):
    """The page videos' caption: the grip and lift checks' failure strings read in Chinese (they showed in English on
    the grasp_atomic pages, 2026-10-23)."""

    def test_the_grip_and_lift_strings_have_a_chinese_reading(self):
        import re

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsi" / "report"))
        import visualize as V

        for why in ("nothing between the fingers", "not between the fingers after the lift"):
            self.assertIsNone(re.search("[A-Za-z]{3}", V.why_zh(why)), why)
