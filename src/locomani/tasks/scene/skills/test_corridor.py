"""The side grasp's corridor (pick.corridor_intrusion, pick.corridor_pass): how deep a free neighbour stands in the
hand's strip, and the plan taken instead when it stands deeper than CORRIDOR_DEPTH_M -- another edge first, then a
turned approach with the far hand, else the least entered one CORRIDOR_GAIN_M shallower, else the first plan. Imports
the module beside it, so it checks this copy of the package wherever it lives."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from . import pick


def yaw_quat(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


TOP = 0.4


class World:
    """The target X (a 5 x 5 x 10 cm box, origin at its bottom) at (0, 0) on the surface S; free neighbours added."""

    def __init__(self):
        box = dict(half_extent=[0.025, 0.025, 0.05], centre=[0.0, 0.0, 0.05])
        self.meta = dict(
            objects={"X": dict(surface="S", free=True, box=box, init_pos=[0.0, 0.0, TOP], init_quat=yaw_quat(0.0))},
            surfaces={"S": dict(xy=[0.0, 0.0], half_extent=[0.6, 0.45], yaw=0.0, top=TOP)},
        )
        self.occ = SimpleNamespace(body_radius=0.35)

    def add(self, key, xy, half=(0.05, 0.02, 0.1), yaw=0.0, z=TOP):
        box = dict(half_extent=list(half), centre=[0.0, 0.0, half[2]])
        self.meta["objects"][key] = dict(surface="S", free=True, box=box, init_pos=[*xy, z], init_quat=yaw_quat(yaw))

    def object_pose(self, key):
        o = self.meta["objects"][key]
        return np.asarray(o["init_pos"], float), np.asarray(o["init_quat"], float)


LOW = TOP + 0.005  # the hand's lowest point


class Intrusion(unittest.TestCase):
    def test_a_neighbour_behind_on_the_line_enters_the_whole_half_width(self):
        w = World()
        w.add("N", (-0.15, 0.0))  # approach +x: the hand comes from -x
        d, lat, k = pick.corridor_intrusion(w, "X", 0.0, LOW)
        self.assertEqual(k, "N")
        self.assertAlmostEqual(d, pick.STRIP_HALF_M)
        self.assertAlmostEqual(lat, 0.0)

    def test_the_depth_is_the_half_width_less_the_gap_to_the_line_signed_by_side(self):
        w = World()
        w.add("N", (-0.15, 0.07))  # box 2 cm deep across: its near face 5 cm left of the line
        d, lat, _ = pick.corridor_intrusion(w, "X", 0.0, LOW)
        self.assertAlmostEqual(d, pick.STRIP_HALF_M - 0.05)
        self.assertGreater(lat, 0.0)
        w = World()
        w.add("N", (-0.15, 0.08))  # its near face 6 cm off: outside the strip
        self.assertEqual(pick.corridor_intrusion(w, "X", 0.0, LOW), (0.0, 0.0, None))
        w = World()
        w.add("N", (-0.15, -0.05))
        d, lat, _ = pick.corridor_intrusion(w, "X", 0.0, LOW)
        self.assertAlmostEqual(d, pick.STRIP_HALF_M - 0.03)
        self.assertLess(lat, 0.0)

    def test_ahead_of_x_beyond_the_strip_below_the_hand_or_fixed_none(self):
        back = pick.STRIP_BACK_M + pick.grasp_program.PREGRASP_BACK
        for xy, half, free in (((0.12, 0.0), (0.05, 0.02, 0.1), True),  # ahead of X
                               ((-back - 0.06, 0.0), (0.05, 0.02, 0.1), True),  # behind the arm
                               ((-0.15, 0.0), (0.05, 0.02, 0.002), True),  # 4 mm tall, under the hand
                               ((-0.15, 0.0), (0.05, 0.02, 0.1), False)):  # fixed: strip_blocker's  # fmt: skip
            w = World()
            w.add("N", xy, half)
            w.meta["objects"]["N"]["free"] = free
            self.assertEqual(pick.corridor_intrusion(w, "X", 0.0, LOW), (0.0, 0.0, None), (xy, half, free))

    def test_a_turned_box_counts_by_its_footprint(self):
        w = World()
        w.add("N", (-0.2, 0.1), half=(0.1, 0.01, 0.1), yaw=np.radians(45.0))  # its corner reaches toward the line
        d, _, _ = pick.corridor_intrusion(w, "X", 0.0, LOW)
        self.assertAlmostEqual(d, pick.STRIP_HALF_M - (0.1 - (0.1 + 0.01) / np.sqrt(2)), places=6)


class Pass(unittest.TestCase):
    """corridor_pass over a mocked plan_approach: (face, depth, psi, stance, edge)."""

    def _plans(self, edges, turns):
        """plan_approach answering the other edges in order (`edges`, faces) and turns by (hand, yaw_off deg)."""

        def plan_approach(world, start, key, side, max_depth=None, yawed=False, yaw_off=None, top_tol=None,
                          exclude=()):  # fmt: skip
            if yaw_off is None:
                rest = [f for f in edges if all(abs(f - x) > 1e-6 for x in exclude)]
                return ((rest[0], 0.2, rest[0], np.zeros(2), np.zeros(2)), "") if rest else (None, "none")
            deg = round(float(np.degrees(yaw_off)), 6)
            ok = (side, deg) in turns
            return ((np.pi, 0.2, np.pi + yaw_off, np.zeros(2), np.zeros(2)), "") if ok else (None, "none")

        return plan_approach

    def _run(self, w, edges, turns, depth):
        straight = (np.pi, 0.2, np.pi, np.zeros(2), np.zeros(2))  # approach -x: the hand comes from +x

        def intrusion(world, key, psi, low):
            return depth(psi)

        with (
            mock.patch.object(pick, "plan_approach", side_effect=self._plans(edges, turns)),
            mock.patch.object(pick, "corridor_intrusion", side_effect=intrusion),
            mock.patch.object(pick, "nav_map", return_value=(None, None, [])),
        ):
            return pick.corridor_pass(w, np.array([2.0, 0.0]), "X", "right", straight)

    def test_a_clear_first_plan_stands(self):
        self.assertIsNone(self._run(World(), [0.5 * np.pi], [], lambda psi: (0.01, 0.05, "N")))

    def test_another_clear_edge_comes_before_a_turn(self):
        def depth(psi):
            return (0.05, 0.05, "N") if abs(psi - np.pi) < 1e-6 else (0.0, 0.0, None)

        out = self._run(World(), [0.5 * np.pi], [("right", 20.0)], depth)
        self.assertEqual((out[0][0], out[1]), (0.5 * np.pi, "right"))
        self.assertEqual(pick._corridor_note["corridor_pass"], "edge")

    def test_the_far_hand_turns_away_first_smallest_turn_that_clears(self):
        def depth(psi):  # the blocker on the approach's left (+): the right hand, psi + turn, clears from 35 deg
            off = np.degrees(pick.R.wrap(psi - np.pi))
            return (0.0, 0.0, None) if off >= 34.0 else (0.05, 0.05, "N")

        turns = {("right", 20.0), ("right", 35.0), ("right", 50.0), ("left", -20.0)}
        out = self._run(World(), [], turns, depth)
        self.assertEqual(out[1], "right")
        self.assertAlmostEqual(np.degrees(pick.R.wrap(out[0][2] - out[0][0])), 35.0)

    def test_none_clear_the_least_entered_only_when_gain_shallower(self):
        def depth(psi):
            off = round(float(np.degrees(pick.R.wrap(psi - np.pi))))
            return {0: (0.058, -0.05, "N"), -20: (0.030, 0.0, "M"), -35: (0.045, 0.0, "M")}.get(off, (0.058, 0, "N"))

        out = self._run(World(), [], {("left", -20.0), ("left", -35.0)}, depth)
        self.assertEqual(out[1], "left")
        self.assertAlmostEqual(np.degrees(pick.R.wrap(out[0][2] - out[0][0])), -20.0)
        with mock.patch.object(pick, "CORRIDOR_GAIN_M", 0.03):  # 2.8 cm shallower is not enough: the first stands
            self.assertIsNone(self._run(World(), [], {("left", -20.0), ("left", -35.0)}, depth))

    def test_no_turn_for_a_box_that_fits_one_way_or_whose_turned_jaw_does_not_straddle_it(self):
        w = World()
        w.meta["objects"]["X"]["box"]["half_extent"] = [0.025, 0.06, 0.05]  # 12 cm one way

        def depth(psi):
            return (0.05, 0.05, "N") if abs(psi - np.pi) < 1e-6 else (0.0, 0.0, None)

        self.assertIsNone(self._run(w, [], {("right", 20.0)}, depth))
        w = World()
        w.meta["objects"]["X"]["box"]["half_extent"] = [0.045, 0.045, 0.05]  # 9 cm square: 12.7 across at 45 deg
        out = self._run(w, [], {("right", 20.0), ("right", 50.0)}, depth)
        self.assertIsNone(out)  # 20 deg: 11.5 cm, 50 deg: 12.6 cm, both past JAW_SPAN_M

    def test_a_top_down_object_takes_no_corridor_pass(self):
        w = World()
        w.meta["objects"]["X"]["box"]["half_extent"] = [0.01, 0.1, 0.01]  # flat and long
        self.assertIsNone(self._run(w, [0.5 * np.pi], [], lambda psi: (0.05, 0.05, "N")))


if __name__ == "__main__":
    unittest.main()
