"""TODO 49.2: the SIMPLE bed's judge (judge.py: comply_simple's cases ported from its --selftest, and loss_of)."""

import unittest

from locomani.bench.simple import judge as J


class Comply(unittest.TestCase):
    def test_the_old_selftest_cases(self):
        J.selftest()  # the pick rules, the two placing tasks, the four articulated tasks

    def test_pick_needs_a_held_run_of_two_tenths(self):
        w = J._tl("+R:10")
        self.assertEqual(J.verdict("XMovePickTeleop", True, w.state()), dict(ok=True, why=""))
        self.assertEqual(J.verdict("XMovePickTeleop", True, J._tl("+R:9").state())["why"], "short")

    def test_numpy_only(self):
        import ast
        import pathlib

        src = pathlib.Path(J.__file__).read_text()
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import)}
        mods |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertEqual(mods - {"numpy"}, set(), "the environment server loads judge.py in SIMPLE's venv")


class Loss(unittest.TestCase):
    base = dict(episode_id="XMovePickTeleop/Q/0", limit_s=16.0)

    def test_classes(self):
        r = dict(self.base)
        self.assertIsNone(J.loss_of(dict(r, t_success=3.0)))
        self.assertEqual(J.loss_of(dict(r, t_success=20.0)), "timeout")
        self.assertEqual(J.loss_of(dict(r, error="x")), "error")
        self.assertEqual(J.loss_of(dict(r, fell=1.0)), "fall")
        self.assertEqual(J.loss_of(dict(r, why="no target seen")), "perceive")
        self.assertEqual(J.loss_of(dict(r, why="no grasp")), "deliver")
        marks = dict(before_close=dict(moved_cm=5.0, tilt_deg=0.0), after_lift=dict(held=False, rise_cm=0.0))
        self.assertEqual(J.loss_of(dict(r, marks=marks)), "collide")
        marks["before_close"]["moved_cm"] = 0.0
        self.assertEqual(J.loss_of(dict(r, marks=marks, palm_err_cm=3.0)), "deliver")
        self.assertEqual(J.loss_of(dict(r, marks=marks, palm_err_cm=1.0, reach_ok=[True])), "grasp")

    def test_articulated(self):
        art = dict(sign=1, best_deg=1.0, start_deg=0.0)
        r = dict(episode_id="OpenOvenTeleop/Q/0", limit_s=20.0, art=art)
        self.assertEqual(J.loss_of(dict(r, art_touch={"t_body": 1.0})), "collide")
        self.assertEqual(J.loss_of(dict(r, art_touch={"t_hand": None})), "deliver")
        self.assertEqual(J.loss_of(dict(r, art_touch={"t_hand": 1.0})), "grasp")
        self.assertEqual(J.loss_of(dict(r, art=dict(art, best_deg=10.0), art_touch={"t_hand": 1.0})), "short")


if __name__ == "__main__":
    unittest.main()
