"""The skills catalog's own tests (src/locomani/tasks/scene/skills/; split from test_scene_task_runner.py on
2026-10-20 so a sandbox started from another seed can drop them, TODO 31): the pick chooses an edge from which a side
of the object fits the jaw; the ported hand program's laws; `apply_set` reaches modules and dict entries."""

import unittest

import numpy as np

from locomani.tasks.scene.skills.pick import JAW_SPAN_M, facings, grasp_yaw


def yaw_quat(y):
    return [float(np.cos(y / 2)), 0.0, 0.0, float(np.sin(y / 2))]


TABLE = dict(xy=[0.0, 0.0], half_extent=[0.4, 0.3], flat_half_extent=[0.4, 0.3], yaw=0.0, top=0.7,
             approach_edges=["+x", "-y"], room="r", of_object="Table")  # fmt: skip


class Pick(unittest.TestCase):
    def test_pick_faces_an_edge_whose_approach_closes_across_a_fitting_side(self):
        f = facings(TABLE, [0.3, 0.2])
        self.assertEqual(len(f), 2)
        self.assertLessEqual(f[0][1], f[1][1])
        # 8 x 12 cm can lying along x: only an approach along x (from the +x edge, facing -x) closes across 8 cm
        half = [0.06, 0.04, 0.09]
        self.assertIsNone(grasp_yaw(yaw_quat(0.0), half, -np.pi / 2))  # from the -y edge, facing +y
        self.assertAlmostEqual(abs(grasp_yaw(yaw_quat(0.0), half, np.pi)), np.pi)
        self.assertEqual(grasp_yaw(yaw_quat(0.3), [JAW_SPAN_M / 2] * 3, 1.0), 1.0)  # both sides fit: face it


class GraspProgram(unittest.TestCase):
    """The ported hand program's laws (skills/grasp_program.py = bfm_grasp's)."""

    def test_entry_command_stops_past_the_grip_by_the_overshoot(self):
        from locomani.tasks.scene.skills.grasp_program import entry_captured, entry_command

        d = np.array([1.0, 0.0, 0.0])
        cmd, grip = np.zeros(3), np.array([0.05, 0.0, 0.0])
        for _ in range(200):
            cmd = entry_command(cmd, grip, d, 0.06, 0.02, 0.02)
        np.testing.assert_allclose(cmd, [0.07, 0.0, 0.0])
        self.assertTrue(entry_captured(np.array([0.045, 0, 0]), grip, d, 0.01))
        self.assertFalse(entry_captured(np.array([0.035, 0, 0]), grip, d, 0.01))

    def test_tracking_keeps_its_origin_and_moves_at_most_its_speed(self):
        from locomani.tasks.scene.skills.grasp_program import TRACKING, Tracking

        tr = Tracking()
        first = tr.step(np.array([0.1, 0, 0]), np.zeros(3), 0.02)
        np.testing.assert_allclose(first, [0.1, 0, 0])  # the first error is the origin, kept
        prev = first.copy()
        for _ in range(50):  # the hand now lags 3 cm more: the feedback follows, no faster than its speed
            fb = tr.step(np.array([0.1, 0, 0]), np.array([-0.03, 0, 0]), 0.02)
            self.assertLessEqual(np.linalg.norm(fb - prev), TRACKING["speed"] * 0.02 + 1e-12)
            prev = fb.copy()
        self.assertGreater(prev[0], 0.1)

    def test_span_across_the_jaw(self):
        from locomani.tasks.scene.skills.pick import span

        half = [0.06, 0.04, 0.09]
        self.assertAlmostEqual(span(yaw_quat(0.0), half, 0.0), 0.08)  # approach along x: closes across y (8 cm)
        self.assertAlmostEqual(span(yaw_quat(0.0), half, np.pi / 2), 0.12)


class SkillParameters(unittest.TestCase):
    def test_apply_set_reaches_modules_and_dict_entries_and_refuses_typos(self):
        import importlib

        import locomani.tasks.scene.skills as skills

        try:
            done = skills.apply_set(skills, {"PLAN_HELD": "pick_simple", "pick.STAND_AHEAD_M": 0.38,
                                             "grasp_program.ENTRY_GATE.lateral_m": 0.008})  # fmt: skip
            self.assertEqual(len(done), 3)
            self.assertEqual(skills.PLAN_HELD, "pick_simple")
            self.assertEqual(skills.pick.STAND_AHEAD_M, 0.38)
            self.assertEqual(skills.grasp_program.ENTRY_GATE["lateral_m"], 0.008)
            for bad in ("pick.STAND_AHEAD", "grasp_program.ENTRY_GATE.lateral", "NOPE"):
                with self.assertRaises(KeyError):
                    skills.apply_set(skills, {bad: 1})
            self.assertIn("grasp_program.DASH_SPEED", skills.parameters(skills))
        finally:
            for m in (skills.grasp_program, skills.pick, skills):
                importlib.reload(m)


if __name__ == "__main__":
    unittest.main()
