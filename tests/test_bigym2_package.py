"""TODO 49.5: the seed package loads by path, takes parameters by name, runs against a fake facade; its guard refuses what a skill may not
touch; the truth view reads goals from a fake Privileged; the capability table agrees with the bed's scope."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomani.bench import options, registry
from locomani.bench.bigym2 import config as cfgmod
from locomani.bench.bigym2 import guard, run
from locomani.bench.bigym2.truth import TruthView, UnsupportedTask
from locomani.bench.spec import READY

SEED = cfgmod.SEED_DIR


class Package(unittest.TestCase):
    def test_loads_by_path_twice_without_clashing(self):
        a, b = run.load_package(SEED), run.load_package(SEED)
        self.assertNotEqual(a.__name__, b.__name__)
        self.assertEqual(a.parameters("drawer_top_open"), b.parameters("drawer_top_open"))

    def test_a_missing_file_is_named(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "__init__.py").write_text("")
            with self.assertRaises(FileNotFoundError):
                run.load_package(t)

    def test_apply_set_and_unknown_names(self):
        pkg = run.load_package(SEED)
        self.assertEqual(pkg.apply_set({"walk_speed": 0.1})["walk_speed"], 0.1)
        with self.assertRaises(KeyError):
            pkg.apply_set({"walk_spede": 0.1})

    def test_package_hash_changes_with_the_source(self):
        with tempfile.TemporaryDirectory() as t:
            shutil.copytree(SEED, t, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
            h1 = run.package_hash(run.load_package(t))
            (Path(t) / "skill.py").write_text((Path(t) / "skill.py").read_text() + "\n# x\n")
            self.assertNotEqual(h1, run.package_hash(run.load_package(t)))

    def test_geometry_helpers(self):
        pkg = run.load_package(SEED)
        S = pkg.skill
        R = S.facing((1, 0, 0))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(R[:, 0], [1, 0, 0])
        np.testing.assert_allclose(R[:, 1], [0, 0, 1])
        q = S.quat_of(R)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0)
        np.testing.assert_allclose(S.quat_of(np.eye(3)), [1, 0, 0, 0], atol=1e-12)

    def test_run_refuses_a_task_it_has_no_flow_for(self):
        pkg = run.load_package(SEED)

        class Env:
            K, task = pkg.parameters(), "pick_box"

        with self.assertRaises(ValueError):
            pkg.run(Env())

    def test_reach_flow_on_a_fake_facade(self):
        """The reach flow against a robot whose hand follows the joint targets exactly: it stands, reaches and holds, and never
        reads anything but the facade."""
        pkg = run.load_package(SEED)
        point = np.array([0.4, 0.0, 0.9])

        class Me:
            def __init__(s):
                s.pos = np.zeros(3)
                s.hand_pos = np.array([0.3, 0.1, 0.8])

            def pelvis(s):
                return s.pos.copy(), 0.0

            def hand(s, side):
                return s.hand_pos.copy()

            def arm(s, side):
                return np.zeros(7)

            def ik(s, side, pos, quat=None, q0=None):
                err = float(np.linalg.norm(pos - (s.pos + np.array([0.0, 0.1, 0.4]))) > 0.5) * 0.1
                return np.ones(7), err

        class View:
            def goal(s):
                return dict(kind="reach", side="left", point=point)

        class Env:
            K, task, instruction = pkg.parameters(), "reach_target_single", "touch"
            view, me, t = View(), Me(), 0.0

            def __init__(s):
                s.steps, s.marks = 0, []

            def drive(s, vx=0.0, vy=0.0, wz=0.0, height=None, pitch=None, left=None, right=None, grip=None):
                s.steps += 1
                s.me.pos = s.me.pos + np.array([vx * 0.02, 0, 0])
                if left is not None:
                    s.me.hand_pos = point.copy()  # the hand arrives as soon as an arm target is given

            def mark(s, name):
                s.marks.append(name)

        env = Env()
        info = pkg.run(env)
        self.assertEqual(env.marks, ["start", "stand", "reached"])
        self.assertLess(info["hand_err"], 0.01)


class Guard(unittest.TestCase):
    def test_selftest(self):
        self.assertEqual(guard.selftest(), "selftest ok")

    def test_the_seed_passes_its_own_guard(self):
        self.assertTrue(guard.check(SEED, SEED)["ok"])

    def with_skill(self, extra_line):
        t = tempfile.mkdtemp()
        shutil.copytree(SEED, t, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        (Path(t) / "skill.py").write_text((SEED / "skill.py").read_text() + "\n" + extra_line + "\n")
        self.addCleanup(shutil.rmtree, t, True)
        return guard.check(SEED, t)

    def test_forbidden_reaches(self):
        for line, rule in [
            ("x = env.seed", "R4"),
            ("x = env._ep", "R4"),
            ("import mujoco", "R4"),
            ("from locomani.bench.bigym2 import client", "R8"),
            ("from locomani.bench.bigym2.privileged import BiGymPrivileged", "R8"),
            ("y = getattr(env, 'me')", "R6"),
            ("import requests_toolbelt", "R5"),
        ]:
            res = self.with_skill(line)
            self.assertFalse(res["ok"], line)
            self.assertTrue(any(v.startswith(rule) for v in res["violations"]), (line, res["violations"]))

    def test_a_clean_edit_passes(self):
        self.assertTrue(self.with_skill("K['walk_speed'] = 0.15")["ok"])

    def test_extra_files_and_a_changed_init_are_refused(self):
        t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, t, True)
        shutil.copytree(SEED, t, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        (Path(t) / "extra.py").write_text("x = 1\n")
        (Path(t) / "__init__.py").write_text("# changed\n")
        res = guard.check(SEED, t)
        self.assertFalse(res["ok"])
        self.assertTrue(any("not allowed" in v for v in res["violations"]))
        self.assertTrue(any("frozen" in v for v in res["violations"]))

    def test_the_seed_package_names_nothing_forbidden(self):
        import ast

        self.assertEqual(guard.forbidden_names(ast.parse((SEED / "skill.py").read_text())), [])


class FakePriv:
    def __init__(self, task, facts, joints, poses):
        self.task, self._f, self._j, self._p = task, facts, joints, poses

    def episode(self):
        return dict(task=self.task)

    def facts(self):
        return self._f

    def joints(self):
        return self._j

    def body_pose(self, name):
        return self._p[name]


class Truth(unittest.TestCase):
    def test_reach_goal(self):
        p = FakePriv("reach_target_single", dict(targets=[dict(pos=[0.5, -0.1, 0.9], hands=["left"])]), {}, {})
        g = TruthView(p).goal()
        self.assertEqual((g["kind"], g["side"]), ("reach", "left"))
        np.testing.assert_allclose(g["point"], [0.5, -0.1, 0.9])

    def test_drawer_goal_takes_the_top_drawer_and_its_handle(self):
        def pose(z):
            T = np.eye(4)
            T[2, 3] = z
            return T

        joints = {
            "cab/drawer_big_1": dict(type=2, body="cab/big", range=[0, 0.38], axis_world=[-1, 0, 0], q=0.0),
            "cab/drawer_small_4": dict(type=2, body="cab/small4", range=[0, 0.38], axis_world=[-1, 0, 0], q=0.03),
            "cab/door": dict(type=3, body="cab/door", range=[0, 1.5], axis_world=[0, 0, 1], q=0.0),
        }
        facts = dict(
            drawers=[0.0, 0.0, 0.07],
            handles={"cab/small4": [dict(body="b52", lo=[0.58, -0.06, 0.72], hi=[0.61, 0.06, 0.73])]},
        )
        p = FakePriv(
            "drawer_top_open", facts, joints, {"cab/big": pose(0.15), "cab/small4": pose(0.66), "cab/door": pose(0.4)}
        )
        g = TruthView(p).goal()
        self.assertEqual((g["kind"], g["name"]), ("drawer", "cab/drawer_small_4"))
        np.testing.assert_allclose(g["handle"], [0.595, 0.0, 0.725])
        np.testing.assert_allclose(g["bar_axis"], [0, 1, 0])
        np.testing.assert_allclose(g["pull"], [-1, 0, 0])
        self.assertAlmostEqual(g["travel"], 0.38)
        self.assertAlmostEqual(g["open_frac"], 0.07)

    def test_other_tasks_have_no_goal(self):
        with self.assertRaises(UnsupportedTask):
            TruthView(FakePriv("pick_box", {}, {}, {})).goal()


class Switches(unittest.TestCase):
    def resolve(self, *sets, allow_planned=False):
        return options.resolve("bigym2", list(sets), allow_planned=allow_planned)

    def test_profile_is_truth_native(self):
        o = self.resolve().options
        self.assertEqual(
            (o.target, o.sensor, o.self_pose, o.controller_root, o.controller),
            ("truth", "none", "truth", "truth", "native"),
        )

    def test_ready_values_are_the_beds_scope(self):
        spec = registry.spec("bigym2")
        for axis, values in spec.supports.items():
            ready = {v for v, s in values.items() if s == READY}
            scope = set(cfgmod.SCOPE[axis]) - ({"none"} if axis in ("sensor", "depth_noise") else set())
            if axis == "sensor":
                scope = {"rgbd"}  # the channel is ready, a truth run reads none of it (target=truth takes no sensor)
            if axis == "depth_noise":
                scope = {"none"}
            self.assertEqual(ready, scope, axis)

    def test_planned_values_stay_planned_with_a_reason(self):
        spec = registry.spec("bigym2")
        for axis, value in [("target", "perception"), ("self_pose", "odometry"), ("depth_noise", "realsense")]:
            self.assertEqual(spec.status(axis, value), "planned")
            self.assertIn(f"{axis}={value}", spec.notes)
        self.assertIsNone(spec.status("sensor", "stereo"))
        self.assertIsNone(spec.status("controller", "scalebfm"))
        self.assertIn("sensor=stereo", spec.notes)

    def test_planned_values_are_refused_unless_allowed(self):
        for sets in (["target=perception", "sensor=rgbd"], ["self_pose=odometry"], ["controller_root=odometry"]):
            with self.assertRaises(ValueError, msg=sets):
                self.resolve(*sets)
        self.resolve("target=perception", "sensor=rgbd", allow_planned=True)

    def test_unsupported_values_are_refused_with_their_reason(self):
        with self.assertRaises(ValueError) as cm:
            self.resolve("controller=scalebfm")
        self.assertIn("GR00T-WBC", str(cm.exception))
        with self.assertRaises(ValueError):
            self.resolve("target=perception", "sensor=stereo", allow_planned=True)

    def test_check_scope_names_what_is_not_wired(self):
        o = self.resolve("target=perception", "sensor=rgbd", allow_planned=True).options
        with self.assertRaises(cfgmod.ScopeError) as cm:
            cfgmod.check_scope(o)
        self.assertIn("target=perception", str(cm.exception))
        cfgmod.check_scope(self.resolve().options)


if __name__ == "__main__":
    unittest.main()
