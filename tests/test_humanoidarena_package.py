"""TODO 49.5b: the skill package loads by path, its parameters are overridden by name, the seed flow runs on a fake facade with simple
kinematics, the guard refuses reaches for the simulator, the truth view reads the goal from the privileged facts, the capability table
says exactly what the bed implements."""

import importlib
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomani.bench import options, registry
from locomani.bench.humanoidarena import client as C
from locomani.bench.humanoidarena import config as cfgmod
from locomani.bench.humanoidarena import guard, overlay, run
from locomani.bench.humanoidarena.truth import TruthView, UnsupportedTask
from locomani.bench.spec import READY

SEED = cfgmod.SEED_DIR


class Loading(unittest.TestCase):
    def test_loads_by_path_twice_without_clashing(self):
        a, b = run.load_package(SEED), run.load_package(SEED)
        self.assertIsNot(a, b)
        self.assertEqual(a.parameters("pp_box"), b.parameters("pp_box"))

    def test_a_missing_file_is_named(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "__init__.py").write_text("")
            with self.assertRaises(FileNotFoundError) as e:
                run.load_package(t)
            self.assertIn("skill.py", str(e.exception))

    def test_apply_set_and_unknown_names(self):
        pkg = run.load_package(SEED)
        self.assertEqual(pkg.apply_set({"slow_s": 2.0}, "pp_box")["slow_s"], 2.0)
        with self.assertRaises(KeyError):
            pkg.apply_set({"slow_ss": 1}, "pp_box")

    def test_package_hash_changes_with_the_source(self):
        with tempfile.TemporaryDirectory() as t:
            shutil.copytree(SEED, t, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
            h1 = run.package_hash(run.load_package(t))
            (Path(t) / "skill.py").write_text((Path(t) / "skill.py").read_text() + "\n# x\n")
            self.assertNotEqual(h1, run.package_hash(run.load_package(t)))

    def test_the_skill_names_a_task_it_has_no_flow_for(self):
        pkg = run.load_package(SEED)
        env = type("E", (), dict(task="double_desk"))()
        with self.assertRaises(NotImplementedError):
            pkg.run(env)


class Geometry(unittest.TestCase):
    def test_rot6d_is_the_first_two_columns(self):
        self.assertTrue(np.allclose(C.rot6d_of_yaw(0.0), [1, 0, 0, 1, 0, 0]))
        c, s = np.cos(0.5), np.sin(0.5)
        self.assertTrue(np.allclose(C.rot6d_of_yaw(0.5), [c, -s, s, c, 0, 0]))

    def test_yaw_of(self):
        self.assertAlmostEqual(C.yaw_of([np.cos(0.4), 0, 0, np.sin(0.4)]), 0.8, places=6)

    def test_overlay_projection(self):
        K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
        T = np.eye(4)
        u, v, z = overlay.project(K, T, [0.1, 0.2, 2.0])
        self.assertEqual((round(u, 3), round(v, 3), z), (55.0, 50.0, 2.0))
        self.assertIsNone(overlay.project(K, T, [0, 0, -1.0]))


class FakeMe:
    A = {"left": np.array([[0.30, 0, 0, 0.12, 0, 0, 0], [0, -0.28, 0, 0, 0, 0, 0], [-0.4, 0, 0, -0.1, 0, 0, 0]]),
         "right": np.array([[0.30, 0, 0, 0.12, 0, 0, 0], [0, 0.28, 0, 0, 0, 0, 0], [-0.4, 0, 0, -0.1, 0, 0, 0]])}  # fmt: skip
    OFF = {"left": np.array([0.08, 0.29, -0.04]), "right": np.array([0.08, -0.29, -0.04])}

    def __init__(self, env):
        self.env = env
        self.ids = {"left": list(range(11, 28, 2))[:7], "right": [i + 1 for i in list(range(11, 28, 2))[:7]]}

    def pelvis(self):
        return np.array([*self.env.xy, 0.79]), self.env.yaw

    def joints(self):
        return self.env.q.copy()

    def arm_ids(self, side):
        return self.ids[side]

    def _rot(self):
        c, s = np.cos(self.env.yaw), np.sin(self.env.yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def wrist(self, side):
        dq = self.env.q[self.ids[side]] - self.env.q0[self.ids[side]]
        return np.array([*self.env.xy, 0.79]) + self._rot() @ (self.OFF[side] + self.A[side] @ dq)

    def jac(self, side):
        return self._rot() @ self.A[side], self.wrist(side)


class FakeEnv:
    """Perfect tracking: the pelvis is where the reference puts it; the wrists follow the joint targets linearly."""

    def __init__(self, K):
        self.K, self.task, self.instruction, self.start_height = K, "pp_box", "carry", 0.8
        self.xy, self.yaw = np.array([-1.56, -4.99]), 0.0
        self.q = np.zeros(29)
        self.q0 = self.q.copy()
        self.steps, self.marks, self.actions = 0, [], []
        self.me = FakeMe(self)
        self.t = 0.0
        self.leg_ids = [0, 1, 3, 4, 6, 7, 9, 10]
        self.gait = type("G", (), dict(period=1.3, speed=0.8, pose=lambda s, ph: np.zeros(29)))()
        self.view = type("V", (), dict(goal=lambda s: GOAL))()

    def hold(self, joints=None, dxy=(0.0, 0.0), yaw=0.0, height=None, hands=(0.0, 0.0)):
        q = self.q if joints is None else joints
        return np.concatenate(
            [dxy, [self.start_height if height is None else height], C.rot6d_of_yaw(yaw), q, hands]
        ).astype(np.float32)

    def drive(self, a):
        assert a.shape == (40,) and np.isfinite(a).all()
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.xy = self.xy + np.array([[c, -s], [s, c]]) @ a[:2]
        self.yaw = float(np.arctan2(a[5], a[3]))  # rot6d [c, -s, s, c, 0, 0] sits at a[3:9]
        self.q = a[9:38].astype(float)
        self.actions.append(a)
        self.steps += 1
        self.t = self.steps * 0.02
        if self.steps > 9000:
            raise C.Done()

    def mark(self, name, note=None):
        self.marks.append(name)


GOAL = dict(kind="pp_box", box=np.array([-0.51, -4.52, 0.745]), box_center=np.array([-0.51, -4.52, 0.85]), yaw=0.0, half=0.105,
            place_xy=np.array([-1.51, -2.81]), place_bottom_z=0.648, place_rect=np.array([-2.54, -0.49, -3.29, -2.34]),
            table=np.array([[-0.64, -6.24, 0.0], [0.12, -3.76, 0.75]]), shelf=np.array([[-2.54, -3.29, 0.0], [-0.49, -2.34, 2.18]]))  # fmt: skip


class Flow(unittest.TestCase):
    def test_the_seed_flow_runs_all_its_phases_on_a_fake_facade(self):
        pkg = run.load_package(SEED)
        env = FakeEnv(pkg.apply_set({"arm_step": 0.03}, "pp_box"))
        env.q0 = env.q.copy()
        out = pkg.run(env)
        self.assertEqual(
            env.marks,
            [
                "settled",
                "ready",
                "at_table",
                "reached",
                "squeezed",
                "lifted",
                "at_shelf",
                "lowered",
                "released",
            ],
        )
        self.assertLess(out["reach_err"], 0.035)  # the servo stops within servo_tol (0.03)
        # at the shelf the box would ride in front of the pelvis: the pelvis ended in front of the shelf, facing +y
        self.assertGreater(env.yaw, 1.2)
        self.assertLess(env.xy[1], -3.0)
        self.assertTrue(all(a.shape == (40,) for a in env.actions))

    def test_the_arms_are_raised_before_the_last_leg_of_the_walk_and_standing_holds_the_heading(self):
        pkg = run.load_package(SEED)
        env = FakeEnv(pkg.apply_set({"arm_step": 0.03}, "pp_box"))
        pkg.run(env)
        # the ready mark comes before the table is near: the pelvis is still more than the lead-in short of the stand
        self.assertLess(env.marks.index("ready"), env.marks.index("at_table"))
        ctl = importlib.import_module(pkg.__name__ + ".skill").Ctl(env)
        ctl.hold_yaw = (
            ctl.yaw0 + 0.3
        )  # the pelvis stands at 0 and is to be held at +0.3 rad: the reference heading turns toward it
        ctl.stand_still(20)
        self.assertGreater(ctl.yaw, 0.0)

    def test_the_hands_close_on_the_squeeze_and_open_at_the_end(self):
        pkg = run.load_package(SEED)
        env = FakeEnv(pkg.apply_set({"arm_step": 0.03}, "pp_box"))
        pkg.run(env)
        hands = np.array([a[38:40] for a in env.actions])
        self.assertTrue((hands.max(axis=0) == 1.0).all())
        self.assertTrue((hands[-1] == 0.0).all())


class Truth(unittest.TestCase):
    class Priv:
        def __init__(self, accepted):
            self.accepted = accepted

        def episode(self):
            return dict(task="pp_box", instruction="x")

        def objects(self):
            return {"box": [-0.5, -4.5, 0.745, 1, 0, 0, 0]}

        def facts(self):
            return dict(
                box_half=[0.105] * 3,
                box_center=[-0.5, -4.5, 0.85],
                table=[[-0.64, -6.2, 0], [0.1, -3.7, 0.75]],
                shelf=None,
                accepted=self.accepted,
            )

    def spot(self, cx, cy, z=0.648, w=2.0):
        return dict(
            surface=0,
            center_xy=[cx, cy],
            rect=[cx - w / 2, cx + w / 2, cy - 0.45, cy + 0.45],
            bottom_z=z,
            center_z=z + 0.105,
        )

    def test_goal_is_the_nearest_accepted_spot_with_room(self):
        far, near, tiny = self.spot(5.0, 5.0), self.spot(-1.5, -2.8), self.spot(-0.6, -4.4, w=0.2)
        g = TruthView(self.Priv([far, tiny, near])).goal()
        self.assertEqual(g["kind"], "pp_box")
        self.assertTrue(np.allclose(g["place_xy"], [-1.5, -2.8]))
        self.assertAlmostEqual(g["place_bottom_z"], 0.648)
        self.assertAlmostEqual(g["yaw"], 0.0)
        self.assertIsNone(g["shelf"])

    def test_no_spot_is_refused(self):
        with self.assertRaises(UnsupportedTask):
            TruthView(self.Priv([self.spot(0, 0, z=2.1)])).goal()

    def test_other_tasks_have_no_goal(self):
        p = self.Priv([])
        p.episode = lambda: dict(task="open_door", instruction="x")
        with self.assertRaises(UnsupportedTask):
            TruthView(p).goal()


class Guard(unittest.TestCase):
    def test_selftest(self):
        self.assertEqual(guard.selftest(), "selftest ok")

    def test_the_seed_passes_its_own_guard(self):
        self.assertTrue(guard.check(SEED, SEED)["ok"], guard.check(SEED, SEED))

    def edited(self, extra):
        t = tempfile.mkdtemp()
        shutil.copytree(SEED, t, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        (Path(t) / "skill.py").write_text((SEED / "skill.py").read_text() + "\n" + extra)
        return guard.check(SEED, Path(t))

    def test_forbidden_reaches(self):
        for line in (
            "x = env.seed",
            "import torch",
            "import isaaclab.sim",
            "y = env._ep",
            "z = getattr(env, 'me')",
            "from locomani.bench.humanoidarena import server",
        ):
            self.assertFalse(self.edited(line)["ok"], line)

    def test_a_clean_edit_passes(self):
        self.assertTrue(self.edited("K['slow_s'] = 2.0")["ok"])

    def test_the_seed_package_names_nothing_forbidden(self):
        import ast

        tree = ast.parse((SEED / "skill.py").read_text())
        self.assertEqual(guard.forbidden_names(tree), [])


class Table(unittest.TestCase):
    def test_ready_values_are_the_beds_scope(self):
        spec = registry.spec("humanoidarena")
        ready = {(a, v) for a, v, s in spec.table() if s == READY}
        want = {
            ("target", "truth"),
            ("sensor", "rgbd"),
            ("depth_noise", "none"),
            ("self_pose", "truth"),
            ("controller_root", "truth"),
            ("controller", "native"),
        }
        self.assertEqual(ready, want)

    def test_planned_values_stay_planned_with_a_reason(self):
        spec = registry.spec("humanoidarena")
        for ax, v in (("target", "perception"), ("depth_noise", "realsense")):
            self.assertEqual(spec.status(ax, v), "planned")
            self.assertIn(f"{ax}={v}", spec.notes)

    def test_the_profile_resolves_to_truth_native(self):
        r = options.resolve("humanoidarena", [], allow_planned=False)
        o = r.options
        self.assertEqual(
            (o.target, o.sensor, o.self_pose, o.controller_root, o.controller),
            ("truth", "none", "truth", "truth", "native"),
        )

    def test_check_scope_names_what_is_not_wired(self):
        r = options.resolve("humanoidarena", ["target=perception", "sensor=rgbd"], allow_planned=True)
        with self.assertRaises(cfgmod.ScopeError) as e:
            cfgmod.check_scope(r.options)
        self.assertIn("perception", str(e.exception))


if __name__ == "__main__":
    unittest.main()
