"""TODO 49.4: the seed skill package of LocoManiLab: loaded by path (two copies side by side, nothing on sys.path), the parameters of
the old package, the cache key, the task table against the benchmark's prompts, the facade a package gets, the guard on the seed."""

import ast
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from locomani.bench.locomanilab import client, families, run
from locomani.bench.locomanilab.config import SEED_DIR, LocoManiLabConfig

OLD = Path("/workspace/code/locomanipulation-esc/scripts/experiments/lml_bed3_20261121/seed_pkg_lml_drawer")
TASKS_YAML = Path(LocoManiLabConfig().lml_root) / "src/locomanilab/benchmark/tasks.yaml"
GUARD = Path(run.__file__).parent / "guard.py"


def literal(path: Path, name: str):
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n.value
    raise KeyError(name)


class Loading(unittest.TestCase):
    def test_two_copies_load_side_by_side_and_nothing_goes_on_the_path(self):
        before = list(sys.path)
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            shutil.copytree(SEED_DIR, a, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(SEED_DIR, b, ignore=shutil.ignore_patterns("__pycache__"))
            (b / "skill.py").write_text((b / "skill.py").read_text().replace("start_hold=1.0", "start_hold=2.5"))
            pa, pb = run.load_package(a), run.load_package(b)
            self.assertNotEqual(pa.__name__, pb.__name__)
            self.assertEqual(pa.parameters("")["start_hold"], 1.0)
            self.assertEqual(pb.parameters("")["start_hold"], 2.5)
            self.assertEqual(pa.path, a.resolve())
        self.assertEqual(sys.path, before)

    def test_a_failed_load_leaves_no_module_behind(self):
        with tempfile.TemporaryDirectory() as t:
            c = Path(t) / "c"
            shutil.copytree(SEED_DIR, c, ignore=shutil.ignore_patterns("__pycache__"))
            (c / "flows.py").write_text("import nothing_like_this\n")
            n = len([k for k in sys.modules if k.startswith("_lml_skill_")])
            with self.assertRaises(ImportError):
                run.load_package(c)
            self.assertEqual(len([k for k in sys.modules if k.startswith("_lml_skill_")]), n)
            (c / "flows.py").unlink()
            with self.assertRaises(FileNotFoundError):
                run.load_package(c)

    def test_the_flows_the_bed_dispatches_exist(self):
        pkg = run.load_package(SEED_DIR)
        for name in run.FLOW_NAMES.values():
            self.assertTrue(callable(getattr(pkg.flows, name)), name)


class Parameters(unittest.TestCase):
    def test_apply_set_rejects_unknown_names_and_bed_switches_are_not_parameters(self):
        pkg = run.load_package(SEED_DIR)
        with self.assertRaises(KeyError):
            pkg.apply_set(dict(lift_speeed=0.3))
        for k in run.BED_SWITCHES:
            self.assertNotIn(k, pkg.parameters(""))
        self.assertEqual(pkg.apply_set(dict(lift=0.4))["lift"], 0.4)

    def test_the_seed_parameters_are_the_old_packages(self):
        """K_ORIG and K_CUT are literally the old skill_simple.py's, and the package's K for an LML task is K_CUT + EXTRA."""
        if not OLD.exists():
            self.skipTest("old package not on this machine")
        pkg = run.load_package(SEED_DIR)
        for name in ("K_ORIG", "K_CUT", "MOTOR_KEYS"):
            old = ast.dump(literal(OLD / "skill_simple.py", name))
            new = ast.dump(literal(SEED_DIR / "skill.py", name))
            self.assertEqual(old, new, name)
        self.assertEqual(
            ast.dump(literal(OLD / "__init__.py", "EXTRA")), ast.dump(literal(SEED_DIR / "__init__.py", "EXTRA"))
        )
        K = pkg.parameters("IThorKitchenAppleIntoBox_scene13")
        self.assertEqual(
            (K["lift"], K["nav_gap"], K["start_hold"], K["lift_speed"], K["walk_hold"]), (0.34, 0.30, 1.0, 0.1, 0.0)
        )
        self.assertEqual(K, pkg.parameters(""))

    def test_the_seed_choose_grasp_is_the_old_one(self):
        if not OLD.exists():
            self.skipTest("old package not on this machine")

        def fn(path, name):
            return next(
                n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name
            )

        for name in ("choose_grasp", "grasp_desc", "one_step"):
            self.assertEqual(
                ast.dump(fn(OLD / "skill_simple.py", name)), ast.dump(fn(SEED_DIR / "skill.py", name)), name
            )
        for name in ("Map", "PlanGrid", "MapGrid", "near_fixture", "polyline_cut", "clear_heading"):
            a = next(
                n for n in ast.parse((OLD / "skill_lml_cam.py").read_text()).body if getattr(n, "name", None) == name
            )
            b = next(n for n in ast.parse((SEED_DIR / "flows.py").read_text()).body if getattr(n, "name", None) == name)
            self.assertEqual(ast.dump(a), ast.dump(b), name)


class CacheKey(unittest.TestCase):
    def test_key_follows_the_package_files_and_switches_only(self):
        pkg = run.load_package(SEED_DIR)
        k0 = run.cache_key(pkg)
        self.assertEqual(k0, run.cache_key(pkg))
        self.assertEqual(len(k0), 10)
        self.assertNotEqual(k0, run.cache_key(pkg, ("perception", "rgbd", "realsense")))
        self.assertNotEqual(k0, run.cache_key(pkg, control=("truth", "odometry")))
        with tempfile.TemporaryDirectory() as t:
            for f in run.PACKAGE_FILES:
                c = Path(t) / f.replace(".py", "")
                shutil.copytree(SEED_DIR, c, ignore=shutil.ignore_patterns("__pycache__"))
                self.assertEqual(run.cache_key(run.load_package(c)), k0)  # a copy keys alike
                (c / f).write_text((c / f).read_text() + "\n# a change\n")
                self.assertNotEqual(run.cache_key(run.load_package(c)), k0, f)


class Calls(unittest.TestCase):
    def test_every_sam_call_names_the_service(self):
        """Regression (found by the fridge episode of the deterministic check): `articulate` called Sam.find(rgb, names) after Sam took
        SAM 3's address as its first argument. Every call of Sam.find / Sam.best in the package passes a `sam3` first."""
        for f in ("camera_view.py", "flows.py"):
            tree = ast.parse((SEED_DIR / f).read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("find", "best"):
                    if (
                        isinstance(n.func.value, ast.Name)
                        and n.func.value.id == "Sam"
                        or (isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "Sam")
                    ):
                        first = ast.unparse(n.args[0])
                        self.assertIn("sam3", first, f"{f}:{n.lineno} {ast.unparse(n)}")

    def test_flows_run_on_a_fake_episode_without_a_simulator(self):
        """articulate on recorded-like inputs: the look for the handle goes through env.sam3 (a fake service answers)."""
        import contextlib
        import types
        from unittest import mock

        pkg = run.load_package(SEED_DIR)
        H, W = 12, 16
        rgb, pts, ok = np.zeros((H, W, 3), np.uint8), np.zeros((H, W, 3)), np.ones((H, W), bool)
        bel = types.SimpleNamespace(scene=lambda: (rgb, pts, ok), pelvis=lambda: (np.zeros(3), 0.0))
        row = {}
        env = types.SimpleNamespace(want=dict(approach=False, handle="fridge door handle"), row=row, bel=bel, sam3=("h", 1),
                                    clock=types.SimpleNamespace(phase=lambda n: contextlib.nullcontext()), K={})  # fmt: skip
        with mock.patch(
            "locomani.perception.head.sam_results", lambda sam3, img, phrases: {p: [] for p in phrases}
        ) as m:
            pkg.flows.articulate(env)
        self.assertEqual(row["art_look"]["found"], False)


class Table(unittest.TestCase):
    def test_every_task_is_in_the_benchmark_and_its_prompt_parses_to_the_table_kind(self):
        if not TASKS_YAML.exists():
            self.skipTest("LocoManiLab not on this machine")
        rows = yaml.safe_load(TASKS_YAML.read_text())["tasks"]
        pkg = run.load_package(SEED_DIR)
        for task, info in families.TASKS.items():
            self.assertIn(task, rows)
            want = pkg.camera_view.parse(rows[task]["prompt"])
            self.assertEqual(want["kind"], info.kind, task)
            looks = want["kind"] in ("pick", "place") or (want["kind"] == "articulate" and bool(want.get("obj")))
            self.assertEqual(info.looks, looks, task)

    def test_the_specs_file_has_every_task_with_a_judge_family(self):
        import json

        f = Path(LocoManiLabConfig().specs_file)
        if not f.exists():
            self.skipTest("specs not on this machine")
        specs = json.loads(f.read_text())
        from locomani.bench.locomanilab import comply

        for task in families.TASKS:
            self.assertIn(specs[task]["family"], comply.NAV + comply.LIFT + comply.PLACE + comply.ART, task)


class Facade(unittest.TestCase):
    def test_a_package_env_holds_no_simulator(self):
        class Fake:
            pass

        sim = Fake()
        sim.fingers_cmd = {"right": np.zeros(7)}
        env = client.Env("t", dict(target="x"), sim, Fake(), Fake(), dict(a=1), None, spec=dict(family="lift"))
        env.clock = Fake()
        env.want = dict(kind="pick")
        env.view = "v"
        f = env.facade()
        names = set(vars(f))
        self.assertLessEqual(
            names, {"_env", "K", "stance", "row", "want", "sam3", "cloud_dir", "bel", "mo", "hand", "clock"}
        )
        for k in ("sim", "info", "spec", "task", "marks", "bodies", "rec", "art"):
            self.assertFalse(hasattr(f, k), k)
        self.assertIs(f.hand.cmd, sim.fingers_cmd)
        self.assertFalse(hasattr(f.hand, "sim"))
        self.assertEqual(f.view, "v")

    def test_flow_done_reaches_the_recorder_and_the_view_proxy_records_the_grasp(self):
        calls = []

        class Rec:
            def after_approach(self):
                calls.append("approach")

            def after_pick(self):
                calls.append("pick")

        env = client.Env("t", {}, None, None, None, {}, None)
        env.rec = Rec()
        env.flow_done("approach")
        env.flow_done("pick")
        self.assertEqual(calls, ["approach", "pick"])

        class View:
            s = 1

            def grasp(self, tgt, yaw):
                return ("g", tgt, yaw)

        seen = []
        v = client.GraspRecorded(View(), lambda tgt, g: seen.append((tgt, g)))
        self.assertEqual(v.grasp("T", 0.5), ("g", "T", 0.5))
        self.assertEqual(seen, [("T", ("g", "T", 0.5))])
        self.assertEqual(v.s, 1)
        v.s = 2
        self.assertEqual(v._view.s, 2)


class Guard(unittest.TestCase):
    def py(self, *args):
        return subprocess.run([sys.executable, "-I", str(GUARD), *args], capture_output=True, text=True)

    def test_selftest_and_the_seed_against_itself(self):
        r = self.py("--selftest")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "selftest ok"), r.stdout)
        r = self.py(str(SEED_DIR), str(SEED_DIR))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_a_package_may_not_import_the_bed_the_server_or_the_truth_side(self):
        for imp in (
            "from locomani.bench.locomanilab import run",
            "from locomani.bench.locomanilab.server import loop_wall",
            "import locomani.bench.locomanilab.truth",
            "from locomani.bench.locomanilab import truth as T",
            "from locomani.bench.locomanilab.client import Env",
        ):
            with tempfile.TemporaryDirectory() as t:
                c = Path(t) / "c"
                shutil.copytree(SEED_DIR, c, ignore=shutil.ignore_patterns("__pycache__"))
                (c / "flows.py").write_text((c / "flows.py").read_text() + "\n" + imp + "\n")
                r = self.py(str(SEED_DIR), str(c))
                self.assertEqual(r.returncode, 1, imp)
                self.assertIn("R8", r.stdout, imp)

    def test_the_package_never_names_the_simulator(self):
        for f in ("skill.py", "pick.py", "camera_view.py", "flows.py"):
            tree = ast.parse((SEED_DIR / f).read_text())
            attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            self.assertNotIn("sim", attrs | names, f)


if __name__ == "__main__":
    unittest.main()
