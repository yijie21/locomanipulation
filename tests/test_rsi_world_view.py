"""Steps 0 and 1 of the vision RSI design (docs/handoff/vision-rsi-design-20261023.md): a skills package that sets
WORLD_VIEW reads the scene only through locomani.tasks.scene.view -- no simulator handle, model arrays or contact readers
in its source, the robot view refusing them at run time -- and the ported incumbent keeps the original's parameters;
the L1 PerturbedView errs on the free objects' poses as its Perturbation says, repeatably."""

import importlib.util
import io
import re
import sys
import tokenize
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "scripts" / "rsi" / "seeds"
sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene.view import Perturbation, PerturbedView, RobotView, TruthView  # noqa: E402

# what a view package's code may not name: the simulator handle, MuJoCo model/data arrays, the motor's contact readers,
# MuJoCo itself beyond its math helpers (mju_*), private attributes of the views
FORBIDDEN = [
    r"\.sim\b",
    r"\.(xpos|xquat|xmat|qpos|qvel|ncon|contact)\b",
    r"\.(geom|mesh|body|jnt|site)_(?!radius\b)[a-z]+\b",
    r"\.(between_fingers|gripping|touching)\b",
    r"\bmujoco\.(?!mju_)\w+",
    r"\b(world|w|ctx\.world|ctx\.motor|m)\._\w+",
]


def code_only(path: Path) -> str:
    """The file's source without comments and string literals (docstrings may name what the code must not touch)."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            out.append(" ")
        elif tok.type in (tokenize.NL, tokenize.NEWLINE):
            out.append("\n")
        else:
            out.append(tok.string + " " if tok.type == tokenize.NAME else tok.string)
    return "".join(out)


def view_packages() -> list[Path]:
    return [d for d in sorted(SEEDS.iterdir()) if (d / "__init__.py").exists() and re.search(
        r"^WORLD_VIEW\s*=\s*True", (d / "__init__.py").read_text(), re.M)]  # fmt: skip


def load(root: Path, name: str):
    for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class ViewPackageSource(unittest.TestCase):
    def test_the_ported_incumbent_is_a_view_package(self):
        self.assertIn(SEEDS / "grasp_c11_view", view_packages())

    def test_view_packages_name_no_simulator_state(self):
        for pkg in view_packages():
            for f in sorted(pkg.rglob("*.py")):
                src = code_only(f)
                for pat in FORBIDDEN:
                    hits = [m.group(0) for m in re.finditer(pat, src)]
                    with self.subTest(file=str(f.relative_to(REPO)), pattern=pat):
                        self.assertEqual(hits, [])

    def test_the_scan_catches_what_the_original_reads(self):
        # the unported incumbent reads the simulator: the patterns must find it there
        src = "\n".join(code_only(f) for f in sorted((SEEDS / "grasp_c11").rglob("*.py")))
        self.assertTrue(any(re.search(p, src) for p in FORBIDDEN[:4]))


class Views(unittest.TestCase):
    def motor(self):
        return SimpleNamespace(sim="SIM", pol="POL", gripping=lambda b: {"right"} if b == 7 else set(), reach=lambda: "r",
                               t=1.5, fell_at=None)  # fmt: skip

    def test_robot_view_refuses_the_simulator_and_contacts(self):
        rv = RobotView(self.motor())
        for name in ("sim", "pol", "gripping", "touching", "between_fingers", "place", "reset"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                getattr(rv, name)
        with self.assertRaises(AttributeError):
            rv.sim = None

    def test_robot_view_passes_commands_and_state_through(self):
        m = self.motor()
        rv = RobotView(m)
        self.assertEqual(rv.reach(), "r")
        self.assertEqual(rv.t, 1.5)
        rv.primitive = "hand_entry"  # skills set motor fields; the write reaches the motor
        self.assertEqual(m.primitive, "hand_entry")

    def test_truth_view_answers_from_the_world_without_exposing_it(self):
        pose = (np.zeros(3), np.array([1.0, 0, 0, 0]))
        meta = {"objects": {}}
        world = SimpleNamespace(motor=self.motor(), meta=meta, free={"mug": 7, "spoon": 9}, occ="OCC", nav="NAV",
                                object_pose=lambda k: pose, sim="SIM")  # fmt: skip
        v = TruthView(world)
        self.assertIs(v.meta, meta)  # the same object each time: skills key their caches on it
        self.assertEqual(v.free, ("mug", "spoon"))
        self.assertEqual((v.occ, v.nav), ("OCC", "NAV"))
        self.assertIs(v.object_pose("mug"), pose)
        self.assertEqual(v.held("mug"), {"right"})
        self.assertEqual(v.held("spoon"), set())
        self.assertIsInstance(v.motor, RobotView)
        for name in ("sim", "_floor_statics", "free_bodies"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                getattr(v, name)
        with self.assertRaises(AttributeError):
            v.cache = 1


def _scene_batch():
    """scripts/rsi/batch.py under its own name (the warehouse loop has a batch.py too), sys.modules left as it was."""
    names = ("gate", "batch", "family", "scene_round")
    saved = {k: sys.modules.pop(k, None) for k in names}
    here = str(REPO / "scripts" / "rsi")
    sys.path.insert(0, here)
    try:
        return load_module("scene_rsi_batch_view", REPO / "scripts" / "rsi" / "batch.py")
    finally:
        sys.path.remove(here)
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Perturbed(unittest.TestCase):
    """L1: errors on the free objects' poses only, repeatable per round and condition."""

    def world(self, trace=None, t=0.0):
        quat = np.array([np.cos(0.15), 0.0, 0.0, np.sin(0.15)])  # yaw 0.3 rad
        poses = {
            "mug": (np.array([1.0, 2.0, 0.8]), quat),
            "table": (np.array([1.0, 2.0, 0.0]), np.array([1.0, 0, 0, 0])),
        }
        motor = SimpleNamespace(trace=trace if trace is not None else [], t=t, gripping=lambda b: set())
        meta = {"objects": {"mug": {"body": "mug_b", "free": True}, "table": {}}}
        return SimpleNamespace(motor=motor, meta=meta, free={"mug": 7}, occ=None, nav=None,
                               object_pose=lambda k: tuple(np.copy(a) for a in poses[k]))  # fmt: skip

    def test_no_perturbation_is_the_truth(self):
        w = self.world()
        p, t = PerturbedView(w, Perturbation(), 610001), TruthView(w)
        for k in ("mug", "table"):
            for a, b in zip(p.object_pose(k), t.object_pose(k)):
                np.testing.assert_array_equal(a, b)

    def test_the_bias_has_its_size_and_stays_for_the_round(self):
        w = self.world()
        truth = w.object_pose("mug")[0]
        v = PerturbedView(w, Perturbation(obj_bias_m=0.02), 610001)
        p1, p2 = v.object_pose("mug")[0], v.object_pose("mug")[0]
        self.assertAlmostEqual(float(np.linalg.norm(p1 - truth)), 0.02, places=12)
        np.testing.assert_array_equal(p1, p2)
        again = PerturbedView(w, Perturbation(obj_bias_m=0.02), 610001).object_pose("mug")[0]
        np.testing.assert_array_equal(p1, again)  # the same round and condition repeat
        other = PerturbedView(w, Perturbation(obj_bias_m=0.02), 610002).object_pose("mug")[0]
        self.assertFalse(np.allclose(p1, other))
        np.testing.assert_array_equal(v.object_pose("table")[0], w.object_pose("table")[0])  # the map is the truth

    def test_the_heading_error_turns_about_the_vertical(self):
        w = self.world()
        q0 = w.object_pose("mug")[1]
        q = PerturbedView(w, Perturbation(obj_yaw_deg=10.0), 610001).object_pose("mug")[1]
        self.assertAlmostEqual(abs(_yaw(q) - _yaw(q0)), np.radians(10.0), places=9)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=12)

    def test_the_jitter_is_fresh_at_every_read(self):
        w = self.world()
        v = PerturbedView(w, Perturbation(obj_jitter_m=0.01), 610001)
        reads = np.array([v.object_pose("mug")[0] for _ in range(4000)]) - w.object_pose("mug")[0]
        self.assertLess(float(np.abs(reads.mean(axis=0)).max()), 0.001)
        np.testing.assert_allclose(reads.std(axis=0), 0.01, rtol=0.1)

    def test_the_delay_reads_the_trace_row_that_long_ago(self):
        trace = [dict(t=j * 0.02, objects={"mug_b": np.r_[float(j), 0.0, 0.0, 1.0, 0, 0, 0]}) for j in range(100)]
        w = self.world(trace, t=100 * 0.02)
        pos, quat = PerturbedView(w, Perturbation(latency_s=0.105), 610001).object_pose("mug")
        self.assertEqual(float(pos[0]), 94.0)  # the last row at or before t - 0.105 = 1.895 s
        np.testing.assert_array_equal(quat, [1.0, 0, 0, 0])
        early = self.world(trace[:3], t=3 * 0.02)
        self.assertEqual(float(PerturbedView(early, Perturbation(latency_s=1.0), 1).object_pose("mug")[0][0]), 0.0)

    def test_the_spec_round_trips_and_unknown_keys_raise(self):
        p = Perturbation(obj_bias_m=0.01, latency_s=0.1)
        self.assertEqual(Perturbation.of(p.spec()), p)
        self.assertIsNone(Perturbation.of({}))
        with self.assertRaises(TypeError):
            Perturbation.of({"bias": 0.01})

    def test_a_perturbed_round_has_its_own_signature(self):
        B = _scene_batch()
        self.assertEqual(B.with_perception("abc", None), "abc")  # the truth keeps every earlier record current
        self.assertEqual(B.with_perception("abc", {}), "abc")
        a, b = B.with_perception("abc", {"obj_bias_m": 0.01}), B.with_perception("abc", {"obj_bias_m": 0.02})
        self.assertNotEqual(a, b)
        self.assertEqual(a, B.with_perception("abc", {"obj_bias_m": 0.01}))


class Sweep(unittest.TestCase):
    def test_conditions_pair_by_start_against_the_truth(self):
        names = ("gate", "batch", "family", "scene_round")
        saved = {k: sys.modules.pop(k, None) for k in names}
        here = str(REPO / "scripts" / "rsi")
        sys.path.insert(0, here)
        try:
            S = load_module("scene_rsi_perception_sweep", REPO / "scripts" / "rsi" / "perception_sweep.py")
        finally:
            sys.path.remove(here)
            for k, v in saved.items():
                sys.modules.pop(k, None)
                if v is not None:
                    sys.modules[k] = v
        for spec in S.CONDITIONS.values():
            Perturbation.of(spec)
        r = lambda ok: dict(ok=ok, fell=False, dropped=False)  # noqa: E731
        base = {"a": r(True), "b": r(True), "c": r(False), "d": r(True)}
        cond = {"a": r(True), "b": r(False), "c": r(True), "d": r(False), "e": r(True)}  # e: no L0 round, not paired
        got = S.paired(base, cond)
        self.assertEqual((got["n"], got["ok"], got["base_ok"], got["b"], got["c"]), (4, 2, 3, 1, 2))
        self.assertEqual((got["only_base"], got["only_cond"]), (["b", "d"], ["c"]))
        self.assertAlmostEqual(got["p_worse"], 0.5)  # 2 losses vs 1 win: P(>= 2 of 3) = 4/8


class Parameters(unittest.TestCase):
    def test_the_port_keeps_every_parameter_and_value(self):
        a = load(SEEDS / "grasp_c11", "c11_original")
        pa = a.parameters(a)
        b = load(SEEDS / "grasp_c11_view", "c11_view")
        pb = b.parameters(b)
        self.assertEqual(pa, pb)
        self.assertFalse(getattr(a, "WORLD_VIEW", False))
        self.assertTrue(b.WORLD_VIEW)


if __name__ == "__main__":
    unittest.main()
