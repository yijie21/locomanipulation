"""Step 0 of the vision RSI design (docs/handoff/vision-rsi-design-20261023.md): a skills package that sets WORLD_VIEW
reads the scene only through locomani.tasks.scene.view -- no simulator handle, model arrays or contact readers in its
source, the robot view refusing them at run time -- and the ported incumbent keeps the original's parameters."""

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

from locomani.tasks.scene.view import RobotView, TruthView  # noqa: E402

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
