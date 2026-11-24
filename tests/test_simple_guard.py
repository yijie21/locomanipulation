"""TODO 49.2: the static guard of the SIMPLE skill package (guard.py): the old --selftest cases, the new forbidden imports,
and the real seed package against itself."""

import shutil
import tempfile
import unittest
from pathlib import Path

from locomani.bench.simple import guard as G
from locomani.bench.simple.config import SEED_DIR


class Guard(unittest.TestCase):
    def test_selftest_cases(self):
        self.assertEqual(G.selftest(), [])

    def _copy(self, t, extra=None):
        d = Path(t) / "cand"
        shutil.copytree(SEED_DIR, d, ignore=shutil.ignore_patterns("__pycache__"))
        for name, text in (extra or {}).items():
            (d / name).write_text(text)
        return d

    def test_seed_against_itself_is_clean(self):
        with tempfile.TemporaryDirectory() as t:
            r = G.check(SEED_DIR, self._copy(t))
        self.assertTrue(r["ok"], r)
        self.assertEqual(set(r["files"]), {"skill.py", "pick.py", "camera_view.py"})

    def test_editable_names(self):
        self.assertEqual(set(G.EDITABLE), {"skill.py", "handover.py", "pick.py", "camera_view.py"})
        self.assertEqual(G.FROZEN, ("__init__.py",))

    def test_frozen_init_and_unknown_file(self):
        with tempfile.TemporaryDirectory() as t:
            d = self._copy(t, {"__init__.py": "x = 1\n", "evil.py": "x = 1\n"})
            rules = {v["rule"] + ":" + v["file"] for v in G.check(SEED_DIR, d)["violations"]}
        self.assertEqual(rules, {"R1:__init__.py", "R1:evil.py"})

    def test_the_package_may_not_reach_the_simulator_side(self):
        for line in (
            "from locomani.bench.simple import run",
            "import locomani.bench.simple.client",
            "from locomani.bench.simple.server import x",
            "from locomani.bench.simple import episodes",
            "from locomani.perception.truth import TruthView",
        ):
            with tempfile.TemporaryDirectory() as t:
                d = self._copy(t)
                p = d / "skill.py"
                p.write_text(line + "\n" + p.read_text())
                rules = {v["rule"] for v in G.check(SEED_DIR, d)["violations"]}
            self.assertIn("R8", rules, line)

    def test_simulator_reads_still_counted(self):
        with tempfile.TemporaryDirectory() as t:
            d = self._copy(t)
            p = d / "camera_view.py"
            p.write_text(p.read_text() + "\n\ndef peek(env):\n    return env.sim.d.qpos\n")
            rules = {v["rule"] for v in G.check(SEED_DIR, d)["violations"]}
        self.assertIn("R4", rules)

    def test_seed_package_has_no_simulator_truth_class(self):
        text = (SEED_DIR / "skill.py").read_text()
        for name in ("class TruthView", "class Belief", "def body_verts"):
            self.assertNotIn(name, text)


if __name__ == "__main__":
    unittest.main()
