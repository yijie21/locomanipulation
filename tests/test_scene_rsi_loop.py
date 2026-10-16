"""The scene RSI loop's gate arithmetic and candidate identity (TODO 27.2.4, scripts/experiments/scene_rsi_20261016/):
the paired tally counts wins, losses, falls and drops over the shared starts; the exact one-sided McNemar test; a
candidate's hash covers its code and its overrides and ignores byte caches."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "experiments" / "scene_rsi_20261016"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# scripts/gate.py and the warehouse loop's batch / family-like modules share these generic names: load this loop's
# under its own names and leave sys.modules as it was
_names = ("gate", "batch", "family", "scene_round")
_saved = {k: sys.modules.pop(k, None) for k in _names}
_path = list(sys.path)
sys.path.insert(0, str(HERE))
try:
    gate = _load("scene_rsi_gate", HERE / "gate.py")
finally:
    sys.path[:] = _path
    for _k, _v in _saved.items():
        sys.modules.pop(_k, None)
        if _v is not None:
            sys.modules[_k] = _v


def rec(ok, fell=False, dropped=False, why=""):
    return dict(ok=ok, fell=fell, dropped=dropped, why=why, plan_why=why)


class Tally(unittest.TestCase):
    def test_paired_counts_over_shared_starts(self):
        cand = {"a": rec(True), "b": rec(True), "c": rec(False, why="x"), "d": rec(True, fell=True), "extra": rec(True)}
        inc = {"a": rec(False), "b": rec(True), "c": rec(True), "d": rec(False, dropped=True)}
        t = gate.tally(cand, inc)
        self.assertEqual((t["n"], t["b"], t["c"]), (4, 2, 1))
        self.assertEqual((t["candidate_ok"], t["incumbent_ok"]), (3, 2))
        self.assertEqual((t["candidate_fell"], t["incumbent_dropped"]), (1, 1))
        self.assertEqual(t["candidate_why"], {"x": 1})

    def test_mcnemar(self):
        self.assertEqual(gate.mcnemar_one_sided(0, 0), 1.0)
        self.assertAlmostEqual(gate.mcnemar_one_sided(5, 0), 1 / 32)
        self.assertGreater(gate.mcnemar_one_sided(4, 0), 0.05)


class CandidateHash(unittest.TestCase):
    def test_hash_covers_code_and_overrides_not_caches(self):
        with tempfile.TemporaryDirectory() as d:
            c = Path(d)
            (c / "skills" / "__pycache__").mkdir(parents=True)
            (c / "skills" / "__init__.py").write_text("A = 1\n")
            (c / "candidate.json").write_text(json.dumps({"set": {"A": 2}}))
            h0 = gate.candidate_hash(c)
            (c / "skills" / "__pycache__" / "x.pyc").write_bytes(b"junk")
            self.assertEqual(gate.candidate_hash(c), h0)
            (c / "candidate.json").write_text(json.dumps({"set": {"A": 3}}))
            h1 = gate.candidate_hash(c)
            self.assertNotEqual(h1, h0)
            (c / "skills" / "__init__.py").write_text("A = 1\nB = 2\n")
            self.assertNotEqual(gate.candidate_hash(c), h1)


if __name__ == "__main__":
    unittest.main()


class Overrides(unittest.TestCase):
    def test_long_inline_overrides_parse_as_json_not_as_a_path(self):
        _saved = {k: sys.modules.pop(k, None) for k in _names}
        sys.path.insert(0, str(HERE))
        try:
            sr = _load("scene_rsi_round", HERE / "scene_round.py")
        finally:
            sys.path.remove(str(HERE))
            for _k, _v in _saved.items():
                sys.modules.pop(_k, None)
                if _v is not None:
                    sys.modules[_k] = _v
        long = json.dumps({f"grasp_program.SIMPLE.key_{i}": 0.001 * i for i in range(40)})
        self.assertGreater(len(long), 255)
        self.assertEqual(len(sr.parse_set(long)), 40)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "set.json"
            f.write_text('{"A": 1}')
            self.assertEqual(sr.parse_set(str(f)), {"A": 1})
        self.assertEqual(sr.parse_set(""), {})
