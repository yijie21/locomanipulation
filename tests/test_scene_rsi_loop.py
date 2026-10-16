"""The scene RSI loop's gate arithmetic and candidate identity (TODO 27.2.4, scripts/experiments/scene_rsi_20261016/):
the paired tally counts wins, losses, falls and drops over the shared starts; the exact one-sided McNemar test; a
candidate's hash covers its code and its overrides and ignores byte caches."""

import json
import os
import sys
import tempfile
import time
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


class RunCount(unittest.TestCase):
    """`loop.py run` does --max-iters iterations, and a re-probed iteration counts as one of them (0597b20 dropped
    the loop: a plain run did nothing and --reprobe always went on to one more iteration)."""

    def calls(self, max_iters, reprobe=None):
        import argparse
        from unittest import mock

        _saved = {k: sys.modules.pop(k, None) for k in _names}
        sys.path.insert(0, str(HERE))
        try:
            lp = _load("scene_rsi_loop", HERE / "loop.py")
        finally:
            sys.path.remove(str(HERE))
            for _k, _v in _saved.items():
                sys.modules.pop(_k, None)
                if _v is not None:
                    sys.modules[_k] = _v
        seen = []
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pool_diag").mkdir()
            with mock.patch.multiple(lp, STATE=Path(d) / "state", LOOP=Path(d), accept_pool=lambda: Path(d),
                                     ledger=lambda e: None, stop_reason=lambda m: None,
                                     iterate=lambda n, m, s: seen.append(("iterate", n)),
                                     reprobe=lambda n, s: seen.append(("reprobe", n))):  # fmt: skip
                lp.cmd_run(argparse.Namespace(max_iters=max_iters, reprobe=reprobe, stub=True))
        return seen

    def test_plain_run_iterates_max_iters_times(self):
        self.assertEqual(self.calls(2), [("iterate", 0), ("iterate", 1)])

    def test_reprobe_counts_as_an_iteration(self):
        self.assertEqual(self.calls(1, reprobe=0), [("reprobe", 0)])
        self.assertEqual(self.calls(2, reprobe=0), [("reprobe", 0), ("iterate", 0)])


class RoleLeftovers(unittest.TestCase):
    """A role that has answered but waits on a shell it left running is unblocked: the driver kills the leftover after
    the grace period and still gets the role's output (iteration 2, 2026-10-16, stalled on a self-matching pgrep)."""

    def test_leftover_child_is_killed_after_the_answer(self):
        from unittest import mock

        _saved = {k: sys.modules.pop(k, None) for k in _names}
        sys.path.insert(0, str(HERE))
        try:
            lp = _load("scene_rsi_loop_roles", HERE / "loop.py")
        finally:
            sys.path.remove(str(HERE))
            for _k, _v in _saved.items():
                sys.modules.pop(_k, None)
                if _v is not None:
                    sys.modules[_k] = _v
        role = (
            "import json, subprocess, sys\n"
            "sys.stdin.read()\n"
            "open(sys.argv[1], 'w').write('Structured output provided successfully')\n"
            "subprocess.Popen(['sleep', '600']).wait()\n"
            "print(json.dumps({'ok': 1}))\n"
        )
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "transcript.jsonl"
            with mock.patch.object(lp, "transcript", lambda s: t if t.exists() else None):
                t0 = time.time()
                out, _err, rc, killed = lp.run_role([sys.executable, "-c", role, str(t)], "prompt", dict(os.environ),
                                                    "s", grace=1.0, poll=0.2)  # fmt: skip
        self.assertLess(time.time() - t0, 30)
        self.assertEqual((json.loads(out), rc), ({"ok": 1}, 0))
        self.assertTrue(killed and "sleep" in killed[0])


if __name__ == "__main__":
    unittest.main()
