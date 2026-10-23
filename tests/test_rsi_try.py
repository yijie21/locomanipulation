"""TODO 37.1 (user 2026-10-23: "B 到 E 按顺序做"): scripts/rsi/try.py, the roles' bounded, queue-backed check command --
its start selection and caps, the per-iteration round budget, the table it prints, and the family loader that no
longer crashes while the hidden tree is locked (the reason run C's roles wrapped scene_round.py themselves)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _try_module():
    """scripts/rsi/try.py under its own name; the warehouse loop has a batch.py too, so sys.modules is restored."""
    names = ("batch", "digest", "family", "scene_round")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load("scene_rsi_try", HERE / "try.py")
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _round(label, task, ok, cls=None, steps=None, fell=False):
    return dict(label=label, task=task, ok=ok, why="" if ok else "not held", fell=fell, dropped=False,
                failure_class=cls, steps=steps or [], t_end=40.0, rule=2)  # fmt: skip


class Starts(unittest.TestCase):
    def setUp(self):
        self.T = _try_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.pool = Path(self.tmp.name) / "pool_diag"
        self.pool.mkdir()
        for i in range(45):
            task = "pick_Mug_2" if i % 3 == 0 else "pick_Orange_A"
            (self.pool / f"{task}_s{i:03d}.json").write_text(json.dumps(dict(task=task, start={})))

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_a_task_or_all_within_the_cap(self):
        T = self.T
        self.assertEqual(T.pick_labels(self.pool, "pick_Mug_2_s000,pick_Mug_2_s003", None, False),
                         ["pick_Mug_2_s000", "pick_Mug_2_s003"])  # fmt: skip
        self.assertEqual(len(T.pick_labels(self.pool, "", "pick_Mug_2", False)), 15)
        with self.assertRaises(SystemExit) as c:
            T.pick_labels(self.pool, "", None, True)  # 45 starts
        self.assertIn("at most 40", str(c.exception))
        with self.assertRaises(SystemExit):
            T.pick_labels(self.pool, "nope_s1", None, False)
        with self.assertRaises(SystemExit):
            T.pick_labels(self.pool, "", None, False)

    def test_hidden_and_outside_pools_are_refused(self):
        T = self.T
        root = Path(self.tmp.name)
        hidden = root / "outputs" / "scene_rsi_20261016_hidden" / "f" / "pool_accept"
        hidden.mkdir(parents=True)
        self.assertIn("hidden", T.pool_allowed(hidden, repo=root, scratch=root / "scratch"))
        self.assertIsNone(T.pool_allowed(self.pool, repo=root, scratch=root / "scratch"))
        with tempfile.TemporaryDirectory() as other:
            self.assertIn("outside", T.pool_allowed(Path(other), repo=root, scratch=root / "scratch"))


class Budget(unittest.TestCase):
    def test_the_ledger_counts_rounds_and_the_cap_refuses_the_rest(self):
        T = _try_module()
        with tempfile.TemporaryDirectory() as d:
            led = Path(d) / "role" / "try_iter_03.jsonl"
            self.assertEqual(T.used(led), 0)
            T.charge(led, 230, ["a"] * 230, Path(d) / "out", {})
            self.assertEqual(T.used(led), 230)
            T.charge(led, 5, ["b"] * 5, Path(d) / "out2", {"pick.X": 1})
            self.assertEqual(T.used(led), 235)
            rows = [json.loads(x) for x in led.read_text().splitlines()]
            self.assertEqual((rows[-1]["n"], rows[-1]["set"]), (5, {"pick.X": 1}))
            with mock.patch.dict(os.environ, {"RSI_TRY_LEDGER": str(led)}):
                self.assertEqual(T.ledger_path(), led)
            with mock.patch.dict(os.environ, {"TMPDIR": d}, clear=False):
                os.environ.pop("RSI_TRY_LEDGER", None)
                self.assertEqual(T.ledger_path(), Path(d) / "try_ledger.jsonl")

    def test_main_runs_only_the_missing_rounds_and_charges_them(self):
        T = _try_module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pool = root / "outputs" / "scene_rsi_20261016" / "fam" / "pool_diag"
            pool.mkdir(parents=True)
            for i in range(4):
                (pool / f"pick_A_s{i}.json").write_text(json.dumps(dict(task="pick_A", start={})))
            skills = root / "skills"
            skills.mkdir()
            out, ref = root / "out", root / "ref"
            ref.mkdir()
            for i in range(4):
                (ref / f"pick_A_s{i}.json").write_text(json.dumps(_round(f"pick_A_s{i}", "pick_A", i < 2)))
            out.mkdir()
            (out / "pick_A_s0.json").write_text(json.dumps(dict(_round("pick_A_s0", "pick_A", True), motor="sig")))
            ran = []

            def fake_run_jobs(jobs, family, p, procs, log, gpus):
                for x in jobs[0].labels:
                    f = jobs[0].out / f"{x}.json"
                    if not T.B.done(f, "sig"):
                        ran.append(x)
                        f.write_text(json.dumps(dict(_round(x, "pick_A", x.endswith("3"), "slip"), motor="sig")))
                return {0: []}

            led = root / "try.jsonl"
            env = {"RSI_TRY_LEDGER": str(led), "TMPDIR": d}
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.object(T, "REPO", root),
                mock.patch.object(T.B, "motor_signature", lambda *a, **k: "sig"),
                mock.patch.object(T.B, "run_jobs", fake_run_jobs),
                mock.patch("builtins.print") as out_print,
            ):
                T.main(["fam", "--skills", str(skills), "--all", "--out", str(out), "--ref", str(ref)])
                text = "\n".join(str(c.args[0]) for c in out_print.call_args_list)
            self.assertEqual(ran, ["pick_A_s1", "pick_A_s2", "pick_A_s3"])  # s0 was done: not run, not charged
            self.assertEqual(T.used(led), 3)
            self.assertIn("2/4 succeed: pick_A 2/4", text)
            self.assertIn("fixed 1 ['pick_A_s3'], broken 1 ['pick_A_s1']", text)
            self.assertIn("pick_A_s2 fail [slip]", text)
            self.assertIn("3 run now, 3 of", text)
            with (
                mock.patch.dict(os.environ, dict(env, RSI_TRY_BUDGET="2")),
                mock.patch.object(T, "REPO", root),
                mock.patch.object(T, "BUDGET", 2),
                mock.patch.object(T.B, "motor_signature", lambda *a, **k: "sig"),
            ):
                with self.assertRaises(SystemExit) as c:  # 3 used of 2: a call that needs more rounds is refused
                    T.main(["fam", "--skills", str(skills), "--labels", "pick_A_s1", "--out", str(root / "o2")])
                self.assertIn("budget", str(c.exception))
                T.main(["fam", "--skills", str(skills), "--all", "--out", str(out)])  # nothing missing: free


class Table(unittest.TestCase):
    def test_outcomes_first_fail_and_steps(self):
        T = _try_module()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            steps = [dict(step="walk", ok=True, t0=1.0, t1=5.0, stop_err_cm=1.2),
                     dict(step="entry", ok=False, t0=5.0, t1=9.0, why="jaw 4.1 cm off", jaw_err_cm=4.1)]  # fmt: skip
            (out / "pick_A_s1.json").write_text(json.dumps(_round("pick_A_s1", "pick_A", False, "side_reach", steps)))
            (out / "pick_A_s2.json").write_text(json.dumps(_round("pick_A_s2", "pick_A", True, fell=True)))
            (out / "_motorx_pick_A_s3.json").write_text("{}")  # a retired record is not a round
            t = T.table(out, None, False)
            self.assertIn("1/2 succeed: pick_A 1/2", t)
            self.assertIn("pick_A_s1 fail [side_reach] t 40s | first fail entry: jaw 4.1 cm off", t)
            self.assertIn("pick_A_s2 OK   F t 40s", t)
            t2 = T.table(out, None, True)
            self.assertIn("entry x [5.0-9.0] jaw_err_cm 4.1", t2)


class HiddenSeedsLocked(unittest.TestCase):
    def test_load_survives_a_locked_hidden_tree(self):
        # Python 3.10: Path.exists() raises PermissionError under a mode-000 directory; run C's Engineers stubbed
        # family.hidden_seeds to get scene_round.py to run during their session
        names = ("family",)
        saved = {k: sys.modules.pop(k, None) for k in names}
        sys.path.insert(0, str(HERE))
        try:
            F = _load("scene_rsi_family_try", HERE / "family.py")
        finally:
            sys.path.remove(str(HERE))
            for k, v in saved.items():
                sys.modules.pop(k, None)
                if v is not None:
                    sys.modules[k] = v
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            hid = root / "outputs" / "scene_rsi_20261016_hidden" / "fam"
            hid.mkdir(parents=True)
            (hid / "seeds.json").write_text('{"accept": 7}')
            with mock.patch.object(F, "REPO", root):
                self.assertEqual(F.hidden_seeds("fam"), {"accept": 7})
                os.chmod(hid, 0)
                try:
                    self.assertEqual(F.hidden_seeds("fam"), {})
                finally:
                    os.chmod(hid, 0o755)


if __name__ == "__main__":
    unittest.main()
