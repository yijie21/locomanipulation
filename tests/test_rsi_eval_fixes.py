"""TODO 36 (review 2026-10-22): the scene RSI evaluation side's fixes (scripts/rsi/): a gate call with an incomplete
pairing is a void ledger line, one sentinel start per task, plan_for errors are failed rounds, the motor signature
versions round records (batch.done) and makes a regression reference-only, the episode time limit, write-back of
dict-entry overrides, the sealed version order."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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


_names = ("gate", "batch", "family", "scene_round", "regress")
_saved = {k: sys.modules.pop(k, None) for k in _names}
_path = list(sys.path)
sys.path.insert(0, str(HERE))
try:
    gate = _load("rsi_fix_gate", HERE / "gate.py")
    regress = _load("rsi_fix_regress", HERE / "regress.py")
    writeback = _load("rsi_fix_writeback", HERE / "writeback.py")
    B = gate.B
finally:
    sys.path[:] = _path
    for _k, _v in _saved.items():
        sys.modules.pop(_k, None)
        if _v is not None:
            sys.modules[_k] = _v


def _round(ok=True, motor=None, **kw):
    return dict(ok=ok, fell=False, dropped=False, t_end=1.0, why="", rule=B.SUCCESS_RULE, motor=motor, **kw)


class SentinelTest(unittest.TestCase):
    def test_one_per_task(self):
        labels = [f"{t}_s{i}" for t in "abcdefg" for i in range(4)]
        got = gate.sentinel_labels(labels)
        self.assertEqual(got, ["a_s0", "b_s0", "c_s0", "d_s0", "e_s0"])

    def test_fewer_tasks_than_sentinels(self):
        labels = [f"{t}_s{i}" for t in "ab" for i in range(4)]
        self.assertEqual(gate.sentinel_labels(labels), ["a_s0", "a_s1", "a_s2", "b_s0", "b_s1"])


class GateVoidTest(unittest.TestCase):
    def _run(self, drop: dict):
        """gate.main with a fake run_jobs; `drop` {job index: labels not written}."""
        tmp = Path(tempfile.mkdtemp())
        pool = tmp / "pool"
        pool.mkdir()
        for t in "abcdefg":
            for i in range(6):
                (pool / f"{t}_s{i}.json").write_text("{}")
        cand = tmp / "cand"
        (cand / "skills").mkdir(parents=True)
        (cand / "skills" / "__init__.py").write_text("def apply_set(pkg, ov):\n    return []\n")
        (cand / "candidate.json").write_text('{"set": {}}')

        def fake(jobs, family, pool_, procs, log, gpus="1", stop=None):
            for k, job in enumerate(jobs):
                job.out.mkdir(parents=True, exist_ok=True)
                for x in job.labels or B.pool_labels(pool_):
                    if x not in drop.get(k, ()):
                        (job.out / f"{x}.json").write_text(json.dumps(_round(ok="candidate" in str(job.out))))
            return {}

        real = B.run_jobs
        B.run_jobs = fake
        try:
            rc = gate.main(["f", "--gate", str(tmp / "gate"), "--pool", str(pool), "--candidate", str(cand)])
        finally:
            B.run_jobs = real
        lines = (tmp / "gate" / "ledger.jsonl").read_text().splitlines()
        return rc, [json.loads(x) for x in lines]

    def test_incomplete_candidate_is_void(self):
        rc, led = self._run({2: ["a_s0"]})
        self.assertEqual(rc, 1)
        self.assertEqual(len(led), 1)
        self.assertTrue(led[0]["void"])
        self.assertFalse(led[0]["accepted"])
        self.assertIn("pairing incomplete", led[0]["reason"])
        self.assertIn("a_s0", led[0]["reason"])
        self.assertIn("hash", led[0])
        self.assertIn("call", led[0])

    def test_missing_incumbent_round_is_void_not_keyerror(self):
        rc, led = self._run({0: ["b_s0"]})
        self.assertEqual(len(led), 1)
        self.assertTrue(led[0]["void"])
        self.assertIn("b_s0", led[0]["reason"])

    def test_complete_pairing_has_no_void(self):
        rc, led = self._run({})
        self.assertEqual(len(led), 1)
        self.assertNotIn("void", led[0])


class MotorSignatureTest(unittest.TestCase):
    def test_signature_follows_the_settings(self):
        from locomani.backends.scalebfm.motor import MotorConfig

        base = B.motor_signature()
        self.assertEqual(base, B.motor_signature("scalebfm", MotorConfig(), True, True))
        self.assertNotEqual(base, B.motor_signature("scalebfm", MotorConfig(walk_speed=0.5)))
        self.assertNotEqual(base, B.motor_signature(sleep=False))
        self.assertNotEqual(base, B.motor_signature(cuda_graph=False))
        self.assertNotEqual(base, B.motor_signature("amo"))
        # a default-off switch is left out while off (the records made before it existed stay done), counted when on
        self.assertEqual(base, B.motor_signature(motor_cfg=MotorConfig(reach_compensation=False)))
        self.assertNotEqual(base, B.motor_signature(motor_cfg=MotorConfig(reach_compensation=True)))
        for env, on in (("", False), ("0", False), ("1", True)):  # the batch's config: on with RSI_REACH_COMPENSATION=1
            with mock.patch.dict(os.environ, {"RSI_REACH_COMPENSATION": env}):
                self.assertEqual(B.motor_config().reach_compensation, on)
                self.assertEqual(B.motor_signature() == base, not on)

    def test_done_and_retire(self):
        sig = B.motor_signature()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.json"
            self.assertFalse(B.done(f, sig))
            f.write_text(json.dumps(_round(motor=sig)))
            self.assertTrue(B.done(f, sig))
            self.assertTrue(B.done(f))
            self.assertFalse(B.done(f, "other"))
            f.write_text(json.dumps(_round(motor=None)))  # a record without a signature
            self.assertFalse(B.done(f, sig))
            self.assertTrue(B.stale(f, sig).startswith("motor"))
            B.retire(f)
            self.assertFalse(f.exists())
            self.assertEqual(len(list(Path(d).glob("_motor*_x.json"))), 1)


class RegressReferenceTest(unittest.TestCase):
    def test_reference_when_signature_differs_or_missing(self):
        old = {"a": _round(motor="m1", task="t")}
        self.assertFalse(regress.compare(old, {"a": _round(motor="m1", task="t")})["reference"])
        self.assertTrue(regress.compare(old, {"a": _round(motor="m2", task="t")})["reference"])
        self.assertTrue(
            regress.compare({"a": _round(motor=None, task="t")}, {"a": _round(motor="m2", task="t")})["reference"]
        )

    def test_reference_family_is_not_worse_and_last_moves(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            reg = {"fam": dict(sandbox=str(d), kind="none", pool=str(d / "pool"), last="old")}
            (d / "pool").mkdir()
            old_dir = regress.rounds_dir("fam", reg["fam"], "old")
            old_dir.mkdir(parents=True)
            new_dir = regress.rounds_dir("fam", reg["fam"], "new")
            new_dir.mkdir(parents=True)
            for i in range(10):  # the old batch won everything, the new one nothing: p(worse) tiny
                (old_dir / f"s{i}.json").write_text(json.dumps(_round(True, motor=None, task="t")))
                (new_dir / f"s{i}.json").write_text(json.dumps(_round(False, motor=B.motor_signature(), task="t")))
                (d / "pool" / f"s{i}.json").write_text("{}")
            saved = (regress.REGISTRY, regress.load, B.run_jobs)
            regress.REGISTRY = d / "outer" / "registry.json"
            regress.REGISTRY.parent.mkdir()
            regress.load = lambda: reg
            B.run_jobs = lambda *a, **k: {}
            try:
                out = regress.run("new", None, 1, "0", regress.CATALOG)
            finally:
                regress.REGISTRY, regress.load, B.run_jobs = saved
            self.assertEqual(out["worse"], [])
            self.assertEqual(out["reference"], ["fam"])
            self.assertLessEqual(out["families"]["fam"]["p_worse"], 0.05)
            self.assertEqual(json.loads((d / "outer" / "registry.json").read_text())["fam"]["last"], "new")


class EpisodeLimitTest(unittest.TestCase):
    def _world(self):
        from locomani.backends.scalebfm.motor import Outcome

        class Motor:
            fell_at = None

            def __init__(self):
                self.k = 0
                self.track = {}

            @property
            def t(self):
                return self.k * 0.02

            def reset(self, *a):
                self.k = 0

            def place(self, *a):
                pass

            def tick(self):
                self.k += 1
                return True

            def hold(self, s):
                t0 = self.t
                while self.t - t0 < s:
                    self.tick()
                return Outcome(True, "", t0, self.t)

            def arrays(self):
                return {}

        world = SimpleNamespace(
            motor=Motor(), meta={"objects": {}}, free={}, seat=lambda k: 0.0,
            check=lambda goal: (lambda st: (True, "")), state=lambda keys: {},
        )  # fmt: skip
        start = SimpleNamespace(placements={}, robot_xy=(0, 0), robot_yaw=0.0, seed=1, to_dict=lambda: {})
        return world, SimpleNamespace(objects=[], goal=[], id="t"), start

    def test_endless_skill_fails_with_limit(self):
        from locomani.tasks.scene.episode import run_episode

        world, task, start = self._world()

        def plan(ctx):
            while True:
                try:
                    ctx.motor.tick()
                except Exception:  # a skill swallowing errors must not stop the cap
                    pass

        res, _ = run_episode(world, task, start, plan, limit_s=5.0)
        self.assertFalse(res.ok)
        self.assertEqual(res.why, "episode time limit")
        self.assertLess(res.execution["t_end"], 5.1)
        self.assertNotIn("tick", vars(world.motor))  # the cap is removed again

    def test_short_plan_unaffected(self):
        from locomani.backends.scalebfm.motor import Outcome
        from locomani.tasks.scene.episode import run_episode

        world, task, start = self._world()
        res, _ = run_episode(world, task, start, lambda ctx: ctx.motor.hold(1.0) and Outcome(True), limit_s=5.0)
        self.assertTrue(res.ok)
        self.assertEqual(res.why, "")


class WritebackEntryTest(unittest.TestCase):
    SRC = (
        "A = 1  # a\n"
        "ENTRY_GATE = {\n"
        '    "lateral_m": 0.02,  # the gate\n'
        '    "depth_m": 0.05,\n'
        '    "nest": {"x": 3},\n'
        "}\n"
    )

    def test_fold_entry(self):
        new, found = writeback.fold_entry(self.SRC, "ENTRY_GATE", ["lateral_m"], 0.03, "fam")
        self.assertTrue(found)
        self.assertIn('"lateral_m": 0.03,  # (was 0.02 until fam\'s set) the gate', new)
        self.assertIn('"depth_m": 0.05,', new)
        compile(new, "x", "exec")
        new, found = writeback.fold_entry(self.SRC, "ENTRY_GATE", ["nest", "x"], 4, "fam")
        self.assertTrue(found)
        self.assertEqual(
            eval(new.split("ENTRY_GATE = ")[1].replace("# ", "#").split("\n}")[0] + "\n}", {})["nest"], {"x": 4}
        )
        self.assertFalse(writeback.fold_entry(self.SRC, "ENTRY_GATE", ["nope"], 1, "fam")[1])
        self.assertFalse(writeback.fold_entry(self.SRC, "A", ["k"], 1, "fam")[1])

    def test_plan_folds_entries_and_reports_dotless_keys(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cat = d / "catalog"
            cat.mkdir()
            (cat / "grasp_program.py").write_text(self.SRC)
            gate_dir = d / "outputs" / "scene_rsi_20261016_hidden" / "fam" / "gate"
            for name in ("incumbent_seed", "final"):
                (gate_dir / name / "skills").mkdir(parents=True)
                (gate_dir / name / "skills" / "m.py").write_text("X = 1\n" if name == "incumbent_seed" else "X = 2\n")
            (gate_dir / "final" / "candidate.json").write_text(
                json.dumps({"set": {"grasp_program.ENTRY_GATE.lateral_m": 0.03, "TOP": 1, "grasp_program.A": 5}})
            )
            (gate_dir / "incumbent.json").write_text(json.dumps(dict(dir=str(gate_dir / "final"))))
            (cat / "m.py").write_text("X = 1\n")
            p = writeback.plan(d, "fam", cat)
            self.assertIn("grasp_program.ENTRY_GATE.lateral_m", p["folded"])
            self.assertIn("grasp_program.A", p["folded"])
            self.assertEqual(len(p["conflicts"]), 1)
            self.assertIn("TOP", p["conflicts"][0])
            self.assertIn("0.03", p["texts"]["grasp_program.py"])


if __name__ == "__main__":
    unittest.main()
