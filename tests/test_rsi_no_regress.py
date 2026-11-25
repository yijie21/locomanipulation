"""TODO 48.19 (user 2026-11-23, "三条都加上"), TODO 54: no task may lose significantly at the gate or the check half (caps
alpha_task, alpha_task_check: verdict.py's per-task sign test, Holm across the tasks), the sealed evaluation when a run
stops (caps sealed_at_stop, sealed.confirm), and the roles' hit rates (hit_rate.py)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"
import rsi_loop_ns as NS  # noqa: E402

CLASH = ("gate", "batch", "family", "scene_round", "digest", "loop", "sealed", "hit_rate")


def _mod(name: str):
    """scripts/rsi/<name>.py under a name of its own, sys.modules left as it was."""
    import importlib.util

    saved = {k: sys.modules.pop(k, None) for k in CLASH}
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location(f"noregress_{name}", HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"noregress_{name}"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _rounds(spec: dict) -> dict:
    """{label: ok} -> outcomes as batch.outcomes gives them."""
    return {
        k: dict(ok=v, task=k.rsplit("_s", 1)[0], fell=False, dropped=False, why="" if v else "x")
        for k, v in spec.items()
    }


class GateTaskRule(unittest.TestCase):
    def setUp(self):
        self.g = _mod("gate")
        # TaskA: 6 wins; TaskB: 2 losses -> net +4 overall, TaskB -2
        inc = {f"TaskA_s{i}": False for i in range(6)} | {f"TaskB_s{i}": True for i in range(4)}
        cand = {f"TaskA_s{i}": True for i in range(6)} | {
            "TaskB_s0": False,
            "TaskB_s1": False,
            "TaskB_s2": True,
            "TaskB_s3": True,
        }
        self.t = self.g.tally(_rounds(cand), _rounds(inc))

    def test_per_task_counts(self):
        self.assertEqual(self.t["per_task"]["TaskA"], dict(n=6, b=6, c=0))
        self.assertEqual(self.t["per_task"]["TaskB"], dict(n=4, b=0, c=2))

    def test_two_losses_on_a_task_no_longer_reject_but_a_significant_loss_does(self):
        self.assertEqual(self.g.verdict_fails(self.t, 3, None), [])
        # 2 losses to 0 wins on a task of four is within the noise (p 0.25): the old fixed rule rejected it
        self.assertEqual(self.g.verdict_fails(self.t, 3, None, alpha_task=0.05), [])
        inc = {f"TaskA_s{i}": False for i in range(20)} | {f"TaskB_s{i}": True for i in range(6)}
        cand = {f"TaskA_s{i}": True for i in range(20)} | {f"TaskB_s{i}": False for i in range(6)}
        t = self.g.tally(_rounds(cand), _rounds(inc))
        self.assertEqual(self.g.verdict_fails(t, 3, 0.05), [])  # the total is 20 : 6
        fails = self.g.verdict_fails(t, 3, 0.05, alpha_task=0.05)
        self.assertEqual(len(fails), 1)
        self.assertIn("task TaskB regresses", fails[0])

    def test_a_tally_without_per_task_is_judged_as_before(self):
        t = {k: v for k, v in self.t.items() if k != "per_task"}
        self.assertEqual(self.g.verdict_fails(t, 3, None, alpha_task=0.05), [])


class SealedConfirm(unittest.TestCase):
    def setUp(self):
        self.s = _mod("sealed")

    def test_confirmed(self):
        seed = _rounds({f"A_s{i}": False for i in range(8)})
        final = _rounds({f"A_s{i}": i < 6 for i in range(8)})
        c = self.s.confirm(final, seed, 0.05, 0.05)
        self.assertTrue(c["confirmed"], c)
        self.assertEqual((c["b"], c["c"]), (6, 0))

    def test_a_task_that_went_down_significantly_since_the_seed_is_not_confirmed(self):
        seed = _rounds({f"A_s{i}": False for i in range(30)} | {f"B_s{i}": True for i in range(7)})
        final = _rounds({f"A_s{i}": True for i in range(30)} | {f"B_s{i}": False for i in range(7)})
        c = self.s.confirm(final, seed, 0.05, 0.05)
        self.assertFalse(c["confirmed"])
        self.assertIn("task B regresses", c["why"][0])
        self.assertTrue(self.s.confirm(final, seed, 0.05, None)["confirmed"])
        small = _rounds(
            {f"A_s{i}": True for i in range(30)}
            | {"B_s0": False, "B_s1": False}
            | {f"B_s{i}": True for i in range(2, 7)}
        )
        self.assertTrue(self.s.confirm(small, seed, 0.05, 0.05)["confirmed"])  # 2 losses: within the noise

    def test_no_gain_is_not_confirmed(self):
        seed = _rounds({"A_s0": True, "A_s1": False})
        final = _rounds({"A_s0": False, "A_s1": True})
        c = self.s.confirm(final, seed, 0.05, 0.05)
        self.assertFalse(c["confirmed"])
        self.assertTrue(any("sign test p" in w for w in c["why"]))


class LoopHooks(unittest.TestCase):
    def setUp(self):
        self.lp = NS.fresh_loop(scene=False)
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        root = Path(self.d.name)
        self.hidden = root / "hidden"
        (self.hidden / "check").mkdir(parents=True)
        (self.hidden / "gate").mkdir(parents=True)
        p = mock.patch.multiple(self.lp, FAMILY="fam", LOOP=root / "vis", HIDDEN=self.hidden, STATE=root / "vis" / "loop",
                                GATE_DIR=self.hidden / "gate", CAPS={"check_half": True, "alpha_task_check": 0.10,
                                                                     "check_min_net": 1, "sealed_at_stop": True})  # fmt: skip
        p.start()
        self.addCleanup(p.stop)
        (root / "vis" / "loop").mkdir(parents=True)

    def _check(self, cand, base):
        def outcomes(path):
            return _rounds(cand if "candidate" in str(path) else base)

        with (
            mock.patch.multiple(self.lp, incumbent=lambda: dict(hash="inc", dir="/x", set={}), candidate_hash=lambda c: "h1",
                                check_pool=lambda: Path("/pool"), overrides=lambda c: {}),
            mock.patch.object(self.lp.B, "run_jobs", lambda *a, **k: {}),
            mock.patch.object(self.lp.B, "outcomes", outcomes),
        ):  # fmt: skip
            return self.lp.check_candidate(1, Path(self.d.name), Path("/cand"))

    def test_check_half_names_a_task_whose_loss_is_significant(self):
        base = {f"TaskA_s{i}": False for i in range(4)} | {f"TaskB_s{i}": True for i in range(7)}
        cand = {f"TaskA_s{i}": True for i in range(4)} | {f"TaskB_s{i}": False for i in range(7)}
        out = self._check(cand, base)
        self.assertEqual((out["net"], out["sent"]), (-3, False))
        self.assertEqual(out["task_regress"], {"TaskB": -7})
        cand["TaskB_s0"] = cand["TaskB_s1"] = cand["TaskB_s2"] = cand["TaskB_s3"] = True  # 3 losses: within the noise
        out = self._check(cand, base)
        self.assertEqual((out["net"], out["task_regress"], out["sent"]), (1, {}, True))

    def test_sealed_runs_once_for_a_final_incumbent_and_never_for_a_stub_or_the_seed(self):
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return mock.Mock(returncode=0)

        gate = self.hidden / "gate"
        (gate / "incumbent.json").write_text(json.dumps(dict(dir=str(gate / "incumbent_seed"), rounds="r")))
        with mock.patch.object(self.lp.subprocess, "run", fake_run):
            self.lp.run_sealed("manager: stop", False)
            self.assertEqual(calls, [])  # nothing accepted
            (gate / "incumbent.json").write_text(json.dumps(dict(dir=str(gate / "03_abc" / "cand"), rounds="r")))
            self.lp.run_sealed("manager: stop", True)  # stub
            self.assertEqual(calls, [])
            self.lp.run_sealed("role call failed: x", False)  # TODO 54: every way a run ends runs it, a failed role too
            self.assertEqual(len(calls), 1)
            self.assertIn("sealed.py", str(calls[0][1]))
            (self.hidden / "sealed" / "history.jsonl").write_text(json.dumps(dict(final="03_abc")) + "\n")
            self.lp.run_sealed("manager: stop", False)
            self.assertEqual(len(calls), 1)  # this final was evaluated already
            with mock.patch.object(self.lp, "CAPS", {}):
                (self.hidden / "sealed" / "history.jsonl").write_text("")
                self.lp.run_sealed("manager: stop", False)
                self.assertEqual(len(calls), 1)  # a family without the cap
        row = [e for e in self.lp.entries() if e.get("step") == "sealed"]
        self.assertEqual(row[0]["final"], "03_abc")
        self.assertNotIn("b", row[0])  # the numbers stay in the hidden tree


class HitRate(unittest.TestCase):
    def test_funnel_from_a_ledger_and_a_composer_context(self):
        hr = _mod("hit_rate")
        with tempfile.TemporaryDirectory() as d:
            loop = Path(d)
            rows = [
                dict(step="probe", iter=0, results=[dict(fixed=["a"], broken=[], status="verified", predicted_fixed_hit=1,
                                                         predicted_fixed_n=2),
                                                    dict(fixed=[], broken=["b"], status="refuted"),
                                                    dict(fixed=[], broken=[], status="refuted")]),
                dict(step="engineer", iter=0, candidate=1), dict(step="engineer", iter=0, candidate=2, void=True),
                dict(step="compose", iter=0, decision="merge"),
                dict(step="check", iter=0, sent=True, net=2), dict(step="check", iter=1, sent=False, net=-1),
                dict(step="gate", iter=0, accepted=False, reason="net wins -2 < 3", result=dict(b=1, c=3)),
                dict(step="gate", iter=1, accepted=False, reason="stage 1: net wins -1 <= 0", stage1=dict(b=0, c=1)),
                dict(step="gate", iter=2, accepted=True, reason="accepted", result=dict(b=9, c=1)),
                dict(step="gate", iter=3, accepted=False, reason="dry run: static checks passed", dry_run=True),
            ]  # fmt: skip
            (loop / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            (loop / "iter_00").mkdir()
            (loop / "iter_00" / "composer_context.json").write_text(json.dumps(dict(candidates=[
                dict(fixed="['x', 'y']", broken="['z']"), dict(fixed=[], broken=["w"])])))  # fmt: skip
            out = hr.compute(loop)
        r, e, c, g = out["reviewer"], out["engineer"], out["check"], out["gate"]
        self.assertEqual((r["probed"], r["better"], r["same"], r["worse"], r["broke_any"]), (3, 1, 1, 1, 1))
        self.assertEqual((r["predicted_fixed_hit"], r["predicted_fixed_n"]), (1, 2))
        self.assertEqual((e["built"], e["void"], e["seen"], e["visible_better"], e["visible_worse"]), (1, 1, 2, 1, 1))
        self.assertEqual((c["n"], c["sent"]), (2, 1))
        self.assertEqual((g["calls"], g["stage1_passed"], g["accepted"], g["rejected_net_negative"]), (3, 2, 1, 2))
        self.assertIn("gate 1/3 accepted", hr.line(out))


if __name__ == "__main__":
    unittest.main()
