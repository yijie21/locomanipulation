"""TODO 56.1 (user 2026-11-27): scripts/rsi/verdict.py, the one verdict function of the gate (stage 1, the full pool,
the `sync` rule) and the sealed set: the exact binomial edges, Holm, the per-task regression report that never rejects,
the scene rule against recorded gate tallies, and gate.py reading its three verdicts and `per_task` from it."""

import importlib.util
import json
import sys
import tempfile
import unittest
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _isolated(name: str, path: Path):
    """A scripts/rsi module under its own name (the warehouse loop has a gate.py and a batch.py too)."""
    names = ("gate", "batch", "family", "scene_round", "verdict")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load(name, path)
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


V = _load("rsi_verdict_under_test", HERE / "verdict.py")


def tally(b, c, per_task=None, **kw):
    t = dict(n=b + c, b=b, c=c, candidate_ok=b + 10, incumbent_ok=10 + c, candidate_fell=0, incumbent_fell=0,
             candidate_dropped=0, incumbent_dropped=0, per_task=per_task or {})  # fmt: skip
    return {**t, **kw}


class SignTest(unittest.TestCase):
    def test_exact_edges(self):
        self.assertEqual(V.sign_p(0, 0), 1.0)
        self.assertAlmostEqual(V.sign_p(5, 0), 1 / 32)
        self.assertAlmostEqual(V.sign_p(4, 0), 1 / 16)
        self.assertAlmostEqual(V.sign_p(6, 1), sum(comb(7, i) for i in (6, 7)) / 2**7)
        self.assertAlmostEqual(V.sign_p(3, 1), 5 / 16)
        self.assertAlmostEqual(V.sign_p(1, 3), 15 / 16)

    def test_the_smallest_tally_at_p_005_is_five_to_nothing(self):
        least = {c: next(b for b in range(60) if V.sign_p(b, c) <= 0.05) for c in range(5)}
        self.assertEqual(least, {0: 5, 1: 7, 2: 9, 3: 10, 4: 12})  # why gate rule 1's p <= 0.05 needed 5:0

    def test_holm_step_down(self):
        out = V.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}, 0.05)
        self.assertEqual([out[k]["reject"] for k in "abcd"], [True, False, False, False])  # 0.03 > 0.05 / 3 ends it
        self.assertEqual((out["a"]["p_holm"], out["c"]["p_holm"], out["b"]["p_holm"], out["d"]["p_holm"]),
                         (0.04, 0.09, 0.09, 0.5))  # fmt: skip
        out = V.holm({"a": 0.001, "b": 0.01, "c": 0.02}, 0.05)
        self.assertTrue(all(v["reject"] for v in out.values()))
        self.assertEqual(V.holm({}, 0.05), {})


class DecideTest(unittest.TestCase):
    def test_stage_one_rejects_on_no_net_win(self):
        """The scene gate's stage-1 rule since TODO 27: net wins must be positive (main rejects on < 0 only)."""
        for b, c in ((0, 0), (2, 2), (1, 2)):
            r = V.decide(tally(b, c), "stage1")
            self.assertFalse(r["decision"], (b, c))
            self.assertIn(f"stage 1: net wins {b - c} <= 0 on {b + c} pairs", r["reasons"])
        self.assertTrue(V.decide(tally(1, 0), "stage1")["decision"])
        self.assertTrue(V.decide(tally(4, 3), "stage1")["decision"])

    def test_gate_is_net_three_losses_a_third_falls_and_drops(self):
        for b, c in ((5, 1), (4, 0), (3, 0), (6, 2)):
            self.assertTrue(V.decide(tally(b, c), "gate")["decision"], (b, c))
        for b, c in ((3, 1), (2, 0), (6, 3)):
            self.assertFalse(V.decide(tally(b, c), "gate")["decision"], (b, c))
        self.assertEqual(V.decide(tally(6, 3), "gate")["reasons"], ["losses 3 > wins 6 / 3"])
        self.assertEqual(V.decide(tally(2, 0), "gate")["reasons"], ["net wins 2 < 3"])
        self.assertEqual(V.decide(tally(5, 0, candidate_fell=2), "gate")["reasons"], ["falls 2 > 0 + 1"])
        self.assertEqual(V.decide(tally(5, 0, candidate_dropped=2), "gate")["reasons"], ["drops 2 > 1"])
        self.assertTrue(V.decide(tally(2, 0), "gate", dict(min_net=2))["decision"])  # the params override

    def test_drop_limit_from_a_zero_start(self):
        t = tally(14, 0, candidate_ok=14, incumbent_ok=0, incumbent_dropped=0)
        self.assertEqual(V.drop_limit(t), 7)
        self.assertEqual(V.drop_limit(dict(t, candidate_ok=2)), 1)  # never below incumbent + 1
        self.assertEqual(V.drop_limit(dict(t, incumbent_ok=9)), 1)

    def test_sync_accepts_no_more_losses_than_wins(self):
        self.assertTrue(V.decide(tally(0, 0), "sync")["decision"])
        self.assertTrue(V.decide(tally(3, 3), "sync")["decision"])
        r = V.decide(tally(2, 3), "sync")
        self.assertEqual((r["decision"], r["reasons"]), (False, ["losses 3 > wins 2"]))

    def test_the_per_task_report_never_rejects(self):
        """A task lost 6:0 is significant after Holm across two tasks (p 1/64 <= 0.05 / 2) and is reported as such;
        the verdict still follows the pool-wide rule (user 2026-11-27: a report for the human and the roles)."""
        per = {"A": dict(n=30, b=30, c=0), "B": dict(n=10, b=0, c=6)}
        r = V.decide(tally(30, 6, per), "gate")
        self.assertTrue(r["decision"])
        self.assertEqual(r["reasons"], [])
        self.assertTrue(r["per_task"]["B"]["regress"])
        self.assertAlmostEqual(r["per_task"]["B"]["p_regress"], 1 / 64)
        self.assertAlmostEqual(r["per_task"]["B"]["p_holm"], 2 / 64)
        self.assertEqual((r["per_task"]["B"]["b"], r["per_task"]["B"]["c"], r["per_task"]["B"]["net"]), (0, 6, -6))
        self.assertFalse(r["per_task"]["A"]["regress"])
        self.assertEqual(r["per_task"]["A"]["p_regress"], 1.0)
        # 4 losses to 0 (p 1/16) does not survive Holm across two tasks at 0.05
        r = V.decide(tally(14, 4, {"A": dict(n=14, b=14, c=0), "B": dict(n=10, b=0, c=4)}), "gate")
        self.assertTrue(r["decision"])
        self.assertFalse(r["per_task"]["B"]["regress"])
        # alpha_task None: the p values stay, nothing is marked
        r = V.decide(tally(30, 6, per), "gate", dict(alpha_task=None))
        self.assertFalse(any(v["regress"] for v in r["per_task"].values()))
        self.assertAlmostEqual(r["per_task"]["B"]["p_regress"], 1 / 64)

    def test_a_tally_already_carrying_the_report_is_passed_through(self):
        t = tally(5, 0, {"A": dict(n=5, b=5, c=0, net=5, p_regress=1.0, p_holm=1.0, regress=False)})
        self.assertEqual(V.decide(t, "gate")["per_task"], t["per_task"])

    def test_sealed_judges_rounds_with_the_gate_rule_and_the_task_test(self):
        final = {f"T1_s{i}": dict(ok=True, task="T1") for i in range(20)} | {
            f"T2_s{i}": dict(ok=False, task="T2") for i in range(10)
        }
        seed = {f"T1_s{i}": dict(ok=False, task="T1") for i in range(20)} | {
            f"T2_s{i}": dict(ok=True, task="T2") for i in range(10)
        }
        r = V.judge(final, seed, "sealed")
        self.assertEqual((r["n"], r["b"], r["c"]), (30, 20, 10))
        self.assertFalse(r["decision"])  # 10 losses > 20 / 3
        self.assertEqual(r["reasons"], ["losses 10 > wins 20 / 3"])
        self.assertTrue(r["per_task"]["T2"]["regress"])  # lost 10 to 0: p 1 / 1024
        self.assertFalse(r["per_task"]["T1"]["regress"])
        seed = {k: dict(v, ok=(v["task"] == "T2" and k not in ("T2_s0", "T2_s1"))) for k, v in seed.items()}
        r = V.judge(final, seed, "sealed")  # T2 lost 8 of 10, 2 ties: 20:8 passes the loss share (8 * 3 <= 20 fails)
        self.assertFalse(r["decision"])
        self.assertEqual(r["per_task"]["T2"]["c"], 8)
        r = V.judge(final, {k: dict(v, ok=False) for k, v in seed.items()}, "sealed")
        self.assertTrue(r["decision"])
        self.assertEqual(r["per_task"]["T2"], dict(n=10, b=0, c=0, net=0, p_regress=1.0, p_holm=1.0, regress=False))

    def test_task_of_falls_back_to_the_label(self):
        self.assertEqual(V.task_of("Box_Small_s12"), "Box_Small")
        self.assertEqual(V.task_of("x_s1", dict(task="T")), "T")


# a sample of recorded gate tallies of the scene families (gate rule 2): (accepted, b, c, candidate_fell,
# incumbent_fell, candidate_dropped, incumbent_dropped, candidate_ok, incumbent_ok, n), from ~/workspace/code/rsi-scene
RECORDED = [
    (False, 2, 4, 0, 0, 22, 20, 78, 80, 150), (False, 2, 0, 0, 0, 3, 3, 27, 25, 40),
    (False, 5, 2, 1, 1, 0, 1, 12, 9, 40),
    (True, 3, 0, 0, 0, 1, 1, 29, 26, 32), (True, 5, 1, 1, 1, 2, 1, 20, 16, 40),
    (True, 10, 0, 0, 0, 1, 7, 26, 16, 32), (True, 18, 1, 1, 1, 1, 1, 26, 9, 40),
    (True, 20, 3, 2, 1, 0, 1, 26, 9, 40), (True, 23, 1, 1, 1, 1, 1, 31, 9, 40),
    (True, 26, 0, 1, 1, 0, 1, 35, 9, 40), (True, 28, 1, 1, 1, 0, 1, 36, 9, 40),
    (True, 43, 0, 0, 0, 0, 0, 71, 28, 90), (True, 61, 0, 3, 3, 15, 0, 61, 0, 90),
]  # fmt: skip


class RecordedTest(unittest.TestCase):
    def test_the_scene_rule_gives_the_recorded_verdicts(self):
        for row in RECORDED:
            acc, b, c, cf, inf, cd, ind, cok, iok, n = row
            t = tally(b, c, n=n, candidate_fell=cf, incumbent_fell=inf, candidate_dropped=cd, incumbent_dropped=ind,
                      candidate_ok=cok, incumbent_ok=iok, p_one_sided=round(V.sign_p(b, c), 5))  # fmt: skip
            self.assertIs(V.decide(t, "gate")["decision"], acc, row)


def _round(ok=True, **kw):
    return dict(ok=ok, fell=False, dropped=False, t_end=1.0, why="", **kw)


class GateReadsVerdictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = _isolated("rsi_verdict_gate", HERE / "gate.py")

    def test_tally_carries_the_per_task_report_on_stages(self):
        """A pack family's wins are higher stages (gso_pack_all.md D15); per_task counts them the same way as b / c."""
        cand = {"A_s0": _round(False, stage=3, task="A"), "A_s1": _round(True, stage=4, task="A"),
                "B_s0": _round(False, stage=0, task="B"), "B_s1": _round(False, stage=1, task="B")}  # fmt: skip
        inc = {"A_s0": _round(False, stage=1, task="A"), "A_s1": _round(False, stage=2, task="A"),
               "B_s0": _round(True, stage=4, task="B"), "B_s1": _round(False, stage=1, task="B")}  # fmt: skip
        t = self.gate.tally(cand, inc)
        self.assertEqual((t["b"], t["c"]), (2, 1))
        self.assertEqual(t["per_task"]["A"], dict(n=2, b=2, c=0, net=2, p_regress=1.0, p_holm=1.0, regress=False))
        self.assertEqual((t["per_task"]["B"]["b"], t["per_task"]["B"]["c"], t["per_task"]["B"]["p_regress"]),
                         (0, 1, 0.5))  # fmt: skip
        self.assertEqual(self.gate.verdict_fails(t), ["net wins 1 < 3", "losses 1 > wins 2 / 3"])
        self.assertEqual(self.gate.non_inferior_fails(t), [])
        self.assertIs(self.gate.mcnemar_one_sided, self.gate.V.sign_p)
        self.assertEqual(self.gate.drop_limit(t), 1)

    def _run(self, inc_ok: dict | None = None):
        """gate.main with a fake run_jobs: the candidate succeeds everywhere, the incumbent only on `inc_ok` labels."""
        inc_ok = inc_ok or {}
        B = self.gate.B
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
            for job in jobs:
                job.out.mkdir(parents=True, exist_ok=True)
                for x in job.labels or B.pool_labels(pool_):
                    ok = ("candidate" in str(job.out)) or x in inc_ok
                    (job.out / f"{x}.json").write_text(json.dumps(_round(ok=ok, rule=B.SUCCESS_RULE)))
            return {}

        real = B.run_jobs
        B.run_jobs = fake
        try:
            rc = self.gate.main(["f", "--gate", str(tmp / "gate"), "--pool", str(pool), "--candidate", str(cand)])
        finally:
            B.run_jobs = real
        return rc, [json.loads(x) for x in (tmp / "gate" / "ledger.jsonl").read_text().splitlines()]

    def test_the_ledger_row_reports_per_task_p_values(self):
        rc, led = self._run()
        self.assertEqual(rc, 0)
        row = led[-1]
        self.assertTrue(row["accepted"])
        per = row["result"]["per_task"]
        self.assertEqual(sorted(per), list("abcdefg"))
        self.assertEqual(per["a"], dict(n=6, b=6, c=0, net=6, p_regress=1.0, p_holm=1.0, regress=False))
        self.assertIn("per_task", row["stage1"])
        self.assertEqual(sum(v["b"] for v in row["stage1"]["per_task"].values()), row["stage1"]["b"])

    def test_stage_one_still_rejects_a_tie(self):
        """Every start a tie (both arms succeed): stage 1's net 0 rejects, through verdict.decide."""
        rc, led = self._run(inc_ok={f"{t}_s{i}" for t in "abcdefg" for i in range(6)})
        self.assertEqual(rc, 1)
        self.assertFalse(led[-1]["accepted"])
        self.assertEqual(led[-1]["reason"], "stage 1: net wins 0 <= 0 on 30 pairs")
        self.assertNotIn("result", led[-1])


if __name__ == "__main__":
    unittest.main()
