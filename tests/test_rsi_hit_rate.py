"""TODO 56.4 (user 2026-11-27, main's TODO 48.19.3): scripts/rsi/hit_rate.py counts each role's funnel from a loop's
ledger and candidates.json; the loop writes hit_rate.json after every iteration and at a stop and `status` prints
the line; `alternate.py status` prints one line per family."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"
spec = importlib.util.spec_from_file_location("rsi_hit_rate_under_test", HERE / "hit_rate.py")
HR = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = HR
spec.loader.exec_module(HR)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _isolated(name: str, path: Path):
    names = ("gate", "batch", "family", "scene_round", "digest", "loop", "merge", "verdict", "hit_rate", "knowledge",
             "sandbox", "writeback")  # fmt: skip
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load(name, path)
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def probe(n, *results):
    return dict(step="probe", iter=n, results=[dict(id=f"h{i}", fixed=["x"] * f, broken=["y"] * b, status=s,
                                                    predicted_fixed_hit=ph, predicted_fixed_n=pn)
                                               for i, (f, b, s, ph, pn) in enumerate(results)])  # fmt: skip


def write_loop(loop: Path) -> None:
    rows = [
        dict(step="review", iter=0), probe(0, (3, 0, "verified", 2, 3), (0, 0, "null", 0, 0), (1, 2, "refuted", 0, 1)),
        dict(step="engineer", iter=0, candidate=1), dict(step="engineer", iter=0, candidate=2),
        dict(step="engineer", iter=0, candidate=3, void=True, leaks=["x"]),
        dict(step="compose", iter=0, decision="merge", merged=[1, 2], by="program", mode="merge"),
        dict(step="gate", iter=0, accepted=True, reason="accepted", result=dict(b=9, c=1), stage1=dict(b=5, c=0)),
        dict(step="review", iter=1), probe(1, (2, 1, "verified", 1, 2)),
        dict(step="engineer", iter=1, candidate=1), dict(step="engineer", iter=1, candidate=2, error="late", late=True),
        dict(step="compose", iter=1, decision="adopt", adopted=1),
        dict(step="gate", iter=1, accepted=False, reason="stage 1: net wins -1 <= 0 on 30 pairs", stage1=dict(b=2, c=3)),
        dict(step="gate", iter=1, shot=2, accepted=False, reason="net wins 2 < 3", result=dict(b=4, c=2)),
        dict(step="gate", iter=None, accepted=True, reason="accepted (non-inferior)", result=dict(b=1, c=1),
             sync={"from": "other"}),
        dict(step="gate", iter=2, accepted=False, void=True, reason="pairing incomplete: final lacks 1"),
        dict(step="gate", iter=2, accepted=False, dry_run=True, reason="dry run: static checks passed"),
        dict(step="gate", iter=2, accepted=False, reason="this exact candidate was rejected before"),
        dict(step="review", iter=2), dict(step="compose", iter=2, decision="keep", void=True, leaks=["x"]),
        dict(step="compose", iter=2, decision="keep"),
    ]  # fmt: skip
    loop.mkdir(parents=True, exist_ok=True)
    (loop / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (loop / "iter_00").mkdir()
    (loop / "iter_00" / "candidates.json").write_text(json.dumps([
        dict(number=1, fixed=["a", "b"], broken=[]), dict(number=2, fixed=["c"], broken=["d"])]))  # fmt: skip
    (loop / "iter_01").mkdir()
    (loop / "iter_01" / "candidates.json").write_text(json.dumps([dict(number=1, fixed=[], broken=["d"])]))


class Funnel(unittest.TestCase):
    def test_compute_counts_each_role(self):
        with tempfile.TemporaryDirectory() as d:
            loop = Path(d) / "loop"
            write_loop(loop)
            h = HR.compute(loop)
        r, e, c, g, s = h["reviewer"], h["engineer"], h["composer"], h["gate"], h["sync"]
        self.assertEqual((r["probed"], r["better"], r["same"], r["worse"], r["broke_any"]), (4, 2, 1, 1, 2))
        self.assertEqual(r["status"], {"verified": 2, "null": 1, "refuted": 1})
        self.assertEqual((r["predicted_fixed_hit"], r["predicted_fixed_n"], r["predicted_fixed_rate"]), (3, 6, 0.5))
        self.assertEqual((e["built"], e["void"], e["error"], e["late"]), (3, 1, 1, 1))
        self.assertEqual((e["screened"], e["diag_better"], e["diag_same"], e["diag_worse"], e["diag_broke_any"]),
                         (3, 1, 1, 1, 2))  # fmt: skip
        self.assertEqual(c, dict(decisions={"merge": 1, "adopt": 1, "keep": 1}, by={"program": 1, "llm": 2}))
        self.assertEqual(
            (g["calls"], g["stage1_passed"], g["accepted"], g["second_shots"], g["static"]), (3, 2, 1, 1, 1)
        )
        self.assertEqual((g["rejected_net_negative"], g["stage1_rate"], g["accept_rate"]), (1, 0.667, 0.333))
        self.assertEqual((s["calls"], s["accepted"]), (1, 1))
        self.assertEqual(h["per_iter"]["0"], dict(probed=3, probe_better=1, probe_same=1, probe_worse=1, drafts=2,
                                                 gate=1, stage1_passed=1, accepted=1))  # fmt: skip
        self.assertEqual(h["per_iter"]["1"]["gate"], 2)
        self.assertEqual(HR.line(h), "hit rate: Reviewer 4 probed (better 2, same 1, worse 1); Engineer 3 drafts "
                                     "built, 3 screened (better 1, broke a start 2); Composer adopt 1, keep 1, merge 1 "
                                     "(program 1, model 2); gate 1/3 accepted (2 past stage 1, 1 second shots); "
                                     "sync 1/1 accepted")  # fmt: skip

    def test_an_empty_or_missing_loop_counts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            h = HR.compute(Path(d) / "nothing")
        self.assertEqual((h["reviewer"]["probed"], h["engineer"]["built"], h["gate"]["calls"]), (0, 0, 0))
        self.assertIsNone(h["gate"]["accept_rate"])
        self.assertTrue(HR.line(h).startswith("hit rate: Reviewer 0 probed"))
        self.assertNotIn("sync", HR.line(h))

    def test_the_recorded_run(self):
        """gso-alt-seed (2026-11-25), when present on this machine: the funnel the user asked to see."""
        loop = Path.home() / "workspace/code/rsi-scene/gso-alt-seed/a/outputs/scene_rsi_20261016/gso_single_grasp/loop"
        if not (loop / "ledger.jsonl").exists():
            self.skipTest("gso-alt-seed is not on this machine")
        h = HR.compute(loop)
        self.assertEqual((h["gate"]["calls"], h["gate"]["accepted"], h["sync"]["calls"]), (5, 3, 1))
        self.assertEqual((h["engineer"]["built"], h["engineer"]["void"], h["composer"]["by"]), (13, 2, {"llm": 4}))


class LoopWrites(unittest.TestCase):
    def setUp(self):
        self.lp = _isolated("rsi_hit_rate_loop", HERE / "loop.py")
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        root = Path(self.d.name)
        (root / "pool_diag").mkdir()
        for p in (mock.patch.multiple(self.lp, STATE=root / "state", LOOP=root, HIDDEN=root / "hid",
                                      accept_pool=lambda: root, stop_reason=lambda m: None),
                  mock.patch.dict(os.environ)):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("RSI_MAX_WALL_H", None)

    def test_hit_rate_json_after_each_iteration_and_at_the_stop_and_the_status_line(self):
        lp = self.lp
        seen = []

        def iterate(n, meta, stub):
            lp.ledger(dict(step="review", iter=n))
            lp.ledger(dict(step="gate", iter=n, accepted=n == 1, reason="r", result=dict(b=4, c=0)))
            seen.append(json.loads((lp.STATE / "hit_rate.json").read_text())["gate"]["calls"] if n else None)
            return "manager: stop" if n == 1 else None

        with mock.patch.object(lp, "iterate", iterate):
            self.assertEqual(lp.cmd_run(Namespace(max_iters=3, reprobe=None, stub=True)), 0)
        self.assertEqual(seen, [None, 1])  # the file written after iteration 0 was read at iteration 1
        h = json.loads((lp.STATE / "hit_rate.json").read_text())
        self.assertEqual((h["gate"]["calls"], h["gate"]["accepted"]), (2, 1))
        self.assertEqual([e["step"] for e in lp.entries()][-1], "stop")
        import io
        from contextlib import redirect_stdout

        out = io.StringIO()
        with redirect_stdout(out), mock.patch.object(lp, "jload", lambda p, d=None: {} if d is not None else None):
            lp.cmd_status(None)
        self.assertIn("hit rate: Reviewer 0 probed (better 0, same 0, worse 0); Engineer 0 drafts built, 0 screened "
                      "(better 0, broke a start 0); Composer none (program 0, model 0); gate 1/2 accepted "
                      "(2 past stage 1, 0 second shots)", out.getvalue())  # fmt: skip


class AlternateStatus(unittest.TestCase):
    def test_status_prints_one_line_per_family_with_a_ledger(self):
        AL = _isolated("rsi_hit_rate_alternate", HERE / "alternate.py")
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            st = dict(families=["fa", "fb"], tags=["a", "b"], iter=0, turn=0, phase="loop", stopped={}, blocked=None,
                      events=[], active_s=0.0)  # fmt: skip
            (run / AL.STATE_FILE).write_text(json.dumps(st))
            (run / "knowledge").mkdir()
            write_loop(run / "a" / AL.VISIBLE / "fa" / "loop")
            shown = AL.status(run)
        self.assertIn("fa hit rate: Reviewer 4 probed", shown)
        self.assertNotIn("fb hit rate", shown)


if __name__ == "__main__":
    unittest.main()
