"""TODO 37.2 (user 2026-10-23: "B 到 E 按顺序做"): the per-mechanism table every role reads -- one row per hypothesis
so far with its probe status, the Manager's link, the drafts built from it, the Composer's and the gate's verdicts and
whether it is still open -- and the compact history as every role's default (run C's Reviewer read the whole ledger,
203-255 KB, and paged through it with scripts)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rsi_loop_ns as NS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scene_loop():
    names = ("gate", "batch", "family", "scene_round", "digest", "loop")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return NS.fresh_loop()
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


class MechanismTable(unittest.TestCase):
    def test_one_row_per_hypothesis_with_its_fate(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            st = Path(d)
            hyps = {
                0: [dict(id="H1", claim="narrow side to the jaw", instead="turn the approach", delta={"set": {"A": 1}}),
                    dict(id="H2", claim="stance nearer", instead="stance 0.4", delta={"set": {"B": 1}})],
                1: [dict(id="H1", claim="narrow side again", instead="turn", delta={"set": {"A": 2}}),
                    dict(id="H3", claim="needs code", instead="a recipe", delta={"set": {}})],
            }  # fmt: skip
            for n, hs in hyps.items():
                (st / f"iter_{n:02d}").mkdir()
                (st / f"iter_{n:02d}" / "hypotheses.json").write_text(json.dumps(dict(hypotheses=hs)))
            (st / "iter_01" / "candidates.json").write_text(json.dumps([
                dict(number=1, hypothesis="H3", fixed=["a", "b"], broken=[]),
                dict(number=2, hypothesis="H3", fixed=[], broken=["c"]),
            ]))  # fmt: skip
            es = [
                dict(step="review", iter=0, n_hypotheses=2),
                dict(step="probe", iter=0, results=[
                    dict(id="H1", n=30, ok_before=10, ok_after=10, fixed=[], broken=[], status="refuted"),
                    dict(id="H2", n=30, ok_before=10, ok_after=12, fixed=["x", "y"], broken=[], status="verified"),
                ]),
                dict(step="manager", iter=0, decision="engineer", hypothesis="H2", hypotheses=["H2"], reason="r"),
                dict(step="engineer", iter=0, candidate=1, hypothesis="H2", mechanism="m", void=False),
                dict(step="compose", iter=0, decision="adopt", adopted=1, notes="n"),
                dict(step="gate", iter=0, accepted=False, hash="h0", reason="p", result=dict(b=2, c=1)),
                dict(step="review", iter=1, n_hypotheses=2),
                dict(step="probe", iter=1, results=[
                    dict(id="H1", n=30, ok_before=10, ok_after=11, fixed=["z"], broken=[], status="verified"),
                    dict(id="H3", error="no rounds ran"),
                ]),
                dict(step="manager", iter=1, decision="engineer", hypotheses=["H3", "H3"], reason="r",
                     repeats=[dict(now="H1", same_as=["00:H1"])], skip={"H1": "changed_code_reached"}),
                dict(step="engineer", iter=1, candidate=1, hypothesis="H3", mechanism="m", void=False),
                dict(step="engineer", iter=1, candidate=2, hypothesis="H3", mechanism="m", void=False),
                dict(step="engineer", iter=1, candidate=3, hypothesis="H3", mechanism="m", void=True, late=True),
                dict(step="compose", iter=1, decision="merge", notes="n"),
                dict(step="gate", iter=1, accepted=True, hash="h1", reason="accepted", result=dict(b=4, c=0)),
            ]  # fmt: skip
            with mock.patch.object(lp, "STATE", st):
                rows = lp.mechanism_table(es)
        by = {(r["iter"], r["id"]): r for r in rows}
        self.assertEqual(sorted(by), [(0, "H1"), (0, "H2"), (1, "H1"), (1, "H3")])
        h1 = by[(0, "H1")]
        self.assertEqual((h1["status"], h1["probe"], h1["open"]), ("refuted", "10->10/30, fixed 0 broken 0", False))
        self.assertNotIn("built", h1)
        h2 = by[(0, "H2")]
        self.assertEqual((h2["status"], h2["built"], h2["composer"], h2["gate"], h2["open"]),
                         ("verified", "draft 1: built", "adopted", "rejected b 2 c 1", False))  # fmt: skip
        h1b = by[(1, "H1")]  # linked to 00:H1, skipped by the Manager, probed verified: open, never built
        self.assertEqual((h1b["same_as"], h1b["skip"], h1b["status"], h1b["open"]),
                         (["00:H1"], "changed_code_reached", "verified", True))  # fmt: skip
        self.assertNotIn("gate", h1b)
        h3 = by[(1, "H3")]
        self.assertEqual(h3["status"], "hypothesis")
        self.assertEqual(h3["probe"], "error: no rounds ran")
        self.assertEqual(h3["built"], "draft 1: fixed 2 broken 0; draft 2: fixed 0 broken 1; "
                                      "draft 3: late: stopped before it answered")  # fmt: skip
        self.assertEqual((h3["composer"], h3["gate"], h3["open"]), ("merge", "accepted b 4 c 0", False))

    def test_a_single_engineer_row_belongs_to_the_managers_hypothesis(self):
        # K = 1 rows carry no `hypothesis` (after_engineer); the Manager's choice names it
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            st = Path(d)
            (st / "iter_00").mkdir()
            (st / "iter_00" / "hypotheses.json").write_text(json.dumps(dict(hypotheses=[
                dict(id="H1", claim="c", instead="i", delta={"set": {"A": 1}}),
                dict(id="H2", claim="c2", instead="i2", delta={"set": {"A": 2}})])))  # fmt: skip
            es = [dict(step="manager", iter=0, decision="engineer", hypothesis="H2", reason="r"),
                  dict(step="engineer", iter=0, mechanism="m", void=False),
                  dict(step="gate", iter=0, accepted=False, hash="h", reason="p", result=dict(b=1, c=1))]  # fmt: skip
            with mock.patch.object(lp, "STATE", st):
                rows = {r["id"]: r for r in lp.mechanism_table(es)}
        self.assertEqual(
            (rows["H2"]["built"], rows["H2"]["gate"], rows["H2"]["open"]), ("built", "rejected b 1 c 1", False)
        )
        self.assertNotIn("built", rows["H1"])
        self.assertTrue(rows["H1"]["open"])

    def test_every_role_context_is_compact_and_carries_the_table(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            st = Path(d)
            es = []
            for n in range(4):
                (st / f"iter_{n:02d}").mkdir()
                (st / f"iter_{n:02d}" / "hypotheses.json").write_text(
                    json.dumps(dict(hypotheses=[dict(id="H1", claim="c")]))
                )
                es += [dict(t="t", step="review", iter=n, n_hypotheses=1, n_claims=0, usage={"x": 1}),
                       dict(step="gate", iter=n, accepted=False, hash=f"h{n}", reason="p", result=dict(b=1, c=0))]  # fmt: skip
            (st / "meta.json").write_text("{}")
            with (
                mock.patch.object(lp, "STATE", st),
                mock.patch.object(lp, "entries", lambda: es),
                mock.patch.object(lp, "run_gates", lambda es, meta: []),
                mock.patch.object(lp, "run_cost", lambda es, meta: 0.0),
            ):
                ctx = lp.context(dict(dir="x", hash="h", set={}), dict(extra=1))
        self.assertEqual(len(ctx["history"]), 1 + 2 * 2)  # one note on iterations 0 and 1, 2 and 3 row by row
        self.assertIsInstance(ctx["history"][0], str)
        self.assertEqual([r["iter"] for r in ctx["mechanisms"]], [0, 1, 2, 3])
        self.assertEqual(ctx["extra"], 1)


if __name__ == "__main__":
    unittest.main()
