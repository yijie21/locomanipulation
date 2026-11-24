"""TODO 33.15-33.17 (user 2026-10-22: "把 33.15 到 33.18 先做了吧，开始下一轮的 RSI"; rules.md 31-33, version 15):
a hypothesis's status from its probe, the forced build of a repeated, unbuilt, unrefuted mechanism (and the Manager's
`skip`), the repair notes after a rejection, the gate's tally per failure class, the digest's class lead, and the run's
`--stop-rate`."""

import json
import os
import sys
import tempfile
import types
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


def _isolated(name: str, path: Path):
    """A scripts/rsi module under its own name (the warehouse loop has a loop.py, gate.py, batch.py too)."""
    names = ("gate", "batch", "family", "scene_round", "digest", "loop")
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


def _rec(label, ok, cls=None, **kw):
    r = dict(label=label, task=label.rsplit("_s", 1)[0], ok=ok, why="" if ok else "[goal] held(X): no hand touches it",
             fell=False, dropped=False, steps=[], **kw)  # fmt: skip
    if cls is not None:
        r["failure_class"] = cls
    return r


class _LoopCase(unittest.TestCase):
    def setUp(self):
        self.lp = _isolated("rsi_forced_build_loop", HERE / "loop.py")
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = Path(self.d.name)
        hid = self.root / "outputs" / "scene_rsi_20261016_hidden" / "fam"  # never HIDDEN at Path() (TODO 36)
        hid.mkdir(parents=True)
        (self.root / "state").mkdir()
        self.lp.FAMILY = "fam"
        for p in (mock.patch.multiple(self.lp, STATE=self.root / "state", LOOP=self.root, HIDDEN=hid,
                                      GATE_DIR=hid / "gate", transcript_leaks=lambda s: []),
                  mock.patch.dict(os.environ)):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)

    def hyps(self, n, *ids):
        d = self.lp.STATE / f"iter_{n:02d}"
        d.mkdir(exist_ok=True)
        hs = [dict(id=i, claim=f"claim {i}", instead="x", evidence="e", delta={"set": {}}, predicted_fixed=[],
                   predicted_broken=[]) for i in ids]  # fmt: skip
        self.lp.jdump(d / "hypotheses.json", dict(hypotheses=hs))
        return dict(hypotheses=hs)


class Status(_LoopCase):
    def test_status_from_the_probe(self):
        st = self.lp.probe_status
        h = dict(delta={"set": {"pick.X": 1}})
        self.assertEqual(st(h, dict(fixed=["a"], broken=[])), "verified")
        self.assertEqual(st(h, dict(fixed=[], broken=["a"])), "refuted")
        self.assertEqual(st(h, dict(fixed=[], broken=[])), "refuted")
        self.assertEqual(st(h, dict(fixed=["a"], broken=["b"])), "mixed")
        self.assertEqual(st(h, dict(error="no rounds ran")), "hypothesis")
        self.assertEqual(st(dict(delta={"set": {}}, needs_code="a new rule"), dict(fixed=[], broken=[])), "hypothesis")

    def test_statuses_land_on_hypotheses_and_probe_rows(self):
        hyps = dict(hypotheses=[dict(id="H1", delta={"set": {"a": 1}}), dict(id="H2", delta={"set": {}})])
        pr = [dict(id="H1", fixed=["x"], broken=[]), dict(id="H2", fixed=[], broken=[])]
        self.lp.set_statuses(hyps, pr)
        self.assertEqual([h["status"] for h in hyps["hypotheses"]], ["verified", "hypothesis"])
        self.assertEqual([r["status"] for r in pr], ["verified", "hypothesis"])


class ForcedBuild(_LoopCase):
    def history(self, latest_status="mixed"):
        """The dish soap's narrow side: proposed in iterations 3 (H2) and 5 (H1), never built; iteration 6's H3 is it."""
        lp = self.lp
        lp.ledger(dict(step="manager", iter=3, decision="engineer", hypotheses=["H1"]))
        lp.ledger(dict(step="manager", iter=5, decision="engineer", hypotheses=["H2"],
                       repeats=[dict(now="H1", same_as=["03:H2"])]))  # fmt: skip
        lp.ledger(dict(step="probe", iter=6, results=[dict(id="H3", fixed=["a"], broken=["b"], status=latest_status)]))
        lp.ledger(dict(step="manager", iter=6, decision="engineer", hypotheses=["H1", "H2", "H1"],
                       repeats=[dict(now="H3", same_as=["05:H1"])]))  # fmt: skip
        return self.hyps(6, "H1", "H2", "H3")

    def test_an_unbuilt_repeat_takes_the_last_slot(self):
        lp = self.lp
        hyps = self.history()
        dec = dict(decision="engineer", hypotheses=["H1", "H2", "H1"])
        with mock.patch.object(lp, "CANDIDATES", 3):
            out = lp.force_build(6, dec, hyps, lp.STATE / "iter_06")
        self.assertEqual(out["hypotheses"], ["H1", "H2", "H3"])
        row = lp.entries()[-1]
        self.assertEqual(
            (row["step"], row["hypothesis"], row["replaced"]), ("forced", "H3", None)
        )  # a free slot: nothing is replaced
        self.assertEqual(row["mechanism"], ["03:H2", "05:H1", "06:H3"])
        self.assertEqual(dec["hypotheses"], ["H1", "H2", "H1"])  # the Manager's own decision is kept as it was
        self.assertTrue((lp.STATE / "iter_06" / "manager_forced.json").exists())

    def test_a_short_list_is_filled_before_the_last_slot_is_taken(self):
        lp = self.lp
        hyps = self.history()
        with mock.patch.object(lp, "CANDIDATES", 3):
            out = lp.force_build(6, dict(decision="engineer", hypotheses=["H1"]), hyps, lp.STATE / "iter_06")
        self.assertEqual(
            out["hypotheses"], ["H1", "H2", "H3"]
        )  # no id twice: the short list is topped up, then H3 takes the last slot

    def test_one_engineer_gets_the_forced_hypothesis(self):
        lp = self.lp
        hyps = self.history()
        with mock.patch.object(lp, "CANDIDATES", 1):
            out = lp.force_build(6, dict(decision="engineer", hypothesis="H1"), hyps, lp.STATE / "iter_06")
        self.assertEqual(out["hypothesis"], "H3")

    def test_a_named_failing_check_skips_it(self):
        lp = self.lp
        hyps = self.history()
        dec = dict(decision="engineer", hypotheses=["H1", "H2", "H1"], skip={"H3": "changed_code_reached"})
        with mock.patch.object(lp, "CANDIDATES", 3):
            out = lp.force_build(6, dec, hyps, lp.STATE / "iter_06")
        self.assertEqual(out["hypotheses"], ["H1", "H2", "H1"])
        self.assertEqual(lp.entries()[-1]["step"], "forced_skip")

    def test_a_skip_without_a_valid_check_does_not(self):
        lp = self.lp
        hyps = self.history()
        dec = dict(decision="engineer", hypotheses=["H1", "H2", "H1"], skip={"H3": "not now"})
        with mock.patch.object(lp, "CANDIDATES", 3):
            out = lp.force_build(6, dec, hyps, lp.STATE / "iter_06")
        self.assertEqual(out["hypotheses"][-1], "H3")

    def test_a_refuted_mechanism_is_not_forced(self):
        lp = self.lp
        hyps = self.history(latest_status="refuted")
        dec = dict(decision="engineer", hypotheses=["H1", "H2", "H1"])
        with mock.patch.object(lp, "CANDIDATES", 3):
            out = lp.force_build(6, dec, hyps, lp.STATE / "iter_06")
        self.assertEqual(out, dec)
        self.assertNotIn(lp.entries()[-1]["step"], ("forced", "forced_skip"))

    def test_a_mechanism_the_manager_assigns_or_built_before_is_not_forced(self):
        lp = self.lp
        hyps = self.history()
        with mock.patch.object(lp, "CANDIDATES", 3):
            lp.ledger(dict(step="manager", iter=6, decision="engineer", hypotheses=["H3", "H1", "H2"]))
            out = lp.force_build(6, dict(decision="engineer", hypotheses=["H3", "H1", "H2"]), hyps, lp.STATE)
        self.assertEqual(out["hypotheses"], ["H3", "H1", "H2"])
        self.assertNotEqual(lp.entries()[-1]["step"], "forced")

    def test_only_engineer_decisions_are_changed(self):
        lp = self.lp
        hyps = self.history()
        dec = dict(decision="reopen", reopen_claim="C1")
        self.assertIs(lp.force_build(6, dec, hyps, lp.STATE / "iter_06"), dec)

    def test_the_manager_schema_takes_a_skip_with_the_four_checks(self):
        sch = self.lp.manager_schema(True)["properties"]["skip"]
        self.assertEqual(sch["additionalProperties"]["enum"], list(self.lp.SKIP_CHECKS))
        self.assertIn("instead", self.lp.HYPOTHESES_SCHEMA["properties"]["hypotheses"]["items"]["required"])


class RepairNotes(_LoopCase):
    def diag(self):
        d = self.root / "diag"
        d.mkdir()
        for r in (_rec("pick_A_s1", False, "too_wide"), _rec("pick_A_s2", False, "too_wide"),
                  _rec("pick_B_s1", False, "slip"), _rec("pick_B_s2", True), _rec("pick_C_s1", False)):  # fmt: skip
            (d / f"{r['label']}.json").write_text(json.dumps(r))
        return d

    def test_after_a_rejection_the_classes_the_gate_tally_and_suggestions(self):
        lp = self.lp
        per = {"too_wide": dict(b=1, c=0, cand_fail=1, inc_fail=2)}
        es = [dict(step="gate", iter=8, accepted=False, reason="net wins 2 < 3", result=dict(b=2, c=0, per_class=per))]
        notes = lp.repair_notes(es, self.diag(), 9)
        self.assertIn("rejected", notes["trigger"])
        self.assertEqual(notes["diag_failures_by_class"], {"too_wide": 2, "slip": 1, "unknown": 1})
        self.assertEqual(notes["gate_by_class"], per)
        self.assertEqual(set(notes["suggestions"]), {"too_wide", "slip"})

    def test_a_kept_incumbent_triggers_them_an_acceptance_does_not(self):
        lp = self.lp
        d = self.diag()
        kept = [dict(step="compose", iter=8, decision="keep")]
        self.assertIn("kept the incumbent", lp.repair_notes(kept, d, 9)["trigger"])
        acc = [dict(step="gate", iter=8, accepted=True, reason="accepted", result=dict(b=4, c=0))]
        self.assertIsNone(lp.repair_notes(acc, d, 9))
        self.assertIsNone(lp.repair_notes([], d, 9))


class GateAndDigest(unittest.TestCase):
    def test_gate_tally_by_failure_class(self):
        gate = _isolated("rsi_forced_build_gate", HERE / "gate.py")
        inc = {"a": _rec("a", False, "too_wide"), "b": _rec("b", True), "c": _rec("c", False, "slip"),
               "d": _rec("d", False)}  # fmt: skip
        cand = {"a": _rec("a", True), "b": _rec("b", False, "walk_off"), "c": _rec("c", False, "slip"),
                "d": _rec("d", True)}  # fmt: skip
        t = gate.tally(cand, inc)
        self.assertEqual((t["b"], t["c"]), (2, 1))
        self.assertEqual(t["per_class"]["too_wide"], dict(b=1, c=0, cand_fail=0, inc_fail=1))
        self.assertEqual(t["per_class"]["walk_off"], dict(b=0, c=1, cand_fail=1, inc_fail=0))
        self.assertEqual(t["per_class"]["slip"], dict(b=0, c=0, cand_fail=1, inc_fail=1))
        self.assertEqual(t["per_class"]["unknown"], dict(b=1, c=0, cand_fail=0, inc_fail=1))
        self.assertFalse(gate.verdict_fails(dict(t, b=4, c=0)))  # the class tally does not judge

    def test_digest_leads_with_classes_and_lists_the_unplaced(self):
        dg = _isolated("rsi_forced_build_digest", HERE / "digest.py")
        rounds = [_rec("pick_A_s1", False, "too_wide"), _rec("pick_A_s2", False, "other"), _rec("pick_B_s1", True),
                  _rec("pick_B_s2", False)]  # fmt: skip
        lines = dg.class_lines(rounds)
        self.assertTrue(lines[0].startswith("Failure classes: "))
        for k in ("too_wide 1", "other 1", "unknown 1"):
            self.assertIn(k, lines[0])
        listed = "\n".join(lines)
        self.assertIn("pick_A_s2", listed)
        self.assertIn("pick_B_s2", listed)
        self.assertNotIn("pick_A_s1:", listed)
        self.assertEqual(dg.class_lines([_rec("x", True)]), [])


class StopRate(unittest.TestCase):
    def setUp(self):
        self.lp = _isolated("rsi_forced_build_loop_sr", HERE / "loop.py")

    def args(self, **kw):
        return types.SimpleNamespace(candidates=None, scoped_review=None, stop_rate=None, **kw)

    def test_cli_then_meta_on_resume_then_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RSI_MAX_WALL_H", None)
            rs = self.lp.resolve_settings
            self.assertEqual(rs(self.args(), {}, False)["stop_rate"], 0.70)
            self.assertEqual(rs(self.args(), {"stop_rate": 0.9}, True)["stop_rate"], 0.9)
            self.assertEqual(rs(self.args(), {"stop_rate": 0.9}, False)["stop_rate"], 0.70)
            a = self.args()
            a.stop_rate = 0.85
            self.assertEqual(rs(a, {"stop_rate": 0.9}, True)["stop_rate"], 0.85)

    def test_a_resume_records_a_changed_rate_even_where_meta_has_none(self):
        ch = self.lp.settings_changes
        cfg = dict(candidates=3, scoped_review=False, max_wall_h=5.0, stop_rate=0.9)
        self.assertEqual(ch(cfg, dict(candidates=3, max_wall_h=5.0), True), {"stop_rate": dict(old=0.70, new=0.9)})
        self.assertEqual(ch(cfg, dict(stop_rate=0.9, candidates=3), True), {})
        self.assertEqual(ch(cfg, {}, False), {})

    def test_the_flag_parses(self):
        ap_ok = {}

        def fake_run(a):
            ap_ok["stop_rate"] = a.stop_rate
            return 0

        with mock.patch.object(self.lp, "cmd_run", fake_run), mock.patch.object(self.lp, "configure", lambda f: None):
            self.lp.main(["run", "fam", "--stop-rate", "0.85"])
        self.assertEqual(ap_ok["stop_rate"], 0.85)


if __name__ == "__main__":
    unittest.main()
