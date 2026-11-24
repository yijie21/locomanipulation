"""2026-11-20: the six changes to the RSI loop -- program search for parameters (sweeps, tuned drafts), the Reviewer's
working files for the Engineers, no duplicate assignments, drafts that run where they act, the next review while the
gate runs, and the straggler deadline counted from the session's end."""

import json
import shutil
import stat
import sys
import tempfile
import threading
import time
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


def _scene_loop():
    names = ("gate", "batch", "family", "scene_round", "digest", "loop")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load("scene_rsi_loop_program_search", HERE / "loop.py")
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _put_rounds(d: Path, ok: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for k, v in ok.items():
        (d / f"{k}.json").write_text(json.dumps(dict(ok=bool(v), task=k.rsplit("_s", 1)[0], why="")))


LABELS = [f"A_s{i}" for i in range(6)] + [f"B_s{i}" for i in range(6)]


class SweepPure(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()

    def test_specs_are_capped_deduplicated_and_need_two_values(self):
        lp = self.lp
        h = dict(sweep=[dict(key="K", values=[1, 1, 2, 3, 4, 5, 6]), dict(key="K", values=[9, 8]),
                        dict(key="L", values=[1]), dict(key="M", values=[0.1, 0.2]), dict(key="N", values=[1, 2])])  # fmt: skip
        self.assertEqual(
            lp.sweep_specs(h), [dict(key="K", values=[1, 2, 3, 4, 5])]
        )  # 2 items taken, K twice, L one value
        self.assertEqual(lp.sweep_specs(dict(sweep=[dict(key="M", values=[0.1, 0.2]), dict(key="N", values=[1, 2])])),
                         [dict(key="M", values=[0.1, 0.2]), dict(key="N", values=[1, 2])])  # fmt: skip
        self.assertEqual(lp.sweep_specs({}), [])

    def test_best_value_is_the_largest_net_then_fewer_broken_then_earlier(self):
        lp = self.lp
        row = lambda net, broken: dict(net=net, broken=broken)  # noqa: E731
        self.assertEqual(lp.best_row([row(1, 0), row(3, 2), row(2, 0)]), 1)
        self.assertEqual(lp.best_row([row(3, 2), row(3, 1), row(3, 1)]), 1)
        self.assertEqual(lp.best_row([row(0, 0), row(0, 0)]), 0)

    def test_sweep_row_counts_fixed_and_broken(self):
        lp = self.lp
        before = dict(a=False, b=True, c=True, d=False)
        after = dict(a=True, b=False, c=True, d=False, e=True)  # e: not in the incumbent's rounds, counts as a fix
        r = lp.sweep_row("K", 2, before, after)
        self.assertEqual((r["fixed"], r["broken"], r["net"], r["n"]), (2, 1, 1, 5))

    def test_subsample_takes_failures_first_up_to_two_thirds_then_successes(self):
        lp = self.lp
        labels = [f"s{i:02d}" for i in range(12)]
        before = {x: i % 3 == 0 for i, x in enumerate(labels)}  # successes s00 s03 s06 s09
        got = lp.subsample_starts(labels, before, 6)
        lose = [x for x in labels if not before[x]]
        self.assertEqual(got, sorted(lose[:4] + ["s00", "s03"]))  # ceil(12/3) = 4 failures, then 2 successes
        self.assertEqual(lp.subsample_starts(labels, before, 6), got)  # deterministic
        self.assertEqual(lp.subsample_starts(labels, before, 50), labels)
        few_wins = {x: x == "s00" for x in labels}  # one success only: filled up with more failures
        got = lp.subsample_starts(labels, few_wins, 6)
        self.assertEqual((len(got), "s00" in got), (6, True))

    def test_budget_reserves_at_most_what_is_left(self):
        b = self.lp.SweepBudget(10)
        self.assertEqual((b.reserve(6), b.reserve(6), b.left), (6, 4, 0))


class ProbeCase(unittest.TestCase):
    """probe() on fake rounds: a job's rounds are written from `rule(overrides, label)`."""

    def setUp(self):
        self.lp = _scene_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _put_rounds(self.root / "pool_diag", {x: True for x in LABELS})
        self.base = self.root / "base"
        _put_rounds(self.base, {x: x.startswith("A_") and int(x[-1]) < 3 for x in LABELS})  # A0-A2 won, rest lost
        self.inc = dict(dir=str(self.root / "inc"), hash="inc", set={"X": 1})
        self.jobs = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_probe(self, hyps, rule, caps=None):
        lp = self.lp

        def fake_run_jobs(jobs, family, pool, procs, log, gpus, **k):
            for j in jobs:
                self.jobs.append(j)
                every = lp.B.pool_labels(pool)
                _put_rounds(j.out, {x: rule(j.overrides, x) for x in (j.labels or every)})
            return {}

        with (
            mock.patch.multiple(
                lp, LOOP=self.root, FAMILY="fam", CAPS={**lp.CAPS, "sweep_rounds": 1200, **(caps or {})}
            ),
            mock.patch.object(lp.B, "run_jobs", fake_run_jobs),
        ):
            return lp.probe(hyps, self.inc, self.base, self.root / "it")

    @staticmethod
    def hyp(id_, **kw):
        return {**dict(id=id_, delta=dict(set={}), predicted_fixed=[], predicted_broken=[]), **kw}

    def test_needs_code_is_not_run_and_stays_untested(self):
        lp = self.lp
        res = self.run_probe([self.hyp("H1", needs_code="a new step", delta=dict(set={"K": 1}))], lambda o, x: True)
        self.assertEqual(res, [dict(id="H1", status="untested", reason="needs code")])
        self.assertEqual(self.jobs, [])
        hyps = dict(hypotheses=[self.hyp("H1", needs_code="a new step", delta=dict(set={"K": 1}))])
        lp.set_statuses(hyps, res)
        self.assertEqual((hyps["hypotheses"][0]["status"], res[0]["status"]), ("untested", "untested"))

    def test_a_hypothesis_without_sweep_runs_once_and_its_best_set_is_its_delta(self):
        res = self.run_probe([self.hyp("H1", delta=dict(set={"K": 5}))], lambda o, x: o.get("K") == 5)
        self.assertEqual(len(self.jobs), 1)
        r = res[0]
        self.assertEqual(r["best_set"], {"K": 5})
        self.assertNotIn("sweep", r)
        self.assertEqual((len(r["fixed"]), len(r["broken"])), (9, 0))

    def test_sweep_runs_one_job_per_value_then_the_second_key_at_the_best_first(self):
        # K=2 fixes all of B and keeps A; K=3 fixes B too but breaks A0; L only matters with K=2
        def rule(o, x):
            if o.get("K") == 2:
                return x.startswith("B_") or (x.startswith("A_") and (o.get("L") != 9 or x != "A_s0"))
            if o.get("K") == 3:
                return x.startswith("B_") and int(x[-1]) < 2
            return o.get("K") == 1 and x == "A_s0"

        h = self.hyp(
            "H1", delta=dict(set={"D": 7}), sweep=[dict(key="K", values=[1, 2, 3]), dict(key="L", values=[8, 9])]
        )
        res = self.run_probe([h], rule)
        self.assertEqual(len(self.jobs), 3 + 2)
        self.assertTrue(
            all(j.overrides["X"] == 1 and j.overrides["D"] == 7 for j in self.jobs)
        )  # incumbent set + delta
        self.assertTrue(all(j.overrides.get("K") == 2 for j in self.jobs[3:]))  # the first key held at its best
        r = res[0]
        self.assertEqual(r["best_set"], {"D": 7, "K": 2, "L": 8})
        self.assertEqual([(x["key"], x["value"], x["net"]) for x in r["sweep"]],
                         [("K", 1, -2), ("K", 2, 9), ("K", 3, -1), ("L", 8, 9), ("L", 9, 8)])  # fmt: skip
        self.assertEqual((r["ok_before"], r["ok_after"]), (3, 12))
        self.assertTrue(r["best_dir"].endswith("H1__s2_0"))

    def test_a_second_key_that_only_hurts_is_left_out(self):
        h = self.hyp("H1", sweep=[dict(key="K", values=[1, 2]), dict(key="L", values=[1, 2])])
        res = self.run_probe([h], lambda o, x: o.get("K") == 2 and "L" not in o)
        self.assertEqual(res[0]["best_set"], {"K": 2})
        self.assertTrue(res[0]["second_key_dropped"])

    def test_values_times_starts_over_the_share_subsample_the_starts(self):
        h1 = self.hyp("H1", sweep=[dict(key="K", values=[1, 2, 3])])
        h2 = self.hyp("H2", sweep=[dict(key="K", values=[1, 2, 3])])
        res = self.run_probe(
            [h1, h2], lambda o, x: o.get("K") == 2, caps=dict(sweep_rounds=36)
        )  # share 18, 3 values: m = 6
        self.assertEqual(len(self.jobs), 6)
        for r in res:
            self.assertTrue(r["subsampled"])
            self.assertEqual(len(r["starts"]), 6)
            # ceil(12/3) = 4 failures, then the incumbent's successes in label order
            self.assertEqual(r["starts"], sorted(["A_s3", "A_s4", "A_s5", "B_s0", "A_s0", "A_s1"]))
            self.assertEqual(r["n"], 6)
        self.assertEqual(res[0]["starts"], res[1]["starts"])

    def test_probe_scope_and_needs_code_beside_a_sweep(self):
        lp = self.lp
        with mock.patch.dict(lp.CAPS, {"probe_scope": True}):
            h = self.hyp("H1", sweep=[dict(key="K", values=[1, 2])], tasks=["B"])
            res = self.run_probe([h, self.hyp("H2", needs_code="x")], lambda o, x: o.get("K") == 2)
        self.assertEqual(res[0]["scope"], ["B"])
        self.assertEqual(res[0]["ok_before"], 0)  # counted over its own starts
        self.assertEqual(res[1]["status"], "untested")


class AssignCase(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()

    @staticmethod
    def hs(*specs):
        return [dict(id=i, tasks=t, status=s) for i, t, s in specs]

    def test_duplicates_are_dropped_and_never_cycled(self):
        lp = self.lp
        hyps = self.hs(("H1", ["a"], "verified"), ("H2", ["a"], "verified"))
        ids, dups = lp.assign_ids(dict(hypotheses=["H1", "H1", "H1"]), hyps, 3)
        self.assertEqual((ids, dups), (["H1"], 2))  # H2 overlaps H1's task: no top-up

    def test_top_up_takes_unrefuted_untuned_non_overlapping_hypotheses_in_reviewer_order(self):
        lp = self.lp
        hyps = self.hs(("H1", ["a"], "verified"), ("H2", ["a"], "verified"), ("H3", ["b"], "refuted"),
                       ("H4", ["c"], "hypothesis"), ("H5", ["d"], "untested"), ("H6", [], "hypothesis"))  # fmt: skip
        ids, dups = lp.assign_ids(dict(hypotheses=["H1", "H1"]), hyps, 3, tune=["H5"])
        self.assertEqual((ids, dups), (["H1", "H4", "H6"], 1))
        ids, _ = lp.assign_ids(dict(hypotheses=["H1", "H4", "H6", "H2"]), hyps, 3)
        self.assertEqual(ids, ["H1", "H4", "H6"])  # never more than K

    def test_empty_hypotheses_with_tune_run_no_engineer_and_without_it_fall_back(self):
        lp = self.lp
        hyps = self.hs(("H1", [], "verified"), ("H2", [], "verified"))
        self.assertEqual(lp.assign_ids(dict(hypotheses=[]), hyps, 3, tune=["H1"]), ([], 0))
        self.assertEqual(lp.assign_ids(dict(hypotheses=[]), hyps, 3)[0], ["H1", "H2"])  # first one, topped up

    def test_manager_schema_allows_an_empty_list_and_names_tune(self):
        sch = self.lp.manager_schema(True, ["H1", "H2"])
        self.assertEqual(sch["properties"]["tune"]["items"]["enum"], ["H1", "H2"])
        self.assertNotIn("minItems", sch["properties"]["hypotheses"])


class DraftLabels(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.pool = [f"{t}_s{i}" for t in ("A", "B", "C") for i in range(5)]
        # the incumbent won B_s1, B_s3, C_s0 and nothing else
        self.ok = dict(B_s1=True, B_s3=True, C_s0=True)

    def test_all_empty_or_unknown_starts_run_the_whole_pool(self):
        lp = self.lp
        for declared in (["all"], [], None, ["Z_s1"]):
            labels, ran = lp.draft_labels(self.pool, declared, self.ok, 3)
            self.assertIsNone(labels)
            self.assertEqual((ran["n"], ran["tasks"], ran["sentinels"]), (15, ["A", "B", "C"], []))

    def test_declared_tasks_run_whole_and_other_tasks_give_sentinels_winners_first(self):
        lp = self.lp
        labels, ran = lp.draft_labels(self.pool, ["A_s2"], self.ok, 3)
        self.assertEqual(ran["tasks"], ["A"])
        self.assertEqual(ran["sentinels"], ["B_s1", "B_s3", "B_s0", "C_s0", "C_s1", "C_s2"])
        self.assertEqual(ran["n"], len(labels))
        self.assertEqual(labels, tuple(sorted([f"A_s{i}" for i in range(5)] + ran["sentinels"])))
        labels2, ran2 = lp.draft_labels(self.pool, ["A_s2", "C_s4"], self.ok, 1)
        self.assertEqual((ran2["tasks"], ran2["sentinels"]), (["A", "C"], ["B_s1"]))

    def test_footprint_counts_only_starts_both_ran_and_reports_not_run(self):
        lp = self.lp
        ref, cand = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        for x in self.pool:
            (ref / f"{x}.json").write_text(json.dumps(dict(ok=True, steps=[], t_end=1)))
        for x in self.pool[:5]:
            (cand / f"{x}.json").write_text(json.dumps(dict(ok=False, steps=[], t_end=1)))
        fp = lp.footprint(ref, cand, dict(starts=["A_s1"]))
        self.assertEqual((fp["n"], len(fp["changed"]), fp["outside"]), (5, 5, ["A_s0", "A_s2", "A_s3", "A_s4"]))
        self.assertEqual(fp["not_run"], sorted(self.pool[5:]))

    def test_a_draft_flips_compare_only_starts_present_in_both(self):
        lp = self.lp
        a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        _put_rounds(a, dict(A_s0=True, A_s1=False, B_s0=True))
        _put_rounds(b, dict(A_s0=False, A_s1=True))
        f = lp.flips(a, b)
        self.assertEqual((f["n"], f["fixed"], f["broken"]), (2, ["A_s1"], ["A_s0"]))


class TunedDrafts(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _put_rounds(self.root / "pool_diag", {x: True for x in LABELS})
        _put_rounds(self.root / "diag" / "incumbent_inc", {x: x == "A_s0" for x in LABELS})
        self.src = self.root / "inc"
        (self.src / "skills").mkdir(parents=True)
        (self.src / "skills" / "__init__.py").write_text("X = 1\n")
        (self.src / "candidate.json").write_text('{"set": {"X": 1}}')
        self.inc = dict(dir=str(self.src), hash="inc", set={"X": 1})
        self.it = self.root / "it"
        self.it.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_tuned_draft_copies_the_probe_rounds_and_runs_the_rest(self):
        lp = self.lp
        best = self.root / "probe_best"
        _put_rounds(best, {x: True for x in LABELS if x.startswith("B_")})
        rows, jobs = [], []
        known = dict(H1=dict(id="H1", tasks=["B"]), H2=dict(id="H2", needs_code="x"), H3=dict(id="H3"))
        pr = [dict(id="H1", fixed=["B_s1", "B_s2"], broken=["A_s0"], best_set={"K": 4}, best_dir=str(best)),
              dict(id="H3", fixed=["B_s1"], broken=["B_s2"], best_set={"K": 5})]  # fmt: skip
        with mock.patch.multiple(
            lp, LOOP=self.root, ledger=rows.append, CAPS={**lp.CAPS, "draft_sentinels": 2}, FAMILY="fam"
        ):
            with mock.patch.object(
                lp,
                "probe_labels",
                lambda h: tuple(x for x in LABELS if x.startswith("B_")) if h["id"] == "H1" else None,
            ):
                out = lp.tuned_drafts(2, self.it, self.inc, known, pr, ["H1", "H2", "H3"], self.src, None, 3, {}, jobs)
        self.assertEqual(len(out), 1)  # H2 needs code, H3's net is 0
        c = out[0]
        self.assertEqual((c["kind"], c["number"], c["hypothesis"]), ("tuned", 3, "H1"))
        self.assertEqual(c["mechanism"], "program sweep: K=4")
        self.assertEqual(json.loads((Path(c["dir"]) / "candidate.json").read_text()), {"set": {"X": 1, "K": 4}})
        self.assertEqual(c["ran"]["tasks"], ["B"])
        self.assertEqual(len(c["ran"]["sentinels"]), 2)  # task A is the only other task: 2 starts
        copied = {p.stem for p in Path(c["diag"]).glob("*.json")}
        self.assertEqual(copied, {x for x in LABELS if x.startswith("B_")})  # the probe's rounds, copied
        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            sorted(jobs[0].labels), sorted(c["ran"]["sentinels"] + [x for x in LABELS if x.startswith("B_")])
        )
        self.assertEqual([r.get("skipped") for r in rows], [None, "needs_code", "probe net 0 < 1"])

    def test_engineer_tunables_pick_the_best_value_and_move_the_candidate(self):
        lp = self.lp
        w = self.it / "cand_1"
        shutil.copytree(self.src / "skills", w / "skills")
        cand = self.it / "candidate_1"
        shutil.copytree(self.src / "skills", cand / "skills")
        (cand / "candidate.json").write_text('{"set": {"X": 1}}')
        h0 = lp.candidate_hash(cand)
        _put_rounds(self.root / "diag" / f"candidate_{h0}", {x: x == "A_s0" or x == "B_s0" for x in LABELS})
        eng = dict(mechanism="m", set={"X": 1}, scope=dict(starts=["A_s0"]),
                   tunable=[dict(key="T", values=[1, 2, 3]), dict(key="Bad", values=[1, 2]), dict(key="U", values=[5, 6])])  # fmt: skip
        calls = []

        def fake_run_jobs(jobs, family, pool, procs, log, gpus, **k):
            for j in jobs:
                calls.append((j.overrides, len(j.labels)))
                _put_rounds(j.out, {x: x == "A_s0" or (j.overrides.get("T") == 2 and x == "A_s1") for x in j.labels})
            return {}

        with (
            mock.patch.multiple(
                lp,
                LOOP=self.root,
                FAMILY="fam",
                CAPS={**lp.CAPS, "draft_sentinels": 2, "sweep_rounds": 1200},
                unknown_keys=lambda sk, s, items: ["Bad"],
                early_diag=lambda *a: calls.append("early"),
            ),
            mock.patch.object(lp.B, "run_jobs", fake_run_jobs),
        ):
            lp.SWEEP = lp.SweepBudget(1200)
            out = lp.tune_draft(1, w, eng, self.inc, self.it)
        self.assertEqual(out["tunable_dropped"], ["Bad"])
        self.assertEqual(out["tuned"], {"T": 2})  # U changes nothing: not adopted
        self.assertEqual(out["set"], {"X": 1, "T": 2})
        self.assertEqual(json.loads((cand / "candidate.json").read_text()), {"set": {"X": 1, "T": 2}})
        self.assertEqual(calls[-1], "early")
        table = {t["key"]: t for t in out["tunable_sweep"]}
        self.assertEqual(table["T"]["adopted"], 2)
        self.assertNotIn("adopted", table["U"])
        self.assertTrue(all(n == 6 for _, n in calls[:-1]))  # the 6 starts of task A, sentinels left out
        h1 = lp.candidate_hash(cand)
        self.assertNotEqual(h0, h1)
        self.assertTrue((self.root / "diag" / f"candidate_{h1}" / "A_s1.json").exists())  # the best variant's rounds


class ReviewNotes(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_new_small_text_files_of_the_top_level_are_copied_with_an_index(self):
        lp = self.lp
        tmp = self.root / "tmp"
        (tmp / "engineer_1").mkdir(parents=True)
        (tmp / "old.py").write_text("old")
        t0 = time.time() + 0.5
        time.sleep(0.6)
        (tmp / "probe.py").write_text("print(1)")
        (tmp / "notes.md").write_text("# n")
        (tmp / "big.txt").write_text("x" * (lp.NOTE_MAX_FILE + 1))
        (tmp / "pic.png").write_bytes(b"x")
        (tmp / "engineer_1" / "mine.py").write_text("other role")
        out = lp.copy_review_notes(self.root / "it", [("", tmp, False)], t0)
        names = sorted(p.name for p in (self.root / "it" / "review_notes").iterdir())
        self.assertEqual(names, ["INDEX.md", "notes.md", "probe.py"])
        self.assertEqual(out["files"], 2)
        idx = (self.root / "it" / "review_notes" / "INDEX.md").read_text()
        self.assertIn("`probe.py` (8 bytes)", idx)

    def test_a_total_over_the_limit_drops_the_largest_first_and_joint_reviewers_are_tagged(self):
        lp = self.lp
        d = self.root / "reviewer_A"
        d.mkdir()
        t0 = time.time() - 1
        for i, size in enumerate((200_000, 150_000, 100_000)):
            (d / f"f{i}.txt").write_text("x" * size)
        with mock.patch.object(lp, "NOTE_MAX_ALL", 300_000):
            out = lp.copy_review_notes(self.root / "it", [("A", d, True)], t0)
        got = sorted(p.name for p in (self.root / "it" / "review_notes" / "A").iterdir())
        self.assertEqual(got, ["f1.txt", "f2.txt"])
        self.assertEqual((out["files"], out["skipped"]), (2, 1))
        self.assertEqual(lp.review_notes_dir(self.root / "it2").joinpath("INDEX.md").read_text().count("(none)"), 1)


class PrivateGate(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.hidden = self.root / "scene_rsi_20261016_hidden" / "fam"
        (self.hidden / "gate" / "01_abc").mkdir(parents=True)
        (self.hidden / "pool_accept").mkdir()
        (self.hidden / "pool_accept" / "A_s0.json").write_text("{}")
        (self.hidden / "practice").mkdir()
        (self.hidden / "practice" / "A.jsonl").write_text("{}\n")
        (self.hidden / "seeds.json").write_text("{}")
        g = self.hidden / "gate"
        (g / "incumbent.json").write_text(json.dumps(dict(dir=str(g / "incumbent_seed"), rounds=str(g / "r"), since=0)))
        (g / "ledger.jsonl").write_text(json.dumps(dict(hash="old", accepted=False)) + "\n")
        (g / "01_abc" / "round.json").write_text(json.dumps(dict(path=str(g / "x"))))
        (g / "01_abc" / "blob.bin").write_bytes(b"\x00" + str(self.hidden).encode())  # binary: not rewritten
        (g / "_rule1_x.json").write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_copy_names_only_its_own_paths_and_the_round_trip_returns_the_originals(self):
        lp = self.lp
        before = {
            p.relative_to(self.hidden).as_posix(): p.read_bytes()
            for p in (self.hidden / "gate").rglob("*")
            if p.is_file()
        }
        p, manifest = lp.private_gate_copy(self.hidden, self.hidden / "pool_accept")
        self.assertEqual(p.parent, self.hidden.parent)
        self.assertRegex(p.name, r"^\.[0-9a-f]{24}$")
        self.assertTrue((p / "pool_accept" / "A_s0.json").exists() and (p / "practice" / "A.jsonl").exists())
        inc = json.loads((p / "gate" / "incumbent.json").read_text())
        self.assertTrue(inc["dir"].startswith(str(p)) and inc["rounds"].startswith(str(p)))
        self.assertNotIn(str(self.hidden), (p / "gate" / "01_abc" / "round.json").read_text())
        # the gate: appends a ledger line, writes a new round, retires one, moves the incumbent
        with (p / "gate" / "ledger.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(hash="new", accepted=True, dir=str(p / "gate" / "02_new"))) + "\n")
        (p / "gate" / "02_new").mkdir()
        (p / "gate" / "02_new" / "r.json").write_text(json.dumps(dict(at=str(p / "gate" / "02_new"))))
        (p / "gate" / "_rule1_x.json").unlink()
        (p / "gate" / "_rule1_y.json").write_text("{}")
        (p / "gate" / "incumbent.json").write_text(json.dumps(dict(dir=str(p / "gate" / "02_new"), since=1)))
        written = lp.copy_back_gate(p, self.hidden, manifest)
        self.assertEqual(sorted(written), ["02_new/r.json", "_rule1_y.json", "incumbent.json", "ledger.jsonl"])
        g = self.hidden / "gate"
        lines = (g / "ledger.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(lines[0]), dict(hash="old", accepted=False))  # old lines come back as they were
        self.assertEqual(json.loads(lines[1])["dir"], str(g / "02_new"))
        self.assertEqual(json.loads((g / "incumbent.json").read_text())["dir"], str(g / "02_new"))
        self.assertFalse((g / "_rule1_x.json").exists())  # the retirement is mirrored
        self.assertEqual((g / "01_abc" / "blob.bin").read_bytes(), before["gate/01_abc/blob.bin"])
        self.assertEqual((g / "01_abc" / "round.json").read_bytes(), before["gate/01_abc/round.json"])

    def test_an_untouched_copy_writes_nothing_back(self):
        lp = self.lp
        p, manifest = lp.private_gate_copy(self.hidden, self.hidden / "pool_accept")
        self.assertEqual(lp.copy_back_gate(p, self.hidden, manifest), [])

    def test_the_parent_is_0o311_while_locked_and_restored_also_after_an_exception(self):
        lp = self.lp
        state = self.root / "state"
        state.mkdir()
        self.hidden.parent.chmod(0o755)
        with mock.patch.multiple(lp, HIDDEN=self.hidden, STATE=state):
            with lp.parent_locked():
                self.assertEqual(stat.S_IMODE(self.hidden.parent.stat().st_mode), 0o311)
                self.assertTrue(lp.hidden_parent_file().exists())
            self.assertEqual(stat.S_IMODE(self.hidden.parent.stat().st_mode), 0o755)
            with self.assertRaises(RuntimeError):
                with lp.parent_locked():
                    raise RuntimeError("x")
            self.assertEqual(stat.S_IMODE(self.hidden.parent.stat().st_mode), 0o755)
            self.assertFalse(lp.hidden_parent_file().exists())
            # a crash leaves the file: unlock_hidden puts the mode back and removes the private copy
            p, _ = lp.private_gate_copy(self.hidden, self.hidden / "pool_accept")
            lp.hidden_parent_file().write_text("0o755")
            lp.private_gate_file().write_text(str(p))
            self.hidden.parent.chmod(0o311)
            lp.unlock_hidden()
            self.assertEqual(stat.S_IMODE(self.hidden.parent.stat().st_mode), 0o755)
            self.assertFalse(p.exists())


class SpecDecision(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_adopt_only_for_the_same_incumbent(self):
        lp = self.lp
        spec = dict(for_incumbent="h1", from_iter=3, account=dict(cost_usd=1.5, seconds=9), leaks=[])
        self.assertEqual(lp.spec_decision(spec, "h1"), "adopt")
        self.assertEqual(lp.spec_decision(spec, "h2"), "discard")
        self.assertIsNone(lp.spec_decision(None, "h1"))

    def test_discard_renames_the_directory_and_writes_its_cost(self):
        lp = self.lp
        rows = []
        st = self.root / "loop"
        it = st / "iter_04"
        it.mkdir(parents=True)
        (it / "spec.json").write_text("{}")
        with mock.patch.multiple(lp, STATE=st, ledger=rows.append):
            gone = lp.discard_spec(
                4, it, dict(for_incumbent="h1", from_iter=3, account=dict(cost_usd=1.5, seconds=9)), "why", "h2"
            )
            self.assertTrue(it.is_dir() and not (it / "spec.json").exists())
            self.assertTrue((gone / "spec.json").exists() and "spec_discarded" in gone.name)
            (st / "iter_03").mkdir()
            (st / "iter_05").mkdir()
            (st / "iter_05" / "spec.json").write_text("{}")  # a review no iteration took up yet is not an iteration
            self.assertEqual([d.name for d in lp.iteration_dirs()], ["iter_03", "iter_04"])
        self.assertEqual((rows[0]["step"], rows[0]["iter"], rows[0]["cost_usd"], rows[0]["incumbent"]),
                         ("spec_discarded", 4, 1.5, "h2"))  # fmt: skip

    def test_spec_runs_only_for_a_backend_family_with_the_cap_on_and_not_in_a_plain_stub(self):
        lp = self.lp
        with mock.patch.object(lp, "BK") as bk:
            bk.of.return_value = object()
            with mock.patch.dict(lp.CAPS, {"spec_review": True, "claims": False, "diag_redraw": False}):
                self.assertTrue(lp.spec_enabled(False))
                self.assertFalse(lp.spec_enabled(True))
                with mock.patch.dict("os.environ", {"RSI_STUB_REAL_GATE": "1"}):
                    self.assertTrue(lp.spec_enabled(True))
            with mock.patch.dict(lp.CAPS, {"spec_review": False}):
                self.assertFalse(lp.spec_enabled(False))
            bk.of.return_value = None
            with mock.patch.dict(lp.CAPS, {"spec_review": True}):
                self.assertFalse(lp.spec_enabled(False))


class StragglerDeadline(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()

    def test_the_deadline_counts_from_the_session_end_not_the_threads_end(self):
        lp = self.lp
        stops = {k: threading.Event() for k in (1, 2, 3)}
        sessions, release, stopped_at = {}, threading.Event(), []

        def t1():
            sessions[1] = time.time()

        def t2():  # its session is over; the thread goes on with its rounds and knob sweep
            sessions[2] = time.time()
            release.wait(30)

        def t3():  # its session is still running
            stops[3].wait(30)
            stopped_at.append(time.time())

        threads = {1: threading.Thread(target=t1), 2: threading.Thread(target=t2), 3: threading.Thread(target=t3)}
        for t in threads.values():
            t.start()
        threading.Timer(2.5, release.set).start()
        with mock.patch.object(lp, "ENGINEER_LATE_MIN", 0.01):  # 0.6 s
            t0 = time.time()
            late = lp.wait_drafts(threads, stops, poll=0.1, sessions=sessions)
        self.assertEqual(late, [3])  # only the session still running is stopped
        self.assertFalse(stops[2].is_set())
        self.assertLess(stopped_at[0] - t0, 1.8)  # counted from the sessions' end, not from thread 2's end at 2.5 s
        self.assertGreater(time.time() - t0, 2.4)  # and the draft still working was waited for


if __name__ == "__main__":
    unittest.main()
