"""The scene RSI loop's gate arithmetic and candidate identity (TODO 27.2.4, scripts/rsi/):
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
HERE = REPO / "scripts" / "rsi"


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

    def test_stage1_draws_every_task_by_its_share(self):
        labels = [f"book_s{i}" for i in range(30)] + [f"wallet_s{i}" for i in range(30)]
        s1 = gate.stage1_labels(labels)
        self.assertEqual((len(s1), sum(x.startswith("wallet") for x in s1)), (30, 15))
        self.assertEqual(s1, sorted(s1))
        labels = [f"t{k}_s{i}" for k in range(8) for i in range(7 if k < 4 else 8)]  # 60 starts over 8 tasks
        s1 = gate.stage1_labels(labels)
        per = {k: sum(x.startswith(f"t{k}_") for x in s1) for k in range(8)}
        self.assertEqual(len(s1), 30)
        self.assertEqual(set(per.values()), {3, 4})  # every task in, none more than one above another
        self.assertEqual([per[k] for k in range(8)], [4, 4, 3, 3, 4, 4, 4, 4])  # 3.5 each: remainders by name

    def test_drop_limit_from_a_zero_start(self):
        t = dict(incumbent_ok=0, incumbent_dropped=0, candidate_ok=15)
        self.assertEqual(gate.drop_limit(t), 7)
        self.assertEqual(gate.drop_limit(dict(t, candidate_ok=2)), 1)  # never below incumbent + 1
        self.assertEqual(gate.drop_limit(dict(t, incumbent_ok=9)), 1)  # an incumbent with successes: + 1 as before

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


def _fresh(name: str, file: str):
    """Load one of this loop's scripts under its own module name, leaving sys.modules as it was."""
    saved = {k: sys.modules.pop(k, None) for k in _names}
    sys.path.insert(0, str(HERE))
    try:
        return _load(name, HERE / file)
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


class SuccessRuleVersion(unittest.TestCase):
    """TODO 27.9: a round judged under an older success rule is not done; it is set aside, not overwritten."""

    def test_old_rule_rounds_are_set_aside(self):
        bt = _fresh("scene_rsi_batch_rule", "batch.py")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "mug_pickup_s1.json"
            f.write_text(json.dumps({"ok": True}))  # rule 1 rounds carry no rule
            f.with_suffix(".npz").write_bytes(b"x")
            self.assertFalse(bt.done(f))
            bt.retire(f)
            self.assertFalse(f.exists())
            self.assertTrue((Path(d) / "_rule1_mug_pickup_s1.json").exists())
            self.assertTrue((Path(d) / "_rule1_mug_pickup_s1.npz").exists())
            self.assertEqual(bt.outcomes(Path(d)), {})
            f.write_text(json.dumps({"ok": False, "rule": bt.SUCCESS_RULE}))
            self.assertTrue(bt.done(f))


class Infeasibility(unittest.TestCase):
    """D13 (plan section 8): claims need all their parts, parks need the check and the attempts, parks reopen when the
    incumbent finishes a parked start, the stop rule's second number leaves task-wide parks out, exits group by
    capability."""

    def setUp(self):
        from unittest import mock

        self.lp = _fresh("scene_rsi_loop_d13", "loop.py")
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.diag = ["book_pickup_s1", "book_pickup_s2", "mug_pickup_s3", "mug_pickup_s4"]
        (root / "pool_diag").mkdir()
        (root / "pool_accept").mkdir()
        for x in self.diag:
            (root / "pool_diag" / f"{x}.json").write_text(json.dumps({"task": x.rsplit("_s", 1)[0], "start": {"n": 1}}))
        self.check = root / "check.py"
        self.check.write_text(
            "def check(start, meta):\n    return dict(holds=start['task'] == 'book_pickup', value=1)\n"
        )
        self.patches = [mock.patch.multiple(self.lp, LOOP=root, STATE=root / "state", GATE_DIR=root / "gate",
                                            HIDDEN=root, FAMILY="pick_single", PROCS=1)]  # fmt: skip
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def claim(self, **kw):
        c = dict(id="book wedged", task="book_pickup", kind="current_skills", starts=["book_pickup_s1", "book_pickup_s2"],
                 actions_assumed="side and top-down grasps only", bound=dict(statement="10.4 cm < 18.5 cm", check=str(self.check)),
                 exits=[dict(capability="Pre-grasp tilt", how="tip the book out first")], refuted_by="any real lift",
                 attempts=[dict(name="top", set={"pick.TOP_FLAT_M": 0.2})])  # fmt: skip
        c.update(kw)
        return c

    def test_a_claim_needs_every_part(self):
        self.assertEqual(self.lp.claim_problems(self.claim(), self.diag), [])
        self.assertTrue(any("exits" in x for x in self.lp.claim_problems(self.claim(exits=[]), self.diag)))
        physical = self.claim(kind="physical", exits=[], no_exit_reason="out of reach for any posture")
        self.assertEqual(self.lp.claim_problems(physical, self.diag), [])
        self.assertTrue(self.lp.claim_problems(self.claim(starts=["mug_pickup_s3"]), self.diag))
        self.assertTrue(self.lp.claim_problems(self.claim(attempts=[]), self.diag))
        self.assertTrue(self.lp.claim_problems(self.claim(refuted_by=" "), self.diag))
        self.assertTrue(self.lp.claim_problems(self.claim(bound=dict(statement="x", check="/nope.py")), self.diag))

    def _fake_rounds(self, ok_labels=()):
        lp = self.lp

        def run_jobs(jobs, *a, **k):
            for job in jobs:
                job.out.mkdir(parents=True, exist_ok=True)
                for x in job.labels:
                    rec = dict(task=x.rsplit("_s", 1)[0], ok=x in ok_labels, why="", rule=lp.B.SUCCESS_RULE)
                    (job.out / f"{x}.json").write_text(json.dumps(rec))
            return {}

        return run_jobs

    def test_check_and_attempts_park_a_task_and_success_reopens_it(self):
        from unittest import mock

        lp, it = self.lp, Path(self.tmp.name) / "state" / "iter_00"
        inc = dict(dir=str(Path(self.tmp.name) / "inc"), set={})
        claims = [self.claim(), self.claim(id="mug", task="mug_pickup", starts=["mug_pickup_s3"])]
        with mock.patch.object(lp.B, "run_jobs", self._fake_rounds()):
            res = lp.verify_claims(claims, inc, it)
        self.assertEqual([r["parked"] for r in res], [True, False])  # the mug's check does not hold
        self.assertEqual(res[0]["scope"], "task")
        lp.park(res, claims, 0)
        self.assertEqual(list(lp.jload(lp.STATE / "parked.json")), ["book_wedged"])
        with mock.patch.object(lp.B, "run_jobs", self._fake_rounds(ok_labels={"book_pickup_s1"})):
            res = lp.verify_claims(claims[:1], inc, it)
        self.assertFalse(res[0]["parked"])  # an attempt that succeeds refutes the claim
        base = Path(self.tmp.name) / "base"
        self._fake_rounds(ok_labels={"book_pickup_s2"})([lp.B.Job(Path("."), base, None, tuple(self.diag))])
        self.assertEqual(lp.unpark(base, 1), ["book_wedged"])
        self.assertEqual(lp.jload(lp.STATE / "parked.json"), {})
        self.assertTrue(any(e.get("step") == "unpark" and e["claim_wrong"] for e in lp.entries()))

    def test_reopen_runs_only_the_parked_starts_and_lifts_on_a_win(self):
        """TODO 27.8.7: a candidate built for a park's exit runs on that park's starts; a win lifts the park without
        calling the claim wrong, no win drops the candidate, an unknown park is refused."""
        from unittest import mock

        lp, root = self.lp, Path(self.tmp.name)
        cand = root / "cand"
        (cand / "skills").mkdir(parents=True)
        (cand / "candidate.json").write_text(json.dumps({"set": {"pick.X": 1}}))
        starts = ["book_pickup_s1", "book_pickup_s2"]
        lp.jdump(lp.STATE / "parked.json", {"book": dict(task="book_pickup", scope="task", starts=starts, exits=[])})
        ran = []

        def spy(ok):
            fake = self._fake_rounds(ok_labels=ok)

            def run_jobs(jobs, *a, **k):
                ran.extend(j.labels for j in jobs)
                return fake(jobs, *a, **k)

            return run_jobs

        with mock.patch.object(lp.B, "run_jobs", spy(set())):
            self.assertFalse(lp.reopen_test(5, root / "it5", cand, "book"))
        self.assertEqual(ran, [tuple(starts)])
        self.assertIn("book", lp.jload(lp.STATE / "parked.json"))
        with mock.patch.object(lp.B, "run_jobs", spy({"book_pickup_s2"})):
            self.assertTrue(lp.reopen_test(6, root / "it6", cand, "book"))
        self.assertEqual(lp.jload(lp.STATE / "parked.json"), {})
        up = [e for e in lp.entries() if e["step"] == "unpark"]
        self.assertEqual((up[-1]["by"], up[-1]["claim_wrong"], up[-1]["won"]), ("reopen", False, ["book_pickup_s2"]))
        self.assertFalse(lp.reopen_test(7, root / "it7", cand, "book"))
        self.assertEqual(lp.entries()[-1]["error"], "no such park")

    def test_unparked_view_and_gap_candidates(self):
        from unittest import mock

        lp, root = self.lp, Path(self.tmp.name)
        accept = ["book_pickup_s7", "mug_pickup_s8", "mug_pickup_s9"]
        for x in accept:
            (root / "pool_accept" / f"{x}.json").write_text("{}")
        rounds = root / "inc_rounds"
        self._fake_rounds(ok_labels={"mug_pickup_s8", "mug_pickup_s9"})(
            [lp.B.Job(Path("."), rounds, None, tuple(accept))]
        )
        self.assertIsNone(lp.unparked_view(rounds))  # no parks
        lp.jdump(lp.STATE / "parked.json", {"book": dict(task="book_pickup", scope="task", starts=["book_pickup_s1"],
                 exits=[dict(capability="Pre-grasp tilt ", how="tip it")]),
                 "mug": dict(task="mug_pickup", scope="starts", starts=["mug_pickup_s3"],
                 exits=[dict(capability="pre-grasp tilt", how="tip the mug")])})  # fmt: skip
        v = lp.unparked_view(rounds)
        self.assertEqual((v["ok"], v["n"], v["all_ok"], v["all_n"]), (2, 2, 2, 3))
        lp.jdump(root / "gate" / "incumbent.json", dict(rounds=str(rounds)))
        self.assertIn("outside the parked tasks", lp.stop_reason(dict(started_s=lp.time.time())))
        with mock.patch.object(lp, "UNPARKED_MIN", 0.8):  # 2 of 3 starts outside the parks: too few for D13.4
            self.assertIsNone(lp.stop_reason(dict(started_s=lp.time.time())))
        g = lp.gap_candidates()
        self.assertEqual(list(g), ["pre-grasp tilt"])
        self.assertEqual(
            (g["pre-grasp tilt"]["tasks"], g["pre-grasp tilt"]["starts"]), (["book_pickup", "mug_pickup"], 2)
        )

    def test_parks_over_half_the_pool_redraw_it_from_the_live_tasks(self):
        from unittest import mock

        lp, root = self.lp, Path(self.tmp.name)
        for x in ["book_pickup_s7", "book_pickup_s8", "mug_pickup_s9"]:
            (root / "pool_accept" / f"{x}.json").write_text("{}")
        lp.jdump(lp.STATE / "parked.json", {"book": dict(task="book_pickup", scope="task", starts=["book_pickup_s1"])})
        self.assertEqual(lp.accept_pool(), root / "pool_accept")  # nothing drawn yet
        fam = dict(name="pick_single", tasks=["book_pickup", "mug_pickup"], rows={"book_pickup": {}, "mug_pickup": {}},
                   pools={"accept": dict(seed=1200000, n=3)})  # fmt: skip

        def draw(f, pool, out):
            out.mkdir()
            self.assertEqual((f["tasks"], f["pools"]["accept"]), (["mug_pickup"], dict(seed=1205000, n=3)))
            for i in range(3):
                (out / f"mug_pickup_s{1205000 + i}.json").write_text("{}")
            return dict(written=3, unavailable={})

        with mock.patch.object(lp.F, "load", return_value=fam), mock.patch.object(lp.F, "draw", side_effect=draw):
            lp.redraw_accept()
            lp.redraw_accept()  # drawn once
        pool = lp.accept_pool()
        self.assertTrue(pool.name.startswith("pool_accept_") and len(lp.B.pool_labels(pool)) == 3)
        self.assertEqual([e["tasks"] for e in lp.entries() if e["step"] == "redraw_accept"], [["mug_pickup"]])
        lp.jdump(lp.STATE / "parked.json", {})  # the park lifted: the family's own pool again
        self.assertEqual(lp.accept_pool(), root / "pool_accept")


class ClaimCheck(unittest.TestCase):
    def test_a_broken_check_does_not_hold(self):
        cc = _fresh("scene_rsi_claim_check", "claim_check.py")
        with tempfile.TemporaryDirectory() as d:
            chk = Path(d) / "c.py"
            chk.write_text(
                "def check(start, meta):\n    return dict(holds=meta['top'] > start['start']['z'], value=1)\n"
            )
            st = Path(d) / "a_s1.json"
            st.write_text(json.dumps({"task": "a", "start": {"z": 1}}))
            self.assertEqual(cc.run(chk, {"top": 2}, [st]), {"a_s1": {"holds": True, "value": 1}})
            self.assertIn("error", cc.run(chk, {}, [st])["a_s1"])


class Digest(unittest.TestCase):
    """The Reviewer's digest (digest.py, plan section 12): failure stages, separating numbers, slip in the hand."""

    def test_separators_rank_the_numbers_that_split_successes_from_failures(self):
        dg = _fresh("scene_rsi_digest", "digest.py")
        rows = [dict(ok=ok, steps=[dict(step="walk", ok=True, stop_err_cm=e, legs=2)]) for ok, e in
                [(True, 1.0), (True, 1.2), (False, 4.0), (False, 5.0)]]  # fmt: skip
        sep = dg.separators(rows)
        self.assertEqual(len(sep), 1)  # legs is the same everywhere
        self.assertIn("walk.stop_err_cm: successes 1.10 (n 2), failures 4.50 (n 2), rank-biserial -1.00", sep[0])

    def test_first_fail_and_the_slip_between_grasp_and_release(self):
        import numpy as np

        dg = _fresh("scene_rsi_digest", "digest.py")
        steps = [dict(step="plan_ahead", ok=True, hand="right", t0=1.0, t1=1.0),
                 dict(step="grasp", ok=True, t0=2.0, t1=3.0), dict(step="carry", ok=False, why="lost it", t0=3.0, t1=5.0),
                 dict(step="release", ok=True, t0=5.0, t1=6.0)]  # fmt: skip
        r = dict(label="x_s1", task="x", ok=False, steps=steps)
        self.assertEqual(dg.first_fail(r), ("carry", "lost it"))
        t = np.arange(0, 7, 0.02, dtype=np.float32)
        obj = np.zeros((len(t), 7), np.float32)
        obj[:, 2] = np.where(t > 3, 1.0, 0.7)
        obj[:, 0] = np.where(t >= 4, 0.05, 0.0)  # 5 cm away from the wrist during the carry
        wrists = np.zeros((len(t), 2, 3), np.float32)
        wrists[:, 1, 2] = np.where(t > 3, 1.0, 0.7)
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "x_s1.npz", t=t, wrists=wrists, obj_box_object=obj)
            tr = dg.trace(Path(d) / "x_s1.npz", r, ["box"], {"objects": {"box": {"body": "box_object"}}})
        self.assertEqual(tr["box"]["lift_cm"], 30.0)
        self.assertEqual(tr["box"]["slip_cm"], 5.0)
        self.assertNotIn("knocked_cm", tr["box"])


class SimSlots(unittest.TestCase):
    def test_slots_are_shared_and_freed_with_their_holder(self):
        from unittest import mock

        bt = _fresh("scene_rsi_batch_slots", "batch.py")
        with tempfile.TemporaryDirectory() as d, mock.patch.multiple(bt, SIM_SLOTS=2, SLOTS_DIR=Path(d)):
            a, b = bt.take_slot(), bt.take_slot()
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            self.assertIsNone(bt.take_slot())  # both held
            a.close()
            c = bt.take_slot()
            self.assertIsNotNone(c)
            b.close()
            c.close()


class OuterGroups(unittest.TestCase):
    def test_a_task_whose_pick_has_a_gap_joins_the_pick_family(self):
        sys.path.insert(0, str(HERE))
        try:
            outer = _load("scene_rsi_outer", HERE / "outer.py")
        finally:
            sys.path.remove(str(HERE))
        meta = {"objects": {"wallet": {"semantic": {"graspable": "single"}}, "box": {}}}
        row = dict(goal=["in(wallet, box)"], checks=[
            dict(step="grasp_single(wallet)", name="object_size", outcome="untested", gap="grasp_single.object_size"),
            dict(step="place(wallet in box)", name="rim_height", outcome="untested", gap="place_into.rim_height")])  # fmt: skip
        self.assertEqual(outer.group(row, meta), "held_single")
        row["checks"][0].update(outcome="ok", gap="")
        self.assertEqual(outer.group(row, meta), "in_single")


class OuterRun(unittest.TestCase):
    def setUp(self):
        self.run_mod = _fresh("scene_rsi_outer_run", "outer_run.py")

    def test_kind_follows_the_goals(self):
        k = self.run_mod.kind_of
        self.assertEqual([k(f) for f in ("r2_held_near_single", "r2_in_single", "r2_on_single", "r2_held_single",
                                         "r2_in_room_none")], ["near", "into", "place", "held", "none"])  # fmt: skip

    def test_a_blocked_step_stops_the_family_and_resumes_after(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            f = self.run_mod.Family(Path(d), "r2_on_single")
            calls = []
            ok = lambda name: lambda: calls.append(name)  # noqa: E731

            def lint():
                raise self.run_mod.Blocked("lint: a goal holds at every start")

            with mock.patch.multiple(f, sandbox_=ok("sandbox"), lint_=lint, loop_=ok("loop")):
                self.assertFalse(f.advance("writeback"))
            self.assertEqual((f.state["step"], calls), ("lint", ["sandbox"]))
            self.assertIn("goal holds", json.loads(f.state_file.read_text())["blocked"])
            g = self.run_mod.Family(Path(d), "r2_on_single")  # a restarted driver reads the state
            with mock.patch.multiple(g, lint_=ok("lint"), loop_=ok("loop"), sealed_=ok("sealed")):
                self.assertTrue(g.advance("writeback"))
            self.assertEqual((g.state["step"], g.state["blocked"]), ("writeback", None))
            self.assertEqual(calls, ["sandbox", "lint", "loop", "sealed"])


class ReviewReplay(unittest.TestCase):
    """The digest replay (review_replay.py, plan section 12.2)."""

    def test_one_probe_per_parameter_set_and_the_plain_arm_drops_the_digest(self):
        saved = {k: sys.modules.pop(k, None) for k in ("loop", "digest")}
        try:
            rr = _fresh("scene_rsi_review_replay", "review_replay.py")
        finally:
            for k, v in saved.items():
                sys.modules.pop(k, None)
                if v is not None:
                    sys.modules[k] = v
        key = rr.delta_key
        self.assertEqual(key(dict(delta={"set": {"a.X": 1, "a.Y": 2}})), key(dict(delta={"set": {"a.Y": 2, "a.X": 1}})))
        self.assertNotEqual(key(dict(delta={"set": {"a.X": 1}})), key(dict(delta={"set": {"a.X": 1.0001}})))
        prompt = (HERE / "prompts" / "reviewer.md").read_text()
        self.assertIn("{DIGEST}", prompt)
        self.assertNotIn("{DIGEST}", rr.DIGEST_ITEM.sub("", prompt))
        lib = rr.DIGEST_ITEM.sub(lambda m: m.group(0) + rr.LIBRARY_ITEM, prompt)  # the library arm: item 5 after 4
        self.assertLess(lib.index("4. {DIGEST}"), lib.index("5. {LIBRARY}"))
        cmd = "cd /workspace/code/rsi-scene/replay2-r2on; sed -n 1,9p /workspace/code/locomanipulation/docs/x.md | head"
        self.assertEqual(rr.PATH.findall(cmd), ["/workspace/code/rsi-scene/replay2-r2on",
                                                "/workspace/code/locomanipulation/docs/x.md"])  # fmt: skip


class CreepSection(unittest.TestCase):
    """The checker's creep rule as the capability table carries it (carry_time.section and fit, plan section 12.3)."""

    def test_skill_constants_are_read_from_the_catalog_and_the_fit_adds_a_put_in_term(self):
        ct = _fresh("scene_rsi_carry_time", "carry_time.py")
        self.assertEqual(ct.skill_constant("grasp_program", "PAD_Z"), 0.014)  # PAD_Y, PAD_Z = 0.0475, 0.014
        with self.assertRaises(KeyError):
            ct.skill_constant("place", "NO_SUCH_CONSTANT")
        rows = [dict(d=d, move=m, t_med=30 + 8 * d + (25 if m == "in" else 0), same_surface=False)
                for d, m in [(1, "on"), (3, "on"), (5, "on"), (2, "in"), (4, "in")]]  # fmt: skip
        rows.append(dict(d=None, move="in", t_med=22.0, same_surface=True))
        m = ct.fit(rows)
        self.assertEqual((m["base"], m["per_m"], m["put_in"], m["near_drop"]), (30.0, 8.0, 25.0, 22.0))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fit.json").write_text(json.dumps(dict(model=m)))
            c = ct.section(fit=Path(d) / "fit.json")
            self.assertIsNone(ct.section(fit=Path(d) / "missing.json"))
        self.assertEqual((c["rate_m_s"], c["pad_half_m"]), (ct.skill_constant("place", "CREEP_RATE_M_S"), 0.014))
        self.assertEqual(c["put_in"], bool(ct.skill_constant("into", "CREEP")))
        self.assertEqual(c["t_est_s"], dict(base=30.0, per_m=8.0, put_in=25.0, near_drop=22.0))


class CalibrationRegistry(unittest.TestCase):
    """A calibration sample registered like a family (regress.py; its tasks count as measured, user 2026-10-19)."""

    def test_its_rounds_live_in_its_sample_dir_and_capability_write_reads_them_with_its_pool(self):
        saved = {k: sys.modules.pop(k, None) for k in ("gate",)}
        try:
            rg = _fresh("scene_rsi_regress", "regress.py")
        finally:
            for k, v in saved.items():
                sys.modules.pop(k, None)
                if v is not None:
                    sys.modules[k] = v
        cal = dict(sandbox="/r", kind="calibration", pool="/r/cal/c1/pool", last="rounds", dir="/r/cal/c1")
        fam = dict(sandbox="/s", kind="place", pool=None, last="t0")
        self.assertEqual(rg.rounds_dir("c1", cal, "wb8"), Path("/r/cal/c1/wb8"))
        self.assertEqual(rg.rounds_dir("f", fam, "t0"), Path("/s") / rg.HIDDEN / "f" / "sealed" / "t0")
        args = rg.capability_args({"c1": cal, "f": fam})
        self.assertEqual(args[args.index("--calibration") + 1], "/r:c1:/r/cal/c1/rounds:/r/cal/c1/pool")
        self.assertIn("--place-loop", args)


class Successes(unittest.TestCase):
    """The tasks the loop has made work (successes.py, plan section 13; user 2026-10-19)."""

    def test_a_goal_pools_its_families_and_low_rate_successes_are_kept_apart(self):
        from unittest import mock

        saved = {k: sys.modules.pop(k, None) for k in ("gate", "regress", "calibrate")}
        try:
            su = _fresh("scene_rsi_successes", "successes.py")
        finally:
            for k, v in saved.items():
                sys.modules.pop(k, None)
                if v is not None:
                    sys.modules[k] = v
        reg = {
            "a": dict(kind="place", last="t"),
            "b": dict(kind="place", last="t"),
            "c": dict(kind="calibration", last="t"),
        }
        tasks = {"a": {"x": dict(goal=["on(Jam, Table)"])}, "b": {"y": dict(goal=["on(Jam, Table)"])},
                 "c": {"z": dict(goal=["held(Cup)"]), "w": dict(goal=["held(Box)"])}}  # fmt: skip
        rounds = {("a", "t"): [("x", 1, True), ("x", 2, True)], ("a", "seed"): [("x", 1, False), ("x", 9, True)],
                  ("b", "t"): [("y", 3, False), ("y", 4, False)], ("b", "seed"): [],
                  ("c", "t"): [("z", 5, True), ("z", 6, False), ("z", 7, False), ("w", 8, False)]}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            for (f, tag), recs in rounds.items():
                (Path(d) / f / tag).mkdir(parents=True)
                for task, i, ok in recs:
                    (Path(d) / f / tag / f"{task}_s{i}.json").write_text(json.dumps(dict(task=task, ok=ok)))
            with mock.patch.multiple(su.R, load=lambda: reg, rounds_dir=lambda f, row, tag: Path(d) / f / tag):
                with mock.patch.object(su, "tasks_of", lambda f: tasks[f]):
                    gs = su.goals()
        jam, cup, box = (gs[su.norm(g)] for g in (["on(Jam, Table)"], ["held(Cup)"], ["held(Box)"]))
        self.assertEqual((jam["ok"], jam["n"], jam["seed_ok"], jam["seed_n"]), (2, 4, 0, 1))  # seed paired by start
        self.assertEqual((jam["status"], jam["trained"], len(jam["runs"])), ("works", True, 2))
        self.assertEqual((cup["status"], cup["trained"], cup["seed_n"]), ("low", False, 0))
        self.assertEqual(box["status"], "none")


class Calibrate(unittest.TestCase):
    """The checker's calibration sample (calibrate.py, plan section 12.3)."""

    def test_allocation_follows_stratum_size_with_one_each(self):
        cal = _fresh("scene_rsi_calibrate", "calibrate.py")
        self.assertEqual(
            cal.allocate({"in": 55, "on": 30, "held": 10, "near": 5}, 10), {"held": 1, "in": 5, "near": 1, "on": 3}
        )
        self.assertEqual(sum(cal.allocate({"a": 1, "b": 1, "c": 1}, 2).values()), 2)
        self.assertEqual(cal.allocate({"a": 2, "b": 50}, 30), {"a": 1, "b": 29})  # 1.15 and 28.85

    def test_wilson_interval(self):
        cal = _fresh("scene_rsi_calibrate", "calibrate.py")
        self.assertEqual(gate.wilson(0, 0), [0.0, 1.0])
        lo, hi = gate.wilson(8, 8)
        self.assertTrue(0.6 < lo < 0.7 and hi == 1.0)
        self.assertEqual(gate.wilson(4, 8), [0.215, 0.785])
        self.assertEqual(cal.kind(["held(x)", "near(x, y, 0.3)", "upright(x)"]), "held+near+upright")


if __name__ == "__main__":
    unittest.main()
