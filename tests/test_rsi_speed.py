"""TODO 32 (the scene loop made faster): batch.py's round queue hands every round to exactly one worker; the scene's
compiled-model cache is reused while the XML and the files it names are unchanged, and rebuilt when one changes;
replay.py finds a gate call's iteration from its candidate path and rebuilds the parks of a run without snapshots."""

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


# the warehouse loop has its own batch.py: load this loop's under its own name, leave sys.modules as it was
_saved = {k: sys.modules.pop(k, None) for k in ("batch", "sandbox")}
_path = list(sys.path)
sys.path.insert(0, str(HERE))
try:
    batch = _load("scene_rsi_batch", HERE / "batch.py")
    replay = _load("scene_rsi_replay", HERE / "replay.py")
finally:
    sys.path[:] = _path
    for _k, _v in _saved.items():
        sys.modules.pop(_k, None)
        if _v is not None:
            sys.modules[_k] = _v


class RoundQueue(unittest.TestCase):
    def test_every_round_goes_to_one_taker_in_job_then_label_order(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = [batch.Job(Path(d) / "a", Path(d) / "out_a", {"X": 1}), batch.Job(Path(d) / "b", Path(d) / "out_b")]
            q = batch.make_queue(jobs, {0: ["s1", "s2"], 1: ["s1"]}, Path(d))
            self.assertEqual(batch.queued(q), 3)
            spec = json.loads((q / "jobs.json").read_text())
            self.assertEqual(spec[0]["overrides"], {"X": 1})
            self.assertEqual(spec[1]["overrides"], {})
            taken = [batch.claim_round(q) for _ in range(4)]
            self.assertEqual(taken[:3], [dict(job=0, label="s1"), dict(job=0, label="s2"), dict(job=1, label="s1")])
            self.assertIsNone(taken[3])
            self.assertEqual(batch.queued(q), 0)
            self.assertEqual(len(list((q / "taken").iterdir())), 3)

    def test_a_round_another_worker_took_first_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            q = batch.make_queue([batch.Job(Path(d), Path(d))], {0: ["s1", "s2"]}, Path(d))
            first = sorted((q / "todo").iterdir())[0]
            os.rename(first, q / "taken" / "other_worker.json")  # another process got there first
            self.assertEqual(batch.claim_round(q), dict(job=0, label="s2"))


class ModelCache(unittest.TestCase):
    XML = '<mujoco><asset><mesh name="m" file="m.stl"/></asset><worldbody><geom type="box" size="{s} .1 .1"/></worldbody></mujoco>'

    def setUp(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from locomani.backends.scalebfm import runtime

        self.runtime = runtime
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        import locomani.paths as P

        self._old, P.MJB_CACHE = P.MJB_CACHE, str(self.cache)

    def tearDown(self):
        import locomani.paths as P

        P.MJB_CACHE = self._old
        self.tmp.cleanup()

    def _scene(self, d: Path, size: str) -> str:
        import struct

        tri = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # a tetrahedron, binary STL
        faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        data = b"\0" * 80 + struct.pack("<I", len(faces))
        for f in faces:
            data += struct.pack("<3f", 0, 0, 0) + b"".join(struct.pack("<3f", *tri[i]) for i in f) + b"\0\0"
        (d / "m.stl").write_bytes(data)
        (d / "s.xml").write_text(self.XML.format(s=size))
        return str(d / "s.xml")

    def test_cache_is_used_until_the_xml_or_a_named_file_changes(self):
        d = Path(self.tmp.name) / "scene"
        d.mkdir()
        xml = self._scene(d, ".1")
        m1 = self.runtime.load_model(xml)
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 1)
        self.assertEqual(sorted(p.name for p in d.iterdir()), ["m.stl", "s.xml"])  # nothing beside the XML
        m2 = self.runtime.load_model(xml)  # from the cache
        self.assertEqual(m1.geom_size.tolist(), m2.geom_size.tolist())
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 1)
        Path(xml).write_text(self.XML.format(s=".2"))
        self.assertAlmostEqual(float(self.runtime.load_model(xml).geom_size[0][0]), 0.2)
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 2)
        os.utime(d / "m.stl", ns=(time.time_ns(), time.time_ns() + 10**9))  # a named file touched
        self.runtime.load_model(xml)
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 3)


class ReplayState(unittest.TestCase):
    def test_iteration_of_a_gate_call(self):
        self.assertEqual(replay.iter_of_candidate("/s/outputs/x/fam/loop/iter_03/candidate"), 3)
        self.assertIsNone(replay.iter_of_candidate("/s/outputs/x/fam/gate/incumbent_seed"))

    def test_parks_before_an_iteration_with_the_unparks_applied(self):
        claim = dict(task="pick_Mug", kind="current_skills", starts=["pick_Mug_s2", "pick_Mug_s1"],
                     bound=dict(statement="too wide"), actions_assumed="side grasp", refuted_by="a mug lifted")  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            loop = Path(d)
            for k in range(4):
                (loop / f"iter_{k:02d}").mkdir()
            (loop / "iter_01" / "hypotheses.json").write_text(json.dumps(dict(hypotheses=[], infeasible=[claim])))
            (loop / "iter_01" / "claims.json").write_text(json.dumps([dict(id="C1", parked=True, scope="task")]))
            (loop / "ledger.jsonl").write_text(json.dumps(dict(step="unpark", iter=3, claim="C1")) + "\n")
            self.assertEqual(replay.rebuild_parks(loop, 1), {})
            parks = replay.rebuild_parks(loop, 3)
            self.assertEqual(sorted(parks), ["C1"])
            self.assertEqual(parks["C1"]["starts"], ["pick_Mug_s1", "pick_Mug_s2"])
            self.assertEqual(parks["C1"]["iter"], 1)
            self.assertEqual(replay.rebuild_parks(loop, 4), {})  # unparked in iteration 3


if __name__ == "__main__":
    unittest.main()


def _scene_loop():
    """scripts/rsi/loop.py under its own name (the warehouse loop has a loop.py too), sys.modules left as it was."""
    names = ("gate", "batch", "family", "scene_round", "digest", "loop")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load("scene_rsi_loop_32", HERE / "loop.py")
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


class LoopProcedure(unittest.TestCase):
    """TODO 32.9-32.11: the scoped Reviewers' hypotheses merge under their scope letter; a candidate's diagnostic flips
    are counted start by start; the Manager and the Reviewer see the multi-candidate and library notes only when on."""

    def setUp(self):
        self.lp = _scene_loop()

    def test_scoped_reviewers_merge_two_hypotheses_each(self):
        from unittest import mock

        def fake(role, prompt, schema, it_dir, tools, stub):
            tag = role.split("_")[1]
            self.assertIn("within your scope", prompt)
            self.assertIn(self.lp.SCOPES[tag][:40], prompt)
            hyps = [dict(id=f"H{i}", claim=f"{tag}{i}") for i in (1, 2, 3)]
            claim = dict(task="t", starts=[f"{tag}_s1"])
            return dict(notes=tag, hypotheses=hyps, infeasible=[claim]), dict(
                cost_usd=1.0, session=tag, seconds=10 * len(tag)
            )

        with mock.patch.object(self.lp, "claude", fake):
            hyps, acct = self.lp.scoped_review("propose at most 4 interventions", Path("."), None)
        self.assertEqual([h["id"] for h in hyps["hypotheses"]], ["PH1", "PH2", "OH1", "OH2"])
        self.assertEqual({h["scope"] for h in hyps["hypotheses"]}, {"P", "O"})
        self.assertEqual(len(hyps["infeasible"]), 2)
        self.assertEqual(acct["cost_usd"], 2.0)
        self.assertEqual(sorted(acct["sessions"]), ["O", "P"])

    def test_a_failed_scoped_reviewer_raises_role_failed(self):
        # 2026-10-21: a usage limit inside one Reviewer's thread surfaced as KeyError 'P' and crashed the run
        from unittest import mock

        def fake(role, prompt, schema, it_dir, tools, stub):
            if role.endswith("_P"):
                raise self.lp.RoleFailed("reviewer_P: session limit")
            return dict(notes="", hypotheses=[], infeasible=[]), dict(cost_usd=1.0, session="O", seconds=1)

        with mock.patch.object(self.lp, "claude", fake), self.assertRaises(self.lp.RoleFailed):
            self.lp.scoped_review("propose at most 4 interventions", Path("."), None)

    def test_flips_pair_the_candidate_with_the_incumbent_start_by_start(self):
        with tempfile.TemporaryDirectory() as d:
            inc, cand = Path(d) / "inc", Path(d) / "cand"
            inc.mkdir()
            cand.mkdir()
            for lab, a, b in (("s1", False, True), ("s2", True, False), ("s3", True, True), ("s4", False, False)):
                (inc / f"{lab}.json").write_text(json.dumps(dict(ok=a)))
                (cand / f"{lab}.json").write_text(json.dumps(dict(ok=b)))
            (cand / "s5.json").write_text(json.dumps(dict(ok=True)))  # not run for the incumbent: not paired
            f = self.lp.flips(inc, cand)
        self.assertEqual((f["n"], f["ok"], f["incumbent_ok"], f["fixed"], f["broken"]), (4, 2, 2, ["s1"], ["s2"]))

    def test_notes_appear_only_when_switched_on(self):
        self.lp.configure("grasp_atomic")
        self.lp.CANDIDATES = 1
        self.assertEqual(self.lp.prompt_vars()["CANDIDATES_NOTE"], "")
        self.lp.CANDIDATES = 3
        note = self.lp.prompt_vars()["CANDIDATES_NOTE"]
        self.assertIn("up to 3 hypothesis ids", note)
        self.assertNotIn("{K}", note)
        self.lp.CANDIDATES = 1
        lib = (HERE / "experience" / "INDEX.md").exists()
        self.assertEqual(bool(self.lp.prompt_vars()["LIBRARY"]), lib)


class RunFlags(unittest.TestCase):
    """`loop.py run` takes every TODO 32 switch on its command line (the first --scoped-review never reached the
    parser: a text edit missed the reformatted line, and 8 scoped replays ran without it, 2026-10-21)."""

    def test_the_switches_reach_cmd_run(self):
        from unittest import mock

        lp = _scene_loop()
        seen = {}

        def fake_run(a):
            seen.update(vars(a))
            return 0

        with mock.patch.object(lp, "cmd_run", fake_run), mock.patch.object(lp, "configure", lambda f: None):
            lp.main(["run", "grasp_atomic", "--candidates", "3", "--scoped-review", "--stop-after", "probe",
                     "--effort", "engineer=medium", "--engineer-from", "/x/iter_02"])  # fmt: skip
        self.assertEqual((seen["candidates"], seen["scoped_review"], seen["stop_after"]), (3, True, "probe"))
        self.assertEqual((seen["effort"], str(seen["engineer_from"])), (["engineer=medium"], "/x/iter_02"))

    def test_three_engineers_and_a_composer_by_default(self):
        # TODO 32.15 (user 2026-10-22): the 32.12 configuration is the default; --candidates 1 is the old loop.
        # TODO 36 (review 2026-10-22, L3): the parser leaves unset flags at None; resolve_settings decides
        from argparse import Namespace
        from unittest import mock

        lp = _scene_loop()
        seen = {}
        with (
            mock.patch.object(lp, "cmd_run", lambda a: seen.update(vars(a)) or 0),
            mock.patch.object(lp, "configure", lambda f: None),
        ):
            lp.main(["run", "grasp_atomic"])
            self.assertEqual((seen["candidates"], seen["scoped_review"]), (None, None))
            lp.main(["run", "grasp_atomic", "--candidates", "1"])
            self.assertEqual(seen["candidates"], 1)
        none = Namespace(candidates=None, scoped_review=None)
        self.assertEqual(lp.resolve_settings(none, {}, False), dict(candidates=3, scoped_review=False, max_wall_h=6.0))
        old = dict(started="x", candidates=1, scoped_review=True, max_wall_h=5.0)
        with mock.patch.dict("os.environ", clear=False):
            os.environ.pop("RSI_MAX_WALL_H", None)
            self.assertEqual(
                lp.resolve_settings(none, old, True), dict(candidates=1, scoped_review=True, max_wall_h=5.0)
            )
            self.assertEqual(lp.resolve_settings(none, old, False)["candidates"], 3)  # fresh run ignores stale meta
            self.assertEqual(
                lp.resolve_settings(Namespace(candidates=2, scoped_review=None), old, True)["candidates"], 2
            )
            os.environ["RSI_MAX_WALL_H"] = "4"
            self.assertEqual(lp.resolve_settings(none, old, True)["max_wall_h"], 4.0)

    def test_effort_names_are_validated(self):
        lp = _scene_loop()
        self.assertEqual(lp.parse_effort(["engineer=medium", "engineer_2=low", "reviewer_P=high"]),
                         {"engineer": "medium", "engineer_2": "low", "reviewer_P": "high"})  # fmt: skip
        with self.assertRaises(SystemExit) as c:
            lp.parse_effort(["enginer=medium"])
        self.assertIn("composer", str(c.exception))


class RunExit(unittest.TestCase):
    """TODO 36 (review 2026-10-22, L3, L4): exit codes, the stop row of a crash or a SIGTERM, and a resume that keeps
    the run's recorded settings (with a `settings` row when a flag changes one)."""

    def setUp(self):
        from unittest import mock

        self.lp = _scene_loop()
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

    def run_(self, iterate, **kw):
        from argparse import Namespace
        from unittest import mock

        a = Namespace(max_iters=1, reprobe=None, stub=True, **kw)
        with mock.patch.object(self.lp, "iterate", iterate):
            return self.lp.cmd_run(a)

    def test_normal_stops_exit_0_and_abnormal_ones_3(self):
        self.assertEqual(self.run_(lambda n, m, s: None), 0)
        self.assertEqual(self.run_(lambda n, m, s: "manager: stop"), 0)
        self.assertEqual(self.run_(lambda n, m, s: "gate calls 12 >= 12"), 0)
        self.assertEqual(self.run_(lambda n, m, s: "role call failed: x"), 3)
        self.assertEqual(self.run_(lambda n, m, s: "reviewer of iteration 1 named off-limits paths (gate)"), 3)
        self.assertEqual(self.run_(lambda n, m, s: "gate error at iteration 2 (see gate.log)"), 3)
        self.assertEqual(self.run_(lambda n, m, s: "audit: iterations 1, 2 voided in a row (x)"), 3)  # TODO 36.7

    def test_a_crash_and_a_sigterm_write_a_stop_row(self):
        import signal

        def boom(n, m, s):
            raise ValueError("boom")

        self.assertEqual(self.run_(boom), 3)
        last = self.lp.entries()[-1]
        self.assertEqual(last["step"], "stop")
        self.assertIn("crashed: ValueError", last["reason"])

        def term(n, m, s):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)

        self.assertEqual(self.run_(term), 3)
        self.assertEqual(self.lp.entries()[-1]["reason"], "terminated")
        meta = self.lp.jload(self.lp.STATE / "meta.json")
        self.lp.resume(meta)  # the crash's downtime is a pause
        self.assertEqual(self.lp.entries()[-1]["step"], "resume")

    def test_a_resume_keeps_the_recorded_settings_and_logs_a_change(self):
        self.run_(lambda n, m, s: None, candidates=1, scoped_review=None)
        meta = self.lp.jload(self.lp.STATE / "meta.json")
        self.assertEqual((meta["candidates"], self.lp.CANDIDATES), (1, 1))
        self.run_(lambda n, m, s: None, candidates=None, scoped_review=None)
        self.assertEqual(self.lp.CANDIDATES, 1)
        self.assertNotIn("settings", [e["step"] for e in self.lp.entries()])
        self.run_(lambda n, m, s: None, candidates=2, scoped_review=None)
        row = [e for e in self.lp.entries() if e["step"] == "settings"][-1]
        self.assertEqual(row["candidates"], dict(old=1, new=2))


class GateVerdict(unittest.TestCase):
    """TODO 36 (review 2026-10-22, L1, L2, L6, L12): the verdict is the gate's new ledger line for this candidate."""

    def setUp(self):
        from unittest import mock

        self.lp = _scene_loop()
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = Path(self.d.name)
        self.cand = self.root / "cand"
        (self.cand / "skills").mkdir(parents=True)
        (self.cand / "skills" / "__init__.py").write_text("A = 1\n")
        (self.cand / "candidate.json").write_text('{"set": {}}')
        self.gate = self.root / "gate"
        self.gate.mkdir()
        for p in (mock.patch.multiple(self.lp, STATE=self.root / "state", LOOP=self.root, HIDDEN=self.root,
                                      GATE_DIR=self.gate, accept_pool=lambda: self.root,
                                      redraw_accept=lambda: None),
                  mock.patch.object(self.lp, "unparked_view", lambda r: None)):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)

    def gate_run(self, line, rc=1):
        from types import SimpleNamespace
        from unittest import mock

        def run(*a, **k):
            if line is not None:
                with (self.gate / "ledger.jsonl").open("a") as fh:
                    fh.write(json.dumps(line) + "\n")
            return SimpleNamespace(returncode=rc, stdout="", stderr="")

        with mock.patch.object(self.lp.subprocess, "run", run):
            return self.lp.gate_candidate(3, self.root / "it", self.cand, True)

    def test_a_crashed_gate_stops_the_run_and_keeps_no_stale_verdict(self):
        (self.root / "it").mkdir()
        h = self.lp.candidate_hash(self.cand)
        stale = dict(hash="old", accepted=True, reason="accepted")
        (self.gate / "ledger.jsonl").write_text(json.dumps(stale) + "\n")
        why = self.gate_run(None, rc=1)  # exits 1 with no new line: a crash, not a rejection
        self.assertIn("gate error at iteration 3", why)
        self.assertEqual(self.lp.entries()[-1]["step"], "gate_error")
        self.assertIn("gate error", self.gate_run(dict(hash="other", accepted=False, reason="x")))
        self.assertIn("gate error", self.gate_run(dict(hash=h, accepted=False, reason="x"), rc=2))
        self.assertNotIn("gate", [e["step"] for e in self.lp.entries()])

    def test_a_void_verdict_is_a_void_gate_row_and_a_rejection_keeps_the_park(self):
        (self.root / "it").mkdir()
        h = self.lp.candidate_hash(self.cand)
        self.lp.jdump(self.lp.STATE / "parked.json", {"p": dict(task="t", scope="task", starts=["t_s1"], exits=[])})
        self.gate_run(dict(hash=h, accepted=False, void=True, reason="pairing incomplete"), rc=0)
        row = [e for e in self.lp.entries() if e["step"] == "gate"][-1]
        self.assertEqual((row["void"], row["accepted"]), (True, False))
        self.assertEqual(self.lp.stop_reason.__name__, "stop_reason")
        # the park stays after a reopen candidate the gate rejected or voided
        from unittest import mock

        with mock.patch.object(self.lp, "subprocess", self.lp.subprocess):
            pass
        self.lp.settle_reopen(3, "p", ["t_s1"], accepted=False)
        self.assertIn("p", self.lp.jload(self.lp.STATE / "parked.json"))

    def test_budget_counts_this_runs_cost_and_the_stop_rate_uses_the_current_pool(self):
        es = [dict(t="2026-10-01T00:00:00Z", cost_usd=5.0), dict(t="2026-10-22T00:00:00Z", cost_usd=2.0)]
        self.assertEqual(self.lp.run_cost(es, dict(started="2026-10-20T00:00:00Z")), 2.0)
        pool = str(self.root)
        old = dict(step="gate", accepted=True, result=dict(candidate_ok=45, incumbent_ok=30, n=60), pool="/old")
        rej = dict(step="gate", accepted=False, result=dict(candidate_ok=20, incumbent_ok=30, n=60), pool=pool)
        self.assertIsNone(self.lp.incumbent_ok_here([old]))  # other pool: keep today's rule
        self.assertEqual(self.lp.incumbent_ok_here([old, rej]), 30)
        self.assertEqual(
            self.lp.incumbent_ok_here([old, rej, dict(rej, void=True, result=dict(candidate_ok=1, incumbent_ok=1))]), 30
        )


class RoleSafety(unittest.TestCase):
    """TODO 36 (review 2026-10-22, L5, L8, L9, L10, L13): hidden tree lock, commits undone, climb-out audit, voided
    drafts hidden, the role env."""

    def setUp(self):
        self.lp = _scene_loop()
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = Path(self.d.name)
        # every test here runs with a hidden dir of its own: a role call with HIDDEN at Path() once locked the repo root
        self.hid = self.root / "outputs" / "scene_rsi_20261016_hidden" / "fam"
        self.hid.mkdir(parents=True)
        self.patch(HIDDEN=self.hid)

    def patch(self, **kw):
        from unittest import mock

        p = mock.patch.multiple(self.lp, **kw)
        p.start()
        self.addCleanup(p.stop)

    def test_hidden_dir_is_locked_while_roles_run_and_restored_after(self):
        hid = self.hid
        hid.chmod(0o750)
        self.patch(STATE=self.root / "state")
        (self.root / "state").mkdir()
        with self.lp.hidden_locked():
            with self.lp.hidden_locked():  # a parallel role shares the lock
                self.assertEqual(hid.stat().st_mode & 0o777, 0)
            self.assertEqual(hid.stat().st_mode & 0o777, 0)
        self.assertEqual(hid.stat().st_mode & 0o777, 0o750)
        with self.assertRaises(RuntimeError), self.lp.hidden_locked():
            raise RuntimeError()
        self.assertEqual(hid.stat().st_mode & 0o777, 0o750)
        # a crash left it locked: unlock_hidden puts the mode back
        self.lp.hidden_mode_file().write_text("0o750")
        hid.chmod(0)
        self.lp.unlock_hidden()
        self.assertEqual(hid.stat().st_mode & 0o777, 0o750)

    def test_a_role_commit_becomes_working_tree_changes(self):
        import subprocess

        git = lambda *a: subprocess.run(["git", *a], cwd=self.root, capture_output=True, text=True, check=True)  # noqa: E731
        git("init", "-q")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        (self.root / "a.txt").write_text("1")
        git("add", "a.txt")
        git("commit", "-qm", "base")
        self.patch(REPO=self.root)
        head0 = self.lp.git_head()
        self.assertIsNone(self.lp.undo_commits(head0))
        (self.root / "a.txt").write_text("2")
        (self.root / "b.txt").write_text("new")
        git("add", "-A")
        git("commit", "-qm", "role")
        note = self.lp.undo_commits(head0)
        self.assertEqual(note["was"], head0)
        self.assertEqual(self.lp.git_head(), head0)
        self.assertEqual(self.lp.changed_files(), ["a.txt", "b.txt"])

    def test_climbing_out_through_a_link_is_flagged(self):
        sandbox, origin = self.root / "sb", self.root / "origin"
        (origin / "third_party").mkdir(parents=True)
        (origin / "outputs" / "scene_discovery" / "scn" / "mesh").mkdir(parents=True)
        (origin / "outputs" / "secret").mkdir()
        (sandbox / "outputs" / "scene_discovery").mkdir(parents=True)
        os.symlink(origin / "third_party", sandbox / "third_party")
        os.symlink(origin / "outputs" / "scene_discovery" / "scn", sandbox / "outputs" / "scene_discovery" / "scn")
        self.patch(REPO=sandbox, ORIGIN=origin)
        targets = self.lp.own_link_targets()
        bad = "ls outputs/scene_discovery/scn/mesh/../../../"
        self.assertTrue(self.lp.climbs_out(json.dumps([["Bash", {"command": bad}, None]]), targets))
        self.assertTrue(self.lp.climbs_out("cd third_party/..", targets))
        self.assertFalse(self.lp.climbs_out("ls outputs/scene_discovery/scn/mesh/../ third_party/x/../y", targets))
        self.assertFalse(self.lp.climbs_out("ls ./src/../outputs", targets))

    def test_voided_drafts_move_to_the_hidden_tree(self):
        hid = self.root / "hid"
        hid.mkdir()
        self.patch(HIDDEN=hid)
        cand = self.root / "loop" / "cand_2"
        cand.mkdir(parents=True)
        (cand / "f.py").write_text("x")
        patch = self.root / "loop" / "code_2.patch"
        patch.write_text("p")
        self.lp.hide_void(4, cand, patch)
        self.assertFalse(cand.exists() or patch.exists())
        self.assertTrue((hid / "void" / "iter_04" / "cand_2" / "f.py").exists())

    def test_the_role_env_drops_claude_md_files_and_scratch_is_per_role(self):
        from unittest import mock

        seen = {}

        def run_role(cmd, prompt, env, session, *a, **k):
            seen.update(env)
            return json.dumps(dict(structured_output={"ok": 1}, total_cost_usd=0)), "", 0, []

        self.patch(SCRATCH_ROOT=self.root / "scratch", REPO=self.root)
        with mock.patch.object(self.lp, "run_role", run_role), mock.patch.object(self.lp, "git_head", lambda: ""):
            self.lp.claude("engineer_2", "p", {}, self.root, "")
        self.assertEqual(seen["CLAUDE_CODE_DISABLE_CLAUDE_MDS"], "1")
        self.assertTrue(seen["TMPDIR"].endswith("engineer_2"))
        self.assertEqual(self.lp.role_dir("manager"), self.lp.role_tmp())


class LeakPolicy(unittest.TestCase):
    """TODO 36.7 (user 2026-10-22, "按你的建议的来"): a leak voids what the leaking role produced and the loop goes on;
    two voided iterations in a row stop it abnormally; a voided Manager's `stop` decides nothing; the streak stop
    ("3 rejections in a row and no new hypothesis") is gone."""

    def setUp(self):
        from unittest import mock

        self.lp = _scene_loop()
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        root = Path(self.d.name)
        (root / "pool_diag").mkdir()
        hid = root / "outputs" / "scene_rsi_20261016_hidden" / "fam"  # as RoleSafety: never HIDDEN at Path()
        hid.mkdir(parents=True)
        for p in (mock.patch.multiple(self.lp, STATE=root / "state", LOOP=root, HIDDEN=hid, GATE_DIR=hid / "gate",
                                      accept_pool=lambda: root),
                  mock.patch.dict(os.environ)):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("RSI_MAX_WALL_H", None)
        (root / "state").mkdir()

    def rows(self, *rows):
        for r in rows:
            self.lp.ledger(r)

    def test_voided_in_a_row_counts_consecutive_iterations_back_from_the_newest(self):
        v = self.lp.voided_in_a_row
        self.assertEqual(v([]), [])
        es = [dict(step="review", iter=0), dict(step="void_iter", iter=0), dict(step="review", iter=1)]
        self.assertEqual(v(es), [])
        es.append(dict(step="void_iter", iter=1))
        self.assertEqual(v(es), [0, 1])
        self.assertEqual(v([dict(step="void_iter", iter=0), dict(step="void_iter", iter=2)]), [2])  # 1 has no rows
        es.append(dict(step="void", iter=1))  # --reprobe 1 takes the iteration up again
        self.assertEqual(v(es), [])

    def test_two_voided_iterations_in_a_row_stop_the_loop_abnormally(self):
        from argparse import Namespace
        from unittest import mock

        def iterate(n, meta, stub):  # every Reviewer leaks
            self.rows(dict(step="review", iter=n, void=True, leaks=["gate"]))
            self.lp.void_iteration(n, "reviewer", ["gate"])
            return None

        with mock.patch.object(self.lp, "iterate", iterate):
            rc = self.lp.cmd_run(Namespace(max_iters=5, reprobe=None, stub=True))
        self.assertEqual(rc, 3)
        es = self.lp.entries()
        self.assertEqual([e["iter"] for e in es if e["step"] == "void_iter"], [0, 1])
        self.assertEqual(es[-1]["step"], "stop")
        self.assertTrue(es[-1]["reason"].startswith("audit: iterations 0, 1 voided in a row"))

    def test_one_voided_iteration_and_the_loop_goes_on(self):
        from argparse import Namespace
        from unittest import mock

        def iterate(n, meta, stub):  # iterations 0 and 2 leak, 1 and 3 run through
            self.rows(dict(step="review", iter=n, void=n % 2 == 0))
            if n % 2 == 0:
                self.lp.void_iteration(n, "composer", ["x"])
            return None

        with mock.patch.object(self.lp, "iterate", iterate):
            rc = self.lp.cmd_run(Namespace(max_iters=4, reprobe=None, stub=True))
        self.assertEqual(rc, 0)
        self.assertEqual(len([e for e in self.lp.entries() if e["step"] == "review"]), 4)
        self.assertFalse([e for e in self.lp.entries() if e["step"] == "stop"])

    def test_a_voided_managers_stop_decides_nothing_and_rejections_never_stop(self):
        meta = dict(started_s=time.time())
        self.rows(dict(step="manager", iter=0, decision="stop", void=True))
        self.assertIsNone(self.lp.stop_reason(meta))
        for i in range(1, 6):  # five rejections in a row, the last review without hypotheses: no stop any more
            self.rows(
                dict(step="review", iter=i, n_hypotheses=0), dict(step="gate", iter=i, accepted=False, hash=f"h{i}")
            )
        self.assertIsNone(self.lp.stop_reason(meta))
        self.assertFalse(hasattr(self.lp, "MAX_REJECT_STREAK"))
        self.rows(dict(step="manager", iter=6, decision="stop"))
        self.assertEqual(self.lp.stop_reason(meta), "manager: stop")


class ManagerHistory(unittest.TestCase):
    """TODO 36.7 (user 2026-10-22): the Manager reads the last two iterations' ledger rows in full and each earlier
    iteration as one line (32.12's iteration 5 Manager context: 93 -> 42 KB)."""

    def test_older_iterations_become_one_line_each(self):
        import inspect
        from unittest import mock

        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            st = Path(d)
            for n in range(4):
                (st / f"iter_{n:02d}").mkdir()
                (st / f"iter_{n:02d}" / "hypotheses.json").write_text(
                    json.dumps(dict(hypotheses=[dict(id="H1", claim=f"claim of iteration {n} " + "x" * 300)]))
                )
            es = []
            for n in range(4):
                es += [
                    dict(t="t", step="review", iter=n, n_hypotheses=1, n_claims=0, usage={"big": 1}),
                    dict(step="probe", iter=n, results=[dict(id="H1", n=30, ok_before=10, ok_after=12,
                                                             predicted_fixed_hit=1, predicted_fixed_n=2)]),
                    dict(step="manager", iter=n, decision="engineer", hypotheses=["H1", "H1"], reason="r" * 400),
                    dict(step="engineer", iter=n, candidate=1, hypothesis="H1", mechanism="m", void=False),
                    dict(step="engineer", iter=n, candidate=2, hypothesis="H1", mechanism="m", void=True),
                    dict(step="compose", iter=n, decision="adopt", adopted=1, notes="n"),
                    dict(step="gate", iter=n, accepted=False, hash=f"h{n}", reason="p",
                         result=dict(b=5, c=1, p_one_sided=0.109375)),
                ]  # fmt: skip
                if n == 0:
                    es.append(dict(t="t", step="stop", reason="wall clock >= 5 h"))
            with mock.patch.object(lp, "STATE", st):
                h = lp.compact_history(es)
        self.assertEqual(len(h), 3 + 2 * 7)  # iteration 0, the stop, iteration 1 as lines; 2 and 3 row by row
        line0, stop, line1 = h[:3]
        self.assertTrue(line0.startswith("iteration 0 | review: 1 hypotheses"))
        self.assertIn("H1 claim of iteration 0", line0)
        self.assertIn("probe H1 10->12/30 (predicted fixed 1/2)", line0)
        self.assertIn("manager engineer H1,H1", line0)
        self.assertIn("engineer 2 H1: void", line0)
        self.assertIn("composer adopt 1", line0)
        self.assertIn("gate rejected b 5 c 1 p 0.109 h0", line0)
        self.assertLess(len(line0), 1200)
        self.assertIn("stop", stop)
        self.assertTrue(line1.startswith("iteration 1 |"))
        self.assertEqual([e["iter"] for e in h[3:]], [2] * 7 + [3] * 7)
        self.assertNotIn("usage", h[3])
        self.assertEqual(lp.compact_history(es, full=10)[0]["step"], "review")  # nothing older: all rows
        self.assertIn("compact=True", inspect.getsource(lp.after_review))  # only the Manager reads it compact


class ExperienceLibrary(unittest.TestCase):
    """2026-10-21: the library's patches named other sandboxes' hidden trees, and a Reviewer that writes such a name
    voids its iteration. Patches carry a/ and b/ roots; the library holds no path a role may not name."""

    def test_a_patch_is_written_with_relative_roots(self):
        import re

        _path = list(sys.path)
        sys.path.insert(0, str(HERE))
        try:
            reflect = _load("scene_rsi_reflect", HERE / "reflect.py")
        finally:
            sys.path[:] = _path
        a, b = "/w/s/outputs/x/incumbents/h/skills", "/w/s/outputs/x_hidden/f/gate/02/cand/skills"
        raw = f"diff -ruN {a}/pick.py {b}/pick.py\n--- {a}/pick.py\t2026\n+++ {b}/pick.py\t2026\n@@ -1 +1 @@\n-x\n+y\n"
        out = reflect.relabel(raw, a, b)
        self.assertNotIn("_hidden", out)
        self.assertTrue(re.search(r"^--- a/pick\.py", out, re.M) and re.search(r"^\+\+\+ b/pick\.py", out, re.M))

    def test_a_sandbox_gets_the_library_without_its_own_family(self):
        sb = _load("scene_rsi_sandbox", HERE / "sandbox.py")
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "scripts" / "rsi" / "experience"
            for fam in ("grasp_atomic", "pick_single"):
                (lib / fam).mkdir(parents=True)
                (lib / fam / "entry.json").write_text(json.dumps(dict(experience=dict(cue=f"cue of {fam}"))))
            (lib / "INDEX.md").write_text("grasp_atomic pick_single\n")
            sb.library_without(Path(d), "grasp_atomic")
            index = (lib / "INDEX.md").read_text()
            self.assertFalse((lib / "grasp_atomic").exists())
            self.assertIn("cue of pick_single", index)
            self.assertNotIn("grasp_atomic", index)
            sb.library_without(Path(d) / "none", "grasp_atomic")  # a sandbox without the library: nothing to do

    def test_the_library_names_no_off_limits_path(self):
        import re

        bad = re.compile(r"_hidden|rsi_kit|\.claude/|/workspace/")
        for p in sorted((HERE / "experience").rglob("*")):
            if p.is_file():
                self.assertIsNone(bad.search(p.read_text(errors="replace")), p)


def _line(*blocks) -> str:
    """An assistant line as Claude Code writes it, with `wireToolInputs`, its copy of every call's input by id."""
    wire = {b["id"]: b["input"] for b in blocks if b["type"] == "tool_use"}  # every block: stop_reason "tool_use"
    msg = dict(role="assistant", content=list(blocks), stop_reason="tool_use")
    return json.dumps(dict(type="assistant", message=msg, wireToolInputs=wire)) + "\n"


def _call(name: str, **inp) -> dict:
    return dict(type="tool_use", id=f"t_{name}", name=name, input=inp)


class TranscriptAudit(unittest.TestCase):
    """TODO 32.13 (user 2026-10-22): the audit reads the inputs of every tool call but the role's answer, and not its
    thinking or text -- 32.12's iteration 4 was voided because the Reviewer's StructuredOutput said it had not used
    the hidden tree. A review voided that way is
    audited again before `--reprobe` reuses its hypotheses; a run resumed after its own stop does not count the
    hours it stood stopped."""

    HID = "outputs/scene_rsi_20261016_hidden/grasp_atomic"

    def setUp(self):
        from unittest import mock

        self.lp = _scene_loop()
        self.lp.configure("grasp_atomic")
        self.d = tempfile.TemporaryDirectory()
        self.state = Path(self.d.name)
        self.files = {}
        for p in (mock.patch.object(self.lp, "STATE", self.state),
                  mock.patch.object(self.lp, "transcript", lambda s: self.files.get(s))):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.d.cleanup)

    def session(self, name: str, *lines: str) -> str:
        p = self.state / f"{name}.jsonl"
        p.write_text("".join(lines))
        self.files[name] = p
        return name

    def test_the_answer_is_not_audited_and_every_other_call_is(self):
        answer = _call("StructuredOutput", notes=f"Not used: anything under {self.HID}")
        said = [dict(type=k, **{k: f"I did not open {self.HID}"}) for k in ("thinking", "text")]
        peek = _call("Bash", command=f"ls {self.HID}/")
        self.assertEqual(self.lp.transcript_leaks(self.session("a", _line(answer), *(_line(b) for b in said))), [])
        blob = json.dumps(dict(type="api-request-blob", message=dict(content=[*said, answer]))) + "\n"
        self.assertEqual(self.lp.transcript_leaks(self.session("e", blob)), [])
        self.assertEqual(self.lp.transcript_leaks(self.session("b", _line(peek))), ["gate"])
        self.assertEqual(self.lp.transcript_leaks(self.session("c", _line(peek, answer))), ["gate"])
        self.assertEqual(self.lp.transcript_leaks(self.session("d", _line(_call("Read", file_path=f"/x/{self.HID}")))),
                         ["gate"])  # fmt: skip

    def review(self, session: str) -> None:
        self.lp.ledger(dict(step="review", iter=4, n_hypotheses=4, void=True, leaks=["gate"], session=session))

    def test_a_voided_review_passes_the_fixed_audit_before_reuse(self):
        self.review(self.session("ok", _line(_call("StructuredOutput", notes=f"Not used: {self.HID}"))))
        self.lp.reaudit(4)
        last = self.lp.entries()[-1]
        self.assertEqual((last["step"], last["iter"], last["leaks"], last["was"]), ("reaudit", 4, [], ["gate"]))

    def test_a_review_that_still_names_the_tree_or_lost_its_transcript_is_refused(self):
        self.review(self.session("bad", _line(_call("Bash", command=f"cat {self.HID}/pool"))))
        with self.assertRaises(SystemExit):
            self.lp.reaudit(4)
        self.review("gone")
        with self.assertRaises(SystemExit):
            self.lp.reaudit(4)
        self.assertNotIn("reaudit", [e["step"] for e in self.lp.entries()])

    def stopped(self, hours_ago: float) -> None:
        t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours_ago * 3600))
        with (self.state / "ledger.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(t=t, step="stop", reason="x")) + "\n")

    def test_a_resumed_run_does_not_count_the_hours_it_stood_stopped(self):
        meta = dict(started_s=time.time() - 25 * 3600)
        self.stopped(21.5)  # ran 3.5 h, stood stopped 21.5 h
        self.lp.resume(meta)
        self.assertAlmostEqual(self.lp.wall_h(meta), 3.5, places=2)
        self.assertEqual(self.lp.entries()[-1]["step"], "resume")
        self.lp.resume(meta)  # the last entry is no longer a stop: nothing more is taken off
        self.assertAlmostEqual(self.lp.wall_h(meta), 3.5, places=2)

    def test_a_stop_from_before_the_run_started_is_not_a_pause(self):
        meta = dict(started_s=time.time() - 3600)
        self.stopped(2)  # a replay's ledger copied from its source run
        self.lp.resume(meta)
        self.assertAlmostEqual(self.lp.wall_h(meta), 1.0, places=2)
        self.assertNotIn("paused_s", meta)
