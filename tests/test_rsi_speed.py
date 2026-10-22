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
