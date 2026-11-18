"""Mode 3 of the scene RSI inner loop, scripts/rsi/alternate.py (user 2026-11-18): two families take turns, one
iteration each, share one knowledge library and one skills package. The merge (three-way, the receiver's side kept
where both changed the same lines), the folding of the sender's overrides, the sync candidate, the driver's state
machine with a fake runner (turns, stops, blocks, resumes, a failed build, a rejected or broken sync gate), the lock,
publishing; gate.py's non-inferior rule and loop.py's `sync` ledger row."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import knowledge as KN  # noqa: E402


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "rsi" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


AL = _load("rsi_alternate_under_test", "alternate.py")


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


BASE = {"pick.py": "A = 1\nB = 2\nC = 3\n", "place.py": "def f():\n    return 1\n", "bimanual.py": "D = 4\n"}


class Merge(unittest.TestCase):
    def merge(self, ours: dict, theirs: dict, base: dict = BASE) -> dict:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rep = AL.merge_trees(write(root / "b", base), write(root / "o", ours), write(root / "t", theirs))
        return rep

    def test_a_file_only_the_sender_changed_comes_over_and_its_additions_and_deletions_too(self):
        theirs = dict(BASE, **{"pick.py": "A = 10\nB = 2\nC = 3\n", "new.py": "X = 1\n"})
        del theirs["place.py"]
        rep = self.merge(dict(BASE), theirs)
        self.assertEqual(rep["texts"], {"pick.py": "A = 10\nB = 2\nC = 3\n", "new.py": "X = 1\n", "place.py": None})
        self.assertEqual((rep["merged"], rep["copied"], rep["deleted"], rep["conflicts"]),
                         (["pick.py"], ["new.py"], ["place.py"], []))  # fmt: skip

    def test_a_file_only_the_receiver_changed_stays(self):
        rep = self.merge(dict(BASE, **{"bimanual.py": "D = 40\n"}), dict(BASE))
        self.assertEqual(rep["texts"], {})

    def test_both_changing_different_lines_merges_cleanly(self):
        rep = self.merge(dict(BASE, **{"pick.py": "A = 1\nB = 2\nC = 30\n"}),
                         dict(BASE, **{"pick.py": "A = 10\nB = 2\nC = 3\n"}))  # fmt: skip
        self.assertEqual(rep["texts"], {"pick.py": "A = 10\nB = 2\nC = 30\n"})
        self.assertEqual(rep["conflicts"], [])

    def test_both_changing_the_same_lines_keeps_the_receivers_and_records_it(self):
        base = {"pick.py": "A = 1\n\n\n\nB = 2\n"}
        rep = self.merge({"pick.py": "A = 7\n\n\n\nB = 2\n"}, {"pick.py": "A = 9\n\n\n\nB = 20\n"}, base)
        self.assertEqual(rep["texts"]["pick.py"], "A = 7\n\n\n\nB = 20\n")  # A: the receiver's; B: the sender's
        self.assertEqual(rep["conflicts"], ["pick.py: 1 hunk(s) changed on both sides"])

    def test_deleted_on_one_side_and_changed_on_the_other_keeps_the_receivers_whole(self):
        theirs = dict(BASE)
        del theirs["place.py"]
        rep = self.merge(dict(BASE, **{"place.py": "def f():\n    return 2\n"}), theirs)
        self.assertEqual(rep["texts"], {})
        self.assertEqual(rep["conflicts"], ["place.py: deleted on one side, changed on the other"])

    def test_the_same_change_on_both_sides_is_no_change(self):
        same = dict(BASE, **{"pick.py": "A = 5\nB = 2\nC = 3\n"})
        self.assertEqual(self.merge(same, dict(same))["texts"], {})


class Fold(unittest.TestCase):
    def test_overrides_are_written_into_the_constants_and_a_package_level_one_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            skills = write(Path(d), {"pick.py": 'LIFT_M = 0.10  # lift\nTABLE = {"k": 1}\n'})
            texts, unfolded = AL.fold_set(skills, {"pick.LIFT_M": 0.3, "pick.TABLE.k": 2, "PLAN": "x"}, "fam")
        self.assertIn("LIFT_M = 0.3", texts["pick.py"])
        self.assertIn('"k": 2', texts["pick.py"])
        self.assertEqual(unfolded, ["override PLAN: a package-level parameter has no module"])


def git_init(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                                                 "--allow-empty", "-m", "x"]):  # fmt: skip
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def fake_run(root: Path, fams=("fa", "fb")) -> Path:
    """A run directory as init leaves it, without building real sandboxes: two git sandboxes with the skills package,
    the empty library, the merge base and the state."""
    run = root / "run"
    for side in AL.SIDES:
        sb = run / side
        write(sb / AL.SKILLS_REL, BASE)
        (sb / ".gitignore").write_text("scripts/rsi/knowledge/KNOWLEDGE.md\noutputs/\n")
        git_init(sb)
    (run / "knowledge").mkdir()
    AL.copy_skills(run / "a" / AL.SKILLS_REL, run / "sync" / "base" / "skills")
    AL.jdump(run / AL.STATE_FILE, dict(families=list(fams), tags=["single", "bimanual"], ref="x", created="t", iter=0,
                                       turn=0, phase="loop", loop_mark=None, stopped={}, blocked=None, events=[]))  # fmt: skip
    return run


def set_incumbent(run: Path, side: str, fam: str, files: dict, overrides: dict | None = None) -> Path:
    """The family's gate incumbent: a candidate directory with these skills files (on top of BASE)."""
    d = run / side / AL.HIDDEN / fam / "gate" / "01_x" / "cand"
    write(d / "skills", dict(BASE, **files))
    (d / "candidate.json").write_text(json.dumps({"set": overrides or {}}))
    (run / side / AL.HIDDEN / fam / "gate" / "incumbent.json").write_text(json.dumps({"dir": str(d)}))
    return d


class Sync(unittest.TestCase):
    def test_the_candidate_is_the_receivers_package_with_the_senders_folded_changes_and_its_own_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            st = json.loads((run / AL.STATE_FILE).read_text())
            set_incumbent(run, "a", "fa", {"pick.py": "A = 1\nB = 2\nC = 3\nE = 5\n"}, {"pick.A": 11})
            set_incumbent(run, "b", "fb", {"bimanual.py": "D = 44\n"}, {"bimanual.D": 45})
            prep = AL.prepare_sync(run, st, 0, 1)
            cand = Path(prep["cand"])
            self.assertIn("A = 11", (cand / "skills/pick.py").read_text())  # the sender's override, folded
            self.assertIn("E = 5", (cand / "skills/pick.py").read_text())
            self.assertEqual((cand / "skills/bimanual.py").read_text(), "D = 44\n")  # the receiver's own code
            self.assertEqual(json.loads((cand / "candidate.json").read_text()), {"set": {"bimanual.D": 45}})
            self.assertEqual(prep["report"]["merged"], ["pick.py"])

    def test_nothing_to_merge_gives_no_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            st = json.loads((run / AL.STATE_FILE).read_text())
            set_incumbent(run, "b", "fb", {"bimanual.py": "D = 44\n"})
            self.assertIsNone(AL.prepare_sync(run, st, 0, 1)["cand"])  # fa has no incumbent: the base, unchanged


class FakeRunner:
    """Records the calls; `loops` maps a family to the exit code and an optional stop reason per call."""

    def __init__(self, run: Path, stops=None, rc=None, gate=None, build_error=False):
        self.run, self.calls = run, []
        self.stops, self.rc, self.gate, self.build_error = stops or {}, rc or {}, gate, build_error

    def loop(self, sb: Path, family: str, tag: str) -> int:
        self.calls.append(("loop", family))
        # what the knowledge step will read: the library is in the sandbox and committed
        assert (sb / "scripts/rsi/knowledge").is_dir()
        led = sb / AL.VISIBLE / family / "loop" / "ledger.jsonl"
        led.parent.mkdir(parents=True, exist_ok=True)
        n = sum(1 for c in self.calls if c == ("loop", family)) - 1
        with led.open("a") as fh:
            fh.write(json.dumps(dict(step="gate", iter=n, accepted=True)) + "\n")
            if family in self.stops:
                fh.write(json.dumps(dict(step="stop", reason=self.stops[family])) + "\n")
        if family == "fa":  # fa's iteration changes pick.py
            set_incumbent(self.run, "a", "fa", {"pick.py": f"A = {n + 100}\nB = 2\nC = 3\n"})
        return self.rc.get(family, 0)

    def build(self, sb: Path, family: str) -> dict:
        self.calls.append(("build", family))
        if self.build_error:
            raise RuntimeError("consolidator down")
        lib = self.run / "knowledge"
        KN.save_entries(lib, [e for e in KN.entries(lib) if KN.source_family(e) != family]
                        + [dict(id=f"{family}/m", tag="single" if family == "fa" else "bimanual",
                                source=dict(family=family))])  # fmt: skip
        return dict(entries=1, cost_usd=0.0, tag="x")

    def sync_gate(self, sb: Path, family: str, cand: Path, source: dict, tag: str):
        self.calls.append(("sync", family))
        if self.gate == "error":
            return 3, {"error": "gate error"}
        accepted = self.gate != "reject"
        if accepted:  # as gate.py: the accepted candidate is the receiver's incumbent
            inc = sb / AL.HIDDEN / family / "gate" / "incumbent.json"
            inc.parent.mkdir(parents=True, exist_ok=True)
            inc.write_text(json.dumps({"dir": str(cand)}))
        return (0 if accepted else 1), dict(accepted=accepted, reason="ok" if accepted else "losses 2 > wins 0",
                                            result=dict(b=0, c=0 if accepted else 2), hash="h")  # fmt: skip


class Driver(unittest.TestCase):
    def test_one_iteration_is_a_half_of_each_family_with_build_and_sync_after_each(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            r = FakeRunner(run)
            why = AL.run_all(run, 1, r)
            st = json.loads((run / AL.STATE_FILE).read_text())
            syncs = KN.jl(run / "knowledge" / KN.SYNCS)
            base = (run / "sync/base/skills/pick.py").read_text()
            in_b = (run / "b/scripts/rsi/knowledge/knowledge.jsonl").read_text()
            git_clean = subprocess.run(["git", "status", "--porcelain"], cwd=run / "b", capture_output=True,
                                       text=True).stdout  # fmt: skip
        self.assertEqual(why, "1 iterations done")
        # fa's half: loop, build, sync into fb (fa changed pick.py); fb's half: loop, build, nothing to merge back
        self.assertEqual(r.calls, [("loop", "fa"), ("build", "fa"), ("sync", "fb"), ("loop", "fb"), ("build", "fb")])
        self.assertEqual((st["iter"], st["turn"], st["phase"]), (1, 0, "loop"))
        self.assertEqual([(s["from"], s.get("accepted"), s.get("skipped")) for s in syncs],
                         [("fa", True, None), ("fb", None, "the merge changes nothing")])  # fmt: skip
        self.assertEqual(base, "A = 100\nB = 2\nC = 3\n")  # the accepted merge is the next base
        self.assertIn("fa/m", in_b)  # fb's loop read fa's entries
        self.assertEqual(git_clean.strip(), "")  # the installed library is committed: no stray for the loop

    def test_a_rejected_sync_leaves_the_base_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            AL.run_all(run, 1, FakeRunner(run, gate="reject"))
            syncs = KN.jl(run / "knowledge" / KN.SYNCS)
            base = (run / "sync/base/skills/pick.py").read_text()
        self.assertEqual((syncs[0]["accepted"], syncs[0]["reason"]), (False, "losses 2 > wins 0"))
        self.assertEqual(base, BASE["pick.py"])

    def test_a_stopped_family_skips_its_halves_and_nothing_is_merged_into_it(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            r = FakeRunner(run, stops={"fb": "the incumbent wins 34/40 >= 80%: sealed set"})
            AL.run_all(run, 2, r)
            st = json.loads((run / AL.STATE_FILE).read_text())
        self.assertEqual(list(st["stopped"]), ["fb"])
        self.assertEqual(r.calls.count(("loop", "fb")), 1)
        self.assertEqual(r.calls.count(("loop", "fa")), 2)
        self.assertEqual(r.calls.count(("sync", "fb")), 1)  # iteration 2: fb has stopped, fa's code is not merged in

    def test_an_abnormal_loop_exit_blocks_the_run_and_a_new_run_takes_it_up_again(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            r = FakeRunner(run, rc={"fa": 3})
            why = AL.run_all(run, 1, r)
            self.assertTrue(why.startswith("blocked: fa's loop exited 3"))
            st = json.loads((run / AL.STATE_FILE).read_text())
            self.assertEqual((st["phase"], st["turn"]), ("loop", 0))
            r2 = FakeRunner(run)
            self.assertEqual(AL.run_all(run, 1, r2), "1 iterations done")
            self.assertEqual(r2.calls[0], ("loop", "fa"))  # the blocked half runs again

    def test_a_failed_build_is_recorded_and_the_run_goes_on(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            why = AL.run_all(run, 1, FakeRunner(run, build_error=True))
            st = json.loads((run / AL.STATE_FILE).read_text())
        self.assertEqual(why, "1 iterations done")
        self.assertTrue(any("consolidator down" in (e.get("error") or "") for e in st["events"]))

    def test_a_broken_sync_gate_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            why = AL.run_all(run, 1, FakeRunner(run, gate="error"))
            st = json.loads((run / AL.STATE_FILE).read_text())
        self.assertTrue(why.startswith("blocked: fb's sync gate failed"))
        self.assertEqual(st["phase"], "sync")  # run again: the sync is tried again

    def test_the_lock_refuses_a_second_driver(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "alternate.lock").write_text(str(os.getpid()))
            with self.assertRaises(SystemExit):
                with AL.Lock(run):
                    pass
            (run / "alternate.lock").write_text("999999999")  # a dead driver's lock is taken over
            with AL.Lock(run):
                self.assertEqual((run / "alternate.lock").read_text(), str(os.getpid()))

    def test_publish_replaces_the_run_families_entries_and_keeps_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            run = fake_run(Path(d))
            KN.save_entries(run / "knowledge", [dict(id="fa/new", tag="single", source=dict(family="fa"),
                                                     scope="shared", status="candidate", condition={},
                                                     evidence=dict(by_object={}), signature={})])  # fmt: skip
            out = Path(d) / "lib"
            KN.save_entries(out, [dict(id="fa/old", source=dict(family="fa")), dict(id="zz/keep", source=dict(family="zz"),
                                                                                   scope="s", status="c", condition={},
                                                                                   evidence=dict(by_object={}), signature={})])  # fmt: skip
            AL.publish(run, out)
            ids = sorted(e["id"] for e in KN.entries(out))
        self.assertEqual(ids, ["fa/new", "zz/keep"])


class NonInferiorGate(unittest.TestCase):
    def test_c_at_most_b_passes(self):
        GA = _load("rsi_gate_under_test", "gate.py")
        self.assertEqual(GA.non_inferior_fails(dict(b=0, c=0)), [])
        self.assertEqual(GA.non_inferior_fails(dict(b=2, c=2)), [])
        self.assertEqual(GA.non_inferior_fails(dict(b=1, c=2)), ["losses 2 > wins 1"])
        self.assertNotEqual(GA.SYNC_RULE, GA.GATE_RULE)


class LoopSync(unittest.TestCase):
    def test_the_sync_verdict_is_a_gate_row_without_an_iteration(self):
        names = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "design", "sealed", "loop")
        saved = {k: sys.modules.pop(k, None) for k in names}
        try:
            LP = _load("rsi_loop_sync_under_test", "loop.py")
        finally:
            for k, v in saved.items():
                sys.modules.pop(k, None)
                if v is not None:
                    sys.modules[k] = v
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            orig = (LP.REPO, LP.subprocess.run)
            LP.REPO = root
            LP.configure("fam")
            (LP.HIDDEN / "pool_accept").mkdir(parents=True)
            cand = write(root / "cand", {"skills/pick.py": "A = 1\n", "candidate.json": '{"set": {}}'})
            h = LP.candidate_hash(cand)

            def fake(args, **kw):  # gate.py: one verdict line in the gate's ledger
                LP.GATE_DIR.mkdir(parents=True, exist_ok=True)
                with (LP.GATE_DIR / "ledger.jsonl").open("a") as fh:
                    fh.write(json.dumps(dict(hash=h, accepted=True, reason="accepted (non-inferior)", rule="sync",
                                             result=dict(b=1, c=0, n=40, candidate_ok=20, incumbent_ok=19))) + "\n")  # fmt: skip
                assert "--non-inferior" in args
                return subprocess.CompletedProcess(args, 0, "", "")

            LP.subprocess.run = fake
            try:
                rc = LP.main(["sync", "fam", "--candidate", str(cand), "--source", json.dumps({"from": "other"})])
            finally:
                LP.REPO, LP.subprocess.run = orig
            rows = [json.loads(x) for x in (LP.STATE / "ledger.jsonl").read_text().splitlines()]
        self.assertEqual(rc, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["step"], rows[0]["accepted"], rows[0]["sync"], rows[0]["rule"]),
                         ("gate", True, {"from": "other"}, "sync"))  # fmt: skip
        self.assertNotIn("iter", rows[0])  # a gate call, not an iteration


if __name__ == "__main__":
    unittest.main()
