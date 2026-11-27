"""TODO 37.4 (E, user 2026-10-23: "B 到 E 按顺序做"): after the gate rejects the composed candidate, the best draft
it has not judged goes to the gate in the same iteration (one more gate call instead of another iteration), and each
draft's diagnostic rounds start as soon as its Engineer answers."""

import json
import sys
import tempfile
import threading
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
        return _load("scene_rsi_loop_37_4", HERE / "loop.py")
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _cand(number, h, fixed, broken, hyp="H1"):
    return dict(number=number, hypothesis=hyp, hash=h, dir=f"/x/candidate_{number}", fixed=fixed, broken=broken)


class SecondShot(unittest.TestCase):
    def test_pick_prefers_the_largest_net_then_fewer_broken_and_skips_judged_hashes(self):
        lp = _scene_loop()
        cands = [_cand(1, "a", ["s1", "s2"], ["s3"]), _cand(2, "b", ["s1", "s2", "s4"], ["s3", "s5"]),
                 _cand(3, "c", ["s1"], []), _cand(4, "d", ["s1", "s2"], ["s3"])]  # fmt: skip
        # every draft nets 1 start: the fewest broken wins (draft 3), then the lower number (1 before 4)
        self.assertEqual(lp.second_shot_pick(cands, "zzz", set())["number"], 3)
        self.assertEqual(lp.second_shot_pick(cands, "c", set())["number"], 1)  # the gated hash is out; ties by number
        self.assertEqual(lp.second_shot_pick(cands, "c", {"a"})["number"], 4)  # rejected before: out
        self.assertIsNone(lp.second_shot_pick([_cand(1, "a", ["s1"], ["s2"])], "z", set()))  # net 0: nothing to send
        self.assertIsNone(lp.second_shot_pick([dict(number=1, hash="a", dir="/x")], "z", set()))  # no flips: not sent

    def test_a_rejection_sends_the_best_draft_and_keeps_the_first_verdict(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            st = Path(d)
            it_dir = st / "iter_03"
            it_dir.mkdir()
            (st / "meta.json").write_text(json.dumps(dict(started="2026-10-23T00:00:00Z")))
            (it_dir / "gate.json").write_text(json.dumps(dict(accepted=False, hash="composed", result=dict(b=2, c=1))))
            (it_dir / "gate.log").write_text("first")
            cands = [
                _cand(1, "composed", ["s1", "s2"], []),
                _cand(2, "b", ["s1"], []),
                _cand(3, "c", ["s1", "s4"], ["s2"]),
            ]
            rows, gated = [], []
            es = [dict(t="2026-10-23T01:00:00Z", step="gate", iter=2, accepted=False, hash="c", rule=lp.GATE_RULE)]
            with mock.patch.multiple(lp, STATE=st, entries=lambda: es, ledger=rows.append,
                                     gate_candidate=lambda n, it, cand, stub, **k: gated.append((str(cand), k)) or None):  # fmt: skip
                self.assertIsNone(lp.second_shot(3, it_dir, cands, "composed", False))
            self.assertEqual(
                gated, [("/x/candidate_2", {"shot": 2})]
            )  # c was rejected before, composed is the one judged
            self.assertEqual((rows[0]["step"], rows[0]["draft"], rows[0]["after"]), ("second_shot", 2, "composed"))
            self.assertEqual(json.loads((it_dir / "gate_shot1.json").read_text())["hash"], "composed")
            self.assertEqual((it_dir / "gate_shot1.log").read_text(), "first")
            # an accepted or void first verdict, or a spent budget: no second shot
            gated.clear()
            (it_dir / "gate.json").write_text(json.dumps(dict(accepted=True, hash="composed")))
            with mock.patch.multiple(
                lp, STATE=st, entries=lambda: es, ledger=rows.append, gate_candidate=lambda *a, **k: gated.append(a)
            ):
                self.assertIsNone(lp.second_shot(3, it_dir, cands, "composed", False))
            (it_dir / "gate.json").write_text(json.dumps(dict(accepted=False, void=True, hash="composed")))
            with mock.patch.multiple(
                lp, STATE=st, entries=lambda: es, ledger=rows.append, gate_candidate=lambda *a, **k: gated.append(a)
            ):
                self.assertIsNone(lp.second_shot(3, it_dir, cands, "composed", False))
            (it_dir / "gate.json").write_text(json.dumps(dict(accepted=False, hash="composed", result=dict(b=2, c=1))))
            full = [
                dict(t="2026-10-23T01:00:00Z", step="gate", iter=i, accepted=False, hash=f"g{i}")
                for i in range(lp.MAX_GATE_CALLS)
            ]
            with mock.patch.multiple(
                lp, STATE=st, entries=lambda: full, ledger=rows.append, gate_candidate=lambda *a, **k: gated.append(a)
            ):
                self.assertIsNone(lp.second_shot(3, it_dir, cands, "composed", False))
            self.assertEqual(gated, [])

    def test_the_gate_row_marks_the_second_shot_and_the_table_reads_it(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            st = Path(d)
            (st / "iter_00").mkdir()
            (st / "iter_00" / "hypotheses.json").write_text(json.dumps(dict(hypotheses=[
                dict(id="H1", claim="c1", instead="i", delta={"set": {}}), dict(id="H2", claim="c2", instead="i", delta={"set": {}})])))  # fmt: skip
            (st / "iter_00" / "candidates.json").write_text(json.dumps([
                dict(number=1, hypothesis="H1", fixed=["a"], broken=[]), dict(number=2, hypothesis="H2", fixed=["b"], broken=[])]))  # fmt: skip
            es = [dict(step="manager", iter=0, decision="engineer", hypotheses=["H1", "H2"], reason="r"),
                  dict(step="engineer", iter=0, candidate=1, hypothesis="H1", mechanism="m", void=False),
                  dict(step="engineer", iter=0, candidate=2, hypothesis="H2", mechanism="m", void=False),
                  dict(step="compose", iter=0, decision="adopt", adopted=1, notes="n"),
                  dict(step="gate", iter=0, accepted=False, hash="h1", reason="p", result=dict(b=1, c=1)),
                  dict(step="second_shot", iter=0, draft=2, hypothesis="H2", hash="h2", fixed=1, broken=0, after="h1"),
                  dict(step="gate", iter=0, accepted=True, hash="h2", reason="accepted", result=dict(b=3, c=0), shot=2)]  # fmt: skip
            with mock.patch.object(lp, "STATE", st):
                rows = {r["id"]: r for r in lp.mechanism_table(es)}
                line = lp.iteration_line(0, es)
        self.assertEqual((rows["H1"]["composer"], rows["H1"]["gate"]), ("adopted", "rejected b 1 c 1"))
        self.assertEqual(
            (rows["H2"]["composer"], rows["H2"]["gate"]), ("not adopted", "accepted b 3 c 0 (second shot)")
        )
        self.assertIn("second shot: draft 2 H2 (fixed 1 broken 0) after h1", line)
        self.assertIn("gate rejected h1: p | second shot", line)
        self.assertTrue(line.endswith("gate accepted h2: accepted"))  # the one-line history shows b c only with p

    def test_each_draft_runs_its_diagnostic_rounds_as_soon_as_it_answers(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            st = root / "loop"
            it_dir = st / "iter_00"
            it_dir.mkdir(parents=True)
            inc_dir = root / "inc"
            (inc_dir / "skills").mkdir(parents=True)
            (inc_dir / "skills" / "__init__.py").write_text("X = 1\n")
            (inc_dir / "candidate.json").write_text('{"set": {}}')
            inc = dict(dir=str(inc_dir), hash=lp.candidate_hash(inc_dir), set={})
            hyps = dict(hypotheses=[dict(id="H1", claim="c", instead="i", delta={"set": {}})])
            dec = dict(decision="engineer", hypotheses=["H1", "H1", "H1"], reason="r")
            rows, runs, gated = [], [], []
            gate_of_2 = threading.Event()

            def fake_claude(role, prompt, schema, it_dir_, tools, stub=None, stop=None):
                if role.startswith("engineer_"):
                    k = int(role.split("_")[1])
                    if k == 3:
                        gate_of_2.wait(20)  # the third answers only after draft 2's rounds have started
                    (it_dir / f"cand_{k}" / "skills" / "__init__.py").write_text(f"X = {k + 1}\n")
                    return (dict(mechanism=f"m{k}", files_changed=["__init__.py"], set={}, tests="t"),
                            dict(cost_usd=1.0, session=f"s{k}", seconds=1))  # fmt: skip
                return (dict(decision="adopt", adopted=1, notes="n", mechanism="m1"),
                        dict(cost_usd=1.0, session="sc", seconds=1))  # fmt: skip

            def fake_run_jobs(jobs, family, pool, procs, log, gpus, **k):
                runs.append((Path(log).name, [j.out.name for j in jobs]))
                if Path(log).name == "diag_2.log":
                    gate_of_2.set()
                return {}

            common = dict(RULES="r", PLAN="p", SIM_PY="py", POOL="pool", FAMILY="fam", FAMILY_FILE="ff", N_DIAG=30,
                          LIBRARY="", DIAG_INC="d", TMP="t")  # fmt: skip
            with (
                mock.patch.multiple(lp, STATE=st, LOOP=root, FAMILY="fam", CANDIDATES=3, ENGINEER_LATE_MIN=0,
                                    SECOND_SHOT=False, claude=fake_claude, ledger=rows.append,
                                    transcript_leaks=lambda s, samples=None: [], revert_strays=lambda b: [],
                                    put_skills=lambda p: None, named_objects=lambda s, names=None: [],
                                    context=lambda *a, **k: {}, pool_view=lambda *a, **k: [],
                                    gate_candidate=lambda n, it, cand, stub, **k: gated.append(str(cand)) or None),
                mock.patch.object(lp.B, "run_jobs", fake_run_jobs),
                mock.patch.object(lp, "flips", lambda a, b: dict(ok=0, incumbent_ok=0, n=0, fixed=[], broken=[])),
            ):  # fmt: skip
                self.assertIsNone(lp.multi_engineer_step(0, it_dir, inc, hyps, [], dec, common, [], stub=False))
            early = [r for r in runs if r[0].startswith("diag_")]
            self.assertEqual(sorted(r[0] for r in early), ["diag_1.log", "diag_2.log", "diag_3.log"])
            hashes = {r[0]: r[1][0] for r in early}
            self.assertTrue(all(h.startswith("candidate_") for h in hashes.values()))
            final = [r for r in runs if r[0] == "prober.log"]  # the main step's run covers whatever is missing
            self.assertEqual(len(final), 1)
            self.assertEqual(sorted(final[0][1]), sorted(hashes.values()))
            self.assertEqual(len(gated), 1)

    def test_the_flag_reaches_the_settings(self):
        from argparse import Namespace

        lp = _scene_loop()
        seen = {}
        with (
            mock.patch.object(lp, "cmd_run", lambda a: seen.update(vars(a)) or 0),
            mock.patch.object(lp, "configure", lambda f: None),
        ):
            lp.main(["run", "fam", "--no-second-shot"])
            self.assertIs(seen["second_shot"], False)
            lp.main(["run", "fam"])
            self.assertIsNone(seen["second_shot"])
        none = Namespace(second_shot=None)
        self.assertIs(lp.resolve_settings(none, {}, False)["second_shot"], True)
        self.assertIs(lp.resolve_settings(none, dict(started="x", second_shot=False), True)["second_shot"], False)


if __name__ == "__main__":
    unittest.main()
