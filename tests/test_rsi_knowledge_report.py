"""report/knowledge_report.py (user 2026-11-25): what the two families of a mode 3 run shared, from synthetic records."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("knowledge_report",
                                               REPO / "scripts" / "rsi" / "report" / "knowledge_report.py")  # fmt: skip
KR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(KR)
RR = KR.RR

T0 = datetime(2026, 11, 24, tzinfo=timezone.utc)


def at(minute: float) -> str:
    return (T0 + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def write_jl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def rounds(d: Path, wins: dict) -> None:
    for task, (w, n) in wins.items():
        for i in range(n):
            write(d / f"{task}_s{i}.json", dict(task=task, ok=i < w))


def family(sb: Path, name: str, task: str, obj: str) -> None:
    write(sb / "scripts" / "rsi" / "families" / f"{name}.json",
          dict(scene="scene", tasks_file=f"scripts/rsi/families/{name}_tasks.jsonl", tasks=[task]))  # fmt: skip
    write_jl(sb / "scripts" / "rsi" / "families" / f"{name}_tasks.jsonl", [dict(id=task, goal=[f"held({obj})"])])
    write(sb / "scene" / "scene_meta.json", dict(objects={obj: dict(free=True)}))


def entry(eid: str, reads_as: str, works: str, gate) -> dict:
    return dict(id=eid, tag="single" if eid.startswith("fa/") else "bimanual", generality="multi_object", version=1,
                status="candidate", condition={"all": []}, signature=dict(reads_as=reads_as), repair=dict(works=works),
                evidence=dict(by_object={"o1": {}, "o2": {}}), source=dict(family=eid.split("/")[0], gate=gate))  # fmt: skip


def cite(eid: str, family: str, it: int, hyp: str, picked: bool, accepted, outcome: str, status="verified") -> dict:
    return dict(entry=eid, run="x", family=family, iter=it, hypothesis=hyp, status=status, picked=picked,
                accepted=accepted, outcome=outcome)  # fmt: skip


class KnowledgeSharing(unittest.TestCase):
    """Two iterations of families fa (sandbox a) and fb (sandbox b). In iteration 2 fb's H2 cites fa's walk entry,
    is built as draft 2 and merged by the Composer into the accepted candidate; fa's `walk` cites fb's H2 entry and has
    no draft. fb's iteration-1 citation of its own entry does not cross."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = run = Path(self.tmp.name) / "run"
        a, b = run / "a", run / "b"
        family(a, "fa", "pick_a", "obj_a")
        family(b, "fb", "lift_b", "obj_b")
        la, lb = a / RR.OUT_REL / "fa" / "loop", b / RR.OUT_REL / "fb" / "loop"
        write_jl(la / "ledger.jsonl", [
            dict(t=at(10), step="gate", iter=0, hash="ha1", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=0, candidate_ok=1, n=2)),
            dict(t=at(12), step="manager", iter=1, hypothesis="walk", hypotheses=["walk"]),
            dict(t=at(20), step="gate", iter=1, hash="ha2", accepted=False, reason="stage 1: net wins 0 <= 0",
                 stage1=dict(n=2, candidate_ok=1, incumbent_ok=1)),
        ])  # fmt: skip
        write_jl(lb / "ledger.jsonl", [
            dict(t=at(30), step="gate", iter=0, hash="hb1", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=0, candidate_ok=1, n=2)),
            dict(t=at(40), step="manager", iter=1, hypothesis="H3", hypotheses=["H3", "H2", "H3"]),
            dict(t=at(45), step="compose", iter=1, decision="merge", merged=[1, 2],
                 mechanism="Draft 1 plus draft 2, at the new stance. Changed in one file."),
            dict(t=at(50), step="gate", iter=1, hash="hb2", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=1, candidate_ok=2, n=2)),
        ])  # fmt: skip
        write(la / "iter_01" / "hypotheses.json", dict(hypotheses=[
            dict(id="walk", claim="The hands sweep it.", knowledge_refs=["fb/i01_H2"])]))  # fmt: skip
        write(lb / "iter_01" / "hypotheses.json", dict(hypotheses=[
            dict(id="H2", claim="The walk sweeps a tall object. It topples. It is then lost.",
                 evidence="Seen in the contacts. The one-hand family learned it (fa/i00_walk): step in. Done.",
                 instead="End the walk 0.20 m behind the stance.", knowledge_refs=["fa/i00_walk"]),
            dict(id="H3", claim="Too far.")]))  # fmt: skip
        write(lb / "iter_01" / "candidates.json", [dict(number=1, hypothesis="H3"), dict(number=2, hypothesis="H2")])
        ha, hb = a / RR.HIDDEN_REL / "fa", b / RR.HIDDEN_REL / "fb"
        write_jl(ha / "gate" / "ledger.jsonl", [
            dict(call=1, hash="ha1", accepted=True, reason="accepted", started=at(5)),
            dict(call=2, hash="hsync", accepted=True, reason="accepted (non-inferior)", started=at(55))])  # fmt: skip
        write_jl(hb / "gate" / "ledger.jsonl", [
            dict(call=1, hash="hb1", accepted=True, reason="accepted", started=at(25)),
            dict(call=2, hash="hb2", accepted=True, reason="accepted", started=at(46))])  # fmt: skip
        rounds(ha / "gate" / "incumbent_seed_rounds", {"pick_a": (0, 2)})
        rounds(ha / "gate" / "01_ha1" / "candidate", {"pick_a": (1, 2)})
        rounds(ha / "gate" / "02_hsync" / "candidate", {"pick_a": (1, 2)})
        rounds(hb / "gate" / "incumbent_seed_rounds", {"lift_b": (0, 2)})
        rounds(hb / "gate" / "01_hb1" / "candidate", {"lift_b": (1, 2)})
        rounds(hb / "gate" / "02_hb2" / "candidate", {"lift_b": (2, 2)})
        write(run / "alternate.json", dict(families=["fa", "fb"], ref="abc123", seed="seeds/s", stopped={}, events=[
            dict(t=at(11), iter=1, family="fa", step="loop", stop=None),
            dict(t=at(31), iter=1, family="fb", step="loop", stop=None),
            dict(t=at(21), iter=2, family="fa", step="loop", stop=None),
            dict(t=at(51), iter=2, family="fb", step="loop", stop=None),
            dict(t=at(60), iter=3, step="end", why="2 iterations done")]))  # fmt: skip
        lib = run / "knowledge"
        write_jl(lib / "knowledge.jsonl", [
            entry("fa/i00_walk", "The hands sweep a tall object. It falls over. The pre-grasp then misses it by far.",
                  "End the walk 0.20 m behind the stance, then step in.", dict(accepted=False, via="merged")),
            entry("fb/i00_own", "Short lift.", "Lift higher.", None),
            entry("fb/i01_H2", "The turn sweeps a finger.", "Walk in.", dict(accepted=True, via="merged")),
        ])  # fmt: skip
        write_jl(lib / "usage.jsonl", [
            cite("fb/i00_own", "fb", 0, "H1", True, "draft", "helped"),
            cite("fa/i00_walk", "fb", 1, "H2", True, "merged", "helped"),
            cite("fb/i01_H2", "fa", 1, "walk", True, None, "unclear", status="hypothesis"),
        ])  # fmt: skip
        write_jl(lib / "sync.jsonl", [
            dict(iter=1, **{"from": "fa"}, to="fb", merged=["pick.py"], copied=["test_x.py"],
                 skipped="the merged package fails its tests: 1 failed, 2 passed"),
            dict(iter=1, **{"from": "fb"}, to="fa", merged=[], copied=[], skipped="the merge changes nothing"),
            dict(iter=2, **{"from": "fb"}, to="fa", merged=["bimanual.py"], copied=[], accepted=True,
                 reason="accepted (non-inferior)", hash="hsync", result=dict(n=2, incumbent_ok=1, candidate_ok=1, b=0, c=0)),
        ])  # fmt: skip
        (run / "logs").mkdir()
        (run / "logs" / "01_a_fa_sync_tests.log").write_text(
            "F\n_____ test _____\n    x = pick.graspability\nE       AttributeError: module 'pick' has no attribute "
            "'graspability'\nFAILED test_x.py::t\n1 failed, 2 passed\n")  # fmt: skip
        with mock.patch.object(RR, "PROJECTS", Path(self.tmp.name) / "projects"):
            self.data = KR.collect(run)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_citations_across_the_families_count(self):
        self.assertEqual(len(self.data["usage"]), 3)
        self.assertEqual([(u["family"], u["entry"]) for u in self.data["cross"]],
                         [("fb", "fa/i00_walk"), ("fa", "fb/i01_H2")])  # fmt: skip
        self.assertEqual(self.data["by_other"], {"fa/i00_walk": 1, "fb/i01_H2": 1})
        self.assertEqual([(cs["family"], cs["hyp"]) for cs in self.data["cases"]], [("fa", "walk"), ("fb", "H2")])

    def test_the_case_follows_the_hypothesis_to_the_gate(self):
        cs = self.data["cases"][1]
        self.assertEqual(cs["quotes"], ["The one-hand family learned it (fa/i00_walk): step in."])
        self.assertEqual((cs["picks"], cs["drafts"]), (["H3", "H2"], [2]))
        self.assertEqual(cs["gate"], ["candidate hb2 accepted: 1 -> 2 / 2"])
        self.assertEqual(KR.composer_text(cs), "draft 2; the Composer merged drafts 1, 2 (this one included): "
                                               "<i>Draft 1 plus draft 2, at the new stance.</i>")  # fmt: skip
        page = KR.case_html(self.data, cs)
        self.assertIn("its own family's gate: not accepted (merged)", page)
        self.assertIn("End the walk 0.20 m behind the stance, then step in.", page)
        self.assertIn("The walk sweeps a tall object. It topples. It is then lost.", page)  # claim up to CLAIM_N

    def test_a_citation_without_a_draft(self):
        cs = self.data["cases"][0]
        self.assertEqual(KR.composer_text(cs), "no draft of this hypothesis")
        self.assertIn("rejected at stage 1 (1/2 against the incumbent's 1/2)", cs["gate"][0])
        self.assertIn("names the entry only in its knowledge_refs", KR.case_html(self.data, cs))

    def test_library_table(self):
        table = KR.library_table(self.data)
        self.assertIn('<td>1</td><td>2</td><td class="l">not accepted (merged)</td>', table)
        self.assertIn("never reached a gate", table)
        # reads_as: whole sentences up to READS_AS_N characters
        self.assertIn("The hands sweep a tall object. It falls over. The pre-grasp then misses it by far.", table)

    def test_syncs_and_their_test_errors(self):
        self.assertEqual(
            self.data["syncs"][0]["error"], "AttributeError: module 'pick' has no attribute 'graspability'"
        )
        table = KR.sync_table(self.data)
        self.assertIn("1 &rarr; 1 / 2 (won 0, lost 0) &mdash; no start changed", table)
        self.assertIn("the merge changes nothing", table)

    def test_summary(self):
        s = "\n".join(KR.summary(self.data))
        self.assertIn(
            "3 entries (1 from <code>fa</code>, 2 from <code>fb</code>); 3 citations, 2 of them by the other", s
        )
        self.assertIn("<code>fa</code> &rarr; <code>fb</code>:</b> 1 citation(s) by 1 hypothesis(es); 1 built, 1 shipped",
                      s)  # fmt: skip
        self.assertIn("<code>fb</code> &rarr; <code>fa</code>:</b> 1 citation(s) by 1 hypothesis(es); 1 built, 0 shipped",
                      s)  # fmt: skip
        self.assertIn(
            "3 attempts; 1 accepted, 1 skipped because the merged package failed its tests, 1 with nothing", s
        )

    def test_results_table_marks_the_synced_version(self):
        table = KR.results_table(self.data)
        self.assertIn('<span class="tag synced">synced</span> <code>02_hsync</code>', table)
        self.assertIn('<span class="tag accepted">accepted</span> <code>02_hb2</code>', table)

    def test_results_table_marks_both_ways_in_order(self):
        col = dict(label="Iteration 1", name="02_hsa", how="accepted + synced", acc={"t": [2, 2]}, seal={})
        data = dict(alt={}, report=dict(families=[dict(family="fa", cols=[col])]))
        self.assertIn('<span class="tag accepted">accepted</span> + <span class="tag synced">synced</span> '
                      "<code>02_hsa</code>", KR.results_table(data))  # fmt: skip

    def test_main_writes_the_page_next_to_the_run(self):
        with mock.patch.object(RR, "PROJECTS", Path(self.tmp.name) / "projects"), mock.patch("builtins.print"):
            self.assertEqual(KR.main([str(self.run)]), 0)
        page = (self.run / KR.OUT_NAME).read_text()
        self.assertTrue(
            page.startswith('<!doctype html><meta charset="utf-8"><title>RSI knowledge sharing run</title>')
        )
        for title in ("Table 1: both families per iteration", "Table 2: what is shared and where",
                      "Table 3: the library at the end of the run (3 entries)", "Table 4: citations across the families",
                      "The cases: each citing hypothesis", "Table 5: every code sync"):  # fmt: skip
            self.assertIn(title, page)

    def test_a_run_without_alternate_json(self):
        with self.assertRaises(SystemExit):
            KR.collect(self.run / "a")


class Text(unittest.TestCase):
    def test_lead_keeps_whole_sentences(self):
        self.assertEqual(KR.lead("One two. Three four. Five.", 10), "One two. Three four.")
        self.assertEqual(KR.lead("At 0.20 m it works. Next.", 5), "At 0.20 m it works.")
        self.assertEqual(KR.lead("", 5), "")


if __name__ == "__main__":
    unittest.main()
