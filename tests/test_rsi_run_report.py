"""report/run_report.py (user 2026-11-25): the per-run HTML page's four tables, from synthetic run records."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_report", REPO / "scripts" / "rsi" / "report" / "run_report.py")
RR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RR)

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
    """{task: (wins, n)} as round files, plus a side file and a record without a task that the tally skips."""
    for task, (w, n) in wins.items():
        for i in range(n):
            write(d / f"{task}_s{i}.json", dict(task=task, ok=i < w, fell=False, dropped=False))
    write(d / "_summary.json", dict(task="pick_a", ok=True))
    write(d / "notes.json", dict(ok=True))


def usage(i: int, cw: int, cr: int, o: int) -> dict:
    return dict(input_tokens=i, cache_creation_input_tokens=cw, cache_read_input_tokens=cr, output_tokens=o)


def role(step: str, it: int, minute: float, seconds: float, session: str, cost: float, **kw) -> dict:
    return dict(t=at(minute), step=step, iter=it, seconds=seconds, session=session, cost_usd=cost,
                usage=usage(1, 2, 3, 4), **kw)  # fmt: skip


def family(sb: Path, name: str, tasks: list[str], objects: dict) -> None:
    """The family file, its tasks file and the scene's catalog: each task's goal names one free object."""
    write(sb / "scripts" / "rsi" / "families" / f"{name}.json",
          dict(scene="scene", tasks_file=f"scripts/rsi/families/{name}_tasks.jsonl", tasks=tasks))  # fmt: skip
    write_jl(sb / "scripts" / "rsi" / "families" / f"{name}_tasks.jsonl",
             [dict(id=t, goal=[f"held({o})", f"lifted({o}, 0.2, 60)"]) for t, o in zip(tasks, objects)])  # fmt: skip
    cat = {o: dict(free=True, gso=dict(name=n)) for o, n in objects.items()}
    cat["platform"] = dict(free=False)
    write(sb / "scene" / "scene_meta.json", dict(objects=cat))


class SingleFamily(unittest.TestCase):
    """Mode 1: one sandbox, two iterations -- the first accepted, the second stopped at stage 1."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sb = sb = Path(self.tmp.name) / "sb"
        family(sb, "fam", ["pick_b", "pick_a"], {"obj_a": "Red_Mug", "obj_b": "Blue_Box"})
        loop = sb / RR.OUT_REL / "fam" / "loop"
        write(loop / "meta.json", dict(started=at(0), model="claude-test"))
        write_jl(loop / "ledger.jsonl", [
            dict(t=at(1), step="diag_pool", iter=0, n=6),
            role("review", 0, 15, 600, "rev0", 1.0),  # Diagnosis 0 -> 5, Reviewer 5 -> 15
            dict(t=at(18), step="probe", iter=0, results=[{}, {}, {}]),
            dict(t=at(19), step="design", iter=0, results=[{}]),
            role("manager", 0, 20, 60, "mgr0", 0.5),
            role("engineer", 0, 30, 500, "eng0a", 2.0, candidate=1),
            role("engineer", 0, 32, 600, "eng0b", 3.0, candidate=2, void=True),
            role("compose", 0, 35, 120, "cmp0", 0.25),
            dict(t=at(50), step="gate", iter=0, hash="aaa111", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=0, candidate_ok=3, n=4), stage1=dict(n=4, candidate_ok=3, incumbent_ok=0)),
            role("review", 1, 60, 300, "rev1", 1.0),
            role("manager", 1, 62, 60, "mgr1", 0.5),
            role("engineer", 1, 70, 400, "eng1", 2.0, candidate=1),
            role("compose", 1, 72, 60, "cmp1", 0.25),
            dict(t=at(80), step="gate", iter=1, hash="bbb222", accepted=False, reason="stage 1: net wins 0 <= 0",
                 result=None, stage1=dict(n=4, candidate_ok=3, incumbent_ok=3)),
            dict(t=at(80), step="stop", reason="iterations 2 >= 2"),
        ])  # fmt: skip
        # the Reviewer's raw file: modelUsage wins over the ledger's own (undercounted) usage
        write(loop / "iter_00" / "review_raw.json",
              dict(session_id="rev0", total_cost_usd=1.5,
                   modelUsage={"m1": dict(inputTokens=10, cacheCreationInputTokens=20, cacheReadInputTokens=30,
                                          outputTokens=40),
                               "m2": dict(inputTokens=1, cacheCreationInputTokens=0, cacheReadInputTokens=0,
                                          outputTokens=2)}))  # fmt: skip
        hid = sb / RR.HIDDEN_REL / "fam"
        write_jl(hid / "gate" / "ledger.jsonl", [
            dict(call=1, hash="aaa111", accepted=True, reason="accepted", dry_run=False, started=at(36),
                 result=dict(incumbent_ok=0, candidate_ok=3, n=4)),
            dict(call=2, hash="bbb222", accepted=False, reason="stage 1: net wins 0 <= 0", dry_run=False,
                 started=at(73)),
        ])  # fmt: skip
        rounds(hid / "gate" / "incumbent_seed_rounds", {"pick_a": (0, 2), "pick_b": (0, 2)})
        rounds(hid / "gate" / "01_aaa111" / "candidate", {"pick_a": (2, 2), "pick_b": (1, 2)})
        rounds(hid / "sealed" / "seed", {"pick_a": (0, 3), "pick_b": (1, 3)})
        rounds(hid / "sealed" / "01_aaa111", {"pick_a": (3, 3), "pick_b": (2, 3)})
        self.data = RR.collect(sb, None)
        self.rep = self.data["families"][0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_columns_seed_accepted_kept(self):
        cols = self.rep["cols"]
        self.assertEqual([(c["label"], c["name"], c["how"]) for c in cols],
                         [("Seed", "", "start"), ("Iteration 1", "01_aaa111", "accepted"),
                          ("Iteration 2", "01_aaa111", "kept")])  # fmt: skip
        self.assertEqual(cols[1]["acc"], {"pick_a": [2, 2], "pick_b": [1, 2]})
        self.assertEqual(cols[0]["seal"], {"pick_a": [0, 3], "pick_b": [1, 3]})
        self.assertEqual(cols[2]["seal"], {"pick_a": [3, 3], "pick_b": [2, 3]})
        self.assertEqual(cols[2]["rate"], "acceptance 3/4 (75.0 %), sealed 5/6 (83.3 %)")

    def test_tasks_objects_and_notes(self):
        self.assertEqual(self.rep["tasks"], ["pick_a", "pick_b"])
        self.assertEqual(self.rep["objects"], {"pick_b": "Red Mug", "pick_a": "Blue Box"})
        self.assertEqual(self.rep["cols"][1]["notes"], ["candidate aaa111 accepted: 0 -> 3 / 4"])
        self.assertIn("rejected at stage 1 (3/4 against the incumbent's 3/4)", self.rep["cols"][2]["notes"][0])
        self.assertEqual(self.rep["stop"], ["iterations 2 >= 2"])

    def test_steps_time_tokens_and_cost(self):
        st = {s["step"]: s for s in self.rep["its"][0]["steps"]}
        self.assertEqual(st["Diagnosis"]["sec"], 300)
        self.assertEqual((st["Reviewer"]["sec"], st["Reviewer"]["calls"], st["Reviewer"]["cost"]), (600, 1, 1.5))
        self.assertEqual(st["Reviewer"]["tok"], dict(input=11, cache_write=20, cache_read=30, output=42))
        self.assertEqual(st["Probe"]["sec"], 240)
        self.assertEqual(st["Probe"]["note"], "3 hypotheses probed, 1 parameter design(s)")
        eng = st["Engineers (×2, parallel)"]
        self.assertEqual((eng["sec"], eng["calls"], eng["cost"], eng["note"]), (720, 2, 5.0, "1 draft(s) void"))
        self.assertEqual(eng["tok"], dict(input=2, cache_write=4, cache_read=6, output=8))
        self.assertEqual((st["Gate"]["sec"], st["Gate"]["note"]), (900, "accepted"))
        # iteration 2 starts where iteration 1 ended; its gate has no own verdict past stage 1
        st2 = {s["step"]: s for s in self.rep["its"][1]["steps"]}
        self.assertEqual(st2["Diagnosis"]["sec"], 300)
        self.assertIsNone(st2["Probe"]["sec"])
        self.assertNotIn("Knowledge build", st2)

    def test_page_tables_and_totals(self):
        page = RR.render(self.data)
        self.assertIn("Table 1.1: acceptance test per object", page)
        self.assertIn("Table 2.1: sealed test per object", page)
        self.assertIn("Table 3: time and tokens per iteration and step", page)
        self.assertIn("<td>3/4</td>", page)  # Table 1's total row of iteration 1
        self.assertIn("Run total", page)
        self.assertIn("final acceptance 3/4 (75.0 %), sealed 5/6 (83.3 %)", page)
        self.assertIn("Stopped</b> after iteration 2", page)

    def test_table_4_iteration_totals(self):
        page = RR.render(self.data)
        self.assertIn("Table 4: time, tokens and cost per iteration and family", page)
        table = RR.family_table(self.data)
        self.assertNotIn("All families", table)  # one family: no combined columns
        # iteration 1: 5 + 10 + 4 + 1 + 12 + 3 + 15 min; the Reviewer's raw file 103 tokens, four more calls 10 each
        self.assertIn('<tr><td class="l">1</td><td>50.0</td><td>143</td><td>7.25</td></tr>', table)
        self.assertIn('<tr><td class="l">2</td><td>30.0</td><td>40</td><td>3.75</td></tr>', table)
        self.assertIn('Run total</td><td>80.0<br><span class="muted">1.33 h</span></td><td>183</td><td>11.00</td>',
                      table)  # fmt: skip

    def test_main_writes_the_page_next_to_the_run(self):
        with mock.patch("builtins.print"):
            self.assertEqual(RR.main([str(self.sb)]), 0)
        page = (self.sb / "rsi_report.html").read_text()
        self.assertTrue(page.startswith('<!doctype html><meta charset="utf-8">'))
        self.assertIn("<title>RSI report sb</title>", page)

    def test_no_sealed_rounds_names_the_command(self):
        import shutil

        shutil.rmtree(self.sb / RR.HIDDEN_REL / "fam" / "sealed")
        page = RR.render(RR.collect(self.sb, None))
        self.assertIn("No sealed rounds yet", page)
        self.assertIn("scripts/rsi/sealed.py fam --all", page)


class Alternating(unittest.TestCase):
    """Mode 3: families fa (sandbox a) and fb (sandbox b), two iterations; in iteration 2 fa's change reaches fb by
    the sync gate before fb's half starts, so only sync.jsonl ties it to iteration 2."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = run = Path(self.tmp.name) / "run"
        self.projects = Path(self.tmp.name) / "projects"
        a, b = run / "a", run / "b"
        family(a, "fa", ["pick_a"], {"obj_a": "Mug"})
        family(b, "fb", ["lift_c"], {"obj_c": "Carton"})
        la, lb = a / RR.OUT_REL / "fa" / "loop", b / RR.OUT_REL / "fb" / "loop"
        write(la / "meta.json", dict(started=at(0), model="claude-test"))
        write(lb / "meta.json", dict(started=at(34), model="claude-test"))
        write_jl(la / "ledger.jsonl", [
            role("review", 0, 10, 300, "ra0", 1.0),
            role("manager", 0, 12, 60, "ma0", 0.5),
            role("engineer", 0, 20, 400, "ea0", 2.0, candidate=1),
            role("compose", 0, 22, 60, "ca0", 0.25),
            dict(t=at(30), step="gate", iter=0, hash="ha1", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=0, candidate_ok=2, n=2)),
            dict(t=at(31), step="pause", reason="--max-iters 1 done"),
            dict(t=at(54), step="resume"),
            role("review", 1, 60, 300, "ra1", 1.0),
            role("manager", 1, 62, 60, "ma1", 0.5),
            role("engineer", 1, 68, 300, "ea1", 2.0, candidate=1),
            role("compose", 1, 69, 60, "ca1", 0.25),
            dict(t=at(70), step="gate", iter=1, hash="ha2", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=2, candidate_ok=2, n=2)),
            dict(t=at(71), step="pause", reason="--max-iters 1 done"),
        ])  # fmt: skip
        write_jl(lb / "ledger.jsonl", [
            role("review", 0, 40, 300, "rb0", 1.0),
            role("manager", 0, 42, 60, "mb0", 0.5),
            role("engineer", 0, 46, 200, "eb0", 2.0, candidate=1),
            role("compose", 0, 47, 60, "cb0", 0.25),
            dict(t=at(50), step="gate", iter=0, hash="hb1", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=0, candidate_ok=1, n=2)),
            dict(t=at(51), step="pause", reason="--max-iters 1 done"),
            dict(t=at(75), step="resume"),
            role("review", 1, 80, 300, "rb1", 1.0),
            role("manager", 1, 82, 60, "mb1", 0.5),
            role("engineer", 1, 86, 200, "eb1", 2.0, candidate=1),
            role("compose", 1, 87, 60, "cb1", 0.25),
            dict(t=at(90), step="gate", iter=1, hash="hb3", accepted=False, reason="net wins -1",
                 result=dict(incumbent_ok=2, candidate_ok=1, n=2)),
            dict(t=at(91), step="stop", reason="the incumbent wins 2/2 >= 95%: sealed set"),
        ])  # fmt: skip
        ha, hb = a / RR.HIDDEN_REL / "fa", b / RR.HIDDEN_REL / "fb"
        write_jl(ha / "gate" / "ledger.jsonl", [
            dict(call=1, hash="ha1", accepted=True, reason="accepted", dry_run=False, started=at(23)),
            dict(call=2, hash="ha2", accepted=True, reason="accepted", dry_run=False, started=at(69)),
        ])  # fmt: skip
        write_jl(hb / "gate" / "ledger.jsonl", [
            dict(call=1, hash="hb1", accepted=True, reason="accepted", dry_run=False, started=at(48)),
            dict(call=2, hash="hsync", accepted=True, reason="accepted (non-inferior)", dry_run=False, started=at(73)),
            dict(call=3, hash="hb3", accepted=False, reason="net wins -1", dry_run=False, started=at(88)),
        ])  # fmt: skip
        rounds(ha / "gate" / "incumbent_seed_rounds", {"pick_a": (0, 2)})
        rounds(ha / "gate" / "01_ha1" / "candidate", {"pick_a": (1, 2)})
        rounds(ha / "gate" / "02_ha2" / "candidate", {"pick_a": (2, 2)})
        rounds(hb / "gate" / "incumbent_seed_rounds", {"lift_c": (0, 2)})
        rounds(hb / "gate" / "01_hb1" / "candidate", {"lift_c": (1, 2)})
        rounds(hb / "gate" / "02_hsync" / "candidate", {"lift_c": (2, 2)})
        write(run / "alternate.json", dict(families=["fa", "fb"], ref="abc123", seed="scripts/rsi/seeds/s", events=[
            dict(t=at(31), iter=1, family="fa", step="loop", stop=None),
            dict(t=at(33), iter=1, family="fa", step="build", entries=2, cost_usd=0.5),
            dict(t=at(34), iter=1, family="fa", step="sync", skipped="the merge changes nothing"),
            dict(t=at(51), iter=1, family="fb", step="loop", stop=None),
            dict(t=at(52), iter=1, family="fb", step="build", entries=1, cost_usd=0.25),
            dict(t=at(53), iter=1, family="fb", step="sync", skipped="the merge changes nothing"),
            dict(t=at(71), iter=2, family="fa", step="loop", stop=None),
            dict(t=at(72), iter=2, family="fa", step="build", entries=3, cost_usd=0.5),
            dict(t=at(74), iter=2, family="fa", step="sync", accepted=True),
            dict(t=at(91), iter=2, family="fb", step="loop", stop="sealed set"),
            dict(t=at(92), iter=3, step="end", why="the families stopped"),
        ]))  # fmt: skip
        write_jl(run / "knowledge" / "sync.jsonl", [
            dict(iter=1, **{"from": "fa"}, to="fb", skipped="the merge changes nothing"),
            dict(iter=1, **{"from": "fb"}, to="fa", skipped="the merge changes nothing"),
            dict(iter=2, **{"from": "fa"}, to="fb", accepted=True, hash="hsync",
                 result=dict(incumbent_ok=1, candidate_ok=2, n=2)),
        ])  # fmt: skip
        # fa's first knowledge build: one session, one API message logged twice (counted once)
        slug = "".join(ch if ch.isalnum() else "-" for ch in str(run.resolve()))
        msg = dict(type="assistant", timestamp=at(31.5), message=dict(id="m1", usage=usage(5, 6, 7, 8)))
        write_jl(self.projects / f"{slug}-ctx-fa" / "s.jsonl", [dict(type="user", timestamp=at(31.2)), msg, msg])
        with mock.patch.object(RR, "PROJECTS", self.projects):
            self.data = RR.collect(run, None)
        self.fa, self.fb = self.data["families"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_sandboxes_and_labels(self):
        self.assertEqual(self.data["mode"], "alternating (mode 3)")
        self.assertEqual([r["sandbox"].name for r in self.data["families"]], ["a", "b"])
        self.assertEqual([it["label"] for it in self.fa["its"]], [1, 2])
        self.assertEqual([it["label"] for it in self.fb["its"]], [1, 2])

    def test_sync_into_the_second_family_belongs_to_its_iteration(self):
        self.assertEqual([(c["label"], c["name"], c["how"]) for c in self.fb["cols"]],
                         [("Seed", "", "start"), ("Iteration 1", "01_hb1", "accepted"),
                          ("Iteration 2", "02_hsync", "synced")])  # fmt: skip
        self.assertIn("sync from fa: accepted as non-inferior, 1 -> 2 / 2", self.fb["cols"][2]["notes"])
        self.assertIn("sync from fb: the merge changes nothing", self.fa["cols"][1]["notes"])
        self.assertEqual(self.fb["cols"][2]["acc"], {"lift_c": [2, 2]})

    def test_half_starts_at_resume_and_builds_count(self):
        st = {s["step"]: s for s in self.fb["its"][1]["steps"]}
        self.assertEqual(st["Diagnosis"]["sec"], 0)  # resume 75, review started 80 - 5
        st = {s["step"]: s for s in self.fa["its"][0]["steps"]}
        kb = st["Knowledge build"]
        self.assertEqual((kb["sec"], kb["calls"], kb["cost"], kb["note"]), (120, 1, 0.5, "2 entries"))
        self.assertEqual(kb["tok"], dict(input=5, cache_write=6, cache_read=7, output=8))
        self.assertEqual((st["Sync gate"]["sec"], st["Sync gate"]["note"]),
                         (60, "to the other family: the merge changes nothing"))  # fmt: skip
        # fb's second half stopped: no build and no sync row
        self.assertNotIn("Knowledge build", {s["step"] for s in self.fb["its"][1]["steps"]})

    def test_run_total_row(self):
        page = RR.render(self.data)
        self.assertIn("Run total", page)
        self.assertIn("fa: final acceptance 2/2 (100.0 %), sealed not run", page)
        self.assertIn("Iteration 2 total", page)
        self.assertIn("the families stopped", page)

    def test_table_4_families_side_by_side(self):
        table = RR.family_table(self.data)
        self.assertIn('<th colspan="3"><code>fa</code></th><th colspan="3"><code>fb</code></th>'
                      '<th colspan="3">All families</th>', table)  # fmt: skip
        # iteration 1: fa 33 min with its build and sync, 40 + 26 build tokens; fb 18 min; then the two summed
        self.assertIn('<tr><td class="l">1</td><td>33.0</td><td>66</td><td>4.25</td><td>18.0</td><td>40</td>'
                      "<td>4.00</td><td>51.0</td><td>106</td><td>8.25</td></tr>", table)  # fmt: skip
        self.assertIn('Run total</td><td>52.0<br><span class="muted">0.87 h</span></td><td>106</td><td>8.50</td>'
                      '<td>33.0<br><span class="muted">0.55 h</span></td><td>80</td><td>7.75</td>'
                      '<td>85.0<br><span class="muted">1.42 h</span></td><td>186</td><td>16.25</td></tr>', table)  # fmt: skip
        # the same run total as Table 3's last row
        t3 = RR.steps_table(self.data).split('<tr class="total">')[-1]
        self.assertIn('<td>85.0<br><span class="muted">1.42 h</span></td>', t3)
        self.assertIn("<td>186</td><td>16.25</td>", t3)
        self.assertIn("run one after the other", RR.render(self.data))

    def test_table_4_a_family_without_the_iteration(self):
        short = dict(self.data, families=[self.fa, dict(self.fb, its=self.fb["its"][:1])])
        dash = '<td class="muted">&mdash;</td>' * 3
        self.assertIn(f'<tr><td class="l">2</td><td>19.0</td><td>40</td><td>4.25</td>{dash}<td>19.0</td><td>40</td>'
                      "<td>4.25</td></tr>", RR.family_table(short))  # fmt: skip


class BothInOneIteration(unittest.TestCase):
    """Mode 3, one iteration in which each family accepts its own candidate and a sync. fa's half comes first: its gate
    accepts ha1, then fb's sync into fa (hsa, after fb's half). fb first takes fa's sync (hsb, before its half), then
    its gate accepts hb1. Each column shows the later version and both tags in the order they happened."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        run = Path(self.tmp.name) / "run"
        a, b = run / "a", run / "b"
        family(a, "fa", ["pick_a"], {"obj_a": "Mug"})
        family(b, "fb", ["lift_c"], {"obj_c": "Carton"})
        write_jl(a / RR.OUT_REL / "fa" / "loop" / "ledger.jsonl", [
            dict(t=at(10), step="gate", iter=0, hash="ha1", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=0, candidate_ok=1, n=2))])  # fmt: skip
        write_jl(b / RR.OUT_REL / "fb" / "loop" / "ledger.jsonl", [
            dict(t=at(30), step="gate", iter=0, hash="hb1", accepted=True, reason="accepted",
                 result=dict(incumbent_ok=1, candidate_ok=2, n=2))])  # fmt: skip
        ha, hb = a / RR.HIDDEN_REL / "fa", b / RR.HIDDEN_REL / "fb"
        write_jl(ha / "gate" / "ledger.jsonl", [
            dict(call=1, hash="ha1", accepted=True, reason="accepted", started=at(5)),
            dict(call=2, hash="hsa", accepted=True, reason="accepted (non-inferior)", started=at(35))])  # fmt: skip
        write_jl(hb / "gate" / "ledger.jsonl", [
            dict(call=1, hash="hsb", accepted=True, reason="accepted (non-inferior)", started=at(12)),
            dict(call=2, hash="hb1", accepted=True, reason="accepted", started=at(25))])  # fmt: skip
        rounds(ha / "gate" / "incumbent_seed_rounds", {"pick_a": (0, 2)})
        rounds(ha / "gate" / "01_ha1" / "candidate", {"pick_a": (1, 2)})
        rounds(ha / "gate" / "02_hsa" / "candidate", {"pick_a": (2, 2)})
        rounds(hb / "gate" / "incumbent_seed_rounds", {"lift_c": (0, 2)})
        rounds(hb / "gate" / "01_hsb" / "candidate", {"lift_c": (1, 2)})
        rounds(hb / "gate" / "02_hb1" / "candidate", {"lift_c": (2, 2)})
        write(run / "alternate.json", dict(families=["fa", "fb"], ref="abc123", events=[
            dict(t=at(11), iter=1, family="fa", step="loop", stop=None),
            dict(t=at(12), iter=1, family="fa", step="build", entries=1, cost_usd=0.1),
            dict(t=at(14), iter=1, family="fa", step="sync", accepted=True),
            dict(t=at(31), iter=1, family="fb", step="loop", stop=None),
            dict(t=at(32), iter=1, family="fb", step="build", entries=1, cost_usd=0.1),
            dict(t=at(36), iter=1, family="fb", step="sync", accepted=True),
            dict(t=at(37), iter=2, step="end", why="1 iterations done")]))  # fmt: skip
        write_jl(run / "knowledge" / "sync.jsonl", [
            dict(iter=1, **{"from": "fa"}, to="fb", accepted=True, hash="hsb",
                 result=dict(incumbent_ok=0, candidate_ok=1, n=2)),
            dict(iter=1, **{"from": "fb"}, to="fa", accepted=True, hash="hsa",
                 result=dict(incumbent_ok=1, candidate_ok=2, n=2))])  # fmt: skip
        with mock.patch.object(RR, "PROJECTS", Path(self.tmp.name) / "projects"):
            self.fa, self.fb = RR.collect(run, None)["families"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_both_tags_in_order_and_the_later_version(self):
        self.assertEqual([(c["label"], c["name"], c["how"]) for c in self.fa["cols"]],
                         [("Seed", "", "start"), ("Iteration 1", "02_hsa", "accepted + synced")])  # fmt: skip
        self.assertEqual([(c["label"], c["name"], c["how"]) for c in self.fb["cols"]],
                         [("Seed", "", "start"), ("Iteration 1", "02_hb1", "synced + accepted")])  # fmt: skip
        self.assertEqual(self.fa["cols"][1]["acc"], {"pick_a": [2, 2]})  # the later version's rounds
        self.assertEqual(self.fb["cols"][1]["acc"], {"lift_c": [2, 2]})

    def test_the_header_shows_both_tags(self):
        self.assertIn('<span class="tag accepted">accepted</span> + <span class="tag synced">synced</span> '
                      "<code>02_hsa</code>", RR.pool_table(self.fa, "acc"))  # fmt: skip
        self.assertIn('<span class="tag synced">synced</span> + <span class="tag accepted">accepted</span> '
                      "<code>02_hb1</code>", RR.pool_table(self.fb, "acc"))  # fmt: skip
        self.assertNotIn('class="kept"', RR.pool_table(self.fa, "acc"))


if __name__ == "__main__":
    unittest.main()
