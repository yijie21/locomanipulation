"""TODO 48.17 (2026-11-23): the check half of the diagnostic pool, its stop rule, the gate's stage-1 file and the
diagnostic count the roles are told."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"

import rsi_loop_ns as NS  # noqa: E402


def _scene_loop():
    return NS.fresh_loop()


class CheckHalf(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        root = Path(self.d.name)
        self.loop, self.hidden = root / "loop_fam", root / "hidden_fam"
        for p in (mock.patch.multiple(self.lp, FAMILY="fam", LOOP=self.loop, HIDDEN=self.hidden,
                                      STATE=self.loop / "loop", CAPS={"check_half": True, "stop_no_progress": 2}),):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)
        pool = self.loop / "pool_diag"
        pool.mkdir(parents=True)
        self.labels = [f"TaskA_s{i}" for i in range(5)] + [f"TaskB_s{i}" for i in range(4)]
        for x in self.labels:
            (pool / f"{x}.json").write_text(json.dumps(dict(task=x.rsplit("_s", 1)[0], episode_id=x)))
        (pool / "_pool.json").write_text("{}")
        inc = self.loop / "diag" / "incumbent_abc"
        inc.mkdir(parents=True)
        for x in self.labels:
            (inc / f"{x}.json").write_text("{}")
            (inc / f"_old_{x}.json").write_text("{}")
            (inc / f"{x}.npz").write_text("")
        (inc / "_run_diag.json").write_text("{}")

    def test_half_of_each_task_by_hash_and_the_same_every_time(self):
        chk = self.lp.check_labels_of(self.labels)
        self.assertEqual(sum(x.startswith("TaskA") for x in chk), 2)  # 5 starts: 2 kept from the roles, 3 seen
        self.assertEqual(sum(x.startswith("TaskB") for x in chk), 2)
        self.assertEqual(chk, self.lp.check_labels_of(list(reversed(self.labels))))
        with mock.patch.object(self.lp, "FAMILY", "other"):  # another family draws another half
            self.assertEqual(len(self.lp.check_labels_of(self.labels)), 4)

    def test_split_moves_the_starts_and_every_file_of_them_once(self):
        row = self.lp.split_check_half()
        chk = self.lp.check_labels_of(self.labels)
        self.assertEqual((row["n_check"], row["n_build"], row["files_moved"]), (4, 5, 12))
        left = sorted(p.stem for p in (self.loop / "pool_diag").glob("*.json") if not p.name.startswith("_"))
        self.assertEqual(left, sorted(set(self.labels) - set(chk)))
        self.assertTrue((self.loop / "pool_diag" / "_pool.json").exists())
        moved = self.hidden / "check" / "incumbent_abc"
        self.assertEqual(sorted(p.name for p in moved.iterdir() if not p.name.startswith("_") and p.suffix == ".json"),
                         sorted(f"{x}.json" for x in chk))  # fmt: skip
        self.assertTrue(all((moved / f"_old_{x}.json").exists() and (moved / f"{x}.npz").exists() for x in chk))
        self.assertTrue((self.loop / "diag" / "incumbent_abc" / "_run_diag.json").exists())
        self.assertEqual(json.loads((self.hidden / "pool_check" / "_split.json").read_text())["labels"], chk)
        self.assertIsNone(self.lp.split_check_half())  # once per loop
        self.assertEqual(self.lp.entries()[-1]["step"], "check_split")

    def test_no_split_without_the_cap(self):
        with mock.patch.object(self.lp, "CAPS", {}):
            self.assertIsNone(self.lp.split_check_half())
        self.assertFalse((self.hidden / "pool_check").exists())

    def test_two_iterations_in_a_row_without_a_candidate_that_passes_the_check_stop_the_run(self):
        """TODO 54: iterations, not candidates: one whose Composer kept the incumbent has no check row and counts."""
        meta = dict(started="2026-11-23T00:00:00Z")
        review = lambda n, t: dict(step="review", iter=n, t=t)  # noqa: E731
        check = lambda n, sent, t: dict(step="check", iter=n, sent=sent, net=1, t=t)  # noqa: E731
        es = [
            review(0, "2026-11-23T01:00:00Z"),
            check(0, False, "2026-11-23T01:30:00Z"),
            review(1, "2026-11-23T02:00:00Z"),
        ]
        with mock.patch.object(self.lp, "CAPS", {"check_half": True, "stop_no_progress": 2}):
            self.assertIn("no progress: iterations 0, 1", self.lp.no_progress_stop(es, meta))
            es2 = es + [check(1, True, "2026-11-23T02:30:00Z")]
            self.assertIsNone(self.lp.no_progress_stop(es2, meta))  # iteration 1 passed
            self.assertIsNone(self.lp.no_progress_stop(es[:1], meta))  # one iteration is not two
            self.assertIsNone(
                self.lp.no_progress_stop(es, dict(started="2026-11-23T03:30:00Z"))
            )  # this run's rows only
            kept = es + [review(2, "2026-11-23T03:00:00Z")]  # a kept incumbent: no check row
            self.assertIn("iterations 1, 2", self.lp.no_progress_stop(kept + [], meta))
        with mock.patch.object(self.lp, "CAPS", {"check_half": True}):
            self.assertIsNone(self.lp.no_progress_stop(es, meta))  # no cap, no rule
        with mock.patch.object(self.lp, "CAPS", {"stop_no_progress": 2}):  # no check half: gate stage 1 counts
            gate = lambda n, reason: dict(
                step="gate", iter=n, reason=reason, stage1=dict(b=1, c=0), t="2026-11-23T05:00:00Z"
            )  # noqa: E731
            e3 = [
                review(0, "2026-11-23T01:00:00Z"),
                review(1, "2026-11-23T02:00:00Z"),
                gate(1, "stage 1: net wins -1 < 0"),
            ]
            self.assertIn("gate's stage 1", self.lp.no_progress_stop(e3, meta))
            e3[2] = gate(1, "net wins 2 < 3")  # it passed stage 1 and failed later: progress
            self.assertIsNone(self.lp.no_progress_stop(e3, meta))

    def test_the_roles_are_told_the_starts_they_see(self):
        """TODO 54: counted from the pool directory (the check split has halved it), in all and per task."""
        v = self.lp.prompt_vars()
        self.assertEqual((v["N_DIAG"], v["N_DIAG_TASK"]), (9, "4-5"))
        with mock.patch.object(self.lp, "check_pool", lambda: self.hidden / "pool_check"):
            (self.hidden / "pool_check").mkdir(parents=True)
            (self.hidden / "pool_check" / "TaskA_s9.json").write_text("{}")
            self.assertEqual(self.lp.prompt_vars()["N_DIAG_FULL"], 10)


def _round(label, ok, blame=None, steps=("walk", "reach", "lift")):
    """A round with `steps` of 2 s each; `blame`: the step its failure is blamed on."""
    return dict(label=label, task=label.rsplit("_s", 1)[0], ok=ok,
                steps=[dict(step=s, ok=s != blame, t0=2.0 * i, t1=2.0 * i + 2) for i, s in enumerate(steps)])  # fmt: skip


class Segments(unittest.TestCase):
    """TODO 48.17.6 (rules 48): hot and stable steps from the rounds, the cards' line and where a draft acted."""

    def setUp(self):
        sys.path.insert(0, str(HERE))
        self.addCleanup(sys.path.remove, str(HERE))
        import segments

        self.sg = segments
        self.rounds = ([_round(f"T_s{i}", True) for i in range(6)] + [_round(f"T_s{i}", False, "lift") for i in range(6, 9)]
                       + [_round("T_s9", False, "reach")])  # fmt: skip

    def test_the_step_with_most_losses_is_hot_and_a_step_none_failed_at_is_stable(self):
        rows = {r["step"]: r for r in self.sg.segment_table(self.rounds)["T"]}
        self.assertEqual([r["step"] for r in self.sg.segment_table(self.rounds)["T"]], ["walk", "reach", "lift"])
        self.assertEqual((rows["lift"]["kind"], rows["lift"]["failed"], rows["lift"]["loss_share"]), ("hot", 3, 0.75))
        self.assertEqual((rows["walk"]["kind"], rows["walk"]["reached"]), ("stable", 10))
        self.assertEqual(rows["reach"]["kind"], "")  # one loss: neither
        self.assertEqual(rows["lift"]["t_reach"], 4.0)

    def test_the_cards_line_and_the_time_before_the_blamed_step(self):
        line = self.sg.table_lines(self.sg.segment_table(self.rounds), self.rounds)[0]
        self.assertIn("T (4 lost): walk 0/10 stable > reach 1/10 > lift 3/10 HOT", line)
        self.assertIn("58 % of their time before the blamed step", line)  # (3*4 + 2) / (4*6) = 14 / 24
        self.assertEqual(self.sg.table_lines(self.sg.segment_table(self.rounds[:6]), self.rounds[:6]), [])

    def test_a_drafts_changes_are_counted_by_the_kind_of_step_they_start_in(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            ref, cand = Path(d) / "ref", Path(d) / "cand"
            ref.mkdir()
            cand.mkdir()
            for r in self.rounds:
                (ref / f"{r['label']}.json").write_text(json.dumps(r))
                (cand / f"{r['label']}.json").write_text(json.dumps(r))
            moved = _round("T_s0", True)
            moved["steps"][0]["t1"] = 2.5  # the walk took longer: the change starts in a stable step
            (cand / "T_s0.json").write_text(json.dumps(moved))
            (cand / "T_s6.json").write_text(json.dumps(_round("T_s6", True)))  # fixed at the hot step
            table = self.sg.segment_table(self.rounds)
            t = lp.segments_touched(ref, cand, ["T_s0", "T_s6"], table)
            self.assertEqual(t, {"stable": {"T:walk": 1}, "hot": {"T:lift": 1}, "other": {}})


class Stage1File(unittest.TestCase):
    def test_only_this_calls_file_counts(self):
        lp = _scene_loop()
        with tempfile.TemporaryDirectory() as d:
            g = Path(d)
            (g / "03_abc").mkdir()
            f = g / "03_abc" / "stage1_passed.json"
            f.write_text("{}")
            old = time.time() - 3600
            os.utime(f, (old, old))
            t0 = time.time()
            self.assertFalse(lp.stage1_passed(g, "abc", t0))  # an earlier call's file
            (g / "04_abc").mkdir()
            (g / "04_abc" / "stage1_passed.json").write_text("{}")
            self.assertTrue(lp.stage1_passed(g, "abc", t0))
            self.assertFalse(lp.stage1_passed(g, "other", t0))


if __name__ == "__main__":
    unittest.main()
