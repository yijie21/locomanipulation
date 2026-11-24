"""TODO 49.4: the LocoManiLab session (run_lml_q.sh in Python): slot choice and release, the cache pass with its markers, retries when
the bed process dies, the row check, exit codes, the process groups -- with fake bed and GraspGen processes (no simulator)."""

import fcntl
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path

from locomani.bench.locomanilab import session as S
from locomani.bench.locomanilab.config import LocoManiLabConfig

PICK = "IThorBathroomLiftSoapBottle_scene401"
NAV = "IThorKitchenApproachFridge_scene13"
# stands in for `python -m locomani.bench.locomanilab <sub> ...`: argv is `-m locomani.bench.locomanilab <sub> <args>`
FAKE_BED = """#!/bin/bash
sub=$3; shift 3
ids=; out=; task=$1
while [ $# -gt 0 ]; do case $1 in --ids) ids=$2; shift 2;; --out) out=$2; shift 2;; --set) echo "$2" >> "$FAKE_SETS"; shift 2;; *) shift;; esac; done
echo "$sub" >> "$FAKE_LOG"
case $sub in
  cachedir) [ -n "$FAKE_NOCACHE" ] && exit 1; echo "$FAKE_CACHE";;
  look) while read -r id; do n=${id##*/}; k=${id#*/}; k=${k%%/*}
        [ -n "$FAKE_NOTFOUND" ] && [ "$id" = "$FAKE_NOTFOUND" ] && { echo "{\\"episode_id\\": \\"$id\\", \\"found\\": false}" >> "$out"; continue; }
        touch "$FAKE_CACHE/${id%%/*}_L${k}_e$n.npz"; echo "{\\"episode_id\\": \\"$id\\", \\"found\\": true}" >> "$out"; done < "$ids";;
  batch)
    n=$(grep -c batch "$FAKE_LOG")
    if [ -n "$FAKE_FAIL_FIRST" ] && [ "$n" = 1 ]; then head -1 "$ids" | while read -r id; do echo "{\\"episode_id\\": \\"$id\\"}" >> "$out"; done; exit 3; fi
    if [ -n "$FAKE_LEAVE" ]; then (sleep 300 & echo $! > "$FAKE_LEAVE"; wait) & fi
    while read -r id; do grep -q "\\"$id\\"" "$out" 2>/dev/null || { [ -n "$FAKE_SKIP" ] && [ "$id" = "$FAKE_SKIP" ] && continue; echo "{\\"episode_id\\": \\"$id\\"}" >> "$out"; }; done < "$ids";;
esac
"""
FAKE_GRASP = """#!/bin/bash
echo graspgen >> "$FAKE_LOG"
cache=$2; only=$4
for f in "$cache"/${only}_L*_e*.npz; do b=$(basename "$f"); [ -n "$FAKE_NOGRASP" ] && [ "$b" = "$FAKE_NOGRASP" ] && continue; touch "$cache/grasp_$b"; done
"""


def script(path: Path, text: str) -> str:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        t = self.t = Path(self.td.name)
        (t / "cache").mkdir()
        self.cfg = LocoManiLabConfig(
            slots_dir=str(t / "slots"), run_dir=str(t / "run"), cache_dir=str(t / "cache"),
            sim_python=script(t / "bed.sh", FAKE_BED), graspgen_python=script(t / "grasp.sh", FAKE_GRASP),
            graspgen_root=str(t), slots=(1, 2),
        )  # fmt: skip
        self.log, self.sets = t / "fake.log", t / "fake.sets"
        self.log.write_text("")
        self.sets.write_text("")
        os.environ.update(FAKE_LOG=str(self.log), FAKE_CACHE=str(t / "cache"), FAKE_SETS=str(self.sets))
        for k in ("FAKE_NOCACHE", "FAKE_FAIL_FIRST", "FAKE_SKIP", "FAKE_NOTFOUND", "FAKE_NOGRASP", "FAKE_LEAVE"):
            os.environ.pop(k, None)
        self.ids = t / "ids.txt"
        self.ids.write_text(f"{PICK}/Q/0\n{PICK}/Q/1\n")
        self.out = t / "out" / "rows.jsonl"

    def tearDown(self):
        self.td.cleanup()

    def session(self, task=PICK, **kw):
        return S.Session(self.cfg, task, self.ids, self.out, sleep=lambda s: None, **kw)

    def calls(self):
        return self.log.read_text().split()


class Slots(Fixture):
    def test_first_free_slot_is_taken_and_released_on_close(self):
        k, f = S.take_slot(self.cfg)
        self.assertEqual(k, 1)
        k2, f2 = S.take_slot(self.cfg)
        self.assertEqual(k2, 2)
        with self.assertRaises(S.SlotTimeout):
            S.take_slot(self.cfg, timeout_s=0.0, sleep=lambda s: None)
        f.close()
        k3, f3 = S.take_slot(self.cfg)
        self.assertEqual(k3, 1)
        for x in (f2, f3):
            x.close()

    def test_a_lock_held_by_another_process_is_skipped(self):
        Path(self.cfg.slots_dir).mkdir(parents=True)
        other = open(Path(self.cfg.slots_dir) / "slot1.lock", "w")
        fcntl.flock(other, fcntl.LOCK_EX)
        k, f = S.take_slot(self.cfg)
        self.assertEqual(k, 2)
        f.close()
        other.close()


class Helpers(Fixture):
    def test_missing_ids_markers_and_looks_nothing(self):
        c = self.t / "cache"
        ids = [f"{PICK}/Q/{n}" for n in range(4)]
        (c / f"{PICK}_LQ_e0.npz").touch()
        (c / f"grasp_{PICK}_LQ_e0.npz").touch()
        (c / f"{PICK}_LQ_e1.npz").touch()  # a cloud without grasps: missing
        (c / f"{PICK}_LQ_e2.notfound").touch()
        (c / f"{PICK}_LQ_e3.nograsp").touch()
        self.assertEqual(S.missing_ids(ids, PICK, c), [f"{PICK}/Q/1"])
        self.assertEqual(S.missing_ids([f"{NAV}/Q/0"], NAV, c), [])  # a navigation task looks at nothing
        self.assertEqual(S.missing_ids(["IThorKitchenOpenFridge_scene13/Q/0"], "IThorKitchenOpenFridge_scene13", c), [])
        self.assertEqual(
            S.missing_ids(["IThorLivingRoomLowDrawerPnP_scene209/Q/0"], "IThorLivingRoomLowDrawerPnP_scene209", c),
            ["IThorLivingRoomLowDrawerPnP_scene209/Q/0"],
        )
        self.assertEqual(S.cloud_name(PICK, f"{PICK}/P/7"), f"{PICK}_LP_e7")

    def test_row_check(self):
        self.out.parent.mkdir()
        self.out.write_text(json.dumps(dict(episode_id="a")) + "\n")
        self.assertEqual(S.rows_missing(["a", "b"], self.out), ["b"])


class Round(Fixture):
    def test_full_round_exit_0_markers_and_run_dir(self):
        self.assertEqual(self.session().run_round(), 0)
        self.assertEqual(self.calls(), ["cachedir", "look", "graspgen", "batch"])
        rows = [json.loads(line)["episode_id"] for line in open(self.out)]
        self.assertEqual(rows, [f"{PICK}/Q/0", f"{PICK}/Q/1"])
        timing = Path(str(self.out) + ".timing").read_text()
        for k in ("slot_wait", "looks_2_ids", "graspgen", "batch", "total"):
            self.assertIn(k, timing)
        self.assertTrue(
            all(s.startswith("run_dir=") and "call_" in s for s in self.sets.read_text().split())
        )  # the call's own run_dir

    def test_fresh_round_removes_old_rows(self):
        self.out.parent.mkdir()
        self.out.write_text(json.dumps(dict(episode_id=f"{PICK}/Q/0", old=True)) + "\n")
        self.assertEqual(self.session().run_round(), 0)
        self.assertNotIn("old", self.out.read_text())

    def test_navigation_round_makes_no_looks(self):
        self.ids.write_text(f"{NAV}/Q/0\n")
        self.assertEqual(self.session(NAV).run_round(), 0)
        self.assertEqual(self.calls(), ["cachedir", "batch"])

    def test_cache_hit_skips_look_and_graspgen(self):
        for n in (0, 1):
            (self.t / "cache" / f"{PICK}_LQ_e{n}.npz").touch()
            (self.t / "cache" / f"grasp_{PICK}_LQ_e{n}.npz").touch()
        self.assertEqual(self.session().run_round(), 0)
        self.assertEqual(self.calls(), ["cachedir", "batch"])

    def test_not_found_and_no_grasp_leave_markers_and_are_not_looked_at_again(self):
        os.environ["FAKE_NOTFOUND"] = f"{PICK}/Q/0"
        os.environ["FAKE_NOGRASP"] = f"{PICK}_LQ_e1.npz"
        self.assertEqual(self.session().run_round(), 0)
        c = self.t / "cache"
        self.assertTrue((c / f"{PICK}_LQ_e0.notfound").exists())
        self.assertTrue((c / f"{PICK}_LQ_e1.nograsp").exists())
        self.log.write_text("")
        self.assertEqual(self.session().run_round(), 0)
        self.assertEqual(self.calls(), ["cachedir", "batch"])

    def test_a_bed_that_dies_is_started_again_for_the_ids_without_a_row(self):
        os.environ["FAKE_FAIL_FIRST"] = "1"
        self.assertEqual(self.session().run_round(), 0)
        self.assertEqual(self.calls().count("batch"), 2)

    def test_a_missing_row_is_exit_1_after_the_tries(self):
        os.environ["FAKE_SKIP"] = f"{PICK}/Q/1"
        self.assertEqual(self.session().run_round(), 1)
        self.assertEqual(self.calls().count("batch"), self.cfg.batch_tries)

    def test_no_cache_folder_is_exit_3(self):
        os.environ["FAKE_NOCACHE"] = "1"
        self.assertEqual(self.session().run_round(), 3)

    def test_all_slots_busy_times_out_and_the_lock_is_released_after_a_round(self):
        held = [S.take_slot(self.cfg) for _ in range(2)]
        self.assertEqual(self.session(slot_timeout_s=0.0).run_round(), 1)
        for _, f in held:
            f.close()
        self.assertEqual(self.session().run_round(), 0)
        k, f = S.take_slot(self.cfg)  # the round left no lock
        f.close()

    def test_a_child_leaves_no_process_behind(self):
        leave = self.t / "leftover.pid"
        os.environ["FAKE_LEAVE"] = str(leave)
        self.assertEqual(self.session().run_round(), 0)
        time.sleep(0.5)
        pid = int(leave.read_text())
        with self.assertRaises(ProcessLookupError):  # the group was signalled when the child ended
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
