"""TODO 49.2: the session (run_official_q.sh in Python): slot choice and lock release, readiness, the cache pass, retries on a
lost server, the row check and the exit codes -- with fake server, depth, bed and GraspGen processes (no simulator, no GPU)."""

import fcntl
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from locomani.bench.simple import session as S
from locomani.bench.simple.config import SimpleConfig

FAKE_SERVER = """#!/bin/bash
echo listening
exec sleep 120
"""
FAKE_DEPTH = """#!/bin/bash
touch "${@: -1}"
exec sleep 120
"""
# stands in for `python -m locomani.bench.simple <sub> ...`: argv is `-m locomani.bench.simple <sub> <args>`
FAKE_BED = """#!/bin/bash
sub=$3; shift 3
ids=; out=; task=$1
while [ $# -gt 0 ]; do case $1 in --ids) ids=$2; shift 2;; --out) out=$2; shift 2;; *) shift;; esac; done
echo "$sub" >> "$FAKE_LOG"
case $sub in
  cachedir) [ -n "$FAKE_NOCACHE" ] && exit 1; echo "$FAKE_CACHE";;
  look) while read -r id; do n=${id##*/}; k=${id#*/}; k=${k%%/*}; touch "$FAKE_CACHE/${id%%/*}_L${k}_e$n.npz"; done < "$ids";;
  batch)
    n=$(grep -c batch "$FAKE_LOG")
    if [ -n "$FAKE_FAIL_FIRST" ] && [ "$n" = 1 ]; then head -1 "$ids" | while read -r id; do echo "{\\"episode_id\\": \\"$id\\"}" >> "$out"; done; exit 3; fi
    while read -r id; do grep -q "\\"$id\\"" "$out" 2>/dev/null || { [ -n "$FAKE_SKIP" ] && [ "$id" = "$FAKE_SKIP" ] && continue; echo "{\\"episode_id\\": \\"$id\\"}" >> "$out"; }; done < "$ids";;
esac
"""
FAKE_GRASP = """#!/bin/bash
echo graspgen >> "$FAKE_LOG"
cache=$2; only=$4
for f in "$cache"/${only}_L*_e*.npz; do b=$(basename "$f"); touch "$cache/grasp_$b"; done
"""


def script(path: Path, text: str) -> str:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        t = Path(self.td.name)
        self.t = t
        (t / "cache").mkdir()
        self.cfg = SimpleConfig(
            slots_dir=str(t / "slots"), run_dir=str(t / "run"), cache_dir=str(t / "cache"),
            simple_python=script(t / "server.sh", FAKE_SERVER), sam3d_python=script(t / "depth.sh", FAKE_DEPTH),
            sim_python=script(t / "bed.sh", FAKE_BED), graspgen_python=script(t / "grasp.sh", FAKE_GRASP),
            graspgen_root=str(t), server_ready_s=10, depth_ready_s=10, slots=(1, 2),
        )  # fmt: skip
        self.log = t / "fake.log"
        self.log.write_text("")
        os.environ["FAKE_LOG"], os.environ["FAKE_CACHE"] = str(self.log), str(t / "cache")
        for k in ("FAKE_NOCACHE", "FAKE_FAIL_FIRST", "FAKE_SKIP"):
            os.environ.pop(k, None)
        self.ids = t / "ids.txt"
        self.ids.write_text("XMovePickTeleop/Q/0\nXMovePickTeleop/Q/1\n")
        self.out = t / "rows.jsonl"

    def tearDown(self):
        for k in ("FAKE_LOG", "FAKE_CACHE", "FAKE_NOCACHE", "FAKE_FAIL_FIRST", "FAKE_SKIP"):
            os.environ.pop(k, None)
        self.td.cleanup()

    def session(self, **kw):
        return S.Session(
            self.cfg, "XMovePickTeleop", self.ids, self.out, sleep=lambda s: time.sleep(min(s, 0.05)), **kw
        )

    def calls(self):
        return self.log.read_text().split()


class Slots(Fixture):
    def test_first_free_slot_is_taken_and_released_on_close(self):
        k1, f1 = S.take_slot(self.cfg)
        k2, f2 = S.take_slot(self.cfg)
        self.assertEqual((k1, k2), (1, 2))
        with self.assertRaises(S.SlotTimeout):
            S.take_slot(self.cfg, timeout_s=0.1, sleep=lambda s: time.sleep(0.02))
        f1.close()  # the lock goes with the file
        k3, f3 = S.take_slot(self.cfg)
        self.assertEqual(k3, 1)
        f2.close()
        f3.close()

    def test_a_lock_held_by_another_process_is_skipped(self):
        Path(self.cfg.slots_dir).mkdir(parents=True)
        other = open(
            Path(self.cfg.slots_dir) / "slot1.lock", "w"
        )  # (the esc bed's run_official_q.sh uses the same files)
        fcntl.flock(other, fcntl.LOCK_EX)
        k, f = S.take_slot(self.cfg)
        self.assertEqual(k, 2)
        f.close()
        other.close()


class Helpers(Fixture):
    def test_missing_ids_and_row_check(self):
        c = Path(self.cfg.cache_dir)
        (c / "XMovePickTeleop_LQ_e0.npz").touch()
        (c / "grasp_XMovePickTeleop_LQ_e0.npz").touch()
        (c / "XMovePickTeleop_LQ_e1.npz").touch()  # the look without its grasps is still missing
        ids = ["XMovePickTeleop/Q/0", "XMovePickTeleop/Q/1", "XMovePickTeleop/Q/2"]
        self.assertEqual(S.missing_ids(ids, "XMovePickTeleop", c), ids[1:])
        self.assertEqual(
            S.missing_ids(["CloseDoorTeleop/Q/0"], "CloseDoorTeleop", c), []
        )  # the articulated tasks keep no cache
        self.assertEqual(S.rows_missing(ids, self.out), ids)
        self.out.write_text(json.dumps(dict(episode_id=ids[0])) + "\n")
        self.assertEqual(S.rows_missing(ids, self.out), ids[1:])


class Round(Fixture):
    def test_full_round_exit_0_and_no_orphans(self):
        s = self.session()
        self.assertEqual(s.run_round(), S.EXIT_OK)
        self.assertEqual(self.calls(), ["cachedir", "look", "graspgen", "batch"])
        self.assertEqual(len(self.out.read_text().splitlines()), 2)
        self.assertIsNone(s.server)
        for pid_file in ("server_XMovePickTeleop.pid", "depth_XMovePickTeleop.pid"):
            pid = int((Path(self.cfg.run_dir) / pid_file).read_text())
            time.sleep(0.2)
            self.assertFalse(
                Path(f"/proc/{pid}").exists() and "sleep" in Path(f"/proc/{pid}/cmdline").read_text(), pid_file
            )
        timing = (Path(self.cfg.run_dir) / "q_XMovePickTeleop.timing").read_text()
        for name in ("slot_wait", "server_start", "depth_start", "looks_2_ids", "graspgen", "batch", "total"):
            self.assertIn(name, timing)
        k, f = S.take_slot(self.cfg)  # the slot lock was released
        self.assertEqual(k, 1)
        f.close()

    def test_cache_hit_skips_look_and_graspgen(self):
        c = Path(self.cfg.cache_dir)
        for n in (0, 1):
            (c / f"XMovePickTeleop_LQ_e{n}.npz").touch()
            (c / f"grasp_XMovePickTeleop_LQ_e{n}.npz").touch()
        self.assertEqual(self.session().run_round(), S.EXIT_OK)
        self.assertEqual(self.calls(), ["cachedir", "batch"])

    def test_lost_server_restarts_and_runs_only_ids_without_a_row(self):
        os.environ["FAKE_FAIL_FIRST"] = "1"
        s = self.session()
        self.assertEqual(s.run_round(), S.EXIT_OK)
        self.assertEqual(self.calls().count("batch"), 2)
        ids = [json.loads(x)["episode_id"] for x in self.out.read_text().splitlines()]
        self.assertEqual(sorted(ids), ["XMovePickTeleop/Q/0", "XMovePickTeleop/Q/1"])  # no id twice

    def test_a_missing_row_is_exit_1_after_the_tries(self):
        os.environ["FAKE_SKIP"] = "XMovePickTeleop/Q/1"
        self.assertEqual(self.session().run_round(), S.EXIT_ROWS)
        self.assertEqual(self.calls().count("batch"), 1)  # a clean exit of the bed is not retried

    def test_no_cache_folder_is_exit_3(self):
        os.environ["FAKE_NOCACHE"] = "1"
        self.assertEqual(self.session().run_round(), S.EXIT_NO_CACHE)
        k, f = S.take_slot(self.cfg)
        self.assertEqual(k, 1)  # the slot was released on the way out
        f.close()

    def test_all_slots_busy_times_out(self):
        held = [S.take_slot(self.cfg)[1] for _ in (1, 2)]
        self.assertEqual(self.session(slot_timeout_s=0.1).run_round(), S.EXIT_ROWS)
        for f in held:
            f.close()


class Readiness(Fixture):
    def test_server_that_never_listens_times_out_and_is_stopped(self):
        self.cfg = SimpleConfig(
            **{
                **self.cfg.__dict__,
                "simple_python": script(self.t / "mute.sh", "#!/bin/bash\nexec sleep 120\n"),
                "server_ready_s": 1,
            }
        )
        s = self.session()
        s.slot = 1
        Path(self.cfg.slots_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.assertFalse(s.start_server())
        self.assertLess(time.time() - t0, 5)
        pid = s.server.pid
        s.stop_all()
        self.assertIsNone(s.server)
        time.sleep(0.2)
        self.assertFalse(Path(f"/proc/{pid}").exists())

    def test_a_server_that_dies_ends_the_wait_early(self):
        self.cfg = SimpleConfig(
            **{
                **self.cfg.__dict__,
                "simple_python": script(self.t / "dies.sh", "#!/bin/bash\nexit 1\n"),
                "server_ready_s": 30,
            }
        )
        s = self.session()
        s.slot = 1
        Path(self.cfg.slots_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.assertFalse(s.start_server())
        self.assertLess(time.time() - t0, 5)

    def test_depth_ready_is_its_socket(self):
        s = self.session()
        s.slot = 1
        Path(self.cfg.slots_dir).mkdir(parents=True, exist_ok=True)
        self.assertTrue(s.start_depth())
        self.assertTrue(s.sock("depth").exists())
        s.stop_all()


if __name__ == "__main__":
    unittest.main()


def alive(pid: int) -> bool:
    try:
        return "sleep" in Path(f"/proc/{pid}/cmdline").read_text()
    except OSError:
        return False


class NoLeak(Fixture):
    """TODO 52: the services (own sessions) must not outlive a killed session (four stereo servers leaked 13 GB on 2026-11-24)."""

    def test_sigterm_to_the_session_stops_both_services(self):
        slow = self.t / "slow_bed.sh"
        script(slow, FAKE_BED.replace("  batch)", "  batch) sleep 30;;\n  batchx)"))
        self.cfg = SimpleConfig(**{**self.cfg.__dict__, "sim_python": str(slow)})
        s = self.session()
        threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        with self.assertRaises(SystemExit):
            s.run_round()
        time.sleep(0.3)
        for f in ("server_XMovePickTeleop.pid", "depth_XMovePickTeleop.pid"):
            self.assertFalse(alive(int((Path(self.cfg.run_dir) / f).read_text())), f)
        self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)  # handlers restored

    def test_sigkill_of_the_session_takes_a_service_with_it(self):
        code = (
            "import subprocess, sys, os, time\n"
            "from locomani.bench.simple.session import die_with_parent\n"
            "p = subprocess.Popen(['sleep', '100'], start_new_session=True, preexec_fn=die_with_parent)\n"
            "print(p.pid, flush=True); time.sleep(100)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONPATH=str(S.SRC))
        )
        pid = int(parent.stdout.readline())
        self.assertTrue(alive(pid))
        parent.kill()
        parent.wait()
        for _ in range(40):
            if not alive(pid):
                break
            time.sleep(0.1)
        self.assertFalse(alive(pid))
