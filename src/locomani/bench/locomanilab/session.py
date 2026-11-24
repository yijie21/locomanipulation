"""One round of LocoManiLab episodes of one task, the loop's unit (TODO 49.4): run_lml_q.sh / run_lml_bed2.sh / run_lml_bed3.sh in Python.

    python -m locomani.bench.locomanilab run <Task> --ids F --setting J [--stance J] --out F [--skills DIR] ...

A session
  1. waits for one of the slots (flock on <slots_dir>/slot<k>.lock): one slot = one simulator instance (the bed process on GPU
     `gpu` and the benchmark's environment server, which the bed starts itself and which dies with it). The slots_dir is shared
     with the esc worktree's runners, so old and new runs together share the slots;
  2. makes the look and the GraspGen-X grasps of the ids whose cache is missing (clouds and grasps <Task>_L<kind>_e<n>.npz and
     grasp_...npz in the cache folder of this package; shared by every round and every instance; one look + GraspGen-X pass per
     task and cache at a time, by a lock). The navigation, door and drawer-only tasks look at nothing (families.py). An id whose
     object was not seen, or whose cloud has no grasps, leaves a marker (.notfound / .nograsp) so that a second call does not
     look again;
  3. runs the bed's `batch` (one row per id into --out; an error is a row with loss "error"; a bed process that dies is
     started again, and only ids without a row run again, at most `batch_tries` tries);
  4. stops what it started (each child is the leader of a process group of its own; the group is signalled, nothing else is) and
     exits 0 only when every id has a row.

Rows already in --out are removed first (a call is a fresh round). Exit codes as the old runner: 0 every id has a row; 1 some id
has none; 2 bad arguments; 3 no cache folder (the bed could not name it); 143 stopped by INT / TERM.
Timing lines (seconds) go to <out>.timing: slot_wait, looks_<n>_ids, graspgen, batch, total. The scenes, server files and logs of a
call are in <run_dir>/call_<Task>.<pid> (handed to the bed as its run_dir): nothing is shared between two calls except the cache and
the slot locks.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from locomani.bench.locomanilab import families
from locomani.bench.locomanilab.config import LocoManiLabConfig
from locomani.bench.simple.session import (  # the slot lock and the row check are the same mechanics on both benchmarks
    EXIT_ARGS,
    EXIT_NO_CACHE,
    EXIT_OK,
    EXIT_ROWS,
    SlotTimeout,
    rows_missing,
    take_slot,
)

__all__ = ["EXIT_ARGS", "EXIT_NO_CACHE", "EXIT_OK", "EXIT_ROWS", "Session", "missing_ids", "rows_missing", "take_slot"]
EXIT_STOPPED = 143  # stopped by INT or TERM
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]  # the src/ this package is imported from: the bed's child processes get it as PYTHONPATH


def cloud_name(task: str, episode_id: str) -> str:
    _, kind, n = episode_id.split("/")
    return f"{task}_L{kind}_e{n}"


def missing_ids(ids: list[str], task: str, cache: Path) -> list[str]:
    """The ids whose look or grasps are not in the cache folder (and have no marker saying the look found nothing or the cloud
    has no grasps). A task that looks at nothing (families.TASKS[task].looks False) misses nothing."""
    if not families.info(task).looks:
        return []
    out = []
    for i in ids:
        f = cloud_name(task, i)
        if (cache / f"{f}.npz").is_file() and (cache / f"grasp_{f}.npz").is_file():
            continue
        if (cache / f"{f}.notfound").is_file() or (cache / f"{f}.nograsp").is_file():
            continue
        out.append(i)
    return out


def write_markers(miss: list[str], task: str, cache: Path, looks: Path) -> None:
    """After a look pass: `.notfound` for an id whose look ran and saw nothing (an errored look is tried again by the next call),
    `.nograsp` for a cloud GraspGen-X made no grasps for."""
    seen = {}
    if looks.exists():
        for line in open(looks):
            r = json.loads(line)
            seen[r["episode_id"]] = r
    for i in miss:
        name = cloud_name(task, i)
        if not (cache / f"{name}.npz").exists():
            if i in seen and seen[i].get("found") is False:
                (cache / f"{name}.notfound").touch()
        elif not (cache / f"grasp_{name}.npz").exists():
            (cache / f"{name}.nograsp").touch()


class Session:
    def __init__(
        self,
        cfg: LocoManiLabConfig,
        task: str,
        ids_file: Path,
        out: Path,
        setting: str = "{}",
        stance: str | None = None,
        forward: list[str] | None = None,
        slot_timeout_s: float | None = None,
        sleep=time.sleep,
        look: bool = True,
    ):
        # look: the look + GraspGen-X pass is needed (target=perception)
        self.cfg, self.task, self.ids_file, self.out = cfg, task, Path(ids_file).resolve(), Path(out).resolve()
        self.setting, self.stance, self.forward = setting, stance, list(forward or [])
        self.slot_timeout_s, self.sleep, self.look = slot_timeout_s, sleep, look
        self.ids = [s.strip() for s in open(self.ids_file) if s.strip()]
        self.run0 = Path(cfg.run_dir)
        self.call = self.run0 / f"call_{task}.{os.getpid()}"  # the scenes, server files and logs of this call
        self.timing = Path(str(self.out) + ".timing")
        self.log_path = self.call / "run_q.log"
        self.slot: int | None = None
        self.child: subprocess.Popen | None = None
        self._locks: list = []

    # ---- environment and commands

    def env(self, **extra) -> dict:
        """The children's environment: this process's, with the process hygiene the bed always had."""
        e = dict(os.environ)
        tmp = os.environ.get("LOCOMANI_TMP")
        if tmp:
            e["TMPDIR"] = tmp
        e.update(NUMPY_MADVISE_HUGEPAGE="0", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYTHONPATH=str(SRC))
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def bed_cmd(self, sub: str, *args: str) -> list[str]:
        """`python -m locomani.bench.locomanilab <sub> ...` under the sim venv, with this call's configs forwarded and the call's
        own run_dir last (it wins over a run_dir the caller gave)."""
        return [self.cfg.sim_python, "-m", "locomani.bench.locomanilab", sub, *args, *self.forward, "--set", f"run_dir={self.call}"]  # fmt: skip

    def tim(self, name: str, t0: float) -> None:
        with open(self.timing, "a") as f:
            f.write(f"{name} {time.time() - t0:.1f}\n")

    # ---- children: each in a session of its own, signalled by its group

    def run_child(self, cmd: list[str], env: dict, cwd=None, capture: bool = False) -> subprocess.CompletedProcess:
        """Run `cmd` in a new session and wait; the whole group is signalled afterwards (anything it left behind)."""
        self.call.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as lf:
            self.child = subprocess.Popen(
                cmd, env=env, cwd=cwd, stdout=subprocess.PIPE if capture else lf, stderr=lf if capture else subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, text=capture,
            )  # fmt: skip
            try:
                stdout, _ = self.child.communicate()
            finally:
                self.kill_group(self.child)
                rc, self.child = self.child.returncode, None
        return subprocess.CompletedProcess(cmd, rc, stdout)

    @staticmethod
    def kill_group(proc: subprocess.Popen | None) -> None:
        """SIGTERM, then SIGKILL, to the child's process group (it is the leader: pgid == pid); nothing else is signalled."""
        if proc is None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            if sig == signal.SIGTERM:
                for _ in range(40):
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        return
                    time.sleep(0.05)

    def stop_all(self) -> None:
        self.kill_group(self.child)
        self.child = None

    # ---- the bed's processes

    def bed(self, sub: str, *args: str, gpu: int | None = None) -> int:
        return self.run_child(
            self.bed_cmd(sub, "--setting", self.setting, *args),
            self.env(CUDA_VISIBLE_DEVICES=self.cfg.gpu if gpu is None else gpu), cwd=SRC.parent,
        ).returncode  # fmt: skip

    def cache_dir(self) -> Path | None:
        """The cache folder of this package (the bed names it: its key hashes the package's files)."""
        r = self.run_child(
            self.bed_cmd("cachedir", self.task, "--setting", self.setting), self.env(), cwd=SRC.parent, capture=True
        )
        lines = [x for x in (r.stdout or "").splitlines() if x.strip()]
        return Path(lines[-1]) if lines and r.returncode == 0 else None

    def graspgen(self, cache: Path) -> None:
        cfg = self.cfg
        grasps = SRC / "locomani" / "perception" / "grasps.py"
        self.run_child(
            [cfg.graspgen_python, str(grasps), str(cache), "--only", self.task, "--assets-dir", cfg.graspgen_assets],
            self.env(CUDA_VISIBLE_DEVICES=cfg.graspgen_gpu), cwd=cfg.graspgen_root,
        )  # fmt: skip

    def lock_cache(self, cache: Path):
        f = open(Path(self.cfg.slots_dir) / f"cache_{cache.name}_{self.task}.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)  # one look + GraspGen-X pass per task and cache at a time (blocking)
        return f

    # ---- the round

    def look_pass(self, cache: Path, miss_file: Path) -> None:
        looks = self.call / "looks.jsonl"
        t = time.time()
        for _ in range(self.cfg.look_tries):
            if self.bed("look", "--ids", str(miss_file), "--out", str(looks)) == 0:
                break
            looks.unlink(missing_ok=True)  # (a failed try's rows are done ids: a new try starts over)
        self.tim(f"looks_{len(open(miss_file).read().split())}_ids", t)
        t = time.time()
        self.graspgen(cache)
        self.tim("graspgen", t)
        write_markers([s.strip() for s in open(miss_file) if s.strip()], self.task, cache, looks)

    def batch_pass(self) -> None:
        t = time.time()
        extra = ["--stance", self.stance] if self.stance else []
        for _ in range(self.cfg.batch_tries):
            self.bed("batch", "--ids", str(self.ids_file), "--out", str(self.out), *extra)
            if not rows_missing(
                self.ids, self.out
            ):  # (a bed process that died leaves ids without a row: they run again)
                break
        self.tim("batch", t)

    def run_round(self) -> int:
        t0 = time.time()
        self.call.mkdir(parents=True, exist_ok=True)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        for p in (self.out, *(Path(str(self.out) + s) for s in (".part0", ".part0.ids", ".part0.log"))):
            p.unlink(missing_ok=True)  # a call is a fresh round
        self.timing.write_text("")
        try:
            self.slot, lock = take_slot(self.cfg, self.slot_timeout_s, self.sleep)
        except SlotTimeout as e:
            print(e, file=sys.stderr)
            return EXIT_ROWS
        self._locks.append(lock)
        self.tim("slot_wait", t0)
        try:
            cache = self.cache_dir()
            if cache is None:
                print("no cache folder", file=sys.stderr)
                return EXIT_NO_CACHE
            cache.mkdir(parents=True, exist_ok=True)
            miss_file = self.call / "missing.txt"
            miss = missing_ids(self.ids, self.task, cache) if self.look else []  # target=truth has no look pass
            miss_file.write_text("".join(i + "\n" for i in miss))
            if miss:
                with self.lock_cache(cache) as cl:
                    miss = missing_ids(self.ids, self.task, cache)  # (another call may have made them while we waited)
                    miss_file.write_text("".join(i + "\n" for i in miss))
                    if miss:
                        self.look_pass(cache, miss_file)
                    fcntl.flock(cl, fcntl.LOCK_UN)
            self.batch_pass()
            self.stop_all()
            gone = rows_missing(self.ids, self.out)
            print(f"rows {len(self.ids) - len(gone)}/{len(self.ids)}", "missing " + ",".join(gone[:5]) if gone else "")
            rc = EXIT_ROWS if gone else EXIT_OK
            self.tim("total", t0)
            return rc
        finally:
            self.stop_all()
            for f in self._locks:
                f.close()  # releases the slot lock
            self._locks.clear()
