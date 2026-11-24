"""One round of SIMPLE episodes of one task, the loop's unit (TODO 49.2): run_official_q.sh in Python.

    python -m locomani.bench.simple run <Task> --ids F --setting J [--stance J] --out F [--skills DIR] ...

A session
  1. waits for one of the slots (flock on <slots_dir>/slot<k>.lock): an Isaac server of its own (GPU 0) and a stereo depth
     service (GPU 1) on the sockets of that slot; SAM 3 is expected to be up and is never touched. The slots_dir is shared with
     the esc worktree's run_official_q.sh, so old and new runs together never hold more Isaac servers than there are slots;
  2. makes the look and the GraspGen-X grasps of the ids whose cache is missing (clouds and grasps <Task>_L<kind>_e<n>.npz and
     grasp_...npz in the cache folder of this package and these parameters; shared by every round and every instance;
     look + GraspGen-X are serialised per task by a lock);
  3. runs the bed's `batch` (one row per id into --out); a server that dies takes the running episode with it: the server is
     restarted and only ids without a row run again (at most `batch_tries` tries); a driver error is a row with loss "error";
  4. stops what it started (by pid; nothing else is ever signalled) and exits 0 only when every id has a row.

Exit codes: 0 every id has a row; 1 some id has none; 2 bad arguments; 3 no cache folder (the bed could not name it).
Timing lines (seconds) go to <run_dir>/q_<Task>.timing: slot_wait, server_start, depth_start, looks_<n>_ids, graspgen, batch, total.
Files per task in run_dir: server_<Task>.pid, depth_<Task>.pid, server_q_<Task>.log, depth_q_<Task>.log, run_q_<Task>.<pid>.log,
missing_<Task>.<pid>.txt, look_<Task>.<pid>.jsonl, scene/<id>/. Two sessions of one task must not share a run_dir (the loop
gives each unit its own).
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

from locomani.bench.simple.config import SimpleConfig

EXIT_OK, EXIT_ROWS, EXIT_ARGS, EXIT_NO_CACHE = 0, 1, 2, 3
ARTICULATED = ("OpenOvenTeleop", "CloseDoorTeleop", "OpenFaucetTeleop", "OpenTrashCanTeleop")
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]  # the src/ this package is imported from: the bed's child processes get it as PYTHONPATH


class SlotTimeout(RuntimeError):
    pass


def take_slot(cfg: SimpleConfig, timeout_s: float | None = None, sleep=time.sleep):
    """Wait for a free slot: the first of `cfg.slots` whose lock file can be flocked without blocking. -> (slot, open file); the
    lock is held for as long as the file is open (closing it, or the process ending, releases it)."""
    Path(cfg.slots_dir).mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        for k in cfg.slots:
            f = open(Path(cfg.slots_dir) / f"slot{k}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return k, f
            except OSError:
                f.close()
        if timeout_s is not None and time.time() - start >= timeout_s:
            raise SlotTimeout(f"no free slot among {tuple(cfg.slots)} in {timeout_s}s")
        sleep(cfg.slot_poll_s)


def missing_ids(ids: list[str], task: str, cache: Path) -> list[str]:
    """The ids whose look or grasps are not in the cache folder. The articulated tasks keep no cache (the skill's one look is
    taken in the episode), so nothing is missing for them."""
    if task in ARTICULATED:
        return []
    out = []
    for i in ids:
        _, kind, n = i.split("/")
        name = f"{task}_L{kind}_e{n}"
        if not ((cache / f"{name}.npz").is_file() and (cache / f"grasp_{name}.npz").is_file()):
            out.append(i)
    return out


def rows_missing(ids: list[str], out: Path) -> list[str]:
    """The ids that have no row in `out` (the final check of a session)."""
    try:
        have = {json.loads(line)["episode_id"] for line in open(out)}
    except FileNotFoundError:
        have = set()
    return [i for i in ids if i not in have]


class Session:
    def __init__(
        self,
        cfg: SimpleConfig,
        task: str,
        ids_file: Path,
        out: Path,
        setting: str = "{}",
        stance: str | None = None,
        forward: list[str] | None = None,
        slot_timeout_s: float | None = None,
        sleep=time.sleep,
    ):
        self.cfg, self.task, self.ids_file, self.out = cfg, task, Path(ids_file).resolve(), Path(out).resolve()
        self.setting, self.stance, self.forward = setting, stance, list(forward or [])
        self.slot_timeout_s, self.sleep = slot_timeout_s, sleep
        self.ids = [s.strip() for s in open(self.ids_file) if s.strip()]
        self.run = Path(cfg.run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run / f"run_q_{task}.{os.getpid()}.log"
        self.timing = self.run / f"q_{task}.timing"
        self.slot: int | None = None
        self.server: subprocess.Popen | None = None
        self.depth: subprocess.Popen | None = None
        self._locks: list = []

    # ---- environment and commands

    def env(self, **extra) -> dict:
        """The children's environment: this process's, minus EXAM_OPEN (never set), with the process hygiene the bed always had."""
        e = dict(os.environ)
        e.pop("EXAM_OPEN", None)
        tmp = os.environ.get("LOCOMANI_TMP")
        if tmp:
            e["TMPDIR"] = tmp
        e.update(NUMPY_MADVISE_HUGEPAGE="0", OMP_NUM_THREADS="2", MUJOCO_GL="egl")
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def sock(self, kind: str) -> Path:
        return Path(self.cfg.slots_dir) / f"s{self.slot}_{kind}.sock"

    def bed_cmd(self, sub: str, *args: str) -> list[str]:
        """`python -m locomani.bench.simple <sub> ...` under the sim venv, with this call's configs forwarded."""
        return [self.cfg.sim_python, "-m", "locomani.bench.simple", sub, *args, *self.forward]

    def bed_env(self) -> dict:
        return self.env(PYTHONPATH=str(SRC))

    def tim(self, name: str, t0: float) -> None:
        with open(self.timing, "a") as f:
            f.write(f"{name} {time.time() - t0:.1f}\n")

    def logfile(self):
        return open(self.log_path, "ab")

    # ---- the two services of the slot

    def stop(self, proc: subprocess.Popen | None) -> None:
        """SIGTERM to this process (by pid, as the shell's kill did), SIGKILL if it does not go; nothing else is signalled."""
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        except ProcessLookupError:
            pass

    def stop_all(self) -> None:
        self.stop(self.server)
        self.stop(self.depth)
        self.server = self.depth = None

    def start_server(self) -> bool:
        """The environment server of this slot (GPU `isaac_gpu`) in SIMPLE's venv; ready when its log says "listening"."""
        cfg = self.cfg
        self.stop(self.server)
        t0 = time.time()
        sock = self.sock("isaac")
        sock.unlink(missing_ok=True)
        log = self.run / f"server_q_{self.task}.log"
        log.write_text("")
        env = self.env(
            MUJOCO_GL="egl",
            OMNI_KIT_ACCEPT_EULA="YES",
            CUDA_VISIBLE_DEVICES=cfg.isaac_gpu,
            MUJOCO_EGL_DEVICE_ID=cfg.isaac_gpu,
            PYTHONPATH="src",  # SIMPLE's own src (the working directory is its root)
        )
        with open(log, "ab") as lf:
            self.server = subprocess.Popen(
                [cfg.simple_python, str(HERE / "server.py"), str(sock), cfg.xdir, cfg.env_names],
                cwd=cfg.simple_root, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )  # fmt: skip
        (self.run / f"server_{self.task}.pid").write_text(f"{self.server.pid}\n")
        ready = self.wait_for(lambda: "listening" in log.read_text(errors="replace"), cfg.server_ready_s, self.server)
        self.tim("server_start", t0)
        return ready

    def start_depth(self) -> bool:
        """The stereo depth service of this slot (GPU `depth_gpu`) in the sam3d env; ready when its socket exists."""
        cfg = self.cfg
        self.stop(self.depth)
        t0 = time.time()
        sock = self.sock("depth")
        sock.unlink(missing_ok=True)
        env = self.env(
            PYTHONPATH=cfg.lingbot_path, HF_HOME=cfg.hf_home, HF_HUB_OFFLINE="1",
            CUDA_VISIBLE_DEVICES=cfg.depth_gpu, PYTHONNOUSERSITE="1",
        )  # fmt: skip
        stereo = SRC / "locomani" / "perception" / "stereo_server.py"
        with open(self.run / f"depth_q_{self.task}.log", "ab") as lf:
            self.depth = subprocess.Popen(
                [cfg.sam3d_python, "-s", str(stereo), str(sock)],
                cwd=cfg.graspgen_root, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )  # fmt: skip
        (self.run / f"depth_{self.task}.pid").write_text(f"{self.depth.pid}\n")
        ready = self.wait_for(sock.exists, cfg.depth_ready_s, self.depth)
        self.tim("depth_start", t0)
        return ready

    def wait_for(self, ready, limit_s: float, proc: subprocess.Popen, step: float = 2.0) -> bool:
        """Poll `ready()` every `step` s up to `limit_s`; stops early when `proc` has died (then it is not ready)."""
        t0 = time.time()
        while time.time() - t0 < limit_s:
            if ready():
                return True
            if proc.poll() is not None:
                return ready()
            self.sleep(step)
        return ready()

    def alive(self, proc: subprocess.Popen | None) -> bool:
        return proc is not None and proc.poll() is None

    # ---- the bed's processes

    def bed(self, sub: str, *args: str) -> int:
        with self.logfile() as lf:
            return subprocess.run(
                self.bed_cmd(sub, "--slot", str(self.slot), "--setting", self.setting, *args),
                env=self.bed_env(), cwd=SRC.parent, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            ).returncode  # fmt: skip

    def cache_dir(self) -> Path | None:
        """The cache folder of this package and these parameters (the bed names it: its key hashes the package's camera view)."""
        with self.logfile() as lf:
            r = subprocess.run(
                self.bed_cmd("cachedir", self.task, "--setting", self.setting),
                env=self.bed_env(), cwd=SRC.parent, stdout=subprocess.PIPE, stderr=lf, text=True, stdin=subprocess.DEVNULL,
            )  # fmt: skip
        lines = [x for x in r.stdout.splitlines() if x.strip()]
        return Path(lines[-1]) if lines and r.returncode == 0 else None

    def graspgen(self, cache: Path) -> None:
        cfg = self.cfg
        grasps = SRC / "locomani" / "perception" / "grasps.py"
        with self.logfile() as lf:
            subprocess.run(
                [cfg.graspgen_python, str(grasps), str(cache), "--only", self.task, "--assets-dir", cfg.graspgen_assets],
                cwd=cfg.graspgen_root, env=self.env(CUDA_VISIBLE_DEVICES=cfg.graspgen_gpu),
                stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )  # fmt: skip

    def lock_cache(self):
        f = open(Path(self.cfg.slots_dir) / f"cache_{self.task}.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)  # one look + GraspGen-X pass per task at a time (blocking)
        return f

    # ---- the round

    def look_pass(self, cache: Path, miss_file: Path) -> None:
        """Depth service and server up, the look of the missing ids (tries), both stopped, then GraspGen-X on the cache folder."""
        self.start_depth()
        self.start_server()
        t = time.time()
        for _ in range(self.cfg.look_tries):
            rc = self.bed(
                "look", "--ids", str(miss_file), "--out", str(self.run / f"look_{self.task}.{os.getpid()}.jsonl")
            )
            if rc == 0:
                break
            self.stop(self.server)
            self.sleep(5)
            if not self.alive(self.depth):
                self.start_depth()
            self.start_server()
        self.tim(f"looks_{len(open(miss_file).read().split())}_ids", t)
        self.stop_all()
        t = time.time()
        self.graspgen(cache)
        self.tim("graspgen", t)

    def batch_pass(self) -> None:
        if not self.alive(self.depth):
            self.start_depth()
        if not self.alive(self.server):
            self.start_server()
        t = time.time()
        extra = ["--stance", self.stance] if self.stance else []
        for _ in range(self.cfg.batch_tries):
            if not self.alive(self.server):
                self.start_server()
            if not self.alive(self.depth):
                self.start_depth()
            if self.bed("batch", "--ids", str(self.ids_file), "--out", str(self.out), *extra) == 0:
                break
            self.stop(self.server)
            self.sleep(5)
            if not self.alive(self.depth):
                self.start_depth()
            self.start_server()
        self.tim("batch", t)

    def run_round(self) -> int:
        t0 = time.time()
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
            miss_file = self.run / f"missing_{self.task}.{os.getpid()}.txt"
            miss = missing_ids(self.ids, self.task, cache)
            miss_file.write_text("".join(i + "\n" for i in miss))
            if miss:
                with self.lock_cache() as cl:
                    miss = missing_ids(
                        self.ids, self.task, cache
                    )  # (another instance may have made them while we waited)
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
