"""Own a render job and stop only its marked processes, including orphan workers.

Children inherit an opaque job ID and canonical output path in their launch
environment. No process-group or command-substring match grants ownership.
Linux pidfds keep every signal attached to the verified process identity.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

JOB_KEY = "LOCOMANI_RENDER_JOB"
OUTPUT_KEY = "LOCOMANI_RENDER_OUTPUT"


@dataclass(frozen=True)
class Process:
    pid: int
    start: int
    job: str
    output: str


def _read_process(pid: int) -> Process | None:
    try:
        proc = Path(f"/proc/{pid}")
        if proc.stat().st_uid != os.getuid():
            return None
        env = dict(item.split(b"=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
        job = env.get(JOB_KEY.encode(), b"").decode()
        output = env.get(OUTPUT_KEY.encode(), b"").decode()
        if not job or not output:
            return None
        # comm can contain spaces/parentheses; field 22 follows its closing ')'.
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return Process(pid, int(fields[19]), job, output)
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def processes() -> list[Process]:
    return [p for path in Path("/proc").iterdir() if path.name.isdecimal() if (p := _read_process(int(path.name)))]


def select_job(population: list[Process], pattern: str = "") -> tuple[str, str] | None:
    jobs = {(p.job, p.output) for p in population if pattern in p.output}
    if not jobs:
        return None
    if len(jobs) != 1:
        names = "\n".join(f"  {output} ({job})" for job, output in sorted(jobs))
        raise RuntimeError(f"Ambiguous render job; use a unique output-directory pattern:\n{names}")
    return jobs.pop()


def _libc_call(name: str, *args: int) -> int:
    # The bundled Python builds omit os.pidfd_open (and sometimes the signal
    # wrapper) even on kernels/libc that support pidfds. Use the named libc API;
    # never fall back to signalling a numeric PID after an ownership check.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = getattr(libc, name)
    except AttributeError as error:
        raise RuntimeError("Stopping renders requires Linux libc pidfd support") from error
    result = function(*args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _open_pidfd(pid: int) -> int:
    return _libc_call("pidfd_open", pid, 0)


def _stopped(proc: Process) -> bool:
    try:
        state = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state in {"T", "t", "Z"} or _read_process(proc.pid) != proc
    except (OSError, IndexError):
        return True


def _signal(fd: int, value: int) -> None:
    try:
        _libc_call("pidfd_send_signal", fd, value, ctypes.c_void_p(), 0)
    except ProcessLookupError:
        pass


def stop_job(job: tuple[str, str], *, grace_seconds: float = 3.0) -> int:
    """Quiesce a job before termination, so children cannot escape the snapshot."""
    handles: dict[Process, int] = {}
    try:
        for _ in range(100):
            members = [p for p in processes() if (p.job, p.output) == job]
            if any(p.pid == os.getpid() for p in members):
                raise RuntimeError("Run stop_render from outside the job being stopped")
            new = [p for p in members if p not in handles]
            if not new and all(_stopped(p) for p in members):
                break
            for proc in new:
                try:
                    fd = _open_pidfd(proc.pid)
                except ProcessLookupError:
                    continue
                if _read_process(proc.pid) != proc:
                    os.close(fd)
                    continue
                handles[proc] = fd
                _signal(fd, signal.SIGSTOP)
            time.sleep(0.01)
        else:
            raise RuntimeError("Render job kept spawning processes; refused incomplete shutdown")

        for fd in handles.values():
            _signal(fd, signal.SIGTERM)
            _signal(fd, signal.SIGCONT)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and any(_read_process(p.pid) == p for p in handles):
            time.sleep(0.05)
        for proc, fd in handles.items():
            if _read_process(proc.pid) == proc:
                _signal(fd, signal.SIGKILL)
        return len(handles)
    finally:
        for fd in handles.values():
            _signal(fd, signal.SIGCONT)
            os.close(fd)


def run_job(command: list[str], output: Path) -> int:
    """Launch a marked process tree; lock its output directory against concurrent writers."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".render.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another render owns {output}") from error
        # Some workers close inherited descriptors when spawning. Their launch
        # identity still prevents reuse after the supervisor/driver has died.
        if any(p.output == str(output) for p in processes()):
            raise RuntimeError(f"An orphan render worker still owns {output}; stop that job first")
        job = uuid.uuid4().hex, str(output)
        env = os.environ.copy()
        env.update({JOB_KEY: job[0], OUTPUT_KEY: job[1]})
        # Keep the lock alive even if this supervisor exits before its worker.
        child = subprocess.Popen(command, env=env, pass_fds=(lock.fileno(),), start_new_session=True)
        try:
            return child.wait()
        except KeyboardInterrupt:
            stop_job(job)
            child.wait()
            return 130


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    stop = commands.add_parser("stop")
    stop.add_argument("pattern", nargs="?", default="")
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                parser.error("run requires a command after --")
            return run_job(command, args.out)
        job = select_job(processes(), args.pattern)
        if job is None:
            print("No managed render job matches; unmarked processes are left alone")
            return 0
        count = stop_job(job)
        print(f"Stopped {count} process(es) belonging to {job[1]}")
        return 0
    except RuntimeError as error:
        print(f"REFUSED: {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
