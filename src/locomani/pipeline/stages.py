"""Retain what a stage acted on, what it delivered, and how it ended.

A stage that spawns a child records the child's argv (`run` / `check` / `record`). A stage that
runs IN THIS PROCESS has no argv to record, so it records the configuration it resolved instead
(`record_config` / `timed`) -- the same question answered, which is what the receipt is for
(docs/handoff/flat-functional-scripts-20260922.md, decision 4).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from locomani import paths


def recorded_settings(env: dict[str, str]) -> dict[str, str]:
    """Record runtime configuration without copying host credentials into artifacts."""
    names = {
        "INSTANCE",
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_EGL_DEVICE_ID",
        "SONIC_POLICY",
        "SONIC_DEPLOY_ROOT",
        "NO_ENCODER",
        "HAND",
        "WRIST_CAMS",
        "SIM_VIS",
        "SCENE_RANDOM",
        "CLEAN_DEPLOY",
        "EVAL_PARALLEL",
        "SONIC_DDS_DOMAIN",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "DATA",
        "RUN",
        "OUT_ROOT",
        "MAX_STEPS",
        "SAVE_STEPS",
        "SAVE_LIMIT",
        "WORKERS",
        "GPUS",
        "SINGLE_GPU",
        "SQUAT_MAX",
        "WANDB",
    }
    prefixes = ("LOCOMANI_", "TF_", "SIM_", "LOCKSTEP", "GYM_", "TRANSPORT_")
    return {k: str(v) for k, v in env.items() if k in names or k.startswith(prefixes)}


class Stage:
    def __init__(self, out: Path):
        self.out = Path(out)

    def run(self, name: str, cmd, env: dict | None = None, cwd=None) -> int:
        command = [str(x) for x in cmd]
        cwd = str(cwd or paths.REPO)
        env = dict(os.environ) if env is None else dict(env)
        started = time.time()
        receipt = dict(stage=name, argv=command, cwd=cwd, settings=recorded_settings(env or {}), start=started)
        with (self.out / "commands.jsonl").open("a") as f:
            f.write(json.dumps(receipt) + "\n")
        log = self.out / f"{name}.log"
        print(f"{name}: start  log={log}", flush=True)
        rc = -1
        try:
            with log.open("a") as f:
                f.write("\nCOMMAND " + json.dumps(command) + "\n")
                f.flush()
                rc = subprocess.call(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
            return rc
        finally:
            with (self.out / "timings.jsonl").open("a") as f:
                f.write(json.dumps(dict(stage=name, start=started, seconds=time.time() - started, rc=rc)) + "\n")
            print(f"{name}: exit {rc}", flush=True)

    def record(self, name: str, argv, env: dict | None = None) -> None:
        """Record a command another object executed (the stack manager's children)."""
        receipt = dict(
            stage=name,
            argv=[str(x) for x in argv],
            cwd=str(paths.REPO),
            settings=recorded_settings(env or {}),
            start=time.time(),
        )
        with (self.out / "commands.jsonl").open("a") as f:
            f.write(json.dumps(receipt) + "\n")

    def check(self, name: str, cmd, env: dict | None = None, cwd=None) -> None:
        rc = self.run(name, cmd, env=env, cwd=cwd)
        if rc:
            raise RuntimeError(f"{name} exited {rc}: {self.out / (name + '.log')}")

    def record_config(self, name: str, config: dict, env: dict | None = None) -> None:
        """Record the configuration an in-process stage resolved, where an argv would have gone.

        `config` is what the stage will act on -- for the gym, `GymOptions.report()`, which carries
        every option AND where each one came from. That is strictly more than the argv it replaced:
        an argv showed the values, not whether a value was the config's or the code's default.
        """
        receipt = dict(
            stage=name,
            config=config,
            cwd=str(paths.REPO),
            settings=recorded_settings(env or dict(os.environ)),
            start=time.time(),
        )
        with (self.out / "commands.jsonl").open("a") as f:
            f.write(json.dumps(receipt, default=str) + "\n")

    @contextlib.contextmanager
    def timed(self, name: str):
        """Time an in-process stage, keep its output in `<name>.log`, and record how it ended.

        The log is what `run` leaves for a child. The stages that moved in-process on 2026-09-22
        left none, while the runbooks still sent readers to `collect_NN.log` / `visual_NN.log`
        for per-round outcomes -- so an unattended run's only record was a terminal (2026-09-23).
        """
        started = time.time()
        log = self.out / f"{name}.log"
        print(f"{name}: start  log={log}", flush=True)
        rc = 0
        try:
            with tee_output(log, f"===== {name} start {time.strftime('%FT%TZ', time.gmtime())} ====="):
                try:
                    yield
                except BaseException:
                    traceback.print_exc()  # into the log: a caller may catch it and carry on
                    raise
        except BaseException:
            rc = 1
            raise
        finally:
            with (self.out / "timings.jsonl").open("a") as f:
                f.write(json.dumps(dict(stage=name, start=started, seconds=time.time() - started, rc=rc)) + "\n")
            print(f"{name}: exit {rc}", flush=True)


@contextlib.contextmanager
def tee_output(log: Path, header: str):
    """Copy everything written to fds 1 and 2 -- Python, C extensions, children -- into `log` too.

    At the descriptor level rather than by swapping `sys.stdout`, because the gym's clients and the
    libraries under them write to the fds directly, and a child a stage spawns inherits them.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    read, write = os.pipe()
    with Path(log).open("ab") as f:
        f.write(("\n" + header + "\n").encode())
        f.flush()

        def pump() -> None:
            try:
                while chunk := os.read(read, 65536):
                    try:
                        os.write(saved[0], chunk)
                    except OSError:
                        pass  # the terminal went away; the log still gets it
                    f.write(chunk)
                    f.flush()
            except (OSError, ValueError):
                pass  # the stage ended with a child still holding the pipe; its output stops here

        thread = threading.Thread(target=pump, name=f"tee:{Path(log).name}", daemon=True)
        thread.start()
        os.dup2(write, 1)
        os.dup2(write, 2)
        os.close(write)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            # EOF arrives once no copy of the write end is left; a child that outlived the stage
            # still holds one, so bound the wait rather than hang on it.
            thread.join(timeout=5)
            os.close(read)
            os.close(saved[0])
            os.close(saved[1])
