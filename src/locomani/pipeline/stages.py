"""Run a child process and retain its command, delivered settings and exit status."""

from __future__ import annotations

import json
import os
import subprocess
import time
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
