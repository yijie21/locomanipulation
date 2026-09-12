"""The one place a config object becomes shell environment (decision 8).

`scripts/stack/start_stack.sh` and everything it launches (the MuJoCo sim, the
C++ deploy, `restart_deploy.sh`) take their knobs as inherited environment
variables -- that boundary is not moving. This module is the only Python that
builds that env dict; every other caller passes a `SonicBackendConfig` (and,
for `start_stack`, a task-ish object carrying `prompt`/`init_token_file`)
instead of writing `os.environ` inline.
"""

from __future__ import annotations

import os
import subprocess

from locomani import paths
from locomani.backends.sonic.config import SonicBackendConfig

START_STACK = os.path.join(paths.SCRIPTS, "stack", "start_stack.sh")
STOP_INSTANCE = os.path.join(paths.SCRIPTS, "stack", "stop_instance.sh")


def backend_env(cfg: SonicBackendConfig) -> dict[str, str]:
    """The env dict `start_stack.sh` (and its children) expect for `cfg`."""
    env = {
        "NO_ENCODER": "1" if cfg.no_encoder else "0",
        "HAND": cfg.hand,
        "GYM_UPPER_SLEW_DEG": str(cfg.upper_slew_deg),
        "TF_GRIP_DELAY": str(cfg.grip_delay),
        "LOCOMANI_REPLACE": "1" if cfg.replace else "0",
        "WRIST_CAMS": "1" if cfg.wrist_cams else "0",
        "SIM_VIS": cfg.sim_vis,
        "SCENE_RANDOM": "1" if cfg.scene_random else "0",
        "TF_DATASET_FPS": str(cfg.dataset_fps),
        "CLEAN_DEPLOY": "1" if cfg.clean_deploy else "0",
        "EVAL_PARALLEL": "1" if cfg.eval_parallel else "0",
        "LOCKSTEP": "1" if cfg.lockstep.enabled else "0",
    }
    if cfg.obs_dump:
        env["TF_OBS_DUMP"] = cfg.obs_dump
    if cfg.sonic_dds_domain is not None:
        env["SONIC_DDS_DOMAIN"] = str(cfg.sonic_dds_domain)
    return env


def export_env(cfg: SonicBackendConfig) -> None:
    """Update THIS process's environment so a subprocess that inherits it
    (e.g. `grasp_gym.fresh_deploy` -> `restart_deploy.sh`) sees `cfg`."""
    os.environ.update(backend_env(cfg))


def start_stack(
    backend_cfg: SonicBackendConfig,
    task_cfg=None,
    *,
    instance: int = 0,
    gpu: str | None = None,
    video_dir: str | None = None,
    extra_env: dict | None = None,
) -> dict[str, str]:
    """Run `scripts/stack/start_stack.sh` with `backend_cfg` as environment.

    `task_cfg`, if given, supplies the task prompt (`task_cfg.prompt`) and the
    initial-token file (`task_cfg.init_token_file`) the deploy should validate
    against. Returns the env dict actually used, so a caller that goes on to
    invoke other subprocesses (e.g. `restart_deploy.sh` mid-round) can reuse it.
    """
    env = dict(os.environ)
    env.update(backend_env(backend_cfg))
    env["INSTANCE"] = str(instance)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # MUJOCO_EGL_DEVICE_ID is a PHYSICAL device index -- CUDA_VISIBLE_DEVICES
        # does not remap it, so setting only the latter leaves the sim rendering
        # on GPU 0 while its tensors run on GPU 1. Every hand-written runner in
        # scripts/ sets the two to the same single value; this does the same,
        # and only for a single device (a list is a CUDA-side choice with no
        # single EGL answer). An explicit extra_env entry still wins.
        if str(gpu).isdigit():
            env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    if video_dir is not None:
        env["SIM_VIDEO_DIR"] = str(video_dir)
    init_token_file = getattr(task_cfg, "init_token_file", None)
    if init_token_file:
        env["TF_INIT_TOKEN_FILE"] = str(init_token_file)
    if extra_env:
        env.update(extra_env)
    prompt = getattr(task_cfg, "prompt", None) or "walk forward"
    rc = subprocess.call(["bash", START_STACK, prompt], cwd=paths.REPO, env=env)
    if rc:
        raise RuntimeError(f"start_stack.sh exited {rc}")
    return env


def stop_instance(instance: int = 0, port: int | None = None) -> None:
    """Run `scripts/stack/stop_instance.sh`, scoped to one instance."""
    args = ["bash", STOP_INSTANCE, str(instance)]
    if port is not None:
        args.append(str(port))
    subprocess.call(args, cwd=paths.REPO)
