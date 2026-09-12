"""Resolve SONIC child settings and start/stop one instance through the shared shell launcher."""

from __future__ import annotations

import json
import os
import subprocess

from locomani import paths
from locomani.backends.sonic.config import ENV_FIELDS, SonicBackendConfig, with_shell_overrides

START_STACK = os.path.join(paths.SCRIPTS, "stack", "start_stack.sh")
STOP_INSTANCE = os.path.join(paths.SCRIPTS, "stack", "stop_instance.sh")


def backend_env(cfg: SonicBackendConfig) -> dict[str, str]:
    """The env dict `start_stack.sh` (and its children) expect for `cfg`."""
    env = {}
    for name, (field_name, kind) in ENV_FIELDS.items():
        value = cfg
        for part in field_name.split("."):
            value = getattr(value, part)
        if value is not None:
            env[name] = str(int(value)) if kind is bool else str(value)
    env["LOCOMANI_BACKEND_JSON"] = json.dumps(cfg.to_dict(), sort_keys=True)
    return env


def export_env(cfg: SonicBackendConfig) -> None:
    """Update THIS process's environment so a subprocess that inherits it
    (e.g. `grasp_gym.fresh_deploy` -> `restart_deploy.sh`) sees `cfg`."""
    os.environ.update(backend_env(cfg))


def resolve_env(
    backend_cfg: SonicBackendConfig,
    task_cfg=None,
    *,
    instance: int = 0,
    gpu: str | None = None,
    video_dir: str | None = None,
    extra_env: dict | None = None,
    inherit: dict | None = None,
) -> dict[str, str]:
    """Resolve once; reuse this exact environment for the stack and its expert.

    Host bootstrap settings are inherited. Configuration-owned optional settings
    and instance ports are cleared before explicit values are applied.
    """
    if type(instance) is not int or instance < 0:
        raise ValueError("instance must be a non-negative integer")
    env = dict(os.environ if inherit is None else inherit)
    for name in (
        "TF_CAMERA_PORT",
        "TF_SONIC_PORT",
        "TF_OUT_PORT",
        "TF_KEYBOARD_PORT",
        "TF_PRIVILEGED_PORT",
        "TF_EPISODE_PORT",
        "SONIC_DDS_DOMAIN",
        "TF_OBS_DUMP",
        "TF_INIT_TOKEN_FILE",
        "SIM_VIDEO_DIR",
    ):
        env.pop(name, None)
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
        env.update(backend_env(with_shell_overrides(backend_cfg, extra_env)))
    return env


def start_stack(
    backend_cfg: SonicBackendConfig,
    task_cfg=None,
    *,
    instance: int = 0,
    gpu: str | None = None,
    video_dir: str | None = None,
    extra_env: dict | None = None,
    inherit: dict | None = None,
    runner=None,
    name: str = "start_stack",
) -> dict[str, str]:
    """Start one stack, optionally recording the launch through Stage.run."""
    env = resolve_env(
        backend_cfg, task_cfg, instance=instance, gpu=gpu, video_dir=video_dir, extra_env=extra_env, inherit=inherit
    )
    command = ["bash", START_STACK, getattr(task_cfg, "prompt", None) or "walk forward"]
    rc = runner(name, command, cwd=paths.REPO, env=env) if runner else subprocess.call(command, cwd=paths.REPO, env=env)
    if rc:
        raise RuntimeError(f"start_stack.sh exited {rc}")
    return env


def stop_instance(instance: int = 0, port: int | None = None, *, runner=None) -> None:
    """Run `scripts/stack/stop_instance.sh`, scoped to one instance."""
    args = ["bash", STOP_INSTANCE, str(instance)]
    if port is not None:
        args.append(str(port))
    if runner:
        runner("stop_stack", args, cwd=paths.REPO)
    else:
        subprocess.call(args, cwd=paths.REPO)
