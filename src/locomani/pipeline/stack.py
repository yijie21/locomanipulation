"""Resolve SONIC child settings and start/stop one instance through `locomani.stack.Stack`."""

from __future__ import annotations

import json
import os
import subprocess

from locomani import paths
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.sim.ports import instance_ports


def backend_env(cfg: SonicBackendConfig) -> dict[str, str]:
    """The one variable every child and the agent read the backend config from."""
    return {"LOCOMANI_BACKEND_JSON": json.dumps(cfg.to_dict(), sort_keys=True)}


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
        "LOCOMANI_BACKEND_JSON",
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
    env.update({name: str(port) for name, port in instance_ports(instance).items()})
    env["INSTANCE"] = str(instance)
    if backend_cfg.sonic_dds_domain is not None:  # otherwise the stack derives it from the instance
        env["SONIC_DDS_DOMAIN"] = str(backend_cfg.sonic_dds_domain)
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
        for key in ("INSTANCE", *instance_ports(instance)):
            if key in extra_env and str(extra_env[key]) != env[key]:
                raise ValueError(f"{key} conflicts with instance {instance}")
        env.update(extra_env)
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
    receipt_path=None,
) -> dict[str, str]:
    """Start one stack through `locomani.stack.Stack` and return the environment its agent process needs.

    `runner`, when it is a `Stage.run`, only records the children's commands (nothing is executed
    through a shell any more); the live `Stack` is kept in `LIVE[instance]` for `stop_instance`.
    """
    from locomani.stack import Stack, StackSpec

    env = resolve_env(
        backend_cfg, task_cfg, instance=instance, gpu=gpu, video_dir=video_dir, extra_env=extra_env, inherit=inherit
    )
    cfg = SonicBackendConfig.from_dict(json.loads(env["LOCOMANI_BACKEND_JSON"]))
    spec = StackSpec(
        instance=int(instance),
        gpu=str(env.get("CUDA_VISIBLE_DEVICES", gpu if gpu is not None else "0")),
        scene_dir=env.get("LOCOMANI_SIM_DIR", paths.SIM_DIR),
        prompt=getattr(task_cfg, "prompt", None) or "walk forward",
        record=(int(instance) == 0) or env.get("RECORD_INSTANCE", "0") == "1",
        policy=env.get("SONIC_POLICY", paths.SONIC_POLICY),
        deploy_root=env.get("SONIC_DEPLOY_ROOT", paths.SONIC_DEPLOY),
        deploy_bin=env.get("SONIC_DEPLOY_BIN") or None,
        encoder=not cfg.no_encoder,
        lockstep=bool(cfg.lockstep.enabled),
        auto_start=bool(cfg.clean_deploy or cfg.lockstep.enabled),
        default_motion="neutral_kick_R_001__A543" if (cfg.clean_deploy or cfg.lockstep.enabled) else None,
        wrist_cams=bool(cfg.wrist_cams),
        video_dir=env.get("SIM_VIDEO_DIR") or None,
        sim_vis=env.get("SIM_VIS", "none"),
        scene_random=env.get("SCENE_RANDOM", "0") == "1",
        dds_domain=cfg.sonic_dds_domain,
        env={
            k: v
            for k, v in env.items()
            if k
            in (
                "LOCOMANI_BACKEND_JSON",
                "TF_INIT_TOKEN_FILE",
                "TF_OBS_DUMP",
                "TF_GRIP_DELAY",
                "HAND",
                "TF_DATASET_FPS",
                "TF_POSE_OBJECTS",
                "EXPORTER_THREADS",
            )
        },
        receipts=str(receipt_path) if receipt_path is not None else None,
    )
    record = None
    if runner is not None and hasattr(runner, "__self__") and hasattr(runner.__self__, "record"):
        record = runner.__self__.record
    if record is not None:
        record(name, ["locomani.stack", "up", str(instance)], env)
    stack_obj = Stack(spec, base_env=env, record_command=record)
    stack_obj.up()
    LIVE[int(instance)] = stack_obj
    return env


LIVE: dict[int, "object"] = {}


def stop_instance(instance: int = 0, port: int | None = None, *, runner=None) -> None:
    """Stop one instance's stack (this process's `Stack` if it started it, else the one in the state file)."""
    from locomani.stack import Stack

    stack_obj = LIVE.pop(int(instance), None)
    if stack_obj is None:
        try:
            stack_obj = Stack.attach(int(instance))
        except FileNotFoundError:
            stack_obj = None
    if stack_obj is not None:
        stack_obj.down()
    if port is not None:
        subprocess.call(["pkill", "-9", "-f", f"serve_policy[.]py .*--port {int(port)}( |$)"])
    if runner is not None and hasattr(runner, "__self__") and hasattr(runner.__self__, "record"):
        runner.__self__.record("stop_stack", ["locomani.stack", "down", str(instance)], {})
