"""selfmodel_hacollect_20261126 (development, 2026-11-26): how HA's physics cost grows with the number of environments in
one process. ha_sim.Sim's start-up (HA's sim_eval_vla helpers, ppbox's env-config YAML) with num_envs = N instead of 1,
no action provider, no controller: every robot holds its default joint targets; per physics substep the wall time of
write_data_to_sim + sim.step(render=False) + scene.update, as ha_g1ctrl's provider loop does them. Only the physics side
of a multi-env collector; the controllers' own per-env cost (policy, our host) adds N times its single-env value.

    hc_physn.sh <gpu> --n 1,4,16 (one process per N: Kit starts once per process)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

W_REPO = Path(os.environ.get("SX_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(W_REPO / "scripts/experiments/selfmodel_xfer_20261125"))
import sx_ha as SXH  # noqa: E402

ha_sim = SXH.ha_sim
sev = ha_sim.sev


def main() -> int:
    p = ha_sim.build_parser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--skip-reset", action="store_true", help="no HA episode reset (its task code indexes env 0 only: N > 1 asserts)")
    a = p.parse_args()
    _orig = sev._configure_renderer_gpu

    def _cfg(args):
        _orig(args)
        SXH.kit_single_gpu(args)

    sev._configure_renderer_gpu = _cfg
    a = ha_sim.finish_args(a, a.ha_task)
    sev._install_interrupt_handlers()
    a.enable_cameras = True
    a.multi_gpu = False
    sev._append_unique_kit_arg(a, "--/renderer/multiGpu/enabled=False")
    sev._configure_renderer_gpu(a)
    sev._normalize_control_routing(a)
    sev.apply_task_runtime_profile(a)
    t0 = time.time()
    app = sev.AppLauncher(a).app  # noqa: F841
    import gymnasium as gym
    import tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_env_config import apply_env_config_yaml
    from tasks.common_runtime import apply_optional_runtime_augments

    env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=int(a.n))
    env_cfg.env_name = a.task
    apply_env_config_yaml(env_cfg, a.env_config_yaml, task_name=a.task, route_name=a.gmt_backend or a.action_source)
    sev._seed_runtime_rngs(100)
    sev._configure_episode_seed_state(env_cfg, 100)
    env = gym.make(a.task, cfg=env_cfg).unwrapped
    sev._initialize_task_scene(env, env_cfg, a, apply_optional_runtime_augments)
    if not a.skip_reset:
        sev._reset_environment_for_episode(env, env_cfg, 100)
    up = time.time() - t0
    rob = env.scene["robot"]
    tgt = rob.data.default_joint_pos.clone()
    dt = env.physics_dt
    T = {"write": 0.0, "step": 0.0, "update": 0.0}
    z = []
    for k in range(a.steps):
        t1 = time.perf_counter()
        rob.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()
        t2 = time.perf_counter()
        env.sim.step(render=False)
        t3 = time.perf_counter()
        env.scene.update(dt=dt)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if k >= 50:
            T["write"] += t2 - t1
            T["step"] += t3 - t2
            T["update"] += t4 - t3
        if k % 100 == 0:
            z.append(rob.data.root_state_w[:, 2].detach().cpu().numpy().round(3).tolist())
    n = a.steps - 50
    ms = {k: round(1000 * v / n, 3) for k, v in T.items()}
    tot = sum(ms.values())
    print("HC_PHYSN " + json.dumps(dict(n_envs=int(a.n), skip_reset=bool(a.skip_reset), up_s=round(up, 1), ms_per_substep=ms, total_ms=round(tot, 3),
                                        ms_per_env_substep=round(tot / a.n, 3), env_spacing=getattr(env_cfg.scene, "env_spacing", None),
                                        max_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20), pelvis_z=z[-1][:4])), flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    ha_sim.main_guard(main)
