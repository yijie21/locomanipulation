#!/usr/bin/env python3
"""ScaleBFM / HEFT inside HumanoidArena: integration smoke (one Kit process per controller), 2026-11-25.

The process is probe_sonic.py's (HA's own sim_eval_vla.py helpers, imported: env creation, env-config YAML, seeding,
episode reset, fall detector, RobotController with use_rl_action_mode) with SonicActionProvider replaced by
ha_g1ctrl's provider (g1_backend's ScaleBFM or HEFT, LocoManiLab's control path). The native SONIC path is not touched:
run_ctrl.sh's switch CTRL=sonic runs HA's probe_sonic.py; CTRL=scalebfm|heft runs this file.

`HADriver` gives HA lml_host.HostDriver's surface (reset, start_frames, push(world qpos frame), step, sample, pl, to_mj,
ref_delay_s), so smoke_common's phases run here exactly as lml_ref.py runs them in LocoManiLab.

Phases (--phases, comma separated):
  actuators  HA's actuator config and what PhysX holds for the 29 body joints: at spawn (HA's own gains), after the
             controller's gains are written, and after one more HA episode reset without re-writing (does HA's
             reset revert written gains?); the physics step and control frequency; the three joint orders checked.
  kin        HA's G1 against robot_fk (the model our planner and g1_backend's FK use): wrist and torso positions relative
             to the pelvis at two poses (spawn, then after 3 s on the stance reference).
  stand / lag / reach / walk   smoke_common (stand --stand_s, default 30 s).
  seed       probe_sonic's seed phase (episode A, B, A; --seed_check_steps control steps each on the stance reference):
             the task's own reset, object placement and reward run end to end under this controller.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", Path.cwd())).resolve()
if str(ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_ROOT))
_spec = importlib.util.spec_from_file_location("sim_eval_vla", ISAACLAB_ROOT / "script" / "eval_scripts" / "sonic" / "sim_eval_vla.py")
sev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sev)


def _gpu_mem_used_mb(phys_gpu: str) -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", phys_gpu],
                             capture_output=True, text=True, timeout=10).stdout.strip()  # fmt: skip
        return float(out.splitlines()[0])
    except Exception:
        return -1.0


def jdump(o):
    import numpy as np

    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.bool_):
            return bool(x)
        return str(x)

    return json.dumps(o, default=conv)


def main() -> int:
    parser = sev._build_parser()
    parser.add_argument("--controller", choices=("scalebfm", "heft"), required=True)
    parser.add_argument("--probe_out", type=str, required=True)
    parser.add_argument("--phases", type=str, default="actuators,kin,stand,lag,reach,walk,seed")
    parser.add_argument("--stand_s", type=float, default=30.0)
    parser.add_argument("--reach_dist", type=float, default=0.30)
    parser.add_argument("--walk_dist", type=float, default=0.50)
    parser.add_argument("--seed_check_steps", type=int, default=100)
    parser.add_argument("--seed_b", type=int, default=12345)
    args_cli = parser.parse_args()
    phys_gpu = os.environ.get("HA_PHYS_GPU", str(sev._cuda_device_index(args_cli.device) or 0))
    phases = [p.strip() for p in args_cli.phases.split(",") if p.strip()]
    out_dir = Path(args_cli.probe_out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- exactly as sim_eval_vla.main() / probe_sonic.py ---
    sev._install_interrupt_handlers()
    sev._ensure_unique_multi_image_shm_name(args_cli)
    args_cli.enable_cameras = True
    args_cli.multi_gpu = False
    sev._append_unique_kit_arg(args_cli, "--/renderer/multiGpu/enabled=False")
    sev._configure_renderer_gpu(args_cli)
    sys.path.insert(0, str(HERE))
    import ha_g1ctrl as HC

    report_kit = HC.kit_single_gpu(args_cli)  # CUDA_VISIBLE_DEVICES=<card>: renderer on the physical card
    sev._normalize_control_routing(args_cli)
    sev.apply_task_runtime_profile(args_cli)
    t_launch = time.time()
    app_launcher = sev.AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import tasks  # noqa: F401
    from common_env_objects import get_current_episode_object_seed_info
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from layeredcontrol.robot_control_system import ControlConfig, RobotController
    from tasks.common_env_config import apply_env_config_yaml
    from tasks.common_runtime import apply_optional_runtime_augments

    HC.setup_paths()
    import smoke_common as SC

    import lml_host as LH
    import lml_selfmodel as L

    report = {"controller": args_cli.controller, "task": args_cli.task, "seed": args_cli.seed, "phys_gpu": phys_gpu,
              "gpu_mem_before_mb": _gpu_mem_used_mb(phys_gpu), "kit_args": report_kit,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}  # fmt: skip
    episode_seed = int(args_cli.episode_seed if args_cli.episode_seed is not None else args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    apply_env_config_yaml(env_cfg, args_cli.env_config_yaml, task_name=args_cli.task,
                          route_name=args_cli.gmt_backend or args_cli.action_source)  # fmt: skip
    sev._seed_runtime_rngs(episode_seed)
    sev._configure_episode_seed_state(env_cfg, episode_seed)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    sev._initialize_task_scene(env, env_cfg, args_cli, apply_optional_runtime_augments)
    sev._reset_environment_for_episode(env, env_cfg, episode_seed)
    physics_dt = float(getattr(env, "physics_dt", None) or env_cfg.sim.dt)
    control_dt = physics_dt * env_cfg.decimation
    control_hz = int(round(1.0 / control_dt))
    report.update(physics_dt=physics_dt, decimation=env_cfg.decimation, control_hz=control_hz, startup_s=time.time() - t_launch)
    print("[probe_ctrl] env up", jdump({k: report[k] for k in ("physics_dt", "decimation", "control_hz", "startup_s")}), flush=True)
    if control_hz != 50:  # the controllers run at 50 Hz (rl_rate 50, motion_dt_s 0.02): stop, do not adapt
        raise SystemExit(f"HA control frequency is {control_hz} Hz, the controllers need 50 Hz")

    robot = env.scene["robot"]
    from g1_bridge.g1_runtime import G1_JOINT_NAMES  # noqa: E402  (after setup_paths)

    names = list(robot.data.joint_names)
    mj_idx = [names.index(n) for n in G1_JOINT_NAMES]
    gains_spawn = HC.physx_gains(robot, mj_idx)  # HA's own drives before anything of ours is written

    t0 = time.time()
    provider = HC.make_action_provider(env, args_cli.controller)
    env.action_provider = provider
    report["controller_build_s"] = time.time() - t0
    controller = RobotController(env, ControlConfig(step_hz=control_hz, replay_mode=False, use_rl_action_mode=True))
    controller.set_action_provider(provider)
    controller._step_interval = 0.0  # unpaced
    fall_detector = sev._build_fall_detector(env, args_cli)
    ctrl = provider.ctrl
    kp, kd = ctrl.kp, ctrl.kd

    class HADriver(LH.HostDriver):
        """lml_host.HostDriver's surface on HA (smoke_common drives only this)."""

        def __init__(self):
            from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

            self.env, self.robot = env, robot
            self.pl = L.Planner()
            self.q_idx = torch.tensor([names.index(n) for n in L.SCALEBFM_ORDER], dtype=torch.long, device=env.device)
            self.to_mj = np.array([L.SCALEBFM_ORDER.index(n) for n in MUJOCO_JOINT_NAMES])
            self.mj_idx_t = torch.tensor(mj_idx, dtype=torch.long, device=env.device)
            bn = list(robot.data.body_names)
            self.b_pelvis = bn.index("pelvis")
            self.wrist = {s: bn.index(f"{s}_wrist_yaw_link") for s in ("left", "right")}
            self.torso = bn.index("torso_link")
            self.feet_b = {s: bn.index(f"{s}_ankle_roll_link") for s in ("left", "right")}
            self.buf = None
            self.c = np.zeros(3)
            self.steps = 0
            self.t_ctrl = 0.0
            self.max_force = None
            self.reset()
            self.c, self.kin0 = self.kin_check()

        # -- conventions ------------------------------------------------------------------------------------------
        def _pose(self, b):
            d = self.robot.data
            p = d.body_link_pos_w[0, b].detach().cpu().numpy().astype(np.float64)
            q = d.body_link_quat_w[0, b].detach().cpu().numpy().astype(np.float64)
            return p, L._qmat(q / np.linalg.norm(q)), q

        def kin_check(self) -> tuple[np.ndarray, dict]:
            """HostDriver._calibrate on HA's bodies (not asserted: reported): c = HA-body-relative-to-HA-pelvis minus
            robot_fk-body-relative-to-robot_fk-pelvis, in the pelvis frame, for both wrists and the torso."""
            pp, Rp, _ = self._pose(self.b_pelvis)
            q_mj = self.robot.data.joint_pos[0, self.mj_idx_t].detach().cpu().numpy().astype(np.float64)
            out = self.pl.fk_world(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), q_mj)
            cs, rot = {}, {}
            for s, b in self.wrist.items():
                pw, Rw, _ = self._pose(b)
                fk_rel = out[s][0] - out[s][1] @ L.HAND_OFF
                cs[s] = Rp.T @ (pw - pp) - fk_rel
                dR = (Rp.T @ Rw).T @ out[s][1]
                rot[s] = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            # torso through robot_fk's own model
            m, dd = self.pl.ik.model, self.pl.d
            tb = m.body("torso_link").id
            pt_fk = dd.xpos[tb].copy()  # fk_world left robot_fk's data at this pose (root at the origin, identity)
            pt, _, _ = self._pose(self.torso)
            cs["torso"] = Rp.T @ (pt - pp) - pt_fk
            c = np.mean([cs["left"], cs["right"]], axis=0)
            spread = max(float(np.linalg.norm(cs[k] - c)) for k in cs)
            return c, dict(c=c.tolist(), per_body={k: v.tolist() for k, v in cs.items()}, spread_m=spread,
                           wrist_rot_diff_deg=rot, q_mj=q_mj.tolist())  # fmt: skip

        def mass_report(self) -> dict:
            """Masses and the whole-body centre of mass (for the comparison with LocoManiLab's body): total, per body,
            the CoM relative to the pelvis (pelvis frame) and to the midpoint of the two ankle_roll_links (heading frame)."""
            m = self.robot.root_physx_view.get_masses()[0].detach().cpu().numpy().astype(np.float64)
            com_w = self.robot.data.body_com_pos_w[0].detach().cpu().numpy().astype(np.float64)
            pp, Rp, _ = self._pose(self.b_pelvis)
            com = (m[:, None] * com_w).sum(0) / m.sum()
            mid = 0.5 * (self._pose(self.feet_b["left"])[0] + self._pose(self.feet_b["right"])[0])
            Ry = L.rot_z(L.yaw_of(Rp))
            bn = list(self.robot.data.body_names)
            return dict(total_kg=float(m.sum()), per_body_kg={bn[i]: float(m[i]) for i in range(len(bn))},
                        com_in_pelvis=(Rp.T @ (com - pp)).tolist(), com_minus_ankle_mid_heading=(Ry.T @ (com - mid)).tolist())  # fmt: skip

        def pelvis_pose(self):
            p, R, _ = self._pose(self.b_pelvis)
            return p + R @ self.c, R

        def sample(self) -> dict:
            d = self.robot.data
            p, R = self.pelvis_pose()
            out = dict(root=np.r_[p, LH.mat_quat(R)],
                       q=d.joint_pos[0, self.q_idx].detach().cpu().numpy().astype(np.float64),
                       dq=d.joint_vel[0, self.q_idx].detach().cpu().numpy().astype(np.float64),
                       w=d.root_link_ang_vel_b[0].detach().cpu().numpy().astype(np.float64))  # fmt: skip
            out["wr"] = {s: self._pose(b)[:2] for s, b in self.wrist.items()}
            out["torso"] = self._pose(self.torso)[:2]
            out["feet"] = {s: self._pose(b)[0] for s, b in self.feet_b.items()}
            return out

        @property
        def ref_delay_s(self) -> float:
            return float(ctrl.delay_s)

        # -- the reference ----------------------------------------------------------------------------------------
        def push(self, fw):
            f = self.to_frame(fw) if self.buf is not None else np.asarray(fw, np.float64)
            provider.set_vla_action(dict(ref_root_pos=f[:3].astype(np.float32), ref_root_quat=f[3:7].astype(np.float32),
                                         ref_joint_pos=f[7:].astype(np.float32)))  # fmt: skip

        def step(self):
            n0 = provider.calls
            t = time.perf_counter()
            controller.step()
            self.t_ctrl += time.perf_counter() - t
            self.steps += 1
            if provider.calls != n0 + 1:
                raise RuntimeError("the provider was not stepped by RobotController")
            if self.buf is None:
                self.buf = ctrl.motion_buffer  # after the first payload: the episode transform is set
            if self.steps_since_reset == 1 and ctrl.settling:
                raise RuntimeError("the bridge is still settling: our reference is not being tracked")
            self.steps_since_reset += 1

        def extra(self) -> dict:
            fell, detail = sev._detect_robot_fall(env, fall_detector)
            d = self.robot.data
            q = d.joint_pos[0, self.mj_idx_t].detach().cpu().numpy()
            dq = d.joint_vel[0, self.mj_idx_t].detach().cpu().numpy()
            tq = ctrl.last[0] if ctrl.last is not None else q
            tau = kp * (tq - q) - kd * dq  # the PD law PhysX's drive applies (before its force clamp)
            ratio = np.abs(tau) / np.maximum(self.max_force, 1e-6)
            return dict(ha_fall=bool(fell), ha_fall_detail=str(detail) if fell else None,
                        sat_joints=int((ratio >= 1.0).sum()), pd_over_limit=float(ratio.max()),
                        pd_over_limit_joint=G1_JOINT_NAMES[int(ratio.argmax())] if ratio.max() >= 0.9 else None)  # fmt: skip

        # -- one reset --------------------------------------------------------------------------------------------
        def reset(self, seed: int | None = None):
            sev._reset_environment_for_episode(env, env_cfg, int(episode_seed if seed is None else seed))
            sev._notify_action_provider_env_reset(provider)  # -> provider.on_env_reset: new session, gains written
            controller.start()
            self.buf = None
            self.steps_since_reset = 0
            mf = HC.physx_gains(robot, mj_idx)["max_force"]
            self.max_force = np.asarray(mf, np.float64)

    h = HADriver()
    report["kin_spawn"] = h.kin0
    print("[probe_ctrl] driver up; kin at spawn", jdump(h.kin0), flush=True)

    tick0 = {}

    def start_phase():
        tick0.update(pol=provider.t_policy, phys=provider.t_phys)
        return time.time(), h.steps

    def finish_phase(name, t0, n0):
        dt = time.time() - t0
        n = h.steps - n0
        r = report[name]
        if isinstance(r, dict):
            tr = r.pop("_trace", None)
            if tr is not None:
                open(out_dir / f"{name}_trace.json", "w").write(jdump(tr))
            r["wall_s"] = dt
            r["control_steps"] = n
            r["ms_per_control_step"] = 1000 * dt / max(n, 1)
            r["ms_per_step_controller"] = 1000 * (provider.t_policy - tick0.get("pol", 0.0)) / max(n, 1)
            r["ms_per_step_physics"] = 1000 * (provider.t_phys - tick0.get("phys", 0.0)) / max(n, 1)
        print(f"[probe_ctrl] {name}", jdump(report[name]), flush=True)
        json.dump(report, open(out_dir / "report.json", "w"), indent=2, default=str)

    # ---------------- actuators ----------------
    if "actuators" in phases:
        t0, n0 = start_phase()
        acfg = {}
        for k, a in robot.cfg.actuators.items():
            acfg[k] = {f: (getattr(a, f) if not isinstance(getattr(a, f, None), dict) else dict(getattr(a, f)))
                       for f in ("joint_names_expr", "stiffness", "damping", "effort_limit", "effort_limit_sim",
                                 "velocity_limit", "velocity_limit_sim", "armature", "friction") if hasattr(a, f)}  # fmt: skip
            acfg[k]["class"] = type(a).__name__
        h.reset()
        gains_written = HC.physx_gains(robot, mj_idx)
        sev._reset_environment_for_episode(env, env_cfg, episode_seed)  # HA's reset alone, no re-write
        gains_after_reset = HC.physx_gains(robot, mj_idx)
        h.reset()
        sim = env.cfg.sim
        try:
            physx = sim.physx.to_dict()
        except Exception as e:  # noqa: BLE001
            physx = f"unavailable: {e}"
        try:
            from action_provider.action_provider_sonic import SONIC_EFFORT_LIMIT, SONIC_ISAACLAB_JOINT_ORDER, SONIC_PD_KD, SONIC_PD_KP

            sonic = {n: dict(kp=float(SONIC_PD_KP[i]), kd=float(SONIC_PD_KD[i]), effort=float(SONIC_EFFORT_LIMIT[i]))
                     for i, n in enumerate(SONIC_ISAACLAB_JOINT_ORDER)}  # fmt: skip
        except Exception as e:  # noqa: BLE001
            sonic = f"unavailable: {e}"

        def per_joint(g):
            return {n: {k: (v[i] if isinstance(v, list) else v) for k, v in g.items()} for i, n in enumerate(G1_JOINT_NAMES)}

        def same(a, b, key):
            if not isinstance(a.get(key), list) or not isinstance(b.get(key), list):
                return None
            return float(np.abs(np.asarray(a[key]) - np.asarray(b[key])).max())

        report["actuators"] = dict(
            physics=dict(sim_dt=float(sim.dt), decimation=int(env_cfg.decimation), control_hz=control_hz,
                         render_interval=getattr(sim, "render_interval", None), gravity=list(sim.gravity), physx=physx),
            controller=dict(backend=args_cli.controller, policy_config=ctrl.policy_config, delay_s=ctrl.delay_s,
                            kp={n: float(kp[i]) for i, n in enumerate(G1_JOINT_NAMES)},
                            kd={n: float(kd[i]) for i, n in enumerate(G1_JOINT_NAMES)}),
            ha_actuator_cfg=acfg, sonic_gains=sonic,
            physx_spawn=per_joint(gains_spawn), physx_written=per_joint(gains_written),
            written_stiffness_vs_kp_maxdiff=float(np.abs(np.asarray(gains_written["stiffness"]) - kp).max()),
            written_damping_vs_kd_maxdiff=float(np.abs(np.asarray(gains_written["damping"]) - kd).max()),
            after_reset_stiffness_vs_kp_maxdiff=float(np.abs(np.asarray(gains_after_reset["stiffness"]) - kp).max()),
            after_reset_stiffness_vs_spawn_maxdiff=same(gains_after_reset, gains_spawn, "stiffness"),
            after_reset_damping_vs_kd_maxdiff=float(np.abs(np.asarray(gains_after_reset["damping"]) - kd).max()),
            orders=HC.check_orders(ctrl.host, ctrl.session.tracking),
        )  # fmt: skip
        finish_phase("actuators", t0, n0)

    # ---------------- kin ----------------
    if "kin" in phases:
        t0, n0 = start_phase()
        h.reset()
        fr = h.start_frames()
        c0, k0 = h.kin_check()
        for f in range(150):
            h.push(fr[min(f, len(fr) - 1)])
            h.step()
        c1, k1 = h.kin_check()
        report["kin"] = dict(spawn=k0, after_3s_stance=k1, c_change_between_poses_m=float(np.linalg.norm(c1 - c0)),
                             c_used=h.c.tolist(), mass_after_3s_stance=h.mass_report())  # fmt: skip
        finish_phase("kin", t0, n0)

    # ---------------- smoke phases (shared with LocoManiLab) ----------------
    for name, fn in (("stand", lambda: SC.phase_stand(h, args_cli.stand_s)), ("lag", lambda: SC.phase_lag(h)),
                     ("reach", lambda: SC.phase_reach(h, "right", args_cli.reach_dist)),
                     ("walk", lambda: SC.phase_walk(h, args_cli.walk_dist))):  # fmt: skip
        if name in phases:
            t0, n0 = start_phase()
            try:
                report[name] = fn()
            except Exception as e:  # noqa: BLE001 -- the next phase still runs
                import traceback

                traceback.print_exc()
                report[name] = dict(error=f"{type(e).__name__}: {e}")
            finish_phase(name, t0, n0)

    # ---------------- seed (probe_sonic's flow) ----------------
    if "seed" in phases:
        t0, n0 = start_phase()

        def object_poses():
            from pxr import UsdGeom
            import omni.usd

            poses = {}
            stage = omni.usd.get_context().get_stage()
            xc = UsdGeom.XformCache()
            for rec in getattr(env.cfg, "deterministic_object_resets", []) or []:
                nm = rec.get("record_name")
                for key in rec.get("scene_keys", []) or []:
                    if key in env.scene.keys():
                        poses[nm] = env.scene[key].data.root_state_w[0, :7].cpu().numpy().astype(np.float64).tolist()
                        break
                else:
                    for p in rec.get("prim_paths", []) or []:
                        prim = stage.GetPrimAtPath(p.format(env_idx=0))
                        if prim and prim.IsValid():
                            m = xc.GetLocalToWorldTransform(prim)
                            t = m.ExtractTranslation()
                            q = m.ExtractRotationQuat()
                            poses[nm] = [t[0], t[1], t[2], q.GetReal(), *q.GetImaginary()]
                            break
            return poses

        def snap():
            d = robot.data
            return {"root_state_w": d.root_state_w[0].cpu().numpy().astype(np.float64),
                    "joint_pos_mj": d.joint_pos[0, h.mj_idx_t].cpu().numpy().astype(np.float64),
                    "joint_vel_mj": d.joint_vel[0, h.mj_idx_t].cpu().numpy().astype(np.float64)}  # fmt: skip

        def episode_trace(seed):
            h.reset(seed)
            info = {"object_seed": get_current_episode_object_seed_info(env.cfg), "objects": object_poses(), "s0": snap()}
            fr = h.start_frames()
            fell, rew = None, []
            for f in range(args_cli.seed_check_steps):
                h.push(fr[min(f, len(fr) - 1)])
                h.step()
                rew.append(float(sev._extract_reward_info(env)["raw_total"]))
                fd, det = sev._detect_robot_fall(env, fall_detector)
                if fd and fell is None:
                    fell = (f, str(det))
            info.update(s1=snap(), objects_after=object_poses(), reward_max=max(rew), reward_end=rew[-1], fall=fell,
                        gains_writes=provider.gains_written)  # fmt: skip
            return info

        A1 = episode_trace(episode_seed)
        B = episode_trace(args_cli.seed_b)
        A2 = episode_trace(episode_seed)

        def maxdiff(a, b):
            return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))

        n = args_cli.seed_check_steps
        report["seed"] = {
            "seed_a": episode_seed, "seed_b": args_cli.seed_b,
            "object_seed_a1": A1["object_seed"], "object_seed_a2": A2["object_seed"], "object_seed_b": B["object_seed"],
            "objects_a1": A1["objects"], "objects_b": B["objects"],
            "objects_a1_vs_a2_maxdiff": {k: maxdiff(A1["objects"][k], A2["objects"][k]) for k in A1["objects"]},
            "objects_a1_vs_b_maxdiff": {k: maxdiff(A1["objects"][k], B["objects"][k]) for k in A1["objects"]},
            "reset_state_a1_vs_a2_maxdiff": {k: maxdiff(A1["s0"][k], A2["s0"][k]) for k in A1["s0"]},
            f"after_{n}_steps_a1_vs_a2_maxdiff": {k: maxdiff(A1["s1"][k], A2["s1"][k]) for k in A1["s1"]},
            f"after_{n}_steps_a1_vs_b_maxdiff": {k: maxdiff(A1["s1"][k], B["s1"][k]) for k in A1["s1"]},
            "objects_after_a1_vs_a2_maxdiff": {k: maxdiff(A1["objects_after"][k], A2["objects_after"][k]) for k in A1["objects_after"]},
            "fall": {"a1": A1["fall"], "b": B["fall"], "a2": A2["fall"]},
            "reward_max": {"a1": A1["reward_max"], "b": B["reward_max"], "a2": A2["reward_max"]},
            "root_z_end": {"a1": float(A1["s1"]["root_state_w"][2]), "b": float(B["s1"]["root_state_w"][2]), "a2": float(A2["s1"]["root_state_w"][2])},
            "gains_writes": A2["gains_writes"],
        }  # fmt: skip
        finish_phase("seed", t0, n0)

    report["controller_ms_per_step_total"] = 1000 * h.t_ctrl / max(h.steps, 1)
    report["gpu_mem_end_mb"] = _gpu_mem_used_mb(phys_gpu)
    report["torch_max_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
    json.dump(report, open(out_dir / "report.json", "w"), indent=2, default=str)
    print("[probe_ctrl] DONE", out_dir / "report.json", flush=True)
    try:
        controller.cleanup()
        env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
