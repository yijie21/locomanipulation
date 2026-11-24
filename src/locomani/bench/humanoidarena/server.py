"""The HumanoidArena environment server (TODO 49.5b): one standalone file, run under the benchmark's own venv (Isaac Sim 5.0, Isaac Lab 2.2).

    third_party/HumanoidArena/venv/bin/python server.py SOCK --root .../repo/isaaclab_twist2_g1 --kit-gpu 1

It imports nothing of locomani, listens on one unix socket and answers pickled messages from one client (client.ServerLink). It dies
with the client. It drives the benchmark's own evaluation worker (script/eval_scripts/twist2/sim_eval_vla.py) as a library: the same
environment, task config, TWIST2 tracker (the ONNX the benchmark ships), reward and fall detector; only the policy's HTTP client is
replaced, in process, by one that returns the action the bed sends. No file of the benchmark is edited. Messages (a tuple, verb first):

    ("hello",)                          -> versions
    ("reset", task, group, repeat)      -> the episode; the seed is derived here (sha256 of "<gym task>|<group>|<repeat>")
    ("step", action40)                  -> one 50 Hz control step with the 40-D reference action; success / fell / state
    ("state",)                          -> the robot's readings
    ("rgbd",)                           -> front camera colour, metric depth, intrinsics, the camera in the pelvis frame
    ("truth",)                          -> the simulator's truth: object pose, task facts, hand and body-link positions
    ("jac", side)                       -> the arm joints' index list, the hand position and its position Jacobian (world frame)
    ("outcome",)                        -> the benchmark's own verdict and the per-step contact record of robot links on the task object
    ("close",)

Seeds. The benchmark's evaluation episodes are the 60 seeds sha256("<task>|<g>|<r>") for g in {0,1,2}, r in 0..19. `reset` derives the
episode seed itself and refuses it when it is one of those 60, and refuses any group in {0,1,2} (the last of the layers,
tests/test_humanoidarena_episodes.py). Physics runs on the CPU (the benchmark's own setting), rendering on the GPU named by --kit-gpu.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener

import numpy as np

TASKS = {
    "pp_box": dict(
        gym="Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
        yaml="tasks/common_test_config/base_test/pp_box_twist2_test.yaml",
        obj_prim="/World/envs/env_.*/Box",
        obj_key="box",
        max_steps=1450,
        instruction="Pick up the box and place it on the shelf.",
    ),
    "double_desk": dict(
        gym="Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
        yaml="tasks/common_test_config/base_test/doubledesk_twist2_test.yaml",
        obj_prim="/World/envs/env_.*/ObjectL",
        obj_key="object_l",
        max_steps=2000,
        instruction="Pick up the hammer and put it in the basket.",
    ),
    "open_door": dict(
        gym="Isaac-Move-Open-Door-G129-Dex3-Wholebody",
        yaml="tasks/common_test_config/base_test/open_door_twist2_test.yaml",
        obj_prim=None,
        obj_key=None,
        max_steps=1800,
        instruction="Open the door and walk through it.",
    ),
}
EVAL_GROUPS = (0, 1, 2)
EVAL_REPEATS = 20
HAND_TOKENS = ("hand", "palm", "thumb", "index", "middle")


def derive_seed(gym_task: str, group: int, repeat: int) -> int:
    """The benchmark's own formula (eval_vla_suite.py:_derive_episode_seed)."""
    payload = f"{gym_task}|{int(group)}|{int(repeat)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big", signed=False) & 0x7FFFFFFF


def eval_seed_set(gym_task: str) -> set[int]:
    return {derive_seed(gym_task, g, r) for g in EVAL_GROUPS for r in range(EVAL_REPEATS)}


def die_with_parent():
    import ctypes

    ctypes.CDLL("libc.so.6").prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM


class ActionClient:
    """Stands in for the benchmark's LeRobotVLAHttpClient inside the TWIST2 provider: returns the action the bed set."""

    def __init__(self):
        self.next = None
        self.rgb = None
        self.obs_state = None
        self.calls = 0

    def infer_chunk(self, front_rgb, observation_state, robot_type, task=None):
        self.rgb, self.obs_state = np.asarray(front_rgb), np.asarray(observation_state, np.float32)
        self.calls += 1
        if self.next is None:
            raise RuntimeError("the provider asked for an action before the bed set one")
        return np.asarray(self.next, np.float32).reshape(1, -1)

    def reset(self, seed=None):
        self.next = None


class Server:
    def __init__(self, root: str, kit_gpu: int, device: str):
        self.root, self.kit_gpu, self.device = os.path.abspath(root), kit_gpu, device
        self.key = None
        self.env = None
        self.client = ActionClient()
        self.over = None

    # ---- boot (first reset): the benchmark's own worker as a library

    def boot(self, key: str):
        spec = TASKS[key]
        os.chdir(self.root)
        sys.path.insert(0, self.root)
        p = os.path.join(self.root, "script/eval_scripts/twist2/sim_eval_vla.py")
        s = importlib.util.spec_from_file_location("ha_sim_eval_vla", p)
        sev = importlib.util.module_from_spec(s)
        sys.modules["ha_sim_eval_vla"] = sev
        s.loader.exec_module(sev)
        self.sev = sev
        argv = ["--task", spec["gym"], "--env_config_yaml", spec["yaml"], "--seed", "0", "--max_steps", "1",
                "--model_path", os.path.join(self.root, "../TWIST2/assets/ckpts/twist2_1017_20k.onnx"),
                "--lerobot_server_url", "http://unused.invalid", "--device", self.device, "--enable_cameras", "--headless",
                f"--kit_args=--/renderer/activeGpu={self.kit_gpu}"]  # fmt: skip
        sys.argv = [sys.argv[0], *argv]  # Kit and the benchmark's helpers re-read sys.argv
        args = sev._build_parser().parse_args(argv)
        sev._ensure_unique_multi_image_shm_name(args)
        args.enable_cameras, args.multi_gpu = True, False
        args.kit_args = (args.kit_args + " --/renderer/multiGpu/enabled=False").strip()
        sev._normalize_control_routing(args)
        sev.apply_task_runtime_profile(args)
        self.args = args
        self.app = sev.AppLauncher(args).app

        import gymnasium as gym
        import tasks  # noqa: F401
        from action_provider.create_action_provider import create_action_provider
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from layeredcontrol.robot_control_system import ControlConfig, RobotController
        from tasks.common_env_config import apply_env_config_yaml
        from tasks.common_runtime import apply_optional_runtime_augments

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        cfg.env_name = args.task
        apply_env_config_yaml(
            cfg, args.env_config_yaml, task_name=args.task, route_name=args.gmt_backend or args.action_source
        )
        first = derive_seed(spec["gym"], 1000, 0)
        sev._seed_runtime_rngs(first)
        sev._configure_episode_seed_state(cfg, first)
        if spec["obj_prim"]:  # robot links touching the task object: the compliance record (no change to the physics)
            cfg.scene.obj_contacts = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", filter_prim_paths_expr=[spec["obj_prim"]],
                                                      history_length=20, update_period=0.0, debug_vis=False)  # fmt: skip
        self.cfg = cfg
        self.env = gym.make(args.task, cfg=cfg).unwrapped
        sev._initialize_task_scene(self.env, cfg, args, apply_optional_runtime_augments)
        sev._reset_environment_for_episode(self.env, cfg, first)
        physics_dt = getattr(self.env, "physics_dt", None) or cfg.sim.dt
        hz = int(round(1.0 / (physics_dt * cfg.decimation)))
        self.provider = create_action_provider(self.env, args)
        sev._disable_action_provider_internal_recording(self.provider)
        self.env.action_provider = self.provider
        self.provider._lerobot_http_client = self.client
        self.ctl = RobotController(self.env, ControlConfig(step_hz=hz, replay_mode=False, use_rl_action_mode=True))
        self.ctl.set_action_provider(self.provider)
        self.key, self.spec = key, spec
        self.robot = self.env.scene["robot"]
        self.body_names = list(self.robot.body_names)
        self.joint_names = list(self.robot.joint_names)
        self.hand_ids = [
            i for i, n in enumerate(self.body_names) if any(t in n for t in HAND_TOKENS) and "hip" not in n
        ]
        self.side_wrist = {s: self.body_names.index(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        from action_provider.vla_smpl_runtime import CANONICAL_G1_JOINT_NAMES_29

        self.canon = list(CANONICAL_G1_JOINT_NAMES_29)
        self.arm_joint_ids = {
            s: [self.joint_names.index(f"{s}_{j}_joint") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
            for s in ("left", "right")
        }  # fmt: skip

    # ---- episodes

    def reset(self, key: str, group: int, repeat: int):
        if key not in TASKS:
            raise ValueError(f"task {key!r}")
        spec = TASKS[key]
        seed = derive_seed(spec["gym"], group, repeat)
        if group in EVAL_GROUPS or seed in eval_seed_set(spec["gym"]) or seed < 0:
            raise ValueError(
                f"episode (group {group}, repeat {repeat}, seed {seed}) refused: it is one of the benchmark's evaluation episodes"
            )
        if self.key is None:
            self.boot(key)
        elif self.key != key:
            raise ValueError(f"this server runs {self.key}; start a new one for {key}")
        sev = self.sev
        if self.ctl.is_running:
            self.ctl.stop()
        sev._reset_environment_for_episode(self.env, self.cfg, seed)
        sev._reset_lerobot_runtime_seed(self.provider, seed)
        sev._notify_action_provider_env_reset(self.provider)
        self.fall = sev._build_fall_detector(self.env, self.args)
        self.streak = 0
        self.steps, self.over = 0, None
        self.success = self.fell = False
        self.contacts: dict[str, list[int]] = {}
        self.client.next = None
        self._aa = None
        self.ctl.start()
        self.obj0 = self.obj_pos()
        self.obj_max_move = 0.0
        return dict(seed=seed, instruction=spec["instruction"], max_steps=spec["max_steps"], dt=0.02, body_names=self.body_names,
                    joint_names=self.joint_names, canonical_names=self.canon, hand_bodies=[self.body_names[i] for i in self.hand_ids], state=self.state(),
                    twist2_indices=[int(i) for i in self.provider.twist2_action_indices])  # fmt: skip

    def step(self, action):
        a = np.asarray(action, np.float32).reshape(-1)
        if a.shape != (40,) or not np.isfinite(a).all():
            raise ValueError("the action is 40 finite floats")
        if self.over:
            raise RuntimeError("the episode is over")
        self.client.next = a
        self.ctl.step()
        self.steps += 1
        sev = self.sev
        info = sev._extract_reward_info(self.env)
        self.success = info["raw_total"] >= 1.0
        det, _ = sev._detect_robot_fall(self.env, self.fall)
        self.streak = self.streak + 1 if det else 0
        self.fell = self.fell or self.streak >= int(self.fall["confirm_steps"])
        self.record_contacts()
        self.over = bool(self.success or self.fell or self.steps >= self.spec["max_steps"])
        return dict(over=self.over, success=self.success, fell=self.fell, steps=self.steps, state=self.state())

    # ---- readings

    def obj_pos(self):
        k = self.spec.get("obj_key")
        if not k:
            return None
        try:
            return self.env.scene[k].data.root_pos_w[0].detach().cpu().numpy().astype(float)
        except Exception:
            return None

    def record_contacts(self, thresh=1.0):
        sensor = self.env.scene.sensors.get("obj_contacts") if hasattr(self.env.scene, "sensors") else None
        if sensor is None:
            return
        fm = sensor.data.force_matrix_w_history  # (1, hist, bodies, filters, 3)
        if fm is None:
            return
        f = np.linalg.norm(fm[0].detach().cpu().numpy(), axis=-1).max(axis=0).sum(axis=-1)  # per sensed body
        names = list(sensor.body_names)
        for i in np.nonzero(f > thresh)[0]:
            self.contacts.setdefault(names[int(i)], []).append(self.steps)
        p = self.obj_pos()
        if p is not None:
            self.obj_max_move = max(self.obj_max_move, float(np.linalg.norm(p - self.obj0)))

    def state(self) -> dict:
        d = self.robot.data
        b = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        out = dict(t=self.steps * 0.02, root=b(d.root_state_w[0]), joint_pos=b(d.joint_pos[0]), joint_vel=b(d.joint_vel[0]),
                   hands={s: b(d.body_pos_w[0, self.side_wrist[s]]) for s in ("left", "right")},
                   hand_bodies={self.body_names[i]: b(d.body_pos_w[0, i]) for i in self.hand_ids})  # fmt: skip
        try:  # the 64-D policy state (heading-canonical root rot6d, canonical joint positions, velocities), as the provider builds it
            out["policy_state"] = self.provider._build_lerobot_vla_observation_state().astype(float)
        except Exception as e:  # noqa: BLE001
            out["policy_state_error"] = f"{type(e).__name__}: {e}"
        return out

    def rgbd(self):
        cam = self.env.scene["front_camera"]
        out = cam.data.output
        rgb = out["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
        depth = (
            out["distance_to_image_plane"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(rgb.shape[0], rgb.shape[1])
        )
        K = cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
        pw, qw = (
            cam.data.pos_w[0].detach().cpu().numpy(),
            cam.data.quat_w_ros[0].detach().cpu().numpy(),
        )  # ROS: x right, y down, z forward
        Tw = np.eye(4)
        Tw[:3, :3] = quat_to_mat(qw)
        Tw[:3, 3] = pw
        r = self.robot.data.root_state_w[0].detach().cpu().numpy().astype(float)
        Tr = np.eye(4)
        Tr[:3, :3] = quat_to_mat(r[3:7])
        Tr[:3, 3] = r[:3]
        return dict(rgb=rgb, depth=depth, K=K, T_body_cam=np.linalg.inv(Tr) @ Tw, t=self.steps * 0.02)

    def truth(self) -> dict:
        out = dict(task=self.key, root=self.state()["root"], t=self.steps * 0.02)
        scene = self.env.scene
        objs = {}
        for name in list(scene.keys()):
            ent = scene[name]
            data = getattr(ent, "data", None)
            if data is not None and hasattr(data, "root_pos_w") and name != "robot":
                objs[name] = np.concatenate(
                    [data.root_pos_w[0].detach().cpu().numpy(), data.root_quat_w[0].detach().cpu().numpy()]
                ).astype(float)
        out["objects"] = objs
        facts = {}
        try:
            if self.key == "pp_box":
                from tasks.g1_tasks.move_pickplace_box_g1_29dof_dex3_wholedoby.mdp import rewards as R

                facts["surfaces"] = [list(map(float, s)) for s in R._get_shelf_support_surfaces_world(self.env)[0]]
                facts["box_center"] = (
                    R._get_box_centers_world(self.env)[0].detach().cpu().numpy().astype(float).tolist()
                )
                facts["box_half"] = [float(x) for x in R._BOX_HALF_EXTENTS_M]
                tz, base = R._estimate_target_support_top_z(
                    self.env, 0, R._get_box_centers_world(self.env)[0], R._get_shelf_support_surfaces_world(self.env)[0]
                )
                facts["target_top_z"] = None if tz is None else float(tz)
                facts["base_top_z"] = None if base is None else float(base)
                # poses the benchmark's own check accepts: a box centred on each surface's rectangle at the layer height that
                # check derives for that spot (the benchmark's functions applied to a hypothetical box position)
                import torch

                surfs = R._get_shelf_support_surfaces_world(self.env)[0]
                half = float(R._BOX_HALF_EXTENTS_M[2])
                ok = []
                for i, sf in enumerate(surfs):
                    cx, cy = 0.5 * (sf[0] + sf[1]), 0.5 * (sf[2] + sf[3])
                    c0 = torch.tensor([cx, cy, 0.0], dtype=torch.float32)
                    t_top, _ = R._estimate_target_support_top_z(self.env, 0, c0, surfs)
                    c = torch.tensor([cx, cy, float(t_top) + half], dtype=torch.float32)
                    if R._is_box_on_support_surface(c, sf, target_support_top_z=float(t_top)):
                        ok.append(
                            dict(
                                surface=i,
                                center_xy=[float(cx), float(cy)],
                                rect=[float(v) for v in sf[:4]],
                                bottom_z=float(t_top),
                                center_z=float(t_top) + half,
                            )
                        )
                facts["accepted"] = ok
                if getattr(self, "_aa", None) is None:  # per episode (the shelf is placed at reset)
                    self._aa = self.aabbs(3)
                aa = self._aa
                facts["table"] = next((v for k, v in aa.items() if k.rsplit("/", 1)[-1].startswith("table")), None)
                facts["shelf"] = aa.get("/World/envs/env_0/Shelf")
        except Exception as e:  # noqa: BLE001 - reported in the reply, the caller decides
            facts["error"] = f"{type(e).__name__}: {e}"
        out["facts"] = facts
        return out

    def jac(self, side: str):
        """Position Jacobian of the wrist (world frame) over the arm's seven joints, and where the wrist is now."""
        r = self.robot
        bi = self.side_wrist[side]
        J = (
            r.root_physx_view.get_jacobians()[0].detach().cpu().numpy()
        )  # (bodies-1 or bodies, 6, dofs+6) for a floating base
        row = bi - 1 if J.shape[0] == len(self.body_names) - 1 else bi
        cols = [6 + j for j in self.arm_joint_ids[side]]
        names = [self.joint_names[j] for j in self.arm_joint_ids[side]]
        return dict(
            joint_ids=self.arm_joint_ids[side],
            canon_ids=[self.canon.index(n) for n in names],
            J=J[row, :3][:, cols].copy(),
            hand=self.state()["hands"][side],
            q=self.state()["joint_pos"][self.arm_joint_ids[side]],
        )

    def aabbs(self, depth: int = 2) -> dict:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        out = {}
        root = stage.GetPrimAtPath("/World/envs/env_0")

        def walk(prim, d):
            if d > depth:
                return
            for c in prim.GetChildren():
                if c.GetName() in ("Robot",):
                    continue
                r = cache.ComputeWorldBound(c).ComputeAlignedRange()
                if not r.IsEmpty():
                    out[str(c.GetPath())] = [list(r.GetMin()), list(r.GetMax())]
                walk(c, d + 1)

        walk(root, 1)
        return out

    def outcome(self) -> dict:
        return dict(
            success=self.success, fell=self.fell, steps=self.steps, contacts=self.contacts, obj_move=self.obj_max_move
        )

    def handle(self, msg):
        v = msg[0]
        if v == "hello":
            import platform

            return dict(python=platform.python_version(), tasks=sorted(TASKS), booted=self.key)
        if v == "reset":
            return self.reset(msg[1], int(msg[2]), int(msg[3]))
        if v == "step":
            return self.step(msg[1])
        if v == "state":
            return self.state()
        if v == "rgbd":
            return self.rgbd()
        if v == "truth":
            return self.truth()
        if v == "jac":
            return self.jac(msg[1])
        if v == "outcome":
            return self.outcome()
        if v == "worldcam":  # development: the benchmark's third-person camera, for a person to look at
            cam = self.env.scene["world_camera"]
            cam.update(0.0) if hasattr(cam, "update") else None
            return dict(rgb=cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8))
        if v == "aabbs":  # the static scene's prims under env_0 (name, world AABB): what is in the way
            return self.aabbs(msg[1] if len(msg) > 1 else 2)
        if v == "probe":  # development: what the scene holds
            sc = self.env.scene
            return dict(
                keys=list(sc.keys()), sensors=list(sc.sensors.keys()), bodies=self.body_names, joints=self.joint_names
            )
        raise ValueError(f"unknown verb {v!r}")


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sock")
    ap.add_argument("--root", required=True)
    ap.add_argument("--kit-gpu", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep", action="store_true", help="development: outlive the client and accept another one")
    a = ap.parse_args(argv)
    if not a.keep:
        die_with_parent()
    srv = Server(a.root, a.kit_gpu, a.device)
    with Listener(a.sock, family="AF_UNIX") as lis:
        conn = lis.accept()
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                if not a.keep:
                    return 0
                conn = lis.accept()
                continue
            if msg[0] == "close":
                conn.send(dict(ok=True))
                os._exit(0)  # Kit's own shutdown can hang; the bed has what it needs
            try:
                conn.send(srv.handle(msg))
            except Exception as e:  # noqa: BLE001 - the client re-raises it
                conn.send(dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:]))


if __name__ == "__main__":
    t0 = time.time()
    raise SystemExit(main())
