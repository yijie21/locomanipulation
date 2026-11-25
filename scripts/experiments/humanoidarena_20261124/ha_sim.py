"""HumanoidArena under its own controller (SONIC: GEAR-SONIC encoder/decoder ONNX, then PD; Isaac Sim 5.0, Isaac Lab
2.2) with our commands in place of the VLA: the shared start-up of humanoidarena_20261124's scripts.

Built as the install's probe ($R/scripts/probe_sonic.py) builds it, from the harness's own
script/eval_scripts/sonic/sim_eval_vla.py (its helpers imported, not copied): the same env creation, env-config YAML,
seeding, reset, SonicActionProvider (input_source=vla, semantic_v3 action), RobotController, reward and fall detector.
The only change is the documented hook: SonicActionProvider._fetch_lerobot_action_chunk (where the VLA chunk would be
fetched) returns our one 40-D frame per control step (Sim.cmd):

  [0:2]  root xy delta (m per control step) in the reference root's heading frame
  [2]    root z (m)
  [3:9]  root rot6d relative to the episode's initial heading (row layout; identity [1,0,0,1,0,0])
  [9:38] 29 absolute joint positions, SONIC/IsaacLab order (ORDER below; right shoulder pitch = frame[21])
  [38:40] hands (left, right): >= 0.5 closes the Dex3 hand (the provider's fixed close pose)

Run inside $R/env.sh, cwd = $ISAACLAB_ROOT (see run_ha.sh). An episode is stepped as sim_eval_vla._run_episode_once
steps it: controller.step(), then the reward (success when raw_total >= 1), then the fall detector (5 confirming steps),
then the step budget. Kit hangs on exceptions: callers end with os._exit (main_guard).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

R_ROOT = Path("/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena")
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", Path.cwd())).resolve()
SONIC_ROOT = Path(
    os.environ.get(
        "SONIC_POLICY_ROOT", ISAACLAB_ROOT.parent / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
    )
)

TASKS = {  # the official SONIC test configs and step budgets (script/eval_scripts/sonic/HOI_*/HSI_* launchers)
    "ppbox": dict(yaml="tasks/common_test_config/base_test/pp_box_sonic_test.yaml", max_steps=1450),
    "doubledesk": dict(yaml="tasks/common_test_config/base_test/doubledesk_sonic_test.yaml", max_steps=2000),
    "opendoor": dict(yaml="tasks/common_test_config/base_test/open_door_sonic_test.yaml", max_steps=1800),
}
OFFICIAL_SEEDS = (0, 1, 2)  # the test set: never run by these scripts (guard in check_seed)

ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip
ARM = {
    s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint", f"{s}_elbow_joint",
        f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint", f"{s}_wrist_yaw_joint"]
    for s in ("left", "right")
}  # fmt: skip
ARM_IDX = {s: [ORDER.index(n) for n in ARM[s]] for s in ARM}
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
UPPER = sorted([ORDER.index(n) for n in WAIST + ARM["left"] + ARM["right"]])  # what our commands own


def episode_seed(task_name: str, seed: int, repeat: int) -> int:
    """eval_vla_suite._derive_episode_seed: sha256("task|seed|repeat"), last 4 bytes, positive int32."""
    payload = f"{task_name}|{int(seed)}|{int(repeat)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big", signed=False) & 0x7FFFFFFF


def check_seed(seed: int) -> None:
    if int(seed) in OFFICIAL_SEEDS and os.environ.get("HA_ALLOW_OFFICIAL") != "registered":
        raise SystemExit(
            f"seed {seed} is an official test seed: refused (set HA_ALLOW_OFFICIAL=registered for the registered run)"
        )


def yaw_of(q_wxyz) -> float:
    w, x, y, z = (float(v) for v in q_wxyz)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def mat_of(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def quat_of(Rm) -> np.ndarray:
    Rm = np.asarray(Rm, float)
    w = np.sqrt(max(0.0, 1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - Rm[0, 0] + Rm[1, 1] - Rm[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - Rm[0, 0] - Rm[1, 1] + Rm[2, 2])) / 2
    x, y, z = (
        np.copysign(x, Rm[2, 1] - Rm[1, 2]),
        np.copysign(y, Rm[0, 2] - Rm[2, 0]),
        np.copysign(z, Rm[1, 0] - Rm[0, 1]),
    )
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot6d_yaw(a: float) -> np.ndarray:
    """vla_smpl_runtime.quat_to_rot6d_wxyz of a yaw: the rotation's first two columns, row-major."""
    c, s = np.cos(a), np.sin(a)
    return np.array([c, -s, s, c, 0.0, 0.0], np.float32)


def _load_sev():
    if str(ISAACLAB_ROOT) not in sys.path:
        sys.path.insert(0, str(ISAACLAB_ROOT))  # sim_eval_vla imports task_runtime_profiles before it fixes sys.path
    spec = importlib.util.spec_from_file_location(
        "sim_eval_vla", ISAACLAB_ROOT / "script" / "eval_scripts" / "sonic" / "sim_eval_vla.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # sets PROJECT_ROOT and sys.path like the harness; imports isaaclab.app only
    return mod


sev = _load_sev()


def build_parser():
    p = sev._build_parser()
    p.add_argument("--ha_task", choices=sorted(TASKS), required=True)
    p.add_argument("--no_walker", action="store_true", help="no planner: the legs hold SONIC's standing default")
    p.add_argument("--lead", type=int, default=8, help="frames the walker's reference runs ahead of the plan")
    return p


def finish_args(a, task: str):
    """The harness's own arguments for `task` (as run_vla_eval_parallel.sh passes them)."""
    import yaml

    spec = TASKS[task]
    a.env_config_yaml = str(ISAACLAB_ROOT / spec["yaml"])
    a.task = yaml.safe_load(open(a.env_config_yaml))["task_name"]
    a.max_steps = spec["max_steps"]
    a.sonic_encoder_path = a.sonic_encoder_path or str(SONIC_ROOT / "model_encoder.onnx")
    a.sonic_decoder_path = a.sonic_decoder_path or str(SONIC_ROOT / "model_decoder.onnx")
    return a


class Sim:
    """One Kit process, one task; episodes reset in place (persistent_sim, as the official eval runs them)."""

    def __init__(self, args):
        a = args
        self.a = a
        a.phys_gpu = str(sev._cuda_device_index(a.device) or 0)
        sev._install_interrupt_handlers()
        sev._ensure_unique_multi_image_shm_name(a)
        a.enable_cameras = True
        a.multi_gpu = False
        sev._append_unique_kit_arg(a, "--/renderer/multiGpu/enabled=False")
        sev._configure_renderer_gpu(a)
        sev._normalize_control_routing(a)
        sev.apply_task_runtime_profile(a)
        t0 = time.time()
        self.app = sev.AppLauncher(a).app

        import gymnasium as gym
        import tasks  # noqa: F401  (registers the gym ids)
        import torch
        from action_provider.action_provider_sonic import SONIC_DEFAULT_POS, SONIC_ISAACLAB_JOINT_ORDER
        from action_provider.create_action_provider import create_action_provider
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from layeredcontrol.robot_control_system import ControlConfig, RobotController
        from tasks.common_env_config import apply_env_config_yaml
        from tasks.common_runtime import apply_optional_runtime_augments

        assert list(SONIC_ISAACLAB_JOINT_ORDER) == ORDER
        self.torch = torch
        self.default = np.asarray(SONIC_DEFAULT_POS, np.float64).copy()
        self.env_cfg = env_cfg = parse_env_cfg(a.task, device=a.device, num_envs=1)
        env_cfg.env_name = a.task
        apply_env_config_yaml(env_cfg, a.env_config_yaml, task_name=a.task, route_name=a.gmt_backend or a.action_source)
        seed0 = int(a.episode_seed if a.episode_seed is not None else a.seed)
        sev._seed_runtime_rngs(seed0)
        sev._configure_episode_seed_state(env_cfg, seed0)
        self.env = env = gym.make(a.task, cfg=env_cfg).unwrapped
        sev._initialize_task_scene(env, env_cfg, a, apply_optional_runtime_augments)
        sev._reset_environment_for_episode(env, env_cfg, seed0)
        physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
        self.dt = float(physics_dt * env_cfg.decimation)
        self.provider = provider = create_action_provider(env, a)
        sev._disable_action_provider_internal_recording(provider)
        env.action_provider = provider
        provider._lerobot_http_client = None  # no LeRobot server: nothing may POST /reset or /infer
        self.controller = RobotController(
            env, ControlConfig(step_hz=int(round(1 / self.dt)), replay_mode=False, use_rl_action_mode=True)
        )
        self.controller.set_action_provider(provider)
        self.controller._step_interval = 0.0  # unpaced (the harness sleeps to 1x real time)
        self.robot = env.scene["robot"]
        self.sonic_idx = provider._sonic_idx
        self.body_names = list(self.robot.data.body_names)
        self.joint_names = list(self.robot.data.joint_names)
        self.calls = 0
        self.frame = None
        provider._fetch_lerobot_action_chunk = self._hook  # <-- the documented hook
        # the camera on its mount link, measured at start-up (fresh then): ROS convention, x right, y down, z forward
        cam = env.scene["front_camera"]
        env.sim.render()
        cam.update(dt=0.0, force_recompute=True)
        b = self.body("d435_link")
        Rb = mat_of(b[3:])
        self.cam_off_p = Rb.T @ (cam.data.pos_w[0].cpu().numpy().astype(np.float64) - b[:3])
        self.cam_off_R = Rb.T @ mat_of(cam.data.quat_w_ros[0].cpu().numpy())
        self.startup_s = time.time() - t0
        self.fall_detector = sev._build_fall_detector(env, a)
        self.walker = None
        if not getattr(a, "no_walker", False):
            import walker as W

            self.W = W
            dev = int(a.phys_gpu) if os.environ.get("CUDA_VISIBLE_DEVICES") is None else 0
            self.walker = W.Walker(ORDER, device_id=dev, lead=getattr(a, "lead", 0))
        self.cmd_reset()

    # ---- the command
    def cmd_reset(self):
        """The command. Without the walker: root_dxy, root_z, yaw_rel and joints are sent as they are. With it (the
        default), every step takes the planner's frame for `move` (walker.Movement) and splices `upper` (waist and
        both arms, UPPER joints) over its joints: root_dxy, root_z, rot6d and joints are then the walker's."""
        self.root_dxy = np.zeros(2)  # m per control step, in the reference root's heading frame
        self.root_z = 0.8
        self.yaw_rel = 0.0  # rad, relative to the episode's initial heading (no walker)
        self.rot6d = None  # set by the walker
        self.joints = self.default.copy()
        self.upper = self.default.copy()  # only UPPER entries are read
        self.hands = np.zeros(2)
        if self.walker is not None:
            self.move = self.W.Movement()

    def _hook(self):
        self.calls += 1
        r6 = self.rot6d if self.rot6d is not None else rot6d_yaw(self.yaw_rel)
        f = np.concatenate([self.root_dxy, [self.root_z], r6, self.joints, self.hands])
        self.frame = f.astype(np.float32)
        return self.frame.reshape(1, -1)

    # ---- episodes
    def reset(self, ep_seed: int):
        sev._reset_environment_for_episode(self.env, self.env_cfg, int(ep_seed))
        sev._reset_lerobot_runtime_seed(self.provider, int(ep_seed))  # no client: a no-op, as in the harness
        sev._notify_action_provider_env_reset(self.provider)
        self.cmd_reset()
        self.root_z = float(self.robot.data.root_state_w[0, 2].item())
        self.yaw0 = yaw_of(self.robot.data.root_state_w[0, 3:7].cpu().numpy())
        if self.walker is not None:
            self.walker.reset(self.q(), self.root()[3:7])
        self.controller.start()
        self.k = 0
        self.fall_streak = 0
        self.fell_at = None
        self.success_at = None
        self.reward = 0.0

    def step(self) -> None:
        """One control step and the harness's checks after it (success, fall)."""
        if self.walker is not None:
            w = self.walker.tick(self.move, self.q())
            self.wf = w
            self.joints = w["q"].copy()
            self.joints[UPPER] = self.upper[UPPER]
            self.root_z, self.rot6d, self.root_dxy = w["z"], self.W.rot6d_of(w["quat_rel"]), w["dxy"]
        c0 = self.calls
        self.controller.step()
        if self.calls != c0 + 1:  # get_action swallows exceptions and returns None: catch a silent hook failure
            raise RuntimeError(f"reference hook called {self.calls - c0} times in one control step")
        self.k += 1
        info = sev._extract_reward_info(self.env)
        self.reward = float(info["raw_total"])
        if self.reward >= 1.0 and self.success_at is None:
            self.success_at = self.k
        fell, _ = sev._detect_robot_fall(self.env, self.fall_detector)
        self.fall_streak = self.fall_streak + 1 if fell else 0
        if self.fell_at is None and self.fall_streak >= int(self.fall_detector["confirm_steps"]):
            self.fell_at = self.k

    # ---- the robot's own state
    def q(self) -> np.ndarray:
        return self.robot.data.joint_pos[0, self.sonic_idx].cpu().numpy().astype(np.float64)

    def root(self) -> np.ndarray:
        return self.robot.data.root_state_w[0, :7].cpu().numpy().astype(np.float64)

    def body(self, name: str) -> np.ndarray:
        i = self.body_names.index(name)
        p = self.robot.data.body_link_pos_w[0, i].cpu().numpy()
        q = self.robot.data.body_link_quat_w[0, i].cpu().numpy()
        return np.concatenate([p, q]).astype(np.float64)

    def pelvis(self):
        """The pelvis link's position and rotation in the world."""
        b = self.body("pelvis")
        return b[:3], mat_of(b[3:])

    def up(self) -> float:
        return float((-self.robot.data.projected_gravity_b[0, 2]).item())

    # ---- the look
    def look(self) -> dict:
        """One front-camera RGB-D frame (640 x 480) and where the camera sits on the pelvis now."""
        cam = self.env.scene["front_camera"]
        for _ in range(3):  # the annotators lag a frame or two behind a reset (2026-11-24: a look right after reset
            self.env.sim.render()  # showed the scene before it, the box missing)
        cam.update(dt=0.0, force_recompute=True)
        rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
        d = cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze(-1).astype(np.float32)
        K = cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64)
        # the camera's pose from the robot's own kinematics: its mount link (d435_link) now, times the mount offset
        # measured once at start-up (the camera's own pos_w is stale once the robot has moved: 2026-11-24 dev probe)
        b = self.body("d435_link")
        cp = b[:3] + mat_of(b[3:]) @ self.cam_off_p
        cR = mat_of(b[3:]) @ self.cam_off_R
        pp, pR = self.pelvis()
        return dict(rgb=rgb, depth=d, K=K, cam_p=pR.T @ (cp - pp), cam_R=pR.T @ cR, pelvis_p=pp, pelvis_R=pR,
                    q=self.q(), k=self.k)  # fmt: skip

    def close(self):
        try:
            self.controller.cleanup()
            self.env.close()
        finally:
            self.app.close()


def cam_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Each pixel's point in the camera's ROS frame (x right, y down, z forward), H x W x 3."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64)
    return np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], axis=2)


def main_guard(fn) -> None:
    try:
        code = fn()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)  # Kit's threads keep a failed process alive otherwise
