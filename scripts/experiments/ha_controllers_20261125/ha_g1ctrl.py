"""ScaleBFM and HEFT (LocoManiLab's g1_backend) as HumanoidArena controllers, 2026-11-25.

HumanoidArena (HA) ships one whole-body controller, SONIC, inside `SonicActionProvider`. This module adds the two
other motion-reference controllers we run in LocoManiLab (LML), on HA's own IsaacLab/PhysX G1, without touching HA's
files or the SONIC path:

* The controller is g1_backend's own control path, unchanged and imported in place from LML's checkout (`G1B`): the
  `g1_bridge.session.TeleopSession` that LML's benchmark host (`robocasa.g1_host.bridge.G1BridgeComposite`) drives, its
  generic tracking backend, `RefQposAdapter` (qpos frame -> FK on g1_backend's robot_fk.xml -> RealtimeMotionBuffer,
  sampled `delay_s` behind on the host clock) and sim2real's `BridgeTracking` (the ONNX policy, observations, PD gains).
  `IsaacG1Host` is the `TeleopHost` the session reads the robot through: HA's articulation instead of MuJoCo.
* What LML's host does around the session is reproduced in `G1BackendController` (one new session per episode as
  robosuite rebuilds the composite, `reset` + `vla_reset`, and on the first policy tick under a VLA action the one-shot
  yaw calibration on the live pose: `set_anchor_heading`, `push_standing_frame`, `calibrate_alignment`; then
  `vla_step_backend` every 20 ms of sim time).
* The joint targets drive HA's own actuators exactly as HA drives SONIC's (SonicActionProvider's default "position
  mode"): the controller's kp / kd written per joint into PhysX's implicit drives, `set_joint_position_target` held
  over the 4 physics steps (200 Hz) of each 50 Hz control step. Everything else about the body stays HA's (armature,
  effort limits, friction, solver, masses).
* `make_action_provider` builds an HA `ActionProvider` (SonicActionProvider's contract: get_action steps physics itself,
  RobotController(use_rl_action_mode=True) syncs rewards), so HA's control loop runs it unchanged.

Two things of g1_backend are switched off, both outside the control path: the GMR retargeting subprocess and its ZMQ
relay (operator teleop only; under a VLA action LML's host never reads them).

Joint orders (checked by name at construction, asserted again in `check_orders`):
  MuJoCo / Unitree order  G1_JOINT_NAMES: the session's q, q_target, kp, kd and the qpos frames' joints
  policy order            policy_joint_names (= SONIC's IsaacLab order): inside g1_backend only
  HA articulation order   robot.data.joint_names (29 body + 14 Dex3, PhysX's order): `IsaacG1Host.idx`
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

G1B = Path(os.environ.get("HA_G1_BACKEND", "/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend"))
CTRL_SITE = Path(os.environ.get("HA_CTRL_SITE", "/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena/ctrl_site"))
BACKENDS = ("scalebfm", "heft")


def setup_paths() -> None:
    """g1_backend and the packages it needs that HA's venv lacks (ctrl_site: mujoco 3.3.1, mjhub, any4hdmi,
    tensordict, loguru, tyro 0.9, termcolor; nothing in it shadows an HA package) go AFTER HA's own paths."""
    for p in (str(CTRL_SITE), str(G1B)):
        if p not in sys.path:
            sys.path.append(p)
    os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "1")


def kit_single_gpu(args) -> str | None:
    """Under CUDA_VISIBLE_DEVICES=<one card> (HA_PHYS_GPU = its physical index): Kit's renderer is Vulkan and still
    enumerates every card, so /renderer/activeGpu must name the PHYSICAL card, while CUDA (PhysX, torch) sees it as
    device 0. sim_eval_vla._configure_renderer_gpu derives both from --device cuda:0; this rewrites activeGpu only.
    Tested 2026-11-25: activeGpu=0 under CUDA_VISIBLE_DEVICES=1 -> "No device could be created"; activeGpu=1 -> the
    card is active as CUDA device 0 and PhysX steps."""
    phys = os.environ.get("HA_PHYS_GPU")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if phys is None or "," in vis or not vis.strip():
        return None
    toks = [t for t in (getattr(args, "kit_args", "") or "").split() if not t.startswith("--/renderer/activeGpu=")]
    toks.append(f"--/renderer/activeGpu={int(phys)}")
    args.kit_args = " ".join(toks)
    return args.kit_args


def _disable_gmr() -> None:
    import g1_bridge.gmr_utils.publisher_proc as pp
    import g1_bridge.gmr_utils.relay as relay

    class _NoRelay:
        def __init__(self, *a, **k):
            pass

        def latest(self):
            return None

        def default(self):
            return None

        def stop(self, timeout: float = 1.0):
            pass

    pp.start_gmr_if_available = lambda cfg: None
    relay.GMRRelay = _NoRelay


class IsaacG1Host:
    """g1_bridge's TeleopHost over HA's articulation (G1CompositeBase's readers, from PhysX instead of MuJoCo)."""

    def __init__(self, robot, clock_s):
        import torch
        from g1_bridge.g1_runtime import G1_JOINT_NAMES

        self.robot = robot
        self._clock_s = clock_s
        names = list(robot.data.joint_names)
        missing = [n for n in G1_JOINT_NAMES if n not in names]
        if missing:
            raise ValueError(f"HA robot lacks joints {missing}")
        self.joint_names_mj = tuple(G1_JOINT_NAMES)
        self.idx_list = [names.index(n) for n in G1_JOINT_NAMES]
        self.idx = torch.tensor(self.idx_list, dtype=torch.long, device=robot.data.joint_pos.device)
        self.body_names = list(robot.data.body_names)
        # what TeleopSession passes through to backends; none are used on the VLA path of heft / scalebfm
        self.sim = None
        self.robot_model = None
        self.qpos_addrs = None
        self.qvel_addrs = None
        self.hand_mapping = None

    def read_joint_state(self):
        d = self.robot.data
        q = d.joint_pos[0, self.idx].detach().cpu().numpy().astype(np.float32)
        dq = d.joint_vel[0, self.idx].detach().cpu().numpy().astype(np.float32)
        return q, dq

    def read_pelvis_state(self):
        """(wxyz, body-frame angular velocity) of the root link (pelvis): MuJoCo's free joint qpos[3:7] / qvel[3:6]."""
        d = self.robot.data
        quat = d.root_link_quat_w[0].detach().cpu().numpy().astype(np.float32)
        n = float(np.linalg.norm(quat))
        quat = quat / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        w = d.root_link_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        return quat, w

    def read_root_pos(self):
        return self.robot.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float32)

    def read_body_pose_w(self, body_name: str):
        i = self.body_names.index(body_name)
        d = self.robot.data
        return (d.body_link_pos_w[0, i].detach().cpu().numpy().astype(np.float32),
                d.body_link_quat_w[0, i].detach().cpu().numpy().astype(np.float32))  # fmt: skip

    def read_hand_state(self):
        return None

    def clock_ms(self) -> int:
        return int(round(self._clock_s() * 1000.0))


class G1BackendController:
    """LML's G1_BRIDGE control path (VLA branch of G1BridgeComposite.run_controller) on an IsaacLab articulation."""

    def __init__(self, backend: str, robot, inference_backend: str = "onnx-cpu", policy_config: str | None = None):
        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r} not in {BACKENDS}")
        setup_paths()
        import g1_bridge._cuda_order  # noqa: F401  (torch before sim2real)

        _disable_gmr()
        from g1_bridge.config import build_bridge_spec

        self.backend = backend
        self.robot = robot
        self.t_s = 0.0  # the host clock: sim seconds since the episode's reset (robosuite's sim.data.time)
        self.host = IsaacG1Host(robot, lambda: self.t_s)
        self.spec = build_bridge_spec(backend, "mock", policy_config=policy_config, inference_backend=inference_backend)
        self.session = None
        self._new_session()
        tr = self.session.tracking
        am = tr.action_manager
        if tuple(am.joint_names) != self.host.joint_names_mj:
            raise ValueError("g1_backend action manager joint order is not G1_JOINT_NAMES")
        self.kp = np.asarray(am.joint_kp_unitree, np.float32)  # MuJoCo order
        self.kd = np.asarray(am.joint_kd_unitree, np.float32)
        self.policy_config = str(tr.args.policy_config)
        self.delay_s = float(tr.motion_buffer.delay_s)
        self.settling = True
        self.last = None  # (q_target, kp, kd) of the last policy tick, MuJoCo order
        self.ticks = 0

    # -- per episode ----------------------------------------------------------------------------------------------
    def _new_session(self) -> None:
        from g1_bridge.session import TeleopSession

        self.session = TeleopSession(self.host)  # robosuite builds a new composite (and session) every env.reset()
        self.session.configure(dict(self.spec))  # the policy runtime comes from g1_bridge.session's process cache

    def episode_reset(self) -> None:
        """After HA's env reset: what robosuite's env.reset() does to the G1_BRIDGE composite (new session, reset,
        vla_reset, settling re-armed), with the host clock back at 0 (MuJoCo's sim.data.time on reset)."""
        self.t_s = 0.0
        self._new_session()
        self.session.reset()
        self.session.vla_reset()
        self.settling = True
        self.last = None
        self.ticks = 0

    # -- per 50 Hz control step -----------------------------------------------------------------------------------
    def policy_tick(self, vla_action: dict):
        """One policy step on the current state: G1BridgeComposite.run_controller's policy-due branch with a VLA
        action. Returns (q_target, kp, kd) in MuJoCo order."""
        from g1_bridge.backends import TickInput

        s = self.session
        qpos, qvel = self.host.read_joint_state()
        tick = TickInput(qpos=qpos, qvel=qvel, sim_time=self.t_s, mode=s.mode_manager.current, buttons=None, sample=None)
        if self.settling:  # SETTLE_UNDER_VLA = False: calibrate on the live pose and hand over in the same tick
            s.set_anchor_heading(self.host.read_pelvis_state()[0])
            s.push_standing_frame()
            s.calibrate_alignment()
            self.settling = False
        target_q, kps, kds = s.vla_step_backend(tick, vla_action)
        s.cache_command(target_q, kps, kds, self.t_s)
        self.last = (np.asarray(target_q, np.float32), np.asarray(kps, np.float32), np.asarray(kds, np.float32))
        self.ticks += 1
        return self.last

    @property
    def motion_buffer(self):
        return self.session.tracking.motion_buffer


def write_gains(robot, idx, kp, kd) -> None:
    """The controller's kp / kd into HA's implicit PhysX drives of the 29 body joints (as SonicActionProvider's
    position mode writes SONIC's)."""
    import torch

    dev = robot.data.joint_pos.device
    robot.write_joint_stiffness_to_sim(torch.as_tensor(kp, dtype=torch.float32, device=dev).unsqueeze(0), joint_ids=idx)
    robot.write_joint_damping_to_sim(torch.as_tensor(kd, dtype=torch.float32, device=dev).unsqueeze(0), joint_ids=idx)


def physx_gains(robot, idx_list) -> dict:
    """What PhysX actually holds for the given joints (not IsaacLab's cached buffers)."""
    v = robot.root_physx_view
    out = {}
    for key, fn in (("stiffness", "get_dof_stiffnesses"), ("damping", "get_dof_dampings"), ("armature", "get_dof_armatures"),
                    ("max_force", "get_dof_max_forces"), ("max_velocity", "get_dof_max_velocities"),
                    ("friction", "get_dof_friction_coefficients")):  # fmt: skip
        f = getattr(v, fn, None)
        if f is None:
            continue
        try:
            a = f()
            a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            out[key] = np.asarray(a[0])[idx_list].astype(float).tolist()
        except Exception as e:  # noqa: BLE001 -- reported, not fatal
            out[key] = f"error {type(e).__name__}: {e}"
    return out


class _ProviderBody:
    """HA ActionProvider (SonicActionProvider's contract) running ScaleBFM or HEFT. The reference comes in as LML's
    VLA action: `set_vla_action({ref_root_pos, ref_root_quat, ref_joint_pos})`, held until replaced; hands from
    `set_hands(left_closed, right_closed)` (HA's Dex3 open / close poses, as SONIC's hand binaries). Mixed into HA's
    ActionProvider by `make_action_provider` (importable only inside HA's tree)."""

    def _init(self, env, backend: str, inference_backend: str, policy_config: str | None):
        import torch

        self.env = env
        self.robot = robot = env.scene["robot"]
        self.ctrl = G1BackendController(backend, robot, inference_backend, policy_config)
        self.idx = self.ctrl.host.idx
        self.idx_list = self.ctrl.host.idx_list
        self.decimation = int(env.cfg.decimation)
        self.physics_dt = float(env.physics_dt)
        self.control_dt = self.physics_dt * self.decimation
        names = list(robot.data.joint_names)
        self.hand_idx = {}
        for side in ("left", "right"):
            hn = [f"{side}_hand_{f}_joint" for f in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
            self.hand_idx[side] = torch.tensor([names.index(n) for n in hn if n in names], dtype=torch.long, device=env.device)
        self.hand_target = {}
        self.set_hands(False, False)  # SonicActionProvider's initial hand targets: both open
        self.default_pos = robot.data.default_joint_pos[0].clone()
        self.vla_action = None
        self.gains_written = 0
        self.calls = 0
        self.t_policy = 0.0  # wall seconds in the controller (state read, policy, reference) and in physics
        self.t_phys = 0.0

    # -- HA provider protocol -------------------------------------------------------------------------------------
    def on_env_reset(self) -> None:
        """HA calls this after each env reset (sim_eval_vla._notify_action_provider_env_reset). Gains are written
        again every episode, in case HA's reset re-initialises the articulation from its actuator config."""
        self.ctrl.episode_reset()
        write_gains(self.robot, self.idx, self.ctrl.kp, self.ctrl.kd)
        self.gains_written += 1
        self.vla_action = None

    def set_vla_action(self, action: dict | None) -> None:
        self.vla_action = action

    def set_hands(self, left_closed: bool, right_closed: bool) -> None:
        import torch
        from pico_server.data_utils.params import DEFAULT_HAND_POSE  # as action_provider_sonic

        SONIC_HAND_POSE_ROBOT_NAME = "unitree_g1_with_hands"  # action_provider_sonic.SONIC_HAND_POSE_ROBOT_NAME

        for side, closed in (("left", left_closed), ("right", right_closed)):
            pose = DEFAULT_HAND_POSE[SONIC_HAND_POSE_ROBOT_NAME][side]["close" if closed else "open"]
            self.hand_target[side] = torch.as_tensor(np.asarray(pose, np.float32).reshape(-1), device=self.env.device)

    def get_action(self, env):
        import torch

        if self.vla_action is None:
            raise RuntimeError("g1_backend provider: no reference (set_vla_action) before the first step")
        t0 = time.perf_counter()
        q_target, _, _ = self.ctrl.policy_tick(self.vla_action)
        t1 = time.perf_counter()
        full = self.default_pos.clone()
        full[self.idx] = torch.as_tensor(q_target, dtype=torch.float32, device=full.device)
        for side in ("left", "right"):
            if self.hand_idx[side].numel():
                full[self.hand_idx[side]] = self.hand_target[side]
        robot = env.scene["robot"]
        for _ in range(self.decimation):  # as SonicActionProvider.get_action's position mode
            robot.set_joint_position_target(full)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        torch.cuda.synchronize()
        self.t_policy += t1 - t0
        self.t_phys += time.perf_counter() - t1
        self.ctrl.t_s += self.control_dt
        self.calls += 1
        return full

    def cleanup(self) -> None:
        pass


def make_action_provider(env, backend: str, inference_backend: str = "onnx-cpu", policy_config: str | None = None):
    """A G1BackendActionProvider: HA's ActionProvider with _ProviderBody's methods."""
    from action_provider.action_base import ActionProvider

    cls = type("G1BackendActionProvider", (_ProviderBody, ActionProvider), {})
    p = cls.__new__(cls)
    ActionProvider.__init__(p, f"g1_backend_{backend}")
    p._init(env, backend, inference_backend, policy_config)
    return p


def check_orders(host: IsaacG1Host, tracking) -> dict:
    """Name-level checks of the three joint orders (HA articulation, MuJoCo, policy) and of the yaml gains' mapping."""
    from sim2real.utils.strings import resolve_matching_names_values

    names_ha = list(host.robot.data.joint_names)
    mj = list(host.joint_names_mj)
    pol = list(tracking.policy_joint_names)
    cfg = tracking.policy_config
    ok_ha = all(names_ha[host.idx_list[i]] == mj[i] for i in range(29))
    # kp from the yaml resolved independently in policy order, then mapped by name to MuJoCo order
    ids, _, vals = resolve_matching_names_values(cfg["joint_kp"], pol, preserve_order=True, strict=False)
    kp_pol = np.zeros(29)
    kp_pol[ids] = vals
    kp_mj_by_name = np.array([kp_pol[pol.index(n)] for n in mj])
    am = tracking.action_manager
    return dict(
        ha_index_of_mujoco_joint=host.idx_list,
        ha_order_matches_by_name=bool(ok_ha),
        policy_order=pol,
        policy_order_is_permutation_of_mujoco=sorted(pol) == sorted(mj),
        policy_to_mujoco=[mj.index(n) for n in pol],
        controlled_joint_indices=list(map(int, tracking.controlled_joint_indices)),  # policy action i -> MuJoCo index
        controlled_matches_names=[mj[k] for k in tracking.controlled_joint_indices] == pol,
        kp_yaml_vs_action_manager_maxdiff=float(np.abs(kp_mj_by_name - am.joint_kp_unitree).max()),
    )


def rot_from_quat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def tilt_deg(quat_wxyz) -> float:
    R = rot_from_quat(quat_wxyz)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))))
