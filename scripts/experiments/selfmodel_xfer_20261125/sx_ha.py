"""selfmodel_xfer_20261125, combination 3: data3's reaches in HumanoidArena's body under its native SONIC (Isaac Sim 5.0 /
Isaac Lab 2.2, PhysX; HA's own SonicActionProvider, semantic_v3 40-D frames through the documented VLA hook), recorded
in data3's layout (sx_collect.py's record: data/ data_both/ arm.npz, d_this / d_prev). Runs in HA's runtime:

    sx_ha.sh <phase> [args]   (env.sh, cwd = isaaclab_twist2_g1, CUDA_VISIBLE_DEVICES = one card, HA_PHYS_GPU = it)
    phases: probe (conventions, scene clearance, reference round trip), lag (d3's ramps), drive (--shard K --shards P)

The body: ha_sim.Sim (humanoidarena_20261124: HA's sim_eval_vla start-up, the task's env-config YAML, episode resets in
place) with --no_walker: the whole 40-D frame is ours every control step (50 Hz; physics 200 Hz, decimation 4).
The reference is our IK's (data3's: d3_collect.plan_reach on robot_fk, protocol 2, xy shift onto the reference root;
data3's stance: LocoManiLab's default pose at the measured pelvis xy / heading, 0.8 m), as world qpos frames (pelvis,
wxyz, 29 joints), each turned into HA's frame as its runtime (vla_robot_current_local_runtime_v3) reads it back:
  [0:2] root xy delta in the target root's frame: (R_target^T (p_t - p_{t-1}))[:2] (the runtime rotates [dx, dy, dz']
        by R_target with dz' solved from the z change, so this inverts it exactly); p_{-1} = the root at the reset (the
        runtime's own start)
  [2]   root z;  [3:9] rot6d of conj(episode heading) * q_t (quat_to_rot6d_wxyz; the runtime composes the heading back)
  [9:38] the 29 joints in HA's order;  [38:40] hands open
and, every step, the runtime's integrated root (xy, z, attitude) is compared with the frame sent (run.json / logs).
State: HA's own robot data (pelvis link pose + c, joints, joint velocities, the root's body-frame angular velocity, the
wrist_yaw_link and torso poses): S, W, J, C, KP's hand point are HA's. The arm record (spheres, finger tips; arm.npz)
comes from a MuJoCo mirror (SIMPLE's G1 + Dex3 scene) posed at HA's measured pelvis, joints and fingers each row.
Fall: d3's 0.45 m (HA's own detector is logged, not used). The command time's reference: frames[f], the frame sent at
the command row (what d3_collect uses); the runtime's own integrated root is checked equal to it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W_REPO = Path(os.environ.get("SX_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
HA_EXP = Path(os.environ.get("SX_HA_EXP", W_REPO / "scripts/experiments/humanoidarena_20261124"))
sys.path.insert(0, str(HA_EXP))
import ha_sim  # noqa: E402  (loads sim_eval_vla: isaaclab.app only)

# ha_controllers_20261125/ha_g1ctrl.py (md5 72a7bc22, 2026-11-25), its two launch helpers copied here (that file is
# still being developed elsewhere): the paths HA's venv lacks (mujoco 3.3.1 ...) after HA's own, and the Vulkan renderer
# pinned to the physical card under CUDA_VISIBLE_DEVICES=<one card> ("activeGpu=0 under CUDA_VISIBLE_DEVICES=1 -> No
# device could be created; activeGpu=1 -> the card is active as CUDA device 0 and PhysX steps").
CTRL_SITE = Path(os.environ.get("HA_CTRL_SITE", "/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena/ctrl_site"))
G1B = Path(os.environ.get("HA_G1_BACKEND", "/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend"))


def setup_paths() -> None:
    for p in (str(CTRL_SITE), str(G1B)):
        if p not in sys.path:
            sys.path.append(p)
    os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "1")


def kit_single_gpu(args) -> None:
    phys, vis = os.environ.get("HA_PHYS_GPU"), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if phys is None or "," in vis or not vis.strip():
        return
    toks = [t for t in (getattr(args, "kit_args", "") or "").split() if not t.startswith("--/renderer/activeGpu=")]
    toks.append(f"--/renderer/activeGpu={int(phys)}")
    args.kit_args = " ".join(toks)

SCENE = W_REPO / "outputs/scene/simple_g1_sonic/scene_locomani.xml"
# ha_controllers_20261125's ScaleBFM / HEFT provider (ha_g1ctrl.py; the as-run copy is pinned by SX_HA_CTRL)
HA_CTRL = Path(os.environ.get("SX_HA_CTRL", "/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena/scripts/ha_ctrl"))
SX_FPS = 50
SONIC_DELAY_F = 6  # HA's SONIC: d3's LAG_MOVES in this body, median lag at 50 % = 6 frames (dev/ha_lag.log, 2026-11-25)


def parser():
    p = ha_sim.build_parser()
    p.add_argument("--phase", choices=["probe", "scene", "hold", "lag", "drive", "substep"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controller", choices=["sonic", "scalebfm", "heft"], default="sonic")
    p.add_argument("--delay-f", type=int, default=None, help="default: sonic SONIC_DELAY_F, g1 controllers the buffer's")
    p.add_argument("--stance", choices=["native", "lml"], default=None, help="default: sonic native, g1 controllers lml")
    p.add_argument("--resets", type=int, default=400)
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default="")
    p.add_argument("--sx-seed", type=int, default=79_000_000)
    p.add_argument("--sx-id0", type=int, default=40_000_000)
    return p


class HAHost:
    """d3_collect's host interface on HA's body (as sx_collect.SimpleHost: m, d, mj, c, pl, to_mj, qadr_mj, rq, pelvis,
    ref_delay_f, reset, start_frames, follow, step, sample, pelvis_pose)."""

    def __init__(self, sim, L, D3, delay_f: int, stance: str = "native"):
        import mujoco

        from action_provider.vla_smpl_runtime import quat_to_rot6d_wxyz
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.sim, self.L, self.D3, self.mj = sim, L, D3, mujoco
        self.r6 = lambda q: np.asarray(quat_to_rot6d_wxyz(np.asarray(q, np.float32).reshape(1, 4)), np.float64).reshape(6)
        rb = sim.robot
        names = list(rb.data.joint_names)
        bn = list(rb.data.body_names)
        self.q_idx = np.array([names.index(n) for n in L.SCALEBFM_ORDER])
        self.i_mj = np.array([names.index(n) for n in MUJOCO_JOINT_NAMES])
        self.to_mj = np.array([L.SCALEBFM_ORDER.index(n) for n in MUJOCO_JOINT_NAMES])
        self.mj_to_ha = np.array([list(MUJOCO_JOINT_NAMES).index(n) for n in ha_sim.ORDER])
        self.b_pelvis, self.b_torso = bn.index("pelvis"), bn.index("torso_link")
        self.b_wrist = {s: bn.index(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        fj = [f"{s}_hand_{f}_joint" for s in D3.SIDES for f in D3.FINGERS]
        self.i_fing = np.array([names.index(n) for n in fj])
        # the MuJoCo mirror (arm record only)
        self.m = m = load_mirror(mujoco, SCENE.read_text())
        self.d = mujoco.MjData(m)
        jad = lambda n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])  # noqa: E731
        self.qadr_mj = np.array([jad(n) for n in MUJOCO_JOINT_NAMES])
        self.fadr = np.array([jad(n) for n in fj])
        self.rq = jad("floating_base_joint")
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.pl = L.Planner()
        self.stance = stance
        self.q_default = np.asarray(sim.default, float)[[list(ha_sim.ORDER).index(n) for n in MUJOCO_JOINT_NAMES]]
        self.ref_delay_f = int(delay_f)
        # scene contact (revision 2): the task's own ContactSensor on every robot body (pp_box: contact_forces); self
        # collisions are off in HA's G1, so a force on any body but the feet (ankle links) is the scene (floor, table, box)
        cs = sim.env.scene["contact_forces"]
        self.cs = cs
        self.cs_keep = np.array([i for i, n in enumerate(cs.body_names) if "ankle" not in n])
        self.cs_names = [cs.body_names[i] for i in self.cs_keep]
        self.cs_body = np.array([bn.index(n) for n in self.cs_names])
        self.touch = {}
        self.c = np.zeros(3)
        self.frames, self.ep_seed = None, 0
        self.dev = dict(xy=0.0, z=0.0, rot_deg=0.0)
        self.reset()
        self.c, self.calib = self._calibrate()

    # -- HA's state ---------------------------------------------------------------------------------------------------
    def _pose(self, b):
        d = self.sim.robot.data
        p = d.body_link_pos_w[0, b].detach().cpu().numpy().astype(np.float64)
        q = d.body_link_quat_w[0, b].detach().cpu().numpy().astype(np.float64)
        q = q / np.linalg.norm(q)
        return p, self.L._qmat(q), q

    def _calibrate(self):
        """lml_host.HostDriver._calibrate on HA's bodies at the spawn stance: c from both wrists (positions, attitudes)
        and the torso, relative to robot_fk at the measured joints (HA's G1 is robot_fk's: c ~ 0, asserted < 2 mm)."""
        L = self.L
        pp, Rp, _ = self._pose(self.b_pelvis)
        q_mj = self.sim.robot.data.joint_pos[0, self.i_mj].detach().cpu().numpy().astype(np.float64)
        out = self.pl.fk_world(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), q_mj)
        cs, rot = [], []
        for s, b in self.b_wrist.items():
            pw, Rw, _ = self._pose(b)
            cs.append(Rp.T @ (pw - pp) - (out[s][0] - out[s][1] @ L.HAND_OFF))
            dR = (Rp.T @ Rw).T @ out[s][1]
            rot.append(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))))
        c = np.mean(cs, axis=0)
        cal = dict(c=c.tolist(), spread_m=max(float(np.linalg.norm(x - c)) for x in cs), wrist_rot_deg=max(rot))
        assert np.linalg.norm(c) < 2e-3 and cal["spread_m"] < 2e-3 and cal["wrist_rot_deg"] < 0.2, f"HA vs robot_fk: {cal}"
        return c, cal

    def pelvis_pose(self):
        p, R, _ = self._pose(self.b_pelvis)
        return p + R @ self.c, R

    def touch_now(self) -> tuple[float, int, np.ndarray]:
        """The largest contact force on a non-foot body now (N), that body (index into cs_names) and its world position."""
        F = self.cs.data.net_forces_w[0, self.cs_keep].detach().cpu().numpy().astype(np.float64)
        n = np.linalg.norm(F, axis=1)
        i = int(np.argmax(n))
        pos = self.sim.robot.data.body_link_pos_w[0, self.cs_body[i]].detach().cpu().numpy().astype(np.float64)
        return float(n[i]), i, pos

    def sample(self) -> dict:
        d = self.sim.robot.data
        p, R = self.pelvis_pose()
        _, _, qp = self._pose(self.b_pelvis)
        jp = d.joint_pos[0].detach().cpu().numpy().astype(np.float64)
        jv = d.joint_vel[0].detach().cpu().numpy().astype(np.float64)
        x = dict(root=np.r_[p, qp], q=jp[self.q_idx], dq=jv[self.q_idx],
                 w=d.root_link_ang_vel_b[0].detach().cpu().numpy().astype(np.float64),
                 wr={s: self._pose(b)[:2] for s, b in self.b_wrist.items()}, torso=self._pose(self.b_torso)[:2])  # fmt: skip
        md = self.d  # the mirror at this state (free joint = HA's pelvis link; the arm record's spheres and tips)
        md.qpos[self.rq : self.rq + 3] = p - R @ self.c
        md.qpos[self.rq + 3 : self.rq + 7] = qp
        md.qpos[self.qadr_mj] = jp[self.i_mj]
        md.qpos[self.fadr] = jp[self.i_fing]
        self.mj.mj_kinematics(self.m, md)
        return x

    # -- the reference --------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.sim.reset(int(self.ep_seed))
        self.frames, self.touch = None, {}
        rs = self.sim.robot.data.root_state_w[0, :7].detach().cpu().numpy().astype(np.float64)
        self.last_p = rs[:3].copy()  # the runtime's own start: the root xy at the reset
        self.last_z = None
        rt = self.sim.provider._lerobot_vla_runtime
        h0 = getattr(rt, "_episode_ref_to_world_heading_quat_wxyz", None)
        yaw0 = ha_sim.yaw_of(h0) if h0 is not None else ha_sim.yaw_of(rs[3:7])
        self.head_inv = np.array([math.cos(yaw0 / 2), 0.0, 0.0, -math.sin(yaw0 / 2)])
        self.yaw0 = yaw0

    def start_frames(self) -> np.ndarray:
        """stance native (default): HA's own standing reference, SONIC's default pose (SONIC_DEFAULT_POS, what HA's
        reset spawns and its probe holds) at the measured pelvis (xy, height, heading); lml: data3's (LocoManiLab's
        default pose at 0.8 m), which SONIC here does not hold still (dev hold test: 8 cm xy drift, hands 3-5 cm/s at
        4-6 s, against 1.5 cm and 0.5-2 cm/s for the native stance)."""
        L = self.L
        x = self.sample()
        yaw = L.yaw_of(L._qmat(x["root"][3:]))
        qy = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        if self.stance == "lml":
            start = (np.r_[x["root"][:2], self.pl.start_pose()[0][2]], qy, L.DEFAULT_Q.copy())
        else:
            start = (x["root"][:3].copy(), qy, self.q_default.copy())
        return L.hold(start, L.START_HOLD_S + L.MARGIN_S + L.PRE_S)

    def follow(self, frames: np.ndarray, repin_at: int | None = None) -> None:
        self.frames = np.asarray(frames, np.float64)

    @property
    def ha_fell_at(self):  # HA's own fall detector (logged; d3's 0.45 m rule decides)
        return self.sim.fell_at

    def frame40(self, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = self.L
        R = L._qmat(fr[3:7])
        p = fr[:3] - R @ self.c
        dz = 0.0 if self.last_z is None else p[2] - self.last_z
        loc = R.T @ np.r_[p[:2] - self.last_p[:2], dz]
        qrel = _qmul(self.head_inv, fr[3:7] / np.linalg.norm(fr[3:7]))
        return np.r_[loc[:2], p[2]], self.r6(qrel), p

    def step(self, f: int) -> bool:
        fr = self.frames[min(f, len(self.frames) - 1)]
        head, r6, p = self.frame40(fr)
        s = self.sim
        s.root_dxy, s.root_z, s.rot6d = head[:2].copy(), float(head[2]), r6.astype(np.float32)
        s.joints = fr[7:][self.mj_to_ha].copy()
        s.hands = np.zeros(2)
        s.step()
        self.touch[f] = self.touch_now()
        self.last_p, self.last_z = p, p[2]
        rt = s.provider._lerobot_vla_runtime  # what SONIC was told, integrated by HA's runtime
        bxy, bz, bq = getattr(rt, "_body_xy_world", None), getattr(rt, "_body_z_world", None), getattr(rt, "_prev_target_root_quat_wxyz", None)
        if bxy is not None:
            self.dev["xy"] = max(self.dev["xy"], float(np.linalg.norm(np.asarray(bxy, float) - p[:2])))
        if bz is not None:
            self.dev["z"] = max(self.dev["z"], abs(float(bz) - p[2]))
        if bq is not None:
            q = np.asarray(bq, float)
            self.dev["rot_deg"] = max(self.dev["rot_deg"], math.degrees(2 * math.acos(min(1.0, abs(float(q @ fr[3:7]) / np.linalg.norm(q) / np.linalg.norm(fr[3:7]))))))
        return True


class HAG1Host(HAHost):
    """d3_collect's host on HA's body under g1_backend's ScaleBFM or HEFT: ha_controllers_20261125's provider
    (ha_g1ctrl.make_action_provider: LocoManiLab's G1_BRIDGE control path on HA's articulation, the controller's kp / kd
    written into PhysX every episode), stepped by HA's RobotController as probe_ctrl.py does. The reference goes in as
    data3 fed LocoManiLab (d3_collect.run_reset with lml_host.HostDriver): the world qpos frame pushed as the VLA action
    every control step, frame 0 as it is, later frames through the motion buffer's episode transform
    (lml_host.HostDriver.to_frame); the settle window's delay is the buffer's (round(delay_s * 50), d3_collect.ref_delay_f).
    Stance: data3's (LocoManiLab's default pose at 0.8 m, the measured xy and heading: lml_host.HostDriver.start_frames).
    The SONIC provider ha_sim.Sim builds stays idle (never stepped, never notified)."""

    def __init__(self, sim, L, D3, backend: str, stance: str = "lml", delay_f: int | None = None):
        from layeredcontrol.robot_control_system import ControlConfig, RobotController

        if str(HA_CTRL) not in sys.path:
            sys.path.insert(0, str(HA_CTRL))
        import ha_g1ctrl as HC

        import lml_host as LH

        self.HC, self.LH, self.backend = HC, LH, backend
        env = sim.env
        self.prov = HC.make_action_provider(env, backend)
        env.action_provider = self.prov
        self.ctl = RobotController(env, ControlConfig(step_hz=int(round(1 / sim.dt)), replay_mode=False, use_rl_action_mode=True))
        self.ctl.set_action_provider(self.prov)
        self.ctl._step_interval = 0.0  # unpaced
        self.buf, self.k, self.fall_streak, self.fell_at = None, 0, 0, None
        super().__init__(sim, L, D3, 0, stance)
        self.ref_delay_f = int(round(float(self.prov.ctrl.delay_s) * SX_FPS)) if delay_f is None else int(delay_f)

    def transform(self):  # lml_host.HostDriver's (the buffer's episode transform), used by its to_frame
        return self.LH.HostDriver.transform(self)

    def reset(self) -> None:
        sev, s = ha_sim.sev, self.sim
        sev._reset_environment_for_episode(s.env, s.env_cfg, int(self.ep_seed))
        sev._notify_action_provider_env_reset(self.prov)  # -> on_env_reset: new session, the controller's gains written
        self.ctl.start()
        self.frames, self.buf, self.k, self.fall_streak, self.fell_at, self.touch = None, None, 0, 0, None, {}
        rs = s.robot.data.root_state_w[0, :7].detach().cpu().numpy().astype(np.float64)
        self.yaw0 = ha_sim.yaw_of(rs[3:7])

    def step(self, f: int) -> bool:
        fr = self.frames[min(f, len(self.frames) - 1)]
        fp = self.LH.HostDriver.to_frame(self, fr) if self.buf is not None else np.asarray(fr, np.float64)
        self.prov.set_vla_action(dict(ref_root_pos=fp[:3].astype(np.float32), ref_root_quat=fp[3:7].astype(np.float32),
                                      ref_joint_pos=fp[7:].astype(np.float32)))  # fmt: skip
        n0 = self.prov.calls
        self.ctl.step()
        if self.prov.calls != n0 + 1:
            raise RuntimeError("the g1 provider was not stepped by RobotController")
        if self.buf is None:
            self.buf = self.prov.ctrl.motion_buffer  # after the first payload: the episode transform is set
            if self.prov.ctrl.settling:
                raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        self.k += 1
        self.touch[f] = self.touch_now()
        fell, _ = ha_sim.sev._detect_robot_fall(self.sim.env, self.sim.fall_detector)  # HA's own detector: logged only
        self.fall_streak = self.fall_streak + 1 if fell else 0
        if self.fell_at is None and self.fall_streak >= int(self.sim.fall_detector["confirm_steps"]):
            self.fell_at = self.k
        return True

    @property
    def ha_fell_at(self):
        return self.fell_at



def add_touch(h, segs: list[dict], reaches: list[dict]) -> dict:
    """Scene contact per reach (revision 2): the largest non-foot contact force from the command to the end label (the
    last frame driven when there is none), on the segment as touch_n (N) and touch_z (that body's height, m: near 0 is
    the floor); the reset's log gets them with the body's name and world position, and the stance's (frames before the
    first command) as the baseline."""
    T = h.touch
    last = max(T) if T else -1

    def peak(f0, f1):
        fr = [f for f in range(f0, f1 + 1) if f in T]
        if not fr:
            return dict(n=0.0, body=None, pos=None)
        f = max(fr, key=lambda k: T[k][0])
        n, i, pos = T[f]
        return dict(n=n, body=h.cs_names[i], pos=np.round(pos, 3).tolist(), f=f)

    out = dict(stance=peak(0, (reaches[0]["f_cmd"] - 1) if reaches else last), reach=[])
    for rc in reaches:
        f1 = rc["f_label"] if rc.get("f_label") is not None else last
        out["reach"].append(peak(int(rc["f_cmd"]), int(f1)))
    for sg in segs:
        t = out["reach"][sg["chain"]]
        sg["touch_n"] = np.float32(t["n"])
        sg["touch_z"] = np.float32(t["pos"][2] if t["pos"] is not None else np.nan)
    return out


def load_mirror(mujoco, text: str):
    """The scene under HA's older mujoco (ctrl_site 3.3.1): every attribute its schema rejects is dropped from that
    element (newer render-only attributes: texture colorspace, light type, ...; meshdir is absolute), up to 20."""
    import re

    dropped = []
    for _ in range(20):
        try:
            m = mujoco.MjModel.from_xml_string(text)
            if dropped:
                print(f"mirror scene: dropped attributes {dropped} for mujoco {mujoco.__version__}", flush=True)
            return m
        except ValueError as e:
            g = re.search(r"unrecognized attribute: '([^']+)'\s*Element '([^']+)'", str(e))
            if not g:
                raise
            att, el = g.group(1), g.group(2)
            dropped.append(f"{el}.{att}")
            text = re.sub(rf'(<{el}\b[^>]*?)\s+{att}="[^"]*"', r"\1", text)
    raise RuntimeError(f"mirror scene does not load: {dropped}")


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])  # fmt: skip


def scene_clearance(sim, p0) -> dict:
    """The task's objects (env.scene entries with a root state, not the robot) and their xy distance to the pelvis."""
    out = {}
    env = sim.env
    for k in env.scene.keys():
        if k == "robot":
            continue
        try:
            rs = env.scene[k].data.root_state_w[0, :3].detach().cpu().numpy().astype(np.float64)
        except Exception:  # noqa: BLE001  (sensors, terrain)
            continue
        out[k] = dict(pos=rs.round(3).tolist(), xy_dist=float(np.linalg.norm(rs[:2] - p0[:2])))
    return out


def run(a) -> int:
    _orig = ha_sim.sev._configure_renderer_gpu

    def _cfg(args):  # the renderer on the physical card under CUDA_VISIBLE_DEVICES=<one card>
        _orig(args)
        kit_single_gpu(args)

    ha_sim.sev._configure_renderer_gpu = _cfg
    a = ha_sim.finish_args(a, a.ha_task)
    a.no_walker = True
    ha_sim.check_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    sim = ha_sim.Sim(a)
    if os.environ.get("SX_NORENDER") == "1":  # no camera render / observation refresh after each control step (the
        # provider renders whenever its one-frame chunk queue is empty, i.e. every step; nothing here reads a camera)
        sim.provider._should_refresh_lerobot_visuals_next_step = lambda: False
    setup_paths()  # mujoco 3.3.1 etc. after HA's own paths
    sys.path.insert(0, str(W_REPO / "src"))
    sys.path.insert(0, str(HERE))
    import sx_collect as SX  # d3_collect (frozen, next to this file's parent) and its helpers

    D3, L = SX.D3, SX.L
    if a.controller == "sonic":
        h = HAHost(sim, L, D3, SONIC_DELAY_F if a.delay_f is None else a.delay_f, a.stance or "native")
    else:
        h = HAG1Host(sim, L, D3, a.controller, a.stance or "lml", a.delay_f)
    h.ep_seed = a.sx_id0
    rig = SX.SRig(h)
    h.sample()
    tag = f"sx_ha {a.controller} {a.ha_task} {a.phase}"
    print(f"{tag}: up in {sim.startup_s:.0f} s; dt {sim.dt:.3f} s; c {np.round(h.c, 6).tolist()} {h.calib}; root check "
          f"{rig.check_root():.1e} m", flush=True)  # fmt: skip
    if a.phase == "scene":  # dev: every collision prim of the stage near the robot (world AABBs, pelvis heading frame)
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        h.reset()
        x = h.sample()
        p0, Rh = x["root"][:3], L.rot_z(h.yaw0)
        rcfg = sim.env_cfg.scene.robot
        print(f"{tag}: robot init_state pos {getattr(rcfg.init_state, 'pos', None)} rot {getattr(rcfg.init_state, 'rot', None)}; "
              f"pelvis {np.round(p0, 3).tolist()} heading {math.degrees(h.yaw0):.1f} deg; robot prim {rcfg.prim_path}", flush=True)
        ev = getattr(sim.env_cfg, "events", None)
        if ev is not None:
            for k, v in vars(ev).items():
                if v is not None and ("root" in k or "robot" in k or "pose" in k):
                    print(f"{tag}: event {k}: {getattr(v, 'func', v)} params {getattr(v, 'params', None)}", flush=True)
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        out = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/Robot" in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if r.IsEmpty():
                continue
            mn, mx = np.array(r.GetMin(), float), np.array(r.GetMax(), float)
            cs = np.array([[a, b] for a in (mn[0], mx[0]) for b in (mn[1], mx[1])]) - p0[:2]
            ch = cs @ Rh[:2, :2]  # corners in the heading frame
            q = np.clip(np.zeros(2), mn[:2] - p0[:2], mx[:2] - p0[:2])
            dxy = float(np.linalg.norm(q))
            if dxy < 1.6 and mx[2] > 0.02:
                out.append((dxy, path, ch[:, 0].min(), ch[:, 0].max(), ch[:, 1].min(), ch[:, 1].max(), mn[2], mx[2]))
        for o in sorted(out)[:40]:
            print(f"  {o[1]}: xy dist {o[0]:.2f} m; heading frame x {o[2]:.2f}..{o[3]:.2f}, y {o[4]:.2f}..{o[5]:.2f}, z {o[6]:.2f}..{o[7]:.2f}", flush=True)
        return 0
    if a.phase == "hold":  # dev: how still each stance holds (0.1 s speeds of pelvis / hands), and the runtime's root
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        rt = sim.provider._lerobot_vla_runtime
        for stance in ("lml", "native"):
            h.reset()
            print(f"{tag}: runtime attributes {sorted(k for k in vars(rt) if 'body' in k or 'quat' in k)}", flush=True)
            if stance == "lml":
                fr = h.start_frames()[:1]
            else:  # SONIC's default pose (HA's standing reference) at the reset root
                x = h.sample()
                q = np.asarray(sim.default, float)[[list(ha_sim.ORDER).index(n) for n in MUJOCO_JOINT_NAMES]]
                fr = np.r_[x["root"][:3], x["root"][3:], q][None]
            fr = np.repeat(fr, 300, 0)
            h.follow(fr)
            P = []
            for f in range(len(fr)):
                h.step(f)
                x = h.sample()
                P.append(np.stack([x["root"][:3]] + [x["wr"][s][0] + x["wr"][s][1] @ L.HAND_OFF for s in ("right", "left")]))
                if f in (0, 1, 50, 299):
                    print(f"  {stance} f {f}: runtime body_xy {getattr(rt, '_body_xy_world', None)}, z {getattr(rt, '_body_z_world', None)}, "
                          f"frame {np.round(fr[f, :3] - L._qmat(fr[f, 3:7]) @ h.c, 4).tolist()}; robot {np.round(x['root'][:3], 4).tolist()}", flush=True)
            P = np.array(P)
            V = 100 * np.linalg.norm(P[5:] - P[:-5], axis=2) / 0.1
            for f0 in range(0, len(V), 25):
                print(f"  {stance} {f0 / 50:.1f}-{(f0 + 25) / 50:.1f} s: max speed pelvis {V[f0:f0 + 25, 0].max():.1f} right "
                      f"{V[f0:f0 + 25, 1].max():.1f} left {V[f0:f0 + 25, 2].max():.1f} cm/s; pelvis z {P[f0, 0, 2]:.3f}, xy from start "
                      f"{100 * np.linalg.norm(P[f0, 0, :2] - P[0, 0, :2]):.1f} cm", flush=True)  # fmt: skip
        return 0
    if a.phase == "substep":  # diagnostic (11-26, not a registered metric): joint velocity readout at every physics substep
        # of a 3 s stance hold, the stance and gains of the drive (reset as drive resets): robot.data's joint_pos / joint_vel
        # (all 43 DOFs incl. Dex3) after every substep's scene.update, against what sample() records and the controller reads.
        rob = sim.robot
        names = list(rob.data.joint_names)
        pv = rob.root_physx_view
        fing = [i for i, nm in enumerate(names) if "hand" in nm]
        JN = list(L.SCALEBFM_ORDER)
        rec = []
        scene = sim.env.scene  # the providers' substep loops: set target, write, sim.step, scene.update(physics_dt)
        orig_upd = scene.update

        def upd_rec(*args, **kw):
            out = orig_upd(*args, **kw)
            rec.append((rob.data.joint_pos[0].detach().cpu().numpy().astype(np.float64).copy(),
                        rob.data.joint_vel[0].detach().cpu().numpy().astype(np.float64).copy()))
            return out

        scene.update = upd_rec
        h.reset()
        show = [17, 18, 13, 14, 9, 10, 19, 21]  # SCALEBFM_ORDER (S layout): ankle roll, ankle pitch, knee, left shoulder yaw, left elbow
        print(f"{tag}: joint properties after the reset (PhysX view, joints {show}: {[JN[j] for j in show]})", flush=True)
        for nm in ("get_dof_armatures", "get_dof_friction_coefficients", "get_dof_stiffnesses", "get_dof_dampings",
                   "get_dof_max_velocities", "get_dof_max_forces", "get_dof_limits"):
            fn = getattr(pv, nm, None)
            if fn is None:
                print(f"  {nm}: not in this PhysX view", flush=True)
                continue
            try:
                v = fn()[0].detach().cpu().numpy().astype(np.float64)
                print(f"  {nm}: {np.round(v[[int(h.q_idx[j]) for j in show]], 4).tolist()}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {nm}: {type(e).__name__} {e}", flush=True)
        for nm in ("joint_armature", "joint_friction_coeff", "joint_friction", "joint_damping", "joint_stiffness"):
            v = getattr(rob.data, nm, None)
            if v is not None:
                v = v[0].detach().cpu().numpy().astype(np.float64)
                print(f"  data.{nm}: {np.round(v[[int(h.q_idx[j]) for j in show]], 4).tolist()}", flush=True)
        fr = h.start_frames()
        fr = np.concatenate([fr, np.repeat(fr[-1:], max(0, 150 - len(fr)), 0)])[:150]
        h.follow(fr)
        rec.clear()
        ctl = []
        hc = getattr(getattr(getattr(h, "prov", None), "ctrl", None), "host", None)
        for f in range(len(fr)):
            n0 = len(rec)
            h.step(f)
            x = h.sample()
            row = dict(n_sub=len(rec) - n0, dq_rec=np.asarray(x["dq"], np.float64).copy(),
                       dq_data=rob.data.joint_vel[0].detach().cpu().numpy().astype(np.float64)[h.q_idx],
                       raw=rec[-1][1][h.q_idx].copy() if len(rec) > n0 else None)
            if hc is not None:  # read_joint_state is in the host's joint_names_mj order: into S's (SCALEBFM_ORDER)
                cq = np.asarray(hc.read_joint_state()[1], np.float64)
                row["dq_ctrl"] = cq[[list(hc.joint_names_mj).index(n) for n in JN]]
            ctl.append(row)
        scene.update = orig_upd
        Qs, Vs = np.array([r[0] for r in rec]), np.array([r[1] for r in rec])
        pdt = float(getattr(sim.env, "physics_dt", sim.dt / 4))
        ns = [r["n_sub"] for r in ctl]
        print(f"{tag}: {len(rec)} substeps recorded over {len(ctl)} control steps (per step {min(ns)}-{max(ns)}), physics dt {pdt:.4f} s", flush=True)
        if not rec:
            return 1
        rows = [r for r in ctl if r["raw"] is not None]
        d1 = max(np.abs(r["dq_rec"] - r["raw"]).max() for r in rows)
        d2 = max(np.abs(r["dq_rec"] - r["dq_data"]).max() for r in rows)
        d3 = max(np.abs(r["dq_rec"] - r["dq_ctrl"]).max() for r in rows) if hc is not None else float("nan")
        print(f"{tag}: recorded dq vs joint_vel after the step's last substep update max |d| {d1:.2e}; vs robot.data.joint_vel "
              f"{d2:.2e}; vs the controller's read_joint_state {d3:.2e} rad/s (nan: no g1 controller host)", flush=True)
        sk = slice(len(Vs) // 6, None)  # from 0.5 s on
        for j in [17, 18, 13, 14, 9, 10, 19, 21]:
            i = int(h.q_idx[j])
            v, q = Vs[sk, i], Qs[sk, i]
            fd = np.diff(q) / pdt
            nz = (np.abs(v[1:]) > 1e-6) & (np.abs(v[:-1]) > 1e-6)
            flips = float(np.mean(np.sign(v[1:][nz]) != np.sign(v[:-1][nz]))) if nz.any() else float("nan")
            r_fd = float(np.corrcoef(v[1:], fd)[0, 1]) if fd.std() > 0 and v.std() > 0 else float("nan")
            ctl_v = np.array([r["dq_rec"][j] for r in rows])[len(rows) // 6 :]
            cf = [float(np.corrcoef(v, Vs[sk, k])[0, 1]) if Vs[sk, k].std() > 0 and v.std() > 0 else 0.0 for k in fing]
            kb = int(np.nanargmax(np.abs(cf))) if fing else -1
            rel = min(float(np.median(np.abs(v - Vs[sk, k])) / max(np.median(np.abs(v)), 1e-9)) for k in fing) if fing else float("nan")
            print(f"  joint {j:2d} {JN[j]}: substep |dq| median {np.median(np.abs(v)):.3f} rad/s; (i) sign flips between consecutive "
                  f"substeps {100 * flips:.0f} %; (ii) integral of dq {np.sum(v[1:]) * pdt:+.4f} rad vs q change {q[-1] - q[0]:+.4f} rad, "
                  f"q substep std {q.std():.5f} rad, corr(dq, substep q difference) {r_fd:+.2f}; (iii) best Dex3 match "
                  f"{names[fing[kb]] if fing else '-'} r {cf[kb] if fing else float('nan'):+.2f}, closest Dex3 dq median |difference| / median |dq| {rel:.2f}; control-step samples median {np.median(ctl_v):+.3f}, "
                  f"positive {100 * np.mean(ctl_v > 0):.0f} %", flush=True)  # fmt: skip
        print(f"  first 12 substeps from 0.5 s, joint 18 dq: {np.round(Vs[sk, int(h.q_idx[18])][:12], 3).tolist()}", flush=True)
        print(f"  first 12 substeps from 0.5 s, joint 18 q x1e3: {np.round(1e3 * Qs[sk, int(h.q_idx[18])][:12], 4).tolist()}", flush=True)
        np.savez(a.out / f"substep_{a.controller}.npz", q=Qs, dq=Vs, names=np.array(names), q_idx=h.q_idx, physics_dt=pdt,
                 dq_rec=np.array([r["dq_rec"] for r in ctl]), n_sub=np.array(ns))  # fmt: skip
        return 0
    if a.phase == "probe":
        h.reset()
        x = h.sample()
        print(f"{tag}: scene objects vs the pelvis at {np.round(x['root'][:3], 3).tolist()} (heading {math.degrees(h.yaw0):.1f} deg): "
              f"{json.dumps(scene_clearance(sim, x['root'][:3]))}", flush=True)
        if os.environ.get("ENABLE_DEPTH") == "1":  # what the front camera sees inside the reach table's box
            lk = sim.look()
            P = ha_sim.cam_points(lk["depth"], lk["K"])[::3, ::3]
            ok = np.isfinite(P[..., 2]) & (P[..., 2] > 0.05) & (P[..., 2] < 5.0)
            pw = (lk["pelvis_R"] @ (lk["cam_R"] @ P[ok].T + lk["cam_p"][:, None])).T + lk["pelvis_p"]  # world
            Rh = L.rot_z(h.yaw0)
            ph = (pw - x["root"][:3]) @ Rh  # heading frame of the pelvis
            body = sim.robot.data.body_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
            near = np.min(np.linalg.norm(pw[:, None, :] - body[None, :, :], axis=2), axis=1) < 0.12
            zw = pw[:, 2]
            box = (ph[:, 0] > -0.1) & (ph[:, 0] < 0.95) & (ph[:, 1] > -0.85) & (ph[:, 1] < 0.85) & (zw > 0.05) & (zw < 1.6) & ~near
            print(f"{tag}: camera points in the reach box (heading frame x -0.1..0.95, |y| < 0.85, z 0.05-1.6 m, not the "
                  f"robot): {int(box.sum())} of {int(ok.sum())}; nearest x {ph[box, 0].min() if box.any() else float('nan'):.2f} m", flush=True)
            try:
                from PIL import Image

                Image.fromarray(lk["rgb"]).save(a.out / "look_probe.png")
            except Exception as e:  # noqa: BLE001
                print(f"{tag}: no image ({e})", flush=True)
        fr = h.start_frames()
        h.follow(fr)
        t0 = time.time()
        for f in range(len(fr)):
            h.step(f)
            x = h.sample()
        dt = (time.time() - t0) / len(fr)
        print(f"{tag}: stance {len(fr)} steps, {1000 * dt:.0f} ms/step; reference round trip (runtime vs frame) max xy "
              f"{100 * h.dev['xy']:.3f} cm, z {100 * h.dev['z']:.3f} cm, attitude {h.dev['rot_deg']:.3f} deg; pelvis z "
              f"{x['root'][2]:.3f} (asked {fr[-1, 2]:.3f}); xy drift {100 * np.linalg.norm(x['root'][:2] - fr[-1, :2]):.2f} cm; "
              f"fell_at {h.ha_fell_at}", flush=True)  # fmt: skip
        table = json.loads(Path(D3.TABLE).read_text())
        for i in range(a.limit or 2):
            plan_d = D3.reset_plan(np.random.default_rng(a.sx_seed + i), table, False)
            h.ep_seed = a.sx_id0 + i
            t0 = time.time()
            rows, info, frames, reaches = SX.run_reset_x(h, rig, plan_d, "lml")
            segs = D3.segments(rows, info, frames, reaches, a.sx_id0 + i, h.pl, rig)
            tc = add_touch(h, segs, reaches)
            print(f"{tag}: reset {i}: scene contact, stance {json.dumps(tc['stance'])}; reaches {json.dumps(tc['reach'])}", flush=True)
            print(f"{tag}: reset {i}: {info['steps']} steps in {time.time() - t0:.0f} s, slots "
                  f"{[(s['status'], round(s.get('wait_s', 0), 2)) for s in info['slots']]}, why '{info['why']}', segments "
                  f"{len(segs)}, end errors {[round(float(s['jaw_err_cm']), 2) for s in segs]} cm, settled "
                  f"{[bool(s['settled_end']) for s in segs]}; round trip max xy {100 * h.dev['xy']:.3f} cm z "
                  f"{100 * h.dev['z']:.3f} cm rot {h.dev['rot_deg']:.3f} deg", flush=True)  # fmt: skip
        return 0
    if a.phase == "lag":
        out = []
        for j, amp in D3.LAG_MOVES:
            h.reset()
            fr = h.start_frames()
            f0, n = len(fr) + 10, SX.FPS
            fr = np.concatenate([fr, np.repeat(fr[-1:], 10 + n + 3 * SX.FPS, 0)])
            s_ = np.arange(n + 1) / n
            fr[f0 : f0 + n + 1, 7 + j] += amp * (10 * s_**3 - 15 * s_**4 + 6 * s_**5)
            fr[f0 + n + 1 :, 7 + j] += amp
            h.follow(fr)
            q = []
            for f in range(len(fr)):
                h.step(f)
                q.append(float(sim.robot.data.joint_pos[0, h.i_mj[j]].item()))
            q = np.array(q)
            cross = lambda v, lvl: int(np.argmax((v[f0 - 1 :] - v[f0 - 1]) / (v[-1] - v[f0 - 1]) >= lvl))  # noqa: E731
            out.append((j, cross(q, 0.5) - cross(fr[:, 7 + j], 0.5), cross(q, 0.9) - cross(fr[:, 7 + j], 0.9), round((q[-1] - q[f0 - 1]) / amp, 3)))
        med = float(np.median([o[1] for o in out]))
        print(f"{tag}: per move (joint, lag at 50 % / 90 % in frames, reached share of the ramp): {out}; median lag at 50 % "
              f"{med:.1f} frames ({med / SX.FPS:.2f} s); round trip max xy {100 * h.dev['xy']:.3f} cm", flush=True)  # fmt: skip
        return 0
    # drive
    ids = [int(v) for v in a.ids.split(",")] if a.ids else list(range(a.shard, a.resets, a.shards))
    n = min(a.limit or len(ids), len(ids))
    od, ob = a.out / "data", a.out / "data_both"
    for d in (od, ob):
        d.mkdir(parents=True, exist_ok=True)
    rp = a.out / "run.json"
    if a.shard == 0 and not rp.exists():
        rp.write_text(json.dumps(dict(controller=a.controller, planner="lml", stance=h.stance, delay_f=h.ref_delay_f, seed=a.sx_seed,
                                      id0=a.sx_id0, resets=a.resets, calib=h.calib, task=a.ha_task, host="humanoidarena",
                                      body_fk="robot_fk", norender=os.environ.get("SX_NORENDER") == "1",
                                      body=dict(REACH_SCENE_DIR=str(SCENE.parent)), md5=SX.code_md5() | {"sx_ha.py": SX.md5(Path(__file__))}),
                                 indent=1))  # fmt: skip
    table = json.loads(Path(D3.TABLE).read_text())
    t0, nres, nseg = time.time(), 0, 0
    for c in range(0, n, D3.CHUNK):
        name = f"w{a.shard:03d}_{c // D3.CHUNK:03d}"
        if (od / f"{name}_log.json").exists():
            continue
        segs, segb, log = [], [], []
        for j in range(c, min(c + D3.CHUNK, n)):
            i = ids[j]
            plan_d = D3.reset_plan(np.random.default_rng(a.sx_seed + i), table, False)
            h.ep_seed = a.sx_id0 + i
            h.dev = dict(xy=0.0, z=0.0, rot_deg=0.0)
            rows, info, frames, reaches = SX.run_reset_x(h, rig, plan_d, "lml")
            s = D3.segments(rows, info, frames, reaches, a.sx_id0 + i, h.pl, rig)
            SX.add_fields(s, reaches)
            info["touch"] = add_touch(h, s, reaches)
            segs += [x for x in s if x["hands"] != 2]
            segb += [x for x in s if x["hands"] == 2]
            log.append(dict(reset=a.sx_id0 + i, n_segs=len(s), walk_asked=plan_d["walk"] is not None, ha_fell_at=h.ha_fell_at,
                            roundtrip=dict(h.dev), **info))  # fmt: skip
            nres, nseg = nres + 1, nseg + len(s)
            print(f"{tag} shard {a.shard}: {j + 1}/{n} resets, {nseg} segments this run, {time.time() - t0:.0f} s "
                  f"({(time.time() - t0) / nres:.1f} s/reset); last: {info['why']!r}, round trip xy {100 * h.dev['xy']:.2f} cm", flush=True)  # fmt: skip
        SX.save_chunk(segs, od, name, rig)
        SX.save_chunk(segb, ob, name, rig)
        (od / f"{name}_log.json").write_text(json.dumps(log, default=D3._json))
    print(f"{tag} shard {a.shard} done: {nres} resets, {nseg} segments, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    args = parser().parse_args()
    ha_sim.main_guard(lambda: run(args))
