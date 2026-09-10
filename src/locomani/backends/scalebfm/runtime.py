"""ScaleBFM runtime: the in-process sim + policy pair every ScaleBFM expert runs on.

`Sim` steps our MuJoCo scene with ScaleBFM's PD (200 Hz, decimation 4) plus our own dex1
gripper PD; `Policy` assembles the 9-tensor observation and history buffer and calls the
vendored network (network.py) in mode 2 (pelvis + both wrists as keypoint targets); `Refs`
turns time-parameterised world targets into the 6 future frames the policy consumes.
No SONIC, no ZMQ, no wall clock: one process, seeded, replayable tick for tick.

Grew out of the 2026-09-08 crate spike (docs/ledger/scalebfm.md); the spike's own episode
script was deleted with that ruling, only the runtime survives.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch

from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.network import load_policy, quat_apply_inverse, quat_mul_inverse_left
from locomani.paths import REPO as ROOT

# --- assets ------------------------------------------------------------------
MODE_TABLE = os.path.join(ROOT, "assets/scalebfm/mode_table.pt")
META = os.path.join(ROOT, "assets/scalebfm/model_22200_tensorrt_metadata.json")
FK_XML = os.path.join(ROOT, "assets/scalebfm/g1_29dof_fk.xml")

# --- scene object + gripper -------------------------------------------------
TOP_Z = 0.085
OFF = np.array([0.1515, 0.0, 0.0])  # wrist frame -> pad centre (dex1)
GRIP_OPEN, GRIP_CLOSED = 0.0245, -0.023
# gripper PD (ctrl units x gear 250 N). The joint itself clamps actuator force at +-20 N (actuatorfrcrange),
# so GRIP_CLIP above 0.08 is inert and kp 40 saturates beyond 2 mm error: the 87 g finger limit-cycles
# (+-6.5 mm sd at rest with kd 1.0) and the asymmetric close flicks the bottle out. A/B on run_bottle.py,
# seed 2, 10 rounds each (2026-09-08): kp40/kd1.0 6/10, kp40/kd0.5 10/10, kp8/kd0.15 9/10 (least chatter,
# 3.4 mm sd, lowest carry tilt). Collection uses GRIP_KD=0.5; kp 8 / kd 0.15 is the clamp-matched setting.
# GRIP_KP / GRIP_KD live in ScaleBfmBackendConfig (config.py); the A/B above is why its defaults are 1.0 / 0.025.

# --- episode script (pilot 23 numbers) ------------------------------------------
DT = 0.02
SETTLE_S = 1.5
WALK_V = 0.5  # m/s pelvis reference
WALK_ACC = 0.6
SPAWN_LAT = 0.15
PELVIS_Z = 0.76  # ScaleBFM's root_height
MODE = 2


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_wxyz(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def mat(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 3 * u * u - 2 * u**3


class Sim:
    """Our crate scene, ScaleBFM's PD on the 29 joints, our PD on the grippers."""

    def __init__(self, meta, video=False, backend: ScaleBfmBackendConfig | None = None):
        self.cfg = backend or ScaleBfmBackendConfig()
        scene = self.cfg.scene_path()
        self.m = mujoco.MjModel.from_xml_path(os.path.join(scene, "scene_locomani.xml"))
        self.d = mujoco.MjData(self.m)
        self.smeta = json.load(open(os.path.join(scene, "scene_meta.json")))
        self.low_dt = 0.005
        self.decim = 4
        self.m.opt.timestep = self.low_dt
        jn = meta["joint_names"]
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
        assert min(jid) >= 0, "joint missing in the scene"
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        self.act = np.array([next(a for a in range(self.m.nu) if self.m.actuator_trnid[a, 0] == j) for j in jid])
        self.kp = np.asarray(meta["stiffness"], dtype=float)
        self.kd = np.asarray(meta["damping"], dtype=float)
        self.tlim = np.asarray(meta["torque_limit"], dtype=float)
        self.default = np.asarray(meta["default_dof_pos"], dtype=float)
        gj = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in (
                "left_hand_gripper_1_joint",
                "left_hand_gripper_2_joint",
                "right_hand_gripper_1_joint",
                "right_hand_gripper_2_joint",
            )
        ]
        self.gq = np.array([self.m.jnt_qposadr[j] for j in gj])
        self.gv = np.array([self.m.jnt_dofadr[j] for j in gj])
        self.gact = np.array([next(a for a in range(self.m.nu) if self.m.actuator_trnid[a, 0] == j) for j in gj])
        self.grip_target = np.full(4, GRIP_OPEN)
        self.grip_clip = 4.0  # ctrl clamp on the gripper PD (x gear 250 N); lower for small objects
        self.bid = {
            n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in (
                "pelvis",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "torso_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            )
        }
        self.crate_b = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, self.smeta["objects"][self.cfg.object_name]["body"]
        )
        self.crate_q = int(self.m.jnt_qposadr[self.m.body_jntadr[self.crate_b]])
        self.crate0 = np.asarray(self.smeta["objects"][self.cfg.object_name]["init_pos"], dtype=float)
        self.crate_quat0 = np.asarray(self.smeta["objects"][self.cfg.object_name]["init_quat"], dtype=float)
        surf = self.smeta["surfaces"]["A"]
        self.edge_x = float(surf["xy"][0] - surf["flat_half_extent"][0])
        self.table_top = float(surf["top"])
        self.renderer = mujoco.Renderer(self.m, height=480, width=640) if video else None
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # -- reset ------------------------------------------------------------------
    def reset(self, spawn_xy, spawn_yaw, crate_dy):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:3] = [spawn_xy[0], spawn_xy[1], PELVIS_Z + 0.02]
        self.d.qpos[3:7] = [np.cos(spawn_yaw / 2), 0, 0, np.sin(spawn_yaw / 2)]
        self.d.qpos[self.qadr] = self.default
        self.d.qpos[self.gq] = GRIP_OPEN
        c = self.crate0.copy()
        c[1] += crate_dy
        self.d.qpos[self.crate_q : self.crate_q + 3] = c
        self.d.qpos[self.crate_q + 3 : self.crate_q + 7] = self.crate_quat0
        self.d.ctrl[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.grip_target[:] = GRIP_OPEN

    # -- state ------------------------------------------------------------------
    def state(self):
        d = self.d
        return dict(
            root_pos=d.qpos[:3].copy(),
            root_quat=d.qpos[3:7].copy(),
            ang_vel=d.qvel[3:6].copy(),
            dof_pos=d.qpos[self.qadr].copy(),
            dof_vel=d.qvel[self.vadr].copy(),
        )

    def body(self, name):
        b = self.bid[name]
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()

    def pad(self, side):
        p, q = self.body(f"{side}_wrist_yaw_link")
        return p + mat(q) @ OFF

    def crate(self):
        return self.d.xpos[self.crate_b].copy(), self.d.xquat[self.crate_b].copy()

    def fallen(self):
        return self.d.qpos[2] < 0.40  # SONIC evaluation fall threshold (user alignment, 2026-09-10)

    # -- step -------------------------------------------------------------------
    def step(self, tgt):
        d = self.d
        for _ in range(self.decim):
            tau = self.kp * (tgt - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
            d.ctrl[self.act] = np.clip(tau, -self.tlim, self.tlim)
            d.ctrl[self.gact] = np.clip(
                self.cfg.grip_kp * (self.grip_target - d.qpos[self.gq]) - self.cfg.grip_kd * d.qvel[self.gv],
                -self.grip_clip,
                self.grip_clip,
            )
            mujoco.mj_step(self.m, d)

    def frame(self):
        pel = self.d.xpos[self.bid["pelvis"]]
        self.cam.lookat[:] = [pel[0] + 0.25, pel[1], 0.72]
        self.cam.distance = 2.0
        self.cam.azimuth = 135
        self.cam.elevation = -20
        self.renderer.update_scene(self.d, camera=self.cam)
        return self.renderer.render()


class Policy:
    def __init__(self, device="cuda", backend: ScaleBfmBackendConfig | None = None):
        self.cfg = backend or ScaleBfmBackendConfig()
        self.w, self.meta, _, _ = load_policy(self.cfg.checkpoint, MODE_TABLE, META, FK_XML, device)
        self.dev = device
        self.H = self.meta["history_buffer_size"]
        self.fidx = list(self.meta["future_idx"])
        self.F = len(self.fidx)
        self.sel = self.meta["selected_body_names"]
        self.N = len(self.sel)
        self.toff = torch.tensor(self.fidx, dtype=torch.long, device=device)[None, :, None]
        self.mode = torch.tensor([MODE], dtype=torch.long, device=device)
        self.isel = {n: i for i, n in enumerate(self.sel)}

    def reset(self, st):
        H = self.H
        self.buf = dict(
            root_quat=np.tile(st["root_quat"], (H, 1)),
            ang_vel=np.tile(st["ang_vel"], (H, 1)),
            dof_pos=np.tile(st["dof_pos"], (H, 1)),
            dof_vel=np.tile(st["dof_vel"], (H, 1)),
            action=np.zeros((H, 29)),
        )

    def push(self, st):
        for k in ("root_quat", "ang_vel", "dof_pos", "dof_vel"):
            self.buf[k] = np.roll(self.buf[k], -1, 0)
            self.buf[k][-1] = st[k]

    def __call__(self, st, targets):
        """targets: dict body_name -> (pos_w[F,3], quat_w[F,4]); unlisted bodies get zeros
        (masked out by mode 2 anyway)."""
        self.push(st)
        pw = np.zeros((self.F, self.N, 3))
        qw = np.zeros((self.F, self.N, 4))
        qw[..., 0] = 1
        for n, (p, q) in targets.items():
            pw[:, self.isel[n]] = p
            qw[:, self.isel[n]] = q
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.dev)
        rq = t(self.buf["root_quat"])[None]
        root_q = rq[:, -1]
        pw_t = t(pw)[None]
        qw_t = t(qw)[None]
        tp = quat_apply_inverse(root_q, pw_t - t(st["root_pos"])[None, None, None, :])
        tq = quat_mul_inverse_left(root_q[:, None, None, :].expand(-1, self.F, self.N, -1), qw_t)
        with torch.inference_mode():
            tgt, act = self.w(
                rq,
                t(self.buf["ang_vel"])[None],
                t(self.buf["dof_pos"])[None],
                t(self.buf["dof_vel"])[None],
                t(self.buf["action"])[None],
                tp,
                tq,
                self.mode,
                self.toff,
            )
        self.buf["action"] = np.roll(self.buf["action"], -1, 0)
        self.buf["action"][-1] = act[0].cpu().numpy()
        return tgt[0].cpu().numpy()


# --- reference builders ---------------------------------------------------------
class Refs:
    """Time-parameterised world targets for pelvis and both wrists. Each phase sets
    a callable ref(t) -> dict name -> (pos, quat); futures are sampled at t + j*DT."""

    def __init__(self, pol: Policy, sim: Sim):
        self.pol, self.sim = pol, sim
        # wrist pose relative to the pelvis in the default (arms hanging) pose
        st = sim.state()
        pp, pq = sim.body("pelvis")
        Rp = mat(pq)
        self.hand_rel = {}
        for sd in ("left", "right"):
            wp, wq = sim.body(f"{sd}_wrist_yaw_link")
            self.hand_rel[sd] = (Rp.T @ (wp - pp), Rp.T @ mat(wq))

    def hands_at_side(self, pel_pos, pel_yaw):
        R = rot_z(pel_yaw)
        out = {}
        for sd in ("left", "right"):
            dp, dR = self.hand_rel[sd]
            out[f"{sd}_wrist_yaw_link"] = (pel_pos + R @ dp, quat_wxyz(R @ dR))
        return out

    def pelvis(self, xy, yaw, z=PELVIS_Z):
        return {"pelvis": (np.array([xy[0], xy[1], z]), np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))}

    def futures(self, ref, t):
        out = {}
        for j in self.pol.fidx:
            r = ref(t + j * DT)
            for n, (p, q) in r.items():
                out.setdefault(n, ([], []))
                out[n][0].append(p)
                out[n][1].append(q)
        return {n: (np.array(v[0]), np.array(v[1])) for n, v in out.items()}


def wrist_from_pad(pad_w, R_palm):
    return pad_w - R_palm @ OFF, quat_wxyz(R_palm)


# --- one episode ----------------------------------------------------------------
