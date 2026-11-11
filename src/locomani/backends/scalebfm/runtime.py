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
from locomani.sim import perturb as perturbation

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


def load_model(xml: str) -> mujoco.MjModel:
    """`xml` compiled, through a binary cache (`paths.MJB_CACHE/<key>.mjb`): the ADT apartment takes 6.8 s to compile
    and 0.13 s to load as MJB, and the MJB model is the same model (re-saved byte for byte; 2000 steps give the same
    qpos, TODO 32.2). The key covers the XML's bytes, the size and mtime of every file it names and the MuJoCo
    version; nothing is written beside the XML (tests compile fixtures). An unwritable cache falls back to compiling.
    Old keys are not pruned (the apartment's is 269 MB)."""
    import hashlib
    import re

    from locomani.paths import MJB_CACHE

    text = open(xml, "rb").read()
    h = hashlib.sha256(text + mujoco.__version__.encode())
    for f in sorted(set(re.findall(rb'file="([^"]+)"', text))):
        p = os.path.join(os.path.dirname(xml), f.decode())
        st = os.stat(p) if os.path.exists(p) else None
        h.update(f + (f"{st.st_size}:{st.st_mtime_ns}" if st else "missing").encode())
    cache = os.path.join(MJB_CACHE, h.hexdigest()[:32] + ".mjb")
    if os.path.exists(cache):
        try:
            return mujoco.MjModel.from_binary_path(cache)
        except Exception:  # a truncated file from a killed writer: compile and replace it
            pass
    m = mujoco.MjModel.from_xml_path(xml)
    try:
        os.makedirs(MJB_CACHE, exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp"
        mujoco.mj_saveModel(m, tmp, None)
        os.replace(tmp, cache)
    except OSError:
        pass
    return m


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
    """Our crate scene (or any scene_locomani.xml with the G1; the tracked object is optional), ScaleBFM's PD on
    the 29 joints, our PD on the grippers."""

    def __init__(self, meta, video=False, backend: ScaleBfmBackendConfig | None = None, perturb: dict | None = None):
        self.cfg = backend or ScaleBfmBackendConfig()
        scene = self.cfg.scene_path()
        self.m = load_model(os.path.join(scene, "scene_locomani.xml"))
        self.d = mujoco.MjData(self.m)
        self.smeta = json.load(open(os.path.join(scene, "scene_meta.json")))
        self.low_dt = 0.005
        self.decim = 4
        self.m.opt.timestep = self.low_dt
        if self.cfg.sleep:
            self.m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
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
        self.grip_clip = self.cfg.grip_clip  # ctrl clamp on the gripper PD (config.py); callers may lower it per round
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
        # A scene without the tracked object (the walk scene, outputs/scene/walk) has no crate and no table:
        # the locomotion diagnostics (wbc_compare brake line, 2026-10-13) run there; crate()/table_top unused.
        obj = self.smeta["objects"].get(self.cfg.object_name)
        if obj is None:
            self.crate_b = self.crate_q = self.crate0 = self.crate_quat0 = None
        else:
            self.crate_b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, obj["body"])
            self.crate_q = int(self.m.jnt_qposadr[self.m.body_jntadr[self.crate_b]])
            self.crate0 = np.asarray(obj["init_pos"], dtype=float)
            self.crate_quat0 = np.asarray(obj["init_quat"], dtype=float)
        surf = self.smeta["surfaces"].get("A")
        self.edge_x = None if surf is None else float(surf["xy"][0] - surf["flat_half_extent"][0])
        self.table_top = None if surf is None else float(surf["top"])
        # TODO 25.5: the SONIC sim's perturbations (sim/perturb.py) -- delay / gain scale in step(), noise in state()
        self.pert = perturbation.Perturbation(perturb, len(jid), self.low_dt) if perturb else None
        if perturb and perturb.get("friction") is not None:
            perturbation.set_friction(
                self.m, float(perturb["friction"]), ("left_ankle_roll_link", "right_ankle_roll_link")
            )
        self.pelvis_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
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
        if self.crate_b is not None:
            c = self.crate0.copy()
            c[1] += crate_dy
            self.d.qpos[self.crate_q : self.crate_q + 3] = c
            self.d.qpos[self.crate_q + 3 : self.crate_q + 7] = self.crate_quat0
        self.d.ctrl[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.grip_target[:] = GRIP_OPEN
        if self.pert is not None:
            self.pert.redraw()
            self.pert.queue.clear()
            self.pert.steps = 0
            perturbation.PUSH.update(force=None, steps=0)

    # -- state ------------------------------------------------------------------
    def state(self):
        d = self.d
        st = dict(
            root_pos=d.qpos[:3].copy(),
            root_quat=d.qpos[3:7].copy(),
            ang_vel=d.qvel[3:6].copy(),
            dof_pos=d.qpos[self.qadr].copy(),
            dof_vel=d.qvel[self.vadr].copy(),
        )
        if self.pert is not None and self.pert.noise:
            st["ang_vel"] = self.pert.noisy("gyro", st["ang_vel"])
            st["dof_pos"] = self.pert.noisy("q", st["dof_pos"])
            st["dof_vel"] = self.pert.noisy("dq", st["dof_vel"])
        return st

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
        gain = 1.0 if self.pert is None else self.pert.gain
        for _ in range(self.decim):
            if self.pert is not None:
                self.pert.tick()
            perturbation.apply_push(d, self.pelvis_id)
            q_cmd = tgt if self.pert is None else self.pert.delayed(np.asarray(tgt, dtype=float).copy())
            tau = gain * self.kp * (q_cmd - d.qpos[self.qadr]) - gain * self.kd * d.qvel[self.vadr]
            d.ctrl[self.act] = np.clip(tau, -self.tlim, self.tlim)
            d.ctrl[self.gact] = np.clip(
                self.cfg.grip_kp * (self.grip_target - d.qpos[self.gq]) - self.cfg.grip_kd * d.qvel[self.gv],
                -self.grip_clip,
                self.grip_clip,
            )
            mujoco.mj_step(self.m, d)

    def push(self, force, seconds: float = 0.2) -> None:
        """TODO 25.5 g: `force` (N, world xyz) on the pelvis for `seconds`."""
        perturbation.request_push(force, seconds, self.low_dt)

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
        self.w, self.meta, _, _ = load_policy(self.cfg.checkpoint_path(), MODE_TABLE, META, FK_XML, device)
        self.dev = device
        self.H = self.meta["history_buffer_size"]
        self.fidx = list(self.meta["future_idx"])
        self.F = len(self.fidx)
        self.sel = self.meta["selected_body_names"]
        self.N = len(self.sel)
        self.toff = torch.tensor(self.fidx, dtype=torch.long, device=device)[None, :, None]
        self.mode = torch.tensor([MODE], dtype=torch.long, device=device)
        self.isel = {n: i for i, n in enumerate(self.sel)}
        self.graph = self.cfg.cuda_graph and str(device).startswith("cuda")
        self.root_source = None  # optional (state, targets) -> the pelvis position the policy is told, an onboard estimate; None: as cfg.reference_forcing says
        self.eager_calls, self.g = 0, None
        # one host buffer for every input, each part starting on a 512-byte boundary as a tensor of its own would
        # (cuBLAS may choose its kernel by alignment, and the outputs must stay bit for bit those of the eager path)
        shapes = dict(rq=(1, self.H, 4), av=(1, self.H, 3), dp=(1, self.H, 29), dv=(1, self.H, 29),
                      ac=(1, self.H, 29), pw=(1, self.F, self.N, 3), qw=(1, self.F, self.N, 4), rp=(3,))  # fmt: skip
        self.parts, o = {}, 0
        for k, sh in shapes.items():
            self.parts[k] = (o, sh)
            o += -(-int(np.prod(sh)) // 128) * 128
        self.host = np.zeros(o, np.float32)

    # eager calls before the capture: the first call differs by ~6e-8 (cuBLAS set-up; motor.reset warms up)
    EAGER_FIRST = 3

    def forward(self, flat: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        """The packed inputs to [tgt..., action...] (flat): the targets into the root frame, then the network in `mode`
        (the motor switches `self.mode` between 2 and 4, so the graph takes it as an input like the rest)."""
        x = {k: flat[o : o + int(np.prod(sh))].view(sh) for k, (o, sh) in self.parts.items()}
        root_q = x["rq"][:, -1]
        tp = quat_apply_inverse(root_q, x["pw"] - x["rp"][None, None, None, :])
        tq = quat_mul_inverse_left(root_q[:, None, None, :].expand(-1, self.F, self.N, -1), x["qw"])
        tgt, act = self.w(x["rq"], x["av"], x["dp"], x["dv"], x["ac"], tp, tq, mode, self.toff)
        self.n_tgt = tgt[0].numel()
        self.tgt_shape = tuple(tgt[0].shape)
        return torch.cat([tgt[0].reshape(-1), act[0].reshape(-1)])

    def net(self, host: np.ndarray) -> np.ndarray:
        """`forward` on the packed inputs: eager, or (cfg.cuda_graph) the same kernels captured once into a CUDA graph
        and replayed on a static copy of the inputs (TODO 32.3: 7.8 -> 3.5 ms a call with twelve rounds running;
        600 calls bit for bit the eager outputs)."""
        if not self.graph or self.eager_calls < self.EAGER_FIRST:
            self.eager_calls += 1
            with torch.inference_mode():
                return self.forward(torch.as_tensor(host, device=self.dev), self.mode).cpu().numpy()
        if self.g is None:
            self.pinned = torch.from_numpy(host.copy()).pin_memory()
            self.gin = torch.as_tensor(host, device=self.dev).clone()
            self.gmode = self.mode.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.inference_mode():
                self.forward(self.gin, self.gmode)
            torch.cuda.current_stream().wait_stream(side)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g), torch.inference_mode():
                self.gout = self.forward(self.gin, self.gmode)
        self.pinned.numpy()[:] = host
        self.gin.copy_(self.pinned, non_blocking=True)
        self.gmode.copy_(self.mode)
        self.g.replay()
        return self.gout.cpu().numpy()

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
        h = self.host
        if self.root_source is not None:
            root_pos = self.root_source(st, targets)
        else:
            root_pos = targets["pelvis"][0][0] if self.cfg.reference_forcing and "pelvis" in targets else st["root_pos"]
        for k, a in (("rq", self.buf["root_quat"]), ("av", self.buf["ang_vel"]), ("dp", self.buf["dof_pos"]),
                     ("dv", self.buf["dof_vel"]), ("ac", self.buf["action"]), ("pw", pw), ("qw", qw),
                     ("rp", root_pos)):  # fmt: skip
            o, sh = self.parts[k]
            h[o : o + int(np.prod(sh))] = np.asarray(a, dtype=np.float32).reshape(-1)
        out = self.net(h)
        self.buf["action"] = np.roll(self.buf["action"], -1, 0)
        self.buf["action"][-1] = out[self.n_tgt :]
        return out[: self.n_tgt].reshape(self.tgt_shape)


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
