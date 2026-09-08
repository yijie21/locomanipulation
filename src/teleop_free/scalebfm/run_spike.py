"""ScaleBFM spike: drive the bimanual crate scene with the ScaleBFM policy in
mode 2 (pelvis + both wrists as keypoint targets), no SONIC, no ZMQ.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $VENV_SIM_PY src/teleop_free/scalebfm/run_spike.py \
        --rounds 10 --out outputs/videos/scalebfm_spike [--video] [--seed 0]
    ... --envelope   # reach-envelope sweep instead of episodes

Episode = the pilot-23 script expressed as keypoint references: settle, walk
the pelvis reference to the stand, hold, one Cartesian Bezier reach to the
hover, descend until contact / floor, close, lift 10 cm, hold. The policy is
called in-process at 50 Hz (PD at 200 Hz with ScaleBFM's gains), the dex1
grippers are our own PD. Everything is measured from privileged sim state,
this is a controller test, not data.

Outputs (under --out): manifest.jsonl (one line per round, same keys the
wall-pilot summariser reads), trace/wall_s0_NN_state_trace.npz (qpos at 50 Hz,
same layout as the pilots so the render scripts work), NN.mp4 when --video.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from bfm_policy import load_policy, quat_apply_inverse, quat_mul_inverse_left  # noqa: E402

# --- assets ------------------------------------------------------------------
CKPT = os.environ.get("SCALEBFM_CKPT", "/workspace/models/scalebfm/humanoid_transformer_m_22200.pt")
MODE_TABLE = os.path.join(ROOT, "assets/scalebfm/mode_table.pt")
META = os.path.join(ROOT, "assets/scalebfm/model_22200_tensorrt_metadata.json")
FK_XML = os.path.join(ROOT, "assets/scalebfm/g1_29dof_fk.xml")
SCENE = os.environ.get("SCENE_DIR", os.path.join(ROOT, "outputs/scene/crate_bimanual"))

# --- crate geometry (bimanual_wall.py, rail feature) --------------------------
OBJ = "blue_crate"
DEPTH_AXIS, WALL_AXIS = 1, 0
WALL_Y, NEAR_X = 0.1933, 0.1347
TOP_Z = 0.085
GRIP_BELOW = 0.050
HOVER_ABOVE = 0.06
ACROSS_BIAS = 0.012
ACROSS_TOL = 0.015
PALM_PITCH_DEG = 60.0
REACH_SUM = 0.38
OFF = np.array([0.1515, 0.0, 0.0])          # wrist frame -> pad centre (dex1)
GRIP_OPEN, GRIP_CLOSED = 0.0245, -0.023
# gripper PD (ctrl units x gear 250 N). The joint itself clamps actuator force at +-20 N (actuatorfrcrange),
# so GRIP_CLIP above 0.08 is inert and kp 40 saturates beyond 2 mm error: the 87 g finger limit-cycles
# (+-6.5 mm sd at rest with kd 1.0) and the asymmetric close flicks the bottle out. A/B on run_bottle.py,
# seed 2, 10 rounds each (2026-09-08): kp40/kd1.0 6/10, kp40/kd0.5 10/10, kp8/kd0.15 9/10 (least chatter,
# 3.4 mm sd, lowest carry tilt). Collection uses GRIP_KD=0.5; kp 8 / kd 0.15 is the clamp-matched setting.
GRIP_KP = float(os.environ.get("GRIP_KP", 40.0))
GRIP_KD = float(os.environ.get("GRIP_KD", 1.0))

# --- episode script (pilot 23 numbers) ------------------------------------------
DT = 0.02
SETTLE_S = 1.5
WALK_V = 0.5                                # m/s pelvis reference
WALK_ACC = 0.6
STAND_OFF = 0.24
WALK_LAT_BIAS = 0.0                         # SONIC's gait parked 4 cm left; measure ours first
SPAWN_BACK = (1.0, 1.3)
SPAWN_LAT = 0.15
CRATE_DY = (-0.08, 0.08)
ARRIVE_HOLD_S = 1.5
REACH_S = 2.5
REACH_UP, REACH_OUT, REACH_OVER = 0.25, 0.15, 0.10
REACH_PITCH0_DEG = 90.0
DESCEND_SPEED = 0.04
DESCEND_TIMEOUT = 8.0
CONTACT_STALL_S, CONTACT_STALL_MM = 0.3, 3.0
CLOSE_S, CLOSE_HOLD_S = 0.8, 0.5
LIFT_HEIGHT, LIFT_S, LIFT_HOLD_S = 0.10, 2.0, 2.0
LIFT_MIN, TILT_OK_DEG = 0.06, 20.0
PELVIS_Z = 0.76                             # ScaleBFM's root_height
MODE = 2
# closed-loop correction of the references on the measured error (the SONIC
# expert re-solved its IK on the measured residual the same way): integral
# gain 1/s and cap in metres, hands and pelvis separately, on from the hover
HOVER_SETTLE_S = 1.0
SERVO_KI_HAND, SERVO_CAP_HAND = float(os.environ.get("SERVO_KI_HAND", 2.0)), 0.08
SERVO_KI_PEL, SERVO_CAP_PEL = float(os.environ.get("SERVO_KI_PEL", 1.0)), 0.10


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
    return 3 * u * u - 2 * u ** 3


class Sim:
    """Our crate scene, ScaleBFM's PD on the 29 joints, our PD on the grippers."""

    def __init__(self, meta, video=False):
        self.m = mujoco.MjModel.from_xml_path(os.path.join(SCENE, "scene_locomani.xml"))
        self.d = mujoco.MjData(self.m)
        self.smeta = json.load(open(os.path.join(SCENE, "scene_meta.json")))
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
        gj = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in
              ("left_hand_gripper_1_joint", "left_hand_gripper_2_joint", "right_hand_gripper_1_joint", "right_hand_gripper_2_joint")]
        self.gq = np.array([self.m.jnt_qposadr[j] for j in gj]); self.gv = np.array([self.m.jnt_dofadr[j] for j in gj])
        self.gact = np.array([next(a for a in range(self.m.nu) if self.m.actuator_trnid[a, 0] == j) for j in gj])
        self.grip_target = np.full(4, GRIP_OPEN)
        self.grip_clip = 4.0                     # ctrl clamp on the gripper PD (x gear 250 N); lower for small objects
        self.bid = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                    ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")}
        self.crate_b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, self.smeta["objects"][OBJ]["body"])
        self.crate_q = int(self.m.jnt_qposadr[self.m.body_jntadr[self.crate_b]])
        self.crate0 = np.asarray(self.smeta["objects"][OBJ]["init_pos"], dtype=float)
        self.crate_quat0 = np.asarray(self.smeta["objects"][OBJ]["init_quat"], dtype=float)
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
        c = self.crate0.copy(); c[1] += crate_dy
        self.d.qpos[self.crate_q:self.crate_q + 3] = c
        self.d.qpos[self.crate_q + 3:self.crate_q + 7] = self.crate_quat0
        self.d.ctrl[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.grip_target[:] = GRIP_OPEN

    # -- state ------------------------------------------------------------------
    def state(self):
        d = self.d
        return dict(root_pos=d.qpos[:3].copy(), root_quat=d.qpos[3:7].copy(), ang_vel=d.qvel[3:6].copy(),
                    dof_pos=d.qpos[self.qadr].copy(), dof_vel=d.qvel[self.vadr].copy())

    def body(self, name):
        b = self.bid[name]
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()

    def pad(self, side):
        p, q = self.body(f"{side}_wrist_yaw_link")
        return p + mat(q) @ OFF

    def crate(self):
        return self.d.xpos[self.crate_b].copy(), self.d.xquat[self.crate_b].copy()

    def fallen(self):
        return self.d.qpos[2] < 0.45

    # -- step -------------------------------------------------------------------
    def step(self, tgt):
        d = self.d
        for _ in range(self.decim):
            tau = self.kp * (tgt - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
            d.ctrl[self.act] = np.clip(tau, -self.tlim, self.tlim)
            d.ctrl[self.gact] = np.clip(GRIP_KP * (self.grip_target - d.qpos[self.gq]) - GRIP_KD * d.qvel[self.gv], -self.grip_clip, self.grip_clip)
            mujoco.mj_step(self.m, d)

    def frame(self):
        pel = self.d.xpos[self.bid["pelvis"]]
        self.cam.lookat[:] = [pel[0] + 0.25, pel[1], 0.72]
        self.cam.distance = 2.0; self.cam.azimuth = 135; self.cam.elevation = -20
        self.renderer.update_scene(self.d, camera=self.cam)
        return self.renderer.render()


class Policy:
    def __init__(self, device="cuda"):
        self.w, self.meta, _, _ = load_policy(CKPT, MODE_TABLE, META, FK_XML, device)
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
        self.buf = dict(root_quat=np.tile(st["root_quat"], (H, 1)), ang_vel=np.tile(st["ang_vel"], (H, 1)),
                        dof_pos=np.tile(st["dof_pos"], (H, 1)), dof_vel=np.tile(st["dof_vel"], (H, 1)),
                        action=np.zeros((H, 29)))

    def push(self, st):
        for k in ("root_quat", "ang_vel", "dof_pos", "dof_vel"):
            self.buf[k] = np.roll(self.buf[k], -1, 0); self.buf[k][-1] = st[k]

    def __call__(self, st, targets):
        """targets: dict body_name -> (pos_w[F,3], quat_w[F,4]); unlisted bodies get zeros
        (masked out by mode 2 anyway)."""
        self.push(st)
        pw = np.zeros((self.F, self.N, 3)); qw = np.zeros((self.F, self.N, 4)); qw[..., 0] = 1
        for n, (p, q) in targets.items():
            pw[:, self.isel[n]] = p; qw[:, self.isel[n]] = q
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.dev)
        rq = t(self.buf["root_quat"])[None]
        root_q = rq[:, -1]
        pw_t = t(pw)[None]; qw_t = t(qw)[None]
        tp = quat_apply_inverse(root_q, pw_t - t(st["root_pos"])[None, None, None, :])
        tq = quat_mul_inverse_left(root_q[:, None, None, :].expand(-1, self.F, self.N, -1), qw_t)
        with torch.inference_mode():
            tgt, act = self.w(rq, t(self.buf["ang_vel"])[None], t(self.buf["dof_pos"])[None], t(self.buf["dof_vel"])[None],
                              t(self.buf["action"])[None], tp, tq, self.mode, self.toff)
        self.buf["action"] = np.roll(self.buf["action"], -1, 0); self.buf["action"][-1] = act[0].cpu().numpy()
        return tgt[0].cpu().numpy()


# --- crate frame helpers (bimanual_wall.wall_frame / wall_point) -----------------
def wall_frame(sim):
    c, q = sim.crate()
    R = mat(q)
    ex = R[:2, DEPTH_AXIS] / max(np.linalg.norm(R[:2, DEPTH_AXIS]), 1e-6)
    pel = sim.d.qpos[:2]
    sign = 1.0 if np.dot(ex, c[:2] - pel) >= 0 else -1.0
    n = sign * ex
    y_hat = np.array([-n[1], n[0]])
    psi = float(np.arctan2(n[1], n[0]))
    return c, n, y_hat, psi, sign, R


def wall_point(sim, side, depth, dz, inset=0.0):
    c, n, y_hat, psi, sign, R = wall_frame(sim)
    e_wall = np.zeros(3); e_wall[WALL_AXIS] = WALL_Y
    yl = WALL_Y if np.dot((R @ e_wall)[:2], y_hat) > 0 else -WALL_Y
    if side == "right":
        yl = -yl
    yl -= np.sign(yl) * inset
    v = np.zeros(3)
    v[DEPTH_AXIS] = sign * (-NEAR_X + depth)
    v[WALL_AXIS] = yl
    v[2] = TOP_Z + dz
    return c + R @ v


def measure_stand(sim):
    c, n, y_hat, psi, _, _ = wall_frame(sim)
    pel = sim.d.qpos[:2].copy()
    near = wall_point(sim, "left", 0.0, 0.0)
    so = float(np.dot(near[:2] - pel, n))
    lat = float(np.dot(pel - c[:2], y_hat))
    yaw = float(np.degrees(wrap(psi - yaw_of(sim.d.qpos[3:7]))))
    return so, lat, yaw, psi


def crate_tilt_deg(q):
    R = mat(q)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))


# --- reference builders ---------------------------------------------------------
class Refs:
    """Time-parameterised world targets for pelvis and both wrists. Each phase sets
    a callable ref(t) -> dict name -> (pos, quat); futures are sampled at t + j*DT."""

    def __init__(self, pol: Policy, sim: Sim):
        self.pol, self.sim = pol, sim
        # wrist pose relative to the pelvis in the default (arms hanging) pose
        st = sim.state()
        pp, pq = sim.body("pelvis"); Rp = mat(pq)
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
                out[n][0].append(p); out[n][1].append(q)
        return {n: (np.array(v[0]), np.array(v[1])) for n, v in out.items()}


def wrist_from_pad(pad_w, R_palm):
    return pad_w - R_palm @ OFF, quat_wxyz(R_palm)


# --- one episode ----------------------------------------------------------------
def run_round(sim: Sim, pol: Policy, rnd: int, rng, out_dir, video):
    m = dict(round=rnd, ok=False, why="", close_trigger=None, stand_attempts=[])
    crate_dy = float(rng.uniform(*CRATE_DY))
    crate_x = sim.edge_x + NEAR_X
    stand_xy = np.array([crate_x - NEAR_X - STAND_OFF, crate_dy + WALK_LAT_BIAS])
    spawn_back = float(rng.uniform(*SPAWN_BACK)); spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
    spawn_xy = stand_xy - np.array([spawn_back, -spawn_lat])
    yaw0 = float(np.arctan2(*(np.array([crate_x, crate_dy]) - spawn_xy)[::-1])) + float(rng.uniform(-0.17, 0.17))
    m.update(crate_dy=crate_dy, spawn_back=spawn_back, spawn_lat=spawn_lat, spawn_yaw_deg=float(np.degrees(yaw0)))
    sim.reset(spawn_xy, yaw0, crate_dy)
    pol.reset(sim.state())
    refs = Refs(pol, sim)
    trace = []; frames = []; log = []
    ffm = None
    if video:
        ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                                "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                                os.path.join(out_dir, f"{rnd:02d}.mp4")], stdin=subprocess.PIPE)
    T = [0.0]
    crate_p0, _ = sim.crate()
    first_move = [None]
    corr = {"left_wrist_yaw_link": np.zeros(3), "right_wrist_yaw_link": np.zeros(3), "pelvis": np.zeros(3)}
    servo_on = [False]

    def corrected(ref):
        def f(t):
            out = {}
            for n, (p, q) in ref(t).items():
                out[n] = (p + corr[n], q)
            return out
        return f

    def tick(ref):
        st = sim.state()
        tgt = pol(st, refs.futures(corrected(ref), T[0]))
        sim.step(tgt)
        if servo_on[0]:
            r = ref(T[0])
            for sd in ("left", "right"):
                n_ = f"{sd}_wrist_yaw_link"
                pad_ref = r[n_][0] + mat(r[n_][1]) @ OFF
                corr[n_] += SERVO_KI_HAND * DT * (pad_ref - sim.pad(sd))
                corr[n_] = np.clip(corr[n_], -SERVO_CAP_HAND, SERVO_CAP_HAND)
            e_p = r["pelvis"][0][:2] - sim.d.qpos[:2]
            corr["pelvis"][:2] = np.clip(corr["pelvis"][:2] + SERVO_KI_PEL * DT * e_p, -SERVO_CAP_PEL, SERVO_CAP_PEL)
        T[0] += DT
        k = len(trace)
        trace.append(sim.d.qpos.copy())
        cp, cq = sim.crate()
        if first_move[0] is None and (np.linalg.norm(cp[:2] - crate_p0[:2]) > 0.005 or abs(np.degrees(wrap(yaw_of(cq) - yaw_of(sim.crate_quat0)))) > 1.0):
            first_move[0] = T[0]
        if ffm is not None and k % 2 == 0:
            ffm.stdin.write(sim.frame().tobytes())
        return sim.fallen()

    def finish(why=None):
        if why:
            m["why"] = why
        m["move_s"] = first_move[0]
        m["len_s"] = T[0]
        m["log"] = log
        np.savez_compressed(os.path.join(out_dir, "trace", f"wall_s0_{rnd:02d}_state_trace.npz"), qpos=np.array(trace), dt=DT)
        if ffm is not None:
            ffm.stdin.close(); ffm.wait()
        return m

    # --- settle: hold the spawn pose --------------------------------------------
    p0 = sim.d.qpos[:2].copy()
    ref = lambda t: {**refs.pelvis(p0, yaw0), **refs.hands_at_side(np.array([p0[0], p0[1], PELVIS_Z]), yaw0)}
    for _ in range(int(SETTLE_S / DT)):
        if tick(ref):
            return finish("[settle] fell")
    # --- walk: pelvis reference along the straight line, trapezoid speed ---------
    c, n, y_hat, psi, _, _ = wall_frame(sim)
    start = sim.d.qpos[:2].copy(); yaw_s = yaw_of(sim.d.qpos[3:7])
    seg = stand_xy - start; L = float(np.linalg.norm(seg)); u_dir = seg / max(L, 1e-6)
    t_acc = WALK_V / WALK_ACC; d_acc = 0.5 * WALK_ACC * t_acc ** 2
    if 2 * d_acc > L:
        t_acc = np.sqrt(L / WALK_ACC); d_acc = L / 2; v_peak = WALK_ACC * t_acc
    else:
        v_peak = WALK_V
    t_cruise = (L - 2 * d_acc) / v_peak
    t_total = 2 * t_acc + t_cruise
    t_walk0 = T[0]

    def s_of(tau):
        tau = float(np.clip(tau, 0.0, t_total))
        if tau < t_acc:
            return 0.5 * WALK_ACC * tau ** 2
        if tau < t_acc + t_cruise:
            return d_acc + v_peak * (tau - t_acc)
        r = t_total - tau
        return L - 0.5 * WALK_ACC * r ** 2

    def walk_ref(t):
        s = s_of(t - t_walk0)
        xy = start + u_dir * s
        yaw = yaw_s + wrap(psi - yaw_s) * smoothstep(s / max(L, 1e-6))
        return {**refs.pelvis(xy, yaw), **refs.hands_at_side(np.array([xy[0], xy[1], PELVIS_Z]), yaw)}

    for _ in range(int((t_total + ARRIVE_HOLD_S) / DT)):
        if tick(walk_ref):
            return finish("[walk] fell")
    so, lat, yaw_err, psi = measure_stand(sim)
    pel_err = float(np.linalg.norm(sim.d.qpos[:2] - stand_xy))
    m.update(stand_off_meas=so, stand_lat=lat, yaw_err_deg=yaw_err, pelvis_err=pel_err,
             stand_attempts=[dict(so=so, lat=lat, yaw=yaw_err)], walk_s=t_total)
    log.append(f"stand: so {so:.3f} lat {lat:+.3f} yaw {yaw_err:+.1f} deg | pelvis vs target {pel_err * 100:.1f} cm")
    if first_move[0] is not None:
        m["why"] = "[walk] crate moved during the walk-in"
    # --- reach: one Bezier per pad, pelvis pinned at the reference stand ---------
    depth = float(np.clip(REACH_SUM - so, 0.06, 0.25))
    m["depth"] = depth
    pel_hold = refs.pelvis(stand_xy, psi)
    pad0 = {sd: sim.pad(sd) for sd in ("left", "right")}
    p3 = {sd: wall_point(sim, sd, depth, HOVER_ABOVE, ACROSS_BIAS) for sd in ("left", "right")}
    out_w = {"left": np.array([y_hat[0], y_hat[1], 0.0]), "right": -np.array([y_hat[0], y_hat[1], 0.0])}
    p1 = {sd: pad0[sd] + out_w[sd] * REACH_OUT + np.array([0, 0, REACH_UP]) for sd in pad0}
    p2 = {sd: p3[sd] + np.array([0, 0, REACH_OVER]) for sd in p3}
    R0 = {sd: mat(sim.body(f"{sd}_wrist_yaw_link")[1]) for sd in ("left", "right")}
    t_reach0 = T[0]

    def reach_ref(t):
        u = float(np.clip((t - t_reach0) / REACH_S, 0, 1))
        out = dict(pel_hold)
        for sd in ("left", "right"):
            pad = (1 - u) ** 3 * pad0[sd] + 3 * (1 - u) ** 2 * u * p1[sd] + 3 * (1 - u) * u * u * p2[sd] + u ** 3 * p3[sd]
            pitch = REACH_PITCH0_DEG + (PALM_PITCH_DEG - REACH_PITCH0_DEG) * u
            R = rot_z(psi) @ rot_y(np.radians(pitch))
            out[f"{sd}_wrist_yaw_link"] = wrist_from_pad(pad, R)
        return out

    for _ in range(int(REACH_S / DT)):
        if tick(reach_ref):
            return finish("[reach] fell")
    e_h = {sd: sim.pad(sd) - p3[sd] for sd in ("left", "right")}
    m["hover_err_cm"] = {sd: [round(float(v) * 100, 1) for v in e_h[sd]] for sd in e_h}
    so_h = measure_stand(sim)[0]
    log.append(f"hover err open-loop (cm): " + " ".join(f"{sd} {np.round(e_h[sd] * 100, 1)}" for sd in e_h) + f" | pelvis so {so_h:.3f} (stand was {so:.3f})")
    servo_on[0] = True
    hover_ref = lambda t: reach_ref(t_reach0 + REACH_S)
    for _ in range(int(HOVER_SETTLE_S / DT)):
        if tick(hover_ref):
            return finish("[hover] fell")
    e_h = {sd: sim.pad(sd) - p3[sd] for sd in ("left", "right")}
    m["hover_err_servo_cm"] = {sd: [round(float(v) * 100, 1) for v in e_h[sd]] for sd in e_h}
    log.append(f"hover err after {HOVER_SETTLE_S:.0f} s servo (cm): " + " ".join(f"{sd} {np.round(e_h[sd] * 100, 1)}" for sd in e_h)
               + " | corr cm " + " ".join(f"{k[:5]} {np.round(v * 100, 1)}" for k, v in corr.items()))
    # --- descend: reference goes down until contact stall / floor ------------------
    R_palm = rot_z(psi) @ rot_y(np.radians(PALM_PITCH_DEG))
    floor_dz = -GRIP_BELOW
    t_d0 = T[0]
    hist = []
    trigger = None

    def desc_ref(t):
        dz = max(HOVER_ABOVE - DESCEND_SPEED * (t - t_d0), floor_dz)
        out = dict(pel_hold)
        for sd in ("left", "right"):
            out[f"{sd}_wrist_yaw_link"] = wrist_from_pad(wall_point(sim, sd, depth, dz, ACROSS_BIAS), R_palm)
        return out

    while T[0] - t_d0 < DESCEND_TIMEOUT:
        if tick(desc_ref):
            return finish("[descend] fell")
        e = {}
        for sd in ("left", "right"):
            d = sim.pad(sd) - wall_point(sim, sd, depth, 0.0)
            e[sd] = np.array([float(d[:2] @ n), float(d[:2] @ y_hat), float(d[2])])
        hist.append((T[0], e))
        dz_ref = max(HOVER_ABOVE - DESCEND_SPEED * (T[0] - t_d0), floor_dz)
        across_ok = all(abs(e[sd][1]) <= ACROSS_TOL + 0.005 for sd in e)
        at_top = all(e[sd][2] <= 0.005 for sd in e)
        if dz_ref > floor_dz and across_ok and at_top and len(hist) > int(CONTACT_STALL_S / DT):
            t_old, e_old = hist[-1 - int(CONTACT_STALL_S / DT)]
            stalled = abs(np.mean([e[sd][2] for sd in e]) - np.mean([e_old[sd][2] for sd in e_old])) < CONTACT_STALL_MM / 1000
            if stalled:
                trigger = "contact"; break
        if dz_ref <= floor_dz and across_ok and at_top:
            trigger = "floor"; break
    m["close_trigger"] = trigger
    m["stand_off_close"] = measure_stand(sim)[0]
    m["seat_err_cm"] = {sd: [round(float(v) * 100, 1) for v in e[sd]] for sd in e}
    log.append(f"close trigger {trigger} after {T[0] - t_d0:.1f} s; seat err (cm): " + " ".join(f"{sd} {np.round(e[sd] * 100, 1)}" for sd in e))
    if trigger is None:
        return finish("[descend] never seated")
    # --- close ------------------------------------------------------------------------
    hold_ref = desc_ref
    t_c0 = T[0]
    for _ in range(int((CLOSE_S + CLOSE_HOLD_S) / DT)):
        a = smoothstep((T[0] - t_c0) / CLOSE_S)
        sim.grip_target[:] = GRIP_OPEN + a * (GRIP_CLOSED - GRIP_OPEN)
        if tick(hold_ref):
            return finish("[close] fell")
    # --- lift ---------------------------------------------------------------------------
    crate_pc, _ = sim.crate()
    lift0 = {sd: sim.pad(sd).copy() for sd in ("left", "right")}
    t_l0 = T[0]

    def lift_ref(t):
        z = LIFT_HEIGHT * smoothstep((t - t_l0) / LIFT_S)
        out = dict(pel_hold)
        for sd in ("left", "right"):
            out[f"{sd}_wrist_yaw_link"] = wrist_from_pad(lift0[sd] + np.array([0, 0, z]), R_palm)
        return out

    for _ in range(int((LIFT_S + LIFT_HOLD_S) / DT)):
        if tick(lift_ref):
            return finish("[lift] fell")
    cp, cq = sim.crate()
    rise = float(cp[2] - crate_pc[2]); tilt = crate_tilt_deg(cq)
    m.update(rise_cm=rise * 100, tilt_deg=tilt)
    log.append(f"lift: rose {rise * 100:.1f} cm, tilt {tilt:.1f} deg, pads " + " ".join(f"{sd} {np.round((sim.pad(sd) - lift0[sd]) * 100, 1)}" for sd in lift0))
    if rise < LIFT_MIN:
        return finish(f"[lift] rose {rise * 100:.1f} cm (< {LIFT_MIN * 100:.0f})")
    if tilt > TILT_OK_DEG:
        return finish(f"[lift] tilt {tilt:.1f} deg")
    if m["why"]:
        return finish()
    m["ok"] = True
    return finish()


# --- reach envelope sweep ---------------------------------------------------------------
def run_envelope(sim: Sim, pol: Policy, out_dir, video):
    """Robot standing alone (crate moved away). Both wrists are sent to a grid of
    pad targets in the pelvis frame; each held 2 s; delivered pad error and pelvis
    displacement recorded. Answers 'how far / how low can mode 2 reach'."""
    xs = np.arange(0.20, 0.76, 0.05)
    zs = np.array([0.10, 0.0, -0.10, -0.20, -0.30, -0.40, -0.50])
    rows = []
    sim.reset(np.array([-1.5, 0.0]), 0.0, 0.0)
    sim.d.qpos[sim.crate_q:sim.crate_q + 3] = [3.0, 3.0, 0.8]      # crate out of the way
    mujoco.mj_forward(sim.m, sim.d)
    pol.reset(sim.state()); refs = Refs(pol, sim)
    p0 = sim.d.qpos[:2].copy(); yaw = 0.0
    pel = refs.pelvis(p0, yaw)
    ref = lambda t: {**pel, **refs.hands_at_side(np.array([p0[0], p0[1], PELVIS_Z]), yaw)}
    for _ in range(int(1.5 / DT)):
        pol_step(sim, pol, refs, ref)
    R_palm = rot_y(np.radians(PALM_PITCH_DEG))
    ffm = None
    if video:
        ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                                "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                                os.path.join(out_dir, "envelope.mp4")], stdin=subprocess.PIPE)
    k = 0
    for z in zs:
        for x in xs:
            tg = {sd: np.array([p0[0] + x, p0[1] + (0.20 if sd == "left" else -0.20), PELVIS_Z + z]) for sd in ("left", "right")}
            ref = lambda t, tg=tg: {**pel, **{f"{sd}_wrist_yaw_link": wrist_from_pad(tg[sd], R_palm) for sd in tg}}
            for _ in range(int(2.0 / DT)):
                pol_step(sim, pol, refs, ref)
                k += 1
                if ffm is not None and k % 2 == 0:
                    ffm.stdin.write(sim.frame().tobytes())
                if sim.fallen():
                    break
            err = {sd: sim.pad(sd) - tg[sd] for sd in tg}
            pel_d = sim.d.qpos[:3] - np.array([p0[0], p0[1], PELVIS_Z])
            torso_pitch = float(np.degrees(np.arcsin(-mat(sim.body("torso_link")[1])[2, 0])))
            row = dict(x=float(x), z=float(z), err_cm={sd: [round(float(v) * 100, 1) for v in err[sd]] for sd in err},
                       err_norm_cm=float(np.mean([np.linalg.norm(err[sd]) for sd in err]) * 100),
                       pelvis_dxyz_cm=[round(float(v) * 100, 1) for v in pel_d], torso_pitch_deg=torso_pitch, fallen=bool(sim.fallen()))
            rows.append(row)
            print(f"x {x:.2f} z {z:+.2f}: err {row['err_norm_cm']:5.1f} cm  pelvis d {row['pelvis_dxyz_cm']}  torso pitch {torso_pitch:+.0f}  {'FELL' if row['fallen'] else ''}", flush=True)
            if sim.fallen():
                break
        if sim.fallen():
            # stand the robot up again for the next height row
            sim.reset(np.array([-1.5, 0.0]), 0.0, 0.0); sim.d.qpos[sim.crate_q:sim.crate_q + 3] = [3.0, 3.0, 0.8]
            mujoco.mj_forward(sim.m, sim.d); pol.reset(sim.state())
            ref = lambda t: {**pel, **refs.hands_at_side(np.array([p0[0], p0[1], PELVIS_Z]), yaw)}
            for _ in range(int(1.5 / DT)):
                pol_step(sim, pol, refs, ref)
    if ffm is not None:
        ffm.stdin.close(); ffm.wait()
    json.dump(rows, open(os.path.join(out_dir, "envelope.json"), "w"), indent=1)
    # table: mean pad error (cm) by (z, x)
    print("\nmean pad error (cm), rows z rel. pelvis, cols x rel. pelvis")
    print("      " + " ".join(f"{x:5.2f}" for x in xs))
    for z in zs:
        vals = [next((r["err_norm_cm"] for r in rows if abs(r["x"] - x) < 1e-6 and abs(r["z"] - z) < 1e-6), float("nan")) for x in xs]
        print(f"{z:+.2f} " + " ".join(f"{v:5.1f}" for v in vals))


_T = [0.0]


def pol_step(sim, pol, refs, ref):
    st = sim.state()
    sim.step(pol(st, refs.futures(ref, _T[0])))
    _T[0] += DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/videos/scalebfm_spike"))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--envelope", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "trace"), exist_ok=True)
    pol = Policy()
    sim = Sim(pol.meta, video=a.video)
    if a.envelope:
        run_envelope(sim, pol, a.out, a.video)
        return
    rng = np.random.default_rng(a.seed)
    mf = open(os.path.join(a.out, "manifest.jsonl"), "a")
    oks = 0
    for r in range(a.rounds):
        t0 = time.time()
        m = run_round(sim, pol, r, rng, a.out, a.video)
        oks += m["ok"]
        print(f"round {r}: {'OK' if m['ok'] else '--'} {m['why']} | so {m.get('stand_off_meas', float('nan')):.3f} lat {m.get('stand_lat', float('nan')):+.3f} "
              f"yaw {m.get('yaw_err_deg', float('nan')):+.1f} | trig {m.get('close_trigger')} rise {m.get('rise_cm', float('nan')):.1f} tilt {m.get('tilt_deg', float('nan')):.1f} "
              f"| {m['len_s']:.0f} s sim, {time.time() - t0:.0f} s wall", flush=True)
        for line in m.get("log", []):
            print("   ", line, flush=True)
        mf.write(json.dumps(m) + "\n"); mf.flush()
    print(f"OK {oks}/{a.rounds}")


if __name__ == "__main__":
    main()
