"""selfmodel_xfer_20261125: data3's reaches executed in another simulator's body, recorded in data3's layout, so the
self models trained in LocoManiLab (selfmodel_train3 / train3b) can be scored there offline (2026-11-25).

    R=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $R
    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    $PY sx_collect.py calib --controller scalebfm                 # conventions: pelvis offset, three G1 models' FK
    $PY sx_collect.py lag   --controller scalebfm                 # the controller's real lag in this body (d3's ramps)
    $PY sx_collect.py drive --controller scalebfm --out DIR --resets 400 --shards 16 --shard K --delay-f N
    $PY sx_collect.py merge --out DIR;  $PY sx_collect.py check --out DIR;  $PY sx_collect.py summary --out DIR

The SIMPLE body (`host simple`): gen_reach.py's SIMPLE configuration through selfwm_20261029/env.py -- the scene
outputs/scene/simple_g1_sonic (NVIDIA's G1 + Dex3, SIMPLE's g1_sonic robot), the controller in process behind our Motor
(ScaleBFM: keypoint futures in mode 4, ankles pinned at each command; SONIC v1.1: SonicMotor's VR 3-point targets and its
planner, networks on the CPU), BFM_ROOT=odo (the controller is told the pelvis position by leg odometry), and
REACH_HAND_OFF=0.13,0,0 (the hand point is data3's: 13 cm ahead of wrist_yaw_link, LocoManiLab's PALM_OFFSET; SIMPLE's own
datasets used the palm, 0.0415).

What is data3's (selfmodel_data3_20261125/d3_collect.py, the frozen copy next to this file's parent: its functions are
called, not copied): the per-reset draws (reset_plan with walk_ok False, so a reset never walks but draws exactly what
data3 draws: hand mode right / left / both 40 / 40 / 20 %, standing 45 % else pelvis 0.52-0.76 m, the target from
d3_reach_table.json with CORE / WIDE, wrist yaw +-90, pitch -30..90, roll +-90 deg), two reaches a reset, the settle rule
(pelvis and both hand points slower than 2 cm/s for 0.5 s, displacement over 0.1 s, at most 3 s, the wait opening at the
reference end plus the controller's delay, the first at 0.2 s), the fall / lost rules, the record (segments(): S, W, J,
C, HQ, KP and every per-segment field, data/ one-hand, data_both/ two-hand, arm.npz) and merge / check / summary.
run_reset_x below is d3_collect.run_reset with the walk branch removed and two hooks: the planner and the body's
reference (follow()).

The planner (--planner):
  native  the reach is planned as SIMPLE plans it: Motor.reach's whole-body IK (on the walk_mars G1, the model Motor
          plans on; measured pelvis and joints, its own seed, weights 1 / 0.3, one or both hands, 0.15 m/s, 3 cm
          waypoints, 0.6 s minimum), from the reference pose at the command, no xy shift (d_this = d_prev = 0).
  lml     d3_collect.plan_reach (LocoManiLab's IK on robot_fk.xml, protocol 2, shifted in x/y onto the reference root).
Either way the IK poses become 50 Hz frames by d3's min-jerk blend (L.blend_frames, the same interpolation as
Motor._blend) and the controller follows them through Motor's own interface (FrameRef: the frame at the Motor's clock,
joint velocity the forward difference; ScaleBFM reads ref(t + j dt) for its futures, SONIC ref(t)).
The stance (--stance): native = the Motor's standing pose after its reset (SIMPLE's default joints, measured pelvis);
lml = data3's stance (LocoManiLab's default pose at the measured pelvis xy / heading, pelvis 0.8 m).

The controller (--controller): CONTROLLERS maps a name to the environment of env.Env (the motor it builds). A new
controller (HEFT, selfmodel_xfer revision 1) plugs in as an entry whose Env.m offers what FrameRef drives: reset(xy, yaw),
tick() -> bool, k, t, ref (callable t -> Pose), _stand(pose), sim (m, d, fallen) and, for --planner native, reach() and ik.

Added to data3's record: d_this / d_prev (3, world: the xy shift this / the previous plan put on the reference; 0 with
--planner native and the stance), so a reach's reference start is X = C[k0] (world) - (d_this - d_prev) for train3b's MX.
run.json: controller, planner, stance, delay, the conventions measured by calib, the code's md5.
Seeds 79000000 + i, reset ids 40000000 + i (data3 73000000 / 20000000, data3b 77000000 / 30000000).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D3_DIR = Path(os.environ.get("SX_D3", HERE.parent / "selfmodel_data3_20261125"))
REPO = Path(os.environ.get("SX_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(D3_DIR))
import d3_collect as D3  # noqa: E402

L = D3.L
FPS, TICKS, DT = L.FPS, L.TICKS, L.DT
SEED, ID0 = 79_000_000, 40_000_000
CONTROLLERS = {  # env.Env's switches for each controller in the SIMPLE body (gen_reach's datasets and data_via_sonic)
    "scalebfm": dict(REACH_CONTROLLER="scalebfm", REACH_BFM_DEVICE=os.environ.get("SX_BFM_DEVICE", "cpu")),
    "sonic": dict(REACH_CONTROLLER="sonic", REACH_SONIC_NETS="allcpu", REACH_SONIC_THREADS="1"),
    # selfmodel_xfer revision 1, combination 6 (2026-11-25): HEFT in process (backends.heft: g1_backend's observation and
    # ONNX, the scene's joint dynamics, HeftConfig reset_pose "lml"); its reference read from the motor's clock on
    # (REACH_HEFT_DELAY 0, no buffer delay, as ScaleBFM and SONIC on this bed). HEFT reads no pelvis position, so BFM_ROOT
    # (odometry) only integrates alongside.
    "heft": dict(REACH_CONTROLLER="heft", REACH_HEFT_DELAY="0"),
}
SIMPLE_BODY = dict(REACH_SCENE_DIR=str(REPO / "outputs" / "scene" / "simple_g1_sonic"), REACH_HAND_OFF="0.13,0,0", BFM_ROOT="odo")


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# -- the SIMPLE body as a d3 host ------------------------------------------------------------------------------------
class FrameRef:
    """Motor's reference from 50 Hz qpos frames (root xyz, wxyz, 29 joints in MuJoCo order): the frame at the clock."""

    def __init__(self, frames: np.ndarray, Pose):
        self.F, self.Pose = np.asarray(frames, np.float64), Pose

    def __call__(self, t: float):
        i = min(max(int(round(t / DT)), 0), len(self.F) - 1)
        a, b = self.F[i], self.F[min(i + 1, len(self.F) - 1)]
        q = a[3:7] / np.linalg.norm(a[3:7])
        return self.Pose(a[:3].copy(), q, a[7:].copy(), (b[7:] - a[7:]) / DT)


class NativeFK(L.Planner):
    """L.Planner's fk_world (wrist_yaw_link and the hand point 13 cm ahead) on another G1 model (Motor's IK model)."""

    def __init__(self, ik):
        import mujoco

        self.ik, self.mj = ik, mujoco
        m = ik.model
        self.wrist = {s: m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}
        self.feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        self.stand_z = 0.8
        self.d = mujoco.MjData(m)  # private: the IK solver keeps ik.data


class SimpleHost:
    """d3_collect's host interface (m, d, mj, c, pl, to_mj, qadr_mj, ref_delay_f, reset, start_frames, sample,
    pelvis_pose) on gen_reach's SIMPLE body; follow(frames) / step(f) drive the controller's Motor."""

    def __init__(self, controller: str, stance: str = "native", delay_f: int = 0):
        assert controller in CONTROLLERS, f"unknown controller {controller}; known: {sorted(CONTROLLERS)}"
        os.environ.update(SIMPLE_BODY)
        os.environ.update(CONTROLLERS[controller])
        sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "experiments" / "selfwm_20261029"), str(REPO / "scripts" / "experiments" / "reachwm_20261030")]
        import env as E
        import mujoco

        from locomani.backends.scalebfm import motor as MO
        from locomani.backends.scalebfm import runtime as RT
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.controller, self.stance, self.mj, self.MO, self.RT = controller, stance, mujoco, MO, RT
        self.env = E.Env("open", controller=controller, root="odo")
        self.mot = self.env.m
        assert np.allclose(RT.OFF, L.HAND_OFF), f"REACH_HAND_OFF did not take: OFF {RT.OFF}"
        sim = self.env.sim
        self.m, self.d = m, d = sim.m, sim.d
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.qadr = np.array([m.jnt_qposadr[jid(n)] for n in L.SCALEBFM_ORDER])
        self.vadr = np.array([m.jnt_dofadr[jid(n)] for n in L.SCALEBFM_ORDER])
        self.qadr_mj = np.array([m.jnt_qposadr[jid(n)] for n in MUJOCO_JOINT_NAMES])
        fj = jid("floating_base_joint")
        assert m.jnt_type[fj] == mujoco.mjtJoint.mjJNT_FREE
        self.rq, self.rv = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
        self.pelvis = bid("pelvis")
        assert int(m.jnt_bodyid[fj]) == self.pelvis
        self.wrist = {s: bid(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        self.torso = bid("torso_link")
        self.to_mj = np.array([L.SCALEBFM_ORDER.index(n) for n in MUJOCO_JOINT_NAMES])
        assert np.array_equal(self.qadr[self.to_mj], self.qadr_mj)
        self.pl = L.Planner()  # LocoManiLab's FK model (robot_fk.xml): plan_reach (--planner lml) and the checks
        self.nfk = NativeFK(self.mot.ik) if getattr(self.mot, "ik", None) is not None else None
        self.ref_delay_f = int(delay_f)
        self.frames = None
        self.mot.reset((0.0, 0.0), 0.0)
        self.c = self._calibrate()

    # -- conventions --------------------------------------------------------------------------------------------------
    def _calibrate(self) -> np.ndarray:
        """lml_host.HostDriver._calibrate: the pelvis offset c between this body's pelvis frame and robot_fk's, from both
        wrists (positions and attitudes) at the joints' zero, where the two G1 versions' arm chains agree."""
        mj, m = self.mj, self.m
        d = mj.MjData(m)  # the live pelvis, every joint at 0: the two G1 versions' waists differ (calib), so a bent
        d.qpos[:] = self.d.qpos  # waist would put that difference into c
        d.qpos[self.qadr_mj] = 0.0
        mj.mj_kinematics(m, d)
        Rp, pp = d.xmat[self.pelvis].reshape(3, 3), d.xpos[self.pelvis]
        out = self.pl.fk_world(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), d.qpos[self.qadr_mj].copy())
        cs, rot = [], []
        for s, b in self.wrist.items():
            cs.append(Rp.T @ (d.xpos[b] - pp) - (out[s][0] - out[s][1] @ L.HAND_OFF))
            Rw = Rp.T @ d.xmat[b].reshape(3, 3)
            rot.append(float(np.degrees(np.arccos(np.clip((np.trace(Rw.T @ out[s][1]) - 1) / 2, -1, 1)))))
        c = np.mean(cs, axis=0)
        self.calib = dict(c=c.tolist(), spread_m=max(float(np.linalg.norm(x - c)) for x in cs), wrist_rot_deg=max(rot))
        assert self.calib["spread_m"] < 2e-3 and self.calib["wrist_rot_deg"] < 0.1, f"body vs robot_fk: {self.calib}"
        return c

    def pelvis_pose(self) -> tuple[np.ndarray, np.ndarray]:
        R = self.d.xmat[self.pelvis].reshape(3, 3).copy()
        return self.d.xpos[self.pelvis] + R @ self.c, R

    def sample(self) -> dict:
        d = self.d
        p, R = self.pelvis_pose()
        return dict(root=np.r_[p, _mat_quat(R)], q=d.qpos[self.qadr].copy(),
                    dq=d.qvel[self.vadr].copy(), w=d.qvel[self.rv + 3 : self.rv + 6].copy(),
                    wr={s: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for s, b in self.wrist.items()},
                    torso=(d.xpos[self.torso].copy(), d.xmat[self.torso].reshape(3, 3).copy()))  # fmt: skip

    # -- the reference --------------------------------------------------------------------------------------------------
    def pose(self, fr: np.ndarray):
        """A frame (robot_fk's pelvis convention) as the Motor's Pose (its pelvis = the free joint)."""
        R = L._qmat(fr[3:7])
        return self.MO.Pose(np.asarray(fr[:3], float) - R @ self.c, np.asarray(fr[3:7], float), np.asarray(fr[7:], float), np.zeros(29))

    def follow(self, frames: np.ndarray, repin_at: int | None = None) -> None:
        """The controller follows `frames` from its next tick on; repin_at: re-pin the ankles where they stand (mode 4)
        first, as Motor.reach does at every command (Motor._stand with a pose: the odometry keeps its anchor)."""
        F = np.asarray(frames, np.float64).copy()
        if np.any(self.c):
            for i in range(len(F)):
                F[i, :3] -= L._qmat(F[i, 3:7]) @ self.c
        if repin_at is not None:
            self.mot._stand(self.MO.Pose(F[repin_at, :3], F[repin_at, 3:7], F[repin_at, 7:], np.zeros(29)))
        self.frames = frames
        self.mot.ref = FrameRef(F, self.MO.Pose)

    def step(self, f: int) -> bool:
        assert self.mot.k == f, f"motor clock {self.mot.k} != frame {f}"
        return bool(self.mot.tick())

    def reset(self) -> None:
        self.mot.reset((0.0, 0.0), 0.0)  # Motor.reset / SonicMotor.reset: stand, policy history, odometry anchor
        self.frames = None

    def start_frames(self) -> np.ndarray:
        x = self.sample()
        if self.stance == "lml":  # data3's: LocoManiLab's default pose at the measured pelvis xy and heading
            yaw = L.yaw_of(L._qmat(x["root"][3:]))
            start = (np.r_[x["root"][:2], self.pl.start_pose()[0][2]], _quat_yaw(yaw), L.DEFAULT_Q.copy())
        else:  # the Motor's own standing pose after its reset (Motor._stand: measured pelvis, heading, joints)
            p = self.mot.ref(0.0)
            R = L._qmat(p.quat)
            start = (np.asarray(p.pos, float) + R @ self.c, np.asarray(p.quat, float), np.asarray(p.joints, float))
        return L.hold(start, L.START_HOLD_S + L.MARGIN_S + L.PRE_S)


def _mat_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# -- planners ---------------------------------------------------------------------------------------------------------
def command_world(x: dict, cmd: dict) -> dict:
    """d3_collect.plan_reach's placement: the command in the measured pelvis heading frame at the command row."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    hands = {}
    for s, c in cmd["hands"].items():
        tw = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
        rw = L.rot_z(yaw + c["dyaw"]) @ L.rot_y(c["pitch"]) @ D3.rot_x(c["roll"])
        hands[s] = dict(target_w=tw, rot_w=rw, dyaw=c["dyaw"], pitch=c["pitch"], roll=c["roll"], core=c["core"],
                        cmd_target=np.asarray(c["target"], float))  # fmt: skip
    return hands


def plan_native(h: SimpleHost, x: dict, ref_frame: np.ndarray, cmd: dict):
    """Motor.reach's plan (its whole-body IK, waypoints and durations), not run: its _run is stubbed and its _blend
    records the poses. Returns d3_collect.plan_reach's (blend, last, rec) with no xy shift (delta 0)."""
    mot = h.mot
    hands = command_world(x, cmd)
    cap = {}
    blend0 = mot._blend

    def blend(poses, durations):
        cap["poses"], cap["durs"] = list(poses), list(durations)
        return blend0(poses, durations)

    saved = {k: mot.__dict__[k] for k in ("_blend", "_run") if k in mot.__dict__}
    mot._blend, mot._run = blend, (lambda name, update, timeout: None)
    t0 = time.time()
    try:
        if len(hands) == 1:
            (s,) = hands
            mot.reach(s, hands[s]["target_w"], hands[s]["rot_w"], pelvis_z=cmd["pelvis_z"], servo=False)
        else:
            mot.reach("both", (hands["left"]["target_w"], hands["right"]["target_w"]),
                      (hands["left"]["rot_w"], hands["right"]["rot_w"]), pelvis_z=cmd["pelvis_z"], servo=False)  # fmt: skip
    finally:  # back to what the motor had (the class's methods)
        for k in ("_blend", "_run"):
            mot.__dict__.pop(k, None)
        mot.__dict__.update(saved)
    plan_s = time.time() - t0
    poses, durs = cap["poses"], cap["durs"]
    # the plan starts at the reference pose at the command (Motor: ref(t) one tick later, the same frame in a hold)
    p0 = poses[0]
    R0 = L._qmat(p0.quat)
    first = np.r_[np.asarray(p0.pos) + R0 @ h.c, p0.quat, p0.joints]
    gap = float(np.abs(first - ref_frame).max())
    tup = [(np.asarray(q.pos, float) + L._qmat(q.quat) @ h.c, np.asarray(q.quat, float), np.asarray(q.joints, float)) for q in poses]
    ref_pose = (ref_frame[:3].astype(float), ref_frame[3:7].astype(float), ref_frame[7:].astype(float))
    shifted = [ref_pose] + tup[1:]
    last = tup[-1]
    for s in hands:  # the last waypoint's hand point to the target on the model planned on (Motor's IK model)
        hands[s]["ik_err"] = float(np.linalg.norm(h.nfk.fk_world(last[0] - L._qmat(last[1]) @ h.c, last[1], last[2])[s][0] - hands[s]["target_w"]))
    rec = dict(mode=cmd["mode"], hands=hands, pelvis_z=cmd["pelvis_z"], reach_s=float(sum(durs)), plan_s=plan_s,
               ref_q=last[2].copy(), ref_pel=np.r_[last[0], last[1]], delta=np.zeros(3), RT=np.r_[0.0, np.cumsum(durs)],
               RQ=np.array([pp[2] for pp in tup]), RP=np.array([np.r_[pp[0], pp[1]] for pp in tup]), start_gap=gap)  # fmt: skip
    return L.blend_frames(shifted, durs), last, rec


def plan(h: SimpleHost, planner: str, x: dict, ref_frame: np.ndarray, cmd: dict):
    if planner == "lml":
        return D3.plan_reach(h, x, ref_frame, cmd)
    return plan_native(h, x, ref_frame, cmd)


# -- one reset: d3_collect.run_reset, in place, the planner and the body's reference as hooks ------------------------------
def run_reset_x(h: SimpleHost, rig, plan_d: dict, planner: str):
    t_wall = time.time()
    h.reset()
    rig.clear()
    frames = h.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(D3.HOLD_S * FPS), 0)])
    h.follow(frames)
    rows, reaches, slots = {}, [], []
    fell = lost = None
    why, f = "", 0
    assert not (plan_d["walk"] and plan_d["walk"]["done"]), "selfmodel_xfer runs in place"
    stage, k, open_f = "wait", 0, D3.FIRST_OPEN_F
    anchor, cur_delta, active = None, np.zeros(3), None
    winfo = dict(asked=None, real=None, dev_cm=0.0, plan=None)
    delay, plan_s, gaps = h.ref_delay_f, 0.0, []
    off0 = None
    while True:
        if not h.step(f):  # the Motor refuses a fallen robot (pelvis below 0.40 m); d3's 0.45 m fires first
            fell, why = f, "height"
            break
        if off0 is None and hasattr(getattr(h, "mot", None), "offset"):
            off0 = np.array(h.mot.offset, float)  # HeftMotor's xy offset once our frames are fed (its reset's settle drift)
        x = rows[f] = h.sample()
        rig.record(x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, "height"
            break
        if anchor is not None:
            fr = frames[min(max(f - delay, anchor[2]), len(frames) - 1)]
            dev = float(np.linalg.norm((x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])))
            if dev > L.LOST_M:
                fell, lost, why = f, f, "lost"
                break
        act = None
        if stage in ("wait", "label"):
            on_grid = active is None or (f - active["f_cmd"]) % TICKS == 0
            if f >= open_f + D3.SETTLE_F and on_grid and rig.quiet():
                act = "settled"
            elif f >= open_f + D3.SETTLE_MAX_F and on_grid:
                act = "timeout"
        if act in ("settled", "timeout"):
            if active is not None:
                active.update(f_label=f, settled_end=act == "settled", settle_s=(f - active["f_end"]) / FPS)
                active = None
            if act == "timeout":
                if k < D3.CHAIN:
                    slots.append(dict(slot=k, mode=plan_d["cmds"][k]["mode"], status="unsettled", wait_s=(f - open_f) / FPS))
                    k += 1
                open_f = f
                frames = np.concatenate([frames[: f + 1], np.repeat(frames[f : f + 1], int(D3.HOLD_S * FPS), 0)])
                h.follow(frames)
            if k >= D3.CHAIN:
                break
            if act == "settled":
                cmd = plan_d["cmds"][k]
                try:
                    blend, last, rec = plan(h, planner, x, frames[f], cmd)
                except (np.linalg.LinAlgError, ValueError) as e:
                    why = f"ik {type(e).__name__}: {e}"
                    break
                plan_s += rec["plan_s"]
                gaps.append(rec.get("start_gap", 0.0))
                rec.update(slot=k, chain=len(reaches), f_cmd=f, f_end=f + len(blend), pre_wait_s=(f - open_f) / FPS,
                           d_prev=cur_delta.copy(), f_label=None, settled_end=False, settle_s=-1.0)  # fmt: skip
                slots.append(dict(slot=k, mode=cmd["mode"], status="done", wait_s=(f - open_f) / FPS))
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, D3.HOLD_S)])
                h.follow(frames, repin_at=f + 1)
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                cur_delta = rec["delta"]
                reaches.append(rec)
                active = rec
                k += 1
                open_f = rec["f_end"] + delay
                stage = "label"
        if f >= len(frames) - 1:
            why = "motion ended"
            break
        f += 1
    while k < D3.CHAIN:
        slots.append(dict(slot=k, mode=plan_d["cmds"][k]["mode"], status="not reached"))
        k += 1
    info = dict(fell=fell, lost=lost, why=why, steps=len(rows), walked=False, walk=winfo, slots=slots, planned=len(reaches),
                wall_s=time.time() - t_wall, plan_s=plan_s, delay_f=delay, start_gap_max=max(gaps, default=0.0))  # fmt: skip
    if off0 is not None:  # HeftMotor adds an xy offset to the frames it feeds HEFT, to keep them continuous: set by its
        # reset (the settle's drift; HEFT reads only displacements between its frames, so a constant offset is invisible
        # to it); any change after our first frame would mean a frame jump it absorbed (expected 0)
        info["heft_offset_cm"] = 100 * float(np.linalg.norm(off0))
        info["heft_offset_change_cm"] = 100 * float(np.linalg.norm(np.asarray(h.mot.offset, float) - off0))
    return rows, info, frames, reaches


# -- the record -----------------------------------------------------------------------------------------------------
class SRig(D3.Rig):
    """d3_collect.Rig on a scene whose bodies carry no robosuite prefix (robot0_)."""

    def __init__(self, h):  # d3_collect.Rig.__init__ with the names unprefixed and the free joint found by type
        from collections import deque

        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        mj, m = h.mj, h.m
        self.h, self.mj = h, mj
        bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        jad = lambda n: int(m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)])  # noqa: E731
        self.links = [s[0] for s in D3.SPHERES]
        self.radius = np.array([s[2] for s in D3.SPHERES], np.float32)
        self.body, self.local = {}, {}
        for side in D3.SIDES:
            mir = np.array([1.0, 1.0 if side == "right" else -1.0, 1.0])
            ids, loc = [], []
            for link, c, _ in D3.SPHERES:
                b = bid(f"{side}_wrist_yaw_link" if link == "hand_palm_link" else f"{side}_{link}")
                assert b >= 0, f"{side}_{link} is not a body of the scene"
                ids.append(b)
                loc.append((D3.PALM_OFF[side] if link == "hand_palm_link" else 0.0) + np.array(c) * mir)
            self.body[side], self.local[side] = np.array(ids), np.array(loc)
        self.fing = {s: np.array([jad(f"{s}_hand_{f}_joint") for f in D3.FINGERS]) for s in D3.SIDES}
        self.qadr_mj = np.array([jad(n) for n in MUJOCO_JOINT_NAMES])
        self.rq, self.pelvis = h.rq, h.pelvis
        self.scratch = mj.MjData(m)
        self.c = np.asarray(h.c, float)
        self.tip = {}
        for side in D3.SIDES:
            for f in ("thumb_2", "index_1"):
                b = bid(f"{side}_hand_{f}_link")
                g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
                g = max(g, key=lambda k: int(m.geom_contype[k]))
                mid = int(m.geom_dataid[g])
                V = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                Rg = np.zeros(9)
                mj.mju_quat2Mat(Rg, m.geom_quat[g])
                Vb = V @ Rg.reshape(3, 3).T + m.geom_pos[g]
                # the link's direction: body origin -> mesh centre (d3 uses geom_pos, which is 0 for these meshes here)
                u = Vb.mean(0) if np.linalg.norm(m.geom_pos[g]) < 1e-9 else m.geom_pos[g]
                u = u / np.linalg.norm(u)
                self.tip[(side, f)] = (b, Vb[int(np.argmax(Vb @ u))].copy())
        self._q = np.zeros(4)
        self.v = deque(maxlen=D3.SETTLE_F)
        self.pts = deque(maxlen=TICKS + 1)


def add_fields(segs: list[dict], reaches: list[dict]) -> None:
    """d_this / d_prev (world xy shift of this / the previous plan, z 0) per segment (reach found by chain)."""
    for sg in segs:
        rc = reaches[sg["chain"]]
        sg["d_this"] = np.asarray(rc["delta"], np.float32)
        sg["d_prev"] = np.asarray(rc["d_prev"], np.float32)


def save_chunk(segs: list[dict], d: Path, name: str, rig) -> None:
    """d3_collect.save_chunk plus d_this / d_prev."""
    (d / "arm").mkdir(parents=True, exist_ok=True)
    if not segs:
        return
    z, a = D3.pack(segs)
    z["d_this"] = np.stack([s["d_this"] for s in segs])
    z["d_prev"] = np.stack([s["d_prev"] for s in segs])
    for k in ("touch_n", "touch_z"):  # HumanoidArena (revision 2): scene contact during the reach
        if all(k in s for s in segs):
            z[k] = np.array([s[k] for s in segs], np.float32)
    a.update(radius=rig.radius, link=np.array(rig.links))
    for path, arr in ((d / f"{name}.npz", z), (d / "arm" / f"{name}.npz", a)):
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, path)


def code_md5() -> dict:
    out = {p.name: md5(p) for p in (Path(__file__).resolve(), D3_DIR / "d3_collect.py", D3_DIR / "d3_reach_table.json",
                                    Path(L.__file__).resolve())}  # fmt: skip
    for rel in ("scripts/experiments/selfwm_20261029/env.py", "scripts/rsi/capability_probe.py",
                "scripts/experiments/reachwm_20261030/leg_odo.py", "src/locomani/backends/scalebfm/motor.py",
                "src/locomani/backends/scalebfm/runtime.py", "src/locomani/backends/sonic_onnx/motor.py",
                "src/locomani/backends/sonic_onnx/runtime.py", "src/locomani/backends/heft/motor.py",
                "src/locomani/backends/heft/runtime.py", "src/locomani/backends/heft/obs.py",
                "src/locomani/primitives/whole_body_ik.py",
                "src/locomani/primitives/arm_ik.py", "outputs/scene/simple_g1_sonic/scene_locomani.xml"):  # fmt: skip
        if (REPO / rel).exists():
            out[rel] = md5(REPO / rel)
    return out


def drive(a) -> None:
    ids = [int(v) for v in a.ids.split(",")] if a.ids else list(range(a.shard, a.resets, a.shards))
    n = min(a.limit or len(ids), len(ids))
    od, ob = a.out / "data", a.out / "data_both"
    for d in (od, ob):
        d.mkdir(parents=True, exist_ok=True)
    chunks = [range(c, min(c + D3.CHUNK, n)) for c in range(0, n, D3.CHUNK)]
    done = lambda c: (od / f"w{a.shard:03d}_{c:03d}_log.json").exists()  # noqa: E731
    before = sum(len(r) for c, r in enumerate(chunks) if done(c))
    tag = f"sx {a.controller}/{a.planner}/{a.stance} shard {a.shard}"
    print(f"{tag}: {before}/{n} resets already driven", flush=True)
    if before == n:
        return
    table = json.loads(Path(a.table).read_text())
    h = SimpleHost(a.controller, a.stance, a.delay_f)
    rig = SRig(h)
    run = dict(controller=a.controller, planner=a.planner, stance=a.stance, delay_f=h.ref_delay_f, seed=a.seed, id0=a.id0,
               resets=a.resets, calib=h.calib, body=SIMPLE_BODY, env=CONTROLLERS[a.controller], md5=code_md5(),
               table_md5=md5(Path(a.table)))  # fmt: skip
    rp = a.out / "run.json"
    if a.shard == 0 and not rp.exists():
        rp.write_text(json.dumps(run, indent=1))
    print(f"{tag}: pelvis offset c {np.round(h.c, 4).tolist()} (spread {h.calib['spread_m']:.1e} m, wrist attitude "
          f"{h.calib['wrist_rot_deg']:.3f} deg vs robot_fk), delay {h.ref_delay_f} frames, root check {rig.check_root():.1e} m", flush=True)  # fmt: skip
    t0, nres, nseg, nfall, nlost, nskip, nslot = time.time(), 0, 0, 0, 0, 0, 0
    for c, rng_ids in enumerate(chunks):
        if done(c):
            continue
        segs, segb, log = [], [], []
        for j in rng_ids:
            i = ids[j]
            plan_d = D3.reset_plan(np.random.default_rng(a.seed + i), table, False)
            rows, info, frames, reaches = run_reset_x(h, rig, plan_d, a.planner)
            s = D3.segments(rows, info, frames, reaches, a.id0 + i, h.pl if a.planner == "lml" else h.nfk, rig)
            add_fields(s, reaches)
            segs += [x for x in s if x["hands"] != 2]
            segb += [x for x in s if x["hands"] == 2]
            log.append(dict(reset=a.id0 + i, n_segs=len(s), walk_asked=plan_d["walk"] is not None, **info))
            nres, nseg = nres + 1, nseg + len(s)
            nfall += info["fell"] is not None and info["lost"] is None
            nlost += info["lost"] is not None
            nskip += sum(sl["status"] == "unsettled" for sl in info["slots"])
            nslot += sum(sl["status"] != "not reached" for sl in info["slots"])
            if nres % 5 == 0 or nres == n - before:
                print(f"{tag}: {before + nres}/{n} resets, {nseg} segments, falls {nfall}, lost {nlost}, unsettled {nskip}/{nslot}, "
                      f"{time.time() - t0:.0f} s ({(time.time() - t0) / nres:.1f} s/reset)", flush=True)  # fmt: skip
        name = f"w{a.shard:03d}_{c:03d}"
        save_chunk(segs, od, name, rig)
        save_chunk(segb, ob, name, rig)
        (od / f"{name}_log.json").write_text(json.dumps(log, default=D3._json))
    print(f"{tag} done: {before + nres} resets ({nres} this run, {nseg} segments, falls {nfall}, lost {nlost}, "
          f"unsettled {nskip}/{nslot}), {time.time() - t0:.0f} s", flush=True)  # fmt: skip


# -- phases: calib, lag, check ----------------------------------------------------------------------------------------
def calib(a) -> None:
    """The conventions this record relies on: the body's pelvis offset c vs robot_fk (and the wrist attitudes); robot_fk
    vs the SIMPLE body vs Motor's IK model (walk_mars) on 200 random poses: wrist_yaw_link position and attitude
    relative to the pelvis; the hand point (13 cm) and the Motor's jaw_centre on the live state."""
    h = SimpleHost(a.controller, a.stance, 0)
    print(f"calib {a.controller}: c {np.round(h.c, 5).tolist()}, spread {h.calib['spread_m']:.2e} m, wrist attitude "
          f"{h.calib['wrist_rot_deg']:.4f} deg; R.OFF {h.RT.OFF.tolist()}; IK model {h.mot.ik.model.nbody} bodies")  # fmt: skip
    rng = np.random.default_rng(0)
    lo = np.array([h.m.jnt_range[h.mj.mj_name2id(h.m, h.mj.mjtObj.mjOBJ_JOINT, n)] for n in _mj_names()])
    scr = h.mj.MjData(h.m)
    worst = dict(sim_vs_fk=0.0, ik_vs_fk=0.0, rot_sim=0.0, rot_ik=0.0)
    for _ in range(200):
        q = rng.uniform(lo[:, 0], lo[:, 1])
        scr.qpos[:] = 0
        scr.qpos[h.rq + 3] = 1.0
        scr.qpos[h.qadr_mj] = q
        h.mj.mj_kinematics(h.m, scr)
        fk = h.pl.fk_world(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
        ik = h.nfk.fk_world(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
        for s, b in h.wrist.items():
            Rs = scr.xmat[b].reshape(3, 3)
            ps = scr.xpos[b] + Rs @ L.HAND_OFF - h.c  # relative to robot_fk's pelvis point (the free joint + c)
            worst["sim_vs_fk"] = max(worst["sim_vs_fk"], float(np.linalg.norm(ps - fk[s][0])))
            worst["ik_vs_fk"] = max(worst["ik_vs_fk"], float(np.linalg.norm(ik[s][0] - fk[s][0])))
            worst["rot_sim"] = max(worst["rot_sim"], _ang(Rs, fk[s][1]))
            worst["rot_ik"] = max(worst["rot_ik"], _ang(ik[s][1], fk[s][1]))
    print("  200 random joint sets (pelvis at the origin), hand point and wrist attitude, max over both hands: "
          f"SIMPLE body vs robot_fk {100 * worst['sim_vs_fk']:.3f} cm / {worst['rot_sim']:.4f} deg; Motor IK model (walk_mars) "
          f"vs robot_fk {100 * worst['ik_vs_fk']:.3f} cm / {worst['rot_ik']:.4f} deg")  # fmt: skip
    x = h.sample()
    for s in ("left", "right"):
        pw, Rw = x["wr"][s]
        j = h.mot.jaw_centre(s)[0]
        print(f"  live {s}: hand point (wrist + R 0.13) vs Motor.jaw_centre {100 * np.linalg.norm(pw + Rw @ L.HAND_OFF - j):.4f} cm")
    print(f"  stance native: joints at the reset vs LocoManiLab's default, max |d| "
          f"{np.degrees(np.abs(h.mot.ref(0.0).joints - L.DEFAULT_Q).max()):.1f} deg; pelvis z {h.mot.ref(0.0).pos[2]:.3f} m")


def _mj_names():
    from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

    return MUJOCO_JOINT_NAMES


def _ang(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def lag(a) -> None:
    """d3_collect.lag in this body: one arm joint follows a 1 s min-jerk ramp from the stance, then holds 3 s; the lag
    is the frames between the robot's joint and the reference crossing half (and 90 % of) their own displacement."""
    h = SimpleHost(a.controller, a.stance, 0)
    out = []
    for j, amp in D3.LAG_MOVES:
        h.reset()
        fr = h.start_frames()
        f0, n = len(fr) + 10, FPS
        fr = np.concatenate([fr, np.repeat(fr[-1:], 10 + n + 3 * FPS, 0)])
        s_ = np.arange(n + 1) / n
        fr[f0 : f0 + n + 1, 7 + j] += amp * (10 * s_**3 - 15 * s_**4 + 6 * s_**5)
        fr[f0 + n + 1 :, 7 + j] += amp
        h.follow(fr)
        q = []
        for f in range(len(fr)):
            h.step(f)
            q.append(float(h.d.qpos[h.qadr_mj[j]]))
        q = np.array(q)
        da = q[-1] - q[f0 - 1]
        cross = lambda v, lvl: int(np.argmax((v[f0 - 1 :] - v[f0 - 1]) / (v[-1] - v[f0 - 1]) >= lvl))  # noqa: E731
        out.append((j, cross(q, 0.5) - cross(fr[:, 7 + j], 0.5), cross(q, 0.9) - cross(fr[:, 7 + j], 0.9), round(da / amp, 3)))
    med = float(np.median([o[1] for o in out]))
    print(f"sx lag {a.controller} ({a.stance} stance): per move (joint, lag at 50 % / 90 % in frames, reached share of the "
          f"ramp): {out}; median lag at 50 % {med:.0f} frames ({med / FPS:.2f} s)")  # fmt: skip


def check(a) -> None:
    """d3_collect.check on the record (its ik_err_cm recomputation on the model planned on: robot_fk for --planner lml,
    Motor's walk_mars G1 for native, whose wrist_yaw_link sits 0.5 cm from robot_fk's), then this experiment's:
      (1) the direct error recomputed from W's last row and the target = jaw_err_cm;
      (2) the hand point: W's hands = KP's hand point (both wrist_yaw_link + R (0.13, 0, 0)); = the same point by this
          body's own FK (the scene's MjModel) from the recorded pelvis (W xyz + yaw, S gravity) and joints; and, reported
          only, the same point by robot_fk (data3's FK model: the G1 model versions' kinematic difference, not a
          definition difference);
      (3) the reference features: ref_q / ref_pel / ref_s = the last waypoint (RQ / RP) and RT's end; the last waypoint's
          hand point on the model planned on (run.json planner) to the target = ik_err_cm;
      (4) C at the end label = the plan's last hand point (pelvis frame), settled segments labelled after the plan;
      (5) d_this / d_prev recorded (0 with --planner native)."""
    sys.path[:0] = [os.environ.get("D3_REACHWM", str(REPO / "scripts" / "experiments" / "reachwm_20261030")), str(REPO / "src")]
    import data as DT
    import mujoco
    from scipy.spatial.transform import Rotation

    from locomani.primitives.arm_ik import ArmIK


    run = json.loads((a.out / "run.json").read_text())
    rfk = L.Planner()
    pfk = rfk if run["planner"] == "lml" else NativeFK(ArmIK(str(REPO / "outputs" / "scene" / "walk_mars" / "scene_locomani.xml")))
    planner0 = L.Planner
    L.Planner = lambda: pfk  # d3's check recomputes ik_err_cm on the model planned on (robot_fk, or walk_mars for native)
    try:
        D3.check(a.out, a.id0, a.resets)
    finally:
        L.Planner = planner0
    bm = mujoco.MjModel.from_xml_path(str(Path(run["body"]["REACH_SCENE_DIR"]) / "scene_locomani.xml"))  # SIMPLE's
    # scene (HumanoidArena's runs record it too: their arm record's mirror; their body FK is robot_fk, run["body_fk"])
    bd = mujoco.MjData(bm)
    names = _mj_names()
    bq = np.array([bm.jnt_qposadr[bm.joint(nm).id] for nm in names])
    brq = int(bm.jnt_qposadr[bm.joint("floating_base_joint").id])
    bw = {s: bm.body(f"{s}_wrist_yaw_link").id for s in ("right", "left")}
    c_off = np.asarray(run["calib"]["c"], float)
    to_mj = np.array([L.SCALEBFM_ORDER.index(nm) for nm in names])
    bad = 0
    for sub in ("data", "data_both"):
        p = a.out / sub / "all.npz"
        if not p.exists():
            continue
        z = dict(np.load(p))
        A = dict(np.load(a.out / sub / "arm.npz"))
        n = z["n_steps"].astype(int)
        so, last = np.r_[0, np.cumsum(n)[:-1]], np.cumsum(n) - 1
        side = z["side"]
        right = side > 0
        st = z["start"].astype(float)
        c, s_ = np.cos(st[:, 3]), np.sin(st[:, 3])
        t = z["target"].astype(float)
        tw = np.c_[st[:, 0] + c * t[:, 0] - s_ * t[:, 1], st[:, 1] + s_ * t[:, 0] + c * t[:, 1], t[:, 2]]
        W, S = z["W"].astype(float), z["S"].astype(float)
        jaw = np.where(right[:, None], W[last, 4:7], W[last, 7:10])
        d1 = np.abs(100 * np.linalg.norm(jaw - tw, axis=1) - z["jaw_err_cm"]).max()
        # (2) every row of 300 segments
        rng = np.random.default_rng(1)
        idx = rng.choice(len(n), size=min(300, len(n)), replace=False)
        rows = np.concatenate([np.arange(so[i], so[i] + n[i]) for i in idx])
        g = S[rows, 58:61]
        th, ph = np.arcsin(np.clip(g[:, 0], -1, 1)), np.arctan2(-g[:, 1], -g[:, 2])
        Rr = Rotation.from_euler("ZYX", np.c_[W[rows, 3], th, ph])
        quat, Rm = Rr.as_quat()[:, [3, 0, 1, 2]], Rr.as_matrix()
        e_kp, e_body, e_rfk = 0.0, [], []
        lab = set(last[i] for i in idx if z["settled_end"][i] and not z["fell"][i])
        e_lab = []
        for r, qq, R_ in zip(rows, quat, Rm):
            q_mj = S[r, :29][to_mj]
            e_kp = max(e_kp, float(np.linalg.norm(z["KP"][r, 0, 0] - W[r, 4:7])), float(np.linalg.norm(z["KP"][r, 1, 0] - W[r, 7:10])))
            bd.qpos[:] = 0
            bd.qpos[brq : brq + 3], bd.qpos[brq + 3 : brq + 7], bd.qpos[bq] = W[r, :3] - R_ @ c_off, qq, q_mj
            mujoco.mj_kinematics(bm, bd)
            fk = rfk.fk_world(W[r, :3], qq, q_mj)
            for s, col in (("right", slice(4, 7)), ("left", slice(7, 10))):
                hb = bd.xpos[bw[s]] + bd.xmat[bw[s]].reshape(3, 3) @ L.HAND_OFF
                if run.get("body_fk") == "robot_fk":  # HumanoidArena: its G1 is robot_fk's (c ~ 1e-7 m, calibrated)
                    hb = fk[s][0]
                e_body.append(float(np.linalg.norm(hb - W[r, col])))
                if r in lab:
                    e_lab.append(e_body[-1])
                e_rfk.append(float(np.linalg.norm(fk[s][0] - W[r, col])))
        # (3) the reference features
        nr = z["n_ref"].astype(int)
        ro = np.r_[0, np.cumsum(nr)[:-1]]
        jl = ro + nr - 1
        d3q = float(np.abs(z["ref_q"] - A["RQ"][jl]).max())
        d3p = float(np.abs(z["ref_pel"] - A["RP"][jl]).max())
        d3s = float(np.abs(z["ref_s"] - z["RT"][jl]).max())
        hp_last = {}
        e3 = []
        for i in idx:
            hp = pfk.fk_world(A["RP"][jl[i], :3].astype(float), A["RP"][jl[i], 3:7].astype(float), A["RQ"][jl[i]].astype(float))
            hp_last[i] = hp["right" if side[i] > 0 else "left"][0]
            e3.append(abs(100 * np.linalg.norm(hp_last[i] - tw[i]) - z["ik_err_cm"][i]))
        # (4) C at the end label vs the plan's last hand point (RP is unshifted: FK(shifted) - d_this = FK(RP))
        D = DT.load(str(p))
        ok = (~z["fell"].astype(bool)) & z["settled_end"].astype(bool) & (z["settle_s"] >= 0)
        e4 = []
        for i in [i for i in idx if ok[i]]:
            r = last[i]
            Rp = Rotation.from_euler("ZYX", [W[r, 3], np.arcsin(np.clip(S[r, 58], -1, 1)), np.arctan2(-S[r, 59], -S[r, 60])]).as_matrix()
            e4.append(100 * float(np.linalg.norm(Rp.T @ (hp_last[i] - W[r, :3]) - z["C"][r, :3])))
        have = "d_this" in z and "d_prev" in z
        dmax = (100 * float(np.abs(z["d_this"]).max()), 100 * float(np.abs(z["d_prev"]).max())) if have else (np.nan, np.nan)
        print(f"xfer {sub}: {len(n)} segments ({run['controller']}, planner {run['planner']}, stance {run['stance']})")
        print(f"  (1) direct error recomputed from W's last row vs jaw_err_cm: max |d| {d1:.4f} cm")
        print(f"  (2) hand point, {len(rows)} rows of {len(idx)} segments: KP's hand point vs W max {100 * e_kp:.5f} cm; this "
              f"body's FK (wrist_yaw_link + R 0.13) from the recorded joints and pelvis vs W: end labels of settled segments "
              f"({len(e_lab)}) max {100 * max(e_lab, default=0):.4f} cm, all rows median {100 * np.median(e_body):.4f} / max "
              f"{100 * max(e_body):.3f} cm (after mj_step the bodies' xpos are one 5 ms substep behind qpos, so moving rows "
              f"differ); robot_fk (data3's model) vs W median {100 * np.median(e_rfk):.3f} cm, p99 "
              f"{100 * np.percentile(e_rfk, 99):.3f}, max {100 * max(e_rfk):.3f} (reported: model versions' kinematics)")
        print(f"  (3) ref_q / ref_pel / ref_s vs the last waypoint max |d| {d3q:.1e} / {d3p:.1e} / {d3s:.1e}; ik_err_cm from the "
              f"last waypoint on the plan's model ({'robot_fk' if run['planner'] == 'lml' else 'walk_mars'}) max |d| {max(e3):.4f} cm")
        print(f"  (4) C at the end label vs the plan's last hand point, {len(e4)} settled segments labelled after the plan: "
              f"median {np.median(e4) if e4 else float('nan'):.4f} cm, max {max(e4) if e4 else float('nan'):.4f} cm")
        print(f"  (5) d_this / d_prev recorded: {have}; max |component| {dmax[0]:.3f} / {dmax[1]:.3f} cm; data.py err_cm vs "
              f"jaw_err_cm max |d| {np.abs(D['err_cm'] - D['jaw_err_cm']).max():.4f} cm")  # fmt: skip
        bad += int(d1 > 0.01) + int(e_kp > 1e-6) + int(max(e_lab, default=0) > 2e-4) + int(max(d3q, d3p, d3s) > 1e-5) + int(max(e3) > 0.05)
        bad += int(bool(e4) and max(e4) > 0.05) + int(not have)
    print("XFER CHECK", "PASSED" if bad == 0 else f"FAILED ({bad})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["calib", "lag", "drive", "merge", "check", "summary"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--controller", default="scalebfm")
    ap.add_argument("--planner", choices=["native", "lml"], default="native")
    ap.add_argument("--stance", choices=["native", "lml"], default="native")
    ap.add_argument("--delay-f", type=int, default=0, help="the controller's reference delay in this body, frames (lag)")
    ap.add_argument("--resets", type=int, default=400)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--id0", type=int, default=ID0)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default="", help="dev: these reset indices (comma-separated) instead of the shard's")
    ap.add_argument("--table", type=Path, default=D3.TABLE)
    a = ap.parse_args()
    if a.phase == "calib":
        calib(a)
    elif a.phase == "lag":
        lag(a)
    elif a.phase == "drive":
        drive(a)
    elif a.phase == "merge":
        D3.merge(a.out)
    elif a.phase == "check":
        check(a)
    else:
        D3.summary(a.out)
