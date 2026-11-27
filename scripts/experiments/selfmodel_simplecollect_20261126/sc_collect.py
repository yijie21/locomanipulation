"""selfmodel_simplecollect_20261126: data3's and data3b's collection inside SIMPLE's body, for ScaleBFM and SONIC as SIMPLE
drives them (the user's 11-26 decision, "SIMPLE 用方案一"), so their self models can be trained in SIMPLE (2026-11-26).

    R=<repo copy>; cd $R; PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    export SX_REPO=$R OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
    $PY sc_collect.py drive --controller scalebfm --out DIR --resets N --shards P --shard K [--b] [--delay-f F]
    $PY sc_collect.py merge --out DIR;  $PY sc_collect.py check --out DIR [--b];  $PY sc_collect.py summary --out DIR [--b]

What is whose:
  the body and the controllers   selfmodel_xfer_20261125/sx_collect.py's SimpleHost (gen_reach's SIMPLE configuration:
                 scene simple_g1_sonic, BFM_ROOT=odo, hand point 13 cm ahead of wrist_yaw_link; ScaleBFM in mode 4 with
                 the ankles re-pinned at every command, SONIC on SonicMotor's VR 3-point targets and its kPlanner), its
                 record rig (SRig) and its planner "native" (SIMPLE's own Motor.reach whole-body IK; the reference not
                 shifted, so d_this = d_prev = 0), the stance "native" (the Motor's standing pose after its reset).
  the reset, the reaches, the record   selfmodel_data3_20261125/d3_collect.py: its per-reset draws (reset_plan;
                 --b: reset_plan_b / command_b, data3b's kinds), the settle rule (pelvis and both hand points under 2 cm/s
                 for 0.5 s, 0.1 s displacements, at most 3 s; the wait opens at the reference end plus the controller's
                 delay, the first at 0.2 s), the fall / lost rules (lost not judged during a walk, data3 revision 1), the
                 segment layout (segments(), augment_b()), merge, check and the summaries.
  what is this file's
    walk     a walk-first reset (a third, data3's draw: 0.3-1.5 m in a uniform direction, the heading turned 0-90 deg
             either way) walks as SIMPLE walks: Motor.walk_to (--walk-face travel, default: turn toward the goal, walk the
             profile, turn to the goal heading, data3's turn-walk-turn; fixed: the heading kept, then the turn, as
             gen_reach.open_walk; mode 2 for ScaleBFM, told its pelvis by leg odometry, the stop correction off as env.py
             sets it under BFM_ROOT=odo; SONIC's kPlanner SLOW_WALK toward the end pose on its pelvis estimate), started
             at data3's 0.7 s, the goal placed by the simulator pelvis there (as data3). Every tick of it is recorded
             (rows, the rig's settle speeds) and its reference kept as the frames; it ends as walk_to ends (settle_s on
             the end pose, then the Motor's _stand: the measured pose, ankles pinned). The wait opens at the frame the
             walk's reference stops moving plus the delay; lost is re-anchored at the first row after walk_to returns.
    task walk  (selfmodel_simpledata_20261126) half of the walk-first resets, picked by the reset id's hash
             (task_walk(): crc32 of "taskwalk:<reset id>", even = task; no random draw shared with anything else), walk
             as SIMPLE's task does instead (simple_bench's approach): gen_reach.open_walk(forward, left, turn 0), i.e.
             Motor.walk_to(qpos xy + R(yaw) (forward, left), yaw = the measured heading, face "fixed") -- the heading kept
             and no turn segment at all (walk_to's a2 = 0), the displacement uniform in gen_reach.APPROACH_BOX (forward
             -0.40..1.00 m, left -0.30..0.30 m: one leg, as simple_bench splits at 1.0 m), drawn from
             default_rng([seed + i, 2]). The d3 walk draw (reset_plan's) is still made, so the commands are unchanged,
             and kept in the log as walk.d3. walk_cmd = (forward, left, 0). The walk's log entry has kind (d3 / task),
             ref_yaw_span_deg (the reference's heading range over the walk; 0 for a task walk) and the hip yaw joints
             (left, right, deg) at the walk's last row and at the first reach's command.
    start perturbation   (fdvel's independent review: every first reach of the xfer records started from one state)
             as LocoManiLab's host does it (robosuite EmptyEnv, initialization_noise "default": N(0, 0.02 rad) on the
             robot's joints at spawn, the reference the fixed default stance), its size set so the in-place first
             reaches spread as data3's (PERTURB): at each reset the 29 body joints are
             spawned at the stance plus N(0, --perturb rad), clipped to their ranges (Motor.reset then drops the feet
             to 5 mm above the floor and starts the policy history there; SONIC's reset settles 2 s on its own plan
             from it), while the stance reference stays the unperturbed one (the clean reset's standing joints and
             pelvis height at the measured xy and heading). --perturb 0 gives sx_collect's native stance exactly.
             The noise is drawn from default_rng([seed + i, 1]), apart from the reset's own draws (default_rng(seed + i),
             data3's order), so the commands do not depend on it.
    --b      data3b's kinds on SIMPLE's planner: pre (Motor.reach to the pre-grasp point at pre_speed), entry (Motor.reach
             to the grasp point at the entry speed, the path starting entry_cont of the way from the measured hand to
             the pre command, the IK built on the reference pose: simple_bench's entry, ik_from_ref), rel (the target and
             attitude command_b draws, at its speed), detour (Motor.reach through command_b's via points at 0.15 m/s;
             Motor.reach turns the wrist over the first leg, then holds it). Two-hand entries start from the measured
             hands (Motor.reach takes `start` for one side only); a two-hand detour keeps the shorter hand's via count.
Records as data3 (data/, data_both/, --b: data_detour/; data.py reads them unchanged) plus d_this / d_prev (0) and
pert_rms_deg (the spawn noise's RMS over the 29 joints); w*_log.json also holds each reset's noise vector.
Seeds 81000000 + i, reset ids 50000000 + i; --b 83000000 + i / 60000000 + i (data3 73M / 20M, data3b 77M / 30M, xfer
79M / 40M).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SX_DIR = Path(os.environ.get("SC_SX", HERE.parent / "selfmodel_xfer_20261125"))
sys.path.insert(0, str(SX_DIR))
import sx_collect as SX  # noqa: E402

D3, L = SX.D3, SX.L
FPS, TICKS = L.FPS, L.TICKS
SEED, ID0, SEED_B, ID0_B = 81_000_000, 50_000_000, 83_000_000, 60_000_000
# spawn noise std (rad): robosuite's initialization_noise "default" is 0.02 (EmptyEnv, LocoManiLab's data3 host), but
# SIMPLE's controllers carry it further to the first command; these give data3's in-place first-reach spread (per-joint
# std median 0.34 deg ScaleBFM, 0.42 SONIC): ScaleBFM 0.007 -> 0.35 deg (dev smoke 11-26, 36 resets; 0.02 -> 1.02);
# SONIC 0.02 -> 0.73, 0.012 -> 0.58, so 0.005 is an extrapolation, not yet measured
PERTURB = {"scalebfm": 0.007, "sonic": 0.005, "heft": 0.007}
DELAY_F = {"scalebfm": 0, "sonic": 4, "heft": 0}  # frames: selfmodel_xfer's lag in this body (sx_collect lag)
HIP_YAW = ("left_hip_yaw_joint", "right_hip_yaw_joint")


def task_walk(reset_id: int) -> bool:
    """Whether a walk-first reset walks as SIMPLE's task (see the module docstring): half of them, by the reset id's hash."""
    return zlib.crc32(f"taskwalk:{int(reset_id)}".encode()) % 2 == 0


def approach_box() -> tuple[np.ndarray, np.ndarray]:
    return _gen_reach().APPROACH_BOX


def _gen_reach():
    d = os.environ.get("D3_REACHWM", str(SX.REPO / "scripts" / "experiments" / "reachwm_20261030"))
    if d not in sys.path:
        sys.path.insert(0, d)
    import gen_reach as GR

    return GR


def walk_plan(plan: dict, reset_id: int, rng) -> dict:
    """plan["walk"] for this reset: d3's draw (kind "d3"), or for task_walk(reset_id) the task walk drawn from rng."""
    w = plan["walk"]
    if w is None:
        return plan
    if task_walk(reset_id):
        lo, hi = approach_box()
        fwd, left = (float(v) for v in rng.uniform(lo, hi))
        plan["walk"] = dict(kind="task", fwd=fwd, left=left, turn=0.0, done=w["done"], d3=w)
    else:
        plan["walk"] = dict(w, kind="d3")
    return plan


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


class _Fell(Exception):
    def __init__(self, f: int):
        super().__init__(f)
        self.f = f


# -- the host ---------------------------------------------------------------------------------------------------------
class Host(SX.SimpleHost):
    """SimpleHost (native planner and stance) with the spawn perturbation and the unperturbed stance reference."""

    def __init__(self, controller: str, delay_f: int, perturb: float):
        super().__init__(controller, "native", delay_f)  # its __init__ reset the motor once, unperturbed
        p = self.mot.ref(0.0)
        self.clean_z, self.clean_q = float(p.pos[2]), np.asarray(p.joints, float).copy()
        self.sigma, self.noise = float(perturb), None
        self.walk_face = "travel"  # Motor.walk_to's face: "travel" (data3's turn-walk-turn) or "fixed" (gen_reach.open_walk)
        sim, m = self.mot.sim, self.m
        jid = [self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_JOINT, n) for n in L.SCALEBFM_ORDER]  # S order (self.qadr)
        self.lim = np.array([m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf) for j in jid], float)
        orig = sim.reset

        def reset(xy, yaw, crate_dy):  # Sim.reset (SonicSim's too), then the noise on the 29 body joints
            orig(xy, yaw, crate_dy)
            if self.noise is not None:
                sim.d.qpos[self.qadr] = np.clip(sim.d.qpos[self.qadr] + self.noise, self.lim[:, 0], self.lim[:, 1])
                self.mj.mj_forward(sim.m, sim.d)

        sim.reset = reset

    def reset(self, rng=None) -> np.ndarray:
        """The Motor's reset with the 29 joints spawned at the stance plus N(0, sigma); returns the noise (rad, S order)."""
        self.noise = rng.standard_normal(29) * self.sigma if (rng is not None and self.sigma > 0) else None
        try:
            self.mot.reset((0.0, 0.0), 0.0)
        finally:
            noise, self.noise = self.noise, None
        self.frames = None
        return np.zeros(29) if noise is None else noise

    def start_frames(self) -> np.ndarray:
        """The unperturbed stance (the clean reset's standing joints and pelvis height) at the measured xy and heading."""
        p = self.mot.ref(0.0)  # Motor._stand after the reset: measured xy, yaw-only attitude
        R = L._qmat(p.quat)
        start = (np.r_[np.asarray(p.pos, float)[:2], self.clean_z] + R @ self.c, np.asarray(p.quat, float), self.clean_q.copy())
        return L.hold(start, L.START_HOLD_S + L.MARGIN_S + L.PRE_S)


def pose_frame(h, pose) -> np.ndarray:
    """A Motor Pose (pelvis = the free joint) as a frame in robot_fk's pelvis convention (root xyz, wxyz, 29 joints)."""
    q = np.asarray(pose.quat, float)
    return np.r_[np.asarray(pose.pos, float) + L._qmat(q) @ h.c, q, np.asarray(pose.joints, float)]


# -- the walk -----------------------------------------------------------------------------------------------------------
def walk_s(h, rig, rows: dict, frames: np.ndarray, f: int, x: dict, walk: dict, delay: int):
    """SIMPLE's walk from row f (see the module docstring): every tick recorded into rows; returns the frames (the walk's
    reference, then the Motor's stand pose held), the last recorded row, the frame the wait opens at and the walk's info."""
    mot = h.mot
    task = walk.get("kind") == "task"
    if not task:
        p, Rq = x["root"][:2].astype(float), L._qmat(x["root"][3:])
        yaw = L.yaw_of(Rq)
        g_xy = p + walk["d"] * np.array([math.cos(yaw + walk["dir"]), math.sin(yaw + walk["dir"])])
        goal = g_xy - (L.rot_z(yaw + walk["turn"]) @ h.c)[:2]  # Motor plans the free joint's xy; c at the end heading
    a0, a1, af = x["root"][:2].copy(), frames[f, :2].copy(), f
    wf: dict = {}
    dev = [0.0]
    prev = mot.__dict__.get("tick")
    orig = mot.tick

    def tick():
        k = mot.k
        wf[k] = pose_frame(h, mot.ref(mot.t))
        if not orig():
            return False
        xk = rows[k] = h.sample()
        rig.record(xk)
        if float(xk["root"][2]) < L.FALL_Z:
            raise _Fell(k)
        j = max(k - delay, af)
        fr = wf[j] if j in wf else frames[min(j, len(frames) - 1)]
        dev[0] = max(dev[0], float(np.linalg.norm((xk["root"][:2] - a0) - (fr[:2] - a1))))
        return True

    mot.tick = tick
    t0 = time.time()
    try:
        if task:  # simple_bench's approach walk: the heading kept, no turn
            out = _gen_reach().open_walk(mot, walk["fwd"], walk["left"], 0.0)
        else:
            out = mot.walk_to(goal, yaw + walk["turn"], face=h.walk_face)
    finally:
        if prev is None:
            del mot.tick
        else:
            mot.tick = prev
    if mot.fell_at is not None:
        raise _Fell(mot.k)
    k_end = mot.k
    W = np.array([wf[k] for k in range(f + 1, k_end)])
    moving = np.where(np.abs(W - W[-1]).max(1) > 1e-9)[0]
    k_stop = f + 1 + (int(moving[-1]) + 1 if len(moving) else 0)  # the first frame of the reference at rest
    stand = pose_frame(h, mot.ref(mot.t))  # walk_to's last act: Motor._stand, the measured pose held
    frames = np.concatenate([frames[: f + 1], W, np.repeat(stand[None], int(D3.HOLD_S * FPS), 0)])
    h.follow(frames)
    ry = np.array([L.yaw_of(L._qmat(w[3:7])) for w in W])
    span = float(np.degrees(np.ptp(np.unwrap(ry)))) if len(ry) else 0.0
    asked = [walk["fwd"], walk["left"], 0.0] if task else [walk["d"] * math.cos(walk["dir"]), walk["d"] * math.sin(walk["dir"]), walk["turn"]]
    info = dict(asked=asked, dev_cm=100 * dev[0], kind="task" if task else "d3", ref_yaw_span_deg=span,
                plan=dict(wall_s=time.time() - t0, ok=bool(out.ok), why=out.why, walk_s=(k_end - f - 1) / FPS,
                          ref_stop_s=(k_stop - f - 1) / FPS, start_f=f, end_f=k_end - 1, ref_stop_f=k_stop, info=out.info))  # fmt: skip
    return frames, k_end - 1, k_stop + delay, info


# -- data3b's kinds on SIMPLE's planner ------------------------------------------------------------------------------------
def plan_native_b(h, x: dict, ref_frame: np.ndarray, cmd: dict):
    """sx_collect.plan_native for command_b's commands: Motor.reach with the kind's speed, start, via points and IK base;
    returns (blend, last, rec) with the data3b fields per hand (d_start_cm, pre_back_cm, att_change_deg, via)."""
    mot = h.mot
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    Rh = L.rot_z(L.yaw_of(L._qmat(quat)))
    hs = cmd["hands"]
    kw = dict(speed=float(cmd["speed"]), pelvis_z=cmd["pelvis_z"], servo=False, ik_from_ref=cmd["kind"] == "entry")
    if len(hs) == 1:
        (s,) = hs
        c = hs[s]
        if c["start"] is not None:
            kw["start"] = np.asarray(c["start"], float)
        used = {s: [np.asarray(v, float) for v in c["vias"]]}
        if used[s]:
            kw["via"] = used[s]
        args = (s, np.asarray(c["target_w"], float), np.asarray(c["rot_w"], float))
    else:
        nv = min(len(hs["left"]["vias"]), len(hs["right"]["vias"]))
        used = {s: [np.asarray(v, float) for v in hs[s]["vias"][:nv]] for s in hs}
        if nv:
            kw["via"] = [(used["left"][i], used["right"][i]) for i in range(nv)]
        args = ("both", tuple(np.asarray(hs[s]["target_w"], float) for s in ("left", "right")),
                tuple(np.asarray(hs[s]["rot_w"], float) for s in ("left", "right")))  # fmt: skip
    cap: dict = {}
    blend0 = mot._blend

    def blend(poses, durations):
        cap["poses"], cap["durs"] = list(poses), list(durations)
        return blend0(poses, durations)

    saved = {k: mot.__dict__[k] for k in ("_blend", "_run") if k in mot.__dict__}
    mot._blend, mot._run = blend, (lambda name, update, timeout: None)
    t0 = time.time()
    try:
        mot.reach(*args, **kw)
    finally:
        for k in ("_blend", "_run"):
            mot.__dict__.pop(k, None)
        mot.__dict__.update(saved)
    plan_s = time.time() - t0
    poses, durs = cap["poses"], cap["durs"]
    p0 = poses[0]
    first = np.r_[np.asarray(p0.pos) + L._qmat(p0.quat) @ h.c, p0.quat, p0.joints]
    gap = float(np.abs(first - ref_frame).max())
    tup = [(np.asarray(q.pos, float) + L._qmat(q.quat) @ h.c, np.asarray(q.quat, float), np.asarray(q.joints, float)) for q in poses]
    shifted = [(ref_frame[:3].astype(float), ref_frame[3:7].astype(float), ref_frame[7:].astype(float))] + tup[1:]
    last = tup[-1]
    hk = h.nfk.fk_world(last[0] - L._qmat(last[1]) @ h.c, last[1], last[2])
    hands = {}
    for s, c in hs.items():
        tw, rw = np.asarray(c["target_w"], float), np.asarray(c["rot_w"], float)
        dyaw, pitch, roll = D3.euler_zyx(Rh.T @ rw)
        pw, Rw = x["wr"][s]
        hp = pw + Rw @ L.HAND_OFF
        hands[s] = dict(target_w=tw, rot_w=rw, dyaw=dyaw, pitch=pitch, roll=roll, core=c["core"],
                        ik_err=float(np.linalg.norm(hk[s][0] - tw)),
                        cmd_target=np.r_[(Rh.T @ np.r_[tw[:2] - p[:2], 0.0])[:2], tw[2]],
                        d_start_cm=100 * float(np.linalg.norm(tw - hp)), pre_back_cm=100 * float(c["back"]),
                        att_change_deg=SX._ang(Rw, rw), via=[(rw, v) for v in used[s]])  # fmt: skip
    rec = dict(mode=cmd["mode"], kind=cmd["kind"], speed=cmd["speed"], pair_att=cmd["pair_att"], hands=hands,
               pelvis_z=cmd["pelvis_z"], reach_s=float(sum(durs)), plan_s=plan_s, ref_q=last[2].copy(),
               ref_pel=np.r_[last[0], last[1]], delta=np.zeros(3), RT=np.r_[0.0, np.cumsum(durs)],
               RQ=np.array([pp[2] for pp in tup]), RP=np.array([np.r_[pp[0], pp[1]] for pp in tup]), start_gap=gap)  # fmt: skip
    return L.blend_frames(shifted, durs), last, rec


# -- one reset ----------------------------------------------------------------------------------------------------------
def run_reset_s(h: Host, rig, plan: dict, b: bool, table: dict, cfg: dict, noise_rng):
    """d3_collect.run_reset (b: run_reset_b) on SIMPLE's body: the perturbed spawn, SIMPLE's walk, SIMPLE's planner."""
    t_wall = time.time()
    noise = h.reset(noise_rng)
    rig.clear()
    frames = h.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(D3.HOLD_S * FPS), 0)])
    h.follow(frames)
    rows, reaches, slots, stat = {}, [], [], {}
    items = plan["slots"] if b else [dict(kind="d3", mode=c["mode"], cmd=c) for c in plan["cmds"]]
    fell = lost = None
    why, f = "", 0
    walk = plan["walk"] if plan["walk"] and plan["walk"]["done"] else None
    stage = "walk" if walk else "wait"  # walk | walking | wait (for the next reach) | label (of the last reach)
    k, open_f = 0, D3.FIRST_OPEN_F
    anchor, cur_delta, active = None, np.zeros(3), None
    winfo = dict(asked=None, real=None, dev_cm=0.0, plan=None)
    if walk is not None:
        winfo.update(kind=walk.get("kind", "d3"), d3=walk.get("d3"), ref_yaw_span_deg=None, hip_yaw_end_deg=None, hip_yaw_cmd_deg=None)
    hy = [list(L.SCALEBFM_ORDER).index(n) for n in HIP_YAW]
    hip_first = None  # the hip yaw joints (deg) at the first reach's command, every reset
    x_walk0 = None
    delay, plan_s, gaps = h.ref_delay_f, 0.0, []
    while True:
        if not h.step(f):  # the Motor refuses a fallen robot (pelvis below 0.40 m); d3's 0.45 m fires first
            fell, why = f, "height"
            break
        x = rows[f] = h.sample()
        rig.record(x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, "height"
            break
        if anchor is not None:
            fr = frames[min(max(f - delay, anchor[2]), len(frames) - 1)]
            dev = float(np.linalg.norm((x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])))
            if stage == "walking":
                winfo["dev_cm"] = max(winfo["dev_cm"], 100 * dev)
            elif dev > L.LOST_M:
                fell, lost, why = f, f, "lost"
                break
        act = None
        if stage == "walk" and f == D3.WALK_F:
            act = "walk"
        elif stage == "walking" and f >= open_f:
            stage = "wait"
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
        if stage in ("wait", "label") and act is None:
            on_grid = active is None or (f - active["f_cmd"]) % TICKS == 0
            if f >= open_f + D3.SETTLE_F and on_grid and rig.quiet():
                act = "settled"
            elif f >= open_f + D3.SETTLE_MAX_F and on_grid:
                act = "timeout"
        if act == "walk":
            x_walk0 = x
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            try:
                frames, f, open_f, wi = walk_s(h, rig, rows, frames, f, x, walk, delay)
            except _Fell as e:
                fell, why = e.f, "height"
                break
            winfo.update(asked=wi["asked"], plan=wi["plan"], dev_cm=max(winfo["dev_cm"], wi["dev_cm"]), ref_yaw_span_deg=wi["ref_yaw_span_deg"],
                         hip_yaw_end_deg=np.degrees(np.asarray(rows[f]["q"], float)[hy]).round(3).tolist())  # fmt: skip
            stage = "walking"
        elif act in ("settled", "timeout"):
            if active is not None:  # the previous reach's end label is this row
                active.update(f_label=f, settled_end=act == "settled", settle_s=(f - active["f_end"]) / FPS)
                active = None
            if act == "timeout":
                if k < len(items):
                    slots.append(dict(slot=k, kind=items[k]["kind"], mode=items[k]["mode"], status="unsettled", wait_s=(f - open_f) / FPS))
                    k += 1
                    if b and items[k - 1]["kind"] == "pre" and k < len(items):
                        slots.append(dict(slot=k, kind="entry", mode=items[k]["mode"], status="pair skipped"))
                        k += 1
                open_f = f
                frames = np.concatenate([frames[: f + 1], np.repeat(frames[f : f + 1], int(D3.HOLD_S * FPS), 0)])
                h.follow(frames)
            if k >= len(items):
                break
            if act == "settled":
                it = items[k]
                if x_walk0 is not None and winfo["real"] is None:
                    p0, y0 = x_walk0["root"][:2].astype(float), L.yaw_of(L._qmat(x_walk0["root"][3:]))
                    dp = L.rot_z(y0).T @ np.r_[x["root"][:2] - p0, 0.0]
                    winfo["real"] = [float(dp[0]), float(dp[1]), D3.wrap(L.yaw_of(L._qmat(x["root"][3:])) - y0)]
                    winfo["hip_yaw_cmd_deg"] = np.degrees(np.asarray(x["q"], float)[hy]).round(3).tolist()
                try:
                    if b:
                        blend, last, rec = plan_native_b(h, x, frames[f], D3.command_b(x, it, table, cfg, stat))
                    else:
                        blend, last, rec = SX.plan_native(h, x, frames[f], it["cmd"])
                except (np.linalg.LinAlgError, ValueError) as e:
                    why = f"ik {type(e).__name__}: {e}"
                    break
                plan_s += rec["plan_s"]
                gaps.append(rec.get("start_gap", 0.0))
                if hip_first is None:
                    hip_first = np.degrees(np.asarray(x["q"], float)[hy]).round(3).tolist()
                rec.update(slot=k, chain=len(reaches), f_cmd=f, f_end=f + len(blend), pre_wait_s=(f - open_f) / FPS,
                           d_prev=cur_delta.copy(), f_label=None, settled_end=False, settle_s=-1.0)  # fmt: skip
                slots.append(dict(slot=k, kind=it["kind"], mode=it["mode"], status="done", wait_s=(f - open_f) / FPS, f=f))
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
    while k < len(items):
        slots.append(dict(slot=k, kind=items[k]["kind"], mode=items[k]["mode"], status="not reached"))
        k += 1
    info = dict(fell=fell, lost=lost, why=why, steps=len(rows), walked=walk is not None, walk=winfo, slots=slots, planned=len(reaches),
                wall_s=time.time() - t_wall, plan_s=plan_s, delay_f=delay, start_gap_max=max(gaps, default=0.0),
                noise=np.round(noise, 6).tolist(), pert_rms_deg=float(np.degrees(np.sqrt(np.mean(noise**2)))), hip_yaw_first_deg=hip_first,
                **stat)  # fmt: skip
    return rows, info, frames, reaches


# -- the record ---------------------------------------------------------------------------------------------------------
def save_chunk(segs: list[dict], d: Path, name: str, rig, scalars: tuple) -> None:
    """d3_collect.save_chunk (with the mode's scalars) plus d_this / d_prev."""
    (d / "arm").mkdir(parents=True, exist_ok=True)
    if not segs:
        return
    z, a = D3.pack(segs, scalars)
    z["d_this"] = np.stack([s["d_this"] for s in segs])
    z["d_prev"] = np.stack([s["d_prev"] for s in segs])
    a.update(radius=rig.radius, link=np.array(rig.links))
    for path, arr in ((d / f"{name}.npz", z), (d / "arm" / f"{name}.npz", a)):
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, path)


def code_md5() -> dict:
    out = SX.code_md5()
    out.update({p.name: md5(p) for p in (Path(__file__).resolve(), SX_DIR / "sx_collect.py")})
    return out


EXTRA = ("settled_2cm", "jit_pp_cm", "jit_rms_cm")  # selfmodel_hacollect_20261126's hc_ha.EXTRA


def settle(a) -> dict | None:
    """--settle-v / --settle-max-s / --hold-s (2026-11-26, selfmodel_simplesonicdata_20261126: SONIC in place at 5 cm/s,
    6 s, 7.5 s, the user's decision, as selfmodel_hacollect's hc_ha sets them): written on d3_collect's module constants,
    which its rig (Rig.quiet: SETTLE_V) and this file's waits and splices (D3.SETTLE_MAX_F, D3.HOLD_S) read every time.
    None when no option is given: d3_collect's rule, nothing changes (no EXTRA, run.json as before)."""
    if a.settle_v is None and a.settle_max_s is None and a.hold_s is None:
        return None
    if a.settle_v is not None:
        D3.SETTLE_V = float(a.settle_v)
    if a.settle_max_s is not None:
        D3.SETTLE_MAX_F = int(round(a.settle_max_s * D3.FPS))
    if a.hold_s is not None:
        D3.HOLD_S = float(a.hold_s)
    return dict(settle_v=D3.SETTLE_V, settle_f=D3.SETTLE_F, settle_max_f=D3.SETTLE_MAX_F, hold_s=D3.HOLD_S)


def settle_fields(segs: list[dict], rows: dict, reaches: list[dict]) -> None:
    """EXTRA per segment, hc_ha.settle_fields' rule: settled_2cm (the end label settled and v_end_cms < 2, the same
    25-step window), jit_pp_cm / jit_rms_cm (the segment's own hand point over the 25 steps ending at its settled end
    label: the largest distance between two of its positions, the RMS distance to their mean; NaN without a settled
    label). A two-hand segment (side 2) takes the right hand's, as hc_ha's does."""
    for sg in segs:
        rc = reaches[sg["chain"]]
        side = "right" if int(sg["side"]) == 1 else "left"
        f = rc.get("f_label")
        ok = bool(sg["settled_end"]) and not bool(sg["fell"]) and f is not None
        sg["settled_2cm"] = bool(ok and float(sg["v_end_cms"]) < 2.0)
        if ok:
            P = np.array([rows[g]["wr"][side][0] + rows[g]["wr"][side][1] @ L.HAND_OFF
                          for g in range(f - D3.SETTLE_F + 1, f + 1) if g in rows], float)  # fmt: skip
            sg["jit_pp_cm"] = float(100 * np.linalg.norm(P[:, None] - P[None], axis=2).max())
            sg["jit_rms_cm"] = float(100 * np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
        else:
            sg["jit_pp_cm"] = sg["jit_rms_cm"] = float("nan")


def drive(a) -> None:
    ids = [int(v) for v in a.ids.split(",")] if a.ids else list(range(a.shard, a.resets, a.shards))
    n = min(a.limit or len(ids), len(ids))
    subs = ("data", "data_both") + (("data_detour",) if a.b else ())
    dirs = {k: a.out / k for k in subs}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    od = dirs["data"]
    chunks = [range(c, min(c + D3.CHUNK, n)) for c in range(0, n, D3.CHUNK)]
    done = lambda c: (od / f"w{a.shard:03d}_{c:03d}_log.json").exists()  # noqa: E731
    before = sum(len(r) for c, r in enumerate(chunks) if done(c))
    tag = f"sc{'b' if a.b else ''} {a.controller} shard {a.shard}"
    print(f"{tag}: {before}/{n} resets already driven", flush=True)
    if before == n:
        return
    table = json.loads(Path(a.table).read_text())
    cfg = D3.b_config(a.b_config) if a.b else {}
    h = Host(a.controller, a.delay_f, a.perturb)
    h.walk_face = a.walk_face
    rig = SX.SRig(h)
    scalars = (D3.SCALARS_B if a.b else D3.SCALARS) + ("pert_rms_deg", "walk_task") + (EXTRA if a.settle else ())
    if a.settle:  # the hold must outlast the reference delay and the longest wait (d3_collect: 4.5 s for 3 s)
        assert D3.HOLD_S * D3.FPS > D3.SETTLE_MAX_F + h.ref_delay_f, (a.settle, h.ref_delay_f)
    run = dict(controller=a.controller, planner="native", stance="native, unperturbed reference", mode="data3b" if a.b else "data3",
               walk=not a.no_walk, walk_face=a.walk_face, task_walk="half of the walk-first resets, task_walk(reset id)",
               approach_box=np.asarray(approach_box()).tolist(), perturb_rad=a.perturb, delay_f=h.ref_delay_f, seed=a.seed, id0=a.id0, resets=a.resets,
               calib=h.calib, body=SX.SIMPLE_BODY, env=SX.CONTROLLERS[a.controller], b_config=cfg or None, md5=code_md5(),
               table_md5=md5(Path(a.table)), clean_stance=dict(z=h.clean_z, q=np.round(h.clean_q, 6).tolist()))  # fmt: skip
    if a.settle:
        run["settle"] = a.settle
    rp = a.out / "run.json"
    if a.shard == 0 and not rp.exists():
        rp.write_text(json.dumps(run, indent=1, default=D3._json))
    if a.b:
        cp = a.out / "b_config.json"
        if cp.exists():
            assert json.loads(cp.read_text()) == json.loads(json.dumps(cfg)), f"{cp} differs from this run's data3b settings"
        else:
            cp.write_text(json.dumps(cfg, indent=1))
    print(f"{tag}: pelvis offset c {np.round(h.c, 4).tolist()}, delay {h.ref_delay_f} frames, perturb {a.perturb} rad, clean "
          f"stance z {h.clean_z:.4f} m, root check {rig.check_root():.1e} m", flush=True)  # fmt: skip
    if a.settle:
        print(f"{tag}: settle rule {a.settle} (EXTRA {EXTRA} per segment)", flush=True)
    t0, nres, nseg, nfall, nlost, nwalk, nskip, nslot = time.time(), 0, 0, 0, 0, 0, 0, 0
    for c, rng_ids in enumerate(chunks):
        if done(c):
            continue
        out, log = {k: [] for k in subs}, []
        for j in rng_ids:
            i = ids[j]
            rng = np.random.default_rng(a.seed + i)
            plan = D3.reset_plan_b(rng, table, not a.no_walk, cfg) if a.b else D3.reset_plan(rng, table, not a.no_walk)
            plan = walk_plan(plan, a.id0 + i, np.random.default_rng([a.seed + i, 2]))
            rows, info, frames, reaches = run_reset_s(h, rig, plan, a.b, table, cfg, np.random.default_rng([a.seed + i, 1]))
            s = D3.segments(rows, info, frames, reaches, a.id0 + i, h.nfk, rig)
            SX.add_fields(s, reaches)
            if a.b:
                D3.augment_b(s, reaches)
            if a.settle:
                settle_fields(s, rows, reaches)
            for x in s:
                x["pert_rms_deg"] = info["pert_rms_deg"]
                x["walk_task"] = int(bool(info["walked"]) and info["walk"].get("kind") == "task")
                key = "data_detour" if a.b and x["kind"] == D3.KIND["detour"] else "data_both" if x["hands"] == 2 else "data"
                out[key].append(x)
            extra = dict(kinds=[sl["kind"] for sl in plan["slots"]]) if a.b else {}
            log.append(dict(reset=a.id0 + i, n_segs=len(s), walk_asked=plan["walk"] is not None, **extra, **info))
            nres, nseg, nwalk = nres + 1, nseg + len(s), nwalk + bool(info["walked"])
            nfall += info["fell"] is not None and info["lost"] is None
            nlost += info["lost"] is not None
            nskip += sum(sl["status"] == "unsettled" for sl in info["slots"])
            nslot += sum(sl["status"] not in ("not reached", "pair skipped") for sl in info["slots"])
            if nres % 5 == 0 or nres == n - before:
                print(f"{tag}: {before + nres}/{n} resets, {nseg} segments, walked {nwalk}, falls {nfall}, lost {nlost}, unsettled "
                      f"{nskip}/{nslot}, {time.time() - t0:.0f} s ({(time.time() - t0) / nres:.1f} s/reset)", flush=True)  # fmt: skip
        name = f"w{a.shard:03d}_{c:03d}"
        for k_, d in dirs.items():
            save_chunk(out[k_], d, name, rig, scalars)
        (od / f"{name}_log.json").write_text(json.dumps(log, default=D3._json))
    print(f"{tag} done: {before + nres} resets ({nres} this run, {nseg} segments, walked {nwalk}, falls {nfall}, lost {nlost}, "
          f"unsettled {nskip}/{nslot}), {time.time() - t0:.0f} s", flush=True)  # fmt: skip


# -- check: d3_collect.check and sx_collect.check's body-side checks, data_detour/ included ------------------------------
def check(a) -> None:
    """d3_collect.check (ik_err_cm recomputed on Motor's IK model, walk_mars), then sx_collect.check's (1)-(5) on every
    sub-directory: the direct error from W's last row, the hand point by this body's own FK from the recorded joints and
    pelvis (settled end labels, the C-row offset of mj_step's substep reported), the reference features, C at the end
    label vs the plan's last hand point, d_this / d_prev; and this file's: (6) the walk-first resets' first reaches are
    commanded after walk_to returned (the command row after the walk's last row, wait >= 0, the walk measured), (7) the
    spawn noise logged per reset differs between resets (perturb > 0), (8) the task walk: each walk-first reset's kind is
    task_walk(reset id)'s, a task walk's displacement lies in APPROACH_BOX with no turn asked and its reference keeps the
    heading (range <= 1e-6 deg), and the hip yaw twist (max |hip yaw|) at the walk's end and the first command is reported
    (task, d3, in place; not a check item: the entry's criterion); and (5) also builds data.py's x0 with train3b's D
    switches and fdvel's (REACHWM_GOAL_REF, APPROACH, QFIX, FDVEL = 1 unless set) and asserts it finite."""
    for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL"):  # train3b's D features + fdvel
        os.environ.setdefault(k, "1")
    sys.path[:0] = [os.environ.get("D3_REACHWM", str(SX.REPO / "scripts" / "experiments" / "reachwm_20261030")), str(SX.REPO / "src")]
    import data as DT
    import mujoco
    from scipy.spatial.transform import Rotation

    from locomani.primitives.arm_ik import ArmIK

    run = json.loads((a.out / "run.json").read_text())
    pfk = SX.NativeFK(ArmIK(str(SX.REPO / "outputs" / "scene" / "walk_mars" / "scene_locomani.xml")))
    planner0 = L.Planner
    L.Planner = lambda: pfk
    try:
        D3.check(a.out, run["id0"], run["resets"])
    finally:
        L.Planner = planner0
    bm = mujoco.MjModel.from_xml_path(str(Path(run["body"]["REACH_SCENE_DIR"]) / "scene_locomani.xml"))
    bd = mujoco.MjData(bm)
    names = SX._mj_names()
    bq = np.array([bm.jnt_qposadr[bm.joint(nm).id] for nm in names])
    brq = int(bm.jnt_qposadr[bm.joint("floating_base_joint").id])
    bw = {s: bm.body(f"{s}_wrist_yaw_link").id for s in ("right", "left")}
    c_off = np.asarray(run["calib"]["c"], float)
    to_mj = np.array([L.SCALEBFM_ORDER.index(nm) for nm in names])
    bad = 0
    for sub in ("data", "data_both", "data_detour"):
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
        rng = np.random.default_rng(1)
        idx = rng.choice(len(n), size=min(300, len(n)), replace=False)
        rws = np.concatenate([np.arange(so[i], so[i] + n[i]) for i in idx])
        g = S[rws, 58:61]
        th, ph = np.arcsin(np.clip(g[:, 0], -1, 1)), np.arctan2(-g[:, 1], -g[:, 2])
        Rr = Rotation.from_euler("ZYX", np.c_[W[rws, 3], th, ph])
        quat = Rr.as_quat()[:, [3, 0, 1, 2]]
        e_kp, e_body, e_lab, e_lab5 = 0.0, [], [], []
        lab = set(last[i] for i in idx if z["settled_end"][i] and not z["fell"][i])
        lab5 = set()
        if run.get("settle"):  # 2026-11-26 (selfmodel_simplesonicdata_20261126): under a relaxed settle rule the 2e-4 m bound
            # (the recorded xpos is one 5 ms substep behind, so it holds for hands under 2 cm/s) is checked on the end labels
            # that also settled under 2 cm/s (settled_2cm), the rule it was set for; every settled label is printed beside
            lab5 = lab
            lab = set(last[i] for i in idx if z["settled_end"][i] and not z["fell"][i] and z["settled_2cm"][i])
        for r, qq, R_ in zip(rws, quat, Rr.as_matrix()):
            q_mj = S[r, :29][to_mj]
            e_kp = max(e_kp, float(np.linalg.norm(z["KP"][r, 0, 0] - W[r, 4:7])), float(np.linalg.norm(z["KP"][r, 1, 0] - W[r, 7:10])))
            bd.qpos[:] = 0
            bd.qpos[brq : brq + 3], bd.qpos[brq + 3 : brq + 7], bd.qpos[bq] = W[r, :3] - R_ @ c_off, qq, q_mj
            mujoco.mj_kinematics(bm, bd)
            for s, col in (("right", slice(4, 7)), ("left", slice(7, 10))):
                hb = bd.xpos[bw[s]] + bd.xmat[bw[s]].reshape(3, 3) @ L.HAND_OFF
                e_body.append(float(np.linalg.norm(hb - W[r, col])))
                if r in lab:
                    e_lab.append(e_body[-1])
                if r in lab5:
                    e_lab5.append(e_body[-1])
        nr = z["n_ref"].astype(int)
        ro = np.r_[0, np.cumsum(nr)[:-1]]
        jl = ro + nr - 1
        d3q = float(np.abs(z["ref_q"] - A["RQ"][jl]).max())
        d3p = float(np.abs(z["ref_pel"] - A["RP"][jl]).max())
        d3s = float(np.abs(z["ref_s"] - z["RT"][jl]).max())
        hp_last, e3 = {}, []
        for i in idx:
            hp = pfk.fk_world(A["RP"][jl[i], :3].astype(float), A["RP"][jl[i], 3:7].astype(float), A["RQ"][jl[i]].astype(float))
            hp_last[i] = hp["right" if side[i] > 0 else "left"][0]
            e3.append(abs(100 * np.linalg.norm(hp_last[i] - tw[i]) - z["ik_err_cm"][i]))
        D = DT.load(str(p))
        ok = (~z["fell"].astype(bool)) & z["settled_end"].astype(bool) & (z["settle_s"] >= 0)
        e4 = []
        for i in [i for i in idx if ok[i]]:
            r = last[i]
            Rp = Rotation.from_euler("ZYX", [W[r, 3], np.arcsin(np.clip(S[r, 58], -1, 1)), np.arctan2(-S[r, 59], -S[r, 60])]).as_matrix()
            e4.append(100 * float(np.linalg.norm(Rp.T @ (hp_last[i] - W[r, :3]) - z["C"][r, :3])))
        have = "d_this" in z and "d_prev" in z
        dmax = (100 * float(np.abs(z["d_this"]).max()), 100 * float(np.abs(z["d_prev"]).max())) if have else (np.nan, np.nan)
        e5 = float(np.abs(D["err_cm"] - D["jaw_err_cm"]).max())
        x0_ok = bool(np.isfinite(D["x0"]).all())
        print(f"simple {sub}: {len(n)} segments ({run['controller']}, {run['mode']}, perturb {run['perturb_rad']} rad)")
        print(f"  (1) direct error recomputed from W's last row vs jaw_err_cm: max |d| {d1:.4f} cm")
        print(f"  (2) hand point, {len(rws)} rows of {len(idx)} segments: KP vs W max {100 * e_kp:.5f} cm; this body's FK from the "
              f"recorded joints and pelvis vs W: settled end labels ({len(e_lab)}) max {100 * max(e_lab, default=0):.4f} cm, all rows "
              f"median {100 * np.median(e_body):.4f} / max {100 * max(e_body):.3f} cm (moving rows: xpos one 5 ms substep behind)"
              + (f"; settle rule {run['settle']['settle_v']} m/s: the bound on the settled_2cm labels, every settled label "
                 f"({len(e_lab5)}) max {100 * max(e_lab5, default=0):.4f} cm (not checked)" if run.get("settle") else ""))  # fmt: skip
        print(f"  (3) ref_q / ref_pel / ref_s vs the last waypoint max |d| {d3q:.1e} / {d3p:.1e} / {d3s:.1e}; ik_err_cm from the last "
              f"waypoint on walk_mars max |d| {max(e3):.4f} cm")  # fmt: skip
        print(f"  (4) C at the end label vs the plan's last hand point, {len(e4)} settled segments: median "
              f"{np.median(e4) if e4 else float('nan'):.4f} cm, max {max(e4) if e4 else float('nan'):.4f} cm")
        print(f"  (5) d_this / d_prev recorded: {have}, max |component| {dmax[0]:.3f} / {dmax[1]:.3f} cm; data.py err_cm vs "
              f"jaw_err_cm max |d| {e5:.4f} cm; data.py k0 {sorted(set(D['k0'].tolist()))}; x0 {D['x0'].shape[1]} columns "
              f"(FDVEL {DT.FDVEL}, GOAL_REF {DT.GOAL_REF}, APPROACH {DT.APPROACH}, QFIX {DT.QFIX}) finite {x0_ok}")  # fmt: skip
        bad += int(d1 > 0.01) + int(e_kp > 1e-6) + int(max(e_lab, default=0) > 2e-4) + int(max(d3q, d3p, d3s) > 1e-5) + int(max(e3) > 0.05)
        bad += int(bool(e4) and max(e4) > 0.05) + int(not have) + int(e5 > 0.01) + int(int(D["k0"].min()) < 3) + int(not x0_ok)
    R = D3.logs(a.out)
    wk = [r for r in R if r["walked"] and any(s["status"] == "done" for s in r["slots"])]
    first = [next(s for s in r["slots"] if s["status"] == "done") for r in wk]
    late = [r for r, s in zip(wk, first) if not (s["f"] > r["walk"]["plan"]["end_f"] and s["wait_s"] >= 0 and r["walk"]["real"] is not None)]
    walked = [float(np.hypot(*r["walk"]["real"][:2])) for r in wk if r["walk"]["real"] is not None]
    nz = np.array([r["noise"] for r in R])
    distinct = len(np.unique(np.round(nz, 6), axis=0))
    print(f"  (6) walk-first resets with a reach: {len(wk)}; first command after walk_to returned, wait >= 0, walk measured: "
          f"{len(wk) - len(late)} / {len(wk)}; distance walked by then min {min(walked, default=float('nan')):.2f} m, median "
          f"{np.median(walked) if walked else float('nan'):.2f} m")  # fmt: skip
    print(f"  (7) spawn noise: {len(R)} resets, {distinct} distinct vectors, RMS median {np.median([r['pert_rms_deg'] for r in R]):.3f} deg")
    bad += len(late) + int(run["perturb_rad"] > 0 and distinct != len(R))
    # (8) the task walk: the hash split, the box, no turn; the hip yaw twist (max |hip yaw| of the two, deg)
    lo, hi = (np.asarray(v, float) for v in run["approach_box"])
    wa = [r for r in R if r["walk_asked"]]
    split_bad = [r["reset"] for r in wa if r["walked"] and r["walk"]["kind"] != ("task" if task_walk(r["reset"]) else "d3")]
    tw = [r for r in R if r["walked"] and r["walk"]["kind"] == "task"]
    dw = [r for r in R if r["walked"] and r["walk"]["kind"] == "d3"]
    box_bad = [r["reset"] for r in tw if not (np.all(np.asarray(r["walk"]["asked"][:2]) >= lo) and np.all(np.asarray(r["walk"]["asked"][:2]) <= hi)
                                              and r["walk"]["asked"][2] == 0.0)]  # fmt: skip
    turn_bad = [r["reset"] for r in tw if r["walk"]["ref_yaw_span_deg"] is None or r["walk"]["ref_yaw_span_deg"] > 1e-6]
    q0 = np.degrees(np.asarray(run["clean_stance"]["q"], float)[[names.index(nm) for nm in HIP_YAW]])  # the Motor's (MuJoCo) order
    tw_ = lambda v: float(np.abs(v).max()) if v is not None else np.nan  # noqa: E731

    def share(rs, key):
        v = np.array([tw_(r["walk"][key]) for r in rs])
        v = v[np.isfinite(v)]
        return (f"{int((v <= 15).sum())}/{len(v)} = {100 * (v <= 15).mean():.1f} % <= 15 deg, median {np.median(v):.1f}, max {v.max():.1f}"
                if len(v) else "none")  # fmt: skip

    inp = np.array([tw_(r["hip_yaw_first_deg"]) for r in R if not r["walked"] and r.get("hip_yaw_first_deg") is not None])
    print(f"  (8) walk-first resets {len(wa)} (walked {len(tw) + len(dw)}: task {len(tw)}, d3 {len(dw)}); kind vs task_walk(reset id) "
          f"mismatches {len(split_bad)}; task walks outside the box or turned {len(box_bad)}; with a reference heading range > 1e-6 deg "
          f"{len(turn_bad)} (max {max([r['walk']['ref_yaw_span_deg'] or 0 for r in tw], default=0):.2e})")  # fmt: skip
    print(f"      hip yaw twist (stance {q0.round(2).tolist()} deg): task walk end {share(tw, 'hip_yaw_end_deg')}; first command "
          f"{share(tw, 'hip_yaw_cmd_deg')}")  # fmt: skip
    print(f"      d3 walk end {share(dw, 'hip_yaw_end_deg')}; first command {share(dw, 'hip_yaw_cmd_deg')}; in place first command "
          f"{f'{int((inp <= 15).sum())}/{len(inp)} <= 15 deg, max {inp.max():.1f}' if len(inp) else 'none'}")  # fmt: skip
    bad += len(split_bad) + len(box_bad) + len(turn_bad)
    print("SIMPLE CHECK", "PASSED" if bad == 0 else f"FAILED ({bad})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["drive", "merge", "check", "summary"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--controller", default="scalebfm", choices=sorted(SX.CONTROLLERS))
    ap.add_argument("--b", action="store_true", help="data3b's kinds (d3_collect --b)")
    ap.add_argument("--b-config", default=None)
    ap.add_argument("--delay-f", type=int, default=None, help="the controller's reference delay in this body, frames (default: xfer's lag, ScaleBFM 0, SONIC 4)")
    ap.add_argument("--perturb", type=float, default=None, help="spawn noise std, rad, on the 29 body joints (default PERTURB)")
    ap.add_argument("--no-walk", action="store_true")
    ap.add_argument("--walk-face", choices=["travel", "fixed"], default="travel",
                    help="travel: turn toward the goal, walk, turn (data3's walk); fixed: heading kept, then the turn (gen_reach.open_walk)")
    ap.add_argument("--resets", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--id0", type=int, default=None)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default="", help="dev: these reset indices (comma-separated) instead of the shard's")
    ap.add_argument("--table", type=Path, default=D3.TABLE)
    ap.add_argument("--settle-v", type=float, default=None,
                    help="(selfmodel_simplesonicdata_20261126) the settle threshold, m/s (default: d3_collect's 0.02); any of the "
                         "three settle options also adds EXTRA to every segment")  # fmt: skip
    ap.add_argument("--settle-max-s", type=float, default=None, help="the longest settle wait, s (default: d3_collect's 3.0)")
    ap.add_argument("--hold-s", type=float, default=None, help="the reference hold after each splice, s (default: d3_collect's 4.5)")
    a = ap.parse_args()
    a.settle = settle(a)
    a.resets = a.resets if a.resets is not None else (15000 if a.b else 30000)
    a.seed = a.seed if a.seed is not None else (SEED_B if a.b else SEED)
    a.id0 = a.id0 if a.id0 is not None else (ID0_B if a.b else ID0)
    a.delay_f = a.delay_f if a.delay_f is not None else DELAY_F[a.controller]
    a.perturb = a.perturb if a.perturb is not None else PERTURB[a.controller]
    if a.phase == "drive":
        drive(a)
    elif a.phase == "merge":
        D3.merge(a.out)
    elif a.phase == "check":
        check(a)
    else:
        D3.summary_b(a.out) if a.b else D3.summary(a.out)
