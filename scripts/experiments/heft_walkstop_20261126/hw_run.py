"""heft_walkstop_20261126: walk to a finish line on an empty floor and stop as close before it as possible without
crossing -- HEFT in LocoManiLab's EmptyEnv (data3's host), the walking program it gets today against faster gaits and
re-anchored re-issues; ScaleBFM's program as a reference row; HEFT's braking carry past a marker as a measurement.
2026-11-26.

The protocol is the 09-23 / 10-13 finish-line test (wbc_compare_20261012/bfm_brake.py, backends/sonic/diag_brake): reset
facing the start heading, settle standing, walk straight toward a line perpendicular to the start heading, stop; the
body's frontmost low point (any robot geom whose world AABB reaches below OBSTACLE_TOP = 0.37 m, bfm_brake's
frontmost_low) must stay behind the line. Moved into LocoManiLab:

  host     lml_host.HostDriver(EmptyEnv): backend heft (lmlreal's HeftBase, heft/pmg) or scalebfm, as fs_run builds it.
           Start perturbation: robosuite's own robot init noise ("default", 0.02 rad Gaussian per joint: data3's), its
           generator set to default_rng([TAG, rep, 1]) before every reset, so repeat `rep` starts from the same state
           in every arm and cell (the pairing).
  walk     HEFT: gait_heft.walk_frames (lmlcmp's heft_gait/ copies through lc3_run.heft_gait) with the upper body
           blended to DEFAULT_Q -- lc3_run.walk_ctrl's body with the gait as a parameter; with gait_heft.HEFT it is
           walk_ctrl bit for bit (`selftest`). ScaleBFM: hg_walk.walk_frames (walk_ctrl's other branch).
  belief   hg_walk.LegOdo (lmlreal / lmlroute's leg odometry: joint encoders, the IMU's orientation, the most loaded
           foot anchored), started at the walk command on the measured pelvis (the robot's map origin; the line is
           given in that frame, as a look's points are). Every decision reads only the odometry, the encoders and the
           IMU; the simulator's pelvis and geoms only score (fields marked truth).
  settle   the robot's own rule (lc3_run.PropSettle, lmlcmp (a)): both hand points in the pelvis frame (encoder FK) <
           2 cm/s and the 12 leg joints < 0.05 rad/s on each of the last 0.5 s of steps; data3's windows: before the
           first walk the wait opens at 0.2 s (FIRST_OPEN_F); after a walk at the start of its held end stance + the
           buffer delay (data3's walk rule); it goes at the first step >= 0.5 s after the opening where the rule holds,
           at most 3 s after the opening (then it goes anyway, flagged `timeout`). data3's truth rule (Rig.quiet:
           simulator pelvis and hand points < 2 cm/s) is recorded beside it.

One trial (arm, cell, rep):
  1. reset; data3's start reference (start_frames + a long hold); the stand wait.
  2. at the command row: p0 / yaw0 = the measured pelvis (odometry origin), h = heading, n = left. reach_own = the
     frontmost point of the two feet's geoms ahead of the pelvis, by FK of the encoders with the IMU's attitude on the
     host's robot model (the robot's own geometry); reach_true = frontmost_low on the simulator (truth). The line at
     s_line = reach_true + MARGIN + d + jitter (uniform +-JIT, default_rng([TAG, cid, rep, 2])) along h from p0. The
     pelvis target the robot plans for: T = p0 + (s_line - reach_own - MARGIN) h + lat n (the walk ends in the start
     stance, so its frontmost point lands MARGIN before the line).
  3. the program (ARMS): one walk to T (one shot), or one walk then re-issues (re-anchor): after each walk's settle
     wait, rem = T - odometry pelvis; |rem| >= REM_MIN and fewer than REISSUE_MAX re-issues -> a new walk to T planned
     from the believed pose (walk_frames carries goal - believed pose onto the current reference root: the reference
     starts from where the robot believes it is), else stop. A walk whose planned end would put the frontmost point
     past the line is never issued (cannot happen with T as above; checked). P3's short-move rule (--p3): a walk
     whose commanded displacement is shorter than dmin is split into a step back by dmin along -h from the believed
     pose, then (after its settle wait) the walk to T, re-anchored again; a split counts as one re-issue.
     marker (measurement only): a walk to p0 + MARKER_RUN h; the step the odometry pelvis passes MARKER_M ahead, a
     stop walk to the believed pose is spliced from the current reference frame (zero displacement for HEFT, which
     reads only displacements: "stop where you are"); carry = settled pelvis - pelvis at that step (truth).
  4. the final read at the last settle row (truth: frontmost point, pelvis, yaw; own: odometry), then POST_S more
     seconds in which the crossing check continues. A fall (pelvis < 0.45 m) ends the trial; nothing else does (no
     lost rule: the reference running ahead is what is measured; its largest lead is recorded per walk).

Per trial (runs/<tag>.jsonl): cell, rep, line, target, reach_own / reach_true, every walk (asked from the believed pose,
the truth pose at the command, the plan's steps and duration, lift-offs during it, the wait and how it ended, truth
pelvis moved, largest reference lead), and the final metrics: cross (truth frontmost past the line at any step after
the command, m; > 0 = crossed), gap (s_line - frontmost at the read, m), lat / fwd error of the pelvis to T, yaw error,
[revision 1: "frontmost" is the feet's (ankle_roll_link geoms' world AABB), the toes, as reach_own already was; the
registered bfm_brake.frontmost_low over every robot geom with an AABB bottom below 0.37 m is kept as cross_low_m /
gap_low_m / reach_true_low, with the body that set cross_low_m -- in the smoke it was the hip-yaw and knee links'
AABBs, which swell ahead of the toes in swing],
liftoffs (foot load < 10 N for 3 consecutive steps, faildiag's rule, after the command), settle_s (command to the final
read), n_reissue, odo_err (odometry - truth at the read), fell, success. Traces (traces/<arm>_c<cid>_r<rep>.npz) per step:
truth pelvis xy / yaw, odometry xy / yaw, foot loads, frontmost - line (feet; low), the pushed reference root (world), stage.

heft_reanchor2x2_20261126 (the independent review's 2x2 on P2's re-issues; P2's own path is unchanged when both are
at their defaults): spec "src" = where a re-issue decision reads the pelvis -- "odo" (the leg odometry, P2) or "true"
(the simulator's pelvis and yaw: a MEASUREMENT arm that isolates the odometry's share, never a method); spec "aim" =
what the re-issues bring to the line -- "pelvis" (T fixed at the command, P2) or "toe" (P4: after every settle the feet's
reach by the robot's own FK in the stance it now stands in, r_now = reach_own, re-derives the pelvis target
T = p0 + (s_line - r_now - MARGIN) h + lat n and the would-cross guard uses r_now). The first walk is the same in all
four (the belief at the command is p0 / yaw0 either way). Every walk also records, at its settle, the feet's own reach
(reach_own_end) and the truth toes past the line (front_past_end). Cells 11 / 12 (2.00 / 3.00 m straight) are the 2x2's.

    cd $G; PYTHONPATH=$G:$W/src $PY hw_run.py run --arms p0,p1a,p1b,p1c,p2 --cells 0:11 --reps 0:10 --shard K --shards N
    $PY hw_run.py selftest
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "selfmodel_lmlcmp3_20261126")]
import lc3_run as LC  # noqa: E402  (lmlcmp3's frozen runner: walk_ctrl, heft_gait, PropSettle, the drivers through lr_run)

LR, D3, L = LC.LR, LC.D3, LC.L
import hg_walk as HW  # noqa: E402  (on the path through lr_run: hardgrasp's walk and leg odometry)

ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
OUT = Path(os.environ.get("HW_OUT", str(ROOT / "heft_walkstop_20261126")))
TAG = 20261126
FPS, DT = L.FPS, L.DT
SETTLE_F, SETTLE_MAX_F = D3.SETTLE_F, D3.SETTLE_MAX_F  # 0.5 s, 3 s
MARGIN, JIT = 0.025, 0.02  # m: the planned clearance (the success band's centre), the line's jitter
REISSUE_MAX, REM_MIN = 3, 0.02
GAP_OK, LAT_OK = (0.0, 0.05), 0.03
LOAD_OFF_N, LOAD_OFF_F = 10.0, 3  # faildiag's lift-off: a foot's normal force < 10 N on 3 consecutive steps
OBSTACLE_TOP = 0.37  # locomani.primitives.foot_watch.OBSTACLE_TOP
POST_S, T_MAX_S, HOLD_S = 1.0, 90.0, 40.0
MARKER_M, MARKER_RUN = 1.5, 3.0
# cells: (run-up d = the distance to walk, lateral offset of the target), straight first
CELLS = ((0.05, 0.0), (0.10, 0.0), (0.20, 0.0), (0.30, 0.0), (0.50, 0.0), (1.00, 0.0), (1.50, 0.0),
         (0.20, 0.20), (0.20, -0.20), (1.00, 0.20), (1.00, -0.20),
         (2.00, 0.0), (3.00, 0.0))  # fmt: skip  (11 / 12: heft_reanchor2x2_20261126 only)
MARKER_CID = 100
# arms: backend, gait, program (plan | marker), re-anchor
ARMS = {
    "p0": dict(backend="heft", gait="p0", program="plan", reanchor=False),
    "p1a": dict(backend="heft", gait="p1a", program="plan", reanchor=False),
    "p1b": dict(backend="heft", gait="p1b", program="plan", reanchor=False),
    "p1c": dict(backend="heft", gait="p1c", program="plan", reanchor=False),
    "p2": dict(backend="heft", gait="p0", program="plan", reanchor=True),
    "p3": dict(backend="heft", gait=None, program="plan", reanchor=True),  # gait / dmin from --p3 (hw_p3rule.py)
    "s0": dict(backend="scalebfm", gait="hg", program="plan", reanchor=False),
    # SONIC (secondary rows, the coordinator 11-26): its LML walk (hg_walk.walk_frames, as lmlcmp (c)), one shot / re-anchored
    "q0": dict(backend="sonic", gait="hg", program="plan", reanchor=False),
    "q2": dict(backend="sonic", gait="hg", program="plan", reanchor=True),
    # revision 2 (diagnostic, LML and HA): HEFT on the walk HA's task episodes give it today (hr_real: hg_walk.walk_frames,
    # 0.20 m / 0.55 s steps, 6 cm swing) -- one of the LML-vs-HA candidates is this reference, not the simulator
    "p0hg": dict(backend="heft", gait="hg", program="plan", reanchor=False),
    # heft_reanchor2x2_20261126: P2 (odo, pelvis) x {re-anchor source} x {aim}; *t = truth pelvis, measurement only
    "p2t": dict(backend="heft", gait="p0", program="plan", reanchor=True, src="true"),
    "p4": dict(backend="heft", gait="p0", program="plan", reanchor=True, aim="toe"),
    "p4t": dict(backend="heft", gait="p0", program="plan", reanchor=True, src="true", aim="toe"),
    "m050": dict(backend="heft", gait="p0", program="marker", reanchor=False),
    "m035": dict(backend="heft", gait="m035", program="marker", reanchor=False),
    "m025": dict(backend="heft", gait="m025", program="marker", reanchor=False),
    # plumbing only (dev, never in the entry's tables): ScaleBFM through the re-anchor and split code paths
    "s2dev": dict(backend="scalebfm", gait="hg", program="plan", reanchor=True),
    "s3dev": dict(backend="scalebfm", gait="hg", program="plan", reanchor=True),
}


def gaits() -> dict:
    """The HEFT gaits. Cruise speed = the walk step's length / its duration; a move shorter than move_min (0.30 m) is
    omnidirectional steps of the whole move (one step under step_len, plus the closing step), so below step_len only
    the step's duration changes with the gait. p0 = gait_heft.HEFT (0.20 m / 0.40 s = 0.50 m/s); p1a: cadence to 0.30 s
    (LAFAN's fastest steps), 0.67 m/s; p1b / p1c: then longer steps, 0.30 / 0.40 m, 1.00 / 1.33 m/s (LAFAN's forward
    speed 90th / 95th percentile 1.21 / 1.33 m/s); m035 / m025: slower cadence (0.571 / 0.80 s) for the marker rows at the
    old test's 0.35 / 0.25 m/s. Turn and side steps unchanged."""
    _, GH = LC.heft_gait()
    H, rp = GH.HEFT, dataclasses.replace
    return {
        "p0": H,
        "p1a": rp(H, walk=rp(H.walk, step_t=0.30)),
        "p1b": rp(H, walk=rp(H.walk, step_t=0.30), step_len=0.30),
        "p1c": rp(H, walk=rp(H.walk, step_t=0.30), step_len=0.40),
        "m035": rp(H, walk=rp(H.walk, step_t=0.20 / 0.35)),
        "m025": rp(H, walk=rp(H.walk, step_t=0.80)),
    }


def walk(dr, backend: str, g, x: dict, ref_frame: np.ndarray, goal_xy, goal_yaw: float, pose, info: dict | None = None):
    """lc3_run.walk_ctrl with the HEFT gait `g` as a parameter (g = gait_heft.HEFT: walk_ctrl bit for bit)."""
    if backend != "heft" or g is None:  # ScaleBFM / SONIC, and HEFT on hg_walk (p0hg)
        return HW.walk_frames(dr, x, ref_frame, goal_xy, goal_yaw, pose=pose, info=info)
    G, GH = LC.heft_gait()
    if not hasattr(dr, "_heft_legs"):
        dr._heft_legs = G.LegModel(dr.pl.ik)
    wf = GH.walk_frames(dr._heft_legs, ref_frame, pose, goal_xy, goal_yaw, g, info)
    q0 = np.asarray(ref_frame[7:], float)
    t = (np.arange(len(wf)) + 1) / FPS
    bl = np.array([L.minjerk(min(1.0, v / HW.BLEND_S)) for v in t])
    wf[:, 7 + 12 :] = (q0[12:][None] + bl[:, None] * (L.DEFAULT_Q[12:] - q0[12:])[None]).astype(np.float32)
    return wf


def hold_end_f(backend: str, g) -> int:
    return int(round((g.hold_end_s if backend == "heft" and g is not None else HW.HOLD_END_S) * FPS))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_driver(backend: str):
    if backend == "sonic":
        import d3_sonic

        base = d3_sonic.SonicHost
    else:
        base = LR.HeftBase if backend == "heft" else LR.LH.HostDriver
    cls = type(f"HW_{backend}", (base,), {"plan": LR.plan_rec})
    dr = cls(env_name="EmptyEnv", seed=0)
    if not hasattr(dr, "plan_s"):
        dr.plan_s = 0.0
    return dr


class Body:
    """Geometry of the robot on the host model: the feet's loads (foot force sensor), the frontmost low point (truth, on
    a scratch copy of the state so the live data are never touched) and the feet's reach by the robot's own FK."""

    def __init__(self, dr):
        mj, m = dr.mj, dr.m
        self.mj, self.m = mj, m
        bn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        self.robot_body = np.array([n.startswith("robot0_") for n in bn])
        self.robot_geoms = np.array([g for g in range(m.ngeom) if self.robot_body[m.geom_bodyid[g]]])
        feet = [next(i for i, n in enumerate(bn) if n == f"robot0_{s}_ankle_roll_link") for s in ("left", "right")]
        self.foot_geoms = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in feet])
        self.gside = {int(g): k for k, b in enumerate(feet) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b}
        self.scratch = mj.MjData(m)
        self.f6 = np.zeros(6)
        self.rq, self.qadr, self.c = dr.rq, dr.qadr, np.asarray(dr.c, float)
        self.bname = [bn[int(m.geom_bodyid[g])] for g in range(m.ngeom)]

    def load(self, d) -> np.ndarray:
        """Normal force on each foot (left, right) from everything that is not the robot."""
        w = np.zeros(2)
        m = self.m
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            s = self.gside.get(g1, self.gside.get(g2))
            if s is None:
                continue
            o = g2 if g1 in self.gside else g1
            if self.robot_body[m.geom_bodyid[o]]:
                continue
            self.mj.mj_contactForce(m, d, i, self.f6)
            w[s] += abs(float(self.f6[0]))
        return w

    def _front(self, d, geoms, origin, h, top) -> tuple[float, int]:
        m = self.m
        rot = d.geom_xmat[geoms].reshape(-1, 3, 3)
        centre = d.geom_xpos[geoms] + np.einsum("kij,kj->ki", rot, m.geom_aabb[geoms, :3])
        half = np.einsum("kij,kj->ki", np.abs(rot), m.geom_aabb[geoms, 3:])
        low = centre[:, 2] - half[:, 2] < top if top is not None else np.ones(len(geoms), bool)
        v = np.where(low, (centre[:, :2] - origin) @ h + half[:, :2] @ np.abs(h), -np.inf)
        k = int(np.argmax(v))
        return float(v[k]), int(geoms[k])

    def frontmost(self, d, origin, h) -> tuple[float, float, str]:
        """Truth, on a scratch copy of the state: (the feet's frontmost point -- the ankle_roll_link geoms' world AABB, the
        toes: revision 1's crossing and gap --, bfm_brake.frontmost_low over every robot geom whose world AABB bottom is
        below OBSTACLE_TOP -- the registered definition, kept as the `_low` fields --, the body that gives the latter)."""
        s = self.scratch
        s.qpos[:] = d.qpos
        self.mj.mj_kinematics(self.m, s)
        low, g = self._front(s, self.robot_geoms, origin, h, OBSTACLE_TOP)
        feet, _ = self._front(s, self.foot_geoms, origin, h, None)
        return feet, low, self.bname[g]

    def reach_own(self, x: dict, h) -> float:
        """The feet's frontmost point ahead of the pelvis (robot_fk convention) along h: FK of the encoders with the IMU's
        attitude, pelvis at the origin -- the robot's own kinematics, no simulator position."""
        s = self.scratch
        s.qpos[:] = 0.0
        s.qpos[self.rq + 3 : self.rq + 7] = x["root"][3:]
        s.qpos[self.qadr] = x["q"]
        self.mj.mj_kinematics(self.m, s)
        pel = L._qmat(x["root"][3:]) @ self.c  # the pelvis point (host body origin 0 + R c)
        return self._front(s, self.foot_geoms, pel[:2], h, None)[0]


class Lift:
    """Lift-off events per foot: load < LOAD_OFF_N for LOAD_OFF_F consecutive steps (faildiag's rule)."""

    def __init__(self):
        self.run = np.zeros(2, int)
        self.off = np.zeros(2, bool)
        self.n = np.zeros(2, int)

    def update(self, w: np.ndarray) -> None:
        below = w < LOAD_OFF_N
        self.run = np.where(below, self.run + 1, 0)
        now = self.run >= LOAD_OFF_F
        self.n += (now & ~self.off).astype(int)
        self.off = now

    def total(self) -> int:
        return int(self.n.sum())


def cells_of(arm: str) -> list[int]:
    return [MARKER_CID] if ARMS[arm]["program"] == "marker" else list(range(len(CELLS)))


def trial(dr, body: Body, rig, ps, arm: str, cid: int, rep: int, G: dict, p3: dict | None, trace_dir: Path | None) -> dict:
    spec = dict(ARMS[arm])
    backend = spec["backend"]
    dmin = None
    if arm == "p3":
        spec["gait"], dmin = p3["gait"], p3.get("dmin")
        if dmin:  # the step back must stay an omnidirectional move (a move of move_min or more turns toward its direction)
            assert dmin < G[spec["gait"]].move_min, (dmin, spec["gait"])
    elif arm == "s3dev":
        dmin = p3.get("dmin")
    g = G.get(spec["gait"]) if backend == "heft" else None
    marker = spec["program"] == "marker"
    d_run, lat = (MARKER_M, 0.0) if marker else CELLS[cid]
    t0 = time.time()
    dr.env.rng = np.random.default_rng([TAG, rep, 1])
    dr.reset()
    rig.clear()
    ps.clear()
    frames = dr.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(HOLD_S * FPS), 0)])
    R = dict(arm=arm, backend=backend, gait=spec["gait"], dmin=dmin, cid=cid, rep=rep, d=d_run, lat=lat, program=spec["program"],
             reanchor=spec["reanchor"])  # fmt: skip
    f, fell, stage, open_f, delay = 0, None, "stand", D3.FIRST_OPEN_F, None
    lift = Lift()
    odo = None
    p0 = h = n = None
    yaw0 = 0.0
    s_line = target = None
    walks: list[dict] = []
    queue: list[np.ndarray] = []
    n_re, cross_max, cross_low, low_body, trig, f_fin, final = 0, -np.inf, -np.inf, "", None, None, None
    anchor = None
    tr = []
    src, aim = spec.get("src", "odo"), spec.get("aim", "pelvis")
    reach_cur = None

    def belief(x) -> tuple:
        """The pose a decision reads: the odometry's (P2), or the simulator's for the measurement arms (src true)."""
        if src == "true":
            return x["root"][:2].astype(float).copy(), L.yaw_of(L._qmat(x["root"][3:]))
        return odo.pose()

    def fwd_lat(p) -> tuple[float, float]:
        v = np.asarray(p, float)[:2] - p0
        return float(v @ h), float(v @ n)

    def issue(x, goal_xy, kind: str) -> bool:
        nonlocal frames, open_f, stage, anchor
        pose = belief(x)
        gf, gl = fwd_lat(goal_xy)
        if not marker and gf + (reach_cur if aim == "toe" else R["reach_own"]) > R["s_line"]:  # the planned end stance's frontmost point past the line
            walks.append(dict(kind=kind, f_cmd=f, refused="would cross"))
            return False
        info: dict = {}
        wf = walk(dr, backend, g, x, frames[f], np.asarray(goal_xy, float), yaw0, pose, info)
        frames = np.concatenate([frames[: f + 1], wf, np.repeat(wf[-1:], int(HOLD_S * FPS), 0)])
        anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
        bf, bl = fwd_lat(pose[0])
        tf, tl = fwd_lat(x["root"][:2])
        walks.append(dict(kind=kind, f_cmd=f, goal_fwd=gf, goal_lat=gl, asked_fwd=gf - bf, asked_lat=gl - bl,
                          belief=[bf, bl, math.degrees(wrap(pose[1] - yaw0))], truth=[tf, tl, math.degrees(wrap(L.yaw_of(L._qmat(x["root"][3:])) - yaw0))],
                          plan=dict(n_steps=info.get("n_steps"), walk_s=info.get("walk_s"), total_s=info.get("total_s"), kinds=info.get("kinds"), ik_err_cm=info.get("ik_err_cm")),
                          lift0=lift.total(), len_f=len(wf), lead_max_cm=0.0))  # fmt: skip
        open_f = f + len(wf) - hold_end_f(backend, g) + delay
        stage = "walk"
        return True

    def plan_walk(x) -> bool:
        """The next walk toward T from the believed pose (split under dmin)."""
        rem = target - belief(x)[0]
        if dmin and float(np.linalg.norm(rem)) < dmin:
            queue.append(target.copy())
            return issue(x, belief(x)[0] - dmin * h, "back")
        return issue(x, target, "walk" if not walks else "reissue")

    while True:
        dr.push(frames[min(f, len(frames) - 1)])
        if dr.buf is None:
            dr.buf = dr._find_buffer()
        dr.step()
        if delay is None:
            delay = D3.ref_delay_f(dr)
        if f == 1 and getattr(getattr(dr, "comp", None), "_settling", False):
            raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        x = dr.sample()
        rig.record(x)
        ps.update(x)
        w = body.load(dr.d)
        if odo is not None:
            odo.update(dr, x)
            lift.update(w)
            fm, fl, fb = body.frontmost(dr.d, p0, h)
            fm, fl = fm - (MARKER_M if marker else s_line), fl - (MARKER_M if marker else s_line)
            cross_max = max(cross_max, fm)
            if fl > cross_low:
                cross_low, low_body = fl, fb
            if walks and stage == "walk" and anchor is not None:
                fr = frames[min(max(f - delay, anchor[2]), len(frames) - 1)]
                lead = float(((fr[:2] - anchor[1]) - (x["root"][:2] - anchor[0])) @ h)
                walks[-1]["lead_max_cm"] = max(walks[-1]["lead_max_cm"], 100 * lead)
            if trace_dir is not None:
                op, oy = odo.pose()
                ref_w = frames[min(f, len(frames) - 1)]
                tr.append(np.r_[f, x["root"][:2], L.yaw_of(L._qmat(x["root"][3:])), op, oy, w, fm, fl, ref_w[:2],
                                ("walk", "wait", "post").index(stage) if stage in ("walk", "wait", "post") else -1])  # fmt: skip
        if float(x["root"][2]) < L.FALL_Z:
            fell = f
            break
        if stage == "stand":
            go = "settled" if (f >= open_f + SETTLE_F and ps.quiet()) else "timeout" if f >= open_f + SETTLE_MAX_F else None
            if go:
                p0 = x["root"][:2].astype(float).copy()
                yaw0 = L.yaw_of(L._qmat(x["root"][3:]))
                h = np.array([math.cos(yaw0), math.sin(yaw0)])
                n = np.array([-h[1], h[0]])
                odo = HW.LegOdo(dr, x, (p0.copy(), yaw0))
                reach_true, reach_low, _ = body.frontmost(dr.d, p0, h)
                reach_own = body.reach_own(x, h)
                reach_cur = reach_own
                j = 0.0 if marker else float(np.random.default_rng([TAG, cid, rep, 2]).uniform(-JIT, JIT))
                s_line = reach_true + MARGIN + d_run + j
                R.update(f_cmd0=f, stand=go, stand_wait_s=(f - open_f) / FPS, jitter=j, reach_true=reach_true, reach_true_low=reach_low, reach_own=reach_own,
                         s_line=s_line, p0=p0.tolist(), yaw0=yaw0, delay_f=delay)  # fmt: skip
                if marker:
                    target = p0 + MARKER_RUN * h
                    issue(x, target, "walk")
                else:
                    target = p0 + (s_line - reach_own - MARGIN) * h + lat * n
                    R["target"] = [float((target - p0) @ h), float((target - p0) @ n)]
                    plan_walk(x)
        elif stage == "walk":
            if marker and trig is None:
                bf, _ = fwd_lat(odo.pose()[0])
                if bf >= MARKER_M:
                    tf, tl = fwd_lat(x["root"][:2])
                    trig = dict(f=f, belief_fwd=bf, truth_fwd=tf, truth_lat=tl)
                    issue(x, odo.pose()[0], "stop")
            if stage == "walk" and f >= open_f:
                stage = "wait"
        elif stage == "wait":
            go = "settled" if (f >= open_f + SETTLE_F and ps.quiet()) else "timeout" if f >= open_f + SETTLE_MAX_F else None
            if go:
                wk = walks[-1]
                tf, tl = fwd_lat(x["root"][:2])
                op, oy = odo.pose()
                bf, bl = fwd_lat(op)
                wk.update(f_go=f, how=go, wait_s=(f - open_f) / FPS, truth_settled=bool(rig.quiet()), lift=lift.total() - wk["lift0"],
                          moved_fwd=tf - wk["truth"][0], moved_lat=tl - wk["truth"][1], belief_end=[bf, bl], truth_end=[tf, tl])  # fmt: skip
                if not marker:
                    wk.update(reach_own_end=body.reach_own(x, h), front_past_end=body.frontmost(dr.d, p0, h)[0] - s_line)
                nxt = False
                if queue:  # the forward leg of a split
                    nxt = issue(x, queue.pop(0), "fwd")
                elif spec["reanchor"]:
                    if aim == "toe":  # P4: the stance the robot now stands in sets where the pelvis must go
                        reach_cur = wk["reach_own_end"]
                        target = p0 + (s_line - reach_cur - MARGIN) * h + lat * n
                        wk["aim_fwd"] = float((target - p0) @ h)
                    rem = target - belief(x)[0]
                    wk["rem_belief_cm"] = 100 * float(np.linalg.norm(rem))
                    if n_re < REISSUE_MAX and float(np.linalg.norm(rem)) >= REM_MIN:
                        n_re += 1
                        nxt = plan_walk(x)
                if not nxt:
                    f_fin, stage = f, "post"
                    yaw = L.yaw_of(L._qmat(x["root"][3:]))
                    fm, fl, _ = body.frontmost(dr.d, p0, h)
                    final = dict(how=go, fwd=tf, lat_pos=tl, yaw_deg=math.degrees(wrap(yaw - yaw0)), front=fm, front_low=fl,
                                 odo=[bf, bl, math.degrees(wrap(oy - yaw0))],
                                 odo_err_cm=100 * float(np.linalg.norm(op - x["root"][:2])), odo_yaw_err_deg=math.degrees(wrap(oy - yaw)))  # fmt: skip
        elif stage == "post":
            if f >= f_fin + int(POST_S * FPS):
                break
        if f * DT > T_MAX_S:
            R["timeout"] = True
            break
        f += 1
    R.update(walks=walks, n_reissue=n_re, fell=fell, steps=f, liftoffs=lift.total(), liftoffs_lr=lift.n.tolist(), wall_s=round(time.time() - t0, 2))
    if odo is not None:
        R.update(cross_m=float(cross_max), cross_low_m=float(cross_low), cross_low_body=low_body)
    if final is not None and fell is None:
        R["final"] = final
        R["settle_s"] = (f_fin - R["f_cmd0"]) / FPS
        if marker:
            R.update(trigger=trig, pelvis_past_m=final["fwd"] - MARKER_M, toes_past_m=final["front"] - MARKER_M, low_past_m=final["front_low"] - MARKER_M,
                     carry_m=None if trig is None else final["fwd"] - trig["truth_fwd"])  # fmt: skip
        else:
            tf_, tl_ = R["target"]
            gap = s_line - final["front"]
            R.update(gap_m=gap, gap_low_m=s_line - final["front_low"], err_fwd_m=final["fwd"] - tf_, err_lat_m=final["lat_pos"] - tl_,
                     pelvis_err_m=math.hypot(final["fwd"] - tf_, final["lat_pos"] - tl_))  # fmt: skip
            ok = (cross_max <= 0.0 and GAP_OK[0] <= gap <= GAP_OK[1] and abs(R["err_lat_m"]) <= LAT_OK and final["how"] == "settled"
                  and not R.get("timeout"))  # fmt: skip
            R["success"] = bool(ok)
    else:
        R["success"] = False
    if walks:
        w0 = walks[0]
        R["first"] = dict(asked_fwd=w0.get("asked_fwd"), moved_fwd=w0.get("moved_fwd"), lift=w0.get("lift"), how=w0.get("how"),
                          n_steps=w0["plan"]["n_steps"] if "plan" in w0 else None)  # fmt: skip
    if trace_dir is not None and tr:
        trace_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_dir / f"{arm}_c{cid}_r{rep}.npz", rows=np.array(tr),
                            cols=np.array(["f", "x", "y", "yaw", "ox", "oy", "oyaw", "load_l", "load_r", "front_past", "front_low_past", "ref_x", "ref_y", "stage"]))  # fmt: skip
    return R


def code_check(tag: str) -> None:
    """lc3_run.code_check without its env.py requirement (this runner loads no self model): with LC_CODE_ROOT set, every
    loaded module from the shared repo tree or the copy, and this file and lc3_run themselves, must lie in the copy."""
    root = os.environ.get("LC_CODE_ROOT")
    if not root:
        return
    root_p, shared = Path(root).resolve(), LC.SHARED_TREE.resolve()
    inside = lambda p: p == root_p or root_p in p.parents  # noqa: E731
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and shared in Path(f).resolve().parents and not inside(Path(f).resolve()):
            bad.append(f"{name} {Path(f).resolve()}")
    own = [Path(__file__).resolve(), Path(LC.__file__).resolve()]
    bad += [f"{p}" for p in own if not inside(p)]
    print(f"hw code check {tag}: {Path(__file__).resolve()}; modules outside {root_p}: {len(bad)}", flush=True)
    if bad:
        raise SystemExit(f"hw code check {tag}: FAIL -- outside the copy: {bad[:8]}")


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def done_jobs(out: Path) -> set:
    s = set()
    for fn in glob.glob(str(out / "*" / "runs" / "*.jsonl")):
        for line in open(fn):
            try:
                r = json.loads(line)
                s.add((r["arm"], r["cid"], r["rep"]))
            except Exception:  # noqa: BLE001 -- a line cut by a kill
                pass
    return s


def rng_of(spec: str) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(v) for v in spec.split(",")]


def report(r: dict) -> None:
    """The trial's one-line summary."""
    arm, cid, rep = r["arm"], r["cid"], r["rep"]
    if "error" in r:
        print(f"hw {arm} c{cid} r{rep}: ERROR {r['error']}", flush=True)
        return
    fin = r.get("final") or {}
    print(f"hw {arm} c{cid} r{rep} d {r['d']:.2f} lat {r['lat']:+.2f}: "
          + (f"carry {100 * r['carry_m']:+.1f} cm pelvis past {100 * r['pelvis_past_m']:+.1f} toes past {100 * r['toes_past_m']:+.1f}" if r.get("carry_m") is not None else
             f"success {r.get('success')} gap {100 * r['gap_m']:+.1f} cm cross {100 * r['cross_m']:+.1f} lat {100 * r['err_lat_m']:+.1f} fwd err {100 * r['err_fwd_m']:+.1f} "
             f"yaw {fin.get('yaw_deg', 0):+.1f} lift {r['liftoffs']} reissue {r['n_reissue']} settle {r.get('settle_s')} s {fin.get('how')} odo {fin.get('odo_err_cm', 0):.1f} cm"
             f" (low: gap {100 * r['gap_low_m']:+.1f} cross {100 * r['cross_low_m']:+.1f} {r['cross_low_body'].replace('robot0_', '')})"
             if "gap_m" in r else f"fell {r.get('fell')} timeout {r.get('timeout')}")
          + f" first: asked {100 * (r.get('first') or {}).get('asked_fwd', float('nan')):.1f} moved {100 * ((r.get('first') or {}).get('moved_fwd') or float('nan')):.1f} cm"
          f" [{r['wall_s']} s]", flush=True)  # fmt: skip


def run(a) -> None:
    out = Path(a.out)
    arms = a.arms.split(",")
    backends = {ARMS[x]["backend"] for x in arms}
    assert len(backends) == 1, "one backend per process"
    backend = backends.pop()
    p3 = None
    if "p3" in arms or "s3dev" in arms:
        p3 = json.loads(Path(a.p3).read_text()) if a.p3 and Path(a.p3).exists() else json.loads(a.p3)
    cells = rng_of(a.cells)
    jobs = [(arm, cid, rep) for arm in arms for cid in cells_of(arm) if (cid in cells or cid == MARKER_CID) for rep in rng_of(a.reps)]
    jobs = jobs[a.shard :: a.shards]
    done = done_jobs(out)
    todo = [j for j in jobs if j not in done]
    print(f"hw {backend} shard {a.shard}/{a.shards}: {len(jobs) - len(todo)}/{len(jobs)} done", flush=True)
    if not todo:
        return
    code_check("start")
    dr = make_driver(backend)
    body, rig, ps = Body(dr), D3.Rig(dr), LC.PropSettle(dr)
    G = gaits()
    d = out / backend / "runs"
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"n{a.shards}_{a.shard}_{os.getpid()}.jsonl"
    for arm, cid, rep in todo:
        if (out / "STOP").exists():
            break
        try:
            r = trial(dr, body, rig, ps, arm, cid, rep, G, p3, out / backend / "traces" if a.traces else None)
        except Exception as e:  # noqa: BLE001 -- recorded, the shard goes on
            r = dict(arm=arm, cid=cid, rep=rep, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:])
        with open(fn, "a") as fh:
            fh.write(json.dumps(r, default=_js) + "\n")
        report(r)
    code_check("end")


def selftest(a) -> None:
    """walk(gait_heft.HEFT) == lc3_run.walk_ctrl on a live start state, and the gait table (steps, durations)."""
    dr = make_driver("heft")
    dr.env.rng = np.random.default_rng([TAG, 0, 1])
    dr.reset()
    fr = dr.start_frames()
    for k in range(40):
        dr.push(fr[min(k, len(fr) - 1)])
        if dr.buf is None:
            dr.buf = dr._find_buffer()
        dr.step()
    x = dr.sample()
    p, yaw = x["root"][:2].astype(float), L.yaw_of(L._qmat(x["root"][3:]))
    hh = np.array([math.cos(yaw), math.sin(yaw)])
    G = gaits()
    for dd in (0.05, 0.2, 0.5, 1.5):
        g_xy = p + dd * hh
        a1 = walk(dr, "heft", G["p0"], x, fr[-1], g_xy, yaw, (p, yaw))
        a2 = LC.walk_ctrl(dr, "heft", x, fr[-1], g_xy, yaw, (p, yaw))
        print(f"selftest d {dd}: walk(p0) == walk_ctrl: {a1.shape == a2.shape and np.array_equal(a1, a2)} ({a1.shape})", flush=True)
    print("gait table (straight ahead): d, gait, steps, walk_s, total_s, cruise m/s")
    for gk in ("p0", "p1a", "p1b", "p1c", "m035", "m025"):
        g = G[gk]
        for dd in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 1.50):
            info: dict = {}
            walk(dr, "heft", g, x, fr[-1], p + dd * hh, yaw, (p, yaw), info)
            print(f"  {dd:.2f} {gk:5s} steps {info['n_steps']} walk {info['walk_s']:.2f} s total {info['total_s']:.2f} s cruise {g.step_len / g.walk.step_t:.2f} "
                  f"(step {g.step_len:.2f} m / {g.walk.step_t:.3f} s) kinds {info['kinds']}", flush=True)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["run", "selftest"])
    ap.add_argument("--arms", default="p0")
    ap.add_argument("--cells", default="0:11")
    ap.add_argument("--reps", default="0:10")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--p3", default="", help="p3's rule: a JSON file from hw_p3rule.py or an inline JSON {gait, dmin}")
    ap.add_argument("--traces", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    run(a) if a.phase == "run" else selftest(a)


if __name__ == "__main__":
    main()
