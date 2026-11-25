"""HumanoidArena bench for humanoidarena_20261124: episodes of P&PBox, DoubleDesk and OpenDoor under HumanoidArena's own
SONIC controller, the right hand's reach done by method A or J, one jsonl row and one npz film per episode.

Every episode (same for both methods, see TASK RULES below):
  1. reset (the official episode seed of task|seed|repeat), 50 steps standing;
  2. look: one front-camera RGB-D frame; SAM 3 masks for the task's text prompts; the target from a written geometric
     rule on the masked points (perceive.py). Nothing reads the simulator's objects: only the image, the depth, and the
     robot's own state (joints, root pose, its own link poses for the camera mount and to drop its own pixels);
  3. approach (all three tasks start 1-3 m from their target, out of reach): per round, the stand point from the
     latest estimate; within 0.12 m: done; within 20 deg of the stand heading: walk straight along it; else turn to
     the stand point, walk, turn to the stand heading; look again; up to 3 rounds. Walking is SONIC's own velocity
     planner (walker.py) feeding the 40-D frame's legs: WALK at 0.5 m/s until 0.10 m short (it coasts the rest), turns
     with the planner's facing plus an integrator on the measured yaw (to 3 deg);
     Stand rule for a target on a support (table, desk, shelf board, the basket's table): the support's top-level points
     within 1.2 m of the target, their min-area rectangle's axes; of the four directions with the robot on the near side
     the one whose boundary is nearest the target (1st percentile of the points within 0.3 m sideways) is the edge we
     face squarely; the pelvis CLEAR = 0.16 m off that edge (the edge kept in the world from the first look that saw
     it), the target at least the task's nominal distance ahead;
  4. a final look at the stand: the perceived grasp and the pregrasp (8 cm out along the approach), in the pelvis frame;
  5. the reach, the only part that differs between methods:
       A  the right hand's grasp point on a straight line to the pregrasp (orientation slerped), min-jerk in time over
          max(0.6 s, length / 0.15 m/s) (Motor.reach_joints' timing), damped least squares on the 7 right-arm joints
          every control step from the reference's own arm pose; every other joint and the root hold;
       J  cuRobo (plan_server.py) plans the 7 right-arm joints, waist locked (given everything in the virtual pelvis
          of the reference's waist), from the measured arm to the pregrasp with the final look's voxels as obstacles;
          the path is played as Motor.reach_joints plays it (min-jerk in the grasp point's arc length, the start gap
          faded); no plan -> A, recorded;
     then up to 1.5 s of settling (ends early at 1 cm after 0.3 s), and the hand error to the perceived pregrasp is
     scored from the sim's palm pose (this score only);
  6. the task's fixed post-reach sequence (TASK RULES), identical across methods;
  7. scoring as sim_eval_vla's episode loop: success when the official reward reaches 1, a fall confirmed by its fall
     detector, the official step budget. The episode ends at the first of these; when the sequence finishes first, the
     robot holds still for --hold steps more (to the budget with --full_budget).
  The row also carries failure-analysis diagnostics read from the sim every 10 steps (never fed back): the object's
  pose (box, hammer) or the door's latch / handle / leaf state.

TASK RULES (fixed before the smoke test on dev episodes 100/101; tried on dev seeds 104/105 only)
  ppbox (two hands; the 21 cm box does not fit one Dex3 hand): target "cardboard box"/"box": the top face's points
    (within 2 cm of the 95th-percentile height) give the box's centre and yaw (min-area rectangle), the support level
    under it its height. Stand: the stand rule on the table, the box centre at least 0.34 m ahead. Squeeze axis: the box
    axis nearest the stand's left; grasp: each hand's palm-frame point (0.06, +-0.06, 0) (the grasp side's surface: the
    open thumb) on its side face's centre, palm x along the box's other axis (forward), y along the squeeze axis (left),
    z up; pregrasp 8 cm out along the face normal. Sequence: the LEFT hand to its pregrasp (scripted line, before the
    reach, same for both methods), the right hand's reach (A/J), both hands 10 cm in (2 cm into the faces: a squeeze),
    close both hands, lift 10 cm, look, then turn to the episode's initial heading +45/+90/+135 deg until SAM 3 sees
    "blue metal shelf" ("shelf" also matches the table's own lower shelf), the target board = the highest horizontal
    level of the shelf's points below the box's bottom (its min-area rectangle's centre), the stand rule on that board
    with its centre at least 0.34 m ahead, move the box (both hands) over the board centre, its bottom 3 cm above the
    board, open both hands, hold.
  doubledesk (one hand): target "hammer" (looks: front, then 45 and 90 deg right): median of its points; the handle
    axis from the points' principal axis in the plane; grasp height = top of the points within 3 cm of the median - 1.2
    cm. Stand: the stand rule on the desk, the hammer at least 0.36 m ahead and 0.20 m right. Top-down grasp: palm grasp
    side down, the handle along palm z (the fist closes around it), fingers level and across the handle, toward
    forward-and-right. Grasp point (palm frame) (0.07, 0.04, 0.015): the closed fist's fingertip centre. Pregrasp 8 cm
    above. Sequence: down 8 cm, close the right hand, lift 15 cm, look, then turn to the initial heading +45/+90/+135
    deg until SAM 3 sees "bin"/"basket", its points' top 3 cm (rim) give the centre, the stand rule on its table with
    the rim centre at least 0.40 m ahead and 0.20 m right, the grasp point to 15 cm over the rim centre, open, hold.
  opendoor (one hand): target "door handle" (looks: front, then 45 deg left, then 45 deg right): the handle's points'
    median; the door's plane from "door" points within 0.6 m (normal toward the robot). Stand: facing the door, the
    handle 0.40 m ahead and 0.20 m right. Grasp: a hook over the lever: grasp point (0.07, 0.04, 0.015), fingers into
    the door, grasp side down; pregrasp 8 cm out from the door. Sequence: 8 cm in, close the hand, press 6 cm down, push
    15 cm into the door, open the hand, walk 1.0 m forward (the door is pushed: whether it opens away is not known to
    the script), hold.
  No lean: HumanoidArena's SONIC does not track the waist's pitch (0.3 rad commanded, 0.02-0.07 measured), so every
  stance is upright (LEAN = 0).

    GPU=3 bash run_ha.sh ha_bench.py --ha_task ppbox --method A --seeds 100,101 --repeats 0-4 --out <dir> \
        [--plan_sock /tmp/ha_plan3.sock]
-> <out>/<task>_<method>.jsonl (one row per episode), <out>/film/<task>_<method>_s<seed>_r<repeat>.npz, looks
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing.connection import Client

import g1arm
import ha_sim as H
import numpy as np
import perceive as P

PRE = 0.08  # m, pregrasp distance along the approach
G_FIST = (0.07, 0.04, 0.015)  # right palm frame: the closed fist's fingertip centre (Dex3 FK, 2026-11-24)
G_BOX = {"right": (0.06, 0.06, 0.0), "left": (0.06, -0.06, 0.0)}  # the grasp side's surface (the open thumb)
REACH_SPEED, REACH_MIN_S, SETTLE_S, SETTLE_TOL = 0.15, 0.6, 1.5, 0.01  # Motor.reach_joints' defaults
WALK_SPEED, WALK_STOP_LEAD, WALK_MIN = 0.5, 0.10, 0.12  # gait_dev 2026-11-24: WALK at 0.5 coasts ~0.06-0.13 m
VOX_BOX = ((0.0, 1.0), (-0.9, 0.6), (-0.6, 0.8))  # pelvis frame: what can reach the right arm
SHOULDER_R = np.array([0.0, -0.1, 0.292])
CLEAR = 0.16  # m, pelvis to a support's near edge at the stand (WALK at 0.5 coasts a few cm past its stop)
# rad, waist pitch at the stand. 0: HumanoidArena's SONIC does not track the waist's pitch (debug 2026-11-24: 0.3 rad
# commanded, 0.02-0.07 measured, and the arms then miss), so the stance stays upright
LEAN = dict(ppbox=0.0, doubledesk=0.0, opendoor=0.0)


class End(Exception):
    pass


def expm(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def minjerk(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def quat_wxyz(Rm):
    """Shepperd's method (robust near 180 deg)."""
    m = np.asarray(Rm, float)
    t = np.trace(m)
    i = int(np.argmax([t, m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        r = np.sqrt(1 + t)
        q = [0.5 * r, (m[2, 1] - m[1, 2]) / (2 * r), (m[0, 2] - m[2, 0]) / (2 * r), (m[1, 0] - m[0, 1]) / (2 * r)]
    else:
        j, k = (i % 3) + 1, ((i + 1) % 3) + 1
        a, b, c = i - 1, j - 1, k - 1
        r = np.sqrt(1 + m[a, a] - m[b, b] - m[c, c])
        q = np.zeros(4)
        q[0] = (m[c, b] - m[b, c]) / (2 * r)
        q[i], q[j], q[k] = 0.5 * r, (m[a, b] + m[b, a]) / (2 * r), (m[a, c] + m[c, a]) / (2 * r)
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def heading_R(yaw):
    return H.rot_z(yaw)


class Bench:
    def __init__(self, a):
        self.a = a
        self.sim = H.Sim(a)
        self.W = self.sim.W
        self.chain = {s: g1arm.Chain(tip=f"{s}_hand_palm_link") for s in ("left", "right")}
        self.torso = g1arm.Chain(tip="torso_link")
        self.sam = P.Sam()
        self.plan_conn = None

    # ---- stepping, recording
    def step(self, phase: str):
        s = self.sim
        s.step()
        f = self.film
        f["ref"].append(s.frame[9:38].copy())
        f["q"].append(s.q().astype(np.float32))
        f["root"].append(s.root().astype(np.float32))
        f["wrist"].append(s.body("right_wrist_yaw_link").astype(np.float32))
        f["palm"].append(s.body("right_hand_palm_link").astype(np.float32))
        f["phase"].append(self.phase_id.setdefault(phase, len(self.phase_id)))
        self.phase = phase
        if s.k % 10 == 0:
            self.diag_step()
        if s.success_at is not None:
            raise End("success")
        if s.fell_at is not None:
            raise End("fell")
        if s.k >= self.a.max_steps:
            raise End("budget")

    def diag_step(self):
        """Failure analysis only (never fed to the controller): the object's pose, or the door's state, every 10 steps."""
        env = self.sim.env
        try:
            if self.a.ha_task == "opendoor":
                d = env.cfg.get_open_door_reward_diagnostics(env)
                v = [float(np.asarray(d["latch_unlocked"].cpu() if hasattr(d["latch_unlocked"], "cpu") else d["latch_unlocked"]).ravel()[0]),
                     float(d["handle_angle_deg"].ravel()[0]), float(d["leaf_angle_deg"].ravel()[0])]  # fmt: skip
            else:
                key = "box" if self.a.ha_task == "ppbox" else "object_l"
                v = env.scene[key].data.root_pos_w[0].cpu().numpy().astype(float).tolist()
        except Exception as e:  # diagnostics must never end an episode
            v = [np.nan, np.nan, np.nan]
            if not self.diag_err:
                self.diag_err = f"{type(e).__name__}: {e}"
        self.diag.append([self.sim.k] + list(v))

    def diag_summary(self):
        D = np.asarray(self.diag, float)
        if not len(D) or np.isnan(D[:, 1:]).all():
            return dict(err=self.diag_err)
        if self.a.ha_task == "opendoor":
            return dict(latch_unlocked=bool(np.nanmax(D[:, 1]) > 0.5), handle_min_deg=round(float(np.nanmin(D[:, 2])), 1),
                        leaf_max_abs_deg=round(float(np.nanmax(np.abs(D[:, 3]))), 1))  # fmt: skip
        z0 = D[0, 3]
        return dict(obj_start=D[0, 1:].round(3).tolist(), obj_end=D[-1, 1:].round(3).tolist(),
                    obj_max_lift_cm=round(100 * float(np.nanmax(D[:, 3]) - z0), 1),
                    obj_moved_cm=round(100 * float(np.linalg.norm(D[-1, 1:3] - D[0, 1:3])), 1))  # fmt: skip

    def hold(self, n, phase="hold"):
        for _ in range(int(n)):
            self.step(phase)

    # ---- the robot's own state
    def yaw(self):
        return H.yaw_of(self.sim.root()[3:])

    def pelvis(self):
        return self.sim.pelvis()

    def to_pelvis(self, pw):
        pp, pR = self.pelvis()
        return pR.T @ (np.asarray(pw, float) - pp)

    def ref_q(self):
        return {H.ORDER[i]: float(self.sim.upper[i]) for i in H.UPPER}

    def set_arm(self, side, qd):
        for n in H.ARM[side]:
            self.sim.upper[H.ORDER.index(n)] = qd[n]

    def grasp_point_meas(self, side, off):
        """The grasp point and palm rotation from the sim's palm link, in the pelvis frame now (scoring only)."""
        b = self.sim.body(f"{side}_hand_palm_link")
        pp, pR = self.pelvis()
        Rw = H.mat_of(b[3:])
        return pR.T @ (b[:3] + Rw @ np.asarray(off) - pp), pR.T @ Rw

    # ---- locomotion (closed on the measured root)
    def face(self, yaw):
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def turn_to(self, yaw_tgt, tol_deg=3.0, n_max=300, phase="turn"):
        """IDLE with the planner's facing; the planner's turn lands ~20% short (gait_dev), so every 50 steps the
        remaining error is added to the commanded facing (an integrator)."""
        off, e = 0.0, wrap(yaw_tgt - self.yaw())
        if abs(e) < np.radians(tol_deg):
            return e
        for k in range(n_max):
            if k % 25 == 24:
                e = wrap(yaw_tgt - self.yaw())
                if abs(e) < np.radians(tol_deg):
                    break
                if k % 50 == 49:
                    off += e
            self.sim.move = self.W.Movement(self.W.IDLE, -1.0, np.zeros(3), self.face(yaw_tgt + off))
            self.step(phase)
        return wrap(yaw_tgt - self.yaw())

    def walk_forward(self, dist, phase="walk", n_max=500):
        if dist < WALK_MIN:
            return 0.0
        y = self.yaw()
        d = self.face(y)
        p0 = self.sim.root()[:3].copy()
        stopped = None
        for k in range(n_max):
            prog = float((self.sim.root()[:3] - p0) @ d)
            if stopped is None and prog >= dist - WALK_STOP_LEAD:
                stopped = k
            if stopped is None:
                self.sim.move = self.W.Movement(self.W.WALK, WALK_SPEED, d, d)
            else:
                self.sim.move = self.W.Movement(self.W.IDLE, -1.0, np.zeros(3), d)
                if k >= stopped + 50:
                    break
            self.step(phase)
        return float((self.sim.root()[:3] - p0) @ d)

    # ---- looking
    def own_points(self):
        pp, pR = self.pelvis()
        Pw = self.sim.robot.data.body_link_pos_w[0].cpu().numpy().astype(np.float64)
        return (Pw - pp) @ pR

    def look(self, prompts, tag):
        L = self.sim.look()
        lk = P.Look(L, own_links=self.own_points(), own_r=0.10)
        ms = self.sam.masks(L["rgb"], prompts, floor=0.3)
        self.looks.append(dict(tag=tag, k=self.sim.k, prompts=prompts, n=len(ms), best=ms[0][0] if ms else 0.0))
        if self.a.save_looks:
            np.savez_compressed(f"{self.look_dir}/{self.ep_name}_{len(self.looks):02d}_{tag}.npz", rgb=L["rgb"],
                                depth=L["depth"], K=L["K"], cam_p=L["cam_p"], cam_R=L["cam_R"], pelvis_p=L["pelvis_p"],
                                pelvis_R=L["pelvis_R"], masks=np.array([m for _, m, _ in ms[:4]]) if ms else np.zeros((0,)))  # fmt: skip
        return lk, ms

    def pick(self, lk, ms, near=None, min_px=40, erode_px=1):
        """The instance to use: the best-scoring one with enough valid points, or (with `near`, a world point) the one
        nearest to it among those scoring at least half the best."""
        cands = []
        for s, m, p in ms:
            pts = lk.object_points(m, erode_px=erode_px)
            if len(pts) >= min_px:
                cands.append((s, m, pts))
        if not cands:
            return None, None
        if near is not None:
            best = cands[0][0]
            cands = [c for c in cands if c[0] >= 0.5 * best]
            cands.sort(key=lambda c: np.linalg.norm(np.median(c[2], axis=0)[:2] - np.asarray(near)[:2]))
        return cands[0][1], cands[0][2]

    # ---- arms
    def arm_lines(self, goals: dict, phase, speed=REACH_SPEED, min_s=REACH_MIN_S, settle=0):
        """Each side's grasp point (palm-frame offset) on a straight line from where the reference has it to the goal
        (pelvis frame), orientation slerped, min-jerk over max(min_s, longest / speed); DLS IK every control step."""
        q = self.ref_q()
        start = {}
        for side, (pg, Rg, off) in goals.items():
            p0, R0 = self.chain[side].point(q, off)
            start[side] = (p0, R0, g1arm.rotvec(R0.T @ Rg))
        length = max(float(np.linalg.norm(goals[s][0] - start[s][0])) for s in goals)
        total = max(min_s, length / speed)
        n = int(np.ceil(total / self.sim.dt))
        for k in range(1, n + 1 + int(settle)):
            u = minjerk(min(k * self.sim.dt / total, 1.0))
            for side, (pg, Rg, off) in goals.items():
                p0, R0, rv = start[side]
                q, _, _ = self.chain[side].ik(q, H.ARM[side], p0 + u * (pg - p0), R0 @ expm(u * rv), off=off, iters=8)
                self.set_arm(side, q)
            self.step(phase)
        return length, total

    def arm_shift(self, sides, dp, phase, offs, speed=REACH_SPEED):
        q = self.ref_q()
        goals = {}
        for s in sides:
            p0, R0 = self.chain[s].point(q, offs[s])
            goals[s] = (p0 + np.asarray(dp, float), R0, offs[s])
        return self.arm_lines(goals, phase, speed=speed)

    def settle_score(self, side, pg, off, phase="settle"):
        """Up to SETTLE_S; ends early when the measured grasp point is within SETTLE_TOL after 0.3 s."""
        n = int(round(SETTLE_S / self.sim.dt))
        for k in range(n):
            self.step(phase)
            if k * self.sim.dt >= 0.3 and np.linalg.norm(self.grasp_point_meas(side, off)[0] - pg) <= SETTLE_TOL:
                break
        return float(np.linalg.norm(self.grasp_point_meas(side, off)[0] - pg))

    def reach_A(self, pg, Rg, off):
        L, T = self.arm_lines({"right": (pg, Rg, off)}, "reach")
        return dict(path_cm=round(100 * L, 2), dur_s=round(T, 3))

    def plan(self, req):
        if self.plan_conn is None:
            self.plan_conn = Client(self.a.plan_sock, "AF_UNIX", authkey=b"jref")
        self.plan_conn.send(req)
        return self.plan_conn.recv()

    def reach_J(self, pg, Rg, off, vox):
        qm = self.sim.q()
        qd = {n: float(qm[H.ORDER.index(n)]) for n in H.ARM["right"]}
        palm_p = pg - Rg @ np.asarray(off)
        # cuRobo's waist is locked at 0: give it everything in the VIRTUAL pelvis, where a pelvis with the waist at 0
        # carries the torso where the reference's waist carries it (the arm hangs off the torso: the plan is the same)
        M = self.torso.fk({}) @ np.linalg.inv(self.torso.fk(self.ref_q()))
        palm_p, Rv = M[:3, :3] @ palm_p + M[:3, 3], M[:3, :3] @ Rg
        vox = np.asarray(vox, float).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        req = dict(q=qd, vox=vox, vox_size=P.VOX, goal_p=palm_p, goal_q=quat_wxyz(Rv), attempts=4, timeout=10.0,
                   free=0.01, free_goal=0.02)  # fmt: skip
        t0 = time.time()
        try:
            r = self.plan(req)
        except Exception as e:  # the server is gone: record, fall back
            r = dict(ok=False, status=f"client {type(e).__name__}: {e}", Q=None, t_plan=0.0)
        info = dict(plan_ok=bool(r["ok"]), plan_status=r.get("status", ""), t_plan=r.get("t_plan"),
                    t_plan_wall=round(time.time() - t0, 3), n_vox=r.get("n_vox"), n_vox_in=r.get("n_vox_in"),
                    n_drop_start=r.get("n_drop_start"), n_drop_goal=r.get("n_drop_goal"), goal_ik_ok=r.get("goal_ik_ok"))  # fmt: skip
        if not r["ok"]:
            info.update(self.reach_A(pg, Rg, off), fallback=True)
            return info
        names = list(r["names"])
        Pth = np.asarray(r["Q"], float)[:, [names.index(n) for n in H.ARM["right"]]]
        # Motor.reach_joints: base = the reference now; the start gap faded along the arc; min-jerk in arc length
        base = self.ref_q()
        gap = np.array([base[n] for n in H.ARM["right"]]) - Pth[0]

        def grasp_of(qa):
            qq = dict(base)
            qq.update(zip(H.ARM["right"], qa))
            return self.chain["right"].point(qq, off)[0]

        jaw = np.array([grasp_of(qa) for qa in Pth])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(jaw, axis=0), axis=1))]
        length = float(arc[-1])
        total = max(REACH_MIN_S, length / REACH_SPEED)
        frac = arc / max(length, 1e-9)
        n = int(np.ceil(total / self.sim.dt))
        for k in range(1, n + 1):
            u = minjerk(min(k * self.sim.dt / total, 1.0))
            qa = np.array([np.interp(u, frac, Pth[:, i]) for i in range(7)]) + (1.0 - u) * gap
            self.set_arm("right", dict(zip(H.ARM["right"], qa)))
            self.step("reach")
        info.update(fallback=False, path_cm=round(100 * length, 2), dur_s=round(total, 3), n_wp=len(Pth),
                    start_gap_deg=round(float(np.degrees(np.abs(gap).max())), 2))  # fmt: skip
        return info

    def reach(self, pg, Rg, off, vox):
        """The method's reach to (pg, Rg) for the right hand's point `off` (pelvis frame), then the settle and score."""
        k0 = self.sim.k
        t0 = time.time()
        info = self.reach_A(pg, Rg, off) if self.a.method == "A" else self.reach_J(pg, Rg, off, vox)
        k1 = self.sim.k
        err = self.settle_score("right", pg, off)
        p_m, R_m = self.grasp_point_meas("right", off)
        info.update(reach_k0=k0, reach_k1=k1, settle_k=self.sim.k, hand_err_cm=round(100 * err, 2),
                    hand_err_xyz_cm=(100 * (p_m - pg)).round(2).tolist(),
                    hand_rot_err_deg=round(float(np.degrees(np.linalg.norm(g1arm.rotvec(Rg.T @ R_m)))), 2),
                    reach_wall_s=round(time.time() - t0, 2))  # fmt: skip
        return info

    def vox_for_plan(self, lk):
        Vw = lk.voxels(box=((-5, 5), (-5, 5), (-5, 5)))
        pp, pR = self.pelvis()
        V = (Vw - pp) @ pR if len(Vw) else np.zeros((0, 3))
        (x0, x1), (y0, y1), (z0, z1) = VOX_BOX
        V = V[(V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 1] > y0) & (V[:, 1] < y1) & (V[:, 2] > z0) & (V[:, 2] < z1)]
        V = V[np.argsort(np.linalg.norm(V - SHOULDER_R, axis=1))][:6000]  # the server's cap: keep the nearest
        return V

    # ---- the approach
    def approach(self, find, prompts, rounds=3, phase="approach"):
        """find(lk, ms, prev) -> target dict (or None): world point "p", stand heading "yaw", where the target is to be
        at the stand ("x" ahead, "y" left). Per round: the stand point S from the latest estimate; within 0.12 m: done;
        within 20 deg of the stand heading: walk straight along it (no turns); else turn to S, walk, turn to the heading;
        then look. Ends with the stand heading, the task's lean and a look there."""
        tgt = self.target
        for it in range(rounds):
            S = np.asarray(tgt["p"][:2]) - heading_R(tgt["yaw"])[:2, :2] @ np.array([tgt["x"], tgt["y"]])
            d = S - self.sim.root()[:2]
            if np.linalg.norm(d) < 0.12:
                break
            f = self.face(tgt["yaw"])[:2]
            if abs(wrap(np.arctan2(d[1], d[0]) - tgt["yaw"])) < np.radians(20):
                if d @ f < WALK_MIN:
                    break  # only a sideways error or a short one: we cannot step that finely
                self.turn_to(tgt["yaw"], phase=phase)
                self.walk_forward(float(d @ f), phase=phase)
            else:
                self.turn_to(float(np.arctan2(d[1], d[0])), phase=phase)
                self.walk_forward(float(np.linalg.norm(S - self.sim.root()[:2])), phase=phase)
                self.turn_to(tgt["yaw"], phase=phase)
            lk, ms = self.look(prompts, f"approach{it}")
            t = find(lk, ms, tgt)
            tgt = tgt if t is None else t
            self.target = tgt
        self.turn_to(tgt["yaw"], phase=phase)
        if self.lean_rad:
            self.lean(self.lean_rad)
        lk, ms = self.look(prompts, "stand")
        t = find(lk, ms, tgt)
        self.target = tgt if t is None else t
        S = np.asarray(self.target["p"][:2]) - heading_R(self.target["yaw"])[:2, :2] @ np.array(
            [self.target["x"], self.target["y"]]
        )
        self.stand_err = (heading_R(self.target["yaw"])[:2, :2].T @ (self.sim.root()[:2] - S)).round(3).tolist()
        return lk, t is not None

    def lean(self, rad, dur=1.0):
        """The waist's pitch to `rad` (min-jerk, `dur` s): a stance for both methods (the reach is arm-only)."""
        i = H.ORDER.index("waist_pitch_joint")
        a0 = float(self.sim.upper[i])
        n = int(round(dur / self.sim.dt))
        for k in range(1, n + 1):
            self.sim.upper[i] = a0 + minjerk(k / n) * (rad - a0)
            self.step("lean")
        self.hold(25, "lean")

    def turn_until_seen(self, find, prompts, steps_deg, tag, from_start=False):
        """Look; not found: turn to each of steps_deg (deg, from the heading now, or with from_start from the
        episode's initial heading) and look again."""
        lk, ms = self.look(prompts, f"{tag}0")
        t = find(lk, ms, None)
        y0 = self.sim.yaw0 if from_start else self.yaw()
        for i, dy in enumerate(steps_deg):
            if t is not None:
                break
            self.turn_to(y0 + np.radians(dy), phase=f"search_{tag}")
            lk, ms = self.look(prompts, f"{tag}{i + 1}")
            t = find(lk, ms, None)
        return t

    # ---- supports (table, desk, shelf board): where to stand
    def support_edge(self, lk, p, support_z):
        """The support's top-level points within 1.2 m of the target and their min-area rectangle's axes: of the four
        axis directions with the robot on the near side, the one whose boundary (1st percentile of the points within
        0.3 m of the target sideways) is nearest the target: we face that edge squarely. Returns the heading and a
        world point on that edge."""
        W, ok = lk.world, lk.ok
        sel = ok & (np.abs(W[..., 2] - support_z) < 0.012) & (np.linalg.norm(W[..., :2] - p[:2], axis=2) < 1.2)
        Ps = W[sel][:, :2]
        if len(Ps) < 200:
            return None
        Ps = Ps[:: max(1, len(Ps) // 4000)]
        _, yaw_t, _ = P.min_area_rect(Ps)
        pp, _ = self.pelvis()
        d = Ps - p[:2]
        best = None
        for k in range(4):
            yaw_k = yaw_t + k * np.pi / 2
            f = self.face(yaw_k)[:2]
            if (p[:2] - pp[:2]) @ f <= 0:
                continue  # the robot is not on this side
            lat = np.array([-f[1], f[0]])
            m = (np.abs(d @ lat) < 0.3) & (d @ f < 0)
            if m.sum() < 20:
                continue
            off = -float(np.percentile((d @ f)[m], 1))
            if best is None or off < best[1]:
                best = (wrap(yaw_k), off, f)
        if best is None:
            return None
        return dict(yaw=float(best[0]), pt=p[:2] - best[1] * best[2], off=best[1])

    def stand_on_support(self, key, lk, p, support_z, x_nom, y_nom):
        """Stand heading and the target's place at the stand: facing the support squarely, the pelvis CLEAR off its near
        edge (the edge is kept, in the world, from the first look that saw it), the target at least x_nom ahead."""
        e = self.edges.get(key)
        if e is None and support_z is not None:
            e = self.support_edge(lk, p, support_z)
            if e is not None:
                self.edges[key] = e
        if e is None:
            pp, _ = self.pelvis()
            return dict(yaw=float(np.arctan2(p[1] - pp[1], p[0] - pp[0])), x=x_nom, y=y_nom, edge=None)
        f = self.face(e["yaw"])[:2]
        return dict(yaw=e["yaw"], x=max(x_nom, float((p[:2] - e["pt"]) @ f) + CLEAR), y=y_nom, edge=round(e["off"], 3))

    # ======================= tasks
    def find_box(self, lk, ms, prev):
        m, pts = self.pick(lk, ms, near=None if prev is None else prev["p"])
        if m is None:
            return None
        top = float(np.percentile(pts[:, 2], 95))
        zs, _ = P.support_level(lk.world, lk.ok, m)
        Pt = pts[pts[:, 2] > top - 0.02]
        if len(Pt) < 10:
            return None
        ctr, yaw_box, half = P.min_area_rect(Pt[:, :2])
        h = top - zs
        c = np.array([ctr[0], ctr[1], zs + 0.5 * h])
        t = dict(p=c, yaw_box=yaw_box, half=half + 0.004, h=float(h), support=zs, n=len(pts))  # +1 px of erosion
        t.update(self.stand_on_support("table", lk, c, zs, 0.34, 0.0))
        return t

    def box_faces(self, t):
        """The squeeze axis: the box axis nearest the stand's left (signed to the left), the other one signed forward;
        per side the face centre (world) and the palm rotation (world): fingers along the forward axis, grasp side in."""
        f, l = self.face(t["yaw"]), self.face(t["yaw"] + np.pi / 2)
        u = [self.face(t["yaw_box"]), self.face(t["yaw_box"] + np.pi / 2)]
        ia = int(np.argmax([abs(v @ l) for v in u]))
        a = u[ia] * np.sign(u[ia] @ l)
        b = u[1 - ia] * np.sign(u[1 - ia] @ f)
        half_a = float(t["half"][ia])
        Rw = np.stack([b, a, np.cross(b, a)], axis=1)  # x fingers, y left (right hand's grasp side, left's back), z up
        return {"right": t["p"] - half_a * a, "left": t["p"] + half_a * a}, a, Rw, half_a

    def find_shelf(self, lk, ms, prev, box_bottom):
        m, pts = self.pick(lk, ms, near=None if prev is None else prev["p"], min_px=200)
        if m is None:
            return None
        z = pts[:, 2]
        hist, e = np.histogram(z, bins=np.arange(0.0, 2.0, 0.01))
        levels = [0.5 * (e[i] + e[i + 1]) for i in range(len(hist)) if hist[i] >= max(60, 0.04 * len(z))]
        below = [lv for lv in levels if lv < box_bottom - 0.02]
        if not below:
            return None
        lv = max(below)
        B = pts[np.abs(z - lv) < 0.008]
        if len(B) < 30:
            return None
        top = float(np.median(B[:, 2]))
        ctr, _, half = P.min_area_rect(B[:, :2])
        p = np.array([ctr[0], ctr[1], top])
        t = dict(p=p, board_top=top, half=half.tolist(), levels=levels)
        t.update(self.stand_on_support("shelf", lk, p, top, 0.34, 0.0))
        return t

    def run_ppbox(self, row):
        prom = ["cardboard box", "box"]
        lk, ms = self.look(prom, "reset")
        t = self.find_box(lk, ms, None)
        if t is None:
            return "box not seen"
        self.target = t
        row["box_first"] = dict(p=np.round(t["p"], 3).tolist(), h=round(t["h"], 3), half=np.round(t["half"], 3).tolist(),
                                stand_x=round(t["x"], 3), edge=t["edge"])  # fmt: skip
        lk, seen = self.approach(self.find_box, prom)
        t = self.target
        faces, a, Rw, half_a = self.box_faces(t)
        pp, pR = self.pelvis()
        Rg = pR.T @ Rw
        a_p = pR.T @ a
        grasp = {s: self.to_pelvis(faces[s]) for s in faces}
        sq = {"right": grasp["right"] + 0.02 * a_p, "left": grasp["left"] - 0.02 * a_p}  # 2 cm in: the squeeze
        pre = {"right": grasp["right"] - PRE * a_p, "left": grasp["left"] + PRE * a_p}
        row["perceived"] = dict(box_p=np.round(self.to_pelvis(t["p"]), 3).tolist(), grasp=np.round(grasp["right"], 3).tolist(),
                                pregrasp=np.round(pre["right"], 3).tolist(), half_a=round(half_a, 3), h=round(t["h"], 3),
                                stand_x=round(t["x"], 3), edge=t["edge"], seen_at_stand=seen, stand_err=self.stand_err)  # fmt: skip
        self.arm_lines({"left": (pre["left"], Rg, G_BOX["left"])}, "left_pregrasp", speed=0.25)
        row["reach"] = self.reach(pre["right"], Rg, G_BOX["right"], self.vox_for_plan(lk))
        self.arm_lines({s: (sq[s], Rg, G_BOX[s]) for s in ("left", "right")}, "squeeze")
        self.sim.hands[:] = 1.0
        self.hold(25, "close")
        self.arm_shift(("left", "right"), pR.T @ np.array([0, 0, 0.10]), "lift", G_BOX)
        self.lean(0.0)  # upright: the box comes up and back with the torso
        q = self.ref_q()
        box_bottom_w = float((pR @ (0.5 * (self.chain["left"].point(q, G_BOX["left"])[0]
                                           + self.chain["right"].point(q, G_BOX["right"])[0])) + pp)[2] - 0.5 * t["h"])  # fmt: skip
        sprom = ["blue metal shelf"]  # "shelf" also matches the table's own lower shelf (dev look, 2026-11-24)

        def find_s(lk_, ms_, prev):
            return self.find_shelf(lk_, ms_, prev, box_bottom_w)

        self.lean_rad = 0.0
        s = self.turn_until_seen(find_s, sprom, (45, 90, 135), "shelf", from_start=True)
        if s is None:
            return "shelf not seen"
        self.target = s
        row["shelf_first"] = dict(p=np.round(s["p"], 3).tolist(), levels=np.round(s["levels"], 3).tolist(),
                                  stand_x=round(s["x"], 3), edge=s["edge"])  # fmt: skip
        self.approach(find_s, sprom, phase="carry")
        s = self.target
        pp, pR = self.pelvis()
        q = self.ref_q()
        box_now = 0.5 * (
            self.chain["left"].point(q, G_BOX["left"])[0] + self.chain["right"].point(q, G_BOX["right"])[0]
        )
        tgt = self.to_pelvis(s["p"])
        tgt_box = tgt + pR.T @ np.array([0, 0, 0.5 * t["h"] + 0.03])  # the box centre: its bottom 3 cm over the board
        row["perceived"]["board_p"] = np.round(tgt, 3).tolist()
        self.arm_shift(("left", "right"), tgt_box - box_now, "place", G_BOX)
        self.sim.hands[:] = 0.0
        self.hold(50, "release")
        return "sequence done"

    def find_hammer(self, lk, ms, prev):
        m, pts = self.pick(lk, ms, near=None if prev is None else prev["p"])
        if m is None:
            return None
        c = np.median(pts, axis=0)
        d = pts[:, :2] - c[:2]
        _, _, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
        near = np.linalg.norm(d, axis=1) < 0.03
        ztop = float(np.percentile(pts[near, 2], 90)) if near.sum() >= 5 else float(np.percentile(pts[:, 2], 90))
        g = np.array([c[0], c[1], ztop - 0.012])
        zs, _ = P.support_level(lk.world, lk.ok, m)
        t = dict(p=g, axis=np.r_[vt[0], 0.0], support=zs, n=len(pts))
        t.update(self.stand_on_support("desk", lk, g, zs, 0.36, -0.20))
        return t

    def topdown_R(self, axis_w, fwd_w, left_w):
        """Right palm, world frame: grasp side (+y) down, the handle along z (the fist closes around it), fingers (+x)
        level and across the handle, toward forward-and-right (outward: reachable for the right arm, offline IK check)."""
        x = np.cross([0, 0, 1.0], axis_w)
        x = x / np.linalg.norm(x)
        if x @ (fwd_w - left_w) < 0:
            x = -x
        y = np.array([0, 0, -1.0])
        return np.stack([x, y, np.cross(x, y)], axis=1)

    def run_doubledesk(self, row):
        prom = ["hammer"]
        t = self.turn_until_seen(self.find_hammer, prom, (-45, -90), "hammer")
        if t is None:
            return "hammer not seen"
        self.target = t
        row["hammer_first"] = dict(p=np.round(t["p"], 3).tolist(), stand_x=round(t["x"], 3), edge=t["edge"])
        lk, seen = self.approach(self.find_hammer, prom)
        t = self.target
        pp, pR = self.pelvis()
        y = self.yaw()
        Rg = pR.T @ self.topdown_R(t["axis"], self.face(y), self.face(y + np.pi / 2))
        g = self.to_pelvis(t["p"])
        up = pR.T @ np.array([0, 0, 1.0])
        pre = g + PRE * up
        row["perceived"] = dict(grasp=np.round(g, 3).tolist(), pregrasp=np.round(pre, 3).tolist(),
                                stand_x=round(t["x"], 3), edge=t["edge"], seen_at_stand=seen, stand_err=self.stand_err)  # fmt: skip
        row["reach"] = self.reach(pre, Rg, G_FIST, self.vox_for_plan(lk))
        self.arm_lines({"right": (g, Rg, G_FIST)}, "descend")
        self.sim.hands[1] = 1.0
        self.hold(30, "close")
        self.arm_shift(("right",), 0.15 * up, "lift", {"right": G_FIST})
        self.lean(0.0)
        self.lean_rad = 0.0
        bprom = ["bin", "basket"]

        def find_basket(lk_, ms_, prev):
            m, pts = self.pick(lk_, ms_, near=None if prev is None else prev["p"], min_px=20)
            if m is None:
                return None
            top = float(np.percentile(pts[:, 2], 95))
            rim = pts[pts[:, 2] > top - 0.03]
            ctr = P.min_area_rect(rim[:, :2])[0] if len(rim) >= 10 else np.median(pts[:, :2], axis=0)
            p = np.array([ctr[0], ctr[1], top])
            zs, _ = P.support_level(lk_.world, lk_.ok, m)
            bt = dict(p=p)
            bt.update(self.stand_on_support("basket_table", lk_, p, zs, 0.40, -0.20))
            return bt

        b = self.turn_until_seen(find_basket, bprom, (45, 90, 135), "basket", from_start=True)
        if b is None:
            return "basket not seen"
        self.target = b
        row["basket_first"] = dict(p=np.round(b["p"], 3).tolist(), stand_x=round(b["x"], 3), edge=b["edge"])
        self.approach(find_basket, bprom, phase="carry")
        b = self.target
        over = self.to_pelvis(b["p"] + np.array([0, 0, 0.15]))
        row["perceived"]["basket_over"] = np.round(over, 3).tolist()
        _, R0 = self.chain["right"].point(self.ref_q(), G_FIST)
        self.arm_lines({"right": (over, R0, G_FIST)}, "over_basket")
        self.sim.hands[1] = 0.0
        self.hold(75, "release")
        return "sequence done"

    def find_handle(self, lk, ms, prev):
        hm = [(s, m, p) for s, m, p in ms if p == "door handle"]
        dm = [(s, m, p) for s, m, p in ms if p == "door"]
        m, pts = self.pick(lk, hm, near=None if prev is None else prev["p"], min_px=10, erode_px=0)
        if m is None:
            return None
        h = np.median(pts, axis=0)
        n = None
        md, dpts = self.pick(lk, dm, near=h, min_px=200)
        if md is not None:
            dpts = dpts[np.linalg.norm(dpts[:, :2] - h[:2], axis=1) < 0.6]
            if len(dpts) > 100:
                _, n = P.plane_fit(dpts)
                n = np.r_[n[:2], 0.0]
                n = n / max(np.linalg.norm(n), 1e-9)
        pp, _ = self.pelvis()
        if n is None:  # no door plane: face the handle
            n = np.r_[pp[:2] - h[:2], 0.0]
            n = n / np.linalg.norm(n)
        if n[:2] @ (pp[:2] - h[:2]) < 0:
            n = -n  # toward the robot
        return dict(
            p=h, n=n, yaw=float(np.arctan2(-n[1], -n[0])), x=0.40, y=-0.20, edge=None, door_plane=md is not None
        )

    def run_opendoor(self, row):
        prom = ["door handle", "door"]
        t = self.turn_until_seen(self.find_handle, prom, (45, -45), "handle")
        if t is None:
            return "handle not seen"
        self.target = t
        row["handle_first"] = np.round(t["p"], 3).tolist()
        lk, seen = self.approach(self.find_handle, prom)
        t = self.target
        pp, pR = self.pelvis()
        x = -t["n"]
        y = np.array([0, 0, -1.0])
        Rg = pR.T @ np.stack(
            [x, y, np.cross(x, y)], axis=1
        )  # a hook over the lever: fingers into the door, grasp side down
        g = self.to_pelvis(t["p"])
        n_p = pR.T @ t["n"]
        pre = g + PRE * n_p
        row["perceived"] = dict(grasp=np.round(g, 3).tolist(), pregrasp=np.round(pre, 3).tolist(),
                                door_plane=t["door_plane"], seen_at_stand=seen, stand_err=self.stand_err)  # fmt: skip
        row["reach"] = self.reach(pre, Rg, G_FIST, self.vox_for_plan(lk))
        self.arm_lines({"right": (g, Rg, G_FIST)}, "in")
        self.sim.hands[1] = 1.0
        self.hold(30, "close")
        self.arm_shift(("right",), pR.T @ np.array([0, 0, -0.06]), "press", {"right": G_FIST})
        self.arm_shift(("right",), -0.15 * n_p, "push", {"right": G_FIST})
        self.sim.hands[1] = 0.0
        self.hold(10, "release")
        self.walk_forward(1.0, phase="walk_through")
        return "sequence done"

    # ======================= episodes
    def episode(self, seed, repeat):
        a, s = self.a, self.sim
        H.check_seed(seed)
        es = H.episode_seed(a.task, seed, repeat)
        self.ep_name = f"{a.ha_task}_{a.method}_s{seed}_r{repeat}"
        self.film = {k: [] for k in ("ref", "q", "root", "wrist", "palm", "phase")}
        self.phase_id, self.looks, self.phase, self.target, self.edges = {}, [], "reset", None, {}
        self.lean_rad = LEAN[a.ha_task]
        self.stand_err = None
        self.diag, self.diag_err = [], None
        t0 = time.time()
        s.reset(es)
        row = dict(task=a.ha_task, seed=seed, repeat=repeat, episode_seed=es, method=a.method)
        end = None
        try:
            self.hold(50, "stand")
            fn = dict(ppbox=self.run_ppbox, doubledesk=self.run_doubledesk, opendoor=self.run_opendoor)[a.ha_task]
            row["sequence"] = fn(row)
            n_more = (a.max_steps - s.k) if a.full_budget else a.hold
            self.hold(n_more, "hold_end")
            end = "held"
        except End as e:
            end = str(e)
        except Exception as e:  # a harness error: record it, keep the process (Kit) alive for the next episode
            import traceback

            traceback.print_exc()
            end = f"error {type(e).__name__}: {e}"
        row.update(end=end, success=s.success_at is not None, success_step=s.success_at,
                   success_time_s=None if s.success_at is None else round(s.success_at * s.dt, 2),
                   fell=s.fell_at is not None, fell_step=s.fell_at, steps=s.k, phase_at_end=self.phase,
                   max_steps=a.max_steps, looks=self.looks, wall_s=round(time.time() - t0, 1),
                   diag=self.diag_summary())  # fmt: skip
        r = row.get("reach", {})
        row["plan"] = None if a.method == "A" else ("ok" if r.get("plan_ok") else ("fallback" if r else "no reach"))
        np.savez_compressed(f"{a.out}/film/{self.ep_name}.npz", phase_names=np.array(list(self.phase_id)),
                            **{k: np.asarray(v) for k, v in self.film.items()})  # fmt: skip
        return row


def to_json(o):
    if isinstance(o, dict):
        return {k: to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> int:
    p = H.build_parser()
    p.add_argument("--method", choices=("A", "J"), required=True)
    p.add_argument("--seeds", default="100,101")
    p.add_argument("--repeats", default="0-4")
    p.add_argument("--out", required=True)
    p.add_argument("--plan_sock", default="/tmp/ha_plan3.sock")
    p.add_argument("--hold", type=int, default=100, help="steps held after the sequence (unless --full_budget)")
    p.add_argument("--full_budget", action="store_true", help="hold to the official step budget after the sequence")
    p.add_argument("--save_looks", type=int, default=1)
    a = p.parse_args()
    H.finish_args(a, a.ha_task)
    seeds = [int(x) for x in a.seeds.split(",")]
    lo, _, hi = a.repeats.partition("-")
    repeats = list(range(int(lo), int(hi or lo) + 1))
    for sd in seeds:
        H.check_seed(sd)
    os.makedirs(f"{a.out}/film", exist_ok=True)
    os.makedirs(f"{a.out}/looks", exist_ok=True)
    b = Bench(a)
    b.look_dir = f"{a.out}/looks"
    path = f"{a.out}/{a.ha_task}_{a.method}.jsonl"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["seed"], r["repeat"]))
    print(f"[bench] startup {b.sim.startup_s:.1f} s; {len(done)} done", flush=True)
    for sd in seeds:
        for rp in repeats:
            if (sd, rp) in done:
                continue
            row = b.episode(sd, rp)
            with open(path, "a") as f:
                f.write(json.dumps(to_json(row)) + "\n")
            print(f"[bench] {a.ha_task} {a.method} s{sd} r{rp}: end={row['end']} success={row['success']} "
                  f"seq={row.get('sequence')} hand_err={row.get('reach', {}).get('hand_err_cm')} plan={row['plan']} "
                  f"steps={row['steps']} wall={row['wall_s']}", flush=True)  # fmt: skip
    print("[bench] DONE", flush=True)
    b.sim.close()
    return 0


if __name__ == "__main__":
    H.main_guard(main)
