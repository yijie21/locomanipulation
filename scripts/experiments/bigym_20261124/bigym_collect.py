"""bigym_selfmodel_20261124: the self model's training data on BiGym 2.0's own controller, the jrefwm way
(simple_eval_20261112/jref_collect.py): cuRobo plans of the right arm from BiGym stances, run by the controller with the
rest of the body held, the scene not colliding from the command on. The registration is
docs/handoff/notebook-20261120/bigym_selfmodel_20261124.md.

    export BIGYM_EGL_GPU=0; source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh; cd <repo>
    python scripts/experiments/bigym_20261124/bigym_collect.py --task reach_target_multi_modal --sids 0:20 \
        --out <dir>/rows/x.jsonl --film <dir>/films --plan-socket <socket>        # collect.sh runs the real thing

A stance (id sid, BiGym seed SEED_BASE + sid: neither the eval block 620000-629999 nor the dev seeds 720000-720099):
  1. reset(seed); the task's goal from the SIMULATOR (training data, not an evaluation): the reach target's centre; for
     the drawer the top drawer's handle bar (the box geom of the drawer's body), its pregrasp the bench's rule's: the
     bar's centre GRASP_OUT out along the drawer's opening direction (the grasp; out_cm 2.6-2.75 on the dev looks) and
     PRE_M more.
  2. bigym_bench.py --stance's walk with the stance point moved by U(-JIT_XY, JIT_XY) ahead and sideways (in the pelvis's
     frame at the reset), then for the drawer the squat to SQUAT_H + U(-JIT_H, JIT_H). The gripper is open throughout.
  3. per sample k (K of them): hold PRE_S (phase pre<k>); the state at the command is the last one recorded. From the
     first command on, every geom that is not the robot's and not a plane (the floor) stops colliding (restored before
     the next reset). A draw in the heading frame at the command (origin the pelvis's x, y; z world; x its heading):
       goal   half near the task's goal (the target's centre / the pregrasp; N(0, NEAR_SD) a coordinate), half uniform
              in FAR, the box ahead of the right shoulder (x, y; z relative to the pelvis's height at the command); the
              wrist the task's own attitude turned by up to YAW_DEG yaw and PITCH_DEG pitch and roll
              (rot_z . rot_y . rot_x . R_task, jref's order). The task's attitude: the reach's is the bench's aim
              (DLS IK from the measured arm to the centre keeping the start's attitude at GOAL_ROT_W); the drawer's is
              the bench's grasp attitude (approach along the drawer's normal, the fingers across the bar, of the two the
              one off the robot's body, then the smaller joint motion: bigym_bench.py's rule)
       boxes  0, 1 or 2 (30 / 45 / 25 %) between the hand and the goal (jref_collect.draw's rule, no slab: BiGym's
              tasks have no table to rise over), 4-20 cm a side, turned up to 45 deg about z, none within CLEAR of a
              start sphere (every fitted sphere of the right arm and gripper: what the planner checks) or GOAL_CLEAR
              of the goal; for the planner only
     The planner (curobo/plan_server.py: the right arm, the waist locked, the world in the virtual pelvis's frame)
     gets up to ATTEMPTS draws; none plans: the sample is skipped. The plan is run as bigym_bench.py's J: min-jerk
     joint targets by Motor.reach_joints' timing (0.15 m/s of the hand point, the start gap faded) to the right arm,
     everything else the hold action (phase jref<k>); then the hand settles (arm joint speeds under SETTLE_V for 0.1
     s, at most SETTLE_MAX_S) and holds POST_S (phase post<k>). Between samples the arm goes back along the plan
     reversed (same timing, phase back<k>) and holds 1 s (phase backhold<k>).
  The task's own episode end is switched off on the instance (success hold and time limit: nothing ends a stance but a
  fall or a physics error); the env is made without cameras (no observation is read; the controller is the same).

Output: a row per stance (jsonl: its samples as stages, jref_collect's fields) and a film per stance (npz, bigym_bench's
Run film: every control step's arm targets sent, the whole robot's readings, phases).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bigym_bench as B  # noqa: E402
from spheres24 import Spheres24  # noqa: E402

SEED_BASE = 800000
TASKS = ("reach_target_multi_modal", "drawer_top_open")
ATTEMPTS, CLEAR, GOAL_CLEAR = 20, 0.035, 0.08
JIT_XY, JIT_H = 0.06, 0.03
PRE_S, POST_S, SETTLE_MAX_S, SETTLE_V, BACK_HOLD_S = 0.3, 1.0, 1.5, 0.05, 1.0
NEAR_SD, YAW_DEG, PITCH_DEG = 0.05, 35.0, 25.0
# heading frame at the command, z above the pelvis. The right shoulder sits at x 0.01, y -0.10, z +0.29 there (both
# tasks); the box is where a DLS IK from the measured arm or 3 random starts reaches 5 mm / 3 deg with draw()'s wrist
# attitudes and no arm-body penetration on 70 % (reach) and 57 % (drawer) of uniform points (6 stances a task, 400 points
# each, over x 0..0.75, y -0.65..0.30, z -0.45..0.55: 12-13 % overall, none under z -0.15 or past x 0.5); y stops at the
# pelvis's midline: the plans do not see the torso
FAR = dict(x=(0.15, 0.45), y=(-0.40, 0.00), z=(0.00, 0.35))
GRASP_OUT = 0.026  # m, the bar's centre to the bench rule's grasp point along the normal
BOX_P = (0.30, 0.45, 0.25)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def box_clear(P: np.ndarray, c: np.ndarray, R: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Signed distance of the points P to the box (centre c, rotation R, full sizes dims): jref_collect.box_clear."""
    d = np.abs((P - c) @ R) - 0.5 * dims
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


def draw(rng, s_h, task_h, R_task_h, X_h, radius, pz: float) -> dict:
    """A goal (the pinch point and the wrist's attitude) and boxes in the heading frame at the command."""
    near = rng.random() < 0.5
    if near:
        g = task_h + rng.normal(0.0, NEAR_SD, 3)
    else:
        g = np.array([rng.uniform(*FAR["x"]), rng.uniform(*FAR["y"]), pz + rng.uniform(*FAR["z"])])
    yaw, pit, rol = np.radians([rng.uniform(-YAW_DEG, YAW_DEG), rng.uniform(-PITCH_DEG, PITCH_DEG),
                                rng.uniform(-PITCH_DEG, PITCH_DEG)])  # fmt: skip
    Rg = rot_z(yaw) @ rot_y(pit) @ rot_x(rol) @ R_task_h
    boxes = []
    for _ in range(int(rng.choice(3, p=BOX_P))):
        for _ in range(20):
            c = s_h + rng.uniform(0.25, 0.75) * (g - s_h) + rng.normal(0.0, 0.04, 3)
            dims, a = rng.uniform(0.04, 0.20, 3), math.radians(rng.uniform(-45, 45))
            Rb = rot_z(a)
            if (box_clear(X_h, c, Rb, dims) - radius).min() >= CLEAR and box_clear(g[None], c, Rb, dims)[
                0
            ] >= GOAL_CLEAR:
                boxes.append((c, a, dims))
                break
    return dict(goal=g, R=Rg, boxes=boxes, near=bool(near))


class Collector:
    def __init__(self, a):
        from bigym.loco import make

        self.a, self.task = a, a.task
        self.env = make(a.task, camera_keys=())  # no observation is read
        self.inner = self.env.inner_env
        self.m, self.d = self.inner.model, self.inner.data
        self.ns = self.inner.robot.namespace
        self.kin = B.Kin(self.env)
        self.right = self.kin.right  # the right arm's and gripper's bodies (right_shoulder_pitch_link's subtree)
        self.sph = Spheres24()
        self.keep = self.m.geom_contype.copy(), self.m.geom_conaffinity.copy()
        self.conn = None

    # -- the scene's truth (where to stand and what to aim near: training data only) --------------------------------
    def task_goal(self) -> dict:
        m, d = self.m, self.d
        if self.task.startswith("reach"):
            return dict(p=np.asarray(self.inner.targets[0].get_position(), float))
        j = m.joint(self.inner.cabinet_drawers.joints[-1].name).id  # the top drawer (get_state()[-1])
        b = int(m.jnt_bodyid[j])
        Rb = d.xmat[b].reshape(3, 3)
        n = Rb @ m.jnt_axis[j]  # the opening direction: out of the cabinet, toward the robot
        n = np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2])

        def under(c: int) -> bool:  # c is b or a body below it (the handle sits in a child body)
            while c > 0 and c != b:
                c = m.body_parentid[c]
            return c == b

        bars = [g for g in range(m.ngeom) if under(int(m.geom_bodyid[g])) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                and self.keep[0][g] and m.geom_size[g].max() < 0.1]  # fmt: skip
        g = max(bars, key=lambda g: m.geom_size[g].max() / m.geom_size[g].min())  # the bar: the slenderest box
        Rg = d.geom_xmat[g].reshape(3, 3)
        bar = Rg[:, int(np.argmax(m.geom_size[g]))]
        grasp = d.geom_xpos[g] + GRASP_OUT * n
        return dict(p=grasp + B.PRE_M * n, grasp=grasp, n=n, bar=bar, size=m.geom_size[g].tolist())

    def free_scene(self) -> int:
        m, k = self.m, 0
        for g in range(m.ngeom):
            if self.kin.robot_bodies[m.geom_bodyid[g]] or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
                k += 1
        return k

    def plan(self, req: dict) -> dict:
        for attempt in (0, 1):
            try:
                if self.conn is None:
                    self.conn = Client(self.a.plan_socket, "AF_UNIX", authkey=B.PLAN_AUTHKEY)
                self.conn.send(req)
                return self.conn.recv()
            except (EOFError, OSError):
                self.conn = None
                if attempt:
                    raise
        raise RuntimeError("unreachable")

    # -- a stance --------------------------------------------------------------------------------------------------
    def stance(self, sid: int) -> dict:
        a, env, kin, m, d = self.a, self.env, self.kin, self.m, self.d
        seed = SEED_BASE + sid
        t_wall = time.time()
        m.geom_contype[:], m.geom_conaffinity[:] = self.keep
        env.reset(seed=seed)
        self.inner._success_hold_steps = 10**9  # no episode end on success (the instance only)
        env.bigym._time_limit = 10**9  # bigym.loco's env under the time-step wrapper (a wrapper attribute is not read)
        run = B.Run(env, kin, self.task)
        rng = np.random.default_rng([a.collect_seed, sid])
        jx, jy = rng.uniform(-JIT_XY, JIT_XY, 2)
        jh = float(rng.uniform(-JIT_H, JIT_H)) if self.task == "drawer_top_open" else 0.0
        row = dict(task=self.task, sid=sid, seed=seed, jitter=[round(float(jx), 4), round(float(jy), 4), round(jh, 4)])
        goal = self.task_goal()
        p0, R0 = run.pelvis()
        g0 = R0.T @ (goal["p"] - p0)
        x_goal = R0.T @ ((goal["grasp"] if "grasp" in goal else goal["p"]) - p0)  # the bench stands by the grasp
        dist = float(x_goal[0]) - (B.STAND_X[self.task] + jx)
        side = float(x_goal[1]) - (B.STAND_Y[self.task] + jy)
        moved = moved_y = 0.0
        if dist > B.WALK_MIN:
            moved = run.walk("walk", dist, B.WALK_VX, dist / B.WALK_VX + 2.0)
            run.wait("walk_settle", B.WALK_SETTLE_S)
        if abs(side) > B.WALK_MIN:
            moved_y = math.copysign(
                run.walk("side", abs(side), math.copysign(B.WALK_VX, side), abs(side) / B.WALK_VX + 2.0, 1), side
            )
            run.wait("side_settle", B.WALK_SETTLE_S)
        h_cmd = -1.0
        if self.task == "drawer_top_open":
            h_cmd = B.SQUAT_H + jh
            run.hold[2] = h_cmd
            run.wait("squat", B.SQUAT_S)
        row.update(goal0_pelvis=B.r3(g0), walk=dict(want_cm=round(100 * dist, 1), moved_cm=round(100 * moved, 1),
                   side_want_cm=round(100 * side, 1), side_moved_cm=round(100 * moved_y, 1)), h_cmd=round(h_cmd, 4))  # fmt: skip
        stages, n_free = [], 0
        for k in range(a.k):
            if run.done or env.episode_fell():
                break
            run.wait(f"pre{k}", PRE_S)
            if run.done:
                break
            if k == 0:
                n_free = self.free_scene()
            s, Q = self.sample(run, rng, k, goal)
            stages.append(s)
            if Q is None or run.done or env.episode_fell() or k == a.k - 1:
                continue
            kin.load(d)
            back, _ = B.joint_refs(kin, Q[::-1], run.ref.copy())
            run.path(f"back{k}", back)
            run.wait(f"backhold{k}", BACK_HOLD_S)
        row.update(stages=stages, n_free=n_free, fell=bool(env.episode_fell()), done=bool(run.done),
                   termination=env.episode_termination() if run.done else None, steps=run.k,
                   wall_s=round(time.time() - t_wall, 2))  # fmt: skip
        if a.film:
            meta = dict(task=self.task, sid=sid, seed=seed, h_cmd=h_cmd)
            np.savez_compressed(f"{a.film}/{self.task}_{seed}.npz", meta=json.dumps(meta), arm=np.array(B.ARM),
                                jnt=np.array(kin.jnt), phases=np.array(run.phases), dt=B.DT,
                                **{k_: np.array(v) for k_, v in run.film.items()})  # fmt: skip
        return row

    def frame(self, run, goal: dict, s: dict) -> dict:
        """The command's frames (the heading frame, the planner's virtual pelvis) and the task's goal and attitude in
        them (bigym_bench.py's rules); s gets the twin choice."""
        kin, d, m = self.kin, self.d, self.m
        kin.load(d)
        p1, Rp1 = run.pelvis()
        Rh = rot_z(math.atan2(Rp1[1, 0], Rp1[0, 0]))
        org = np.array([p1[0], p1[1], 0.0])
        q_meas = d.qpos[kin.arm_q].copy()
        tw, Rw = kin.torso_pose()
        tw0, Rw0 = kin.torso_pose(np.zeros(3))
        Rv, tv = Rw @ Rw0.T, tw - (Rw @ Rw0.T) @ tw0  # the virtual pelvis in the pelvis frame
        gp_task = Rp1.T @ (goal["p"] - p1)
        if self.task.startswith("reach"):
            _, R_start = kin.site_pose(q_meas)
            q_g = kin.ik(gp_task, R_start, q_meas, B.GOAL_ROT_W, iters=300, tol_p=1e-4)[0]
            R_task, s["twin"] = kin.site_pose(q_g)[1], None
        else:
            n_p, b_p = Rp1.T @ goal["n"], Rp1.T @ goal["bar"]
            x = -n_p
            y = np.cross(x, b_p)
            y /= np.linalg.norm(y)
            R1 = np.column_stack([x, y, np.cross(x, y)])
            score = []
            for R_ in (R1, R1 @ np.diag([1.0, -1.0, -1.0])):
                q_, e_, _ = kin.ik(gp_task, R_, q_meas, 1.0, iters=300, tol_p=1e-4)
                qg = kin.ik(gp_task - B.IN_M * n_p, R_, q_, 1.0, iters=200)[0]
                pen = max(kin.self_contact(q_), kin.self_contact(qg))
                score.append((e_ >= 0.01, round(pen) if pen > 1.0 else 0, float(np.abs(q_ - q_meas).sum())))
            R_task = R1 if score[0] <= score[1] else R1 @ np.diag([1.0, -1.0, -1.0])
            s["twin"] = int(score[1] < score[0])
        kin.site_pose(q_meas)  # the scratch arm back at its reading
        mujoco.mj_kinematics(m, d)

        def to_h(X):  # world -> heading frame (origin the pelvis's x, y; z world)
            return (np.asarray(X, float) - org) @ Rh

        return dict(p1=p1, Rp1=Rp1, Rh=Rh, org=org, q_meas=q_meas, Rv=Rv, tv=tv, R_task=R_task, to_h=to_h,
                    X_h=to_h(self.sph.world_full(m, d, self.ns)), s_h=to_h(d.site_xpos[kin.site]), task_h=to_h(goal["p"]),
                    R_task_h=Rh.T @ Rp1 @ R_task)  # fmt: skip

    def body_contact(self, seen: set) -> float:
        """The deepest penetration (mm) now between a right-arm or gripper body and another robot body (0 if none);
        the pairs touching go into seen (arm body, other body, the robot's namespace dropped)."""
        m, d, kin, worst = self.m, self.d, self.kin, 0.0
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if kin.robot_bodies[b1] and kin.robot_bodies[b2] and ((b1 in self.right) != (b2 in self.right)):
                worst = max(worst, -1000.0 * float(c.dist))
                a_, o_ = (b1, b2) if b1 in self.right else (b2, b1)
                seen.add((m.body(a_).name[len(self.ns) :], m.body(o_).name[len(self.ns) :]))
        return worst

    def sample(self, run, rng, k: int, goal: dict):
        """One sample from the state now (the command). -> (stage dict, the plan (T, 7) in ARM order, or None)."""
        a, kin, d = self.a, self.kin, self.d
        s = dict(stage="jref", k=k, t_cmd=round(run.k * B.DT, 3), i_cmd=len(run.film["t"]) - 1)
        F = self.frame(run, goal, s)
        p1, Rp1, Rh, org, q_meas, Rv, tv = (F[k_] for k_ in ("p1", "Rp1", "Rh", "org", "q_meas", "Rv", "tv"))
        X_h, s_h, task_h, R_task_h = F["X_h"], F["s_h"], F["task_h"], F["R_task_h"]
        ctrl0 = run.ref.copy()
        q = {n: float(v) for n, v in zip(B.ARM, q_meas)}
        tries, rep, dr = [], None, None
        for _ in range(ATTEMPTS):
            dr = draw(rng, s_h, task_h, R_task_h, X_h, self.sph.full_radius, float(p1[2]))
            gp = Rp1.T @ (org + Rh @ dr["goal"] - p1)
            gR = Rp1.T @ Rh @ dr["R"]
            boxes = []
            for c, ang, dims in dr["boxes"]:
                c_p = Rp1.T @ (org + Rh @ c - p1)
                boxes.append(((c_p - tv) @ Rv, B.quat_of(Rv.T @ Rp1.T @ Rh @ rot_z(ang)), dims))
            t_ = time.time()
            rep = self.plan(dict(q=q, vox=np.zeros((0, 3), np.float32), vox_size=B.VOX, goal_p=(gp - tv) @ Rv,
                                 goal_q=B.quat_of(Rv.T @ gR), attempts=4, timeout=a.plan_timeout, free=0.01, boxes=boxes))  # fmt: skip
            tries.append(dict(ok=bool(rep.get("ok")), status=str(rep.get("status")), t=round(time.time() - t_, 2), n_box=len(boxes),
                              near=dr["near"]))  # fmt: skip
            if rep.get("ok"):
                break
        s.update(tries=tries, task_h=B.r3(task_h), R_task_h=B.r3(R_task_h, 5))
        if not rep or not rep.get("ok") or rep.get("Q") is None:
            s["skip"] = "no plan"
            return s, None
        names = list(rep["names"])
        Q = np.asarray(rep["Q"], float)[:, [names.index(n) for n in B.ARM]]
        if rep.get("fk_p") is not None:  # the planner's start hand against BiGym's (virtual pelvis frame)
            s["fk_mm"] = round(
                1000 * float(np.linalg.norm(np.asarray(rep["fk_p"], float) - (kin.site_pose(q_meas)[0] - tv) @ Rv)), 3
            )
        refs, info = B.joint_refs(kin, Q, ctrl0)
        s.update(goal_h=B.r3(dr["goal"]), goal_R_h=B.r3(dr["R"], 5), near=dr["near"],
                 boxes=[(B.r3(c), round(ang, 4), B.r3(dims)) for c, ang, dims in dr["boxes"]],
                 plan_T=len(Q), dur_s=info["dur_s"], path_cm=info["path_cm"], gap_deg=info["gap_deg"],
                 end_err_mm=rep.get("end_err_mm"), plan_clear_cm=rep.get("plan_clear_cm"))  # fmt: skip
        s = {k_: (v.item() if isinstance(v, np.generic) else v) for k_, v in s.items()}
        pen, seen = [], set()
        for r_ in refs:
            if run.step(f"jref{k}", r_):
                break
            pen.append(self.body_contact(seen))
        still = n_settle = 0
        for n_settle in range(1, int(round(SETTLE_MAX_S / B.DT)) + 1):
            if run.step(f"post{k}"):
                break
            pen.append(self.body_contact(seen))
            still = still + 1 if np.abs(d.qvel[kin.arm_v]).max() < SETTLE_V else 0
            if still >= 5:
                break
        for _ in range(int(round(POST_S / B.DT))):
            if run.step(f"post{k}"):
                break
            pen.append(self.body_contact(seen))
        s["body_contact_mm"] = round(max(pen, default=0.0), 1)  # the arm against the robot's own body, worst
        s["body_contact_steps"] = int(sum(p_ > 0 for p_ in pen))
        s["body_contact_pairs"] = sorted(seen)
        s.update(settle_s=round(n_settle * B.DT, 2), settled=still >= 5, i_end=len(run.film["t"]) - 1, fell=bool(self.env.episode_fell()),
                 done=bool(run.done))  # fmt: skip
        return s, Q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--sids", required=True, help="a:b (stance ids; seed = SEED_BASE + sid)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--film", default=None)
    ap.add_argument("--plan-socket", required=True)
    ap.add_argument("--plan-timeout", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=2, help="samples a stance")
    ap.add_argument("--collect-seed", type=int, default=0)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.sids.split(":"))
    for sid in (lo, hi - 1):
        seed = SEED_BASE + sid
        if B.EVAL_LO <= seed < B.EVAL_HI or 720000 <= seed < 720100:
            raise SystemExit(f"seed {seed} is an eval or dev seed")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.film:
        os.makedirs(a.film, exist_ok=True)
    c = Collector(a)
    for sid in range(lo, hi):
        row = c.stance(sid)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        st = row["stages"]
        print(json.dumps(dict(task=a.task, sid=sid, fell=row["fell"], samples=len(st), planned=sum("skip" not in s for s in st),
                              tries=[len(s["tries"]) for s in st], wall_s=row["wall_s"])), flush=True)  # fmt: skip
    print(json.dumps(dict(done=True, task=a.task, sids=a.sids)), flush=True)


if __name__ == "__main__":
    main()
