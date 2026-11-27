"""Walking, turning and the leg odometry for selfmodel_hardgrasp_20261125 (LocoManiLab host, ScaleBFM), 2026-11-25.

ScaleBFM tracks a whole-body reference; it has no velocity-command mode. A walk is therefore a synthetic gait
reference: a footstep plan (turn toward the goal in place, walk straight, turn to the goal heading in place; or, for a
short move, one omnidirectional shuffle), the pelvis carried through the step knots (linear, Gaussian-smoothed, a small
sway toward the stance foot, lowered to WALK_Z while walking), each swing foot on a min-jerk arc lifted SWING_H, and the
12 leg joints solved per frame by damped least squares on robot_fk (ankle_roll_link position + flat foot at its yaw).
The upper body blends to DEFAULT_Q; the walk ends in the default stance at stand_z, held HOLD_END_S, so lml_tasks'
reaches (planned from the measured state) follow it unchanged.

Frames are in reference coordinates, as lml_tasks' frames: the goal is given relative to the pelvis the caller believes
in (`pose`, default the measured one in x) and carried onto the current reference frame's root.

LegOdo: pelvis xy / yaw without simulator positions: joint encoders (FK of the 29 joints on robot_fk), the IMU's
orientation (the pelvis rotation: roll / pitch / yaw, as lml_tasks' feedback method), the most loaded foot (normal
contact force on its sole: a foot force sensor's reading) as the stance foot, its sole centre anchored in the world
while it stays the stance foot.

    python hg_walk.py test --out DIR --task FreeSpace --case walk2.0 --reps 5 --worker K
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LML = HERE.parent / "locomanilab_20261117"
sys.path.insert(0, str(LML))
import lml_selfmodel as L  # noqa: E402

_E = lambda k, v: float(os.environ.get(f"HG_{k}", v))  # noqa: E731 (dev overrides; the frozen run sets none)
STEP_LEN = _E("STEP_LEN", 0.20)  # m per step, forward
STEP_SIDE = _E("STEP_SIDE", 0.10)  # m per step, sideways (short moves)
STEP_YAW = math.radians(_E("STEP_YAW", 15.0))  # per step
STEP_T = _E("STEP_T", 0.55)  # s per step
DS = 0.25  # double-support fraction of a step (split before / after the swing)
SWING_H = _E("SWING_H", 0.06)  # m
SWAY = _E("SWAY", 0.025)  # m, pelvis toward the stance foot at mid-swing
WALK_Z = 0.77  # pelvis height while walking (stand_z 0.8: the knees keep slack)
BLEND_S = 0.5  # in / out blends (pelvis height, upper body)
HOLD_END_S = 1.5  # default stance held at the end
SMOOTH_S = 0.12  # Gaussian sigma of the pelvis path
# a move shorter than this is one omnidirectional shuffle instead of turn-walk-turn. Dev 11-25: ScaleBFM does not follow
# a sideways shuffle (0.3 m asked, 0.0 m moved, 5/5, with 10 or 15 cm steps); turn-walk-turn moved it to 6 cm. So 0:
# every move of MOVE_MIN or more is turn-walk-turn; a shorter one is a turn in place carrying the small shift along
SHORT_M = _E("SHORT_M", 0.0)
MOVE_MIN = _E("MOVE_MIN", 0.15)
TURN_FIRST = math.radians(10.0)  # turn toward the travel direction first when it is off the heading by more
SOLE_C = np.array([0.05, 0.0, -0.025])  # ankle_roll_link -> the centre of the sole's collision capsules (robot_fk)
# odometry (dev 11-25, 5 cases x 5 reps, error to the true pelvis at the end): "max" (one stance foot, the most loaded;
# switch when the other carries >= LOAD_MIN of the feet's force) 0.5-3.4 cm median; "weighted" (every foot with >= LOAD_MIN,
# load-weighted) 3-17 cm; the lower foot as stance, ankle point (v1) 3.6-19 cm (turns worst)
LOAD_MIN = _E("LOAD_MIN", 0.5)
ODO_MODE = os.environ.get("HG_ODO_MODE", "max")


def rot_z(a):
    return L.rot_z(a)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class LegModel:
    """robot_fk legs: FK of the ankle_roll_links and per-frame leg IK (own MjData, the Planner's model)."""

    def __init__(self, pl):
        import mujoco

        self.mj = mujoco
        self.m = pl.ik.model
        self.d = mujoco.MjData(self.m)
        self.base = int(pl.ik.base_qadr)
        self.qadr = np.asarray(pl.ik.qadr)  # 29 joints, MuJoCo order
        self.foot = {s: self.m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")}
        jn = [self.m.joint(int(self.m.body(self.foot["left"]).jntadr[0])).name]  # noqa: F841 (sanity)
        self.leg = {"left": np.arange(0, 6), "right": np.arange(6, 12)}  # indices into the 29
        dof = []
        for i in range(29):
            jid = int(np.where(self.m.jnt_qposadr == self.qadr[i])[0][0])
            dof.append(int(self.m.jnt_dofadr[jid]))
        self.dof = np.array(dof)
        self.lo = self.m.jnt_range[[int(np.where(self.m.jnt_qposadr == a)[0][0]) for a in self.qadr], 0]
        self.hi = self.m.jnt_range[[int(np.where(self.m.jnt_qposadr == a)[0][0]) for a in self.qadr], 1]

    def set(self, p, quat, q):
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.base : self.base + 3] = p
        d.qpos[self.base + 3 : self.base + 7] = quat
        d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, d)

    def feet(self, p, quat, q) -> dict:
        self.set(p, quat, q)
        return {s: (self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()) for s, b in self.foot.items()}

    def rel_feet(self, q) -> dict:
        """Ankle_roll_link positions in the pelvis frame (encoders)."""
        f = self.feet(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
        return {s: v[0] for s, v in f.items()}

    def solve(self, p, quat, q, targets: dict, iters: int = 8) -> tuple[np.ndarray, float]:
        """Leg joints of q so each ankle_roll_link reaches targets[s] = (pos, R); returns (q, worst position error)."""
        mj, m, d = self.mj, self.m, self.d
        q = q.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = 0.0
        for _ in range(iters):
            self.set(p, quat, q)
            mj.mj_comPos(m, d)
            err = 0.0
            for s, (tp, tR) in targets.items():
                b = self.foot[s]
                ep = tp - d.xpos[b]
                Rc = d.xmat[b].reshape(3, 3)
                Re = tR @ Rc.T
                er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                err = max(err, float(np.linalg.norm(ep)))
                mj.mj_jacBody(m, d, jacp, jacr, b)
                idx = self.leg[s]
                J = np.vstack([jacp[:, self.dof[idx]], 0.5 * jacr[:, self.dof[idx]]])
                e = np.r_[ep, 0.5 * er]
                lam = 1e-3
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
                q[idx] = np.clip(q[idx] + dq, self.lo[idx], self.hi[idx])
        return q, err


def _plan_knots(S, G) -> list[tuple[np.ndarray, float]]:
    """Pelvis poses (xy, yaw) at the step knots from S to G (both included)."""
    (s_xy, s_yaw), (g_xy, g_yaw) = S, G
    knots = [(s_xy.copy(), s_yaw)]
    d = g_xy - s_xy
    dist = float(np.linalg.norm(d))

    def seg(a_xy, a_yaw, b_xy, b_yaw, omni: bool):
        dyaw = wrap(b_yaw - a_yaw)
        dl = rot_z(a_yaw).T @ np.r_[b_xy - a_xy, 0.0]
        n = max(1, int(math.ceil(abs(dyaw) / STEP_YAW)))
        if omni:
            n = max(n, int(math.ceil(abs(dl[0]) / STEP_LEN)), int(math.ceil(abs(dl[1]) / STEP_SIDE)))
        else:
            n = max(n, int(math.ceil(float(np.linalg.norm(b_xy - a_xy)) / STEP_LEN)))
        for k in range(1, n + 1):
            u = k / n
            knots.append((a_xy + u * (b_xy - a_xy), a_yaw + u * dyaw))

    if dist < max(SHORT_M, MOVE_MIN):
        seg(s_xy, s_yaw, g_xy, s_yaw + wrap(g_yaw - s_yaw), True)
        return knots
    h = math.atan2(d[1], d[0])
    yaw = s_yaw
    if abs(wrap(h - s_yaw)) > TURN_FIRST:
        seg(s_xy, s_yaw, s_xy, s_yaw + wrap(h - s_yaw), False)
        yaw = s_yaw + wrap(h - s_yaw)
    seg(s_xy, yaw, g_xy, yaw, False)
    if abs(wrap(g_yaw - yaw)) > math.radians(1.0):
        seg(g_xy, yaw, g_xy, yaw + wrap(g_yaw - yaw), False)
    return knots


def walk_frames(dr, x: dict, ref_frame: np.ndarray, goal_xy, goal_yaw: float, pose=None, info: dict | None = None):
    """Reference frames (N, 36) at 50 Hz, reference coordinates, from the frame after ref_frame to a held default stance
    at the goal. goal_xy / goal_yaw: world, in the frame of `pose` = (pelvis xy, yaw) the caller believes in (default
    the measured pelvis in x). info (optional dict) gets the plan: knots, steps, IK error, walk duration."""
    if not hasattr(dr, "_legs"):
        dr._legs = LegModel(dr.pl)
    lm: LegModel = dr._legs
    if pose is None:
        pose = (np.asarray(x["root"][:2], float), L.yaw_of(L._qmat(x["root"][3:])))
    b_xy, b_yaw = np.asarray(pose[0], float), float(pose[1])
    r_xy = np.asarray(ref_frame[:2], float)
    r_yaw = L.yaw_of(L._qmat(ref_frame[3:7]))
    dy = wrap(r_yaw - b_yaw)
    g_xy = r_xy + (rot_z(dy) @ np.r_[np.asarray(goal_xy, float) - b_xy, 0.0])[:2]
    g_yaw = r_yaw + wrap(goal_yaw + dy - r_yaw)
    knots = _plan_knots((r_xy, r_yaw), (g_xy, g_yaw))
    n = len(knots) - 1

    p0 = np.asarray(ref_frame[:3], float)
    q0 = np.asarray(ref_frame[7:], float)
    feet0 = lm.feet(p0, np.asarray(ref_frame[3:7], float), q0)
    # nominal foot offsets in the pelvis heading frame (default stance)
    nom = lm.feet(np.array([0.0, 0.0, dr.pl.stand_z]), np.array([1.0, 0, 0, 0]), L.DEFAULT_Q)
    off = {s: nom[s][0] for s in nom}  # xyz, z = ankle height at stand

    def nominal(xy, yaw, s):
        return np.r_[xy + (rot_z(yaw) @ np.r_[off[s][:2], 0.0])[:2], off[s][2]]

    # first swing foot: the one on the side of the first move (turn left / move left -> left)
    if n > 0:
        k1_xy, k1_yaw = knots[1]
        lat = (rot_z(r_yaw).T @ np.r_[k1_xy - r_xy, 0.0])[1]
        turn = wrap(k1_yaw - r_yaw)
        first = "left" if (turn > 1e-3 or (abs(turn) <= 1e-3 and lat > 1e-3)) else "right"
    else:
        first = "right"
    other = {"left": "right", "right": "left"}
    # footsteps: step j (1..n) swings `sw` to nominal(knot j); one closing step puts the other foot at nominal(knot n)
    foot = {s: (feet0[s][0].copy(), L.yaw_of(feet0[s][1])) for s in ("left", "right")}
    steps = []
    sw = first
    for j in range(1, n + 1):
        xy, yaw = knots[j]
        steps.append((sw, foot[sw], (nominal(xy, yaw, sw), yaw), j))
        foot[sw] = (nominal(xy, yaw, sw), yaw)
        sw = other[sw]
    xy, yaw = knots[n]
    for s in (sw, other[sw]):  # close: both feet at the final nominal (skipped when already there)
        tgt = nominal(xy, yaw, s)
        if float(np.linalg.norm(foot[s][0][:2] - tgt[:2])) > 0.01 or abs(wrap(foot[s][1] - yaw)) > math.radians(2):
            steps.append((s, foot[s], (tgt, yaw), n))
            foot[s] = (tgt, yaw)

    lead = BLEND_S  # pelvis lowers before the first step
    T_walk = lead + len(steps) * STEP_T
    T = T_walk + BLEND_S + HOLD_END_S
    N = int(round(T * L.FPS))
    t = (np.arange(N) + 1) / L.FPS
    # pelvis knots in time: knot j reached at the end of step j (the closing steps keep knot n)
    kt = [0.0, lead] + [lead + (i + 1) * STEP_T for i in range(len(steps))]
    kxy = [r_xy, r_xy] + [knots[st[3]][0] for st in steps]
    kyaw = [r_yaw, r_yaw] + [knots[st[3]][1] for st in steps]
    kt.append(T)
    kxy.append(kxy[-1])
    kyaw.append(kyaw[-1])
    kt = np.array(kt)
    pxy = np.stack([np.interp(t, kt, np.array(kxy)[:, i]) for i in range(2)], 1)
    pyaw = np.interp(t, kt, np.unwrap(np.array(kyaw)))
    # sway toward the stance foot during each swing
    sway = np.zeros((N, 2))
    for i, (s, a, b, _) in enumerate(steps):
        t0 = lead + i * STEP_T
        st_xy = None
        u = (t - t0) / STEP_T
        msk = (u >= 0) & (u <= 1)
        # stance foot during this step = the other foot's current placement
        st_side = other[s]
        st_xy = _foot_at(steps, i, st_side, feet0)[:2]
        mid = np.interp(t[msk], kt, np.array(kxy)[:, 0]), np.interp(t[msk], kt, np.array(kxy)[:, 1])
        v = st_xy[None, :] - np.stack(mid, 1)
        nv = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        sway[msk] += SWAY * np.sin(np.pi * u[msk])[:, None] * v / nv
    pxy = pxy + sway
    # Gaussian smoothing (edge-padded)
    sig = SMOOTH_S * L.FPS
    kr = int(3 * sig)
    if kr > 0:
        w = np.exp(-0.5 * (np.arange(-kr, kr + 1) / sig) ** 2)
        w /= w.sum()
        pad = lambda a: np.concatenate([np.repeat(a[:1], kr, 0), a, np.repeat(a[-1:], kr, 0)])  # noqa: E731
        pxy = np.stack([np.convolve(pad(pxy[:, i]), w, "valid") for i in range(2)], 1)
        pyaw = np.convolve(pad(pyaw), w, "valid")
    # pelvis height: ref z -> WALK_Z over the lead blend, back to stand_z after the last step
    z0, zs = float(p0[2]), dr.pl.stand_z
    pz = np.where(t < lead, z0 + (WALK_Z - z0) * np.array([L.minjerk(min(1, v / lead)) for v in t]),
                  np.where(t < T_walk, WALK_Z, WALK_Z + (zs - WALK_Z) * np.array([L.minjerk(min(1, max(0, (v - T_walk) / BLEND_S))) for v in t])))  # fmt: skip
    # feet over time
    fpos = {s: np.zeros((N, 3)) for s in ("left", "right")}
    fyaw = {s: np.zeros(N) for s in ("left", "right")}
    for s in ("left", "right"):
        fpos[s][:] = feet0[s][0]
        fyaw[s][:] = L.yaw_of(feet0[s][1])
    for i, (s, a, b, _) in enumerate(steps):
        t0 = lead + i * STEP_T
        u = (t - t0) / STEP_T
        su = np.clip((u - DS / 2) / (1 - DS), 0, 1)
        msk = u >= 0
        mj_ = np.array([L.minjerk(v) for v in su[msk]])
        a_p, a_y = a
        b_p, b_y = b
        dyaw = wrap(b_y - a_y)
        fpos[s][msk] = a_p[None] + mj_[:, None] * (b_p - a_p)[None]
        fpos[s][msk, 2] += SWING_H * np.sin(np.pi * su[msk])
        fyaw[s][msk] = a_y + mj_ * dyaw
    # upper body: ref -> DEFAULT_Q over the lead blend
    out = np.empty((N, 36), np.float32)
    q = q0.copy()
    worst = 0.0
    for f in range(N):
        bl = L.minjerk(min(1.0, t[f] / BLEND_S))
        q[12:] = q0[12:] + bl * (L.DEFAULT_Q[12:] - q0[12:])
        quat = quat_yaw(pyaw[f])
        p = np.r_[pxy[f], pz[f]]
        tg = {s: (fpos[s][f], rot_z(fyaw[s][f])) for s in ("left", "right")}
        q, e = lm.solve(p, quat, q, tg)
        if t[f] > T_walk + BLEND_S:  # the held stance: exactly the default pose at the final knot
            q = q.copy()
        worst = max(worst, e)
        out[f] = np.r_[p, quat, q]
    # end exactly in the default stance (legs from IK at stand_z equal DEFAULT_Q up to the IK error; snap the hold)
    fin = np.r_[pxy[-1], zs, quat_yaw(pyaw[-1]), L.DEFAULT_Q]
    nh = int(round(HOLD_END_S * L.FPS))
    nb = int(round(0.3 * L.FPS))
    for k in range(nb):  # 0.3 s blend from the IK'd stance to the exact default
        f = N - nh - nb + k
        w = L.minjerk((k + 1) / nb)
        out[f, 7:] = (1 - w) * out[f, 7:] + w * L.DEFAULT_Q
    out[N - nh :] = fin
    if info is not None:
        info.update(n_knots=n, n_steps=len(steps), walk_s=T_walk, total_s=T, ik_err_cm=100 * worst, first=first,
                    goal_ref=[float(g_xy[0]), float(g_xy[1]), float(g_yaw)])  # fmt: skip
    return out


def _foot_at(steps, i, side, feet0):
    """Where `side` stands during step i (its last landing before step i)."""
    p = feet0[side][0]
    for k in range(i):
        if steps[k][0] == side:
            p = steps[k][2][0]
    return p


class LegOdo:
    """Pelvis xy / yaw from encoders + IMU orientation + the loaded feet anchored (no simulator positions).

    Each foot's sole centre (SOLE_C on its ankle_roll_link, FK of the encoders, rotated by the IMU) is anchored in the
    world while the foot carries >= LOAD_MIN of the feet's normal force (the foot's contact force: what a foot force
    sensor reads); the pelvis is the load-weighted mean of anchor - sole offset. A foot that unloads drops its anchor
    and is re-anchored where it lands (from the pelvis estimate at that moment)."""

    def __init__(self, dr, x0: dict, pose0=None):
        if not hasattr(dr, "_legs"):
            dr._legs = LegModel(dr.pl)
        self.lm = dr._legs
        self.to_mj = dr.to_mj
        self._foot_geoms(dr)
        R = L._qmat(x0["root"][3:])
        if pose0 is None:
            pose0 = (np.asarray(x0["root"][:2], float), L.yaw_of(R))
        self.yaw_off = wrap(float(pose0[1]) - L.yaw_of(R))
        Rw = rot_z(self.yaw_off) @ R
        pt = self._soles(x0, Rw)
        z = self.lm.feet(np.array([0.0, 0, 0.8]), np.array([1.0, 0, 0, 0]), L.DEFAULT_Q)["left"][0][2] - min(
            v[2] - (Rw @ SOLE_C)[2] for v in pt.values()
        )
        self.p = np.r_[np.asarray(pose0[0], float), z]
        self.anch = {s: self.p + pt[s] for s in pt}
        self.R = Rw
        self.switches = 0

    def _foot_geoms(self, dr) -> None:
        m, mj = dr.m, dr.mj
        self.gfoot = {}
        for g in range(m.ngeom):
            b = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            for s in ("left", "right"):
                if b == f"robot0_{s}_ankle_roll_link":
                    self.gfoot[g] = s
        self._f6 = np.zeros(6)

    def load(self, dr) -> dict:
        """Normal contact force on each foot from everything that is not the robot."""
        m, d, mj = dr.m, dr.d, dr.mj
        w = {"left": 0.0, "right": 0.0}
        for i in range(d.ncon):
            c = d.contact[i]
            s = self.gfoot.get(int(c.geom1)) or self.gfoot.get(int(c.geom2))
            if s is None:
                continue
            o = int(c.geom2) if int(c.geom1) in self.gfoot else int(c.geom1)
            ob = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[o])) or ""
            if ob.startswith("robot0_"):
                continue
            mj.mj_contactForce(m, d, i, self._f6)
            w[s] += abs(float(self._f6[0]))
        return w

    def _soles(self, x, R) -> dict:
        f = self.lm.feet(np.zeros(3), np.array([1.0, 0, 0, 0]), x["q"][self.to_mj])
        return {s: R @ (p + Rf @ SOLE_C) for s, (p, Rf) in f.items()}

    def update(self, dr, x: dict) -> None:
        R = rot_z(self.yaw_off) @ L._qmat(x["root"][3:])
        pt = self._soles(x, R)
        w = self.load(dr)
        W = sum(w.values())
        if W > 1.0 and ODO_MODE == "max":
            top = max(w, key=w.get)
            cur = next(iter(self.anch)) if len(self.anch) == 1 else None
            if cur is None or (top != cur and w[top] >= LOAD_MIN * W):
                p_now = self.p if cur is None else self.anch[cur] - pt[cur]
                self.anch = {top: p_now + pt[top]}
                self.switches += 1
        elif W > 1.0:
            for s in ("left", "right"):
                if w[s] >= LOAD_MIN * W:
                    if s not in self.anch:
                        self.anch[s] = self.p + pt[s]
                        self.switches += 1
                else:
                    self.anch.pop(s, None)
        if self.anch:
            ws = {s: max(w[s], 1e-6) for s in self.anch}
            self.p = sum(ws[s] * (self.anch[s] - pt[s]) for s in self.anch) / sum(ws.values())
        self.R = R

    def pose(self) -> tuple[np.ndarray, float]:
        return self.p[:2].copy(), L.yaw_of(self.R)


# -- tests --------------------------------------------------------------------------------------------------------
CASES = {  # name: (forward m, left m, turn deg)
    "walk1.5": (1.5, 0.0, 0.0), "walk2.0": (2.0, 0.0, 0.0), "walk3.0": (3.0, 0.0, 0.0),
    "turn45": (0.0, 0.0, 45.0), "turn90": (0.0, 0.0, 90.0), "turn160": (0.0, 0.0, 160.0),
    "diag2.0": (1.6, 1.2, 40.0), "back2.0": (-1.2, 1.5, 158.0), "side0.3": (0.1, -0.3, 0.0),
    "fwd0.4": (0.4, 0.0, 0.0), "side0.3turn": (0.1, -0.3, -30.0),
}  # fmt: skip


SCENES = {"IThorKitchenApproachTableAndLiftApple": 13, "IThorLivingRoomApproachTableAndLiftDumbbell": 207}


def test(out: Path, task: str, case: str, reps: int, worker: int) -> None:
    import lml_tasks as LT

    out.mkdir(parents=True, exist_ok=True)
    LT.TASKS.setdefault(task, dict(scene=SCENES.get(task), max_steps=0, genuine=0))
    dr = LT.TaskDriver(task, seed=worker)
    for rep in range(reps):
        fn = out / f"{task}_{case}_r{rep}.json"
        if fn.exists():
            continue
        t0 = time.time()
        dr.reset_episode(rep)
        rng = np.random.default_rng(1000 + rep)
        frames = dr.start_frames()
        f, rows = 0, {}
        f_walk = int(round(L.START_HOLD_S * L.FPS))
        res = dict(task=task, case=case, rep=rep)
        odo = None
        maxdev = 0.0
        anchor = None
        while True:
            dr.push(frames[min(f, len(frames) - 1)])
            if dr.buf is None:
                dr.buf = dr._find_buffer()
            dr.step()
            x = rows[f] = dr.sample()
            if odo is None:
                odo = LegOdo(dr, x)
            else:
                odo.update(dr, x)
            if float(x["root"][2]) < L.FALL_Z:
                res.update(fell=f)
                break
            if anchor is not None:
                dev = (x["root"][:2] - anchor[0]) - (frames[min(f, len(frames) - 1), :2] - anchor[1])
                maxdev = max(maxdev, float(np.linalg.norm(dev)))
            if case.startswith("squat") and f == f_walk:
                # squat: a low reach ahead, pelvis_z asked (case squat<pz>_<z>)
                pz, tz = (float(v) for v in case[5:].split("_"))
                c = dict(side="right", target=np.array([0.45 + 0.1 * rng.random(), -0.15, tz]), dyaw=0.0,
                         pitch=math.radians(60), pelvis_z=pz)  # fmt: skip
                blend, last, rec = dr.plan(x, frames[f], c)
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, 1.5)])
                res.update(ik_err_cm=100 * rec["ik_err"], target_w=rec["target_w"], f_end=f + len(blend))
                anchor = (x["root"][:2].copy(), frames[f, :2].copy())
            elif f == f_walk:
                fw, lw, tw = CASES[case]
                p, yaw = x["root"][:2], L.yaw_of(L._qmat(x["root"][3:]))
                j = rng.uniform(-1, 1, 3) * np.array([0.05, 0.05, math.radians(10)])
                g_xy = p + (rot_z(yaw) @ np.r_[fw + j[0], lw + j[1], 0.0])[:2]
                g_yaw = yaw + math.radians(tw) + j[2]
                info = {}
                tic = time.time()
                wf = walk_frames(dr, x, frames[f], g_xy, g_yaw, info=info)
                info["plan_s"] = time.time() - tic
                frames = np.concatenate([frames[: f + 1], wf])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy())
                res.update(goal=[*g_xy.tolist(), g_yaw], start=[*p.tolist(), yaw], plan=info)
            if f >= len(frames) - 1:
                break
            f += 1
        x = rows[max(rows)]
        p, yaw = x["root"][:2], L.yaw_of(L._qmat(x["root"][3:]))
        op, oyaw = odo.pose()
        res.update(fell=res.get("fell"), steps=len(rows), maxdev_cm=100 * maxdev, wall_s=time.time() - t0,
                   end=[*p.tolist(), yaw], odo=[*op.tolist(), oyaw], odo_err_cm=100 * float(np.linalg.norm(op - p)),
                   odo_yaw_err_deg=math.degrees(wrap(oyaw - yaw)), odo_switches=odo.switches)  # fmt: skip
        if "goal" in res:
            g = np.array(res["goal"])
            res.update(pos_err_cm=100 * float(np.linalg.norm(p - g[:2])), yaw_err_deg=math.degrees(wrap(yaw - g[2])))
        if "target_w" in res and res.get("fell") is None:
            fe = min(res["f_end"] + 50, max(rows))
            pw, Rw = rows[fe]["wr"]["right"]
            res["hand_err_cm"] = 100 * float(np.linalg.norm(pw + Rw @ L.HAND_OFF - np.array(res["target_w"])))
            res["pelvis_z_min"] = float(min(r["root"][2] for r in rows.values()))
        fn.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items() if k not in ("plan", "goal", "start", "end", "odo", "target_w")}), flush=True)  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="FreeSpace")
    ap.add_argument("--case", default="walk2.0")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()
    test(a.out, a.task, a.case, a.reps, a.worker)
