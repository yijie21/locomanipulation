"""A walk as a whole-body gait reference, for a controller that only tracks motion (HEFT has no velocity command).

Vendored from scripts/experiments/hardgrasp_20261125/hg_walk.py (`LegModel`, `_plan_knots`, `walk_frames`,
`_foot_at`; 2026-11-25), the gait reference LocoManiLab's HEFT walked on in selfmodel_data3_20261125 (d3_collect.py):
a footstep plan (turn toward the goal in place, walk straight, turn to the goal heading in place; a move shorter than
MOVE_MIN is a turn in place carrying the shift along, in omnidirectional steps), the pelvis carried through the step
knots (linear, Gaussian-smoothed, a small sway toward the stance foot, lowered to WALK_Z while walking), each swing foot
on a min-jerk arc lifted SWING_H, and the 12 leg joints solved per frame by damped least squares (ankle_roll_link
position + flat foot at its yaw). Constants as hg_walk's frozen run (no HG_* overrides).

Changed for the Motor layer, nothing else: the FK model is the motor's IK model (`ik`: ArmIK's model, base and joint
addresses; hg_walk took lml_selfmodel.Planner's robot_fk); the walk ends in the stance it started from -- the start
frame's joints and pelvis height (`Motor.walk_to` keeps both too) -- where hg_walk ended in LocoManiLab's default pose at
its stand_z, so the nominal foot offsets are the start stance's and the upper body stays as it was (hg_walk blended it to
DEFAULT_Q); the pelvis is not raised to WALK_Z when the start stands lower; `pose` is required (hg_walk defaulted it to
the measured pelvis of its host sample). The lml_selfmodel helpers it used (FPS, minjerk, rot_z, yaw_of, _qmat) are
copied here.
"""

from __future__ import annotations

import math

import numpy as np

FPS = 50
STEP_LEN = 0.20  # m per step, forward
STEP_SIDE = 0.10  # m per step, sideways (short moves)
STEP_YAW = math.radians(15.0)  # per step
STEP_T = 0.55  # s per step
DS = 0.25  # double-support fraction of a step (split before / after the swing)
SWING_H = 0.06  # m
SWAY = 0.025  # m, pelvis toward the stance foot at mid-swing
WALK_Z = 0.77  # pelvis height while walking
BLEND_S = 0.5  # in / out blends (pelvis height)
HOLD_END_S = 1.5  # the stance held at the end
SMOOTH_S = 0.12  # Gaussian sigma of the pelvis path
SHORT_M = 0.0  # hg_walk: every move of MOVE_MIN or more is turn-walk-turn
MOVE_MIN = 0.15
TURN_FIRST = math.radians(10.0)  # turn toward the travel direction first when it is off the heading by more


def minjerk(u: float) -> float:
    return u * u * u * (10 - 15 * u + 6 * u * u)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def qmat(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


class LegModel:
    """FK of the ankle_roll_links and per-frame leg IK on `ik`'s model (own MjData)."""

    def __init__(self, ik):
        import mujoco

        self.mj = mujoco
        self.m = ik.model
        self.d = mujoco.MjData(self.m)
        self.base = int(ik.base_qadr)
        self.qadr = np.asarray(ik.qadr)  # 29 joints, MuJoCo order
        self.foot = {s: self.m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")}
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


def plan_knots(S, G) -> list[tuple[np.ndarray, float]]:
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


def _foot_at(steps, i, side, feet0):
    """Where `side` stands during step i (its last landing before step i)."""
    p = feet0[side][0]
    for k in range(i):
        if steps[k][0] == side:
            p = steps[k][2][0]
    return p


def walk_frames(lm: LegModel, ref_frame: np.ndarray, pose, goal_xy, goal_yaw: float, info: dict | None = None):
    """Reference frames (N, 36: pelvis xyz, wxyz, 29 joints MuJoCo order) at 50 Hz, the frame after `ref_frame` first,
    to the start's stance held at the goal. `pose` = (pelvis xy, yaw) the caller believes the robot is at, and goal_xy /
    goal_yaw are in the same world; the plan is carried onto the reference frame's root (hg_walk's convention). `info`
    (optional dict) gets the plan: knots, steps, IK error, walk duration."""
    other = {"left": "right", "right": "left"}
    b_xy, b_yaw = np.asarray(pose[0], float), float(pose[1])
    r_xy = np.asarray(ref_frame[:2], float)
    r_yaw = yaw_of(qmat(ref_frame[3:7]))
    dy = wrap(r_yaw - b_yaw)
    g_xy = r_xy + (rot_z(dy) @ np.r_[np.asarray(goal_xy, float) - b_xy, 0.0])[:2]
    g_yaw = r_yaw + wrap(goal_yaw + dy - r_yaw)
    knots = plan_knots((r_xy, r_yaw), (g_xy, g_yaw))
    n = len(knots) - 1

    p0 = np.asarray(ref_frame[:3], float)
    q0 = np.asarray(ref_frame[7:], float)
    zs = float(p0[2])  # the stance's own height (hg_walk: robot_fk's stand_z)
    walk_z = min(WALK_Z, zs)
    feet0 = lm.feet(p0, np.asarray(ref_frame[3:7], float), q0)
    # nominal foot offsets in the pelvis heading frame (the start's stance, upright)
    nom = lm.feet(np.array([0.0, 0.0, zs]), np.array([1.0, 0, 0, 0]), q0)
    off = {s: nom[s][0] for s in nom}  # xyz, z = ankle height at the stance

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
    # footsteps: step j (1..n) swings `sw` to nominal(knot j); one closing step puts the other foot at nominal(knot n)
    foot = {s: (feet0[s][0].copy(), yaw_of(feet0[s][1])) for s in ("left", "right")}
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
    N = int(round(T * FPS))
    t = (np.arange(N) + 1) / FPS
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
    for i, (s, _a, _b, _) in enumerate(steps):
        t0 = lead + i * STEP_T
        u = (t - t0) / STEP_T
        msk = (u >= 0) & (u <= 1)
        st_xy = _foot_at(steps, i, other[s], feet0)[:2]  # stance foot during this step: the other foot's placement
        mid = np.interp(t[msk], kt, np.array(kxy)[:, 0]), np.interp(t[msk], kt, np.array(kxy)[:, 1])
        v = st_xy[None, :] - np.stack(mid, 1)
        nv = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        sway[msk] += SWAY * np.sin(np.pi * u[msk])[:, None] * v / nv
    pxy = pxy + sway
    # Gaussian smoothing (edge-padded)
    sig = SMOOTH_S * FPS
    kr = int(3 * sig)
    if kr > 0:
        w = np.exp(-0.5 * (np.arange(-kr, kr + 1) / sig) ** 2)
        w /= w.sum()
        pad = lambda a: np.concatenate([np.repeat(a[:1], kr, 0), a, np.repeat(a[-1:], kr, 0)])  # noqa: E731
        pxy = np.stack([np.convolve(pad(pxy[:, i]), w, "valid") for i in range(2)], 1)
        pyaw = np.convolve(pad(pyaw), w, "valid")
    # pelvis height: ref z -> walk_z over the lead blend, back to the stance height after the last step
    pz = np.where(t < lead, zs + (walk_z - zs) * np.array([minjerk(min(1, v / lead)) for v in t]),
                  np.where(t < T_walk, walk_z, walk_z + (zs - walk_z) * np.array([minjerk(min(1, max(0, (v - T_walk) / BLEND_S))) for v in t])))  # fmt: skip
    # feet over time
    fpos = {s: np.zeros((N, 3)) for s in ("left", "right")}
    fyaw = {s: np.zeros(N) for s in ("left", "right")}
    for s in ("left", "right"):
        fpos[s][:] = feet0[s][0]
        fyaw[s][:] = yaw_of(feet0[s][1])
    for i, (s, a, b, _) in enumerate(steps):
        t0 = lead + i * STEP_T
        u = (t - t0) / STEP_T
        su = np.clip((u - DS / 2) / (1 - DS), 0, 1)
        msk = u >= 0
        mj_ = np.array([minjerk(v) for v in su[msk]])
        a_p, a_y = a
        b_p, b_y = b
        dyaw = wrap(b_y - a_y)
        fpos[s][msk] = a_p[None] + mj_[:, None] * (b_p - a_p)[None]
        fpos[s][msk, 2] += SWING_H * np.sin(np.pi * su[msk])
        fyaw[s][msk] = a_y + mj_ * dyaw
    out = np.empty((N, 36), np.float32)
    q = q0.copy()
    worst = 0.0
    for f in range(N):
        quat = quat_yaw(pyaw[f])
        p = np.r_[pxy[f], pz[f]]
        tg = {s: (fpos[s][f], rot_z(fyaw[s][f])) for s in ("left", "right")}
        q, e = lm.solve(p, quat, q, tg)
        worst = max(worst, e)
        out[f] = np.r_[p, quat, q]
    # end exactly in the start's stance (legs from IK at its height equal its legs up to the IK error; snap the hold)
    fin = np.r_[pxy[-1], zs, quat_yaw(pyaw[-1]), q0]
    nh = int(round(HOLD_END_S * FPS))
    nb = int(round(0.3 * FPS))
    for k in range(nb):  # 0.3 s blend from the IK'd stance to the exact one
        f = N - nh - nb + k
        w = minjerk((k + 1) / nb)
        out[f, 7:] = (1 - w) * out[f, 7:] + w * q0
    out[N - nh :] = fin
    if info is not None:
        info.update(n_knots=n, n_steps=len(steps), walk_s=T_walk, total_s=T, ik_err_cm=100 * worst, first=first,
                    goal_ref=[float(g_xy[0]), float(g_xy[1]), float(g_yaw)])  # fmt: skip
    return out


__all__ = ["LegModel", "plan_knots", "walk_frames"]
