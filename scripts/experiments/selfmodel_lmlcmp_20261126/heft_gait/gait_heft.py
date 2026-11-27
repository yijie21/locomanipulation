"""A footstep gait reference HEFT keeps up with: gait.py's plan with the step shapes HEFT walks on.

HEFT is a motion tracker trained on human motion (its policy.yaml's motion_path is any4hdmi's LAFAN G1 walk1_subject1).
On gait.py's reference (hg_walk's constants, tuned under ScaleBFM: every step 0.55 s, a 6 cm swing, a 2.5 cm sway) it
walked about 60 % of a 0.5 m walk and almost none of a sideways move, in SIMPLE and in LocoManiLab alike, while it tracks
LAFAN walk1_subject1 itself to 2-5 cm (heftsimple_20261125). The LAFAN walks lift the ankle about 10 cm on slow forward
steps but only a few cm when turning on the spot (lafan_stats.py), and HEFT follows that split: forward steps need a high
swing and a weight shift (on gait.py's they land short), turns in place a low one (on gait.py's 15-degree steps a
90-degree turn ends 11 degrees short and 7-12 cm off its spot; with a high swing the spot creeps 5-10 cm backward).

The plan is gait.py's (turn toward the goal on the spot, walk straight, turn to the goal heading; a move shorter than
`move_min` steps there directly), with one `Shape` per kind of step -- `walk` (a knot ahead, or a short move mostly
ahead), `turn` (a knot rotated on the spot), `side` (a short move mostly aside) -- each with its own duration, double
support, swing height and sway:

  swing      the swing foot's horizontal move and yaw finish `land_lead` of the swing before touchdown and it descends
             over the last `descend` of it, so a reference followed a few frames late still lands where it was planned
             (land_lead 0, descend 0.5: gait.py's single min-jerk arc under a sine lift).
  sway       the pelvis toward the stance foot at mid-swing.

Other changes from gait.py: the first swing foot is the one on the side of the move unless the first knot moves less
than 2 cm aside (gait.py let a heading correction of a few tenths of a degree pick the trailing foot of a sidestep).
No feedback: the reference is a function of the plan only (Motor.walk_to's goal and the start stance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from locomani.backends.heft import gait as G
from locomani.backends.heft.gait import LegModel, minjerk, qmat, quat_yaw, rot_z, wrap, yaw_of


@dataclass(frozen=True)
class Shape:
    """One kind of step."""

    step_t: float = 0.55  # s a step
    ds: float = 0.25  # double-support share of a step (half before, half after the swing)
    swing_h: float = 0.06  # m, ankle lift
    sway: float = 0.025  # m, pelvis toward the stance foot at mid-swing
    land_lead: float = 0.0  # share of the swing the horizontal move finishes before touchdown
    descend: float = 0.5  # share of the swing the foot descends over (0.5 with land_lead 0: gait.py's sine arc)


@dataclass(frozen=True)
class HeftGait:
    walk: Shape = field(default_factory=Shape)
    turn: Shape = field(default_factory=Shape)
    side: Shape = field(default_factory=Shape)
    step_len: float = 0.20  # m a step, forward
    step_side: float = 0.10  # m a step, sideways (moves under move_min)
    step_yaw: float = math.radians(15.0)  # rad a step, turning
    walk_z: float = 0.77  # m, pelvis height while walking (not above the start stance's)
    blend_s: float = 0.5  # s, pelvis height blends in and out
    hold_end_s: float = 1.5  # s, the end stance held
    smooth_s: float = 0.12  # s, Gaussian sigma of the pelvis path
    move_min: float = 0.15  # m, shorter moves are omnidirectional steps instead of turn-walk-turn
    turn_first: float = math.radians(10.0)  # rad, turn toward the travel direction first when off by more


HG_WALK = HeftGait()  # gait.py's constants (hg_walk's frozen run) in this shape
HEFT = HeftGait(
    walk=Shape(step_t=0.40, swing_h=0.14, sway=0.05, land_lead=0.25, descend=0.3),
    turn=Shape(step_t=0.45, swing_h=0.05, sway=0.0, ds=0.4),
    side=Shape(step_t=0.45, swing_h=0.14, sway=0.0),
    step_yaw=math.radians(45.0),
    move_min=0.30,
)
# HEFT: heftsimple_20261125's sweeps (gait_try.py in SIMPLE, lml_gait.py in LocoManiLab), 5 starts each in both, stop
# error SIMPLE / LocoManiLab: 0.5 m ahead 2.2-2.4 / 3.2-3.4 cm, 0.23 m to the left 4.3-4.4 / 2.7-3.4 cm, a 90-degree turn
# 2.3-2.4 / 3.4-3.7 degrees short; no falls. A 0.23 m move turned, walked and turned back (move_min 0.15) stopped
# 9.0-9.8 / 7.3-8.3 cm off (the turn steps land under-rotated), hence move_min 0.30.


def plan_knots(S, Gl, g: HeftGait) -> list[tuple[np.ndarray, float, str]]:
    """gait.plan_knots with `g`'s step sizes: pelvis poses (xy, yaw, kind of the step to it) at the knots from S to Gl
    (both included; the first knot's kind is "")."""
    (s_xy, s_yaw), (g_xy, g_yaw) = S, Gl
    knots = [(s_xy.copy(), s_yaw, "")]
    d = g_xy - s_xy
    dist = float(np.linalg.norm(d))

    def seg(a_xy, a_yaw, b_xy, b_yaw, omni: bool):
        dyaw = wrap(b_yaw - a_yaw)
        dl = rot_z(a_yaw).T @ np.r_[b_xy - a_xy, 0.0]
        n = max(1, int(math.ceil(abs(dyaw) / g.step_yaw)))
        if omni:
            n = max(n, int(math.ceil(abs(dl[0]) / g.step_len)), int(math.ceil(abs(dl[1]) / g.step_side)))
        else:
            n = max(n, int(math.ceil(float(np.linalg.norm(b_xy - a_xy)) / g.step_len)))
        if float(np.linalg.norm(b_xy - a_xy)) < 1e-9:
            kind = "turn"
        else:
            kind = "side" if omni and abs(dl[1]) > abs(dl[0]) else "walk"
        for k in range(1, n + 1):
            u = k / n
            knots.append((a_xy + u * (b_xy - a_xy), a_yaw + u * dyaw, kind))

    if dist < g.move_min:
        seg(s_xy, s_yaw, g_xy, s_yaw + wrap(g_yaw - s_yaw), True)
        return knots
    h = math.atan2(d[1], d[0])
    yaw = s_yaw
    if abs(wrap(h - s_yaw)) > g.turn_first:
        seg(s_xy, s_yaw, s_xy, s_yaw + wrap(h - s_yaw), False)
        yaw = s_yaw + wrap(h - s_yaw)
    seg(s_xy, yaw, g_xy, yaw, False)
    if abs(wrap(g_yaw - yaw)) > math.radians(1.0):
        seg(g_xy, yaw, g_xy, yaw + wrap(g_yaw - yaw), False)
    return knots


def swing_profile(su: np.ndarray, sh: Shape) -> tuple[np.ndarray, np.ndarray]:
    """(horizontal progress 0..1, lift 0..1) over the swing phase su in [0, 1]."""
    hor = np.array([minjerk(v) for v in np.clip(su / max(1e-6, 1.0 - sh.land_lead), 0, 1)])
    if sh.land_lead == 0.0 and sh.descend == 0.5:
        return hor, np.sin(np.pi * su)  # gait.py's arc
    rise = max(1e-6, min(0.5, 1.0 - sh.descend))
    up = np.array([minjerk(v) for v in np.clip(su / rise, 0, 1)])
    down = np.array([minjerk(v) for v in np.clip((su - (1.0 - sh.descend)) / max(1e-6, sh.descend), 0, 1)])
    return hor, up * (1.0 - down)


def walk_frames(lm: LegModel, ref_frame: np.ndarray, pose, goal_xy, goal_yaw: float, g: HeftGait = HEFT,
                info: dict | None = None) -> np.ndarray:  # fmt: skip
    """gait.walk_frames with `g`'s step shapes: frames (N, 36: pelvis xyz, wxyz, 29 joints MuJoCo order) at 50 Hz from
    the one after `ref_frame` to the start's stance held at the goal; `pose` = (pelvis xy, yaw) the caller believes the
    robot is at, goal_xy / goal_yaw in the same world, the plan carried onto the reference frame's root."""
    fps = G.FPS
    other = {"left": "right", "right": "left"}
    b_xy, b_yaw = np.asarray(pose[0], float), float(pose[1])
    r_xy = np.asarray(ref_frame[:2], float)
    r_yaw = yaw_of(qmat(ref_frame[3:7]))
    dy = wrap(r_yaw - b_yaw)
    g_xy = r_xy + (rot_z(dy) @ np.r_[np.asarray(goal_xy, float) - b_xy, 0.0])[:2]
    g_yaw = r_yaw + wrap(goal_yaw + dy - r_yaw)
    knots = plan_knots((r_xy, r_yaw), (g_xy, g_yaw), g)
    n = len(knots) - 1

    p0 = np.asarray(ref_frame[:3], float)
    q0 = np.asarray(ref_frame[7:], float)
    zs = float(p0[2])
    walk_z = min(g.walk_z, zs)
    feet0 = lm.feet(p0, np.asarray(ref_frame[3:7], float), q0)
    nom = lm.feet(np.array([0.0, 0.0, zs]), np.array([1.0, 0, 0, 0]), q0)
    off = {s: nom[s][0] for s in nom}  # the start stance's ankle offsets in the pelvis heading frame

    def nominal(xy, yaw, s):
        return np.r_[xy + (rot_z(yaw) @ np.r_[off[s][:2], 0.0])[:2], off[s][2]]

    def apart(f, tgt, yaw) -> bool:  # a foot (pos, yaw) not yet at (tgt, yaw)
        return float(np.linalg.norm(f[0][:2] - tgt[:2])) > 0.01 or abs(wrap(f[1] - yaw)) > math.radians(2)

    if n > 0:
        k1_xy, k1_yaw, _ = knots[1]
        lat = (rot_z(r_yaw).T @ np.r_[k1_xy - r_xy, 0.0])[1]
        turn = wrap(k1_yaw - r_yaw)
        first = "left" if (lat > 0.02 or (abs(lat) <= 0.02 and turn > 1e-3)) else "right"
    else:
        first = "right"
    # footsteps: step j (1..n) swings `sw` to nominal(knot j); closing steps put the other foot at nominal(knot n)
    foot = {s: (feet0[s][0].copy(), yaw_of(feet0[s][1])) for s in ("left", "right")}
    steps = []  # (side, from (pos, yaw), to (pos, yaw), knot, shape)
    sw = first
    for j in range(1, n + 1):
        xy, yaw, kind = knots[j]
        steps.append((sw, foot[sw], (nominal(xy, yaw, sw), yaw), j, getattr(g, kind)))
        foot[sw] = (nominal(xy, yaw, sw), yaw)
        sw = other[sw]
    xy, yaw, kind = knots[n]
    for s in (sw, other[sw]):
        tgt = nominal(xy, yaw, s)
        if apart(foot[s], tgt, yaw):
            steps.append((s, foot[s], (tgt, yaw), n, getattr(g, kind or "walk")))
            foot[s] = (tgt, yaw)

    lead = g.blend_s
    t_step = lead + np.r_[0.0, np.cumsum([st[4].step_t for st in steps])]  # step i runs t_step[i] .. t_step[i + 1]
    T_walk = float(t_step[-1])
    T = T_walk + g.blend_s + g.hold_end_s
    N = int(round(T * fps))
    t = (np.arange(N) + 1) / fps
    kt = np.r_[0.0, lead, t_step[1:], T]  # the pelvis reaches knot j at the end of step j (closing steps keep knot n)
    kxy = np.array([r_xy, r_xy] + [knots[st[3]][0] for st in steps] + [knots[steps[-1][3]][0] if steps else r_xy])
    kyaw = np.array([r_yaw, r_yaw] + [knots[st[3]][1] for st in steps] + [knots[steps[-1][3]][1] if steps else r_yaw])
    pxy = np.stack([np.interp(t, kt, kxy[:, i]) for i in range(2)], 1)
    pyaw = np.interp(t, kt, np.unwrap(kyaw))
    sway = np.zeros((N, 2))
    for i, (s, _a, _b, _, sh) in enumerate(steps):
        u = (t - t_step[i]) / sh.step_t
        msk = (u >= 0) & (u <= 1)
        if sh.sway == 0.0 or not msk.any():
            continue
        st_xy = G._foot_at(steps, i, other[s], feet0)[:2]  # stance foot during this step
        mid = np.stack([np.interp(t[msk], kt, kxy[:, 0]), np.interp(t[msk], kt, kxy[:, 1])], 1)
        v = st_xy[None, :] - mid
        sway[msk] += sh.sway * np.sin(np.pi * u[msk])[:, None] * v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    pxy = pxy + sway
    sig = g.smooth_s * fps
    kr = int(3 * sig)
    if kr > 0:
        w = np.exp(-0.5 * (np.arange(-kr, kr + 1) / sig) ** 2)
        w /= w.sum()
        pad = lambda a: np.concatenate([np.repeat(a[:1], kr, 0), a, np.repeat(a[-1:], kr, 0)])  # noqa: E731
        pxy = np.stack([np.convolve(pad(pxy[:, i]), w, "valid") for i in range(2)], 1)
        pyaw = np.convolve(pad(pyaw), w, "valid")
    inout = np.where(t < lead, np.array([minjerk(min(1, v / lead)) for v in t]),
                     np.where(t < T_walk, 1.0, 1.0 - np.array([minjerk(min(1, max(0, (v - T_walk) / g.blend_s))) for v in t])))  # fmt: skip
    pz = zs + (walk_z - zs) * inout
    fpos = {s: np.tile(feet0[s][0], (N, 1)) for s in ("left", "right")}
    fyaw = {s: np.full(N, yaw_of(feet0[s][1])) for s in ("left", "right")}
    for i, (s, a, b, _, sh) in enumerate(steps):
        u = (t - t_step[i]) / sh.step_t
        su = np.clip((u - sh.ds / 2) / (1 - sh.ds), 0, 1)
        msk = u >= 0
        hor, lift = swing_profile(su[msk], sh)
        fpos[s][msk] = a[0][None] + hor[:, None] * (b[0] - a[0])[None]
        fpos[s][msk, 2] += sh.swing_h * lift
        fyaw[s][msk] = a[1] + hor * wrap(b[1] - a[1])
    out = np.empty((N, 36), np.float32)
    q = q0.copy()
    worst = 0.0
    for f in range(N):
        quat = quat_yaw(pyaw[f])
        p = np.r_[pxy[f], pz[f]]
        q, e = lm.solve(p, quat, q, {s: (fpos[s][f], rot_z(fyaw[s][f])) for s in ("left", "right")})
        worst = max(worst, e)
        out[f] = np.r_[p, quat, q]
    # end exactly in the start's stance (legs from IK at its height equal its legs up to the IK error; snap the hold)
    fin = np.r_[pxy[-1], zs, quat_yaw(pyaw[-1]), q0]
    nh = int(round(g.hold_end_s * fps))
    nb = int(round(0.3 * fps))
    for k in range(nb):
        f = N - nh - nb + k
        w = minjerk((k + 1) / nb)
        out[f, 7:] = (1 - w) * out[f, 7:] + w * q0
    out[N - nh :] = fin
    if info is not None:
        info.update(n_knots=n, n_steps=len(steps), walk_s=T_walk, total_s=T, ik_err_cm=100 * worst, first=first,
                    kinds=[knots[st[3]][2] for st in steps], goal_ref=[float(g_xy[0]), float(g_xy[1]), float(g_yaw)])  # fmt: skip
    return out


__all__ = ["HEFT", "HG_WALK", "HeftGait", "Shape", "plan_knots", "swing_profile", "walk_frames"]
