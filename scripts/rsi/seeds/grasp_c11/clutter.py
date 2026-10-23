"""Low fixed clutter under the forearm of a side grasp on a high support (grasp_atomic iteration 8, draft 1, H1).

Cause (orange s410000, iteration 7 FK on the logged joints): on the 0.895 m kitchen island the level side grasp puts
the jaw 3.9 cm above the top; the wrist and forearm hang about 4.5 cm below the jaw axis from 0.10 m back (r1_in_single:
pick.WRIST_LOW_POINT (-0.15, -0.045)), so they rest on a low static tray in the arm's line (CandleDishMedium, 2.7 cm
tall). The jaw then sits 2.5 cm high, the command leads it 6-8 cm down, and the pads close above the equator on nothing
that lifts. Raising the jaw (H1 probe, JAW_ABOVE_TOP_M 0.046) flipped nothing and broke two held starts.

`pitch` is the gate and the angle, run after the walk on the measured approach `a` (shoulder to object):
- trigger: a side grasp (pick.side_grasp: not flat, not a rim pinch) of an object resting less than HEIGHT_MAX_M tall
  on a support whose top is at least SUPPORT_MIN_M;
- strip test (`blockers`): static objects of the scene file on the same support (their scene-file box; free bodies only
  with FREE) whose box reaches within STRIP_HALF_M of the arm's line, STRIP_NEAR_M..STRIP_FAR_M behind the jaw (an
  axis-aligned bound in the approach frame, as pick.hand_corridor), and whose top rises 0 < excess <= LOW_MAX_M above
  the arm's underside (jaw z - ARM_BELOW_M). The narrow strip is iteration 7 draft 2's: at +-4.5 cm (and with free
  bodies) it flags held s410010 (a dish 2.1 cm beside the line) and plant s410008 (a free plant the hand shoves aside);
- angle: theta = min(PITCH_MAX_DEG, max over blockers of atan((excess + PITCH_MARGIN_M) / near-edge distance)), the
  near edge being where the blocker's box begins behind the jaw (at least STRIP_NEAR_M). A nose-down pitch about the
  jaw centre raises a point L behind the jaw by L sin(theta), with no jaw rise.
pick.side_grasp then turns the grasp attitude by rot_y(theta) about the wrist y, keeps the incumbent's grasp point,
and puts the pre-grasp and the entry along the pitched wrist x. With no blocker (theta 0) every value is the
incumbent's.

`grip_lift` (pitched rounds only, REGRIP): after the grip, an opening under REGRIP_FRAC of the object's hull width at
the jaw height along the closing axis (`hull_width`, the collision mesh's vertices in a +-HULL_BAND_M band) is a pinch
above the equator: release, lower the jaw REGRIP_DOWN_M below the lower of the commanded entry and the measured jaw,
grip once more, then pick's lift. Off by default (iteration 8 Composer): it did not fire on orange s410000, the one
start the pitch fixed (grip opening 8.1 cm), so no trace shows it working; with REGRIP 0 a pitched round grips and
lifts as pick.grip_and_lift."""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.backends.scalebfm.motor import WRIST

from . import pick as P

CLUTTER = 1  # 0: the whole branch off (every round is the incumbent's)
PITCH = 1  # 0: blockers are logged, the grasp stays level
REGRIP = 0  # 1: one lowered re-grip on a pinch above the equator (pitched rounds only; untested, see above)
FREE = 0  # 1: free bodies count as blockers too (live poses)
# trigger
SUPPORT_MIN_M = 0.80
HEIGHT_MAX_M = 0.18  # resting height of the object
# strip
STRIP_HALF_M = 0.015
STRIP_NEAR_M = 0.10  # behind the jaw centre
STRIP_FAR_M = 0.45
ARM_BELOW_M = 0.045  # the wrist and forearm's underside below the jaw axis (r1_in_single)
LOW_MAX_M = 0.06  # taller statics are not in the forearm's band (vase +17.8, bowl +14.2 cm on the island)
# angle
PITCH_MARGIN_M = 0.01
PITCH_MAX_DEG = 25.0
# re-grip
REGRIP_FRAC = 0.85
REGRIP_DOWN_M = 0.015
HULL_BAND_M = 0.01


def triggered(world, key: str, top: float) -> bool:
    _, Ro, half = P.object_frame(world, key)
    return bool(CLUTTER) and top >= SUPPORT_MIN_M and 2 * P.resting_half_height(Ro, half) < HEIGHT_MAX_M


def _quat_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def blockers(world, key: str, a, g) -> list[tuple]:
    """(object, excess above the arm's underside (m), near-edge distance behind the jaw (m)) of every object on `key`'s
    support whose box lies in the arm's strip along `a` behind the jaw at `g`, with 0 < excess <= LOW_MAX_M."""
    support = world.meta["objects"][key]["surface"]
    ax = np.stack([np.r_[a, 0.0], np.r_[P._left(a), 0.0]], axis=1)  # along, lateral
    under = float(g[2]) - ARM_BELOW_M
    out = []
    for k, o in world.meta["objects"].items():
        if k == key or "box" not in o:
            continue
        if k in world.free:
            if not FREE:
                continue
            q, quat = world.object_pose(k)
        elif o.get("surface") == support:
            q, quat = world.object_pose(k)
        else:
            continue
        Rq = _quat_mat(quat)
        hq = np.asarray(o["box"]["half_extent"], float)
        c3 = np.asarray(q, float) + Rq @ np.asarray(o["box"].get("centre", (0.0, 0.0, 0.0)), float)
        c, e = ax.T @ (c3 - g), np.abs(ax.T @ Rq) @ hq
        if c[0] - e[0] > -STRIP_NEAR_M or c[0] + e[0] < -STRIP_FAR_M or abs(c[1]) - e[1] > STRIP_HALF_M:
            continue
        excess = float(c3[2] + np.abs(Rq[2]) @ hq) - under
        if 0.0 < excess <= LOW_MAX_M:
            out.append((k, excess, max(STRIP_NEAR_M, -float(c[0] + e[0]))))
    return out


def pitch(ctx, key: str, top: float, a, g) -> float:
    """The nose-down grasp pitch (rad) for the side grasp at `g` along `a`; 0 when the branch does not apply."""
    w = ctx.world
    if not triggered(w, key, top):
        return 0.0
    found = blockers(w, key, a, g)
    if not found:
        return 0.0
    th = min(np.radians(PITCH_MAX_DEG), max(float(np.arctan((ex + PITCH_MARGIN_M) / d)) for _, ex, d in found))
    P._note(ctx, "clutter", theta_deg=float(np.degrees(th)) if PITCH else 0.0,
            blockers=[[k, round(100 * ex, 2), round(100 * d, 2)] for k, ex, d in found])  # fmt: skip
    return th if PITCH else 0.0


def hull_width(world, key: str, z: float, axis) -> float:
    """Width (m) along the horizontal unit `axis` of `key`'s collision mesh vertices within HULL_BAND_M of height `z`,
    from the body's pose now (the box's width when no vertex lies in the band)."""
    m, d = world.sim.m, world.sim.d
    b = world.free[key]
    pb, Rb = d.xpos[b].copy(), _quat_mat(d.xquat[b])
    ax3 = np.r_[axis, 0.0]
    vals = []
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] != b or not (m.geom_contype[gi] or m.geom_conaffinity[gi]):
            continue
        if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        adr, n = m.mesh_vertadr[m.geom_dataid[gi]], m.mesh_vertnum[m.geom_dataid[gi]]
        v = pb + (m.geom_pos[gi] + m.mesh_vert[adr : adr + n] @ _quat_mat(m.geom_quat[gi]).T) @ Rb.T
        sel = np.abs(v[:, 2] - z) <= HULL_BAND_M
        vals.extend((v[sel] @ ax3).tolist())
    if len(vals) >= 2:
        return float(max(vals) - min(vals))
    _, Ro, half = P.object_frame(world, key)
    return float(2 * np.abs(Ro.T @ ax3) @ half)


def grip_lift(ctx, key: str, Rg, entry) -> object:
    """pick.grip_and_lift for a pitched grasp, with one lowered re-grip on a pinch above the equator."""
    w, m = ctx.world, ctx.motor
    jaw, _ = m.jaw_centre(P.SIDE)
    po, _ = w.object_pose(key)
    wrist, _ = m.sim.body(WRIST[P.SIDE])
    P._note(ctx, "pitched_at_grip", cmd_minus_meas_z=float(m.commanded_jaw(P.SIDE)[2] - jaw[2]),
            jaw_minus_centre_z=float(jaw[2] - po[2]), wrist_z=float(wrist[2]))  # fmt: skip
    if not REGRIP:
        return P.grip_and_lift(ctx, Rg)
    out = ctx.note("grip", m.grip(P.SIDE))
    if m.fell_at is not None:
        return out
    jaw, _ = m.jaw_centre(P.SIDE)
    width = hull_width(w, key, float(jaw[2]), Rg[:2, 1] / max(np.linalg.norm(Rg[:2, 1]), 1e-9))
    opening = m.jaw_opening(P.SIDE)
    if opening < REGRIP_FRAC * width:
        P._note(ctx, "regrip", opening_cm=100 * opening, hull_width_cm=100 * width)
        ctx.note("release", m.release(P.SIDE))
        if m.fell_at is not None:
            return out
        low = np.r_[entry[:2], min(float(entry[2]), float(jaw[2])) - REGRIP_DOWN_M]
        ctx.note("lower", m.reach(P.SIDE, low, Rg, speed=P.ENTRY_SPEED))
        if m.fell_at is not None:
            return out
        out = ctx.note("grip", m.grip(P.SIDE))
    if not out.ok or m.fell_at is not None:
        return out
    lift = m.jaw_centre(P.SIDE)[0] + np.array([0.0, 0.0, P.LIFT_M])
    out = ctx.note("lift", m.reach(P.SIDE, lift, Rg, speed=P.LIFT_SPEED))
    out.ok = m.fell_at is None
    out.info["lift_to"] = lift
    return out
