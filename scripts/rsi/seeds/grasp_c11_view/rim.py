"""held(X) for an open container with thin box walls: a rim pinch across one wall (grasp_atomic iteration 3, park C3).

Mug_2's collision body is a tub of 1 cm box walls, 9.0 x 11.6 cm outside. A level side grasp fits the 9.5 cm jaw only
within 2.6 deg of the tub's short axis, and the tub lies 14.6-18.4 deg off every table axis the stances approach along,
so pick.pick closed on its corners (0/3 diagnostic, park C3). `plan_for` sends an object here only when the scene file
marks it a `container` and the loaded model has RIM_MIN_WALLS thin upright box walls on its body (`walls`): Mug_2 only.
Every other object keeps pick.pick unchanged.

1. Stance (`rim_stance`): pick's stance candidates in pick's order, skipping those whose arm corridor (the strip from
   the pelvis to the object on the right arm's side, at the raised hand's height and above) or whose raise point (where
   the hanging hands turn) holds a static collider of another piece of furniture (`arm_obstacles`). Mug s410013: the
   tucked chair's back (z up to 0.914) stood 14 cm in front of the edge-normal stance; the turn there pushed the robot
   12.5 cm and the hand stayed trapped behind the back for the whole reach.
2. Raised-hand approach (support tops RAISE_TOP_MIN_M..RAISE_TOP_MAX_M): the walk stops RAISE_LEG_M before the stance
   (or at its last leg's start, if nearer), turns to the stance facing, raises the jaw to the top + the object's height
   + RAISE_CLEAR_M, RAISE_FWD_M ahead of and RAISE_LAT_M beside the pelvis, and walks the rest face-fixed: walk_to
   carries the reference's arm pose, so the hand stays over the objects. The hanging hand (jaw 3-4.5 cm above the
   0.746 m top) had pushed the mug 3.9 / 7.6 cm during the final turn (s410003 / s410023).
3. Rim pinch across the wall with the least twist (angle between the wall and the approach), among walls whose outer
   finger has RIM_ROOM_M to every other object's box on the support; ties within TWIST_TIE_DEG: the right-most. The
   closing axis is the wall's normal; wrist x is pitched RIM_PITCH_DEG nose-down along the wall, away from the body
   (RIM_PITCH_FAR_DEG when the object is more than RIM_FAR_M from the shoulder: the pitched jaw puts its 0.15 m offset
   downward, so the wrist itself must reach about 0.45 m out, and the arm then cannot also deliver 60 deg); the jaw
   centre on the wall's mid-length RIM_BELOW_M under the rim. Pre-grasp RIM_ABOVE_M straight above, descend at
   RIM_DESCENT_SPEED, then pick's grip, lift and carry-out.
All geometry comes from ctx.world: its scene description, `object_geoms` and `static_colliders` (local geom frames
composed with body poses: the static scene's geom_xpos stays zero under island sleeping)."""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene.navigate import path as nav_path
from locomani.tasks.scene.navigate import walk_path

from . import pick as P

# which objects: a box collision geom this thick or thinner, its thin axis horizontal, is a wall
RIM_WALL_MAX_M = 0.015
RIM_MIN_WALLS = 2
# stance: static colliders of other furniture the arm must not meet
ARM_LAT_RIGHT_M = 0.32  # the right arm's strip runs from the pelvis line this far to the right ...
ARM_LAT_LEFT_M = 0.0  # ... and this far to the left
ARM_ABOVE_TOP_M = 0.10  # colliders reaching above the support top + this block the raised arm
ARM_TOP_M = 0.45  # up to the support top + this
TURN_RADIUS_M = 0.35  # the hanging hands' sweep about the pelvis while it turns at the raise point
TURN_Z_M = (0.55, 1.00)  # the hanging arms' height band
# raised-hand approach
RAISE_TOP_MIN_M = 0.60  # the band of support tops the hanging hand rides onto
RAISE_TOP_MAX_M = 0.88
RAISE_LEG_M = 0.30  # the raised hand walks at most this far, face-fixed
RAISE_FWD_M = 0.20  # jaw centre ahead of the pelvis ...
RAISE_LAT_M = -0.22  # ... and beside it (right: negative)
RAISE_CLEAR_M = 0.05  # above the top + the object's height
# rim pinch
RIM_PITCH_DEG = 60.0  # wrist x below horizontal (iteration 1 scratch: 90 deg delivered 50-80 deg off, 60 deg 20-28)
RIM_FAR_M = 0.47  # shoulder to object beyond this: the arm cannot also pitch the jaw 60 deg down, ...
RIM_PITCH_FAR_DEG = 40.0  # ... so it pitches this much (iteration 3 scratch mugs: at 0.50-0.58 m the 60 deg pitch
# arrived 25-46 deg off, 2/6 held; 40 deg arrived 3-16 deg off, 4/5 held; 30 deg let the housing stop on the rim)
RIM_BELOW_M = 0.015  # jaw centre under the rim
RIM_ABOVE_M = 0.08  # pre-grasp above the grasp
RIM_DESCENT_SPEED = 0.04
RIM_ROOM_M = 0.06  # outer finger to the nearest other object's box
JAW_HALF_OPEN_M = 0.0475  # the open jaw's half width: where the outer finger sits
TWIST_TIE_DEG = 5.0  # walls this close to the least twist count as equally parallel


def _quat_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def walls(world, key: str) -> list[dict]:
    """The thin box walls of a free container as it lies now: centre (world), outward normal and along-wall direction
    (world xy, unit), half length, rim z. Empty for anything else, or with fewer than RIM_MIN_WALLS walls. Reads the
    collision shapes and the body pose only (no kinematics call), so a plan that does not take the branch is unchanged."""
    if not world.meta["objects"][key].get("container") or key not in world.free:
        return []
    p, q = world.object_pose(key)
    Rb = R.mat(q)
    out = []
    for geom in world.object_geoms(key):
        if geom["type"] != "box":
            continue
        g = geom["id"]
        half = np.asarray(geom["size"], float)
        Rg = Rb @ _quat_mat(geom["quat"])
        i = int(np.argmin(half))
        if 2 * half[i] > RIM_WALL_MAX_M or abs(Rg[2, i]) > 0.5:
            continue
        j = min((k for k in range(3) if k != i), key=lambda k: abs(Rg[2, k]))
        c = p + Rb @ np.asarray(geom["pos"], float)
        n = Rg[:2, i] / np.linalg.norm(Rg[:2, i])
        if n @ (c[:2] - p[:2]) < 0:
            n = -n
        t = Rg[:2, j] / np.linalg.norm(Rg[:2, j])
        out.append(dict(geom=g, centre=c, normal=n, along=t, half_len=float(half[j]),
                        rim=float(c[2] + np.abs(Rg[2]) @ half)))  # fmt: skip
    return out if len(out) >= RIM_MIN_WALLS else []


def arm_obstacles(world, p, top: float, radius: float = 2.0) -> np.ndarray:
    """World AABBs (n x 2 x 3: low, high) of the static scene's box colliders that the robot's arms collide with,
    within `radius` of `p` (xy) and reaching above the support top (the support's own slab and apron stay below it)."""
    boxes = []
    for c, e in world.static_colliders():
        if c[2] + e[2] <= top + 0.005 or np.linalg.norm(np.maximum(np.abs(c[:2] - p[:2]) - e[:2], 0.0)) > radius:
            continue
        boxes.append((c - e, c + e))
    return np.array(boxes, float).reshape(-1, 2, 3)


def arm_blocked(boxes, pel, u, p, top: float) -> int:
    """How many of `boxes` overlap the right arm's strip in front of a stance (pelvis `pel`, facing `u`): from the
    pelvis to the object's centre along `u`, ARM_LAT_RIGHT_M right to ARM_LAT_LEFT_M left, from the support top +
    ARM_ABOVE_TOP_M to + ARM_TOP_M. Boxes are bounded axis-aligned in the stance frame."""
    if not len(boxes):
        return 0
    lft = P._left(u)
    c, e = boxes.mean(1), (boxes[:, 1] - boxes[:, 0]) / 2
    rel = c[:, :2] - pel
    cf, cl = rel @ u, rel @ lft
    ef, el = e[:, :2] @ np.abs(u), e[:, :2] @ np.abs(lft)
    reach = float((p[:2] - pel) @ u)
    hit = (cf + ef > 0.0) & (cf - ef < reach) & (cl + el > -ARM_LAT_RIGHT_M) & (cl - el < ARM_LAT_LEFT_M)
    hit &= (c[:, 2] + e[:, 2] > top + ARM_ABOVE_TOP_M) & (c[:, 2] - e[:, 2] < top + ARM_TOP_M)
    return int(hit.sum())


def turn_blocked(boxes, xy) -> int:
    """How many of `boxes` come within TURN_RADIUS_M of `xy` inside the TURN_Z_M band: the hanging hands' sweep
    while the robot turns there."""
    if not len(boxes):
        return 0
    gap = np.linalg.norm(np.maximum(np.maximum(boxes[:, 0, :2] - xy, xy - boxes[:, 1, :2]), 0.0), axis=1)
    return int(((gap < TURN_RADIUS_M) & (boxes[:, 1, 2] > TURN_Z_M[0]) & (boxes[:, 0, 2] < TURN_Z_M[1])).sum())


def raise_point(corners):
    """Where the raised-hand approach turns and raises: RAISE_LEG_M before the stance on the last leg, or the last
    leg's start when that is nearer."""
    a, b = np.asarray(corners[-2], float), np.asarray(corners[-1], float)
    last = float(np.linalg.norm(b - a))
    return b + (a - b) * (RAISE_LEG_M / last) if last > RAISE_LEG_M else a


def rim_stance(world, key: str, start_xy, top: float, raised: bool):
    """pick's stance candidates in order: the first with a path whose arm strip and raise point are clear of
    `arm_obstacles`; the first with a path when none is. Returns (stance, corners, what to log) or (None, None, {})."""
    p, _, _ = P.object_frame(world, key)
    surf = world.meta["surfaces"][world.meta["objects"][key]["surface"]]
    boxes = arm_obstacles(world, p, top)
    first, skipped = None, []
    for c in P.stance_candidates(world, p[:2], surf, P.STANCE_MIN_D_M):
        corners = nav_path(world.occ, world.nav, start_xy, c[2])
        if corners is None:
            continue
        if first is None:
            first = (c, corners)
        arm = arm_blocked(boxes, c[2], c[3], p, top)
        turn = turn_blocked(boxes, raise_point(corners)) if raised else 0
        if arm or turn:
            skipped.append((c[4], round(c[1], 2), arm, turn))
            continue
        return c, corners, dict(obstacles=len(boxes), skipped=skipped[:6], n_skipped=len(skipped))
    if first is None:
        return None, None, {}
    return first[0], first[1], dict(obstacles=len(boxes), skipped=skipped[:6], n_skipped=len(skipped), all_blocked=True)


def room(world, key: str, point) -> tuple[float, str]:
    """Distance from `point` (world) to the nearest box of another object on the same support (free or static), and
    its key. Only the support's own objects: the scene's environment box holds every point (an earlier draft counted
    it: 0 cm from every wall)."""
    support = world.meta["objects"][key]["surface"]
    best, who = np.inf, ""
    for k, o in world.meta["objects"].items():
        if k == key or o.get("surface") != support or not o.get("box"):
            continue
        if k not in world.free and "init_pos" not in o:
            continue
        pk, qk = world.object_pose(k)
        Rk = R.mat(qk)
        half = np.asarray(o["box"]["half_extent"], float)
        c = pk + Rk @ np.asarray(o["box"].get("centre", (0.0, 0.0, 0.0)), float)
        loc = Rk.T @ (np.asarray(point, float) - c)
        dist = float(np.linalg.norm(loc - np.clip(loc, -half, half)))
        if dist < best:
            best, who = dist, k
    return best, who


def choose_wall(world, key: str, ws: list, p, a) -> dict:
    """The wall to pinch: among walls whose outer finger has RIM_ROOM_M (all walls if none has), those within
    TWIST_TIE_DEG of the least twist (angle between the wall and the approach `a`); of these the right-most (the arm's
    own side), or, when no wall has room, the one with the most."""
    rows = []
    for wl in ws:
        outer = np.r_[wl["centre"][:2] + JAW_HALF_OPEN_M * wl["normal"], wl["rim"] - RIM_BELOW_M]
        r, who = room(world, key, outer)
        twist = float(np.degrees(np.arccos(np.clip(abs(wl["along"] @ a), 0.0, 1.0))))
        lat = float((wl["centre"][:2] - p[:2]) @ P._left(a))
        rows.append(dict(wall=wl, room=r, nearest=who, twist=twist, lat=lat))
    roomy = [r for r in rows if r["room"] >= RIM_ROOM_M]
    pool = roomy or rows
    tmin = min(r["twist"] for r in pool)
    pool = [r for r in pool if r["twist"] <= tmin + TWIST_TIE_DEG]
    best = min(pool, key=lambda r: r["lat"]) if roomy else max(pool, key=lambda r: r["room"])
    alts = [(round(r["twist"], 1), round(100 * r["lat"], 1), round(100 * min(r["room"], 9.99), 1), r["nearest"])
            for r in rows]  # fmt: skip
    return dict(best, alternatives=alts)


def rim_rotation(along, a, pitch_deg: float):
    """Wrist rotation: x pitched `pitch_deg` below horizontal along the wall, pointing away from the body (along `a`);
    y (the closing axis) the wall's normal; z up and forward."""
    t = along if along @ a >= 0 else -along
    th = np.radians(pitch_deg)
    x = np.r_[np.cos(th) * t, -np.sin(th)]
    z = np.r_[np.sin(th) * t, np.cos(th)]
    return np.stack([x, np.cross(z, x), z], axis=1)


def level_rotation(f):
    """Wrist x level along `f` (world xy), z up."""
    return np.stack([np.r_[f, 0.0], np.r_[P._left(f), 0.0], np.array([0.0, 0.0, 1.0])], axis=1)


def walk_raised(ctx, corners, yaw: float, top: float, height: float) -> Outcome:
    """`corners`' legs up to `raise_point`, a turn to `yaw`, the jaw raised over the support, then the rest
    face-fixed."""
    m = ctx.motor
    b = np.asarray(corners[-1], float)
    r = raise_point(corners)
    legs = [np.asarray(c, float) for c in corners[1:-1]]
    if np.linalg.norm(r - np.asarray(corners[-2], float)) > 1e-6:
        legs.append(r)
    for c in legs:
        out = m.walk_to(c, None)
        if not out.ok or m.fell_at is not None:
            return out
    out = ctx.note("turn", m.turn(yaw))
    if not out.ok or m.fell_at is not None:
        return out
    pel_now, _ = m.pelvis()
    f = np.array([np.cos(m.yaw()), np.sin(m.yaw())])
    up = np.r_[pel_now[:2] + RAISE_FWD_M * f + RAISE_LAT_M * P._left(f), top + height + RAISE_CLEAR_M]
    jaw0, _ = m.jaw_centre(P.SIDE)
    out = ctx.note(
        "raise", m.reach(P.SIDE, up, level_rotation(f)), raise_to=np.round(up, 4), jaw_from=np.round(jaw0, 4)
    )
    if m.fell_at is not None:
        return out
    out = m.walk_to(b, yaw, face="fixed")
    out.info["legs"] = len(legs) + 1
    out.info["raised_leg_m"] = round(float(np.linalg.norm(b - r)), 3)
    return out


def finger_offsets(ctx, wl) -> list:
    """Each right finger body's offset from the wall (cm): across (+ outside), along, below the rim."""
    m = ctx.motor
    out = []
    for q in m.finger_positions(P.SIDE):
        d = q[:2] - wl["centre"][:2]
        out.append([round(100 * float(d @ wl["normal"]), 2), round(100 * float(d @ wl["along"]), 2),
                    round(100 * (wl["rim"] - float(q[2])), 2)])  # fmt: skip
    return out


def pick(ctx, key: str) -> Outcome:
    """`rim_stance`, the raised-hand walk and the rim pinch."""
    w, m = ctx.world, ctx.motor
    p, Ro, half = P.object_frame(w, key)
    surf = w.meta["surfaces"][w.meta["objects"][key]["surface"]]
    top = float(surf["top"])
    raised = RAISE_TOP_MIN_M <= top <= RAISE_TOP_MAX_M
    st, corners, sinfo = rim_stance(w, key, m.pelvis()[0][:2], top, raised)
    if st is None:
        return ctx.note("stance", Outcome(False, "no stance with a path", m.t, m.t))
    _, s, pel, u, kind, ang = st
    height = 2 * P.resting_half_height(Ro, half)
    P._note(ctx, "stance", kind=kind, shoulder_to_obj=s, normal_angle_deg=ang, pelvis=pel,
            clear=P.clearance(w.occ, pel), rim=True, raised=raised, height=height, **sinfo)  # fmt: skip
    yaw = float(np.arctan2(u[1], u[0]))
    p0 = p.copy()
    out = walk_raised(ctx, corners, yaw, top, height) if raised else walk_path(m, w.occ, w.nav, pel, yaw)
    p, Ro, half = P.object_frame(w, key)
    pel_now, _ = m.pelvis()
    yaw_m = m.yaw()
    f = np.array([np.cos(yaw_m), np.sin(yaw_m)])
    jaw0, _ = m.jaw_centre(P.SIDE)
    rel = jaw0[:2] - pel_now[:2]
    ctx.note("walk", out, obj_moved_cm=round(100 * float(np.linalg.norm(p[:2] - p0[:2])), 2),
             jaw_fwd=round(float(rel @ f), 3), jaw_lat=round(float(rel @ P._left(f)), 3),
             jaw_z_minus_top=round(float(jaw0[2]) - top, 4))  # fmt: skip
    if not out.ok or m.fell_at is not None:
        return out
    shoulder = pel_now[:2] - P.SHOULDER_Y_M * P._left(f)
    a = p[:2] - shoulder
    a = a / np.linalg.norm(a)
    ws = walls(w, key)
    if not ws:
        return ctx.note("rim", Outcome(False, "no upright thin wall after the walk", m.t, m.t))
    ch = choose_wall(w, key, ws, p, a)
    wl = ch["wall"]
    reach = float(np.linalg.norm(p[:2] - shoulder))
    pitch = RIM_PITCH_DEG if reach <= RIM_FAR_M else RIM_PITCH_FAR_DEG
    Rg = rim_rotation(wl["along"], a, pitch)
    g = np.r_[wl["centre"][:2], wl["rim"] - RIM_BELOW_M]
    pre = g + np.array([0.0, 0.0, RIM_ABOVE_M])
    P._note(ctx, "rim", walls=len(ws), wall_len_cm=200 * wl["half_len"], twist_deg=ch["twist"],
            room_cm=100 * min(ch["room"], 9.99), nearest=ch["nearest"], wall_lat_cm=100 * ch["lat"],
            alternatives=ch["alternatives"], rim_z=wl["rim"], grasp=g, shoulder_to_obj=reach, pitch_deg=pitch,
            approach_deg=float(np.degrees(np.arctan2(a[1], a[0]))))  # fmt: skip
    via = [] if raised else [P.first_via(jaw0, np.r_[a, 0.0], top, pre)]
    out = ctx.note("pregrasp", m.reach(P.SIDE, pre, Rg, via=via))
    if m.fell_at is not None:
        return out
    out = ctx.note("descend", m.reach(P.SIDE, g, Rg, speed=RIM_DESCENT_SPEED))
    if m.fell_at is not None:
        return out
    jaw, _ = m.jaw_centre(P.SIDE)
    off = jaw - g
    P._note(ctx, "close", across_cm=100 * float(off[:2] @ wl["normal"]), along_cm=100 * float(off[:2] @ wl["along"]),
            below_rim_cm=100 * (wl["rim"] - float(jaw[2])), fingers_cm=finger_offsets(ctx, wl))  # fmt: skip
    out = P.grip_and_lift(ctx, Rg)
    now = next((x for x in walls(w, key) if x["geom"] == wl["geom"]), wl)  # the wall where the mug is now
    P._note(ctx, "fingers", at="lift" if out.ok else "grip", fingers_cm=finger_offsets(ctx, now),
            opening_cm=100 * m.jaw_opening(P.SIDE))  # fmt: skip
    if not out.ok or m.fell_at is not None:
        return out
    P.carry_out(ctx, key, np.r_[a, 0.0], out.info["lift_to"], Rg)
    if m.fell_at is not None:
        return Outcome(False, f"fell at {m.fell_at:.2f} s", m.t, m.t)
    held = bool(w.held(key))
    return Outcome(held, "" if held else "not between the fingers after the lift", m.t, m.t)
