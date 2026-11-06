"""Two-hand skills for an object wider than a jaw: the hug pick (TODO 38, plan section 7; the box line's seed
`scripts/rsi/seeds/box_strict_start/bimanual.py`, ported into the library on 2026-11-05, user Q18) and the two-hand
put-in `place_into` (the pack line, 2026-11-05).

Hug pick: stand square to the support edge nearest the object, crouch, bring both open jaws beside the object's two
side walls (the walls whose normals run along the edge), press them together with `squeeze` and lift by arms and legs
at once, still pressing. Geometry (from the scene file and the object's measured pose, never from its name):
- the approach `v` is the box axis closest to the inward normal of the nearest approach edge; `u` is the robot's left
  when facing `v`; `w` the box's half width along `u`;
- the stance pelvis stands STANDOFF_M outside that edge on the line through the box centre along `v`, facing `v`;
- each wrist points along `v` with z up, so the jaw opens along `u` and the inner finger's outer flank (FLANK_M from the
  jaw axis) is what touches the wall: the jaw centre goes to the wall plane + FLANK_M + CLEAR_M, at the box centre's
  depth and GRASP_BELOW_RIM_M under the rim;
- the hands come in outward first, then forward PRE_OUT_M beside the walls, then inward onto them.
The jaws stay open (a closed jaw touches with its base, which `held` does not count). The lift (lift_margin_legs) asks for
LIFT_DZ_M with the pelvis up to LIFT_PELVIS_Z, holds LIFT_HOLD_S, and checks the box rose LIFT_MIN_RISE_M (held()'s
gap) with both hands on it; a miss re-arms the squeeze and lifts again (or re-grips when a hand came off).

Two-hand put-in (`place_into`, untested before the pack line's first run): the hug pick, then both hands raised with
the squeeze kept until the object's underside is PUT_CLEAR_M + PUT_OVER_M above the container's rim, a step back, a
walk on the skill's map (pick.walk_route, the carry turn rate) to a stance square to the container's edge with the
object over the opening (the carried offset ahead of the pelvis kept where it can be, the pelvis at least
EDGE_OUT_MIN_M outside the edge), the hands shifted so the object's box centre is over the opening's and lowered to
PUT_CLEAR_M above the rim, let go only when `into.release_fit` says the footprint is inside the inner box (one
correction), then rise, retract and step back. The object and the hands together do not fit through the opening, so
the release is above the rim and the object falls onto the container's floor. Done when in(X, C) holds and no finger
touches X. A failure after the lift keeps the object in both hands (there is no two-hand set-down yet)."""

from __future__ import annotations

import math

import numpy as np

from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene import navigate
from locomani.tasks.scene import stage as S

from . import into, place
from .pick import nav_map, walk_route

SIDES = ("left", "right")

# stance and crouch
STANDOFF_M = 0.38  # pelvis outside the support edge
HUG_PELVIS_Z = 0.72
STANCE_TOL_M = 0.05
STANCE_TOL_DEG = 8.0
STANCE_RETRY_STRAIGHT_M = 0.30  # a missed stance this close is re-walked straight, farther by a new path
POSTURE_TOL_M = 0.05
# hand placement
GRASP_BELOW_RIM_M = 0.10  # jaw-centre height under the box's rim
GRASP_DEPTH_M = 0.0  # jaw centres along the approach, from the box centre
FLANK_M = 0.0635  # the open jaw's finger flank from the jaw axis
CLEAR_M = 0.005  # gap to the wall before the squeeze
PRE_OUT_M = 0.06  # pre-grasp and via beside the wall, outward
PRE_BACK_M = 0.12  # pre-grasp short of the grasp along the approach
PRE_TOL_M = 0.08
MIN_TRACK = 0.12  # reach_check stall ratio
# the entry is judged by where each jaw centre is against its wall, not by its distance to the target: the squeeze
# closes a lateral gap, so a hand short along the approach or a little high still presses the wall
BESIDE_EDGE_M = 0.04  # jaw centre inside the wall's ends along the approach by this much
BESIDE_LOW_M = 0.06  # ... above the box's bottom
BESIDE_RIM_M = 0.03  # ... under its rim
BESIDE_IN_M = 0.02  # ... at most this far inside the target's wall-plane offset
BESIDE_OUT_M = 0.04  # ... at most this far outside the target
BOX_MOVED_MAX_M = 0.03  # the box knocked by a hand during the entry
# squeeze and lift
SQUEEZE_M = 0.06
SQUEEZE_S = 1.0
LIFT_DZ_M = 0.15
LIFT_PELVIS_Z = 0.79
LIFT_SPEED = 0.15  # m/s, the motor's reach speed
LIFT_HOLD_S = 1.0
LIFT_MIN_RISE_M = 0.02  # held()'s gap: a box already held is kept as it is (re-squeezing and lifting higher dropped it)
LIFT_RETRY_DZ_M = 0.10  # a retry lifts this far above where the hands are
# retries per step (the round's stage budget caps them all)
STANCE_RETRIES = 1
REACH_RETRIES = 1
SQUEEZE_RETRIES = 1
LIFT_RETRIES = 1
# the two-hand put-in
PUT_CLEAR_M = (
    0.06  # the object's underside this far above the rim at the release (the fingers reach under the jaw centre)
)
PUT_OVER_M = 0.03  # ... and this much higher while it is carried and brought over the opening
EDGE_OUT_MIN_M = 0.30  # the drop stance's pelvis at least this far outside the container's edge (the platform's height)
AHEAD_MAX_M = (
    0.73  # the object's centre at most this far ahead of the pelvis at the drop (the hug's farthest grasp line)
)
BACK_OFF_M = 0.30  # step back this far after the raise, before the walk
PUT_LOWER_SPEED = 0.06  # m/s of the last lowering
PUT_RISE_M = 0.08  # after the release both open jaws rise this far ...
PUT_RETRACT_M = 0.15  # ... back out this far along the approach ...
PUT_STEP_BACK_M = 0.25  # ... and the robot steps back this far
PUT_RETRIES = 1


def yaw_of(q) -> float:
    w, x, y, z = (float(c) for c in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


_edge_normals = {"+x": (-1.0, 0.0), "-x": (1.0, 0.0), "+y": (0.0, -1.0), "-y": (0.0, 1.0)}  # inward, surface frame


def two_handed(world, key: str) -> bool:
    """Whether the scene file says `key` takes two hands: `semantic.graspable` bimanual or a `bimanual` flag (features,
    never the name; rules.md 30)."""
    o = world.meta["objects"].get(key) or {}
    return (o.get("semantic") or {}).get("graspable") == "bimanual" or bool(o.get("bimanual"))


def nearest_edge(surface: dict, xy) -> tuple[str, np.ndarray, float]:
    """(edge name, inward unit normal in the world, distance of `xy` inside that edge) over the approach edges."""
    yaw = float(surface.get("yaw", 0.0))
    local = _rot2(-yaw) @ (np.asarray(xy, float)[:2] - np.asarray(surface["xy"], float))
    hx, hy = (float(h) for h in surface["half_extent"][:2])
    dist = {"+x": hx - local[0], "-x": hx + local[0], "+y": hy - local[1], "-y": hy + local[1]}
    edges = [e for e in (surface.get("approach_edges") or dist) if e in dist]
    e = min(edges, key=lambda k: dist[k])
    return e, _rot2(yaw) @ np.asarray(_edge_normals[e]), float(dist[e])


def geometry(world, key: str) -> dict:
    """The hug's frame and targets from the box's measured pose: approach `v`, left `u`, the stance, each hand's grasp
    target, the wrist rotation, the box's half width and depth along them."""
    meta = world.meta["objects"][key]
    pos, quat = world.object_pose(key)
    yaw = yaw_of(quat)
    half = np.asarray(meta["box"]["half_extent"], float)
    centre_off = np.asarray(meta["box"].get("centre", [0.0, 0.0, half[2]]), float)
    c = np.asarray(pos, float)[:2] + _rot2(yaw) @ centre_off[:2]
    surface = world.meta["surfaces"][meta["surface"]]
    edge, a, d = nearest_edge(surface, c)
    axes = [
        (_rot2(yaw) @ np.array([1.0, 0.0]), half[0], half[1]),
        (_rot2(yaw) @ np.array([0.0, 1.0]), half[1], half[0]),
    ]
    best = max(
        ((s * ax, depth, width) for ax, depth, width in axes for s in (1.0, -1.0)), key=lambda t: float(t[0] @ a)
    )
    v, depth_half, w = best
    u = np.array([-v[1], v[0]])
    d_v = d / max(float(v @ a), 0.5)  # box centre to the edge along v
    stance = c - v * (d_v + STANDOFF_M)
    bottom = float(pos[2]) + float(centre_off[2]) - float(half[2])
    rim = bottom + 2 * float(half[2])
    z = rim - GRASP_BELOW_RIM_M
    lateral = float(w) + FLANK_M + CLEAR_M
    target = {s: np.r_[c + sgn * u * lateral + v * GRASP_DEPTH_M, z] for s, sgn in (("left", 1.0), ("right", -1.0))}
    out = {s: np.r_[sgn * u * PRE_OUT_M, 0.0] for s, sgn in (("left", 1.0), ("right", -1.0))}
    rot = np.column_stack([np.r_[v, 0.0], np.r_[u, 0.0], [0.0, 0.0, 1.0]])
    return dict(edge=edge, depth=d, depth_v=d_v, v=v, u=u, w=float(w), depth_half=float(depth_half), centre=c, stance=stance,
                stance_yaw=math.atan2(v[1], v[0]), target=target, out=out, rot=rot, grasp_z=z, bottom=bottom, rim=rim)  # fmt: skip


def box_centre(world, key: str) -> np.ndarray:
    """The object's box centre in the world now (its origin plus the turned `box.centre`)."""
    pos, quat = world.object_pose(key)
    box = world.meta["objects"][key]["box"]
    return np.asarray(pos, float) + np.asarray(box.get("centre") or (0, 0, 0), float) @ _rot3(quat).T


def underside(world, key: str) -> float:
    """The object's lowest box corner (m) at its pose now."""
    pos, quat = world.object_pose(key)
    return float(pos[2]) - into.low_offset(quat, world.meta["objects"][key]["box"])


def _rot3(quat) -> np.ndarray:
    w, x, y, z = (float(c) for c in quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _pair(d: dict) -> tuple:
    return tuple(np.asarray(d[s], float) for s in SIDES)


def _reach_both(m, goal: dict, rot, via=(), **kw) -> Outcome:
    return m.reach("both", _pair(goal), None if rot is None else (rot, rot), via=[_pair(p) for p in via], **kw)


def _shift_both(m, delta, **kw) -> Outcome:
    """Both jaw centres moved by the world vector `delta`, wrists as they are, the squeeze kept."""
    goal = {s: m.jaw_centre(s)[0] + np.asarray(delta, float) for s in SIDES}
    return _reach_both(m, goal, None, keep_squeeze=True, **kw)


def _hands_check(m, goal: dict, tol: float, since: float | None = None):
    ok, values = True, {}
    for s in SIDES:
        ok_s, v = S.reach_check(m, s, goal[s], tol_m=tol, min_track=MIN_TRACK, since=since)
        ok = ok and ok_s
        values[f"{s}_jaw_err_cm"], values[f"{s}_track"] = v["jaw_err_cm"], v["track_ratio"]
    return ok, values


def _beside_check(m, g: dict):
    """Each jaw centre beside its wall, in the box frame `g` was computed in: along the approach within the wall,
    between the box's bottom and rim, laterally from BESIDE_IN_M inside to BESIDE_OUT_M outside the target's offset."""
    ok, values = True, {}
    v3, u3 = np.r_[g["v"], 0.0], np.r_[g["u"], 0.0]
    c3 = np.r_[g["centre"], g["bottom"]]
    lateral, rim = g["w"] + FLANK_M + CLEAR_M, g["rim"]
    for s, sgn in (("left", 1.0), ("right", -1.0)):
        p = m.jaw_centre(s)[0]
        along, out = float((p - c3) @ v3), sgn * float((p - c3) @ u3) - lateral
        ok_s = (abs(along) <= g["depth_half"] - BESIDE_EDGE_M and g["bottom"] + BESIDE_LOW_M <= p[2] <= rim - BESIDE_RIM_M
                and -BESIDE_IN_M <= out <= BESIDE_OUT_M)  # fmt: skip
        ok = ok and ok_s
        values[f"{s}_jaw_err_cm"] = round(100 * float(np.linalg.norm(p - g["target"][s])), 2)
        values[f"{s}_along_cm"], values[f"{s}_out_cm"] = round(100 * along, 2), round(100 * out, 2)
        values[f"{s}_dz_cm"] = round(100 * float(p[2] - g["grasp_z"]), 2)
    return ok, values


def pick(ctx, key: str) -> Outcome:
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    g = geometry(world, key)
    ctx.note("hug_plan", Outcome(True, t0=m.t, t1=m.t, info=dict(
        edge=g["edge"], depth_m=round(g["depth"], 3), half_width_m=round(g["w"], 3), grasp_z=round(g["grasp_z"], 3),
        grasp_line_m=round(g["depth_v"] + STANDOFF_M + GRASP_DEPTH_M, 3), stance=[round(float(x), 3) for x in g["stance"]],
        stance_yaw_deg=round(math.degrees(g["stance_yaw"]), 1))))  # fmt: skip

    # 1. stance: walk around the furniture to the edge, square to the box
    xy, yaw = g["stance"], g["stance_yaw"]

    def re_walk(k, v):
        if k >= STANCE_RETRIES:
            return None
        if float(v.get("stop_err_cm", 1e9)) <= 100 * STANCE_RETRY_STRAIGHT_M:  # close: a straight step, no path
            return lambda: m.walk_to(xy, yaw)
        return lambda: navigate.walk_path(m, world.occ, world.nav, xy, yaw)

    out = S.stage(
        ctx, "hug_stance",
        lambda: navigate.walk_path(m, world.occ, world.nav, xy, yaw),
        lambda o: S.stance_check(m, xy, yaw, STANCE_TOL_M, STANCE_TOL_DEG),
        re_walk,
    )  # fmt: skip
    if not out.ok:
        return out

    # 2. crouch
    def posture_check(o):
        z = float(m.pelvis()[0][2])
        return abs(z - HUG_PELVIS_Z) <= POSTURE_TOL_M, dict(pelvis_z=round(z, 3))

    out = S.stage(ctx, "hug_crouch", lambda: m.posture(pelvis_z=HUG_PELVIS_Z), posture_check,
                  lambda k, v: (lambda: m.posture(pelvis_z=HUG_PELVIS_Z)) if k < 1 else None)  # fmt: skip
    if not out.ok:
        return out

    # 3. pre-grasp: outward to the pre-grasp's side and height first, then forward
    g = geometry(world, key)
    pre = {s: g["target"][s] + g["out"][s] - np.r_[g["v"], 0.0] * PRE_BACK_M for s in SIDES}

    def to_pre():
        via1 = {}
        for s in SIDES:
            j = m.jaw_centre(s)[0]
            v3 = np.r_[g["v"], 0.0]
            via1[s] = pre[s] - v3 * float((pre[s] - j) @ v3)
        return _reach_both(m, pre, g["rot"], via=[via1])

    out = S.stage(ctx, "hug_pregrasp", to_pre, lambda o: _hands_check(m, pre, PRE_TOL_M),
                  lambda k, v: (lambda: _reach_both(m, pre, g["rot"])) if k < REACH_RETRIES else None)  # fmt: skip
    if not out.ok:
        return out

    # 4. entry: forward beside the walls, then inward onto them
    box0 = world.object_pose(key)[0].copy()
    state = dict(g=g)

    def entry():
        gg = state["g"]
        beside = {s: gg["target"][s] + gg["out"][s] for s in SIDES}
        return _reach_both(m, gg["target"], gg["rot"], via=[beside])

    def entry_check(o):
        ok, values = _beside_check(m, state["g"])
        moved = float(np.linalg.norm(world.object_pose(key)[0][:2] - box0[:2]))
        values["box_moved_cm"] = round(100 * moved, 2)
        return ok and moved <= BOX_MOVED_MAX_M, values

    def re_entry(k, v):
        if k >= REACH_RETRIES:
            return None

        def go():
            gg = state["g"]
            back = {s: m.jaw_centre(s)[0] + gg["out"][s] for s in SIDES}  # off the wall first
            _reach_both(m, back, gg["rot"])
            state["g"] = gg = geometry(world, key)  # on the box as it stands now
            box0[:] = world.object_pose(key)[0]
            beside = {s: gg["target"][s] + gg["out"][s] for s in SIDES}
            return _reach_both(m, gg["target"], gg["rot"], via=[beside])

        return go

    out = S.stage(ctx, "hug_entry", entry, entry_check, re_entry)
    if not out.ok:
        return out

    # 5. squeeze: both hands' fingers on the box
    out = S.stage(ctx, "hug_squeeze", lambda: m.squeeze(SQUEEZE_M, SQUEEZE_S),
                  lambda o: S.bimanual_grip_check(m, body),
                  lambda k, v: (lambda: m.squeeze(SQUEEZE_M, SQUEEZE_S)) if k < SQUEEZE_RETRIES else None)  # fmt: skip
    if not out.ok:
        return out

    # 6. lift by arms and legs, still pressing; hold, then check the rise with both hands on
    z0 = float(world.object_pose(key)[0][2])
    rot = state["g"]["rot"]

    def lift(goal: dict) -> Outcome:
        o = _reach_both(m, goal, rot, keep_squeeze=True, pelvis_z=LIFT_PELVIS_Z, speed=LIFT_SPEED)
        if m.fell_at is None:
            m.hold(LIFT_HOLD_S)
        o.t1 = m.t
        return o

    up = {s: state["g"]["target"][s] + np.array([0.0, 0.0, LIFT_DZ_M]) for s in SIDES}

    def re_lift(k, v):
        if k >= LIFT_RETRIES:
            return None
        if v.get("touching") == list(SIDES):  # still in both hands, too low: press again and lift higher

            def higher():
                m.squeeze(SQUEEZE_M, SQUEEZE_S)
                goal = {s: np.r_[up[s][:2], float(m.jaw_centre(s)[0][2]) + LIFT_RETRY_DZ_M] for s in SIDES}
                return lift(goal)

            return higher

        def regrip():  # a hand came off: back off the walls, onto them again on the box as it stands, press, lift
            back = {s: m.jaw_centre(s)[0] + state["g"]["out"][s] for s in SIDES}
            _reach_both(m, back, rot)
            gg = geometry(world, key)
            beside = {s: gg["target"][s] + gg["out"][s] for s in SIDES}
            _reach_both(m, gg["target"], gg["rot"], via=[beside], pelvis_z=HUG_PELVIS_Z)
            m.squeeze(SQUEEZE_M, SQUEEZE_S)
            return lift({s: gg["target"][s] + np.array([0.0, 0.0, LIFT_DZ_M]) for s in SIDES})

        return regrip

    return S.stage(ctx, "hug_lift", lambda: lift(up),
                   lambda o: S.bimanual_lift_check(world, m, key, z0, LIFT_MIN_RISE_M), re_lift)  # fmt: skip


# -- the two-hand put-in -----------------------------------------------------------------------------------------------


def drop_stance(world, container: str, ahead: float) -> dict:
    """Where to stand to hold a carried object over `container`'s opening: on the hug's approach line of the container
    (`geometry`: square to its nearest edge, facing in), the object's centre `ahead` of the pelvis as it is carried,
    clipped so the pelvis stays EDGE_OUT_MIN_M outside the edge and the centre at most AHEAD_MAX_M ahead. dict(xy, yaw,
    ahead (the clipped value), v, edge, depth)."""
    g = geometry(world, container)
    ahead_s = float(min(max(ahead, g["depth_v"] + EDGE_OUT_MIN_M), AHEAD_MAX_M))
    return dict(xy=g["centre"] - g["v"] * ahead_s, yaw=g["stance_yaw"], ahead=ahead_s, v=g["v"], edge=g["edge"],
                depth=g["depth_v"])  # fmt: skip


def _kept(m, why: str) -> Outcome:
    return Outcome(False, f"{why} [kept in both hands]", m.t, m.t)


def place_into(ctx, key: str, container: str) -> Outcome:
    """Hug-pick `key`, carry it to `container` and let it go over the opening (module doc)."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    out = pick(ctx, key)
    if not out.ok:
        return Outcome(False, f"hug pick failed: {out.why}", out.t0, out.t1, dict(out.info))

    # 1. raise: the underside clear of the rim, the squeeze kept
    cxy, rim, chalf, wall, cyaw = into.opening(world, container)
    carry_z = rim + PUT_CLEAR_M + PUT_OVER_M

    def raise_check(o):
        ok, values = S.bimanual_grip_check(m, body)
        under = underside(world, key)
        values["under_rim_cm"] = round(100 * (under - rim), 1)
        return ok and under >= rim + PUT_CLEAR_M, values

    def re_raise(k, v):
        if k >= PUT_RETRIES:
            return None

        def again():
            m.squeeze(SQUEEZE_M, SQUEEZE_S)
            return _shift_both(m, [0.0, 0.0, max(carry_z - underside(world, key), 0.0)], speed=LIFT_SPEED)

        return again

    dz = max(carry_z - underside(world, key), 0.0)
    out = S.stage(ctx, "put_raise", lambda: _shift_both(m, [0.0, 0.0, dz], speed=LIFT_SPEED), raise_check, re_raise)
    if not out.ok:
        return _kept(m, f"put_raise: {out.why}")

    # 2. back off the edge, then walk to the drop stance with the object where it rides
    out = ctx.note("put_back_off", m.sidestep(forward=-BACK_OFF_M))
    if m.fell_at is not None:
        return Outcome(False, f"put_back_off: {out.why}", out.t0, out.t1)
    heading = np.array([math.cos(m.yaw()), math.sin(m.yaw())])
    ahead = float((box_centre(world, key)[:2] - m.pelvis()[0][:2]) @ heading)
    spot = drop_stance(world, container, ahead)
    occ, nav, _ = nav_map(world)
    ctx.note("put_plan", Outcome(True, t0=m.t, t1=m.t, info=dict(
        edge=spot["edge"], depth_m=round(spot["depth"], 3), carried_ahead_m=round(ahead, 3),
        ahead_m=round(spot["ahead"], 3), stance=[round(float(x), 3) for x in spot["xy"]],
        stance_yaw_deg=round(math.degrees(spot["yaw"]), 1), rim=round(rim, 3))))  # fmt: skip

    def walk():
        with place._carry_turns(m):
            return walk_route(m, occ, nav, spot["xy"], spot["yaw"])

    def re_walk(k, v):
        if k >= STANCE_RETRIES:
            return None
        if float(v.get("stop_err_cm", 1e9)) <= 100 * STANCE_RETRY_STRAIGHT_M:
            return lambda: m.walk_to(spot["xy"], spot["yaw"])
        return walk

    out = S.stage(ctx, "put_stance", walk,
                  lambda o: S.stance_check(m, spot["xy"], spot["yaw"], STANCE_TOL_M, STANCE_TOL_DEG), re_walk)  # fmt: skip
    if not out.ok:
        return _kept(m, f"put_stance: {out.why}")
    if sorted(m.touching(body)) != list(SIDES):
        return Outcome(False, "dropped during the carry: a hand no longer touches the object", m.t, m.t)

    # 3. over the opening at the carry height, then down to the release height; let go only on release_fit
    cxy, rim, chalf, wall, cyaw = into.opening(world, container)  # as the container stands now
    shift = np.r_[cxy - box_centre(world, key)[:2], 0.0]
    for attempt in range(2):
        out = ctx.note("put_over", _shift_both(m, shift, speed=LIFT_SPEED), move_cm=np.round(100 * shift, 1),
                       attempt=attempt)  # fmt: skip
        if m.fell_at is not None:
            return Outcome(False, f"put_over: {out.why}", out.t0, out.t1)
        down = underside(world, key) - (rim + PUT_CLEAR_M)
        out = ctx.note(
            "put_lower", _shift_both(m, [0.0, 0.0, -down], speed=PUT_LOWER_SPEED), down_cm=round(100 * down, 1)
        )
        if m.fell_at is not None:
            return Outcome(False, f"put_lower: {out.why}", out.t0, out.t1)
        fit, why, move = into.release_fit(into.object_points(world, m, key), cxy, rim, chalf, wall, cyaw)
        ctx.note("release_fit", Outcome(fit, why, m.t, m.t), attempt=attempt, move_cm=np.round(100 * move[:2], 1),
                 under_rim_cm=round(100 * (underside(world, key) - rim), 1))  # fmt: skip
        if fit or sorted(m.touching(body)) != list(SIDES):
            break
        shift = np.r_[move[:2], PUT_OVER_M]  # once more: up a little, over by the measured error, down again
    if not fit:
        return _kept(m, f"no release over {container}: {why}")

    # 4. release both, rise, retract along the approach, step back
    r = ctx.note("open", m.release("both"))
    if m.fell_at is None:
        r = ctx.note("rise", _reach_both(m, {s: m.jaw_centre(s)[0] + [0.0, 0.0, PUT_RISE_M] for s in SIDES}, None))
    if m.fell_at is None:
        back = -PUT_RETRACT_M * np.r_[spot["v"], 0.0]
        r = ctx.note("retract", _reach_both(m, {s: m.jaw_centre(s)[0] + back for s in SIDES}, None))
    if m.fell_at is None:
        r = ctx.note("step_back", m.sidestep(forward=-PUT_STEP_BACK_M))
    if m.fell_at is not None:
        return Outcome(False, f"after the release: {r.why}", r.t0, r.t1)
    touching = sorted(m.touching(body))
    if touching:
        return Outcome(False, f"still touching after release: {'/'.join(touching)} hand", r.t0, r.t1)
    ok, why = world.check([f"in({key}, {container})"])(world.state([key, container]))
    return Outcome(ok, why, r.t0, r.t1)
