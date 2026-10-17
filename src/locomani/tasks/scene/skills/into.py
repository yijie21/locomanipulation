"""Skill `place_into` (r1_in_single loop, iteration 0; the family's gap place_into.single, rules 24): put a free object
into a container C. place.place_on's pick (its switches and SIMPLE overrides), its plan before the pick with the creep
refusal, its hand-off and carry; the spot is C's opening centre (place.plan_spot would find C itself resting there).

Stances (`stances`): C on a surface -- per edge of that surface with C's opening centre at most MAX_DEPTH_M in, the
opening pick.STAND_AHEAD_M ahead and pick.STAND_SIDE_M toward the hand, or up to DEEP_MAX_M in with the pelvis EDGE_OUT_M
outside the edge; listed edges first, pick's floor-object and other-edge clearances. C on the floor (no surface) --
FLOOR_FACES bearings with the opening FLOOR_AHEAD_M ahead. The first with a route wins; RING (off) then tries stances on
a ring around C, facing it.

Near drop (`near_spot`): C on the object's own support with a stance at the edge the pick stands at, within NEAR_M of
the pick stance, is reached by one walk_to along that edge keeping the heading (no carry turns), no back-off and no
route, and T_est is that path's (`near_seconds`). The Reviewer's apple s130009: lift to release about 21 s; the back-off and route plan was 30.3 s,
past the apple's 22.75 s creep budget. Its pick lifts the object (grasp_program.LIFT_M while it runs) until the
underside clears by NEAR_CLEAR_M every object resting on the support whose footprint the carried path crosses
(`path_top`: the jaw from the pick to the drop stance, widened by the object's half width + NEAR_BAND_M; `near_lift`),
at least LIFT_M; a near drop needing more than NEAR_LIFT_MAX_M is not taken. The corner apples (iteration 2): at the
0.10 m lift the hand swept the static Cracker_Anon (top 0.949 m) on the walk along the -y edge; 0.22 m cleared it.
Routed carries keep LIFT_M.

Release (`release_fit`): the object is reached over the opening through a point VIA_OUT_M outside the edge and lowered
to its underside DROP_CLEAR_M above the rim. The hand opens only when the object's collision footprint lies
RELEASE_MARGIN_M inside C's inner box (the `in` predicate's: the half extent less the wall) and its underside is above
the rim; otherwise it is moved once by the measured error and lowered again. Still outside, the hand stays shut: the
object is set down on a surface under it (place.set_down), else raised back to RAISE_BELOW_START_M below its start
height (scene_round counts an object 0.15 m below its start as a drop, and C on the floor has no surface under it).
Every other failure after the lift keeps place's set_down. Done when in(X, C) holds and no finger touches X."""

from __future__ import annotations

import dataclasses

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.scene_discovery.scene_file import to_surface_frame
from locomani.tasks.scene.start import overlap

from . import grasp_program, pick, place

DROP_CLEAR_M = 0.03  # object underside this far above the rim at the release
ABOVE_M = 0.06  # the reach over the opening ends this much higher, through a point VIA_OUT_M outside the edge
VIA_OUT_M = 0.10
RISE_M = 0.08  # after the release the open jaw rises this far, then backs out RETRACT_M level
RETRACT_M = 0.12
MAX_DEPTH_M = 0.40  # opening centre at most this far in from the edge the robot stands at ...
DEEP_MAX_M = 0.7  # (0.55 until r1_in_single's set) ... or up to this, the pelvis EDGE_OUT_M outside the edge (<= MAX_DEPTH_M: off)
EDGE_OUT_M = 0.15  # (0.05 until r1_in_single's set)
FLOOR_FACES = 16  # a container on the floor: stances on this many bearings ...
FLOOR_AHEAD_M = 0.42  # ... the opening centre this far ahead of the pelvis
CREEP = True  # place's creep refusal before the pick and grip-margin check after it (False: both off)
NEAR_M = 1.6  # (0.6 until r1_in_single's set) > 0: a drop stance at the pick's edge within this of the pick stance is reached by one walk_to ...
NEAR_YAW_DEG = 30.0  # ... when the pick ends in the planned hand, facing within this of that edge
RELEASE_MARGIN_M = 0.01  # the hand opens only with the object's footprint this far inside the inner box ...
RAISE_BELOW_START_M = 0.05  # ... else, over no surface, the object is raised back to this below its start height
NEAR_CLEAR_M = 0.03  # near drops: the carried underside this far above what rests on the support along its path ...
NEAR_BAND_M = 0.05  # ... the path widened by the object's larger half width + this
NEAR_LIFT_MAX_M = 0.30  # a near drop whose pick would lift higher than this is not taken
RING = False  # no edge-normal stance with a route: stances on a ring around the container, facing it
RING_AHEAD_M = (0.40, 0.45, 0.50)
RING_OUTSIDE_M = 0.10  # pelvis at least this far outside the support's footprint ...
RING_OTHER_M = 0.15  # ... and this clear of other furniture

_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def opening(world, c: str):
    """(xy of C's opening centre, rim z, box half extent, wall, yaw), by C's pose now (the `in` predicate's box)."""
    o = world.meta["objects"][c]
    pos, quat = world.object_pose(c)
    box = o["box"]
    centre = np.asarray(box.get("centre") or (0, 0, 0), float)
    half = np.asarray(box["half_extent"], float)
    return pos[:2] + centre[:2], float(pos[2] + centre[2] + half[2]), half, float(box.get("wall", 0.01)), R.yaw_of(quat)


def low_offset(quat, box: dict) -> float:
    """Depth (m) of the object's lowest box corner below its origin at `quat`."""
    corners = np.asarray(box.get("centre") or (0, 0, 0), float) + _CORNERS * np.asarray(box["half_extent"], float)
    return float(-(corners @ R.mat(quat).T)[:, 2].min())


def object_points(world, m, key: str) -> np.ndarray:
    """The object's collision vertices in the world now (pick.object_vertices), or its box corners without a mesh."""
    v = pick.object_vertices(m, world.free[key])
    if len(v):
        return v
    pos, quat = world.object_pose(key)
    box = world.meta["objects"][key]["box"]
    corners = np.asarray(box.get("centre") or (0, 0, 0), float) + _CORNERS * np.asarray(box["half_extent"], float)
    return pos + corners @ R.mat(quat).T


def release_fit(points, cxy, rim: float, half, wall: float, yaw: float) -> tuple[bool, str, np.ndarray]:
    """May the object, by `points` (world), be let go over the opening (`opening`'s cxy, rim, half, wall, yaw)? Its
    footprint in C's frame RELEASE_MARGIN_M inside the inner box and its lowest point above the rim. (ok, why, the world
    move putting the footprint's centre over the opening's and the underside DROP_CLEAR_M above the rim)."""
    p = np.asarray(points, float)
    rc = R.rot_z(yaw)[:2, :2]
    local = (p[:, :2] - np.asarray(cxy, float)) @ rc  # rows: C's frame
    lo, hi = local.min(axis=0), local.max(axis=0)
    past = float(np.max(np.maximum(-lo, hi) - (np.asarray(half, float)[:2] - wall - RELEASE_MARGIN_M)))
    under = float(p[:, 2].min())
    move = np.r_[-rc @ (0.5 * (lo + hi)), rim + DROP_CLEAR_M - under]
    why = []
    if past > 0:
        why.append(f"footprint {100 * past:.1f} cm past the inner box less {100 * RELEASE_MARGIN_M:.1f} cm")
    if under < rim:
        why.append(f"underside {100 * (rim - under):.1f} cm below the rim")
    return not why, "; ".join(why), move


def stances(world, container: str, side: str, start_xy) -> list[dict]:
    """Drop stances at `container` (see the module doc), best first by listed edge, depth, distance from `start_xy`.
    Each dict(xy (the opening centre), face, stance, edge_xy (where the reach crosses the support's edge or C's side),
    edge, depth)."""
    meta = world.meta
    cxy, _, chalf, _, cyaw = opening(world, container)
    sk = meta["objects"][container].get("surface")
    _, _, boxes = pick.nav_map(world)
    sign = -1.0 if side == "right" else 1.0
    cands = []
    if sk:
        s = meta["surfaces"][sk]
        rs = R.rot_z(float(s.get("yaw", 0.0)))[:2, :2]
        half = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)
        local = rs.T @ (cxy - np.asarray(s["xy"], float))
        listed = list(s.get("approach_edges") or pick.NORMALS)
        boxed = not s.get("approach_edges")
        for e, n in pick.NORMALS.items():
            n = np.asarray(n, float)
            depth = float(half[0 if "x" in e else 1] - local @ n)
            if depth > max(MAX_DEPTH_M, DEEP_MAX_M):
                continue
            w = rs @ n
            face = float(np.arctan2(-w[1], -w[0]))
            ahead = pick.STAND_AHEAD_M if depth <= MAX_DEPTH_M else depth + EDGE_OUT_M
            stance = cxy - (R.rot_z(face) @ np.array([ahead, sign * pick.STAND_SIDE_M, 0.0]))[:2]
            if pick.floor_clearance(boxes, stance) < pick.STANCE_CLEAR_M:
                continue
            if (e not in listed or boxed) and pick.other_clearance(world, sk, stance) < pick.OTHER_EDGE_CLEAR_M:
                continue
            cands.append((0 if e in listed else 1, depth, float(np.linalg.norm(stance - start_xy)),
                          dict(xy=cxy, face=face, stance=stance, edge_xy=cxy + w * depth, edge=e, depth=depth)))  # fmt: skip
    else:
        rc = R.rot_z(cyaw)[:2, :2]
        for k in range(FLOOR_FACES):
            face = 2 * np.pi * k / FLOOR_FACES
            u = np.array([np.cos(face), np.sin(face)])
            stance = cxy - (R.rot_z(face) @ np.array([FLOOR_AHEAD_M, sign * pick.STAND_SIDE_M, 0.0]))[:2]
            if pick.floor_clearance(boxes, stance) < pick.STANCE_CLEAR_M:
                continue
            reach = float(abs(u @ rc[:, 0]) * chalf[0] + abs(u @ rc[:, 1]) * chalf[1])  # C's side along u
            cands.append((0, 0.0, float(np.linalg.norm(stance - start_xy)),
                          dict(xy=cxy, face=face, stance=stance, edge_xy=cxy - u * reach, edge=f"floor{k}", depth=reach)))  # fmt: skip
    cands.sort(key=lambda z: z[:3])
    return [cd[3] for cd in cands]


def ring_stances(world, container: str, side: str, start_xy) -> list[dict]:
    """RING: stances RING_AHEAD_M from C's opening on bearings 10 deg apart, facing it, the pelvis RING_OUTSIDE_M
    outside its support's footprint, STANCE_CLEAR_M from floor objects and RING_OTHER_M from other furniture; nearest
    ring first, then by distance from `start_xy`. edge_xy: where the line to C leaves the support."""
    meta = world.meta
    cxy, _, _, _, _ = opening(world, container)
    sk = meta["objects"][container]["surface"]
    s = meta["surfaces"][sk]
    half = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)
    _, _, boxes = pick.nav_map(world)
    sign = -1.0 if side == "right" else 1.0
    ring = []
    for ahead in RING_AHEAD_M:
        for bearing in np.radians(np.arange(0, 360, 10)):
            stance = cxy - (R.rot_z(bearing) @ np.array([ahead, sign * pick.STAND_SIDE_M, 0.0]))[:2]
            if float(np.max(np.abs(to_surface_frame(stance, s)) - half)) < RING_OUTSIDE_M:
                continue
            if pick.floor_clearance(boxes, stance) < pick.STANCE_CLEAR_M:
                continue
            if pick.other_clearance(world, sk, stance) < RING_OTHER_M:
                continue
            u = np.array([np.cos(bearing), np.sin(bearing)])
            d = 0.0
            while np.all(np.abs(to_surface_frame(cxy - u * d, s)) <= half) and d < 1.0:
                d += 0.01
            ring.append((ahead, float(np.linalg.norm(stance - start_xy)),
                         dict(xy=cxy, face=float(bearing), stance=stance, edge_xy=cxy - u * d, edge="ring", depth=d)))  # fmt: skip
    ring.sort(key=lambda z: z[:2])
    return [cd[2] for cd in ring]


def container_spot(world, container: str, side: str, start_xy) -> tuple[dict | None, str]:
    """(the first of `stances` -- then, with RING, of `ring_stances` -- among place.MAX_ROUTE_TRIES with a route on
    pick's map from `start_xy`, or None; why not)."""
    occ, nav, _ = pick.nav_map(world)
    cands = stances(world, container, side, start_xy)
    for cd in cands[: place.MAX_ROUTE_TRIES]:
        if pick.route(occ, nav, start_xy, cd["stance"]) is not None:
            return cd, ""
    if RING and world.meta["objects"][container].get("surface"):
        for cd in ring_stances(world, container, side, start_xy)[: place.MAX_ROUTE_TRIES]:
            if pick.route(occ, nav, start_xy, cd["stance"]) is not None:
                return cd, ""
    if not cands:
        return None, f"no stance at {container}"
    return None, f"no route to {len(cands)} stances at {container}"


def near_spot(world, container: str, side: str, stance, face: float) -> dict | None:
    """The first of `stances` at the pick's edge (`face`) within NEAR_M of the pick `stance`, with a route from it (it
    is no dead end) and every point of the straight walk to it as clear as a stance (floor objects; other furniture at
    an unlisted edge), or None."""
    sk = world.meta["objects"][container]["surface"]
    s = world.meta["surfaces"][sk]
    occ, nav, boxes = pick.nav_map(world)
    stance = np.asarray(stance, float)[:2]
    for cd in stances(world, container, side, stance):
        if abs(R.wrap(cd["face"] - face)) > 1e-6 or np.linalg.norm(cd["stance"] - stance) > NEAR_M:
            continue
        unlisted = cd["edge"] not in (s.get("approach_edges") or ())
        line = [stance + f * (cd["stance"] - stance) for f in np.linspace(0.0, 1.0, 7)]
        if any(pick.floor_clearance(boxes, p) < pick.STANCE_CLEAR_M for p in line):
            continue
        if unlisted and any(pick.other_clearance(world, sk, p) < pick.OTHER_EDGE_CLEAR_M for p in line):
            continue
        if pick.route(occ, nav, stance, cd["stance"]) is not None:
            return cd
    return None


def path_top(world, key: str, a, b) -> tuple[float, list[str]]:
    """The highest top (m) among the objects resting on `key`'s support (place.neighbours, `key` excepted) whose
    footprint crosses the band from xy `a` to `b`, widened by `key`'s larger half width + NEAR_BAND_M, and their keys
    (-inf and [] with none)."""
    o = world.meta["objects"][key]
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    w = float(max(o["box"]["half_extent"][:2])) + NEAR_BAND_M
    d = b - a
    band = (0.5 * (a + b), np.array([0.5 * float(np.linalg.norm(d)) + w, w]), float(np.arctan2(d[1], d[0])))
    top, crossed = -np.inf, []
    for k, fp in place.neighbours(world, o["surface"], key).items():
        if overlap(band, fp, 0.0):
            pos, _ = world.object_pose(k)
            box = world.meta["objects"][k]["box"]
            top = max(top, float(pos[2] + (box.get("centre") or (0, 0, 0))[2] + box["half_extent"][2]))
            crossed.append(k)
    return top, crossed


def near_lift(world, key: str, stance, spot: dict) -> tuple[float, list[str]]:
    """The near drop's lift (m): at least grasp_program.LIFT_M, else the underside NEAR_CLEAR_M above `path_top` of the
    jaw's path from the object now to where it rides at the drop stance (the same offset from the pelvis, the heading
    kept); and the objects crossed."""
    pos, quat = world.object_pose(key)
    box = world.meta["objects"][key]["box"]
    xy = pos[:2] + np.asarray(box.get("centre") or (0, 0, 0), float)[:2]
    top, crossed = path_top(world, key, xy, xy + np.asarray(spot["stance"], float) - np.asarray(stance, float)[:2])
    under = float(pos[2]) - low_offset(quat, box)
    return max(float(grasp_program.LIFT_M), top + NEAR_CLEAR_M - under), crossed


def reach_seconds(cfg, depth: float, rise: float) -> float:
    """place.lift_to_release_s's reach terms: over the opening (legs of `depth` + VIA_OUT_M, the first also `rise`)
    and the lowering, each with its settle."""
    leg = depth + VIA_OUT_M
    t = sum(max(cfg.reach_min_s, d / cfg.reach_speed) for d in (float(np.hypot(leg, rise)), leg)) + cfg.reach_settle_s
    return t + max(cfg.reach_min_s, ABOVE_M / place.DESCENT_SPEED) + cfg.reach_settle_s


def near_seconds(cfg, stance, face: float, spot: dict, rise: float) -> float:
    """T_est of the near drop: the pick's lift and hold, one walk_to keeping the heading from the pick `stance` to the
    drop stance (its final turn at place.CARRY_TURN_RATE), the reach over the opening and the lowering."""
    t = grasp_program.SIMPLE["lift_seconds"] + grasp_program.HOLD_S
    carry = dataclasses.replace(cfg, turn_rate=place.CARRY_TURN_RATE)
    t += place.walk_seconds(carry, stance, face, [np.asarray(stance, float)[:2], spot["stance"]], spot["face"], True)
    return t + reach_seconds(cfg, spot["depth"], rise)


def plan_ahead(world, key: str, container: str, side: str, start_xy, cfg) -> dict | None:
    """The drop planned before any contact, as place.plan_ahead: None when the pick has no plan, else dict(hand, face,
    stance, edge_xy, xy, top, spot, why, near, lift, crossed, corners, t_est, margin). The near drop (`near_spot`, C on
    the object's support) when there is one and its `near_lift` is at most NEAR_LIFT_MAX_M, its T_est `near_seconds`;
    else `container_spot` from the back-off's end, the carry route and place.lift_to_release_s (lift None)."""
    plan, hand, _ = pick.plan_hand(world, start_xy, key, side)
    if plan is None:
        return None
    face, _, _, stance, edge_xy = plan
    pos, _ = world.object_pose(key)
    obj = world.meta["objects"][key]
    out = dict(hand=hand, face=face, stance=stance, edge_xy=edge_xy, xy=pos[:2].copy(), top=pick.is_top(world, key),
               spot=None, why="", near=False, lift=None, crossed=[], corners=None, t_est=None,
               margin=place.grip_margin(obj["box"]["half_extent"][2], (obj["box"].get("centre") or (0, 0, 0))[2]))  # fmt: skip
    _, rim, _, _, _ = opening(world, container)
    rise = abs(float(world.meta["surfaces"][obj["surface"]]["top"]) + pick.LIFT_M - rim - DROP_CLEAR_M - ABOVE_M)
    if NEAR_M > 0 and world.meta["objects"][container].get("surface") == obj["surface"]:
        spot = near_spot(world, container, hand, stance, face)
        lift, crossed = (None, []) if spot is None else near_lift(world, key, stance, spot)
        if spot is not None and lift <= NEAR_LIFT_MAX_M:
            t_est = near_seconds(cfg, stance, face, spot, rise)
            out.update(spot=spot, near=True, lift=lift, crossed=crossed, t_est=t_est)
            return out
    back = stance - place.BACK_AFTER_PICK_M * np.array([np.cos(face), np.sin(face)])
    out["spot"], out["why"] = container_spot(world, container, hand, back)
    if out["spot"] is None:
        return out
    occ, nav, _ = pick.nav_map(world)
    out["corners"] = pick.route(occ, nav, back, out["spot"]["stance"])
    out["t_est"] = place.lift_to_release_s(cfg, stance, face, out["corners"], out["spot"]["face"],
                                           out["spot"]["depth"], rise)  # fmt: skip
    return out


@place._pick_switches()
def place_into(ctx, key: str, container: str, side: str = "right") -> Outcome:
    """Plan the drop (`plan_ahead`), pick `key` as place.place_on does, carry it to the drop stance and let it go into
    `container` (`_drop_in`); see the module doc."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    blocker = place.blocking_neighbour(world, key)
    if blocker is not None:
        why = f"pick refused: a neighbour blocks the jaw ({blocker} overlaps it)"
        return ctx.note("pick_check", Outcome(False, why, m.t, m.t), neighbour=blocker)
    ahead = plan_ahead(world, key, container, side, m.sim.d.qpos[:2].copy(), m.cfg)
    if ahead is not None and not CREEP:
        ahead = dict(ahead, top=True)  # `top` alone exempts a pick from place's creep refusal and grip-margin check
    refused = place._refusal(ctx, ahead)
    if refused is not None:
        return refused
    z0 = float(world.object_pose(key)[0][2])
    n0 = len(ctx.log)
    saved = {k: grasp_program.SIMPLE[k] for k in ("disturb_limit_m", "settle_abort_m", "entry_capture_depth_m")}
    grasp_program.SIMPLE.update(disturb_limit_m=place.PICK_DISTURB_LIMIT_M, settle_abort_m=place.PICK_SETTLE_ABORT_M,
                                entry_capture_depth_m=place.PICK_CAPTURE_DEPTH_M)  # fmt: skip
    lift0 = grasp_program.LIFT_M
    if ahead is not None and ahead["near"]:
        ctx.note("near_lift", Outcome(True, "", m.t, m.t), lift_cm=round(100 * ahead["lift"], 1),
                 crossed=ahead["crossed"])  # fmt: skip
        grasp_program.LIFT_M = ahead["lift"]
    try:
        out = pick.pick(ctx, key, side)
    finally:
        grasp_program.SIMPLE.update(saved)
        grasp_program.LIFT_M = lift0
    if not out.ok:
        refused = not any(r["step"] == "walk" for r in ctx.log[n0:])
        return place._fail(ctx, key, f"{'pick refused' if refused else 'pick failed'}: {out.why}")
    hands = sorted(m.gripping(body)) or sorted(m.touching(body))
    if not hands:
        return place._fail(ctx, key, "pick failed: the object is not in the hand after the pick")
    side = hands[0]
    source = world.meta["objects"][key]["surface"]
    lifted = tuple(dict.fromkeys(k for k in (source, world.meta["objects"][container].get("surface")) if k))
    home = dict(stance=ahead["stance"], face=ahead["face"], xy=ahead["xy"], edge_xy=ahead["edge_xy"], surface=source)
    spot = ahead["spot"]
    if side != ahead["hand"]:
        return place._fail(ctx, key, f"pick ended in the {side} hand, the drop is planned for the {ahead['hand']}",
                           lifted, home)  # fmt: skip
    if ahead["near"]:
        dist = float(np.linalg.norm(spot["stance"] - m.sim.d.qpos[:2]))
        off = float(np.degrees(abs(R.wrap(m.yaw() - spot["face"]))))
        if dist > NEAR_M or off > NEAR_YAW_DEG:
            return place._fail(ctx, key, f"pick ended {dist:.2f} m, {off:.0f} deg from the near drop's walk", lifted,
                               home)  # fmt: skip
        failed = place._grip_checks(ctx, key, side, lifted, ahead, home)
        if failed is not None:
            return failed
        with place._carry_turns(m):
            walked = m.walk_to(spot["stance"], spot["face"], face="fixed")
    else:
        failed = place._handoff(ctx, key, side, lifted, ahead, home)
        if failed is not None:
            return failed
        occ, nav, _ = pick.nav_map(world)
        with place._carry_turns(m):
            walked = pick.walk_route(m, occ, nav, spot["stance"], spot["face"])
    out = ctx.note("carry", walked, near=ahead["near"], spot=spot["xy"], edge=spot["edge"], depth=spot["depth"],
                   stance=spot["stance"])  # fmt: skip
    if not out.ok or m.fell_at is not None:
        return place._fail(ctx, key, f"carry: {out.why}", lifted, home)
    if side not in m.gripping(body):
        return place._fail(ctx, key, f"dropped during the carry: the {side} hand no longer grips it", lifted)
    return _drop_in(ctx, key, container, side, spot, lifted, z0)


def _drop_in(ctx, key: str, container: str, side: str, spot: dict, lifted, z0: float) -> Outcome:
    """At the drop stance: reach the object over the opening and lower it, let go only on `release_fit` (one
    correction), rise, retract, step back; or keep it (`_keep`). Done with in(key, container) and no finger on it."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    cxy, rim, half, wall, cyaw = opening(world, container)
    pos, quat = world.object_pose(key)
    jaw, rot = m.jaw_centre(side)
    rel = pos - jaw  # object origin from the jaw centre, as carried
    target = np.r_[cxy, rim + DROP_CLEAR_M + low_offset(quat, world.meta["objects"][key]["box"])] - rel
    above = target + np.array([0.0, 0.0, ABOVE_M])
    n_in = np.array([np.cos(spot["face"]), np.sin(spot["face"])])
    via = (np.r_[np.asarray(spot["edge_xy"], float) - n_in * VIA_OUT_M - rel[:2], above[2]],)
    legs = [("over", above, dict(via=via)), ("lower", target, dict(speed=place.DESCENT_SPEED))]
    for attempt in range(2):
        for name, goal, kw in legs:
            out = ctx.note(name, m.reach(side, goal, rot, **kw), target=np.round(goal, 3))
            if m.fell_at is not None:
                return Outcome(False, f"{name}: {out.why}", out.t0, out.t1)
        fit, why, move = release_fit(object_points(world, m, key), cxy, rim, half, wall, cyaw)
        p = world.object_pose(key)[0]
        ctx.note("release_fit", Outcome(fit, why, m.t, m.t), attempt=attempt, move_cm=np.round(100 * move, 1),
                 off_cm=round(100 * float(np.linalg.norm(p[:2] - cxy)), 1), rim=round(rim, 3))  # fmt: skip
        if fit:
            break
        target = target + move  # the commanded target, so a steady landing error of the reach is corrected too
        legs =[("raise_again", target + np.array([0.0, 0.0, place.LOWER_RETRY_UP_M]), {}),
                ("lower_again", target, dict(speed=place.DESCENT_SPEED))]  # fmt: skip
    if not fit:
        return _keep(ctx, key, side, f"no release over {container}: {why}", lifted, z0)
    r = ctx.note("open", m.release(side))
    if m.fell_at is None:
        jaw, rot = m.jaw_centre(side)
        r = ctx.note("rise", m.reach(side, jaw + np.array([0.0, 0.0, RISE_M]), rot))
    if m.fell_at is None:
        jaw, rot = m.jaw_centre(side)
        r = ctx.note("retract", m.reach(side, jaw - RETRACT_M * np.r_[n_in, 0.0], rot))
    if m.fell_at is None:
        r = ctx.note("step_back", m.sidestep(forward=-place.BACK_AFTER_PLACE_M))
    if m.fell_at is not None:
        return Outcome(False, f"after the release: {r.why}", r.t0, r.t1)
    touching = sorted(m.touching(body))
    if touching:
        return Outcome(False, f"still touching after release: {'/'.join(touching)} hand", r.t0, r.t1)
    ok, why = world.check([f"in({key}, {container})"])(world.state([key, container]))
    return Outcome(ok, why, r.t0, r.t1)


def _keep(ctx, key: str, side: str, why: str, lifted, z0: float) -> Outcome:
    """No release over the opening: set the object down on the surface of `lifted` under it (place._fail), else keep
    it shut in the hand, raised to RAISE_BELOW_START_M below its start height `z0` when it is lower."""
    world, m = ctx.world, ctx.motor
    if place.surface_below(world, key, lifted) is not None:
        return place._fail(ctx, key, why, lifted)
    dz = z0 - RAISE_BELOW_START_M - float(world.object_pose(key)[0][2])
    if dz > 0:
        jaw, rot = m.jaw_centre(side)
        ctx.note("raise_back", m.reach(side, jaw + np.array([0.0, 0.0, dz]), rot), dz_cm=round(100 * dz, 1))
    return Outcome(False, f"{why} [kept in the {side} hand]", m.t, m.t)
