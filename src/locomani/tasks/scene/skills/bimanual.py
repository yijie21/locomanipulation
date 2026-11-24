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
  jaw axis) is what touches the wall: the jaw centre goes to the wall + FLANK_M + CLEAR_M, at the box centre's depth;
- its height comes from the collision hull (`grip_height`): scanning down from GRASP_BELOW_RIM_M under the rim to just
  above the centre of mass, the highest band where both walls are at their full width at the jaw's depth (a wall that
  narrows towards the top squeezes the object down out of the jaws; a squeeze line below the centre of mass lets it
  swing about the line between the hands), each wall's offset read there; the box's rim and width without a hull;
- the hands come in outward first, then forward PRE_OUT_M beside the walls, then inward onto them.
The jaws stay open (a closed jaw touches with its base, which `held` does not count). The lift (lift_margin_legs) asks for
LIFT_DZ_M with the pelvis up to LIFT_PELVIS_Z, holds LIFT_HOLD_S, and checks the box rose LIFT_MIN_RISE_M (held()'s
gap) with both hands on it; a miss re-arms the squeeze and lifts again (or re-grips when a hand came off). When the
task's goal asks for lifted(X, c, t) the check is that goal after the hold (`lift_goal_check`: the lowest box corner
c + LIFT_MARGIN_M above the support top, upright under t): the box hangs from the line between the hands and swings
about it, so a held box can sit short of c; it is then raised by the deficit with the squeeze kept (no re-squeeze),
the hands at most LIFT_MAX_DZ_M above the grasp, and held LIFT_HOLD_S again.

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

from . import grasp_common as GC
from . import into, place
from .pick import nav_map, walk_route

SIDES = ("left", "right")

# stance and crouch
STANDOFF_M = 0.3  # (0.38 until gso_bimanual_lift's set) pelvis outside the support edge
HUG_PELVIS_Z = 0.72
STANCE_TOL_M = 0.05
STANCE_TOL_DEG = 8.0
STANCE_RETRY_STRAIGHT_M = 0.30  # a missed stance this close is re-walked straight, farther by a new path
POSTURE_TOL_M = 0.05
# hand placement
GRASP_BELOW_RIM_M = 0.045  # (0.10 until gso_bimanual_lift's set) jaw-centre height under the box's rim: the highest grip the hull scan may choose
GRASP_DEPTH_M = 0.0  # jaw centres along the approach, from the box centre
# the grip height from the collision hull's width along u (`grip_height`): the scan runs down from GRASP_BELOW_RIM_M
# under the rim in SCAN_STEP_M steps to SCAN_ABOVE_COM_M above the centre of mass (and SCAN_LOW_M above the bottom); it
# takes the highest band (the finger pad's FINGER_HALF_M above and below the jaw centre, FINGER_DEPTH_M either side of
# the jaw's depth along v) where each wall stays within WALL_TOL_M of its outermost section at that depth, so the pads
# press a wall at its full width rather than one that narrows towards the top (which squeezes the object down and out)
# or a handle cut-out; when no band is, the band whose walls fall least short
SCAN_STEP_M = 0.01
SCAN_ABOVE_COM_M = 0.01
SCAN_LOW_M = 0.04
FINGER_HALF_M = 0.015
FINGER_DEPTH_M = 0.02
WALL_TOL_M = 0.005
FLANK_M = 0.0635  # the open jaw's finger flank from the jaw axis
CLEAR_M = 0.005  # gap to the wall before the squeeze
PRE_OUT_M = 0.06  # pre-grasp and via beside the wall, outward
PRE_BACK_M = 0.12  # pre-grasp short of the grasp along the approach
PRE_TOL_M = 0.08
MIN_TRACK = 0.12  # reach_check stall ratio
# the entry is judged by where each jaw centre is against its wall, not by its distance to the target: the squeeze
# closes a lateral gap, so a hand short along the approach or a little high still presses the wall
BESIDE_EDGE_M = 0.015  # (0.04 until gso_bimanual_lift's set) jaw centre inside the wall's ends along the approach by this much (at least half the half depth)
BESIDE_DZ_M = 0.03  # ... within this of the chosen grip height
BESIDE_LOW_M = 0.02  # (0.06 until gso_bimanual_lift's set) ... and above the box's bottom
BESIDE_RIM_M = 0.01  # (0.03 until gso_bimanual_lift's set) ... and under its rim
BESIDE_IN_M = 0.02  # ... at most this far inside the target's wall-plane offset
BESIDE_OUT_M = 0.04  # ... at most this far outside the target
BOX_MOVED_MAX_M = 0.03  # the box knocked by a hand during the entry
# squeeze and lift
SQUEEZE_M = 0.06
SQUEEZE_S = 1.0
LIFT_DZ_M = 0.4  # (0.15 until gso_bimanual_lift's set)
LIFT_PELVIS_Z = 0.79
LIFT_SPEED = 0.15  # m/s, the motor's reach speed
LIFT_HOLD_S = 3  # (1.0 until gso_bimanual_lift's set)
LIFT_MIN_RISE_M = 0.02  # held()'s gap: a box already held is kept as it is (re-squeezing and lifting higher dropped it)
LIFT_RETRY_DZ_M = 0.10  # a retry lifts this far above where the hands are
# against the goal's lifted(X, c, t) (`lift_goal_check`): a held box whose lowest corner is short of c + LIFT_MARGIN_M
# after the hold is raised by the deficit, the squeeze kept, the hands at most LIFT_MAX_DZ_M above the grasp,
# LIFT_TOPUPS times (the round's stage budget caps them too)
LIFT_MARGIN_M = 0.03
LIFT_MAX_DZ_M = 0.45
LIFT_TOPUPS = 1
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


def grip_height(pieces, c, u, v, bottom: float, rim: float, com_z: float) -> dict | None:
    """The jaw-centre height and the walls' offsets along u there, from the collision hull (module constants above):
    dict(z, left (the outermost s on +u over the pad's band), right (on -u, negative), short_cm (how far the band's
    nearer wall falls inside its outermost section at the jaw's depth), band_cm / full_cm (the band's narrowest and the
    hull's widest width there), how: "full_width" (the highest band within WALL_TOL_M), "widest" (no band is: the one
    falling least short) or "top" (the object is too short to scan below the top)), or None without a hull section at
    that depth. `pieces` are grasp_common pieces (`GC.pieces`) or plain vertex arrays (a test's blocks)."""
    hp = GC.prepare(pieces)
    strip = (np.asarray(v, float), GRASP_DEPTH_M, FINGER_DEPTH_M)
    c, u = np.asarray(c, float), np.asarray(u, float)
    secs = [
        s
        for s in (GC.section_extent(hp, z, c, u, strip) for z in np.arange(bottom + 0.01, rim - 0.005, SCAN_STEP_M))
        if s
    ]
    if not secs:
        return None
    out_left, out_right = max(s[1] for s in secs), min(s[0] for s in secs)  # each wall's outermost section
    full = max(s[1] - s[0] for s in secs)
    top = rim - GRASP_BELOW_RIM_M
    low = max(com_z + SCAN_ABOVE_COM_M, bottom + SCAN_LOW_M)
    n = int(math.floor((top - low) / SCAN_STEP_M + 1e-9)) + 1 if top >= low else 1
    best = None
    for z in (top - k * SCAN_STEP_M for k in range(n)):
        band = [GC.section_extent(hp, z + dz, c, u, strip) for dz in np.linspace(-FINGER_HALF_M, FINGER_HALF_M, 5)]
        if any(b is None for b in band):
            continue
        short = max(out_left - min(b[1] for b in band), max(b[0] for b in band) - out_right)
        cand = dict(z=float(z), left=max(b[1] for b in band), right=min(b[0] for b in band),
                    short_cm=round(100 * short, 2), band_cm=round(100 * min(b[1] - b[0] for b in band), 2),
                    full_cm=round(100 * full, 2))  # fmt: skip
        if short <= WALL_TOL_M:
            return dict(cand, how="full_width" if top >= low else "top")
        if best is None or short < best[0]:
            best = (short, cand)
    return None if best is None else dict(best[1], how="widest")


def geometry(world, key: str) -> dict:
    """The hug's frame and targets from the box's measured pose: approach `v`, left `u`, the stance, each hand's grasp
    target, the wrist rotation, the box's half width and depth along them. The jaw-centre height and each wall's
    offset come from the collision hull's width along u at the jaw's depth (`grip_height`), or from the box,
    GRASP_BELOW_RIM_M under its rim, without a hull."""
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
    body = (getattr(world, "free", None) or {}).get(key)
    sim = getattr(world, "sim", None)
    com_z = float(sim.d.xipos[body][2]) if sim is not None and body is not None else bottom + float(half[2])
    grip = grip_height(GC.pieces(world, key), c, u, v, bottom, rim, com_z)
    if grip is None:
        z, walls = rim - GRASP_BELOW_RIM_M, {"left": float(w), "right": float(w)}
        grip = dict(z=z, how="box")
    else:
        z, walls = grip["z"], {"left": grip["left"], "right": -grip["right"]}
    lateral = {s: walls[s] + FLANK_M + CLEAR_M for s in SIDES}
    target = {s: np.r_[c + sgn * u * lateral[s] + v * GRASP_DEPTH_M, z] for s, sgn in (("left", 1.0), ("right", -1.0))}
    out = {s: np.r_[sgn * u * PRE_OUT_M, 0.0] for s, sgn in (("left", 1.0), ("right", -1.0))}
    rot = np.column_stack([np.r_[v, 0.0], np.r_[u, 0.0], [0.0, 0.0, 1.0]])
    return dict(edge=edge, depth=d, depth_v=d_v, v=v, u=u, w=float(w), depth_half=float(depth_half), centre=c, stance=stance,
                stance_yaw=math.atan2(v[1], v[0]), target=target, out=out, rot=rot, grasp_z=z, bottom=bottom, rim=rim,
                lateral=lateral, com_z=com_z, grip=grip)  # fmt: skip


def box_centre(world, key: str) -> np.ndarray:
    """The object's box centre in the world now (its origin plus the turned `box.centre`)."""
    pos, quat = world.object_pose(key)
    box = world.meta["objects"][key]["box"]
    return np.asarray(pos, float) + np.asarray(box.get("centre") or (0, 0, 0), float) @ _rot3(quat).T


underside = GC.underside  # the object's lowest box corner (m) now
lift_goal = GC.lift_goal  # the goal's lifted(key, clearance, tilt), or None
clearance_now = GC.clearance_now  # (the lowest corner above the support top, m; the upright axis off vertical, deg)


def lift_goal_check(world, m, key: str, z0: float, goal: tuple[float, float] | None):
    """`bimanual_lift_check` (the rise and both hands on), and with the goal's (clearance, tilt) the goal itself with
    LIFT_MARGIN_M to spare: the lowest corner clearance + LIFT_MARGIN_M above the support top, upright under tilt.
    Values add low_cm, need_cm, tilt_deg."""
    ok, values = S.bimanual_lift_check(world, m, key, z0, LIFT_MIN_RISE_M)
    return GC.goal_check(world, key, goal, LIFT_MARGIN_M, ok, values)


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


def beside_window(g: dict) -> tuple[float, float, float]:
    """The beside check's bounds in the box frame `g`: (|along| at most, lowest z, highest z) -- inside the wall's ends
    by BESIDE_EDGE_M or at most half the half depth (a thin object's +-0.4 cm is not a window), within BESIDE_DZ_M of
    the chosen grip height and between BESIDE_LOW_M above the bottom and BESIDE_RIM_M under the rim."""
    along = max(g["depth_half"] - BESIDE_EDGE_M, 0.5 * g["depth_half"])
    lo = max(g["grasp_z"] - BESIDE_DZ_M, g["bottom"] + BESIDE_LOW_M)
    hi = min(g["grasp_z"] + BESIDE_DZ_M, g["rim"] - BESIDE_RIM_M)
    return along, lo, hi


def _beside_check(m, g: dict):
    """Each jaw centre beside its wall, in the box frame `g` was computed in: along the approach within the wall,
    near the chosen grip height (`beside_window`), laterally from BESIDE_IN_M inside to BESIDE_OUT_M outside the
    target's offset."""
    ok, values = True, {}
    v3, u3 = np.r_[g["v"], 0.0], np.r_[g["u"], 0.0]
    c3 = np.r_[g["centre"], g["bottom"]]
    along_max, z_lo, z_hi = beside_window(g)
    for s, sgn in (("left", 1.0), ("right", -1.0)):
        p = m.jaw_centre(s)[0]
        along, out = float((p - c3) @ v3), sgn * float((p - c3) @ u3) - g["lateral"][s]
        ok_s = abs(along) <= along_max and z_lo <= p[2] <= z_hi and -BESIDE_IN_M <= out <= BESIDE_OUT_M
        ok = ok and ok_s
        values[f"{s}_jaw_err_cm"] = round(100 * float(np.linalg.norm(p - g["target"][s])), 2)
        values[f"{s}_along_cm"], values[f"{s}_out_cm"] = round(100 * along, 2), round(100 * out, 2)
        values[f"{s}_dz_cm"] = round(100 * float(p[2] - g["grasp_z"]), 2)
    return ok, values


def pick(ctx, key: str) -> Outcome:
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    g = geometry(world, key)
    grip = g["grip"]
    ctx.note("hug_plan", Outcome(True, t0=m.t, t1=m.t, info=dict(
        edge=g["edge"], depth_m=round(g["depth"], 3), half_width_m=round(g["w"], 3), grasp_z=round(g["grasp_z"], 3),
        grasp_line_m=round(g["depth_v"] + STANDOFF_M + GRASP_DEPTH_M, 3), stance=[round(float(x), 3) for x in g["stance"]],
        stance_yaw_deg=round(math.degrees(g["stance_yaw"]), 1), grip_how=grip["how"],
        grip_below_rim_cm=round(100 * (g["rim"] - g["grasp_z"]), 1),
        grip_above_com_cm=round(100 * (g["grasp_z"] - g["com_z"]), 1),
        wall_left_cm=round(100 * (g["lateral"]["left"] - FLANK_M - CLEAR_M), 1),
        wall_right_cm=round(100 * (g["lateral"]["right"] - FLANK_M - CLEAR_M), 1),
        wall_short_cm=grip.get("short_cm"), band_cm=grip.get("band_cm"), full_cm=grip.get("full_cm"))))  # fmt: skip

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

    # 6. lift by arms and legs, still pressing; hold, then check the rise with both hands on (and the goal's clearance)
    z0 = float(world.object_pose(key)[0][2])
    rot = state["g"]["rot"]
    goal_lift = lift_goal(ctx, key)
    tries = dict(retry=0, topup=0, goal=None)

    def lift(goal: dict) -> Outcome:
        tries["goal"] = goal
        o = _reach_both(m, goal, rot, keep_squeeze=True, pelvis_z=LIFT_PELVIS_Z, speed=LIFT_SPEED)
        if m.fell_at is None:
            m.hold(LIFT_HOLD_S)
        o.t1 = m.t
        return o

    up = {s: state["g"]["target"][s] + np.array([0.0, 0.0, LIFT_DZ_M]) for s in SIDES}

    def top_up(v):
        """Held, upright enough, short of the goal's clearance: both hands up by the deficit from where they were sent,
        the squeeze kept, at most LIFT_MAX_DZ_M above the grasp; None when nothing is left to raise."""
        if tries["topup"] >= LIFT_TOPUPS:
            return None
        last = tries["goal"]
        z = GC.top_up(v, goal_lift, [float(last[s][2]) for s in SIDES], float(state["g"]["grasp_z"]) + LIFT_MAX_DZ_M)
        if z is None:
            return None
        goal = {s: np.r_[last[s][:2], zs] for s, zs in zip(SIDES, z)}
        tries["topup"] += 1

        def again():
            o = lift(goal)
            o.info["topup_cm"] = round(100 * max(float(goal[s][2] - last[s][2]) for s in SIDES), 1)
            return o

        return again

    def re_lift(k, v):
        held = v.get("touching") == list(SIDES) and float(v.get("rise_cm", 0.0)) >= 100 * LIFT_MIN_RISE_M
        if goal_lift is not None and held:  # in both hands and off the support, short of the goal or tipped
            return top_up(v)
        if tries["retry"] >= LIFT_RETRIES:
            return None
        tries["retry"] += 1
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
                   lambda o: lift_goal_check(world, m, key, z0, goal_lift), re_lift)  # fmt: skip


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
