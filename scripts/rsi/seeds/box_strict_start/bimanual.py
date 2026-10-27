"""A two-hand hug pick of a box wider than a jaw (TODO 38, plan section 7): stand square to the support edge nearest the
box, crouch, bring both open jaws beside the box's two side walls (the walls whose normals run along the edge), press
them together with `squeeze` and lift by arms and legs at once, still pressing.

Geometry (from the scene file and the box's measured pose, never from its name):
- the approach `v` is the box axis closest to the inward normal of the nearest approach edge; `u` is the robot's left
  when facing `v`; `w` the box's half width along `u`;
- the stance pelvis stands STANDOFF_M outside that edge on the line through the box centre along `v`, facing `v`;
- each wrist points along `v` with z up, so the jaw opens along `u` and the inner finger's outer flank (FLANK_M from the
  jaw axis) is what touches the wall: the jaw centre goes to the wall plane + FLANK_M + CLEAR_M, at the box centre's
  depth and GRASP_BELOW_RIM_M under the rim;
- the hands come in outward first, then forward PRE_OUT_M beside the walls, then inward onto them.

The jaws stay open (a closed jaw touches with its base, which `held` does not count). The lift (lift_margin_legs) asks for
LIFT_DZ_M with the pelvis up to LIFT_PELVIS_Z, holds LIFT_HOLD_S, and checks the box rose LIFT_MIN_RISE_M (held()'s
gap) with both hands on it; a miss re-arms the squeeze and lifts again (or re-grips when a hand came off)."""

from __future__ import annotations

import math

import numpy as np

from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene import navigate
from locomani.tasks.scene import stage as S

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


def yaw_of(q) -> float:
    w, x, y, z = (float(c) for c in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


_edge_normals = {"+x": (-1.0, 0.0), "-x": (1.0, 0.0), "+y": (0.0, -1.0), "-y": (0.0, 1.0)}  # inward, surface frame


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


def _pair(d: dict) -> tuple:
    return tuple(np.asarray(d[s], float) for s in SIDES)


def _reach_both(m, goal: dict, rot, via=(), **kw) -> Outcome:
    return m.reach("both", _pair(goal), (rot, rot), via=[_pair(p) for p in via], **kw)


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
