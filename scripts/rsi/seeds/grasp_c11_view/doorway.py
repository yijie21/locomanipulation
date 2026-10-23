"""The narrow-door walk (grasp_atomic iteration 5, reopen of park C1): a start whose spawn is in another room than the
object, with no stance in the object's room reachable on the runner's occupancy map (body radius 0.35 m), first walks
through the doorway on a map of its own at DOOR_BODY_RADIUS_M, then hands back to the normal skill on the runner's map.

Cause (DisplayWolf s410019): the spawn (1.40, 1.14) is in the bedroom and the wolf (-0.16, 0.77) in the living room.
The bedroom door is about 0.65 m wide on the map, so at 0.35 m it is shut and every stance pick could walk to was on
the bedroom side of the wall: (0.374, 0.669), whose grip closed on the wall ('adt_env'). At 0.30 m the door opens.

`walk` is None (nothing done, no sim effect) unless both the spawn and the object lie in room polygons of the scene file
and they differ, and no stance of `pick.stance_candidates` inside the object's room has a path on the runner's map.
Then:
- handoff: of the cells free on the runner's map, inside the object's room and connected there to one of those in-room
  stances, the one the 0.30 m map reaches first from the spawn (the doorway's exit: the 0.30 m walk stays short, rule
  16 records an arm that caught the kitchen island and a fall at 0.26 m);
- the stance pick would choose from the handoff must be inside the object's room, so the hand never reaches through
  a wall;
- walk the 0.30 m path to the handoff (arms at the walk's reference), facing along its last leg; check the stance again
  from where the walk stopped;
- walk on to the start of that stance's last leg (its nearest free cell on the runner's map) on a copy of the runner's
  map that also holds the scene file's floor objects the map leaves out (`floor_statics`). s410019's first smoke walked
  on from the handoff on the runner's map: its path ran 9 cm past WhiteLiddedTrashBin (0.33 m tall, against the
  bedroom wall west of the door), the pelvis stalled 25 cm behind its reference and the body tilted 45 deg.
The normal skill (rim or pick) then walks its last leg on the runner's map. When no handoff or no in-room stance is
found, the door step is logged and the normal skill runs as before; with no path on the copy, it walks from the
handoff."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.primitives.navigator import UNREACHABLE, build
from locomani.tasks.scene.navigate import path as nav_path
from locomani.tasks.scene.navigate import walk_path

from . import pick

DOOR_BODY_RADIUS_M = 0.30  # the doorway map's inflation (the runner's is 0.35 m)
SAMPLE_M = 0.025  # clearance samples along the doorway legs
# floor objects the occupancy map leaves out: not free, no surface, bottom below this, no half extent above the next
STATIC_BOTTOM_M = 0.15
STATIC_MAX_HALF_M = 1.5  # the apartment shell's box is 6.6 x 4.9 m

_MAPS: dict = {}


def door_maps(world):
    """(occ, nav) at DOOR_BODY_RADIUS_M for this scene, built once per scene file and radius."""
    k = (id(world.meta), float(DOOR_BODY_RADIUS_M))
    if k not in _MAPS or _MAPS[k][0] is not world.meta:
        _MAPS[k] = (world.meta, *build(world.meta, body_radius=float(DOOR_BODY_RADIUS_M)))
    return _MAPS[k][1], _MAPS[k][2]


def floor_statics(meta) -> list:
    """(centre xy, half extent along its yaw axes, yaw) of every static object standing on the floor (not free, not a
    support, bottom below STATIC_BOTTOM_M) other than the apartment shell: trash bins, a step stool, door leaves."""
    out = []
    for o in meta["objects"].values():
        if o.get("free") or o.get("surface"):
            continue
        p, h = np.asarray(o["init_pos"], float), np.asarray(o["box"]["half_extent"], float)
        if p[2] - h[2] >= STATIC_BOTTOM_M or h.max() >= STATIC_MAX_HALF_M:
            continue
        Ro = R.mat(np.asarray(o["init_quat"], float))
        yaw = float(np.arctan2(Ro[1, 0], Ro[0, 0]))
        out.append((p[:2], (np.abs(R.rot_z(yaw).T @ Ro) @ h)[:2], yaw))
    return out


def approach_maps(world):
    """(occ, nav) at the runner's body radius with `floor_statics` added, built once per scene file."""
    k = (id(world.meta), "approach", float(world.occ.body_radius), float(STATIC_BOTTOM_M), float(STATIC_MAX_HALF_M))
    if k not in _MAPS or _MAPS[k][0] is not world.meta:
        occ, nav = build(world.meta, body_radius=float(world.occ.body_radius))
        for c, h, yaw in floor_statics(world.meta):
            occ.add_box(c, h, yaw)
        occ.recompute()
        _MAPS[k] = (world.meta, occ, nav)
    return _MAPS[k][1], _MAPS[k][2]


def inside(poly, xy) -> bool:
    """Even-odd test of a point against a world-xy polygon."""
    x, y = float(xy[0]), float(xy[1])
    c = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            c = not c
    return c


def room_of(meta, xy) -> str | None:
    """The scene file's room whose polygon holds `xy` (None: outside every room)."""
    for k, r in ((meta.get("room") or {}).get("rooms") or {}).items():
        if inside([tuple(v[:2]) for v in r["polygon"]], xy):
            return k
    return None


def _note(ctx, step: str, ok: bool = True, why: str = "", **info) -> Outcome:
    t = ctx.motor.t
    return ctx.note(step, Outcome(ok, why, t, t, info={k: pick._round(v) for k, v in info.items()}))


def _samples(corners) -> list:
    out = []
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / SAMPLE_M)) + 1)
        out += [a + u * (b - a) for u in np.linspace(0.0, 1.0, n)]
    return out


def handoff(world, nav_d, start, goals, poly):
    """The cell free on the runner's map, inside `poly` and connected there to one of `goals` (stance points, snapped
    to their nearest free cell as `navigate.path` does), that the doorway map's field from `start` reaches first; None
    when there is none."""
    occ, nav = world.occ, world.nav
    mask = np.zeros_like(occ.free)
    for g in goals:
        c = occ.clip_cell(occ.cell_of(g))
        if not occ.free[c[0], c[1]]:
            c = occ.nearest_free(g)
        mask[c[0], c[1]] = True
    if not nav.set_goal_mask(mask) or not nav_d.set_goal_point(start):
        return None
    d35, d30 = nav.dist, nav_d.dist
    ii, jj = np.nonzero(occ.free & (d35 < UNREACHABLE * 0.5) & (d30 < UNREACHABLE * 0.5))
    for i in np.argsort(d30[ii, jj], kind="stable"):
        xy = occ.xy_of(np.array([ii[i], jj[i]]))
        if inside(poly, xy):
            return xy
    return None


def walk(ctx, key: str) -> Outcome | None:
    """The doorway walk when this start needs it (see the module doc), else None."""
    w, m = ctx.world, ctx.motor
    p, _, _ = pick.object_frame(w, key)
    start = m.pelvis()[0][:2]
    room, here = room_of(w.meta, p[:2]), room_of(w.meta, start)
    if room is None or here is None or room == here:
        return None
    surf = w.meta["surfaces"][w.meta["objects"][key]["surface"]]
    min_d = pick.TOP_MIN_D_M if pick.is_flat(w, key) else pick.STANCE_MIN_D_M
    inroom = [c for c in pick.stance_candidates(w, p[:2], surf, min_d) if room_of(w.meta, c[2]) == room]
    if not inroom or any(nav_path(w.occ, w.nav, start, c[2]) is not None for c in inroom):
        return None
    occ_d, nav_d = door_maps(w)
    poly = [tuple(v[:2]) for v in w.meta["room"]["rooms"][room]["polygon"]]
    hand = handoff(w, nav_d, start, [c[2] for c in inroom], poly)
    corners = None if hand is None else nav_path(occ_d, nav_d, start, hand)
    st = None if corners is None else pick.choose_stance(w, key, hand, min_d)
    st_room = None if st is None else room_of(w.meta, st[2])
    if st_room != room:
        _note(ctx, "door", False, "no doorway walk to a stance in the object's room; the normal skill runs", room=room,
              spawn_room=here, handoff=hand, stance_room=st_room)  # fmt: skip
        return None
    pts = _samples(corners)
    cl = [pick.clearance(w.occ, q) for q in pts]
    i = int(np.argmin(cl))
    leg = corners[-1] - corners[-2]
    yaw = float(np.arctan2(leg[1], leg[0]))
    k0 = len(m.trace)
    out = ctx.note("door", walk_path(m, occ_d, nav_d, hand, yaw), room=room, spawn_room=here, stance=st[2],
                   stance_kind=st[4], handoff=hand, handoff_clear=pick.clearance(w.occ, hand),
                   corners=np.round(np.array(corners), 3), door=np.round(pts[i], 3), min_clear=round(cl[i], 3),
                   pelvis_z_min=round(min((float(r["robot"][2]) for r in m.trace[k0:]), default=float("nan")), 3))  # fmt: skip
    if not out.ok or m.fell_at is not None:
        return out
    here_xy = m.pelvis()[0][:2]
    st = pick.choose_stance(w, key, here_xy, min_d)
    st_room = None if st is None else room_of(w.meta, st[2])
    if st_room != room:
        return _note(ctx, "door_stance", False, "no stance in the object's room after the doorway", room=room,
                     stance_room=st_room, at=here_xy)  # fmt: skip
    _note(ctx, "door_stance", stance=st[2], stance_room=st_room, kind=st[4], at=here_xy)
    occ_s, nav_s = approach_maps(w)
    goal = st[2] if w.occ.is_free(st[2]) else w.occ.xy_of(w.occ.nearest_free(st[2]))
    corners = nav_path(occ_s, nav_s, here_xy, goal)
    if corners is None:
        _note(ctx, "approach", False, "no path clear of the floor objects; the normal skill walks from here", goal=goal)
        return Outcome(True, "", m.t, m.t)
    leg = corners[-1] - corners[-2] if len(corners) > 1 else np.array([np.cos(m.yaw()), np.sin(m.yaw())])
    if np.linalg.norm(leg) < 1e-6:
        leg = np.array([np.cos(m.yaw()), np.sin(m.yaw())])
    near = min((float(np.linalg.norm(np.maximum(np.abs(R.rot_z(a)[:2, :2].T @ (q - c)) - h, 0.0)))
                for q in _samples(corners) for c, h, a in floor_statics(w.meta)), default=float("nan"))  # fmt: skip
    out = ctx.note("approach", walk_path(m, occ_s, nav_s, goal, float(np.arctan2(leg[1], leg[0]))), goal=goal,
                   corners=np.round(np.array(corners), 3), min_static_m=round(near, 3))  # fmt: skip
    out.ok = m.fell_at is None  # a stop outside tolerance still hands over; only a fall ends the plan
    return out
