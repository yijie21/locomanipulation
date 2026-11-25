"""Skill `bring_near` (r1_held_near_single loop, iteration 0; the family's gap goal.near, rules 24): held(X) and
near(X, T, d) -- pick X, carry it to a stance from which it sits within d of T, and end still gripping it.

- The pick is held(X)'s own (`held.pick_held` over the PLAN_HELD pick, upright(X) passed through); the put-down's pick
  switches stay off.
- The hand-off is the put-down's (`place._handoff`: the fingertip check, a side grasp's creep budget, the low-pick
  pull-back and stand, the back-off, the stand-up below place.STAND_UP_BELOW_M), but its failures keep holding X
  (`_keep`) instead of setting it down: the set-down after a canister held 6.2 cm out at the fingertips released it on
  the cart's edge and it fell to the floor, a drop (s140004). Its T_est (`t_est`) is the lift, the hold, the back-off,
  the carry over the end stance's route planned from where the back-off will leave the robot, and the episode's final
  hold. Its pull-back runs with `follow`: one that ends outside the reach's 2 cm but brought the object along is not
  stuck (canister s140016 followed 18.4 of 18.5 cm and ended 3.0 cm off, remote s140017 5.2 cm off).
- The end stance (`plan_end`) is planned after the hand-off from X's offset from the pelvis (in the pelvis yaw frame)
  and X's underside, both measured then. Cells GRID_M apart within d + |offset| + REACH_PAD_M of T's centre xy (a
  surface's "xy" or an object's box centre: what near() measures to), free on the skill's map and pick.STANCE_CLEAR_M
  from the floor objects, each facing so the offset points at T. A cell is kept when the predicted X xy is within
  d - NEAR_MARGIN_M of T and not `barred`; the MAX_TRIES nearest the start are routed, the first with a route wins.
- When that finds none (`plan_stance`), the same search runs on a map at NARROW_RADIUS_M (`narrow_map`) with
  NARROW_MARGIN_M, and a route counts only if it passes `low_route`: wherever it comes within the skill's body radius
  of an occupied cell, every furniture hull and surface slab within that radius tops out at NARROW_TOP_MAX_M or lower,
  so only legs can meet them (rule 17). Both gaps around the sofa seat (0.53 and 0.60 m) close on the skill's 0.35 m
  map; the corridor north of the coffee table is 0.30 m clear between the coffee table (hull 0.20 m, top 0.45 m) and
  the red armchair (hull 0.41 m, seat 0.42 m), while the island's 0.88 m hull caught an arm at 0.26 m (rule 16).
- `barred` is required, not a nicety: predicates._held reads X inside a surface's footprint with its underside below
  that top + 0.02 m as resting on it, and the hand carrying X there runs it into the slab's edge. After the hand-off
  the canister's underside is at 0.85-0.92 m, the island's top at 0.896 m.
- The carry walks the route at place.CARRY_TURN_RATE and stops. The hand never opens after the lift: a failed check,
  no end stance, a failed walk or a lost grip ends the skill as it is (the episode's 2 s hold follows; an object let go
  off its support is a drop at the gate)."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter
from scipy.spatial import ConvexHull

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.primitives.navigator import Navigator
from locomani.primitives.navigator import build as build_navigation
from locomani.scene_discovery.scene_file import to_surface_frame
from locomani.tasks.scene import episode

from . import clearance, grasp_program, held, pick, place

NEAR_MARGIN_M = 0.15  # the predicted object xy at most d - this from T's xy (stops 1.5-2 cm off, turns 2-5 deg off)
BAR_OUTSIDE_M = 0.05  # the predicted object xy this far outside the flat footprint of ...
BAR_BELOW_M = 0.05  # ... every surface whose top is above the object's underside less this
GRID_M = 0.05  # end-stance candidate grid
REACH_PAD_M = 0.2  # candidates within d + |offset| + this of T's xy
MAX_TRIES = 60  # candidates routed, nearest the start first
NARROW_RADIUS_M = 0.28  # the fallback map's body radius when the skill's map has no end stance ...
NARROW_MARGIN_M = 0.05  # ... its NEAR_MARGIN_M (6 narrow rounds ended -0.18 to +0.014 m off their plan) ...
NARROW_TOP_MAX_M = 0.46  # ... its route kept only where the hulls and slabs within the skill's body radius top out at
#                          this or lower (coffee table top 0.454 m; the island's 0.876 m hull caught an arm, rule 16)
HULL_CONTYPE = 4  # the furniture collision hulls' contype (they collide with the robot only; plan 7.3)

_NARROW: dict = {}  # (id(meta), radius) -> (meta, occupancy, navigator, floor boxes, top within the body radius)


PICK_SIDE_AHEAD_M = 0.46  # pick.SIDE_AHEAD_M for the whole carry-to (r1_held_near_single's set; held(X)'s stays None)


def target_xy(world, target: str) -> np.ndarray:
    """What near() measures to: a surface's "xy", an object's box centre."""
    if target in world.meta["surfaces"]:
        return np.asarray(world.meta["surfaces"][target]["xy"], float)
    return box_xy_bottom(world, target)[0]


def box_xy_bottom(world, key: str) -> tuple[np.ndarray, float]:
    """The object's box centre xy and underside z now (rotation ignored, as the predicates do)."""
    pos, _ = world.object_pose(key)
    box = world.meta["objects"][key]["box"]
    centre = np.asarray(box.get("centre") or (0, 0, 0), float)
    return pos[:2] + centre[:2], float(pos[2] + centre[2] - box["half_extent"][2])


def barred(world, xy, bottom: float) -> list[str]:
    """The surfaces whose flat footprint, grown by BAR_OUTSIDE_M, holds `xy` and whose top is above `bottom` -
    BAR_BELOW_M."""
    out = []
    for k, s in world.meta["surfaces"].items():
        half = np.asarray(s.get("flat_half_extent") or s["half_extent"], float) + BAR_OUTSIDE_M
        if np.all(np.abs(to_surface_frame(xy, s)) <= half) and float(s["top"]) > bottom - BAR_BELOW_M:
            out.append(k)
    return out


def plan_end(world, m, key: str, target: str, d: float, start=None, maps=None, margin: float | None = None,
             accept=None) -> tuple[dict | None, dict]:  # fmt: skip
    """(end or None, info): the end stance for carrying `key` near `target` (module doc), routed from `start` (default:
    the pelvis now) on `maps` ((occupancy, navigator, floor boxes); default: the skill's map), the predicted object at
    most d - `margin` (default NEAR_MARGIN_M) from T, its route passing `accept(corners)` when given. The offset and
    underside are measured now; a back-off moves the object with the pelvis, so they hold for a `start` behind it. end:
    stance, face, corners, dist (predicted object to T); info: off, bottom, n_free, n_near, n_clear (candidate counts),
    and n_refused (routes `accept` turned down) with `accept`."""
    pel = m.sim.d.qpos[:2].copy()
    start = pel if start is None else np.asarray(start, float)[:2]
    xy, bottom = box_xy_bottom(world, key)
    off = R.rot_z(m.yaw())[:2, :2].T @ (xy - pel)
    c = target_xy(world, target)
    occ, nav, boxes = maps or pick.nav_map(world)
    margin = NEAR_MARGIN_M if margin is None else margin
    reach = d + float(np.linalg.norm(off)) + REACH_PAD_M
    cands = []
    n_free = n_near = 0
    for x in np.arange(c[0] - reach, c[0] + reach + 1e-9, GRID_M):
        for y in np.arange(c[1] - reach, c[1] + reach + 1e-9, GRID_M):
            p = np.array([x, y])
            if not occ.is_free(p) or pick.floor_clearance(boxes, p) < pick.STANCE_CLEAR_M:
                continue
            n_free += 1
            v = c - p
            face = R.wrap(float(np.arctan2(v[1], v[0]) - np.arctan2(off[1], off[0])))
            o = p + R.rot_z(face)[:2, :2] @ off
            dist = float(np.linalg.norm(o - c))
            if dist > d - margin:
                continue
            n_near += 1
            if barred(world, o, bottom):
                continue
            cands.append((float(np.linalg.norm(p - start)), p, face, dist))
    info = dict(off=off, bottom=bottom, n_free=n_free, n_near=n_near, n_clear=len(cands))
    if accept is not None:
        info["n_refused"] = 0
    cands.sort(key=lambda z: z[0])
    for _, p, face, dist in cands[:MAX_TRIES]:
        corners = pick.route(occ, nav, start, p)
        if corners is not None and accept is not None and not accept(corners):
            info["n_refused"] += 1
            continue
        if corners is not None:
            return dict(stance=p, face=face, corners=corners, dist=dist), info
    return None, info


def hulls(world) -> list[tuple[np.ndarray, float]]:
    """(xy outline, top z) of each furniture collision hull (HULL_CONTYPE geoms): its vertices' xy and highest z."""
    sim = world.sim
    out = []
    for g in np.flatnonzero((sim.m.geom_contype == HULL_CONTYPE) & (sim.m.geom_conaffinity == 0)):
        if sim.m.geom_type[g] != 7:  # mjGEOM_MESH
            continue
        i = sim.m.geom_dataid[g]
        a, n = sim.m.mesh_vertadr[i], sim.m.mesh_vertnum[i]
        v = sim.d.geom_xpos[g] + sim.m.mesh_vert[a : a + n] @ sim.d.geom_xmat[g].reshape(3, 3).T
        out.append((v[:, :2], float(v[:, 2].max())))
    return out


def tops_within(world, occ, radius: float) -> np.ndarray:
    """Per cell of `occ`: the highest hull (`hulls`, its xy hull's cells grown by one) or surface slab (a surface's
    box as the map marks it, at its top) within `radius`; inf where an occupied cell is neither (walls, the room's
    obstacles)."""
    top = np.full(occ.occupied.shape, -np.inf)
    scratch = copy.copy(occ)
    for xy, z in hulls(world):
        scratch.occupied = np.zeros_like(occ.occupied)
        scratch.add_polygon(xy[ConvexHull(xy).vertices])
        cells = binary_dilation(scratch.occupied)
        top[cells] = np.maximum(top[cells], z)
    for s in world.meta["surfaces"].values():
        if s.get("half_extent") is None:
            continue
        scratch.occupied = np.zeros_like(occ.occupied)
        scratch.add_box(s["xy"], s["half_extent"], s.get("yaw", 0.0))
        top[scratch.occupied] = np.maximum(top[scratch.occupied], float(s["top"]))
    top[occ.occupied & np.isinf(top)] = np.inf
    k = int(np.ceil(radius / occ.res))
    u = np.arange(-k, k + 1)
    disk = np.hypot(*np.meshgrid(u, u, indexing="ij")) * occ.res <= radius
    return maximum_filter(top, footprint=disk, mode="nearest")


def narrow_map(world):
    """(occupancy, navigator, floor boxes, tops): `pick.nav_map`'s map at NARROW_RADIUS_M, and `tops_within` the
    skill's body radius on it; built once per scene and radius."""
    meta = world.meta
    hit = _NARROW.get((id(meta), NARROW_RADIUS_M))
    if hit is None or hit[0] is not meta:
        occ, _ = build_navigation(meta, body_radius=NARROW_RADIUS_M)
        boxes = pick.floor_boxes(meta, occ)
        for c, he, yaw in boxes:
            occ.add_box(c, he, yaw)
        occ.recompute()
        tops = tops_within(world, occ, world.occ.body_radius)
        hit = _NARROW[(id(meta), NARROW_RADIUS_M)] = (meta, occ, Navigator(occ), boxes, tops)
    return hit[1:]


def low_route(corners, wide, tops, out: dict) -> bool:
    """Whether every point of `corners` (sampled at half a cell) that the skill's map `wide` does not leave free has
    `tops` at most NARROW_TOP_MAX_M; out["top"] gets the highest such top."""
    high = 0.0
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (wide.res / 2))) + 1)
        for p in a + np.linspace(0.0, 1.0, n)[:, None] * (b - a):
            if wide.is_free(p):
                continue
            c = wide.clip_cell(wide.cell_of(p))
            high = max(high, float(tops[c[0], c[1]]))
    out["top"] = high
    return high <= NARROW_TOP_MAX_M


def plan_stance(world, m, key: str, target: str, d: float, start=None) -> tuple[dict | None, dict]:
    """`plan_end` on the skill's map; when that finds none, on `narrow_map` at NARROW_MARGIN_M with `low_route`
    (end["narrow"], end["top"]; info["narrow_n"]: that pass's counts)."""
    end, info = plan_end(world, m, key, target, d, start)
    if end is not None:
        return end, info
    occ, nav, boxes, tops = narrow_map(world)
    wide = pick.nav_map(world)[0]
    seen = {}
    end, narrow = plan_end(world, m, key, target, d, start, (occ, nav, boxes), NARROW_MARGIN_M,
                           lambda corners: low_route(corners, wide, tops, seen))  # fmt: skip
    info["narrow_n"] = {k: v for k, v in narrow.items() if k.startswith("n_")}
    if end is not None:
        end.update(narrow=True, top=seen["top"])
    return end, info


def t_est(cfg, pel, yaw: float, end: dict | None, c) -> float:
    """T_est (s) for `place._handoff`: the pick's lift and hold, the back-off from `pel` facing `yaw`, the carry over
    the end's corners at place.CARRY_TURN_RATE (no end: one straight leg to T's xy `c`), the episode's final hold."""
    back = np.asarray(pel, float)[:2] - place.BACK_AFTER_PICK_M * np.array([np.cos(yaw), np.sin(yaw)])
    t = grasp_program.SIMPLE["lift_seconds"] + grasp_program.HOLD_S
    t += place.walk_seconds(cfg, pel, yaw, [pel, back], yaw, fixed=True)
    corners, face = (end["corners"], end["face"]) if end is not None else ([back, c], None)
    t += place.walk_seconds(dataclasses.replace(cfg, turn_rate=place.CARRY_TURN_RATE), back, yaw, corners, face)
    return t + episode.HOLD_S


def _keep(ctx, key: str, why: str, lifted=(), home: dict | None = None) -> Outcome:
    """`place._handoff`'s failure for bring_near: `why`, nothing set down, no hand opened."""
    return Outcome(False, why, ctx.motor.t, ctx.motor.t)


def bring_near(ctx, key: str, target: str, d: float, skill, upright: bool = False, side: str = "right") -> Outcome:
    """Pick `key` with `skill` through held(X)'s `held.pick_held`, hand off, carry it to the end stance (`plan_end`) and
    stop, still gripping it. ok iff the hand that picked it still grips it at the end; see the module doc."""
    world, m = ctx.world, ctx.motor
    out = held.pick_held(ctx, key, skill, upright)
    if not out.ok or key not in world.free:
        return out
    body = world.free[key]
    source = world.meta["objects"][key]["surface"]
    hands = sorted(m.gripping(body))
    if not hands:
        return _keep(ctx, key, "pick failed: not gripped after the pick")
    side = hands[0]
    pel, yaw = m.sim.d.qpos[:2].copy(), m.yaw()
    back = pel - place.BACK_AFTER_PICK_M * np.array([np.cos(yaw), np.sin(yaw)])
    pre, _ = plan_stance(world, m, key, target, d, start=back)
    ahead = dict(top=pick.is_top(world, key), t_est=t_est(m.cfg, pel, yaw, pre, target_xy(world, target)))
    ctx.note("near_ahead", Outcome(True, "", m.t, m.t), t_est_s=round(ahead["t_est"], 1), routed=pre is not None)
    failed = place._handoff(ctx, key, side, (source,), ahead, None, _keep, follow=True)
    if failed is not None:
        return failed

    end, info = plan_stance(world, m, key, target, d)
    noted = dict(info, off=np.round(info["off"], 3).tolist(), bottom=round(info["bottom"], 3))
    if end is None:
        why = f"no end stance: {info['n_free']} free cells, {info['n_near']} near, {info['n_clear']} clear, none routed"
        n = info["narrow_n"]
        why += f"; at {NARROW_RADIUS_M} m {n['n_clear']} clear, {n['n_refused']} routes too high"
        return ctx.note("near_plan", Outcome(False, why, m.t, m.t), **noted)
    if end.get("narrow"):
        noted.update(narrow=True, radius=NARROW_RADIUS_M, top_passed=round(end["top"], 3))
    ctx.note("near_plan", Outcome(True, "", m.t, m.t), stance=end["stance"].round(3).tolist(),
             face_deg=round(float(np.degrees(end["face"])), 1), dist=round(end["dist"], 3),
             corners=[np.round(c, 2).tolist() for c in end["corners"]], **noted)  # fmt: skip
    k0 = len(m.trace)
    with place._carry_turns(m):
        walked = place.walk_corners(m, end["corners"], end["face"])
    xy, bottom = box_xy_bottom(world, key)
    ctx.note("carry", walked, obj_xy=xy.round(3).tolist(), obj_bottom=round(bottom, 3),
             obj_dist=round(float(np.linalg.norm(xy - target_xy(world, target))), 3), gripping=sorted(m.gripping(body)),
             sweep=clearance.sweep(world, m, side, k0))  # fmt: skip
    if m.fell_at is not None or not walked.ok:
        return Outcome(False, f"carry: {walked.why}", walked.t0, m.t)
    if side not in m.gripping(body):
        return Outcome(False, f"dropped during the carry: the {side} hand no longer grips it", walked.t0, m.t)
    return Outcome(True, "", out.t0, m.t)
