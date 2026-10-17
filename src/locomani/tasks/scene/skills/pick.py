"""Seed skill `pick` (TODO 27.2.2-27.2.3): a single-hand side grasp of an object on a surface. Walk around the
furniture to a stance at an approachable edge from which a side of the object fits the jaw (its own box, `grasp_yaw`),
reach the jaw to the pre-grasp with the motor layer (whole-body IK + servo), then the 65-object line's closed-loop hand
program (grasp_program.py: settle, gated entry with lateral correction and retry, close, lift, hold; plan D4, user
2026-10-16). v0 (motor reaches + grip, no closed loop) stays as `pick_simple` for comparison.

This is the pick_single loop's final incumbent (0109bde905a8: the code of iterations 1-3, the parameter set of every
accepted candidate folded into the constants here and in grasp_program.py), written back to the skill catalog for the
outer loop (plan §2 step 5, TODO 27.4): sealed pool 41/80.

The approach (`plan_approach`, loop iteration 1): the scene file's approach_edges first, the surface's other edges only
when those refuse or have no path; walks on the skill's own map, which adds the static floor objects the scene's map
leaves out (a step stool next to the utility cart and a trash bin at the kitchen island's end were behind both walk
falls); a stance too close to one of them, or without a path, shifts sideways. The reach to the pre-grasp rises outside
the surface's edge first when its fingertips would otherwise cross the edge below the top (they hooked the slab's lip
from below: pre-grasp misses of 10-30 cm).

A low flat object on a low surface (the TV remote on the coffee table, `is_top`, loop iteration 2) has no workable side
grasp: `pick_top` pinches it from above, wrist x down, closing across its narrow side.

Fallback plans (`plan_hand`, loop iteration 3) only where the planner above finds none for the hand: a top-down pinch
out to TOP_MAX_DEPTH_M, then the other hand from the mirrored stance (on the kitchen island's free west side every
right-hand stance stands against a bar stool; the left hand's stance is on the stool's far side). A fallback plan is
refused when the robot starts against a floor object (a mug start spawns with its wrist inside a trash bin's footprint
and falls on the first step whatever the plan).

A default-off switch the put-down (place.py, r1_on_single loop iteration 4) turns on around its pick, so held(X)'s pick
is unchanged: SIDE_AHEAD_M stands the pelvis farther back for a right-hand side grasp on a surface above
SIDE_AHEAD_TOP_MIN_M (`stand_ahead`); the left-hand fallback and the top-down pinch keep STAND_AHEAD_M.

The gripper's own shape (r1_held_single loop iteration 1, `grasp_pitch`): the Dex1 housing reaches 3.1-3.4 cm below the
jaw axis from 4.1-4.9 cm behind the jaw centre (robot.xml), so a level side grasp at the centre of an object standing
less than that above its support (the coffee-table and top-shelf candles, the supply bin: 2.5-2.6 cm) drives the
housing into the slab: the hand stalled at the coffee table's edge and rode on the 1.07-1.12 m shelves. Such a grasp
pitches the wrist GRASP_PITCH_DEG nose-down, which lifts the housing behind the jaw, and holds the jaw at least
`min_jaw` above the top. An object under another shelf stays level (pitched, the shelf plant was knocked off its shelf),
and its pre-grasp reach rises COVERED_VIA_OUT_M outside the edge instead of VIA_OUT_M (the wolf under the corner
display's top shelf; farther out everywhere, it broke pitched top-shelf grasps).

The pinch (r1_held_single loop iteration 2, `pick_pinch`): a pitched grasp whose pads would still pass over the object
(`needs_pinch`, the 1.5 cm car key on the 0.92 m counter: the closed jaw's fingers ended 0.4 cm above it, and where the
pads did reach down they closed on the key's tapered blade, its 1.5 cm fob beyond the fingertips). The Dex1 fingers'
underside tapers from 1.45 cm below the jaw axis at 0.4 cm ahead of the jaw centre to 0.2 cm at the tip, so
GRIPPER_LOW_POINTS' fingertip is 1.26 cm lower than the finger; with the true outline (PINCH_LOW_POINTS) a 25 deg pitch
brings the fingers' front 2.7 cm to PINCH_CLEAR_M above the top with the housing 0.7 cm higher. The jaw aims past the
part of the object the pads can overlap (`grip_centre`, from its collision mesh) and enters level, so the entry's stop
1-2.5 cm short leaves that part between the pads without lifting the jaw.

The yawed pass (iteration 2, `plan_hand`'s last pass): a side grasp up to FACE_TOLERANCE_YAWED_DEG off the facing,
with the stance placed from the wrist, only where no other pass plans, on surfaces up to YAWED_TOP_MAX_M.

The retry (iteration 4, `pick` over `pick_once`): a side grasp or pinch whose hand program failed without a fall,
leaving the object on its own support, under no other surface and between no fingers (`retry_ok`), backs out
(`back_out`) and plans and runs again from the object's pose now, up to RETRY_ATTEMPTS times. Part of what fails is
one attempt, not the start: a coffee-table candle knocked 5 cm in the settle by the controller's lunge after a 3.2 cm
pre-grasp, a top-shelf candle whose entry timed out 4.7 cm short; a second attempt finished both. Planning refusals
and the top-down pinch are not retried; a success never reaches the retry, so its steps are unchanged."""

from __future__ import annotations

import copy

import numpy as np
from scipy.ndimage import distance_transform_edt

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import SIDES, Outcome
from locomani.primitives.navigator import Navigator
from locomani.primitives.navigator import build as build_navigation
from locomani.tasks.scene.navigate import path as nav_path

from . import cover, grasp_program

JAW_SPAN_M = 0.092  # widest side the open Dex1 jaw (9.5 cm) closes across (0.085 until the pick_single loop's set)
STAND_AHEAD_M = 0.40  # object this far ahead of the pelvis ...
STAND_SIDE_M = 0.12  # ... and this far toward the grasping hand
PREGRASP_BACK_M = 0.12  # jaw centre short of the object centre before the entry
ENTRY_PAST_M = 0.01
ENTRY_SPEED = 0.08
LIFT_M = 0.10
FACE_TOLERANCE_DEG = 40.0
MAX_DEPTH_M = 0.35  # object centre at most this far in from the edge the robot stands at (0.30 until the loop's set)
NORMALS = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}
FLOOR_BOTTOM_M = 0.30  # the skill's map adds static floor objects whose bottom is below this ...
FLOOR_MAPPED_MAX = 0.5  # ... and whose footprint the scene's map covers less than this (furniture is on it already)
STANCE_CLEAR_M = 0.12  # pelvis at least this far from those footprints
STAND_SIDE_RETRY_M = (0.22, 0.30)  # lateral offsets tried after STAND_SIDE_M when a stance is not clear or unreachable
OTHER_EDGE_CLEAR_M = 0.25  # a stance at an edge the scene file does not list: this clear of all but its own furniture
VIA_OUT_M = 0.08  # a reach whose fingertips would cross the edge below the top rises this far outside the edge first
COVERED_VIA_OUT_M = 0.20  # ... this far for an object under another surface (cover.overhead: the wolf)
VIA_JAW_ABOVE_TOP_M = 0.03  # only reaches that start with the jaw below the top + this
FINGERTIP_M = 0.033  # fingertips ahead of the jaw centre along the wrist x
TOP_FLAT_M = 0.02  # an object with box half-height below this on a surface below TOP_SURFACE_MAX_M: top-down pinch
TOP_SURFACE_MAX_M = 0.6  # (the wallet's 1.12 m shelf: the IK sits at its limit, 48 deg off, iteration 1)
TOP_TIP_CLEAR_M = 0.004  # top-down pinch: fingertips this far above the surface top
TOP_ABOVE_M = 0.10  # ... from a pre-grasp this far above the grasp
TOP_DESCENT_SPEED = 0.05
TOP_MAX_DEPTH_M = 0.39  # a top-down pinch's depth limit (fallback pass; MAX_DEPTH_M stays the side grasps')
SPAWN_CLEAR_M = 0.08  # a fallback plan needs the start pelvis this far from every floor object footprint
SIDE_AHEAD_M = None  # default-off switch (place.py's pick): object this far ahead for a right-hand side grasp ...
SIDE_AHEAD_TOP_MIN_M = 0.6  # ... on a surface whose top is above this (None: STAND_AHEAD_M, held(X)'s pick)
GRIPPER_LOW_POINTS = ((-0.049, -0.034), (-0.0407, -0.031), (0.0333, -0.0145))  # (x along the approach, z up) in the
# jaw frame, robot.xml: the housing cylinder's and the housing mesh's front-bottom, the fingertip's front-bottom
GRASP_CLEAR_M = 0.005  # a pitched side grasp keeps those points this far above the support top ...
GRASP_PITCH_DEG = 15.0  # ... pitching the wrist this far nose-down when level they would be below it
PINCH_OVERLAP_M = 0.005  # a pitched grasp whose pads' lower edge at the object centre overlaps its top less: pinch
PINCH_PITCH_DEG = 25.0  # the pinch's wrist pitch nose-down (at 20 deg the housing cleared the top by 0.13 cm more
# than the fingers; delivered at 17 deg and 0.4 cm low, it rode on the counter and tipped the wrist level)
PINCH_X_M = -0.015  # the grippable part's centre this far ahead of the pinch's jaw target along the approach (the
# entry stops 1-2.5 cm short; at 0 the key's fob ended half past the fingertips and slipped out in the lift)
PINCH_DROP_M = 0.04  # the pinch's pre-grasp reach ends this far above its level entry line, then drops onto it
PINCH_CLEAR_M = 0.003  # the pinch keeps PINCH_LOW_POINTS this far above the support top (GRASP_CLEAR_M's role; at
# 0.005 1 of the 4 planned diag keys was held, at 0.003 and at 0.0015 3 of 4: iteration 2's smoke runs)
PINCH_LOW_POINTS = ((-0.0441, -0.031), (-0.0407, -0.0272), (-0.078, -0.034), (0.0041, -0.0145), (0.0064, -0.014),
                    (0.0126, -0.0114), (0.0321, -0.003), (0.0333, -0.0019))  # fmt: skip
# the lower outline (x, z) of the Dex1 collision hulls in the jaw frame (robot.xml): the housing mesh's bottom-front
# corner, the housing cylinder's, and the finger's underside, flat to x 0.4 cm then tapering to the tip
WRIST_JAW_M = 0.1515  # the jaw centre ahead of the wrist along its x
FACE_TOLERANCE_YAWED_DEG = 70.0  # the yawed fallback pass's tolerance, its stance placed from the wrist ...
YAWED_TOP_MAX_M = 1.0  # ... on a surface whose top is at most this (both 1.12 m shelf supply bins fell, 41 deg yawed)
RETRY_ATTEMPTS = 1  # a failed attempt `retry_ok` admits runs again this many times at most, after `back_out`:
RETRY_OUT_M = 0.15  # the jaw pulls this far straight back, level, the robot steps RETRY_BACK_M back when that point
RETRY_BACK_M = 0.25  # is free on the skill's map, and the arm returns to the pose it walked with
_NAV: dict = {}  # id(meta) -> (meta, occupancy, navigator, floor boxes): one map per scene and process
_CLEAR: dict = {}  # (id(occupancy), furniture) -> (occupancy, clearance grid without that furniture's surfaces)


def facings(surface: dict, xy, edges=None) -> list[tuple[float, float]]:
    """(robot yaw into the surface, that edge's distance to `xy`) per approachable edge (or per edge of `edges`),
    nearest first."""
    yaw_s = float(surface.get("yaw", 0.0))
    Rs = R.rot_z(yaw_s)[:2, :2]
    local = Rs.T @ (np.asarray(xy, float)[:2] - np.asarray(surface["xy"], float))
    half = np.asarray(surface["half_extent"], float)
    out = []
    for e in edges or surface.get("approach_edges") or NORMALS:
        n = np.asarray(NORMALS[e])
        w = Rs @ n
        out.append((float(np.arctan2(-w[1], -w[0])), float(half[0 if "x" in e else 1] - n @ local)))
    return sorted(out, key=lambda f: f[1])


def grasp_yaw(quat, half, face: float, tolerance_deg: float | None = None) -> float | None:
    """The approach yaw (wrist x) closing the jaw across a side of the object's box that fits it, nearest `face`,
    within `tolerance_deg` of it (default FACE_TOLERANCE_DEG)."""
    yaw_o = R.yaw_of(quat)
    fit_y, fit_x = 2 * half[1] <= JAW_SPAN_M, 2 * half[0] <= JAW_SPAN_M  # closing across local y / local x
    if fit_x and fit_y:
        return face
    cands = []
    if fit_y:
        cands += [yaw_o, yaw_o + np.pi]
    if fit_x:
        cands += [yaw_o + np.pi / 2, yaw_o - np.pi / 2]
    if not cands:
        return None
    best = min(cands, key=lambda a: abs(R.wrap(a - face)))
    limit = FACE_TOLERANCE_DEG if tolerance_deg is None else tolerance_deg
    return float(best) if abs(R.wrap(best - face)) <= np.radians(limit) else None


def pick_simple(ctx, key: str, side: str = "right") -> Outcome:
    world, m = ctx.world, ctx.motor
    plan = approach(ctx, key, side)
    if not plan.ok:
        return plan
    side = plan.info["side"]
    pos, quat = world.object_pose(key)  # the object as it is now
    Rg = R.rot_z(plan.info["psi"])
    out = ctx.note("pregrasp", m.reach(side, pos - Rg @ np.array([PREGRASP_BACK_M, 0.0, 0.0]), Rg))
    if m.fell_at is not None:
        return out
    out = ctx.note("entry", m.reach(side, pos + Rg @ np.array([ENTRY_PAST_M, 0.0, 0.0]), Rg, speed=ENTRY_SPEED))
    if m.fell_at is not None:
        return out
    out = ctx.note("grip", m.grip(side))
    if not out.ok:
        return out
    out = ctx.note("lift", m.reach(side, m.jaw_centre(side)[0] + np.array([0.0, 0.0, LIFT_M]), Rg, speed=ENTRY_SPEED))
    held = world.free[key] in m.between_fingers(side)
    return Outcome(held, "" if held else "not between the fingers after the lift", out.t0, out.t1)


def pick(ctx, key: str, side: str = "right") -> Outcome:
    """`pick_once`; after a failed attempt that `retry_ok` admits, a "retry" step, `back_out` and `pick_once` again
    (planned anew from where the robot and the object are), up to RETRY_ATTEMPTS times."""
    rest: dict = {}
    out = pick_once(ctx, key, side, rest)
    for k in range(int(RETRY_ATTEMPTS)):
        if not retry_ok(ctx, key, out, rest):
            break
        m = ctx.motor
        ctx.note("retry", Outcome(True, "", m.t, m.t), attempt=k + 1, after=out.why)
        back = back_out(ctx, rest)
        if m.fell_at is not None:
            return back
        rest = {}
        out = pick_once(ctx, key, side, rest)
    return out


def retry_ok(ctx, key: str, out: Outcome, rest: dict) -> bool:
    """Whether `pick` retries the attempt that ended with `out`: it failed in the hand program (`rest` holds the
    walking pose pick_once records just before the pre-grasp: planning refusals, failed walks and the top-down pinch
    never record it), the robot has not fallen, and X is free, still resting on its own scene-file support
    (cover.resting_on), under no other surface (cover.overhead) and between no fingers. An object under a shelf is
    not retried: on the diag pool the second entry knocked 3 of the 5 retried shelf plants off their shelf (their hand
    stalls under the shelf above, pre-grasps 11-22 cm off) and pushed all 5 books again, while the 10 retries of
    objects under no shelf dropped nothing."""
    world, m = ctx.world, ctx.motor
    if out.ok or not rest or m.fell_at is not None or key not in world.free:
        return False
    on = cover.resting_on(world, key)
    if world.meta["objects"][key]["surface"] not in on or cover.overhead(world, key):
        return False
    return not any(world.free[key] in m.between_fingers(s) for s in SIDES)


def back_out(ctx, rest: dict) -> Outcome:
    """Out of a failed attempt: drop the hand's servo integrals and the hand program's override, pull the jaw
    RETRY_OUT_M straight back along the robot's facing, level, keeping its rotation ("back_out"), open the jaw
    ("open"), step RETRY_BACK_M back when that point is free on the skill's map ("step_back"; otherwise the step is
    skipped), then return the arm to the walking pose `rest` (relative to the pelvis now, "arm_rest"). The step comes
    before the arm: lowered first, a jaw still over the slab after the pull came down on it (top-shelf candle s120035,
    the wrist held 30 cm above its target for 4.8 s on the 1.1 m shelf until the step carried it off). The last step's
    outcome."""
    world, m = ctx.world, ctx.motor
    side = rest["side"]
    for k in ("pos", "turn"):
        m.servo.pop((k, side), None)
    m.hand(side)
    jaw, rot = m.jaw_centre(side)
    yaw = m.yaw()
    out = ctx.note("back_out", m.reach(side, jaw - RETRY_OUT_M * np.array([np.cos(yaw), np.sin(yaw), 0.0]), rot))
    if m.fell_at is not None:
        return out
    out = ctx.note("open", m.release(side))
    if m.fell_at is not None:
        return out
    occ, _, _ = nav_map(world)
    yaw = m.yaw()
    back = m.sim.d.qpos[:2] - RETRY_BACK_M * np.array([np.cos(yaw), np.sin(yaw)])
    if not occ.is_free(back):
        ctx.note("step_back", Outcome(True, "", m.t, m.t), skipped="not free on the skill's map")
    else:
        out = ctx.note("step_back", m.sidestep(forward=-RETRY_BACK_M))
        if m.fell_at is not None:
            return out
    pel, _ = m.pelvis()
    Rz = R.rot_z(m.yaw())
    return ctx.note("arm_rest", m.reach(side, pel + Rz @ rest["jaw"], Rz @ rest["rot"]))


def pick_once(ctx, key: str, side: str = "right", rest: dict | None = None) -> Outcome:
    """Walk, reach the pre-grasp, then the ported closed-loop hand program (grasp_program.run), the wrist pitched and
    the jaw raised for a short object (`grasp_pitch`); a low flat object (`is_top`) is pinched from above instead
    (`pick_top`). A side grasp or pinch records the hand and its walking jaw pose (relative to the pelvis, in its yaw
    frame) in `rest` before the pre-grasp, for `back_out`."""
    world, m = ctx.world, ctx.motor
    plan = approach(ctx, key, side)
    if not plan.ok:
        return plan
    side = plan.info["side"]
    if is_top(world, key):
        return pick_top(ctx, key, side, plan)
    pel, _ = m.pelvis()
    Rz = R.rot_z(m.yaw())
    jaw0, rot0 = m.jaw_centre(side)
    (rest if rest is not None else {}).update(side=side, jaw=Rz.T @ (jaw0 - pel), rot=Rz.T @ rot0)
    pos, quat = world.object_pose(key)
    psi = plan.info["psi"]
    Rg = R.rot_z(psi)
    pitch, rise = grasp_pitch(world, key, pos)
    if pitch and needs_pinch(world, key, pos, pitch, rise):
        return pick_pinch(ctx, key, side, plan)
    pitched = {}
    if pitch:
        Rg = Rg @ R.rot_y(pitch)
        pitched = dict(pitch_deg=round(float(np.degrees(pitch)), 1), jaw_rise_cm=round(100 * rise, 2))
    up = np.array([0.0, 0.0, rise])
    pre = pos + up - Rg @ np.array([grasp_program.PREGRASP_BACK, 0.0, 0.0])
    via = edge_via(ctx, key, side, plan, pre)
    out = ctx.note("pregrasp", m.reach(side, pre, Rg, via=via), via=[np.round(v, 3).tolist() for v in via], **pitched)
    if m.fell_at is not None:
        return out
    pos, _ = world.object_pose(key)
    return grasp_program.run(
        ctx, key, side, Rg, pos + up, span(quat, world.meta["objects"][key]["box"]["half_extent"], psi)
    )


def min_jaw(pitch: float, points=None, clear: float | None = None) -> float:
    """The least jaw-centre height (m) above the support top of a side grasp pitched nose-down by `pitch` (rad) that
    keeps every point of `points` (default GRIPPER_LOW_POINTS) `clear` (default GRASP_CLEAR_M) above the top
    (rot_y(pitch) puts the point (x, z) at -x sin + z cos): 3.9 cm level, 2.76 cm at 15 deg; with PINCH_LOW_POINTS 2.13
    cm at 25 deg."""
    s, c = np.sin(pitch), np.cos(pitch)
    low = float(max(x * s - z * c for x, z in (GRIPPER_LOW_POINTS if points is None else points)))
    return low + (GRASP_CLEAR_M if clear is None else clear)


def grasp_pitch(world, key: str, pos) -> tuple[float, float]:
    """(pitch nose-down (rad), jaw rise (m)) of a side grasp at the object centre `pos`: (0, 0) unless the level jaw
    there would put the gripper below the support top (centre less than min_jaw(0) - GRASP_CLEAR_M above it) and no
    surface covers the object (`cover.overhead`); then GRASP_PITCH_DEG, the jaw raised to the top +
    min_jaw(GRASP_PITCH_DEG) where the centre is lower."""
    top = float(world.meta["surfaces"][world.meta["objects"][key]["surface"]]["top"])
    h = float(pos[2]) - top
    if h >= min_jaw(0.0) - GRASP_CLEAR_M or cover.overhead(world, key):
        return 0.0, 0.0
    pitch = float(np.radians(GRASP_PITCH_DEG))
    return pitch, max(0.0, min_jaw(pitch) - h)


def needs_pinch(world, key: str, pos, pitch: float, rise: float) -> bool:
    """A pitched side grasp (`grasp_pitch`'s pitch and rise at the object centre `pos`) whose pads' lower edge there
    (grasp_program.PAD_Z below the jaw axis) would overlap the object's top by less than PINCH_OVERLAP_M: the car key
    (1.5 cm high) by 0.1 cm, the wallet by 1.35 cm, the candles and the bin by 3.6-3.9 cm."""
    o = world.meta["objects"][key]
    top_obj = float(pos[2]) + float(o["box"]["half_extent"][2])
    pad_low = float(pos[2]) + rise - grasp_program.PAD_Z * np.cos(pitch)
    return top_obj - pad_low < PINCH_OVERLAP_M


def object_vertices(m, body: int) -> np.ndarray:
    """The world positions now of the vertices of `body`'s collision meshes (the simulator collides their hulls)."""
    sim = m.sim
    out = []
    for g in np.flatnonzero(sim.m.geom_bodyid == body):
        if sim.m.geom_type[g] != 7:  # mjGEOM_MESH
            continue
        i = sim.m.geom_dataid[g]
        a, n = sim.m.mesh_vertadr[i], sim.m.mesh_vertnum[i]
        out.append(sim.d.geom_xpos[g] + sim.m.mesh_vert[a : a + n] @ sim.d.geom_xmat[g].reshape(3, 3).T)
    return np.concatenate(out) if out else np.zeros((0, 3))


def grip_centre(verts, pos, top: float, approach) -> np.ndarray:
    """The xy centre, along `approach` and across it, of the vertices at least GRASP_CLEAR_M + PINCH_OVERLAP_M above
    the support top: the part the pads can overlap. The car key's 1.5 cm fob fills one end of its box (its blade
    tapers to 0.4 cm), 2.4 cm from the box centre. Without such vertices, the object centre `pos`."""
    v = np.asarray(verts, float)
    v = v[v[:, 2] >= top + GRASP_CLEAR_M + PINCH_OVERLAP_M] if len(v) else v
    if not len(v):
        return np.asarray(pos, float)[:2].copy()
    d = np.asarray(approach, float)[:2]
    n = np.array([-d[1], d[0]])
    s, t = v[:, :2] @ d, v[:, :2] @ n
    return 0.5 * (s.min() + s.max()) * d + 0.5 * (t.min() + t.max()) * n


def pick_pinch(ctx, key: str, side: str, plan: Outcome) -> Outcome:
    """A pinch with the fingers' tapered underside (`needs_pinch`; the car key on the 0.92 m counter): the wrist
    pitched PINCH_PITCH_DEG nose-down with every PINCH_LOW_POINTS point PINCH_CLEAR_M above the support top, the jaw
    aimed PINCH_X_M short of the grippable part's centre (`grip_centre`), the pre-grasp reach PINCH_DROP_M above the
    level entry line, a drop onto it, then grasp_program.run entering level (entry_frame) so the jaw keeps its height
    wherever the entry stops."""
    world, m = ctx.world, ctx.motor
    pos, quat = world.object_pose(key)
    o = world.meta["objects"][key]
    top = float(world.meta["surfaces"][o["surface"]]["top"])
    psi = plan.info["psi"]
    level = R.rot_z(psi)
    pitch = float(np.radians(PINCH_PITCH_DEG))
    Rg = level @ R.rot_y(pitch)
    d = level[:, 0]
    centre = grip_centre(object_vertices(m, world.free[key]), pos, top, d)
    tool = np.r_[centre - PINCH_X_M * d[:2], top + min_jaw(pitch, PINCH_LOW_POINTS, PINCH_CLEAR_M)]
    pre = tool - grasp_program.PREGRASP_BACK * d
    above = pre + np.array([0.0, 0.0, PINCH_DROP_M])
    via = edge_via(ctx, key, side, plan, above)
    out = ctx.note("pregrasp", m.reach(side, above, Rg, via=via), via=[np.round(v, 3).tolist() for v in via],
                   pitch_deg=PINCH_PITCH_DEG, jaw_above_top_cm=round(100 * (tool[2] - top), 2),
                   grip_offset_cm=round(100 * float((centre - pos[:2]) @ d[:2]), 2))  # fmt: skip
    if m.fell_at is not None:
        return out
    out = ctx.note("drop", m.reach(side, pre, Rg))
    if m.fell_at is not None:
        return out
    now, _ = world.object_pose(key)
    tool = np.r_[tool[:2] + (now - pos)[:2], tool[2]]
    width = span(quat, o["box"]["half_extent"], psi)
    return grasp_program.run(ctx, key, side, Rg, tool, width, entry_frame=R.rot_y(-pitch))


def span(quat, half, psi: float) -> float:
    """The object's width across the jaw (closing along the wrist y) for approach yaw `psi`."""
    a = R.wrap(psi - R.yaw_of(quat))
    return float(2 * (abs(np.sin(a)) * half[0] + abs(np.cos(a)) * half[1]))


def floor_boxes(meta: dict, occ) -> list[tuple]:
    """(centre xy, half extent xy, yaw) of the static floor objects `occ` (the scene's map: walls and surfaces) leaves
    out: bottom below FLOOR_BOTTOM_M and less than FLOOR_MAPPED_MAX of the footprint occupied (in the ADT apartment the
    step stool, the trash bins, a speaker, a laundry basket and a cat bowl)."""
    out = []
    for k, o in meta["objects"].items():
        if o.get("free") or o.get("surface") is not None or k.startswith("Apartment"):
            continue
        p, he = np.asarray(o["init_pos"], float), np.asarray(o["box"]["half_extent"], float)
        if p[2] - he[2] > FLOOR_BOTTOM_M:
            continue
        yaw = R.yaw_of(np.asarray(o["init_quat"], float))
        Ro = R.rot_z(yaw)[:2, :2]
        grid = np.linspace(-1.0, 1.0, 9)
        cells = [occ.cell_of(p[:2] + Ro @ (he[:2] * [u, v])) for u in grid for v in grid]
        mapped = np.mean([not occ.in_bounds(c) or occ.occupied[c[0], c[1]] for c in cells])
        if mapped < FLOOR_MAPPED_MAX:
            out.append((p[:2], he[:2], yaw))
    return out


def nav_map(world):
    """(occupancy, navigator, floor boxes): the scene's map plus `floor_boxes`, built once per scene."""
    meta = world.meta
    hit = _NAV.get(id(meta))
    if hit is None or hit[0] is not meta:
        occ, _ = build_navigation(meta, body_radius=world.occ.body_radius)
        boxes = floor_boxes(meta, occ)
        for c, he, yaw in boxes:
            occ.add_box(c, he, yaw)
        occ.recompute()
        hit = _NAV[id(meta)] = (meta, occ, Navigator(occ), boxes)
    return hit[1:]


def floor_clearance(boxes, xy) -> float:
    """Distance from `xy` to the nearest floor box footprint (inf without any)."""
    d = np.inf
    for c, he, yaw in boxes:
        q = np.maximum(np.abs(R.rot_z(yaw)[:2, :2].T @ (np.asarray(xy, float)[:2] - c)) - he, 0.0)
        d = min(d, float(np.linalg.norm(q)))
    return d


def other_clearance(world, surface_key: str, xy) -> float:
    """Distance from `xy` to the nearest occupied cell of the skill's map that is not one of the surfaces of
    `surface_key`'s furniture (keys "<furniture>__<n>"): walls, other furniture, the floor objects."""
    occ, _, _ = nav_map(world)
    furniture = surface_key.split("__")[0]
    hit = _CLEAR.get((id(occ), furniture))
    if hit is None or hit[0] is not occ:
        own = copy.copy(occ)
        own.occupied = np.zeros_like(occ.occupied)
        for k, s in world.meta["surfaces"].items():
            if k.split("__")[0] == furniture and s.get("half_extent") is not None:
                own.add_box(s["xy"], s["half_extent"], s.get("yaw", 0.0))
        grid = distance_transform_edt(~(occ.occupied & ~own.occupied), sampling=occ.res)
        hit = _CLEAR[(id(occ), furniture)] = (occ, grid)
    c = occ.clip_cell(occ.cell_of(xy))
    return float(hit[1][c[0], c[1]])


def plan_approach(world, start_xy, key: str, side: str, max_depth: float | None = None, yawed: bool = False):
    """((face, edge depth, psi, stance xy, edge xy) or None, why): the first edge -- the surface's approach_edges
    nearest first, then its other edges nearest first -- from which a side of the object fits the jaw (`grasp_yaw`) with
    a stance that is STANCE_CLEAR_M clear of the floor objects (at another edge also OTHER_EDGE_CLEAR_M clear of all but
    the surface's own furniture: the kitchen island's unlisted west side stands against a bar stool) and has a path on
    the skill's map; the stance is STAND_SIDE_M toward the hand, else each of STAND_SIDE_RETRY_M. The utility cart's
    crowded top leaves the canister lying along one edge: only the edge at its end faces its narrow side. The object
    centre is at most `max_depth` (default MAX_DEPTH_M) in from the edge and `stand_ahead` ahead of the stance.
    `yawed` (plan_hand's last pass): a side within FACE_TOLERANCE_YAWED_DEG of the facing, and the stance placed from
    the wrist (WRIST_JAW_M behind the jaw along the approach yaw psi) instead of the jaw, which it equals at psi = face:
    at a large yaw the jaw-placed stance put the wrist 13-15 cm farther toward the hand, at or past the arm's reach.
    No yawed plan on a surface above YAWED_TOP_MAX_M."""
    obj = world.meta["objects"][key]
    surf = world.meta["surfaces"][obj["surface"]]
    pos, quat = world.object_pose(key)
    occ, nav, boxes = nav_map(world)
    listed = list(surf.get("approach_edges") or NORMALS)
    others = [e for e in NORMALS if e not in listed]
    sign = -1.0 if side == "right" else 1.0
    edges = [(f, dep, True) for f, dep in facings(surf, pos, listed)]
    edges += [(f, dep, False) for f, dep in (facings(surf, pos, others) if others else [])]
    top_pinch = is_top(world, key)
    fit = _nearest_closing_yaw if top_pinch else grasp_yaw
    why = "no side fits the jaw within reach of an approachable edge"
    if yawed and float(surf["top"]) > YAWED_TOP_MAX_M:
        return None, why
    max_depth = MAX_DEPTH_M if max_depth is None else max_depth
    ahead = stand_ahead(float(surf["top"]), side, top_pinch)
    for f, dep, is_listed in edges:
        if dep > max_depth:
            psi = None
        elif yawed and not top_pinch:
            psi = grasp_yaw(quat, obj["box"]["half_extent"], f, FACE_TOLERANCE_YAWED_DEG)
        else:
            psi = fit(quat, obj["box"]["half_extent"], f)
        if psi is None:
            continue
        why = "no clear stance with a path at an edge where a side fits the jaw"
        for lateral in (STAND_SIDE_M, *STAND_SIDE_RETRY_M):
            stance = pos[:2] - (R.rot_z(f) @ np.array([ahead, sign * lateral, 0.0]))[:2]
            if yawed:  # wrist = pos - WRIST_JAW_M u(psi); stance = wrist - R(f) [ahead - WRIST_JAW_M, sign lateral]
                stance = stance + WRIST_JAW_M * (np.array([np.cos(f), np.sin(f)]) - [np.cos(psi), np.sin(psi)])
            if floor_clearance(boxes, stance) < STANCE_CLEAR_M:
                continue
            if not is_listed and other_clearance(world, obj["surface"], stance) < OTHER_EDGE_CLEAR_M:
                continue
            if route(occ, nav, start_xy, stance) is None:
                continue
            return (f, dep, psi, stance, pos[:2] - dep * np.array([np.cos(f), np.sin(f)])), ""
    return None, why


def stand_ahead(top: float, side: str, top_pinch: bool) -> float:
    """How far ahead of the pick stance the object is: SIDE_AHEAD_M for a right-hand side grasp on a surface whose top is
    above SIDE_AHEAD_TOP_MIN_M when the switch is set (from 0.40 m the soap's reach on the bathroom sink counter missed
    the pre-grasp by 13-22 cm), else STAND_AHEAD_M."""
    if SIDE_AHEAD_M is None or side != "right" or top_pinch or top <= SIDE_AHEAD_TOP_MIN_M:
        return STAND_AHEAD_M
    return float(SIDE_AHEAD_M)


def plan_hand(world, start_xy, key: str, side: str):
    """(plan, hand, why): `plan_approach` for `side`; only when that finds none, the fallback passes -- the same hand
    with a top-down pinch out to TOP_MAX_DEPTH_M, then the other hand (stance mirrored), then for a side grasp the
    yawed pass (`plan_approach` yawed, `side` first: a mug whose narrow side is 58 deg off the facing) -- whose plan is
    refused when the start is less than SPAWN_CLEAR_M from a floor object. A start no pass plans keeps the first pass's
    why."""
    plan, why = plan_approach(world, start_xy, key, side)
    if plan is not None:
        return plan, side, ""
    depth = TOP_MAX_DEPTH_M if is_top(world, key) else MAX_DEPTH_M
    other = "left" if side == "right" else "right"
    passes = ([(side, depth, False)] if depth > MAX_DEPTH_M else []) + [(other, depth, False)]
    passes += [] if is_top(world, key) else [(side, MAX_DEPTH_M, True), (other, MAX_DEPTH_M, True)]
    for hand, limit, yawed in passes:
        fallback, _ = plan_approach(world, start_xy, key, hand, limit, yawed)
        if fallback is None:
            continue
        if floor_clearance(nav_map(world)[2], start_xy) < SPAWN_CLEAR_M:
            return None, hand, "start against a floor object"
        return fallback, hand, ""
    return None, side, why


def route(occ, nav, start_xy, goal_xy) -> list | None:
    """navigate.path, or -- when the start is inside a floor object's inflation, where the field cannot start -- a
    straight leg out to the nearest free cell first."""
    corners = nav_path(occ, nav, start_xy, goal_xy)
    if corners is None and not occ.is_free(start_xy):
        out = occ.xy_of(occ.nearest_free(start_xy))
        rest = nav_path(occ, nav, out, goal_xy)
        corners = None if rest is None else [np.asarray(start_xy, float)[:2], *rest]
    return corners


def walk_route(m, occ, nav, goal_xy, goal_yaw: float) -> Outcome:
    """navigate.walk_path on the skill's map, with `route`'s way out of an inflation."""
    corners = route(occ, nav, m.sim.d.qpos[:2].copy(), goal_xy)
    if corners is None:
        return Outcome(False, "no path on the occupancy map", m.t, m.t)
    out = None
    for k, c in enumerate(corners[1:], start=1):
        out = m.walk_to(c, goal_yaw if k == len(corners) - 1 else None)
        if not out.ok:
            return out
    out.info["legs"] = len(corners) - 1
    return out


def approach(ctx, key: str, side: str) -> Outcome:
    """Choose the hand, edge, grasp yaw and stance (`plan_hand`), walk there. info: side (the hand), psi (approach
    yaw), edge_depth_m, face (the robot's yaw into the surface), edge_xy (where the object's line along `face` crosses
    that edge)."""
    world, m = ctx.world, ctx.motor
    plan, side, why = plan_hand(world, m.sim.d.qpos[:2].copy(), key, side)
    if plan is None:
        return ctx.note("pick", Outcome(False, why, m.t, m.t))
    face, depth, psi, stance, edge = plan
    occ, nav, _ = nav_map(world)
    off = float(np.degrees(R.wrap(psi - face)))
    yawed = {} if is_top(world, key) or abs(off) <= FACE_TOLERANCE_DEG else dict(yaw_off_deg=round(off, 1))
    out = ctx.note("walk", walk_route(m, occ, nav, stance, face), edge_depth_m=round(depth, 3),
                   face_deg=round(float(np.degrees(face)), 1), stance=np.round(stance, 3), side=side,
                   **yawed)  # fmt: skip
    out.info.update(side=side, psi=psi, edge_depth_m=depth, face=face, edge_xy=edge)
    return out


def edge_via(ctx, key: str, side: str, plan: Outcome, pre) -> tuple:
    """The via point for the reach to the pre-grasp `pre`, or (): when the jaw starts below the surface top +
    VIA_JAW_ABOVE_TOP_M, the pre-grasp is above the top, and the straight leg's fingertips (the jaw + FINGERTIP_M along
    the starting wrist x) would cross the edge's plane below the top, go first to the jaw's xy moved VIA_OUT_M
    (COVERED_VIA_OUT_M for an object under another surface) outside the edge (or where it is, when farther out) at the
    pre-grasp's height; the last leg goes in level."""
    world, m = ctx.world, ctx.motor
    surf = world.meta["surfaces"][world.meta["objects"][key]["surface"]]
    top = float(surf["top"])
    jaw, rot = m.jaw_centre(side)
    if jaw[2] > top + VIA_JAW_ABOVE_TOP_M or pre[2] < top:
        return ()
    f, edge = plan.info["face"], plan.info["edge_xy"]
    n_in = np.array([np.cos(f), np.sin(f)])
    tips = [jaw + u * (pre - jaw) + FINGERTIP_M * rot[:, 0] for u in np.linspace(0.0, 1.0, 51)]
    if not any(float((t[:2] - edge) @ n_in) >= 0.0 and t[2] < top for t in tips):
        return ()
    s_jaw = float((jaw[:2] - edge) @ n_in)
    out = COVERED_VIA_OUT_M if cover.overhead(world, key) else VIA_OUT_M
    return (np.r_[jaw[:2] - n_in * max(0.0, s_jaw + out), pre[2]],)


def is_top(world, key: str) -> bool:
    """A low flat object on a low surface (box half-height below TOP_FLAT_M, surface top below TOP_SURFACE_MAX_M): the
    TV remote on the coffee table. Its only side that fits the jaw is its 3.6 cm width, so a side entry runs along its
    16.4 cm length and the palm meets its near end about 12 cm short."""
    o = world.meta["objects"][key]
    top = float(world.meta["surfaces"][o["surface"]]["top"])
    return o["box"]["half_extent"][2] < TOP_FLAT_M and top < TOP_SURFACE_MAX_M


def _nearest_closing_yaw(quat, half, face: float) -> float:
    """The wrist yaw of a top-down pinch closing across the object's local y when that fits the jaw, else across its
    local x, nearest `face` (no FACE_TOLERANCE_DEG: the hand comes from above)."""
    yaw_o = R.yaw_of(quat)
    cands = [yaw_o, yaw_o + np.pi] if 2 * half[1] <= JAW_SPAN_M else [yaw_o + np.pi / 2, yaw_o - np.pi / 2]
    return float(min(cands, key=lambda a: abs(R.wrap(a - face))))


def pick_top(ctx, key: str, side: str, plan: Outcome) -> Outcome:
    """Top-down pinch after `approach`: wrist x down, closing across the narrow side (plan psi); the pre-grasp
    TOP_ABOVE_M above the grasp (fingertips TOP_TIP_CLEAR_M above the surface top), descend, grip, lift."""
    world, m = ctx.world, ctx.motor
    pos, _ = world.object_pose(key)
    surf = world.meta["surfaces"][world.meta["objects"][key]["surface"]]
    Rg = R.rot_z(plan.info["psi"]) @ R.rot_y(np.pi / 2)
    grasp = np.r_[pos[:2], float(surf["top"]) + FINGERTIP_M + TOP_TIP_CLEAR_M]
    above = grasp + np.array([0.0, 0.0, TOP_ABOVE_M])
    via = edge_via(ctx, key, side, plan, above)
    out = ctx.note("pregrasp", m.reach(side, above, Rg, via=via), via=[np.round(v, 3).tolist() for v in via])
    if m.fell_at is not None:
        return out
    out = ctx.note("descend", m.reach(side, grasp, Rg, speed=TOP_DESCENT_SPEED))
    if m.fell_at is not None:
        return out
    out = ctx.note("grip", m.grip(side))
    if not out.ok:
        return out
    out = ctx.note("lift", m.reach(side, m.jaw_centre(side)[0] + np.array([0.0, 0.0, LIFT_M]), Rg, speed=ENTRY_SPEED))
    held = world.free[key] in m.between_fingers(side)
    return Outcome(held, "" if held else "not between the fingers after the lift", out.t0, out.t1)
