"""held(X) for one free object, right hand, from the motor layer and the scene's geometry (grasp_atomic iteration 1).

`pick` walks to a stance straight out from an edge of the object's support (for a side grasp whose hand corridor
another free body blocks, the nearest rotated approach line that is clear: `corridor_stance`), then either
- side grasp (H1): wrist x level from the right shoulder to the object, jaw at (object xy, max(centre z, top +
  JAW_ABOVE_TOP_M)); over-the-edge via points to a pre-grasp PRE_BACK_M behind, a slow entry, grip, lift; or
- top-down pinch (H3, `is_flat`): an object whose resting half-height is below FLAT_HALF_M (the TV remote, the wooden
  spoon) leaves the side grasp's pads above it, so the wrist points down, the jaw closes across the narrowest
  horizontal box axis, the fingertips TOP_TIP_CLEAR_M above the support top; pre-grasp TOP_ABOVE_M straight above,
  descend, grip, lift. A high pinch (`high_pinch`: a flat object resting under HIGH_REST_M tall on a support whose top
  is HIGH_SUPPORT_M or higher, the island spoon) takes the yaw of the symmetric jaw whose twist from the approach lies in
  [TWIST_LO_DEG, TWIST_LO_DEG + 180) and pinches lower and slower (SLOW_PINCH; iteration 9).
Both end with the carry-out: `held` counts any surface whose footprint holds the object and whose top is above its
underside as resting on it (a shelf overhead), so a lifted object under one is pulled back out of its footprint.

Geometry (plan section 7, Reviewer iteration 1): the jaw centre is R.OFF ahead of the wrist; the housing hangs 3.1-3.4
cm below the jaw axis, hence a level jaw sits at least 3.9 cm above the top; the fingers reach 3.3 cm along wrist x."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import motor as M
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome, expmap
from locomani.primitives.whole_body_ik import rotvec
from locomani.scene_discovery.scene_file import to_surface_frame
from locomani.tasks.scene.navigate import path as nav_path
from locomani.tasks.scene.navigate import walk_path
from locomani.tasks.scene.stage import stage, stance_check

from . import clutter, recipes

SIDE = "right"
# stance: the right shoulder on the support-edge normal through the object, the pelvis SHOULDER_Y_M to its left
STANCE_CLEAR_M = 0.17  # occupancy clearance at the pelvis
STANCE_MIN_D_M = 0.30  # shoulder to object, side grasps
STANCE_MAX_D_M = 1.00
REACH_MAX_M = 0.65  # a stance farther than this ranks after every nearer one (spoon s410015: 0.72 m, 9 cm off)
SHOULDER_Y_M = 0.10
NARROW_EDGE_FRAC = 0.6  # an edge shorter than this fraction of the support's longest edge is a narrow end ...
NARROW_PENALTY_M = 0.15  # ... whose stance counts this much farther (plant s410018 failed across the narrow end)
CONE_DEG = 30.0  # no normal stance: the nearest cell within this of a (non-narrow) edge normal, then the nearest
# floor statics (iteration 11, H2): the runner's map leaves out most fixed floor objects that are not furniture (of the
# inner half of the footprint it covers 0 % of the bins, the step stool and the speaker, 8 % of the laundry basket, 33 %
# of a cat bowl; 100 % of furniture and the other low objects), so the island's south-end normal put the pelvis inside
# the kitchen's bin (spoon 951007: walk 15 cm off, nothing between the fingers; 951005 fell on the last leg). Under
# FLOOR_STANCE a stance's pelvis (stance_candidates, corridor_stance) keeps STANCE_CLEAR_M from every low fixed floor
# box (FIT_FLOOR_BOTTOM_M, FIT_FLOOR_TOP_M) the map covers under FLOOR_MAPPED_FRAC (`floor_statics`); fit_stance's
# FIT_FLOOR_GUARD already keeps it from all of them, mapped or not
FLOOR_STANCE = 1  # 0: every stance is 1fe9d1e7dbdb's
FLOOR_MAPPED_FRAC = 0.60
# hand corridor (iteration 2, H2): the box the open hand sweeps from behind the pre-grasp to the object's near face must
# not hold another free body (vase s410007: VaseBig_2 10 cm in front, 6 cm off the line, ended between the fingers)
CORRIDOR_HALF_W_M = 0.06  # half the open jaw (4.75 cm) and a finger
CORRIDOR_BACK_M = 0.05  # behind the pre-grasp's jaw centre
CORRIDOR_BELOW_M = 0.034  # the housing's underside below the jaw axis (flat things under the hand do not block)
CORRIDOR_ABOVE_M = 0.08
CORRIDOR_TOL_M = 0.005  # a free body's box may reach this deep into it
CORRIDOR_ROT_DEG = 45.0  # blocked: the approach line rotated up to this either way about the object, ...
CORRIDOR_STEP_DEG = 5.0  # ... smallest rotation first; none clear keeps the stance
# yaw fit (iteration 9, H2): a tall non-container whose hull section across the jaw at the grasp height is wider than
# the open jaw on the chosen approach (recipes.features margin < 0: the soap's wide side, 9.4-12 cm) rotates the
# approach about the object to where the section fits with FIT_MARGIN_MIN_M..FIT_MARGIN_MAX_M a side, so
# recipes.tall_narrow takes it (s410012 held at +25 deg; s410002 needed -50 deg at 0.65 m and the entry stalled)
FIT_ON = 1  # 0: every round is c54e923f2234's
FIT_HEIGHT_MIN_M = 0.18  # resting this tall or taller (tall_narrow's SIGNATURE)
FIT_ROT_DEG = 35.0  # the approach line rotated up to this either way about the object, ...
FIT_STEP_DEG = 5.0  # ... in these steps
FIT_MARGIN_MIN_M = 0.0025  # a margin of 0.01-0.02 cm a side lost the recipe after the walk's heading error
FIT_MARGIN_MAX_M = 0.010  # tall_narrow's SIGNATURE margin_max_m
FIT_MAX_D_M = 0.58  # shoulder to object
# 1: the widest margin that fits, then the smaller rotation; 0: the smallest rotation that fits (fresh soaps: 0 stalled
# s902005's entry at 0.26 cm a side, 1 held it at 0.47 cm and held the other 4 the fit turns, as 0 did)
FIT_WIDEST = 1
FIT_FLOOR_GUARD = 1  # the fit's pelvis keeps STANCE_CLEAR_M from low fixed floor objects (box bottom and top below):
FIT_FLOOR_BOTTOM_M = 0.10
FIT_FLOOR_TOP_M = 0.60
# raised approach (iteration 10, H1): a side grasp of a non-container with nothing overhead on a support whose top lies
# in the hanging hand's band (rim.RAISE_TOP_MIN_M..RAISE_TOP_MAX_M, the 0.746 m dining table) takes rim's stance (arm
# strip and turn point clear of static colliders) and walks the last leg and the turn with the jaw raised over the top
# (rim.walk_raised): the hanging hand swept cans over in the final turn or shuffle (fresh 940011, 940031), and a cone
# stance between dining chairs trapped both hands on their backs (940001: walk 33.8 cm off; diag s410001 18.6 cm)
RAISED_ON = 1  # 0: every round is 011262a9020f's
RAISED_STOP_TOL_M = 0.06  # stance check after the raised walk (prototype walks: 1.1-2.7 cm, 0.5-6.7 deg off) ...
RAISED_YAW_TOL_DEG = 12.0
RAISED_REWALKS = 1  # ... a miss walks face-fixed to the stance again this many times, then gives up
# raise fit (iteration 11, H1): a level side grasp of a tall non-container (FIT_HEIGHT_MIN_M) whose hull section on the
# walked approach is still wider than the open jaw after fit_stance (no rotation within FIT_ROT_DEG fits: the soap's
# 10-11 cm side, margin -0.7..-0.8 cm, pushed and tipped by the level jaw) takes the grasp at the lowest height, at most
# RAISE_BELOW_TOP_M under the object's top, where the section fits with FIT_MARGIN_MIN_M..FIT_MARGIN_MAX_M a side (the
# bottle tapers to 8.1-8.9 cm at 6.5-8.5 cm above its centre, its shoulder above), so recipes.tall_narrow takes it there
RAISE_ON = 1  # 0: every round is 1fe9d1e7dbdb's
RAISE_STEP_M = 0.005
RAISE_BELOW_TOP_M = 0.035
# only within this shoulder-to-object distance after the walk (as FIT_MAX_D_M): tall_narrow's hand program keeps the
# body at the pregrasp pose, and from 0.58-0.65 m the raised jaw ended 0.9-2.7 cm short (one timed out), a fingertip
# grip under the bottle's shoulder that slid out in 3 of 6 fresh soaps, as many as the level grasp lost there; at
# 0.31-0.54 m it ended 0.15-1.2 cm off and held 11 of 11, where the level grasp held none of 10
RAISE_MAX_D_M = 0.58
# side grasp (H1)
JAW_ABOVE_TOP_M = 0.039
PRE_BACK_M = 0.12
VIA_BACK_M = 0.06
VIA_UP_M = 0.10  # the first via point rises to the top + this (and to the pre-grasp + VIA_OVER_M)
VIA_OVER_M = 0.04
ENTRY_PAST_M = 0.01
ENTRY_SPEED = 0.05
LIFT_M = 0.10
LIFT_SPEED = 0.08
# top-down pinch (H3)
FLAT_HALF_M = 0.02  # resting half-height below this: pinch from above
TOP_MIN_D_M = 0.40  # shoulder to object (pick_single's pinches stood 0.40 m off)
TOP_TIP_CLEAR_M = 0.004  # fingertips above the support top
FINGERTIP_M = 0.033  # jaw centre to fingertip along wrist x
TOP_ABOVE_M = 0.10
TOP_DESCENT_SPEED = 0.05
# high pinch (iteration 9, H1): on the island the right arm delivered the outward twist (-65..-90 deg from the approach)
# 63-98 deg off and the inward twist of the same symmetric jaw (+92..+112) 16-23 deg off; a full-force grip (jaw_close,
# about 20 N) squeezed the rounded 4.3-4.5 cm section out of the pads in the lift (Reviewer, s410025)
HIGH_SUPPORT_M = 0.8  # support top at or above this ...
HIGH_REST_M = 0.025  # ... and a flat object resting under this tall: a high pinch
TWIST_LO_DEG = -60.0  # high pinch: the twist window's low end (-90 is the low pinches' window)
SLOW_PINCH = 1  # 0: a high pinch with TOP_ABOVE_M, TOP_DESCENT_SPEED, LIFT_M, LIFT_SPEED
SLOW_TOP_ABOVE_M = 0.06
SLOW_DESCENT_SPEED = 0.025
SLOW_LIFT_M = 0.06
SLOW_LIFT_SPEED = 0.03  # also the carry-out's speed after a slow lift
# shallow grip (closing a few mm past the hull section): dropped (iteration 9, Engineer 1: the fingers stalled 4-6 mm
# short of the spoon on s410015 and s410025, and s410005 lost it in the lift; shallow 1/3 held, the full close 2/3)
# high pinch re-grasp (iteration 10, H2): the high pinch's lift runs through stage() with a check (the object between
# both fingers, risen PINCH_RISE_M); when it fails with the object not lifted (risen under PINCH_REST_M), the jaw opens,
# rises PINCH_UP_M, and if the settled object lies within PINCH_REST_M of its resting height, tilted under
# PINCH_TILT_DEG, pinches once more from its new pose (s410025: rolled in the lift, fell back flat; successes rose 3.7 cm)
PINCH_REGRASP = 1  # 0: every round is 011262a9020f's
PINCH_REGRASP_MAX = 1
PINCH_RISE_M = 0.03
PINCH_REST_M = 0.015
PINCH_TILT_DEG = 15.0
PINCH_UP_M = 0.04
# relative entry (selfwm 2026-10-29, docs/handoff/selfwm-20261029.md item 12): the side grasp's entry and the low
# pinch's descent steered by the grasp point as seen against the jaw as seen (grasp frame: x approach, y closing, z
# third), the object re-read at every look, in place of one IK leg to where the object was read before the pregrasp.
# The grasp point is the object's estimate plus the offset the plan chose before the pregrasp. Under
# REL_TOP_FROM_OBJECT the support top under the grasp is the object's own underside, not the map's surface.
# REL_MODE "look" (`rel_look`): whole-body IK legs (Motor.reach, so the torso helps at long reach) with stops to look:
# a leg to the stage, the fingertips REL_STAGE_CLEAR_M short of the object's near face; up to REL_CORRECTIONS legs
# that take out the y and z error seen there; a leg onto the grasp point as seen. REL_MODE "servo" (`rel_entry`):
# a hand-only servo, u = REL_GAIN x (grasp - seen jaw), no advance along x while y or z is off by more than its
# tolerance, each axis at most REL_VMAX, done after REL_INSIDE reads inside every tolerance or REL_MAX_S. With the body
# left at the pregrasp pose the arm alone did not get the jaw forward from 0.55-0.61 m (oranges, L0 2026-10-29: the
# command 5-8 cm ahead of a jaw that did not move), so under REL_BODY_FOLLOW the body's reference blends to a
# whole-body IK pose for the seen goal (as Motor.reach's legs do) while the servo drives the wrist; the goal is solved
# again when it moves REL_BODY_RESOLVE_M. The look mode's IK legs lost a high-shelf plant the servo held (s610061).
REL_ON = 1  # 0: every round is grasp_c11_view's
REL_MODE = "servo"
REL_BODY_FOLLOW = 0  # L0 2026-10-29: 37/50 on, 38 off (lead cap 0.15), no start won by it alone
REL_BODY_RESOLVE_M = 0.02
REL_BODY_SPEED = 0.10  # m/s: the blend to a new body pose lasts the goal's distance over this, ...
REL_BODY_MIN_S = 0.5  # ... at least this long
REL_STAGE_CLEAR_M = 0.015
REL_CORRECTIONS = 2
REL_TOP_FROM_OBJECT = 1
REL_PELVIS_Z = 0.0  # > 0: the side grasp's pregrasp and lift with the pelvis at this height (0: the IK's own choice)
REL_PERIOD_S = 0.1
REL_GAIN = 2.0  # 1/s
REL_VMAX = 0.15  # m/s per axis
REL_TOL_APPROACH_M = 0.012  # side grasp, along the approach
REL_TOL_DESCENT_M = 0.008  # pinch, along the wrist x (down)
REL_TOL_THIRD_M = 0.02
REL_TOL_CLOSE_MIN_M = 0.005  # closing axis: half the open jaw's spare width less REL_TOL_CLOSE_SLACK_M, within
REL_TOL_CLOSE_MAX_M = 0.02  # REL_TOL_CLOSE_MIN_M..REL_TOL_CLOSE_MAX_M
REL_TOL_CLOSE_SLACK_M = 0.003
REL_INSIDE = 2
# > 0: done only after REL_INSIDE reads within this of the grasp point on every axis (and the tolerances), not as soon
# as inside the tolerances: stopping at the tolerance box left the true grip 0.7 cm (median, closing axis) off with an
# exact camera where the IK leg, settling onto its goal, left 0.2 (L2 head camera, 2026-10-29)
REL_FINE_M = 0.004
REL_GATE_SLACK_M = 0.005  # the gate on the approach lets it advance while y and z are within tolerance + this (0:
# an orange, 0.5 cm a side, waited at the gate for most of REL_MAX_S; L0 39/50 vs 38)
REL_MAX_S = 6.0
REL_KI_ROT = 2.0  # 1/s wrist-attitude integral (recipes.tall_narrow's), on the turn carried from the pregrasp
REL_TURN_CAP = 0.5  # rad
REL_LEAD_CAP_M = 0.15  # the command runs at most this far (per axis) ahead of the jaw beyond the carried lead: a hand
# stalled on the object must not wind the command up (the grip-probe servo's command ran ahead of a stalled hand); at
# 0.05 the oranges at 0.55-0.61 m stalled 6-10 cm short (L0 35/50), the controller needing a longer lead out there
JAW_OPEN_M = 0.095
# carry-out
CARRY_GAP_M = 0.03  # a surface whose top is above the underside - this, with the object in its footprint, is overhead
CARRY_STEP_M = 0.01
CARRY_MAX_M = 0.50
CARRY_MARGIN_M = 0.05


def goal_object(task):
    """X of a task whose goal is held(X) of one object, else None. A task with no `objects` (a bare goal, as the load
    check passes) takes X from held(X); every real task carries its objects."""
    if len(task.goal) != 1 or not str(task.goal[0]).startswith("held("):
        return None
    objects = getattr(task, "objects", None) or [str(task.goal[0])[5:-1]]
    return objects[0] if len(objects) == 1 else None


def plan(task):
    """The plan for held(X) of one free object, or None."""
    key = goal_object(task)
    if key is None:
        return None
    return lambda ctx: pick(ctx, key)


def _note(ctx, step: str, **info) -> None:
    t = ctx.motor.t
    ctx.note(step, Outcome(True, t0=t, t1=t, info={k: _round(v) for k, v in info.items()}))


def _round(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    if isinstance(v, np.ndarray):
        return np.round(v, 4)
    return v


def object_frame(world, key: str):
    """(position, rotation matrix, box half extent) of a free object now."""
    p, q = world.object_pose(key)
    return p, R.mat(q), np.asarray(world.meta["objects"][key]["box"]["half_extent"], float)


def resting_half_height(Ro, half) -> float:
    """Half the object's box extent along the world vertical, as it lies."""
    return float(np.abs(Ro[2]) @ half)


def is_flat(world, key: str) -> bool:
    _, Ro, half = object_frame(world, key)
    return resting_half_height(Ro, half) < FLAT_HALF_M


def support_edges(surf: dict) -> list:
    """The support's four edges: (outward normal (world xy), edge length, narrow)."""
    hx, hy = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    yaw = float(surf.get("yaw", 0.0))
    ex, ey = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    edges = [(ex, 2 * hy), (-ex, 2 * hy), (ey, 2 * hx), (-ey, 2 * hx)]
    longest = 2 * max(hx, hy)
    return [(n, length, bool(length < NARROW_EDGE_FRAC * longest)) for n, length in edges]


def clearance(occ, xy) -> float:
    c = occ.cell_of(xy)
    return float(occ.clearance[c[0], c[1]]) if occ.in_bounds(c) else 0.0


def _left(u):
    return np.array([-u[1], u[0]])


_floor_statics_cache: dict = {}  # (id(meta), parameters) -> (meta, statics): built once per scene description


def floor_statics(world) -> list:
    """(centre xy, half extent xy) of the low fixed floor objects the runner's map leaves out: fixed objects of the
    scene file with a box whose bottom is at most FIT_FLOOR_BOTTOM_M and top at most FIT_FLOOR_TOP_M (as
    `floor_static_near` counts them; world-aligned bounds of the box as it stands), of the inner half of whose footprint
    fewer than FLOOR_MAPPED_FRAC of the map's cells are occupied (the inner half: a box spans legs and handles the map's
    surface leaves out, the round side table's 0.33 m half to its top's 0.25 m). Built once per world and parameters."""
    k = (id(world.meta), float(FIT_FLOOR_BOTTOM_M), float(FIT_FLOOR_TOP_M), float(FLOOR_MAPPED_FRAC))
    got = _floor_statics_cache.get(k)
    if got is not None and got[0] is world.meta:
        return got[1]
    occ, out = world.occ, []
    for key, o in world.meta["objects"].items():
        if key in world.free or o.get("free") or "box" not in o:
            continue
        q, quat = world.object_pose(key)
        Rq = R.mat(quat)
        c3 = np.asarray(q, float) + Rq @ np.asarray(o["box"].get("centre", (0.0, 0.0, 0.0)), float)
        e = np.abs(Rq) @ np.asarray(o["box"]["half_extent"], float)
        if c3[2] - e[2] > FIT_FLOOR_BOTTOM_M or c3[2] + e[2] > FIT_FLOOR_TOP_M:
            continue
        lo, hi = occ.clip_cell(occ.cell_of(c3[:2] - e[:2] / 2)), occ.clip_cell(occ.cell_of(c3[:2] + e[:2] / 2))
        if float(occ.occupied[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1].mean()) < FLOOR_MAPPED_FRAC:
            out.append((c3[:2].copy(), e[:2].copy()))
    _floor_statics_cache[k] = (world.meta, out)
    return out


def floor_blocked(world, xy) -> bool:
    """Under FLOOR_STANCE: whether a pelvis at `xy` lies within STANCE_CLEAR_M of a `floor_statics` footprint."""
    if not FLOOR_STANCE:
        return False
    xy = np.asarray(xy, float)
    gaps = (float(np.linalg.norm(np.maximum(np.abs(xy - c) - e, 0.0))) for c, e in floor_statics(world))
    return any(g < STANCE_CLEAR_M for g in gaps)


def stance_candidates(world, p_xy, surf: dict, min_d: float) -> list:
    """Stances (rank, shoulder-to-object distance, pelvis xy, facing unit vector, kind, angle to the nearest edge normal
    in deg), best first. Kinds: "normal" / "narrow" -- on an edge's outward normal ray through the object, the first
    point whose pelvis has STANCE_CLEAR_M (a narrow end's counts NARROW_PENALTY_M farther); "cone" -- a cell within
    CONE_DEG of a non-narrow edge normal; "nearest" -- any cell. Ranked: normal rays within REACH_MAX_M, cone cells
    within it, normal rays beyond, cone cells beyond, the nearest cells. No pelvis is `floor_blocked`."""
    occ = world.occ
    edges = support_edges(surf)
    out = []
    for n, _, narrow in edges:
        u = -n
        for s in np.arange(min_d, STANCE_MAX_D_M + 1e-9, 0.01):
            pel = p_xy + s * n + SHOULDER_Y_M * _left(u)
            if clearance(occ, pel) >= STANCE_CLEAR_M and not floor_blocked(world, pel):
                rank = (0 if s <= REACH_MAX_M else 2, s + (NARROW_PENALTY_M if narrow else 0.0))
                out.append((rank, float(s), pel, u, "narrow" if narrow else "normal", 0.0))
                break
    ii, jj = np.nonzero(occ.clearance >= STANCE_CLEAR_M)
    xy = occ.origin + (np.stack([ii, jj], -1) + 0.5) * occ.res
    d = np.linalg.norm(xy - p_xy, axis=1)
    keep = (d >= min_d) & (d <= STANCE_MAX_D_M)
    xy, d = xy[keep], d[keep]
    u = (p_xy - xy) / d[:, None]
    wide = [-n for n, _, narrow in edges if not narrow] or [-n for n, _, _ in edges]
    ang = np.degrees(np.arccos(np.clip(np.max(np.stack([u @ w for w in wide], 1), axis=1), -1.0, 1.0)))
    cone = ang <= CONE_DEG
    for tier, sel in ((1, cone & (d <= REACH_MAX_M)), (3, cone & (d > REACH_MAX_M)), (4, ~cone)):
        idx, kept = np.nonzero(sel)[0], 0
        for i in idx[np.argsort(d[idx], kind="stable")]:
            pel = xy[i] + SHOULDER_Y_M * _left(u[i])
            if clearance(occ, pel) < STANCE_CLEAR_M - 0.03:
                pel = xy[i]
            if floor_blocked(world, pel):
                continue
            out.append(((tier, float(d[i])), float(d[i]), pel, u[i], "cone" if tier < 4 else "nearest", float(ang[i])))
            kept += 1
            if kept == 10:
                break
    out.sort(key=lambda c: c[0])
    return out


def choose_stance(world, key: str, start_xy, min_d: float):
    """The first stance of `stance_candidates` with a path from `start_xy`, or None."""
    p, _, _ = object_frame(world, key)
    surf = world.meta["surfaces"][world.meta["objects"][key]["surface"]]
    for c in stance_candidates(world, p[:2], surf, min_d):
        if nav_path(world.occ, world.nav, start_xy, c[2]) is not None:
            return c
    return None


def hand_corridor(world, key: str, u, top: float):
    """The deepest overlap (m) of another free body's box with the box a side grasp's open hand sweeps along `u`
    (horizontal unit, shoulder to object): from CORRIDOR_BACK_M behind the pre-grasp to the object's near face,
    CORRIDOR_HALF_W_M either side of the line, from the housing's underside to CORRIDOR_ABOVE_M over the jaw; and that
    body (None when nothing overlaps). Boxes are bounded axis-aligned in the approach frame, at their poses now."""
    p, Ro, half = object_frame(world, key)
    ax = np.stack([np.r_[u, 0.0], np.r_[_left(u), 0.0], np.array([0.0, 0.0, 1.0])], axis=1)  # along, lateral, up
    gz = max(float(p[2]), top + JAW_ABOVE_TOP_M) - float(p[2])
    near = float(np.abs(ax[:, 0] @ Ro) @ half)
    lo = np.array([-PRE_BACK_M - CORRIDOR_BACK_M, -CORRIDOR_HALF_W_M, gz - CORRIDOR_BELOW_M])
    hi = np.array([-near, CORRIDOR_HALF_W_M, gz + CORRIDOR_ABOVE_M])
    worst, body = 0.0, None
    for k in world.free:
        if k == key:
            continue
        q, Rq, hq = object_frame(world, k)
        c, e = ax.T @ (q - p), np.abs(ax.T @ Rq) @ hq
        d = float(np.min(np.minimum(hi, c + e) - np.maximum(lo, c - e)))
        if d > worst:
            worst, body = d, k
    return worst, body


def corridor_stance(world, key: str, st, start_xy, min_d: float):
    """`st` (a `choose_stance` result) when its hand corridor is clear. Otherwise the stance on the ray through the
    object rotated by the smallest angle (CORRIDOR_STEP_DEG steps up to CORRIDOR_ROT_DEG either way, from `st`'s
    approach) whose corridor is clear: its first pelvis point with STANCE_CLEAR_M (not `floor_blocked`) between `min_d`
    and REACH_MAX_M, with a path; `st` again when none is. Returns (stance, what to log)."""
    p, _, _ = object_frame(world, key)
    surf = world.meta["surfaces"][world.meta["objects"][key]["surface"]]
    top = float(surf["top"])
    u0 = st[3]
    d0, body = hand_corridor(world, key, u0, top)
    info = dict(corridor_cm=100 * d0, blocker=body)
    if d0 <= CORRIDOR_TOL_M:
        return st, info
    inward = [-n for n, _, _ in support_edges(surf)]
    steps = np.arange(CORRIDOR_STEP_DEG, CORRIDOR_ROT_DEG + 1e-9, CORRIDOR_STEP_DEG)
    for th in [t for s in steps for t in (-s, s)]:
        c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
        u = np.array([c * u0[0] - sn * u0[1], sn * u0[0] + c * u0[1]])
        d, _ = hand_corridor(world, key, u, top)
        if d > CORRIDOR_TOL_M:
            continue
        for s in np.arange(min_d, REACH_MAX_M + 1e-9, 0.01):
            pel = p[:2] - s * u + SHOULDER_Y_M * _left(u)
            if clearance(world.occ, pel) >= STANCE_CLEAR_M and not floor_blocked(world, pel):
                break
        else:
            continue
        if nav_path(world.occ, world.nav, start_xy, pel) is None:
            continue
        ang = float(np.degrees(np.arccos(np.clip(max(float(u @ w) for w in inward), -1.0, 1.0))))
        info.update(rotated_deg=float(th), rotated_corridor_cm=100 * d)
        return ((5, float(s)), float(s), pel, u, "corridor", ang), info
    info.update(rotated_deg=None)
    return st, info


def _rot2(u, deg: float):
    c, sn = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * u[0] - sn * u[1], sn * u[0] + c * u[1]])


def floor_static_near(world, xy, r: float) -> bool:
    """Whether a fixed object of the scene file whose box rests on the floor (bottom at most FIT_FLOOR_BOTTOM_M, top at
    most FIT_FLOOR_TOP_M) has its footprint within `r` of `xy`."""
    for k, o in world.meta["objects"].items():
        if k in world.free or o.get("free") or "box" not in o:
            continue
        q, quat = world.object_pose(k)
        Rq = R.mat(quat)
        hq = np.asarray(o["box"]["half_extent"], float)
        c3 = np.asarray(q, float) + Rq @ np.asarray(o["box"].get("centre", (0.0, 0.0, 0.0)), float)
        e = np.abs(Rq) @ hq
        if c3[2] - e[2] > FIT_FLOOR_BOTTOM_M or c3[2] + e[2] > FIT_FLOOR_TOP_M:
            continue
        if np.linalg.norm(np.maximum(np.abs(np.asarray(xy, float) - c3[:2]) - e[:2], 0.0)) < r:
            return True
    return False


def fit_stance(world, key: str, st, start_xy, min_d: float):
    """`st` unless the object is a tall non-container (FIT_HEIGHT_MIN_M) whose hull section across the jaw at the grasp
    height does not fit the open jaw on `st`'s approach (`recipes.features` margin < 0). Then the stance on the ray
    through the object rotated by FIT_STEP_DEG steps up to FIT_ROT_DEG either way where the margin is in
    FIT_MARGIN_MIN_M..FIT_MARGIN_MAX_M (so recipes.tall_narrow takes it), the hand corridor is clear, and a pelvis
    point with STANCE_CLEAR_M (and, under FIT_FLOOR_GUARD, clear of low floor objects) lies between `min_d` and
    FIT_MAX_D_M with a path: the smallest such rotation, or under FIT_WIDEST the widest margin. `st` again when none
    is. Returns (stance, what to log; empty when the object does not trigger)."""
    o = world.meta["objects"][key]
    p, Ro, half = object_frame(world, key)
    if not FIT_ON or o.get("container") or 2 * resting_half_height(Ro, half) < FIT_HEIGHT_MIN_M:
        return st, {}
    surf = world.meta["surfaces"][o["surface"]]
    top = float(surf["top"])
    gz = max(float(p[2]), top + JAW_ABOVE_TOP_M)
    u0 = st[3]
    obj = recipes.features(world, key, np.r_[u0, 0.0], gz, flat=False, cover=False)
    if obj.margin is None or obj.height < FIT_HEIGHT_MIN_M or obj.margin >= 0.0:
        return st, {}
    info = dict(fit_margin_cm=100 * obj.margin)
    inward = [-n for n, _, _ in support_edges(surf)]
    steps = np.arange(FIT_STEP_DEG, FIT_ROT_DEG + 1e-9, FIT_STEP_DEG)
    found = []
    for th in [t for s in steps for t in (-s, s)]:
        u = _rot2(u0, th)
        mg = recipes.features(world, key, np.r_[u, 0.0], gz, flat=False, cover=False).margin
        if mg is None or not FIT_MARGIN_MIN_M <= mg < FIT_MARGIN_MAX_M:
            continue
        if hand_corridor(world, key, u, top)[0] > CORRIDOR_TOL_M:
            continue
        for s in np.arange(min_d, FIT_MAX_D_M + 1e-9, 0.01):
            pel = p[:2] - s * u + SHOULDER_Y_M * _left(u)
            if clearance(world.occ, pel) >= STANCE_CLEAR_M and not (
                FIT_FLOOR_GUARD and floor_static_near(world, pel, STANCE_CLEAR_M)
            ):
                break
        else:
            continue
        if nav_path(world.occ, world.nav, start_xy, pel) is None:
            continue
        found.append((float(th), float(mg), float(s), pel, u))
        if not FIT_WIDEST:
            break
    if not found:
        info.update(fit_deg=None)
        return st, info
    th, mg, s, pel, u = max(found, key=lambda f: (round(f[1], 4), -abs(f[0]))) if FIT_WIDEST else found[0]
    ang = float(np.degrees(np.arccos(np.clip(max(float(u @ w) for w in inward), -1.0, 1.0))))
    info.update(fit_deg=th, fit_margin_after_cm=100 * mg)
    return ((6, s), s, pel, u, "fit", ang), info


def banded(world, key: str, flat: bool, top: float) -> bool:
    """Whether a side grasp of `key` takes the raised approach: RAISED_ON, not flat, not a container, no surface
    overhead, and the support top in rim.RAISE_TOP_MIN_M..RAISE_TOP_MAX_M."""
    from . import rim  # rim imports pick

    o = world.meta["objects"][key]
    if not RAISED_ON or flat or o.get("container") or not rim.RAISE_TOP_MIN_M <= top <= rim.RAISE_TOP_MAX_M:
        return False
    po, _ = world.object_pose(key)
    return not overhead(world, po[:2], float(po[2]) - float(o["box"]["half_extent"][2]), o["surface"])


def walk_raised_checked(ctx, key: str, corners, pel, yaw: float, top: float, height: float) -> Outcome:
    """rim.walk_raised along `corners` to the stance (`pel`, `yaw`) through stage(): stance_check within
    RAISED_STOP_TOL_M / RAISED_YAW_TOL_DEG; a miss walks face-fixed to the stance again (RAISED_REWALKS), then gives
    up. Logs the jaw over the top and the object's drift and tilt at the walk's end."""
    from . import rim

    w, m = ctx.world, ctx.motor
    p0, R0, _ = object_frame(w, key)

    def check(out):
        ok, values = stance_check(m, pel, yaw, RAISED_STOP_TOL_M, RAISED_YAW_TOL_DEG)
        values["walk_ok"] = bool(out.ok)
        p, Ro, _ = object_frame(w, key)
        jaw, _ = m.jaw_centre(SIDE)
        tilt = float(np.degrees(np.arccos(np.clip(float(Ro[:, 2] @ R0[:, 2]), -1.0, 1.0))))
        values.update(jaw_z_minus_top=round(float(jaw[2]) - top, 4), obj_moved_cm=round(100 * float(np.linalg.norm(
            p[:2] - p0[:2])), 2), obj_tilt_deg=round(tilt, 2))  # fmt: skip
        return ok, values

    def on_fail(attempt, values):  # a walk that did not end (timeout) is not retried along a straight line
        return (lambda: m.walk_to(pel, yaw, face="fixed")) if values["walk_ok"] and attempt < RAISED_REWALKS else None

    return stage(ctx, "walk", lambda: rim.walk_raised(ctx, corners, yaw, top, height), check, on_fail)


def pick(ctx, key: str) -> Outcome:
    w, m = ctx.world, ctx.motor
    flat = is_flat(w, key)
    p, Ro, half = object_frame(w, key)
    surf = w.meta["surfaces"][w.meta["objects"][key]["surface"]]
    top = float(surf["top"])
    start_xy = m.pelvis()[0][:2]
    raised, corners, rinfo = banded(w, key, flat, top), None, {}
    if raised:
        from . import rim

        st, corners, rinfo = rim.rim_stance(w, key, start_xy, top, True)
    else:
        st = choose_stance(w, key, start_xy, TOP_MIN_D_M if flat else STANCE_MIN_D_M)
    if st is None:
        return ctx.note("stance", Outcome(False, "no stance with a path", m.t, m.t))
    corridor = {}
    if not flat:  # the pinch comes down from above, not along a corridor
        st, corridor = corridor_stance(w, key, st, start_xy, STANCE_MIN_D_M)
        st, fit = fit_stance(w, key, st, start_xy, STANCE_MIN_D_M)
        corridor.update(fit)
    _, s, pel, u, kind, ang = st
    if raised:
        if np.linalg.norm(np.asarray(corners[-1], float) - pel) > 1e-6:  # the corridor or the fit moved the stance
            corners = nav_path(w.occ, w.nav, start_xy, pel)
        corridor.update(raised=True, **rinfo)
    _note(ctx, "stance", kind=kind, shoulder_to_obj=s, normal_angle_deg=ang, pelvis=pel,
          clear=clearance(w.occ, pel), flat=flat, half_height=resting_half_height(Ro, half), **corridor)  # fmt: skip
    yaw_st = float(np.arctan2(u[1], u[0]))
    if raised and corners is not None:
        out = walk_raised_checked(ctx, key, corners, pel, yaw_st, top, 2 * resting_half_height(Ro, half))
    else:
        out = ctx.note("walk", walk_path(m, w.occ, w.nav, pel, yaw_st))
    if not out.ok or m.fell_at is not None:
        return out
    p, Ro, half = object_frame(w, key)
    pel_now, _ = m.pelvis()
    yaw = m.yaw()
    shoulder = pel_now[:2] - SHOULDER_Y_M * np.array([-np.sin(yaw), np.cos(yaw)])
    a = p[:2] - shoulder
    a = a / np.linalg.norm(a)
    ah = np.r_[a, 0.0]
    jaw0, _ = m.jaw_centre(SIDE)
    _note(ctx, "geom", shoulder_to_obj=float(np.linalg.norm(p[:2] - shoulder)), jaw_z_minus_top=float(jaw0[2]) - top,
          approach_deg=float(np.degrees(np.arctan2(a[1], a[0]))))  # fmt: skip
    if flat:
        out, Rg = pinch_top(ctx, key, p, Ro, half, top, a, jaw0)
    else:
        out, Rg = side_grasp(ctx, key, p, top, a, jaw0)
    if not out.ok or m.fell_at is not None:
        return out
    if not out.info.get("carried"):  # a recipe's hand program carries out itself
        carry_out(ctx, key, ah, out.info["lift_to"], Rg, speed=out.info.get("lift_speed"))
    if m.fell_at is not None:
        return Outcome(False, f"fell at {m.fell_at:.2f} s", m.t, m.t)
    held = bool(w.held(key))
    return Outcome(held, "" if held else "not between the fingers after the lift", m.t, m.t)


def first_via(jaw0, ah, top: float, pre):
    """Back VIA_BACK_M from where the jaw is, raised to the top + VIA_UP_M and the pre-grasp + VIA_OVER_M: the hand
    leaves a resting pose below a slab's edge before it goes over it."""
    back = jaw0 - VIA_BACK_M * ah
    return np.r_[back[:2], max(float(jaw0[2]), top + VIA_UP_M, float(pre[2]) + VIA_OVER_M)]


def raise_fit(ctx, key: str, ux, g):
    """`g` unless the object is a tall non-container (FIT_HEIGHT_MIN_M) whose hull section across the jaw along `ux`
    at `g`'s height does not fit the open jaw (`recipes.features` margin < 0) and the right shoulder stands within
    RAISE_MAX_D_M of it. Then `g` raised in RAISE_STEP_M steps to the first height, up to the object's top -
    RAISE_BELOW_TOP_M, where the margin is in FIT_MARGIN_MIN_M..FIT_MARGIN_MAX_M; `g` again when none is. Logs a
    raise_fit note when the object triggers."""
    w, m = ctx.world, ctx.motor
    o = w.meta["objects"][key]
    p, Ro, half = object_frame(w, key)
    hh = resting_half_height(Ro, half)
    if o.get("container") or 2 * hh < FIT_HEIGHT_MIN_M:
        return g
    z0 = float(g[2])
    m0 = recipes.features(w, key, ux, z0, flat=False, cover=False).margin
    if m0 is None or m0 >= 0.0:
        return g
    pel, yaw = m.pelvis()[0], m.yaw()
    reach = float(np.linalg.norm(p[:2] - (pel[:2] - SHOULDER_Y_M * np.array([-np.sin(yaw), np.cos(yaw)]))))
    if reach > RAISE_MAX_D_M:
        _note(ctx, "raise_fit", dz_cm=None, z=z0, margin_cm=100 * m0, margin_after_cm=None, reach=reach)
        return g
    for z in np.arange(z0 + RAISE_STEP_M, float(p[2]) + hh - RAISE_BELOW_TOP_M + 1e-9, RAISE_STEP_M):
        mg = recipes.features(w, key, ux, float(z), flat=False, cover=False).margin
        if mg is not None and FIT_MARGIN_MIN_M <= mg < FIT_MARGIN_MAX_M:
            _note(ctx, "raise_fit", dz_cm=100 * (float(z) - z0), z=float(z), margin_cm=100 * m0,
                  margin_after_cm=100 * mg, reach=reach)  # fmt: skip
            return np.r_[g[:2], float(z)]
    _note(ctx, "raise_fit", dz_cm=None, z=z0, margin_cm=100 * m0, margin_after_cm=None, reach=reach)
    return g


def side_grasp(ctx, key: str, p, top: float, a, jaw0) -> Outcome:
    """H1: wrist x level along `a` (shoulder to object), z up; pitched nose-down about the jaw centre when low fixed
    clutter lies under the forearm on a high support (`clutter.pitch`); else, for a tall object too wide for the jaw at
    its centre, raised to where it fits (`raise_fit`). A level grasp of an object whose features match a flow under
    recipes/ hands over to that flow after the pregrasp."""
    m = ctx.motor
    top = object_top(ctx, key, top)
    pel_z = REL_PELVIS_Z if REL_ON and REL_PELVIS_Z > 0 else None
    ux = np.r_[a, 0.0]
    Rg = np.stack([ux, np.array([-a[1], a[0], 0.0]), np.array([0.0, 0.0, 1.0])], axis=1)
    g = np.array([p[0], p[1], max(float(p[2]), top + JAW_ABOVE_TOP_M)])
    th = clutter.pitch(ctx, key, top, a, g)
    if th > 0.0:
        Rg = Rg @ R.rot_y(th)
        ux = Rg[:, 0]
    elif RAISE_ON:
        g = raise_fit(ctx, key, ux, g)
    pre = g - PRE_BACK_M * ux
    via = [first_via(jaw0, np.r_[a, 0.0], top, pre), np.r_[pre[:2], float(pre[2]) + VIA_OVER_M]]
    out = ctx.note("pregrasp", m.reach(SIDE, pre, Rg, via=via, pelvis_z=pel_z), grasp=g)
    if m.fell_at is not None:
        return out, Rg
    if th == 0.0:
        w = ctx.world
        po, _ = w.object_pose(key)
        cover = overhead(w, po[:2], float(po[2]) - float(w.meta["objects"][key]["box"]["half_extent"][2]),
                         w.meta["objects"][key]["surface"])  # fmt: skip
        obj = recipes.features(w, key, ux, float(g[2]), flat=False, cover=bool(cover))
        flow = recipes.choose(obj)
        if flow is not None:  # a flow for this kind of object (recipes/), chosen by its features
            _note(ctx, "recipe", recipe=flow.__name__.rsplit(".", 1)[-1], width_cm=100 * obj.width,
                  margin_cm=100 * obj.margin, near_cm=100 * obj.near, height=obj.height)  # fmt: skip
            return flow.grasp(ctx, key, obj, pre, Rg, ux), Rg
    entry = g + ENTRY_PAST_M * ux
    if REL_ON and th == 0.0:
        po, _ = ctx.world.object_pose(key)
        feat = recipes.features(ctx.world, key, ux, float(g[2]), flat=False, cover=False)
        tol = rel_tolerances(REL_TOL_APPROACH_M, feat.width)
        near = feat.near
        if near is None:  # no hull section at the grasp height: the box's half extent along the approach
            _, Ro, half = object_frame(ctx.world, key)
            near = float(np.abs(ux @ Ro) @ half)
        out = ctx.note("rel_entry", relative(ctx, key, "rel_entry", entry - po, Rg, tol, near, ENTRY_SPEED))
        if m.fell_at is not None:
            return out, Rg
        return grip_and_lift(ctx, Rg, pel_z), Rg
    out = ctx.note("entry", m.reach(SIDE, entry, Rg, speed=ENTRY_SPEED))
    if m.fell_at is not None:
        return out, Rg
    if th > 0.0:
        return clutter.grip_lift(ctx, key, Rg, entry), Rg
    return grip_and_lift(ctx, Rg), Rg


def pinch_axis(Ro, half, a, twist_lo: float | None = None):
    """psi of a wrist-down pinch (R = rot_z(psi) rot_y(pi/2): wrist x down, y = closing, z = (cos psi, sin psi)) closing
    across the narrowest horizontal box axis, wrist z away from the body (along `a`); and that axis' width. With
    `twist_lo` (deg), of psi and psi + pi (the jaw is symmetric) the one whose twist from `a` lies in [twist_lo,
    twist_lo + 180) instead."""
    horiz = [i for i in range(3) if abs(Ro[2, i]) < 0.5]
    i = min(horiz, key=lambda k: half[k])
    n = Ro[:2, i] / np.linalg.norm(Ro[:2, i])
    psi = float(np.arctan2(-n[0], n[1]))  # wrist y = (-sin psi, cos psi) = n
    if twist_lo is None:
        if np.cos(psi) * a[0] + np.sin(psi) * a[1] < 0:
            psi += np.pi
    else:
        t = float(np.degrees(R.wrap(psi - float(np.arctan2(a[1], a[0])))))
        if t < twist_lo:
            psi += np.pi
        elif t >= twist_lo + 180.0:
            psi -= np.pi
    return float(R.wrap(psi)), float(2 * half[i])


def high_pinch(Ro, half, top: float) -> bool:
    """A flat object resting under HIGH_REST_M tall on a support whose top is HIGH_SUPPORT_M or higher."""
    return 2 * resting_half_height(Ro, half) < HIGH_REST_M and top >= HIGH_SUPPORT_M


def pinch_top(ctx, key: str, p, Ro, half, top: float, a, jaw0) -> Outcome:
    """H3: wrist down over the object, fingertips TOP_TIP_CLEAR_M above the top; pre-grasp TOP_ABOVE_M above."""
    m = ctx.motor
    if not high_pinch(Ro, half, top):
        top = object_top(ctx, key, top)
        psi, width = pinch_axis(Ro, half, a)
        Rg = R.rot_z(psi) @ R.rot_y(np.pi / 2)
        g = np.array([p[0], p[1], top + FINGERTIP_M + TOP_TIP_CLEAR_M])
        pre = g + np.array([0.0, 0.0, TOP_ABOVE_M])
        via = [first_via(jaw0, np.r_[a, 0.0], top, pre)]
        yaw = m.yaw()
        out = ctx.note("pregrasp", m.reach(SIDE, pre, Rg, via=via), grasp=g, width_cm=round(100 * width, 2),
                       psi_from_facing_deg=round(float(np.degrees(R.wrap(psi - yaw))), 1))  # fmt: skip
        if m.fell_at is not None:
            return out, Rg
        if REL_ON:
            po, _ = ctx.world.object_pose(key)
            tol = rel_tolerances(REL_TOL_DESCENT_M, width)
            out = ctx.note("rel_descend", relative(ctx, key, "rel_descend", g - po, Rg, tol,
                                                   resting_half_height(Ro, half), TOP_DESCENT_SPEED))  # fmt: skip
        else:
            out = ctx.note("descend", m.reach(SIDE, g, Rg, speed=TOP_DESCENT_SPEED))
        if m.fell_at is not None:
            return out, Rg
        return grip_and_lift(ctx, Rg), Rg
    return high_pinch_top(ctx, key, p, Ro, half, top, a, jaw0)


def high_pinch_top(ctx, key: str, p, Ro, half, top: float, a, jaw0) -> Outcome:
    """`pinch_top` of a high pinch: the TWIST_LO_DEG yaw, SLOW_PINCH's pre-grasp, descent and lift, the full close.
    Under PINCH_REGRASP the lift is a stage() whose check is the object between both fingers and risen PINCH_RISE_M,
    and whose on_fail (the object not lifted) is `_pinch_again`, up to PINCH_REGRASP_MAX times."""
    if not PINCH_REGRASP:
        return _high_pinch_once(ctx, key, p, Ro, half, top, a, jaw0)
    m, w = ctx.motor, ctx.world
    out, Rg, lift, speed = _high_pinch_grip(ctx, key, p, Ro, half, top, a, jaw0)
    if lift is None:
        return out, Rg
    st = dict(Rg=Rg, lift=lift, speed=speed, grip=None)
    z0, up = float(p[2]), int(np.argmax(np.abs(Ro[2])))  # resting height, the box axis vertical at rest

    def lifted(_out):
        if st["grip"] is not None:
            return False, dict(grip_failed=True)
        po, _ = w.object_pose(key)
        rise, both = float(po[2]) - z0, bool(w.held(key))
        return both and rise >= PINCH_RISE_M, dict(rise_cm=round(100 * rise, 2), gripping=both)

    def again(attempt, values):
        if attempt >= PINCH_REGRASP_MAX or st["grip"] is not None or values.get("rise_cm", 0.0) >= 100 * PINCH_REST_M:
            return None
        return lambda: _pinch_again(ctx, key, top, a, z0, up, st)

    res = stage(ctx, "lift", lambda: m.reach(SIDE, lift, Rg, speed=speed), check=lifted, on_fail=again)
    if st["grip"] is not None:  # the re-grasp's grip closed on nothing: as the first grip's failure
        return st["grip"], st["Rg"]
    res.ok = m.fell_at is None
    res.info.update(lift_to=st["lift"], lift_speed=st["speed"])
    return res, st["Rg"]


def _pinch_again(ctx, key: str, top: float, a, z0: float, up: int, st: dict) -> Outcome:
    """The high pinch's re-grasp: open, rise PINCH_UP_M, re-read the object; if it lies within PINCH_REST_M of its
    resting height `z0`, its box axis `up` under PINCH_TILT_DEG from the vertical, pinch it again (pre-grasp, descent,
    grip) and return the lift's Outcome; else the Outcome that it did not."""
    m = ctx.motor
    ctx.note("release", m.release(SIDE))
    out = ctx.note("retreat", m.reach(SIDE, m.jaw_centre(SIDE)[0] + np.array([0.0, 0.0, PINCH_UP_M]), st["Rg"],
                                      speed=st["speed"]))  # fmt: skip
    if m.fell_at is not None:
        return out
    p, Ro, half = object_frame(ctx.world, key)
    dz, tilt = float(p[2]) - z0, float(np.degrees(np.arccos(min(1.0, abs(float(Ro[2, up]))))))
    flat = abs(dz) < PINCH_REST_M and tilt < PINCH_TILT_DEG
    _note(ctx, "regrasp", dz_cm=100 * dz, tilt_deg=tilt, again=flat)
    if not flat:
        return Outcome(False, "not back flat on its support", m.t, m.t)
    out, Rg, lift, speed = _high_pinch_grip(ctx, key, p, Ro, half, top, a, m.jaw_centre(SIDE)[0])
    st["Rg"] = Rg
    if lift is None:
        if m.fell_at is None:
            st["grip"] = out
        return out
    st.update(lift=lift, speed=speed)
    return m.reach(SIDE, lift, Rg, speed=speed)


def _high_pinch_once(ctx, key: str, p, Ro, half, top: float, a, jaw0) -> Outcome:
    """The high pinch without the lift check (c54e923f2234's and 011262a9020f's)."""
    m = ctx.motor
    out, Rg, lift, speed = _high_pinch_grip(ctx, key, p, Ro, half, top, a, jaw0)
    if lift is None:
        return out, Rg
    out = ctx.note("lift", m.reach(SIDE, lift, Rg, speed=speed))
    out.ok = m.fell_at is None
    out.info["lift_to"] = lift
    out.info["lift_speed"] = speed
    return out, Rg


def _high_pinch_grip(ctx, key: str, p, Ro, half, top: float, a, jaw0):
    """The high pinch up to the grip: (Outcome, Rg, lift target, lift speed); the target is None when the pre-grasp or
    the descent ended in a fall or the grip closed on nothing (the Outcome says which)."""
    m = ctx.motor
    psi, width = pinch_axis(Ro, half, a, TWIST_LO_DEG)
    Rg = R.rot_z(psi) @ R.rot_y(np.pi / 2)
    g = np.array([p[0], p[1], top + FINGERTIP_M + TOP_TIP_CLEAR_M])
    slow = bool(SLOW_PINCH)
    pre = g + np.array([0.0, 0.0, SLOW_TOP_ABOVE_M if slow else TOP_ABOVE_M])
    via = [first_via(jaw0, np.r_[a, 0.0], top, pre)]
    out = ctx.note("pregrasp", m.reach(SIDE, pre, Rg, via=via), grasp=g, width_cm=round(100 * width, 2),
                   psi_from_facing_deg=round(float(np.degrees(R.wrap(psi - m.yaw()))), 1),
                   twist_deg=round(float(np.degrees(R.wrap(psi - float(np.arctan2(a[1], a[0]))))), 1),
                   high=True, slow=slow)  # fmt: skip
    if m.fell_at is not None:
        return out, Rg, None, None
    out = ctx.note("descend", m.reach(SIDE, g, Rg, speed=SLOW_DESCENT_SPEED if slow else TOP_DESCENT_SPEED))
    if m.fell_at is not None:
        return out, Rg, None, None
    out = ctx.note("grip", m.grip(SIDE))
    if not out.ok or m.fell_at is not None:
        return out, Rg, None, None
    speed = SLOW_LIFT_SPEED if slow else LIFT_SPEED
    lift = m.jaw_centre(SIDE)[0] + np.array([0.0, 0.0, SLOW_LIFT_M if slow else LIFT_M])
    return out, Rg, lift, speed


def grip_and_lift(ctx, Rg, pelvis_z: float | None = None) -> Outcome:
    m = ctx.motor
    out = ctx.note("grip", m.grip(SIDE))
    if not out.ok or m.fell_at is not None:
        return out
    if SIDE in m.hand_override:  # after a relative entry: the reference takes the hand where the servo left it
        m.adopt_hand(SIDE)
    lift = m.jaw_centre(SIDE)[0] + np.array([0.0, 0.0, LIFT_M])
    out = ctx.note("lift", m.reach(SIDE, lift, Rg, speed=LIFT_SPEED, pelvis_z=pelvis_z))
    out.ok = m.fell_at is None
    out.info["lift_to"] = lift
    return out


def object_top(ctx, key: str, top: float) -> float:
    """The support top under the grasp: under REL_ON and REL_TOP_FROM_OBJECT the object's own underside as it is read
    now (the map's surface top `top` can sit off where the object stands), else `top`. Logs both."""
    if not (REL_ON and REL_TOP_FROM_OBJECT):
        return top
    p, Ro, half = object_frame(ctx.world, key)
    under = float(p[2]) - resting_half_height(Ro, half)
    _note(ctx, "object_top", map_top=top, object_underside=under)
    return under


def rel_tolerances(along: float, width) -> np.ndarray:
    """(approach, closing, third) tolerances of the relative entry for an object `width` across the jaw (None: the
    narrowest closing tolerance)."""
    spare = REL_TOL_CLOSE_MIN_M if width is None else 0.5 * (JAW_OPEN_M - float(width)) - REL_TOL_CLOSE_SLACK_M
    return np.array([along, float(np.clip(spare, REL_TOL_CLOSE_MIN_M, REL_TOL_CLOSE_MAX_M)), REL_TOL_THIRD_M])


def seen_jaw(ctx) -> np.ndarray:
    """The jaw centre as the robot sees it against the objects it reads: a world view with camera errors (`seen`) puts
    it through the same cameras as the objects; otherwise the motor's own reading."""
    jaw = ctx.motor.jaw_centre(SIDE)[0]
    seen = getattr(ctx.world, "seen", None)
    return jaw if seen is None else np.asarray(seen(jaw), float)


def relative(ctx, key: str, name: str, off, Rg, tol, near: float, speed: float) -> Outcome:
    """The relative entry of REL_MODE onto the object's estimate + `off` (world) in the grasp frame `Rg`; `near`: how far
    the object's near face lies behind its origin along the approach."""
    if REL_MODE == "servo":
        return rel_entry(ctx, key, name, off, Rg, tol)
    return rel_look(ctx, key, name, off, Rg, tol, near, speed)


def rel_look(ctx, key: str, name: str, off, Rg, tol, near: float, speed: float) -> Outcome:
    """REL_MODE "look": IK legs to goals the jaw's own reading + the seen error (Motor.reach's servo delivers a goal in
    the motor's frame, so the seen error is what moves it), with a look at the object before each."""
    w, m = ctx.world, ctx.motor
    t0 = m.t
    Rg, off = np.asarray(Rg, float), np.asarray(off, float)
    stage = float(off @ Rg[:, 0]) + near + FINGERTIP_M + REL_STAGE_CLEAR_M  # jaw behind the grasp point at the stage

    def look() -> np.ndarray:
        po, _ = w.object_pose(key)
        return Rg.T @ (np.asarray(po, float) + off - seen_jaw(ctx))

    def leg(d) -> Outcome:
        m.primitive = name
        return m.reach(SIDE, m.jaw_centre(SIDE)[0] + Rg @ d, Rg, speed=speed)

    seen, legs = [], 0
    rel = look()
    seen.append(rel)
    out = leg(np.r_[max(0.0, rel[0] - stage), rel[1], rel[2]])
    legs += 1
    for _ in range(REL_CORRECTIONS):
        if m.fell_at is not None:
            break
        rel = look()
        seen.append(rel)
        if abs(rel[1]) < tol[1] and abs(rel[2]) < tol[2]:
            break
        out = leg(np.r_[0.0, rel[1], rel[2]])
        legs += 1
    if m.fell_at is None:
        rel = look()
        seen.append(rel)
        out = leg(rel)
        legs += 1
    if m.fell_at is not None:
        return Outcome(False, f"fell at {m.fell_at:.2f} s", t0, m.t)
    rel = look()
    done = bool(np.all(np.abs(rel) < tol))
    return Outcome(True, "" if done else "not inside", t0, m.t,
                   info=dict(inside=done, legs=legs, stage_cm=round(100 * stage, 2), jaw_err_cm=out.info.get("jaw_err_cm"),
                             rel_cm=[round(100 * float(v), 2) for v in rel], tol_cm=[round(100 * float(v), 2) for v in tol],
                             looks_cm=[[round(100 * float(v), 2) for v in r] for r in seen]))  # fmt: skip


def body_follow(m, goal, Rg, seconds: float) -> None:
    """Blend the body's reference from where it is to a whole-body IK pose with SIDE's jaw centre at `goal` (wrist
    `Rg`, the other jaw where it is, Motor.reach's weights) over `seconds`; the wrist itself stays on Motor.hand."""
    solver = m._solver()
    cur = {s: m.jaw_centre(s) for s in M.SIDES}
    tp = np.array([np.asarray(goal, float) if s == SIDE else cur[s][0] for s in M.SIDES])
    tr = np.array([Rg if s == SIDE else cur[s][1] for s in M.SIDES])
    pose = m._pose_of(m._solve(solver, tp, tr, weights=tuple(1.0 if s == SIDE else 0.3 for s in M.SIDES)), m.yaw())
    m.ref, _ = m._blend([m.ref(m.t), pose], [seconds])


def rel_entry(ctx, key: str, name: str, off, Rg, tol) -> Outcome:
    """Servo the jaw onto the object's estimate + `off` (world) in the grasp frame `Rg` (module constants REL_*). The
    command starts at the pregrasp's last one (Motor.commanded_jaw and its wrist turn), so it does not step; the hand
    stays commanded at the end (grip_and_lift adopts it)."""
    w, m = ctx.world, ctx.motor
    m.primitive, t0 = name, m.t
    Rg, off = np.asarray(Rg, float), np.asarray(off, float)
    turn0 = rotvec(R.mat(m.last_fut[M.WRIST[SIDE]][1][0]) @ Rg.T)
    turn = turn0.copy()
    target = m.commanded_jaw(SIDE).copy()
    lead0 = target - m.jaw_centre(SIDE)[0]
    n_tick = max(1, int(round(REL_PERIOD_S / R.DT)))
    inside = reads = fresh = waits = solves = 0
    rel = np.zeros(3)
    solved = None  # the goal (motor frame) the body's reference was last solved for
    last, goal = None, None
    while m.t - t0 < REL_MAX_S:
        # one image, one instant: when the object's estimate is new (the view refreshed it from the cameras at this
        # read), the grasp point and the jaw as the camera shows them now give the jaw's way to go, which the jaw's own
        # reading turns into a goal in the motor's frame; until the next refresh the goal stays and the jaw's own
        # reading alone measures the way left. A kept estimate against the jaw seen later would carry the camera's
        # error at another pose of the body (the torso leans as the arm reaches): 1.3 cm on a coffee can at 2 deg.
        po, _ = w.object_pose(key)
        po = np.asarray(po, float)
        reads += 1
        jaw = m.jaw_centre(SIDE)[0]
        if goal is None or last is None or not np.array_equal(po, last):
            goal = jaw + (po + off - seen_jaw(ctx))
            fresh += 1
        last = po
        rel = Rg.T @ (goal - jaw)
        stop = np.minimum(tol, REL_FINE_M) if REL_FINE_M > 0 else tol
        inside = inside + 1 if bool(np.all(np.abs(rel) < stop)) else 0
        if inside >= REL_INSIDE:
            break
        if REL_BODY_FOLLOW and (solved is None or float(np.linalg.norm(goal - solved)) > REL_BODY_RESOLVE_M):
            body_follow(m, goal, Rg, max(REL_BODY_MIN_S, float(np.linalg.norm(rel)) / REL_BODY_SPEED))
            solved, solves = goal, solves + 1
        lateral_ok = abs(rel[1]) < tol[1] + REL_GATE_SLACK_M and abs(rel[2]) < tol[2] + REL_GATE_SLACK_M
        waits += not lateral_ok
        u = np.clip(REL_GAIN * rel * np.array([1.0 if lateral_ok else 0.0, 1.0, 1.0]), -REL_VMAX, REL_VMAX)
        for _ in range(n_tick):
            jaw = m.jaw_centre(SIDE)[0]
            target = jaw + lead0 + np.clip(target + Rg @ u * R.DT - jaw - lead0, -REL_LEAD_CAP_M, REL_LEAD_CAP_M)
            d = turn + REL_KI_ROT * R.DT * rotvec(Rg @ m.jaw_centre(SIDE)[1].T) - turn0
            n = float(np.linalg.norm(d))
            turn = turn0 + (d * (REL_TURN_CAP / n) if n > REL_TURN_CAP else d)
            m.hand(SIDE, target, expmap(turn) @ Rg, Rg @ u)
            if not m.tick():
                return Outcome(False, f"fell at {m.fell_at:.2f} s", t0, m.t)
    m.hand(SIDE, target, expmap(turn) @ Rg, np.zeros(3))
    done = bool(np.all(np.abs(rel) < tol))
    return Outcome(True, "" if done else f"not inside after {REL_MAX_S:.0f} s", t0, m.t,
                   info=dict(inside=done, reads=reads, fresh=fresh, waits=waits, solves=solves,
                             rel_cm=[round(100 * float(v), 2) for v in rel],
                             tol_cm=[round(100 * float(v), 2) for v in tol]))  # fmt: skip


def overhead(world, xy, bottom: float, support: str) -> list[str]:
    """Surfaces other than `support` whose footprint holds `xy` and whose top is above `bottom` - CARRY_GAP_M (the
    predicate's resting test with a margin)."""
    out = []
    for k, sf in world.meta["surfaces"].items():
        if k == support:
            continue
        half = np.asarray(sf.get("flat_half_extent") or sf["half_extent"], float)[:2]
        if np.all(np.abs(to_surface_frame(xy, sf)) <= half) and bottom - float(sf["top"]) <= CARRY_GAP_M:
            out.append(k)
    return out


def carry_out(ctx, key: str, ah, lift_to, Rg, speed: float | None = None) -> None:
    """While the lifted object's box centre lies under a surface (`overhead`), pull the hand back from the lift's
    target along -ah at `speed` (LIFT_SPEED) until the object is out of every such footprint, plus CARRY_MARGIN_M."""
    w, m = ctx.world, ctx.motor
    support = w.meta["objects"][key]["surface"]
    po, _ = w.object_pose(key)
    bottom = float(po[2]) - float(w.meta["objects"][key]["box"]["half_extent"][2])  # as the predicate (no rotation)
    over = overhead(w, po[:2], bottom, support)
    if not over:
        _note(ctx, "carry_out", need=0.0, bottom_minus_top=bottom - float(w.meta["surfaces"][support]["top"]))
        return
    need = CARRY_MAX_M
    for s in np.arange(0.0, CARRY_MAX_M + 1e-9, CARRY_STEP_M):
        if not overhead(w, po[:2] - s * ah[:2], bottom, support):
            need = float(s) + CARRY_MARGIN_M
            break
    speed = LIFT_SPEED if speed is None else speed
    ctx.note("carry_out", m.reach(SIDE, np.asarray(lift_to) - need * ah, Rg, speed=speed), need=round(need, 3),
             under=over)  # fmt: skip
