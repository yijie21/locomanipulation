"""Skill `place_on` (r1_on_single loop, iteration 0; the family's gap place_on_surface.single, rules 24): put a free
object down on a named surface. `pick.pick` as it is, step back (stand up if the pick left the pelvis low), walk to a
stance facing a free spot near an edge of the target (`plan_spot`), reach the object over the spot through a point
outside that edge, lower it, release, retract, step back.

The target height is the named surface's own top, never another surface sharing its footprint (NightStand__1's middle
shelf lies under NightStand__2's top). A spot must clear every object resting on the target by its current pose, not
the scene file's `surface` field (two_novels' second book has to see the first; the bed's pillows have no surface).

`on(X, S)` does not require a release: an object still held up to 0.30 m over S passes it. So the skill never ends
holding the object -- every failure after the pick opens the hand -- and it reports success only when, after the
retract, no finger touches X and X rests on S's own top. Each stage fails with its own string (pick refused / pick
failed / no spot / no stance or route / carry / dropped during the carry / place reach failed / still touching after
release / not resting on S) so failures split by layer.

Not tucked before the carry: raising a top-down-pinched remote to 0.8 m left it 34 deg off, and side-grasped objects
carried fine without a tuck (Reviewer's closed-loop checks). The in-place turns of the carry run at CARRY_TURN_RATE: the
one jam that slipped did so in a 120 deg final turn at the motor's 0.8 rad/s.

Drop-safe (loop iteration 1; iteration 0 won the gate 12-0 and was rejected on 10 drops):
- An object whose footprint overlaps a neighbour on its support is refused before the pick (`blocking_neighbour`): the
  open fingers have no room. Every book start and one vase start overlap by 1.2-1.5 cm, every other start is 2 cm or
  more clear, and none of the overlapping ones was ever put down.
- The put-down's pick aborts once the object has moved PICK_DISTURB_LIMIT_M (successes moved at most 0.7 cm; knocked
  objects passed 2 cm seconds before the 5 cm limit, then fell off the shelf). held(X)'s pick keeps its own limit.
- An object held more than HELD_OUT_MAX_M out along the fingers after the lift (`held_out`) is set back on its support
  instead of carried: those slid out of the grip in the carry, the successes sat 1.5-2.6 cm out.
- After a low pick (reference pelvis tilt above UPRIGHT_TILT_DEG) the jaw first pulls CLEAR_BACK_M straight back, level,
  and the robot stands up before it walks: walk_to's yaw-only pelvis otherwise snaps the reference wrist up 18-33 cm at
  the first tick. On that path the grip is checked after each step, and a lost grip ends the skill before any walk (a
  robot that walked on after a lost book fell).
- A failure after the lift sets the object down on whichever of its support and the target is under it (`set_down`)
  before the hand opens; it opens in the air only over neither.

Planned before the pick (loop iteration 2; iteration 1 won 15-0 and was rejected on 6 drops, 3 of them in the carry):
- The spot and the carry route are planned before any contact (`plan_ahead`), from where the back-off will leave the
  robot, with the object's yaw relative to the robot predicted from the grasp. No spot or route: refused untouched.
  After the lift the spot is planned again from where the robot is and only refines that plan.
- Creep budget, side grasps only: a carried object slides down between the pads at 0.04-0.14 cm/s whatever the grip
  force (the clock lost after 4.2 cm in 39 s). The pick is refused when the pad margin (`grip_margin`: the object's top
  above the grasp height less grasp_program.PAD_Z) is under CREEP_RATE_M_S times the planned lift-to-release time
  (`lift_to_release_s`); after the lift the margin measured in the jaw is checked against the rest of that time, and a
  short one is set back on its support. The top-down pinch (the remote: 0.3-0.5 cm in 21-23 s) is exempt.
- The put-down's pick aborts before the entry when the settle ends more than PICK_SETTLE_ABORT_M from the pre-grasp
  (grasp_program.SIMPLE settle_abort_m; every knock followed such a miss, every success settled within 1.3 cm).
- Never opened in the air while it still grips: a failure with no surface under the object walks back to the pick
  stance and puts it back where it was (`set_down` with `home`); at the target the hand opens only over it, after one
  more lowering when the first ended more than LOWER_RETRY_OFF_M off.

Deep counters and end spots (loop iteration 4; iteration 3 won 8-1 and was rejected on 3 drops, all put-downs released
2.1-2.6 cm from the target's end):
- The put-down's pick stands back for a right-hand side grasp on a surface above 0.6 m (pick.SIDE_AHEAD_M at
  PICK_SIDE_AHEAD_M, `_pick_switches`): from 0.40 m the soap's reach missed the pre-grasp by 13-22 cm. The left-hand
  fallback, the top-down pinch and every put-down stance keep 0.40 m. Its entry closes PICK_CAPTURE_DEPTH_M deep (the
  2.5 cm capture left objects 1.9-3.0 cm out toward the fingertips; near 3 cm they were set back or slid out), and a
  settle abort backs out from where the hand is (grasp_program; a retreat with wound-up feedback swept a vase off).
- A side-grasped object goes down PLACE_SIDE_MIN_M from the edge's ends, its footprint at the carried yaw SPOT_INSIDE_M
  inside the flat top (clearance.inside): one vase start fell off the table's end from 2.6 cm and stood from 6.1 cm. A
  top-down-pinched one keeps SPOT_SIDE_M (the TV leaves one remote start only a spot 3.8 cm from the stand's end).
- On a surface that lists no approach edge every put-down stance needs pick.OTHER_EDGE_CLEAR_M, as at an unlisted edge:
  a remote's carry to the TV stand's walled north end fell there 0.15 m from the wall (the successes stood 0.35 m clear).
- CREEP_RATE_M_S 0.12 cm/s (was 0.15): carries measured 0.067-0.117 cm/s on the successes and 0.08-0.13 cm/s on the
  clocks that dropped, and T_est overestimates the carry by 2.3-9.8 s.
- Logged only: the gripper's least clearance to static objects over the carry and the reach over the spot
  (clearance.sweep; a carried hand passed 1-4 cm from the thermostat beside the bathroom sink).

Closed-loop release, side grasps only (loop iteration 5; iterations 3 and 4 each won 8-1 and were rejected on 3 drops,
all of them put-downs that toppled on release or retract -- a jam or soap lying on the target is a drop):
- Levelled (`_lower_onto`): the wrist turns by the least rotation taking the object's measured up axis to vertical,
  at most LEVEL_MAX_DEG (clearance.level_rotation), before the reach over the spot, and holds it down: lowered at the
  attitude the carry left, objects reached the top tilted 1-16 deg.
- Released (`_release`): the servo's position lead and turn integral are dropped and the hand re-commanded where it
  is before the jaw opens (with the lead, up to 4.7 cm, the open jaw sprang up 1.5-4.3 cm and left the object leaning
  on one finger); the open jaw moves onto the object's centre along the wrist y (clearance.centre_move), backs out
  RETRACT_M along the fingers level, then rises RETRACT_RISE_M (the rising retract dragged leaning objects over).
- The put-down, `set_down` and `_put_back` (through `_lower_onto`) all release this way. A top-down-pinched object
  (pick.is_top, the remote) keeps the open-loop lowering, release and rising retract."""

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import FF_SHORT, FF_SPEEDS, SIDES, Outcome, Profile, tilt, turn_seconds
from locomani.scene_discovery.scene_file import to_surface_frame
from locomani.tasks.scene.start import footprint, overlap

from . import clearance, grasp_program, pick

BACK_AFTER_PICK_M = 0.30  # step back this far after the pick ...
STAND_UP_BELOW_M = 0.72  # ... and stand up when the pelvis is still lower than this
SPOT_CLEAR_M = 0.04  # the object's footprint at the spot this clear of every object resting on the target
SPOT_EDGE_M = 0.03  # spot depth in from its edge: from the object's smaller half extent + this ...
SPOT_DEPTH_MAX_M = 0.30  # ... to this (and the spot centre stays as far from the opposite edge)
SPOT_SIDE_M = 0.02  # spot centre at least this far in from the ends of its edge
SPOT_STEP_M = 0.02  # depth and along-edge grid
RESTING_BAND_M = 0.05  # an object whose underside is this close to the target's top rests on it
MAX_ROUTE_TRIES = 40  # stances tried for a route, best first
CARRY_TURN_RATE = 0.4  # rad/s peak in-place turn while carrying (the motor's 0.8 restored after the carry)
PLACE_CLEAR_M = 0.015  # object underside this far above the top at the end of the descent
ABOVE_M = 0.06  # the descent starts this far above that
VIA_OUT_M = 0.10  # the reach over the top passes with the object this far outside the edge, at the descent's start
DESCENT_SPEED = 0.04
RETRACT_M = 0.12  # after the release the jaw backs out this far along -wrist x ...
RETRACT_UP_M = 0.03  # ... and rises this far
BACK_AFTER_PLACE_M = 0.25
REST_GAP_M = 0.02  # done only with the object's underside at most this above the target's top
PICK_MIN_NEIGHBOUR_GAP_M = 0.0  # refuse the pick when a neighbour on the support comes closer than this (overlap)
PICK_DISTURB_LIMIT_M = 0.02  # grasp_program's disturb limit during the put-down's pick only (held(X)'s stays 0.05)
HELD_OUT_MAX_M = 0.03  # object centre at most this far out along the fingers after the lift, else set back
UPRIGHT_TILT_DEG = 10.0  # reference pelvis tilt above which the robot stands up before its first walk
CLEAR_BACK_M = 0.20  # ... after pulling the jaw this far straight back, level
SET_DOWN_MARGIN_M = 0.03  # set_down: the object's centre this far inside the surface's footprint
CREEP_RATE_M_S = 0.0009  # (0.0012 until r1_in_single's set) side grasps: the object slides down the pads this fast from the lift to the release
LEG_SETTLE_S = 3.0  # lift_to_release_s: per walk leg after its profile (walk_to: settle_s, up to its stop_timeout)
PICK_SETTLE_ABORT_M = 0.035  # grasp_program's settle_abort_m during the put-down's pick only (held(X)'s stays None)
LOWER_RETRY_OFF_M = 0.015  # a lowering that ends farther than this from its target ...
LOWER_RETRY_UP_M = 0.03  # ... rises this far and lowers once more
PICK_SIDE_AHEAD_M = 0.46  # pick.SIDE_AHEAD_M during the put-down (held(X)'s stays None)
PICK_CAPTURE_DEPTH_M = 0.015  # grasp_program's entry_capture_depth_m during the put-down's pick (held(X)'s stays 0.025)
PLACE_SIDE_MIN_M = 0.06  # a side-grasped object's spot centre at least this far in from the ends of its edge ...
SPOT_INSIDE_M = 0.015  # ... and its footprint at the carried yaw this far inside the flat top on every side
LEVEL_MAX_DEG = 25.0  # side grasps: the wrist levels the object's measured up axis by at most this before lowering
CENTRE_MIN_M = 0.002  # after the release the open jaw moves onto the object's centre (wrist y) when off more ...
CENTRE_MAX_M = 0.03  # ... than this, by at most this
RETRACT_RISE_M = 0.05  # side grasps: the jaw backs out RETRACT_M level, then rises this far


def neighbours(world, surface_key: str, exclude: str) -> dict[str, tuple]:
    """Footprints (start.footprint) by key of the objects, static or free, whose underside is now within
    RESTING_BAND_M of the surface's top and whose footprint can reach the surface's, except `exclude`."""
    s = world.meta["surfaces"][surface_key]
    top, c = float(s["top"]), np.asarray(s["xy"], float)
    reach = float(np.hypot(*(s.get("flat_half_extent") or s["half_extent"])))
    out = {}
    for k, o in world.meta["objects"].items():
        if k == exclude:
            continue
        pos, quat = world.object_pose(k)
        half, centre = np.asarray(o["box"]["half_extent"], float), np.asarray(o["box"].get("centre") or (0, 0, 0))
        if abs(float(pos[2] + centre[2] - half[2]) - top) > RESTING_BAND_M:
            continue
        xy = pos[:2] + centre[:2]
        if np.linalg.norm(xy - c) > reach + np.hypot(half[0], half[1]) + SPOT_CLEAR_M:
            continue
        out[k] = footprint(o, xy, quat)
    return out


def resting_on(world, surface_key: str, exclude: str) -> list[tuple]:
    """The footprints of `neighbours`."""
    return list(neighbours(world, surface_key, exclude).values())


def blocking_neighbour(world, key: str) -> str | None:
    """The first object resting on `key`'s support (`neighbours`) whose footprint comes within
    PICK_MIN_NEIGHBOUR_GAP_M of `key`'s, by current poses, or None."""
    o = world.meta["objects"][key]
    pos, quat = world.object_pose(key)
    me = footprint(o, pos[:2] + np.asarray(o["box"].get("centre") or (0, 0, 0))[:2], quat)
    for k, fp in neighbours(world, o["surface"], key).items():
        if overlap(me, fp, PICK_MIN_NEIGHBOUR_GAP_M):
            return k
    return None


def held_out(jaw, rot, obj_pos) -> float:
    """How far (m) the object's centre sits out along the fingers (the wrist x) from the jaw centre."""
    return float(grasp_program.object_in_palm(rot, jaw, obj_pos)[0])


def grip_margin(half_z: float, centre_z: float = 0.0, grasp_dz: float = 0.0) -> float:
    """Planned: how far (m) a side-grasped object can slide down between the pads before its top passes theirs -- its
    top above the grasp height (the jaw at the object's origin + `grasp_dz`) less grasp_program.PAD_Z."""
    return float(centre_z + half_z - grasp_dz - grasp_program.PAD_Z)


def grip_margin_now(jaw, rot, pos, quat, box: dict) -> float:
    """Measured: the object's top centre above the jaw centre along the wrist z, less grasp_program.PAD_Z."""
    centre = np.asarray(box.get("centre") or (0, 0, 0), float)
    top = np.asarray(pos, float) + R.mat(quat) @ (centre + np.array([0.0, 0.0, box["half_extent"][2]]))
    return float((np.asarray(rot).T @ (top - np.asarray(jaw, float)))[2] - grasp_program.PAD_Z)


def creep_short(margin: float, seconds: float) -> bool:
    """The margin is less than the creep over `seconds` at CREEP_RATE_M_S."""
    return margin < CREEP_RATE_M_S * seconds


def walk_seconds(cfg, xy, yaw: float, corners, goal_yaw: float | None, fixed: bool = False) -> float:
    """The planned duration of walking `corners` (pick.walk_route: one walk_to per corner after the first, the last
    one ending at `goal_yaw`) from `xy` facing `yaw`: per leg walk_to's turn toward it (none when `fixed`), its speed
    profile, its final turn, plus LEG_SETTLE_S."""
    t, p, yaw = 0.0, np.asarray(xy, float)[:2], float(yaw)
    ff = float(np.interp(cfg.walk_speed, FF_SPEEDS, FF_SHORT))
    for k, c in enumerate(corners[1:], start=1):
        delta = np.asarray(c, float)[:2] - p
        dist = float(np.linalg.norm(delta))
        heading = yaw
        if not fixed and dist > cfg.face_travel_min_m:
            travel = float(np.arctan2(delta[1], delta[0]))
            if abs(R.wrap(travel - yaw)) > np.radians(cfg.face_travel_deg):
                heading = travel
        end = goal_yaw if k == len(corners) - 1 and goal_yaw is not None else heading
        for a in (R.wrap(heading - yaw), R.wrap(end - heading)):
            t += turn_seconds(a, cfg.turn_rate) if abs(a) > 1e-3 else 0.0
        if dist > 1e-3:
            t += Profile(cfg.walk_speed, cfg.walk_acc, dist + ff if dist > 0.05 else dist).t_end
        t += LEG_SETTLE_S
        p, yaw = np.asarray(c, float)[:2], end
    return t


def lift_to_release_s(cfg, stance, face: float, corners, spot_face: float, depth: float, rise: float) -> float:
    """T_est (s), the lift to the release of a put-down planned before the pick: the pick's lift and hold, the back-off
    from `stance`, the carry over `corners` at CARRY_TURN_RATE ending at `spot_face`, the reach over the spot (legs of
    `depth` + VIA_OUT_M, the first also `rise` up or down) and the lowering, each reach with its whole settle."""
    back = np.asarray(stance, float)[:2] - BACK_AFTER_PICK_M * np.array([np.cos(face), np.sin(face)])
    t = grasp_program.SIMPLE["lift_seconds"] + grasp_program.HOLD_S
    t += walk_seconds(cfg, stance, face, [stance, back], face, fixed=True)
    t += walk_seconds(dataclasses.replace(cfg, turn_rate=CARRY_TURN_RATE), back, face, corners, spot_face)
    leg = depth + VIA_OUT_M
    t += sum(max(cfg.reach_min_s, d / cfg.reach_speed) for d in (float(np.hypot(leg, rise)), leg)) + cfg.reach_settle_s
    return t + max(cfg.reach_min_s, ABOVE_M / DESCENT_SPEED) + cfg.reach_settle_s


def surface_below(world, key: str, surfaces, margin: float = SET_DOWN_MARGIN_M) -> str | None:
    """Of `surfaces`, the highest whose flat footprint holds the object's centre `margin` inside its sides and whose
    top is below the object's underside + REST_GAP_M (an object still on it counts), or None."""
    pos, _ = world.object_pose(key)
    box = world.meta["objects"][key]["box"]
    centre = np.asarray(box.get("centre") or (0, 0, 0), float)
    xy, bottom = pos[:2] + centre[:2], float(pos[2] + centre[2] - box["half_extent"][2])
    best = None
    for k in dict.fromkeys(surfaces):
        s = world.meta["surfaces"][k]
        half = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)
        if np.any(np.abs(to_surface_frame(xy, s)) > half - margin) or float(s["top"]) > bottom + REST_GAP_M:
            continue
        if best is None or float(s["top"]) > float(world.meta["surfaces"][best]["top"]):
            best = k
    return best


def plan_spot(world, key: str, surface_key: str, side: str, start_xy, yaw_rel: float):
    """(spot or None, why). Spots: per edge of the target -- approach_edges first, then the others -- depths from the
    object's smaller half extent + SPOT_EDGE_M to SPOT_DEPTH_MAX_M and offsets along the edge, SPOT_STEP_M apart, the
    centre SPOT_SIDE_M from the edge's ends (a side-grasped object: PLACE_SIDE_MIN_M, and its footprint `inside` the
    flat top by SPOT_INSIDE_M); the object's footprint there, at yaw face + `yaw_rel` (its yaw relative to the robot's
    now, clearance.carried_footprint), SPOT_CLEAR_M clear of `resting_on`. Stance: the spot at pick.STAND_AHEAD_M ahead,
    pick.STAND_SIDE_M toward the hand `side`, with pick's floor-object and other-edge clearances (the latter at every
    edge of a surface that lists none: the TV stand's walled north end, where a remote's carry fell turning). Best first
    by listed edge, depth, distance from `start_xy`; the first of MAX_ROUTE_TRIES with a route on pick's map wins. spot:
    dict(xy, face, stance, edge_xy, edge, depth)."""
    meta = world.meta
    s = meta["surfaces"][surface_key]
    hx, hy = (float(v) for v in meta["objects"][key]["box"]["half_extent"][:2])
    taken = resting_on(world, surface_key, key)
    Rs = R.rot_z(float(s.get("yaw", 0.0)))[:2, :2]
    half = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)
    c = np.asarray(s["xy"], float)
    occ, nav, boxes = pick.nav_map(world)
    listed = list(s.get("approach_edges") or pick.NORMALS)
    boxed = not s.get("approach_edges")  # every edge counts as listed, and every stance needs the other-edge clearance
    edges = [(e, True) for e in listed] + [(e, False) for e in pick.NORMALS if e not in listed]
    sign = -1.0 if side == "right" else 1.0
    d_min = min(hx, hy) + SPOT_EDGE_M
    side_grasp = not pick.is_top(world, key)
    n_inside, n_free, cands = 0, 0, []
    for e, is_listed in edges:
        n = np.asarray(pick.NORMALS[e])  # outward normal, surface frame
        t = np.array([-n[1], n[0]])
        ax = 0 if "x" in e else 1
        along = half[1 - ax] - (PLACE_SIDE_MIN_M if side_grasp else SPOT_SIDE_M)
        d_max = min(SPOT_DEPTH_MAX_M, 2 * half[ax] - d_min)
        if along < 0 or d_max < d_min:
            continue
        w = Rs @ n
        face = float(np.arctan2(-w[1], -w[0]))
        for depth in np.arange(d_min, d_max + 1e-9, SPOT_STEP_M):
            for u in np.arange(-along, along + 1e-9, SPOT_STEP_M):
                xy = c + Rs @ (n * (half[ax] - depth) + t * u)
                fp = clearance.carried_footprint(xy, (hx, hy), face, yaw_rel)
                if side_grasp and not clearance.inside(fp, s, SPOT_INSIDE_M):
                    continue
                n_inside += 1
                if any(overlap(fp, tk, SPOT_CLEAR_M) for tk in taken):
                    continue
                n_free += 1
                stance = xy - (R.rot_z(face) @ np.array([pick.STAND_AHEAD_M, sign * pick.STAND_SIDE_M, 0.0]))[:2]
                if pick.floor_clearance(boxes, stance) < pick.STANCE_CLEAR_M:
                    continue
                if (not is_listed or boxed) and pick.other_clearance(
                    world, surface_key, stance
                ) < pick.OTHER_EDGE_CLEAR_M:
                    continue
                cands.append((0 if is_listed else 1, round(float(depth), 3), float(np.linalg.norm(stance - start_xy)),
                              dict(xy=xy, face=face, stance=stance, edge_xy=c + Rs @ (n * half[ax] + t * u), edge=e,
                                   depth=round(float(depth), 3))))  # fmt: skip
    if not n_inside:
        return None, f"no spot: {key} does not fit {SPOT_INSIDE_M} m inside {surface_key}"
    if not n_free:
        return None, f"no spot: nothing on {surface_key} clears the {len(taken)} objects resting there"
    cands.sort(key=lambda z: z[:3])
    for cd in cands[:MAX_ROUTE_TRIES]:
        if pick.route(occ, nav, start_xy, cd[3]["stance"]) is not None:
            return cd[3], ""
    return None, f"no stance or route: {n_free} free spots, {len(cands)} clear stances, none of the first " \
                 f"{min(len(cands), MAX_ROUTE_TRIES)} reachable"  # fmt: skip


def plan_ahead(world, key: str, surface_key: str, side: str, start_xy, cfg) -> dict | None:
    """The put-down planned before any contact. None when the pick has no plan (pick.pick refuses it with its own why);
    else dict(hand, face, stance, edge_xy (pick.plan_hand, as pick.pick will make it), xy (the object's now), top (a
    top-down pinch), spot, why (plan_spot from the back-off's end, the object's yaw on the robot predicted from the
    grasp: its yaw in the hand, yaw - psi, plus the hand's on the body, psi - face), corners (the carry route),
    t_est (`lift_to_release_s`, motor config `cfg`), margin (`grip_margin`))."""
    plan, hand, _ = pick.plan_hand(world, start_xy, key, side)
    if plan is None:
        return None
    face, _, psi, stance, edge_xy = plan
    pos, quat = world.object_pose(key)
    obj = world.meta["objects"][key]
    back = stance - BACK_AFTER_PICK_M * np.array([np.cos(face), np.sin(face)])
    spot, why = plan_spot(world, key, surface_key, hand, back, R.wrap(R.yaw_of(quat) - face))
    out = dict(hand=hand, face=face, stance=stance, edge_xy=edge_xy, xy=pos[:2].copy(), top=pick.is_top(world, key),
               spot=spot, why=why, corners=None, t_est=None,
               margin=grip_margin(obj["box"]["half_extent"][2], (obj["box"].get("centre") or (0, 0, 0))[2]))  # fmt: skip
    if spot is None:
        return out
    occ, nav, _ = pick.nav_map(world)
    out["corners"] = pick.route(occ, nav, back, spot["stance"])
    tops = (float(world.meta["surfaces"][k]["top"]) for k in (obj["surface"], surface_key))
    rise = abs(next(tops) + grasp_program.LIFT_M - next(tops) - PLACE_CLEAR_M - ABOVE_M)
    out["t_est"] = lift_to_release_s(cfg, stance, face, out["corners"], spot["face"], spot["depth"], rise)
    return out


def _refusal(ctx, ahead: dict | None) -> Outcome | None:
    """Logs `plan_ahead` (step "plan_ahead"); the refusal before any contact -- no spot or route, or a side grasp whose
    margin is short of the creep over T_est -- or None."""
    m = ctx.motor
    if ahead is None:
        return None
    why = ""
    if ahead["spot"] is None:
        why = f"{ahead['why']} (planned before the pick)"
    elif not ahead["top"] and creep_short(ahead["margin"], ahead["t_est"]):
        why = f"pick refused: carry of {ahead['t_est']:.0f} s exceeds the grip's creep margin " \
              f"({100 * ahead['margin']:.1f} cm < {100 * CREEP_RATE_M_S * ahead['t_est']:.1f} cm)"  # fmt: skip
    t_est = None if ahead["t_est"] is None else round(ahead["t_est"], 1)
    ctx.note("plan_ahead", Outcome(not why, why, m.t, m.t), t_est_s=t_est, margin_cm=round(100 * ahead["margin"], 2),
             top=ahead["top"], hand=ahead["hand"], spot=None if ahead["spot"] is None else ahead["spot"]["xy"])  # fmt: skip
    return Outcome(False, why, m.t, m.t) if why else None


@contextlib.contextmanager
def _pick_switches():
    """pick.SIDE_AHEAD_M at PICK_SIDE_AHEAD_M inside, so plan_ahead and pick.pick plan the same stance (plan_spot does
    not read it); pick's own value (held(X)'s: off) restored after."""
    saved = pick.SIDE_AHEAD_M
    pick.SIDE_AHEAD_M = PICK_SIDE_AHEAD_M
    try:
        yield
    finally:
        pick.SIDE_AHEAD_M = saved


@_pick_switches()
def place_on(ctx, key: str, surface_key: str, side: str = "right") -> Outcome:
    """Plan the put-down (`plan_ahead`), pick `key`, carry it to a free spot on `surface_key` and put it down, pick's
    put-down switch on throughout (`_pick_switches`); see the module doc."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    blocker = blocking_neighbour(world, key)
    if blocker is not None:
        why = f"pick refused: a neighbour blocks the jaw ({blocker} overlaps it)"
        return ctx.note("pick_check", Outcome(False, why, m.t, m.t), neighbour=blocker)
    ahead = plan_ahead(world, key, surface_key, side, m.sim.d.qpos[:2].copy(), m.cfg)
    refused = _refusal(ctx, ahead)
    if refused is not None:
        return refused
    n0 = len(ctx.log)
    saved = {k: grasp_program.SIMPLE[k] for k in ("disturb_limit_m", "settle_abort_m", "entry_capture_depth_m")}
    grasp_program.SIMPLE.update(disturb_limit_m=PICK_DISTURB_LIMIT_M, settle_abort_m=PICK_SETTLE_ABORT_M,
                                entry_capture_depth_m=PICK_CAPTURE_DEPTH_M)  # fmt: skip
    try:
        out = pick.pick(ctx, key, side)
    finally:
        grasp_program.SIMPLE.update(saved)
    if not out.ok:
        refused = not any(r["step"] == "walk" for r in ctx.log[n0:])
        return _fail(ctx, key, f"{'pick refused' if refused else 'pick failed'}: {out.why}")
    hands = sorted(m.gripping(body)) or sorted(m.touching(body))
    if not hands:
        return _fail(ctx, key, "pick failed: the object is not in the hand after the pick")
    side = hands[0]
    source = world.meta["objects"][key]["surface"]
    lifted = (source, surface_key)  # where a failure from here on sets it down
    home = dict(stance=ahead["stance"], face=ahead["face"], xy=ahead["xy"], edge_xy=ahead["edge_xy"], surface=source)
    failed = _handoff(ctx, key, side, lifted, ahead, home)
    if failed is not None:
        return failed

    _, quat = world.object_pose(key)
    spot, why = plan_spot(world, key, surface_key, side, m.sim.d.qpos[:2].copy(), R.wrap(R.yaw_of(quat) - m.yaw()))
    corners = None
    if spot is None:  # the plan made before the pick stands
        ctx.note("place_plan", Outcome(False, why, m.t, m.t), fallback="planned before the pick")
        spot, corners = ahead["spot"], ahead["corners"]
    k0 = len(m.trace)
    with _carry_turns(m):
        if corners is None:
            occ, nav, _ = pick.nav_map(world)
            walked = pick.walk_route(m, occ, nav, spot["stance"], spot["face"])
        else:
            walked = walk_corners(m, corners, spot["face"])
    out = ctx.note("carry", walked, spot=spot["xy"], edge=spot["edge"], depth=spot["depth"], stance=spot["stance"],
                   face_deg=round(float(np.degrees(spot["face"])), 1), sweep=clearance.sweep(world, m, side, k0))  # fmt: skip
    if not out.ok or m.fell_at is not None:
        return _fail(ctx, key, f"carry: {out.why}", lifted, home)
    if side not in m.gripping(body):
        return _fail(ctx, key, f"dropped during the carry: the {side} hand no longer grips it", lifted)
    return _put_down(ctx, key, surface_key, side, spot, lifted, home)


def _put_down(ctx, key: str, surface_key: str, side: str, spot: dict, lifted, home: dict) -> Outcome:
    """At the place stance: lower the object over the spot (`_lower_onto`), release over the target only, retract, step
    back; done only with no finger on it and it resting on the target's own top."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    off, level = _lower_onto(ctx, key, side, spot["xy"], surface_key, spot["edge_xy"], spot["face"])
    if m.fell_at is not None:
        return _fail(ctx, key, f"place reach failed: {'; '.join(off)}")
    if surface_below(world, key, (surface_key,), margin=0.0) is None:
        return _fail(ctx, key, f"place reach failed: not over {surface_key} ({'; '.join(off)})", lifted, home)
    if pick.is_top(world, key):
        ctx.note("release", m.release(side))
        jaw, rot = m.jaw_centre(side)
        out = ctx.note("retract", m.reach(side, jaw - RETRACT_M * rot[:, 0] + np.array([0.0, 0.0, RETRACT_UP_M]), None))
    else:
        out = _release(ctx, key, side, level)
    if m.fell_at is None:
        out = ctx.note("step_back", m.sidestep(forward=-BACK_AFTER_PLACE_M))
    if m.fell_at is not None:
        return Outcome(False, f"after the release: {out.why}", out.t0, out.t1)
    touching = sorted(m.touching(body))
    if touching:
        return Outcome(False, f"still touching after release: {'/'.join(touching)} hand", out.t0, out.t1)
    rests, why = resting(world, key, surface_key)
    if not rests:
        return Outcome(False, f"place reach failed: {'; '.join(off)} ({why})" if off else why, out.t0, out.t1)
    return Outcome(True, "", out.t0, out.t1, info=dict(spot=np.round(spot["xy"], 3).tolist(), edge=spot["edge"]))


def _lower_onto(ctx, key: str, side: str, xy, surface_key: str, edge_xy, face: float) -> tuple[list[str], float | None]:
    """Reach the held object over `xy` on `surface_key` through a point VIA_OUT_M outside the edge point `edge_xy` (the
    robot facing `face`) and lower it to PLACE_CLEAR_M above the surface's own top at DESCENT_SPEED; a lowering that
    ends more than LOWER_RETRY_OFF_M off rises LOWER_RETRY_UP_M and lowers once more. A side-grasped object is levelled
    first: every leg holds the wrist turned by clearance.level_rotation (at most LEVEL_MAX_DEG) of its measured up
    axis; a top-down-pinched one keeps the wrist as carried. (the reaches that failed, the levelling turn in deg or None
    for a pinch; the caller checks m.fell_at)."""
    world, m = ctx.world, ctx.motor
    top = float(world.meta["surfaces"][surface_key]["top"])
    obj = world.meta["objects"][key]
    box = obj["box"]
    pos, quat = world.object_pose(key)
    jaw, rot = m.jaw_centre(side)
    rel = pos - jaw  # object centre from the jaw centre, as carried
    rot_goal, level = None, None
    if not pick.is_top(world, key):
        corr, level = clearance.level_rotation(clearance.up_axis(quat, obj["init_quat"]), LEVEL_MAX_DEG)
        rot_goal, rel = corr @ rot, corr @ rel
    target = np.r_[xy, top + box["half_extent"][2] - (box.get("centre") or (0, 0, 0))[2] + PLACE_CLEAR_M] - rel
    above = target + np.array([0.0, 0.0, ABOVE_M])
    n_in = np.array([np.cos(face), np.sin(face)])
    via = (np.r_[np.asarray(edge_xy, float) - n_in * VIA_OUT_M - rel[:2], above[2]],)
    legs = [("over", above, dict(via=via)), ("lower", target, dict(speed=DESCENT_SPEED))]
    off = []
    while legs:
        name, goal, kw = legs.pop(0)
        k0 = len(m.trace)
        out = m.reach(side, goal, rot_goal, **kw)
        sweep = clearance.sweep(world, m, side, k0) if name == "over" else None
        ctx.note(name, out, target=np.round(goal, 3), **({"sweep": sweep} if sweep else {}))
        if m.fell_at is not None:
            return off + [f"{name}: {out.why}"], level
        if not out.ok:
            off.append(f"{name}: {out.why}")
        if name == "lower" and float(np.linalg.norm(m.jaw_centre(side)[0] - target)) > LOWER_RETRY_OFF_M:
            up = target + np.array([0.0, 0.0, LOWER_RETRY_UP_M])
            legs = [("raise_again", up, {}), ("lower_again", target, dict(speed=DESCENT_SPEED))]
    return off, level


def _release(ctx, key: str, side: str, level: float | None) -> Outcome:
    """Let go of a side-grasped object resting on a surface: drop `side`'s servo lead and turn integral and re-command
    the hand where it is ("unload"), open ("open"), move the open jaw onto the object's centre along the wrist y
    (clearance.centre_move, "centre"), back out RETRACT_M along -wrist x level ("retract"), rise RETRACT_RISE_M
    ("rise"), the attitude held throughout; stops at a fall. Logs one "release" step: the object's lean at the end of
    the lowering, the levelling turn `level`, the lead dropped, the centring offset, the lean after the opening and
    after the retract, the hands touching it after the opening. Returns the last motion's outcome (the caller checks
    m.fell_at)."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    rest = world.meta["objects"][key]["init_quat"]
    t0 = m.t
    info = dict(lean_lowered_deg=round(clearance.lean_deg(world.object_pose(key)[1], rest), 1),
                level_deg=None if level is None else round(level, 1))  # fmt: skip
    lead = m.servo.pop(("pos", side), None)
    m.servo.pop(("turn", side), None)
    info["lead_cm"] = None if lead is None else np.round(100 * lead, 1).tolist()
    jaw, rot = m.jaw_centre(side)
    out = ctx.note("unload", m.reach(side, jaw, rot))
    if m.fell_at is None:
        out = ctx.note("open", m.release(side))
        info.update(lean_open_deg=round(clearance.lean_deg(world.object_pose(key)[1], rest), 1),
                    touching_open=sorted(m.touching(body)))  # fmt: skip
    if m.fell_at is None:
        jaw, rot = m.jaw_centre(side)
        d = float(grasp_program.object_in_palm(rot, jaw, world.object_pose(key)[0])[1])
        move = centre_move(d)
        info["centre_cm"] = round(100 * d, 2)
        if move:
            out = ctx.note("centre", m.reach(side, jaw + move * rot[:, 1], rot), move_cm=round(100 * move, 2))
    if m.fell_at is None:
        jaw, rot = m.jaw_centre(side)
        out = ctx.note("retract", m.reach(side, jaw - RETRACT_M * rot[:, 0], rot))
    if m.fell_at is None:
        jaw, rot = m.jaw_centre(side)
        out = ctx.note("rise", m.reach(side, jaw + np.array([0.0, 0.0, RETRACT_RISE_M]), rot))
    info["lean_retract_deg"] = round(clearance.lean_deg(world.object_pose(key)[1], rest), 1)
    ctx.note("release", Outcome(m.fell_at is None, "" if m.fell_at is None else out.why, t0, m.t), **info)
    return out


def centre_move(d: float) -> float:
    """clearance.centre_move at CENTRE_MIN_M, CENTRE_MAX_M."""
    return clearance.centre_move(d, CENTRE_MIN_M, CENTRE_MAX_M)


def _handoff(ctx, key: str, side: str, lifted, ahead: dict, home: dict) -> Outcome | None:
    """From the pick stance to standing clear with the object, ready to carry: the fingertip check (`held_out`) and, for a
    side grasp, the grip margin measured in the jaw against the creep over the rest of T_est; after a low pick the level
    pull-back and the stand-up, the grip checked after each step and after the back-off; the back-off (and the stand-up
    when the pelvis is still low). None when ready, else the failure (`_fail`, object set down)."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    failed = _grip_checks(ctx, key, side, lifted, ahead, home)
    if failed is not None:
        return failed

    tilt_deg = float(np.degrees(tilt(m.ref(m.t).quat)))
    low_pick = tilt_deg > UPRIGHT_TILT_DEG
    if low_pick:
        jaw, rot = m.jaw_centre(side)
        back = jaw - CLEAR_BACK_M * np.array([np.cos(m.yaw()), np.sin(m.yaw()), 0.0])
        out = ctx.note("pull_back", m.reach(side, back, rot), ref_tilt_deg=round(tilt_deg, 1))
        if m.fell_at is not None:
            return _fail(ctx, key, f"carry: {out.why}")
        if not out.ok:
            return _fail(ctx, key, f"pick failed: stuck after the lift ({out.why})", lifted, home)
        if side not in m.gripping(body):
            return _fail(ctx, key, f"dropped before the carry: the {side} hand lost it in the pull_back", lifted)
        out = m.posture()
        ctx.note("stand", out, ref_tilt_after_deg=round(float(np.degrees(tilt(m.ref(m.t).quat))), 1))
        if m.fell_at is not None:
            return _fail(ctx, key, f"carry: {out.why}")
        if side not in m.gripping(body):
            return _fail(ctx, key, f"dropped before the carry: the {side} hand lost it in the stand", lifted)

    k0 = len(m.trace)
    out = ctx.note("back_off", m.sidestep(forward=-BACK_AFTER_PICK_M), ref_wrist_jump_cm=_ref_jump(m, side, k0))
    if m.fell_at is not None:
        return _fail(ctx, key, f"carry: {out.why}")
    if low_pick and side not in m.gripping(body):
        return _fail(ctx, key, f"dropped before the carry: the {side} hand lost it in the back_off", lifted)
    if float(m.sim.d.qpos[2]) < STAND_UP_BELOW_M:
        out = ctx.note("stand", m.posture())
        if m.fell_at is not None:
            return _fail(ctx, key, f"carry: {out.why}")
    return None


def _grip_checks(ctx, key: str, side: str, lifted, ahead: dict, home: dict) -> Outcome | None:
    """`_handoff`'s checks after the lift, no motion: the fingertip check (`held_out`) and, for a side grasp, the grip
    margin measured in the jaw against the creep over the rest of T_est. None when both pass, else the failure."""
    world, m = ctx.world, ctx.motor
    jaw, rot = m.jaw_centre(side)
    pos, quat = world.object_pose(key)
    out_m = held_out(jaw, rot, pos)
    ctx.note("held_out", Outcome(out_m <= HELD_OUT_MAX_M, "", m.t, m.t), cm=round(100 * out_m, 2))
    if out_m > HELD_OUT_MAX_M:
        return _fail(ctx, key, f"pick failed: held at the fingertips ({100 * out_m:.1f} cm out)", lifted, home)
    if not ahead["top"]:
        margin = grip_margin_now(jaw, rot, pos, quat, world.meta["objects"][key]["box"])
        rest = ahead["t_est"] - grasp_program.SIMPLE["lift_seconds"] - grasp_program.HOLD_S
        short = creep_short(margin, rest)
        ctx.note("grip_margin", Outcome(not short, "", m.t, m.t), cm=round(100 * margin, 2),
                 creep_cm=round(100 * CREEP_RATE_M_S * rest, 2), rest_s=round(rest, 1))  # fmt: skip
        if short:
            return _fail(ctx, key, f"pick failed: grip margin {100 * margin:.1f} cm < creep over {rest:.0f} s", lifted,
                         home)  # fmt: skip
    return None


def walk_corners(m, corners, goal_yaw: float) -> Outcome:
    """pick.walk_route's walk over route corners already planned (the first is where the robot was to start)."""
    if len(corners) < 2:
        return m.walk_to(corners[-1], goal_yaw)
    out = None
    for k, c in enumerate(corners[1:], start=1):
        out = m.walk_to(c, goal_yaw if k == len(corners) - 1 else None)
        if not out.ok:
            return out
    out.info["legs"] = len(corners) - 1
    return out


@contextlib.contextmanager
def _carry_turns(m):
    """The motor's in-place turns at CARRY_TURN_RATE inside, its own config restored after."""
    cfg0 = m.cfg
    m.cfg = dataclasses.replace(cfg0, turn_rate=CARRY_TURN_RATE)
    try:
        yield
    finally:
        m.cfg = cfg0


def resting(world, key: str, surface_key: str) -> tuple[bool, str]:
    """`on(key, surface_key)` with the underside at most REST_GAP_M above the surface's own top (`on` alone admits
    0.30 m)."""
    ok, why = world.check([f"on({key}, {surface_key})"])(world.state([key]))
    if not ok:
        return False, f"not resting on {surface_key}: {why}"
    pos, _ = world.object_pose(key)
    box = world.meta["objects"][key]["box"]
    top = float(world.meta["surfaces"][surface_key]["top"])
    gap = float(pos[2] + (box.get("centre") or (0, 0, 0))[2] - box["half_extent"][2]) - top
    return (True, "") if gap <= REST_GAP_M else (False, f"not resting on {surface_key}: underside {gap:+.3f} m")


def set_down(ctx, key: str, surfaces, home: dict | None = None) -> str:
    """Lower the held object onto `surface_below` among `surfaces` (underside PLACE_CLEAR_M above its top), open every
    hand that touches it, retract the holding one as the put-down does (a side grasp: `_release`). Over neither, while a
    hand still grips it and `home` (the pick's stance, face, edge point, the object's start xy and support) is given:
    walk back there and put it back where it was first (`_put_back`). Over none: just open. Returns what happened, for
    the failure string."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    hands = sorted(m.gripping(body)) or sorted(m.touching(body))
    under = surface_below(world, key, surfaces)
    back = ""
    if under is None and home is not None and hands and hands[0] in m.gripping(body):
        under, back = _put_back(ctx, key, hands[0], home)
        if m.fell_at is not None:
            return f"fell {back}"
    if under is not None:
        pos, _ = world.object_pose(key)
        box = world.meta["objects"][key]["box"]
        bottom = float(pos[2] + (box.get("centre") or (0, 0, 0))[2] - box["half_extent"][2])
        drop = max(0.0, bottom - float(world.meta["surfaces"][under]["top"]) - PLACE_CLEAR_M)
        jaw, _ = m.jaw_centre(hands[0])
        ctx.note("set_down", m.reach(hands[0], jaw - np.array([0.0, 0.0, drop]), None, speed=DESCENT_SPEED),
                 surface=under, drop_cm=round(100 * drop, 1))  # fmt: skip
        if m.fell_at is not None:
            return f"fell setting it down on {under}"
    if under is not None and not pick.is_top(world, key):
        for s in sorted(m.touching(body) - {hands[0]}):
            ctx.note("let_go", m.release(s))
        _release(ctx, key, hands[0], None)
    else:
        for s in sorted(m.touching(body)):
            ctx.note("let_go", m.release(s))
        if under is None:
            return f"opened over no surface{f' ({back})' if back else ''}"
        jaw, rot = m.jaw_centre(hands[0])
        ctx.note("retract", m.reach(hands[0], jaw - RETRACT_M * rot[:, 0] + np.array([0.0, 0.0, RETRACT_UP_M]), None))
    rests, _ = resting(world, key, under)
    done = "put back" if back == "walked back" else "set down"
    return f"{done} on {under}" if rests else f"{done}, not resting on {under}"


def _put_back(ctx, key: str, side: str, home: dict) -> tuple[str | None, str]:
    """Walk the gripped object back to the pick's stance and lower it over its start xy on its support (`_lower_onto`).
    (the support when the object is over it now, else None; what happened)."""
    world, m = ctx.world, ctx.motor
    occ, nav, _ = pick.nav_map(world)
    with _carry_turns(m):
        out = ctx.note("walk_back", pick.walk_route(m, occ, nav, home["stance"], home["face"]))
    if m.fell_at is not None or not out.ok:
        return None, f"walking back: {out.why}"
    if side not in m.gripping(world.free[key]):
        return None, "lost it walking back"
    off, _ = _lower_onto(ctx, key, side, home["xy"], home["surface"], home["edge_xy"], home["face"])
    if m.fell_at is not None:
        return None, f"putting it back: {'; '.join(off)}"
    return surface_below(world, key, (home["surface"],), margin=0.0), "walked back"


def _fail(ctx, key: str, why: str, lifted=(), home: dict | None = None) -> Outcome:
    """The failure `why`, after opening every hand that still touches the object when the robot stands: the skill never
    ends holding it (a hover over the target passes `on`). After a lift (`lifted`: its support and the target) it is
    first set down on the one under it, or put back at `home` (`set_down`)."""
    m = ctx.motor
    if m.fell_at is None and m.touching(ctx.world.free[key]):
        if lifted:
            why = f"{why} [{set_down(ctx, key, lifted, home)}]"
        else:
            for s in sorted(m.touching(ctx.world.free[key])):
                ctx.note("let_go", m.release(s))
    return Outcome(False, why, m.t, m.t)


def _ref_jump(m, side: str, k0: int) -> float | None:
    """The reference wrist's jump (cm) across the first tick of the primitive that began at trace row `k0`."""
    if k0 < 1 or len(m.trace) < k0 + 2:
        return None
    i = SIDES.index(side)
    return round(100 * float(np.linalg.norm(m.trace[k0 + 1]["ref_wrists"][i] - m.trace[k0 - 1]["ref_wrists"][i])), 1)
