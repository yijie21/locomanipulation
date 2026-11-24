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
display's top shelf; farther out everywhere, it broke pitched top-shelf grasps). The wrist and forearm hang lower still
(WRIST_LOW_POINT, r1_in_single loop iteration 3), and count where they lie over the support (WRIST_OVER_M in from the
edge, an apple 0.17 m in on a straight approach): a level grasp of an apple (centre 4.0-4.1 cm above the dining table)
deeper than that laid right_wrist_yaw on the slab's edge, which pitched the hand 10-22 deg nose-up and stalled the entry
6-15 cm short (3 of 8 apples 0.18-0.31 m in); pitched, the wrist rises 3.4-3.9 cm and all 8 were grasped. Shallower
(0.09-0.16 m), the wrist is off the slab and the level grasp took all 4 apples, while pitched ones carried to the bowl
0.65 m ahead slipped out of the jaw in 2 of 3 (the hand 23-34 deg off its pitched attitude at the reach limit).

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
and the top-down pinch are not retried; a success never reaches the retry, so its steps are unchanged.

The narrow covered grasp (r1_slide_then_grasp loop iteration 1, `narrow_covered`): an object under another surface
whose only side that fits the jaw is narrower than NARROW_SIDE_MAX_M (the 4.7 cm thick novel packed in a row on the
bedroom display's bottom shelf, 16.6 cm deep, 23.9 cm tall) runs the same level grasp with NARROW's values. The
module's jaw closed 7-8 cm in front of the book's centre with about 10 N: a two-point hinge about which the lift
pivoted the book on its rear-bottom edge (it toppled, or rose 2 cm on an arc as the pads slid up). A narrower open jaw,
an entry stopping 5 cm short instead of 2.5 (the jaw 3-4.6 cm from the centre), no lateral correction, twice the
squeeze (about 20 N), a 1.5 cm rise threshold for a pivoted book, and held.pick_held's pull out to NARROW pull_m
(NARROW max_m in all when the book is still under the shelves above) turn the pivot into a slide-out that ends held.
The pull travels on the pick's outcome (info "withdraw").

The low grasp (r2_on_single iteration 1, `low_grasp`): a level side grasp whose pads at the centre would meet a
collision hull wider than LOW_SPAN_M (the round big white vase, 9.2 cm at its equator in the 9.5 cm open jaw) closes
at the highest jaw height below the centre where the hull spans at most LOW_WIDTH_M over the pads' height (wrist
pitch LOW_PITCH_DEG), so the pads meet it just under its widest part. Its straight approach, when it would sweep an
object fixed on the support, is turned CLEAR_YAW_DEG away with the far hand (`clear_pass`). Its outcome's info "low"
tells place.place_on to level the object by up to place.LOW_LEVEL_MAX_DEG before the put-down (r2_on_single iteration
2). A failed low grasp is not retried, a pre-grasp that ends more than LOW_PREGRASP_MISS_M off runs no hand program
(a snagged hand's settle swept the vase off its shelf), an entry whose grasp point drifts more than LOW_LATERAL_ABORT_M
across the jaw fails before the fingers reach it (one that went on struck the vase off its shelf), and a failure's
set-down levels it (place.set_down).

The end-on long box (gso_single_grasp iteration 0, `end_on`): a box whose only side that fits the jaw is its thickness
and whose other side is longer than END_ON_M (the logistics room's game, snack, cartridge, candy, block and software
boxes and the hammer: 14-29 cm long, 2-6 cm thick) is entered end-on along its length by every side grasp: the housing
4.1-4.9 cm behind the jaw centre meets the near end once the half-length passes about 6.6 cm, and every one of their 27
diag rounds was knocked, closed on nothing or found no side. `is_top` takes it top-down: `pick_top` closes across the
thickness with the jaw TOP_INSET_M below the object's own top under the jaw (`top_grasp`; the descent to the support
tipped every tall box and stalled the palm on the low ones' tops 4-7 cm short), over its box centre (the hammer's
centre of mass lies 8 cm toward its head: gripped over it, one of 3 diag starts slipped in the lift and one missed the
pre-grasp; at the box centre the probe held 2 and the third hung 45 deg, its low corner 19.1 cm up), each step
through stage() (a missed descent or an empty close rises and comes down again), then lifts TOP_LIFT_M
(the jaw no higher than TOP_JAW_MAX_M) and holds TOP_HOLD_S so the goal holds over the judging window. A top-down
wrist is delivered when it is turned toward the body's midline (`top_yaw_pref`): on the right hand, over the 24
top-down rounds of iteration 0's probe, the pre-grasp attitude error was 8-24 deg with the closing yaw within 10 deg of
the facing and at most 2.5 deg from 50 deg on (the lift's 39-62 deg against 7-23); plan_approach takes the closing yaw
nearest TOP_YAW_PREF_DEG off the facing, and `top_pass` first the hand (the left one mirrored) and edge where it comes
within TOP_YAW_TOL_DEG.

The corridor (gso_single_grasp iteration 1, `corridor_pass`): `clear_pass` sees only objects fixed on the support and
only for a low grasp; the pallet's free neighbours stood in the side grasp's hand strip in 8 of the 10 side-grasp diag
failures, and the pre-grasp or the entry moved them before the hand met X, which they then shoved or tipped. A first
plan a free neighbour enters deeper than CORRIDOR_DEPTH_M (`corridor_intrusion`, the strip back past the pre-grasp)
takes the same hand at another edge, else a turned approach (CORRIDOR_TURN_DEG, the far hand first) whose jaw still
straddles X, whichever first has a clear strip, else the least entered one CORRIDOR_GAIN_M shallower; else it stands."""

from __future__ import annotations

import copy

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import SIDES, Outcome
from locomani.primitives.navigator import Navigator
from locomani.primitives.navigator import build as build_navigation
from locomani.tasks.scene import graspability
from locomani.tasks.scene import stage as S
from locomani.tasks.scene.navigate import path as nav_path

from . import cover, grasp_program
from . import grasp_common as GC

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
OTHER_EDGE_CLEAR_M = 0.15  # (0.25 until r2_on_single's set) a stance at an edge the scene file does not list: this clear of all but its own furniture
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
END_ON_M = 0.13  # a box whose only side fitting the jaw is entered end-on along more than this: top-down (`end_on`;
# the mug's 12.6 cm and the hard drive's 10.4 cm stay side grasps, the software box's 14.2 cm does not)
TOP_INSET_M = 0.035  # top-down: the jaw centre this far below the object's top under the jaw (housing 4.07 cm above it)
TOP_FOOT_M = 0.02  # ... that top read on vertical lines this far either side of the jaw along the object's long axis
TOP_YAW_PREF_DEG = 70.0  # top-down: the closing yaw this far off the facing toward the body's midline (+ right hand)
TOP_YAW_TOL_DEG = 35.0  # ... a hand and edge where it comes within this first (`top_pass`; measured -70 to +88 deg)
TOP_PRE_TOL_M = 0.04  # top-down pre-grasp check: the jaw within this of the point above the grasp
TOP_ACROSS_MARGIN_M = 0.0  # descent check: the jaw across its closing axis within the open jaw's room a side less this
TOP_HIGH_TOL_M = 0.025  # ... and at most this above the grasp (a finger standing on the top)
TOP_RETRIES = 1  # a missed pre-grasp, descent or close: this many re-approaches (stage's round budget too)
TOP_LIFT_M = 0.40  # top-down lift (the proxy's 0.40: 35-40 cm delivered) ...
TOP_JAW_MAX_M = 0.95  # (0.86 until gso_single_grasp's set) ... the jaw no higher than this (a tall box's top grasp starts at 0.60 m)
TOP_LIFT_MIN_M = 0.08  # lift check: X rose at least this
TOP_HOLD_S = 2.0  # hold after the lift, so the goal holds over the judging window
SPAWN_CLEAR_M = 0.08  # a fallback plan needs the start pelvis this far from every floor object footprint
SIDE_AHEAD_M = None  # default-off switch (place.py's pick): object this far ahead for a right-hand side grasp ...
SIDE_AHEAD_TOP_MIN_M = 0.6  # ... on a surface whose top is above this (None: STAND_AHEAD_M, held(X)'s pick)
GRIPPER_LOW_POINTS = ((-0.049, -0.034), (-0.0407, -0.031), (0.0333, -0.0145))  # (x along the approach, z up) in the
# jaw frame, robot.xml: the housing cylinder's and the housing mesh's front-bottom, the fingertip's front-bottom
WRIST_LOW_POINT = (-0.15, -0.045)  # the wrist and forearm (robot.xml right_wrist_yaw/pitch/roll and elbow links hang
# 2.9-3.5 cm below the jaw axis from 4 to 37 cm behind the jaw centre), with about 5 deg of nose-up attitude error ...
WRIST_OVER_M = 0.02  # ... counted where that point lies at least this far in from the edge (over the support)
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
NARROW_SIDE_MAX_M = 0.06  # an object under another surface whose only side fitting the jaw is narrower: NARROW
NARROW = dict(  # the narrow covered grasp's own values (grasp_program.run's and held.pick_held's)
    jaw_open=0.004,  # grasp_program.JAW_OPEN 0.0245
    jaw_closed=-0.06,  # JAW_CLOSED -0.02: the PD depth into the book, about 20 N instead of 10
    entry_capture_depth_m=0.05,  # SIMPLE's 0.025: the entry ends 5 cm short of the centre ...
    entry_overshoot_m=-0.05,  # ... and its command stops there (-0.01)
    disturb_limit_m=0.10,  # 0.05
    lateral_correction_max_m=0.0,  # 0.04
    gate_hold_s=0.0,  # ENTRY_GATE hold_s 3.0
    rise_min=0.015,  # RISE_MIN 0.05: a pivoted book still goes on to the pull
    pull_m=0.30,  # held.WITHDRAW_PULL_M 0.20
    max_m=0.35,  # the pull in all when after pull_m X is still under the shelves above
)
LOW_SPAN_M = 0.089  # a level side grasp whose pads would meet a wider hull (under 3 mm per side in the open jaw) ...
LOW_WIDTH_M = 0.091  # ... closes at the highest jaw below the centre where the collision hull is at most this wide
LOW_STEP_M = 0.0025  # over the pads' height (`low_grasp`), searched down in these steps
LOW_PITCH_DEG = 0.0  # the low grasp's wrist pitch nose-down (the jaw no lower than min_jaw of it allows)
CLEAR_YAW_DEG = 35.0  # a low grasp whose straight approach sweeps an object fixed on the support (`strip_blocker`)
# comes in this far off the facing, turned away from it, with the hand on the far side (`plan_hand`)
STRIP_HALF_M = 0.0575  # the hand's strip: grasp_program.PAD_Y and a finger either side of the approach line ...
STRIP_BACK_M = 0.35  # ... from the object centre back this far (fingers, housing, wrist and forearm)
CORRIDOR_DEPTH_M = 0.02  # a side grasp whose strip (back to STRIP_BACK_M + the pre-grasp's PREGRASP_BACK) a free
# neighbour enters deeper than this (`corridor_intrusion`) takes another edge or a turned approach (`corridor_pass`) ...
CORRIDOR_GAIN_M = 0.02  # ... the first whose strip is clear, else the least entered one when this much shallower
CORRIDOR_TURN_DEG = (20.0, 35.0, 50.0)  # ... the turned approaches it tries, each hand (the far one first)
LOW_PREGRASP_MISS_M = 0.035  # a low grasp's pre-grasp ending farther off fails without the hand program (place's
# PICK_SETTLE_ABORT_M, which aborts only after the settle)
LOW_LATERAL_ABORT_M = 0.03  # a low grasp's entry fails once its grasp point sits more than this across the jaw
# (grasp_program.SIMPLE lateral_abort_m): the vase's grasped entries stayed within 0.7 cm, the one that struck it
# drifted past 3 cm 4.1 s before the outer finger hit and it fell off its shelf
_NAV: dict = {}  # id(meta) -> (meta, occupancy, navigator, floor boxes): one map per scene and process
_CLEAR: dict = {}  # (id(occupancy), furniture) -> (occupancy, clearance grid without that furniture's surfaces)
_corridor_note: dict = {}  # the last plan_hand's corridor numbers (`corridor_pass`), logged on the walk step


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


def grasp_yaw(quat, half, face: float, tolerance_deg: float | None = None, hull=None) -> float | None:
    """The approach yaw (wrist x) closing the jaw across a side of the object's box that fits it, nearest `face`,
    within `tolerance_deg` of it (default FACE_TOLERANCE_DEG). A box that fits both ways is closed across at `face`
    itself unless `hull` (approach yaw -> the collision hull's span across the jaw, `hull_span`; None: not known) says
    the hull there spans more than JAW_SPAN_M: then across the nearest box face (object yaw + k 90 deg) where it fits,
    or None when that is beyond the tolerance (plan_hand's yawed pass takes it); `face` when no box face fits either.
    The square tapered plant pot (gso_single_grasp iteration 1) is 8.8 x 8.8 cm in its box but 10.7-10.9 cm across its
    corners at the pads' band: at the facing the jaw closed on two corners 34-48 deg off a face (two diag starts; one
    squirted out in the close, one was squeezed up out of the jaw in the hold), within 5 deg of a face it was held."""
    yaw_o = R.yaw_of(quat)
    fit_y, fit_x = 2 * half[1] <= JAW_SPAN_M, 2 * half[0] <= JAW_SPAN_M  # closing across local y / local x
    limit = FACE_TOLERANCE_DEG if tolerance_deg is None else tolerance_deg
    if fit_x and fit_y:
        wide = (lambda a: (hull(a) or 0.0) > JAW_SPAN_M) if hull is not None else (lambda a: False)
        if not wide(face):
            return face
        faces = [a for a in (yaw_o + k * np.pi / 2 for k in range(4)) if not wide(a)]
        if not faces:
            return face
        turn = min((R.wrap(a - face) for a in faces), key=abs)
        return float(face + turn) if abs(turn) <= np.radians(limit) else None
    cands = []
    if fit_y:
        cands += [yaw_o, yaw_o + np.pi]
    if fit_x:
        cands += [yaw_o + np.pi / 2, yaw_o - np.pi / 2]
    if not cands:
        return None
    best = min(cands, key=lambda a: abs(R.wrap(a - face)))
    return float(best) if abs(R.wrap(best - face)) <= np.radians(limit) else None


def hull_span(world, key: str, psi: float) -> float | None:
    """The span (m) across the jaw of a side grasp at approach yaw `psi` of X's collision hulls at its pose now, over
    the pads' height (grasp_program.PAD_Z either side) at the side grasp's jaw height (X's origin + `grasp_pitch`'s
    rise; graspability.section's cuts); None without a simulator model or a section there."""
    pieces = GC.pieces(world, key)
    if not pieces:
        return None
    pos, _ = world.object_pose(key)
    z = float(pos[2]) + grasp_pitch(world, key, pos)[1]
    d = max(0.0, grasp_program.PAD_Z - graspability.SECTION_BAND_M)
    b = graspability.SECTION_BAND_M
    levels = [z + s + k for s in (-d, d) for k in (-b, 0.0, b)]
    return GC.cuts_width(pieces, levels, (-np.sin(psi), np.cos(psi)))


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
    objects under no shelf dropped nothing. A low grasp (`low_grasp`, its outcome's info "low") is not retried either:
    on a fresh vase start (r2_on_single iteration 2) the first entry snagged and leaned the vase on the hand, and the
    back-out pulled the hand away under it; it fell to the floor."""
    world, m = ctx.world, ctx.motor
    if out.ok or not rest or m.fell_at is not None or key not in world.free or out.info.get("low"):
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
    the jaw raised for a short object (`grasp_pitch`), pitched and lowered below the centre for one too wide to enter
    level (`low_grasp`, closing across its hull's span there; its outcome's info "low" True; failed without the hand
    program when its pre-grasp ends more than LOW_PREGRASP_MISS_M off), with NARROW's values for
    a `narrow_covered` one (its outcome's info "withdraw" carries NARROW's pull for held.pick_held); a low flat object
    (`is_top`) is pinched from above instead (`pick_top`). A side grasp or pinch records the hand and its walking jaw
    pose (relative to the pelvis, in its yaw frame) in `rest` before the pre-grasp, for `back_out`."""
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
    pitch, rise = grasp_pitch(world, key, pos, wrist_in(plan.info))
    if pitch and needs_pinch(world, key, pos, pitch, rise):
        return pick_pinch(ctx, key, side, plan)
    narrow = narrow_covered(world, key)
    low = None if pitch or narrow else low_grasp(world, key, pos, quat, psi, wrist_in(plan.info))
    pitched = {}
    if low is not None:
        pitch, rise, _ = low
        pitched = dict(low_width_cm=round(100 * low[2], 2))
    if pitch:
        Rg = Rg @ R.rot_y(pitch)
        pitched.update(pitch_deg=round(float(np.degrees(pitch)), 1), jaw_rise_cm=round(100 * rise, 2))
    up = np.array([0.0, 0.0, rise])
    pre = pos + up - Rg @ np.array([grasp_program.PREGRASP_BACK, 0.0, 0.0])
    via = edge_via(ctx, key, side, plan, pre)
    out = ctx.note("pregrasp", m.reach(side, pre, Rg, via=via), via=[np.round(v, 3).tolist() for v in via], **pitched)
    if m.fell_at is not None:
        return out
    miss = None if low is None else float(np.linalg.norm(m.jaw_centre(side)[0] - pre))
    if miss is not None and miss > LOW_PREGRASP_MISS_M:  # a snagged hand's settle swept a vase off its shelf
        return Outcome(False, f"pregrasp missed by {100 * miss:.1f} cm (no settle after a low grasp's)", out.t0, m.t,
                       dict(low=True))  # fmt: skip
    pos, _ = world.object_pose(key)
    width = span(quat, world.meta["objects"][key]["box"]["half_extent"], psi) if low is None else low[2]
    if not narrow:
        if low is None:
            return grasp_program.run(ctx, key, side, Rg, pos + up, width)
        out = grasp_program.run(ctx, key, side, Rg, pos + up, width, simple=dict(lateral_abort_m=LOW_LATERAL_ABORT_M))
        return Outcome(out.ok, out.why, out.t0, out.t1, dict(out.info, low=True))
    n = NARROW
    simple = {k: n[k] for k in ("entry_capture_depth_m", "entry_overshoot_m", "disturb_limit_m",
                                "lateral_correction_max_m")}  # fmt: skip
    out = grasp_program.run(ctx, key, side, Rg, pos + up, width, simple=simple, gate=dict(hold_s=n["gate_hold_s"]),
                            jaw=(n["jaw_open"], n["jaw_closed"]), rise_min=n["rise_min"])  # fmt: skip
    withdraw = {k: n[k] for k in ("pull_m", "max_m")}
    return Outcome(out.ok, out.why, out.t0, out.t1, dict(out.info, withdraw=withdraw))


def narrow_covered(world, key: str) -> bool:
    """X is under another surface (cover.overhead) and the only side of its box that fits the jaw (JAW_SPAN_M) is
    narrower than NARROW_SIDE_MAX_M: the thick novel (4.7 cm; 16.6 cm along the other side), not the wallet (8.7 cm,
    under no shelf)."""
    half = world.meta["objects"][key]["box"]["half_extent"]
    fits = [2 * float(h) for h in half[:2] if 2 * float(h) <= JAW_SPAN_M]
    return len(fits) == 1 and fits[0] < NARROW_SIDE_MAX_M and bool(cover.overhead(world, key))


def min_jaw(pitch: float, points=None, clear: float | None = None) -> float:
    """The least jaw-centre height (m) above the support top of a side grasp pitched nose-down by `pitch` (rad) that
    keeps every point of `points` (default GRIPPER_LOW_POINTS) `clear` (default GRASP_CLEAR_M) above the top
    (rot_y(pitch) puts the point (x, z) at -x sin + z cos): 3.9 cm level, 2.76 cm at 15 deg; with WRIST_LOW_POINT too
    5.0 cm level and still 2.76 cm at 15 deg; with PINCH_LOW_POINTS 2.13 cm at 25 deg."""
    s, c = np.sin(pitch), np.cos(pitch)
    low = float(max(x * s - z * c for x, z in (GRIPPER_LOW_POINTS if points is None else points)))
    return low + (GRASP_CLEAR_M if clear is None else clear)


def low_points(wrist_in: float | None = None) -> tuple:
    """The gripper's low outline for `min_jaw`: GRIPPER_LOW_POINTS, and WRIST_LOW_POINT when it lies `wrist_in` (m) in
    from the edge, at least WRIST_OVER_M (None: not known, not counted)."""
    over = wrist_in is not None and wrist_in >= WRIST_OVER_M
    return (*GRIPPER_LOW_POINTS, WRIST_LOW_POINT) if over else tuple(GRIPPER_LOW_POINTS)


def wrist_in(info: dict) -> float | None:
    """How far (m) in from the edge WRIST_LOW_POINT lies for the approach `info` (approach's: edge_depth_m, psi, face)
    with the jaw at the object centre; None without an edge depth."""
    if info.get("edge_depth_m") is None:
        return None
    return float(info["edge_depth_m"]) + WRIST_LOW_POINT[0] * float(np.cos(info["psi"] - info["face"]))


def grasp_pitch(world, key: str, pos, wrist: float | None = None) -> tuple[float, float]:
    """(pitch nose-down (rad), jaw rise (m)) of a side grasp at the object centre `pos`, the wrist point `wrist` m in
    from the edge (`low_points`): (0, 0) unless the level jaw there would put the gripper below the support top (centre
    less than min_jaw(0) - GRASP_CLEAR_M above it) and no surface covers the object (`cover.overhead`); then
    GRASP_PITCH_DEG, the jaw raised to the top + min_jaw(GRASP_PITCH_DEG) where the centre is lower."""
    top = float(world.meta["surfaces"][world.meta["objects"][key]["surface"]]["top"])
    h = float(pos[2]) - top
    points = low_points(wrist)
    if h >= min_jaw(0.0, points) - GRASP_CLEAR_M or cover.overhead(world, key):
        return 0.0, 0.0
    pitch = float(np.radians(GRASP_PITCH_DEG))
    return pitch, max(0.0, min_jaw(pitch, points) - h)


def needs_pinch(world, key: str, pos, pitch: float, rise: float) -> bool:
    """A pitched side grasp (`grasp_pitch`'s pitch and rise at the object centre `pos`) whose pads' lower edge there
    (grasp_program.PAD_Z below the jaw axis) would overlap the object's top by less than PINCH_OVERLAP_M: the car key
    (1.5 cm high) by 0.1 cm, the wallet by 1.35 cm, the candles and the bin by 3.6-3.9 cm."""
    o = world.meta["objects"][key]
    top_obj = float(pos[2]) + float(o["box"]["half_extent"][2])
    pad_low = float(pos[2]) + rise - grasp_program.PAD_Z * np.cos(pitch)
    return top_obj - pad_low < PINCH_OVERLAP_M


def low_grasp(world, key: str, pos, quat, psi: float, wrist: float | None = None) -> tuple[float, float, float] | None:
    """(pitch nose-down (rad), jaw rise (m, negative), width (m)) of a side grasp below the centre `pos` of an object
    under no other surface whose collision hull spans more than LOW_SPAN_M across the jaw (approach yaw `psi`) over
    the pads' height at the centre (grasp_program.PAD_Z either side, `band_width`): GRASP_PITCH_DEG, the highest jaw
    height, LOW_STEP_M steps down from the centre and last the lowest the gripper allows, at which the hull spans at
    most LOW_WIDTH_M over the pads' height, and that span. None for a narrower hull (its box is read first: the mesh
    only when the box spans more than LOW_SPAN_M -- a round can whose box fits both ways spans 9.5 cm at 45 deg), a
    covered object, or when no such
    height keeps the pitched gripper above the support top (`min_jaw` of `low_points(wrist)`). The big white vase
    (r2_on_single iteration 1) is round, 9.2 cm across at its equator, where the level grasp closes: 1.6 mm per side
    in the 9.5 cm open jaw, and a finger met it in the entry. About 5 cm below the centre it spans 8.6 cm and widens
    above the pads, so its weight wedges it in; closed 4.5 cm above the centre, where it narrows upward, it entered
    cleanly and slid out of the jaw within 4 s. Closed-loop (the clear pass's 35 deg approach, 2-3 diag starts each):
    pitched 15 deg, 5.25 cm below the centre (LOW_WIDTH_M 0.086), the vase tipped about the pads' contact line as the
    wrist pitched to 22-31 deg and fell within 9 s; level at the centre it slid down the pads 0.13-0.16 cm/s and fell
    after 5 cm; level 3 cm below it (9.08 cm across the pads' top, 2.1 mm per side) it slid 0.9 cm in 70 s and was
    carried to the TV stand, but leaned 28-31 deg there (the wrist drifted 10-24 deg nose-up in the carry). With the
    wrist over the shelf the lowest jaw is 3.09 cm below the vase's centre: the steps alone stopped at 3 cm, where it
    spans 9.06-9.10 cm by its yaw, and missed diag s48 (9.10, 9.09 cm at the lowest; iteration 2)."""
    o = world.meta["objects"][key]
    if span(quat, o["box"]["half_extent"], psi) <= LOW_SPAN_M or cover.overhead(world, key):
        return None
    verts = object_vertices(world.motor, world.free[key])
    if len(verts) < 4:
        return None
    across = R.rot_z(psi)[:, 1]
    z0 = float(pos[2])
    if band_width(verts, z0 - grasp_program.PAD_Z, z0 + grasp_program.PAD_Z, across) <= LOW_SPAN_M:
        return None
    pitch = float(np.radians(LOW_PITCH_DEG))
    floor = float(world.meta["surfaces"][o["surface"]]["top"]) + min_jaw(pitch, low_points(wrist))
    steps = [z0 - k * LOW_STEP_M for k in range(1, int((z0 - floor) / LOW_STEP_M) + 1)]
    for z in steps + ([floor] if floor < z0 else []):
        w = band_width(verts, z - grasp_program.PAD_Z, z + grasp_program.PAD_Z, across)
        if w <= LOW_WIDTH_M:
            return pitch, z - z0, w
    return None


def band_width(verts, lo: float, hi: float, across) -> float:
    """The extent along the horizontal `across` of the convex hull of `verts` (the simulator collides a mesh's hull)
    between the heights `lo` and `hi`: of its vertices there and its edges' crossings of both planes (0 without any);
    grasp_common.band_width of the one hull."""
    return GC.band_width(GC.prepare([verts]), lo, hi, across)


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


def plan_approach(
    world,
    start_xy,
    key: str,
    side: str,
    max_depth: float | None = None,
    yawed: bool = False,
    yaw_off: float | None = None,
    top_tol: float | None = None,
    exclude: tuple = (),
):
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
    No yawed plan on a surface above YAWED_TOP_MAX_M. `yaw_off` (rad, with `yawed`; `plan_hand`'s clear pass): psi is
    the facing + yaw_off, for a box that fits the jaw both ways. A top-down plan (`is_top`) takes the closing yaw
    nearest the facing + `top_yaw_pref(side)`; with `top_tol` (rad, `top_pass`) only at edges where it comes that near.
    `exclude` (`corridor_pass`): facings (rad) whose edges are skipped."""
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
    half = obj["box"]["half_extent"]
    pref = top_yaw_pref(side)
    spans: dict = {}

    def hull(a):  # hull_span, once per yaw
        k = round(float(R.wrap(a)), 6)
        if k not in spans:
            spans[k] = hull_span(world, key, a)
        return spans[k]

    def fit(q, h, f):
        if not top_pinch:
            return grasp_yaw(q, h, f, hull=hull)
        psi = _nearest_closing_yaw(q, h, f, pref)
        return None if top_tol is not None and abs(R.wrap(psi - f - pref)) > top_tol else psi

    why = "no side fits the jaw within reach of an approachable edge"
    if yawed and float(surf["top"]) > YAWED_TOP_MAX_M:
        return None, why
    max_depth = MAX_DEPTH_M if max_depth is None else max_depth
    ahead = stand_ahead(float(surf["top"]), side, top_pinch)
    for f, dep, is_listed in edges:
        if any(abs(R.wrap(f - x)) < 1e-6 for x in exclude):
            continue
        if dep > max_depth:
            psi = None
        elif yawed and not top_pinch and yaw_off is not None:
            fits = all(2 * float(h) <= JAW_SPAN_M for h in obj["box"]["half_extent"][:2])
            psi = float(f + yaw_off) if fits else None
        elif yawed and not top_pinch:
            psi = grasp_yaw(quat, obj["box"]["half_extent"], f, FACE_TOLERANCE_YAWED_DEG, hull=hull)
        else:
            psi = fit(quat, half, f)
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
    why. A top-down object (`is_top`) first takes `top_pass`, the hand and edge whose wrist yaw is delivered. The first
    pass's side grasp, when it is no `clear_pass` and a free neighbour stands in its hand's strip, takes `corridor_pass`
    (another edge or a turned approach whose strip is clear)."""
    _corridor_note.clear()
    if is_top(world, key):
        found = top_pass(world, start_xy, key, side)
        if found is not None:
            return found
    plan, why = plan_approach(world, start_xy, key, side)
    if plan is not None:
        found = clear_pass(world, start_xy, key, plan) or corridor_pass(world, start_xy, key, side, plan)
        return found or (plan, side, "")
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


def top_pass(world, start_xy, key: str, side: str) -> tuple | None:
    """`plan_hand`'s first pass for a top-down object: (plan, hand, why) of the first of `side`, the other hand (stance
    mirrored), each out to MAX_DEPTH_M and then TOP_MAX_DEPTH_M, that plans (`plan_approach`) at an edge where the
    closing yaw comes within TOP_YAW_TOL_DEG of `top_yaw_pref` -- the two closing yaws 180 deg apart put one of the
    hands there from 110 deg of the 180 the object's yaw takes against an edge; None when none does (the plain passes
    then take the nearest yaw). Any but the first is refused as plan_hand's fallbacks are, from a start against a floor
    object."""
    other = "left" if side == "right" else "right"
    tol = float(np.radians(TOP_YAW_TOL_DEG))
    for k, (hand, limit) in enumerate(((side, MAX_DEPTH_M), (other, MAX_DEPTH_M), (side, TOP_MAX_DEPTH_M),
                                       (other, TOP_MAX_DEPTH_M))):  # fmt: skip
        plan, _ = plan_approach(world, start_xy, key, hand, limit, top_tol=tol)
        if plan is None:
            continue
        if k and floor_clearance(nav_map(world)[2], start_xy) < SPAWN_CLEAR_M:
            return None, hand, "start against a floor object"
        return plan, hand, ""
    return None


def clear_pass(world, start_xy, key: str, plan) -> tuple | None:
    """`plan_hand`'s pass for a low grasp: when the side grasp `plan` (plan_approach's) would be a `low_grasp` whose
    straight approach sweeps an object fixed on the support (`strip_blocker`), (plan, hand, "") of plan_approach turned
    CLEAR_YAW_DEG away from it (yaw_off, the stance placed from the wrist), with the hand on the far side: the left one
    when the blocker is on the approach's right. None otherwise, or when that has no plan (the straight plan stands).
    The big white vase on the wide display unit's middle shelf stands behind a fixed small white vase 5-9 cm to its
    side (every diag start): the fingers of every straight entry met it below its 0.82 m top, the entries stalled
    13-14 cm short or knocked the vase; turned 35 deg with the left hand, 3 of 3 entered within 2 mm."""
    face, depth, psi, _, _ = plan
    pos, quat = world.object_pose(key)
    wrist = wrist_in(dict(edge_depth_m=depth, psi=psi, face=face))
    if is_top(world, key) or grasp_pitch(world, key, pos, wrist)[0] or narrow_covered(world, key):
        return None
    if low_grasp(world, key, pos, quat, psi, wrist) is None:
        return None
    lateral = strip_blocker(world, key, psi)
    if lateral is None:
        return None
    hand, off = ("left", -np.radians(CLEAR_YAW_DEG)) if lateral < 0 else ("right", np.radians(CLEAR_YAW_DEG))
    turned, _ = plan_approach(world, start_xy, key, hand, None, True, off)
    return None if turned is None else (turned, hand, "")


def strip_blocker(world, key: str, psi: float) -> float | None:
    """The signed offset (m, + to the left of the approach yaw `psi`) of the first object fixed on X's support (not
    free, its underside within 5 cm of the top) whose box footprint at its pose now meets the hand's strip:
    STRIP_HALF_M either side of the approach line through X's centre, from STRIP_BACK_M behind it to it; None."""
    o = world.meta["objects"][key]
    top = float(world.meta["surfaces"][o["surface"]]["top"])
    pos, _ = world.object_pose(key)
    u = np.array([np.cos(psi), np.sin(psi)])
    n = np.array([-u[1], u[0]])
    for k, other in world.meta["objects"].items():
        if k == key or other.get("free"):
            continue
        p, q = world.object_pose(k)
        half = np.asarray(other["box"]["half_extent"], float)
        if abs(float(p[2] - half[2]) - top) > 0.05:
            continue
        Ro = R.rot_z(R.yaw_of(q))[:2, :2]
        c = np.array([Ro @ (half[:2] * [a, b]) for a in (-1, 1) for b in (-1, 1)]) + (p[:2] - pos[:2])
        s, t = c @ u, c @ n
        if s.max() >= -STRIP_BACK_M and s.min() <= 0.0 and t.max() >= -STRIP_HALF_M and t.min() <= STRIP_HALF_M:
            return float((p[:2] - pos[:2]) @ n)
    return None


def _clip(poly: np.ndarray, d, lo: float, hi: float) -> np.ndarray:
    """The convex polygon `poly` (k, 2; in order) clipped to lo <= p @ d <= hi (Sutherland-Hodgman, two planes)."""
    for sign, bound in ((1.0, hi), (-1.0, -lo)):
        out = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            fa, fb = sign * float(a @ d) - bound, sign * float(b @ d) - bound
            if fa <= 0.0:
                out.append(a)
            if fa * fb < 0.0:
                out.append(a + (fa / (fa - fb)) * (b - a))
        poly = np.asarray(out, float).reshape(-1, 2)
        if not len(poly):
            break
    return poly


def corridor_intrusion(world, key: str, psi: float, low: float) -> tuple[float, float, str | None]:
    """(depth (m), signed offset (m, + to the left of the approach yaw `psi`), key) of the free object that enters the
    side grasp's hand strip deepest at its pose now: STRIP_HALF_M either side of the approach line through X's centre,
    from STRIP_BACK_M + grasp_program.PREGRASP_BACK behind it (the arm at the pre-grasp) to it, its box's footprint
    reaching up above the hand's lowest point `low` (world z); the depth is STRIP_HALF_M less the footprint's least
    distance from the line inside that span. (0, 0, None) when none enters. Iteration 1's diag pool (incumbent
    6b5f36afa742): of 33 side-grasp rounds, the 8 a neighbour entered 2.9-5.8 cm deep failed (the neighbour moved in
    the pre-grasp or the entry before the hand met X, then shoved or tipped X), the 6 it entered 0.5-2.5 cm all held;
    two held at 5.8 cm."""
    pos, _ = world.object_pose(key)
    u = np.array([np.cos(psi), np.sin(psi)])
    n = np.array([-u[1], u[0]])
    back = float(STRIP_BACK_M) + float(grasp_program.PREGRASP_BACK)
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    best = (0.0, 0.0, None)
    for k, other in world.meta["objects"].items():
        if k == key or not other.get("free") or not other.get("box"):
            continue
        p, q = world.object_pose(k)
        half = np.asarray(other["box"]["half_extent"], float)
        centre = np.asarray(other["box"].get("centre") or (0.0, 0.0, 0.0), float)
        corners = np.asarray(p, float) + (centre + signs * half) @ R.mat(q).T
        if float(corners[:, 2].max()) <= low:
            continue
        xy = corners[:, :2] - pos[:2]
        try:
            poly = xy[ConvexHull(xy).vertices]
        except Exception:  # noqa: BLE001 -- a degenerate footprint (no area) is no blocker
            continue
        part = _clip(poly, u, -back, 0.0)
        if not len(part):
            continue
        t = part @ n
        gap = 0.0 if t.min() <= 0.0 <= t.max() else float(np.abs(t).min())
        depth = float(STRIP_HALF_M) - gap
        if depth > best[0]:
            best = (depth, float(0.5 * (t.min() + t.max())), k)
    return best


def _hand_low(world, key: str, plan) -> float:
    """The world z of the side grasp's lowest gripper point at X (`grasp_pitch`'s pitch and rise, `low_points`)."""
    face, depth, psi, _, _ = plan
    pos, _ = world.object_pose(key)
    wrist = wrist_in(dict(edge_depth_m=depth, psi=psi, face=face))
    pitch, rise = grasp_pitch(world, key, pos, wrist)
    return float(pos[2]) + rise - (min_jaw(pitch, low_points(wrist)) - GRASP_CLEAR_M)


def _turn_fits(world, key: str, psi: float, z: float) -> bool:
    """A turned approach's jaw straddles X at the grasp height: graspability.fit on its collision hulls where a model
    is at hand, else the box's span across the jaw within JAW_SPAN_M."""
    if getattr(world, "sim", None) is not None or getattr(world, "m", None) is not None:
        return graspability.fit(world, key, psi, z)["verdict"] != "too_wide"
    _, quat = world.object_pose(key)
    return span(quat, world.meta["objects"][key]["box"]["half_extent"], psi) <= JAW_SPAN_M


def corridor_pass(world, start_xy, key: str, side: str, plan) -> tuple | None:
    """`plan_hand`'s pass for a side grasp whose first plan `plan` (plan_approach's, hand `side`) has a free neighbour
    in its hand's strip deeper than CORRIDOR_DEPTH_M (`corridor_intrusion`): (plan, hand, "") of the first clear one
    (at most CORRIDOR_DEPTH_M) of -- the same hand at each other edge, nearest first; for a box that fits the jaw both
    ways the approach turned by each of CORRIDOR_TURN_DEG away from the neighbour with the far hand, then the other way
    with the other hand (plan_approach yaw_off, as `clear_pass`), where its jaw still straddles X (`_turn_fits`) --
    else of the least entered one (the first of equals) when CORRIDOR_GAIN_M shallower than
    `plan`'s. None (the first plan stands) otherwise, for a top-down
    or narrow covered object, or from a start against a floor object. Iteration 1's diag pool: in 8 of
    the 10 side-grasp failures the pre-grasp or the entry moved a neighbour standing in the strip before the hand met
    X, which then shoved or tipped X; the strip test against fixed objects only (`strip_blocker`) never saw them.
    Records the numbers in `_corridor_note` for the walk step."""
    if is_top(world, key) or narrow_covered(world, key):
        return None
    pos, _ = world.object_pose(key)
    face, depth, psi, _, _ = plan
    _, rise = grasp_pitch(world, key, pos, wrist_in(dict(edge_depth_m=depth, psi=psi, face=face)))
    deep, lateral, blocker = corridor_intrusion(world, key, psi, _hand_low(world, key, plan))
    _corridor_note.update(corridor_cm=round(100 * deep, 1), blocker=blocker)
    if deep <= CORRIDOR_DEPTH_M or floor_clearance(nav_map(world)[2], start_xy) < SPAWN_CLEAR_M:
        return None
    cands, seen = [], [face]
    while True:  # the same hand at the other edges, nearest first
        alt, _ = plan_approach(world, start_xy, key, side, exclude=tuple(seen))
        if alt is None:
            break
        seen.append(alt[0])
        cands.append(("edge", alt, side))
    half = world.meta["objects"][key]["box"]["half_extent"]
    if all(2 * float(h) <= JAW_SPAN_M for h in half[:2]):
        hands = [("right", 1.0), ("left", -1.0)] if lateral >= 0 else [("left", -1.0), ("right", 1.0)]
        for hand, sign in hands:
            for deg in CORRIDOR_TURN_DEG:
                alt, _ = plan_approach(world, start_xy, key, hand, None, True, sign * float(np.radians(deg)))
                if alt is not None and _turn_fits(world, key, alt[2], float(pos[2]) + rise):
                    cands.append(("turn", alt, hand))
    scored = []
    for kind, alt, hand in cands:
        d, _, b = corridor_intrusion(world, key, alt[2], _hand_low(world, key, alt))
        scored.append((d, kind, alt, hand, b))
    clear = [c for c in scored if c[0] <= CORRIDOR_DEPTH_M]
    best = clear[0] if clear else min(scored, key=lambda c: c[0], default=None)
    if best is None or (not clear and best[0] > deep - CORRIDOR_GAIN_M):
        _corridor_note.update(corridor_kept=len(scored))
        return None
    d, kind, alt, hand, b = best
    _corridor_note.update(corridor_pass=kind, corridor_new_cm=round(100 * d, 1), corridor_new_blocker=b,
                          corridor_turn_deg=round(float(np.degrees(R.wrap(alt[2] - alt[0]))), 1))  # fmt: skip
    return alt, hand, ""


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
    half = world.meta["objects"][key]["box"]["half_extent"]
    turned = _corridor_note.get("corridor_pass") == "turn"  # `corridor_pass`'s turn is logged as corridor_turn_deg
    if not is_top(world, key) and not turned and abs(off) > 0.5 and all(2 * float(h) <= JAW_SPAN_M for h in half[:2]):
        yawed["hull_turn_deg"] = round(off, 1)  # `grasp_yaw`: a box that fits both ways, turned to a face by its hull
    out = ctx.note("walk", walk_route(m, occ, nav, stance, face), edge_depth_m=round(depth, 3),
                   face_deg=round(float(np.degrees(face)), 1), stance=np.round(stance, 3), side=side,
                   **yawed, **dict(_corridor_note))  # fmt: skip
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
    """An object on a low surface (top below TOP_SURFACE_MAX_M) taken top-down: a low flat one (box half-height below
    TOP_FLAT_M: the TV remote on the coffee table; its only side that fits the jaw is its 3.6 cm width, so a side entry
    runs along its 16.4 cm length and the palm meets its near end about 12 cm short), or an end-on long box (`end_on`)."""
    o = world.meta["objects"][key]
    top = float(world.meta["surfaces"][o["surface"]]["top"])
    return top < TOP_SURFACE_MAX_M and (o["box"]["half_extent"][2] < TOP_FLAT_M or end_on(world, key))


def end_on(world, key: str) -> bool:
    """X's box has one horizontal side that fits the jaw (JAW_SPAN_M) and the other longer than END_ON_M: every side
    grasp across the one that fits (`grasp_yaw`) enters along that length, and the housing meets its near end."""
    w = sorted(2.0 * float(h) for h in world.meta["objects"][key]["box"]["half_extent"][:2])
    return w[0] <= JAW_SPAN_M and w[1] > END_ON_M


def top_yaw_pref(side: str) -> float:
    """The top-down closing yaw's preferred offset from the facing (rad): TOP_YAW_PREF_DEG toward the body's midline,
    + for the right hand, - for the left (mirrored)."""
    return float(np.radians(TOP_YAW_PREF_DEG)) * (1.0 if side == "right" else -1.0)


def _nearest_closing_yaw(quat, half, face: float, pref: float = 0.0) -> float:
    """The wrist yaw of a top-down pinch closing across the object's local y when that fits the jaw, else across its
    local x, nearest `face` + `pref` (no FACE_TOLERANCE_DEG: the hand comes from above)."""
    yaw_o = R.yaw_of(quat)
    cands = [yaw_o, yaw_o + np.pi] if 2 * half[1] <= JAW_SPAN_M else [yaw_o + np.pi / 2, yaw_o - np.pi / 2]
    return float(min(cands, key=lambda a: abs(R.wrap(a - face - pref))))


def _line_top(parts, xy) -> float:
    """The highest point of the convex point sets `parts` (world) on the vertical line through `xy` (-inf: none)."""
    best = -np.inf
    for v in parts:
        if len(v) < 4:
            continue
        try:
            eq = ConvexHull(v).equations
        except Exception:  # noqa: BLE001 -- a flat or degenerate part has no volume to stand on
            continue
        rhs = -eq[:, 3] - eq[:, 0] * xy[0] - eq[:, 1] * xy[1]
        flat = np.abs(eq[:, 2]) < 1e-9
        if np.any(rhs[flat] < -1e-9):
            continue
        up, down = eq[:, 2] > 1e-9, eq[:, 2] < -1e-9
        hi = float(np.min(rhs[up] / eq[up, 2])) if up.any() else np.inf
        lo = float(np.max(rhs[down] / eq[down, 2])) if down.any() else -np.inf
        if lo <= hi:
            best = max(best, hi)
    return best


def top_grasp(world, key: str, psi: float) -> tuple[np.ndarray, dict]:
    """(jaw centre, numbers) of the top-down grasp closing along rot_z(psi)'s y at X's pose now: over X's box centre,
    the jaw TOP_INSET_M below X's top under the jaw -- the highest point of its collision hulls on vertical lines
    TOP_FOOT_M either side along the long axis and across the open jaw (`_line_top`; the box's top when none meets it,
    or in a world without a simulator) -- and no lower than the support top + FINGERTIP_M + TOP_TIP_CLEAR_M (a flat
    object's fingertips over the slab)."""
    o = world.meta["objects"][key]
    pos, quat = world.object_pose(key)
    box = o["box"]
    centre = np.asarray(pos, float) + R.mat(quat) @ np.asarray(box.get("centre", (0.0, 0.0, 0.0)), float)
    u = np.array([np.cos(psi), np.sin(psi)])  # the long axis (the jaw closes across it)
    n = np.array([-u[1], u[0]])
    xy = centre[:2].copy()
    parts = graspability.world_parts(world, key) if getattr(world, "sim", None) is not None else []
    across = np.linspace(-1.0, 1.0, 5) * grasp_program.PAD_Y
    lines = [xy + a * u + b * n for a in (-TOP_FOOT_M, 0.0, TOP_FOOT_M) for b in across]
    top_obj = max((_line_top(parts, p) for p in lines), default=-np.inf)
    if not np.isfinite(top_obj):
        top_obj = float(centre[2]) + float(box["half_extent"][2])
    support = float(world.meta["surfaces"][o["surface"]]["top"])
    z = max(top_obj - TOP_INSET_M, support + FINGERTIP_M + TOP_TIP_CLEAR_M)
    info = dict(top_above_support_cm=round(100 * (top_obj - support), 1),
                jaw_below_top_cm=round(100 * (top_obj - z), 1))  # fmt: skip
    return np.r_[xy, z], info


def pick_top(ctx, key: str, side: str, plan: Outcome) -> Outcome:
    """Top-down pinch after `approach`: wrist x down, closing across the narrow side (plan psi) at `top_grasp`, each
    step through stage(): the pre-grasp TOP_ABOVE_M above it (within TOP_PRE_TOL_M, else reached again); the descent
    (the jaw across its closing axis within the open jaw's room a side less TOP_ACROSS_MARGIN_M and at most
    TOP_HIGH_TOL_M above the grasp, else up to the pre-grasp at X's pose now and down again); the close (both fingers
    on X, else open, up, down and close again), TOP_RETRIES re-approaches each; the lift TOP_LIFT_M (the jaw no higher
    than TOP_JAW_MAX_M; X must rise TOP_LIFT_MIN_M), then TOP_HOLD_S held."""
    world, m = ctx.world, ctx.motor
    body = world.free[key]
    psi = plan.info["psi"]
    Rg = R.rot_z(psi) @ R.rot_y(np.pi / 2)
    _, quat = world.object_pose(key)
    width = span(quat, world.meta["objects"][key]["box"]["half_extent"], psi)
    across_tol = max(0.005, grasp_program.PAD_Y - 0.5 * width - TOP_ACROSS_MARGIN_M)  # beyond: a finger on X's top
    goal: dict = {}

    def aim():
        goal["grasp"], goal["info"] = top_grasp(world, key, psi)
        goal["above"] = goal["grasp"] + np.array([0.0, 0.0, TOP_ABOVE_M])

    def retry(fn):
        return lambda attempt, values: fn if attempt < int(TOP_RETRIES) else None

    def descend():
        aim()
        return m.reach(side, goal["grasp"], Rg, speed=TOP_DESCENT_SPEED)

    def descend_check(_out):
        d = m.jaw_centre(side)[0] - goal["grasp"]
        across, high = abs(float(d @ Rg[:, 1])), float(d[2])
        values = dict(jaw_err_cm=round(100 * float(np.linalg.norm(d)), 2), across_cm=round(100 * across, 2),
                      high_cm=round(100 * high, 2), track_ratio=S.track_ratio(m, side), **goal["info"])  # fmt: skip
        return across <= across_tol and high <= TOP_HIGH_TOL_M, values

    def again():  # up to the pre-grasp over X as it is now, and down again
        aim()
        up = ctx.note("top_rise", m.reach(side, goal["above"], Rg))
        return up if m.fell_at is not None else descend()

    def regrip():
        opened = ctx.note("top_open", m.release(side))
        if m.fell_at is not None:
            return opened
        down = again()
        return down if m.fell_at is not None else m.grip(side)

    aim()
    via = edge_via(ctx, key, side, plan, goal["above"])
    pre = goal["above"]
    out = S.stage(ctx, "pregrasp", lambda: m.reach(side, pre, Rg, via=via),
                  lambda o: S.reach_check(m, side, pre, TOP_PRE_TOL_M), retry(lambda: m.reach(side, pre, Rg)))  # fmt: skip
    if not out.ok:
        return out
    out = S.stage(ctx, "descend", descend, descend_check, retry(again))
    if not out.ok:
        return out
    out = S.stage(ctx, "grip", lambda: m.grip(side), lambda o: S.grip_check(m, side, body), retry(regrip))
    if not out.ok:
        return out
    jaw0 = m.jaw_centre(side)[0]
    z0 = float(world.object_pose(key)[0][2])
    rise = max(0.0, min(float(TOP_LIFT_M), float(TOP_JAW_MAX_M) - float(jaw0[2])))
    up = jaw0 + np.array([0.0, 0.0, rise])
    out = S.stage(ctx, "lift", lambda: m.reach(side, up, Rg, speed=ENTRY_SPEED),
                  lambda o: S.lift_check(world, key, z0, TOP_LIFT_MIN_M))  # fmt: skip
    if not out.ok:
        return out
    out = ctx.note("hold_up", m.hold(TOP_HOLD_S), lift_cm=round(100 * rise, 1))
    held = body in m.between_fingers(side)
    return Outcome(held, "" if held else "not between the fingers after the lift", out.t0, out.t1)
