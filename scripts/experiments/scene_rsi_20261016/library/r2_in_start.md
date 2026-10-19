# Experience library — scene RSI, as of r2_in_single's start (round 1, then r2_on_single and write-back 7)

What earlier families established about this robot, its skills and the scene, one entry per trigger. A trigger is
what you see in a round: a step name, a `why` / `plan_why` string, or a number in a step. Each entry gives the rule
(with the parameter that controls it), the evidence behind it, and its limits. The values are the incumbent's; the
context's parameter list is authoritative, and where the code or a round disagrees with an entry, trust them and say
which entry is wrong. Sources: the skill docstrings (`pick.py`, `place.py`, `into.py`, `near.py`, `grasp_program.py`)
and the plan (round-1 sections, r2_on_single).

## Pick (`pick.py`, `grasp_program.py`)

**P1. A side must fit the jaw.** Trigger: `pick refused: no side fits the jaw within reach of an approachable edge`.
Rule: a box side fits when its full width is at most `pick.JAW_SPAN_M` 0.092 (the open Dex1 jaw is 9.5 cm) and it
faces the robot within `FACE_TOLERANCE_DEG` 40; a last pass allows `FACE_TOLERANCE_YAWED_DEG` 70, but only on tops
below `YAWED_TOP_MAX_M` 1.0. Evidence: probe table, 9 cm closed 2/3 and 9.5 cm 0/3; both 1.12 m supply bins fell when
grasped 41 deg yawed. Limits: the test is on the box, not the mesh.

**P2. Stance, depth and clearances.** Trigger: `no clear stance with a path at an edge where a side fits the jaw`.
Rule: the object centre at most `pick.MAX_DEPTH_M` 0.35 in from the edge stood at; pelvis `STAND_AHEAD_M` 0.40 behind
the object and `STAND_SIDE_M` 0.12 to the side (retries 0.22, 0.30); the stance `STANCE_CLEAR_M` 0.12 from floor
objects and `OTHER_EDGE_CLEAR_M` 0.15 (0.25 until r2_on_single's write-back) from other furniture at an edge the scene file does not list (bar stools). The
skill's map inflates by a 0.35 m body radius. Evidence: at 0.26 m a route passed 0.33 m from the island hull, an arm
caught it and the robot fell. Limits: a route that exists only with a tighter radius is a gap, not out of range.

**P3. Floor objects the scene map lacks.** Trigger: a fall during the walk; `start against a floor object`; a wrist
target inside a bin. Rule: `floor_boxes` adds static objects whose bottom is below `FLOOR_BOTTOM_M` 0.30 (ADT's step
stool, trash bins, speaker, laundry basket, cat bowl); fallback plans need `SPAWN_CLEAR_M` 0.08 from the spawn.
Evidence: two pick_single falls. Limits: the start sampler does not avoid floor objects.

**P4. Via point over the slab lip.** Trigger: pre-grasp misses of 10–30 cm; fingertips hooking the slab lip from
below. Rule: approach through a via point `VIA_OUT_M` 0.08 out (`COVERED_VIA_OUT_M` 0.20 under a shelf) with the jaw
`VIA_JAW_ABOVE_TOP_M` 0.03 above the top (fingertips reach `FINGERTIP_M` 0.033 past the jaw centre). Limits: a farther
via everywhere broke pitched top-shelf grasps.

**P5. Low objects: pitch the wrist.** Trigger: a level side grasp would put the jaw lower than the gripper allows over
the slab — the hand stalls at the edge or rides on the top. Geometry: the gripper housing reaches 3.1–3.4 cm below the
jaw axis from 4.1–4.9 cm behind it, so a level jaw needs its centre ≥ 3.9 cm above the top (5.0 cm when the wrist's
low point is over the support, `WRIST_OVER_M` 0.02 in from the edge); pitched `pick.GRASP_PITCH_DEG` 15 nose-down it
needs 2.76 cm (`GRASP_CLEAR_M` 0.005 margin). Evidence: candles sit 2.5–2.6 cm above their top; apples 0.18–0.31 m
in: level missed 3/8, pitched grasped 8/8. Limits: the evidence is from held(X) picks.

**P6. Thin objects: the pinch.** Trigger: `needs_pinch` — the pads' lower edge would overlap the object by less than
`PINCH_OVERLAP_M` 0.005 (car key 0.1 cm; wallet 1.35 cm; candles 3.6–3.9 cm do not need it). Rule: `PINCH_PITCH_DEG`
25, `PINCH_CLEAR_M` 0.003, `PINCH_X_M` -0.015. Evidence: diagonal keys held 1/4 at a 0.005 clearance, 3/4 at 0.003.
Limits: smoke runs only.

**P7. Flat objects on low tops: top-down pinch.** Trigger: `is_top` — box half height under `TOP_FLAT_M` 0.02 on a
top below `TOP_SURFACE_MAX_M` 0.6 (the TV remote on the coffee table). Rule: `TOP_TIP_CLEAR_M` 0.004, descent from
`TOP_ABOVE_M` 0.10 at 0.05 m/s. Evidence: the remote 10/10 in pick_single's sealed pool. Limits: the 1.12 m shelf is
out (wallet, F2); the put-down's creep refusal does not apply to it.

**P8. One retry after a failed hand program.** Rule: `RETRY_ATTEMPTS` 1, only for an object still on its support,
under no surface and between no fingers; backs off `RETRY_BACK_M` 0.25. Evidence: a coffee-table candle knocked 5 cm
by a 3.2 cm pre-grasp lunge was finished by the retry; under shelves the retry knocked 3 of 5 plants off, so it is not
used there.

**P9. Entry: lag, pushes and timeouts.** Trigger: `pushed away on entry`, `entry timeout`, `no bilateral pad contact
after the close`. Facts: ScaleBFM lags the commanded wrist by about 10 cm at entry; `grasp_program.SIMPLE` uses
`settle_tolerance_m` 0.02, `entry_capture_depth_m` 0.025 (the jaw centre this close to the object centre),
`entry_timeout_s` 9.0, `disturb_limit_m` 0.05, `lift_seconds` 2.0; pad half height `PAD_Z` 0.014, half span `PAD_Y`
0.0475. A continuous close (`CONTINUOUS_CLOSE`) avoids the +2.2/+3.3 cm wrist jumps of target resets that cost the
top-shelf candle its pads. Evidence: pick_single round 1 had 6 pushed and 9 entry timeouts of 60.

**P10. A narrow book under a shelf: slide it out first.** Trigger: `narrow_covered` — under a surface and its only
fitting side under `NARROW_SIDE_MAX_M` 0.06. Rule: `NARROW` program — harder close (`jaw_closed` -0.06, about 20 N),
capture 0.05, pull 0.30 m before lifting. Evidence: a 4.7 cm book pivoted on its rear edge when closed 7–8 cm off its
centre at about 10 N; with the slide-out the held(X) books reached 38 sealed.

## The put-down's pick and the carry (`place.py`)

**C1. Creep budget.** Trigger (before contact): `pick refused: carry of N s exceeds the grip's creep margin (a cm <
b cm)`; (after the lift): `grip margin a cm < creep over N s` and a set-back. Rule: a side-grasped object slides down
the pads during the carry; the margin is its top above the grasp height less `PAD_Z` (1.4 cm); the pick is refused
when the margin is under `place.CREEP_RATE_M_S` × T_est, T_est the planned lift-to-release time
(`lift_to_release_s`: lift, hold, back-off, carry route with `LEG_SETTLE_S` 3.0 per leg and turns at
`CARRY_TURN_RATE` 0.4 rad/s, reach, descent). Current rate: 0.00062 m/s (0.0009 until r2_on_single: that rate had been measured on a 0.6 kg object, and r2_on's lighter objects slid slower). The top
pinch is exempt; `into.CREEP` False switches it off for put-ins. Evidence: carries slid 0.067–0.117 cm/s on successes
and 0.08–0.13 cm/s on dropped clocks; T_est overestimates the actual carry by 2.3–9.8 s; a clock was lost after
4.2 cm in 39 s.

**C2. Turning while loaded.** Rule: carry turns at `CARRY_TURN_RATE` 0.4 rad/s (the motor default is 0.8). Evidence: one
slip in a 120 deg final turn at 0.8 rad/s.

**C3. The put-down's pick is stricter than held(X)'s.** Rule: `PICK_DISTURB_LIMIT_M` 0.02 (held 0.05),
`PICK_SETTLE_ABORT_M` 0.035, `PICK_CAPTURE_DEPTH_M` 0.025 (0.015 until r2_on_single), `PICK_SIDE_AHEAD_M` 0.46. Evidence: successes
moved their object at most 0.7 cm and every object knocked more than 2 cm fell later; successes settled within 1.3 cm;
a 2.5 cm capture had left round-1 objects 1.9–3.0 cm out along the fingers, but the Dex1 housing's front edge is 4.07 cm behind the jaw centre, so on long objects (the canister, the wolf figurine) the housing pushed the object away before a 1.5 cm capture (r2_on_single iteration 0); the soap reach from 0.40 m missed by 13–22 cm.

**C4. Fingertip check.** Trigger: `held_out` step, a set-back after the lift. Rule: the object centre at most
`HELD_OUT_MAX_M` 0.03 out along the fingers. Evidence: successes sat 1.5–2.6 cm out; objects held farther out slid out
in the carry.

**C5. Blocking neighbour.** Trigger: `pick refused: a neighbour blocks the jaw (K overlaps it)`. Rule: refused when a
neighbour resting on the support (underside within `RESTING_BAND_M` 0.05 of the top) comes closer than
`PICK_MIN_NEIGHBOUR_GAP_M` 0.0. Evidence: every book start and one vase start overlapped a neighbour by 1.2–1.5 cm;
none was put down.

## Putting down on a surface (`place.py`)

**O1. The spot must sit inside the top.** Trigger: `no spot: ...`, an object falling off an end. Rule: `PLACE_SIDE_MIN_M`
0.06 from the sides, `SPOT_INSIDE_M` 0.015, `SPOT_EDGE_M` 0.03, `SPOT_CLEAR_M` 0.035 (0.04 until r2_on_single) from every object resting there.
Evidence: a vase fell off the end from 2.6 cm and stood from 6.1 cm; a remote fell 0.15 m from a walled end.

**O2. Level, then open, then retract.** Rule: level within `LEVEL_MAX_DEG` 25; the servo lead is dropped before the
hand opens (a lead up to 4.7 cm sprang the jaw up 1.5–4.3 cm on opening); `RETRACT_M` 0.12 straight back
(`RETRACT_RISE_M` 0.05). Evidence: objects reached the top tilted 1–16 deg without levelling; a rising retract dragged
leaning objects over.

**O3. One more lowering.** Rule: lowered to `PLACE_CLEAR_M` 0.015 above the top from `ABOVE_M` 0.12 (0.06 until r2_on_single); when the release
lands more than `LOWER_RETRY_OFF_M` 0.015 off, raise `LOWER_RETRY_UP_M` 0.03 and lower again. Evidence: releases
landed 2.1–2.6 cm from the target's end before the spot margins existed.

## Putting into a container (`into.py`)

**I1. Release fit.** Trigger: `no release over C: footprint a cm past the inner box less 1.0 cm` or `underside a cm
below the rim`. Rule: the hand opens only with the footprint `into.RELEASE_MARGIN_M` 0.01 inside C's inner box (half
extent less the wall) and the underside above the rim (`DROP_CLEAR_M` 0.03); one corrective move, then `_keep`: set
down on a surface below, else raised to `RAISE_BELOW_START_M` 0.05 below its start.

**I2. Where the drop stance is.** Rule: C's opening at most `into.MAX_DEPTH_M` 0.40 in from the edge, or up to
`DEEP_MAX_M` 0.7 with the pelvis `EDGE_OUT_M` 0.15 outside the edge; a drop at the pick's own edge within `NEAR_M`
1.6 of the pick stance is a near drop (one walk, no back-off), its lift at most `NEAR_LIFT_MAX_M` 0.30.

**I3. Near drops: timing and lift.** Evidence: an apple's near drop took about 21 s lift-to-release against a 22.75 s
creep budget (a back-off plan would have been 30.3 s); at a 0.10 m lift the hand swept a static cracker box (top
0.949 m), at 0.22 m it cleared (`NEAR_CLEAR_M` 0.03 over what rests along the path).

## Walking with a load (`near.py`)

**N1. Narrow passages and barred tops.** Rule: a narrow fallback at `NARROW_RADIUS_M` 0.28 where the obstacle tops are
below `NARROW_TOP_MAX_M` 0.46 (the 0.30 m corridor between the coffee table and the armchair); tops are barred
`BAR_OUTSIDE_M` 0.05 around and `BAR_BELOW_M` 0.05 below a carried object's underside. Fact: the predicates read an
object as resting on a top when its underside is within the top + 0.02 m, so a hand carrying it past a slab edge runs
into it.

## Controller and measured envelope (phase 1, capability table)

**E1. Hand reach.** Right hand, level, standing: forward at most 0.5 m at 0.25 m height, 0.7 m at 0.55–1.05 m, 0.5 m
at 1.25 m; wrist-down only at 0.15–0.55 m high and at most 0.40 m forward; 0.80–0.90 m forward unreachable. Counted
within 2.0 cm and 10 deg after a 1.5 s hold.

**E2. Mass and width.** Lifted up to 1.5 kg; 2 kg 0/3. A 9.5 cm wide side 0/3.

**E3. Torso and squat.** Mode 4 does not track a torso reference (10–50 deg asked gives 2–18 deg). Squat references
0.62–0.74 m hold; below 0.58 m it stops at 0.61 m.

**E4. Objects spin at rest.** 33 of 64 free objects turn more than 0.1 deg/s at rest (wallet 2.0, plant 4.3, soap
3.0 deg/s); torsional friction alone does not stop it.

## Diagnosis method

**D1. A one-round simulation is evidence.** The simulator is deterministic: the same start and skills give the same
round. Round 1's Prober got 16 of 17 predicted flips right (some starts the Reviewer had simulated itself).

**D2. A probe with 0 episodes is a driver problem first.** It was a bug once: inline overrides over 255 bytes were read
as file names.

**D3. Test an infeasibility with a closed-loop round before parking it.** Hard constraints alone would have parked
feasible starts: the orange behind the candle (the claim said a 1.75 cm lift was needed, over the 1.4 cm pad half
height; the closed loop succeeded at s100056) and the cup from above (a side approach succeeded at s100037, s100072).

**D4. A claim needs four parts** (rules.md); parking removes the repair goal only — parked starts stay in every
denominator, and parked points are re-tested after each adoption.

## Parked or physically impossible (with the evidence)

**F1. A book between books** (held(X), round 1): neighbours overlap it 1.2–1.46 cm, the pinch needs a 1.6 cm gap,
top-down needs 18.5 cm of headroom against 10.4 cm; 0 of 18 closed-loop attempts. Later solved by P10's slide-out: a
park is about the skills at the time.

**F2. The wallet on the 1.12 m top shelf:** a flat grasp needs the housing 3.9 cm into the slab; top-down leaves a
4.1 mm margin against 10 mm needed; 0 of 18.

**F3. A candle into the small dish** is physically impossible: lying flat on the dish's floor its centre is above the
rim, so in() can never hold (checker rule `under_rim`).

## Round 2 so far: `r2_on_single` (10-18, ten put-downs, 5 gates, 89.64 USD)

**R1. What fixed it.** Iteration 0: C3's housing push on long objects and C1's slower slide of light objects; the
diagnostic pool went 4 → 14/50 and the gate 19 vs 8. Iterations 2–4: candidates at 25 vs 19 were refused for 3 drops
(> 2) until iteration 4's 2 drops. Sealed: seed 0 → 25 of 72 (b 25, c 0); per task wolf figurine → coffee table 8/8,
big white vase → TV stand 6/8, coffee can → cart 6/8, canister → island 5/8, the other five 0/8.

**R2. Claims and parks.** Parked: the novel upright (its `upright_axis` is the thickness axis, so upright() wants it
lying flat — the task's meaning inverted, TODO 19.15); two novel put-downs (the only side that fits is the 4.7 cm
thickness, and the housing pushes the book); the plant under a shelf (4 starts); the coffee can (1 start); the plate
holder (2 starts). The Manager's `reopen` lifted the big vase's park (4/5 on those diagnostic starts), but that gate's
first stage split 10 vs 10. The Reviewer's claim that the high-shelf candle was infeasible (iteration 2) was
overturned by a closed-loop probe (D3 again). Proposed gaps: `reorient_onto_face`, `slide_on_surface`,
`two_hand_carry`, `set_down_and_regrasp`, `deep_low_reach`.

**R3. The scene changed under it.** The plate holder's targets (`ApartmentEnv__5`, `__4`) were cabinet tops 3.6 cm
under the long counter's slab: putting the holder anywhere over them made on() true, so the task held at 7/8 starts
from the start. Surfaces ≥ 90 % covered by one less than 10 cm above are removed (`surfaces.shadowed`).

**R4. Write-back 7 on the other families** (all sealed pools re-run): `r1_in_single` 9 → 13 (apple into the black
bowl 4 → 7/10), `r1_held_single` 34 → 36 (coffee can 6 → 8/8), `pick_single` 59 → 60, `r1_on_single` 25 → 23 (b 2,
c 4), the rest unchanged.
