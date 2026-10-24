# A two-hand box pickup from the motor layer: the plan the roles read (TODO 38, 2026-10-23; strict goal 2026-10-24)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). The robot, the controller and the loop are the same as in every scene RSI run; what differs is the scene (a
scanned logistics room instead of the ADT apartment), the object (one open cardboard box wider than a jaw, so it takes
both hands) and the goal: the box lifted high and held.

## 0. Goal

The robot walks to a pallet platform and lifts a cardboard box off it with both hands, high, level and steady. This
run is one task family, `box_bimanual`, with the strict goal of 2026-10-24 (rules.md 36):
- `held(nor_cal_box)`: the fingers of both hands on the box and the box off every surface;
- `lifted(nor_cal_box, 0.20, 30)`: the box's lowest corner 20 cm or more above the platform top (0.371 m), its bottom
  less than 30 deg off level;
- both checked every 0.1 s and holding at every check over the round's last 5 s (`judge_s`): the end of the plan and
  then 2 s standing still (`hold_s`). The skill may still stand up or lift during the window; the box may not slip,
  tip or drop in it.

Two tasks differ only in how the box sits on the platform (`box_long_side`: its 0.45 m side faces the nearest
platform edge; `box_short_side`: its 0.32 m side does), its near face within 0.15 m of that edge. The robot starts
anywhere in the room it can walk from, facing a random way, 0.5 m or more from the box.

- **Fixed inputs**: the scene file, the robot (mars G1 with Dex1 two-finger grippers), the controller ScaleBFM M 22200
  (not trained, not changed), MuJoCo, privileged poses from the simulator, the motor layer, `rules.md`.
- **The seed**: not empty. It is a two-hand hug pick in `bimanual.py` (stand square to the platform edge nearest the
  box, crouch, open jaws beside the two side walls, squeeze, lift by arms and legs). It lifts the box only 3-16 cm, so
  under this goal it fails on every start (low_lift, a few tilted).
- **Done by the loop**: diagnosing and changing the pick -- the lift height, standing up, keeping the box level and
  steady through the window -- accepting a change only through the gate.
- **Not done by people**: no parameters tuned, no skill code written beyond that dispatch line. Every human
  intervention is recorded.

## 2. Structure

One inner loop on one family (no outer loop in this run). One iteration:

1. **Reviewer** (Claude) reads the incumbent's rounds on the diagnostic pool (records and 50 Hz traces) and writes at
   most 4 hypotheses: the cause, a parameter test, predicted flips. It may also claim that some starts cannot be done
   with the present actions (section 8).
2. **Prober** (program) runs each hypothesis' parameter test closed loop on the diagnostic pool.
3. **Manager** (Claude) reads the ledger and decides: `engineer` (which hypotheses to build), `reopen`, `review`, or
   `stop`.
4. **Engineers** (Claude, three in parallel) write code in the skills package and/or parameter overrides, one draft
   each; the **Composer** (Claude) makes one candidate of them.
5. **Gate** (program) compares candidate and incumbent start by start on the hidden acceptance pool.

Pools (D7, sized as grasp_atomic's): diagnostic 30 starts (15 per task), acceptance 40 (20 per task), sealed 50 (25
per task, drawn only after the loop ends). The roles see only the diagnostic pool.

## 3. What exists below the skills

| Layer | What it is | Who may change it |
|---|---|---|
| Controller | ScaleBFM M 22200, follows keypoint targets (pelvis, wrists, ankles) | nobody |
| Motor layer | `src/locomani/backends/scalebfm/motor.py`: walk_to, turn, sidestep, posture, reach (one hand or `"both"`), grip and release (one hand or `"both"`), squeeze, hold, plus per-tick hand commands and sensors | default-off switches only |
| Skills | `src/locomani/tasks/scene/skills/`: empty at the start | the Engineer |
| Scene runner | `src/locomani/tasks/scene/` world, start, navigate (`walk_path`), episode, stage (`bimanual_grip_check`, `bimanual_lift_check`) | nobody |
| Goal predicates | `src/locomani/scene_discovery/predicates.py`: `held(X)` = both fingers of one hand, or the fingers of both hands, on X and X off every surface by more than 2 cm | nobody |
| Scene | `outputs/scene_discovery/logistics_room_mars/` (scene_meta.json: the platform as surface `A` with its approach edges, the box with its `box` extents and `semantic`, the room's bounds, obstacles and spawn region) | nobody inside the loop |

The scene (measured from the pack, assets/scene/logistics_room_box_v1):
- The room is about 5.7 x 4.4 m of walkable floor inside a scanned warehouse room 8.1 x 6.9 m; storage racks stand
  along three walls and a cart against the fourth. Racks and walls are visual meshes that do not collide; the
  occupancy map carries them as obstacles, so `navigate.walk_path` keeps the body 0.35 m from them and the start
  sampler spawns 0.45 m or more from them. The platform collides.
- The platform (surface `A`): 1.2 x 0.9 m, top 0.371 m, centre (0, -0.105); all four edges are approach edges, with
  1.05-1.63 m of clear floor in front of each.
- The box (`nor_cal_box`): outer 0.45 x 0.32 x 0.33 m, 1 cm cardboard walls and floor, open top, 1.2 kg, friction 1.4;
  its body origin is at the bottom centre, `box.centre` [0, 0, 0.165]. The jaw opens 9.5 cm: the box does not fit a
  jaw on any side (`semantic.graspable` bimanual), so one hand cannot hold it; two hands pressing opposite walls can.
- The default start (`robot_start`) stands 1.1 m from the platform's -y edge facing it; the pools spawn anywhere.

The two-hand actions (rules.md 35, TODO 38.4):
- `reach("both", (left_xyz, right_xyz), (left_R, right_R))` moves the two jaw centres together with the jaw-centre and
  attitude servo on both.
- `grip("both")` / `release("both")` close and open the two jaws together (a closed jaw presents the finger pads flat).
- `squeeze(inward_m, seconds)` pushes the two jaw centres towards each other along the line between them and keeps
  pushing: the offsets ride on each hand's servo reference in the pelvis frame through `posture`, `hold` and `walk_to`
  until `release` or a `reach` of that hand; blocked by the box, each servo's lead winds up to its cap (8 cm per axis)
  and the controller keeps pressing. Its outcome says whether the fingers of both hands touch one common body.
- `posture()` with no arguments stands back up with the arms as they are: a lift, when the hands hold the box.

Measured ScaleBFM facts (controller comparison, 2026-10-13; the box line, TODO 25.3):
- With the right mode, keyframe clips that carry with both arms and hand over between hands succeed 10/10.
- Replaying the SONIC box expert's trajectories on ScaleBFM, the box came up only with the jaw-centre and attitude
  servo on (open loop 3/8; the hands landed 2-9 cm and 10-30 deg off). The motor layer's reach and squeeze carry that
  servo.
- Standing in mode 2 the feet slide apart; mode 4 pins them. Mode switches that add keypoints are smooth.

## 4. Decisions

| # | Decision |
|---|---|
| D4 | The ninth motor action stays atomic: `grip` / `release` take `"both"`, `squeeze` presses the two jaw centres together and keeps pressing. Where to stand, where to put the hands, how hard to squeeze and how to lift are the skills' (the loop's). |
| D6 | The Engineer may add and change skills in the skills package (code, tests, parameters). The motor layer takes only default-off switches; the controller, simulator, scene, predicates, success rule and gate are frozen. |
| D7 | One family = one group of tasks; pools as in section 2, from the family's random box placements and robot spawns; disjoint seeds; the sealed pool is drawn after the loop. |
| D8 | The loop stops when the incumbent wins 70 % of the acceptance pool, or at a hard stop: 12 gate calls, the wall-clock cap (6 h), or the Manager's stop. |
| D11 | The only human knowledge is the versioned `rules.md` and this plan; interventions are recorded. |
| D12 | The success rule (2026-10-24, user): `held(nor_cal_box)` and `lifted(nor_cal_box, 0.20, 30)` at every 0.1 s check over the round's last 5 s. |
| D13 | Infeasibility is handled as in section 8. |
| D14 | Graspability, the chooser, the body table and the preparatory layer describe one jaw; on this family they say nothing about the two-hand hold (rules.md 35). The failure classes are the two-hand set: walk_off, side_reach, one_hand_short, knocked, no_lift, low_lift, tilted, dropped_hold, slip, fell, other. |
| 2026-10-23 | User for the first run: the seed is the motor layer only; grasps are computed from the scene's geometry by code the loop writes; earlier skill catalogs, loop records and grasp programs are not in the sandbox; the experience library holds the single-hand lines' lessons, not their code. |

## 7. What the layers below deliver (measured)

Motor layer on the open floor (2026-10-16, one trial each; the walk, turn, sidestep, reach, squat and lean rows of
grasp_atomic's plan hold here unchanged):

| Action | Result |
|---|---|
| walk 2 m | stops 2.0 cm from the goal, heading 0.7 deg off |
| turn in place 180 / 90 deg | 4.8 / 1.7 deg off |
| right jaw centre to 0.35 / 0.55 / 0.90 / 1.15 m height | 0.75 / 0.99 / 0.85 / 1.0 cm off, attitude 0.4-4 deg |
| squat to 0.62 m | 0.638 m; delivered lean 25 deg (mode 4 does not track the torso) |
| stand up (back to the reset stance) | 0.786 m, lean 4-5 deg |

What the layers below deliver on this scene (hand-written probes, `scripts/rsi/probe_box.py`, 2026-10-23 and 24;
the default start, 10 random box placements, 5 per task; sequences the loop does not see):

| Step | What worked | What did not |
|---|---|---|
| stance | `walk_path` to 0.38 m outside the platform edge nearest the box, facing it; stops within 1.5-2 cm | -- |
| hands | `posture(pelvis_z=0.72)`, then `reach("both")` with the wrists pointing forward (x = the approach, z up) and the jaws open: each hand's outer finger flank, 6.35 cm from the jaw axis, is what touches a wall; delivered 1-2 cm, attitude 2-6 deg | a pinch on each wall's top edge with the wrist pointing down: the attitude came out 50-75 deg off and the hands knocked the box; closed jaws touch the wall with the gripper base, which `held` does not count |
| squeeze | `squeeze(0.06, 1 s)`: both hands' fingers on the box | 0.04: the fingers slid up the walls |
| wall pinch (the human-defined grasp of this box, the crate line's expert) | one grasp point on each side wall, 8 mm inside its outer face and 1 cm under the rim, the open jaw straddling the 1 cm wall; the palm pitched 60 deg down; pre-grasp 9 cm above and 4 cm back, then straight down; `grip("both")`; `reach("both", +0.32 m, pelvis_z=0.79)`: 3 / 10 on this goal, all three with the box's long side towards the edge (3 / 5); the rim pivot sits above the centre of mass, so a box held by both pinches hangs level | the jaws arrive 3-8 cm off and the attitude 22-37 deg off, so often only one jaw closes on its wall and the box slides or is knocked; with the short side towards the edge the walls to pinch are 25-37.5 cm deep and 0 / 5 succeeded; a 45 deg pitch arrived closer (12-28 deg) but four boxes fell inside the 5 s window |
| lift to the goal (hug) | `reach("both", +0.25 m, keep_squeeze=True, pelvis_z=0.79)` raises the box 15-29 cm: the height is deliverable | 0 / 10 on this goal in five settings: hands at 55 % of the box height -- the line between them passes almost through the box's centre of mass -- the box tipped 30-59 deg; at 75-85 % it slid down out of the finger flanks; an 8 cm squeeze at 75 % still tipped. `posture()` (stand up) keeps the arms in joint space and swings them over the head |

While a squeeze is on, each hand's servo lead is capped at the offset along the push and at the offset's size across it
(motor.py `_servo`): the command sits at most 2 x inward_m inside the wall. The gripper base (2.9 cm half width) is wider
than the closed fingers (1.9 cm), so a hug with closed jaws touches the box with the base, which `held()` does not count.

- Modes: walking is mode 2 (pelvis + wrists; the policy places its own steps); standing still is mode 4 (pelvis +
  wrists + ankles pinned). Walks run at 0.25 m/s, where mode 2 shuffles; straight walks stop within 1.5-2 cm (0.5 m/s, tried 2026-10-24, changed nothing in the probe).
- The platform collides with the robot and the box; the box rests on it directly (no slab). Racks and walls do not
  collide: a hand through a rack is a navigation error the map should have prevented, not a physics event.
- `walk_to` walks straight lines only; `navigate.walk_path` plans around the platform and the racks on the occupancy
  map (body radius 0.35 m) and cuts the path into straight legs; a stance at the platform is reached by a last
  straight leg.
- A round: reset the robot at the start's spawn, place the box, settle 1 s, run `plan_for(task)(ctx)`, hold 2 s, judge
  the goal on the privileged state. A skill exception is the round's failure, recorded. `ctx.world` holds the scene
  file (`meta`), object poses, the free bodies, the occupancy map and navigator; `ctx.motor` the motor layer;
  `ctx.note(step, outcome, **numbers)` logs a step into the round record.
- A candidate is the skills package snapshot plus parameter overrides (`candidate.json` {"set": ...}, keys as
  `skills.apply_set` takes them). The gate copies the candidate and runs it; the incumbent runs from its own snapshot;
  a 5-round sentinel of the incumbent must reproduce its stored outcomes.

## 8. Starts that cannot be done (D13)

| # | Rule |
|---|---|
| D13.1 | Three kinds: physically impossible (out of reach, over the payload, inside something that cannot open) -- only the user confirms this; cannot with the present actions -- a skill gap with an exit; not done yet -- the loop keeps working. |
| D13.2 | A claim names the actions it assumed, a hard constraint code can recompute (a check file), at least one exit ("if the robot could first ..., it could") or why no action could, and what would refute it. Code refuses a claim missing any part. |
| D13.3 | Two pieces of evidence, run by the driver: the check holds on every claimed start, and the strongest present attempts (1-3, closed loop) all fail. A narrative alone does not park anything. |
| D13.4 | A park only removes those starts as repair targets; they still count as failures in every pool. The 70 % stop also counts on the unparked starts when they are at least half the acceptance pool. |
| D13.5 | Any incumbent that finishes a parked start lifts the park, and the claim is recorded as wrong. |
| D13.6 | A new skill that fits one Engineer call stays in the loop (for a park's exit: the Manager's `reopen`); one that does not becomes a gap proposal. |
