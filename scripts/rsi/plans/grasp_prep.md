# Preparing a grasp: the plan the roles read (TODO 33.5, 2026-10-23)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). The robot, the controller, the scene and the loop are the same as in every scene RSI run; what differs is the
seed (the grasp_atomic line's incumbent) and the tasks (objects placed where no stance serves them).

## 0. Goal

The seed already picks the ten living-room objects up where they usually stand (run C's incumbent: sealed 38/50). This
family, `grasp_prep`, puts three of them where the chooser (`locomani.tasks.scene.choose`) finds no stance from which
the controller delivers the hand, and asks the loop to write the preparatory actions a person would use first: the
dish soap bottle deep on the counter with its narrow side lying along the edge (turn it so the narrow side faces you;
the seed's raised grasp still serves about a third of these starts), the small vase and the tall plant on the display unit's shelves with a
neighbour standing between them and the shelf's only open side (move the neighbour aside). No surface in this room is deep
enough behind a blocked side to need a pull toward the edge, so that kind stays in the interface for other rooms. The `placement` option of each task draws exactly those situations.

- **Fixed inputs**: the scene file, the robot (mars G1 with Dex1 two-finger grippers), the controller ScaleBFM M 22200
  (not trained, not changed), MuJoCo, privileged poses from the simulator, the motor layer, `rules.md`, and three
  pieces of infrastructure the loop calls but may not change: the body model `locomani.tasks.scene.body` (the odds the
  controller lands a hand 3 cm or more off a target), the chooser `locomani.tasks.scene.choose` (grasp, hand, stance
  and posture together, with the fallback order standing / lowered pelvis / preparatory action) and the preparatory-
  action layer `locomani.tasks.scene.prep` (the goals turn / pull / clear, the selector, the registry an action is
  put in with `prep.register`).
- **The seed**: `scripts/rsi/seeds/grasp_c11`, the grasp_atomic incumbent after run C (pick, rim, clutter, doorway,
  recipes/tall_narrow). It has no preparatory action: `prep.implemented()` is empty, so on these starts the seed
  mostly fails.
- **Done by the loop**: calling the chooser, writing the actions (the motions that turn, pull and clear through the
  motor layer, each registered under its kind and ending with a re-choice on the measured state), and whatever the
  pick needs afterwards; accepting a change only through the gate.
- **Not done by people**: no parameters tuned, no skill code written. Every human intervention is recorded.

## 2. Structure

One inner loop on one family (no outer loop in this run). One iteration:

1. **Reviewer** (Claude) reads the incumbent's rounds on the diagnostic pool (records and 50 Hz traces) and writes at
   most 4 hypotheses: the cause, a parameter test, predicted flips. It may also claim that some starts cannot be done
   with the present actions (section 8).
2. **Prober** (program) runs each hypothesis' parameter test closed loop on the diagnostic pool.
3. **Manager** (Claude) reads the ledger and decides: `engineer` (which hypothesis to build), `reopen`, `review`, or
   `stop`.
4. **Engineer** (Claude) writes code in the skills package and/or parameter overrides: one candidate.
5. **Gate** (program) compares candidate and incumbent start by start on the hidden acceptance pool.

Pools (D7, sized for this run): diagnostic 30 starts (3 per object), acceptance 40 (4 per object), sealed 50 (5 per
object, drawn only after the loop ends). The roles see only the diagnostic pool.

## 3. What exists below the skills

| Layer | What it is | Who may change it |
|---|---|---|
| Controller | ScaleBFM M 22200, follows keypoint targets (pelvis, wrists, ankles) | nobody |
| Motor layer | `src/locomani/backends/scalebfm/motor.py`: walk_to, turn, sidestep, posture, reach, grip, release, hold, plus per-tick hand commands and sensors | default-off switches only |
| Skills | `src/locomani/tasks/scene/skills/`: the grasp_atomic incumbent after run C (2026-10-23) | the Engineer |
| Body model, chooser, prep layer | `src/locomani/tasks/scene/body.py`, `choose.py`, `prep.py`: measured reach odds, the stance / grasp choice, the preparatory-action goals and selector | nobody; the actions themselves: the Engineer |
| Scene runner | `src/locomani/tasks/scene/` world, start, navigate (`walk_path`), episode | nobody |
| Goal predicates | `src/locomani/scene_discovery/predicates.py`: `held(X)` = both fingers of one hand (or both hands) on X and X off every surface | nobody |
| Scene | `outputs/scene_discovery/adt_apartment_mars/` (scene_meta.json: objects with box, mesh, surface; surfaces with height, extent, approachable edges) | nobody inside the loop |

Measured ScaleBFM facts (controller comparison, 2026-10-13):
- With the right mode, keyframe clips that reach the jaw down to 0.16 m, step back, sidestep, turn in place, carry with
  both arms and hand over between hands all succeed 10/10.
- Not delivered: hands at 1.39 m (attitude 14-30 deg off), closing the jaw while walking, a 65 deg forward bend
  (attitude 31-46 deg off).
- Standing in mode 2 the feet slide apart; open loop the jaw lands 2-9 cm off, so the jaw centre and wrist attitude need
  a servo (the motor layer's reach has one).
- Mode switches that add keypoints are smooth; removing shoulder/elbow keypoints drops the wrist 21-28 cm.

## 4. Decisions

| # | Decision |
|---|---|
| D5 | The motor layer has 8 actions: `walk_to` (pose), `turn`, `sidestep`, `posture` (pelvis height, torso lean), `reach` (one or both jaw centres to a pose, jaw-centre and attitude servo), `grip`, `release`, `hold`; each in its best measured mode. |
| D6 | The Engineer may add and change skills in the skills package (code, tests, parameters). The motor layer takes only default-off switches; the controller, simulator, scene, predicates, success rule and gate are frozen. |
| D7 | One family = one group of tasks; pools as in section 2, from the family's random object placements and robot spawns; disjoint seeds; the sealed pool is drawn after the loop. |
| D8 | The loop stops when the incumbent wins 70 % of the acceptance pool, or at a hard stop: 12 gate calls, the wall-clock cap (5 h in this run), 3 rejections in a row with no new hypothesis, or the Manager's stop. |
| D11 | The only human knowledge is the versioned `rules.md`; interventions are recorded in `interventions.jsonl`. |
| D13 | Infeasibility is handled as in section 8. |
| 2026-10-23 | User for this run (TODO 33.5, D2, D5, D9): the preparatory actions follow the user's order -- turn the object so the narrow side faces you, pull it to the edge, move the neighbour -- and are tried only when the chooser says `needs_prep`; the interface and selector are infrastructure, the actions are the loop's; the tasks' `placement` options draw the situations. |
| 2026-10-20 | User for the seed's run: the seed is the motor layer only; grasps are computed from the scene's geometry by code the loop writes (no learned grasp sampler); earlier skill catalogs, earlier loop records and earlier grasp programs are not in the sandbox. |

## 7. What the layers below deliver (measured)

Body model (envelope grid of 15,120 reaches, A800 2026-10-21; `body.reach`): at table height a wrist pointing
straight down is not delivered 0.3-0.6 m ahead (83-100 % of cells miss by 3 cm or more) while a 45-degree tilt
and a level wrist are (0 %, 0.3-0.7 m); reaches 0.2 m ahead miss 42-88 %, 0.8 m and beyond 83-95 %; a low shelf
(0.40-0.55 m) is fine level from 0.4-0.7 m. On the grasp_atomic line's pre-grasps the model's AUC is 0.78
against 0.60 for the IK residual. It has no furniture in it: a tight stance (under 0.20 m from the table) and a
neighbour in the corridor are the chooser's checks, not the model's.


Motor layer (2026-10-16, open floor, one trial each):

| Action | Result |
|---|---|
| walk 2 m | stops 2.0 cm from the goal, heading 0.7 deg off |
| diagonal 1.8 m with a 90 deg turn | 1.5 cm, 0.8 deg |
| turn in place 180 / 90 deg | 4.8 / 1.7 deg off |
| sidestep 0.3 m / step back 0.3 m | both arrive (sidestep 2 cm short) |
| right jaw centre to 0.35 / 0.55 / 0.90 / 1.15 m height | 0.75 / 0.99 / 0.85 / 1.0 cm off, attitude 0.4-4 deg |
| squat to 0.62 m | 0.638 m; upright reference torso, delivered lean 25 deg (mode 4 does not track the torso) |
| lean 20 / 34 deg (reference) | 4-8 / 16 deg delivered: in mode 4 the controller chooses the waist |
| stand up (back to the reset stance) | 0.786 m, lean 4-5 deg |

- Modes: walking is mode 2 (pelvis + wrists; the policy places its own steps); standing still is mode 4 (pelvis +
  wrists + ankles pinned). Standing 15 s in the apartment, mode 2 widened the stance 27.7 -> 36.1 cm, mode 4
  26.5 -> 28.0 cm. Walks run at 0.25 m/s, where mode 2 shuffles (15 cm steps, 2 cm foot lift); straight walks stop within 1.5-2 cm.
- Furniture collides with the robot only below its lowest surface (a convex hull cut 2 cm under it); above that, hands
  and arms pass through cabinet sides. Free objects rest on 2 cm surface slabs. Walls and the upper parts of tall
  furniture carry extra collision boxes against the robot's arms and head.
- `walk_to` walks straight lines only; `navigate.walk_path` plans around furniture on the occupancy map (body radius
  0.35 m) and cuts the path into straight legs; a stance against a table is reached by a last straight leg.
- A round: reset the robot at the start's spawn, place the task objects, settle 1 s, run `plan_for(task)(ctx)`, hold
  2 s, judge the goal on the privileged state. A skill exception is the round's failure, recorded. `ctx.world` holds
  the scene file (`meta`), object poses, the free bodies, the occupancy map and navigator; `ctx.motor` the motor layer;
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
