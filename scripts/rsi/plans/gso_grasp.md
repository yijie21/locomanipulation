# One-hand picks of scanned household objects from the motor layer: the plan the roles read (logistics_room_1030, 2026-11-03)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). The robot, the controller and the loop are the same as in every scene RSI run; what differs is the scene (the
scanned logistics room with ten Google Scanned Objects drawn onto its pallet platform) and the objects: a different set
of household items in every scene instance, none of them known to the skills by name (rules.md 30).

## 0. Goal

The robot walks to a pallet platform and lifts one of the scanned objects off it with one hand. This run is one task
family, `gso_grasp_s<seed>`, one task per single-hand object of the instance (user Q23, 2026-11-03):
- `held(X)`: both fingers of one hand on X and X off every surface;
- `lifted(X, 0.10, 60)`: X's lowest point 10 cm or more above the platform top (0.371 m), its upright axis less than
  60 deg off vertical;
- both checked at the end of the plan and then 2 s standing still (`hold_s`).

The objects come from `assets/mujoco_scanned_objects/catalog.json`: an object is `single` when its narrow side fits the
open Dex1 jaw (0.095 m), it stands 0.03 m or more, it stands at rest in a 24 s drop test, and its grasp point (half its
height above the platform top) is 0.40 m or more above the floor. The instance's ten objects are drawn once with its
seed (`gso.seed` in scene_meta.json), the carton is always present; the objects that are not the task's are distractors
on the same platform. The robot starts anywhere in the room it can walk from, facing a random way, 0.5 m or more from
the object; the task object is moved to a random spot and yaw within `reach_depth_m` (0.35 m) of a platform edge (the
`placement` row option, D13), and then every other object on the platform is moved to a random spot clear of the task
objects (`shuffle_others`, 2026-11-06: with them fixed, 13 of s0's 21 tasks found no start).

- **Fixed inputs**: the scene file, the robot (mars G1 with Dex1 two-finger grippers), the controller ScaleBFM M 22200
  (not trained, not changed), MuJoCo, privileged poses from the simulator, the motor layer, `rules.md`.
- **The seed**: the skills package has no skill (`seeds/atomic`). `plan_for(task)` returns None for every goal, so the
  seed succeeds on no start. The first candidates may be a plain end-to-end pick built on the motor layer, improved
  one Engineer call at a time (rules.md 24).
- **Done by the loop**: diagnosing, writing and changing skills, choosing grasps from the object's extents and
  `semantic` fields (never from its name), accepting a change only through the gate.
- **Not done by people**: no parameters tuned, no skill code written. Every human intervention is recorded.

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

Pools (D7): 3 diagnostic, 10 acceptance and 10 sealed starts per task (18 / 60 / 60 on 6 tasks; user 2026-11-24, a fixed 30 / 40 / 50 before; the sealed pool is drawn only after the loop ends),
rotated over the family's tasks. The roles see only the diagnostic pool.

## 3. What exists below the skills

| Layer | What it is | Who may change it |
|---|---|---|
| Controller | ScaleBFM M 22200, follows keypoint targets (pelvis, wrists, ankles) | nobody |
| Motor layer | `src/locomani/backends/scalebfm/motor.py`: walk_to, turn, sidestep, posture, reach, grip, release, squeeze, hold, plus per-tick hand commands and sensors | default-off switches only |
| Skills | `src/locomani/tasks/scene/skills/`: empty at the start | the Engineer |
| Scene runner | `src/locomani/tasks/scene/` world, start, navigate (`walk_path`), episode, stage | nobody |
| Goal predicates | `src/locomani/scene_discovery/predicates.py`: `held(X)`, `lifted(X, h, tilt)` | nobody |
| Scene | `outputs/scene_discovery/logistics_room_1030_s<seed>/` (scene_meta.json: the platform as surface `A` with its approach edges, every object with its `box` extents, `semantic` and a `gso` block, the room's bounds, obstacles and spawn region) | nobody inside the loop |

The scene (the logistics room of box_bimanual's plan, plus the objects):
- The room is about 5.7 x 4.4 m of walkable floor inside a scanned warehouse room 8.1 x 6.9 m; storage racks stand
  along three walls and a cart against the fourth. Racks and walls are visual meshes that do not collide; the
  occupancy map carries them as obstacles. The platform collides.
- The platform (surface `A`): 1.2 x 0.9 m, top 0.371 m, centre (0, -0.105); all four edges are approach edges, with
  1.05-1.63 m of clear floor in front of each. Its middle is out of reach from every edge (D13).
- The objects: scanned household items (boxes of games and food, bottles, bags, electronics, household goods; no shoes, toys, Legos or hats) with their scan
  texture, 0.05-0.3 kg (hull volume x 300 kg/m^3, clamped; the cap was 1.5 until 2026-11-07), friction 0.8, collision by the dataset's convex hulls (up
  to 32 per object). `scene_meta.json` gives each one `box.half_extent` and `box.centre` (the body origin is the mesh
  origin, not the bottom centre), `semantic.phrase`, `semantic.graspable` (`single` here), `semantic.mass_kg`,
  `semantic.upright_axis`, and `gso.grasp_height_m` (the catalog's predicted grasp point above the floor).
- The carton (`nor_cal_box`, 0.45 x 0.32 x 0.33 m, 0.3 kg since 2026-11-07 (was 1.2)) sits on the same platform as a distractor of this family.
- The default start (`robot_start`) stands 1.1 m from the platform's -y edge facing it; the pools spawn anywhere.

Measured ScaleBFM facts (controller comparison, 2026-10-13): with the right mode, keyframe clips that reach the jaw
down to 0.16 m, step back, sidestep, turn in place, carry with both arms and hand over between hands all succeed
10/10; open loop the jaw lands 2-9 cm off, so the jaw centre and wrist attitude need a servo (the motor layer's reach
has one). Not delivered: hands at 1.39 m, closing the jaw while walking, a 65 deg forward bend.

## 4. Decisions

| # | Decision |
|---|---|
| D5 | The motor layer's actions: `walk_to` (pose), `turn`, `sidestep`, `posture` (pelvis height, torso lean), `reach` (one or both jaw centres to a pose, jaw-centre and attitude servo), `grip`, `release`, `squeeze`, `hold`; each in its best measured mode. |
| D6 | The Engineer may add and change skills in the skills package (code, tests, parameters). The motor layer takes only default-off switches; the controller, simulator, scene, predicates, success rule and gate are frozen. |
| D7 | One family = one group of tasks; pools as in section 2, from the family's random task-object placements and robot spawns; disjoint seeds; the sealed pool is drawn after the loop. |
| D8 | The loop stops at the first of: the incumbent wins 95 % of the acceptance pool, the run has been going 15 h, it has taken 5 iterations (the family file's `stop_rate`, `max_wall_h`, `max_run_iters`; user 2026-11-24, 80 % and 5 h before), 12 gate calls, or the Manager's stop. |
| D11 | The only human knowledge is the versioned `rules.md` and this plan; interventions are recorded. |
| D12 | The success rule (user Q23, 2026-11-03): `held(X)` and `lifted(X, 0.10, 60)` at the end of the round. |
| D13 | Infeasibility is handled as in section 8. The task object always starts within 0.35 m of an approach edge (the `placement` option); a distractor may sit in the unreachable middle. |
| 2026-11-03 | User: a scene instance draws its objects once (`--gso-seed`); the loop never sees a different set in the same run. Skills read extents and `semantic` fields, never names (rules.md 30): a skill that works on seed 0's six objects and on seed 1's is the aim. Held-out objects (10 % of the catalog, by name hash) never appear in a `train` instance; `--pool eval_unseen` instances exist for evaluation only. |

## 7. What the layers below deliver (measured)

Motor layer (2026-10-16, open floor, one trial each; grasp_atomic's plan):

| Action | Result |
|---|---|
| walk 2 m | stops 2.0 cm from the goal, heading 0.7 deg off |
| turn in place 180 / 90 deg | 4.8 / 1.7 deg off |
| sidestep 0.3 m / step back 0.3 m | both arrive (sidestep 2 cm short) |
| right jaw centre to 0.35 / 0.55 / 0.90 / 1.15 m height | 0.75 / 0.99 / 0.85 / 1.0 cm off, attitude 0.4-4 deg |
| squat to 0.62 m | 0.638 m; delivered lean 25 deg (mode 4 does not track the torso) |
| stand up (back to the reset stance) | 0.786 m, lean 4-5 deg |

On this scene (box_bimanual's probes, 2026-10-23 and 24): `walk_path` to 0.38 m outside the platform edge nearest the
object, facing it, stops within 1.5-2 cm; `reach` delivers the jaw 1-2 cm and 2-6 deg off at the platform's height.
No probe has been run on the scanned objects themselves yet (2026-11-03): the first Reviewer reads the seed's empty
rounds and the scene file. Objects at rest: every instance passes `scripts/scene_lint.py` (0 errors) after
`scripts/scene_settle.py --write`; objects that tip in the catalog's drop test are never drawn.

- Modes: walking is mode 2 (pelvis + wrists; the policy places its own steps); standing still is mode 4 (pelvis +
  wrists + ankles pinned). Walks run at 0.25 m/s; straight walks stop within 1.5-2 cm.
- The platform collides with the robot and the objects; the objects rest on it directly. Racks and walls do not
  collide.
- `walk_to` walks straight lines only; `navigate.walk_path` plans around the platform and the racks on the occupancy
  map (body radius 0.35 m) and cuts the path into straight legs.
- A round: reset the robot at the start's spawn, place the task object, settle 1 s, run `plan_for(task)(ctx)`, hold
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
| D13.4 | A park only removes those starts as repair targets; they still count as failures in every pool. The 95 % stop also counts on the unparked starts when they are at least half the acceptance pool. |
| D13.5 | Any incumbent that finishes a parked start lifts the park, and the claim is recorded as wrong. |
| D13.6 | A new skill that fits one Engineer call stays in the loop (for a park's exit: the Manager's `reopen`); one that does not becomes a gap proposal. |
