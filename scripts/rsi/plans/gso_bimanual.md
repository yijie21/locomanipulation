# Two-hand lifts of scanned objects and the carton from the motor layer: the plan the roles read (logistics_room_1030, 2026-11-03)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). The robot, the controller and the loop are the same as in every scene RSI run; what differs is the scene (the
scanned logistics room with ten Google Scanned Objects drawn onto its pallet platform) and the objects: wide scanned
items that no single jaw holds, plus the open carton of the box_bimanual line, every one at a random spot on the
platform's edge band.

## 0. Goal

The robot walks to a pallet platform and lifts a wide object off it with both hands, high, level and steady. This run
is one task family, `gso_bimanual_s<seed>`, one task per two-hand object of the instance and one for the carton
(user, 2026-11-03), with the strict goal of box_bimanual (rules.md 36):
- `held(X)`: the fingers of both hands on X and X off every surface;
- `lifted(X, 0.20, 60)`: X's lowest point 20 cm or more above the platform top (0.371 m), its upright axis less than
  60 deg off vertical;
- both checked every 0.1 s and holding at every check over the round's last 5 s (`judge_s`): the end of the plan and
  then 2 s standing still (`hold_s`).

The objects come from `assets/mujoco_scanned_objects/catalog.json`: an object is `bimanual` when its wide side is
0.20-0.45 m, it stands 0.10 m or more and it stands at rest in a 24 s drop test (user Q17, 2026-11-03: the band was
widened from the jaw-based rule and these objects are untested; the first run tells which of them two Dex1 hands can
hold). The carton (`nor_cal_box`, 0.45 x 0.32 x 0.33 m, 1.2 kg, open top) is drawn into every instance. The robot
starts anywhere in the room it can walk from, facing a random way, 0.5 m or more from the object; the task object is
moved to a random spot and yaw within `reach_depth_m` (0.35 m) of a platform edge (the `placement` row option, D13),
and then every other object on the platform is moved to a random spot clear of the task objects and of the hands'
corridors (`shuffle_others`, 2026-11-06: with them fixed, 13 of s0's 21 tasks found no start).

- **Fixed inputs**: the scene file, the robot (mars G1 with Dex1 two-finger grippers), the controller ScaleBFM M 22200
  (not trained, not changed), MuJoCo, privileged poses from the simulator, the motor layer, `rules.md`.
- **The seed**: none (`sandbox.seed` null since 2026-11-05): the family runs on the main library, which holds the
  box line's two-hand hug pick ported from `seeds/box_strict_start` (`skills/bimanual.py`; `plan_for` sends `held(X)`
  of an object the scene file marks `bimanual` to it): stand square to the platform edge nearest the object, crouch,
  open jaws beside its two side walls, squeeze, lift by arms and legs. Measured once on the carton in the warehouse
  (lifts of 3-16 cm) and once here (gso_0945, both pregrasps 15 cm off: the pack line's smoke test, runbook 5a);
  it has passed no gate on a scanned object, so expect the seed to fail most starts.
- **Done by the loop**: generalising the hug from the carton to objects of other sizes and shapes, computed from
  `box.half_extent` and `semantic` (never from names, rules.md 30); the lift height, standing up, keeping the object
  level and steady through the window; accepting a change only through the gate.
- **Not done by people**: no parameters tuned, no skill code written beyond the seed. Every human intervention is
  recorded.

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

Pools (D7, sized as box_bimanual's): diagnostic 30 starts, acceptance 40, sealed 50 (drawn only after the loop
ends), rotated over the family's tasks. The roles see only the diagnostic pool.

## 3. What exists below the skills

| Layer | What it is | Who may change it |
|---|---|---|
| Controller | ScaleBFM M 22200, follows keypoint targets (pelvis, wrists, ankles) | nobody |
| Motor layer | `src/locomani/backends/scalebfm/motor.py`: walk_to, turn, sidestep, posture, reach (one hand or `"both"`), grip and release (one hand or `"both"`), squeeze, hold, plus per-tick hand commands and sensors | default-off switches only |
| Skills | `src/locomani/tasks/scene/skills/`: the seed's `bimanual.py` | the Engineer |
| Scene runner | `src/locomani/tasks/scene/` world, start, navigate (`walk_path`), episode, stage (`bimanual_grip_check`, `bimanual_lift_check`) | nobody |
| Goal predicates | `src/locomani/scene_discovery/predicates.py`: `held(X)` = both fingers of one hand, or the fingers of both hands, on X and X off every surface by more than 2 cm; `lifted(X, h, tilt)` | nobody |
| Scene | `outputs/scene_discovery/logistics_room_1030_s<seed>/` (scene_meta.json: the platform as surface `A` with its approach edges, every object with its `box` extents, `semantic` and a `gso` block, the room's bounds, obstacles and spawn region) | nobody inside the loop |

The scene (the logistics room of box_bimanual's plan, plus the objects):
- The room is about 5.7 x 4.4 m of walkable floor inside a scanned warehouse room 8.1 x 6.9 m; storage racks stand
  along three walls and a cart against the fourth. Racks and walls are visual meshes that do not collide; the
  occupancy map carries them as obstacles. The platform collides.
- The platform (surface `A`): 1.2 x 0.9 m, top 0.371 m, centre (0, -0.105); all four edges are approach edges, with
  1.05-1.63 m of clear floor in front of each. Its middle is out of reach from every edge (D13).
- The scanned objects: 0.05-1.5 kg (hull volume x 300 kg/m^3, clamped), friction 0.8, collision by the dataset's
  convex hulls (up to 32 per object), so a side wall may be curved or stepped where the carton's is flat.
  `scene_meta.json` gives each one `box.half_extent` and `box.centre` (the body origin is the mesh origin, not the
  bottom centre), `semantic.phrase`, `semantic.graspable` (`bimanual` here), `semantic.mass_kg`,
  `semantic.upright_axis`, and `gso.grasp_height_m`.
- The carton: 1 cm cardboard walls and floor, open top, friction 1.4, body origin at the bottom centre, `box.centre`
  [0, 0, 0.165]; `semantic.graspable` bimanual, `semantic.container` true.

The two-hand actions (rules.md 35): `reach("both", ...)` moves the two jaw centres together with the jaw-centre and
attitude servo on both; `grip("both")` / `release("both")` close and open the two jaws together; `squeeze(inward_m,
seconds)` pushes the two jaw centres towards each other and keeps pressing until `release` or a `reach` of that hand
(each servo's lead capped at 8 cm per axis); `posture()` with no arguments stands back up with the arms as they are.

## 4. Decisions

| # | Decision |
|---|---|
| D4 | The two-hand actions stay atomic: `grip` / `release` take `"both"`, `squeeze` presses the two jaw centres together and keeps pressing. Where to stand, where to put the hands, how hard to squeeze and how to lift are the skills' (the loop's). |
| D6 | The Engineer may add and change skills in the skills package (code, tests, parameters). The motor layer takes only default-off switches; the controller, simulator, scene, predicates, success rule and gate are frozen. |
| D7 | One family = one group of tasks; pools as in section 2, from the family's random task-object placements and robot spawns; disjoint seeds; the sealed pool is drawn after the loop. |
| D8 | The loop stops when the incumbent wins 70 % of the acceptance pool, or at a hard stop: 12 gate calls, the wall-clock cap, or the Manager's stop. |
| D11 | The only human knowledge is the versioned `rules.md` and this plan; interventions are recorded. |
| D12 | The success rule (user Q23, 2026-11-03): `held(X)` and `lifted(X, 0.20, 60)` at every 0.1 s check over the round's last 5 s. |
| D13 | Infeasibility is handled as in section 8. The task object always starts within 0.35 m of an approach edge (the `placement` option); a distractor may sit in the unreachable middle. |
| D14 | Graspability, the chooser, the body table and the preparatory layer describe one jaw; on this family they say nothing about the two-hand hold (rules.md 35). The failure classes are the two-hand set: walk_off, side_reach, one_hand_short, knocked, no_lift, low_lift, tilted, dropped_hold, slip, fell, other. |
| 2026-11-03 | User: the bimanual band is 0.20-0.45 m wide and 0.10 m or more high, untested on the scanned objects; the carton is a random object of the instance like the others and one task of this family; skills read extents and `semantic` fields, never names (rules.md 30). |

## 7. What the layers below deliver (measured)

Motor layer on the open floor (2026-10-16, one trial each): walk 2 m stops 2.0 cm from the goal, heading 0.7 deg off;
turn in place 180 / 90 deg 4.8 / 1.7 deg off; right jaw centre to 0.35-1.15 m height 0.75-1.0 cm off, attitude 0.4-4
deg; squat to 0.62 m gives 0.638 m with a 25 deg lean; standing up gives 0.786 m, lean 4-5 deg.

On the carton in this room (box_bimanual's probes, `scripts/rsi/probe_box.py`, 2026-10-23 and 24):

| Step | What worked | What did not |
|---|---|---|
| stance | `walk_path` to 0.38 m outside the platform edge nearest the object, facing it; stops within 1.5-2 cm | -- |
| hands | `posture(pelvis_z=0.72)`, then `reach("both")` with the wrists pointing forward and the jaws open: each hand's outer finger flank, 6.35 cm from the jaw axis, touches a wall; delivered 1-2 cm, attitude 2-6 deg | a pinch on a wall's top edge with the wrist pointing down (attitude 50-75 deg off); closed jaws touch the wall with the gripper base, which `held` does not count |
| squeeze | `squeeze(0.06, 1 s)`: both hands' fingers on the box | 0.04: the fingers slid up the walls |
| lift (hug) | `reach("both", +0.25 m, keep_squeeze=True, pelvis_z=0.79)` raises the box 15-29 cm | 0 / 10 on the strict goal: hands at 55 % of the height, the box tipped 30-59 deg; at 75-85 % it slid out of the flanks |
| wall pinch | one grasp point on each side wall under the rim, palm pitched 60 deg down, `grip("both")`, then `reach("both", +0.32 m)`: 3 / 10, all with the long side towards the edge | the jaws arrive 3-8 cm and 22-37 deg off; with the short side towards the edge 0 / 5 |

No probe has been run on the scanned two-hand objects yet (2026-11-03): their walls are the hulls of scanned shapes
(game boxes, cereal boxes, lunch bags, backpacks; no shoes or toys), and whether a squeeze holds them is what the first rounds measure. Objects at rest:
every instance passes `scripts/scene_lint.py` (0 errors) after `scripts/scene_settle.py --write`.

- Modes: walking is mode 2 (pelvis + wrists), standing still is mode 4 (pelvis + wrists + ankles pinned). Walks run at
  0.25 m/s; straight walks stop within 1.5-2 cm.
- The platform collides with the robot and the objects; the objects rest on it directly. Racks and walls do not
  collide.
- `walk_to` walks straight lines only; `navigate.walk_path` plans around the platform and the racks on the occupancy
  map (body radius 0.35 m) and cuts the path into straight legs.
- A round: reset the robot at the start's spawn, place the task object, settle 1 s, run `plan_for(task)(ctx)`, hold
  2 s, judge the goal on the privileged state over the last 5 s. A skill exception is the round's failure, recorded.
  `ctx.world` holds the scene file (`meta`), object poses, the free bodies, the occupancy map and navigator;
  `ctx.motor` the motor layer; `ctx.note(step, outcome, **numbers)` logs a step into the round record.
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
