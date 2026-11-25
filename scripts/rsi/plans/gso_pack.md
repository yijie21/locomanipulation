# Packing scanned objects into the carton, then lifting it: the plan the roles read (logistics_room_1030, 2026-11-05)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). The robot, the controller, the loop and the room are those of `gso_bimanual.md` (the scanned logistics room
with ten Google Scanned Objects and the open carton drawn onto its pallet platform); what differs is the goal: one
object goes into the carton, then the carton is lifted with both hands.

## 0. Goal

The robot walks to the pallet platform, picks one scanned object X, drops it into the open carton standing on the same
platform, then hugs the carton with both hands and lifts it, X inside. Two task families per instance (user 2026-11-05):
`gso_pack_s<seed>`, one task per single-hand object (6), and `gso_pack_bimanual_s<seed>`, one per two-hand object (4),
each with the goal
- `in(X, nor_cal_box)`: X's box centre inside the carton's inner box (the carton's half extents less its 1 cm wall,
  from the carton's measured pose; the scanned object's box centre, not its mesh origin, 2026-11-05);
- `held(nor_cal_box)`: the fingers of both hands on the carton and the carton off every surface;
- `lifted(nor_cal_box, 0.20, 60)`: the carton's lowest point 20 cm or more above the platform top (0.371 m), its
  upright axis less than 60 deg off vertical (no `upright` term: the box line's 15 deg was not asked, user Q11);
- all three checked every 0.1 s and holding at every check over the round's last 5 s (`judge_s`): the end of the plan
  and then 2 s standing still (`hold_s`).

Every object drawn into an instance fits the carton (`catalog.json` `fits_carton`: its resting footprint 2 cm inside
the 0.43 x 0.30 m inner box on every side, either way round, and no taller than the 0.33 m rim). Each episode stands
alone: the carton starts empty; the start moves X to a random spot and yaw within 0.35 m of a platform edge (a two-hand
X with its wide side along that edge) and the carton to the two-hand stance band, 0.18-0.31 m in from a random edge
with its long side along it (`placement.objects`); then every other object on the platform is moved to a random spot
clear of the task objects and of the hands' corridors (`shuffle_others`, 2026-11-06: with them fixed, 13 of s0's 21
tasks found no start); the robot starts anywhere in the room it can walk from, 0.5 m or more from X.

- **Fixed inputs**: the scene file, the robot (mars G1 with Dex1 two-finger grippers), the controller ScaleBFM M 22200
  (not trained, not changed), MuJoCo, privileged poses from the simulator, the motor layer, `rules.md`.
- **The start of the skills** (no seed directory: `sandbox.seed` null keeps the main library as it stands):
  `skills/pack.py` chains `into.place_into` (one hand; the r1_in_single line's put-in) or `bimanual.place_into` (two
  hands, new on 2026-11-05: hug lift, raise over the rim, carry, lower over the opening, release) with
  `bimanual.pick` of the carton (the box line's hug, ported from `seeds/box_strict_start`). The two-hand put-in, the
  two-hand carry on foot and the lift of a loaded carton are untested before this run (section 7).
- **Done by the loop**: making the put-in land X inside for objects of every shape and the carton at every band
  position; the hug of a carton with an object in it; the lift height, standing up, keeping the carton level and steady
  through the window; accepting a change only through the gate.
- **Not done by people**: no parameters tuned, no skill code written beyond what the library holds at the run's
  commit. Every human intervention is recorded.

## 2. Structure

One inner loop on one family (no outer loop in this run); the user picks the instance by the family name
(`gso_pack_s3`). One iteration, as in every scene RSI run:

1. **Reviewer** (Claude) reads the incumbent's rounds on the diagnostic pool (records and 50 Hz traces) and writes at
   most 4 hypotheses: the cause, a parameter test, predicted flips. It may also claim that some starts cannot be done
   with the present actions (section 8).
2. **Prober** (program) runs each hypothesis' parameter test closed loop on the diagnostic pool.
3. **Manager** (Claude) reads the ledger and decides: `engineer` (which hypotheses to build), `reopen`, `review`, or
   `stop`.
4. **Engineers** (Claude, three in parallel) write code in the skills package and/or parameter overrides, one draft
   each; the **Composer** (Claude) makes one candidate of them.
5. **Gate** (program) compares candidate and incumbent start by start on the hidden acceptance pool.

Pools (D7): 3 diagnostic, 10 acceptance and 10 sealed starts per task (18 / 60 / 60 on the 6 one-hand tasks, 12 / 40 / 40 on the 4 two-hand ones; user 2026-11-24, a fixed 30 / 40 / 50 before; the sealed pool is drawn only after the loop ends),
rotated over the family's tasks; seeds 460000 / 560000 / 660000 (`gso_pack`) and 470000 / 570000 / 670000
(`gso_pack_bimanual`), disjoint from every other family's. The roles see only the diagnostic pool.

## 3. What exists below the skills

| Layer | What it is | Who may change it |
|---|---|---|
| Controller | ScaleBFM M 22200, follows keypoint targets (pelvis, wrists, ankles) | nobody |
| Motor layer | `src/locomani/backends/scalebfm/motor.py`: walk_to, turn, sidestep, posture, reach (one hand or `"both"`, `keep_squeeze`), grip and release (one hand or `"both"`), squeeze, hold, plus per-tick hand commands and sensors | default-off switches only |
| Skills | `src/locomani/tasks/scene/skills/`: `pack.py` (the chain), `bimanual.py` (hug pick, two-hand put-in), `into.py` (one-hand put-in), `pick.py` / `grasp_program.py` / `place.py` / `held.py` (what the one-hand put-in is built on) | the Engineer |
| Scene runner | `src/locomani/tasks/scene/` world, start (`placement.objects`), navigate (`walk_path`), episode, stage (`bimanual_grip_check`, `bimanual_lift_check`) | nobody |
| Goal predicates | `src/locomani/scene_discovery/predicates.py`: `in(X, C)` by the rotated box centres; `held(X)`; `lifted(X, h, tilt)` | nobody |
| Scene | `outputs/scene_discovery/logistics_room_1030_s<seed>/` (scene_meta.json: the platform as surface `A` with its approach edges, every object with its `box` extents, `semantic` and a `gso` block, the carton's `box.wall`, the room's bounds, obstacles and spawn region) | nobody inside the loop |

The room, the platform, the scanned objects and the carton are as `gso_bimanual.md` section 3 describes them. What the
pack goal adds:
- The carton's rim stands 0.70 m above the floor (platform 0.371 + walls 0.33); its opening is 0.43 x 0.30 m inside,
  its long side along the edge it is nearest to. From that edge its centre is 0.18-0.31 m in.
- X and the hands together do not fit through the opening (a two-hand hold is wider than the carton's short inner
  side), so a release happens above the rim and X falls 3-40 cm onto the carton's floor.
- `into.place_into` (one hand) stands at the carton's edge with the opening at most 0.40 m in and releases with X's
  footprint 1 cm inside the inner box and its underside 3 cm above the rim (measured 2026-10-28 on bowls and bins at
  0.547-0.832 m; the carton's rim is within that range).
- `bimanual.place_into` (two hands) hugs X as the pick does, raises both hands with the squeeze kept until X's underside
  clears the rim, backs off, walks around the platform to a stance square to the carton's edge with X over the opening,
  lowers X to its underside a hand's clearance above the rim, releases both hands, rises, retracts and steps back.

## 4. Decisions

| # | Decision |
|---|---|
| D4 | The two-hand actions stay atomic (`grip` / `release("both")`, `squeeze`, `reach("both", keep_squeeze=True)`); where to stand, where to put the hands, how high to carry and where to let go are the skills' (the loop's). |
| D6 | The Engineer may add and change skills in the skills package (code, tests, parameters). The motor layer takes only default-off switches; the controller, simulator, scene, predicates, success rule and gate are frozen. |
| D7 | One family = one group of tasks; pools as in section 2, from the family's random placements and spawns; disjoint seeds; the sealed pool is drawn after the loop. |
| D8 | The loop stops at the first of: the incumbent wins 95 % of the acceptance pool, the run has been going 20 h, it has taken 5 iterations (the family file's `stop_rate`, `max_wall_h`, `max_run_iters`; user 2026-11-24, 80 % and 5 h before), 12 gate calls, or the Manager's stop. |
| D11 | The only human knowledge is the versioned `rules.md` and this plan; interventions are recorded. |
| D12 | The success rule (user 2026-11-05): `in(X, nor_cal_box)`, `held(nor_cal_box)` and `lifted(nor_cal_box, 0.20, 60)` at every 0.1 s check over the round's last 5 s. |
| D13 | Infeasibility is handled as in section 8. X always starts within 0.35 m of an approach edge and the carton on the hug's band (the `placement` option); every drawn object fits the carton by the catalog's rule. |
| D14 | The failure classes are the two-hand set of `gso_bimanual.md` for the carton's lift plus the put-in's: pick_failed, carry_dropped, missed_opening (released outside), stuck_on_rim, in_but_not_lifted. |
| 2026-11-05 | User: X into the carton, then the carton held and lifted (no `upright`); the hug pick and both put-ins live in the main library as atomic skills, no seed directory; ten instances s0-s9, the user picks one per inner loop; code and smoke test first, no RSI run yet. |

## 7. What the layers below deliver (measured)

Motor layer on the open floor and the carton hug: `gso_bimanual.md` section 7 (walk 2 m stops 2.0 cm off; the hug
lifts the carton 15-29 cm with `squeeze(0.06)` and `reach("both", keep_squeeze=True, pelvis_z=0.79)`).

The one-hand put-in (r1_in_single, 2026-10-28, bowls and bins at 0.55-0.83 m rim height): release over the opening
lands the object inside in the accepted incumbent's rounds; what failed was the pick's creep budget and stances with
no route, not the drop.

Not measured before this run (2026-11-05):
- a two-hand hug of any scanned object (the `gso_bimanual` families have not run either);
- walking with an object held in both hands by a kept squeeze: the squeeze offsets ride on the hand servos through
  `walk_to`, but no round has carried a hugged object a step;
- raising a hugged object until its underside clears a 0.70 m rim (jaw centres at 0.73-0.96 m by X's height) and
  lowering it over the opening;
- hugging and lifting the 0.3 kg carton with 0.05-0.3 kg inside it (until 2026-11-07: the 1.2 kg carton with up to 1.5 kg; the payload of the strict box line was the empty 1.2 kg
  carton).

The smoke test in `docs/rsi_logistic_room_1030.md` (one episode of each family on s0) is the first measurement.

## 8. Starts that cannot be done (D13)

| # | Rule |
|---|---|
| D13.1 | Three kinds: physically impossible (out of reach, over the payload, inside something that cannot open) -- only the user confirms this; cannot with the present actions -- a skill gap with an exit; not done yet -- the loop keeps working. |
| D13.2 | A claim names the actions it assumed, a hard constraint code can recompute (a check file), at least one exit ("if the robot could first ..., it could") or why no action could, and what would refute it. Code refuses a claim missing any part. |
| D13.3 | A start where X cannot be released over the opening from any stance (the carton's opening deeper than the two-hand reach from every edge) is a `cannot with the present actions` claim with the exit "reach farther" or "stand closer"; the carton's band keeps its centre within 0.31 m of an edge, so such a claim must show the measured reach falls short. |
