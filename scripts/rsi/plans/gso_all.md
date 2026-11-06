# Every task of one logistics_room_1030 instance in one loop: the plan the roles read (2026-11-06)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). The robot, the controller, the loop and the room are those of `gso_grasp.md`, `gso_bimanual.md` and
`gso_pack.md`; what differs is that one family holds all their tasks on one scene instance, so every iteration
diagnoses, changes and gates the one-hand pick, the two-hand hug and the pack chain together (user 2026-11-06).

## 0. Goal

The robot walks to the pallet platform and, depending on the task, lifts a scanned object with one hand, lifts a wide
object or the carton with both hands, or puts an object into the open carton and then lifts the carton with both
hands. This run is one task family, `gso_all_s<seed>`: every task of the instance's four families (21 tasks on each of
s0-s9), each with its own family's goal, placement and judging:

| Tasks | From | Goal | Judged |
|---|---|---|---|
| `pick_X` (6) | `gso_grasp_s<seed>`, every single-hand object | `held(X)`, `lifted(X, 0.10, 60)` | at the end of the plan, then 2 s still (`hold_s`) |
| `lift_X` (5) | `gso_bimanual_s<seed>`, every two-hand object and the carton | `held(X)`, `lifted(X, 0.20, 60)` | every 0.1 s over the round's last 5 s (`judge_s`), the plan's end and 2 s still |
| `pack_X` (6) | `gso_pack_s<seed>`, every single-hand object | `in(X, nor_cal_box)`, `held(nor_cal_box)`, `lifted(nor_cal_box, 0.20, 60)` | as `lift_X` |
| `pack_X` (4) | `gso_pack_bimanual_s<seed>`, every two-hand object but the carton | as above, X put in with both hands | as `lift_X` |

The timing is on each task row (`hold_s` / `judge_s` in `gso_s<seed>_tasks.jsonl`, read by `family.timing`); a row
without them is judged at the end of the plan. The objects, their placements and the robot's spawn are as the three
plans' section 0 describe: the task object within 0.35 m of a platform edge, the carton of a pack task on the hug's
band, 0.18 m free beside each wall a hug squeezes, every other object on the platform moved to a random spot clear of
the task objects and those corridors (`shuffle_others`), the robot anywhere in the room it can walk from, 0.5 m or more
away.

- **Fixed inputs**: the scene file, the robot (mars G1 with Dex1 two-finger grippers), the controller ScaleBFM M 22200
  (not trained, not changed), MuJoCo, privileged poses from the simulator, the motor layer, `rules.md`.
- **The start of the skills** (`sandbox.seed` null keeps the main library as it stands): `plan_for` sends a one-hand
  `held` + `lifted` to the one-hand pick (`held.pick_held` over `pick.py`'s PLAN_HELD pick), a `held` of an object the scene file marks
  `bimanual` to the two-hand hug (`bimanual.pick`), and an `in(X, carton)` + the carton's lift to `pack.py`, which
  chains the one-hand (`into.place_into`) or two-hand (`bimanual.place_into`) put-in with `bimanual.pick` of the
  carton. None of them has passed a gate on these scanned objects.
- **Done by the loop**: all three chains at once. A change made for one kind of task is gated on every task of the
  family, so it must not break the others: the hug is shared by `lift_X` and the second half of every `pack_X`, the
  one-hand pick by `pick_X` and the first half of the one-hand `pack_X`.
- **Not done by people**: no parameters tuned, no skill code written beyond what the library holds at the run's
  commit. Every human intervention is recorded.

## 2. Structure

One inner loop on one family (no outer loop in this run); the user picks the instance by the family name
(`gso_all_s3`). One iteration, as in every scene RSI run:

1. **Reviewer** (Claude) reads the incumbent's rounds on the diagnostic pool (records and 50 Hz traces) and writes at
   most 4 hypotheses: the cause, a parameter test, predicted flips. It may also claim that some starts cannot be done
   with the present actions (section 8).
2. **Prober** (program) runs each hypothesis' parameter test closed loop on the diagnostic pool.
3. **Manager** (Claude) reads the ledger and decides: `engineer` (which hypotheses to build), `reopen`, `review`, or
   `stop`.
4. **Engineers** (Claude, three in parallel) write code in the skills package and/or parameter overrides, one draft
   each; the **Composer** (Claude) makes one candidate of them.
5. **Gate** (program) compares candidate and incumbent start by start on the hidden acceptance pool.

Pools (D7): 2 diagnostic, 2 acceptance and 5 sealed starts per task (42 / 42 / 105 on 21 tasks; the sealed pool is
drawn only after the loop ends), rotated over the tasks in the order of the table in section 0; seeds 480000 / 580000
/ 680000, disjoint from every other family's. The roles see only the diagnostic pool. With two diagnostic starts per
task, a hypothesis about one kind of task rests on few rounds: read the kinds together where they share a skill.

## 3. What exists below the skills

| Layer | What it is | Who may change it |
|---|---|---|
| Controller | ScaleBFM M 22200, follows keypoint targets (pelvis, wrists, ankles) | nobody |
| Motor layer | `src/locomani/backends/scalebfm/motor.py`: walk_to, turn, sidestep, posture, reach (one hand or `"both"`, `keep_squeeze`), grip and release (one hand or `"both"`), squeeze, hold, plus per-tick hand commands and sensors | default-off switches only |
| Skills | `src/locomani/tasks/scene/skills/`: `pick.py` / `grasp_program.py` / `place.py` / `held.py` (one hand), `bimanual.py` (hug pick, two-hand put-in), `into.py` (one-hand put-in), `pack.py` (the chain) | the Engineer |
| Scene runner | `src/locomani/tasks/scene/` world, start (`placement`, `placement.objects`), navigate (`walk_path`), episode, stage (`bimanual_grip_check`, `bimanual_lift_check`) | nobody |
| Goal predicates | `src/locomani/scene_discovery/predicates.py`: `held(X)`, `lifted(X, h, tilt)`, `in(X, C)` | nobody |
| Scene | `outputs/scene_discovery/logistics_room_1030_s<seed>/` (scene_meta.json: the platform as surface `A` with its approach edges, every object with its `box` extents, `semantic` and a `gso` block, the carton's `box.wall`, the room's bounds, obstacles and spawn region) | nobody inside the loop |

The room, the platform, the scanned objects and the carton: `gso_bimanual.md` section 3. The two-hand actions:
`gso_bimanual.md` section 3 (rules.md 35). What the pack goal adds (the rim height, the opening, where each put-in
releases): `gso_pack.md` section 3.

## 4. Decisions

| # | Decision |
|---|---|
| D4 | The motor actions stay atomic, the two-hand ones included; where to stand, where to put the hands, how hard to squeeze, how high to carry and where to let go are the skills' (the loop's). |
| D6 | The Engineer may add and change skills in the skills package (code, tests, parameters). The motor layer takes only default-off switches; the controller, simulator, scene, predicates, success rule and gate are frozen. |
| D7 | One family = every task of one instance; pools as in section 2, from the tasks' random placements and spawns; disjoint seeds; the sealed pool is drawn after the loop. |
| D8 | The loop stops at the first of: the incumbent wins 80 % of the acceptance pool, the run has been going 5 h, it has taken 5 iterations (the family file's `stop_rate`, `max_wall_h`, `max_run_iters`; user 2026-11-06); or at a hard stop: 12 gate calls, the Manager's stop. |
| D11 | The only human knowledge is the versioned `rules.md` and this plan; interventions are recorded. |
| D12 | The success rule is each task's own family's (section 0's table). |
| D13 | Infeasibility is handled as in section 8. The task object always starts within 0.35 m of an approach edge and a pack task's carton on the hug's band (the `placement` option). |
| D14 | The failure classes follow the task object: the one-hand set for a single-hand object, the two-hand set of `gso_bimanual.md` for a two-hand object or the carton, plus the put-in's of `gso_pack.md` on a pack task. |
| 2026-11-06 | User: one iteration runs on one instance and covers all four families' tasks (21), so all the skills evolve together. |

## 7. What the layers below deliver (measured)

The motor layer, the carton hug and the probes behind them: `gso_bimanual.md` section 7. The one-hand put-in and what
was not measured before the pack line: `gso_pack.md` section 7. The one-hand pick on scanned objects: `gso_grasp.md`
section 7. No round of any gso family has passed a gate before this run (2026-11-06); the smoke test in
`docs/rsi_logistic_room_1030.md` section 5a stopped both pack chains at the grasp on s0.

## 8. Starts that cannot be done (D13)

| # | Rule |
|---|---|
| D13.1 | Three kinds: physically impossible (out of reach, over the payload, inside something that cannot open) -- only the user confirms this; cannot with the present actions -- a skill gap with an exit; not done yet -- the loop keeps working. |
| D13.2 | A claim names the actions it assumed, a hard constraint code can recompute (a check file), at least one exit ("if the robot could first ..., it could") or why no action could, and what would refute it. Code refuses a claim missing any part. |
| D13.3 | Two pieces of evidence, run by the driver: the check holds on every claimed start, and the strongest present attempts (1-3, closed loop) all fail. A narrative alone does not park anything. |
| D13.4 | A park only removes those starts as repair targets; they still count as failures in every pool. The 80 % stop also counts on the unparked starts when they are at least half the acceptance pool. |
| D13.5 | Any incumbent that finishes a parked start lifts the park, and the claim is recorded as wrong. |
| D13.6 | A new skill that fits one Engineer call stays in the loop (for a park's exit: the Manager's `reopen`); one that does not becomes a gap proposal. |
