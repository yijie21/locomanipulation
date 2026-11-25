# The whole pack task of one logistics_room_1030 instance in one loop: the plan the roles read (2026-11-07)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). The robot, the controller, the loop, the room and the chain are those of `gso_pack.md` (the put-in and the
carton's lift) and `gso_bimanual.md` (the two-hand actions); what differs is what the loop optimises: one family holds
the pack task of every object of the instance, and the gate compares two versions of the skills start by start on
how far along the chain each round got, not only on whether it finished it (user 2026-11-07: "what I want most is one
complete task: pick the target, walk to the box, put it in, lift the box").

## 0. Goal

The robot walks to the pallet platform, picks one scanned object (with one hand or, a wide one, with both), carries it
to the open carton on the same platform, puts it in, and lifts the carton with both hands. This run is one task family,
`gso_pack_all_s<seed>`: every pack task of the instance (10 on each of s0-s9: 6 one-hand objects, 4 two-hand ones; the
carton has no pack task), every task with the same goal, placement and judging:

| Tasks | From | Goal | Judged |
|---|---|---|---|
| `pack_X` (6) | `gso_pack_s<seed>`, every single-hand object | `in(X, nor_cal_box)`, `held(nor_cal_box)`, `lifted(nor_cal_box, 0.20, 60)` | every 0.1 s over the round's last 5 s (`judge_s`), the plan's end and 2 s still (`hold_s`) |
| `pack_X` (4) | `gso_pack_bimanual_s<seed>`, every two-hand object but the carton | as above, X put in with both hands | as above |

The timing is on each task row (`hold_s` / `judge_s` in `gso_s<seed>_tasks.jsonl`, read by `family.timing`). The
objects, their placements and the robot's spawn are as `gso_pack.md` section 0 describes: the task object within
0.35 m of a platform edge, the carton on the hug's band, 0.18 m free beside each wall a hug squeezes, every other
object on the platform moved to a random spot clear of the task objects and those corridors (`shuffle_others`), the
robot anywhere in the room it can walk from, 0.5 m or more away.

**The progress stage (D15).** Every round record carries `stage` (0-4) and `stage_name`, computed by the driver from
the trace: 4 `goal` -- the round succeeded; 3 `in_container` -- the object was inside the carton at some moment
(the `in` predicate's box), even if the carton's lift then failed or the object came out; 2 `over_opening` -- the
object's underside was above the rim over the carton's inner footprint at some moment; 1 `picked` -- it rose 8 cm
above its settled height; 0 `none`. A stage is what the round *ever* reached, not its end state: a round that put the
object in and then dropped the carton is a 3, which the end state alone would hide. The gate's win / loss on a start
is the candidate's stage against the incumbent's on that start (section 2 step 5, rules.md 25 and 32).

- **Fixed inputs**: the scene file, the robot (mars G1 with Dex1 two-finger grippers), the controller ScaleBFM M 22200
  (not trained, not changed), MuJoCo, privileged poses from the simulator, the motor layer, `rules.md`.
- **The start of the skills** (`sandbox.seed` null keeps the main library as it stands): `plan_for` sends an
  `in(X, carton)` + the carton's lift to `pack.py`, which chains the one-hand (`into.place_into`) or two-hand
  (`bimanual.place_into`) put-in with `bimanual.pick` of the carton. No round of it has passed a gate on these
  objects; in the s0 diagnostic rounds of 2026-11-06 no pack round reached stage 3.
- **Done by the loop**: the whole chain. A change made for the pick is gated on the put-in and the lift of every
  task too: the hug is the second half of every task, the one-hand pick the first half of six of them.
- **Not done by people**: no parameters tuned, no skill code written beyond what the library holds at the run's
  commit. Every human intervention is recorded.

## 2. Structure

One inner loop on one family (no outer loop in this run); the user picks the instance by the family name
(`gso_pack_all_s3`). One iteration, as in every scene RSI run:

1. **Reviewer** (Claude) reads the incumbent's rounds on the diagnostic pool (records and 50 Hz traces) and writes at
   most 4 hypotheses: the cause, a parameter test, predicted flips. It may also claim that some starts cannot be done
   with the present actions (section 8). The digest and the summary give the stage counts per task: read where the
   chain breaks before reading the rounds.
2. **Prober** (program) runs each hypothesis' parameter test closed loop on the diagnostic pool.
3. **Manager** (Claude) reads the ledger and decides: `engineer` (which hypotheses to build), `reopen`, `review`, or
   `stop`.
4. **Engineers** (Claude, three in parallel) write code in the skills package and/or parameter overrides, one draft
   each; the **Composer** (Claude) makes one candidate of them.
5. **Gate** (program) compares candidate and incumbent start by start on the hidden acceptance pool: a start the
   candidate reaches a higher stage on is a win (b), a lower one a loss (c), the same stage a tie. The rule is the
   usual one (rules.md 25: b - c >= 3 and c <= b / 3, the falls' and drops' limits), so a change that moves three
   starts from the pick to the put-in is accepted before any start completes the chain. The verdict reports the
   stages behind b and c (`stage`: wins, losses, ties, each arm's mean stage, per task) beside the per-class tally.

Pools (D7): 3 diagnostic, 10 acceptance and 10 sealed starts per task (30 / 100 / 100 on 10 tasks; user 2026-11-24, 3 / 4 / 5 before; the sealed pool is
drawn only after the loop ends), rotated over the tasks in the order of the table in section 0; seeds 490000 / 590000
/ 690000, disjoint from every other family's. The roles see only the diagnostic pool. The stops (D8) and the sealed
pool count stage 4 alone: the run's success rate is the share of starts that finished the whole chain.

## 3. What exists below the skills

| Layer | What it is | Who may change it |
|---|---|---|
| Controller | ScaleBFM M 22200, follows keypoint targets (pelvis, wrists, ankles) | nobody |
| Motor layer | `src/locomani/backends/scalebfm/motor.py`: walk_to, turn, sidestep, posture, reach (one hand or `"both"`, `keep_squeeze`), grip and release (one hand or `"both"`), squeeze, hold, plus per-tick hand commands and sensors | default-off switches only |
| Skills | `src/locomani/tasks/scene/skills/`: `pick.py` / `grasp_program.py` / `place.py` / `held.py` (one hand), `bimanual.py` (hug pick, two-hand put-in), `into.py` (one-hand put-in), `pack.py` (the chain) | the Engineer |
| Scene runner | `src/locomani/tasks/scene/` world, start (`placement`, `placement.objects`), navigate (`walk_path`), episode, stage (`bimanual_grip_check`, `bimanual_lift_check`) | nobody |
| Goal predicates | `src/locomani/scene_discovery/predicates.py`: `held(X)`, `lifted(X, h, tilt)`, `in(X, C)` | nobody |
| Progress stage | `scripts/rsi/failure_class.py` `stage()` from the pack events (`x_rise_max_cm`, `x_over_opening_ever`, `x_in_ever`) | nobody |
| Scene | `outputs/scene_discovery/logistics_room_1030_s<seed>/` (scene_meta.json: the platform as surface `A` with its approach edges, every object with its `box` extents, `semantic` and a `gso` block, the carton's `box.wall`, the room's bounds, obstacles and spawn region) | nobody inside the loop |

The room, the platform, the scanned objects and the carton: `gso_bimanual.md` section 3. The two-hand actions:
`gso_bimanual.md` section 3 (rules.md 35). What the pack goal adds (the rim height, the opening, where each put-in
releases): `gso_pack.md` section 3.

## 4. Decisions

| # | Decision |
|---|---|
| D4 | The motor actions stay atomic, the two-hand ones included; where to stand, where to put the hands, how hard to squeeze, how high to carry and where to let go are the skills' (the loop's). |
| D6 | The Engineer may add and change skills in the skills package (code, tests, parameters). The motor layer takes only default-off switches; the controller, simulator, scene, predicates, success rule, stage and gate are frozen. |
| D7 | One family = the pack task of every object of one instance; pools as in section 2, from the tasks' random placements and spawns; disjoint seeds; the sealed pool is drawn after the loop. |
| D8 | The loop stops at the first of: the incumbent wins 95 % of the acceptance pool (stage 4), the run has been going 20 h, it has taken 5 iterations (the family file's `stop_rate`, `max_wall_h`, `max_run_iters`; user 2026-11-24, 80 % and 5 h before; user 2026-11-06); or at a hard stop: 12 gate calls, the Manager's stop. |
| D11 | The only human knowledge is the versioned `rules.md` and this plan; interventions are recorded. |
| D12 | The success rule is `gso_pack.md`'s: `in`, `held` and `lifted` of the carton over the last 5 s. |
| D13 | Infeasibility is handled as in section 8. The task object always starts within 0.35 m of an approach edge and the carton on the hug's band (the `placement` option). |
| D14 | The failure classes are `gso_pack.md`'s (pick_failed, carry_dropped, missed_opening, stuck_on_rim, in_but_not_lifted), with the object's own class in `events.pick_class` and the carton lift's in `events.lift_class`. |
| D15 | The gate compares the two versions start by start on the progress stage (section 0; user 2026-11-07): higher wins, lower loses, equal ties; the acceptance rule, the stops and the sealed pool are unchanged, the stops and the sealed pool counting stage 4 alone. A stage is what the round ever reached. Rejected: the binary ok / fail gate (with 0 successes every candidate ties on every start and nothing can be accepted), end-state stages (hide the put-in that preceded a lost lift), and changing `gso_pack.md`'s per-kind families (they stay as they are). |
| 2026-11-07 | User: one family of the ten pack tasks, one-hand and two-hand objects together; the gate's stage comparison is a recorded human intervention (rules.md version 24). |

## 7. What the layers below deliver (measured)

The motor layer, the carton hug and the probes behind them: `gso_bimanual.md` section 7. The one-hand put-in and what
was not measured before the pack line: `gso_pack.md` section 7. The one-hand pick on scanned objects: `gso_grasp.md`
section 7. No round of any gso family has passed a gate before this run; the 20 pack rounds of the s0 diagnostic pool
of `gso_all_s0` (2026-11-06, the main library as it stood) were pick_failed 13 and carry_dropped 7 -- no object
reached the opening, so the put-in and the carton's lift after a put-in are unmeasured.

## 8. Starts that cannot be done (D13)

| # | Rule |
|---|---|
| D13.1 | Three kinds: physically impossible (out of reach, over the payload, inside something that cannot open) -- only the user confirms this; cannot with the present actions -- a skill gap with an exit; not done yet -- the loop keeps working. |
| D13.2 | A claim names the actions it assumed, a hard constraint code can recompute (a check file), at least one exit ("if the robot could first ..., it could") or why no action could, and what would refute it. Code refuses a claim missing any part. |
| D13.3 | Two pieces of evidence, run by the driver: the check holds on every claimed start, and the strongest present attempts (1-3, closed loop) all fail. A narrative alone does not park anything. |
| D13.4 | A park only removes those starts as repair targets; they still count as failures in every pool. The 95 % stop also counts on the unparked starts when they are at least half the acceptance pool. |
| D13.5 | Any incumbent that finishes a parked start lifts the park, and the claim is recorded as wrong. |
| D13.6 | A new skill that fits one Engineer call stays in the loop (for a park's exit: the Manager's `reopen`); one that does not becomes a gap proposal. |
