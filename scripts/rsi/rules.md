# Rules for the scene RSI loop (TODO 27, version 7, 2026-10-20)

The only human knowledge the loop reads (plan D11). It is distilled from the project's notes and the warehouse loop's
rules. Every rule says where it is enforced: **code** (the gate, the runner or the driver refuses) or **prompt**. A new
rule is a new version of this file, committed, and counted as one human intervention.

## The task and its limits (code)

1. The scene, the controller (ScaleBFM M 22200), the motor layer, the simulator, the goal predicates and the success
   rule are frozen. Making a task easier is not an improvement. The Engineer may change only the skills package
   (`src/locomani/tasks/scene/skills/`); any other changed file voids the candidate (the driver checks and restores).
2. A start is never removed from a pool. A start a skill refuses ("no side fits the jaw...") still counts as a failure.
3. A candidate is accepted only by `gate.py` on the hidden acceptance pool. Nothing else counts as evidence that a
   change helps. The gate's per-round files are off limits; its verdict JSON is what a role may read.
4. A candidate is the skills package plus parameter overrides (`candidate.json` {"set": ...}, `skills.apply_set` keys:
   "NAME", "module.NAME", "module.NAME.key"). An unknown key is refused (code).

## Evidence (prompt; code where marked)

5. Read the per-round records and the 50 Hz traces, not the failure strings. A failure string names where the code gave
   up, not the cause.
6. Use per-round statistics (the value at the deciding moment, a round's min or max), then aggregate across rounds.
   Per-frame aggregates are weighted by dwell time; stalled rounds dominate them.
7. Only closed-loop re-runs from a start are evidence (the Prober's). Replaying commands or re-solving IK offline is not.
8. A diagnostic pool of 50 starts measures direction only; the same configuration has moved 6/10 to 2/10 on repeats.
9. Subsets picked because they succeeded fall back towards the mean. Do not tune on the rounds that succeed now.
10. Before saying information is absent, look for it for a few minutes. Before calling a constant decisive, find every
    caller of it.
11. When a list of fixes keeps growing, the layer is probably wrong: say so instead of adding item N+1.

## The robot and the controller (prompt)

12. mars G1 with Dex1 two-finger grippers: the jaw opens 9.5 cm, fingers close along the wrist's y axis, the jaw centre
    sits 0.1515 m ahead of the wrist along its x axis, finger force is the PD depth into the object (about 20 N max).
13. ScaleBFM follows keypoint targets (pelvis, wrists, ankles in the motor layer's modes 2 and 4; the torso only in
    mode 7). It crouches and leans when the hands ask for it. In mode 4 the waist is the controller's choice: a 20 deg
    reference lean came out as 8 deg, a squat with an upright reference leaned 21-56 deg.
14. Open loop, ScaleBFM's hands land 2-9 cm and 10-30 deg off at grasp poses; the motor layer's reach servo (jaw centre
    + wrist attitude integral) brings them to about 1 cm. During a fast entry the hand lags the command by up to 10 cm.
15. Standing in mode 2 the feet slide apart (27.7 -> 36.1 cm in 15 s); mode 4 pins them.
16. Walks run at 0.5 m/s (version 7, user 2026-10-20: at the earlier 0.25 m/s ScaleBFM shuffled, its feet sliding on the
    floor). Straight walks stop within 1.5-2 cm; in-place turns end 2-5 deg off. The motor layer walks straight lines only;
    `navigate.walk_path` plans around furniture on the occupancy map (body radius 0.35 m: at 0.26 m an arm caught the
    kitchen island and the robot fell).
17. Furniture collides with the robot only below its lowest surface; free objects rest on surface slabs.

## Operations (code where marked)

18. ScaleBFM rounds run through `batch.py` (12 processes, GPUs 0 and 1, OMP 2). For your own one-round checks use one
    process: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 <sim python> scripts/rsi/scene_round.py ...`.
19. Long jobs run under `nohup setsid` with a log; a foreground call that times out loses its children.
20. Stop and write an escalation instead of acting when a step needs the scene, the controller or its training, the
    motor layer beyond a default-off switch, the predicates, the success rule, the frozen rules or the budget.

## Infeasibility (code; plan section 8, D13; version 3)

21. A claim that starts cannot be done names the actions it assumed, a hard constraint as a check file the driver runs,
    exits (a new capability that would open them) or, only for a physical claim, why there are none, what would
    refute it, and 1-3 closed-loop attempts. Code refuses a claim missing any part. "Cannot with the present actions"
    is not "impossible"; only the user confirms impossible.
22. A claim parks its starts only if the check holds on every claimed start and every attempt fails. Parked starts
    still count as failures in every pool and are no longer repair targets. A park reopens, and is recorded as wrong,
    as soon as an incumbent finishes one of its starts. A skill too big for one Engineer call goes to the outer loop
    as a gap proposal. An exit that fits one Engineer call is built on the Manager's `reopen` (version 3): the
    candidate runs on that park's diagnostic starts only, and goes to the gate only if it finishes one of them (the
    park then lifts).
23. `held(X)` needs both fingers of one hand on X (or both hands) and X off every surface (success rule 2,
    2026-10-16). Under rule 1 one finger was enough: a mug hanging on closed fingers counted. Rounds judged under rule 1
    are set aside as `_rule1_*` and run again.
24. A family the outer loop made for a gap (its file's `about` names the new skills, e.g. `r1_on_single`: no skill
    puts an object down) exists to build those skills here: the first candidates may be a plain end-to-end version
    (`plan_for` covering the goal, the existing pick reused), improved one Engineer call at a time. They are not gap
    proposals for the outer loop; only a further skill outside the family's gap is (version 4, phase 2).

## The gate and the stops (code; version 5, user 2026-10-18)

25. The gate's stage 1 takes its 30 starts from every task in proportion to the task's share of the acceptance pool,
    so a candidate that changes one task is judged on that task's starts too. Against an incumbent with no success
    on the pool, a candidate may drop up to half its successes (otherwise, as before, the incumbent's drops + 1). The
    D13.4 stop counts only when the starts outside the parked tasks are at least half the acceptance pool.

26. When task-wide parks cover more than half the acceptance pool, the gate judges on an acceptance pool of the same
    size drawn afresh from the tasks that are not parked (new seeds; drawn again whenever that set changes), so a
    candidate's gain on the live tasks can be told from chance. Parked starts still count in the sealed pool and in
    the diagnostic pool (version 6, user 2026-10-19).
