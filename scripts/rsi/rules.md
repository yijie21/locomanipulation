# Rules for the scene RSI loop (TODO 27, version 14, 2026-10-22)

With the family's plan (the sections of its file's `plan` the prompts include), the only human knowledge the loop
reads (plan D11). It is distilled from the project's notes and the warehouse loop's rules. Every rule says where it is
enforced: **code** (the gate, the runner or the driver refuses) or **prompt**. A new rule is a new version of this
file; it, or a change to a family plan's text, is committed and counted as one human intervention.

## The task and its limits (code)

1. The scene, the controller (ScaleBFM M 22200), the motor layer, the simulator, the goal predicates and the success
   rule are frozen. Making a task easier is not an improvement. The Engineer may change only the skills package
   (`src/locomani/tasks/scene/skills/`); any other changed file voids the candidate (the driver checks and restores).
2. A start is never removed from a pool (rule 26's redraw of the acceptance pool aside). A start a skill refuses ("no
   side fits the jaw...") still counts as a failure. A round still running at 300 simulated seconds ends as a failure,
   "episode time limit" (code, version 12).
3. A candidate is accepted only by `gate.py` on the hidden acceptance pool. Nothing else counts as evidence that a
   change helps. The gate's per-round files are off limits; its verdict JSON is what a role may read. While a role
   runs the hidden tree cannot be read, and the sandbox's family file has no acceptance or sealed seeds (code, version
   12). A role whose tool calls name the hidden tree or a path outside the sandbox voids what it produced -- its
   iteration, or only its draft when it is one of several Engineers -- and the loop goes on; two voided iterations in a
   row stop it (code, version 13; until then a voided Reviewer, Manager or Composer stopped the loop).
4. A candidate is the skills package plus parameter overrides (`candidate.json` {"set": ...}, `skills.apply_set` keys:
   "NAME", "module.NAME", "module.NAME.key"). An unknown key is refused (code).

## Evidence (prompt; code where marked)

5. Read the per-round records and the 50 Hz traces, not the failure strings. A failure string names where the code gave
   up, not the cause.
6. Use per-round statistics (the value at the deciding moment, a round's min or max), then aggregate across rounds.
   Per-frame aggregates are weighted by dwell time; stalled rounds dominate them.
7. Only closed-loop re-runs from a start are evidence (the Prober's). Replaying commands or re-solving IK offline is not.
8. A diagnostic pool (a few starts per task) measures direction only; the same configuration has moved 6/10 to 2/10
   on repeats.
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
16. Walks run at 0.25 m/s (version 10; versions 7-9 ran at 0.5 m/s). At this speed ScaleBFM shuffles, its feet
    sliding on the floor. Straight walks stop within 1.5-2 cm; in-place turns end 2-5 deg off. The motor layer walks
    straight lines only;
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
    candidate runs on that park's diagnostic starts only, and goes to the gate only if it finishes one of them. The
    park lifts only if the gate accepts it (version 12; until then a rejected reopen lifted it too).
23. `held(X)` needs both fingers of one hand on X (or both hands) and X off every surface (success rule 2,
    2026-10-16). Rounds recorded under an older success rule or other motor settings are set aside and run again
    (code).
24. A family the outer loop made for a gap (its file's `about` names the new skills, e.g. `r1_on_single`: no skill
    puts an object down) exists to build those skills here: the first candidates may be a plain end-to-end version
    (`plan_for` covering the goal, the existing pick reused), improved one Engineer call at a time. They are not gap
    proposals for the outer loop; only a further skill outside the family's gap is (version 4, phase 2).

## The gate and the stops (code; version 5, user 2026-10-18)

25. The gate's stage 1 takes its 30 starts from every task in proportion to the task's share of the acceptance pool,
    so a candidate that changes one task is judged on that task's starts too. Against an incumbent with no success
    on the pool, a candidate may drop up to half its successes (otherwise, as before, the incumbent's drops + 1). The
    D13.4 stop counts only when the starts outside the parked tasks are at least half the acceptance pool. The verdict
    (version 14, TODO 33.8, user 2026-10-22: early research): accepted when b - c >= 3 and c <= b / 3 on the full
    pool; the one-sided McNemar p is reported, not required (until then p <= 0.05 was, which alone needs 5:0, so a
    change that fixes one object's 4 acceptance starts could never pass). A candidate rejected under the older rule
    may be judged again; the sealed pool at the end still decides what the run reports.

26. When task-wide parks cover more than half the acceptance pool, the gate judges on an acceptance pool of the same
    size drawn afresh from the tasks that are not parked (new seeds; drawn again whenever that set changes), so a
    candidate's gain on the live tasks can be told from chance. Parked starts still count in the sealed pool and in
    the diagnostic pool (version 6, user 2026-10-19).

27. A family whose skills package starts from an empty seed (its file's `sandbox.seed`; `grasp_atomic`, TODO 31)
    builds its goal's skills from the motor layer: the seed covers no goal and has no parameters, so the first
    hypotheses are code (`needs_code`) with an empty parameter test, and the first candidates may be a plain
    end-to-end version, improved one Engineer call at a time (rule 24 without an existing pick to reuse). Grasps are
    computed from the scene's geometry (the object's box and mesh, its surface, the gripper's size). The sandbox holds
    no earlier skill catalog, loop record or grasp program; do not look for them outside it (version 8, user
    2026-10-20). Its experience library keeps the lessons, not the code earlier runs ended with (code: no
    `final.patch`; version 12 -- in TODO 32.12 an Engineer read one).

28. An iteration builds several candidates (`run --candidates K`, TODO 32.9; K = 3 by default since version 11, and
    K = 1 is one Engineer without a Composer): the Manager lists up to K hypotheses,
    K Engineers build them in parallel from the incumbent (each in its own copy of the skills package), each candidate
    runs on the diagnostic pool, and the Composer merges into one candidate the edits whose diagnostic flips show
    their mechanism, adopts one candidate, or keeps the incumbent. Only that one candidate reaches the gate, so each
    gate call stays one paired test (code); the Composer ties every kept or dropped edit to the starts and the trace
    moment that decided it (prompt) (version 9, user 2026-10-21). A reopen uses one Engineer. Identical drafts run
    once. A file changed outside the skills copies voids every draft of the iteration: the Engineers share the sandbox,
    so it cannot be told whose it is (code, version 12). Version 14 (TODO 33.8, 33.14, user 2026-10-22):
    - the gate's best 2 rejected candidates with more wins than losses since the incumbent wait in
      `loop/candidates_pool.json` (cleared when the incumbent changes); the Manager may name one as `base`, so the
      Engineers build on it (the gate still compares with the incumbent), and the Composer may merge its edits or adopt
      it as it stands (code);
    - a mechanism Reviewers proposed in two or more iterations and no Engineer was given is built first, unless the
      Manager writes why not; the Manager links repeats to earlier ids (prompt; the driver counts them);
    - at most two iterations in a row reopen a park; after two, `reopen` is not offered (code).

29. The sandbox holds the experience library (scripts/rsi/experience, TODO 32.10): one entry per family that finished
    before, without this family's own or those its file's `library_exclude` names (code: sandbox.py), and the outer
    loop adds the family's entry when it ends
    (outer_run.py). Its numbers come from other objects and surfaces; this family's rounds decide (prompt) (version
    11, user 2026-10-22: three candidates, the Composer and the library are the default after TODO 32.12).

30. A skills package may hold flows for one object or one kind of object (version 14, TODO 33.13, user 2026-10-22:
    "允许给每个物体进化出它独有的抓取 pipeline……在之后抓类似物体的时候也需要复用"). A flow lives under `recipes/`,
    carries a `SIGNATURE` of the features it is for (hull width at the grasp height, height, flat or tall, container,
    wall, mass, friction, support height, a cover above) and `match(obj) -> score`, and the package's dispatcher picks
    by features, the generic pick as the fallback, so a similar object reuses it. A flow is never chosen by an
    object's name: a string constant equal to a scene object's key, body, ADT instance name or name stem (Mug_2,
    Mug_2_object, Mug) outside a docstring voids the draft (code); semantic fields (category, prototype) are features.
