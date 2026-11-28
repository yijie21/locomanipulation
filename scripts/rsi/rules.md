# Rules for the scene RSI loop (TODO 27, version 31, 2026-11-27)

With the family's plan (the sections of its file's `plan` the prompts include), the only human knowledge the loop
reads (plan D11). It is distilled from the project's notes and the warehouse loop's rules. Every rule says where it is
enforced: **code** (the gate, the runner or the driver refuses) or **prompt**. A new rule is a new version of this
file; it, or a change to a family plan's text, is committed and counted as one human intervention.

## The task and its limits (code)

1. The scene, the controller (ScaleBFM M 22200), the motor layer, the simulator, the goal predicates and the success
   rule are frozen. Making a task easier is not an improvement. The Engineer may change only the skills package
   (`src/locomani/tasks/scene/skills/`); any other changed file voids the candidate (the driver checks and restores).
2. A start is never removed from a pool. Two exceptions, both the driver's and neither a role's: rule 26's redraw of
   the acceptance pool, and the diagnostic pool's redraw when the incumbent fails fewer than 10 of its starts -- the
   driver draws fresh starts from the diagnostic seed range, runs the incumbent on them and swaps the failing ones in
   for passing starts until 15 fail; retired starts move to `pool_diag_retired/`, every task keeps two starts and one
   passing start, parked starts stay (code, version 17, TODO 33.23). A start a skill refuses ("no
   side fits the jaw...") still counts as a failure. A round still running at 300 simulated seconds ends as a failure,
   "episode time limit" (code, version 12).
3. A candidate is accepted only by `gate.py` on the hidden acceptance pool. Nothing else counts as evidence that a
   change helps. The gate's per-round files are off limits; its verdict JSON is what a role may read. While a role
   runs the hidden tree cannot be read, and the sandbox's family file has no acceptance or sealed seeds (code, version
   12). A role whose tool calls name the hidden tree or a path outside the sandbox voids what it produced and the
   loop goes on. A Reviewer, Manager or Composer that does is called once more, in a fresh session on the same inputs
   and told which path the first call named; the second call's leak voids the iteration (code, version 29, user
   2026-11-27: until then the first leak voided it, and before version 13 it stopped the loop). One of several
   Engineers' leak voids only its draft. Two voided iterations in a row stop the loop (code, version 13).
4. A candidate is the skills package plus parameter overrides (`candidate.json` {"set": ...}, `skills.apply_set` keys:
   "NAME", "module.NAME", "module.NAME.key"). An unknown key is refused (code).

## Evidence (prompt; code where marked)

5. Read the per-round records and the 50 Hz traces, not the failure strings. A failure string names where the code gave
   up, not the cause.
6. Use per-round statistics (the value at the deciding moment, a round's min or max), then aggregate across rounds.
   Per-frame aggregates are weighted by dwell time; stalled rounds dominate them.
7. Only closed-loop re-runs from a start are evidence (the Prober's). Replaying commands or re-solving IK offline is not.
8. A diagnostic pool (a few starts per task) measures direction only; the same configuration has moved 6/10 to 2/10
   on repeats. A pool the incumbent passes measures nothing, so the driver re-draws it (rule 2, version 17).
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
16. Walks run at 0.25 m/s (version 10; versions 7-9 ran at 0.5 m/s; a family file's `motor` block may change it: box_bimanual walks at 0.5 m/s, TODO 38.21, user 2026-10-24). At this speed ScaleBFM shuffles, its feet
    sliding on the floor. Straight walks stop within 1.5-2 cm; in-place turns end 2-5 deg off. The motor layer walks
    straight lines only;
    `navigate.walk_path` plans around furniture on the occupancy map (body radius 0.35 m: at 0.26 m an arm caught the
    kitchen island and the robot fell).
17. Furniture collides with the robot only below its lowest surface; free objects rest on surface slabs.

## Operations (code where marked)

18. ScaleBFM rounds run through `batch.py` (12 processes, GPUs 0 and 1, OMP 2). A role's own closed-loop checks go
    through `scripts/rsi/try.py` (version 16, TODO 37.1): the starts it names (`--labels`, `--task`, `--all`; at most
    40 per call), one skills package with its `--set`, run in parallel through the same queue and machine-wide slots,
    a table back (the outcome, the failure class, the first failing step, and against `--ref <the incumbent's
    diagnostic rounds>` the starts that flip). 240 rounds per role per iteration (code: the driver's ledger file; a
    call past it is refused). Do not run `scene_round.py` yourself, write loops around it, or wait on background
    work with `until` / `sleep` polls: run C's roles spent 10-minute waits on one-process runs while the queue stood
    idle. The Prober's probes and the gate remain the evidence (rules 3, 7); `try.py` is for looking.
19. Long jobs run under `nohup setsid` with a log; a foreground call that times out loses its children. A `try.py`
    call of 30 starts takes about 2-5 minutes and is run in the foreground.
20. Stop and write an escalation instead of acting when a step needs the scene, the controller or its training, the
    motor layer beyond a default-off switch, the predicates, the success rule, the frozen rules or the budget. When
    the incumbent fails two or more diagnostic rounds of a recurring class (now: falls on the walk), the driver runs
    each round's failing command alone from its start (`scripts/rsi/isolate.py`, the motor's call log) and gives the
    verdict in the context (`attribution`): motor -- the command alone fails the same way, the motor layer's or the
    controller's limit, written as an escalation and a body-model constraint; route around it (another first move,
    another stance), never patch the layer below; skill -- alone it does not fail (code: loop.py attribute; TODO
    33.20, D20) (version 20).

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
    may be judged again; the sealed pool at the end still decides what the run reports. The loop stops for the sealed
    pool when the incumbent wins 70 % of the acceptance pool; a run continued from a strong incumbent sets its own
    share with `run --stop-rate` (version 15, user 2026-10-22: grasp_atomic continued from 29/40). On a family whose
    rounds carry a progress stage (rule 32; the pack families, `gso_pack_all.md` D15; version 24, user 2026-11-07) a
    win is a start the candidate reaches a higher stage on than the incumbent and a loss a lower one, the same stage a
    tie; b and c are judged by the same rule, so a change that carries three starts further along the chain is
    accepted before any start completes it. The verdict reports the stages behind b and c (`stage`). The 80 % stop,
    `candidate_ok` and the sealed pool count stage 4 (the goal) alone. The verdict's `per_task` (version 30, TODO
    56.1, user 2026-11-27) gives each task's wins, losses and the Holm-corrected one-sided p that the candidate is
    worse on it (`regress` when significant at 0.05); it is a report for the roles and the human, never a reason to
    reject -- a candidate that lifts the pool while a task regresses is still accepted, and the Reviewer's next
    hypothesis should start from that task.

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
    so it cannot be told whose it is (code, version 12). Once half the drafts are in, the rest get
    `--engineer-late-min` minutes (10 by default), then the driver stops them: a stopped draft is void and `late` in
    the ledger, its hypothesis counts as not built, and rule 33 builds it first next time (code, version 16, TODO
    37.3). Each draft's diagnostic rounds start as soon as its Engineer answers. When the gate rejects the composed
    candidate, the best draft it has not judged (more diagnostic starts fixed than broken, the largest net first)
    goes to the gate in the same iteration -- one more gate call of the budget instead of another iteration
    (`--second-shot`, on by default; code, version 16, TODO 37.4). Version 14 (TODO 33.8, 33.14, user 2026-10-22):
    - the gate's best 2 rejected candidates with more wins than losses since the incumbent wait in
      `loop/candidates_pool.json` (cleared when the incumbent changes); the Manager may name one as `base`, so the
      Engineers build on it (the gate still compares with the incumbent), and the Composer may merge its edits or adopt
      it as it stands (code);
    - a mechanism Reviewers proposed in two or more iterations and no Engineer was given is built first, unless the
      Manager writes why not; the Manager links repeats to earlier ids (prompt; the driver counts them);
    - at most two iterations in a row reopen a park; after two, `reopen` is not offered (code).
    Under the family file's `compose: program` (`run --compose program`; version 31, TODO 56.2, user 2026-11-27; the
    gso families) the program composes first: the drafts that fix more diagnostic starts than they break, the largest
    net first, are folded into one package by a 3-way merge against their base (scripts/rsi/merge.py), their sets
    merged the same way; one positive draft goes to the gate as it stands. The Composer is called only when a
    draft's edits overlap another's (its working copy then holds the clean part of the merge, the context names the
    conflicts) or when no draft is positive (code). The compose ledger row says who decided (`by`, `mode`).

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

31. Every grasp step of a skill goes through `stage()` (src/locomani/tasks/scene/stage.py, TODO 33.16, D12): the motor
    call, a check of the measured state after it (`stance_check`, `reach_check` with its stall ratio `track_ratio`,
    `grip_check`, `lift_check`, or the skill's own) and an `on_fail` action (re-walk, re-approach, open and grip again,
    give up), within the round's retry budget. A step that missed is not followed by the next one as if it had worked:
    32.12's incumbent closed the jaw after reaching 10-36 cm off. The thresholds and the on_fail actions are the loop's
    to evolve (prompt) (version 15, user 2026-10-22: "把 33.15 到 33.18 先做了吧").

32. Every round record carries `failure_class` (too_wide, side_reach, pinch_reach, walk_off, knocked, slip, fell,
    other) and `events` (the walk's end, the pre-grasp's end, the grip, the lift, the hand's closest approach, the
    object's first move and the hand that moved it), computed by the driver from the trace and the scene file
    (scripts/rsi/failure_class.py, TODO 33.15); the gate reports its tally per class and the digest leads with the
    classes. A pack round (a goal with `in(X, C)`) is classed by its put-in (pick_failed, carry_dropped,
    missed_opening, stuck_on_rim, in_but_not_lifted; the object's own class in `events.pick_class`, the carton lift's
    in `events.lift_class`) and carries `stage` / `stage_name` (version 24; `gso_pack_all.md` D15): 0 none, 1 picked
    (rose 8 cm), 2 over the opening (above the rim over the container's inner footprint), 3 in the container (the `in`
    predicate's box), 4 the goal -- what the round *ever* reached, from the trace (`failure_class.stage`); the gate
    compares versions on it (rule 25), the digest and the summary count it per task. `src/locomani/tasks/scene/graspability.py` (and scripts/rsi/graspability.py) says, without simulating,
    whether an object fits the open jaw on each side, how far to turn so a side that fits faces the jaw, and which
    preparatory action would make it fit (TODO 33.18, D13). `src/locomani/tasks/scene/body.py` (scripts/rsi/body.py)
    says, from a table of 15,120 measured reaches, how likely the controller is to land a hand 3 cm or more off a
    target, and `src/locomani/tasks/scene/choose.py` (scripts/rsi/choose.py) lists for an object at a start every
    grasp, hand, stance and posture that stands clear of the furniture, scored by that table, best first, with the
    fallback order standing, lowered pelvis, preparatory action (TODO 33.1.4, 33.12; D1, D2, D9, D22). All three are
    infrastructure: skills call them, the loop does not change them, the driver rebuilds the table. None is evidence
    that a change helps; only the gate is (code: the driver writes them; prompt) (version 15; version 18 for the body
    model and the chooser).

33. Each hypothesis gets a `status` from its probe: verified (it fixed a diagnostic start and broke none), refuted (it
    fixed none), mixed, or hypothesis (not probed: its test needs code); fixed and broken are counted as the gate
    counts wins and losses (rule 25: a higher or lower stage on a pack family, ok / fail elsewhere; version 24). A mechanism Reviewers proposed in two or more
    iterations, given to no Engineer and not refuted takes one Engineer slot, unless the Manager's `skip` names the
    check that fails: interface_kept, safety_checks_kept, changed_code_reached or evidence_not_refuting. After a
    rejection the Manager reads the failure classes, the gate's per-class tally and a standing suggestion per class
    (code: loop.py force_build, repair_notes; TODO 33.17, D14) (version 15). A hypothesis whose test needs code may
    name a `measure` -- {feature, op, value, expect: fails | succeeds, tasks?, scope: incumbent | all} over a column of
    the record library (`scripts/rsi/records.py features` lists them: start, events, each step's numbers) -- and the
    driver sets its status from the library instead: verified when the rounds on the measure's side of the split fail
    (or succeed) more, at least 3 a side and Fisher's p <= 0.05; refuted when at least 10 a side show no difference or
    the opposite one; else it stays a hypothesis. Only rounds the roles may see, on this machine, under the incumbent's
    own version unless scope is all. A correlation chooses what to build; the gate still decides what stays (rule 7)
    (code: loop.py records_statuses, records.judge; TODO 33.15.1) (version 20).

34. Preparatory actions (TODO 33.5; D2, D5, D9): when the chooser (rule 32) marks an object `needs_prep`, the skill
    asks `locomani.tasks.scene.prep` which goal would give it a standing stance -- turn the object about the vertical,
    pull or push it toward an edge of its support, move a neighbour out of the hand's corridor, in the user's order
    (turn before pull before clear) -- and runs the action registered for that kind (`prep.register`), then chooses
    again on the measured state. The goals, the selector and the registry are infrastructure (the loop does not change
    them); the actions are skills the loop writes. A kind no action implements is a gap the Reviewer reports
    (`prep.wanted`). A task family may give its tasks a `placement` option (`depth_m`, `yaw_deg`, `blocker`) so its
    pools hold the situations that need these actions; it changes no other family's pools (code: start.sample_start,
    family.draw; prompt) (version 19).

35. Design probes (TODO 33.19; D16): a hypothesis may ask for one, `design` {factor, levels, starts, tasks?, name?}:
    one factor forced at a few levels on the same diagnostic starts, under the incumbent -- `param` (an overridable
    parameter `name` set to each level), `spawn_bearing` (the object `level` degrees off the robot's heading at the
    spawn), `spawn_distance` (the robot `level` metres from the object along the line to its spawn), `object_yaw` (the
    object turned `level` degrees in place). At most 2 an iteration, at most 300 rounds each; a start the forcing
    makes invalid is left out. The per-level outcome goes to the Manager, the next Reviewer and the body model's
    curves; it says how the factor changes the outcome, never that a change should stay (rule 7) (code: design.py,
    loop.py run_designs) (version 20).

36. Two-hand actions (TODO 38.4, D4; version 21): the motor layer has nine actions. `reach("both", (left, right), ...)`
    moves the two jaw centres together; `grip("both")` / `release("both")` close and open the two jaws together;
    `squeeze(inward_m, seconds)` pushes the two jaw centres towards each other along the line between them and keeps
    pushing -- the offsets ride on each hand's servo reference in the pelvis frame through posture, hold and walk_to
    until `release` or a `reach` of that hand drops them (`reach(..., keep_squeeze=True)` keeps them: a two-hand lift
    by the arms); while a squeeze is on, the servo's lead is capped at the offset, so the push is at most twice
    `inward_m`. Nothing in them decides the stance, the hand placement or the lift; a family whose object is wider than a jaw (`semantic.graspable` bimanual; `box_bimanual`, TODO 38) builds
    those from these actions. `held(X)` then needs the fingers of both hands on X and X off every surface (rule 23's
    second branch). The step checks for a two-hand hold are `stage.bimanual_grip_check` and `stage.bimanual_lift_check`
    (rule 31). The failure classes are the two-hand set when the family's object is bimanual (rule 32; TODO 38.11).
    Graspability, the chooser, the body table and the preparatory layer (rules 32, 34) describe one jaw: on a bimanual
    family they say nothing about the two-hand hold, and a hypothesis may not cite them for it (prompt).

37. Strict goals and judging windows (TODO 38.19, user 2026-10-24; version 22). `lifted(X, clearance_m, tilt_deg)`
    holds when the lowest corner of X's box is `clearance_m` or more above the top of the surface the scene file
    rests X on, and X's bottom face is less than `tilt_deg` off level. A family may set `judge_s`: the goal is then
    checked every 0.1 s and must hold at every check over the round's last `judge_s` seconds -- the end of the plan
    included, then the runner's `hold_s` standing still -- so a skill may still stand up or lift in the window, but
    the box may not slip, tip or drop at any check in it. The round record carries `judge` (the window, how many
    checks held, the first one that failed and why, `ever_ok`). `box_bimanual` uses `held(X)` + `lifted(X, 0.20, 60)`
    with `judge_s` 5 and `hold_s` 2; its failure classes add low_lift, tilted and dropped_hold (code: predicates.py,
    episode.judge_window, failure_class.py). Its seed is the first run's incumbent, not an empty package, so rule 27
    does not apply to it: change the pick that is there rather than writing a new one (prompt).

38. Scene-graph records (TODO 39, version 23; plan docs/handoff/past-scene-graph-plan-20261030.md). The runner writes
    `<label>.past.jsonl` beside every round: one record per skill step and retry, built after the round from the step
    log, the primitive calls, the trace with its contact channel and the scene file (`locomani.tasks.scene.past`). It
    is a measuring instrument: the skills package may not import it, read a `.past.jsonl` or call
    `scripts/rsi/past.py` -- a draft whose skills code names `past.jsonl`, `scene.past`, `scene import past` or
    `rsi/past.py` is void, like rule 30's object names (code: loop.py named_objects / the draft check). The records
    never enter the gate, the sentinel or the regression; they are data for the Reviewer (prompt: the digest's
    "Scene-graph transitions" section, `past.py show` / `compare`). Their `expected` transitions are a fixed table by
    primitive (walk lands and nothing blocks the stance; reach lands within 3 cm, the target and its neighbours do not
    move, nothing blocks the corridor; grip -> gripped by both fingers; a rising reach while gripping -> held and off
    its support; release -> not gripped and, when the goal names it, on the goal surface or in the container), not
    evidence that a change should stay (rule 7). Entities enter a record by relevance (roles, neighbours within
    0.30 m, furniture in the stance footprint or the approach corridor, surfaces within 0.5 m above the target,
    regions), not by contact; a value the recorder could not compute is `unknown`, never false.
39. The knowledge library (plan docs/handoff/shared-knowledge-plan-20261111.md; version 25). `scripts/rsi/knowledge.py
    build` turns a finished family's ledger and scene-graph records into entries: a condition over record-library
    columns, the failure signature it goes with, what repaired it and what did not, and the per-object evidence;
    the driver gives each entry a `generality` -- `multi_object` (held and helped on two or more objects, hurt
    none), `single_object` (one object in evidence) or `contested` (some object lost rounds, after narrowing to the
    positive objects' start.* ranges failed to exclude it; user 2026-11-23) -- and never by an object's name (rule
    30's check runs on the entry's text). Generality is about objects within the learning family; the `tag` below
    says which kind of task it came from. The entries live in one file, scripts/rsi/knowledge/knowledge.jsonl, each tagged by the kind of
    task it was learned on (`single` one-hand pick, `bimanual` two-hand pick; version 26, user 2026-11-18). Before
    each Reviewer the driver writes scripts/rsi/knowledge/KNOWLEDGE.md: the whole library, every entry marked with
    the incumbent's diagnostic rounds its condition holds on (a mark, not a filter), and an alternating run's code
    syncs; the Reviewer and the Engineer read it whole (prompt: what worked elsewhere is a hypothesis here; this
    family's rounds decide; cite an entry's id in `knowledge_refs`). Each citation is recorded with how the citing
    hypothesis ended (scripts/rsi/knowledge/usage.jsonl; version 27): an entry that helped where it was cited is
    `tested`, one cited at least twice without effect is `doubtful`, one whose condition holds only where the
    reader's incumbent already succeeds is `resolved` for that reader; KNOWLEDGE.md shows tested and candidate
    entries in full, doubtful and resolved ones a line each (code). The skills package may not read it: a draft
    whose skills code names `rsi/knowledge`, `knowledge_hits`, `HITS.md`, `KNOWLEDGE.md` or `knowledge.jsonl` is
    void, as rule 38's recorder (code: loop.py named_objects). The library never enters the gate, the sentinel or
    the regression.
40. Two families sharing one skills package (the alternating driver alternate.py, run outside the sandboxes; version
    26, user 2026-11-18). An alternating run gives each family one iteration in turn. After a family's iteration its
    incumbent is merged into the other family's skills package and enters there only through that family's gate
    under the non-inferior rule (gate.py --non-inferior: it may not lose more of the other family's acceptance
    starts than it wins); the verdict counts as one of that family's gate calls (code). Where both families changed
    the same lines since the last accepted sync the sync fails: nothing is merged, the conflict is recorded in
    sync.jsonl (KNOWLEDGE.md shows it to both families' Engineers) and the merge base stays, so the next sync tries
    again (code; version 28, user 2026-11-23: until then the receiver's side was kept silently and the sender's work
    there lost). Before the gate the merged package's own tests (skills/test_*.py) run in the receiving sandbox; a
    failure is a failed sync too (code). Each family's prompts name the other family and its skill file; a change
    to a module both skills use must stay correct for both, and an Engineer does not edit the other family's skill
    file for its own tasks (prompt).
41. The shared stage module skills/grasp_common.py (version 28, user 2026-11-23; the first alternating run, gso-alt-a,
    reached the same two principles in both families by different code). The computations both grasp skills need
    live in this one file, two sections: the lift judged against the goal's lifted(X, c, t) -- the goal read, the
    lowest box corner's clearance and the tilt measured, the check with a margin, the top-up when held but short --
    and the collision hull's cross-section (pieces, cut, band, extent within a strip, widths), from which the one-hand
    skill reads a width across the jaw and the two-hand skill the band where both walls are at full width. An
    Engineer may change it and add pure functions to it; a change must keep every test of the package passing, the
    other family's included (code for a sync: rule 40's tests; prompt for a family's own candidate). It holds no
    policy constants: every tolerance, margin, scan step and retry count is a parameter, set by a constant in the
    family's own module (pick.py, grasp_program.py, bimanual.py), so the families may tune them apart and `set`
    overrides keep their family keys; it reads no family configuration and no candidate overrides (prompt).
