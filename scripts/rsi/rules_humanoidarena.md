# Rules for the RSI loop on the HumanoidArena benchmark (adapted from rules_lml_bench.md version 9; this file is version 1, 2026-11-24 (version 1 of rules_humanoidarena.md, the rules for the bench adapter `locomani.bench.humanoidarena`, TODO 49.7a: the package is `skill.py` with a frozen `__init__.py`; rules 1, 12-17, 30-32, 35, 39-46 name the benchmark's tracker, action space, compliance check and guard; the family file is humanoidarena_pp_box))

With the family's plan (if its file names one), the only human knowledge the loop reads. The numbering follows
`scripts/rsi/rules.md`, `rules_simple.md` and the LocoManiLab bench rules, because the driver's messages cite rules by number. A rule that does not
apply on this backend keeps its number and says so in one line. Rules 39 to 49 are the ones written for the bench adapters. Every rule says where it is enforced: **code** (the gate, the runner, the guard or the driver refuses) or
**prompt**. A new rule is a new version of this file; it, or a change to the family plan, is committed and counted as
one human intervention.

## The task and its limits (code)

1. The benchmark is frozen: the simulator (Isaac Sim and Isaac Lab), the family's task (`pp_box`), its instruction, its step
   budget (1450 control steps), the benchmark's own success check, the compliance check (rule 40), the native tracker (TWIST2),
   the gait the tracker follows, the truth view and the failure classes. Making the task easier is not an improvement. The
   Engineer may change only the skill package (`skill.py`, rule 44); any other changed file voids the candidate (the driver
   checks and restores).
2. A start is never removed from a pool. A start the skill refuses ("no target seen", "no grasp") still counts as a
   failure. A round that has not succeeded when the task's time limit passes is a failure (rule 39 says what the limit
   is). The diagnostic pool is not redrawn on this backend. (code)
3. A candidate is accepted only by `gate.py`, in the benchmark's own simulator, on the hidden acceptance pool. Nothing
   else counts as evidence that a change helps. The gate's per-round files are off limits; its verdict JSON is what a
   role may read. While a role runs the hidden tree cannot be read. A role whose tool calls name the hidden tree, the
   benchmark's evaluation episodes or their results, or a path outside the sandbox voids what it produced -- its
   iteration, or only its draft when it is one of several Engineers -- and the loop goes on. A Reviewer, Manager or
   Composer whose output is voided runs once more in the same iteration, told why; only a second voided output
   voids the iteration (TODO 48.17). Two voided iterations in a row stop the loop. (code)
4. A candidate is the skill package plus parameter overrides (`candidate.json` {"set": ...}). An unknown key is
   refused. The bed's fixed switches (rule 46) are laid over every set and may not appear in one. (code)

## Evidence (prompt; code where marked)

5. Read the per-round records, not the failure strings. A failure string names where the code gave up, not the cause.
   The record's phase list, marks (object lift, movement and tilt at four moments), palm error and release error are
   the facts; the class is a summary of them.
6. Use per-round statistics (the value at the deciding moment), then aggregate across rounds. A time summed over a
   whole round is weighted by whatever the robot waited on.
7. Only closed-loop re-runs from a start are evidence (the Prober's). Reasoning from the code, re-solving grasps or
   inverse kinematics offline, or replaying a recorded command is not.
8. The diagnostic pool is small and measures direction only: you see half of each task's diagnostic starts, the
   driver keeps the other half (rule 47); see rules 41 and 42.
9. Subsets picked because they succeeded fall back towards the mean. Do not tune on the rounds that succeed now.
10. Before saying information is absent, look for it for a few minutes. Before calling a constant decisive, find every
    caller of it.
11. When a list of fixes keeps growing, the layer is probably wrong: say so instead of adding item N+1.

## The robot and the controller (prompt)

12. A Unitree G1 (29 degrees of freedom) with a three-finger Dex3 hand on each wrist. A hand takes one value (0 open .. 1
    closed); the fingers close by joint targets and the hand is not a pinch tool: the seed carries the box between two palms,
    squeezing from the sides. The box is a 21 cm cube standing on a table whose top is 0.745 m; the arms are position
    controlled. The simulator is slow (7 to 10 control steps a second at 50 Hz on four cores), so an episode takes one to three
    minutes.
13. The robot is driven by the native TWIST2 tracker: a learned whole-body policy that follows a reference motion. The package
    does not drive joints directly; it gives a reference and the tracker turns it into leg and waist motion (rule 14). What
    follows from that: the root step alone does not walk (the tracker follows the reference's legs), standing still while the
    arms move is NOT stable under this tracker (with the legs held at the standing pose and only arm targets changing the
    pelvis wandered 1 to 1.5 m and yawed up to 50 degrees within 6 s), and the policy's state reads the first two matrix
    columns of the root rotation (written as rows the robot falls within 28 steps). There is no ScaleBFM here.
14. The action space, native: `env.drive(action)` takes the 40-D reference the tracker follows: the reference root's step (2:
    dx, dy in the heading frame), its height (1), its heading as the first two columns of a rotation matrix (6, rot6d), the 29
    canonical joint positions (29) and the two hand values (2). `env.hold(joints, dxy, yaw, height, hands)` builds a native action
    that keeps the robot as it is; `env.gait` gives the leg joints of a stride (`gait_cycle.json`: the near-straight strides of
    TWIST2's example motions averaged, period 1.31 s, 0.80 m/s), without which the pelvis does not walk or steer. `env.me`
    gives the pelvis, the wrists, the arm Jacobians and `policy_state()`. Every speed, distance and wait of a flow is a key
    of `K` in `skill.py`.
15. A walk with the gait stops within 2 to 3 cm of where it is told to stop and steers; a turn in place is slower. The legs
    hold the standing pose only while the arms are still (rule 13): moving the arms while standing is where the seed's
    episodes fall or wander. The tracker follows the reference with a lag, so a hand's target is reached after a few steps.
16. The flow. `pp_box` ("Pick up the box and place it on the shelf"): the seed settles (30 steps), walks to the table's edge
    (one thigh's depth short, level with the box, facing it), reaches both wrists to the box's two sides (a palm's length behind
    the box's middle, the arms raised by the Jacobian servo), squeezes in a few centimetres and holds the targets, lifts both
    wrists, backs off, turns to the shelf and carries, lowers to the shelf layer's height (the box bottom to 0.648 m: the box
    starts on a table at 0.745 m, so the carry also lowers it by 0.1 m), steps in, opens the arms and hands and backs away. The
    benchmark's check: the box's bottom within 0.06 m of 0.648 m and its centre inside the shelf surface's rectangle. Every
    step is a `mark` of the package (settled, at_table, mid, reached, squeezed, lifted, ...); the marks are the segments of
    rule 48. Measured on the seed (5 practice episodes, 2026-11-24): all reach `at_table`; none places the box; three fall
    while carrying.
17. What the package knows of the scene comes through the truth view: `env.view.goal()` gives the box's pose and extent and the
    shelf layer's height and rectangle in the world frame. This benchmark runs with `target=truth`: the package is told where the
    box and the shelf are. No detector is involved and none is to be built (the front camera exists through `env.cameras` and
    is not needed). The scene (table, box start, shelf) is the same up to the seed's small differences, so a goal value written
    into code as a number is a memorised episode.
## Operations (code where marked)

18. A role's own closed-loop checks go through `scripts/rsi/try.py`: the starts it names (`--labels`, `--task`;
    the call's limit is printed by the tool), one skill package with its `--set`, run through the shared queue, a
    table back (outcome, failure class, first failing phase; against `--ref` the starts that flip). The round budget
    per role and iteration is the driver's (a call past it is refused). Do not run the bed or the scene runner
    yourself, write loops around it, or wait on background work with `until` / `sleep` polls. A round takes about half
    a minute and only two simulator instances run at a time, so a call with many starts queues: choose starts that
    discriminate between causes. The Prober's probes and the gate remain the evidence (rules 3, 7); `try.py` is for
    looking.
19. Long jobs run under `nohup setsid` with a log; a foreground call that times out loses its children.
20. Stop and write an escalation instead of acting when a step needs something locked (rule 1), the controller or its
    training, or the budget. A fall that comes from the controller (the same command alone falls the same way) is a
    limit of the layer below: route around it with another stance or a gentler first move; never patch the layer.
    Rule 45 says how a controller limit is reported.

## Infeasibility (not used on this backend)

21.-23. Infeasibility claims, parks and the `held(X)` predicate belong to the scene backend. Do not file an
    `infeasible` claim; an episode the present skill cannot do is a failure to explain (and, when the controller is
    the limit, a level 3 request, rule 45).
24. A family's new skills (scene backend) do not apply: the skill package here starts from a working seed.

## The gate and the stops (code)

25. The gate's stage 1 takes its starts from every task in proportion to the task's share of the acceptance pool. The
    verdict compares the candidate with the incumbent on the same starts: accepted when wins minus losses reaches the
    gate's net threshold (3 unless the verdict says otherwise) and the paired sign test over wins and losses gives a
    one-sided p of at most 0.05 (5 wins against 0 losses pass, 10 against 3, 12 against 4; 8 against 2 and 6 against 3
    do not). A re-run of the same code may flip some starts both ways (rule 41), which is why the test is on the pairs,
    and the gate's verdict JSON says what it counted. The sealed evaluation at the end decides what the run reports:
    the number of sealed starts won. The candidate's diagnostic rounds and the early review of the next iteration start only once the gate's
    stage 1 has passed, so a call stage 1 rejects costs only stage 1 (code, TODO 48.17).
    No task may lose: when a task's losses exceed its wins by 2 or more, the candidate is rejected
    whatever the other tasks gained (code, TODO 48.19; the gate's `per_task` gives each task's wins and losses). A
    change that helps one task and hurts another is two changes: keep the part that helps, and condition it so it
    stays out of the other task's episodes (rule 38).
26. (scene backend: redraw of the acceptance pool after parks; not used here.)
27. (scene backend: families that start from an empty seed; not used here.)
28. An iteration builds several candidates (K Engineers in parallel, each in its own copy of the skill package), each
    candidate runs on the diagnostic pool, and the Composer merges into one candidate the edits whose diagnostic flips
    show their mechanism, adopts one candidate, or keeps the incumbent. Only that one candidate reaches the gate, so
    each gate call stays one paired test, and gate calls are budgeted: a change that satisfies the benchmark's check
    without doing what the instruction says (rule 40) wastes one. The Composer ties every kept or dropped edit to the
    starts and the trace moment that decided it. The Manager links a mechanism proposed again to its earlier ids; a
    mechanism proposed in two or more iterations and given to no Engineer is built first unless the Manager writes
    which check fails (rule 33). The best rejected candidates wait in the pool and may serve as a `base`. A file
    changed outside the skill copies voids every draft of the iteration. The Manager's `hypotheses` are different ones, preferably on different tasks: the same hypothesis is never given to two Engineers (the driver removes duplicates, may run fewer Engineers, or tops up with an untested or verified hypothesis on another task). The Reviewer's working files go to the Manager, the Engineers and the Composer as review notes with an INDEX.md. (code, prompt)
29. The experience library of earlier families, when the sandbox holds one (`scripts/rsi/experience/INDEX.md`), comes
    from the scene backend. Its numbers come from other robots, objects and grippers; read it for cues, trust this
    family's rounds. (code: sandbox.py)
30. A flow for one kind of task is allowed in the skill package. The skill may branch on `env.task` and on what the truth view's
    goal says (how far the box is, how high the layer is). It may never branch on the episode's id or seed; the guard (rule 44)
    refuses these names. (code)
31. (scene backend: `stage()` retries.) The skill here is a straight line of marks. A step that missed is not followed by the next
    as if it had worked: check the measured state (the wrists' distance to the box sides, whether the box rose with the hands,
    the pelvis's height and drift) before the next mark, and give up or retry only within the step budget. A check reads only
    what `env` offers (rule 40).
32. Every round record carries `failure_class`, computed by the driver from the row (`backend_seeded.failure_class`). The
    classes are error, timeout, fall, comply and short; rule 43 says what each means and what usually helps. A success that the
    compliance check rejects is put in `comply`. The gate reports its tally per class. None of the classes is evidence that a
    change helps; only the gate is. (code)
33. Each hypothesis gets a `status` from its probe: verified (it fixed a diagnostic start, broke none), refuted (it
    fixed none), mixed, or untested (not probed: its mechanism needs code). A mechanism Reviewers proposed in two or more
    iterations, given to no Engineer and not refuted takes one Engineer slot, unless the Manager's `skip` names the
    check that fails: interface_kept, safety_checks_kept, changed_code_reached or evidence_not_refuting. After a
    rejection the Manager reads the failure classes and the gate's per-class tally. A `measure` over the record
    library may set a status when the record library has columns for this family (code). (code)
    Version 29: parameters are searched by the program; the roles name the knobs. A hypothesis may carry `sweep`, at most 2
    items {"key", "values"} (2-5 values each) besides `delta.set`, the fixed values. The program runs each value on the
    starts of the hypothesis's tasks (the second key with the first held at its best), compares each run with the
    incumbent start by start and keeps the best value: most starts fixed minus broken, ties to fewer broken, then to the
    earlier in the list. The probe result carries `sweep` (key, value, n, fixed, broken, net) and `best_set`. Values
    bracket the change the role expects and come from the quantity the value comes from, never from particular
    episodes. A hypothesis whose mechanism needs code (`needs_code`) is not probed: its status is `untested`, which is
    not a refutation. The Manager lists in `tune` the parameter-only hypotheses whose best net is at least 1, and the
    program builds a draft for each (the incumbent's code plus `best_set`) without an Engineer (code, prompt).
    Since TODO 48.17 the probe runs beside the Engineers, after the Manager has decided: the Manager decides on the
    review and earlier probes, the program picks the tuned drafts itself (at most two, best net at least 1, of
    hypotheses no Engineer builds), and the Composer gets the probe's results (code, prompt).
34. (scene backend: preparatory actions.) Not used here.
35. (scene backend: design probes.) Switched off on this backend. There is no stance to probe: the package walks itself.
36. (scene backend: keypoint grasps.) Not used here.
37. What the Reviewer gets of the incumbent's diagnostic rounds is the failure cards: one card per failed round under
    a summary of the pool. To look closer at a round the Reviewer runs it again with try.py. The Prober and the later
    roles read the rounds as before. (code, prompt)
38. A draft says where it acts and the driver measures it. The Engineer returns `scope`: the condition in measured
    quantities and the diagnostic starts it expects to change. Rounds from the same start and the same code are
    close to the same (rule 41), so a round that differs from the one the draft started from in a phase or its timing
    is read as a round the code reached; the driver records that as the draft's `footprint` for the Composer. With
    repair on, a draft that breaks a start the incumbent finishes gets one more turn. (code, prompt)
    Version 29: a draft's diagnostic rounds run on the tasks its declared `scope.starts` belong to, plus a few sentinel
    starts of every other task (starts the incumbent won first); `["all"]` runs the whole pool. Its flips and footprint
    are over those starts (`ran`; `footprint.not_run` counts the rest); the composed candidate runs on the whole pool
    beside the gate. A threshold the new code needs is a parameter key the Engineer lists in `tunable` (at most 3 items
    {"key", "values"}, 2-5 values each); after the diagnostic run the program sweeps them on the declared scope and
    keeps the best (rule 33) (code, prompt).

## What is new on this backend

39. **Two levels of change, cheapest first.** Every proposal says which level it is and what evidence put it there.
    - Level 1: parameters and the skill's own decision code: the walk's speed, ramp and stop, the reference's arm motion (how
      fast the arms move, how they are held while the legs stand), the hands' path to the box, the squeeze depth, the lift and
      carry heights, the order and timing of the marks, what a step checks before the next one.
    - Level 3: the tracker cannot do it (it does not hold the standing pose while the arms move, it does not carry the weight,
      it falls at a speed the gait asks for). Edit nothing. File a request (rule 45). The loop does not train the tracker.
    There is no level 2 on this backend: there are no stance keys (a `stance_` key is refused); where the robot stands is part
    of the flow. A level 1 explanation that has not been tried closed loop does not justify level 3.
    The seed (`humanoidarena/seed/skill.py`) is a start, not a solution: it walks to the table and reaches, and loses the box or
    the balance after that. The loop looks for what keeps the robot upright while the arms work.
40. **A win is HumanoidArena's own check holding inside the step budget, the robot never having fallen, and the hands having
    done it.** The row carries all of it: `success` (compliant), `bench_success` (the benchmark's reward reached 1 before a fall),
    `why`, `hand_steps`, `body_steps`, `body_parts` and `obj_move` (how far the box travelled, m). A round counts as won only
    when `success` holds. The compliance check, from the environment server's contact record (every step on which a robot link
    pressed on the task object with more than 1 N):
    - a hand is a link of the Dex3 hand (a link name containing `hand`: base, palm, finger links); every other link is the body
      (wrist, forearm, elbow, shoulder, torso, pelvis, legs);
    - ok = the benchmark's check and no fall, a hand on the box on at least 5 control steps (0.1 s), the body on it on fewer
      than 3 steps (a graze), and the box moved at least 0.10 m (it did not just start where the check wants it).
    The benchmark's own check needs no contact at all: a shove, a kick or a body push passes it. Here those are `body_push`;
    `why` is "" or one of `not_success`, `fell`, `no_record`, `no_hand`, `body_push`, `not_moved`, `error`. A change that
    satisfies the check without the hands doing it is a loss, even when the benchmark would score it.
41. **A re-run may not repeat exactly.** Isaac's contact and the learned tracker may let the same code from the same start
    finish differently. So: do not argue from one flip. A change is argued from several episodes that flip the same way for the
    same mechanism, with the mark where the mechanism acted; a single flip is noise until it recurs. A round that is borderline
    (the box 2 cm off the layer's height, a fall at the very end) is a reason to look at the margin the code leaves, not at the
    one episode. The gate's verdict says what its sentinel rounds did.
42. **The diagnostic pool is small** (you see half of each task's starts, rule 47; the context says how many). Do not tune a number to the
    individual episodes in it. A parameter value picked because it fixes two specific episodes falls back on the next
    draw (rule 9). Prefer a mechanism stated in measured quantities (the palm error at the grasp, the box's travel
    as the truth view gives it, the time a phase takes) with a value that has a reason, and prefer a change that moves the
    margin on many episodes a little to one that rescues one a lot. Every threshold in new code needs the quantity it
    comes from.
43. **Failure classes on this backend** (from `backend_seeded.failure_class`; one line each: what it means, what usually helps;
    the mechanism, not a setting):
    - `error`: the round did not run (an exception in the package, a crash, the server's error): read the row's `error`. A
      package that raises ends its episode.
    - `timeout`: the episode ran to its 1450 steps without the check holding (the robot standing, the box on the table): find
      the mark the clock went to (rule 48).
    - `fall`: the robot fell: the arms moved while the legs stood (rule 13), a carry too fast, a turn with the box held, a step
      into the shelf; keep the legs stepping (the gait) while the arms work, slow the carry, widen the margin to the shelf.
    - `comply`: the check held and the hands did not do it (`no_hand`, `body_push`, `not_moved`): `body_parts` name the links that
      touched the box; carry it on the palms and keep the forearms and the torso off it.
    - `short`: the episode ended with the check not held and the budget not run out: the last mark reached and what the flow
      did next say where it stopped (the box set down 5 cm short of the layer, off the rectangle).
44. **What an edit may not do (the static guard `humanoidarena/guard.py`; code, applied to the candidate before anything runs;
    an edit that fails it is not run and costs no round).**
    - R1 Only `skill.py` changes. `__init__.py` is byte-identical, no file is added or missing, nothing else is in the package.
    - R2 Every editable file parses. R3 (locked definitions): none in this package.
    - R4 No forbidden name as an identifier, attribute, argument, keyword or imported name: the link and the episode's
      identity (`_ep`, `link`, `ServerLink`, `Episode`, `ask`, `priv`, `Privileged`, `HumanoidArenaPrivileged`, `TruthView`,
      `facts`, `episode_id`, `seed`, `outcome`, `contacts`, `verdict`, `objects`, `obj_move`); the simulator (`isaaclab`,
      `isaacsim`, `omni`, `torch`, `carb`, `sim`, `model`, `data`, `root_state_w`, `body_pos_w`, `contact`, `sensor`, and any
      name starting `isaac`, `omni` or `carb`); files, the environment, processes and code from strings (`os`, `sys`, `pathlib`,
      `shutil`, `pickle`, `json`, `ctypes`, `threading`, `multiprocessing`, `environ`, `getenv`, `open`, `load`, `save`,
      `subprocess`, `socket`, `requests`, `urllib`, `eval`, `exec`, `compile`, `__import__`, `importlib`, `glob`, `Path`,
      `listdir`, `system`, `popen`, `__file__`).
    - R5 No new imports beyond numpy, math, dataclasses, typing, functools, itertools, collections, heapq and scipy.
    - R6 No dynamic attribute access (`getattr`, `setattr`, `vars`, `globals`, `locals`, `__dict__`, ...).
    - R7 The file grows by at most 300 lines.
    - R8 No import of the simulator side of the bed (`locomani.bench.humanoidarena.client`, `server`, `run`, `session`,
      `episodes`, `config`, `guard`, `judge`, `record`, `privileged`, `truth`; `locomani.perception.truth`,
      `locomani.perception.services`).
    What the package gets is the facade `env`: `env.K`, `env.task`, `env.instruction`, `env.t`, `env.view.goal()` (rule 17),
    `env.me` (the robot's own readings and arm solutions), `env.drive(action)`, `env.hold(...)`, `env.gait`, `env.mark(name,
    note)` and `env.cameras`. It never gets the link, a Privileged, the outcome or the contact record.
    A change that needs something the guard refuses is not a level 1 change: it is a locked part, and the Manager treats it as
    outside the loop (rule 20).
45. **Level 3 requests.** When the evidence says the tracker itself cannot do it (it does not hold the standing pose while the
    arms move, it falls at the gait's speed, it does not carry the box's weight), nothing is edited. The Reviewer reports it in
    `notes` with the label `LEVEL 3` and the Manager writes the request in `escalation`, with the probe evidence (the episodes,
    the references given, the pelvis's drift and yaw against the command). The request is for people; the loop goes on with what
    it can do at level 1. A symptom that sits with the bed rather than with the tracker is reported the same way with the label
    `BED`: a reading stays at zero, a joint does not follow its target, the first frame after a reset still shows the last
    episode, a task behaves as no other task with the same motion does. Nothing is edited; the Manager puts it in `escalation`
    and people check the bed's interface (rule 1).
46. **The bed's fixed switches (code).** The family file's `bed_fixed` lists switches that correct the bed's interface to the
    benchmark. This family has none. A new fixed switch is a new version of this file and one human intervention.
47. **The check half (code, TODO 48.17).** Half of each task's diagnostic starts, chosen by a hash of the start's
    name, is kept from the roles in the hidden tree; the pool you see is the other half. A candidate the Composer
    sends first runs on the kept half beside the incumbent, start by start, and goes to the gate only when it wins
    there at least once more than it loses. The ledger's `check` row gives the counts, never the starts. Two
    candidates in a row that do not pass stop the run: what the visible half shows no longer carries to starts the
    roles have not seen (in the runs of 2026-11-20 to 22, late candidates gained 17-35 visible starts net and nothing
    at the gate). Until 2026-11-23 the whole pool was visible; rounds of the kept half in earlier iteration
    directories stay where they were.
    The same per-task rule holds there: a task that loses 2 or more net on the kept half stops the candidate before the
    gate (the `check` row names the task, not the starts).
48. **Segments (code, prompt; TODO 48.17.6).** A task's steps are its segments. Over the incumbent's diagnostic
    rounds the driver counts, per task and step, the rounds that reached the step and the losses blamed on it (the
    cards' first table; `segments` in the roles' contexts). The step that holds most of a task's losses (at least 40 %
    and two rounds) is `hot`; a step at least five rounds reached and none failed at is `stable`. Work where the losses
    are: a hypothesis names the step it acts on, and one that acts before the hot step says why the hot step's cause
    lies there (a grasp chosen at the approach that fails at the lift). A change that alters a stable step risks every
    round that passes through it: the Composer gets, per draft, the steps where its changed rounds first left the
    rounds it started from (`segments_touched`) and keeps a change that starts in a stable step only with the flips
    that show what it buys. The verdict is still the whole episode's (rules 3, 25): a segment is where to look, not
    what is judged.
49. **The sealed check at the end (code, TODO 48.19).** When a stop rule ends the run (not a failed role, a crash or the
    iteration limit), the driver runs the final incumbent and the seed once on the sealed pool, which no role and no
    gate call has seen, in the gate's environment. The final version is confirmed when, start by start, it wins more
    than it loses against the seed with a one-sided sign test p of at most 0.05 and no task loses 2 or more net. The
    numbers stay with people (the hidden tree); the ledger only records that it ran. A version the sealed check does
    not confirm is reported as not confirmed, whatever the gate said along the way: many gate calls in a row each
    risk a lucky acceptance, and the sealed pool is where that shows.
