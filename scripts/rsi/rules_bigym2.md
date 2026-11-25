# Rules for the RSI loop on the BiGym 2.0 benchmark (adapted from rules_lml_bench.md version 9; this file is version 2, 2026-11-25: rules 19, 25, 47, 49 and a version note (TODO 54, review 2026-11-25): no Manager and no Prober in the benchmark iteration, the stop on no progress, the per-task sign test replaces "a task net <= -2 rejects", the check half sends at net 4, the program merges the drafts, rule 19 removed, rule 25 no longer says rounds are bit for bit equal; version 1, 2026-11-24 (version 1 of rules_bigym2.md, the rules for the bench adapter `locomani.bench.bigym2`, TODO 49.7a: the package is `skill.py` with a frozen `__init__.py`; rules 1, 12-17, 30-32, 35, 39-46 name the benchmark's controller, action space, compliance check and guard; the family file is bigym2_reach_drawer))

**Version note (TODO 54, rsi-lean, review 2026-11-25).** The benchmark iteration has no Manager call and no separate Prober phase. After the Reviewer, the program hands the Reviewer's hypotheses, in the Reviewer's order, to up to 3 Engineers, one distinct hypothesis each, with non-overlapping `tasks` (fewer Engineers when fewer are disjoint). The only parameter search is the program's sweep of the Engineers' `tunable` keys, within one budget per iteration. Each draft is screened on its declared starts plus sentinel starts; drafts with a positive screen are merged by the program (3-way merge against the incumbent), and the Composer is called only on a conflict or when no draft is positive. Where a rule below speaks of the Manager (its decision, `tune`, `skip`, `base`, `reopen`) or of the Prober's results (a hypothesis's `status`, `sweep`, `best_set`), read: the program, no such field, no such result. What a role is told changed in rules 19, 25, 47 and 49 (below). Phase 3 of the review rewrites this file; until then this note wins where the text disagrees.

With the family's plan (if its file names one), the only human knowledge the loop reads. The numbering follows
`scripts/rsi/rules.md`, `rules_simple.md` and the LocoManiLab bench rules, because the driver's messages cite rules by number. A rule that does not
apply on this backend keeps its number and says so in one line. Rules 39 to 49 are the ones written for the bench adapters. Every rule says where it is enforced: **code** (the gate, the runner, the guard or the driver refuses) or
**prompt**. A new rule is a new version of this file; it, or a change to the family plan, is committed and counted as
one human intervention.

## The task and its limits (code)

1. The benchmark is frozen: the simulator (MuJoCo), the family's tasks (two in `bigym2_reach_drawer`), their instructions,
   their step budgets, the benchmark's own success check, the compliance check (rule 40), the robot's own controller
   (BiGym's GR00T-WBC lower-body policy), the server's inverse kinematics, the truth view and the failure classes. Making a
   task easier is not an improvement. The Engineer may change only the skill package (`skill.py`, rule 44); any other changed
   file voids the candidate (the driver checks and restores).
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

12. A Unitree G1 with a Dex1 gripper on each wrist: two jaws that close on a pinch centre (the site `left_end_effector`
    / `right_end_effector`). The pads are 6.6 cm long and reach about 3.9 cm past the pinch site, so the pinch site stops about
    4 cm short of a bar the jaws are to close on. Jaws open and close by one command each (0 open .. 1 closed); how hard they
    hold is the command's depth. The arms are position controlled and sag under gravity: a joint target is reached only after
    a few steps, and the hand ends a few millimetres to centimetres below the aim. The seed integrates the measured hand error
    for that, and only once the joint targets have arrived (earlier it overshoots into the thing it reaches for).
13. The robot is driven by BiGym's own controller: the action is 20-D at 50 Hz (rule 14). The lower body is the benchmark's
    GR00T-WBC walking policy: it takes a velocity (forward, sideways, turn) and a pelvis height and keeps the robot upright;
    the arms and grippers take absolute joint targets and gripper commands directly. There is no keypoint tracker and no
    ScaleBFM here, and the package does not choose the controller (rule 1).
14. The action space, native: `env.drive(...)` takes the 20 values (vx, vy, pelvis height, wz, the left arm's 7 joint targets,
    the right arm's 7, the two grippers); the 21-D pitch tasks are mapped by the server and the seed does not use them. The
    seed's flows build their commands with the helpers of `skill.py` and the robot's own inverse kinematics (`env.me.ik`: a
    damped least squares on the arm's 7 joints to a hand target given in the pelvis frame, shoulder roll kept outward
    (>= 0.12 rad) because the shoulder-yaw link otherwise meets the torso). Walking stops below a pelvis height of about 0.66 m
    (0.55 and 0.62 move 3 cm in 2 s, 0.68 and 0.74 move 22 cm), so a crouch to reach low things is 0.68 at the lowest if the
    robot has to walk afterwards. Every speed, distance and wait of a flow is a key of `K` in `skill.py`.
15. Standing, the robot stays; a walk is slow (0.2 m/s in the seed) and stops within a few centimetres of where it is told; a
    walk with the grip closed drags what the hand holds. The robot's start differs with the pool's seed (the spawn pose and, on
    the drawer, the opening); `dishwasher_close`, `wall_cupboard_open` and `wall_cupboard_close` give one start for every seed
    (not in this family).
16. The flows. `reach_target_single` ("Touch the red sphere with your left hand"): a 5 cm sphere half a metre ahead of the
    settled pelvis, just out of the arm's reach: walk forward until the arm solution reaches the sphere, stop, move the pinch
    centre there and hold it (the check wants the pinch centre in the ball for one second). `drawer_top_open` ("Open the top
    drawer of the kitchen cabinet"): the handle is a thin plate (6 mm) standing 2 cm off the drawer face; the robot walks to
    it, crouches so the shoulder is level with it, turns the hand so the jaws open vertically (above and below the bar), puts
    the pinch centre on the bar, closes, walks backwards with the grip closed (the arms fixed in joint space) until the drawer
    is open, and lets go. The check wants the drawer at least 90 % open for one second; the seed's pull ends at 96 %. Every
    step of a flow is a `mark` of the package; the marks are the segments of rule 48.
17. What the package knows of the scene comes through the truth view: `env.view.goal()` gives the reach's sphere {side, point}
    or the drawer's {handle, bar_axis, pull, travel, open_frac}, in the pelvis frame. This benchmark runs with `target=truth`: the
    package is told where the thing is. No detector is involved and none is to be built (the head camera exists through
    `env.cameras` and is not needed). The package knows nothing of the room beyond the goal; the episodes differ in the
    robot's spawn pose and the drawer's opening, so a goal value written into code as a number is a memorised episode.
## Operations (code where marked)

18. A role's own closed-loop checks go through `scripts/rsi/try.py`: the starts it names (`--labels`, `--task`;
    the call's limit is printed by the tool), one skill package with its `--set`, run through the shared queue, a
    table back (outcome, failure class, first failing phase; against `--ref` the starts that flip). The round budget
    per role and iteration is the driver's (a call past it is refused). Do not run the bed or the scene runner
    yourself, write loops around it, or wait on background work with `until` / `sleep` polls. A round takes about half
    a minute and only two simulator instances run at a time, so a call with many starts queues: choose starts that
    discriminate between causes. The Prober's probes and the gate remain the evidence (rules 3, 7); `try.py` is for
    looking.
19. (removed in version bump of TODO 54: a role does not background work, and rule 18 forbids waiting on it; a foreground call that times out loses its children.)
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
    verdict compares the candidate with the incumbent on the same starts: accepted when the paired sign test over wins and losses gives a
    one-sided p of at most 0.05 (the net-win threshold of 3 is not read where p_max is set: no tally of fewer than 5 net wins passes; 5 wins against 0 losses pass, 10 against 3, 12 against 4; 8 against 2 and 6 against 3
    do not). A re-run of the same code may flip some starts both ways (rule 41), which is why the test is on the pairs,
    and the gate's verdict JSON says what it counted. The sealed evaluation at the end decides what the run reports:
    the number of sealed starts won. The candidate's diagnostic rounds and the early review of the next iteration start only once the gate's
    stage 1 has passed, so a call stage 1 rejects costs only stage 1 (code, TODO 48.17).
    A task may not lose significantly: for each task the gate runs the one-sided exact sign test on that task's discordant pairs (its losses among its wins and losses, against a half), Holm-corrected across the tasks, at alpha 0.05 (0.10 at the check half, rule 47); a task with a significant loss rejects the candidate whatever the other tasks gained. Small losses on a task no longer reject unless they are significant: 2 losses to 0 wins on one task of four is within the noise, 7 to 0 is not (code, TODO 54; the verdict's `per_task` and `task_tests` give the counts and the p values). A
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
30. A flow for one kind of task is allowed in the skill package. The skill may branch on `env.task` (the benchmark's task name)
    and on what the truth view's goal says (a sphere or a handle, how far it is). It may never branch on the episode's id or
    seed; the guard (rule 44) refuses these names. (code)
31. (scene backend: `stage()` retries.) The skill here is a straight line of marks. A step that missed is not followed by the
    next as if it had worked: check the measured state (the hand error after a reach, whether the jaws stopped on the bar,
    whether the drawer moved) before the next mark, and give up or retry only within the task's step budget. A check reads
    only what `env` offers (rule 40).
32. Every round record carries `failure_class`, computed by the driver from the row (`backend_seeded.failure_class`). The
    classes are error, timeout, fall, comply and short; rule 43 says what each means and what usually helps. A success that
    the compliance check rejects is put in `comply`. The gate reports its tally per class. None of the classes is evidence that
    a change helps; only the gate is. (code)
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
    - Level 1: parameters and the skill's own decision code: speeds, waits, tolerances, the hand's path (where it enters from,
      how it leaves), the crouch height, the order and timing of the marks, the pull speed, how the jaws are closed, what a
      step checks before the next one.
    - Level 3: the controller cannot do it (the walking policy does not walk below a pelvis height, does not hold a posture
      under a pull, the arm's joint limits). Edit nothing. File a request (rule 45). The loop does not train the controller.
    There is no level 2 on this backend: the robot has no stance keys (a `stance_` key is refused); where it stands is part of
    the flow. A level 1 explanation that has not been tried closed loop does not justify level 3.
    The seed (`bigym2/seed/skill.py`) is a working start: it solves both tasks on its practice episodes, so the loop looks
    for the starts where it does not (a re-grasp that takes 990 steps, a drawer that stops short of 90 %).
40. **A win is BiGym's own check holding inside the task's step budget, the robot never having fallen, and the hands having
    done it.** The row carries all of it: `success` (compliant), `bench_success` (BiGym's own verdict: the task reported success
    and the robot never fell; bigym/loco/eval/protocol.py), `why`, `hand_steps`, `body_steps` and `body_parts`. A round counts
    as won only when `success` holds. The compliance check, from the environment server's contact record (every robot body
    touching a scene body that has joints of its own: a drawer, a door, a free object, per control step; no floor, no
    robot-robot pairs):
    - a hand is a body of the Dex1 gripper (a body name containing `dex1`: the base link and the two finger links); every other
      robot body is the body (wrist, forearm, elbow, shoulder, torso, pelvis, legs);
    - the reach task needs no contact: `success` is the benchmark's check and no fall;
    - the drawer needs a hand on the drawer on at least 5 control steps (0.1 s) and the body on it on fewer than 3 steps (a
      graze, 0.06 s). A knee, a hip or a forearm that pushes the drawer passes the benchmark's check and is a loss here:
      `body_push`; a drawer opened without a hand touching it is `no_hand`.
    `why` is "" or one of `not_success`, `fell`, `no_hand`, `body_push`, `error`. A change that satisfies the check without the
    hands doing it is a loss, even when the benchmark would score it.
41. **A re-run may not repeat exactly.** Rounds from the same code on the same practice episode are close to deterministic in
    this simulator, but a different step count in a walk or a re-grasp (the seed's drawer once took 990 steps instead of 650)
    shows that contact can still differ. So: do not argue from one flip. A change is argued from several episodes that flip the
    same way for the same mechanism, with the mark where the mechanism acted; a single flip is noise until it recurs. A round
    that is borderline (the drawer 88 % open at the end, a hand 0.5 cm outside the sphere) is a reason to look at the margin the
    code leaves, not at the one episode. The gate's verdict says what its sentinel rounds did.
42. **The diagnostic pool is small** (you see half of each task's starts, rule 47; the context says how many). Do not tune a number to the
    individual episodes in it. A parameter value picked because it fixes two specific episodes falls back on the next
    draw (rule 9). Prefer a mechanism stated in measured quantities (the palm error at the grasp, the drawer's travel
    as the truth view gives it, the time a phase takes) with a value that has a reason, and prefer a change that moves the
    margin on many episodes a little to one that rescues one a lot. Every threshold in new code needs the quantity it
    comes from.
43. **Failure classes on this backend** (from `backend_seeded.failure_class`; one line each: what it means, what usually helps;
    the mechanism, not a setting):
    - `error`: the round did not run (an exception in the package, a crash, the server's error): read the row's `error`. A
      package that raises ends its episode.
    - `timeout`: the episode ran to its step budget without the check holding: find the mark the clock went to (rule 48) and
      shorten what buys nothing; a retry that cannot finish in the budget left is a failure that costs the same.
    - `fall`: the robot fell: a walk too fast for the crouch, a pull that outweighs the walk, a hand pushing against the drawer
      while the legs move; lower the speeds, keep the pelvis higher, release before the legs run out of margin.
    - `comply`: the check held and the hands did not do it (`no_hand`, `body_push`): the row's `body_parts` name the robot
      links that touched the drawer; bring the gripper (not the wrist or forearm) to the handle and keep the arm off the
      drawer face.
    - `short`: the episode ended with the check not held and no budget run out (the drawer open 80 %, the hand outside the
      sphere): the last mark reached and what the flow did next say where it stopped.
44. **What an edit may not do (the static guard `bigym2/guard.py`; code, applied to the candidate before anything runs; an
    edit that fails it is not run and costs no round).**
    - R1 Only `skill.py` changes. `__init__.py` is byte-identical, no file is added or missing, nothing else is in the package.
    - R2 Every editable file parses. R3 (locked definitions): none in this package.
    - R4 No forbidden name as an identifier, attribute, argument, keyword or imported name: the link and the episode's
      identity (`_ep`, `link`, `ServerLink`, `Episode`, `ask`, `priv`, `Privileged`, `BiGymPrivileged`, `TruthView`, `facts`,
      `joints`, `episode_id`, `seed`, `outcome`, `contacts`, `verdict`); the simulator (`mujoco`, `bigym`, `sim`, `model`, `data`,
      `xpos`, `xmat`, `qpos`, `qvel`, `contact`, `ncon`, `geom_xpos`, `site_xpos`, and any name starting `mj_` or `mju_`); files,
      the environment, processes and code from strings (`os`, `sys`, `pathlib`, `shutil`, `pickle`, `json`, `ctypes`,
      `threading`, `multiprocessing`, `environ`, `getenv`, `open`, `load`, `save`, `subprocess`, `socket`, `requests`,
      `urllib`, `eval`, `exec`, `compile`, `__import__`, `importlib`, `glob`, `Path`, `listdir`, `system`, `popen`, `__file__`).
    - R5 No new imports beyond numpy, math, dataclasses, typing, functools, itertools, collections, heapq and scipy.
    - R6 No dynamic attribute access (`getattr`, `setattr`, `vars`, `globals`, `locals`, `__dict__`, ...).
    - R7 The file grows by at most 300 lines.
    - R8 No import of the simulator side of the bed (`locomani.bench.bigym2.client`, `server`, `run`, `session`, `episodes`,
      `config`, `guard`, `judge`, `record`, `privileged`, `truth`; `locomani.perception.truth`, `locomani.perception.services`).
    What the package gets is the facade `env`: `env.K` (the parameters), `env.task`, `env.instruction`, `env.t` (seconds),
    `env.view.goal()` (the truth view of rule 17), `env.me` (the robot's own readings: `pelvis`, `hand`, `hand_R`, `arm`,
    `tilt`, and the arm solution `ik`), `env.drive(...)` (one control step with a 20-D command), `env.mark(name, note)` and
    `env.cameras`. It never gets the link, a Privileged, the outcome or the contact record.
    A change that needs something the guard refuses is not a level 1 change: it is a locked part, and the Manager treats it as
    outside the loop (rule 20).
45. **Level 3 requests.** When the evidence says the controller itself cannot do it (a pose the walking policy will not hold, a
    pelvis height below which it does not walk, an arm that does not reach a hand target in its joint limits), nothing is edited.
    The Reviewer reports it in `notes` with the label `LEVEL 3` and the Manager writes the request in `escalation`, with the
    probe evidence (the episodes, the commands, the quantities). The request is for people; the loop goes on with what it can do
    at level 1. A symptom that sits with the bed rather than with the controller is reported the same way with the label `BED`: a
    reading stays at zero, a joint does not follow its target, a task behaves as no other task with the same motion does.
    Nothing is edited; the Manager puts it in `escalation` and people check the bed's interface (rule 1).
46. **The bed's fixed switches (code).** The family file's `bed_fixed` lists switches that correct the bed's interface to the
    benchmark. This family has none. A new fixed switch is a new version of this file and one human intervention.
47. **The check half (code, TODO 48.17).** Half of each task's diagnostic starts, chosen by a hash of the start's
    name, is kept from the roles in the hidden tree; the pool you see is the other half. A candidate the Composer
    sends first runs on the kept half beside the incumbent, start by start, and goes to the gate only when it wins there by at least `check_min_net` (4) and the per-task test below does not reject it. The ledger's `check` row gives the counts, never the starts. Two
    iterations in a row without a candidate that passes it stop the run (an iteration whose Composer kept the incumbent counts; a family without a check half counts a candidate that passes the gate's stage 1; caps `stop_no_progress`): what the visible half shows no longer carries to starts the
    roles have not seen (in the runs of 2026-11-20 to 22, late candidates gained 17-35 visible starts net and nothing
    at the gate). Until 2026-11-23 the whole pool was visible; rounds of the kept half in earlier iteration
    directories stay where they were.
    The per-task test of rule 25 holds there too, at alpha 0.10 (the `check` row names the tasks, not the starts).
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
49. **The sealed check at the end (code, TODO 48.19).** Whenever the run ends
    (a stop rule, the iteration limit, the wall clock, a failed role; not a crash), the driver runs the final incumbent and the seed once on the sealed pool, which no role and no
    gate call has seen, in the gate's environment. The final version is confirmed when, start by start, it wins more
    than it loses against the seed with a one-sided sign test p of at most 0.05 and no task's loss is significant (the per-task test of rule 25, alpha 0.05). The
    numbers stay with people (the hidden tree); the ledger only records that it ran. A version the sealed check does
    not confirm is reported as not confirmed, whatever the gate said along the way: many gate calls in a row each
    risk a lucky acceptance, and the sealed pool is where that shows.
