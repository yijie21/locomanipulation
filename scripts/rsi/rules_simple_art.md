# Rules for the RSI loop on SIMPLE's articulated tasks (family simple_art; this file is version 5, 2026-11-25 (version 5: rule 25 says that the diagnostic, probe and draft rounds start from saved states of earlier rounds (TODO 53: `bench.memo`, the states are on disk in the loop's scratch area), so a round no longer re-simulates the beginning another round already ran; the gate and the sealed evaluation never do; version 4: rule 25 as rules_simple.md version 10, every round renders only at the robot's looks after 48 warm-up renders, the gate and the sealed evaluation included (TODO 52); version 3: the loop runs on the bench adapter `locomani.bench.simple` (TODO 49.6a); the package is one folder skill.py, pick.py, camera_view.py, handover.py with a frozen __init__.py, rules 1 and 44 name those files, the imports a draft may use and what it receives: a view, a body and an env facade, never the simulator's bed or the truth side; version 2: rule 30 allows the task's name, rule 24 gives the seed's development numbers), adapted from rules_simple.md version 10 (rules 25, 47, 49 as there): rules 1, 2, 12, 14, 16, 17, 24, 30, 31, 32, 39, 40, 43, 44, 45 and 46 rewritten for the four articulated tasks; the other rules are rules_simple.md's)

With the family's plan (if its file names one), the only human knowledge the loop reads. The numbering follows
`scripts/rsi/rules.md`, because the driver's messages cite rules by number. A rule that does not apply on this backend
keeps its number and says so in one line. Rules 39 to 48 are new. Every rule says where it is enforced: **code** (the
gate, the runner, the guard or the driver refuses) or **prompt**. A new rule is a new version of this file; it, or a
change to the family plan, is committed and counted as one human intervention.

## The task and its limits (code)

1. The benchmark is frozen: the simulator, the four tasks, their instructions, their time limits, the environment's
   success check, the compliance check (rule 40), the failure classes, the controller (ScaleBFM), the motor layer, the
   frame bridge `Belief` and the truth view (all fixed modules of locomani, outside the package: the package never
   contains them). Making a task easier is not an improvement. The Engineer may change only
   the skill package (the files the static guard lists as editable, rule 44); any other changed file voids the
   candidate (the driver checks and restores).
2. A start is never removed from a pool. A start the skill refuses ("no target seen", "no part seen") still counts as a
   failure. A round that has not succeeded when the task's step limit passes is a failure (rule 39 says what the limit
   is). The diagnostic pool is not redrawn on this backend. (code)
3. A candidate is accepted only by `gate.py`, in the official environment, on the hidden acceptance pool. Nothing else
   counts as evidence that a change helps. The gate's per-round files are off limits; its verdict JSON is what a role
   may read. While a role runs the hidden tree cannot be read. A role whose tool calls name the hidden tree, the
   benchmark's released evaluation episodes or their results, or a path outside the sandbox voids what it produced --
   its iteration, or only its draft when it is one of several Engineers -- and the loop goes on. A Reviewer, Manager or
   Composer whose output is voided runs once more in the same iteration, told why; only a second voided output
   voids the iteration (TODO 48.17). Two voided iterations in a row stop the loop. (code)
4. A candidate is the skill package plus parameter overrides (`candidate.json` {"set": ...}). An unknown key is
   refused. The bed's fixed switches (rule 46) are laid over every set and may not appear in one. (code)

## Evidence (prompt; code where marked)

5. Read the per-round records, not the failure strings. A failure string names where the code gave up, not the cause.
   The record's phase list, palm errors, the hinge's angle (`art`: start, furthest toward the goal, end, needed) and the
   hand's contact with the moving part (`art_touch`) are the facts; the class is a summary of them.
6. Use per-round statistics (the value at the deciding moment), then aggregate across rounds. A time summed over a
   whole round is weighted by whatever the robot waited on.
7. Only closed-loop re-runs from a start are evidence (the Prober's). Reasoning from the code, re-solving reaches or
   inverse kinematics offline, or replaying a recorded command is not.
8. The diagnostic pool is small and measures direction only: you see 15 of each task's 30 episodes, the driver keeps
   the other 15 (rule 47); see rules 41 and 42.
9. Subsets picked because they succeeded fall back towards the mean. Do not tune on the rounds that succeed now.
10. Before saying information is absent, look for it for a few minutes. Before calling a constant decisive, find every
    caller of it.
11. When a list of fixes keeps growing, the layer is probably wrong: say so instead of adding item N+1.

## The robot and the controller (prompt)

12. A Unitree G1 with a three-finger Dex3 hand on each arm, the model ScaleBFM was trained on (SIMPLE's g1_sonic), in all
    four tasks. Fingers close by joint targets; how hard they hold is how far past the contact the joints are told to
    go, and SIMPLE drives each finger joint up to 0.7 N m. The objects are hinged assets (PartNet-Mobility); each hinge
    has damping and dry friction (from the asset files):
    - OpenOvenTeleop: a double wall oven with two glass doors, each on a hinge of 0 to 90 deg, damping 1, dry friction
      4 N m on one door and 2 N m on the other: a door does not move until it is pushed or pulled harder than that.
    - CloseDoorTeleop: a room door, hinge -10 to 80 deg, damping 2, dry friction 1 N m. It starts open at a random
      angle between 0 and 80 deg.
    - OpenFaucetTeleop: a faucet lever that turns up to 90 deg either way, damping 0.2, friction 0.3 N m; a small part.
    - OpenTrashCanTeleop: a trash can with a hinged lid, 0 to 90 deg.
    Every task also places an apple and a few other objects nearby; they play no part in the task. Where each hinge's
    axis lies and which way each part swings is not written here: the camera shows it, and a probe measures it.
13. ScaleBFM follows keypoint targets (pelvis, wrists, ankles). It crouches and leans when the hands ask for it, and
    the waist is its own choice. The motor layer's reach servo brings a hand to about 1 cm of a target after it has
    settled; during a fast entry the hand lags the command by several centimetres.
14. Open loop, ScaleBFM's hands land 2-9 cm and 10-30 deg off a target pose. Reaches to a small part are therefore
    stepped (in front of the part, then onto it) and slow near it; each step waits for the palm to come near the
    target, and every wait costs time on a clock that is short (rule 39). A part on a hinge moves only along its arc:
    a hand that holds it and moves along another path pulls against the hinge, and the grip slips or the arm is
    dragged off its path.
15. Standing, the feet slide apart; a walk shuffles. A straight walk stops within a few centimetres, a turn in place
    within a few degrees. The walk's end point is a stance, and the hand's reach starts from wherever the walk ended.
16. The robot walks to a stance in front of the object, reaches the right hand to the part, closes the fingers on it
    (or sets the palm against it for a push), moves the part, lets go and takes the hand away. The seed's steps are
    named walk, reach, close, move and release (rule 48's segments). Speeds, waits, tolerances and offsets of every
    step, and the path of the move, are parameters of the skill (the context's `parameters`, with the skill file's
    comments).
17. The part is found in the head camera's images: a segmentation model (SAM 3) is prompted with words for the part,
    and depth comes from the stereo pair. The camera sees the object from where the robot stands: a handle is small, a
    glass door shows what is behind it, a door leaf seen edge-on is a thin strip, and the arm hides the part once the
    hand is on it. Where the hinge is and which way the part swings is not given: the skill infers it from what it
    sees (the part's outline, the frame around it) or from how the hand moves when it pushes or pulls (the robot's own
    joint readings), never from the simulator.

## Operations (code where marked)

18. A role's own closed-loop checks go through `scripts/rsi/try.py`: the starts it names (`--labels`, `--task`;
    the call's limit is printed by the tool), one skill package with its `--set`, run in parallel through the shared
    queue, a table back (outcome, failure class, first failing phase; against `--ref` the starts that flip). The round
    budget per role and iteration is the driver's (a call past it is refused). Do not run the bed or the scene runner
    yourself, write loops around it, or wait on background work with `until` / `sleep` polls. A round is slow (a
    minute or more in the official environment), so choose starts that discriminate between causes. The Prober's
    probes and the gate remain the evidence (rules 3, 7); `try.py` is for looking.
19. Long jobs run under `nohup setsid` with a log; a foreground call that times out loses its children.
20. Stop and write an escalation instead of acting when a step needs something locked (rule 1), the controller or its
    training, or the budget. A fall that comes from the controller (the same command alone falls the same way) is a
    limit of the layer below: route around it with another stance or a gentler first move; never patch the layer.
    Rule 45 says how a controller limit is reported.

## Infeasibility (not used on this backend)

21.-23. Infeasibility claims, parks and the `held(X)` predicate belong to the scene backend. Do not file an
    `infeasible` claim; an episode the present skill cannot do is a failure to explain (and, when the controller is
    the limit, a level 3 request, rule 45).
24. A family's new skills (scene backend) do not apply. The skill package here starts from a seed that runs every
    step on all four tasks. Its author set it up on development episodes, which are in none of the loop's pools: there
    it won 23 of 30 (oven), 13 of 30 (door), 9 of 30 (faucet) and 22 of 30 (trash can) by the compliance rule.

## The gate and the stops (code)

25. The gate's stage 1 takes its starts from every task in proportion to the task's share of the acceptance pool. The
    verdict compares the candidate with the incumbent on the same starts: accepted when wins minus losses reaches the
    gate's net threshold (3 unless the verdict says otherwise) and the paired sign test over wins and losses gives a
    one-sided p of at most 0.05 (5 wins against 0 losses pass, 10 against 3, 12 against 4; 8 against 2 and 6 against 3 do not). The
    official environment does not repeat exactly (rule 41): the same code re-run flips some starts both ways, which
    is why the test is on the pairs, and the gate's verdict JSON says what it counted. The sealed
    evaluation at the end decides what the run reports. The candidate's diagnostic rounds and the early review of the next iteration start only once the gate's
    stage 1 has passed, so a call stage 1 rejects costs only stage 1 (code, TODO 48.17).
    No task may lose: when a task's losses exceed its wins by 2 or more, the candidate is rejected
    whatever the other tasks gained (code, TODO 48.19; the gate's `per_task` gives each task's wins and losses). A
    change that helps one task and hurts another is two changes: keep the part that helps, and condition it so it
    stays out of the other task's episodes (rule 38).
    Every round renders the cameras only at the robot's looks, each look after 48 warm-up renders of the same state (the
    renderer's auto-exposure settles; one render without them sees another image, brighter by about 5 %, and on the faucet
    SAM 3 can miss the object), the gate and the sealed evaluation included (bench block `render`, default "lazy"; the
    family's `bed_env_gate` is the esc bed's). On the 60 diagnostic starts of the four articulated tasks this gave the same
    success rate (83 % against 78 % rendering every step, Fisher p 0.33) and per-start agreement with every-step runs (0.85 against 0.92 between two
    every-step runs). The diagnostic, probe and draft rounds (not the gate, not the sealed evaluation) also start from saved
    states: a round whose first commands to the motor (walk_to, turn, sidestep, posture, reach, adopt_hand, grip, release,
    squeeze, hold, with the same arguments) are those of an earlier round of the same start restores the simulation as that
    round left it, instead of running it again, and its looks at that state show it the images the earlier round saw (the
    renderer is not repeatable, so a re-run would see other pixels and could place the stance a hair differently). The
    skill's Python between the commands runs as always, so a change that comes after the shared commands (the grasp, the
    arc, the speed) changes the result exactly as a full run would, and a change before them (the approach, the look, the
    stance) shares nothing. Two rounds of one setting on one start are therefore the same round, bit for bit (code,
    TODO 52 and 53; the store is in the loop's scratch area, outside the sandbox and the hidden tree: a role cannot read it).
26. (scene backend: redraw of the acceptance pool after parks; not used here.)
27. (scene backend: families that start from an empty seed; not used here.)
28. An iteration builds several candidates (K Engineers in parallel, each in its own copy of the skill package), each
    candidate runs on the diagnostic pool, and the Composer merges into one candidate the edits whose diagnostic flips
    show their mechanism, adopts one candidate, or keeps the incumbent. Only that one candidate reaches the gate, so
    each gate call stays one paired test, and gate calls are budgeted: a change that satisfies the environment's check
    without doing what the instruction says (rule 40) wastes one. The Composer ties every kept or dropped edit to the
    starts and the trace moment that decided it. The Manager links a mechanism proposed again to its earlier ids; a
    mechanism proposed in two or more iterations and given to no Engineer is built first unless the Manager writes
    which check fails (rule 33). The best rejected candidates wait in the pool and may serve as a `base`. A file
    changed outside the skill copies voids every draft of the iteration. The Manager's `hypotheses` are different ones, preferably on different tasks: the same hypothesis is never given to two Engineers (the driver removes duplicates, may run fewer Engineers, or tops up with an untested or verified hypothesis on another task). The Reviewer's working files go to the Manager, the Engineers and the Composer as review notes with an INDEX.md. (code, prompt)
29. The experience library of earlier families, when the sandbox holds one (`scripts/rsi/experience/INDEX.md`), comes
    from the scene backend. Its numbers come from other robots, objects and grippers; read it for cues, trust this
    family's rounds. (code: sandbox.py)
30. The skill may branch on the task's name or the words of its instruction (each task has one fixed instruction, so
    the two say the same thing: the object and what to do with it -- open, close, turn) to choose that task's
    parameters and flow, as the seed's `ART_TASK` does, and on what the camera measures (the part's size, shape, height
    and distance). It may not branch on the episode, its seed or the benchmark's level, and it may not hold positions
    or angles remembered from rounds: the guard (rule 44) refuses the identity and the seed, review refuses remembered
    layouts. (code, prompt)
31. (scene backend: `stage()` retries.) The skill here is a straight line of phases. A step that missed is not followed
    by the next as if it had worked: check the measured state (the palm error, whether the fingers stopped on
    something, whether the hand moved when it pulled or pushed) before the next phase, and give up or retry only
    within the task's time limit. A check reads only what the robot may use at run time (rule 40: images, a
    segmentation, joint readings, odometry).
32. Every round record carries `failure_class`, computed by the driver from the record (loss_simple.py). The classes
    are error, timeout, fall, perceive, collide, deliver, grasp, short and comply; rule 43 says what each means and what
    usually helps. The gate reports its tally per class. None of the classes is evidence that a change helps; only the
    gate is. (code)
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
35. (scene backend: design probes.) Switched off on this backend. To find out whether a stance matters, run the
    incumbent with the stance parameters set to a few values on the same starts through a hypothesis' `delta` (rule
    39's level 2).
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

39. **Three levels of change, cheapest first.** Every proposal says which level it is and what evidence put it there.
    - Level 1: parameters and the skill's own decision code: speeds, waits, tolerances, the words the segmentation is
      asked for, where on the part the hand goes and how it closes, the path of the move (its direction, the arc and
      its radius, how far, how fast), what the skill does with the camera's points.
    - Level 2: where and how the robot stands: the stand-off distance from the target or the table edge, a sideways
      shift, the heading at the stance, and the same for the placing stance. Move up to level 2 only when a probe
      shows that no level 1 setting wins these episodes while another stance does (the same starts, the incumbent and
      the stance values, closed loop, at least three episodes that flip, rule 41). A stance is tried through the
      `set` keys `stance_d` (m, added to the stand-off), `stance_t` (m, a shift to the robot's left along the
      table's edge), `stance_yaw` (deg, added to the heading) and `stance_here` (true: no walk); a stance that holds is then built into the skill's
      own choice of where to stand, from what the camera measures. A stance is not a cure for a grip
      the hand cannot take or a perception error: show that the palm error or the reach moved with the stance.
    - Level 3: the controller cannot do it, for example the hand cannot reach or hold the pose from any stance the
      probe tried. Edit nothing. File a request (rule 45). The loop does not train the controller by itself.
    A level 1 explanation that has not been tried closed loop does not justify level 2; a level 2 stance sweep that
    has not been tried does not justify level 3.
40. **A win is the environment's check holding inside the time limit and the robot having done what the instruction
    says.** The record carries both: `success` (the environment's own check, within the limit) and `comply` (did the
    robot do what the instruction says). A round counts as won only when both hold. The environment's checks read one
    hinge angle and nothing else, so a shove with the body, a fall into the door or a kick would pass them; none of
    that is a win. For all four tasks, from the start until the check first holds:
    - no part of the robot outside the two hands (the wrist's last link, the palm and the fingers) touches a moving
      part of the object (`comply.why` `body_touch`);
    - a hand touched a moving part, and the part moved at least 5 deg toward the goal while a hand was on it
      (`no_hand`).
    A part that swings on after the hand pushed it and let go is fine: a person pushes a door and lets it swing.
    What each task asks:
    - OpenOvenTeleop (limit 20 s): "move forward to the oven and open it". One of the two doors opens to at least 80
      of its 90 deg, held for 9 steps.
    - CloseDoorTeleop (limit 9 s): "move forward to the door and close it". The door is pushed shut against its stop
      (past -9 deg), held for 9 steps.
    - OpenFaucetTeleop (limit 20 s): "turn the faucet". The lever turns at least 40 deg either way, held for 4 steps.
    - OpenTrashCanTeleop (limit 16 s): "move forward to the trash can and open it". The lid opens at least 29 deg, held
      for 15 steps.
    What the robot may use at run time: the head's stereo colour images (depth computed from them), a segmentation of
    what the skill asks for, the robot's joint readings and leg odometry. Never the simulator's object poses, hinge
    angles or contacts, and never the identity of the episode or its seed. The guard (rule 44) refuses code that
    reaches for them. The record's `art` and `art_touch` (rule 5) are the judge's, for the cards: the skill never sees
    them.
41. **The official environment is not exactly repeatable.** The same code from the same start can finish differently
    on a second run (small differences in contact and in the camera's depth). So: do not argue from one flip. A
    change is argued from several episodes that flip the same way for the same mechanism, with the trace moment where
    the mechanism acted; a single flip is noise until it recurs. A round that is borderline (a failure by a few
    hundredths of a second, a hinge a few degrees short) is a reason to look at the margin the code leaves, not at the
    one episode. The gate's sentinel rounds may differ between runs; the verdict says what they did.
42. **The diagnostic pool is small** (15 episodes per task that you see, rule 47). Do not tune a number to the individual episodes in it.
    A parameter value picked because it fixes two specific episodes falls back on the next draw (rule 9). Prefer a
    mechanism stated in measured quantities (the palm error at the reach, the part's size as the camera sees it,
    the time a phase takes) with a value that has a reason, and prefer a change that moves the margin on many
    episodes a little to one that rescues one a lot. Every threshold in new code needs the quantity it comes from.
43. **Failure classes on this backend** (from `loss_simple.py`, first match wins; one line each: what it means, what
    usually helps; the mechanism, not a setting):
    - `error`: the round did not run (an exception, a crash). Read the traceback; it is a bug in the code or the
      environment, not a robot failure. Fix the code, never the number.
    - `timeout`: the environment's check passed, but after the task's step limit. Find which phases dominate the clock
      and shorten the ones that buy nothing (a wait for a settled hand that is already settled, a speed lower than the
      arm needs), never skip a check that the next phase relies on. CloseDoorTeleop has 9 s in all.
    - `fall`: the robot fell. Usually a walk with an arm out, a large lean from a reach too far from the stance, or a
      pull or push that drags the body. Stand nearer, keep the hands in before walking, move the part more slowly.
    - `perceive`: the object or its part was not found ("no target seen", "no part seen"). Look at what the image
      showed, which words the segmentation was asked for and where the head looked from; the remedy is a better look or
      better words, not a guessed position.
    - `collide`: a robot link outside the hands touched a moving part (the body walked into the door, a knee met an
      oven door swinging toward the robot, a forearm leaned on the lid). Change the stance, the arm's path, or where
      the body is while the part moves.
    - `deliver`: no hand touched a moving part. Distinguish a reach the controller did not deliver (palm error large
      though the command was fine: a nearer stance, a slower entry) from a part the camera located wrongly (the palm
      arrived where the cloud said) from a hand that stopped on the frame or the fixed body.
    - `grasp`: a hand touched the moving part, but it moved less than 3 deg toward the goal: the hand slipped off,
      pushed against the hinge or the wrong way, or held too weakly against the hinge's friction (rule 12).
    - `short`: the part moved toward the goal but not as far as the check needs, or the check held and the hand did
      not make the move (rule 40): the move ended too early, the hand lost the part on the arc, the path left the arc
      (rule 14), or time ran out.
    - `comply`: the check held but the compliance did not, and none of the classes above names why. Read
      `comply.why`.
44. **What an edit may not do (the static guard; code, applied to the candidate before anything runs; an edit that
    fails it is not run and costs no round).**
    - The package is one folder. Only the files the guard lists as editable change: skill.py, pick.py and
      camera_view.py (a candidate may also add handover.py, which is checked against an empty file: it may hold none
      of the refused names at all). No other file may appear in the package and `__init__.py` (the package's
      parameters and `apply_set`) stays byte for byte as it is. A file may grow by at most about 400 lines.
    - The truth view, the frame bridge `Belief`, the helper that reads body vertices and the grasp library are not in
      the package at all (they are fixed modules of locomani); a locked name that a package still defines stays
      exactly as it is.
    - What a draft receives: a view (the head camera's `camera_view.CameraView`), a body and an environment facade
      (`env`). It never receives the simulator's bed or its truth side. It may import, besides numpy and the standard
      helpers, only relative modules of its own folder and the public modules the seed already imports:
      `locomani.perception.camera_tools` and `locomani.perception.head`, `locomani.bench.simple.body`,
      `locomani.backends.scalebfm` (runtime, motor). It may not import the adapter's run, client, server, episodes,
      session, feed, config, guard, judge or privileged modules, nor `locomani.perception` truth, rgbd, noise,
      services, odometry or grasps (the guard's rule R8: the simulator side of the bed).
    - No new reads of simulator state: names that reach the physics data (positions, orientations, velocities,
      contacts of bodies or geoms, anything of the simulator library) may not appear more often than before.
    - No branching on the identity of the episode or on the seed, and no use of the benchmark's evaluation hooks or
      the process environment.
    - No file or network access: no opening, loading, listing or globbing files, no subprocess, socket or web call.
    - No dynamic access that goes round the name check: no getattr, setattr, vars, globals, locals, no `eval`,
      `exec`, `compile` or dynamic import.
    - No new top-level module beyond those the base already imports and numpy and the standard helpers math,
      dataclasses, typing, functools, itertools and collections (a new import anywhere in a file counts, function
      bodies too).
    A change that needs something the guard refuses is not a level 1 or 2 change: it is a locked part, and the Manager
    treats it as outside the loop (rule 20).
45. **Level 3 requests.** When the evidence says the controller itself cannot do it from any stance (a hand that does
    not reach or hold a pose, a wrist attitude it does not deliver, a pose it will not hold under the object's
    weight), nothing is edited. The Reviewer reports it in `notes` with the label `LEVEL 3` and the Manager writes the
    request in `escalation`: it names the hand-target region where data for the controller should be collected (the
    palm's position in the pelvis frame, its attitude, and the pelvis height, as ranges) and gives the probe evidence
    (the episodes, the stances tried, the palm errors against the command). The request is for people; the loop goes
    on with what it can do at levels 1 and 2.
    A symptom that sits with one robot model or one part of the bed rather than with the controller is reported the
    same way with the label `BED`: a reading stays at zero, a joint does not
    follow its target, the hand lags a command the other model follows. Nothing is edited; the Manager puts it in
    `escalation` with the rounds and the quantities that show it, and people check the bed's interface. The loop does
    not change the bed (rule 1); what people correct becomes a fixed switch (rule 46).
46. **The bed's fixed switches (code).** The family file's `bed_fixed` lists switches that correct the bed's interface
    to the benchmark: the bed sends the robot what the benchmark's robot takes, or reads what the robot really senses.
    People found them, by hand, outside the loop. They are on in every round of every role (diagnostic rounds,
    probes, try.py, drafts, the gate, the sealed evaluation) and in the exam, and they are part of the bed's signature,
    so no round of a bed without them is read again. A set that names one is refused. Now on:
    - `feed_torque`: leg odometry reads the feet's loads from the robot's own torque command. Without it the loads read
      zero in the official environment, and a foot that moves during a reach is read as half a pelvis move (by hand,
      the release error of PickBetweenTables went from about 20 cm to 1 cm).
    This family starts with it on. A new fixed switch is a new version of
    this file and one human intervention.
47. **The check half (code, TODO 48.17).** Half of each task's diagnostic starts, chosen by a hash of the start's
    name, is kept from the roles in the hidden tree; the pool you see is the other half. A candidate the Composer
    sends first runs on the kept half beside the incumbent, start by start, and goes to the gate only when it wins
    there at least once more than it loses. The ledger's `check` row gives the counts, never the starts. Two
    candidates in a row that do not pass stop the run: what the visible half shows no longer carries to starts the
    roles have not seen (in the runs of 2026-11-20 to 22, late candidates gained 17-35 visible starts net and nothing
    at the gate).
    The same per-task rule holds there: a task that loses 2 or more net on the kept half stops the candidate before the
    gate (the `check` row names the task, not the starts).
48. **Segments (code, prompt; TODO 48.17.6).** A task's steps are its segments. Over the incumbent's diagnostic
    rounds the driver counts, per task and step, the rounds that reached the step and the losses blamed on it (the
    cards' first table; `segments` in the roles' contexts). The step that holds most of a task's losses (at least 40 %
    and two rounds) is `hot`; a step at least five rounds reached and none failed at is `stable`. Work where the losses
    are: a hypothesis names the step it acts on, and one that acts before the hot step says why the hot step's cause
    lies there (a hand placed at the reach that slips during the move). A change that alters a stable step risks every
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
