# Rules for the RSI loop on the SIMPLE benchmark (adapted from rules.md version 28; this file is version 9, 2026-11-23: rules 25, 47, 49: no task may lose at the gate or the check half, the sealed check at the end (TODO 48.19); version 8, 2026-11-23: rule 25: the gate and the sealed evaluation render every step; version 7, 2026-11-23: rule 48: segments, hot and stable steps (TODO 48.17.6); version 6, 2026-11-23: rules 3, 8, 25, 33, 42, 47: role reruns, the check half, stage 1 first, the probe beside the Engineers (TODO 48.17); version 5, 2026-11-23: rules 4, 12, 17, 43, 45, 46: the bed's fixed switches, the two robot models and the objects' weight, where on the object the grasp sits, `BED` reports; version 4, 2026-11-20: rule 25's sign test, rule 40's held run for the pick tasks; rules 28, 33, 38: parameters searched by the program, review notes, no duplicate assignments, drafts run where they act)

With the family's plan (if its file names one), the only human knowledge the loop reads. The numbering follows
`scripts/rsi/rules.md`, because the driver's messages cite rules by number. A rule that does not apply on this backend
keeps its number and says so in one line. Rules 39 to 48 are new. Every rule says where it is enforced: **code** (the
gate, the runner, the guard or the driver refuses) or **prompt**. A new rule is a new version of this file; it, or a
change to the family plan, is committed and counted as one human intervention.

## The task and its limits (code)

1. The benchmark is frozen: the simulator, the five tasks, their instructions, their time limits, the environment's
   success check, the compliance check (rule 40), the failure classes, the controller (ScaleBFM), the motor layer, the
   frame bridge `Belief` and the truth view. Making a task easier is not an improvement. The Engineer may change only
   the skill package (the files the static guard lists as editable, rule 44); any other changed file voids the
   candidate (the driver checks and restores).
2. A start is never removed from a pool. A start the skill refuses ("no target seen", "no grasp") still counts as a
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
   The record's phase list, marks (object lift, movement and tilt at four moments), palm error and release error are
   the facts; the class is a summary of them.
6. Use per-round statistics (the value at the deciding moment), then aggregate across rounds. A time summed over a
   whole round is weighted by whatever the robot waited on.
7. Only closed-loop re-runs from a start are evidence (the Prober's). Reasoning from the code, re-solving grasps or
   inverse kinematics offline, or replaying a recorded command is not.
8. The diagnostic pool is small and measures direction only: you see 15 of each task's 30 episodes, the driver keeps
   the other 15 (rule 47); see rules 41 and 42.
9. Subsets picked because they succeeded fall back towards the mean. Do not tune on the rounds that succeed now.
10. Before saying information is absent, look for it for a few minutes. Before calling a constant decisive, find every
    caller of it.
11. When a list of fixes keeps growing, the layer is probably wrong: say so instead of adding item N+1.

## The robot and the controller (prompt)

12. A Unitree G1 with a three-finger Dex3 hand on each arm, in two models. XMovePickTeleop, HandoverTeleop and
    LocomotionPickBetweenTablesTeleop use the model ScaleBFM was trained on. BendPickMP and TabletopGraspMP use
    SIMPLE's AMO model of the same robot: its arm joints are position servos (the bed sends them the target angle,
    rule 46), and its finger joints have ten times the damping (0.5 against 0.05) and five times the armature, so the
    fingers lag a fast close even in the air, and a close that stops when the fingers fall behind the command can stop
    before they touch. SIMPLE drives the AMO hand up to 3-5 N m, the other model's hand up to 0.7 N m. A difference
    between the two MP tasks and the three others that the controller does not explain may come from the model
    (rule 45, `BED`). The right hand picks; in Handover the left hand takes the object. Fingers close by joint targets;
    how hard they hold is how far past the contact the joints are told to go.
    Every target object weighs 0.1 kg in all, whatever its size (SIMPLE's `engines/mujoco.py`: the mass is split over
    the object's convex parts; friction 0.8, stiff contact). Holding that weight is easy. Keeping the object where it
    is while the hand arrives and closes is not: the finger or the palm that touches it first pushes or turns it, a
    fast close can flick it, and a tall box tips from a touch high up.
13. ScaleBFM follows keypoint targets (pelvis, wrists, ankles). It crouches and leans when the hands ask for it, and
    the waist is its own choice. The motor layer's reach servo brings a hand to about 1 cm of a target after it has
    settled; during a fast entry the hand lags the command by several centimetres.
14. Open loop, ScaleBFM's hands land 2-9 cm and 10-30 deg off at grasp poses. The skill's reaches are therefore
    stepped (behind the pregrasp, the pregrasp, the grasp) and slow near the object; each step waits for the palm to
    come near the target, and every wait costs time on a clock that is short (rule 39).
15. Standing, the feet slide apart; a walk shuffles. A straight walk stops within a few centimetres, a turn in place
    within a few degrees. The walk's end point is a stance, and the hand's reach starts from wherever the walk ended.
16. The robot walks to a stance, reaches, grips, lifts; the two placing tasks carry the object, walk to the
    container, lower it, let go and take the hand away. Speeds, waits, tolerances and offsets of every step are
    parameters of the skill (the context's `parameters`, with the skill file's comments).
17. The grasp comes from a grasp model's candidates on the target's visible points, chosen by the skill; the hand
    takes only candidates it can reach with the palm and fingers facing a way the arm can deliver. The model
    (GraspGen, Dex3 palm poses with a score) sees the points of the head camera's look inside the target's
    segmentation: the front and the top, so the middle of those points is not the middle of the object. The seed's
    choice takes the `grasp_k`-th candidate, in score order, that passes the reach tests (`ahead_cos`, `roll_lo` to
    `roll_hi`, `clear`); nothing in it looks at where on the object the candidate grips, and the score knows nothing of
    the object's weight or of the hand's lag. A grip off the middle of the face the fingers close on is touched by
    one finger first, which pushes and turns a 0.1 kg object (rule 12). The card's `grasp chosen` line gives the palm's
    place from the middle of the seen points, in the robot's heading axes (ahead, left, up), and the palm's normal in
    the same axes: with the normal pointing down (a grasp from above) the ahead and left parts say how far off the
    middle it grips, with the normal sideways the ahead and up parts.

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
24. A family's new skills (scene backend) do not apply: the skill package here starts from a working seed.

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
    The gate and the sealed evaluation render the cameras in every step (the family's `bed_env_gate`); diagnostic
    rounds, probes and drafts render them only at the robot's looks, which is faster and, on the final version of
    2026-11-22, won 4 of 150 diagnostic starts fewer and none more. Read a diagnostic number as from that mode (code).
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
30. A flow for one kind of object is allowed in the skill package, chosen by measured features of the object as the
    camera sees it (its size and shape, how high it sits, how far), never by its name or its identity in the
    benchmark: a string constant equal to an object's name, or any use of the episode's identity, is refused by the
    guard (rule 44) and by the driver. (code)
31. (scene backend: `stage()` retries.) The skill here is a straight line of phases. A step that missed is not followed
    by the next as if it had worked: check the measured state (palm error, whether the fingers stopped on something,
    whether the object rose) before the next phase, and give up or retry only within the task's time limit. A check
    reads only what the robot may use at run time (rule 40: images, a segmentation, grasp candidates, joint readings,
    odometry).
32. Every round record carries `failure_class`, computed by the driver from the record (loss_simple.py). The classes
    are error, timeout, fall, perceive, deliver, grasp, collide, place and comply; rule 43 says what each means and
    what usually helps. The gate reports its tally per class. None of the classes is evidence that a change helps; only
    the gate is. (code)
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
    - Level 1: parameters and the skill's own decision code: speeds, waits, tolerances, the choice among grasp
      candidates, the hand's path (where it enters from, how it leaves), what the skill does with the camera's
      points (clean-up of the target's cloud, where it puts the surface level), carrying and release geometry.
    - Level 2: where and how the robot stands: the stand-off distance from the target or the table edge, a sideways
      shift, the heading at the stance, and the same for the placing stance. Move up to level 2 only when a probe
      shows that no level 1 setting wins these episodes while another stance does (the same starts, the incumbent and
      the stance values, closed loop, at least three episodes that flip, rule 41). A stance is tried through the
      `set` keys `stance_d` (m, added to the stand-off), `stance_t` (m, a shift to the robot's left along the
      table's edge), `stance_yaw` (deg, added to the heading), `stance_d2`, `stance_t2`, `stance_yaw2` (the same at
      the placing stance) and `stance_here` (true: no walk); a stance that holds is then built into the skill's
      own choice of where to stand, from what the camera measures. A stance is not a cure for a grasp
      the hand cannot take or a perception error: show that the palm error or the reach moved with the stance.
    - Level 3: the controller cannot do it, for example the hand cannot reach or hold the pose from any stance the
      probe tried. Edit nothing. File a request (rule 45). The loop does not train the controller by itself.
    A level 1 explanation that has not been tried closed loop does not justify level 2; a level 2 stance sweep that
    has not been tried does not justify level 3.
40. **A win is the environment's check holding inside the time limit and the robot having done what the instruction
    says.** The record carries both: `success` (the environment's own check, within the limit) and `comply` (did the
    robot do what the instruction says). A round counts as won only when both hold. A change that satisfies the check
    without doing the instruction is a loss, even when the environment would score it. What each task requires:
    - BendPickMP (limit 8 s): "bend to pick up the cracker box". The right hand takes the box off its support and
      lifts it; the check is the box rising by about 9 cm. The box must be held while the check holds: touching a
      hand and nothing that is neither the robot nor the container, for at least 0.2 s in a row, the first of those
      steps inside the limit. Resting on or leaning against a support, or pushed up by the robot's body, is not held.
    - XMovePickTeleop (limit 16 s): "move forward to pick up the {object}". The robot walks to the table and picks the
      object up with the hand; the check is a rise of about 9 cm. Held as above.
    - TabletopGraspMP (limit 16 s): "grasp the object on the tabletop". The same, with a lift of about 2 cm. The
      environment's check is weak here: it holds when closing fingers tip the object up on the table, when the object
      is knocked into the air, and in a few episodes by itself before the robot arrives. None of that is a win.
    In the three pick tasks a step where the check holds without the object being held does not end the episode: the
    skill goes on, and the round is won when a held run of 0.2 s begins inside the limit. The record has `t_check` (the
    first step the environment's check held) beside `t_success`; `comply.why` says what was missing (`no_hand`,
    `on_support`, `short`: held for less than 0.2 s).
    - HandoverTeleop (limit 16 s): "Hand over {object} from right hand to left hand and place it on the container".
      The right hand picks the object up; the left hand takes it from the right hand; the left hand alone then
      carries it to the container and puts it in. The environment's check is only a contact between the object and the
      container held for a short time, so it passes for an object dropped, pushed or thrown in; that does not count.
      Compliance needs: the object was on the right hand and in contact with nothing but the robot, then for at least
      0.2 s on the left hand with no contact with the right hand or anything else, all before the first contact with
      the container.
    - LocomotionPickBetweenTablesTeleop (limit 24 s): "pick up the {object} from table1, locomotion to table2, and
      place on table2". The robot picks the object up, walks to the second table, and puts it on the container there.
      The check holds when the object rests on the container, within about 12 cm of its centre, with no hand on it.
      Compliance needs: at some moment the object was in a hand and touching nothing but the robot (it was lifted
      and carried, not slid), and the object does not stay in contact with a hand at the end.
    One object per task, placed within a few centimetres of where the task puts it; the instruction names it.
    What the robot may use at run time: the head's stereo colour images (depth computed from them), a segmentation of
    the target, grasp candidates from a grasp model, joint readings and leg odometry. Never the simulator's object
    poses or contacts, and never the identity of the episode or its seed. The guard (rule 44) refuses code that
    reaches for them.
41. **The official environment is not exactly repeatable.** The same code from the same start can finish differently
    on a second run (small differences in contact and in the camera's depth). So: do not argue from one flip. A
    change is argued from several episodes that flip the same way for the same mechanism, with the trace moment where
    the mechanism acted; a single flip is noise until it recurs. A round that is borderline (a failure by a few
    hundredths of a second, a lift a centimetre short) is a reason to look at the margin the code leaves, not at the
    one episode. The gate's sentinel rounds may differ between runs; the verdict says what they did.
42. **The diagnostic pool is small** (15 episodes per task that you see, rule 47). Do not tune a number to the individual episodes in it.
    A parameter value picked because it fixes two specific episodes falls back on the next draw (rule 9). Prefer a
    mechanism stated in measured quantities (the palm error at the grasp, the object's width as the camera sees it,
    the time a phase takes) with a value that has a reason, and prefer a change that moves the margin on many
    episodes a little to one that rescues one a lot. Every threshold in new code needs the quantity it comes from.
43. **Failure classes on this backend** (from `loss_simple.py`; one line each: what it means, what usually helps; the
    mechanism, not a setting):
    - `error`: the round did not run (an exception, a crash). Read the traceback; it is a bug in the code or the
      environment, not a robot failure. Fix the code, never the number.
    - `timeout`: the environment's check passed, but after the task's step limit. Time is lost in waits, slow reach
      legs, a long walk or a second attempt; the cure is to find which phases dominate the clock and shorten the ones
      that buy nothing (a wait for a settled hand that is already settled, a speed lower than the arm needs), never to
      skip a check that the next phase relies on.
    - `fall`: the robot fell. Usually a walk with an arm out, a large lean from a reach that is too far from the
      stance, or a push by the table. Stand nearer, keep the hands in before walking, ask for less reach.
    - `perceive`: the target or the container was not found (the segmentation or the depth gave nothing usable).
      Look at what the cloud lacked (a glossy surface, a small object, a view blocked by the arm or by the carried
      object) and where the head looked from; the remedy is a better look (another standing place or heading, clean-up
      of flying points) rather than a guessed position.
    - `deliver`: the palm ended more than 2 cm from the grasp or the release point, or no grasp candidate that the
      hand could take was left. Distinguish a reach the controller did not deliver (palm error large though the
      command was fine: a nearer stance, a smaller or slower entry, a hand attitude the arm delivers) from a grasp
      choice that was out of the hand's reach (choose among candidates by what the arm can deliver) from a target
      read wrongly by the camera (perceive-like, the palm arrived where the cloud said).
    - `collide`: the target was pushed 3 cm or tipped 20 degrees before the fingers closed, or the container was
      pushed 3 cm before the release. Something swept it: the hand's entry path, the idle hand beside the object, the
      body leaning into the reach, a thumb, or the first finger of a grip off the middle of the object (rule 17).
      Change the path of the part that hit it (come from above or from the side with free space, draw the idle hand
      out of the way), where on the object the grip sits, or the stance.
    - `grasp`: the fingers closed and the object did not come up, or was lost on the way. Whether the fingers found
      the object (did they stop on it?), whether the grip is enough for the object's weight and surface, whether the
      lift was gentle enough; a candidate that holds more of the object's width, or one nearer its middle (rule 17);
      how far the palm was from where the fingers wrap. On the AMO model, whether the fingers reached the object at
      all before the close stopped (rule 12).
    - `place`: carried to the container and let go, and it did not stay. Where the object is let go (its foot's height
      over the rim, how far in), how fast the hand opens and leaves, whether the hand drags the object out on the way
      away, whether a tall or light object tips on the container's near side.
    - `comply`: the environment's check held, but the robot did not do what the instruction says (rule 40). The cause
      is in the flow, not in a number: for Handover, a transfer that did not happen between the hands; for the pick
      tasks, a lift that was the body or a support moving the object. Fix the flow so the instruction is done.
44. **What an edit may not do (the static guard; code, applied to the candidate before anything runs; an edit that
    fails it is not run and costs no round).**
    - Only the files the guard lists as editable change: the skill file, the pick routine and the camera view code.
      The seed has no hand-over flow; a candidate may add the one file `handover_simple.py` for it, and that new
      file may hold none of the refused names at all. No other file may appear in the package and `__init__.py`
      stays as it is. A file may grow by at most about 400 lines.
    - The locked definitions stay exactly as they are: the truth view, the frame bridge `Belief`, the helper that
      reads body vertices, and the grasp-library path.
    - No new reads of simulator state: names that reach the physics data (positions, orientations, velocities,
      contacts of bodies or geoms, anything of the simulator library) may not appear more often than before.
    - No branching on the identity of the episode or on the seed, and no use of the benchmark's evaluation hooks or
      the process environment.
    - No file or network access: no opening, loading, listing or globbing files, no subprocess, socket or web call.
    - No dynamic access that goes round the name check: no getattr, setattr, vars, globals, locals, no `eval`,
      `exec`, `compile` or dynamic import.
    - No new imports beyond numpy and the standard helpers math, dataclasses, typing, functools, itertools and
      collections, and none inside a function body.
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
    same way with the label `BED`: it appears on the AMO tasks only (rule 12), a reading stays at zero, a joint does not
    follow its target, the hand lags a command the other model follows. Nothing is edited; the Manager puts it in
    `escalation` with the rounds and the quantities that show it, and people check the bed's interface. The loop does
    not change the bed (rule 1); what people correct becomes a fixed switch (rule 46).
46. **The bed's fixed switches (code).** The family file's `bed_fixed` lists switches that correct the bed's interface
    to the benchmark: the bed sends the robot what the benchmark's robot takes, or reads what the robot really senses.
    People found them, by hand, outside the loop. They are on in every round of every role (diagnostic rounds,
    probes, try.py, drafts, the gate, the sealed evaluation) and in the exam, and they are part of the bed's signature,
    so no round of a bed without them is read again. A set that names one is refused. Now on:
    - `amo_arm_pos`: the AMO model's arm actuators are position servos and get the target angle. Until 2026-11-23 the
      loop's rounds sent them the torque value as if it were an angle (by hand, BendPickMP went from 0 to 6 of 10 when
      this was corrected).
    - `feed_torque`: leg odometry reads the feet's loads from the robot's own torque command. Without it the loads read
      zero in the official environment, and a foot that moves during a reach is read as half a pelvis move (by hand,
      the release error of PickBetweenTables went from about 20 cm to 1 cm).
    Rounds, probes and verdicts from before 2026-11-23 were made without both: read their numbers as from another
    bed, and test again what was refuted there before calling it refuted here. A new fixed switch is a new version of
    this file and one human intervention.
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
