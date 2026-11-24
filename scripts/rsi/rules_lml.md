# Rules for the RSI loop on the LocoManiLab benchmark (adapted from rules_simple.md version 4; this file is version 8, 2026-11-23: rules 25, 47, 49: no task may lose at the gate or the check half, the sealed check at the end (TODO 48.19); version 7, 2026-11-23: rule 48: segments, hot and stable steps (TODO 48.17.6); version 6, 2026-11-23: rules 3, 8, 25, 33, 42, 47: role reruns, the check half, stage 1 first, the probe beside the Engineers (TODO 48.17); version 5, 2026-11-23: rules 4, 12, 45, 46: the bed's fixed switches, the objects' weight, `BED` reports; version 4, 2026-11-21: rules 1, 16, 40, 43: the drawer tasks of the drawer family; version 3, 2026-11-21: rules 16, 40, 43: the door tasks of the fridge family; version 2, 2026-11-20: rules 28, 33, 38: parameters searched by the program, review notes, no duplicate assignments, drafts run where they act)

With the family's plan (if its file names one), the only human knowledge the loop reads. The numbering follows
`scripts/rsi/rules.md` and `rules_simple.md`, because the driver's messages cite rules by number. A rule that does not
apply on this backend keeps its number and says so in one line. Rules 39 to 48 are the ones written for this
benchmark. Every rule says where it is enforced: **code** (the gate, the runner, the guard or the driver refuses) or
**prompt**. A new rule is a new version of this file; it, or a change to the family plan, is committed and counted as
one human intervention.

## The task and its limits (code)

1. The benchmark is frozen: the simulator, the family's tasks (twenty in `lml_q`, three in `lml_fridge`, three in `lml_drawer`), their instructions, their time limits, the benchmark's own
   success check, the compliance check (rule 40), the `genuine` flag, the failure classes, the controller (ScaleBFM),
   the motor layer, the frame bridge and the robot's kinematics, the segmentation client, the grasp-file loader and the
   truth view. Making a task easier is not an improvement. The Engineer may change only the skill package (the files
   the static guard lists as editable, rule 44); any other changed file voids the candidate (the driver checks and
   restores).
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

12. A Unitree G1 with a three-finger Dex3 hand on each arm. Fingers close by joint targets; how hard they hold is how
    far past the contact the joints are told to go (small: the hand's torque limit is low against a pinch, so a palm
    under the object carries weight better than a pinch).
    The objects are light. Each takes its density from its asset (friction 0.9), and most of the things the tasks
    move weigh a tenth of the real thing or less: an apple 10-15 g, a cup 9 or 69 g, bread 26 g, a candle 30 g, a can
    51 g, a boxed food 54 g, a sponge 22 g, a book 115 g, the basket 306 g (from the compiled scenes, 2026-11-23).
    Holding that weight is easy. Keeping the object where it is while the hand arrives and closes is not: the finger
    or the palm that touches it first pushes or turns it, a fast close can flick it, and a tall object tips from a touch
    high up.
13. ScaleBFM follows keypoint targets (pelvis, wrists, ankles). It crouches and leans when the hands ask for it, and
    the waist is its own choice. The motor layer's reach servo brings a hand to about 1 cm of a target after it has
    settled; during a fast entry the hand lags the command by several centimetres.
14. Open loop, ScaleBFM's hands land 2-9 cm and 10-30 deg off at grasp poses. The skill's reaches are therefore
    stepped (behind the pregrasp, the pregrasp, the grasp) and slow near the object; each step waits for the palm to
    come near the target, and every wait costs time on a clock that is short (rule 39).
15. Standing, the feet slide apart; a walk shuffles. A straight walk stops within a few centimetres, a turn in place
    within a few degrees. The walk's end point is a stance, and the hand's reach starts from wherever the walk ended.
16. The flows: the walking tasks walk to the thing the instruction names and stand. The lifting tasks walk to a
    stance, reach, grip and lift. The putting and moving tasks carry the object to the destination, lower it, let go
    and take the hand away. The door tasks (family `lml_fridge`) open or close a hinged door: the door turns about
    its hinge, so a hand on it has to follow the door's arc, and the door has to stay past the goal for the check's
    hold time. The drawer tasks (family `lml_drawer`) open or close a dresser drawer, which slides straight out
    toward the robot and back in; its handle is a bar low on the drawer's front, below where a standing robot's
    hands hang, so the robot has to get a hand low: how it stands for that (how far the pelvis goes down, how far
    the torso leans: the motor layer's `posture` and the `pelvis_z` of `reach`) is level 2, what the controller does
    not deliver from any such stance is level 3. The drawer task with the apple composes a pick, the opening, a put
    into the drawer and the closing. In the seed package `articulate(env)` (skill_lml_cam.py; `view_lml.parse` gives
    the kind `articulate`) walks when the instruction asks for it and looks, and moves no door or drawer: writing
    that flow is level 1 work.
    Speeds, waits, tolerances and offsets of every step are parameters of the skill (the context's `parameters`,
    with the skill file's comments).
17. The grasp comes from a grasp model's candidates on the target's visible points, chosen by the skill; the hand
    takes only candidates it can reach with the palm and fingers facing a way the arm can deliver.

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
    the number of the twenty tasks solved on sealed starts. The candidate's diagnostic rounds and the early review of the next iteration start only once the gate's
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
30. A flow for one kind of task or object is allowed in the skill package. The skill may branch on what the
    instruction says (the object, the destination, the kind of task: walk, lift, put) and on what the camera measures
    (an object's size and shape, how high it sits, how far). It may never branch on the task's id, the episode's
    identity or the seed; the guard (rule 44) and the driver refuse these names. (code)
31. (scene backend: `stage()` retries.) The skill here is a straight line of phases. A step that missed is not followed
    by the next as if it had worked: check the measured state (palm error, whether the fingers stopped on something,
    whether the object rose) before the next phase, and give up or retry only within the task's time limit. A check
    reads only what the robot may use at run time (rule 40: images, a segmentation, grasp candidates, joint readings,
    odometry).
32. Every round record carries `failure_class`, computed by the driver from the record (loss_lml.py). The classes
    are error, timeout, fall, perceive, navigate, deliver, grasp, collide and place; rule 43 says what each means and
    what usually helps. A success that the compliance check rejects is put in the class its reason names. The gate
    reports its tally per class. None of the classes is evidence that a change helps; only the gate is. (code)
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
    incumbent with the stance settings at a few values on the same starts through a hypothesis' `delta` (rule 39's
    level 2).
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
      candidates, the hand's path (where it enters from, how it leaves), how the robot searches for and looks at what
      the instruction names, what the skill does with the camera's points (clean-up of the cloud, where it puts a
      surface level), the walking planner's clearances, carrying and release geometry.
    - Level 2: where and how the robot stands, including whether it walks at all before the reach: the stand-off
      distance from the target or the destination, a sideways shift, the heading at the stance, and the same for the
      placing stance. Move up to level 2 only when a probe shows that no level 1 setting wins these episodes while
      another stance does (the same starts, the incumbent and the stance values, closed loop, at least three
      episodes that flip, rule 41). A stance is tried through `set` keys with the prefix `stance_` (the loop hands them to the bed as the
      stance, not as parameters; they are not in the context's `parameters`; a stance key not listed here, or any key that is neither of these nor a parameter, is refused):
      for the pick, `stance_d` (metres added to the stand-off from the grasp point), `stance_t` (metres, a sideways
      shift to the robot's left), `stance_yaw` (degrees added to the heading at the stance) and `stance_here` (true:
      no walk, reach from where the robot stands); for the placing stance of the putting and moving tasks `stance_d2`
      (metres, stand-off) and `stance_t2` (metres, sideways) and `stance_stay` (true: no second walk, let go from the
      pick's stance). The five walking tasks have no stance keys: a walk's end point is the skill's own decision
      (level 1). Values are plain numbers in the units given. A stance that holds is then built into the skill's own choice of
      where to stand, from what the camera measures. A stance is not a cure for a grasp the hand cannot take or a
      perception error: show that the palm error or the reach moved with the stance.
    - Level 3: the controller cannot do it, for example the hand cannot reach or hold the pose from any stance the
      probe tried. Edit nothing. File a request (rule 45). The loop does not train the controller by itself.
    A level 1 explanation that has not been tried closed loop does not justify level 2; a level 2 stance sweep that
    has not been tried does not justify level 3.
40. **A win is the benchmark's check holding inside the task's time limit, the robot having done what the
    instruction says, and the benchmark's own `genuine` flag.** The record carries all three: `success` (the
    benchmark's check, within the limit), `comply` (`ok` and `why`: did the robot do what the instruction says) and
    `genuine` (the check did not already hold at the start, so the round was an attempt). A round counts as won only
    when all three hold. A change that satisfies the check without doing the instruction is a loss, even when the
    benchmark would score it. A success whose `genuine` is false is counted as an `error` (a measurement fault, not
    an attempt). The record also has `t_check` (the first moment the check held), `t_end`, `limit_s` and
    `instruction`.

    The compliance rules, by kind of task (the benchmark's check is weaker than its instructions):
    - Walking tasks: the robot is upright at the moment the check holds. (`why`: `fell`.) A fall earlier, from which
      the robot got up, does not count.
    - Lifting tasks: with the check holding, the object has been on a hand and touching nothing but the robot for
      0.2 s in a row. Resting on or leaning against its support or a container, being pushed up by the body, or no
      hand on it at all is not held. (`why`: `short` a hand but under 0.2 s, `on_support` a hand and something
      else, `no_hand`.)
    - Putting and moving tasks: before the check first holds, the object was carried: on a hand and touching nothing
      but the robot for at least 0.1 s in a row (`why`: `not_carried`); and at the moment the check first holds no
      hand is on the object (`why`: `hand_on`). A hand is a wrist link or a finger link; the forearm, the torso and
      the legs count as the robot, not as a hand.
    - Door tasks: the robot has not fallen at or before the moment the check holds (`why`: `fell before the door
      check held`), and the round is an attempt (`genuine`). Which part of the robot moves the door is free.
    - Drawer tasks: not fallen and `genuine` as for the doors, and the drawer was moved by a hand: a hand among
      what touched the drawer while it moved, and no thigh, knee, shin or foot (`why`: `drawer moved by a leg`,
      `drawer not moved by a hand`; a forearm or the torso touching it too is allowed). The instructions say
      "crouch down and open it": how low the robot goes is recorded, not required; a kick is not opening or
      closing it. The task with the apple adds the putting rule above for the apple (carried in a hand first; no
      hand on it when the check holds).

    The twenty tasks (name, time limit, instruction, what the benchmark's check requires, what compliance adds). Every
    check must hold for about a second, except row 1, where 0.2 s is enough.
    1. HolodeckApproachDesk, 9.3 s, "walk to the desk": the robot's base within about a metre of the desk's centre.
       Compliance: upright.
    2. IThorBathroomApproachToilet, 17.6 s, "walk over to the toilet and stand next to it": the base within about
       35 cm of the toilet itself. Compliance: upright.
    3. IThorBedroomApproachBed, 11.1 s, "walk to the bed": the base within about 35 cm of the bed itself. Compliance:
       upright.
    4. IThorKitchenApproachFridge, 16.7 s, "walk over to the fridge and stand next to it": the base inside an area
       about a metre or two across beside the fridge. Compliance: upright.
    5. IThorLivingRoomApproachSofa, 11.8 s, "walk over to the sofa and stand next to it": the base inside an area
       about a metre or two across beside the sofa. Compliance: upright.
    6. IThorBathroomLiftSoapBottle, 16.5 s, "pick up the soap bottle": a hand closed on the bottle and touching it,
       and the bottle about 30 cm above where it started. Compliance: held for 0.2 s.
    7. HolodeckPickBookFromDresser, 15.7 s, "pick up the closed book from the side table by the window and lift it":
       the book about 30 cm above where it started; the check asks nothing about a hand, so any way of raising it
       passes. Compliance: held in a hand for 0.2 s.
    8. IThorBedroomLiftBasketFromFloor, 11.8 s, "pick up the basket from the low bedroom stool": as row 6.
       Compliance: held for 0.2 s.
    9. IThorBathroomLiftScrubBrush, 23.6 s, "pick up the scrub brush": as row 6. Compliance: held for 0.2 s.
    10. HolodeckMoveCupToCenter, 20.9 s, "move the glass cup to the center": the cup ends within about 15 to 30 cm of
        the desk's centre and touching the desk. A cup knocked there passes. Compliance: carried in a hand, and no
        hand on it when the check holds.
    11. HolodeckMoveCupToCorner, 11.8 s, "move the glass cup to either robot-side desk corner": the cup ends within
        about 12 cm of either of the desk's two corners on the robot's side. Compliance: as row 10.
    12. IThorBathroomPutSpongeInBathtub, 15.0 s, "pick up the sponge from the bathtub rim and put it in the
        bathtub": the sponge inside the tub's basin and no hand touching it. Compliance: carried in a hand first.
    13. IThorBedroomBoxedFoodIntoBasket, 9.7 s, "put the boxed food in the basket": the object inside the basket and
        touching it, with no part of a hand within about 18 cm. Compliance: carried in a hand first; no hand on it.
    14. IThorBedroomCanIntoBasket, 14.6 s, "put the can in the basket": as row 13.
    15. IThorKitchenAppleIntoBox, 10.0 s, "put the apple into the box": the apple's middle inside the box for about
        a second; the check asks nothing about a hand, so an apple still in the hand passes. Compliance: carried in
        a hand first, and no hand on it when the check holds.
    16. IThorKitchenBendAndPutAppleOnPlate, 10.9 s, "bend down and put the apple on the plate on the low table": the
        apple touching the plate, within about 13 cm of the plate's centre. The check asks nothing about a hand or
        about bending. Compliance: carried in a hand first, and no hand on it.
    17. IThorKitchenBreadOnPlate, 8.8 s, "put the bread on the plate on the table": as row 16 (without the
        bending).
    18. IThorKitchenCupToSink, 10.3 s, "pick up the cup beside the sink and put it into the sink": the cup inside
        the sink and no part of a hand within about 25 cm. Compliance: carried in a hand first; no hand on it.
    19. IThorLivingRoomBookIntoBasket, 11.4 s, "pick up the book and place it in the basket": the book touching the
        basket within about 12 cm of its centre and no part of a hand within about 25 cm. A book lying on the rim
        or leaning in from outside may pass. Compliance: carried in a hand first; no hand on it.
    20. IThorLivingRoomCandleToBasket, 10.7 s, "pick up the candle and place it in the basket": as row 19.

    The fridge family's three tasks (`lml_fridge`; the door's open fraction is its angle over 90 degrees, 0 closed,
    1 wide open; the target is the left door of the fridge in front of the robot):
    21. IThorKitchenOpenFridge, 9.4 s, "open the fridge door in front of you": the door starts at 0.10 and must be at
        0.50 or more for a second in a row. Compliance: not fallen.
    22. IThorKitchenCloseFridge, 10.4 s, "close the fridge door in front of you": the door starts at 0.30 and must be
        at 0.10 or less for a second in a row. Compliance: not fallen.
    23. IThorKitchenApproachOpenFridge, 11.5 s, "walk to the fridge from far away and open the fridge door": the
        door starts closed and must reach 0.40 with any part of the robot within 0.75 m of the fridge; no hold.
        Compliance: not fallen.
    The drawer family's three tasks (`lml_drawer`; the drawer's open fraction is how far it is out over its 20 cm
    of travel, 0 shut, 1 all the way out; the target is the lower-left drawer of the dresser; its handle is a bar
    across the drawer's front):
    24. IThorLivingRoomLowDrawerPnP, 38.0 s, "pick up the apple from the top of the dresser, open its lower-left
        drawer, put the apple inside, and close the drawer": in this order, the drawer at 0.80 or more for 0.1 s,
        then the apple inside the drawer and the drawer at 0.20 or less, both for a second in a row. The drawer
        starts at 0.20, the apple on the dresser's top right above it, the robot facing the dresser close by.
        Compliance: not fallen; the drawer moved by a hand; the apple carried in a hand first and no hand on it.
    25. IThorLivingRoomWalkAndOpenLowDrawer, 97.2 s, "walk to the lower-left drawer of the dresser, then crouch down
        and open it": first the robot within 0.35 m of the drawer (any part of it, 0.1 s), then the drawer at 0.80
        or more for a second in a row. The drawer starts at 0.20; the robot starts a few metres away with
        furniture in between. Compliance: not fallen; the drawer moved by a hand.
    26. IThorLivingRoomWalkAndCloseLowDrawer, 28.6 s, "walk to the lower-left drawer of the dresser, then crouch
        down and close it": as row 25, but the drawer starts at 0.80 and must be at 0.20 or less for a second in a
        row. Compliance: not fallen; the drawer moved by a hand.
    A drawer round's card has a `drawer:` line like the door's, plus what was on the drawer while it moved and how
    far the pelvis went below its standing height (cm); the palm's distance is to the handle bar's axis.
    A door round's card has a `door:` line: the open fraction at the start, its maximum, minimum and end, when it
    first moved, when it first entered the goal and how long it stayed there, which parts of the robot touched the
    door (hand, arm, torso, leg), and the palm nearest the handle: its distance, how far above the handle it was
    then (negative: below), how far off sideways, the highest any palm got against the handle's height, and whether
    the fingers closed while the palm was within 4 cm of the handle. Distances are relative; nothing on a card gives
    a position in the room.

    What the robot may use at run time: the instruction text, the head camera's colour and depth image, a
    segmentation of what the instruction names, grasp candidates from a grasp model, joint readings and leg odometry.
    Never the simulator's object poses or contacts, never the target coordinates or regions of the benchmark's check,
    and never the identity of the episode or its seed. The scene of a task is the same in practice and in the final
    evaluation, with objects moved by a few centimetres. So a position in the room's frame written into code as a
    number (a stand point, a target spot, a heading to a piece of furniture) is knowledge of the evaluation, not
    something the robot saw: do not write one, whether you got it from a trace, a probe or the task's description.
    Offsets, distances and angles measured from what the camera sees (from the target, from the destination, from the
    robot's own pelvis) are fine. The guard (rule 44) refuses code that reaches for the simulator's data.
41. **A re-run may not repeat exactly.** Small differences in contact and in the camera's depth may let the same code
    from the same start finish differently. So: do not argue from one flip. A change is argued from several episodes
    that flip the same way for the same mechanism, with the trace moment where the mechanism acted; a single flip is
    noise until it recurs. A round that is borderline (a failure by a few hundredths of a second, a lift a centimetre
    short) is a reason to look at the margin the code leaves, not at the one episode. The gate's verdict says what
    its sentinel rounds did.
42. **The diagnostic pool is small** (you see half of each task's starts, rule 47; the context says how many). Do not tune a number to the
    individual episodes in it. A parameter value picked because it fixes two specific episodes falls back on the next
    draw (rule 9). Prefer a mechanism stated in measured quantities (the palm error at the grasp, the object's width
    as the camera sees it, the time a phase takes) with a value that has a reason, and prefer a change that moves the
    margin on many episodes a little to one that rescues one a lot. Every threshold in new code needs the quantity it
    comes from.
43. **Failure classes on this backend** (from `loss_lml.py`; one line each: what it means, what usually helps; the
    mechanism, not a setting):
    - `error`: the round did not run (an exception, a crash), or the check held at the start (`genuine` false: not an
      attempt). Read the traceback; it is a bug in the code or the environment, not a robot failure. Fix the code,
      never the number.
    - `timeout`: the time limit was reached with nothing else wrong: the flow was still on its way (it had not reached
      the grasp, or it held the object and had not let go). Time is lost in waits, slow reach legs, a long walk, a
      look that repeats or a second attempt; the cure is to find which phases dominate the clock and shorten the ones
      that buy nothing, never to skip a check that the next phase relies on.
    - `fall`: the robot fell (also a success with `comply.why` `fell`). Usually a walk with an arm out, a large lean
      from a reach that is too far from the stance, or a push by furniture. Stand nearer, keep the hands in before
      walking, ask for less reach.
    - `perceive`: the target or the destination was not found (the segmentation or the depth gave nothing usable).
      Look at what the cloud lacked (a glossy or small object, a view blocked by the arm or by the carried object, the
      thing out of the field of view) and where the head looked from; the remedy is a better look (another standing
      place or heading, a search, clean-up of flying points, another phrase for the segmentation drawn from the
      instruction's words) rather than a guessed position.
    - `navigate`: a walking task whose walk ended outside what the check accepts. Read where the walk stopped against
      what the camera saw of the thing, how the walking planner treated the clearance round obstacles, and whether
      the stop was cut short.
    - `deliver`: the palm ended more than 2 cm from the grasp or the release point, a reach step failed, or the flow
      ended before any grasp. Distinguish a reach the controller did not deliver (palm error large though the
      command was fine: a nearer stance, a smaller or slower entry, a hand attitude the arm delivers) from a grasp
      choice that was out of the hand's reach (choose among candidates by what the arm can deliver) from a target
      read wrongly by the camera (perceive-like, the palm arrived where the cloud said).
    - `collide`: the target was pushed 3 cm or tipped 20 degrees before the fingers closed, or the destination was
      pushed 3 cm before the release (also a success with `comply.why` `on_support`). Something swept it: the hand's
      entry path, the idle hand beside the object, the body leaning into the reach, a thumb. Change the path of the
      part that hit it (come from above or from the side with free space, draw the idle hand out of the way) or the
      stance.
    - `grasp`: the fingers closed and the object did not come up, or was lost on the way; also a success that
      compliance rejects as `no_hand`, `short` or `not_carried`. Whether the fingers found the object (did they stop
      on it?), whether the grip is enough for the object's weight and surface, whether the lift was gentle enough; a
      candidate that holds more of the object's width; how far the palm was from where the fingers wrap.
    - `place`: carried to the destination and let go, and it did not stay; also a success with `comply.why`
      `hand_on`. Where the object is let go (its foot's height over the rim or the surface, how far in), how fast the
      hand opens and leaves, whether the hand drags the object out on the way away, whether a tall or light object
      tips on the near side.
    Door tasks add four classes, after `error`, `fall` and `perceive` (the handle or the door was not found):
    - `art_not_reached`: no palm came within 5 cm of the handle and no hand or arm touched the door. Tell a reach
      the controller did not deliver (the palm's error against the command; the highest palm against the handle's
      height) from a target the camera put in the wrong place, and from a stance too far or too low.
    - `art_no_move`: a hand got there, but the door's fraction never moved 0.03 from its start: the fingers did not
      hold the handle, or the hand pulled or pushed in a direction the hinge does not allow.
    - `art_partial`: the door moved but never reached the goal: the hand slipped off along the arc, the arm ran out
      of reach, or the body was in the door's way.
    - `art_not_held`: the door reached the goal but left it before the hold time ended (it swung back, or the hand
      let go too early or pulled it back).
    Drawer tasks use the same four classes for the drawer (the handle is its bar; a hand, forearm or upper arm on
    the drawer counts as reached), and a success the compliance check refuses because a leg moved the drawer or no
    hand did is `art_not_reached`. The task with the apple goes by its steps: the apple's pick first (`collide`,
    `deliver`, `grasp`, as for the lifting tasks), then the drawer never opened to 0.80 (`art_*`), then the apple
    lost on the way or not let go (`grasp`, `deliver`), then let go but the drawer left open (`art_partial`), the
    apple not inside or not staying (`place`), the drawer shut for less than a second (`art_not_held`).
44. **What an edit may not do (the static guard; code, applied to the candidate before anything runs; an edit that
    fails it is not run and costs no round).**
    - Only the four files the guard lists as editable change: `skill_simple.py` and `pick.py` (carried over from
      the SIMPLE seed: the flow's shared parts and the pick routine) and `skill_lml_cam.py` and `view_lml.py` (the
      walking and placing flows on the camera, and the camera view). No file may be added or removed and
      `__init__.py` stays as it is. A file may grow by at most about 400 lines.
    - The locked definitions stay exactly as they are (formatting and comments aside): the truth view, the frame
      bridge and the robot's head kinematics, the segmentation client, the grasp-file loader and the cloud name
      function, and in `pick.py` the helpers the guard lists.
    - No reads of simulator state. The names that reach the physics data or the task's definition (positions,
      orientations, velocities and contacts of bodies or geoms, anything of the simulator library, the task's spec,
      the truth view, the frame bridge to the simulator's world, the episode's id or seed) may not appear at all in
      `skill_lml_cam.py` and `view_lml.py`, and in `skill_simple.py` and `pick.py` no more often than in the
      incumbent. What the package gets of the simulator comes through narrow accessors the seed already uses: the
      head camera's colour and depth image with its field of view, the finger joint readings, the cloud name it is
      given.
    - No branching on the identity of the episode or on the seed, and no use of the benchmark's evaluation hooks or
      the process environment.
    - No file or network access: no opening, loading, listing or globbing files, no subprocess, socket or web call.
    - No dynamic access that goes round the name check: no getattr, setattr, vars, globals, locals, `__dict__`, no
      `eval`, `exec`, `compile` or dynamic import.
    - No new imports beyond numpy and the standard helpers math, dataclasses, typing, functools, itertools,
      collections and heapq (the guard also lets scipy through).
    A change that needs something the guard refuses is not a level 1 or 2 change: it is a locked part, and the Manager
    treats it as outside the loop (rule 20).
45. **Level 3 requests.** When the evidence says the controller itself cannot do it from any stance (a hand that does
    not reach or hold a pose, a wrist attitude it does not deliver, a pose it will not hold under the object's
    weight), nothing is edited. The Reviewer reports it in `notes` with the label `LEVEL 3` and the Manager writes the
    request in `escalation`: it names the hand-target region where data for the controller should be collected (the
    palm's position in the pelvis frame, its attitude, and the pelvis height, as ranges) and gives the probe evidence
    (the episodes, the stances tried, the palm errors against the command). The request is for people; the loop goes
    on with what it can do at levels 1 and 2.
    A symptom that sits with the bed rather than with the controller is reported the same way with the label `BED`:
    a reading stays at zero, a joint does not follow its target, the robot's estimate of its own pelvis or hand moves
    when the true one does not (or the reverse), a task behaves as no other task with the same motion does. Nothing is
    edited; the Manager puts it in `escalation` with the rounds and the quantities that show it, and people check the
    bed's interface. The loop does not change the bed (rule 1); what people correct becomes a fixed switch (rule 46).
46. **The bed's fixed switches (code).** The family file's `bed_fixed` lists switches that correct the bed's interface
    to the benchmark: the bed sends the robot what the benchmark's robot takes, or reads what the robot really senses.
    People found them, by hand, outside the loop. They are on in every round of every role (diagnostic rounds,
    probes, try.py, drafts, the gate, the sealed evaluation) and in the exam, and they are part of the bed's signature,
    so no round of a bed without them is read again. A set that names one is refused. Now on:
    - `feed_torque`: leg odometry reads the feet's loads from the robot's own torque command. Without it the loads read
      zero in the benchmark's environment, and a foot that moves during a reach is read as half a pelvis move (by
      hand, the scrub brush: the estimate of the hand at the end of the reach went from 8 cm off to 1.5 cm).
    Rounds, probes and verdicts from before 2026-11-23 were made without it: read their numbers as from another bed,
    and test again what was refuted there before calling it refuted here. A new fixed switch is a new version of this
    file and one human intervention.
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
