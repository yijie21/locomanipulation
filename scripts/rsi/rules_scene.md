# Scene line appendix

Hand-written rules of the scene line (apartment and warehouse grasp families: grasp_atomic, grasp_floor, grasp_prep,
box_bimanual, pick_*, r1_*, r2_*). It adds to the core. Ids defined here: [scene-robot],
[scene-park], [scene-held], [scene-seed], [scene-redraw], [scene-candidates], [scene-status], [scene-library],
[scene-stage], [scene-classes], [scene-prep], [scene-probe], [scene-keypoints], [scene-twohand], [scene-strict].

### [scene-robot] The robot and the scene (prompt)
Mars G1 with Dex1 two-finger grippers: the jaw opens 9.5 cm, the fingers close along the wrist's y axis, the jaw centre
sits 0.1515 m ahead of the wrist along its x axis, finger force is the PD depth into the object (about 20 N at most). In
mode 4 the waist is the controller's choice: a 20 deg reference lean came out as 8 deg, a squat with an upright reference
leaned 21-56 deg. In mode 2 standing the feet slide apart (27.7 to 36.1 cm in 15 s); mode 4 pins them. The family file's
motor block sets the walk speed; the context's parameters show it. Straight walks stop within 1.5-2 cm, in-place turns end
2-5 deg off. The motor layer walks straight lines only; `navigate.walk_path` plans around furniture on the occupancy map
(body radius 0.35 m: at 0.26 m an arm caught the kitchen island).  Furniture collides with the robot only below its lowest surface.

### [scene-park] Infeasibility claims and parks (code)
A claim that starts cannot be done names the actions it assumed, a hard constraint as a check file the driver runs, and the
exits (a new capability that would open them) or, only for a physical claim, why there are none, what would refute it and
1-3 closed-loop attempts; code refuses a claim missing a part. "Cannot with the present actions" is not "impossible"; only
the user confirms impossible.  Starts park only if the check holds on every claimed start and every attempt fails; they still count as failures in every pool and are no longer repair targets. A park reopens, recorded as wrong, as soon as
an incumbent finishes one of its starts. An exit that fits one Engineer call is built on the Manager's `reopen` (one
Engineer, run on that park's starts only, to the gate only if it finishes one; the park lifts only if the gate accepts it;
at most two iterations in a row).

### [scene-held] The held(X) predicate (code)
`held(X)` needs both fingers of one hand on X (or both hands) and X off every surface.

### [scene-seed] Gap families and empty seeds (code; prompt)
A family the outer loop made for a gap (its file's `about` names the new skills) exists to build them: the first candidates may be a plain end-to-end version (`plan_for` covering the goal,
the existing pick reused), improved one Engineer call at a time; only a skill outside the family's gap is a gap proposal.
A family whose package starts empty (`sandbox.seed`, e.g. grasp_atomic) has no goal covered and no parameters: the first
hypotheses are code (`needs_code`) and grasps are computed from the scene's geometry (box, mesh, surface, gripper size).
A family seeded from an incumbent (box_bimanual) changes the pick that is there.

### [scene-redraw] Pool redraws (code, the driver's, never a role's)
When task-wide parks cover more than half the acceptance pool, the gate judges on a pool of the same size drawn afresh from
the tasks not parked (redrawn whenever that set changes); parked starts still count in the sealed and diagnostic pools. When
the incumbent fails fewer than 10 diagnostic starts, the driver draws fresh starts and swaps failing ones in for passing
ones until 15 fail.

### [scene-candidates] Candidates per iteration (code; prompt)
An iteration builds up to K candidates (`run --candidates K`, default 3; K = 1 is one Engineer, no Composer). The Manager lists up to K different hypotheses, preferably on different tasks; none goes to two Engineers. Each Engineer
builds from the incumbent in its own copy of the skills package; each draft runs on the diagnostic pool when its Engineer
answers.  The Composer merges the edits whose flips show their mechanism, adopts one draft or keeps the incumbent, tying each edit to the starts and trace moment that decided it; only that candidate reaches the gate. A draft stopped as late is void and builds first next time. The gate's best 2 rejected candidates with more
wins than losses wait in `loop/candidates_pool.json` (cleared when the incumbent changes): the Manager may name one as `base`
, the Composer may merge its edits or adopt it. A mechanism proposed in two or
more iterations, given to no Engineer and not refuted takes one slot unless the Manager's `skip` names the failing check:
interface_kept, safety_checks_kept, changed_code_reached or evidence_not_refuting.

### [scene-status] Hypothesis statuses (code; prompt)
A probe gives each hypothesis a `status`: verified (fixed a diagnostic start, broke none), refuted (fixed none), mixed, or
untested (not probed because its mechanism needs code; not a refutation). After a rejection the Manager reads the failure
classes, the gate's per-class tally and the standing suggestion per class. A hypothesis whose test needs code may name a
`measure` {feature, op, value, expect: fails | succeeds, tasks?, scope} over a column of the record library
(`scripts/rsi/records.py features`); the driver then sets the status: verified when the rounds on the measure's side fail (or
succeed) more (at least 3 a side, Fisher p <= 0.05), refuted when at least 10 a side show no difference or the opposite. A
hypothesis may carry `sweep`, at most 2 items {"key", "values"} (2-5 values) beside the fixed `delta.set`. The Manager lists in `tune` the parameter-only hypotheses
whose best net is at least 1; the program builds a draft for each without an Engineer.

### [scene-library] The experience library (code; prompt)
One entry per earlier family, without this family's own and its file's `library_exclude`; it keeps lessons, not the code
earlier runs ended with. Numbers come from other objects and surfaces: cues only ([library]).

### [scene-stage] stage() and the grasp flow (code; prompt)
Every grasp step goes through `stage()` (`src/locomani/tasks/scene/stage.py`): the motor call, a check of the measured state
 (`stance_check`, `reach_check` with `track_ratio`, `grip_check`, `lift_check`, or the skill's own) and an
`on_fail` action (re-walk, re-approach, open and grip again, give up) within the round's retry budget. A missed step is not
followed by the next as if it had worked . A flow for one kind of object lives under `recipes/` with a `SIGNATURE` of features
and
`match(obj) -> score`; the dispatcher picks by features, the generic pick is the fallback. Infrastructure the skills call and
the loop does not change: `graspability.py` (does the object fit the open jaw on each side, how far to turn, which
preparatory action would fit it), `body.py` (from 15,120 measured reaches, how likely a hand lands 3 cm or more off),
`choose.py` (every grasp, hand, stance, posture clear of furniture, scored by that table, best first), `body.best_level` (measured probe curves; None unless one level clearly wins, then keep the default).

### [scene-classes] The failure classes of the scene (code)
 Look hard at `other` and where `events` disagree with the class. no_plan: no plan for the goal.
time_limit: 300 simulated seconds passed. fell_walk / fell: fell before / after the first reach. walk_off: the stance walk
stopped over 5 cm off. pinch_reach / side_reach: the last reach before the grip ended over 5 cm off (pinch when the wrist
points down). too_wide: the object's section across the closing axis does not fit the jaw, and it was pushed or tipped or
the jaw closed on nothing. knocked: moved over 2 cm or tipped over 15 deg before the grip. slip: gripped, then not held.
miss_near: the jaw closed on nothing near the object. The strict goal ([scene-strict]) adds low_lift (rose, never to the
clearance), tilted and dropped_hold (held, then failed inside the window).

### [scene-prep] Preparatory actions (code; prompt)
When the chooser marks an object `needs_prep`, the skill asks `locomani.tasks.scene.prep` which goal would give it a standing
stance (turn the object about the vertical, then pull or push it toward an edge of its support, then clear a neighbour out
of the hand's corridor) and runs the action registered for that kind (`prep.register`), then chooses again on the measured
state. With `movable` floor furniture a fourth kind `floor` moves a chair or stool 20-30 cm off the object or to a side; then call `floor.refresh(world)`. A kind no action implements is a gap the Reviewer reports (`prep.wanted`).

### [scene-probe] Design probes (code; prompt)
A hypothesis may ask for `design` {factor, levels, starts, tasks?, name?}: one factor forced at levels on the same
diagnostic starts under the incumbent. Factors: `param` (a parameter `name` at each level), `spawn_bearing` (object `level`
degrees off the heading), `spawn_distance` (robot `level` metres from the object), `object_yaw` (object turned `level` degrees,
slid up to 0.15 m when it no longer fits), `primitive` (one motor primitive `name`: walk, back, side or turn, alone on empty
floor). At most 2 per iteration. A level that lost more than half its starts says little . A curve says how the factor changes the outcome, never that a change should stay.

### [scene-keypoints] Keypoint grasps (code; prompt, optional route)
`src/locomani/tasks/scene/keypoints.py` turns constraints on an object's points (`centre`, `top`, `face:+x`, `hull:+x`, or
coordinates) into ranked grasps: a pinch, a jaw centre, an approach, a closing axis, a depth, a twist. The chooser ranks them
and `pick` executes the best through stage(). A skill writes which points and constraints by the object's geometry, never
its name.

### [scene-twohand] Two-hand actions (code; prompt)
`reach("both", (left, right), ...)`, `grip("both")`, `release("both")` and `squeeze(inward_m, seconds)` (pushes the jaw
centres together; rides on the servo references until `release` or a reach of that hand; `keep_squeeze=True` keeps it for a
lift by the arms; the push is at most twice `inward_m`). `held(X)` then
needs the fingers of both hands on X off every surface; the checks are `stage.bimanual_grip_check` and `bimanual_lift_check`.
Graspability, the chooser, the body table and prep describe one jaw: a hypothesis may not cite them for a two-hand hold.

### [scene-strict] Strict goals and judging windows (code)
`lifted(X, clearance_m, tilt_deg)`: the lowest corner of X's box is `clearance_m` or more above its support's top and its
bottom face is less than `tilt_deg` off level. With `judge_s` the goal is checked every 0.1 s and must hold at every check
over the last `judge_s` seconds (then `hold_s` standing still): a skill may still lift in the window, but the box may not
slip, tip or drop.  `box_bimanual`: `held(X)` + `lifted(X, 0.20, 60)`, `judge_s` 5, `hold_s` 2.
