# RSI core rules (TODO 54 phase 2, 2026-11-25)

What every benchmark and scene family shares. Each rule has a stable id (`[win]`, not a number) and says what enforces it:
**code** (the driver, the guard, the gate or the runner refuses or decides), **prompt** (only your reading counts) or
both. A family's appendix adds what is specific to it (tasks, robot, objects, the guard's list, the switches). Role
prompts get only the rules they act on, inline; `scripts/rsi/rules_build.py` assembles them. A change to this file is a
change for every family and counts as a human intervention.

### [frozen] What is frozen (code)
The simulator, the tasks and their instructions, the time limits, the environment's success check, the compliance check
([win]), the failure classes, the controller, the motor layer, the frame bridge and the truth view are fixed modules
outside what you edit. Making a task easier is not an improvement. Only the files the guard lists as editable change
([guard]); any other changed file voids the candidate (the driver checks and restores). A start is never removed from a
pool: a start the skill refuses ("no target seen") is a failure, and so is a round not finished when the limit passes.

### [hidden] The hidden tree (code)
The acceptance pool, the check half of the diagnostic starts, the sealed pool and the gate's per-round files are in a
tree that cannot be read while a role runs. Never open it, the benchmark's released evaluation episodes or their results,
or a path outside the sandbox, and never name the hidden directory in a command, not even to exclude it (`grep -v`,
`--exclude-dir`): a command that names it voids your work. A gate verdict JSON and the ledger's `check` row (counts, never
the starts) are what you may read of it.

### [void] Voided work (code)
A role whose tool calls reach the hidden tree voids what it produced: its iteration, or only its draft when it is one of
several Engineers. A Reviewer or Composer whose output is voided runs once more in the same iteration, told why; a second
voided output voids the iteration. Two voided iterations in a row stop the loop. A file changed outside the skill
copies voids every draft of the iteration.

### [candidate] What a candidate is (code)
The skill package plus parameter overrides (`candidate.json` `{"set": ...}`). An unknown key is refused. It must differ
from the incumbent and from every candidate rejected under the present gate rule (`rejected_hashes`). One mechanism per
candidate; keep it as small as the evidence supports.

### [guard] What an edit may not do (code, before anything runs; an edit that fails costs no round)
The static guard compares your package with the incumbent's. Your appendix lists the editable files, the growth cap and
the refused modules. Everywhere: no new reads of simulator state (positions, orientations, velocities, contacts of bodies
or geoms, anything of the simulator library); no branching on the episode's identity or seed; no use of the process
environment or the benchmark's evaluation hooks; no file or network access (open, load, glob, listing, subprocess,
socket, web call); no getattr, setattr, vars, globals, locals, eval, exec, compile or dynamic import; no new top-level
import beyond the base's own and the appendix's list (an import inside a function counts); a locked definition stays byte
for byte. A change that needs something the guard refuses is a locked part, not a level 1 or 2 change ([escalate]).

### [bed-fixed] The bed's fixed switches (code)
The family file's `bed_fixed` lists switches that correct the bed's interface to the benchmark. People found them by hand,
outside the loop. They are on in every round of every role (diagnostics, try.py, drafts, the gate, the sealed set) and are
part of the bed's signature. A set that names one is refused: do not set or "discover" one. Your appendix lists which are
on and why. A new fixed switch is one human intervention.

### [win] What a win is (code: the record; prompt: your reading)
A win is the benchmark's own check holding inside the time limit AND the robot having done what the instruction says (the
record's `comply`, or `success` where the benchmark folds both in). The benchmark's checks are weaker than its
instructions: a shove, a fall into the object, a kick or a knocked object pass them. A change that satisfies the check
without doing the instruction is a loss, and it wastes a gate call. Count only rounds that satisfy both; a round that
flips to the check but not to compliance is not a flip. The appendix says what compliance means per task.

### [runtime] What the skill may use at run time (code: the guard; prompt)
What the robot senses: the camera images and what a segmentation of them gives, the joint readings, odometry; where the
appendix says so, a facade the benchmark gives. Never the simulator's object poses, hinge angles or contacts, and never
the identity of the episode or its seed.

### [branch] What the skill may branch on (code: the guard; prompt: review)
The task's name or the words of its instruction (each task has one fixed instruction) to choose that task's parameters
and flow, and what the sensors measure (size, shape, height, distance). Never the episode, its seed or the level, and no
positions or angles remembered from rounds: the guard refuses the identity and the seed, review refuses remembered
layouts. Scene families: a flow chosen by an object's name voids the draft.

### [levels] Three levels of change, cheapest first (prompt)
Every proposal says its level and what evidence put it there.
- Level 1: parameters and the skill's own decision code (speeds, waits, tolerances, the segmentation's words, where on the
  part the hand goes and how it closes, the path of the move).
- Level 2: where and how the robot stands (stand-off, sideways shift, heading, also at the placing stance). Only when a
  closed-loop probe shows that no level 1 setting wins these episodes while another stance does (same starts, incumbent and
  stance values, at least three episodes that flip) and the palm error or the reach moved with the stance; a stance is not a
  cure for a grip the hand cannot take or a perception error. Stances are tried through the `set` keys the appendix names
  (`stance_*`), then built into the skill's own choice from what it senses.
- Level 3: the controller cannot do it from any stance tried. Edit nothing; file a request ([escalate]).
A level 1 explanation not tried closed loop does not justify level 2; a stance sweep not tried does not justify level 3.

### [escalate] Level 3 and BED reports (prompt)
When a step needs something locked, the controller or its training, stop and say so rather than patch the layer below. A
fall that comes from the controller (the same command alone falls the same way) is its limit: route around it (another
stance, a gentler first move). The Reviewer reports a controller limit in `notes` in a paragraph that starts `LEVEL 3`: the
hand-target region (palm position in the pelvis frame, attitude, pelvis height, as ranges) and the probe evidence (episodes,
stances tried, palm errors against the command). A symptom that sits with one robot model or one part of the bed (a reading
that stays at zero, a joint that does not follow its target) starts `BED`. The driver forwards both to people; the loop goes
on at levels 1 and 2.

### [envelope] The ScaleBFM controller's envelope (prompt; families that use it, the appendix says otherwise for others)
ScaleBFM follows keypoint targets (pelvis, wrists, ankles); it crouches and leans when the hands ask for it and the waist is
its own choice. The motor layer's reach servo brings a hand to about 1 cm after it settles; in a fast entry the hand lags
the command by several centimetres. Open loop the hands land 2-9 cm and 10-30 deg off a target pose, so reaches to a
small part are stepped (in front of it, then onto it) and slow near it, and every wait costs clock. Standing, the feet
slide apart; a straight walk stops within a few centimetres and a turn in place within a few degrees; the reach starts
from wherever the walk ended. A part on a hinge moves only along its arc: a hand moving it along another path pulls
against the hinge and slips.

### [evidence] What counts as evidence (prompt; the gate in code)
Read the per-round records, not the failure strings: a string names where the code gave up, not the cause; the class is a
summary of the record. Use per-round statistics (the value at the deciding moment), then aggregate; a time summed over a
whole round is weighted by whatever the robot waited on. Only closed-loop re-runs from a start are evidence. Reasoning from
the code, offline inverse kinematics or a replayed command is not. Before saying information is absent, look for it a few
minutes; before calling a constant decisive, find every caller of it.

### [noise] The environment is not exactly repeatable (prompt)
The same code from the same start can finish differently on a second run (small differences in contact and depth), and the
gate's and the sealed rounds are fresh. Rounds of the visible pool that share their first commands with an earlier round
are replayed from saved states, so repeating a visible round is cheap but not independent. Do not argue from one flip: argue
from several episodes that flip the same way for the same mechanism, with the trace moment where it acted. A borderline
round (a failure by hundredths of a second, a hinge a few degrees short) is a reason to look at the margin the code leaves,
not at that episode.

### [pool] The diagnostic pool is small (prompt)
The pool has {N_DIAG_TASK} episodes per task that you see (the driver keeps the rest, [check-half]); the pool measures direction only. Do
not tune a number to individual episodes: a value picked because it fixes two named episodes falls back on the next draw.
State a mechanism in measured quantities (the palm error at the reach, the part's size as the camera sees it, the time a
phase takes) with a value that has a reason; every threshold in new code names the quantity it comes from. Prefer a
change that moves the margin on many episodes a little to one that rescues one a lot.

### [regress] Selected subsets regress to the mean (prompt)
Rounds picked because they succeeded now, or failures picked because they fail now, fall back on the next draw. Do not tune
on the rounds that succeed now, and do not read a drop on the worst starts as the effect of a change.

### [layer] When the list keeps growing (prompt)
When a list of fixes keeps growing, the layer is probably wrong: say so instead of adding item N+1.

### [try] try.py is the only runner (code: the budget)
Your own closed-loop checks go through `scripts/rsi/try.py`: the starts it names (`--labels`, `--task`), one skill package
with its `--set`, run in parallel through the shared queue, a table back (outcome, class, first failing phase; against
the reference rounds the starts that flip). One foreground call; the driver's round budget per role and iteration refuses a call past it
and the tool prints what is left. Do not run the bed or the runner yourself, write loops around it, or wait with `until` /
`sleep`. A round is slow, so choose starts that discriminate between causes.

### [classes] The failure classes (code: the driver computes them)
Every round record carries a `failure_class`; your appendix lists the classes of this family with what each usually means.
No class is evidence that a change helps; only the gate is. The gate reports its tally per class.

### [cards] The failure cards (code)
The Reviewer gets, per incumbent diagnostic round that failed, one card computed from the record, under a pool summary. A
round the card cannot answer is run again with try.py; the incumbent's round files are not readable while you work.

### [segments] Segments (code computes; prompt reads)
A task's steps are its segments. Per task and step the driver counts the rounds that reached the step and the losses blamed
on it (the cards' first table). The step with most of a task's losses (at least 40 % and two rounds) is `hot`; a step at
least five rounds reached and none failed at is `stable`. Work where the losses are: a hypothesis names the step it acts on,
and one that acts before the hot step says why its cause lies there. A change that alters a stable step risks every round
through it: keep it only with flips that show what it buys. A segment is where to look, not what is judged.

### [mechanisms] The mechanism table and the history (code)
`mechanisms` is one row per hypothesis so far (its claim and `instead`, what was built, each draft's flips, the merge's and
the gate's verdicts, `open` when nothing of it was built): the index of what was tried. `history` is only the last two
iterations' ledger rows. Read `mechanisms` first. Name a repeated mechanism as its earlier row does; do not re-propose a
refuted one without new evidence; a mechanism the gate rejected in the same form needs a reason to go again.

### [scope] Scope and footprint (code measures; prompt declares)
A draft says where it acts and keeps it there. The Engineer returns `scope`: `when` (the measured condition) and `starts`
(the diagnostic labels it expects to change, `["all"]` only when it acts in every task). Its rounds run on the tasks those
starts belong to plus a few sentinel starts of every other task; `["all"]` runs the pool. The driver records `footprint`:
`changed` (rounds that differ from the ones the draft started from), `outside` (changed, not declared), `missed` (declared,
never reached), `not_run`. Outside `when` the incumbent's path stays as it is. A change that helps one task and hurts
another is two changes: keep the part that helps and condition it.

### [tunable] Parameter search is the program's (code)
When new code needs a threshold, make it a parameter key (an upper-case module constant with a comment saying what it
measures, default taken from that quantity) and list it in `tunable`: at most 3 items `{"key", "values": [2 to 5 values]}`
bracketing the change expected. After the draft's diagnostic run the program sweeps them on the declared scope, once per
iteration, and keeps the best (most starts fixed minus broken, ties to fewer broken, then the earlier value). A knob that
is not listed is never tried. Do not hand-tune numbers.

### [gate] The acceptance gate (code: verdict.py, gate.py)
Only `gate.py` accepts a candidate, in the official environment, on the hidden acceptance pool; nothing else is evidence
that a change helps. Stage 1 takes starts from every task in proportion and rejects only on net < 0; the full pool then
compares candidate and incumbent start by start. Accepted when the one-sided exact sign test on wins b against losses c
gives p <= 0.05 (with p set, the net threshold never binds: 5 against 0, 10 against 3 and 12 against 4 pass; 8 against 2 and
6 against 3 do not). For each task the one-sided exact sign test on its discordant pairs, Holm-corrected across the tasks at
alpha 0.05, must not show a significant loss: 2 losses to 0 wins on one task is noise, 7 to 0 is not. Scene families (no p):
net wins at least 3 and losses at most a third of the wins. Gate calls are budgeted; the verdict JSON gives the tally per
class and per task.

### [check-half] The check half (code)
Half of each task's diagnostic starts, chosen by a hash of the start's name, is kept from the roles in the hidden tree. The
candidate the Composer sends first runs there beside the incumbent, start by start, and goes to the gate only when it wins by
more than it loses with a one-sided sign test p of at most `check_alpha` (0.10 on the benchmarks: 4:0 is sent, 6:2 is not;
two runs of one code differ on 8-17 % of starts, so a small net is noise) and the per-task test ([gate], alpha 0.10) does
not reject it. The ledger's
`check` row gives counts only. Late candidates gained 17-35 visible starts net and nothing at the gate: what the visible
half shows no longer carries to starts the roles have not seen.

### [overlap] The previous candidate is still being judged (code)
The check half and the gate run beside the next iteration: you start from the incumbent as it stood, the context's
`pending` names the candidate in flight (its hash, never its result; no verdict is written or shown before the driver has
it). If it is accepted your draft is rebased onto the new incumbent: dropped when its target starts are already won or
its edits overlap the accepted change, else run again on its scope. Keep edits small and local.

### [stop] Stops (code)
The run stops on the iteration limit, the wall clock, the cost cap, the gate-call budget, two voided iterations in a row, or
`stop_no_progress` iterations in a row (2 on the benchmarks) without a candidate that passed the check half (an iteration
whose Composer kept the incumbent counts; with no check half a candidate that passes the gate's stage 1 counts). Every way
the run ends runs the sealed set once; a stop with a gate call in flight waits for it, records it, then runs the sealed set.

### [sealed] The sealed check (code)
At the end the driver runs the final incumbent and the seed once on the sealed pool, which no role and no gate call has
seen. The final version is confirmed when it wins more than it loses against the seed (one-sided sign test p <= 0.05) and no
task's loss is significant (alpha 0.05). The numbers stay with people; a version not confirmed is reported as not confirmed
whatever the gate said, since many gate calls in a row each risk a lucky acceptance.

### [library] The experience library (code: sandbox)
When the sandbox holds `scripts/rsi/experience/INDEX.md`, it comes from earlier scene families: numbers from other robots,
objects and grippers. Read it for cues; trust this family's rounds.
