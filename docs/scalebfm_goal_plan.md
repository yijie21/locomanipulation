# ScaleBFM visual expert: registered experiment plan

Created 2026-09-08. Goal state: `docs/scalebfm_goal_state.md`. Design contract:
`docs/scalebfm_visual_expert_design.md`. Experiment root:
`outputs/experiments/scalebfm_visual_goal_20260908/`.

## Objective and constraints

Deliver a reproducible collection strategy for a dual-RGB + proprioception VLA
that approaches a table, raises the hand during approach, grasps upright when
comfortably reachable, and bends forward when needed. No object ground truth is
given to the VLA or its deployed action executor. Expert-only geometry must pass
through a visibility/rate/latency-limited observation adapter. Preserve all prior
data and other sessions. No commits without user instruction. Do not use either
GPU or instance 0 before its reservation is explicitly released. Never invoke
stop_stack.sh. CPU inference, FK, unit tests and software rendering are allowed.

## Checkpoints (execute sequentially)

- C0: freeze baseline, metrics, budgets, manifests and restart procedure.
- C1: continuous wrist reference and command-space tests; CPU paired probes.
- C2: restricted scene estimates; progress-driven approach/reach/recovery;
  posture-conditioned reach calibration and camera evidence checks.
- C3: common collection/deployment action executor and explicit temporal
  observation contract; exact target round-trip and perturbed-state replay.
- C4: freeze candidate experts; collect/audit matched-budget datasets.
- C5: fixed-budget VLA training, selection evaluation, untouched confirmation.
- C6: compare candidates, package winner (or explicitly report no acceptable
  winner), reproduce commands, limitations, final review and checkpoint.

Implement and test smaller substeps within each checkpoint. Do not skip C3
because the expert succeeds. Do not declare the goal complete before C5/C6.

## Pre-training development budget and scene splits

CPU development: at most 12 named candidate configurations, up to 12 paired
rounds each, plus focused contract/diagnostic probes. Changes after seeing these
results must receive new IDs. Development scene seeds 101..112; edges x-, y-, y+
balanced (counter obstructs x+, excluded explicitly). Start with table height
0.621 m and inboard 0.04..0.08 m. Add visible bottle-depth strata 0.10..0.16 and
0.18..0.24 m to elicit reaching and bending once the near-edge case is stable.
Distance strata are scene factors, not hidden per-episode style choices.

Candidate qualification: 30 fresh scenes, seeds 1101..1130, balanced across
edges and calibrated reach strata. Recovery tests use 12 separate cases:
stop-short 0.10 m, wrist offset 0.03 m, temporary observation blackout (0.4 s),
and visibly shifted bottle 0.04 m, each on three edges. Invalid physics is a
failed case, never silently discarded. A reach stratum unsupported by the robot
must be documented as outside the domain, not quietly removed after evaluation.

## Measurable gates (v1, freeze before qualification)

Numerical thresholds below are engineering acceptance targets, not claims about
human biomechanics. Any amendment is logged with reason and pre-amendment result.

| Property | Gate / reported measure |
|---|---|
| Physics | Zero nonfinite/reset/QACC instability; no fall (pelvis z <0.45 m) in accepted data |
| Task success | Bottle rises >=0.08 m, tilt <=30 deg, held for >=1 s; no pre-close knock |
| Expert reliability | >=24/30 qualification successes; >=6/10 in each calibrated reach stratum |
| Command continuity | At 50 Hz, wrist increments <=0.02 m and <=5 deg, including phase/replan boundaries; log velocity/acceleration |
| Side lean | Accepted episode torso |roll| peak <=20 deg, time above 15 deg <=0.2 s; report waist roll separately |
| Twist | Accepted episode waist |yaw| peak <=25 deg; report signed trace and duration above 20 deg |
| Forward bend | Allowed up to 40 deg torso pitch; report upright and bend strata separately, never reward zero pitch everywhere |
| Walk/reach coordination | >=80% of successful normal approaches start lifting before pelvis speed falls below 0.05 m/s; speed estimated from history |
| Visual evidence | At least one camera has >=32 actual target pixels at 320x240 and >=20% visible fraction at >=90% of pre-close decision updates; log image margin and blackout duration |
| Occlusion causality | Hidden object motion cannot change task commands until a new permitted observation arrives; low confidence stops precision entry |
| Timing causality | Delaying approach must delay reach/close appropriately; no transition solely from episode clock |
| Recovery | >=8/12 successful recoveries; every failure retained in manifest |
| Action fidelity | Same state + recorded command gives identical ScaleBFM targets (pos <=1e-6 m, rot <=1e-5 rad); no future root/object truth needed |
| Replay | Same-state deterministic replay succeeds for >=9/10 qualified expert episodes; perturbation replay reported separately |

The original baseline is evaluated with these metrics but keeps its original
acceptance rule as the control arm. A candidate must pass the new gates before
being advertised as the improved collection strategy. A threshold failure is not
fixed by selecting only an unrepresentative subset of easy scenes.

## Fixed VLA budget

Maximum three frozen collection arms: A current edge expert (baseline snapshot),
B best qualified continuous/observable expert, C B with recovery data replacing
25% of normal demonstrations. C is included only if its recovery generator passes
C2; otherwise use A/B and do not silently spend its budget on additional tuning.

For every included arm:

- 200 accepted train episodes +20 validation episodes; maximum 880 attempts.
  Same scene distribution and seeds, independent train/validation pools. Record
  failed-attempt cost and total frames; fix optimization budget despite differing
  episode lengths. C uses 150 normal +50 recovery train episodes.
- Same GR00T-N1.7-3B base, camera layout, observation/action contract, optimizer,
  effective batch 64, learning rate and augmentation. Freeze the resolved config
  before the first comparative train; rerun the frozen baseline task generator through the common executor and
  record those executed demonstrations (see C3 amendment below).
- Two training seeds (17,29), 3000 optimizer steps each, checkpoint at final step;
  at most 6 full runs /18000 optimizer steps. No selecting checkpoints using test
  outcomes. Failed infrastructure runs and restarts count in the ledger.
- Selection evaluation: 60 identical fresh scenes per model, scene seed pool
  21001..21060, fixed policy RNG protocol, 0.4 s replan and 0.08 s latency unless
  C3 preregisters a different common runtime before training.
- Primary selection metric: mean closed-loop task success across both train
  seeds, with per-stratum breakdown. Also report roll/twist, visibility, early
  stopping, recovery and latency. Qualified candidates within one success of one
  another per 60 scenes are tied; prefer lower posture violations then simpler
  implementation. Do not claim statistical superiority from a point estimate.
- Lock winner before confirmation: selected candidate and baseline, both train
  seeds, 100 fresh cases each, pool 31001..31100; no retuning on this set. Report
  paired discordant outcomes/McNemar per train seed and uncertainty intervals;
  do not treat two seeds on the same scene as 200 independent scenes.

If no candidate qualifies or improves, retain baseline as best measured task
performer, state unmet naturalness/learnability gates, and document the next
bounded experiment rather than declaring the engineering objective achieved.

## Durable experiment record

Each run has unique ID, source hashes, config, scene IDs, command, process/log
paths, start/end UTC, outcome, metrics, failure classification and next action.
Save config before launch and results atomically on completion. Never reuse a
completed ID for a different configuration. Snapshots/traces live in outputs;
compact conclusions and links live in docs. Record every checkpoint immediately.

On resume: read STATE, goal_state and this plan; inspect current code and live
processes; verify hashes/artifacts for the next operation. Do not restart a live
process or rerun completed experiments just because conversation memory is gone.

## C3 preregistration amendment (2026-09-08, before qualification/training)

The formal A arm will rerun the frozen legacy task generator through the same
executor and stable gripper PD used by B/C. The original a00 remains the unchanged
diagnostic baseline. We cannot truthfully call post-hoc conversion of its raw
six-frame forecasts exact: the 29-D codec linearizes them (observed wrist forecast
error <=0.36 cm in the first contract probe), and the old gripper PD oscillates
4–6 mm even while commanded open. Applying those corrections equally to all arms
avoids confounding the collection-policy comparison with different runtime
physics. Report original-a00 versus common-runtime-A expert behavior separately.
This amendment does not change the training/evaluation budget or acceptance gates.

Provisional common runtime: native 29-D rows, canonical float32 before teacher
execution, 25 Hz rows held for two 50 Hz control ticks, live pelvis yaw frame,
shared visual_executor.ActionExecutor and visual_action.decode_action. No extra
object-dependent correction in deployment. 25 Hz float32 contract probe: 3/3
success and all three deterministic replays match every qpos and wrist command
exactly; full qualified 10-case replay is still required.

Provisional observation: two RGB views at t-0.4 s and t; 91-D robot state (old
53-D proprioception +9 own robot velocity values +29 previous submitted action).
No absolute root position, object state, contact success, phase or scene estimate
is a VLA modality. History is actual past frames; first-frame padding follows
the dataset index convention. Same configuration for every formal arm.

## C2 reach-domain registration (2026-09-08, before qualification)

The unloaded mode-2 calibration on the original 0.621 m table shows 20–33 deg
forward pitch across 0.25–0.40 m front distances. A 0.761 m table reduces this
to 7–21 deg with bottle-facing posture. Therefore depth alone on the original
low table does not supply an upright stratum. Add a visibly taller table as
a scene factor, using an isolated XML copy with vertically scaled table/legs.
Shared scene XML is untouched. Original arm IK with nose-up feedforward failed
the strict pose target across these samples; it is not proof of infeasibility
for other wrist orientations and is not used as a privileged policy input.

Before qualification, test these three registered strata equally:
1. high_near: table top 0.7611386 m, bottle inboard 0.04–0.08 m; intended
   upright stratum, median absolute torso pitch during close <=15 deg.
2. low_near: original table top 0.6211386 m, inboard 0.04–0.08 m.
3. low_deep: original table top, inboard 0.18–0.24 m; preserve at least
   0.22 m commanded pelvis clearance from the observed edge.
All original posture/physics/visibility gates remain. If the high_near upright
gate or deep reach fails, report that limitation rather than relabeling cases.
Development a09: 12/12 near successes with a bounded steady-entry tracking
correction, 25 Hz float32 shared executor and visible carry-height feedback.
Development a10 is the first three-stratum candidate (same controller plus
visible edge-clearance constraint). No formal VLA budget has been spent.

## Bounded development repair amendment (2026-09-08, before qualification)

Initial a00..a11 budget is spent; no configuration is yet qualified across all
three strata. a10/a11 high_near cases used a faulty scene copy: visual meshes
were scaled, but the invisible contact box was not. All affected failures stay
in the ledger and cannot support high-table policy conclusions. A regression
test now checks collider height and bottle settling. a11 also tightened the
pre-alignment tolerance to 2 cm, below its measured residual tracking error.

Allocate within this implementation plan one bounded repair block: at most two
additional candidate configurations (a12/a13), each <=12 development cases.
a12 repairs scene geometry and restores the originally proposed 3.5 cm pre
corridor while retaining bounded correction during reach. No VLA steps/eval
cases are added, and no qualification or test outcomes have been seen. If these
fail, report the unmet gates and use the best measured supported domain; do not
quietly advertise an upright/three-stratum solution.

The common executor now caps actual world wrist increments at 2 cm /5 deg,
including live-pelvis motion between held rows (a10 had one 2.031 cm step).
Its previous applied wrist command is deterministic executor state; exact replay
starts with the same state/history. All formal arms use this same limiter.

Mode-2 repair block outcome: a12=9/12 (high near successes pitch24–27 deg, fails
upright requirement); a13=8/12, low-near/deep8/8, high-near0/4. No three-stratum
qualification or VLA training authorized by these results. Next is a focused
mode-6 feasibility diagnosis, not another unregistered mode-2 parameter sweep.
C3 camera/parser/export/runtime work can proceed independently.

## Geometry-based follow-up (2026-09-08, no VLA budget change)

Focused mode6_fk_feasibility (six unloaded cases) did not justify switching
control modes: wrist errors were 4.9–8.4 cm in mode6 versus 3.1–3.4 cm in mode2.
With valid FK references and no legacy overdrive, mode2 high-near pitch was
-3.5 deg. The high-near arm IK was feasible (position error0.0006cm), so the prior
claim that high-near might require mode6 is not supported. The old +10cm position
bias and -13deg wrist feedforward were fitted to distant low-table approaches.

Register one additional candidate a14 (12 development cases), replacing those
legacy feedforwards with direct level-palm references plus bounded own-state
tracking feedback. This is a targeted test of the identified reference-geometry
issue, with the same mode2/action29/runtime and existing three-stratum setup.
All earlier tuning/repair attempts remain counted; total candidate allowance
becomes15 IDs (a00..a14). Formal training/evaluation budget remains unchanged.

Further diagnosis of a14: low/deep8/8 but high-near0/4. In seed101 the intended
pre-wrist lies only5cm ahead and8cm to the LEFT of the pelvis-heading origin;
actual pad lateral error15.5cm. Facing the bottle from the nominal stand cancels
the intended right-shoulder offset, and actual base lag makes the cross-body
reach worse. This violates the design's heading/shoulder-alignment intent.
Register a15 (12 cases), changing only facing to the approached table normal
relative to a14. Visibility remains an actual-image gate; off-center head view
must be covered by the wrist view. This increases recorded CPU candidate count
to16; the fixed VLA comparison budget is unchanged. No qualification has begun.

### Qualification freeze

a15 is selected for the registered30 fresh qualification cases; development
audit12/12, walking/arm-extension overlap12/12, exact executor replay12/12.
Arm-motion onset is measured from actual FK relative to the pelvis: >=3cm
forward extension or>=1cm rise after reach starts; base speed from the preceding
0.1s must exceed0.05m/s. This avoids mistaking a logged phase switch for motion.
Qualification may record RGB/state/actions for replay/export tests but these
seeds are excluded from training and validation datasets. No tuning on these
results; failures remain in the record. Recovery12 and full C3 remain required.

Qualification recording split (performance only): keep full25Hz RGB/state/actions
for1101..1103; CPU recording cost~100s/episode. Stop own recorder before finishing
1104 and retain the incomplete artifact. Run1104..1130 without RGB under the same
policy and original case index offset3; no completed scene is repeated. Outcome
set is still the same30 preregistered seeds, no policy tuning.

C3 data-loader correction: set data.allow_padding=True explicitly; otherwise
negative image history indices wrap to future end-of-episode images in the
current loader. This was verified with the real episode loader and a raw qualified
recording. Use shortest_image_edge240, crop_fraction1.0 (full audited FOV) for all
formal arms. Spatial cropping cannot remove evidence the expert was allowed to
use. Keep VLA step/seed/case budgets unchanged.

## C3 final runtime and C4 pool freeze (2026-09-08, before formal data/training)

Common baseline interface pilot:12/12 accepted (1401..1412), all three strata
covered. Legacy task source is the original snapshot, with stable PD1/.025,
common1.68s startup, right jaw only (left open), native29 float32 at25Hz and
shared world-wrist limiter. These common-runtime changes are not claimed as
new expert task logic. All task failures and new-gate violations are reported.

Formal collection and evaluation both use Isaac-GR00T/.venv (torch2.9cu128,
MuJoCo3.11), GPU0. Simvenv2.13 versus2.9 caused tiny replay differences despite
3/3 success; same-environment delayed-chunkreplay is exactly equal3/3 in each
environment. Fresh GR00Tenv seeds1501..1503 recorded and replayed with zero
qpos/command error; excluded from train/val. Common inference uses4 denoising
steps,40rows, capture every20controlticks, latency4ticks,25Hz actions/50Hzcontrol.
First capturetick80(1.60s), first rolloutrowtick84(1.68s); during later pending
replans the previous chunk continues. Policy RNG per scene=900000+scene_seed,
identical across arms/train seeds. Max rollout24s including startup.

A/B proposed train pool4001..4800 (at most800), val6001..6080 (at most80).
Take first200/20 accepted in predetermined order. Scene indexi maps stratum
[high_near,low_near,low_deep][i%3], edge[x-,y-,y+][(i//3+i%3)%3]. Acceptance
may skew actual training distribution; report it, not silently rebalance.
C reuses first150 B normal train +15 B validation examples, adds50/5 recovery
examples from8001..8400 and9001..9040. Recovery kinds rotate every3scenes:
stop_short,wrist_offset,blackout,object_shift. Max440 new attempts +165 reused
accepted episodes is below880; reuse/failure cost reported separately. These
pools are disjoint from development, qualification, selection and confirmation.
Model memory preflight uses ONLY the qualification-derived schema fixture, one
forward/backward, zero optimizer steps or saved weights. Effective batch64 is
fixed; microbatch/accumulation may be adjusted solely for memory before training.

Within the existing60-case selection budget, use48normal +12recovery scenes.
Within100-case confirmation use88normal +12recovery. The last12 seeds comprise
three cases each of stop_short,wrist_offset,blackout,object_shift, covering all
three strata; all earlierseeds are normal. Report total(primary), normal and
recovery separately. This fixes the previously unspecified secondary recovery
measurement without adding evaluations. Recovery intervention timing in VLA is
triggered by measured robot-relative arm extension/rise, then1.8s, never an
expert-phase input. The scorer requires intervention to occur before accepting
a recovery success. All intervention/reset truth remains outside predictor.

Storage-only pre-training setting: save_total_limit=1, retaining the most recent
optimizer/RNG checkpoint every500steps and final3000. Final weights also saved
by Trainer at modelroot. No additional training/checkpoint selection. Held-out
validation diagnostic uses3fixedtimes per20episodes on the finalmodel only,
reports per-dimension actionMAE; winner still chosen byclosed-loopresults.

## Review correction before first VLA update

Independent review found a15 could emit zero pelvis velocity when its internal
reference reached stand but the measured body remained>10cm short. A deterministic
held-plant regression reproduces this. This violates the design even though
a15's normal expert qualifications passed. Retire in-progress cohort_B; preserve
every attempt and debit its physical executions against B's880-attempt budget.
No training optimizer update has happened. Register bounded repair candidatea16
(12paired development scenes), adding measured-distance moving futures only
outside the10cm arrival corridor in walk/reach. Keep all othertask parameters.
Require fresh30qualification seeds1601..1630,12recovery1701..1712 and same-runtime
replay. No new VLA arms, optimizer steps orselection/confirmationcases.

Also record two startup rows fromt1.60 before task starts1.68 in all formal arms,
matching VLA's first observation and compensatedchunk origin. Fix completion
markers, interrupted-attempt accounting and explicit unsupported-recovery replay
rejection. These repairs require a new cohort ID; do not overwrite the retired
a15 data. Earlier qualified/script/baseline results remain in ledger.

Base model references pinned beforetraining in base_model_snapshots.json:
GR00T2fc962b973bccdd5d8ce4f67cc63b264d6886495;
Cosmos9ce19a195e423419c349abfc86fd07178b230561.
Training uses those existing local snapshots, local_files_only=True. These are
the cached weights used in preflight; no model/hyperparameter candidate change.

VLA posture audit uses robotanglesduringthefirst0.8s after the firstcommanded
gripper decrease>2mm from open as the close window. In high_near its median
absolute torso pitch>15deg counts as an upright violation in the registered
posture tie-break. This is command-derived audit timing, never a VLAinput.
Report stationary_without_close (last1s base speed<.02m/s and no closecommand),
actual inference latency and visibility separately; they do not replace task
success as primary metric. Defined before any formaltraining ormodel evaluation.

## Training startup compatibility repair (20:43UTC, zero optimizer updates)

First A17 initialization failed in OmegaConf structured-config serialization:
upstream TrainingConfig declares resume_from_checkpoint asbool, whereas launcher
assignedNone (and acceptedstringpathsforresume). No model/pipeline setup or
optimizerupdates occurred. CPU regression reproducesfresh/resumefailures and
passesafterusingFalse/True. Explicit --resume must match latestcheckpoint;
Trainerthenloads it withoptimizer/RNG. Dry-run nowvalidatestheactualOmegaConf
serialization boundary. Allsix initialfreezes andfailedstage arearchived in
training_startup_failure_01; comparison confirmed onlyNone→False compatibility
change, no scientificconfig changes. Refreezeall6 before retryA17; completed
collection/export/stats arenotrepeated. Oneinfrastructurefailure,0stepsdebited.

## Validation startup repair (2026-09-09 01:28UTC)

A17 completed all3000 updates. First evaluation initialization loaded the model
and computed one fixed validation prediction, then failed before any closed-loop
case: validation_diagnostic removed action from the same dictionary retained by
LeRobotEpisodeLoader, so scoring labels were absent. Preserve the failed source,
config, log, stage ledger and cost in evaluation_startup_failure_01 and the
pipeline infrastructure_attempts. Zero optimizer updates or closed-loop cases;
one additional diagnostic prediction, not a new model or selection experiment.

Give the label loader an independent deep copy, retaining the observation-only
config used for prediction. Real20-episode/60-frame regression reproduces KeyError
before repair and verifies raw-label MAE parity afterward; prediction inputs
still exclude action labels. Scientific configs, model weights, scene pools,
seeds and inference/executor behavior are unchanged. Retry only the failed
selection stage; completed training and all preceding stages are preserved.
Real-model validation subsequently completed60/60 finite predictions and the
original60-case closed-loop evaluation started normally.
