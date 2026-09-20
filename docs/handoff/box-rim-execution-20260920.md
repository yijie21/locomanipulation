# Box rim execution pilot — 2026-09-20

Branch: `feat/box-rim-execution-20260920`, based on `7710591`; integration target is
`main` working tree. This extends the [static planner](box-rim-planner-20260920.md).
Current usage and scope: [execution guide](../box_rim_execution.md).

## Scene and protocol

Artifacts are under `outputs/box_rim_execution_20260920/`. The source logistics
scene is preserved. The runtime snapshot wraps the existing platform collision
geom in a named static body and adds metadata; it does not change its geometry.
`scene_validation.json` records the loaded box mass (1.2 kg), local COM, friction
and five active collision geoms. Actual box wall thickness is 0.010 m.

`validation_protocol.json` declares a near initialization and the original farther
initialization. Reset is allowed only before the attempt; all subsequent stance
correction must be actual SONIC movement. No formal dataset collection is enabled.
Both live processes are owned by the existing managed Stack launcher.

## Preserved implementation pilots

- `near_stance/result.json`: initialization and admission ran, but a diagnostic
  `phase` field collided with the journal method argument. The integration error
  masked the underlying admission timeout. Source video is preserved. This run
  contains no grasp-success evidence.
- `near_stance_v2/result.json`: after the journal fix, admission recorded 64
  samples and exited `robot_fallen`. The first recorded pelvis was
  (0.0066, -1.0165, 0.7867) m, speed 0.0355 m/s and tilt 0.0170 rad; the last
  was (0.7110, -1.3142, 0.4673) m, speed 2.6227 m/s and tilt 1.0838 rad.
  No approach or close phase was reached. This motivates checking the hold
  command's measured-joint velocity feedback; it does not establish causality.

`fixed_root_diagnostic.json` is an offline check, not a dynamic result: the
previous selected targets, seeded with their planned joint solution, leave
201 mm hand-position residual at the original root y=-1.205 m and 20.3 mm at
root y=-0.980 m. Thus static whole-body feasibility cannot be substituted for
feasibility of a fixed-root arm-only reference.

- `near_stance_v3/result.json`: holding a fixed upper-body reference with zero
  commanded velocity passed admission and exited `no_safe_reposition`. The actual
  planning root was (0.02869, -1.03536, 0.78820) m, with approximately 108° yaw.
  `measured.plan.json` reproduces initial `no_coarse_ik_candidate`. Five offline
  preview replays found coarse candidates but rejected every refined candidate.
  For the forward 0.10 m preview, candidate `x:-0.0615` was rejected only because
  foot-floor contacts compressed 2.7–3.6 mm, exceeding the static 2 mm allowance.
  Candidate `x:-0.1230` additionally had a real non-pad box-wall contact.
  These replays motivate a bounded measured-support allowance, not relaxing
  arbitrary scene collisions. `source_snapshot/` preserves this run's code.

- `near_stance_v4/result.json`: bounded baseline foot-support checking removed
  the false floor rejection. This different settled state had approximately 119°
  yaw; five body-relative previews still had real shoulder/torso or finger/wall
  collisions, so execution again refused before moving. Six full planning records
  are retained in the result. Offline `partial_10.plan.json` instead translates
  0.10 m toward the box and corrects yaw by at most 0.35 rad; it finds two static
  valid candidates with the same collision/acceptance thresholds. This motivates
  box-relative candidate generation; it is not evidence of physical repositioning.

The support change compares the same measured foot/environment geom pair and
upward normal, with an absolute baseline compression cap and a cap on additional
compression. Configurable values live in `PlannerOptions`; arbitrary overlaps and
non-pad target contacts remain rejected. The replay
`near_stance_v3/preview_0.support_baseline.plan.json` accepts `x:-0.0615` while
preserving `x:-0.1230`'s non-pad wall rejection.

- `near_stance_v5/result.json`: first physical reposition attempt. The selected
  target was (0.0096, -0.9077) m from approximately (0.0119, -1.0057) m.
  It exited `reposition_travel_budget_exceeded` at 0.23684 m observed path length
  (22 cm configured budget, sampled detection). The robot reached the target's
  forward coordinate but had lateral drift; chasing a strict XY-radius stop
  condition caused overshoot. `result.trace.jsonl` retains 23 movement samples.
  This motivates progress-based braking and replanning from the actual stop.
  No arm approach or grasp was attempted.

- `near_stance_v6/result.json`: target-plane braking triggered at
  (0.05369, -0.89685) m, before the target (0.06412, -0.87214) m. The robot then
  coasted during `reposition_settle`; the observed path reached 0.22437 m and
  the same budget guard stopped the attempt. This distinguishes the brake-command
  location from a quiet stopped stance. Speed-dependent earlier braking remains
  necessary; no arm approach or grasp was attempted.

## Verification so far

The integrated v6 source passed `make check` (full repository checks and default
CPU suite). The log is `outputs/box_rim_execution_20260920/make_check.log`.
Focused worker planner/execution tests passed 18/18 before integration.
Physical success has not been demonstrated by the above attempts.

- `original_stance/result.json`: the original farther initialization, tested on
  integrated v6, exited `no_safe_reposition` before moving. Full current-state and
  hypothetical-stance planning diagnostics are retained. This is a finite-search
  refusal under the configured movement budget, not a proof of impossibility.

- `near_stance_v7/result.json`: speed-aware braking fired with 0.148 m projected
  distance remaining and 0.301 m/s measured forward speed. The chosen move was
  0.20 m because this initialization settled farther back. The robot approached
  the target, but gait sway/coasting consumed 0.22347 m path before quietness;
  the unchanged 0.22 m travel gate refused it. No settled stop is claimed.
- `near_stance_budget30/result.json`: explicitly separate 0.30 m travel-budget
  diagnostic, with unchanged collision/grasp gates. It achieved an actual quiet
  stop at (0.01920, -0.89337, 0.78792) m and replanned successfully. Observed travel
  was only 0.15252 m, so this particular run did not consume the extra allowance;
  initialization differed and it is not a paired proof that the larger budget
  fixed braking. The next gate rejected the pregrasp joint interpolation at
  fraction 0.727: left fingers crossed box walls by 0.53–0.94 mm. No hand approach
  command was issued. `measured_stopped_pose.png` reconstructs the final measured
  state with actual jaw positions; it is not a planned ghost or grasp result.

## Supported approach development

The saved quiet state from `near_stance_budget30` exposed two separate issues.
Using the measured fully open Dex1 joints also detects a small right finger/hip
contact at the initial pose; neutral jaw joints remove it without a contact waiver.
A fixed-root raised-front → above-rim → pregrasp → grasp route cleared collisions,
but moved the robot COM by 26–45 mm, exceeding the unchanged 15 mm limit.
It was rejected rather than executed.

The replacement connects whole-body waypoints with foot anchors, COM and joint
margin checks, opens the jaws after raising them, and checks sampled segments with
their actual or commanded aperture. Planned pelvis translation is a feasibility
prediction, never a root-position command. Only measured tracking can validate it.
Offline admission and physical grasp success remain separate results.

The exact-state replay is preserved as
`near_stance_budget30/supported_route_replay.json` (411,052 bytes), with matching
source hashes and the replay script alongside it. Four phases each have six
waypoints. All 288 segment samples and 12 jaw-aperture samples passed; maximum
COM shift was 13.723 mm, maximum foot-anchor error 0.290 mm, and final hand
position residual 0.0041 mm. Solving took 1.683 s; the complete replay took 3.56 s.
These are offline results, using hypothetical neutral jaws at the saved body state.

The frozen implementation passed 23 focused tests and the full `make check`
(`make_check_supported.log`, exit 0). The execution watchdog now requires advancing
simulation time, and request validation binds the observation key to the target
body in scene metadata.

## Pre-follow-up live validation and next action

`near_stance_supported/result.json` used the frozen integrated source and
`near_stance_supported.request.json` (only the recording label changed from the
budget30 request). It achieved a real settled stop at
(-0.009408, -0.909656, 0.788013) m after 0.174834 m of observed travel, then selected
`x:-0.1230:+0.0000:+0.0000:0.0060:+45.00` from fresh measurements.

The route was rejected before any approach command:
`unsafe_swept_reference`, `raised_front`, step 1, fraction 0.0. The measured
starting pose had right hip / right wrist penetration 0.09256 mm and right hip /
right finger penetration 0.00708 mm. These remain disallowed self-collisions.
No closure or lift was attempted; no physical grasp success is established.
This different stopped posture does not invalidate the earlier saved-state
kinematic replay, and that replay does not establish transfer to this stop.

Next action is to make the walking/ready upper-body posture clear the hips with
positive margin across achieved stops, then repeat supported-route execution
with the existing gates. Do not waive initial contact or infer success from IK.

Source hashes in the live result match the integrated files. `source_snapshot/`,
managed logs, request, trace and raw recording are preserved in the run directory.
The result was written before recorder shutdown and lists no video; the finalized
raw file is `videos/near_stance_supported.mp4` (1,886,501 bytes), preserved and not
exported for viewing. No video preview was delivered. Managed GPU 0 / instance 2
exited and was released; no other workload was stopped. Original scenes, failed
pilots, datasets and weights remain intact. No commit or push was made.

## Follow-up: visible grasp attempt and room camera

User requested an actual grasp attempt and the third-person camera used by
`outputs/logistics_room/renders/pair_mujoco_isaac.png`. The native recorder now
accepts `scene_meta.video_camera`; `scene_reverse_camera/` preserves identical
scene XML and selects `reverse_room`. The real recorder frame in
`camera_validation/reverse_room_recording.png` was inspected: robot and carton
are visible. Thirteen camera/inset tests passed. Default tracking remains intact.
GPU 0 became occupied by another task; subsequent managed pilots use GPU 1,
instance 2, with independent requests and source snapshots.

The initial supported-waypoint arm escape kept measured legs, waist and root.
`near_stance_escape` reached `ready_escape` but strict observed monotonicity
rejected a 0.198 mm returning hip/wrist contact during POSE takeover.
Reference paths still require monotonically resolving exact initial pairs;
measured execution now has a finite tracking envelope of initial depth +0.25 mm,
capped at 0.5 mm, and requires no remaining contact at completion. This allowance
is restricted to the short escape, not normal approach/closure or grasp success.
`near_stance_escape_v2` exposed a bookkeeping bug: a starting finger/hip pair
briefly vanished during takeover and was wrongly classified as a new pair when
it returned at 0.00154 mm. Starting pairs are now preserved across observations.

`near_stance_escape_v3` moved preparation to the legacy PlannerArm IDLE controller
and made real arm motion, but finger/hip contact reached 0.264 mm at 9.4% reference
progress, over its 0.258 mm tracking envelope. Coupled shoulder/wrist references
were replaced by a shoulder-only lateral clearance (+/-0.15 rad), leaving wrists,
waist and legs unchanged before the normal supported reach. This path passed
reference checks from three recorded states; `check_lateral_ready.py` and
`lateral_ready_replay.json` retain the calculation. It is not physical success.
Twenty-six execution/planner tests pass for this revision; the next live run is
`near_stance_escape_v4`.

### PlannerArm execution follow-up results

All attempts below retain their request, result, raw video and source snapshot.
- `near_stance_escape_v4` and `nearer_stance_lateral`: no safe reposition. The
  latter's previews exposed measured sole compression around 7.5 mm above the
  former 6 mm baseline cap. `dense_planning_replay.json` separately reproduces
  refusal with 6 mm and a valid preview with 10 mm; the diagnostic request uses
  9 rim samples and 10 mm support-baseline cap (same pair/normal/growth checks).
- `nearer_stance_lateral_dense`: refused reposition travel budget. No grasp.
- `grasp_stance_lateral`: nearer initialization, yaw pi/3, found a static plan
  without walking; whole-body raised-front route exceeded 15 mm COM-change gate.
- Explicit `execution.approach_mode=planner_arms` now reuses the legacy controller
  for approach, opening, alignment, closure and lift. Legs/root remain measured
  references in IK; physical root translation is never commanded. The default
  `supported_pose` mode remains available. The first two PlannerArm attempts
  stopped at COM-change and joint-margin admission respectively. Exact replays
  of their routes show COM projection within the foot contact hull by at least
  30.3/38.9 mm (`planner_arms_support_replay.json`, `planner_arms_margin_replay.json`).
  The explicit diagnostic COM-change limit is 75 mm, not the default 15 mm.
  Fixed-root IK now reserves the same joint margin checked at admission.
- `planner_arms_attempt_v3`: a shallow initial finger/box contact was refused.
  Preparation therefore now happens before planning/walking, matching the old
  expert's ordering, using shoulder-only clearance via PlannerArm.
- `planner_arms_ready_first`: actual clearance reached 85% before measured foot
  motion 5.025 mm exceeded the static 5 mm gate. Measured tracking tolerance is
  now an explicit separate execution field (default still 5 mm); this diagnostic
  uses 20 mm, while planned geometry retains its 5 mm foot-anchor threshold.
- `planner_arms_ready_first_v2`: actual preparation, 0.112 m reposition, raised
  hands, opened jaws, and started above-rim approach; stopped at 35.93 mm hand
  error versus 35 mm. New PlannerArm mode now reprojects each waypoint at the
  latest measured root and waits 0.25 s at each commanded target.
- **`planner_arms_ready_first_v3`**: latest live, no reposition needed, completed
  clearance, all six raised-front waypoints, jaw opening, and five above-rim
  waypoints. At above-rim waypoint 6 it stopped with 49.915 mm hand position error
  (orientation 0.173 rad); support gates passed (feet 10.0/14.5 mm displacement,
  COM displacement about 33 mm). **No closure, probe or grasp success.** This is
  a real visible approach attempt, not a ghost animation. Remaining work is
  measured hand tracking through final approach; success must not be inferred
  from the reference reaching the rim.

Latest preview: `planner_arms_ready_first_v3/previews/grasp_attempt.mp4`,
317,651 bytes, 25.2 seconds. It preserves full raw duration at normal speed;
full decode and start/intermediate/end frames were inspected. The raw 7,323,130
byte recording remains in `videos/`. The main view uses the same reverse_room
camera as the paired MuJoCo/Isaac still, with unchanged wrist/ego insets.

Final validation for this follow-up: 27 focused execution/planner tests and 13
camera/inset tests passed; full `make check` exited 0
(`make_check_camera_planner_arms.log`). The final live result's source hashes
match the integrated files. Managed GPU 1 / instance 2 stopped; other GPU jobs
were left running. All failed trials and raw videos remain preserved. No commit
or push was made. The next action is feedback correction of measured hand
tracking through above-rim/pregrasp, followed by physical closure validation.

## Follow-up: legacy low-reach physical attempt

`legacy_low_reach_attempt_v6` used the opt-in `execution.approach_mode=legacy_low_reach`; the default remains `supported_pose`. The controller performed the initial lateral escape, tracked all six PlannerArm upright-clearance waypoints, replanned from the fresh measured state, entered StreamedArm once, and completed the historical smootherstep whole-body bend in 4.04 s. This satisfies the requested physical bend/reach attempt; it does not demonstrate a grasp.

The trace contains 100 `low_reach_bend` samples from simulation time 11.40 to 15.36 s. During that phase, measured torso pitch reached 22.00°, pelvis pitch reached 19.61°, waist pitch ranged from 2.12° to 9.30°, and pelvis height fell from 0.7882 m to a minimum of 0.7773 m. The following held-posture `transit_open` samples reached 22.64° torso pitch and 20.09° pelvis pitch. Left and right palms moved 46.95 mm and 33.64 mm during the bend. The box moved less than 1 nm in the trace and reported no touch/contact links.

The next `above_rim` admission stopped at interpolation fraction 0 with `unsupported_swept_reference`. Its reported 20.40/19.48 mm foot-anchor deltas came from `_admit_supported_segment`'s hypothetical reference FK and its static 5 mm planning tolerance. That calculation mixed measured start joints and pelvis translation with the held commanded root quaternion. It is an unresolved reference-admission mismatch after the bend, not evidence that the observed feet physically slipped by 20 mm; the preceding live observed-support checks passed. Closure, contact probing, and lift were never attempted, so `produced=false` and no grasp success is claimed.

Preserved evidence:

- Request: `outputs/box_rim_execution_20260920/legacy_low_reach_attempt_v6.request.json`
- Result and full trace: `legacy_low_reach_attempt_v6/result.json` and `legacy_low_reach_attempt_v6/result.trace.jsonl`
- Source snapshot manifest: `legacy_low_reach_attempt_v6/source_snapshot/manifest.json`
- Motion audits: `legacy_low_reach_attempt_v6/motion_audit.json` and `legacy_low_reach_attempt_v6/main_motion_audit.json`
- Raw recording: `legacy_low_reach_attempt_v6/videos/legacy_low_reach_attempt_v6.mp4`, 4,222,125 bytes, 15.04 s
- Full-duration normal-speed preview: `legacy_low_reach_attempt_v6/previews/legacy_low_reach_attempt_v6_full_preview.mp4`, 394,984 bytes, 15.00 s, H.264 640x480 at 10 fps. Full decode passed.

The final source passed 31 focused box execution/whole-body tests and a subsequent full `make check`. The live run did not modify source. GPU 1 returned to 3 MiB, the instance-2 lock was free, and no owned stack process remained; GPU 0's unrelated job was untouched. No commit or push was made.

## One-only 10 cm closer-start attempt

The user authorized exactly one execution of the unchanged current policy from 10 cm closer to the box, with no tuning or retry. `legacy_low_reach_closer_10cm_once.request.json` was cloned from the v6 request. An exhaustive parsed-request comparison found only the authorized changes: `runtime.robot_dxy` changed from `[0, 0.275]` to `[0, 0.375]`, and `runtime.video_label` changed for artifact bookkeeping. Yaw, scene, camera, backend, planner, execution settings, thresholds, and policy were identical. All nine relevant source hashes matched the v6 snapshot before launch, and the new request and sources were snapshotted in the run directory.

Exactly one managed episode was launched on GPU 1 / instance 2. During settling, the left wrist was already against the box: 19 of 21 admission trace samples reported the `L.wrist_yaw` contact link. Relative to v6's planning snapshot, the measured pelvis settled at `(0.01249, -0.82302, 0.78653)` m, a delta of `(0.02226, +0.15257, -0.00180)` m. The box settled at `(0.01653, -0.41923, 0.37098)` m, displaced `(0.01653, -0.02923, -0.00001)` m from v6. The robot-box planar distance was 0.40381 m versus 0.58567 m in v6. These are observed settled states; the requested spawn offset itself changed by exactly +0.10 m in world y.

The strict initial-escape gate returned `initial_escape_not_admissible` in `planning`, before any escape, bend, arm reach, closure, probe, or lift command. It rejected a 0.431 mm non-pad collision between the box's `nor_cal_ym` geom and `left_wrist_yaw_link`, together with a 0.095 mm `right_hip_roll_link`/`right_wrist_yaw_link` self-contact. `produced=false`; no success is claimed. Per the one-run instruction, there was no retry or configuration change after this outcome.

Preserved artifacts:

- Request: `outputs/box_rim_execution_20260920/legacy_low_reach_closer_10cm_once.request.json`
- Result and trace: `legacy_low_reach_closer_10cm_once/result.json` and `legacy_low_reach_closer_10cm_once/result.trace.jsonl`
- Source/request snapshot: `legacy_low_reach_closer_10cm_once/source_snapshot/manifest.json`
- Single-attempt audit: `legacy_low_reach_closer_10cm_once/single_attempt_audit.json`
- Raw video: `legacy_low_reach_closer_10cm_once/videos/legacy_low_reach_closer_10cm_once.mp4`, 510,161 bytes, 2.08 s
- Full-duration normal-speed preview: `legacy_low_reach_closer_10cm_once/previews/legacy_low_reach_closer_10cm_once_full_preview.mp4`, 458,503 bytes, 2.08 s, H.264 720x540 at 12.5 fps. All 26 frames decoded, and start/middle/end frames were inspected in `previews/verification_frames.png`.

The managed stack exited after the single result. GPU 1 returned to 3 MiB, the instance-2 lock was free, and no owned stack process remained. GPU 0's unrelated workload was untouched. No source change, test run, commit, or push was made.

## Approved square-ready legacy-low-reach execution

The approved static review configuration is
`square_ready_initial_review/proposed_initial_state.json`: requested pelvis XY
`(0, -0.83)` m at yaw pi/2, zero waist, symmetric low arms, and open hands.
Runtime initialization commands only those 17 upper-body joints while retaining
SONIC's live settled legs. It pins the upper body across reset and same-channel
PlannerArm takeover, then releases the pins before recording. The first launch,
`legacy_low_reach_square_ready_once`, failed before recording and before
`BoxRimExecutor.execute()` because initialization passed streamed timing metadata
through the inherited `PlannerArm.hold()` contract. Its result and snapshot remain
unchanged. The two initialization holds now use the supported legacy PlannerArm
call shape; a real-class inheritance regression covers that dispatch. The focused
execution suite passes 25 tests, and final `make check` passes in
`make_check_square_ready_once.log`.

`legacy_low_reach_square_ready_execution.request.json` changes only the video
label from the failed pre-episode request. Its source snapshot and manifest are in
`legacy_low_reach_square_ready_execution/source_snapshot/`; all planner,
execution, collision, backend, pose, and threshold values are unchanged. This was
the single actual grasp episode. The measured initialization heading was
1.54596 rad (88.577 degrees), 1.423 degrees below the requested 90 degrees, and
the largest upper-joint error was 0.05665 rad (3.246 degrees). Active open-hand
error was at most 0.187 mm. The pelvis settled at
`(0.01225, -0.92107, 0.78245)` m, 9.19 cm in XY from the requested pose, mostly
9.11 cm farther back. The box shifted 9.59 mm from its requested position. Thus
the run validates the explicit upper pose and heading initialization path, but is
not an exact physical reproduction of the static reviewed state.

The unchanged strategy selected
`x:-0.1230:+0.0000:+0.0000:0.0060:+45.00`, completed all six
`low_reach_arm_clearance` waypoints and the 4.06 s `low_reach_bend`, then stopped
before the first above-rim waypoint with `unsupported_swept_reference` at
fraction 0. The rejection reported hypothetical-reference FK foot-anchor errors
of 15.894 and 14.157 mm against the unchanged 5 mm static gate. This is not a
measurement of actual foot slip; the preceding live support checks passed. No
closure, contact probe, lift, or grasp success occurred. No retry or strategy
tuning was performed.

The full result and 452-sample trace are
`legacy_low_reach_square_ready_execution/result.json` and
`result.trace.jsonl`. The 4,090,938-byte source recording is preserved at
`videos/legacy_low_reach_square_ready_execution.mp4`. The normal-speed,
full-duration delivery preview is
`previews/legacy_low_reach_square_ready_execution_preview_crf27.mp4`: 369,187
bytes, 14.80 s, 640x480, 10 fps, and 148 frames. Full decode passed; source and
preview durations are both 14.80 s; start, middle, and end frames were visually
checked in `previews/main_check_frames.png`, with the reverse-room view and
wrist/ego insets visible. Hashes, probe metadata, initialization measurements,
outcome, and resource release are recorded in `artifact_manifest.json`. GPU 1
returned to 3 MiB, instance 2 is unlocked with no stack state, and GPU 0's
unrelated workload was left untouched.


## Saved-attempt diagnosis: preparatory closure and blocked approach

Read-only CPU FK audit of the square-ready attempt is preserved in
`legacy_low_reach_square_ready_execution/cause_audit.py` and `cause_audit.json`
under the output root above. No policy change or new simulation was made.

The prior statement “no closure” means no formal grasp-close phase. Preparation
does explicitly command `HAND_READY=0` in arm-clearance and bend. Dex1 collision
mesh geometry gives a jaw gap of 45.877 mm at zero, versus 94.877 mm at the
initial open command (0.0245 m per finger). Measured joints follow this
open → partly closed → open sequence.

Initialization ends 91.1 mm behind requested pelvis y; admission then advances
it, leaving the planning snapshot 45.5 mm behind requested y. During recorded
bend, pelvis y retreats 42.87 mm but ankle world displacements are only
2.19/2.47 mm. `travel_m=0`: this is not a commanded backward walk. Initializer
heading/upper-pose holds do not guarantee pelvis XY, and there is no final XY
correction. The saved observations do not isolate each pre-recording source of
drift, nor prove that a closer stance is infeasible.

Static IK was admitted. The first above-rim reference check stops at fraction
zero, before approach/descent. It combines measured joints/root translation
with held commanded root orientation. Replaying the last saved observation
with the held bend root (1.2012 degrees from observed orientation) produces
15.905/14.156 mm FK ankle deviations, closely matching logged failure
15.894/14.157 mm against the static 5 mm gate. This replay is near the boundary,
not the exact unrecorded gate observation. These are hypothetical-reference
deviations, not measured foot slip and not an IK-infeasibility result.
The separate observed-motion tolerance is 20 mm.

The opening trace currently records observed root orientation in its reference
field, although the command uses the held bend root. The audit therefore uses
the bend reference explicitly; this trace discrepancy is not fixed in this
diagnostic-only task.

Annotated full-duration preview:
`legacy_low_reach_square_ready_execution/diagnostic/legacy_low_reach_square_ready_diagnostic_full_preview_v2.mp4`,
319,297 bytes, 14.80 s, 640x480, 148 frames at 10 fps. Actual size, duration,
full decode and four decoded phase frames verified. Displays commanded/measured
jaw gap, pelvis drift, projected planned/observed palm points and retrospective
final rejection reason. Approximate trace/video synchronization uncertainty is
±0.08 s. This is an annotation of the existing attempt, not a new episode.
`make check-docs` and `make check` passed after the diagnostic documentation.


## Original coordinated expert migration: one square-ready attempt

Working tree: `main`, base commit `7710591`; no commit or push. The new opt-in
`execution.approach_mode=coordinated` uses the actual `VisualApproach` and
`CoordinatedReach` classes. Object key, box-bottom-local grasp points, capture
axes and bounded along-rim region are explicit inputs. Historical crate defaults
remain compatible. Targets are simulation truth; camera visibility admission has
not been migrated. Nonempty forbidden rim intervals are explicitly rejected.

Coordinated initialization prepares the configured upper pose with OPEN hands,
then resets/reseeds the requested stance a second time before releasing pins and
recording. No reset or object attachment occurs during manipulation. Stance
feedback respects the measured travel budget. Reference joint/root/position
checks use one coherent solution; measured checks use measured state. Closure
requires measured capture readiness, and lifting retains bilateral contact,
contact-duration and slip checks.

Validation: original/controller/approach/box focused tests, two initialization
consumer cases, a real Coor/BimanualIK pitch-versus-all integration regression,
and final `make check` passed. The real-scene CPU replay of a saved observation
entered reach with OPEN hands, micrometre-scale IK residuals and no rejected
reference contacts. This is a kinematic check, not a dynamics success. It also
exposed and led to fixing inconsistent pitch-only solver bounds versus an
all-waist controller before the live attempt.

Exactly one managed attempt used
`outputs/box_rim_execution_20260920/coordinated_square_ready_once.request.json`,
with explicit planner `waist_dofs=all`, `waist_pitch_max_rad=0.52`, existing
reverse-room scene/camera, GPU 1 and instance 2. GPU 1 was shared with an unrelated
training process at launch; that process and GPU 0 were not modified. The attempt
ran 2026-09-20 09:08:54–09:10:01 UTC including stack startup and teardown.

Measured initialization: requested pelvis `(0,-0.83)` m; actual
`(-0.03541,-0.83496)` m, yaw error −6.29 degrees. Thus forward/back placement
improved to 4.96 mm, but lateral/yaw placement was not exact. Both grippers
remained open at approximately 94.9 mm. Two initialization resets were recorded.

Outcome: `coordinated_collision`, before the first continuous reach reference
was sent. Static and original-controller admission both passed. The retained
reference collision check rejected right ankle/floor contact: reference
penetration 2.5966 mm versus measured baseline 0.7679 mm, an increase of
1.8287 mm above the 1.5000 mm growth allowance. Rejected measured contacts were
empty. This differs from the earlier mixed-root foot-anchor rejection: the
reference here is coherent, but fails a reference contact-depth gate. The
result does not prove actual collision, slip, IK infeasibility, or safe execution
of the unissued motion. No reach, close or lift was delivered; no retry or gate
relaxation followed.

Artifacts under `coordinated_square_ready_once/`: `result.json`,
`result.trace.jsonl`, `launch_receipt.json`, `source_snapshot/manifest.json`,
`resolved_control_config.json`, `cpu_saved_state_check.json`, `make_check.log`,
`artifact_manifest.json`, source recording and full `state_trace/*.npz`.
Production sources were unchanged between launch and delivery. Instance 2 lock
was released, stack state removed, and both owned child processes exited.

Full-duration diagnostic preview:
`diagnostic/coordinated_grasp_diagnostic_preview.mp4`, 378,940 bytes, 3.36 s,
42 frames at 12.5 fps, 768x576. Full decode and decoded start/middle/end frames
passed inspection. Source video frame k maps to physics state `seq[3+4*k]`;
measured jaw opening, pelvis position and grasp centres use exact-frame FK.
Selected targets and the final rejection card are retrospective labels.

Next action: review the applicability of the reference foot/floor penetration
growth gate at planner-to-POSE handoff before another authorized physical trial;
do not interpret the gate as a measured collision or simply widen it to force
progress. Post-handoff continuous tracking and physical grasp remain unverified.

Independent post-run audit: `motion_audit.json` and `motion_audit.md`. All nine
stance samples commanded zero walking speed; `walk_step` was not called. The
4.988 mm reported stance travel is measured pelvis settling motion, not a
commanded walk. No coordinated `pose_step` occurred before the rejection.


## Original-expert support policy and one attempt (09:41 UTC)

Branch `main`, base `7710591`, uncommitted shared working tree. User authorized removing
new support rejection rules and exactly one new attempt. Coordinated mode now selects
`diagnostic_floor` in static planning and observed/reference collision checks: only
foot/ankle contacts against the structurally verified lowest active horizontal world
plane are support diagnostics. Per-tick observed/reference foot-anchor errors remain
logged but no longer abort. Original initial `WholeBodyPose.reachable` admission and
continuous IK foot/COM residuals remain; non-floor, non-foot and self collisions retain
rejection. Other approach modes keep `bounded_growth` defaults.

Validation: 60 focused tests, scoped Ruff/format, and final `make check` passed. The
first final check found an import-order issue, fixed before the successful check and
launch. CPU replay of the previous floor rejection reproduces its reference/baseline
boundary within 4.266 micrometres (nearest saved state, not the exact unrecorded state);
old policy rejects 1.826 mm growth, new policy records it without rejection.

One managed launch only: `coordinated_expert_support_once`, GPU 1 / instance 2,
2026-09-20 09:40:40–09:41:38 UTC. Request is identical to
`coordinated_square_ready_once` except video label. **Failed in initial standing
admission with `robot_fallen`, before planning or controller handoff.** All 32 trace
rows are `admission`; no reach, closure, or walking command was issued by that path.
Hands stayed OPEN (measured 94.7–95.4 mm in command trace), no bilateral box contact,
and box displacement was below 0.001 mm. Pelvis retreated 0.713 m between first/last
logged samples and fell from 0.757 to 0.422 m; this was observed instability, not a
commanded reposition. The unchanged `hold_ready`/`_settle` paths were verified against
the previous source snapshot. This does not establish the cause of the instability;
control/reset timing and effects of shared GPU load have not been isolated.

Artifacts under `outputs/box_rim_execution_20260920/coordinated_expert_support_once/`:
`result.json`, `result.trace.jsonl`, `motion_audit.json`, `support_gate_replay.json`,
`source_snapshot/manifest.json`, `launch_receipt.json`, `make_check.log`,
`focused_checks.log`, `stack_sim.log`, `stack_deploy.log`, and original video/physics
trace. Full-duration diagnostic preview:
`diagnostic/expert_support_grasp_preview.mp4`, **386,171 bytes**, **1.36 s**, 17 frames
at 12.5 fps. Full decode passed and four decoded start/middle/end frames were reviewed;
readouts use exact-frame physics (`seq[3+4*k]`), with command-trace sample age shown.
`diagnostic/verification.json` records export checks.

Owned stack stopped; PIDs 16674/16737/17058 absent, instance lock free and stack state
removed (`resource_release.json`). Unrelated GPU workloads were not stopped. No
additional trial or threshold tuning was performed. Next action: diagnose the initial
unpin/reset-to-standing instability before another physical trial. Live reach with
the corrected support handling is **not yet validated**.


## Standing-fall investigation (09:56 UTC)

User requested diagnosis, not another grasp trial. Read-only CPU investigation; no
production changes or additional live launch. Evidence is in
`outputs/box_rim_execution_20260920/coordinated_expert_support_once/standing_diagnosis/`.
The shared deploy log appends across launches: `failed_deploy.log` and
`previous_deploy.log` isolate its last two process starts, verified against receipts.
Both runs used identical scene, deploy binary, encoder and decoder hashes.

**Observed:** failed recording starts with world-y pelvis velocity -0.387 m/s,
pitch -8.49 degrees and height 0.752 m; previous run starts at -0.089 m/s,
-3.17 degrees and 0.783 m and recovers. The failed run's COM falls progressively
behind the feet; at 0.8 s after first saved state it is 0.131 m behind mean ankle y.
No reach/planning/support-policy path ran. Physics reconstruction and four comparative
plots are in `physics_comparison.json` and `standing_comparison.png`. Contact geometry
is CPU reconstruction, not recorded contact forces.

**Confirmed initialization defects / causal candidates:**

- `box_rim_execution_cli.py:288-305` sends the second reset, sleeps 0.3 wall seconds,
  then calls `PlannerArm.start(..., settle_seconds=0)` without confirming a post-reset
  observation. `run_live:343-351` releases pins, holds 0.8 wall seconds, reports
  initialization and starts video. Its `standing_legs` metadata says live settled
  despite lacking a completed measured-quiet admission there.
- `EpisodeControlClient.reset` (`sim/privileged_pub.py:650-675`) only PUSH-sends;
  simulator `_apply:487-538` resets MuJoCo pose/velocity but has no acknowledged epoch
  or coordinated free-running deploy reset. Deploy observation history and previous
  action/planner state survive this physical teleport. Existing lockstep
  `ResetLockstepSession` explicitly clears/reseeds those states; that path is not used.
- `PlannerArm.start:779-817` documents immediate takeover for an already settled
  owned channel; it reads one observation for root/facing/joints with no reset
  freshness barrier. With settle_seconds=0 its hold loop sends no frame. The caller
  releases pins on a separate port before the subsequent hold sends the next planner
  command. The ordering vulnerability is confirmed; actual cross-port scheduling
  in the failed run is unrecorded.

**Original expert comparison:** `expert.py:91-116` also uses a second reset and short
sleep, so it shares the underlying reset discontinuity. It does not introduce the
box wrapper's arm/waist pin-release transition and requires upright pose, z >= 0.72 m
and speed < 0.06 m/s for a continuous measured interval before stance-ready. Replay
of those conditions over the available saved samples (`quiet_criteria_replay.json`)
finds only one quiet sample and no ready interval in the failed run; previous samples
support a 1 s quiet interval by simulation t=2.20 s. Sparse recorded-sample replay is
not a counterfactual simulation. Adding this gate alone would reject an unstable
initialization; it does not prove the physical fall would disappear, because the
box executor already commands a ready hold during admission.

**Limits:** the evidence localizes the failure to reset/standing takeover and proves
the above synchronization defects, but cannot uniquely attribute the backward
actuator response to retained deploy history, reset phase, or pin-release ordering.
Pre-reset/control-action traces and reset/frame acknowledgments were not recorded.
Yaw offset is compensated by the commander; the roughly 12-degree streamed root
and roughly 70-degree planner facing are not by themselves a yaw bug. Sparse
LowState-age maxima were 143.9 ms in the previous stable run and 69.3 ms in the failed
run; admission wall intervals were similar. GPU-load causation is unsupported.

Next correction belongs to initialization/control synchronization: confirm reset
and fresh state, coordinate deploy history/reference reseeding, publish/confirm a
planner command before release, then apply the original measured-quiet admission.
Do not treat a longer sleep or a relaxed fall threshold as a demonstrated repair.
A subsequent diagnostic should record the entire reset-to-release transition and
consumed controller inputs/actions before evaluating physical standing recovery.


## Unchanged-code repeat (10:00 UTC)

User explicitly authorized one more run without code changes.
`coordinated_expert_support_repeat_once` ran once on GPU 1 / instance 2; source hashes
match `coordinated_expert_support_once` and request changes only `video_label`.
Standing admission passed at sim t=1.62 s, then approach stopped with
`reposition_travel_budget_exceeded`, cumulative measured pelvis path 0.300160622 m.
Trace has 30 admission and 42 approach rows; no selected grasp plan, reach, closure
or lift. No threshold change or further retry. Full-duration diagnostic preview
`diagnostic/expert_support_grasp_preview.mp4`: 352,709 bytes, 2.88 s, 36 frames at
12.5 fps. Full decode, matching source duration and four decoded frames verified.
Results, exact-state trace, source identity and release receipt retained in the run.
Owned PIDs 44746/44811/45298 exited; instance lock free/state removed. No code edits;
initialization synchronization diagnosis remains unresolved.
