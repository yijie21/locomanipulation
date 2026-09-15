# Bottle central grasp, contact recovery, tracking and stance

Date: 2026-09-15. Implementation branch: `fix/bottle-contact-20260915` in
`/workspace/code/locomanipulation-bottle-contact-20260915`. The baseline is main
at `fcf23d9` plus the entire existing dirty tree, including the preceding
[natural expert implementation](bottle-natural-expert-20260915.md). Preserve
other sessions' changes. Evidence: `outputs/bottle_contact_20260915`.

## Current result and remaining acceptance gaps

Current integrated version: **checkpoint11**. Fresh seeds30700/30800 completed
**10 successes /15 attempts**. Ten full-duration normal-speed previews have been
verified (425454–490786bytes each; ZIP4552364bytes) in
`outputs/bottle_contact_20260915/final_collect_v3/previews/`.

The central-grasp request is supported by reconstructed contact geometry: all10
successful post-handback tails retained bilateral contact, with contact x median
152.682mm against the installed pad midpoint152.674mm. Current depth target is
+1.2mm ±5mm relative to the calibrated tool centre; chronological entries below
include earlier rejected parameter values and must not be used as current config.

High reliability and fully natural motion are **not established**: three wrist/
bottle clearance failures, one finger/table clearance failure and one recovery
slip remain. Reaching takes6.04–12.64s in the ten successful samples, and terminal
handback still has substantial root/hand movement. Primary remaining priorities
are close-pose tracking, measured terminal stabilization with a natural carry
pose, gait-aware stopping/reachability, and force-informed grasp regulation.
No VLA training or learning-benefit validation was performed.

Checkpoint12 was tested on the four fixed scenes and achieved2/4, matching the
original accepted count while regressing one original success. It is **not adopted**.
Main restored checkpoint11 after verifying exact source hashes; candidate12 remains
in its checkpoint snapshot and worker branch for further investigation. Final main
`make check` passed on the restored checkpoint11 worktree; documentation-only
follow-up is checked separately.

## Authorized scope and delivery

The user requested implementation of the three remaining priorities: reduce
actual tracking jitter, coordinate grasp/lift/recovery with contact, and improve
stopping and whole-body posture. The added explicit requirement is to move the
gripper farther forward and close with the bottle in the middle of the pads.
Implementation, focused checks, physical validation and viewable recordings are
available; the high-reliability and natural-motion goals remain unmet. No new VLA training, physics/friction change or artificial attachment
is part of this work.

The main agent owns planning/review, integration, GPU 0 / instance 0, evidence and
documentation. One GPT-5.6 Sol worker owns scoped implementation and focused tests
in the isolated tree; checkpoints are copied only after freezing and verifying
that main still matches the prior source hashes. `baseline.patch` and
`baseline_hashes.json` retain the starting state. Final main validation uses
`make check`; no commit is requested.

## Central grasp and contact checkpoint

The old entry target asymptoted at +18 mm relative to the calibrated pad centre,
and the measured close gate accepted +5 through +20 mm. The calibrated Dex1
centre remains wrist-frame x=0.1515 m. The new whole-body target is depth
-0.002 m with a 0.005 m tolerance; the approach target and measured gate use the
same convention. Other strategies retain their existing geometry defaults.

CPU inspection of the installed hard-pad collision geometry found shared pad
coverage x=0.120588..0.184759 m and palm/base end x=0.110800 m. For the installed
bottle radius 0.0221589 m, the full configured capture interval leaves minimum
axial clearances of 1.7531 mm at the proximal pad edge, 8.1001 mm at the distal
edge and 11.5411 mm at the palm. An incompatible radius/window is rejected.
These one-dimensional checks supplement the existing swept mesh/palm/table
checks; they are not a proof of complete three-dimensional clearance.

Loading and recovery now use physical-time bilateral-contact hysteresis and
filtered bottle speed in the rotating measured palm frame. Progress decelerates
on contact degradation and resumes after stable contact; closure remains held.
Reacquisition is bounded by 0.8 s. The 4 cm slip limit is measured in the palm
frame so rigid wrist rotation is not itself labelled slip. Existing support,
clearance, lift-height and final physical-success gates remain active.

`checkpoint01/` preserves six changed paths. The worker ran 74 relevant tests,
Ruff, generated-config-reference and diff checks. `contact01_pilot4/` replays four
baseline configurations; physical results will be recorded below.

Checkpoint 01 achieved **3/4**, including the baseline recovery-slip case.
Measured depth at closure was +0.989, +2.469, +1.872 and +0.234 mm: all four
entered the new central window. The fourth attempt failed the unchanged
`right:right_wrist_yaw_link:water_bottle_object` clearance guard during closure.
At admission its palm-frame relative speed was 0.0574 m/s, actual tool position
differed from the active command by 4.01 mm, and orientation differed by 9.03°.
This supports adding a distinct capture-settling gate before closure; it does
not justify weakening the wrist guard or returning to shallow tip capture.
Raw manifests, state traces, copied runtime logs and `pilot_summary.json` remain
under `contact01_pilot4/`. The pilot runner returned 1 because four successes
were requested in four attempts; collection and owned cleanup returned 0.

## Baseline replay and stopping evidence correction

The preceding natural expert collected 10/11 on seeds 30100 and 30200. Its saved
commands were replayed in CPU sample-only mode to reconstruct spawn/target
inputs. Appearance seeds match exactly; manifest-rounded XY values differ from
the reconstructed values by less than 5 micrometres. Sample-only bottle yaw uses
a different random draw from the original execution, so the fixed replay bank
explicitly uses each actual manifest's recorded object yaw. Evidence is in
`case_reconstruction_audit.json`, `baseline_pilot4.json`, and
`baseline_fixed11.json`. Physical replay is not assumed bit-deterministic.

An initial interpretation of `arrival_measurements` as full stopping travel was
incorrect. `round_phases_approach.py` records its `end_depth_m` without a settle
hold; despite the log wording "stopped", supported `_brake()` still follows.
The 6–15 mm depth/13–40 mm edge changes therefore describe navigation-loop exit
only and must not be used to reduce the safety lead.

The corrected audit anchors the start to the first planner IDLE packet after
the brake event and the end to the supported controller's initial state. The
state trace and journal seq alignment was independently checked against actual
pelvis positions. In the old 11 attempts, last SLOW_WALK to first IDLE gaps were
0.225–0.297 s; after that first IDLE, maximum forward excursions were about
0.10–0.20 m. Packet context seq may precede the send slightly, so this is still a
bounded timing approximation. Details: `baseline_idle_braking_analysis.json`.
Earlier nearest-depth and loop-exit calculations are preserved as superseded
diagnostics, not calibration inputs.

The next stopping change should issue IDLE as soon as a fresh observation meets
the stop condition, before arm-planning or grasp setup work, and log actual
onset, maximum excursion and settled state. Its envelope must include control
latency and gait excursion. Existing conservative margins remain until those
measurements support a change.

## Tracking, posture and capture-settling checkpoint

Checkpoint 02 retains tracking gain 1 and the initial decoder bias. Its opt-in
critically damped correction has a 2 mm translational deadband, translation
speed/acceleration limits of 0.04 m/s and 0.2 m/s², and rotation limits of
0.15 rad/s and 0.6 rad/s². The IK adds a 0.03-weight velocity residual around
the predicted next reference, within the existing hard joint, velocity,
acceleration and support bounds. A comfortable posture prior blends into
reach (waist yaw/roll weight 0.12, right elbow/wrist 0.08) and freezes at
capture. Shared callers retain their prior behavior unless they opt in.

Central capture now requires 0.10 s of advancing physical observations with
in-window geometry, instantaneous palm-frame relative speed ≤0.03 m/s,
measured palm angular speed ≤0.20 rad/s and desired-attitude error ≤0.15 rad.
Each failing observation resets qualification. The existing reach reference
continues toward the central target; this gate does not insert a new reset or
fixed waiting pose. Post-close contact loading retains its separate 0.08 m/s
threshold. Capture-window validation precedes observer resource creation.

`checkpoint02/` records integrated source hashes. The worker ran 76 focused
tests covering bottle capture/tracking plus shared crate IK consumers, Ruff,
formatting, compilation and generated configuration checks. Physical pilot
`contact02_pilot4/` uses the same four configurations as checkpoint 01.

Motion diagnostics retain actual 50 Hz joint-velocity finite-difference
acceleration. An additional >5 Hz velocity RMS uses a third-order Butterworth
forward/backward high-pass filter on each of the seven right-arm joints,
trims 0.2 s at either endpoint and requires contiguous samples spanning over
1 s. Both metrics use first-to-last journal sample per phase; neither proves
perceptual smoothness or VLA learning benefit.

Checkpoint 02 physical results were **0/4**: all four failed the unchanged
wrist/bottle clearance guard during reach, before capture. For the same four
case configurations, median reach acceleration p95 fell from 8.964 to
7.117 rad/s² and >5 Hz velocity RMS from 0.05965 to 0.04798 rad/s, but the
failure regression prevents adopting this parameter combination. These four
trials do not isolate which new term caused it.

The replay traces show repeated central-window crossings with residual wrist
attitude error around 0.16–0.35 rad; gate qualification never completed.
Offline FK of the 50 Hz traces approximating the visual target with the
bottle-local grip point found maximum geometry-only residence intervals of
0.72, 0.72, 1.62 and 0.10 s. Adding the speed and angular-speed conditions
reduced these to 0.10, 0.02, 0.10 and 0 s, before the desired-attitude gate.
`capture_dynamics.json` records this approximation. Journal `samples.target`
is the commanded tool target, not the observed bottle point; using it to
reconstruct capture depth would be incorrect.

`gate_only_pilot4/` retains the central geometry and new settling gate while
disabling the three new tracking/continuity/posture options to separate gate
behavior from those options. Its exact profile is `task_gate_only.json`.

The gate-only ablation achieved **1/4**; the other three also failed the wrist
guard before closure. Thus the new strict gate itself is a material regression.
The next gate uses a physical-time window with bounded relative displacement
and angular excursion, while continuing to reject geometry departures and
fast approach drift. The requested depth is being moved to +1.2 mm, the
measured shared-pad midpoint (x=0.1526735 m) relative to the calibrated
x=0.1515 m tool frame. This still inserts roughly 17 mm deeper than the old
+18 mm target and adds proximal clearance versus the -2 mm candidate.

## Stopping-order checkpoint and complete braking measurements

Checkpoint 03 checks fresh observations before coordinated arm planning and
again after it; normal and protective exits send IDLE immediately. Supported
braking sends IDLE before further arm work. Four regression tests cover this
ordering; the worker ran 65 focused tests plus Ruff and diff checks.

`approach_stop_requests` records the request observation and sampled maximum
excursion through supported brake completion. In this first schema `complete`
means diagnostic completion, including failures; it must not be interpreted
as proof of quiet stance. The follow-up adds an explicit settled flag, reason,
send-time bracket and a snapshot of the active stopping thresholds.

`stop03_pilot4/` retained the strict capture gate and achieved **0/4** (three
wrist-clearance failures, one disturbed target/base). Its stopping evidence
is nevertheless usable because all four completed the supported brake before
those reach failures. Last SLOW_WALK-to-first-IDLE gaps were 21–73 ms,
versus 225–297 ms in the earlier 11 episodes. A 50 Hz trace audit found maximum
forward excursions after the request observation of 0.213, 0.289, 0.133 and
0.105 m; request-observation/trace position differences were 1.5–2.8 mm.
Runtime sampled maxima were 0.215, 0.292, 0.134 and 0.106 m. The current 0.10 m
lead cap cannot represent this braking envelope. See `braking_analysis.json`.

`lead025_pilot2/` tests a 0.25 m lead cap on the largest- and smallest-excursion
cases (original rounds 1 and 3) before changing the default. This is an
explicit stopping/reachability calibration, not evidence that all gait phases
will stop at the geometric tangent point.

Both attempts were rejected as out of reach before supported braking ran:
navigation exited at about 0.62 m depth, while the existing stand admission
checked the still-moving observation. Settlement remained incomplete. This
exposes an ordering gap; it does not establish that the eventual stationary
stance is unreachable. The follow-up moves supported braking ahead of stand
admission and retains the existing reachability, visibility and edge gates.

## Central-window and terminal-control checkpoint

Checkpoint 04 uses the +1.2 mm pad midpoint with a ±5 mm capture window.
Qualification requires at least six distinct physical observations spanning
at least 0.10 s; slower control periods retain six samples over a longer span.
Every observation must satisfy capture geometry. Pairwise position diameter
must be ≤4 mm and least-squares drift ≤0.03 m/s; rotation diameter must be
≤0.035 rad and angular drift ≤0.20 rad/s. Geometry departures clear the window,
duplicate timestamps add no qualification and backwards time is rejected.
The previous instantaneous speed and arbitrary desired-attitude-error fields
were removed. Existing jaw-tilt and full collision checks remain active.

Terminal approach limits only positive motion along the entry axis using
measured approach speed, 0.20 m/s² braking acceleration and 0.10 s lookahead.
It retains lateral, vertical and retreat corrections. At capture, inverse
compensation selects rotation targets whose corrected value is the current
supported reference. Rotation compensation then freezes through closing and
lifting, preventing further pursuit of the old arbitrary yaw. Reference
position and velocity are not reset; existing acceleration bounds and IK
continuity decelerate the remaining motion. The comfortable prior also freezes.

Supported braking now precedes stand admission and is reused by execution;
direct execution still performs braking. Diagnostics distinguish completion
from successful settling and record the command-send time bracket and active
stopping thresholds. `observer_target` and `capture_geometry` distinguish
perception from the existing commanded `samples.target` field.

`checkpoint04/` records integrated source hashes and focused regression tests
cover irregular clock cadence, bounded oscillation, fast crossings, terminal
directional braking, compensated attitude freezing and admission ordering.
`contact04_pilot4/` runs the same four cases with a 0.20 m lead-cap override
recorded in `task_contact04_lead020.json`; the default is unchanged pending
physical results.

The first checkpoint-04 physical run aborted in its first reach iteration:
the capture-recording block was incorrectly outside the ready-gate branch
and referenced `transition` before assignment. There are no completed
manifests (one rollout began and aborted), so the summary's zero completed
attempts must not be read as a successful or empty physical validation.
`contact04_pilot4/collect_00.log` and copied runtime logs preserve the failure;
checkpoint 05 corrects the branch and adds executable consumer-path coverage
before rerunning the same cases.

Checkpoint 05's six-line main hotfix first kept only capture recording inside
the ready branch; the existing posture freeze was already correctly nested.
`checkpoint05_hotfix/` preserves that exact physical source. The worker then
added `CaptureAdmissionConsumer`, placing all close-entry effects in its
callback, plus an executable false/first-true/repeated-true regression.
`checkpoint05/` is integrated; 105 focused tests and static checks passed.

`contact05_pilot4/` tested the hotfix with the 0.20 m lead override and achieved
**0/4**: two reach wrist/bottle clearance failures, one disturbed target/base,
one measured foot-drift failure. No central capture was admitted. Final
settled object depths were 0.222, 0.289, 0.430 and 0.432 m, demonstrating that
increasing a fixed lead cap did not yield consistent final stances. The last
geometry sample in the first case had 41 mm height error; the second had
about 24 mm lateral error. An axial approach limiter alone does not resolve
these three-dimensional tracking errors.

`contact05_lead010_pilot4/` repeats the same four cases on the integrated
consumer code with the default 0.10 m lead, to separate the stance override
from the current capture/tracking changes. Offline diagnosis separately checks
post-brake foot movement and tracking/IK residuals before further changes.

## Checkpoint 05 diagnosis and next correction

The default-cap run `contact05_lead010_pilot4/` completed **0/4**; every
failure was the unchanged wrist/bottle clearance guard during reach. Runtime
logs and the summary are preserved in that run. Each case entered the actual
central capture geometry (3, 1, 14 and 7 sampled observations), but none passed
the static settling gate. Together with checkpoint 01's 4/4 captures and 3/4
successes, this supersedes the earlier recommendation that a stricter quietness
gate was the appropriate remedy for the closing collision. It does not establish
that every dynamic crossing is safe.

The next minimal comparison restores direct admission on the actual central
geometry and holds the active compensated reference during closure, without
resetting joint position or velocity. Existing acceleration bounds, collision,
contact, slip and final-success checks remain in force. Static window statistics
must not block capture. Dynamic prediction is a possible later safeguard only
if closure trajectories show a specific failure it can distinguish.

## Checkpoint 06 implementation

`checkpoint06/` holds the integrated source snapshot. Capture consumes the
actual central geometry directly; the old settling window remains a diagnostic
for this comparison. Closing freezes the current compensated tool position and
rotation. Translation feedback is held and its correction rate starts from zero
when lifting resumes. The generalized reference position and velocity are never
reset. A capture on the first reach observation initializes the existing decoder
bias before constructing the held target, avoiding an uninitialized-bias jump
at lifting.

The worker passed **106 focused tests** across approach braking, supported
smoothing, whole-body grasp, walking reach, ready grasp and transport guards,
plus Ruff/format, generated configuration documentation and diff checks. The
main agent reviewed the changes and started `contact06_pilot4/` using the same
four cases, the default 0.10 m stopping cap and unchanged physical guards.
The detailed offline diagnosis is retained at
`outputs/bottle_contact_20260915/checkpoint06_offline_diagnosis.md`.

The completed checkpoint-06 pilot achieved **2/4**. Rounds 2 and 3 succeeded;
round 0 failed wrist/bottle clearance 0.385 s after entering closure, and round 1
failed because the bottle toppled during braking. All three reach attempts
admitted the central geometry; their closure target freeze diagnostics report
zero compensated position discontinuity. Actual motion and success still need
improvement; target continuity alone is not sufficient.

For round 1, offline MuJoCo forward evaluation of the recorded 50 Hz states found
right finger 2 contacting the bottle at sequence 3614 (0.659 mm penetration),
0.44 s after the stop request at 3592. Bottle tilt exceeded 4 degrees at 3615,
before the quiet completion at 3658. Maximum measured base forward excursion
was 0.248 m despite a 0.10 m capped lead. Tool world displacement from the request
to first finger contact was 0.331 m, including base motion and continued arm
preparation. These records identify an actual braking collision, not an
acceptance-only failure. Raw contact and selected tool-state records are
`contact06_pilot4/brake_bottle_contacts_case1.json` and
`contact06_pilot4/brake_tool_excursion_case1.json`.

## Closing residual and the larger stopping-cap diagnostic

Checkpoint-06 round 0 retained about 33.9 mm of uncompensated target-to-measured
position residual at capture. Although both frozen Cartesian targets were
constant, the actual tool advanced 14.8 mm along its capture axis over 0.28 s,
while the bottle followed by only 6.0 mm. The measured axial depth moved from
+3.80 mm to -6.87 mm, outside the original central window. The guard reported a
predicted wrist collision while actual contact was still finger-only; subsequent
recorded states show wrist contact. This supports holding the measured tool
position through the existing bounded joint controller. It does not support
removing the wrist collision guard or resetting joint reference state.

`contact06_lead025_pilot4/` tested a 0.25 m stopping-lead cap with the corrected
brake-before-admission order. It achieved **1/4**: the prior braking-collision
case succeeded, two cases were rejected by the standing-arm/squat reach map
at settled relative XY of [0.48, -0.32] and [0.55, -0.28] m, and one slipped in
recovery. The 0.25 m cap is **not adopted**. The reach-map implementation extends
only vertical squat reach; whether its rejections agree with actual bent-waist
whole-body feasibility remains unverified. No reachability or physical gate was
weakened for this diagnostic.

## Checkpoint 07: brake toward the measured hand position

`checkpoint07/` is the integrated main snapshot; the worker's separate four-file
hash list is preserved under `checkpoint07_worker_report/`. At capture the raw
translation target now equals the measured tool position, with compensation held.
This deliberately changes the Cartesian goal to remove tracking residual; it
preserves the sent joint reference and its velocity/acceleration constraints.
Rotation handling is unchanged. Diagnostics distinguish Cartesian goal retreat
from sent-reference continuity. The focused six-file suite passed **106 tests**,
with Ruff/format, configuration reference and diff checks passing.

`contact07_pilot4/` tests this single change on the same four cases with the
unchanged default stopping cap. The checkpoint-06 closing residual calculation
is saved in `contact06_pilot4/r0_capture_freeze_diagnosis.json`.

The complete checkpoint-07 pilot also achieved **2/4**, with a different split:
rounds 0 and 1 succeeded, round 2 disturbed the target during reach, and round 3
failed wrist/bottle clearance during closure. Round 3 had entered the axial
central window while remaining 22.4 mm off laterally and 16.8 mm off in world
height. The two successful captures had lateral errors 12.3/2.5 mm and height
errors 6.9/1.9 mm. These observations motivate checking three-dimensional
alignment before entry; they do not yet prove that tighter scalar thresholds
alone would fix the failures. The current entry law adds axial separation when
the hand is high but does not add separation for lateral misalignment.

## Checkpoint 08 design: align continuously before deep entry

Recorded-state MuJoCo contact geometry in checkpoint 07 shows round 2 first
contacting a single finger with bottle-centre palm-frame lateral offset +27.35 mm
and visual grip depth about 42.24 mm; round 3 first contacts the opposite finger
at lateral offset -26.08 mm near the central depth. Successful rounds first
achieve bilateral geometric contact at centre offsets +1.36/+1.20 mm. Static
forward evaluation establishes contact geometry, not measured normal force.
Runtime contact/load and lift/success records remain the acceptance evidence.

The next change uses a continuous lateral/height alignment-dependent entry gap,
with whole-body-only capture tolerances of 12 mm lateral and 15 mm world height.
Large errors retain a non-contacting axial gap derived from the hard-pad forward
edge, calibrated tool offset, actual bottle radius and existing guard margin:
184.759 - 151.5 + 22.639 + 4 = 59.898 mm for the sampled radius. The gap converges
smoothly to the 1.2 mm central depth as alignment improves. Terminal braking
uses this current guarded goal instead of always using the final central goal.
There is no static dwell condition or reference-state reset. Legacy grasp
strategies keep their 26/20 mm geometry defaults; the stopping cap stays 0.10 m.

Checkpoint 08 was integrated after 110 focused tests, 21 legacy capture tests,
Ruff/format/config-doc checks and the worker-tree full `make check` passed. The
integrated main final `make check` is still pending. `contact08_pilot4/` achieved
**2/4**: all four reached the tighter central capture geometry, with no reach
failure, but rounds 0 and 2 failed wrist/bottle clearance during closing. Round
0 entered with only 2.9 mm lateral and -2.25 mm height error, then moved from
+5.67 mm depth to -6.85 mm within about 0.16 s without bilateral contact.

`contact08_no_velocity_prior_pilot4/` tests only
`ik_velocity_continuity_weight=0`, retaining all hard velocity/acceleration
bounds, tracking correction, posture, alignment, contact checks and the 1.2 s
closing time. The hypothesis is that the existing continuity cost around
`x + v*dt` can encourage coasting even after the position goal is held. This is a
runtime ablation, not an adopted configuration or an established cause.

## Checkpoint 08 velocity-prior and closure ablations

The fixed four-case run `contact08_no_velocity_prior_pilot4/` completed **3/4**
with only the velocity-continuity cost disabled. The hard speed/acceleration
limits and bounded tracking correction remained active. Across the four cases,
median actual right-arm reach acceleration p95 was 7.140 rad/s² and above-5-Hz
velocity RMS was 0.04918 rad/s, versus 8.964 and 0.05965 in checkpoint 01.
These are descriptive small-sample comparisons with asynchronous physics, not
isolated causal estimates or VLA-learning evidence.

Restricting the comparison to checkpoint 08 with versus without its velocity
cost gives acceleration p95 medians 7.578→7.140 rad/s² (−5.8%) and above-5-Hz
velocity RMS 0.05385→0.04918 rad/s (−8.7%). The roughly 18% reduction against
checkpoint 01 includes the other tracking/posture/alignment changes as well;
it must not be attributed solely to disabling the velocity cost.

The additional `close_seconds=0.5` trial `contact08_fastclose_pilot4/` completed
**2/4**: one wrist/bottle failure during closing and one finger/bottle failure
before capture. Its two successes retained visual grip-point depths near the
pad centre: at first bilateral geometric contact −2.0/−2.4 mm, and at loaded
upright +0.0/−3.0 mm. The 1.2 s variant's three successes ended at +5.4/+8.0/+10.0 mm.
The lower aggregate success count does not isolate closure duration because one
failure preceded closure. Faster closure remains an experiment pending diagnosis.
Runtime logs, exact overrides, summary and FK audit are preserved in each run.

The corrected audit helper isolates manifests, traces and sidecars by batch and
case using sequence/time anchors. Its complete prior-baseline audit includes
**11 attempts, 10 accepted**, across two batches; the original partial baseline
report remains preserved. Baseline first-bilateral geometric contact positions
had median palm x=0.170575 m (successful cases: 0.170390 m), toward the distal
part of the shared pad interval [0.120588, 0.184759] m. These reconstructed
contacts identify geometry, not measured force or load.

## Frozen checkpoint 09 and fresh collection

Checkpoint 09 adopts only `ik_velocity_continuity_weight=0` and updates the
profile expectation test; closing remains 1.2 s. The source/configuration hashes
are preserved in `checkpoint09/` and `final_collect/expert_checkpoint.json`.
The worker's focused profile suite passed 33 tests. The integrated main final
check remains pending until collection, artifact review and documentation finish.

The closure diagnosis is recorded in `checkpoint09_offline_diagnosis.json`.
Examined failed closures inherit actual forward speed of roughly 49–80 mm/s;
the Cartesian target and frozen bias are stationary, but the actual hand keeps
advancing and trips the existing wrist collision guard. This does not justify
relaxing the guard. An actual-velocity-aware capture/braking controller remains
future work; a strict static dwell gate previously failed physical validation.

Fresh collection uses the frozen production profile, no task overrides, and
previously unused seed 30300, requesting ten accepted episodes with a maximum
of three ten-attempt batches. It runs only through collection on GPU 0 / instance 0.
All attempts, raw recordings and traces remain in `final_collect/`; earlier pilot
or prior-task videos are not counted as the new delivery.

## Checkpoint 09 held-out regression: 6/10, delivery deferred

Seed 30300 completed ten attempts with six accepted. Rounds 2/3/4 entered
capture and failed wrist/bottle clearance during closure; round 5 failed finger/bottle
clearance during reach. The next batch was interrupted during setup, before its
first round, and owned launcher cleanup completed successfully. GPU 0 / instance 0
were released. `final_collect/interruption.json` records why this run was stopped;
its directory name does not designate a delivered final version.

This 6/10 held-out result is insufficient to claim high success. All completed
attempts and source videos are retained. The new ten-video delivery remains
pending a correction for inherited forward hand motion. The proposed correction
must let the axial goal brake an already-ahead reference using measured stopping
travel; merely limiting the next positive command increment has not done so.

The schema-3 audit adds the last task-outcome state as the final holding endpoint.
It validates that endpoint against the unique case trace and sidecar; literal trace
ends can extend after outcome. For the six successes, final bilateral geometric
contact medians in palm x are 165.12, 151.19, 147.61, 152.81, 161.23 and 141.77 mm,
compared with the pad midpoint 152.67 mm. Capture centrality and final holding
centrality are reported separately; reconstructed geometry is not contact-force evidence.

## Checkpoint 10: active measured-momentum braking

The reach now low-passes positive measured hand speed with attack time equal
to half the existing lookahead and release time equal to the lookahead (default
0.05/0.10 s). The common timed/per-tick target adds stopping retreat
`v * lookahead + v² / (2 * acceleration)` to the alignment-dependent axial gap.
At 80 mm/s and current settings this is 24 mm. Unlike the earlier forward-step
cap, this can retract an already-ahead command continuously. Its 0.10 m target
depth cap bounds retreat; it does not certify the actual stopping distance.

Capture requires the current geometry and predicted stopped depth to be in the
existing central window. Prediction uses the larger of raw positive speed and
filtered speed and the full uncapped stopping distance, so the goal filter/cap
cannot hide a fresh speed pulse. It adds no dwell or reference-state reset. Hard
joint/reference bounds, collision/support checks and contact/success gates remain.

The worker's 112 focused tests, 21 legacy capture tests, Ruff/format checks and
isolated-tree full `make check` passed. Main's final integrated check remains
pending. `checkpoint10/` freezes code hashes. `contact10_momentum_pilot4/` tests
recorded seed30300 cases 0/2/3/4 (one success, three closure failures). Conditions
were reconstructed using sample-only sampling, verified against appearance and
manifest-rounded XY, then given exact recorded object yaw; replay is not assumed
physically bit-deterministic.

The checkpoint-10 targeted replay completed **3/4**, versus 1/4 previously for
these selected case configurations: all three original closure failures passed,
while the former successful case failed wrist/bottle clearance during closing.
Capture prediction speeds were 8.97–24.54 mm/s. The three successful reaches took
8.50, 15.52 and 8.02 s; their loaded bilateral contact medians were palm x=157.5,
143.2 and 144.6 mm. This is a targeted regression result, not a random success-rate
estimate. The remaining failure passed predicted stopping admission, confirming
that the constant-acceleration approximation is not a stopping guarantee.

`final_collect_v2/` now runs frozen checkpoint10 on fresh seed30500 with no task
overrides, seeking ten accepted videos in at most three ten-attempt batches.
Only this new version's completed collection can become the delivery; the aborted
6/10 checkpoint09 run remains diagnostic evidence.

## Post-handback endpoint audit correction

`whole_body.outcome.state` is recorded before the `execute()` cleanup calls
`arm.stop()`. It is retained as an outcome anchor, not described as the final
post-handback holding state. The schema-4 audit adds a separate
`post_handback_video_endpoint` from the uniquely matched case trace tail.
For these zero-retry runs, `round_loop.py` holds IDLE for four seconds, writes the
manifest, then calls `reset.end_video()` before the next reset. Lifecycle, tail
continuity screens and terminal image review are all checked. Tail physical time
is explicitly cadence-extrapolated from exact sidecar anchors; exact image/state
synchronization is not established by this audit.

The completed checkpoint09 batch has ten validated tails and six successful
bilateral holds. Successful tail contact medians are palm x=166.12, 148.46,
147.06, 150.27, 169.94 and 141.40 mm. One remains more distal than the other five,
so admission at the midpoint alone does not establish equally central final
holding in every case. Source videos and prior audit versions are preserved.

## Checkpoint 10 fresh result and identified closing-control defect

`final_collect_v2/` completed **9/13** over seeds30500/30600: three closing
wrist/bottle failures and one measured-foot-drift failure after capture. A further
case30600:3 was interrupted during spawn/settle, separately from the thirteen
completed attempts. Owned launcher cleanup passed; all originals and partial
recordings remain. Post-run hashes still match all eleven frozen source paths.
This run is diagnostic, not the requested ten-video delivery.

The targeted case30300:0 exposes a defect in freezing moving-reach compensation
through closing. Its frozen axial translation bias was +16.432 mm. Once the raw
world hold was fixed, live reference-minus-measured axial error fell to roughly
9.3–10.5 mm at21.64s, 6.3–7.1mm at21.76s, 4.6–5.0mm at21.88s and2.4–3.0mm
at22.00s. Thus the frozen moving bias overcompensated the stationary hold and
continued drawing the hand forward. This explains why predicting only incoming
hand momentum could still admit a failing closure. The next narrow correction
keeps the raw close target fixed but lets the existing bounded translation
feedback unwind the changing lag, without resetting joint references or weakening
collision/contact/support checks. Physical validation and new videos remain required.

## Checkpoint 11: fixed world hold with live bounded correction

The narrow fix replaces the capture-time call to the shared translation-freeze
primitive with `begin_live_translation_hold()`. It snapshots measured world tools
and the active reference without changing joints, joint velocity, translation
correction value or correction rate. Raw close targets stay fixed; existing live
bounded correction keeps unwinding motion lag. Shared freeze behavior and the
rotation/posture freezes remain unchanged. Lift resumption is already a no-op on
active correction state.

The diagnosed case entered capture with +16.432mm axial correction already
unwinding at −26.744mm/s (27.079mm/s vector norm). The previous freeze zeroed
that rate. The actual hand then advanced13.718mm past the fixed raw hold, and
FK grip depth drifted +0.637→−12.026mm before wrist collision. The regression
checks fixed raw hold versus frozen-bias retention, q/v/value/rate continuity,
and the existing40mm/s correction-speed and200mm/s² acceleration bounds.
The worker's113 focused tests,21 legacy tests, Ruff/diff checks and isolated
full `make check` passed. Main's final integrated check remains pending.

`checkpoint11/` contains source hashes; `contact11_live_hold_pilot4/` replays the
same targeted four cases. No closure time, geometry window, friction, scene,
low-level control limits or collision/contact/support/success guard changed.

The checkpoint11 targeted replay completed **3/4**. Cases30300:0/2/3 captured
and succeeded; case30300:4 failed with disturbed target/base before capture, so
its closing-feedback change never activated. No closing wrist failure occurred
in the three captures. Successful reach durations were5.58/9.62/9.32s; post-handback
bilateral contact medians were palm x=157.8/153.8/161.8mm. This supports keeping
the hold correction, without establishing a general success-rate guarantee.

`final_collect_v3/` now collects the frozen checkpoint11 on fresh seed30700,
requesting ten accepted episodes with no task overrides. The preceding diagnostic
runs and prior task videos are excluded from this final collection.

## Pending work

- Validate direct central-geometry admission and the held closure reference,
  with bounded tracking correction and comfortable posture active.
- Verify immediate stopping and report the remaining braking-envelope spread;
  navigation exit must not be equated with completed braking.
- Compare actual acceleration/high-frequency motion and central capture, collect
  ten new accepted videos, complete final repository checks and verify previews.

## Checkpoint11 complete fresh collection and visual review

`final_collect_v3/` completed with seeds30700/30800: **10/15** (7/10 +3/5).
Three wrist/bottle clearance failures, one finger/table clearance failure before
capture, and one recovery slip remain. Inspected wrist failures30700:1/3 are
predicted-step guard failures, not observed wrist contacts in reconstructed state.
Source/config hashes were unchanged after collection; owned launcher cleanup
returned0 and instance0 lock was free. GPU1 was not touched.

Schema4 audit maps all15 manifest cases to their unique traces and endpoints. All
10 successful post-handback tails retain bilateral finger contact. Their35 contact
points have palm x142.448–161.156mm, median152.682mm, against pad midpoint152.674mm;
all lie within installed pad axial bounds. This is geometry evidence, not force
or stationary-hold evidence. Report: `final_collect_v3/central_grasp_audit_schema4_full_postrun.json`.

Main inspected all10 twelve-frame timeline sheets and original/preview endpoint
pairs; bottles remain in the sampled final frames. This is keyframe review, not
every-frame playback. Case30700:4 has12.64s reaching and marked outward/upward
arm extension through recovery/handback;30800:4 has11.16s reaching. Natural carry
and hesitation remain unresolved. Successful reach median observed span7.28s,
arm acceleration p95 median7.577rad/s² and >5Hz arm-velocity RMS median0.05183rad/s;
these descriptive fresh-scene metrics are not a controlled A/B or VLA result.
Across successful outcome-to-tail intervals6.66–8.78s, root net displacement is
0.286–0.559m (median0.357m), palm0.398–0.661m (median0.523m). The terminal
controller handback therefore still needs measured stance/pose stabilization;
contact retention must not be described as a stationary final hold.

Ten full-duration25.04–31.52s previews are425454–490786bytes, at original FPS/frame
counts, fully decoded, with matching endpoints. ZIP4552364bytes contains precisely
these ten MP4s and byte-for-byte archive verification passed. Originals and failed
attempts remain. `previews/verification.json`, `visual_review/review.json` and
`postrun_source_integrity.json` retain checks. This is checkpoint11 evidence while
the narrow full6D close-hold proposal is reviewed; final main checks remain pending.

Checkpoint11 recovery-slip case30800:3 still reported bilateral contact in its
last logged lift samples, while first-bilateral-to-outcome bottle displacement
in the palm frame was(-15.00,-3.85,+38.25)mm. Progress paused and resumed but the
4cm relative-slip guard eventually failed. Presence of bilateral geometry is
therefore insufficient evidence of loaded stability; force-informed regulation
and relative-slip sensing remain follow-ups, with no guard relaxed here.

Terminal-hold review: `BottleWholeBody._brake_handoff` checks the predicted swept
braking reference and calls `StreamedArm.brake_to_stop`; the latter records
observed joint-rate diagnostics but does not enforce measured stoppedness.
`execute` stores outcome before `arm.stop(preserve_reference=True)`, which primes
IDLE/frozen-upper-body planner references and exits the streamed controller.
The observed tail movement therefore warrants measured root/hand stabilization
and a compatible standing/carry reference at this interface. This identifies a
missing acceptance condition, not a proven sole cause of the physical drift.

## Candidate12: measured6D close hold and reviewed transition boundaries

Main reviewed and integrated three source/test updates in `checkpoint12/`.
`SupportedMotion.begin_measured_pose_hold` stores actual world position and
rotation while preserving initialized calibration, correction values/rates and
command state. Close uses takeover bias plus deadbanded raw hold error as the
bounded correction target; IK/reference lag is excluded from this close-only
estimator. Corrections remain acceleration/speed bounded. This proportional
compensation can leave error when decoder bias changes with posture.

`end_measured_pose_hold` returns `reference - gain*feedback`; the policy uses
those targets on the **same transition tick** before computing the torso-relative
lift offset. This fixes a caller mismatch found during main review, where a
correct lift start had been computed but stale close targets were still sent.
Both hands are continuous; rotation state remains unchanged on lift entry.
First-tick capture is possible because the settle gate is diagnostic-only.
Missing origins therefore initialize with normal first-sample calibration while
inherited correction rates and command state are preserved.

Worker checks:78 smoothing/whole-body tests,39 approach/walking/ready-grasp/crate
consumer tests, compile, Ruff and diff check passed. Main physical regression uses
`pose_hold_regression4.json` (two prior successes and two prior close failures)
in `contact12_pose_hold_pilot4/`, with unchanged official config and scene physics.
No new configuration fields, guard changes or faster closure were introduced.

## Candidate12 physical result and final version selection

`contact12_pose_hold_pilot4/` completed **2/4**:30700:0 and30700:1 succeeded;
30700:2 (originally successful) hit the wrist/bottle clearance guard;30700:3
(original close failure) slipped during recovery. Both former close failures
passed closure, but one subsequently slipped and one former success failed.
This selected asynchronous replay provides no support for an overall reliability
improvement, so no further candidate tuning or new collection was launched.

Main verified all checkpoint12 source hashes unchanged after the run, preserved
runtime logs and original videos/traces, and restored exactly the three changed
paths from checkpoint11. `checkpoint12_rejection_and_restore.json` records the
restoration; all cumulative production hashes again equal checkpoint11. The
worker tree and checkpoint12 source snapshot retain the unadopted candidate.
The owned launcher stopped cleanly and instance0 lock was verified free.

Final artifacts are the ten chronological successful checkpoint11 videos from
`final_collect_v3/previews/`, full-duration and normal speed, with original
recordings and all15 attempted episodes preserved. Do not combine successes
from checkpoint12 with this delivery or report these ten as10/10 attempts.

Worker read-only candidate12 audit confirms measured6D mode active on every
logged close sample for30700:1/3, zero spread of raw right-hand position targets,
and correction speed/acceleration within configured bounds. Both previously
failing close motions entered lift after1.334/1.304s; case3 slipped1.943s into
recovery. Its corrected-target-minus-actual residual grew40.470→66.198mm despite
bounded hold feedback, reinforcing the need to model posture-dependent decoder
bias/dynamics. Coarse0.12s boundary samples were consistent with continuity but
cannot verify every20ms transition tick. Candidate code remains unadopted.

## Final validation and artifact delivery

Main ran `make check` on restored checkpoint11 plus the existing shared dirty
worktree. Full scope selected115 changed paths; Ruff formatting/checks, repository/
configuration/environment checks and complete default CPU pytest suite passed.
Size advisories and existing retired-config-key warnings were reported, without
check failures. Log: `outputs/bottle_contact_20260915/main_final_make_check.log`.
Later documentation-only status changes use `make check-docs`.

Final source/config hashes match the10/15 recording run exactly. Main verified
that previews select the ten chronological accepted episodes, every actual MP4
size is below500000bytes, decoded frame counts/FPS/durations match originals,
and ZIP content equals the ten delivered files byte-for-byte. Timeline review
covered12 sampled frames per episode, plus every original/preview endpoint pair;
this is not a claim of watching every frame. The post-handback FK/contact audit
covers all15 attempted episodes, and all10 successful tails retain bilateral
central-pad contact. Original recordings and all failed attempts are preserved.

GPU0/instance0 is released; final live GPU query showed no compute applications.
GPU1 was never stopped or used. No commit, VLA training or physical-parameter
change was made for this follow-up.
