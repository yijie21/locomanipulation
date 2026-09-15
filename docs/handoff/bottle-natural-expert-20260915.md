# Natural bottle expert: preparation, wrist framing and continuous references

Date: 2026-09-15 UTC. Implementation branch: `fix/bottle-natural-20260915`,
isolated from `main` at `fcf23d9` with the existing uncommitted working tree copied
before editing. Integration must preserve those earlier edits.

## Requested behavior and scope

The user requested implementation of three reviewed changes, followed by ten
recorded bottle episodes and compressed full-duration videos:

1. Choose a natural preparation/pregrasp posture together with a feasible table
   stopping position, replacing the fixed tucked walking hand target.
2. Use camera-admitted wrist projection to maintain a useful image region with
   hysteresis and bounded slow corrections, after ego size triggers preparation.
3. Preserve reference motion through planner/POSE takeover and make the complete
   future reference window bounded and consistent with its reported velocities.

The existing physical success criteria, recorded-action contract and hardened-pad
scene remain the validation conditions. Failed attempts must remain inspectable;
ten accepted recordings do not imply ten attempts or a measured population success
rate. No new VLA training or closed-loop evaluation is part of this request.

## Review findings motivating the change

- The previous walking goal fixed tool x at 0.08 m, clipped lateral position, and
  strongly constrained wrist orientation. A weak prior around the previous IK
  solution could preserve a twisted arm configuration.
- The previous table-edge rule was a constant threshold combined with the bottle
  depth threshold. Historical 36 px preparation triggered at spawn, and wrist
  visibility did not drive image-region control.
- Planner-to-POSE takeover retained joint position but issued constant holds and
  cleared velocity; the supported solver initialized its velocity to zero.
- The entire history/future POSE window is transmitted to deploy. The historical
  comparison of smooth first-row references with rough realized joints does not
  isolate the cause to deploy: future samples also depend on commanded velocity.
  Subsequent inspection found the preserved deployed source and headers under
  `outputs/bottle_continuous_20260914/deploy/`. Its build command names that source;
  the launcher records the actual binary identity in each run's receipt. The
  selected encoder configuration uses ten joint-position, joint-velocity and
  orientation frames with five-tick spacing (offsets 0 through 45, about 0.9 s).
  `deploy.cpp` gathers those future frames and clamps to the buffer end. Baseline
  deploy logs also record incoming windows and merge/playback offsets. The whole
  future window therefore needs validation, not only the earliest logged row.

Historical evidence and earlier interpretations remain in
[the prior experiment record](bottle-hanging-start-20260914.md).

## Validation and evidence

Implementation, repository checks and live collection are complete. Evidence root:
`outputs/bottle_natural_20260915`. The pre-change snapshot is retained as
`baseline.patch` and `baseline_hashes.json`; `baseline3/` is an independently
launched check of the original main-tree configuration and recording pipeline.

Final results, commands, checks, artifacts and remaining optimization work will
be recorded here after implementation and collection.

### Original-main pipeline check

`baseline3/`: three sampled attempts, seed 30000, GPU 0 / instance 0, original
task profile and `outputs/scene/sim_hardpads`. Results: 2/3 accepted (35.0 and
34.8 s); one failure, no sustained bilateral contact (23.0 s). All three videos,
state traces and manifests are retained. The pipeline launched, recorded and
stopped its owned stack successfully; its final nonzero exit reports that the
three-success target was not met within the intentionally single three-round
batch. This small run is a pipeline/diagnostic baseline, not a success-rate
estimate. Runtime logs were copied to `baseline3/runtime/` before another launch.

### Checkpoint 01 diagnostic pilot

`checkpoint01/` preserves the exact integrated source files and their SHA-256
identities. The worker reported 48 passing focused tests before this pilot.
`candidate01_pilot4/`: four attempts, seed 30000, unchanged scene and success
criteria, 0/4 accepted. Three failed on right wrist/bottle clearance during
walking/braking; one failed on left hip/table clearance. None reached supported
whole-body initialization. All four acquired the wrist target. The inspected
first video shows the bottle knocked over, so the wrist rejection reflects real
interference and is not a reason to weaken the guard.

The first, third and fourth attempts selected approximately 0.37 m bottle depth
and a walking tool x near 0.25 m including ROI correction. Failure-state forward
speed was approximately 0.35–0.43 m/s: nominal candidate-path clearance did not
cover actual gait/tracking and stopping behavior. `pilot_summary.json` retains
the compact findings; complete states and candidate checks remain in manifests.
This checkpoint is a failed diagnostic candidate, not the collection delivery.

### Checkpoint 02 diagnostic pilot

`checkpoint02/` preserves the integrated source identity. Natural preparation
keeps a 0.18 m commanded pregrasp gap, including a 0.02 m tracking allowance.
A speed-dependent braking lead, capped and latched at 0.10 m, advances both
stopping thresholds; raw target depth also caps ROI-corrected hand extension.

`candidate02_pilot4/`: the same four seeds produced 2/4 accepted episodes
(34.6 and 33.7 s). All four acquired the wrist target, passed walking/braking
clearance and entered supported whole-body control. One failed during closure
with target/base disturbance (23.0 s); another completed lifting and upright
recovery but lost the bottle during subsequent holding (39.1 s). The latter
remains a failure despite the earlier whole-body success diagnostic.

The first attempt's takeover reported zero commanded position and velocity
start errors, with incoming joint speed about 0.70 rad/s. This proves continuity
of those boundary references, not smoothness of actual robot motion. Actual
right-arm acceleration remains substantial: finite-difference qvel at 50 Hz,
pooled over seven joints within logged reach spans, has p95 absolute values
7.23–8.74 rad/s² across these four episodes. The original three-run diagnostic
was 6.25–6.89 rad/s² by the same calculation. These small, non-deterministic
samples do not establish a causal change or a population success rate.

Remaining observed discontinuities motivated the next checkpoint: capture
reset the commanded hand targets to measured palms, and successful recovery
could return control while the supported reference was still moving. Guards,
success criteria and hard-pad scene physics were retained.

### Checkpoint 03 continuity correction

Capture now retains the active commanded hand targets and rotations, recording
their discrepancy from the measured palms. Successful recovery predicts a
bounded stopping reference, admits its sweep with the bottle attached, and
emits that braking path with the right hand closed before returning to the
planner. Joint and root angular velocities reach zero at the command endpoint.
Packet acceleration and separately observed motion are recorded in `handback`.

Review caught an additional lifecycle defect before GPU validation: explicit
`_push` calls sent new velocity/time values without persisting them for the next
handoff. The sender now retains the latest sent state, including preview-derived
root angular velocity when none is supplied. A regression covers moving packets
followed by stopping prediction and braking, not just an isolated synthetic
braking call. The worker passed 54 focused tests, including crate compatibility,
plus Ruff and diff checks. Main integrated the frozen source into `checkpoint03/`
and launched the repository check and same-seed diagnostic pilot.

`candidate03_pilot4/` completed 3/4 accepted episodes (35.2, 35.7 and 35.5 s).
The previous closure-disturbance and post-recovery-loss cases passed this run;
the fourth case failed to sustain bilateral contact (22.9 s). This is a small
same-seed diagnostic, not evidence that those failure types are eliminated.
The bounded pilot runner exited nonzero because its four-success target was
not met in four attempts; collection and owned-stack cleanup both exited zero.

`make check` completed on integrated main. A fresh collection uses seed 30100,
ten-success target, ten attempts per batch and at most three batches. All
attempts, task/config snapshots, deploy receipts, videos and state traces remain
under `final_collect/`; accepted-video delivery is separate from attempt counts.

The outgoing-brake audit separates packet history from future commands. Across
the three accepted diagnostic episodes, retained pre-current lift frames reach
2.96–3.55 rad/s² in the full-packet acceleration calculation. Reconstructing
overlapping frame indices places those extrema in the five history frames;
current-and-future braking frames remain at or below 2.000001 rad/s². Each
56-frame packet advances its epoch by one and has current at first + 5. The
encoder gathers current through current + 45. The configured 2.0 rad/s² bound
must therefore not be described as a bound on every retained history frame.
Braking reference-log context time remains stale, although payload frame epochs
and separately sampled actual-state times are continuous; splitting future and
whole-packet acceleration diagnostics is a remaining logging improvement.

## Final collection result

`final_collect/` finished successfully: **10 accepted episodes from 11 attempts**
(90.9% in this small sample). Batch 00, seed 30100, was 9/10; batch 01, seed
30200, supplied the final 1/1. The one failure was bottle slip during upright
recovery. Success episodes took 35.3–42.1 s by the episode timer; videos cover the
complete recorded task span, whose duration excludes the unrecorded prelude.
The runner and owned-stack cleanup exited zero; GPU 0 / instance 0 were released.
GPU 1 was not used by this task. Scene physics and physical acceptance gates
were unchanged.

All 11 attempts acquired the wrist target. The ten accepted episodes recorded
zero joint-reference discrepancy at close entry and zero commanded joint/root
angular velocity at handback completion. Realized movement remains less smooth
than the reference: accepted-episode right-arm p95 absolute acceleration was
7.63–9.17 rad/s² during logged reach spans (median 8.42), 4.85–8.88 during close,
and 6.94–12.22 during lift. These are pooled seven-joint finite differences of
actual qvel at 50 Hz, not perceptual scores or proof of improvement over the
small baseline sample. The new sample is encouraging for task success but does
not establish a higher population success rate or improved VLA learning.

`previews/index.html` is the ten-video gallery; `previews/verification.json`
records source/preview frame counts, durations, encoding and decode checks.
The ZIP contains only the ten size-checked previews. Original recordings and
all failure manifests/state traces remain in the collection directory.
All ten previews passed full decode, equal source/output frame counts, duration
and frame-rate checks, plus visual first/last-frame comparison. They retain
12.5 fps and full source durations of 22.24–28.00 s at 640×480. Individual files
are 406,303–484,696 bytes; the ZIP is 4,452,304 bytes and passed its CRC check.
SHA-256 values are retained in the verification record. Final documentation and
diff checks passed, and all 15 owned source paths still match `checkpoint03/`.

## Remaining optimization priorities

1. Coordinate closure, lifting and upright recovery with measured finger contact
   and bottle motion. Stable bilateral contact and a small relative palm/bottle
   speed should govern loading and recovery progress. Higher friction cannot
   compensate for a finger losing contact or the wrist disturbing the bottle.
2. Reduce actual tracking oscillation. Preserve synchronized actual q/qvel and
   the consumed future references, then isolate whole-body feedback, SONIC
   tracking and contact contributions. Zero reference handoff errors and bounded
   future acceleration are not evidence of smooth realized joint motion.
3. Refine the stopping envelope from actual braking and tracking error, and
   jointly optimize the remaining reach with waist/arm posture costs. The present
   conservative lead parks farther away and can increase subsequent body motion;
   a comfortable walking prior does not guarantee natural full-reach or carry
   posture. Recheck the previously observed VLA parked-state failure mode.
4. Validate on a larger held-out spawn/target set, reporting physical success,
   actual motion and image/action timing together. Only subsequent VLA training
   and closed-loop evaluation can establish that these demonstrations improve
   learning; this request does not include that training.
