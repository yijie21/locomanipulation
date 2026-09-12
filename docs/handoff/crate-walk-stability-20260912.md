# Crate: walking approach and smoother grasping

User request, 2026-09-12: start 1.2–1.5 m from the table edge, record approaching
and grasping in one episode, and reduce the abrupt upper-body motion before contact.
Baseline: `6aa0b2f`. Branch: `fix/crate-walk-stability-20260912`.
Current usage: [crate RGB guide](../crate_visual_collection.md).

## Required behavior and implementation

- Distance means the initial pelvis horizontal coordinate to the near table edge,
  not the table centre. Reset samples the distance; the result records its measured
  value. All placement ends before recording; no approach-end teleport occurs.
- Approach decisions use head RGB marker geometry and robot observations. The
  visible top pair supports coarse range/bearing under a fixed upright/frontal
  basket prior. Full RGB pose estimation must be available before reaching.
- Walking and braking use the same `PlannerArm` owner as grasping. Duplicate
  camera reads hold the last locomotion command; missing/stale vision stops motion.
- A continuous quiet interval of observed target motion and simulated IMU angular
  motion gates reaching. Base quaternion, movement, speed and approach source are logged.
- Joint velocity reversals are acceleration-limited using measured control intervals.
  Palm orientation references blend from the measured starting orientation and
  finish alignment before closure. The existing coupled waist/two-arm IK remains.
- Object/world truth is confined to reset and after-episode measurement, with
  explicitly recorded external fall aborts. Base orientation is the recorded
  simulated proprioceptive/IMU signal; it does not supply hidden object information.

## Diagnosis and preliminary evidence

Artifacts: `outputs/crate_walk_stability_20260912/` (all preserved).
The deterministic recorded-symptom check was run before modifying control:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  outputs/crate_walk_stability_20260912/measure_shake.py \
  outputs/crate_visual_expert_20260912/collection_10_v2 --assert-stable
```

It fails on abrupt reference reversal/body motion. The diagnostic checks maximum
pre-contact torso angular speed ≤45°/s and reference acceleration ≤4 rad/s²;
these are debugging thresholds, not a general balance certificate. `baseline_shake.log`
retains per-episode values. Torso velocity comes from the recorded simulation
qpos/qvel and the model; reference acceleration uses successive command intervals.

Ranked hypotheses were command reversals without acceleration limiting, abrupt palm
orientation targets, and residual body motion when reaching starts. In the old
code, the reference velocity could reverse from +0.7 to −0.7 rad/s in a camera tick.
The first three old episodes had acceleration peaks of 57.18/56.93/54.09 rad/s².

`acceleration_only_01` repeats those three seeds at the original near stand with
only the new joint-reference filter. Peaks became 2.55/2.62/2.58 rad/s², and torso
angular-speed peaks changed from 41.11/29.83/19.09 to 18.48/28.90/17.36°/s.
Task outcomes were 1/3 success; two basket-tilt exits remain. This supports smoothing
the command reversals, not a claim that loaded grasp stability was fully solved.
After this trial the limiter's interval calculation was corrected for irregular
camera intervals, with a dedicated regression test.

`walk_01` is the first complete walking pilot: measured initial edge distance
1.480 m; walk at 0 s, brake/settle at 3.31 s, pregrasp at 5.90 s, close at 15.32 s,
and successful completion at 22.76 s. Object rise was 9.29 cm, final truth tilt
5.70°, with bilateral contact. Pre-contact torso angular-speed peak was 27.33°/s.
`manifest_00.jsonl`, `observations/`, `trace/`, `config_snapshot.json` and
`deploy_receipts.jsonl` retain actual measurements, source hashes, scene and model
identity for each pilot. No original video is replaced by a compressed preview.

## Final batch and checks

A fixed-version ten-episode batch is pending. Full-batch outcomes, compressed
video links and final review results will be recorded here after verification.

The current scope is a frontal, obstacle-free approach to the marked basket.
The top-pair method is a coarse navigation estimate, not a six-DoF grasp pose.
Arbitrary starting headings, unseen basket shapes, transport and obstacle avoidance
remain unvalidated. VLA training/inference history and action synchronization
remain separate work; these videos do not certify student learnability.
