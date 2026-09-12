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

The first fixed batch `collection_10_v1` at `1445f98` completed 6/10 successes.
Independent Standards review found no material issues. Spec review found two:
intervals above 0.1 s were truncated by the acceleration limiter, and the closure
gate tested commanded orientation without checking measured wrist orientation.
Both regressions were reproduced by failing CPU tests, then fixed. Actual elapsed
time is retained; both measured wrists must be within 10° of the desired rotation
and the reference rotation must finish blending. At first closure in the old batch,
measured wrist errors ranged from 1.92° to 8.06°; this informs the tolerance but
does not validate physical contact. Tests cover 0.2/0.3 s gaps and nontracking wrists.

The corrected fixed batch `collection_10_v2` at `958f5f9`, seed 20260912,
completed all ten attempts: **3 success, 3 approach timeout, 3 reach/hold timeout,
1 visual-tilt exit**. Measured initial edge distances were 1.278–1.479 m.
Seven episodes entered reaching after walking and a quiet interval; three remained
in approach until its timeout. Those approach failures showed persistent lateral
error of roughly 5–11 cm while minimum-speed commands did not converge. This is
an observed navigation limitation, not evidence that the basket was unreachable.
Two of the three grasp timeouts reached hold; one remained in pregrasp.

The seven episodes with pre-contact samples had torso angular-speed peaks
17.76–25.33°/s and reference acceleration peaks 2.000 rad/s² (rounding error
<2e-13). The original recorded-symptom gate passes for those seven; the three
without reaching are explicitly **not evaluated**. The original grasp-only batch
still fails the same checker. This comparison supports reduced abrupt reference
motion, not equivalent success probabilities or fully solved loaded stability.
Raw metrics: `collection_10_v2/shake_evaluated.log`; empty-phase handling was added
to the diagnostic script after its original version failed on non-reaching episodes.
The original script and traceback are retained, as is `baseline_shake_recheck.log`.

Reproduce the corrected collection:

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.pipeline.run_crate_pilot --expert visual --rounds 10 \
  --seed 20260912 --gpu 0 --instance 1 \
  --scene outputs/crate_visual_expert_20260912/scene_marked_v3 \
  --out outputs/crate_walk_stability_20260912/collection_10_v2
```

Use a new output directory when rerunning. Scene/model identity, resolved config,
source hashes, all ten outcomes, RGB observations and state traces are retained.
The command records a fresh asynchronous trial; it does not promise identical results.

All ten compressed previews retain complete duration and normal speed, with full
FFmpeg decode checks and duration difference within one output frame. Individual
sizes are **286,394–471,610 bytes**; ZIP size **3,679,111 bytes**. The exporter,
checksums, codec/duration records and original-to-preview start/end contact sheet
are in `collection_10_v2/download_preview/`. Its `index.html` links only small
previews; `videos_10_small.zip` contains those ten and small metadata. Originals
and both ten-episode batches remain untouched.

Full CPU checks and independent Standards/Spec reviews pass. Spec findings were
reproduced and fixed; follow-up tested 4,500 limiter updates with intervals up to
0.69 s, reversals and starts near joint limits without bound violations. The ten
corrected GPU trials provide live behavior evidence with the failures above.

The current scope is a frontal, obstacle-free approach to the marked basket.
The top-pair method is a coarse navigation estimate, not a six-DoF grasp pose.
Arbitrary starting headings, unseen basket shapes, transport and obstacle avoidance
remain unvalidated. VLA training/inference history and action synchronization
remain separate work; these videos do not certify student learnability.
