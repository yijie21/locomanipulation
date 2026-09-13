# Visual low reach: handoff, capture region and information audit

Baseline: `3fa486a`, last physical candidate `f16119c`. Work branch:
`fix/crate-visual-low-reach-20260913`. Runtime evidence is retained in
`outputs/crate_visual_low_reach_20260913/`; source videos remain untouched.

## Findings and changes

The user accepted the previous deep hip hinge and reported a jump after walking,
slow descent/closure, and uncertainty about whether those episodes were visual.
The following implementation facts explain why those concerns require separate checks.

1. `StreamedArm.start()` emitted 20 planner frames carrying neither upper-body nor
   wrist references before POSE takeover. The actual-consumer regression fails
   with that count. Whole-body callers now prime POSE while the planner reference
   remains present, bridge the latch-clearing interval with measured wrist/neck
   poses and retain the full base orientation. The final seed composes the fresh
   planner joint/root reference with the currently owned upper-body override,
   rather than turning measured tracking error into a new reference step.
   Legacy callers retain their previous default handoff.
2. A nominal 2.2 s close lasts 4.44 s in a reproduced 40 ms control loop. Recording
   also performed an extra blocking observation read during body/arm feedback.
   Carry now reuses the control observation and advances close/lift by elapsed
   time. The baseline's successful lifts/holds took 10.9–11.8 s for a nominal 5 s.
3. The current crate asset has no handle-slot posts, in both mesh and collision.
   The old carry's +/-25 mm along-bar limit still cited the removed 73 mm slot.
   `RimRegion` represents the open segment with end-wall clearance, transverse
   alignment and vertical overlap. Both hands may use opposite along-rim offsets; their
   midpoint must stay within 15 mm of the basket center. Closure does not require returning to the old center or matching one wrist pose.

4. When the measured body lagged the end of bending, alignment reconstructed a
   new target using commanded joints and measured base pose. A consumer regression
   reproduces a 39 cm vertical target step under an intentionally stationary body.
   The transition now retains the planned world staging target. This synthetic
   magnitude is not a measurement of the live robot's jump.

The RGB low-reach policy shares the existing whole-body geometry solver and physical
waist limits. Its observation interface accepts only a visual pose, encoders, IMU,
finger openings and time. It reconstructs a stationary support frame from foot FK;
it has no simulator object-pose/contact argument. Body planning remains a geometric
admission check, not a claim of guaranteed balance or grasp success.

Two perception failures were reproduced from saved input frames. Lower illumination
hid otherwise visible labels: the first replay recovered 34/54 frames with the old
brightness floor, 54/54 after correction. Later the far-wall label physically left
all three images. Floor paint adds visibility at deep lean without changing mass or
collision. Candidate selection preserves multiple observed colour components and
uses geometric agreement; the largest background patch cannot discard a smaller
valid label before fitting. Planar wall/floor and mixed-plane hypotheses follow
the [OpenCV PnP contract](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).

Signed visual velocity is filtered before taking its magnitude. The previous order
interpreted zero-mean 3 mm pixel-fit noise as continuous motion and could prevent
stance admission. A regression also verifies that sustained 0.12 m/s motion still
prevents admission. Lift attachment checks use observed hand advancement, with an
explicit deadline, rather than assuming that the decoder delivered the reference
after a fixed 1.5 s.

## Preserved physical attempts

Every row below is a separate source/scene protocol, never a pooled success rate.
GPU 0 / instance 1 is used; GPU 1 is unrelated. The launcher records commands,
resolved configuration, source identity and deployment receipts in each directory.
The underlying 30 cm scene is the unchanged fixed-footprint scene from the preceding
height investigation; painted variants change only visible labels.

| Directory | Source | Attempts / successes | Observed outcome |
|---|---|---|---|
| `handoff_only` | `2a71b68` | 3 / 2 | Both reached attempts lift/hold; one table-edge rejection |
| `visual_pilot` | `cec1167` | 3 / 0 | Two losses during bend, one during approach; wall paint |
| `visual_shadow` | `ef2d7b9` | 3 / 0 | All bend, then lose the far-wall label while reaching |
| `visual_floor` | `dd80359` | 3 / 0 | Two close bilaterally, then premature lift-probe rejection; one approach loss |
| `visual_flow` | `2dac78e` | 3 / 0 | One reaches lift but tilts; one approach loss, one approach timeout |
| `visual_region` | `e4f2960` | 5 / 1 | Two insufficient lifts, one tilt failure, one align timeout with invalid 0.97 m reset distance |
| `visual_reference` | `c6e6f03` | 3 / 0 | Two tilted lifts, one invalid initial-distance rejection before recording |
| `visual_balanced` | `68c24d4` | 3 / 1 | One low-tilt lift, one align timeout, one approach loss; all start distances valid |
| `visual_final` | `68c24d4` | 10 / 4 | Five timeouts and one approach loss with physics instability; every attempt retained |

In the first handoff-only experiment, the 65 published state samples beginning at
body-plan admission had peak base angular rates 2.344/2.355 rad/s. The earlier
posture batch had 3.375–4.201 rad/s in the same named window. This is an observed
reduction, not disappearance of all transients; stands and asynchronous histories
differ. The balanced pilot measured peaks 2.358/2.795 rad/s and P95 1.367/1.661
rad/s for its two reached attempts. A brief handoff transient remains.

In the floor-label pilot, the two reached attempts progressed from bend completion
to closure in 2.64/4.04 s, then closed in about 1 s. The earlier successful privileged
posture batch took 5.7–16.4 s for its grasp phase. Those are different experts and
placements, so they establish actual phase durations rather than a controlled
success-rate improvement. Both new attempts failed the subsequent probe.

Reproduce visual phase measurements with
`scripts/experiments/crate_visual_low_reach_report_20260913.py RUN... --out REPORT.json`.
It joins recorded state sequence numbers to complete simulator traces and measures
actual torso angle, base motion, crate rise and phase durations after the attempt.
`early_report.json` covers the first three visual pilots; raw results remain available.

The pair-balance constraint follows a physical failure comparison: the one successful
`visual_region` grasp had an along-rim pair midpoint 5.5 mm from the basket center;
five insufficient/tilted lifts in `visual_region` and `visual_reference` were
28–59 mm off-center. Both hands on the same end can individually capture the rim
while leaving a grip axis that does not support the basket center. This association
motivates the constraint; the final batch below tests the resulting behavior.

Deep-carry recovery also moved the base before recording in two earlier attempts.
Initialization now finishes upright reference preparation, reapplies the same
sampled reset before recording, then measures the actual starting distance. Starts
outside 1.2–1.5 m are rejected before video; no placement is resampled or reset
during the recorded approach.

The balanced pilot's success took 3.237 s from the end of the bend reference to
closure, 1.017 s to close, and 4.283 s to reach the observed lift hold. It reached
58.9 degrees torso pitch at closing, with 6.83 cm final rise and 1.60 degrees tilt.
The rejected align attempt retained a 3–6 cm height error in one hand; it was not
waiting at a geometrically valid pinch. Its timeout remains a failure. The approach
loss in this pilot includes a basket ejection in the recorded physics trace; do not
attribute every missing RGB fit to a perception-threshold problem. Its mechanism
has not been isolated.

## Fixed ten-attempt batch

`visual_final` ran one continuous ten-attempt process with seed 41 (placement seeds
41 through 50), the existing 30 cm tabletop / approximately 44.5 cm grip height,
SONIC v1.1, encoder enabled, GPU 0 / instance 1. Actual initial table-edge distances
were 1.225–1.486 m. The control process loaded `68c24d4`; the later observation-boundary
correction is `1ca11b3` and is evaluated separately, not retroactively attributed here.

```bash
LOCOMANI_ROOT=/workspace/code/locomanipulation-crate-visual-low-reach-20260913 PYTHONPATH=src \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u -m locomani.pipeline.run_crate_pilot \
  --expert visual --rounds 10 --seed 41 --gpu 0 --instance 1 \
  --config configs/tasks/crate_visual_low_reach.json \
  --scene outputs/crate_visual_low_reach_20260913/scene_visual30_final \
  --out outputs/crate_visual_low_reach_20260913/visual_final
```

Use a fresh `--out` when reproducing. `commands.jsonl`, `config_snapshot.json`,
`task_config.json`, `deploy_receipts.jsonl`, `sim_full.log` and `deploy_full.log`
retain the complete launch protocol. Identities for this batch:

| Input | SHA-256 |
|---|---|
| Resolved task configuration | `3522485811484d1f62f05f20dfd03884920707c9a0f7b6f0df3fc3b0d903dd3c` |
| Copied scene XML | `56e98f609c5e0c60575b675b94965ac0ab3c6fe9ec8e52f1c93eaffd4084b8ee` |
| Scene metadata | `ea8607574509e22d1c9291f0cf12d76d2ab1f108b4eb42ea47f1c2a4d6e9b3f4` |
| Encoder | `fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae` |
| Decoder | `34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509` |
| Deploy binary | `da1cc6119374ac5890a91639f9c75f587b95e0f4030db3854293dc82aff2b83f` |

Results below retain the original manifest metrics, measured after cleanup in this
revision. `final_report.json` additionally joins the last recorded state to the
physics trace; it does not rewrite the original verdicts.

| Episode suffix | Result | Initial edge m | Wall seconds | Manifest rise cm | Manifest tilt degrees |
|---|---|---:|---:|---:|---:|
| 00 | timeout | 1.442 | 40.70 | -0.00 | 0.00 |
| 01 | timeout | 1.403 | 40.81 | 4.86 | 10.73 |
| 02 | approach_vision_lost | 1.445 | 4.70 | 0.00 | 0.00 |
| 03 | completed | 1.486 | 21.18 | 7.82 | 4.52 |
| 04 | timeout | 1.430 | 40.84 | 0.00 | 0.00 |
| 05 | timeout | 1.380 | 40.70 | 0.00 | 0.00 |
| 06 | timeout | 1.225 | 40.25 | 0.00 | 0.00 |
| 07 | completed | 1.402 | 20.67 | 8.58 | 6.92 |
| 08 | completed | 1.381 | 23.68 | 8.15 | 5.69 |
| 09 | completed | 1.486 | 20.98 | 8.98 | 1.42 |

For the **four successful attempts**, bend-reference completion to closure took
3.066–6.188 s (median 3.480 s), closure took 1.003–1.033 s, and lift-reference/response
took 4.205–4.456 s before the two-second observed hold. Closing torso pitch was
56.6–61.2 degrees. Recorded end-state rise was 8.05–8.89 cm and tilt 1.24–7.18 degrees.
These timings do not include rejected attempts as if they were successful grasps.

Across all nine reached attempts, the handoff window's base angular peak was
1.895–2.795 rad/s and P95 was 0.332–1.727 rad/s. This remains a visible physical
transient; reference continuity has improved but zero shaking is not demonstrated.
The walking brake/quiet interval plus handoff still took about two seconds in
successful episodes. No video frames or delays were removed to improve these numbers.

Episode 02 is a physics failure as well as a visual abort. `sim_full.log` reports
`Nan, Inf or huge value in QACC at DOF 39` at simulation time 5.14 s; the raw trace
contains an implausible basket excursion. The post-cleanup zero rise in the old
manifest must not be interpreted as an undisturbed basket. Keep this failed attempt;
its numerical/contact cause is still unresolved. Five other failures timed out:
four during alignment and one during an insufficient/stalled loaded hold.

This batch establishes a working observed low-reach path and measured improvements
within successful attempts. **4/10 is not reliable collection readiness**, and these
marked visual attempts are not directly comparable to the earlier 6/10 privileged
mechanical pilot. The scene, reset distribution and decision inputs differ.

## Information and training audit

| Signal or operation | Role and boundary |
|---|---|
| Basket RGB, visible colour labels, fixed camera/object geometry | Authorizes navigation and grasp targets; scene priors are explicitly visible |
| Measured joints, base orientation, finger openings | FK, support-frame estimate, tracking and physical jaw opening; recorded |
| Previous RGB estimates, references, phase and elapsed valid time | Explicit expert history; occlusion pauses progress and resets acceptance intervals |
| Object pose and contact truth | Reset before recording and after-attempt evaluation; never fills a missing visual target |
| Fall flag | Marked external abort, not a successful expert recovery |
| Full reference and root orientation | Recorded with image/state timestamps for inspection; mode takeover remains in overview/trace |

The previous `--expert carry` videos are **privileged mechanical pilots**, not VLA
visual demonstrations. The new `--expert visual` low-reach path satisfies the above
decision boundary in the tested scope. Missing/stale RGB, shadowed labels, visible
placement changes, off-center capture, and untracked lifting have consumer tests.
A further consumer regression catches stale nonzero alignment velocity on the
first reacquired frame; pausing now resets the joint-reference velocity as well as
its position clock. This does not establish successful student learning or universal
grasping.

Formal VLA collection remains blocked. `SONIC_CRATE` already names a 64-dimensional
motion token and both grippers; it is inaccurate to infer from the new body reference
alone that the action schema necessarily lacks waist control. What is unverified is
token/observation synchronization, complete mode-handoff labels and replay through
the real consumer. `configs/gr00t_sonic_3cam_bimanual.py` uses `delta_indices=[0]` for
images and state, while the expert uses history. Merely writing history to JSON does
not make it a VLA input. Validate that temporal contract before enabling `--record`.

Known scope: marked crate, fixed known geometry, flat-floor double support, frontal
approach, no transport. Foot slip and severe marker ambiguity can invalidate the
support/pose estimate. Preserve rejection and failed-hold episodes as failures.

A separate evaluation regression found that reading final state after video
finalization and mode exit could measure cleanup behavior rather than the attempt.
The evaluator now uses the last control observation and records its state sequence.
Earlier manifests remain unchanged; the report additionally measures the first/last
recorded trace states to expose any difference. This correction does not change
actions or upgrade an earlier failed verdict.

## Checks and artifacts

Selected consumer tests and the full pre-commit `make check` pass for committed
candidates. Red/green logs, saved RGB replays, every manifest, full overview/trace,
scene copy and deploy receipt are under the evidence root. Viewing exports follow
the mandatory [compressed preview rules](../rendering.md#video-export).
The ten `previews_final/*.mp4` files are 352,315–453,437 bytes each, 640×480 at
10 FPS. All passed full decode, original-duration rounding (at most one output
frame), and first/last-frame inspection. The source overview's fixed frame rate
can differ from wall-clock elapsed time; exports preserve its full timeline and
speed. Use recorded timestamps for timing analysis.

- Representative: `previews_final/visual_crate_s41_09.mp4`, 419,098 bytes.
- All ten, including failures: `crate_visual_low_reach_10_previews.zip`, 4,132,022 bytes.
- ZIP SHA-256: `ab7f321997ca0fc93b4de5a2340c777ce3c0e0f7f3b0b0969b2121489653b095`.
- Package contents: ten previews, README, results CSV and `video_checks.json`;
  originals, consumed RGB streams and traces remain outside the download package.

The `carry_regression` run used source `1ca11b3`, the unpainted 30 cm scene,
`--expert carry --config configs/tasks/crate_low_reach.json --rounds 3 --seed 41`
and the same GPU/instance. It completed 1/3: one successful lift, one unilateral
contact after closing, one table-edge rejection. The two reached attempts closed
in 2.217/2.214 s; the successful lift plus hold took 5.003 s, with 8.12 cm rise and
3.14 degrees tilt. This exercises the actual shared timing/handoff path; it does
not establish unchanged success probability or remove the retained failures.

The final `visual_boundary_check` used source `1ca11b3`, the same visual profile
and painted scene, three attempts with seed 41. It completed **1/3**: `00` aligned
but timed out, `01` completed with 8.05 cm rise and 3.74 degrees tilt, `02` was
rejected by the geometric reach admission. All three recorded `evaluation_state_seq`
and started 1.403–1.445 m from the table edge. This separate check validates the
current entry and end-state recording; it is not pooled with the ten-attempt batch.
The full CPU check and actual-consumer regressions pass, including stale-velocity
reacquisition and evaluation before cleanup. GPU 0 / instance 1 is released; GPU 1
was not used by this work.

The final-version success is `previews_boundary_check/visual_crate_s41_01.mp4`
(427,226 bytes). Its alignment took 3.373 s, closure 1.023 s, lift 4.199 s and hold
2.039 s; closing torso pitch was 58.9 degrees. `boundary_report.json` joins evaluation
sequence 5570 to the final trace and agrees within 0.02 mm / 0.002 degrees with the
recorded evaluation. The three final-version previews (308,216–427,226 bytes) also
pass size, complete decode, full-duration and first/last-frame checks.
