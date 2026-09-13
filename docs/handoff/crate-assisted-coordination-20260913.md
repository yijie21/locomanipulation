# Ordinary basket and simulation-assisted whole-body coordination

Date: 2026-09-13. Branch: `fix/crate-assisted-coordination-20260913`.
Base: `6f41bb8`. This implements the user's clarification that simulation may
assist planning throughout an ordinary-basket demonstration. It replaces the
low-table profile's RGB-only restriction, without retracting historical measurements.
Current usage and information contract: [pilot guide](../crate_visual_collection.md).

## Implemented behavior

- `visual.pose_source=assisted` in the low-table profile uses simulation basket and
  robot pose with camera-frustum/scene-ray visibility admission. Each record identifies
  the source; it does not claim that a simulation pose was inferred from RGB.
  The historical `markers` implementation remains explicitly selectable.
- New run scenes omit the `crate_visual_*` labels/plates. The original source and
  all old videos remain intact. An audit compares compiled mass, inertia, collision
  parameters and exact XML identity against the original ordinary-basket scene.
- `CoordinatedReach` re-solves the supported whole-body reference during descent,
  closure and lift. `WholeBodyIK` owns feet, COM and physical limits. Hip/leg joints
  continue moving with the waist and two hands; the lower body is no longer frozen
  after a separate bend phase. A filtered, bounded tracking correction uses measured
  simulation feedback. Capture still requires balanced rim overlap and jaw alignment.
- The reference starts from the planner command actually consumed by SONIC. All
  waist axes stay within physical bounds; pelvis pitch/height and shallow knees
  bound posture. Joint/root speed and acceleration are bounded within the solve.
  The timing correction uses adjacent observation intervals, because camera periods
  vary in actual operation. Existing backend handoff and post-attempt evaluation
  contracts are retained.
- The final profile increases the existing approach speed from 0.10 to 0.16 m/s.
  This shortens the Cartesian descent without increasing joint speed/acceleration.
  It does not increase capture tolerances or move the basket into reach.

## Reproducible inputs and artifacts

Runtime root: `outputs/crate_assisted_coordination_20260913/` (untracked).
All runs use GPU 0 / instance 1, SONIC v1.1 encoder/decoder unchanged, seeds starting
at 41, a 30 cm table with approximately 44.5 cm grip height, and the same source:
`outputs/crate_table30_review_20260913/scene_table30`.
XML SHA256: `a2af77f13691b642fa19dc3a0cb7d674d575f7ad64d375c401036afcb01bc760`.

```bash
LOCOMANI_ROOT=/workspace/code/locomanipulation-crate-assisted-coordination-20260913 \
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
  -m locomani.pipeline.run_crate_pilot --expert visual --rounds 10 --seed 41 \
  --gpu 0 --instance 1 --config configs/tasks/crate_visual_low_reach.json \
  --scene /workspace/code/locomanipulation/outputs/crate_table30_review_20260913/scene_table30 \
  --out /workspace/code/locomanipulation/outputs/crate_assisted_coordination_20260913/assisted_delivery
```

`assisted_probe` is source `a9e9178`, three diagnostic attempts at 0.10 m/s.
`assisted_final` is the **intermediate** source `b8b54f0`, ten attempts at 0.16 m/s
with the adjacent-interval correction. `assisted_delivery` uses source `b256d03`,
which also corrects the loaded reference and adds bounded failure exits.
Rates must not be pooled across these revisions. Each run contains config/source
snapshots, deployment receipts, full manifests, raw videos, actual consumed RGB,
observation records, state traces and preserved sim/deploy logs.

`measure.py RUN` in the runtime root produces `RUN/motion_measurements.json`.
Phase durations use adjacent observation wall timestamps. Torso pitch is FK plus
observed base rotation at the first close frame (last body frame for failures).
Handoff angular speed comes from trace base angular velocity, from the last settle
record through one simulation second after the first body record. Foot displacement
compares trace ankle positions with their first body-frame anchors. These diagnostics
are not a VLA learning test or proof of physical equivalence across asynchronous runs.

## Diagnostic batch

`assisted_probe`: **2/3 success**. Successful torso inclinations at closure were
59.89° and 59.49°, knees 10.40–12.10°. Maximum observed foot XY drift over body
motion was 6.49–9.54 mm. Hip references still changed by 20–26° after the first
three seconds of descent. Reach lasted 11.35 and 8.28 s, closure 1.01–1.03 s,
lift 4.01–4.12 s and hold 2.03–2.07 s. Actual basket rises were 6.90 and 6.47 cm.

The third attempt failed during approach: basket speed jumped from approximately
zero to 15.90 m/s between trace sequences 5651 and 5652, before body planning.
Replay of sampled frames shows table/basket contact before the jump and a finger
contact afterward; it cannot identify an unsampled triggering contact or establish
a root cause. No QACC warning was found in this run's sim log. Preserve this as a
physical ejection with `approach_vision_lost`, not as evidence against ordinary-basket
pose availability. Details: `assisted_probe/walk_ejection_diagnostic.json`.

The diagnostic source's reference acceleration reached 2.82–2.99 rad/s² when
computed across irregular camera periods, above the configured 2.0. This actual
failure motivated the adjacent-interval correction and its consumer regression test.

## Intermediate batch and discriminating corrections

`assisted_final` (folder name retained, **not** the delivered revision): **4/10**.
Successes were 00, 01, 07 and 09; failures were one approach ejection, two reach
timeouts, one hold timeout after a drop and two lift tilt aborts. The first two
successful reach times fell to 7.05 and 7.50 s; this is not a controlled causal
success-rate comparison with the three-attempt diagnostic batch.

An actual-consumer regression reproduced a reference error at closure → lift:
with a 3 cm constant tracking lag, the already compensated held FK was used as the
new lift baseline and then compensated again. Both reference hands descended
about 22 mm before lifting. `lift_boundary_red.log` records that failure;
`lift_boundary_green.log` and `attachment_checks.log` record the corrected tests.
Removing the previously applied compensation from the stored lift baseline keeps
the target continuous. This proves the reference bug, not that it caused every
physical slip or tilt.

Additional corrections in `b256d03`:

- Reserve more braking distance near joint limits. The intermediate body's measured
  acceleration still reached 2.16 rad/s² near waist-roll saturation with irregular
  intervals; no physical-limit jump is accepted as normal continuous motion.
- Abort when a loaded gripper becomes empty or held attachment is clearly lost.
  A stable hold has a bounded waiting period; it cannot run to the 35 s global
  timeout after the basket has already dropped.
- Abort when an ungrasped basket is pushed more than the rim half-length (9 cm),
  or after nominal descent when the hand error makes no 5 mm improvement for 3 s.
  These are failure exits, not permissive success criteria or hidden recovery.

Reference-derivative measurement uses only consecutive active body-command records.
Planner-mode `arm.q` does not contain the consumed walking leg reference, so its
transition to a full POSE record is not an actual commanded joint jump. Terminal
abort holds are also excluded. `unfiltered_reference_metrics.json` preserves the
initial mixed-record calculation; use `motion_measurements.json` for the corrected
body-command scope. Trace angular speed independently measures the actual handoff.

The historical wall gate was run against the intermediate manifest and failed all
three comparisons (yield, exit label, new failure categories). It uses a standing
wall-grasp protocol and an 8/10 baseline, while this task walks to a 30 cm table.
This cross-protocol result does not establish physical equivalence or regression;
no PASS is claimed. See `gate_comparison.log` and
`assisted_vs_historical_wall_gate.json`. Bottle behavior was not exercised in this task.

## Ordinary-basket ten-attempt batch

`assisted_delivery`, source `b256d03`: **4/10 success**, labels 00, 01, 03 and 08.
The six failures are one approach ejection (02), three stalled reaches (04, 05, 07),
one geometric admission rejection (06), and one lift tilt abort (09). Every actual
initial table-edge distance was within 1.218–1.485 m. No failed trial was removed.

For the four successes:

| Measurement | Observed range |
|---|---|
| Torso pitch at closure | 57.87–61.92° |
| Knee flexion at closure | 9.51–13.57° |
| Actual basket rise | 7.57–10.09 cm |
| Final basket tilt | 0.34–8.49° |
| Reach including continuous bending | 7.44–9.24 s (median 7.76 s) |
| Closure / lift / observed hold | 1.01–1.05 / 4.00–4.02 / 2.03–2.06 s |
| Maximum observed foot XY drift | 7.36–14.57 mm |
| Handoff base angular peak / P95 | 1.11–2.99 / 0.30–1.27 rad/s |

This establishes ordinary-basket grasping with a deep hip hinge and shallow knees
in successful trials. It does **not** establish high reliability, elimination of
the handoff transient, or a success-rate improvement over the earlier painted 4/10
batch. Their conditions and asynchronous trajectories differ. Remaining grasp
tracking, contact/tilt failures and approach ejections stay open.

A second timing regression reproduced a smaller acceleration violation when a
camera period exceeded 100 ms: the capped trajectory interval was also used to
calculate the previous reference velocity. An actual-consumer replay measured
2.38 rad/s² instead of the configured 2.0; the live batch reached 2.35. Source
`a8871e2` separates capped trajectory progress from the actual interval used for
rate/acceleration limits. `delayed_frame_red.log` and `delayed_frame_green.log`
record the regression and fix. The ten-attempt rate above belongs to `b256d03`;
the final timing revision receives its own check, without pooling attempts.

`delivery_previews/video_checks.json` verifies all ten complete-duration previews:
234,573–480,368 bytes each, original aspect/speed, maximum duration difference
0.04 s, full ffmpeg decode, and inspected first/last frames. The original basket's
mesh is visible without the added wall/floor plates or colored dots.
`ordinary_basket_10_previews.zip` is 4,031,378 bytes and contains only these previews
plus a small result index. Original recordings remain in `assisted_delivery`.

CPU checks passed at every committed controller revision, including full pre-commit
format/lint/navigation/config checks and unit suites. Consumer tests cover copied
scene restoration, source/config delivery, visibility/occlusion, supported body
motion, irregular/delayed camera periods, loaded-reference continuity and immediate
empty-gripper rejection. No VLA learning or formal dataset export was validated.

## Final timing check and delivery boundary

`timing_check`, final controller source `a8871e2`: **2/3 success**, seeds 41–43.
The third attempt again ejects the basket during approach, before body planning.
Both successful active body-reference streams stay at or below 0.70 rad/s and
2.00 rad/s² under the corrected measurement. Their reach times are 6.45 and
7.29 s, closure 1.02–1.03 s, lift 4.01–4.02 s and hold 2.04–2.05 s.
Closure torso pitch is 60.63° / 59.19°, with knees 9.32–11.05°.
Actual handoff angular peaks remain 2.21 / 2.40 rad/s: reference continuity does
not mean the physical mode-transition wobble has disappeared.

This small check supports the delayed-frame correction in its measured scope.
It is not a new ten-attempt reliability estimate. The ordinary-basket ten-video
package above is explicitly from `b256d03`; latest-source previews are separate
under `timing_previews/`, with their own full-duration and size checks.
GPU 0 / instance 1 has been released. GPU 1 was not controlled by this task.
