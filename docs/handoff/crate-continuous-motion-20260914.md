# Assisted crate motion continuity — 2026-09-14

Implementation branch: `fix/crate-continuous-motion-20260914`, based on `e29857a`.
The requested scope is the four changes following the [continuity audit](crate-continuity-audit-20260914.md): independent camera/control updates, overlapping preparation/reach, continuous loaded recovery, and a training boundary before verification. The current usage contract is in the [pilot guide](../crate_visual_collection.md).

## Implementation

1. `VisibleTarget` caches a camera-admitted world pose and its original timestamp. Each new physical state advances control, including ticks without a new image. Robot motion re-expresses that same target rather than moving it in world space. A newly occluded view invalidates it; the reference brakes with bounded acceleration and stale input aborts.
2. `ApproachPreparation` lowers/orients both hands within a small walking lean envelope while approaching the visible basket. Assisted stance admission uses current measured base velocity and rotation, maintaining the existing continuous quiet interval and supported-reach check. Full reach couples torso rotation, hands and hips through one progress variable; all waist axes remain available.
3. After measured attachment is demonstrated by the probe, the lift continues while recovery of torso/hips begins. The phase label changes without resetting the motion clock. The paired nominal hand-height path is nondecreasing. Actual attachment, rise, displacement, tilt, standing and hold acceptance still apply; failed contact is not relabeled.
4. The transport journals every successfully sent planner/POSE reference, including priming and reused-image ticks. Successful results publish an inclusive `training_tail.end_seq` at entry to the upright loaded hold, plus command/image/time joins. This marker is withheld if subsequent validation fails; raw video and traces retain the entire attempt.

No task tuning parameters were added. The existing lift/stand durations now describe overlapping ramps in the assisted path. Marker and mechanical experts retain their previous transport mode. Formal LeRobot/token/VLA synchronization remains separate and `--record` is still refused for visual pilots.

## Additional cause found during validation

Fixing the repeated-image zero-velocity send and trajectory boundaries alone did not remove the large reference-to-body delay. `probe_v1` still delivered the torso roughly two seconds after its reference and failed all three attempts (two unstable holds, one tilt failure).

The matched deploy's `deploy.cpp` reserves `(num_frames - 1) * step_size + 1` samples for future observations and stops streamed playback when it enters that reserved tail. With the v1.1 configuration (10 frames, stride 5), this is 46 frames. The old sender supplied only past samples. In the inspected live log the merged window reached 53 frames, with 48 copied before the five incoming frames: the newest command was being queued behind the observation reserve. A slow sender stretched those frames over physical seconds.

`PosePreview` now resamples the current reference on the physical 50 Hz timeline and sends five historical ticks plus 50 future ticks. The prediction follows the current joint/root tangent while continuously braking to rest, and is replaced at the next update. Each pose carries its own root and consistent joint velocity. These are explicit future predictions, not labels of already executed actions. No deploy binary or policy weights were changed.

Source of this contract: `outputs/crate_upright_handoff_20260913/deploy/deploy.cpp` (`saved_frame_for_observation_window_`, `UpdateMotionFrame`) and its v1.1 configuration. The run receipts identify the exact binary and source inputs. The CPU consumer test exercises the sent packet and the 46-frame playable-window condition. GPU results test the physical effect; asynchronous trials do not establish a precise causal success-rate estimate.

## Protocol and preserved attempts

Artifacts: `outputs/crate_continuous_motion_20260914/`.

- Baseline: `outputs/crate_upright_handoff_20260913/clock_delivery`, seeds 41–50, 6/10 successes. The [audit](crate-continuity-audit-20260914.md) verifies identity of the eight relevant baseline control sources with `e29857a` and documents metric definitions.
- `probe_v1`: first four-change implementation, past-only sender, seeds 41–43; 0/3 (two `unstable_hold`, one `basket_tilt`). Preserved to distinguish code implementation from observed continuity.
- `probe_v2`: adds the future window, seeds 41–43; 3/3. This is a small diagnostic batch, not a reliability claim.
- `probe_v3`: uses current base velocity for stance admission and stops sending stale nonzero velocity on terminal safety failures, seeds 41–43; 2/3, one `basket_tilt` during hold.
- `final10`: fixed seeds 41–50; **6/10 successes**. `00,01,03,05,07,09` complete; `02,04` fail on basket tilt, `08` on stalled reaching, and `06` is rejected before recording because its measured initial distance is 1.19165 m. Nine raw videos exist; no substitute video or resampling hides the rejected initialization.

All use the ordinary basket, 30 cm table, same scene source and matched v1.1 `POSE_UPPER_OWNERSHIP_V2` binary. The actual initial distance is checked to remain within 1.2–1.5 m. GPU 0 / instance 1 is launched through the existing pilot runner; GPU 1 belongs to another project and is untouched.

```bash
LOCOMANI_ROOT=/workspace/code/locomanipulation-crate-continuity-audit-20260914 \
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
SONIC_DEPLOY_BIN=/workspace/code/locomanipulation/outputs/crate_upright_handoff_20260913/deploy/target/release/g1_deploy_onnx_ref \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
  -m locomani.pipeline.run_crate_pilot --expert visual --rounds 10 --seed 41 \
  --gpu 0 --instance 1 --config configs/tasks/crate_visual_low_reach.json \
  --scene /workspace/code/locomanipulation/outputs/crate_table30_review_20260913/scene_table30 \
  --out /workspace/code/locomanipulation/outputs/crate_continuous_motion_20260914/final10
```

## Measurement and limitations

`measure.py` joins observation `state_seq` to exact recorded MuJoCo qpos/qvel and computes world tool/basket positions and Jacobian velocities. It uses simulation time; missing sequence matches are excluded, never interpolated into physical evidence. `transitions.py` checks basket drawdown/rebend at the loaded transition. `command_metrics.py` checks the current sample separately from the future tail, counts control updates reusing images, and also measures maximum drawdown across the *entire* loaded trajectory so moving a dip outside the first two seconds cannot hide it.

The baseline and candidate are asynchronous physical experiments with matching seeds/assets/policy, not deterministic paired trajectories. Phase meanings intentionally changed: candidate `stand` begins at the attachment probe while the lift continues. Changes in task progress and physical motion matter more than shorter labels. Packet logs prove sends and their values, not which frames the deploy actually consumed. The reference-window mechanism, measured FK traces and rendered videos provide complementary evidence.

A continuous policy still needs braking, contact closure and a measured stable stance. Zero pauses under every physical condition is not claimed. Retain real tilt/contact/navigation failures; the user permits imperfect crate reproducibility. Training success has not been demonstrated and the final verification hold must not become repeated idle supervision.


## Final GPU measurements

Ranges below cover each batch's six successful attempts (different success cohorts,
not a paired statistical estimate). All times use the physical simulation clock.

| Metric | Baseline | Final candidate |
|---|---:|---:|
| Stop-walking command → 5° additional torso bend from reach entry | 4.40–4.94 s | 2.88–3.10 s |
| Reach entry → closure | 6.26–9.56 s | 2.62–3.48 s |
| Reach beyond its nominal trajectory duration | 2.22–5.56 s | 0–0.48 s |
| Largest basket drawdown in first 2 s of recovery | 1.72–4.06 cm | 0.13–1.16 cm |
| Entire recorded physical duration | 24.02–27.90 s | 16.52–17.70 s |
| Longest measured quiet interval in the entire recording | 0.42–0.94 s | 0.10–0.34 s |

“Quiet” requires both hands below 2 cm/s, base speed below 3 cm/s and torso angular
speed below 5°/s; gaps over 0.25 s are excluded from that metric. There is still
real braking/stance admission: stop-command → reach spans 1.60–2.02 s. It is not
reported as zero dwell. The intended 0.3 s continuously quiet admission interval
remains, along with support solving and the bounded mode handoff.

Across the **whole loaded trajectory**, the candidate's maximum basket drawdown
is 0.86–1.16 cm. Thus small physical backtracking remains despite the monotone
nominal height path. Successful end torso lean is 0.33–2.55°; maximum reach lean
is 59.87–62.08°. This improves motion continuity but does not establish general
reliable grasping: the ten-attempt success count remains 6/10, with the failures above.

The nine recorded attempts contain **4,330 sent reference messages**. Of these,
120 moving POSE updates reuse the previous image. None holds the current joint
position while claiming a nonzero joint velocity. For current samples outside
priming/initialization, maximum velocity-versus-position finite-difference error
is 5.52e-6 rad/s (float32 transport precision); maximum joint speed is 0.70 rad/s.
Successful episodes still have individual control-observation gaps up to
0.32–0.38 s, covered by the bounded prediction, so this is not a claim of a
hard real-time 50 Hz planning loop. The sender's trajectory samples are at 50 Hz.

Each recorded attempt has one leading observation sequence absent from its
physics trace; that row is excluded from FK/velocity measurements. Successful
training boundaries remove 2.04–2.06 s of final verification from future supervision;
all raw recordings remain complete. Failure results contain no successful boundary.

`final10/config_snapshot.json` includes the new untracked-at-run-time modules and
matches the delivered source hashes. The manifest SHA-256 is
`6843f5da3213e1cb5d807e01a7324e9f19a71f5314df7ed2c12cdd614b113ac7`.
`artifact_sha256.json` also identifies receipts, raw observations/commands, physics
traces, runtime logs, previews and ZIP.

## Checks, artifacts and next work

Development checks cover actual repeated-image execution, world-frame caching and
occlusion, braking/resume, supported coordinated motion, attachment failures,
monotone nominal loaded targets, rate/acceleration limits, transport payloads and
future-window availability, handoff ownership, preparation/admission, per-send
recording, and withholding a successful training boundary after failed validation.
The delivery commit uses the installed repository hook for final lint and the
complete default CPU suite; the separate `make check` entry is not repeated.

All nine previews pass actual-size checks, complete ffmpeg decode, duration and
aspect-ratio checks. All start/end frames were inspected, with full-duration contact
sheets for `00` and `08` and the `00` physical-motion plot. Previews are
237,926–371,179 bytes; `crate_continuous_motion_10_attempts.zip` is **2,995,887 bytes**,
contains the nine previews plus a small README/results journal, and passes ZIP CRC
verification. The README explicitly explains why attempt `06` has no video.
Source videos, failed probes and earlier large artifacts are preserved.

The four requested code changes and live validation are implemented. Remaining
work is improving contact/tilt robustness and initialization drift in this scene,
then validating token/history synchronization, both-gripper inference, dataset
conversion/trimming and actual VLA learning. These are not claimed complete by
this continuity revision. GPU 0 / instance 1 was released; GPU 1 was untouched.
