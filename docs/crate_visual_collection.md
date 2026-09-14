# Camera-guided crate video pilots

Use `--expert visual` for demonstrations that approach a visible basket and grasp
its rim. The low-table profile uses simulation-assisted planning with the original
basket appearance. It records the walk from 1.2–1.5 m away through the attempt.
These are video/control pilots; they do not yet certify VLA training or learning.
The [continuity revision](handoff/crate-continuous-motion-20260914.md) records the
current implementation and live checks. The earlier [continuity audit](handoff/crate-continuity-audit-20260914.md)
and [loaded-standing batch](handoff/crate-upright-handoff-20260913.md) retain the pre-change evidence.

## Information used by the expert

“Visual guidance” describes the relationship between the visible scene and the
action. It does **not** require an expert restricted to RGB pose estimation.
Simulation object pose, robot state and geometric constraints may help produce
smooth, reproducible references. Their source must be recorded honestly.

`visual.pose_source` selects one implementation:

| Source | Planning inputs | Appearance and admission |
|---|---|---|
| `assisted` (low-table profile) | Simulation basket pose, measured joints/base pose, camera geometry, scene occlusion geometry | Original basket; fresh camera frames and a basket visible in at least one recorded camera |
| `markers` (historical RGB experiment) | RGB marker fits, measured joints/IMU, known marker geometry | Added wall/floor labels; RGB fit required near grasping |

The assisted path uses frustum projection and scene rays to admit a visible basket.
It is **not** an RGB detector. Three visible surface samples spanning at least eight
pixels in one camera are required; blank frames are rejected. Camera mount bodies
are excluded from their own rays, and hidden geometry groups are excluded. This
geometric check is an approximation of recorded visibility, not a pixel-level
segmentation guarantee. RGB/state timestamps and source labels remain inspectable.

Simulation may improve the target and trajectory; it cannot teleport the basket,
reset the robot during the demonstration, move the object into reach, advance
an unseen target indefinitely, or turn failed contact into a successful label.
A newly occluded view stops target progress and brakes the reference; stale input eventually aborts. The same
visible placement determines the target; arbitrary hidden episode flags do not
select grasp styles or unrecorded recovery branches. This constrains potential
hidden action causes but does not prove that a particular VLA can learn the policy.

The historical [RGB low-reach batch](handoff/crate-visual-low-reach-20260913.md)
used a painted scene and succeeded 4/10. Its observations remain historical
measurements; that rate must not be attributed to the assisted controller.
Earlier [walking](handoff/crate-walk-stability-20260912.md) and
[grasp-only](handoff/crate-visual-pilot-20260912.md) evidence is preserved.
The original [design discussion](handoff/crate-visual-expert-design.md) predates
this explicit permission to use simulation throughout planning.

## Run

Follow [AGENTS](../AGENTS.md) for GPU/instance admission. The example uses instance
1 and GPU 0; check availability first. Build a crate scene following the
[crate guide](crate_collection.md). For a lowered table use the
[table-height procedure](crate_collection.md#table-height-validation).
The whole-body entry also requires the [POSE ownership deploy patch](../patches/sonic_pose_ownership.md).

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.tasks.crate.bimanual_scene outputs/scene/crate_bimanual
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/crate_table_height_20260913.py \
  outputs/scene/crate_bimanual outputs/scene/crate_table30 --height 0.30

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.pipeline.run_crate_pilot --expert visual \
  --config configs/tasks/crate_visual_low_reach.json \
  --scene outputs/scene/crate_table30 --out outputs/crate_assisted_pilot \
  --instance 1 --gpu 0 --rounds 10 --seed 41
```

The profile does not alter table height. It selects `carry.reach_control=whole_body`
and `visual.pose_source=assisted`. The runner copies the source scene and removes
the experimental `crate_visual_*` plates/markers from the copy. The support table's
box collider is represented by an equal-size eight-vertex convex mesh to avoid
MuJoCo's analytic box/box phantom contacts. Basket appearance and its gripper
contact geometry remain unchanged; the original source is preserved. Use a fresh output
directory for each experiment; old artifacts must remain intact.

Configuration is resolved as defaults → profile → explicit `--set` overrides,
then serialized to the expert process. See the generated
[field reference](reference/config_fields.md). Assisted control requires the
whole-body path. To reproduce the older marker path, build a distinct scene with
`python -m locomani.tasks.crate.visual_scene SOURCE DESTINATION` and explicitly
set `--set visual.pose_source=markers`.

## Low tables and a free waist

`CoordinatedReach` owns one supported body reference throughout descent, closure
lifting and loaded standing. `WholeBodyIK` owns foot anchors, balance and physical posture limits.
The existing `carry` expert remains a separate mechanical diagnostic entry.

- Walk from the sampled 1.2–1.5 m table-edge distance. Prepare the two hands and a
  small waist lean as the visible basket approaches, with joint rate limits while
  the walking planner still owns the feet. A continuously quiet measured base
  interval authorizes a support check before handoff. The assisted path uses
  current base velocity/rotation, avoiding extra waits for the visual speed filter. Unsupported reaches use the remaining
  braking allowance for a closer visible approach, bounded by `visual.stand_off`;
  unsupported placements can still fail admission.
- Preserve the consumed planner joint/root reference at POSE takeover. The backend
  switches directly into a primed POSE stream and releases the planner override
  atomically. Descent begins from that reference; there is no intermediate VR stage.
- Move both hands, hips and all three waist axes together. Compensate bounded
  measured tracking error with the whole body; do not freeze the legs after bending.
  Torso rotation and hand descent share the same progress variable.
  Joint/root rates, acceleration, planted-foot error, support COM, pelvis height,
  waist limits and shallow knee flexion constrain the solution.
- Treat each no-post rim as a segment, allowing different along-rim hand positions.
  Keep the pair midpoint within 15 mm of the basket center. End-wall clearance,
  jaw alignment and vertical overlap still constrain closure.
- Maintain the paired reference during closure and lift with actual attachment,
  basket rise and tilt checks. The lift baseline must not apply tracking feedback
  twice. Stop on empty grippers, clear attachment loss, displaced baskets or stalled
  reaching; retain failed and timed-out attempts.
- Once the observed attachment probe passes, continue lifting while moving the
  paired grasp toward its upright preparation centre and recovering hips/torso.
  The lift and recovery ramps overlap on one clock: changing the phase label does
  not reset the reference or its velocity. Nominal hand height never decreases;
  measured height can still deviate and must be checked in physics traces. Success requires actual torso lean within
  `visual.upright_tolerance_deg`, preserved attachment and a quiet load/body throughout
  `visual.hold_seconds`. The load must stay within `visual.hold_position_tolerance`
  of its mean position over that complete window, allowing small stance sway while
  rejecting sustained drift. Reaching a commanded upright pose alone is insufficient.
  Failed attachment, failed standing and unstable holding have separate terminal reasons.

`VisualLowReach` retains the older marker-based policy for historical experiments.
It reconstructs base translation from stationary foot FK; that information contract
must not be confused with the assisted controller's measured simulation base pose.

Camera updates and reference execution have separate lifecycles. New images admit
or invalidate a basket pose cached in world coordinates. Between images the pose
is re-expressed using current robot proprioception, retaining its original camera
timestamp. Each new physical state advances the controller; reused images do not
insert a stationary joint command with zero velocity. Actual simulation object
state independently checks displacement, attachment, tilt and stability.

The assisted POSE sender uses [PosePreview](../src/locomani/backends/sonic/pose_preview.py).
SONIC v1.1 reserves 46 future frames for its encoder; sending only past samples
makes execution lag behind the newest reference. The sender resamples references
at 50 Hz in physical time, retains five historical ticks and supplies 50 future
ticks that continuously brake along the current body trajectory's tangent to rest.
Each update replaces that prediction; it is neither a new grasp target nor an
unbounded extrapolation. Every pose keeps its own root rotation and joint velocity.
The older marker and mechanical experts keep their existing send mode.

The assisted expert uses **simulation time** for motion rates, settling, phase
durations and task deadlines. Camera freshness and the watchdog for a stalled
simulator use wall time. A busy machine therefore gives the physical simulation
time to execute its reference without accepting stale images. Before recording,
the reset must deliver a measured quiet upright stance; unsuccessful initialization
is reported with zero video frames. The actual 1.2–1.5 m start-distance check remains.
This clock contract does not make the asynchronous neural controller deterministic
or eliminate sensitivity to missed control updates.

## Outputs and limits

Before sharing or packaging videos, follow mandatory
[video export](rendering.md#video-export): every preview is at most 500,000 bytes,
with complete duration/speed and verified playback. Preserve original recordings.

- `videos_00/*.mp4`: raw simulator overview and camera insets.
- `observations/<label>/*.mp4`: raw camera frames consumed by the expert.
- `observations/<label>/observations.jsonl`: matching frame indices, camera/state
  timestamps, motion clock, joint/base observations, guidance source/visibility, issued joint/root
  references, body feedback, capture errors, closure and phase. For assisted poses,
  the legacy `vision` container has `source=simulation` and `reprojection_px=null`.
- `observations/<label>/commands.jsonl`: every successfully sent planner/POSE
  reference, including handoff priming and commands between camera frames.
  `command_id`, send wall time, the decision's state sequence/simulation time,
  image index/timestamp and phase join it to observations and the physics trace.
  Payloads retain actual protocol ordering: planner upper-body order or POSE
  Isaac joint order, buffered frame indices, roots, velocities and both grippers.
  The decision state can precede the send (especially during priming). A send log
  does not certify deploy consumption. POSE packets include future predictions;
  their latest 50 frames must not be mistaken for already executed training actions.
- Successful results include an inclusive `training_tail.end_seq`, with image,
  command and physical-time boundaries at entry to the upright loaded hold.
  That boundary is published only after the full hold succeeds. Future training
  conversion should retain the prefix through this boundary and omit the final
  verification hold; raw video/control/physics files remain complete. Failed
  attempts have no successful training boundary. This uses the shared
  [sequence-boundary meaning](../src/locomani/pipeline/verification_tail.py), but
  these pilot manifests are not yet inputs to the formal dataset converter.
- `attempts.jsonl`, `manifest_00.jsonl`, per-attempt `result.json`: every sampled
  attempt and its actual success/failure, rise/tilt/contact, source kind and exceptions.
- `config_snapshot.json`, `deploy_receipts.jsonl`, stage logs and `trace/`: source,
  deployment and simulation evidence. Artifacts stay untracked.

Camera MP4s use nominal 25 FPS; use JSON timestamps for measurements. The overview
uses simulator timing. A good-looking video does not establish a causal explanation.
The current task covers a frontal, obstacle-free approach to one known basket/table
geometry. General navigation and transport remain outside this pilot.

Formal `--record` is refused. Token/history synchronization, emitted 64-dimensional
motion tokens, both-gripper VLA inference and end-to-end replay/learning still need
validation before these attempts can be called ready-to-train LeRobot episodes.
