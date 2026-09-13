# Camera-guided crate video pilots

Use `--expert visual` for demonstrations that approach a visible basket and grasp
its rim. The low-table profile uses simulation-assisted planning with the original
basket appearance. It records the walk from 1.2–1.5 m away through the attempt.
These are video/control pilots; they do not yet certify VLA training or learning.
Implementation and measured limits: [assisted coordination](handoff/crate-assisted-coordination-20260913.md).

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
Missing/stale observations pause the reference and eventually abort. The same
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

- Walk from the sampled 1.2–1.5 m table-edge distance. A quiet target/base interval
  authorizes a support check before handoff. Unsupported reaches use the remaining
  braking allowance for a closer visible approach, bounded by `visual.stand_off`;
  unsupported placements can still fail admission.
- Preserve the consumed planner joint/root reference at POSE takeover. The backend
  switches directly into a primed POSE stream and releases the planner override
  atomically. Descent begins from that reference; there is no intermediate VR stage.
- Move both hands, hips and all three waist axes together. Compensate bounded
  measured tracking error with the whole body; do not freeze the legs after bending.
  Joint/root rates, acceleration, planted-foot error, support COM, pelvis height,
  waist limits and shallow knee flexion constrain the solution.
- Treat each no-post rim as a segment, allowing different along-rim hand positions.
  Keep the pair midpoint within 15 mm of the basket center. End-wall clearance,
  jaw alignment and vertical overlap still constrain closure.
- Maintain the paired reference during closure and lift with actual attachment,
  basket rise and tilt checks. The lift baseline must not apply tracking feedback
  twice. Stop on empty grippers, clear attachment loss, displaced baskets or stalled
  reaching; retain failed and timed-out attempts.
- After the attached lift, move the paired grasp toward the upright ready-pose
  centre while recovering hips and torso. Success requires actual torso lean within
  `visual.upright_tolerance_deg`, preserved attachment and a quiet load/body throughout
  `visual.hold_seconds`. The load must stay within `visual.hold_position_tolerance`
  of its mean position over that complete window, allowing small stance sway while
  rejecting sustained drift. Reaching a commanded upright pose alone is insufficient.
  Failed attachment, failed standing and unstable holding have separate terminal reasons.

`VisualLowReach` retains the older marker-based policy for historical experiments.
It reconstructs base translation from stationary foot FK; that information contract
must not be confused with the assisted controller's measured simulation base pose.

## Outputs and limits

Before sharing or packaging videos, follow mandatory
[video export](rendering.md#video-export): every preview is at most 500,000 bytes,
with complete duration/speed and verified playback. Preserve original recordings.

- `videos_00/*.mp4`: raw simulator overview and camera insets.
- `observations/<label>/*.mp4`: raw camera frames consumed by the expert.
- `observations/<label>/observations.jsonl`: matching frame indices, camera/state
  timestamps, joint/base observations, guidance source/visibility, issued joint/root
  references, body feedback, capture errors, closure and phase. For assisted poses,
  the legacy `vision` container has `source=simulation` and `reprojection_px=null`.
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
