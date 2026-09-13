# Crate RGB video pilots

Use this guide for the painted-crate expert. Its intended design and remaining
training work are in [the design](handoff/crate-visual-expert-design.md).
The expert now records walking from the table edge through stable grasping.
Low-table validation and the latest changes: [visual low reach](handoff/crate-visual-low-reach-20260913.md).
The earlier [walking and stability evidence](handoff/crate-walk-stability-20260912.md) is preserved.
The earlier [grasp-only batch](handoff/crate-visual-pilot-20260912.md) remains preserved.
No VLA learning result is implied.

## Information used by the expert

The pose estimator consumes three RGB streams, measured robot joints and fixed
camera/object geometry. Visible labels on the inner far wall and basket floor
provide complementary views during walking and deep bending. Its scene builder changes appearance only: zero mass,
no new contacts. The ordinary unmarked basket is a separate perception task.

At long range, only the two top markers may be visible. Their known spacing gives
a coarse head-camera range/bearing for a frontal approach to the fixed upright basket.
This estimate cannot authorize grasping; full RGB pose recovery is required near the table.

Task decisions consume RGB estimates, encoder FK, finger openings, base orientation
and their history. Base orientation is the simulated IMU signal and is recorded. A simultaneous solver controls two arms and the configured waist axes. Scene
placement occurs before recording: the pelvis starts 1.2–1.5 m from the near table
edge. The whole walk, brake, observed settle and grasp are recorded. No robot
placement/reset occurs after walking begins. Object pose/contact truth is used by the
after-episode evaluator; falling can trigger an explicitly marked external abort.

Joint references limit velocity and acceleration; palm rotations blend continuously.
Reaching starts only after a continuous quiet interval in visual target motion and
base angular motion. Loaded references wait for bilateral tracking and attachment. Completion requires
an uninterrupted observed hold interval; occlusion resets that interval and pauses
gripper closure progress. The policy stops advancing when visual estimates disappear. It never refreshes a
hidden target from simulator object state. Camera streams, source timestamps,
robot state and issued references are saved for inspection.

## Run

Follow [AGENTS](../AGENTS.md) for GPU/instance admission. The following example uses
instance 1 and GPU 0; check live availability first. The configured simulation
Python is `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python`.

```bash
# Start from the crate scene built by tasks/crate/bimanual_scene.py (see crate guide).
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.tasks.crate.visual_scene outputs/scene/crate_bimanual outputs/scene/crate_visual

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.pipeline.run_crate_pilot --expert visual \
  --scene outputs/scene/crate_visual --out outputs/crate_visual_pilot \
  --instance 1 --gpu 0 --rounds 10 --seed 101
```

Configuration uses `--config` and `--set visual.FIELD=VALUE`. Fields and validation
belong to `CrateTaskConfig.visual`; see the generated [field reference](reference/config_fields.md).
Use a fresh output directory for each distinct experiment. The runner owns startup,
provenance, instance cleanup and the attempt manifests.

## Outputs and limits

Before sharing or packaging these videos, follow the mandatory
[video export procedure](rendering.md#video-export). Recorded originals below are
audit inputs; viewing/download links must use the verified compressed previews.

- `videos_00/*.mp4`: simulator overview with the camera insets, one per completed video attempt.
- `observations/<label>/*.mp4`: the RGB frames actually consumed by the expert.
- `observations/<label>/observations.jsonl`: matching frame indices, timestamps,
  robot observations including base orientation, references, visual fits, gripper
  closures, walking direction/speed, approach-estimate source and phase transitions.
- `attempts.jsonl`: sampled labels/seeds journaled before reset or camera setup.
- `manifest_00.jsonl` and per-attempt `result.json`: success, failure reason,
  measured rise/tilt/contact and any exception/intervention. Failed attempts remain present.
- `config_snapshot.json`, `deploy_receipts.jsonl`, stage logs and `trace/`: source,
  deployment and simulation evidence. A video does not certify a causal explanation.

The RGB MP4s are frame-indexed at a nominal 25 FPS; use the JSON timestamps for
measurement rather than assuming fixed wall-clock spacing. The overview is recorded
by the simulator's existing video path. Runtime artifacts remain untracked.

This pilot does not produce a validated LeRobot training dataset (`--record` is
refused). Token-label/history synchronization, both-gripper VLA inference and
student learning validation remain separate work before formal data collection.
Walking currently covers a frontal, obstacle-free approach to one known basket/table
geometry. General navigation, transport and unmarked-scene perception are unvalidated.
The saved IMU/history/command signals still need end-to-end VLA consumer validation.

## Low tables and a free waist

For a 30 cm table, select `--expert visual --config configs/tasks/crate_visual_low_reach.json`.
The `carry.reach_control=whole_body` selector reuses the existing body/waist limits;
the RGB entry owns object-directed decisions. `--expert carry` remains a privileged
mechanical diagnostic and must not be relabeled as a visual training expert.
The [low-reach investigation](handoff/crate-visual-low-reach-20260913.md) records
the measured motion, unsuccessful attempts and remaining limits. Its fixed ten-attempt
batch succeeded 4/10; low-table collection reliability is still insufficient.

Build a fresh painted scene from the lowered, unpainted crate scene. The current
builder adds visible wall and floor labels: the wall supports the long approach,
and the floor remains visible when bending puts the far wall outside the images.
The patches have zero mass and no collision response. Their known locations are
object geometry, not per-episode pose truth. This is still a marked-basket task.

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.tasks.crate.visual_scene outputs/scene/crate_table30 outputs/scene/crate_visual30

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.pipeline.run_crate_pilot --expert visual \
  --config configs/tasks/crate_visual_low_reach.json \
  --scene outputs/scene/crate_visual30 --out outputs/crate_visual30_pilot \
  --gpu 0 --instance 1 --rounds 10 --seed 41
```

The height profile does not build the scene. Prepare the 30 cm source using the
[table-height procedure](crate_collection.md#table-height-validation), then paint it.
Use new scene/output directories so older evidence remains reproducible.

Low reaching uses the following contracts:

- Walk from the sampled 1.2–1.5 m table-edge distance using RGB range/bearing.
  Observed quiet motion authorizes reaching; the full-body solver still rejects
  unreachable stands. All three waist axes obey the physical model limits.
- Preserve the consumed planner joint/root reference and owned arm override at
  POSE takeover; measured wrist/neck references bridge the mode transition.
  Bend and reach with bounded joint speed/acceleration. Use elapsed observation
  time rather than assuming that every camera/control tick took 20 ms.
- Treat the no-post rim as a graspable segment. Individual along-rim offsets may
  differ, while the pair midpoint stays within 15 mm of the basket center. End-wall
  clearance, jaw alignment and vertical overlap still constrain closure; the wrists need not match a single preferred 6D pose.
- RGB fits, encoder FK, finger openings and IMU drive the controller. A planted-foot
  frame estimates stationary base motion from those observations. Missing or stale
  vision pauses every trajectory, including bending and closure, and eventually aborts.
  Simulator object position and contacts are consumed only by the final evaluator.
- Check actual delivered hand motion before deciding whether the basket follows
  the lift. A deadline still rejects an undelivered lift. Preserve all failures.

`observations.jsonl` also records the full 29-joint reference, its velocity, root
orientation reference, reconstructed support origin, capture-region errors and
visual lift/attachment checks. Image indices and source timestamps remain authoritative.
The bounded backend mode handshake appears in the overview/trace but is not a
complete token-label recording. Formal `--record` remains blocked: the current
SONIC crate training configuration consumes single-frame observations, while this
expert uses history. Validate history inputs, emitted 64-dimensional motion tokens,
both grippers and synchronized replay before calling these VLA training episodes.
