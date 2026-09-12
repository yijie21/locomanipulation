# Crate RGB video pilots

Use this guide for the painted-crate expert. Its intended design and remaining
training work are in [the design](handoff/crate-visual-expert-design.md).
The expert now records walking from the table edge through stable grasping.
Current validation: [walking and stability evidence](handoff/crate-walk-stability-20260912.md).
The earlier [grasp-only batch](handoff/crate-visual-pilot-20260912.md) remains preserved.
No VLA learning result is implied.

## Information used by the expert

The pose estimator consumes three RGB streams, measured robot joints and fixed
camera/object geometry. A visible four-colour label on the basket's inner far wall
stays clear of the grippers. Its scene builder changes appearance only: zero mass,
no new contacts. The ordinary unmarked basket is a separate perception task.

At long range, only the two top markers may be visible. Their known spacing gives
a coarse head-camera range/bearing for a frontal approach to the fixed upright basket.
This estimate cannot authorize grasping; full RGB pose recovery is required near the table.

Task decisions consume RGB estimates, encoder FK, finger openings, base orientation
and their history. Base orientation is the simulated IMU signal and is recorded. A simultaneous solver controls two arms and shared waist pitch. Scene
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
