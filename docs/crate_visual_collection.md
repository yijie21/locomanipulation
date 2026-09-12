# Crate RGB video pilots

Use this guide for the painted-crate expert. Its intended design and remaining
training work are in [the design](handoff/crate-visual-expert-design.md).
Implementation is being validated with video pilots; no VLA learning result is implied.
Measured outcomes and revisions: [video evidence](handoff/crate-visual-pilot-20260912.md).

## Information used by the expert

The pose estimator consumes three RGB streams, measured robot joints and fixed
camera/object geometry. A visible four-colour label on the basket's inner far wall
stays clear of the grippers. Its scene builder changes appearance only: zero mass,
no new contacts. The ordinary unmarked basket is a separate perception task.

Task decisions consume RGB pose estimates, encoder FK, finger openings and their
history. A simultaneous solver controls two arms and shared waist pitch. Scene
placement occurs before recording. Object pose/contact truth is used by the
after-episode evaluator; falling can trigger an explicitly marked external abort.

Loaded references wait for bilateral tracking and attachment. Completion requires
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

- `videos_00/*.mp4`: simulator overview with the camera insets, one per completed video attempt.
- `observations/<label>/*.mp4`: the RGB frames actually consumed by the expert.
- `observations/<label>/observations.jsonl`: matching frame indices, timestamps,
  robot observations, references, visual fits, gripper closures and phase transitions.
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
Current controls use a fixed ready stand; general walking and transport are outside
this first experiment. No claim of unmarked-scene generalization is made.
