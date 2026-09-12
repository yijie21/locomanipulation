# Crate RGB expert: implementation and video evidence

Date: 2026-09-12. Current usage: [collection guide](../crate_visual_collection.md).
This is a video pilot, with no claim of a validated VLA training dataset.

## Preserved development evidence

Artifacts are under `outputs/crate_visual_expert_20260912/` (untracked and preserved).
`pilot_04/manifest_00.jsonl` records the first complete grasp: 1/1, object rise
9.10 cm, final world tilt 12.33°, bilateral truth contact. Its command and measured
waist pitch maxima were 22.92° and 22.30°. Code file hashes, scene hashes and actual
SONIC v1.1 deployment hashes are in that run's `config_snapshot.json` and
`deploy_receipts.jsonl`; it predates the first implementation commit.
[First successful video](../../outputs/crate_visual_expert_20260912/pilot_04/videos_00/visual_crate_s101_00.mp4).

The fixed `2f89d1b` batch `collection_10_v1` ran seeds 20260912–20260921: 9/10
successes, one alignment timeout. All ten videos are retained in its
[viewing index](../../outputs/crate_visual_expert_20260912/collection_10_v1/index.html).
This preliminary batch predates corrections to loaded-motion coordination,
continuous stable hold, closure resumption after occlusion and attempt journaling.
It does not validate those later corrections.

`scene_physics_comparison.json` compares the compiled source and marked scene:
body mass/inertia/inertial frames and all original collision geometry parameters
are identical. Markers are visible patches on the existing inner far wall.
`pilot_00` preserves an initial gripper-shape exception; `pilot_01`–`pilot_03`
preserve failed colour/occlusion trials. They were development attempts with
different implementations/scenes, not part of the fixed batch denominator.

## Final validation

The corrected implementation is undergoing a separate ten-episode collection.
Final results and review resolution will be recorded here after it completes.
