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

The delivered batch is **`collection_10_v2`: 7/10 successes, 3 hold timeouts**.
No attempts were excluded. All ten reached the hold phase; the three failures
never met its observed stability conditions for an uninterrupted two seconds.
No external interventions or runtime errors occurred in this batch.
[Play all ten videos](../../outputs/crate_visual_expert_20260912/collection_10_v2/index.html).

For quick review, this index now serves compressed full-duration previews:
**238–492 KB each**, 640×480, 10 FPS, H.264. The
[ten-video ZIP](../../outputs/crate_visual_expert_20260912/collection_10_v2/download_preview/videos_10_small.zip)
is **3.40 MB**, reduced from the original 71.7 MB package. All files pass the
[mandatory video export procedure](../rendering.md#video-export): byte limits,
full decoding, start/end inspection and duration within one output frame of the source.
`download_preview/export.py`, `export_manifest.json` and `validation.json` preserve
the commands, hashes and checks. Original recordings, the original index and the
large package remain retained; the previews do not replace scientific evidence.

GPU-tested control revision: `3ac447a616914a8bf4f20471785a95a6162000cd`.
SONIC policy: `policy/sonic_v1_1`, encoder enabled, GPU 0 / instance 1.
Fixed marked scene: `scene_marked_v3`, metadata marker version 2; placement jitter
±2 cm in each horizontal axis and yaw 90° ±5°, identical seed list to v1.
The actual encoder, decoder, observation-config and deploy-binary SHA256 values
are retained in `deploy_receipts.jsonl`; source/config/scene identities are in
`config_snapshot.json`. The final exception-reporting correction was CPU-tested
separately; it does not change normal commands and was not exercised by this GPU batch.

```bash
LOCOMANI_ROOT=/workspace/code/locomanipulation-crate-visual-20260912 PYTHONPATH=src \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
  -m locomani.pipeline.run_crate_pilot --expert visual --rounds 10 --seed 20260912 \
  --gpu 0 --instance 1 \
  --scene outputs/crate_visual_expert_20260912/scene_marked_v3 \
  --out outputs/crate_visual_expert_20260912/collection_10_v2
```

The output directory already exists; choose a fresh directory when repeating.
The exact executed commands and process exit codes are in `commands.jsonl`.

| Episode suffix | Result | Truth rise (cm) | Final truth tilt (°) | Measured waist peak (°) |
|---|---|---:|---:|---:|
| 00 | completed | 9.92 | 7.65 | 7.33 |
| 01 | completed | 9.64 | 6.68 | 10.39 |
| 02 | completed | 7.60 | 9.38 | 22.42 |
| 03 | timeout | 8.36 | 6.62 | 11.49 |
| 04 | completed | 10.75 | 7.14 | 7.68 |
| 05 | completed | 9.90 | 12.36 | 10.10 |
| 06 | completed | 8.69 | 12.50 | 22.89 |
| 07 | timeout | 7.14 | 4.24 | 8.15 |
| 08 | completed | 9.69 | 3.47 | 6.97 |
| 09 | timeout | 9.55 | 12.37 | 18.26 |

Success combines the policy's uninterrupted observed hold with independent final
truth checks: rise ≥60% of the 10 cm reference, tilt <15°, bilateral contact and
no fall. The table reports object-centre rise from the pre-action state, not a
minimum-distance measurement between every basket point and the table.

Failed episodes 03/07/09 had longest observed stable intervals of 1.195/1.395/0.511 s.
The replay attributes failed checks to the visual attachment residual (03/07)
and hand tracking plus attachment residual (09). These are measured guard failures;
physical slip, estimator error and asynchronous sensor mismatch are not distinguished.
All three still had bilateral truth contact at the final snapshot. Do not relabel
them successful, or treat these diagnostics as proof of physical slip.

`audit.json` records full ffprobe frame counts, durations and overview hashes.
All 10 overview videos and 30 consumed-camera videos decode; all camera frame
counts match the 6,628 JSON records. There are 6,617 valid visual fits; missing
fits remain explicit. Maximum measured waist pitch was 22.89°, within the tested
reference regime. This is delivery evidence, not a locked-waist workspace comparison.

`policy_replay.json` reconstructs all ten policy trajectories from recorded visual
fits, encoder FK, gripper openings and timestamps: maximum reference-position and
closure discrepancies are both zero, with zero phase/reason mismatches. This check
covers the decision state/history contract. It does **not** replay lossy RGB pose
estimation, SONIC joint transport/tokens or the VLA input pipeline. Recompute with
the preserved `audit_collection.py` and `replay_policy.py` in the parent artifact directory.

## Checks and boundaries

Full `make check` and commit hooks pass. Independent Standards and Spec reviews
found configuration validation, result typing, evidence linking, loaded-motion
coordination, continuous hold, occlusion closure and attempt-error logging issues;
all material findings were corrected and rechecked. See [review record](crate-visual-review-20260912.md).

The v1 and v2 success counts are separate populations. Policy changes and asynchronous
trajectories prevent treating their difference as a controlled causal effect.
Development failures and both complete batches remain preserved.

This delivers an expert and ten reviewable video episodes for a fixed, visibly
marked basket. It does not certify unmarked perception, broad reachability,
collision-free planning, active recovery/leveling, locomotion, VLA history/action
synchronization, or student learnability. Those parts of the [design](crate-visual-expert-design.md)
remain future work before formal training collection. No weights or datasets were deleted.
