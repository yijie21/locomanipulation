# G1 floor-only dynamic paired-target reach — 2026-09-20

**Later correction:** this batch has a confirmed malformed reset causing violent initialization. The 6/14 result covers only subsequent reach, not whole-trial stationarity. See [startup diagnosis](g1-startup-jump-diagnosis-20260920.md).

Branch: `main`, dirty working tree retained. No commit or push. User requested a physical attempt for every V2 ghost target, with the original bimanual expert, no walking, a measured success criterion, and exported video.

## Implemented behavior

Entry: `scripts/probe_ghost_reach.py REQUEST --output RESULT`. Managed Stack owns the simulator/deploy lifecycle. Owners are `src/locomani/tasks/crate/ghost_reach_execution.py` and `ghost_reach_execution_cli.py`; `CoordinatedTarget.reach_only` defaults off, preserving ordinary crate behavior.

The scene contains only the floor and G1. Each of 14 trials resets to the original bimanual ready posture, releases initialization arm pins, and requires measured quiet standing. Target capture centres are transformed once from the settled robot xy/yaw into world coordinates; height remains floor-relative. They cannot slide along the rim or follow the robot. Ghosts exert no forces. Original WholeBodyIK admission, whole-body interpolation and SONIC POSE tracking remain in use. The hands stay open; there is no close/lift or locomotion command. Bending, knees and pelvis adjustments are allowed.

This first run uses the original 60-degree palm pitch, equivalent to **30 degrees toward the body from downward**. It does not search the full allowed 0–45-degree wrist inclination range.

Success requires both measured capture-centre errors ≤20 mm and both orientation errors ≤15 degrees for ≥0.5 simulation seconds. A trial ends after 12 simulation seconds without success. Either foot moving horizontally >30 mm or changing height >30 mm fails stationarity; this is not a pelvis-motion restriction or a floor-penetration-growth gate. Initialization failures, IK rejection, non-support collision, stationarity failure and timeout have separate reasons.

## Observed result

The completed batch is `outputs/g1_ghost_reach_20260920/`:

- **6/14 success**, cases **3, 9, 10, 12, 13, 14**.
- **8/14 timeout**, cases **1, 2, 4, 5, 6, 7, 8, 11**.
- All 14 passed initial IK admission and executed reach. All recorded walk-command counts are zero. No stationarity, collision, fall or initialization failure was reported.
- Maximum measured horizontal foot displacement across the batch: **14.123 mm**; maximum foot height change: **0.455 mm** (rounded upward).
- Native video duration: **187.44 seconds** across 14 complete trials. Each includes initialization and post-decision controller stopping/holding. A successful trial's hand error may increase during that explicitly labelled finishing interval; acceptance is the earlier completed 0.5-second window.

The native `seq/qpos/qvel` traces were independently replayed at 50 Hz. Every runtime classification agrees with the longest qualifying window recomputed from native FK (`all_windows_audit.json`), and all six reported success windows pass the denser replay (`dense_hold_audit.json`). The native qpos joints exactly match the recorded controller observations. Root-derived tool errors have small online-versus-replay differences (maximum **1.825 mm / 0.213 degrees**, rounded upward); these do not change any result. This comes from an existing publisher timing mismatch: `mj_step` integrates qpos while cached pelvis `xpos/xquat` remain from the preceding 5 ms physics step; online tool FK combines newer joints with that older root pose. The native-qpos 50 Hz audits above independently verify all 14 classifications, so the mismatch does not change this batch's outcome. Native raw RGB also uses cached transforms; frame numbering aligns to `3+4*k`, but the offline native-qpos geometry is 5 ms later. Do not claim pixel-identical replay of raw RGB.

These are **task/controller outcomes at one wrist inclination**, not a geometric impossibility proof or an estimated probability of success. A timeout means the dynamic execution did not sustain the thresholds within 12 seconds. Fourteen samples are insufficient to identify a general reach boundary.

## Artifacts and verification

- Request/source identities: `request.json`, `execution_code_identity.json`.
- Full results and controller diagnostics: `result.json`.
- Original recordings and native states: `videos/`, `videos/trace/`.
- Replayed robot plus corrected V2 ghost geometry: `rendered_sources/`.
- Delivery previews and continuous-play page: `previews/index.html`; verification: `previews/verification.json`.
- Download package: `g1_reach_all14_previews.zip` (**5,940,671 bytes**), containing only compressed previews and their index/verification. All 14 previews are **417,963–433,180 bytes** each; full decoding, duration preservation, start/end screenshots and ZIP integrity passed.
- Independent audits: `result_audit.json`, `dense_hold_audit.json`, `all_windows_audit.json`.

The first run was stopped because its launcher omitted `SIM_STATE_TRACE_DIR`; its original videos are preserved in `aborted_missing_state_trace/`. The replacement batch enabled native traces. No targets, controller behavior or thresholds were tuned using that interrupted run. Core/CLI hashes also changed due to Ruff formatting/import sorting; the substantive launcher change only enabled recording.

CPU validation: 27 controller/execution tests passed before launch; seven ghost tests passed after the recording fix. Final `make check` passed (one skip; existing warnings), log `make_check.log`. Physical/replay model joint order was checked exactly (34 joints, 40 qpos). For every trial, native sequence numbers are consecutive and native video frame count is `floor(trace_rows/4)`; replay uses rows `3+4*k`, at 12.5 FPS.

Next action: inspect successes and near misses, then decide whether to expand the target grid, repeat samples for variability, or search allowed wrist inclinations. The publisher's 5 ms mixed-state observation is a known follow-up issue for precise future boundary measurements.
