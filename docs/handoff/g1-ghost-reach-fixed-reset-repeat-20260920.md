# G1 ghost reach: user-authorized repeat after reset-vector fix

Branch `main`; prior dirty work preserved. This run repeats the same 14 V2 targets, seed42, physical scene, original expert, 30-degree wrist inclination and acceptance criteria. Production source was not modified during the repeat; the saved file hashes were checked after execution. No failed trial was retried.

Artifacts: `outputs/g1_ghost_reach_20260920_fixed_reset_repeat/`.

## Observed results

- **4 reach successes:** cases10,12,13,14.
- **9 reach timeouts:** cases2–9 and11.
- **1 initialization failure:** case1 fell before reaching a stable admitted stance; it never entered reach. Its complete9.12s recording is preserved.
- All14 first native frames have the correct reset values: waist0, both elbows0.8rad, and WBC default legs. The 17/29 vector mismatch is fixed.
- **Startup dynamics are still not reliable.** Five cases exceeded30mm horizontal foot displacement during initialization: **1,2,4,6,12**. Case1's maximum was0.923231m, with a46.201-degree joint change over20ms; minimum root height was0.083569m and the recorded final observation reported `fallen=true`.
- Case12 reached successfully after its initialization foot displacement of0.187882m. Its reach success does not establish whole-trial stationarity.
- No walking commands were issued. Balance-controller motion can still move the feet; absence of walk commands is not evidence of stationary startup.

Independent native50Hz FK audit agrees with all13 runtime classifications that entered reach: four valid windows lasting0.52–0.54s, nine failures to sustain0.5s. All13 reach-stage foot displacements stayed below the30mm limit, measured from the later ready baseline. The failed initialization has no target/IK/reach trajectory to audit and is not fabricated as a reach attempt.

## Video and audit delivery

Native recording duration across14 clips: **196.96s**, normal speed and full duration, including initialization and finishing intervals. Ghosts appear only after reach begins. HUD displays foot displacement both from reset and from ready, and labels results as **reach-stage** outcomes. Case1 explicitly shows the fall and lack of a reach attempt.

- `request.json`, `dry_run.json`, `code_identity.json`: exact inputs and source identity.
- `result.json`: original runtime output and per-tick diagnostics.
- `videos/` and `videos/trace/`: preserved source recordings and seq/qpos/qvel.
- `fixed_reset_audit.json`, `audit_fixed_reset.py`: reset posture, startup movement, native FK windows and frame-count evidence.
- `rendered_sources/`: actual-qpos replay plus visual-only targets and labels.
- `previews/index.html`: all14 compressed clips and continuous playback.
- `previews/verification.json`: measured sizes, durations, decode checks and metadata.
- `g1_reach_all14_previews.zip`: only compressed previews, index and verification.

All14 delivered previews are **403,895–437,016 bytes** each; ZIP is **5,971,594 bytes**. Full decoding, source/preview duration preservation and ZIP integrity passed; decoded first/last images were inspected for the initialization fall and a successful case. Documentation checks passed.

Every native trace is consecutive; all14 raw video frame counts equal `floor(trace_rows/4)`. Replay uses `3+4*k` at12.5FPS. The previously documented5ms cached-root/online observation mismatch remains; classification was independently verified against native qpos. The0.65m audit height threshold denotes standing admission, not the simulator's fallen flag.

GPU1 / instance2 were released after the batch. No additional runtime tuning or retry was performed. This repeats the corrected reset request but does **not** demonstrate that the remaining reset/controller synchronization problem has been fixed. [Prior cause and correction](g1-startup-jump-diagnosis-20260920.md).

Follow-up: [comparison against the actual original expert startup](g1-startup-vs-original-expert-20260920.md) identifies missing preparation/interpolation, an added PD-pin transition, and different recording boundaries. No new physical trial was performed for that comparison.
