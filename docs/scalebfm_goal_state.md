# ScaleBFM visual VLA Goal — current checkpoint

Updated 2026-09-09 07:07 UTC. USER CANCELLED; work stopped and comparison incomplete.
Read docs/STATE.md first, then this file. Recover from evidence; do not repeat experiments.
Experiment root: outputs/experiments/scalebfm_visual_goal_20260908/ (paths below relative to root).
Prior detailed checkpoint archived in docs/scalebfm_goal_history_20260909_c5_044144.md; it contains historical statuses, not current work.

## User cancellation — STOP

The user explicitly said “先暂停，取消之前的 goal” on2026-09-09.
All Goal work is stopped as of2026-09-09 07:07UTC. Do not resume for automatic
Goal continuations; require a new explicit human request. No objective-completion claim.
Platform Goal was initially still active. After three consecutive automatic continuations
without renewed human execution authorization, it was marked blocked for that authorization
impediment. No experiments resumed, no completion/cancelled-platform claim. See
goal_authorization_block_20260909.json. Available tools do not provide platform cancellation.

- Driver1470213/session89332 and watcher2145035/session78100 terminated viaSIGTERM,
  both exit143. A29trainer2378682 and all Goal GPU processes are absent. Other sessions untouched.
- A17/B17/C17 each completed3000steps and60selectioncases: successes13/17/11.
  C17 selection finished07:03:17UTC, auditPASS; high/low-near/low-deep0/8/3 of20 each,
  normal10/48, recovery1/12 (wrist_offset1/3, others0), postureviolations0.
  C17 failures timeout24, preclosemotion11, heighttermination14 (9lowdeep+5lownear),
  no nonfinite qpos or MuJoCo warnings. C extension proxy8/10 successfulnormal overlap,
  but no world-pad rise>=1cm at its detected onsets; naturalness not accepted.
- A29 stopped after40logged optimizer updates; resumable checkpoint directories:[].
  Total observed updates9040/18000. No A29model or full-comparison completion claim.
  Do not restart from step0 automatically; any authorized restart must account for these updates.
-18/36checkpoint audits complete; latest C17final3000PASS. Watcher markedcancelled_by_user.
- C17 diagnostics and interim paired results saved. CvsB: both5,C-only6,B-only12,p=.2379;
  CvsA: both1,C-only10,A-only12,p=.8318. Single-seed exploratory results only; no winner lock.
  seed17_interim_comparison.json. A29actualprocessor checkPASS before cancellation.
- C fixed-three rendering finished exit0, image exists; visual review NOT performed before stop.
  Preserve vla_selection_visuals_s17/C_first_three.png/C_manifest.json/C_render.log.
- Durable cancellation:user_cancel_checkpoint_20260909.json and user_cancel_request_20260909.json.
  pipeline_v2_state_before_user_cancel.json archives the live ledger; current ledger and watcher
  explicitly mark user cancellation. All data, models, logs, diagnostics and videos preserved.

## Resource and observation constraints

- GPU0 was explicitly released by9d at18:50 Sep8 and claimed for this Goal18:56.
  GPU1 belongs to9d seed7 work started03:50UTC; instance0/port5590 reserved until
  about07:30UTC per latest STATE.md. Leave alone. No stack/ports; NEVER stop_stack.sh.
- At06:15UTC an additional SONIC deployment PID2296604 was observed on GPU0 (~1.3GiB),
  alongside C17 trainer (~26GiB). gpu0_shared_load_20260909_0615.json records the observation.
  No external process/stack changed; throughput causality unproven. GPU0 reservation
  is not evidence of exclusive actual use. Preserve load context for wall-clock timings.
- Formal jobs use third_party/Isaac-GR00T/.venv/bin/python (torch2.9cu128,MuJoCo3.11).
  GPU env: CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0.
  CPU work hides CUDA; render with MUJOCO_GL=osmesa, LP_NUM_THREADS=1,
  OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. No shared-library/env changes.
- VLA sees two RGB cameras, own proprioception/history/previous action, fixed task text.
  No object truth, contact flags, expert phase or success inputs. Native action29,
  proprio/history state91. Expert RGB-localization proxy remains a documented limitation.
- Preserve other sessions' dirty files/data; no commits. docs/STATE.md has shared history.

## Completed A17/B17 evidence — do not retrain or re-evaluate

| Run | Training | Selection | Registered posture violations | Task plus all posture gates |
|---|---|---|---|---|
| A17 legacy/common executor |3000,exit0|13/60|49/60|0/60|
| B17 a16 |3000,exit0|17/60|0/60|17/60|

- A17 ended01:24:14UTC, runtime16832.36s, final loss.0157. B17 ended04:03:52UTC,
  runtime8923.33s, final loss.0189. model_A_s17_completion.json and model_B_s17_completion.json.
- Each500-step checkpoint audited; only checkpoint-3000 and final root model retained.
  checkpoint_A17_3000_audit.json / checkpoint_B17_3000_audit.json PASS. Old audit JSONs retained.
- A17 selection ended01:34:42UTC; high/low-near/low-deep8/3/2 of20 each.
  Normal11/48; recovery2/12 (blackout2/3, other kinds0). Failures: pre-close motion24,
  height termination17, timeout6. No-close27, stationary_without_close14.
- B17 selection ended04:13:01UTC; high/low-near/low-deep0/10/7 of20 each.
  Normal14/48; recovery3/12 (one each stop_short/wrist_offset/object_shift, blackout0).
  Failures: timeout24, pre-close motion15, height termination4 (all low-deep).
  No-close15, stationary_without_close7. Visibility mean.996535; median of case medians
  66.36ms, median of case p95s72.52ms (not pooled p95).
- Both real-model validation diagnostics60/60 finite. selection_A_s17_audit.json and
  selection_B_s17_audit.json check all registered scenes, per-case/summary equality, final model.
  Reusable audit_selection_run.py --arm C --training-seed17 for future completed runs;
  refuses to overwrite completed audits. No additional inference/physics.
- Paired B-only17/A-only13/both0, exact McNemar p.585 (single training seed only).
  paired_B_vs_A_s17_selection.json. B has better measured torso/waist posture and low-table
  results, but high-table success drops to0/20; no superiority/winner claim yet.
- A/B failure-state audits inspect all60 saved qpos and warnings: no nonfinite qpos or
  MuJoCo warnings. Their17/4 combined physics warning/fall labels match pelvisz<.45m.
  Do not call these numerical instability or visually proven falls. Height termination
  can occur with crouching while all registered torso/waist posture gates pass.
  selection_{A,B}_s17_failure_states.json; audit_selection_failure_states.py --arm C later.
- Pre-close knock means horizontal bottle motion>5mm before grip decreases>2mm;
  it does not require toppling. Use 提前碰动, not confirmed 碰倒.
- Supplementary existing evaluator arm-start diagnostic COMPLETE for all60 A17/B17 traces:
  successful-normal before/at-close moving overlap A11/11, B12/14; all-normal47/48,45/48.
  The pelvis-relative up threshold triggers59/60 A and58/60 B onsets, while world-pad
  rise>=1cm from1.68s at the detected onset occurs only3/60 in each arm. Pelvis lowering
  can trigger this proxy; it does NOT prove natural hand raising or deny later raising.
  selection_{A,B}_s17_extension_diagnostic.json; student_extension_interpretation_s17.json.
  Four specified cases cross-checked against Sim.pad/mj_forward with zero displacement error.
  Diagnostic only, not the expert phase-relative coordination gate or a selection change.
  Student natural-coordination acceptance remains open. C/seed29: audit_student_extension.py
  --arm C --training-seed17 after corresponding complete selection audit; no repeat A/B.

## User-requested videos and visual checks COMPLETE

- Expert a16 accepted train seed4001 high_near,10.12s:
  outputs/videos/scalebfm_a16_delivery_20260909/a16_expert_seed4001_third_person.mp4
- Corresponding trained VLA B17 final3000, held-out seed21001 high_near,22.4s:
  outputs/videos/scalebfm_a16_delivery_20260909/vla_B17_seed21001_third_person.mp4
  Episode outcome TIMEOUT24s; clip omits1.6s startup,560frames/25fps/640x552.
  ffprobePASS, encoded frames0/280/559 inspected, labels readable. Both links delivered.
  Different expert/student scene seeds; not a matched reset or a successful student clip.
  a16_video_delivery.json and delivery-directory delivery.json/inference_visual_review.json.
  Renderer render_vla_video.py, CPU session68508 exited0; no rerender needed.
- posture_ab_v1: paired expert A/B images high/low/deep and three B expert clips
  seeds4001/4002/4003 COMPLETE, visually checked, ffprobePASS. Expert imagery only.
- vla_selection_visuals_s17: A_first_three.png and B_first_three.png COMPLETE and viewed;
  fixed scenes21001/21002/21003 at2/4/6s+final, same frozen cameras.json across arms.
  B first two timeout24s, deep crouches and terminates20.22s; all3 samples fail though
  B overall17/60 succeeds. A_manifest/B_manifest and A_visual_review/B_visual_review saved.
  render_vla_selection_samples.py --arm C only after C17 selection; no A/B rerender.
- A_height_termination_samples.png shows first such case per stratum21004/21005/21009;
  three samples visibly crouch, not prone at terminal frame. JSON/review saved;
  layout_v1/v2 preserved. Not a continuous visual review of all17 cases.
- All visuals use recorded qpos and mj_forward/render only, zero additional rollouts.

## C0–C4 COMPLETE

- Plans/gates/amendments: docs/scalebfm_goal_plan.md; design: docs/scalebfm_visual_expert_design.md.
- a16 measured-arrival fix keeps native forward motion while body remains>10cm short
  despite reference arrival; stops in corridor/blackout. dev12/12, qualification30/30
  task+audit, each stratum10/10; recovery11/12 (seed1710 object_shift pre-close motion failed).
- Qualification close-window absolute pitch medians: high2.98deg, low25.23deg, deep30.61deg.
  startup_contract30/30: first capture1.60s, first action1.68s. Labels/previous actions exact.
  Same-env delayed chunk replay10/10 exact qpos/actions; actual3cm wrist perturbation
  replay3/3 success over original recording duration only. c3_final_acceptance.json ready.
- Review repairs complete;35tests PASS, independent reviews resolved. No repeat reviews to wait.
  Frozen-base real batch4 forward/backward finite, zero optimizer updates; memory preflight passed.
  processor_batch_contract/result.json passed. No task rollout from preflight.

| Arm | Accepted train/val | Train/val frames | Accounted attempts |
|---|---|---|---|
|A legacy|200/20|56522/5693|225/880|
|B a16|200/20|50572/5102|367/880 incl145retired|
|C recovery replacement|200/20|47616/4758|225/880 incl165reused|

- All cohort_{A,B,C}_v2 exports/stats complete; dataset_{A,B,C}_{train,val} immutable.
  Old a15 cohort_B retired, never enters training; preserve retirement.json and145attempt cost.
  B new222 physical attempts; C new60 plus165reused B normal. C train150normal+50recovery,
  val15normal+5recovery. Recovery train12stop_short/15wrist_offset/12blackout/11object_shift.
  C validation recovery2stop_short/3wrist_offset only; blackout/object_shift val coverage absent.
  cohort_data_report.json; real dataset_B_contract.json6/6 passed. Do not recollect/re-export.

## Fixed protocol and remaining work

- A/B/C each train seeds17,29, exactly3000 optimizer updates, six runs/max18000.
  Micro4×acc16=64, LR1e-4, AdamW8bit, weight decay1e-5,warmup.05. Vision/LLM frozen,
  projector+diffusion tuned, state dropout.2, fullFOV240/crop1, allow_padding=True.
  Base snapshots GR00T2fc962b973bccdd5d8ce4f67cc63b264d6886495 and
  Cosmos9ce19a195e423419c349abfc86fd07178b230561; local_files_only=True.
- six_config_contract.json verified/frozen before first update. Save each500 including
  optimizer/RNG, retain latest1, final root model also saved. Only final3000 evaluated.
- Selection60/model21001..21060 =48normal+12recovery,3/kind. Four denoising steps,
  40 action rows, .4s replan/.08s delay, previous chunk while pending, max24s.
  Policy RNG900000+scene seed; first capture1.60s/first action1.68s.
- Winner only after all six full selections: mean task success across two train seeds;
  within1/60 tie, fewer registered posture violations then simpler A/B/C.
  Lock selection_locked.json BEFORE confirmation. Winner and A both seeds100scenes
  31001..31100 (88normal+12recovery), no duplicate A if A wins. Per-seed paired McNemar
  and scene-joint bootstrap, no treating same scene across two seeds as independent.
- Registered posture: roll peak>20deg or roll>15 duration>.2s, waist yaw>25, pitch>40;
  high-near close-window median abs pitch>15 also violates. Height stop remains task failure.
  Validation3fixed frames/20episodes MAE diagnostic only, never selects checkpoint/winner.

## Completed infrastructure repairs and limitations

- training_startup_failure_01: initial OmegaConf resume-bool failure, zero updates.
  Own launcher repaired bool resume/latest-path check; red→green regression; six configs
  refrozen before first actual update. Do not restore old freezes or repeat training.
- evaluation_startup_failure_01: A17 validation loader shared mutable modality dict;
  one validation prediction, zero closed-loop cases. Fixed independent deepcopy;
  real20episode/60frame regression red→green, label MAE parity, no labels in predictor input.
  A17/B17 real diagnostics and selection now complete. No scientific config change.
- normalization_tail_audit.json: q01/q99 clipping is lossy; wrist error>1cm frames
  A2.39%,B1.74%,C2.00%, maxima10.25/6.00/6.21cm. Actual saved A/B/C settings/stats checked.
  Model closed-loop impact is unproven; do not attribute failures causally without experiment.
- environment_manifest.json/environment_versions.txt145packages captured after A17step500,
  not a retrospective full environment lock. Checkpoint audits do not test resumed updates.
- C is synthetic recovery replacement, not actual student-state DAgger. Expert localization
  is an RGB-visible proxy, not a trained perception network. Preserve domain/coverage limits.
- docs/scalebfm_visual_collection.md operator guide; docs/scalebfm_acceptance_evidence.md
  requirement matrix; docs/scalebfm_vla_comparison_results.md results and visuals.
  Older development: docs/scalebfm_goal_history_20260908.md and
  docs/scalebfm_goal_history_20260908_c4.md. Goal is NOT complete: remaining training,
  selections, lock, confirmation and final requirement-by-requirement audit are required.
