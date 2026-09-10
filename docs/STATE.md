# STATE (updated 2026-09-09 17:30 UTC by 9d -- READ THE INCIDENT FIRST)

## INCIDENT 2026-09-09 17:23 UTC: outputs/ was deleted by a git merge (session 9d, Claude)
- Cause: refactor commit cf6a08e (`git add -A` in a worktree that had a symlink named `outputs`) tracked that symlink;
  `.gitignore` had `/outputs/` (directory-only pattern) so the symlink was not ignored. Merging refactor/step1-tooling
  into main (3344e45) replaced the real `outputs/` directory with the symlink; git removed the ignored contents.
  /workspace usage fell 819 GB -> 563 GB. XFS, no block-device access: NOT recoverable by undelete.
- LOST: everything under outputs/ -- all experiments (datasets, models, evals, manifests, videos, gate dirs), scene
  builds, eval history, renders, and the Codex thread's scalebfm_visual_goal_20260908 / scalebfm_yplus200_s17_20260909.
- SURVIVES: git-tracked code/docs/tests (tests/gate/baseline_*.json included); raw exporter sessions in
  /workspace/tmp/p03/outputs (82 sessions: 39 on 09-08, 43 on 09-09, 3.9 GB) and /workspace/tmp/p03_i1/outputs;
  /workspace/huggingface_cache (143 GB, base models); the refactor worktrees.
- Fixed: commit 4a109bf untracks the symlink, recreates outputs/, ignores `/outputs` and `/third_party` as paths.
- Recovery paths: datasets can be re-converted from the exporter sessions WITHOUT manifests (ok/failed rounds are
  in each session's info.json discarded_episode_indices; the verification-tail trim marker is lost) or recollected;
  models retrain in ~77 min; evals rerun. Waiting for the user's decision before any further GPU work.

## Refactor merged and verified (2026-09-10 03:00 UTC, session 9d; main 1d35aad+)
- Steps 1-6 of docs/handoff/refactor-plan.md are on main. No functional change, evidence in tests/gate/results/:
  (1) converter A/B (78536ae vs main, 4 exporter sessions): 34/34 parquet+stats byte-identical; meta differs only in
  the source path string and the deleted a18 block. (2) bottle GPU gate on GPU0: 7/10, geometry / stages / durations
  all PASS; the single red row is a pre-existing failure string missing from the 30-round baseline. Bisect p2/p4/p6:
  6/10, 6/10, 8/9. An earlier 3/10 with four close-phase topples was the sim on GPU1 under 97 % foreign utilisation.
  (3) crate GPU gate: 8/10 = baseline; stand-off / yaw / depth PASS; one pre-existing failure string.
- Gate-found and fixed: exporter launcher lost its import path (never started); 'approach' stage marker dropped
  with the legacy tail; script-vs-package double module copy made `--set wall_walk=true` a no-op in the crate CLI.
- Not yet verified: the eval path (needs a trained model; all models were lost). Size-cap debt: 5 files / 79
  functions (scene_lib, privileged_pub, convert_to_hf_format, isaac_scene, isaac/stage), lint stays in report mode.
- Data recovery after the incident: waiting for the user's decision (re-convert from /workspace/tmp/p03 sessions
  without tail trim, or recollect). No GPU jobs running.

## Now
- ScaleBFM post-deletion audit2026-09-10: outputs is empty; ABCD process absent, NOT running.
  Core src/tests/config/docs survive;3ablation unit tests pass. ABCD run.py/config/progress/results,
  yplus200 checkpoint/data/evals, scene builds and runtime manifests are missing. Cannot resume.
  Last conversational2/120 is historical, not confirmed final deletion-time progress. No recovery GPU jobs launched.
  Details:docs/scalebfm_outputs_loss_20260910.md. ACTIVE statements below are pre-incident history.
- Pelvis ABCD protocol v2 ACTIVE: old EGL baseline not repeatable (identical own state,1-level RGB differences).
 OSMesa+deterministic math produced bit-identical full24s original/diagnosticA forseed31001.
 Fresh30 A +90 B/C/D =120 formal cases, same30 seeds/interventions; VLA+ScaleBFM inferenceGPU0,
 renderingCPU. Do not compare with oldEGL25/100 as paired baseline. PROTOCOL_REVISION.md documents
 change before B/C/D outcomes; run_v2.log and progress.json track execution. Old failure_v1.json is archived evidence.
 Root:outputs/experiments/scalebfm_pelvis_abcd_20260909/. No retraining or cancelledGoal restart.
- NEW user-authorized pelvis ABCD diagnostic on GPU0:30paired existing scenes (10/stratum),
 A original;B smooth76cm height hold;C restore after<72cm for0.1s;D right-wrist1s hold then0.5s release
 at same trigger.3baseline reproducibility probes then90 B/C/D cases. No training/oldGoal restart.
 Root:outputs/experiments/scalebfm_pelvis_abcd_20260909/. Read PLAN.md,plan.json,progress.json first.
 GPU0 checked idle16:07UTC;GPU1 other workload untouched. All interventions and outputs auditable.
- Failure analysis2026-09-09:docs/scalebfm_failure_analysis_20260909.md. Third-person export has
 missing mj_camlight: light runtime direction remainszero; same-frame luma42.28->65.95 after update.
 This is an export bug, not training RGB lighting change; current ZIP not rerendered yet.
 38VLA cases descend below0.60m;29fall labels are solelyrootz<0.45 with0physics warnings.
 Relative pelvis-z action is clamped bycheckpoint q99 to+1.836cm, restricting recovery from low poses.
 7cases have>=3gripper switches;55never command near-full closure. No strategy or formal experiment changes.
 Evidence:outputs/diagnostics/scalebfm_failure_analysis_20260909/. Read report before proposing fixes.
- User-requested300-video compact export COMPLETE:200 collection +100 evaluation (including75failures),
  right-front full-body follow, saved-qpos rendering only,480x400/12.5fps. ZIP53,748,107bytes;
  videos86-375KB, median163KB. All300 frame counts and ZIPintegrity checked;14decoded sample frames reviewed.
  Download:outputs/videos/scalebfm_yplus200_eval100_small/scalebfm_200_collection_100_evaluation.zip.
  Export script/manifest/verification saved alongside; do not repeat physics or inference for these videos.
- yplus200 experiment COMPLETE2026-09-09 12:39:49UTC:200 accepted new demos, seed17/3000steps,
 100 ordinary y+ closed-loop cases,25/100 success (high10/34,low-near10/33,deep5/33).
 Failures29timeout/17pre-close bottle knock/29physics warning or fall. All stages exited0 after
 documented startup recovery. Evidence:outputs/experiments/scalebfm_yplus200_s17_20260909/RESULTS.md.
 No further training/evaluation queued by this experiment; older running progress below is historical.
- yplus200 update: collection COMPLETE200/200 attempts accepted (67high/67low-near/66deep),
  export/statistics complete. Training startup offline-tokenizer failure repaired by removing driver-only
  HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE; no training configuration or expert change,0 updates lost.
  GPU0 training now verified at12/3000 steps; driver2458090,trainer2458112.100-case evaluation pending.
  Logs/recovery:outputs/experiments/scalebfm_yplus200_s17_20260909/{pipeline_v2_state.json,RECOVERY.md}.
- NEW user-authorized experiment2026-09-09: current expert200 accepted episodes -> VLA seed17/3000steps
  ->100 ordinary closed-loop cases on GPU0, yplus_center_v1. Existing information gaps recorded,
  intentionally not changed. GPU0 checked idle at08:30UTC; GPU1 belongs to another task.
  Root:outputs/experiments/scalebfm_yplus200_s17_20260909/. Read PLAN.md and pipeline_v2_state.json
  before doing anything; per-stage PID/logs survive context loss. This is separate from cancelled ABC Goal.
- Current ScaleBFM policy audit (2026-09-09, bounded user request): initial HEAD-camera visibility
  enforced at reset,1.60s,and actual first recorded frame; invalid episodes retained/rejected.
  Fixed eligible-camera selection, entry wrist/root heading mismatch, fixed-bottle-height lift test.
  49 visual tests passed; CPU three normal scenes3/3 task+audit, two injected hidden starts2/2 rejected.
  Remaining major limitation: proxy still reads exact object/table geometry; bottle visibility does
  not establish table-edge visibility. Full audit:docs/scalebfm_policy_audit_20260909.md.
  Evidence:outputs/diagnostics/scalebfm_policy_audit_20260909/. No Goal/training restart.
- User-requested bounded video inspection: trigger "检查采集视频" -> bash scripts/scalebfm_video_check.sh.
  CPU only,10 current-expert coverage cases, right-front full-body follow videos; no Goal/training restart.
  Contract:docs/scalebfm_video_check.md. Current export:outputs/videos/scalebfm_check10_yplus_center_v1_26cc7ae54f7a/.
  COMPLETE:10/10 videos,10/10 expert task+audit passes;40 decoded representative frames visually reviewed.
  No repeated physical cases, no Goal restart. This is a coverage demo, not random success-rate evaluation.
- ScaleBFM follow-up (2026-09-09, separately authorized): new collection defaults to yplus_center_v1,
  bottle x centred on table +/-8cm, near inboard12-18cm, deep18-24cm; robot approaches only y+.
  CPU three-case smoke3/3 and42visual tests passed. No batch collection/training or Goal restart.
  Details:docs/scalebfm_yplus_collection_20260909.md. Old frozen three-edge results remain historical.
- ScaleBFM visual-expert Goal (this Codex thread): USER CANCELLED at2026-09-09 07:07UTC.
  Driver1470213, A29trainer2378682 and watcher2145035 stopped; all Goal GPU processes absent.
  Goal releases its GPU0 reservation; an external SONIC deployment still uses GPU0.
  Other workloads/stacks untouched. Do not restart Goal without new explicit human authorization.
  A17/B17/C17 complete3000steps and60selectioncases each, successes13/17/11.
  A29cancelled after40logged updates, no saved500step checkpoint. No winner lock or confirmation.
  All artifacts preserved. Cancellation/recovery:docs/scalebfm_goal_state.md and
  outputs/experiments/scalebfm_visual_goal_20260908/user_cancel_checkpoint_20260909.json.
  Platform Goal now blocked: three automatic continuations lacked renewed human authorization.
  This is not objective completion; no experiments resumed. Platform cancellation remains a UI action.
- 9d: RUNNING since 09:25 UTC -- ready_visible200_20260909 (visible-expert collection, 200 ep, instance 0
  stack on GPU1; training/eval intended for GPU0 once the Codex training there ends). DO NOT delete
  outputs/experiments/ready_visible200_20260909 or ready_visible_smoke_* (a cleanup at ~09:00 removed the
  smoke dir while it was being written). Code: visible_expert flag in reach_config/ready_grasp_policy/
  grasp_gym/recoverable_grasp_policy, config configs/collection/ready_grasp_visible.json, tests/test_visible_expert.py.
  Smoke 8/10 (1 spawn-turn, 1 infeasible discard), 9/9 brakes at depth 0.30-0.31, no recovery stage.
  Previous: seed-7 pilot torn down 07:55 UTC; instance 0 / port 5590 free; the 20 GB on GPU1 is
  not ours -- check nvidia-smi before claiming it). Next item 1 DONE: ready07d_200_seed7_20260909 = 46/100
  (same 176 ep, GR00T_DATA_SEED=7) vs seed 42's 57/100, paired 21:32 p 0.17; vs decisive first 60: 27 vs 29,
  p 0.87. The +8 of 07d-200 over decisive is seed variance, not data; failure mix flips with seed alone
  (static 15 -> 39, knock 19 -> 7). docs/ready_grasp_arms_20260907.md §9. Model kept (24 GB).
  Earlier today: ensemble N3 M0.3 screen 3/20 vs 9/20, harmful (§8.1). Last result before that: 07d-200 57/100
- ScaleBFM: nothing running. Three pilots today (v3 18-d, v3 26-d offline velocity, v4 23-d native
  velocity) all 0/60; the failure moved from "never leaves the spawn" to "stops 10 cm short of the stand".
  The scene geometry was then found to be wrong (corner approach, 0.45-0.49 m reach) and replaced by the
  SONIC edge geometry; the expert works there (x-/y-/y+ edges 2/2 each). No dataset recorded on it yet.
- Branch (65): idle. Do not start a recording stack and never run stop_stack.sh (kills every instance)
  until 9d says instance 0 / GPU1 are free (after the 07d eval).

## Changed (2026-09-08 session, COMMITTED as 343d9f9 and pushed)
- Commit 343d9f9 holds everything below. ~50 untracked files from the Codex "ScaleBFM visual-expert" thread
  (docs/scalebfm_goal_*, scalebfm_visual_*, src/teleop_free/scalebfm/*visual*.py, tests/test_bfm_visual_*,
  configs/gr00t_bfm_visual_2cam.py) are NOT committed -- that thread owns them.
- docs/bimanual_collection_tutorial.md rewritten (160 -> 463 lines) in collection.md's structure: scene, one
  episode with log reading, pilot + acceptance, control logic with the episode flowchart and per-phase
  criteria table, demo shape, randomisation, cameras, render, path to a dataset, dated pitfalls.
- Disk cleanup (user-approved tiers A/B/C, /workspace 97% -> 77%): DELETED the bfm_trainsmoke, bfm_pilot_v3
  (model + model_vel) and bfm_pilot_v4 checkpoints (their datasets, evals, pilot.log stay), the v3 partial
  evals/probes, outputs/train/{transition_pilot20,prefix_*}, outputs/diag/{transition_*,grasp_handover,
  prefix_target_audit,transport_dynamic}, outputs/videos/{cl_cover400_tok100,wall_pilot_try*},
  outputs/sim_video, /workspace/tmp/{gr00t_val,gso_probe,robocasa,*.zip} and exporter sessions before
  2026-09-08. Kept: ready07d_decisive, ready07_learn, ready07d_200 models; all datasets; HF cache (tier D).
- src/teleop_free/scalebfm/eval_bfm_bottle.py: (1) the settle-hold fallback during the 4 latency ticks
  after each replan is gone (a `live` chunk keeps executing) -- it was 20% of ticks; (2) `--anchor live`
  re-anchors chunk rows on the live pelvis and builds ScaleBFM's futures from velocity columns (4 or 7);
  (3) `--level-pelvis` keeps only the yaw of the pelvis orientation target (the relative rotation compounds
  into a 0.4 rad backward lean); (4) `--pelvis-anchor world|plan` (world: no gain; plan: diverges -- keep
  `live`); (5) `--replay-mode body`, `--replay-round`, `--interp`, `--no-vel`, `--vel-scale`; (6) traces carry
  `pred` (row 0 of the chunk, wr0/wr39, grip_min), `exec` (executed world targets) and `wrist_w`; (7) the
  scene sample is run_bottle.sample_scene (shared with the recorder).
- src/teleop_free/scalebfm/run_bottle.py: `sample_scene` with BFM_SCENE=edge (default, SONIC geometry:
  bottle 4-8 cm inboard of its nearest edge, face-on approach, spawn 1.1-1.3 m, +-20 deg, +-0.20 m) or
  corner (v3/v4 legacy); BFM_EDGE=x-|x+|y-|y+ override; BFM_STAND_D, BFM_REACH_START, BFM_REACH_MIN_S env
  knobs; the wrist-servo BIAS is now rotated into the stand frame (it was a world constant tuned on +x,
  which put the hand 10 cm sideways on +-y approaches).
- src/teleop_free/scalebfm/record_bottle.py: skips the settle (BFM_RECORD_SETTLE=1 restores) and writes
  action.bfm_vel (4: pelvis plan velocity + yaw rate from ref(t+DT)-ref(t)); run_bottle.tick passes cmd_next.
- New: scalebfm/bfm_action.py (offline 7-col velocities), scalebfm/augment_bfm_velocity.py (in-place
  augmentation + idle-prefix trim + video re-cut), configs/gr00t_bfm_vel_2cam.py (26-d),
  configs/gr00t_bfm_vel4_2cam.py (23-d, the current one).
- scripts/bfm_pilot.sh: COLLECT_GPU/TRAIN_GPU/EVAL_GPU, MODALITY, EVAL_EXTRA overrides.
- src/teleop_free/tools/eval_open_loop.py: scores non-SONIC layouts (nmse_body/body_mae) and skips the
  SONIC decoder when the columns are absent.
- src/teleop_free/tools/run_learn_pilot.py: `--stack-env` (e.g. CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1)
  for both stacks; PORT env reaches train_bottle.sh (torchrun port collision with a second training).
- 07d-200 data: collect/inputs.json excludes sessions 15-17 (10:09-10:40 UTC; original inputs_all28.json).
  Their exporter episodes were 1-6 s stubs after deploy self-heal restarts caused by ScaleBFM sim probes on
  the same GPU; the ordered manifest join cannot see that, only check_dataset's grasp-causality pass did.

## Refactor in progress (docs/handoff/refactor-plan.md, confirmed 2026-09-09)
- Two-week feature freeze on main from 2026-09-09: experiments only, no new features outside scalebfm/.
- Step 0 DONE (no main-code change): scripts/gate.py + tests/gate/baseline_{bottle,crate}.json. bottle baseline is
  from the first 30 rounds of ready_visible200 (re-record when the run completes); crate from outputs/videos/wall_pilot.
  Self-compare PASS 19/19 and 8/8; 07d-200 manifests FAIL against it on exit reason / recovery stages / failure set only.
- Paths being moved: none yet. Steps 1-6 (tooling, delete retired, knobs->config, skeleton, crate, docs) not started.
- Cleanup rule: deleting outputs/ dirs or weights needs a 24 h notice here first (a cleanup removed a running smoke dir today).
- Codex thread: please commit the ~50 untracked scalebfm/visual files as-is before step 2 starts.

## Next
1. Ready-grasp line: data volume is flat at 176 ep (seed 7 = decisive level). Attack the failure both seeds
   share -- standing at the table with the arm never leaving rest (s7 39/54, s42 15/43): DAgger with near-table
   spawns covering the stand -> raise transition (failure_anatomy ARM-NEVER-UP prescription), 2 seeds per arm.
2. ScaleBFM decision (user): (a) record 200 episodes on the edge geometry as-is (STAND_D 0.45-0.55, reach
   overlapping the walk -- the best-scoring expert combination, 5/6) and run the v4 pipeline (~2.5 h), or
   (b) first a mode-6 spike (pelvis + shoulder + elbow + wrist targets) to remove the ~20 deg waist twist,
   which needs an arm IK for the elbow target. The 10-cm stop-short of v4 is a policy stop-rule problem
   (gait-state keyed) that a cleaner expert does not fix by itself; DAgger from stopped-short states or a
   STAND_D curriculum remain the candidates for that.
3. Branch (65): symmetric jaw command (drive q1-q2 to 0) against close-time ejection.
4. Progress-parameterised expert (replace time u with path progress) before the DAgger stage.

## Blocked / waiting on
- nothing.

## Open questions
- ScaleBFM twist: waist yaw/roll ~20 deg peaks 1 s after the reach starts, at every STAND_D from 0.25 to
  0.55; reach-after-arrival trims yaw 23->14-18 deg, roll unchanged, success drops (3/6). It is mode 2's
  reaching strategy (no torso constraint), not the stand distance. Closer stands (0.25-0.40) fail in the
  carry (bottle dropped when pulled 0.15 m toward the body near the edge) -- CARRY_BACK needs retuning
  before any closer stand is used.
- ScaleBFM stop-short: v4 stops at 0.55-0.61 m and drifts to 0.62-0.70 m while leaning; its stand offset
  stays +0.02 because no training frame shows a stopped robot that must keep walking.
- check_dataset's rise check needs root_position to be frame-correct; SONIC datasets lack it, so their
  "min rise" numbers under-read (the 07d-200 audit passed 32/32 after the stub sessions were removed).
- Ready-grasp line: 07d-200 (176 ep) 57/100 vs 07d (60 ep) 29/60 paired 37 vs 29, p 0.24. Data helped the
  stand (reach-fail min dist 0.24 m vs 0.37 m) but 'at the table, never closes' is still 34/43 of the
  failures; a second training seed on the same dataset is the cheapest next fact.
- failure_anatomy.py's arm-up threshold is inverted for the ready-grasp line (rest 0.90 m > close 0.72 m): its
  'arm never up = 0' is wrong there; use palm_z min vs the successes' close height instead.
- The v3/v4 ScaleBFM checkpoints are gone (cleanup); their numbers live in pilot.log/STATE/memory only.
  Re-deriving them means re-running bfm_pilot.sh on the kept datasets (~80 min each).

## Parked (do not forget)
- SONIC lockstep collection: docs/handoff/sonic_lockstep_collection_plan.md
- Isaac render + MuJoCo physics: memory isaac-render-mujoco-physics-plan
- v4 clockless expert (0/3, abandoned; lessons in memory scalebfm-interface-facts)

- Disk cleanup 2026-09-09 (branch 65, user-approved): outputs/experiments 195G -> 101G. Removed the checkpoint-3000
  safetensors + optimizer.pt in all 7 model dirs (byte-identical to the top-level weights, verified with cmp;
  WEIGHTS_REMOVED.txt left in each), collect/ + train_dataset/ of 07b/07c/07e/07s2 (eval traces kept), and the
  bfm_trainsmoke / ready_visible_smoke dirs. policy_goal kept. Resuming those trainings is no longer possible.

- 2026-09-09 (65): SIMPLE moved physically from /workspace/code/SIMPLE to third_party/SIMPLE (paths.sh/paths.py
  defaults, docs/setup.md step 7, venv .pth + shebangs, locomanipulation-render's two hard-coded paths all updated).

## Recent results (last 3)
- 2026-09-09 07d-200 seed 7: 46/100 vs seed 42 57/100 (21:32, p 0.17); vs decisive 27 vs 29 (p 0.87) -- data gain was seed noise
- 2026-09-09 07d-200 + ensemble N3 M0.3: 3/20 vs 9/20 paired (3:9, p 0.15); ensemble delays/blocks the arm drop, negative
- 2026-09-08 ready07d_200 (176 ep): 57/100; paired first 60 37 vs 29 (p 0.24); reach-fail stops closer (0.24 m) but still no close
- 2026-09-08 ScaleBFM edge geometry: expert 2/2 on x-, y-, y+ edges after the BIAS fix; STAND_D sweep
  (6 rounds each) 0.25-0.30 3/6, 0.30-0.35 4/6, 0.35-0.40 3/6, 0.40-0.45 2/6, 0.45-0.55 5/6; twist ~20 deg
  everywhere (outputs/videos/scalebfm_edge_scene_20260908/)
- 2026-09-08 bfm_pilot_v4 (native velocity, 23-d): 0/60, stops 10 cm short; open-loop body_mae 0.44 cm
- 2026-09-08 bfm_pilot_v3: 18-d 0/29 (never left the spawn); 26-d offline velocity 0/60 + 0/60 (closes 6 cm
  short / 8 cm high, open-loop 0.90 cm)
