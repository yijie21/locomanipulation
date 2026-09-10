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
- 9d: NOTHING RUNNING (2026-09-10 03:30 UTC). GPU0 idle; GPU1 holds a foreign AnyImageNav job (~15 GB, 97 % util,
  load ~15) -- do NOT put a sim on GPU1 while it runs (that alone produced 4 close-phase topples in 10 rounds).
  Instance 0 / 1 stacks down, port 5590 free. outputs/ holds only: scene/ (reinstalled + gate_922 + crate_bimanual),
  experiments/gate_bottle_* and gate_crate_*, gate/. Raw exporter sessions: /workspace/tmp/p03/outputs (82).
- ScaleBFM: nothing running. Three pilots today (v3 18-d, v3 26-d offline velocity, v4 23-d native
  velocity) all 0/60; the failure moved from "never leaves the spawn" to "stops 10 cm short of the stand".
  The scene geometry was then found to be wrong (corner approach, 0.45-0.49 m reach) and replaced by the
  SONIC edge geometry; the expert works there (x-/y-/y+ edges 2/2 each). No dataset recorded on it yet.
- Branch (65): idle. Do not start a recording stack and never run stop_stack.sh (kills every instance)
  until 9d says instance 0 / GPU1 are free (after the 07d eval).

## Changed (2026-09-09/10 session 9d, COMMITTED on main through 507be6a; NOT pushed)
- Refactor steps 1-6 merged (docs/handoff/refactor-plan.md, 13 decisions + D-group ruling; execution ledgers in
  docs/handoff/refactor-step2-candidates.md, refactor-step3-config.md, refactor-step4-layout.md):
  1 tooling (ruff, pre-commit local hooks, scripts/lint_sizes.py + lint_env.py, test tiers, Makefile, docs/reference/standards.md);
  2 ~12k lines of retired branches deleted (A/B/D groups; RTC cluster + --client blocking KEPT by ruling);
  3 knobs -> config dataclasses (BottleTaskConfig / CrateTaskConfig / SonicBackendConfig / EvalConfig / PilotConfig,
  configs/{tasks,backends,eval,pilot}/*.json, --config/--set CLIs, config_snapshot.json replaces provenance.json,
  docs/reference/config_fields.md generated, lint_env --strict = 0);
  4a package move src/teleop_free -> src/locomani/{pipeline,backends/sonic,sim,tasks/bottle,tasks/crate,primitives,rtc},
  stack shell in scripts/stack/, 8 alias stubs under src/teleop_free/ for the frozen scalebfm/;
  4b pure-extraction splits of grasp_gym, stream_reach, vla_round, crate run_round, Primitives (13 -> 5 files over 600);
  5 Task seam (tasks/base.py, registry, locomani.cli, tests/test_task_contract.py);
  6 docs into four kinds (tutorials / ledger / reference / state) + docs/archive/INDEX.md; CLAUDE.md updated.
- Verification (no functional change): see 'Refactor merged and verified' above; tests/gate/results/.
- Gate-found fixes after the merge: run_instance_exporter bootstrap (326450b), 'approach' stage marker + gate.py rules
  (a88c5f5), script-vs-package module alias in bimanual_wall / bimanual_carry / grasp_gym + Codex tests scoped out of the
  unit tier + reach_config test fixture (1d35aad).
- Visible-expert collection ready_visible200_20260909: 200 episodes over 39 batches + 3 top-ups; 8 sessions excluded
  for deploy-restart stubs/truncation; convert + audit passed 32/32 -- then the dataset, manifests and everything else
  under outputs/ were deleted by the merge incident. The 40 exporter sessions survive.
- Incident fixes: outputs symlink untracked, /outputs and /third_party ignored as paths (4a109bf); memory
  never-git-add-all-with-symlinks.

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

## Refactor status
- Steps 1-6 DONE and merged (see Changed above). Freeze stays until 2026-09-23 for the follow-ups below.
- Follow-ups: (a) eval-path A/B once a model exists (20 rounds seed 6000, old vs new eval client on the same checkpoint);
  (b) size caps: 5 files / 79 functions left, lint_sizes --strict not yet flipped; (c) step 4c (pure `decide(snapshot,
  cfg)` expert + offline replay gate) deferred to the DAgger stage; (d) delete the stale worktrees under .claude/worktrees
  (refactor-step1, old-78536ae, agent-*) after the user's review; (e) push main.

## Next
1. Data recovery (USER DECISION): (a) re-convert ready_visible200 from the 40 surviving exporter sessions without
   manifests (ok/failed rounds via each session's discarded_episode_indices; verification-tail trim marker lost, ~100
   extra hold frames per episode; the 8 excluded sessions are listed in this session's scratch scripts and must be
   left out), ~20 min CPU; or (b) recollect 200 episodes with the refactored code on GPU0, ~3.5 h.
2. Train the visible-expert model (3000 steps, ~77 min, GPU0) and eval 100 rounds seed 6000. No paired baseline
   survives (57/100 manifest lost): compare against the numbers in docs/ledger/bottle.md, and run the eval-path A/B
   (old 78536ae eval client vs new, 20 rounds) on that checkpoint first.
3. Then the plan's next lever: DAgger with near-table spawns for the 'arm never leaves rest at the table' failure,
   built on tasks/bottle (step 4c seam), 2 seeds per arm.
4. ScaleBFM / branch 65 items unchanged (below).

## Blocked / waiting on
- User decision on data recovery (Next 1). Nothing runs until then.
- GPU1 is unusable for sims while the foreign AnyImageNav job holds it at 97 %.

## Open questions
- Untrimmed verification tails (if recovery path (a)): does training on ~100 extra 'hold the lifted bottle' frames per
  episode change closed-loop behaviour? Unknown; 07d always trimmed. If it matters, path (b) is the clean answer.
- Deploy self-heal restarts truncated 8/40 collection sessions on 2026-09-09; cause is host contention (foreign GPU/CPU
  load), the stubs are caught only by check_dataset's grasp-causality audit + the manifest/episode count check.
  A per-session sanity check right after each batch would save the re-convert loops.
- Gate baselines are 30 rounds (bottle) / 10 rounds (crate): rare failure strings show as 'new'; widen the baselines to
  100 / 30 rounds once the machine is quiet.
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
- 2026-09-10 refactor gates on main (GPU0): converter 34/34 byte-identical; bottle 7/10 all metrics PASS; crate 8/10 = baseline
- 2026-09-09 visible expert collection: 200 ep in 297+ rounds, yield ~0.62, 9/9 brakes at 0.30-0.31 m, no recovery stage (dataset lost)
- 2026-09-09 07d-200 seed 7: 46/100 vs seed 42 57/100 (21:32, p 0.17); vs decisive 27 vs 29 (p 0.87) -- data gain was seed noise
- 2026-09-09 07d-200 + ensemble N3 M0.3: 3/20 vs 9/20 paired (3:9, p 0.15); ensemble delays/blocks the arm drop, negative
- 2026-09-08 ready07d_200 (176 ep): 57/100; paired first 60 37 vs 29 (p 0.24); reach-fail stops closer (0.24 m) but still no close
- 2026-09-08 ScaleBFM edge geometry: expert 2/2 on x-, y-, y+ edges after the BIAS fix; STAND_D sweep
  (6 rounds each) 0.25-0.30 3/6, 0.30-0.35 4/6, 0.35-0.40 3/6, 0.40-0.45 2/6, 0.45-0.55 5/6; twist ~20 deg
  everywhere (outputs/videos/scalebfm_edge_scene_20260908/)
- 2026-09-08 bfm_pilot_v4 (native velocity, 23-d): 0/60, stops 10 cm short; open-loop body_mae 0.44 cm
- 2026-09-08 bfm_pilot_v3: 18-d 0/29 (never left the spawn); 26-d offline velocity 0/60 + 0/60 (closes 6 cm
  short / 8 cm high, open-loop 0.90 cm)
