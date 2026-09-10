# STATE (updated 2026-09-10 by the branch session: ScaleBFM reorg; incident block below still applies)

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

## 固定规则:ScaleBFM正式批量采集
- 必须GPU EGL渲染;同时显式配置CUDA设备与EGL设备,不得将OSMesa诊断配置沿用到正式批量采集。
- 启动或切换渲染/设备/场景前先做独立pilot:覆盖三种桌面条件、检查成功/可见性/质量/视频和墙钟吞吐。
  当前200集runner门槛12集入选、24视频解码通过、每集墙钟中位数<=60s;CPU余量>=6核且无限流。
- 软件渲染仅显式诊断例外;采集入口需要--allow-software-rendering。断点续采保留完整集及渲染来源历史。
- 完整规则与命令:docs/scalebfm_collection.md「正式采集必须遵循的渲染与吞吐规则」。

## Now
- SONIC phase2 STOPPED before any action test: supervisor2545023 exited1 after10min resource timeout,
  private diagnostic deploy on GPU1 instance1; no production binary/source changes, no exporter.
  outputs/diagnostics/sonic_lean_phase2_20260910/{plan.json,runner.json,resources.jsonl,cases.log,progress.json,run.exit}.
  Five conditions x3: legacy waist0.52, matched waist-only, +legs, +root orientation, full official static pose.
  Official squat frame85: torso~35deg/rootz0.558m (rootz not explicitly commanded). Actual encoder input/token
  capture in encoder.bin; report script sonic_encoder_capture_report_20260910.py. Do not command instance1.
  No cases executed and no encoder.bin produced; instance1 automatically cleaned up. Last free CPU samples~8cores
  still reset streak (gate also requires zero throttling). Diagnose resource timing before retry, not model results.
- RUNNING ScaleBFM200 on GPU1 EGL (Codex,user-authorized): driver2516911/child2518813. EGL_DEVICE_ID=1,
  CUDA_VISIBLE_DEVICES=1. 12/12 independent pilot passed all audits,24/24video decode, median5.497s/episode
  (OSMesa prior33 median175.94s). make check passed. 37/200 accepted at last check; new train episodes5.1–5.7s.
  Old34 complete episodes retained, interruptedseed4035preserved/skipped; frozen expert source unchanged.
  Root outputs/experiments/scalebfm200_noretract_sonic_20260910; plan/rendering_history distinguish oldOSMesa
  and newEGL cases. Automatic collect->convert->stats continues; RESULTS.json only after all complete.
- DONE (Codex SONIC GPU1 isolated-waist verification): batch.exit=0, instance1 stopped. All10 formal cases
  (neutral/reach x5 repeats, each planner/streamed x6 angles) complete:120 angle conditions, no recorded falls.
  live/phase1_summary.json: at request29.8deg, mean actual waist neutral planner/streamed0.04/0.68deg,
  reach1.82/2.30deg; torso1.95/6.33 and3.46/6.31deg. Reach motor waist target is opposite sign,
  mean-9.57/-11.11deg; neutral motor target0.95/1.11deg. Confirms output is not faithful requested waist tracking;
  cannot yet distinguish learned coordination response from online-reference/mode problem. No fix applied.
  Next: coordinated full-body reference and encoder-input capture, not more identical waist sweeps.
  ScaleBFM original collection still running,32/200 accepted at status check. Ledger bottle §13 final results.
# Re-applied 2026-09-10 ~09:50 UTC by session cf from /workspace/tmp/STATE_cf_fragment_20260910.md (Codex's hunks lost in the
# STATE truncation, see 65's notice below); Codex owns these lines and may rewrite them.
- HISTORICAL (superseded by EGL restart above): ScaleBFM 200-success collection + conversion, driver 2460752 / child
  2460783 on GPU1. At 2026-09-10 09:08 UTC: 11/200 accepted, 11 completed attempts; conversion not started.
  outputs/experiments/scalebfm200_noretract_sonic_20260910/progress.json says collect.
  Original waiting GPU0 driver 2452516 stopped before any episode; plan_gpu0_before_collection.json archived.
  GPU1 had 8977/32607 MiB occupied, five 0% utilization samples; shared GPU allowed by this user request,
  CPU headroom checked, OSMesa + single CPU threads retained. Do not schedule additional work on GPU1 without checking.
- Codex SONIC bend audit 2026-09-10: current ready-grasp excludes waist IK; generic IK pitch capped at
  0.20 rad. Historical isolated-waist 2–10% result does not establish whole-body inability, raw traces lost.
  CPU-only real serialization/encoder/decoder audit passed (plus dropped-pitch negative control); no physical
  reproduction while both collections run. Findings/commands/next controlled test: docs/ledger/bottle.md §12.
  No expert/model changes. Await collaborator's interface/checkpoint/reference details.
- Codex 2026-09-10: ScaleBFM expert/eval now share sonic_lift6cm_hold2s_v1 (>6 cm lift for >2 s; fall <40 cm).
  Removed phase/grip/tilt/hand-speed success gates and preclose-knock early abort; 24 s budget and collection quality
  audit retained. 17 focused tests + thread-limited make check passed; no GPU run. Ledger scalebfm §17 supersedes §16's hold criterion.
- Codex 2026-09-10: user cancelled ScaleBFM post-lift retraction. GraspConfig.carry_back and the bottle_visual JSON
  default are now 0 (was 0.07 m); collection and video-check defaults inherit it. Lift/height hold remain unchanged.
  No GPU job started; SONIC visible200 collection is unaffected. See docs/ledger/scalebfm.md latest entry.
- visible200_20260910 DONE 2026-09-10 07:02-10:43 UTC (session cf, instance 0 / GPU0, stack stopped): 34 batches, 335 rounds
  -> 200 kept (yield 0.60), convert rc=0, check_dataset 32/32 PASS on train (190 ep, 81,425 frames, 25 fps, 910 MB) and val
  (10 ep, 4,743 frames). 34/34 exporter sessions match their manifests, kept episodes 831-1433 rows. Dataset:
  outputs/experiments/visible200_20260910/{train_dataset,val_dataset}; sessions /workspace/tmp/p03/outputs/2026-09-10-07-02-53 ..
  10-3x. Sample reel outputs/videos/visible200_20260910/reel_sample10.mp4. Caveats: 65's crate chain shared GPU0 07:45-08:30
  (batches 07-14: 43/80 vs 157/255 elsewhere; no deploy trips, audits clean); round 1 of EVERY batch (34/34) failed at the
  spawn -- root cause + fix on branch fix/first-round-anchor (ledger bottle §12), verification running on GPU0 now
  (outputs/experiments/firstround_fix_20260910, 3 batches x 4 rounds, ~8 min). Not trained.
- RUNNING (session cf): visible200 collection + convert on instance 0 / GPU0 since 2026-09-10 07:02 UTC, ends ~10:30 UTC
  (outputs/experiments/visible200_20260910, run_learn_pilot through=convert). Do NOT start a stack on instance 0, put
  anything on GPU0, or run sim probes on either GPU until this line is gone. Batch results in collect/manifest_*.jsonl.
- visible10 look-see collection DONE 2026-09-10 06:32-06:42 UTC (branch session cf, instance 0 / GPU0, stack stopped):
  bottle_visible.json + learn_200.json with through=collect successes=10, seed0 31000, first run with restart_deploy=false.
  2 batches, 18 rounds -> 10 kept (yield 0.56): batch 0 4/10 (3x "toppled in the close phase", 1 view-at-spawn, 1 IK,
  1 stand infeasible), batch 1 6/8 (1 turn timeout, 1 stand infeasible). Exporter sessions
  /workspace/tmp/p03/outputs/2026-09-10-06-{32-40,37-49}: discard indices match the manifests, kept episodes 853-1074 rows.
  Runner scripts/experiments/visible10_20260910.sh (waits on GPU0 memory + cpu_headroom); reel of the 10 successes
  outputs/videos/visible10_20260910/reel_10ok.mp4. Not converted, not trained. Open: the 3 close-phase topples in batch 0.
- Codex sync 2026-09-10 through 092b042: Claude review fixes inspected; see docs/ledger/refactor.md latest entry.
  Backend fingerprints, scene_dir, input hashes and causal/Task wording repaired; config file load still bypasses
  schema_version validation; old diagnostic references remain. Thread-limited make check passed, 9 focused tests passed.
  Observed another session's visible10 collection process active; no GPU/stack actions or implementation edits by Codex.
- Codex review 2026-09-10: audited refactor through 1b32a48 (base 78536ae; ScaleBFM base a3e2fee). Findings in
  docs/ledger/refactor.md: missing config validation, ScaleBFM backend omitted from source hashes, ignored scene_dir,
  lost SONIC input-asset hashes, and stale/overstated documentation. No implementation fixes or GPU jobs started.
  make check hit the known wall-clock-sensitive offhand test; CPU thread-limited default suite passed 224, skipped 9.
- ScaleBFM line REORGANISED 2026-09-10 (branch refactor/scalebfm, worktree .claude/worktrees/refactor-scalebfm; the
  Codex thread has finished, user-confirmed): its working tree was committed as-is (a3e2fee), then moved into
  src/locomani/backends/scalebfm + src/locomani/tasks/bottle_visual (f8fc71b), knobs into ScaleBfmBackendConfig /
  BottleVisualTaskConfig with no os.environ (51a4d25), docs into docs/ledger/scalebfm.md (§1-14, all numbers) +
  docs/scalebfm_collection.md (tutorial) + docs/archive/scalebfm_*. Verified: round.py replays 3 seeds byte-identical
  to a3e2fee after each step; make lint + make test green; bottle_visual registered as the third Task (contract test
  passes). Nothing of this line runs; its checkpoints/data/scene builds are gone (ledger §13). Follow-ups in
  docs/ROADMAP.md "ScaleBFM 线重组后的后续". Everything Codex reported before the incident (Goal A/B/C 13/17/11 of 60,
  yplus200 25/100, pelvis ABCD 2/120, policy audit, video check) is in the ledger, not here any more.
- 9d: NOTHING RUNNING (2026-09-10 03:30 UTC). GPU0 idle; GPU1 holds a foreign AnyImageNav job (~15 GB, 97 % util,
  load ~15) -- do NOT put a sim on GPU1 while it runs (that alone produced 4 close-phase topples in 10 rounds).
  Instance 0 / 1 stacks down, port 5590 free. outputs/ holds only: scene/ (reinstalled + gate_922 + crate_bimanual),
  experiments/gate_bottle_* and gate_crate_*, gate/. Raw exporter sessions: /workspace/tmp/p03/outputs (82).
- ScaleBFM: nothing running. Three pilots today (v3 18-d, v3 26-d offline velocity, v4 23-d native
  velocity) all 0/60; the failure moved from "never leaves the spawn" to "stops 10 cm short of the stand".
  The scene geometry was then found to be wrong (corner approach, 0.45-0.49 m reach) and replaced by the
  SONIC edge geometry; the expert works there (x-/y-/y+ edges 2/2 each). No dataset recorded on it yet.
- Branch (65) 2026-09-10 09:40 UTC: NOTHING RUNNING from this session; instance 2 free. TWO NOTICES. (1) OVERLAP: my queued
  crate chain (pilot 07:45-07:55 on instance 2, Isaac renders 07:55-08:30, both GPU0) started inside session cf's visible200
  per-batch stack restart -- its port poll saw the gap. cf's batches 07-14 in that window yielded 43/80 vs 66/100 outside,
  no deploy trips; within batch noise but not excluded as my contention. (2) STATE.md CLOBBER at ~09:40 UTC: a branch-session
  script truncated docs/STATE.md before reading it; the file was restored from HEAD b68c1f9, but cf/Codex's UNCOMMITTED
  STATE hunks were lost from disk -- the first 60 diff lines (Codex ScaleBFM RUNNING/QUEUED lines, cf visible200 RUNNING
  line) are saved verbatim in /workspace/tmp/STATE_cf_fragment_20260910.md; anything beyond them must be re-entered by
  its author. Rule from now: read the RUNNING lines here + nvidia-smi compute apps before any GPU0 job; a queued runner
  re-checks right before it starts (memory gpu0-port-gap-overlap); never open a file for writing before reading it.
- Branch (65): idle. Do not start a recording stack and never run stop_stack.sh (kills every instance)
  until 9d says instance 0 / GPU1 are free (after the 07d eval).

## Changed (2026-09-10 ~09:40 UTC, new pick-and-place session; UNCOMMITTED, CPU-only, no GPU touched)
- SIMPLE's basket scene brought into MuJoCo: assets/simple/{basket,sugar_box,soup_can}/model.xml (visual OBJ +
  CoACD convex pieces from third_party/SIMPLE/data/assets, basket rotated y-up -> z-up, origin = centre). scene_lib:
  three MESH_TARGETS entries + SCENE_PRESET=pick_place (basket at (-0.34, 0) yaw 90, sugar box (-0.40, +0.28),
  soup can (-0.40, -0.28); target soup_can, goal basket via a new per-preset goal_object) and a `container=True`
  flag that skips the jaw-span check; compose_scene.py --preset. Checks: OSMesa placement renders + 2 s settle
  (objects sink <= 0.4 cm, no launch) + soup can dropped into the basket rests inside at 0.4 cm off centre:
  outputs/scene_checks/pick_place_20260910/. make lint green. No expert / pipeline / Task yet.

## Changed (2026-09-10 branch session 65, COMMITTED on main through b68c1f9; pushed through ebac697)
- Collection default: no per-round deploy restart (restart_deploy=False, ebac697) with a reset-phase reanchor
  (b2310c8); paired A/B 18/30 vs 16/30 at 34 vs 64 s a round (ledger bottle §11); gate baseline re-recorded;
  eval_closed_loop pins the restart. SIM_VIS default none (18dd6dd). scripts/cpu_headroom.sh (container quota 15.36
  cores, 55cb77c). Crate: WALL_WALK/WALL_FEATURE env vars documented dead (b07caef); placement design
  crate_setback_levels_m x crate_yaw_levels_deg with the walk-in stand along the crate normal (166cd37); ghost hands
  (IK reference pose) in MuJoCo + Isaac sbs renders, on by default (5bc1e0d, 3e18990); mujoco_render --insets;
  summarize_wall_pilot label fix. Ledger crate.md: ready-stand 8/10, walk-in 4/10 (CPU-starved), hard placement
  5/5/7 of 10 with the stand-off >= 0.26 boundary. Videos: outputs/renders_isaac/crate_hard10r_20260910/ (final),
  outputs/videos/restart_ab_20260910/ (A vs B' pairs).

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
5. Branch 65 next: (a) crate walk-in stand biased toward the far rail end or a [stand] far-hand reach gate, then re-run the
   hard-placement 10 (memory crate-standoff-times-yaw-boundary); (b) speed levers 2-4 from ledger bottle §10 (post-'c'
   hold 4 -> 1 s, restart-script polling, 20-round batches) -- each a gate run; (c) a GPU0 lock file both sessions honour.

## Blocked / waiting on
- PRE-EXISTING unit failure on main (not from the ScaleBFM merge, fails on 6857fbb too, 2026-09-10):
  tests/test_pose_feedback.py::PoseFeedbackTests::test_offhand_guard_reuses_current_observation raises
  'stale observation for offhand guard' from backends/sonic/streamed_arm.py:235; make check is red until fixed.
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
- No-restart collection parks the base 3 cm farther (park 0.332 vs 0.301) and adds 'local alignment disturbed base' 2/30:
  does it change the trained policy's stand? Only the next 200-episode dataset + closed loop can answer (ledger bottle §11).
- Crate hard placement: is the fix the stand bias, the reach gate, or a lower STAND_OFF_WALK? Untested; the boundary is
  clean (stand-off >= 0.26 fails), the gait scatter 0.15-0.30 is the variable.
- Did my GPU0 renders cost cf's visible200 batches 07-14 their yield (0.54 vs 0.66)? Unknowable after the fact; the
  dataset audit (check_dataset) on those batches is the only check.
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
- 2026-09-10 crate hard placement (setback x yaw) 5/5/7 of 10 same seed; stand-off >= 0.26 fails in descend; final videos with ghost hands
- 2026-09-10 restart A/B paired 30: no-restart+reanchor 18 vs restart 16, 34 vs 64 s/round, gate 13/15 -> default flipped
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
