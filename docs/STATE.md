# STATE (updated 2026-09-11 00:50 UTC by session cf; several sessions write this file, keep edits surgical)

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
- **CRATE/BIMANUAL ENTRY POINTS REFACTORED: driver and expert are separate now, like the single-arm line
  (branch session, 2026-09-11 14:45-15:2x UTC). NOTHING RUNNING of mine; instance 1 released, GPU1 2 MiB.
  Another session holds instance 0 / GPU0 (stagee_az29b_20260911), untouched by this work.** Ledger crate.md, entry 2026-09-11. NOT COMMITTED YET.
  User asked for it: the double-arm and pick-and-place lines called their scripts differently from the
  single-arm one. `scripts/run_wall_pilot.sh` (83 lines) and `run_bimanual_pilot.sh` (72) were the same text
  twice over (33 of 155 lines differed, 10 of them comments) and each did four jobs at once -- pin
  instance/GPU, start the stack, run the expert, stop it with a hand-rolled `pgrep -f`. Now
  `src/locomani/pipeline/run_crate_pilot.py` is the driver (stack lifecycle, run dir, manifests, provenance)
  and the experts are launched the way `grasp_gym.py` is, from a 50-line shell + two 3-line shims.
  **Four latent defects fell out of it:** (1) `pipeline/stack.py`'s `start_stack`/`stop_instance` were DEAD
  CODE -- decision 8's "one place a config becomes shell env" had no caller outside `export_env`;
  (2) `SonicBackendConfig.no_encoder` defaults **True** (the EVAL protocol) while `start_stack.sh` defaults 0,
  so routing the crate stack through that seam naively sets NO_ENCODER=1 and silently drops the planner's arm
  and crouch references; (3) the experts open the manifest in APPEND mode and both runners pointed every run
  at `outputs/videos/wall_pilot/manifest.jsonl` -- which is what `gate.py record crate` reads, so two pilots
  became one population; (4) the experts resolve ports with `os.environ.setdefault`, so an inherited
  instance-0 `TF_*` block (paths.sh exports one whenever INSTANCE is unset) beats `locomani.sim.ports`.
  **Four more asymmetries fixed on the user's instruction:** pad hardening had two implementations with two
  constant sets (`build_sim_scene` opt-in solref 0.004 / `bimanual_scene` unconditional 0.002, applied on top)
  -> one implementation + named `PAD_LEVELS{object,crate}`, and it is PROVEN physics-neutral (new vs old crate
  XML differ only in two `contype=0` ghost geoms the old prefix match caught; **0 of 47 collidable geoms differ
  in priority/condim/solref/solimp**, so no baseline needs re-recording); `CrateTaskConfig` gained `prompt`
  (stack.py reads it; the crate stack had been falling back to "walk forward") and `deploy_self_heal_rounds`;
  the deploy self-heal was three private copies (grasp_gym had a success check, bimanual_wall had none,
  bimanual_carry had no self-heal at all) -> `backends/sonic/deploy.{restart,SelfHeal}`; and **`--record` is
  wired**, so the crate line can produce a LeRobot dataset for the first time (crate_collection.md §11 was
  "the missing step"): same key protocol as the bottle line, `produced` in the manifest, forced to instance 0
  with `ports.require_exporter()` refusing an empty record. `cli.py`'s dispatch table now matches the registry
  (pick_place had no row at all despite being run_learn_pilot's task).
  **GATE: three 10-round runs, all `SONIC_POLICY=policy/release` (baseline_crate.json is 2026-09-09 and the
  default flipped to v1.1 today), seed 0, instance 1/GPU1.** runner-refactor-only 10/10 **PASS 8/8**;
  +C/D/E/F 8/10 FAIL 6/8 (`stand_off_m` 0.2149 -> 0.1889, `failure_reasons`); **the SAME code again** 8/10
  FAIL 7/8 (`stand_off_m` 0.2065 -> PASS, `failure_reasons`).
  **Both failing checks are gate resolution, not a regression, and it is now measured, not argued:** the same
  code twice moves `stand_off_m` by **0.0176 m = 1.8x its 0.010 tolerance**, and that tolerance is
  `TOL["depth_m"]` borrowed for a quantity that is the ACHIEVED stand-off after a walk-in (per round, same
  seed, up to 11 cm apart between runs). `failure_reasons` reported DIFFERENT new strings in the two
  same-code runs -- this file's own open item ("crate baselines are 10 rounds: rare failure strings show as
  'new'"). `yield` passed all three (0.8/1.0/0.8 vs baseline 0.8), as did depth_m, exit_majority, stage_s.*
  and yaw_err_deg. Scene XML md5-identical across runs and the planned stands identical, so the divergence is
  downstream of anything that was edited. **I did not widen any tolerance to make the change pass.**
  **The root cause is the walk-in's stand delivery, and the user parked it on 2026-09-11 ("先放一放"),
  so the gate is parked with it -- see Parked below. No tolerance was widened.**
  Artifacts outputs/experiments/{crate_runner_refactor_20260911_release,crate_tasks_defg_20260911,
  crate_variance_run3_20260911}/ (each with gate_crate.json). Instance 1 released, GPU1 2 MiB.
  **Doc sweep afterwards caught a real break of my own:** C had changed `--harden-pads` to an
  optional-value flag, and the command recorded in `docs/reference/runbook.md:75` /
  `docs/pick_place_collection.md:26` is `--harden-pads outputs/scene outputs/scene/pick_place` --
  the optional value swallows the first positional (`invalid choice: 'outputs/scene'`, exit 2).
  Now `--harden-pads` (store_true) + a separate `--pad-level {object,crate}`; the documented form is
  restored verbatim and the crate XML is unchanged. Rule worth keeping: grep the docs for a flag's
  existing spelling before changing its arity.
- **STAGES A-E CLOSED OUT (session cf, 2026-09-11 15:40-16:08 UTC). NOTHING RUNNING on instance 0; GPU0 2 MiB.**
  Ledger bottle.md §28-29. Branch `v11-recalibration-20260911`, 8 commits, main untouched.
  **(D) The wrist delivery ratio is NOT measured and TRACK_ROT_BOOST stays 4.5.** Two runs of the restored
  diag_wrist disagree, and the second reports the channel INERT on policy/release -- where the 22% the constant
  inverts was measured. Reproducing a known-true prior is what identifies which run is wrong, and it says both
  are: the first had the robot fall (sim log, height 0.199 m) while still printing ratios; the second shows all
  four cases at the same palm position, so `set_wrists` took no effect at all. The tool now resets per case,
  checks `fallen` while holding, and prints the per-axis ratio spread -- the next step is to establish which
  stream mode consumes set_wrists (agent_server.py:293; default PLANNER_VR_3PT) and enter it after the reset.
  Do not poke this constant on an unreproducible measurement: its failure mode is an integrator limit cycle.
  **(E) The bottle gate baseline is re-recorded on v1.1**, 30 rounds, collection-style (sampled cases, repo
  profile). The release one is kept as tests/gate/baseline_bottle_release_20260910.json. A first 10-round attempt
  was discarded -- the gate's yield tolerance is +-0.15 and the binomial sd at n=10, p=0.3 is 0.145. Self-compare
  PASS 15/15.
      metric        release(30)  v1.1(30)        yield 0.600 -> 0.233, end_depth 0.3011 -> 0.2998,
      squat_drop 0.13 both, seconds 33.7 -> 19.95, exit_majority identical
  **The two baselines are NOT comparable** (different sampling era, different profile): 0.600 vs 0.233 is not a
  checkpoint comparison. The paired one is §27: 7/20 vs release's 9/20, p=0.754.
  **PREREQUISITE LANDED:** `stand_azimuth_deg: 29.0` is now in `configs/collection/ready_grasp_visible.json` and
  `configs/tasks/bottle_visible.json`; the other six ready_grasp profiles stay at 0.0 (only the visible line was
  measured). Recording the baseline before this would have enshrined the 0/20 configuration.
  **OPEN (§29.1):** the sampled batch produced failure kinds the fixed-bank batch never did -- fingers against the
  dining table, `coupled safe-hand path blocked`, `coupled transport disturbed`. Either the 29 deg stand faces the
  object more squarely and spends table clearance, or the sampled distribution simply covers harder placements.
  Discriminate by fixing the offending cases into a bank and running plan 37.5 vs plan 29 paired. Until then
  0.233 must NOT be attributed to the azimuth.
- **CONFIRMED ON 20 PAIRED ROUNDS: v1.1 is back at release's level (session cf, 2026-09-11 15:30-15:39 UTC).**
  Ledger bottle.md §27. Same 20 cases, bank differing only in `rel`/`stand`:
      release, plan 37.5          9/20   azimuth 31.1 [13,45]  depth 0.307  lateral 0.204  park 0.369
      v1.1 old, no fixes          0/20   azimuth 40.9 [30,49]  depth 0.332  lateral 0.305  park 0.467
      v1.1 all fixes, plan 29     7/20   azimuth 35.2 [25,40]  depth 0.298  lateral 0.207  park 0.363
  Paired McNemar: **7:0 vs the old v1.1 arm, exact p=0.0156**; 4:6 vs release, **p=0.754 -- statistically
  indistinguishable**. lateral and park land on release's values and the azimuth spread narrows from [30,49] to
  [25,40]. "All fixes" = APPR_COARSE 0.40 (§22) + the reset seating fix (§24) + TRANSPORT_ROTATION_TOL 0.35 (§23)
  + planned azimuth 29 (§26).
  **So STATE's long-standing "v1.1 cannot drop into the collection line" is LIFTED.** It was true (0/20) but the
  cause was four things calibrated on -- or hidden by -- release, and two of them (the reset seating, the planned
  azimuth) were checkpoint-independent defects that release's own execution defect was masking.
  Remaining failures on the new arm: `stand infeasible` 6/20 (release 8/20, fewer), `local alignment disturbed
  target or base` **5/20 -- NEW**, release had 0; it fires in the last alignment before the grasp. Read it as
  stage progress, not regression: far more rounds now reach alignment at all.
  **RETRACTED from §26.3:** "apply backshift radially" is wrong. `project_squat` models the squat retreat as
  `backshift * forward` -- a pure DEPTH retreat -- so compensating it with a depth shift is dimensionally right;
  radial would inject backshift*sin(az) ~ 4 cm of lateral error. The real structure: the planned azimuth is a
  PRE-squat quantity while arm comfort is POST-squat. Adding the measured 0.089 m retreat: release 28.5 -> 22.9,
  v1.1 plan 37.5 42.8 -> 35.2, v1.1 plan 29 32.0 -> 26.0 deg; successes sit at 23-26 post-squat, the failure at 35.
  Expressing the plan in post-squat coordinates is the better next step than a radial backshift.
- **PLANNED STAND AZIMUTH 37.5 -> 29 deg: 5/10 on v1.1, and the failure modes it predicted are gone (session cf,
  2026-09-11 15:12-15:24 UTC). NOTHING RUNNING on instance 0; GPU0 2 MiB. Another session holds instance 1 / GPU1.**
  Ledger bottle.md §26. New knob `ReachConfig.stand_azimuth_deg` (default 0 = unchanged, so all 7 repo profiles are
  untouched) rotates the planned stand at constant plan_r. NOT reusing the dead `azimuth_deg`: every profile sets
  it to 42.0, which is inside the failure band.
  **The first run was VOID, and the reason is worth keeping:** with `--cases` the planned stand comes from the BANK
  (round_loop.py:99 reads can_xy/stand/rel straight out of it), so `sample_stand` -- and the new field -- is never
  called. Config written, loaded, validated, not read on that path. The corrected run derives a bank from the SAME
  ten cases with only `rel` rotated and `stand` recomputed (every other field byte-identical, checked); the planned
  stand moves 4.8 cm.
      arm                  plan   achieved az   depth   lateral   park    ok
      release              37.5      28.5       0.306    0.167    0.349   4/10
      v1.1 + seat fix      37.5      42.8       0.297    0.272    0.400   1/10
      v1.1 (the void run)  37.5      41.4       0.298    0.257    0.388   1/10
      v1.1 plan 29         29.0      32.0       0.299    0.189    0.350   5/10
  The achieved azimuth follows the plan (-10.8 deg for -8.5 of plan, the same +3 bias §25 measured); lateral and
  park land on release's values while DEPTH does not move (the control); `stand infeasible` 4 -> 2, `local
  alignment outside comfortable IK` 2 -> 0, `coupled safe-hand IK did not track` 1 -> 0. The one new failure is
  downstream progress (`[hold] object slipped ...` -- it had already grasped). Paired McNemar 4:0 vs the seat-fix
  run (p=0.125), 4:0 vs the void run, 2:1 vs release (p=1.000). Ten rounds still cannot measure a RATE (§23.3):
  the evidence is the telemetry, which moved exactly as §25 predicted.
  **WHY RELEASE ALWAYS WORKED:** the FK template's comfortable centre is at 30.3 deg; `backshift = 0.08` is applied
  along DEPTH ONLY, which rotates the plan to 37.5; release's dead lateral channel never executed that rotation and
  it landed at 28.5. v1.1 executes it. So this is a long-standing PLAN-geometry error that release's execution
  defect was hiding -- not a v1.1 defect.
  **Cleaner long-term fix, NOT done (one variable at a time):** apply `backshift` RADIALLY, which returns the plan
  to the template's own 30.3 deg with no extra knob, at the cost of plan_r 0.324 -> 0.310.
  Artifacts outputs/experiments/stagee_az29{,b}_20260911/. Branch `v11-recalibration-20260911`.
- **THE LATERAL RESIDUAL IS SOLVED, AND IT IS A PLAN problem, not an execution one (session cf, 2026-09-11
  14:19-15:04 UTC). NOTHING RUNNING on instance 0; GPU0 2 MiB. NOTE another session took instance 1 / GPU1 at
  15:00 (deploy on port 5566, GPU1 2948 MiB) -- not mine, and my sweeps ran on instance 0 / GPU0.**
  Ledger bottle.md §25. Committed work is on branch `v11-recalibration-20260911` (789f696 seat fix, aae3d72 the
  v1.1 line); main is untouched, merge with `git merge --ff-only` when you want it.
  **(1) The crab measurement refutes "v1.1's lateral channel is weak".** diag_shuffle gained `--crab` (the angle
  between the movement vector and the FACING; the robot still travels along the cleared heading, so a crab leg
  keeps a forward leg's obstacle clearance) plus fwd/lat/travel_err/yaw_turned -- the first pass recorded only the
  displacement MAGNITUDE and read 90 deg as FASTER than forward, which is only possible if the robot reorients.
  It does not: at 0.40 both checkpoints move sideways 0.17-0.29 m/s within 9-18 deg of the commanded direction
  while turning only 11-15 deg. **Lateral capability is a function of the SPEED COMMAND, not the checkpoint**:
  at 0.30 both deliver 0.002-0.046 m/s, at 0.40 both deliver ~0.25. Stage B moved v1.1's COARSE to 0.40, which is
  where that channel switches on.
  **(2) So release never executed the planned stand.** The visible expert's stand is a function of grip height
  alone (sample_stand's visible branch does not draw), so all ten cases plan the SAME azimuth: **37.4 deg**,
  plan_r 0.324. Achieved minus planned: release **-8.9 deg** (lands at 31.1), v1.1 coarse 0.30 +5.2, coarse 0.40
  +2.5, +seat fix **+5.4** (lands at 42.8, spread 36-46). v1.1 executes the plan; release undershoots it by 9 deg
  because at COARSE 0.30 its lateral channel delivers ~0.01 m/s and the steering's lateral component is simply
  not executed.
  **(3) And the downstream is tuned to what release DELIVERED, not to the plan.** On release's own 20 rounds the
  achieved azimuth predicts the outcome: [0,30) **5/7 ok**, [30,38) 3/6, [38,50) **1/6**; successes median 28.5
  deg, failures 35.5. The v1.1 arms agree ([30,38) 2/2, [38,50) 1/6 and 1/7). **The planned 37.4 deg sits on the
  edge of the failure band** -- the line works because release drags it back to ~29.
  **RETRACTED mid-diagnosis:** I said "the config asks for 42 deg". `ReachConfig.azimuth_deg` (42.0 in the
  profile) has NO consumer -- declared and validated in reach_config.py, read nowhere in src/locomani. The stand
  azimuth comes from `templates()`. A declared-validated-unread field is a hole lint_env cannot see.
  **NEXT is a DESIGN DECISION, not a calibration, so it is not done:** move the PLANNED stand azimuth to the
  line's real operating point (~29 deg = about 5.5 cm of lateral at depth 0.257). There is no knob today --
  sample_stand's visible branch subtracts `backshift` along DEPTH only, and raising it pushes plan_r past
  post_max 0.40; the approach's `appr_lat` is exactly the lever but was retired to 0.0 in refactor step 2.
  It changes the collected arrival distribution, so it wants your call.
  Artifacts outputs/diagnostics/gait_lateral{,2}_20260911/.
- **RESET SEATING BUG FOUND AND FIXED -- REPO-WIDE, NOT v1.1 (session cf, 2026-09-11 14:00-14:11 UTC). NOTHING
  RUNNING; GPUs 2 MiB, ports free.** Ledger bottle.md §24.
  `privileged_pub._seat_on_floor()` used `geom_rbound` -- the bounding-SPHERE radius -- as each geom's downward
  extent. Measured on the composed scene at the controller's nominal stance: the estimate puts the lowest point at
  -0.0994 m so it lifts the robot by that, seating the pelvis at **0.8924**, while the TRUE lowest surface point is
  +0.0084 (seated 0.7846). The deciding foot mesh has rbound 0.1291 against a real downward extent of 0.0213.
  0.8924 matches the z0 = 0.892 recorded in ALL 62 rounds; 0.7846 matches the observed settled 0.785. So **every
  reset in every task line was a 10.8 cm free fall**, and the "walked 0.5 m off the spawn while standing up"
  failures are its LANDING TRANSIENT, not walking: the whole displacement happens in the first second while the
  command is IDLE with a zero movement vector (sidecars record the command per tick).
  FIXED: true lowest point per geom type (mesh vertices; analytic for sphere/capsule/box/cylinder; rbound as a
  conservative fallback), seated with a 2 mm clearance. tests/test_seat_on_floor.py, 4 cases.
  **ACCEPTANCE on its own statistic (not a success rate, §23.3): first-second displacement after the reset** --
  before: release median 0.071 / max 0.211 (0/20 over the 0.35 m budget), v1.1 pooled over 42 rounds median 0.099
  / max 0.489 (4/42 over, 1 fall). After, 10 rounds: **teleport z 0.787, median 0.034, p90 0.071, max 0.083, 0/10
  over budget, no falls, and zero `[stand]`-stage failures** -- tighter than release was BEFORE the fix.
  **THIS CHANGES THE INITIAL CONDITION OF EVERY EPISODE IN EVERY LINE.** Every dataset collected before today
  started from a 10.8 cm drop. The crate and pick_place lines share this reset path; adopting it there is the
  user's call, the code is already shared.
  **WITH THE LANDING NOISE GONE, THE RESIDUAL IS 100% LATERAL:** park decomposed as park^2 = depth^2 + lateral^2
  (medians) -- release 0.349 = 0.306 depth + 0.167 lateral; v1.1 coarse 0.30: 0.467 = 0.332 + 0.350; v1.1 coarse
  0.40: 0.393 = 0.300 + 0.250; v1.1 + seat fix: 0.400 = 0.297 + **0.272**. Depth is SOLVED (0.297 vs 0.306);
  the lateral error is **+10.5 cm** against release with barely overlapping distributions (v1.1 min 0.219 vs
  release max 0.250). This is §17's "11 cm further sideways", untouched by both fixes -- the seat fix only made it
  deterministic. **So "Stage C is not needed" (§22/§23) is CORRECTED**: the brake DEPTH needs nothing, but the
  10.5 cm lateral is now the direct source of `stand infeasible` x4 and `local alignment outside comfortable IK`
  x2 of 10.
  **NEXT: measure the LATERAL gait delivery.** The approach's last 0.3 m is a crab walk (`mv` aims at the planned
  stand while `facing` stays on the approach heading, and the deploy normalises the movement vector's magnitude
  away), and §21's speed map measured FORWARD walking only. Same protocol with a pure-lateral movement vector,
  both checkpoints -- if v1.1 delivers less lateral per command than forward, that is the 10.5 cm.
  Artifacts outputs/experiments/staged_seat_20260911/. NOT COMMITTED YET.
- **GRASP-PHASE DIAGNOSIS (session cf, 2026-09-11 13:45-13:56 UTC). NOTHING RUNNING; GPUs 2 MiB, ports free.**
  Ledger bottle.md §23. `coupled safe-hand IK did not track` is a ROTATION residual, not reach: instrumented the
  block (it recorded only a tick count before) and got pos_err 0.0088-0.0106 m against a 0.08 tolerance -- 9x
  INSIDE -- while rot_err ran 0.2013/0.2208/0.2323 rad against a 0.20 tolerance, at phase 0.94-0.99, the bottom of
  the descent. **Mechanism:** the transport holds the palm's WORLD attitude while the pelvis pitches through the
  crouch (the target is recomputed each tick as `br.T @ target_rotation` against the live pelvis rotation), so the
  pelvis's own pitch becomes a rotation demand on the arm. Measured from the pose sidecars at the deepest squat:
  release +9.8/+11.1/+11.8/+11.4 deg (peaks to +14.5), v1.1 +12.1/+12.5 (peak +14.4) -- i.e.
  `TRANSPORT_ROTATION_TOL = 0.20 rad = 11.5 deg` is SMALLER THAN THE PITCH THE SQUAT DELIVERS, both checkpoints sit
  on it and v1.1's extra ~1 deg crosses it. Both TOLs were bare literals with no provenance.
  **RETRACTED mid-diagnosis:** `pregrasp_ahead_of_pelvis_m` separates the §22 rounds perfectly (OK <= -0.044,
  blocked >= -0.080) and I read it as "the target folds behind the shoulder, out of reach" -- the position residual
  refutes that; it is a proxy correlated with wrist extremity, not the cause.
  **CHANGED:** TRANSPORT_ROTATION_TOL 0.20 -> 0.35 rad (covers the measured 14.4 deg peak with margin), derivation
  in the comment. Chosen over holding the attitude in the pelvis frame because the tolerance only decides whether
  to GIVE UP -- it changes no command when the IK succeeds, while a pelvis-frame attitude would change the palm's
  motion and therefore the recorded data. `screen_motion` + `reference_check` still guard the object, and the
  alignment stage servos the attitude to ~0.03 rad anyway.
  **THE VERIFICATION DOES NOT STAND, and this is the bigger finding:** the same ten cases scored 0/10, but all 8
  failures were UPSTREAM of the tolerance (4 `stand infeasible`, 4 `[stand]`: teleport 0.10 m off, did not
  stabilise at pelvis z 0.085, walked 0.55 m off the spawn, could not turn -24 deg). Rounds REACHING the coupled
  squat across three identical-config runs: **6/10, 2/3, 2/10**. So a 10-round pilot cannot measure this line's
  success rate on v1.1 at all -- §22's "3/10 vs 4/10 is unresolvable" must be strengthened to "two runs of the
  SAME config are unresolvable", and neither 3/10 nor 0/10 should be quoted as a rate. The tolerance's intended
  effect IS visible per round (tol 0.20: blocked_ticks [0,0,0,5,11,11]+[2,5], two 11-tick aborts; tol 0.35: [2,3],
  no abort, no topple and no `path blocked`) but n=2 -- keep the change, marked unverified.
  **NEXT, in this order:** (1) fix the UPSTREAM stand variance first -- it is now the dominant term and "walked
  0.5 m off the spawn while standing up" is v1.1-specific; nothing downstream is measurable until it settles;
  (2) then settle the tolerance with a POOLED blocked_ticks distribution (n>=10 rounds that reach the squat, ~30
  rounds or only the cases that reach it), not a success rate. Unverified protocol suspicion worth controlling
  for: three stacks inside 20 minutes with `restart_deploy: false`, and the 0/10 run opened with teleport/stabilise
  failures -- leave a cooldown between runs or restart the deploy per round, and log a CPU gate.
  Artifacts outputs/experiments/{stageb_ikblock,stagec_rottol}_20260911/. NOT COMMITTED YET.
- **STAGE B DONE: APPR_COARSE 0.30 -> 0.40 on v1.1, and the arrival is fixed (session cf, 2026-09-11 13:36-13:41
  UTC). NOTHING RUNNING; GPUs 2 MiB, ports free, instance 0 free.** Ledger bottle.md §22.
  New module `src/locomani/backends/sonic/gait_speeds.py`: the approach's (coarse, mid, shuffle) commands now come
  from §21.1's MEASURED table keyed by checkpoint -- release keeps (0.30, 0.215, 0.20) byte-for-byte so every
  pre-switch dataset stays reproducible, v1.1 gets (0.40, 0.215, 0.20). 0.40 not the inverse lookup's 0.41: 0.40
  is a measured point (0.187 m/s, +-8%, 92% of release's 0.203). An unmeasured checkpoint RAISES with the
  diag_shuffle command instead of falling back. MID/SHUFFLE deliberately untouched (v1.1's step jumps over the
  0.016-0.018 m/s they deliver on release) so the pilot reads one variable. The two dormant sites that carried the
  same literals now read the same table (`_creep_to_band`, no production caller; `recoverable_grasp_policy`,
  recovery_attempts 0) -- dormant is exactly why they would otherwise keep release-era numbers until someone
  turned them on. NAV_SPEED/NAV_SPEED_MIN and place_phase's 0.20 left alone, unmeasured targets.
  **PILOT, same ten cases as the 2026-09-10 A/B, three arms paired per case_id:**
  release (coarse 0.203 m/s) 4/10, v1.1 old (0.044) 0/10, **v1.1 new (0.187) 3/10**. Every telemetry acceptance
  from §21.4 passed: `stalled` approach exits **5/10 -> 0/10** (release 1/10); `arrival left the object out of
  reach` **x3 -> x0**; park median **0.467 -> 0.393** (release 0.349) with the spread down from [0.43,0.68] to
  [0.36,0.43]; brake/stop geometry now matches release (0.307/0.300 vs 0.309/0.306), **so Stage C does not need to
  move `visible_brake_depth`**; 10 rounds in 4 min 03 (24 s/round) with the far field back from ~13 s to ~3 s.
  Paired McNemar: 3:0 vs the old arm (p=0.250), 2:3 vs release (p=1.000) -- ten rounds cannot resolve 3 vs 4, so
  "matches release" rests on the telemetry, not on the rate. NOTE release's own brake->stop carry is 0.003 m; the
  old "5-18 cm of momentum slide" comments are from the superseded settle-loop design.
  **THE BINDING CONSTRAINT MOVED TO THE GRASP PHASE:** the new arm's 7 failures are coupled safe-hand IK did not
  track x2, stand infeasible x2, local alignment timed out x1, "could not turn -24 deg" x1 (release fails the SAME
  case with the SAME string -- a case property, not the checkpoint), and "walked 0.54 m off the spawn while
  standing up" x1, which IS v1.1-specific and is a stand-up/heading-anchor behaviour, not a gait speed.
  Artifacts outputs/experiments/stageb_coarse_pilot_20260911/ + captioned reel
  outputs/videos/stageb_coarse_pilot_20260911/reel.mp4; scripts stageb_{coarse_pilot,report,reel}_20260911.*.
  Suite green, ruff/lint clean. NOT COMMITTED YET.
- **STAGE A OF THE v1.1 RE-CALIBRATION IS DONE (session cf, 2026-09-11 12:50-13:27 UTC). NOTHING RUNNING;
  GPUs 2 MiB, ports free, instance 0 free.** Ledger bottle.md §21. Measurement only -- no control constant changed.
  Restored `diag_shuffle.py` (deleted by refactor step 2 ec0eb1c) as `src/locomani/backends/sonic/diag_shuffle.py`,
  ported the imports, added an IDLE control leg (without it the carry column cannot be read) and made a fall cost
  one leg instead of the whole grid. Both checkpoints swept in one session, 11 speeds x 3 repeats + IDLE controls.
  **THE RESULT: both curves are STEPS, and the step is displaced by +0.065 of command.** release jumps
  0.016 -> 0.123 m/s between 0.215 and 0.26 (7.9x); v1.1 jumps 0.007 -> 0.044 between 0.28 and 0.30 (6.6x).
  Below the step the two are the same order (release 0.013-0.018, v1.1 0.007-0.014); above it v1.1 is 28-68% of
  release. So `APPR_COARSE=0.30` sits above release's step (0.203 m/s) and barely above v1.1's (0.044) -- 4.6x
  slower -- which is the one badly wrong constant. release's own 0.215 does NOT reproduce its documented 0.040
  (measured 0.016, three legs 0.007/0.016/0.039): it is sitting ON release's step edge, which is why the
  documented MID band was never stable.
  **IN-PATH BUDGET (depth 1.2 -> brake 0.31; do_approach has 45 s):** release 2.9+12.2+7.2 = 22.2 s; v1.1 as-is
  13.1+14.0+10.8 = **37.8 s** of pure depth-closing, leaving 7 s for the 11 cm lateral correction, the arm rise
  and any hesitation, with only 4.5x / 2.9x margin on the stall rule (1.5 cm in 5 s = 0.003 m/s) in the mid and
  decel phases. With COARSE re-picked to 0.41 it is 27.6 s. That is the mechanism of §17's 0/20, and it produces
  exactly the failure string §17 recorded ("arrival left the object out of reach", x5) because the visible profile
  runs ZERO forward-only repair walks.
  **TWO MECHANISMS I GAVE EARLIER TODAY ARE RETRACTED** (both in §21.3): (1) "the creep's 8 s budget times out" --
  `stream_reach._creep_to_band` has NO caller in the production path (tests call it directly; the comment at
  round_phases_approach.py:77 points at dead code); (2) "open-floor net speed refutes the stall rule" -- net speed
  and the approach's progress are not the same quantity, the loop scores only the DEPTH component.
  **RE-PICK for Stage B:** `APPR_COARSE` 0.30 -> 0.41 (delivers 0.187, +-8% over repeats). `APPR_MID` /
  `APPR_SHUFFLE` CANNOT be matched -- v1.1's step skips 0.016-0.044 entirely, so keep them or set both 0.215;
  never interpolate into 0.28-0.30, that is the bistable step interior. Consequence to carry into Stage E:
  v1.1's approach dynamic range is 3x (0.067-0.203) against release's 11x, i.e. it has no equivalent slow crawl,
  and memory policy-stop-point-is-set-by-the-crawl says that crawl is what sets the learned stop point -- this is
  a DATA-SHAPE change, not only a control change. carry is 1.9-3.5 cm on v1.1 in the bands that matter
  (release 12-17 cm at 0.40/0.45), so Stage C's `visible_brake_depth` should need only a small correction.
  Artifacts outputs/diagnostics/gait_speed_v11_20260911{,_knee}/ (legs.json per arm + report.json); scripts
  gait_speed_map_20260911.sh / gait_speed_report_20260911.py. Provenance bug found and fixed in §21.5: the first
  runner passed SONIC_POLICY as a command prefix, so the release arm's legs.json labelled itself sonic_v1_1 --
  the MEASUREMENT is unaffected (stack.log and deploy.log both show policy/release) and the report now takes
  stack.log as authoritative and prints the disagreement. NOT COMMITTED YET.
- **SONIC DEFAULT CHECKPOINT IS NOW policy/sonic_v1_1 (user ruling 2026-09-11, session cf). NOTHING RUNNING;
  GPUs 2 MiB, ports free, stack down. No GPU work was needed for this and none was done.**
  One default, two halves: `scripts/paths.sh` exports `SONIC_POLICY` (default `policy/sonic_v1_1`) and
  `locomani.paths.SONIC_POLICY` / `SONIC_POLICY_DIR` / `sonic_policy_file()` is the python side;
  start_stack.sh and restart_deploy.sh carry the same value as a fallback and now FAIL LOUDLY if that policy
  directory is missing from the deploy tree they run in (SIMPLE's own lockstep tree has policy/release only).
  `sonic_encoder.py` no longer hard-codes release's block table: `EncoderLayout.from_config()` reads the
  checkpoint's own observation_config.yaml, so v1.1 is 1751 inputs with the anchor block
  `motion_anchor_orientation_heading_10frame_step5` @584 computed in **orientation_mode 1** (left-multiply by
  the robot's HEADING quaternion, not its full base quat) against release's 1762 / @601 / mode 0. The decoder
  side needed no branch -- the 994-dim policy observation is block-for-block identical between the two.
  Offline decoders (`sonic_decoder`, `eval_open_loop`, `rtc/analyze_transition_capture`, `rtc/probe_prefix_models`)
  follow SONIC_POLICY. `config_snapshot.json` now records `sonic_policy` (name, dir, obs-config sha256, ONNX sizes).
  **WHAT THIS INVALIDATES, and it is not small:** a token is a CHECKPOINT-PRIVATE latent. Both encoders share the
  codebook exactly (1/16 grid, saturating at [-1.0, 0.9375], measured 2026-09-11), so every "is this token legal"
  check passes across checkpoints -- only the MEANING changes. Every dataset collected before today and every
  GR00T checkpoint trained on one is release-space: deploying those on a v1.1 deploy is silently meaningless, and
  so is decoding their tokens with v1.1's decoder. Reproducing an old number means `SONIC_POLICY=policy/release`
  for the whole chain.
  **NOT RE-CALIBRATED, so no v1.1 grasp number means anything yet:** release walks 17 cm/step at speed 0.30, v1.1
  4.3 cm -- and on one fixed case bank v1.1 scored 0/20 against release's 9/20 (McNemar p=0.0039, ledger §17),
  mechanism arrival geometry (parks 12 cm further out, 11 cm further sideways, approach "stalled" 10/20). The
  approach/brake timings, `primitives_tuning.TRACK_ROT_BOOST=4.5` (kept, NOT blind-reset to 1.0) and the
  reach_config stand calibration are all release-era. Re-measure the timings FIRST. The reason to switch is real
  too: v1.1 tracks the waist 3.3x better (waist RMSE 3.9 vs 12.9 deg, §16) and is a self-consistent model+config
  pair, while policy/release/observation_config.yaml is a hand-edited third variant.
  Verified: `sonic_encoder --selfcheck` PASSES on BOTH checkpoints; unit suite green (exit 0, no failures --
  note test_pose_feedback, listed below as a pre-existing failure, passes now); ruff/lint_env/lint_sizes add no
  new debt. `make check` is still red ONLY on `scripts/experiments/scalebfm200_reel_20260910.py` (untracked,
  another session's, ruff-format, red before this work). Ledger bottle.md §20 has the full record and the
  re-calibration list; HANDOFF.md §1b is marked superseded. NOT COMMITTED YET.
- **IMAGE-SERVO LINE, session cf 2026-09-11 08:1x-12:3x. NOTHING RUNNING; GPUs 2 MiB, ports free, stack down.**
  Commits c02354c -> f330238. Ledger bottle.md §19.7-19.11. Codex's design (§19.1-19.6) is intact and now committed;
  its tests are at 12 and green. Picked up its work after it stopped at attempt08 (8 attempts x 4 rounds, 0 success).
  **STILL 0 SUCCESSES**, but four confounds are gone and the remaining blocker is one measured geometry problem.
  **WHAT WAS FIXED**
  1. **A 0.545 m perception error.** Auditing 622 accepted fits against truth: fused ego+wrist is calibrated
     (0.017 reported / 0.018 true), but wrist-ONLY fits reported 0.076 and were off by 0.545 m, 98% pure depth,
     and passed the 0.15 gate. Cause: on a disagreeing pair `fit_bottle` fell back to the best-CONDITIONED single
     view = the larger silhouette = the wrist, the camera its own comment blames. Now keeps the head view.
     (A width-sensitivity uncertainty term was tried and REVERTED: a confidently wrong monocular depth is
     self-consistent and it still admitted 98% of them. Only cross-view agreement separates them.)
  2. **A latent pirouette that perception fix exposed.** Far-field facing aimed at `atan2(target[1], target[0])`,
     the bearing to the BOTTLE, but the stand is deliberately offset `approach_lateral=0.195`, so that bearing
     converges to -0.195 and never to zero -- a standing ~10 deg right turn every tick. attempt08 never reached it
     because the wrong-near depth kept it in the other branch. attempt10 swept 2738 deg of heading inside a 3-8 cm
     box. Now aims at the approach goal, whose bearing does converge.
  3. **The geometry-derived finger clearance.** Measured off the MJCF: fingers reach 0.1941 m from the wrist, the
     grip point (OFFSET) is at 0.1515 m, so they extend **0.0426 m past the grip**; required stand-off =
     finger reach past grip + object radius + margin = 0.100 m for this bottle, against the fixed entry_gap 0.075.
     Now derived from geometry + the object's own radius; `grip_standoff_m` logged per frame.
  **THE PRIVILEGED-TEACHER PATTERN (user's idea, §19.9), implemented behind `reach.image_servo_truth_control`
  (default False, Codex's RGB-controlled path untouched):** the expert CONTROLS from object truth so it is reliable,
  and the SAME `decide()` is evaluated a second time on the RGB fit every frame, recording `intent_rgb` / `agree` /
  `aim_delta_m` / `rgb_usable`. Truth never enters the recorded observation; it selects the action and the audit
  certifies reproducibility. **Criterion: Var(action | observation) ~ 0, not "no privileged info".**
  **PERCEPTION BUDGET (1562 fused fits vs truth):** 0.25-0.40 m -> 0.009 median / 0.015 p90; 0.40-0.60 ->
  0.005/0.016; 0.60-0.90 -> 0.013/0.041; 0.90-1.40 -> 0.028/0.084. A switch is only admissible where its tolerance
  greatly exceeds the p90 at the range it fires: "when to close" (0.25-0.40 m) can be one, **"when to raise the arm"
  (0.6-1.2 m, p90 41-84 mm) cannot** and must be continuous.
  **AGREEMENT RATE (attempt12_truth, 143 frames): 80.4% overall, 82.5% of frames had a usable fit, 97.5% agreement
  when usable. By range: 0.25-0.40 m 100%, 0.90-1.50 m 49%.** Of 28 disagreements, 25 are "RGB not usable at all"
  and only 3 are a different intent -- all 3 on the `prealign_depth=0.44` boundary. So the near-field grasp
  decisions ARE reproducible; everything unlearnable is far field and is mostly "could not see".
  **WHAT IS STILL BROKEN -- one thing.** The dex1 long fingers foul the bottle. Per-round minimum grip stand-off:
  r1 +0.050, r2 +0.057 (both swept; the geometric limit is 0.0653, and with fingers 0.0426 past the grip that puts
  the tips 0.007 and 0.014 m from the bottle centre, inside its 0.0226 m radius -- exactly what the model predicts),
  r3 -0.065 (the 62 s timeout, arm hooded over the bottle).
  **Base-speed coupling is NOT the lever**: the entry axis is near vertical (near field fwd +0.180, **up -0.860**),
  so the clearance is spent by the ARM's own descent. `clear_entry` fired on 335 of 522 frames trying to lift and
  still lost. The fix is to make the clearance a PROJECTION on the commanded aim rather than a reactive back-off.
  **Acceptance is already instrumented**: per-frame minimum `grip_standoff_m` must stop falling below 0.0653 m --
  no need to wait for a success rate.
  **TWO CORRECTIONS I OWE**, both caught by the user asking: §19.10 claimed the stand-off was negative and the
  geometry model refuted -- that was an aggregate weighted by r3's 62 s dwell; per round the model predicts the
  fouling correctly (§19.11). Only the lever was refuted, not the model.
  **ARTIFACTS:** attempt09/10/11/12_truth/13_clearance under outputs/experiments/image_servo_20260911/;
  captioned videos with the failure string and each round's minimum stand-off in
  outputs/videos/image_servo_attempt13_clearance/{r0..r3,reel}.mp4 (r3 shows the hooded pose that blocks its own
  wrist camera). Exporter scripts/experiments/image_servo_export_videos_20260911.py.
- [STALE, left for Codex to rewrite -- its RUNNING line below refers to attempt08, which finished at 08:08 UTC;
  the machine has been free since, and session cf continued the line through attempt13. See the IMAGE-SERVO
  entry at the top.]
- **Codex 2026-09-11: 单臂视觉反馈专家正在实现;用户已授权采集 1 集成功视频。**
  RUNNING GPU0 / instance 0: outputs/experiments/image_servo_20260911/attempt08/launch.sh。
  新 RGB cap 定位 + 无阶段历史的纠偏控制正在烟测,最多 4 轮、1 集成功即停,退出按实例锁归属清理。
  本次不训练 VLA、不做 DAgger;保留其他会话的 ScaleBFM 未提交修改。
- **Codex 2026-09-11: 单臂专家设计建议已记录于 docs/ledger/bottle.md §19;实施与烟测见上一条。**
  用户询问如何按视觉线索重设计;只审阅代码与历史实验,本轮未启动 GPU/stack/采集/训练。
  建议以当前视觉夹取误差驱动局部纠偏,直接覆盖桌边静止/部分伸手的原生初始状态,遵守 NO DAgger。
  对 Next 0a 的补充:仅令 arm_pose=f(base_distance) 仍可能卡住;“从第一帧手臂一直动/腕高单调”
  不能代替安全的任务进展。检测器须区分恒定目标下的实际收敛、无效抖动、远场保持与近场死锁;
  无平台期不保证训练后无固定点。§19 含具体观测接口、四类 pilot 覆盖、反例与双 seed 闭环验收。
- **SESSION cf 2026-09-11: the visible200 2/100 is DIAGNOSED. NOTHING RUNNING; both GPUs at 2 MiB, ports free,
  no stack, instance 0 free.** Commits bf5bfd6 -> ee70bcc. Full record in ledger bottle.md §18.2-18.9.
  **THE ANSWER.** The policy walks to the table, adopts the expert's approach posture to within ~2 deg and ~5 mm,
  and then FREEZES for 85 s (shoulder pitch varies 0.7 deg, wrist ~2 mm). The gripper never closes because there is
  nothing between the jaws -- that decision was always correct. The miss is almost entirely VERTICAL: successes sit
  +0.029 m above the bottle, failures +0.191 m, and failures are horizontally CLOSER (0.108 vs 0.134 m). Min
  palm-to-bottle separates the 99 rounds PERFECTLY (successes <= 0.138 m, all 97 failures >= 0.156 m, zero overlap)
  while base-to-bottle overlaps completely -- measure the PALM, not the base. Cleanest discriminator: WRIST ROLL,
  expert 0.03 -> 0.24 rad at the wind-up, live 0.018 all round.
  **RULED OUT, each with numbers, not argument:** observation encoding (§18.3); the model refusing to close (912
  live queries never left the +0.0245 open rail); the execution chain (the arm DOES track the token, r +0.54..+0.72
  zero-lag, and commanded/achieved motion scale together, §18.7); stop position and azimuth (§18.2); distribution
  shift (the stuck state is nearer to training than training frames are to each other, and 100% matches the
  approach phase, §18.8); missing velocity (a LINEAR probe on the leg angles the policy already gets predicts
  motion at R^2 0.665 and moving-vs-still at 90%, §18.9).
  **WHAT IT IS.** A behaviour-cloning fixed point in a fully in-distribution state, created by the expert's
  STRUCTURE: through the whole approach the ARM IS A PASSENGER (wrist z flat at 0.111 for 100+ frames, roll flat at
  0.03) and all task progress comes from the BASE; then a sharp switch at close-60. When the policy parks the base
  (measured 0.79 cm/s), the only source of progress is gone, "hold" is the correct action on the plateau, the
  observation stops changing, and the same action repeats for 85 s.
  **RETRACTED along the way (both after the user challenged them):** "the right wrist camera triggers the descent"
  (§18.5 -- single-shot reads of a statistic whose sampling sd is 0.002-0.017); "the policy cannot tell walking from
  parked" (§18.9 -- probe says it can). Two method rules now in the ledger: repeated sampling + an sd before
  believing any token-nearest-neighbour difference; and a linear probe before claiming information is ABSENT.
  **CODE FIXED (all committed, make test green):**
  (1) trace files were named from the VLA client's own call counter, so `traces/round_NNN.json` described manifest
      round NNN+1 and every analysis joining them by round number was reading mismatched pairs. `round_id` now
      travels round_loop -> run_round -> _run_round -> RoundContext -> the three clients, and `round` is a field in
      the JSON. tests/test_trace_round_join.py (12 assertions, all red pre-fix). Verified on live data: offset 0.
  (2) `--set backend.obs_dump=DIR` wrote NOTHING -- its only consumer was stack.py exporting TF_OBS_DUMP into the
      STACK's env, while the dump is gated by `vla_round.OBS_DUMP`, a module global in the eval-client process that
      nothing assigned. `grip_delay`, `replace`, `lockstep.no_seed` were inert the same way (their config defaults
      equal the module defaults, so NO past result is affected). All four wired;
      tests/test_eval_config_reaches_the_client.py names all four on the pre-fix HEAD. NOTE: a "is the field name
      referenced" check does NOT catch this class -- obs_dump was referenced, in the wrong process.
  **NEW TOOLS:** `src/locomani/pipeline/replay_obs.py` -- ask the policy offline what it would have done, no sim, no
  stack, seconds not hours (`--dump` replays OBS_DUMP npz, `--dataset --near-close` rebuilds observations from a
  LeRobot episode in the eval's obs format, `--substitute KEY --donor-dump` bisects the train/deploy seam,
  `--expect-close` makes the exit code the verdict). Needs serve_policy on :5590 and the Isaac-GR00T venv.
  `src/locomani/pipeline/dataset_ambiguity.py` -- written as a design gate but it FAILED its own acceptance test
  (the wind-up scores the LOWEST ambiguity), so it is UNFIT for this failure and must not gate anything.
  **ARTIFACTS:** outputs/experiments/visible200_obsdump_20260911/ (912 npz / 580 MB in obs/, 4 traces, 5 videos,
  wrist_live_vs_dataset.png, wrist_series_live_vs_expert.png, trigger_controls.json, reach_trigger.json);
  outputs/experiments/visible200_20260910/eval/trace_join_20260911.json; ambiguity_train.json.
  The 580 MB obs/ dump can be deleted once the new expert design is settled.
- **USER RULING 2026-09-11: NO DAgger.** The goal is that natively collected data trains a good VLA on its own.
  The fix is therefore the EXPERT REDESIGN (see Next), not an iterative rollout-label-retrain loop.
- visible200 TRAIN + EVAL DONE (session cf, 2026-09-10 12:01-19:56 UTC, GPU0 released, instance 0 free).
  **[MECHANISM SUPERSEDED 2026-09-11 -- see the SESSION cf entry at the top and ledger §18.2-18.9. The 2/100
  and the failure strings below are correct; every causal claim in this entry and in §18/§18.1 has been
  re-derived or retracted.]**
  10k steps, loss 1.2866 -> 0.0625; closed loop **2/100** (seed 6000, async, clean protocol).
  97 of 100 failures are `[vla] at the table but never commanded a close`. Open loop on its own val is
  EXCELLENT: nmse_gripper 0.0024, close_frame_err_abs 2.3 frames, deg_mean at the noise floor. So the model
  learned the mapping and the closed loop puts it outside the training distribution: the never-closed stop
  distances are p10 0.18 / median 0.27 / p90 0.35 m against the expert's 0.30-0.31 brake band, only 20/96
  inside 0.29-0.32. Mechanism and ruling in docs/ledger/bottle.md §18. Not a seed-variance result (floor is
  57 vs 46) and not the GPU0 overlap (0.54 vs 0.62 yield cannot move 55 points).
  CORRECTION (ledger §18.1, after the user challenged the number): the mechanism in §18 is WRONG. Per-round
  traces show azimuth and distance do NOT separate success from failure (successes sit at 0.25 m / -39 deg,
  inside the failures' spread). The only discriminator is the gripper: 94/99 rounds the model never predicts a
  close at all (chunk_grip_min >= 0), 2 predict+execute and both succeed, 3 predict but it never reaches the
  executed stream. So it is not an execution bug and not a stop-position problem -- offline it predicts the
  close perfectly, in closed loop it almost never decides to. Not yet localised. The restart_deploy A/B meant
  to test the anchor mismatch scored 0/20, ALL [stand] collapses, so that suspect is untested, not excluded.
  Next: dump the observation at a failing decision moment and ask the model offline whether it closes there.
  FIXED on the way (23b5436): eval_closed_loop.py's src bootstrap reached src/locomani instead of src, so
  BOTH eval entry points had been broken since the 4a package move (29bdf84); it was the refactor follow-up
  this file listed as "not yet verified: the eval path".
- CONVENTION (user ruling, 2026-09-10): cross-session coordination goes in THIS FILE by default, not in
  cross-session messages. Send a message only for a blocking resource conflict. Rationale: message volume
  was too high today and every one costs a human approval; several expired undelivered anyway, so STATE is
  both cheaper and more reliable. Session 29 is following this from now on.
- DONE (session 8b, user-asked, 2026-09-10 15:34-18:08 UTC, GPU1 only, GPU1 now FREE): ScaleBFM / bottle_visual arm
  `scalebfm200_noretract_s17_20260910` -- GR00T seed-17 3000-step finetune on the no-retract 200-episode dataset
  (2.8 s/it, 2 h 21) then the registered 100-case closed loop (seeds 31001:31101, yplus_center_v1, no recovery
  perturbation, 13 min). **41/100** under the new rule: high_near 25/34, low_near 11/33, low_deep 5/33; failures are
  34 physics/fall + 25 timeout, and "knocked before close" no longer exists as a label.
  **The success rule changed in 2ba1a89, so 41 is not 25+16.** Re-scored from the same traces under the 2026-09-09
  rule this arm is **<=38/100** (upper bound: object_audit has no bottle velocity, so the old |v|<0.06 term cannot be
  reproduced and can only lower it). 24/100 knocked the bottle before closing but only 1 of them ended a success, so
  dropping that abort was worth ~3 points, not 16. Like-for-like the gain over ledger section 9 is ~13 points and it
  is **entirely high_near** (10/34 -> 24-25/34); low_near 10 -> 10-11 and low_deep 5 -> 4-5 did not move. Fisher
  two-sided 41 vs 25 p=0.024, 38 vs 25 p=0.067; section 9's per-case results were deleted in the incident so this can
  only be unpaired, and section 11 already ruled the old EGL baseline not bit-reproducible.
  Ledger scalebfm.md section 22; RESULTS.json + RESCORE.json in the experiment dir; reel
  outputs/videos/scalebfm200_s17_20260910/reel_10.mp4 (8 cases, 3 strata x ok/timeout/fall).
  This line runs a standalone in-process MuJoCo sim -- no SONIC deploy, no stack, no ports -- so instance 0 and GPU0
  were never touched. Checkpoint outputs/experiments/scalebfm200_noretract_s17_20260910/model/checkpoint-3000, 27 GB.
  FOUR CODE FIXES, all UNCOMMITTED (the tree already carries several sessions' work; eval.py was modified by another
  session before I touched it): (1) `pipeline/train.py` had `ROOT = parents[3]` (= src/locomani) after the package
  move, so base snapshots, modality config and the src/gr00t hashes all pointed at a directory that does not exist --
  now `locomani.paths.REPO`. (2) the frozen `base_model_snapshots.json` AND the default manifest for
  `verify_runtime_assets()` both died with `scalebfm_visual_goal_20260908/`: snapshots regenerated as
  `configs/scalebfm/base_model_snapshots.json` (same surviving HF cache, weight digests beside it, marked a rebuild)
  and `eval.py` gained `--runtime-assets` (mirrors collect.py), pointed at the collection's surviving manifest --
  103/103 assets verified unchanged; the eval scene is byte-identical to the collection's frozen copy.
  (3) `follow_video.py` now calls `mj_camlight` -- ledger section 10 recorded that darkening bug on 2026-09-09 and it
  was never fixed. Anyone restarting this line needs (1) and (2).
  Runner source scripts/experiments/scalebfm200_train_eval_20260910.sh (+ _rescore_ and _reel_ scripts).
- 2026-09-10 15:1x (session 29): session cf flagged that `--task-config` could re-calibrate a closed-loop eval.
  VERIFIED NOT to happen today -- no eval path passes --task-config (run_learn_pilot's eval passes an explicit
  --reach-config; eval_closed_loop.py has no such flag; run_eval_arm.sh passes neither, so it stays on
  DEFAULT_REACH). But cf is right that the reach block is NOT inert for a VLA round: sample_stand reads
  max_drop / backshift / ready_grasp_recovery, so it moves the sampled stand and the case distribution.
  _apply_task_config now withholds `reach` when VLA_POLICY is set and prints "skipped reach"; an eval that
  wants the collection calibration must pass --reach-config explicitly and record the protocol change.
  Test: tests/test_reach_config.py::CalibrationGateTests::test_a_task_config_never_recalibrates_a_vla_round.
- FIXED 2026-09-10 15:0x (session 29), SECOND regression of the same class, found by self-audit before it bit:
  `require_runnable` would have refused every CLOSED-LOOP EVAL that did not pass --reach-config (cf's
  visible200 eval at ~15:45 included). A VLA round returns from `_p_handover` before `_p_approach`/`_p_grasp`,
  so `execute_grasp` never runs and the calibration is irrelevant; `eval_closed_loop.py` passes --reach-config
  only when one was asked for. Now `reach_config.require_expert_calibration(VLA_POLICY, ...)` gates it, with
  both branches tested (tests/test_reach_config.py::CalibrationGateTests). Verified end to end: expert run
  without a calibration still refused, VLA run proceeds. Eval and gate paths are unblocked; suite green.
- FIXED 2026-09-10 14:5x (session 29): `ports.require_exporter()` refused every run_learn_pilot / run_ready_pilot
  collect on a healthy stack. Those launchers set TF_EXPORTER_SCRIPT to run_instance_exporter.py while
  start_stack.sh's default is gear_sonic's run_data_exporter.py, and the probe matched only the default name
  (found by session 65 on a live gate run). ports.py now carries an explicit _EXPORTER_ENTRY_POINTS tuple with
  both, plus tests/test_instance_ports.py. Gate/eval runs are unblocked; suite green.
- FREE since 2026-09-10 14:10 UTC: **instance 0 and GPU1**, released by session 29. Verified at handback:
  no python run_sim_locomani / run_data_exporter, no g1_deploy, 5555/5556/5580/5581 with no LISTEN, GPU1 at 2 MiB.
  Session 65 was waiting on a direct ping for this and TWO cross-session messages expired unapproved, so this
  line is the notice. GPU0 remains session cf's training; cf takes instance 0 from ~15:45 for the visible200 eval.
- DONE (session 29, 2026-09-10 14:10 UTC): clean 20-round A/B, policy/release 9/20 vs policy/sonic_v1_1 0/20 on one
  fixed case bank (exact McNemar p=0.0039). Mechanism is arrival geometry, not grasping: v1_1 arrives 11 cm further
  sideways (t=-6.38) and parks 12 cm further out (0.498 vs 0.377, plan_r 0.32), and its approach ends "stalled" 10/20
  against release's 2/20. Do NOT switch the production checkpoint without re-calibrating the approach timings.
  Ledger bottle.md §17; artifacts outputs/experiments/v11_bottle_ab20_20260910/. Instance 0 handed back at 14:10.
- DONE (session 29): the four fixes for the config-partially-applied class that cost three stack cycles today --
  locomani.sim.ports (one port formula, INSTANCE-derived), reach_config.require_runnable, ports.require_exporter,
  and --task-config now carrying the whole profile; scripts/collect_pilot.sh un-broken (it produced zero episodes
  from refactor step 2 until now). make check passes except docs/handoff/3dgs_background_plan.md, which fails
  ruff format in HEAD too and is not mine. Not committed -- the tree carries several sessions' uncommitted work.
- RUNNING (session 29, user-asked, since 2026-09-10 13:26 UTC): paired 10-round bottle pilot on **instance 0** with
  CUDA_VISIBLE_DEVICES=1 / MUJOCO_EGL_DEVICE_ID=1 (all GPU work on GPU1; GPU0 stays with session cf's training).
  policy/sonic_v1_1 THEN policy/release over the SAME fixed case bank (outputs/experiments/v11_bottle_pilot_20260910/
  cases10.json, seed 7100, narrow spawn --spawn-edge 0.7 --face-object --spawn-yaw-err 20 --spawn-lateral 0.20,
  --edge-jitter 0.20, --stand-retries 0, --record, videos per arm). Purpose: how much error does switching the SONIC
  checkpoint cost on the real grasp line. Runner scripts/experiments/v11_bottle_pilot_i0_20260910.sh (scratch copy).
  Instance 0 ports 5555-5582 and the exporter are in use; session cf notified, they need instance 0 only after
  training (~16:30 UTC).
  GOTCHA found doing this: **instance 1 cannot record** -- start_stack prints "[3/3] data exporter skipped (instance 1
  records no episodes)", and the gym resolves its ports from the instance-0 defaults regardless of INSTANCE, so an
  instance-1 run dies with "no privileged state on :5581". The first attempt is kept as
  {sonic_v1_1,release}_portfail_i1/ -- not results.
- INCIDENT 2026-09-10 13:40:25 UTC (pick-place session, Claude): my bottle gate runner (outputs/experiments/
  gate_bottle_pickplace_20260910) admitted itself onto instance 0 / GPU1 on GPU-memory + CPU headroom alone, without
  re-reading the RUNNING lines above, and raced session 29's runner for the port block in the same second (their
  sonic_v1_1 arm had finished and its teardown freed the ports; their release arm's gym died with "Address already in
  use :5556"). Lost: one arm of theirs (~10 min, relaunched 13:41). My 10 rounds are all "teleport landed 1.85 m off"
  (two sims on one port block); gate.txt there is VOID, not a regression. Session 29 is landing an admission check
  that gates on the ports (not GPU memory); the gate for the pick_place changes still has to be re-run on a quiet
  instance (window: after 29's run ~13:55, before cf's eval ~15:45).
- pick-place session: NOTHING RUNNING (15:15 UTC); instance 0 / GPU1 free, lock free. Bottle gate for the pick_place
  changes PASSED 16/16 (7/10, outputs/experiments/gate_bottle_pickplace3_20260910; ledger pick_place §6).
  (inherited fd, self-releasing) and stop_instance.sh is scoped to its own instance's exporter (comm=python + TF_CAMERA_PORT
  in the environment); agreed with session 29, who drop their runner-side port loop. Untested on a live stack yet (instance 0
  busy) -- the first start after this is the test.
- UPDATE 2026-09-10 12:03 UTC (Codex): SONIC phase3 retry CPU gate timed out after10min;
  retry_reset/run.exit=1, runner.log="CPU gate timeout; no stack started"; PID2600051 exited.
  No retry action test occurred. Chrome SwiftShader PID2598614 from Claude3265590/Playwright
  measured14.76 CPU cores of15.36 quota; left untouched. Restore from HANDOFF.md, not older queued status.
- RUNNING (session cf): visible200_20260910 TRAINING on GPU0 since 2026-09-10 12:01 UTC. GR00T 10k steps on the 190/10
  dataset, 1.42 s/it, **ends ~16:00 UTC**; the runner then starts the 100-round closed-loop eval (seed 6000,
  configs/eval/closed_loop_100.json) on **instance 0 / GPU0**, ~3 h. Keep GPU0 free until this line is gone; instance 0
  is free until ~15:45 (session 29 has it until ~13:50, agreed).
  **HAZARD, cannot be patched:** the runner's eval gate checks only GPU0 memory + cpu_headroom, NOT whether instance 0
  is free -- another session working on GPU1 would pass that gate and get its ports taken. The executing copy is
  outputs/experiments/visible200_20260910/launch_train_eval.sh and editing a running bash script corrupts it
  (memory never-edit-a-running-bash-script), so the 15:45 hand-back time is the only guard today.
  Runner source scripts/experiments/visible200_train_eval_20260910.sh; logs in the experiment dir
  {train.log,pilot_train.log,pilot_eval.log,runner_train_eval.log}.
- SONIC phase3 native dynamic replay DONE (2026-09-10 12:15-12:33 UTC, session 29, GPU1/instance1). Two 3-repeat
  runs under outputs/diagnostics/sonic_native_phase3_20260910/: retry_reset_02 (policy/release) and v1_1
  (policy/sonic_v1_1). Both 3/3 valid, run.exit=0, no falls, 423/424 frames at exactly 50.00 fps (8.46 s).
  Startup protocol that made it valid: wait "Init Done" -> reset -> ']' -> 1 s settle (pelvis 0.890->0.786, ncon 33)
  -> 't'. The old single-sample failure stays in native_release_1.json as recorded.
  RESULT (mean of 3, waist/torso pitch at the reference's deepest-lean frame, ref torso 40.3 / ref waist 29.3):
    policy/release   torso 14.5  waist  8.1  waist RMSE 12.9  torso RMSE 12.4
    policy/sonic_v1_1 torso 25.3  waist 21.4  waist RMSE  3.9  torso RMSE  5.7
  Same scene, same PD, same protocol -> the checkpoint is the dominant factor, 3.3x on waist RMSE. Reports:
  native_report.json (frame index recovered by nearest-neighbour match, no assumed play rate) and
  channel_audit.json (29x29 command-vs-reference correlation) in each dir.
  Ruled out as causes, with numbers: reference->encoder 2.1e-7 rad; reference is self-consistent with the official
  MJCF (FK torso pitch reproduces body_quat to 0.01 deg, so "+waist = forward" holds for reference/official/our sim);
  deploy observations vs privileged truth <=1.2 deg (release) / <=2.3 deg (v1_1); last_action vs sim cmd_q <=0.05 rad;
  both configs pass the deploy's own dimension check (release 1762/994, v1_1 1751/994).
  Open, NOT a checkpoint difference: waist_pitch is the only channel whose command is anti-correlated with its own
  reference (-0.90 in all 6 runs, both checkpoints, ~-60 deg commanded where the reference asks +29.3). Legs/arms are
  +0.95..0.99. So the achieved lean is not driven by that command; waist kp is 28.5 (2*STIFFNESS_5020, armature*w^2
  from policy_parameters.hpp) against a measured gravity load of 12-23 Nm. Do not call this a per-checkpoint bug.
  Note: policy/release/observation_config.yaml is a third, hand-edited variant, equal to neither
  observation_config_sonic_release.yaml nor _sonic_v1_1.yaml (it adds motion_root_z_position*, +11 dims, and uses the
  non-heading-normalised anchor). policy/sonic_v1_1/ is a self-consistent model+config pair.
  Scripts: scripts/experiments/sonic_native_{replay,report,channel_audit}_20260910.py (replay is now parameterised by
  SONIC_NATIVE_POLICY / SONIC_NATIVE_VIDEO / SONIC_NATIVE_INIT_TIMEOUT). parity_report.json from the earlier pass still
  stands: LowCmd PD matches the C++ at CSV-rounding precision, 29 body joint params match the official local MJCF,
  robot mass Dex1=35.864952 vs Dex3=36.1652352 kg. No production policy, PD or expert changes.
  GPU1 released at 12:34 UTC: the optional release-vs-v1_1 video A/B was killed inside its CPU gate (no stack was ever
  created) when the pick-place session claimed GPU1 at 12:33. outputs/.../video_release/ is that aborted stub, not a
  result; a retry needs a new directory name.
- Video comparison DONE (2026-09-10 12:55 UTC, session 29, no simulation): reference vs release vs v1_1 as one
  side-by-side clip in outputs/diagnostics/sonic_native_phase3_20260910/compare/ (mp4 real-time + half speed,
  key_frames.png, 424 png frames). Offline kinematic replay of the recorded traces via
  scripts/experiments/sonic_native_render_20260910.py, rendered with EGL on GPU1 in 7.7 s wall / ~2.5 cores, so it
  did not need a stack and did not disturb the pick-place pilot. Root yaw and xy are removed so the three panels
  differ only in posture.
- Video v2 (2026-09-10 13:06 UTC): v1 had a RENDER bug -- the scene's 29 motor joints are not a contiguous qpos
  slice (left-gripper SLIDE joints sit between the arms), so a flat qpos[7:36] write put a shoulder angle in metres
  onto a gripper slide and threw the jaw 2 m off, and shifted the right arm by two slots. Legs and waist are before
  the corruption, so no measurement or conclusion changes. Fixed by indexing jnt_qposadr by NAME. v2 also adds an
  orange reference-outline ghost on both delivery panels and a waist/torso pitch curve strip with a playhead.
  EGL failed transiently while another session's stack was starting, so v2 was rendered with OSMesa (explicit
  diagnostic exception, ~4 min on free CPU); EGL was verified healthy again at 13:20. Ledger bottle.md 16.7/16.8.
- Waist bend ceiling (measured): waist_pitch range is +-0.52 rad = +-29.79 deg in EVERY G1 MJCF in the repo, and the
  official references are clipped to it -- squat_001__A359 sits at exactly 0.520000 rad for 118 of 424 frames (28%),
  and 11 of the 13 official motions peak at exactly 29.8 deg. The reference's 40 deg torso lean is pelvis 11 + waist
  29.8; a deeper lean has to come from hip and squat, not the waist.
  Commands/evidence/remaining comparisons: docs/ledger/bottle.md section15.
- MERGE PENDING (session cf): branch fix/first-round-anchor (round 1 of every batch failed at the spawn; verified 3/3,
  ledger bottle §14) waits for the uncommitted edits in main's tree (Codex: docs/collection.md, scene_lib.py, bottle_visual/*,
  ledger bottle §12 弯腰审计) to be committed -- it touches docs/collection.md too. Whoever commits those: ping cf or just
  `git merge --no-ff fix/first-round-anchor` (both-appended conflicts only, keep both).
- COMPLETE SONIC phase2:15/15 cases,5conditions x3,run.exit0,instance1 stopped. Results:
  outputs/diagnostics/sonic_lean_phase2_20260910/{summary.json,encoder_report.json,encoder_root_check.json}.
  Matched waist-only->+legs->+root->full pose actual torso means4.90/7.38/13.06/16.52deg;
  requested25.17/25.17/34.97/34.97deg. Actual encoder joint inputs maxerr5.89e-8rad;
  reconstructed encoder rootpitch matches request within0.05deg(all mode0). Input signal is present.
  Coordination improves bending but still undertracks; not proof G1/SONIC cannot bend. Next: dynamic
  official replay/model+PD parity, not more identical static tests. No production expert/model change.
- COMPLETE ScaleBFM200 collection/conversion/stats: RESULTS.json verified200episodes,49387frames,400videos,
  no training started. Dataset:outputs/experiments/scalebfm200_noretract_sonic_20260910/dataset.
  OSMesa34 complete cases retained plus166 EGL cases, one interruptedseed4035 skipped; rendering_history preserved.
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

## Changed (session cf, 2026-09-10 02:48 - 09-11 00:50 UTC, ALL COMMITTED AND PUSHED through 12554b2)

- **ScaleBFM line reorganised and merged** (a3e2fee, f8fc71b, 51a4d25, 0ad4b3e, merge f73d4ec) -- see the Now entry.
- **README rewritten** (b7881f1): it still pointed at src/teleop_free/start_stack.sh, docs/reference.md,
  docs/runbook.md and a 2026-08-27 status. Now the locomani layout, the four-kind doc map and a 2026-09-10 status.
  The five stray root step1/2/3/5/6.md (the scoop_check skill's notes, swept in by b41692a's `git add -A`) are one
  file under docs/archive/scoop_check_2026-08-26.md.
- **Codex's 7 review findings fixed** (f1f7452, merge 9af89f8, b26fcb0, 1bd6498; docs/ledger/refactor.md is his report):
  config_base now rejects unknown fields, mismatched schema_version and wrong scalar types, and enum fields declare
  metadata["choices"] so `--set client=asnyc` / `through=collec` are errors; the ScaleBFM provenance path was
  parents[4] pointing at a directory that does not exist, so **every ScaleBFM run's source_sha256 silently omitted the
  whole backend** (now parents[3], raises if incomplete, snapshots keep package-relative paths); scene.base_scene()
  makes `--set-backend scene_dir=` actually take effect; run_learn_pilot's config_snapshot.json carries `inputs`
  (frozen scene digests, initial token, reach_config) like the retired provenance.json did; silent_failures'
  TF_DATASET_FPS row, the scalebfm ledger's over-stated "cannot recover from the crouch", and tasks/base.py's
  over-claimed Task contract corrected. tests/test_review_2026_09_10.py reproduces each one.
- **sim_vis default made consistent** (c122a6d, 9156e88): 18dd6dd changed sim_vis.py's bare default to none but
  SonicBackendConfig.sim_vis stayed "full", and stack.py writes cfg.sim_vis -- so every pipeline-launched stack still
  drew the debug art. Config, JSON and the field table now agree. runbook §1 records the 2026-09-10 defaults
  (collection no longer restarts the deploy, eval pins the restart, SIM_VIS none). standards §1: a one-off runner runs
  from outputs/experiments/<name>/launch.sh and its source is committed under scripts/experiments/.
- **visible10 + visible200 collections** (0ffbb14 and the Now entries): runners under scripts/experiments/.
- **fix/first-round-anchor** (384381e, branch, NOT merged): see MERGE PENDING in Now.
- **Everything uncommitted in the shared tree committed** (2dd6d9c pick-and-place line, 2ba1a89 ScaleBFM line,
  2a67ea1 session 29's stack lock + SONIC runners + doc corrections, 4934d4c my two missed visible200 runners and
  the doc map, d73c194 a lint fix of my own): 61 paths, split by line, each commit naming whose work it is and that
  the 15:07 bottle gate (PASS 16/16, yield 0.60) covered them -- every one of those files was already in the tree
  when that gate ran. Tree clean, make check green.
- **README pick-and-place entries** (87bcef6): the line registered its docs in docs/README.md and CLAUDE.md but the
  root README carried a SECOND doc map that missed them. Added the rows, and stated that docs/README.md is the
  authoritative index and the root table a curated subset, so a new doc is registered in one place.
- **eval_closed_loop.py's src bootstrap fixed** (23b5436): it reached src/locomani instead of src, so BOTH eval
  entry points had been unlaunchable since the 4a package move (29bdf84). Found by the first eval attempt crashing.
  Audited all 267 files under src/ for the same off-by-N; it was the only one.
- **EvalConfig.restart_deploy** (83adaa4, default True so no historical protocol changes): added to test whether the
  collection/eval heading-anchor mismatch explains §18's 2/100. The test failed to run (see Open questions).
- **Eval provenance capture** (27dd88a, running): `run_learn_pilot` writes `config_snapshot.json` ONCE, at whichever
  stage runs first -- for this pilot the 07:02 collect, `git_sha 56617ae`, 260 digests -- while the eval starts ~16:00
  against a tree several commits ahead plus ~31 files modified by other sessions. The eval's number would carry a
  nine-hour-old source stamp. `scripts/experiments/visible200_eval_provenance_20260910.sh` waits for the runner's
  stage markers and records git sha, `git status`, a full `git diff HEAD` patch and `source_digests()` at eval START and
  eval END, then writes a verdict line saying whether the tree moved mid-eval. Does not touch anyone's work.
- **Eval-calibration provenance thread** (STATE open questions, 50e6068 -> 6498afb -> 88baac1; session 29 wrote it up as
  ledger bottle §17.7 plus dated qualifiers on `docs/reference/runbook.md:7` and `docs/evaluation.md:941`, still
  uncommitted on its side): the stand-sampling band moved on 2026-09-07 (`8c621fb`), so flow200's 50/100 cannot be the
  external reference for a post-09-07 arm, while the dart-vs-flow 2x2's internal comparisons are untouched. Two of my
  own intermediate conclusions were wrong and are recorded as such in §17.7.
- **Verified session 29's uncommitted guards against my eval path** rather than trusting them: the calibration gate only
  fires when `VLA_POLICY is None`, `main()` never rebinds `VLA_POLICY` so there is no UnboundLocalError, the eval runs
  without `--record` so the exporter probe cannot fire, all 31 modified modules compile and `make test` is green.
- **3DGS background plan** (96184be): docs/handoff/3dgs_background_plan.md, 430 lines -- research + grilling record,
  two decisions taken, four rendering routes left open. ROADMAP's 3DGS entry points at it and records that its
  2026-08-25 "UE for diversity, 3DGS for one-scene fidelity" criterion is superseded.

## Changed (2026-09-10 pick-and-place session, 09:40-13:15 UTC; UNCOMMITTED; GPU1 used with the user's authorisation)
- pick_place task IMPLEMENTED and piloted: tasks/bottle/place_phase.py (raise, turn-to-band, carry-over, descend, release,
  retreat, verify) hooked after the bottle lift by BottleTaskConfig.goal; PlaceConfig; --task-config in the gym; Task
  `pick_place` registered (configs/tasks/pick_place.json); run_learn_pilot takes target/prompt from the task config;
  build_sim_scene --harden-pads; reach.grip_below_center may be negative. Acceptance pilot 10/19 (all 9 failures on the
  grasp side, place 10/10 once lifted, release xy 1.4-2.6 cm, basket in view 100 %); shakedown 10/15 before it. bottle gate PASS 16/16 (ledger §6). Ledger
  docs/ledger/pick_place.md, reels outputs/videos/pick_place_pilot{,2}_20260910/reel_10ok.mp4. Object shape lesson: the
  ready-grasp line needs grip_z 0.70-0.73 with <= 4-5 cm of object above the grip -> 6 x 14 cm can gripped 3 cm above centre.
- SIMPLE's basket scene brought into MuJoCo: assets/simple/{basket,soup_can}/model.xml (visual OBJ +
  CoACD convex pieces from third_party/SIMPLE/data/assets, basket rotated y-up -> z-up, origin = centre). scene_lib:
  two MESH_TARGETS entries + SCENE_PRESET=pick_place (basket at (-0.34, 0) yaw 90, soup can (-0.40, -0.28) scaled to 9.0 cm
  tall / 6.0 cm body for a 1.2 cm jaw margin a side; the sugar box was removed on request; target soup_can, goal basket via a new per-preset goal_object) and a `container=True`
  flag that skips the jaw-span check; compose_scene.py --preset. Checks: OSMesa placement renders + 2 s settle
  (objects sink <= 0.4 cm, no launch) + soup can dropped into the basket rests inside at 0.4 cm off centre:
  outputs/scene_checks/pick_place_20260910/. make lint green. No expert / pipeline / Task yet.
- Design for the pick-and-place expert (walk, right-hand grasp, turn ~30 deg, place in the basket to the left) written to
  docs/handoff/pick_place_plan.md: 9 phases (1-4 reuse bottle verbatim), ego-frame check passed offline at spawn / stand /
  place (both objects inside 0.80/0.90), 7 new failure leaves, 10-round pilot acceptance. Nothing implemented.

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
0-SONIC. RE-CALIBRATE THE GRASP LINE ON v1.1 (new 2026-09-11, user ruling made v1.1 the default).
   **(i) AND (ii) ARE DONE -- ledger §21 (the gait map) and §22 (COARSE 0.40, arrival fixed: stalled 5/10 -> 0/10,
   park 0.467 -> 0.393). Stage C does NOT need `visible_brake_depth`: brake/stop now match release to 2 mm.
   NEXT is the grasp phase, which is where the failures moved: "coupled safe-hand IK did not track" x2 and
   "local alignment timed out" x1 of 10, plus one v1.1-specific "walked 0.54 m off the spawn while standing up"
   (a stand-up/heading-anchor behaviour, not a speed). Then (iv) TRACK_ROT_BOOST with diag_wrist.**
   The other sites of the same hardcoded constants, all to be handled together or the config is half-applied:
   stream_reach.py:250-267 (_creep_to_band, dead in production but test-covered),
   recoverable_grasp_policy.py:356 (recovery, off in the visible profile), primitives_tuning.py:306-317
   (NAV_SPEED 0.45 -> 0.273 m/s on v1.1, NAV_SPEED_MIN 0.35 -> 0.083; both far above the stall rule, not blocking),
   place_phase.py:68 (pick_place's turn_command(speed=0.20)).
   Order matters: (i) re-measure the speed->velocity map -- v1.1 delivers 4.3 cm/step at speed 0.30 against
   release's 17, and that single fact explains §17's 0/20. NOTE the tool for it, diag_shuffle.py, was DELETED by
   refactor step 2 (ec0eb1c) and survives only in .claude/worktrees/old-78536ae/src/teleop_free/ -- restoring it
   is an import port (agent_server/primitives/privileged_pub/paths -> locomani.*). diag_speed / diag_wrist /
   diag_squat / diag_idle_drift are in the same boat and stage (iv) needs diag_wrist; (ii) then the brake band and
   arrival_advance, then reach_config's stand (0.30/42.0 are release-era); (iii) `TRACK_ROT_BOOST=4.5` is kept
   and marked pending -- the 2026-08 sweep that called the boost "meaningless on v1.1" was an orientation sweep,
   not the grasp line. Until (i) lands, a v1.1 grasp success rate is not a property of the checkpoint.
   Also note for every training/eval decision: release-era datasets and checkpoints are a DIFFERENT token space
   (ledger §20.3) -- to reproduce an old number, set SONIC_POLICY=policy/release for the whole chain.
0-IMG. IMAGE-SERVO (the active line). Make the finger clearance a PROJECTION on the commanded aim instead of a
   reactive back-off: while unaligned, clamp the aim along the entry axis so the grip point can never sit closer
   than `clearance` to the object, making entry into its column geometrically impossible. Keep decide()'s L-shape.
   ACCEPTANCE, already instrumented and not needing a success rate: per-frame minimum `grip_standoff_m` must stop
   falling below 0.0653 m (the geometric limit), and the agreement rate must not fall (attempt12 was 80% overall /
   97.5% when the fit is usable).
   THEN, in order: (a) the far field should stop requiring a precise fit -- bearing survives poor depth and 21% of
   far-field frames have no usable fit at all; (b) `prealign_depth=0.44` should become a continuous blend, which
   removes both the 3 ambiguous labels and the passenger phase (§19.8); (c) re-measure the near-field agreement
   rate, whose 100% currently rests on 16 frames because episodes are cut short at 18-21 s.
   DO NOT re-tune the far field further, and do not add a new discrete stage -- the L-shape already exists.
0a. DONE 2026-09-11 -- the 2/100 is diagnosed (ledger §18.2-18.9, and the SESSION cf entry in Now).
   THE NEXT PIECE OF WORK, in order:
   (i) BUILD AND VALIDATE THE GATE. A "no-progress plateau" detector: per frame, the commanded change for each
       subsystem (base / arm / gripper), reporting the longest run where the ARM is near-static while the task is
       unfinished, and its share of the dataset. It MUST go red on visible200's existing data first (wrist z is
       flat at 0.111 for 100+ frames of every episode) -- that is the acceptance test the ambiguity metric failed.
       No GPU. This is the ruler the new expert design is scored with.
   (ii) REDESIGN THE EXPERT so the arm is never a passenger: arm pose a continuous function of what is visible
       (bottle size/position -- ego far field, wrist near field), monotone from the first frame, so there is no
       phase whose only source of progress is the base. Keep the gripper DISCRETE (it is the one component that
       is provably correct, and its plateau is not absorbing because the arm is moving through it).
   (iii) Spawn distance 0.5-1.5 m + speed jitter (today: --spawn-edge 1.1 1.3, a +-10% band). Close spawns start
       the episode STATIONARY near the table -- exactly the state the policy sticks in -- labelled with the
       correct continuation.
   (iv) Stand azimuth stays 42 deg (bottle to the robot's right); nothing this session touched it.
   (v) 20-episode pilot -> gate (i) -> only then 200 episodes. There is NO validated offline gate for a
       behaviour-cloning fixed point, so the real check is still train + a short closed loop (77 min + ~40 min).
   NOT DOING: DAgger (user ruling 2026-09-11). NOT DOING: the delta_indices=[-10,0] retrain -- its premise died
   in §18.9. NOT DOING: a continuous gripper -- it would change the only part that works.
0d. cf: the refactor follow-up (a) is still undone -- the eval-path A/B (old 78536ae eval client vs new, 20 rounds,
   same checkpoint). A checkpoint now exists, so it is finally possible.
0b. cf: merge fix/first-round-anchor as soon as Codex's uncommitted tree edits land (see MERGE PENDING).
0c. cf, after the eval: make `run_learn_pilot` write a config snapshot per STAGE, not once per pilot
   (memory provenance-snapshot-is-per-pilot-not-per-stage). Not done now because that file is the one the running
   pilot executes and session 29 has uncommitted changes in it.
0. pick_place next (this session): same-seed water-bottle control for the grasp-side yield; randomise basket/can
   (plan §4); a 200-episode collection + training once the yield question is settled.
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
- [RESOLVED? 2026-09-11, session cf] PRE-EXISTING unit failure on main (not from the ScaleBFM merge, failed on
  6857fbb too, 2026-09-10): tests/test_pose_feedback.py::PoseFeedbackTests::test_offhand_guard_reuses_current_observation
  raised 'stale observation for offhand guard' from backends/sonic/streamed_arm.py:235. It PASSES now, 3 runs out of 3,
  and the whole suite is green -- either someone's uncommitted fix landed in the tree or it was order/timing dependent.
  Whoever owns it: confirm and delete this line. `make check` is red only on another session's untracked
  scripts/experiments/scalebfm200_reel_20260910.py (ruff format).
- User decision on data recovery (Next 1). Nothing runs until then.
- GPU1 is unusable for sims while the foreign AnyImageNav job holds it at 97 %.

## Open questions
- **Will a projected clearance actually stop the sweep?** The finger-extent model predicts the two swept rounds
  correctly (minima +0.050 and +0.057 against a 0.0653 m limit), so enforcing it geometrically should work. Untested.
- **Why does the expert approach from almost straight above?** The entry axis measures fwd +0.180 / up -0.860 in the
  near field, i.e. the grasp descends onto the bottle. Nobody chose that deliberately -- it falls out of
  `templates()[4]`'s terminal attitude. A more horizontal approach would need far less vertical clearance and might
  remove the whole problem. Not investigated.
- **The far field has no usable RGB fit on 21% of frames.** Those are unlearnable labels. Is the fix a wider
  uncertainty gate for coarse motion, a better ego-side fit, or simply not demonstrating there?
- **Does the base-speed coupling do any harm?** It is kept because §19.8's "the base outruns the arm" is real, but
  it caused one deadlock (r3's 62 s timeout). It may need a floor so the base never fully stops.
- **ANSWERED 2026-09-11 (was: why does the policy never command a close?)** Because nothing is between the jaws.
  The arm never leaves the approach posture; the gripper decision was always correct. Ledger §18.2-18.9, and the
  SESSION cf entry in Now. What replaced it is the question below.
- **OPEN, and now the main one: will an expert with no passenger phase actually remove the fixed point?**
  The diagnosis says the policy parks in a fully in-distribution state where "hold" is the correct action and the
  base was the only source of progress. Making the arm advance continuously with what it sees removes that
  structure. That is a well-motivated bet, NOT a proven fix -- a fixed point is a property of the learned function,
  not of the data, so no offline gate can settle it. Only collect + train + closed loop can.
- **OPEN: is there any offline gate that predicts a behaviour-cloning fixed point?** The ambiguity metric
  (dataset_ambiguity.py) failed its acceptance test. The plateau detector in Next 0a(i) at least measures the
  structure we intend to change, but "no plateau" has not been shown to imply "no fixed point". Treat it as a
  necessary condition, never a sufficient one.
- **OPEN: 2 of 99 rounds predicted a close that never reached the executed stream** (rounds 053, 066; §18.2). Small,
  real, and nobody has looked at the async client's chunk handling for it.
- **OPEN: what does `video.ego_view` do to the arm command?** It was the only key to move the descent statistic
  significantly, and only on 1 of 2 bases (§18.5). Under-powered; would need the repeated-sampling protocol to say
  anything. Possibly nothing.
- **OPEN: `state.right_hand` / `state.left_hand` layout mismatch.** Live puts the two jaw components at indices
  0,1; the dataset's observation vector has them at 4,5 (same values, different slots; per-index std confirms).
  So at eval the model reads zeros in the slots it learned from. PROVEN NOT to be the cause of the 2/100 (§18.3:
  substituting the live hand vector into a closing observation still closes 5/5), but it is a real defect and
  should be fixed before it matters for something else.
- **Does the collection/eval heading-anchor mismatch matter?** Collection (restart_deploy=false) re-anchors with
  hold+reanchor after each reset; the eval restarts the deploy and takes "anchor = landing yaw". 65 measured the two
  8.5 deg apart. UNTESTED, not excluded: the A/B scored 0/17, every round a [stand] failure, degrading from "did not
  stabilise (pelvis z 0.094)" to "teleport landed 0.81-0.92 m off" -- without the per-round restart the eval path
  cannot even stand the robot up, which is itself a finding about how much that restart is carrying.
- **3 rounds of 99 predicted a close that never reached the executed stream** (round 027 chunk -0.0175 vs executed
  0.0048, also 052 and 065). Small, but it is an execution gap in the async client and nobody has looked at it.
- **站位采样带在 2026-09-07 变过,所以 flow200 的 50/100 不能当 ready-grasp 线的外部对照**
  (2026-09-10 查证,与今天的改动无关;此条是同日两次过头表述的最终版)。
  事实链:`reach_config.py` 与全部 `configs/collection/ready_grasp_*.json` 都是 8c621fb(2026-09-07)新建的,
  而 flow200 的 50/100 是 2026-09-02/03(台账 §5、§8),那时 `--reach-config` 这个 flag 还不存在。
  `ReachConfig` 的 docstring 写的是"对**当前**桌面 reach 控制器的显式、带版本的标定",其默认
  post_min 0.22 / azimuth 36.5 与改动前 `grasp_gym.py` 注释里反复出现的 0.22 带一致,
  所以 flow200 落在旧带上;ready-grasp 系列(全部 7 个 profile 与 `bottle_visible.json`、`pick_place.json`)
  统一是 0.30 / 42.0,经 `ready_grasp_policy.sample_stand` 改变站位与出生分布。
  backshift 0.08、max_drop 0.13 两边相同,差异只有这两个字段。
  **推论**:ready-grasp 线各臂内部可比(07d、07d-200 的 57/100、即将出的 visible200);
  flow200 的 50/100 属于旧带,`docs/reference/runbook.md:7` 与 `docs/evaluation.md:941` 把它写成"当前正式基线",
  作为**新臂的外部参照**这一用法已经失效,需要加限定。
  **但 dart-vs-flow 的 2×2 本身完好**:四格(flow200 base 50 / ens 46,dart200 base 24 / ens 56)是同一实验、
  同一协议一次跑出来的,组内比较全部成立,结论"DART 数据教会桌边停滞"不受影响。
  区别是**跨实验 vs 组内**,不是有效 vs 作废。
  一般规则(与本条独立):比较前看各自记录的 argv 而不是脚本名,因为 `run_eval_arm.sh:31` 的 `EXTRA=${7:-}`
  会被 71 行原样拼进 eval 的 argv;这一条对 flow200 不适用,因为那时 flag 还不存在。
- Did the GPU0 overlap cost visible200 anything? 65's crate chain shared GPU0 during batches 07-14 (43/80 vs 157/255
  elsewhere). Session integrity and both audits are clean, so nothing is provably damaged; if the trained model shows a
  stratified oddity, those 8 sessions (07-45-01 .. 08-24-52) are the first suspects. Unknowable after the fact.
- "stand infeasible, no reposition" was 72 of 101 in-round failures on 2026-09-10 against ~19% on 2026-09-09. The
  no-restart default parks 3 cm farther (ledger §11 park 0.332 vs 0.301), which points the same way. If the trained
  model's stand distribution looks different from 07d's, this is why. Not measured.
- A background runner's admission check: gating on GPU memory alone is not enough, another session can hold the
  instance while working on the other GPU. Today this was covered by a hand-agreed time window. Session 29 is writing
  it up as a FAQ row together with two of its own findings (instance 1 cannot record; the gym does not follow INSTANCE).
- 3DGS background: 8 decisions still open, listed in docs/handoff/3dgs_background_plan.md §9. The one with teeth is
  that D2 (no separate acceptance) means the closed-loop eval can only show "no harm", never "helps".
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
- **Crate walk-in stand delivery (user ruling 2026-09-11: "先放一放,之后再调整").** The walk-in delivers
  the planned stand with sd 5.7 cm (range -9.8..+12.0), while success needs the ACHIEVED stand-off under
  ~0.27 m: 29 rounds separate perfectly, 26 successes 0.148-0.265, 3 `[descend] never seated` at
  0.282/0.285/0.301. Planned stands span 8.7 cm and are deterministic; achieved span 15.3 cm -- the gait
  noise is larger than the whole planned distribution, so scatter, not the plan, picks the winners.
  Consequence for collection (worse than for the gate): every kept episode is one where the scatter landed
  near, so a crate dataset would be biased to near stands and hold no correct behaviour for the far ones --
  the shape of memory policy-stop-point-is-set-by-the-crawl. Cheapest first step when it is picked up:
  does the walk-in's retry criterion cover the DEPTH axis or only the lateral one (`[stand] lateral
  -14.1 cm after 3 walk-ins` was one of the 30 rounds)? The single-arm line has
  tasks/bottle/arrival_validation.py for this; the crate line has no equivalent.
  Ledger crate.md 2026-09-11; memory crate-standoff-times-yaw-boundary. NOT to be tuned unasked.
- **Crate gate `stand_off_m` tolerance, parked with the above** because it is the same root cause: it gates
  an ACHIEVED, gait-noisy quantity at `TOL["depth_m"]=0.01` and the same code twice moves it 1.8x that.
  Do NOT widen it to make a change pass -- that hides the defect above. Honest fixes when the line is
  picked up: re-record the baseline at 30 rounds, or gate the DELIVERY ERROR (|achieved - planned| sd)
  instead of the median. Memory crate-gate-stand-off-tolerance.
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
- 2026-09-11 stages A-E closed: gate baseline re-recorded on v1.1 (30 rounds, yield 0.233, self-compare 15/15,
  release's kept separately and NOT comparable); TRACK_ROT_BOOST left at 4.5 because the wrist measurement could
  not reproduce a known-true prior on release. Ledger §28-29
- 2026-09-11 20 paired rounds: v1.1 with all four fixes 7/20 vs the unfixed v1.1 arm 0/20 (McNemar 7:0, p=0.0156)
  and vs release 9/20 (4:6, p=0.754, indistinguishable). lateral 0.207 vs release's 0.204, park 0.363 vs 0.369.
  The "v1.1 cannot drop into the collection line" verdict is lifted. Ledger §27
- 2026-09-11 planned stand azimuth 37.5 -> 29 deg on v1.1: achieved 42.8 -> 32.0 deg, lateral 0.272 -> 0.189, park
  0.400 -> 0.350 (release 0.349), depth unchanged; `stand infeasible` 4->2, `local alignment outside comfortable
  IK` 2->0; 5/10 against 1/10 twice at plan 37.5 and release's 4/10 (paired 4:0, p=0.125). The plan was distorted
  by applying `backshift` along depth only, and release never executed it. Ledger §26
- 2026-09-11 the v1.1 lateral residual is a PLAN problem: both checkpoints crab identically at the same speed
  command (0.17-0.29 m/s at 0.40, ~0.01 at 0.30), so v1.1 at COARSE 0.40 executes the planned 37.4 deg stand
  azimuth (+5 deg) while release undershoots it by 9 deg and lands at 29 -- and the downstream is tuned to
  release's 29: achieved azimuth [0,30) 5/7 ok, [38,50) 1/6 on release's own rounds. Ledger §25
- 2026-09-11 reset seating bug (repo-wide): `_seat_on_floor` read the bounding-SPHERE radius as the foot's extent,
  so every reset seated the pelvis at 0.892 instead of 0.785 and dropped the robot 10.8 cm. Fixed; first-second
  displacement median 0.099 -> 0.034 m, max 0.489 -> 0.083, no falls. It changes every episode's initial
  condition. With that noise gone the v1.1 arrival residual is 100% LATERAL: +10.5 cm vs release, depth solved.
  Ledger §24
- 2026-09-11 grasp phase: `coupled safe-hand IK did not track` is a ROTATION residual (pos_err 9x inside its
  tolerance) caused by holding the palm's world attitude while the pelvis pitches +12 deg through the squat --
  against a 0.20 rad = 11.5 deg tolerance. Raised to 0.35 (derived from the measured pitch), but the 10-round
  verification does NOT stand: 8 of 10 failures were upstream, and rounds reaching the squat are 6/10, 2/3, 2/10
  across identical configs. Ledger §23
- 2026-09-11 Stage B: APPR_COARSE 0.30 -> 0.40 on v1.1 (from the measured map, per-checkpoint table in
  gait_speeds.py). Same ten cases, paired: release 4/10, v1.1-old 0/10, v1.1-new 3/10; `stalled` 5/10 -> 0/10,
  park median 0.467 -> 0.393 m, brake geometry now release-like. Failures moved to the grasp phase. Ledger §22
- 2026-09-11 Stage A gait map, both checkpoints in one session: the speed command -> velocity curve is a STEP, and
  the step is displaced +0.065 by the checkpoint (release 0.215->0.26 = 0.016->0.123 m/s; v1.1 0.28->0.30 =
  0.007->0.044). APPR_COARSE=0.30 therefore delivers 0.044 on v1.1 against release's 0.203, spending 37.8 s of a
  45 s approach budget. Two of my own mechanisms retracted with it. Ledger bottle.md §21
- 2026-09-11 DEFAULT SONIC CHECKPOINT -> policy/sonic_v1_1 (user ruling; config change, no GPU). Encoder layout is
  now read from the checkpoint's own config (1751 / heading anchor @584 vs release 1762 / @601); the codebook is
  identical across checkpoints but the latent is not, so every pre-2026-09-11 dataset and model is release-space.
  Grasp timings still release-calibrated (§17 0/20 vs 9/20). Ledger bottle.md §20
- 2026-09-11 visible200 2/100 DIAGNOSED: the arm freezes in the expert's own approach posture for 85 s; the gripper
  was never at fault. Palm-to-bottle separates perfectly (<=0.138 vs >=0.156 m), the miss is vertical (+0.029 vs
  +0.191 m), wrist roll 0.24 vs 0.018. A behaviour-cloning fixed point in an in-distribution state, caused by the
  arm being a passenger through the whole approach. Ledger §18.2-18.9
- 2026-09-11 two claims of mine RETRACTED after the user asked for controls: "the wrist camera triggers the descent"
  (sampling noise) and "the policy cannot tell walking from parked" (a linear probe says it can, R^2 0.665 / 90%)
- 2026-09-10 visible200 10k steps -> closed loop 2/100; open loop on its own val is at the noise floor
  (nmse_gripper 0.0024, close timing 2.3 frames). Mechanism unlocalised, ledger bottle §18 + §18.1
- 2026-09-11 restart_deploy=false eval arm: 0/17, every round [stand], degrading -- the eval needs the per-round restart
- 2026-09-10 ScaleBFM no-retract 200 ep -> seed17/3000 -> 100 closed loop: 41/100 (<=38/100 under the old rule vs ledger §9's 25/100); the whole gain is high_near, low_deep unchanged at 5/33
- 2026-09-10 visible200 collection: 34 batches / 335 rounds -> 200 kept (yield 0.60), convert rc=0, check_dataset 32/32
  PASS on both splits; first round of EVERY batch failed at the spawn (34/34) until the anchor fix
- 2026-09-10 first-round anchor fix verified: round 1 past the spawn 3/3 on the fix branch vs 0/34 on main
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
