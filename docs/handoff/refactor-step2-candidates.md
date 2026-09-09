# Refactor step 2 — retired-branch deletion candidates (evidence table, 2026-09-09)

For `docs/handoff/refactor-plan.md` decision 4 ("删代码留裁定; ReachConfig 收缩到当前协议字段; 例外: `--ensemble` 推理开关和
lockstep 评测保留; RTC 训练侧和 planner 退化模式删"). Evidence only; nothing deleted here.

**Baseline = current official protocol**: `configs/collection/ready_grasp_visible.json` and
`outputs/experiments/ready07d_200_20260908/reach_config.json`. Both set only: `ready_grasp_baseline, ready_grasp_recovery,
transport_guard, pre/post_min/max, azimuth_deg, max_drop, recovery_attempts, recovery_timeout, grip_below_center,
arrival_advance, measured_squat_handover, coupled_squat, timely_capture, brake_preplan=false, squat_prediction_*`
(+ `visible_*`). Launch argv (`outputs/experiments/*/launch.sh`): `run_learn_pilot.py --gym-env GYM_SHUFFLE_BAND=0`, gym gets
`--face-object`, never `--one-shot` or `--dart-frac`.

Structural fact: `stream_reach.py:2073-2076` returns `coordinated.execute(...)` whenever `ready_grasp_recovery` is true, so the
rest of `_execute_grasp` (≈2079-2312: `_creep_to_band`, `GYM_IDLE_SWEEP`, `GYM_ONE_SHOT`, `_nudge_*`, reach retry) is dead for
the official line.

## A. Safe to delete (retired verdict, no live dependency) — ≈800 src lines, ≈400 test lines, 12-14 configs, 4 docs

| candidate | ~LOC | files | used-by | verdict | risk |
|---|---|---|---|---|---|
| `coordinated_arrival` flag + strategy | 255 | `coordinated_arrival.py`, `grasp_gym.py:1021,1524-1530`, `configs/collection/coordinated_arrival.json`, `docs/coordinated_arrival.md` | grasp_gym only; `ReachConfig:124` forbids it with `ready_grasp_baseline` | "The historical collector remains the default." `docs/coordinated_arrival.md:3` | it imports `_servo_continuous/toppled/PlannerArm` from stream_reach — those stay |
| `continuous_capture` | 167 | `continuous_grasp_policy.py`, `capture_feedback.py`, `coupled_grasp_policy.py:297-301`, `measured_grasp.py:80,150-155`, `grasp_gym.py:1538-1541,1728-1730`, `reach_config.py`, `ready_grasp_continuous.json`, `docs/continuous_grasp_policy.md` | grasp_gym:1539 only; `capture_feedback` only inside `if continuous` | "not promoted to the default configuration" `docs/continuous_grasp_policy.md:70` | confirm `capture_feedback` has no other caller |
| `tool_point_grasp`, `horizontal_entry`, `grasp_depth_preference`, `local_point_lead/speed` | 83 | `ready_grasp_policy.py:26-42` (terminal_rotation level branch), `coupled_grasp_policy.py:158-172`, `measured_grasp.py:81-83,251-253,274-275,288`, `tabletop_grasp.py:91`, `reach_config.py`, `ready_grasp_{level_bottle,tool_point}.json`, `docs/level_bottle_grasp.md` | no live config; `tests/test_level_bottle_grasp.py`, `test_measured_grasp.py:118,203` | "has not passed expert acceptance and is not the pilot default" `docs/level_bottle_grasp.md:5`; `EXPERIMENT_STATUS.md:11` | `terminal_rotation` is live: remove only the level-entry branch |
| `capture_reference_handover` | 25 | `stream_reach.py:1382-1396`, `coupled_grasp_policy.py:214`, `measured_grasp.py:207`, `reach_config.py`, 5 configs | `tests/test_capture_reference_handover.py`, `test_coupled_grasp_policy.py:99,114` | "It is not promoted." `EXPERIMENT_STATUS.md:8-9` | inside `_servo_continuous` (live): excise the branch only |
| `recovery_region_stop` | 22 | `recoverable_grasp_policy.py:336-350`, `coupled_grasp_policy.py:89-91`, `reach_config.py`, 5 configs | `tests/test_recovery_region_stop.py`, `test_coupled_grasp_policy.py:84,92`, `test_visible_expert.py:40` | `EXPERIMENT_STATUS.md:11` "retained experimental intermediate profiles, not recommendations" | simplifies the visible_expert validator |
| `coupled_hand_lead` | 15 | `coupled_grasp_policy.py:46-62,328`, `reach_config.py`, 4 configs | `tests/test_transport_ik_fallback.py` | `EXPERIMENT_STATUS.md:13` | keep `solve_transport_ik`, drop the lead retry + 4th return |
| `brake_preplan` | 16 | `coupled_grasp_policy.py:94-103`, `reach_config.py` | `test_visible_expert.py:41` | false in both official configs | none |
| `pregrasp_lateral_jitter` | 8 | `coupled_grasp_policy.py:244-246`, `reach_config.py` | `test_visible_expert.py:39` | 07c 20/60 vs 07d 29/60 `docs/ready_grasp_arms_20260907.md:40-41,81` | doc says "可叠在 07d 上作下一个单变量": record the verdict so it can be re-derived |
| `arrival_advance_jitter` | 12 | `grasp_gym.py:1632-1638`, `reach_config.py` | `test_visible_expert.py:38,49` | 07e 19/60 vs 07d 29/60 `ready_grasp_arms:43,80` | both official configs leave it 0 |
| `pregrasp_min_forward` | 7 | `coupled_grasp_policy.py:247-250`, `reach_config.py` | none | never set by any config | dead default |
| `GYM_FACE_OBJ` env | 13 | `grasp_gym.py:188,1706-1716` | none | "实测更差" `docs/runbook.md:46` | NOT the CLI `--face-object` (live) |
| `GYM_CREEP_OBS` env | 8 | `grasp_gym.py:184-187,1885`, `tools/diag_transition_cross.py:255` | none | no-op plumbing (`docs/transition_control.md`) | none |
| `GYM_SHUFFLE_BAND>0` branch | 15 | `grasp_gym.py:1617-1624,1717-1720,1727` | official runs pass 0 | 07d 30%→48% p 0.052 `docs/experiments.md:234` | keep `APPR_SHUFFLE` (visible decel floor), delete the gate |
| DART `--dart-frac/--dart-far` | 80 | `grasp_gym.py:433-465,472-473,489-493,778-783,806,880-883,953,1004,1063-1074,1667-1675`, `scripts/collect.sh`, `docs/collection.md:960-993` | none | dart200 24/100 vs flow200 50/100 p 0.0002 `docs/experiments.md:11,151` | `--one-shot` exclusion check goes with it |
| `GYM_IDLE_SWEEP` env | 25 | `stream_reach.py:2117-2140` | dead tail | comment calls it "unproven" | only via legacy `collect_episode` line (D) |
| `GYM_STAND_COVER` + `TargetPlan.stand3` + `grasp_gym.sample_stand` + `reach_map.stand_cells` | 45 | `grasp_gym.py:119,172,182,191-206,554,1630,1885`, `reach_map.py:405-` | unreachable under `ready_grasp_baseline` (`grasp_gym.py:636-641`) | creep hidden variable `ready_grasp_arms:13-18` | `docs/runbook.md:45` still lists it for flow200: fix in step 6 |

Tests to delete with A: `test_coordinated_arrival.py`, `test_capture_feedback.py`, `test_level_bottle_grasp.py`,
`test_capture_reference_handover.py`, `test_recovery_region_stop.py`, `test_transport_ik_fallback.py`, plus cases in
`test_coupled_grasp_policy.py`, `test_measured_grasp.py`, `test_visible_expert.py`, `test_ready_grasp_policy.py`.
Configs to delete: `coordinated_arrival, ready_grasp_continuous, ready_grasp_level_bottle, ready_grasp_tool_point,
ready_grasp_region_stop, ready_grasp_bounded_transport, ready_grasp_lower_body, ready_grasp_reference_region,
ready_grasp_reference_handover, ready_grasp_stable_reference, ready_grasp_stable_bounded_reference, ready_grasp_capture_continuity`
(+ `ready_grasp_closer`, `ready_grasp_baseline` if the no-recovery path goes). Keep `ready_grasp_visible.json`.

## B. Delete with care (eval / live modules touched) — ≈620 lines (+230 conditional)

| candidate | ~LOC | files | verdict | risk |
|---|---|---|---|---|
| a18 / 18-D action space | 540 | `action18_client.py`, `tools/export_action18.py`, `tools/convert_to_hf_format.py:225,530-531,603-608,768-798`, `configs/gr00t_sonic_2cam_a18.py`, `vla_round.py:258-260,479,793-797,978-982`, `tools/eval_closed_loop.py:306-323`, `scripts/run_eval_arm.sh:29-36,56-57,65` | refactor-plan "不在范围: a18 可运行性" | cuts into live `vla_round` send path and the converter join; needs the gate |
| planner-degraded deploy mode | 50 | `scripts/run_eval_arm.sh:25-27,54`, `scripts/restart_deploy.sh:44-73,120`, `tools/eval_closed_loop.py:96-102,319-323,375-401` | "自此正式数字一律用干净协议" `docs/experiments.md:105` | = make clean mandatory: drop `--clean-deploy` as an option, KEEP `CleanClock`/`install_clean_shim` (`vla_round.py:1481-1615`); coupled with a18 |
| RTC training side (`RTC_TRAIN_P/KMAX`) | 28 | `src/gr00t/model/gr00t_n1d7/gr00t_n1d7.py:239-260,302-306` | "RTC 三个变体全部显著负收益" `docs/experiments.md:12`; `docs/evaluation.md:907-909` | local patch inside src/gr00t: separate commit; do not touch `action_prefix_loss` |
| legacy `_execute_grasp` tail (`GYM_ONE_SHOT`, `_nudge_*`, reach retry, `_creep_to_band`) | 230 | `stream_reach.py:712-896,2079-2312` | one-shot = flow200 recipe `docs/runbook.md:32,59`, not the current argv | reachable only via `collect_episode.py:458` and `tools/diag_transition_cross.py:218`; tests patch `_creep_to_band` (`test_grasp_transition`, `test_ready_grasp_policy:98`, `test_recoverable_grasp_policy:204`); bimanual imports `PlannerArm`/`reanchor` which are above this range |

## C. Keep

`--ensemble/--ensemble-m` (decision 4 exception); lockstep evaluation (`vla_round.py:1237-1480,1628-1877`, `lockstep_sim.py`,
`lockstep_vendor/`; calibration unresolved, `docs/experiments.md:13,134`); clean-deploy shim; `visible_*`; the whole bimanual line and
its transitive deps (`paths, upper_ref, agent_server, g1_joints, primitives, primitives_const, privileged_pub, stream_reach
(PlannerArm, reanchor + StreamedArm, arm_ik, execution_record, reach_config), grasping, navigator, wrist_map, scene_random,
run_sim_locomani, build_sim_scene, scene_lib, sim_vis, sim_video`); eval/deploy core (`eval_closed_loop`, `make_vla_policy_async`,
`serve_policy` minus `RTCPolicy`, `initial_pose`, `tools/sonic_*`, `failure_*`, `eval_open_loop`, `latency_dist`, `seq_mcnemar`,
`frame_rate_guard`, `instance_watchdog`, `run_eval_arm.sh` MODE=wall, `run_closed_loop.sh`, `seq_ab.sh`); collection core
(`grasp_gym`, live half of `stream_reach`, `ready/recoverable/coupled_grasp_policy`, `measured_grasp`, `grasp_capture`,
`adaptive_grasp`, `tabletop_grasp`, `transport_guard`, `arrival_validation`, `experiment_cases`, `execution_record`, `reach_map`,
`run_learn_pilot`, `convert_to_hf_format`, `check_dataset`, `verification_tail`, `gate.py`). ReachConfig fields that survive:
`post/pre_min/max, backshift, depth_margin, azimuth_deg, approach_timeout, max_drop, transport_margin, sweep_joint_step,
transport_guard, ready_grasp_baseline, ready_grasp_recovery, grip_below_center, arrival_advance, measured_squat_handover,
coupled_squat, timely_capture, squat_prediction_*, recovery_attempts/timeout, pregrasp_rise, pregrasp_gap, local_reach_limit,
visible_*` (43 → ~29 fields; several become task constants in step 4).

## D. Needs a user ruling — ≈8,000 lines

| candidate | ~LOC | files | verdict | note |
|---|---|---|---|---|
| test-time RTC + transition-commit + prefix-sampling cluster | 2,100 | `serve_policy.py` `RTCPolicy` (110), `vla_round.py:247-252,465,864-912`, `eval_closed_loop.py:195-209,485-496,586`, `transition_execution.py`, `tools/{transition_capture,transition_events,analyze_transition_capture,label_transition_pilot,train_transition_pilot,prefix_sampling,probe_prefix_models,audit_prefix_execution,audit_rtc_prefix,diag_rtc_offline,diag_transition_live,diag_transition_cross}.py`, tests `test_transition_execution/test_prefix_capture/test_prefix_sampling` | "RTC 两种形式都是负收益,不再投入" `docs/evaluation.md:912-915`; prefix "0/30,不启用" `docs/prefix_sampling_experiment.md:34` | decision 4 names only the training side; docs verdict says retired |
| `--client blocking` (`make_vla_policy`) | 246 | `vla_round.py:488-733`, `eval_closed_loop.py:90,105,405-410` | "仅作 A/B 对照保留" `docs/evaluation.md:136,315` | docs say keep |
| 47 unimported `diag_*.py` | 7,147 (3,303 for 24 orphans with no doc/script mention) | `diag_angled_grasp, diag_can_watch, diag_core_visit, diag_core_zone, diag_dex1, diag_hand, diag_heading, diag_idle_drift, diag_map, diag_movement, diag_nav, diag_pitch, diag_plan_scan, diag_reach_error, diag_reach_sweep, diag_reach_volume, diag_ready_walk, diag_repeat, diag_settle, diag_spawn_view, diag_speed, diag_stand, diag_station_keep, diag_stream` (orphans); the other 23 appear only in `docs/reference.md` / `collection.md` / `ROADMAP.md` catalogues | none | the target layout has no home for them |
| unimported `tools/` | 1,400 | `analyze_pose_feedback, audit_frame_sources, audit_object_topple, probe_handover_geometry, summarize_transport_trials, diagnostic_transport, motion_smoothness, reach_wall_sweep, walk_pose_preflight, reach_preview, reach_preflight, sweep_reach_feasibility, analyze_stance_grid, analyze_posture_stages, diag_stance_grid, diag_crouched_handover, diag_arm_response, analyze_arm_response` | none | `reach_preflight, reach_preview, reach_wall_sweep, walk_pose_preflight` are cited by the crate tutorial |
| legacy episode line: `collect_episode.py` + `run_batch.py` + `stream_reach.final_approach/place_streamed/grasp_streamed` | 1,030 | `collect_episode.py`, `run_batch.py`, `stream_reach.py:907-1064,1953-1992` | `FULL_TASK` "尚未迁移到新控制逻辑" `docs/reference.md` | grasp_gym imports 4 constants from `collect_episode`; deleting this unlocks B row 4 |
| isaac helpers (`isaac/frames.py, resolve_scene.py, stage.py`) | 1,015 | reachable from `scripts/build_isaac_assets.sh`, `collect.sh`, `render_wall_pilot.sh` | "闭环评测用 Isaac 观测不需要 hybrid clock" `docs/experiments.md:228` | KEEP (bimanual render deliverable) |

## Totals

| group | src LOC | tests | configs |
|---|---|---|---|
| A | ~800 | ~400 | 12-14 |
| B | ~620 (+230) | ~120 | 1 |
| D (pending) | ~8,000 | ~150 | 0 |

`src/teleop_free/*.py` = 29,242 lines; tools+isaac = 13,238; tests = 3,655. Decision 4 as written removes ~4 %; resolving D removes ~28 %.

## Ordering

1. A first, one commit per family; this is the ReachConfig shrink.
2. a18 + planner-degraded together (they reconcile each other in `eval_closed_loop.py:314-323`); RTC-train as its own commit.
3. Re-record the bottle gate baseline from the completed ready_visible200 run before certifying any deletion.
4. Env knobs surviving A (`GYM_RESTART_DEPLOY, GYM_ARM_CHANNEL, GYM_ARM_VEL, GYM_ARM_PIN, GYM_LEFT_GUARD, GYM_UPPER_SLEW_DEG,
   LOCOMANI_POSE_SIDECAR, SIM_VIDEO_INCLUDE_INIT, TF_INIT_TOKEN_FILE, FULL_TASK, TF_ACTION18`) move into config in step 3.
5. Blocked on: Codex committing `scalebfm/*`; a user ruling on D rows 1-5.

## Executed

One commit per family, on `refactor/step1-tooling` (worktree `.claude/worktrees/refactor-step1`), in the order
above. `git show --stat <sha>` gives the file list; totals below are that same diffstat's insertions/deletions.

| # | family | commit | files | lines (+/-) | status |
|---|---|---|---|---|---|
| 1 | `coordinated_arrival` | `4ab0c6b` | 7 | +3 / -377 | done |
| 2 | `continuous_capture` + `capture_feedback` | `6297f03` | 9 | +3 / -382 | done |
| 3 | `tool_point_grasp`/`horizontal_entry`/`grasp_depth_preference`/`local_point_lead`/`local_point_speed` | `0d335dd` | 10 | +26 / -223 | done |
| 4 | `capture_reference_handover` | `d7bd223` | 11 | +3 / -265 | done |
| 5 | `recovery_region_stop` | `432d314` | 9 | +3 / -166 | done |
| 6 | `coupled_hand_lead` + `brake_preplan` + `pregrasp_lateral_jitter` + `arrival_advance_jitter` + `pregrasp_min_forward` | `eeff157` | 6 | +53 / -115 | done |
| 7 | `GYM_FACE_OBJ` + `GYM_CREEP_OBS` + `GYM_SHUFFLE_BAND>0` | `7a6841f` | 1 | +10 / -33 | **partial** -- see skip below |
| 7b | `GYM_STAND_COVER` + `TargetPlan.stand3` + `grasp_gym.sample_stand` + `reach_map.stand_cells` | -- | -- | -- | **skipped** |
| 8 | DART `--dart-frac`/`--dart-far` (+ its `--one-shot` exclusion check) | `c9e758d` | 3 | +7 / -126 | done |
| 9 | `GYM_IDLE_SWEEP` | -- | -- | -- | **skipped** |

Totals for the 7 completed/partial families: 56 files touched, +108 / -1687 lines (net -1579).

**Skipped, and why (both touch group B, out of this task's authorized scope):**

- **7b. `GYM_STAND_COVER` / `TargetPlan.stand3` / `grasp_gym.sample_stand` / `reach_map.stand_cells`.**
  `grasp_gym.sample_stand` (the module-level one, distinct from `ready_grasp_policy.sample_stand`) is the
  *sole* stand-sampling implementation for the `else` branch of
  `if reach_config.ready_grasp_baseline: ... else: rel, squat_hint = sample_stand(pl, rng)` (`grasp_gym.py`
  `run_round`, ~line 635). That branch is what runs whenever `ready_grasp_baseline` is False -- the
  flow200/legacy collection recipe, which routes into `stream_reach.py`'s group-B "legacy `_execute_grasp`
  tail" (`stream_reach.py:2079-2312`, delete-with-care, untouched here). Deleting `sample_stand`/`STAND_COVER`/
  `stand3` would leave that branch calling a name that no longer exists -- not a group-A-safe deletion. The
  candidate table's own risk note already flagged this ("`docs/runbook.md:45` still lists it for flow200: fix
  in step 6"), i.e. it anticipated needing the doc-reorg/reconciliation step before this can go. Left in place.
- **9. `GYM_IDLE_SWEEP`.** Its only occurrence (`stream_reach.py:2110`) sits inside the exact group-B "legacy
  `_execute_grasp` tail" range this task was told not to touch (`stream_reach.py:2079-2312` -- the code after
  `coordinated.execute(...)`/`coordinated.finish(...)` short-circuits, dead for the official
  `ready_grasp_recovery=True` line but still live for the group-B path). No safe group-A-only edit exists;
  skipped entirely, no commit.

Both skips are candidates for the group-B reconciliation step (refactor-plan.md execution step 4, "抽骨架"),
not for a group-A follow-up commit on this branch.
