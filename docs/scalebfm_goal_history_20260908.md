# ScaleBFM Goal checkpoint

Updated 2026-09-08 19:20 UTC. Goal active; C0/C1 complete, C2 passed, C3 final integration and C4 preparation.

## Read on every context recovery

1. `docs/STATE.md` (resource reservations take precedence).
2. This file.
3. `docs/scalebfm_goal_plan.md` (registered budgets/gates).
4. `docs/scalebfm_visual_expert_design.md` and relevant experiment results.

## Confirmed requirements

- VLA: two RGB cameras + proprioception, no object ground truth.
- Human-like continuous approach/lift/reach, necessary forward bend, little
  unnecessary side lean/twist. Judge learnability through fixed-budget VLA tests.
- Preserve existing uncommitted work/data. No user authorization to commit.
- GPU0 explicitly released by9d and claimed by this Goal. GPU1 has another live workload; leave it alone. No stop_stack.sh or recording stack.

## Current artifacts

- Plan: `docs/scalebfm_goal_plan.md`.
- Frozen original sources and hashes:
  `outputs/experiments/scalebfm_visual_goal_20260908/baseline_source/manifest.json`.
- Prior diagnosis: `docs/scalebfm_sidelean_diagnosis_20260908.md`.
- Prior CPU counterfactuals: `outputs/diagnostics/scalebfm_sidelean_20260908/`.
  These are DONE; do not rerun just to reconstruct the conversation.
- No new dataset or VLA run has started. No subagents started.

## Established evidence

- At reach onset, current wrist command jumps ~60 deg and ~12 cm.
- CPU seeds 1/2/3: smooth wrist rotation alone reduces reach torso roll from
  18.4/17.2/19.9 to 8.9/8.8/9.7 deg; grasp remains 1/3 for both variants.
- Baseline accepted v4 corner data: 199/210 episodes exceed 15 deg torso roll.
- Mode 2 only constrains pelvis/wrists; mode 6 shoulders/elbows too but NOT torso.
- Changing yaw or delaying reach did not remove roll. One after-arrival probe
  reported QACC instability; its task success is not a trustworthy endorsement.
- Current VLA config has one-frame video/state, no explicit history.
- Current expert feedback is 50 Hz but eval task replans at 2.5 Hz +80 ms delay.

## Next concrete operation

Candidate a15 PASSED all 12 development cases AND all per-case audits;
walk/reach overlap12/12, visibility gates passed, high-near close pitch about
-2deg, low-near23–26deg, low-deep30–32deg. All12 deterministic shared-executor
replays exactly match every qpos and command (a15_executor_replay.json).

Qualification DONE: original three RGB cases1101..1103 plus27 disjoint suffix
cases1104..1130 =29/30 task successes (1115 preclose knock). Remaining27 audit
accepted26/27; original-three audit session75459 should be drained/inspected.
Recovery DONE:11/12 (1210 object_shift timeout), all11 successes pass audit and
walk/reach overlap. Qualification ten-case replay running/finished session76849;
inspect qualification_replay10.json before further replay. All12 development
replays were exact. Do not repeat completed experiments.

RESOURCE UPDATE: docs/STATE.md was explicitly updated by9d18:50: GPU0/GPU1 and
instance0 released after eval ended18:45. We claimed GPU0 in STATE after checking
it idle. GPU1 has other live work; leave it alone. GPU0 is authorized for this
Goal's standalone sim/render/train/eval (no stack/ports, never stop_stack.sh).
GPU0 contract run DONE:1301/1302/1303 all successful, recording~5s/case versus
~100s CPU. Path contract_gpu0_a15; session71320. Next audit and SAME-GPU replay.
run_visual_bottle supports --policy-device cuda; use CUDA_VISIBLE_DEVICES=0,
MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID=0. replay uses saved policy_device (CUDA when
recorded on CUDA). Exact equality is asserted within a device, not across CPU/GPU.

C3 schema smoke PASSED: raw student previous_action is the preceding row exactly;
25Hz labels equal the actual shared-executor inputs; actual GR00T loader reads
state91/action29/two-image-time history. schema_smoke_dataset is ONLY a fixture
from qualification1101, never training data. Train entry train_visual_vla.py
implemented; CPU dry-run succeeded at train_config_smoke. TODO change optimizer
string to existing pipeline's adamw_bnb_8bit (currently adamw_8bit), then new dry-run
ID; do not rewrite the previous frozen smoke config. Real training not started.
It explicitly sets data.allow_padding=True (False wraps negative history into
future episode-tail frames), shortest_image_edge240/crop_fraction1.0, batch4 x
accumulation16 =64, seeds17/29, steps3000, saves optimizer state every500 steps.

OUTSTANDING: finish common VLA chunk scheduler/evaluator (including replay through
0.4s request/80ms delayed chunk switching); baseline frozen-task generator through
same29D executor/PD/runtime; bounded per-arm collection/export orchestration;
GPU model/processor memory preflight without optimizer training; fixed-budget
A/B/C data collection and six training/eval runs; selection+confirmation report.
30 unittest tests passed. No formal training dataset or VLA run started.

## Run ledger

| ID | Status | Conclusion / next |
|---|---|---|
| C0-baseline-freeze | done | Source snapshot and v1 gates/budget saved; C1 implementation next |
| dev_a00_baseline | done | 9/12 success; ~60 deg and 12 cm wrist jumps; one failed case QACC warning |
| dev_a01_pose_servo | done, rejected | 5/12; jumps bounded but persistent filtering worsens roll and grasp |
| dev_a02_pose_transition | done, provisional | 9/12; successful cases ~1.3–1.6 cm and <1 deg steps; some sustained roll remains |
| a02_visibility | done | All sampled ticks in all 12 cases have >=1 camera passing pixel/fraction gate; a02_visibility.json |
| dev_a03_grip_matched | done, rejected | 7/12 with kp8/kd0.15; no improvement over a02 |
| dev_a04_close_shortcarry | done, provisional best | 10/12, all torso roll peaks ~10–12 deg; .35–.40 stance and .07 carry; grasp failures remain |
| dev_a05_observable | done | 0/3 timeout reach; physical pre-target error 4–5 cm exceeds 3.5 cm gate; visibility all passed |
| dev_a06_pre_corridor | done | 1/3, 6 cm pre corridor and continuous settle handover; roll 9–10 deg; entry collision/ejection remain |

| gripper_free_response | done, diagnostic | kp40/kd1 free-jaw std 4–6 mm vs kp1/kd.025 ~0.2 mm; JSON saved |
| dev_a07_grip_stable | done | 6/12 strict success, no preclose knock/ejection; most failures carry height 6–7.4 cm, one entry timeout |
| contract_native29_a07 | done, contract probe | 2/3 strict success; forecast approximation <0.36 cm; 3/3 deterministic replays exactly match qpos and commands |
| dev_a08_visual_clearance | done | 11/12; native29 + lift .12 + visible carry height correction; remaining seed108 entry timeout |
| contract_25hz_f32_a08 | done | 3/3 success; all three exact deterministic replays, qpos/command error=0 |
| reach_envelope_mode2 | done | bottle-facing near/far pitch20–33 deg; fixed low table inherently favors bending |
| arm_ik_envelope | done, diagnostic | strict feedforward pose target residuals; not proof of global infeasibility |
| reach_envelope_high_table | done | unloaded higher target yields pitch7–21 deg; lateral tracking bias4–6 cm needs correction |
| dev_a09_tracking_bias | done | 12/12 low-near success with bounded steady-entry correction, 25 Hz f32 |
| dev_a10_reach_strata | done | 8/12; all four high-table cases invalid geometry, low/deep 8/8; one wrist step2.031 cm |
| dev_a11_pre_alignment | done | 3/12; invalid high geometry, low-near pre alignment stalled, one deep knock |

| dev_a12_scene_repair | done | 9/12: high near2/4 (pitch24–27 deg, not upright), low near4/4, deep3/4; deep failure from route switching near corner |
| dev_a13_stable_route | done | 8/12: low-near/deep8/8, high-near0/4 reach timeout with closer stand; approach edge latched from initial heading |

| mode6_fk_feasibility | done | mode6 wrist error4.9–8.4cm vs mode2 3.1–3.4; no reason to switch. Valid mode2 high-near reference nearly upright |
| dev_a14_direct_reference | done | low/deep8/8, high-near0/4; remaining high failure is cross-body pre-reference caused by body facing nominal bottle |
| dev_a15_shoulder_alignment | done, selected for qualification | 12/12 task and audit passes; high upright, low appropriate bend; 12/12 exact executor replay |
| observation_contract_smoke | done | actual RGB/state observer through GR00T parser: state91, images [1,2,240,320,3] per camera |

## Recovery precautions

Do not infer GPU release from 0% utilization or the estimated 19:00 finish time.
Verify current STATE and explicit release before using it. This Goal is separate
from the SONIC 07d eval and must not alter its processes/files.


New C3 modules: visual_executor.py, visual_state.py, replay_visual_actions.py; exploratory configs/gr00t_bfm_visual_2cam.py (state91, action29, 0.4 s RGB history). Common-runtime baseline amendment registered in goal_plan before training. No GPU work.

New visual_camera.py shares current RGB + state91 and 0.4s image history for collection/inference; tests/test_bfm_visual_motion.py added. high-table collision-box regression is tests/test_bfm_visual_scene.py. Last CPU development source snapshots remain per-run.

Final candidate args: --pre-tolerance .035 --grip-kp 1 --grip-kd .025 --native-codec --lift-height .12 --carry-clearance .12 --action-hz 25 --integral-gain 1 --strata high_near,low_near,low_deep --edge-clearance .22 --high-table-front .20 --reference-style direct --facing edge. Add --record-observations for raw student state/actions/videos.

Qualification RGB split: qual_a15 contains3 completed recorded cases and an intentionally interrupted incomplete1104; qual_a15_remaining preserves original assignment via --case-offset3. Raw student recorder/exporter paths: visual_recording.py, export_visual_dataset.py. Recovery module visual_recovery.py is assessment/reset-only; policy sees no intervention flag.

## C3 integration update (19:20 UTC)

- qualification_replay10.json:10/10 exact, every qpos/command error0.
- gpu0_executor_replay.json:3/3 exact; contract_gpu0_a15 audit3/3.
- New eval_visual_vla.py uses actual RobotCameraObserver + ChunkScheduler +
  ActionExecutor; first capturetick80, first delayed executiontick84,0.4s/80ms.
- chunk_replay_sim_env:3/3 exact including delayed chunk switching and settle.
- chunk_replay_gr00t_env:3/3 task success, tiny nonzero qpos errors3e-5..1.6e-4.
  Sim venv torch2.13cu130, GR00T venv2.9cu128, other numpy/mujoco/scipy identical.
  Formal collection AND VLA inference will use GR00T venv to remove this variable.
  Need fresh GR00T-env3case recording and same-envchunkreplay before C4.
- 32 unittests passed (unittest_c3.log). Optimizer corrected to adamw_bnb_8bit;
  train_config_smoke_v2 dryrun exists. No actual optimizer steps.
- New visual_baseline.py adapts frozen task generator through common runtime;
  common settle1.68s and left jaw held open are explicit runtime differences.
- New collect_visual_cohort.py: fixed attempt pools, durable manifests, A/B/C,
  C reuses B150normal+50newrecovery, val15+5. Baseline common pilot running
  session89882 in GR00T venv/GPU0,12freshseeds1401..1412; inspect its log/results.
- Next: finish same-GR00Tenvcontract; memory preflight model+4-imagehistory;
  freeze pool/runtime in plan, then launch bounded formal collection.

## C4 started

- contract_gr00t_a15:3/3 audit accepted; chunk_replay_gr00t_same3/3 exact.
- baseline_common_pilot complete12/12 allstrata4/4, wall63s.
- model_memory_preflight PASSED realbatch4 forward/backward, finite loss1.5023,
  allocated19.43GB/reserved20.63GB, 8bitoptimizerestimated3.24GB, nooptimizersteps.
  Batch4 xacc16 retained; no hardware-driven protocol change needed.
- Reviewed posture_contact_sheet.png under contract_gr00t_a15: hightable upright,
  lowtable/deep forwardbend and recoveruprightafterlift; fullbodyimagesconsistent
  with auditedangles.
- Formal cohort_B launched in GR00Tenv/GPU0, log cohort_B.log. Seeds/pools and
  runtime frozen in plan; source snapshots in cohort_B/source. Do not edit its
  frozen dependencies during collection or restart completed attempts.
- Next afterB complete: collectA throughfrozenbaseline and C recovery; export
  matched200/20 datasets, generate stats, trainseed17/29 final3000/arm.

## Automatic sequential driver

run_visual_comparison.py running session99188, log pipeline.log; durable
pipeline_state.json is the authoritative stage/PID/command/returncode ledger.
It adopts the already-runningBcollector (session19121), then A,C, exports/stats,
freezes6trainingconfigs, runs A17/B17/C17/A29/B29/C29 final3000 with60-case
selection each, locks winner, confirms winner/A2seeds100cases. Stagefailures stop
the driver and require diagnosis; it refuses to repeat a running/failed stage.
No training budget has yet been spent. Do not launch another driver.

Processor-only actual-batch contract passed:4samples ->16images, state padded132
with91configured inputs, action40x132 with29configured outputs, attention370tokens.
Original preflight batch_shapes.json was only a BatchFeature type string; the
correct actual shape evidence is processor_batch_contract/result.json.
Training retention save_total_limit changed2->1 before anytraining, preserving
optimizer/RNG checkpoints every500 and final3000 while limiting disk use.

Evaluation now includes last12cases as fixed recoverytests within60/100 existing
budgets (48/88normal); reset/environment mutation never enters VLAobservations.
VLA recoverystart based on ownarmFKextension, notexpertphase. Formal runtime
freeze and seedassignment documented in goal_plan. validation_diagnostic samples
3fixedobservations/episode on20heldoutval episodes, allow_padding=True, reports
per-dimensionMAE but never selects checkpoints orwinner.

Next useful work whileBcollects: review implementation, especially evaluation
causality/recovery and formal-selection/accounting. Do not edit frozen collector
dependencies in cohort_B/source during active collection.
