# G1 startup jump: wrong reset vector and missed initialization checks

Branch `main`; prior dirty work preserved. This follows the user's report that the 14-case ghost reach videos again show violent startup motion. No new physical simulation or GPU trial was run for this investigation.

## Proven cause in this batch

`ghost_reach_execution_cli._initialize_trial` sent `robot.dof=ready[12:]` (17 upper-body values), with `dof_from=12`. The reset consumer `EpisodeControlServer._apply` expects all **29** joint values; `dof_from` only restricts which entries of that full vector are applied. It does not offset a short vector.

The former consumer zipped the vector against full joint addresses and then filtered `i >= dof_from`. Thus only the last five values of the short array were applied to full-body joints 12–16. The right elbow's intended `0.8 rad` became **waist_roll=0.8 rad (45.8 degrees)**; shoulders and elbows remained at reset defaults, including **both elbows=0** instead of 0.8.

This is not an inference from the picture: CPU execution of the actual reset consumer on the run's real model, with its WBC default configuration, reproduced all 29 joints and pelvis height of **all 14 native first frames exactly**. Evidence:

- `outputs/g1_ghost_reach_20260920/startup_diagnosis/reset_consumer_reproduction.json` (pre-fix consumer; full29 control has zero upper-body error).
- `startup_diagnosis/native_reset_match.json` (all14 zero residual).
- `startup_diagnosis/native_jump_metrics.json` (time-resolved physical displacement).
- `startup_diagnosis/startup_jump_evidence.png` (cases1/2 joint and pelvis plots).

The arm/waist “pin” is a stiff PD motor target, **not an atomic write of physical joint position**. `run_sim_locomani.py` uses KP120, KD3 and a 30 Nm torque cap. The wrong reset therefore starts roughly 46 degrees away from the intended waist posture; that PD target and the balance controller rapidly pull the body back. At zero initial velocity the erroneous 0.8 rad waist/elbow gaps exceed the pin torque cap. Actual sampled motion reaches **21.525 degrees in 20 ms** (case1 hip yaw), with most other cases showing roughly **19 degrees of elbow movement in 20 ms** immediately after reset.

## Why earlier fixes and checks did not prevent it

1. Prior Planner→POSE continuity fixes and whole-body reference interpolation operate after initialization. The new ghost initializer is a separate caller and introduced a malformed reset request before those mechanisms run. The box initializer already sent a full29 vector.
2. The reset boundary documented 29 entries but did not enforce length. Python `zip` silently truncated instead of rejecting the incompatible request.
3. The ghost tests covered reach criteria, fixed targets, no-walk commands and recordings. They did not exercise the new initializer through the actual reset consumer. Passing those tests did not demonstrate correct startup posture.
4. Quiet standing was checked only after the initial holds and pin release. It could admit a robot that had already moved violently and then recovered. Reach foot displacement was measured from that later admitted state, hiding the earlier movement.

The zero leg entries in the initializer's planner reference are misleading but are not a second demonstrated leg command in this path: PlannerArm publishes upper-body values; the POSE takeover reads the full-body planner packet and overlays upper-body ownership. Cross-process reset acknowledgement/deploy-history reseeding remain separate previously recorded gaps, not proven to be the primary cause of this deterministic wrong waist pose.

## Correction to the previous experiment claim

The earlier **6/14 successes / 8 timeouts** remain reproducible outcomes of the **post-initialization reach stage**, including the independent native 50 Hz FK audit. They do **not** certify an end-to-end stationary experiment.

Case1 moved its pelvis horizontally by up to **0.8185 m** and a foot by up to **0.8758 m** during initialization. Across all14, the maximum initialization foot displacement ranged from **0.0427 to 0.8758 m**, exceeding the intended 30 mm stationary criterion in every trial. The earlier “feet remained within the limit” statement applied only to the later reach stage. Preserve these videos as diagnostic evidence; do not use this batch as a clean standardized startup/reachability baseline.

## Local correction and validation scope

The local fix sends the full29 reference and retains `dof_from=12`, preserving the simulator's standing legs. The reset consumer now validates the full-vector shape, finite values and index before appearance/reset/state mutation; malformed reset messages are rejected without killing the polling loop. Regression coverage must call the actual initializer and actual reset joint-writing logic, asserting neutral waist, ready elbows, unchanged default legs, and no mutation on malformed requests.

Validation completed: 25 focused tests passed (`test_episode_reset.py`, `test_ghost_reach_execution.py`, `test_box_rim_initialization.py`, `test_streamed_handoff.py`); `make check` exited 0. Record: `startup_diagnosis/verification.json`. An additional CPU-only run through the corrected consumer on the actual saved physical scene confirms waist `[0,0,0]`, both elbows `0.8`, preserved default legs, and rejection of the old 17-value request without modifying qpos/qvel (`fixed_reset_real_model_verification.json`). Documentation checks passed after this note was added. This is a CPU-verified correction of the malformed reset contract; it is **not** a claim that every asynchronous reset/control-history transient has been eliminated. No fixed-code physical rerun has yet been collected. Future admission needs to cover the complete reset-to-ownership interval, with fresh reset state and controller/reference synchronization, instead of accepting only eventual quiet or resetting the drift baseline after a transient.
