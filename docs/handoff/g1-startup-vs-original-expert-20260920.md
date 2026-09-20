# Why the ghost startup differs from the original bimanual expert

Read-only comparison following the user's report after the corrected-reset repeat. No production code, controller parameters, or physical trials were changed in this investigation.

## Establish the actual original owner

`configs/experiments/bimanual_20ep_20260919.yaml` selects `expert: visual` and `reach_control: whole_body`. `tasks/crate/prepare.py` maps this to `tasks/crate/expert.py`. The relevant original routine is `reset_ready(..., whole_body=True)`, not an assumed generic carry initializer.

## Confirmed sequence differences

| Stage | Original whole-body visual expert | Current ghost reach |
| --- | --- | --- |
| Per-session preparation | `round_runner.prime`: reset, start control, 2.5s IDLE | No equivalent session prime |
| Before per-case reset | Exit POSE, 1.5s IDLE using current facing; clear sim pins | Exit POSE, 0.2s IDLE using +X; enable ready waist/arm PD pins |
| Physical reset | First placement reset into simulator defaults | Reset directly into ready upper joints; corrected full29 vector |
| After start/reference readiness | 1.5s IDLE, reanchor, then 3s IDLE | 1.5s IDLE and reanchor; no subsequent 3s recovery |
| Establish ready reference | `PlannerArm.start()` with observed pose and default0.6s settle;125-frame smoothstep to ready (~2.5s at50Hz), then1.2s hold | New runtime calls `PlannerArm.start(explicit ready, settle_seconds=0)` and pushes ready once |
| Sim pin transition | Original `pin_arms()` has no arguments and releases pins; no explicit ready-PD pin takeover | Explicit waist/arm pins are released immediately after runtime construction |
| Placement/admission | Second reset under an already-established ready reference, then continuously publish ready until measured quiet | Single reset; measured quiet only after shortened preparation and pin release |
| Quiet thresholds | Height≥0.72m; configured upright tolerance (12degrees in formal config); speed<0.06m/s | Height≥0.65m, upright tolerance15degrees, speed<0.06m/s |
| Video/dataset begins | After all of `reset_ready` and admission, then initial-distance check | Immediately after the physical reset, before start-control/reanchor/pin release |

Owners: `expert.py:43–118,149–195,556–560`; `round_runner.py:50–57`; `ghost_reach_execution_cli.py:194–262`; `box_rim_execution.py:89–140`.

The `reanchor` implementation (`backends/sonic/streamed_arm.py:933`) explicitly documents that its mode-toggle twitch occurs before recording. The ghost entry records that operation. This is a certain explanation for showing startup movements that the old videos excluded; it does **not** establish that the old unrecorded startup never moved, and does **not** dismiss the new startup fall as a presentation issue.

## What the current evidence proves

The corrected-repeat audit proves all14 first physical states now have the intended waist, elbow and default leg values. Nevertheless case1 falls before admission, and cases1,2,4,6,12 exceed30mm initialization foot displacement. Therefore the former17/29-value reset mistake is no longer the explanation for this repeat.

A physical ready pose is not an established controller ready state. The new path changes physical joints while retaining the free-running controller's history/reference and adds a strong PD pin/release transition; it omits the original preparation and smooth reference transitions. These are substantive migration differences, not just different grasp targets.

Both the old and new ordinary reset paths remain asynchronous: simulator reset changes MuJoCo state; `AgentCommander.reset_pose()` only clears Python command caches; `start_control()`/`exit_streamed()` send planner-ownership commands. Neither invokes the C++ lockstep `ResetLockstepSession`, which explicitly clears observation/action history and token/input state. Do not claim that the original expert had an atomic controller reset that the new path omitted.

The original videos begin after admission. The inspected formal `visual_crate_s7000_00` trace starts at seq2478 with root height0.789684m and speed0.04091m/s; it cannot reveal earlier reset transients. Pre-record failures may have zero frames/no video. Consequently, comparing video frame0 alone is not a like-for-like startup test.

The strongest candidates for the remaining physical instability are shortened controller/reference preparation, control-mode transitions while not settled, and the added pin/release sequence. Existing recordings do not contain synchronized pre-reset command, reference, controller-history and acknowledgement traces, so they do not uniquely attribute the fall to one of those components.

## Correct migration boundary

The earlier migration reused `CoordinatedReach`, but did not reuse the complete initialization workflow. The next implementation should share the original controller preparation/ready owner rather than maintain a second shortened initializer. Scene placement can be injected independently so the floor-only test needs no crate and no walking.

Preserve preparation order, smooth ready interpolation and continuous ownership/reference handoff. Keep initialization diagnostic recording from reset; begin the main reach/dataset clip after admission. Evaluate initialization foot movement and the ready transition explicitly instead of accepting eventual quiet and clearing the earlier drift baseline. Source-compatible reuse of the original flow is the starting point; complete physical stability still requires a dedicated reset-to-ready test and cannot be claimed from static pose equality or successful reach alone.
