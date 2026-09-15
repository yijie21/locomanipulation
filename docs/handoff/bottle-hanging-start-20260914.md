# Bottle continuous expert: hanging-arm start and no-retract lift — 2026-09-14

Scope: the user asked that the single-arm episode start with the arms hanging, as the crate
line does, and raise the arm gradually instead of the recorded raise-then-drop. Usage contract:
[单臂采集](../collection.md). Baseline `fcf23d9`; no deploy binary, weights, scene, case bank or
gate thresholds changed. Fixed cases `outputs/bottle_continuous_20260914/cases_fixed10.json`.

## Cause

`_phase_spawn_and_settle` primed both arms to READY through `prepare_ready` before the spawn
gates, then `_phase_navigate` seeded the continuous takeover from the deploy's live reference
(`upper17()`), which still held the hanging arms. Every trace of
`outputs/bottle_direct_repeat_20260914b` shows the right elbow going 1.06 → −0.2 rad at 0.5 s
and back to 1.0 at 1.5 s before `WalkingReach` raised it again during the walk (9/9 rounds).

`prepare_ready` did a second job nobody had written down: it brought the waist from the
deploy's engage lean (0.14 rad forward) to neutral before the ego view gate.

## Implementation

- `ready_grasp_policy.ready_target(seed, arms=)` and `prepare_ready(p, arms=, waist_tolerance=)`:
  with `arms=False` the target keeps the seed's arms and neutralises the waist only, and the
  measured gate is the waist within 0.03 rad instead of the arms within 0.30 rad.
- `round_phases._phase_spawn_and_settle`: the continuous expert (`prepares_during_navigation`)
  calls `prepare_ready(p, arms=False)`; other strategies are unchanged. Failure string
  `[stand] neutral waist not delivered`.
- Tests: `test_ready_vla_handover` (whole_body prepares only the waist),
  `test_ready_grasp_policy` (`ready_target`). Docs: `docs/collection.md`.

## Batches (cases 31000:0–9, instance 0 / GPU 0, all attempts retained, previews ≤500,000 bytes)

| Batch | Code | Result | Notes |
|---|---|---|---|
| `outputs/bottle_direct_20260914/final10` | `fcf23d9` | 6/10 | previous handoff |
| `outputs/bottle_direct_repeat_20260914b` | `fcf23d9` | 4/10 | reference for this change; raise-then-drop 9/9; one observer clock gap, one reset teleport miss |
| `outputs/bottle_hanging_start_20260914` | READY skipped entirely | 2/10 | 6/10 rejected at spawn: object below the ego view gate (cam y 0.92–0.96 vs 0.90) because the waist kept its lean |
| `outputs/bottle_hanging_start_20260914b` | waist only, 0.30 rad gate | 3/10 | still 6/10 view rejections: the gate admitted before the waist had moved (0.14 → 0.10 at 1 s) |
| `outputs/bottle_hanging_start_20260914c` | waist only, 0.03 rad gate | 6/10 | no view rejections; raise-then-drop 0/9; waist at 0.01 by 0.5 s; failures: wrist/bottle clearance (2), hip/table clearance (9), no bilateral contact (3), walked 0.63 m off the spawn while standing (4) |

Ten rounds cannot resolve a success rate on this line; what the batches establish is that the
recorded start is now monotone (elbow 1.07 → 0.2 rad over the first 3 s of walking, no
intermediate READY) and that the view gate passes again.

## Remaining

- After `prepare_ready` returns, the takeover seeds from the deploy's reference, which still
  reads the 0.14 rad lean; the waist rebounds to ~0.10–0.12 rad at 1 s and `WalkingReach`
  brings it back to neutral by ~2.5 s. A 6° waist bob at the start of the walk; seeding the
  takeover from the ramp's settled command would remove it but changes the "start from the
  consumed reference" decision of the previous handoff.
- Grasp-stage clearance rejections (`transport_guard` predictions on the candidate pose) remain
  the dominant failure across all four batches; they are gate decisions, not measured contacts.

## Second change: lift without retracting the arm

The user asked that the arm lift the bottle to a height and stop, instead of folding into the
awkward pose seen after every success. Cause: `lift_and_recover` held a **world** hand target
(capture point + 0.16 m + a 6 cm retract toward the body) while the torso straightened, so the
shoulder moved up and back and the arm folded to keep the hand there. At the terminal hold of
`outputs/bottle_hanging_start_20260914c` the right shoulder pitch was 0.75–0.92 rad, shoulder
yaw −0.5 to −0.9 rad, wrist yaw ≈1 rad, waist pitched back 0.2 rad.

Implementation (`whole_body_policy.py`): `torso_offset` / `lift_target` express the capture
hand point in the commanded torso frame; each tick the target is that offset in the current
reference torso plus `reach.whole_body.lift_height` (new typed field, 0.16 m) in world z. The
retract is gone. The torso recovery itself now raises the bottle further (total rise at
`loaded_upright` 0.30–0.42 m against 0.13–0.18 before). Test: the target keeps its torso-frame
offset when the reference waist pitch changes.

| Batch | Result | Terminal-hold right arm |
|---|---|---|
| `outputs/bottle_hanging_start_20260914c` (retract) | 6/10 | shoulder pitch 0.75–0.92, shoulder yaw −0.5…−0.9, waist −0.2 |
| `outputs/bottle_no_retract_20260914` | 7/10 | shoulder pitch −0.64…0.44, waist ≈0; arm extended forward, bottle at chest height |

Failures in the new batch are all transport-guard clearance predictions before the grasp
(hip/table 3, finger/bottle 4, wrist/bottle 6), unrelated to the lift. Ten rounds do not
resolve the rate. The lift height is a knob: with the torso adding ~0.2 m, a lower
`lift_height` gives a lower hold if the extended arm proves to be a walking disturbance on the
0.35 m back-off (no fall in 7/7 here).

## Third change: transport-guard semantics at handoff and during the lift

The 200-episode sampled collection (`outputs/bottle_vla200_20260914`, seed 30000, stopped by
the user at 29/81) ran at 36% against 57% on the fixed bank the same day. Not the
distribution (planned stand identical in every round: 0.331 m, −36.7°; spawn, placement and
bottle size ranges match the bank), not contention (34–37 s per successful round vs 33.5 s;
the foreign GPU-0 job started after most batches). 46 of the 52 failures were
`transport_guard` predictions:

| Family | n | When | Park (m) |
|---|---|---|---|
| lift: `right_wrist_yaw_link` vs bottle | 21 | lift progress ≈0.27, bottle already 8 cm (2.4–41) off the table | 0.43–0.54, same as successes |
| handoff: `right_dex1_finger_1_link` vs bottle | 12 | brake → reach transition, before any reach sample | 0.37–0.44 |
| handoff: `right_hip_roll_link` vs table | 12 | same | 0.41–0.48 |

The first family aborted rounds with the bottle in hand: the guard only allowed *finger*
contact with the target, so the wrist link near the held bottle counted as a collision. The
no-retract lift (hand riding the torso) plausibly raises that rate (1/10 on the fixed bank,
26% here) but ten-round batches cannot show it. The second family is a prepared hand already
at the bottle's surface when the walk parked under 0.44 m, which is the intended arrival.

Implementation: `TransportGuard.check(..., ignore_target=)` drops the target from one side's
obstacles; `WholeBodyGraspPolicy._clear(..., attached=)` passes it for the right arm during
`lift`; the walk-tick handoff check now allows finger contact (`contact=True`). Table and
body checks are unchanged everywhere; lift height, hold and slip acceptance are unchanged.
Tests: guard (attached target ignored, other obstacles still hit) and policy call semantics.

Validation (GPU 0 / instance 0, after the foreign job released it), failures recounted by
phase × link:

| Run | Before | After |
|---|---|---|
| sampled, seed 30000, 20 rounds (`outputs/bottle_guard_sampled20_20260914`) | 8/20: guard@lift wrist/bottle 8, guard@handoff hip/table 2, finger/bottle 1, foot drift 1 | 15/20: guard@handoff hip/table 1, hold slip 3.4 and 4.7 cm, recovery slip 1, reset teleport miss 1 |
| fixed 10 cases (`outputs/bottle_guard_fixed10_20260914`) | 7/10 (no-retract batch) | 6/10: guard@handoff hip/table 2, hold slip 111 cm (bottle dropped after back-off), recovery slip 1 |

The two targeted families are gone; the rounds they used to abort mostly complete. What the
change exposes is real: with the torso-anchored lift the bottle rises 35–41 cm in the 3 s
ramp (about 13 cm/s, against 13–18 cm total in the retract version) and the waist ends 9–14°
behind vertical; the object-to-tool offset starts drifting at lift progress ≈0.86 in the
slipped rounds. Same seeds are not the same gait, so 8 → 15 is indicative, not paired.
Next lever, not applied: a lower `reach.whole_body.lift_height` and a longer lift ramp
(the 3 s is a literal in `whole_body_policy.py`), then the slip count on the same seeds.

## Fourth change: lift height/duration, failure handback, and the finger pads

Same 20 sampled seeds (30000) after the guard change, run in sequence:

| Run | Change | Result | Failure families |
|---|---|---|---|
| `outputs/bottle_lift_sampled20_20260914` | `lift_height` 0.16 → 0.06 m, new `lift_seconds` (3 → 5 s) | 11/20 | 3 drops at the end of the lift, 1 hip/table, 1 disturbed, **4 consecutive reset misses** (rounds 6–9) |
| `outputs/bottle_lift2_sampled20_20260914` | + failure path calls `arm.stop(preserve_reference=True)` | 12/20 | 4 hold slips (2 drops), 1 recovery slip, 2 finger/bottle guard hits in reach/close, 1 disturbed; no stand-stage failures |
| `outputs/bottle_lift3_sampled20_20260914` | + wait for a still reference before the planner switch (reverted) | 0/10, stopped | the reference never drops under 0.15 rad/s; the policy waited 8–14 s in the lift and the bottle crept out in 9/10 |
| `outputs/bottle_hardpads_sampled20_20260914` | scene `outputs/scene/sim_hardpads` (`build_sim_scene.py --harden-pads`, level `object`) | **15/20** | 3 hip/table at handoff (park 0.42–0.44), 1 finger/table in the lift, 1 no bilateral contact; **0 slips** |

What each step established:

- The reset-miss cluster (every stand-stage failure today followed a whole-body abort, four in
  one unbroken run) was the failure path leaving the deploy in streamed mode; `execute()` now
  pins the stopped reference and hands the same pose back to the planner on failure too.
- Slower and lower lifting did not stop the slides: the object-to-tool offset starts drifting
  in the last second of the lift, 0–0.8 s before the planner switch, and a held bottle creeps
  out even while the reference is held still (run 3). The production bottle scene has the
  MuJoCo default pads (priority 0, condim 3, solref 0.02, solimp 0.9/0.95); the crate scene has
  been hardened since 2026-09-11 and `build_sim_scene.py` ships an `object` level calibrated on
  a 0.3 kg bottle (24.7 s carry, 0.8 cm drift) that was "off by default so the gate baselines
  keep their physics". With it, the same 20 seeds go from 5 slips to none.
- The still-reference switch is not usable on this line: the supported reference keeps moving
  at 0.34–0.7 rad/s after the lift completes, so the condition never fires. Reverted; the
  0.8 m/s switch jolt remains and is tolerated by the hardened grip.

Decision left open: making `sim_hardpads` the bottle collection scene changes contact physics
against every earlier bottle dataset and invalidates the fixed case bank's `scene_sha256`.
The remaining failures (hip/table at near parks, 3/20) are the stand-scatter family, not the grasp.

## Fifth step: 200-episode collection, training and 100-round closed loop (2026-09-14/15)

All on GPU 0 / instance 0, `outputs/bottle_vla200b_20260914`, hardened-pad default scene,
`run_learn_pilot` stages collect → convert → train → eval:

| Stage | Result |
|---|---|
| Collect | 272 attempts, 200 successes (74%), seed 30000, 28 batches. Failures: 27 hip/table clearance at handoff, 19 no bilateral contact, 8 disturbed target, 6 slipped in recovery, 3 finger/table, 2 wrist/bottle, 2 foot drift, 1 ego view at stand |
| Convert | `train_dataset` 190 episodes / 93,624 frames, `val_dataset` 10 / 4,963, 25 Hz, verification tails trimmed; val audit's alignment-lag check failed and was overridden by hand (`convert.done` written, see the stage note in `audit_val.log`) |
| Train | `checkpoint-10000`, 4.2 h, mean train loss 0.087, last logged 0.043 |
| Eval | **17/100** (Wilson 95% 0.11–0.26), seed 6000, async client, horizon 16, 90 s budget, `whole_body_initial_token.json` prelude, `closed_loop.json` in `eval/` |

Eval failure families (from `eval/closed_loop.manifest.jsonl`):

| Family | Rounds | Note |
|---|---|---|
| `[vla] reach: at the table but never commanded a close` | 60 | closest approach median 0.32 m (0.21–0.58); the arm never fires the jaw |
| `[view] object not in ego view at spawn` | 11 | pre-VLA discard, spawn-side |
| `[vla] close: commanded a grasp but no lift` | 7 | |
| `[vla] walk: never closer than 0.60–0.86 m` | 3 | |
| `[stand]` (reset unstable, teleport 0.11 m off) | 2 | |

Successful rounds take a median 63 s. This is the same absorbing state `visible200`
(`outputs/experiments/visible200_20260910`, 2/100 on 2026-09-10) showed: the
policy stands at the table and the close is never predicted, so the sampled expert's 74%
yield did not carry into the learned policy. All 87 overview videos are kept under
`eval/videos/`; ≤500,000-byte full-duration previews with `verification.json` are under
`eval/previews/`. Nothing here changes the crate line, which is evaluated separately in
[crate-formal-export](crate-formal-export-20260914.md).

## Sixth step: why the policy never closes, and the brake depth (2026-09-15)

Offline bisection of the 17/100 (`outputs/bottle_vla200b_20260914/eval_dump10`, 10 rounds re-run
with `backend.obs_dump`, 2/10): on val frames around the expert's close the checkpoint closes
10/10 episodes; on the failed rounds' live observations it closes 0/70 and replay reproduces the
live answer. Substituting keys from the success round's close observation: right-wrist image
alone 0/70, ego alone 0/70, **both images 58/70**; any state group 0/70. The reverse (success
observation with the failed round's images) drops 12/12 to 0/12, state swaps stay 12/12. The close
is a joint function of the two images; proprioception is irrelevant, so it is geometry, not timing.

The geometry: the expert brakes at pelvis-frame depth `visible_brake_depth` 0.40 m (park
0.46 m, p10–p90 0.42–0.52) and the whole-body phase then moves the pelvis a further 10.6 cm
(6–16 cm) before reaching. The failed eval rounds stop at 0.37–0.52 m, i.e. at the expert's
park, and never make that second move; successes stood at 0.28–0.31 m. The park frame is a
static picture whose next action is triggered by the expert's phase, not by anything visible.

Change: `configs/tasks/bottle_whole_body.json` `visible_brake_depth` 0.40 → 0.31 (the value every
other bottle profile uses; `visible_decel_band` 0.24 unchanged). Same 20 seeds (30000), hardened
pads, `outputs/bottle_brake031_sampled20_20260915`:

| | 0.40 (`bottle_hardpads_sampled20_20260914`) | 0.31 (`bottle_brake031_sampled20_20260915`) |
|---|---|---|
| Successes | 15/20 | **17/20** |
| Hip/table guard at handoff | 3 (parks 0.43–0.44) | 2 (park 0.37) |
| Other failures | finger/table 1, no bilateral contact 1 | wrist/bottle guard 1 |
| Park (pelvis–bottle) | 0.461 m (0.43–0.52) | 0.355 m (0.34–0.39) |
| Park → whole-body start shift | 10.6 cm (6–16) | 3.4 cm (0.6–5.7) |

Paired on batch 0's ten seeds: 7 both succeed, 1 each way, 1 both fail. The hip/table family did
not grow. Previews of all 20 rounds (≤500,000 bytes, `verification.json`) are under
`outputs/bottle_brake031_sampled20_20260915/previews/`. Whether the shorter park-to-reach gap
transfers to the policy needs a re-collection and closed loop; not run.

## Seventh step: the crate-style walking preparation does not fit this table (2026-09-15)

Asked why the hand rises to a fixed height during the walk and only then descends, and whether
the crate line's walking preparation (`crate/visual_approach.ApproachPreparation`: progress over
the last `walk_speed × 3 s`, hands blended toward a pose defined on the perceived rim) could be
borrowed. `walking_reach.py` clips its goal to a pelvis-frame box (x 0.16–0.26, y −0.26..−0.14,
z ≥ pelvis + 0.05); the grip point is 11–16 cm below the pelvis, so the +0.05 floor always wins
and the tool point sits at 0.86 m from the first plan, 16.6 cm above the grip; `_reach` then
slides 11 cm forward at that height and descends 17 cm (5.5 s).

Tried (`outputs/bottle_walkprep_sampled20_20260915`, same 20 seeds, brake 0.31, hardened pads):
progress = clip((brake + 0.9 − depth)/0.9), goal = initial tool point + smoothstep(progress) ×
(pre-entry − initial) with the pre-entry 12 cm behind the bottle along the heading, clearance
0.10 → 0.04 m above the grip, palm attitude slerped 75 % toward the entry attitude (a full turn is
arm-only unreachable at that height: IK 1.5–3 cm), IK seeded with the walking pose. Offline replay
of six recorded tracks: IK ≤ 0.6 cm. Result **2/20**: 8 rounds aborted during the walk at progress
0.94–1.0 by the transport guard, `left_dex1_finger_2_link` × 4 and `right_dex1_finger_1_link` × 4
against the table top, 1 disturbed, 1 no contact.

Replaying the guard on the recorded failure states settles it (sweep fraction 0, i.e. the measured
pose itself is in contact):

- The dex1 finger collision mesh reaches **19 cm below the palm** when the fingers point down. The
  hanging palm is at 0.72 m, so the hanging fingertips are at 0.53 m, under the 0.62 m table top.
  The crate table is 0.30 m, which is why hanging hands are harmless there.
- Left hand: yesterday's run raised the left arm from hanging (elbow 1.02) to READY (elbow −0.3)
  during the walk as a side effect of `WalkingReach` seeding the IK with `ready_q()` and returning
  the whole solved vector. Seeding with the walking pose left it hanging, and it met the table edge
  in 4/10 rounds. That side effect is what keeps the left hand clear today.
- Right hand: the commanded reference at the failures had the tool point at −0.17 m (pelvis frame,
  0.61 m world) with the fingers still pointing down (x-axis z −0.94), not the +0.04 m above the grip
  that was solved for. The joint-space slew between "hanging, fingers down" and "forward, fingers
  turned" passes through palm-at-0.75 / fingers-down, and the weak rotation weight (0.2) lets the
  solver keep the fingers down at low heights. Measured lag was not the cause (reference vs measured
  tool dz ≤ 1.8 cm).

So the fixed height is a clearance: with fingers that long, the palm must be ≥ table + 0.19 + margin
≈ 0.84 m before the hand is over the table, or the fingers must already be horizontal, and a
fingers-forward pose at grip height is outside the arm-only workspace (the supported reach reaches
it with the waist and the lean). Reverted `walking_reach.py`, the four config fields and the test;
the run, its videos and manifests are kept. Ways to actually remove the rise, none tried: a small
waist envelope during the walk (the crate allows ±0.1 rad), turning the wrist early with a strong
rotation weight at a height where it is feasible and only then lowering, or a lower table.

## Eighth step: walking preparation with a waist envelope (2026-09-15)

Three attempts on the same 20 seeds (30000), brake 0.31, hardened pads; all runs, videos and
manifests kept. Baseline for comparison: `outputs/bottle_brake031_sampled20_20260915`, 17/20.

| Run | Walking hand | Result | Failures |
|---|---|---|---|
| `bottle_walkprep_sampled20_20260915` (v1, seventh step) | progress-blended low pre-entry from the hanging pose, arm only, 0.45 rad/s | 2/20 | 8 finger/table during the walk |
| `bottle_walkprep2_sampled20_20260915` (v2) | wrist turned from the first plan, tool 6 cm above the grip, pre-entry "12 cm behind the bottle" (filtered), waist envelope, 0.7 rad/s | 0/20 | 5 toppled in the settle, 6 wrist/finger vs bottle, 4 disturbed, 2 hip/table, 2 finger/table, 1 view |
| `bottle_walkprep3_sampled20_20260915` (v3, current code) | as v2 but x is the pelvis-frame constant `HAND_X` 0.08 with a "≥ entry gap behind the RAW bottle point" bound | **14/20** | 5 hip/table at the brake (parks 0.31–0.39), 1 wrist/bottle |

What each run showed:

- v1: the 0.45 rad/s walking slew cannot turn the wrist (~1.3 rad) inside a 3.5 s walk; the sent
  reference trailed the IK solution by 30 cm and passed through palm-0.75 / fingers-down. The
  crate's walking preparation runs at `joint_speed` 0.7, now `walk_joint_speed`.
- v2: a pre-entry defined relative to an approaching bottle moves backward in the pelvis frame at
  base speed, faster than the arm retreats; the sent hand sat at the bottle's depth at bottle height
  and its lateral sweep toppled it. The crate never chases in x: its hands are clamped at
  `stand_off − walk_brake_margin`. The filtered relative also under-reads depth by ~0.1 m while
  braking (velocity prediction), so a bound on it pushes the goal to an unreachable x.
- v3: the motion asked for is delivered. At the reach start the tool point is 4.8 cm above the grip
  and 15.6 cm behind it (baseline 16.3 cm above, 7.4 cm behind); the right palm is at 0.761 m
  instead of 0.846 m; the reach takes 3.4 s instead of 5.1 s and is a slide-in with a 5 cm
  descent. Success-round time 35 s vs 36 s. Paired: 13 both succeed, 4 baseline-only (all
  hip/table), 1 v3-only (hip/table), 2 both fail.
- The cost is the hip/table family (5 vs 2). Stop depth and edge distance are unchanged (stopped x
  median 0.295 vs 0.300, edge 0.270 vs 0.275). What differs is the waist: the walking IK saturates
  the envelope's roll at +0.10 in every round (measured 0.07–0.12 at the brake; baseline −0.02), the
  pelvis rolls to −0.06..−0.11, and the right hip roll link meets the table top on the near parks.
  Offline, removing roll from the envelope (yaw ±0.1, pitch 0–0.12) keeps the walking pose within
  0.2–1.7 cm of IK error, so a roll-free envelope is the next single change to test; not run.

Previews of the v3 rounds (≤500,000 bytes, `verification.json`) are under
`outputs/bottle_walkprep3_sampled20_20260915/previews/`. The v3 code is what the tree carries
(`walking_reach.py`, fields `walk_clearance`, `walk_joint_speed`); the baseline behaviour is
commit `fcf23d9`'s `walking_reach.py`.

## Ninth step: where the shake comes from, and three smoothing variants (2026-09-15)

Method: the sim's 50 Hz state trace (`collect/state_trace_NN/<label>_state_trace.npz`, `qvel`)
and the journaled references (`whole_body.references`, `seq`-aligned), per phase. Metrics: RMS
joint acceleration, velocity sign reversals per second, share of velocity power above 2 Hz.
Correction to the first reading: the journal keeps two frames per POSE send, the earliest
history tick and the 50-tick future endpoint of `PosePreview`; the "sent reference" numbers first
quoted (6.5–9.9 rad/s², ~24 % > 2 Hz) were the future endpoint. Tracked reference (frame 0) vs
body, v3 control, medians over successful rounds:

| phase | tracked reference acc / > 2 Hz | body right arm acc / > 2 Hz | body waist acc |
|---|---|---|---|
| walk | 0.4 rad/s² / 0.4 % | 6.0 / 31 % (8 reversals/s) | 8.6 (16 reversals/s) |
| reach | 1.2 / 3 % | 3.4 / 33 % | 2.5 |
| close | 1.3 / 14 % | 3.0 / 44 % | 2.3 |
| lift | 1.5 / 3 % | 4.0 / 24 % | 3.8 |

So in every phase the body is rougher than what it is told to track. The walk is SONIC's gait
(the crate's walking third reads the same: waist 8.1 rad/s², 72 % > 2 Hz). In the manipulation
phases the tracked stream is smooth and the roughness is the deploy's tracking plus contact; the
preview's future endpoint is rough (20–38 % > 2 Hz) because it extrapolates the per-tick velocity.

Three config-gated variants on v3 (`outputs/bottle_smooth_*_20260915`, same 20 seeds, previews
under each run's `previews/`, configs in `outputs/bottle_smooth_20260915/task_*.json`):

| variant | field | result | body reach arm acc / > 2 Hz / rev | body lift arm acc | failures |
|---|---|---|---|---|---|
| v3 control | — | 14/20 | 3.40 / 0.33 / 1.27 | 4.03 | 5 hip/table at the brake, 1 wrist/bottle |
| A low-pass | `pose_lowpass_hz` 4 | 7/20 | 2.94 / 0.30 / 2.41 | 4.51 | 9 hip/table at the brake, 4 no bilateral contact |
| B feedback gain | `tracking_gain` 0.3 | 6/20 | 3.03 / 0.24 / 0.86 | 3.34 | 7 reach timed out, 3 no contact, 3 disturbed, 1 foot drift |
| C min-jerk reach | `reach_seconds` 3 | 6/20 | 3.43 / 0.33 / 1.06 | 3.87 | 5 disturbed, 4 wrist/bottle, 1 finger/bottle, 3 hip/table |

- A: the 4 Hz second-order lag (~80 ms) leaves the hand drifting when the reach-to-close switch
  freezes the target at the measured hand; and deriving the velocity from the filtered stream made
  the preview's future endpoint four times rougher (81 % > 2 Hz). Body reach slightly smoother,
  lift rougher. Not adoptable as implemented.
- B: the only variant that smooths the body in every phase (reach > 2 Hz 0.24 vs 0.33, reversals
  0.86 vs 1.27, lift acc 3.3 vs 4.0), so the 0.4 s feedback is a real jitter source; but the
  feedback is what closes the decoder's tracking bias, and at 0.3 the hand stops short of the
  ready geometry (reach 5.5 s, 7 timeouts). A gain between 0.3 and 1, or a slower feedback filter
  at gain 1, is the untested middle.
- C: reach 2.5 s instead of 3.5 s, body not smoother; the path ends exactly at the grip point
  whereas `entry_step` ends 1.8 cm behind it and lets the readiness check trigger the close, so
  the wrist/fingers ran into the bottle (5 guard hits, 5 disturbed). Fixable (end the path at
  target − gap), not rerun.
- The brake-time hip/table family swings 0–9 per 20 across runs with identical walking code
  (A 9, B 0, C 3, v3 5): v3's saturated waist roll makes it gait-phase dependent, and it makes
  20-round success comparisons between these variants unreadable.

All three fields default to off; the tree still runs v3's law. Videos: 20 raw rounds per variant
under `collect/videos_NN/`, previews ≤ 500,000 bytes with `verification.json` under `previews/`.

## Tenth step: the "capsule tangent to the table edge" plan (2026-09-15)

User's plan: stop where the walking posture's collision capsule is tangent to the table edge;
keep the arms hanging until the bottle is large enough in the ego view, then raise; after the
stop let arm and waist solve together and lean as far as the bottle needs; treat the pads as
high-friction. Geometry checked first: with the compact fingers-forward walking posture the
fingertips are 0.12 m ahead of the pelvis, so the tangent stop is ≈0.20 m from the edge; the
real-mesh guard puts the right hip roll link on the table top when standing upright at ≤0.23 m
and clears at 0.27 m, while leaning 20–40° with the pelvis lowered clears even at 0.20 m; the
bottle sits 4–8 cm inboard of its edge (`CAN_INBOARD`), so a 0.27 m edge stop leaves it
0.31–0.35 m from the pelvis. Arm reach at grip height tops out at ≈0.60 m even with a 70° torso
pitch (waist 0.52 + pelvis 0.70 rad; the crate's deep bend buys height, its palms reach 0.29–0.32 m
forward), so the "forearm held in front" capsule (0.40 m stop) would be out of reach.

Implemented (all fields in `docs/reference/config_fields.md`):
`visible_brake_edge` 0.27 (stop at the edge distance or `visible_brake_depth` 0.31, whichever
first; the deceleration band applies to both), `raise_trigger_px` 36 (the observer reports the
bottle's ego pixel height; the arm holds until it is reached), the walking waist envelope without
roll, `lean_angle` = max(height term, distance term 0.30→0.60 m) capped by `lean_max` (1.0 in
`bottle_whole_body.json`, default 0.55 keeps the old law); the reach end stays `entry_step`.

`outputs/bottle_edge_sampled20_20260915`, same 20 seeds: **12/20** (v3 14/20, brake-0.31 baseline
17/20). Paired v3→edge: 9 both succeed, 5 v3-only, 3 edge-only, 3 both fail. Failures: 3 no
bilateral contact, 2 hip/table at the brake (was 5 with roll), 1 slip in recovery, 1 wrist/bottle,
1 finger/bottle. What each element did:

- Edge brake: fired in 2/20 stops; the depth rule fires first because the bottle is only 2–5 cm
  beyond the table box edge, so the stop moved nowhere (park 0.365 vs 0.359). To make the edge
  rule operative the depth brake must be ≥0.33 or dropped.
- Pixel trigger: the bottle subtends 36–37 px at the first plan (spawn ≈1.5 m), so the trigger
  fired at plan 0 in all 20 rounds; behaviour identical to "raise from the first plan".
- Roll-free envelope: hip/table 5 → 2, as predicted from the guard replay.
- Distance-driven lean: the distance term is ≤0.37 rad at these depths, below the height term;
  no lean changed.
- The grasp-side losses (+4 vs v3) coincide with the different hand configuration at takeover
  once the waist roll is gone (capture depth/lateral at the close are inside the successes'
  band: 1.9 cm / ≤2 cm), i.e. within what 20 rounds cannot separate from noise.

Previews under `outputs/bottle_edge_sampled20_20260915/previews/`. The tree runs this plan
(`bottle_whole_body.json` carries `visible_brake_edge` 0.27 and `lean_max` 1.0).
