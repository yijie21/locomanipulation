# STATE (updated 2026-09-08 07:25 UTC by 9d; main line = session 9d, instance 0 / GPU0; branch = 65)

## Now
- Main line (9d): ScaleBFM v3 bottle pilot, outputs/experiments/bfm_pilot_v3_20260908/ (pilot.log is the
  stage log). Collection DONE 07:20: 261 rounds, 210 accepted (80.5%), train 200 ep / 61025 frames,
  val 10; GRIP_KP=40 GRIP_KD=0.5, seed 1, fixed scene. Training RUNNING on GPU0 since 07:22
  (3000 steps, ~1.4 s/it, ETA ~08:35), then open-loop on val + 60-round in-process eval on GPU1
  (seed 6000, replan 0.4 s, latency 4 ticks). GPU1 is idle until the eval starts.
- Branch session (65): idle, nothing running.

## Next
1. Read the pilot eval (outputs/experiments/bfm_pilot_v3_20260908/eval/closed_loop.json). Pass bar for the
   overfitting test: >50% on the fixed scene. Pass -> scale the collection (randomised spawn, 2 seeds);
   fail -> open-loop error + replay round-trip decide between data/label bug and control bug.
2. Branch (65): symmetric jaw command (drive q1-q2 to 0) against close-time ejection, after 1.
3. Progress-parameterised expert (replace time u with path progress) before the DAgger stage.

## Blocked / waiting on
- 65: GPU1 is 9d's until the pilot eval ends (9d messages when free).

## Open questions
- check_dataset's grasp-causality rise check was frame-confounded (object pose is pelvis-frame; the carry
  lowers the pelvis) and failed 19/200 valid episodes; fixed 2026-09-08 by rebuilding world z from
  root_position/root_orientation. Older SONIC datasets passed only because their lift was larger --
  any earlier "min rise" numbers from this check under-read the true lift.
- Ready-grasp line (docs/ready_grasp_arms_20260907.md): closer stand + no shuffle 29/60 is the best
  arm; "never squatted" is training-seed variance, not data. Needs 200+ episodes and 2 seeds per
  config before the next claim. Parked while ScaleBFM is evaluated.

## Parked (do not forget)
- SONIC lockstep collection: docs/handoff/sonic_lockstep_collection_plan.md
- Isaac render + MuJoCo physics: memory isaac-render-mujoco-physics-plan
- v4 clockless expert (0/3, abandoned; lessons in memory scalebfm-interface-facts)

## Recent results (last 3)
- 2026-09-08 bfm_pilot_v3_20260908 collection: 210/261 accepted; audit 26/26 after the rise fix
  (world-frame rise after close: median 9.3 cm, min 8.8 cm)
- 2026-09-08 outputs/videos/scalebfm_bottle_v3_grip_{A,B,C,B_s3,C_s3}: kd1.0 6/10, kd0.5 10/10+9/10, kp8/kd0.15 9/10+10/10
- 2026-09-07 ready-grasp six arms (docs/ready_grasp_arms_20260907.md): 07 18/60, 07d closer+no-shuffle 29/60 (p .052)
