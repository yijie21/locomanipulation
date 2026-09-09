# STATE (updated 2026-09-08 18:50 UTC by 9d; main line = session 9d, instance 0; branch = 65)

## Now
- ScaleBFM visual-expert Goal (this Codex thread): ACTIVE. GPU0 reserved for this
  Goal after 9d's explicit 18:50 release (claimed 18:56 UTC); GPU1 currently has
  another live workload and is left alone. No recording stack/ports are used.
  Plan/checkpoint: docs/scalebfm_goal_plan.md and docs/scalebfm_goal_state.md.
  Review correction: a15 cohort retired (145 attempts counted); a16 passed30/30
  qualification,11/12recovery. Formal v2 comparison driver active onGPU0, currently
  allABCdata/export/stats complete200train+20val each. AttemptcostA225/880,
  B367/880includingretired,C225/880includingreuse. C5A17trainingLIVE,
  2841/3000updates at2026-09-09 01:13UTC; finiteinitialloss/gradients,GPU0~26GB.
  DriverPID1081396/session17266; trainerPID1081656. Firstinitializationfailure
  repaired/archivedwith0updates; sixconfigsrefrozen. Checkpoint2500auditPASS
  (optimizer/scheduler/RNG/model/processor; latest-onlyretentionverified). Ledger:pipeline_v2_state.json.
  User authorized staged implementation and fixed-budget VLA comparison (2026-09-08);
  supersedes the ScaleBFM choice in Next item 2; resource release is the 9d entry below.
- DONE 18:45 (9d): 07d-200 closed-loop eval 57/100 (docs/ready_grasp_arms_20260907.md §8). Paired first 60:
  37/60 vs ready07d_decisive 29/60, discordant 22:14, McNemar p 0.24 -- not significant; needs a second seed.
  Instance 0 / GPU0 / GPU1 are FREE (stack stopped by the runner). Model: ready07d_200_20260908/model (24 GB).
- ScaleBFM: nothing running. Three pilots today (v3 18-d, v3 26-d offline velocity, v4 23-d native
  velocity) all 0/60; the failure moved from "never leaves the spawn" to "stops 10 cm short of the stand".
  The scene geometry was then found to be wrong (corner approach, 0.45-0.49 m reach) and replaced by the
  SONIC edge geometry; the expert works there (x-/y-/y+ edges 2/2 each). No dataset recorded on it yet.
- Branch (65): idle. Do not start a recording stack and never run stop_stack.sh (kills every instance)
  until 9d says instance 0 / GPU1 are free (after the 07d eval).

## Changed today (all UNCOMMITTED; commit is the user's call)
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

## Next
1. 07d-200 second seed: same train_dataset (176 ep), GR00T_DATA_SEED=7 (see run_learn_pilot / launch_finetune),
   100 rounds seed 6000 -> the data-vs-seed question of docs/ready_grasp_arms §5/§8 (n=1 cannot carry 8 points).
2. ScaleBFM decision (user): (a) record 200 episodes on the edge geometry as-is (STAND_D 0.45-0.55, reach
   overlapping the walk -- the best-scoring expert combination, 5/6) and run the v4 pipeline (~2.5 h), or
   (b) first a mode-6 spike (pelvis + shoulder + elbow + wrist targets) to remove the ~20 deg waist twist,
   which needs an arm IK for the elbow target. The 10-cm stop-short of v4 is a policy stop-rule problem
   (gait-state keyed) that a cleaner expert does not fix by itself; DAgger from stopped-short states or a
   STAND_D curriculum remain the candidates for that.
3. Branch (65): symmetric jaw command (drive q1-q2 to 0) against close-time ejection.
4. Progress-parameterised expert (replace time u with path progress) before the DAgger stage.

## Blocked / waiting on
- nothing: instance 0 and both GPUs are free as of 18:45.

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
- Ready-grasp line (docs/ready_grasp_arms_20260907.md): closer stand + no shuffle 29/60 is the best arm;
  "never squatted" is training-seed variance. Needs 200+ episodes and 2 seeds per config -- the 07d-200
  run is the first of those.

## Parked (do not forget)
- SONIC lockstep collection: docs/handoff/sonic_lockstep_collection_plan.md
- Isaac render + MuJoCo physics: memory isaac-render-mujoco-physics-plan
- v4 clockless expert (0/3, abandoned; lessons in memory scalebfm-interface-facts)

## Recent results (last 3)
- 2026-09-08 ready07d_200 (176 ep): 57/100; paired first 60 37 vs 29 (p 0.24); reach-fail stops closer (0.24 m) but still no close
- 2026-09-08 ScaleBFM edge geometry: expert 2/2 on x-, y-, y+ edges after the BIAS fix; STAND_D sweep
  (6 rounds each) 0.25-0.30 3/6, 0.30-0.35 4/6, 0.35-0.40 3/6, 0.40-0.45 2/6, 0.45-0.55 5/6; twist ~20 deg
  everywhere (outputs/videos/scalebfm_edge_scene_20260908/)
- 2026-09-08 bfm_pilot_v4 (native velocity, 23-d): 0/60, stops 10 cm short; open-loop body_mae 0.44 cm
- 2026-09-08 bfm_pilot_v3: 18-d 0/29 (never left the spawn); 26-d offline velocity 0/60 + 0/60 (closes 6 cm
  short / 8 cm high, open-loop 0.90 cm)
