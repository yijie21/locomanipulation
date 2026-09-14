# Bottle: filtered walking preparation and continuous entry — 2026-09-14

This continues the [earlier whole-body candidate](bottle-continuous-20260914.md).
The user requested early, gradual arm alignment during walking, without following
high-frequency body sway or imposing a raise/backward/forward grasp sequence.
The opt-in profile remains `configs/tasks/bottle_whole_body.json`; the default
`bottle_visible.json` is unchanged. Usage is in [single-arm collection](../collection.md).

## Implementation

- `policy.py` owns whether preparation starts before the initial navigation leg.
  `round_phases.py` supplies the walk callback. The continuous controller starts
  from the current upper reference and slews toward its geometry-derived command;
  it does not replace that reference with READY in one frame. The short fixed
  samples generally skip the initial navigation leg, so this routing change alone
  does not explain their previous pauses.
- `walking_reach.py` filters camera-admitted relative geometry before 5 Hz IK
  planning. A 0.45 s filter suppresses gait oscillation; filtered planar body
  velocity predicts forward travel so filtering does not lag the approaching table.
  Upper commands use physical-time speed/acceleration bounds (at most 0.45 rad/s
  and 1 rad/s² during walking). Legs remain with SONIC; the waist returns gradually
  to a neutral walking reference. The supported solver only takes over after braking.
- Preparation uses a conservative arm workspace and clearance. The same admitted
  bottle and entry direction continue into the supported reach. The profile brakes
  at 0.40 m nominal depth with a 0.24 m deceleration band, replacing 0.31 / 0.12 m;
  the stop still depends on measured geometry, not a per-case hidden waypoint.
- The previous `clear → descend → insert` state sequence is removed. A continuous
  entry corridor opens when the hand is above the bottle and shrinks near capture
  height. Descent slows while the open hand lacks cap clearance; horizontal
  alignment continues. A collision-free approach at capture height is direct.
  Arbitrarily straight descent through the cap is not a valid grasp path.
- `SupportedMotion` seeds feedback from the measured/reference discrepancy at
  takeover, then bounds and filters deviations around that initial bias. Starting
  feedback at zero previously changed an otherwise held reference while feedback
  warmed up. The CPU test checks this effect explicitly; physical video/metrics
  remain necessary to assess any remaining mode-transition response.
- A rejecting navigation callback sends a stop and aborts. Contact, disturbance,
  support, slip, real lift, upright recovery and loaded transport acceptance remain
  active. Finger contact above the capture region is screened more strictly.

This expert uses simulation geometry admitted through fresh student-camera streams
and approximate frustum/ray visibility. It is not an RGB pose estimator. A cached
visible point bridges at most the configured one-second interval; hidden truth does
not refresh it. No object teleport, attachment or acceptance relaxation was added.

The `BottleTargetObserver` worker owns image receive/decode, visibility work and
private MuJoCo scratch data. The control thread submits a state snapshot and reads
the latest completed result. The original image timestamp controls expiry. This
addresses an observed 3.579 s synchronous camera-observation call in
`candidate_contact10` case 0; its following state gap was 3.40 s. The final run has
no >0.5 s walking control-clock failure. Last per-case control-thread observer calls
were 0.085–0.140 ms; these last-sample timings are not latency quantiles.

## Reproduction and identity

Worktree: `/workspace/code/locomanipulation-bottle-direct-20260914`, branch
`fix/bottle-direct-20260914`, base `c0e994cf81d3d48dd8d6aae0ba24cf0a9b6091e1`.
Runtime artifacts are under `outputs/bottle_direct_20260914/` in the main workspace.
`candidate_source/`, `balanced_source/`, `contact_source/` and `final_source/`
contain the source/SHA-256 inventories for each fixed batch. Subsequent edits are
documentation and tests; the runner retains resolved task/scene/model/deploy identities.

The previous candidate is `outputs/bottle_continuous_20260914/candidate10` (3/10).
Reuse its `cases_fixed10.json`: target/stand/spawn/appearance/object yaw and scene
metadata identity are frozen. The table is still 62 cm and the bottle is unchanged.
The same private patched v1.1 deploy binary is used for both versions.

```bash
SONIC_DEPLOY_BIN="$PWD/outputs/bottle_continuous_20260914/deploy/target/release/g1_deploy_onnx_ref" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
  -m locomani.pipeline.run_learn_pilot \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_whole_body.json \
  --scene outputs/scene/sim --out outputs/bottle_direct_repeat \
  --cases outputs/bottle_continuous_20260914/cases_fixed10.json \
  --set through=collect --set successes=10 --set batch_rounds=10 \
  --set max_batches=1 --set seed0=31000 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0
```

Use an idle GPU/instance and a fresh output directory. The private binary is a local
artifact; other developers must build the tracked ownership patches per
[setup](../setup.md). Fixed sampling does not make asynchronous dynamics deterministic.

## Validation artifacts and interpretation

Development probes are retained separately: `probe1` 0/2, `probe2` 0/2, `probe3`
0/2, `probe4` 1/4 and `probe5` 1/4. They exposed filter-induced approach lag,
premature finger/cap contact, occasional control-clock gaps, and support limits.
They are not pooled with the fixed final comparison. The first full `candidate10`
batch was 0/10: six COM-bound failures, two finger-clearance failures, one
body-clearance failure and one standing initialization failure.
It is retained as a failed intermediate revision. The subsequent controller bounds
torso-goal rotation to 0.25 rad/s and strengthens COM retention in `SupportedMotion`;
the default `WholeBodyIK` weight used by crate is unchanged.

The motion report script is preserved as `motion_report.py`. It measures physical
wrist trajectories from MuJoCo qpos traces and joins reference phases by sequence.
Its path length averages 0.2-second blocks to reduce jitter; the last positive walk
command is a reproducible braking marker, not proof that the base is already still.
Action audits separately check token/gripper schema and consumed-command receipts.
Neither smoothness metrics nor matching labels establish VLA learning performance.


## Final fixed comparison

The two intermediate full batches, `candidate10` and `candidate_balanced10`, were
both 0/10. The corrected cap-entry batch `candidate_contact10` was 8/10, with one
camera-clock failure and one wrist-clearance failure. After moving observation to
the worker, the final `final10` batch is **6/10**, versus the previous candidate's
3/10. These are separate revisions/runs, not pooled samples or a claim of statistical
significance. All original attempts, scenes, videos and traces are preserved.

Final successful case IDs: `31000:1,2,3,5,6,8`. Failures:

- `31000:0`: 3.5 cm palm-frame slip at terminal hold; rejected after lift/transport.
- `31000:4`: hip/table clearance during braking.
- `31000:7`: wrist/bottle clearance.
- `31000:9`: disturbed target/base gate.

Metrics below are medians across successful episodes (three previous, six final):

| Measurement | Previous | Final |
|---|---:|---:|
| Raw frames / nominal 50 Hz duration | 1116 / 22.32 s | 931.5 / 18.63 s |
| Last positive walk command → capture | 9.84 s | 6.68 s |
| Post-brake wrist path, 0.2 s block average | 0.970 m | 0.689 m |
| Path length / displacement | 3.522 | 3.109 |
| Longest identical token run | 0.38 s | 0.27 s |
| Body joint-step p99 | 0.0247 rad | 0.0342 rad |
| WBC joint-command-step p99 | 0.0702 rad | 0.1085 rad |
| Post-brake wrist upward excursion | 1.34 cm | 7.78 cm |

The last three rows are material limits: the new expert is faster and prepares
while walking, but **overall physical smoothness has not been established**.
There is still a visible lift/rebound during deceleration/POSE takeover. Joint-step
statistics mix purposeful arm movement, gait and tracking response; they are not
jerk bounds or isolated vibration measurements. The quiet detector decreases from
0.26 to 0.03 s, but jitter can hide a visually stationary pose, so it cannot certify
that all hesitation disappeared. No VLA training improvement is claimed.

Successful measured recoveries reached 12.7–16.6 cm actual bottle rise and torso
lean 1.6–4.7 degrees, then passed back-off and the unchanged terminal hold checks.
The control channel still changes from the planner to POSE; reference continuity
and a slower torso target do not remove every physical response to that switch.

`comparison.json`, `previous_motion.json`, `final_motion.json` and
`final_actions.json` preserve the calculations. The unchanged behavior gate is
**FAIL, 10/13 checks pass** (`final_gate.json` / `.log`): changed braking depth,
failure categories and reach-stage duration fail equivalence. Its tolerances were
not adjusted, and this result is not relabeled PASS.

The next motion investigation should separate walking, braking and POSE execution
when measuring joint steps, then test the physical response to the reference
handoff. The current evidence does not justify solving the remaining rebound by
adding another sequence of grasp waypoints or loosening collision acceptance.
