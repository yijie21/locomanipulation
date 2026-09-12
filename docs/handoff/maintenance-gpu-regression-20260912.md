# Maintenance GPU behavior regression — 2026-09-12

Completed simulation comparison: baseline `4cc52ba` → candidate `3eca6bb`.
Bottle gates pass on the matched samples; crate wall fails. Overall behavioral
acceptance is **not complete**. No control, physical or gate thresholds were tuned.

## Results

| Final comparison | Baseline successes | Candidate successes | Existing gate |
|---|---:|---:|---|
| Bottle coupled | 4/30 | 6/30 | PASS, 15/15 checks |
| Bottle teleop | 9/10 | 10/10 | PASS, 16/16 checks |
| Crate wall, ready start | 4/10 | 2/10 | FAIL, 4/7 checks |

Success counts use the existing gate: bottle requires `ok` and recorded data;
crate uses `ok`. A progress message incorrectly counted crate successes with the
bottle rule and reported zero. This reporting error is corrected above; the raw
manifests and gate calculations were unaffected.

Crate wall failed three unchanged checks:
- Yield fell from 0.4 to 0.2, exceeding the 0.15 allowed drop.
- New observed failure categories: `[hold] crate tilted N deg` and
  `[lift] crate tilted N deg`.
- Successful descent-time median changed from 2.65 s to 1.4 s, outside 30% tolerance.

Stand offset (0.2468 → 0.2464 m), depth (0.11 → 0.11 m) and yaw error
(0.4872 → 0.952 degrees) pass. The printed `exit_majority` PASS compares absent
values; this ready-start protocol does not exercise walking approach. With
`wall_walk=false`, the controller directly ramps to hover, so continuous `reach_s`
is not applicable. Successful `descend_s` samples number four and two respectively.

## Protocol and identity audit

- Pinned trees: `/workspace/code/locomanipulation-baseline-4cc52ba` and
  `/workspace/code/locomanipulation-maintenance-20260912`. Their recorded revisions
  and tracked runtime code were unchanged at the final audit.
- Seed `20260912`; SONIC v1.1, encoder enabled, dex1 hands, GPU EGL, asynchronous
  controller. GPU 0 runs sequentially: bottle instance 0, crate instance 1.
- Bottle uses `bottle_visible.json` and `bottle_visible_teleop.json`, ten rounds per
  fresh stack, recording/video initialization enabled, no stand retries.
- Crate uses `run_crate_pilot.py --expert wall`, ten rounds per revision, the same
  crate scene and unchanged task defaults: `wall_feature=rail`, `wall_walk=false`.
- All 40 final bottle pairs have identical recorded initial conditions, including
  commanded object yaw. Runtime ReachConfig values match after version migration.
- All ten wall pairs have equal sampled crate/stand offsets and angles. Task and
  backend snapshots match. Model-directory strings differ by worktree but resolve
  to the same directory; the other recorded policy settings match.
- Nine model/deploy files and five scene files retain their initial SHA-256 values.
  Hashes establish file identity within this captured set, not complete determinism.
- Crate task, SONIC backend, stack launcher, crate pilot and config-base source are
  unchanged between the pinned revisions. This does not establish the cause of the
  wall differences. No same-revision A/A variance control was run.

GPU 1 acquired an unrelated job before the original crate admission. The guard
refused that launch; subsequent crate batches used GPU 0. Other host load and
asynchronous timing remain uncontrolled. These small samples cannot establish
improved physical success rates or certify historical claims. Coupled's passing
comparison still has low absolute success on both revisions.

## Preserved protocol corrections

The first run executed 80 bottle episodes. Its raw gates were coupled 4/30 → 3/30
(FAIL 14/15) and teleop 9/10 → 7/10 (FAIL 14/16). A manifest audit found all 40
baseline/candidate commanded yaws differed: generated case banks omit `object_yaw`,
and `plan.sample_object_yaw(None)` uses global NumPy randomness outside the seeded
scene generator. Matching seeds and case-bank files were insufficient.

The correction retained all 40 baseline episodes, including failures, and replayed
all 40 candidate cases with the baseline's recorded yaw. No cases were selected by
outcome. The original candidate episodes remain unmatched diagnostic observations.
The first run's `coupled_failure_audit.json` describes its unmatched case; it cannot
support a paired causal interpretation. Neither raw results nor thresholds changed.

The initial gate help also named carry, while its geometry/timing fields are
emitted by wall phases. Twenty carry diagnostic episodes produced 0/10 successes
on each revision, incomplete metric coverage and raw FAIL 3/4 (yaw median
5.2239 → 3.6208 degrees). The four sampled carry inputs match; derived stand targets
can differ at floating-point scale. Carry does not provide successful-carry evidence.
Twenty additional wall episodes produced the final crate result above. Gate help
now names wall; metric calculations and tolerances are unchanged.

Total executed: 160 episodes. The final comparisons use 100 episodes; the remaining
60 are preserved diagnostics (40 unmatched bottle candidate, 20 carry).

## Evidence locations and repeatable checks

All run directories below are under `outputs/experiments/` and remain untracked:

| Directory | Evidence |
|---|---|
| `maintenance_gpu_20260912/` | Original 80 bottle episodes, baseline case banks, initial fingerprints and GPU admission failure |
| `maintenance_gpu_paired_20260912/` | Corrected 40 candidate bottle episodes, pairing checks, final bottle gates, 20 carry diagnostics, final audit and CPU checks |
| `maintenance_gpu_wall_20260912/` | 20 wall episodes, final crate gate and metric-coverage report |

Each retains its runner source, protocol, actual command/settings, manifests and
stack logs; videos/traces remain alongside batches. Original exporter sessions
remain under `/workspace/tmp/p03/outputs`. `done.json` means execution completed,
not gate acceptance. Historical protocol metadata may describe intended scheduling;
per-batch `*.command.json` and this report identify what actually ran.

The [experiment runner](../../scripts/experiments/maintenance_gpu_regression_20260912.py)
now pins yaw for fresh runs and rejects mismatched bottle manifests before gating.
Use a fresh output directory; existing result directories are deliberately refused.
For example, the corrected bottle replay is reproducible with:

```bash
python3 -u scripts/experiments/maintenance_gpu_regression_20260912.py \
  --out outputs/experiments/maintenance_gpu_replay_new \
  --replay-candidate-from outputs/experiments/maintenance_gpu_20260912
```

The recorded per-batch wall commands reproduce wall runs. The retained `run_wall.py`
uses the same helper and standard driver. Existing tracked gate baselines were not
replaced: each comparison recorded a fresh baseline from the pinned old revision.

`maintenance_gpu_paired_20260912/audit_results.py` recomputes all gates from retained
manifests, checks pairing/configuration/file identity and writes `final_audit.json`.
Run it from the repo with `PYTHONPATH=src:.` using the simulation Python environment.
The audit exited 0. CPU validation is `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make
check`; protocol tests cover yaw/geometry mismatch, missing/duplicate rows, retaining
failed reference cases, incomplete crate metrics and unrecorded ready-wall success.

## Handoff

Owned stacks are stopped, instance 0/1 locks are free, and GPU 0 was at 2 MiB / 0%
at release; `gpu-after.csv` records that observation. Check live occupancy before
reuse. The unrelated GPU 1 job was left running. All raw evidence is preserved.

Next: investigate crate wall with a predeclared same-revision repeat and matched
A/B schedule to distinguish run variance from a revision effect; inspect tilt
traces at the failed stages. Keep this FAIL until new discriminating evidence exists.
Do not change acceptance tolerances or tune control parameters to obtain a pass.
