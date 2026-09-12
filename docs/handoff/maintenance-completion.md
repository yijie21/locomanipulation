# Remaining repository maintenance

Worktree: `/workspace/code/locomanipulation-maintenance-final-20260912`.
Branch: `refactor/maintenance-final`; baseline: `2e9af84`.
User authorization: continue the remaining repository cleanup, 2026-09-12.

## Acceptance

- Bottle and crate share environment resolution, process launch records and scoped
  stack lifecycle. Settings received by real child processes are tested.
- Backend modules no longer import tasks. Bottle grasp orchestration lives with
  the bottle task; generic arm control accepts explicit settings.
- Task execution returns a typed result; acceptance reads that result without
  starting another episode. Dataset schemas have a shared definition used by
  conversion, training configuration and evaluation.
- Unit tests run in a fresh checkout without generated `outputs/scene` files.
- Recorded run/checkpoint identities include content hashes and stage settings.
- Short navigation names real owners and extension examples. Fresh-context review
  records files read, correct entry selection and representative edit verification.
- Run relevant tests, full repository checks and independent standards/spec review;
  integrate reviewable commits with no runtime artifacts tracked.

Crate scope clarification from the user: low success or non-reproducibility does
not block this maintenance delivery; the scene needs separate future design.
Retain observed failures and do not tune crate control or acceptance thresholds.
The already-started 40-round ABBA run is diagnostic only. Bottle compatibility
remains subject to CPU comparisons and a matched GPU regression.

## Progress and evidence

- Fresh worktree baseline unit suite fails on implicit `outputs/scene/sim/scene_meta.json`
  access in `test_bottle_cli.py`; the fixture migration must remove that dependency.
- Original crate wall FAIL reproduced by the existing gate from retained manifests.
- Crate ABBA diagnostic complete: blocks A=2/10, B=2/10, B=4/10, A=3/10;
  pooled A=5/20, B=6/20. Same-version repeats also fail some unchanged gate checks.
  Raw: primary `outputs/experiments/maintenance_crate_control_20260912/`.
- Implementation and validation are in progress; no completion claim yet.

## Implemented contracts

- Common stack resolution and Stage receipts; full backend JSON reaches Python
  consumers, including nested lockstep fields. Optional paths do not leak from a
  previous instance. Tests run real child processes with controlled settings.
- Bottle execution/servo code moved from SONIC to the bottle task. Generic arms
  accept IK and guard settings; backend-to-task exception list is empty.
- GymRuntime supplies policy/restart/robot construction; immutable VlaConfig
  reaches link, clock and policy factories. Production evaluation uses per-clock
  commander classes. Legacy diagnostic instrumentation is explicitly separate.
- RoundResult represents completed execution; acceptance is pure. Shared schema
  preserves SONIC storage slices and bottle/crate/visual model ordering.
- Portable geometry fixture preserves FK without generated scene dependencies.
- Policy identity hashes weights and normalization files; serving reports actual
  model modalities. Evaluation rejects a mismatched schema before a round.

## Intentional corrections and limits

- Nondefault dataset FPS now reaches clients before they capture their rate;
  repeated evaluation no longer retains prior module flags. Previous nondefault
  runs need their actual artifacts checked before timing conclusions are reused.
- Arm channel/velocity/guard settings now reach the selected generic arm driver.
  Default planner settings are preserved. Teleop transport has its own backend.
- Typed results do not make asynchronous physical control deterministic. Geometry
  fixtures certify tested kinematics, not photorealism or simulation dynamics.
- New evaluation expects the updated server's identity endpoint; restart an older
  server before evaluating. Hashes establish content identity, not model quality.

## Validation in progress

`make check` passed in the isolated worktree with no generated `outputs/scene`.
The three training modality configurations loaded in the real training venv.
Focused delivery tests verify observation dumping/snap, per-client gain/ensemble/
stall decisions, dataset rate, inherited shell overrides, and real child settings.
Matched bottle GPU comparison and independent review follow the code freeze.
