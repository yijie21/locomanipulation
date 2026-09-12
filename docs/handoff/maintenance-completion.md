# Repository maintenance delivery

Worktree: `/workspace/code/locomanipulation-maintenance-final-20260912`.
Branch: `refactor/maintenance-final`; baseline: `2e9af84`.
User authorization: continue the remaining repository cleanup, 2026-09-12.

The listed structural migrations and their CPU/review acceptance are complete.
Runtime validation is complete as an experiment, but bottle physical equivalence
remains **unverified**: original A/B and unchanged-revision A/A gates fail. No
physical commands or gate tolerances were tuned to obtain a passing comparison.

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
- Crate ABBA diagnostic (`4cc52ba` -> `3eca6bb`) complete:
  blocks A=2/10, B=2/10, B=4/10, A=3/10;
  pooled A=5/20, B=6/20. Same-version repeats also fail some unchanged gate checks.
  Raw: primary `outputs/experiments/maintenance_crate_control_20260912/`.
- Source migrations are committed as `bac6492` and reviewed corrections as
  `ec7ca1d`. The final integration adds only documentation after this tested source.

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

## CPU and configuration validation

`make check` passed in the isolated worktree with no generated `outputs/scene`.
The three training modality configurations loaded in the real training venv.
Focused delivery tests verify observation dumping/snap, per-client gain/ensemble/
stall decisions, dataset rate, inherited shell overrides, and real child settings.
Pre-commit passed on both source commits. The final source is frozen at `ec7ca1d`;
subsequent delivery edits are documentation. `EvalConfig.to_dict()` matches the
fixed baseline exactly, as do resolved coupled/teleop gym options. Source-body
comparisons record unchanged control routines and distinguish constructor/input
wiring changes; they do not certify equivalent physical trajectories.

Evidence under primary `outputs/maintenance_final_20260912/`:
`make-check-ec7ca1d.log`, `precommit-ec7ca1d.log`, `eval-defaults.json`,
`bottle_profile_defaults.json`, `control_source_audit.json` and its audit script.
An additional fixed-input/clock replay of the real default UpperRamp emits exactly
equal positions, velocities, residuals and slew limits over 200 target steps on both
revisions (`upper_ramp_comparison.json`, replay script and arrays in that directory).

## Independent review and corrections

Initial Standards review found four delivery/lifecycle defects; Spec review found
three (overlapping). They were reproduced and fixed before integration:

- Startup failure now cleans only children launched after this shell acquired the
  lock; admission refusal leaves the other owner untouched. Real shell tests use
  CPU stand-ins and verify lock release and explicit DDS delivery.
- Instance ports, upper-reference slew and all lockstep settings reach actual
  consumers; fresh-child probes exercise defaults read from serialized config.
- Ready pilot generates the current evaluation config/argv, including 25 FPS.
  Its normal and instrumented routes pass the actual parser.
- Start and restart append final-cwd/binary/model hashes to `deploy_receipts.jsonl`.
  Foreign deployment trees and same-size weight changes are covered. Initial
  snapshots are explicitly pre-launch requests; receipts describe launch selection.
- Learn-pilot save interval reaches the child; final weights are selected even
  off a periodic save step. Existing partial datasets/models are retained and
  require inspection plus a new output directory instead of automatic deletion.

Both independent re-reviews report no remaining findings (Standards: 27 focused
CPU tests plus fresh-process consumers; Spec: 39 focused CPU tests). Both then
confirmed the final source commit `ec7ca1d`; the Spec reviewer also passed the two
final training-child/receipt cases.
Artifacts: primary `outputs/maintenance_final_20260912/navigation_standards/`
and `navigation_spec/`. Source review closure does not claim GPU acceptance.

## Fresh-context navigation observations

Two independent agents started only from AGENTS/change-map, edited isolated probe
worktrees, and passed targeted tests plus `make check`, without reading historical
STATE/archives/ledgers. The acceptance extension read 11 files / 29,374 UTF-8 bytes /
664 lines; the FPS validation extension read 13 files / 61,814 bytes / 1,334 lines.
These are measured reading-cost proxies, not model token counts or a demonstrated
percentage reduction from the old repository. One probe initially used a Python
without pytest; AGENTS now gives `make test TESTS=...` using the configured venv.
Probe patches and reading logs remain in the two artifact directories above.

## Runtime evidence

The first 80-round comparison uses `2e9af84` and intermediate source `bac6492`.
It retains coupled 2/30 -> 3/30 (FAIL 5/6 checks) and teleop 10/10 -> 8/10
(FAIL 14/16 checks). Coupled gained one observed failure category; its baseline
had too few successes for the gate's stage-time comparison. Teleop failed yield
and new failure categories. These observations are not replaced by the final replay.

The final `ec7ca1d` replay completed with coupled 2/30 -> 3/30 (FAIL 5/6 checks)
and teleop 10/10 -> 9/10 (FAIL 15/16 checks). Both fail new observed failure
categories; teleop's yield and timing checks pass. Coupled's baseline has only two
successes, so no stage-time comparison is available. All the same 30 coupled and
10 teleop cases were used against the retained baseline.
Recorded case geometry, appearance and commanded object yaw are
checked pairwise. Seed `20260912`, SONIC v1.1, encoder, dex1 hands, scene files,
model/deploy hashes and existing gate tolerances stay fixed. Batches run sequentially
on GPU 0 with ten episodes per fresh stack; other host load remains uncontrolled.

A complete 40-case repeat of `2e9af84` was predeclared after the intermediate FAIL,
before the final replay result, to examine same-version variability. Both original
and repeated baseline comparisons remain available. It is a single fixed repeat,
not a search for a passing baseline. Asynchronous physical runs are not deterministic.

Additional checks use the real GR00T server/checkpoint and PolicyLink on three
captured observations with fixed denoising seeds, and two final-version crate pilot
episodes to exercise startup, completed results, receipts and cleanup. The inference
check covers plain inference and identity, not RTC performance or closed-loop yield.
Crate success is diagnostic, consistent with the user's scope clarification.

Real-server plain inference passed: all six token/gripper arrays returned by
PolicyLink (default token snapping) are elementwise identical across revisions,
and all 16 checkpoint file hashes match before/after.
The first baseline startup failed because the experiment set `HF_HUB_OFFLINE=1`,
which blocked a tokenizer metadata request despite cached model files. The successful
retry restores normal network access, with unchanged source/checkpoint/inputs/seeds;
both attempts remain. The server, crate smoke and baseline repeat use `attempt2/`
under their directories. The root protocols retain the original predeclarations.

Crate smoke passed: two completed manifests (2/2 physical successes), all three
stage exits zero, command/config/deploy receipts present, owned stack stopped.
This two-episode observation does not certify crate scene quality or success rate.
The coupled baseline repeat is 2/30 -> 2/30 (FAIL 5/6 checks), gaining the same
`coupled transport disturbed target or base` category observed in the intermediate
candidate. Thus that category also occurs on the unchanged pre-refactor revision.
Repeat-baseline/final is 2/30 -> 3/30 (FAIL 5/6), with different newly observed
categories. This supports observed same-version variability; it neither identifies
every cause nor certifies physical equivalence.

| Unchanged-gate comparison | Coupled successes / checks | Teleop successes / checks |
|---|---|---|
| Original baseline A1 -> final C | 2/30 -> 3/30; FAIL 5/6 | 10/10 -> 9/10; FAIL 15/16 |
| Original baseline A1 -> repeated baseline A2 | 2/30 -> 2/30; FAIL 5/6 | 10/10 -> 9/10; FAIL 15/16 |
| Repeated baseline A2 -> final C | 2/30 -> 3/30; FAIL 5/6 | 9/10 -> 9/10; FAIL 15/16 |

All six failures above concern newly observed failure categories. Teleop A/A gains
`teleop squat was not delivered`; A2/C gains a height-related local-alignment timeout.
Both coupled baselines have only two successes, leaving stage-time coverage absent.
No comparison is promoted to PASS by choosing another baseline or combining runs.
These data do not establish whether every difference is caused by the refactor or
by other run variation; this uncertainty remains open.

This phase executed 160 bottle episodes, 40 earlier-revision crate ABBA episodes
and two final crate smoke episodes. Three captured observations were also replayed
through each real policy server. Initial offline-mode startup produced no inference
comparison; its failure and the dependent queue cancellations are retained.

Final audit exited zero: all gate results recompute, case inputs pair correctly,
14 captured SONIC/deploy/scene files retain their hashes, four final bottle launch
receipts match actual executable/weights, and all 16 GR00T checkpoint files plus
three observations retain their identities. The audit verifies the report, not
physical equivalence. Logs/results: `outputs/maintenance_final_20260912/` contains
`final_audit.json`, `final_audit.log` and `resources_after.json`.

Owned resources were released at 2026-09-12 13:31 UTC: GPU 0 at 2 MiB / 0%,
instance 0/1 locks free; the unrelated GPU 1 job was left running. Check live state
before reuse. Runtime artifacts, datasets and weights remain untracked and preserved.
Local raw evidence remains untracked
under primary `outputs/experiments/`:

| Directory | Scope |
|---|---|
| `maintenance_final_gpu_20260912/` | 80 original bottle episodes, fixed inputs, fingerprints and intermediate gates |
| `maintenance_final_replay_20260912/` | 40 final-source bottle episodes and original-baseline comparisons |
| `maintenance_server_replay_20260912/` | Same checkpoint/observations/seeds, actual server identity and action arrays |
| `maintenance_crate_smoke_20260912/` | Two final pilot episodes; physical outcomes are diagnostic |
| `maintenance_bottle_repeat_20260912/` | 40 repeated baseline episodes, A/A and repeat-baseline/final comparisons |

`done.json` denotes completed execution, not gate acceptance. Each directory retains
its runner and protocol; per-stage commands/logs identify the actual revision and
environment. The repeat reuses the helper's `candidate` directory label for the
second **baseline** run; its command cwd and protocol explicitly identify `2e9af84`.
`outputs/maintenance_final_20260912/audit_final.py` recomputes retained gates and
checks pairing/identities without turning a FAIL into a PASS.
