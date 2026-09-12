# Agent maintenance migration

User authorization: 2026-09-12, implement the proposed repository cleanup for
low-context, consistent development and independently checkable evidence.
Starting revision: `4cc52ba`. Branch: `refactor/agent-maintenance`.

## First delivery

- One shared short AGENTS entry, CLAUDE delegation and a checked change map naming
  real entries/configs/tests. STATE holds current work only.
- Preserve old STATE, ROADMAP, README and runbook byte-for-byte in archive.
  Historical observations and causal interpretations remain unverified until checked.
- Enforce navigation budgets/links, forbidden dependencies with explicit existing
  debt, and exclusion of tracked runtime artifacts through make/pre-commit.
- Replace ReachConfig's five strategy booleans with a version-2 strategy selector.
  Validate version-1 JSON migration. Existing predicates become derived read-only
  compatibility properties rather than independent knobs.
- Pure bottle argument resolution: explicit CLI values override profiles even when
  equal to old defaults. VLA retains its reach calibration unless explicitly changed.
  Reuse the loaded task profile in the round driver.
- Centralize strategy construction and test the real factory. Provide a CPU-only
  gym-options report with sources, without claiming full runtime provenance.
- Verify old profiles preserve values, implementations and sampling behavior.
  Test precedence corrections/invalid inputs. Run make check and independent
  standards/spec reviews. Report live behavior gates separately.

## Scope

Retain control algorithms, physical thresholds, sampling distributions and success
criteria. This does not certify historical conclusions or finish all runner/global
setting/backend dependency migrations. ROADMAP records remaining acceptance criteria.

## Validation

Baseline: 24 config/reach/Task-descriptor tests passed at `4cc52ba` with CPU
thread counts set to one. Captured candidate code: `38b3c3b`; subsequent edits
update documentation and hook descriptions only.

Candidate CPU validation:
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make check` exited 0 after copying
  the existing 33 MB scene assets into this
  worktree. The unit tier still depends on generated scene/hand-point assets;
  a fresh-worktree fixture migration is recorded in ROADMAP.
- The capture and comparison commands below compared `4cc52ba` with `38b3c3b`:
  11 profiles have identical effective values; 176 real sample_stand calls match
  exactly (168 accepted, 8 rejected). This does not test physical success rates.
- Four archived entry documents match their `4cc52ba` blobs byte-for-byte.
- Intentional CLI changes: explicit values equal to defaults now win; an explicit
  spawn offset clears inherited edge sampling; one-shot/video flags inherit profiles;
  facing/one-shot/video can explicitly be disabled; ambiguous option abbreviations
  and >2 spawn distances are rejected. The resolver is pure; its entry loads once.
- Root guide 39 lines; change map 44. These are reading-size measurements,
  not a measured reduction in task-completion token cost.
- Live GPU bottle/crate gates have not been run. No physical commands or thresholds
  were tuned. Historical experiment conclusions remain independently unverified.
- Standards review found missing capture/compare evidence, supplied below. Spec
  review found two precedence omissions, fixed with actual scene-setup and expert/VLA
  boolean tests; independent re-review found neither issue remaining.
- Local pre-commit installed in the shared clone. No hosted CI was added. Size lint
  remains a report: 8 files over 600 lines and 95 functions over 80 at validation.

## Reproduce the CPU comparison

Raw files are retained locally under `outputs/maintenance/` in the maintenance
worktree and copied to `outputs/maintenance/agent-entry-20260912/` in the primary
checkout: `baseline.json`, `candidate.json`, `comparison.json`, `resolved-teleop.json`,
`make-check.log`. Runtime artifacts remain untracked. Captures contain source hashes,
effective config values, sampled decisions, code revision and scene hash.

Commands ran from `/workspace/code/locomanipulation-maintenance-20260912` while
the primary checkout was still at `4cc52ba`:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LOCOMANI_SIM_DIR=/workspace/code/locomanipulation/outputs/scene/sim
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/reach_config_migration_20260912.py \
  --repo /workspace/code/locomanipulation --out outputs/maintenance/baseline.json
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/reach_config_migration_20260912.py \
  --repo /workspace/code/locomanipulation-maintenance-20260912 --out outputs/maintenance/candidate.json
python3 scripts/experiments/reach_config_migration_20260912.py \
  --compare outputs/maintenance/baseline.json outputs/maintenance/candidate.json \
  > outputs/maintenance/comparison.json
```

Comparison exit 0: `scene_equal=true`, `profiles_equal=true`, 11 profiles,
176 samples (168 accepted, 8 rejected). Scene SHA-256:
`0e8899db8c8438bb50b8fe51306bd35bb968d602bbf6723d57e4c86feafa7fd9`.
Equality covers the entire effective-value/sample mappings, including rejected
samples. A temporary capture with one altered effective value correctly exited 1.
To repeat after integration, supply detached worktrees at the recorded revisions
instead of assuming the primary checkout is still the baseline; reuse the scene
and hand-point/reach-map assets. This comparison does not execute robot control.
