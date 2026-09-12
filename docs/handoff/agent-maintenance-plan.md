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
thread counts set to one. Candidate results will be recorded before handoff.

Candidate CPU validation:
- `make check` exited 0 after copying the existing 33 MB scene assets into this
  worktree. The unit tier still depends on generated scene/hand-point assets;
  a fresh-worktree fixture migration is recorded in ROADMAP.
- `reach_config_migration_20260912.py` compared main at `4cc52ba` with this tree:
  11 profiles have identical effective values; 176 real sample_stand calls match
  exactly (168 accepted, 8 rejected). This does not test physical success rates.
- Four archived entry documents match their `4cc52ba` blobs byte-for-byte.
- Intentional CLI changes: explicit values equal to defaults now win; boolean
  facing can explicitly be disabled; ambiguous option abbreviations and >2 spawn
  distances are rejected. The resolver performs no I/O; its entry loads inputs once.
- Root guide 39 lines; change map 44; STATE 30. These are reading-size measurements,
  not a measured reduction in task-completion token cost.
- Live GPU bottle/crate gates have not been run. No physical commands or thresholds
  were tuned. Historical experiment conclusions remain independently unverified.
- Independent standards and spec reviews pending before local integration.
