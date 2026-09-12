# STATE — 2026-09-12

## Current delivery

- First repository maintenance delivery complete: short agent entry, checked change
  map, architecture constraints and bottle configuration migration. CPU checks pass.
- Branch: `refactor/agent-maintenance`; worktree:
  `/workspace/code/locomanipulation-maintenance-20260912`.
- [Scope and validation](handoff/agent-maintenance-plan.md).
- This session has not started a GPU job. Recheck live occupancy before GPU work.

## Next

- Next migration: unify runner/environment resolution; use its ROADMAP acceptance
  condition. Live bottle/crate behavior comparisons are still unverified.
- Follow-up migrations and completion conditions: [ROADMAP](ROADMAP.md).

## Evidence

- Historical experimental conclusions, including their retractions, have not been
  independently verified by this maintenance session. Apply the
  [evidence rules](reference/standards.md#evidence) before relying on them.
- The previous 1473-line STATE is preserved verbatim in
  [the archive](archive/state_20260912.md), sourced from commit `4cc52ba`.
  Its RUNNING lines describe old observations, not current resource availability.

## Parked constraints carried from the previous record

- The historical record attributes NO DAgger and parking crate stand tuning to the
  user. This maintenance task does not change either policy or physical tuning.
- Resource/artifact rules: [AGENTS](../AGENTS.md).
