# Repository maintenance roadmap

Current task: [first migration](handoff/agent-maintenance-plan.md).
Earlier research plans remain verbatim in [the archive](archive/roadmap_20260912.md);
claims and priorities there require review before reuse.

| Work | Completion condition |
|---|---|
| Short entry and navigation — delivered | Shared AGENTS; checked paths/tests and reading budgets; historical material is reached on demand |
| Bottle configuration path — delivered | Explicit strategy, tested legacy migration, pure argument resolution and actual consumer tests; live gates remain unverified |
| Runner/environment convergence | Bottle and crate launch through the same environment/lifecycle seam; resolved child settings verified |
| Backend/task separation | Remove every listed backend-to-task edge; task geometry stays with tasks and backend interfaces receive explicit inputs |
| Runtime/acceptance contracts | Real typed entry contracts exercised by tests; acceptance can inspect results without executing another round |
| Test assets | Unit geometry fixtures have explicit inputs; a fresh worktree does not need implicit reads from generated `outputs/scene` assets |
| Schema and provenance | Conversion and eval agree on field layout/checkpoint identity; stage-specific resolved snapshots verify consumer delivery |
| Experiment evidence | Recompute decision-relevant observations; distinguish observations, interpretations and design choices |
| Agent navigation evaluation | Fresh-context representative edits record context read, correct file selection, verification and rework |

Choose one complete path at a time. Validate compatibility against a pinned revision
and separate intentional behavior changes from structural migration. Keep useful
existing tests, helpers and formats until their replacements are exercised.
