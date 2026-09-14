# Agent workflow adjustment — 2026-09-14

Read this for the first delivery scope; the [follow-up report](agent-guideline-completion-20260914.md) completes the remaining work. Daily instructions
remain in [AGENTS](../../AGENTS.md) and the [session workflow](../agents/session-workflow.md).
Baseline: `c2b5686`; implementation branch: `chore/agent-workflow-20260914`.

## Delivered scope

- Local edits start at a specified file or known owner. The change map applies to
  new features, unclear ownership and cross-module work; architecture, STATE and
  historical documents retain explicit reading conditions. CLAUDE and the human
  README point to the same contract.
- `scripts/check_changes.py` owns final scope selection for both `make check` and
  the installed commit hook. Root/docs Markdown-only changes run navigation,
  changed-document links and tracked-artifact checks. Code, configuration, unknown
  paths and clean checkouts run full lint and the default CPU suite. `make check-all`
  forces the latter. Renames inspect both paths; untracked inputs participate.
- Development uses relevant tests; final delivery chooses the hook or `make check`.
  No successful-result cache was added. Explicitly running both still repeats work;
  choose the hook when a commit is part of delivery. Checks no longer rewrite source.
- Completion means requested behavior, matching code/config/docs, required checks,
  and inspection of requested runtime results/artifacts. Routine authorized work
  continues without repeated approval. Failed experimental targets remain failures;
  preserve attempts and limits within the agreed scope and compute budget.

## Validation

`make test TESTS="tests/test_check_workflow.py tests/test_repository_rules.py"`:
14 tests passed. The new tests use real Git diffs and disposable make targets to
check document-only, staged/unstaged/untracked code, code renamed to Markdown,
Markdown fixtures, clean/forced-full runs, and failure propagation. The full route
always requests the entire default suite even if `TESTS` is narrowed elsewhere.

An actual installed pre-commit hook in a disposable repository selected documentation
checks for a staged README edit, preserved an unstaged code edit, and then invoked
full lint and the CPU target once when that code was committed. This probe validates
routing and staging behavior; its make targets are fixtures, not the project suite.
An initial real-hook run exposed inherited `GIT_*` variables redirecting fixture
Git commands into the implementation worktree. The unpublished fixture commit was
quarantined, the worktree index/branch and shared `core.bare` setting restored;
main's files and commit stayed unchanged. The fixture now clears inherited Git
variables. All 14 focused tests also pass with injected foreign Git/index paths,
which remain untouched. See `hook_environment_tests.log` and `commit_first_attempt.log`.
Final project-wide validation runs through this change's commit hook.
Logs: `outputs/agent_workflow_20260914/{focused.log,hook_probe.json,commit.log}`.
No physical commands, simulation protocol or acceptance thresholds changed; this
workflow adjustment does not require a GPU behavior comparison.

## Remaining guideline work at this delivery

All four follow-ups were subsequently handled in [the completion report](agent-guideline-completion-20260914.md); the original scope is retained below.

1. Shorten the detailed change map by task domain; conditional reading is delivered,
   but the full table remains about 11.6 KB. Avoid replacing it with many tiny routers.
2. Audit current task guides for historical procedures presented as current rules,
   especially the long crate guide. Preserve original evidence and scope corrections.
3. Audit user/plugin skills for overlapping triggers and rigid generic workflows.
   Those skills are outside this repository and were not changed in this delivery.
   Add project skills only for demonstrated reusable operations that need them.
4. Compare representative local-edit, config-change and GPU-task sessions for reading
   volume, repeated checks, unnecessary pauses and missed defects. This delivery tests
   executable routing; it does not establish a measured token-saving percentage or
   equal instruction-following behavior across coding models.
