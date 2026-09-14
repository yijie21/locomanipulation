# Session workflow

Use [AGENTS](../../AGENTS.md) for reading conditions and resources; use this page for verification, completion and handoffs.

## Verification

| Change | Development feedback |
|---|---|
| Prose / links | `make check-docs`: navigation, changed Markdown links and tracked-artifact rules |
| Local code | Relevant tests: `make test TESTS="tests/test_name.py"` |
| Config / shared interfaces | Actual consumer, precedence and delivery tests |
| Physical commands / timing / sampling / acceptance | Relevant tests plus authorized live validation under [standards](../reference/standards.md#checks) |

Choose one final entry: the installed commit hook when committing, or `make check`
for delivery without a commit. Both run `scripts/check_changes.py`. A commit with
no installed hook needs `make check` before committing; installation is not automatic.
The default CPU suite uses disposable fixtures and skips live simulation. Run it and
fix failures caused by the requested change within existing authorization.

Only root Markdown and `docs/**/*.md` changes take the documentation route. Other
paths, Markdown fixtures, unknown inputs and a clean checkout take full lint and the
complete default CPU suite. `make check-all` explicitly forces full validation.
The hook sees staged content after pre-commit hides unstaged tracked edits; untracked
inputs remain visible. Direct `make check` considers staged, unstaged and untracked
paths. Renames/deletions also affect scope. A narrowed `TESTS` does not narrow final checks.

Checks do not rewrite source; use `make format` when formatting needs repair.
After a successful final check, repeat only for changed inputs, failures or unresolved
concerns. There is no result cache: explicitly invoking both final entries repeats
work. Prefer the commit hook if a commit is already part of the requested delivery.

## Completion

Infer the requested behavior, constraints and delivery target from the task. Resolve
routine local choices and continue authorized work through implementation and verification.
Delivery means the requested behavior is implemented, affected code/config/docs agree,
required checks pass, and requested runtime results or artifacts have been inspected.
A first implementation or a passing unit test alone does not establish physical behavior.
Report checks actually run, observed results and remaining limits; a blocked required
check remains unverified. Fix related failures before claiming completion.

For experiments, keep the agreed scope and compute budget, preserve failed attempts,
and report unmet targets. Repeated trials need a specific unresolved question; retain
the original acceptance criteria. Implementation completion and experimental success
are separate claims. Follow [evidence rules](../reference/standards.md#evidence).

## Handoff

Name the branch/commit, changed behavior, validation, uncertainty and next action.
Keep active status/resources in [STATE](../STATE.md), linked evidence outside it, and durable rules in versioned documents.
