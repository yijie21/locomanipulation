# Working in locomanipulation

Application code is in `src/locomani/`; `src/gr00t/` is vendored training code.
Use this shared development contract with any coding agent.

## Find the change

Start with the specified file or known owner for local edits. Use scoped searches.
For new features, unclear ownership or cross-module changes, consult the relevant
domain of [the change map](docs/reference/change-map.md); expand reading along actual
callers and dependencies as needed.
Read [architecture and standards](docs/reference/standards.md) before changing an
interface, dependency direction or configuration format.

Read [STATE](STATE.md) (open tasks in [TODO](TODO.md), reasons in [DECISIONS](DECISIONS.md))
when resuming work or using shared compute. Use
[the document index](docs/README.md) when adding/reorganizing documentation or when
a task needs historical evidence. Historical experiments are not startup reading.

## Make and verify a change

- Parse configuration at the entry; pass explicit values to consumers. Keep changes
  local to the owning module and extend its existing interface.
- During implementation, use related tests with `make test TESTS="tests/test_name.py"`.
  For final validation choose the installed commit hook, or `make check` when
  delivering without a commit. Both use the [same scope rules](docs/agents/session-workflow.md#verification).
- Continue authorized implementation and validation through the
  [completion criteria](docs/agents/session-workflow.md#completion); resolve routine
  local choices without repeated approval.
- Record checks actually run and remaining uncertainty. Experimental claims follow
  the [evidence rules](docs/reference/standards.md#evidence).
- Preserve other sessions' edits. Stage explicit paths. Use isolated worktrees for
  refactors and reconcile against the current branch before integrating.

## Shared resources and artifacts

- Every video exported, linked for viewing or packaged for download MUST be a
  compressed preview: target 300–500 KB per video, hard cap **500,000 bytes**.
  Verify each file's actual size before delivery; ZIPs must contain these previews.
  Preserve source recordings. Follow [video export](docs/rendering.md#video-export)
  for compression, full-duration playback and verification; report download sizes.
- Before GPU/stack work, check live GPU use, instance locks and STATE. Acquire
  resources through the existing launcher. Stop only the owned instance.
- Formal ScaleBFM collection uses GPU EGL and pilot admission checks; read the
  [collection requirements](docs/scalebfm.md) before running it.
- `outputs/`, `third_party/` and `datasets/` are runtime paths. Never track them,
  including symlinks. Preserve datasets and weights; planned deletion of outputs
  or weights must be recorded in STATE at least 24 hours beforehand.
- Keep STATE short: active work, blockers and links. Handoffs name the branch,
  changed behavior, validation and next action; preserve history outside STATE.
