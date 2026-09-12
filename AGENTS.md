# Working in locomanipulation

Application code is in `src/locomani/`; `src/gr00t/` is vendored training code.
Use this shared development contract with any coding agent.

## Find the change

Read [the change map](docs/reference/change-map.md), then the row relevant to the
request. It names the entry, configuration, reference implementation and tests.
Read [architecture and standards](docs/reference/standards.md) before changing an
interface, dependency direction or configuration format.

Read [STATE](docs/STATE.md) when resuming work or using shared compute. Historical
experiments are reached through [the document index](docs/README.md) when a task
needs them. They are not startup reading.

## Make and verify a change

- Parse configuration at the entry; pass explicit values to consumers. Keep changes
  local to the owning module and extend its existing interface.
- Run selected tests with `make test TESTS="tests/test_name.py"`; this uses the
  configured CPU environment. Then run `make check`. The repository checker rejects
  new forbidden imports, broken navigation and oversized entry documents.
- Record checks actually run and remaining uncertainty. Experimental claims follow
  the [evidence rules](docs/reference/standards.md#evidence).
- Preserve other sessions' edits. Stage explicit paths. Use isolated worktrees for
  refactors and reconcile against the current branch before integrating.

## Shared resources and artifacts

- Before GPU/stack work, check live GPU use, instance locks and STATE. Acquire
  resources through the existing launcher. Stop only the owned instance.
- Formal ScaleBFM collection uses GPU EGL and pilot admission checks; read the
  [collection requirements](docs/scalebfm_collection.md) before running it.
- `outputs/`, `third_party/` and `datasets/` are runtime paths. Never track them,
  including symlinks. Preserve datasets and weights; planned deletion of outputs
  or weights must be recorded in STATE at least 24 hours beforehand.
- Keep STATE short: active work, blockers and links. Handoffs name the branch,
  changed behavior, validation and next action; preserve history outside STATE.
