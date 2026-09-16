# Architecture and development standards

These are the current rules. [The change map](change-map.md) lists implemented
entry points and reference tests. Historical plans describe earlier intentions.

## Design principles

Organize code by responsibility, with names and explicit interfaces that explain
its behavior. Extend the owning module's established pattern; extract shared code
when consumers share a contract. Keep each parameter, schema and development rule
in one authoritative place. Keep navigation short and disclose detail by task.
Use inspectable evidence to distinguish implementation, observation and interpretation.

## Module ownership

| Module | Owns | Consumer contract |
|---|---|---|
| `pipeline/` | Lifecycle, recording, conversion, training, evaluation | Explicit run inputs and recorded results |
| `tasks/<task>/` | Sampling, task decisions, acceptance | Observations, explicit state/config, decisions/results |
| `backends/<backend>/` | Controller protocol, observations, execution | Backend execution behind a task-independent interface |
| `primitives/` | Shared geometry and joint calculations | Values in, values out |
| `sim/` | Simulator integration, assets and rendering | Scene and simulation inputs |

The entry composes tasks and backends. Task-specific grasp geometry belongs in the
task; a backend receives data or a callable. New backend-to-task imports are rejected
by `scripts/lint_repository.py`. The historical exception list in `tests/fixtures/architecture_debt.json` is now
empty; do not add an exception to bypass ownership.
Production modules cannot import experiment runners. The checker enforces these
specific rules, not every architectural property or dynamic dependency.

## Interfaces and configuration

- A caller should need a small explicit interface, without module-attribute mutation
  or import-order setup. Keep internal helpers with their owning module.
- Keep state/time explicit at decision seams. A stateful controller owns its state.
  Claim deterministic replay only for a tested input/state/clock contract. Physical
  asynchronous runs are not assumed bit-deterministic.
- Define each parameter/default in its owning configuration. Resolve code defaults,
  a profile and explicit CLI overrides once, in that order. An explicit value equal
  to the default is still an override. Validate before starting resources.
- Use one selector for mutually exclusive implementations and separate fields for
  independent capabilities. Historical formats use versioned migrations; current
  formats reject obsolete fields and unknown choices.
- At subprocess boundaries serialize resolved configuration and verify delivery.
  `pipeline/stack.py` resolves the child environment; `pipeline/stages.py` records
  the actual command, settings and exit status. Bottle and crate use both.
- Task entries return `RoundResult`; acceptance consumes this completed result.
  Do not start an episode from acceptance. Phase inputs remain task-specific.
- `schema.py` owns stored/model field order; conversion, training configurations
  and observation builders consume it. A policy server reports its checkpoint
  file hashes and actual model modalities; evaluation checks and records them.
- Runs (infra plan 2026-09-16): a run is composed from `configs/` groups by `locomani.conf`; a version is a file under `configs/experiment/` (`+experiment=<name>`), never edited after it has produced results.
- Stages: one process each (`python -m locomani.cli <stage>`), reading only the upstream `<out>/<stage>/manifest.json` (config-group hash, upstream hash, outputs, metrics, resources) and writing its own; stacks start only through `locomani.stack.Stack`; parallel work goes through `locomani.launch`, which appends every finished stage to `outputs/registry.jsonl`.

## Checks

Final validation is selected by `scripts/check_changes.py`, shared by `make check`
and the commit hook. Use the [verification scope and completion criteria](../agents/session-workflow.md)
when choosing checks or deciding whether work is delivered. Static checks protect
specified boundaries and links; interface quality and parameter necessity still need
review. Size advisories locate complexity, not interface quality or a reading quota.

Test observable contracts and important failure modes through real consumers.
Configuration changes need precedence, migration, invalid-input and delivery coverage.
For structural migration compare resolved values, selected implementations and
replayable decisions with a fixed pre-change revision. Record intended differences.

Main-code refactor PRs retain the bottle/crate `scripts/gate.py compare` requirement.
Check baseline/candidate protocol compatibility before interpreting a gate. An
unavailable or incompatible comparison is unverified, never an implicit pass.
For this maintenance delivery, the user explicitly excluded crate physical
success/reproducibility as a blocker (2026-09-12); retain its diagnostic failures.
Changes to physical commands, timing, sampling or acceptance also need live validation.

## Evidence

Separate implementation facts, measured observations, causal explanations and design
decisions. A date, confident summary or apology does not establish a claim.
For a decision-relevant claim record its statement/scope; code/config/model/scene
revision; command/raw artifacts; metric calculation/result; controls/limitations.
Missing artifacts mean unavailable evidence. Hashes establish identity, not correctness
of a measurement. Conflicting interpretations remain disputed until discriminating
evidence exists; a retraction alone proves neither interpretation.

Code/tests establish implementation behavior within their checked scope. Changing a
historical conclusion requires new evidence or an inspectable reasoning/calculation
correction. Preserve the original claim and link the correction.

## Documentation and integration

Keep the root guide, change map and STATE within the checked reading budgets.
Link task-specific details. The document index distinguishes current navigation from
historical evidence; tutorials do not certify experiments they recount. Generate
field tables from configuration code. Commands must point to existing entries.
Place new material using [documentation ownership](../README.md#documentation-ownership)
and finish code/documentation updates through the [change workflow](change-map.md#change-workflow).

Use isolated worktrees and reviewable commits. Preserve other sessions' work and
stage explicit paths. Resource/artifact rules are in [AGENTS](../../AGENTS.md).
