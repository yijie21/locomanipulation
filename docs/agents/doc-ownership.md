# Documentation ownership

Each kind of material has one home; other documents link to it. Moved here from
`docs/README.md` on 2026-09-28, when the root README became the reader's document list.

| Material to add or change | Owning document / update rule |
|---|---|
| Project summary, the three main lines, the one-line list of every other document | [Root README](../../README.md). A new document gets one line there saying what it is for; a new main line gets a table row and a runbook |
| **How to run a main line**: commands, what each does, what it produces, the few common errors | That line's runbook ([bottle](../single_arm.md), [crate](../bimanual.md), [warehouse](../warehouse.md)); commands and results only, no design reasoning, no dated history, no default values the field table owns |
| Shared mechanics: experiment files, CLI arguments, skip/resume, parallel runs, the stack, the eval clock | [Running](../running.md); runbooks link to it instead of repeating it |
| **Why a line is built this way**, its extra troubleshooting, and detail the runbook leaves out | That line's design page ([bottle](../single_arm_design.md), [crate](../bimanual_design.md), [warehouse](../warehouse_design.md)) |
| **What a run produced on a given day** | That line's ledger ([bottle](../ledger/bottle.md), [crate and warehouse](../ledger/crate.md)) |
| Installation | [Setup](../setup.md): each step is the command, what it does and how to verify it; pitfalls go in its closing troubleshooting section |
| A term the guides use without explaining | [Glossary](../reference/glossary.md); define it once and link it |
| Configuration fields and defaults | Owning code; regenerate the [field reference](../reference/config_fields.md) with `make config-doc` |
| Shared entry or resource rules | `AGENTS.md`; keep it short and point to conditional detail |
| Code entry, config owner, reference implementation or tests | [Change map](../reference/change-map.md); maintain the affected domain row with the code |
| Module boundaries, interface/config conventions or evidence rules | [Standards](../reference/standards.md) |
| Verification scope, final check execution and completion criteria | [Session workflow](session-workflow.md) |
| Active progress, blockers and resources | [STATE](../../STATE.md) (written by `/handoff` with TODO and DECISIONS) |
| Experiment claims, corrections, finished work, superseded documents | Ledger, `docs/handoff/`, or `docs/archive/` (listed in [the history index](../README.md)) |

## Writing rules for the reader-facing pages (README, runbooks, running, setup)

- One sentence says one thing; say what a command does before any reason.
- Every command: a code block, then what it does, then what it produces (files, the log line to look for, time, expected result).
- No symbols as verbs (`=`, `→`, `/`) and no coined shorthand; identifiers stay in English, everything else in plain Chinese.
- At most one or two bold spans per section.
- Numbers only where the reader uses them: time, round counts, expected success.
- `scripts/lint_repository.py` caps each page's length (`GUIDES`), checks every `scripts/ configs/ src/ tests/` path a
  runbook names (`COMMAND_DOCUMENTS`), and requires any multi-word inline code without a `/` to be text the code prints.

Update an existing owning document first. Separate current usage from dated experiments;
preserve originals and link corrections when evidence changes.
