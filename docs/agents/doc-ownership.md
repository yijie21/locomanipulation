# Documentation ownership

Each kind of material has one home; other documents link to it. Moved here from
`docs/README.md` on 2026-09-28, when the root README became the reader's document list.

| Material to add or change | Owning document / update rule |
|---|---|
| Project summary, the main line (scene RSI since 2026-10-20), the one-line list of every other document, the archive section | [Root README](../../README.md). A new document gets one line there saying what it is for. Archived pages (the VLA collection and training line, 2026-10-20) stay where they are, carry a one-line archive note at the top, are listed only in the README's archive section and are not updated with the code |
| **How to run the main line**: commands, what each does, what it produces, the few common errors | [Scene RSI runbook](../rsi.md); commands and results only, no design reasoning, no dated history. The archived VLA runbooks ([bottle](../single_arm.md), [crate](../bimanual.md), [warehouse](../warehouse.md)) keep their old rule |
| Shared mechanics of the archived VLA line: experiment files, CLI arguments, skip/resume, parallel runs, the stack, the eval clock | [Running](../running.md) (archived) |
| **Why the main line is built this way**, its extra troubleshooting, and detail the runbook leaves out | [Scene RSI design](../rsi_design.md); the dated decisions stay in the [plan](../handoff/scene-rsi-plan-20261016.md) |
| **What a run produced on a given day** | Scene RSI: the [progress report](../handoff/scene-rsi-report-20261019.md) and the plan; the archived lines' ledgers ([bottle](../ledger/bottle.md), [crate and warehouse](../ledger/crate.md)) |
| **This branch's main line**, the self world model (branch `selfwm`; scene RSI is `main`'s main line and is not maintained here): commands for the run switches on public benchmarks; the room-grasp and per-experiment commands; why it is built this way; what a run produced | [Benchmark runbook](../selfwm_bench.md); [runbook](../selfwm.md); [design](../selfwm_design.md); [dated record](../handoff/selfwm-20261029.md) and the [experiment entries](../handoff/notebook-20261120/) |
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
