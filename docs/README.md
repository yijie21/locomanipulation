# Document index

项目功能、安装路线和首次运行示例见 [仓库 README](../README.md)。

## Current development navigation

- [AGENTS](../AGENTS.md): shared entry and resource rules.
- [Change map](reference/change-map.md): select code/configuration/tests by intent.
- [Standards](reference/standards.md): architecture, checks and evidence rules.
- [Configuration fields](reference/config_fields.md): generated from code.
- [Runbook](reference/runbook.md): implemented command entry points.
- [STATE](STATE.md): active work; [ROADMAP](ROADMAP.md): maintenance scope and status.

## Documentation ownership

Each rule has one authoritative home; other documents link to it. Humans and coding
agents share [AGENTS](../AGENTS.md); [CLAUDE](../CLAUDE.md) points to that same contract.

| Material to add or change | Owning document / update rule |
|---|---|
| Project overview, installation routes and feature-to-guide links | [Root README](../README.md); address users running the project, and link development rules from the contribution section |
| Shared entry or resource rules | `AGENTS.md`; keep it short and point to conditional detail |
| Code entry, config owner, reference implementation or tests | [Change map](reference/change-map.md); maintain the affected row with the code |
| Module boundaries, interface/config conventions or evidence rules | [Standards](reference/standards.md); link the rule instead of copying it into task guides |
| Configuration fields and defaults | Owning code; regenerate [field reference](reference/config_fields.md) with `make config-doc` |
| Supported launch commands | [Runbook](reference/runbook.md); point to implemented entry points |
| Task-specific usage or design explanation | Relevant task guide below; explain when the reader needs it |
| Active progress, blockers and resources | [STATE](STATE.md); link detailed work instead of accumulating transcripts |
| Maintenance scope and status | [ROADMAP](ROADMAP.md); link implementation and validation records |
| Experiment claims, corrections and completed work | Task ledger or `handoff/`; preserve historical material in `archive/` with evidence links |

Update an existing owning document first. Add a document when a distinct task needs
separate detail; link it from this index or the relevant guide and state when to read
it. Move historical material with links to the retained original and its correction.
Follow [the evidence rules](reference/standards.md#evidence) for claims and revisions.
Keep current rules in versioned documents reachable from these entries, so future
contributors can find them without conversation history or an agent's private memory.

## Task and pipeline detail — read when needed

[Setup](setup.md), [collection](collection.md), [dataset](dataset.md),
[training](training.md), [evaluation](evaluation.md), [rendering and video export](rendering.md),
[crate collection](crate_collection.md), [ScaleBFM collection](scalebfm_collection.md),
[pick and place](pick_place_collection.md).
Crate expert redesign: [visual guidance and privileged-information contract](handoff/crate-visual-expert-design.md)
(proposal; read when implementing the new expert or reviewing its training data).
[Crate camera-guided video pilots](crate_visual_collection.md): simulation-assisted ordinary-basket control, historical RGB path and artifact meanings.
[Carry motion comparison](handoff/crate-motion-flow-20260913.md): timing, command bounds and GPU evidence.
These tutorials include historical experiments and explanations. Their presence in
this index does not validate their measurements or causal conclusions.

## Historical evidence and investigations

- Ledgers: [bottle](ledger/bottle.md), [crate](ledger/crate.md),
  [ScaleBFM](ledger/scalebfm.md), [pick and place](ledger/pick_place.md),
  [previous refactor](ledger/refactor.md).
- [Silent-failure notes](reference/silent_failures.md): historical troubleshooting
  leads; check referenced code and evidence before applying a recommendation.
- [Archive index](archive/INDEX.md): original reports and superseded entry documents.
- [First migration](handoff/agent-maintenance-plan.md); [maintenance delivery and validation](handoff/maintenance-completion.md).

A newer claim or a retraction has no automatic evidentiary priority. Follow
[the evidence rules](reference/standards.md#evidence). Retain original records,
link corrections, and state the exact scope of any newly verified observation.
