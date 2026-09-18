# Document index

项目功能、安装路线和首次运行示例见 [仓库 README](../README.md)。

## Current development navigation

- [AGENTS](../AGENTS.md): shared entry and resource rules.
- [Change map](reference/change-map.md): select code/configuration/tests by intent.
- [Standards](reference/standards.md): architecture and evidence rules.
- [Session workflow](agents/session-workflow.md): verification scope, completion and handoffs.
- [Configuration fields](reference/config_fields.md): generated from code.
- [Runbook](reference/runbook.md): command reference and manual stack operation.
- [Run infrastructure](single_arm.md): one expanded experiment file per run, one flat script per stage, the process stack, and what parallelism means here (a second terminal). Read it before adding an experiment; the [2026-09-16](handoff/infra-plan-20260916.md) and [2026-09-17](handoff/refactor-flat-scripts-20260917.md) refactors are history.
- [STATE](STATE.md): active work; [ROADMAP](ROADMAP.md): maintenance scope and status.

## Documentation ownership

Each rule has one authoritative home; other documents link to it. Humans and coding
agents share [AGENTS](../AGENTS.md); [CLAUDE](../CLAUDE.md) points to that same contract.

| Material to add or change | Owning document / update rule |
|---|---|
| Project overview, installation routes and feature-to-guide links | [Root README](../README.md); one row and one canonical guide per task family; keep cases and alternate commands inside that guide |
| Shared entry or resource rules | `AGENTS.md`; keep it short and point to conditional detail |
| Code entry, config owner, reference implementation or tests | [Change map](reference/change-map.md); maintain the affected domain row with the code |
| Module boundaries, interface/config conventions or evidence rules | [Standards](reference/standards.md); link the rule instead of copying it into task guides |
| Configuration fields and defaults | Owning code; regenerate [field reference](reference/config_fields.md) with `make config-doc` |
| Task operation, runnable examples and capability limits | Canonical task guide below; keep the shared section order and update the affected example with the implementation |
| Cross-task CLI reference, wrappers and manual stack operation | [Runbook](reference/runbook.md); link task guides for complete collection recipes |
| Detailed task design or protocol | Link from the owning task guide; do not create a competing usage entry or copy its launch recipe |
| Verification scope, final check execution and completion criteria | [Session workflow](agents/session-workflow.md); keep Makefile and commit-hook behavior consistent |
| Active progress, blockers and resources | [STATE](STATE.md); link detailed work instead of accumulating transcripts |
| Maintenance scope and status | [ROADMAP](ROADMAP.md); link implementation and validation records |
| Experiment claims, corrections and completed work | Task ledger or `handoff/`; preserve historical material in `archive/` with evidence links |

Update an existing owning document first. Add a document for distinct detail, link it
from its task guide or this index, and state when to read it. Separate current usage
from dated experiments; preserve originals and link corrections when evidence changes.
Follow [the evidence rules](reference/standards.md#evidence). Keep current rules in
versioned documents reachable here, without requiring conversation history or private memory.

## Task and pipeline detail — read when needed

| 采集任务 | 唯一使用入口 |
|---|---|
| SONIC 单臂 | [单臂采集](single_arm.md) |
| SONIC 双臂（视觉抓篮、carry / wall 诊断、回放） | [双臂采集](bimanual.md) |
| Pick-and-place | [Pick-and-place 采集](single_arm.md) |
| ScaleBFM 后端 | [ScaleBFM 采集](scalebfm.md) |

单臂、双臂和 pick-and-place 指南统一按以下顺序维护：
**任务与能力 → 环境与场景 → 跑一局 → 批量采集 → 产物与验收 → 调整任务 → 排错与停止 → 进一步阅读**。
新 case 放进所属任务指南；新任务沿用此结构。明确区分尝试数与成功数、
视频 pilot 与训练数据，以及已实现接口与已验证学习效果；不适用或未验证的部分直接注明。
历史数字和逐版诊断放入台账/归档，首次运行不要求阅读；底层字段只链接生成的配置表。

通用流程：[安装](setup.md)、[数据转换](single_arm.md)、[训练](single_arm.md)、
[评测](single_arm.md)、[MuJoCo 仿真、Isaac 离线渲染与视频导出](rendering.md)。
理解在线控制、状态记录和离线重渲如何衔接时，读 [渲染整体流程](rendering.md#整体流程)。
专家设计、控制协议和对应实验从各采集指南的“进一步阅读”进入。

## Historical evidence and investigations

- Ledgers: [bottle](ledger/bottle.md), [crate](ledger/crate.md),
  [ScaleBFM](ledger/scalebfm.md), [pick and place](ledger/pick_place.md), [previous refactor](ledger/refactor.md).
- [Silent-failure notes](reference/silent_failures.md): historical troubleshooting leads; verify against code and evidence.
- [F10: silent exporter frame loss](handoff/exporter-frame-loss-20260917.md): open, root cause not located; also the index of the 2026-09-17 cold-start walkthrough's eleven findings (F1-F11).
- [Bimanual eval start-of-round lurch](handoff/crate-eval-prelude-twitch-20260917.md): why the deploy's FIRST round crouches and steps out of it, and the five fixes that did not work.
- [Archive index](archive/INDEX.md): original reports and superseded entry documents.
- [First migration](handoff/agent-maintenance-plan.md); [maintenance delivery](handoff/maintenance-completion.md).
- [Guideline audit and probes](handoff/agent-guideline-completion-20260914.md).

A newer claim or a retraction has no automatic evidentiary priority. Retain original
records, link corrections, and state the exact scope of any newly verified observation.
