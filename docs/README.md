# Document index

项目功能、安装路线和首次运行示例见 [仓库 README](../README.md)。

## Current development navigation

- [AGENTS](../AGENTS.md): shared entry and resource rules.
- [Change map](reference/change-map.md): select code/configuration/tests by intent.
- [Standards](reference/standards.md): architecture and evidence rules.
- [Session workflow](agents/session-workflow.md): verification scope, completion and handoffs.
- [Configuration fields](reference/config_fields.md): generated from code.
- [Runbook](reference/runbook.md): command reference and manual stack operation.
- One expanded experiment file per run, one flat bash script per stage over one python module, parallelism as a second terminal: each runbook carries this for its own line, and the decisions, the byte A/B evidence and the gate still outstanding are in [the 2026-09-21 refactor](handoff/refactor-flat-bash-20260921.md). The [2026-09-16](handoff/infra-plan-20260916.md) and [2026-09-17](handoff/refactor-flat-scripts-20260917.md) refactors are history.
- [STATE](../STATE.md): active work, with [TODO](../TODO.md) and [DECISIONS](../DECISIONS.md); [ROADMAP](ROADMAP.md): maintenance scope and status.

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
| **How to run it** -- commands, their arguments, artifacts, troubleshooting | That line's runbook ([single arm](single_arm.md), [bimanual](bimanual.md)); capped at 180 lines, and it never restates a default value that the generated field table owns |
| **Why it is built this way** -- principles, rejected alternatives, protocols | That line's design page ([single arm](single_arm_design.md), [bimanual](bimanual_design.md)) |
| **What a run produced on a given day** -- rounds, seeds, success rates | That line's ledger ([bottle](ledger/bottle.md), [crate](ledger/crate.md)) |
| Cross-task CLI reference, wrappers and manual stack operation | [Runbook](reference/runbook.md); the runbooks own the complete per-line recipes |
| Verification scope, final check execution and completion criteria | [Session workflow](agents/session-workflow.md); keep Makefile and commit-hook behavior consistent |
| Active progress, blockers and resources | [STATE](../STATE.md) (written by `/handoff` with TODO and DECISIONS); link detailed work instead of accumulating transcripts |
| Maintenance scope and status | [ROADMAP](ROADMAP.md); link implementation and validation records |
| Experiment claims, corrections and completed work | Task ledger or `handoff/`; preserve historical material in `archive/` with evidence links |

Update an existing owning document first. Add a document for distinct detail, link it
from its task guide or this index, and state when to read it. Separate current usage
from dated experiments; preserve originals and link corrections when evidence changes.
Follow [the evidence rules](reference/standards.md#evidence). Keep current rules in
versioned documents reachable here, without requiring conversation history or private memory.

## Task and pipeline detail — read when needed

| 线 | 怎么跑（采集→转换→训练→评测，一页跑完） | 为什么这么设计 | 实测数字 |
|---|---|---|---|
| SONIC 单臂（含 pick-and-place、bottle_visual） | [单臂](single_arm.md) | [单臂详解](single_arm_design.md) | [台账](ledger/bottle.md) |
| SONIC 双臂（视觉抓篮、carry / wall 诊断、回放） | [双臂](bimanual.md) | [双臂详解](bimanual_design.md) | [台账](ledger/crate.md) |
| SONIC 物流房间抓纸箱（离线渲染、数据集相机事后补渲） | [warehouse](warehouse.md) | [DECISIONS](../DECISIONS.md) | [DECISIONS](../DECISIONS.md) |
| ScaleBFM 后端 | [ScaleBFM](scalebfm.md) | 同页 | [台账](ledger/scalebfm.md) |

三份 runbook 按同一顺序维护：**准备 → 一次运行=一份实验文件 → 新建实验 → 每次都要改的字段 → 跑 → 产物 → 排错 → 进一步阅读**。新 case 放进所属的线；新线沿用此结构。
明确区分尝试数与成功数、视频 pilot 与训练数据，以及已实现接口与已验证学习效果。

[双臂前伸包络](reach_envelope.md)是复现 2026-09-21 边界搜索的 runbook（空地场景、`search.json`、双栈逐轮搜索、表/图/视频），[记录](handoff/g1-capability-envelope-20260921.md)里是它的测量与推断；固定站姿下的[双臂可达实验](bimanual_reachability.md)说明 ghost 验证、条件域计算和区间查询；[五维联合实测](handoff/g1-joint-reach-search-20260920.md)记录1,277次尝试、原生复核、倾角对照与独立复测；[早期条件切片](handoff/g1-reachability-atlas-20260920.md)保留原atlas的覆盖限制。可迁移双臂抓箱：[箱体边沿抓点规划](box_rim_planning.md)根据尺寸和实际停车状态生成 ghost；[仿真执行](box_rim_execution.md)说明低速接近、闭合与试提的验证边界。正式采集仍使用现有专家。

双臂可达实验的环境与采样范围预览见 [G1 平地 ghost hand 预览](handoff/g1-reachability-preview-20260920.md)；预览仅展示目标分布。后续 [14组动态 reach 实验](handoff/g1-ghost-reach-execution-20260920.md) 已完成，6组成功、8组超时仅指 reach 阶段；[后续诊断](handoff/g1-startup-jump-diagnosis-20260920.md) 确认初始化缺陷，整段原地要求未通过。

跨线通用：[安装](setup.md)、[MuJoCo 仿真、Isaac 离线渲染与视频导出](rendering.md)。
数据集格式（LeRobot 布局与 schema）各线共用，在 [单臂详解](single_arm_design.md) 里。
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

历史记录：G1 修正 reset 后的同14组重跑见 [本次结果与视频](handoff/g1-ghost-reach-fixed-reset-repeat-20260920.md)：当时4组reach成功、9组超时、1组初始化倒地，启动同步问题尚未解决；当前状态见上方可达实验与本轮证据。

G1 [原 expert 实跑与任务录像起点修正](handoff/g1-original-expert-task-boundary-20260920.md)：原版行走抓取成功；当前单组reach视频3.92秒，从站稳状态直接开始，准备诊断另存。前轮证据见 [同14组重录](handoff/g1-shared-expert-ready-repeat-20260920.md)。
