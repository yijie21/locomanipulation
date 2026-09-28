# 历史记录与开发者索引

三条主线、通用文档、支线和台账的清单在[仓库 README](../README.md)。这一页只收 README 不列的东西：给开发者的入口，以及按时间留下的交接记录和归档。

## 开发者入口

- [AGENTS.md](../AGENTS.md)：改代码的共同约定，从这里开始。
- [代码导航](reference/change-map.md)：按要改的东西找代码、配置和测试。
- [开发规范](reference/standards.md)：模块边界、接口和配置约定、证据规则。
- [会话流程](agents/session-workflow.md)：验证范围、完成标准和交接。
- [文档维护规则](agents/doc-ownership.md)：每类内容写在哪一页，以及面向读者的页面怎么写。
- [STATE](../STATE.md)、[TODO](../TODO.md)、[DECISIONS](../DECISIONS.md)：当前进度与卡点、任务账本、每次决策的理由与被否决的方案。
- 流水线的结构（一次运行一份实验文件、每个阶段一个脚本、并行就是多开终端）怎么定下来的：[2026-09-21 重构](handoff/refactor-flat-bash-20260921.md)；更早的 [2026-09-16](handoff/infra-plan-20260916.md)、[2026-09-17](handoff/refactor-flat-scripts-20260917.md) 两次只作历史。

## 台账

每一轮实验的实测数字和结论：[bottle](ledger/bottle.md)、[crate 与 warehouse](ledger/crate.md)、[pick-and-place](ledger/pick_place.md)、[ScaleBFM](ledger/scalebfm.md)、[Blender 场景](ledger/blender_scenes.md)、[上一次重构](ledger/refactor.md)。

## 排查线索

- [隐性失败](reference/silent_failures.md)：历史上的排查线索，路径和默认值要对照代码逐项核实。
- [F10：exporter 静默丢帧](handoff/exporter-frame-loss-20260917.md)：仍未关闭，根因没找到；也是 2026-09-17 冷启动走查 11 条发现（F1–F11）的索引。
- [双臂评测开局的突然动作](handoff/crate-eval-prelude-twitch-20260917.md)：为什么 deploy 的第一局会先蹲下再站起，以及试过但无效的五种修法。

## 双臂可达性的历次记录

- [G1 能力包络](handoff/g1-capability-envelope-20260921.md)：[双臂前伸包络](reach_envelope.md)那张图的测量与推断。
- [五维联合实测](handoff/g1-joint-reach-search-20260920.md)：1,277 次尝试、原生复核、倾角对照与独立复测。
- [早期条件切片](handoff/g1-reachability-atlas-20260920.md)：最初 atlas 的覆盖范围与局限。
- [平地 ghost 手预览](handoff/g1-reachability-preview-20260920.md)：可达实验的环境和采样范围，只展示目标分布。
- [14 组动态伸手](handoff/g1-ghost-reach-execution-20260920.md)：6 组成功，8 组在伸手阶段超时；[初始化诊断](handoff/g1-startup-jump-diagnosis-20260920.md)确认了初始化缺陷。
- [修正重置后的同 14 组重跑](handoff/g1-ghost-reach-fixed-reset-repeat-20260920.md)：4 组成功、9 组超时、1 组初始化倒地，启动同步问题当时没有解决。
- [原专家实跑与录像起点修正](handoff/g1-original-expert-task-boundary-20260920.md)：原版行走抓取成功；前一轮见[同 14 组重录](handoff/g1-shared-expert-ready-repeat-20260920.md)。

## 交接与归档

- `handoff/`：各阶段的交接记录和计划，文件名带日期。
- [归档索引](archive/INDEX.md)：冻结的原始报告和被取代的入口文档，包括 2026-09-14 的[维护路线图](archive/roadmap_maintenance_20260914.md)。
- [第一次迁移](handoff/agent-maintenance-plan.md)、[维护交付](handoff/maintenance-completion.md)、[规范审计与探针](handoff/agent-guideline-completion-20260914.md)。

较新的说法或撤回不会自动比旧记录更可信。保留原始记录，链接到更正，并写清新核实的观察适用的范围。
