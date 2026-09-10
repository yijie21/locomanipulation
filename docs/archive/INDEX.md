# 归档文档索引

本索引收集了文档重组期间冻结的历史诊断、设计讨论和实验报告。这些文档的内容未作修改，仅用于历史查找；活跃的进行中计划保留在 `docs/handoff/`，agent 规则保留在 `docs/agents/`。

## 已归档文档

这些文档记录了过去的诊断过程、备选方案和实验结果，不再作为运行时参考，但保存供历史追溯。

- [Planner 手臂参考的实际响应](arm_response.md) — SONIC 链路手臂位姿控制的实测误差和可跟踪性诊断 (dated 2026-09-06)
- [双手搬箱 expert 设计(A–D)](bimanual_grasp_design.md) — 双手物体搬运场景中 expert 行为规范和数据记录要求的设计
- [机器人是怎么"边想边动"的:blocking 与 async 图解](client-mechanisms.md) — blocking 和 async 两种客户端时序差异及其对推理期间执行的影响分析
- [数据采集与 VLA 的验收收尾单](collection_acceptance.md) — 专家数据源、训练集一致性、VLA 闭环评估的三项可验收条件 (dated 2026-09-06)
- [Collection transition timing](collection_timing.md) — 采集过程中走近到下蹲的时序控制和抓取接触准则的实验配置
- [数据采集的执行容错原则](collection_tolerance_principles.md) — 执行与规划中不同的接受条件和容差原则，防止伪造成功 (dated 2026-09-06)
- [动作控制流的渐进重构](control_flow_design.md) — 桌面任务控制从脚本实验中逐步分离的架构迁移和执行记录设计
- [下蹲与手臂协同接近（实验配置）](coupled_grasp_policy.md) — 骨盆和末端协同下蹲的闭环控制策略和实测验收结果 (dated 2026-09-06)
- [Retired diagnostic/tool script catalogue](diag_catalogue.md) — 重构期间删除的 43 个一次性诊断工具的清单和来源文献索引 (dated 2026-09-09)
- [Lockstep 评测:仿真时钟 + 模拟延迟](lockstep.md) — 逐帧确定性评测框架的设计与校准进展 (dated 2026-09-02)
- [Measured squat prediction experiment](measured_squat_prediction.md) — 实测下蹲后退的回归模型用于可达性预测的实验配置
- [官方 tutorials 逐步对照审计(2026-08-28)](official_parity_audit.md) — 官方文档与实现的系统对照，区分自研、遗漏、有意偏离和版本漂移 (dated 2026-08-28)
- [sidecar 位姿 ↔ parquet 对齐：结论与修法](pose_parquet_alignment.md) — 机器人位姿两个记录通道的对齐诊断与修复方案 (dated 2026-08-27)
- [动作切换进入执行前缀的对照](prefix_sampling_experiment.md) — 下蹲预测改善的边界采样实验设计和诊断框架
- [从 5.9% 到 47.5%：G1 loco-manipulation VLA 的四轮进展报告](progress_report_2026-08.md) — 四轮闭环实验的成功率进展、失败分类和原始数据源查证 (dated 2026-08-27)
- [桌面接近与运输可行性](reach_feasibility.md) — 接近、下蹲、运输各阶段的可达性定义和碰撞检查机制
- [双臂准备抓取基线（实验配置）](ready_grasp_baseline.md) — 序列控制的双臂准备抓取策略与姿态先于站位的设计原则
- [双手准备姿势的数据与部署初始化](ready_grasp_initialization.md) — 双臂采集配置与评估部署的初始化 token 对齐与验证
- [最新策略的小规模学习实验](ready_grasp_pilot.md) — 固定预算下采集、训练、评估的小规模学习流程设计
- [Real-Time Action Streaming:async client 的三个机制](realtime-action-streaming.md) — 实时分块、超时补偿索引和端点框架的三个执行时机制设计
- [可恢复的双臂准备抓取策略](recoverable_grasp_policy.md) — 当前站位不可达时的局部搜索恢复和二次尝试策略
- [真实感渲染全景调研（2026-08）](rendering_survey.md) — 四个渲染家族（网格、3DGS、生成式、神经场）的特点和 sim2real 应用前景 (dated 2026-08)
- [Research Statement — Scripted Expert 何以替代人类 Teleop（草稿 v1）](research_statement.md) — 脚本化专家替代遥操作数据的理论框架和三条核心主张
- [Scripted Expert 创新点定位讨论记录](scripted_expert_innovation_discussion.md) — Claude 与用户对研究创新点的深化讨论和接缝分解分析 (dated 2026-08-26)
- [SONIC + GR00T / SRPO / DAgger / Agent-based VLA 讨论记录](sonic_gr00t_srpo_dagger_agent_vla_discussion.md) — ChatGPT 与用户的技术讨论记录，涉及 RL、DAgger 和 agent 路线选择
- [桌前站位动态验证](stance_validation.md) — 站位网格上的诊断方法与后续验收顺序定义
- [走近、下蹲和抓取的切换](transition_control.md) — 各阶段边界检查的分离和实验性保留下蹲过渡的执行设计

## docs/handoff/ (进行中的计划,不在归档里,原地保留)

这些文档记录了跨会话交接的活跃计划、设计决策和实现路线，保持动态更新。

- [Agentic expert:设计决策记录(2026-09-03)](../handoff/agentic-plan.md) — 任务扩展型 agent 的十二项核心决策与选项排除 (dated 2026-09-03)
- [Handoff: agentic iteration on the scripted policy and scene generation](../handoff/agentic-scripted-policy.md) — 脚本化策略和场景生成的 agent 迭代方向（Reading A/B）与代码地图
- [仓库重构与规范计划(2026-09-09 grill 定稿,待用户确认后执行)](../handoff/refactor-plan.md) — 架构重组、文档分类、多会话协同的十三项决定与五阶段执行计划 (dated 2026-09-09)
- [Refactor step 2 — retired-branch deletion candidates (evidence table, 2026-09-09)](../handoff/refactor-step2-candidates.md) — 淘汰分支和配置的删除裁定与依赖证据表 (dated 2026-09-09)
- [Refactor step 3 — knobs into config objects (decision 8)](../handoff/refactor-step3-config.md) — 配置对象设计、字段映射和环境变量白名单规则
- [Refactor step 4a — package layout move (decision 12)](../handoff/refactor-step4-layout.md) — 从 `src/teleop_free` 到 `src/locomani/` 的包重组目标结构与导入改写
- [ScaleBFM 分支交接:控制器替换、v3 单手 expert、准备用它采集训练数据](../handoff/scalebfm_handoff.md) — ScaleBFM 在仿真中的集成现状、资产清单和建议的 v3 采集配置 (dated 2026-09-08)
- [SONIC 采集的 lockstep 化:可加速、逐帧可复现,wall-clock 路径零改动](../handoff/sonic_lockstep_collection_plan.md) — lockstep 化采集的设计方案与当前阻碍分析（搁置中） (dated 2026-09-08)

## docs/agents/ (agent 规则,不在归档里,原地保留)

这些文档定义了 agent 的工作流规范和工具使用规则。

- [Domain Docs](../agents/domain.md) — 代码库领域文档的组织规范（CONTEXT.md、ADR、多上下文布局）与探索指南
- [Issue tracker: GitHub](../agents/issue-tracker.md) — GitHub Issues 的 CLI 操作、标签体系和 PR 作为 triage surface 的约定
- [Session workflow (token discipline)](../agents/session-workflow.md) — 单任务会话的五步工作流（读 STATE.md、分工、检查点、交接、清空）与成本管理原则
- [Triage Labels](../agents/triage-labels.md) — 五角色标签映射表（needs-triage、needs-info、ready-for-agent、ready-for-human、wontfix）

## ScaleBFM 线(2026-09-08 → 09-10,Codex 线程;数字全部在 `docs/ledger/scalebfm.md`)

- [ScaleBFM spike 交接](scalebfm_spike_handoff_20260908.md) — 分支会话写给主线:接口事实、v3 expert、资产清单、不要踩的坑 (dated 2026-09-08)
- [视觉 expert 设计](scalebfm_visual_expert_design.md) — privileged 信息边界、连续动作逻辑、可达性与人体风格、验收四条的设计 rationale
- [Goal 预注册协议](scalebfm_goal_plan.md) — A/B/C 固定预算比较的门槛表、池 seed 分配、恢复扰动定义、选择/确认统计规则 (dated 2026-09-08)
- [Goal 运行史 a00–a15](scalebfm_goal_history_20260908.md) — 开发池逐版本理由与 run ledger (dated 2026-09-08)
- [验收证据矩阵](scalebfm_acceptance_evidence.md) — 逐要求→证据→判定与最终核验顺序 (dated 2026-09-09)
- [视觉采集操作手册](scalebfm_visual_collection.md) — 观测/动作协议表、检查点与中断恢复(命令引用已丢失的目录)
- [yplus_center_v1 几何](scalebfm_yplus_collection_20260909.md) — 新采集分布的定义与隔离旧分布的规则 (dated 2026-09-09)
- [policy 审计](scalebfm_policy_audit_20260909.md) — 可见性门修复、学生接口检查、仍存在的先验 (dated 2026-09-09)
- ["检查采集视频"契约](scalebfm_video_check.md) — 固定功能的 8 条要求与复用/锁/恢复规则 (dated 2026-09-09)
- [Codex 线程终版交接](scalebfm_handoff_20260910.md) — 决策理由表、恢复顺序、事故批注链 (dated 2026-09-10)
