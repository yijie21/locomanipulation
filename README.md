# locomanipulation

Unitree G1 人形机器人的移动操作仿真。现在的主线是**场景 RSI**：机器人在一套公寓场景里，由 Claude 提出任务、写技能代码，再由验收测试决定留不留下改动，能做成的任务一轮比一轮多。
物理仿真用 MuJoCo，全身控制器用 ScaleBFM，场景来自 Aria Digital Twin 的公寓；报告视频用 Unreal 渲染。

## 主线：场景 RSI

先照[安装指南](docs/setup.md)装好“场景 RSI”一列要求的步骤，再按 runbook 从 L0 冒烟跑起。

| 文档 | 写什么 |
|---|---|
| [场景 RSI runbook](docs/rsi.md) | 复现的全部命令，分四级：L0 冒烟、L1 一个任务族的内环、L2 一轮外环、L3 从头复现；另有核对已记录结果和看视频的命令 |
| [场景 RSI 详解](docs/rsi_design.md) | 系统怎么工作、为什么这样设计：分层、内环的五个角色、三个起点池、验收门、写回与回归、人插手的地方、试过没采用的 |
| [进展报告](docs/handoff/scene-rsi-report-20261019.md) | 进化到哪一步：每次被接受的改动、做成了哪些任务、穿模和接触问题的修正与重测 |
| [场景文件格式](docs/handoff/scene-file-format-20261003.md) | 场景文件的每个字段，建场景后的面板、碰撞盒、沉降、检查几步各做什么 |
| [方案与开发记录](docs/handoff/scene-rsi-plan-20261016.md) | 按天记的决策、实测和改动，D1–D13 的取舍 |
| [从原子技能开始的抓取进化](docs/handoff/grasp-atomic-line-20261023.md) | 10-20 起的那条线：任务族、四次运行的结果、每版改了什么、还败在哪、从空种子跑和续跑的命令、做好还没跑的部件 |

## 本分支：自我模型抓取

分支 `selfwm`。全身控制器 ScaleBFM 保持冻结，在它外面加走路模型、抓取选择和只读关节的连续伺服，让它抓得准：没见过的摆放上 19 % 提到 80 %。

| 文档 | 写什么 |
|---|---|
| [自我模型抓取 runbook](docs/selfwm.md) | 准备材料、跑一组评测、看结果和看过程的命令 |
| [自我模型抓取详解](docs/selfwm_design.md) | 误差怎么分两类、系统组成、一局怎么跑、伺服怎么不读真值、评测方法、结果、试过没采用的、已知限制 |
| [开发记录](docs/handoff/selfwm-20261029.md) | 10-29 起按天记的实验、数字和复现命令 |

## 其他文档

- [术语表](docs/reference/glossary.md)：文档里反复出现的词。
- [安装](docs/setup.md)：每一步的命令、做什么、怎么确认装好了；开头的表写着每条线要装哪几步。
- 对照与评估：[ScaleBFM 和 SONIC 的对比](docs/handoff/controller-compare-20261013.md)、[AMO 换掉 ScaleBFM 的试验](docs/handoff/amo-compare-20261020.md)、[ASPIRE 评估](docs/handoff/aspire-v1-review-20261019.md)。
- [历史记录与开发者索引](docs/README.md)：阶段交接记录、归档文档、文档的维护规则。
- [AGENTS.md](AGENTS.md)：改代码从这里开始，包括代码导航、开发规范和当前进度。

## 归档：VLA 数据采集与训练（2026-10-20 起暂不使用）

2026-10-16 之前的主线：机器人走到目标前抓起物体，采集示范数据，微调 GR00T 视觉-语言-动作策略（VLA），在仿真里闭环评测；全身控制用 NVIDIA 的 SONIC。下面的文档都留在原处，但不再随代码更新，命令可能已经跑不通。

**三条主线的 runbook 与详解**

| 主线 | 做什么 | runbook | 详解 |
|---|---|---|---|
| bottle | 单臂：走到桌前，单手抓起水瓶 | [bottle](docs/single_arm.md) | [bottle 详解](docs/single_arm_design.md) |
| crate | 双臂：走到矮桌前，俯身用双手抓起筐并站直 | [crate](docs/bimanual.md) | [crate 详解](docs/bimanual_design.md) |
| warehouse | 双臂：从托盘的一条边走近，双手扣住纸箱沿，抬起并站直 | [warehouse](docs/warehouse.md) | [warehouse 详解](docs/warehouse_design.md) |

**通用**：[运行机制](docs/running.md)（实验文件、参数、并行、仿真栈）、[仿真与渲染](docs/rendering.md)（Isaac 与 Unreal 出图、视频导出）、[配置字段表](docs/reference/config_fields.md)、[旧数据转成当前格式](docs/migrate_old_data.md)、[与官方教程的对照](docs/official_comparison.md)、[真机部署](src/real_robot/README.md)。

**支线**：[双臂抓箱的参数范围](docs/box_grasp_envelope.md)、[场景能力扫参](docs/scene_sweep.md)、[Blender 场景](docs/blender_scenes.md)、[ScaleBFM 视觉抓瓶](docs/scalebfm.md)、[任意物体抓取](docs/object_grasp.md)（[详解](docs/object_grasp_design.md)）、[walk 场景](docs/walk.md)、[可拼接的技能（SONIC 版）](docs/skills.md)。

**已结束的专题实验**：[箱沿抓点规划](docs/box_rim_planning.md)、[箱沿抓取执行](docs/box_rim_execution.md)、[双臂可达实验](docs/bimanual_reachability.md)、[双臂前伸包络](docs/reach_envelope.md)。

**台账**：[bottle](docs/ledger/bottle.md)、[crate 与 warehouse](docs/ledger/crate.md)、[pick-and-place](docs/ledger/pick_place.md)、[ScaleBFM](docs/ledger/scalebfm.md)、[Blender 场景](docs/ledger/blender_scenes.md)。
