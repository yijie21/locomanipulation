# locomanipulation

Unitree G1 人形机器人的移动操作仿真。这个分支的主线是**自我模型抓取**：全身控制器 ScaleBFM 保持冻结，在它外面加走路模型、抓取选择和只读关节的连续伺服，让它抓得准，没见过的摆放上从 19 % 提到 80 %。
物理仿真用 MuJoCo，手是 Dex1 两指夹爪，物体来自 NVIDIA SimReady。

## 投稿计划：RSS 2027（2026-11-15 的判断）

第一阶段的 5 页扩展摘要 2026-12-04 截止，实验可以是初步的；终稿 2027-04-16 截止。完整的整理、证据表、日程和文献对照见[投稿故事与缺口](docs/handoff/rss-story-20261115.md)。

- 故事：通用全身控制器抓不准，差的东西分两种。把手送到位，靠反馈纠偏就够；站哪、用哪个抓取、用什么姿态，要在出发前由自身模型预测。
- 原来的说法不能再写：不用任何模型的纠偏在 294 个留出摆法里抓起 220 个，学出来的落点模型是 215 个。
- **最弱的一环是自身模型本身**：它在纠偏之上只多 15 个，统计上不显著；换成相机感知后是 179 对 218，输给纠偏。
- 现有材料够交第一阶段，不够交终稿。

| 2026-12-04 前要做的 | 现状 |
|---|---|
| 训练逐局选站位和抓取的结果预测模型，在难房间的 203 局上对纠偏。纠偏抓起 103 个，四个站位里事后挑最好的是 129 个 | 没做，训练数据已有 |
| 按停下、抬手看、再走一小段的流程补数据重训，重跑相机感知那张对比表 | 没做 |
| 清掉手臂逆运动学读的真实骨盆位置 | 没做 |
| 写 5 页摘要 | 没开始 |

终稿前还要补：真机 G1，SIMPLE 六个任务在官方环境里对 HITCH，第二个控制器，成组的统计。

## 主线：自我模型抓取

先照[安装指南](docs/setup.md)装好仿真环境和 ScaleBFM 权重，再按 runbook 准备材料、跑评测。

| 文档 | 写什么 |
|---|---|
| [自我模型抓取 runbook](docs/selfwm.md) | 准备材料、跑一组评测、看结果和看过程的命令 |
| [自我模型抓取详解](docs/selfwm_design.md) | 误差怎么分两类、系统组成、一局怎么跑、伺服怎么不读真值、评测方法、结果、试过没采用的、已知限制 |
| [开发记录](docs/handoff/selfwm-20261029.md) | 10-29 起按天记的实验、数字和复现命令 |

## 其他文档

- [术语表](docs/reference/glossary.md)：文档里反复出现的词。
- [安装](docs/setup.md)：每一步的命令、做什么、怎么确认装好了；开头的表写着每条线要装哪几步。
- 场景 RSI，`main` 分支的主线，这个分支不维护：[runbook](docs/rsi.md)、[详解](docs/rsi_design.md)、[进展报告](docs/handoff/scene-rsi-report-20261019.md)。
- 对照与评估：[ScaleBFM 和 SONIC 的对比](docs/handoff/controller-compare-20261013.md)、[AMO 换掉 ScaleBFM 的试验](docs/handoff/amo-compare-20261020.md)、[ASPIRE 评估](docs/handoff/aspire-v1-review-20261019.md)。
- [历史记录与开发者索引](docs/README.md)：阶段交接记录、归档文档、文档的维护规则。
- [AGENTS.md](AGENTS.md)：改代码从这里开始，包括代码导航、开发规范和当前进度。

## 归档：VLA 数据采集与训练（2026-10-20 起暂不使用）

`main` 分支 2026-10-16 之前的主线：机器人走到目标前抓起物体，采集示范数据，微调 GR00T 视觉-语言-动作策略（VLA），在仿真里闭环评测；全身控制用 NVIDIA 的 SONIC。下面的文档都留在原处，但不再随代码更新，命令可能已经跑不通。

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
