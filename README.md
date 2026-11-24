# locomanipulation：分支 selfwm（自身世界模型）

Unitree G1 人形机器人移动操作的仿真研究。这个分支做**自身世界模型**：全身控制器保持冻结，另外学一个只描述机器人自己的模型，给控制器一个命令，预测身体实际会到哪、手会落在哪，在出发前用它挑命令或改命令。模型和不用模型的做法放在公开基准上比，基准、控制器、感知来源都可以换。

和其他分支的关系：`main` 的主线是场景 RSI，这里不维护；`bench-options` 做 RSI 在公开基准上的评测，运行开关的写法和这里一样，两边的任务互不影响。

## 能跑哪些组合

一次运行由几个开关定下来，`python -m locomani.bench table` 打印完整的表和每一项的原因。

| 开关 | 现在能跑 | 登记为计划中 | 默认 |
|---|---|---|---|
| 基准 `bench` | SIMPLE 的四个抓取任务 | LocoManiLab、BiGym 2.0、HumanoidArena | SIMPLE |
| 全身控制器 `controller` | ScaleBFM、SONIC | AMO、GR00T-WBC；其他基准用自带的控制器 | ScaleBFM |
| 目标和抓取 `target` | 感知、仿真真值 | | 感知 |
| 感知的输入 `sensor` | 单目 RGB-D、双目 | | 单目 RGB-D |

默认组合是 SIMPLE、ScaleBFM、单目 RGB-D（2026-11-24 起）。2026-11-20 到 11-24 记录的数都是双目，重复它们要加 `--bench-set sensor=stereo`。

## 上手

1. 安装：照[安装指南](docs/setup.md)做准备、第 0–2 步、第 7 步、第 11 步和第 13 步。
2. 准备材料、跑默认组合、换组合：照[基准 runbook](docs/selfwm_bench.md)。跑一个任务的命令是

```bash
PYTHONPATH=src REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl SIMPLE_ROOT=$PWD/third_party/SIMPLE \
    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/simple_eval_20261112/simple_bench.py \
    TabletopGraspMP --method direct --out outputs/try/rows.jsonl
```

做什么：默认组合下跑 SIMPLE 桌面抓取的 30 局，直线伸手。得到什么：每局一行记录，开头打印这次的开关 `switches simple/perception-rgbd/...`。换控制器加 `--bench-set controller=sonic`，换双目加 `--bench-set sensor=stereo`。

主指标是 SIMPLE 四个抓取任务感知栏的成功局数，45 秒内和官方时限内都报，重点是 TabletopGrasp。现在的数在基准 runbook 的第 6 节。

## 这个分支的文档

| 文档 | 写什么 |
|---|---|
| [基准 runbook](docs/selfwm_bench.md) | 按开关跑组合：准备材料、跑一个任务、四个任务五种伸手做法、看结果、现在的数 |
| [自身模型抓取 runbook](docs/selfwm.md) | 房间抓取（11-11 之前的主线）和 SIMPLE 上每次实验当时的命令 |
| [自身模型抓取详解](docs/selfwm_design.md) | 自身模型是什么、怎么采数据和评测、试过没采用的、运行开关为什么这样分 |
| [实验条目](docs/handoff/notebook-20261120/) | 每次实验事先登记的预测和判据、原始输出、复核 |
| [开发记录](docs/handoff/selfwm-20261029.md)、[4090 的记录](docs/handoff/selfmodel-4090-20261123.md)、[SIMPLE 换 Isaac 画面的记录](docs/handoff/simple-isaac-eyes-20261120.md) | 按天记的实验、数字、机器和路径 |
| [基准的代码导航](docs/reference/change-map-bench.md) | 开关、各基准的能力表、SIMPLE 的接法分别在哪个文件 |

投稿计划（RSS 2027，第一阶段 2026-12-04）见[投稿故事与缺口](docs/handoff/rss-story-20261115.md)。那份是 11-15 写的，以连续伺服为基线；11-19 起不再展示伺服和查表这类做法，那一部分待改。

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
