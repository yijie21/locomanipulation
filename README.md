# locomanipulation

Unitree G1 人形机器人的移动操作仿真：机器人走到目标前、抓起物体，一路采集示范数据，再用这些数据微调 GR00T 视觉-语言-动作策略（VLA），最后在仿真里闭环评测。
物理仿真用 MuJoCo，全身控制用 NVIDIA 的 SONIC；Isaac Sim 只负责出图。

## 三条主线

每条线一页 runbook，按顺序跑完建场景、采集、转换、训练、评测。先照[安装指南](docs/setup.md)装好环境。

| 主线 | 做什么 | runbook | 需要 |
|---|---|---|---|
| bottle | 单臂：走到桌前，单手抓起水瓶 | [bottle](docs/single_arm.md) | 1 张卡 |
| crate | 双臂：走到矮桌前，俯身用双手抓起筐并站直 | [crate](docs/bimanual.md) | 1 张卡 |
| warehouse | 双臂：在物流房间里从托盘四条边中随机一条走近，双手扣住纸箱沿，抬起并站直 | [warehouse](docs/warehouse.md) | 2 张卡、Isaac |

## 其他文档

**通用**

- [安装](docs/setup.md)：从零装好运行三条主线所需的全部环境，每一步都附验证命令。
- [运行机制](docs/running.md)：实验文件、命令行参数、跳过与重跑、并行、仿真栈、评测时钟。想改参数或同时跑几个实验时读。
- [仿真与渲染](docs/rendering.md)：MuJoCo 和 Isaac 各管什么，怎样用 Isaac 重渲已录的动作，评测时的 Isaac 镜像，怎样导出视频。
- [术语表](docs/reference/glossary.md)：文档里反复出现的词，例如 deploy、exporter、trace、token。
- [配置字段表](docs/reference/config_fields.md)：实验文件里每个字段的含义和默认值，由代码自动生成。

**各主线的详解**：为什么这样设计、试过什么被否掉、runbook 里省略的细节。

- [bottle 详解](docs/single_arm_design.md)：专家的控制逻辑、观测与动作、训练和评测的细节、数据集格式。
- [crate 详解](docs/bimanual_design.md)：视觉专家的信息来源与连续控制、双臂闭环评测器、评测的时间模型。
- [warehouse 详解](docs/warehouse_design.md)：和 crate 的不同之处、补渲、MuJoCo 图与 Isaac 图、评测前要核对的项、对照实验的做法。

**支线**：不在三条主线上，但还在用。

- [双臂抓箱的参数范围](docs/box_grasp_envelope.md)：台上抓纸箱的默认配置，成功率随箱子尺寸、站位、朝向等布局参数的变化，以及布场景时该怎么取值。
- [场景能力扫参](docs/scene_sweep.md)：上一篇结果的扫参方法和运行命令。
- [Blender 场景](docs/blender_scenes.md)：把 Blender 做的房间导出成 MuJoCo 和 Isaac 两套场景，物流房间的尺寸和已知问题。
- [ScaleBFM](docs/scalebfm.md)：不用 SONIC、改用 ScaleBFM 控制器的视觉抓瓶采集与评测。
- [walk 场景](docs/walk.md)：只有一台 G1 的空场景，用来测 SONIC 的行走性能。

**已结束的专题实验**：留作查阅，现在的流程不用。

- [箱沿抓点规划](docs/box_rim_planning.md)、[箱沿抓取执行](docs/box_rim_execution.md)：按箱沿几何和机器人停稳后的状态规划双手抓点，并在仿真里低速接近、合爪、试提（2026-09-20 至 22）。
- [双臂可达实验](docs/bimanual_reachability.md)、[双臂前伸包络](docs/reach_envelope.md)：站着不动时双手能够到哪里，含测量方法和复现步骤（2026-09-20 至 24）。

**台账与开发**

- 台账：每一轮实验的实测数字和结论。[bottle](docs/ledger/bottle.md)、[crate 与 warehouse](docs/ledger/crate.md)、[pick-and-place](docs/ledger/pick_place.md)、[ScaleBFM](docs/ledger/scalebfm.md)、[Blender 场景](docs/ledger/blender_scenes.md)。
- [历史记录与开发者索引](docs/README.md)：阶段交接记录、归档文档、文档的维护规则。
- [AGENTS.md](AGENTS.md)：改代码从这里开始，包括代码导航、开发规范和当前进度。
