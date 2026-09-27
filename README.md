# locomanipulation

面向 **Unitree G1 的移动操作（locomanipulation）仿真与数据采集项目**：让机器人走近桌子、抓取物体或完成放置，并提供单臂任务的数据转换、VLA 微调和评测工具。
主要使用 MuJoCo 运行物理仿真、SONIC v1.1 执行全身控制；另有 ScaleBFM 控制路线。Isaac Sim 只出图：离线重渲已录轨迹，或作为策略训练图与闭环评测图的来源（`images.source: isaac`，目前是双臂纸箱线）；完整分工见 [MuJoCo 与 Isaac 工作流程](docs/rendering.md#整体流程)。

[安装环境](#安装环境) · [快速开始](#快速开始) · [功能与文档](#功能与文档) · [查看运行结果](#查看运行结果)

## 功能与文档

一条线一页，**采集 → 转换 → 训练 → 评测都在同一页里**，不用跳转：

| 想做什么 | 提供的功能 | 怎么跑 | 为什么这么设计 |
|---|---|---|---|
| 单臂 | 走近桌子、单手抓瓶；含 pick-and-place 变体 | [单臂](docs/single_arm.md) | [单臂详解](docs/single_arm_design.md) |
| 双臂 | 视觉指导走近、俯身抓篮、抬起并站直 | [双臂](docs/bimanual.md) | [双臂详解](docs/bimanual_design.md) |
| 双臂抓纸箱（物流房间 / 台上） | 双臂扣住纸箱沿、抬起、站直；仿真时不渲染，数据集相机事后从轨迹补渲（MuJoCo 或 Isaac 出图）；默认配置与布局参数范围 | [warehouse](docs/warehouse.md) | [默认配置与参数范围](docs/box_grasp_envelope.md) |
| 可迁移双臂抓箱 | 箱沿抓点规划、原 expert 仿真执行、固定站姿五维可达搜索 | [规划与执行](docs/box_rim_planning.md) | [可达实验](docs/bimanual_reachability.md) |
| 导入 Blender 场景 | 物流间纸箱场景的 MuJoCo / Isaac 资产、Blender 导入与尺寸说明 | [Blender 场景](docs/blender_scenes.md) | 同页 |
| 使用 ScaleBFM | 独立于 SONIC 的视觉抓瓶采集与评测路线 | [ScaleBFM](docs/scalebfm.md) | 同页 |
| MuJoCo 仿真与 Isaac 渲染 | EmbodiedGen 背景导入、轨迹重渲与视频导出 | [仿真与渲染](docs/rendering.md) | 同页 |

逐轮实测数字和历次裁定在台账：[单臂](docs/ledger/bottle.md)、[双臂](docs/ledger/crate.md)、[ScaleBFM](docs/ledger/scalebfm.md)、[Blender 场景](docs/ledger/blender_scenes.md)。

双臂视觉抓篮子目前是**视频试验流程**：低桌配置使用普通篮子，允许仿真信息辅助规划，按相机可见性约束动作；尚未提供经过验证的 LeRobot 训练数据集。各任务的适用范围和已知问题见对应指南。
**接手开发和最近的试验结果从这三个文件开始**：[STATE.md](STATE.md)（当前进度、卡点、下一步和命令）、[TODO.md](TODO.md)（任务账本）、[DECISIONS.md](DECISIONS.md)（每次决策的理由与被否决的方案）。

## 安装环境

准备 Linux、NVIDIA GPU 与能跑 CUDA 13 的驱动、CUDA 13 运行时开发文件（安装指南第 3 步第 0 条先查），以及 Git、Git LFS、CMake 和 uv。
Git LFS 用于下载上游机器人网格和库；本仓库附带的场景包与房间背景直接随 Git 提供。
先克隆本仓库，再按 [完整安装指南](docs/setup.md) 配置所需组件：

```bash
git clone https://github.com/yijie21/locomanipulation.git
cd locomanipulation
git lfs pull  # 下载新场景的网格、贴图、USD 和 Blender 源文件
```

| 要运行的功能 | 安装范围 |
|---|---|
| SONIC 单臂 / 双臂 / warehouse 仿真采集与转换（到 LeRobot 数据集） | 安装指南第 **0–4 步**：路径、上游仓库、仿真与采集环境、SONIC 权重及补丁、C++ deploy、场景 |
| GR00T / VLA 训练与评测 | 在上述基础上完成第 **5 步**，安装微调环境 |
| Isaac 渲染（离线重渲；`images.source: isaac` 的训练图与评测图） | 额外完成第 **7 步**，安装 SIMPLE 渲染运行时（装有 Isaac Sim 的 Python），再按 [资产构建](docs/rendering.md#资产构建) 生成 USD；纸箱场景的 Isaac stage 见 [warehouse §1](docs/warehouse.md#1-准备一次) |
| ScaleBFM 采集与评测 | 按 [ScaleBFM 指南](docs/scalebfm.md) 配置独立后端、权重和运行资产 |
| 重新生成场景或训练 SONIC | 按安装指南第 **6 步**及相应生成工具配置可选环境 |

安装指南包含逐步命令和每一步的验证方法。`LOCOMANI_WORKSPACE` 可指定依赖、日志和缓存所在目录；
默认使用 `/workspace`。已有场景可直接安装，无需先运行 EmbodiedGen。

## 快速开始

一次运行就是一份实验文件：`configs/experiments/<name>.yaml` 里写着这次运行读的**每一个**值，没有继承也没有组合，新实验就是复制一份改几行。先完成安装指南第 0–4 步，再按[单臂 §1](docs/single_arm.md#1-准备一次)
建一次场景（`bottle_simple_*.yaml` 用的 `outputs/scene/sim_lowcom` 不由 `install_scene.sh` 建），在仓库根目录执行：

```bash
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 仅原作者机器；新机器按安装第 3 步打补丁后不要这行

# 无人值守跑完整条线：采集 → 转换 → 训练 → 闭环评测
bash scripts/run_overnight.sh configs/experiments/bottle_simple_200.yaml --gpu 0
# 或者一个阶段一个终端；采集是两个：起录制栈 / 跑 agent（每个脚本都吃 --dry-run）
bash scripts/start_stack.sh collect          configs/experiments/bottle_simple_200.yaml --gpu 0
bash scripts/single_arm/run_agent_server.sh  configs/experiments/bottle_simple_200.yaml --gpu 0
bash scripts/convert_data.sh outputs/runs/bottle_simple_200   # 不用卡
bash scripts/train.sh        outputs/runs/bottle_simple_200 --gpu 0
# 闭环评测三个终端：栈 / 策略服务 / 跑局
bash scripts/start_stack.sh eval             outputs/runs/bottle_simple_200 --gpu 0
bash scripts/run_vla_server.sh               outputs/runs/bottle_simple_200 --gpu 0
bash scripts/single_arm/run_vla_inference.sh outputs/runs/bottle_simple_200 --gpu 0
```

每个脚本都是十几行 bash、一个 `exec`，直接调功能代码，没有一层套一层；`run_overnight.sh` 是**唯一**
会调用其它脚本的脚本（无人值守批量用），读代码请从单阶段脚本读起。并行就是多开一个终端换一张卡，
没有作业文件也没有调度器；阶段产物已存在就自动跳过，要重跑就删掉那个产物。

首次启动 SONIC 可能需要约 10 分钟构建 TensorRT engine，后续会使用缓存。仿真栈由 `start_stack.sh collect` /
`start_stack.sh eval` 起、由驱动脚本或 `stop_stack.sh` 收（`src/locomani/stack/`）。完整流程见 [单臂](docs/single_arm.md) / [双臂](docs/bimanual.md) / [warehouse](docs/warehouse.md)。

## 查看运行结果

以上示例的结果保存在 `outputs/runs/<name>/`：

| 路径 | 内容 |
|---|---|
| `config.yaml` | 实验文件的逐字节副本，这次运行的 provenance（`--name` 只改目录名，副本里的 `name:` 仍是来源实验的） |
| `summary.json` | 每个阶段一条记录：保留数、产出率、失败家族、数据集、checkpoint、成功率 |
| `stack_env.json`、`commands.jsonl`、`timings.jsonl` | 仿真栈告诉子进程的全部设置、每个子进程的命令行与耗时 |
| `collect/manifest_00.jsonl`、`collect/videos/`、`collect/state_trace/` | 每局的成败与原因、第三人称录像（含机器人相机画中画）、仿真状态轨迹。manifest 按批分；录像和轨迹一个运行一个目录，因为栈跨批存活。**双臂**写在顶层 |
| `train_dataset/`、`model/checkpoint-<steps>/`、`eval/closed_loop.json` | 转换后的 LeRobot 数据集、微调 checkpoint、闭环评测结果（录像在 `eval/videos/`） |

比较多次运行就是 `cat outputs/runs/*/summary.json`；产物是否齐全查契约：`python scripts/artifact_contract.py check outputs/runs/<name> bottle_eval`。
分享视频前按 [视频导出步骤](docs/rendering.md#video-export) 生成每段不超过 500,000 字节的预览，原片保留；调参数查 [配置字段表](docs/reference/config_fields.md)（由 dataclass 生成，覆盖 task / backend / eval）。

## 仓库与更多文档

`scripts/` 是每个阶段一个的扁平 bash（采集 / 转换 / 训练 / 评测 / `run_overnight.sh`），`src/locomani/` 是仿真、任务、专家和栈管理（`stack/`）代码，`configs/experiments/` 是全展开的实验文件、`configs/{task,backend,eval}/` 是按名字引用的 profile 库，`assets/` 是随仓库提供的资产。
`third_party/` 存放另行安装的上游依赖，`outputs/` 存放运行产物，两者都不上传 Git。
完整文档目录见 [文档索引](docs/README.md)，后续计划见 [路线图](docs/ROADMAP.md)；文档里反复出现的名词（栈、deploy、exporter、trace、token、镜像……）见 [术语表](docs/reference/glossary.md)。

需要修改代码或贡献功能时，从 [开发入口 AGENTS.md](AGENTS.md) 开始。已知文件的局部修改可直接进入；归属不明或新增功能时查 [代码导航](docs/reference/change-map.md)，改变接口或配置格式时查 [开发规范](docs/reference/standards.md)。
