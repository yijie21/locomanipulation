# locomanipulation

面向 **Unitree G1 的移动操作（locomanipulation）仿真与数据采集项目**：
让机器人走近桌子、抓取物体或完成放置，并提供单臂任务的数据转换、VLA 微调和评测工具。
主要使用 MuJoCo 运行物理仿真、SONIC v1.1 执行全身控制；另有 ScaleBFM 控制路线。
Isaac Sim 用于已录轨迹的离线渲染；完整分工见 [MuJoCo 与 Isaac 工作流程](docs/rendering.md#整体流程)。

[安装环境](#安装环境) · [快速开始](#快速开始) · [功能与文档](#功能与文档) · [查看运行结果](#查看运行结果)

## 功能与文档

一条线一页，**采集 → 转换 → 训练 → 评测都在同一页里**，不用跳转：

| 想做什么 | 提供的功能 | 怎么跑 | 为什么这么设计 |
|---|---|---|---|
| 单臂 | 走近桌子、单手抓瓶；含 pick-and-place 变体 | [单臂](docs/single_arm.md) | [单臂详解](docs/single_arm_design.md) |
| 双臂 | 视觉指导走近、俯身抓篮、抬起并站直 | [双臂](docs/bimanual.md) | [双臂详解](docs/bimanual_design.md) |
| 导入 Blender 场景 | 把一个 `.blend` 房间变成 MuJoCo 物理场景与 Isaac 渲染场景 | [Blender 场景](docs/blender_scenes.md) | 同页 |
| 使用 ScaleBFM | 独立于 SONIC 的视觉抓瓶采集与评测路线 | [ScaleBFM](docs/scalebfm.md) | 同页 |
| MuJoCo 仿真与 Isaac 渲染 | EmbodiedGen 背景导入、轨迹重渲与视频导出 | [仿真与渲染](docs/rendering.md) | 同页 |

逐轮实测数字和历次裁定在台账：[单臂](docs/ledger/bottle.md)、[双臂](docs/ledger/crate.md)、[ScaleBFM](docs/ledger/scalebfm.md)、[Blender 场景](docs/ledger/blender_scenes.md)。

双臂视觉抓篮子目前是**视频试验流程**：低桌配置使用普通篮子，允许仿真信息辅助规划，
按相机可见性约束动作；尚未提供经过验证的 LeRobot 训练数据集。各任务的适用范围和已知问题见对应指南，最近的试验结果见 [项目状态](docs/STATE.md)。

## 安装环境

准备 Linux、NVIDIA GPU 与可用驱动，以及 Git、Git LFS、CMake 和 uv。
Git LFS 用于下载上游机器人网格和库；本仓库附带的场景包与房间背景直接随 Git 提供。
先克隆本仓库，再按 [完整安装指南](docs/setup.md) 配置所需组件：

```bash
git clone https://github.com/yijie21/locomanipulation.git
cd locomanipulation
```

| 要运行的功能 | 安装范围 |
|---|---|
| SONIC 单臂 / 双臂仿真采集 | 安装指南第 **0–4 步**：路径、上游仓库、仿真与采集环境、SONIC 权重及补丁、C++ deploy、场景 |
| GR00T / VLA 训练与评测 | 在上述基础上完成第 **5 步**，安装微调环境 |
| Isaac 离线渲染 | 额外完成第 **7 步**，安装 SIMPLE 渲染运行时，再按 [资产构建](docs/rendering.md#资产构建) 生成 USD |
| ScaleBFM 采集与评测 | 按 [ScaleBFM 指南](docs/scalebfm.md) 配置独立后端、权重和运行资产 |
| 重新生成场景或训练 SONIC | 按安装指南第 **6 步**及相应生成工具配置可选环境 |

安装指南包含逐步命令和每一步的验证方法。`LOCOMANI_WORKSPACE` 可指定依赖、日志和缓存所在目录；
默认使用 `/workspace`。已有场景可直接安装，无需先运行 EmbodiedGen。

## 快速开始

一次运行就是一份实验文件：`configs/experiments/<name>.yaml` 里写着这次运行读的**每一个**值，
没有继承也没有组合。新实验就是复制一份改几行。先完成安装指南第 0–4 步，在仓库根目录执行：

```bash
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 打过夹爪补丁的 deploy

# 先看会发生什么：打印每个阶段将执行的命令，什么都不写
python scripts/pipeline.py configs/experiments/bottle_simple_gate8.yaml --gpu 1 --dry-run

# 跑起来：采集 → 转换 → 训练 → 闭环评测
python scripts/pipeline.py configs/experiments/bottle_simple_200.yaml --gpu 1

# 单个阶段也能单独跑
python scripts/collect_bottle.py configs/experiments/bottle_simple_gate8.yaml --gpu 1
python scripts/eval_bottle.py outputs/runs/bottle_simple_200 --gpu 1
```

并行就是**多开一个终端**、换一张卡：`--gpu 0` 和 `--gpu 1` 各跑一份实验，没有作业文件也没有调度器。
阶段产物已存在就自动跳过，要重跑就删掉那个产物（数据集目录、`model/`、`eval/closed_loop.json`）。

首次启动 SONIC 可能需要约 10 分钟构建 TensorRT engine，后续会使用缓存。`collect` 和 `eval`
自己启停仿真、控制器和导出器（`src/locomani/stack/`，手工控制用
`python -m locomani.stack.cli up|status|down --instance N`）。完整流程见 [单臂](docs/single_arm.md) / [双臂](docs/bimanual.md)。

## 查看运行结果

以上示例的结果保存在 `outputs/runs/<name>/`：

| 路径 | 内容 |
|---|---|
| `config.yaml` | 实验文件的逐字节副本，这次运行的 provenance（`--name` 只改目录名，副本里的 `name:` 仍是来源实验的） |
| `summary.json` | 每个阶段一条记录：保留数、产出率、失败家族、数据集、checkpoint、成功率 |
| `stack_env.json`、`commands.jsonl`、`timings.jsonl` | 仿真栈告诉子进程的全部设置、每个子进程的命令行与耗时 |
| `collect/manifest_00.jsonl`、`collect/videos_00/`、`collect/state_trace_00/` | 每局的成败与原因、第三人称录像（含机器人相机画中画）、仿真状态轨迹。**双臂**写在顶层（`manifest_00.jsonl`、`videos_00/`），状态轨迹是全批共用的 `trace/` |
| `train_dataset/`、`model/checkpoint-<steps>/`、`eval/closed_loop.json` | 转换后的 LeRobot 数据集、微调 checkpoint、闭环评测结果（录像在 `eval/videos/`） |

比较多次运行就是 `cat outputs/runs/*/summary.json`。产物是否齐全可以查契约：
`python scripts/artifact_contract.py check outputs/runs/<name> bottle_eval`。
分享视频前按 [视频导出步骤](docs/rendering.md#video-export) 生成每段不超过 500,000 字节 的预览，原片保留。
需要调整参数时查 [配置字段表](docs/reference/config_fields.md)（它由 dataclass 生成，覆盖 task / backend / eval；`collect` 和 `train` 两块是普通 YAML，逐字段说明在 `configs/experiments/bottle_simple_200.yaml` 的注释里）；排查安装问题先看 [安装指南](docs/setup.md) 的对应验证步骤。

## 仓库与更多文档

`scripts/` 是每个阶段一个的扁平脚本（采集 / 转换 / 训练 / 评测 / `pipeline.py`），`src/locomani/` 是仿真、任务、专家和栈管理（`stack/`）代码，`configs/experiments/` 是全展开的实验文件、`configs/{task,backend,eval}/` 是按名字引用的 profile 库，`assets/` 是随仓库提供的资产。
`third_party/` 存放另行安装的上游依赖，`outputs/` 存放运行产物，两者都不上传 Git。
完整文档目录见 [文档索引](docs/README.md)，后续计划见 [路线图](docs/ROADMAP.md)。

需要修改代码或贡献功能时，从 [开发入口 AGENTS.md](AGENTS.md) 开始。已知文件的局部修改可直接进入；
归属不明或新增功能时查 [代码导航](docs/reference/change-map.md)，改变接口或配置格式时查 [开发规范](docs/reference/standards.md)。
