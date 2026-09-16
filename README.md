# locomanipulation

面向 **Unitree G1 的移动操作（locomanipulation）仿真与数据采集项目**：
让机器人走近桌子、抓取物体或完成放置，并提供单臂任务的数据转换、VLA 微调和评测工具。
主要使用 MuJoCo 运行物理仿真、SONIC v1.1 执行全身控制；另有 ScaleBFM 控制路线。
Isaac Sim 用于已录轨迹的离线渲染；完整分工见 [MuJoCo 与 Isaac 工作流程](docs/rendering.md#整体流程)。

[安装环境](#安装环境) · [快速开始](#快速开始) · [功能与文档](#功能与文档) · [查看运行结果](#查看运行结果)

## 功能与文档

| 想做什么 | 提供的功能 | 从这里开始 |
|---|---|---|
| 单臂采集 | 走近桌子、单手抓瓶，单局与批量录制 | [单臂采集指南](docs/collection.md) |
| 双臂采集 | 视觉指导走近、俯身抓篮、抬起并站直；含其他专家场景与回放 | [双臂采集指南](docs/crate_collection.md) |
| Pick-and-place 采集 | 走近桌子、右手抓起罐头并放入篮子，单局与批量录制 | [Pick-and-place 采集指南](docs/pick_place_collection.md) |
| 使用 ScaleBFM | 独立于 SONIC 的视觉抓瓶采集与评测路线 | [ScaleBFM 指南](docs/scalebfm_collection.md) |
| 整理训练数据 | 转换 LeRobot 数据集，检查图像、状态、动作与数据格式 | [数据集转换与检查](docs/dataset.md) |
| 训练和测试 VLA | GR00T 微调、离线预测评测、仿真闭环评测 | [训练](docs/training.md) · [评测](docs/evaluation.md) |
| 多卡并行、多配置对比 | 一个 job 文件跑多个实验，按 manifest 跳过已完成阶段，结果注册表 | [运行基础设施指南](docs/infra.md) · `configs/jobs/example.yaml` |
| MuJoCo 仿真与 Isaac 渲染 | 仿真与采集流程、EmbodiedGen 背景导入、轨迹重渲与视频导出 | [仿真与渲染指南](docs/rendering.md) |

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
| ScaleBFM 采集与评测 | 按 [ScaleBFM 指南](docs/scalebfm_collection.md) 配置独立后端、权重和运行资产 |
| 重新生成场景或训练 SONIC | 按安装指南第 **6 步**及相应生成工具配置可选环境 |

安装指南包含逐步命令和每一步的验证方法。`LOCOMANI_WORKSPACE` 可指定依赖、日志和缓存所在目录；
默认使用 `/workspace`。已有场景可直接安装，无需先运行 EmbodiedGen。

## 快速开始

所有阶段只有一个命令：`python -m locomani.cli <collect|convert|train|eval|stats> [覆盖项]`。一次运行由七个配置组组成
（`task` × `embodiment` × `scene` × `backend` × `collect` × `train` × `eval`，各在 `configs/<组>/` 下一个 YAML 一个版本），
`configs/experiment/<name>.yaml` 固定一整套组合，用 `+experiment=<name>` 选中。先完成安装指南第 0–4 步，在仓库根目录执行：

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 只打印组合后的配置，不运行
"$VENV_SIM_PY" -m locomani.cli show +experiment=bottle_simple_lowcom

# 单臂抓瓶：固定样本库 8 局，GPU 1
"$VENV_SIM_PY" -m locomani.cli collect +experiment=bottle_simple_lowcom name=my_first_run gpu=1

# 同一个运行目录上接着做：转换 → 训练 → 闭环评测 → 汇总
"$VENV_SIM_PY" -m locomani.cli convert name=my_first_run
"$VENV_SIM_PY" -m locomani.cli train   name=my_first_run gpu=1
"$VENV_SIM_PY" -m locomani.cli eval    name=my_first_run gpu=1 eval.rounds=20

# 多个实验并行（一卡一 job，起跑前实测 CPU / pid / 显存）；`launch ls` 看注册表
"$VENV_SIM_PY" -m locomani.launch run configs/jobs/example.yaml --gpus 0,1
```

首次启动 SONIC 可能需要约 10 分钟构建 TensorRT engine，后续会使用缓存。每个阶段是一个独立进程：`collect` 和 `eval`
自己启停仿真、控制器和导出器（`src/locomani/stack/`，手工控制用 `python -m locomani.stack.cli up|status|down --instance N`），
结果写进 `outputs/runs/<name>/<阶段>/manifest.json`，下一阶段只读这个文件。双臂抓篮子的命令见 [双臂采集指南](docs/crate_collection.md)。

## 查看运行结果

以上示例的结果保存在 `outputs/runs/<name>/`：

| 路径 | 内容 |
|---|---|
| `run_config.yaml`、`run_config.json` | 本次运行完整解析后的配置（provenance） |
| `<阶段>/manifest.json` | 该阶段依赖的配置组 hash、上游 manifest hash、产物路径、指标、起跑时的 CPU / pid / GPU 实测 |
| `collect/manifest_00.jsonl`、`collect/videos_00/`、`collect/state_trace_00/` | 每局的成败与原因、第三人称录像（含机器人相机画中画）、仿真状态轨迹 |
| `train_dataset/`、`model/checkpoint-<steps>/`、`eval/closed_loop.json` | 转换后的 LeRobot 数据集、微调 checkpoint、闭环评测结果（录像在 `eval/videos/`） |

并行 launcher 每完成一个阶段在 `outputs/registry.jsonl` 追加一行（`launch ls` 查看，`launch compare A B --stage eval` 并排比较）。
分享视频前按 [视频导出步骤](docs/rendering.md#video-export) 生成每段不超过 500,000 字节 的预览，原片保留。
需要调整参数时查 [配置字段表](docs/reference/config_fields.md)；排查安装问题先看 [安装指南](docs/setup.md) 的对应验证步骤。

## 仓库与更多文档

`src/locomani/` 是仿真、任务、阶段（`stages/`）、栈管理（`stack/`）和 launcher（`launch.py`）代码，`configs/` 是按组划分的运行配置（YAML，`conf.py` 用 Hydra 组合），`assets/` 是随仓库提供的资产。
`third_party/` 存放另行安装的上游依赖，`outputs/` 存放运行产物，两者都不上传 Git。
完整文档目录见 [文档索引](docs/README.md)，后续计划见 [路线图](docs/ROADMAP.md)。

需要修改代码或贡献功能时，从 [开发入口 AGENTS.md](AGENTS.md) 开始。已知文件的局部修改可直接进入；
归属不明或新增功能时查 [代码导航](docs/reference/change-map.md)，改变接口或配置格式时查 [开发规范](docs/reference/standards.md)。
