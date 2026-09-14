# locomanipulation

面向 **Unitree G1 的移动操作（locomanipulation）仿真与数据采集项目**：
让机器人走近桌子、抓取物体或完成放置，并提供单臂任务的数据转换、VLA 微调和评测工具。
主要使用 MuJoCo 运行物理仿真、SONIC v1.1 执行全身控制；另有 ScaleBFM 控制路线。
Isaac Sim 用于已录轨迹的离线渲染。

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
| 渲染和分享视频 | 构建 Isaac 场景、离线重渲、压缩下载预览 | [渲染与视频导出](docs/rendering.md) |

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

下面以**30 cm 低桌上的双臂抓篮子，运行 3 次尝试**为例。先完成安装指南第 0–4 步，
在仓库根目录的 Bash 终端执行。示例使用 GPU 0、实例 1；共享机器上先确认它们空闲。

```bash
export INSTANCE=1
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 构建原始篮子场景，准备保持桌面宽度不变的 30 cm 低桌
"$VENV_SIM_PY" -m locomani.tasks.crate.bimanual_scene outputs/scene/crate_bimanual
"$VENV_SIM_PY" scripts/experiments/crate_table_height_20260913.py \
  outputs/scene/crate_bimanual outputs/scene/crate_table30 --height 0.30

# 启动仿真与控制器，采集完整的走近、抓取过程
"$VENV_SIM_PY" -m locomani.pipeline.run_crate_pilot --expert visual \
  --config configs/tasks/crate_visual_low_reach.json \
  --scene outputs/scene/crate_table30 --out outputs/crate_first_run \
  --instance "$INSTANCE" --gpu 0 --rounds 3 --seed 101
```

首次启动 SONIC 可能需要约 10 分钟构建 TensorRT engine，后续会使用缓存。
runner 负责启停自己的实例并记录每次尝试；失败也会留下结果。重跑时换一个新的输出目录。
单臂任务从 [单臂采集指南](docs/collection.md) 开始；本例的批量采集、参数与回放见 [双臂采集指南](docs/crate_collection.md)。

## 查看运行结果

以上示例的结果保存在 `outputs/crate_first_run/`：

| 路径 | 内容 |
|---|---|
| `videos_00/` | 第三人称录像，包含机器人相机的画中画 |
| `observations/` | 专家实际看到的 RGB、时间戳、机器人状态和动作参考 |
| `manifest_00.jsonl`、`attempts.jsonl` | 每次尝试的标签、成败及失败原因 |
| `trace/` | 仿真状态轨迹，可用于检查与离线重渲 |
| `config_snapshot.json`、`deploy_receipts.jsonl` | 本次配置、代码和实际控制模型的记录 |

分享视频前按 [视频导出步骤](docs/rendering.md#video-export) 生成每段不超过 500,000 字节 的预览，原片保留。
需要调整任务参数时查 [配置字段表](docs/reference/config_fields.md)；排查安装问题先看 [安装指南](docs/setup.md) 的对应验证步骤。

## 仓库与更多文档

`src/locomani/` 是仿真和任务代码，`configs/` 是运行配置，`assets/` 是随仓库提供的资产。
`third_party/` 存放另行安装的上游依赖，`outputs/` 存放运行产物，两者都不上传 Git。
完整文档目录见 [文档索引](docs/README.md)，后续计划见 [路线图](docs/ROADMAP.md)。

需要修改代码或贡献功能时，从 [开发入口 AGENTS.md](AGENTS.md) 开始。已知文件的局部修改可直接进入；
归属不明或新增功能时查 [代码导航](docs/reference/change-map.md)，改变接口或配置格式时查 [开发规范](docs/reference/standards.md)。
