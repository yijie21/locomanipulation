# 单臂数据采集

让 G1 走近桌子，用右手抓起水瓶并保持。本文是 SONIC 单臂任务的使用入口；
抓起后放入篮子见 [pick-and-place](pick_place_collection.md)，双手抓篮子见 [双臂采集](crate_collection.md)。

## 1. 任务与能力

| 项目 | 当前支持 |
|---|---|
| 默认任务 | `bottle_only` 场景 + `configs/tasks/bottle_visible.json` |
| 动作过程 | 随机出生 → 走近可见目标 → 减速、准备抓取 → 伸手、闭合、抬起 → 退手与验收 |
| 专家输入 | 相机可见性、机器人状态与仿真物体信息辅助规划；不是纯 RGB 策略 |
| 采集能力 | 单局检查、按成功数批量采集、exporter session、manifest、视频与状态轨迹 |
| 后续链路 | 已有 LeRobot 转换、GR00T 训练与评测入口；新配置的数据质量和学习效果需单独验证 |

成功率取决于场景和控制配置。历史实验数字见文末，不作为本页命令的成功率承诺。

## 2. 环境与场景

先完成 [安装指南](setup.md)第 0–4 步，准备仿真、采集环境和 SONIC v1.1。
以下命令均在仓库根目录的 **Bash** 终端执行；GPU 0、实例 0 是示例，
启动前按 [资源规则](../AGENTS.md#shared-resources-and-artifacts)确认空闲。

```bash
export INSTANCE=0
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
SCENE_PRESET=bottle_only bash scripts/install_scene.sh
"$VENV_SIM_PY" -c 'import json; m=json.load(open("outputs/scene/sim/scene_meta.json")); print(m.get("target_pool"))'
```

目标池应包含 `water_bottle`。安装脚本解包仓库自带的场景资产、改写当前路径，
并使用仿真环境重新构建 `outputs/scene/sim`；不需要先运行 EmbodiedGen。
若提示跳过重建，先补齐仿真环境。漏掉 `SCENE_PRESET` 会构建多物体桌面。

## 3. 跑一局

这里运行 **1 次尝试**并录制；失败时保留诊断记录，runner 会因未收够 1 条成功数据退出。
它负责启动 sim / SONIC deploy / exporter，再启动专家，结束后停止本次实例。

```bash
mkdir -p /workspace/tmp/p03/outputs
"$VENV_SIM_PY" -u -m locomani.pipeline.run_learn_pilot \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json \
  --scene outputs/scene/sim --out outputs/bottle_first_run \
  --set through=collect --set successes=1 \
  --set batch_rounds=1 --set max_batches=1 --set seed0=30000 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0
```

此自动采集入口固定使用 **实例 0**，目前从 `/workspace/tmp/p03/outputs` 查找 exporter session；
示例采用默认 `LOCOMANI_TMP=/workspace/tmp`。自定义临时目录时可用文末的手动入口。
首次部署可能较久地构建 TensorRT 缓存，进度见输出目录的 `start_stack_00.log` 和栈日志。

`through=collect` 表示只采集；`learn_200.json` 本身默认运行到评测，不要遗漏这个覆盖。
每次新实验使用新的 `--out`。打开 `collect_00.log`，检查 `EPISODE OK` 或带标签的失败原因，
再结合完整视频判断实际动作。

## 4. 批量采集

目标是 **10 条成功示范，最多 3 批、每批 10 次尝试**：

```bash
"$VENV_SIM_PY" -u -m locomani.pipeline.run_learn_pilot \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json \
  --scene outputs/scene/sim --out outputs/bottle_collect_10 \
  --set through=collect --set successes=10 \
  --set batch_rounds=10 --set max_batches=3 --set seed0=30100 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0
```

`successes` 是保留条数，`batch_rounds` / `max_batches` 限制尝试预算；达到预算仍不足会明确报错。
每批重启栈，限制 exporter 的资源增长；一个 batch 对应一个 session 和一个 manifest。
扩到更多数据时调整这三个字段，保留采样分布和失败记录。

中断后的同目录重跑会读取 `collect/inputs.json` 和阶段完成标记。先核对未完成 batch 的日志、
session 和 manifest；这不是任意时刻的 episode 级恢复。更改场景或任务配置时使用新目录，
不要修改正在运行的配置快照或覆盖旧产物。

## 5. 产物与验收

以下路径相对于本次 `--out`：

| 路径 | 内容 / 查看方式 |
|---|---|
| `task_config.json`、`reach_config.json`、`scene/` | 本次任务配置与场景副本 |
| `config_snapshot.json`、`deploy_receipts.jsonl` | 配置来源、输入身份及实际部署模型 |
| `collect_00.log`、`commands.jsonl`、`timings.jsonl` | 专家日志、runner 启动命令与阶段耗时 |
| `collect/manifest_00.jsonl` | 每次尝试的目标、指令、结果及 `produced` 标志 |
| `collect/inputs.json` | session / manifest 的有序配对、保留条数与批次进度 |
| `collect/videos_00/*.mp4`、`collect/state_trace_00/` | 第三人称原片与物理状态轨迹 |
| `/workspace/tmp/p03/outputs/<时间戳>/` | exporter 原始 session；实际配对以 `inputs.json` 为准 |

### 相机与动作

默认 exporter 录 ego、左腕、右腕三路；单臂训练配置
[gr00t_sonic_2cam.py](../configs/gr00t_sonic_2cam.py)使用 ego + 右腕。
原始记录、转换后的训练字段和控制器命令不是同一种数据格式，字段说明见 [数据指南](dataset.md)。
第三人称画中画用于检查，不能替代训练相机。

验收时核对完整走近与抓取过程、真实抬升和持物结果，以及 manifest / session 的对应关系。
`--record` 才触发 episode 录制，失败尝试不作为成功示范保留；
`produced` 用于判断该行是否对应 exporter episode，不能只数 `ok`。
逐局指令通过 manifest 传给转换器，不应全部替换为 exporter 的单条 session prompt。

转换步骤、图像/动作对齐、`--trim-verification-tail` 与数据检查统一见 [数据集转换](dataset.md)；
训练和闭环结果分别见 [训练](training.md)、[评测](evaluation.md)。
分享或打包视频执行 [视频导出规则](rendering.md#video-export)：保留原片，
每条预览压到几百 KB，最多 **500,000 字节**，保持完整时长和正常速度。

## 6. 调整任务

| 想调整什么 | 修改入口 | 注意事项 |
|---|---|---|
| 出生距离、朝向、横向偏移、目标沿桌沿散布 | 任务 JSON 的 `spawn` | 当前 profile 的桌边距离为 1.1–1.3 m；不等于裸 CLI 的默认分布 |
| 目标与语言指令 | `target`、`prompt` 与场景目标池 | 指令必须对应场景中实际存在的目标 |
| 抓取策略、站位和接近行为 | `reach` | 沿用已有策略入口；改控制逻辑用 [bottle 代码导航](reference/change-map-bottle.md) |
| 桌高、物体形状和尺寸 | `scene_lib.py` 与场景构建入口 | 重建场景、使用新输出目录；配置文件不会自动改几何 |
| 外观与相机 | sim 外观设置、相机配置 | 自动 runner 固定 `SCENE_RANDOM=0`；改变记录相机时同步数据配置 |

直接调用 gym 时，已支持的选项按默认值 → 任务 JSON → 显式 CLI 合并。
先用无需模拟器的命令检查实际参数来源：

```bash
"$VENV_SIM_PY" -m locomani.tasks.bottle.cli_args \
  --task-config configs/tasks/bottle_visible.json
```

自动 runner 的 `--config` / `--set` 修改 **PilotConfig**（采集规模、阶段和运行环境），
`--task` 选择 **BottleTaskConfig**。修改任务参数应编辑单独的任务 profile，
不能把 `spawn` / `reach` 字段直接塞进 runner 的 `--set`。
字段含义和默认值以 [生成的配置表](reference/config_fields.md)为准。

## 7. 排错与停止

| 现象 | 先查哪里 |
|---|---|
| 场景里没有水瓶 / 目标池不符 | 安装场景时的 preset、重建是否完成、传入的 `--scene` |
| 启动失败或端口占用 | `start_stack_00.log`；`$LOCOMANI_TMP/p03/{sim,deploy,exporter}.log`；实例锁 |
| `[stand]` / `[view]` / `[grasp]` / `[hold]` | 分别对应站位、可见性、抓取和保持失败；检查该次视频及配置 |
| 跑完却没有成功数据 | manifest 的 `ok` / `produced`、exporter 日志、`inputs.json`；失败尝试不等于成功 episode |
| 同目录重跑被拒绝 / session 对不上 | 中断 batch 和阶段标记；保留现场，核对配对后再决定恢复方式 |

前台运行用 Ctrl-C 停止 runner。需要额外清理时先确认本次 runner / expert 已退出，
再执行 `bash scripts/stack/stop_instance.sh 0`，仅作用于本次拥有的实例。保留数据与日志。

## 8. 进一步阅读

- [手动起栈、gym 与命令入口](reference/runbook.md)：只检查动作、不录 episode，或使用自定义临时目录时阅读。
- [代码导航](reference/change-map-bottle.md)：修改采样、抓取策略或参数消费者时阅读。
- [bottle 台账](ledger/bottle.md)：按版本查实验数据、动作质量和失效记录。
- [历史指南全文](archive/collection_20260914.md)：保留旧批处理脚本、相机标定、
  控制层分析及带日期的结论；其中旧命令和成功率不作为当前使用规则。
