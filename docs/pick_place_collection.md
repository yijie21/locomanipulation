# Pick-and-place 数据采集

让 G1 走近桌子，用右手抓起罐头，再放进旁边的篮子。本文是该任务的使用入口；
只抓起物体见 [单臂采集](collection.md)，双手抬篮子见 [双臂采集](crate_collection.md)。

## 1. 任务与能力

| 项目 | 当前支持 |
|---|---|
| 默认任务 | `pick_place` 场景 + `configs/task/pick_place.yaml`；目标 `soup_can`、容器 `basket` |
| 动作过程 | 随机出生 → 走近、抓起 → 抬高避开篮沿 → 转向放置区域 → 移到篮上、下降松手 → 退手与验收 |
| 专家输入 | 复用单臂专家；仿真罐头/篮子位姿、机器人状态和相机可见性辅助规划 |
| 采集能力 | 单局检查、按成功数批量采集、exporter session、manifest、视频与状态轨迹 |
| 后续链路 | 复用单臂数据接口；该任务的 LeRobot 转换、训练及闭环学习效果尚未验证 |

当前是固定篮子、右手放置的实验任务，没有开放任意容器、左右手选择或通用搬运。
放置阶段仍分开执行移步与手臂伺服，不能据此承诺全过程无停顿。
旧 10 成功 pilot 的测量保留在 [pick-and-place 台账](ledger/pick_place.md)，不代表当前成功率。

## 2. 环境与场景

先完成 [安装指南](setup.md)第 0–4 步。篮子和罐头源资产位于仓库的 `assets/simple/`，
使用该场景不需要额外安装 SIMPLE；Isaac 重渲是单独的可选环境。
以下命令均在仓库根目录的 **Bash** 终端执行，示例 GPU 0、实例 0
须按 [资源规则](../AGENTS.md#shared-resources-and-artifacts)确认空闲。

```bash
export INSTANCE=0
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
bash scripts/install_scene.sh
"$VENV_SIM_PY" -m locomani.sim.build_sim_scene \
  --preset pick_place --harden-pads outputs/scene outputs/scene/pick_place
"$VENV_SIM_PY" -c 'import json; m=json.load(open("outputs/scene/pick_place/scene_meta.json")); print(m.get("target_object"), m.get("goal_object"))'
```

元数据应显示 `soup_can` 和 `basket`。预设使用约 62 cm 桌面、6 × 14 cm 罐头和
30 × 20 × 15 cm 空心篮子；具体几何以生成的场景为准。
`--harden-pads` 使用共享的 `object` 指面接触配置，是**构建时**开关；
启动 runner 不会补上遗漏的接触设置。

## 3. 跑一局

这里运行 **1 次完整抓取与放置尝试**并录制。失败会留下诊断记录，
runner 因未收够 1 条成功数据退出；只有抓起罐头不算任务完成。

```bash
mkdir -p $LOCOMANI_TMP/p03/outputs
export PYTHONPATH=src
"$VENV_SIM_PY" -m locomani.cli collect task=pick_place \
  scene.dir=outputs/scene/pick_place collect.successes=1 \
  collect.batch_rounds=1 collect.max_batches=1 collect.seed0=32000 \
  gpu=0
```

runner 负责启动 sim / SONIC deploy / exporter 和专家，并在结束后停止本次实例。
它固定使用 **实例 0**，目前从 `$LOCOMANI_TMP/p03/outputs` 查找 session；
上述命令使用默认临时目录。自定义目录或只检查动作时见 [手动命令入口](reference/runbook.md#manual-bottle-or-pick-and-place-check)。

`through=collect` 必须保留，否则 pilot 配置默认继续到转换、训练和评测。
检查 `collect_00.log` 中的抬升、`[place]` 阶段和最终 `EPISODE OK` / 失败标签，
再看视频确认罐头确实释放在篮内。首次部署的缓存构建进度见 `start_stack_00.log` 及栈日志。

## 4. 批量采集

目标是 **10 条成功示范，最多 3 批、每批 10 次尝试**：

```bash
export PYTHONPATH=src
"$VENV_SIM_PY" -m locomani.cli collect task=pick_place \
  scene.dir=outputs/scene/pick_place collect.successes=10 \
  collect.batch_rounds=10 collect.max_batches=3 collect.seed0=32300 \
  gpu=0 name=pick_place_collect_10
```

`successes` 是保留条数；每批重启栈并产生独立 session / manifest，
达到尝试预算仍未收够会报错。失败原因应区分抓取侧和放置侧，不能只统计已经抓起的子集。

同目录重跑使用 `collect/inputs.json` 和阶段标记恢复。先检查中断 batch 的配对，
不要假定支持 episode 级断点续采。改任务或场景时换新目录；任务、reach 和场景副本在首次运行时保存，
不要修改正在运行的 `task_config.json`，否则快照与同批实际行为会不一致。

## 5. 产物与验收

以下路径相对于本次 `--out`：

| 路径 | 内容 / 查看方式 |
|---|---|
| `task_config.json`、`reach_config.json`、`scene/` | 本次任务、抓取配置及场景副本 |
| `config_snapshot.json`、`deploy_receipts.jsonl` | 配置、输入身份与部署模型 |
| `collect_00.log`、`commands.jsonl`、`timings.jsonl` | 专家日志、runner 启动命令与阶段耗时 |
| `collect/manifest_00.jsonl` | 每次尝试的指令、结果、`produced`；放置指标和轨迹在 `execution.metrics.place` / `execution.metrics.place_trace` |
| `collect/inputs.json` | session / manifest 的有序配对及采集进度 |
| `collect/videos_00/*.mp4`、`collect/state_trace_00/` | 完整过程原片及物理轨迹 |
| `/workspace/tmp/p03/outputs/<时间戳>/` | exporter 原始 session；配对以 `inputs.json` 为准 |

### 相机与动作

exporter 默认记录 ego、左腕、右腕；该任务复用单臂 schema，
[训练配置](../configs/gr00t_sonic_2cam.py)选 ego + 右腕和右手动作。
任务指令由 profile 的 `prompt` 写入 manifest，包含抓取和放置两个目标。
第三人称画中画不能替代训练相机；字段、转换与对齐见 [数据指南](dataset.md)。

验收需要罐头真实落在篮内且低于篮沿、手已离开，篮子移动和退手后的扰动在配置允许范围内。
当前判据不要求罐头最终竖直。检查以下内容：

- 完整走近、抓取、转向、放置和退手过程；不要在刚抓起或刚松手时截断视频。
- manifest 的 `ok` / `produced` 与 session 配对；失败尝试保留诊断，不作为成功示范。
- `execution.metrics.place` 的释放误差和最终偏移，以及 `execution.metrics.place_trace` 的掌系漂移。
- 篮子在决策与放置过程中的可见性；记录中可见不等于已经证明 VLA 可学。

成功结果中的 `training_tail.end_seq` 为后续裁去验证尾段提供边界。
使用 [转换与检查流程](dataset.md)验证后才能称为训练数据；本任务尚未完成这一验证。
分享视频执行 [视频导出规则](rendering.md#video-export)：保留原片，预览每条最多
**500,000 字节**，保留完整时长和正常速度。Isaac 下篮子贴图仍需单独检查。

## 6. 调整任务

| 想调整什么 | 修改入口 | 注意事项 |
|---|---|---|
| 出生与罐头沿桌沿随机化 | 任务 JSON 的 `spawn` | 当前距离为桌边 1.1–1.3 m，罐头采样还受 `place.goal_gap` 约束 |
| 抓取高度和策略 | `reach` | 更换罐头后检查抓取模板覆盖、碰撞和实际持物稳定性 |
| 放置区域与运动速度 | `place.band_x/band_y`、`v_travel/v_descend` | 区域在骨盆坐标系；同时检查脚步能否到位和持物 IK 可达性 |
| 下降与松手 | `release_clearance`、`place_tol/place_tol_late`、相关 timeout | 深度相对篮沿；放宽容差不能代替篮内落物检查 |
| 容器和任务指令 | `goal`、`prompt`、场景几何及 `goal_inner_half` | 当前篮子固定；更换尺寸或位置需同步验收范围与可见性 |

字段含义和默认值见 [配置表](reference/config_fields.md)。
runner 的 `--config` / `--set` 只调整 PilotConfig；任务参数用单独的 `--task` profile。
无需启动仿真即可查看 gym 对该 profile 的解析：

```bash
"$VENV_SIM_PY" -m locomani.tasks.bottle.cli_args \
  --task-config configs/task/pick_place.yaml
```

改变物体或容器几何后，先检查抓取模板、持物放置带 IK、出生/抓取/放置时的相机视野，
再跑实际单局。旧测试坐标和成功率不能代替新场景的验证。
当前控制流程及参数消费者位于 [place_phase.py](../src/locomani/tasks/bottle/place_phase.py)，
改代码从 [任务导航](reference/change-map-bottle.md)进入。

## 7. 排错与停止

| 现象 | 先查哪里 |
|---|---|
| 目标名不符或罐头出生时碰篮子 | `scene_meta.json`、目标间距与 `place.goal_gap` |
| 抓取模板覆盖失败 | 罐头 grip 高度、`reach` 配置；更换物体后重新检查几何 |
| `[carry]` / `[turn]` | 持物漂移、转向超时、篮心是否进入骨盆坐标系的放置带 |
| `[place]` / `[retreat]` | 释放误差、篮内判据、手臂退出后的碰撞或扫动 |
| 启动失败 / 没有 session | `start_stack_00.log`、`$LOCOMANI_TMP/p03/{sim,deploy,exporter}.log`、实例锁与路径 |

前台运行用 Ctrl-C 停止 runner。需要额外清理时，确认本次 runner / expert 已退出，
再执行 `python -m locomani.stack.cli down --instance 0`，只停止自己拥有的实例；保留结果和日志。

## 8. 进一步阅读

- [手动命令与通用入口](reference/runbook.md)：检查动作或更换运行路径时阅读。
- [任务代码导航](reference/change-map-bottle.md)：定位复用的单臂实现和 place 参数消费者。
- [原始设计](handoff/pick_place_plan.md)、[实验台账](ledger/pick_place.md)：追溯几何选择与对应版本的测试。
- [重组前指南](archive/pick_place_collection_20260914.md)：保留旧操作、实验数字和已知问题的原文；
  不作为当前运行步骤。
