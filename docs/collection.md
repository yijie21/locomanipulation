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

### 固定采样比较新策略

`bottle_whole_body.json` 是显式选择的连续全身候选策略，默认仍为 `bottle_visible.json`。
使用前安装 [POSE 与夹爪交接补丁](setup.md)，确认所选 deploy 包含两个 capability 标记。
当前实现与实验结果见 [行走准备和连续抓取](handoff/bottle-direct-20260914.md)，
上一版基线见 [固定样本验证记录](handoff/bottle-continuous-20260914.md)，
不能把“能运行候选策略”理解为已提高成功率或已验证 VLA 学习效果。

候选策略的动作设计是：

- 走近时由 ego / 右腕相机的实际可见性准入仿真目标点，结合本体状态补偿身体移动。
  视觉观测在独立线程处理，先滤掉步态摆动，再以 5 Hz 更新手臂规划；关节指令连续发送，
  并限制速度和加速度。迟到的图像结果不会延长目标缓存的有效期。
- 行走阶段做保守的方向、伸手距离准备；腿部仍由 SONIC 控制，腰部平滑回到行走中性参考。
  相机短暂遮挡只沿用最后可见目标，超时不继续追踪隐藏的物体位置。
- 减速后延续同一瓶子目标与抓取方向，接入双脚支撑的全身求解，协调腰、髋和手臂。
  最终伸手不再强制经过“高处保持 → 下降 → 前插”三个路点；实际碰撞、接触与承载验收仍保留。
- 接管从当前参考开始，不瞬间跳到 READY；抬起和身体恢复继续重叠进行。

`reach.whole_body.walk_plan_hz` 控制规划频率，`walk_filter_seconds` 控制滤波时间常数，
默认 5 Hz / 0.45 s。配置定义在 [whole_body_config.py](../src/locomani/tasks/bottle/whole_body_config.py)，
行走引导在 [walking_reach.py](../src/locomani/tasks/bottle/walking_reach.py)，
全身抓取在 [whole_body_policy.py](../src/locomani/tasks/bottle/whole_body_policy.py)。
本次固定 10 次尝试为 6/10 成功（上一版 3/10），刹车到闭合的成功样本中位数从 9.84 s 降到 6.68 s。
但实测关节步进 p99 和接管后的抬手幅度未改善，因此它仍是候选策略，不能等同于全面更平滑。
原始动作标签检查 10/10 通过；转换后的 6 集训练格式检查为 30/32 通过，尚未通过全部验收。
世界坐标轨迹复核确认 6 集都真实抬升，但图像与状态的 1–2 帧对齐疑点仍待解决，不能直接视为训练就绪。
完整指标、检查器的坐标系问题与全部失败记录见上述验证文档。

这是使用仿真几何辅助的专家，并非从 RGB 直接估计抓取位姿；数据标签是否正确与 VLA 是否学会，
需要分别检验。动作更直接不能代替真实接触成功或训练评测。

先冻结同一组目标位置、站位、出生扰动、外观和物体 yaw：

```bash
LOCOMANI_SIM_DIR="$LOCOMANI_ROOT/outputs/scene/sim" "$VENV_SIM_PY" \
  -m locomani.tasks.bottle.grasp_gym --task-config configs/tasks/bottle_visible.json \
  --targets water_bottle --rounds 10 --seed 31000 \
  --sample-only --write-cases outputs/bottle_cases10.json
```

然后对同一 case bank 分别运行基线和候选；保留全部十次尝试，不用成功后重采样填满：

```bash
for variant in visible whole_body; do
  "$VENV_SIM_PY" -u -m locomani.pipeline.run_learn_pilot \
    --config configs/pilot/learn_200.json --task "configs/tasks/bottle_${variant}.json" \
    --scene outputs/scene/sim --out "outputs/bottle_fixed_${variant}" \
    --cases outputs/bottle_cases10.json \
    --set through=collect --set successes=10 --set batch_rounds=10 \
    --set max_batches=1 --set seed0=31000 \
    --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0
done
```

不足 10 条成功时 runner 返回非零退出码，但失败记录和视频仍保留。
case bank 绑定场景 metadata hash，并须覆盖整个尝试预算；换场景须生成新 bank。
固定的是采样条件，异步控制、接触和渲染时序仍可能使重复实验产生不同结果。

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

### 连续全身候选的动作原则

`reach.strategy=whole_body` 沿用相同走近和采样入口，在走近时逐渐准备手臂，减速后让
腰、髋、手共用带速度/加速度约束的支撑参考。三个腰部自由度均可参与。
接近可抓区域时开始闭合，持续双侧接触后抬升；物体离桌后，抬升、收手和恢复直立
重叠进行。回到走路控制前预发保持夹爪和上肢的 planner 命令，再退离桌边并验收稳持。

仿真位姿辅助几何规划，但目标须通过真实 ego/右腕图像的新鲜度和近似可见性检查，
短时遮挡只沿用上次可见目标；不会用隐藏的新位置无限追踪。可见性由相机几何和 ray
近似判定，尚不是 RGB 位姿估计器。碰撞、物体被推走、失去接触或身体不稳定仍判失败。
支持约束和连续参考不能保证 SONIC 的实际动作同样平滑，应同时检查参考与执行轨迹。

运动参数集中在 `reach.whole_body`，通过类型化配置传到消费者；不增加散落的环境变量。
抓取后的退离与稳持验收沿用原标准。训练转换只截去已标记的末端验收尾段，不删除
任务中间的停顿来掩盖控制问题。参数和证据分别见配置表与验证记录。

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
