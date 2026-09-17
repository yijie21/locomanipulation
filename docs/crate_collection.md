# 双臂数据采集

让 G1 从桌边 1.2–1.5 m 外走近低桌，双手抓住篮沿，抬起篮子并恢复直立。
本文是双臂任务的使用入口，默认介绍**视觉指导、仿真辅助规划的抓篮流程**；
机械可达性、箱壁接触和轨迹回放作为同一任务下的细分用途。

## 1. 任务与能力

| 项目 | 当前支持 |
|---|---|
| 默认任务 | 30 cm 低桌、篮子向桌内退入 4 cm + 原始外观 + `configs/task/crate_visual_low_reach.yaml` |
| 动作过程 | 走近并准备双手 → 减速、支撑交接 → 协调俯身与伸手 → 闭合、试提 → 抬升并恢复直立 → 持物验收 |
| 专家输入 | 仿真物体位姿与机器人状态辅助规划，由采集相机的可见性和新鲜度约束动作 |
| 采集能力 | 完整尝试视频、相机观测、发送的动作参考、物理轨迹和成败记录 |
| 后续链路 | `collect.record=true` 经 exporter 录制正式 LeRobot episode（第 4 节），转换与审计已验证；闭环评测 `eval_crate_closed_loop.py` 已实现，10k 步模型 71/100（[评测](evaluation.md#双臂-closed-loopsrclocomanipipelineeval_crate_closed_looppy)） |

当前范围是已知篮子/桌面、正面无障碍接近；不保证任意深度可达或通用搬运。
仿真信息可辅助生成动作，失败接触仍须如实保留。观测、动作与训练边界的详细含义见
[视觉专家说明](crate_visual_collection.md)。

## 2. 环境与场景

先完成 [安装指南](setup.md)第 0–4 步，准备 SONIC v1.1，
并应用 [POSE ownership 补丁](../patches/sonic_pose_ownership.md)。
以下命令均在仓库根目录的 **Bash** 终端执行；GPU 0、实例 1 是示例，
启动前按 [资源规则](../AGENTS.md#shared-resources-and-artifacts)确认空闲。

```bash
export INSTANCE=1
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
bash scripts/install_scene.sh
"$VENV_SIM_PY" -m locomani.tasks.crate.bimanual_scene outputs/scene/crate_bimanual_depth04 --crate-setback 0.04
"$VENV_SIM_PY" scripts/experiments/crate_table_height_20260913.py \
  outputs/scene/crate_bimanual_depth04 outputs/scene/crate_table30_depth04 --height 0.30
```

场景资产随仓库提供；构建结果包含 `scene_locomani.xml` 和 `scene_meta.json`。
低桌配置文件**不会改变桌面几何**，因此降低桌高和选择控制 profile 是两个步骤。
`--crate-setback 0.04` 同时修改 XML 初始位置和 reset 元数据；任务 JSON 不代替这项场景构建参数。
此前本页漏写了这 4 cm，生成的场景与 6/10 实验不同；见 [复现核对](handoff/crate-reproduction-20260914.md)。
篮子外观、桌面碰撞处理与场景副本的具体约定见 [场景处理](crate_visual_collection.md#run)。

## 3. 跑一局

这里运行 **1 次尝试**，从走近开始录视频，成功局到持篮直立验收结束：

```bash
# 只跑一局：复制实验文件，把 collect 改成 batch_rounds 1 / max_batches 1 / successes 1
python scripts/collect_crate.py configs/experiments/crate_table30_record.yaml \
  --gpu 0 --name crate_first_attempt
```

采集阶段启停自己的仿真和控制实例。首次 SONIC 部署可能较久地构建 TensorRT 缓存，
进度见本次运行目录的启动日志。每次新实验使用新的 `--name`，查看 manifest 和完整视频判断结果。

`collect.record=false` 时本流程只保存视频、相机和动作记录（诊断用途，不进转换器）；
正式训练数据见第 4 节的录制方式。命令里的 `gpu=0` 是显式选择，不要照抄别处的卡号。

## 4. 批量采集

下面采集 **10 次尝试**，包含成功和失败，不是“收够 10 条成功”：

```bash
# 只跑一局：复制实验文件，把 collect 改成 batch_rounds 1 / max_batches 1 / successes 1
python scripts/collect_crate.py configs/experiments/crate_table30_record.yaml \
  --gpu 0 --name crate_collect_10
```

`collect.batch_rounds` 控制每批尝试数，`collect.max_batches` 控制批数上限；每批重启一次栈，
默认整个 visual pilot 使用一批。`collect.successes` 是要保留的成功条数。

### 录制成正式训练数据

`collect.record=true`（默认）时，采集阶段会在本实例上启动 data exporter（`RECORD_INSTANCE=1`），
专家在任务开始时发 `c`、评估后发 `c`（成功保存）或 `x`（失败丢弃），
每批一个 exporter session，写入 `$LOCOMANI_TMP/p03_i<实例>/outputs/<时间戳>/`，
并记录在 `<out>/collect/inputs.json` 的 `sessions`，与 `manifests` 同序一一对应。
exporter 约占 500 个线程，容器 pid 上限 3840，实测两个录制栈约 1800，
不要再并行第三个。实例不再被强制成 0，因此可以在 GPU 1 / 实例 1 上与
单臂采集（GPU 0 / 实例 0）并行录制；同一块 GPU 上不能跑两个仿真。

```bash
python scripts/collect_crate.py configs/experiments/crate_table30_record.yaml \
  --gpu 1 --name crate_record_10
```

manifest 行带 `task`、`produced`、`ok` 和成功局的
`execution.metrics.training_tail.end_seq`（进入直立持物验收的序号），
转换器按 [数据指南](dataset.md) 的方式裁掉验收尾段：

```bash
"$VENV_DATA_PY" -m locomani.pipeline.convert_to_hf_format \
  --sessions $(python3 -c "import json;print(' '.join(json.load(open('outputs/runs/crate_record_10/inputs.json'))['sessions']))") \
  --manifest $(python3 -c "import json;print(' '.join(json.load(open('outputs/runs/crate_record_10/inputs.json'))['manifests']))") \
  --trim-verification-tail --out outputs/runs/crate_record_10/dataset --jobs 4
"$VENV_DATA_PY" -m locomani.pipeline.check_dataset outputs/runs/crate_record_10/dataset \
  --run-dir outputs/runs/crate_record_10
```

`--run-dir` 让审计从本次运行的 `trace/` 按 exporter 的 `priv_seq` 精确取出箱子的
世界高度来判"闭合后抬升"；没有它，检查器只能用骨盆坐标系的位姿列，
而躯干直立时这一列会把 20 cm 的真实抬升读成几厘米（2026-09-14 验证批：
7/7 世界抬升 21.2–21.6 cm，骨盆系只判出 2/7）。双臂时贴近度用两腕中点。

训练用 `configs/gr00t_sonic_3cam_bimanual.py`（60 维状态、66 维动作、三路相机）；
`check_dataset` 在任务文本含 crate 时同时审计这个 profile。

**预期成功率**（2026-09-14，本页第 2 节的场景 + `crate_visual_low_reach.json` + SONIC v1.1
+ 带补丁的 deploy，GPU 1 / 实例 1 录制）：260 次尝试的正式录制前 194 次 163 成功（84%），
10 局验证批 7/10；失败以 `basket_tilt`、`invalid_initial_distance` 和接触类为主。
另一台机器数字不同时，先按第 7 节"别人的成功率与本机不同"逐项核对。
训练后的闭环评测见 [双臂 Closed-loop](evaluation.md#双臂-closed-loopsrclocomanipipelineeval_crate_closed_looppy)。
专家由 `task.expert` 选择（默认 `visual`；`wall` / `carry` 是诊断专家，
`scripts/run_wall_pilot.sh`、`run_bimanual_pilot.sh` 跑的就是它们，不代表这里的连续 visual 专家）。
同名重跑从首个未完成 batch 继续（读 `collect/inputs.json`），不是 episode 级续采；
先检查原有尝试记录和未完成 batch。
更改场景或配置时使用新目录，保留失败、零帧初始化失败及其原因，不以重跑覆盖原始样本。

## 5. 产物与验收

以下路径相对于本次运行目录 `outputs/runs/<name>/`：

| 路径 | 内容 / 查看方式 |
|---|---|
| `collect/inputs.json`、`task_config.json`、`run_config.yaml`、`crate_table30_depth04/` | 运行输入、解析后的任务配置及场景副本；副本目录沿用传入场景名 |
| `config_snapshot.json`、`deploy_receipts.jsonl` | 配置来源、场景身份与实际部署模型 |
| `videos_00/*.mp4` | 第三人称原片及机器人相机画中画 |
| `observations/<label>/*.mp4`、`observations/<label>/observations.jsonl` | 专家消费的相机帧及对应机器人状态、时间戳、可见性与阶段 |
| `observations/<label>/commands.jsonl` | 实际发送的 planner / POSE 动作参考；区别于运行目录根部的 `commands.jsonl` |
| `attempts.jsonl`、`manifest_00.jsonl`、每次尝试的 `result.json` | 全部采样尝试的真实成败、接触、抬升、倾斜与异常 |
| `trace/*_state_trace.npz` | 物理状态轨迹；用于动作核对和离线回放 |

### 相机与动作

ego、左腕和右腕用于可见性检查及动作审查。相机 MP4 按名义 25 FPS 保存，
时序测量应读 JSON 时间戳；第三人称原片使用仿真时间。
发送记录不是控制器已经执行的证明，POSE 包含未来参考帧，不能把它们直接当作当前训练 action。
字段和对齐限制见 [观测与动作记录](crate_visual_collection.md#outputs-and-limits)。

验收应核对实际初始距离、完整走近过程、双手接触、真实离桌抬升及恢复直立后的持物状态。
停止到俯身、试提到站直之间的停顿和抖动需要结合视频与物理时间测量；
manifest 的成功标签不证明动作质量或 VLA 可学性。

成功结果的 `training_tail.end_seq` 标记进入最终直立持物验证时的训练前缀边界，
仅在完整验证成功后发布；原片仍保留验证尾段，失败尝试不提供成功边界。
正式 token / history 对齐、双爪推理和学习验证仍待完成。

分享或打包执行 [视频导出规则](rendering.md#video-export)：保留原片，
每条预览最多 **500,000 字节**，保持完整时长、正常速度和比例。

### 渲染并排视频

回放需要该次采集的 trace、场景副本和 Isaac 资产，不需要重新启动 SONIC。
先按 [渲染环境与资产构建](rendering.md#资产构建)准备环境和房间/桌子源文件。
下面以第 6 节的单集 wall 诊断为例：

```bash
V="$LOCOMANI_ROOT/outputs/runs/crate_wall_example"
R="$LOCOMANI_ROOT/outputs/renders_isaac/crate_collect_10"
export LOCOMANI_SIM_DIR="$V/crate_table30_depth04"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  bash scripts/build_isaac_assets.sh --snapshot "$LOCOMANI_SIM_DIR"
mkdir -p "$R"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  MUJOCO_RENDER_ARGS="--insets" ISAAC_RENDER_ARGS="--insets" \
  bash scripts/render_wall_pilot.sh "$V" "$R" 10 visual_crate_s41 > "$R/render_all.log" 2>&1


```

结果为 `sbs_roundNN.mp4`（左 MuJoCo、右 Isaac）及 `rNN/` 中间文件。
visual / carry 回放须改成实际场景副本和 trace 标签。
两个 `*_RENDER_ARGS` 独立控制画中画；Isaac 使用第二遍重放，缺失或陈旧相机帧会报错。

缺失 trace 可能被跳过，`RENDER_ALL_DONE` 本身不证明每集均成功。
核对预计集数、完整解码与首尾画面、房间和桌子、相机视角；超时残片不能当完整视频交付。
Ghost 只表示规划目标。所有回放视频交付前同样执行上述压缩规则。

## 6. 调整任务

| 想调整什么 | 修改入口 | 注意事项 |
|---|---|---|
| 出生距离、接近与停车 | `CrateTaskConfig.visual` | 检查真实采样距离和完整行走，不仅是配置值 |
| 俯身、双手与持物站直 | `carry` / `visual` 的对应字段 | 参考 [crate 代码导航](reference/change-map-crate.md)定位控制器 |
| 桌高、篮子深度与形状 | 场景构建工具与 `--scene` | 控制 profile 不改变几何；改场景后重新验证接触与可达性 |
| 专家输入 | `visual.pose_source` | 当前默认 profile 为 `assisted`；历史 `markers` 路径需要专门的标记场景 |

配置就是那份实验文件本身：`configs/experiments/<name>.yaml` 全展开，无继承无组合，
运行开始时逐字节复制到运行目录的 `config.yaml` 并据此交付专家。
字段含义以 [配置表](reference/config_fields.md)为准；新实验使用单独的 profile 和输出目录。

### Waist and reach

低桌 visual / whole-body carry 使用腰部三轴自由度，并协调髋部、支撑和双手。
普通 carry 的自由度由 `carry.waist_dofs` 决定，wall 仍为 pitch-only。
腰关节 pitch、骨盆姿态和躯干世界倾角是不同量，配置上限不能直接当成实际弯腰角。
物体超出可达范围应失败；固定骨盆 CPU IK 不能证明真实 SONIC 跟踪成功。

### Table-height validation

第 2 节的构建命令保持桌面 XY 尺寸并降到 30 cm。改高度时修改 `--height`，
输出到新场景目录，再把 `scene=` / `scene.dir=` 指过去；同时检查控制 profile 是否适用于新几何。
55 cm、30 cm 下蹲和髋部前俯是不同实验，具体证据在文末。

### 其他双臂场景与专家

| 用途 | 入口 / 配置 | 录制边界 |
|---|---|---|
| 全身弯腰与双手机械可达性诊断 | `task=crate_low_reach task.expert=carry` | whole-body 动作 pilot，拒绝录制（`collect.record=false`） |
| 传统 planner 搬箱诊断 | `task.expert=carry`，`task.carry.reach_control=planner` | 可录 exporter session |
| 箱壁 / 箱沿接触诊断 | `task=crate_wall task.expert=wall` | 可录 session、轨迹和可选 ghost |

例如单集 wall 接触诊断使用第 2 节构建的原始高度场景：

```bash
# 只跑一局：复制实验文件，把 collect 改成 batch_rounds 1 / max_batches 1 / successes 1
python scripts/collect_crate.py configs/experiments/crate_table30_record.yaml \
  --gpu 0 --name crate_wall_example
```

`crate_wall.json` 显式开启 `wall_walk`；不加载 profile 时默认为 false。
wall / planner carry 在 `collect.record=true` 时于本实例启动 exporter，未指定 batch 大小时采用 10；
成功 episode 保存、失败丢弃，检查 `produced` 与 session 配对。
转换参考 [数据指南](dataset.md)和 [双臂数据配置](../configs/gr00t_sonic_3cam_bimanual.py)；
有 exporter 录制入口不等于已经验证 VLA 学习，也不等同于默认 visual 的完整持物站直流程。

## 7. 排错与停止

| 现象 | 先查哪里 |
|---|---|
| 低桌没有生效 | 实际 `--scene`、运行目录场景副本与桌高；仅换 JSON 不会改变桌子 |
| 刚启动就失败 / 没有视频 | `attempts.jsonl`、`result.json` 和启动日志；初始化失败可能是零帧 |
| 抓不到、倾斜或站直失败 | manifest 的阶段与失败原因、双手接触、支撑跟踪及 state trace |
| 别人的成功率与本机不同 | 先比 `task.expert`、解析后的配置、桌高与篮子 inset、`deploy_receipts.jsonl` 的模型/二进制身份；再比失败类型，不能只比 git SHA |
| 录制被拒绝 | 是否选了 visual 或 whole-body carry；当前属于能力限制 |
| Isaac 缺少桌子或背景 | 本次场景副本与 [资产构建](rendering.md#资产构建)，重新构建对应缓存 |

用 Ctrl-C 停止前台阶段进程；阶段自己会停掉它起的栈。需要额外清理时，先确认本次
阶段 / expert 已退出，再执行 `python -m locomani.stack.cli down --instance 1`（示例实例）；
只清理自己拥有的实例并保留产物。

### 中途停止渲染 / 清理

```bash
bash scripts/stack/stop_render.sh crate_review_example
```

参数按输出目录筛选，须唯一匹配本人的受管渲染任务。
启动器自动生成内部 job ID，不接受用户指定 `--job-id`；
仅停止归属核验通过的进程，目录仍有 worker 时拒绝重用。
独立启动的未受管渲染器不在该命令范围内。

## 8. 进一步阅读

- [运行基础设施](infra.md)：配置组怎么组合、阶段产物与断点、多卡并行的作业文件与资源准入。
- [视觉专家说明](crate_visual_collection.md)：信息来源、相机可见性、连续动作与记录字段。
- [crate 代码导航](reference/change-map-crate.md)：修改双手、腰部、交接或失败判据时阅读。
- [连续动作版本及验证](handoff/crate-continuous-motion-20260914.md)：当前实现的固定批次证据和剩余限制。
- [55 cm](handoff/crate-table55-20260913.md)、[30 cm 下蹲](handoff/crate-table30-20260913.md)、
  [髋部前俯](handoff/crate-low-reach-20260913.md)、[持物站直](handoff/crate-upright-handoff-20260913.md)：
  按对应配置追溯历史实验。
- [crate 台账](ledger/crate.md)、[历史指南全文](archive/crate_collection_20260914.md)：
  保留早期设计、诊断流程及原始结论，不作为当前运行规则。
