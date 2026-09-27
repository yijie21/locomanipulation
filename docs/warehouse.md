# 物流房间抓纸箱（warehouse）：从采集到评测

G1 在物流房间里走到托盘上的纸箱前，双手扣住箱沿，抬起、站直、后退 0.3 m。**这一页从头到尾能跑完采集、转换、训练和评测**，不需要跳到别的文档。和 [双臂](bimanual.md) 抓篮是**同一条管线**（同一个 `tasks/crate` 专家、同一套扁平脚本），差别只有三处：场景、**仿真时不渲染**、数据集相机在转换前从轨迹补渲。结构与 [单臂](single_arm.md) / [双臂](bimanual.md) 逐节对应。为什么这么定、试过什么被否掉了，在 [DECISIONS](../DECISIONS.md) 2026-09-22 起的各节。

## 1. 准备一次

先做完 [安装指南](setup.md) 第 0–4 步。第 3 步一次打全的三个 deploy 补丁里，这条线要的是 [POSE 控制权补丁](../patches/sonic_pose_ownership.md)（双臂必需，采集入口会检查二进制，没打直接报错）和 [夹爪交接补丁](../patches/sonic_hand_ownership.md)（下面的基线是在打了它的 deploy 上测的），不用再单独打。然后建场景：

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
"$VENV_SIM_PY" scripts/prepare_logistics_scene.py outputs/scene/warehouse_textured
```

资产包 `assets/scene/logistics_room_box_v1/` 随仓库提供（`git lfs pull` 过就有）；生成的场景在 `outputs/` 下，不进 Git。场景名 `warehouse_textured` 指向这个目录（`configs/scenes.yaml`）。旧名 `warehouse` 是贴图提交之前装的，纸箱没贴图，只留给旧运行。

> **本仓库原作者的机器**上打过这两个补丁的 deploy 在别处，用 `export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"` 指过去；新机器按上面装好后**不要**这一行。

`eval.image_source: isaac` 的运行（训练图由 Isaac 补渲、闭环由 Isaac 镜像出图，§5–§6）**必须**另做安装指南第 7 步和 `python3 scripts/build_isaac_warehouse_scene.py`（要 `pxr`，`pip install usd-core`），并有第二张卡给 Isaac；`image_source: mujoco` 的运行只有 §5 最后一条的并排预览要它。

## 2. 一次运行 = 一份实验文件

现行基线是 `configs/experiments/box_room_grasp_baseline_20260924.yaml`：20 局（seed 7000–7019）**6/20**。成功 = 抬起、躯干直立、后退 0.3 m 箱子不掉。实验文件是**完整**的，这次运行读的每一个值都在里面，没有继承。**基线本身不改**：新实验复制它，改 `name` 和**一个**旋钮。这条线同一份代码 10 局之间就能差出 2/10 到 7/10，所以对照一律 20 局，同一组 seed。

## 3. 新建一次实验

```bash
cp configs/experiments/box_room_grasp_baseline_20260924.yaml configs/experiments/my_box.yaml
sed -i 's/^name: .*/name: my_box/' configs/experiments/my_box.yaml
$EDITOR configs/experiments/my_box.yaml     # 改 §4 表里那几个字段，或一个专家旋钮
diff configs/experiments/box_room_grasp_baseline_20260924.yaml configs/experiments/my_box.yaml
```

## 4. 每次都要改的字段

| 字段 | 含义 | 注意 |
|---|---|---|
| `name` | 运行目录 `outputs/runs/<name>/` | 同名运行的实验文件被改过会拒绝启动 |
| `backend.render` | 仿真时渲不渲相机 | **目前用 `offline`**。旧 `warehouse` 场景的纸箱是一整块 46.3 万面网格，超过 MuJoCo 单块网格的 2²⁰ 顶点门槛，实时渲染时 sim 只剩 0.35× 实时，机器人走得比指令快约 3 倍；2026-09-24 按实时渲染跑是 0/20，离线回到 0.95×。现行 `warehouse_textured` 把同样的面拆成 9 块，三路相机离线测 3.2 ms，但实时栈的 RTF 还没测（TODO 4.7.7） |
| `collect.record` | 录不录 LeRobot 数据 | 基线是 `false`（只采 manifest 和轨迹，调参数用）。要出训练数据改成 `true`，并跑 §5 的渲相机一步 |
| `collect.batch_rounds` / `max_batches` | 每批尝试几局 / 最多几批 | 批内一定跑满；要正好跑 N 局就 `batch_rounds: N`、`max_batches: 1`（和 [双臂 §5](bimanual.md) 同一套规则），录制时 `train.holdout` 也要小于 N |
| `collect.successes` | 保留多少条成功 | 批间的验收门槛；不够时采集非零退出，但记录照样保留 |
| `collect.seed0` | 第一局的 seed | 对照要同一个 |
| `train.holdout` | 留几集做验证 | 必须**小于**能保留的集数，否则训练集为空。录制时 `collect.successes` 和 `batch_rounds × max_batches` 都是上限，起栈时就查；实际成功得更少时 convert 仍会拒绝（基线约 6/20，默认 10 就不够） |

专家旋钮（`task.visual.*`）的含义见 [配置字段表](reference/config_fields.md)，历次改动的效果见 [DECISIONS](../DECISIONS.md)。

## 5. 跑

```bash
. scripts/paths.sh
# 采集两个终端：起栈 / 跑专家（每个脚本都吃 --dry-run）；专家跑完自己收栈
bash scripts/start_stack.sh collect       configs/experiments/my_box.yaml --gpu 0
bash scripts/bimanual/run_agent_server.sh configs/experiments/my_box.yaml --gpu 0
# 录像：仿真时没渲，事后从 trace 渲出（第三视角自动画 ghost 夹爪），写 <run>/videos/
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/my_box
# 只在 collect.record: true 时：先补渲数据集相机，再转换
bash scripts/render_dataset_cameras.sh outputs/runs/my_box --gpu 0
#   eval.image_source: isaac 的运行改用 Isaac 补渲（上一行不跑）；可在采集时另一张卡上反复跑，只动已跑完的批
bash scripts/render_dataset_cameras_isaac.sh outputs/runs/my_box --gpu 1
bash scripts/convert_data.sh           outputs/runs/my_box
# 可选：Isaac 并排预览，写 <run>/previews_isaac/
bash scripts/render_isaac_warehouse.sh outputs/runs/my_box
```

**补渲数据集相机在做什么**：离线时 sim 不出图，exporter 先存黑色占位帧，其余列（状态、动作、物体位姿）都是实录的。`render_dataset_cameras.sh` 按每行记下的 sim seq（session 的 `priv_seq/`）从 `trace/` 取出那一刻的 qpos，用和 sim 同一份场景、同一组相机（ego + 两腕）渲出图，覆盖占位帧；每局 6–10 s。对照实时录制的局核过：画面一致，实时帧比渲的晚 1–2 行（实时相机 30 Hz、exporter 取最近一张）。`render_dataset_cameras_isaac.sh` 做同一件事，图来自闭环镜像用的同一个 `IsaacView`（同一 stage、同一曝光），训练图和评测图是同一个函数。为省时间只渲 convert 会读的：成功局（失败局留占位帧），且只渲 50→25 Hz 取到的行，中间行重复前一行（早 20 ms）；每局约 90 s（三路 p50 76–84 ms，GPU1）。convert 按 `eval.image_source` 核对 session 是哪个渲染器补的，不一致就拒绝。整条线无人值守：`bash scripts/run_overnight.sh <实验文件> --gpu 0 --isaac-gpu 1 --line bimanual`（Isaac 补渲和镜像在 `--isaac-gpu`）。

一局连重置约 45 s，20 局约 15 分钟。训练和评测见 §6。

## 6. 训练与评测

训练与双臂相同（§6.2 第一条命令），跑完在 `summary.json` 写 train 条目（指最后一步的 checkpoint）。

### 6.1 起评测前逐条核对

评测**只读运行目录里的 `config.yaml` 副本**，不读 `configs/experiments/` 的原件；已经开跑的运行改下表字段就改 `<run>/config.yaml`，用 `run_overnight.sh` 时原件里的 `eval.image_source` 也要一致（它按原件决定起不起镜像）。每局的起点、夹爪合爪下限（`task.visual.close_target`）和"后退"一列都取自 `task.visual`，与采集同源，见 [双臂详解 §五](bimanual_design.md#五双臂闭环评测器)。

| 项 | 要求 | 不对会怎样 |
|---|---|---|
| `summary.json` 的 train 条目 | `train.sh` 跑完自动写。训练中途停了、或要评别的 checkpoint，按下面的命令手补（整条覆盖） | `summary.json has no checkpoint; run the train stage first` |
| `eval.initial_token` | `configs/deployment/warehouse_initial_token.json`。warehouse 与 `box_platform` 两条线的数据开局手臂都与它差 ≤ 0.03 rad；从旧实验文件抄来的 `crate_initial_token.json` 差 0.6 rad | **不报错**：开局核对只拿 token 文件自己的 `arms` 比，策略从训练里没见过的姿势起步 |
| `eval.image_source` 与 `eval.backend.render` | `isaac` 配 `offline`，外加镜像（§6.2）；`mujoco` 配 `live`（sim 自己出图）。训练图的渲染器必须同源，convert 已核对 | `mujoco` 配 `offline`：sim 不发图，每局 `[vla] camera/state streams never became ready` |
| `<run>/eval/` | 没有上一次的 `closed_loop.manifest.jsonl` / `closed_loop.json` | 前者 `... exists; pick another --out`；后者让 `run_overnight.sh` 直接跳过评测。重评先 `mv outputs/runs/my_box/eval outputs/runs/my_box/eval_<标签>` |
| `eval.rounds` | 先 10 局冒烟看录像，再 100 局；每 `collect.batch_rounds` 局重启一次 deploy | — |

```bash
PYTHONPATH=src python3 -c "from pathlib import Path; from locomani.pipeline.run_context import summarise; r = Path('outputs/runs/my_box').resolve(); summarise(r, 'train', checkpoint=str(r / 'model/checkpoint-5000'), steps=5000)"   # 手补 train 条目（例：停在 5000 步）
```

### 6.2 跑

```bash
. scripts/paths.sh
bash scripts/train.sh outputs/runs/my_box --gpu 0
bash scripts/open_loop_eval.sh --model-path outputs/runs/my_box/model/checkpoint-5000 --dataset-path outputs/runs/my_box/val_dataset --out-dir outputs/runs/my_box/eval/open_loop   # 开环，不起栈，10 集约 7 分钟
# 闭环，一条命令：栈 + 策略服务在 --gpu，Isaac 镜像在 --isaac-gpu（image_source mujoco 不要 --isaac-gpu）
bash scripts/run_overnight.sh configs/experiments/my_box.yaml --gpu 0 --isaac-gpu 1 --line bimanual --from eval
bash scripts/start_stack.sh eval           outputs/runs/my_box --gpu 0   # 或者分终端：栈 / 策略服务 / Isaac 镜像（只 isaac，等它打印 listening）/ rounds
bash scripts/run_vla_server.sh             outputs/runs/my_box --gpu 0
bash scripts/run_isaac_mirror.sh           outputs/runs/my_box --gpu 1
bash scripts/bimanual/run_vla_inference.sh outputs/runs/my_box --gpu 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/my_box --scene outputs/runs/my_box/eval/warehouse_textured --traces outputs/runs/my_box/eval/state_trace --out outputs/runs/my_box/eval/videos   # 录像：评测时 sim 不录，从每局 qpos 轨迹事后渲
```

### 6.3 读结果

- 过程：`eval/rounds_00.log` 每局一行 `[crate-vla] round N: OK/FAIL <原因>`；`eval/serve.log` 有 `listening` 才算策略服务起来了；`eval/isaac_mirror.log`。Isaac 镜像不等物理：策略要观测时渲 sim 此刻状态的三路相机，与训练图同一个 `IsaacView`（[机制](rendering.md#vla-闭环的-isaac-画面按需镜像)）。
- `eval/closed_loop.json`：`successes` = 抬起 ≥ 0.6 × `lift_height`、倾角 < `max_tilt_deg`、两手接触，连续 2 s；失败按 `[vla] walk / reach / close` 分阶段。`retreat_distance` > 0 时成功局接着跑 `retreat_timeout` 秒，`retreated` / `retreat_reasons` 记是否带箱后退了这么远。
  **convert 在 hold 开始处裁掉了训练集的验收尾段，后退不在数据里**，所以这一列目前必然是 `timeout`，读作"成功后又持住这么久没掉"；`dropped` / `fallen` 才是掉了。
- 逐局：`eval/closed_loop.manifest.jsonl`（起点 `placement`、`rise`、`tilt_deg`、`min_dist`）；`eval/traces/round_NNN.json` 每个动作块一行（夹爪指令、延迟、镜像图的 `seq`）。
- 第一个参照：`box_room_grasp_200_cap120_20260925` checkpoint-5000，开环 MSE 0.00123，闭环 9/10，延迟与镜像耗时见 [台账](ledger/crate.md) 2026-09-27 末节。

## 7. 产物

路径相对于 `outputs/runs/<name>/`，布局与双臂相同（采集产物在顶层）：

| 路径 | 内容 |
|---|---|
| `config.yaml`、`summary.json` | 实验文件副本；每阶段一条记录（保留数、产出率、失败家族） |
| `manifest_NN.jsonl` | 逐局成败 `ok` 与原因 `why`，每批一个文件 |
| `trace/<局>_state_trace.npz` | 每个控制帧的 qpos 和 seq：事后录像、补渲相机、审计世界高度都从它来 |
| `trace/<局>_ghost.npz` | 命令的抓取目标，只有进入伸手阶段的局才有 |
| `observations/<局>/observations.jsonl` | 专家逐 tick 记录（伸手误差、跨壁误差、底盘位置） |
| `videos/`、`previews_isaac/` | 事后渲的录像、Isaac 并排预览 |
| `collect/inputs.json` | exporter session 与 manifest 的对应；session 里 `meta/offline_render.json` 记哪些局补渲过 |
| `train_dataset/`、`val_dataset/`、`audit_*.log`、`model/checkpoint-<步>/` | 转换后的 LeRobot 数据集和审计；权重 |
| `eval/closed_loop*`、`eval/state_trace/`、`eval/traces/`、`eval/isaac_mirror.jsonl` | 闭环汇总与逐局记录（§6.3）、每局 qpos 轨迹（渲录像用）、每个动作块的策略记录、镜像的每次渲染 |

## 8. 排错

| 症状 | 原因与处理 |
|---|---|
| 机器人走得太快、过冲撞箱、成批 0 分 | `backend.render` 不是 `offline`。算实时比：trace 行数 × 0.02 s ÷ manifest 的 `seconds`，低于 0.9 就是渲染拖慢 |
| convert 报 `still hold black placeholder frames` | 这是离线录制的运行，先跑 `render_dataset_cameras.sh`（`eval.image_source: isaac` 的跑 `_isaac.sh`） |
| convert 报 `rendered by another renderer` | session 补渲用的渲染器和 `eval.image_source` 不一致；训练图必须和评测图同源，换 `name` 重采 |
| 起栈报 `the training set would be empty`，或 convert 报 `leaves none of N kept episodes` | `train.holdout` 不小于能保留（起栈时查上限）或实际保留的局数。调小实验文件里的 `train.holdout`（0 = 不留验证集）：起栈时被拒的同名重起即可；convert 才拒的，运行开始时配置已冻结，只能换 `name` 重采 |
| 一批里大量 `initial_stance_unstable` | 停栈后不到 20 s 就重起，deploy 收不到 LowState；等 20 s、确认没有残留 sim/deploy 进程再起 |
| 起栈报缺 POSE 补丁 | §1 的 deploy 没打补丁，或 `SONIC_DEPLOY_ROOT` 指错 |
| 评测每局 `[vla] camera/state streams never became ready` | `eval.image_source: mujoco` 却 `eval.backend.render: offline`（sim 不发图）；或 `isaac` 但镜像没起。改 `<run>/config.yaml` 后重起栈 |
| 评测 `[vla] isaac mirror: no reply ...` | 镜像没起或挂了，看 `eval/isaac_mirror.log`；`run_overnight.sh` 要给 `--isaac-gpu` |
| 评测报 `has no checkpoint`、`... exists; pick another --out` | §6.1 表的前两行与第四行 |
| `reach_stalled`、`lost_attachment` 占大头 | 这条线现在的主要失败形态（手停在箱沿上方、合爪后脱开），不是环境问题；进展见 [TODO](../TODO.md) 2.4.10.29 |

## 9. 进一步阅读

- [Blender 场景](blender_scenes.md)：房间和纸箱资产怎么从 Blender 出到 MuJoCo / Isaac、尺寸、已知问题。
- [离线渲染](rendering.md#离线渲染backendrender-offline)：`backend.render offline` 的实现和限制。
- [双臂](bimanual.md) / [双臂详解](bimanual_design.md)：同一条管线的完整说明和专家设计。
