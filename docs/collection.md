# 单臂数据采集

让 G1 走近桌子，用右手抓起水瓶并保持。本文是 SONIC 单臂任务的使用入口；
抓起后放入篮子见 [pick-and-place](pick_place_collection.md)，双手抓篮子见 [双臂采集](crate_collection.md)。

## 1. 任务与能力

| 项目 | 当前支持 |
|---|---|
| 默认任务 | `bottle_only` 场景（硬垫）+ `configs/tasks/bottle_whole_body.json`；`bottle_visible.json` 是 2026-09-14 前的基线入口 |
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

目标池应包含 `water_bottle`，且脚本最后一行回显 `finger pads HARDENED`。安装脚本解包
仓库自带的场景资产、改写当前路径，并使用仿真环境重新构建 `outputs/scene/sim`；
不需要先运行 EmbodiedGen。若提示跳过重建，先补齐仿真环境。漏掉 `SCENE_PRESET`
会构建多物体桌面。指垫硬化是 2026-09-14 起的默认（见 [安装指南](setup.md#4-装场景必需)），
本页所有成功率都在硬垫场景上测得；固定 case bank 用
`outputs/bottle_continuous_20260914/cases_fixed10_hardpads.json`（同样 10 个 case，
按硬垫场景重新盖章；旧文件只匹配软垫场景）。

## 3. 跑一局

这里运行 **1 次尝试**并录制；失败时保留诊断记录，runner 会因未收够 1 条成功数据退出。
它负责启动 sim / SONIC deploy / exporter，再启动专家，结束后停止本次实例。

```bash
mkdir -p /workspace/tmp/p03/outputs
"$VENV_SIM_PY" -u -m locomani.pipeline.run_learn_pilot \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_whole_body.json \
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
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_whole_body.json \
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

### 统一入口与配置组（2026-09-16）

所有阶段只有一个命令：`python -m locomani.cli <collect|convert|train|eval|stats|show> [覆盖项]`（`make run ARGS=...` 同义）。一次运行由七个配置组组成，每组一个 YAML 目录、一个 dataclass 做 schema（YAML 里写了 dataclass 没有的字段会在组合时报错，`__post_init__` 的校验照常触发）：`task/`（专家决策、验收、失败分类）、`embodiment/`（动作/状态维度、相机、GR00T modality；`g1_single_arm`、`g1_bimanual`）、`scene/`（冻结场景目录 + 固定样本库）、`backend/`、`collect/`、`train/`、`eval/`。`configs/config.yaml` 是默认组合；`configs/experiment/<name>.yaml` 固定一个版本，用 `+experiment=<name>` 选中，只增不改。例子：

```
python -m locomani.cli show    +experiment=bottle_simple_lowcom            # 只打印组合结果
python -m locomani.cli collect +experiment=bottle_simple_lowcom gpu=1      # 冻结专家的 8 局固定库
python -m locomani.cli collect task=pick_place scene=sim collect.successes=50
python -m locomani.cli collect +experiment=crate_table30_record            # 双臂 crate 录制线
```

Task 声明它能跑的 embodiment（`conf.TASK_EMBODIMENTS`），`task=crate_table30 embodiment=g1_single_arm` 在组合时就报错。每次运行把完整解析后的配置写到 `<out>/run_config.yaml|json`，这就是这次运行的 provenance。旧的 JSON 配置文件和 `--set` 覆盖仍被底层驱动读取：入口把组合结果落成 `task_config.json`/`pilot_config.json` 再调用它们（基础设施计划第 3 步把阶段拆成独立进程后删除）。

### 复现要点与预期成功率

另一台机器要得到与本页相同的数字，下面每一项都必须一致；任何一项不同都先比它，再谈策略：

| 项目 | 要求 | 核对方法 |
|---|---|---|
| SONIC 权重 | `policy/sonic_v1_1`（`scripts/paths.sh` 默认） | 运行目录 `deploy_receipts.jsonl` |
| deploy 二进制 | 按 [安装指南第 3 步](setup.md#3-编译-c-deploy必需) 编译，含 POSE ownership 与夹爪交接补丁 | 同上的 capability 标记 |
| 场景 | `SCENE_PRESET=bottle_only bash scripts/install_scene.sh`，回显 `finger pads HARDENED` | `outputs/scene/sim/scene_meta.json`、脚本回显 |
| 任务配置 | `configs/tasks/bottle_whole_body.json`，不改字段（2026-09-15 起 `visible_brake_depth` 为 0.31，此前 0.40；见下表最后一行） | 运行目录 `task_config.json`、`config_snapshot.json` |
| pilot 配置 | `configs/pilot/learn_200.json` + 本页的 `--set` | 运行目录 `inputs.json` |
| 资源 | 该 GPU 上没有第二个仿真；CPU 配额有余量 | `nvidia-smi`、`/sys/fs/cgroup/cpu.stat` |
| 闭环评测的初始 token | `--initial-token-file configs/deployment/whole_body_initial_token.json`（与这份数据的垂臂开局一致；默认的 READY token 会让 100 局全部卡在 prelude） | 评测目录 `config_snapshot.json` 的 `initial_token` 摘要 |

2026-09-14 在上述配置下的实测（同一天、同一份代码，失败按"阶段 × 连杆"分类）：

| 运行 | 结果 | 剩余失败 |
|---|---|---|
| 采样 20 局，`seed0=30000`（`outputs/bottle_hardpads_sampled20_20260914`） | 15/20 | 髋/桌接管判定 3（停得 ≤0.44 m 时）、抬升中手指/桌 1、无双侧接触 1；无滑脱 |
| 同 20 局，`visible_brake_depth` 0.40→0.31（`outputs/bottle_brake031_sampled20_20260915`，2026-09-15） | 17/20 | 髋/桌接管判定 2、腕/瓶接管判定 1；park 0.355 m（0.34–0.39），park 到全身阶段开始的骨盆前移由 10.6 cm 降到 3.4 cm |
| 固定 10 case，`seed0=31000`（`outputs/bottle_hardpads_fixed10_20260914`，软垫场景同代码为 6/10） | 7/10 | 抬升中手指/桌 1、髋/桌接管 1、无双侧接触 1；无滑脱 |

10 局批次之间本身有 ±2 局的波动（同 seed 步态不可复现），比较时至少用 20 局并按失败族对照，
不要只看总数。证据与历次改动见 [开局与抬升交接](handoff/bottle-hanging-start-20260914.md)。

### 固定采样比较新策略

`bottle_whole_body.json` 是本页的采集入口（连续全身策略，2026-09-14 起）；
`bottle_visible.json` 是此前的基线入口，只用于与旧数据对照。
使用前安装 [POSE 与夹爪交接补丁](setup.md)，确认所选 deploy 包含两个 capability 标记。
当前实现与实验结果见 [行走准备和连续抓取](handoff/bottle-direct-20260914.md)，
上一版基线见 [固定样本验证记录](handoff/bottle-continuous-20260914.md)，
不能把“能运行候选策略”理解为已提高成功率或已验证 VLA 学习效果。

当前默认 profile 已启用 `reach.whole_body.natural_walk_prep` 和
`wrist_roi_supervisor`。自然准备同时选择停车位置和预抓姿态，以舒适关节先验、
实际网格扫掠检查及腕相机投影筛选候选；无可行候选时保持并低频重试。
停车距离包含速度相关的制动提前量，伸手目标还受实时瓶子距离约束。
ego 尺寸用连续图像确认和滞回触发抬手，腕相机稳定获取目标后停止搜索抬升，
以 2 Hz、带死区和幅度限制的画面修正维持取景。规划之间仍连续发送关节参考。
POSE 接管继承当前参考速度，并让完整未来窗口的位置与速度使用同一条有界轨迹。
开始夹紧时固定实测手部世界位置目标，保留平移跟踪补偿及其变化速度，继续限制补偿的速度和加速度；
腕部旋转补偿和舒适姿态先验保持不变，关节参考继续受速度、加速度约束；
恢复直立后先检查持瓶制动路径，再连续收住参考速度，
保持右爪闭合并交回行走控制。日志分别记录指令连续性和实测运动。
实现、逐版失败原因和新采集结果见
[自然准备、腕相机取景与连续参考](handoff/bottle-natural-expert-20260915.md)。

全身抓取的目标深度现为标定工具点前方 1.2 mm（共享指垫区间的中点），合爪窗口为该点 ±5 mm，
使瓶子进入指垫中部；规划与实测准入使用同一深度定义，保留腕、掌和桌面的碰撞检查。
伸手目标根据实测前进速度连续后退，为剩余制动行程留出空间；实际中央几何和预测停下时的深度都满足窗口后才闭合，静稳窗口仅作诊断。
跟踪补偿限制变化速度和加速度，
保留接管时的解码偏差；全身 IK 使用舒适姿态先验和速度、加速度硬约束。
当前配置关闭速度连续代价，避免其在停止伸手时继续偏好原有前进速度。
停车先发制动命令，再做耗时手臂规划，并在完成支撑制动后检查站位；
当前制动提前量上限仍为 0.10 m，实际制动距离的散布需要继续校准。
抓起和恢复直立的进度受双侧接触与掌心坐标系内的滑动速度调节，短暂失去接触时
减速并保持闭爪，稳定恢复后再继续；超时仍报告失败。具体改动、物理验证及限制见
[中部抓取、接触恢复与运动跟踪](handoff/bottle-contact-expert-20260915.md)。 后续按响应辨识、停车与全身规划、带物轨迹与稳定交接的顺序推进，见 [当前实施记录](handoff/bottle-dynamics-expert-20260915.md)。
这些改动尚不代表已验证 VLA 学习收益。

共同动作原则与此前诊断记录如下；固定髋旁手目标和固定停车距离的记录属于旧版：

- 走近时由 ego / 右腕相机的实际可见性准入仿真目标点，结合本体状态补偿身体移动。
  视觉观测在独立线程处理，先滤掉步态摆动，再以 5 Hz 更新手臂规划；关节指令连续发送，
  并限制速度和加速度。迟到的图像结果不会延长目标缓存的有效期。
- 行走阶段做保守的方向、伸手距离准备；腿部仍由 SONIC 控制，腰部平滑回到行走中性参考。
  相机短暂遮挡只沿用最后可见目标，超时不继续追踪隐藏的物体位置。
- 减速后延续同一瓶子目标与抓取方向，接入双脚支撑的全身求解，协调腰、髋和手臂。
  最终伸手不再强制经过“高处保持 → 下降 → 前插”三个路点；实际碰撞、接触与承载验收仍保留。
- 接管从当前参考开始，不瞬间跳到 READY；抬起和身体恢复继续重叠进行。
- transport guard 的预测碰撞门限（`transport_margin`）在三处含义不同：接管时允许手指
  贴着瓶子（准备好的手本来就该在那里）；伸手阶段接近后允许手指接触；闭合并持续接触
  后进入抬升，手里的瓶子对右臂不再算障碍，桌子和身体的检查不变。2026-09-14 之前
  抬升中腕部连杆碰到手里的瓶子会中止回合（81 局里 21 局，瓶子已离桌中位 8 cm），
  接管时手指贴瓶又中止 12 局。抬升高度、稳持、滑移验收没有放宽。
- 全身抓取失败后先钉住一个零速度的停止参考，再把同一姿势交还 planner 并退出 POSE 流。
  以前失败路径不退出流模式，下一局 reset 后机器人落点偏 0.10–0.12 m，
  并一直重复到批次结束（2026-09-14 当天 8 个 stand 阶段失败全部紧跟在一次全身失败之后）。
  连续专家在 engage 后只把腰部准备到中性参考，不再先送双臂 READY 基线：录制从 SONIC
  自己的垂臂参考开始，右臂在走近中按速度 / 加速度限制逐渐抬起（与双臂采集的开局一致）。
  此前的顺序会在 0.5 s 双臂抬起、1.5 s 又落回，再于走路时重新抬起。腰部准备不能省：
  engage 姿势前倾约 8°，不回中性时出生视野门限会拒绝大部分回合；而且要等实测腰角
  收敛到 0.03 rad 内再放行，手臂那档 0.30 rad 的容差会在腰还没动时就放行。

- 旧版走路阶段的手目标（2026-09-15，自然准备接入前，照双臂 `ApproachPreparation`）：从第一次规划起腕就转向入口
  姿态（手指朝前），工具点保持在抓点上方 `walk_clearance`（0.06 m）、髋旁 x 0.08（骨盆坐标系常量，
  对应双臂的 `stand_off − walk_brake_margin`：手不前伸，瓶子走过来），横向跟随瓶子，且始终留在瓶子
  后方至少 12 cm（基座会越过 0.31 m 刹车点停在约 0.2 m）；腰在双臂线的行走包络内（yaw/roll ±0.1、
  pitch 0–0.12 rad）参与求解，走路限速 `walk_joint_speed` 0.7 rad/s。这样掌心基本保持垂臂高度，
  停下后只剩约 15 cm 滑入和 5 cm 下降。同 20 seed 14/20（基线 17/20），多出的 3 局是刹停时髋/桌接管判定，
  来自走路 IK 把腰 roll 打满 +0.1。
- 2026-09-15 起的"胶囊相切"方案：走路包络去掉腰 roll；抬手由 ego 视野里瓶子的像素高度触发
  （`raise_trigger_px` 36 px ≈ 1.5 m，触发前双臂保持原样）；刹车改为"骨盆离桌沿 `visible_brake_edge`
  0.27 m 或物体深度 `visible_brake_depth` 0.31 m，先到者停"（瓶子在桌沿内 4–8 cm，所以通常是桌沿先到，
  停下时瓶子离骨盆 0.31–0.35 m；直立站在 ≤0.23 m 时右髋连杆会贴到桌面，0.27 m 无碰）；停下后的
  全身伸手里躯干前倾目标取"高度项"和"距离项（骨盆到抓点 0.30→0.60 m 线性到 `lean_max`）"中较大者，
  `bottle_whole_body.json` 把 `lean_max` 设为 1.0 rad，远瓶自动深弯；伸手终点仍由 `entry_step`
  停在抓点后 1.8 cm。之前的目标是骨盆坐标系里的固定盒子
  （z ≥ 骨盆 +0.05，掌心 0.84 m），那是清障高度：dex1 手指从掌心向下伸 19 cm，垂臂指尖 0.53 m
  低于 0.62 m 桌面。第一次尝试（抓点上方 4 cm、不用腰、0.45 rad/s）2/20，见
  [handoff](handoff/bottle-hanging-start-20260914.md#seventh-step-the-crate-style-walking-preparation-does-not-fit-this-table-2026-09-15)。
- `reach.whole_body.walk_plan_hz` 控制规划频率，`walk_filter_seconds` 控制滤波时间常数，
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
  -m locomani.tasks.bottle.grasp_gym --task-config configs/tasks/bottle_whole_body.json \
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
  --task-config configs/tasks/bottle_whole_body.json
```

自动 runner 的 `--config` / `--set` 修改 **PilotConfig**（采集规模、阶段和运行环境），
`--task` 选择 **BottleTaskConfig**。修改任务参数应编辑单独的任务 profile，
不能把 `spawn` / `reach` 字段直接塞进 runner 的 `--set`。
字段含义和默认值以 [生成的配置表](reference/config_fields.md)为准。

### 连续全身候选的动作原则

`reach.strategy=whole_body` 沿用相同走近和采样入口，在走近时逐渐准备手臂，减速后让
腰、髋、手共用带速度/加速度约束的支撑参考。三个腰部自由度均可参与。
接近可抓区域时开始闭合，持续双侧接触后抬升；物体离桌后，抬升和恢复直立重叠进行，
手相对躯干的位置保持不变，不再向身体回收（此前 6 cm 的回收加上躯干直立会把肘折到
胸前的别扭角度）。抬升高度由 `reach.whole_body.lift_height`（默认 0.06 m）、时长由
`lift_seconds`（默认 5 s）给出，躯干直立本身再额外抬高约 0.2 m。此前 0.16 m / 3 s
的组合让瓶子以约 13 cm/s 上升 35–41 cm，2026-09-14 的 30 局里有 4 局从爪中滑出。
回到走路控制前预发保持夹爪和上肢的 planner 命令，再退离桌边并验收稳持。

仿真位姿辅助几何规划，但目标须通过真实 ego/右腕图像的新鲜度和近似可见性检查，
短时遮挡只沿用上次可见目标；不会用隐藏的新位置无限追踪。可见性由相机几何和 ray
近似判定，尚不是 RGB 位姿估计器。碰撞、物体被推走、失去接触或身体不稳定仍判失败。
支持约束和连续参考不能保证 SONIC 的实际动作同样平滑，应同时检查参考与执行轨迹。

运动参数集中在 `reach.whole_body`，通过类型化配置传到消费者；不增加散落的环境变量。
抓取后的退离与稳持验收沿用原标准。训练转换只截去已标记的末端验收尾段，不删除
任务中间的停顿来掩盖控制问题。参数和证据分别见配置表与验证记录。

### 简化单臂专家（`reach.strategy=simple`，2026-09-16）

和双臂线同形：回合开始站立时双臂以 1.0 rad/s 收到配置的下垂姿态（`prepare_walk`，实测到 0.15 rad 内即走，上限 3 s；走路期间不再有任何手臂动作）（`reach.simple.hang_*`：肘 1.2、肩滚外展 0.45、腕关节全零，左臂镜像），手腕不做任何翻转，整臂比规划器的下垂姿态离身体远 8 cm（骨盆外 30 cm），腕相机 80° 视锥内到自身几何 ≥ 35 cm（离线射线扫描；外展 0.25 打到大腿 12 cm；前臂前伸的姿态会改变步态，停位冲到 0.08–0.10 m）；不跟踪瓶子。步态一结束（骨盆 < 0.10 m/s、腿 < 0.50 rad/s，不等静止）就接管 POSE，`stats.stand` 记录站姿；接管时两只手显式张开。之后是一条连续的最小加加速度路径：手在桌面高度带且离桌沿
不足 6 cm 时先后缩 10 cm，竖直抬到 max(抓取高度 +8 cm, 桌面 +12 cm) 并前移 10 cm，按
停稳后的位置重算接近线，滑到预抓取点；然后一段 8 cm/s 的进入，横向按实测掌坐标偏移
在线修正（3 cm/s、最多 4 cm），到位或双侧接触即闭爪 → 抬升 → 躯干回正，交接给规划器。
手指方向不沿骨盆到瓶子的方位，而是随瓶子向右的偏移向内转（1.2 rad/m，上限 20°）：
离线 IK 在 8 个实测站位上扫描，手指朝前到向内 20° 的关节余量最大（65–70°），沿方位
向外 30–40° 只有 42–55°，向内 60° 只剩 17–27°。前倾只按前向深度算（0.30 m 内为 0）。
站位由刹车时的步速决定，不由刹车点决定：规划器停步会随步态相位冲出 0.02–0.26 m，
而 v1.1 的 shuffle 指令（0.20）低于它的速度阶跃（0.28–0.30），减速带原本不减速。
`reach.visible_shuffle_speed` 0.32 配 0.60 m 减速带让最后半米慢速爬行，刹车沿 0.40
（`visible_brake_depth` 0.44）使停稳站位落在离桌沿 0.20–0.31 m。下垂的 Dex1 手几何
最前端在骨盆前 0.15–0.18 m、高 0.55–0.72 m，正是桌面高度带，站位 < 0.22 m 时手在
抬起前就贴着桌面（离线用 `TransportGuard` 在状态轨迹上复现，`stage4_hang8`…`8e`）。
只保留实测硬门：碰撞扫描、摔倒、瓶子被撞离 5 cm、闭爪后双侧指垫接触、抬升后瓶子
上升 ≥4 cm，以及各相位超时；没有终端观测器、异步站姿规划、静止窗和进度调速。
字段在 `reach.simple.*`（`docs/reference/config_fields.md`）。配套场景
`outputs/scene/sim_lowcom` 由 `build_sim_scene.py --object-com-drop 0.05` 生成，把
瓶子重心下移 5 cm，使单边指垫先碰到时瓶子倾向滑动而不是倾倒。

```bash
"$VENV_SIM_PY" -u -m locomani.pipeline.run_learn_pilot --config configs/pilot/learn_200.json \
  --task configs/tasks/bottle_simple.json --scene outputs/scene/sim_lowcom --out outputs/<run> \
  --set through=collect --set successes=8 --set batch_rounds=8 --set max_batches=1 --set seed0=30700 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0
```

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
