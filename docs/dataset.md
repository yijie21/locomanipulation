# 数据集:导出与审计

> exporter 原始 session → LeRobot v2.1,数据 schema,以及转完必跑的四段审计。

## 导出 LeRobot v2.1 并对接 GR00T

exporter 的原始 session 在 `$LOCOMANI_TMP/p03/outputs/`（实例 0）或
`$LOCOMANI_TMP/p03_i<N>/outputs/`（`--record` 在实例 N 上录制，见
[双臂采集](crate_collection.md#录制成正式训练数据)）。转成 LeRobot v2.1
(格式对齐 `hojjunekim/g1_sonic_orange_to_plate_locomani`,含 29-DoF 关节状态 +
64 维 SONIC motion latent + 三路 25fps 视频):

```bash
# 正常路径：转换是一个阶段，session 与 manifest 的配对从 collect 的 manifest 里读
python scripts/convert.py outputs/runs/<run>   # 产出 <run>/train_dataset 与 <run>/val_dataset
```

`train.holdout` 决定切出多少条做验证集（0 表示不切）。手工转换散落的 session
（不属于某次运行、或要自定义裁剪选项）时直接调底层转换器：

```bash
third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
    src/locomani/pipeline/convert_to_hf_format.py \
    --sessions "$LOCOMANI_TMP"/p03/outputs/<session1> <session2> ... \
    --manifest outputs/logs/collect/manifest_1.jsonl \
                outputs/logs/collect/manifest_2.jsonl ... \
    --out datasets/g1_locomani_lerobot
```

**`--sessions` 和 `--manifest` 必须一一对应、同序**:第 k 个 session 是第 k 个
manifest 那一批产生的。`collect` 命令的批次是按 seed 递增跑的,session 目录名是时间戳,
所以两边都按各自的自然顺序排就对上了(seed 超过 9 时记得用 `sort -V`)。

### 只去掉末端验收尾段

新单臂采集的成功 manifest 提供 `execution.metrics.training_tail.end_seq`，
表示退离桌边结束、进入最终持物验收的序号。转换时在上述命令添加
`--trim-verification-tail`，通过 session 的逐行 sequence sidecar 保留到这个边界的
连续前缀。原始数据和视频仍保留完整验收，验收失败的局不会进入成功数据集。
缺失边界或序号不能猜测裁剪；此选项要求 manifest，且不能同时使用 `--drop-idle`
或 `--include-discarded`。任务中间的停顿应修控制逻辑，不能靠删帧掩盖。

对错了不会静默 —— 转换器在解码任何一帧之前**硬校验**四件事,不符直接报错退出:

- manifest 里 `produced` 的行数 == session 里的 episode 数
- 每一行的 `ok` == 该 episode 在 disk 上是否被 discarded
- 没有 0 字节的 episode parquet(exporter 中途死掉会留下没 flush 的空文件)
- 每个 episode 的录制密度 ≥ `MIN_RECORDING_DENSITY`(0.60 行/序号单位)

最后一条挡的是 2026-09-17 发现的静默损坏:健康 session 每个 episode 都在
1.10–1.15 行/序号单位,但机器被其他负载占用时 exporter 会安静地只留下零头 ——
同一天三个 session 出现 0.04–0.44 的 episode,其中一局 37 秒的成功回合只录到
1.2 秒画面。危险的不是报错,而是这种残片只要训练边界恰好落在它确实录到的区间内
就能通过其余所有检查,以 22 帧"示范"的身份进入数据集;当天五个残片里只有一个
响了,而且纯属运气。判定在 `recording_density.py`,拒绝时会打印实测数字;
遇到就重采,不要放宽阈值。

> **`--manifest` 名义上可选,实际上必给。**数据集里的语言标注(`task_index` →
> `meta/tasks.jsonl`)**只**来自 manifest 的逐局 `task` 字段。省掉 `--manifest`
> 不会报错,而是让每一条 episode 都退回转换器里那个硬编码常量
> `TASK = "pick up the red can"`(`convert_to_hf_format.py:43`)—— 于是一整个
> 水瓶数据集会被标成"拿起红罐子",画面和语言对不上,而且这在训练日志里完全看不出来。
>
> 注意 exporter 的 `--task-prompt`(`locomani.stack.cli up` 的那个字符串、
> `collect` 命令的 `task` 参数)**不参与**这条链路:它对整个 session 是固定的,
> 只影响录像叠字和 session 元数据。它写错不会污染数据集,但会让你在排查时看错地方
> —— 所以还是让它和场景里的目标物体保持一致。

得到的数据集可直接喂给 `third_party/Isaac-GR00T` 做 GR00T 微调。上层/底层的详细
数据流见 `docs/architecture.drawio`。

### 转完先查一遍(强烈建议)


加 `--run-dir <运行目录>`：抬升检查改用状态 trace 里的世界坐标高度，不再受骨盆坐标系
位姿列的影响（[world_lift.py](../src/locomani/pipeline/world_lift.py)）。两种目录布局都支持
——双臂线的逐局 `label`，以及单臂线没有 label 时按 `seq` 覆盖区间把 episode 和 trace 对上。
`convert` 阶段默认就带上它：不带时机器人站起来的那部分抬升会被骨盆系抵消，2026-09-17 的
200 集数据里有 15/190 被误判为"没抬起来"（实测世界抬升中位 15 cm、最小 6 cm）。

```bash
third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
    src/locomani/pipeline/check_dataset.py datasets/g1_locomani_lerobot
```

**通过全部检查才去烧 GPU。** 有问题就退出码 1,能直接挂在 CI 或训练前的 gate 上。
四段检查,由便宜到贵:

| 段 | 回答什么 | 内容 |
|---|---|---|
| **structure** | 文件之间自洽吗 | 计数三方一致、`episode_index` 连续、parquet 行数 == 声明长度、`index`/`frame_index`/`timestamp`/`task_index` 自洽、列宽 == `info.json` 声明、无 NaN/Inf、抽检视频帧数 == parquet 行数 |
| **modality** | GR00T 认得出来吗 | 同时检查 stock（state 46 / action 78）与本仓库单臂 2cam（state 53 / action 65）所选字段和切片 |
| **grasp** | 数字描述的是一次抓取吗 | 每局有闭合事件、闭合时腕已在物体 30 cm 内、闭合**之后**物体才升、末段物↔腕距离恒定 |
| **alignment** | 第 t 帧和第 t 行是同一时刻吗 | 见下 |

常用开关:`--quick`(只跑前两段,不解码视频,几秒)、`--videos N`、`--align-episodes N`。

**为什么 alignment 要单独测。** 时序错位是这类数据最贵的 bug,而且**看不出来** ——
拖着视频对曲线,差一秒以内怎么看都像对的。所以这里是**量**出来的:腕相机硬连在手臂上,
所以逐帧图像变化量是手臂速度的代理,而手臂速度在表里也有
(`|d(observation.state)/dt|`)。两者互相关,峰值所在的 lag 就是错位帧数。

对齐检查报告图像变化与关节速度的相关峰值，以及 token 与身体响应的相对 lag。
这些是诊断指标，不能证明每张图的曝光时间与每条 action 精确对应。按 25 fps 换算，
3 帧是 120 ms；旧 30 fps 实验中的 3 帧才是 100 ms。重复采样、周期动作和低运动量
都会影响相关峰值，不能仅凭 lag 正负推断因果关系或宣称 VLA 可学。

**一个反直觉的坑,写在这里免得再踩:** `observation.can_pose` / `plate_pose` 是
**9 维(xyz + rot6d)且在机器人坐标系下**,不是世界坐标。所以"物体相对起始高度上升了吗"
这种判据会在**机器人下蹲**时触发(骨盆降低 → 物体相对高度跳变),在完全正常的数据上报
0/200。判据必须锚在**夹爪闭合那一帧**上。

锚的是**这一局收尾时的那次闭合**,不是第一次闭合:接管瞬间 deploy 会短暂播一下默认片段,
夹爪闭合约 20 帧又张开(2026-09-17 的 200 集里 43/190 有这种毛刺,帧号 74–136)。锚在
第一次闭合上会把毛刺当成抓取,判出"闭合时腕离物体 32 cm"。审计会把这些局逐一列出来,
它同时是训练数据的一个已知瑕疵:约四分之一的示范带一段"离瓶子很远却发出闭合"的监督信号。

### VLA 部署时这条链路怎么变

本仓库单臂模型输出 **65 维**：64 维 token + 1 维右夹爪；部署端恢复两组 7 维夹爪
报文后，通过 protocol v4 发送，**不经过 encoder**。双臂 profile 输出 66 维。

```bash
python -m locomani.stack.cli up --instance 0 --gpu 0 --prompt "pick up the water bottle" --no-encoder   # 关键:必须不加载 encoder
```

`NO_ENCODER=1` 不是省显存。deploy 只要加载了 encoder,`initial_encoder_mode_` 就是 0,
它的 token 路径会不停地翻回 encoder("if encoder mode is not -1, we need to turn on
the encoder",`g1_deploy_onnx_ref.cpp:3096`),外部 token 流永远压不住。不给
`--encoder-file`,mode 变成 -1,才会走 "tokens can be set externally"。

当前数据应分别验证字段选择、实际消费命令、转换对齐和 live token 回放。
历史回放指标不能替代新策略的检查；从不同初始物理状态开环发送一段 token，
只能检查动作通道，不能当作整集任务复现或 VLA 闭环成功证据。

---

## LeRobot 数据集结构

`convert_to_hf_format.py` 吐出的是标准 LeRobot v2.1 目录布局,三块内容各管各的,
靠帧号隐式对齐,不存交叉引用:

```
<dataset_root>/
├── meta/                                          # 元信息,JSON/JSONL,不含大数据
│   ├── info.json                                  # 全局配置 + 完整 feature schema
│   ├── tasks.jsonl                                # task_index → 任务文本
│   ├── episodes.jsonl                             # 每个 episode 的 task_index 和帧数
│   ├── episodes_stats.jsonl                       # 每个 episode、每个字段的统计量
│   ├── modality.json / modality_gr00t_sonic.json  # GR00T/sonic 用的字段命名映射(同内容)
│   └── modality_psi0.json                         # psi0 用的字段命名映射
├── data/
│   └── chunk-000/
│       └── episode_{000000..NNNNNN}.parquet       # 每帧数值数据,不含图像
└── videos/
    └── chunk-000/
        ├── observation.images.ego_view/episode_*.mp4
        ├── observation.images.left_wrist/episode_*.mp4
        └── observation.images.right_wrist/episode_*.mp4
```

(`chunk-000` 是分片目录,一个 chunk 装 `info.json.chunks_size` 条 episode;单个
session 批量不大的话通常只有这一个 chunk。)

### meta/info.json

数据集级配置:`total_episodes`/`total_frames`/`fps`/`splits`、`data_path`/
`video_path` 命名模板,以及 `features` —— 每个字段的 `dtype`/`shape`/关节名。
`script_config` 额外记了转换脚本和原始 session 路径,出问题先看这里对得上哪次导出。

### meta/tasks.jsonl

`{"task_index": N, "task": "..."}`。语言标注只有这一条链路,来源和坑见上文
『导出 LeRobot v2.1 并对接 GR00T』里 `--manifest` 那段。

### meta/episodes.jsonl

每个 episode 一行:`{"episode_index", "tasks", "length"}`。不用打开 parquet
就能拿到 episode 边界,structure 审计的行数校验就是拿这份和 parquet 实际行数对。

### meta/episodes_stats.jsonl

每个 episode 一行,对 `info.json` 里除视频外的**每个字段**都存了逐维
`min`/`max`/`mean`/`std`/`count`,训练前的归一化统计从这里出,不用重新扫 parquet。

### meta/modality*.json

不存数据,是 dataloader 用的**切片表**:`{组名: {start, end, original_key}}`,
把 parquet 里零散的列按名字拼成策略要的 `state`/`action`/`video` 命名空间。
`modality.json` 和 `modality_gr00t_sonic.json` 内容相同(GR00T/sonic 用),
`modality_psi0.json` 是 psi0 简化动作空间用的。哪个策略实际吃多少维、为什么
这么切,见下文「数据 schema」。

> **⚠️ `action.psi0_18` 在 2026-09-12 之前的每一个数据集里都是全零**(ledger bottle.md §35.8)。
> converter 无条件用 `teleop.vr_3pt_position/orientation` 算它(`convert_to_hf_format.compute_psi0_columns`),
> 而 exporter 只在 `stream_mode == 5` 才写那两列 —— 单臂 coupled 线录制窗口里 `stream_mode` 一直是 3,
> 于是那两列恒为零,psi0 这套 modality **是死数据**。抽查过 2026-09-11 的 session:864 帧只有一种取值。
> 只有走 `teleop_arm` 通道(`collection.md` 的通道对照表)时它才有真值,而且**只在抓取段非零**
> (run5 的 episode 0:1089 帧里 621 帧),走路与抱稳段仍是零 —— 要用它训练,**先处理这个不连续**。

### data/chunk-000/episode_*.parquet

一个 episode 一个文件,**一行一帧**。列很多,按用途分四组:

| 分组 | 代表字段(维度) | 说明 |
|---|---|---|
| 机器人本体状态 | `observation.state`(43)、`observation.eef_state`(14)、`observation.root_orientation`(4)、`observation.projected_gravity`(3)、`observation.cpp_rotation_offset`(4)、`observation.init_base_quat`(4) | 关节角、末端位姿、朝向/重力 |
| 动作/标签 | `action.wbc`(43)、`action.motion_token`(64)、`observation.state_psi0`(15)、`action.psi0_18`(18) | WBC 关节目标、SONIC latent、psi0 坐标系简化 state/action |
| 遥操作/动捕溯源(`teleop.*`) | `smpl_joints`/`smpl_pose`/`smpl_frame_index`、`body_quat_w`/`target_body_orientation`、`left`/`right_hand_joints`、`left`/`right_wrist_joints`、`vr_3pt_position`/`vr_3pt_orientation`、`delta_heading`、`stream_mode`/`planner_mode`、`planner_movement`/`facing`/`speed`/`height` | SMPL 动捕原始量、VR 三点追踪、locomotion planner 指令;大部分不进训练,留着做分析/辅助监督。**`vr_3pt_*` 只在 `stream_mode == 5` 有值**:coupled 通道恒为零,`teleop_arm` 通道在抓取段是真实的腕/颈命令位姿(见上方 ⚠️) |
| 索引/标注 | `timestamp`、`frame_index`、`episode_index`、`index`(全局帧号)、`task_index` | 定位用,不是物理量 |

物体位姿(`observation.can_pose` / `plate_pose` / `water_bottle_pose`,按 session
里的物体命名;9 维 = pos3 + rot6d6,机器人骨盆系)只在该 session 跑了物体位姿
跟踪时才存在,是特权量,不进 modality——用它写判据时的坑见上文「一个反直觉的坑」。

`observation.images.*` 在 `info.json` 里声明成 `dtype: video`,但**不出现在
parquet 列里**——像素在 mp4 里,靠帧号对齐,parquet 不存引用。

### videos/chunk-000/observation.images.{cam}/episode_*.mp4

三路 RGB(`ego_view` 头显视角、`left_wrist`/`right_wrist` 手腕相机),和
`info.json.fps` 一致(转换器统一重采样到 `DST_FPS`,**2026-08-28 起是 25fps**,
理由见 [ROADMAP](ROADMAP.md#已做dst_fps--25消掉重采样的-10-hz-伪影2026-08-28);
更早转出来的数据集是 30fps)。每个视频的帧数必须等于对应
parquet 的行数——这正是 structure 审计里"抽检视频帧数 == parquet 行数"要测的。

---

## 数据 schema

2026-09-14 按当前 schema 与模型配置修正本节。[修订前原文](https://github.com/yijie21/locomanipulation/blob/e302d11/docs/dataset.md)
保留了旧 78/46 维说明、量化试验和延迟解释；这些历史说法不作为当前接口或动态行为结论。

原始列、模型选择的字段和部署报文分开定义。唯一代码来源是
[schema.py](../src/locomani/schema.py)；模型使用的字段由对应配置选择。

| 用途 | state | action | 相机 |
|---|---|---|---|
| 本仓库单臂 `gr00t_sonic_2cam.py` | 53 = 43 关节 + 3 重力 + 7 右腕位姿 | 65 = 64 token + 1 右夹爪 | ego、右腕 |
| 本仓库双臂 `gr00t_sonic_3cam_bimanual.py` | 60 = 单臂 state + 7 左腕位姿 | 66 = 64 token + 左右夹爪各 1 | ego、左右腕 |
| 上游 stock `unitree_g1_sonic` | 46 = 43 关节 + 3 重力 | 78 = 64 token + 两组 7 维手部字段 | 由 stock 配置选择 |

单臂右夹爪两个滑轨共用一个标量，左夹爪保持张开。
[sonic_token.py](../src/locomani/backends/sonic/sonic_token.py)负责 65 维 action 与
protocol v4 的转换。若实际采集的左手也变化，或右夹爪两分量不相等，65 维 profile
就无法无损表达这次动作，必须拒绝该数据或显式修改模型接口。

`motion_token` 在 1/16 网格上，范围为 [−1.0, 0.9375]。token 的语义取决于
encoder/decoder checkpoint；检查 `deploy_receipts.jsonl` 的模型和观测配置 hash。
相同网格不能说明旧 release 数据能交给 v1.1 decoder。量化也不意味着细微动作
优化没有价值：需要测最终 token、夹爪命令和实际运动，不能由单关节的静态试验
推导整个动态策略的平滑性上限。

43 维关节列按以下顺序保存，两组手部字段分别紧跟各自手臂：

| 切片 | 0:6 | 6:12 | 12:15 | 15:22 | 22:29 | 29:36 | 36:43 |
|---|---|---|---|---|---|---|---|
| 内容 | 左腿 | 右腿 | 腰 | 左臂 | 左手 | 右臂 | 右手 |

`meta/modality.json` 是可选择字段的映射，不能把里面所有条目相加当成模型维度。
仿真物体位姿可以用于专家规划和离线验收，但不进入这些 VLA 输入字段。

### 夹爪标签必须来自实际消费命令

默认启动脚本现在通过仓库的 `run_instance_exporter.py` 启动 exporter。
它将 `teleop.left_hand_joints` / `teleop.right_hand_joints` 写成同一份 C++ state
报文中的 `last_left_hand_action` / `last_right_hand_action`，与记录的 token 共用来源。
不要绕过此 wrapper 直接启动上游 exporter：joint-only POSE 消息可能没有被 SMPL
订阅者解码，旧 fallback 会把真正闭合的夹爪标成张开。

自动采集的 `collect/frame_audit_*.jsonl` 记录逐行消费命令、状态序号和相机发布时刻。
对原始采集运行执行：

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  -m locomani.pipeline.audit_bottle_actions outputs/bottle_run \
  --out outputs/bottle_run/actions_audit.json
```

检查包括：65 维动作能否恢复原报文、标签是否等于逐行消费命令、重复 token、
关节变化与停顿诊断。旧数据缺少消费命令 receipt 时标为 **未验证**，不能仅凭维度
正确就判 PASS。该审计不替代真实持物验收、转换后的 `check_dataset` 或 VLA 训练评测。
