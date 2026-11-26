# selfmodel_simplesonic_20261126：SIMPLE 上 SONIC 的直接执行 A 对一次性修正 R_D（草稿）

- **后续（2026-11-26T13:10Z）**：用户 11-26 定了"只采原地，放宽到 5 cm/s"，和 HA 的 SONIC 相同。SONIC 这一格改由 [selfmodel_simplesonicdata_20261126](selfmodel_simplesonicdata_20261126.md) 做：在 SIMPLE 里采数据、训 D_S，当天用本条的 F1-0 规则和 simpleheft 的 F1(a) 当闸门，过了再跑 A 对 R_D。本条的正式运行不开；下面的 F1-0 结果是那条的出发点。本条随那条的登记一起提交，内容不改。
- **状态**：**草稿，没登记、没开跑**（协调方 11-26：只写草稿）。下面"开跑前的核对 F1-0"是只读的，不跑仿真。它的规则和预测写于 2026-11-26T11:55Z，在算之前；11:59Z 算完。**结果：两个候选模型都没过 F1-0，按规则正式运行不开**，所以加了一节"SIMPLE 上采 SONIC 数据"的方案，也是只写不跑。
- **日期**：2026-11-26。
- **先要定的一件事：用哪个模型。**
  - 协调方原话是"SONIC 用 simpledata 里训好的 SONIC 自身模型"，但 simpledata 没有训 SONIC：
    - 条目头部写明"只做 ScaleBFM"。理由是 SONIC 在 SIMPLE 现在的接法下采不出合格数据：有些姿态下手持续摆动 2–11 cm/s；先转身再走时摔 3/10；单手可用段大约每 20 次重置才有 1 段。条目写的是"这件事要用户定"。
    - `selfmodel_simpledata_20261126/train/` 下只有 `models_D_s0`，data3 的 run.json 写的是 `controller: scalebfm`。
  - 现有的 SONIC 自身模型，都是在 LocoManiLab（LML）上训的：

    | 候选 | 路径（`/wuji-vefps-D/wuji-il/yijie/loco_outputs/` 下） | 训练数据 | 特征 | LML val 中位数 |
    |---|---|---|---|---|
    | D_sonic | `selfmodel_train3b_20261125/sonic/models_D_s0/E3_ensemble_f1` | LML 的 data3、data3b 和两份 mid，训练 1 079 768 段 | GOAL_REF、APPROACH、QFIX，没有 FDVEL | 3.150 cm |
    | F_sonic | `selfmodel_fdvel_20261126/sonic/models_s0/E3_ensemble_f1` | 只有 LML 的 data3，训练 25 046 段 | 同上再加 FDVEL，和 SIMPLE 的 D 是同一套特征 | 4.275 cm |

  - 这两个模型在 SIMPLE 上已经测过，见 xfer 的组合 2 和 fdvel 的"换仿真记录"：
    - 误差是自己 LML val 的倍数：T3 是 1.785 倍，F 是 2.005 倍；第一次伸手是 2.49 倍和 2.83 倍。
    - 模型误差除以直接执行的偏差：0.93 和 1.01。也就是说，模型几乎不比"不修正"准。
    - 87.5–89.3 % 的段，至少有一项输入超出训练集 6 个标准差。
    - 起点的 29 个关节都在 train3 训练集 0.5–99.5 % 范围里的段：0.0 %。最常出界的是右髋 pitch，占 96.4 %。
    - xfer 当时的结论是"要重采"。
  - 第三条路是在 SIMPLE 里给 SONIC 采数据、训模型，照 HumanoidArena 那边 hadata_sonic 的办法（停稳阈值放宽到 5 cm/s）。这要另开条目，由用户来定。
  - **本草稿按 D_sonic 和 F_sonic 写，F1-0 是闸门。** 两个模型 F1-0 都不过，就停在这里，把选哪条路交给用户，正式运行不开。
- **机器 / GPU**：4090b，或 4090 的 2、3 号卡。0 号卡不碰，每张卡总量 ≤ 20 GB，我们每张卡最多 2 个进程，CPU 合计 ≤ 8 个进程，nice 10。两种跑法见"运行计划"。
- **代码**：分支 `sonic-selfmodel`，在 simplecmp 和 simplecmp2 的开关上加。要改的地方列在"要加的东西"里，全部默认关闭，关着时逐位不变。开跑前用冻结脚本把代码抄进 `asrun/`，记下 md5。这次冻结脚本排除 `__pycache__`，见 simplecmp2 的"意外"第 5 条。
- **关联**：
  - [selfmodel_simplecmp_20261126](selfmodel_simplecmp_20261126.md)：ScaleBFM 上的 A 对 R_D。R 规则、对准手掌、F1 的三项、布局符号检验都照它。
  - [selfmodel_simplecmp2_20261126](selfmodel_simplecmp2_20261126.md)：BendPick 上加站位抖动，结论还在复核。
  - [selfmodel_xfer_20261125](selfmodel_xfer_20261125.md)、[selfmodel_fdvel_20261126](selfmodel_fdvel_20261126.md)：LML 的 SONIC 模型搬到 SIMPLE 上的误差。
  - [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md)：为什么只采了 ScaleBFM。
  - [simple_relook_20261126](simple_relook_20261126.md)：补步把站偏中位数减半，A 不涨；按判据，站位不是瓶颈，转去查 SONIC 的手。另外 SONIC 的网络放 CPU 和放 GPU 时，结果不逐位相同。
  - vrref 的手速和手偏结果见 [simple_sonic_vrref_20261126](simple_sonic_vrref_20261126.md) 的"其他度量"和复核。
  - [simple_sonic_vrref_20261126](simple_sonic_vrref_20261126.md)：主指标的 SONIC 一行仍是 `vr_frame="robot"`，本条也用 `"robot"`。
  - 主指标 `bench_rgbd_20261124/sonic` 的 A 栏：45 s 内 29/120，按任务是 BendPick 11、TabletopGrasp 10、XMoveBendPick 3、XMovePick 5；官方时限内 18，按任务是 0、10、3、5。

## 这个 R_D 怎么回答"SONIC 的手"

- **为什么问手**：simple_relook 的判据落在"补步后站位误差中位数 ≤ 10 cm，但 A 多不到 3 局"那一行：补步把站偏中位数从 19.46 cm 降到 9.76 cm，A 只是 31 对 30。按判据，站位不是瓶颈，停止在站位上投入，转去查 SONIC 的手。
- **关于手，已经知道的**（都是 `"robot"` 坐标，也就是主指标和本条用的坐标）：
  - **手晃**：合手前 0.5 s 的手速中位数 2.18 cm/s，`"reference"` 下是 0.60（vrref）。但同一种坐标内部，手速分不出成败：robot 下成功局 2.28、失败局 2.71 cm/s（vrref 复核）。
  - **手偏**：主指标 A 的手掌偏差，进入段中位数按任务是 5.4–9.1 cm；成功局 3.2–5.7 cm，失败局 6.2–9.6 cm。手偏分得出成败。
- **要回答的问题**：SONIC 的手偏，是"命令 → 落点"里一个跟着状态走、可以预测的偏差，还是下命令那一刻预测不了的东西（手晃、历史、接口的随机性）？前一种，自身模型能学，一次性修正能修；后一种，修命令没用，要改控制或加反馈。
- **R_D 怎么回答**：
  1. **能预测的部分有多大**：AL 发的是 A 的原命令，但每一下都记录模型对落点的预测。拿预测和实测落点比，就是模型误差；拿目标和实测落点比，就是直接执行的手偏。两者之比（模型 / 直接）越小，手偏里能从下命令时的状态预测的部分越大。ScaleBFM 在 BendPick 上大约是 0.2（simplecmp2 的 E1：AL 进入段 D 误差 1.13 cm，手偏 5.74 cm）。
  2. **修掉以后成功跟不跟**：R_D 对 AL 的手掌偏差和成功数。手偏降了、成功也涨，说明手偏就是手的问题；手偏降了、成功不涨，说明剩下的问题在合手和手晃，不在目标。
  3. **只在站得好的局里读**：按 simple_relook 的分组（站偏和里程计偏都小、只是站偏、只是里程计偏、两项都大）分开报。站偏的局，伸手命令修得再准也够不着，不能拿来回答手的问题。
  4. **手晃单独看**：AL 和 R_D 都报合手前的手速。R_D 只改目标、不改控制器，手速应该不变；它是读第 2 条时的对照。
- **前提是模型可信**：模型在 SONIC 的 SIMPLE 起点上不可信时（F1 不过），"模型 / 直接 ≈ 1"读不成"手偏预测不了"，只能读成"这个模型没见过这些状态"。所以 F1-0 是闸门。F1-0 不过，要回答手的问题，就需要在 SIMPLE 里给 SONIC 采数据（第三条路）。采集本身也会量出 SONIC 的手在 SIMPLE 里多久能停稳；simpledata 开发时看到的是，有些姿态下手一直摆 2–11 cm/s。
- **R_D 回答不了的**：站偏；`"reference"` 坐标下骨盆前倾不跟参考的问题（vrref 复核：参考前倾约 40°，SONIC 只有 8–16°），本条用 `"robot"`，不碰这个问题；合手阶段本身。

## 假设

1. **主假设。** 动作和控制器都不变，只把 A 的直线命令换成在 SONIC 自身模型上反解得到的命令 R_D，手就落得更准、成功更多。这和 ScaleBFM 在 BendPick 上看到的一样。
   - 有提高的余地：A 进入段的手掌偏差中位数是 5.4–9.1 cm，按任务不同。成功局是 3.2–5.7 cm，失败局是 6.2–9.6 cm。
2. **前提：模型在 SONIC 的 SIMPLE 伸手起点上可信。** 这是本条最可能不成立的地方，所以先查（F1-0）。
3. **即使模型可信，增益也有上限。**
   - simple_relook 把 SONIC 的 A 失败按两项分过组：91 局失败里，只有 16 局站位偏差和里程计偏差都小；47 局只是站偏，27 局两项都大。站偏中位数是 19.6 cm，TabletopGrasp 主要是横向偏约 25 cm。
   - 一次性修正伸手命令，修不了站得太远、够不着的局。
4. **SONIC 本身的翻转噪声大。** simple_relook 里，什么都不改，网络从 GPU 换到 CPU，A 就有 20/19 局翻转。所以命令的任何改动都会带来一批来回的翻转，配对检验的检验力低。

## 开跑前的核对 F1-0（只读，不跑仿真；规则写于 11:55Z，在算之前）

- **问题**：SONIC 在 SIMPLE 上的伸手起点，有多少落在它自己模型的训练范围里。
- **起点**：主指标 SONIC A 栏的 120 局（`bench_rgbd_20261124/sonic/rows/A_*.jsonl`）。每局有预抓取和进入段两下，取每一下的 `S0`，即下命令时的 29 个关节角。这正是 R_D 要修正的两下。
- **训练范围**：候选模型自己的训练划分（`data.splits` 的 train）。取每段起点行（`so + k0`）的 29 个关节，每个关节取 0.5 % 和 99.5 % 分位。29 个关节都在范围里，才算"在范围内"。这条和 xfer 的 sx_eval 相同，也就是 sd_range 39 项里的关节那 29 项。
- **为什么只看关节**：主指标的行只记了 `S0` 的 29 个关节，没有骨盆高度、重力方向和手点，也没有可以拿来算最近邻的整段输入。
  - 所以 F1-0 是正式 F1(b) 的**上界**。正式 F1(b) 要求 39 项、最近邻、伸手表三项检查都满足，关节在范围内是其中的必要条件。
  - F1-0 已经低于 80 %，正式 F1(b) 一定也低于 80 %。
  - 主指标的行是网络放 GPU 跑出来的。如果按"跑法 C"放 CPU，起点会有小差别（见"运行计划"）。
- **参照**：同样的规则，算 ScaleBFM 主指标 A 栏的起点对 SIMPLE 的 D（simpledata，`models_D_s0`）的训练划分。ScaleBFM 正式 F1(b) 在 R_D 发出的命令上（起点和 AL 几乎一样）是 BendPick 88.3 %、TabletopGrasp 63.3 %（simplecmp），关节这一项只会更高。
- **报**：每个候选、每个任务、预抓取和进入段分开，再合起来。报在范围内的比例，出界关节数的中位数，最常出界的 5 个关节。区间按布局（任务，ep % 10）重抽 1000 次，seed 0。
- **预测**：
  - D_sonic 和 F_sonic，四个任务合起来在范围内 0–30 %，每个任务都 ≤ 40 %。理由：xfer 在 SIMPLE 站立伸手的起点上是 0.0 %，右髋 pitch 出界 96.4 %；任务局先走路、BendPick 还要下蹲，起点离 LML 只会更远。D_sonic 多了 data3b，范围可能宽一些，所以给到 30 %。
  - ScaleBFM 参照：BendPick ≥ 85 %，TabletopGrasp ≥ 60 %。
- **判据**：

  | F1-0 的结果 | 下一步 |
  |---|---|
  | 两个候选在四个任务上都 < 80 % | 不开正式运行。结论写成"SIMPLE 上没有可信的 SONIC 自身模型，R_D 无从检验"，交用户定：在 SIMPLE 采 SONIC 数据（照 hadata_sonic），或者 SIMPLE 上不做 SONIC 的 R_D |
  | 某个候选在至少一个任务上 ≥ 80 % | 用这个候选（两个都满足时，取过线任务多的；一样多取 F_sonic，因为特征和 SIMPLE 的 D 相同），补完"要加的东西"、冒烟和登记前核对，再请协调方定开不开跑。判定集不改（见下），但只有 F1 判"可信"的任务，成功差才记到模型名下 |
  | ScaleBFM 参照 BendPick < 85 % | 规则本身有问题，先停下查清，不读 SONIC 的数 |

### F1-0 的结果（2026-11-26T11:57:33–11:59:04Z，4090 的 CPU，nice 10）

- **跑法**：`pre/ss_f1pre.sh`（284316d1）调 `pre/ss_f1pre.py`（b420de40，Mac 上是 `scripts/experiments/selfmodel_simplesonic_20261126/ss_f1pre.py`，未提交）。代码用 simplecmp2 的冻结副本 `selfmodel_simplecmp2_20261126/asrun/code`，开跑前 CODE_MD5 核对通过。输出在 `loco_outputs/selfmodel_simplesonic_20261126/pre/`：`f1pre.txt`（9f4d1581）和三个 json。
- **训练划分核对**：三个模型的 train 段数都和各自 train.json 的 n_train 相等（1 079 768、25 046、809 804）。

`f1pre.txt` 原文：

```
== D_sonic: train /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3b_20261125/sonic/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/sonic/data/mid_data3.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/sonic/data/mid_data3b.npz; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '0', 'REACHWM_TRAIN_HELD': '', 'REACHWM_SUBSET': ''}
  train split 1079768 segments (train.json n_train 1079768: equal); val 5855, val starts in range 84.3 %
  rows A_XMovePickTeleop.jsonl: 30 episodes, md5 d048ad2939ce24ae3e14a07ec9144335
  rows A_XMoveBendPickTeleop.jsonl: 30 episodes, md5 99fc9d8f73d29c3b0c09e4fc8ac55be8
  rows A_BendPickMP.jsonl: 30 episodes, md5 aeb8038f8cc8626d7d51612a4b085f34
  rows A_TabletopGraspMP.jsonl: 30 episodes, md5 8f91c4a3f98f610ef59dcd320f6050c5
  in range (all 29 joints inside the train split's 0.5 / 99.5 %): share [95 %, layouts resampled]; joints out median; most often out (% of starts)
    XMovePickTeleop      pregrasp n  30   96.7 % [ 90.0, 100.0]  out median 0  [('left_ankle_roll_joint', 3.3)]
    XMovePickTeleop      entry    n  30   46.7 % [ 30.0,  66.7]  out median 1  [('right_elbow_joint', 40.0), ('right_hip_pitch_joint', 33.3), ('waist_pitch_joint', 13.3), ('left_hip_pitch_joint', 3.3)]
    XMovePickTeleop      both     n  60   71.7 % [ 63.3,  78.3]  out median 0  [('right_elbow_joint', 20.0), ('right_hip_pitch_joint', 16.7), ('waist_pitch_joint', 6.7), ('left_hip_pitch_joint', 1.7), ('left_ankle_roll_joint', 1.7)]
    XMoveBendPickTeleop  pregrasp n  30   86.7 % [ 76.7,  96.7]  out median 0  [('right_hip_pitch_joint', 13.3)]
    XMoveBendPickTeleop  entry    n  30   36.7 % [ 20.0,  56.7]  out median 1  [('waist_pitch_joint', 23.3), ('waist_roll_joint', 20.0), ('right_hip_pitch_joint', 16.7), ('right_elbow_joint', 16.7), ('right_wrist_pitch_joint', 13.3)]
    XMoveBendPickTeleop  both     n  60   61.7 % [ 51.7,  73.3]  out median 0  [('right_hip_pitch_joint', 15.0), ('waist_pitch_joint', 11.7), ('waist_roll_joint', 10.0), ('right_elbow_joint', 8.3), ('right_wrist_pitch_joint', 6.7)]
    BendPickMP           pregrasp n  30   76.7 % [ 63.3,  90.0]  out median 0  [('right_hip_pitch_joint', 16.7), ('left_ankle_pitch_joint', 6.7)]
    BendPickMP           entry    n  30    3.3 % [  0.0,  10.0]  out median 1  [('waist_roll_joint', 66.7), ('right_hip_pitch_joint', 26.7), ('waist_pitch_joint', 20.0), ('right_ankle_pitch_joint', 10.0)]
    BendPickMP           both     n  60   40.0 % [ 31.7,  48.3]  out median 1  [('waist_roll_joint', 33.3), ('right_hip_pitch_joint', 21.7), ('waist_pitch_joint', 10.0), ('right_ankle_pitch_joint', 5.0), ('left_ankle_pitch_joint', 3.3)]
    TabletopGraspMP      pregrasp n  30   80.0 % [ 63.3,  93.3]  out median 0  [('waist_roll_joint', 10.0), ('left_hip_roll_joint', 6.7), ('right_hip_roll_joint', 6.7), ('right_hip_pitch_joint', 3.3), ('right_shoulder_yaw_joint', 3.3)]
    TabletopGraspMP      entry    n  30   43.3 % [ 33.3,  53.3]  out median 1  [('right_hip_pitch_joint', 26.7), ('left_ankle_roll_joint', 20.0), ('waist_roll_joint', 13.3), ('left_hip_yaw_joint', 10.0), ('waist_pitch_joint', 10.0)]
    TabletopGraspMP      both     n  60   61.7 % [ 50.0,  73.3]  out median 0  [('right_hip_pitch_joint', 15.0), ('waist_roll_joint', 11.7), ('left_ankle_roll_joint', 10.0), ('left_hip_yaw_joint', 5.0), ('waist_pitch_joint', 5.0)]
    all                  pregrasp n 120   85.0 % [ 78.3,  90.9]  out median 0  [('right_hip_pitch_joint', 8.3), ('waist_roll_joint', 2.5), ('right_hip_roll_joint', 1.7), ('left_hip_roll_joint', 1.7), ('left_ankle_pitch_joint', 1.7)]
    all                  entry    n 120   32.5 % [ 23.3,  41.7]  out median 1  [('right_hip_pitch_joint', 25.8), ('waist_roll_joint', 25.0), ('waist_pitch_joint', 16.7), ('right_elbow_joint', 14.2), ('left_ankle_roll_joint', 5.0)]
    all                  both     n 240   58.8 % [ 52.9,  64.6]  out median 0  [('right_hip_pitch_joint', 17.1), ('waist_roll_joint', 13.8), ('waist_pitch_joint', 8.3), ('right_elbow_joint', 7.1), ('left_ankle_roll_joint', 2.9)]
== F_sonic: train /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic/data/all.npz; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '1', 'REACHWM_TRAIN_HELD': '', 'REACHWM_SUBSET': ''}
  train split 25046 segments (train.json n_train 25046: equal); val 3134, val starts in range 81.7 %
  rows A_XMovePickTeleop.jsonl: 30 episodes, md5 d048ad2939ce24ae3e14a07ec9144335
  rows A_XMoveBendPickTeleop.jsonl: 30 episodes, md5 99fc9d8f73d29c3b0c09e4fc8ac55be8
  rows A_BendPickMP.jsonl: 30 episodes, md5 aeb8038f8cc8626d7d51612a4b085f34
  rows A_TabletopGraspMP.jsonl: 30 episodes, md5 8f91c4a3f98f610ef59dcd320f6050c5
  in range (all 29 joints inside the train split's 0.5 / 99.5 %): share [95 %, layouts resampled]; joints out median; most often out (% of starts)
    XMovePickTeleop      pregrasp n  30   93.3 % [ 83.3, 100.0]  out median 0  [('left_ankle_roll_joint', 6.7)]
    XMovePickTeleop      entry    n  30   33.3 % [ 16.7,  50.0]  out median 1  [('right_elbow_joint', 40.0), ('right_hip_pitch_joint', 36.7), ('waist_pitch_joint', 20.0), ('left_hip_pitch_joint', 16.7), ('right_hip_yaw_joint', 6.7)]
    XMovePickTeleop      both     n  60   63.3 % [ 55.0,  71.7]  out median 0  [('right_elbow_joint', 20.0), ('right_hip_pitch_joint', 18.3), ('waist_pitch_joint', 10.0), ('left_hip_pitch_joint', 8.3), ('left_ankle_roll_joint', 3.3)]
    XMoveBendPickTeleop  pregrasp n  30   70.0 % [ 53.3,  86.7]  out median 0  [('right_hip_pitch_joint', 20.0), ('right_hip_yaw_joint', 6.7), ('left_hip_pitch_joint', 3.3)]
    XMoveBendPickTeleop  entry    n  30   33.3 % [ 16.7,  53.3]  out median 1  [('waist_pitch_joint', 23.3), ('right_wrist_pitch_joint', 23.3), ('waist_roll_joint', 20.0), ('right_hip_pitch_joint', 20.0), ('right_elbow_joint', 16.7)]
    XMoveBendPickTeleop  both     n  60   51.7 % [ 38.3,  65.0]  out median 0  [('right_hip_pitch_joint', 20.0), ('waist_pitch_joint', 11.7), ('right_wrist_pitch_joint', 11.7), ('waist_roll_joint', 10.0), ('left_hip_pitch_joint', 10.0)]
    BendPickMP           pregrasp n  30   76.7 % [ 60.0,  93.3]  out median 0  [('right_hip_pitch_joint', 23.3)]
    BendPickMP           entry    n  30    0.0 % [  0.0,   0.0]  out median 1  [('waist_roll_joint', 66.7), ('right_hip_pitch_joint', 26.7), ('right_ankle_pitch_joint', 23.3), ('waist_pitch_joint', 20.0), ('right_hip_roll_joint', 10.0)]
    BendPickMP           both     n  60   38.3 % [ 30.0,  46.7]  out median 1  [('waist_roll_joint', 33.3), ('right_hip_pitch_joint', 25.0), ('right_ankle_pitch_joint', 11.7), ('waist_pitch_joint', 10.0), ('right_hip_roll_joint', 5.0)]
    TabletopGraspMP      pregrasp n  30   46.7 % [ 26.7,  66.7]  out median 1  [('right_hip_yaw_joint', 23.3), ('right_shoulder_yaw_joint', 20.0), ('right_ankle_roll_joint', 16.7), ('right_hip_pitch_joint', 10.0), ('right_hip_roll_joint', 10.0)]
    TabletopGraspMP      entry    n  30   10.0 % [  0.0,  20.0]  out median 2  [('right_hip_yaw_joint', 60.0), ('right_hip_pitch_joint', 36.7), ('left_ankle_roll_joint', 20.0), ('waist_roll_joint', 13.3), ('right_shoulder_yaw_joint', 13.3)]
    TabletopGraspMP      both     n  60   28.3 % [ 18.3,  38.3]  out median 1  [('right_hip_yaw_joint', 41.7), ('right_hip_pitch_joint', 23.3), ('right_shoulder_yaw_joint', 16.7), ('waist_roll_joint', 11.7), ('left_ankle_roll_joint', 11.7)]
    all                  pregrasp n 120   71.7 % [ 61.7,  80.8]  out median 0  [('right_hip_pitch_joint', 13.3), ('right_hip_yaw_joint', 7.5), ('right_shoulder_yaw_joint', 5.0), ('right_ankle_roll_joint', 4.2), ('right_hip_roll_joint', 2.5)]
    all                  entry    n 120   19.2 % [ 11.7,  27.5]  out median 1  [('right_hip_pitch_joint', 30.0), ('waist_roll_joint', 25.0), ('right_hip_yaw_joint', 20.0), ('waist_pitch_joint', 18.3), ('right_elbow_joint', 14.2)]
    all                  both     n 240   45.4 % [ 39.2,  52.1]  out median 1  [('right_hip_pitch_joint', 21.7), ('waist_roll_joint', 13.8), ('right_hip_yaw_joint', 13.8), ('waist_pitch_joint', 9.2), ('right_elbow_joint', 7.1)]
== ScaleBFM_D_reference: train /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/data3/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/data3b/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/train/data/mid_data3.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/train/data/mid_data3b.npz; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '1', 'REACHWM_TRAIN_HELD': '', 'REACHWM_SUBSET': ''}
  train split 809804 segments (train.json n_train 809804: equal); val 6272, val starts in range 76.3 %
  rows A_XMovePickTeleop.jsonl: 30 episodes, md5 bd79e07f5255f380922f486ecce1082f
  rows A_XMoveBendPickTeleop.jsonl: 30 episodes, md5 b96f3f74f372fa6231024bea7cc0e421
  rows A_BendPickMP.jsonl: 30 episodes, md5 837431be79e987dd9ed39be74709fe3d
  rows A_TabletopGraspMP.jsonl: 30 episodes, md5 022b528da474eef929ed2c855ba957e6
  in range (all 29 joints inside the train split's 0.5 / 99.5 %): share [95 %, layouts resampled]; joints out median; most often out (% of starts)
    XMovePickTeleop      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    BendPickMP           pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    BendPickMP           entry    n  30   76.7 % [ 56.7,  96.7]  out median 0  [('right_wrist_roll_joint', 23.3)]
    BendPickMP           both     n  60   88.3 % [ 76.7,  98.3]  out median 0  [('right_wrist_roll_joint', 11.7)]
    TabletopGraspMP      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    TabletopGraspMP      entry    n  30   20.0 % [  6.7,  36.7]  out median 1  [('right_wrist_roll_joint', 56.7), ('right_wrist_pitch_joint', 16.7), ('left_knee_joint', 10.0), ('right_knee_joint', 3.3), ('right_wrist_yaw_joint', 3.3)]
    TabletopGraspMP      both     n  60   60.0 % [ 51.7,  68.3]  out median 0  [('right_wrist_roll_joint', 28.3), ('right_wrist_pitch_joint', 8.3), ('left_knee_joint', 5.0), ('right_knee_joint', 1.7), ('right_wrist_yaw_joint', 1.7)]
    all                  pregrasp n 120  100.0 % [100.0, 100.0]  out median 0  []
    all                  entry    n 120   74.2 % [ 61.7,  85.8]  out median 0  [('right_wrist_roll_joint', 20.0), ('right_wrist_pitch_joint', 4.2), ('left_knee_joint', 2.5), ('right_knee_joint', 0.8), ('right_wrist_yaw_joint', 0.8)]
    all                  both     n 240   87.1 % [ 80.4,  92.9]  out median 0  [('right_wrist_roll_joint', 10.0), ('right_wrist_pitch_joint', 2.1), ('left_knee_joint', 1.2), ('right_knee_joint', 0.4), ('right_wrist_yaw_joint', 0.4)]
F1-0 done 2026-11-26T11:59:04Z
```

**对照判据：落在第 1 行，两个候选在四个任务上都 < 80 %。** 按规则，正式运行不开。

| 任务（两下合起来） | D_sonic | F_sonic | ScaleBFM 参照 |
|---|---|---|---|
| XMovePick | 71.7 % | 63.3 % | 100.0 % |
| XMoveBendPick | 61.7 % | 51.7 % | 100.0 % |
| BendPick | 40.0 % | 38.3 % | 88.3 % |
| TabletopGrasp | 61.7 % | 28.3 % | 60.0 % |
| 四个任务 | 58.8 % [52.9, 64.6] | 45.4 % [39.2, 52.1] | 87.1 % [80.4, 92.9] |

- **参照过线**：ScaleBFM BendPick 88.3 % ≥ 85 %，TabletopGrasp 60.0 % 正好压在预测下限上。规则本身没问题。BendPick 的 88.3 % 恰好等于 simplecmp 里 R_D 命令上 39 项检查的 88.3 %。
- **坏在进入段**：D_sonic 的预抓取四个任务合起来是 85.0 %，进入段只有 32.5 %，BendPick 的进入段是 3.3 %。F_sonic 是 71.7 % 和 19.2 %，BendPick 进入段 0.0 %。进入段出界最多的是右髋 pitch、腰 roll、腰 pitch、右肘，也就是 SONIC 在 SIMPLE 里伸手时躯干和手臂的姿势，LML 的数据里没有。simplecmp2 的 E2 说明，ScaleBFM 在 BendPick 上的提高全部来自进入段的修正，而 SONIC 的进入段起点恰恰最不在范围里。
- **预测落空**：预测 0–30 %、每个任务 ≤ 40 %；实际 D_sonic 58.8 %、F_sonic 45.4 %，只有 BendPick（40.0 % / 38.3 %）和 F_sonic 的 TabletopGrasp（28.3 %）在 40 % 以下。预测是从 xfer 的 0.0 % 推的，但 xfer 的起点是 SIMPLE 宿主自己的原地伸手（xfer 做法：每次重置原地伸两次手），不是任务局里走到桌前以后的起点。任务局的预抓取起点反而大多在范围里。
- **不改读法**：判据按任务、两下合起来读，和 simplecmp 登记的 F1(b) 一致。只看预抓取时，D_sonic 有三个任务 ≥ 80 %。但那是看到结果以后换的读法，不用。而且 F1-0 只是关节的上界，xfer 里 LML 的 SONIC 模型第一次伸手的误差已经是 val 的 2.5–2.8 倍，F1(a) 也过不了。
- **所以**：现有的 SONIC 模型回答不了"SONIC 的手"这个问题，要回答，先在 SIMPLE 上采 SONIC 数据，方案见下一节。

## SIMPLE 上采 SONIC 数据（方案，不登记、不开跑）

- **为什么采**：F1-0 不过，SIMPLE 上没有可信的 SONIC 自身模型。没有可信的模型，"SONIC 的手偏能不能从状态预测"就回答不了（见"前提是模型可信"）。采集本身也会量出 SONIC 的手在 SIMPLE 里停不停得住。
- **配方**：照 [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md)，只把控制器换成 SONIC。
  - 采集器 `sc_collect.py`（包 xfer 的 SIMPLE 宿主，抽样、静止判据、分段、检查调 d3_collect）。SONIC 走 SIMPLE 自己的接口，`vr_frame="robot"`，和主指标相同。
  - SONIC 用 CPU 推理：`REACH_SONIC_NETS=allcpu`，`REACH_SONIC_THREADS=1`（xfer 的 sx_collect 对 SONIC 的设置），`CUDA_VISIBLE_DEVICES` 为空。一个进程约占 1 个核。采集不渲染，不用 GPU。
  - **data3 式**：30 000 次重置。先走后伸的重置里，一半照 d3 的规则（转、走、转），一半用任务式走法（不转身，位移落在 APPROACH_BOX 里），按 `crc32("taskwalk:<重置编号>") % 2` 分。起始扰动 0.007 rad。
  - **data3b 式**：15 000 次重置，四种伸手照 data3b，走法分法和 data3 式相同。
  - **中间点**：照 simpledata 的 `sd_mid` 做 mid_data3、mid_data3b。
  - **训练**：E3_ensemble，3 万步，batch 1024，5 个成员，`REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1`，frac 1.0 和 0.3，seed 0。和 SIMPLE 的 D 同一套。
  - **评估**：照 simpledata 的评估 1–4：SIMPLE 自己的 val 和 test_iid；xfer 的 SONIC·SIMPLE 记录（400 次重置）当额外测试集；真实任务局（SONIC 的 120 局 A，带记录器）；起点范围检查。真实任务局那一份，就是本条 AL 要的记录。
  - **编号**：重置号和种子要和已有的不重叠。已经用过的有 50M、60M（simpledata）、89M–95M（HA、simplecmp 等）、107M、200M 等。候选是重置号 140 000 000 / 141 000 000，种子 142 000 000 / 143 000 000，开发阶段查重以后定。
- **配方里要先定的两件事**（照 simpledata 的说法："要用户定"）：
  1. **停稳阈值**。simpledata 开发时，SONIC 有些姿态下手一直摆 2–11 cm/s，单手可用段约每 20 次重置才有 1 段。照原阈值，45 000 次重置只能出两千多段，不够训练（SIMPLE 的 ScaleBFM 是 14.2 万段，LML 的 SONIC data3 是 4.66 万段）。HumanoidArena 的 SONIC 遇到过同样的问题，用户定的是放宽到 5 cm/s、最多等 6 s：每次重置的单手伸手，data3 式从 0.8 升到 1.4，data3b 式从 0.33 升到 1.67（hadata_sonic 的开发阶段）。SIMPLE 上可以照这个办法，但要用户点头。放宽的代价是下命令时 SONIC 还在动，FDVEL 特征里有速度，模型看得到。
  2. **转身再走的那一半**。simpledata 开发时，SONIC 先转身再走摔 3/10，而且 SIMPLE 的任务走路时不转身。可以照配方保留，也可以全用任务式走法。全用任务式会少一种起点，要和协调方定。
- **开发阶段**（正式采集之前，在开发编号上做，不进正式数据）：定停稳阈值以后，data3 式和 data3b 式各跑 20 次重置，量每次重置的可用单手段数、摔倒率、每次重置的用时和每个进程的内存；H20b 和 4090 上各跑同一个分片，日志和每段数组逐位相同，才开正式采集。
- **机器**：H20b（新来的 8 卡 H20，171 个 CPU，接入核对中）。只用 CPU，nice 10。启动方式照 4090b 的 `py4090b.py`，按 vePFS 的真实路径重建 sys.path，另设 `LOCOMANI_WORKSPACE`、`SONIC_POLICY`。训练用 H20b 的 GPU，每张卡总量不超过 20 GB（照现在的规矩，H20b 的上限等接入核对时再定）。
- **局数和耗时**：
  - **每次重置的用时**（单进程，SONIC 在 CPU 上）：开发日志里，原地重置 22.3–23.6 s，先走后伸 33.7–42.5 s，data3b 式 39.4 s（`selfmodel_simplecollect_20261126/dev/` 的 `cal/sonic012_0.log`、`try_b_sonic.log`、`try1s.log`、`try1sb.log`、`face/sonic_0.log`，每份只有 2–8 次重置）。ScaleBFM 正式采集比冒烟慢 1.2–1.4 倍（data3 冒烟中位 9.5 s，正式最慢分片平均 11.4 s；data3b 11.5 s 对 16.2 s）。照这个比例估计：
    - data3 式：先走后伸约占 1/3（simpledata 是 9 927 / 30 000），平均约 28 s，正式约 34–39 s；
    - data3b 式：正式约 47–55 s。
  - **合计**：data3 式 30 000 × 34–39 s ≈ 283–325 核·小时；data3b 式 15 000 × 47–55 s ≈ 196–229 核·小时。**合计约 480–555 核·小时**。停稳阈值放宽以后，等待会少，但每次重置伸手多了，两边可能抵掉，所以区间要放宽到 400–700 核·小时。
  - **墙钟**（一个进程一个核）：

    | 进程数（= 核数） | 采集墙钟 |
    |---|---|
    | 64 | 约 6.3–11 h |
    | 96 | 约 4.2–7.3 h |
    | 128 | 约 3.1–5.5 h |

    这还不算开发阶段约 1–2 h，以及训练和评估约 1 h（simpledata 的 D 训练用了 545 s，加上 mid 和评估）。
  - **预计的段数**：照 HA 放宽以后的产出（每次重置 1.4–1.67 个单手伸手），约 4–7 万个单手伸手。SIMPLE 上实际多少，开发阶段量。
- **之后**：模型训好、评估 4（范围检查）过了，再回到本条：用新模型重算 F1-0，过了才补完"要加的东西"、冒烟和登记前核对。

## 做法（F1-0 过了才用）

| 组 | 预抓取 | 进入段 | 局 |
|---|---|---|---|
| AL（A 加记录） | 目标；R 的反解照算，只记录（dryR） | 同左 | 120 |
| R_D | R：在候选模型上反解，候选缩回到 IK ≤ 2 cm | R | 120 |

- **局号和种子**：4 个任务 × 第 0–29 局，和主指标 SONIC 一行相同，按局配对。种子沿用 simple_bench 的默认，不加别的随机源。
- **控制器**：SONIC，`vr_frame="robot"`，sensor=rgbd，里程计根（主指标的 `controller_root: odometry`）。
- **反解**：照 simplecmp 的 R：6 步不动点迭代，对准手掌，缩回到 IK ≤ 2 cm，损失加 1.0 × 成员离散度。只把模型换成候选，候选的特征开关照它训练时的设置：D_sonic 关 FDVEL，F_sonic 开。
- **不跑的组**：
  - S_D：simplecmp 里 S_D 和 R_D 逐位相同；
  - 路线 J：协调方这一条只要 R_D；
  - simplecmp2 的 E2 和 V1：只拆 ScaleBFM 的问题。

## 局、布局和判定集

- **布局**：同一个 index 的三个 level 几乎一样，所以 120 局是 40 个布局。配对检验在布局上做。
- **排除**：照 simplecmp 和 lmlcmp3，三类分开计：感知失败；看完以后、发预抓取之前 IK 失败；没有那一眼。主指标 SONIC A 的 120 局都走到了预抓取和进入段，没有摔倒。
- **到顶**：A ≥ 80 % 的任务排除。SONIC 四个任务都远低于 80 %，所以：
  - 45 s：判定集是四个任务，120 局、40 个布局；
  - 官方时限：BendPick 8 s 够不着，各组都是 0，判定集是另外三个任务。

## 度量

- **主比较 P1**：R_D 对 AL，45 s，判定集 40 个布局。每个布局比三个 level 的成功数，做精确的双侧符号检验，p < 0.05。McNemar 一起报，但不作判据。成功率差的 95 % 区间按布局重抽 4000 次，seed 0。
- **按任务**：四个任务各自报 R_D 对 AL，两个时限，同样的检验和区间。
- **F1，模型在修正后的命令上还可信吗**：照 simplecmp，按任务算三项，任何一项成立就判"不可信"：
  - (a) 模型在任务标签处的误差中位数，除以它自己的 LML val 中位数（D_sonic 3.150 cm，F_sonic 4.275 cm），> 1.3；
  - (b) sd_range 三项检查（39 项、最近邻、伸手表）都满足的比例，训练集换成候选自己的，< 80 %；
  - (c) 发出命令中 IK 误差 > 2 cm 的比例，比 AL 高 ≥ 10 个百分点。

  AL 的三项一起报，作参照。
- **机制，也就是"手"的问题**（见"这个 R_D 怎么回答'SONIC 的手'"）：
  - 每一下手掌偏差的中位数，AL 和 R_D，按任务、预抓取和进入段分开；
  - 模型对发出命令的预测和实测落点之差（手掌），AL 和 R_D 各一个；
  - **模型 / 直接**：AL 上每一下的模型误差除以同一下的手掌偏差，取中位数，按任务、按段，区间按布局重抽；
  - 合手前 0.5 s 的手速中位数（vrref 的量法），AL 和 R_D；
  - 按 simple_relook 的分组（两项都小、只是站偏、只是里程计偏、两项都大）报 R_D 的成功差和手掌偏差。分组用的是 AL 这一局自己的站偏和里程计偏，阈值照 simple_relook，两组同局同值。站偏的局本来就修不好。
- **命令本身**：修正量、缩回次数、思考时间。
- **一致性**：
  - 跑法 G（网络放 GPU）：AL 和主指标 A 逐局相同，只去掉 wall_s 和开关自己加的字段；
  - 跑法 C（网络放 CPU）：AL 和主指标 A 的翻转局数只记录，不作判据（simple_relook 量过是 20/19）；
  - 两种跑法都要：plan_same 全为真，在线、离线预测之差 < 1e-4 m。

## 预测（F1-0 过了、开跑之前，还要再按 F1-0 的数修订一次）

- F1(a)：1.8–3 倍，判"不可信"。依据是 xfer 和 fdvel 的 1.8–2.8 倍。
- P1：R_D − AL 在 45 s 是 −6 到 +6 局，符号检验不显著。
- 手掌偏差：R_D 对 AL，进入段中位数变化 −30 % 到 +20 %。模型几乎不比直接执行准（0.93–1.01），所以不会降很多，也可能变差。
- 模型 / 直接（AL）：0.8–1.2。依据同上。这个数本身不能回答手的问题，因为模型不可信（见"前提是模型可信"）。
- 手速：AL 和 R_D 都在 1.5–3 cm/s，差不到 0.5 cm/s（一格记录精度）。

## 判据（F1-0 过了才用）

| 结果 | 结论和下一步 |
|---|---|
| P1 成立，而且带来提高的任务上 F1 判"可信" | SIMPLE 上，SONIC 的一次性修正提高了成功率，写明是哪个任务带来的 |
| P1 成立，但带来提高的任务上 F1 判"不可信" | 照报，不归功于自身模型 |
| P1 不成立 | 照实报各任务的差和区间，不合并成"赢" |
| 某任务 R_D − AL ≤ −3 局，且 F1 判"不可信" | 模型不可信时修正会伤害，SIMPLE 上的 SONIC 不用 R_D |
| 手掌偏差没降，而 F1 判"可信" | 反解没起作用，先查模型对修正命令的预测误差有没有变大 |
| **手**：F1 可信，AL 的模型 / 直接 ≤ 0.5，R_D 进入段手偏降 ≥ 40 %，且"两项都小"那组 R_D 成功多 | SONIC 的手偏主要是能从状态预测的偏差，自身模型能修。手的问题至少有一大块是这个 |
| 手：F1 可信，手偏降 ≥ 40 %，但"两项都小"那组成功不涨 | 手偏不是限制，剩下的在合手和手晃。下一步查合手阶段，或 vrref 复核提的 b1 / a2 |
| 手：F1 可信，AL 的模型 / 直接 ≥ 0.8 | 手偏在下命令时预测不了，修命令没用。手的问题是手晃、历史或接口，要改控制或加反馈 |
| 手：F1 不可信 | 回答不了手的问题。只报数，不下结论；要回答，先在 SIMPLE 里给 SONIC 采数据 |
| 一致性不满足 | 停下查清，不做评估 |

## 要加的东西（F1-0 过了才做，都默认关闭）

1. `scmp_correct.install` 现在断言 `controller == scalebfm`，要放开 SONIC。只有给了 `--correct` 才会走到这里，所以不影响默认行为。
2. 按模型选特征开关：D_sonic 关 FDVEL。还要一份 SONIC 数据的模板，提供输入的维度。
3. 先查 SIMPLE 的 SONIC 电机是不是也走 `Motor.reach` 和 `_blend`。候选命令的规划要在 `_blend` 里截停，电机状态不动，和 ScaleBFM 一样。
4. 手掌和 13 cm 点的换算（REACH_HAND_OFF 0.0415、HAND_OFF 0.13），先核对 SONIC 用的是同一只 Dex3 手。
5. 登记前的核对照 simplecmp：开关关着时和主指标逐位相同（跑法 G），dryR 和开关关着时逐位相同，冒烟 2 局。

## 运行计划（F1-0 过了才用）

- **11:48Z 的显存**：4090 的 1、2、3 号卡各用了 16.4 GB，4090b 的 0、1 号卡也各用了 16.4 GB（应该是 HA 采集）。
- **跑法 G：网络放 GPU，能和主指标逐位对上。** SONIC 网络放 GPU 时一个进程约 7.4 GB（simple_relook），现在哪张卡都放不下。要等 HA 让出 4090 的 2、3 号卡，每张卡 1 个进程。AL 必须和主指标 A 逐局相同。
- **跑法 C：网络放 CPU，现在就能跑。** 放 4090b 的 CPU（`REACH_SONIC_NETS=allcpu`，`py4090b.py` 的启动方式），渲染每个进程约 1 GB，放 4090b 两张卡，每卡 ≤ 2 个。AL 在同一次运行里重跑，作为基线；主指标的行不复用。
- 两种跑法都是 240 局。时长等冒烟时再量，超过 5 分钟就登记到灵动岛。

## 预测修订

## 结果（贴原始输出，不转述）

## 意外

## 结论与下一步

## 独立复核
