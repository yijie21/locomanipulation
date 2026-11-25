# selfmodel_xfer_20261125：LocoManiLab 里训出的自身模型，在 SIMPLE 和 HumanoidArena 里还准不准（检验"身体一不一样"）

- **日期**：2026-11-25。所有运行之前写好，单独提交。
- **用户的问题和决定（2026-11-25）**：
  - 用户问："这个自身模型不应该只跟控制器有关吗，为什么会跟我跑哪个 benchmark 有关"。
  - 我的回答：模型描述的是"控制器 + 它驱动的那具身体 + 我们发命令的方式"。每个 benchmark 都自带一具仿真的身体和一套接口，身体到底一不一样，可以直接测出来。
  - 用户说："好，按这个来"。BiGym 不跑，因为它用的是 GR00T-WBC。
- **机器**：
  - SIMPLE 用 4090，或者用已经装好 SIMPLE 的机器。
  - HumanoidArena 用 4090（`tools/humanoidarena`）。
  - 每个组合只采几百次伸手，进程数少，不和正在跑的任务抢。
- **代码**：`selfwm`。在各自的仿真里用各自已有的接口发伸手命令，不改控制器。
  - SIMPLE：`gen_reach.py` 的 SIMPLE 身体配置。
  - HumanoidArena：原生 SONIC，用我们的参考帧驱动。
  - 记录格式和 data3 相同，这样 data.py 能直接读。
  - 跑之前把代码抄进 `asrun/`，记下 md5。
- **组合**：

  | 组合 | 用到的模型 |
  |---|---|
  | ScaleBFM → SIMPLE | LocoManiLab 的 ScaleBFM 模型 |
  | SONIC → SIMPLE | LocoManiLab 的 SONIC 模型 |
  | SONIC → HumanoidArena | LocoManiLab 的 SONIC 模型 |

  HEFT 只在 LocoManiLab 里有，不参加。
- **模型**：用 [selfmodel_train3b_20261125](selfmodel_train3b_20261125.md) 按判据选出的那个模型。train3b 还没出结果的话，先用 [selfmodel_train3_20261125](selfmodel_train3_20261125.md) 的 frac 1.0（T3），等 train3b 出来再在同一份记录上离线重算一遍。

## 做法

在每个组合的仿真里，按 data3 的采法直接执行伸手：
- 每次重置原地伸两次手，每次伸手前都按 data3 的停稳规则等身体停稳；
- 目标、手腕朝向、站或蹲的抽法都和 data3 相同；
- 每个组合采 400 次重置，约 800 段。

记录时，手点要换成 data3 的定义，也就是 wrist_yaw_link 前方 13 cm。每个仿真的 IK 参考都用它自己的生成方式，参考特征照样记下来。

把记录喂给 LocoManiLab 的模型，算预测落点和实际落点的距离。同时算直接执行的误差（手到命令的距离），和 LocoManiLab 同类伸手的直接执行误差对比。

## 假设

同一个控制器，换一个仿真，身体的物理参数和发命令的接口都可能不同，所以 LocoManiLab 的模型在别的仿真里会明显变差。差多少要看这两样差多少。

SONIC 到 HumanoidArena 应该差得最多，因为送参考的方式不同：LocoManiLab 里 SONIC 落后参考 0.94 s，HumanoidArena 里只落后 0.16 s。

## 度量

- **主要**：预测误差的中位数，除以同一个模型在 LocoManiLab val 上的误差。第一次伸手和接着上一次的伸手分开报，再报一个合计。
- **次要**：
  - 直接执行误差的中位数，和 LocoManiLab 的比值；
  - 直接执行误差向量的平均方向，和 LocoManiLab 的方向夹角；
  - 起点在训练范围里的比例（29 个关节），以及集成模型的成员分歧。
- **区间**：按重置重抽样，取 95 % 区间。

## 预测

| 组合 | 预测误差 / LocoManiLab val |
|---|---|
| ScaleBFM → SIMPLE | 1.5–4 倍 |
| SONIC → SIMPLE | 2–5 倍 |
| SONIC → HumanoidArena | 2–6 倍 |

直接执行误差和 LocoManiLab 的比值，三个组合都会落在 0.7 倍以下或 1.4 倍以上，也就是说至少差三成。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| 预测误差 ≤ 1.3 倍 val，区间上限 ≤ 1.5 倍 | 身体基本一样。直接拿 LocoManiLab 的模型去做这个 benchmark 的对比（直接执行对用模型修正） |
| 预测误差 > 1.5 倍 val | 身体不一样。在这个仿真里按 data3 + data3b 的采法重采、重训，之后再对比。重采另行登记 |
| 1.3–1.5 倍之间 | 先在 200 段上做一次微调式的小样本校准（另行登记），再决定 |

## 修订 1：三个控制器在三个 benchmark 里都要跑（2026-11-25，新组合都还没开跑）

**用户的要求（2026-11-25）**：SIMPLE、HumanoidArena、LocoManiLab 三个 benchmark 都跑 SONIC、ScaleBFM、HEFT 三个控制器。原来的三个组合，预测和判据都不变。

**新增三个组合**。三个都要先把控制器接进对应的仿真，接好、冒烟通过以后才开跑：

| 组合 | 现状 | 要做的事 |
|---|---|---|
| HEFT → SIMPLE | SIMPLE 里没有 HEFT | 写一个 SIMPLE 接口。用 g1_backend 的 policy.onnx 和观测构造，喂和 LocoManiLab 里一样的参考帧 |
| ScaleBFM → HumanoidArena | HumanoidArena 只有原生 SONIC | 在 IsaacLab 的仿真循环里换掉 SONIC，改由 ScaleBFM 输出关节目标 |
| HEFT → HumanoidArena | 同上 | 同上，换成 HEFT |

**关节参数怎么定**：同一个 benchmark 里，三个控制器要用同一套身体设置。
- SIMPLE：先查清 ScaleBFM 和 SONIC 用的是哪一套关节参数，HEFT 照用同一套。
- HumanoidArena：用 IsaacLab 自带的 G1 关节参数，PD 增益按各控制器自己的配置。

两边实际怎么设，都在冒烟报告里写明。

**新组合的预测**：
- 预测误差是 LocoManiLab val 的几倍：
  - HEFT → SIMPLE：2–6 倍。HEFT 对关节参数最敏感，在 LocoManiLab 里也最容易摔。
  - ScaleBFM → HumanoidArena：2–6 倍。物理引擎换了，从 MuJoCo 换到 PhysX。
  - HEFT → HumanoidArena：2–8 倍。
- 直接执行误差和 LocoManiLab 比，至少差三成，和原来三个组合一样。

**判据**：和原登记相同。

**接不上时**：哪个组合接不上，就写明卡在哪，那一格留空，不影响其他组合。

## 结果：组合 1，ScaleBFM → SIMPLE（2026-11-25）

**冻结**：2026-11-25T16:04:48Z，代码在 `loco_outputs/selfmodel_xfer_20261125/asrun/`。

| 文件 | md5 |
|---|---|
| sx_collect.py | 962a5fb6 |
| sx_eval.py | 81f61381 |
| sx_batch.sh | c3ed3382 |
| sx_freeze.sh | 73bce510 |
| d3_collect.py | 581fadf5（直接调用它的抽样、停稳、记录、merge、check） |

另外记录了仓库副本和 git rev bc426d5e。

**实现**
- 宿主换成 gen_reach 的 SIMPLE 身体：场景用 simple_g1_sonic，BFM_ROOT=odo。
- ScaleBFM 在进程内经 Motor 跑，用 mode 4 关键点模式，每次下命令都重新钉住脚踝。
- 手点用 REACH_HAND_OFF=0.13,0,0，和 data3 的定义一样。
- 参考用 SIMPLE 自己的生成方式：Motor.reach 在 walk_mars G1 上做全身 IK，min-jerk 插值到 50 Hz。
- 站姿也是 SIMPLE 自己的：默认关节和 LML 的最多差 34°，骨盆高 0.762 m。
- d_this 和 d_prev 都存了，SIMPLE 里没有 xy 平移，所以都是 0。

**一致性检查**：CHECK PASSED，XFER CHECK PASSED。
- 直接执行误差的重算值和记录值，最大差 0.0000 cm。
- 用 SIMPLE 身体的 FK 重算手点，停稳标签上最大差 0.012 cm。
- C 行和规划末位手点的差为 0.0000 cm。
- data.py 能原样读入。
- 身体运动学和 LML 不同：torso_link 原点高 1 cm，随机姿态下手点最多差 0.68 cm；Motor 的 IK 模型手腕差 0.5 cm。

**和登记不一样的地方**
1. SIMPLE 没有参考缓冲，实测延迟为 −2 帧，所以停稳窗口按 0 帧延迟算。
2. 分母用 LML 的 val，里面含先走再伸的重置；SIMPLE 这边只有原地伸手。
3. ScaleBFM 在 CPU 上跑。
4. 冻结前在 PLAN.txt 里加了一条诊断臂，不进判据：在 SIMPLE 身体上用 data3 的参考生成方式（robot_fk 上的 IK，LML 的站姿），seed 相同。目的是把"身体不同"和"参考生成方式不同"分开看。

**采集**：400 次重置，800 次伸手。没有摔倒，丢失 0.2 %，停不稳跳过 1.0 %，单手段停稳率 98.0 %。

**原始输出**（eval.txt 原文）：

```
xfer one-hand segments 638: kept 624 from 378 resets (1 fell / lost, 13 not settled at the end label); LocoManiLab val kept 3066
primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:
  first reach : n  314 (314 resets) xfer 6.989 cm [6.628, 7.636] / val 1.701 cm (n 1580) = 4.109x [3.877, 4.503] (val resampled too [3.811, 4.541])
  chained     : n  310 (310 resets) xfer 7.594 cm [7.116, 8.051] / val 5.209 cm (n 1486) = 1.458x [1.358, 1.545] (val resampled too [1.336, 1.562])
  all         : n  624 (378 resets) xfer 7.308 cm [6.967, 7.699] / val 3.077 cm (n 3066) = 2.375x [2.269, 2.526] (val resampled too [2.220, 2.545])
secondary: direct error median (cm), xfer / LocoManiLab in place ...; mean direct error vector ... angle ...; model / direct on xfer:
  first reach : direct 6.54 / 8.82 = 0.741x [0.680, 0.807]; mean vector xfer [0.71, -2.0, -0.94] (|2.32|), LML [-0.03, -0.81, -1.67] (|1.86|), angle 42.8 deg [29.9, 57.8]; model / direct 1.069
  chained     : direct 9.17 / 10.30 = 0.890x [0.790, 0.956]; mean vector xfer [1.42, -2.8, -1.88] (|3.66|), LML [-0.05, -2.19, -1.98] (|2.96|), angle 25.3 deg [14.5, 37.5]; model / direct 0.828
  all         : direct 7.59 / 9.53 = 0.797x [0.751, 0.854]; mean vector xfer [1.06, -2.4, -1.4] (|2.98|), LML [-0.04, -1.48, -1.82] (|2.35|), angle 29.8 deg [22.4, 38.0]; model / direct 0.963
training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer 60.7 % [57.9, 63.6] (val 81.1 %), joints out median 0; prediction error inside 7.162 cm (n 379), outside 7.599 cm (n 245); most often out: right_elbow 12.5, left_elbow 10.6, left_wrist_roll 7.2, right_shoulder_pitch 6.9, right_wrist_roll 6.7 (%)
ensemble disagreement (|member std| of the predicted hand point, median cm): xfer 4.874 [4.746, 4.947], val 2.506, ratio 1.94x
```

补充取法（末尾没停稳的也算）：合计 2.42 倍，结论不变。

**对照判据**
- 合计 2.38 倍 [2.27, 2.53]，大于 1.5 倍，落在"身体不一样"那一格，需要在 SIMPLE 里重采重训，另行登记。
- 分开看，第一次伸手 4.11 倍，接着上一次 1.46 倍 [1.36, 1.55]。
- 合计的倍数落在预测的 1.5–4 倍之内。
- 直接执行误差是 LML 的 0.80 倍，预测说会在 0.7 以下或 1.4 以上，这条没中：误差大小只差两成，平均误差方向差了 30°。
- LML 的模型在 SIMPLE 里几乎修不动：模型 / 直接执行合计 0.96，第一次伸手 1.07，比不修还差。

**先不下"重采"的结论**：要等诊断臂的结果。现在的差别里混着两样东西：身体不同，以及参考生成和站姿不同。站姿不同这一样，从"在训练范围里的比例只有 60.7 %"就能看出来，val 是 81.1 %。
- 如果诊断臂（SIMPLE 身体 + LML 的参考生成）回到 1.3 倍以内，说明身体基本一样，差别出在我们怎么生成命令。这样可以考虑让三个 benchmark 统一用同一套参考生成，不用重采。
- 如果诊断臂仍然大于 1.5 倍，就按判据在 SIMPLE 里重采。

### 诊断臂：ScaleBFM → SIMPLE，换成 data3 的参考生成和 LML 站姿（冻结前已写进 PLAN.txt，不进判据）

**eval.txt 原文**：

```
xfer scalebfm (lml planner, lml stance, delay 0 frames): /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/scalebfm_simple_lmlref; xfer segments kept: not fallen and settled at the end label (d3_eval)
  model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/models_s0/E3_ensemble_f1 (built on /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/data/all.npz)
  xfer one-hand segments 638: kept 619 from 375 resets (3 fell / lost, 16 not settled at the end label); LocoManiLab val kept 3066
  primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:
    first reach : n  315 (315 resets) xfer 6.774 cm [6.227, 7.330] / val 1.701 cm (n 1580) = 3.983x [3.662, 4.309] (val resampled too [3.586, 4.436])
    chained     : n  304 (304 resets) xfer 7.471 cm [7.084, 7.998] / val 5.209 cm (n 1486) = 1.434x [1.360, 1.504] (val resampled too [1.324, 1.548])
    all         : n  619 (375 resets) xfer 7.166 cm [6.892, 7.527] / val 3.077 cm (n 3066) = 2.329x [2.240, 2.451] (val resampled too [2.187, 2.478])
  secondary: direct error median (cm), xfer / LocoManiLab in place (kept, every split) [95 %]; mean direct error vector (heading frame, cm; left y mirrored) and the angle between them [95 %, xfer resampled]; model / direct on xfer:
    first reach : direct 6.90 / 8.82 = 0.782x [0.723, 0.836]; mean vector xfer [0.26, -2.31, -1.26] (|2.64|), LML [-0.03, -0.81, -1.67] (|1.86|), angle 36.1 deg [25.7, 47.0]; model / direct 0.981
    chained     : direct 8.83 / 10.30 = 0.857x [0.786, 0.949]; mean vector xfer [2.14, -2.68, -1.74] (|3.85|), LML [-0.05, -2.19, -1.98] (|2.96|), angle 35.9 deg [24.4, 49.0]; model / direct 0.846
    all         : direct 7.65 / 9.53 = 0.803x [0.754, 0.853]; mean vector xfer [1.18, -2.49, -1.49] (|3.14|), LML [-0.04, -1.48, -1.82] (|2.35|), angle 30.3 deg [22.8, 39.6]; model / direct 0.937
  training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer 65.9 % [62.8, 69.0] (val 81.1 %), joints out median 0; prediction error inside 6.939 cm (n 408), outside 7.513 cm (n 211); most often out: [('right_elbow_joint', 7.6), ('left_wrist_roll_joint', 7.1), ('right_wrist_roll_joint', 7.1), ('right_wrist_yaw_joint', 5.2), ('right_shoulder_pitch_joint', 5.0)] (%)
  ensemble disagreement (|member std| of the predicted hand point, median cm): xfer 2.991 [2.869, 3.158], val 2.506, ratio 1.19x
```

- 补充取法（不论末尾停没停稳都算）：3.98 / 1.46 / 2.37 倍，结论不变。
- CHECK PASSED，XFER CHECK PASSED。

**读法**：换了参考生成和站姿以后，预测误差和主臂几乎一样（合计 2.33 倍，主臂 2.38 倍）。同时起点更像训练数据了：在训练范围里的比例从 60.7 % 升到 65.9 %，集成分歧从 val 的 1.94 倍降到 1.19 倍。也就是说，模型更"自信"了，误差却照样大。所以差别不在规划器和站姿上。

剩下的差别是：身体的物理参数，包括关节 armature、阻尼、摩擦，以及 MJCF 的几何（躯干高 1 cm）；还有 ScaleBFM 在两边的驱动方式，SIMPLE 用 Motor 的 mode 4 关键点模式、钉住脚踝、里程计给根。诊断臂没有把这两样拆开，所以这里只能说"不是参考生成"，还不能说"是身体"。

## 结果：组合 2，SONIC → SIMPLE（2026-11-26）

**运行**：和组合 1 用同一份 asrun（冻结于 16:04:48Z）。主臂 16:14:00Z–16:26:18Z，诊断臂 16:26:18Z–16:38:05Z。各 400 次重置，16 个进程。

**实现**
- SONIC 走 SIMPLE 自己的接口：SonicMotor 下发 VR 三点目标（两手腕和头），腿由 SONIC 自带的 kPlanner 规划。参考骨盆低于 0.70 m 时才下蹲。
- 延迟用 LAG_MOVES 实测，中位数 3.5 帧，所以停稳窗口取 4 帧。LML 是 47 帧。
- SONIC 在 CPU 上跑。
- d_this 和 d_prev 都存了：主臂两者都是 0，诊断臂有 data3 的平移。

**一致性检查**
- 两臂的 CHECK 都通过；诊断臂 XFER CHECK 通过。
- 主臂 XFER CHECK 第 (2) 项没过：用 FK 从关节重算的手点和 W 比，最大差 0.0255 cm，阈值是 0.02 cm，56 个标签里有 2 个超了。
  - 原因：mj_step 之后 xpos 比 qpos 落后一个 5 ms 子步，而这几个标签上手腕还在转（|dq| 最大 0.23 rad/s）。把关节退回一个子步再算，56 个全部是 0.0000 cm。
  - KP 和 W 的差是 0.00000 cm。
  - 判定：不是记录错误。

**停稳超时和手速**（sx_speed.py，冻结之后写的只读分析）：

| 组合 | 命令前停不稳被跳过 | 段末没停稳 | 末尾 0.5 s 的规则速度，中位数 / p90 |
|---|---|---|---|
| SONIC·SIMPLE 主臂 | 42.1 % | 86.0 % | 4.23 / 11.26 cm/s |
| SONIC·SIMPLE 诊断臂 | 44.2 % | 89.4 % | 4.83 / 12.06 cm/s |
| SONIC·LML data3 | 1.6 % | 4.8 % | 0.79 / 1.53 cm/s |
| ScaleBFM·SIMPLE | 1.0 % | 2.9 % | 0.67 / 1.39 cm/s |

主臂上，骨盆、伸手的手、另一只手都在动（中位数 1.7 / 3.3 / 2.8 cm/s），不只是手在抖。

**主臂 eval.txt 原文**：

```
xfer sonic (native planner, native stance, delay 4 frames): /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/sonic_simple; xfer segments kept: not fallen and settled at the end label (d3_eval)
  model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic/models_s0/E3_ensemble_f1 (built on /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic/data/all.npz)
  xfer one-hand segments 367: kept 56 from 49 resets (7 fell / lost, 304 not settled at the end label); LocoManiLab val kept 2935
  primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:
    first reach : n   49 (49 resets) xfer 7.122 cm [5.891, 12.669] / val 2.860 cm (n 1557) = 2.490x [2.060, 4.430] (val resampled too [2.034, 4.438])
    chained     : n    7 (7 resets) xfer 14.307 cm [8.089, 18.227] / val 6.837 cm (n 1378) = 2.093x [1.183, 2.666] (val resampled too [1.153, 2.718])
    all         : n   56 (49 resets) xfer 7.589 cm [6.301, 13.131] / val 4.252 cm (n 2935) = 1.785x [1.483, 3.088] (val resampled too [1.474, 3.118])
  secondary: direct error median (cm), xfer / LocoManiLab in place (kept, every split) [95 %]; mean direct error vector (heading frame, cm; left y mirrored) and the angle between them [95 %, xfer resampled]; model / direct on xfer:
    first reach : direct 7.96 / 13.50 = 0.589x [0.477, 0.901]; mean vector xfer [-0.25, 0.36, 5.42] (|5.44|), LML [-3.58, -0.58, 8.12] (|8.89|), angle 22.4 deg [8.9, 41.5]; model / direct 0.895
    chained     : direct 14.21 / 15.58 = 0.912x [0.205, 1.224]; mean vector xfer [-0.71, -0.95, 7.64] (|7.73|), LML [-4.47, -1.62, 9.3] (|10.45|), angle 20.3 deg [4.2, 66.9]; model / direct 1.007
    all         : direct 8.21 / 14.52 = 0.565x [0.459, 0.897]; mean vector xfer [-0.31, 0.2, 5.7] (|5.71|), LML [-4.0, -1.07, 8.67] (|9.61|), angle 23.2 deg [11.3, 41.6]; model / direct 0.925
  training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer 0.0 % [0.0, 0.0] (val 82.8 %), joints out median 1; prediction error inside None cm (n 0), outside 7.589 cm (n 56); most often out: [('right_hip_pitch_joint', 96.4), ('waist_roll_joint', 3.6), ('left_shoulder_roll_joint', 1.8), ('left_knee_joint', 1.8), ('left_elbow_joint', 1.8)] (%)
  ensemble disagreement (|member std| of the predicted hand point, median cm): xfer 4.938 [4.436, 5.447], val 3.132, ratio 1.58x
```

**补充取法 eval_ended.txt**（执行 agent 摘出的行）：

```
  xfer one-hand segments 367: kept 360 from 319 resets (7 fell / lost, 304 not settled at the end label); LocoManiLab val kept 2935
    first reach : n  313 (313 resets) xfer 11.704 cm [10.493, 12.671] / val 2.860 cm (n 1557) = 4.093x [3.652, 4.430] (val resampled too [3.577, 4.493])
    chained     : n   47 (47 resets) xfer 15.111 cm [13.555, 16.970] / val 6.837 cm (n 1378) = 2.210x [2.093, 2.519] (val resampled too [1.995, 2.542])
    all         : n  360 (319 resets) xfer 12.201 cm [10.935, 13.585] / val 4.252 cm (n 2935) = 2.869x [2.570, 3.171] (val resampled too [2.527, 3.181])
    first reach : direct 12.93 / 13.50 = 0.958x [0.884, 1.020]; ... angle 20.6 deg [12.7, 29.0]; model / direct 0.905
    chained     : direct 16.04 / 15.58 = 1.029x [0.812, 1.127]; ... angle 8.2 deg [2.8, 32.4]; model / direct 0.942
    all         : direct 13.09 / 14.52 = 0.902x [0.846, 0.964]; ... angle 18.5 deg [11.3, 26.6]; model / direct 0.932
  training range: xfer 2.5 % [1.1, 4.3] (val 82.8 %); most often out: right_hip_pitch_joint 93.1 %
  ensemble disagreement: xfer 5.261 [5.174, 5.480], val 3.132, ratio 1.68x
```

**诊断臂**（换成 data3 的参考生成和 LML 站姿，不进判据）：

| 取法 | 合计 | 第一次 | 接着上一次 | 直接执行比值 | 模型/直接 |
|---|---|---|---|---|---|
| 主取法 | 2.455 倍 [1.561, 3.015]，n 37 | 3.638 倍 | 1.923 倍（n 3） | 0.752 | 0.956 |
| 补充取法 | 3.264 倍 | 4.746 倍 | 2.774 倍 | 0.941 | 1.016 |

训练范围内的比例：主取法 0.0 %，补充取法 0.9 %。

**对照判据**
- 主取法合计 1.79 倍 [1.48, 3.09]，超过 1.5，落在"身体不一样"那一格。但这只是不摔的段里能停稳的 15.6 %，有选择偏差。
- 补充取法 2.87 倍 [2.57, 3.17]，同一格。
- 预测 2–5 倍：主取法略低于下限，补充取法在范围内。
- 模型几乎修不动：模型/直接是 0.93。
- 起点 100 % 落在训练范围外，几乎全是 right_hip_pitch 一个关节出界：SIMPLE 里 kPlanner 让机器人站得更直，中位数 +0.04 rad，LML 是 −0.22 rad。

## ScaleBFM 和 SONIC 在 LML 与 SIMPLE 里的设置对比（只读查代码，执行 agent 整理，附出处）

**两边相同**
- PD 的 kp/kd。
- 关节 armature / damping / frictionloss：两边都取 MJCF 的 joint class，腿、腰、臂是 0.01 / 0.05 / 0.2，腕 pitch 和 yaw 是 0.01 / 0.05 / 0.1。policy.yaml 里写的 dynamics 两边都没用上。
- 力矩上限：髋 88、膝 139、踝 50、腰 yaw 88、腰 roll/pitch 50、肩/肘/腕 roll 25、腕 pitch/yaw 5 N·m。
- 几何、质量、惯量：只有 right_wrist_yaw_link 的质心 z 差约 1 mm。
- 策略频率：都是 50 Hz。

**两边不同**

| 项 | ScaleBFM·LML | ScaleBFM·SIMPLE | SONIC·LML | SONIC·SIMPLE |
|---|---|---|---|---|
| 控制模式 | mode 7，14 个关键点全跟 | mode 4，跟骨盆、两踝、两腕 | g1 编码器，29 个关节，10 帧未来 | VR 三点 + kPlanner 管腿 |
| 根位置输入 | 无（reference_forcing，策略认为骨盆就在参考骨盆处） | 腿里程计 | 无 | 里程计位置 + 真值姿态 |
| 踝钉住 | 否 | 每次下命令时重新钉住 | 否 | 不适用 |
| 参考延迟 | 0.14 s | 0，未来帧按速度外推 | 0.94 s | 0 |
| 仿真步长和 PD | 2 ms × 10，PD 500 Hz，Euler，impratio 20 | 5 ms × 4，PD 200 Hz，implicitfast，impratio 10，noslip 2 | 同 ScaleBFM·LML | 同 ScaleBFM·SIMPLE |

**更正前面的说法**："身体不一样"是判据那一格的名字，不是原因。查下来，两边的物理身体（MJCF 几何、质量、关节参数、力矩上限）基本相同，唯一的物理层差别是求解器设置。ScaleBFM 和 SONIC 在 SIMPLE 里迁移不过来，主要原因是两边驱动控制器的方式不一样：控制模式、根位置的来源、踝钉住、延迟；SONIC 在 SIMPLE 里还多了 kPlanner 管腿。诊断臂只换了参考生成和站姿，没换驱动方式，所以和主臂结果一样。

**由此看 HEFT → SIMPLE**：HEFT 在 SIMPLE 里用的就是 g1_backend 的观测和 ONNX，和 LML 里的驱动方式相同，身体也相同，只差求解器设置和参考延迟（0 对 0.16 s，只影响时间，不影响停稳后的位置）。所以 HEFT → SIMPLE 可以直接检验："控制器和驱动方式都不变时，模型能不能跨 benchmark 迁移"。

## 修订 2：HumanoidArena 三个组合的做法（2026-11-26，HA 还没开跑）

预测和判据不变。

1. **规模不减。** 每个组合 400 次重置、2 个 HA 进程，每张卡一个进程，估计每个组合用时 1.6–1.9 h。
2. **SONIC 关掉渲染。** 我们直接喂自己的 40 维帧，用不到相机。同一组 2 次重置，开渲染跑 2 次、关渲染跑 1 次，三次的步数和末端误差逐位相同。另外有一次更早的开渲染运行，结果不同（749 步）：PhysX 在不同进程之间偶尔不能复现，原因没查到，作为注意事项记下。
3. **桌子保留，场景不改。** ppbox 场景里，桌子近边在骨盆前方 0.96 m，台面高 0.75 m。
   - 用 ppbox 自带的 contact_forces 传感器，逐段记录除脚以外的刚体有没有碰到场景。正式运行前，先在站姿里确认基线为 0。
   - 主结果按登记的取法算，不剔除碰到场景的段；另外报碰到场景的比例，以及剔除这些段之后的敏感性数字。
   - 按 SIMPLE 两个组合的抽样估计，手臂越过 0.90 m 的段只占 0.4–0.8 %。ScaleBFM 在 HA 里会往前迈步，实际比例可能更高。
4. **SONIC 的接法。**
   - 参考帧用 data3 的 IK 生成，按 HA runtime v3 的读法转成 40 维帧。runtime 积分出来的根和我发的帧逐步相差 0。
   - 停稳延迟按 LAG_MOVES 实测，中位数 6 帧。
   - 站姿用 SONIC 在 HA 里的默认姿态，不用 LML 站姿，这一点和 data3 不同。原因是用 LML 站姿时，SONIC 在 HA 里停不住：xy 漂 8 cm，冒烟里一段都没采到。
5. **ScaleBFM 和 HEFT 的接法。** 和 data3 喂 LML 的方式一样，世界系 qpos 帧经 motion buffer 推入，延迟取 buffer 的 delay_s，站姿用 LML 的。驱动代码用 ha_controllers 的 `ha_g1ctrl.py`（md5 72a7bc22）。
6. **运动学一致。** HA 的运动学和 robot_fk 相差约 1e-7 m。

## 预测修订：HumanoidArena 三个组合（2026-11-25 约 17:15Z 写下，即北京时间 11-26 01:15）

**写下时的状态**：ScaleBFM·HA 已经开跑，但还没看到任何结果；HEFT·HA 和 SONIC·HA 还没开跑。原来登记的预测和判据保留不动，这里只补充一条新的预测。

**这条预测怎么来的**
- SIMPLE 的三个组合显示，两边物理身体相同、驱动方式也相同时，HEFT 能迁移（组合 6）。
- HA 的情况不同：驱动方式和 LML 相同（ScaleBFM 和 HEFT 都走 motion buffer），但物理引擎换成了 PhysX，身体也不同。接入冒烟时查到：
  - 整机 38.6 kg，LML 是 36.2 kg；
  - 踝力矩上限 40 N·m，LML 是 50 N·m；
  - armature、阻尼、摩擦都是 0。
- 所以如果"模型对应的是控制器、驱动方式和物理身体"成立，在 HA 里就不该迁移。

**预测（预测误差 / LML val，合计）**
- ScaleBFM·HA：> 1.5 倍。冒烟里 30 cm 前伸时它往前倒、迈步。
- HEFT·HA：1.3–3 倍。冒烟里前伸误差 3.7 cm，LML 是 1.3 cm。
- SONIC·HA：> 1.5 倍。HA 的 40 维帧接口和站姿都和 LML 不同。

**什么结果会推翻这条解释**：ScaleBFM 或 HEFT 在 HA 里落进 ≤ 1.3 倍那一格。那就说明连物理引擎和身体参数的差别都不要紧，"驱动方式"之外的解释都要重新考虑。
