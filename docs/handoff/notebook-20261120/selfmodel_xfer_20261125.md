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

## 结果：组合 6，HEFT → SIMPLE（2026-11-26）

**冻结**：2026-11-25T16:49:19Z，`asrun_heft/`。从 4090 的 `loco_heft` 整份拷出用到的代码（772 个文件），再加 3 个 HEFT 权重文件。sx_batch.sh 开跑前先校验 MD5.check。运行计划冻结前已写进 PLAN_heft.txt。

| 文件 | md5 |
|---|---|
| sx_collect.py | 08e78066（加了 heft 条目，以及 HEFT 的 xy 偏移记录） |
| sx_eval.py | 81f61381（未变） |
| sx_batch.sh | 3801ec2a |
| sx_freeze_heft.sh | 8e167d0f |
| sx_speed.py | 3db13e7a |
| d3_collect.py | 581fadf5 |
| backends/heft | motor aefef4a1、runtime c690e646、obs 227e8eca、gait 83d99421 |
| HEFT 权重 | policy.onnx a1e2e40c，和 g1_backend 里的相同 |

和组合 1、2 的 repo 副本相比，只有 paths.py、env.py、capability_probe.py 三个文件不同，改动都是新加的 HEFT 分支。

**实现**
- HEFT 在进程内跑，用 g1_backend 的观测和 ONNX。设置：
  - REACH_HEFT_DELAY=0；
  - reset_pose 用 lml；
  - 关节参数用 MJCF 自带的，PD 用 HEFT 自己的。
- 延迟：LAG_MOVES 实测中位数 0 帧，停稳窗口按 0 帧取。
- 两臂的参考生成和站姿：
  - 主臂用 SIMPLE 的 IK 和 HEFT 复位后的站姿。这个站姿和 LML 默认关节最多差 3.2°。
  - 诊断臂用 data3 的 IK 和 LML 站姿。
  - 两臂和组合 1、2 用同一套种子，命令逐条配对。
- HeftMotor 会把参考的 xy 跳变吸收成一个偏移。偏移在第一帧之后的变化：中位数 0.02 cm，最大 0.45 cm。
- 两臂都是 CHECK PASSED、XFER CHECK PASSED。FK 重算手点，最大差 0.0046 cm。

**主臂 eval.txt 原文**：

```
xfer heft (native planner, native stance, delay 0 frames): /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/heft_simple; xfer segments kept: not fallen and settled at the end label (d3_eval)
  model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/models_s0/E3_ensemble_f1 (built on /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/data/all.npz)
  xfer one-hand segments 481: kept 258 from 179 resets (190 fell / lost, 33 not settled at the end label); LocoManiLab val kept 1492
  primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:
    first reach : n  168 (168 resets) xfer 2.990 cm [2.566, 3.336] / val 2.964 cm (n 1022) = 1.009x [0.858, 1.133] (val resampled too [0.840, 1.152])
    chained     : n   90 (90 resets) xfer 7.516 cm [6.057, 8.900] / val 6.989 cm (n 470) = 1.075x [0.867, 1.271] (val resampled too [0.854, 1.334])
    all         : n  258 (179 resets) xfer 4.227 cm [3.489, 4.880] / val 3.993 cm (n 1492) = 1.059x [0.862, 1.222] (val resampled too [0.880, 1.249])
  secondary: direct error median (cm), xfer / LocoManiLab in place (kept, every split) [95 %]; mean direct error vector (heading frame, cm; left y mirrored) and the angle between them [95 %, xfer resampled]; model / direct on xfer:
    first reach : direct 8.75 / 10.35 = 0.845x [0.768, 0.984]; mean vector xfer [2.89, -3.17, -2.67] (|5.05|), LML [2.2, -4.89, -0.98] (|5.45|), angle 27.3 deg [16.5, 39.4]; model / direct 0.342
    chained     : direct 11.39 / 10.43 = 1.092x [0.900, 1.356]; mean vector xfer [1.02, -2.72, -0.48] (|2.94|), LML [-0.47, -4.55, -0.68] (|4.63|), angle 26.1 deg [5.5, 73.9]; model / direct 0.660
    all         : direct 9.62 / 10.38 = 0.927x [0.827, 1.038]; mean vector xfer [2.24, -3.01, -1.91] (|4.21|), LML [1.32, -4.78, -0.88] (|5.04|), angle 26.2 deg [14.4, 40.6]; model / direct 0.440
  training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer 85.7 % [81.9, 89.4] (val 87.2 %), joints out median 0; prediction error inside 3.786 cm (n 221), outside 8.364 cm (n 37); most often out: [('left_shoulder_pitch_joint', 2.3), ('right_wrist_roll_joint', 2.3), ('left_shoulder_roll_joint', 1.6), ('right_hip_pitch_joint', 1.6), ('right_wrist_pitch_joint', 1.6)] (%)
  ensemble disagreement (|member std| of the predicted hand point, median cm): xfer 3.014 [2.805, 3.173], val 2.915, ratio 1.03x
```

**补充取法 eval_ended.txt**（执行 agent 摘出的行）：

```
  xfer one-hand segments 481: kept 291 from 202 resets (190 fell / lost, 33 not settled at the end label)
    first reach : n  191 (191 resets) xfer 3.277 cm [2.837, 3.979] / val 2.964 cm (n 1022) = 1.106x [0.957, 1.310] (val resampled too [0.966, 1.362])
    chained     : n  100 (100 resets) xfer 8.373 cm [6.784, 10.071] / val 6.989 cm (n 470) = 1.198x [0.971, 1.377] (val resampled too [0.922, 1.450])
    all         : n  291 (202 resets) xfer 4.880 cm [4.152, 5.536] / val 3.993 cm (n 1492) = 1.222x [1.036, 1.378] (val resampled too [1.029, 1.392])
    all direct 10.75 / 10.38 = 1.036x [0.925, 1.172]; angle 25.1 deg [12.5, 38.8]; model / direct 0.454
  training range: xfer 85.9 % [82.3, 89.5] (val 87.2 %); ensemble disagreement ratio 1.07x
```

**诊断臂**（data3 的 IK 加 LML 站姿，不进判据）
- 主取法合计 1.170 倍 [1.052, 1.319]：第一次 1.283 倍，接着上一次 0.987 倍。
- 补充取法 1.296 倍。
- 直接执行比值 1.018 倍，方向夹角 13.6°，模型/直接 0.442。
- 训练范围内的比例 86.3 %。

**停稳和丢失**

| 组合 | 命令前停不稳被跳过 | 段末没停稳 | 末尾规则速度 中位数 / p90 |
|---|---|---|---|
| HEFT·SIMPLE 主臂 | 4.6 % | 12.7 % | 0.79 / 3.07 cm/s |
| HEFT·LML data3（原地，同一口径） | 4.3 % | 11.5 % | 0.86 / 2.21 cm/s |

原地重置里，丢失 62.5 %、摔倒 1.5 %；LML data3 原地是丢失 59.3 %、摔倒 1.2 %。丢失按 d3 的 0.5 m 规则判，这么高是 HEFT 本身的特点。两边都只保留幸存下来的段。

**对照判据**
- 主取法合计 1.06 倍 [0.86, 1.22]：点值 ≤ 1.3，区间上限 ≤ 1.5，落在"身体基本一样"那一格，可以直接拿 LML 的模型做 SIMPLE 的对比。补充取法（1.22 倍 [1.04, 1.38]）和诊断臂（1.17 倍）也落在这一格。
- 登记的预测是 2–6 倍，没中，实际低得多。
- 直接执行比值预测没中：实际 0.93 倍 [0.83, 1.04]，误差大小基本相同，平均方向差 26°。
- 模型修正有效：模型/直接合计 0.44，第一次伸手 0.34。

**读法**：HEFT 在两边吃的是同一种参考、同一套观测和 ONNX，关节参数也一样。剩下的差别只有求解器设置和参考延迟。在这种条件下，模型能迁移。再结合组合 1、2 和设置对比表来看：模型跟的是"控制器加上驱动方式"，不是 benchmark。ScaleBFM 和 SONIC 在 SIMPLE 里迁移不过来，是因为驱动方式不同。这个结论目前只靠一个控制器得出，求解器和延迟两项也没有单独拆开验证。

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

## 独立复核：组合 1、2、6（2026-11-26，独立 agent，只读；以下是要点，原意照录）

**总体**：数字都对得上，只有一处笔误。"HEFT 能迁移"很稳。"ScaleBFM 和 SONIC 失败是因为驱动方式"还没有被单独验证，因为驱动方式和求解器是一起变的。"模型对应控制器加驱动方式，不是 benchmark"目前只有一个成功的例子，而且两边是同一具 MJCF 身体，推不到这么远。

**1. 数字核对**
- 在 4090 上重新载入 E3 模型，对 val 和 xfer 重新预测：
  - 组合 6：1.059 倍，val 也重抽样时区间 [0.877, 1.255]；
  - 组合 1：2.375 倍 [2.22, 2.54]；
  - 组合 2：1.785 倍；
  - 模型/直接：0.963 / 0.925 / 0.440。
- 停稳表、丢失率、诊断臂、补充取法逐项对上。
- 笔误：ScaleBFM 诊断臂补充取法的第一次伸手应为 4.054，条目写成了 3.98。
- 漏报：组合 2 主取法的直接执行比值是 0.565 < 0.7，登记的这条预测其实中了；补充取法 0.90，没中。

**2. 四类陷阱**
- **分母。** 把 LML val 限定到原地重置以后：
  - ScaleBFM 2.375 → 2.464，第一次伸手 4.11 → 4.48；
  - SONIC 1.785 → 1.819；
  - HEFT 1.059 → 1.087 [0.90, 1.29]，第一次伸手 1.009 → 1.092。

  先走再伸的段更难，混进分母会压低倍数，但偏差只有 1–9 %，判据落格都不变。唯一要注意的是：HEFT 诊断臂第一次伸手改用原地分母后是 1.39 [1.22, 1.71]，落进 1.3–1.5 那一格。
- **幸存者偏差。** 两边的筛法是同一段代码。HEFT：
  - SIMPLE 单手段丢失 39.5 %，LML 原地 33.3 %；
  - 按站姿比例重新加权后是 1.064；
  - 把多丢的约 30 段按最大误差补回去，是 1.22，用原地分母是 1.25，仍然不超过 1.3；
  - 在 HEFT 留下来的那批命令上算 ScaleBFM，是 2.02 倍（全量 2.38），说明筛选解释不了两者的差别。

  SONIC：停稳的那 15 % 是容易的命令（HEFT 在这批命令上只有 0.73 倍），筛法让 SONIC 看起来更好而不是更差，失败的结论是稳的。但主取法区间下限 1.46–1.48，样本只有 56 段，而且 86 % 的段打标签时手还在动，这些段的落点本身就不明确。
- **驱动方式的证据。** 在远端把两边的模型实际建出来读数值，抽查 7 项，全部属实：
  - ScaleBFM 在 LML 用 mode 7 加 reference_forcing，在 SIMPLE 用 mode 4 加踝钉住；
  - 关节参数两边都是 0.01 / 0.05 / 0.2，腕 0.1；
  - 力矩上限一致；
  - kp/kd 在 29 个关节、三个控制器上完全相同；
  - 44 个刚体里只有 right_wrist_yaw_link 的质心差 0.95 mm；
  - 求解器设置不同；
  - HEFT 的 ONNX 和 SONIC 的编码器、解码器，两边 md5 相同。

  表里没列出来的两点：
  - ScaleBFM 两边不是同一份实现。LML 用 g1_backend 导出的 ONNX，SIMPLE 用 humanoid_transformer_m_22200.pt 加我们自己的运行时，权重是否相同没核实。
  - SONIC 和 ScaleBFM 训练时的步长就是 5 ms × 4（200 Hz），所以它们在 SIMPLE 里跑的反而是训练步长，在 LML 里不是。"求解器对这两个控制器影响更大"这个解释排除不了。
- **"HEFT 误差模式更简单"不成立。** 用线性基线（命令加起点状态）在 LML 原地 val 上只能降 23–26 %，三个控制器差不多；E3 能降 64–70 %。
- **支持驱动方式的机制证据。** SONIC 在 SIMPLE 里根本不跟关节参考，腿由 kPlanner 定，96 % 的起点在 right_hip_pitch 上出界；ScaleBFM 在起点落在训练范围内的段上照样失败（7.16 cm 对 7.60 cm）。
- **预测有没有在看结果之后被改。** 预测和判据从登记起没有删过一行。修订 1 在任何运行之前；修订 2 在 HEFT 运行之前。"驱动方式"这个解释是看了组合 1、2 之后才提出的，HEFT 没有据此登记新的数值预测，所以 HEFT 的成功只能算一次定性印证，不是数值预测命中。

**3. 可能推翻核心论断的实验**
- **实验 A（最关键）：在 SIMPLE 身体里对 ScaleBFM 做 2×2，驱动方式 × 求解器。**
  - 统一用诊断臂的设置，400 次重置。
  - 四个格：
    - a：SIMPLE 驱动 + SIMPLE 求解器，即现有诊断臂，2.33 倍；
    - b：SIMPLE 驱动 + LML 求解器（2 ms × 10、Euler、impratio 20、noslip 0）；
    - c：LML 驱动（g1_backend 的 ScaleBFM，照 HEFT 那样接）+ SIMPLE 求解器；
    - d：LML 驱动 + LML 求解器，闭合检查，应该在 1.0 附近。
  - 怎么算推翻：b ≤ 1.3，或者 c > 1.5。d 如果回不到 1 附近，说明还有表外的差别，比如两边 ScaleBFM 的实现或权重不同。
- **实验 B：在 LML 宿主里只换成 SIMPLE 的求解器**，ScaleBFM 和 HEFT 各跑 400 次原地重置。如果 ScaleBFM 超过 1.5 倍而 HEFT 不超过 1.3 倍，就推翻驱动方式的解释。另外，LML 的 iTHOR 任务场景 XML 里写的是 implicitfast、impratio 10、noslip 4，和 data3 用的 EmptyEnv 不同。如果这些设置真的生效，这个问题在 LML 自己的任务里也存在。
- **实验 C：用 HA 的 ScaleBFM 和 HEFT 组合来检验第三条论断**，并在正式开跑前把预测登记下来。

**4. 判定**

| 结论 | 判定 | 理由 |
|---|---|---|
| ScaleBFM 和 SONIC 失败，原因是驱动方式，不是物理身体 | 证据不足 | "不是物理身体"属实，但驱动方式和求解器是一起变的，没有单独拆开；而且这两个控制器按 SIMPLE 的步长训练，求解器的影响排除不了 |
| HEFT 两边驱动方式相同，所以能迁移 | 支持 | 分母、取法、加权、最坏情况补回，倍数都在 1.06–1.25；仅限原地伸手、约 40 % 幸存段、MuJoCo 到 MuJoCo |
| 模型对应"控制器加驱动方式"，不是 benchmark | 证据不足 | 只有一个成功例子，而且是同一具 MJCF 身体、同一个物理引擎 |

## 复核后的更正（2026-11-26）

1. ScaleBFM 诊断臂补充取法的第一次伸手是 4.054 倍，不是 3.98。
2. 组合 2 主取法的直接执行比值是 0.565，登记的直接执行预测在这一格算中；补充取法 0.90，没中。
3. 结论收窄。
   - 能确定的：SIMPLE 和 LML 的 G1 物理身体相同。HEFT 两边驱动方式相同，模型能迁移（1.06–1.25 倍，原地伸手）。
   - 还不能确定的："ScaleBFM 和 SONIC 失败是因为驱动方式"，因为驱动方式和求解器设置是一起变的。
   - "模型对应控制器加驱动方式"只算一个有一例支持的假设。
4. 给用户的 SIMPLE 方案二（把 ScaleBFM 和 SONIC 改成 LML 的接法）能不能行，要看实验 A 的 c 格。这一点要告诉用户。
5. HA 组合的预测已经补登记（7826b351）。写下时 ScaleBFM·HA 已经开跑，但还没看到结果。

## 结果：组合 4，ScaleBFM → HumanoidArena（2026-11-26）

**冻结**：2026-11-25T17:03:43Z，`asrun_ha/`。MD5.check 共 631 行，sx_ha_batch.sh 开跑前校验通过。运行计划冻结前写进了 PLAN_ha.txt。冻结前用 dev 拷贝跑过 4 次重置的流水线测试。

| 文件 | md5 |
|---|---|
| sx_collect.py | cd290c2e（加了可选的接触字段；HA 的身体 FK 用 robot_fk；没有 motor 的宿主跳过 HEFT 偏移记录；SIMPLE 逻辑没变） |
| sx_eval.py | 81f61381（未变） |
| sx_ha.py / sx_ha.sh / sx_ha_batch.sh / sx_freeze_ha.sh | ba8b4f7f / 6f8234c0 / be93a4f2 / d51f4026 |
| sx_speed.py / d3_collect.py | 3db13e7a / 581fadf5 |
| ha_sim.py / ha_g1ctrl.py | 1592d3c2 / 72a7bc22 |
| 外部文件 | HA action_provider_sonic e5e058f9、runtime_v3 b25b8f8b、sim_eval_vla a5301545、pp_box_sonic_test.yaml f1e47d9d、robot_configs ed9957ea；SONIC encoder a21244b1、decoder 1d4391ad；g1_backend 25d262a（ScaleBFM onnx d32d8a0c / yaml 9d914d3c，HEFT onnx a1e2e40c / yaml 9b33fe1d）；repo 拷贝来自 bc426d5e |

**实现**：按修订 2 的接法。
- 参考用 data3 的 IK 和 LML 站姿，世界系 qpos 帧经 motion buffer 推入；停稳延迟按 buffer 取 7 帧。
- 场景 ppbox，桌子保留。
- 400 次重置，2 个进程，用卡 1、2（PLAN 写的是 2、3）。驱动阶段耗时 1 h 26 min。
- 站姿阶段，除脚以外的接触力全部是 0 N。
- CHECK PASSED，XFER CHECK PASSED。robot_fk 重算手点，最大差 0.0002 cm。

**eval.txt 原文**：

```
xfer scalebfm (lml planner, lml stance, delay 7 frames): /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/scalebfm_ha; xfer segments kept: not fallen and settled at the end label (d3_eval)
  model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/models_s0/E3_ensemble_f1 (built on /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/data/all.npz)
  xfer one-hand segments 487: kept 276 from 192 resets (118 fell / lost, 93 not settled at the end label); LocoManiLab val kept 3066
  primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:
    first reach : n  179 (179 resets) xfer 21.694 cm [20.659, 22.587] / val 1.701 cm (n 1580) = 12.755x [12.350, 13.306] (val resampled too [11.870, 13.744])
    chained     : n   97 (97 resets) xfer 16.575 cm [15.264, 18.318] / val 5.209 cm (n 1486) = 3.182x [2.853, 3.517] (val resampled too [2.878, 3.567])
    all         : n  276 (192 resets) xfer 20.359 cm [19.459, 21.213] / val 3.077 cm (n 3066) = 6.617x [6.346, 6.950] (val resampled too [6.225, 7.043])
  secondary: direct error median (cm), xfer / LocoManiLab in place (kept, every split) [95 %]; mean direct error vector (heading frame, cm; left y mirrored) and the angle between them [95 %, xfer resampled]; model / direct on xfer:
    first reach : direct 10.84 / 8.82 = 1.229x [1.104, 1.387]; mean vector xfer [2.54, -1.48, -5.62] (|6.35|), LML [-0.03, -0.81, -1.67] (|1.86|), angle 26.9 deg [15.1, 38.4]; model / direct 2.001
    chained     : direct 13.54 / 10.30 = 1.315x [1.160, 1.403]; mean vector xfer [2.49, -0.81, -8.12] (|8.53|), LML [-0.05, -2.19, -1.98] (|2.96|), angle 45.2 deg [32.7, 58.0]; model / direct 1.224
    all         : direct 12.02 / 9.53 = 1.262x [1.133, 1.351]; mean vector xfer [2.52, -1.25, -6.5] (|7.08|), LML [-0.04, -1.48, -1.82] (|2.35|), angle 35.2 deg [26.4, 45.3]; model / direct 1.694
  training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer 77.5 % [73.8, 81.5] (val 81.1 %), joints out median 0; prediction error inside 21.011 cm (n 214), outside 17.764 cm (n 62); most often out: [('right_ankle_roll_joint', 5.1), ('left_ankle_roll_joint', 5.1), ('left_hip_roll_joint', 4.7), ('right_hip_pitch_joint', 4.3), ('right_ankle_pitch_joint', 4.0)] (%)
  ensemble disagreement (|member std| of the predicted hand point, median cm): xfer 31.395 [30.981, 31.547], val 2.506, ratio 12.53x
```

**补充取法**（eval_ended.txt，执行 agent 摘出的数）
- 保留 369 段，来自 264 次重置。
- 第一次伸手 14.088 倍 [13.318, 15.428]，接着上一次 3.400 倍 [3.059, 3.761]，合计 7.151 倍 [6.881, 7.600]。
- 直接执行 1.471 倍，方向夹角 57.3°，模型/直接 1.570。
- 训练范围内 79.7 %，集成分歧 12.51 倍。

**停稳、手速、摔倒**（sx_speed，同一口径，只算原地重置）

| | HA | LML data3 ScaleBFM |
|---|---|---|
| 下命令前停不稳、被跳过 | 12.8 %（90/702） | 0.9 %（360/39654） |
| 段末没停稳 | 27.9 %（151/541） | 3.0 %（1357/45998） |
| 末尾 0.5 s 规则速度 中位数 / p90 | 1.19 / 19.06 cm/s | 0.79 / 1.46 cm/s |
| 摔倒 / 丢失 | 13.5 % / 25.8 % | 1.0 % / 3.0 % |

**碰到场景**（接触力 > 1 N，不算脚）
- 按计划，碰到任何刚体都算：主取法保留段里有 12.3 % 碰到。剔除后合计 6.617 倍，第一次伸手 12.84 倍，接着上一次 3.02 倍。
- 事后加的一行，只算桌子和纸箱（刚体高度 ≥ 0.2 m）：碰到的占 3.6 %，剔除后 6.533 倍。
- 碰到的刚体大多是手指碰地面。两种剔除都不改结论。

**事后诊断：HA 里两个踝 roll 关节的速度读数不对**（原写"两个左臂关节"，关节名查错了，见组合 5 的更正）（看到结果以后做的，不进判据；sx_dq.py md5 3b7e2ed8）
- 关节 17、18（按 ScaleBFM 的关节顺序是左、右踝 roll；原写成左肩 yaw 和左肘）的 joint_vel 读数，和它们自己位置的 10 Hz 差分几乎不相关：相关系数 0.08 和 0.01。其余 27 个关节在 0.57–0.98 之间。
- 每次第一次伸手下命令时，这两个关节读到 +1.70 / −1.90 rad/s，可前 0.4 s 里位置只变了 1e-4 rad。
- 全部行的 p1/p99：HA 是 ±2.7 rad/s，LML 是 ±0.2 rad/s。
- 执行 agent 推测（未验证）：200 Hz 物理步长下这两个关节按 2 个子步为周期颤动，50 Hz 读数总落在同一相位。
- 同一个模型、同一批段，只换掉 x0 里的 dq：

| x0 里的 dq | HA 第一次 / 接着上一次 / 合计 | 集成分歧 | LML val 合计 | SIMPLE 诊断臂合计 |
|---|---|---|---|---|
| 原样 | 12.76 / 3.18 / 6.62 倍 | 31.4 cm | 3.08 cm | 2.33 倍 |
| 只把关节 17、18 置 0 | 6.04 / 2.35 / 3.57 倍 | 5.5 cm | 3.12 cm | 2.30 倍 |
| 全部置 0 | 4.02 / 2.06 / 2.79 倍 | 3.6 cm | 3.13 cm | 2.42 倍 |
| 改用 q 的 10 Hz 差分 | 4.72 / 1.92 / 2.68 倍 | 3.8 cm | 3.13 cm | 2.30 倍 |

- 读法：
  - 换掉 dq 以后，LML val 和 SIMPLE 几乎不变，HA 从 6.6 倍降到约 2.7 倍。所以 6.6 倍里有一大半来自这两个关节的读数。
  - 剩下的约 2.7 倍仍然 > 1.5。这一部分对应的是 ScaleBFM 在 HA 里更晃、更常迈步，误差方向也偏了 35°。
- ScaleBFM 自己的观测里也有这两个关节的速度，它在 HA 里摔得多，可能部分也是因为这个读数。这一点没验证。

**对照判据**
- 登记的主结果：合计 6.62 倍 [6.35, 6.95]，大于 1.5，落在"身体不一样，要重采"那一格。补充取法 7.15 倍，同一格。
- 原登记的预测 2–6 倍：点值略高于上限，没中。
- 预测修订（7826b351，写下时组合 4 已开跑、还没看到结果）的预测是 > 1.5 倍：中。
- 直接执行比值的预测（< 0.7 或 > 1.4）：主取法 1.26 倍，没中；补充取法 1.47 倍，中。
- 模型/直接 1.69：在 HA 里直接用 LML 的模型，比不修正还差。
- 集成分歧是 val 的 12.5 倍。模型自己"知道"这些起点它没见过，这一点可以用作上线前的检查。
- 扣掉读数问题后的约 2.7 倍是事后数字，不能拿来对判据。

**和计划不一样的地方**：
1. 用的卡是 1、2，PLAN 里写的是 2、3。
2. "只算桌子和纸箱"那一行是看到结果后加的。sx_touch.py 第一版（5aad8213）在任何 HA 结果之前就写好了，现在的版本是 0c34d09c，默认算法不变。
3. 速度读数诊断完全是事后做的。

## 结果：组合 5，HEFT → HumanoidArena（2026-11-26）

**更正（组合 4 的诊断）**：关节 17、18 的名字查错了。S 里的关节按 ScaleBFM 的顺序排（IsaacLab 的交错顺序），当时误按 MuJoCo 顺序查了名字。
- 17、18 是左、右踝 roll，不是左肩 yaw 和左肘。
- 同样，关节 10 是右膝，13、14 是左、右踝 pitch。
- 所有计算都按下标做，LML 和 HA 两边顺序一致，所以数字和结论都不受影响。

**冻结**：和组合 4 用同一份 asrun_ha（2026-11-25T17:03:43Z），开跑前 MD5.check 校验通过。
- 参考延迟按 buffer 取 8 帧（0.16 s），用 LML 站姿。
- 卡 1、2，400 次重置，2 个进程，驱动阶段 1 h 17 min。
- 站姿阶段的接触基线是 0。
- CHECK PASSED，XFER CHECK PASSED；robot_fk 重算手点，最大差 0.0002 cm。

**eval.txt 原文**：

```
xfer heft (lml planner, lml stance, delay 8 frames): /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/heft_ha; xfer segments kept: not fallen and settled at the end label (d3_eval)
  model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/models_s0/E3_ensemble_f1 (built on /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/data/all.npz)
  xfer one-hand segments 441: kept 176 from 137 resets (193 fell / lost, 72 not settled at the end label); LocoManiLab val kept 1492
  primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:
    first reach : n  128 (128 resets) xfer 21.236 cm [20.360, 22.090] / val 2.964 cm (n 1022) = 7.164x [6.869, 7.428] (val resampled too [6.634, 7.774])
    chained     : n   48 (48 resets) xfer 37.227 cm [24.495, 41.842] / val 6.989 cm (n 470) = 5.326x [3.709, 5.974] (val resampled too [3.417, 6.185])
    all         : n  176 (137 resets) xfer 21.934 cm [20.952, 22.660] / val 3.993 cm (n 1492) = 5.493x [5.258, 5.677] (val resampled too [5.040, 5.889])
  secondary: direct error median (cm), xfer / LocoManiLab in place (kept, every split) [95 %]; mean direct error vector (heading frame, cm; left y mirrored) and the angle between them [95 %, xfer resampled]; model / direct on xfer:
    first reach : direct 10.21 / 10.35 = 0.986x [0.838, 1.195]; mean vector xfer [7.56, -3.47, -3.59] (|9.06|), LML [2.2, -4.89, -0.98] (|5.45|), angle 41.3 deg [32.2, 51.0]; model / direct 2.080
    chained     : direct 12.59 / 10.43 = 1.207x [0.901, 1.585]; mean vector xfer [1.43, -3.22, -4.21] (|5.49|), LML [-0.47, -4.55, -0.68] (|4.63|), angle 48.5 deg [18.6, 84.9]; model / direct 2.956
    all         : direct 10.63 / 10.38 = 1.024x [0.917, 1.208]; mean vector xfer [5.89, -3.4, -3.76] (|7.77|), LML [1.32, -4.78, -0.88] (|5.04|), angle 45.7 deg [35.0, 56.0]; model / direct 2.064
  training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer 87.5 % [82.6, 91.6] (val 87.2 %), joints out median 0; prediction error inside 21.902 cm (n 154), outside 38.593 cm (n 22); most often out: [('left_ankle_pitch_joint', 2.8), ('left_ankle_roll_joint', 2.8), ('right_wrist_roll_joint', 2.8), ('right_wrist_yaw_joint', 1.7), ('right_shoulder_roll_joint', 1.7)] (%)
  ensemble disagreement (|member std| of the predicted hand point, median cm): xfer 60.655 [60.258, 60.913], val 2.915, ratio 20.81x
```

**补充取法**（eval_ended.txt，执行 agent 摘出的数）
- 保留 248 段，来自 194 次重置。
- 第一次伸手 7.202 倍，接着上一次 5.438 倍，合计 5.551 倍 [5.347, 5.802]。
- 直接执行 1.345 倍，方向夹角 52.0°，模型/直接 1.588。
- 训练范围内 88.3 %，集成分歧 20.77 倍。

**停稳和丢失**（只算原地重置）

| | HEFT·HA | HEFT·LML data3 |
|---|---|---|
| 下命令前停不稳、被跳过 | 9.9 %（60/609） | 4.3 % |
| 段末没停稳 | 29.7 %（92/310） | 11.5 % |
| 末尾 0.5 s 规则速度 中位数 / p90 | 1.29 / 14.69 cm/s | 0.86 / 2.21 cm/s |
| 丢失 / 摔倒 | 66.8 % / 0.8 % | 59.3 % / 1.2 % |

**碰到场景**
- 主取法保留段里有 14.2 %（25/176）碰到场景，碰到的基本都是桌子。
- 剔除后合计 5.496 倍，不改结论。

**速度读数诊断**（sx_dq，事后做的，不进判据）

| x0 里的 dq | HA 第一次 / 接着上一次 / 合计 | 集成分歧 | LML val 对照 |
|---|---|---|---|
| 原样 | 7.16 / 5.33 / 5.49 倍 | 60.7 cm | 3.99 cm |
| 只把关节 17、18 置 0 | 合计 3.59 倍 | 19.2 cm | — |
| 全部置 0 | 合计 2.11 倍 | — | 4.42 cm（val 也变了，不干净） |
| 改用 q 的 10 Hz 差分 | 3.29 / 1.59 / 2.50 倍 | 5.7 cm | 4.07 cm |

**对照判据**
- 主取法合计 5.49 倍 [5.26, 5.68]，大于 1.5，落在"身体不一样，要重采"那一格。补充取法 5.55 倍，同一格。
- 原登记的预测 2–8 倍：中。
- 预测修订写的 1.3–3 倍：没中，实际高得多。扣掉读数问题后约 2.5 倍，落在修订的区间里，但这是事后数字，不能拿来对判据。
- 直接执行比值的预测：主取法 1.02 倍，补充取法 1.35 倍，都没中。
- 模型/直接 2.06：在 HA 里直接用 LML 的模型修正，误差反而大一倍。

## 诊断：HA 里踝 roll 的速度读数（逐子步检验，不进判据；diag_ha_dq.txt）

**做法**
- 用 dev 版 sx_ha.py --phase substep（md5 5caa230b），冻结拷贝没动。单进程跑在卡 1。
- 每个物理子步 scene.update 之后，读全部 43 个自由度（含 Dex3 手指）的 q 和 dq。
- 三个控制器各站 3 s，从 0.5 s 开始分析。

**结果**（左 / 右踝 roll）

| 控制器 | (i) 相邻子步 dq 变号的比例 | (ii) 2.5 s 内 ∫dq·dt，对比实际 q 变化 | (iii) 最接近的 Dex3 dq：中位 \|差\| / 中位 \|dq\| |
|---|---|---|---|
| ScaleBFM | 0 % / 0 % | +4.50 / −4.37 rad，对 +0.006 / +0.001 rad | 1.00 / 1.00 |
| HEFT | 0 % / 1 % | +3.27 / −1.81 rad，对 −0.023 / −0.022 rad | 1.00 / 1.00 |
| SONIC | 95 % / 41 % | +0.001 / +0.20 rad，对 −0.003 / −0.003 rad | 1.00 / 1.00 |

**读法**
- ScaleBFM 和 HEFT 的踝增益相同（kp 28.5、kd 1.81），都是情况 (ii)：读数约 1.3–1.9 rad/s，从不变号，但关节几乎不动。这是读数偏置，不是颤动，也不是混叠。
- SONIC 的踝增益是 kp 40、kd 2.0。它的左踝是情况 (i)：每个子步换一次号，幅度约 0.3 rad/s，每 4 个子步采一次，总落在同一相位。
- 三个控制器都不是 (iii)：读到的不是手指关节。
- 这几个关节在 HA 里的 armature 和 friction 都是 0。
- 控制器的 read_joint_state 读到的 dq，和我们记录的逐步完全相同，最大差 0。也就是说，ScaleBFM 和 HEFT 的策略在 HA 里一直看到一个不存在的踝速度。它们在 HA 里更晃、更常迈步，可能和这个有关，没验证。
- 原因只能推测：踝 roll 是脚掌着地前的最后一个关节，armature 为 0，接触下 PhysX 报的关节速度和位置积分对不上。没验证。

**SONIC·HA（组合 3）**
- 2026-11-25T19:55:33Z 开跑。
- 偏离计划：外层循环停掉后手动开跑，参数和 run_ha.sh 相同。命令记在 run_ha_manual.log：`SX_ASRUN=$O/asrun_ha bash $O/asrun_ha/selfmodel_xfer_20261125/sx_ha_batch.sh sonic_ha sonic 1 2 400`。

## 结果：组合 3，SONIC → HumanoidArena（2026-11-26）

**冻结和运行**：同一份 asrun_ha，开跑前 MD5.check 校验通过。2026-11-25T19:55:33Z 手动开跑，参数和 run_ha.sh 相同。卡 1、2，400 次重置，2 个进程，驱动阶段 1 h 38 min。run.json 记的是 controller sonic、stance native、delay_f 6、norender True。站姿用 SONIC 在 HA 里的默认姿态，见修订 2 第 4 条。CHECK PASSED，XFER CHECK PASSED。

**eval.txt 原文**：

```
xfer sonic (lml planner, native stance, delay 6 frames): /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/sonic_ha; xfer segments kept: not fallen and settled at the end label (d3_eval)
  model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic/models_s0/E3_ensemble_f1 (built on /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic/data/all.npz)
  xfer one-hand segments 433: kept 155 from 123 resets (3 fell / lost, 275 not settled at the end label); LocoManiLab val kept 2935
  primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:
    first reach : n  119 (119 resets) xfer 21.569 cm [20.050, 22.606] / val 2.860 cm (n 1557) = 7.543x [7.012, 7.915] (val resampled too [6.846, 8.044])
    chained     : n   36 (36 resets) xfer 36.303 cm [27.286, 50.029] / val 6.837 cm (n 1378) = 5.310x [3.885, 7.492] (val resampled too [3.878, 7.592])
    all         : n  155 (123 resets) xfer 22.225 cm [21.567, 23.279] / val 4.252 cm (n 2935) = 5.227x [5.072, 5.475] (val resampled too [5.010, 5.569])
  secondary: direct error median (cm), xfer / LocoManiLab in place (kept, every split) [95 %]; mean direct error vector (heading frame, cm; left y mirrored) and the angle between them [95 %, xfer resampled]; model / direct on xfer:
    first reach : direct 12.18 / 13.50 = 0.902x [0.800, 1.010]; mean vector xfer [-1.82, 2.72, 4.92] (|5.91|), LML [-3.58, -0.58, 8.12] (|8.89|), angle 31.3 deg [19.5, 44.6]; model / direct 1.771
    chained     : direct 13.05 / 15.58 = 0.837x [0.576, 1.475]; mean vector xfer [0.5, 2.3, 6.75] (|7.15|), LML [-4.47, -1.62, 9.3] (|10.45|), angle 40.5 deg [20.6, 67.3]; model / direct 2.782
    all         : direct 12.30 / 14.52 = 0.847x [0.759, 0.965]; mean vector xfer [-1.28, 2.62, 5.35] (|6.09|), LML [-4.0, -1.07, 8.67] (|9.61|), angle 33.7 deg [21.5, 46.9]; model / direct 1.807
  training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer 87.7 % [83.0, 92.0] (val 82.8 %), joints out median 0; prediction error inside 21.716 cm (n 136), outside 39.476 cm (n 19); most often out: [('waist_yaw_joint', 5.2), ('waist_roll_joint', 3.2), ('right_ankle_pitch_joint', 1.9), ('right_shoulder_roll_joint', 1.9), ('right_shoulder_yaw_joint', 1.9)] (%)
  ensemble disagreement (|member std| of the predicted hand point, median cm): xfer 18.997 [18.872, 19.237], val 3.132, ratio 6.07x
```

**补充取法**（执行 agent 摘出的数）
- 保留 430 段，来自 334 次重置。
- 第一次伸手 7.872 倍，接着上一次 8.151 倍，合计 5.742 倍 [5.419, 6.016]。
- 直接执行 0.993 倍，方向夹角 41.1°，模型/直接 1.693。

**停稳和摔倒**（只算原地重置）

| | SONIC·HA | SONIC·LML data3 |
|---|---|---|
| 下命令前停不稳、被跳过 | 32.0 %（253/791） | 1.6 % |
| 段末没停稳 | 66.0 %（413/626） | 4.8 % |
| 末尾 0.5 s 规则速度 中位数 / p90 | 2.14 / 9.70 cm/s | 0.79 / 1.53 cm/s |
| 摔倒 / 丢失 | 0.0 % / 2.5 % | 0.0 % / 7.0 % |

SONIC 在 HA 里不摔，但站不稳：伸手那只手的速度中位数 1.68 cm/s，p90 8.62 cm/s。主取法只留下 36 % 停稳的段，存在选择偏差。补充取法把这些段都算上，结果同一格。

**碰到场景**：433 段里只有 1 段碰到。

**速度读数诊断**（事后做的）
- SONIC 下，命令时刻偏离 LML 的有两个踝 roll（0.30 / 0.17 rad/s）和左右膝（0.28 rad/s，LML val 是 0.004）。这几维在 100 % 的段里都超出训练范围。
- 只把 17、18 置 0：5.39 倍。全部置 0：3.11 倍。改用 q 的 10 Hz 差分：3.06 倍，LML val 对照 4.24 cm，不变。

**对照判据**
- 主取法合计 5.23 倍 [5.07, 5.48]，大于 1.5，落在"身体不一样，要重采"那一格。补充取法 5.74 倍，同一格。
- 原登记的预测 2–6 倍：中。预测修订写的 > 1.5 倍：中。
- 直接执行比值的预测：主取法 0.85 倍，补充取法 0.99 倍，都没中。
- 模型/直接 1.81。

## 汇总、意外和结论（2026-11-26，HA 部分未复核）

**六个组合一览**（主取法合计，判据按登记读）

| 组合 | 倍数 [95 %] | 判据格 | 模型/直接 | 登记预测 | 只换速度特征（事后） |
|---|---|---|---|---|---|
| 1 ScaleBFM → SIMPLE | 2.38 [2.22, 2.54] | 要重采 | 0.96 | 1.5–4，中 | — |
| 2 SONIC → SIMPLE | 1.79 [1.48, 3.09] | 要重采 | 0.93 | 2–5，主取法略低于下限，补充取法 2.87 在区间内 | — |
| 6 HEFT → SIMPLE | 1.06 [0.86, 1.22] | 直接对比 | 0.44 | 2–6，没中 | — |
| 4 ScaleBFM → HA | 6.62 [6.35, 6.95] | 要重采 | 1.69 | 2–6，没中（略超） | 2.68 |
| 5 HEFT → HA | 5.49 [5.26, 5.68] | 要重采 | 2.06 | 2–8，中 | 2.50 |
| 3 SONIC → HA | 5.23 [5.07, 5.48] | 要重采 | 1.81 | 2–6，中 | 3.06 |

HA 三个组合另有预测修订（7826b351）：ScaleBFM 和 SONIC > 1.5 倍，两个都中；HEFT 1.3–3 倍，没中。

**意外**
1. HEFT → SIMPLE 能迁移，预测的是 2–6 倍。
2. HA 里有两三个关节的速度读数和实际运动对不上（踝 roll，SONIC 下还有膝），倍数里一大半来自这个。ScaleBFM 和 HEFT 的策略自己也读到这个值。
3. 三个控制器在 HA 里直接执行的误差大小，和在 LML 里差不多（0.85–1.26 倍），但方向偏了 30–50°。所以模型修正的方向是错的，结果比不修正还差。
4. 集成分歧在 HA 里是 val 的 6–21 倍，在 SIMPLE 的 HEFT 上是 1.03 倍。模型能分辨出自己没见过的情形。

**结论**
- LML 训出来的自身模型，只有 HEFT → SIMPLE 这一个组合能直接用。其余五个组合都要在各自的 benchmark 里重采。HA 里直接用不但没帮助，还更差。
- 能迁移的这一个组合，两边物理身体相同、驱动方式也相同。另外两个 SIMPLE 组合换了驱动方式，同时也换了求解器设置；HA 三个组合换了物理引擎、身体参数和关节速度读数。所以到底是哪个因素决定能不能迁移，现有数据只能分到"benchmark 和接法整体"这一层，分不到单个因素（见复核的实验 A、B）。
- 实际做法：每个控制器 × benchmark 组合，在 benchmark 自己的设置里采数据。上线前用集成分歧检查有没有跑出训练范围。
- 证据强度：每个组合 400 次重置，1 个模型种子（s0）。保留段 155–276 段，丢失和没停稳的比例各组合差别很大。

**四类陷阱**
- **benchmark 选择**：六个组合都是事先登记的。BiGym 按用户要求不跑。
- **数据泄漏**：xfer 的数据没有进过训练，分母只用 LML val。
- **指标误用**：主指标没换过。"只换速度特征"和"只算桌子"两项是事后加的，都标了出来，也不拿来对判据。
- **事后挑选**：主取法和补充取法都报了。SONIC 两个组合的停稳比例低（15 %、36 %），主取法有选择偏差，但补充取法落在同一格。

**下一步**（要用户定的，见 TODO）
- SIMPLE 的 ScaleBFM 和 SONIC：方案一，按现有接法在 SIMPLE 里重采；或方案二，改成 LML 的接法。
- HA：在我们的接入层里改用差分速度，同时喂给控制器和自身模型；或者照 HA 原样。
- 2×2（驱动方式 × 求解器）检验，做不做。

## 独立复核：HA 三个组合（2026-11-26，独立 agent，只读；要点，原意照录）

**总体**：四条核心论断里，第 1 条成立；第 2 条成立，但范围写窄了；第 3、4 条证据不足。另外发现一处事后改动：组合 4 的预测修订并不是"没看到任何结果"时写的。

**1. 数字核对**
- 重新筛段、重新预测，以下数字都一致：
  - 组合 4 合计 6.617 倍 [6.33, 6.95]；
  - 组合 5 模型/直接 2.064；
  - 三个组合"改用 q 的 10 Hz 差分"后 2.68 / 2.50 / 3.06 倍；
  - 集成分歧、保留段数、补充取法、接触剔除、两边的停稳表。
- 记录问题：组合 4 一节写的 sx_dq.py md5 3b7e2ed8 不对，diag_ha_dq.txt 和现在的文件都是 1dce476a。

**2. 四类陷阱**
- **"只换速度特征"能说明多少。**
  - 其他输入：骨盆角速度这一块，82–97 % 的段超出训练分布 6 个标准差。HA 静止时读数 0.04–0.07 rad/s，按姿态变化反推只有约 0.007。HEFT 和 SONIC 换成差分以后，速度块仍有约 80 % 的段超出 6 个标准差，只有 ScaleBFM 换完是干净的。
  - "LML val 上换掉 dq 几乎不变"不能推出"模型不用 dq"。把 HA 的 dq 原样注入 LML 原地 val，误差就变成 5–7.7 倍；只注入 3 % 的幅度，也有 1.15–1.95 倍。分布内的 dq 带的信息很少，但网络对越界的 dq 极其敏感：HA 的踝 roll 读数落在 350–620 个标准差之外。
  - "剩 2.5–3 倍"里还混着没清掉的越界输入。把角速度也换成 val 中位数，是 2.64 / 2.05 / 2.77 倍。所以这个数只能当上限看。
- **逐子步检验。**
  - 第一次伸手的起点是确定的，所以 3 s 站立的检验对第一次伸手够用。
  - 积分 4.5 rad 是这个关节全行程的 9 倍，不可能是真实运动。
  - 并联机构、mimic、tendon 都排除了：USD 里只有 43 个转动关节、11 个固定关节，踝是串联的 pitch → roll。站立时离关节限位也很远。
  - **新发现：读数对不上的远不止两三个关节。**
    - 逐子步数据里，29 个身体关节有 23–24 个（SONIC 下 16 个）的 0.1 s 平均 dq 和位置变化相差超过 0.02 rad/s。
    - 10 Hz 记录里有 10–11 个关节相差超过 0.05 rad/s，包括膝、髋、踝 pitch、腕 roll。LML 里是 0 个。
    - 再加上骨盆角速度也对不上，更像是整条关节链在接触下的速度读数有问题，比如 PhysX 速度迭代阶段（8 次位置迭代、4 次速度迭代），而不只是"末端关节 armature 为 0"。这一点没验证。
  - SONIC 下踝 roll 换号的 dq 和 q 的抖动吻合（相关 +0.67），是真实的 100 Hz 抖动被采样混叠，不是读数偏置。
  - HA 的关节配置（unitree.py 的 G129_CFG_WITH_DEX3_WHOLEBODY）：
    - 全部是 ImplicitActuatorCfg；
    - armature 0，friction 0；
    - 踝 roll 力矩上限 40 N·m；
    - 自碰撞关闭；
    - 踝 roll 的等效惯量只有 0.0037。
- **分母和选择偏差。**
  - 改用原地 val 当分母：6.86 / 5.64 / 5.33 倍。改用按命令找的最近邻 val：9.18 / 6.05 / 7.24 倍。HA 留下的是较容易的命令，所以登记的分母偏保守，换分母都不改落格。
  - HEFT 丢失率 66.8 % 对 59.3 %，用的是同一条规则，可以比，区间约 ±4.6 %。
- **事后改动。**
  - 时间线：冻结 17:03:43Z，开跑 17:03:57Z，预测修订提交于 17:16Z，第一个数据块 17:27Z 写出。
  - 但冻结前的流水线测试 `dev/ha_batch_test/eval.txt` 生成于 17:01:35Z，就是 ScaleBFM·HA 的完整评估：6 段，6.23 倍，模型/直接 2.0，集成分歧 12.2 倍。所以 ScaleBFM 那条"> 1.5 倍"不能算盲预测；HEFT 和 SONIC 那两条是盲的。
  - sonic_ha 手动开跑的命令和 run_ha.sh 的循环体相同。MD5.check 现在仍然通过。

**3. 可能推翻核心论断的实验**
- **A. 给控制器喂差分速度**（检验第 3、4 条）
  - 做法：在 ha_g1ctrl 的 read_joint_state / read_pelvis_state 里，改喂位置差分出的关节速度和姿态差分出的角速度。ScaleBFM 和 HEFT 各跑 400 次重置。
  - 怎么算推翻：ScaleBFM 摔倒率仍 ≥ 10 %，推翻第 4 条；方向夹角 < 15° 且倍数 ≤ 1.5，推翻第 3 条；倍数 ≤ 1.3，第 1 条在 HA 上也被推翻。
  - 成本：改代码和冒烟约半天，每个组合约 1.5 h。
- **B. 读数机制扫描**（检验第 2 条的机制）
  - 做法：扫 armature 0 / 0.01、速度迭代 4 / 1 / 0、踝增益两种。
  - 成本：约 1 h。
- **C. 不依赖速度读数的自身模型**（检验第 1、3 条）
  - 做法：在 data3 上重训 E3，dq 换成 10 Hz 差分，角速度换成重力方向差分或者去掉；先确认 LML val 不变，再直接在现有三份 HA 记录上评估。
  - 怎么算推翻：任一组合 ≤ 1.3 倍，推翻第 1 条；≤ 1.5 倍，说明"剩下的倍数"主要是外推，不是身体差异。
  - 成本：训练 2–4 h，评估几分钟。

**4. 判定**

| 结论 | 判定 | 理由 |
|---|---|---|
| 1. 三个组合都要重采 | 支持 | 怎么换分母、换取法、换速度特征，最低也有 2.05 倍；限制是只有 1 个模型种子 |
| 2. 倍数一大半来自速度读数 | 支持，要改措辞 | HA 的 dq 注入 LML val 就能单独造出 5–7.7 倍；但对不上的是整块关节速度加骨盆角速度，SONIC 的踝 roll 贡献为 0，"armature 为 0"的机制被削弱 |
| 3. 扣掉读数后剩 2.5–3 倍，对应控制器更晃、方向偏 | 证据不足 | 输入里还有越界；差分后模型/直接是 0.70 / 0.96 / 1.06，"修正比不修正还差"主要是读数造成的 |
| 4. 策略读到错的速度，所以更容易摔 | 证据不足 | HEFT 在 HA 里并不比 LML 更容易摔（0.8 % 对 1.2 %），SONIC 读同样的量却 0 % 摔倒；其他混杂因素没排除 |

## 复核后的更正（HA 部分，2026-11-26）

1. **预测修订写下时的状态**：应改为"正式运行还没出结果，但冻结前 4 次重置的流水线测试已经给出 ScaleBFM·HA 6.2 倍"。这个测试结果写修订时已经存在，执行 agent 也看过，所以 ScaleBFM·HA 那条预测不算盲预测。HEFT·HA 和 SONIC·HA 那两条是盲的。
2. **读数问题的范围**：不只是两个踝 roll 加膝，而是 HA 里大部分身体关节的速度读数，以及骨盆角速度，在着地站立时都和实际运动对不上。SONIC 的踝 roll 是真实的高频抖动。原因（armature 为 0、PhysX 速度迭代）没有验证。
3. **"扣掉读数后剩 2.5–3 倍"只是上限**：把角速度也清掉后是 2.05–2.77 倍。"误差方向偏了，所以修正比不修正还差"这个读法收回。差分以后模型/直接是 0.70 / 0.96 / 1.06，修正变差主要是读数造成的。
4. **集成分歧能用作上线前检查**，但在 HA 里它报的主要是读数问题：差分后分歧只剩 1.5–2.2 倍。
5. **组合 4 的 sx_dq.py md5** 是 1dce476a，不是 3b7e2ed8。
6. **"策略读到错的速度所以更易摔"**：证据不足。HEFT 在 HA 里并不更易摔。

## 补充更正：第一次伸手只有一个起点（2026-11-26，来自 fdvel 的独立复核）

六份 xfer 记录里，每份的第一次伸手都从同一个起始状态出发。仿真是确定的，重置时只换命令，起始状态不加扰动。比如 HEFT·HA 的 128 段第一次伸手只有 1 个起点，SONIC·HA 的 119 段也只有 1 个。LML val 里则是每段的起点都不同。

所以表里"第一次伸手"那一格，检验的只是一个起点上的不同命令，不能说明起始状态变化时迁移得怎么样；它的区间也只反映命令之间的差异。接着上一次的段，起点才各不相同。

HEFT→SIMPLE 第一次伸手 1.01 倍这个结论，也只在这一个起点上成立。以后的换仿真检验，重置要加上和 LML 相当的起始扰动。

## 用户的决定（2026-11-26）

> "SIMPLE 用方案一，HA 也重采，检验先不做"

- **SIMPLE**：保留现在的接法。ScaleBFM 和 SONIC 在 SIMPLE 里重新采数据、重新训练。HEFT 继续直接用 LML 的模型。
- **HA**：三个控制器都在 HA 里重新采数据。控制器的接法照现在的不变，包括它们读到的速度。模型输入按 fdvel 的结论改用差分速度。重置时加上起始扰动，这条来自 fdvel 的复核。
- **2×2 检验**（驱动方式 × 求解器）：先不做。
