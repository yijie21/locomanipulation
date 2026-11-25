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
