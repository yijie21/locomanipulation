# Research Statement — Scripted Expert 何以替代人类 Teleop（草稿 v1）

> 2026-08-26。基于 scoop check（docs/../step1–6.md、papers/）与三轮实验。
> 标 ⏳ 的槽位等 round 3 闭环数字（预计当日）落地后填入。

## Thesis（一句话）

LEGS (arXiv:2606.01458) 证明了 scripted 数据**可以**替代人类 teleop 微调
humanoid loco-manip VLA；本工作回答它**何以成立、何时失败、代价几何** ——
核心发现是：scripted expert 与人类演示的差距不在动作的空间正确性，而在
**时间纹理（cadence）**，它会被 BC 克隆成闭环 freeze attractor，且只能在
数据源头治理，DAgger 与事后重采样都不根除。

## 三条 claim（scoop check 后的定位）

**C1（设定，不再是主张）— 零人工数据引擎。**
自验证、无人值守：privileged scripted expert（reach map / 可见性门 /
grasp-causality audit / per-episode manifest）+ 停滞看门狗。200 episodes
无人值守采集，expert 成功率 90.5%（200 OK / 16 grasp / 5 hold，walk 与
approach 零失败）。LEGS 占了"能不能"的旗；我们的信封宽 4 倍
（spawn-edge 0.7 m、yaw ±20°、lateral ±0.20 m vs LEGS 的 ±10 cm/±10°），
导航是真实发生的，不是排练路径。

**C2（主线）— cadence 是 scripted→BC 链路的一等数据质量轴。**
- 实测差距：scripted 到位停顿中位 **4.6 s**；NVIDIA 人类 teleop 参考 **0.2 s**；
  action chunk 视界 **1.33 s**（40 帧 @30 Hz）→ 96% 的停顿不可被 chunk 吸收。
- 机制：freeze 不是（纯）covariate shift，而是 **in-distribution 模仿** ——
  demo 教会了"到位后站着"，站着是自延续 attractor。旁证：靶向 DAgger
  仅 6%→13.7%（若纯属 recovery 缺失，杠杆应更大；且 recovery 数据本身
  携带同样的停顿）。
- 治理只能在源头：廉价 idle 过滤分不开到位停顿与闭合前精调保持
  （0.11 vs 0.06 rad/s，故 --drop-idle 被 park）；源头行为工程把停顿
  4.6→3.0 s、episode 35→21 s（信号密度 ≈2×）。
- **判决实验（round 3）**：同 200 episodes、仅治理 cadence 重训。
  闭环成功率：round 1 = 6/101，round 2 (+DAgger) = 14/102，
  round 3 = **⏳ __/100**。若显著超过 round 2，则以同等数据量隔离出
  cadence 变量。

**C3（实证刻画，非机制主张）— latent-WBC-token 动作空间把失败压缩为意图错误。**
三轮闭环失败分类学中**零摔倒**（失败全为 reach 停滞 / close 失误）：任何
token 解码出的都是 SONIC 稳定的全身运动，探索空间从"既要对又要不摔"
降维到"只要对"。机制 rationale 属 LeVERB/SONIC 一脉共有，不据为己有；
我们贡献的是测量。天然对照轴：LEGS 用同款 SONIC 的 18 维运动学命令线 ——
补一个同数据、命令线动作空间的对照训练即可将 C3 升回主张（backlog）。

## 与最近邻的分界（每篇一行）

| 工作 | 它做了 | 它没做（我们做了） |
|---|---|---|
| LEGS 2606.01458 | scripted+3DGS 替代 teleop，真机 1110 trials | 生成器一段带过；±10cm 信封；零 cadence/recovery/闭环失败分析；科学主张在渲染轴 |
| ISR 2606.22907 | 人类 teleop 演示的事后 pause 重采样，+25% | scripted vs human 的类别差距；chunk 视界机制；闭环 freeze 因果链；源头治理 |
| LeVERB 2506.13751 | 共训 latent verb + RL WBC | 固定预训练 WBC 的现成 token wire；物体 loco-manip；失败分类学 |
| SIMPLE 2606.08278 | benchmark + planner 自动生成 + 离线重渲染 | 行为塑形 expert；数据质量科学（它是基础设施，将来引用其重渲染架构） |
| MimicGen 一族 | 臂式自动数据生成 | locomotion / WBC / cadence |

## 证伪条件（诚实清单）

- round 3 ≤ round 2 → C2 的"cadence 是主要残余瓶颈"被削弱，
  回到 coverage 叙事（DAgger 加码 / spawn 课程）。⏳
- 命令线对照训练若与 token 线同等成功率 → C3 的动作空间优势不成立。
- 人类 teleop 上界对照（同信封 200 条 teleop 数据）缺失 —— 目前只能
  引 NVIDIA 参考数据的 0.2 s 停顿统计，不能引其成功率。写作时须声明。

## 实验阶梯

已完成：round 1 (6/101) → round 2 (+50 DAgger, 14/102) → round 3 数据
（200 eps，cadence 治理后）→ 训练中。
进行中：round 3 open-loop sweep + closed-loop 100 rounds ⏳。
计划：18 维命令线对照（C3 升格）；failure-state fork + privileged-progress
advantage weighting（sim 免 V-JEPA 的 SRPO 变体，见讨论文档）；
3DGS/UE 重渲染（根位姿通道已在 99 处排队）；真机（LEGS 已验通
SONIC sim→real 链路，可引用）。
