# selfmodel_simplecmp2_20261126：BendPick 换了站位 R_D 还赢不赢（E1），只修一段（E2），路线组只改时长（V1）

- **日期**：2026-11-26。在任何正式运行之前写好，单独提交。
- **机器 / GPU**：4090（`ssh -p 27984 root@14.103.60.188`），只用 2、3 号卡，0 号卡不碰。
  - 每张卡总量 ≤ 20 GB：每个作业开跑前读 nvidia-smi，等到该卡已用显存加 1024 MiB ≤ 19800 MiB 才起；
  - 我们的仿真每张卡最多 2 个；CPU 合计 ≤ 8 个进程，nice 10。
- **代码**：底子是 [selfmodel_simplecmp_20261126](selfmodel_simplecmp_20261126.md) 的冻结副本 `$O1/asrun/code`（`$O1` = `/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126`），它的行就是这份代码跑出来的。上面只换两样：
  - `simple_eval_20261112/simple_bench.py`：加默认关闭的 `--joint-dur`。改前 8d873cbb（= 那次冻结），改后 8e40e961；
  - 新目录 `scripts/experiments/selfmodel_simplecmp2_20261126/`：
    - `scmp2_jdur.py`（3218792d）：`--joint-dur` 本体，只在开关打开时才 import；
    - `scmp2_jdur_make.py`（fa497ded）：从另一组的经过点文件和行生成时长文件；
    - `scmp2_run.sh`（c2bd1931）：正式运行；
    - `scmp2_eval.py`（002deb3f）：登记的评估，统计函数全部复用 simplecmp 的 `scmp_eval.py`；
    - `scmp2_post.sh`（b9df22ce）：运行后的评估流水；
    - `scmp2_smoke.py`（eac17f80）：登记前核对的比较；
    - `scmp2_copy.sh`（fbbce9b5）、`scmp2_freeze.sh`（4fcbd7c3）：私有副本和冻结。

  Mac 工作区里另有 12 个别的实验的文件在那次冻结之后被别人改过，都不在本条目的路径上，不带进来。
  - 本条目的输出目录是 `/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126`（下称 `$O`；simplecmp 的是 `$O1`）。正式运行前用 `scmp2_freeze.sh` 生成 `$O/asrun/code`，同时记下 CODE_MD5、冻结时刻、KEY_MD5，以及输入的 md5（INPUTS_MD5）：模型 D、模板、主指标 A 的行、simplecmp 的 RD / J0 / JD 行和规划、V1 的两个时长文件。
  - 每一部分开跑前都核对 CODE_MD5。共享树一个字节都不写：场景写到 `$O/evalout/scene/`，其余条目是指向共享树的只读链接。
- **配置**：和 simplecmp 完全相同。ScaleBFM，感知栏（sensor=rgbd），腿部里程计（BFM_ROOT=odo），手点是 Dex3 手掌（REACH_HAND_OFF=0.0415,0,0），REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1。
  - 模型 D：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/train/models_D_s0/E3_ensemble_f1`（frac 1.0，seed 0），模板 `real/task/data/all.npz`；
  - R 规则、对准手掌、7 个候选，都照 simplecmp。
- **关联**：
  - [selfmodel_simplecmp_20261126](selfmodel_simplecmp_20261126.md)：结论 1 的限定一（BendPick 的结论只对那 10 个布局成立）和限定二（这批局已经被看过）、结论 5（路线组少的 7 局是每局慢约 0.2 s）；独立复核"可能推翻结论的实验"的 E1、E2。E3（没人看过的 BendPick 布局）这次不做；
  - selfmodel_precision_20261124：站位抖动开关 `--stance-jitter` 的来历，当时只用在 TabletopGrasp 上。BendPick 加抖动是第一次。

## 假设

1. **E1（主项）：换了伸手几何，R_D 在 BendPick 上还是比 A 多成功。**
   - simplecmp 里 R_D 在 BendPick 上 45 s 30/30，A 10/30，翻过来的 8 个布局是同一种失败：姿态相同，进入段都偏向同一侧（y 向 −5.6 到 −6.9 cm）。所以 +20 局可能只是"一种失败被修好"的 8 个实例。
   - 站位前后左右各挪最多 6 cm，伸手的距离和方向就变了，A 的失败也可能换一种样子。如果 R_D 的好处来自 D 真的会预测落点，它在新站位上也应当成立。
   - 这批局严格说已经被看过："对准手掌"和预测是登记前在这 120 局的 A 记录上定的。抖动后的伸手谁都没见过，部分补上这一点；物体的摆放还是原来那 10 个布局。
2. **E2：BendPick 失败的原因是进入段的偏差。**
   - A 在 BendPick 上失败的局，进入段偏差中位数 6.55 cm，成功的局 2.34 cm；预抓取只差 3.10 对 1.95 cm。
   - 所以只修进入段（RE）应当收回 R_D 那 +20 局的大部分，只修预抓取（RP）收不回。
3. **V1：路线组在官方时限下少的 7 局，是因为预抓取每局慢了约 0.2 s，不是落点变差。**
   - 少掉的 7 局，J0 在 15.86–16.00 s 成功，J_D 在 16.06–16.22 s 成功；J_D 的预抓取时长 28/28 局都比 J0 长，中位数 +0.23 s。
   - 只改时长、路径不变：J0 的路径放慢到 J_D 的时长（J0S），应当丢掉这 7 局的大部分；J_D 的路径加快到 J0 的时长（JDF），应当收回大部分。

## 做法

| 组 | 部分 | 站位 | 预抓取 | 进入段 | 局 |
|---|---|---|---|---|---|
| AJ1、AJ2 | E1 | 抖动，rep 1 / rep 2 | 目标 | 目标 | BendPick 0–29 × 2 = 60 |
| ALJ1、ALJ2 | E1 | 同上 | 目标；R 的反解照算、只记录（dryR） | 同左 | 60 |
| RDJ1、RDJ2 | E1 | 同上 | R：在 D 上反解，候选缩回到 IK ≤ 2 cm | R | 60 |
| RE | E2 | 不抖 | 目标 | R | BendPick 0–29 |
| RP | E2 | 不抖 | R | 目标 | BendPick 0–29 |
| J0S | V1 | 不抖 | simplecmp 的 J0 规划（关节轨迹），时长换成同一局 J_D 的预抓取时长 | 目标 | 28 个有路线的 TabletopGrasp 局 |
| JDF | V1 | 不抖 | simplecmp 的 J_D 规划，时长换成同一局 J0 的预抓取时长 | 目标 | 同上 |

E2 和 V1 的对照组（不抖动的 A、R_D、J0、J_D）直接用 simplecmp 的行，不重跑：仿真是确定的，simplecmp 已经核对过 AL = A 120/120。

- **站位抖动**（已有的开关，没有改）：`--stance-jitter 0.06 --jitter-seed 0 --jitter-rep k`。
  - 每局按（seed 0，crc32("任务:level:index")，rep，11）抽一次，前后、左右各在 ±6 cm 里均匀取值，按起步朝向转到世界系，加到站位上；
  - 同一局的三组用同一次抽样，行里记 `jitter` 和 `jitter_rep`；
  - 感知目标不变：那一眼在起步时看，抖动只改走到哪里停；
  - 原有的两个限制照旧：站位不越过桌边前 table_gap，走位裁剪到 APPROACH_BOX。所以实际挪动可能小于抽到的值，行里的 `walk`、`walk_err_cm` 记了实际走位。
- **只修一段**（已有的开关）：`--correct-stages entry` 或 `pregrasp`。没列出的那一段照原样发；反解的记录照样数每一下，所以进入段知道自己是第二下。
- **`--joint-dur <json>`（新开关，默认关）**：只在 `--joint` 时可用。
  - reach_joints 的时长是 max(reach_min_s, 掌心弧长 / 速度)。这里把速度设成"这条路径的弧长 / 要的时长"，弧长按 reach_joints 自己的算法算（命令时刻的参考姿态、路径的关节、掌心点）。所以这一下正好走那么久，路径、起点偏差的淡出、最小加加速度曲线都不变。
  - "要的时长"是另一组同一局的那条规划在这里会得到的时长：文件每局给出另一组的规划文件（`{"<任务>_L<level>_e<index>": {"plan": ..., "speed": null, "dur_s": 记录值}}`），开关在命令时刻用同样的参考姿态现算它的弧长，时长 = max(reach_min_s, 弧长 / reach_speed)。两组在预抓取命令之前逐位相同，所以这就是那一组当时的时长，而且不经过取整。
    - 起先用的是行里记录的时长（4 位小数）。冒烟时让 J0 的路径按它自己记录的时长重走（J0Ik），第 30 局的 t_success 就差了一个控制周期（15.52 → 15.54 s），见"登记前的核对"。J0 有 3 局正好 16.00 s 成功，这点取整就可能翻局，所以改成现算。
  - 这一下多记 `joint_dur_set`：要的时长、这条路径的弧长、速度、另一组的规划、它的弧长和记录的时长。
  - 时长文件在冻结时生成：`jdur_J0S.json` 列 simplecmp 每局 J_D 的规划（P/JD/via 的 src），`jdur_JDF.json` 列 J0 的。
  - 这是诊断用的干预，时长来自另一组同一局的规划，不是机器人的决策，也不是一个方法。
- 其余都和 simplecmp 一样：A 的选项（bench_via.sh 那一套，不录像），同样的局号和种子，`--trace`。

## 局、布局和判定集

- **E1**：BendPick 第 0–29 局 × rep 1、2，每组 60 次伸手。布局是 index：10 个布局，每个 6 次（3 个 level × 2 个 rep）。符号检验在 10 个布局上做；区间按布局重抽 4000 次，seed 0。
- **E2**：BendPick 第 0–29 局，不抖动，10 个布局。
- **V1**：simplecmp 的 28 个有路线的 TabletopGrasp 局，10 个布局。
- **时限**：
  - E1、E2 看 45 s。BendPick 的官方时限是 8 s，simplecmp 里各组成功都在 11.7–13.7 s，官方时限下各组都是 0，没有检验力；照报，不作判据。
  - V1 看官方时限（16.0 s），45 s 一起报。
- **分组之前的失败**：沿用 lmlcmp3 的排除规则，三类分开计（感知失败、发预抓取之前 IK 失败、没有那一眼），预计都是 0。

## 度量

- **E1 主比较**：R_D（RDJ）对 A（AJ），45 s，60 次。
  - 每个布局算 6 次的成功数，两组不同记 + 或 −，精确双侧符号检验；
  - 局上的 McNemar 精确检验一起报，局之间不独立，不作判据；
  - 成功率差的 95 % 区间，按布局重抽；
  - rep 1、rep 2 分开也报。
- **E1 机制**：每一下手掌偏差的中位数，A 和 R_D，预抓取和进入段分开；A 的失败局和成功局分开；D 对发出命令的预测误差（ALJ 对 A 的命令，RDJ 对 R_D 的命令）。
- **E1 的 F1：抖动后的起点上，D 在 R_D 的命令上还可信吗**。在 RDJ 实际发出的命令上算三项，和 simplecmp 相同：
  - (a) D 在任务标签处的误差中位数（13 cm 点，没摔倒的段），÷ 1.886 cm（登记的 SIMPLE val），> 1.3 判不可信；÷ 0.993 一起报；区间按布局重抽；
  - (b) "在范围内"的比例（sd_range 的三项检查：39 个量、最近邻、伸手表；阈值、训练集、表都和评估 4 相同，scmp_range.py 逐下算），< 80 % 判不可信；
  - (c) 发出命令中 IK 误差 > 2 cm 的比例，比 ALJ（A 的命令）高 ≥ 10 个百分点判不可信。

  ALJ 的三项作为参照一起报。
- **E2**：RE 对 A、RP 对 A、RE 对 R_D、RP 对 R_D，45 s，同样的检验和区间；每一下的手掌偏差，A、R_D、RE、RP。
- **V1**：J0S 对 J0、JDF 对 J_D、JDF 对 J0、J0S 对 J_D，官方时限和 45 s；预抓取和进入段的手掌偏差；预抓取时长；t_success；"45 s 内成功但超过 16.0 s"的局。
- **命令本身**：RDJ、RE、RP 的修正量、缩回次数、发出命令 IK > 2 cm 的个数、思考时间。
- **一致性**（不满足就停下查清，不做评估）：
  - AJk = ALJk 逐局相同，每个 rep 30/30（比较行和每一下的全部字段，只去掉 wall_s、开关自己加的字段、`bench.options.sonic_vr_frame`）；
  - 三组的抖动逐局相同；
  - RE 的预抓取 = A 的预抓取（30/30，比较这一下除 `correct` 外的全部字段）；RP 的预抓取 = R_D 的（30/30）；
  - J0S 的路径和命令时的关节 = J0 的，JDF 的 = J_D 的（28/28）；J0S 的预抓取时长 = J_D 的、JDF 的 = J0 的（记录里的 `correct.dur_s`，差 ≤ 1e-4 s，即记录的取整）；
  - plan_same 全为真；在线、离线预测之差 < 1e-4 m。

## 预测

**E1**（BendPick，抖动，每组 60 次）

| | 45 s | 官方时限 |
|---|---|---|
| A | 15–40（中间估计 25） | 0 |
| R_D | 50–60（中间估计 57） | 0 |
| R_D − A | +12 到 +40 | 0 |

- 理由：
  - A 在 BendPick 上是两种状态：进入段偏 ≈ 2–2.6 cm（成功），或 ≈ 5.7–6.9 cm（失败；index 5 的两局偏 5.7 cm 也成功了）。抖动改变伸手几何，哪种状态出现说不准，所以 A 的区间放得很宽；
  - R_D 在 30 局上把进入段压到 0.3–1.7 cm，两种状态都修好了。D 是在 data3 / data3b 的大量随机走位和伸手上训练的，不是在这 120 局上，新站位不一定出范围。
- 布局：+ 的布局 6–10 个，− 的 0–1 个。布局符号检验 p < 0.05 的可能约 75 %（要 ≥ 6 个 + 且 0 个 −，或 9 个 + 且 1 个 −；8 个 + 加 1 个 − 是 p 0.070，不够）。
- 机制：A 进入段偏差中位数 3–6.5 cm（看失败多少），R_D 预抓取 0.7–1.5 cm、进入段 0.9–2.0 cm。
- F1（RDJ，BendPick，两个 rep 合起来，120 下）：D 误差中位数 0.9–1.6 cm（0.48–0.85x）；在范围内 78–95 %，不抖动时是 88.3 %，(b) 可能擦线；IK > 2 cm 0 %。判"可信"的可能约 80 %。
- 一致性：AJ = ALJ 60/60。
- 落在判据哪一行：成立约 70 %，推翻约 10 %，证据不足约 5 %，两者之间约 15 %。

**E2**（BendPick，不抖动，30 局）

| 组 | 45 s 成功 | 预抓取偏差中位数 | 进入段偏差中位数 |
|---|---|---|---|
| A（simplecmp） | 10 | 3.01 | 6.19 |
| R_D（simplecmp） | 30 | 0.96 | 1.25 |
| RE（只修进入段） | 24–30 | 3.01（= A） | 1.0–2.2 |
| RP（只修预抓取） | 8–16 | 0.96（= R_D） | 4.5–7.0 |

- 理由：A 的成功局和失败局主要差在进入段，预抓取只差约 1 cm；进入段的反解从预抓取实际停下的地方算，D 看得到那个状态。
- RE − A ≥ +14 的可能约 70 %。
- 另一种可能：如果进入段落到哪种状态是由预抓取结束时的姿态决定的，RP 也会收回不少。那时"进入段偏 6.2 cm"只是症状，不是原因。

**V1**（28 个有路线的 TabletopGrasp 局）

| 组 | 官方时限 | 45 s | 预抓取偏差中位数 |
|---|---|---|---|
| J0（simplecmp） | 23 | 28 | 3.86 |
| J_D（simplecmp） | 16 | 27 | 2.09 |
| J0S（J0 的路径，J_D 的时长） | 15–19 | 27–28 | 3.4–4.4 |
| JDF（J_D 的路径，J0 的时长） | 20–24 | 26–28 | 1.8–2.8 |

- 理由：有路线的局都在 14.1–16.5 s 成功，J0 有 8 次成功落在 (15.8, 16.0] s；J_D 比 J0 慢的中位数 +0.23 s（0.02–0.74 s）。如果只是时间，J0S 的 t_success ≈ J0 的加上时长差，离 16.0 s 不到 0.2 s 的局会翻。
- 速度变了，跟踪误差也可能变，所以落点给了 ±0.5 cm 左右的区间。
- "时长解释成立"的可能约 65 %。

## 判据

**E1**（照独立复核写的；"几乎不失败"定成 A ≥ 54/60）

| 结果 | 结论和下一步 |
|---|---|
| **推翻**：R_D − A < 0；或 R_D − A = 0 且 A < 54；或 10 个布局的符号检验方向相反（− 的布局多于 +）；或 R_D 进入段偏差中位数 > 2.8 cm | BendPick 上的提高只限原来那 10 个布局的那一种失败。simplecmp 的结论 1 收回到"只在这 10 个布局、这一种失败上成立"，不能作为"SIMPLE 上直线伸手默认用 R_D"的依据 |
| **成立**：R_D − A ≥ +5，布局符号检验 p < 0.05 且 + 多于 −，并且 RDJ 在 BendPick 上 F1 可信 | 换了伸手几何，R_D 仍提高 BendPick 的 45 s 成功率，限定一放宽到"这 10 个物体摆放、多种站位"。simplecmp 判据第 1 行照走：直线伸手默认用 R_D，下一步在 SIMPLE 上跑 SONIC。官方时限下 BendPick 各组都是 0，这一条只在 45 s 下成立 |
| **证据不足**：A ≥ 54/60（且没有落在推翻） | 抖动后 A 自己几乎不失败，没有提高的余地，不算支持。结论 1 维持限定一 |
| **两者之间**：R_D − A 在 +1 到 +4，或 p ≥ 0.05，或 F1 不可信 | 照实报差和区间。结论 1 维持限定一；F1 不可信时，差别不归功于自身模型 |

- 54 的来由：A 失败 ≤ 6 次时，要 p < 0.05 至少得 6 个布局全是 + 且没有 −，也就是每次失败各在一个布局、全部修好，实际上够不着"成立"那一行。

**E2**（45 s）

| 结果 | 结论 |
|---|---|
| RE − A ≥ +14（收回 +20 的七成以上） | "进入段的偏差是 BendPick 失败的主要原因"站得住 |
| RE − A ≤ +10（一半或更少） | "进入段偏 6.2 cm 是失败原因"这个解释是错的，哪怕 R_D 整体的提高依然成立。再看 RP：RP − A ≥ +10 时原因在预抓取的终点 |
| RE − A 在 +11 到 +13 | 两段都有份，照实写 |
| RP − A ≥ +10 | 只修预抓取也收回不少（意外），记下，和上面一起读 |

**V1**（官方时限）

| 结果 | 结论 |
|---|---|
| J0S ≤ 18 且 JDF ≥ 21 | 时长解释成立：路线组少的 7 局是慢了的代价，平移本身没有让落点变差。保持原时长的路线修正值得做，另登记 |
| J0S ≥ 21 或 JDF ≤ 18 | 时长解释不成立，路线组变差另有原因（路径形状或落点）。simplecmp 的结论 5 要改写 |
| 其余 | 部分成立，照实写 |
| JDF 的预抓取偏差中位数比 J_D 大 > 0.5 cm | 加快本身让落点变差，"保持时长"不是免费的，记下 |

**一致性**任何一条不满足：停下查清，不做评估。

## 注意

- **思考时间不计时**：仿真在反解时暂停，和 simplecmp 一样。墙钟的思考时间逐下记录，一起报。
- **E1、E2 不会改变官方时限下的结论**：BendPick 的官方时限下各组都是 0，simplecmp 官方时限下四个任务合计 73 → 78 里，BendPick 贡献 0。
- **E1 换的是站位，不是物体摆放**：物体还是原来那 10 个布局。"没人看过的布局"（E3）这次不做。
- **决策输入不变**：决策只用感知、关节读数、里程计和自身模型。V1 的时长来自另一组的记录，是干预，不进任何方法。
- **局号、种子都和主指标一样**：E1 只多了站位抖动这一项（seed 0）。

## 登记前的核对（只用开发局，或开关关着、只发 A / J / R_D 原命令的测试局；原始输出见 `$O/checks/`）

所有命令第一行都写了 `# notebook: skip ...`，都在 2、3 号卡上（当时 2 号卡已用 17.9 GB、3 号卡 17.2–18.9 GB，每张卡我们最多 1 个仿真）。代码是 `$O/code_dev`（scmp2_copy.sh 生成，和要冻结的副本同一个做法）。开发树只有 TabletopGrasp 的那一眼，所以开发局的冒烟只能用 TabletopGrasp 30、36；BendPick 只在测试局上做"开关关着逐位相同"的核对。

**1. 开关关着，逐位相同（测试局，只发原来的命令）**：
- 新副本、不加任何新开关：BendPick 第 0、1 局和主指标的 A 逐位相同；TabletopGrasp 第 0 局按 J 的选项（主指标的经过点、`--joint`）和主指标的 J 逐位相同，所以 `--joint-dur` 关着时关节路径这一支没变；
- BendPick 第 0 局按 R_D 跑，和 simplecmp 的 R_D 逐位相同，发出的命令也相同。新副本复现了 simplecmp。

**2. 站位抖动（开发局 30，rep 1，抽到前 2.67 cm、左 2.48 cm）**：三组用的是同一次抽样；dryR（ALJk）和不加开关（AJk）逐位相同。

**3. 只修一段（开发局 30）**：
- 只修进入段（REk）时，预抓取和不加开关的 A（Ak）逐位相同，只有进入段带反解；
- 只修预抓取（RPk）时，预抓取和两段都修（Rk）逐位相同，进入段照原样发。

开发局 30 是挡边局，三组都失败，这里只看机制。

**4. `--joint-dur`（开发局 30、36，simplecmp checks/route1 的两套规划）**：
- 把 J0 的规划原样重放（J0k），和 route1 的 J0 逐位相同；
- J0 的路径按它自己的时长重走（J0Ik），除了新记的 `joint_dur_set` 外逐位相同；
- J0 的路径按 J_D 的时长走（J0Sk）、J_D 的路径按 J0 的时长走（JDFk）：时长和另一组记录的 `correct.dur_s` 一致到 4 位小数，路径不变。开发局上的结果（不计分）：J0Sk 比 J0 晚 0.18、0.32 s 成功，JDFk 比 J_D 早 0.28、0.32 s；第 36 局 JDFk 15.94 s，J_D 16.26 s。

起先时长取行里记录的 4 位小数，J0Ik 就不是逐位相同：第 30 局 t_success 差了一个控制周期。所以改成现算（见"做法"），改后重跑了第 4 项。改前的输出（`$O/checks/smoke.txt`，84a2c201，摘这三行）：

```
  ep 30 J0Ik: joint_dur_set {'asked_s': 5.2949, 'arc_cm': 79.424, 'speed': 0.150001}; correct.dur_s 5.2949 (the asked arm's 5.2949); joint_dur_s 5.29; joint_path_cm 79.42 (J0 79.42); pregrasp miss 3.374 (J0 3.374, J_D 1.503); t_success 15.54 (J0 15.52, J_D 15.64)
  ep 30: J0Ik vs route1 J0: differ in 12 keys: ['hand_rise_cm', 'rise_cm', 'stages.0.joint_dur_set (only in a)', 't_success', 'trace.104.2.1', 'trace.135.2.2', 'trace.171.2.1', 'trace.183.2.1']
  ep 36: J0Ik vs route1 J0: differ in 9 keys: ['stages.0.joint_dur_set (only in a)', 'trace.102.2.0', 'trace.160.2.1', 'trace.162.2.0', 'trace.163.2.0', 'trace.187.2.1', 'trace.225.2.2', 'trace.362.2.1']
```

`$O/checks/smoke3.txt`（d94e8888）。第 2、3 项来自 `checks/dev`（simple_bench 0b71e2dd，和最终的 8e40e961 只差 `--joint-dur` 打开时那一支），第 4 项来自 `checks/dev2`（现算时长之后，8e40e961）。第 1 项来自 `checks/test2`：在最终的 simple_bench（8e40e961）上重跑的，早先 0b71e2dd 上的 `checks/test` 结果相同（smoke2.txt 和 smoke3.txt 逐字节相同）：

```
rows: Ak 1, AJk 1, ALJk 1, RDJk 1, Rk 1, REk 1, RPk 1, J0k 2, J0Ik 2, J0Sk 2, JDFk 2, A0 2, Jt 1, Rt 1

== stance jitter (dev TabletopGrasp 30, rep 1)
  AJk: jitter [0.0267, 0.0248] rep 1; stand 0.3; success_45 False; pregrasp miss 27.339, entry 31.484 cm; --correct: pregrasp -, entry -
  ALJk: jitter [0.0267, 0.0248] rep 1; stand 0.3; success_45 False; pregrasp miss 27.339, entry 31.484 cm; --correct: pregrasp dryR, entry dryR
  RDJk: jitter [0.0267, 0.0248] rep 1; stand 0.3; success_45 False; pregrasp miss 28.99, entry 33.217 cm; --correct: pregrasp R, entry R
  Ak (no jitter): stand 0.3; success_45 False
  AJk = ALJk (dryR records, sends the goal): identical

== only one stage corrected (dev TabletopGrasp 30)
  Rk: --correct: pregrasp R, entry R; pregrasp miss 29.536, entry 32.66 cm; success_45 False
  REk: --correct: pregrasp -, entry R; pregrasp miss 27.862, entry 33.265 cm; success_45 False
  RPk: --correct: pregrasp R, entry -; pregrasp miss 29.536, entry 32.92 cm; success_45 False
  REk's pregrasp = Ak's (every stage key but correct): identical
  RPk's pregrasp = Rk's: identical
  RPk's entry vs Rk's (RPk's not corrected): differ in 18 keys

== joint paths (dev TabletopGrasp 30 / 36; simplecmp checks/route1's plans)
  ep 30: J0k = route1 J0 (the replay): identical
  ep 30 J0Ik: joint_dur_set {'asked_s': 5.294935, 'arc_cm': 79.424, 'speed': 0.15, 'plan': '/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/checks/route1/P/J0/step2/plan_TabletopGraspMP_L3_e0.npz', 'other_arc_cm': 79.424, 'other_dur_rec_s': 5.2949}; correct.dur_s 5.2949 (the asked arm's 5.2949); joint_dur_s 5.29; joint_path_cm 79.42 (J0 79.42); pregrasp miss 3.374 (J0 3.374, J_D 1.503); t_success 15.52 (J0 15.52, J_D 15.64)
  ep 30 J0Sk: joint_dur_set {'asked_s': 5.481898, 'arc_cm': 79.424, 'speed': 0.14488416, 'plan': '/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/checks/route1/P/JD/step2/plan_TabletopGraspMP_L3_e0.npz', 'other_arc_cm': 82.228, 'other_dur_rec_s': 5.4819}; correct.dur_s 5.4819 (the asked arm's 5.4819); joint_dur_s 5.48; joint_path_cm 79.42 (J0 79.42); pregrasp miss 3.374 (J0 3.374, J_D 1.503); t_success 15.7 (J0 15.52, J_D 15.64)
  ep 30 JDFk: joint_dur_set {'asked_s': 5.294935, 'arc_cm': 82.228, 'speed': 0.15529648, 'plan': '/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/checks/route1/P/J0/step2/plan_TabletopGraspMP_L3_e0.npz', 'other_arc_cm': 79.424, 'other_dur_rec_s': 5.2949}; correct.dur_s 5.2949 (the asked arm's 5.2949); joint_dur_s 5.29; joint_path_cm 82.23 (J0 79.42); pregrasp miss 1.579 (J0 3.374, J_D 1.503); t_success 15.36 (J0 15.52, J_D 15.64)
  ep 30: J0Ik vs route1 J0: differ in 1 keys: ['stages.0.joint_dur_set (only in a)']
  ep 30: JDFk's path = route1 J_D's: True
  ep 36: J0k = route1 J0 (the replay): identical
  ep 36 J0Ik: joint_dur_set {'asked_s': 5.735836, 'arc_cm': 86.038, 'speed': 0.15, 'plan': '/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/checks/route1/P/J0/step2/plan_TabletopGraspMP_L3_e6.npz', 'other_arc_cm': 86.038, 'other_dur_rec_s': 5.7358}; correct.dur_s 5.7358 (the asked arm's 5.7358); joint_dur_s 5.74; joint_path_cm 86.04 (J0 86.04); pregrasp miss 8.543 (J0 8.543, J_D 2.785); t_success 16.080000000000002 (J0 16.080000000000002, J_D 16.26)
  ep 36 J0Sk: joint_dur_set {'asked_s': 6.049219, 'arc_cm': 86.038, 'speed': 0.14222917, 'plan': '/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/checks/route1/P/JD/step2/plan_TabletopGraspMP_L3_e6.npz', 'other_arc_cm': 90.738, 'other_dur_rec_s': 6.0492}; correct.dur_s 6.0492 (the asked arm's 6.0492); joint_dur_s 6.05; joint_path_cm 86.04 (J0 86.04); pregrasp miss 8.577 (J0 8.543, J_D 2.785); t_success 16.400000000000002 (J0 16.080000000000002, J_D 16.26)
  ep 36 JDFk: joint_dur_set {'asked_s': 5.735836, 'arc_cm': 90.738, 'speed': 0.1581954, 'plan': '/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/checks/route1/P/J0/step2/plan_TabletopGraspMP_L3_e6.npz', 'other_arc_cm': 86.038, 'other_dur_rec_s': 5.7358}; correct.dur_s 5.7358 (the asked arm's 5.7358); joint_dur_s 5.74; joint_path_cm 90.74 (J0 86.04); pregrasp miss 2.788 (J0 8.543, J_D 2.785); t_success 15.940000000000001 (J0 16.080000000000002, J_D 16.26)
  ep 36: J0Ik vs route1 J0: differ in 1 keys: ['stages.0.joint_dur_set (only in a)']
  ep 36: JDFk's path = route1 J_D's: True

== the switches off, test episodes, against earlier rows
  BendPickMP 0: A0 = the main metric's A: identical
  BendPickMP 1: A0 = the main metric's A: identical
  TabletopGraspMP 0: Jt = the main metric's J: identical
  BendPickMP 0: Rt = simplecmp's R_D: identical
  BendPickMP 0: Rt's sent commands = R_D's: True
```

**5. 评估脚本**：用 simplecmp 的行拼了一个假的运行（`$O/checks/fake_run`：AJ = 主指标 A，ALJ = AL，RDJ = RD，加上假的抖动字段），scmp2_eval.py 跑通，数字和 simplecmp 的 BendPick 一致（R_D 30 对 A 10 每个 rep，进入段 6.19 对 1.25 cm，F1 1.135 cm、88.3 %）。

## 运行计划

| 部分 | 组 | 局 | 每局（冒烟时的墙钟） | 进程分钟 |
|---|---|---|---|---|
| E1 | RDJ1、RDJ2 | 60 | 约 20 s | 约 20 |
| E1 | ALJ1、ALJ2 | 60 | 约 20 s | 约 20 |
| E1 | AJ1、AJ2 | 60 | 约 15 s | 约 15 |
| E2 | RE、RP | 60 | 约 20 s | 约 20 |
| V1 | J0S、JDF | 56 | 约 20 s | 约 19 |

- 一个队列，每个作业 5 局，共 60 个作业，E1 的作业排在最前；2、3 号卡上各 2 个工作进程。
- 每个作业开跑前读 nvidia-smi：该卡已用 + 1024 MiB ≤ 19800 MiB 才起，否则 30 s 后再看；起一个作业后占着锁 45 s，让下一次读数看得到新进程。所以每 45 s 最多起一个作业，合计约 50 min–1 h。
- 冒烟时（10:30Z）2 号卡 17.9 GB、3 号卡 17.2–18.9 GB，别人的进程在涨落：实际常常每张卡只放得下 1 个，最慢约 1.5 h。
- CPU 同时 ≤ 4 个进程。
- 启动：`asrun/launch.sh`（第一行 `# notebook: selfmodel_simplecmp2_20261126`），nohup 在 4090 上跑；进度登记到灵动岛 `~/.claude/longjobs/simplecmp2.json`。
- 跑完后：`scmp2_post.sh`（CPU），先看一致性，再看结果。

## 预测修订

## 结果（贴原始输出，不转述）

## 意外

## 结论与下一步

## 独立复核
