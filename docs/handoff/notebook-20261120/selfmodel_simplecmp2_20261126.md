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

### 运行

- **冻结**：2026-11-26T10:49:00Z，`$O/asrun/code`，1798 个文件，CODE_MD5 文件本身 ad9c7927。
  - scmp2_freeze.sh 写 KEY_MD5 时停了：`md5sum: scripts/experiments/selfmodel_simplecmp_20261126/__pycache__: Is a directory`。simplecmp 的冻结副本里有它评估时留下的 `__pycache__` 目录，脚本用了 `set -e`。
  - CODE_MD5 在这之前已经写好（不含 `__pycache__`），核对通过。
  - 剩下三步照脚本手工补完，冻结的代码没动：KEY_MD5 只列文件（28 个）；两个时长文件（各 28 局，jdur_J0S.json 记录的时长 5.167–6.088 s，jdur_JDF.json 4.996–5.801 s）；INPUTS_MD5（375 个输入）。
- **正式运行**：10:49:34Z 到 11:34:43Z，`asrun/launch.sh`（d46f28a1），nohup 在 4090 上跑。
  - 60 个作业全部 exit 0，296 行；
  - 每次起作业时，2、3 号卡已用最多 18633 / 18624 MiB，加上这个作业约 1 GB，最高约 19.7 GB；0 号卡没碰。
- **评估**：`scmp2_post.sh`，CPU，输出在 `$O/run/eval/`。

launch.log 和 run.log（去掉 60 行 `start`、60 行 `exit 0`）：

```
10:49:34 start: 0, 6207 MiB 1, 17923 MiB 2, 17998 MiB 3, 18984 MiB 
11:34:43 run end (exit 0): 0, 6207 MiB 1, 16376 MiB 2, 16466 MiB 3, 16373 MiB 
[10:49:34] parts E1 E2 V1; routed TabletopGrasp episodes 0,1,2,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
[10:49:34] queue all: 60 jobs, GPUs 2 3 x 2, need 1024 of cap 19800 MiB
[11:34:43] AJ1: 30 rows
[11:34:43] AJ2: 30 rows
[11:34:43] ALJ1: 30 rows
[11:34:43] ALJ2: 30 rows
[11:34:43] RDJ1: 30 rows
[11:34:43] RDJ2: 30 rows
[11:34:43] RE: 30 rows
[11:34:43] RP: 30 rows
[11:34:43] J0S: 28 rows
[11:34:43] JDF: 28 rows
[11:34:43] DONE all
```

### 一致性：eval/cmp_AJ1_ALJ1.txt、cmp_AJ2_ALJ2.txt（头两行、末行）

```
a /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/rows (30 rows), b /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/rows (30 rows): 30 episodes in both
  BendPickMP ep 0: identical; success_45 True / True; pregrasp: err 3.288 / 3.288, ik 0.0 / 0.0 | entry: err 3.146 / 3.146, ik 0.0 / 0.0
  ...
identical 30 / 30 (compared: every row and stage key but ['correct_reset', 'film', 'wall_s'] and stages' ['correct'])
a /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/rows (30 rows), b /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/rows (30 rows): 30 episodes in both
  BendPickMP ep 0: identical; success_45 True / True; pregrasp: err 2.265 / 2.265, ik 0.0 / 0.0 | entry: err 2.843 / 2.843, ik 0.0 / 0.0
  ...
identical 30 / 30 (compared: every row and stage key but ['correct_reset', 'film', 'wall_s'] and stages' ['correct'])
```

### 在线、离线对照：eval/offline.txt

"against simpledata's A record" 那一行在这里不作检查：只有不抖、发 A 原命令的组（simplecmp 的 AL）才应当是 0。抖动后的起点、R 组发的命令都和 A 的记录不同；simplecmp 的 R_D 那一行也是 2.839。

```
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_ALJ1: 60 segments; online = offline on 60 sent stages (plan_same): max |pred diff| 4.977e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (60 reaches): max |x0 diff| 6.765e-01, |u diff| 9.513e-02, |pred diff| 7.948e-02 m
  BendPickMP           n  60: D error at the task label median 0.963 cm = 0.511x SIMPLE val, 0.970x test_iid task-walk
  all                  n  60: 0.963 cm = 0.511x / 0.970x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_ALJ2: 60 segments; online = offline on 60 sent stages (plan_same): max |pred diff| 4.908e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (60 reaches): max |x0 diff| 5.981e-01, |u diff| 8.981e-02, |pred diff| 8.709e-02 m
  BendPickMP           n  60: D error at the task label median 1.065 cm = 0.565x SIMPLE val, 1.073x test_iid task-walk
  all                  n  60: 1.065 cm = 0.565x / 1.073x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_RDJ1: 60 segments; online = offline on 60 sent stages (plan_same): max |pred diff| 4.954e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (60 reaches): max |x0 diff| 6.765e-01, |u diff| 1.192e-01, |pred diff| 1.073e-01 m
  BendPickMP           n  60: D error at the task label median 1.004 cm = 0.532x SIMPLE val, 1.011x test_iid task-walk
  all                  n  60: 1.004 cm = 0.532x / 1.011x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_RDJ2: 60 segments; online = offline on 60 sent stages (plan_same): max |pred diff| 4.995e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (60 reaches): max |x0 diff| 5.981e-01, |u diff| 1.419e-01, |pred diff| 1.270e-01 m
  BendPickMP           n  60: D error at the task label median 1.145 cm = 0.607x SIMPLE val, 1.153x test_iid task-walk
  all                  n  60: 1.145 cm = 0.607x / 1.153x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_RE: 60 segments; online = offline on 30 sent stages (plan_same): max |pred diff| 4.936e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (30 reaches): max |x0 diff| 8.882e-01, |u diff| 6.886e-02, |pred diff| 5.705e-02 m
  BendPickMP           n  60: D error at the task label median 1.170 cm = 0.620x SIMPLE val, 1.178x test_iid task-walk
  all                  n  60: 1.170 cm = 0.620x / 1.178x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_RP: 60 segments; online = offline on 30 sent stages (plan_same): max |pred diff| 4.945e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (30 reaches): max |x0 diff| 2.257e-01, |u diff| 4.060e-02, |pred diff| 3.541e-02 m
  BendPickMP           n  60: D error at the task label median 1.122 cm = 0.595x SIMPLE val, 1.130x test_iid task-walk
  all                  n  60: 1.122 cm = 0.595x / 1.130x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_J0S: 56 segments; online = offline on 56 sent stages (plan_same): max |pred diff| 4.997e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (56 reaches): max |x0 diff| 3.220e+00, |u diff| 1.561e-01, |pred diff| 1.043e-01 m
  TabletopGraspMP      n  56: D error at the task label median 1.978 cm = 1.049x SIMPLE val, 1.992x test_iid task-walk
  all                  n  56: 1.978 cm = 1.049x / 1.992x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126/run/corr_JDF: 56 segments; online = offline on 56 sent stages (plan_same): max |pred diff| 4.967e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (56 reaches): max |x0 diff| 2.973e+00, |u diff| 1.561e-01, |pred diff| 1.039e-01 m
  TabletopGraspMP      n  56: D error at the task label median 1.667 cm = 0.884x SIMPLE val, 1.679x test_iid task-walk
  all                  n  56: 1.667 cm = 0.884x / 1.679x
```

### 逐下的范围检查：eval/range.txt

```
training set 809804 segments; nearest-neighbour p99 21.152 (sd_range's eval4: 21.152)
ALJ1: 60 reaches, in range 83.3 % (kept 60), D median 0.963 cm
ALJ2: 60 reaches, in range 86.7 % (kept 60), D median 1.065 cm
RDJ1: 60 reaches, in range 83.3 % (kept 60), D median 1.004 cm
RDJ2: 60 reaches, in range 86.7 % (kept 60), D median 1.145 cm
RE: 60 reaches, in range 88.3 % (kept 60), D median 1.170 cm
RP: 60 reaches, in range 88.3 % (kept 60), D median 1.122 cm
J0S: 56 reaches, in range 83.9 % (kept 56), D median 1.978 cm
JDF: 56 reaches, in range 92.9 % (kept 56), D median 1.667 cm
```

### 登记的评估：eval/eval.txt（d78c701d）

```
== rows: AJ 60, ALJ 60, RDJ 60, A 120, RD 120, RE 30, RP 30, J0 28, JD 28, J0S 28, JDF 28

== 1. consistency
  E1 rep 1: AJ = ALJ (every row and stage key but wall_s, --correct's fields, sonic_vr_frame): 30 / 30
  E1 rep 2: AJ = ALJ (every row and stage key but wall_s, --correct's fields, sonic_vr_frame): 30 / 30
  E1: the same stance jitter in AJ, ALJ and RDJ: 60 / 60; jitter |ahead|, |left| max 0.0544, 0.0557 m
  E2: RE's pregrasp = A's (every stage key but correct): 30 / 30; RP's pregrasp = R_D's: 30 / 30
  V1 J0S: path and the command's joints = J0's 28 / 28; pregrasp duration (correct.dur_s) - JD's: max |diff| 0.00000 s (28 / 28 within the record's 1e-4 s)
  V1 JDF: path and the command's joints = JD's 28 / 28; pregrasp duration (correct.dur_s) - J0's: max |diff| 0.00000 s (28 / 28 within the record's 1e-4 s)
  RDJ: corrected stages 120, plan_same 120, fallbacks 0
  RE: corrected stages 30, plan_same 30, fallbacks 0
  RP: corrected stages 30, plan_same 30, fallbacks 0

== 2. E1: BendPick, stance jittered (up to 6 cm), A vs R_D, 60 episodes (2 reps x 30); a layout = the index
  45 s     rep 1: A 15, AL 15, R_D 30 (of 30)
  45 s     rep 2: A 14, AL 14, R_D 30 (of 30)
  45 s     both : A 29, AL 29, R_D 60 (of 60)
  official rep 1: A 0, AL 0, R_D 0 (of 30)
  official rep 2: A 0, AL 0, R_D 0 (of 30)
  official both : A 0, AL 0, R_D 0 (of 60)
  E1 R_D vs A 45 s    : 60 / 60 vs 29 / 60 (+51.7 pp [+35.0, +68.3] by layout); layouts + 9 / - 0 of 10, sign test p 0.003906; episodes + 31 / - 0, McNemar p 9.313e-10
  E1 R_D vs A official: 0 / 60 vs 0 / 60 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  E1 R_D vs A 45 s rep 1: 30 / 30 vs 15 / 30 (+50.0 pp [+30.0, +70.0] by layout); layouts + 8 / - 0 of 10, sign test p 0.007812; episodes + 15 / - 0, McNemar p 6.104e-05
  E1 R_D vs A 45 s rep 2: 30 / 30 vs 14 / 30 (+53.3 pp [+36.7, +70.0] by layout); layouts + 9 / - 0 of 10, sign test p 0.003906; episodes + 16 / - 0, McNemar p 3.052e-05
  pregrasp palm miss median (n 60): A 2.90, R_D 0.85 cm | A's failures 3.14, A's successes 2.21 cm | D's error of the sent command AL 0.93, R_D 0.87 cm
  entry    palm miss median (n 60): A 5.74, R_D 1.18 cm | A's failures 6.20, A's successes 3.15 cm | D's error of the sent command AL 1.13, R_D 1.15 cm
  F1 on the jittered starts (BendPick, both reps; D's error at the task label, sd_range's in-range, the sent commands' IK):
  ALJ  : n 120, D median 1.035 cm [0.959, 1.147] = 0.55x SIMPLE val [0.51, 0.61], 1.04x test_iid task-walk; in range 85.0 %; IK > 2 cm 0.0 %
  RDJ  : n 120, D median 1.103 cm [0.956, 1.158] = 0.58x SIMPLE val [0.51, 0.61], 1.11x test_iid task-walk; in range 85.0 %; IK > 2 cm 0.0 %
  F1 RDJ BendPick: trustworthy
  E1 verdict (45 s): R_D - A = +31, A 29 / 60, R_D entry miss median 1.18 cm -> HOLDS (R_D - A >= +5, layout sign test p < 0.05, F1 trustworthy)

== 3. E2: BendPick, no jitter: only the entry corrected (RE), only the pregrasp (RP); A and R_D from before
  45 s    : A 10, R_D 30, RE 30, RP 10 (of 30)
  official: A 0, R_D 0, RE 0, RP 0 (of 30)
  E2 RE vs A   45 s: 30 / 30 vs 10 / 30 (+66.7 pp [+40.0, +90.0] by layout); layouts + 8 / - 0 of 10, sign test p 0.007812; episodes + 20 / - 0, McNemar p 1.907e-06
  E2 RP vs A   45 s: 10 / 30 vs 10 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  E2 RE vs R_D 45 s: 30 / 30 vs 30 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  E2 RP vs R_D 45 s: 10 / 30 vs 30 / 30 (-66.7 pp [-90.0, -43.3] by layout); layouts + 0 / - 8 of 10, sign test p 0.007812; episodes + 0 / - 20, McNemar p 1.907e-06
  pregrasp palm miss median (n 30): A 3.01, RD 0.96, RE 3.01, RP 0.96 cm
  entry    palm miss median (n 30): A 6.19, RD 1.25, RE 1.36, RP 6.10 cm
  E2 verdict (45 s): RE - A = +20, RP - A = +0 -> the entry's miss explains it (RE - A >= +14)

== 4. V1: TabletopGrasp's routed episodes, the paths of one arm at the other's pregrasp durations
  45 s    : J0 28, J_D 27, J0S 28, JDF 27 (of 28)
  official: J0 23, J_D 16, J0S 14, JDF 25 (of 28)
  V1 J0S vs J0  45 s    : 28 / 28 vs 28 / 28 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  V1 J0S vs J0  official: 14 / 28 vs 23 / 28 (-32.1 pp [-50.0, -12.5] by layout); layouts + 0 / - 5 of 10, sign test p 0.0625; episodes + 0 / - 9, McNemar p 0.003906
  V1 JDF vs J_D 45 s    : 27 / 28 vs 27 / 28 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  V1 JDF vs J_D official: 25 / 28 vs 16 / 28 (+32.1 pp [+7.7, +57.1] by layout); layouts + 4 / - 0 of 10, sign test p 0.125; episodes + 9 / - 0, McNemar p 0.003906
  V1 JDF vs J0  45 s    : 27 / 28 vs 28 / 28 (-3.6 pp [-10.7, +0.0] by layout); layouts + 0 / - 1 of 10, sign test p 1; episodes + 0 / - 1, McNemar p 1
  V1 JDF vs J0  official: 25 / 28 vs 23 / 28 (+7.1 pp [+0.0, +16.7] by layout); layouts + 2 / - 0 of 10, sign test p 0.5; episodes + 2 / - 0, McNemar p 0.5
  V1 J0S vs J_D 45 s    : 28 / 28 vs 27 / 28 (+3.6 pp [+0.0, +10.7] by layout); layouts + 1 / - 0 of 10, sign test p 1; episodes + 1 / - 0, McNemar p 1
  V1 J0S vs J_D official: 14 / 28 vs 16 / 28 (-7.1 pp [-16.7, +0.0] by layout); layouts + 0 / - 2 of 10, sign test p 0.5; episodes + 0 / - 2, McNemar p 0.5
  pregrasp palm miss median (n 28): J0 3.86, JD 2.09, J0S 3.86, JDF 2.10 cm
  entry    palm miss median (n 28): J0 2.34, JD 2.94, J0S 2.30, JDF 3.01 cm
  pregrasp duration median: J0 5.39, JD 5.58, J0S 5.58, JDF 5.39 s; t_success median: J0 15.77, JD 15.94, J0S 16.01, JDF 15.70 s
  J0 : within 45 s but after the 16.0 s limit 5 [1, 2, 6, 11, 21]
  JD : within 45 s but after the 16.0 s limit 11 [0, 1, 2, 6, 7, 10, 11, 12, 16, 17, 22]
  J0S: within 45 s but after the 16.0 s limit 14 [0, 1, 2, 6, 7, 10, 11, 12, 16, 17, 18, 21, 22, 26]
  JDF: within 45 s but after the 16.0 s limit 2 [1, 11]
  V1 verdict (official): J0S 14, JDF 25 -> the duration explains it (J0S <= 18 and JDF >= 21); JDF's pregrasp miss - J_D's +0.02 cm

== 5. the commands
  RDJ pregrasp: n 60, sent IK > 2 cm 0; correction median 3.14 cm, max 4.22; shrunk 0 of 0; think median 4.8 s
  RDJ entry   : n 60, sent IK > 2 cm 0; correction median 5.98 cm, max 7.83; shrunk 0 of 0; think median 1.4 s
  RE  entry   : n 30, sent IK > 2 cm 0; correction median 5.80 cm, max 7.17; shrunk 0 of 0; think median 1.7 s
  RP  pregrasp: n 30, sent IK > 2 cm 0; correction median 3.27 cm, max 4.09; shrunk 0 of 0; think median 4.6 s
```

### 事后，未登记：eval/posthoc.txt（posthoc.py 5dfb1251，只读）

A 在抖动后失败的是不是同一种；每个布局的成功数；V1 里 t_success 的变化和时长变化的关系；翻局的逐局时刻。

```
== E1 post hoc: A's entry miss vector (miss, cm) by outcome; no jitter (main A) vs jittered (AJ1 + AJ2)
  no jitter failures  n 20: miss x/y/z median [0.38, -6.53, -0.3], y range [-6.92, -5.55], |miss| median 6.54
  no jitter successes n 10: miss x/y/z median [-1.26, -1.89, -0.16], y range [-5.63, -0.88], |miss| median 2.34
  jittered  failures  n 31: miss x/y/z median [0.1, -6.18, -0.12], y range [-6.84, -5.13], |miss| median 6.20
  jittered  successes n 29: miss x/y/z median [-1.26, -2.42, 0.16], y range [-5.86, -0.68], |miss| median 3.15
  per layout (index): successes A no jitter (of 3) | A jittered rep1+rep2 (of 6) | R_D jittered (of 6)
    0: 2 | 4 | 6
    1: 0 | 2 | 6
    2: 3 | 6 | 6
    3: 0 | 2 | 6
    4: 1 | 3 | 6
    5: 3 | 4 | 6
    6: 0 | 0 | 6
    7: 0 | 1 | 6
    8: 0 | 4 | 6
    9: 1 | 3 | 6
  drawn jitter (cm): ahead [-5.3, 5.4], left [-5.6, 5.5]
  A fails    n 31: jitter ahead median +0.8, left median -1.6 cm
  A succeeds n 29: jitter ahead median -0.1, left median +1.4 cm

== V1 post hoc: t_success change against the pregrasp duration change, per routed episode (official 16.0 s)
  J0S - J0 : n 28; duration change median +0.231 [+0.020, +0.739] s; t_success change median +0.220 [+0.040, +0.740] s; t_success change - duration change median +0.003 [-0.073, +0.055] s
  JDF - J_D: n 27; duration change median -0.231 [-0.739, -0.020] s; t_success change median -0.200 [-0.740, +0.000] s; t_success change - duration change median -0.001 [-0.063, +0.058] s
  J0 -> J0S: official flips 9: ep 0 16.0 -> 16.26, ep 7 15.920000000000002 -> 16.16, ep 10 16.0 -> 16.22, ep 12 16.0 -> 16.38, ep 16 15.899999999999999 -> 16.32, ep 17 15.86 -> 16.240000000000002, ep 18 15.82 -> 16.02, ep 22 15.96 -> 16.22, ep 26 15.740000000000002 -> 16.080000000000002
  J_D -> JDF: official flips 9: ep 0 16.16 -> 15.879999999999999, ep 2 16.22 -> 15.86, ep 6 16.26 -> 15.96, ep 7 16.06 -> 15.780000000000001, ep 10 16.18 -> 15.879999999999999, ep 12 16.22 -> 15.84, ep 16 16.16 -> 15.760000000000002, ep 17 16.1 -> 15.760000000000002, ep 22 16.080000000000002 -> 15.8
  J0 -> JDF: official flips 2: ep 2 16.02 -> 15.86, ep 6 16.080000000000002 -> 15.96
```

## 意外

预测的对照：E1、E2 全部落在预测区间里。V1 的两个成功数都出了区间，方向都是"时长的作用比预测的大"。

| 项 | 预测 | 结果 |
|---|---|---|
| E1 A，45 s | 15–40 / 60 | 29 |
| E1 R_D，45 s | 50–60 / 60 | 60 |
| E1 R_D − A | +12 到 +40 | +31 |
| E1 布局 + / − | 6–10 / 0–1 | 9 / 0（p 0.0039） |
| E1 A 进入段偏差中位数 | 3–6.5 cm | 5.74 |
| E1 R_D 预抓取 / 进入段偏差 | 0.7–1.5 / 0.9–2.0 cm | 0.85 / 1.18 |
| E1 F1（RDJ）：D 误差 / 在范围内 / IK > 2 cm | 0.9–1.6 cm / 78–95 % / 0 % | 1.103 cm（0.58x）/ 85.0 % / 0 %，可信 |
| E2 RE / RP，45 s | 24–30 / 8–16 | 30 / 10 |
| E2 RE / RP 进入段偏差 | 1.0–2.2 / 4.5–7.0 cm | 1.36 / 6.10 |
| V1 J0S，官方时限 | 15–19 | **14** |
| V1 JDF，官方时限 | 20–24 | **25** |
| V1 J0S / JDF，45 s | 27–28 / 26–28 | 28 / 27 |
| V1 J0S / JDF 预抓取偏差 | 3.4–4.4 / 1.8–2.8 cm | 3.86 / 2.10 |

1. **V1：时长一换，成功时刻就一比一地跟着移，比预测的还干净**（事后 posthoc.txt）。
   - t_success 的变化减去时长的变化，中位数 J0S +0.003 s、JDF −0.001 s，每局都在 ±0.075 s 以内。
   - J0 放慢后丢的 9 局里，有 simplecmp 里 J_D 少的那 7 局，另外 2 局是 ep 18（15.82 → 16.02 s）和 ep 26（15.74 → 16.08 s）。
   - J_D 加快后收回的 9 局里也有那 7 局，另外 2 局是 ep 2、6：这两局 J0 自己也超时（16.02、16.08 s），JDF 在 15.86、15.96 s 成功。JDF 和 J0 的预抓取时长相同，所以这 2 局 JDF 更早，是在预抓取之后的阶段。JDF 的预抓取偏差 2.10 cm，和 J_D（2.09）一样，没有 J0（3.86）那么大。
   - 结果是 JDF 官方时限 25/28，比 J0 的 23 还多 2 局（2 个布局 +，0 个 −，p 0.5，不显著）。保持原时长的平移修正可能不止"不亏"，这只是 2 局，没有检验力。
2. **抖动后 A 自己的成功率变高了，但失败的还是同一种**（事后）。
   - A 在 BendPick 上 45 s 从 10/30（33 %）到 29/60（48 %）。
   - A 失败的局，进入段 y 向偏 −5.1 到 −6.8 cm（中位数 −6.18），不抖时是 −5.55 到 −6.92；成功局 y 中位数 −2.42（不抖时 −1.89）。抖动没有带来新的失败方式。
   - A 失败的局，抽到的左右抖动中位数 −1.6 cm（往右），成功的局 +1.4 cm：站位往右挪更容易落进那种失败。
   - 所以 E1 说明的是：在前后左右 ±5.5 cm 的多种站位上，R_D 都修好了这一种失败；它没有说明 R_D 能修好别的失败。
3. **E2 比预测干脆**：RE 和 R_D 逐局相同（30 局都成功），RP 和 A 逐局相同（同样的 10 局成功）。预抓取修到 0.96 cm，进入段偏差还是 6.10 cm（A 6.19）：预抓取对进入段几乎没有影响。
4. **布局 2 抖不抖 A 都是全成功**（3/3、6/6），这个布局没有提高余地，所以 + 的布局最多 9 个，结果正好 9 个。布局 6 抖动后 A 仍然 0/6。
5. **冻结脚本撞到 `__pycache__`，停在 KEY_MD5**（见"结果 · 运行"）。
   - 本条目的副本是从 simplecmp 的冻结副本抄来的，那里有它评估时留下的 `scripts/experiments/selfmodel_simplecmp_20261126/__pycache__/`。
   - scmp2_freeze.sh 用 `md5sum <目录>/*` 列 KEY_MD5，碰到这个目录返回非零，`set -e` 下脚本停了，时长文件和 INPUTS_MD5 都没生成。
   - CODE_MD5 用的是 `find -type f -not -path '*/__pycache__/*'`，不受影响，核对通过。剩下的步骤手工补完，冻结的代码没动。
   - **建议**：以后的冻结脚本在所有列文件的地方都排除 `__pycache__`（用 `find -type f -not -path '*/__pycache__/*'`，不用 `目录/*`）；从别的条目的冻结副本起手时，先删掉副本里的 `__pycache__` 再算 md5。

## 结论与下一步

> 2026-11-26 先写了草稿，独立复核回来以后按复核收窄（复核员只读了"结果"和预注册，E1、E2、V1 三项都判"支持"，原文见"独立复核"）。引用写成"结果 · 文件 · 节或行"，eval.txt 的节号就是它自己的 `== n.`；引用复核写成"复核 · 节"。本节和结果、意外、复核一样，还没提交。

### 结论

1. **E1：换了站位，R_D 在 BendPick 上仍然提高 45 s 成功率。适用范围只到：45 s 时限、那 10 个物体摆放、那一种"进入段侧偏约 6 cm"的失败。**
   - R_D 60/60，A 29/60，+51.7 pp [+35.0, +68.3]；布局 + 9 / − 0，符号检验 p 0.0039；局上 + 31 / − 0（结果 · eval.txt §2 `E1 R_D vs A 45 s`）。两个 rep 分开也各自成立：+15，p 0.0078；+16，p 0.0039（§2 `rep 1`、`rep 2`）。复核独立重算，全部一致（复核 · 1 E1）。
   - **适用范围**（复核 · "抖动后的失败还是同一种"的"E1 能说明的范围"）：
     - 只在 45 s 下成立。官方 8 s 时限下三组都是 0/60（§2 `official both`）。
     - 只有那 10 个物体摆放。站位挪动最多约 5.5 cm（抽到的前后 −5.3 到 +5.4 cm、左右 −5.6 到 +5.5 cm，结果 · posthoc.txt `drawn jitter`），朝向几乎没转：命令时在 −3.9° 到 −0.8° 之间。
     - 只有一种失败：进入段侧偏约 6 cm。抖动后 A 失败的 31 局，进入段 y 偏差全在 −6.84 到 −5.13 cm（中位数 −6.18），全是在合手阶段把物体碰开；不抖时失败的 20 局是 −6.92 到 −5.55（posthoc.txt 两行 `failures`；复核同上节）。抖动改的是哪种状态出现，没有带出新的失败方式。R_D 修好了这种失败 31/31。
     - 不能推出：R_D 能修别种失败；能推到别的物体摆放或别的任务。
   - **F1 可信，但进入段单看在线下。** 抖动后的起点上，D 误差中位数 1.103 cm = 0.58x，在范围内 85.0 %，IK > 2 cm 0.0 %（§2 `RDJ` 行、`F1 RDJ BendPick: trustworthy`）；A 的命令上是 1.035 cm、85.0 %、0.0 %（§2 `ALJ` 行）。
     - 85.0 % 是两下合起来的。只看进入段是 47/60 = 78.3 %，ALJ 也一样；不抖时只有 23/30（复核 · 2 度量误用）。起作用的恰好是进入段，而它单独看在 80 % 线下。
     - 登记的判据是两下合起来，所以判"可信"不变，这 +31 按判据记到自身模型名下。
     - 出范围的进入段，D 的误差并不更大（RDJ1：0.706 对 1.132 cm），这 13 下全部成功（同上）。
   - **D 的修正跟着每局的状态变，不是加一个常数**（复核 · "抖动后的失败还是同一种"）：
     - 在 A 自己的命令上（ALJ），D 预测的进入段 y 偏差和实际偏差相关 0.978。失败局预测 −5.10、实际 −6.18 cm；成功局预测 −2.72、实际 −2.42 cm。
     - R_D 的 y 修正和 A 的偏差相关 −0.965：A 失败的局中位数 6.23 cm，A 成功的局 3.20 cm。
     - D 对失败局少估约 1 cm：|预测 − 实际| 失败局 1.334 cm，成功局 0.826 cm。
   - 机制和成功一致：R_D 预抓取偏差 0.85 cm、进入段 1.18 cm，A 是 2.90 / 5.74 cm（§2 `pregrasp`、`entry`）。进入段 1.18 cm 远低于推翻线 2.8 cm。
   - A 自己的成功率随站位变：10/30（33 %）→ 29/60（48 %）（§2 `45 s both`）；往右挪的站位更容易落进那种失败（posthoc.txt 末两行，左右抖动中位数 −1.6 对 +1.4 cm）。两组用同一次抽样配对，这部分在差里抵掉了。
2. **E2：BendPick 的失败全部来自进入段的偏差。**
   - 只修进入段（RE）30/30，和 R_D 逐局相同；只修预抓取（RP）10/30，和 A 逐局相同（§3 `RE vs R_D`、`RP vs A`，都是 + 0 / − 0）。RE − A = +20 ≥ +14，落在"进入段偏差是主要原因"那一行（§3 `E2 verdict`）。
   - 预抓取修到 0.96 cm，进入段仍偏 6.10 cm，A 是 6.19（§3 `entry palm miss`）：预抓取的终点对进入段的偏差几乎没有影响。
   - 所以 simplecmp 里 R_D 在 BendPick 上的 +20，可以全部记在进入段的修正上。这只在不抖动的原 30 局上测过。
3. **V1：路线组在官方时限下的损失是时长造成的，不是落点变差。**
   - 只改时长、路径不变：J0 的路径放慢到 J_D 的时长（J0S），官方时限 23 → 14；J_D 的路径加快到 J0 的时长（JDF），16 → 25（§4 `official`）。
     - 局上：J0S 对 J0 + 0 / − 9，JDF 对 J_D + 9 / − 0，McNemar p 都是 0.0039；
     - 布局上：− 5 / + 0（p 0.0625），+ 4 / − 0（p 0.125）（§4 `V1 J0S vs J0 official`、`V1 JDF vs J_D official`）。
     - 落在判据"J0S ≤ 18 且 JDF ≥ 21"那一行（§4 `V1 verdict`）。45 s 下各组不变：28、27、28、27（§4 `45 s`）。
   - 成功时刻一比一地跟着时长移（事后，未登记）：t_success 的变化减去时长的变化，中位数 +0.003 s（J0S）、−0.001 s（JDF），每局都在 ±0.075 s 以内（posthoc.txt V1 两行）。simplecmp 里 J_D 少的那 7 局，J0S 全部丢掉，JDF 全部收回（posthoc.txt `official flips`）。
   - 平移本身没有让落点变差：JDF 预抓取偏差 2.10 cm，J_D 2.09；J0S 3.86，J0 3.86；进入段 3.01 对 2.94、2.30 对 2.34 cm（§4 `pregrasp palm miss`、`entry palm miss`）。
   - **时长相同时，J_D 的路径反而多 2 局**：JDF 对 J0 是 25 对 23，多的是第 2、6 局；J0S 对 J_D 是 14 对 16，少的是第 18、26 局；p 0.5（§4 `V1 JDF vs J0 official`；复核 · 1 V1）。没有检验力，只能说"不亏"，不能说"更好"。
   - 所以路线输掉的是时长：J_D 在 simplecmp 里少的那 7 局，换成 J0 的时长就全部收回。复核重算的斜率是 0.996 / 1.021，时长差和成功时刻差之差 ≤ 0.06 s（3 个控制周期）的局是 27/28 和 26/27（复核 · 1 V1）。
   - JDF 是诊断用的干预：它的时长取自同一局 J0 的规划，不是一个能用的方法。
   - 背景：16.0 s 的官方时限是一条刀刃，每局约 0.2 s 就翻 7–9 局。simplecmp 结论 5 的"每局慢了约 0.2 s"现在有了干预证据，"待验证"可以去掉。

### 证据强度

- **一次运行，一个模型**：D 是 seed 0、frac 1.0 的 E3_ensemble，没有换 seed。只有 ScaleBFM 一个控制器，只有感知栏（sensor=rgbd）一种设置。仿真是确定的（AJ = ALJ 30/30 × 2，§1；登记前新副本复现了主指标的 A、J 和 simplecmp 的 R_D），重跑同一组只得同样的行，不确定性只来自局的抽样。
- **复核的重算**：E1、E2、V1 的计数、检验和偏差都由复核员用自己的脚本从行里重算，和 eval.txt 一致；有 4 处区间端点差一格，原因是 eval 的各个比较共用一条随机流，没有判据用到区间（复核 · 1 一致性）。在线、离线预测之差和"在范围内"没有对训练集独立重算（复核 · 怎么核的）。
- **E1**：每组 60 次伸手、60 个不同站位，但只有 10 个物体摆放。布局 2 两组都是 6/6，没有提高余地（posthoc.txt 每布局表），能变的 9 个布局全是 +。9 个布局时符号检验能达到的最小 p 就是 0.0039，已经到了上限。同一布局的 6 次伸手不独立，McNemar 只作参考。失败都是同一种，有效的"失败多样性"只有一种。
- **E2**：30 局、10 个布局，RE、RP 和对照组逐局相同，这批局上的读法是确定的；没有抖动版本，也只在 BendPick 上。
- **V1**：28 局、10 个布局。局上 McNemar p 0.0039，布局符号检验 0.0625 / 0.125，不到 0.05。但这是干预，不是相关：只动了时长，路径、命令时的关节都和原组相同（§1 `V1 J0S`、`V1 JDF` 两行，28/28，时长差 0）；成功时刻一比一跟着移（事后）。JDF 比 J0 多 2 局只是计数，没有检验力。

### 其他解释

1. **状态相关预测和平均偏移的贡献没有拆开**：E1 里 A 的失败都是同一侧、差不多大的偏差，这次的结果仍分不开两者各占多少。有关的观察：
   - R_D 的修正跟着每局的状态变：D 预测的进入段 y 偏差和实际相关 0.978，R_D 的 y 修正和 A 的偏差相关 −0.965，A 失败局修 6.23 cm、成功局修 3.20 cm（复核 · "抖动后的失败还是同一种"）。所以 R_D 发出的不是一个常数修正。
   - R_D 每次修正的大小有起伏，预抓取中位 3.14 cm、最大 4.22 cm，进入段中位 5.98 cm、最大 7.83 cm（§5 `RDJ`）；A 成功的 29 局在 R_D 下也都成功（§2，R_D 60/60）。
   - 但"修正随状态变"不等于"随状态变才有用"。要回答后一个问题，需要复核提的归因对照 X2（"下一步"第 4 条，待用户定）。
2. **E1 的提高只是因为抖动后 A 变差或变好了**：已排除。两组用同一次抽样逐局配对（§1 `the same stance jitter ... 60 / 60`），A 的成功率随站位的变化在差里抵掉了。
3. **E2 里"只修进入段就够"，是因为反解从 A 的预抓取终点出发，正好修到了 A 的那种偏差**：这正是 E2 想测的，不算另一种解释。但它意味着 E2 的读法和 A 的失败方式绑在一起：换一种失败方式（比如预抓取就偏得很远），预抓取的修正可能就有用了。这次没测。
4. **V1 的差别来自速度变了以后跟踪误差变了，不是时间本身**：可能性小。落点几乎没变（预抓取 2.10 对 2.09、3.86 对 3.86 cm，§4），而成功时刻的变化和时长变化一比一（事后，posthoc.txt）。
5. **V1 的差别来自规划噪声**：已排除。J0S、JDF 重放的是 simplecmp 同一轮的规划文件，路径逐局相同（§1），没有重新规划。

### 四类陷阱

- **benchmark 选择**：
  - BendPick 0–29 局、28 个有路线的 TabletopGrasp 局，都是主指标原样的局号和种子，登记时定。
  - 抖动的幅度 6 cm、rep 1 和 2、seed 0 是复核提出、登记时定的，没有挑 rep，两个 rep 都报了（§2）。
  - BendPick 是 simplecmp 里效果最大的那个任务。这一轮是专门针对它的复查，不是新任务上的证据（复核 · 2 benchmark 选择）。
  - 局限：只有 SIMPLE、只有 ScaleBFM；E1 只有 10 个物体摆放、一种失败方式；BendPick 够不着 8 s 的官方时限，E1、E2 的结论只在 45 s 口径下成立。
- **数据泄漏**：
  - D 和它的训练数据（data3、data3b）都没变，没有用这些局训练。
  - E1 的 60 个站位登记前谁都没见过；但物体摆放就是 simplecmp 那 10 个，"对准手掌"也是在那批局的 A 记录上定的。
  - E2、V1 用的就是 simplecmp 已经看过的局，这是有意的：它们要查的就是 simplecmp 在那批局上的解释。
  - 登记前的冒烟只用开发局；测试局上只跑过开关关着、照原样发命令的逐位核对（"登记前的核对"第 1 项）。
- **指标误用**：
  - E1、E2 看 45 s、V1 看官方时限，都是登记时定的；另一个时限照报（§2 `official`、§3 `official`、§4 `45 s`）。
  - F1 用登记的任务标签和两个分母（§2 `ALJ`、`RDJ` 行）。F1(b) 按登记是两下合起来（85.0 %）；只看进入段是 78.3 %，在 80 % 线下，照实写在结论 1 里，判"可信"不变（复核 · 2 度量误用）。
  - 手掌偏差用了仿真真值，只作度量，不进入任何决策。V1 的时长来自另一组的规划，是干预，不进任何方法。
- **事后挑选**：
  - 一次正式运行，所有组全部报了；预测和判据在开跑前提交（286c9fc0）。
  - V1 的两个数出了预测区间，照报（"意外"第 1 条）。
  - 登记前改过一次 `--joint-dur` 的时长来源（记录值 → 现算）。原因是开发局上的逐位核对不过（"登记前的核对"第 4 项），和测试局的结果无关，登记里写明了。
  - 事后分析一项（posthoc.txt，标了未登记），只用来写结论 1 的限定和结论 3 的机制，不改判据的读法。

### 对照判据

| 判据行 | 结果 | 成立吗 |
|---|---|---|
| E1 推翻：R_D − A < 0，或 = 0 且 A < 54，或布局方向相反，或 R_D 进入段 > 2.8 cm | +31；+ 9 / − 0；1.18 cm（§2） | 不成立 |
| E1 成立：R_D − A ≥ +5，布局 p < 0.05 且 + 多于 −，F1 可信 | +31，p 0.0039，可信（§2；进入段单看 78.3 %） | **成立**（只在 45 s 下；那 10 个摆放；那一种进入段侧偏约 6 cm 的失败） |
| E1 证据不足：A ≥ 54/60 | A 29/60 | 不成立 |
| E1 两者之间 | — | 不适用 |
| E2：RE − A ≥ +14 | +20（§3） | **成立**：进入段的偏差是主要原因 |
| E2：RE − A ≤ +10 / +11 到 +13 | — | 不成立 |
| E2：RP − A ≥ +10 | 0（§3） | 不成立：只修预抓取收不回 |
| V1：J0S ≤ 18 且 JDF ≥ 21 | 14、25（§4） | **成立**：时长解释成立 |
| V1：J0S ≥ 21 或 JDF ≤ 18 | — | 不成立 |
| V1：JDF 预抓取偏差比 J_D 大 > 0.5 cm | +0.02 cm（§4 `V1 verdict`） | 不成立：加快没让落点变差 |
| 一致性不满足 | 全部满足（§1、offline.txt、cmp_AJ*_ALJ*.txt） | 不成立 |

### 下一步

1. **直线伸手**：E1 成立，按 simplecmp 判据第 1 行走，SIMPLE 上直线伸手默认用 R_D。写的时候带上三个限定：
   - 只在 45 s 口径下成立，官方时限下 BendPick 三组都是 0/60；
   - 只有那 10 个物体摆放，站位挪动最多约 5.5 cm，朝向几乎没转；
   - 修好的是那一种失败：进入段侧偏约 6 cm。
2. **SONIC**：SIMPLE 上跑 SONIC 的 R_D，草稿是 [selfmodel_simplesonic_20261126](selfmodel_simplesonic_20261126.md)，没有登记、没有开跑。注意 simpledata 只训了 ScaleBFM，SIMPLE 上还没有 SONIC 自己的模型，草稿里写了怎么处理。修哪一段仍按登记的 R_D 两段都修：E2 只是解释 ScaleBFM 在 BendPick 上为什么赢，不据此去掉预抓取的修正，XMove 两个任务上预抓取的修正有没有用，这次没测。
3. **路线组**：simplecmp 结论 5 改成"已由 V1 的干预证实"。平移修正本身不伤落点，代价全在路径变长以后的时长上；时长相同时 J_D 的路径反而多 2 局（p 0.5）。如果要在官方时限上用平移修正的路线，需要一个能用的定时办法，例如沿用原规划的时长；这要另登记，由协调者和用户定要不要做。在 16.0 s 的刀刃上追这几局值不值，也一起定。
4. **复核提的两个实验，待用户定，不开跑**（复核 · 3）：
   - **X1（主）：新摆放加朝向抖动。** 在没看过的 BendPick 物体摆放上，站位抖动 ±6 cm，再加起步朝向抖动 ±15°，A 对 R_D 配对，至少 10 个新布局 × 6 次。登记前先在 A 上定好失败分类：侧偏型（进入段 y 在 −7.5 到 −4.5 cm）和其他型，分开报。这一条替代原来的 E3，用来检验"那一种失败"和"那 10 个摆放"这两个限定能不能去掉。前提是先查 SIMPLE 的 BendPick 生成器能不能给出新摆放；不能的话，退一步，在现有 10 个布局上只加朝向抖动。
   - **X2（可选）：打乱 x0 做归因对照。** R 所用的 D 起点特征 x0，换成同一 rep 里另一局的 x0（固定置换，优先把 A 成败不同的两局配对），其余不变。R_shuf ≥ 57/60 说明提高不需要 D 对每一局的预测，"自身模型"这个归因不成立；R_shuf ≤ A + 5 说明归因站得住。它回答的是"其他解释"第 1 条。
5. **冻结脚本**：以后排除 `__pycache__`（"意外"第 5 条）。
6. **beliefs.md**：等用户定了这次的改动再增量更新。

## 独立复核（2026-11-26）

只读了假设、做法、局和判定集、度量、预测、判据、注意、登记前的核对、运行计划、预测修订和"结果"。没读"意外""结论与下一步"。

**怎么核的**
- **预注册**：`git diff 286c9fc0 -- 条目` 只有一处改动，从"## 结果"标题之后开始，235 行全是新增，0 行删除。
  - 假设、做法、度量、预测、判据、注意都没改；"预测修订"为空。
  - 预注册提交在 10:48:33Z，早于冻结（10:49:00Z）和开跑（10:49:34Z）。
- **冻结和输入**：在 4090 上重核了 `asrun/code/CODE_MD5`（1798 个文件）和 `INPUTS_MD5`（375 个输入），全部通过。
  - 10 个行文件拉回本地，md5 和远端一致。
  - 读过的代码，本地 md5 和冻结副本相同：scmp_correct.py、reachwm 的 data / models / train、d3_collect.py、wm.py、scmp2_run.sh。
- **数字**：用自己的脚本 `/Users/zed/.claude/jobs/f13ab968/tmp/review_scmp2/scripts/`（e1、e2、v1、cons、dpred）。
  - 直接读本实验的 `run/rows/*.jsonl`、simplecmp 的 RD / J0 / JD 行和主指标的 A 行。
  - 不调用 scmp2_eval.py 或 scmp_eval.py。
  - 区间按布局（index）重抽 4000 次，seed 0。
- **没能独立重算的**：
  - 在线、离线预测之差：要重跑 D，没做。
  - "在范围内"：只按 range.json 的逐下标记重数，没有对训练集重算。
  - D 在任务标签处的误差：取 range.json 的 err_cm，另用行里在线记录的 pred_err13_cm 交叉核对。

**1. 数字**

E1（BendPick，抖动，每组 60 次）
- **成功数**：
  - 45 s：A 29，AL 29，R_D 60。rep 1：A 15，R_D 30；rep 2：A 14，R_D 30。
  - 官方时限：三组都是 0 / 60。
- **主比较**：+51.7 pp [+35.0, +68.3]。
  - 布局 + 9 / − 0，符号检验 p 0.003906。
  - 逐局 + 31 / − 0，McNemar p 9.313e-10。
  - rep 1：+50.0 pp [+30.0, +70.0]；布局 + 8 / − 0，p 0.007812；McNemar 6.104e-05。
  - rep 2：+53.3 pp [+36.7, +70.0]；布局 + 9 / − 0，p 0.003906；McNemar 3.052e-05。
  - 以上全部和 eval.txt 一致。
- **每个布局的成功数**（A / R_D，各 6 次）：0: 4/6，1: 2/6，2: 6/6，3: 2/6，4: 3/6，5: 4/6，6: 0/6，7: 1/6，8: 4/6，9: 3/6。只有布局 2 打平。
- **手掌偏差中位数**（cm）：
  - 预抓取：A 2.895，R_D 0.847；A 失败局 3.138，A 成功局 2.206。
  - 进入段：A 5.7415，R_D 1.18；A 失败局 6.201，A 成功局 3.146。
  - D 对发出命令的误差（手掌）：预抓取 AL 0.9311、R_D 0.87275；进入段 AL 1.12855、R_D 1.1473。
- **F1**：
  - RDJ，n 120：D 误差中位数 1.1029 cm = 0.585x。我得的区间 [0.956, 1.154]，eval 是 1.103 [0.956, 1.158]。在范围内 102/120 = 85.0 %。IK > 2 cm 0.0 %。
  - ALJ：1.0347 cm，85.0 %，0.0 %。
  - 三项都过线，判"可信"成立。
  - 行里在线记录的 pred_err13_cm 中位数：RDJ 1.0977，ALJ 1.0296，和离线值差 ≤ 0.007 cm。
- **判据**：落在"成立"行。
  - R_D − A = +31 ≥ +5；p 0.003906，且 + 多于 −；F1 可信。
  - 推翻行的四条都不沾：R_D 进入段偏差 1.18 cm ≤ 2.8 cm。
  - A 29 < 54，不是"证据不足"。

E2（不抖动，30 局）
- **成功数**：45 s 下 A 10，R_D 30，RE 30，RP 10；官方时限下都是 0。
- **比较**：
  - RE 对 A：+66.7 pp [+40.0, +90.0]；布局 + 8 / − 0，p 0.007812；逐局 + 20 / − 0，McNemar 1.907e-06。
  - RP 对 A：差 0。RE 对 R_D：差 0。
  - RP 对 R_D：−66.7 pp。我得的区间 [−90.0, −40.0]，eval 是 [−90.0, −43.3]。布局 − 8，p 0.007812。
- **"RE = R_D、RP = A 逐局相同"属实**：逐局成败 30/30 相同；RP 失败的 20 局正是 A 失败的那 20 局。
- **偏差中位数**（cm）：
  - 预抓取：A 3.015，R_D 0.958，RE 3.015，RP 0.958。
  - 进入段：A 6.194，R_D 1.252，RE 1.357，RP 6.10。
  - RP 和 A 逐局的进入段偏差之差：中位数 −0.016 cm，范围 −0.17 到 +0.12。也就是说，预抓取修到 0.96 cm，进入段的偏差几乎不变。
- **修正量**：
  - RE 进入段中位数 5.80 cm（最大 7.17）。
  - RP 预抓取中位数 3.27 cm（最大 4.09）。
  - 缩回 0，回退 0，IK > 2 cm 0。

V1（28 个有路线的 TabletopGrasp 局）
- **成功数**：
  - 官方时限：J0 23，J_D 16，J0S 14，JDF 25。
  - 45 s：J0 28，J_D 27，J0S 28，JDF 27。
- **比较**：
  - J0S 对 J0：官方时限下少 9 局，McNemar 0.003906；布局 − 5，p 0.0625。
  - JDF 对 J_D：官方时限下多 9 局，McNemar 0.003906；布局 + 4，p 0.125。
- **原来少掉的 7 局**（0、7、10、12、16、17、22）：J0S 全丢（0/7），JDF 全收回（7/7）。
- **时长相同时**，J_D 的路径反而多 2 局：JDF 对 J0 多 2 局（第 2、6 局），J0S 对 J_D 少 2 局（第 18、26 局），p 0.5。
- **t_success 和时长一比一变化**：
  - J0S − J0：时长变化中位数 +0.231 s [+0.020, +0.739]，t_success 变化 +0.220 s [+0.040, +0.740]，两者之差中位数 +0.003 s [−0.073, +0.055]，回归斜率 0.996。
  - JDF − J_D（n 27，第 21 局两组都失败）：之差中位数 −0.001 s [−0.063, +0.058]，斜率 1.021。
  - 之差 ≤ 0.06 s（3 个控制周期）的局：27/28 和 26/27。
- **预抓取偏差中位数**（cm）：J0 3.86，J_D 2.088，J0S 3.863，JDF 2.104。JDF − J_D = +0.016 cm，没碰到判据里 0.5 cm 那一条。
- **"45 s 内成功但超过 16.0 s"的局**：J0 5 局，J_D 11 局，J0S 14 局，JDF 2 局；局号和 eval.txt 相同。
- **对照预测**：J0S 低于预测区间 15–19，JDF 高于 20–24，两边都偏向时长解释更强的一侧。

一致性
- **AJk = ALJk**：每个 rep 30/30。
  - 比较了行和每一下的全部字段，只去掉 wall_s、correct_reset、film、sonic_vr_frame 和每一下的 correct。trace 也逐位相同。
  - ALJ 的 120 下都是 dryR，发出的命令 = 目标。
- **抖动**：
  - 三组的抖动逐局相同，60/60。
  - 抽到的 |前后| 最大 5.44 cm，|左右| 最大 5.57 cm。
  - 实际走位比不抖的 A 挪了中位数 4.82 cm（x −5.3 到 5.5 cm，y −5.6 到 5.5 cm），裁剪几乎没起作用。
  - AJ 和 RDJ 在发预抓取之前的全部字段、预抓取的起点 S0，逐局相同。
- **只修一段**：
  - RE 的预抓取 = A 的，30/30；RP 的预抓取 = R_D 的，30/30（比较除 correct 外的全部字段）。
  - RE、RP 没修的那一段都没有 correct 记录。
- **V1 的路径和时长**：
  - J0S 的路径（joint_src、joint_path_cm、joint_names、joint_gap_deg）和起点 S0 = J0 的，28/28；JDF 的 = J_D 的，28/28。
  - 时长和另一组记录的 correct.dur_s 差 0。
  - joint_dur_set 记的规划、时长和时长文件逐局对得上；弧长等于本组路径的弧长，28/28。
- **plan_same**：RDJ 120/120，RE 30/30，RP 30/30，J0S 28/28，JDF 28/28。回退 0，缩回 0。
- **分组之前的失败**：0。296 行都发了两下，感知 score 最低 0.945。
- **区间端点**：4 处和 eval.txt 差一格。
  - RP 对 R_D：见上。
  - J0S 对 J0 官方时限：我得 [−53.3, −13.3]，eval [−50.0, −12.5]。
  - JDF 对 J0、J0S 对 J_D 的 45 s：我得 ±11.5，eval ±10.7。
  - 原因是 eval 的各个比较共用一条随机流，属于重抽误差。没有一条判据用到区间。

**2. 四类陷阱**
- **benchmark 选择**：
  - BendPick 是 simplecmp 里效果最大的那个任务。这一轮是专门针对它的复核，不是新任务上的证据。
  - 官方 8 s 时限下三组都是 0/60，提高只在团队自定的 45 s 时限下存在。这一点预注册里就写明了。
  - 物体摆放还是那 10 个看过的布局（E3 没做）。
- **数据泄漏**（按冻结代码逐条查，没发现）：
  - 训练集：simpledata 的 data3、data3b 和两个 mid 文件，809804 段，不含任务局。
  - 标准化统计量：Norm 的 mu / sd 和 sd_d 都是 buffer，load_state_dict 时被 checkpoint 覆盖。模板 real/task/data/all.npz（A 的任务记录）只提供形状。
  - 仿真真值：
    - D 的输入只有记录器的关节读数、IMU 和感知的信念；仿真骨盆只用作坐标原点和高度。
    - land_hp、pred_err 都在这一下结束后才写进记录。
    - features() 里用仿真骨盆算的 walk_real 不进 x0。x0 的走位尾部是 D["walk"]，测试和训练时都是 0。
  - 剩下的一点："对准手掌"和 7 个候选是看过 simplecmp 那 120 局 A 之后定的。抖动后的伸手是新的，布局不是。
- **度量误用**：
  - 45 s 和官方时限都报了，判定用的是预注册的那一个时限。McNemar 只作参考。
  - F1(b) 两下合起来是 85.0 %，但只看进入段是 47/60 = 78.3 %（ALJ 也一样），不抖动时也只有 23/30。
    - 登记的判据是两下合起来，所以判"可信"没错；但起作用的恰好是进入段，而它单独看在 80 % 线下。
    - 出范围的进入段 D 误差并不更大（RDJ1：0.706 对 1.132 cm），这 13 下全部成功。
  - 思考时间不计时（预抓取中位数 4.8 s，进入段 1.4 s），已照报。
- **事后挑选**：
  - 开跑后预测、判据、做法都没改。posthoc.txt 标了"未登记"。E1、E2、V1 的判定都按登记的判据行走。
  - offline.txt 里 "against simpledata's A record" 那一行不在登记的一致性列表里，条目说不作检查，站得住。
  - 冻结脚本在 KEY_MD5 那一步停了，后几步是手工补完的；我重核 CODE_MD5 和 INPUTS_MD5，都通过。

**抖动后的失败还是同一种**
- **偏差方向和大小**：A 抖动后失败的 31 局，进入段偏差的 y 全在 −6.84 到 −5.13 cm，三轴中位数 [0.1, −6.18, −0.12]。不抖时失败的 20 局，y 在 −6.92 到 −5.55。
- **失败的样子**：31 局全是在合手阶段把物体碰开，没有摔倒。不抖时的 20 局也都是合手阶段碰开，其中 19 局首次接触也在合手阶段。
- **边界不是一刀切**：A 成功局的 y 在 −5.86 到 −0.68，和失败局在 −5.86 到 −5.13 之间有重叠。
- **抖动改的是哪种状态出现，不是失败的种类**：A 从 10/30 变成 29/60；布局 8 从 0/3 变成 4/6；布局 6 还是 0。
- **D 的修正跟着每局状态变，不是加一个常数**：
  - 在 A 自己的命令上（ALJ），D 预测的进入段 y 偏差和实际偏差相关 0.978。失败局预测 −5.10、实际 −6.18；成功局预测 −2.72、实际 −2.42。
  - R_D 的 y 修正：A 失败的局中位数 6.23 cm，A 成功的局 3.20 cm，和 A 的偏差相关 −0.965。
  - D 对失败局的偏差少估约 1 cm：|预测 − 实际| 在失败局是 1.334 cm，成功局 0.826 cm。
- **E1 能说明的范围**：
  - 能说明：在那 10 个物体摆放上，站位挪动最多约 5.5 cm 时，R_D 修好了那一种"进入段侧偏约 6 cm"的失败，31/31，而且修正量随每局状态变化。站位几乎没转向，命令时朝向在 −3.9° 到 −0.8° 之间。
  - 不能说明：R_D 能修别种失败；也不能推到别的物体摆放或别的任务。

**3. 还能推翻它的实验**
- **X1（主）**：在没看过的 BendPick 物体摆放上，站位抖动 ±6 cm，再加起步朝向抖动 ±15°，A 对 R_D 配对，至少 10 个新布局 × 6 次。
  - 登记前先在 A 上定好失败分类：侧偏型（进入段 y 在 −7.5 到 −4.5 cm 之间）和其他型。
  - 推翻：R_D − A ≤ 0；或布局符号检验方向相反；或 R_D 进入段偏差中位数 > 2.8 cm。
  - 收窄：A 的其他型失败 ≥ 6 局，而 R_D 只修好其中 ≤ 1/3。这时结论收到"只修侧偏这一种"。
  - 证据不足：A ≥ 54/60。
  - 成立：同 E1 的成立行（≥ +5，p < 0.05，F1 可信），并按失败类型分开报。
  - 前提：先查 SIMPLE 的 BendPick 生成器能不能给出新摆放；不能的话，退一步，在现有 10 个布局上只加朝向抖动。
- **X2（归因，可选）**：R 所用的 D 起点特征 x0，换成同一 rep 里另一局的 x0（固定置换，优先把 A 成败不同的两局配对），其余不变。
  - R_shuf ≥ 57/60：提高不需要 D 对每一局的预测，"自身模型"这个归因不成立。
  - R_shuf ≤ A + 5：归因站得住。

**4. 判定**
- **E1：支持。** 60/60 对 29/60，布局 + 9 / − 0（p 0.003906），F1 三项都过，落在"成立"行；只适用于 45 s 时限、那 10 个布局和那一种侧偏失败。
- **E2：支持。** RE 和 R_D 逐局相同（30/30），RP 和 A 逐局相同（10/30），RE − A = +20 ≥ +14；预抓取修好后，进入段偏差逐局几乎不变（中位数 −0.016 cm）。
- **V1：支持。** J0S 14 ≤ 18，JDF 25 ≥ 21；少掉的 7 局 J0S 全丢、JDF 全收回；t_success 跟时长一比一（斜率 0.996 / 1.021，残差中位数 +0.003 / −0.001 s），落点不变（+0.016 cm）。
