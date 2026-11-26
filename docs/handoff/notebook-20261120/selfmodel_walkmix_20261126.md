# selfmodel_walkmix_20261126：走路数据只留一部分够不够，省下的时间全采原地值不值（nowalk 的两个后续，SIMPLE，ScaleBFM）

- **状态**：11-26 用户要求收窄范围，暂停，未开跑。第一部分（Ma、Mb、Cm、Dh）和第二部分（T 的采集和训练）都没起，H20b 不分给本实验。登记（8e06d85a）留着不撤。
- **日期**：2026-11-26。在任何正式训练和新采集之前写好，单独提交。划分检查和冒烟在提交之前做完，结果写在"开跑前的准备"里。
- **来源**：
  - [selfmodel_nowalk_20261126](selfmodel_nowalk_20261126.md) 第一阶段落在判据的中间那一行，按判据要"试一个更省的中间方案，另行登记"。第二阶段 LML 上 ScaleBFM 总体过线，但走路后的第一下也差约 30 %。
  - 设计草稿 11-26 交给协调人。协调人定了两点：中间方案用一个共用的随机对照；加一条"主比值离阈值不到 0.02 不下结论"。
  - 时间对齐那一组要新采数据，协调人 11-26 转达：用户同意。
  - 协调人 11-26 又转达：用户给了新机器 H20b，"尽量快的把所有任务跑完"。两部分的训练和 T 的采集都改到 H20b 上。
- **两部分**：
  - **第一部分（中间方案）**：从现有数据里取子集，不新采。马上可以跑。
  - **第二部分（T，时间对齐）**：用 D 的采集时间全部采原地伸手，要新采数据。在 H20b 上采，不用等 E2a。
- **机器**：H20b（`ssh -p 16093 root@14.103.60.188`，8 张 H20、171 个 CPU）。**以接入核对为准**：协调人正在核对，大约 1 小时。核对回来之前不在 H20b 上起任何东西。
  - 要核对的有：LML、SIMPLE 采集是否逐位相同，EGL，训练能不能和 H20 混着比，进程数。
  - 训练：8 张卡一卡一个，一起跑。每张卡总显存不超过 80 GB；开训练前要求已用显存加 12 GB 不超过 80 GB。
  - CPU 和采集的进程数，按接入核对给的数定。核对没给之前，容器里所有进程合计按不超过 150 核算。按每个进程 10 s 内的 CPU 时间量，开每个训练前量一次，要求量到的核数加 5 不超过上限。
  - **一组对比的训练放在同一台机器上**：Ma、Mb、Cm、Dh、T 都在 H20b 上训。除非核对证明 H20 和 H20b 的训练逐位相同，否则不和 H20 上训的模型混着当主比较。
  - H20b 的 keepalive 还没开，开了以后不停、不动。不起 Isaac：这里的采集用 MuJoCo，评估只用 CPU。
  - 冒烟和划分检查是在 H20（`ssh -p 27840`）上做的，见"开跑前的准备"。
- **代码**：`selfwm` 分支，新目录 `scripts/experiments/selfmodel_walkmix_20261126/`：
  - `wm_split.py`：nowalk 的 `nw_split.py`，只换了组；
  - `wm_train.py`：nowalk 的 `nw_train.py`，只换了组名；
  - `wm_eval.py`：nowalk 的 `nw_eval.py`，改成命令行给模型和配对，按种子配对后取平均，和第二阶段的 `nw2_eval.py` 一样；
  - `wm_run.sh`：第一部分的训练和评估流程。
  - 第二部分的采集、中间点、划分脚本，在采集完、训练之前用一个修订补上，并在修订里写清实现。
  - 共用代码取 simpledata 冻结的 `asrun/repo_b`，和 nowalk 第一阶段相同：`data.py`（174071cb，FDVEL 版）、`train.py`、`models.py`，train3b 冻结的 `t3b_train.py`（c7ecf581）。
  - 正式运行前，本实验的脚本拷进 `asrun/code/`，md5 记在 `asrun/MD5.check`。
- **输出**：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_walkmix_20261126/`，下面简称"本目录"。simpledata 和 nowalk 的输出只读。
- **关联**：
  - [selfmodel_nowalk_20261126](selfmodel_nowalk_20261126.md)：N、C、C2 的做法、第一阶段的模型和评估；
  - [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md)：SIMPLE 的数据、模型 D、120 局真实局。

## 背景

- nowalk 第一阶段（SIMPLE，ScaleBFM，4090b）：
  - 去掉先走后伸的数据（N），对同样大小的随机对照（C），真实局 N/C 是 1.161 / 1.086，落在中间那一行。
  - 差距几乎全在第一下（1.294 / 1.300），第二下看不出来。
  - 全换成原地重置只省约 17 % 的采集时间。
- nowalk 第二阶段（LML）：ScaleBFM 总体 1.055，落在第一行；但第一下 1.300，走过去后的第一下 1.362，和 SIMPLE 一样。
- D 的训练集（809 804 段，29 260 个重置）按重置类型分（nowalk 的设计草稿，`midsurvey.py`，只读）：

  | 类型 | 重置 | 段 | 这些重置的采集用时（进程·小时） | 走路距离中位数 |
  |---|---|---|---|---|
  | 原地 | 19 523 | 542 459 | 56.9 | — |
  | 任务式走法 | 4 914 | 134 411 | 17.7 | 0.39 m |
  | d3 走法（先转身再走） | 4 823 | 132 934 | 25.2 | 0.91 m |
  | 走路 ≤ 0.6 m（两种都算） | 4 648 | 127 666 | 17.8 | — |

  - 距离是日志里 `walk.asked` 的平面距离，也就是命令距离。
  - 任务式走法 70 % 在 0.6 m 以内，d3 走法只有 25 %。
  - 每个重置的平均用时：原地 9.11 s，任务式 11.65 s，d3 17.28 s（data3 型）。
- 真实局里的走路都是任务式的：走到站位，扭转小。d3 走法要先转身 30–70°，真实局里没有这种起点。
- 这里问两个问题：
  1. 走路数据只留一部分，能不能补上第一下的差距；
  2. 把采集时间全花在原地伸手上，能不能追上 D。

## 假设

1. **中间方案**：第一下的差距要靠先走后伸的数据来补，但不需要全部。只留任务式走法（Ma），或者只留 0.6 m 以内的短走（Mb），和留同样多的混合数据（Cm）一样好。
   - 理由：真实局都是任务式的短走；d3 走法的起点离真实局远，对真实局帮助小。
2. **时间对齐**：同样的采集时间全采原地（T，原地重置是现在的 1.76 倍），也追不上 D，T/D > 1.05。
   - 理由：nowalk 两个阶段的意外都说明，N 输掉的是"刚走完、停下"的那种状态。多采原地数据补不了这种状态。所以 T 的差距应该主要在第一下。

## 做法

### 第一部分：中间方案（不新采数据）

**各组**（ScaleBFM，SIMPLE，配方和 nowalk 第一阶段相同，种子 0、1；段数是划分脚本给的实际数，见"开跑前的准备"）

| 组 | 训练数据 | 训练段数 | 和 Cm 的差 |
|---|---|---|---|
| **Ma** | D 去掉 d3 走法的重置（所有桶），即原地 + 任务式走法 | 676 870 | +0.500 % |
| **Mb** | D 去掉命令距离 > 0.6 m 的走路重置（所有桶，两种走法都算） | 670 125 | −0.502 % |
| **Cm**（共用的随机对照） | D 按 md5("walkmix_cm:<重置编号>") 的顺序去掉整个训练重置，所有类型都可能被去掉，留下的段数最接近 (Ma + Mb) / 2 = 673 497.5 | 673 503 | — |
| **Dh** | D 的全部训练数据 | 809 804 | +20.2 % |

- **段数**：Ma 比 Cm 多 3 367 段，Mb 比 Cm 少 3 378 段。
  - 按 nowalk 第一阶段的数据量效应：C 只有 D 2/3 的数据，C/D 是 1.24–1.33。也就是每少 1 % 数据，误差约高 0.7–1.0 %。
  - 0.5 % 的段数差别，对比值的影响约 0.004–0.005。下面"离阈值不到 0.02 不下结论"这条规则，留出的余量大于这个数。
- **Cm 里各类型的比例**和 D 差不多：先走后伸 222 314 段（任务式 112 495，d3 109 819）。
- **为什么加 Dh**：
  - 第二部分的主比较是 T 对 D，要在同一台机器上、用同一个 val 挑 checkpoint 才公平。
  - D 本身（simpledata 的模型）是在 4090b 上训的，而且挑 checkpoint 用的是全部 val。Dh 用 D 的数据，在 H20b 上训，挑 checkpoint 用原地 val，和其他组一样。
  - 第一部分里，Ma/Dh、Mb/Dh、Cm/Dh 也是同一台机器上的比较。
- **走路标志**：四组都保留（不置 0）。nowalk 第一阶段 C2/C 是 1.046，标志本身没用，这里不再单独做。
- **挑 checkpoint**：四组都用 nowalk 的原地 val，4260 段、2534 个重置，和 nowalk 的 N、C 是同一批段（划分脚本核对过）。每组都去掉 val 桶里的先走后伸重置。
- **训练**：`wm_run.sh "0 1 2 3 4 5 6 7" 1 <CPU 上限>` 在 H20b 的 tmux `walkmix` 里跑。
  - 8 个训练：Ma、Mb、Cm、Dh × 种子 0、1，一卡一个，一起跑。每个都是 train3b 冻结的 `t3b_train.py E3_ensemble --frac 1`，迭代数用缺省值。
  - 开关：`REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1`，X0_REF 不开。nice 10，每个 4 个 CPU 线程。
  - 每次开训练前按上面"机器"一节查显存和 CPU，不满足就等。

**评估**（`wm_eval.py`，只用 CPU，跑在全部训练之后）
- **主评估**：和 nowalk 第一阶段一样，真实局主标签 settled，235 段、120 局、40 个布局。区间按布局重抽 4000 次。
- **登记的统计量**：种子 0、1 各自配对，求 Ma_k/Cm_k，再取平均；它的 95 % 区间由同样的重抽得到。每个种子自己的比值也报。Mb 同样。
- **分格**：第一下（116 段）、第二下（119 段）、去掉 TabletopGraspMP 的三个任务、每个任务。
- **一起报，不进判据**：
  - Ma/Dh、Mb/Dh、Cm/Dh（同一台机器）；
  - Ma/N、Mb/N、Cm/N、Ma/C、Mb/C。N、C 是 nowalk 第一阶段在 4090b 上训的模型，比值里混着换卡的差别；
  - Dh/D：同样的数据，换了机器、换了挑 checkpoint 的 val；
  - N/C：重算 nowalk 第一阶段的数，作核对。
- **次要评估**：data3 / data3b 的留出集，切法和 nowalk 第一阶段相同。重点看两格：任务式走法后的第一下（data3 test_iid），d3 走法后的第一下（data3 val）。
- **核对**（对不上就停）：
  - simpledata 的 D 重现它的四个中位数（|差| ≤ 2e-4 cm）；
  - nowalk 的 N0、N1、C0、C1 重现 nowalk `eval_s<k>/eval.json` 里真实局"全部"和"第一下"的中位数（|差| ≤ 2e-4 cm）。

### 第二部分：T，时间对齐，全采原地（新采数据）

**采多少**（按日志里每个重置的 wall_s 算，D 的全部 45 000 个重置，包括 val 和 test 的）
- data3 型：30 000 个重置共 90.6 进程·小时。全采原地（9.11 s 一个），同样的时间能采 35 805 个，比现有的 20 073 个原地重置多 **15 732 个**。
- data3b 型：15 000 个重置共 63.0 进程·小时。全采原地（13.31 s 一个），同样的时间能采 17 047 个，多 **7 007 个**。
- 合计新采 22 739 个原地重置，原地重置从 30 113 个变成 52 852 个，是原来的 1.76 倍。按仿真步数算也差不多，所以换一种口径算时间，量也在这附近。

**怎么采**
- 用冻结的 simpledata `asrun/repo_b`，和 data3 / data3b 同一份代码、同样的参数（perturb 0.007 rad 等缺省值），只是加 `--no-walk`，不走路：
  - data3 型：`sc_collect.py drive --controller scalebfm --no-walk --resets 15732 --id0 52000000 --seed 82000000`；
  - data3b 型：再加 `--b`，`--resets 7007 --id0 62000000 --seed 84000000`；
  - 在 H20b 上采。进程数按接入核对给的数定（预计多于 36 个）：data3 型和 data3b 型各分成同样多的片，每个进程先采 data3 型的一片，再采同号的 data3b 型。nice 10，只用 CPU，`CUDA_VISIBLE_DEVICES` 为空，每个进程 1 个线程。
- 重置编号是 52 000 000–52 015 731 和 62 000 000–62 007 006，种子是 82 000 000 起和 84 000 000 起。
  - 仓库里 grep 过，已经占用的有：重置编号 20M、30M、40M、50M、60M、90M、93M / 94M、200M 以上、9.1M / 9.2M；种子 73M、77M、79M、81M、83M。
  - 52M、62M、82M、84M 都没人用。
  - 不接着 data3 的编号往后排，是为了和以后可能的 data3 扩采分开。
- **开采前**：量一次容器里的 CPU，要求量到的核数加进程数不超过接入核对给的上限；不满足就等。
- **采集中途查一次**：新写出的日志里，重置编号是不是落在 52M / 62M 段，种子是不是 82M / 84M 起。
- **采完**：用 sc_collect 的 `merge`、`check`、`summary` 收尾，再用 simpledata 的 `sd_mid.py` 建中间点，和 D 的做法一样。
- 新编号照样由 data.py 按 crc32 分到 train / val / test 桶。

**T 这一组**
- T = N 的训练数据（D 里的原地重置）+ 新数据 train 桶里的重置 + 它们的中间点，约 95 万段，比 D 的 809 804 段还多。
- 配方和第一部分一样。走路标志置 0，和 N 相同：训练数据里没有走过的段，评估时真实局的标志也置 0。
- 挑 checkpoint 仍然只用 nowalk 的那 4260 段原地 val。新数据的 val 桶不参与挑选，和 test 桶一起作额外的留出集。
- 种子 0、1，在 H20b 的 GPU 上训，规则同第一部分。

**评估**：同第一部分。主比值是 T/Dh（种子各自配对后取平均）。一起报 T/N、T/C、T/D、T/Ma、T/Mb，以及新数据的留出集。

**要多久**
- 按每个重置 9.11 / 13.31 s 算，新采一共 65.7 进程·小时：data3 型 39.8，data3b 型 25.9。36 个进程约 1.8 小时，进程越多越快。
- 加上收尾、中间点和划分检查约 0.5 小时；训练两个种子约 15–20 分钟；评估约 10 分钟。

## 预测（启动前写，首次启动后锁定）

写预测之前已经知道的：划分脚本的输入检查（见"开跑前的准备"）。按 39 个起点量算，真实局第一下的起点落在训练范围里的比例：Dh 和 Cm 100 %，Mb 98.3 %，Ma 90.5 %（nowalk 的 N 是 84.5 %）。nowalk 第一阶段里，N 在范围外的段误差反而更小，所以这个比例不一定预示误差，下面的预测没有因此收窄。

**第一部分**

| 项 | 预测 | 理由 |
|---|---|---|
| 训练时间 | 每个 8–20 分钟 | 第二阶段 LML 的训练是 8–14 分钟，数据量相近；这次 8 个一起跑，一卡一个，CPU 可能成为瓶颈 |
| 挑 checkpoint 的原地 val，Ma/Cm、Mb/Cm | 0.98–1.02 | 三组原地数据的量差不多 |
| **Ma/Cm，全部 235 段** | **0.95–1.12**，最可能约 1.03 | 真实局都是任务式走法，d3 走法帮不上忙；Ma 的原地数据和 Cm 差不多 |
| **Mb/Cm，全部 235 段** | **0.97–1.15**，最可能约 1.05 | Mb 少了长距离的任务式走法（任务式里有 30 % 超过 0.6 m） |
| Ma/Cm，第一下 | 0.92–1.20 | 差距在第一下，比值也在第一下摆得最开 |
| Mb/Cm，第一下 | 0.95–1.25 | |
| Ma/N，第一下 | 0.70–0.95 | 补回 N 在第一下上的大部分差距（nowalk N/C 1.29–1.30）。这里混着换卡的差别 |
| Cm/Dh，全部 | 1.00–1.25 | Cm 只有 D 83 % 的数据；nowalk 里 2/3 的数据 C/D 是 1.24–1.33 |
| Dh/D，全部 | 0.80–1.25 | 只差机器、挑 checkpoint 的 val 和种子；nowalk 里两个 C 种子在真实局上就差了 8 % |
| data3 test_iid，任务式走法后的第一下，Ma/Cm | 0.90–1.10 | Ma 的任务式数据比 Cm 多 |
| data3 val，d3 走法后的第一下，Ma/Cm | 1.20–2.50 | Ma 完全没有 d3 走法的数据，nowalk 里 N 在这一格是 2.17 |

- 两个种子的平均，95 % 区间宽度约 ±0.12–0.20。nowalk 第一阶段单个种子的区间是 −0.19 / +0.36。
- 落在判据各行的把握：
  - Ma：第一行 50 %，中间（包括贴线）40 %，> 1.25 10 %；
  - Mb：第一行 45 %，中间 45 %，> 1.25 10 %。

**第二部分**

| 项 | 预测 | 理由 |
|---|---|---|
| 采集用的 CPU 时间 | 55–85 进程·小时 | 按每个重置 9.11 / 13.31 s 算是 65.7；H20b 的 CPU 和 4090 / 4090b 不同，用时可能偏一些 |
| 新数据的段数 | 每个原地重置 25–30 段（D 的原地重置平均 27.8 段），合计 57–68 万段，含中间点 | |
| **T/Dh，全部 235 段** | **1.05–1.30**，最可能约 1.2 | 按 val 的学习曲线（指数约 0.26），原地数据多 1.75 倍，误差约降到 0.86 倍：nowalk 的 N/D 1.44 × 0.86 ≈ 1.24 |
| T/Dh，第一下 | 1.15–1.50 | 多采的原地数据补不了"刚停下"的状态 |
| T/Dh，第二下 | 0.90–1.20 | 第二下 N 本来就不差 |
| T/N | 0.75–0.95 | 原地数据多 1.75 倍；混着换卡的差别 |
| 挑 checkpoint 的原地 val，T/Dh | 0.85–0.98 | T 的原地数据比 Dh 多 |

- 落在判据各行的把握：T 第一行 10 %，中间 35 %，> 1.15 55 %。

## 判据（什么结果会让我放弃或修改假设）

**两部分共用**
- **停下的情况**：核对失败（D0 的四个中位数，或 nowalk 的 N0、N1、C0、C1 对不上），或划分检查失败。停下查清，不看比较结果。
- **登记的统计量**：两个种子各自配对的比值取平均，配它按布局重抽的 95 % 区间，用全部 235 段。
- **附加规则 1，种子**：两个种子各自的点估计要和平均落在同一行（按点估计和它自己的区间判断），否则降为中间那一行：补训种子 2、3，再按四个种子的平均判。
- **附加规则 2，第一下**：第一下（116 段）用同样的阈值单独判一次。如果它落在比全部 235 段更差的一行，结论取第一下那一行。原因是 nowalk 的差距就在第一下。

**第一部分**：Ma 和 Mb 各自对 Cm 判。

| 结果 | 结论和下一步 |
|---|---|
| **Ma/Cm ≤ 1.10 且区间上限 ≤ 1.25** | 只保留任务式走法就够。以后采数据可以去掉 d3 走法：同样的数据量，采集时间约是 D 的 75 %（74.6 对 99.8 进程·小时） |
| **Ma/Cm > 1.25** | d3 走法有用，要保留 |
| 其余 | 不下结论，补训种子 2、3 再判 |

- Mb 同样判。Mb 过线的意思是：只留 0.6 m 以内的短走就够，两种走法都可以。
- **附加规则 3，离阈值太近不下结论**：主比值的点估计或区间上限，离它要比的阈值（1.10 或 1.25）不到 0.02，这一组就不下结论，记为"贴线"，补训种子 2、3 再判。这一条也用在附加规则 2 的第一下上。原因是 Ma、Mb 和 Cm 的段数相差约 0.5 %，不完全相同。
- 如果 Ma 和 Mb 都落在第一行，建议用 Ma：采集用时一样（74.6 对 74.7 进程·小时），但 Ma 只需要去掉 d3 这一种走法，不用按距离筛。
- Ma/N、Mb/N 这些混着换卡差别的比值只报，不进判据。

**第二部分**：T 对 Dh 判。

| 结果 | 结论和下一步 |
|---|---|
| **T/Dh ≤ 1.05 且区间上限 ≤ 1.15** | 同样的采集时间全采原地，不比 D 差。以后 SIMPLE 里 ScaleBFM 的自身模型数据改成只采原地，重置数按时间对齐 |
| **T/Dh > 1.15** | 先走后伸的数据换不成更多的原地数据。保留走路数据；如果第一部分有一组过线，就用那一组的采法 |
| 其余（1.05 < T/Dh ≤ 1.15，或 T/Dh ≤ 1.05 但区间上限 > 1.15） | 差得不多。按第一部分决定：第一部分有一组过线，就用那一组的采法；都没过线，不下结论，补种子 |

- 附加规则 1、2 照用。第一部分的附加规则 3 不用在 T 上：T 的大小是按时间对齐有意定的，不是要和 D 一样大。
- 采集中途查编号时，如果重置编号或种子不在 52M / 62M、82M / 84M 段，就停下，已采的数据不用。

## 预测修订
<!-- 首次启动后，预测/判据已锁定。在此追加修订，写明时间和原因，不要改上面两节。 -->

## 开跑前的准备（2026-11-26，正式训练之前）

三件事都在 H20（`ssh -p 27840`）上做，只读 simpledata 和 nowalk 的输出，只写本目录。H20b 上还没有起任何东西。

### 1. 划分检查（只读，H20 CPU）

`wm_split.py --nowalk $P/run_b/splits.npz --out 本目录/splits`。完整输出在 `本目录/splits/split.log`，下面原样摘出检查和各组的行（输入检查那一段每组只摘第一行）：

```
logs: 45000 resets, walk-first 14887 (task 7512, d3 7375; asked > 0.6 m 7814)
checks:
  [ok] every segment's reset is in the collection logs
  [ok] segment walked == the log's walked, all 851294 segments (data3, data3b, mid)
  [ok] segment walk_task == the log's kind 'task', all 851294 segments
  [ok] mid_data3_src rows 480254 == mid_data3 segments 480254
  [ok] mid_data3: reset and walked equal to the source segment's (mid_*_src.npz)
  [ok] mid_data3b_src rows 280150 == mid_data3b segments 280150
  [ok] mid_data3b: reset and walked equal to the source segment's (mid_*_src.npz)
  [ok] x0 column 204 == walked
  [ok] x0 column 205 == chain
  [ok] x0 columns 274-277 (REACHWM_APPROACH) zero everywhere
  [ok] D's training split 809804 == 809804 (simpledata run.log)

arms (training split after the arm's drop; val = data.splits' val, fell excluded as train.train does):
  Dh: train 809804 segments (29260 resets; first 306654, chained 503150), walk-first 267345 segments / 9737 resets (task 134411, d3 132934); ... val 4260 (2534 resets, walk-first 0); test_iid 6130; drop 1392 resets; zero walk flag False
  Ma: train 676870 segments (24437 resets; first 254647, chained 422223), walk-first 134411 segments / 4914 resets (task 134411, d3 0); ... val 4260 (2534 resets, walk-first 0); test_iid 6130; drop 7375 resets; zero walk flag False
  Mb: train 670125 segments (24171 resets; first 251934, chained 418191), walk-first 127666 segments / 4648 resets (task 94906, d3 32760); ... val 4260 (2534 resets, walk-first 0); test_iid 5552; drop 8144 resets; zero walk flag False
  Cm: train 673503 segments (24342 resets; first 255816, chained 417687), walk-first 222314 segments / 8093 resets (task 112495, d3 109819); ... val 4260 (2534 resets, walk-first 0); test_iid 6130; drop 6310 resets; zero walk flag False
  [ok] Dh's training split 809804 == D's 809804
  [ok] Ma: 676870 training segments, 0 from d3-walk resets
  [ok] Mb: 670125 training segments, 0 from walks asked > 0.6 m
  [ok] every arm: val (checkpoint pick) in place only
  [ok] every arm picks on the same val resets and segments
  [ok] the val is nowalk's N / C val: 4260 segments == 4260
  [ok] Cm's training split 673503 vs target (Ma + Mb) / 2 = 673497.5: +0.0008 % (tolerance 0.1 %)
  sizes: Ma 676870 (+0.500 % vs Cm), Mb 670125 (-0.502 % vs Cm), Cm 673503, Dh 809804
  [ok] real main: x0 columns 274-277 zero
  [ok] real main settled: walked flag 1 on every segment

input check, real main settled (235 segments; first 116, second 119): share inside the arm's training p0.5 / p99.5 (sd_range's 39 quantities), and x0 columns under Norm's floor
  Dh: joints 86.8 % (first 100.0, second 73.9); pelvis 98.7 % (first 100.0, second 97.5); hands 100.0 % (first 100.0, second 100.0); all 39 86.8 % (first 100.0, second 73.9)
  Ma: joints 81.7 % (first 90.5, second 73.1); pelvis 98.7 % (first 100.0, second 97.5); hands 100.0 % (first 100.0, second 100.0); all 39 81.7 % (first 90.5, second 73.1)
  Mb: joints 85.5 % (first 98.3, second 73.1); pelvis 98.7 % (first 100.0, second 97.5); hands 100.0 % (first 100.0, second 100.0); all 39 85.5 % (first 98.3, second 73.1)
  Cm: joints 86.8 % (first 100.0, second 73.9); pelvis 98.7 % (first 100.0, second 97.5); hands 100.0 % (first 100.0, second 100.0); all 39 86.8 % (first 100.0, second 73.9)

SPLIT CHECK PASS
78fc77cb7164b08041f5061b7ccf7d73  /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_walkmix_20261126/splits/splits.npz
```

- `...` 处删掉的是各组按文件的段数，原文在 split.log 里。
- 训练日志里 val 是 4263 段，划分检查是 4260 段：差的 3 段是摔倒的段，`train.train` 挑 checkpoint 时去掉，和 nowalk 的 N 一样（nowalk N_s0 的训练日志也是 `'val': 4263`）。
- Mb 的 test_iid 一栏少了 578 段（5552 对 6130），是划分脚本按组去掉重置以后报的数：去掉的走得远的重置里有一些在 test_iid。训练不用 test_iid；评估的 held-out 一节用 `data.splits` 原来的 val / test_iid，和组无关，各组评的是同一批段。

### 2. 冒烟（H20，GPU 0、1 各一个训练，2026-11-26 11:54:51–11:59:12Z）

```
# notebook: skip smoke of the walkmix runner (300 iterations, checkpoints deleted after)
tmux new -d -s walkmix_smoke 'WM_TEST=1 WM_OUT=$Q/smoke WM_ITERS=300 WM_JOBS="Ma:0 Cm:0" WM_CODE=$Q/code bash $Q/code/wm_run.sh > $Q/smoke_tmux.log 2>&1'
# $Q = 本目录
```

- 这是旧版 `wm_run.sh`（只认 H20 的卡 0、1）。之后改成命令行给卡号、每卡训练数和 CPU 上限，训练和评估的部分没改，冻结的是改过的版本。
- `WM_TEST=1` 跳过 md5 检查，`WM_ITERS=300` 只训 300 步。

训练日志（原样）：

```
logs/train_Ma_s0.log:2:851294 segments, splits {'train': 676870, 'val': 4263, 'test_iid': 6130, ...}
logs/train_Ma_s0.log:4:E3_ensemble frac 1.0: best val median 4.246 cm (10.1 s) -> .../smoke/models_Ma_s0/E3_ensemble_f1
logs/train_Cm_s0.log:2:851294 segments, splits {'train': 673503, 'val': 4263, 'test_iid': 6130, ...}
logs/train_Cm_s0.log:4:E3_ensemble frac 1.0: best val median 4.302 cm (7.7 s) -> .../smoke/models_Cm_s0/E3_ensemble_f1
```

评估的复现检查（`本目录/smoke/eval.txt`，原样）：

```
  reproduction: D0 median 0.70526 vs simpledata 0.70526 (+0.00000)
  reproduction: N0 all median 1.01416 vs nowalk eval_s0 1.01416 (+0.00000)
  reproduction: N0 first reach (pre-grasp) median 1.04411 vs nowalk eval_s0 1.04411 (+0.00000)
  reproduction: C0 all median 0.87344 vs nowalk eval_s0 0.87344 (+0.00000)
  reproduction: C0 first reach (pre-grasp) median 0.80674 vs nowalk eval_s0 0.80674 (+0.00000)
  reproduction: N1 all median 1.02078 vs nowalk eval_s1 1.02078 (+0.00000)
  reproduction: N1 first reach (pre-grasp) median 1.14668 vs nowalk eval_s1 1.14668 (+0.00000)
  reproduction: C1 all median 0.94012 vs nowalk eval_s1 0.94012 (+0.00000)
  reproduction: C1 first reach (pre-grasp) median 0.88200 vs nowalk eval_s1 0.88200 (+0.00000)
  reproduction: D0 median 0.82076 vs simpledata 0.82076 (+0.00000)
  reproduction: data3 val all D0 median 1.88601 vs simpledata 1.88601 (+0.00000)
  reproduction: data3 test_iid all D0 median 1.57564 vs simpledata 1.57564 (+0.00000)
REPRODUCTION PASS
```

- 冒烟里 Ma0、Cm0 只训了 300 步，eval.txt 里它们的数（比如 Ma/Cm 0.85–0.99）不含任何信息，不看、不用。冒烟只证明流程走通、复现检查能过。
- 冒烟的 checkpoint（`smoke/models_Ma_s0`、`smoke/models_Cm_s0`）11:59Z 删了，日志和 eval.txt 留着。

### 3. 冻结

`本目录/asrun/code/` 是正式运行用的脚本，`asrun/MD5.check`（FROZEN_UTC 2026-11-26T11:59:45Z），和本分支 `scripts/experiments/selfmodel_walkmix_20261126/` 里的文件 md5 相同：

```
9bc9d4f764b6fe1e4ac610d9dea475c4  code/wm_eval.py
bce77db3d55df8ebf3e9a68df1826344  code/wm_run.sh
1e5d2ef5ca95bddadfb39f2b8155baa8  code/wm_split.py
20d53fa298f20e4381c5e4c8388a5829  code/wm_train.py
78fc77cb7164b08041f5061b7ccf7d73  ../splits/splits.npz
```

- 正式运行时 `wm_run.sh` 先查这份 md5，再查 simpledata 的 `MD5.check`、`MD5_r2.check` 和 `t3b_train.py`（c7ecf581），再查 split.log 的 PASS。任何一项不对就不训。
- H20b 上要用的话，本目录在 vePFS 上，H20b 挂了 vePFS 就直接用；没挂就拷过去，拷完重新核 md5。这一点等接入核对。

## 结果（贴原始输出，不转述）
<!-- 取指标的命令 + 原始输出；完整日志、checkpoint、评测输出的路径。 -->

## 意外
<!-- 预测和结果哪里对不上？有什么没想到的现象？ -->

## 结论与下一步
<!-- 每个论断引用上面的证据；写明证据强度（运行次数、种子、episode 数）；至少一个其他解释；
     四类陷阱检查：benchmark 选择 / 数据泄漏 / 指标误用 / 事后挑选；结果落在判据哪一侧、下一步做什么。 -->

## 独立复核
<!-- 结论要写进论文、用来否掉或投入一个方向、或作为发现报告时必须做。贴独立 agent 的原始回复。 -->
