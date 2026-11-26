# selfmodel_simplesonicdata_20261126：在 SIMPLE 里给 SONIC 采原地数据（停稳 5 cm/s），训自身模型 D_S，当天在真实局上检验，过了再跑 A 对 R_D

- **日期**：2026-11-26。在任何正式采集之前写好，单独提交。
- **用户的决定（11-26，协调方转达）**："只采原地，放宽到 5 cm/s"，和 HumanoidArena（HA）的 SONIC 相同。**停稳阈值放宽到 5 cm/s 是用户 11-26 批准的。**
- **为什么要采**：[selfmodel_simplesonic_20261126](selfmodel_simplesonic_20261126.md) 的 F1-0 里，现有的两个 SONIC 自身模型（都是在 LocoManiLab 上训的）在 SIMPLE 任务局的起点上都没过：四个任务合起来在训练范围里的比例，D_sonic 58.8 %，F_sonic 45.4 %，每个任务都低于 80 %。坏在进入段：D_sonic 进入段只有 32.5 %，BendPick 的进入段 3.3 %。所以 SIMPLE 上没有可信的 SONIC 自身模型，SONIC 的 R_D 无从检验。
- **这一条做什么**：照 [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md) 的配方，只采原地那部分，在 SIMPLE 里给 SONIC 采数据，训模型 D_S。当天在 SIMPLE 的真实任务局上检验 D_S：预测误差，以及起点在不在训练范围里（闸门）。闸门过了，当天跑 A 对 R_D，同机按局配对。
- **和别的条目的关系**：
  - 采集、训练、留出评估：照 simpledata。不同的只有三处：控制器换成 SONIC，只采原地，停稳规则换成 HA 的 SONIC 规则（见"采集"）。
  - 停稳规则和额外记录：照 [selfmodel_hadata_sonic_20261126](selfmodel_hadata_sonic_20261126.md) 和 hc_ha.py：5 cm/s，最多等 6 s，参考保持 7.5 s。
  - 只采原地：[selfmodel_nowalk_20261126](selfmodel_nowalk_20261126.md) 的复核对 SONIC 判"支持"（协调方 11-26）。
  - 闸门、F1(a) 的算法、修订 1 的读法：照 [selfmodel_simpleheft_20261126](selfmodel_simpleheft_20261126.md)。
  - R 规则、F1 的三项、判定集和检验：照 [selfmodel_simplecmp_20261126](selfmodel_simplecmp_20261126.md)。
  - F1-0 的规则和脚本 `ss_f1pre.py`：照 simplesonic。
- **机器**：H20b（`ssh -p 16093`），只用这一台。
  - 采集只用 CPU，nice 10，一个进程一个线程。
  - 训练和任务局的渲染用 H20b 的卡。每张卡总量 ≤ 80 GB，keepalive 每卡约 900 MiB 算在里面。
  - 主指标的 SONIC 行是 4090 上网络放 GPU 跑的。CPU 和 GPU 的结果本来就不逐位相同（simple_relook），所以 A 在 H20b 上用 CPU 网络重跑，R_D 只和这次的 A 配对。
  - **核数**：协调方给 SIMPLE 的两格（HEFT、SONIC）合计 ≤ 140 核，按我们自己进程的 CPU 时间算；整机我们 ≤ 160 核，另外 20 核留给 HA 训练。本条采集用 90 个进程，约 90 核（开发时 40 个进程实测 40.0 核）。HEFT 那一格过了闸门，不在 H20b 采数据。`checks/ssd_cpu.py` 每 60 s 记一次两格的 CPU 时间（`checks/cpu.log`）。
- **代码**：分支 `sonic-selfmodel`，叠在别人未提交的改动上。改了两个未提交的文件，都是默认关闭：
  - `selfmodel_simplecollect_20261126/sc_collect.py`（md5 改前 f69baf4b，改后 7b011795）：
    - 加 `--settle-v`、`--settle-max-s`、`--hold-s`。给了其中任何一个，就照 hc_ha 的做法改写 d3_collect 的模块常量，并给每段加 hc_ha 的三个额外字段。都不给时，规则和输出都不变（"登记前核对"第 1 条）。
    - check 的第 (2) 项：放宽规则时，"结束标签处本体 FK 和记录的手点差 ≤ 2e-4 m"这条只在 2 cm/s 下也停稳的标签上查，所有停稳标签的最大差照打印、不判。理由：记录的 xpos 比关节晚一个 5 ms 子步，这条界是按 2 cm/s 的手速定的；5 cm/s 时手还在动，开发数据上所有停稳标签的最大差是 0.52 mm，超过了界，但这说明的是手在动，不是记录错了。没有放宽规则的数据，这一项不变。
  - `selfmodel_simplecmp_20261126/scmp_correct.py`：`--correct` 放开 SONIC。SonicMotor 沿用 Motor 的 reach、_blend 和参考逻辑，只改参考怎么交给控制器，所以记录器、候选命令的规划、手点的换算都是同一份代码。另外把本条的 D_S 登记为 FDVEL = 1。只有给了 `--correct` 才走到这里。md5 改前 d80b3ed1。
  - 本条自己的脚本在 `scripts/experiments/selfmodel_simplesonicdata_20261126/`，前缀 `ssd_`。`ssd_eval.py` 是 simpleheft 的 `sh_eval.py`（含它的修订 1）换成 D_S 的文件名，F1(b) 换成 sd_range 的三项。
  - **用到的未提交文件**（不在本次提交里，md5 前 8 位；冻结副本的 KEY_MD5 有全文 md5）：sc_collect.py 7b011795、py4090b.py 23664f5d（simplecollect）；scmp_correct.py ac39342c、scmp_eval.py 297ff474、scmp_range.py 15098f3d（simplecmp）；sd_mid.py 5c46a94d、sd_eval.py 500927fc、sd_real.py 3703c1aa、sd_mid_check.py bb940b71、sd_split_check.py 86d68705（simpledata）；sx_collect.py cd290c2e、sx_eval.py 81f61381（xfer）；d3_collect.py dc92703a；py4090b_rl.py 49c82ebc（simple_relook）；simple_bench.py 8e40e961（有别人未提交的改动）。
  - 开跑前用 `ssd_freeze.sh` 把代码抄进 H20b 上的 `asrun/code`，记下 md5。冻结副本里的 `outputs/simple_eval_20261112` 是副本自己的真目录，`scene/` 也是真目录，其余条目只读地链接到共享树（协调方 11-26 要求）。开跑前用 `find -type l` 和 realpath 核对。
- **输出**：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplesonicdata_20261126/`（H20b 上的 realpath 就是这个路径）。

## 假设

1. **数据假设**：在 SIMPLE 里原地采、停稳阈值放宽到 5 cm/s，训出的 D_S 在 SONIC 的 SIMPLE 任务局上可信，至少一个任务过闸门：起点在训练范围里 ≥ 80 %，预测误差不超过 D_S 自己 val 误差的 1.3 倍。
   - 理由：LocoManiLab 模型的起点出界，最常出界的是右髋 pitch、腰 roll、腰 pitch、右肘，也就是 SONIC 在 SIMPLE 里伸手时躯干和手臂的姿势。这些姿势在 SIMPLE 自己的原地伸手里都有。
   - 可能不成立的地方：任务局的起点是走到桌前以后的姿势，原地数据里没有走路。nowalk 的复核对 SONIC 判"支持"只采原地，但这在 SIMPLE 上没有直接量过。
2. **主假设**（闸门过了才问，同 simplesonic）：动作和控制器都不变，只把 A 的两下直线命令（预抓取和进入）换成在 D_S 上反解出来的命令 R_D，SONIC 的手落得更准、成功更多。
3. **增益上限**：一次性修正只改目标，修不了站位偏。simple_relook 里 SONIC 的 91 局 A 失败，只有 16 局站位偏差和里程计偏差都小。
4. **翻转噪声大**：网络从 GPU 换到 CPU，什么都不改，A 就有 20/19 局翻转（simple_relook）。所以配对检验的检验力低。

## 做法

### 采集（H20b，CPU）

- **采集器**：`sc_collect.py drive --controller sonic`。SIMPLE 的宿主、抽样、分段和检查都照 simpledata，SONIC 的接法是 sx_collect 的：`REACH_SONIC_NETS=allcpu`，`REACH_SONIC_THREADS=1`，`vr_frame="robot"`。起始扰动 0.005 rad，参考延迟 4 帧，都是 sc_collect 给 SONIC 的缺省。
- **只采原地**：`--no-walk`。抽样的随机数照旧，抽到"先走后伸"的重置也不走，原地伸手。simpledata 里另一半"转身再走"和"任务式走法"都不采。
- **停稳规则**（用户 11-26 批准；照 HA 的 SONIC）：`--settle-v 0.05 --settle-max-s 6 --hold-s 7.5`。
  - 骨盆和两只手点都慢于 5 cm/s，持续 0.5 s（25 步），才算停稳。simpledata 是 2 cm/s。
  - 最多等 6 s，等不到就跳过下一下伸手。simpledata 是 3 s。
  - 每次拼接后参考保持 7.5 s。保持时间必须长过参考延迟加最长的等待，simpledata 是 4.5 s 对 3 s。
- **额外记录**（hc_ha 的 `settle_fields`，每段都记）：
  - `settled_2cm`：这一段的结束标签在 2 cm/s 下也算停稳；
  - `jit_pp_cm`、`jit_rms_cm`：结束标签前 25 步里手点的最大两点距离和到均值的 RMS，即"抖动噪声"。
- **数量**：照 simpledata。data3 式 30 000 次重置，data3b 式 15 000 次重置（四种伸手照 data3b）。
- **编号**（和已有的都不重叠；已用过的百万段最大到 107M，另有 200M）：
  - data3 式：重置号 140 000 000 起，种子 142 000 000 起；
  - data3b 式：重置号 141 000 000 起，种子 143 000 000 起；
  - 开发：重置号 144M / 145M，种子 146M / 147M，不进正式数据；
  - 任务局的记录：95M + 1M × 41（AL）/ 42（RD），即 136M、137M。
- **进程**：data3 式 54 片，data3b 式 36 片，共 90 个进程。片数按开发时 40 个进程下量的每次重置用时分（data3 式中位数 30.5 s，data3b 式 41.5 s），让两边差不多同时跑完。每个进程约 1 核、约 2.85 GB 内存。
- **采完以后**：所有片的退出码都是 0，才对两份数据做 merge、check、summary。check 要有 "CHECK PASSED" 和 "SIMPLE CHECK PASSED"。simpledata 的"任务式走法扭转"判据不适用，因为不走路。

### 训练和留出评估（H20b 的两张卡）

- **训练**：照 simpledata 的 D。
  - t3b_train.py E3_ensemble，3 万步，batch 1024，5 个成员；
  - 数据：data3、data3b，以及 `sd_mid.py --backend sonic` 做的两份中间点；
  - `REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1`，seed 0；
  - frac 1.0 是 D_S，frac 0.3 是学习曲线的对照，两个并排训。
  - 训练前照 simpledata 做 t3b_mid 和 sd_mid 的对照检查、划分检查。
- **评估 1**：sd_eval.py，SIMPLE 自己的 val、test_iid 等划分。"SIMPLE val" 是 data3 val 全部伸手的位置误差中位数。
- **评估 2**：xfer 的 sx_eval.py 在 xfer 的 SONIC·SIMPLE 记录上（`selfmodel_xfer_20261125/sonic_simple`，400 次重置），settled 和 ended 两种保留。这是"换一批 SIMPLE 原地伸手"的测试。

### 真实任务局（H20b，当天）

- **局**：主指标的测试局。四个抓取任务（XMovePick、XMoveBendPick、BendPick、TabletopGrasp），每个 30 局（level 0–2 × index 0–9），局号和种子都和主指标相同。
- **跑法**：
  - 感知栏，`--bench-set controller=sonic --bench-set sensor=rgbd`。"那一眼"是每局缓存好的 RGB-D 结果，和其他控制器读同一份文件。
  - simple_bench 的 A 选项和 bench_via.sh 一样，加 `--trace`，不录像。`REACH_HAND_OFF=0.0415,0,0`。
  - SONIC 三个网络都放 CPU（`REACH_SONIC_NETS=allcpu`），卡只用来渲染。H20b 接入核对里，这样跑的 SIMPLE 局和 4090b 的 CPU 记录逐位相同。
  - 树：副本自己的 `outputs/simple_eval_20261112`。阶段跑完后，列出共享树里比阶段开始时间新的文件，应该没有。
- **组**：
  - **A**：直接执行，不加 `--correct`；
  - **AL**：A 加 `--correct D_S --correct-rule dryR`。发出去的还是 A 的命令，只把 D_S 对落点的预测记下来，所以 AL 必须和 A 逐局相同；
  - **R_D**（过闸门才跑）：`--correct D_S --correct-rule R`。R 规则照 simplecmp：6 步不动点迭代，对准手掌，目标收缩到 IK 误差 ≤ 2 cm，损失里加 λ = 1.0 乘集成的分散度。
- **先后**：阶段 A（A 和 AL，每任务先 A 后 AL，每个作业 5 局，8 张卡各 5 个进程）→ 一致性 → 闸门 → 过了当天跑阶段 R（R_D，120 局）→ F1 → 评估。这一串由 `ssd_chain.sh` 接着采集自动跑，每一步的停止条件写在脚本里，和下面的判据相同。

## 闸门（阶段 A 跑完后算；规则写在采集开始之前）

按任务判，两项都满足才算"过"：

1. **F1-0：起点在训练范围里 ≥ 80 %。**
   - `ss_f1pre.py` 原样用。起点是本条 A 栏 120 局每局预抓取和进入两下的 `S0`，即下命令时的 29 个关节角。
   - 训练范围：D_S 自己的训练划分（`data.splits` 的 train），每段起点行的 29 个关节，每个关节取 0.5 % 和 99.5 % 分位。29 个都在里面才算"在范围内"。
   - 按任务、两下合起来算比例。预抓取和进入分开的数一起报。
2. **F1(a)：真实局上的预测误差 ≤ 1.3 倍。**
   - 数据：AL 记录的每一下伸手，在任务标签处（sd_real 的规则），不含摔倒的。
   - 算法：`sh_err.py` 原样用。D_S 对落点的预测（13 cm 点）和实际落点的距离取中位数，除以 D_S 自己 train.json 里的 val 中位数。
   - 按任务、两下合起来算。分开的数、直接执行的偏差、模型 / 直接一起报。区间按布局（任务，index）重抽 1000 次，seed 0。

**结论**：至少一个任务"过"，记 GATE PASS。四个任务都不过，记 GATE FAIL。

**另外报，不进闸门**：
- F1(b)：AL 每一下的起点按 sd_range 的三项检查（39 项、最近邻、伸手表）算在范围里的比例（`scmp_range.py`）。D_S 有 FDVEL，所以这项能算。
- F1(a) 再按"这一下有没有碰到桌子"分开算一遍（行里的 `hits`），和 simpleheft 一样。

## 局和判定集

照 simpleheft 和 simplecmp：

- **布局**：同一个 index 的三个 level 几乎一样，所以 120 局是 40 个布局。配对检验在布局上做。
- **排除**：感知失败、看完以后发预抓取之前规划失败、没有那一眼，这几类在两组分开之前就失败了，不进判定集。两组分开之前就失败的局超过一半的任务，整个不进判定集。
- **到顶**：A ≥ 24/30 的任务不进那个时限的判定集，只作描述。主指标 SONIC A 每个任务都在 11/30 以下，预计没有任务到顶。
- **官方时限**：BendPick 整局约 12 s，超过 8 s 的官方时限，各组都不可能成功。它留在判定集里，但不贡献差别。

## 度量

- **主比较**：R_D 对 A，四个任务合计，45 s 和官方时限都报。旁边同时报只算过闸门任务的合计。每个任务都注明闸门过没过。这是协调方给 simpleheft 定的读法（修订 1），用户对 HA 也这样定："照跑对比，把检验结果注在旁边"。
  - 检验：每个布局比三个 level 的成功数，做精确的双侧符号检验，p < 0.05。McNemar 一起报，不作判据。成功率差的 95 % 区间按布局重抽 4000 次，seed 0。
- **P1**：判定集上的同一检验，45 s。预计判定集就是四个任务，那时 P1 和主比较是同一个数。
- **按任务**：四个任务各自报 A、R_D 的成功数，两个时限都报，加同样的检验和区间。
- **F1，模型在修正后的命令上还可信吗**：按任务算三项，任何一项不满足就判"不可信"：
  - (a) D_S 在 R_D 记录上的误差中位数除以它的 val 中位数 > 1.3；
  - (b) R_D 起点按 sd_range 三项检查在范围里的比例 < 80 %（simplecmp 的定义），F1-0 的比例一起报；
  - (c) 发出命令里 IK 误差 > 2 cm 的比例，比 AL 高 ≥ 10 个百分点。
  - AL 的三项一起报，作参照。
- **成功差记到谁名下**：只有闸门和 F1 都判"可信"的任务，成功差才记到模型名下。
- **机制**（"SONIC 的手"这个问题，见 simplesonic）：
  - 每一下手掌偏差的中位数，A 和 R_D，按任务、按段；
  - D_S 对发出命令的预测误差；
  - 模型 / 直接（AL）：D_S 的误差除以直接执行的偏差。越小，手偏里能从下命令时的状态预测的部分越大；
  - 手臂碰桌子的伸手数，修正量，缩回次数，思考时间。
- **采集本身**：每次重置的单手可用段（没摔、停稳），按 data3 式、data3b 式；只在 5 cm/s 下才算停稳的段占多少（`settled_2cm` 为假的）；抖动噪声 `jit_pp_cm` 的中位数；摔倒和 lost 的比例；每次重置的用时。
- **一致性**：
  - AL 和 A 逐局相同。比较时去掉 `wall_s`、`correct_reset`、`film`、各段的 `correct` 字段和 bench 记录里的 `sonic_vr_frame`，和 simplecmp 的 `clean()` 一样；
  - plan_same 全为真；
  - 在线、离线预测之差 < 1e-4 m（日志按 1e-5 取整）；
  - 那一眼和主指标 SONIC A 栏同一局是否相同，只作描述：主指标是 GPU 网络跑的，站位不同，看到的就可能不同；
  - 共享树没被写。

## 预测

**采集**（开发的 20 + 20 次重置看过以后写的，见"登记前核对"第 2 条）
- 每次重置的单手可用段：data3 式 0.9–1.4，data3b 式 1.7–2.4。开发时是 1.10 和 2.10（各 20 次）、1.07 和 2.00（各 200 次）；HA 放宽以后是 1.4 和 1.67。
- 只在 5 cm/s 下才算停稳的段（`settled_2cm` 为假）：90–100 %。开发时是 95 % 和 98 %、99.1 % 和 99.5 %。
- 抖动噪声 `jit_pp_cm` 的中位数：0.4–1.0 cm。开发时是 0.65 和 0.67 cm。
- 摔倒加 lost：8–25 % 的重置。开发时是 20 % 和 15 %（各 20 次）、10.5 % 和 12 %（各 200 次）。
- 墙钟：90 个进程 5–7.5 h。估法：40 个进程时每次重置 30.5 s 和 41.5 s，比 4 个进程时慢 1.35 倍和 1.27 倍；H20b 接入核对的吞吐表里，90 个进程比 45 个再慢约 1.2 倍；合计约 184 万核·秒，90 个进程约 5.7 h。HEFT 也在 H20b 上采时取上限：进程数过了 90 个物理核，每个进程会再慢约 30 %。
- 单手可用段合计：5–8 万。开发的比例推出来约 6.5 万。LML 上 SONIC 的 data3 是 4.66 万。

**模型**
- SIMPLE val 中位数：2.0–4.0 cm。SIMPLE 上 ScaleBFM 的 D 是 1.856 cm；LML 上 SONIC 的 D 是 3.150 cm，比 ScaleBFM 难。5 cm/s 的标签还带着抖动。
- 学习曲线：D_S / D_S_f0.3 ≤ 0.95。
- xfer 的 SONIC·SIMPLE 记录：D_S 的误差 2.5–4.5 cm。LML 的 D_sonic 在这份记录上是它 val 的 1.785 倍，约 5.6 cm；D_S 低 20–55 %。

**真实局**
- A（H20b，CPU 网络）：45 s 内 18–40/120。主指标是 29/120，CPU 和 GPU 之间翻转约 20 局。
- F1-0（D_S，两下合起来）：
  - 预抓取每个任务 ≥ 80 %，进入 40–90 %；两下合起来每个任务 55–95 %；
  - BendPick 最低，因为它的进入段要下蹲；
  - 至少一个任务 ≥ 80 % 的把握约六成。
- F1(a)：每个任务 0.9–2.0 倍。任务标签是在下一下命令时取的，SONIC 的手那时还在动，比 val（停稳后的标签）难。
- 闸门：GATE PASS 的把握约 45 %。
- R_D（如果跑）：
  - 四个任务合计，45 s 内 R_D − A 在 −6 到 +10 局，符号检验不显著；
  - 过闸门的任务上，进入段的手掌偏差降 10–40 %。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| 采集有片退出码非 0，或 check 没有 "CHECK PASSED" 和 "SIMPLE CHECK PASSED" | 停下查清，不训练 |
| 单手可用段合计 < 15 000，或摔倒加 lost > 40 % 的重置 | 停下查清，不训练。数据量比预测少几倍，或 lost 是开发时的两倍以上，说明停稳规则或原地采集在 SIMPLE 上不成立，报协调方 |
| 训练、留出评估失败 | 停下查清，不跑任务局 |
| 阶段 A 有作业退出码非 0；AL 和 A 不逐局相同；plan_same 有假；在线、离线之差 ≥ 1e-4 m | 停下查清，不算闸门 |
| GATE FAIL（四个任务都不过） | 不跑 R_D。结论写成"SIMPLE 原地数据训出的 D_S 在任务局上仍不可信"，写清是 F1-0 还是 F1(a) 不过，坏在预抓取还是进入段，碰桌子的伸手占多少。报协调方 |
| GATE PASS（≥ 1 个任务过） | 当天跑 R_D，四个任务 120 局，同机按局配对，每个任务注明闸门结果 |
| 主比较或 P1 成立（R_D 多），而且带来提高的任务闸门和 F1 都过 | SIMPLE 上 SONIC 的一次性修正提高了成功率，写明是哪个任务带来的 |
| 主比较或 P1 成立，但带来提高的任务闸门或 F1 不过 | 照报，不归功于自身模型 |
| 都不成立 | 照实报各任务的差和区间，不合并成"赢" |
| 某任务 R_D − A ≤ −3 局 | 修正在这里有害，照报，并写明这个任务的闸门和 F1 |
| 过闸门的任务上手掌偏差没降 | 反解没起作用，先查 D_S 对修正命令的预测误差有没有变大 |
| **手**：过闸门的任务上，AL 的模型 / 直接 ≤ 0.5 | SONIC 的手偏大部分能从下命令时的状态预测，自身模型能修 |
| **手**：过闸门的任务上，AL 的模型 / 直接 ≥ 0.8 | 手偏在下命令时预测不了，修命令没用；手的问题在手晃、历史或接口 |

## 运行计划

1. 冻结（`ssd_freeze.sh`），核对副本的链接和真目录。
2. 开跑前 `nvidia-smi`。正式采集：`ssd_collect.sh formal 54 36`，90 个进程，nice 10。登记到灵动岛 `~/.claude/longjobs/simplesonicdata.json`，跑起来以后把局数和预计时间报协调方。
3. 同时起 `ssd_chain.sh <卡>,<卡>`：它等采集的 `logs/finished`，然后训练、阶段 A、一致性、闸门；GATE PASS 就接着阶段 R、F1、评估；GATE FAIL 就停。
4. 训练用两张卡，每张先要有 ≥ 8 GB 空闲。任务局用 8 张卡，每卡最多 5 个进程，每个进程不到 1 GB。

## 登记前核对

都在 H20b 上、`code_dev` 副本里跑，记录在 `checks/`、`dev/`、`dev2/`。

1. **默认关闭时逐位不变**（`ssd_collect.sh idcheck`，`checks/idcheck/same.txt`）：改过的 sc_collect 不给停稳开关，在 H20b 接入核对的 asrun_test 那组重置上跑 ScaleBFM（d3 的 1、5、9、13、30 号，d3b 的 1、5、12、29、57 号，第 1 片 / 共 4 片），和接入核对用 simpledata 冻结代码跑出的 `sc/asrun_test_h20b` 比：d3 4 个文件 144 个数组、5 条重置日志，d3b 6 个文件 231 个数组、5 条重置日志，全部 SAME。
2. **开发的 20 + 20 次重置**（`dev/`，开发编号，2 + 2 个进程，12:55–13:01Z）：

   | | data3 式 | data3b 式 |
   |---|---|---|
   | 单手段 / 可用（没摔、停稳） | 27 / 22 | 46 / 42 |
   | 每次重置的可用单手段 | 1.10 | 2.10 |
   | 可用段里只在 5 cm/s 下停稳的 | 95 % | 98 % |
   | 结束时手速中位数 | 4.46 cm/s | 4.15 cm/s |
   | 抖动 `jit_pp_cm` / `jit_rms_cm` 中位数 | 0.65 / 0.24 cm | 0.67 / 0.23 cm |
   | 摔倒 / lost 的重置 | 0 / 4 | 0 / 3 |
   | 每次重置用时，中位数 | 22.6 s | 32.8 s |

   - 下一下伸手因为等不到停稳被跳过的：data3 式 8.1 %（双手 22 %）。
   - 直接执行的偏差（13 cm 点）：原地中位数 13.1 cm。
   - 每个进程约 1.0 核（4 个进程 4.0 核；40 个进程 41.5 核）、约 2.85 GB 内存。
   - merge 和 check 都过：CHECK PASSED，SIMPLE CHECK PASSED（第 (2) 项按"代码"里说的改法；改之前这一项在 data 和 data_both 上各不过一次，最大差 0.52 mm 和 0.28 mm）。
   - data.py 自己的 settled 标记（2 cm/s）在我们的停稳段上只有 4.5 % 为真。它只用来训练"停没停稳"的头，不筛位置的训练数据（`models.head_loss`）。
3. **再采 200 + 200 次重置**（`dev2/`，开发编号，40 个进程，13:03–13:12Z，给后面的管线冒烟用；`ssd_datacheck.py` 原文）：

   ```
   data3: 200 resets, 280 one-hand segments, usable 215 (1.07 a reset); settled only under the relaxed rule 99.1 %; jit_pp median 0.69 cm, rms 0.23 cm; resets fell 2, lost 17 (9.5 %); wall 30.5 s / reset median, 31.2 mean
   data3b: 200 resets, 478 one-hand segments, usable 401 (2.00 a reset); settled only under the relaxed rule 99.5 %; jit_pp median 0.66 cm, rms 0.23 cm; resets fell 0, lost 24 (12.0 %); wall 41.5 s / reset median, 42.6 mean
   usable one-hand segments 616; DATA STOP (< 15 000 usable, or fell + lost > 40 %)
   ```

   - 最后一行的 DATA STOP 是开发数据量小的缘故，说明停止判据会动手。merge 和 check 都过（CHECK PASSED，SIMPLE CHECK PASSED）。
   - 40 个进程时 40.0 核、114 GB 内存（`checks/cpu.log`）。
4. **训练管线冒烟**（`SSD_TEST=1`，dev2 的数据，2000 步，卡 6、7，`checks/train_dev/`）：mid（MID CHECKS PASS）、mid 对照（FDVEL 关时逐字节相同，开时只换 96 列，MID FDVEL CHECK PASS）、划分检查（SPLIT CHECK PASS）、两个训练、评估 1、评估 2 都跑通。`ssd_datacheck.py` 判 DATA STOP，但测试模式下不停，这是设计好的。数据太少，误差（val 6.8 cm）不说明什么。
5. **任务局冒烟**（`ssd_run.sh` 的冒烟模式，`checks/smoke/`；测试局 TabletopGrasp 第 0 局和 BendPick 第 0 局，A、AL、R_D 各一局，模型是第 4 条那个 2000 步的模型）。只看管线，不读成败和偏差：
   - 6 个作业退出码都是 0；
   - A = AL：2 / 2 局去掉 `clean()` 去掉的字段后相同；
   - plan_same：AL 4 / 4，R_D 4 / 4；在线、离线之差最大 4.4e-6 m；
   - 那一眼和主指标 SONIC A 栏：2 / 2 相同；
   - R_D 跑得起来，IK 误差 ≤ 1.5 cm，没有退回；思考时间 3.7–15.8 s。修正量 72–96 cm，是因为冒烟模型几乎没训，不说明什么；
   - scene 写在副本自己的 `outputs/simple_eval_20261112/scene/` 下，共享树里没有比冒烟开始时间新的文件。
6. **闸门脚本**（`checks/fakegate/`）：把冒烟的 A 行冒充四个任务的行，`ssd_gate.sh` 跑通，F1-0、F1(a)、F1(b) 和判定都打印出来，判 GATE FAIL（冒烟模型的 F1(a) 是 12–14 倍）。
7. **HEFT 那一格**：simpleheft 的闸门 13:14Z 判 GATE PASS（XMovePick、XMoveBendPick），不需要在 H20b 上采 HEFT 的数据。给它留的 50 个进程用不上，本条仍然只开 90 个进程。

## 预测修订

## 结果（贴原始输出，不转述）

## 意外

## 结论与下一步

## 独立复核
