# selfmodel_nowalk_20261126：自身模型的数据要不要"先走、停下、再伸手"，只采原地伸手够不够

- **日期**：2026-11-26。在任何正式训练之前写好，单独提交。冒烟和划分检查在提交之前做完，结果写在"开跑前的准备"里；正式训练在提交之后。
- **用户的决定**（11-26，协调人转达，最高优先级）：做一个数据消融，回答一个问题——采自身模型的数据时，"先走一段、停下、等稳、再伸手"的重置是不是必须的，还是只采原地伸手就够。如果只采原地伸手就够，以后采数据会快很多：SIMPLE 里原地重置中位 7.5 s，先走后伸 12.0 s（simpledata 修订 1 的冒烟）。
- **机器**：新的 H20 机器（`ssh -p 27840 root@14.103.60.188`，4 张 H20，每张 96 GB）。
  - 训练只用 0、1 号卡。按用户的规定，每张卡总显存不超过 80 GB，所以每张卡最多同时跑我们的 2 个训练；每次开训练前看 nvidia-smi，要求已用显存加 12 GB 不超过 80 GB，不满足就等。2、3 号卡不碰。
  - tmux `keepalive` 覆盖 4 张卡，不停它、不动它。
  - 评估只用 CPU，nice 10，每个进程 4 个线程，同时最多 8 个进程。
  - 启动方式和 4090b 一样：`/usr/bin/python3.10 -S …/py4090b.py`，加 `LOCOMANI_WORKSPACE`、`SONIC_POLICY` 和 simpledata 的 `asrun/repo_b`。在 H20 上验证过能用 CUDA（torch 2.6.0+cu124，H20 是 sm_90）。
- **代码**：`selfwm`。新目录 `scripts/experiments/selfmodel_nowalk_20261126/`：
  - `nw_split.py`：从 D 的训练数据里做出各组的训练集，做所有划分检查和训练前的输入检查；
  - `nw_train.py`：包一层 train3b 冻结的 `t3b_train.py`，只在 `data.load` 之后改两处（见"做法"），其余一行不改；
  - `nw_eval.py`：几组模型在同一批段上打分，配对比较；
  - `nw_range.py`：包一层 simpledata 冻结的 `sd_range.py`（评估 4），把训练集换成 N 的；
  - `nw_run.sh`：正式的训练和评估流程。

  共用的代码全部取 simpledata 冻结的 `asrun/repo_b`：`data.py`（174071cb，FDVEL 版）、`train.py`（8acdaea1）、`models.py`（0ac27f6b），train3b 冻结的 `t3b_train.py`（c7ecf581），simpledata 的 `sd_range.py`（1276163a）。正式运行前把本实验的脚本拷进 `asrun/code/`，md5 记在 `asrun/MD5.check`。
- **输出**：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/`，下面简称"本目录"。simpledata 的输出和共用的代码树都只读。
- **关联**：
  - [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md)：SIMPLE 里的 ScaleBFM 数据、模型 D、120 局真实局；复核指出的 crc32 奇偶问题和按任务的读法；
  - [selfmodel_train3_20261125](selfmodel_train3_20261125.md) 更正 3：LML 上只比第一下伸手，先走后伸的误差是原地的 1.2–1.3 倍；
  - [selfmodel_train3b_20261125](selfmodel_train3b_20261125.md)：配方 D；
  - [selfmodel_lmlreal_20261125](selfmodel_lmlreal_20261125.md)：LML 的真实任务局（第二阶段用）。

## 背景

自身模型预测：冻结的控制器收到一条伸手命令后，手最后停在哪。任务局里每一下伸手都发生在机器人走到桌前、停下之后。这时关节还在微动，髋有扭转，站姿也和默认站姿不同。所以现在采数据时混了两种重置：
- **原地伸手**：出生后直接伸手，起点只加了 0.007 rad 的关节扰动；
- **先走后伸**：约 1/3 的重置先走 0.3–1.5 m，停下、等稳，再伸手。在 SIMPLE 里，这些重置一半用"任务式"走法（不转身，位移落在 APPROACH_BOX 里），一半照 d3 的规则（转、走、转，走完髋扭 30–70°）。

已有的证据：
- LML 上第一下伸手，先走后伸的误差是原地的 1.2–1.3 倍（train3 更正 3）。这是两种起点都训过的模型，说的是"难"，不是"没见过"。
- SIMPLE 的 D：原地第一下 0.80 cm；任务式走法后的第一下 1.03 cm（test_iid）；d3 走法后的第一下 1.95 cm（val）。
- **还没有人训过不含先走后伸数据的模型，再拿真实任务局去测。** 这就是本实验要补的。

## 假设

**只用原地伸手的数据训出来的模型（N），在真实 SIMPLE 任务局上的预测误差，和用同样多、但含先走后伸数据训出来的模型（C）差不多。** 理由：
1. 真实局的第一下，都在任务式走法之后。任务式走法不转身，髋扭不超过 15°（simpledata check 第 8 项，100 %）。训练前的输入检查（见"开跑前的准备"第 3 条）显示，真实局第一下的起点，有 84.5 % 的段 29 个关节都落在 N 的训练集 p0.5–p99.5 之内（C 是 100 %）；按 N 的标准化，真实段 x0 的最大 |z| 是 21.2，按 C 是 18.6，差得不多。也就是说，任务式走法之后的起点离原地伸手的分布不远。
2. 真实局的第二下（进入）从预抓取的姿态出发，是一段连续伸手。原地重置里的连续伸手覆盖这种起点。

反方向的理由：
- 两种起点都训过的模型里，先走后伸的第一下也比原地难 1.2–1.3 倍；没见过这种起点，误差可能更大。
- SIMPLE 的 ScaleBFM 走完以后第一条命令时，关节还在动（0.1 s 差分速度 0.28 rad/s，LML 是 0.02），原地重置里没有这种状态。
- 真实局第一下有 15.5 % 的段至少一个关节落在 N 的训练范围外，最多的是 S 的第 1 个关节（7.7 %）。

## 做法

### 第一阶段：SIMPLE，ScaleBFM（先做）

**各组**。都是配方 D：data3 + data3b + 两个中间点文件，`REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1`，不开 X0_REF；E3_ensemble，frac 1.0，30 000 步，迭代数、batch、学习率、评估网格都用 train.py 的缺省值，和 simpledata 的 `sd_run.sh` 一样。

| 组 | 训练数据 | 走路标志 | 种子 |
|---|---|---|---|
| **N**（不走） | 去掉所有先走后伸重置：它们在 data3、data3b 里的全部段（第一下和连续伸手），以及源段属于它们的中间点样本 | 训练和评估都置 0 | 0；1（重复） |
| **C**（数据量对照） | 按 md5 随机整次去掉重置，不分种类，直到训练段数和 N 相同（±1 %） | 保留 | 0；1（重复） |
| **C2** | 和 C 同一批重置 | 训练和评估都置 0，和 N 一样 | 0 |
| **D**（参照） | 全部数据，simpledata 已训好的模型，不重训 | 保留 | 0（在 4090b 上训的） |

- **主比较是 N 对 C**：两者训练段数相同，只差"有没有先走后伸的数据"和走路标志。学习曲线还在降（frac 1.0 对 0.3 是 0.734），所以不能拿 N 直接和 D 比。
- **C2** 把走路标志的作用单独拿出来：N 对 C2 只差数据，C2 对 C 只差标志。一个训练在 H20 上只要十来分钟，所以加上。
- **种子 1 的 N 和 C** 是我加的重复，看训练本身的波动有多大。用户的设计只要求种子 0；判据以种子 0 为准，种子 1 只用来判断结论稳不稳（见判据）。两张卡、每张两个训练，加这两组不多花墙钟。

**训练集怎么做**（`nw_split.py`，正式数据上已跑过，检查全部通过，见"开跑前的准备"）
- **哪些是先走后伸的重置**：读采集日志（`data3/data`、`data3b/data` 的 `w*_log.json`，每次重置一条）的 `walked`。data3 有 9927 次，data3b 有 4960 次，共 14 887 次。再和每一段的 `walked` 标志对一遍：data3、data3b、两个中间点文件的全部 851 294 段都一致；中间点样本的重置编号和走路标志，经 `mid_*_src.npz` 查到的源段也都一致。
- **怎么去掉**：被去掉的重置，它们所有段的桶号改成 10。这样 `data.splits` 不会把它们放进训练、验证、测试的任何一份。band 和 edge 本来就不参与训练。
- **N**：去掉全部 14 887 次先走后伸的重置。
- **C**：
  1. 先去掉验证桶（桶 8）里先走后伸的重置，让 C 的验证集和 N 的一样；
  2. 训练重置（桶 0–7）按 `md5("nowalk_c:<重置编号>")` 从小到大排，按这个顺序保留，直到保留的训练段数第一次不少于 N 的训练段数，其余的去掉。
  
  不用 crc32：simpledata 里 crc32 取余偶数和划分的桶号相关，md5 和它们无关。
- **C2**：去掉的重置和 C 完全相同。
- **挑 checkpoint 用的验证集**：N、C、C2 都只用桶 8 里的原地重置（4260 段，2534 次重置）。理由：
  - N 要是一个只采原地数据就能部署的模型，它选 checkpoint 时就不能用先走后伸的数据；
  - C 和 C2 用同一份验证集，这样三组的差别只来自训练数据。
  
  D 当时用的是完整的验证集（6266 段），这一点只影响和 D 的比较。D 的 val 曲线在 8k 步以后很平（1.856–1.877 cm），挑 checkpoint 带来的差别应该很小。

**走路相关的输入怎么处理**
- x0 有 278 列。和走路有关的是两处：
  - 第 204 列 `walked`（data.py 里 x0 的 `D["walked"]`，已核对逐段等于走路标志）；
  - 最后 4 列 274–277，是 `REACHWM_APPROACH` 的走路特征。本实验所有数据文件里都没有 `walk` 这个数组，`data.load` 填的是 0，所以这 4 列在训练和真实局里处处为 0（已核对）。D 也一样，这 4 列对 D 本来就是常数。
- 所以实际会变的只有第 204 列。N 的训练数据里它恒为 0，标准化后均值 0、标准差取下限 1e-3。真实局全部是先走后伸，`walked = 1`，如果不处理，标准化后会变成 1000。
- **登记的做法**：N 和 C2 在训练和所有评估里都把第 204 列置 0。对 N 来说训练时是空操作，评估时等于"部署的人不知道、也不告诉模型刚才走没走过"。这正是只采原地数据以后能部署的模型。`nw_eval.py` 加载 N 和 C2 时会核对：每个成员第 204 列的标准化均值是 0、标准差是 1e-3。
- **别的零方差列**：按 N 的训练集算，标准差低于 1e-3 的列只有第 204 列和 274–277 列，没有别的；按 C、D 算只有 274–277 列。

**怎么训**（`nw_train.py`）
- 包一层 train3b 冻结的 `t3b_train.py`：把 `data.load` 换成"先照原样读，再把这一组去掉的重置改成桶 10；N、C2 再把第 204 列置 0"。
- 读完以后核对：训练段数、验证段数等于 `splits.npz` 里记的；N 的训练段里没有一段 `walked = 1`。`t3b_train` 看到的是同一个 data 模块（有断言）。
- 其余全照 `t3b_train` 和 `train.train`：划分、种子、bootstrap、学习率表、按 val 挑 checkpoint、存模型。

### 评估

所有评估都在 CPU 上跑（`nw_eval.py`）。误差的定义和 sd_eval、sx_eval 一样：预测的最终手点到真实手点的距离（cm，起点朝向系），另报姿态误差（度）。保留的段也和 simpledata 一样：没摔、末端停稳（settled）；"ended"指只要没摔。

**主评估：120 局真实 SIMPLE 局，主标签，settled，235 段**
- 就是 simpledata 评估 3 的那批段，四个抓取任务各 30 局，全部先走后伸（任务式走法）。
- 每组报误差中位数，总体、按任务各报一次，附 95 % 区间。
- **区间按布局重抽**：布局是（任务，局号 % 10），局号取自重置编号（90 000 000 + 1000 × 任务 + 100 × 难度 + 局号），共 40 个布局，每个布局 3 局（三个难度）、6 段。重抽 4000 次。理由是 simpledata 的复核发现，同一个局号的三个难度抓取点和走位几乎相同，不是独立样本。
- **配对比值**：N/C 是登记的主指标，同时报 N/C2、C2/C、N/D、C/D、C2/D。每次重抽都在同一批段上算两组的中位数再相除。
- 还要分开报：第一下（预抓取，chain 0，116 段）、第二下（进入，chain 1，119 段）、每个任务、去掉 TabletopGraspMP 的三个任务（180 段，30 个布局）。
- **TabletopGraspMP 单独看**：它被桌沿挡住的那批段是接触造成的失败，不是起点的问题。每组模型在这些段上都会有 28–40 cm 的误差。
- 也报 p90。

**次要评估**
1. **留出集**：用 data.py 自己的桶（crc32），按重置重抽 2000 次，各组配对：
   - data3：原地第一下、原地连续伸手（val + test_iid）；任务式走法后的第一下和连续伸手（只在 test_iid 里有）；d3 走法后的第一下和连续伸手（只在 val 里有）；data3 val 全部（simpledata 的分母）；test_iid 全部；
   - data3b：预抓取第一下加进入第二下，分任务式走法（test_iid）、原地（val + test_iid）、d3 走法（val）。
2. **补充标签**（任务原本的节奏）：ended（simpledata 补充标签的主口径）和 settled；主标签的 ended 也报。
3. **起点在不在 N 的训练范围里**（`nw_range.py`）：simpledata 的 `sd_range.py` 原样跑，只把训练集换成 N 的（同样的去掉重置，标志置 0），模型换成 N。报真实局的段落在 N 训练范围里的比例，以及 N 在范围内外的误差。D 的同一项在 simpledata 的 `train/eval4.txt` 里。
4. **核对**：`nw_eval.py` 用 D 重算 simpledata 记过的四个中位数（真实局主标签 settled 0.7053、补充标签 ended 0.8208、data3 val 1.8860、data3 test_iid 1.5756，都是 CPU 上的数），差超过 2e-4 cm 就算失败、停下。

**和 D 比的时候要记住**：D 是在 4090b 上训的，N、C、C2 在 H20 上训。同一配方、同一冒烟数据、300 步，两种卡的 best val 不一样（见"开跑前的准备"第 1 条）。所以 N/D、C/D 里混着换卡的差别；N/C、N/C2、C2/C 都在 H20 上，没有这个问题。

### 第二阶段：LocoManiLab，三个控制器（第一阶段之后，GPU 有空就做）

ScaleBFM、SONIC、HEFT 各做同样的 N/C 消融：
- 训练数据是 train3b 的 D（`selfmodel_data3_20261125`、`selfmodel_data3b_20261125` 加中间点，见 `selfmodel_train3b_20261125`）；
- 评估用 `selfmodel_lmlreal_20261125` 的真实 LML 任务局；
- 对 SONIC 最重要：它的先走后伸采得最慢、最不稳。

数据文件、局的布局怎么定义、按什么重抽，在开跑前写成一个修订，单独提交。这里先把方向性的预测和判据写下来，修订里只补实现细节，不改数字。

## 开跑前的准备（2026-11-26，正式训练之前）

1. **H20 和 4090b 的对照**（协调人要求）：用 simpledata 的 D 配方、它的冒烟数据（smoke/d3、smoke/d3b 加 dev_b/runtest 的两个中间点文件）、seed 0、300 步，在 H20 的 0 号卡上跑。
   - **不一致**，原因是显卡型号不同。原文：

```
H20   (smoke/h20_equiv.log):   E3_ensemble 1.0 {"it": 300, "train": 0.07785685360431671, "val_median_cm": 1.878440022468567, "s": 16.7}
4090b (simpledata dev_b/runtest/logs/train_D_s0_f1.log):
                               E3_ensemble 1.0 {"it": 300, "train": 0.07601235061883926, "val_median_cm": 1.8745638132095337, "s": 10.2}
model.pt md5: H20 f24a7430 (h20_equiv, h20_equiv2, h20_conc_a, h20_conc_b), 4090b 35c9d5df
```

   - H20 自己是确定的：同样的命令再跑一次，model.pt 逐字节相同（md5 f24a7430）；同一张卡上同时跑两个，也都是 f24a7430。
   - 所以照协调人的安排：N、C、C2 都在 H20 上训，彼此配对比较；D 用现有的模型，标明是在 4090b 上训的。
2. **划分检查**（`splits/split.log`，原文）：

```
  [ok] every segment's reset is in the collection logs
  [ok] segment walked == the log's walked, all 851294 segments (data3, data3b, mid)
  [ok] mid_data3_src rows 480254 == mid_data3 segments 480254
  [ok] mid_data3: reset and walked equal to the source segment's (mid_*_src.npz)
  [ok] mid_data3b_src rows 280150 == mid_data3b segments 280150
  [ok] mid_data3b: reset and walked equal to the source segment's (mid_*_src.npz)
  [ok] x0 column 204 == walked
  [ok] x0 column 205 == chain
  [ok] x0 columns 274-277 (REACHWM_APPROACH) zero everywhere
  [ok] D's training split 809804 == 809804 (simpledata run.log)

arms (training split after the arm's drop; val = data.splits' val, fell excluded as train.train does):
  D : train 809804 segments (29260 resets; first 306654, chained 503150), walk-first 267345 segments / 9737 resets (task 134411, d3 132934); per file {'data3': 25691, 'data3b': 23709, 'mid_data3': 480254, 'mid_data3b': 280150}; val 6266 (3727 resets, walk-first 2006); test_iid 6130; drop 0 resets; zero walk flag False
  N : train 542459 segments (19523 resets; first 204207, chained 338252), walk-first 0 segments / 0 resets (task 0, d3 0); per file {'data3': 17218, 'data3b': 15672, 'mid_data3': 323457, 'mid_data3b': 186112}; val 4260 (2534 resets, walk-first 0); test_iid 4130; drop 14887 resets; zero walk flag True
  C : train 542494 segments (19673 resets; first 206561, chained 335933), walk-first 177597 segments / 6521 resets (task 88803, d3 88794); per file {'data3': 17332, 'data3b': 15759, 'mid_data3': 323308, 'mid_data3b': 186095}; val 4260 (2534 resets, walk-first 0); test_iid 6130; drop 10979 resets; zero walk flag False
  C2: train 542494 segments (19673 resets; first 206561, chained 335933), walk-first 177597 segments / 6521 resets (task 88803, d3 88794); per file {'data3': 17332, 'data3b': 15759, 'mid_data3': 323308, 'mid_data3b': 186095}; val 4260 (2534 resets, walk-first 0); test_iid 6130; drop 10979 resets; zero walk flag True
  [ok] N's training split 542459 == D's training segments of in-place resets 542459
  [ok] N: 0 walk-first segments / resets in training
  [ok] N: walked flag 0 on every training segment
  [ok] N, C: val (checkpoint pick) in place only
  [ok] N and C pick on the same val segments
  [ok] C's training split 542494 vs N's 542459: +0.006 % (tolerance 1 %)
  [ok] real main: x0 columns 274-277 zero
  [ok] real main settled: walked flag 1 on every segment
SPLIT CHECK PASS
```

   - C 里先走后伸的段占 177 597 / 542 494 = 32.7 %，D 是 267 345 / 809 804 = 33.0 %；C 里任务式和 d3 式各一半（88 803 对 88 794）。md5 的抽法没有偏向哪种重置。
   - `splits.npz` 的 md5 是 4a47e698。
3. **训练前的输入检查**（不用模型，只看起点；同一份 `split.log`）：真实局主标签 settled 的 235 段，起点（S[k0]）的 39 个量落在各组训练集 p0.5–p99.5 之内的比例，以及按各组的标准化，真实段 x0 的 |z|：

```
input check, real main settled (235 segments; first 116, second 119): share inside the arm's training p0.5 / p99.5 (sd_range's 39 quantities), and x0 columns under Norm's floor
  D : joints 86.8 % (first 100.0, second 73.9); pelvis 98.7 % (first 100.0, second 97.5); hands 100.0 % (first 100.0, second 100.0); all 39 86.8 % (first 100.0, second 73.9)
      joints most often outside (S index, share): [(24, 0.10212765957446808), (26, 0.02127659574468085), (10, 0.01276595744680851), (9, 0.01276595744680851), (4, 0.0), (5, 0.0)]
      x0 columns with training sd < 1e-3 (col, sd, max |z| on real): [(274, 0.0, 0.0), (275, 0.0, 0.0), (276, 0.0, 0.0), (277, 0.0, 0.0)]
      |z| on real under this arm's normalisation: max 18.5, p99 of row max 6.3, median row max 1.9; top columns [(33, 18.463206301912887), (61, 13.318411546538945), (98, 11.867081785074493), (32, 11.296596975778511), (108, 10.523581116587037), (101, 9.821011898174845)]
  N : joints 78.7 % (first 84.5, second 73.1); pelvis 98.7 % (first 100.0, second 97.5); hands 100.0 % (first 100.0, second 100.0); all 39 78.7 % (first 84.5, second 73.1)
      joints most often outside (S index, share): [(24, 0.10212765957446808), (1, 0.07659574468085106), (26, 0.02127659574468085), (9, 0.01276595744680851), (10, 0.01276595744680851), (3, 0.00851063829787234)]
      x0 columns with training sd < 1e-3 (col, sd, max |z| on real): [(204, 0.0, 0.0), (274, 0.0, 0.0), (275, 0.0, 0.0), (276, 0.0, 0.0), (277, 0.0, 0.0)]
      |z| on real under this arm's normalisation: max 21.2, p99 of row max 6.3, median row max 2.3; top columns [(33, 21.19269788680051), (98, 14.427299971819712), (61, 13.637086470156119), (32, 13.073588954904237), (101, 10.568726779943292), (63, 10.449772051466056)]
  C : joints 87.2 % (first 100.0, second 74.8); pelvis 98.7 % (first 100.0, second 97.5); hands 100.0 % (first 100.0, second 100.0); all 39 87.2 % (first 100.0, second 74.8)
      joints most often outside (S index, share): [(24, 0.10212765957446808), (26, 0.01702127659574468), (10, 0.01276595744680851), (9, 0.01276595744680851), (4, 0.0), (5, 0.0)]
      x0 columns with training sd < 1e-3 (col, sd, max |z| on real): [(274, 0.0, 0.0), (275, 0.0, 0.0), (276, 0.0, 0.0), (277, 0.0, 0.0)]
      |z| on real under this arm's normalisation: max 18.6, p99 of row max 6.3, median row max 1.9; top columns [(33, 18.581140523878407), (61, 13.46825480056895), (98, 11.991599749039873), (32, 11.428561059289173), (108, 10.552259699889099), (101, 9.842305189679708)]
```

   - 第 204 列在 N 下的 max |z| 是 0，因为真实段的这一列也已置 0；没有置 0 的话它是 1000。
   - 输入文件的 md5（`split.log` 第 2 行）：data3 b92cd560、data3b ed71649d、mid_data3 7a7c947f、mid_data3b 4b09df50。

   这一条是在写预测之前看的，只看了输入，没有任何模型的输出。

4. **整条流程的冒烟**（`# notebook: skip smoke`）：正式数据、300 步，N 在 0 号卡、C 在 1 号卡、C2 在 CPU 上，接着跑 nw_eval（含 D 的核对）和 nw_range。输出在 `smoke/run/`，数字没有意义，不记。结果：

   - **训练**：三组都跑完 300 步。`nw_train` 的核对都通过：N 训练 542 459 段、验证 4260 段、训练段里 `walked = 1` 的 0 段、第 204 列均值 0；C、C2 训练 542 494 段。H20 上 300 步约 6 s（N 6.0 s、C 5.9 s），CPU 上 29.3 s。
   - **评估**：`nw_eval` 输出 `REPRODUCTION PASS`，D 的四个中位数和 simpledata 记的到小数点后 5 位都相同：

```
  reproduction: D median 0.70526 vs simpledata 0.70526 (+0.00000)
  reproduction: D median 0.82076 vs simpledata 0.82076 (+0.00000)
  reproduction: data3 val all D median 1.88601 vs simpledata 1.88601 (+0.00000)
  reproduction: data3 test_iid all D median 1.57564 vs simpledata 1.57564 (+0.00000)
```

   - **各格的段数**和 simpledata 一致：主标签 settled 235 段、40 个布局（第一下 116 段，39 个布局；第二下 119 段）；去掉 TabletopGraspMP 180 段、30 个布局；data3 任务式走法后的第一下 517 段、d3 走法后的第一下 543 段、原地第一下 2205 段（1125 + 1080）；data3b 任务式走法的预抓取 + 进入 140 段、86 次重置（复核里"最可比"的那一层）。
   - **nw_range** 跑通。它报的"落在 N 训练范围里的比例"只取决于训练集，不取决于模型，所以这个数在冒烟里已经是定数：主标签 settled 的 "all three" 是 78.7 % [73.8, 83.3]（185 / 50 out），和第 3 条的 39 个量的结果相同，最近邻和伸手表两项没有多排除任何段。范围内外的误差要等正式模型。
   - 冒烟模型的误差和比值我看到了，但那是 300 步的模型（val 4.3 cm，正式的 D 是 1.9 cm），不拿来写预测。
   - 冒烟的 tmux 会话 `nowalk_smoke` 跑完以后自己结束了。

5. **冻结**：本实验的脚本拷进 `asrun/code/`，md5 记在 `asrun/MD5.check`（连同 `splits.npz`），冻结时刻 `asrun/FROZEN_UTC` = 2026-11-26T09:05:49Z。`nw_run.sh` 开跑前核对它，以及 simpledata 的 `asrun/MD5.check`、`MD5_r2.check` 和 `t3b_train.py` 的 c7ecf581。

| 文件 | md5 |
|---|---|
| nw_split.py / nw_train.py / nw_eval.py | 474a4b0b / f000197d / acd4c564 |
| nw_range.py / nw_run.sh | bdc2985a / 47cdc03c |
| splits/splits.npz | 4a47e698 |

6. **正式运行的跑法**（`asrun/code/nw_run.sh`，H20 上 tmux `nowalk`，输出 `run/`）：
   - 训练按顺序开：N s0（0 号卡）、C s0（1 号卡）、C2 s0（0 号卡）、N s1（1 号卡）、C s1（等有空位的那张卡）。每张卡最多同时 2 个，开之前查 nvidia-smi（已用 + 12 GB ≤ 80 GB），每次的 nvidia-smi 记进 `run/run.log`；
   - 全部训完以后在 CPU 上跑：`eval_s0`（D、N、C、C2）、`eval_s1`（D、N s1、C s1）、`range_N`；
   - 进度登记在 `~/.claude/longjobs/nowalk.json`。

## 预测（启动前写，首次启动后锁定）

**训练**
- 每个训练在 H20 上 10–20 分钟（300 步约 6 s，按比例 30 000 步约 10 分钟，加上验证；一张卡两个同时跑会慢一些）。显存每个 5–9 GB。
- 挑 checkpoint 用的原地 val 上，N 不比 C 差：N/C 在 0.88–1.02 之间。N 的原地数据是 C 的约 1.5 倍（C 只有 2/3 是原地重置）。

**主评估：真实局主标签 settled，235 段，种子 0**

| 比值 | 预测（点估计） | 理由 |
|---|---|---|
| **N/C，全部** | **1.00–1.20**，最可能 1.05–1.10 | 任务式走法后的起点大多在原地数据的范围里（第一下 84.5 %），离群程度和 C 差不多；但没见过"刚走完、关节还在动"这种起点，第一下会差一点 |
| N/C，第一下 | 1.00–1.30 | 走路的影响集中在第一下 |
| N/C，第二下 | 0.95–1.15 | 第二下从预抓取出发，原地数据里的连续伸手覆盖这种起点 |
| N/C，两个 XMove 任务 | 各 0.95–1.35 | 误差本来只有 0.36–0.40 cm，绝对值上的一点偏差在比值上会被放大 |
| N/C，BendPickMP | 0.95–1.20 | |
| N/C，TabletopGraspMP | 0.90–1.15 | 被挡的段每组都错 28–40 cm，比值主要看没被挡的段 |
| C2/C | 0.95–1.08 | 走路标志和起点状态是重复的信息，模型从关节和速度就看得出走没走过 |
| C/D | 1.00–1.25 | C 只有 D 2/3 的数据。按 val 上的学习曲线（frac 0.3 对 1.0 是 0.734，指数约 0.26），2/3 的数据误差约高 11 %；再加上换卡的差别 |

- N/C 的 95 % 区间：宽度约 ±0.10–0.15。两组在同一批段上配对，误差高度相关。
- 落在判据各行的把握：第一行（≤ 1.10 且上限 ≤ 1.25）约 50 %，中间约 35 %，> 1.25 约 15 %。

**次要评估**

| 项 | N/C 预测 |
|---|---|
| data3 原地第一下（val + test_iid） | 0.88–1.02 |
| data3 原地连续伸手 | 0.90–1.02 |
| data3 任务式走法后的第一下（test_iid） | 1.05–1.40 |
| data3 任务式走法后的连续伸手（test_iid） | 1.00–1.25 |
| data3 d3 走法后的第一下（val） | 1.40–3.00：髋扭 30–70°，远在原地数据的范围外 |
| data3 d3 走法后的连续伸手（val） | 1.10–1.80 |
| data3b 预抓取 + 进入，任务式走法（test_iid） | 1.00–1.30 |
| 补充标签 ended | 和主标签的 N/C 相差不超过 0.05 |

- **起点范围**（nw_range）：落在 N 训练范围里的比例已经知道（78.7 %，见冒烟），不再预测。预测：N 在范围外的段误差比范围内高，第一下尤其如此。
- **种子 1**：N/C 和种子 0 相差不超过 0.08。

**第二阶段（LML，方向性的预测）**：真实 LML 任务局上，N/C 是 ScaleBFM 1.00–1.35，HEFT 1.00–1.40，SONIC 0.95–1.35。LML 的先走后伸只有 d3 式走法（转、走、转），起点和任务局的走到站位不一定像，所以 C 里的走路数据对真实局的帮助可能比 SIMPLE 里小，N/C 未必更大；但 LML 的模型误差大（val 3–4 cm），起点的影响也可能更大，所以区间放宽。

## 判据（什么结果会让我放弃或修改假设）

判据看主评估：真实局主标签 settled，全部 235 段，种子 0 的 N/C 和它按布局重抽的 95 % 区间。

| 结果 | 结论和下一步 |
|---|---|
| 核对失败（D 的四个中位数对不上），或 N 的训练集里出现先走后伸的段 | 停下查清，不看比较结果 |
| **N/C ≤ 1.10 且区间上限 ≤ 1.25** | 对 SIMPLE 里的 ScaleBFM，先走后伸的数据不需要。建议以后只采原地伸手，接着做第二阶段，看另外两个控制器和 LML 是否也成立 |
| **N/C > 1.25** | 先走后伸的数据要保留 |
| 其余（1.10 < N/C ≤ 1.25，或 N/C ≤ 1.10 但上限 > 1.25） | 试一个更省的中间方案，比如只采短距离的任务式走法，另行登记 |

附加规则（都在看结果之前定下）：
1. **种子 1 必须落在同一行**（按它的点估计 N/C 和区间判断）。如果两个种子落在不同的行，结论降为"中间那一行"：先补训种子 2、3，再决定。
2. **去掉 TabletopGraspMP 的三个任务**要单独按同样的阈值判一次。如果它和全部 235 段落在不同的行，结论也降为中间那一行。TabletopGraspMP 本身单独报，不单独判。
3. **单个任务**：如果三个没被挡的任务里有一个 N/C > 1.25 且区间下限 > 1.0，即使总体过线，"只采原地"的建议也要写明这类任务不适用。
4. **d3 走法的 val 不进判据**：任务里没有这种走法。它只说明 N 在远离训练范围的起点上会怎样。
5. **C2/C**：如果 C2/C > 1.10 且区间下限 > 1.0，说明走路标志本身有用，那么 N 的"标志置 0"是 N 的一个额外的不利条件；结论里要把"N 对 C2"一起写上，说明数据和标志各占多少。

第二阶段用同样的阈值，每个控制器单独判。

## 预测修订
<!-- 首次启动后如需修改预测，只能追加在这里，写明时间和原因。 -->

### 修订 1：第一阶段换到 4090b 重跑（2026-11-26 09:21Z；还没有任何正式结果）

预测和判据都不变，只有训练时间那一条按新机器改写（见最后）。

**H20 那次中断**
- 第一次正式运行 09:06:27Z 在 H20 上开始，输出在本目录的 `run/`。
- 约 09:12Z，H20 的容器重启了：ssh 先被拒、后超时，09:18Z 回来时 tmux 已经没有了，4 张卡都只用了 1 MiB，我们的进程一个都不在。
- 中断时 5 个训练开了 4 个，C s1 还在排队等卡。各自的日志停在：N s0 第 18000 步，C s0 第 16000 步，C2 s0 第 12000 步，N s1 第 12000 步。
- 4 个训练都还没写出 checkpoint，`E3_ensemble_f1/` 是空的。
- 这期间我只看过日志里的步数，没有看任何 val 数字。监视器只在出现 "best val" 时才会报，它一次也没报过。
- `run/` 原样保留，等 H20 稳定以后改名为 `run_aborted_0912`，不用于任何结果。

**换到 4090b**
- 这是协调人 09:19Z 的安排，理由有三：
  - 这个实验是最高优先级；
  - H20 刚崩过，回来以后稳不稳不知道；
  - simpledata 的 D 就是在 4090b 上训的。
- 用 4090b（`ssh -p 62882`）的 0 号卡，一次只跑一个训练。
- 按用户的规定，每张卡总共不超过 20 GB：开训练前要求已用显存加 9 GB 不超过 20 GB，不满足就等，等的时候每分钟记一次显存。
- 0 号卡上同时还有 HA 的一个采集进程 b0a，约 7.8 GB，GPU 利用率 77 %。
- 冻结的脚本、splits.npz 和任务顺序（N s0、C s0、C2 s0、N s1、C s1）都不变，评估也不变。
- 运行脚本是 `nw_run_b.sh`：从 `nw_run.sh` 复制，只改了几处：
  - 只用 0 号卡；
  - 同一时间只跑我们的 1 个训练；
  - 显存上限从 80 GB 改为 20 GB；
  - 输出目录改为 `run_b/`；
  - 多核对一个 `asrun/MD5_r1.check`。
- `nw_run_b.sh` 冻结在 `asrun/code/`，md5 41b69c51，冻结时间 2026-11-26T09:20:25Z（`asrun/FROZEN_R1_UTC`）。原来的 `MD5.check` 也照样核对。

**4090b 的对照**
- 做法：开跑前的准备第 1 条那个 300 步冒烟，用同样的命令在 4090b 的 0 号卡上再跑一次，输出在 `smoke/b_equiv`。
- 结果：和 simpledata 当初在 4090b 上的结果逐字节相同。原文：

```
4090b, this run (smoke/b_equiv.log): E3_ensemble 1.0 {"it": 300, "train": 0.07601235061883926, "val_median_cm": 1.8745638132095337, "s": 23.3}
simpledata dev_b/runtest:            E3_ensemble 1.0 {"it": 300, "train": 0.07601235061883926, "val_median_cm": 1.8745638132095337, "s": 10.2}
model.pt md5: smoke/b_equiv 35c9d5df, simpledata dev_b/runtest/models_D_s0 35c9d5df
```

- 这说明两点：
  - N、C、C2 现在和 D 在同一种卡上训。"评估"里"和 D 比的时候要记住"那一段说的换卡差别不再存在，N/D、C/D 只差训练数据。
  - 和 b0a 共用一张卡，不影响训练的数值。

**训练时间（改写登记里的那一条）**
- 登记时写的是"H20 上每个训练 10–20 分钟"，现在不适用了。
- 4090b 上和 b0a 共用一张卡，300 步用了 23.3 s；simpledata 单独用这张卡时是 10.2 s。
- 按这个比例，30 000 步加验证每个约 20–45 分钟，5 个依次跑完约 2–3.5 小时。
- 显存不再预测。H20 上每个训练实际只占约 2 GB：0 号卡上同时跑两个时，一共 5029 MiB。

### 修订 1b：4090b 上改成两个训练同时跑（2026-11-26 09:26Z；还没有任何正式结果）

- **起因**：09:24Z 协调人告诉我，HA 的 b0a 已经离开 0 号卡；卡上只剩 keepalive 的 808 MiB 和我们的 N s0，共 4233 MiB。所以同意同时跑两个训练，每次开训练前照样检查"已用 + 9 GB ≤ 20 GB"。
- **4090b 上的实际占用**：N s0 开跑以后，卡上已用从 9485 MiB 涨到 11067 MiB，每个训练约 1.6 GB。两个训练加 keepalive 约 4 GB，远在 20 GB 以内。
- **为什么不能直接用原来的流程**：冻结的 `nw_run_b.sh` 规定同一时间只跑我们的一个训练，它不能改。
- **为什么不重开**：重开要停掉 N s0。协调人说 N s0 不用停，所以改成接管：
  1. 把 `nw_run_b.sh`（这时它在等 C s0 的空位）用 SIGSTOP 停住。它的子进程 N s0 照常跑，`nw_run_b.sh` 本身不退出，所以 tmux 不会关，N s0 也不会收到 SIGHUP。
  2. 新脚本 `nw_run_b2.sh` 把 N s0 算作我们的一个训练，按原来的顺序开 C s0、C2 s0、N s1、C s1，同一时间最多两个。
  3. 新脚本等 N s0 结束，核对它的日志里有 best val、model.pt 已经写出。
  4. 然后用 SIGKILL 结束停住的 `nw_run_b.sh`。这时那个 tmux 里已经没有我们在跑的东西了。
  5. 评估照 `nw_run_b.sh` 原样跑，输出还是 `run_b/`。
- **结果不受影响的依据**：
  - 训练的数值和同卡上有没有别的进程无关：H20 上同一张卡同时跑两个，model.pt 逐字节相同；4090b 上和 b0a 共用一张卡，也和单独跑时逐字节相同（修订 1）。
  - 冻结的脚本和 splits.npz 都没改。
- `nw_run_b2.sh` 冻结在 `asrun/code/`，md5 30a53601，冻结时间 2026-11-26T09:25:38Z（`asrun/MD5_r2.check`、`asrun/FROZEN_R2_UTC`）。
- 接管时 N s0 已经跑到第 6000 步。这期间我只看过步数和每 2000 步的用时，没有看 val 数字。

### 修订 2：第二阶段的实现细节（2026-11-26 09:28Z；第二阶段还没有开始正式训练，第一阶段的结果也还没出来）

登记里第二阶段的预测和判据都不改，这里只补实现细节。登记时说过修订"只补实现细节，不改数字"。

**数据和配方**
- 每个控制器（ScaleBFM、SONIC、HEFT）分开做，训练数据就是 train3b 的 D：
  - `selfmodel_data3_20261125/<控制器>/data/all.npz`
  - `selfmodel_data3b_20261125/<控制器>/data/all.npz`
  - `selfmodel_train3b_20261125/<控制器>/data/mid_data3.npz`、`mid_data3b.npz`
- 开关和 train3b 的 D 一样：`REACHWM_GOAL_REF / APPROACH / QFIX = 1`；FDVEL 不开，因为 train3b 的 D 没用它。
- 代码仍然是 simpledata 的 `asrun/repo_b`：
  - `train.py`、`models.py` 和 train3b 冻结的版本 md5 相同；
  - `data.py`（174071cb）比 train3b 的（e0cabb83）只多 65 行，全是 FDVEL 开关下的内容，FDVEL 不开时完全一样；
  - `env.py` 只差 HEFT 控制器的设置，训练和评估都用不到。
- D 本身不重训，只作参考。train3b 的 D 是在 4090 的 1–3 号卡上训的（train3b 条目"卡和运行"），所以 N/D、C/D 里混着换卡的差别。N/C 两组都在 H20 上训，彼此配对，没有这个问题。

**各组**
- 只做 N 和 C，不做 C2。登记里第二阶段写的是"同样的 N/C 消融"。
- 种子 0、1、2 各训一次，一共 3 个控制器 × 2 组 × 3 个种子 = 18 个训练。用三个种子是因为：
  - train3b 的 D 本来就有三个种子；
  - LML 的真实局段数少（ScaleBFM 226 段，SONIC 129 段，HEFT 69 段），模型种子之间的差别会比 SIMPLE 大。
- `nw2_split.py` 就是 `nw_split.py`，只改了这几处，构造规则一样：
  1. 数据路径换掉；D 的训练段数从 train3b 的 `logs/train_D_s0_f1.log` 里读。
  2. train3b 的 t3b_mid 没写 `mid_*_src.npz`，所以中间点的段改成按重置编号核对：编号必须是它来源文件里的；走没走过，按日志核对。
  3. sd_range 的 39 个起点量原样抄进来。原因是 sd_range 一 import 就断言 FDVEL = 1。
  4. 输入检查改成对着本控制器的 lmlreal 记录做，走过的段和原地的段分开报。
- 训练的包装（`nw_train.py`）和第一阶段同一份。
- 和 train3b 的 D 训练命令核对过：`t3b_run.sh` 里 D 的数据顺序是 data3、data3b、mid_data3、mid_data3b，环境变量是 `REACHWM_GOAL_REF / APPROACH / QFIX = 1`、`REACHWM_X0_REF=0`，`t3b_train.py E3_ensemble --seed S --frac 1`，和这里一样。

**划分检查**（`lml/<控制器>/splits/split.log`，每份都是 SPLIT CHECK PASS；节选原文，完整日志见文件）：

```
--- scalebfm (lml/scalebfm/splits/split.log)
  md5 {'data3': '9eeede9a', 'data3b': 'b303a773', 'mid_data3': '9e567fa9', 'mid_data3b': 'b9ca63ac'}
logs: 45000 resets, walk-first 15164
  [ok] D's training split 847551 == 847551 (train3b logs/train_D_s0_f1.log)
  D : train 847551 segments (28786 resets; first 345596, chained 501955), walk-first 283850 segments / 9698 resets (task 0, d3 283850)
  N : train 563701 segments (19088 resets; first 230850, chained 332851), walk-first 0 segments / 0 resets (task 0, d3 0)
  C : train 563723 segments (19116 resets; first 229892, chained 333831), walk-first 190299 segments / 6462 resets (task 0, d3 190299)
  [ok] N's training split 563701 == D's training segments of in-place resets 563701
  [ok] C's training split 563723 vs N's 563701: +0.004 % (tolerance 1 %)
input check, lmlreal scalebfm settled (226 segments; walked 174, in place 52): share inside the arm's training p0.5 / p99.5 (sd_range's 39 quantities; below 'first' = walked, 'second' = in place), and ...
  D : joints 92.5 % (first 90.8, second 98.1); pelvis 92.5 % (first 90.2, second 100.0); hands 100.0 % (first 100.0, second 100.0); all 39 85.0 % (first 81.0, second 98.1)
  N : joints 91.6 % (first 89.7, second 98.1); pelvis 93.8 % (first 92.0, second 100.0); hands 100.0 % (first 100.0, second 100.0); all 39 85.4 % (first 81.6, second 98.1)
  C : joints 92.5 % (first 90.8, second 98.1); pelvis 91.2 % (first 88.5, second 100.0); hands 100.0 % (first 100.0, second 100.0); all 39 83.6 % (first 79.3, second 98.1)
SPLIT CHECK PASS
--- sonic (lml/sonic/splits/split.log)
  md5 {'data3': '51d1afa1', 'data3b': '1cfe41df', 'mid_data3': 'b199b720', 'mid_data3b': '7e22a714'}
logs: 45000 resets, walk-first 15164
  [ok] D's training split 1079768 == 1079768 (train3b logs/train_D_s0_f1.log)
  D : train 1079768 segments (28492 resets; first 428576, chained 651192), walk-first 364173 segments / 9624 resets (task 0, d3 364173)
  N : train 715595 segments (18868 resets; first 286014, chained 429581), walk-first 0 segments / 0 resets (task 0, d3 0)
  C : train 715611 segments (18902 resets; first 284001, chained 431610), walk-first 243033 segments / 6402 resets (task 0, d3 243033)
  [ok] N's training split 715595 == D's training segments of in-place resets 715595
  [ok] C's training split 715611 vs N's 715595: +0.002 % (tolerance 1 %)
input check, lmlreal sonic settled (129 segments; walked 97, in place 32): share inside the arm's training p0.5 / p99.5 (sd_range's 39 quantities; below 'first' = walked, 'second' = in place), and x0  ...
  D : joints 91.5 % (first 88.7, second 100.0); pelvis 99.2 % (first 100.0, second 96.9); hands 99.2 % (first 100.0, second 96.9); all 39 89.9 % (first 88.7, second 93.8)
  N : joints 91.5 % (first 88.7, second 100.0); pelvis 99.2 % (first 100.0, second 96.9); hands 99.2 % (first 100.0, second 96.9); all 39 89.9 % (first 88.7, second 93.8)
  C : joints 91.5 % (first 88.7, second 100.0); pelvis 99.2 % (first 100.0, second 96.9); hands 99.2 % (first 100.0, second 96.9); all 39 89.9 % (first 88.7, second 93.8)
SPLIT CHECK PASS
--- heft (lml/heft/splits/split.log)
  md5 {'data3': '8280b777', 'data3b': '743dafc9', 'mid_data3': '8437471c', 'mid_data3b': '6478dffe'}
logs: 45000 resets, walk-first 15164
  [ok] D's training split 499077 == 499077 (train3b logs/train_D_s0_f1.log)
  D : train 499077 segments (23815 resets; first 277559, chained 221518), walk-first 166848 segments / 7986 resets (task 0, d3 166848)
  N : train 332229 segments (15829 resets; first 185310, chained 146919), walk-first 0 segments / 0 resets (task 0, d3 0)
  C : train 332230 segments (15847 resets; first 184536, chained 147694), walk-first 112410 segments / 5335 resets (task 0, d3 112410)
  [ok] N's training split 332229 == D's training segments of in-place resets 332229
  [ok] C's training split 332230 vs N's 332229: +0.000 % (tolerance 1 %)
input check, lmlreal heft settled (69 segments; walked 40, in place 29): share inside the arm's training p0.5 / p99.5 (sd_range's 39 quantities; below 'first' = walked, 'second' = in place), and x0 co ...
  D : joints 79.7 % (first 65.0, second 100.0); pelvis 100.0 % (first 100.0, second 100.0); hands 98.6 % (first 97.5, second 100.0); all 39 79.7 % (first 65.0, second 100.0)
  N : joints 79.7 % (first 65.0, second 100.0); pelvis 100.0 % (first 100.0, second 100.0); hands 98.6 % (first 97.5, second 100.0); all 39 79.7 % (first 65.0, second 100.0)
  C : joints 79.7 % (first 65.0, second 100.0); pelvis 100.0 % (first 100.0, second 100.0); hands 98.6 % (first 97.5, second 100.0); all 39 79.7 % (first 65.0, second 100.0)
SPLIT CHECK PASS
```

- LML 的先走后伸只有 d3 走法（task 0），日志里 45000 个重置中有 15164 个先走。三个控制器的日志相同。
- 输入检查里，真实局里走过的段落在 N 训练范围里的比例和 C、D 几乎一样（ScaleBFM 全部 39 个量：N 85.4 %、C 83.6 %、D 85.0 %；SONIC 和 HEFT 三组相同）。所以 LML 上 N 的起点覆盖并不比 C 差。这一点在训练之前就知道了，预测是在它之前写的，不改。

**冒烟**（`smoke/lml_run`，ScaleBFM，种子 0，N 和 C 各 300 步，H20 的 0、1 号卡；checkpoint 测完已删）：
- 全流程跑通：划分检查 → 训练 → 评估。
- D 的核对通过，原文：

```
  reproduction: D0 median 1.96050 vs train3b lml_real.json 1.96050 (+0.00001); n 226 vs 226
  reproduction: D1 median 1.85640 vs train3b lml_real.json 1.85640 (+0.00000); n 226 vs 226
  reproduction: D2 median 2.22039 vs train3b lml_real.json 2.22039 (+0.00000); n 226 vs 226
REPRODUCTION PASS
```

- 冒烟也打出了 300 步模型的 N0/C0（0.949）。300 步的模型离收敛很远，这个数不代表任何结论；第二阶段的预测登记时就写好了，不改。

**评估**（`nw2_eval.py`，只用 CPU）
- **主评估**：`selfmodel_lmlreal_20261125/<控制器>/data/all.npz`。保留的段和 train3b 的 lmlreal 指标一样：没摔、没丢，末端停稳；途经点那几下也算进去。
  - 每个模型报误差中位数。
  - 区间按局重抽 4000 次。一局就是一个重置编号，编号 = 9 100 000 + 1000 × 任务 + 局号，7 个任务各 30 局；有记录的局 ScaleBFM 135 局、SONIC 128 局、HEFT 74 局。
  - LML 没有 SIMPLE 那种"同一局号、三个难度、布局几乎相同"的结构：每局的布局是单独抽的。所以不再按布局合并，按局重抽就够了。
- **登记的统计量**：三个种子各自配对，求 N_k/C_k，再取平均；它的 95 % 区间由同样的重抽得到。每个种子自己的比值也报。
- 第一下（reach_no 1）、第二下（reach_no 2）、走过的段、原地的段，各单独报一格，不进判据。
- **次要评估**：本控制器 data3 的留出集（val + test_iid，没摔、停稳），分成原地第一下、先走的第一下、原地连续伸手、走过之后的连续伸手，各组配对，按重置重抽 2000 次。
- **核对**：用 D0–D2 重算 train3b `lml_real.json` 里"全部"那一格的中位数，段数要相同，差不超过 2e-3 cm（那边是在 GPU 上算的）。对不上就停下。要对上的数是：

  | 控制器 | D0 | D1 | D2 |
  |---|---|---|---|
  | ScaleBFM | 1.96050 | 1.85640 | 2.22039 |
  | SONIC | 1.84117 | 1.83337 | 1.85337 |
  | HEFT | 24.30997 | 23.65374 | 24.72060 |

**判据怎么用到第二阶段**（阈值不变）
- 每个控制器单独判，用三个种子平均的 N/C 和它的区间，套登记表里的三行。
- **种子一致**：对应第一阶段的附加规则 1。
  - 如果平均落在第一行，三个种子各自的点估计都必须 ≤ 1.25；
  - 如果平均落在 > 1.25 那一行，三个种子都必须 > 1.10；
  - 不满足就降为中间那一行。
- 第一阶段附加规则 2、3 说的是 SIMPLE 的任务。LML 每个任务只有 10–30 段，按任务不报也不判。
- **HEFT 要单独说明**（在看第二阶段任何结果之前写下）：
  - train3b 里，D 在 HEFT 真实局上的中位数是 23.7–24.7 cm，直接把命令当预测是 28.4 cm，val 只有 3.2 cm。也就是说，所有模型在这批局上都错得很远。
  - 所以 HEFT 的 N/C 就算落在第一行，也只能说"去掉先走后伸的数据，不会让一个本来就错得很远的预测变得更差"，不能当成 HEFT 可以只采原地的证据。
  - HEFT 的结论同时看 data3 留出集里先走的那一格。

**运行**
- `nw2_run.sh` 在 H20 的 tmux `nowalk2` 里跑。
- 冻结的代码放在 `asrun2/code/`，冻结时间 2026-11-26T09:26:50Z。md5 记在 `asrun2/MD5.check`：
  - `nw2_split.py` d3f7c3b2，`nw2_eval.py` bcb8cb74，`nw2_run.sh` c3b2dd01，`nw_train.py` f000197d；
  - splits.npz：ScaleBFM 1d0f9833，SONIC 7ced1558，HEFT eea135d9。
- 卡：H20 只用 0、1 号卡，2、3 号卡留给搭环境的 agent。每张卡最多同时两个我们的训练，开之前要求已用显存加 12 GB ≤ 80 GB。
- 按协调人 09:24Z 的安排，第二阶段和第一阶段同时跑。第一阶段在 4090b 上，两边不抢卡。
- 第一阶段如果出现任何一条停下的情况（核对失败、N 的训练集里出现先走后伸的段），第二阶段的结果先不解读，查清以后再说。

### 修订 3：第二阶段加一项次要分析——HEFT 的走路数据和任务局对不对得上（2026-11-26 09:57Z；第二阶段的评估还没跑，第一阶段的结果也还没出来）

**这一项只做描述，不进判据。** 第二阶段的预测和判据都不变。

**为什么加**（协调人转达，约 09:48Z）
- LML 采数据时（d3_collect），三个控制器都用 hg_walk 走路。
- LML 的任务局里（lc3_run.walk_ctrl），HEFT 走的是 gait_heft；ScaleBFM 和 SONIC 采数据和跑任务都用 hg_walk。
- heft_walkstop 刚发现 HEFT 跟不好 hg_walk：要求走 ≤ 0.2 m 时几乎不迈步。
- 所以 HEFT 训练数据里先走后伸那部分的起点，可能和任务局里"走到站位以后"的起点对不上。
- 这一项也回答论文那边的问题：HEFT 的走路数据和任务局对不对得上。

**做法**（`nw2_walkfit.py`，只读，只用 CPU；三个控制器都报，HEFT 是重点）
- 用的段：lmlreal 记录里保留的段（和主评估一样：没摔、没丢，末端停稳），只看第一下（reach_no 1），按记录的 walked 分成"走过去了"和"原地"。
1. **误差**：直接读 nw2_eval 的 `per_segment.npz`，和它表里的数是同一批。
   - 每个模型（D0–D2、N0–N2、C0–C2）报中位数，95 % 区间按局重抽 4000 次；
   - 报 N/C（三个种子各自配对再平均）、每个种子的 N_k/C_k，以及 C/D；
   - 另报配对的"N_k 比 C_k 错得多"的段的比例，三个种子平均。它不受中位数在两种误差之间跳的影响（见下面 HEFT 的说明）；
   - 每格再限制到 chain 0（前面没有途经点的第一下）报一次。
2. **起点在不在训练范围里**：用 sd_range 的逻辑，对照 D 训练集里的四个参照集：全部；先走路的；先走路的第一下（chain 0）；原地的（就是 N 的训练集）。
   - range：39 个起点量都落在参照集的 0.5 %–99.5 % 之间，按关节、骨盆、手分组报，也报 39 个一起；
   - neighbour：在标准化的 x0 里，到参照集最近一段的距离，不超过参照集自己最近邻距离的 p99 就算在里面。标准化用整个训练集的均值和标准差，去掉常数列，并且去掉走路标志那一列（204），让标志本身不算距离；
   - nearer：真实段离最近的"先走路"训练段，比离最近的"原地"训练段更近的比例，旁边报基准比例（两个参照集大小之比）；
   - 比例都附 95 % 区间，按局重抽 1000 次。
- 运行：第二阶段的 nw2_eval 跑完以后，在 H20 的 CPU 上跑，nice 10，4 个线程。输出 `lml/run/<控制器>/walkfit/walkfit.{txt,json}`。
- 脚本冻结在 `asrun3/code/`，md5 a0cc94c7，冻结时间 2026-11-26T09:57:21Z（H20 的钟）。
- 冻结之前在 ScaleBFM 上冒烟跑过一次，用的是第二阶段冒烟（300 步）的 per_segment.npz。我只看了退出码（0）和行数（40 行），没有看内容，输出随后删掉了。

**已经知道的**（写预测之前就知道，不是这次的结果）
- train3b 的 lml_real.txt：HEFT 真实局的第一下，D 的中位数约 29 cm（27 段），区间从约 2 cm 到 34 cm。也就是两种情况混在一起：一部分约 2 cm，一部分约 30 cm。
- 同一个文件里，chain 0 的第一下（14 段）D 是 1.4–1.8 cm。所以 30 cm 那一类主要出在途经点之后的第一下。
- 修订 2 的输入检查：真实局里走过的段（所有伸手，不只第一下）落在 D 训练范围里（39 个量）的比例，ScaleBFM 81.0 %，SONIC 88.7 %，HEFT 65.0 %。

**预测**（在看这一项任何输出之前写下；第一项要等第二阶段评估，第二项只依赖数据，也还没看）

| 项 | ScaleBFM | SONIC | HEFT |
|---|---|---|---|
| 走过去的第一下，N/C（三个种子平均） | 1.00–1.40 | 0.95–1.40 | **0.85–1.20** |
| 同上，"N 比 C 错得多"的比例 | 0.50–0.70 | 0.45–0.70 | 0.40–0.60 |
| 走过去的第一下落在"先走路"参照集 range (39) 里的比例 | 0.65–0.85 | 0.70–0.90 | 0.35–0.65 |
| 同上，落在"原地"参照集（N 的训练集）里的比例 | 0.70–0.90 | 0.70–0.90 | 0.45–0.70 |
| nearer：离"先走路"训练段更近的比例（基准约 0.33） | 0.40–0.80 | 0.40–0.80 | 0.15–0.45 |

- **协调人要的那条**："HEFT 上走过去的那部分 N/C 比 ScaleBFM 更接近 1"，即 |N/C − 1| HEFT 小于 ScaleBFM。我给的把握是 55 %。
  - 理由：hg_walk 的走路数据如果对谁有用，应该是对采数据和跑任务用同一个步态的 ScaleBFM；HEFT 在 hg_walk 下几乎不迈步，它的"先走路"数据和原地数据差不多，任务局里用 gait_heft 真正走过去的起点，两边的数据都不像。
  - 把握不高的原因有两个：
    - 修订 2 的输入检查里，ScaleBFM 走过的真实段落在 N 的范围里的比例和 C 差不多（85.4 % 对 83.6 %），ScaleBFM 的 N/C 本身也可能接近 1；
    - HEFT 的第一下误差是 2 cm 和 30 cm 两种混在一起，只有十几段，中位数会在两种之间跳，比值可以离 1 很远，而这和走路数据对不对得上无关。所以 HEFT 那格要同时看"N 比 C 错得多"的比例和 chain 0 那一格。
- **第二项的方向**：
  - HEFT 走过去的起点，"先走路"参照集不比"原地"参照集更像（nearer 不高于基准）；
  - ScaleBFM 和 SONIC 的 nearer 高于基准。
  - 如果 HEFT 的 nearer 也明显高于基准，就说明 HEFT 的 hg_walk 数据虽然走得少，起点还是像任务局里走过去的样子，"对不上"的说法就不成立。

**怎么读**（不进判据，只决定结论怎么写）
- 第二项是这一项的主体：它直接回答"HEFT 的走路数据和任务局对不对得上"，不依赖模型。
- 第一项只作旁证：段数太少（HEFT 第一下走过去的预计十几段），区间会很宽。

## 结果（贴原始输出，不转述）
<!-- 取指标的命令 + 原始输出；完整日志、checkpoint、评测输出的路径。 -->

### 第一阶段（SIMPLE，ScaleBFM）

**运行**
- 4090b 的 0 号卡，2026-11-26T09:21:38Z–10:08:58Z。
- 前一次在 H20 上的运行被中断，见修订 1；之后接管成两个训练同时跑，见修订 1b。
- 5 个训练都跑满 30 000 步。挑 checkpoint 只用原地的 val（4260 段，N、C、C2 是同一批）。
- `run_b/run.log`（去掉了每分钟一行的等待记录）：

```
2026-11-26T09:21:38Z code ok (test 0); code /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/asrun/code; out /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b; jobs N:0:0 C:0:0 C2:0:0 N:1:0
SPLIT CHECK PASS
2026-11-26T09:21:38Z train
2026-11-26T09:21:38Z nvidia-smi before N s0: 0, 9701 MiB, 83 % 1, 17282 MiB, 93 % 
2026-11-26T09:21:38Z train N seed 0 on GPU 0: pid 320222
2026-11-26T09:22:28Z   nw_train arm N: {"arm": "N", "splits": "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/splits/splits.npz", "splits_md5": "4a47e698", "n_drop_resets": 14887, "zero_walk_flag": true,
2026-11-26T09:26:14Z code ok (test 0); code /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/asrun/code; out /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b; jobs C:0:0 C2:0:0 N:1:0 C:1:0
SPLIT CHECK PASS
2026-11-26T09:26:14Z train
2026-11-26T09:26:14Z takeover: nw_run_b.sh pid 319991 stopped; N s0 pid 320222 adopted; two of ours at a time on card 0
2026-11-26T09:26:14Z nvidia-smi before C s0: 0, 4247 MiB, 95 % 1, 18189 MiB, 96 % 
2026-11-26T09:26:14Z train C seed 0 on GPU 0: pid 328871
2026-11-26T09:27:04Z   nw_train arm C: {"arm": "C", "splits": "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/splits/splits.npz", "splits_md5": "4a47e698", "n_drop_resets": 10979, "zero_walk_flag": false
2026-11-26T09:38:05Z nvidia-smi before C2 s0: 0, 3675 MiB, 53 % 1, 18184 MiB, 93 % 
2026-11-26T09:38:05Z train C2 seed 0 on GPU 0: pid 339533
2026-11-26T09:38:55Z   nw_train arm C2: {"arm": "C2", "splits": "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/splits/splits.npz", "splits_md5": "4a47e698", "n_drop_resets": 10979, "zero_walk_flag": tru
2026-11-26T09:42:55Z nvidia-smi before N s1: 0, 2778 MiB, 18 % 1, 16370 MiB, 61 % 
2026-11-26T09:42:55Z train N seed 1 on GPU 0: pid 342464
2026-11-26T09:43:45Z   nw_train arm N: {"arm": "N", "splits": "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/splits/splits.npz", "splits_md5": "4a47e698", "n_drop_resets": 14887, "zero_walk_flag": true,
2026-11-26T09:53:46Z nvidia-smi before C s1: 0, 2778 MiB, 19 % 1, 17438 MiB, 94 % 
2026-11-26T09:53:46Z train C seed 1 on GPU 0: pid 360913
2026-11-26T09:54:26Z   nw_train arm C: {"arm": "C", "splits": "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/splits/splits.npz", "splits_md5": "4a47e698", "n_drop_resets": 10979, "zero_walk_flag": false
2026-11-26T10:04:38Z N s0 ended; stopped nw_run_b.sh (pid 319991) killed
E3_ensemble frac 1.0: best val median 1.674 cm (893.3 s) -> /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_C2_s0/E3_ensemble_f1
E3_ensemble frac 1.0: best val median 1.695 cm (900.9 s) -> /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_C_s0/E3_ensemble_f1
E3_ensemble frac 1.0: best val median 1.698 cm (609.7 s) -> /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_C_s1/E3_ensemble_f1
E3_ensemble frac 1.0: best val median 1.658 cm (886.1 s) -> /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_N_s0/E3_ensemble_f1
E3_ensemble frac 1.0: best val median 1.684 cm (801.5 s) -> /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_N_s1/E3_ensemble_f1
2026-11-26T10:04:38Z eval s0
2026-11-26T10:06:09Z eval s1
2026-11-26T10:07:25Z range N
```

**主评估：真实局主标签 settled，235 段**
- 种子 0（`run_b/eval_s0.txt`，md5 451044a9），原文：

```
arms {'D': ('/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/train/models_D_s0', False), 'N': ('/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_N_s0', True), 'C': ('/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_C_s0', False), 'C2': ('/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/run_b/models_C2_s0', True)}; splits /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/splits/splits.npz; FDVEL True; device cpu
== real main label, kept settled (PRIMARY): 235 segments, 120 episodes, 40 layouts; walked 235
  all: n 235 (40 layouts)
    median cm: D 0.705 [0.483, 1.031] | N 1.014 [0.916, 1.239] | C 0.873 [0.635, 1.186] | C2 0.914 [0.723, 1.434]
    p90 cm:    D 2.95 | N 2.84 | C 3.10 | C2 3.90;  attitude deg median: D 2.38 | N 2.55 | C 2.05 | C2 1.83
    ratio:     N/C 1.161 [0.976, 1.520] | N/C2 1.110 [0.838, 1.373] | C2/C 1.046 [0.955, 1.292] | N/D 1.438 [1.051, 2.003] | C/D 1.238 [0.995, 1.592] | C2/D 1.296 [1.081, 1.671]
  first reach (pre-grasp): n 116 (39 layouts)
    median cm: D 0.720 [0.487, 1.009] | N 1.044 [0.918, 1.380] | C 0.807 [0.651, 1.018] | C2 0.999 [0.728, 1.586]
    p90 cm:    D 2.58 | N 2.66 | C 2.92 | C2 3.43;  attitude deg median: D 2.39 | N 2.74 | C 2.20 | C2 2.08
    ratio:     N/C 1.294 [1.071, 1.555] | N/C2 1.045 [0.762, 1.394] | C2/C 1.238 [0.995, 1.669] | N/D 1.451 [1.128, 2.079] | C/D 1.121 [0.905, 1.572] | C2/D 1.388 [1.160, 1.903]
  second reach (entry): n 119 (40 layouts)
    median cm: D 0.622 [0.404, 1.096] | N 0.992 [0.901, 1.131] | C 0.896 [0.575, 1.300] | C2 0.863 [0.662, 1.169]
    p90 cm:    D 3.27 | N 2.82 | C 3.05 | C2 4.09;  attitude deg median: D 1.84 | N 2.50 | C 1.69 | C2 1.37
    ratio:     N/C 1.107 [0.799, 1.637] | N/C2 1.149 [0.903, 1.426] | C2/C 0.963 [0.743, 1.222] | N/D 1.595 [0.969, 2.332] | C/D 1.441 [1.068, 1.737] | C2/D 1.388 [0.919, 1.779]
  three tasks (no TabletopGraspMP): n 180 (30 layouts)
    median cm: D 0.486 [0.440, 0.722] | N 0.924 [0.862, 1.017] | C 0.652 [0.536, 0.895] | C2 0.728 [0.634, 0.922]
    p90 cm:    D 1.21 | N 1.53 | C 1.37 | C2 1.61;  attitude deg median: D 1.93 | N 2.43 | C 1.82 | C2 1.53
    ratio:     N/C 1.416 [1.103, 1.685] | N/C2 1.268 [1.040, 1.444] | C2/C 1.117 [0.984, 1.247] | N/D 1.900 [1.379, 2.077] | C/D 1.342 [1.141, 1.535] | C2/D 1.499 [1.245, 1.649]
  task XMovePickTeleop: n 60 (10 layouts)
    median cm: D 0.361 [0.331, 0.438] | N 0.924 [0.888, 0.957] | C 0.541 [0.520, 0.601] | C2 0.632 [0.592, 0.663]
    p90 cm:    D 0.50 | N 1.05 | C 0.69 | C2 0.73;  attitude deg median: D 1.70 | N 2.70 | C 1.73 | C2 1.24
    ratio:     N/C 1.707 [1.527, 1.761] | N/C2 1.462 [1.383, 1.545] | C2/C 1.168 [1.085, 1.183] | N/D 2.561 [2.092, 2.768] | C/D 1.500 [1.235, 1.657] | C2/D 1.751 [1.388, 1.907]
  task XMoveBendPickTeleop: n 60 (10 layouts)
    median cm: D 0.401 [0.294, 0.485] | N 0.691 [0.538, 0.820] | C 0.454 [0.366, 0.706] | C2 0.529 [0.305, 0.862]
    p90 cm:    D 0.74 | N 1.05 | C 0.87 | C2 1.00;  attitude deg median: D 1.23 | N 1.73 | C 1.86 | C2 1.16
    ratio:     N/C 1.521 [1.088, 2.004] | N/C2 1.306 [0.937, 1.928] | C2/C 1.165 [0.814, 1.614] | N/D 1.723 [1.349, 2.540] | C/D 1.132 [0.922, 1.548] | C2/D 1.319 [0.958, 2.146]
  task BendPickMP: n 60 (10 layouts)
    median cm: D 1.070 [0.994, 1.160] | N 1.363 [1.130, 1.520] | C 1.259 [1.132, 1.339] | C2 1.539 [1.155, 1.612]
    p90 cm:    D 1.33 | N 1.74 | C 1.59 | C2 1.67;  attitude deg median: D 3.46 | N 2.16 | C 2.13 | C2 2.55
    ratio:     N/C 1.082 [0.873, 1.260] | N/C2 0.885 [0.799, 1.096] | C2/C 1.223 [0.916, 1.349] | N/D 1.273 [1.009, 1.470] | C/D 1.177 [1.089, 1.240] | C2/D 1.439 [1.064, 1.565]
  task TabletopGraspMP: n 55 (10 layouts)
    median cm: D 2.263 [1.988, 30.168] | N 1.773 [1.525, 31.538] | C 2.625 [2.114, 29.308] | C2 2.990 [2.246, 29.784]
    p90 cm:    D 35.04 | N 35.98 | C 34.49 | C2 35.36;  attitude deg median: D 4.63 | N 5.96 | C 4.71 | C2 4.64
    ratio:     N/C 0.676 [0.663, 1.110] | N/C2 0.593 [0.586, 1.118] | C2/C 1.139 [0.946, 1.341] | N/D 0.784 [0.752, 1.047] | C/D 1.160 [0.924, 1.226] | C2/D 1.322 [0.931, 1.349]
  reproduction: D median 0.70526 vs simpledata 0.70526 (+0.00000)
```

- 种子 1（`run_b/eval_s1.txt`，md5 3ff3fe2d），原文：

```
== real main label, kept settled (PRIMARY): 235 segments, 120 episodes, 40 layouts; walked 235
  all: n 235 (40 layouts)
    median cm: D 0.705 [0.483, 1.031] | N 1.021 [0.971, 1.103] | C 0.940 [0.806, 1.037]
    p90 cm:    D 2.95 | N 2.62 | C 3.29;  attitude deg median: D 2.38 | N 3.04 | C 1.92
    ratio:     N/C 1.086 [1.002, 1.276] | N/D 1.447 [1.012, 2.099] | C/D 1.333 [0.979, 1.821]
  first reach (pre-grasp): n 116 (39 layouts)
    median cm: D 0.720 [0.487, 1.009] | N 1.147 [1.080, 1.231] | C 0.882 [0.812, 0.990]
    p90 cm:    D 2.58 | N 2.44 | C 2.89;  attitude deg median: D 2.39 | N 3.15 | C 2.24
    ratio:     N/C 1.300 [1.157, 1.443] | N/D 1.593 [1.170, 2.374] | C/D 1.226 [0.912, 1.826]
  second reach (entry): n 119 (40 layouts)
    median cm: D 0.622 [0.404, 1.096] | N 0.943 [0.880, 0.986] | C 0.978 [0.766, 1.117]
    p90 cm:    D 3.27 | N 2.61 | C 3.26;  attitude deg median: D 1.84 | N 2.90 | C 1.45
    ratio:     N/C 0.964 [0.830, 1.227] | N/D 1.518 [0.861, 2.287] | C/D 1.574 [0.993, 1.997]
  three tasks (no TabletopGraspMP): n 180 (30 layouts)
    median cm: D 0.486 [0.440, 0.722] | N 0.975 [0.943, 1.025] | C 0.806 [0.761, 0.882]
    p90 cm:    D 1.21 | N 1.28 | C 1.15;  attitude deg median: D 1.93 | N 2.20 | C 1.71
    ratio:     N/C 1.209 [1.088, 1.316] | N/D 2.005 [1.363, 2.230] | C/D 1.659 [1.193, 1.784]
  task XMovePickTeleop: n 60 (10 layouts)
    median cm: D 0.361 [0.331, 0.438] | N 0.930 [0.905, 0.954] | C 0.774 [0.752, 0.822]
    p90 cm:    D 0.50 | N 1.12 | C 1.00;  attitude deg median: D 1.70 | N 3.37 | C 1.43
    ratio:     N/C 1.201 [1.140, 1.241] | N/D 2.577 [2.088, 2.795] | C/D 2.147 [1.777, 2.287]
  task XMoveBendPickTeleop: n 60 (10 layouts)
    median cm: D 0.401 [0.294, 0.485] | N 1.001 [0.967, 1.058] | C 0.621 [0.487, 0.809]
    p90 cm:    D 0.74 | N 1.34 | C 0.98;  attitude deg median: D 1.23 | N 2.12 | C 1.46
    ratio:     N/C 1.613 [1.217, 2.164] | N/D 2.495 [2.022, 3.578] | C/D 1.547 [1.368, 1.903]
  task BendPickMP: n 60 (10 layouts)
    median cm: D 1.070 [0.994, 1.160] | N 1.074 [0.764, 1.146] | C 1.041 [0.975, 1.125]
    p90 cm:    D 1.33 | N 1.29 | C 1.36;  attitude deg median: D 3.46 | N 1.57 | C 1.92
    ratio:     N/C 1.032 [0.695, 1.132] | N/D 1.004 [0.686, 1.117] | C/D 0.973 [0.917, 1.056]
  task TabletopGraspMP: n 55 (10 layouts)
    median cm: D 2.263 [1.988, 30.168] | N 2.113 [1.635, 30.845] | C 2.419 [1.827, 29.325]
    p90 cm:    D 35.04 | N 35.57 | C 34.68;  attitude deg median: D 4.63 | N 6.37 | C 4.24
    ratio:     N/C 0.873 [0.719, 1.111] | N/D 0.934 [0.706, 1.031] | C/D 1.069 [0.914, 1.121]
  reproduction: D median 0.70526 vs simpledata 0.70526 (+0.00000)
```

- 两份都打出了 `REPRODUCTION PASS`：D 的四个中位数 0.70526、0.82076、1.88601、1.57564，差都是 +0.00000。

**次要评估**：两份 eval 里其余各表的比值行，原文：

```
[eval_s0.txt]
== real main label, kept ended: 240 segments, 120 episodes, 40 layouts; walked 240
  all: n 240 (40 layouts)
    ratio:     N/C 1.139 [0.952, 1.511] | N/C2 1.037 [0.820, 1.363] | C2/C 1.098 [0.954, 1.331] | N/D 1.400 [1.040, 1.988] | C/D 1.229 [0.996, 1.578] | C2/D 1.350 [1.076, 1.664]
  first reach (pre-grasp): n 120 (40 layouts)
    ratio:     N/C 1.231 [1.062, 1.553] | N/C2 1.007 [0.763, 1.394] | C2/C 1.222 [1.000, 1.682] | N/D 1.429 [1.112, 2.078] | C/D 1.161 [0.902, 1.556] | C2/D 1.419 [1.171, 1.883]
  second reach (entry): n 120 (40 layouts)
    ratio:     N/C 1.111 [0.804, 1.641] | N/C2 1.151 [0.892, 1.425] | C2/C 0.965 [0.754, 1.233] | N/D 1.501 [0.966, 2.370] | C/D 1.350 [1.070, 1.742] | C2/D 1.304 [0.931, 1.780]
  three tasks (no TabletopGraspMP): n 180 (30 layouts)
    ratio:     N/C 1.416 [1.102, 1.684] | N/C2 1.268 [1.033, 1.437] | C2/C 1.117 [0.990, 1.253] | N/D 1.900 [1.354, 2.073] | C/D 1.342 [1.134, 1.548] | C2/D 1.499 [1.247, 1.654]
  task XMovePickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.707 [1.538, 1.763] | N/C2 1.462 [1.390, 1.546] | C2/C 1.168 [1.090, 1.183] | N/D 2.561 [2.093, 2.759] | C/D 1.500 [1.237, 1.654] | C2/D 1.751 [1.395, 1.908]
  task XMoveBendPickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.521 [1.088, 2.008] | N/C2 1.306 [0.938, 1.878] | C2/C 1.165 [0.845, 1.622] | N/D 1.723 [1.356, 2.547] | C/D 1.132 [0.917, 1.553] | C2/D 1.319 [0.962, 2.145]
  task BendPickMP: n 60 (10 layouts)
    ratio:     N/C 1.082 [0.859, 1.258] | N/C2 0.885 [0.800, 1.097] | C2/C 1.223 [0.917, 1.343] | N/D 1.273 [1.004, 1.475] | C/D 1.177 [1.085, 1.241] | C2/D 1.439 [1.067, 1.569]
  task TabletopGraspMP: n 60 (10 layouts)
    ratio:     N/C 0.875 [0.672, 1.109] | N/C2 0.679 [0.609, 1.091] | C2/C 1.289 [0.948, 1.324] | N/D 0.872 [0.751, 1.046] | C/D 0.997 [0.924, 1.242] | C2/D 1.285 [0.956, 1.322]
== real task label, kept ended: 240 segments, 120 episodes, 40 layouts; walked 240
  all: n 240 (40 layouts)
    ratio:     N/C 1.171 [0.804, 1.677] | N/C2 1.083 [0.773, 1.519] | C2/C 1.081 [0.869, 1.239] | N/D 1.267 [0.930, 2.425] | C/D 1.082 [0.970, 1.741] | C2/D 1.170 [0.954, 1.854]
  first reach (pre-grasp): n 120 (40 layouts)
    ratio:     N/C 1.218 [0.869, 1.724] | N/C2 1.033 [0.769, 1.603] | C2/C 1.180 [0.937, 1.460] | N/D 1.416 [1.021, 2.690] | C/D 1.163 [0.883, 1.820] | C2/D 1.372 [1.039, 2.009]
  second reach (entry): n 120 (40 layouts)
    ratio:     N/C 1.158 [0.737, 1.721] | N/C2 1.141 [0.794, 1.606] | C2/C 1.014 [0.727, 1.229] | N/D 1.235 [0.826, 2.392] | C/D 1.067 [0.952, 1.683] | C2/D 1.082 [0.809, 1.795]
  three tasks (no TabletopGraspMP): n 180 (30 layouts)
    ratio:     N/C 1.563 [1.066, 1.891] | N/C2 1.426 [1.098, 1.736] | C2/C 1.096 [0.889, 1.243] | N/D 2.295 [1.093, 2.574] | C/D 1.469 [1.019, 1.751] | C2/D 1.610 [0.992, 1.821]
  task XMovePickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.983 [1.691, 2.183] | N/C2 1.890 [1.537, 2.141] | C2/C 1.049 [0.948, 1.184] | N/D 2.825 [2.481, 3.275] | C/D 1.425 [1.339, 1.581] | C2/D 1.495 [1.339, 1.700]
  task XMoveBendPickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.588 [1.130, 2.124] | N/C2 1.394 [1.086, 1.937] | C2/C 1.139 [0.728, 1.532] | N/D 2.050 [1.513, 2.488] | C/D 1.291 [1.040, 1.517] | C2/D 1.470 [0.906, 1.894]
  task BendPickMP: n 60 (10 layouts)
    ratio:     N/C 1.082 [0.701, 1.156] | N/C2 0.974 [0.733, 1.088] | C2/C 1.111 [0.956, 1.157] | N/D 1.293 [0.854, 1.413] | C/D 1.195 [1.142, 1.283] | C2/D 1.328 [1.137, 1.422]
  task TabletopGraspMP: n 60 (10 layouts)
    ratio:     N/C 0.877 [0.597, 1.099] | N/C2 0.667 [0.509, 1.089] | C2/C 1.315 [0.992, 1.334] | N/D 0.890 [0.661, 1.048] | C/D 1.015 [0.942, 1.247] | C2/D 1.335 [0.940, 1.409]
  reproduction: D median 0.82076 vs simpledata 0.82076 (+0.00000)
== data3 held out (kept settled): val 3252, test_iid 3116 segments
  val all (simpledata's denominator): n 3252 (2381 resets)
    ratio:     N/C 1.303 [1.251, 1.373] | N/C2 1.299 [1.240, 1.364] | C2/C 1.003 [0.977, 1.036] | N/D 1.407 [1.350, 1.479] | C/D 1.080 [1.047, 1.114] | C2/D 1.083 [1.045, 1.126]
  first, in place (val + test_iid): n 2205 (2205 resets)
    ratio:     N/C 0.903 [0.876, 0.939] | N/C2 0.903 [0.872, 0.938] | C2/C 1.001 [0.970, 1.036] | N/D 1.032 [1.000, 1.072] | C/D 1.142 [1.100, 1.188] | C2/D 1.143 [1.102, 1.189]
  chained, in place (val + test_iid): n 2066 (2066 resets)
    ratio:     N/C 0.992 [0.953, 1.026] | N/C2 0.986 [0.948, 1.018] | C2/C 1.006 [0.978, 1.034] | N/D 1.082 [1.045, 1.110] | C/D 1.091 [1.051, 1.129] | C2/D 1.098 [1.057, 1.134]
  first, task walk (test_iid): n 517 (517 resets)
    ratio:     N/C 1.460 [1.331, 1.575] | N/C2 1.479 [1.372, 1.620] | C2/C 0.988 [0.917, 1.034] | N/D 1.694 [1.569, 1.836] | C/D 1.160 [1.094, 1.260] | C2/D 1.146 [1.063, 1.217]
  chained, task walk (test_iid): n 512 (512 resets)
    ratio:     N/C 1.054 [0.985, 1.150] | N/C2 1.027 [0.962, 1.160] | C2/C 1.026 [0.944, 1.094] | N/D 1.108 [1.044, 1.200] | C/D 1.051 [0.984, 1.127] | C2/D 1.079 [0.967, 1.151]
  first, d3 walk (val): n 543 (543 resets)
    ratio:     N/C 2.174 [2.013, 2.348] | N/C2 2.152 [1.951, 2.313] | C2/C 1.010 [0.961, 1.084] | N/D 2.461 [2.281, 2.614] | C/D 1.132 [1.060, 1.194] | C2/D 1.144 [1.068, 1.233]
  chained, d3 walk (val): n 525 (525 resets)
    ratio:     N/C 1.481 [1.363, 1.575] | N/C2 1.447 [1.350, 1.551] | C2/C 1.024 [0.966, 1.069] | N/D 1.610 [1.476, 1.690] | C/D 1.087 [1.023, 1.146] | C2/D 1.113 [1.039, 1.159]
  test_iid all: n 3116 (2251 resets)
    ratio:     N/C 1.071 [1.032, 1.119] | N/C2 1.090 [1.049, 1.129] | C2/C 0.982 [0.957, 1.019] | N/D 1.194 [1.153, 1.251] | C/D 1.115 [1.069, 1.165] | C2/D 1.095 [1.060, 1.145]
  reproduction: data3 val all D median 1.88601 vs simpledata 1.88601 (+0.00000)
  reproduction: data3 test_iid all D median 1.57564 vs simpledata 1.57564 (+0.00000)
== data3b held out (kept settled): val 2923, test_iid 2920 segments
  val all: n 2923 (1295 resets)
    ratio:     N/C 1.201 [1.145, 1.261] | N/C2 1.177 [1.129, 1.237] | C2/C 1.020 [0.990, 1.044] | N/D 1.307 [1.250, 1.373] | C/D 1.088 [1.056, 1.123] | C2/D 1.110 [1.076, 1.141]
  pre-grasp first + entry chain 1, task walk (test_iid): n 140 (86 resets)
    ratio:     N/C 1.545 [1.293, 1.744] | N/C2 1.549 [1.288, 1.738] | C2/C 0.998 [0.904, 1.087] | N/D 1.625 [1.354, 1.912] | C/D 1.051 [0.954, 1.207] | C2/D 1.049 [0.965, 1.198]
  pre-grasp first + entry chain 1, in place (val + test_iid): n 582 (333 resets)
    ratio:     N/C 0.972 [0.885, 1.066] | N/C2 0.939 [0.859, 1.031] | C2/C 1.034 [0.963, 1.112] | N/D 1.092 [0.997, 1.174] | C/D 1.124 [1.024, 1.189] | C2/D 1.163 [1.063, 1.255]
  pre-grasp first + entry chain 1, d3 walk (val): n 170 (94 resets)
    ratio:     N/C 2.147 [1.854, 2.578] | N/C2 2.115 [1.780, 2.524] | C2/C 1.015 [0.933, 1.137] | N/D 2.476 [2.193, 2.984] | C/D 1.154 [1.025, 1.303] | C2/D 1.171 [1.074, 1.337]
  test_iid all: n 2920 (1316 resets)
    ratio:     N/C 1.086 [1.043, 1.128] | N/C2 1.090 [1.044, 1.128] | C2/C 0.997 [0.973, 1.028] | N/D 1.198 [1.154, 1.244] | C/D 1.103 [1.068, 1.141] | C2/D 1.099 [1.067, 1.140]
[eval_s1.txt]
== real main label, kept ended: 240 segments, 120 episodes, 40 layouts; walked 240
  all: n 240 (40 layouts)
    ratio:     N/C 1.081 [0.994, 1.258] | N/D 1.414 [0.996, 2.083] | C/D 1.307 [0.969, 1.815]
  first reach (pre-grasp): n 120 (40 layouts)
    ratio:     N/C 1.286 [1.151, 1.434] | N/D 1.567 [1.143, 2.347] | C/D 1.218 [0.907, 1.835]
  second reach (entry): n 120 (40 layouts)
    ratio:     N/C 0.960 [0.827, 1.228] | N/D 1.422 [0.860, 2.302] | C/D 1.481 [0.992, 2.023]
  three tasks (no TabletopGraspMP): n 180 (30 layouts)
    ratio:     N/C 1.209 [1.091, 1.317] | N/D 2.005 [1.343, 2.210] | C/D 1.659 [1.178, 1.785]
  task XMovePickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.201 [1.144, 1.240] | N/D 2.577 [2.093, 2.785] | C/D 2.147 [1.779, 2.284]
  task XMoveBendPickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.613 [1.222, 2.160] | N/D 2.495 [2.024, 3.573] | C/D 1.547 [1.366, 1.922]
  task BendPickMP: n 60 (10 layouts)
    ratio:     N/C 1.032 [0.698, 1.139] | N/D 1.004 [0.694, 1.118] | C/D 0.973 [0.912, 1.055]
  task TabletopGraspMP: n 60 (10 layouts)
    ratio:     N/C 0.844 [0.719, 1.092] | N/D 0.820 [0.733, 1.030] | C/D 0.973 [0.923, 1.112]
== real task label, kept ended: 240 segments, 120 episodes, 40 layouts; walked 240
  all: n 240 (40 layouts)
    ratio:     N/C 1.189 [0.961, 1.432] | N/D 1.390 [0.966, 2.705] | C/D 1.169 [0.908, 2.077]
  first reach (pre-grasp): n 120 (40 layouts)
    ratio:     N/C 1.307 [1.054, 1.569] | N/D 1.547 [1.094, 3.058] | C/D 1.183 [0.849, 2.196]
  second reach (entry): n 120 (40 layouts)
    ratio:     N/C 0.954 [0.773, 1.257] | N/D 1.148 [0.747, 2.309] | C/D 1.203 [0.933, 1.981]
  three tasks (no TabletopGraspMP): n 180 (30 layouts)
    ratio:     N/C 1.318 [1.020, 1.480] | N/D 2.419 [1.075, 2.801] | C/D 1.836 [0.986, 2.071]
  task XMovePickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.471 [1.369, 1.609] | N/D 2.922 [2.738, 3.478] | C/D 1.987 [1.873, 2.271]
  task XMoveBendPickTeleop: n 60 (10 layouts)
    ratio:     N/C 1.530 [1.270, 2.169] | N/D 2.608 [2.187, 3.335] | C/D 1.705 [1.337, 1.964]
  task BendPickMP: n 60 (10 layouts)
    ratio:     N/C 0.929 [0.704, 1.058] | N/D 0.915 [0.709, 1.062] | C/D 0.984 [0.953, 1.062]
  task TabletopGraspMP: n 60 (10 layouts)
    ratio:     N/C 0.768 [0.719, 1.078] | N/D 0.830 [0.754, 1.036] | C/D 1.080 [0.909, 1.206]
  reproduction: D median 0.82076 vs simpledata 0.82076 (+0.00000)
== data3 held out (kept settled): val 3252, test_iid 3116 segments
  val all (simpledata's denominator): n 3252 (2381 resets)
    ratio:     N/C 1.303 [1.238, 1.356] | N/D 1.438 [1.371, 1.506] | C/D 1.104 [1.073, 1.143]
  first, in place (val + test_iid): n 2205 (2205 resets)
    ratio:     N/C 0.866 [0.838, 0.902] | N/D 1.030 [0.999, 1.074] | C/D 1.189 [1.149, 1.237]
  chained, in place (val + test_iid): n 2066 (2066 resets)
    ratio:     N/C 0.999 [0.969, 1.033] | N/D 1.091 [1.053, 1.123] | C/D 1.092 [1.052, 1.120]
  first, task walk (test_iid): n 517 (517 resets)
    ratio:     N/C 1.457 [1.343, 1.538] | N/D 1.749 [1.622, 1.876] | C/D 1.201 [1.136, 1.286]
  chained, task walk (test_iid): n 512 (512 resets)
    ratio:     N/C 1.095 [1.008, 1.151] | N/D 1.124 [1.045, 1.191] | C/D 1.026 [0.976, 1.107]
  first, d3 walk (val): n 543 (543 resets)
    ratio:     N/C 2.162 [2.022, 2.325] | N/D 2.481 [2.316, 2.655] | C/D 1.148 [1.086, 1.212]
  chained, d3 walk (val): n 525 (525 resets)
    ratio:     N/C 1.458 [1.372, 1.551] | N/D 1.579 [1.459, 1.662] | C/D 1.083 [1.013, 1.124]
  test_iid all: n 3116 (2251 resets)
    ratio:     N/C 1.062 [1.027, 1.109] | N/D 1.201 [1.163, 1.263] | C/D 1.131 [1.094, 1.179]
  reproduction: data3 val all D median 1.88601 vs simpledata 1.88601 (+0.00000)
  reproduction: data3 test_iid all D median 1.57564 vs simpledata 1.57564 (+0.00000)
== data3b held out (kept settled): val 2923, test_iid 2920 segments
  val all: n 2923 (1295 resets)
    ratio:     N/C 1.224 [1.164, 1.265] | N/D 1.333 [1.279, 1.385] | C/D 1.089 [1.066, 1.130]
  pre-grasp first + entry chain 1, task walk (test_iid): n 140 (86 resets)
    ratio:     N/C 1.411 [1.194, 1.692] | N/D 1.601 [1.346, 1.940] | C/D 1.134 [1.003, 1.274]
  pre-grasp first + entry chain 1, in place (val + test_iid): n 582 (333 resets)
    ratio:     N/C 0.901 [0.837, 0.978] | N/D 1.028 [0.958, 1.108] | C/D 1.141 [1.059, 1.220]
  pre-grasp first + entry chain 1, d3 walk (val): n 170 (94 resets)
    ratio:     N/C 2.363 [1.965, 2.701] | N/D 2.591 [2.306, 3.025] | C/D 1.096 [1.032, 1.289]
  test_iid all: n 2920 (1316 resets)
    ratio:     N/C 1.106 [1.060, 1.143] | N/D 1.213 [1.171, 1.252] | C/D 1.096 [1.065, 1.140]
```

**起点在不在 N 的训练范围里**（`run_b/range_N.txt`，md5 a88c045a）
- 注意：sd_range 的输出里写的"D median"，这里指的是 N s0 的模型。nw_range 换掉的是训练集和模型，没有改输出的字。
- 原文节选：

```
nw_range: arm N, zero walk flag True; sd_range /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/asrun/repo/scripts/experiments/selfmodel_simpledata_20261126/sd_range.py
table /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/asrun/repo_b/scripts/experiments/selfmodel_data3_20261125/d3_reach_table.json md5 9d4c716a5ad271ef92cb05ac83aa976f; device cpu
nw_range arm N: training split 542459 segments (14887 resets dropped)
training set: 542459 segments (19523 resets) from /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/data3/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/data3b/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/train/data/mid_data3.npz,/wu
SIMPLE val (data3 val, all, kept settled): 3252 segments, D median 2.654 cm
  reference, val inside: joints 59.9 %, pelvis 92.6 %, hands 94.2 %, neighbour 96.1 %, table 100.0 %, range (39) 50.4 %, all three 50.4 %; nearest-neighbour distance median 7.089
== main label, kept settled (primary): 235 segments from 120 episodes (/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/real/main/data/all.npz); in place 0, walk first 235; D median 1.014 cm = 0.382x SIMPLE val; nearest-neighbour distance median 5.727 (training p99 20.960)
  joints     : inside 78.7 % [74.1, 83.1] (185 / 50 out); error inside 1.01 cm (0.38x), outside 1.06 cm (0.40x)
               walk first n 235 inside 78.7 % (err in 1.01 / out 1.06); first reach n 116 inside 84.5 % (err in 1.17 / out 0.54); second reach n 119 inside 73.1 % (err in 0.93 / out 1.32)
  pelvis     : inside 98.7 % [97.0, 100.0] (232 / 3 out); error inside 1.01 cm (0.38x), outside 35.06 cm (13.21x)
               walk first n 235 inside 98.7 % (err in 1.01 / out 35.06); first reach n 116 inside 100.0 % (err in 1.04 / out -); second reach n 119 inside 97.5 % (err in 0.98 / out 35.06)
  neighbour  : inside 99.1 % [97.8, 100.0] (233 / 2 out); error inside 1.01 cm (0.38x), outside 29.36 cm (11.06x)
               walk first n 235 inside 99.1 % (err in 1.01 / out 29.36); first reach n 116 inside 100.0 % (err in 1.04 / out -); second reach n 119 inside 98.3 % (err in 0.98 / out 29.36)
  range (39) : inside 78.7 % [73.8, 83.1] (185 / 50 out); error inside 1.01 cm (0.38x), outside 1.06 cm (0.40x)
               walk first n 235 inside 78.7 % (err in 1.01 / out 1.06); first reach n 116 inside 84.5 % (err in 1.17 / out 0.54); second reach n 119 inside 73.1 % (err in 0.93 / out 1.32)
```

- 对照：D 在同一批段上的结果是 range (39) 86.8 %，第一下 100.0 %，第二下 73.9 %（simpledata `train/eval4.txt`）。

**文件**（本目录下）
- `run_b/`：`run.log`、`logs/train_*.log`、`models_{N,C,C2}_s0`、`models_{N,C}_s1`；
- `eval_s0/eval.json`（md5 1330c526）、`eval_s1/eval.json`（md5 9963a0d9），以及每段误差的 `per_segment.npz`；
- `range_N.json`。

### 第二阶段（LocoManiLab；ScaleBFM、SONIC、HEFT）

**运行**
- H20 的 0、1 号卡，2026-11-26T09:27:52Z–10:33:21Z。18 个训练各跑满 30 000 步，每个 497–825 s。
- 每次开训练前看的 nvidia-smi：0、1 号卡最多用到 5827 MiB。2、3 号卡一直是 912 MiB（只有 keepalive），我们没用。
- 挑 checkpoint 只用 N 和 C 共用的原地 val，和第一阶段一样。
- `lml/run/run.log` 节选。去掉了 18 组"nvidia-smi before / train … on GPU / nw_train arm"启动行和评估打出的 PRIMARY 行，后者见下面的 eval.txt；`$P` = `/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126`：

```
2026-11-26T09:27:52Z code ok (test 0); code $P/asrun2/code; out $P/lml/run; controllers scalebfm sonic heft; seeds 0 1 2
scalebfm: SPLIT CHECK PASS
sonic: SPLIT CHECK PASS
heft: SPLIT CHECK PASS
2026-11-26T09:27:52Z train
train_heft_C_s0.log:E3_ensemble frac 1.0: best val median 3.408 cm (786.2 s) -> heft/models_C_s0/E3_ensemble_f1
train_heft_C_s1.log:E3_ensemble frac 1.0: best val median 3.455 cm (811.0 s) -> heft/models_C_s1/E3_ensemble_f1
train_heft_C_s2.log:E3_ensemble frac 1.0: best val median 3.383 cm (723.8 s) -> heft/models_C_s2/E3_ensemble_f1
train_heft_N_s0.log:E3_ensemble frac 1.0: best val median 3.258 cm (801.7 s) -> heft/models_N_s0/E3_ensemble_f1
train_heft_N_s1.log:E3_ensemble frac 1.0: best val median 3.246 cm (825.0 s) -> heft/models_N_s1/E3_ensemble_f1
train_heft_N_s2.log:E3_ensemble frac 1.0: best val median 3.224 cm (820.5 s) -> heft/models_N_s2/E3_ensemble_f1
train_scalebfm_C_s0.log:E3_ensemble frac 1.0: best val median 2.460 cm (518.0 s) -> scalebfm/models_C_s0/E3_ensemble_f1
train_scalebfm_C_s1.log:E3_ensemble frac 1.0: best val median 2.425 cm (515.7 s) -> scalebfm/models_C_s1/E3_ensemble_f1
train_scalebfm_C_s2.log:E3_ensemble frac 1.0: best val median 2.434 cm (708.0 s) -> scalebfm/models_C_s2/E3_ensemble_f1
train_scalebfm_N_s0.log:E3_ensemble frac 1.0: best val median 2.408 cm (497.4 s) -> scalebfm/models_N_s0/E3_ensemble_f1
train_scalebfm_N_s1.log:E3_ensemble frac 1.0: best val median 2.399 cm (544.5 s) -> scalebfm/models_N_s1/E3_ensemble_f1
train_scalebfm_N_s2.log:E3_ensemble frac 1.0: best val median 2.375 cm (608.9 s) -> scalebfm/models_N_s2/E3_ensemble_f1
train_sonic_C_s0.log:E3_ensemble frac 1.0: best val median 3.344 cm (638.8 s) -> sonic/models_C_s0/E3_ensemble_f1
train_sonic_C_s1.log:E3_ensemble frac 1.0: best val median 3.373 cm (662.2 s) -> sonic/models_C_s1/E3_ensemble_f1
train_sonic_C_s2.log:E3_ensemble frac 1.0: best val median 3.373 cm (671.4 s) -> sonic/models_C_s2/E3_ensemble_f1
train_sonic_N_s0.log:E3_ensemble frac 1.0: best val median 3.348 cm (632.4 s) -> sonic/models_N_s0/E3_ensemble_f1
train_sonic_N_s1.log:E3_ensemble frac 1.0: best val median 3.351 cm (660.9 s) -> sonic/models_N_s1/E3_ensemble_f1
train_sonic_N_s2.log:E3_ensemble frac 1.0: best val median 3.338 cm (703.3 s) -> sonic/models_N_s2/E3_ensemble_f1
2026-11-26T10:30:23Z eval scalebfm
2026-11-26T10:31:26Z eval sonic
2026-11-26T10:32:32Z eval heft
2026-11-26T10:33:21Z done
```

- 挑 checkpoint 用的原地 val，N/C（种子 0 / 1 / 2）：ScaleBFM 0.979 / 0.989 / 0.976，SONIC 1.001 / 0.993 / 0.990，HEFT 0.956 / 0.940 / 0.953。

**主评估：lmlreal 真实局，保留的段**（没摔、没丢、末端停稳；三个种子各自配对，再取平均；按局重抽 4000 次）

- ScaleBFM（`lml/run/scalebfm/eval.txt`，md5 e0f10033），原文：

```
== lmlreal scalebfm (PRIMARY): 226 kept segments (not fallen, settled end) of 316, 129 episodes; reach_no 1 117, 2 82, 0 27; walked 174
  all (via points included): n 226 (129 episodes)
    median cm: D0 1.961 [1.548, 3.020] | D1 1.856 [1.320, 2.809] | D2 2.220 [1.540, 3.079] | N0 2.180 [1.769, 3.064] | N1 2.344 [1.823, 3.019] | N2 2.198 [1.732, 3.217] | C0 2.094 [1.622, 2.667] | C1 2.214 [1.619, 3.322] | C2 2.066 [1.634, 2.659]
    p90 cm:    D0 30.90 | D1 30.71 | D2 30.61 | N0 30.64 | N1 30.62 | N2 30.53 | C0 30.81 | C1 30.75 | C2 30.64
    attitude deg median: D0 3.07 | D1 3.58 | D2 3.35 | N0 3.93 | N1 3.61 | N2 4.05 | C0 3.50 | C1 3.37 | C2 3.51
    N/C: mean over seeds 1.055 [0.954, 1.229]; per seed N0/C0 1.041 [0.949, 1.253], N1/C1 1.059 [0.873, 1.327], N2/C2 1.064 [0.934, 1.312]
    C/D: mean over seeds 1.064 [0.909, 1.207]; per seed C0/D0 1.068 [0.810, 1.230], C1/D1 1.192 [0.991, 1.459], C2/D2 0.930 [0.813, 1.130]
    N/D: mean over seeds 1.122 [0.968, 1.345]; per seed N0/D0 1.112 [0.875, 1.313], N1/D1 1.263 [1.025, 1.657], N2/D2 0.990 [0.877, 1.256]
  第一下 (reach_no 1): n 117 (117 episodes)
    median cm: D0 1.336 [0.991, 1.480] | D1 1.224 [1.063, 1.352] | D2 1.240 [1.084, 1.450] | N0 1.679 [1.201, 2.184] | N1 1.823 [1.221, 2.334] | N2 1.626 [1.278, 2.043] | C0 1.420 [1.215, 1.593] | C1 1.212 [1.022, 1.526] | C2 1.339 [1.070, 1.600]
    p90 cm:    D0 30.57 | D1 29.78 | D2 29.37 | N0 29.48 | N1 29.70 | N2 28.79 | C0 29.28 | C1 29.71 | C2 29.82
    attitude deg median: D0 2.29 | D1 2.74 | D2 2.33 | N0 2.81 | N1 2.79 | N2 3.20 | C0 2.32 | C1 2.34 | C2 2.34
    N/C: mean over seeds 1.300 [1.045, 1.540]; per seed N0/C0 1.182 [0.943, 1.444], N1/C1 1.504 [1.060, 1.859], N2/C2 1.215 [1.023, 1.529]
    C/D: mean over seeds 1.044 [0.944, 1.181]; per seed C0/D0 1.063 [0.936, 1.330], C1/D1 0.990 [0.879, 1.186], C2/D2 1.080 [0.883, 1.265]
    N/D: mean over seeds 1.353 [1.107, 1.586]; per seed N0/D0 1.257 [0.999, 1.550], N1/D1 1.489 [1.092, 1.802], N2/D2 1.312 [1.119, 1.587]
  第二下 (reach_no 2): n 82 (82 episodes)
    median cm: D0 2.985 [2.186, 3.800] | D1 2.543 [1.899, 3.494] | D2 2.793 [2.231, 3.756] | N0 2.517 [1.918, 3.552] | N1 2.793 [1.827, 3.655] | N2 2.588 [1.956, 3.913] | C0 2.632 [2.115, 3.602] | C1 3.155 [2.577, 3.991] | C2 2.659 [2.233, 3.568]
    p90 cm:    D0 9.26 | D1 8.08 | D2 8.50 | N0 8.07 | N1 8.13 | N2 8.77 | C0 9.18 | C1 9.15 | C2 8.62
    attitude deg median: D0 4.23 | D1 4.35 | D2 3.95 | N0 4.32 | N1 4.57 | N2 4.54 | C0 4.49 | C1 3.91 | C2 3.98
    N/C: mean over seeds 0.938 [0.801, 1.114]; per seed N0/C0 0.956 [0.823, 1.168], N1/C1 0.885 [0.679, 1.051], N2/C2 0.973 [0.796, 1.285]
    C/D: mean over seeds 1.025 [0.932, 1.143]; per seed C0/D0 0.882 [0.766, 1.055], C1/D1 1.241 [1.043, 1.461], C2/D2 0.952 [0.850, 1.074]
    N/D: mean over seeds 0.956 [0.862, 1.132]; per seed N0/D0 0.843 [0.742, 1.020], N1/D1 1.098 [0.884, 1.361], N2/D2 0.927 [0.747, 1.245]
  walked: n 174 (96 episodes)
    median cm: D0 2.308 [1.578, 3.467] | D1 2.180 [1.359, 3.249] | D2 2.404 [1.736, 3.319] | N0 2.318 [1.934, 3.322] | N1 2.629 [2.088, 3.383] | N2 2.400 [1.972, 3.554] | C0 2.245 [1.849, 3.282] | C1 2.653 [1.634, 3.717] | C2 2.292 [1.805, 3.202]
    p90 cm:    D0 31.35 | D1 31.31 | D2 31.43 | N0 31.44 | N1 31.17 | N2 31.21 | C0 31.62 | C1 31.57 | C2 32.07
    attitude deg median: D0 3.36 | D1 4.03 | D2 3.63 | N0 4.16 | N1 3.93 | N2 4.60 | C0 4.21 | C1 3.55 | C2 3.80
    N/C: mean over seeds 1.024 [0.934, 1.240]; per seed N0/C0 1.033 [0.926, 1.280], N1/C1 0.991 [0.843, 1.370], N2/C2 1.047 [0.906, 1.358]
    C/D: mean over seeds 1.048 [0.892, 1.217]; per seed C0/D0 0.973 [0.783, 1.244], C1/D1 1.217 [0.984, 1.479], C2/D2 0.954 [0.799, 1.067]
    N/D: mean over seeds 1.070 [0.933, 1.351]; per seed N0/D0 1.005 [0.829, 1.288], N1/D1 1.206 [0.968, 1.684], N2/D2 0.999 [0.833, 1.271]
  in place: n 52 (33 episodes)
    median cm: D0 1.059 [0.770, 2.724] | D1 0.876 [0.549, 2.822] | D2 1.214 [0.749, 2.700] | N0 1.093 [0.764, 2.896] | N1 1.099 [0.915, 2.847] | N2 1.508 [0.831, 2.715] | C0 1.065 [0.726, 2.373] | C1 1.112 [0.737, 2.719] | C2 0.904 [0.745, 2.183]
    p90 cm:    D0 29.88 | D1 29.33 | D2 28.90 | N0 28.60 | N1 29.02 | N2 28.39 | C0 28.40 | C1 29.11 | C2 29.28
    attitude deg median: D0 1.73 | D1 1.62 | D2 1.67 | N0 1.74 | N1 2.14 | N2 1.70 | C0 2.33 | C1 1.64 | C2 1.88
    N/C: mean over seeds 1.228 [0.920, 1.369]; per seed N0/C0 1.026 [0.722, 1.329], N1/C1 0.988 [0.774, 1.315], N2/C2 1.669 [0.904, 1.923]
    C/D: mean over seeds 1.007 [0.855, 1.523]; per seed C0/D0 1.006 [0.782, 1.591], C1/D1 1.270 [0.963, 1.936], C2/D2 0.744 [0.733, 1.294]
    N/D: mean over seeds 1.177 [0.913, 1.617]; per seed N0/D0 1.032 [0.717, 1.547], N1/D1 1.256 [0.988, 1.779], N2/D2 1.242 [0.787, 1.922]
  reproduction: D0 median 1.96050 vs train3b lml_real.json 1.96050 (+0.00001); n 226 vs 226
  reproduction: D1 median 1.85640 vs train3b lml_real.json 1.85640 (+0.00000); n 226 vs 226
  reproduction: D2 median 2.22039 vs train3b lml_real.json 2.22039 (+0.00000); n 226 vs 226
```

- SONIC（`lml/run/sonic/eval.txt`，md5 4969186d），原文：

```
== lmlreal sonic (PRIMARY): 129 kept segments (not fallen, settled end) of 295, 90 episodes; reach_no 1 70, 2 39, 0 20; walked 97
  all (via points included): n 129 (90 episodes)
    median cm: D0 1.841 [1.680, 2.104] | D1 1.833 [1.657, 1.988] | D2 1.853 [1.794, 2.094] | N0 2.146 [1.849, 2.427] | N1 1.994 [1.778, 2.229] | N2 2.078 [1.740, 2.318] | C0 1.962 [1.754, 2.139] | C1 2.176 [1.873, 2.400] | C2 2.067 [1.781, 2.315]
    p90 cm:    D0 4.66 | D1 4.24 | D2 4.92 | N0 5.02 | N1 4.82 | N2 5.30 | C0 4.86 | C1 4.92 | C2 5.54
    attitude deg median: D0 4.90 | D1 4.32 | D2 4.61 | N0 5.01 | N1 5.01 | N2 4.64 | C0 4.51 | C1 5.01 | C2 4.56
    N/C: mean over seeds 1.005 [0.928, 1.100]; per seed N0/C0 1.094 [0.953, 1.222], N1/C1 0.916 [0.837, 1.057], N2/C2 1.005 [0.873, 1.147]
    C/D: mean over seeds 1.123 [1.021, 1.171]; per seed C0/D0 1.065 [0.947, 1.146], C1/D1 1.187 [1.036, 1.307], C2/D2 1.115 [0.964, 1.187]
    N/D: mean over seeds 1.125 [1.021, 1.184]; per seed N0/D0 1.166 [0.978, 1.289], N1/D1 1.087 [0.998, 1.183], N2/D2 1.121 [0.961, 1.186]
  第一下 (reach_no 1): n 70 (70 episodes)
    median cm: D0 1.994 [1.571, 2.203] | D1 1.866 [1.572, 2.324] | D2 1.900 [1.687, 2.214] | N0 2.213 [1.827, 2.524] | N1 2.090 [1.851, 2.474] | N2 2.064 [1.676, 2.539] | C0 2.175 [1.781, 2.457] | C1 2.325 [1.803, 2.730] | C2 2.133 [1.802, 2.814]
    p90 cm:    D0 4.56 | D1 4.16 | D2 4.91 | N0 5.01 | N1 4.80 | N2 5.31 | C0 4.86 | C1 4.91 | C2 5.36
    attitude deg median: D0 4.74 | D1 4.30 | D2 4.55 | N0 4.67 | N1 4.76 | N2 4.44 | C0 4.30 | C1 5.45 | C2 4.65
    N/C: mean over seeds 0.961 [0.848, 1.080]; per seed N0/C0 1.017 [0.864, 1.157], N1/C1 0.899 [0.776, 1.116], N2/C2 0.968 [0.753, 1.143]
    C/D: mean over seeds 1.153 [1.053, 1.283]; per seed C0/D0 1.091 [0.992, 1.303], C1/D1 1.246 [1.010, 1.423], C2/D2 1.123 [0.975, 1.387]
    N/D: mean over seeds 1.105 [1.008, 1.227]; per seed N0/D0 1.110 [0.957, 1.361], N1/D1 1.120 [0.949, 1.271], N2/D2 1.087 [0.929, 1.237]
  第二下 (reach_no 2): n 39 (39 episodes)
    median cm: D0 1.979 [1.484, 2.223] | D1 1.746 [1.494, 2.342] | D2 1.843 [1.547, 2.342] | N0 1.894 [1.590, 2.475] | N1 1.746 [1.470, 2.194] | N2 1.770 [1.571, 2.250] | C0 1.811 [1.455, 2.084] | C1 1.998 [1.534, 2.422] | C2 1.833 [1.540, 2.356]
    p90 cm:    D0 3.48 | D1 3.82 | D2 3.57 | N0 3.82 | N1 3.81 | N2 3.94 | C0 3.39 | C1 3.42 | C2 3.73
    attitude deg median: D0 5.06 | D1 4.28 | D2 4.92 | N0 5.16 | N1 5.01 | N2 4.64 | C0 4.25 | C1 4.55 | C2 4.45
    N/C: mean over seeds 0.962 [0.863, 1.123]; per seed N0/C0 1.046 [0.878, 1.361], N1/C1 0.874 [0.669, 1.039], N2/C2 0.966 [0.809, 1.178]
    C/D: mean over seeds 1.018 [0.921, 1.145]; per seed C0/D0 0.915 [0.810, 1.075], C1/D1 1.144 [0.960, 1.317], C2/D2 0.995 [0.856, 1.203]
    N/D: mean over seeds 0.973 [0.880, 1.144]; per seed N0/D0 0.957 [0.858, 1.280], N1/D1 1.000 [0.773, 1.172], N2/D2 0.961 [0.854, 1.173]
  walked: n 97 (69 episodes)
    median cm: D0 1.896 [1.685, 2.204] | D1 1.960 [1.721, 2.213] | D2 1.884 [1.800, 2.288] | N0 2.321 [1.880, 2.505] | N1 2.194 [1.970, 2.510] | N2 2.220 [1.852, 2.665] | C0 2.031 [1.771, 2.403] | C1 2.272 [1.815, 2.546] | C2 2.128 [1.823, 2.395]
    p90 cm:    D0 5.51 | D1 5.83 | D2 5.53 | N0 6.65 | N1 6.06 | N2 5.94 | C0 5.64 | C1 5.97 | C2 6.46
    attitude deg median: D0 5.52 | D1 4.84 | D2 5.43 | N0 5.36 | N1 5.49 | N2 4.90 | C0 5.18 | C1 5.66 | C2 5.15
    N/C: mean over seeds 1.051 [0.957, 1.169]; per seed N0/C0 1.143 [0.934, 1.267], N1/C1 0.966 [0.881, 1.155], N2/C2 1.043 [0.916, 1.244]
    C/D: mean over seeds 1.120 [0.998, 1.178]; per seed C0/D0 1.071 [0.962, 1.175], C1/D1 1.159 [0.972, 1.284], C2/D2 1.129 [0.940, 1.226]
    N/D: mean over seeds 1.174 [1.049, 1.253]; per seed N0/D0 1.225 [0.996, 1.351], N1/D1 1.119 [1.005, 1.251], N2/D2 1.178 [1.000, 1.293]
  in place: n 32 (21 episodes)
    median cm: D0 1.729 [1.348, 2.129] | D1 1.516 [1.109, 1.878] | D2 1.757 [1.348, 2.142] | N0 1.833 [1.465, 2.259] | N1 1.628 [1.325, 1.881] | N2 1.680 [1.302, 2.097] | C0 1.707 [1.368, 2.142] | C1 1.989 [1.630, 2.603] | C2 1.808 [1.591, 2.386]
    p90 cm:    D0 3.00 | D1 2.60 | D2 2.78 | N0 3.26 | N1 2.64 | N2 2.76 | C0 2.85 | C1 2.88 | C2 3.33
    attitude deg median: D0 3.70 | D1 3.87 | D2 3.38 | N0 3.78 | N1 3.71 | N2 3.76 | C0 3.24 | C1 3.81 | C2 3.80
    N/C: mean over seeds 0.941 [0.824, 1.011]; per seed N0/C0 1.074 [0.884, 1.251], N1/C1 0.819 [0.662, 0.948], N2/C2 0.930 [0.725, 1.065]
    C/D: mean over seeds 1.109 [1.017, 1.277]; per seed C0/D0 0.987 [0.817, 1.221], C1/D1 1.312 [1.129, 1.583], C2/D2 1.029 [0.897, 1.312]
    N/D: mean over seeds 1.030 [0.931, 1.128]; per seed N0/D0 1.060 [0.838, 1.315], N1/D1 1.074 [0.896, 1.195], N2/D2 0.956 [0.804, 1.158]
  reproduction: D0 median 1.84118 vs train3b lml_real.json 1.84117 (+0.00000); n 129 vs 129
  reproduction: D1 median 1.83337 vs train3b lml_real.json 1.83337 (+0.00000); n 129 vs 129
  reproduction: D2 median 1.85337 vs train3b lml_real.json 1.85337 (-0.00000); n 129 vs 129
```

- HEFT（`lml/run/heft/eval.txt`，md5 3b636f21），原文：

```
== lmlreal heft (PRIMARY): 69 kept segments (not fallen, settled end) of 166, 44 episodes; reach_no 1 27, 2 21, 0 21; walked 40
  all (via points included): n 69 (44 episodes)
    median cm: D0 24.310 [2.297, 26.934] | D1 23.654 [2.380, 26.137] | D2 24.721 [2.719, 26.994] | N0 24.470 [2.374, 26.500] | N1 24.549 [2.063, 26.730] | N2 24.819 [2.372, 27.404] | C0 22.293 [2.540, 25.874] | C1 23.656 [1.959, 26.649] | C2 23.178 [2.466, 26.676]
    p90 cm:    D0 36.76 | D1 36.78 | D2 37.10 | N0 37.79 | N1 38.32 | N2 38.96 | C0 36.58 | C1 35.16 | C2 35.87
    attitude deg median: D0 44.89 | D1 44.38 | D2 44.72 | N0 43.59 | N1 43.31 | N2 42.72 | C0 42.74 | C1 43.49 | C2 45.20
    N/C: mean over seeds 1.069 [0.748, 1.098]; per seed N0/C0 1.098 [0.735, 1.132], N1/C1 1.038 [0.769, 1.074], N2/C2 1.071 [0.754, 1.113]
    C/D: mean over seeds 0.952 [0.913, 1.197]; per seed C0/D0 0.917 [0.881, 1.171], C1/D1 1.000 [0.824, 1.255], C2/D2 0.938 [0.886, 1.145]
    N/D: mean over seeds 1.016 [0.896, 1.036]; per seed N0/D0 1.007 [0.838, 1.039], N1/D1 1.038 [0.876, 1.049], N2/D2 1.004 [0.833, 1.034]
  第一下 (reach_no 1): n 27 (27 episodes)
    median cm: D0 28.987 [1.887, 34.397] | D1 29.068 [1.499, 34.492] | D2 29.957 [1.702, 36.142] | N0 30.235 [1.772, 36.785] | N1 30.386 [1.682, 36.870] | N2 30.248 [1.524, 36.235] | C0 28.753 [1.621, 34.776] | C1 28.373 [1.895, 33.387] | C2 28.752 [2.195, 34.062]
    p90 cm:    D0 37.95 | D1 37.56 | D2 38.04 | N0 37.97 | N1 38.37 | N2 39.15 | C0 37.27 | C1 37.12 | C2 37.13
    attitude deg median: D0 45.83 | D1 38.66 | D2 39.83 | N0 38.13 | N1 43.71 | N2 28.93 | C0 41.63 | C1 40.15 | C2 46.70
    N/C: mean over seeds 1.058 [0.830, 1.396]; per seed N0/C0 1.052 [0.884, 1.502], N1/C1 1.071 [0.851, 1.323], N2/C2 1.052 [0.674, 1.381]
    C/D: mean over seeds 0.976 [0.762, 1.256]; per seed C0/D0 0.992 [0.722, 1.239], C1/D1 0.976 [0.842, 1.321], C2/D2 0.960 [0.721, 1.238]
    N/D: mean over seeds 1.033 [0.925, 1.120]; per seed N0/D0 1.043 [0.895, 1.170], N1/D1 1.045 [1.011, 1.190], N2/D2 1.010 [0.758, 1.022]
  第二下 (reach_no 2): n 21 (21 episodes)
    median cm: D0 2.228 [1.902, 2.335] | D1 2.233 [1.859, 2.392] | D2 2.516 [1.893, 2.729] | N0 2.277 [1.855, 2.387] | N1 1.964 [1.628, 2.095] | N2 2.104 [1.905, 2.383] | C0 2.486 [2.298, 2.608] | C1 1.832 [1.748, 2.015] | C2 2.313 [2.161, 2.475]
    p90 cm:    D0 29.40 | D1 31.46 | D2 35.76 | N0 35.43 | N1 38.12 | N2 36.69 | C0 27.68 | C1 30.02 | C2 30.42
    attitude deg median: D0 1.26 | D1 1.44 | D2 1.81 | N0 2.06 | N1 2.42 | N2 1.87 | C0 1.50 | C1 1.09 | C2 2.06
    N/C: mean over seeds 0.966 [0.806, 1.009]; per seed N0/C0 0.916 [0.769, 0.966], N1/C1 1.072 [0.867, 1.140], N2/C2 0.910 [0.745, 0.987]
    C/D: mean over seeds 0.952 [0.926, 1.153]; per seed C0/D0 1.116 [1.062, 1.223], C1/D1 0.820 [0.770, 1.075], C2/D2 0.919 [0.863, 1.154]
    N/D: mean over seeds 0.912 [0.885, 0.976]; per seed N0/D0 1.022 [0.917, 1.058], N1/D1 0.879 [0.811, 0.920], N2/D2 0.836 [0.771, 1.073]
  walked: n 40 (26 episodes)
    median cm: D0 28.792 [26.839, 30.376] | D1 28.318 [25.983, 31.457] | D2 28.620 [26.943, 35.760] | N0 27.982 [25.997, 35.429] | N1 27.886 [26.499, 36.870] | N2 28.314 [27.117, 36.235] | C0 27.605 [25.623, 29.548] | C1 28.151 [26.637, 30.020] | C2 27.575 [26.439, 30.417]
    p90 cm:    D0 46.15 | D1 47.92 | D2 43.11 | N0 44.30 | N1 49.70 | N2 46.87 | C0 40.03 | C1 38.88 | C2 38.86
    attitude deg median: D0 71.76 | D1 71.92 | D2 72.79 | N0 70.74 | N1 71.34 | N2 69.89 | C0 70.53 | C1 72.07 | C2 71.57
    N/C: mean over seeds 1.010 [0.960, 1.255]; per seed N0/C0 1.014 [0.958, 1.259], N1/C1 0.991 [0.919, 1.249], N2/C2 1.027 [0.972, 1.255]
    C/D: mean over seeds 0.972 [0.912, 1.001]; per seed C0/D0 0.959 [0.920, 0.979], C1/D1 0.994 [0.954, 1.026], C2/D2 0.963 [0.799, 1.018]
    N/D: mean over seeds 0.982 [0.942, 1.146]; per seed N0/D0 0.972 [0.914, 1.200], N1/D1 0.985 [0.906, 1.240], N2/D2 0.989 [0.978, 1.031]
  in place: n 29 (18 episodes)
    median cm: D0 1.887 [1.728, 2.193] | D1 1.633 [1.439, 2.006] | D2 1.820 [1.673, 2.123] | N0 1.772 [1.605, 2.167] | N1 1.628 [1.467, 1.943] | N2 1.574 [1.327, 2.008] | C0 2.215 [2.010, 2.326] | C1 1.783 [1.610, 1.833] | C2 2.161 [1.932, 2.287]
    p90 cm:    D0 3.75 | D1 4.06 | D2 4.23 | N0 3.98 | N1 4.09 | N2 3.78 | C0 4.42 | C1 4.45 | C2 4.46
    attitude deg median: D0 1.30 | D1 1.28 | D2 1.38 | N0 1.54 | N1 1.61 | N2 1.58 | C0 1.47 | C1 0.91 | C2 1.74
    N/C: mean over seeds 0.814 [0.768, 0.941]; per seed N0/C0 0.800 [0.733, 0.960], N1/C1 0.913 [0.864, 1.060], N2/C2 0.728 [0.658, 0.892]
    C/D: mean over seeds 1.151 [1.011, 1.200]; per seed C0/D0 1.174 [0.965, 1.275], C1/D1 1.091 [0.902, 1.220], C2/D2 1.188 [1.057, 1.258]
    N/D: mean over seeds 0.934 [0.891, 1.000]; per seed N0/D0 0.939 [0.884, 1.018], N1/D1 0.997 [0.895, 1.131], N2/D2 0.865 [0.750, 0.992]
  reproduction: D0 median 24.30997 vs train3b lml_real.json 24.30997 (+0.00000); n 69 vs 69
  reproduction: D1 median 23.65374 vs train3b lml_real.json 23.65374 (+0.00000); n 69 vs 69
  reproduction: D2 median 24.72060 vs train3b lml_real.json 24.72060 (+0.00000); n 69 vs 69
```

- 三份都打出了 `REPRODUCTION PASS`：D0–D2 的中位数和 train3b 的 `lml_real.json` 相差不超过 0.00001 cm，段数相同。

**次要评估：各控制器 data3 的留出集**（val + test_iid，没摔、停稳；按重置重抽 2000 次）。从 eval.txt 里 `grep -E "^== data3|^  (first|chained|all)|N/C:|C/D:"` 取出的行，原文：

```
[scalebfm]
== data3 scalebfm held out (val + test_iid, kept settled): 6160 segments
  first, in place: n 2154 (2154 resets)
    N/C: mean over seeds 0.964 [0.935, 0.989]; per seed N0/C0 0.956 [0.913, 0.994], N1/C1 0.975 [0.936, 1.015], N2/C2 0.960 [0.917, 0.995]
    C/D: mean over seeds 1.135 [1.106, 1.170]; per seed C0/D0 1.142 [1.098, 1.191], C1/D1 1.148 [1.102, 1.196], C2/D2 1.114 [1.076, 1.168]
  first, walk first: n 1016 (1016 resets)
    N/C: mean over seeds 1.191 [1.148, 1.249]; per seed N0/C0 1.138 [1.083, 1.205], N1/C1 1.221 [1.161, 1.294], N2/C2 1.215 [1.161, 1.290]
    C/D: mean over seeds 1.179 [1.131, 1.227]; per seed C0/D0 1.204 [1.147, 1.282], C1/D1 1.143 [1.082, 1.212], C2/D2 1.191 [1.117, 1.244]
  chained, in place: n 2006 (2006 resets)
    N/C: mean over seeds 0.990 [0.964, 1.013]; per seed N0/C0 0.989 [0.951, 1.023], N1/C1 0.978 [0.950, 1.009], N2/C2 1.003 [0.966, 1.037]
    C/D: mean over seeds 1.084 [1.063, 1.114]; per seed C0/D0 1.071 [1.036, 1.113], C1/D1 1.100 [1.070, 1.145], C2/D2 1.080 [1.050, 1.119]
  chained, after a walk: n 984 (984 resets)
    N/C: mean over seeds 1.012 [0.983, 1.053]; per seed N0/C0 1.003 [0.959, 1.073], N1/C1 1.000 [0.956, 1.052], N2/C2 1.033 [0.983, 1.086]
    C/D: mean over seeds 1.076 [1.042, 1.108]; per seed C0/D0 1.089 [1.023, 1.152], C1/D1 1.073 [1.019, 1.103], C2/D2 1.067 [1.026, 1.124]
  all: n 6160 (4505 resets)
    N/C: mean over seeds 1.027 [1.008, 1.046]; per seed N0/C0 1.022 [0.997, 1.045], N1/C1 1.033 [1.010, 1.059], N2/C2 1.025 [1.000, 1.049]
    C/D: mean over seeds 1.111 [1.094, 1.128]; per seed C0/D0 1.116 [1.094, 1.144], C1/D1 1.101 [1.075, 1.125], C2/D2 1.118 [1.090, 1.142]
[sonic]
== data3 sonic held out (val + test_iid, kept settled): 5940 segments
  first, in place: n 2115 (2115 resets)
    N/C: mean over seeds 0.976 [0.954, 0.999]; per seed N0/C0 0.985 [0.955, 1.018], N1/C1 0.970 [0.940, 1.003], N2/C2 0.973 [0.941, 1.001]
    C/D: mean over seeds 1.097 [1.076, 1.116]; per seed C0/D0 1.088 [1.059, 1.118], C1/D1 1.103 [1.071, 1.127], C2/D2 1.100 [1.068, 1.132]
  first, walk first: n 1015 (1015 resets)
    N/C: mean over seeds 1.016 [0.976, 1.056]; per seed N0/C0 1.017 [0.962, 1.065], N1/C1 1.009 [0.956, 1.073], N2/C2 1.021 [0.973, 1.073]
    C/D: mean over seeds 1.078 [1.043, 1.111]; per seed C0/D0 1.056 [1.014, 1.104], C1/D1 1.065 [1.007, 1.111], C2/D2 1.115 [1.062, 1.159]
  chained, in place: n 1863 (1863 resets)
    N/C: mean over seeds 0.970 [0.949, 0.993]; per seed N0/C0 0.966 [0.934, 1.003], N1/C1 0.975 [0.941, 1.006], N2/C2 0.967 [0.944, 1.002]
    C/D: mean over seeds 1.094 [1.068, 1.119]; per seed C0/D0 1.105 [1.067, 1.140], C1/D1 1.087 [1.054, 1.131], C2/D2 1.089 [1.051, 1.120]
  chained, after a walk: n 947 (947 resets)
    N/C: mean over seeds 0.967 [0.934, 1.011]; per seed N0/C0 0.956 [0.912, 1.006], N1/C1 0.979 [0.931, 1.036], N2/C2 0.966 [0.927, 1.029]
    C/D: mean over seeds 1.098 [1.057, 1.134]; per seed C0/D0 1.084 [1.031, 1.151], C1/D1 1.076 [1.023, 1.122], C2/D2 1.133 [1.073, 1.184]
  all: n 5940 (4371 resets)
    N/C: mean over seeds 0.976 [0.962, 0.992]; per seed N0/C0 0.983 [0.959, 1.004], N1/C1 0.971 [0.955, 0.994], N2/C2 0.975 [0.957, 0.993]
    C/D: mean over seeds 1.100 [1.086, 1.115]; per seed C0/D0 1.096 [1.074, 1.121], C1/D1 1.097 [1.072, 1.117], C2/D2 1.108 [1.089, 1.127]
[heft]
== data3 heft held out (val + test_iid, kept settled): 3068 segments
  first, in place: n 1423 (1423 resets)
    N/C: mean over seeds 0.954 [0.918, 0.989]; per seed N0/C0 0.942 [0.885, 0.983], N1/C1 0.976 [0.920, 1.022], N2/C2 0.943 [0.897, 1.008]
    C/D: mean over seeds 1.166 [1.111, 1.207]; per seed C0/D0 1.157 [1.092, 1.210], C1/D1 1.180 [1.083, 1.240], C2/D2 1.160 [1.098, 1.230]
  first, walk first: n 657 (657 resets)
    N/C: mean over seeds 1.177 [1.095, 1.284]; per seed N0/C0 1.193 [1.081, 1.295], N1/C1 1.112 [1.021, 1.244], N2/C2 1.226 [1.131, 1.394]
    C/D: mean over seeds 1.129 [1.057, 1.187]; per seed C0/D0 1.070 [0.988, 1.153], C1/D1 1.171 [1.068, 1.263], C2/D2 1.146 [1.048, 1.219]
  chained, in place: n 671 (671 resets)
    N/C: mean over seeds 0.975 [0.923, 1.019]; per seed N0/C0 0.964 [0.880, 1.020], N1/C1 0.964 [0.903, 1.040], N2/C2 0.997 [0.919, 1.057]
    C/D: mean over seeds 1.076 [1.029, 1.129]; per seed C0/D0 1.061 [0.997, 1.155], C1/D1 1.082 [1.005, 1.139], C2/D2 1.086 [1.021, 1.162]
  chained, after a walk: n 317 (317 resets)
    N/C: mean over seeds 0.993 [0.904, 1.067]; per seed N0/C0 0.919 [0.853, 1.041], N1/C1 1.008 [0.883, 1.112], N2/C2 1.050 [0.910, 1.129]
    C/D: mean over seeds 1.077 [1.003, 1.143]; per seed C0/D0 1.199 [1.029, 1.260], C1/D1 1.036 [0.951, 1.170], C2/D2 0.996 [0.926, 1.139]
  all: n 3068 (2462 resets)
    N/C: mean over seeds 1.023 [0.987, 1.056]; per seed N0/C0 1.040 [0.983, 1.083], N1/C1 1.004 [0.960, 1.048], N2/C2 1.024 [0.988, 1.076]
    C/D: mean over seeds 1.111 [1.089, 1.147]; per seed C0/D0 1.074 [1.041, 1.133], C1/D1 1.115 [1.080, 1.161], C2/D2 1.145 [1.104, 1.188]
```

**修订 3 的次要分析：真实局走过去后的第一下**（`nw2_walkfit.py`，冻结在 `asrun3/code/`，H20 CPU，三个控制器并行，各 4 线程）。运行命令见 `lml/run/walkfit_launch.sh`，它的第一行是 `# notebook: selfmodel_nowalk_20261126`。从 `lml/run/<控制器>/walkfit/walkfit.txt` 里取误差、比值、range (39)、nearer 这几行，`; nn median` 以后的部分略去：

```
[scalebfm]
scalebfm: lmlreal kept 226 segments; models ['D0', 'D1', 'D2', 'N0', 'N1', 'N2', 'C0', 'C1', 'C2']; eval /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/lml/run/scalebfm/eval
== 1. errors, real first reaches (reach_no 1; kept: not fallen, settled end)
  第一下, walked: n 85 (85 episodes)
    N/C mean over seeds 1.362 [1.085, 1.596]; per seed N0/C0 1.209 [1.007, 1.454], N1/C1 1.569 [1.111, 1.972], N2/C2 1.308 [1.048, 1.591]
    share N_k error > C_k error, mean over seeds 0.545 [0.463, 0.627]; per seed 0.529, 0.494, 0.612
    C/D mean over seeds 1.027 [0.948, 1.171]
  第一下, in place: n 32 (32 episodes)
    N/C mean over seeds 1.134 [0.987, 1.532]; per seed N0/C0 1.174 [0.947, 1.540], N1/C1 1.193 [0.774, 2.216], N2/C2 1.034 [0.807, 1.404]
    share N_k error > C_k error, mean over seeds 0.510 [0.427, 0.604]; per seed 0.656, 0.594, 0.281
    C/D mean over seeds 1.104 [0.817, 1.835]
  第一下 chain 0, walked: n 49 (49 episodes)
    N/C mean over seeds 1.276 [1.037, 1.742]; per seed N0/C0 1.174 [0.909, 1.475], N1/C1 1.271 [0.936, 1.796], N2/C2 1.384 [1.104, 2.148]
    share N_k error > C_k error, mean over seeds 0.612 [0.510, 0.707]; per seed 0.592, 0.469, 0.776
    C/D mean over seeds 0.895 [0.815, 1.025]
  第一下 chain 0, in place: n 32 (32 episodes)
    N/C mean over seeds 1.134 [0.983, 1.522]; per seed N0/C0 1.174 [0.943, 1.532], N1/C1 1.193 [0.774, 2.213], N2/C2 1.034 [0.811, 1.410]
    share N_k error > C_k error, mean over seeds 0.510 [0.417, 0.604]; per seed 0.656, 0.594, 0.281
    C/D mean over seeds 1.104 [0.834, 1.840]
== 2. starts: D's training split 847551 segments (data3, data3b, mid_data3, mid_data3b); reference sets all 847551, walk-first 283850, walk-first chain 0 114746, in place (N's set) 563701; neighbour space 273 x0 columns (204 dropped)
  reference all: 847551 segments, 28786 resets; neighbour p99 20.479
    第一下, walked (n 85, 85 episodes): joints 0.859 [0.788, 0.929]; pelvis 0.988 [0.965, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.847 [0.765, 0.918]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 32, 32 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference walk-first: 283850 segments, 9698 resets; neighbour p99 24.009
    第一下, walked (n 85, 85 episodes): joints 0.859 [0.776, 0.918]; pelvis 0.988 [0.965, 1.000]; hands 0.988 [0.965, 1.000]; range (39) 0.835 [0.753, 0.906]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 32, 32 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference walk-first chain 0: 114746 segments, 5771 resets; neighbour p99 18.318
    第一下, walked (n 85, 85 episodes): joints 0.859 [0.776, 0.918]; pelvis 0.988 [0.965, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.847 [0.765, 0.918]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 32, 32 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference in place (N's set): 563701 segments, 19088 resets; neighbour p99 20.907
    第一下, walked (n 85, 85 episodes): joints 0.859 [0.788, 0.929]; pelvis 0.988 [0.965, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.847 [0.776, 0.918]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 32, 32 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  nearer: nearest walk-first training segment nearer than the nearest in-place one; base rate 0.335 (walk-first 283850, in place 563701)
    第一下, walked: 0.082 [0.024, 0.141]
    第一下, in place: 0.031 [0.000, 0.125]
[sonic]
sonic: lmlreal kept 129 segments; models ['D0', 'D1', 'D2', 'N0', 'N1', 'N2', 'C0', 'C1', 'C2']; eval /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/lml/run/sonic/eval
== 1. errors, real first reaches (reach_no 1; kept: not fallen, settled end)
  第一下, walked: n 51 (51 episodes)
    N/C mean over seeds 1.016 [0.868, 1.242]; per seed N0/C0 1.039 [0.839, 1.213], N1/C1 0.980 [0.783, 1.363], N2/C2 1.028 [0.811, 1.367]
    share N_k error > C_k error, mean over seeds 0.523 [0.425, 0.621]; per seed 0.569, 0.490, 0.510
    C/D mean over seeds 1.127 [0.982, 1.275]
  第一下, in place: n 19 (19 episodes)
    N/C mean over seeds 0.863 [0.772, 1.009]; per seed N0/C0 1.066 [0.783, 1.389], N1/C1 0.704 [0.568, 0.897], N2/C2 0.820 [0.611, 1.052]
    share N_k error > C_k error, mean over seeds 0.316 [0.246, 0.386]; per seed 0.684, 0.105, 0.158
    C/D mean over seeds 1.262 [1.116, 1.417]
  第一下 chain 0, walked: n 31 (31 episodes)
    N/C mean over seeds 1.126 [0.928, 1.278]; per seed N0/C0 1.052 [0.801, 1.367], N1/C1 1.286 [0.884, 1.599], N2/C2 1.041 [0.789, 1.216]
    share N_k error > C_k error, mean over seeds 0.559 [0.430, 0.677]; per seed 0.613, 0.516, 0.548
    C/D mean over seeds 1.057 [0.937, 1.287]
  第一下 chain 0, in place: n 19 (19 episodes)
    N/C mean over seeds 0.863 [0.770, 1.014]; per seed N0/C0 1.066 [0.783, 1.389], N1/C1 0.704 [0.563, 0.897], N2/C2 0.820 [0.610, 1.051]
    share N_k error > C_k error, mean over seeds 0.316 [0.246, 0.386]; per seed 0.684, 0.105, 0.158
    C/D mean over seeds 1.262 [1.121, 1.422]
== 2. starts: D's training split 1079768 segments (data3, data3b, mid_data3, mid_data3b); reference sets all 1079768, walk-first 364173, walk-first chain 0 142562, in place (N's set) 715595; neighbour space 273 x0 columns (204 dropped)
  reference all: 1079768 segments, 28492 resets; neighbour p99 23.372
    第一下, walked (n 51, 51 episodes): joints 0.941 [0.863, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.941 [0.863, 1.000]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 19, 19 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference walk-first: 364173 segments, 9624 resets; neighbour p99 25.419
    第一下, walked (n 51, 51 episodes): joints 0.941 [0.863, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.941 [0.863, 1.000]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 19, 19 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference walk-first chain 0: 142562 segments, 5719 resets; neighbour p99 17.202
    第一下, walked (n 51, 51 episodes): joints 0.882 [0.784, 0.961]; pelvis 1.000 [1.000, 1.000]; hands 0.980 [0.941, 1.000]; range (39) 0.863 [0.764, 0.941]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 19, 19 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference in place (N's set): 715595 segments, 18868 resets; neighbour p99 25.015
    第一下, walked (n 51, 51 episodes): joints 0.941 [0.863, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.941 [0.882, 1.000]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 19, 19 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  nearer: nearest walk-first training segment nearer than the nearest in-place one; base rate 0.337 (walk-first 364173, in place 715595)
    第一下, walked: 0.725 [0.588, 0.843]
    第一下, in place: 0.000 [0.000, 0.000]
[heft]
heft: lmlreal kept 69 segments; models ['D0', 'D1', 'D2', 'N0', 'N1', 'N2', 'C0', 'C1', 'C2']; eval /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_nowalk_20261126/lml/run/heft/eval
== 1. errors, real first reaches (reach_no 1; kept: not fallen, settled end)
  第一下, walked: n 13 (13 episodes)
    N/C mean over seeds 1.030 [1.021, 1.110]; per seed N0/C0 1.011 [1.001, 1.082], N1/C1 1.061 [1.010, 1.138], N2/C2 1.018 [1.008, 1.126]
    share N_k error > C_k error, mean over seeds 0.949 [0.872, 1.000]; per seed 0.923, 0.923, 1.000
    C/D mean over seeds 0.986 [0.966, 0.991]
  第一下, in place: n 14 (14 episodes)
    N/C mean over seeds 0.882 [0.839, 0.970]; per seed N0/C0 1.074 [0.911, 1.280], N1/C1 0.933 [0.872, 0.994], N2/C2 0.640 [0.553, 0.770]
    share N_k error > C_k error, mean over seeds 0.405 [0.262, 0.571]; per seed 0.786, 0.286, 0.143
    C/D mean over seeds 1.076 [0.970, 1.154]
  第一下 chain 0, walked: no segments
  第一下 chain 0, in place: n 14 (14 episodes)
    N/C mean over seeds 0.882 [0.839, 0.971]; per seed N0/C0 1.074 [0.911, 1.303], N1/C1 0.933 [0.869, 0.994], N2/C2 0.640 [0.557, 0.777]
    share N_k error > C_k error, mean over seeds 0.405 [0.262, 0.571]; per seed 0.786, 0.286, 0.143
    C/D mean over seeds 1.076 [0.966, 1.154]
== 2. starts: D's training split 499077 segments (data3, data3b, mid_data3, mid_data3b); reference sets all 499077, walk-first 166848, walk-first chain 0 92249, in place (N's set) 332229; neighbour space 273 x0 columns (204 dropped)
  reference all: 499077 segments, 23815 resets; neighbour p99 24.574
    第一下, walked (n 13, 13 episodes): joints 0.385 [0.154, 0.615]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.385 [0.154, 0.692]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 14, 14 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference walk-first: 166848 segments, 7986 resets; neighbour p99 27.937
    第一下, walked (n 13, 13 episodes): joints 0.385 [0.154, 0.615]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.385 [0.077, 0.617]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 14, 14 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference walk-first chain 0: 92249 segments, 5779 resets; neighbour p99 23.542
    第一下, walked (n 13, 13 episodes): joints 0.385 [0.154, 0.615]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.385 [0.154, 0.615]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 14, 14 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  reference in place (N's set): 332229 segments, 15829 resets; neighbour p99 25.648
    第一下, walked (n 13, 13 episodes): joints 0.385 [0.154, 0.692]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 0.385 [0.154, 0.615]; neighbour 1.000 [1.000, 1.000]
    第一下, in place (n 14, 14 episodes): joints 1.000 [1.000, 1.000]; pelvis 1.000 [1.000, 1.000]; hands 1.000 [1.000, 1.000]; range (39) 1.000 [1.000, 1.000]; neighbour 1.000 [1.000, 1.000]
  nearer: nearest walk-first training segment nearer than the nearest in-place one; base rate 0.334 (walk-first 166848, in place 332229)
    第一下, walked: 0.615 [0.385, 0.846]
    第一下, in place: 0.000 [0.000, 0.000]
```

- walkfit.txt 的 md5：ScaleBFM 94b623bc，SONIC 4545a5e6，HEFT 4a3ec25d。walkfit.json：738cee1b，ec7c0619，feeced67。
- 关节里最常超出范围的（S 序号，按 SCALEBFM_ORDER）：ScaleBFM 是 26 号，占 11.8 %；HEFT 也是 26 号，占 61.5 %。26 号是 right_wrist_pitch_joint。SONIC 走过去的段只有 5.9 % 超出范围。

**HEFT 的三件事（协调人 11-26 在看到 eval.txt 以后加问的，事后分析，只读已有记录，没有跑仿真）**

问的是：
1. 走过去的第一下和原地的第一下误差差多少，走过去的起点有多少在训练范围里；
2. walked 那一格姿态误差约 71°，是不是量错了，比如模型的输入和标签用的骨盆朝向不对；
3. lmlcmp 复核里"HEFT 没碰家具、反解收敛的 27 下伸手：4.4 → 1.7 cm"（lmlcmp 第 449 行，论文 rss_status 第 74 行引用了它），这 27 下由哪些伸手组成，只看走过去后的第一下还成不成立。

- 脚本：`code/heft3.py`（md5 0b01057b）、`code/heft3_b27.py`（md5 d6a7298a）。H20 CPU，nice 10。
- 输出：`lml/run/heft_check/heft3.txt`（md5 95ba92c5）、`b27.txt`（md5 9c45d38c）。
- 在 heft3.txt 的 A 部分，每格依次是：
  - 三个模型的误差；
  - 命令那一行到标签那一行之间骨盆转了多少、移了多少，以及只因为骨盆移动、手在命令时坐标系里显得挪了多少（frame shift）；
  - 记录里不经过模型的 att_err（落点手的朝向和命令的夹角，世界系）和 jaw_err（落点离 IK 目标）；
  - 命令那一刻里程计减真值的差；
  - 伸手期间碰没碰家具，以及 ≥ 50 % 的行都碰着的比例；
  - 走完到位时离计划站位的误差。
- B 部分和 b27.txt 是 lmlcmp HEFT 用模型组的 135 次预抓取。

heft3.txt 原文：

```
== A. heft: lmlreal kept 69 segments (fell 0, settled end); walk = hg_walk (lr_run.py)
  first walked: n 13 | median err cm D0 36.15 N0 36.92 C0 36.53 | pelvis after the command (command row -> label row): |dyaw| median 5.3 p90 9.2 deg, dxy median 20.3 cm; hand moved by the frame change alone 21.1 cm | record att_err (landing vs command, world) 80.3 deg, jaw_err 38.5 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 1.8 cm | pre_v_pel median 0.8 cm/s, pre_settled 0.77 | furniture contact during the reach: any 1.00, >= 50 % of the rows 1.00 | arrive pos err median 25.6 cm
  first in place: n 14 | median err cm D0 1.79 N0 1.55 C0 1.45 | pelvis after the command (command row -> label row): |dyaw| median 0.9 p90 2.0 deg, dxy median 6.7 cm; hand moved by the frame change alone 6.6 cm | record att_err (landing vs command, world) 8.5 deg, jaw_err 4.5 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 0.0 cm | pre_v_pel median 7.8 cm/s, pre_settled 0.00 | furniture contact during the reach: any 0.07, >= 50 % of the rows 0.00 | arrive pos err median nan cm
  second walked: n 6 | median err cm D0 29.31 N0 30.71 C0 27.67 | pelvis after the command (command row -> label row): |dyaw| median 2.7 p90 4.6 deg, dxy median 5.0 cm; hand moved by the frame change alone 5.3 cm | record att_err (landing vs command, world) 82.4 deg, jaw_err 37.8 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 1.4 cm | pre_v_pel median 2.9 cm/s, pre_settled 0.33 | furniture contact during the reach: any 1.00, >= 50 % of the rows 0.83 | arrive pos err median 25.9 cm
  second in place: n 15 | median err cm D0 1.93 N0 1.89 C0 2.33 | pelvis after the command (command row -> label row): |dyaw| median 1.1 p90 1.4 deg, dxy median 1.3 cm; hand moved by the frame change alone 1.1 cm | record att_err (landing vs command, world) 8.3 deg, jaw_err 3.1 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 0.1 cm | pre_v_pel median 0.8 cm/s, pre_settled 0.80 | furniture contact during the reach: any 0.00, >= 50 % of the rows 0.00 | arrive pos err median nan cm
  via walked: n 21 | median err cm D0 26.25 N0 25.73 C0 25.20 | pelvis after the command (command row -> label row): |dyaw| median 1.7 p90 2.5 deg, dxy median 4.9 cm; hand moved by the frame change alone 4.1 cm | record att_err (landing vs command, world) 77.6 deg, jaw_err 31.3 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 2.0 cm | pre_v_pel median 0.4 cm/s, pre_settled 1.00 | furniture contact during the reach: any 1.00, >= 50 % of the rows 1.00 | arrive pos err median 25.6 cm
  via in place: n 0
  HEFT walked first reaches, one per line: task ep chain | err D0 N0 C0 cm | pelvis dyaw after the command deg, dxy cm, frame shift cm | att_err deg | odo-truth dyaw at command deg | arrive yaw err deg, pos err cm | walk asked (x, y, yaw deg) real | pre_v_pel | settle_s | furn rows
    t1 ep 3 c1 |  38.0  37.4  36.5 |   -9.8  12.6  18.6 | 100.7 |  +0.00 |   5.3  25.6 | (+1.91, +0.43,  +7.4) (+1.74, +0.24, +12.7) |  10.1 | 1.30 | 203
    t1 ep 5 c1 |  28.4  37.6  20.5 |   -4.0   6.6   9.4 |  22.4 |  +0.00 |   3.6  26.2 | (+1.83, +0.57, +12.0) (+1.67, +0.36, +15.6) |   1.4 | 0.84 | 150
    t1 ep11 c1 |  35.7  36.8  36.4 |   -6.2  23.8  25.4 |  82.8 |  +0.00 |   3.5  25.9 | (+1.83, +0.62, +13.3) (+1.68, +0.41, +16.8) |   1.5 | 1.10 | 185
    t1 ep15 c1 |  36.2  36.9  36.6 |   -6.6  20.3  21.7 |  84.1 |  +0.00 |   4.8  25.6 | (+1.88, +0.47,  +8.7) (+1.72, +0.27, +13.5) |   0.4 | 1.30 | 196
    t1 ep22 c1 |  36.2  36.9  36.6 |   -6.9  21.0  22.5 |  84.0 |  +0.00 |   3.4  25.9 | (+1.84, +0.57, +11.9) (+1.68, +0.37, +15.3) |   0.4 | 1.28 | 193
    t1 ep25 c1 |  29.6  30.2  29.0 |   -1.0   7.9   8.0 |  62.1 |  +0.00 |   5.5  25.4 | (+1.93, +0.46,  +8.1) (+1.77, +0.26, +13.6) |   0.3 | 0.92 | 161
    t1 ep28 c1 |  33.7  34.8  34.1 |   -3.6  20.3  21.1 |  80.3 |  +0.00 |   4.5  25.0 | (+1.86, +0.41,  +7.0) (+1.70, +0.22, +11.5) |   0.8 | 1.28 | 188
    t2 ep12 c1 |  73.1  76.7  75.0 |   +2.0   2.3   2.2 |  21.9 |  +0.00 |   3.0  26.2 | (+1.96, +0.06,  -3.5) (+1.73, -0.07,  -0.5) |  26.5 | 1.26 | 210
    t2 ep19 c1 |  56.7  58.7  57.7 |  -10.5   7.6   5.0 |  75.1 |  +0.00 |   6.5  25.5 | (+1.90, +0.27,  +2.6) (+1.71, +0.10,  +9.2) |   0.2 | 1.22 | 258
    t3 ep 0 c1 |  37.9  38.5  38.1 |   -5.6  19.7  20.9 |  85.3 |  +0.00 |   4.7  25.4 | (+1.87, +0.45,  +8.0) (+1.71, +0.25, +12.7) |   0.5 | 1.26 | 194
    t3 ep11 c1 |  34.4  34.7  34.8 |   -5.3  24.6  25.7 |  77.2 |  +0.00 |   6.5  26.0 | (+1.97, +0.23,  +1.5) (+1.77, +0.06,  +7.9) |   0.8 | 1.20 | 189
    t3 ep18 c1 |  30.4  30.8  30.7 |   -1.1  27.0  27.0 |  62.7 |  +0.00 |   3.6  25.5 | (+1.93, +0.13,  -1.6) (+1.72, -0.02,  +2.0) |   0.9 | 1.22 | 191
    t3 ep21 c1 |  36.5  36.8  36.7 |   -4.8  22.6  23.6 |  82.1 |  +0.00 |   5.3  25.5 | (+1.90, +0.50,  +9.4) (+1.74, +0.30, +14.7) |   0.2 | 1.24 | 193
  corr(err D0, frame shift) -0.618; corr(err D0, |dyaw|) 0.144; median |err D0 - frame shift| 14.5 cm
== A. scalebfm: lmlreal kept 226 segments (fell 0, settled end); walk = hg_walk (lr_run.py)
  first walked: n 85 | median err cm D0 1.42 N0 1.74 C0 1.44 | pelvis after the command (command row -> label row): |dyaw| median 10.8 p90 14.8 deg, dxy median 6.1 cm; hand moved by the frame change alone 12.1 cm | record att_err (landing vs command, world) 28.1 deg, jaw_err 11.6 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 2.6 cm | pre_v_pel median 0.5 cm/s, pre_settled 0.80 | furniture contact during the reach: any 0.59, >= 50 % of the rows 0.14 | arrive pos err median 3.2 cm
  first in place: n 32 | median err cm D0 0.79 N0 0.90 C0 0.77 | pelvis after the command (command row -> label row): |dyaw| median 8.3 p90 14.8 deg, dxy median 8.1 cm; hand moved by the frame change alone 11.7 cm | record att_err (landing vs command, world) 30.0 deg, jaw_err 6.8 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 0.0 cm | pre_v_pel median 1.8 cm/s, pre_settled 0.00 | furniture contact during the reach: any 0.44, >= 50 % of the rows 0.22 | arrive pos err median nan cm
  second walked: n 62 | median err cm D0 3.39 N0 3.06 C0 2.98 | pelvis after the command (command row -> label row): |dyaw| median 6.7 p90 11.0 deg, dxy median 1.8 cm; hand moved by the frame change alone 6.6 cm | record att_err (landing vs command, world) 27.4 deg, jaw_err 10.3 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 2.6 cm | pre_v_pel median 0.5 cm/s, pre_settled 0.95 | furniture contact during the reach: any 0.81, >= 50 % of the rows 0.53 | arrive pos err median 3.0 cm
  second in place: n 20 | median err cm D0 1.74 N0 1.36 C0 1.68 | pelvis after the command (command row -> label row): |dyaw| median 5.8 p90 11.5 deg, dxy median 1.7 cm; hand moved by the frame change alone 5.1 cm | record att_err (landing vs command, world) 15.0 deg, jaw_err 6.2 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 0.1 cm | pre_v_pel median 0.5 cm/s, pre_settled 0.95 | furniture contact during the reach: any 0.80, >= 50 % of the rows 0.50 | arrive pos err median nan cm
  via walked: n 27 | median err cm D0 30.35 N0 30.39 C0 30.59 | pelvis after the command (command row -> label row): |dyaw| median 13.9 p90 18.9 deg, dxy median 6.9 cm; hand moved by the frame change alone 5.5 cm | record att_err (landing vs command, world) 48.1 deg, jaw_err 25.7 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 2.7 cm | pre_v_pel median 0.4 cm/s, pre_settled 1.00 | furniture contact during the reach: any 1.00, >= 50 % of the rows 0.78 | arrive pos err median 2.8 cm
  via in place: n 0
== A. sonic: lmlreal kept 129 segments (fell 0, settled end); walk = hg_walk (lr_run.py)
  first walked: n 51 | median err cm D0 2.05 N0 2.50 C0 2.40 | pelvis after the command (command row -> label row): |dyaw| median 1.6 p90 4.8 deg, dxy median 6.3 cm; hand moved by the frame change alone 6.1 cm | record att_err (landing vs command, world) 12.5 deg, jaw_err 7.9 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 1.3 cm | pre_v_pel median 2.5 cm/s, pre_settled 0.22 | furniture contact during the reach: any 0.08, >= 50 % of the rows 0.04 | arrive pos err median 12.6 cm
  first in place: n 19 | median err cm D0 1.70 N0 1.77 C0 1.66 | pelvis after the command (command row -> label row): |dyaw| median 2.3 p90 9.2 deg, dxy median 7.6 cm; hand moved by the frame change alone 8.3 cm | record att_err (landing vs command, world) 10.2 deg, jaw_err 3.9 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 0.0 cm | pre_v_pel median 0.8 cm/s, pre_settled 0.00 | furniture contact during the reach: any 0.16, >= 50 % of the rows 0.00 | arrive pos err median nan cm
  second walked: n 26 | median err cm D0 1.97 N0 1.94 C0 1.78 | pelvis after the command (command row -> label row): |dyaw| median 1.1 p90 2.7 deg, dxy median 1.7 cm; hand moved by the frame change alone 1.7 cm | record att_err (landing vs command, world) 8.2 deg, jaw_err 4.6 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 0.9 cm | pre_v_pel median 0.8 cm/s, pre_settled 0.58 | furniture contact during the reach: any 0.00, >= 50 % of the rows 0.00 | arrive pos err median 13.9 cm
  second in place: n 13 | median err cm D0 2.01 N0 1.89 C0 1.87 | pelvis after the command (command row -> label row): |dyaw| median 0.6 p90 3.1 deg, dxy median 0.6 cm; hand moved by the frame change alone 0.9 cm | record att_err (landing vs command, world) 7.2 deg, jaw_err 2.7 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 0.1 cm | pre_v_pel median 0.6 cm/s, pre_settled 0.85 | furniture contact during the reach: any 0.15, >= 50 % of the rows 0.15 | arrive pos err median nan cm
  via walked: n 20 | median err cm D0 1.70 N0 1.92 C0 1.77 | pelvis after the command (command row -> label row): |dyaw| median 3.5 p90 7.0 deg, dxy median 3.4 cm; hand moved by the frame change alone 3.5 cm | record att_err (landing vs command, world) 15.1 deg, jaw_err 8.4 cm | at the command, odometry - truth: |dyaw| 0.00 deg, dxy 3.6 cm | pre_v_pel median 2.7 cm/s, pre_settled 0.20 | furniture contact during the reach: any 0.70, >= 50 % of the rows 0.20 | arrive pos err median 10.2 cm
  via in place: n 0
== B. lmlcmp HEFT selfmodel group, pre-grasps (reach_no 1), jittered eps: 135
  contact class {'stuck': 48, 'light': 36, 'none': 51}
  no contact 51: converged by "pred miss <= 1 cm" 27, not 24
  no contact 51: converged by "not (worse or oscillating)" 5, not 46
 -- converged = pred miss <= 1 cm
  the converged no-contact set: n 27, paired with the direct group 27: err_cm median S 2.60 / A 5.29, S < A 0.667 (18/27)
   walked / in place {'walked': 9, 'in place': 18} | chain (0 = no via point before) {0: 27} | task {0: 17, 4: 1, 5: 6, 6: 3}
   walked: n 9 err_cm median S 10.25 / A 9.32, S < A 1/9; per item (task ep chain S A): t5 ep5 c0 10.1 7.4; t5 ep8 c0 9.6 9.3; t5 ep17 c0 11.2 8.7; t5 ep25 c0 9.0 5.6; t5 ep26 c0 9.2 10.1; t5 ep28 c0 10.2 8.5; t6 ep0 c0 11.5 11.3; t6 ep10 c0 10.3 10.3; t6 ep16 c0 10.5 10.3
   in place: n 18 err_cm median S 1.88 / A 4.64, S < A 17/18; per item (task ep chain S A): t0 ep0 c0 1.8 4.8; t0 ep1 c0 1.7 3.7; t0 ep2 c0 2.7 4.8; t0 ep4 c0 2.6 4.5; t0 ep6 c0 1.8 4.5; t0 ep8 c0 1.8 4.5; t0 ep9 c0 2.0 4.9; t0 ep10 c0 2.0 3.3; t0 ep11 c0 1.4 3.1; t0 ep12 c0 1.5 4.4; t0 ep13 c0 4.1 33.4; t0 ep14 c0 2.5 5.3; t0 ep18 c0 1.6 3.6; t0 ep23 c0 1.0 29.8; t0 ep24 c0 2.7 4.8; t0 ep28 c0 1.7 3.7; t0 ep29 c0 2.4 5.3; t4 ep26 c0 78.9 53.2
 -- converged = not (worse or oscillating)
  the converged no-contact set: n 5, paired with the direct group 5: err_cm median S 2.70 / A 5.29, S < A 0.800 (4/5)
   walked / in place {'walked': 1, 'in place': 4} | chain (0 = no via point before) {0: 5} | task {0: 4, 5: 1}
   walked: n 1 err_cm median S 9.61 / A 9.32, S < A 0/1; per item (task ep chain S A): t5 ep8 c0 9.6 9.3
   in place: n 4 err_cm median S 2.59 / A 5.06, S < A 4/4; per item (task ep chain S A): t0 ep2 c0 2.7 4.8; t0 ep14 c0 2.5 5.3; t0 ep24 c0 2.7 4.8; t0 ep29 c0 2.4 5.3
```

b27.txt 原文：

```
d3 keys with method/ep: ['reset', 'reach_no', 'task_id', 'ep', 'method_id']
n 27
all 27 27
  err_cm at the window end, S / A: n 27 median 2.60 / 5.29; first < second 18/27 = 0.667
  live_err_cm at the label (vs own command target), S / A: n 27 median 4.17 / 4.32; first < second 11/27 = 0.407
  |act_h - goal_h|, S / A: n 27 median 1.67 / 4.41; first < second 25/27 = 0.926
  D predicted miss: chosen / direct command: n 27 median 0.16 / 3.86; first < second 27/27 = 1.000
  S err_cm / D predicted miss of the direct command: n 27 median 2.60 / 3.86; first < second 16/27 = 0.593
  D prediction error S / A: n 27 median 1.83 / 1.90; first < second 12/27 = 0.444
  close_dist_cm S / A: n 26 median 2.15 / 3.30; first < second 17/26 = 0.654
  S live_err / A live_err (label): n 27 median 4.17 / 5.29; first < second 18/27 = 0.667
walked 9
  err_cm at the window end, S / A: n 9 median 10.25 / 9.32; first < second 1/9 = 0.111
  live_err_cm at the label (vs own command target), S / A: n 9 median 3.52 / 2.95; first < second 2/9 = 0.222
  |act_h - goal_h|, S / A: n 9 median 1.13 / 2.95; first < second 8/9 = 0.889
  D predicted miss: chosen / direct command: n 9 median 0.14 / 3.99; first < second 9/9 = 1.000
  S err_cm / D predicted miss of the direct command: n 9 median 10.25 / 3.99; first < second 0/9 = 0.000
  D prediction error S / A: n 9 median 1.02 / 1.20; first < second 5/9 = 0.556
  close_dist_cm S / A: n 9 median 8.86 / 5.30; first < second 0/9 = 0.000
  S live_err / A live_err (label): n 9 median 3.52 / 9.32; first < second 8/9 = 0.889
in place 18
  err_cm at the window end, S / A: n 18 median 1.88 / 4.64; first < second 17/18 = 0.944
  live_err_cm at the label (vs own command target), S / A: n 18 median 4.44 / 4.44; first < second 9/18 = 0.500
  |act_h - goal_h|, S / A: n 18 median 1.84 / 4.59; first < second 17/18 = 0.944
  D predicted miss: chosen / direct command: n 18 median 0.17 / 3.69; first < second 18/18 = 1.000
  S err_cm / D predicted miss of the direct command: n 18 median 1.88 / 3.69; first < second 16/18 = 0.889
  D prediction error S / A: n 18 median 1.96 / 2.03; first < second 7/18 = 0.389
  close_dist_cm S / A: n 17 median 1.87 / 3.05; first < second 17/17 = 1.000
  S live_err / A live_err (label): n 18 median 4.44 / 4.64; first < second 10/18 = 0.556
A group contact on the same episodes (furn rows > 0): 2 / 27
odometry - truth at the pre-grasp command (|xy| cm, yaw deg), S group, the 27:
  walked: n 9 |xy| median 9.8 cm (min 8.2, max 10.6); |yaw| max 0.000 deg; gait gait_heft.HEFT
  in place: n 18 |xy| median 0.1 cm (min 0.0, max 0.1); |yaw| max 0.000 deg; gait None
```

**文件**（`lml/run/` 下）
- `run.log`、`logs/train_<控制器>_<组>_s<种子>.log`；
- `<控制器>/models_{N,C}_s{0,1,2}/`；
- `<控制器>/eval.txt`、`eval/eval.json`、`eval/per_segment.npz`，md5 见下表；
- `<控制器>/walkfit/walkfit.{txt,json}`；
- `heft_check/`。

| 控制器 | eval.json | per_segment.npz |
|---|---|---|
| ScaleBFM | 1f2158f1 | 58a87241 |
| SONIC | 381e8b3f | 5ff5de50 |
| HEFT | ce067333 | c9de8390 |


## 意外
<!-- 预测和结果哪里对不上？有什么没想到的现象？ -->

### 第一阶段

**预测和结果对照**（种子 0；括号里是种子 1）

| 项 | 预测 | 结果 | 落在区间里？ |
|---|---|---|---|
| 训练时间（修订 1 改写的） | 每个 20–45 分钟 | 610–901 s，即 10–15 分钟 | 否，更快：b0a 09:24Z 就走了 |
| 挑 checkpoint 的原地 val，N/C | 0.88–1.02 | 1.658 / 1.695 = 0.978（1.684 / 1.698 = 0.992） | 是 |
| **N/C，全部 235 段** | **1.00–1.20** | **1.161**（1.086） | 是 |
| N/C 的区间宽度 | 约 ±0.10–0.15 | [0.976, 1.520]，即 −0.19 / +0.36 | 否，宽得多，尤其是上侧 |
| N/C，第一下 | 1.00–1.30 | 1.294（1.300） | 是，种子 1 正好在上限 |
| N/C，第二下 | 0.95–1.15 | 1.107（0.964） | 是 |
| N/C，XMovePickTeleop | 0.95–1.35 | 1.707（1.201） | 种子 0 否 |
| N/C，XMoveBendPickTeleop | 0.95–1.35 | 1.521（1.613） | 否 |
| N/C，BendPickMP | 0.95–1.20 | 1.082（1.032） | 是 |
| N/C，TabletopGraspMP | 0.90–1.15 | 0.676（0.873） | 否，偏低 |
| C2/C | 0.95–1.08 | 1.046 | 是 |
| C/D | 1.00–1.25 | 1.238（1.333） | 种子 1 否 |
| 落在判据哪一行 | 第一行 50 %，中间 35 %，第三行 15 % | 中间 | — |
| data3 原地第一下 | 0.88–1.02 | 0.903（0.866） | 种子 1 略低 |
| data3 原地连续伸手 | 0.90–1.02 | 0.992（0.999） | 是 |
| data3 任务式走法后的第一下 | 1.05–1.40 | 1.460（1.457） | 否，偏高 |
| data3 任务式走法后的连续伸手 | 1.00–1.25 | 1.054（1.095） | 是 |
| data3 d3 走法后的第一下 | 1.40–3.00 | 2.174（2.162） | 是 |
| data3 d3 走法后的连续伸手 | 1.10–1.80 | 1.481（1.458） | 是 |
| data3b 任务式走法，预抓取加进入 | 1.00–1.30 | 1.545（1.411） | 否，偏高 |
| 补充标签 ended，和主标签相差 | ≤ 0.05 | 1.171 对 1.161，差 0.010（1.189 对 1.086，差 0.103） | 种子 1 否 |
| 种子 1 和种子 0 相差 | ≤ 0.08 | 0.075 | 是 |
| N 在自己范围外的段误差更高，第一下尤其如此 | 范围外 > 范围内 | 全部 1.06 对 1.01；第一下 0.54 对 1.17；第二下 1.32 对 0.93 | 第一下反了 |

**意外**
1. **N 输在第一下，而且输的不是"起点超出范围"。**
   - 第一下的 N/C 两个种子都是 1.29–1.30；第二下是 0.96–1.11。留出集里"任务式走法后的第一下"N/C 也是 1.46，比预测的上限 1.40 还高。
   - 但第一下的起点 84.5 % 落在 N 自己的训练范围里，而且 N 在范围外那 15.5 % 的段上误差反而更小（0.54 cm 对 1.17 cm）。
   - sd_range 的 39 个量只有关节角、骨盆高度和朝向、两个手点，没有速度。刚走完时关节还在动、骨盆还在晃，这些不在 39 个量里。所以更可能的情况是：N 输在没见过"刚停下、还在动"的状态，而不是没见过那些姿势。这和登记时写的理由一致，但 range 检查测不出来。
2. **差距集中在两个 XMove 任务上。** 种子 0 的 1.71 和 1.52 都超出了预测上限 1.35。BendPickMP 接近 1。TabletopGraspMP 上 N 反而更好（0.68 / 0.87），但这个任务里被桌沿挡住的段误差 28–40 cm，中位数在两种误差之间跳（它的区间上限是 29–31 cm），比值不稳。
3. **按任务的比值被 C 的种子噪声主导。**
   - XMovePickTeleop 上，N 两个种子是 0.924 / 0.930，几乎不变；C 却是 0.541 / 0.774。所以同一个任务的 N/C 从 1.71 掉到 1.20。
   - 两个 C 的 val 几乎一样（1.695 / 1.698）。真实局的中位数在种子之间的波动，比 val 大得多。
4. **数据量和先走后伸一样重要。**
   - C 只有 D 2/3 的数据，C/D 是 1.24 / 1.33；N/C 是 1.16 / 1.09。
   - 不考虑数据量，只问"有没有先走后伸的数据"，会把数据量的影响也算进去。N/D 是 1.44 / 1.45。
5. **区间比预测的宽得多。** 40 个布局里，有几个布局的误差在两种情况之间切换，比值的区间上侧拉得很长（上限 1.52）。
6. 运行上：H20 的容器中途重启过一次（修订 1）。之后改在 4090b 上用接管的方式跑两个训练（修订 1b），没有影响数值。

### 第二阶段

**预测和结果对照**

| 项 | 预测 | 结果 | 落在区间里？ |
|---|---|---|---|
| 训练时间（登记时写的 H20） | 每个 10–20 分钟 | 497–825 s，即 8–14 分钟 | 大体是，ScaleBFM 有 4 个略快 |
| **N/C，ScaleBFM**（三个种子平均） | **1.00–1.35** | **1.055** [0.954, 1.229] | 是 |
| **N/C，SONIC** | **0.95–1.35** | **1.005** [0.928, 1.100] | 是 |
| **N/C，HEFT** | **1.00–1.40** | **1.069** [0.748, 1.098] | 是 |
| 修订 3：走过去后的第一下，N/C | ScaleBFM 1.00–1.40，SONIC 0.95–1.40，HEFT 0.85–1.20 | 1.362，1.016，1.030 | 三个都是 |
| 修订 3：同上，"N 比 C 错得多"的比例 | 0.50–0.70，0.45–0.70，0.40–0.60 | 0.545，0.523，0.949 | HEFT 否 |
| 修订 3：range (39) 在"先走路"参照集里 | 0.65–0.85，0.70–0.90，0.35–0.65 | 0.835，0.941，0.385 | SONIC 否，偏高 |
| 修订 3：range (39) 在"原地"参照集里 | 0.70–0.90，0.70–0.90，0.45–0.70 | 0.847，0.941，0.385 | SONIC 否，偏高；HEFT 否，偏低 |
| 修订 3：nearer（基准约 0.33） | 0.40–0.80，0.40–0.80，0.15–0.45 | 0.082，0.725，0.615 | ScaleBFM 否，远低于基准；HEFT 否，偏高 |
| 修订 3：HEFT 的 \|N/C − 1\| 小于 ScaleBFM（把握 55 %） | 是 | 0.030 对 0.362，是 | 是，但原因不是预测里写的那个（见意外 3） |
| 修订 3 的方向：HEFT 的 nearer 不高于基准，ScaleBFM、SONIC 高于基准 | — | HEFT 0.615 高于基准；ScaleBFM 0.082 低于基准；SONIC 0.725 高于基准 | 只有 SONIC 对 |

**意外**
1. **ScaleBFM 总体过线，但第一下的差距和 SIMPLE 一样大。**
   - 第一下的 N/C 是 1.300 [1.045, 1.540]，三个种子 1.182 / 1.504 / 1.215。走过去后的第一下是 1.362 [1.085, 1.596]。留出集里"先走后的第一下"是 1.191 [1.148, 1.249]。
   - 第一阶段 SIMPLE 的第一下是 1.294 / 1.300。两个 benchmark、五个 N/C 配对，第一下都高约 30 %，区间下限都高于 1。
   - 总体能落在第一行，是因为第二下（0.938）和途经点那几下看不出差别，把第一下的差距冲淡了。
2. **ScaleBFM 走过去后的起点，离原地的训练段反而更近，N 却在这些段上更差。**
   - nearer 只有 0.082，基准是 0.335。预测是 0.40–0.80，差得很远。range (39) 对两个参照集一样（83.5 % 对 84.7 %）。
   - LML 的先走后伸只有 d3 走法（转身、走、再转），起点和任务局里走到站位后的样子不像。但 C 有了这部分数据，第一下还是更准。
   - 和第一阶段的意外 1 一样：差距用起点姿态在不在训练范围里、离训练段近不近，都解释不了。
3. **lmlreal 里 HEFT 走的是 hg_walk，不是 gait_heft。修订 3 的前提对这批局不成立。**
   - `lr_run.py` 走路调用的是 `hg_walk.walk_frames`。gait_heft 是从 lmlcmp 开始才用的（`lc_run.py` 的 `walk_ctrl`）。
   - 修订 3 写"LML 的任务局里 HEFT 走的是 gait_heft"，说的是 lmlcmp 以后的任务局，不是第二阶段评估用的 lmlreal。这一点在写修订 3 时没有核对。
   - 所以在 lmlreal 上，HEFT 训练数据的步态和任务局是同一个。nearer 0.615 高于基准，按修订 3 的读法，"对不上"在这批局上不成立。
4. **HEFT 走过去后的第一下全部卡在家具上，71° 是真的，不是量错。**（事后查的，见结果里的 heft3.txt）
   - 这 13 段全部在途经点之后（chain 1），全部有 ≥ 50 % 的时间手臂碰着家具。走过去的途经点 21/21、第二下 5/6 也是这样。原地的第一下只有 1/14 碰到过。
   - 不经过模型，直接看记录：落点手的朝向和命令差 80°，落点离 IK 目标 38.5 cm。原地分别是 8.5° 和 4.5 cm。
   - 模型的输入和标签都用命令那一刻骨盆的真值位姿（`d3_collect.segments` 的 p0、yaw0）。里程计的朝向和真值每一下都相同（差 0.00°）。命令之后骨盆只转了 5.3°（中位数）。所以没有坐标系问题。
   - 这几局走完离计划站位中位数差 25.6 cm：要它走约 1.9 m，实际走了约 1.7 m。苹果、碗、盐瓶三个桌子任务都是这样。
   - 所以 HEFT 的真实局 N/C 说明不了走路数据有没有用：所有模型错的都是"手被桌子挡住"，各模型都差约 36 cm。"N 比 C 错得多"的比例 0.949，是因为 N 在同样的大误差上一致地多错约 0.5 cm，和走路数据合不合适无关。
   - 协调人那条预测（HEFT 的 |N/C − 1| 更小）应验了，但原因是接触，不是步态。
5. **HEFT 自己的留出集里，N 在先走后的第一下上明显更差**：1.177 [1.095, 1.284]，三个种子都 > 1.11。这是 HEFT 唯一没被接触干扰的走路格。
6. **SONIC 哪一格都看不出差别**：真实局 1.005，第一下 0.961，走过去后的第一下 1.016，留出集里先走后的第一下也只有 1.016 [0.976, 1.056]。ScaleBFM 和 HEFT 的这一格是 1.19 和 1.18。SONIC 的先走后伸数据连在它自己的同类段上都没有帮助。
7. **C/D 比第一阶段小得多**：真实局里 ScaleBFM 1.064、SONIC 1.123、HEFT 0.952；留出集"all"一格是 1.10–1.11。第一阶段 SIMPLE 是 1.24 / 1.33。C/D 里还混着 4090 和 H20 的差别。
8. **HEFT 的区间不可靠**：点估计 1.069 几乎在区间 [0.748, 1.098] 的上端。69 段里一半左右误差约 2 cm，另一半约 25–30 cm，重抽时中位数在两种之间跳。按登记的规则它落在第一行，但这个区间的意义有限。
9. **ScaleBFM 主比值的区间上限 1.229，离 1.25 只差 0.021。** "离阈值不到 0.02 不下结论"那条规则是给中间方案定的，第二阶段登记时没有，这里不追溯使用，只记下来。
10. **lmlcmp 那 27 下，"4.4 → 1.7 cm"只在身体坐标里成立。**（事后查的，见 b27.txt）
    - 这 27 次全是第一下（预抓取），前面都没有途经点。其中 9 次走过去（哑铃 6、浇水壶 3），18 次原地（蜡烛 17、雕像柜 1）。
    - 4.41 → 1.67 cm、25/27 是这样算的：落点用命令那一刻骨盆的真值位姿表示，目标用里程计以为的位姿表示，两者相减。按落点离世界里真实目标算（`err_cm`），是 5.29 → 2.60 cm、18/27。
    - 只看走过去的 9 次：身体坐标 2.95 → 1.13 cm，8/9 更准；世界坐标 9.32 → 10.25 cm，只有 1/9 更准。原因是 gait_heft 走完后里程计的位置差 9.8 cm（8.2–10.6），模型看不到。
    - 原地的 18 次两种算法都成立：世界坐标 4.64 → 1.88 cm，17/18。
11. 运行上没有意外：18 个训练一次跑完，两张卡每张同时最多两个，没有中断。


## 结论与下一步
<!-- 每个论断引用上面的证据；写明证据强度（运行次数、种子、episode 数）；至少一个其他解释；
     四类陷阱检查：benchmark 选择 / 数据泄漏 / 指标误用 / 事后挑选；结果落在判据哪一侧、下一步做什么。 -->

### 第一阶段（SIMPLE，ScaleBFM）

**结论**
1. **落在判据的中间那一行**：对 SIMPLE 里的 ScaleBFM，去掉先走后伸的数据，真实局的误差变大 9–16 %。两个种子都落在中间那一行，所以既不能说"不需要"，也不能说"必须保留"。
   - 种子 0：N/C 1.161 [0.976, 1.520]。点估计在 1.10 和 1.25 之间。
   - 种子 1：N/C 1.086 [1.002, 1.276]。点估计 ≤ 1.10，但区间上限超过 1.25。
   - 证据见"结果"里两份 eval 的 PRIMARY 段。
2. **差距几乎全在第一下**：第一下的 N/C 是 1.294 [1.071, 1.555] 和 1.300 [1.157, 1.443]，两个种子的区间下限都高于 1；第二下是 1.107 和 0.964，看不出差别。
   - 留出集里也一样："任务式走法后的第一下"是 1.460 / 1.457，"原地第一下"是 0.903 / 0.866，N 在原地第一下上反而更好，因为它的原地数据更多。
3. **第一下输掉的，不是"姿势超出范围"**：
   - N 的第一下起点 84.5 % 落在 N 自己的训练范围里；
   - 范围外那些段，N 的误差反而更小（0.54 cm 对 1.17 cm）；
   - 更可能的原因是刚走完时的运动状态（关节速度、骨盆晃动），39 个范围量里没有这些（意外 1）。
4. **走路标志本身几乎没用**：C2/C 是 1.046 [0.955, 1.292]。N 的劣势主要来自数据，不来自把标志置 0（N/C2 1.110）。
5. **省下的采集时间不多**：
   - D 的训练集里，先走后伸的重置约占 1/3：29 260 个里有 9 737 个。
   - 按原地 7.5 s、先走后伸 12.0 s 算（simpledata 修订 1 的冒烟），全换成原地重置只省约 17 % 的时间。
   - 换来的是真实局误差高 9–16 %，第一下高约 30 %。

**证据强度**
- N、C 各两个种子，C2 和 D 各一个模型。
- 主评估 235 段，来自 120 局、40 个布局，区间按布局重抽 4000 次。
- 两个种子的总体 N/C 相差 0.075。

**其他可能的解释**
1. **C 的种子噪声**：
   - 两个 C 的 val 几乎一样（1.695 / 1.698），但在真实局三个任务上差了 24 %（0.652 / 0.806 cm）；
   - 种子 1 的点估计已经低于 1.10；
   - 再多几个种子，平均值可能落在 1.10 附近，在第一行和中间行之间摇摆。
   - 第一下的差距在两个种子里都很稳（1.29 / 1.30），这一条不受影响。
2. **XMove 任务的绝对误差很小**：D 只有 0.36–0.40 cm，N 和 C 差 0.2–0.4 cm 在比值上就是 1.5 倍以上。按 cm 算，差距没有比值看起来那么大。
3. **挑 checkpoint 只用原地的 val，这一点对 C 不利**：C 的 checkpoint 没有按先走后伸的段来挑。如果按那些段挑，C 可能更好，N/C 会更大。所以这里的 N/C 偏保守。

**四类陷阱**
- **benchmark 选择**：主评估就是登记的那 235 段、四个任务，TabletopGraspMP 也报了，没有去掉任何东西。
- **数据泄漏**：
  - 真实局的重置编号从 90 000 000 开始，不在任何训练文件里；
  - 留出集用 data.py 自己的桶；
  - 每组的归一化来自它自己的训练集；
  - checkpoint 按原地 val 挑，没碰真实局。
- **指标误用**：用的是登记的中位数比值。p90、姿态、ended 口径、任务标签都报了，方向一致：任务标签 ended 的 N/C 是 1.171 / 1.189。
- **事后挑选**：
  - 两个种子都报了；每个训练只有一个按 val 挑的 checkpoint；
  - H20 那次中断的运行没有产生任何结果；
  - 接管以后任务不变；
  - 看到结果以后没有重跑任何东西。

**对照判据**
- 主判据：中间那一行，结论是"试一个更省的中间方案，另行登记"。
- 附加规则 1：两个种子同在中间那一行，不改。
- 附加规则 2：种子 0 去掉 TabletopGraspMP 后的三个任务是 1.416 [1.103, 1.685]，落在 > 1.25 那一行；种子 1 是 1.209 [1.088, 1.316]，落在中间。按规则降为中间，结论本来就在中间。
- 附加规则 3：XMoveBendPickTeleop 两个种子都 > 1.25 且区间下限 > 1.0（1.521 [1.088, 2.004]；1.613 [1.217, 2.164]）；XMovePickTeleop 只有种子 0 满足（1.707 [1.527, 1.761]）。以后如果建议只采原地，要写明 XMove 这类任务不适用。
- 附加规则 5：C2/C 1.046，没有触发。

**下一步**
1. **另行登记中间方案**：只采短距离的任务式走法（走一小段、停下、再伸手），大小和 N、C 一样，看第一下的差距能不能补上。意外 1 说明要补的是"刚停下"的那个状态，不一定需要长距离走路。
2. **建议加一组时间对齐的比较**：
   - 用 D 的采集时间全采原地，大约是 N 原地重置数的 1.8 倍，看能不能追上 D；
   - 这回答的是"省下的时间拿去多采原地数据，值不值"；
   - 按 val 的学习曲线粗估（指数约 0.26），1.8 倍数据大约能把 N/D 从 1.44 降到 1.24，仍然不如 D。这只是估计，要跑了才算。
3. 第二阶段（LML，三个控制器）还在 H20 上跑，结论按控制器另写。

### 第二阶段（LocoManiLab；ScaleBFM、SONIC、HEFT）

**结论**
1. **按登记的判据，三个控制器都落在第一行，种子一致的规则也都满足**（证据见"结果"里三份 eval.txt 的 PRIMARY 段）：

   | 控制器 | N/C 平均 [95 %] | 每个种子 | 判据 |
   |---|---|---|---|
   | ScaleBFM | 1.055 [0.954, 1.229] | 1.041 / 1.059 / 1.064 | 第一行 |
   | SONIC | 1.005 [0.928, 1.100] | 1.094 / 0.916 / 1.005 | 第一行 |
   | HEFT | 1.069 [0.748, 1.098] | 1.098 / 1.038 / 1.071 | 第一行，但登记的 HEFT 说明适用，见第 4 条 |

   也就是说，在 LML 的真实局上把全部伸手放在一起看，去掉先走后伸的数据，误差的点估计最多大 7 %，三个区间的上限都低于 1.25。
2. **ScaleBFM 不能就此改成只采原地**：
   - 走路之后的第一下，两个 benchmark 都稳定地差约 30 %：
     - LML 第一下 1.300 [1.045, 1.540]，走过去后的第一下 1.362 [1.085, 1.596]；
     - 留出集里先走后的第一下 1.191 [1.148, 1.249]；
     - 第一阶段 SIMPLE 的第一下 1.294 / 1.300。
   - 总体能过线，是因为第二下和途经点那几下没有差别，把它冲淡了（意外 1）。
   - 判据看的是全部伸手，所以这一条不改判据的结论，但会改建议：对 ScaleBFM，只采原地要以走到站位后的第一下准度为代价，而第一下正是走到站位后要做的事。
3. **SONIC 看不出先走后伸的数据有任何用处**：真实局、第一下、走过去后的第一下、留出集里先走后的第一下，N/C 都在 0.96–1.02 之间（意外 6）。在这次检验的范围里，SONIC 只采原地就够。
4. **HEFT 的真实局回答不了这个问题**：
   - 走过去后的 13 个第一下全部卡在桌子上，所有模型都差约 36 cm。71° 的姿态误差是真的，不是量错，也没有坐标系问题（意外 4）。
   - 原地的段上，N 比 C 好（0.814）。
   - HEFT 自己的留出集里，N 在先走后的第一下上差 18 %（1.177 [1.095, 1.284]，意外 5）。这是唯一能看的走路格，它指向"先走后伸的数据对 HEFT 有用"。
   - 按登记时写好的 HEFT 说明，第一行的结果只能说明：去掉这部分数据，不会让一个本来就错得很远的预测变得更差。不能当成 HEFT 可以只采原地的证据。
5. **修订 3 的问题（HEFT 的走路数据和任务局对不对得上），在 lmlreal 上不成立**：
   - lmlreal 里 HEFT 走的就是 hg_walk，和训练数据一样（意外 3）；
   - 走过去的起点离先走后伸的训练段更近（nearer 0.615，基准 0.334）。
   - 对用 gait_heft 的任务局（lmlcmp 以后），这次没有检验。
   - 这一项只作描述，不进判据。
6. **lmlcmp 那 27 下的"4.4 → 1.7 cm"只在身体坐标里成立**：
   - 走过去的 9 次，按世界里的真实目标算是 9.32 → 10.25 cm，模型没有帮上忙，误差来自走完后里程计差约 10 cm；
   - 原地的 18 次，两种算法都成立：4.6 → 1.9 cm，17/18（意外 10）。
   - 协调人已转给论文那边。

**证据强度**
- 每个控制器 N、C 各 3 个种子，18 个模型都在 H20 上训，N 和 C 彼此配对。
- 真实局保留的段：ScaleBFM 226 段 / 129 局，SONIC 129 / 90，HEFT 69 / 44。第一下 117 / 70 / 27 段。
- HEFT 走过去后的第一下只有 13 段，而且全部卡住，基本没有信息量。
- D 是在 4090 上训的，所以 C/D、N/D 里混着换卡的差别；N/C 没有。

**其他可能的解释**
1. **ScaleBFM 第一下的差距是种子噪声**：三个种子分别是 1.182、1.504、1.215，都高于 1.1。加上第一阶段的两个种子，五个配对没有一个低于 1.18，这个解释站不住。
2. **ScaleBFM 第一下的差距来自 C 多了 d3 走法的"刚停下"状态，而不是姿态覆盖**：nearer 和 range 都显示，走过去后的起点在原地数据里也找得到相近的段（意外 2）。这和第一阶段意外 1 的推测一致：39 个范围量和 x0 的近邻都测不出运动状态上的差别。但这只是推测，这次没有直接测。
3. **挑 checkpoint 用原地 val，对 C 不利**：C 的 checkpoint 没有按先走后伸的段来挑，所以 N/C 偏保守，和第一阶段一样。
4. **HEFT 是幸存者样本**：lmlreal 里 HEFT 有 61 局在走路时跑丢，剩下的局本来就偏向走得近、走得稳的那些。
5. **SONIC 的"没有用"也可能是它的 d3 走路数据和任务局不像**：SONIC 的 nearer 是 0.725，起点像先走后伸的数据，所以这个解释不太可能，但没有排除。

**四类陷阱**
- **benchmark 选择**：三个控制器、7 个任务、登记的保留规则下所有段都报了。按任务的结果按登记不报，没有挑任务。
- **数据泄漏**：
  - 真实局的重置编号从 9 100 000 开始，不在任何训练文件里；
  - 留出集用 data.py 自己的桶；
  - 每组的归一化来自它自己的训练集；
  - checkpoint 按原地 val 挑，没碰真实局。
- **指标误用**：
  - 用的是登记的统计量：三个种子各自配对，求 N_k/C_k 的平均，中位数比值。
  - HEFT 的区间几乎退化（点估计在区间上端），已在意外 8 里标出。
  - 第一下、走过去后这些分格只报，不进判据。第 2 条结论只改建议，不改判据的结论。
- **事后挑选**：
  - 18 个训练全部报了，每个训练只有一个按 val 挑的 checkpoint，看到结果以后没有重跑任何东西；
  - walkfit 是评估前登记的（修订 3）；
  - HEFT 的三件事是看到结果以后协调人加问的，结果里已经标为事后分析，也没有用它改判据的结论。

**对照判据**
- 三个控制器都是第一行，种子一致规则满足，判据结论是"先走后伸的数据不需要"。
- HEFT 按登记的说明降为"没有证据"。
- 第一阶段的附加规则 2、3 是 SIMPLE 的任务规则，第二阶段按登记不适用。

**两个阶段合起来，下一步**
1. 结论按控制器分开：
   - **ScaleBFM**：SIMPLE 落在中间行，LML 总体落在第一行。但两边走路后的第一下都差约 30 %，所以不建议只采原地。
   - **SONIC**：只采原地就够。
   - **HEFT**：真实局没有信息，留出集指向"有用"。
2. **另行登记两个后续**（新条目，协调人 11-26 已同意设计，用户已同意新采数据）：
   - 中间方案 Ma / Mb，共用随机对照 Cm，SIMPLE ScaleBFM，6 个训练，在 H20 的 GPU 上跑；
   - 时间对齐 T：全采原地，新采数据，排在 E2a 之后，用 H20 的 CPU。
   - 两个都只做 ScaleBFM，因为它是唯一在两个 benchmark 上都有清楚差距的控制器。
3. 两个阶段的独立复核由协调人安排。


## 独立复核
<!-- 结论要写进论文、用来否掉或投入一个方向、或作为发现报告时必须做。贴独立 agent 的原始回复。 -->
