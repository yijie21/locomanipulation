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

## 结果（贴原始输出，不转述）
<!-- 取指标的命令 + 原始输出；完整日志、checkpoint、评测输出的路径。 -->

## 意外
<!-- 预测和结果哪里对不上？有什么没想到的现象？ -->

## 结论与下一步
<!-- 每个论断引用上面的证据；写明证据强度（运行次数、种子、episode 数）；至少一个其他解释；
     四类陷阱检查：benchmark 选择 / 数据泄漏 / 指标误用 / 事后挑选；结果落在判据哪一侧、下一步做什么。 -->

## 独立复核
<!-- 结论要写进论文、用来否掉或投入一个方向、或作为发现报告时必须做。贴独立 agent 的原始回复。 -->
