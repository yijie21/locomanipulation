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


## 独立复核
<!-- 结论要写进论文、用来否掉或投入一个方向、或作为发现报告时必须做。贴独立 agent 的原始回复。 -->
