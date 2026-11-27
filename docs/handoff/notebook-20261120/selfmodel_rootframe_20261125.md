# selfmodel_rootframe_20261125：自身模型改在电机参考根的框架里重训，先在已有的 SIMPLE 真实局上离线过关

- 日期：2026-11-25（设计、预测、判据；重建数据和训练以前写下，单独提交）
- 机器 / GPU：4090，1–3 号卡。训练每个控制器一张卡，跑之前查 nvidia-smi，0 号卡上有别人的进程，不碰。
- 代码版本：`selfwm` @ `08499167` 加未提交的改动（`jref_data.py` 加一个框架开关，离线评测脚本新写）。跑时抄进输出目录，md5 记在结果里。
- 配置：
  - 训练数据就是 [selfmodel_jrefwm_20261124](selfmodel_jrefwm_20261124.md) 采好的录像，两个控制器各约一万条，不重采；
  - 模型结构、训练设置、划分都和那次一样：`via_point.py train` 默认，5 个成员，512×4，30000 步，桶 0–7 训练、8 早停、9 测试；
  - ScaleBFM 和 SONIC 都不改参数，不用伺服和查表。
- 关联：
  - 用户 2026-11-25 选"(a) 和 (b) 都做"：先在电机参考根的框架里重训，离线过关，再到伸手几何真的会变的局上执行检验。
  - [selfmodel_precision3_20261125](selfmodel_precision3_20261125.md) 和它的复核：
    - 现在的模型和"参考根终点加常数"（kinc_rr）打平。它学的主要是"参考搬到真实骨盆上"那个框架项。
    - 复核离线拟合（ScaleBFM）：
      - 用发令时刻的本体信息做 ridge，能预测剩下误差 t = 手 − er 的一半左右：0.58 cm，c_rr 是 1.10；
      - 只用开发局拟合的 (er, 目标) 仿射到 0.82；
      - 现在的模型隐含的 t 是 1.09。
    - SONIC 上 ridge 几乎学不到，现在的模型隐含的 t 离线最好（3.47 对 c_rr 4.53）。

## 假设

一次命令下去，手停在哪 = er（电机参考根放出来的参考终点离目标多远，发命令时就能算）+ t（控制器跟这条参考的误差）。

t 有一部分随发令时的身体状态和这条参考变化，所以是可以预测的。现在的模型是在"参考搬到真实骨盆上"的框架里训的，学到的主要是框架项，没学到 t。

**在电机参考根的框架里重训，同样的数据、同样的结构，模型能不能逐局预测 t，并且比"一个常数"（c_rr）、比"开发局上拟合的 (er, 目标) 仿射"都准？**

## 做法

### 1. 重建训练数据（`jref_data.py build --frame root`，新开关，默认行为不变）

- **参考**：不再搬到真实骨盆上。用录像里电机自己的参考（`ref` 的根 `[0:7]` 加关节），原样放。
- **坐标**：发命令时刻**参考根**的朝向坐标，原点在参考根，x 朝前。
- **标签**：实际的 24 个球（仿真里的值，标签可以用真值）减参考的球，时间和停稳点的定义照旧。
- **输入：不用仿真真值。** 原来输入里经过真实骨盆的量（`W` 给的发令时夹爪位置、`J` 给的手腕姿态、朝向坐标本身）都换成从机器人自己能知道的量算：
  - 测得的关节角；
  - 机器人对骨盆的估计（里程计 / 站位时锚定的那个，`bel` 一类的字段），放在参考根的框架里。

  `S3` 如果只含关节量，照原样；如果含骨盆的世界位姿，同样换成估计值。具体用了哪些字段，结果里逐项列出。
- **划分**：同一套布局分桶和种子，两份数据逐条对齐。

### 2. 训练

两个控制器各一份，`via_point.py train`，全用默认设置，和 jrefwm 那次一样；然后 `via_point.py dev`、`via_point.py eval`。

### 3. 不用模型的对手：开跑以前冻结

- **c_rr**：`outputs/simple_dev_20261124/precision_const/ctrack_rr_<控制器>.json`，precision3 用的那份，不改。
- **开发局仿射（aff）**：t = a + B·[er, 目标]，在发命令时刻参考根的朝向坐标里，每个轴一个 ridge。
  - 数据是 precision 开发局（第二次）4 次重复的 J 行：ScaleBFM 44 个局次，SONIC 55 个。
  - λ 在开发局里按留一布局选。
  - 训练完模型以前拟合，存 `outputs/simple_dev_20261124/precision_const/aff_rr_<控制器>.json`，记 md5。
  - 另存一份只用离所有测试位置 5 cm 以上的开发位置拟合的（`aff_rr_far_<控制器>.json`），作稳健性对照。precision3 复核发现开发局和测试局有位置重合。

### 4. 离线评测（新脚本，只读已有的局）

- **评测集**：已经跑过的真实 SIMPLE 局上 J 的行和录像，模型没见过：
  - **主评测集**：precision 测试局（抖动）4 次重复，ScaleBFM 112、SONIC 106 个局次；
  - 开发局 4 次重复（44 / 55）；
  - 官方测试局 JR（28 / 29）。
- **每个局次算的量**：
  - **实际的 t**：停稳时实测的夹爪中心 − er，在参考根朝向坐标里。er 的算法同 precision3。
  - **各做法预测的 t**：
    - 新模型 mR：手掌三个球停稳偏移的平均，和 modelf 的取法一样；
    - c_rr；
    - aff 和 aff_far；
    - 旧模型的隐含 t：p − u，同 precision3 复核。
- **离线停偏**：|实际 t − 预测 t|，按局。等于按这个做法挪了目标、并且执行没有额外误差时手会停的地方。
- **配对和统计**：和 precision 系列一样，按局配对；按布局（1 cm）重抽 4000 次的 95 % 区间；按布局均值的符号检验。另报按布局中位数的符号检验（precision3 复核建议，防止个别局主导）。
- **当天检验**：
  - 新模型在这些局上的预测误差（中位数、90 % 分位），以及"参考本身"的误差；
  - 发令时 29 个关节超出训练数据 0.5–99.5 % 范围的个数（`jref_data.py range`）。

### 5. 之后（这次不做）

过关的控制器，另登记执行检验：
- **局**：在伸手几何真的会变的局上（站位扰动加在桌沿限位之后，目标和骨盆高度变化），用新的位置和新的抖动种子，常数从离测试位置 5 cm 以上的开发位置重估。
- **对手**：kinc_rr 和 aff。
- **成功数**：至少 3 个规划器种子，以 replan 为参照组。

## 度量

- **核心**（每个控制器，主评测集）：mR − c_rr 和 mR − aff 的离线停偏，按局配对的中位数、布局区间、两种符号检验。
- **另报**：
  - mR − 旧模型、mR − aff_far；
  - 开发局和官方局上同样的比较；
  - jref 自己测试桶（桶 9）上新旧模型的误差（`via_point.py eval` 的输出）。

## 预测（写下时看过 precision3 复核的离线数：ridge 0.58、aff 0.82、c_rr 1.10、旧模型 1.09，SONIC 3.47 / 4.53；没看过任何新模型的结果）

**ScaleBFM（主评测集）**：
- **mR 离线停偏中位数**：0.7–1.2 cm。
  - 理由：ridge 的 0.58 是在测试局本身上留一布局拟合的，分布最近。mR 只见过空地上的伸手，训练时关了桌子的碰撞，而测试局每局都碰桌。
  - 只用开发局拟合的 ridge 也只有 0.91。
- **核心比较**：mR − c_rr −0.5 到 +0.1；mR − aff −0.3 到 +0.3。
- **两条都过的可能**约两成半。
- **aff**：本身在 0.75–0.95；aff_far 比 aff 差 0–0.2 cm。

**SONIC（主评测集）**：
- **mR**：3.3–4.8 cm。
- **核心比较**：mR − c_rr −1.2 到 +0.3。aff 在 SONIC 上比 c_rr 好不了多少，4.2–5.0。
- **两条都过的可能**约三成半。

**两个控制器**：
- **jref 测试桶**：新模型的停稳点误差和旧模型各自框架里的相比，差不超过 30 %。框架换了，标签的方差也变了，所以只作参考。
- **关节范围**：中位数 0–1，最大不超过 9。

## 判据（什么结果会让我改做法，下一步做什么）

"赢"的定义同 precision 系列：按布局重抽的 95 % 区间整个在 0 以下，并且按布局均值的符号检验 p < 0.05。每个控制器分开，以主评测集为准。

- **mR 同时赢 c_rr 和 aff**：
  - 读法：在正确的框架里，自身模型能逐局预测跟踪误差，比常数、比开发局拟合的仿射都准。
  - 下一步：登记第 5 步的执行检验。
- **mR 赢 c_rr、不赢 aff**：
  - 读法：可预测的那部分，用开发局几十局拟合一个仿射就够，自身模型的长处最多是"不需要任务局"（数据效率）。
  - 下一步：向用户报告，第 5 步仍做，但对手里必须带 aff，结论写成数据效率。
- **mR 两个都不赢**：
  - 读法：换了框架，模型还是学不到 t，SIMPLE 上自身模型的精度论断没有支撑。
  - 下一步：向用户报告，不进第 5 步的执行，先查原因：训练数据没有桌子碰撞、分布不同，还是模型容量。
- **按布局中位数的符号检验和按均值的结论不同**：照实报两种，以均值为准（登记的定义），但写明。

## 预测修订

2026-11-25，训练和任何离线评测以前（实现的 agent 在冻结 aff 以后、开训以前停下来问）：

- **aff_far 照原文做不出来**：开发位置的开局目标离最近的测试目标都不到 5 cm（ScaleBFM 最远 2.65 cm，SONIC 3.03 cm），符合条件的是 0 个。
- **改为阈值 1 cm**：正好去掉 precision3 复核第 4 条说的那些重合位置（0.21–0.93 cm）。
  - ScaleBFM 剩 6 个位置、24 个局次；SONIC 剩 4 个位置、9 个局次。SONIC 的这份太少，只报不读。
  - 它仍只是稳健性对照，不进核心判据。
- **已冻结的 aff**（agent 原样输出；01:26Z 写好，登记提交 `8bd16b1e` 在 00:59:25Z）：

  ```
  aff_rr_scalebfm.json: n 44 episode-repeats, 11 positions, 9 layouts; lambda per axis [1000.0, 30.0, 10.0]; a [-1.51175, -5.23368, 5.62639] cm; leave-one-layout miss median 0.9836 cm (in sample 0.8157)
  aff_rr_sonic.json: n 55 episode-repeats, 16 positions, 15 layouts; lambda per axis [100.0, 100.0, 1000.0]; a [9.14623, -3.30076, 0.94023] cm; leave-one-layout miss median 7.6564 cm (in sample 6.947)
  ```

  md5 `7c2a7955…` / `7dc30323…`。文件放在 `outputs/simple_eval_20261112/rootframe_20261125/aff/`，没放 `precision_const/`，因为新输出只放新目录。
- **输入核对**（训练以前）：
  - 默认开关重建的数据和原数据逐位相同。
  - 把录像里的真实骨盆挪 5 cm、转 4°，把 W、J 打乱，新框架的输入 x 一位不变，只有标签变。所以输入里没有仿真真值。
  - S3 原样保留：其中的骨盆高度本来就是里程计估计，`simple_bench.py` 第 660 行强制 `root == "odo"`。
  - 估计骨盆的 xy 和朝向用站位锚点（行里的 `bel`），因为发令时控制器自己的里程计 xy 没有记下来。
- **代价，事先写明**：估计的夹爪离真实夹爪，ScaleBFM 中位数 2.0–2.3 cm，SONIC 3.4–7.3 cm。站位以后骨盆漂了多少，模型看不到。
- **预测、判据、核心比较都不变。**

## 结果（贴原始输出，不转述）

### 运行

- **aff_far 冻结**：按修订用 1 cm 阈值，01:28:51Z / 01:28:55Z 写好，开训是 01:32:14Z。aff_rr 那份不动，脚本输出 "exists, kept"。原始输出：

  ```
  aff_rr_far_scalebfm.json: n 24 episode-repeats, 6 positions, 6 layouts; lambda per axis [1000.0, 100.0, 10.0]; a [-1.68657, -3.07032, 4.2606] cm; ...; leave-one-layout miss median 1.1866 cm (in sample 0.8533); md5 b3a8ee6b549eebaef53917c8d4177776
  aff_rr_far_sonic.json: n 9 episode-repeats, 4 positions, 4 layouts; lambda per axis [1000.0, 1000.0, 100.0]; a [-16.06567, 13.6095, 3.0664] cm; ...; leave-one-layout miss median 33.8469 cm (in sample 25.5149); md5 e53316acee1cdf9d522d88d3e130f381
  ```

- **训练**：
  - 方式：tmux `rf_train`，脚本 `rootframe_20261125/logs/train.sh`（md5 6c1c81c9…，第一行 `# notebook: selfmodel_rootframe_20261125`）。ScaleBFM 用 1 号卡，SONIC 用 2 号卡。`via_point.py train` 全默认。
  - 划分：ScaleBFM 8047/592/378/987，SONIC 8054/602/378/975。
  - 时间：train.ok 01:43:00Z / 01:43:08Z，dev.ok 01:43:36Z / 01:43:44Z，eval.ok 01:44:02Z / 01:44:08Z。
  - 当时视频报告的 Isaac 渲染也在 1、2 号卡上，每个成员训了约 110 s（jrefwm 那次 41 s）。
- **模型 md5**：
  - 新模型：ScaleBFM 6431e578…，SONIC eb9449db…。
  - 旧模型（jrefwm）：3eeaa14b… / 8fadf65d…。
- **代码 md5**：rootframe_eval 885c2755…（出表那一版；冻结 aff 那一版 179cabd3…，aff 函数相同，两份都在 `asrun/`），jref_data 9019cbea…，precision 9fa88ceb…，s_check 4106eba5…，via_point addad976…（没改），via_model e4a2e9f9…。
- **`via_point.py dev` 的 STOP LINE**（开发桶，空地伸手）：

  ```
  scalebfm: validation, standing, palm path over 1.0 m (THE STOP LINE: drop of the arm's median >= 0.25): 38 paths
    direction  swept-region error: arm median  1.28 p90  5.44 cm | hand median  0.94 cm | ...
    drop of the arm's median against the reference: 0.85 (95 % interval 0.67 to 0.85, resampling routes); hand 0.83; constant: arm 0.02
  sonic: validation, standing, palm path over 1.0 m (THE STOP LINE: ...): 26 paths
    direction  swept-region error: arm median  5.62 p90 12.34 cm | hand median  4.19 cm | ...
    drop of the arm's median against the reference: 0.46 (95 % interval 0.34 to 0.59, resampling routes); hand 0.46; constant: arm 0.02
  ```

### 离线表

`outputs/simple_eval_20261112/rootframe_20261125/table.txt`（md5 ad044b5391ca8307c4aa3534968c001e，01:44:48Z）。逐局次结果在 `table_rows_<ctrl>.jsonl`，关节范围的原始输出在 `range_<ctrl>.txt`。下面摘相关行，主评测集整段照贴。

```
offline miss = |t - predicted t| per episode-repeat, cm, t = measured jaw centre - jw (the reference-root end), the root's heading frame; paired d = miss(first) - miss(second) < 0: the first is better. Interval: 95 % of the median of d, 4000 layout resamples (precision.by_layout, seed 0); sign tests over layouts (1 cm) on their means (registered) and on their medians. win = interval below 0 and the layout-mean sign p < 0.05 (the verdict reads only the main set)

==== scalebfm test: precision test episodes (jittered), 4 repeats: THE MAIN SET
112 episode-repeats, 28 positions, 18 layouts (1 cm)
  checks: |er - shifts' er_h_cm| max 0.0005 cm; |c_rr - shifts' crr_h_cm| max 0.0000 cm (the full mean); |old p - shifts' pred_h_cm| max 0.0005 cm; root-frame reference end - jw max 4.4e-08 m
  film route (112 of 112 have a check row; S0 max |diff| 0.0e+00): x max |diff| from the plan-built input 4.91e-03 (the reference keys 8.10e-04, the rest 4.91e-03 at column 378); mR from the film's input - from the plan's: max 0.001 cm
  mR's own error (the filmed settled palm offset - mR's, 112 reaches): median 1.15 p90 2.56 cm; the reference itself (the offset): median 2.36 p90 3.51 cm; the filmed palm offset - the measured t: median 1.17 cm
    via_point report, test root-frame checks: 112 paths
    reference  swept-region error: arm median  5.67 p90  6.87 cm | hand median  3.98 cm | margin (median radius)  7.01 cm, inside 0.94 | fixed radius 11.91 cm, inside 1.00
    constant   swept-region error: arm median  5.71 p90  7.62 cm | hand median  3.80 cm | margin (median radius)  7.85 cm, inside 0.96 | fixed radius 11.07 cm, inside 0.99
    direction  swept-region error: arm median  2.76 p90  3.43 cm | hand median  2.53 cm | margin (median radius)  1.46 cm, inside 0.21 | fixed radius  2.80 cm, inside 0.52
    drop of the arm's median against the reference: 0.51 (95 % interval 0.48 to 0.55, resampling paths); hand 0.37; constant: arm -0.01
    fixed radius: direction / reference 0.24 (set on the calibration paths); direction wins on 1.00 of the paths
  joints at the command outside the training paths' 0.5-99.5 % band (112 episode-repeats): median 0.0, mean 0.75, max 9, counts [71, 22, 11, 2, 3, 0, 1, 1, 0, 1]
  offline miss, cm            median   mean    p90
    t itself (kin_rr)          1.88   1.96   3.22
    c_rr                       1.10   1.14   1.72
    aff                        0.81   0.96   1.76
    aff_far                    1.02   1.08   1.65
    old (p - u)                1.09   1.29   2.23
    old, review's p-(er-g)     1.09   1.29   2.19
    mR                         0.96   1.14   1.92
  paired d                    median [95 % interval]      layout means better/worse p   layout medians better/worse p   win
    mR - c_rr      -0.01 [-0.19, +0.14]   10/8  p 0.815   10/8  p 0.815   no
    mR - aff       +0.17 [+0.01, +0.33]    4/14 p 0.031    4/14 p 0.031   no
    mR - old       -0.15 [-0.21, -0.10]   16/2  p 0.001   16/2  p 0.001   WIN
    mR - aff_far   +0.02 [-0.17, +0.32]    9/9  p 1.000    9/9  p 1.000   no
    aff - c_rr      -0.21 [-0.31, -0.12]   17/1  p 0.000   15/3  p 0.008   WIN (extra, not registered)

==== scalebfm dev: precision development episodes, 4 repeats
44 episode-repeats, 11 positions, 9 layouts (1 cm)
  mR's own error (the filmed settled palm offset - mR's, 44 reaches): median 1.04 p90 1.71 cm; the reference itself (the offset): median 2.28 p90 3.47 cm; the filmed palm offset - the measured t: median 1.12 cm
  joints at the command outside the training paths' 0.5-99.5 % band (44 episode-repeats): median 0.0, mean 2.07, max 10, counts [26, 5, 1, 1, 0, 2, 2, 2, 2, 1, 2]
  offline miss, cm            median   mean    p90
    t itself (kin_rr)          1.85   1.99   2.99
    c_rr                       1.16   1.24   2.01
    aff                        0.98   1.10   1.78
    aff_far                    1.07   1.17   2.02
    old (p - u)                1.14   1.19   1.43
    old, review's p-(er-g)     1.15   1.19   1.45
    mR                         0.91   1.03   1.39
  paired d                    median [95 % interval]      layout means better/worse p   layout medians better/worse p   win
    mR - c_rr      -0.14 [-0.75, +0.18]    6/3  p 0.508    6/3  p 0.508   no
    mR - aff       -0.14 [-0.33, +0.15]    6/3  p 0.508    6/3  p 0.508   no
    mR - old       -0.14 [-0.24, -0.07]    7/2  p 0.180    8/1  p 0.039   no
    mR - aff_far   -0.11 [-0.46, +0.04]    6/3  p 0.508    6/3  p 0.508   no
    aff - c_rr      -0.12 [-0.33, +0.04]    7/2  p 0.180    6/3  p 0.508   no (extra, not registered)

==== scalebfm jr: official JR episodes (bench_rgbd_20261124)
28 episode-repeats, 28 positions, 18 layouts (1 cm)
  mR's own error (the filmed settled palm offset - mR's, 28 reaches): median 1.37 p90 2.85 cm; the reference itself (the offset): median 2.63 p90 4.26 cm; the filmed palm offset - the measured t: median 1.19 cm
  joints at the command outside the training paths' 0.5-99.5 % band (28 episode-repeats): median 0.0, mean 1.00, max 9, counts [18, 5, 2, 1, 0, 0, 0, 1, 0, 1]
  offline miss, cm            median   mean    p90
    t itself (kin_rr)          2.16   2.28   3.84
    c_rr                       1.03   1.29   2.32
    aff                        1.01   1.21   2.36
    aff_far                    1.05   1.24   2.28
    old (p - u)                1.19   1.53   2.76
    old, review's p-(er-g)     1.19   1.52   2.70
    mR                         1.04   1.29   2.22
  paired d                    median [95 % interval]      layout means better/worse p   layout medians better/worse p   win
    mR - c_rr      -0.01 [-0.29, +0.18]   11/7  p 0.481   11/7  p 0.481   no
    mR - aff       +0.11 [-0.30, +0.32]    7/11 p 0.481    7/11 p 0.481   no
    mR - old       -0.19 [-0.26, -0.12]   18/0  p 0.000   17/1  p 0.000   WIN
    mR - aff_far   +0.06 [-0.28, +0.27]   10/8  p 0.815   10/8  p 0.815   no
    aff - c_rr      -0.08 [-0.20, +0.10]   12/6  p 0.238   12/6  p 0.238   no (extra, not registered)

==== scalebfm jref test bucket 9 (each model on its own data and frame; via_point eval's lines below)
  new (root frame) 987 paths: settled palm error median 0.29 p90 0.64 cm (the reference itself 2.22 / 3.82); settled, mean over the 24 spheres: median 0.31 cm
  old (jrefwm)     987 paths: settled palm error median 0.32 p90 0.77 cm (the reference itself 3.45 / 5.96); settled, mean over the 24 spheres: median 0.35 cm

==== sonic test: precision test episodes (jittered), 4 repeats: THE MAIN SET
106 episode-repeats, 28 positions, 19 layouts (1 cm)
  checks: |er - shifts' er_h_cm| max 0.0005 cm; |c_rr - shifts' crr_h_cm| max 0.0000 cm (the full mean); |old p - shifts' pred_h_cm| max 0.0005 cm; root-frame reference end - jw max 5.4e-08 m
  film route (105 of 106 have a check row; S0 max |diff| 0.0e+00): x max |diff| from the plan-built input 4.93e-03 (the reference keys 9.37e-04, the rest 4.93e-03 at column 378); mR from the film's input - from the plan's: max 0.008 cm
  mR's own error (the filmed settled palm offset - mR's, 105 reaches): median 4.03 p90 13.84 cm; the reference itself (the offset): median 4.72 p90 14.48 cm; the filmed palm offset - the measured t: median 1.06 cm
    via_point report, test root-frame checks: 105 paths
    reference  swept-region error: arm median 12.15 p90 21.75 cm | hand median  7.70 cm | margin (median radius) 20.76 cm, inside 0.90 | fixed radius 20.24 cm, inside 0.86
    constant   swept-region error: arm median 12.08 p90 20.56 cm | hand median  7.39 cm | margin (median radius) 19.59 cm, inside 0.88 | fixed radius 18.98 cm, inside 0.85
    direction  swept-region error: arm median  8.60 p90 20.29 cm | hand median  6.29 cm | margin (median radius) 12.75 cm, inside 0.78 | fixed radius 10.97 cm, inside 0.61
    drop of the arm's median against the reference: 0.29 (95 % interval 0.14 to 0.37, resampling paths); hand 0.18; constant: arm 0.01
    fixed radius: direction / reference 0.54 (set on the calibration paths); direction wins on 0.75 of the paths
  joints at the command outside the training paths' 0.5-99.5 % band (106 episode-repeats): median 0.0, mean 0.74, max 5, counts [68, 20, 6, 5, 4, 3]
  offline miss, cm            median   mean    p90
    t itself (kin_rr)          4.31   7.13  14.40
    c_rr                       4.53   6.58  11.46
    aff                        6.26   7.53  10.68
    aff_far                   25.50  24.03  27.55
    old (p - u)                3.47   6.15  13.78
    old, review's p-(er-g)     3.43   6.20  14.14
    mR                         3.79   6.16  14.88
  paired d                    median [95 % interval]      layout means better/worse p   layout medians better/worse p   win
    mR - c_rr      -0.24 [-1.48, +0.08]   11/8  p 0.648   11/8  p 0.648   no
    mR - aff       -2.46 [-3.67, -0.46]   14/5  p 0.064   14/5  p 0.064   no
    mR - old       +0.03 [-0.32, +0.19]   12/7  p 0.359    9/10 p 1.000   no
    mR - aff_far  -20.69 [-22.36, -19.43]   19/0  p 0.000   19/0  p 0.000   WIN (SONIC's far fit: reported, not read)
    aff - c_rr      +0.86 [+0.35, +1.68]    3/16 p 0.004    3/16 p 0.004   no (extra, not registered)

==== sonic dev: precision development episodes, 4 repeats
55 episode-repeats, 16 positions, 15 layouts (1 cm)
  mR's own error (the filmed settled palm offset - mR's, 54 reaches): median 2.75 p90 14.08 cm; the reference itself (the offset): median 3.59 p90 14.69 cm; the filmed palm offset - the measured t: median 1.10 cm
  joints at the command outside the training paths' 0.5-99.5 % band (55 episode-repeats): median 0.0, mean 0.71, max 7, counts [38, 8, 4, 2, 1, 0, 1, 1]
  offline miss, cm            median   mean    p90
    t itself (kin_rr)          3.37  10.49  19.96
    c_rr                       4.31  10.21  15.64
    aff                        7.66  12.36  12.88
    aff_far                   25.90  27.68  29.29
    old (p - u)                2.83   9.12  18.24
    old, review's p-(er-g)     2.75   9.32  17.99
    mR                         2.50   8.91  17.96
  paired d                    median [95 % interval]      layout means better/worse p   layout medians better/worse p   win
    mR - c_rr      -1.25 [-2.35, -0.88]   12/3  p 0.035   12/3  p 0.035   WIN
    mR - aff       -4.54 [-5.67, -1.83]   12/3  p 0.035   12/3  p 0.035   WIN
    mR - old       -0.21 [-0.32, -0.05]   10/5  p 0.302   10/5  p 0.302   no
    mR - aff_far  -23.13 [-24.10, -21.31]   15/0  p 0.000   15/0  p 0.000   WIN (SONIC's far fit: reported, not read)
    aff - c_rr      +2.28 [+1.17, +3.70]    2/13 p 0.007    2/13 p 0.007   no (extra, not registered)

==== sonic jr: official JR episodes (bench_rgbd_20261124)
29 episode-repeats, 29 positions, 25 layouts (1 cm)
  mR's own error (the filmed settled palm offset - mR's, 29 reaches): median 6.08 p90 18.50 cm; the reference itself (the offset): median 4.24 p90 16.04 cm; the filmed palm offset - the measured t: median 1.13 cm
    drop of the arm's median against the reference: 0.09 (95 % interval -0.09 to 0.23, resampling paths); hand 0.02; constant: arm -0.01
  joints at the command outside the training paths' 0.5-99.5 % band (29 episode-repeats): median 0.0, mean 0.38, max 3, counts [21, 6, 1, 1]
  offline miss, cm            median   mean    p90
    t itself (kin_rr)          3.75   5.68  15.21
    c_rr                       4.33   5.41  11.01
    aff                        6.41   7.00  11.75
    aff_far                   25.11  23.72  27.30
    old (p - u)                4.90   8.89  17.19
    old, review's p-(er-g)     5.12   8.97  17.12
    mR                         5.25   9.63  17.11
  paired d                    median [95 % interval]      layout means better/worse p   layout medians better/worse p   win
    mR - c_rr      +0.85 [-1.35, +2.09]   11/14 p 0.690   11/14 p 0.690   no
    mR - aff       -0.19 [-3.86, +2.20]   14/11 p 0.690   14/11 p 0.690   no
    mR - old       -0.12 [-0.59, +0.33]   14/11 p 0.690   14/11 p 0.690   no
    mR - aff_far  -19.86 [-22.19, -16.86]   22/3  p 0.000   22/3  p 0.000   WIN (SONIC's far fit: reported, not read)
    aff - c_rr      +0.78 [-0.11, +2.42]    9/16 p 0.230    9/16 p 0.230   no (extra, not registered)

==== sonic jref test bucket 9 (each model on its own data and frame; via_point eval's lines below)
  new (root frame) 975 paths: settled palm error median 1.77 p90 5.30 cm (the reference itself 2.76 / 8.84); settled, mean over the 24 spheres: median 1.92 cm
  old (jrefwm)     975 paths: settled palm error median 1.88 p90 4.24 cm (the reference itself 9.06 / 17.19); settled, mean over the 24 spheres: median 1.86 cm
```

### 和条目不一致的地方（实现 agent 逐条列的，摘要）

1. aff 文件在 `rootframe_20261125/aff/`，阈值 1 cm，修订已写。
2. SONIC 的 aff_far 坏了：在拟合用的局上中位数也偏 25.5 cm。9 个局次里有 t 很大的局，λ 又顶在 1000，预测几乎是一个被大值拉偏的均值。已定为只报不读。
3. 好几个轴的 λ 选在网格上限 1000：ScaleBFM aff 的 x 轴、SONIC aff 的 z 轴、两份 far。
4. `via_point.py eval` 只跑了桶 9，真实局上的模型误差改在 rootframe_eval 里，用 via_point 自己的 report 函数算，也就是表里的 "via_point report" 段。
5. SONIC 主评测集和开发局各有 1 个局次带摔倒标记，没有检查数据。"mR 自身误差"在 105 / 54 局上算，离线停偏用全部局次。
6. mR 是手掌三个球的偏移，t 是夹爪中心的偏移，两者在录像里差 1.1–1.2 cm（中位数）。
   - agent 做了一个没登记的探查：ScaleBFM 上这个差几乎是常数，均值 [−0.02, 0.83, 0.69]，标准差 0.1–0.3 cm。从 mR 减去它的开发局均值，主评测集反而从 0.96 变成 1.01；SONIC 从 3.70 变成 3.57。
   - 所以这个差解释不了打平，只作参考。
7. 表格代码试跑时临时拿旧模型顶 mR，发生在新模型训完以前，结果行当时滤掉没看。
8. 远程仓库在 bc426d5 加别人没提交的改动；本实验用到的文件 md5 都和 Mac 一致。

## 意外

预测和结果的对照，主评测集：

| 项 | 预测 | 结果 | 落在区间里吗 |
|---|---|---|---|
| ScaleBFM mR | 0.7–1.2 | 0.96 | 是 |
| ScaleBFM mR − c_rr | −0.5 到 +0.1 | −0.01 | 是 |
| ScaleBFM mR − aff | −0.3 到 +0.3 | +0.17 | 是 |
| ScaleBFM aff | 0.75–0.95 | 0.81 | 是 |
| ScaleBFM aff_far − aff | 0 到 +0.2 | 1.02 − 0.81 = +0.21 | 刚好出界 |
| SONIC mR | 3.3–4.8 | 3.79 | 是 |
| SONIC mR − c_rr | −1.2 到 +0.3 | −0.24 | 是 |
| SONIC aff | 4.2–5.0 | 6.26 | 否，更差 |
| 桶 9 新旧之差 | 30 % 以内 | 0.29 / 0.32，1.77 / 1.88；SONIC p90 5.30 对 4.24，差 25 % | 是 |
| 关节超范围的最大值 | ≤ 9 | ScaleBFM 开发局 10，其余 ≤ 9 | 开发局出界 1 |

"两条都过"的可能我写的是两成半和三成半，实际两个控制器都没过。

没想到的现象：

1. **ScaleBFM 上 aff 明显赢了常数**：主评测集 aff − c_rr −0.21 [−0.31, −0.12]，17/1 个布局（没登记）。
   - 也就是说，t 里确实有一块不是常数，而且它是 (er, 目标) 的线性函数，用 44 个开发局次就拟合得出来，mR 没学到。
   - 换框架以后，mR 比旧模型好了 0.15 cm（16/2），但也只到常数的水平。
2. **训练分布内外差 4 倍**：ScaleBFM 新模型在自己的测试桶上停稳点误差中位数 0.29 cm，到 SIMPLE 主评测集上是 1.15 cm。SONIC 是 1.77 对 4.03。
3. **SONIC 在官方局上，模型比"参考本身"还差**：mR 自身误差 6.08 cm，参考本身 4.24 cm；扫掠误差的下降只有 0.09 [−0.09, 0.23]。主评测集上是 0.29。
4. **SONIC 的 aff 比常数差**：+0.86，3/16 个布局。~~原因是尾巴重、λ 顶格~~。**复核更正**：真正的原因是拟合里混进了开发局 r2 L3_e4 这个摔倒局（|t| 219.7 cm）。去掉它，aff 是 3.32，赢 c_rr，和 mR 打平。
5. **SONIC 开发局上 mR 两条都赢**（12/3，p 0.035），主评测集和官方局上都没有。开发局只有 15 个布局。
6. ~~mR 没见过 SIMPLE 局，却把平均偏差估到了开发局常数的水平~~。**复核更正，收回**：这是取点不一致造成的。在同一个点（夹爪）上，用模型自己的夹爪预测去比，模型平均偏 0.86 cm，停偏 1.22，显著比开发常数差（+0.27，4/14，p 0.031）。

## 结论与下一步

**对照判据**：两个控制器都落在"mR 两个都不赢"。证据是"结果"里主评测集的 paired d 行：
- **ScaleBFM**：
  - mR − c_rr −0.01 [−0.19, +0.14]，10/8，p 0.815；
  - mR − aff +0.17 [+0.01, +0.33]，4/14，p 0.031。**复核更正**：只是边缘显著。这份 aff 是在和测试位置重合的开发局上拟合的，样本量匹配的对照显示它的优势约一半来自重合；去掉重合后打平。
- **SONIC**：
  - mR − c_rr −0.24 [−1.48, +0.08]，11/8，p 0.648；
  - mR − aff −2.46 [−3.67, −0.46]，14/5，p 0.064，按定义不算赢。**复核更正**：这 −2.46 全是 aff 吃进摔倒局造成的。换成干净的 aff，打平（+0.20，9/10）。所以 SONIC 上应当写：mR 和 c_rr、和干净的 aff 都打平，而且检验力不够，区间排除不了 1 cm 以上的优势。
- **两种符号检验**（按布局均值、按布局中位数）在这四条上结论相同。

**读法（照判据原文）**：换了框架，模型还是学不到 t 中逐局可预测的那部分，SIMPLE 上自身模型的精度论断没有支撑。

更具体地说：
- **ScaleBFM**：
  - ~~自身模型不用任何 SIMPLE 局，就把平均偏差估到了开发局常数的水平~~（复核收回，见意外 6）。
  - 逐局变化的那部分它没有抓到。**复核更正**：aff 对 c_rr 的 0.21 里，约 0.09 来自"44 局的常数比 11 局的准"，斜率只占约 0.12（aff − c44 −0.12，14/4，p 0.031）。在 er + 目标之外再加 mR，只多 −0.05，不显著。
  - 和旧模型比是进步（−0.15，16/2），但到不了能用的程度。
- **SONIC**：主评测集上模型比参考本身略好（下降 0.29），官方局上不比参考好（0.09），和常数比分不出高下。**复核更正**：读法"学不到逐局的部分"在 SONIC 上不成立。mR 的输出里有逐局信息：在测试集内按布局留一做 3×3 标定，到 3.17，常数是 4.09（17/2，p 0.001）。只是幅度大了 2–4 倍，标定不对。这是事后探查，只作参考。

**证据强度**：
- 每个控制器训练一次（5 个成员，一个种子）。
- 主评测集 112 / 106 个局次，28 个位置，18 / 19 个布局。
- 只是离线：离线停偏假设"挪了目标以后执行没有额外误差"，没有真正执行。

**其他可能的解释**：
1. **训练分布没有桌子**：训练用的是空地伸手，关了桌子碰撞；SIMPLE 每局都碰桌。分布内 0.29 cm，SIMPLE 上 1.15 cm（意外 2），差最可能出在这里。
2. **骨盆估计的误差**：输入用的是站位锚点，估计的夹爪离真实夹爪中位数 2 cm（ScaleBFM）。
   - 这个误差在训练数据里同样存在。
   - aff 只用了 er 和目标，也没有真值，却比 mR 准。**复核更正**：aff 看得到目标位置，mR 看不到。只用 mR 也看得到的 jw 来拟合，比 aff 差约 0.07。不是大头，但不能说已经排除。
3. **取点定义不同**：手掌三个球对夹爪中心。**复核更正**：模型自己能预测手掌和夹爪的差，所以取点不一致其实在帮 mR，约 0.26 cm。正确的比法是用模型自己的夹爪预测（8 个球刚体拟合，和 t 只差 0.04 cm）去比 t：停偏 1.22，比开发常数显著差。今后的度量都应当在夹爪上比。
4. **模型容量**：分布内误差 0.29 cm，不像是容量不够。

**四类陷阱**：
- **benchmark 选择**：只看 TabletopGrasp 上 J 有路线的局，这是登记好的唯一有路线的任务，没有挑。主评测集事先指定，结论只读主评测集。
- **数据泄漏**：
  - mR 只用 jref 空地数据训练，没见过 SIMPLE 局。**复核补充**：但训练数据是在 SIMPLE 四个任务的布局上采的，包括 TabletopGraspMP。112 个测试局次里有 92 个落在训练过的布局上，这一点对模型有利。所有开发位置都在测试位置 2.65 cm 以内，这个 benchmark 里没有干净的几何。
  - c_rr 和 aff 是在开发局上拟合的，开发局和测试局有位置重合，这一点对对手有利、对 mR 不利。
  - 用 1 cm 阈值去掉重合以后，ScaleBFM 的 aff_far 和 mR 打平（+0.02 [−0.17, +0.32]，9/9），结论"mR 不赢"不变。
  - 输入不含真值的核对已经通过：挪动真实骨盆，x 不变。
- **指标误用**：度量、统计、"赢"的定义都是登记的；两种符号检验都报了；没换指标。**复核补充**：登记的度量本身就是拿手掌比夹爪；SONIC 的 aff 吃进了模型数据会排除的摔倒局。
- **事后挑选**：每个控制器只训一次，用 via_point 默认的早停（桶 8）。试跑表格代码时拿旧模型顶 mR，那时新模型还没训完，结果行没看。

**下一步（照判据）**：
- 向用户报告，不进第 5 步的执行。
- 先查原因。候选的查法，都还没登记：
  - (a) **零成本，离线**：在 SIMPLE 局里按碰桌的强弱分组，看 mR 的误差是不是集中在碰桌重的局。**复核**：录像里没有接触力，`obj_cm` 也没有变化，只能用手部球和桌面的穿透量来算；区分力最弱。更便宜、应当先做的是：把 mR 在夹爪上的误差拆成平均偏差、增益和残差，再和 jref 测试桶里 TabletopGraspMP 布局的同样拆分比。
  - (b) **要重跑**：把同一批 J 命令在去掉桌子的 SIMPLE 场景里再执行一遍，用 ScaleBFM，28 个位置。
    - **复核**：要在同一批里加一份有桌子的重复作对照，并在夹爪上、用模型自己的夹爪预测来比。可以预先写：如果原因是碰桌，模型偏差里 z +0.6、y +0.5 cm 那一块应当消失。
    - 如果 mR 的误差降到 0.3–0.5 cm，原因就是碰桌，属于场景效应，和"自身模型只描述身体"的范围一致；
    - 如果不降，原因在别处。
  - (c) **只有在 (a) 或 (b) 指向碰桌时才考虑**：在自身模型的数据里加带桌面的伸手。这会改变"空地采数据、场景无关"的设定，要先问用户。

## 独立复核

2026-11-25，独立 agent，新上下文。给它的材料：条目、precision3 的复核、代码、4090 上的输出。它从 `table_rows` 出发重算了核心比较，和表格一致。下面是回复原文，只删了开头的寒暄。上面"意外"和"结论"里标"复核更正"的地方，都已经按它改过。

### 判定

| 说法 | 判定 |
|---|---|
| 主结论：原始 mR 按登记判据，在主评测集上两个控制器都"两个都不赢" | **支持**。数全部对得上 |
| 读法："换了框架，模型还是学不到 t 中逐局可预测的那部分" | ScaleBFM **大体支持，但几条理由要改**；SONIC **不支持** |
| SONIC 上"不比 aff 准" | **证据不足**。那份 aff 的拟合数据里混进了一个摔倒局 |

我先读"结果"，在读"结论"以前自己定的判定是：两个都不赢。具体说，ScaleBFM 上 aff 比 mR 准，但只是勉强；SONIC 上 mR 对 aff 接近赢，可这个对手本身是坏的。这和条目一致，分歧在怎么读。

### 1. 核对数字

条目里贴的 140 行和 `table.txt` 逐行对过，全部原样出现在文件里（md5 ad044b53…）。

我从 `table_rows_<ctrl>.jsonl` 出发独立重算了核心比较。布局是我自己用 bench 行里的目标位置按 1 cm 聚的，重抽用的是另一个种子：

```
scalebfm test rows 112 unique (rep,name) 112
layouts: file 18 mine 18 same partition: True
  c_rr     median 1.096 mean 1.143 p90 1.722  |mine-stored| max 1.0e-04
  aff      median 0.810 mean 0.958 p90 1.756  |mine-stored| max 1.3e-04
  mR       median 0.959 mean 1.143 p90 1.920  |mine-stored| max 1.2e-04
  mR-c_rr     median -0.013 [-0.19, +0.14]  layout means 10/8 p 0.815  layout medians 10/8 p 0.815
  mR-aff      median +0.173 [+0.02, +0.34]  layout means 4/14 p 0.031  layout medians 4/14 p 0.031
  mR-old      median -0.154 [-0.21, -0.09]  layout means 16/2 p 0.001  layout medians 16/2 p 0.001
  mR-aff_far  median +0.022 [-0.18, +0.33]  layout means 9/9 p 1.000  layout medians 9/9 p 1.000
  aff-c_rr     median -0.209 [-0.31, -0.11]  layout means 17/1 p 0.000  layout medians 15/3 p 0.008
sonic test rows 106 unique (rep,name) 106
layouts: file 19 mine 19 same partition: True
  mR-c_rr     median -0.241 [-1.40, +0.08]  layout means 11/8 p 0.648  layout medians 11/8 p 0.648
  mR-aff      median -2.462 [-3.66, -0.41]  layout means 14/5 p 0.064  layout medians 14/5 p 0.064
```

除了重抽区间在小数第二位上有抖动，全部一致。

"结论"里有几个数不在 `table.txt` 里，来源如下：
- "2 cm"：来自 `verify/fields.txt`，测试局中位数 2.03–2.32。
- "0.1–0.3、1.01"：来自 agent 没登记的探查。我复现出 sd [0.09 0.24 0.28]，减掉后 1.01。

### 2. 四类陷阱

**(a) mR 的输入里没有仿真真值：成立。**

- **代码**：`--frame root` 下，骨盆的 xy 和朝向取参考根（站位锚点），高度取 S3 里的里程计，倾斜取重力读数；夹爪和手腕用测得的关节角做正运动学；参考也不再搬到真实骨盆上。
- **`verify/` 的检验**：扰动以后 x 一位不变；默认开关重建的数据逐位相同。
- **我补查的评测路线**：在 8 个测试状态上挪了真实骨盆、打乱了 qpos0 的关节、W 和 J，结果是 `max |x or ref change| 0.00e+00`。
- **两点说明**：
  - 参考根本身是站位时用仿真真值锚定的，但到了根自己的坐标里会抵消掉，只剩高度。
  - 仿真里的里程计高度和真值只差 0.02 cm，真机上会差得多。
- **cuRobo 带来的泄漏不是主因**：cuRobo 是按真实骨盆规划的，所以 er 带着真值的痕迹。但只用"真实骨盆 − 根"预测 t 是 0.90，只用 er 是 0.88，常数是 0.93。

**(b) 对手的优劣。**

- **位置重合**：每个开发位置离最近的测试位置都在 2.65 cm 以内。1 cm 阈值以外还剩三个位置，距离也只有 1.03–1.09 cm。样本量匹配的对照：

  ```
  aff from every 6-position subset ... median 0.92, 10-90 % 0.81-1.06; the far subset [1.02]; its rank 359 of 462
  subsets with 0/1/2/3/4/5 positions within 1 cm of a test goal: 1.02 / 1.01 / 0.96 / 0.92 / 0.89 / 0.86
  ```

  aff 从 0.81 掉到 aff_far 的 1.02，大约一半是因为数据少了，一半是因为去掉了重合的位置。
- **更好的常数**（c44：44 个开发局次的均值）：

  ```
  c44 0.93   mR - it -0.00 [-0.13,+0.19] 9/9   it - c_rr -0.09 [-0.17,+0.06] 12/6 p 0.238   aff - it -0.12 [-0.21,-0.05] 14/4 p 0.031
  ```

  aff 对 c_rr 的 −0.21 里，约 0.09 来自常数本身更准，真正来自斜率的只有约 0.12。
- **λ 顶到网格上限，实际不影响**：ScaleBFM 的 x 轴，交叉验证误差到 1000 还在降，但系数已经都在 0.01 以下，这个轴实际上就是常数。SONIC 的 z 轴同理。
- **SONIC 的 aff 吃进了一个摔倒局**：开发局 r2 L3_e4，|t| = 219.7 cm，被纳入了 aff 和 aff_far 的拟合；而模型自己的数据构建会把摔倒行丢掉。

  ```
  dev mean t [-7.52 4.75 1.2], dev median t [-1.53 1.21 0.9]
  aff without dev |t|>20 cm: 3.32   mR - it +0.20 [-0.30,+0.75] 9/10 p 1.000   it - c_rr -0.82 [-1.04,-0.26] 15/4 p 0.019
  Huber aff: 4.13   mR - it -0.03 [-1.02,+0.30] 8/11
  ```

  aff_far 那 25 cm 的偏差，也就是这个摔倒局的 t 除以 9。
- **模型这边也占了便宜**：训练数据是在四个 SIMPLE 任务的布局上采的，`train.sh` 的数据目录里有 TabletopGraspMP。112 个测试局次里，92 个落在训练过的布局上。

**(c) mR 和 t 不在同一点上比：问题比条目写的大，而且是在帮 mR。**

- **怎么在夹爪上比**：手掌三个球在一条直线上，靠它们解不出手转了多少。我加上三根手指根部的球，用 8 个点做刚体拟合，得到模型自己的夹爪预测。用录像验证，拟合出的夹爪偏移和 t 的中位数只差 0.042 cm。
- **模型自己能预测"手掌 − 夹爪"的差**：模型给的均值是 [0.02 −0.83 −0.79]，录像里是 [0.03 −0.83 −0.72]。
- **在夹爪上比的结果**：

  ```
  test at the jaw (t), model's own jaw by rigid fit: miss median 1.22; - dev const +0.27 [+0.11,+0.50] 4/14 p 0.031; - aff +0.44 [+0.14,+0.63] 4/14 p 0.031
  mean(t - mJ) [-0.15 0.52 0.68]  vs  mean(t - mR) [-0.13 -0.31 -0.10]
  test at the palm, mR: 1.15; dev const 1.19; dev aff 1.00 → ties
  ```

- **读法**：
  - 在同一点上比，模型平均偏 0.86 cm，显著比开发常数差。
  - 表里"拿手掌比夹爪"的做法，恰好让手掌和夹爪之间的差抵消了模型的偏差，替 mR 挣了约 0.26 cm。
  - 所以"不赢"的结论不变，只是更站得住。
  - SONIC 上换哪个点比都是打平。

**(d) SONIC 少的那 1 个局次。**

- 是测试局 r0 L0_e1，t = [−77.3, 20.6, −3.6]，机器人摔倒了。离线停偏仍然用全部 106 局。
- 把 |t| > 20 cm 的 5 局去掉，mR − c_rr 是 −0.08 [−1.23, +0.08]，11/8，判定不变。

**其他。**

- **benchmark 选择**：只有 TabletopGrasp 一个任务，位置又挤，这种设定天然对常数有利。结论要限定在这个设定里。
- **事后挑选**：没登记的 aff − c_rr 那一行被拿去做了读法，读错了，见下面第 4 节更正第 4 条。
- **检验力**：
  - ScaleBFM 的区间是 [−0.19, +0.14]，能排除 mR 有超过约 0.2 cm 的优势。
  - SONIC 的区间是 [−1.48, +0.08]，排除不了 1 cm 以上的优势。

**没登记的探查（只作参考）。**

- **ScaleBFM**：
  - 用开发局拟合 t = a + B·mR，测试局上到 0.62，但这个数不稳：只用远处的开发位置拟合是 0.95，在测试集内部按布局留一是 0.96。
  - 在 er + 目标的基础上再加 mR，只多 −0.05，不显著。
  - 所以"没抓到逐局的部分"在 ScaleBFM 上大体成立。
- **SONIC**：

  ```
  leave-one-layout within the set: const 4.09; t ~ mR 3.17 (-0.67 [-1.13,-0.40] 17/2 p 0.001); mR - aff's features -0.46 14/5 p 0.064
  dev calibration B rows [[0.47,..],[..,0.26,..],[..,0.50]]
  ```

  输出里有逐局的信息，只是原始输出的幅度大了 2–4 倍。

### 3. 可能推翻结论的实验

做法：先冻结标定，再到几何会变的新局上，在同一点比。只需要新跑 J，不用挪目标。

1. **冻结**（记 md5）：
   - 原始 mR；
   - mR 自己的夹爪预测（刚体拟合）；
   - 开发局上标定过的 mR；
   - 去掉摔倒局的 c44 和 aff；
   - c_rr。
2. **新局**：TabletopGraspMP，只跑 J 并录像，用新的抖动种子，重复 4 次。
   - **变化组**：至少 24 个新位置，离所有开发位置 5 cm 以上；站位扰动加在桌沿限位之后；目标高度 ±10 cm，骨盆高度也变。
   - **对照组**：同样数量的新位置，几何保持标称。这样能把"位置是新的"和"几何变了"分开。
3. **阴性对照**：把开发局的 mR 打乱以后再标定，应当只到常数的水平。
4. **判据**：照登记的规则，另报按布局中位数的符号检验，两个控制器分开读。
5. **什么算推翻**：
   - 变化组上，原始 mR 或 mR 自己的夹爪预测同时赢 c44 和 aff → 结论被推翻。
   - 只有标定过的 mR 赢 → 改写成"自身模型加几十局标定"。
   - 常数和 aff 在变化组上变差、mR 没变差 → 原来的打平是 benchmark 造成的。

**对条目提出的三种查法的评价**：
- **(a) 按碰桌强弱分组**：便宜，但区分力最弱。录像里没有接触力，`obj_cm` 在所有局上都是 0.04–0.07 cm，没有变化。
- **(b) 去掉桌子重跑**：最干净，但要改两处：
  - 同一批里加一份有桌子的重复作对照；
  - 在夹爪上、用模型自己的夹爪预测来比。
- **(c) 往数据里加带桌面的伸手**：最贵，排在最后。

**更便宜、应当先做的**：把 mR 的误差拆成平均偏差、增益和残差，再和 jref 测试桶里 TabletopGraspMP 布局的同样拆分比。

### 4. 需要更正的说法

1. ScaleBFM 上"aff 显著比 mR 准"：改为边缘显著。约一半来自位置重合，去掉重合以后打平。
2. SONIC 的 −2.46 和意外 4：原因是开发局里的摔倒局。用干净的数据拟合，aff 是 3.32，赢 c_rr，和 mR 打平。
3. 意外 6 说"不用任何 SIMPLE 局就估到了开发常数的水平"：这是取点不一致造成的假象。在同一点上比，模型偏 0.86 cm、停偏 1.22，显著比开发常数差。
4. "aff 抓到的 0.21 cm"：其中约 0.09 来自更准的常数，来自斜率的只有约 0.12。
5. 其他解释 3：结论对，理由要改。取点不一致其实在帮 mR。今后都应当在夹爪上、用模型自己的夹爪预测来比。
6. "学不到逐局的部分"：ScaleBFM 大体成立；SONIC 不成立，问题在幅度和偏差。
7. 其他解释 2 说"信息不够不是主因"：aff 看得到目标位置，mR 看不到。只用 jw 拟合比 aff 差约 0.07，所以不能说已经排除。
8. 陷阱·数据泄漏：要补上两点：训练数据是在 SIMPLE 的布局上采的；这个 benchmark 里没有干净的几何。
9. 陷阱·指标误用：要写明，登记的度量本身就是拿手掌比夹爪；SONIC 的 aff 吃进了一个摔倒局。
10. 输入核对：补两条说明。站位锚点来自仿真真值，但在根坐标里会抵消；仿真里的里程计高度几乎等于真值。
11. 下一步 (a)：录像里没有接触力，只能用几何算穿透量。
