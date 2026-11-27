# selfmodel_fdvel_20261126：速度类输入改用位置差分，自身模型能不能从 LocoManiLab 迁移到 HumanoidArena

- **日期**：2026-11-26。在任何训练之前写好，单独提交。
- **机器 / GPU**：4090 的 1–3 号卡。0 号卡是别人的，不用。keepalive 照常开着。
- **代码**：`selfwm`。训练和评估用 [selfmodel_train3_20261125](selfmodel_train3_20261125.md) 冻结的 `train.py E3_ensemble`、`data.py`、`d3_eval.py`。
  - 唯一的改动是在 `data.py` 加一个开关 `REACHWM_FDVEL=1`：所有速度类输入，改用同一份记录里位置或姿态的差分。
    - 关节速度：改成关节角的差分。
    - 骨盆角速度：改成相邻两行骨盆姿态之间的相对旋转，取对数再除以时间间隔。
    - 骨盆线速度：如果用到了，改成位置差分。
    - 差分的间隔和记录的行间隔相同。一次重置的第一行没有前一行，规则和其他历史特征的缺省处理一致。
  - 网络结构、配方、其余特征都不变：3 万步，batch 1024，5 个成员，`REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1`。
  - LML 和各宿主的记录，用同一个函数算这些特征。
  - 跑之前把代码抄进 `asrun/`，记下 md5。
- **数据**：
  - [selfmodel_data3_20261125](selfmodel_data3_20261125.md) 的 `data/`，只用单手段，划分沿用 data.py。
  - 只用 data3，不加 data3b 和中间点样本，这样能和换仿真检验用的 T3 模型直接比。
- **评估数据**：
  - 换仿真检验 [selfmodel_xfer_20261125](selfmodel_xfer_20261125.md) 已经采好的 6 份记录：SIMPLE 三个、HA 三个；
  - 两条诊断臂；
  - LML 的 val 和 test_iid。
  - 不重新仿真。
- **关联**：xfer 的 HA 复核实验 C；beliefs B53、B54。

## 模型

| 名字 | 数据 | 特征 | seed |
|---|---|---|---|
| T3 | data3 | 原样（读数速度） | 0，train3 已训好，直接拿来用 |
| F | data3 | `REACHWM_FDVEL=1` | 0、1、2 |

三个控制器（ScaleBFM、HEFT、SONIC）各训一套 F。

## 度量

- **主要**：每个 xfer 组合上，F 的预测误差中位数除以 F 自己在 LML val 上的中位数。
  - 算法和 xfer 完全一样：主取法，sx_eval 的筛段，合计、第一次伸手、接着上一次分开报，区间按重置重抽样。
  - 同时报 T3 的倍数，作为对照。
- **LML 上的代价**：F / T3 在 LML val 和 test_iid 上的误差比，按 train3 的方式拆开报。
- **其他**：
  - 模型/直接执行；
  - 集成分歧倍数；
  - F 的输入里还有多少段超出训练分布 6 个标准差（按网络自己的归一化）；
  - 3 个 seed 各自的倍数。

## 假设

1. LML 里分布内的速度读数带的信息很少（xfer 复核：在 LML val 上换掉 dq，误差几乎不变），所以 F 在 LML 上几乎不吃亏。
2. HA 的大倍数主要是越界的速度读数让网络外推失控。F 不读这些量，所以在 HA 上的倍数会大幅下降。
3. 但 HA 的身体和物理确实不一样：更重、踝更弱、armature 为 0、换成了 PhysX。所以 F 在 HA 上仍会高于 1.5 倍，还是要在 HA 里重采。只是重采时可以直接用 F 的特征，模型也不会被读数带偏。

## 预测

- **LML 上 F / T3**：
  - 合计 0.97–1.10；
  - 第一次伸手和接着上一次，都在 0.95–1.15；
  - 3 个 seed 之间相差不到 5 %。
- **HA 上 F 的倍数**（合计）：
  - ScaleBFM 1.6–2.8；
  - HEFT 1.5–2.6；
  - SONIC 1.8–3.2。
  - 理由：在推断时把 dq 和角速度都换掉后是 2.05–2.77 倍（复核数），重训会更好一点。
- **SIMPLE 上 F 的倍数**：和 T3 相差在 ±15 % 以内。SIMPLE 的读数本来是对的。
  - ScaleBFM 约 2.4；
  - SONIC 约 1.8；
  - HEFT 约 1.06。
- **HA 上 F 的模型/直接执行**：0.6–1.0。T3 是 1.69–2.06。
- **HA 上 F 的集成分歧倍数**：1.3–2.5。T3 是 6–21。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| LML 上 F/T3 合计 ≤ 1.05，第一次和接着上一次都 ≤ 1.10 | 速度改用差分在 LML 上没有代价。以后的模型都默认用 F 的特征（之后的 D 配方也一样），跨宿主更稳 |
| LML 上 F/T3 合计在 1.05–1.15 之间 | 两套特征都留着：同一个宿主内用读数速度，跨宿主用差分 |
| LML 上 F/T3 合计 > 1.15 | 读数速度在 LML 里带了有用的信息，不采用 F。在 HA 里核对或修读数（xfer 复核的实验 A、B） |
| HA 上 F 某个组合 ≤ 1.3 倍，区间上限 ≤ 1.5 | 这个控制器的 F 可以直接拿到 HA 做对比，不用在 HA 里重采 |
| HA 上 F 某个组合在 1.3–1.5 倍之间 | 在 HA 里采 200 段做校准 |
| HA 上 F 某个组合 > 1.5 倍 | 这个控制器要在 HA 里重采（用 F 的特征），另行登记 |
| HA 上 F 的倍数没有比 T3 降一半以上 | 假设 2 不成立，大倍数不只来自读数，先查别的越界输入 |
| SIMPLE 上 F 和 T3 相差超过 15 % | 记下来，查是哪一格变了；SIMPLE 的结论以 T3 为准，除非 LML 上也选了 F |

## 预测修订

### 修订 1：实现细节，开训以前（2026-11-26，冻结于 2026-11-25T22:20:38Z）

预测和判据都不改。这一条只写实现细节和一致性核对。

**换了哪些列**
- x0 是 278 维，不是 284 维。284 是再开 `REACHWM_X0_REF` 多出的 6 维，这次和 T3 一样没开。
  - 278 维的组成：S[k0]、S[k0-1]、S[k0-2] 各 65 维；手点 3 维、手腕朝向 6 维、walked、chain；`GOAL_REF` 68 维；`APPROACH` 4 维。
  - 每个 S 行：关节角 0:29，关节速度 29:58，重力方向 58:61，骨盆角速度 61:64（骨盆系），骨盆高度 64。
- 换掉的是三行里的关节速度和骨盆角速度，共 96 列：
  - 关节速度 87 列：x0 的 29:58、94:123、159:188。改成这一行和上一行关节角之差除以 0.1 s。这和 xfer 的 sx_dq "改用 q 的 10 Hz 差分"逐位相同。
  - 骨盆角速度 9 列：x0 的 61:64、126:129、191:194。骨盆姿态 R 用 W 的偏航加 S 的重力方向拼出来（Rz·Ry·Rx，和 models.pelvis_R 是同一个公式），然后算 log(R_上一行ᵀ · R_这一行) / 0.1 s，得到骨盆系的角速度。
- 骨盆线速度：x0 和命令 u 里都没有（S 里只有骨盆高度），不用换。
- C 行（参考手点 3 维，加上距命令的时间）不是 E3 的输入，只有开 `REACHWM_X0_REF` 时才进 x0，里面也没有速度。
- 其余 182 列和 u 原样不动。

**差分间隔**：0.1 s，等于记录的行间隔。三种采集器都是每 5 个 50 Hz 帧记一行。在 data3 的三个控制器和 9 份 xfer 记录上都核对过：命令行之后，C 里"距命令的时间"每行正好加 0.1 s。

**第一行怎么处理**：所有段的 k0 都是 3，包括 data3 的 131,692 段和 9 份 xfer 记录的全部段，也就是命令前记了 3 行（0.3 s）。
- S[k0-2] 是段里的第 1 行，它的差分用第 0 行，两行在同一段里。所以差分永远用不到"一次重置第一行之前"的数据，也就不需要缺省值。
- 这和历史特征 S[k0-1]、S[k0-2] 本身的处理一致：data.py 不填缺省值，直接读采集器记下的命令前几行；命令前的行不全时，采集器根本不写这一段。
- 代码里加了断言：k0 < 3 就报错，不自己补值。

**同一个函数**：`REACHWM_FDVEL=1` 时，data.py 的 derive() 调用 fdvel_x0()。LML 的记录和各宿主的记录都经过 data.load → derive，所以走的是同一个函数。

**一致性核对**（fd_check.py，在冻结副本上跑；check.txt，md5 ae3eae33）
1. **开关关掉**：data3 三个控制器加 9 份 xfer 记录，x0 和 train3 冻结的 data.py（e0cabb83）逐位相同，最大差 0；其余 84–88 个派生数组也全部相同。
2. **开关打开**：只有这 96 列变了，其余 182 列和其他派生数组都逐位相同；关节速度列和 sx_dq 的 "dq from q" 逐位相同。
3. **LML 上差分和读数的对比**。"x0 行"指 x0 用到的三行，算全部段；"所有行"指段里所有有上一行的行，包括伸手过程。

   | 控制器 | 关节速度 r：x0 行 / 所有行 | x0 行中位差（差分 − 读数，rad/s） | x0 行中位 \|差\| / 中位 \|读数\| | 角速度 r：x0 行 / 所有行 | x0 行角速度中位 \|差\| / 中位 \|读数\| |
   |---|---|---|---|---|---|
   | ScaleBFM | 0.995 / 0.897 | +0.00001 | 0.00019 / 0.0034 | 0.992 / 0.886 | 0.00027 / 0.0021 |
   | HEFT | 0.990 / 0.755 | −0.00000 | 0.00016 / 0.0026 | 0.977 / 0.725 | 0.00025 / 0.0018 |
   | SONIC | 0.866 / 0.869 | −0.00000 | 0.00053 / 0.0032 | 0.616 / 0.756 | 0.00068 / 0.0025 |

   - ScaleBFM 和 HEFT 在 x0 行上几乎完全一致。
   - SONIC 在 x0 行上一致得差一些，主要是角速度。补了一个只读诊断（不在冻结代码里，脚本和输出见 `dev/check_std.*`）：按训练集读数的标准差折算，SONIC 的 |差分 − 读数| 中位是 0.08（关节）/ 0.12（角速度）个标准差，p90 是 0.61 / 0.93，p99 是 2.4 / 3.6；ScaleBFM 和 HEFT 的 p99 不到 1。
   - 判断：这不算严重不一致。中位差接近 0，所有行上的相关系数和另两个控制器在同一范围。SONIC 站立时有高频抖动，xfer 复核在 HA 里也看到 SONIC 踝 roll 的真实抖动：瞬时读数带着这部分，0.1 s 的差分把它平均掉了。所以对 SONIC 来说，F 确实少了一点"站立时瞬时抖动"的信息，这正是 LML 上 F/T3 要测的代价。
   - 对照：HA 的 x0 行上，关节速度的 r 只有 −0.004 / 0.074 / 0.084（ScaleBFM / HEFT / SONIC），读数中位数是差分的 6–17 倍。SIMPLE 的 ScaleBFM 和 HEFT 都在 0.96 以上。SONIC·SIMPLE 是 0.68，因为它的读数中位只有 0.0007 rad/s，几乎是噪声。

**冻结**：`loco_outputs/selfmodel_fdvel_20261126/asrun/`。做法照 xfer 的 asrun_ha：从 4090 的 repo（bc426d5e 加上未提交改动）拷出 src 和 selfwm / reachwm / locomanilab 三个目录，assets 和 third_party 链到原处，只把 data.py 换成 Mac 上的 FDVEL 版本。MD5.check 一共 607 个文件，fd_run.sh 开跑前会先校验。

| 文件 | md5 |
|---|---|
| reachwm_20261030/data.py | 174071cb（和 train3 的 e0cabb83 相比只多了 FDVEL 这一段，diff 在 asrun/data_py.diff） |
| train.py / models.py | 8acdaea1 / 0ac27f6b，和 train3 相同 |
| selfwm env.py / wm.py / fk.py | c551fddc / 28cae81d / 721bd136，和 train3、train3b 相同 |
| d3_eval.py | 5ba9f56f，train3 冻结的那份 |
| sx_eval.py | 81f61381，xfer 冻结的那份 |
| fd_check.py / fd_eval_lml.py / fd_xfer_table.py | eab693fd / f8242224 / accd5b60 |
| fd_run.sh / fd_sx1.sh / fd_freeze.sh | c43c88d0 / e4834253 / d08031d1 |

- 解释器用的是 T3 训练和 xfer 评估用的同一个：`.venv_sim`，torch 2.6.0，Python 3.10.12。
- data.py 和 selfmodel_fdvel_20261126 目录在 Mac 上没有提交（Mac 当时在 a927d8cc）。

**运行和评估计划**（fd_run.sh，tmux `fdvel`）
- **训练**：`train.py E3_ensemble --frac 1 --seed s`，数据是 data3 的 `<控制器>/data/all.npz`，环境变量 `REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1`。
  - 3 万步、batch 1024、5 个成员，都是 train.py 的默认值，和 T3 相同。
  - 9 个训练：seed 0、1、2 分别用 4090 的 1、2、3 号卡，每张卡上跑三个控制器。nice 10，每个进程 4 个 CPU 线程。
- **LML 评估**：
  - 每个 F 模型跑 train3 的 d3_eval（val、test_iid）。
  - fd_eval_lml.py 算 F/T3：段和 d3_eval 一样，按 train3 的格子拆，另加第一次伸手 / 接着上一次，以及两者和原地 / 先走路的交叉；区间按重置配对重抽样。
- **xfer 评估**：冻结的 sx_eval 跑 9 份记录（6 个组合，加 ScaleBFM、SONIC、HEFT 三条 `*_simple_lmlref` 诊断臂），T3 和 F 的三个 seed 都跑，主取法和补充取法各一遍。
  - T3 的输出要和每份记录原来的 eval.txt / eval_ended.txt 逐字节相同。
  - fd_xfer_table.py 并排列出 T3 和 F，并算超出 6 个标准差的输入：按网络自己的 nx / nu，报每段至少有一个输入超出的比例，分速度块、其余 x0、u 三块。
- **读数方式**（实现细节，不改判据）：F 的倍数报 3 个 seed 各自的值，以及 3 个 seed 的平均（LML 上 F/T3 = 3 个 seed 中位误差的平均 / T3 的中位误差；xfer 上是 3 个 seed 倍数的平均）。判据按平均读；3 个 seed 落在不同格时，照实写出。

**冻结前跑过的流水线测试**：在 `dev/smoke/` 里用 200 步的 HEFT 模型跑了一遍，评估只用 heft_simple 和 heft_ha 两份记录，只为确认流程能跑通，不是结果。测试里 T3 的输出和原来的 eval.txt 逐字节相同。预测和判据早在 193d246b 就锁定了，这次测试没有改动它们。

## 结果（2026-11-26，执行 agent 的报告）

**冻结和运行**
- 冻结目录：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_fdvel_20261126/asrun`，FROZEN_UTC 2026-11-25T22:20:38Z。md5 表见修订 1，开跑前 MD5.check 共 607 个文件，全部校验通过。
- 运行：4090 卡 1–3，22:25:58Z 开跑，22:47:27Z 跑完。9 个训练每个约 490–540 s，72 次 sx_eval 的退出码全是 0。
- T3 复现：18 份输出（9 份记录 × 2 种筛段）和原来的 eval.txt / eval_ended.txt 逐字节相同（`xfer/t3_reproduce.txt`）。
- 一致性：check.txt（md5 ae3eae33）FDVEL CHECK PASSED。
  - 开关关掉时，x0 和其余数组与 train3 的 data.py 逐位相同。
  - 开关打开时，只有 96 列变了。
  - LML x0 行上，差分和读数的相关系数（关节速度 / 角速度）：ScaleBFM 0.995 / 0.992，HEFT 0.990 / 0.977，SONIC 0.866 / 0.616。
  - SONIC 的差异按训练标准差算，中位数只有 0.08 / 0.12，p99 是 2.4 / 3.6。执行 agent 判断不算严重，继续跑了。这份 SONIC 补充诊断不在冻结代码里，check_std.txt 是事后手抄的控制台输出。

**LML 原文**（`<控制器>/eval_lml_{val,test_iid}.txt`，只截"all"和"reach"三行；表头见 scalebfm test_iid 的第一行）：

```
scalebfm val: 3066 segments from 2256 resets (d3_eval's kept
    all: all                        3066  2256 |  3.077 |  3.102  3.053  3.089 | 1.008 [0.983, 1.035], 0.992 [0.962, 1.016], 1.004 [0.975, 1.030] | 1.002 [0.978, 1.020] |  1.6 % | 0.320, 0.321
    reach: first reach              1580  1580 |  1.701 |  1.737  1.762  1.769 | 1.021 [0.987, 1.065], 1.036 [0.992, 1.087], 1.040 [1.000, 1.093] | 1.032 [1.003, 1.078] |  1.8 % | 0.193, 0.200
    reach: chained                  1486  1486 |  5.209 |  5.257  5.325  5.283 | 1.009 [0.980, 1.035], 1.022 [0.981, 1.056], 1.014 [0.980, 1.059] | 1.015 [0.987, 1.043] |  1.3 % | 0.492, 0.500
scalebfm test_iid: 3094 segments from 2249 resets (d3_eval's
    all: all                        3094  2249 |  3.002 |  3.007  3.084  3.052 | 1.001 [0.976, 1.024], 1.027 [0.984, 1.058], 1.017 [0.980, 1.051] | 1.015 [0.988, 1.037] |  2.6 % | 0.316, 0.321
    reach: first reach              1590  1590 |  1.568 |  1.629  1.699  1.705 | 1.038 [1.001, 1.074], 1.084 [1.041, 1.132], 1.087 [1.039, 1.130] | 1.070 [1.039, 1.101] |  4.7 % | 0.178, 0.191
    reach: chained                  1504  1504 |  5.100 |  5.178  5.040  5.020 | 1.015 [0.994, 1.040], 0.988 [0.963, 1.027], 0.984 [0.956, 1.017] | 0.996 [0.976, 1.021] |  3.2 % | 0.495, 0.493
heft val: 1492 segments from 1206 resets (d3_eval's kept seg
    all: all                        1492  1206 |  3.993 |  4.032  4.041  4.072 | 1.010 [0.968, 1.049], 1.012 [0.964, 1.055], 1.020 [0.969, 1.059] | 1.014 [0.977, 1.046] |  1.0 % | 0.374, 0.379
    reach: first reach              1022  1022 |  2.964 |  2.910  2.919  3.054 | 0.982 [0.939, 1.038], 0.985 [0.941, 1.028], 1.030 [0.980, 1.097] | 0.999 [0.963, 1.042] |  5.0 % | 0.285, 0.284
    reach: chained                   470   470 |  6.989 |  7.576  7.035  7.192 | 1.084 [0.992, 1.119], 1.006 [0.926, 1.072], 1.029 [0.938, 1.086] | 1.040 [0.959, 1.079] |  7.7 % | 0.630, 0.656
heft test_iid: 1576 segments from 1256 resets (d3_eval's kep
    all: all                        1576  1256 |  3.904 |  4.058  4.037  3.880 | 1.039 [0.995, 1.084], 1.034 [0.990, 1.083], 0.994 [0.961, 1.045] | 1.022 [0.990, 1.058] |  4.6 % | 0.371, 0.379
    reach: first reach              1058  1058 |  2.815 |  2.828  2.821  2.798 | 1.005 [0.956, 1.064], 1.002 [0.950, 1.059], 0.994 [0.947, 1.044] | 1.000 [0.958, 1.041] |  1.1 % | 0.268, 0.268
    reach: chained                   518   518 |  6.336 |  7.094  6.680  6.516 | 1.120 [1.044, 1.192], 1.054 [0.969, 1.111], 1.028 [0.943, 1.067] | 1.067 [1.002, 1.107] |  8.9 % | 0.597, 0.637
sonic val: 2935 segments from 2165 resets (d3_eval's kept se
    all: all                        2935  2165 |  4.252 |  4.153  4.171  4.106 | 0.977 [0.946, 1.001], 0.981 [0.957, 1.009], 0.966 [0.938, 0.992] | 0.974 [0.954, 0.997] |  1.6 % | 0.298, 0.290
    reach: first reach              1557  1557 |  2.860 |  2.762  2.795  2.727 | 0.966 [0.930, 0.996], 0.978 [0.934, 1.007], 0.954 [0.914, 0.990] | 0.966 [0.932, 0.994] |  2.5 % | 0.218, 0.211
    reach: chained                  1378  1378 |  6.837 |  6.766  6.902  6.726 | 0.990 [0.956, 1.031], 1.010 [0.977, 1.044], 0.984 [0.951, 1.017] | 0.994 [0.968, 1.023] |  2.6 % | 0.426, 0.423
sonic test_iid: 3005 segments from 2206 resets (d3_eval's ke
    all: all                        3005  2206 |  4.149 |  4.161  4.078  4.144 | 1.003 [0.974, 1.026], 0.983 [0.956, 1.008], 0.999 [0.967, 1.022] | 0.995 [0.972, 1.015] |  2.0 % | 0.296, 0.294
    reach: first reach              1573  1573 |  2.788 |  2.775  2.790  2.772 | 0.996 [0.967, 1.021], 1.001 [0.961, 1.034], 0.994 [0.963, 1.024] | 0.997 [0.970, 1.019] |  0.6 % | 0.219, 0.218
    reach: chained                  1432  1432 |  6.630 |  6.768  6.676  6.699 | 1.021 [0.983, 1.046], 1.007 [0.965, 1.044], 1.010 [0.967, 1.041] | 1.013 [0.978, 1.041] |  1.4 % | 0.443, 0.448
```

**换仿真原文**（`xfer_table.txt`，md5 382e60f8；每份记录只截主取法的 T3 行、F 汇总行和 F 均值行，以及补充取法的 F 汇总行）：

```
scalebfm_simple (controller scalebfm; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm):
    T3   : all 2.375x [2.269, 2.526] (n 624); first reach 4.109x [3.877, 4.503] (n 314); chained 1.458x [1.358, 1.545] (n 310); model/direct 0.963 / 1.069 / 0.828; disagreement 1.94x; OOD 63.1 % | 62.5 % (62.5, 0.2) | 1.0 % | 0.0 %
    F: all F mean 2.355x (seeds 2.309, 2.410, 2.347; spread 4.4 %), F / T3 0.992; first reach F mean 4.000x (seeds 3.978, 4.041, 3.981; spread 1.6 %), F / T3 0.973; chained F mean 1.414x (seeds 1.384, 1.443, 1.416; spread 4.3 %), F / T3 0.970
    F (mean of seeds): model/direct all 0.956; disagreement 2.48x; OOD any 64.7 %; most often beyond 6 sd: [('dq left_shoulder_roll_joint @k0-2', 54.6), ('dq right_shoulder_roll_joint @k0-2', 54.3), ('dq left_shoulder_roll_joint @k0-1', 54.2), ('dq left_shoulder_roll_joint @k0', 53.8), ('dq right_shoulder_yaw_joint @k0-2', 53.8), ('dq right_shoulder_roll_joint @k0-1', 53.5)]
  [ended] F: all F mean 2.393x (seeds 2.339, 2.455, 2.384; spread 4.9 %), F / T3 0.988; first reach F mean 4.049x (seeds 4.065, 4.070, 4.011; spread 1.5 %), F / T3 0.967; chained F mean 1.433x (seeds 1.411, 1.464, 1.426; spread 3.8 %), F / T3 0.971
  LocoManiLab val (3066 kept): OOD any T3 4.08 %, F_s0 4.11 %, F_s1 4.11 %, F_s2 4.11 % | velocity T3 3.42 %, F_s0 3.46 %, F_s1 3.46 %, F_s2 3.46 %
sonic_simple (controller sonic; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic):
    T3   : all 1.785x [1.483, 3.088] (n 56); first reach 2.490x [2.060, 4.430] (n 49); chained 2.093x [1.183, 2.666] (n 7); model/direct 0.925 / 0.895 / 1.007; disagreement 1.58x; OOD 87.5 % | 87.5 % (87.5, 0.0) | 0.0 % | 0.0 %
    F: all F mean 2.005x (seeds 1.867, 2.143, 2.005; spread 14.8 %), F / T3 1.123; first reach F mean 2.830x (seeds 2.634, 3.013, 2.844; spread 14.4 %), F / T3 1.136; chained F mean 2.065x (seeds 2.078, 1.998, 2.118; spread 6.0 %), F / T3 0.987
    F (mean of seeds): model/direct all 1.012; disagreement 1.45x; OOD any 89.3 %; most often beyond 6 sd: [('dq waist_pitch_joint @k0', 87.5), ('dq right_wrist_yaw_joint @k0-1', 87.5), ('dq waist_pitch_joint @k0-2', 1.8)]
  [ended] F: all F mean 3.185x (seeds 3.263, 3.252, 3.038; spread 7.4 %), F / T3 1.110; first reach F mean 4.648x (seeds 4.767, 4.740, 4.437; spread 7.4 %), F / T3 1.136; chained F mean 2.186x (seeds 2.194, 2.239, 2.124; spread 5.4 %), F / T3 0.989
  LocoManiLab val (2935 kept): OOD any T3 9.44 %, F_s0 7.33 %, F_s1 7.33 %, F_s2 7.33 % | velocity T3 8.18 %, F_s0 5.96 %, F_s1 5.96 %, F_s2 5.96 %
heft_simple (controller heft; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft):
    T3   : all 1.059x [0.862, 1.222] (n 258); first reach 1.009x [0.858, 1.133] (n 168); chained 1.075x [0.867, 1.271] (n 90); model/direct 0.440 / 0.342 / 0.660; disagreement 1.03x; OOD 1.6 % | 0.8 % (0.4, 0.4) | 0.8 % | 0.0 %
    F: all F mean 1.072x (seeds 1.053, 1.076, 1.086; spread 3.2 %), F / T3 1.012; first reach F mean 1.022x (seeds 0.993, 1.021, 1.053; spread 6.1 %), F / T3 1.013; chained F mean 1.125x (seeds 0.970, 1.212, 1.193; spread 25.0 %), F / T3 1.046
    F (mean of seeds): model/direct all 0.451; disagreement 1.05x; OOD any 1.9 %; most often beyond 6 sd: [('plan rot c1 y', 0.4), ('plan rot c1 x', 0.4), ('angvel y @k0', 0.4), ('rot0 4', 0.4), ('dq waist_roll_joint @k0-2', 0.4), ('q right_shoulder_yaw_joint @k0-2', 0.4)]
  [ended] F: all F mean 1.247x (seeds 1.258, 1.255, 1.228; spread 2.4 %), F / T3 1.021; first reach F mean 1.153x (seeds 1.153, 1.130, 1.176; spread 4.1 %), F / T3 1.043; chained F mean 1.201x (seeds 1.113, 1.249, 1.240; spread 12.2 %), F / T3 1.002
  LocoManiLab val (1492 kept): OOD any T3 2.95 %, F_s0 3.08 %, F_s1 3.08 %, F_s2 3.08 % | velocity T3 1.94 %, F_s0 2.08 %, F_s1 2.08 %, F_s2 2.08 %
scalebfm_ha (controller scalebfm; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm):
    T3   : all 6.617x [6.346, 6.950] (n 276); first reach 12.755x [12.350, 13.306] (n 179); chained 3.182x [2.853, 3.517] (n 97); model/direct 1.694 / 2.001 / 1.224; disagreement 12.53x; OOD 100.0 % | 100.0 % (100.0, 85.9) | 0.4 % | 0.0 %
    F: all F mean 2.710x (seeds 2.666, 2.719, 2.746; spread 3.0 %), F / T3 0.410; first reach F mean 4.329x (seeds 4.506, 4.127, 4.355; spread 9.2 %), F / T3 0.339; chained F mean 1.920x (seeds 1.905, 2.023, 1.831; spread 10.4 %), F / T3 0.603
    F (mean of seeds): model/direct all 0.695; disagreement 1.00x; OOD any 4.0 %; most often beyond 6 sd: [('dq right_shoulder_yaw_joint @k0', 1.4), ('dq right_shoulder_yaw_joint @k0-1', 1.4), ('dq right_wrist_roll_joint @k0', 1.1), ('dq right_shoulder_roll_joint @k0', 1.1), ('dq left_shoulder_roll_joint @k0-2', 1.1), ('dq right_wrist_roll_joint @k0-1', 1.1)]
  [ended] F: all F mean 3.600x (seeds 3.453, 3.642, 3.706; spread 7.3 %), F / T3 0.503; first reach F mean 6.092x (seeds 6.067, 6.073, 6.137; spread 1.2 %), F / T3 0.432; chained F mean 2.142x (seeds 2.073, 2.165, 2.188; spread 5.5 %), F / T3 0.630
  LocoManiLab val (3066 kept): OOD any T3 4.08 %, F_s0 4.11 %, F_s1 4.11 %, F_s2 4.11 % | velocity T3 3.42 %, F_s0 3.46 %, F_s1 3.46 %, F_s2 3.46 %
heft_ha (controller heft; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft):
    T3   : all 5.493x [5.258, 5.677] (n 176); first reach 7.164x [6.869, 7.428] (n 128); chained 5.326x [3.709, 5.974] (n 48); model/direct 2.064 / 2.080 / 2.956; disagreement 20.81x; OOD 100.0 % | 100.0 % (100.0, 89.2) | 1.7 % | 0.0 %
    F: all F mean 2.151x (seeds 2.038, 2.150, 2.266; spread 11.2 %), F / T3 0.392; first reach F mean 2.833x (seeds 2.669, 2.868, 2.963; spread 11.0 %), F / T3 0.395; chained F mean 1.454x (seeds 1.372, 1.485, 1.505; spread 9.7 %), F / T3 0.273
    F (mean of seeds): model/direct all 0.820; disagreement 1.95x; OOD any 83.5 %; most often beyond 6 sd: [('dq left_ankle_pitch_joint @k0', 75.0), ('dq right_elbow_joint @k0', 73.9), ('dq right_wrist_pitch_joint @k0-2', 73.9), ('dq left_ankle_pitch_joint @k0-1', 73.9), ('dq right_wrist_pitch_joint @k0-1', 73.9), ('dq left_hip_roll_joint @k0-2', 73.3)]
  [ended] F: all F mean 2.895x (seeds 2.816, 2.815, 3.054; spread 8.5 %), F / T3 0.521; first reach F mean 3.875x (seeds 3.795, 3.880, 3.951; spread 4.1 %), F / T3 0.538; chained F mean 1.721x (seeds 1.573, 1.634, 1.956; spread 24.4 %), F / T3 0.316
  LocoManiLab val (1492 kept): OOD any T3 2.95 %, F_s0 3.08 %, F_s1 3.08 %, F_s2 3.08 % | velocity T3 1.94 %, F_s0 2.08 %, F_s1 2.08 %, F_s2 2.08 %
sonic_ha (controller sonic; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic):
    T3   : all 5.227x [5.072, 5.475] (n 155); first reach 7.543x [7.012, 7.915] (n 119); chained 5.310x [3.885, 7.492] (n 36); model/direct 1.807 / 1.771 / 2.782; disagreement 6.07x; OOD 100.0 % | 100.0 % (100.0, 96.8) | 0.0 % | 0.0 %
    F: all F mean 3.061x (seeds 3.277, 3.055, 2.851; spread 14.9 %), F / T3 0.586; first reach F mean 4.638x (seeds 4.948, 4.672, 4.294; spread 15.2 %), F / T3 0.615; chained F mean 1.738x (seeds 1.698, 1.770, 1.745; spread 4.3 %), F / T3 0.327
    F (mean of seeds): model/direct all 1.031; disagreement 2.17x; OOD any 91.6 %; most often beyond 6 sd: [('angvel z @k0', 80.0), ('dq right_hip_yaw_joint @k0', 79.4), ('dq right_wrist_roll_joint @k0', 78.7), ('angvel z @k0-2', 78.7), ('dq left_shoulder_yaw_joint @k0', 78.1), ('dq right_shoulder_yaw_joint @k0', 77.4)]
  [ended] F: all F mean 3.807x (seeds 3.742, 4.024, 3.655; spread 10.1 %), F / T3 0.663; first reach F mean 5.635x (seeds 5.351, 6.143, 5.412; spread 14.8 %), F / T3 0.716; chained F mean 2.418x (seeds 2.533, 2.381, 2.340; spread 8.2 %), F / T3 0.297
  LocoManiLab val (2935 kept): OOD any T3 9.44 %, F_s0 7.33 %, F_s1 7.33 %, F_s2 7.33 % | velocity T3 8.18 %, F_s0 5.96 %, F_s1 5.96 %, F_s2 5.96 %
scalebfm_simple_lmlref (controller scalebfm; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm):
    T3   : all 2.329x [2.240, 2.451] (n 619); first reach 3.983x [3.662, 4.309] (n 315); chained 1.434x [1.360, 1.504] (n 304); model/direct 0.937 / 0.981 / 0.846; disagreement 1.19x; OOD 12.1 % | 11.3 % (11.3, 0.2) | 1.1 % | 0.0 %
    F: all F mean 2.313x (seeds 2.253, 2.342, 2.343; spread 4.0 %), F / T3 0.993; first reach F mean 3.779x (seeds 3.759, 3.734, 3.843; spread 2.9 %), F / T3 0.949; chained F mean 1.447x (seeds 1.396, 1.485, 1.458; spread 6.4 %), F / T3 1.009
    F (mean of seeds): model/direct all 0.932; disagreement 1.33x; OOD any 64.0 %; most often beyond 6 sd: [('dq left_shoulder_yaw_joint @k0-2', 53.5), ('dq left_elbow_joint @k0-2', 4.7), ('dq left_elbow_joint @k0-1', 3.7), ('dq right_shoulder_pitch_joint @k0-2', 3.7), ('dq right_shoulder_pitch_joint @k0-1', 3.2), ('dq right_shoulder_pitch_joint @k0', 3.1)]
  [ended] F: all F mean 2.343x (seeds 2.303, 2.363, 2.363; spread 2.6 %), F / T3 0.991; first reach F mean 3.820x (seeds 3.786, 3.785, 3.889; spread 2.8 %), F / T3 0.942; chained F mean 1.485x (seeds 1.442, 1.503, 1.510; spread 4.7 %), F / T3 1.017
  LocoManiLab val (3066 kept): OOD any T3 4.08 %, F_s0 4.11 %, F_s1 4.11 %, F_s2 4.11 % | velocity T3 3.42 %, F_s0 3.46 %, F_s1 3.46 %, F_s2 3.46 %
sonic_simple_lmlref (controller sonic; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic):
    T3   : all 2.455x [1.561, 3.015] (n 37); first reach 3.638x [2.256, 4.384] (n 34); chained 1.923x [1.407, 2.146] (n 3); model/direct 0.956 / 0.955 / 0.835; disagreement 1.12x; OOD 2.7 % | 2.7 % (2.7, 0.0) | 0.0 % | 0.0 %
    F: all F mean 2.644x (seeds 2.544, 2.674, 2.715; spread 6.7 %), F / T3 1.077; first reach F mean 3.913x (seeds 3.739, 3.806, 4.195; spread 12.2 %), F / T3 1.076; chained F mean 1.894x (seeds 2.237, 1.915, 1.529; spread 46.2 %), F / T3 0.985
    F (mean of seeds): model/direct all 1.003; disagreement 1.03x; OOD any 5.4 %; most often beyond 6 sd: [('dq left_knee_joint @k0', 2.7), ('dq right_knee_joint @k0', 2.7), ('dq left_wrist_pitch_joint @k0', 2.7), ('dq right_wrist_pitch_joint @k0', 2.7), ('dq left_knee_joint @k0-2', 2.7), ('dq left_knee_joint @k0-1', 2.7)]
  [ended] F: all F mean 3.406x (seeds 3.436, 3.337, 3.444; spread 3.2 %), F / T3 1.043; first reach F mean 4.921x (seeds 4.903, 4.837, 5.024; spread 3.9 %), F / T3 1.037; chained F mean 2.882x (seeds 2.971, 2.747, 2.929; spread 8.2 %), F / T3 1.039
  LocoManiLab val (2935 kept): OOD any T3 9.44 %, F_s0 7.33 %, F_s1 7.33 %, F_s2 7.33 % | velocity T3 8.18 %, F_s0 5.96 %, F_s1 5.96 %, F_s2 5.96 %
heft_simple_lmlref (controller heft; LocoManiLab /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft):
    T3   : all 1.170x [1.052, 1.319] (n 321); first reach 1.283x [1.151, 1.513] (n 206); chained 0.987x [0.793, 1.249] (n 115); model/direct 0.442 / 0.382 / 0.548; disagreement 1.09x; OOD 0.9 % | 0.3 % (0.3, 0.0) | 0.6 % | 0.0 %
    F: all F mean 1.211x (seeds 1.264, 1.189, 1.180; spread 7.1 %), F / T3 1.036; first reach F mean 1.359x (seeds 1.441, 1.349, 1.287; spread 12.0 %), F / T3 1.059; chained F mean 0.968x (seeds 0.917, 1.028, 0.958; spread 12.1 %), F / T3 0.980
    F (mean of seeds): model/direct all 0.464; disagreement 1.14x; OOD any 1.2 %; most often beyond 6 sd: [('plan rot c1 y', 0.3), ('plan rot c1 x', 0.3), ('rot0 4', 0.3), ('dq waist_roll_joint @k0-2', 0.3), ('q right_shoulder_yaw_joint @k0-2', 0.3), ('dq waist_yaw_joint @k0-2', 0.3)]
  [ended] F: all F mean 1.349x (seeds 1.369, 1.344, 1.334; spread 2.6 %), F / T3 1.041; first reach F mean 1.490x (seeds 1.569, 1.499, 1.404; spread 11.7 %), F / T3 1.002; chained F mean 1.027x (seeds 1.007, 1.045, 1.030; spread 3.8 %), F / T3 1.018
  LocoManiLab val (1492 kept): OOD any T3 2.95 %, F_s0 3.08 %, F_s1 3.08 %, F_s2 3.08 % | velocity T3 1.94 %, F_s0 2.08 %, F_s1 2.08 %, F_s2 2.08 %
```


**LML 上的 F / T3**（3 个 seed 平均，[95 %]）

| 控制器 · 划分 | 合计 | 第一次伸手 | 接着上一次 | 合计 seed 差 |
|---|---|---|---|---|
| ScaleBFM val | 1.002 [0.978, 1.020] | 1.032 [1.003, 1.078] | 1.015 | 1.6 % |
| ScaleBFM test_iid | 1.015 [0.988, 1.037] | 1.070 [1.039, 1.101] | 0.996 | 2.6 % |
| HEFT val | 1.014 [0.977, 1.046] | 0.999 | 1.040 | 1.0 % |
| HEFT test_iid | 1.022 [0.990, 1.058] | 1.000 | 1.067 [1.002, 1.107] | 4.6 % |
| SONIC val | 0.974 [0.954, 0.997] | 0.966 | 0.994 | 1.6 % |
| SONIC test_iid | 0.995 [0.972, 1.015] | 0.997 | 1.013 | 2.0 % |

**换仿真记录**（主取法，倍数 = 误差 / 自己的 LML val）

| 组合 | T3 | F 平均（三个 seed） | F/T3 | 模型/直接 T3 → F | 集成分歧 T3 → F | 超 6 SD 的段 T3 → F |
|---|---|---|---|---|---|---|
| ScaleBFM·HA | 6.617 | 2.710（2.666、2.719、2.746） | 0.410 | 1.694 → 0.695 | 12.53 → 1.00 | 100 % → 4.0 % |
| HEFT·HA | 5.493 | 2.151（2.038、2.150、2.266） | 0.392 | 2.064 → 0.820 | 20.81 → 1.95 | 100 % → 83.5 % |
| SONIC·HA | 5.227 | 3.061（3.277、3.055、2.851） | 0.586 | 1.807 → 1.031 | 6.07 → 2.17 | 100 % → 91.6 % |
| ScaleBFM·SIMPLE | 2.375 | 2.355 | 0.992 | 0.963 → 0.956 | 1.94 → 2.48 | 63.1 % → 64.7 % |
| SONIC·SIMPLE（56 段） | 1.785 | 2.005（seed 差 14.8 %） | 1.123 | 0.925 → 1.012 | 1.58 → 1.45 | 87.5 % → 89.3 % |
| HEFT·SIMPLE | 1.059 | 1.072 | 1.012 | 0.440 → 0.451 | 1.03 → 1.05 | 1.6 % → 1.9 % |

- HA 上，每个 seed 的区间下限都高于 1.5，最低是 HEFT s1 的 1.848。
- 补充取法下，HA 的 F/T3 是 ScaleBFM 0.503、HEFT 0.521、SONIC 0.663。

**对照判据**

| 判据 | 结果 | 结论 |
|---|---|---|
| LML 上合计 ≤ 1.05，第一次和接着上一次都 ≤ 1.10 | 三个控制器、两个划分都满足：合计最大 1.022，第一次最大 1.070，接着上一次最大 1.067。有两格的区间上限略过 1.10（1.101、1.107），判据按点值读 | 第 1 格：以后的模型默认用 F 的特征 |
| HA 某组合 > 1.5 倍 | 三个都是：2.71 / 2.15 / 3.06 | 三个控制器都要在 HA 里用 F 的特征重采，另行登记 |
| HA 上 F 没比 T3 降一半以上 | 主取法下 ScaleBFM 0.41、HEFT 0.39，降过一半；SONIC 0.586 没降过。补充取法下三个都没降过一半（0.50 / 0.52 / 0.66） | 对 SONIC 触发"假设 2 不成立，先查别的越界输入"；ScaleBFM 和 HEFT 在主取法下成立，补充取法下压线 |
| SIMPLE 上 F 和 T3 相差超过 15 % | 没有（0.992 / 1.123 / 1.012） | 不触发 |

**预测对照**
- LML 上 F/T3 合计在 0.97–1.10：6/6 中。第一次和接着上一次在 0.95–1.15：12/12 中。
- seed 之间相差不到 5 %：合计格和第一次伸手格全中；接着上一次那格 HEFT 没中（val 7.7 %，test_iid 8.9 %）。
- HA 上 F 的倍数：三个都落在预测区间里。单看 seed，SONIC s0 是 3.277，超出区间。
- SIMPLE 上 F 和 T3 相差在 ±15 % 以内：3/3 中。
- HA 上 F 的模型/直接在 0.6–1.0：ScaleBFM 和 HEFT 中，SONIC 是 1.031，没中。
- HA 上 F 的集成分歧在 1.3–2.5 倍：HEFT 和 SONIC 中；ScaleBFM 是 1.00，比预测还低。

**和计划不一样的地方**
1. 解释器用的是 `.venv_sim`（torch 2.6.0），和 T3 训练、xfer 评估用的一样，修订 1 已写明。
2. x0 是 278 维。284 维是开了 X0_REF 才有的，这次和 T3 一样没开。
3. 诊断臂跑了三条，条目写的是"两条"。
4. SONIC 的补充诊断见上文"一致性"一条。
5. Mac 上 data.py 的 FDVEL 改动和 `scripts/experiments/selfmodel_fdvel_20261126/` 还没提交，冻结副本在 asrun 里。
6. 冻结前在 `dev/smoke` 跑过一次流水线测试，用的是 200 步的 HEFT 模型。
7. LML 上的集成分歧倍数和 test_iid 上的越界比例没有算。

## 意外

1. **HEFT 和 SONIC 在 HA 里，差分速度本身也大面积越界**，分别有 83.5 % 和 91.6 % 的段。HEFT 越界最多的是左踝 pitch 的 dq（75 %），SONIC 是骨盆偏航角速度（80 %）。这两个控制器在命令前那几行，身体确实比在 LML 里动得多，不只是读数有问题。这和 SONIC 的倍数没降到一半对得上。ScaleBFM 的 F 只剩 4.0 % 越界。
2. **ScaleBFM 的 SIMPLE 诊断臂记录里，F 的越界反而变多**，从 12.1 % 涨到 64.0 %，集中在肩关节 k0-2 行的 dq。这说明这份记录在段的前两行之间关节角有跳变，但误差没受影响（0.993）。还没查。
3. **ScaleBFM 的 F 在 HA 上集成分歧是 1.00 倍**，和 val 完全一样，倍数却还有 2.7。这时模型自己"不知道"它错了，所以集成分歧不能当作跨宿主的安全检查。

## 结论与下一步（待复核）

- **默认改用差分速度。** 三个控制器在 LML 上几乎没有代价（合计 0.97–1.02），落在判据第 1 格。唯一稳定的代价是 ScaleBFM 第一次伸手 3–7 %，区间不含 1。
- **HA 的大倍数有一大半来自速度读数。** ScaleBFM 和 HEFT 降掉约六成（6.6 → 2.7，5.5 → 2.2），SONIC 降掉约四成（5.2 → 3.1）。模型修正也从"比不修正还差"变成能消掉一些误差：ScaleBFM 0.70、HEFT 0.82；SONIC 是 1.03，修不动。
- **但三个控制器在 HA 上都还在 2 倍以上，要在 HA 里重采。** HEFT 和 SONIC 在 HA 里命令前身体确实动得更多，起点越界不只是读数造成的。
- **SIMPLE 不受影响**，原来的判断不变：HEFT 能直接用，ScaleBFM 和 SONIC 要重采。
- **集成分歧不能单独当作上线检查。** ScaleBFM·HA 上分歧和 val 一样，倍数却是 2.7。
- **证据强度**：每个控制器 3 个 seed；HA 每个组合 155–276 段，来自 1 次运行；只用了 data3，没加 data3b 和中间点。
- **其他可能的解释**：HA 上剩下的倍数可能部分来自没清干净的越界输入（HEFT、SONIC 仍有八九成的段越界），不一定全是身体差异。在 HA 里重采可以同时回答这个问题。
- **四类陷阱**：
  - benchmark 选择：评估记录都是 xfer 事先采好的，没有挑。
  - 数据泄漏：F 只在 data3 上训，xfer 记录没进过训练。
  - 指标误用：主指标和 xfer 相同，没换。
  - 事后挑选：3 个 seed 和两种取法都报了；判据按点值读，有两格区间上限略过 1.10，已写明。

## 独立复核

（2026-11-26，独立 agent，只读；以下是要点，原意照录）

**总体**：数字都对得上，差分特征的实现也没有问题。第 1、4 条论断可以原样成立，第 2、3 条要改写。理由有三：
- T3 只有一个 seed，它在 HA 上的倍数很不稳，五个集成成员之间能差 2–4 倍。
- 每份 xfer 记录里，所有第一次伸手都从同一个起始状态出发：HEFT·HA 的 128 段只有 1 个起点，SONIC·HA 的 119 段也只有 1 个。LML 里每段的起点都不同。
- 把 F 的速度输入"钉"在训练均值上：HEFT·HA 从 2.15 倍降到约 1.48 倍，SONIC·HA 基本不变。这和作者"SONIC 是因为身体动得更多"的解释方向相反。

**1. 数字核对**
- 用 scipy 独立实现了差分特征，和 data.py 的结果比：LML 上最大差 3e-11，HA 上为 0。
- 下面几项都一致：
  - ScaleBFM·HA 的 F 合计 2.710（F/T3 0.410），补充取法 3.600；
  - LML test_iid 上 ScaleBFM 第一次伸手 F/T3 1.070 [1.037, 1.100]；
  - HEFT·HA 越界 83.5 %；
  - LML 表的 18 格；
  - md5；
  - T3 复现逐字节相同。
- 时间线：登记 21:58Z，修订 1 提交 22:25:39Z，开跑 22:25:58Z。预测和判据一行没动。
- 笔误：训练时长应为 484–539 s。

**2. 指定的检查点**
- **差分实现没有问题。**
  - 第 r 行的差分，和第 r−1、r 两行读数的平均，相关系数 0.999–1.000。也就是说，差分代表前 0.1 s 的平均速度，所有宿主都一样。
  - 坐标系和符号核对过：改成世界系或者反号，相关都会明显变差。
  - k0 都是 3，所以 S[k0-2] 用的是同一段的第 0 行。
  - SONIC 的相关只有 0.87 / 0.62，原因在读数本身：读数相邻两行的相关只有 0.71 / 0.41，但把它积分 0.3 s 后，和实际位移的相关是 0.973。也就是读数没有偏，只是带着高频抖动，差分把抖动平均掉了。
- **"HEFT 和 SONIC 命令前身体确实动得更多"：方向对，但要说清楚是什么、有多大。**
  - 不是记录间隔造成的：HA 每行也是 0.1 s，伸手时手臂"差分对读数"的斜率两边几乎一样。
  - HEFT：躯干点、骨盆姿态、腿关节三路数据都显示一路单向的缓慢漂移。命令前 0.3 s 的 p90，HA 对 LML 分别是 0.52 对 0.23 cm、0.29 对 0.12°、0.010 对 0.003 rad。骨盆速度约 1.25 cm/s，低于 2 cm/s 的停稳阈值，所以停稳规则放它过去了。
  - SONIC：净位移并不比 LML 大，是在来回抖。相邻差分有 58 % 换号（LML 12 %），走过的路程是净位移的 3.5 倍。
  - 幅度都在 1 cm、1° 以内。之所以超出 6 个标准差，是因为 LML 里站得特别稳。
  - 第一次伸手只有一个起点。接着上一次的段，越界比例 HEFT 是 33 %，SONIC 是 64 %。
  - "钉住"测试（事后做的，只看敏感度）：
    - HEFT·HA 2.04–2.27 → 1.47–1.50，第一次伸手的误差从 7.8–9.1 cm 降到 4.6–5.3 cm；
    - SONIC·HA 基本不变（3.05–3.16）；
    - ScaleBFM 也不变。

    所以 HEFT 剩下的倍数里约三分之一，是网络在那一个越界起点上外推出来的错。SONIC 剩下的约 3 倍和速度输入无关，它的其余输入也没有越界。
- **"以后默认用 F"的代价。**
  - 判据按点值读是合理的：LML 这一行没有写区间。
  - 但区间只算了重置抽样，T3 只有 seed 0，训练本身的随机性没算进去。参照：SONIC val 上 F 反而比 T3 好（0.974，区间不含 1）；F 的 seed 之间第一次伸手差 4.7 %。
  - 绝对值很小，约 0.08 cm。LML 的五个留出带上，F/T3 合计 0.975–1.008，ScaleBFM 第一次伸手 1.02–1.04，方向一致。
  - 应改写为："可能有约 1 mm 的小代价，分不清是不是训练随机性。"
- **判据第 3 行（降没降过一半）。**
  - 按主取法读。对 F 的 seed 很稳：ScaleBFM 0.40–0.42，HEFT 0.37–0.41，SONIC 0.55–0.63。
  - 但对 T3 的 seed 不稳：T3 五个成员在 HA 上 ScaleBFM 6.6–14.1、HEFT 5.8–22、SONIC 3.9–8.3。SONIC 有一个成员到了 8.26，可能翻格。
  - 所以"ScaleBFM 和 HEFT 降过一半"大体可信，"SONIC 没降过一半"不稳。
- **ScaleBFM 诊断臂的越界从 12 % 涨到 64 %：不是 bug，也不是跳变。**
  - SIMPLE 里第一次伸手时，左肩 yaw 的速度还在平滑衰减，差分取的是早 0.05 s 的平均，值是 0.054，刚好越过 6 个标准差（0.050）。
  - 315 段第一次伸手共用这一个起点，所以一起越界。作者"记录里有跳变"的解释不对。

**3. 可能推翻论断的实验**
- **补训 T3 的 seed 1、2**：三个控制器 × 2 个 seed，用三个 T3 seed 的平均来算 F/T3。约 1 GPU 小时，要另行登记。
  - 推翻条件：ScaleBFM 或 HEFT 的 F/T3 > 0.5；SONIC 的 F/T3 < 0.5；或者 ScaleBFM 第一次伸手的 F/T3 区间含 1。
- **在 HA 里让起点多样化，并加严停稳**：重置时加上和 LML 相当的起始扰动；下命令前要求差分速度落在 LML 的 p99 以内并保持 0.3 s，最多等 6 s。HEFT 和 SONIC 各采 150 次重置。
  - 越界降到 10 % 以下且 F ≤ 1.5 倍，就推翻"要在 HA 里重采"。
  - 6 s 内停不下来，第 3 条反而更站得住。

**4. 判定**

| 论断 | 判定 | 理由 |
|---|---|---|
| 1. LML 上 F 几乎没有代价，以后默认用 F 的特征 | 支持，措辞要改 | 落在第 1 格，留出带上也成立；"稳定代价"没有算 T3 的 seed 随机性 |
| 2. HA 的大倍数有一大半来自速度读数 | ScaleBFM 和 HEFT 支持；SONIC 证据不足 | T3 只有一个 seed，它的成员之间差 2–4 倍；"一大半"不适用于 SONIC |
| 3. 要在 HA 里重采；HEFT 和 SONIC 命令前身体动得更多 | "要重采"支持；"身体动得更多能解释剩下的倍数"证据不足 | 身体在动，但只在 1 cm、1° 以内，而且只有一个起点；钉住测试和作者对 SONIC 的解释方向相反 |
| 4. 集成分歧不能单独当跨宿主的安全检查 | 支持 | ScaleBFM·HA 分歧 1.00 倍，误差却是 2.7 倍；HA 上分歧和误差的等级相关只有 0.36–0.41，val 上是 0.79 |

## 复核后的更正（2026-11-26）

1. **代价的措辞。** "ScaleBFM 第一次伸手有 3–7 % 的稳定代价，区间不含 1"，改为"可能有约 1 mm 的小代价。T3 只有一个 seed，分不清是不是训练随机性。"以后默认用 F 的特征这一条不变。
2. **SONIC。** 判据已经对 SONIC 触发"假设 2 不成立"。查下来，它剩下的约 3 倍和速度输入无关：钉住速度输入后结果不变，其余输入也不越界。"身体动得更多"解释不了 SONIC。原因待查。
3. **HEFT。** 剩下的 2.15 倍里，约三分之一是网络在一个越界起点上外推出来的错（钉住后是 1.48）。HEFT 在 HA 里的身体漂移很小（1 cm、1° 以内），但单向，而且停稳规则放它过去了。
4. **意外 2 的原因**：是单一起点刚好压在阈值边上，不是记录里有跳变。
5. **xfer 记录的设计缺陷**：每份记录里，第一次伸手全部从同一个起始状态出发，因为仿真是确定的、重置又没有扰动。所以"第一次伸手"那一格，实际只检验了一个起点上的不同命令。这一条也补进了 xfer 条目。
6. **"降六成 / 降四成"这种分法不稳**，因为 T3 在 HA 上的成员间差异很大。以后的跨宿主对比，基准模型也要训 3 个 seed。
