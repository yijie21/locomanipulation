# selfmodel_simpledata_20261126：在 SIMPLE 里给 ScaleBFM 重采数据、训练 SIMPLE 自己的自身模型

- **日期**：2026-11-26。在任何正式采集之前写好，单独提交。
- **用户的决定**（11-26）："SIMPLE 用方案一，HA 也重采，检验先不做"。方案一是保留 SIMPLE 现在的接法，在 SIMPLE 里重采数据、重训模型。
- **机器**：采集只用 CPU。
  - 4090b：32 个进程，nice 10。用开发阶段的启动办法 `py4090b.py`，它按 vePFS 的真实路径重建 sys.path；再加 `LOCOMANI_WORKSPACE`、`SONIC_POLICY` 两个环境变量和 `repo_b`。开发阶段已验证，这样跑出的结果和 4090 逐位相同。4090b 的环境不改。
  - 4090：8 个进程，nice 10。4090 上同时还在跑 lmlcmp3，用 35 个进程。
  - 训练用 GPU，4090 的 1–3 号卡或 4090b 的两张卡，开跑前看 nvidia-smi。
- **代码**：`selfwm`。采集器是 `scripts/experiments/selfmodel_simplecollect_20261126/sc_collect.py`，它包了 xfer 的 SIMPLE 宿主，抽样、静止判据、分段、检查都直接调 d3_collect。开发、冒烟和吞吐测试见执行 agent 的开发报告，要点抄在"预测修订"里。训练和评估沿用 train3b 的 t3b_* 和 fdvel 的改动。正式采集前把代码抄进 `asrun/`，记下 md5。
- **只做 ScaleBFM。** SONIC 在 SIMPLE 现在的接法下采不出合格数据：
  - 有些姿态下手持续摆动 2–11 cm/s；
  - 先转身再走时摔 3/10；
  - 单手可用段约每 20 次重置才有 1 段。

  这件事要用户定，见"开发阶段的发现"。HEFT 在 SIMPLE 里直接用 LML 的模型，xfer 里是 1.06 倍。
- **关联**：
  - [selfmodel_xfer_20261125](selfmodel_xfer_20261125.md)：ScaleBFM 在 SIMPLE 上是 2.38 倍，结论是重采。
  - [selfmodel_fdvel_20261126](selfmodel_fdvel_20261126.md)：模型输入改用差分速度，重置要加扰动。
  - [selfmodel_train3b_20261125](selfmodel_train3b_20261125.md)：训练配方 D。

## 开发阶段的发现（看过冒烟，没看过正式数据）

- **起始扰动**：出生时 29 个关节加高斯噪声，0.007 rad。这时原地第一下伸手的起点分散程度和 LML data3 相当：单关节标准差中位数 0.35° 对 0.34°，骨盆高度标准差 0.05 cm 对 0.03 cm。冒烟用的是 0.02 rad，偏大。
- **先走后伸的起点**：照 d3 的规则，先转 ±90°、再走、再转。走完以后两个髋偏航扭到 30–70°。SIMPLE 的任务走路时不转身，扭转不超过 14°。所以照 d3 的规则，大部分先走后伸的起点是任务里见不到的。
- **SIMPLE 的 ScaleBFM 下第一条命令时，关节还在动**，0.1 s 差分速度 0.28 rad/s；LML 是 0.02。xfer 里不加扰动时也一样，和扰动无关。
- **吞吐**：
  - 4090b 36 个进程时，ScaleBFM data3 每小时约 11.5k 次重置、27k 段，其中可用单手段约 17k。
  - data3b 每小时约 10k 次重置、50k 段。

## 假设

1. ScaleBFM 在 SIMPLE 上 2.38 倍的预测误差，是因为 SIMPLE 的接法和 LML 不同：mode 4 加腿部里程计加踝重钉。换成在 SIMPLE 里采的数据，D 配方训出来的模型在 SIMPLE 上的误差，应该回到 LML 模型在 LML 上的水平。
2. 加了起始扰动以后，第一下伸手的起点不再只有一个。模型在真实 SIMPLE 局的起点上，误差和自己的 val 相当。
3. 先走后伸的重置里混入一半 SIMPLE 任务式的走法（不转身），能覆盖任务里走到桌前的起点。

## 做法

### 采集
- **data3 式**：30k 次重置。重置编号段、抽样、伸手表、静止判据、段布局都照 d3_collect。只改两处：
  1. 起始扰动 0.007 rad；
  2. 先走后伸的重置里，一半照 d3 的规则（转、走、转），另一半用"任务式"走法：不转身，位移落在 APPROACH_BOX 里。哪一半由重置编号的哈希决定，和其他抽样无关。任务式走法要在开发阶段的基础上实现，并通过冒烟。
- **data3b 式**：15k 次重置，四种伸手照 data3b，起始扰动同上。
- 冒烟目录（50M/60M 编号的前 20 次重置，扰动 0.02 rad）不并进正式数据。
- 检查：d3 原有检查加 SIMPLE 检查（FK 重算手点、C 行、直接误差重算、ik_err、参考特征、data.py 读入、划分）。全部通过才能训练。

### 训练
- **配方 D**：data3 + data3b + 中间点样本。
- **标志**：`REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1`，不开 X0_REF。
- **模型和训练设置**：E3_ensemble，步数和评估网格照 train3b 的 D，按 val 选最佳 checkpoint。
- **seed**：0 为主。
- **学习曲线**：frac 0.3 和 1.0，按重置嵌套，都用 seed 0。
- **划分**：照 data.py 按重置哈希划分，0–7 训练、8 验证、9 测试；bands 和 edge 单独留出。

### 评估（模型训完当天做完）
1. **SIMPLE 自己的 val、test_iid、bands、edge**：只统计没摔、末端停稳的段，第一下和连续伸手分开报。
2. **xfer 的 SIMPLE ScaleBFM 记录**：400 次重置，没加扰动，作额外测试集。报这次模型的误差除以 SIMPLE val 的倍数，以及 LML 模型 F 在同一批记录上的倍数（fdvel 里已有）。
3. **真实 SIMPLE 局**：
   - 用主指标的四个抓取任务的感知栏，ScaleBFM，和主指标的 A 栏同一批局。
   - 记录器挂在 `Motor.tick` 上逐拍记；预抓取和进入两次 `Motor.reach`，通过 `_blend` 截下规划，写成 data3 格式。
   - 标签有两种：
     - **主标签**：在每条命令前，按 d3 规则等身体静止（最多 3 s）以后再取末端。这是单独跑的测量局，不替换主指标；
     - **补充标签**：任务原本的节奏，在下一条命令的时刻取末端。
4. **起始状态在不在训练范围里**：对每个真实段算 x0，查三项：
   - 关节、骨盆、手点是否落在训练集的 p0.5–p99.5 之内；
   - 在标准化的 x0 空间里，到训练集的最近邻距离，是否超过训练集内部最近邻距离的 p99；
   - 命令是否在伸手表的范围内。

   误差按"在范围内 / 不在"分开报，原地伸手和先走后伸也分开报。

## 预测

**采集**
- data3：摔倒或丢失 0–1 %；等不到静止被跳过的伸手 1–5 %；末端停稳 ≥ 95 %。
- data3b：摔倒或丢失 0–1 %；末端停稳 ≥ 97 %。
- 任务式走法下，走完髋偏航扭转 ≤ 15° 的重置 ≥ 90 %。
- 用时：4090b 32 个进程加 4090 8 个进程，data3 约 2.5–4 h，data3b 约 1.5–2.5 h，两种并行或先后跑都行。
- 段数：data3 约 6–7.5 万，其中可用单手段 4–5 万；data3b 约 6.5–8 万。

**模型**（D 配方，seed 0）
- SIMPLE val 的落点误差在 1.5–3.0 cm。LML 上 D 的 val 约 2.2 cm。SIMPLE 是确定性的，有踝重钉，动作可能更可重复，所以略偏低也可能。
- 学习曲线：frac 1.0 对 0.3 的误差比在 0.65–0.85，曲线还在降。
- xfer 记录：误差除以 SIMPLE val 在 0.8–1.4。这些记录的起点没加扰动，正好在扰动分布的中心。LML 的 F 在这批记录上是 2.38 倍左右，这次应该明显更低。
- 真实 SIMPLE 局：
  - 主标签下，误差除以 SIMPLE val 在 1.0–2.0；
  - 补充标签下在 1.5–4，因为任务里下一条命令往往在身体停稳之前就发了。
- 起始状态在范围内的比例：原地伸手 ≥ 90 %；先走后伸 60–95 %。范围外的段，误差倍数比范围内的高。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| 采集检查有任何一项不过 | 停下查清，不训练 |
| 真实局主标签倍数 ≤ 1.3，且置信区间上限 ≤ 1.5 | 这个模型直接用于 SIMPLE 的"直接执行 vs 用自身模型修正"对比。反解用 lmlcmp3 定下来的做法，lmlcmp3 没结论时 S 和 R 都跑。当天登记对比 |
| 1.3–1.5 | 先在 SIMPLE val 上做一次校准再用，另行登记 |
| > 1.5 | 不用于对比。先看误差是不是集中在范围外的起点：是的话，就在那一类起点上加采；不是的话，找 SIMPLE 和训练数据在别处的不同（比如第一条命令时关节还在动） |
| SIMPLE val > 3.5 cm | 模型本身没学好，先查数据和训练，不看真实局 |
| 学习曲线 frac 1.0 对 0.3 仍降 10 % 以上 | 第二批加采 50 %，种子沿用，`--resets` 拉大 |
| 任务式走法下，扭转 ≤ 15° 的重置 < 80 % | 任务式走法没做对，查清以后再训练 |

## 预测修订

### 修订 1：采集的实现细节和冒烟，正式采集之前（2026-11-26，冻结于 2026-11-26T02:17:20Z）

预测和判据一行不改。这里只写登记里没写到的实现细节，以及冒烟看到了什么。训练和评估的实现细节另写修订 2，在开训之前提交。

**任务式走法**（`sc_collect.py` 的 `task_walk()`、`walk_plan()`、`walk_s()`）
- **哪一半**：先走后伸的重置里，按 `crc32("taskwalk:<重置编号>") % 2` 分。偶数走任务式，奇数照 d3 规则（转、走、转）。
  - 这个分法不用随机数，和其他抽样无关；
  - d3 那份走路抽样照样抽，伸手命令因此和不分时完全一样；
  - 抽出的 d3 走法记在日志的 `walk.d3` 里。
- **位移**：在 gen_reach 的 `APPROACH_BOX` 里均匀抽：前后 −0.40 到 1.00 m，左右 −0.30 到 0.30 m。随机数用 `default_rng([seed + i, 2])`，自成一路。
  - simple_bench 在 1.0 m 处分段，箱子里的位移都不超过 1.0 m，所以只走一段。
- **怎么走**：直接调 simple_bench 走到站位时用的 `gen_reach.open_walk(m, 前后, 左右, 0.0)`。
  - 它调 `Motor.walk_to`，参数是 `face="fixed"`，终点朝向取当前测得的朝向；
  - 所以 walk_to 里没有原地转身段，走前不转、走完也不转（a2 = 0）。原来的 `--walk-face fixed` 末尾还要转到 d3 抽的朝向，这里没有这一段。
  - 停步修正关闭，和 BFM_ROOT=odo 下 env.py 的设置相同。
  - `walk_cmd` 记 (前后, 左右, 0)。
- **data3b 也一样**：data3b 的先走后伸重置也一半照 d3 规则，一半任务式，分法和 data3 相同。data3b 的抽样里同样有先走后伸。2026-11-26 和协调方确认过，修订 2 里再记一笔。
- **新增的记录**：每次重置的日志里，`walk` 加了几个字段：
  - `kind`：d3 或 task；
  - `ref_yaw_span_deg`：走路参考在整段走路里的朝向变化范围。任务式应为 0；
  - `hip_yaw_end_deg`：walk_to 返回时最后一行的左右髋偏航关节角；
  - `hip_yaw_cmd_deg`：第一下伸手命令那一行的左右髋偏航关节角。

  每次重置还有 `hip_yaw_first_deg`，即第一下命令时的髋偏航，原地重置也记。每段多了一个标量 `walk_task`。data.py 读入不受影响。
- **扭转的口径**：两个髋偏航关节角绝对值取较大的那个。站姿里两个都是 0°。预测里"走完"指 walk_to 返回的那一行，第一下命令那一行同时报。

**起始扰动**：ScaleBFM 正式用 0.007 rad，这是 `PERTURB` 的缺省值。冒烟和正式采集都没另传 `--perturb`。

**检查新增第 8 项**（`sc_collect.py check`）
- 每个先走后伸重置的 `kind` 和 `task_walk(重置编号)` 一致；
- 任务式的位移在箱子里，转角为 0；
- 参考朝向范围不超过 1e-6°。

以上任何一处不对都记为检查失败。扭转比例只报出来，不算检查项，它属于登记里的判据。

**冒烟**（4090，8 个进程，nice 10，只用 CPU；2026-11-26 02:10:58–02:12:16Z）
- **怎么选的重置**：data3 和 data3b 各 20 次，从正式编号段的前面挑，每种 10 次任务式、4 次 d3 走法、6 次原地。
  - data3 编号：0–14、27、30–33；data3b 编号：0–7、11、12、15、22、23、29、34、37、54、57、61、63。
  - 输出在 `loco_outputs/selfmodel_simpledata_20261126/smoke/`，不并进正式数据。正式采集会把这些编号按同样的种子重新跑一遍。
- **检查**：两种都是 `CHECK PASSED` 和 `SIMPLE CHECK PASSED`。第 8 项两种都是：分法不一致 0，越出箱子或有转角 0，参考朝向范围最大 0.00e+00。
- **摔倒、丢失、跳过**：两种都没有摔倒或丢失。等不到静止被跳过的伸手，data3 1/40，data3b 1/80。
- **走路本身**：任务式走完的朝向变化在 ±2.6° 以内，是测得的漂移；走路参考本身朝向不变。
- **髋偏航扭转**（`check.txt` 原文）：

  ```
  data3:  task walk end 10/10 = 100.0 % <= 15 deg, median 5.0, max 7.3; first command 10/10 = 100.0 % <= 15 deg, median 5.9, max 8.4
          d3 walk end 0/4 = 0.0 % <= 15 deg, median 57.2, max 68.3; first command 0/4 = 0.0 % <= 15 deg, median 59.8, max 65.9; in place first command 6/6 <= 15 deg, max 2.4
  data3b: task walk end 10/10 = 100.0 % <= 15 deg, median 4.3, max 10.8; first command 10/10 = 100.0 % <= 15 deg, median 4.5, max 11.1
          d3 walk end 0/4 = 0.0 % <= 15 deg, median 50.9, max 56.0; first command 0/4 = 0.0 % <= 15 deg, median 53.1, max 56.2; in place first command 6/6 <= 15 deg, max 2.6
  ```

  合起来，任务式 20/20 不超过 15°，最大 11.1°；d3 走法 0/8。
- **用时**（4090，8 个进程）：data3 每次重置中位 9.5 s，其中原地 7.5 s、先走后伸 12.0 s；data3b 中位 11.5 s。

**冻结**：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/asrun/`，FROZEN_UTC 2026-11-26T02:17:20Z。
- **两份代码**：
  - `repo/` 给 4090 用，`repo_b/` 给 4090b 用，都是开发阶段那两份的拷贝；
  - src、scripts、configs、pyproject 是实拷，assets 和 third_party 链到原处；
  - 两份的代码逐个文件相同。`repo_b/outputs/scene` 里的 simple_g1_sonic 和 walk_mars 是把 `/workspace/` 换成 vePFS 真实路径的副本。
- **4090b 的启动**：
  - 环境不改。`py4090b.py` 已移进实验目录 `scripts/experiments/selfmodel_simplecollect_20261126/`，它按 vePFS 的真实路径重建 sys.path；
  - 另设 `LOCOMANI_WORKSPACE`、`SONIC_POLICY` 两个环境变量和 `SX_REPO=asrun/repo_b`。
- `MD5.check` 一共 2369 个文件，包括 `asrun/tools/` 里的启动脚本。

| 文件 | md5 |
|---|---|
| selfmodel_simplecollect_20261126/sc_collect.py | f69baf4b |
| selfmodel_simplecollect_20261126/py4090b.py / sc_start.py | 23664f5d / f19b7921 |
| selfmodel_xfer_20261125/sx_collect.py | cd290c2e |
| selfmodel_data3_20261125/d3_collect.py | dc92703a |
| locomanilab_20261117/lml_selfmodel.py | f5168f31 |
| selfwm_20261029/env.py | 62b8856f |
| reachwm_20261030/gen_reach.py / leg_odo.py / data.py | 63f6d90d / 5905796f / 174071cb（fdvel 的 FDVEL 版） |
| src/locomani/backends/scalebfm/motor.py | 23269ad1 |
| tools/formal.sh / finish.sh | 795a7ff2 / 79174b77 |

- **冒烟和冻结版的关系**：冒烟用的 sc_collect 是 f67027ef。它和冻结版 f69baf4b 只差检查里一行打印（站姿髋偏航取关节的下标顺序），驱动部分完全相同。
- **核对**：冻结版在两台机器上各重跑了冒烟的一个分片（data3 5 次、data3b 5 次重置），比较结果在 `tools/asrun_test/`：
  - 4090 和 4090b 的输出逐字节相同；
  - data3b 和冒烟的分片逐字节相同（SAME）；
  - data3 和冒烟共有的 4 次重置，日志和每段数组都相同。第 5 次重置是核对时编号敲错了（30，冒烟里是 31），不在比较之内。

**正式采集的跑法**（`tools/formal.sh`，跑完用 `tools/finish.sh`）
- **分片**：data3 和 data3b 各分 40 片。4090b 跑第 0–31 片，32 个进程；4090 跑第 32–39 片，8 个进程。
- **每个进程**：先跑完自己那片 data3（750 次重置），再跑同号的 data3b（375 次）。nice 10，只用 CPU，`CUDA_VISIBLE_DEVICES` 为空。
- **输出**：`loco_outputs/selfmodel_simpledata_20261126/{data3,data3b}/`；日志在 `logs/{d3,d3b}_w<k>.log`；每个进程结束写 `logs/done_<k>`，里面是两次运行的退出码。
- **收尾**：40 个都结束以后，在 4090 上用冻结版对两种数据各跑 merge、check、summary，输出到各自目录的 `{merge,check,summary}.txt`。

### 修订 2：中间点样本、data3b 的走法、训练和评估的实现细节，开训之前（2026-11-26，冻结于 2026-11-26T02:44:49Z）

预测和判据一行不改。写这一节时正式采集还在跑，没看过正式数据，也没有训练过模型。新脚本都在 `scripts/experiments/selfmodel_simpledata_20261126/`。

**1. 中间点样本多存一行，k0 = 3**
- **原因**：冻结版 D 的中间点 k0 = 2，只存 k−2、k−1、k 三行，再加最后 6 行。fdvel 的 data.py 在 `REACHWM_FDVEL=1` 时断言 k0 ≥ 3，因为 S[k0−2] 的差分要用到它前面那一行。两者冲突，登记时没发现。2026-11-26 和协调方确认了下面的改法；另外两种做法（关掉中间点的 FDVEL、换配方）都和登记不符，不用。
- **改法**（`sd_mid.py`）：
  - 是 train3b 冻结版 `t3b_mid.py`（6a39671c）的拷贝，只改一处：每个起点多存 k−3 一行，k0 设为 3；
  - 其余不动，起点的选法、丢弃规则和自带检查都一样；
  - 附带三处：输入路径必须显式给；运行时断言 FDVEL 关着（自带检查要拿 x0 和原始 S 行比）；多存一个 `mid_<name>_src.npz`，记每个起点的源段和行号，给核对用。
- **核对**（`sd_mid_check.py`）：正式数据上先用冻结版 t3b_mid 和 sd_mid 各生成一份，再核对，不过就停。
  1. FDVEL 关：两份起点顺序相同；x0、u 和标签共 13 个数组逐字节相同；新的 k0 全是 3，每段多一行。
  2. FDVEL 开：`data.fdvel_x0` 换掉的 96 列（三行的 29 个关节速度和 3 个角速度）要等于用源段真实相邻两行算出的差分，误差 ≤ 1e-5；其余 182 列和 FDVEL 关时相同。
  3. 开 FDVEL 的子进程用 `data.load` 读新文件，k0 ≥ 3 的断言通过，读出的 x0 和第 2 步逐字节相同。

  冒烟数据上的结果（原文）：

  ```
  data3: 316 new / 316 old starts; same order True; k0 new [3] old [2]; n_steps new = old + 1: True
    FDVEL off, byte equal: x0 True, u True, pos_f True, rot_f True, settled True, fell True, err_cm True, target True, rot6 True, jaw0 True, rot0 True, pel_f True, q_f True
    FDVEL on, 316 sampled starts vs the source rows: joint velocity max |d| 2.38e-07 rad/s, angular velocity max |d| 2.85e-08 rad/s; 96 columns replaced, the other 182 equal to FDVEL off: True
    child with REACHWM_FDVEL=1: data.load ran (k0 >= 3), its x0 byte equal to fdvel_x0 here: True
  data3b: 513 new / 513 old starts; ...（同上，角速度 4.32e-08）
  MID FDVEL CHECK PASS
  ```
- **划分检查**（`sd_split_check.py`）：train3b 的 `t3b_split_check.py` 用 k0 == 2 认中间点，现在原有段也是 k0 = 3，认不出来了。改成按文件认：从 `mid_*.npz` 来的段算中间点。另加一条：中间点的 k0 全是 3。其余检查不变。冒烟数据上 `SPLIT CHECK PASS`。

**2. data3b 的先走后伸重置**：一半照 d3 规则，一半任务式，分法和 data3 相同，按 `crc32("taskwalk:<重置编号>")` 分。2026-11-26 和协调方确认。实现和冒烟见修订 1，正式采集就是这样跑的。

**3. 训练**（`sd_run.sh`，4090 上从冻结副本跑，tmux `sd_run`）
- **开训前的门槛**：
  - `logs/finished` 已写；
  - data3 和 data3b 的 check.txt 里都有 `CHECK PASSED` 和 `SIMPLE CHECK PASSED`；
  - 判据里的扭转：两种数据 check 第 8 项 "task walk end" 的比例都 ≥ 80 %；
  - 然后等 `TRAIN_GO`。它在本修订提交并推送之后手写。
- **训练本身**：
  - train3b 冻结版 `t3b_train.py`（c7ecf581）训 E3_ensemble；
  - 数据是 data3、data3b 和两个中间点文件；seed 0，frac 1.0 和 0.3 两个并行，各占一张卡；
  - `REACHWM_GOAL_REF/APPROACH/QFIX/FDVEL=1`，不开 X0_REF；
  - 迭代数用 train.py 的缺省值，和 train3b 的 D 相同；nice 10，每个 4 个 CPU 线程。
- **显卡**：开训前 nvidia-smi 记进 run.log，从 4090 的 1–3 号卡里选两张空的。0 号卡不碰。
- **frac 0.3**：data.splits 按每次重置的 frac_rank 嵌套，中间点跟着它的源重置走。

**4. 评估的口径**
- **SIMPLE val**：data3 的 val 集，全部伸手，只算没摔、末端停稳的段，取 D（frac 1.0）落点误差的中位数。
  - 这和"LML 上 D 的 val 约 2.2 cm"是同一个口径：train3b `final_val.txt` 里 data3 val、all 那一格的 D0 是 2.25 cm；
  - 预测里的"1.5–3.0 cm"、判据里的"> 3.5 cm"，以及评估 2、3 的分母，都用这一个数。
- **学习曲线**：同一格里 D 和 D_f0.3 中位数的比值。区间用 1000 次按重置重抽，两者配对。判据"仍降 10 % 以上"指这个比值 < 0.90。test_iid 的比值一起报，但判据只看 val。
- **评估 1**（`sd_eval.py`）：
  - data3 和 data3b 各取 7 个留出集：val、test_iid、四个 band、edge；
  - 每个都分全部、第一下、连续伸手三组报；
  - data3 另外把第一下分成原地、任务式先走、d3 式先走；data3b 另外按入口、预抓取、相对伸手分；
  - 每格报位置误差的中位数和 p90、姿态误差的中位数、直接执行的误差，以及学习曲线比值。
- **评估 2**：
  - 用 xfer 的 `sx_eval.py`（81f61381，和 fdvel 用的是同一个文件），FDVEL=1；
  - `--lml` 指本实验的 data3，所以分母就是 SIMPLE val；`--models` 指本实验的 D；
  - `--keep settled` 为主，ended 也报；
  - F 的倍数取 fdvel 已有的结果：`xfer/scalebfm_simple/F_s0_settled.txt` 的 all 行（2.309x），三个 seed 见 `xfer_table.txt`（2.31–2.41x）。
- **评估 3：真实局**（`sd_real.py` 记录，`sd_real.sh` 跑，`sd_real_merge.py` 合并和检查）
  - **哪些局**：主指标 A 栏的同一批局，四个抓取任务各 30 局（0–29）。
    - 用 simple_bench 自己的 `main()` 跑，参数照 bench_via.sh 的 A 栏：`--method direct --bench-set controller=scalebfm --bench-set sensor=rgbd --trace --episodes 0-29`；
    - 环境变量 `REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1`。
  - **设备**：ScaleBFM 的策略放在 GPU 上，和 A 栏当时一样，`CUDA_VISIBLE_DEVICES` 和 `MUJOCO_EGL_DEVICE_ID` 设成同一张卡。8 个进程（两种标签 × 四个任务），nice 10。
    - 理由：冒烟里同一局（TabletopGraspMP 第 0 局）在 GPU 上的 simple_bench 行和 A 栏逐位相同（手掌误差 33.94 cm），在 CPU 上是 33.765 cm；
    - 用 GPU，补充标签的局就是 A 栏那一局本身；
    - 训练数据是在 CPU 上采的，策略在 CPU 和 GPU 上的数值差别留作一个已知的小差异。
  - **simple_bench 的版本**：A 栏跑的 md5 是 abc246fd，冻结副本里的是 80511140，后来加过别的开关。两者在 A 栏的路径上是否等价，用逐局比对来查：补充标签的每一局，simple_bench 行的 palm_cm 和每个阶段的 err_cm、ik_cm 都必须和 A 栏同一局相同。sd_real_merge 检查这一条，不过就停。
  - **记录器**：
    - 包在 `Motor.tick` 最里层，每拍记一行，和采集用同一个 `SimpleHost.sample` 和 `SRig.record`；同时记这一拍 Motor 的参考位姿作为参考帧；
    - 预抓取和进入两次 `Motor.reach`，调用期间截下 `_blend` 的位姿和时长，即规划，写法照 `plan_native_b`；抬起那一次不记；
    - 命令行取调用前的最后一行；走路记 `open_walk` 各段的起点和要求的位移；
    - 段用采集的同一套函数生成（`D3.segments`、`SX.add_fields`、`D3.augment_b`），格式是 data3b 的，kind 1 是预抓取，kind 2 是进入；
    - 重置编号：主标签 90 000 000 起，补充标签 91 000 000 起，再加 1000 × 任务 + 100 × 难度 + 局号。
  - **手点换算**：采集的手点是 wrist_yaw_link 前方 13 cm（`L.HAND_OFF`），simple_bench 的 Motor 规划的是 Dex3 手掌点（`REACH_HAND_OFF` 0.0415 m）。
    - 一条命令是一个完整的腕部位姿：手掌点 T 加腕部姿态 R；
    - 记录里的命令目标取同一个位姿的 13 cm 点，即 T + R (HAND_OFF − OFF)；
    - ik_err 是规划最后一个位姿的 13 cm 点到它的距离；标签是测得的 13 cm 点。
  - **主标签**：
    - 在预抓取命令前、进入命令前和合手前，都按 d3 规则等：从参考结束起（预抓取前从走完那一刻起），在从命令起算的 0.1 s 网格上，取第一次满足"骨盆和两个手点 0.5 s 内都低于 2 cm/s"的那一行，最多等到参考结束后 3 s；
    - 这一行就是上一下的末端标签，下一条命令也在这一行发；
    - 只改命令什么时候发，不改发什么，命令仍是 simple_bench 自己从感知算出来的。仿真里的速度只决定什么时候发，和采集时的 d3 规则一样。
  - **补充标签**：局照原样跑。预抓取的末端取进入命令之前、网格上的最后一行；进入的末端取第一次合手命令之前、网格上的最后一行。settled_end 记那一行是否满足 d3 规则。
  - **保留哪些段**：
    - 主标签：没摔、末端停稳，和 SIMPLE val 同一口径。判据用这个；
    - 补充标签：只要没摔，停没停稳都算。补充标签本来就不等停稳，预测的 1.5–4 倍说的就是这种情况；
    - 两种标签的另一种保留法也都报。
  - **倍数和区间**：用 sx_eval 的 all 行，倍数 = 真实段误差中位数 / SIMPLE val。判据里的区间用两边都重抽的那一个（"val resampled too"），上限 ≤ 1.5 才算过。第一下和第二下另报。
  - **原地伸手**：A 栏的 120 局都先走（位移 > 2 cm），所以真实段全是"先走后伸"。评估 4 里"原地伸手 ≥ 90 %"那条预测在真实局上没有段可算，报 n = 0。
  - **冒烟**（4090，GPU 2 和 3）：BendPickMP 和 TabletopGraspMP 的第 0–1 局，两种标签各跑一遍；另有 TabletopGraspMP 第 0 局在 CPU 上的两种标签。
    - 每局 2 段，没有摔倒，标签行都在网格上；
    - 补充标签 4/4 局和 A 栏逐位相同；8 段里 7 段末端停稳，TabletopGraspMP 第 1 局的预抓取在进入命令时还没停稳；
    - 主标签 8 段全部停稳。走完后要等 0.70–0.96 s 才发预抓取；进入和合手前只等 0.02–0.08 s，因为 `Motor.reach` 自己会等到停稳，最多 1.5 s；
    - 例子，TabletopGraspMP 第 0 局：补充标签的预抓取在第 168 行发、第 438 行取末端，进入在第 439 行发、第 654 行取末端；主标签的预抓取在第 203 行发；
    - 数据用 data.py（FDVEL=1）读入，x0 有 278 列，全部有限。
- **评估 4**（`sd_range.py`）：
  - **训练集**：D 的训练集，即 data3、data3b 和中间点，data.splits 的 train，frac 1.0。
  - **范围**：在起点行查 39 个量，每个都要落在训练集的 p0.5–p99.5 之内：29 个关节角；骨盆的高度和骨盆系里的重力方向，共 4 个；两个手点在起点朝向系里的位置，xy 相对骨盆，z 原样，共 6 个。按关节、骨盆、手点三组分别报，也报合起来的。
  - **最近邻**：在 x0 空间（FDVEL=1 时的模型输入）按训练集每列的均值和标准差标准化，去掉常数列，算到最近一个训练段的距离。
    - 阈值是训练集自己最近邻距离的 p99：用 seed 0 抽 10 000 个训练段，每个只和别的重置的段比，同一条轨迹自己的中间点不算；
    - 不超过阈值算在范围内。
  - **伸手表**：命令目标在伸手表里它那只手、那个骨盆高度的框内，用采集用的同一张表（md5 9d4c716a）和 `d3_collect.in_table`。
  - **三项都满足**才算在范围内。误差（D，frac 1.0）按在范围内和不在分开报，每一项单独也报，附上相对 SIMPLE val 的倍数；再按原地 / 先走、第一下 / 第二下分组。
  - **参照**：SIMPLE val 自己落在范围内的比例一起报。39 个量各留 1 % 在外，训练分布里的段本身也会有一部分"不在范围内"。
  - 比例的区间用 1000 次按局重抽。
- **整条流程的冒烟**：冒烟数据上用 `SD_TEST=1` 从头跑过一遍 sd_run.sh，训练只跑 300 步，接着跑上面两个真实局的冒烟。mid、两个核对、训练、评估 1–4 都跑通了。数字没有意义，不记。

**5. 冻结**
- 新文件拷进 `asrun/repo/scripts/experiments/selfmodel_simpledata_20261126/`，md5 记在 `asrun/MD5_r2.check`，冻结时刻记在 `asrun/FROZEN_UTC_R2`。sd_run.sh 和 sd_real.sh 开跑前都要核对 MD5.check 和 MD5_r2.check。
- train3b 的 `t3b_train.py` 和 `t3b_mid.py` 直接用 train3b 的冻结副本，sd_run.sh 开跑前核对它们的 md5。
- 修订 1 的 MD5.check 不动。正式采集不读这些新文件。

| 文件 | md5 |
|---|---|
| sd_mid.py / sd_mid_check.py / sd_split_check.py | 5c46a94d / 5521766f / 86d68705 |
| sd_real.py / sd_real_merge.py / sd_real.sh | 3703c1aa / 4cb32665 / c1bcf695 |
| sd_eval.py / sd_range.py / sd_run.sh | 500927fc / 1276163a / a29c1c0d |
| train3b asrun：t3b_train.py / t3b_mid.py | c7ecf581 / 6a39671c |
| selfmodel_xfer_20261125/sx_eval.py | 81f61381 |

- **顺序**：本修订单独提交并推送 → 跑真实局（sd_real.sh，不依赖采集）→ 采集结束、检查通过 → 写 TRAIN_GO → sd_run.sh 训练和评估。

### 修订 2 补记：真实局的显卡、训练改到 4090b（2026-11-26，冻结于 2026-11-26T02:53:20Z）

预测和判据仍然一行不改。写这一节时还没有训练，也没看过真实局的误差。

**1. 原地伸手那条预测**：A 栏的 120 局都是先走后伸，真实局里没有原地伸手的段。评估 4 里"原地伸手 ≥ 90 %"这条在真实局上没有样本，不算中也不算不中。

**2. 真实局的显卡**
- **经过**：
  - 02:45:33Z 用 sd_real.sh 起了 8 个进程，1、2、3 号卡上分别是 3、3、2 个。开之前看过 nvidia-smi，三张卡都是 820 MiB、0 %。
  - 随后协调方说明了卡的安排：2、3 号卡归 `simple_sonic_vrref` 用，到它写出 DONE 为止；HA 采集也在 1–3 号卡上，每张卡总显存不超过 20 GB，我们每张卡最多开 1–2 个进程。
  - 02:47:17–22Z 按 pid 停掉了 2、3 号卡上的 5 个进程：main 的 XMoveBendPickTeleop、BendPickMP，task 的 XMovePickTeleop、XMoveBendPickTeleop、TabletopGraspMP。它们当时分别跑完了 12、12、14、13、12 局。
- **删掉的半截输出**：这 5 组的 `real/<标签>/rows/<任务>.jsonl`、`log/<任务>.jsonl`、`data/<任务>.npz`、`data/arm/<任务>.npz`。进程日志改名为 `real/logs/<标签>_<任务>.stopped.log` 保留，scene 目录会被重跑覆盖。
- **1 号卡上的 3 组照常跑完**：main 的 XMovePickTeleop、TabletopGraspMP，task 的 BendPickMP，各 30 局。sd_real.sh 自己在 02:50:27Z 做的合并因为缺文件失败了，作废；最后的合并由下面的排队脚本重做。
- **重跑**：
  - 用 `tools/real_queue.sh` 把这 5 组整组重跑，命令和环境变量和 sd_real.sh 相同，只用 1 号卡；
  - 同时最多开 2 个进程。每次开新进程前查 nvidia-smi，要求卡上已用显存加 1 GB 不超过 20 GB，不满足就等；
  - 02:51:03Z 开始，开之前 1 号卡已用 16.4 GB，开了 2 个以后是 17.6 GB；
  - 跑完以后，4090 的卡上不留我们的进程。
- **结果不受影响**：在 GPU 上，局是确定的。合并时，补充标签的每一局都要和 A 栏逐位相同，这条检查不变。

**3. 训练改到 4090b**（4090 的卡都会被 HA 占满，4090b 的两张卡空着）
- `sd_run.sh` 加了 `SD_MACHINE=4090b`：
  - 用 `asrun/repo_b` 和 4090b 的启动方式（`/usr/bin/python3.10 -S …/py4090b.py`，外加 `LOCOMANI_WORKSPACE` 和 `SONIC_POLICY`），和正式采集在 4090b 上的跑法一样；
  - 本实验的脚本仍取 `asrun/repo` 里的那份。repo 和 repo_b 里的 data.py、sx_eval.py 是同一个文件（174071cb、81f61381）。
- **显存门槛**：每次训练、评估 1、评估 4 开始前，要求那张卡至少空出 8 GB（nvidia-smi 的总显存减已用），不够就等。以后 HA 在 4090b 上训练时，按这个余量错开。
- `sd_mid_check.py` 第 3 步的子进程改成用 `$SD_PY` 启动（4090b 的启动命令），没设就用当前解释器。检查的内容不变。
- **冒烟**：在 4090b 上用冒烟数据、`SD_TEST=1`、300 步从头跑过一遍。mid、两个核对、训练、评估 1–4 都跑通了。训练出的 best val 和 4090 上那遍一模一样（frac 1.0 是 1.875 cm，frac 0.3 是 3.627 cm）。
- **重新冻结**：两个改过的文件拷进 asrun，`MD5_r2.check` 重写，冻结时刻记在 `asrun/FROZEN_UTC_R2B`。

| 文件 | 修订 2 | 补记 |
|---|---|---|
| sd_run.sh | a29c1c0d | 19e1c6fe |
| sd_mid_check.py | 5521766f | bb940b71 |

其余文件的 md5 不变。

## 结果

## 意外

## 结论与下一步

## 独立复核
