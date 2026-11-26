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

## 结果

## 意外

## 结论与下一步

## 独立复核
