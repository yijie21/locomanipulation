# selfmodel_precision_20261124：一次命令的精度——站位抖动后，自身模型的逐局预测能不能赢过不用模型的固定偏差修正

- 日期：2026-11-24（设计、预测、判据；正式跑之前写下，单独提交）
- 机器 / GPU：4090。
  - ScaleBFM 在 1 号卡，同时跑 3 次重复；SONIC 在 2、3 号卡，同时跑 2 次重复，每次重复 3 种挪法并行。
  - 试跑测得：SONIC 一个进程约 1.8 GB，cuRobo 一个规划约 0.5 GB，每次重复同时 6 个规划。
  - 0、3 号卡上有别人的任务和 HumanoidArena 的调试，BiGym 的规划服务也会用 1–3 号卡，每次启动前看 `nvidia-smi`。
  - 启动：`precision_launch.sh dev 4`，跑完、流程判据过了再 `precision_launch.sh test 4`。
- 代码版本：`selfwm` @ `d308277a` 加未提交的文件，正式跑时抄进每个输出目录的 `asrun/`，md5 记在结果里：
  - `scripts/experiments/simple_eval_20261112/precision.py`（新）：常数、五种挪法、退回 J、按布局的表；
  - `scripts/experiments/simple_eval_20261112/precision_run.sh`（新）：一个控制器、若干次重复的流程；
  - `scripts/experiments/simple_eval_20261112/precision_launch.sh`（新）：开发局或测试局，两个控制器一起；
  - `scripts/experiments/simple_eval_20261112/simple_bench.py`：新加 `--stance-jitter`、`--jitter-rep`、`--jitter-seed`，默认关；
  - `scripts/experiments/simple_eval_20261112/bench_via.sh`：新加 `BENCH_EXTRA`（每种做法都加的选项）、`BENCH_EPISODES`（只跑有路线的局），不设就和以前一样；
  - `scripts/experiments/curobo_avoid_20261120/plan2.py`：新加 `--ik-seed`，默认 1531 就是 cuRobo 自己的默认值；
  - 自身模型用 `outputs/simple_eval_20261112/jrefwm_20261124/model_<控制器>`，`s_check.py` 一行不改。
- 配置：SIMPLE 的 TabletopGrasp，感知栏，单目 RGB-D（默认组合），G1 加 Dex3；ScaleBFM 和 SONIC（`REACH_SONIC_NETS=gpu`），控制器不改参数；不用伺服和查表。
- 关联：
  - 用户 2026-11-24 定论文主论断："我们主要还是 claim one-shot precision，只是比较鲁棒的可以处理 obstacle avoidance 而已"（DECISIONS 11-24）；
  - [selfmodel_s_20261124](selfmodel_s_20261124.md) 的独立复核：测试局上不用模型的固定偏差修正剩 1.26 cm，模型剩 2.04 cm；原因是 TabletopGrasp 的伸手几何几乎不变。这个实验就是复核给的实验 (a)；
  - [selfmodel_jrefwm_20261124](selfmodel_jrefwm_20261124.md)：模型的训练数据，站位就是 ±6 cm 的同一种抖动；
  - TODO 38.10.4。

## 假设

论文要说的是：控制器的自身模型能预测"这一条命令发下去手会停在哪"，按预测把命令挪一下，一次命令就更准。

SIMPLE 的测试局上，这件事被一个常数做到了：每局都按开发局上 J 的平均偏差挪同样的量，比模型还准。复核的解释是伸手几何几乎不变，所以手停偏的方向和大小每局都差不多。

**站位抖动以后伸手几何会变，手停偏的量也跟着变。这时模型的逐局预测能不能比固定常数把手放得更准？**

如果不能，这个基准上能说的就只是"按偏差修正命令有用"（标定），模型不算贡献。

## 做法

### 局

- **开发局**：TabletopGrasp 没用过的 3、4 级（位置 30–49，`outputs/simple_dev_20261124` 这棵树），每个位置抖动 K 次。
- **测试局**：TabletopGrasp 官方的 0–2 级（位置 0–29），每个位置抖动 K 次。开发局跑完、流程没问题才跑。
- K = 4，两边一样（由试跑的用时定，见"试跑"一节）：开发局每个控制器 80 局，测试局 120 局。

### 站位抖动

- `simple_bench.py --stance-jitter 0.06 --jitter-rep k`：站位点在出发时的朝向坐标里前后、左右各加 ±6 cm 内的均匀扰动。
- 种子由任务、等级、局号和第几次重复定，所以同一局同一次重复，所有做法的站位完全一样。
- 和模型训练数据（jrefwm 的采集）是同一段代码、同样的 ±6 cm。之后照常受桌沿限位（离桌沿 0.2 m），往前的扰动多半被桌沿挡掉。实际上是往后 0–6 cm、左右 ±6 cm 在变。
- 不抖动时（默认），新代码的行和录像和以前逐字段相同（见"试跑"）。

### 每次重复的流程（`precision_run.sh`）

1. `bench_via.sh` 的 A（直线伸手）、plan（照开局那一眼规划路线）、JR（J 带录像，记下发令时刻的状态），都带同样的抖动。
2. `s_check.py state`：有路线的局，发令时刻的状态。
3. `precision.py shifts`：五种挪法，都只挪预抓取的目标（cuRobo 规划的终点），障碍物不动：
   - **replan（只重新规划）**：目标不挪，cuRobo 的 IK 种子从 1531 换成 7。试跑发现即使种子不换，cuRobo 每次规划出来的路线也不一样（同一局两次：步数 57 对 63，终点关节差 6–12°），所以这一组就是"换一条路线、目标不变"。
   - **const（固定常数）**：减去开发局上 J 的平均停偏（朝向坐标）。
     - ScaleBFM [−3.168, 0.100, 3.363] cm（11 局），SONIC [−3.878, 1.300, 2.347] cm（15 局）。
     - 来自 `outputs/simple_dev_20261124/bench/<控制器>/rows/JR_TabletopGraspMP.jsonl`，不抖动，开跑前算好，存在 `outputs/simple_dev_20261124/precision_const/const_<控制器>.json`。
     - 开发局上常数就是从这些局来的，所以开发局用**留一布局**的常数：去掉目标离本局 2 cm 以内的那些局再平均。试跑里 ScaleBFM 的位置 35 去掉了 4 局，36 去掉了 1 局。
     - 测试局用全部 11 / 15 局。
   - **model（自身模型）**：减去模型预测的、J 这条路线手掌停稳后的偏差（`s_check.py` 终点检查的第 0 轮，同一个算式）。没有 2 cm 的门槛，一律挪，只挪一次。
   - **sham（随机方向）**：挪的长度和 model 一样，方向随机（每局一个种子）。
   - **oracle（上限）**：减去本局 J 实际测到的停偏。用了实测的手，只作参照，不算方法。
4. 每种挪法用 `plan2.py` 重新规划一次（激活距离和 J 那条路线一样）。**规划不出来就发 J 原来的路线**，不放弃任何一局，记下退回的次数。
5. 五种挪法各跑一遍有路线的局。没有路线的局各做法都和 J 一样是直线伸手，不重跑，主指标里直接用 J 的结果。

### 和 S（selfmodel_s_20261124）的区别

- 没有"修不好就放弃"；
- 没有 2 cm 门槛和第二轮；
- 多了 replan、const、sham、oracle 四个对照；
- 站位抖动。

## 度量

- **主度量：预抓取停偏**。行里 `stages` 的 pregrasp 一段 `err_cm`：伸手结束、停稳以后，夹爪中心离开局那一眼的预抓取点多远（目标挪了也是对原来的点量）。
  - 每种挪法减 J，按局配对；
  - **model 减 const**，按局配对。这是核心。
- **按布局算**：预抓取点在 1 cm 以内的位置算同一个布局（0、1 级大多是同一个布局）。报：
  - 中位数差和 95 % 区间（按布局重抽 4000 次）；
  - 更近的布局数和符号检验（每个布局取它所有局差值的平均）。
- **见过和没见过**：测试局按布局分成"模型的训练数据在这里采过"（0–2 级里 crc32 % 3 ≠ 0 的位置所在的布局）和"没采过"，model 减 const 两边分开报。开发局（3、4 级）全是没采过的。
- entry 一段的停偏，同样报。
- **主指标照报**：TabletopGrasp 的成功局数，45 秒内和官方时限内。各做法分别报全部局和有路线的局，外加 A 和 J。
- 其他：每种挪法退回 J 的次数、摔倒数、有路线的局数。

## 预测（试跑之前写，没看过任何一种挪法的结果）

ScaleBFM：
- **有路线的局**：开发局每次重复 8 到 15 局（不抖动时 11 / 20）；测试局每次 24 到 30 局（不抖动时 28 / 30）。
- **J 的预抓取停偏中位数**：开发局 2.5 到 6 cm（不抖动时 3.38），测试局 3 到 6.5 cm（不抖动时修过的 21 局 4.71）。
- **replan 减 J**：中位数在 ±0.7 cm 以内（复核：单是重新规划最多贡献约 0.6 cm）。
- **const**：开发局（留一）1.0 到 3.0 cm；测试局 1.0 到 2.5 cm（不抖动时 1.26）。
- **model**：开发局 1.5 到 3.5 cm；测试局 1.5 到 3.0 cm（不抖动时 2.04）。
- **model 减 const（核心）**：
  - 测试局中位数在 −0.5 到 +1.0 cm 之间，我猜 +0.4，也就是常数仍然略好；
  - 模型按判据赢的可能我估约两成。
  - 理由：复核看到模型侧向有约 +2.2 cm 的固定偏差，开发局上也有（+2.16 cm），抖动不会让它消失；而 ±6 cm 的站位对手臂姿态改变不大，停偏随之变的那部分可能比 2 cm 小。
- **sham 减 J**：+0.5 到 +3 cm（方向错了的挪动只会更差）。
- **oracle**：0.5 到 1.5 cm（不抖动时 S 的手 ≈ J 的手加挪动量，差中位数 0.58 cm）。
- **entry 停偏**：各做法中位数和 J 差不超过 0.7 cm（这一段从实测的手掌出发，会自己纠正）。
- **主指标（有路线的局，45 秒内）**：J 在 90 % 以上；其他做法和 J 差不超过 10 %；官方时限内差得更多（时间卡在 16 s 上）。
- **退回 J**：各挪法不超过 15 %（S 那次 cuRobo 规划不出挪过的目标：ScaleBFM 7 / 28）。

SONIC：
- J 的预抓取停偏中位数 5 到 9 cm（不抖动时 6.33）。
- const：比 J 低 0 到 1.5 cm（SONIC 的停偏每局差得大，标准差 4.9、2.4、3.1 cm）。
- model：7 到 12 cm，比 J 还差（不抖动时 10.27）。
- model 减 const：+1 到 +5 cm，模型差。
- oracle：3 到 6 cm（不抖动时 S 的手 ≈ J 加挪动量，差中位数 2.42 cm）。
- replan 减 J：±2 cm 以内（SONIC 命令一变结局就变）。
- sham 减 J：0 到 +3 cm。

## 判据（什么结果会让我改做法，下一步做什么）

**核心判据，每个控制器分开，以测试局为准**：model 算赢过 const，要同时满足两条：
1. model 减 const 的中位数，按布局重抽的 95 % 区间整个在 0 以下；
2. model 更近的布局多于更远的布局，符号检验 p < 0.05。

结果分三种：
- **赢了，而且"没见过的布局"（测试的没采过部分加开发局）方向一致**：这个基准上可以说"模型的逐局预测比标定更准"。S 改成这次的 model 挪法（加"规划失败就退回 J"），进主表，主表同时带 const 一行。
- **只在"见过的布局"上赢**：可能是训练数据在这些布局上采过，不能当泛化的证据。论文里只写没见过的部分，并补一组没见过的布局（比如开发局加大 K）。
- **打平或输（常数不比模型差）**：SIMPLE 上的结论写成"按偏差修正命令有用，常数就够"，模型在这里不算贡献。
  - 照用户定的主论断，模型的价值得在常数失效的地方证明：BiGym 抽屉（精度直接决定成败，关节反馈 7 / 10 对 J 3 / 10），以及伸手范围更大的局。
  - 不在 SIMPLE 上继续调模型。

**对照判据**：
- **replan 减 J 的中位数小于 −1 cm**：单是换一条路线就把手放近了 1 cm 以上。几种挪法都改成和 replan 比，而不是和 J 比。
- **sham 不比 J 差**（中位数差 ≤ 0）：挪目标这件事本身就改变了手怎么停（比如碰桌沿的方式变了），model 和 oracle 的好处可能是假象。先查这个，再下结论。
- **ScaleBFM 的 oracle 中位数超过 2 cm**："按偏差反着挪"这个办法在抖动下本身就不准，模型和常数都不可能更准。这时报的是这个办法的上限，不是模型的问题。

**流程判据（开发局，决定能不能跑测试局）**：
- 出现以下任一情况，就先查清、修好再跑测试局，修了什么记在"预测修订"：
  - 有进程失败；
  - 某种挪法退回 J 超过 30 %；
  - 不抖动的对照和以前不同。
- 测试局的预测和判据就是本节这些，开发局的结果不改它们。

## 试跑（`# notebook: skip`，开发局，不进结果）

- **不抖动时新代码和旧代码相同**：ScaleBFM 开发局 30、32 两局，用新代码重跑 JR。行的 45 个字段除 `wall_s` 外全部相同，录像 8 个数组全部相同（`ref` 里有 NaN，按 NaN 相等比较）。
- **cuRobo 不重现自己的路线**：同样的输入、默认种子，30、32 两局的路线和 J 那次都不同，种子 7 也不同（见做法第 3 步的 replan）。
  - 所以同一次重复重跑一遍，A 逐字段相同，J 的路线不同，但手停的位置几乎不变（位置 35：3.583 对 3.656 cm；36：8.718 对 8.717 cm）。
  - 每次重复里各挪法都和同一次的 J 比，不受影响。
- **第一次端到端试跑发现一个错**：五种挪法跑的时候没带站位抖动（启动脚本在 A、plan、JR 之后清掉了抖动的选项）。
  - 挪法站的位置和 J 不同，SONIC 的五种挪法连"只重新规划"都偏到 15–24 cm。
  - 已改：抖动选项留给本次重复的所有做法。表格脚本加了一条检查：任何一行的站位抖动和同局 J 不同就报错停下。拿那次的输出试过，能拦下。
  - 那次的输出移到 `outputs/simple_dev_20261124/precision_trial_v1_arms_unjittered/`，不用。
- **改完再试跑**：位置 35、36 × 2 次重复 × 两个控制器（`precision_trial/`），加一次 ScaleBFM 1 次重复（`precision_trial3/`，五种挪法的规划改成一起跑以后的检查）。都没有失败的进程；SONIC 有 2 次退回 J（模型要挪 12 cm，cuRobo 规划不出来）。
- **试跑看到了各挪法的数**（开发局位置 35、36，每个控制器 4 局，只是流程检查）：
  - ScaleBFM：J 6.29、replan 6.34、const 2.99、model 2.57、sham 9.13、oracle 1.13 cm；
  - SONIC：J 4.15、const 3.47、model 6.55、oracle 2.20 cm。
  - 上面的预测写在看到这些数之前，看到之后没改。正式开发局会重跑这两个位置。
- **用时**（试跑实测，每局 8.5–13 s，每次重复的 A、plan、JR 加五种挪法）：
  - 开发局每次重复 ScaleBFM 约 14 分钟、SONIC 约 21 分钟；K = 4 时两个控制器同时跑约 50 分钟。
  - 测试局每次重复 ScaleBFM 约 26 分钟、SONIC 约 36 分钟；K = 4 时约 85 分钟。
  - K = 5 会让测试局的 SONIC 接近 2 小时，所以取 4。

## 预测修订

**2026-11-25（UTC 11-24 16:55 后），开发局跑完、测试局一局都没跑之前**：开发局触发了流程判据"某种挪法退回 J 超过 30 %"。

- **触发的数**：ScaleBFM 有路线的 44 局里 const 退回 17 局、model 14 局；SONIC 55 局里 const、model 各 19 局。
- **原因**：查了每次重新规划的状态。退回几乎全是 cuRobo 的 IK_FAIL（ScaleBFM const 16、model 13；SONIC const 18、model 19），其余是个别 FINETUNE_TRAJOPT_FAIL、GRAPH_FAIL。
  - 常数把目标往前挪约 3.2 cm、往下挪约 3.4 cm（J 的手停在目标后上方）。挪过的目标落进桌面的安全距离里，IK 就找不到不碰桌的姿态。
  - 而手实际会停在挪过的目标上方约 3 cm，所以这是规划器对参考做检查时过于保守，不是手真会撞桌。
- **顺带查了 replan**：ScaleBFM 上 replan 和 J 的停偏几乎一样。它的路线确实和 J 不同（44 局没有一局相同，终点关节差中位数 0.11 rad），只是手停的位置一样。这是结果，不是错：停偏由目标决定，不由路线决定。
- **改法**（只改流程，核心判据、预测都不改）：
  1. 挪过的目标规划失败时，按同一方向把挪动量缩到 2/3 再试；还不行缩到 1/3；都不行才退回 J 的路线。const、model、sham、oracle 四种挪法用同一条规则，replan 不挪，不受影响。
  2. 每局记下实际用的缩放（1、2/3、1/3、退回），表里各挪法分别报。
  3. 表里 entry 一段改成忽略缺失值的中位数，并报缺失的局数。SONIC 有路线的局里摔了 5 局，这些局没有 entry 的停偏，原来的中位数因此是 nan。
- **流程判据照旧**：改完重跑开发局（原输出移到 `outputs/simple_dev_20261124/precision_20261124_v1_fallback/`，不用）。缩放以后仍有某种挪法退回 J 超过 30 %，或有进程失败，就不跑测试局，先报告。
- **开发局的数我看过了**：ScaleBFM model 减 const 中位数 +0.00 [−0.66, +0.38]；SONIC +1.51 [+0.00, +2.85]。它们不改测试局的预测和判据。

## 结果（贴原始输出，不转述）

### 开发局第一次（2026-11-24 15:55–16:49Z，触发流程判据，不用）

输出移到 `outputs/simple_dev_20261124/precision_20261124_v1_fallback/`。触发的数和原因见"预测修订"。

### 开发局第二次（改过以后，17:02–17:59Z）

- 代码：`precision.py` 8eb33c5e、`precision_run.sh` f51a8321；其余同登记。
- 流程判据：没有进程失败（两个 `run.log` 没有 FAIL，每个 `pipeline.log` 都是 0 failed processes）；退回 J 最多 ScaleBFM 5 %（const）、SONIC 15 %（oracle）。所以跑测试局。
- replan 确实执行自己的路线：伸手段的参考和 JR 每局都不同，ScaleBFM 44 / 44（每局最大差中位数 6.74°），SONIC 55 / 55（13.86°）。
- `outputs/simple_dev_20261124/precision_20261124/<控制器>/table.txt`：

```
######## scalebfm/table.txt
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/precision_20261124/scalebfm  repeats 4, episodes 80, routed episode-repeats 44, layouts 9, falls in routed 0
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 17 / 17  J 40 / 30  replan 39 / 30  const 40 / 30  model 40 / 31  sham 40 / 29  oracle 40 / 31

routes with every arm's row: 44; fallback to J's plan: replan 0 (0 %), const 2 (5 %), model 1 (2 %), sham 0 (0 %), oracle 1 (2 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 44 2/3 0 1/3 0 fallback 0; const 1 28 2/3 4 1/3 10 fallback 2; model 1 31 2/3 3 1/3 9 fallback 1; sham 1 36 2/3 3 1/3 5 fallback 0; oracle 1 36 2/3 4 1/3 3 fallback 1

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   3.70  mean   4.55  missing 0
  replan  median   3.72  mean   4.88  missing 0 | minus J: median +0.00 [-0.01, +0.01], closer in 21/44; layouts closer 2/9 (sign p 0.18)
  const   median   1.98  mean   2.24  missing 0 | minus J: median -2.41 [-3.77, -1.18], closer in 39/44; layouts closer 9/9 (sign p 0.00391)
  model   median   2.52  mean   2.43  missing 0 | minus J: median -1.24 [-3.96, -0.77], closer in 41/44; layouts closer 8/9 (sign p 0.0391)
  sham    median   6.47  mean   6.79  missing 0 | minus J: median +2.51 [+1.66, +2.67], closer in 11/44; layouts closer 0/9 (sign p 0.00391)
  oracle  median   0.71  mean   1.08  missing 0 | minus J: median -3.20 [-5.59, -1.42], closer in 43/44; layouts closer 9/9 (sign p 0.00391)
  model minus const: median +0.33 [-0.36, +0.77] cm, model closer in 14/44 episodes, layouts 2/9 (worse 7, sign p 0.18)
    seen by the model's data: none
    never seen: 44 episodes, model minus const median +0.33 [-0.36, +0.77], layouts closer 2/9 (sign p 0.18)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   2.03  mean   2.36  missing 0
  replan  median   2.05  mean   2.86  missing 0 | minus J: median +0.00 [+0.00, +0.02], closer in 12/44; layouts closer 1/9 (sign p 0.0391)
  const   median   2.61  mean   2.65  missing 0 | minus J: median +0.16 [+0.00, +0.35], closer in 11/44; layouts closer 1/9 (sign p 0.0391)
  model   median   2.51  mean   2.59  missing 0 | minus J: median +0.15 [+0.00, +0.25], closer in 13/44; layouts closer 1/9 (sign p 0.0391)
  sham    median   2.02  mean   2.38  missing 0 | minus J: median -0.07 [-0.13, +0.05], closer in 23/44; layouts closer 7/9 (sign p 0.18)
  oracle  median   2.48  mean   2.67  missing 0 | minus J: median +0.21 [+0.04, +0.41], closer in 9/44; layouts closer 2/9 (sign p 0.18)
  model minus const: median -0.04 [-0.12, +0.00] cm, model closer in 30/44 episodes, layouts 5/9 (worse 4, sign p 1)
    seen by the model's data: none
    never seen: 44 episodes, model minus const median -0.04 [-0.12, +0.00], layouts closer 5/9 (sign p 1)

routes: successes within 45 s / official limit
  J 40 / 30  replan 39 / 30  const 40 / 30  model 40 / 31  sham 40 / 29  oracle 40 / 31
######## sonic/table.txt
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/precision_20261124/sonic  repeats 4, episodes 80, routed episode-repeats 55, layouts 15, falls in routed 6
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 13 / 13  J 36 / 26  replan 36 / 28  const 33 / 21  model 34 / 21  sham 34 / 25  oracle 36 / 26

routes with every arm's row: 55; fallback to J's plan: replan 0 (0 %), const 7 (13 %), model 4 (7 %), sham 3 (5 %), oracle 8 (15 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 55 2/3 0 1/3 0 fallback 0; const 1 36 2/3 4 1/3 8 fallback 7; model 1 36 2/3 9 1/3 6 fallback 4; sham 1 44 2/3 6 1/3 2 fallback 3; oracle 1 39 2/3 3 1/3 5 fallback 8

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   4.79  mean  11.77  missing 0
  replan  median   5.00  mean  11.15  missing 0 | minus J: median -0.03 [-0.19, +0.11], closer in 28/55; layouts closer 9/15 (sign p 0.607)
  const   median   4.03  mean  10.84  missing 0 | minus J: median -0.75 [-1.38, +0.00], closer in 33/55; layouts closer 9/15 (sign p 0.267)
  model   median   7.32  mean  12.62  missing 0 | minus J: median +1.02 [+0.00, +2.44], closer in 19/55; layouts closer 3/15 (sign p 0.0574)
  sham    median   8.28  mean  17.49  missing 0 | minus J: median +2.90 [+2.14, +4.59], closer in 11/55; layouts closer 0/15 (sign p 6.1e-05)
  oracle  median   2.36  mean  10.87  missing 0 | minus J: median -1.58 [-2.27, -0.51], closer in 40/55; layouts closer 11/15 (sign p 0.0574)
  model minus const: median +1.05 [+0.00, +3.21] cm, model closer in 19/55 episodes, layouts 3/15 (worse 11, sign p 0.0574)
    seen by the model's data: none
    never seen: 55 episodes, model minus const median +1.05 [+0.00, +3.21], layouts closer 3/15 (sign p 0.0574)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   2.55  mean   5.21  missing 1
  replan  median   2.74  mean   5.55  missing 1 | minus J: median +0.13 [-0.08, +0.51], closer in 22/54; layouts closer 5/15 (sign p 0.302)
  const   median   2.96  mean   6.28  missing 1 | minus J: median +0.27 [+0.00, +0.47], closer in 18/54; layouts closer 2/15 (sign p 0.0225)
  model   median   3.07  mean   5.46  missing 1 | minus J: median +0.04 [-0.15, +0.58], closer in 24/54; layouts closer 3/15 (sign p 0.0574)
  sham    median   2.89  mean   8.07  missing 1 | minus J: median +0.21 [+0.11, +0.54], closer in 16/54; layouts closer 4/15 (sign p 0.118)
  oracle  median   3.01  mean   7.37  missing 1 | minus J: median +0.25 [+0.00, +0.77], closer in 16/54; layouts closer 5/15 (sign p 0.424)
  model minus const: median -0.07 [-0.56, +0.15] cm, model closer in 28/54 episodes, layouts 8/15 (worse 6, sign p 0.791)
    seen by the model's data: none
    never seen: 54 episodes, model minus const median -0.07 [-0.56, +0.15], layouts closer 8/15 (sign p 0.791)

routes: successes within 45 s / official limit
  J 35 / 25  replan 35 / 27  const 32 / 20  model 33 / 20  sham 33 / 24  oracle 35 / 25
2026-11-24T17:02:36Z
2026-11-24T17:59:39Z
```

### 测试局（18:03:59Z–19:31:58Z）

- 代码同开发局第二次（`precision.py` 8eb33c5e、`precision_run.sh` f51a8321）。ScaleBFM 在 1 号卡，SONIC 在 2、3 号卡。
- 没有进程失败：每个 `pipeline.log` 的每一步都是 0 failed processes，`run.log` 没有 FAIL。
- 退回 J 最多 ScaleBFM 1 %、SONIC 8 %（oracle）。
- `outputs/simple_eval_20261112/precision_20261124/<控制器>/table.txt`：

```
######## scalebfm/table.txt
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/precision_20261124/scalebfm  repeats 4, episodes 120, routed episode-repeats 112, layouts 18, falls in routed 0
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 64 / 64  J 119 / 101  replan 119 / 106  const 112 / 98  model 115 / 98  sham 116 / 91  oracle 116 / 94

routes with every arm's row: 112; fallback to J's plan: replan 0 (0 %), const 1 (1 %), model 0 (0 %), sham 1 (1 %), oracle 1 (1 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 112 2/3 0 1/3 0 fallback 0; const 1 83 2/3 15 1/3 13 fallback 1; model 1 86 2/3 15 1/3 11 fallback 0; sham 1 100 2/3 8 1/3 3 fallback 1; oracle 1 99 2/3 7 1/3 5 fallback 1

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   3.97  mean   4.70  missing 0
  replan  median   4.02  mean   4.70  missing 0 | minus J: median +0.00 [-0.00, +0.00], closer in 51/112; layouts closer 8/18 (sign p 0.815)
  const   median   1.74  mean   3.08  missing 0 | minus J: median -2.63 [-3.84, -1.50], closer in 104/112; layouts closer 16/18 (sign p 0.00131)
  model   median   2.47  mean   2.67  missing 0 | minus J: median -1.55 [-3.16, -0.50], closer in 92/112; layouts closer 15/18 (sign p 0.00754)
  sham    median   6.06  mean   6.30  missing 0 | minus J: median +1.74 [+0.31, +2.70], closer in 40/112; layouts closer 1/18 (sign p 0.000145)
  oracle  median   0.61  mean   1.19  missing 0 | minus J: median -3.41 [-4.22, -2.64], closer in 110/112; layouts closer 17/18 (sign p 0.000145)
  model minus const: median +0.69 [+0.23, +1.14] cm, model closer in 36/112 episodes, layouts 7/18 (worse 11, sign p 0.481)
    seen by the model's data: 92 episodes, model minus const median +0.59 [+0.10, +1.14], layouts closer 7/15 (sign p 1)
    never seen: 20 episodes, model minus const median +1.10 [+1.02, +1.14], layouts closer 0/3 (sign p 0.25)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   2.08  mean   2.28  missing 0
  replan  median   2.10  mean   2.29  missing 0 | minus J: median -0.00 [-0.01, +0.01], closer in 56/112; layouts closer 8/18 (sign p 0.815)
  const   median   2.61  mean   3.81  missing 0 | minus J: median +0.30 [+0.19, +0.45], closer in 14/112; layouts closer 1/18 (sign p 0.000145)
  model   median   2.66  mean   2.85  missing 0 | minus J: median +0.22 [+0.10, +0.36], closer in 29/112; layouts closer 4/18 (sign p 0.0309)
  sham    median   2.28  mean   2.64  missing 0 | minus J: median +0.04 [-0.04, +0.11], closer in 50/112; layouts closer 6/18 (sign p 0.238)
  oracle  median   2.73  mean   3.02  missing 0 | minus J: median +0.28 [+0.18, +0.47], closer in 16/112; layouts closer 3/18 (sign p 0.00754)
  model minus const: median -0.08 [-0.12, -0.05] cm, model closer in 77/112 episodes, layouts 10/18 (worse 8, sign p 0.815)
    seen by the model's data: 92 episodes, model minus const median -0.09 [-0.19, +0.00], layouts closer 7/15 (sign p 1)
    never seen: 20 episodes, model minus const median -0.08 [-0.13, -0.07], layouts closer 3/3 (sign p 0.25)

routes: successes within 45 s / official limit
  J 111 / 93  replan 111 / 98  const 104 / 90  model 107 / 90  sham 108 / 83  oracle 108 / 86
######## sonic/table.txt
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/precision_20261124/sonic  repeats 4, episodes 120, routed episode-repeats 106, layouts 19, falls in routed 4
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 39 / 39  J 72 / 60  replan 69 / 57  const 72 / 59  model 69 / 53  sham 66 / 51  oracle 70 / 50

routes with every arm's row: 106; fallback to J's plan: replan 0 (0 %), const 3 (3 %), model 2 (2 %), sham 3 (3 %), oracle 8 (8 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 106 2/3 0 1/3 0 fallback 0; const 1 93 2/3 3 1/3 7 fallback 3; model 1 86 2/3 15 1/3 3 fallback 2; sham 1 79 2/3 12 1/3 12 fallback 3; oracle 1 73 2/3 12 1/3 13 fallback 8

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   5.92  mean   8.57  missing 0
  replan  median   5.76  mean   9.87  missing 0 | minus J: median +0.11 [-0.05, +0.30], closer in 46/106; layouts closer 8/19 (sign p 0.648)
  const   median   4.63  mean   7.73  missing 0 | minus J: median -1.45 [-2.19, -0.56], closer in 72/106; layouts closer 13/19 (sign p 0.167)
  model   median   9.26  mean  10.79  missing 0 | minus J: median +2.41 [+0.69, +3.71], closer in 34/106; layouts closer 5/19 (sign p 0.0636)
  sham    median   9.12  mean  12.15  missing 0 | minus J: median +2.84 [+1.80, +3.43], closer in 27/106; layouts closer 1/19 (sign p 7.63e-05)
  oracle  median   3.00  mean   6.35  missing 0 | minus J: median -2.73 [-3.47, -1.54], closer in 78/106; layouts closer 15/19 (sign p 0.0192)
  model minus const: median +3.35 [+2.41, +5.01] cm, model closer in 22/106 episodes, layouts 1/19 (worse 18, sign p 7.63e-05)
    seen by the model's data: 94 episodes, model minus const median +3.39 [+2.38, +5.06], layouts closer 1/16 (sign p 0.000519)
    never seen: 12 episodes, model minus const median +2.81 [+2.04, +8.16], layouts closer 0/3 (sign p 0.25)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   3.33  mean   6.82  missing 0
  replan  median   3.41  mean   7.98  missing 0 | minus J: median +0.09 [-0.12, +0.37], closer in 49/106; layouts closer 8/19 (sign p 0.648)
  const   median   3.07  mean   6.82  missing 0 | minus J: median -0.26 [-0.43, -0.13], closer in 62/106; layouts closer 14/19 (sign p 0.0636)
  model   median   3.43  mean   5.61  missing 0 | minus J: median +0.07 [-0.23, +0.40], closer in 50/106; layouts closer 10/19 (sign p 1)
  sham    median   3.63  mean   5.79  missing 0 | minus J: median +0.00 [-0.28, +0.43], closer in 52/106; layouts closer 7/19 (sign p 0.359)
  oracle  median   3.18  mean   7.12  missing 0 | minus J: median -0.04 [-0.23, +0.04], closer in 53/106; layouts closer 10/19 (sign p 1)
  model minus const: median +0.32 [+0.00, +0.46] cm, model closer in 42/106 episodes, layouts 8/19 (worse 11, sign p 0.648)
    seen by the model's data: 94 episodes, model minus const median +0.33 [-0.07, +0.48], layouts closer 7/16 (sign p 0.804)
    never seen: 12 episodes, model minus const median +0.24 [+0.01, +1.81], layouts closer 1/3 (sign p 1)

routes: successes within 45 s / official limit
  J 68 / 56  replan 65 / 53  const 68 / 55  model 65 / 49  sham 62 / 47  oracle 66 / 46
```

## 意外

- **预测几乎全中**：两个控制器的 J、replan、const、model、sham、oracle 的停偏和差，都落在写下的区间里。唯一卡在边上的是 SONIC 的 oracle 3.00 cm（预测 3–6）。
- **ScaleBFM 上模型在没见过的布局上更差**：model 减 const 是 +1.10 cm（见过的布局 +0.59）。但没见过的只有 3 个布局、20 局。
- **常数修正让 entry 一段略差**：ScaleBFM const 比 J 多 0.30 cm，model 多 0.22，oracle 多 0.28。原因是预抓取挪近以后，下一段从更近的地方出发，反而没有 J 那么准；成败几乎不受影响。
- **主指标上各做法差不多**：ScaleBFM 有路线的局 45 秒内 J 111、const 104、model 107、oracle 108（112 局里）；官方时限内 replan 98 最多。手停得更准没有变成更多的成功局，和不抖动时一样。
- **抖动没有让常数失效**：ScaleBFM 的常数在抖动后的测试局上剩 1.74 cm，比不抖动时（1.26）只差一点。往前的抖动多半被桌沿挡掉，实际只在往后 0–6 cm、左右 ±6 cm 里变，伸手的几何变化不大。

## 结论与下一步

- **对照核心判据，两个控制器都落在"打平或输"**。
  - **ScaleBFM**：model 减 const 中位数 +0.69 cm，区间 +0.23 到 +1.14，整个在 0 以上；更近的布局 7 / 18，更远 11。
  - **SONIC**：+3.35 cm（+2.41 到 +5.01），1 / 19。
  - 所以 SIMPLE 上的结论写成："按偏差修正命令有用，常数就够"，模型在 SIMPLE 上不算贡献。
- **对照判据都没触发**：
  - replan 减 J 是 +0.00 / +0.11，不是小于 −1 cm；
  - sham 比 J 差 1.74 / 2.84 cm，挪目标这件事本身没有造成假象；
  - ScaleBFM 的 oracle 0.61 cm，不到 2 cm。

  也就是说"按偏差挪目标"这个办法本身是对的，问题在模型预测偏差不如常数准。
- **能说的话**：
  - 在 SIMPLE 的 TabletopGrasp 上，站位抖动 ±6 cm 以后，两个学出来的全身控制器（ScaleBFM、SONIC）的自身模型，预测手停在哪都不如一个固定常数准；
  - 按实测偏差挪目标能把 ScaleBFM 的手放到 0.61 cm，说明可修正的空间是有的；
  - 对比 BiGym（[bigym_endpoint_20261125](bigym_endpoint_20261125.md)），那里手臂是 PD 跟关节目标，模型赢过所有常数。
- **证据强度**：
  - 每个控制器 30 个位置 × 4 次抖动，有路线的 112 / 106 局，18 / 19 个布局；
  - 一个模型（jrefwm）；
  - 开发局两次（第一次触发流程判据，改后重跑）。
- **其他可能的解释**：
  - **模型训练时没有桌子**：训练数据从发命令那一刻起关掉了场景碰撞，SIMPLE 里桌子是在的。手或手臂碰到桌沿会带来一个模型不知道的固定偏差，常数正好把它吸收了。可以按"这一局手臂碰没碰桌子"把 model 减 const 分开看，来检验这个解释。
  - **抖动太小**：实际只在往后 0–6 cm、左右 ±6 cm 里变（往前被桌沿挡掉），伸手的几何变化不大，常数自然够用。更大的扰动下常数会不会先坏，没测。
  - **模型侧向有固定偏差**（S 那次复核看到约 +2.2 cm），抖动不会让它消失，常数没有这个偏差。
- **四类陷阱**：
  - **benchmark 选择**：只用 TabletopGrasp，因为只有它有路线；其余三个任务各做法相同，不重跑。
  - **数据泄漏**：测试布局里有 15–16 个布局在模型的训练数据里采过。这对模型有利，结果是模型仍然输，没见过的布局上输得更多。常数来自开发局，不在测试局上拟合。
  - **指标误用**：主度量照登记；entry 和主指标都照报。
  - **事后挑选**：开发局第一次触发流程判据，改流程是按登记的流程判据做的，改法在跑测试局之前提交（`abbe9993`），核心判据和预测没改。
- **下一步**：
  1. 独立复核（这条结论会写进论文，并决定 SIMPLE 上不再调模型）。
  2. 主表里 SIMPLE 这一块要带"固定常数修正"一行，S 不写成模型的贡献。
  3. 检验"训练时没有桌子"这个解释：按手臂碰桌与否分开看 model 减 const，不用新仿真。
  4. 论文里模型的精度贡献靠 BiGym（数据效率：模型不用真实局，不用模型的回归要几十局才追上），以及之后伸手几何变化更大的场景。

## 独立复核

2026-11-25，独立的 agent。它拿到的是登记的各节、预测修订、开发局和测试局的原始表、结论（"意外"一节和推理过程没给），以及 4090 上的逐局记录、代码路径。原文（英文）：

> ## Findings
>
> 1. **Every number in the write-up reproduces, and the pipeline checks pass.** The test and dev tables, the v1 dev figures quoted in the revision, the medians, the layout bootstrap, the sign tests, the seen/unseen split and the scale and fallback counts all match exactly from the rows. I found no mismatch in stance, state, plan source or constant.
>
> 2. **The model arm corrected the wrong quantity. This is the main problem, and it decides the core conclusion.**
>    - `precision.py shifts` moves the goal by `-pred`, where `pred = off[-1, palm].mean(0)`. That is `s_check.py`'s `end_off = P_h[-1, palm] - ref0[-1, palm]`: the predicted palm minus the motor's reference palm.
>    - The metric, the constant and the oracle are all measured against the look's goal. Where the reference itself puts the jaw at the end of J's plan is not the goal.
>    - That gap can be computed before sending, by kinematics of J's plan on the state at the command, with no measurement and no self model:
>      - ScaleBFM: mean [+0.40, −1.65, +0.51] cm (forward, left, up), median length 1.96 cm;
>      - SONIC: mean [−1.03, +8.62, +0.50] cm, median length 9.45 cm.
>    - The constant absorbs this gap; the model arm never tried to.
>    - The "+2.2 cm lateral bias of the model" is this gap, not a model error. Against the part it does predict (actual hand minus reference), the ScaleBFM model's error is [−0.23, −0.31, −0.48] cm (sd 1.01 / 0.72 / 0.37).
>    - The executed runs confirm it. At full scale, the model arm's actual miss has mean [0.27, −1.81, 0.33] cm against J's reference gap of [0.31, −1.57, 0.62]. Per episode they correlate 0.50 / 0.78 / 0.91 (ScaleBFM) and 0.31 / 0.60 / 0.72 (SONIC).
>    - SONIC: in J, the reference gap (+8.6 cm left) and the tracking error (−7.5 cm) nearly cancel. The model removes the tracking part and exposes the gap, which is the +3.35 cm loss.
>
> 3. **Offline, the model's full prediction beats the constant on ScaleBFM.** Full prediction here means reference end plus predicted offset. "Offline" means extrapolating each arm's outcome as J's miss plus its shift; on ScaleBFM the executed shifted arms match that within a median of 0.5 cm.
>    - ScaleBFM test, model − const: −0.58 cm [−0.81, −0.35], closer in 18/18 layouts, sign test p = 7.6e−6. Unseen layouts: 3/3, −0.54. Dev: −1.14, 8/9.
>    - It also beats a model-free per-episode baseline (reference gap plus a tracking constant frozen on dev): −0.48 cm, 18/18.
>    - SONIC: a tie, −0.13 [−1.30, +0.42], 9/19.
>    - This was not registered and has not been run, so it is not a result. But "a constant suffices; the model earns no credit" is not established for ScaleBFM.
>
> 4. **The scale mismatch does not explain the loss.**
>    - Model and const ended at different scales in 31/112 (ScaleBFM) and 27/106 (SONIC) episodes.
>    - The scale cuts helped the model slightly: executed +0.69 cm against +1.11 extrapolated at full shift.
>    - Where both arms ran at full scale: ScaleBFM +0.61 [−0.21, +1.14], 5/13 layouts; SONIC +3.23, 17/19 layouts worse.
>
> 5. **Alternative (a), training without the table: not supported.**
>    - Every ScaleBFM episode touches the table during the reach (112/112), mostly the idle left wrist at about 1.6 cm penetration, so a touched/untouched split cannot be made.
>    - On the reaching hand, the lateral error is larger without contact (−2.66 cm, n = 11) than with it (−1.88).
>    - Left-wrist penetration correlates with the lateral error at r = +0.65, but in the direction of less bias with more contact.
>
> 6. **Alternative (b), small jitter: supported for ScaleBFM, more strongly than written.**
>    - Across a position's 4 repeats, the reach vector (goal minus pelvis at the command) varies with sd 0.6 cm forward, 1.4 cm sideways and 0.1 cm vertical.
>    - Only 5 % of the forward draw and 44 % of the sideways draw reach the arm. The write-up's "back 0–6 cm, ±6 cm sideways" does not describe what happened.
>    - The model does follow the variation that exists (within-layout correlation 0.80 / 0.61 / 0.63), but that variation is about 0.9 cm against a roughly 2 cm fixed term.
>    - SONIC's reach varies a lot (sd 4.0 / 8.7 cm), mostly from its own walking error.
>
> 7. **Pregrasp precision does not reach success.** On routed test episodes, ScaleBFM J succeeds within 45 s in 111/112, against 104 for const, 107 for model and 108 for oracle. Every shifted arm has a worse entry miss than J. TabletopGrasp's entry phase re-aims from the measured hand, so this task barely rewards one-shot precision.
>
> ## Verdicts
>
> | Conclusion | Verdict |
> |---|---|
> | By the core criterion, both controllers "tie or lose" (the arm as registered) | Supports |
> | "On SIMPLE a constant suffices; the model earns no credit" | ScaleBFM: does not support (the arm omitted a known kinematic term; offline the full prediction wins 18/18). SONIC: insufficient evidence (offline tie) |
> | "Moving the goal by the offset is right; the problem is the model predicts the offset worse than a constant" | Does not support. It predicts its own target well (mean error 1.29 cm vs the constant's 2.01); what was missing is the reference gap |
> | Both self models predict where the hand stops less accurately than a constant | ScaleBFM: does not support. SONIC: insufficient evidence |
> | replan / sham / oracle controls not triggered; oracle 0.61 cm shows room to correct | Supports |
> | The model has a fixed lateral bias that jitter cannot remove | Does not support as attributed (it is the reference's gap, −1.65 cm, not the model's −0.31) |
> | Alternative: no table in training | Does not support |
> | Alternative: jitter too small | Supports (ScaleBFM), and stronger than stated |
> | Pitfall: data leakage — constant from dev, not refit | Supports (verified below) |
> | Pitfall: post-hoc selection — the revision was legitimate | Supports (details below) |
> | Pitfall: metric misuse — "nothing misused" | Does not support. The model arm's target (reference) differs from the metric's (goal), and the metric does not predict success |
>
> ## How I verified it
>
> - **Integrity, all 218 test and 99 dev episode-repeats.**
>   - Every arm's row has the same `jitter`, `jitter_rep`, walk, posture and pelvis state at the command, and goal as J.
>   - Each arm ran the plan from the scale directory that `pick.jsonl` recorded, and fallbacks ran J's own plan. Zero mismatches.
> - **Model shift.**
>   - Recomputed for every test episode-repeat from `state_*.npz` and J's plan with `s_check.reference`, `model_input` and `VP.predict`: largest difference from `shifts.jsonl` is 0.005 cm.
>   - The saved state matches the JR row, and the plan comes from the same repeat.
>   - The rebuilt reference matches the filmed motor reference within 0.11 cm, so the reference gap is what the motor actually received.
> - **Constant.**
>   - Recomputed exactly from the dev JR rows (levels 3–4, no jitter, n = 11 / 15).
>   - The files date from 15:39Z, before dev v1 at 15:55Z, with identical md5 in the dev v1, dev v2 and test copies.
>   - Test used one constant throughout (n = 11 / 15); dev used the leave-one-layout version (6–10 / 12–15 episodes).
> - **Revision.**
>   - It was triggered by the registered fallback rule and committed in `abbe9993` at 16:52Z, before the test started at 18:03Z.
>   - The same retry code applies to const, model, sham and oracle, and the write-up discloses that the v1 dev numbers had been seen.
>   - The scale cut does depend on the outcome (IK fails when the shift pushes into the table margin; const cut to 2/3 even scored 0.61 cm), but it applied equally and did not hurt the model.
> - **Provenance nit.** `run.log` says the remote checkout is at `bc426d5`, not the registered `d308277a`. Every file in the run copies (`asrun/`) has the same md5 as the Mac working tree, so the code that ran is confirmed.
>
> ## Experiments that could overturn "a constant suffices"
>
> 1. **Re-run with the corrected model arm (primary).** On the test episodes, add an arm shifted by minus (reference end jaw + predicted offset − goal). In `shifts()` that is `pred + 100*to_head(end[0] - goal, yaw)`, with the goal taken from J's `goal_mm`.
>    - Run it beside const and a model-free arm (reference gap plus a tracking constant frozen on dev).
>    - Register the same criteria. Do dev first to check IK failures; the shift sizes are similar (median 4.54 vs 4.62 cm).
>    - Offline prediction: ScaleBFM about −0.6 cm and a win; SONIC a tie.
> 2. **Make the reach geometry really vary.** Enforce the stance offset after the table-edge clamp, or vary target position, pelvis height and heading over a range where the constant's residual measurably grows. Register that the constant degrades while the model does not.
> 3. **Check the earlier S result.** `s_check.py`'s round-0 end check uses the same formula, so the earlier S comparison (const 1.26 vs model 2.04 cm) may carry the same omission.
>
> Temporary files on the remote (`/tmp/rv_prec_9f3a_*`) and on the Mac have been deleted. Nothing was written to the output directories and no GPU was used: the model was recomputed on CPU with `CUDA_VISIBLE_DEVICES=""`.
>
> The code in question is in `/Users/zed/code/loco/scripts/experiments/simple_eval_20261112/`: `precision.py` lines 128–142 (the shift) and `s_check.py` line 303 (`end_off`).

### 复核后的更正（2026-11-25）

数全部重算对上；站位、状态、路线来源、常数都核对过，没有错位；中途的改法合法。**但复核找到一个设计上的错，它决定了核心结论**。

1. **model 这一组挪错了量。** 这个错出在我设计 S 的终点检查时：按 J 自己路线的终点比，而不是按目标比。这次 `precision.py` 照搬了。
   - `precision.py shifts` 挪的是 `−pred`，`pred` 是"预测的手掌减参考的手掌"（`s_check.py` 第 303 行的 `end_off`）。
   - 停偏、常数、oracle 都是对开局那一眼的目标量的，而参考在 J 路线终点把夹爪放在哪，本身就不在目标上。
   - **这段"参考离目标"的差**，在发命令时就能用 J 的路线和发令时刻的状态算出来，不用测、不用模型：ScaleBFM 平均 [+0.40, −1.65, +0.51] cm、长度中位数 1.96 cm；SONIC 平均 [−1.03, +8.62, +0.50] cm、9.45 cm。原因是路线在开局那一眼就规划好了，机器人实际站的位置（抖动、走路误差）和规划时不同。
   - 常数把这段差吸收了，model 这一组没管它。
2. **撤回"模型侧向有约 +2.2 cm 的固定偏差"**（这里和 S 那次复核都这样写过）。那正是参考离目标的差（−1.65 cm），不是模型的错。
   - 模型真正预测的那部分（实际的手减参考），ScaleBFM 的误差只有 [−0.23, −0.31, −0.48] cm。
   - SONIC 上参考离目标的差（左 8.6 cm）和跟踪误差（−7.5 cm）正好差不多抵消，模型去掉了后者，前者就露出来了，所以输 3.35 cm。
3. **核心判据照登记的做法成立，但读法要改。**
   - 照登记，"model 这一组"两个控制器都打平或输，这一条照报。
   - 但"SIMPLE 上常数就够，模型不算贡献"不成立：
     - **ScaleBFM 证据反对**：复核离线外推，把参考离目标的差加进去以后，model 减 const 是 −0.58 cm（−0.81 到 −0.35），18 / 18 个布局更近；也赢不用模型的逐局基线（参考离目标的差加开发局冻结的跟踪常数）−0.48 cm，18 / 18。
     - **SONIC 证据不足**：离线是打平（−0.13）。
   - 离线外推不是登记的结果，要重跑。
4. **备选解释"训练时没有桌子"不成立**：ScaleBFM 每一局伸手时都碰桌（多半是不动的左手腕），分不开碰与不碰；右手侧向误差在不碰桌的局上反而更大。
5. **"抖动太小"成立，而且比写的更小。** ScaleBFM 同一位置 4 次抖动，伸手向量前后只变 0.6 cm、左右 1.4 cm、上下 0.1 cm。往前的抖动只有 5 %、往侧的只有 44 % 传到手臂上。原文"往后 0–6 cm、左右 ±6 cm 在变"不对。
6. **预抓取更准没有变成更多成功**：ScaleBFM 有路线的局，45 秒内 J 111 / 112、const 104、model 107、oracle 108；挪过目标的各组 entry 都比 J 差。TabletopGrasp 的 entry 从实测的手重新瞄，这个任务几乎不奖励一次命令的精度。
7. **S 那次（selfmodel_s_20261124）用的是同一个算式**，所以那次复核的"常数 1.26 cm 比模型 2.04 cm 准"也可能带着同样的遗漏。已在那份条目里加注。
8. **下一步**：
   - 另登记一次：在这次测试局和开发局的同一批局上（同样的抖动、同样的 J 和状态），加三组新挪法：
     - 只去掉参考离目标的差；
     - 再加开发局冻结的跟踪常数；
     - 模型的完整预测（参考离目标的差加模型预测的偏差）。
   - 核心判据是完整预测对 const、对"参考离目标的差加跟踪常数"都要赢。
   - 另外，让伸手几何真的变（桌沿限位之后再加站位扰动，或换目标位置、骨盆高度），作为之后的一步。
