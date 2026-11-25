# selfmodel_s_20261124：S 这一行（J 加自身模型）在 SIMPLE 测试局上，和 A、J 逐局配对

- 日期：2026-11-24（设计、预测、判据：开发局选完以后、测试局任何 S 运行以前写下，单独提交）
- 机器 / GPU：4090。和 [simple_rgbd_20261124](simple_rgbd_20261124.md) 的 A、J 一样：ScaleBFM 在 1 号卡，SONIC 在 2、3 号卡，每个任务一个进程。
- 代码版本：`selfwm` @ `99c0d3fd`（4090 的检出 `bc426d5` 加未提交的文件。跑到的文件和 Mac 逐个 md5 相同，只有 `simple_bench.py` 例外：4090 那份和 RGB-D 那次 J 跑的是同一份（md5 abc246fd），和 Mac 的 `99c0d3fd` 只差模块开头的帮助文字，代码相同），跑的时候抄进输出目录的 `asrun*/`。
- 配置：
  - SIMPLE 官方评测局，四个抓取任务各 30 局，感知栏；
  - 开关 `sensor=rgbd`，签名 `simple/perception-rgbd/pose-odometry/root-odometry/<控制器>`；
  - ScaleBFM 和 SONIC（`REACH_SONIC_NETS=gpu`）都不改参数；
  - 自身模型是 selfmodel_jrefwm_20261124 的两份，不重训；
  - 不用仿真真值，不用伺服和查表。
- 关联：
  - TODO 38.10.4；
  - [selfmodel_s_dev_20261124](selfmodel_s_dev_20261124.md)：开发局上选的检查方式和半径；
  - [simple_rgbd_20261124](simple_rgbd_20261124.md)：同一批局上 A、J 的数，这次配对的对象；
  - 用户 2026-11-24："只需要跑对照中的 A，J，S 就行了"。

## 要回答的问题

在新的默认组合上，给 J 加上该控制器自己的自身模型做检查和修正（S），主指标比 J、比直线伸手 A 多还是少。

S 和 J 的区别只在发令以前：模型预测整条手臂会走到哪，查终点偏差和离点云的间隙，不过关就让 cuRobo 带着修正重新规划，修不好就放弃。

## 做法

**开发局上选的（[selfmodel_s_dev_20261124](selfmodel_s_dev_20261124.md)，照事先写死的选法）：S_end，只开终点检查。**
- 两个控制器开发局 45 秒内成功数之和：S_end 14（ScaleBFM 6、SONIC 8），S_clear 0，S_both 0；J（开发局上是 JR）20（10、10）。
- **开发局上 S 没有赢 J**：三种都比 J 少，触发了开发局的判据"三种都比 J 少"。照判据，这里预测 S 在测试局上不比 J 多。
- S_clear 和 S_both 在开发局上放弃了全部有路线的局（J 自己的参考就贴着点云，离保留的体素 −1.7 到 +1.2 cm，而半径 4.06 / 16.99 cm），触发"放弃一半以上"的判据，记在这里；它们不在测试局上跑。
- 参数：`--tau-end 2 --delta 1e9`，最多 2 轮。间隙检查不开，所以开发局标的半径（ScaleBFM 4.06 cm、SONIC 16.99 cm）在测试局上不起作用，只在度量 4 里报。
- 开发局上的读法：修过的局（挪目标、重新规划）S 和 J 结果逐局相同（6 / 6、10 / 10）；S 少的分全在放弃的局里（挪过的目标 IK 无解，10 / 11 是 IK_FAIL）。
- 开发局跑的时候改了 `s_check.py` 两处（重新规划失败的文件、修过的候选的终点检查对原目标），见开发局条目的"预测修订"；测试局用改过的版本，和开发局相同。

1. **JR**：测试局上 J 带录像重跑一遍（`bench_via.sh` 步骤 JR，四个任务各 30 局，原来的输出目录 `bench_rgbd_20261124/<控制器>/`），录下发令时刻的状态。JR 的结果应当和 J 逐局相同。
2. **检查和修正**：`s_run.sh <控制器目录> <控制器> <模型> <控制器目录>/S 2 --tau-end 2 --delta 1e9`，输出在 `bench_rgbd_20261124/<控制器>/S/`。JR 已经跑完，和 J 逐局相同（[selfmodel_s_jr_20261124](selfmodel_s_jr_20261124.md)，240 / 240）。
3. **S**：`bench_via.sh` 步骤 S，四个任务各 30 局。没有路线的局退回直线伸手，和 J 一样。放弃的局计为失败。
4. **计分**：
   - `bench_jref_table.py` 给 A、J 的数（已有），`s_check.py table` 给 A、J、S 并排，S 放弃的局算失败；
   - J 有路线的局上 S 对 J 逐局配对。

## 度量

1. **主指标**：SIMPLE 的成功，四个任务各 30 局，45 秒内和官方时限内都报，重点是 TabletopGrasp。
2. 有路线的局上，S 对 J 逐局配对：45 秒内和官方时限内的成功数之差、只有一边成功的局数。S 第一轮过关、重新规划后过关、放弃各多少局。
3. 流程核对：
   - JR 和 J 逐局相同的局数；
   - S 第一轮就发 J 的路线的局，S 和 J 相同的局数；
   - 重推参考和录像的最大差。
4. 模型在测试局上的预测误差：在 JR 上按开发局的做法算扫掠误差的分位数，以及开发局半径罩住的比例。这一条只报，不参与选择。
5. 起始状态在不在训练范围里：`jref_data.py range`，JR 发令时 29 个关节超出训练数据 0.5–99.5 % 范围的个数。

## 预测（启动前写，首次启动后锁定）

测试局上 J 有路线的局：ScaleBFM 28 局（全在 TabletopGrasp），SONIC 29 局（28 局 TabletopGrasp、1 局 XMovePick）。J 在这些局上 45 秒内 / 官方时限内：ScaleBFM 28 / 24；SONIC TabletopGrasp 的 28 局 14 / 11，XMovePick 那一局 0 / 0（从 J 的行按有没有路线文件数出来）。其余局 S 和 J 一样退回直线伸手，结果应当和 J 相同。

- **流程**：
  - 重推参考和录像的最大差不超过 0.2 cm（开发局 0.055 / 0.081 cm）；
  - 没有路线的局，S 和 J 逐局相同（ScaleBFM 92 局，SONIC 91 局）。
- **第一轮过关的局**：每个控制器 0 到 3 局（开发局 0 / 11、0 / 15）。
- **放弃的局**：ScaleBFM 28 局里 6 到 20 局，SONIC 29 局里 5 到 18 局（开发局 45 %、33 %，区间放宽，测试局的目标和开发局不同）。
- **修过并过关的局**：S 的 45 秒内成败和 J 相同的局不少于 90 %（开发局 16 / 16）。
- **有路线的局上 S − J**：
  - ScaleBFM：45 秒内 −6 到 −20，官方时限内 −4 到 −18（J 在这 28 局上 45 秒内全成，放弃一局几乎就少一局）；
  - SONIC：45 秒内 −1 到 −10，官方时限内 0 到 −8（J 只成一半，放弃的局约一半本来就失败）；
  - 两个控制器、两个时限都是 S 不比 J 多（开发局的判据要求这样预测）。
- **主指标（四个任务，45 秒内 / 官方时限内）**：
  - ScaleBFM：S 75–89 / 63–77（J 95 / 81，A 83 / 73）；TabletopGrasp S 10–24 / 8–22（J 30 / 26，A 18 / 18）；
  - SONIC：S 24–33 / 12–20（J 34 / 20，A 29 / 18）；TabletopGrasp S 5–14 / 4–12（J 15 / 12，A 10 / 10）。
- **模型在测试局上的预测误差**（JR 录像，`s_check.py calib`，按开发局的做法）：方向预测器扫掠误差的 90 % 分位，ScaleBFM 3 到 7 cm，SONIC 8 到 22 cm；原固定半径罩住的比例 ScaleBFM 0.2 到 0.6，SONIC 0.5 到 0.9。理由：开发局 4.06 / 16.99 cm、0.364 / 0.733，测试局的布局更容易，误差不应更大。
- **起始状态**：`jref_data.py range`，发令时 29 个关节超出训练数据 0.5–99.5 % 范围的个数，中位数 0 到 1，最大不超过 8（jrefwm 那次 SIMPLE 局上中位数 0、最大 7）。

## 判据（什么结果会让我改读法，下一步做什么）

- **流程没过**：重推参考差超过 0.5 cm，或者没有路线的局里 S 和 J 有一局不同。先查代码，不报 S 的数。
- **S 不比 J 少**（有路线的局上，任一控制器任一时限 S − J ≥ 0）：和开发局的读法相反，说明开发局 20 局的结论不可靠；照实报，并在两个控制器上看修过的局里 S 多赢的是哪些。
- **S 比 J 少，落在预测里**：读法是"在 SIMPLE 的 TabletopGrasp 上，自身模型的终点检查和放弃规则只丢分"。下一步不再在 SIMPLE 上调 S，转去障碍物位置会变的任务（TODO 38.10.5，BiGym 2.0 的抽屉）；同时用录像比手实际停下的位置，分开"模型的偏差预测是假的"和"抓取阶段本来就吸收伸手误差"两种解释。
- **修过并过关的局里 S 和 J 成败不同的超过 10 %**：修法本身在改变结果（好或坏），单独报这些局，看是不是模型说对了的那些。
- **预测误差的 90 % 分位超过开发局的 1.5 倍**：模型在测试局上更差，S 的检查依据不可靠，写进结论。

## 预测修订

## 结果（贴原始输出，不转述）

输出在 4090 的 `outputs/simple_eval_20261112/bench_rgbd_20261124/<控制器>/`：`S/`（检查、重新规划、挑选、表）、`rows/S_*.jsonl`、`films_S/`、`calib_test.json`、`range_test.txt`；启动脚本 `run_s.sh`。

`phase.log`：

```
S start 2026-11-24T14:22:10Z bc426d5 + uncommitted 4106eba5 13413386 967b61e0 abc246fd
scalebfm model on the test films: {"reference": {"n": 28, "q50": 8.08, "q80": 10.58, "q90": 12.0, "q95": 12.42, "max": 12.97, "palm_median": 6.02, "inside_fixed": 0.929, "fixed_cm": 12.4}, "direction": {"n": 28, "q50": 3.13, "q80": 3.54, "q90": 3.73, "q95": 3.89, "max": 4.11, "palm_median": 2.88, "inside_fixed": 0.393, "fixed_cm": 2.87}} | range: joints at the command outside the training paths' 0.5-99.5 % band (28 reaches): median 0.0, mean 1.00, max 9, counts [18, 5, 2, 1, 0, 0, 0, 1, 0, 1]
sonic model on the test films: {"reference": {"n": 29, "q50": 17.41, "q80": 23.93, "q90": 27.94, "q95": 30.63, "max": 33.68, "palm_median": 11.52, "inside_fixed": 0.759, "fixed_cm": 22.42}, "direction": {"n": 29, "q50": 11.37, "q80": 17.71, "q90": 22.15, "q95": 25.53, "max": 30.17, "palm_median": 8.48, "inside_fixed": 0.379, "fixed_cm": 10.29}} | range: joints at the command outside the training paths' 0.5-99.5 % band (29 reaches): median 0.0, mean 0.38, max 3, counts [21, 6, 1, 1]
scalebfm S checks: [14:24:54] chosen: 21 via files, given up 7
sonic S checks: [14:25:12] chosen: 23 via files, given up 6
scalebfm S done 2026-11-24T14:29:15Z
sonic S done 2026-11-24T14:31:47Z
S all done 2026-11-24T14:31:47Z
```

各轮（`S/s_pipeline.log`）：

```
scalebfm  state: 28 episodes with J's plan | round 0: 28 checked, 0 passed | replan 1: 28 rows, planned 21 | round 1: 21 checked, 21 passed | chosen: 21 via files, given up 7
sonic     state: 29 episodes with J's plan | round 0: 29 checked, 0 passed | replan 1: 29 rows, planned 23 | round 1: 23 checked, 21 passed | replan 2: 2 rows, planned 2 | round 2: 2 checked, 2 passed | chosen: 23 via files, given up 6
scalebfm  arm S (GPUs 1): 120 rows, 0 failed processes
sonic     arm S (GPUs 2 3): 120 rows, 0 failed processes
```

表（`S/table.txt`，`s_check.py table <控制器>/S <控制器> --arm S --j-arm J`；三列是 A、J、S，45 秒内 / 官方时限内）：

```
== scalebfm
successes within 45 s / official limit (S: 7 given up, counted failures)
  task                             A           J           S
  XMovePickTeleop (30)       28 / 28     28 / 28     28 / 28
  XMoveBendPickTeleop (30)     27 / 27     27 / 27     27 / 27
  BendPickMP (30)             10 / 0      10 / 0      10 / 0
  TabletopGraspMP (30)       18 / 18     30 / 26     23 / 22
  all (120)                  83 / 73     95 / 81     88 / 77

episodes J ran a plan on: 28; S kept J's plan 0 (rows identical to J's: 0), replanned and passed 21, gave up 7
  45 s successes: J 28, S 21, S - J -7 (95 % interval -12 to -3), only S 0, only J 7
  official limit successes: J 24, S 20, S - J -4 (95 % interval -8 to -1), only S 0, only J 4
== sonic
successes within 45 s / official limit (S: 6 given up, counted failures)
  task                             A           J           S
  XMovePickTeleop (30)         5 / 5       5 / 5       5 / 5
  XMoveBendPickTeleop (30)       3 / 3       3 / 3       3 / 3
  BendPickMP (30)             11 / 0      11 / 0      11 / 0
  TabletopGraspMP (30)       10 / 10     15 / 12     15 / 10
  all (120)                  29 / 18     34 / 20     34 / 18

episodes J ran a plan on: 29; S kept J's plan 0 (rows identical to J's: 0), replanned and passed 23, gave up 6
  45 s successes: J 14, S 14, S - J +0 (95 % interval -6 to +5), only S 4, only J 4
  official limit successes: J 11, S 9, S - J -2 (95 % interval -7 to +4), only S 3, only J 5
```

逐局（同一个文件，节选；ScaleBFM 放弃的 7 局 J 都成功）：

```
scalebfm 放弃：TabletopGraspMP_L0_e1, L1_e1, L1_e5, L2_e0, L2_e1, L2_e4, L2_e8（chosen None J 1 S 0）
sonic    放弃：TabletopGraspMP_L0_e1 (J 0), L0_e2 (J 0), L0_e8 (J 1), L2_e0 (J 0, round 0 end 81.45 cm), L2_e1 (J 0), L2_e9 (J 0)
```

流程核对（从 `S/state.jsonl` 和 `rows/{J,S}_*.jsonl` 数，脚本在 Mac 的作业目录 `tmp/s_checks.py`）：

```
scalebfm state 28 rebuilt vs film max 0.068 cm; dur equal 28 path equal 28
scalebfm unrouted 92 S differs from J on 0 [] | S rows 120
sonic state 29 rebuilt vs film max 0.161 cm; dur equal 29 path equal 29
sonic unrouted 91 S differs from J on 0 [] | S rows 120
```

判据里写的下一步：手实际停下的位置（行里 `stages` 的 pregrasp 一段 `err_cm`：伸手结束、停稳后的夹爪中心离开局那一眼的预抓取点多远；entry 一段是 `palm_cm`），只看修过并跑了的局（脚本 `tmp/s_err.py`）：

```
scalebfm: 21 repaired episodes ran; pregrasp miss median J 4.71 S 1.73 cm, S closer in 19; entry miss median J 2.17 S 2.82 cm, S closer in 5
sonic: 23 repaired episodes ran; pregrasp miss median J 6.33 S 10.27 cm, S closer in 5; entry miss median J 3.28 S 3.61 cm, S closer in 8
   TabletopGraspMP_L0_e6    J 0 S 1 | pregrasp miss J 18.71 S 10.27 | entry miss J 8.77 S 2.03 | predicted end 9.39 cm
   TabletopGraspMP_L0_e7    J 0 S 1 | pregrasp miss J 9.68 S 10.27 | entry miss J 1.08 S 2.29 | predicted end 7.89 cm
   TabletopGraspMP_L0_e9    J 1 S 0 | pregrasp miss J 4.51 S 13.38 | entry miss J 3.28 S 9.14 | predicted end 7.04 cm
   TabletopGraspMP_L1_e2    J 0 S 1 | pregrasp miss J 12.90 S 13.09 | entry miss J 13.80 S 6.44 | predicted end 7.49 cm
   TabletopGraspMP_L1_e7    J 1 S 0 | pregrasp miss J 12.88 S 21.70 | entry miss J 4.70 S 5.02 | predicted end 11.8 cm
   TabletopGraspMP_L1_e9    J 1 S 0 | pregrasp miss J 8.40 S 30.88 | entry miss J 3.46 S 13.16 | predicted end 5.59 cm
   TabletopGraspMP_L2_e2    J 0 S 1 | pregrasp miss J 6.33 S 16.88 | entry miss J 4.95 S 6.02 | predicted end 7.6 cm
```

## 意外

逐条对预测：
1. 流程：重推参考差 0.068 / 0.161 cm（预测不超过 0.2）；没有路线的局 S 和 J 逐局相同 92 / 92、91 / 91。符合。
2. 第一轮过关：0 / 28、0 / 29，在 0–3 内。
3. 放弃：ScaleBFM 7 / 28（6–20 内），SONIC 6 / 29（5–18 内）。
4. 修过并过关的局，S 和 J 成败相同的比例：ScaleBFM 21 / 21，符合；SONIC 16 / 23（70 %），预测不少于 90 %，不符合。
5. 有路线的局上 S − J：ScaleBFM −7 / −4，在预测内；SONIC 45 秒内 0，预测 −1 到 −10，不符合；官方时限内 −2，在预测内。
6. 主指标：ScaleBFM S 88 / 77（预测 75–89 / 63–77，官方时限那一格压在上沿），TabletopGrasp 23 / 22（10–24 / 8–22，同样压在上沿）；SONIC S 34 / 18，45 秒内超出预测上限 33，TabletopGrasp 15 / 10，45 秒内超出上限 14；官方时限内都在区间内。
7. 测试局上的预测误差（方向预测器 90 % 分位）：ScaleBFM 3.73 cm（3–7 内），SONIC 22.15 cm，略超上限 22。原固定半径罩住的比例：ScaleBFM 0.393（0.2–0.6 内），SONIC 0.379，低于下限 0.5。
8. 起始状态：中位数都是 0；最大值 ScaleBFM 9（预测不超过 8，有 1 局 9 个关节超出），SONIC 3。
9. 没想到的：
   - ScaleBFM 的修法真的把手放准了：停稳后离预抓取点的中位数从 4.71 cm 降到 1.73 cm，21 局里 19 局更近。但下一段（伸进抓取位置）会自己再纠正一次，entry 的误差 J 2.17、S 2.82 cm，所以成败一局都没变。这说明 ScaleBFM 的自身模型对终点的预测是对的。
   - SONIC 正好相反：修完离得更远（6.33 → 10.27 cm，只有 5 / 23 局更近），模型对 SONIC 的终点预测在这些局上是错的。成败不同的 7 局里，S 赢的 4 局和输的 3 局，修完后的距离都没有一致的方向。
   - SONIC 有一局第 0 轮预测终点偏 81.45 cm（L2_e0），模型在这一局上明显出界。

## 结论与下一步

- **主指标（RGB-D 默认组合，四个抓取任务，45 秒内 / 官方时限内）：**
  - ScaleBFM：A 83 / 73，J 95 / 81，S 88 / 77；TabletopGrasp A 18 / 18，J 30 / 26，S 23 / 22。
  - SONIC：A 29 / 18，J 34 / 20，S 34 / 18；TabletopGrasp A 10 / 10，J 15 / 12，S 15 / 10。
  - 另外三个任务三种做法逐局相同（没有路线，都退回直线伸手；SONIC 的 XMovePick 那一局 J、S 都失败）。
- **对照判据：**
  - 流程判据没有触发。
  - ScaleBFM 落在"S 比 J 少，落在预测里"：S − J 45 秒内 −7（区间 −12 到 −3），官方时限内 −4（−8 到 −1），丢的全是放弃的 7 局，修过的 21 局和 J 逐局相同。
  - SONIC 触发了两条：
    - "S 不比 J 少"（45 秒内 S − J = 0，区间 −6 到 +5）；
    - "修过并过关的局里 S 和 J 成败不同的超过 10 %"（7 / 23）。照判据单独看了这 7 局：修完后手离预抓取点的距离和成败没有一致关系，S 赢的局并不是模型说对了的局。SONIC 的 0 差是噪声的结果：复核已指出 SONIC 上同一目标的两局常常结果不同，命令一变，结局就跟着随机变。官方时限内 S 仍少 2 局。
  - "预测误差的 90 % 分位超过开发局的 1.5 倍"：没有触发（3.73 对 4.06，22.15 对 16.99 × 1.5 = 25.5）。
- **读法：**
  1. 在 SIMPLE 上，S 没有比 J 多拿一局。ScaleBFM 少 7 局，全部来自"修不好就放弃"；SONIC 持平，是噪声。
  2. 自身模型本身对 ScaleBFM 是有用的：按它挪目标，伸手停下的位置误差降到原来的约三分之一（中位数 4.71 → 1.73 cm）。只是 SIMPLE 的抓取流程在预抓取点之后还有一段自己的纠正，把这个好处吃掉了，所以成功局数看不出来。
  3. 对 SONIC，模型的终点预测在测试局上不可靠（误差 90 % 分位 22 cm，原固定半径只罩住 38 %），按它修会把手推得更远。
- **证据强度：** 每个控制器一次运行，120 局，有路线的 28 和 29 局；TabletopGrasp 是同一个薯片罐，0、1 级布局相同，有效样本约 10（RGB-D 那次复核的结论）。伸手停下位置那一条是 21 局里 19 局同向，符号检验 p < 0.001，但同样受布局重复的影响（0、1 级同一布局的两局算两次）。
- **其他可能的解释：**
  - ScaleBFM 上 pregrasp 误差变小，也可能是重新规划换了一条更好的路线，而不是挪目标起的作用。要分开，得在同一个挪过的目标上不用模型、只重新规划一次做对照（"只重新规划"），这次没做。
  - SONIC 的成败变化也可能部分来自修法：修过的局终点不同，抓取起点不同。但 S 赢的局里修完离得更远的也有（L1_e2、L2_e2），不支持"修得好才赢"。
- **四类陷阱：**
  - benchmark 选择：四个任务各 30 局全报；S 只在有路线的局上和 J 不同，这一点写明了。
  - 数据泄漏：**有一种要写明。** 自身模型的训练数据是在 SIMPLE 这 30 局的站位上采的（jrefwm 的设计：控制器在这些站位上执行随机的 cuRobo 路线，留出 1 / 3 的布局）。测试局有路线的局里，ScaleBFM 19 / 28、SONIC 20 / 29 是训练见过的布局（`range_test.txt` 的 seen / held-out 两行）。模型见过的是这些站位下身体怎么动，不是这些局的结果；但它在测试局上的误差因此可能偏小。S 的选择和参数只在开发局（训练没见过的布局）上定，测试局一次没调。
  - 指标误用：两个时限都报；SONIC 在 45 秒内持平、官方时限内少 2，两个都写了。
  - 事后挑选：一次运行、一种 S，就是开发局选的那种；没有在测试局上看了结果再换。手停下位置的比较是判据里事先写好的下一步，改用行里已有的 `stages` 字段，没有挑局。
- **下一步：**
  1. 照判据：不再在 SIMPLE 上调 S，转去障碍物会变、伸手精度直接决定成败的任务：BiGym 2.0 的抽屉和伸手。BiGym 的伸手任务成败就是手到不到目标 5 cm 以内，没有后面那段纠正，正好能检验第 2 条读法。
  2. "修不好就放弃"在 SIMPLE 上只丢分。换成"修不好就发 J 原来的路线"，S 在 ScaleBFM 上就会和 J 持平；这是新做法，要另登记、先在开发局上跑。
  3. 补"只重新规划"的对照，分开挪目标和换路线的作用。

## 独立复核

2026-11-24，独立的 agent。它拿到的是做法、度量、预测、判据、原始结果和结论（"意外"一节和推理过程没给），以及 4090 上的逐局记录、代码路径和开发局条目的"预测修订"。原文（英文）：

> **Review of selfmodel_s_20261124 (arm S on SIMPLE's test episodes)**
>
> Every number in the conclusions reproduces from the raw rows. Two of the readings go further than the data allow. The "the self model is useful for ScaleBFM" reading fails against a baseline the write-up never ran: a fixed offset correction with no model does as well or better. "SONIC's 0 difference is noise" hides harm the model caused, and the official-limit loss on SONIC comes from timing, not reach accuracy. The two mid-run code fixes were legitimate.
>
> ## Main findings
>
> 1. **All numbers match, with no mismatch in what is reported.**
>    - Per-task A/J/S counts at both limits match. So do the paired S−J on routed episodes (ScaleBFM −7 / −4, SONIC 0 / −2), the given-up lists, and only-S / only-J counts.
>    - S equals J on all unrouted episodes (92 and 91). J equals JR on 120/120 per controller.
>    - Rebuilt reference against the film: max 0.068 and 0.161 cm.
>    - Pregrasp miss medians: ScaleBFM J 4.71 → S 1.73 cm (S closer in 19/21); SONIC 6.33 → 10.27 cm (S closer in 5/23). Entry miss medians: 2.17 → 2.82 (5/21) and 3.28 → 3.61 (8/23).
>    - The seven SONIC episodes and the calib and range figures all match.
>    - Only cosmetic difference: my bootstrap seed gives SONIC [−6, +6] / [−7, +3], against the write-up's [−6, +5] / [−7, +4].
>
> 2. **A fixed offset beats the self model on ScaleBFM, so 3a is not supported as stated.**
>    - I took J's mean pregrasp miss on the 11 development episodes, [−3.17, +0.10, +3.36] cm in the heading frame (no model, could have been registered in advance).
>    - On the 21 repaired test episodes, correcting by that constant would leave a median miss of 1.26 cm. The model leaves 2.04 cm, and is better in only 7/21.
>    - The model has a sideways bias of about +2.2 cm (+2.16 cm on the development episodes), and every S hand ends 0.9–3.8 cm to the side. It does track the per-episode variation forward-back and up-down (correlation 0.78 and 0.94), but sideways the correlation is −0.36.
>    - Why a constant works: all 30 TabletopGrasp pregrasp points sit in roughly a 9 × 10 cm patch, and the robot's standing spot is placed relative to the target. The reach the robot makes barely changes from episode to episode.
>    - So "moving the goal brings the hand closer" holds. "The self model's end prediction is what does it" does not.
>
> 3. **The replan-only confound the author names is mostly ruled out by data already on disk.**
>    - S's actual hand position equals J's actual hand position plus the commanded goal shift, to within a median 0.58 cm (ScaleBFM; 2.42 cm on SONIC).
>    - So replanning itself contributes at most about 0.6 cm. The improvement is explained by how well the shift matches J's own miss.
>
> 4. **The entry-absorption reading (3b) cannot be told apart from a ceiling.**
>    - J already succeeds 28/28 within 45 s on ScaleBFM's routed episodes, so S had nothing to gain there.
>    - Official-limit failures are timing: J's success times cluster at 15.2–16.1 s against a 16 s limit.
>    - Consistent with absorption: pooling all arms on TabletopGrasp, ScaleBFM's 45-s success is 100% for pregrasp misses of 0–14 cm (n=71), and the entry miss median stays flat around 2.3 cm. S's better pregrasp gave a *worse* entry (closer in only 5/21, two-sided sign test p = 0.027).
>    - SONIC goes the other way: success is 0.56–0.73 below a 9 cm pregrasp miss, 0.38 at 9–14 cm, and 1/18 at 14 cm or more. Its entry miss tracks the pregrasp miss (rank correlation 0.67), so absorption does not generalise.
>
> 5. **"SONIC's 0 difference is noise" understates harm the model caused.**
>    - All three S losses at 45 s are episodes where S's shift sent the hand far off: 13.4 / 21.7 / 30.9 cm, against J's 4.5 / 12.9 / 8.4. That contradicts the write-up's claim that distance and outcome show no consistent relation, at least for the losses.
>    - Of the four S wins, only L0_e6 is explained by S being closer (18.7 → 10.3 cm). The other three look like noise.
>    - Noise is real: on SONIC, J's same-layout pairs (L0_eX against L1_eX) agree on only 3/10, against 10/10 on ScaleBFM.
>    - Net zero therefore looks like harm plus noisy wins, not pure noise.
>
> 6. **SONIC's official-limit −2 is systematic, not noise.** Four of the five episodes where only J succeeds within the official limit succeed for both within 45 s. In each, S's replanned path is 0.8–2.0 s longer, which pushes S's success time to 16.24–16.60 s.
>
> 7. **What S actually was in this run: "always shift; give up if the planner fails".**
>    - Round 0 passed 0/28 and 0/29; the smallest predicted offset, 3.86 cm, is already above the 2 cm threshold.
>    - After the second fix, a repaired candidate is checked by the same model that made the shift, so it passes almost automatically (21/21 and 21/23 in round 1).
>    - Every give-up came from cuRobo failing on the moved goal (ScaleBFM: 6 IK_FAIL and 1 FINETUNE_TRAJOPT_FAIL; SONIC: 6 IK_FAIL), never from the model rejecting a plan.
>    - Three of ScaleBFM's give-ups are the episodes where J was already most accurate (1.44, 1.45, 1.72 cm), but the model predicted 4.1–4.5 cm.
>    - Given-up episodes physically ran a straight reach. Two of ScaleBFM's seven (L1_e5, L2_e8) and one of SONIC's six (L0_e1) succeeded but were scored as failures, as registered.
>
> 8. **Missed predictions not acknowledged in the conclusions shown:**
>    - SONIC error at 90%: 22.15 cm, above the predicted 8–22.
>    - SONIC fixed-radius coverage: 0.379, below the predicted 0.5–0.9.
>    - ScaleBFM out-of-range joints: max 9, above the predicted ≤ 8.
>    - SONIC main metric within 45 s: S total 34, above the predicted 24–33; TabletopGrasp 15, above 5–14.
>
>    The other misses are covered by the triggered criteria.
>
> ## The four pitfalls
>
> - **Benchmark selection: acceptable.** All four tasks are reported, and the write-up says S only differs on 28–29 TabletopGrasp episodes. Those come from about 14 distinct layouts with nearly constant reach geometry, so this benchmark cannot test whether the model's per-state prediction matters.
> - **Data leakage: a minor problem.**
>   - The training hold-out picks episodes by level and index and ignores that levels 0 and 1 share layouts (pregrasp goals coincide within 6 mm for 8/10 indices).
>   - Three of the nine "held-out" episodes (L0_e0, L1_e1, L1_e9) have a twin within 1.3 mm in the training data, so "19/28 seen" understates exposure.
>   - The model is not more accurate on truly unseen episodes: miss after correction 2.28 cm against 2.13 overall, and 2.48 cm on the fully unseen development episodes. Leakage is unlikely to drive 3a, though the samples are small.
> - **Metric misuse: partly.**
>   - Both limits are reported. But the official limit is a knife edge for ScaleBFM (success at 15.2–16.1 s against 16 s), and SONIC's −2 there is the slower replanned paths.
>   - Bootstrap intervals treat the duplicated layouts as independent. Counting by layout, the 19/21 sign test becomes 13/14, p ≈ 0.002, not "p < 0.001". ScaleBFM's official −4 has exact McNemar p = 0.125.
> - **Post-hoc selection: none in the choice of S or its parameters.**
>   - The S variant was chosen by the fixed rule (S_end 14, S_clear 0, S_both 0).
>   - The test entry was committed (14:21:33Z) before S started (14:22:10Z).
>   - The hand-stop analysis was registered and covers all repaired episodes.
>   - Reading 2's "the model is useful" is an interpretation added without the obvious fixed-offset control.
>
> **The two mid-run fixes were legitimate.**
> - **Fix 1** (cuRobo writes a plan file with no trajectory): a crash fix that matches the registered give-up rule.
> - **Fix 2** (judge a repaired plan by the predicted palm against J's original end): the only reading under which the registered repair "move the goal back by the offset" can ever pass. Both were made before any S episode ran (check rerun at 14:07:50, S rows at 14:15).
> - **Code identity:** I diffed the development run's copy of `s_check.py` against the test version (4106eba5). They differ only in the table function, which got official-limit pairing and a fix for a numpy bool subtraction error; that is reporting only, and my independent recompute agrees with it.
> - **Cost of fix 2:** the check now confirms itself (see finding 7).
>
> ## Experiments that could overturn the conclusions
>
> **(a) "The self model brings the hand ~3× closer."** Run arms on ScaleBFM's 28 routed episodes, the 11 development episodes, and new episodes with jittered standing spots so the reach geometry actually varies. Every arm falls back to J's plan when replanning fails.
> - J;
> - replan only: same goal, forced new plan;
> - constant shift of [+3.17, −0.10, −3.36] cm from the development episodes, no model;
> - S;
> - sham: same shift size as S, random direction;
> - oracle: minus this episode's own J miss.
>
> Primary metric: paired pregrasp miss, clustered by layout. From the measured linearity, I expect about 1.3 cm for the constant, 1.7–2.0 for S, about 0.6 for the oracle, and about 4.7 for replan only. If the constant ties or beats S on the fixed test layouts, the claim becomes "a fixed offset correction helps". The self model only earns credit where the constant fails, on the jittered stances.
>
> **(b) "The entry stage absorbs the reach error."**
> - Dose-response: offset the pregrasp goal by 0, 3, 6, 9 and 12 cm in six directions; record entry miss and success at both limits.
> - Ablation: make the entry start from the commanded pregrasp instead of the measured palm (`ENTRY_CONT=1`).
> - Escape the ceiling: also run on the development set's hard levels, where J is 10/11 rather than all successes.
>
> If success stays flat to about 9 cm with the correction but falls without it, absorption is causal. If it falls even with the correction, 3b is wrong.
>
> **(c) "SONIC's 0 is noise."** The simulator is deterministic (J equals JR on 240/240), so "noise" here means sensitivity to the command.
> - Run J, J′ (J with a 0.5 cm random goal nudge) and S, eight times each with such nudges, on SONIC's 29 routed episodes. Estimate per-episode success rates and the J-against-J′ flip rate.
> - Add an S-capped arm (fall back to J when the shift exceeds 5 cm) and a version of S retimed to J's path duration, to separate timing from accuracy at the official limit.
>
> The noise reading is overturned if S underperforms on episodes where its shift moved the hand more than 10 cm, or if S-capped beats S.
>
> ## Verdicts per conclusion
>
> | Conclusion | Verdict |
> |---|---|
> | Main-metric numbers (A/J/S, both controllers, TabletopGrasp) | Supports |
> | Other three tasks identical; SONIC XMovePick fails for both J and S | Supports |
> | Pipeline criterion not triggered | Supports |
> | ScaleBFM: S below J within prediction; all losses from the 7 give-ups; 21 repaired identical | Supports (give-ups are planner failures, not model checks; official −4 alone not significant) |
> | SONIC triggers "S not below J" and "more than 10% differ" (7/23) | Supports |
> | SONIC's 7: distance and outcome have no consistent relation | Does not support (all 3 losses had model-induced misses of 13–31 cm) |
> | SONIC's 0 difference is noise | Insufficient evidence (high noise, but harm on some episodes) |
> | SONIC official −2 (as a reach result) | Does not support as a reach effect (4/5 losses are timing from longer replanned paths) |
> | Error at 90% not above 1.5× development | Supports (3 other missed predictions not acknowledged) |
> | Reading 1: S gains nothing; ScaleBFM −7 all from give-ups | Supports; "SONIC is noise": insufficient evidence |
> | Reading 2a: moving the goal brings ScaleBFM's hand ~3× closer | Supports (2.6–2.7×; replanning alone ≤ about 0.6 cm) |
> | Reading 2a: the self model's end prediction is what helps | Does not support (fixed offset 1.26 cm against model 2.04; model better in 7/21) |
> | Reading 2b: the entry absorbs the gain, so outcomes don't change | Insufficient evidence (consistent for ScaleBFM, confounded by the 28/28 ceiling, false for SONIC beyond about 9 cm) |
> | Reading 3: SONIC's model is unreliable and pushes the hand farther | Supports (farther in 18/23, p = 0.011; model miss 9.58 cm against 6.34 uncorrected) |
> | Evidence strength "19/21, p < 0.001" | Overstated: by layout 13/14, p ≈ 0.002 |
>
> I ran the scripts by piping them over ssh into the remote environment's Python, so nothing was written into the output directories. The local temp files are deleted and none were left on the remote.

### 复核后的更正（2026-11-24）

复核重算的数全部和原文一致（SONIC 的自助法区间换种子后是 [−6, +6] / [−7, +3]，差别只在随机种子）。上面"结论与下一步"保留原样，读法照下面改：

1. **撤回"自身模型对 ScaleBFM 有用"（读法 2）。** 挪目标确实把手放近了（中位数 4.71 → 1.73 cm，单是重新规划最多贡献约 0.6 cm），但不用模型、只按开发局上 J 的平均偏差挪一个固定量，测试局上剩 1.26 cm，比模型的 2.04 cm 还好，模型只在 7 / 21 局上更好。原因：TabletopGrasp 30 局的预抓取点都在约 9 × 10 cm 的一块里，站位跟着目标走，伸手的几何几乎不变，常数就够。模型在前后、上下方向跟得上逐局变化（相关 0.78、0.94），侧向有约 +2.2 cm 的固定偏差、相关 −0.36。所以这批局证明不了"模型的逐局预测"有用，只能说"按偏差修正命令"有用，而这件事常数就能做。
2. **"抓取的下一段吸收了伸手误差"（读法 2 的后半）证据不足。** ScaleBFM 的 J 在有路线的局上 45 秒内已经 28 / 28，S 没有可赢的余地，分不开"吸收"和"到顶"。SONIC 上不成立：预抓取偏差超过约 9 cm 后成功率明显下降。
3. **SONIC 的"0 差是噪声"改成"模型造成的损失加上噪声里赢的局"。** S 输的 3 局都是模型的修正把手推远了（13.4、21.7、30.9 cm，J 是 4.5、12.9、8.4 cm），原文"距离和成败没有一致关系"对输的局不对。赢的 4 局只有 L0_e6 是修得更近；另外 3 局像噪声（SONIC 上同布局的两局 J 只有 3 / 10 结果一致）。
4. **SONIC 官方时限内 −2 不是伸手精度的结果：** 只有 J 在官方时限内成功的 5 局里，4 局两边 45 秒内都成功；S 重新规划的路线长 0.8–2.0 s，成功时间落到 16.24–16.60 s，刚过 16 s 的时限。
5. **这次 S 实际上是"一律挪目标，规划失败就放弃"。** 第 0 轮没有一局过关（最小的预测偏差 3.86 cm 已超过 2 cm 阈值）；修过的候选用同一个模型检查，几乎自动过关；放弃全部来自 cuRobo 规划不出挪过的目标，没有一次是模型否决了路线。ScaleBFM 放弃的 7 局里有 3 局正是 J 本来最准的（1.44、1.45、1.72 cm），模型却预测偏 4.1–4.5 cm；另外放弃的局实际跑的是直线伸手，ScaleBFM 有 2 局、SONIC 有 1 局其实抓起来了，按登记算失败。
6. **补上没提的预测偏差：** SONIC 误差 90 % 分位 22.15 cm（预测 8–22）；SONIC 原固定半径罩住 0.379（预测 0.5–0.9）；ScaleBFM 超范围关节最大 9（预测不超过 8）；SONIC 主指标 45 秒内 S 34（预测 24–33）、TabletopGrasp 15（5–14）。
7. **证据强度改正：** "19 / 21，p < 0.001"按布局去重是 13 / 14，p ≈ 0.002；ScaleBFM 官方时限内 −4 的精确 McNemar p = 0.125，单看不显著。训练留出按等级和局号分，没管 0、1 级同布局，"留出"的 9 局里有 3 局在训练里有 1.3 mm 以内的孪生局；但模型在真正没见过的局上并不更准（2.28 对 2.13 cm，开发局 2.48 cm），泄漏不大可能是读法 2 的来源。
8. **两处中途改的代码复核认为合法**（第一处是崩溃修复，和登记的放弃规则一致；第二处是登记的修法能过关的唯一读法），都在任何 S 局之前。代价是第二处让检查变成自己确认自己（第 5 条）。
9. **下一步改成：**
   - 要证明的是"模型的逐局预测"比常数强，就得在伸手几何真的会变的局上比：复核给的实验 (a)，J、只重新规划、固定常数修正、S、同大小随机方向、用本局 J 偏差的上限，六种都"规划失败就退回 J 的路线"，按布局配对比预抓取偏差；外加站位抖动的新局。常数在固定布局上打平或赢 S，结论就只能写成"固定偏差修正有用"，模型只在常数失效的地方算贡献。
   - 固定常数修正要加进主表的对手（和 M、D 一样不用模型），否则 S 的任何好处都可能被它解释。
   - S 的"修不好就放弃"改成"退回 J 的路线"，并加一条：模型预测偏差太大（比如超过 5 cm）就不挪，避免 SONIC 上那种越修越远。

### 2026-11-25 补注：第 1、2 条更正可能是同一个遗漏造成的

- [selfmodel_precision_20261124](selfmodel_precision_20261124.md) 的独立复核发现，S 的终点检查（`s_check.py` 第 303 行 `end_off`）比的是"预测的手掌对 J 自己路线终点的参考手掌"，不是对目标。
- J 的路线在开局那一眼规划，机器人实际站的位置和规划时不同，所以参考在终点本来就离目标有一段差：ScaleBFM 长度中位数约 2 cm，SONIC 约 9 cm。这段差在发命令时就能用运动学算出来。
- 常数把这段差吸收了，模型的挪法没管它。上面第 1 条"常数 1.26 cm 比模型 2.04 cm 准"、第 2 条"模型侧向约 +2.2 cm 的固定偏差"，很可能就是这段差，不是模型的错。
- 在补跑把这段差加进去的那一组之前，第 1 条的"撤回'自身模型对 ScaleBFM 有用'"改为**证据不足**。
