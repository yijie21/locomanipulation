# selfmodel_lmlreal_20261125：用 LocoManiLab 真实任务局检验 train3 新模型的预测误差，并检查起始状态在不在训练范围里

- **日期**：2026-11-25。在任何运行之前写好，单独提交。
- **规则**：用户定的规则是"新模型当天就用真实的局检验预测误差，并检查起始状态在不在训练范围里"。train3 的模型是在 LocoManiLab 宿主里训的，所以真实局用 LocoManiLab 的任务局，不用 SIMPLE（SIMPLE 是另一个仿真，模型用不上）。
- **机器**：4090。等 data3b 的 ScaleBFM 和 HEFT 采完（约 15:10Z）以后再开，最多 80 个进程。SONIC 要等 4090b 上 data3b 的 SONIC 采完（约 16:00Z），或者放到 4090 上用 CPU 推理。
- **代码**：`selfwm`，用 `scripts/experiments/hardgrasp_20261125/` 的任务和流程：开局看一眼、走到站位、预抓取、进入、合手。
  - 只跑直接执行这一组。模型只做预测，不改变任何命令。
  - 每一下伸手都按 data3 的布局记下来：命令前的 S、W、J、C、HQ、KP 行，命令，参考特征，停稳后的终点。这样以后任何模型都能离线评估，不用重跑。
  - 跑之前把代码抄进 `asrun/`，记下 md5。
- **任务和局数**：hardgrasp 的 12 个 LocoManiLab 任务，正式局的种子和抖动不变，每个任务 30 局。
- **控制器**：ScaleBFM 必做；SONIC 和 HEFT 能接上就做，接不上就写明原因。
- **模型**：[selfmodel_train3_20261125](selfmodel_train3_20261125.md) 里每个控制器 frac 1.0 的 E3。合并训练出来以后，用同一份记录离线再评估一次。

## 假设

train3 的模型在验证集上误差约 3–4 cm。真实任务里的伸手有两种：
- 第一下（走到站位后伸到预抓取点）：起点和命令大多落在 data3 的范围里，预测误差应该接近验证误差。
- 第二下（从预抓取点伸进去）：data3 几乎没有"停稳的抬手 + 短距离 + 小转腕"这种伸手，预测误差应该明显变大。这正是 data3b 要补的部分。

## 度量

- **预测误差**：模型预测的终点和实际停稳终点之间的距离，取中位数（cm）。第一下和第二下分开报，每个任务分开报；全部局报一次，只算出发前停稳的局再报一次。同时报直接执行的误差（手到命令的距离）。
- **在不在训练范围里**：
  - 29 个关节逐个和训练数据的 0.5–99.5 % 区间比，全部落在区间里才算"在范围里"，记下比例；
  - 集成模型的成员分歧，记下超过 val 90 % 分位的比例；
  - 在范围里和不在范围里的预测误差分开报。
- **和验证误差的倍数**：真实局的预测误差除以同一模型在 val 上的误差。ScaleBFM 是 3.08 cm，HEFT 3.99 cm，SONIC 4.25 cm。

## 预测

以 ScaleBFM 为准，SONIC 和 HEFT 如果跑了也套用同样的倍数：

| 项 | 第一下 | 第二下 |
|---|---|---|
| 预测误差是 val 的几倍 | 1.0–2.0 倍 | 1.5–3.5 倍 |
| 起点在训练范围里的比例（29 个关节） | 60–95 % | 40–85 % |

- 不在范围里的伸手，预测误差是在范围里的 1.3–3 倍。
- 出发前停稳的比例：第一下 20–80 %，第二下 50–95 %。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| 第一下 ≤ 1.5 倍 val | 模型能迁移到真实任务的第一下 |
| 第一下 > 2 倍 val | 找出是哪些量落在范围外，在那里加采 |
| 第二下 > 2 倍 val | 证实需要 data3b。合并模型用同一份记录离线再评估，看倍数降了多少 |
| 第二下 ≤ 1.3 倍 val | data3 已经够用。data3b 的价值要在合并训练里重新衡量 |
| 不在范围里的比例 > 50 % | 先查记录的特征和训练数据在定义上是否一致（坐标系、手点、时刻），再下结论 |

## 修订 1：评估的切法（2026-11-25，正式运行 15:26Z 已开始，还没有看任何结果）

预测和判据都不变。

1. **"第一下"和"第二下"按 lr_run 的 reach_no 切，不按 chain 切。** 第一下是 reach_no = 1（伸到预抓取点），第二下是 reach_no = 2（伸进去）。
   - 这和登记里的定义一致。
   - 有途经点的局里，预抓取那一下是 chain 1，伸进去那一下是 chain 2。按 chain 切会把两种伸手混在一起。
   - 途经点那几下只算进"全部"。
2. **T3 的 chain 输入截到 1。** T3 只在 data3 上训过，没见过 chain 大于 1 的数据，lr_run 的说明也要求评估方截到训练范围。train3b 的模型（M、D、MX、MXa）见过 chain 0–3，不截。chain = 2 的伸手单独再报一格，好看出截断有没有影响。
3. **MX 只评估 chain ≥ 1 的伸手。**
   - 记录里存了 walk_cmd、walk_real、ref_pel，但没有存每一下的 d_this 和 d_prev。
   - chain 0 的伸手原地时，起始参考的根不在原点；先走路时，补走那次的位姿没有记下来。所以按 data3 的公式算不出这一下计划的参考起点。
   - chain ≥ 1 时用"上一下的 ref_pel − 实测根"来算，是精确的。
   - 正式运行不为这个重跑。第一下那一格，MX 留空并写明原因。
   - 以后的宿主（LML 里的修正对比、换仿真检验）每一下都要存 d_this 和 d_prev。

## 结果（ScaleBFM 先出；HEFT 和 SONIC 还没出）

**运行**
- 主冻结：2026-11-25T14:21:49Z，`asrun/MD5`。
- ScaleBFM 正式运行：15:25:59Z 开始，15:31:14Z 结束，210 局，35 个进程。
- 代码从仓库目录运行，因为 env.py 按自己所在的位置找仓库根。运行前后都和 `asrun` 的 md5 核对过。

**评估脚本**
- 冻结的 `lr_eval.py`（ff232ee7）缺修订 1 要求的两样："全部"那一组，和 chain = 2 的单独一格。它不能改：一改 md5，后面的 HEFT 和 SONIC 正式运行就会中止。所以另写了两份：
  - `lr_eval_r1.py`：4b04a7df，冻结于 15:34:18Z，存在 `asrun_r1/`。只补上修订 1 那两样。
  - `lr_eval_r2.py`：c8dd7e50，冻结于 15:37:19Z，存在 `asrun_r2/`。在 r1 的基础上加了 p90、参考分母 `x cval`，以及每个任务按停稳和没停稳再拆开。
- r2 加的这些，是我在看到 0.82 / 1.04 两个主数以后才要求的。判据仍只按登记的分母（val 3.08 cm），只算停稳的落点。
- 修订 1 的两条，原脚本本来就照做了：按 reach_no 切，T3 的 chain 截到 1。

**和登记不一样的地方**（执行 agent 报告，原意照录）：
1. 只跑了 7 个任务，不是 12 个。这 7 个是 hardgrasp 的正式任务集，另外 5 个没有可用的动作、场景或物体。
2. HEFT 和 SONIC 的时间表整体推后了（延迟 − 7）帧：HEFT 推后 1 帧，SONIC 推后 40 帧。ScaleBFM 和 hardgrasp 完全一样。
3. HEFT 和 SONIC 用的是 ScaleBFM 缓存下来的那一眼，看的时候机器人姿态差 24–34 mm。
4. 判定停稳的窗口只到下一条命令或合手为止，不另外再等 3 s。没停稳的落点做了标记，不进主结果，单独报一行。
5. 途经点那一下记为 reach_no 0，只算进"全部"。
6. 事后加了接触计数（碰家具、碰物体），只观察，不影响仿真；210 局和存档逐局相同。

**一致性检查**（`check.txt`，md5 d1d8df65，CHECK PASSED）：
- 终点误差重算、ik_err 重算、data.py 的 err_cm、局内直接算的误差，四者两两相差不超过 0.00003 cm。
- 从 npz 重建的 x0、u 和在线算的完全一样。
- 210/210 局和 hardgrasp 存档逐字段相同。
- val 自检：中位数 3.077 cm。

**评估原始输出**：`$O/scalebfm/eval_r2.txt`，md5 19b8158f，其中 `O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125`。原样如下：

```
scalebfm: model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/scalebfm/models_s0/E3_ensemble_f1; val check: 3066 kept val segments, model median 3.077 cm (registered 3.08); chained val reaches (chain >= 1) 1486, model median 5.209 cm (the x cval denominator, reference only); val spread |pos_std| p90 5.215 cm; training band: 25353 training segments' command rows; training chain max 1, walked [0, 1]
records /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125/scalebfm/data/all.npz: 316 segments from 135 episodes; reach_no {0: 46, 1: 135, 2: 135}; fell / lost 1; label not settled 89; kept 226; chain in the record {0: 135, 1: 135, 2: 46} -> clipped to 1; walked 240
== reach 1 (pre-grasp, 第一下): 135 recorded (135 not fell / lost, 117 kept); settled start (all recorded) 58.5 % (79/135); in range (all recorded) 83.7 % (113/135); label settled (not fell) 86.7 %
    all kept                       n  117 eps 117 | model 2.52 [2.11, 3.38] p90 29.90 cm  x val 0.82 [0.68, 1.10]  x cval 0.48 [0.40, 0.65] | direct 10.30 [6.86, 12.93] p90 35.38 cm  model/direct 0.24 | in range  87.2 % [80.3, 92.3] | spread > val p90   9.4 % | settled start  58.1 %
    kept, settled start            n   68 eps  68 | model 2.25 [1.62, 2.73] p90 27.13 cm  x val 0.73 [0.53, 0.89]  x cval 0.43 [0.31, 0.52] | direct 13.02 [10.83, 13.87] p90 32.26 cm  model/direct 0.17 | in range  85.3 % [76.5, 92.6] | spread > val p90   0.0 % | settled start 100.0 %
    kept, start not settled        n   49 eps  49 | model 4.20 [1.64, 7.31] p90 30.09 cm  x val 1.36 [0.53, 2.37]  x cval 0.81 [0.31, 1.40] | direct 6.06 [5.10, 7.12] p90 39.53 cm  model/direct 0.69 | in range  89.8 % [79.6, 98.0] | spread > val p90  22.4 % | settled start   0.0 %
    kept, in range                 n  102 eps 102 | model 2.23 [1.57, 2.67] p90 25.90 cm  x val 0.72 [0.51, 0.87]  x cval 0.43 [0.30, 0.51] | direct 9.29 [6.73, 13.02] p90 37.54 cm  model/direct 0.24 | in range 100.0 % [100.0, 100.0] | spread > val p90   7.8 % | settled start  56.9 %
    kept, out of range             n   15 eps  15 | model 8.58 [3.89, 30.72] p90 32.98 cm  x val 2.79 [1.26, 9.98]  x cval 1.65 [0.75, 5.90] | direct 12.58 [5.10, 32.57] p90 33.86 cm  model/direct 0.68 | in range   0.0 % [0.0, 0.0] | spread > val p90  20.0 % | settled start  66.7 %
    out / in range (model median): 3.84 [1.85, 16.50]
    kept, chain 2 in the record (fed 1) n    0
    kept, chain <= 1 in the record n  117 eps 117 | model 2.52 [2.11, 3.51] p90 29.90 cm  x val 0.82 [0.68, 1.14]  x cval 0.48 [0.40, 0.67] | direct 10.30 [6.80, 13.11] p90 35.38 cm  model/direct 0.24 | in range  87.2 % [81.2, 93.2] | spread > val p90   9.4 % | settled start  58.1 %
    kept, spread > val p90         n   11 eps  11 | model 7.31 [3.34, 12.85] p90 22.14 cm  x val 2.37 [1.09, 4.17]  x cval 1.40 [0.64, 2.47] | direct 5.25 [4.92, 17.96] p90 20.95 cm  model/direct 1.39 | in range  72.7 % [45.5, 91.1] | spread > val p90 100.0 % | settled start   0.0 %
    kept, spread <= val p90        n  106 eps 106 | model 2.28 [1.59, 3.12] p90 30.15 cm  x val 0.74 [0.52, 1.01]  x cval 0.44 [0.30, 0.60] | direct 10.75 [7.09, 13.22] p90 37.66 cm  model/direct 0.21 | in range  88.7 % [83.0, 94.3] | spread > val p90   0.0 % | settled start  64.2 %
    all labels incl. unsettled (not fell) n  135 eps 135 | model 3.35 [2.39, 4.56] p90 32.34 cm  x val 1.09 [0.78, 1.48]  x cval 0.64 [0.46, 0.87] | direct 12.14 [8.78, 13.95] p90 39.55 cm  model/direct 0.28 | in range  83.7 % [77.8, 89.6] | spread > val p90  10.4 % | settled start  58.5 %
    contact during the reach (kept): furniture 64, target object 6, neither 52  [post hoc]
    kept, no contact [post hoc]    n   52 eps  52 | model 1.35 [0.97, 2.21] p90 6.72 cm  x val 0.44 [0.31, 0.72]  x cval 0.26 [0.19, 0.42] | direct 5.30 [4.73, 6.09] p90 11.54 cm  model/direct 0.25 | in range  94.2 % [86.5, 100.0] | spread > val p90   9.6 % | settled start  40.4 %
    kept, no contact, in range [post hoc] n   49 eps  49 | model 1.29 [0.94, 1.74] p90 6.97 cm  x val 0.42 [0.31, 0.56]  x cval 0.25 [0.18, 0.33] | direct 5.36 [4.72, 6.14] p90 11.94 cm  model/direct 0.24 | in range 100.0 % [100.0, 100.0] | spread > val p90   8.2 % | settled start  42.9 %
    kept, furniture contact [post hoc] n   64 eps  64 | model 5.09 [2.67, 8.86] p90 31.68 cm  x val 1.65 [0.87, 2.88]  x cval 0.98 [0.51, 1.70] | direct 14.39 [13.40, 17.91] p90 39.55 cm  model/direct 0.35 | in range  81.2 % [71.9, 90.6] | spread > val p90   9.4 % | settled start  71.9 %
    joints out of the band (kept segments, count): right_wrist_pitch 10, left_hip_pitch 4, right_shoulder_roll 2, waist_pitch 1, right_wrist_yaw 1; joints out per segment median 0, max 2
    per task:
      candle                         n   18 eps  18 | model 0.80 [0.66, 1.11] p90 1.49 cm  x val 0.26 [0.21, 0.36]  x cval 0.15 [0.13, 0.21] | direct 5.97 [4.70, 6.59] p90 7.08 cm  model/direct 0.13 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      candle, settled start          n    0
      candle, start not settled      n   18 eps  18 | model 0.80 [0.66, 1.10] p90 1.49 cm  x val 0.26 [0.21, 0.36]  x cval 0.15 [0.13, 0.21] | direct 5.97 [4.72, 6.59] p90 7.08 cm  model/direct 0.13 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      apple                          n   22 eps  22 | model 7.14 [4.56, 11.15] p90 31.79 cm  x val 2.32 [1.48, 3.62]  x cval 1.37 [0.87, 2.14] | direct 5.18 [4.59, 10.66] p90 32.53 cm  model/direct 1.38 | in range  63.6 % [45.5, 81.8] | spread > val p90  22.7 % | settled start  45.5 %
      apple, settled start           n   10 eps  10 | model 19.80 [3.12, 31.91] p90 32.28 cm  x val 6.43 [1.01, 10.36]  x cval 3.80 [0.60, 6.13] | direct 22.31 [8.02, 32.57] p90 33.42 cm  model/direct 0.89 | in range  40.0 % [10.0, 70.0] | spread > val p90   0.0 % | settled start 100.0 %
      apple, start not settled       n   12 eps  12 | model 6.20 [4.41, 7.60] p90 11.34 cm  x val 2.01 [1.43, 2.47]  x cval 1.19 [0.85, 1.46] | direct 4.75 [4.15, 5.18] p90 5.55 cm  model/direct 1.31 | in range  83.3 % [58.3, 100.0] | spread > val p90  41.7 % | settled start   0.0 %
      bowl                           n    5 eps   5 | model 26.32 [8.58, 38.00] p90 36.40 cm  x val 8.55 [2.79, 12.34]  x cval 5.05 [1.65, 7.30] | direct 50.13 [17.96, 59.64] p90 55.90 cm  model/direct 0.52 | in range  60.0 % [20.0, 100.0] | spread > val p90  40.0 % | settled start  40.0 %
      bowl, settled start            n    2 eps   2 | model 32.16 [26.32, 38.00] p90 36.83 cm  x val 10.44 [8.55, 12.34]  x cval 6.17 [5.05, 7.30] | direct 54.96 [50.28, 59.64] p90 58.70 cm  model/direct 0.59 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      bowl, start not settled        n    3 eps   3 | model 22.45 [8.58, 34.01] p90 31.70 cm  x val 7.29 [2.79, 11.04]  x cval 4.31 [1.65, 6.53] | direct 28.44 [17.96, 50.13] p90 45.80 cm  model/direct 0.79 | in range  33.3 % [0.0, 100.0] | spread > val p90  66.7 % | settled start   0.0 %
      salt                           n    9 eps   9 | model 3.89 [2.45, 7.63] p90 13.38 cm  x val 1.26 [0.80, 2.48]  x cval 0.75 [0.47, 1.47] | direct 4.88 [3.71, 11.59] p90 16.70 cm  model/direct 0.80 | in range  44.4 % [11.1, 77.8] | spread > val p90   0.0 % | settled start  77.8 %
      salt, settled start            n    7 eps   7 | model 3.38 [2.45, 7.63] p90 19.14 cm  x val 1.10 [0.80, 2.48]  x cval 0.65 [0.47, 1.47] | direct 4.88 [4.19, 11.59] p90 21.81 cm  model/direct 0.69 | in range  42.9 % [14.3, 85.7] | spread > val p90   0.0 % | settled start 100.0 %
      salt, start not settled        n    2 eps   2 | model 4.04 [3.89, 4.20] p90 4.17 cm  x val 1.31 [1.26, 1.36]  x cval 0.78 [0.75, 0.81] | direct 4.50 [3.54, 5.45] p90 5.26 cm  model/direct 0.90 | in range  50.0 % [0.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab                     n   14 eps  14 | model 20.74 [8.72, 30.15] p90 31.04 cm  x val 6.74 [2.83, 9.79]  x cval 3.98 [1.67, 5.79] | direct 31.35 [17.30, 39.56] p90 40.54 cm  model/direct 0.66 | in range 100.0 % [100.0, 100.0] | spread > val p90  28.6 % | settled start   0.0 %
      statue_cab, settled start      n    0
      statue_cab, start not settled  n   14 eps  14 | model 20.74 [8.72, 30.26] p90 31.04 cm  x val 6.74 [2.83, 9.82]  x cval 3.98 [1.67, 5.81] | direct 31.35 [17.11, 39.56] p90 40.54 cm  model/direct 0.66 | in range 100.0 % [100.0, 100.0] | spread > val p90  28.6 % | settled start   0.0 %
      dumbbell                       n   28 eps  28 | model 1.49 [1.22, 2.24] p90 6.72 cm  x val 0.48 [0.40, 0.73]  x cval 0.29 [0.23, 0.43] | direct 12.08 [8.46, 13.58] p90 20.49 cm  model/direct 0.12 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      dumbbell, settled start        n   28 eps  28 | model 1.49 [1.22, 2.24] p90 6.72 cm  x val 0.48 [0.40, 0.73]  x cval 0.29 [0.23, 0.43] | direct 12.08 [8.99, 13.71] p90 20.49 cm  model/direct 0.12 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      dumbbell, start not settled    n    0
      wcan                           n   21 eps  21 | model 1.74 [1.31, 2.39] p90 3.63 cm  x val 0.56 [0.43, 0.78]  x cval 0.33 [0.25, 0.46] | direct 13.82 [10.30, 14.39] p90 15.31 cm  model/direct 0.13 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, settled start            n   21 eps  21 | model 1.74 [1.31, 2.39] p90 3.63 cm  x val 0.56 [0.43, 0.78]  x cval 0.33 [0.25, 0.46] | direct 13.82 [10.30, 14.39] p90 15.31 cm  model/direct 0.13 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, start not settled        n    0
== reach 2 (entry, 第二下): 135 recorded (134 not fell / lost, 82 kept); settled start (all recorded) 88.1 % (119/135); in range (all recorded) 75.6 % (102/135); label settled (not fell) 61.2 %
    all kept                       n   82 eps  82 | model 3.19 [2.35, 3.97] p90 10.33 cm  x val 1.04 [0.76, 1.29]  x cval 0.61 [0.45, 0.76] | direct 7.15 [6.00, 11.19] p90 18.65 cm  model/direct 0.45 | in range  92.7 % [86.6, 97.6] | spread > val p90   2.4 % | settled start  95.1 %
    kept, settled start            n   78 eps  78 | model 3.10 [2.29, 3.89] p90 8.71 cm  x val 1.01 [0.74, 1.26]  x cval 0.60 [0.44, 0.75] | direct 7.03 [5.79, 9.20] p90 17.32 cm  model/direct 0.44 | in range  93.6 % [88.5, 98.7] | spread > val p90   0.0 % | settled start 100.0 %
    kept, start not settled        n    4 eps   4 | model 24.65 [1.80, 34.34] p90 33.28 cm  x val 8.00 [0.58, 11.15]  x cval 4.73 [0.34, 6.59] | direct 19.13 [2.66, 47.37] p90 40.19 cm  model/direct 1.29 | in range  75.0 % [25.0, 100.0] | spread > val p90  50.0 % | settled start   0.0 %
    kept, in range                 n   76 eps  76 | model 3.03 [2.29, 3.66] p90 8.24 cm  x val 0.98 [0.74, 1.19]  x cval 0.58 [0.44, 0.70] | direct 6.88 [5.74, 8.99] p90 16.16 cm  model/direct 0.44 | in range 100.0 % [100.0, 100.0] | spread > val p90   1.3 % | settled start  96.1 %
    kept, out of range             n    6 eps   6 | model 17.81 [5.56, 27.85] p90 27.85 cm  x val 5.78 [1.81, 9.04]  x cval 3.42 [1.07, 5.35] | direct 28.19 [13.97, 47.51] p90 47.51 cm  model/direct 0.63 | in range   0.0 % [0.0, 0.0] | spread > val p90  16.7 % | settled start  83.3 %
    out / in range (model median): 5.88 [1.76, 10.84]
    kept, chain 2 in the record (fed 1) n   16 eps  16 | model 3.89 [1.91, 4.95] p90 27.85 cm  x val 1.26 [0.62, 1.61]  x cval 0.75 [0.37, 0.95] | direct 5.45 [4.77, 12.26] p90 45.14 cm  model/direct 0.71 | in range  81.2 % [62.5, 100.0] | spread > val p90   0.0 % | settled start  93.8 %
    kept, chain <= 1 in the record n   66 eps  66 | model 3.10 [2.36, 3.87] p90 9.06 cm  x val 1.01 [0.77, 1.26]  x cval 0.60 [0.45, 0.74] | direct 7.96 [6.54, 13.63] p90 16.78 cm  model/direct 0.39 | in range  95.5 % [89.4, 100.0] | spread > val p90   3.0 % | settled start  95.5 %
    kept, spread > val p90         n    2 eps   2 | model 24.65 [18.49, 30.81] p90 29.58 cm  x val 8.00 [6.00, 10.00]  x cval 4.73 [3.55, 5.92] | direct 19.13 [14.82, 23.45] p90 22.58 cm  model/direct 1.29 | in range  50.0 % [0.0, 100.0] | spread > val p90 100.0 % | settled start   0.0 %
    kept, spread <= val p90        n   80 eps  80 | model 3.10 [2.29, 3.87] p90 8.94 cm  x val 1.01 [0.74, 1.26]  x cval 0.60 [0.44, 0.74] | direct 7.03 [5.83, 9.88] p90 17.61 cm  model/direct 0.44 | in range  93.8 % [88.8, 98.8] | spread > val p90   0.0 % | settled start  97.5 %
    all labels incl. unsettled (not fell) n  134 eps 134 | model 4.19 [3.42, 4.77] p90 29.34 cm  x val 1.36 [1.11, 1.55]  x cval 0.81 [0.66, 0.92] | direct 10.31 [7.38, 14.81] p90 38.18 cm  model/direct 0.41 | in range  76.1 % [68.7, 83.6] | spread > val p90   5.2 % | settled start  88.1 %
    contact during the reach (kept): furniture 66, target object 17, neither 10  [post hoc]
    kept, no contact [post hoc]    n   10 eps  10 | model 1.62 [1.17, 2.71] p90 3.07 cm  x val 0.53 [0.38, 0.88]  x cval 0.31 [0.23, 0.52] | direct 5.61 [4.85, 7.51] p90 9.17 cm  model/direct 0.29 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
    kept, no contact, in range [post hoc] n   10 eps  10 | model 1.62 [1.15, 2.71] p90 3.07 cm  x val 0.53 [0.37, 0.88]  x cval 0.31 [0.22, 0.52] | direct 5.61 [4.76, 7.22] p90 9.17 cm  model/direct 0.29 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
    kept, furniture contact [post hoc] n   66 eps  66 | model 3.87 [3.00, 4.45] p90 14.53 cm  x val 1.26 [0.97, 1.45]  x cval 0.74 [0.58, 0.86] | direct 8.16 [6.32, 14.58] p90 21.49 cm  model/direct 0.47 | in range  92.4 % [84.8, 98.5] | spread > val p90   3.0 % | settled start  93.9 %
    joints out of the band (kept segments, count): waist_yaw 2, right_hip_yaw 2, left_hip_yaw 2, right_hip_roll 2, left_knee 1, left_ankle_pitch 1; joints out per segment median 0, max 4
    per task:
      candle                         n   18 eps  18 | model 2.42 [1.68, 3.19] p90 3.39 cm  x val 0.78 [0.54, 1.04]  x cval 0.46 [0.32, 0.61] | direct 5.87 [4.90, 6.88] p90 7.51 cm  model/direct 0.41 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      candle, settled start          n   18 eps  18 | model 2.42 [1.61, 3.17] p90 3.39 cm  x val 0.78 [0.52, 1.03]  x cval 0.46 [0.31, 0.61] | direct 5.87 [4.93, 6.82] p90 7.51 cm  model/direct 0.41 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      candle, start not settled      n    0
      apple                          n    7 eps   7 | model 3.91 [2.25, 4.72] p90 4.81 cm  x val 1.27 [0.73, 1.53]  x cval 0.75 [0.43, 0.91] | direct 4.88 [4.60, 5.13] p90 5.45 cm  model/direct 0.80 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      apple, settled start           n    7 eps   7 | model 3.91 [2.25, 4.72] p90 4.81 cm  x val 1.27 [0.73, 1.53]  x cval 0.75 [0.43, 0.91] | direct 4.88 [4.60, 5.13] p90 5.45 cm  model/direct 0.80 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      apple, start not settled       n    0
      bowl                           n    4 eps   4 | model 27.85 [5.87, 34.34] p90 34.03 cm  x val 9.04 [1.91, 11.15]  x cval 5.35 [1.13, 6.59] | direct 45.14 [18.78, 52.10] p90 50.68 cm  model/direct 0.62 | in range  50.0 % [0.0, 100.0] | spread > val p90   0.0 % | settled start  75.0 %
      bowl, settled start            n    3 eps   3 | model 22.39 [5.87, 33.32] p90 31.13 cm  x val 7.27 [1.91, 10.82]  x cval 4.30 [1.13, 6.40] | direct 42.92 [18.78, 52.10] p90 50.27 cm  model/direct 0.52 | in range  33.3 % [0.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      bowl, start not settled        n    1 eps   1 | model 34.34 [nan, nan] p90 34.34 cm  x val 11.15 [nan, nan]  x cval 6.59 [nan, nan] | direct 47.37 [nan, nan] p90 47.37 cm  model/direct 0.72 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
      salt                           n    5 eps   5 | model 1.51 [1.13, 1.92] p90 1.86 cm  x val 0.49 [0.37, 0.62]  x cval 0.29 [0.22, 0.37] | direct 5.74 [4.23, 8.40] p90 7.51 cm  model/direct 0.26 | in range  80.0 % [40.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      salt, settled start            n    5 eps   5 | model 1.51 [1.13, 1.92] p90 1.86 cm  x val 0.49 [0.37, 0.62]  x cval 0.29 [0.22, 0.37] | direct 5.74 [4.23, 8.40] p90 7.51 cm  model/direct 0.26 | in range  80.0 % [40.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      salt, start not settled        n    0
      statue_cab                     n    2 eps   2 | model 23.97 [18.49, 29.44] p90 28.35 cm  x val 7.78 [6.00, 9.56]  x cval 4.60 [3.55, 5.65] | direct 34.04 [23.45, 44.64] p90 42.52 cm  model/direct 0.70 | in range  50.0 % [0.0, 100.0] | spread > val p90  50.0 % | settled start  50.0 %
      statue_cab, settled start      n    1 eps   1 | model 29.44 [nan, nan] p90 29.44 cm  x val 9.56 [nan, nan]  x cval 5.65 [nan, nan] | direct 44.64 [nan, nan] p90 44.64 cm  model/direct 0.66 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start 100.0 %
      statue_cab, start not settled  n    1 eps   1 | model 18.49 [nan, nan] p90 18.49 cm  x val 6.00 [nan, nan]  x cval 3.55 [nan, nan] | direct 23.45 [nan, nan] p90 23.45 cm  model/direct 0.79 | in range   0.0 % [nan, nan] | spread > val p90 100.0 % | settled start   0.0 %
      dumbbell                       n   28 eps  28 | model 3.65 [1.77, 7.82] p90 10.90 cm  x val 1.18 [0.57, 2.54]  x cval 0.70 [0.34, 1.50] | direct 11.61 [6.41, 14.83] p90 18.10 cm  model/direct 0.31 | in range  92.9 % [82.1, 100.0] | spread > val p90   3.6 % | settled start  92.9 %
      dumbbell, settled start        n   26 eps  26 | model 3.65 [1.74, 7.55] p90 9.83 cm  x val 1.18 [0.57, 2.45]  x cval 0.70 [0.33, 1.45] | direct 11.61 [6.41, 14.99] p90 18.51 cm  model/direct 0.31 | in range  92.3 % [80.8, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      dumbbell, start not settled    n    2 eps   2 | model 16.30 [1.80, 30.81] p90 27.91 cm  x val 5.29 [0.58, 10.00]  x cval 3.13 [0.34, 5.92] | direct 8.74 [2.66, 14.82] p90 13.60 cm  model/direct 1.87 | in range 100.0 % [100.0, 100.0] | spread > val p90  50.0 % | settled start   0.0 %
      wcan                           n   18 eps  18 | model 3.69 [2.30, 4.60] p90 5.11 cm  x val 1.20 [0.75, 1.49]  x cval 0.71 [0.44, 0.88] | direct 15.06 [6.54, 15.30] p90 15.66 cm  model/direct 0.24 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, settled start            n   18 eps  18 | model 3.69 [2.29, 4.60] p90 5.11 cm  x val 1.20 [0.74, 1.49]  x cval 0.71 [0.44, 0.88] | direct 15.06 [8.49, 15.30] p90 15.66 cm  model/direct 0.24 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, start not settled        n    0
== all reaches (via points, reach 1 and 2; 全部): 316 recorded (315 not fell / lost, 226 kept); settled start (all recorded) 76.9 % (243/316); in range (all recorded) 82.6 % (261/316); label settled (not fell) 71.7 %
    all kept                       n  226 eps 129 | model 3.19 [2.43, 3.96] p90 30.99 cm  x val 1.04 [0.79, 1.29]  x cval 0.61 [0.47, 0.76] | direct 10.25 [7.08, 13.35] p90 32.35 cm  model/direct 0.31 | in range  90.7 % [86.8, 94.1] | spread > val p90   5.8 % | settled start  76.5 %
    kept, settled start            n  173 eps 104 | model 2.94 [2.30, 3.57] p90 31.46 cm  x val 0.95 [0.75, 1.16]  x cval 0.56 [0.44, 0.69] | direct 12.14 [8.23, 14.23] p90 28.71 cm  model/direct 0.24 | in range  91.3 % [87.2, 94.9] | spread > val p90   0.0 % | settled start 100.0 %
    kept, start not settled        n   53 eps  53 | model 4.26 [2.11, 7.88] p90 30.31 cm  x val 1.38 [0.68, 2.56]  x cval 0.82 [0.40, 1.51] | direct 6.13 [5.25, 7.12] p90 39.55 cm  model/direct 0.69 | in range  88.7 % [79.2, 96.2] | spread > val p90  24.5 % | settled start   0.0 %
    kept, in range                 n  205 eps 126 | model 2.65 [2.26, 3.42] p90 30.46 cm  x val 0.86 [0.74, 1.11]  x cval 0.51 [0.43, 0.66] | direct 8.75 [6.97, 13.27] p90 27.38 cm  model/direct 0.30 | in range 100.0 % [100.0, 100.0] | spread > val p90   4.4 % | settled start  77.1 %
    kept, out of range             n   21 eps  20 | model 10.58 [7.25, 26.59] p90 33.32 cm  x val 3.43 [2.35, 8.63]  x cval 2.03 [1.39, 5.10] | direct 19.53 [10.21, 32.57] p90 37.14 cm  model/direct 0.54 | in range   0.0 % [0.0, 0.0] | spread > val p90  19.0 % | settled start  71.4 %
    out / in range (model median): 3.99 [2.23, 9.51]
    kept, chain 2 in the record (fed 1) n   16 eps  16 | model 3.89 [1.91, 5.30] p90 27.85 cm  x val 1.26 [0.62, 1.72]  x cval 0.75 [0.37, 1.02] | direct 5.45 [4.88, 8.40] p90 45.14 cm  model/direct 0.71 | in range  81.2 % [62.5, 100.0] | spread > val p90   0.0 % | settled start  93.8 %
    kept, chain <= 1 in the record n  210 eps 129 | model 3.13 [2.43, 4.23] p90 30.85 cm  x val 1.02 [0.79, 1.37]  x cval 0.60 [0.47, 0.81] | direct 10.95 [7.26, 13.82] p90 31.74 cm  model/direct 0.29 | in range  91.4 % [88.0, 95.1] | spread > val p90   6.2 % | settled start  75.2 %
    kept, spread > val p90         n   13 eps  13 | model 8.58 [3.77, 18.49] p90 22.39 cm  x val 2.79 [1.22, 6.00]  x cval 1.65 [0.72, 3.55] | direct 5.33 [4.92, 17.96] p90 22.95 cm  model/direct 1.61 | in range  69.2 % [38.5, 92.3] | spread > val p90 100.0 % | settled start   0.0 %
    kept, spread <= val p90        n  213 eps 119 | model 2.94 [2.30, 3.89] p90 31.31 cm  x val 0.95 [0.75, 1.26]  x cval 0.56 [0.44, 0.75] | direct 10.30 [7.12, 13.71] p90 32.85 cm  model/direct 0.29 | in range  92.0 % [88.5, 95.3] | spread > val p90   0.0 % | settled start  81.2 %
    all labels incl. unsettled (not fell) n  315 eps 135 | model 4.03 [3.27, 4.96] p90 32.30 cm  x val 1.31 [1.06, 1.61]  x cval 0.77 [0.63, 0.95] | direct 12.14 [8.00, 14.48] p90 39.12 cm  model/direct 0.33 | in range  82.9 % [78.3, 87.5] | spread > val p90   6.7 % | settled start  76.8 %
    contact during the reach (kept): furniture 157, target object 23, neither 62  [post hoc]
    kept, no contact [post hoc]    n   62 eps  52 | model 1.35 [1.13, 2.15] p90 4.56 cm  x val 0.44 [0.37, 0.70]  x cval 0.26 [0.22, 0.41] | direct 5.30 [4.74, 6.09] p90 11.54 cm  model/direct 0.26 | in range  95.2 % [89.7, 100.0] | spread > val p90   8.1 % | settled start  50.0 %
    kept, no contact, in range [post hoc] n   59 eps  49 | model 1.31 [1.12, 1.82] p90 5.04 cm  x val 0.43 [0.36, 0.59]  x cval 0.25 [0.21, 0.35] | direct 5.36 [4.81, 6.34] p90 11.94 cm  model/direct 0.24 | in range 100.0 % [100.0, 100.0] | spread > val p90   6.8 % | settled start  52.5 %
    kept, furniture contact [post hoc] n  157 eps 107 | model 4.68 [3.46, 7.25] p90 31.82 cm  x val 1.52 [1.12, 2.35]  x cval 0.90 [0.66, 1.39] | direct 14.48 [12.54, 15.61] p90 38.29 cm  model/direct 0.32 | in range  89.2 % [85.0, 93.4] | spread > val p90   5.1 % | settled start  86.0 %
    joints out of the band (kept segments, count): right_wrist_pitch 10, left_hip_pitch 4, left_hip_yaw 2, waist_yaw 2, right_hip_yaw 2, waist_pitch 2; joints out per segment median 0, max 4
    per task:
      candle                         n   36 eps  18 | model 1.21 [0.97, 1.59] p90 3.27 cm  x val 0.39 [0.31, 0.52]  x cval 0.23 [0.19, 0.31] | direct 5.96 [4.88, 6.56] p90 7.19 cm  model/direct 0.20 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  50.0 %
      candle, settled start          n   18 eps  18 | model 2.42 [1.61, 3.15] p90 3.39 cm  x val 0.78 [0.52, 1.02]  x cval 0.46 [0.31, 0.60] | direct 5.87 [4.90, 6.82] p90 7.51 cm  model/direct 0.41 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      candle, start not settled      n   18 eps  18 | model 0.80 [0.66, 1.10] p90 1.49 cm  x val 0.26 [0.21, 0.36]  x cval 0.15 [0.13, 0.21] | direct 5.97 [4.70, 6.59] p90 7.08 cm  model/direct 0.13 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      apple                          n   45 eps  28 | model 7.31 [4.26, 28.80] p90 31.78 cm  x val 2.37 [1.38, 9.35]  x cval 1.40 [0.82, 5.53] | direct 5.95 [5.01, 25.44] p90 29.91 cm  model/direct 1.23 | in range  82.2 % [72.3, 92.3] | spread > val p90  11.1 % | settled start  73.3 %
      apple, settled start           n   33 eps  21 | model 23.37 [3.87, 30.95] p90 31.89 cm  x val 7.59 [1.26, 10.05]  x cval 4.49 [0.74, 5.94] | direct 18.21 [5.58, 26.09] p90 32.04 cm  model/direct 1.28 | in range  81.8 % [71.1, 93.1] | spread > val p90   0.0 % | settled start 100.0 %
      apple, start not settled       n   12 eps  12 | model 6.20 [4.41, 7.60] p90 11.34 cm  x val 2.01 [1.43, 2.47]  x cval 1.19 [0.85, 1.46] | direct 4.75 [4.15, 5.18] p90 5.55 cm  model/direct 1.31 | in range  83.3 % [58.3, 100.0] | spread > val p90  41.7 % | settled start   0.0 %
      bowl                           n   13 eps   7 | model 33.32 [22.42, 35.74] p90 37.82 cm  x val 10.82 [7.28, 11.60]  x cval 6.40 [4.31, 6.86] | direct 41.88 [28.44, 50.13] p90 51.74 cm  model/direct 0.80 | in range  69.2 % [50.0, 88.9] | spread > val p90  15.4 % | settled start  69.2 %
      bowl, settled start            n    9 eps   5 | model 33.32 [22.39, 38.00] p90 38.94 cm  x val 10.82 [7.27, 12.34]  x cval 6.40 [4.30, 7.30] | direct 41.88 [28.77, 52.10] p90 53.61 cm  model/direct 0.80 | in range  77.8 % [69.2, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      bowl, start not settled        n    4 eps   4 | model 28.23 [8.58, 34.34] p90 34.24 cm  x val 9.17 [2.79, 11.15]  x cval 5.42 [1.65, 6.59] | direct 37.90 [17.96, 50.13] p90 49.30 cm  model/direct 0.74 | in range  50.0 % [0.0, 100.0] | spread > val p90  50.0 % | settled start   0.0 %
      salt                           n   21 eps  10 | model 3.38 [1.78, 15.67] p90 35.53 cm  x val 1.10 [0.58, 5.09]  x cval 0.65 [0.34, 3.01] | direct 7.10 [5.17, 15.72] p90 28.49 cm  model/direct 0.48 | in range  71.4 % [56.0, 89.5] | spread > val p90   0.0 % | settled start  90.5 %
      salt, settled start            n   19 eps  10 | model 2.94 [1.52, 23.71] p90 35.55 cm  x val 0.95 [0.49, 7.70]  x cval 0.56 [0.29, 4.55] | direct 7.67 [5.74, 19.85] p90 28.55 cm  model/direct 0.38 | in range  73.7 % [57.1, 92.9] | spread > val p90   0.0 % | settled start 100.0 %
      salt, start not settled        n    2 eps   2 | model 4.04 [3.89, 4.20] p90 4.17 cm  x val 1.31 [1.26, 1.36]  x cval 0.78 [0.75, 0.81] | direct 4.50 [3.54, 5.45] p90 5.26 cm  model/direct 0.90 | in range  50.0 % [0.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab                     n   16 eps  15 | model 23.56 [8.99, 30.05] p90 30.84 cm  x val 7.65 [2.92, 9.76]  x cval 4.52 [1.73, 5.77] | direct 31.35 [17.49, 39.56] p90 41.26 cm  model/direct 0.75 | in range  93.8 % [80.0, 100.0] | spread > val p90  31.2 % | settled start   6.2 %
      statue_cab, settled start      n    1 eps   1 | model 29.44 [nan, nan] p90 29.44 cm  x val 9.56 [nan, nan]  x cval 5.65 [nan, nan] | direct 44.64 [nan, nan] p90 44.64 cm  model/direct 0.66 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start 100.0 %
      statue_cab, start not settled  n   15 eps  15 | model 18.49 [8.72, 30.26] p90 30.94 cm  x val 6.00 [2.83, 9.82]  x cval 3.55 [1.67, 5.81] | direct 24.51 [17.49, 39.56] p90 40.43 cm  model/direct 0.75 | in range  93.3 % [80.0, 100.0] | spread > val p90  33.3 % | settled start   0.0 %
      dumbbell                       n   56 eps  30 | model 2.09 [1.49, 3.42] p90 10.27 cm  x val 0.68 [0.48, 1.11]  x cval 0.40 [0.29, 0.66] | direct 12.08 [7.12, 13.90] p90 19.74 cm  model/direct 0.17 | in range  96.4 % [90.9, 100.0] | spread > val p90   1.8 % | settled start  96.4 %
      dumbbell, settled start        n   54 eps  28 | model 2.09 [1.49, 3.35] p90 9.89 cm  x val 0.68 [0.48, 1.09]  x cval 0.40 [0.29, 0.64] | direct 12.08 [7.17, 13.90] p90 19.82 cm  model/direct 0.17 | in range  96.3 % [90.7, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      dumbbell, start not settled    n    2 eps   2 | model 16.30 [1.80, 30.81] p90 27.91 cm  x val 5.29 [0.58, 10.00]  x cval 3.13 [0.34, 5.92] | direct 8.74 [2.66, 14.82] p90 13.60 cm  model/direct 1.87 | in range 100.0 % [100.0, 100.0] | spread > val p90  50.0 % | settled start   0.0 %
      wcan                           n   39 eps  21 | model 2.32 [1.68, 2.66] p90 5.00 cm  x val 0.75 [0.55, 0.86]  x cval 0.45 [0.32, 0.51] | direct 14.39 [8.78, 14.80] p90 15.64 cm  model/direct 0.16 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, settled start            n   39 eps  21 | model 2.32 [1.68, 2.66] p90 5.00 cm  x val 0.75 [0.55, 0.86]  x cval 0.45 [0.32, 0.51] | direct 14.39 [8.78, 14.78] p90 15.64 cm  model/direct 0.16 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, start not settled        n    0
sensitivity, chain as recorded (unclipped) on the 46 segments with chain 2 (16 kept): model median 3.88 cm vs clipped 3.89 cm
```

冻结版 `eval.txt`（md5 8c454745）里途经点那一下单列了一格，只作描述，原样如下：

```
== via-point reach (info): 46 recorded (46 not fell / lost, 27 kept); settled start (all recorded) 97.8 % (45/46); in range (all recorded) 100.0 % (46/46); label settled (not fell) 58.7 %
    all kept                       n   27 eps  27 | model 30.57 [27.62, 31.73] cm  x val 9.93 [8.97, 10.30] | direct 25.74 [24.20, 26.80] cm  model/direct 1.19 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
    all labels incl. unsettled (not fell) n   46 eps  46 | model 23.54 [1.58, 30.41] cm  x val 7.64 [0.51, 9.87] | direct 19.03 [7.50, 25.56] cm  model/direct 1.24 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  97.8 %
    contact during the reach (kept): furniture 27, target object 0, neither 0  [post hoc]
```

### ScaleBFM 对照判据（初步，三个控制器齐了以后再做独立复核）

- **第一下**：0.82 倍 val [0.68, 1.10]，落在"≤ 1.5 倍 val"那一格，说明模型能迁移到真实任务的第一下。预测是 1.0–2.0 倍，实际低于预测。
- **第二下**：1.04 倍 val [0.76, 1.29]，按登记落在"≤ 1.3 倍 val"那一格，也就是"data3 已经够用"。预测是 1.5–3.5 倍，实际低于预测。这一格不稳：
  - 区间上沿 1.29 正好贴着 1.3；
  - 把没停稳的落点也算进去，是 1.36 倍 [1.11, 1.55]；
  - 第二下只有 61 % 的落点停稳了。被排除的那些很可能是更难的伸手，所以这里有选择偏差。
- **登记的分母和伸手长短混在一起。** 第二下是短距离的进入动作，直接执行的误差本来就小：第二下 7.15 cm，第一下 10.30 cm。所以绝对误差小，不等于模型学会了。换成"模型误差 / 直接执行误差"，第一下是 0.24，第二下是 0.45；val 里接着上一次的伸手是 0.49 左右。按这个相对量看，第二下仍然比第一下差将近一倍，和 train3 复核看到的"接着上一次更差"一致。所以"data3 已经够用"不成立，data3b 的价值照判据在合并训练里衡量。
- **在训练范围里的比例**：第一下 83.7 %，第二下 75.6 %，都在预测区间里，没有触发"> 50 % 先查定义"那一格。
- **出发前停稳的比例**：第一下 58.5 %，第二下 88.1 %，都在预测区间里。
- **不在范围里的误差是在范围里的几倍**：第一下 3.84 倍 [1.85, 16.5]，第二下 5.88 倍 [1.76, 10.8]，点估计高于预测的 1.3–3 倍。

### 意外（ScaleBFM）

1. **大误差几乎都出在手碰到家具的伸手上。**
   - 没碰到任何东西的伸手，模型误差的中位数是第一下 1.35 cm、第二下 1.62 cm，比 val 还好。
   - 碰到家具的是 5.09 cm 和 3.87 cm。
   - 途经点那一下 27 次全部碰到了家具，模型误差 30.6 cm，比直接执行的 25.7 cm 还差。这些伸手是被家具挡住的。自身模型不看场景，所以预测的是"没挡住时手会停在哪"。被挡住的那部分不是模型能预测的，要靠规划时避开。
   - 以后在真实任务里用模型修正时，被挡住的伸手修正了也没用，甚至更差。对比实验里要把这一组单独报。
2. **任务之间差别很大。**
   - 蜡烛、哑铃、浇水壶、盐瓶：0.8–3.9 cm。
   - 苹果：7 cm。
   - 碗和柜子里的雕像：20–33 cm，直接执行也有 31–50 cm，碰家具很多。
3. **75 局一下都没有伸。** 其中看不到物体 45 局，站位或 IK 失败 26 局，其他 4 局，和 hardgrasp 存档完全一致。所以碗只有 8 局有数，盐瓶只有 10 局。

## 结果：HEFT 和 SONIC

**运行时间**（UTC，2026-11-25）：
- HEFT：15:31:33Z 开跑，15:36:27Z 跑完；
- SONIC：15:36:42Z 开跑，15:48:18Z 跑完。

两个都用 35 个进程、nice 19；SONIC 用 onnx-cpu 单线程推理。正式运行 15:48:38Z 全部结束，结束后代码的 md5 和 asrun 一致。评估脚本 `lr_eval_r2.py`（c8dd7e50）在 16:27–16:28Z 跑完。

**一致性检查**：两个控制器的 check.txt 都是 CHECK PASSED，第 1–4 项最大偏差都在 0.00003 cm 以内。第 5 项"和 hardgrasp 存档逐局比对"这次不适用，因为 hardgrasp 没跑过这两个控制器。

**val 自检**：HEFT 3.993 cm，SONIC 4.252 cm。

**和 ScaleBFM 不同的偏离**（执行 agent 报告）：
1. 时间表整体推迟，让动作和机器人的实际时间对齐。参考延迟 HEFT 8 帧、SONIC 47 帧，所以 HEFT 推迟 1 帧，SONIC 推迟 40 帧。
2. 借用 ScaleBFM 缓存的那一眼。看的时候两个控制器的机器人姿态和 ScaleBFM 的不同：HEFT 中位数差 56.7 mm（最大 112.7 mm），SONIC 中位数差 26.9 mm（最大 42.1 mm）。
3. HEFT 大量跑丢，用的是 hardgrasp 的规则：偏离参考超过 0.5 m，走路时也查。结果是走路时跑丢 61 局、第一下跑丢 17 局；210 局里只有 74 局伸了手；成功 8 局，ScaleBFM 是 32 局。SONIC 走路时只跑丢 7 局。
4. SONIC 停得慢。判定窗口截到下一条命令为止，所以第二下只有 30.7 % 的落点停稳。
5. 看不到物体 45 局、站位或 IK 失败 26 局、其他 4 局，三个控制器的这几个数相同。

### HEFT：eval_r2.txt 原文（md5 c2fc61eb）

```
heft: model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/models_s0/E3_ensemble_f1; val check: 1492 kept val segments, model median 3.993 cm (registered 3.99); chained val reaches (chain >= 1) 470, model median 6.989 cm (the x cval denominator, reference only); val spread |pos_std| p90 6.109 cm; training band: 20518 training segments' command rows; training chain max 1, walked [0, 1]
records /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125/heft/data/all.npz: 166 segments from 74 episodes; reach_no {0: 35, 1: 74, 2: 57}; fell / lost 18; label not settled 79; kept 69; chain in the record {0: 74, 1: 57, 2: 35} -> clipped to 1; walked 107
== reach 1 (pre-grasp, 第一下): 74 recorded (57 not fell / lost, 27 kept); settled start (all recorded) 31.1 % (23/74); in range (all recorded) 67.6 % (50/74); label settled (not fell) 47.4 %
    all kept                       n   27 eps  27 | model 28.88 [5.87, 34.82] p90 40.25 cm  x val 7.24 [1.47, 8.73]  x cval 4.13 [0.84, 4.98] | direct 32.35 [4.94, 37.25] p90 42.51 cm  model/direct 0.89 | in range  66.7 % [48.1, 85.2] | spread > val p90  44.4 % | settled start  37.0 %
    kept, settled start            n   10 eps  10 | model 35.25 [32.05, 37.50] p90 39.50 cm  x val 8.83 [8.03, 9.40]  x cval 5.04 [4.59, 5.37] | direct 38.07 [35.49, 39.91] p90 43.84 cm  model/direct 0.93 | in range  10.0 % [0.0, 30.0] | spread > val p90  20.0 % | settled start 100.0 %
    kept, start not settled        n   17 eps  17 | model 5.87 [4.60, 23.83] p90 38.46 cm  x val 1.47 [1.15, 5.97]  x cval 0.84 [0.66, 3.41] | direct 4.94 [4.29, 30.44] p90 37.69 cm  model/direct 1.19 | in range 100.0 % [100.0, 100.0] | spread > val p90  58.8 % | settled start   0.0 %
    kept, in range                 n   18 eps  18 | model 6.09 [4.65, 23.83] p90 44.31 cm  x val 1.53 [1.16, 5.97]  x cval 0.87 [0.66, 3.41] | direct 4.99 [4.38, 30.44] p90 46.77 cm  model/direct 1.22 | in range 100.0 % [100.0, 100.0] | spread > val p90  61.1 % | settled start   5.6 %
    kept, out of range             n    9 eps   9 | model 35.15 [30.88, 37.18] p90 37.31 cm  x val 8.81 [7.74, 9.32]  x cval 5.03 [4.42, 5.32] | direct 37.64 [33.73, 39.91] p90 40.20 cm  model/direct 0.93 | in range   0.0 % [0.0, 0.0] | spread > val p90  11.1 % | settled start 100.0 %
    out / in range (model median): 5.77 [1.43, 7.56]
    kept, chain 2 in the record (fed 1) n    0
    kept, chain <= 1 in the record n   27 eps  27 | model 28.88 [5.87, 34.82] p90 40.25 cm  x val 7.24 [1.47, 8.73]  x cval 4.13 [0.84, 4.98] | direct 32.35 [4.94, 37.25] p90 42.51 cm  model/direct 0.89 | in range  66.7 % [48.1, 81.5] | spread > val p90  44.4 % | settled start  37.0 %
    kept, spread > val p90         n   12 eps  12 | model 9.54 [3.86, 40.50] p90 45.11 cm  x val 2.39 [0.97, 10.15]  x cval 1.36 [0.55, 5.79] | direct 4.76 [4.37, 40.93] p90 51.88 cm  model/direct 2.00 | in range  91.7 % [75.0, 100.0] | spread > val p90 100.0 % | settled start  16.7 %
    kept, spread <= val p90        n   15 eps  15 | model 30.88 [5.87, 35.15] p90 36.45 cm  x val 7.74 [1.47, 8.81]  x cval 4.42 [0.84, 5.03] | direct 33.34 [5.10, 38.49] p90 39.53 cm  model/direct 0.93 | in range  46.7 % [20.0, 73.3] | spread > val p90   0.0 % | settled start  53.3 %
    all labels incl. unsettled (not fell) n   57 eps  57 | model 33.43 [24.96, 34.83] p90 49.15 cm  x val 8.38 [6.26, 8.73]  x cval 4.78 [3.57, 4.98] | direct 35.69 [30.44, 38.97] p90 53.55 cm  model/direct 0.94 | in range  57.9 % [45.6, 70.2] | spread > val p90  49.1 % | settled start  40.4 %
    contact during the reach (kept): furniture 14, target object 0, neither 13  [post hoc]
    kept, no contact [post hoc]    n   13 eps  13 | model 5.62 [3.95, 7.73] p90 28.99 cm  x val 1.41 [0.99, 1.94]  x cval 0.80 [0.56, 1.11] | direct 4.51 [4.19, 5.04] p90 25.37 cm  model/direct 1.25 | in range 100.0 % [100.0, 100.0] | spread > val p90  61.5 % | settled start   0.0 %
    kept, no contact, in range [post hoc] n   13 eps  13 | model 5.62 [3.95, 7.73] p90 28.99 cm  x val 1.41 [0.99, 1.94]  x cval 0.80 [0.57, 1.11] | direct 4.51 [4.19, 5.04] p90 25.37 cm  model/direct 1.25 | in range 100.0 % [100.0, 100.0] | spread > val p90  61.5 % | settled start   0.0 %
    kept, furniture contact [post hoc] n   14 eps  14 | model 35.25 [31.05, 37.82] p90 44.84 cm  x val 8.83 [7.78, 9.48]  x cval 5.04 [4.44, 5.41] | direct 38.07 [34.88, 41.37] p90 50.18 cm  model/direct 0.93 | in range  35.7 % [14.3, 57.3] | spread > val p90  28.6 % | settled start  71.4 %
    joints out of the band (kept segments, count): right_wrist_pitch 8, waist_pitch 1; joints out per segment median 0, max 1
    per task:
      candle                         n   13 eps  13 | model 5.62 [3.95, 7.73] p90 28.99 cm  x val 1.41 [0.99, 1.94]  x cval 0.80 [0.57, 1.11] | direct 4.51 [4.19, 5.04] p90 25.37 cm  model/direct 1.25 | in range 100.0 % [100.0, 100.0] | spread > val p90  61.5 % | settled start   0.0 %
      candle, settled start          n    0
      candle, start not settled      n   13 eps  13 | model 5.62 [3.95, 7.73] p90 28.99 cm  x val 1.41 [0.99, 1.94]  x cval 0.80 [0.57, 1.11] | direct 4.51 [4.19, 5.04] p90 25.37 cm  model/direct 1.25 | in range 100.0 % [100.0, 100.0] | spread > val p90  61.5 % | settled start   0.0 %
      apple                          n    7 eps   7 | model 35.35 [30.88, 37.82] p90 40.79 cm  x val 8.86 [7.74, 9.48]  x cval 5.06 [4.42, 5.41] | direct 37.64 [33.73, 38.97] p90 44.47 cm  model/direct 0.94 | in range  28.6 % [0.0, 71.4] | spread > val p90  28.6 % | settled start  71.4 %
      apple, settled start           n    5 eps   5 | model 35.35 [30.88, 37.82] p90 37.53 cm  x val 8.86 [7.74, 9.48]  x cval 5.06 [4.42, 5.41] | direct 37.64 [33.73, 38.97] p90 38.78 cm  model/direct 0.94 | in range   0.0 % [0.0, 0.0] | spread > val p90  20.0 % | settled start 100.0 %
      apple, start not settled       n    2 eps   2 | model 34.54 [23.83, 45.24] p90 43.10 cm  x val 8.66 [5.97, 11.34]  x cval 4.94 [3.41, 6.47] | direct 42.54 [32.35, 52.73] p90 50.69 cm  model/direct 0.81 | in range 100.0 % [100.0, 100.0] | spread > val p90  50.0 % | settled start   0.0 %
      bowl                           n    2 eps   2 | model 49.30 [43.91, 54.69] p90 53.61 cm  x val 12.36 [11.00, 13.71]  x cval 7.05 [6.28, 7.82] | direct 55.12 [44.22, 66.03] p90 63.85 cm  model/direct 0.89 | in range 100.0 % [100.0, 100.0] | spread > val p90 100.0 % | settled start  50.0 %
      bowl, settled start            n    1 eps   1 | model 54.69 [nan, nan] p90 54.69 cm  x val 13.71 [nan, nan]  x cval 7.82 [nan, nan] | direct 66.03 [nan, nan] p90 66.03 cm  model/direct 0.83 | in range 100.0 % [nan, nan] | spread > val p90 100.0 % | settled start 100.0 %
      bowl, start not settled        n    1 eps   1 | model 43.91 [nan, nan] p90 43.91 cm  x val 11.00 [nan, nan]  x cval 6.28 [nan, nan] | direct 44.22 [nan, nan] p90 44.22 cm  model/direct 0.99 | in range 100.0 % [nan, nan] | spread > val p90 100.0 % | settled start   0.0 %
      salt                           n    4 eps   4 | model 34.19 [28.88, 37.18] p90 36.57 cm  x val 8.57 [7.24, 9.32]  x cval 4.89 [4.13, 5.32] | direct 38.58 [33.07, 41.37] p90 40.94 cm  model/direct 0.89 | in range   0.0 % [0.0, 0.0] | spread > val p90   0.0 % | settled start 100.0 %
      salt, settled start            n    4 eps   4 | model 34.19 [28.88, 37.18] p90 36.57 cm  x val 8.57 [7.24, 9.32]  x cval 4.89 [4.13, 5.32] | direct 38.58 [33.07, 41.37] p90 40.94 cm  model/direct 0.89 | in range   0.0 % [0.0, 0.0] | spread > val p90   0.0 % | settled start 100.0 %
      salt, start not settled        n    0
      statue_cab                     n    1 eps   1 | model 6.31 [nan, nan] p90 6.31 cm  x val 1.58 [nan, nan]  x cval 0.90 [nan, nan] | direct 5.08 [nan, nan] p90 5.08 cm  model/direct 1.24 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab, settled start      n    0
      statue_cab, start not settled  n    1 eps   1 | model 6.31 [nan, nan] p90 6.31 cm  x val 1.58 [nan, nan]  x cval 0.90 [nan, nan] | direct 5.08 [nan, nan] p90 5.08 cm  model/direct 1.24 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
== reach 2 (entry, 第二下): 57 recorded (56 not fell / lost, 21 kept); settled start (all recorded) 50.9 % (29/57); in range (all recorded) 36.8 % (21/57); label settled (not fell) 37.5 %
    all kept                       n   21 eps  21 | model 2.27 [1.85, 5.29] p90 32.23 cm  x val 0.57 [0.46, 1.33]  x cval 0.32 [0.26, 0.76] | direct 3.29 [2.93, 25.94] p90 38.81 cm  model/direct 0.69 | in range  71.4 % [52.4, 90.5] | spread > val p90  14.3 % | settled start  66.7 %
    kept, settled start            n   14 eps  14 | model 2.12 [1.64, 2.77] p90 13.87 cm  x val 0.53 [0.41, 0.69]  x cval 0.30 [0.24, 0.40] | direct 3.27 [2.70, 4.05] p90 19.55 cm  model/direct 0.65 | in range  85.7 % [64.3, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
    kept, start not settled        n    7 eps   7 | model 18.47 [2.13, 32.23] p90 33.72 cm  x val 4.63 [0.53, 8.08]  x cval 2.64 [0.31, 4.61] | direct 26.05 [1.99, 38.81] p90 39.71 cm  model/direct 0.71 | in range  42.9 % [14.3, 71.4] | spread > val p90  42.9 % | settled start   0.0 %
    kept, in range                 n   15 eps  15 | model 2.04 [1.60, 2.66] p90 2.91 cm  x val 0.51 [0.40, 0.67]  x cval 0.29 [0.23, 0.38] | direct 3.07 [2.22, 3.29] p90 3.94 cm  model/direct 0.66 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  80.0 %
    kept, out of range             n    6 eps   6 | model 30.21 [18.52, 36.96] p90 36.96 cm  x val 7.57 [4.64, 9.26]  x cval 4.32 [2.65, 5.29] | direct 37.84 [25.99, 45.19] p90 45.19 cm  model/direct 0.80 | in range   0.0 % [0.0, 0.0] | spread > val p90  50.0 % | settled start  33.3 %
    out / in range (model median): 14.82 [8.41, 21.20]
    kept, chain 2 in the record (fed 1) n    6 eps   6 | model 30.21 [18.52, 36.96] p90 36.96 cm  x val 7.57 [4.64, 9.26]  x cval 4.32 [2.65, 5.29] | direct 37.84 [25.99, 45.19] p90 45.19 cm  model/direct 0.80 | in range   0.0 % [0.0, 0.0] | spread > val p90  50.0 % | settled start  33.3 %
    kept, chain <= 1 in the record n   15 eps  15 | model 2.04 [1.60, 2.28] p90 2.91 cm  x val 0.51 [0.40, 0.57]  x cval 0.29 [0.23, 0.33] | direct 3.07 [2.21, 3.29] p90 3.94 cm  model/direct 0.66 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  80.0 %
    kept, spread > val p90         n    3 eps   3 | model 28.19 [18.47, 35.95] p90 34.40 cm  x val 7.07 [4.63, 9.01]  x cval 4.03 [2.64, 5.14] | direct 38.81 [26.05, 41.06] p90 40.61 cm  model/direct 0.73 | in range   0.0 % [0.0, 0.0] | spread > val p90 100.0 % | settled start   0.0 %
    kept, spread <= val p90        n   18 eps  18 | model 2.17 [1.67, 2.89] p90 22.66 cm  x val 0.54 [0.42, 0.72]  x cval 0.31 [0.24, 0.41] | direct 3.21 [2.58, 3.87] p90 29.22 cm  model/direct 0.68 | in range  83.3 % [66.7, 100.0] | spread > val p90   0.0 % | settled start  77.8 %
    all labels incl. unsettled (not fell) n   56 eps  56 | model 24.22 [15.98, 27.90] p90 45.00 cm  x val 6.07 [4.01, 6.99]  x cval 3.46 [2.29, 3.99] | direct 31.62 [22.31, 36.69] p90 46.40 cm  model/direct 0.77 | in range  37.5 % [25.0, 50.0] | spread > val p90  39.3 % | settled start  51.8 %
    contact during the reach (kept): furniture 6, target object 3, neither 12  [post hoc]
    kept, no contact [post hoc]    n   12 eps  12 | model 2.09 [1.67, 2.46] p90 2.86 cm  x val 0.52 [0.42, 0.62]  x cval 0.30 [0.24, 0.35] | direct 3.16 [2.13, 3.48] p90 4.16 cm  model/direct 0.66 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  83.3 %
    kept, no contact, in range [post hoc] n   12 eps  12 | model 2.09 [1.67, 2.46] p90 2.86 cm  x val 0.52 [0.42, 0.62]  x cval 0.30 [0.24, 0.35] | direct 3.16 [2.13, 3.48] p90 4.16 cm  model/direct 0.66 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  83.3 %
    kept, furniture contact [post hoc] n    6 eps   6 | model 30.21 [18.52, 36.96] p90 36.96 cm  x val 7.57 [4.64, 9.26]  x cval 4.32 [2.65, 5.29] | direct 37.84 [25.99, 45.19] p90 45.19 cm  model/direct 0.80 | in range   0.0 % [0.0, 0.0] | spread > val p90  50.0 % | settled start  33.3 %
    joints out of the band (kept segments, count): right_wrist_pitch 5, right_wrist_roll 4, left_hip_roll 1, right_shoulder_yaw 1; joints out per segment median 0, max 3
    per task:
      candle                         n   15 eps  15 | model 2.04 [1.60, 2.28] p90 2.91 cm  x val 0.51 [0.40, 0.57]  x cval 0.29 [0.23, 0.33] | direct 3.07 [2.22, 3.29] p90 3.94 cm  model/direct 0.66 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  80.0 %
      candle, settled start          n   12 eps  12 | model 1.94 [1.52, 2.46] p90 2.86 cm  x val 0.49 [0.38, 0.62]  x cval 0.28 [0.22, 0.35] | direct 3.16 [2.46, 3.48] p90 4.16 cm  model/direct 0.61 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      candle, start not settled      n    3 eps   3 | model 2.13 [1.41, 5.29] p90 4.66 cm  x val 0.53 [0.35, 1.33]  x cval 0.31 [0.20, 0.76] | direct 1.99 [1.92, 3.15] p90 2.92 cm  model/direct 1.07 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      apple                          n    4 eps   4 | model 34.09 [18.56, 37.97] p90 37.37 cm  x val 8.54 [4.65, 9.52]  x cval 4.88 [2.66, 5.43] | direct 37.84 [25.94, 49.33] p90 46.17 cm  model/direct 0.90 | in range   0.0 % [0.0, 0.0] | spread > val p90  25.0 % | settled start  50.0 %
      apple, settled start           n    2 eps   2 | model 28.27 [18.56, 37.97] p90 36.03 cm  x val 7.08 [4.65, 9.52]  x cval 4.04 [2.66, 5.43] | direct 37.63 [25.94, 49.33] p90 46.99 cm  model/direct 0.75 | in range   0.0 % [0.0, 0.0] | spread > val p90   0.0 % | settled start 100.0 %
      apple, start not settled       n    2 eps   2 | model 34.09 [32.23, 35.95] p90 35.58 cm  x val 8.54 [8.08, 9.01]  x cval 4.88 [4.61, 5.14] | direct 37.84 [36.87, 38.81] p90 38.62 cm  model/direct 0.90 | in range   0.0 % [0.0, 0.0] | spread > val p90  50.0 % | settled start   0.0 %
      bowl                           n    1 eps   1 | model 18.47 [nan, nan] p90 18.47 cm  x val 4.63 [nan, nan]  x cval 2.64 [nan, nan] | direct 26.05 [nan, nan] p90 26.05 cm  model/direct 0.71 | in range   0.0 % [nan, nan] | spread > val p90 100.0 % | settled start   0.0 %
      bowl, settled start            n    0
      bowl, start not settled        n    1 eps   1 | model 18.47 [nan, nan] p90 18.47 cm  x val 4.63 [nan, nan]  x cval 2.64 [nan, nan] | direct 26.05 [nan, nan] p90 26.05 cm  model/direct 0.71 | in range   0.0 % [nan, nan] | spread > val p90 100.0 % | settled start   0.0 %
      salt                           n    1 eps   1 | model 28.19 [nan, nan] p90 28.19 cm  x val 7.07 [nan, nan]  x cval 4.03 [nan, nan] | direct 41.06 [nan, nan] p90 41.06 cm  model/direct 0.69 | in range   0.0 % [nan, nan] | spread > val p90 100.0 % | settled start   0.0 %
      salt, settled start            n    0
      salt, start not settled        n    1 eps   1 | model 28.19 [nan, nan] p90 28.19 cm  x val 7.07 [nan, nan]  x cval 4.03 [nan, nan] | direct 41.06 [nan, nan] p90 41.06 cm  model/direct 0.69 | in range   0.0 % [nan, nan] | spread > val p90 100.0 % | settled start   0.0 %
== all reaches (via points, reach 1 and 2; 全部): 166 recorded (148 not fell / lost, 69 kept); settled start (all recorded) 52.4 % (87/166); in range (all recorded) 63.9 % (106/166); label settled (not fell) 46.6 %
    all kept                       n   69 eps  44 | model 24.49 [5.87, 28.07] p90 37.31 cm  x val 6.14 [1.47, 7.04]  x cval 3.50 [0.84, 4.02] | direct 28.42 [5.04, 32.22] p90 41.12 cm  model/direct 0.86 | in range  78.3 % [70.0, 86.4] | spread > val p90  24.6 % | settled start  65.2 %
    kept, settled start            n   45 eps  35 | model 26.98 [23.57, 28.31] p90 37.56 cm  x val 6.76 [5.91, 7.10]  x cval 3.86 [3.37, 4.05] | direct 31.25 [27.32, 33.00] p90 40.79 cm  model/direct 0.86 | in range  75.6 % [66.7, 85.7] | spread > val p90   8.9 % | settled start 100.0 %
    kept, start not settled        n   24 eps  24 | model 6.09 [4.65, 23.83] p90 35.61 cm  x val 1.53 [1.16, 5.97]  x cval 0.87 [0.66, 3.41] | direct 4.99 [4.29, 30.44] p90 40.38 cm  model/direct 1.22 | in range  83.3 % [70.8, 95.8] | spread > val p90  54.2 % | settled start   0.0 %
    kept, in range                 n   54 eps  42 | model 15.63 [5.13, 26.26] p90 34.40 cm  x val 3.92 [1.29, 6.58]  x cval 2.24 [0.73, 3.76] | direct 16.08 [4.48, 30.24] p90 36.13 cm  model/direct 0.97 | in range 100.0 % [100.0, 100.0] | spread > val p90  24.1 % | settled start  63.0 %
    kept, out of range             n   15 eps  15 | model 33.23 [28.88, 35.95] p90 37.56 cm  x val 8.33 [7.24, 9.01]  x cval 4.75 [4.13, 5.14] | direct 37.64 [33.73, 39.91] p90 41.25 cm  model/direct 0.88 | in range   0.0 % [0.0, 0.0] | spread > val p90  26.7 % | settled start  73.3 %
    out / in range (model median): 2.13 [1.29, 6.72]
    kept, chain 2 in the record (fed 1) n    6 eps   6 | model 30.21 [18.52, 36.96] p90 36.96 cm  x val 7.57 [4.64, 9.26]  x cval 4.32 [2.65, 5.29] | direct 37.84 [26.05, 45.19] p90 45.19 cm  model/direct 0.80 | in range   0.0 % [0.0, 0.0] | spread > val p90  50.0 % | settled start  33.3 %
    kept, chain <= 1 in the record n   63 eps  42 | model 23.83 [5.65, 27.98] p90 37.16 cm  x val 5.97 [1.42, 7.01]  x cval 3.41 [0.81, 4.00] | direct 27.51 [4.65, 31.94] p90 39.72 cm  model/direct 0.87 | in range  85.7 % [77.6, 93.1] | spread > val p90  22.2 % | settled start  68.3 %
    kept, spread > val p90         n   17 eps  16 | model 28.19 [5.84, 43.93] p90 55.92 cm  x val 7.07 [1.46, 11.01]  x cval 4.03 [0.84, 6.28] | direct 30.44 [4.51, 44.22] p90 48.93 cm  model/direct 0.93 | in range  76.5 % [50.0, 94.4] | spread > val p90 100.0 % | settled start  23.5 %
    kept, spread <= val p90        n   52 eps  39 | model 24.16 [5.78, 27.97] p90 33.38 cm  x val 6.06 [1.45, 7.01]  x cval 3.46 [0.83, 4.00] | direct 27.96 [5.06, 31.94] p90 37.21 cm  model/direct 0.86 | in range  78.8 % [70.2, 88.9] | spread > val p90   0.0 % | settled start  78.8 %
    all labels incl. unsettled (not fell) n  148 eps  57 | model 27.39 [24.49, 28.65] p90 50.80 cm  x val 6.87 [6.14, 7.18]  x cval 3.92 [3.50, 4.10] | direct 32.73 [29.35, 34.63] p90 47.28 cm  model/direct 0.84 | in range  60.1 % [53.8, 67.6] | spread > val p90  37.8 % | settled start  58.8 %
    contact during the reach (kept): furniture 41, target object 3, neither 25  [post hoc]
    kept, no contact [post hoc]    n   25 eps  15 | model 3.11 [2.46, 4.60] p90 9.90 cm  x val 0.78 [0.62, 1.15]  x cval 0.45 [0.35, 0.66] | direct 4.15 [3.18, 4.46] p90 5.08 cm  model/direct 0.75 | in range 100.0 % [100.0, 100.0] | spread > val p90  32.0 % | settled start  40.0 %
    kept, no contact, in range [post hoc] n   25 eps  15 | model 3.11 [2.40, 4.60] p90 9.90 cm  x val 0.78 [0.60, 1.15]  x cval 0.45 [0.34, 0.66] | direct 4.15 [3.18, 4.51] p90 5.08 cm  model/direct 0.75 | in range 100.0 % [100.0, 100.0] | spread > val p90  32.0 % | settled start  40.0 %
    kept, furniture contact [post hoc] n   41 eps  27 | model 28.45 [27.48, 30.88] p90 43.91 cm  x val 7.13 [6.89, 7.74]  x cval 4.07 [3.93, 4.42] | direct 33.73 [32.28, 36.87] p90 46.07 cm  model/direct 0.84 | in range  63.4 % [54.3, 73.7] | spread > val p90  22.0 % | settled start  80.5 %
    joints out of the band (kept segments, count): right_wrist_pitch 13, right_wrist_roll 4, left_hip_roll 1, waist_pitch 1, right_shoulder_yaw 1; joints out per segment median 0, max 3
    per task:
      candle                         n   28 eps  17 | model 3.02 [2.24, 4.41] p90 8.81 cm  x val 0.76 [0.56, 1.10]  x cval 0.43 [0.32, 0.63] | direct 3.48 [3.17, 4.37] p90 5.06 cm  model/direct 0.87 | in range 100.0 % [100.0, 100.0] | spread > val p90  28.6 % | settled start  42.9 %
      candle, settled start          n   12 eps  12 | model 1.94 [1.52, 2.46] p90 2.86 cm  x val 0.49 [0.38, 0.62]  x cval 0.28 [0.22, 0.35] | direct 3.16 [2.46, 3.48] p90 4.16 cm  model/direct 0.61 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      candle, start not settled      n   16 eps  16 | model 4.97 [3.76, 6.71] p90 22.37 cm  x val 1.25 [0.94, 1.68]  x cval 0.71 [0.54, 0.96] | direct 4.37 [3.18, 4.94] p90 17.77 cm  model/direct 1.14 | in range 100.0 % [100.0, 100.0] | spread > val p90  50.0 % | settled start   0.0 %
      apple                          n   22 eps  14 | model 28.31 [26.55, 32.23] p90 37.74 cm  x val 7.10 [6.65, 8.08]  x cval 4.05 [3.80, 4.61] | direct 34.02 [31.63, 36.50] p90 38.95 cm  model/direct 0.83 | in range  59.1 % [50.0, 70.6] | spread > val p90  13.6 % | settled start  81.8 %
      apple, settled start           n   18 eps  13 | model 28.11 [26.98, 28.57] p90 37.31 cm  x val 7.04 [6.76, 7.16]  x cval 4.02 [3.86, 4.09] | direct 32.84 [30.97, 35.17] p90 38.64 cm  model/direct 0.86 | in range  61.1 % [45.0, 78.6] | spread > val p90   5.6 % | settled start 100.0 %
      apple, start not settled       n    4 eps   4 | model 34.09 [23.83, 45.24] p90 42.46 cm  x val 8.54 [5.97, 11.34]  x cval 4.88 [3.41, 6.47] | direct 37.84 [32.35, 52.73] p90 48.56 cm  model/direct 0.90 | in range  50.0 % [0.0, 100.0] | spread > val p90  50.0 % | settled start   0.0 %
      bowl                           n    5 eps   4 | model 54.69 [18.47, 57.75] p90 61.18 cm  x val 13.71 [4.63, 14.47]  x cval 7.82 [2.64, 8.26] | direct 46.07 [26.05, 46.40] p90 58.18 cm  model/direct 1.19 | in range  80.0 % [25.0, 100.0] | spread > val p90 100.0 % | settled start  60.0 %
      bowl, settled start            n    3 eps   2 | model 57.75 [57.75, 59.08] p90 62.32 cm  x val 14.47 [14.47, 14.81]  x cval 8.26 [8.26, 8.45] | direct 46.40 [46.40, 56.05] p90 62.10 cm  model/direct 1.24 | in range 100.0 % [100.0, 100.0] | spread > val p90 100.0 % | settled start 100.0 %
      bowl, start not settled        n    2 eps   2 | model 31.19 [18.47, 43.91] p90 41.37 cm  x val 7.82 [4.63, 11.00]  x cval 4.46 [2.64, 6.28] | direct 35.13 [26.05, 44.22] p90 42.40 cm  model/direct 0.89 | in range  50.0 % [0.0, 100.0] | spread > val p90 100.0 % | settled start   0.0 %
      salt                           n   13 eps   8 | model 28.19 [24.58, 30.65] p90 34.76 cm  x val 7.07 [6.16, 7.68]  x cval 4.03 [3.52, 4.38] | direct 33.00 [30.71, 35.27] p90 40.83 cm  model/direct 0.85 | in range  61.5 % [50.0, 80.0] | spread > val p90   7.7 % | settled start  92.3 %
      salt, settled start            n   12 eps   8 | model 28.43 [24.58, 30.65] p90 34.96 cm  x val 7.13 [6.16, 7.68]  x cval 4.07 [3.52, 4.38] | direct 32.73 [30.31, 35.27] p90 39.64 cm  model/direct 0.87 | in range  66.7 % [53.3, 88.9] | spread > val p90   0.0 % | settled start 100.0 %
      salt, start not settled        n    1 eps   1 | model 28.19 [nan, nan] p90 28.19 cm  x val 7.07 [nan, nan]  x cval 4.03 [nan, nan] | direct 41.06 [nan, nan] p90 41.06 cm  model/direct 0.69 | in range   0.0 % [nan, nan] | spread > val p90 100.0 % | settled start   0.0 %
      statue_cab                     n    1 eps   1 | model 6.31 [nan, nan] p90 6.31 cm  x val 1.58 [nan, nan]  x cval 0.90 [nan, nan] | direct 5.08 [nan, nan] p90 5.08 cm  model/direct 1.24 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab, settled start      n    0
      statue_cab, start not settled  n    1 eps   1 | model 6.31 [nan, nan] p90 6.31 cm  x val 1.58 [nan, nan]  x cval 0.90 [nan, nan] | direct 5.08 [nan, nan] p90 5.08 cm  model/direct 1.24 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
sensitivity, chain as recorded (unclipped) on the 35 segments with chain 2 (6 kept): model median 30.10 cm vs clipped 30.21 cm
```

### SONIC：eval_r2.txt 原文（md5 86b9ae78）

```
sonic: model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/sonic/models_s0/E3_ensemble_f1; val check: 2935 kept val segments, model median 4.252 cm (registered 4.25); chained val reaches (chain >= 1) 1378, model median 6.837 cm (the x cval denominator, reference only); val spread |pos_std| p90 6.277 cm; training band: 25046 training segments' command rows; training chain max 1, walked [0, 1]
records /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125/sonic/data/all.npz: 295 segments from 128 episodes; reach_no {0: 40, 1: 128, 2: 127}; fell / lost 1; label not settled 165; kept 129; chain in the record {0: 128, 1: 127, 2: 40} -> clipped to 1; walked 220
== reach 1 (pre-grasp, 第一下): 128 recorded (127 not fell / lost, 70 kept); settled start (all recorded) 17.2 % (22/128); in range (all recorded) 80.5 % (103/128); label settled (not fell) 55.1 %
    all kept                       n   70 eps  70 | model 2.93 [2.33, 3.28] p90 6.92 cm  x val 0.69 [0.55, 0.77]  x cval 0.43 [0.34, 0.48] | direct 5.50 [4.63, 8.37] p90 22.27 cm  model/direct 0.53 | in range  87.1 % [78.6, 94.3] | spread > val p90   2.9 % | settled start  15.7 %
    kept, settled start            n   11 eps  11 | model 3.65 [1.97, 6.41] p90 6.59 cm  x val 0.86 [0.46, 1.51]  x cval 0.53 [0.29, 0.94] | direct 8.13 [5.40, 11.48] p90 42.92 cm  model/direct 0.45 | in range  81.8 % [54.5, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
    kept, start not settled        n   59 eps  59 | model 2.62 [2.28, 3.23] p90 6.94 cm  x val 0.62 [0.54, 0.76]  x cval 0.38 [0.33, 0.47] | direct 5.09 [4.16, 8.86] p90 21.87 cm  model/direct 0.51 | in range  88.1 % [79.7, 94.9] | spread > val p90   3.4 % | settled start   0.0 %
    kept, in range                 n   61 eps  61 | model 2.59 [2.19, 3.33] p90 6.59 cm  x val 0.61 [0.52, 0.78]  x cval 0.38 [0.32, 0.49] | direct 5.40 [4.27, 8.61] p90 22.14 cm  model/direct 0.48 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  14.8 %
    kept, out of range             n    9 eps   9 | model 3.25 [2.99, 6.91] p90 8.10 cm  x val 0.77 [0.70, 1.62]  x cval 0.48 [0.44, 1.01] | direct 8.13 [4.54, 21.80] p90 22.13 cm  model/direct 0.40 | in range   0.0 % [0.0, 0.0] | spread > val p90  22.2 % | settled start  22.2 %
    out / in range (model median): 1.26 [0.97, 2.95]
    kept, chain 2 in the record (fed 1) n    0
    kept, chain <= 1 in the record n   70 eps  70 | model 2.93 [2.35, 3.33] p90 6.92 cm  x val 0.69 [0.55, 0.78]  x cval 0.43 [0.34, 0.49] | direct 5.50 [4.65, 8.37] p90 22.27 cm  model/direct 0.53 | in range  87.1 % [78.6, 94.3] | spread > val p90   2.9 % | settled start  15.7 %
    kept, spread > val p90         n    2 eps   2 | model 9.50 [7.16, 11.84] p90 11.38 cm  x val 2.24 [1.68, 2.79]  x cval 1.39 [1.05, 1.73] | direct 11.90 [3.27, 20.54] p90 18.81 cm  model/direct 0.80 | in range   0.0 % [0.0, 0.0] | spread > val p90 100.0 % | settled start   0.0 %
    kept, spread <= val p90        n   68 eps  68 | model 2.92 [2.30, 3.24] p90 6.62 cm  x val 0.69 [0.54, 0.76]  x cval 0.43 [0.34, 0.47] | direct 5.50 [4.54, 8.37] p90 22.53 cm  model/direct 0.53 | in range  89.7 % [82.4, 97.1] | spread > val p90   0.0 % | settled start  16.2 %
    all labels incl. unsettled (not fell) n  127 eps 127 | model 3.23 [2.92, 3.84] p90 14.53 cm  x val 0.76 [0.69, 0.90]  x cval 0.47 [0.43, 0.56] | direct 8.88 [6.68, 11.93] p90 33.50 cm  model/direct 0.36 | in range  80.3 % [74.0, 86.6] | spread > val p90   7.9 % | settled start  17.3 %
    contact during the reach (kept): furniture 7, target object 11, neither 54  [post hoc]
    kept, no contact [post hoc]    n   54 eps  54 | model 2.37 [2.13, 3.11] p90 6.39 cm  x val 0.56 [0.50, 0.73]  x cval 0.35 [0.31, 0.45] | direct 4.98 [4.05, 6.22] p90 20.69 cm  model/direct 0.48 | in range  88.9 % [79.6, 96.3] | spread > val p90   1.9 % | settled start  14.8 %
    kept, no contact, in range [post hoc] n   48 eps  48 | model 2.30 [2.03, 2.93] p90 6.38 cm  x val 0.54 [0.48, 0.69]  x cval 0.34 [0.30, 0.43] | direct 5.04 [4.05, 6.68] p90 19.32 cm  model/direct 0.46 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  14.6 %
    kept, furniture contact [post hoc] n    7 eps   7 | model 6.09 [3.20, 8.82] p90 10.03 cm  x val 1.43 [0.75, 2.07]  x cval 0.89 [0.47, 1.29] | direct 19.49 [8.86, 20.68] p90 22.38 cm  model/direct 0.31 | in range  85.7 % [57.1, 100.0] | spread > val p90  14.3 % | settled start  14.3 %
    joints out of the band (kept segments, count): waist_pitch 4, waist_roll 2, right_wrist_yaw 2, left_hip_yaw 1, right_elbow 1, right_wrist_pitch 1; joints out per segment median 0, max 3
    per task:
      candle                         n   16 eps  16 | model 1.92 [1.14, 2.28] p90 2.79 cm  x val 0.45 [0.27, 0.54]  x cval 0.28 [0.17, 0.33] | direct 3.66 [2.97, 4.03] p90 4.80 cm  model/direct 0.52 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      candle, settled start          n    0
      candle, start not settled      n   16 eps  16 | model 1.92 [1.14, 2.28] p90 2.79 cm  x val 0.45 [0.27, 0.54]  x cval 0.28 [0.17, 0.33] | direct 3.66 [2.97, 4.15] p90 4.80 cm  model/direct 0.52 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      apple                          n   13 eps  13 | model 4.50 [3.25, 7.08] p90 7.41 cm  x val 1.06 [0.77, 1.67]  x cval 0.66 [0.48, 1.04] | direct 5.73 [4.08, 11.48] p90 30.86 cm  model/direct 0.78 | in range  53.8 % [30.8, 76.9] | spread > val p90  15.4 % | settled start  53.8 %
      apple, settled start           n    7 eps   7 | model 4.14 [2.94, 6.41] p90 6.83 cm  x val 0.97 [0.69, 1.51]  x cval 0.61 [0.43, 0.94] | direct 5.73 [3.94, 11.48] p90 24.09 cm  model/direct 0.72 | in range  71.4 % [28.6, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      apple, start not settled       n    6 eps   6 | model 5.79 [3.12, 9.50] p90 9.50 cm  x val 1.36 [0.74, 2.24]  x cval 0.85 [0.46, 1.39] | direct 6.99 [3.67, 26.99] p90 26.99 cm  model/direct 0.83 | in range  33.3 % [0.0, 66.7] | spread > val p90  33.3 % | settled start   0.0 %
      bowl                           n    6 eps   6 | model 6.63 [4.38, 10.72] p90 10.83 cm  x val 1.56 [1.03, 2.52]  x cval 0.97 [0.64, 1.57] | direct 28.51 [21.97, 43.40] p90 43.40 cm  model/direct 0.23 | in range  66.7 % [32.9, 100.0] | spread > val p90   0.0 % | settled start  16.7 %
      bowl, settled start            n    1 eps   1 | model 6.59 [nan, nan] p90 6.59 cm  x val 1.55 [nan, nan]  x cval 0.96 [nan, nan] | direct 42.92 [nan, nan] p90 42.92 cm  model/direct 0.15 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start 100.0 %
      bowl, start not settled        n    5 eps   5 | model 6.68 [3.23, 14.75] p90 11.61 cm  x val 1.57 [0.76, 3.47]  x cval 0.98 [0.47, 2.16] | direct 23.45 [21.80, 43.88] p90 39.76 cm  model/direct 0.28 | in range  60.0 % [20.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      salt                           n    1 eps   1 | model 3.99 [nan, nan] p90 3.99 cm  x val 0.94 [nan, nan]  x cval 0.58 [nan, nan] | direct 4.54 [nan, nan] p90 4.54 cm  model/direct 0.88 | in range   0.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
      salt, settled start            n    0
      salt, start not settled        n    1 eps   1 | model 3.99 [nan, nan] p90 3.99 cm  x val 0.94 [nan, nan]  x cval 0.58 [nan, nan] | direct 4.54 [nan, nan] p90 4.54 cm  model/direct 0.88 | in range   0.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab                     n    3 eps   3 | model 3.20 [2.99, 4.99] p90 4.63 cm  x val 0.75 [0.70, 1.17]  x cval 0.47 [0.44, 0.73] | direct 20.68 [15.72, 24.92] p90 24.07 cm  model/direct 0.15 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab, settled start      n    0
      statue_cab, start not settled  n    3 eps   3 | model 3.20 [2.99, 4.99] p90 4.63 cm  x val 0.75 [0.70, 1.17]  x cval 0.47 [0.44, 0.73] | direct 20.68 [15.72, 24.92] p90 24.07 cm  model/direct 0.15 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      dumbbell                       n   18 eps  18 | model 2.40 [1.96, 3.57] p90 6.91 cm  x val 0.57 [0.46, 0.84]  x cval 0.35 [0.29, 0.52] | direct 6.68 [5.12, 8.86] p90 13.45 cm  model/direct 0.36 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  16.7 %
      dumbbell, settled start        n    3 eps   3 | model 1.97 [1.74, 3.65] p90 3.32 cm  x val 0.46 [0.41, 0.86]  x cval 0.29 [0.25, 0.53] | direct 8.61 [5.40, 11.46] p90 10.89 cm  model/direct 0.23 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      dumbbell, start not settled    n   15 eps  15 | model 2.43 [2.19, 4.06] p90 7.73 cm  x val 0.57 [0.52, 0.95]  x cval 0.36 [0.32, 0.59] | direct 6.68 [4.83, 9.05] p90 15.10 cm  model/direct 0.36 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      wcan                           n   13 eps  13 | model 2.37 [2.03, 3.37] p90 5.15 cm  x val 0.56 [0.48, 0.79]  x cval 0.35 [0.30, 0.49] | direct 5.60 [2.52, 12.23] p90 12.54 cm  model/direct 0.42 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      wcan, settled start            n    0
      wcan, start not settled        n   13 eps  13 | model 2.37 [2.03, 3.37] p90 5.15 cm  x val 0.56 [0.48, 0.79]  x cval 0.35 [0.30, 0.49] | direct 5.60 [2.52, 12.22] p90 12.54 cm  model/direct 0.42 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
== reach 2 (entry, 第二下): 127 recorded (127 not fell / lost, 39 kept); settled start (all recorded) 59.8 % (76/127); in range (all recorded) 63.8 % (81/127); label settled (not fell) 30.7 %
    all kept                       n   39 eps  39 | model 2.06 [1.79, 2.57] p90 4.42 cm  x val 0.49 [0.42, 0.61]  x cval 0.30 [0.26, 0.38] | direct 3.89 [2.91, 4.85] p90 8.89 cm  model/direct 0.53 | in range  79.5 % [66.7, 92.3] | spread > val p90   0.0 % | settled start  66.7 %
    kept, settled start            n   26 eps  26 | model 1.87 [1.38, 2.48] p90 4.47 cm  x val 0.44 [0.32, 0.58]  x cval 0.27 [0.20, 0.36] | direct 2.95 [2.62, 5.26] p90 16.25 cm  model/direct 0.63 | in range  88.5 % [76.9, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
    kept, start not settled        n   13 eps  13 | model 2.43 [2.13, 3.28] p90 4.31 cm  x val 0.57 [0.50, 0.77]  x cval 0.36 [0.31, 0.48] | direct 4.63 [3.89, 5.71] p90 8.30 cm  model/direct 0.52 | in range  61.5 % [38.5, 84.6] | spread > val p90   0.0 % | settled start   0.0 %
    kept, in range                 n   31 eps  31 | model 1.92 [1.60, 2.42] p90 4.19 cm  x val 0.45 [0.38, 0.57]  x cval 0.28 [0.23, 0.35] | direct 2.98 [2.62, 4.85] p90 8.83 cm  model/direct 0.65 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  74.2 %
    kept, out of range             n    8 eps   8 | model 2.87 [1.84, 5.57] p90 6.03 cm  x val 0.67 [0.43, 1.31]  x cval 0.42 [0.27, 0.82] | direct 4.64 [3.89, 8.11] p90 8.42 cm  model/direct 0.62 | in range   0.0 % [0.0, 0.0] | spread > val p90   0.0 % | settled start  37.5 %
    out / in range (model median): 1.49 [0.91, 2.90]
    kept, chain 2 in the record (fed 1) n    8 eps   8 | model 2.22 [1.38, 2.90] p90 3.24 cm  x val 0.52 [0.32, 0.68]  x cval 0.32 [0.20, 0.42] | direct 2.76 [1.45, 5.71] p90 13.35 cm  model/direct 0.80 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  50.0 %
    kept, chain <= 1 in the record n   31 eps  31 | model 1.94 [1.61, 2.89] p90 4.76 cm  x val 0.46 [0.38, 0.68]  x cval 0.28 [0.23, 0.42] | direct 3.94 [2.98, 5.19] p90 8.83 cm  model/direct 0.49 | in range  74.2 % [58.1, 87.1] | spread > val p90   0.0 % | settled start  71.0 %
    kept, spread > val p90         n    0
    kept, spread <= val p90        n   39 eps  39 | model 2.06 [1.79, 2.57] p90 4.42 cm  x val 0.49 [0.42, 0.61]  x cval 0.30 [0.26, 0.38] | direct 3.89 [2.90, 4.85] p90 8.89 cm  model/direct 0.53 | in range  79.5 % [66.7, 92.3] | spread > val p90   0.0 % | settled start  66.7 %
    all labels incl. unsettled (not fell) n  127 eps 127 | model 3.07 [2.75, 3.95] p90 10.91 cm  x val 0.72 [0.65, 0.93]  x cval 0.45 [0.40, 0.58] | direct 5.71 [4.49, 8.83] p90 35.01 cm  model/direct 0.54 | in range  63.8 % [55.9, 71.7] | spread > val p90  11.0 % | settled start  59.8 %
    contact during the reach (kept): furniture 2, target object 17, neither 20  [post hoc]
    kept, no contact [post hoc]    n   20 eps  20 | model 1.83 [1.38, 2.44] p90 3.54 cm  x val 0.43 [0.32, 0.57]  x cval 0.27 [0.20, 0.36] | direct 3.95 [2.82, 4.92] p90 6.50 cm  model/direct 0.46 | in range  65.0 % [45.0, 85.0] | spread > val p90   0.0 % | settled start  65.0 %
    kept, no contact, in range [post hoc] n   13 eps  13 | model 1.60 [0.97, 2.06] p90 2.77 cm  x val 0.38 [0.23, 0.49]  x cval 0.23 [0.14, 0.30] | direct 2.91 [2.54, 5.19] p90 6.11 cm  model/direct 0.55 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  84.6 %
    kept, furniture contact [post hoc] n    2 eps   2 | model 4.99 [4.76, 5.21] p90 5.17 cm  x val 1.17 [1.12, 1.23]  x cval 0.73 [0.70, 0.76] | direct 26.20 [24.39, 28.01] p90 27.65 cm  model/direct 0.19 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
    joints out of the band (kept segments, count): left_ankle_pitch 7, left_hip_roll 1, waist_roll 1; joints out per segment median 0, max 2
    per task:
      candle                         n   11 eps  11 | model 1.35 [1.00, 1.92] p90 2.42 cm  x val 0.32 [0.23, 0.45]  x cval 0.20 [0.15, 0.28] | direct 2.62 [1.86, 3.16] p90 3.94 cm  model/direct 0.52 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  81.8 %
      candle, settled start          n    9 eps   9 | model 1.35 [1.00, 1.92] p90 2.02 cm  x val 0.32 [0.23, 0.45]  x cval 0.20 [0.15, 0.28] | direct 2.54 [1.86, 2.71] p90 3.18 cm  model/direct 0.53 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      candle, start not settled      n    2 eps   2 | model 2.09 [0.90, 3.28] p90 3.04 cm  x val 0.49 [0.21, 0.77]  x cval 0.31 [0.13, 0.48] | direct 3.58 [3.16, 4.00] p90 3.92 cm  model/direct 0.58 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      apple                          n    6 eps   6 | model 2.02 [1.07, 2.66] p90 2.66 cm  x val 0.48 [0.25, 0.63]  x cval 0.30 [0.16, 0.39] | direct 2.16 [1.19, 3.87] p90 3.87 cm  model/direct 0.94 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  50.0 %
      apple, settled start           n    3 eps   3 | model 1.38 [0.76, 1.92] p90 1.81 cm  x val 0.32 [0.18, 0.45]  x cval 0.20 [0.11, 0.28] | direct 2.63 [1.70, 2.90] p90 2.84 cm  model/direct 0.52 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      apple, start not settled       n    3 eps   3 | model 2.43 [2.13, 2.90] p90 2.80 cm  x val 0.57 [0.50, 0.68]  x cval 0.36 [0.31, 0.42] | direct 1.19 [1.15, 4.85] p90 4.12 cm  model/direct 2.04 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      bowl                           n    1 eps   1 | model 4.04 [nan, nan] p90 4.04 cm  x val 0.95 [nan, nan]  x cval 0.59 [nan, nan] | direct 31.17 [nan, nan] p90 31.17 cm  model/direct 0.13 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start 100.0 %
      bowl, settled start            n    1 eps   1 | model 4.04 [nan, nan] p90 4.04 cm  x val 0.95 [nan, nan]  x cval 0.59 [nan, nan] | direct 31.17 [nan, nan] p90 31.17 cm  model/direct 0.13 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start 100.0 %
      bowl, start not settled        n    0
      salt                           n    1 eps   1 | model 2.31 [nan, nan] p90 2.31 cm  x val 0.54 [nan, nan]  x cval 0.34 [nan, nan] | direct 5.71 [nan, nan] p90 5.71 cm  model/direct 0.40 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
      salt, settled start            n    0
      salt, start not settled        n    1 eps   1 | model 2.31 [nan, nan] p90 2.31 cm  x val 0.54 [nan, nan]  x cval 0.34 [nan, nan] | direct 5.71 [nan, nan] p90 5.71 cm  model/direct 0.40 | in range 100.0 % [nan, nan] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab                     n    2 eps   2 | model 4.99 [4.76, 5.21] p90 5.17 cm  x val 1.17 [1.12, 1.23]  x cval 0.73 [0.70, 0.76] | direct 26.20 [24.39, 28.01] p90 27.65 cm  model/direct 0.19 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      statue_cab, settled start      n    2 eps   2 | model 4.99 [4.76, 5.21] p90 5.17 cm  x val 1.17 [1.12, 1.23]  x cval 0.73 [0.70, 0.76] | direct 26.20 [24.39, 28.01] p90 27.65 cm  model/direct 0.19 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      statue_cab, start not settled  n    0
      dumbbell                       n    9 eps   9 | model 2.89 [1.94, 4.17] p90 4.20 cm  x val 0.68 [0.46, 0.98]  x cval 0.42 [0.28, 0.61] | direct 5.19 [3.01, 6.19] p90 6.71 cm  model/direct 0.56 | in range  66.7 % [33.3, 100.0] | spread > val p90   0.0 % | settled start  66.7 %
      dumbbell, settled start        n    6 eps   6 | model 2.41 [1.38, 3.32] p90 3.32 cm  x val 0.57 [0.32, 0.78]  x cval 0.35 [0.20, 0.49] | direct 4.14 [2.96, 5.38] p90 5.38 cm  model/direct 0.58 | in range  66.7 % [33.3, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      dumbbell, start not settled    n    3 eps   3 | model 4.17 [2.21, 4.34] p90 4.31 cm  x val 0.98 [0.52, 1.02]  x cval 0.61 [0.32, 0.63] | direct 6.19 [4.23, 8.83] p90 8.30 cm  model/direct 0.67 | in range  66.7 % [0.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      wcan                           n    9 eps   9 | model 2.06 [1.60, 5.57] p90 5.88 cm  x val 0.49 [0.38, 1.31]  x cval 0.30 [0.23, 0.82] | direct 4.65 [1.77, 8.11] p90 8.32 cm  model/direct 0.44 | in range  44.4 % [11.1, 77.8] | spread > val p90   0.0 % | settled start  55.6 %
      wcan, settled start            n    5 eps   5 | model 2.06 [0.75, 7.11] p90 5.94 cm  x val 0.49 [0.18, 1.67]  x cval 0.30 [0.11, 1.04] | direct 5.59 [1.72, 8.11] p90 7.35 cm  model/direct 0.37 | in range  80.0 % [40.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, start not settled        n    4 eps   4 | model 2.21 [1.38, 5.57] p90 4.67 cm  x val 0.52 [0.33, 1.31]  x cval 0.32 [0.20, 0.82] | direct 4.64 [3.89, 9.16] p90 7.80 cm  model/direct 0.48 | in range   0.0 % [0.0, 0.0] | spread > val p90   0.0 % | settled start   0.0 %
== all reaches (via points, reach 1 and 2; 全部): 295 recorded (294 not fell / lost, 129 kept); settled start (all recorded) 35.3 % (104/295); in range (all recorded) 75.9 % (224/295); label settled (not fell) 43.9 %
    all kept                       n  129 eps  90 | model 2.62 [2.31, 3.18] p90 6.94 cm  x val 0.62 [0.54, 0.75]  x cval 0.38 [0.34, 0.47] | direct 5.22 [4.49, 6.20] p90 21.87 cm  model/direct 0.50 | in range  86.8 % [81.3, 92.0] | spread > val p90   1.6 % | settled start  31.8 %
    kept, settled start            n   41 eps  37 | model 2.06 [1.82, 3.65] p90 7.11 cm  x val 0.49 [0.43, 0.86]  x cval 0.30 [0.27, 0.53] | direct 5.19 [3.01, 5.91] p90 28.01 cm  model/direct 0.40 | in range  87.8 % [77.5, 95.4] | spread > val p90   0.0 % | settled start 100.0 %
    kept, start not settled        n   88 eps  83 | model 2.90 [2.37, 3.23] p90 6.88 cm  x val 0.68 [0.56, 0.76]  x cval 0.42 [0.35, 0.47] | direct 5.31 [4.65, 6.88] p90 20.58 cm  model/direct 0.55 | in range  86.4 % [78.9, 93.1] | spread > val p90   2.3 % | settled start   0.0 %
    kept, in range                 n  112 eps  81 | model 2.43 [2.22, 2.99] p90 6.67 cm  x val 0.57 [0.52, 0.70]  x cval 0.36 [0.32, 0.44] | direct 5.28 [4.21, 6.20] p90 22.05 cm  model/direct 0.46 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  32.1 %
    kept, out of range             n   17 eps  17 | model 3.23 [2.57, 5.57] p90 7.13 cm  x val 0.76 [0.61, 1.31]  x cval 0.47 [0.38, 0.82] | direct 5.09 [4.23, 8.88] p90 21.04 cm  model/direct 0.63 | in range   0.0 % [0.0, 0.0] | spread > val p90  11.8 % | settled start  29.4 %
    out / in range (model median): 1.33 [0.99, 2.29]
    kept, chain 2 in the record (fed 1) n    8 eps   8 | model 2.22 [1.38, 2.90] p90 3.24 cm  x val 0.52 [0.32, 0.68]  x cval 0.32 [0.20, 0.42] | direct 2.76 [1.19, 5.28] p90 13.35 cm  model/direct 0.80 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  50.0 %
    kept, chain <= 1 in the record n  121 eps  89 | model 2.91 [2.37, 3.23] p90 7.08 cm  x val 0.69 [0.56, 0.76]  x cval 0.43 [0.35, 0.47] | direct 5.40 [4.54, 6.45] p90 21.80 cm  model/direct 0.54 | in range  86.0 % [80.0, 91.7] | spread > val p90   1.7 % | settled start  30.6 %
    kept, spread > val p90         n    2 eps   2 | model 9.50 [7.16, 11.84] p90 11.38 cm  x val 2.24 [1.68, 2.79]  x cval 1.39 [1.05, 1.73] | direct 11.90 [3.27, 20.54] p90 18.81 cm  model/direct 0.80 | in range   0.0 % [0.0, 0.0] | spread > val p90 100.0 % | settled start   0.0 %
    kept, spread <= val p90        n  127 eps  89 | model 2.59 [2.31, 3.19] p90 6.75 cm  x val 0.61 [0.54, 0.75]  x cval 0.38 [0.34, 0.47] | direct 5.22 [4.35, 6.20] p90 21.94 cm  model/direct 0.50 | in range  88.2 % [82.6, 93.5] | spread > val p90   0.0 % | settled start  32.3 %
    all labels incl. unsettled (not fell) n  294 eps 127 | model 3.37 [2.99, 4.06] p90 19.98 cm  x val 0.79 [0.70, 0.95]  x cval 0.49 [0.44, 0.59] | direct 8.12 [6.03, 10.60] p90 33.53 cm  model/direct 0.41 | in range  75.9 % [71.4, 80.0] | spread > val p90   9.5 % | settled start  35.4 %
    contact during the reach (kept): furniture 23, target object 29, neither 80  [post hoc]
    kept, no contact [post hoc]    n   80 eps  62 | model 2.37 [2.10, 2.94] p90 6.37 cm  x val 0.56 [0.49, 0.69]  x cval 0.35 [0.31, 0.43] | direct 4.73 [4.08, 5.60] p90 17.00 cm  model/direct 0.50 | in range  83.8 % [75.9, 90.7] | spread > val p90   1.2 % | settled start  26.2 %
    kept, no contact, in range [post hoc] n   67 eps  55 | model 2.28 [2.03, 2.89] p90 6.38 cm  x val 0.54 [0.48, 0.68]  x cval 0.33 [0.30, 0.42] | direct 4.86 [4.00, 6.03] p90 17.37 cm  model/direct 0.47 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  26.9 %
    kept, furniture contact [post hoc] n   23 eps  20 | model 5.21 [3.20, 7.47] p90 21.04 cm  x val 1.23 [0.75, 1.76]  x cval 0.76 [0.47, 1.09] | direct 14.52 [8.24, 20.68] p90 27.39 cm  model/direct 0.36 | in range  95.7 % [85.7, 100.0] | spread > val p90   4.3 % | settled start  30.4 %
    joints out of the band (kept segments, count): left_ankle_pitch 7, waist_pitch 4, waist_roll 3, right_wrist_yaw 2, left_hip_yaw 1, left_hip_roll 1; joints out per segment median 0, max 3
    per task:
      candle                         n   27 eps  18 | model 1.61 [1.14, 2.22] p90 2.75 cm  x val 0.38 [0.27, 0.52]  x cval 0.23 [0.17, 0.32] | direct 3.16 [2.62, 3.86] p90 4.46 cm  model/direct 0.51 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  33.3 %
      candle, settled start          n    9 eps   9 | model 1.35 [1.00, 1.92] p90 2.02 cm  x val 0.32 [0.23, 0.45]  x cval 0.20 [0.15, 0.28] | direct 2.54 [1.65, 2.98] p90 3.18 cm  model/direct 0.53 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      candle, start not settled      n   18 eps  18 | model 1.92 [1.13, 2.28] p90 3.05 cm  x val 0.45 [0.27, 0.54]  x cval 0.28 [0.17, 0.33] | direct 3.66 [3.06, 4.02] p90 4.78 cm  model/direct 0.52 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      apple                          n   34 eps  20 | model 3.94 [3.09, 5.25] p90 10.67 cm  x val 0.93 [0.73, 1.23]  x cval 0.58 [0.45, 0.77] | direct 5.88 [5.12, 8.88] p90 21.07 cm  model/direct 0.67 | in range  82.4 % [70.6, 93.5] | spread > val p90   5.9 % | settled start  41.2 %
      apple, settled start           n   14 eps  10 | model 3.90 [2.43, 7.47] p90 20.21 cm  x val 0.92 [0.57, 1.76]  x cval 0.57 [0.36, 1.09] | direct 5.67 [3.74, 20.92] p90 27.62 cm  model/direct 0.69 | in range  85.7 % [70.6, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      apple, start not settled       n   20 eps  16 | model 4.07 [3.18, 5.86] p90 7.24 cm  x val 0.96 [0.75, 1.38]  x cval 0.60 [0.47, 0.86] | direct 6.23 [5.09, 8.88] p90 15.12 cm  model/direct 0.65 | in range  80.0 % [63.2, 95.0] | spread > val p90  10.0 % | settled start   0.0 %
      bowl                           n    8 eps   6 | model 6.06 [4.04, 6.91] p90 9.26 cm  x val 1.43 [0.95, 1.62]  x cval 0.89 [0.59, 1.01] | direct 27.31 [22.14, 38.72] p90 43.21 cm  model/direct 0.22 | in range  75.0 % [50.0, 100.0] | spread > val p90   0.0 % | settled start  25.0 %
      bowl, settled start            n    2 eps   2 | model 5.32 [4.04, 6.59] p90 6.34 cm  x val 1.25 [0.95, 1.55]  x cval 0.78 [0.59, 0.96] | direct 37.04 [31.17, 42.92] p90 41.75 cm  model/direct 0.14 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      bowl, start not settled        n    6 eps   6 | model 6.11 [2.41, 10.83] p90 10.83 cm  x val 1.44 [0.57, 2.55]  x cval 0.89 [0.35, 1.58] | direct 22.79 [20.51, 38.72] p90 38.72 cm  model/direct 0.27 | in range  66.7 % [33.3, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      salt                           n    6 eps   5 | model 2.38 [1.50, 24.88] p90 14.44 cm  x val 0.56 [0.35, 5.86]  x cval 0.35 [0.22, 3.64] | direct 5.13 [2.58, 31.36] p90 20.31 cm  model/direct 0.46 | in range  83.3 % [40.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      salt, settled start            n    0
      salt, start not settled        n    6 eps   5 | model 2.38 [1.50, 24.88] p90 14.44 cm  x val 0.56 [0.35, 5.86]  x cval 0.35 [0.22, 3.64] | direct 5.13 [2.58, 31.36] p90 20.31 cm  model/direct 0.46 | in range  83.3 % [40.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      statue_cab                     n    5 eps   3 | model 4.76 [3.20, 5.10] p90 5.12 cm  x val 1.12 [0.75, 1.20]  x cval 0.70 [0.47, 0.75] | direct 24.39 [20.06, 24.92] p90 26.78 cm  model/direct 0.20 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  40.0 %
      statue_cab, settled start      n    2 eps   2 | model 4.99 [4.76, 5.21] p90 5.17 cm  x val 1.17 [1.12, 1.23]  x cval 0.73 [0.70, 0.76] | direct 26.20 [24.39, 28.01] p90 27.65 cm  model/direct 0.19 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      statue_cab, start not settled  n    3 eps   3 | model 3.20 [2.99, 4.99] p90 4.63 cm  x val 0.75 [0.70, 1.17]  x cval 0.47 [0.44, 0.73] | direct 20.68 [15.72, 24.92] p90 24.07 cm  model/direct 0.15 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      dumbbell                       n   27 eps  21 | model 2.43 [2.08, 3.48] p90 5.04 cm  x val 0.57 [0.49, 0.82]  x cval 0.36 [0.30, 0.51] | direct 5.43 [4.71, 8.26] p90 10.93 cm  model/direct 0.45 | in range  88.9 % [77.4, 100.0] | spread > val p90   0.0 % | settled start  33.3 %
      dumbbell, settled start        n    9 eps   9 | model 1.97 [1.74, 3.49] p90 3.52 cm  x val 0.46 [0.41, 0.82]  x cval 0.29 [0.25, 0.51] | direct 5.33 [3.01, 8.61] p90 9.18 cm  model/direct 0.37 | in range  77.8 % [44.4, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      dumbbell, start not settled    n   18 eps  18 | model 2.51 [2.21, 4.11] p90 6.91 cm  x val 0.59 [0.52, 0.97]  x cval 0.37 [0.32, 0.60] | direct 6.44 [4.72, 8.83] p90 12.84 cm  model/direct 0.39 | in range  94.4 % [83.3, 100.0] | spread > val p90   0.0 % | settled start   0.0 %
      wcan                           n   22 eps  17 | model 2.23 [2.00, 3.37] p90 5.57 cm  x val 0.53 [0.47, 0.79]  x cval 0.33 [0.29, 0.49] | direct 5.15 [2.62, 9.16] p90 12.50 cm  model/direct 0.43 | in range  77.3 % [57.9, 92.0] | spread > val p90   0.0 % | settled start  22.7 %
      wcan, settled start            n    5 eps   5 | model 2.06 [0.75, 7.11] p90 5.94 cm  x val 0.49 [0.18, 1.67]  x cval 0.30 [0.11, 1.04] | direct 5.59 [1.72, 8.11] p90 7.35 cm  model/direct 0.37 | in range  80.0 % [40.0, 100.0] | spread > val p90   0.0 % | settled start 100.0 %
      wcan, start not settled        n   17 eps  17 | model 2.37 [1.96, 3.37] p90 5.56 cm  x val 0.56 [0.46, 0.79]  x cval 0.35 [0.29, 0.49] | direct 4.71 [2.88, 11.93] p90 12.54 cm  model/direct 0.50 | in range  76.5 % [52.9, 94.1] | spread > val p90   0.0 % | settled start   0.0 %
sensitivity, chain as recorded (unclipped) on the 40 segments with chain 2 (8 kept): model median 2.23 cm vs clipped 2.22 cm
```

### 冻结版 eval.txt 里途经点那一格（只作描述）

```
HEFT:
== via-point reach (info): 35 recorded (35 not fell / lost, 21 kept); settled start (all recorded) 100.0 % (35/35); in range (all recorded) 100.0 % (35/35); label settled (not fell) 60.0 %
    all kept                       n   21 eps  21 | model 27.87 [25.16, 28.18] cm  x val 6.98 [6.30, 7.06] | direct 31.32 [28.74, 33.00] cm  model/direct 0.89 | in range 100.0 % [100.0, 100.0] | spread > val p90   9.5 % | settled start 100.0 %
    all labels incl. unsettled (not fell) n   35 eps  35 | model 27.15 [24.52, 28.04] cm  x val 6.80 [6.14, 7.03] | direct 31.25 [29.63, 33.00] cm  model/direct 0.87 | in range 100.0 % [100.0, 100.0] | spread > val p90  17.1 % | settled start 100.0 %
SONIC:
== via-point reach (info): 40 recorded (40 not fell / lost, 20 kept); settled start (all recorded) 15.0 % (6/40); in range (all recorded) 100.0 % (40/40); label settled (not fell) 50.0 %
    all kept                       n   20 eps  20 | model 4.07 [2.70, 6.57] cm  x val 0.96 [0.63, 1.55] | direct 8.42 [5.75, 10.32] cm  model/direct 0.48 | in range 100.0 % [100.0, 100.0] | spread > val p90   0.0 % | settled start  20.0 %
    all labels incl. unsettled (not fell) n   40 eps  40 | model 7.41 [3.83, 12.18] cm  x val 1.74 [0.90, 2.87] | direct 9.75 [7.36, 17.09] cm  model/direct 0.76 | in range 100.0 % [100.0, 100.0] | spread > val p90  10.0 % | settled start  15.0 %
```

### 三个控制器汇总（数字从上面的原始输出抄来）

| 控制器 | 第一下 x val | 第一下 模型/直接 | 第二下 x val | 第二下 模型/直接 | 没碰东西（全部） | 碰到家具（全部） |
|---|---|---|---|---|---|---|
| ScaleBFM | 0.82 [0.68, 1.10] | 0.24 | 1.04 [0.76, 1.29] | 0.45 | 1.35 cm | 4.68 cm |
| HEFT | 7.24 [1.47, 8.73] | 0.89 | 0.57 [0.46, 1.33] | 0.69 | 3.11 cm | 28.45 cm |
| SONIC | 0.69 [0.55, 0.77] | 0.53 | 0.49 [0.42, 0.61] | 0.53 | 2.37 cm | 5.21 cm |

## 意外（三个控制器合起来看）

1. **"第二下明显变大"按登记的度量，三个控制器都不成立。** 预测是 1.5–3.5 倍 val，实际是 1.04、0.57、0.49 倍。不过第二下是短距离伸手，直接执行误差本来就小。换成模型 / 直接执行的比值来看，三个控制器各不相同：
   - ScaleBFM 从 0.24 升到 0.45，第二下更差；
   - SONIC 两下都是 0.53，不变；
   - HEFT 从 0.89 降到 0.69，第二下更好。
2. **误差大的伸手主要是手碰到了家具。** 没碰任何东西的伸手，三个控制器的模型误差是 1.35、3.11、2.37 cm。碰到家具的是 4.68、28.45、5.21 cm。途经点那一下，ScaleBFM 和 HEFT 全都碰了家具，模型误差 30.6 cm 和 27.9 cm，ScaleBFM 那一下比直接执行还差。
3. **HEFT 的问题主要出在身体上，不在模型上。**
   - 走路时有 61/210 局偏离参考超过 0.5 m；
   - 第一下停稳的只有 27 下，其中碰家具的 14 下误差 35 cm；
   - 不用走路的蜡烛任务，第一下是 1.41 倍，第二下是 0.51 倍。
   
   登记时预计常跑丢的是 SONIC，实际是 HEFT。
4. **SONIC 第一下出发前停稳的只有 17.2 %**，低于预测的 20–80 %。这是因为 SONIC 的参考延迟 0.94 s，而任务时间表本身没有等停稳。
5. **HEFT 第二下只有 36.8 % 落在训练范围里**，低于预测的 40–85 %。越界最多的是右腕 pitch 和 roll。

## 结论与下一步

**对照判据**（登记写的是"SONIC 和 HEFT 套用同样的倍数"）：

| 控制器 | 第一下 | 第二下 | 不在训练范围里的比例 |
|---|---|---|---|
| ScaleBFM | 0.82 倍，落在"≤ 1.5"格，能迁移 | 1.04 倍，落在"≤ 1.3"格。区间上沿 1.29，把没停稳的也算上是 1.36 倍，不稳 | 16 % / 24 % |
| SONIC | 0.69 倍，"≤ 1.5"格 | 0.49 倍，"≤ 1.3"格。算上没停稳的是 0.72 倍，同一格 | 19.5 % / 36.2 % |
| HEFT | 7.24 倍，"> 2"格，下一步是找出哪些量在范围外，在那里加采 | 0.57 倍，区间上沿 1.33，越过了 1.3 | 第二下 63.2 %，超过 50 %，触发"先查定义"。定义一致性检查（check 第 1–4 项）已经通过 |

**解读**
- **第一下，ScaleBFM 和 SONIC 能迁移到真实任务。** HEFT 的大倍数主要来自身体跑偏和碰家具：直接执行误差本身就有 32 cm，碰家具的那组模型误差是 35 cm。所以"在范围外的地方加采"不是 HEFT 的第一步，第一步应该是先查 HEFT 为什么跟不住 hardgrasp 的走路参考。
- **第二下，"data3 已经够用"这一格要打折扣。** 登记的度量用的是绝对误差，和伸手的长短混在一起。ScaleBFM 换成相对量后，第二下还是比第一下差将近一倍。所以 data3b 的价值照判据在合并训练里衡量：train3b 会用同一份记录离线重算。
- **被家具挡住的伸手，模型帮不上，有时还会更差。** 自身模型不看场景，这是设计如此。被挡住要靠规划时避开。以后在真实任务里做"直接执行 vs 模型修正"的对比时，碰家具的伸手单独报。

**证据强度**
- 每个控制器 210 局；每个控制器用 1 个模型（train3 frac 1.0 的 E3 集成）。
- 保留下来的伸手：ScaleBFM 226 下、HEFT 69 下、SONIC 129 下。
- HEFT 第二下只有 21 下，区间很宽。

**其他可能的解释**
- 只保留停稳的落点，会把难的伸手筛掉。第二下停稳的比例：ScaleBFM 61 %、HEFT 37.5 %、SONIC 30.7 %。被排除的伸手可能误差更大。算上没停稳的落点，三个控制器的第二下分别是 1.36、6.07、0.72 倍。
- 第二下的误差小，可能只是因为伸手距离短。

**四类陷阱**
- **benchmark 选择**：只跑了 7 个任务（hardgrasp 的正式任务集），其中 75 局一下都没伸。碗、盐瓶、雕像这几个难任务样本少。
- **数据泄漏**：真实局不在训练集里，训练范围用 data3 的训练集来算，没有泄漏。
- **指标误用**：r2 版加的 p90、参考分母、按任务拆分停稳，都是看到 ScaleBFM 的主数以后才加的；碰东西的拆分是事后观察。判据只按登记的分母，只算停稳的落点。
- **事后挑选**：只有一次运行、一个模型，没有挑 checkpoint。只保留停稳落点这一点，是登记里的定义，但两种取法都已经报了。

## 独立复核（2026-11-26，独立 agent，只读；原文照录）

总的结论是：数字基本都对得上；四条主要结论里只有"第一下能迁移"站得住，另外三条要么理由不成立，要么只对一部分伸手成立。

**一、数字核对**
- 冻结信息全部对得上：三份 eval_r2.txt 的 md5（19b8158f / c2fc61eb / 86b9ae78），三份 check.txt，三个模型和 data3 的 md5，三个 FROZEN_UTC（14:21:49Z / 15:34:18Z / 15:37:19Z），以及远端仓库脚本和 asrun 的 md5。
- 从 all.npz 独立重算了三个关键数：用同一个模型的预测，中位数自己算，区间按局重抽 2000 次。
  - ScaleBFM 第二下：3.19 cm，1.04 倍 val。我算的区间是 [0.76, 1.27]，json 里是 [0.764, 1.288]，差别只来自重抽的随机数。
  - 没碰东西 / 碰到家具（全部伸手）：ScaleBFM 1.35 / 4.68 cm，HEFT 3.11 / 28.45 cm，SONIC 2.37 / 5.21 cm，全部一致。
  - 模型/直接执行：第一下 0.24 / 0.89 / 0.53，第二下 0.45 / 0.69 / 0.53，一致。这个比值是两个中位数相除，不是逐下比值取中位数。val 里接着上一次的伸手，比值重算是 0.492。
  - 其他抽查一致：HEFT 走路时跑丢 61 局，第一下跑丢 17 局，74 局伸了手；成功 32 / 8 局；没伸手的 75 局是 45 + 26 + 4。
- 发现三处不一致：
  1. 看那一眼时的姿态差，前后两处数字不同。ScaleBFM 部分偏离第 3 条写的是 24–34 mm；后文和局文件写的是 HEFT 中位数 56.7 mm（最大 112.7）、SONIC 26.9 mm。前一处对 HEFT 是错的。
  2. 对 data3 的判断前后强度不一：ScaleBFM 小节写"data3 已经够用不成立"，结论写"要打折扣"。
  3. HEFT 第二下引用的证据不对。原文说触发了"先查定义"，并引用"check 第 1–4 项已通过"。但这几项查的是记录和在线计算是否自洽，不是记录特征和训练数据的定义是否一致。真正能消掉这一条的是：两边用的是同一个 `d3_collect.segments`，walk 特征两边都缺省为 0；越界的是右腕 pitch/roll，更像手被卡住，不像定义错了。结论可以不变，引用的证据要换。

**二、四类陷阱**
1. **benchmark 选择**
   - 7 个任务是 hardgrasp 事先定下的，不是挑的。75 局没伸手的情况三个控制器完全相同，和模型无关。
   - 但所有伸手合在一起取中位数，容易的任务占了大头：ScaleBFM 第一下的 117 下里，蜡烛、哑铃、浇水壶占 67 下。
   - 按任务分别取中位数再取中位数，ScaleBFM 第一下是 1.26 倍，7 个任务里 3 个超过 2 倍：苹果 2.3、碗 8.6、雕像 6.7。SONIC 这样算是 0.75 倍，很稳。
   - HEFT 只剩走路没跑丢的局，样本是"幸存者"。
2. **数据泄漏**：没发现。
   - data3 在空场景（EmptyEnv）里采，真实局不在训练集里；训练范围和 val 分位只用了 data3 的 train/val。
   - 接触计数在 14:21 冻结的 lr_run.py 里就有，正式运行 15:26 才开始，不是看了结果才加的，只是没写进登记。
3. **指标误用：有两处。**
   - **换了指标。** 判据登记的是"x val"，结论却用事后加的"模型/直接"，把"≤ 1.3 倍"那一格改读成"打折扣"，属于看完结果换指标。作者披露了，但换上的指标本身也不合适，见第三节。
   - **"在不在训练范围里"只看关节位置，没看速度。**
     - data3 每一下都先等停稳再发命令：pre_wait_s 全部大于 0，中位数 1.2 s；训练数据里发命令时手速超过 2 cm/s 的一次也没有。真实局按固定时间表发命令，不等停稳。
     - 把"手速超过训练的 99.5 分位"也算越界以后，ScaleBFM 第一下（停稳落点）在范围里的比例从 87 % 降到 71 %。在范围里的误差 1.56 cm，不在的 10.1 cm，是 6.5 倍；登记的口径只显示 3.8 倍。
     - HEFT 第一下，关节和速度都在范围里的只有 7 %。
     - 实验问题的后一半"起始状态在不在训练范围里"，登记的口径明显低估了越界。
4. **事后挑选**
   - 只留停稳落点有偏差。第二下停稳的比例：ScaleBFM 61 %、HEFT 37.5 %、SONIC 30.7 %。两边的筛法也不一样严：这里的停稳窗口截到下一条命令为止，data3 最多等 3 s，val 里 97.8 % 都停稳了。把没停稳的也算上，ScaleBFM 第二下是 1.36 倍，越过 1.3。
   - 按接触拆分和 r2 加的格子，本身不改判据的读法；但"碰家具"那一格被用来支撑一条主要结论，问题见第三节。

**三、两个重点**

**(a) 第二下的分母**
- "分母和伸手长短混在一起"在数据上不成立：val 里模型误差几乎不随距离变短而变小。ScaleBFM 的 val 误差：

  | 伸手距离 | 模型误差 |
  |---|---|
  | 10 cm 以内（只有 18 下） | 3.2–4.0 cm |
  | 10–15 cm | 1.94 cm |
  | 15–20 cm | 2.54 cm |
  | 30–45 cm | 2.89 cm |
  | 45 cm 以上 | 3.25 cm |

  按距离配对 val，第二下是 1.06 倍 [0.77, 1.44]。
- 真正该配的是"接着上一次"。第二下全是 chain ≥ 1，而 val 里接着上一次的误差是第一次的三倍（5.2 cm 对 1.7 cm）。三种配法：

  | 配法 | ScaleBFM 第一下 | ScaleBFM 第二下 | SONIC 第一下 | SONIC 第二下 |
  |---|---|---|---|---|
  | 只按距离 | 0.87 | 1.06 | 0.74 | 0.59 |
  | 按 chain 和是否走路 | 1.12 [0.91, 1.50] | 0.61 [0.45, 0.74] | 0.88 | 0.30 |
  | 按 chain 和距离 | 1.08 | 0.82 | 0.95 | 0.43 |

- 所以登记用的全 val 分母本来就偏向第一下、亏待第二下。作者换上的"模型/直接"更不合适：直接执行误差随伸手变短而变小，模型误差却有 1.5–4 cm 的底，比值会机械地变大。第二下的 0.45 和 val 里接着上一次的 0.49 差不多，看不出"短距离进入"有什么特殊缺口。

**(b) 碰家具**
- 接触怎么算：`lr_run.py` 第 229–231 行，`hg_run.py` 的 `arm_scene_contact`（第 209 行）。
  - 从命令之后到落点那一行，每 20 ms 一帧，只要右手、腕、肘碰到任何"不是机器人、不是目标物体"的东西就记一帧，记到 1 帧就算"碰家具"。
  - 这包括地面和房间本身，蜡烛任务碰到的是名为 "bathroom" 的东西。
  - 轻擦一下和整段卡死算同一类。
  - 窗口越长越容易碰上：ScaleBFM 第一下，碰家具组的窗口中位数 230 帧，没碰组 185 帧。
- 查了几何，倒因为果不是主要问题。碰家具的伸手，实际落点沿模型预测的方向"走不到"：ScaleBFM 第二下只走到预测的 58 %，HEFT 第一下只走到 5 %、手比预测低 18 cm，横向偏离只有 2–4 cm；没碰的组走到 99–100 %。这是被挡住的样子，不是先伸偏再碰上。
  - ScaleBFM 第二下的接触只有 5 % 是上一下带过来的，即上一下结束时手还贴着；HEFT 第二下有 83 % 是上一下就卡着带过来的。
- 真正的问题是这个标签太宽，而且和任务绑在一起。
  - ScaleBFM 第一下和第二下里，碰了家具的有 130 下，其中 126 下模型比直接执行准，79 下模型误差 ≤ 5 cm。哑铃、浇水壶的第一下碰了家具，模型误差 1.8–2.5 cm，直接执行是 12–14 cm。
  - 误差超过 10 cm 的 33 下里，31 下碰了家具。所以"大误差几乎都碰了家具"是对的，"碰了家具模型就帮不上"是错的。
  - 真正帮不上的是整段卡死、接触占窗口一半以上的伸手：这类伸手 ScaleBFM 的模型/直接是 0.53，HEFT 是 0.91。它们集中在苹果、碗、盐瓶（经过途经点后手卡在厨房台面下）和柜子里的雕像。
  - "没碰东西"那一组也被任务带偏：ScaleBFM 第一下没碰组 52 下里，有 18 下是最容易的蜡烛。

**(c) HEFT**
- 7.24 倍主要来自 14 下整段卡死的伸手：接触占窗口 97 %，手几乎没动。这部分确实不是模型的问题。
- 但剩下 13 下干净的伸手（都是蜡烛：不走路、不碰东西、关节都在范围里），模型误差 5.62 cm，比直接执行的 4.51 cm 还差。同一个任务上，ScaleBFM 和 SONIC 的模型是 0.80 和 2.03 cm。
- 这 13 下都是手还在动时发的命令，手速都超过 HEFT 训练数据的 99.5 分位；集成成员分歧中位数 7.4 cm，超过 val 的 90 分位 6.1 cm。
- 这正是登记里"> 2 倍 → 找出哪些量落在范围外，在那里加采"要找的东西，只看关节的口径没看到。

**四、可能推翻结论的实验**

1. **让机器人手臂穿过家具**（检验结论 3、4）
   - 做法：同样 210 局 × 3 个控制器，种子、那一眼、时间表都不变。只关掉右手、腕、肘和非目标场景物体之间的碰撞，脚和地面、手和目标物体照常碰。
   - 为什么能单独看家具的作用：运行是确定的，每局第一次碰家具之前，两次运行逐帧相同（用 check 第 5 项的方法核对）。所以那一下的起始状态、命令和模型预测都完全一样，只有落点变了。
   - 看什么：每局"第一次碰家具的那一下"，穿过家具后的预测误差中位数和区间，分控制器、分"轻碰 / 卡死"报。
   - 怎么判：降到 ≤ 1.5 倍 val（ScaleBFM ≤ 4.6 cm、HEFT ≤ 6.0 cm），结论 3、4 成立；仍 > 2 倍 val（ScaleBFM > 6.2 cm、HEFT > 8.0 cm），说明误差不是家具造成的，两条都被推翻。
   - 成本：按 35 个进程，ScaleBFM 和 HEFT 各约 5 分钟，SONIC 约 12 分钟。
2. **每条命令前先等停稳**（检验结论 1、2、4）
   - 做法：同样的局，每条命令前按 data3 的规则等停稳，最多 3 s；落点也按 data3 最多等 3 s。其余不变。
   - 看什么：第二下停稳落点的比例（应该接近 95 %）；第一下和第二下的 x val，以及按 chain + 距离配对的倍数；HEFT 蜡烛第一下的模型误差和直接执行误差。
   - 怎么判：
     - 第二下几乎全部落点算进来后仍 > 2 倍 val，或高于配对 val，说明 data3 确实不够，复核人的判定被推翻；≤ 1.3 倍且不高于配对 val，"打折扣"被推翻。
     - 第一下 ScaleBFM > 4.6 cm 或 SONIC > 6.4 cm，说明原来的 0.82 是靠只留停稳落点得来的，"能迁移"被推翻。
     - HEFT 蜡烛仍不低于直接执行，说明 HEFT 的模型本身弱，结论 4 被推翻；降到 ≤ 1.5 倍 val 且明显低于直接执行，说明问题出在"边动边发命令"，属于时间表和训练数据覆盖的问题。
   - 补充：train3b 在同一份记录上离线评估时，看第二下改进了多少，要同时看 val 里接着上一次的伸手改进了多少，才能分清补的是"短距离进入"的缺口，还是只改善了"接着上一次"这一类伸手。

**五、判定**

| 结论 | 判定 | 理由 |
|---|---|---|
| ScaleBFM 和 SONIC 的第一下能迁移 | 支持，要写清适用范围 | 换哪种分母都不超过 1.5 倍：ScaleBFM 0.82–1.12，SONIC 0.69–0.95，算上没停稳的也是。但这只是 7 个任务合起来的中位数：ScaleBFM 有 3 个任务超过 2 倍，手还在动时发命令的第一下误差有 10 cm |
| "data3 已经够用"要打折扣 | 不支持，理由不成立 | "模型/直接"会因为短伸手的直接误差小而机械变大。按 chain 和距离配对 val，第二下是 0.61–1.06 倍，不比第一下差。站得住的只有"39 % 没停稳的落点被排除"这一条顾虑，它只能让"够用"变成暂定，撑不起"打折扣" |
| 被家具挡住的伸手，模型帮不上 | 证据不足，只对整段卡死的成立 | 落点沿预测方向走不到，确实是被挡住，不是倒因为果。但"碰家具"这个标签连轻擦和地面都算进去了，ScaleBFM 碰家具的 130 下里有 126 下模型比直接执行准 |
| HEFT 的大倍数主要不是模型的问题 | 证据不足，一半对 | 7 倍主要来自 14 下卡死的伸手，这部分对。但 13 下干净的伸手模型比直接执行还差，而且全是手在动时发的命令，训练数据没覆盖这种起始状态，这正是登记里"加采"那一步该查的 |

## 复核后的更正（2026-11-26）

1. **姿态差的数字。** ScaleBFM 部分偏离第 3 条的"24–34 mm"只对 SONIC 成立。HEFT 是中位数 56.7 mm，最大 112.7 mm。
2. **收回"data3 已经够用要打折扣"。** "模型/直接"这个比值不适合比较长短不同的伸手。按 chain 和距离配对 val 以后，第二下是 0.61–1.06 倍，并不比第一下差。按登记的判据，第二下落在"≤ 1.3 倍"那一格，即"data3 已经够用"，但这个结论只是暂定：第二下有 39 % 没停稳的落点被排除了（ScaleBFM），算上它们是 1.36 倍。等"命令前先等停稳"的运行出来再定。
3. **"被家具挡住，模型帮不上"缩小为"整段卡死的伸手，模型帮不上"。** 这类伸手的接触占窗口一半以上，ScaleBFM 的模型/直接是 0.53，HEFT 是 0.91。轻碰的伸手模型照样有用：ScaleBFM 碰家具的 130 下里，126 下模型比直接执行准。现在的接触标签把地面、房间和轻擦都算了进去，以后要改成"接触占窗口的比例"。
4. **HEFT 的结论改成一半对。** 7.24 倍主要来自卡死的伸手，这一半成立。另一半是：13 下干净的蜡烛伸手，模型比直接执行还差（5.62 cm 对 4.51 cm）。这些都是手还在动时发的命令，手速超出训练范围。所以 HEFT 下一步要查的是"边动边发命令"，不是"在哪里加采关节姿态"。
5. **"在训练范围里"以后要同时看关节速度。** 只看关节位置会低估越界：加上手速以后，ScaleBFM 第一下在范围里的比例从 87 % 降到 71 %，在范围里和不在范围里的误差是 1.56 cm 对 10.1 cm。
6. **HEFT 第二下"先查定义"那一格的依据换成：** 两边用的是同一个 `d3_collect.segments`；越界的是右腕 pitch/roll，像手被卡住，不像定义错了。
7. **适用范围。** 第一下能迁移，指的是 7 个任务合起来的中位数。ScaleBFM 按任务分别取中位数再取中位数是 1.26 倍，苹果、碗、雕像三个任务超过 2 倍。

**下一步**
- 复核人的实验 2（每条命令前先等停稳）并进 LML 的"直接执行 vs 模型修正"对比里一起做：它的直接执行列就是实验 2，模型列用 train3b 选出来的模型。这样也满足用户"避免没停稳的抓取"的要求。
- 实验 1（手臂穿过家具）成本小，作为诊断，等用户决定做不做。
