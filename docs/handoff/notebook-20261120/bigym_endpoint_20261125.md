# bigym_endpoint_20261125：BiGym 自身模型的终点精度——换成"下一步开始那一刻的手"做标签，再在没用过的局上和冻结的常数比

- 日期：2026-11-25（设计、预测、判据；跑之前写下，单独提交）
- 机器 / GPU：4090，0 号卡（训练、渲染、已有的 BiGym 规划服务）。1–3 号卡在跑 SIMPLE 的精度测试，不用。
- 代码版本：`selfwm` 加未提交的 `scripts/experiments/bigym_20261124/`（`bigym_data.py` 加一个终点标签的选项，见做法；其余同 [bigym_selfmodel_20261124](bigym_selfmodel_20261124.md)），跑时抄进输出目录的 `asrun/`，md5 记在结果里。`via_point.py`、`via_model.py` 一行不改。
- 配置：BiGym 2.0 自带的控制器，`--stance` 流程，`reach_target_multi_modal` 和 `drawer_top_open`；只用开发种子；不用仿真真值做决定，不用伺服和查表。
- 关联：
  - [bigym_selfmodel_20261124](bigym_selfmodel_20261124.md) 和它的独立复核：模型在扫过误差上赢过所有常数；抽屉的终点照登记的量法是常数赢（模型 0.59、留一常数 0.28 cm）。复核的离线检查支持"抽屉的停稳点没停稳"的解释：真实局伸手后 0.5 s 就往里推，那时手还在动，而训练标签是停稳 1 s 以后的点。这里做的是复核建议的 (b)1 和 (a)；
  - 用户 2026-11-24 定的论文主论断是一次命令的精度（DECISIONS 11-24），所以终点是要证明的量。

## 假设

1. **标签对上流程以后，抽屉的终点模型能赢常数。** 训练标签从"停稳后的点"换成"伸手结束后 0.5 s 内手的平均位置"，这和真实抽屉局的量法完全一样，也是往里推开始前手所在的地方。用新标签训练的模型，在现有 20 局抽屉真实局上，终点比留一常数准。
2. **模型的优势在没用过的局上还在。** 常数用现有 38 局冻结下来，在 160 局新的开发局上，模型在扫过误差和终点上都比它准。

## 做法

### 第 1 部分：换标签、重新训练

- **新标签**：`bigym_data.py build` 加选项 `--end early`。每条路线的窗口截到参考时长加 0.5 s，"停稳点"取这个窗口最后 5 个 10 Hz 时刻的平均，也就是伸手结束后 0.5 s 内的平均。
  - 训练数据和真实局用同一个定义。
  - 抽屉真实局的窗口本来就是这样，数不变。伸手任务真实局原来的窗口到参考时长加约 0.94 s，现在也截到 0.5 s。
  - 前 24 个时刻的偏移不变。
- **数据**：同一批录像（`outputs/bigym_20261124/selfmodel/films/`，9 999 条），同样的 10 桶划分，用新标签重新 `build`。
- **训练**：`via_point.py train`，全用默认设置（5 个成员，512×4，30000 步），0 号卡。模型叫 M2，原来的叫 M。
- **评测**：
  - 现有 38 局真实局（`sm_real/`）用新标签重新 `build`；
  - 跑 `via_point.py eval` 和 `bigym_data.py loo`（参考、留一常数、M2），按任务分开；
  - 测试桶（9 号）用 `bigym_test.py`。

### 第 2 部分：冻结常数，新局

- **新局**：开发种子 720020–720099，两个任务各 80 局，J，`--stance`，带录像，和现有 38 局同一套流程。走路途中就成功、没有伸手的局照旧跳过，记数。
- **冻结的常数**：开跑前从现有 38 局按任务算好、存成文件（每个任务一份，两种标签各一份），之后不改。
- **比较**：新局上参考、冻结常数、M（原标签）、M2（新标签）。
  - 扫过误差用原标签的数据比 M 和常数；
  - 终点用新标签的数据比 M2 和常数；
  - 按任务分开报；另报留一常数（新局内部）作参照。
- J 在新局上的成功局数照报。

## 度量

- **终点误差**：手掌三个球的平均位置，预测的终点对实际的终点，cm。新标签下就是"伸手结束后 0.5 s 内的平均"。
- **扫过区域误差**：`via_model` 的定义，cm。
- **降幅** = 1 − 模型中位数 ÷ 对手中位数，95 % 自助区间按局重抽。
- 每个比较都报模型更好的局数。

## 预测（跑之前写，没看过任何新标签的结果）

第 1 部分（现有 38 局）：
- 抽屉终点：参考和留一常数不变（2.14、0.28 cm，窗口本来就一样）。M2 在 0.08 到 0.25 cm 之间（复核按训练数据的比例折算 M 得到 0.11）；M2 对留一常数的降幅 0.2 到 0.7。
- 伸手任务终点（新标签）：参考 2.0 到 3.0 cm，留一常数 0.5 到 1.0，M2 0.15 到 0.4。
- M2 的扫过误差（全部 38 局）：0.2 到 0.5 cm。复核按比例折算时升到 0.53，但重新训练会把最后一段学对。
- 测试桶：M2 的终点 0.1 到 0.3 cm，扫过误差 0.4 到 0.6 cm。

第 2 部分（160 局新局）：
- J 成功：伸手 76 到 80 / 80；抽屉 15 到 35 / 80（现有 20 局是 6 / 20）。
- 没有伸手被跳过的伸手局：0 到 10 局。
- 扫过误差：冻结常数 0.85 到 1.3 cm，M 0.2 到 0.4 cm，降幅 0.6 到 0.8。
- 终点：
  - 抽屉 M2 0.08 到 0.3 cm，冻结常数 0.25 到 0.5 cm；
  - 伸手任务 M2 0.15 到 0.4 cm，冻结常数 0.5 到 1.0 cm。
- 冻结常数在新局上比在原 38 局上的留一常数差，但不超过 1.5 倍。

## 判据（什么结果会让我改做法，下一步做什么）

- **第 1 部分**：抽屉终点 M2 对留一常数的降幅不到 0.1，或区间包含 0。
  - 说明即使标签对上，抽屉的终点模型也赢不了常数。BiGym 的 S 只在伸手任务的终点上做，抽屉不做；报告用户，论文里 BiGym 只写伸手任务。
- **第 2 部分**：在新局上，任一任务、扫过误差（M）或终点（M2），模型对冻结常数的降幅不到 0.1 或区间包含 0。
  - 说明模型的优势在没用过的局上不稳；那个任务、那个量上不写模型的贡献。
- **冻结常数在新局上比原 38 局的留一常数差 1.5 倍以上**：真实局之间的差别比 38 局显示的大，常数在新局上失效。这是对模型有利的结果，但要先查新局和旧局的站位、目标分布是不是一样，再下结论。
- **两部分都过**：写 S 在 BiGym 上的登记。抽屉的 S 用 M2，瞄准往里推开始那一刻的手；伸手任务用 M2 或 M，在开发种子上定；对手是 A、J、冻结常数修正、上限；官方种子 620000–620099 只跑一次。

## 预测修订

**跑完以后记（实现细节，不改预测和判据）**：
- `--end early` 截在伸手最后一步之后 0.5 s，不是截在 `dur_s` 加 0.5 s。`dur_s` 只到 0.01 s，截在那里会少一步、过不了长度检查。截在最后一步，抽屉真实局的窗口和原来逐位相同。
- 新局里伸手任务有 9 局被跳过：6 局 `no_reach`（走路途中成功），3 局 `short`（伸手途中就满足了成功保持）。`short` 是转换脚本原有的规则，不是新加的。
- 新局用 4 个进程一个任务、各 20 个种子跑，每个进程单独的 `--out`。

## 结果（贴原始输出，不转述）

输出都在 4090 的 `outputs/bigym_20261124/`：`endpoint/`、`selfmodel/data_early/`、`selfmodel/model_early/`、`sm_real/replay_early/`、`sm_real2/`，原有的目录一个没覆盖。

- 代码：`bigym_data.py` e876ccd0、`bigym_test.py` b0ad7cb8，其余不变（`bigym_bench.py` 5a6b98a0、`via_point.py` addad976、`via_model.py` e4a2e9f9）。
- 冻结的常数写于 4090 时钟 2026-11-24T18:42:55Z，在任何新局开跑之前；用之前 `md5sum -c` 通过。

### 代码和逐位检查

`endpoint/verify.txt`：

```
real 38, table keys 22 differing: [] | arm keys ['X', 'jnt', 'link', 'link_bigym', 'local', 'radius'] differing: []
train 50: seeds/k match True | table keys differing: []
train 50: arm X segments identical True | a, b identical True | radius/link/local/jnt identical True
real 38 early: x, ref, off[:, :24] identical True True True | settled point changed on 13 of 38 | window ticks before/after, median 13.0 12.0
    reach 18 settled point identical on 5 | arm segments identical on 5 | ticks after the command (median) before/after 14.0 12.0
    drawer 20 settled point identical on 20 | arm segments identical on 20 | ticks after the command (median) before/after 13.0 13.0
train 50 early: x, ref, off[:, :24] identical True True True | settled point changed on 50 of 50 | window ticks before/after, median 27.0 19.5
    drawer 50 settled point identical on 0 | arm segments identical on 0 | ticks after the command (median) before/after 27.0 19.5
```

`endpoint/freeze.txt`：

```
outputs/bigym_20261124/endpoint/frozen_orig.npz: per-task mean offsets of outputs/bigym_20261124/sm_real/replay (table.npz md5 2bfd26ec94dcb9fad606ef0c9ee2b2ed): reach 18 episodes, drawer 20; the end point's offset (palm mean, cm): reach [-1.6, -0.66, -1.95], drawer [-0.95, -0.68, -1.74]
outputs/bigym_20261124/endpoint/frozen_early.npz: per-task mean offsets of outputs/bigym_20261124/sm_real/replay_early (table.npz md5 92c18e570155ce13cd7c02a50bf6839d): reach 18 episodes, drawer 20; the end point's offset (palm mean, cm): reach [-1.44, -0.62, -1.94], drawer [-0.95, -0.68, -1.74]
```

```
da109951ee1ff582fca8e235533f5002  frozen_orig.npz
0171a70576246c8c2c52f63a7fa28803  frozen_early.npz
2026-11-24T18:42:55Z
```

### 第 1 部分（现有 38 局，新标签）

`endpoint/build_replay_early.txt`：

```
outputs/bigym_20261124/sm_real/replay_early: 38 paths ({'film': 0, 'short': 0, 'no_reach': 2} skipped), tasks [18, 20], buckets [0, 0, 0, 0, 0, 0, 0, 0, 0, 38]; reference-vs-actual: swept-region error median 2.59 cm (p90 7.08), palm 1.95 cm, palm settled point 2.25 cm (p90 2.65); our FK against the simulator's right_end_effector: max 0.000 mm
  reach: 18 paths, swept median 4.82 cm (p90 7.96), palm settled median 2.45 cm, duration median 0.60 s
  reach, reach 1: 18 paths, swept median 4.82 cm (p90 7.96), palm settled median 2.45 cm, duration median 0.60 s
  drawer: 20 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.14 cm, duration median 0.70 s
  drawer, reach 1: 20 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.14 cm, duration median 0.70 s
```

`endpoint/build_data_early.txt`：

```
outputs/bigym_20261124/selfmodel/data_early: 9999 paths ({'film': 0, 'short': 0, 'no_reach': 0} skipped), tasks [4999, 5000], buckets [1000, 1000, 999, 1000, 1000, 1000, 1000, 1000, 1000, 1000]; reference-vs-actual: swept-region error median 3.76 cm (p90 9.73), palm 2.82 cm, palm settled point 2.66 cm (p90 8.79); our FK against the simulator's right_end_effector: max 0.000 mm
  reach: 4999 paths, swept median 3.86 cm (p90 11.70), palm settled median 2.82 cm, duration median 1.08 s
  reach, reach 1: 2500 paths, swept median 3.88 cm (p90 11.23), palm settled median 2.80 cm, duration median 1.08 s
  reach, reach 2: 2499 paths, swept median 3.84 cm (p90 12.07), palm settled median 2.83 cm, duration median 1.08 s
  drawer: 5000 paths, swept median 3.67 cm (p90 7.69), palm settled median 2.49 cm, duration median 1.29 s
  drawer, reach 1: 2500 paths, swept median 3.76 cm (p90 7.88), palm settled median 2.48 cm, duration median 1.25 s
  drawer, reach 2: 2500 paths, swept median 3.63 cm (p90 7.59), palm settled median 2.50 cm, duration median 1.32 s
```

`selfmodel/model_early/train.out`：

```
datasets ['outputs/bigym_20261124/selfmodel/data_early']; settings {'dropout': 0.1, 'wd': 0.0001, 'seed': 0, 'width': 512, 'depth': 4, 'members': 5, 'iters': 30000, 'dev': 'cuda'}
paths 9999: train 7999, validation for early stopping 498, for calibration 502, test 1000, sector hold-out 0
rows: train 199975, early-stopping validation 12450; input 336; validation squared offset 4.05 cm2 a coordinate with no model
member 0: validation squared offset 0.159 cm2 at iteration 5200 (first check 0.411, last 0.169; 40s)
member 1: validation squared offset 0.166 cm2 at iteration 17600 (first check 0.403, last 0.180; 77s)
member 2: validation squared offset 0.143 cm2 at iteration 24800 (first check 0.475, last 0.149; 117s)
member 3: validation squared offset 0.142 cm2 at iteration 18600 (first check 0.431, last 0.144; 162s)
member 4: validation squared offset 0.164 cm2 at iteration 18000 (first check 0.458, last 0.167; 235s)
ensemble's members: mean validation squared offset 0.155 cm2
direction: 7190 of 7999 training paths have an out-of-bag prediction
radius[reference] member 0: best validation loss 0.0864 at iteration 1000 (16s)
radius[reference] member 1: best validation loss 0.0796 at iteration 1000 (31s)
radius[reference] member 2: best validation loss 0.0781 at iteration 1000 (45s)
radius[reference]: scale 1.290; calibration paths' error median 3.67 cm, 95% level 13.65 cm
radius[constant] member 0: best validation loss 0.0917 at iteration 1000 (15s)
radius[constant] member 1: best validation loss 0.1025 at iteration 1500 (29s)
radius[constant] member 2: best validation loss 0.0915 at iteration 1500 (43s)
radius[constant]: scale 1.340; calibration paths' error median 3.42 cm, 95% level 12.32 cm
radius[direction] member 0: best validation loss 0.2150 at iteration 1000 (15s)
radius[direction] member 1: best validation loss 0.2257 at iteration 500 (30s)
radius[direction] member 2: best validation loss 0.2246 at iteration 1000 (44s)
radius[direction]: scale 1.777; calibration paths' error median 0.45 cm, 95% level 3.96 cm
saved outputs/bigym_20261124/selfmodel/model_early/via_point.pt
```

`selfmodel/model_early/eval.log`：

```
test, ordinary hold-out: 1000 paths
  reference  swept-region error: arm median  3.76 p90  9.26 cm | hand median  2.82 cm | margin (median radius)  4.84 cm, inside 0.96 | fixed radius 13.65 cm, inside 0.96
  constant   swept-region error: arm median  3.48 p90  8.48 cm | hand median  2.79 cm | margin (median radius)  4.68 cm, inside 0.95 | fixed radius 12.32 cm, inside 0.96
  direction  swept-region error: arm median  0.45 p90  1.69 cm | hand median  0.35 cm | margin (median radius)  0.83 cm, inside 0.95 | fixed radius  3.96 cm, inside 0.96
  drop of the arm's median against the reference: 0.88 (95 % interval 0.88 to 0.89, resampling routes); hand 0.87; constant: arm 0.08
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
test, sector hold-out: 0 paths, too few
test, ordinary hold-out, standing: 500 paths
  reference  swept-region error: arm median  3.86 p90 10.47 cm | hand median  2.90 cm | margin (median radius)  5.11 cm, inside 0.96 | fixed radius 13.65 cm, inside 0.93
  constant   swept-region error: arm median  3.89 p90  9.32 cm | hand median  3.14 cm | margin (median radius)  5.23 cm, inside 0.95 | fixed radius 12.32 cm, inside 0.94
  direction  swept-region error: arm median  0.46 p90  2.49 cm | hand median  0.36 cm | margin (median radius)  0.95 cm, inside 0.95 | fixed radius  3.96 cm, inside 0.94
  drop of the arm's median against the reference: 0.88 (95 % interval 0.87 to 0.89, resampling routes); hand 0.88; constant: arm -0.01
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 0.99 of the paths
test, ordinary hold-out, standing, palm path over 1.0 m: 1 paths, too few
real paths replayed, replay_early: 38 paths
  reference  swept-region error: arm median  2.59 p90  7.08 cm | hand median  1.95 cm | margin (median radius)  3.42 cm, inside 0.95 | fixed radius 13.65 cm, inside 1.00
  constant   swept-region error: arm median  2.23 p90  4.64 cm | hand median  2.01 cm | margin (median radius)  3.03 cm, inside 0.97 | fixed radius 12.32 cm, inside 1.00
  direction  swept-region error: arm median  0.41 p90  0.72 cm | hand median  0.33 cm | margin (median radius)  0.59 cm, inside 0.84 | fixed radius  3.96 cm, inside 1.00
  drop of the arm's median against the reference: 0.84 (95 % interval 0.83 to 0.89, resampling paths); hand 0.83; constant: arm 0.14
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
real paths replayed, replay_early, walked first: 38 paths
  reference  swept-region error: arm median  2.59 p90  7.08 cm | hand median  1.95 cm | margin (median radius)  3.42 cm, inside 0.95 | fixed radius 13.65 cm, inside 1.00
  constant   swept-region error: arm median  2.23 p90  4.64 cm | hand median  2.01 cm | margin (median radius)  3.03 cm, inside 0.97 | fixed radius 12.32 cm, inside 1.00
  direction  swept-region error: arm median  0.41 p90  0.72 cm | hand median  0.33 cm | margin (median radius)  0.59 cm, inside 0.84 | fixed radius  3.96 cm, inside 1.00
  drop of the arm's median against the reference: 0.84 (95 % interval 0.83 to 0.89, resampling paths); hand 0.83; constant: arm 0.14
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
```

`endpoint/loo_replay_early_M2.txt`：

```
replay_early, all: 38 episodes
  reference       swept-region error median  2.59 p90  7.08 cm | hand  1.95 cm | palm settled point median  2.25 cm
  loo_same_task   swept-region error median  0.92 p90  3.17 cm | hand  0.68 cm | palm settled point median  0.36 cm
  loo_all         swept-region error median  1.70 p90  3.87 cm | hand  1.28 cm | palm settled point median  0.40 cm
  constant_train  swept-region error median  2.23 p90  4.64 cm | hand  2.01 cm | palm settled point median  1.22 cm
  model           swept-region error median  0.41 p90  0.72 cm | hand  0.33 cm | palm settled point median  0.09 cm
  drops of the median (95 % interval, resampling episodes): loo_same_task vs reference 0.65 (0.54 to 0.72); model vs reference 0.84 (0.83 to 0.88); model vs loo_same_task 0.55 (0.46 to 0.67); model vs loo_all 0.76 (0.71 to 0.80); model vs constant_train 0.81 (0.79 to 0.84)
  drops of the end point median (95 % interval, resampling episodes): loo_same_task vs reference 0.84 (0.80 to 0.89); model vs reference 0.96 (0.95 to 0.97); model vs loo_same_task 0.76 (0.66 to 0.83); model vs loo_all 0.79 (0.74 to 0.85); model vs constant_train 0.93 (0.91 to 0.95)
  better on (swept-region error): loo_same_task than reference 35/38; model than reference 38/38; model than loo_same_task 37/38; model than loo_all 38/38; model than constant_train 38/38
  better on (end point): loo_same_task than reference 38/38; model than reference 38/38; model than loo_same_task 37/38; model than loo_all 38/38; model than constant_train 38/38
replay_early, reach: 18 episodes
  reference       swept-region error median  4.82 p90  7.96 cm | hand  3.87 cm | palm settled point median  2.45 cm
  loo_same_task   swept-region error median  1.97 p90  3.70 cm | hand  1.35 cm | palm settled point median  0.52 cm
  loo_all         swept-region error median  2.55 p90  4.47 cm | hand  1.69 cm | palm settled point median  0.52 cm
  constant_train  swept-region error median  3.30 p90  5.83 cm | hand  2.37 cm | palm settled point median  1.43 cm
  model           swept-region error median  0.48 p90  0.79 cm | hand  0.36 cm | palm settled point median  0.16 cm
  drops of the median (95 % interval, resampling episodes): loo_same_task vs reference 0.59 (0.29 to 0.75); model vs reference 0.90 (0.82 to 0.95); model vs loo_same_task 0.76 (0.55 to 0.88); model vs loo_all 0.81 (0.70 to 0.90); model vs constant_train 0.85 (0.73 to 0.93)
  drops of the end point median (95 % interval, resampling episodes): loo_same_task vs reference 0.79 (0.71 to 0.89); model vs reference 0.93 (0.90 to 0.97); model vs loo_same_task 0.68 (0.48 to 0.86); model vs loo_all 0.69 (0.49 to 0.84); model vs constant_train 0.89 (0.82 to 0.95)
  better on (swept-region error): loo_same_task than reference 15/18; model than reference 18/18; model than loo_same_task 18/18; model than loo_all 18/18; model than constant_train 18/18
  better on (end point): loo_same_task than reference 18/18; model than reference 18/18; model than loo_same_task 17/18; model than loo_all 18/18; model than constant_train 18/18
replay_early, drawer: 20 episodes
  reference       swept-region error median  2.57 p90  2.58 cm | hand  1.94 cm | palm settled point median  2.14 cm
  loo_same_task   swept-region error median  0.78 p90  1.04 cm | hand  0.54 cm | palm settled point median  0.28 cm
  loo_all         swept-region error median  1.50 p90  1.78 cm | hand  1.09 cm | palm settled point median  0.37 cm
  constant_train  swept-region error median  2.09 p90  2.33 cm | hand  1.92 cm | palm settled point median  1.05 cm
  model           swept-region error median  0.41 p90  0.48 cm | hand  0.33 cm | palm settled point median  0.08 cm
  drops of the median (95 % interval, resampling episodes): loo_same_task vs reference 0.70 (0.67 to 0.77); model vs reference 0.84 (0.83 to 0.86); model vs loo_same_task 0.47 (0.32 to 0.52); model vs loo_all 0.73 (0.67 to 0.76); model vs constant_train 0.80 (0.78 to 0.82)
  drops of the end point median (95 % interval, resampling episodes): loo_same_task vs reference 0.87 (0.82 to 0.91); model vs reference 0.96 (0.96 to 0.97); model vs loo_same_task 0.72 (0.55 to 0.82); model vs loo_all 0.79 (0.67 to 0.86); model vs constant_train 0.93 (0.91 to 0.94)
  better on (swept-region error): loo_same_task than reference 20/20; model than reference 20/20; model than loo_same_task 19/20; model than loo_all 20/20; model than constant_train 20/20
  better on (end point): loo_same_task than reference 20/20; model than reference 20/20; model than loo_same_task 20/20; model than loo_all 20/20; model than constant_train 20/20
```

`selfmodel/model_early/test_bucket9.txt`：

```
bucket 9, all: 1000 paths, 500 stances
  reference       swept-region error median  3.76 p90  9.26 cm | hand  2.82 cm | palm settled point median  2.66 p90  8.67 cm
  constant_train  swept-region error median  3.48 p90  8.48 cm | hand  2.79 cm | palm settled point median  2.54 p90  7.24 cm
  model           swept-region error median  0.45 p90  1.69 cm | hand  0.35 cm | palm settled point median  0.16 p90  1.06 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.88 (0.88 to 0.89); constant_train vs reference 0.08 (0.03 to 0.12); model vs constant_train 0.87 (0.87 to 0.88)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.94 (0.93 to 0.94); constant_train vs reference 0.05 (-0.00 to 0.09); model vs constant_train 0.94 (0.93 to 0.94)
  better on (swept-region error): model than reference 997/1000; constant_train than reference 618/1000; model than constant_train 997/1000
  better on (palm settled point): model than reference 998/1000; constant_train than reference 584/1000; model than constant_train 996/1000
bucket 9, reach: 500 paths, 250 stances
  reference       swept-region error median  3.86 p90 10.47 cm | hand  2.90 cm | palm settled point median  2.82 p90 10.62 cm
  constant_train  swept-region error median  3.89 p90  9.32 cm | hand  3.14 cm | palm settled point median  2.99 p90  9.13 cm
  model           swept-region error median  0.46 p90  2.49 cm | hand  0.36 cm | palm settled point median  0.22 p90  1.60 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.88 (0.87 to 0.89); constant_train vs reference -0.01 (-0.10 to 0.08); model vs constant_train 0.88 (0.87 to 0.89)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.92 (0.91 to 0.93); constant_train vs reference -0.06 (-0.12 to 0.01); model vs constant_train 0.93 (0.92 to 0.94)
  better on (swept-region error): model than reference 497/500; constant_train than reference 286/500; model than constant_train 497/500
  better on (palm settled point): model than reference 498/500; constant_train than reference 285/500; model than constant_train 496/500
bucket 9, drawer: 500 paths, 250 stances
  reference       swept-region error median  3.68 p90  7.89 cm | hand  2.74 cm | palm settled point median  2.49 p90  6.52 cm
  constant_train  swept-region error median  3.15 p90  6.68 cm | hand  2.51 cm | palm settled point median  2.07 p90  5.47 cm
  model           swept-region error median  0.43 p90  1.33 cm | hand  0.35 cm | palm settled point median  0.12 p90  0.73 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.88 (0.88 to 0.89); constant_train vs reference 0.14 (0.08 to 0.21); model vs constant_train 0.86 (0.85 to 0.87)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.95 (0.95 to 0.95); constant_train vs reference 0.17 (0.12 to 0.22); model vs constant_train 0.94 (0.93 to 0.95)
  better on (swept-region error): model than reference 500/500; constant_train than reference 332/500; model than constant_train 500/500
  better on (palm settled point): model than reference 500/500; constant_train than reference 299/500; model than constant_train 500/500
bucket 9, reach 1 of the stance: 500 paths, 500 stances
  reference       swept-region error median  3.80 p90  9.68 cm | hand  2.81 cm | palm settled point median  2.65 p90  9.42 cm
  constant_train  swept-region error median  3.61 p90  8.89 cm | hand  2.82 cm | palm settled point median  2.58 p90  7.91 cm
  model           swept-region error median  0.45 p90  1.98 cm | hand  0.36 cm | palm settled point median  0.16 p90  1.13 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.88 (0.88 to 0.89); constant_train vs reference 0.05 (-0.01 to 0.13); model vs constant_train 0.88 (0.87 to 0.88)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.94 (0.93 to 0.95); constant_train vs reference 0.03 (-0.04 to 0.10); model vs constant_train 0.94 (0.93 to 0.94)
  better on (swept-region error): model than reference 499/500; constant_train than reference 306/500; model than constant_train 498/500
  better on (palm settled point): model than reference 500/500; constant_train than reference 285/500; model than constant_train 499/500
bucket 9, reach 2 of the stance: 500 paths, 500 stances
  reference       swept-region error median  3.68 p90  8.98 cm | hand  2.83 cm | palm settled point median  2.66 p90  8.48 cm
  constant_train  swept-region error median  3.41 p90  7.84 cm | hand  2.76 cm | palm settled point median  2.47 p90  6.65 cm
  model           swept-region error median  0.45 p90  1.48 cm | hand  0.35 cm | palm settled point median  0.15 p90  1.05 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.88 (0.87 to 0.89); constant_train vs reference 0.07 (0.03 to 0.15); model vs constant_train 0.87 (0.86 to 0.88)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.94 (0.93 to 0.95); constant_train vs reference 0.07 (0.00 to 0.13); model vs constant_train 0.94 (0.93 to 0.94)
  better on (swept-region error): model than reference 498/500; constant_train than reference 312/500; model than constant_train 499/500
  better on (palm settled point): model than reference 498/500; constant_train than reference 299/500; model than constant_train 497/500
```

### 第 2 部分（开发种子 720020–720099，新局）

```
```

`sm_real2/build_replay.txt`：

```
outputs/bigym_20261124/sm_real2/replay: 151 paths ({'film': 0, 'short': 3, 'no_reach': 6} skipped), tasks [71, 80], buckets [0, 0, 0, 0, 0, 0, 0, 0, 0, 151]; reference-vs-actual: swept-region error median 2.58 cm (p90 7.03), palm 1.94 cm, palm settled point 2.27 cm (p90 2.90); our FK against the simulator's right_end_effector: max 0.000 mm
  reach: 71 paths, swept median 4.23 cm (p90 7.52), palm settled median 2.50 cm, duration median 0.60 s
  reach, reach 1: 71 paths, swept median 4.23 cm (p90 7.52), palm settled median 2.50 cm, duration median 0.60 s
  drawer: 80 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.06 cm, duration median 0.70 s
  drawer, reach 1: 80 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.06 cm, duration median 0.70 s
```

`sm_real2/build_replay_early.txt`：

```
outputs/bigym_20261124/sm_real2/replay_early: 151 paths ({'film': 0, 'short': 3, 'no_reach': 6} skipped), tasks [71, 80], buckets [0, 0, 0, 0, 0, 0, 0, 0, 0, 151]; reference-vs-actual: swept-region error median 2.58 cm (p90 7.03), palm 1.94 cm, palm settled point 2.26 cm (p90 2.66); our FK against the simulator's right_end_effector: max 0.000 mm
  reach: 71 paths, swept median 4.23 cm (p90 7.52), palm settled median 2.35 cm, duration median 0.60 s
  reach, reach 1: 71 paths, swept median 4.23 cm (p90 7.52), palm settled median 2.35 cm, duration median 0.60 s
  drawer: 80 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.06 cm, duration median 0.70 s
  drawer, reach 1: 80 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.06 cm, duration median 0.70 s
```

`sm_real2/range.txt`：

```
joints at the command outside the training paths' 0.5-99.5 % band (151 reaches, 7999 training paths): right arm (7) median 0.0, mean 0.01, max 2, counts [150, 0, 1]; all 29 median 0.0, max 6
```

`sm_real2/frozen_M_orig.txt`：

```
outputs/bigym_20261124/sm_real2/replay against outputs/bigym_20261124/endpoint/frozen_orig.npz (from /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/bigym_20261124/sm_real/replay, [18, 20] episodes; table md5 2bfd26ec94dcb9fad606ef0c9ee2b2ed), model outputs/bigym_20261124/selfmodel/model
replay, all: 151 episodes
  reference       swept-region error median  2.58 p90  7.03 cm | hand  1.94 cm | end point median  2.27 p90  2.90 cm
  frozen_constant swept-region error median  1.08 p90  3.06 cm | hand  0.80 cm | end point median  0.44 p90  0.86 cm
  loo_same_task   swept-region error median  1.06 p90  3.40 cm | hand  0.77 cm | end point median  0.43 p90  0.84 cm
  model           swept-region error median  0.31 p90  0.79 cm | hand  0.24 cm | end point median  0.51 p90  0.66 cm
  drops of the swept-region error median (95 % interval, resampling episodes): frozen_constant vs reference 0.58 (0.47 to 0.64); loo_same_task vs reference 0.59 (0.51 to 0.64); model vs reference 0.88 (0.85 to 0.90); model vs frozen_constant 0.72 (0.67 to 0.76); model vs loo_same_task 0.71 (0.66 to 0.75)
  drops of the end point median (95 % interval, resampling episodes): frozen_constant vs reference 0.81 (0.78 to 0.84); loo_same_task vs reference 0.81 (0.79 to 0.84); model vs reference 0.78 (0.75 to 0.82); model vs frozen_constant -0.15 (-0.44 to 0.15); model vs loo_same_task -0.16 (-0.48 to 0.10)
  better on (swept-region error): frozen_constant than reference 139/151; loo_same_task than reference 141/151; model than reference 151/151; model than frozen_constant 151/151; model than loo_same_task 151/151
  better on (end point): frozen_constant than reference 151/151; loo_same_task than reference 151/151; model than reference 151/151; model than frozen_constant 73/151; model than loo_same_task 70/151
replay, reach: 71 episodes
  reference       swept-region error median  4.23 p90  7.52 cm | hand  3.31 cm | end point median  2.50 p90  3.08 cm
  frozen_constant swept-region error median  2.41 p90  3.33 cm | hand  1.73 cm | end point median  0.64 p90  0.99 cm
  loo_same_task   swept-region error median  2.57 p90  3.66 cm | hand  1.88 cm | end point median  0.64 p90  1.00 cm
  model           swept-region error median  0.61 p90  0.90 cm | hand  0.53 cm | end point median  0.25 p90  0.41 cm
  drops of the swept-region error median (95 % interval, resampling episodes): frozen_constant vs reference 0.43 (0.14 to 0.61); loo_same_task vs reference 0.39 (0.09 to 0.55); model vs reference 0.85 (0.78 to 0.90); model vs frozen_constant 0.75 (0.69 to 0.77); model vs loo_same_task 0.76 (0.73 to 0.80)
  drops of the end point median (95 % interval, resampling episodes): frozen_constant vs reference 0.75 (0.70 to 0.78); loo_same_task vs reference 0.74 (0.72 to 0.78); model vs reference 0.90 (0.89 to 0.91); model vs frozen_constant 0.62 (0.54 to 0.67); model vs loo_same_task 0.62 (0.53 to 0.65)
  better on (swept-region error): frozen_constant than reference 59/71; loo_same_task than reference 61/71; model than reference 71/71; model than frozen_constant 71/71; model than loo_same_task 71/71
  better on (end point): frozen_constant than reference 71/71; loo_same_task than reference 71/71; model than reference 71/71; model than frozen_constant 66/71; model than loo_same_task 63/71
replay, drawer: 80 episodes
  reference       swept-region error median  2.57 p90  2.58 cm | hand  1.94 cm | end point median  2.06 p90  2.53 cm
  frozen_constant swept-region error median  0.77 p90  1.27 cm | hand  0.59 cm | end point median  0.30 p90  0.55 cm
  loo_same_task   swept-region error median  0.79 p90  1.25 cm | hand  0.58 cm | end point median  0.31 p90  0.55 cm
  model           swept-region error median  0.22 p90  0.34 cm | hand  0.18 cm | end point median  0.59 p90  0.69 cm
  drops of the swept-region error median (95 % interval, resampling episodes): frozen_constant vs reference 0.70 (0.67 to 0.72); loo_same_task vs reference 0.69 (0.67 to 0.72); model vs reference 0.91 (0.91 to 0.92); model vs frozen_constant 0.71 (0.68 to 0.74); model vs loo_same_task 0.71 (0.68 to 0.74)
  drops of the end point median (95 % interval, resampling episodes): frozen_constant vs reference 0.85 (0.82 to 0.87); loo_same_task vs reference 0.85 (0.82 to 0.86); model vs reference 0.71 (0.70 to 0.73); model vs frozen_constant -0.96 (-1.14 to -0.62); model vs loo_same_task -0.93 (-1.07 to -0.62)
  better on (swept-region error): frozen_constant than reference 80/80; loo_same_task than reference 80/80; model than reference 80/80; model than frozen_constant 80/80; model than loo_same_task 80/80
  better on (end point): frozen_constant than reference 80/80; loo_same_task than reference 80/80; model than reference 80/80; model than frozen_constant 7/80; model than loo_same_task 7/80
```

`sm_real2/frozen_M2_early.txt`：

```
outputs/bigym_20261124/sm_real2/replay_early against outputs/bigym_20261124/endpoint/frozen_early.npz (from /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/bigym_20261124/sm_real/replay_early, [18, 20] episodes; table md5 92c18e570155ce13cd7c02a50bf6839d), model outputs/bigym_20261124/selfmodel/model_early
replay_early, all: 151 episodes
  reference       swept-region error median  2.58 p90  7.03 cm | hand  1.94 cm | end point median  2.26 p90  2.66 cm
  frozen_constant swept-region error median  1.08 p90  3.05 cm | hand  0.75 cm | end point median  0.41 p90  0.76 cm
  loo_same_task   swept-region error median  1.06 p90  3.40 cm | hand  0.79 cm | end point median  0.38 p90  0.78 cm
  model           swept-region error median  0.42 p90  0.74 cm | hand  0.34 cm | end point median  0.08 p90  0.24 cm
  drops of the swept-region error median (95 % interval, resampling episodes): frozen_constant vs reference 0.58 (0.47 to 0.64); loo_same_task vs reference 0.59 (0.51 to 0.64); model vs reference 0.84 (0.83 to 0.84); model vs frozen_constant 0.61 (0.54 to 0.68); model vs loo_same_task 0.60 (0.54 to 0.66)
  drops of the end point median (95 % interval, resampling episodes): frozen_constant vs reference 0.82 (0.79 to 0.85); loo_same_task vs reference 0.83 (0.80 to 0.86); model vs reference 0.96 (0.96 to 0.97); model vs frozen_constant 0.80 (0.76 to 0.83); model vs loo_same_task 0.79 (0.75 to 0.83)
  better on (swept-region error): frozen_constant than reference 141/151; loo_same_task than reference 141/151; model than reference 151/151; model than frozen_constant 147/151; model than loo_same_task 149/151
  better on (end point): frozen_constant than reference 151/151; loo_same_task than reference 151/151; model than reference 151/151; model than frozen_constant 144/151; model than loo_same_task 145/151
replay_early, reach: 71 episodes
  reference       swept-region error median  4.23 p90  7.52 cm | hand  3.31 cm | end point median  2.35 p90  2.84 cm
  frozen_constant swept-region error median  2.40 p90  3.27 cm | hand  1.67 cm | end point median  0.52 p90  0.90 cm
  loo_same_task   swept-region error median  2.57 p90  3.66 cm | hand  1.81 cm | end point median  0.53 p90  0.84 cm
  model           swept-region error median  0.53 p90  0.87 cm | hand  0.44 cm | end point median  0.15 p90  0.32 cm
  drops of the swept-region error median (95 % interval, resampling episodes): frozen_constant vs reference 0.43 (0.15 to 0.61); loo_same_task vs reference 0.39 (0.11 to 0.55); model vs reference 0.87 (0.81 to 0.92); model vs frozen_constant 0.78 (0.73 to 0.81); model vs loo_same_task 0.79 (0.76 to 0.83)
  drops of the end point median (95 % interval, resampling episodes): frozen_constant vs reference 0.78 (0.72 to 0.80); loo_same_task vs reference 0.77 (0.75 to 0.82); model vs reference 0.94 (0.92 to 0.96); model vs frozen_constant 0.72 (0.63 to 0.81); model vs loo_same_task 0.73 (0.61 to 0.79)
  better on (swept-region error): frozen_constant than reference 61/71; loo_same_task than reference 61/71; model than reference 71/71; model than frozen_constant 71/71; model than loo_same_task 71/71
  better on (end point): frozen_constant than reference 71/71; loo_same_task than reference 71/71; model than reference 71/71; model than frozen_constant 66/71; model than loo_same_task 66/71
replay_early, drawer: 80 episodes
  reference       swept-region error median  2.57 p90  2.58 cm | hand  1.94 cm | end point median  2.06 p90  2.53 cm
  frozen_constant swept-region error median  0.77 p90  1.27 cm | hand  0.59 cm | end point median  0.30 p90  0.55 cm
  loo_same_task   swept-region error median  0.79 p90  1.25 cm | hand  0.58 cm | end point median  0.31 p90  0.55 cm
  model           swept-region error median  0.40 p90  0.47 cm | hand  0.32 cm | end point median  0.07 p90  0.10 cm
  drops of the swept-region error median (95 % interval, resampling episodes): frozen_constant vs reference 0.70 (0.67 to 0.72); loo_same_task vs reference 0.69 (0.67 to 0.72); model vs reference 0.84 (0.84 to 0.85); model vs frozen_constant 0.48 (0.43 to 0.53); model vs loo_same_task 0.49 (0.44 to 0.53)
  drops of the end point median (95 % interval, resampling episodes): frozen_constant vs reference 0.85 (0.82 to 0.87); loo_same_task vs reference 0.85 (0.82 to 0.86); model vs reference 0.97 (0.96 to 0.97); model vs frozen_constant 0.78 (0.74 to 0.83); model vs loo_same_task 0.78 (0.74 to 0.83)
  better on (swept-region error): frozen_constant than reference 80/80; loo_same_task than reference 80/80; model than reference 80/80; model than frozen_constant 76/80; model than loo_same_task 78/80
  better on (end point): frozen_constant than reference 80/80; loo_same_task than reference 80/80; model than reference 80/80; model than frozen_constant 78/80; model than loo_same_task 79/80
```

## 意外

- **换标签以后模型的扫过误差变差了**：38 局全部 M 0.26、M2 0.41 cm；新局全部 M 0.31、M2 0.42 cm，抽屉 0.22 对 0.40 cm。最后一段折线的终点挪近了，扫过误差就不再"白送"那一段。复核事先就说过两个量会对着走。
- **原标签的模型 M 在新局的抽屉终点上照样输给常数**（降幅 −0.96，80 局里只有 7 局更好）。和 38 局的结果一致，说明原来的输法是标签对不上，不是那 20 局碰巧。
- **抽屉终点 M2 0.07 cm**，比预测的下限 0.08 还好一点；38 局上对留一常数的降幅 0.72，略高于预测上限 0.7。
- **新局彼此很像**：冻结常数和新局内部的留一常数几乎一样（扫过 1.08 对 1.06、终点 0.41 对 0.38 cm），所以冻结在别的局上的常数并没有吃亏。
- **绝对的增益很小**：终点上抽屉从 0.30 到 0.07 cm，伸手任务从 0.52 到 0.15 cm，都是几毫米。伸手任务的容差是 5 cm。抽屉的关节反馈诊断（误差 0.29 cm）是 7 / 10，常数修正大概已经接近这个精度，模型多出来的这几毫米能不能变成多开几次抽屉，现在没有证据。

## 结论与下一步

- **对照判据**：
  - **第 1 部分过了**。抽屉终点 M2 对留一常数的降幅 0.72（0.55–0.82），20 / 20 局更好，不是"不到 0.1"。
  - **第 2 部分过了**。新局上两个任务、两个量，模型对冻结常数的降幅都在 0.1 以上，区间都不含 0：
    - 扫过误差（M）：伸手 0.75（0.69–0.77），抽屉 0.71（0.68–0.74）；
    - 终点（M2）：伸手 0.72（0.63–0.81），抽屉 0.78（0.74–0.83）。
  - 冻结常数在新局上只比原来的留一常数差 1.17 倍（扫过 1.08 对 0.92 cm），不到 1.5 倍。

  所以落在"两部分都过：写 S 在 BiGym 上的登记"。
- **能说的话**：在 BiGym 2.0 自带控制器、`--stance` 流程、两个任务的开发局上，自身模型预测"这一条命令发下去手停在哪"，比最好的不用模型的常数准 70–80 %。
  - 标签要对上流程：抽屉往里推之前手还没停稳，模型要预测那一刻的手。
  - 绝对误差：伸手任务从 0.52 到 0.15 cm，抽屉从 0.30 到 0.07 cm。
- **证据强度**：
  - 一次采样、两个模型（M、M2，各 5 个成员）；
  - 38 局旧开发局加 151 局新开发局，一个控制器；
  - 没跑官方种子。
- **其他可能的解释**：
  - 开发局彼此很像（瞄准点只差约 1 cm），模型的优势可能部分来自"在同一类局上学得细"。但常数也是在同样的局上取的，比较是公平的；换到差别更大的局上谁更好，没有测。
  - M2 是看过 38 局的结果以后定的标签（复核建议的），但标签的定义来自流程本身（往里推的时刻），没有按结果调参数，而且第 2 部分的 151 局是新局。
- **四类陷阱**：
  - **benchmark 选择**：两个任务 11-24 定的，没有换。
  - **数据泄漏**：训练是 800000 起的种子；常数冻结在 720000–720019 上，测在 720020–720099 上；冻结文件的 md5 在新局开跑前记下，用前核对过。
  - **指标误用**：两个量、两种标签都报了，包括对模型不利的（M 的终点、M2 的扫过误差）。
  - **事后挑选**：每种标签只训练一个模型，没有挑 checkpoint。
- **下一步**：
  1. 独立复核（这条结论会决定写 S 在 BiGym 上的登记）。
  2. **S 在 BiGym 上的登记**：
     - 做法：A、J、冻结常数修正、S（用 M2 的终点预测挪目标）、上限（按本局实测偏差挪）。
     - 开发种子上定细节，官方种子 620000–620099 只跑一次。
     - 主指标：抽屉的成功局数，伸手任务的终点偏差（成败到顶）。
     - 预测要写明：常数修正可能已经把抽屉的成功率拉到接近上限，S 对常数在成败上可能分不开。

## 独立复核

2026-11-25，独立的 agent。它拿到的是登记的各节、跑后补记、原始结果和结论（"意外"一节和推理过程没给），以及 4090 上的数据、模型、代码路径。原文（英文）：

> ## Verdict summary
>
> The numbers check out exactly, and the bookkeeping is clean: nothing leaked and nothing was timed after the fact. The headline claim is worded more strongly than the evidence allows. The self model does beat a per-task constant by 70–80 % at the end point. A per-task constant is not the best model-free opponent, though. Cheap regressions fitted on the same 38 old episodes cut the margin to about 40–60 %. The few-mm gain is also unlikely to change drawer success.
>
> ## Main findings
>
> 1. **Every reported number reproduces, and every integrity check passes.** I recomputed them with my own end-point and swept-error code. Mismatches found: none.
>    - **Frozen constants:** they equal the per-task means of the 38 old episodes (seeds 720000–720019) to the bit. Both md5s match. They were written at 18:42:55 UTC; the new runs' code copy was made at 18:45:44 and the first new film at 18:45:47.
>    - **Registration:** commit 940915a8 was made at 18:36:46 UTC, before any output (the first is at 18:40:28). Its design, prediction and criteria sections match the write-up word for word.
>    - **New runs vs Part 1:** the new episodes were launched before M2's Part 1 result existed (18:50:30), so Part 2 could not have been started because Part 1 looked good.
>    - **No leakage into the model:** M2 was trained only on `data_early` (seeds 800000–804999). The union of its members' training draws covers buckets 0–7 only, and bucket 8 was used for early stopping and calibration. No real-episode seed overlaps the training data, and the old and new seeds don't overlap each other.
>    - **Same label in training and real episodes:** every window ends exactly 25 steps (0.5 s) after the reach's last step. This held for 189/189 real episodes and 300/300 sampled training paths. On all 100 real drawer episodes, the 25 settle steps are followed directly by `in`.
>    - **Label value:** my own forward kinematics on the raw films reproduces the label to within 1e-4 mm. The 5 ticks start 1–5 steps after the reach and end 0–4 steps before the window end, by the same rule on both sides.
>
> 2. **A per-task constant is a weak opponent.** I fitted cheap model-free predictors on the 38 old episodes only. I picked the best one per task by leave-one-out on those 38 alone, then scored it on the 151 new episodes (early label):
>
>    | | Constant | Best cheap predictor | M2 | Gain of M2 over cheap predictor (95 % CI) |
>    |---|---|---|---|---|
>    | Reach | 5.23 mm | 3 nearest neighbours in goal space: 2.44 mm | 1.46 mm | 0.40 (0.13–0.56), better on 54/71 |
>    | Drawer | 3.03 mm | ridge on goal + hand + joints + duration: 1.59 mm | 0.67 mm | 0.58 (0.47–0.66), better on 70/80 |
>
>    - **Drawer, goal only:** plain least squares on the goal position (3 features, 20 episodes) gets 1.18 mm. M2's gain over it is 0.43 (0.28–0.59).
>    - **With about 80 real episodes:** fitted leave-one-out within the new drawers, a 14-feature ridge reaches 0.85 mm. M2's margin shrinks to 0.20 (0.02–0.34), better on only 54/80.
>    - **Swept error:** a ridge on the goal position, fitted on the 18 old reach episodes, gets 0.74 cm (constant 2.41, M 0.61). M's gain over it is only 0.17 (0.02–0.30), better on 54/71. On the drawer the same ridge gets 0.47 cm, M 0.23, gain 0.53.
>    - **What the claim can say:** the self model beats the best cheap regression by about 40–60 % (reach swept error: about 17 %) while using no real episodes. Its edge over model-free predictors is mostly data efficiency.
>
> 3. **The label is not "the hand at the moment the push starts".** The hand is still moving inside the label window. On real drawers, the palm at reach + 0.5 s sits 3.4 mm (median; p90 4.3 mm, mostly sideways) from the label's 5-tick mean. That is larger than the constant's whole error.
>    - **Raw M2 against the push-start instant:** 3.44 mm, worse than a constant fitted to that instant (2.82 mm).
>    - **M2 plus a fixed shift** taken from the old episodes: 0.58 mm.
>    - The S registration has to aim at the instant, either by retraining on it or by applying that shift.
>
> 4. **The few-mm gain is unlikely to matter for task success.**
>    - **Reach:** J already succeeds on 80/80 with no correction.
>    - **Drawer:** under J, success is decided on a roughly 2 cm scale. Episodes with settled error ≤ 1.99 cm succeed 32/33; above 1.98 cm, 1/67 (AUC 0.999).
>    - **Confound:** how far ahead the goal sits after the walk separates success from failure perfectly (AUC 1.0). It correlates 0.93 with the settled error, so it is unclear whether the error itself causes the failures.
>    - **After correction:** both corrections leave residuals far below that 2 cm scale (constant p90 5.5 mm, M2 p90 1.0 mm). Constant correction and S probably can't be told apart on drawer success.
>    - **Grasp error:** the hand drifts about 1 cm further during the push (error at grasp 2.9 cm on successes, 3.5 cm on failures). The label doesn't cover that drift.
>
> 5. **Part 2 makes the label choice legitimate, but only partly.** It is a proper fresh test: 151 new episodes from the same pipeline, constants frozen before the runs, and the drawer's measured end point unchanged by the relabel. It also confirms the mismatch explanation out of sample: the old-label M loses on the new drawers (0.59 vs 0.30 cm). It does not fix three things:
>    - The reach task's evaluation target was also changed, in the model's favour. Under the old label M beats the constant by 0.62; under the new one M2 beats it by 0.72. The "70–80 %" range relies on the new label.
>    - Only constants were tried as model-free opponents.
>    - The distribution is narrow: goals spread about 1–2 cm in x and y, and the drawer's height spreads 1.6 mm. Old and new episodes come from the same distribution (mean offsets differ by ≤ 1.2 mm), so the 1.17× degradation of the frozen constant was to be expected.
>
> ## Recomputed key numbers (all match the write-up)
>
> - **Part 1, end point (cm):**
>   - Drawer: reference 2.143, leave-one-out constant 0.280, M2 0.077; gain 0.72 (0.55–0.83), better on 20/20.
>   - Reach: reference 2.448, leave-one-out 0.518, M2 0.164; gain 0.68 (0.48–0.86), better on 17/18.
> - **Part 2, end point, new label (cm):**
>   - Reach: frozen constant 0.523, M2 0.146; gain 0.72 (0.63–0.80), better on 66/71.
>   - Drawer: frozen 0.303, M2 0.067; gain 0.78 (0.74–0.82), better on 78/80.
> - **Part 2, swept error, old label (cm):**
>   - Reach: frozen 2.414, M 0.614; gain 0.75.
>   - Drawer: frozen 0.774, M 0.225; gain 0.71.
> - **Frozen constant on new episodes vs leave-one-out on old (swept):** 1.077 / 0.918 = 1.17×.
> - **Registered predictions:** all hit, except drawer M2 (0.067 against a predicted 0.08–0.3) and the Part 1 drawer gain (0.72 against 0.2–0.7). Both missed by being better than predicted, which doesn't matter.
>
> ## Experiments that could overturn the conclusion
>
> 1. **Register a stronger model-free opponent and a learning curve.** Use goal-conditioned ridge or nearest neighbours fitted on N = 10, 20, 40, 80 and 150 real episodes. If the regression matches M2 at N ≈ 80–150 (my drawer number suggests it nearly does), the claim becomes "saves about N real episodes", not "beats model-free".
> 2. **Closed-loop dose-response on the drawer.** Run four arms: oracle correction, oracle plus injected offsets of 0, ±2, ±5, ±10 and ±20 mm along each axis, constant correction, and S. If success is flat for offsets up to 5 mm, or constant equals oracle, the mm gain cannot matter for drawer success. Score the realized error at the push-start instant and at the grasp, not the 5-tick mean.
> 3. **Realized versus predicted gain.** Shifting the goal changes the command, and with it the offset. S should evaluate the model at the shifted command. If the realized end error with S is close to that of constant correction, the one-shot-accuracy story fails.
> 4. **A wider distribution.** Jitter the stance (the stand-off distances, squat height), reach duration and goal position beyond the dev spread. Constants and linear fits may extrapolate badly, or the simulator-trained model may degrade. Also retrain M2 with 2–3 more seeds; the current gains come from a single training run.
>
> ## Verdict per conclusion
>
> - **"Part 1 passed"** (drawer gain 0.72, 20/20): **supports.**
> - **"Part 2 passed"** (all four gains above 0.1 with intervals excluding 0): **supports, as registered.** Against a goal regression the reach swept gain is 0.17 (0.02–0.30), so the 0.1 bar would be marginal.
> - **"Frozen constant only 1.17× worse"**: **supports.**
> - **"Beats the best model-free constant by 70–80 %"**: **does not support as worded.** It is true against a per-task constant. Against the best cheap model-free predictor it is about 40–58 % at the end point and about 17 % for reach swept error.
> - **"The model must predict the hand at the moment the push starts"**: **does not support as worded.** M2 predicts a 5-tick mean that sits about 3.4 mm from that instant. This is fixable.
> - **Absolute errors** (reach 0.52 → 0.15 cm, drawer 0.30 → 0.07 cm): **supports.**
> - **"The comparison is fair because the constant came from the same episodes"**: **insufficient evidence.** The data are the same, but the opponent is weak.
> - **Data leakage pitfall:** **supports** (verified clean).
> - **Post-hoc selection pitfall:** **partly supports.** The label choice is cured by Part 2. The changed reach target and the narrow opponent set are not.
> - **That the few-mm gain matters for drawer success:** **insufficient evidence, and the existing data argue against it.** This backs the author's own warning that S and constant correction may be indistinguishable on drawer success.
>
> All analysis ran read-only on the 4090: CPU only, temporary files in a unique folder under `/tmp`, deleted afterwards on both machines. The code reviewed is in `/Users/zed/code/loco/scripts/experiments/bigym_20261124/` (`bigym_data.py`, `bigym_bench.py`, `bigym_collect.py`, `bigym_test.py`) and `/Users/zed/code/loco/scripts/experiments/selfmodel_via_20261120/` (`via_point.py`, `via_model.py`). Their md5s match the copies saved with each run on the 4090.

### 复核后的更正（2026-11-25）

数全部重算对上；冻结常数、登记、新局的先后顺序都核对过；没有泄漏。读法照下面改：

1. **撤回"比最好的不用模型的常数准 70–80 %"，改成两句。**
   - **对按任务的常数**：准 70–80 %，成立。
   - **对便宜的不用模型的回归**：这种回归只用旧的 38 局拟合，按目标位置找最近的 3 局，或对目标、手、关节、时长做岭回归。在 151 局新局上，终点 M2 比它准 0.40（伸手，0.13–0.56）和 0.58（抽屉，0.47–0.66）；伸手的扫过误差只准 0.17（0.02–0.30）。
   - **真实局多了以后**：抽屉上用约 80 局真实局拟合的回归只差 M2 0.20（0.02–0.34）。
   - **所以模型对不用模型的办法，优势主要是不用真实局**：模型一局真实局都没用，回归要几十局才追得上。论文里要写的是这个数据效率，并且要登记一条学习曲线来量它（见第 5 条）。
2. **撤回"标签对上了往里推开始那一刻"。** 新标签是伸手后 0.5 s 里 5 个时刻的平均，手在这段时间还在动。
   - 抽屉上，往里推开始那一刻的手离这个平均约 3.4 mm（中位数，90 % 分位 4.3 mm，主要在侧向），比常数的整个误差还大。
   - 直接拿 M2 对那一刻比，3.44 mm，比对那一刻拟合的常数（2.82 mm）还差；M2 加一个从旧局取的固定偏移是 0.58 mm。
   - S 要瞄准那一刻：要么用那一刻重新训练，要么加这个偏移。
3. **几毫米的增益在 BiGym 上大概变不成成败。**
   - 伸手任务 J 不修正已经 80 / 80。
   - 抽屉在 J 下成败按约 2 cm 分开：停稳误差 1.99 cm 以下 33 局成功 32 局，以上 67 局成功 1 局。
   - 但"走完路以后目标离多远"把成败分得一样干净（AUC 1.0），和误差相关 0.93，所以误差是不是原因说不清。
   - 常数修正和 S 剩下的误差都远小于 2 cm，两者在抽屉成败上大概分不开。
   - 往里推的过程中手还会再漂约 1 cm，标签不管这一段。
4. **事后挑选只部分解决。** 第 2 部分的新局让"换标签"站得住；但伸手任务的评测目标也跟着换了，换法对模型有利（原标签 0.62，新标签 0.72）。"70–80 %"靠的是新标签，两种标签的数都要报。
5. **下一步改成：**
   1. **学习曲线（离线，不用新仿真，先登记）。** 不用模型的回归（最近邻、岭回归），分别用 N = 10、20、40、80、150 局真实局拟合，和不用真实局的 M2 比终点。结论写成"模型省下多少局真实局"。
   2. **抽屉的剂量反应（要仿真，先登记）**，回答几毫米对成败有没有用：
      - 上限修正，再故意加 0、±2、±5、±10、±20 mm 的偏移；外加常数修正和 S。
      - 量往里推开始那一刻和抓住那一刻的误差。
      - 如果 5 mm 以内成败不变，或者常数修正等于上限，就说明 BiGym 抽屉上毫米级的精度不影响成败，S 的价值只能用精度本身和数据效率来写。
   3. S 在 BiGym 上的登记合进第 2 步：A、J、常数修正、S 都在里面；S 瞄准往里推开始那一刻；挪过目标以后用挪过的命令再算一次预测。
   4. 更宽的分布（站位、下蹲高度、目标都抖得更大），看常数和回归在外推时是不是先坏；另用 2–3 个种子重训 M2。
