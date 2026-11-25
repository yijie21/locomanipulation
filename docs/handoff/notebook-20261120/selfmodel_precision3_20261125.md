# selfmodel_precision3_20261125：电机自己框架下的参考运动学加一个常数，在抖动局上和模型的完整预测比

- 日期：2026-11-25（设计、预测、判据；任何 kin_rr、kinc_rr 运行以前写下，单独提交）
- 机器 / GPU：4090，1–3 号卡，和 precision2 一样。跑之前查 nvidia-smi，0 号卡上有别人的进程，不碰。
- 代码版本：`selfwm` 加未提交的 `scripts/experiments/simple_eval_20261112/precision.py` 和启动脚本，跑时抄进输出目录，md5 记在结果里。
- 配置：同 [selfmodel_precision2_20261125](selfmodel_precision2_20261125.md)。SIMPLE 的 TabletopGrasp，感知栏，RGB-D，ScaleBFM 和 SONIC 不改参数，站位抖动 ±6 cm，同样的局和重复。
- 关联：
  - precision2 的独立复核：
    - **离线打平**：按电机自己的参考根放出来的终点（er）挪，再加一个开发局冻结的常数（kinc_rr），离线外推和 modelf 打平。ScaleBFM +0.09 [−0.06, +0.26]，8 / 18 个布局；常数只用 3 个开发局估也打平。SONIC −0.61 [−1.47, −0.11]，10 / 19。
    - **模型学到的东西**：p 和"参考根终点 − 搬到骨盆上的终点"（u）的相关 0.79 / 0.85 / 0.89。扣掉 u 以后，对剩下跟踪误差的相关只有 0.29 / 0.36 / 0.76。
  - [selfmodel_sfull_20261125](selfmodel_sfull_20261125.md)：官方局上同样加这两组（预测修订）。

## 假设

在 SIMPLE 的抖动局上，模型的完整预测比固定常数准，靠的主要是"按电机实际跟的参考去瞄"。这件事不用模型就能算。

要检验的是：**扣掉这一项以后，自身模型对跟踪误差的逐局预测还剩多少用？** 复核的离线外推说几乎没有。

## 做法

### 局

沿用 precision2 的开发局、测试局、4 次重复。J、const、modelf 的行直接用，不重跑。新两组从同样的状态出发，逐局配对。

### 两种新挪法（只挪预抓取的目标，不用模型、不用仿真真值）

- **er**：J 的参考在终点的夹爪中心，用电机自己的参考根（`ref0[0:7]`）放，不搬到真实骨盆上，减开局那一眼的目标。坐标用参考根的朝向。
- **c_rr**：开发局（和 const 同一批 3、4 级 JR 局，ScaleBFM 11 局、SONIC 15 局，不抖动）上"实测夹爪中心 − er"的平均。
  - 开跑前算好，存 `outputs/simple_dev_20261124/precision_const/ctrack_rr_<控制器>.json`，记 md5，之后不改。
  - 开发局用留一位置的版本，**按位置名排除**本局的位置。precision2 复核发现按目标距离排除，在 SONIC 上漏排。
- **kin_rr**：挪 −er。
- **kinc_rr**：挪 −(er + c_rr)。

规划、缩放重试（全量、2/3、1/3，都不行退回 J）、执行都和 precision2 一样。

### 顺序

1. 开发局，查流程判据。
2. 过了就接着跑测试局，不等人。

## 度量

和 precision2 一样：
- 预抓取停偏，每组减 J，按局配对；
- 按布局（1 cm）重抽 4000 次的 95 % 区间和符号检验；
- 见过 / 没见过的布局分开报；
- entry 停偏、主指标、缩放和退回次数。

核心比较：**modelf − kinc_rr**。另报 kinc_rr − const、kin_rr − J。

## 预测（跑之前写；看过复核的离线外推，没看过 kin_rr、kinc_rr 的任何执行结果）

**ScaleBFM（测试局）**：
- kin_rr 停偏中位数 1.3–2.6 cm。J 自己的手离 er 中位数 1.88 cm。
- kinc_rr 0.7–1.5 cm（离线 1.10）。
- modelf − kinc_rr：−0.3 到 +0.5，中间猜 +0.1。
- modelf 赢的可能约一成；kinc_rr 反过来赢（区间整个在 0 以上，并且 p < 0.05）的可能约两成半。
- kinc_rr − const：−0.9 到 −0.2。
- 开发局 modelf − kinc_rr：−0.5 到 +0.5。

**SONIC（测试局）**：
- kin_rr 3–6 cm，kinc_rr 3–6 cm。
- modelf − kinc_rr：−1.5 到 +1.0，中间猜 −0.4；modelf 赢的可能约一成半。

**两个控制器**：
- 每组退回 J 不超过 15 %。
- 有路线的局 45 秒内的成功，各组和 J 差不超过 10 %。

## 判据（什么结果会让我改做法，下一步做什么）

"赢"的定义同 precision2：按布局重抽的 95 % 区间整个在 0 以下，并且符号检验 p < 0.05。每个控制器分开，以测试局为准。

- **modelf 赢 kinc_rr**：
  - 读法：自身模型对跟踪误差的逐局预测，在参考运动学加常数之外还有用，SIMPLE 上模型算贡献。
  - 下一步：另做一个不用仿真真值的 SF（现在的 SF 经 `s_check.frames` 用了真实骨盆）。
- **打平，或 kinc_rr 赢**：
  - 读法：SIMPLE 上比固定标定更准的那部分，来自"按电机实际跟的参考去瞄，再加一个常数"，不来自学出来的模型。SIMPLE 上自身模型不算贡献。
  - 我向用户建议：主表加"参考运动学加常数"一行；论文里一次命令精度的证据，要放到这个对手会失效的设定上，即伸手几何真的会变的局。那要另登记。

**流程判据（开发局，决定接着跑测试局）**：
- 出现以下任一情况，就停下先查，不跑测试局：
  - 有进程失败；
  - 某组退回 J 超过 30 %；
  - 新两组任一组的站位抖动、发令状态和同局 J 不同。
- 测试局的预测和判据就是本节这些，开发局的结果不改它们。

## 预测修订

2026-11-25，跑完以后补记实现（预测和判据没改）：

- **读行时串组（正式运行以前发现并修掉）**：
  - `precision.py` 按 `kinc_*.jsonl` 这样的文件名读某一组的行，会把 `kinc_rr_*.jsonl` 也读进来，`kin_*` 同理，而且 rr 的行会盖掉原组的行。
  - 修法：文件名去掉组名前缀以后必须以大写字母开头（任务名都大写开头）。两个脚本里只用来记日志的行数统计也一起改了。
  - 修在任何正式运行之前。两个试跑的表是改前的代码写的，已用修好的代码重新生成。
  - `bench_via.sh` 记在 `pipeline.log` 里的行数仍会把两组算在一起（例如 "arm kinc: 56 rows"）。只是日志，没改，以保持它的 md5。
- **c_rr 的坐标**：用参考根的朝向。复核给的数是在 W 的朝向下算的，换过去完全一致：ScaleBFM [−1.619, −0.651, +0.797]，sd [0.969, 0.661, 0.549]。
- **4090 的 git HEAD**：是 `bc426d5`，所以运行日志写 "commit bc426d5 + uncommitted"。代码从 Mac 抄过去，以各输出目录的 asrun 副本为准。
- **代码 md5**（Mac 和 4090 一致）：
  - `precision.py` `9fa88cebd64afc75807740ca6f75bfac`
  - `precision_run.sh` `a888a66c68e0c978e70690807620124b`
  - `precision3_launch.sh` `1a8088744fc61658e881857bfa823de6`
  - `sfull_run.sh` `54b1baf3f4ab79b4ee9c6b017a09830c`
  - `sfull_launch.sh` `08ba150dce509036802f3cc76da300b3`
  - `ctrack_rr_scalebfm.json` `05a1e8f9ce30fbfac28d0e30214c90a0`
  - `ctrack_rr_sonic.json` `805875a6f1feb8622fda56f24241eb3e`
  - 没改的：`s_check.py` `4106eba5…`、`jref_data.py` `8e7e1130…`、`bench_via.sh` `82cd6dc2…`、`simple_bench.py` `80511140…`、`plan2.py` `3161d616…`。
  - const、ctrack 文件同上次。
- **试跑用的是修串组之前的版本**：`precision.py` `49b13629…`、`sfull_run.sh` `70fd3b69…`、`precision_run.sh` `b26a313f…`。正式运行都用上面的最终版本。
- **c_rr 文件的时间**：写于 21:34:55Z / 21:34:57Z。登记提交 `dcb83c3b` 在 21:23:30Z，第二个试跑 21:36Z 开始。
- **启动脚本**：新文件 `precision3_launch.sh`，标记 `started3` / `done3`，进度 `run3.log`，副本 `asrun3/`，表 `table3.txt`；旧那几次的文件都没动。旧的 `table`、`table2` 跑完以后仍和原文件逐字节一致。
- **按位置名留一**：ScaleBFM 开发局的 L3_e0、L3_e7、L3_e9 目标相距不到 0.9 cm，表里算成一个布局，但按名字只排掉本局那一个位置。只影响开发局。

## 结果（贴原始输出，不转述）

```
dev  started3 2026-11-24T22:07:03Z  done3 2026-11-24T22:18:25Z
test started3 2026-11-24T22:19:47Z  done3 2026-11-24T22:43:51Z
```

c_rr 文件（`outputs/simple_dev_20261124/precision_const/ctrack_rr_<控制器>.json`）和核对：

```
scalebfm {'arm': 'JR', 'n': 11, 'frame': "the reference root's yaw", 'mean_crr_h_cm': [-1.617, -0.668, 0.797], 'sd_crr_h_cm': [0.972, 0.644, 0.549], 'mean_er_h_cm': [-1.553, 0.73, 2.564], 'mean_er_plus_crr_h_cm': [-3.17, 0.062, 3.361], 'mean_miss_rr_h_cm': [-3.17, 0.062, 3.363], 'abs_yaw_rr_minus_W_deg': {'median': 0.323, 'max': 1.717}, 'jaw_to_er_cm_median': 1.932, 'jaw_to_carried_end_cm_median': 4.425}
  names equal const: True
sonic {'arm': 'JR', 'n': 15, 'frame': "the reference root's yaw", 'mean_crr_h_cm': [-3.559, 2.234, 1.515], 'sd_crr_h_cm': [4.76, 2.785, 1.87], 'mean_er_h_cm': [-0.39, -1.314, 0.831], 'mean_er_plus_crr_h_cm': [-3.949, 0.921, 2.347], 'mean_miss_rr_h_cm': [-3.947, 0.923, 2.347], 'abs_yaw_rr_minus_W_deg': {'median': 4.008, 'max': 13.453}, 'jaw_to_er_cm_median': 3.387, 'jaw_to_carried_end_cm_median': 7.911}
  names equal const: True
c_rr in W heading frame: scalebfm mean [-1.619 -0.651  0.797] sd [0.969 0.661 0.549]; sonic mean [-3.385 2.541 1.515] sd [4.654 2.912 1.87]
```

precision2 的测试局上，手离 er 和离搬到骨盆上的终点：

```
precision_20261124/scalebfm: 112 episode-repeats; |jaw - er| median 1.88 cm, |jaw - carried end| median 4.65 cm; |yaw rr - W| median 0.28, 90 % 1.02, max 1.79 deg
precision_20261124/sonic: 106 episode-repeats; |jaw - er| median 4.31 cm, |jaw - carried end| median 11.13 cm; |yaw rr - W| median 5.31, 90 % 8.31, max 14.01 deg
```

开发局 `outputs/simple_dev_20261124/precision_20261124/scalebfm/table3.txt`：

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/precision_20261124/scalebfm  repeats 4, episodes 80, routed episode-repeats 44, layouts 9, falls in routed 0
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 17 / 17  J 40 / 30  replan 39 / 30  const 40 / 30  model 40 / 31  sham 40 / 29  oracle 40 / 31  kin 38 / 30  kinc 40 / 32  modelf 40 / 29  kin_rr 40 / 32  kinc_rr 40 / 32

routes with every arm's row: 44; fallback to J's plan: replan 0 (0 %), const 2 (5 %), model 1 (2 %), sham 0 (0 %), oracle 1 (2 %), kin 1 (2 %), kinc 4 (9 %), modelf 1 (2 %), kin_rr 1 (2 %), kinc_rr 1 (2 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 44 2/3 0 1/3 0 fallback 0; const 1 28 2/3 4 1/3 10 fallback 2; model 1 31 2/3 3 1/3 9 fallback 1; sham 1 36 2/3 3 1/3 5 fallback 0; oracle 1 36 2/3 4 1/3 3 fallback 1; kin 1 43 2/3 0 1/3 0 fallback 1; kinc 1 27 2/3 6 1/3 7 fallback 4; modelf 1 33 2/3 7 1/3 3 fallback 1; kin_rr 1 43 2/3 0 1/3 0 fallback 1; kinc_rr 1 32 2/3 6 1/3 5 fallback 1
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0, kin_rr 0, kinc_rr 0
reference root (kin_rr, kinc_rr): |yaw of the root - W's| median 0.31, max 1.49 deg; |er| median 2.35 cm (|g| 1.91); J's jaw to the root's end median 1.85 cm, to the carried end 4.50 cm; c_rr episodes 10-10

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   3.70  mean   4.55  missing 0
  replan  median   3.72  mean   4.88  missing 0 | minus J: median +0.00 [-0.01, +0.01], closer in 21/44; layouts closer 2/9 (sign p 0.18)
  const   median   1.98  mean   2.24  missing 0 | minus J: median -2.41 [-3.77, -1.18], closer in 39/44; layouts closer 9/9 (sign p 0.00391)
  model   median   2.52  mean   2.43  missing 0 | minus J: median -1.24 [-3.96, -0.77], closer in 41/44; layouts closer 8/9 (sign p 0.0391)
  sham    median   6.47  mean   6.79  missing 0 | minus J: median +2.51 [+1.66, +2.67], closer in 11/44; layouts closer 0/9 (sign p 0.00391)
  oracle  median   0.71  mean   1.08  missing 0 | minus J: median -3.20 [-5.59, -1.42], closer in 43/44; layouts closer 9/9 (sign p 0.00391)
  kin     median   4.63  mean   5.76  missing 0 | minus J: median +0.88 [+0.58, +1.30], closer in 7/44; layouts closer 1/9 (sign p 0.0391)
  kinc    median   1.98  mean   2.27  missing 0 | minus J: median -2.38 [-3.34, -1.07], closer in 38/44; layouts closer 9/9 (sign p 0.00391)
  modelf  median   1.44  mean   1.50  missing 0 | minus J: median -2.46 [-4.72, -1.21], closer in 42/44; layouts closer 9/9 (sign p 0.00391)
  kin_rr  median   1.83  mean   2.00  missing 0 | minus J: median -2.01 [-4.63, -0.50], closer in 42/44; layouts closer 9/9 (sign p 0.00391)
  kinc_rr median   1.58  mean   1.54  missing 0 | minus J: median -2.64 [-4.75, -1.23], closer in 42/44; layouts closer 9/9 (sign p 0.00391)
  modelf minus kinc_rr: median -0.04 [-0.51, +0.12] cm, modelf closer in 24/44 episodes, layouts 5/9 (worse 4, sign p 1)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus kinc_rr median -0.04 [-0.51, +0.12], layouts closer 5/9 (sign p 1)
  kinc_rr minus const: median -0.31 [-1.33, +0.03] cm, kinc_rr closer in 27/44 episodes, layouts 7/9 (worse 2, sign p 0.18)
    seen by the model's data: none
    never seen: 44 episodes, kinc_rr minus const median -0.31 [-1.33, +0.03], layouts closer 7/9 (sign p 0.18)
  kin_rr minus J: median -2.01 [-4.63, -0.50] cm, kin_rr closer in 42/44 episodes, layouts 9/9 (worse 0, sign p 0.00391)
    seen by the model's data: none
    never seen: 44 episodes, kin_rr minus J median -2.01 [-4.63, -0.50], layouts closer 9/9 (sign p 0.00391)
  modelf minus const: median -0.38 [-1.62, +0.00] cm, modelf closer in 30/44 episodes, layouts 7/9 (worse 2, sign p 0.18)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus const median -0.38 [-1.62, +0.00], layouts closer 7/9 (sign p 0.18)
  modelf minus kinc: median -0.59 [-1.64, -0.21] cm, modelf closer in 32/44 episodes, layouts 9/9 (worse 0, sign p 0.00391)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus kinc median -0.59 [-1.64, -0.21], layouts closer 9/9 (sign p 0.00391)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   2.03  mean   2.36  missing 0
  replan  median   2.05  mean   2.86  missing 0 | minus J: median +0.00 [+0.00, +0.02], closer in 12/44; layouts closer 1/9 (sign p 0.0391)
  const   median   2.61  mean   2.65  missing 0 | minus J: median +0.16 [+0.00, +0.35], closer in 11/44; layouts closer 1/9 (sign p 0.0391)
  model   median   2.51  mean   2.59  missing 0 | minus J: median +0.15 [+0.00, +0.25], closer in 13/44; layouts closer 1/9 (sign p 0.0391)
  sham    median   2.02  mean   2.38  missing 0 | minus J: median -0.07 [-0.13, +0.05], closer in 23/44; layouts closer 7/9 (sign p 0.18)
  oracle  median   2.48  mean   2.67  missing 0 | minus J: median +0.21 [+0.04, +0.41], closer in 9/44; layouts closer 2/9 (sign p 0.18)
  kin     median   2.13  mean   3.43  missing 0 | minus J: median +0.06 [+0.04, +0.09], closer in 11/44; layouts closer 3/9 (sign p 0.508)
  kinc    median   2.65  mean   2.70  missing 0 | minus J: median +0.22 [+0.00, +0.35], closer in 9/44; layouts closer 1/9 (sign p 0.0391)
  modelf  median   2.46  mean   2.59  missing 0 | minus J: median +0.14 [+0.00, +0.27], closer in 12/44; layouts closer 1/9 (sign p 0.0391)
  kin_rr  median   2.19  mean   2.49  missing 0 | minus J: median +0.03 [-0.03, +0.13], closer in 17/44; layouts closer 2/9 (sign p 0.18)
  kinc_rr median   2.47  mean   2.66  missing 0 | minus J: median +0.25 [+0.00, +0.41], closer in 12/44; layouts closer 1/9 (sign p 0.0391)
  modelf minus kinc_rr: median -0.04 [-0.05, -0.02] cm, modelf closer in 30/44 episodes, layouts 8/9 (worse 1, sign p 0.0391)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus kinc_rr median -0.04 [-0.05, -0.02], layouts closer 8/9 (sign p 0.0391)
  kinc_rr minus const: median +0.00 [-0.07, +0.07] cm, kinc_rr closer in 20/44 episodes, layouts 4/9 (worse 5, sign p 1)
    seen by the model's data: none
    never seen: 44 episodes, kinc_rr minus const median +0.00 [-0.07, +0.07], layouts closer 4/9 (sign p 1)
  kin_rr minus J: median +0.03 [-0.03, +0.13] cm, kin_rr closer in 17/44 episodes, layouts 2/9 (worse 7, sign p 0.18)
    seen by the model's data: none
    never seen: 44 episodes, kin_rr minus J median +0.03 [-0.03, +0.13], layouts closer 2/9 (sign p 0.18)
  modelf minus const: median -0.02 [-0.06, +0.04] cm, modelf closer in 25/44 episodes, layouts 4/9 (worse 5, sign p 1)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus const median -0.02 [-0.06, +0.04], layouts closer 4/9 (sign p 1)
  modelf minus kinc: median -0.03 [-0.09, +0.04] cm, modelf closer in 24/44 episodes, layouts 6/9 (worse 3, sign p 0.508)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus kinc median -0.03 [-0.09, +0.04], layouts closer 6/9 (sign p 0.508)

routes: successes within 45 s / official limit
  J 40 / 30  replan 39 / 30  const 40 / 30  model 40 / 31  sham 40 / 29  oracle 40 / 31  kin 38 / 30  kinc 40 / 32  modelf 40 / 29  kin_rr 40 / 32  kinc_rr 40 / 32
```

开发局 `.../sonic/table3.txt`：

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/precision_20261124/sonic  repeats 4, episodes 80, routed episode-repeats 55, layouts 15, falls in routed 11
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 13 / 13  J 36 / 26  replan 36 / 28  const 33 / 21  model 34 / 21  sham 34 / 25  oracle 36 / 26  kin 35 / 25  kinc 38 / 26  modelf 34 / 26  kin_rr 32 / 24  kinc_rr 37 / 23

routes with every arm's row: 55; fallback to J's plan: replan 0 (0 %), const 7 (13 %), model 4 (7 %), sham 3 (5 %), oracle 8 (15 %), kin 8 (15 %), kinc 8 (15 %), modelf 6 (11 %), kin_rr 0 (0 %), kinc_rr 7 (13 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 55 2/3 0 1/3 0 fallback 0; const 1 36 2/3 4 1/3 8 fallback 7; model 1 36 2/3 9 1/3 6 fallback 4; sham 1 44 2/3 6 1/3 2 fallback 3; oracle 1 39 2/3 3 1/3 5 fallback 8; kin 1 41 2/3 6 1/3 0 fallback 8; kinc 1 36 2/3 8 1/3 3 fallback 8; modelf 1 42 2/3 6 1/3 1 fallback 6; kin_rr 1 53 2/3 2 1/3 0 fallback 0; kinc_rr 1 37 2/3 8 1/3 3 fallback 7
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0, kin_rr 0, kinc_rr 0
reference root (kin_rr, kinc_rr): |yaw of the root - W's| median 4.06, max 12.37 deg; |er| median 2.82 cm (|g| 7.86); J's jaw to the root's end median 3.37 cm, to the carried end 9.71 cm; c_rr episodes 14-15

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   4.79  mean  11.77  missing 0
  replan  median   5.00  mean  11.15  missing 0 | minus J: median -0.03 [-0.19, +0.11], closer in 28/55; layouts closer 9/15 (sign p 0.607)
  const   median   4.03  mean  10.84  missing 0 | minus J: median -0.75 [-1.38, +0.00], closer in 33/55; layouts closer 9/15 (sign p 0.267)
  model   median   7.32  mean  12.62  missing 0 | minus J: median +1.02 [+0.00, +2.44], closer in 19/55; layouts closer 3/15 (sign p 0.0574)
  sham    median   8.28  mean  17.49  missing 0 | minus J: median +2.90 [+2.14, +4.59], closer in 11/55; layouts closer 0/15 (sign p 6.1e-05)
  oracle  median   2.36  mean  10.87  missing 0 | minus J: median -1.58 [-2.27, -0.51], closer in 40/55; layouts closer 11/15 (sign p 0.0574)
  kin     median   8.88  mean  19.17  missing 0 | minus J: median +2.48 [+1.46, +5.62], closer in 4/55; layouts closer 0/15 (sign p 0.000122)
  kinc    median   7.33  mean  13.95  missing 0 | minus J: median +1.25 [+0.56, +2.17], closer in 10/55; layouts closer 1/15 (sign p 0.00183)
  modelf  median   3.09  mean  13.14  missing 0 | minus J: median -0.33 [-1.31, +0.00], closer in 32/55; layouts closer 7/15 (sign p 1)
  kin_rr  median   3.86  mean  11.38  missing 0 | minus J: median -0.51 [-1.62, +0.32], closer in 32/55; layouts closer 8/15 (sign p 1)
  kinc_rr median   3.53  mean  10.54  missing 0 | minus J: median -0.55 [-1.45, +0.00], closer in 32/55; layouts closer 10/15 (sign p 0.0923)
  modelf minus kinc_rr: median +0.00 [-0.23, +0.20] cm, modelf closer in 26/55 episodes, layouts 6/15 (worse 8, sign p 0.791)
    seen by the model's data: none
    never seen: 55 episodes, modelf minus kinc_rr median +0.00 [-0.23, +0.20], layouts closer 6/15 (sign p 0.791)
  kinc_rr minus const: median -0.05 [-0.86, +0.00] cm, kinc_rr closer in 28/55 episodes, layouts 7/15 (worse 6, sign p 1)
    seen by the model's data: none
    never seen: 55 episodes, kinc_rr minus const median -0.05 [-0.86, +0.00], layouts closer 7/15 (sign p 1)
  kin_rr minus J: median -0.51 [-1.62, +0.32] cm, kin_rr closer in 32/55 episodes, layouts 8/15 (worse 7, sign p 1)
    seen by the model's data: none
    never seen: 55 episodes, kin_rr minus J median -0.51 [-1.62, +0.32], layouts closer 8/15 (sign p 1)
  modelf minus const: median +0.00 [-0.44, +0.32] cm, modelf closer in 26/55 episodes, layouts 5/15 (worse 9, sign p 0.424)
    seen by the model's data: none
    never seen: 55 episodes, modelf minus const median +0.00 [-0.44, +0.32], layouts closer 5/15 (sign p 0.424)
  modelf minus kinc: median -2.05 [-3.74, -0.72] cm, modelf closer in 38/55 episodes, layouts 12/15 (worse 2, sign p 0.0129)
    seen by the model's data: none
    never seen: 55 episodes, modelf minus kinc median -2.05 [-3.74, -0.72], layouts closer 12/15 (sign p 0.0129)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   2.55  mean   5.21  missing 1
  replan  median   2.74  mean   5.55  missing 1 | minus J: median +0.13 [-0.08, +0.51], closer in 22/54; layouts closer 5/15 (sign p 0.302)
  const   median   2.96  mean   6.28  missing 1 | minus J: median +0.27 [+0.00, +0.47], closer in 18/54; layouts closer 2/15 (sign p 0.0225)
  model   median   3.07  mean   5.46  missing 1 | minus J: median +0.04 [-0.15, +0.58], closer in 24/54; layouts closer 3/15 (sign p 0.0574)
  sham    median   2.89  mean   8.07  missing 1 | minus J: median +0.21 [+0.11, +0.54], closer in 16/54; layouts closer 4/15 (sign p 0.118)
  oracle  median   3.01  mean   7.37  missing 1 | minus J: median +0.25 [+0.00, +0.77], closer in 16/54; layouts closer 5/15 (sign p 0.424)
  kin     median   2.79  mean   9.98  missing 1 | minus J: median +0.13 [+0.00, +0.62], closer in 19/54; layouts closer 3/15 (sign p 0.0574)
  kinc    median   2.82  mean   6.58  missing 1 | minus J: median +0.00 [-0.06, +0.32], closer in 20/54; layouts closer 6/15 (sign p 0.791)
  modelf  median   2.50  mean   9.09  missing 1 | minus J: median +0.11 [-0.01, +0.21], closer in 20/54; layouts closer 2/15 (sign p 0.0129)
  kin_rr  median   3.06  mean  11.37  missing 0 | minus J: median +0.20 [-0.12, +0.56], closer in 22/54; layouts closer 5/15 (sign p 0.302)
  kinc_rr median   2.72  mean   6.34  missing 1 | minus J: median +0.15 [+0.00, +0.39], closer in 19/54; layouts closer 3/15 (sign p 0.0923)
  modelf minus kinc_rr: median -0.00 [-0.13, +0.02] cm, modelf closer in 27/54 episodes, layouts 7/15 (worse 7, sign p 1)
    seen by the model's data: none
    never seen: 54 episodes, modelf minus kinc_rr median -0.00 [-0.13, +0.02], layouts closer 7/15 (sign p 1)
  kinc_rr minus const: median -0.03 [-0.16, +0.00] cm, kinc_rr closer in 29/54 episodes, layouts 7/15 (worse 6, sign p 1)
    seen by the model's data: none
    never seen: 54 episodes, kinc_rr minus const median -0.03 [-0.16, +0.00], layouts closer 7/15 (sign p 1)
  kin_rr minus J: median +0.20 [-0.12, +0.56] cm, kin_rr closer in 22/54 episodes, layouts 5/15 (worse 10, sign p 0.302)
    seen by the model's data: none
    never seen: 54 episodes, kin_rr minus J median +0.20 [-0.12, +0.56], layouts closer 5/15 (sign p 0.302)
  modelf minus const: median +0.00 [-0.21, +0.21] cm, modelf closer in 26/54 episodes, layouts 6/15 (worse 8, sign p 0.791)
    seen by the model's data: none
    never seen: 54 episodes, modelf minus const median +0.00 [-0.21, +0.21], layouts closer 6/15 (sign p 0.791)
  modelf minus kinc: median +0.00 [-0.01, +0.26] cm, modelf closer in 23/54 episodes, layouts 5/15 (worse 9, sign p 0.424)
    seen by the model's data: none
    never seen: 54 episodes, modelf minus kinc median +0.00 [-0.01, +0.26], layouts closer 5/15 (sign p 0.424)

routes: successes within 45 s / official limit
  J 35 / 25  replan 35 / 27  const 32 / 20  model 33 / 20  sham 33 / 24  oracle 35 / 25  kin 34 / 24  kinc 37 / 25  modelf 33 / 25  kin_rr 31 / 23  kinc_rr 36 / 22
```

开发局的流程判据都没触发：
- 没有失败的进程；
- 退回 J 最多 13 %（SONIC kinc_rr）；
- 新两组的站位和发令状态和 J 不同的局都是 0。

所以按登记接着跑了测试局。

测试局 `outputs/simple_eval_20261112/precision_20261124/scalebfm/table3.txt`：

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/precision_20261124/scalebfm  repeats 4, episodes 120, routed episode-repeats 112, layouts 18, falls in routed 0
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 64 / 64  J 119 / 101  replan 119 / 106  const 112 / 98  model 115 / 98  sham 116 / 91  oracle 116 / 94  kin 120 / 95  kinc 110 / 95  modelf 113 / 99  kin_rr 118 / 106  kinc_rr 116 / 100

routes with every arm's row: 112; fallback to J's plan: replan 0 (0 %), const 1 (1 %), model 0 (0 %), sham 1 (1 %), oracle 1 (1 %), kin 3 (3 %), kinc 1 (1 %), modelf 1 (1 %), kin_rr 1 (1 %), kinc_rr 1 (1 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 112 2/3 0 1/3 0 fallback 0; const 1 83 2/3 15 1/3 13 fallback 1; model 1 86 2/3 15 1/3 11 fallback 0; sham 1 100 2/3 8 1/3 3 fallback 1; oracle 1 99 2/3 7 1/3 5 fallback 1; kin 1 103 2/3 0 1/3 6 fallback 3; kinc 1 75 2/3 20 1/3 16 fallback 1; modelf 1 90 2/3 15 1/3 6 fallback 1; kin_rr 1 105 2/3 6 1/3 0 fallback 1; kinc_rr 1 91 2/3 10 1/3 10 fallback 1
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0, kin_rr 0, kinc_rr 0
reference root (kin_rr, kinc_rr): |yaw of the root - W's| median 0.28, max 1.79 deg; |er| median 2.76 cm (|g| 1.96); J's jaw to the root's end median 1.88 cm, to the carried end 4.65 cm; c_rr episodes 11-11

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   3.97  mean   4.70  missing 0
  replan  median   4.02  mean   4.70  missing 0 | minus J: median +0.00 [-0.00, +0.00], closer in 51/112; layouts closer 8/18 (sign p 0.815)
  const   median   1.74  mean   3.08  missing 0 | minus J: median -2.63 [-3.84, -1.50], closer in 104/112; layouts closer 16/18 (sign p 0.00131)
  model   median   2.47  mean   2.67  missing 0 | minus J: median -1.55 [-3.16, -0.50], closer in 92/112; layouts closer 15/18 (sign p 0.00754)
  sham    median   6.06  mean   6.30  missing 0 | minus J: median +1.74 [+0.31, +2.70], closer in 40/112; layouts closer 1/18 (sign p 0.000145)
  oracle  median   0.61  mean   1.19  missing 0 | minus J: median -3.41 [-4.22, -2.64], closer in 110/112; layouts closer 17/18 (sign p 0.000145)
  kin     median   4.30  mean   4.95  missing 0 | minus J: median +0.37 [+0.05, +0.96], closer in 35/112; layouts closer 3/18 (sign p 0.00754)
  kinc    median   1.94  mean   3.53  missing 0 | minus J: median -2.30 [-3.10, -1.49], closer in 104/112; layouts closer 16/18 (sign p 0.00131)
  modelf  median   1.28  mean   2.35  missing 0 | minus J: median -2.74 [-4.21, -1.65], closer in 107/112; layouts closer 17/18 (sign p 0.000145)
  kin_rr  median   1.84  mean   2.00  missing 0 | minus J: median -2.13 [-3.31, -1.21], closer in 111/112; layouts closer 18/18 (sign p 7.63e-06)
  kinc_rr median   1.26  mean   1.53  missing 0 | minus J: median -2.81 [-3.88, -2.03], closer in 110/112; layouts closer 17/18 (sign p 0.000145)
  modelf minus kinc_rr: median -0.04 [-0.22, +0.40] cm, modelf closer in 59/112 episodes, layouts 10/18 (worse 8, sign p 0.815)
    seen by the model's data: 92 episodes, modelf minus kinc_rr median -0.03 [-0.22, +0.43], layouts closer 9/15 (sign p 0.607)
    never seen: 20 episodes, modelf minus kinc_rr median -0.16 [-0.53, +0.47], layouts closer 1/3 (sign p 1)
  kinc_rr minus const: median -0.23 [-0.70, -0.06] cm, kinc_rr closer in 79/112 episodes, layouts 15/18 (worse 3, sign p 0.00754)
    seen by the model's data: 92 episodes, kinc_rr minus const median -0.27 [-0.91, -0.06], layouts closer 12/15 (sign p 0.0352)
    never seen: 20 episodes, kinc_rr minus const median -0.08 [-0.29, +0.04], layouts closer 3/3 (sign p 0.25)
  kin_rr minus J: median -2.13 [-3.31, -1.21] cm, kin_rr closer in 111/112 episodes, layouts 18/18 (worse 0, sign p 7.63e-06)
    seen by the model's data: 92 episodes, kin_rr minus J median -2.13 [-3.53, -0.97], layouts closer 15/15 (sign p 6.1e-05)
    never seen: 20 episodes, kin_rr minus J median -2.25 [-2.63, -1.82], layouts closer 3/3 (sign p 0.25)
  modelf minus const: median -0.41 [-0.71, -0.12] cm, modelf closer in 80/112 episodes, layouts 15/18 (worse 3, sign p 0.00754)
    seen by the model's data: 92 episodes, modelf minus const median -0.44 [-0.80, -0.08], layouts closer 12/15 (sign p 0.0352)
    never seen: 20 episodes, modelf minus const median -0.30 [-0.49, +0.36], layouts closer 3/3 (sign p 0.25)
  modelf minus kinc: median -0.56 [-0.96, -0.24] cm, modelf closer in 81/112 episodes, layouts 15/18 (worse 3, sign p 0.00754)
    seen by the model's data: 92 episodes, modelf minus kinc median -0.58 [-1.03, -0.12], layouts closer 12/15 (sign p 0.0352)
    never seen: 20 episodes, modelf minus kinc median -0.52 [-0.80, -0.05], layouts closer 3/3 (sign p 0.25)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   2.08  mean   2.28  missing 0
  replan  median   2.10  mean   2.29  missing 0 | minus J: median -0.00 [-0.01, +0.01], closer in 56/112; layouts closer 8/18 (sign p 0.815)
  const   median   2.61  mean   3.81  missing 0 | minus J: median +0.30 [+0.19, +0.45], closer in 14/112; layouts closer 1/18 (sign p 0.000145)
  model   median   2.66  mean   2.85  missing 0 | minus J: median +0.22 [+0.10, +0.36], closer in 29/112; layouts closer 4/18 (sign p 0.0309)
  sham    median   2.28  mean   2.64  missing 0 | minus J: median +0.04 [-0.04, +0.11], closer in 50/112; layouts closer 6/18 (sign p 0.238)
  oracle  median   2.73  mean   3.02  missing 0 | minus J: median +0.28 [+0.18, +0.47], closer in 16/112; layouts closer 3/18 (sign p 0.00754)
  kin     median   2.15  mean   2.31  missing 0 | minus J: median +0.04 [-0.00, +0.06], closer in 39/112; layouts closer 6/18 (sign p 0.238)
  kinc    median   2.64  mean   4.25  missing 0 | minus J: median +0.39 [+0.22, +0.52], closer in 12/112; layouts closer 1/18 (sign p 0.000145)
  modelf  median   2.70  mean   3.56  missing 0 | minus J: median +0.28 [+0.16, +0.45], closer in 20/112; layouts closer 5/18 (sign p 0.0963)
  kin_rr  median   2.33  mean   2.49  missing 0 | minus J: median +0.08 [+0.02, +0.25], closer in 34/112; layouts closer 3/18 (sign p 0.00754)
  kinc_rr median   2.71  mean   2.97  missing 0 | minus J: median +0.29 [+0.16, +0.50], closer in 14/112; layouts closer 2/18 (sign p 0.00131)
  modelf minus kinc_rr: median -0.00 [-0.03, +0.01] cm, modelf closer in 58/112 episodes, layouts 12/18 (worse 6, sign p 0.238)
    seen by the model's data: 92 episodes, modelf minus kinc_rr median -0.00 [-0.05, +0.02], layouts closer 10/15 (sign p 0.302)
    never seen: 20 episodes, modelf minus kinc_rr median -0.02 [-0.03, +0.01], layouts closer 2/3 (sign p 1)
  kinc_rr minus const: median +0.00 [-0.03, +0.02] cm, kinc_rr closer in 55/112 episodes, layouts 8/18 (worse 10, sign p 0.815)
    seen by the model's data: 92 episodes, kinc_rr minus const median +0.00 [-0.04, +0.07], layouts closer 7/15 (sign p 1)
    never seen: 20 episodes, kinc_rr minus const median -0.01 [-0.05, +0.01], layouts closer 1/3 (sign p 1)
  kin_rr minus J: median +0.08 [+0.02, +0.25] cm, kin_rr closer in 34/112 episodes, layouts 3/18 (worse 15, sign p 0.00754)
    seen by the model's data: 92 episodes, kin_rr minus J median +0.18 [+0.04, +0.28], layouts closer 2/15 (sign p 0.00739)
    never seen: 20 episodes, kin_rr minus J median -0.02 [-0.28, +0.04], layouts closer 1/3 (sign p 1)
  modelf minus const: median -0.00 [-0.04, +0.05] cm, modelf closer in 57/112 episodes, layouts 10/18 (worse 8, sign p 0.815)
    seen by the model's data: 92 episodes, modelf minus const median -0.00 [-0.05, +0.07], layouts closer 8/15 (sign p 1)
    never seen: 20 episodes, modelf minus const median -0.03 [-0.07, +0.01], layouts closer 2/3 (sign p 1)
  modelf minus kinc: median -0.04 [-0.12, +0.03] cm, modelf closer in 65/112 episodes, layouts 11/18 (worse 7, sign p 0.481)
    seen by the model's data: 92 episodes, modelf minus kinc median -0.04 [-0.14, +0.05], layouts closer 9/15 (sign p 0.607)
    never seen: 20 episodes, modelf minus kinc median -0.03 [-0.16, +0.04], layouts closer 2/3 (sign p 1)

routes: successes within 45 s / official limit
  J 111 / 93  replan 111 / 98  const 104 / 90  model 107 / 90  sham 108 / 83  oracle 108 / 86  kin 112 / 87  kinc 102 / 87  modelf 105 / 91  kin_rr 110 / 98  kinc_rr 108 / 92
```

测试局 `.../sonic/table3.txt`：

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/precision_20261124/sonic  repeats 4, episodes 120, routed episode-repeats 106, layouts 19, falls in routed 8
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 39 / 39  J 72 / 60  replan 69 / 57  const 72 / 59  model 69 / 53  sham 66 / 51  oracle 70 / 50  kin 73 / 52  kinc 74 / 49  modelf 67 / 46  kin_rr 73 / 59  kinc_rr 69 / 56

routes with every arm's row: 106; fallback to J's plan: replan 0 (0 %), const 3 (3 %), model 2 (2 %), sham 3 (3 %), oracle 8 (8 %), kin 0 (0 %), kinc 5 (5 %), modelf 4 (4 %), kin_rr 4 (4 %), kinc_rr 4 (4 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 106 2/3 0 1/3 0 fallback 0; const 1 93 2/3 3 1/3 7 fallback 3; model 1 86 2/3 15 1/3 3 fallback 2; sham 1 79 2/3 12 1/3 12 fallback 3; oracle 1 73 2/3 12 1/3 13 fallback 8; kin 1 80 2/3 12 1/3 14 fallback 0; kinc 1 82 2/3 10 1/3 9 fallback 5; modelf 1 92 2/3 4 1/3 6 fallback 4; kin_rr 1 96 2/3 1 1/3 5 fallback 4; kinc_rr 1 86 2/3 10 1/3 6 fallback 4
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0, kin_rr 0, kinc_rr 0
reference root (kin_rr, kinc_rr): |yaw of the root - W's| median 5.31, max 14.01 deg; |er| median 3.19 cm (|g| 9.45); J's jaw to the root's end median 4.31 cm, to the carried end 11.14 cm; c_rr episodes 15-15

pregrasp settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   5.92  mean   8.57  missing 0
  replan  median   5.76  mean   9.87  missing 0 | minus J: median +0.11 [-0.05, +0.30], closer in 46/106; layouts closer 8/19 (sign p 0.648)
  const   median   4.63  mean   7.73  missing 0 | minus J: median -1.45 [-2.19, -0.56], closer in 72/106; layouts closer 13/19 (sign p 0.167)
  model   median   9.26  mean  10.79  missing 0 | minus J: median +2.41 [+0.69, +3.71], closer in 34/106; layouts closer 5/19 (sign p 0.0636)
  sham    median   9.12  mean  12.15  missing 0 | minus J: median +2.84 [+1.80, +3.43], closer in 27/106; layouts closer 1/19 (sign p 7.63e-05)
  oracle  median   3.00  mean   6.35  missing 0 | minus J: median -2.73 [-3.47, -1.54], closer in 78/106; layouts closer 15/19 (sign p 0.0192)
  kin     median   9.71  mean  15.27  missing 0 | minus J: median +3.00 [+2.15, +4.72], closer in 25/106; layouts closer 0/19 (sign p 3.81e-06)
  kinc    median   6.82  mean   9.47  missing 0 | minus J: median +0.10 [-0.26, +1.22], closer in 48/106; layouts closer 8/19 (sign p 0.648)
  modelf  median   3.87  mean   6.74  missing 0 | minus J: median -1.99 [-2.64, -1.65], closer in 78/106; layouts closer 14/19 (sign p 0.0636)
  kin_rr  median   4.04  mean   8.00  missing 0 | minus J: median -0.89 [-1.55, -0.05], closer in 67/106; layouts closer 11/19 (sign p 0.648)
  kinc_rr median   3.96  mean   7.61  missing 0 | minus J: median -1.76 [-2.40, -0.65], closer in 72/106; layouts closer 14/19 (sign p 0.0636)
  modelf minus kinc_rr: median -0.37 [-1.16, +0.36] cm, modelf closer in 58/106 episodes, layouts 14/19 (worse 5, sign p 0.0636)
    seen by the model's data: 94 episodes, modelf minus kinc_rr median -0.37 [-1.24, +0.39], layouts closer 12/16 (sign p 0.0768)
    never seen: 12 episodes, modelf minus kinc_rr median -0.33 [-2.07, +2.86], layouts closer 2/3 (sign p 1)
  kinc_rr minus const: median -0.40 [-0.65, +0.00] cm, kinc_rr closer in 63/106 episodes, layouts 11/19 (worse 8, sign p 0.648)
    seen by the model's data: 94 episodes, kinc_rr minus const median -0.28 [-0.63, +0.06], layouts closer 9/16 (sign p 0.804)
    never seen: 12 episodes, kinc_rr minus const median -1.04 [-1.47, +0.70], layouts closer 2/3 (sign p 1)
  kin_rr minus J: median -0.89 [-1.55, -0.05] cm, kin_rr closer in 67/106 episodes, layouts 11/19 (worse 8, sign p 0.648)
    seen by the model's data: 94 episodes, kin_rr minus J median -0.98 [-2.15, -0.08], layouts closer 10/16 (sign p 0.454)
    never seen: 12 episodes, kin_rr minus J median -0.42 [-1.18, +0.21], layouts closer 1/3 (sign p 1)
  modelf minus const: median -0.40 [-1.10, -0.08] cm, modelf closer in 63/106 episodes, layouts 12/19 (worse 7, sign p 0.359)
    seen by the model's data: 94 episodes, modelf minus const median -0.40 [-1.10, -0.15], layouts closer 11/16 (sign p 0.21)
    never seen: 12 episodes, modelf minus const median -0.38 [-2.28, +1.26], layouts closer 1/3 (sign p 1)
  modelf minus kinc: median -2.33 [-2.86, -1.67] cm, modelf closer in 79/106 episodes, layouts 16/19 (worse 3, sign p 0.00443)
    seen by the model's data: 94 episodes, modelf minus kinc median -2.25 [-3.01, -1.53], layouts closer 13/16 (sign p 0.0213)
    never seen: 12 episodes, modelf minus kinc median -2.66 [-4.67, -1.76], layouts closer 3/3 (sign p 0.25)

entry settled miss, cm (jaw centre against the look's goal; medians skip missing rows, pairs with a missing side are dropped)
  J       median   3.33  mean   6.82  missing 0
  replan  median   3.41  mean   7.98  missing 0 | minus J: median +0.09 [-0.12, +0.37], closer in 49/106; layouts closer 8/19 (sign p 0.648)
  const   median   3.07  mean   6.82  missing 0 | minus J: median -0.26 [-0.43, -0.13], closer in 62/106; layouts closer 14/19 (sign p 0.0636)
  model   median   3.43  mean   5.61  missing 0 | minus J: median +0.07 [-0.23, +0.40], closer in 50/106; layouts closer 10/19 (sign p 1)
  sham    median   3.63  mean   5.79  missing 0 | minus J: median +0.00 [-0.28, +0.43], closer in 52/106; layouts closer 7/19 (sign p 0.359)
  oracle  median   3.18  mean   7.12  missing 0 | minus J: median -0.04 [-0.23, +0.04], closer in 53/106; layouts closer 10/19 (sign p 1)
  kin     median   3.11  mean  10.76  missing 0 | minus J: median -0.16 [-0.51, +0.14], closer in 59/106; layouts closer 9/19 (sign p 1)
  kinc    median   3.05  mean   6.59  missing 0 | minus J: median +0.00 [-0.24, +0.25], closer in 51/106; layouts closer 13/19 (sign p 0.167)
  modelf  median   3.05  mean   5.57  missing 0 | minus J: median -0.09 [-0.34, +0.10], closer in 56/106; layouts closer 10/19 (sign p 1)
  kin_rr  median   2.94  mean   7.11  missing 0 | minus J: median -0.00 [-0.13, +0.04], closer in 54/106; layouts closer 9/19 (sign p 1)
  kinc_rr median   3.41  mean   6.72  missing 0 | minus J: median -0.15 [-0.45, +0.10], closer in 59/106; layouts closer 12/19 (sign p 0.359)
  modelf minus kinc_rr: median +0.04 [-0.07, +0.20] cm, modelf closer in 51/106 episodes, layouts 9/19 (worse 10, sign p 1)
    seen by the model's data: 94 episodes, modelf minus kinc_rr median +0.05 [-0.05, +0.22], layouts closer 6/16 (sign p 0.454)
    never seen: 12 episodes, modelf minus kinc_rr median -0.17 [-0.80, -0.06], layouts closer 3/3 (sign p 0.25)
  kinc_rr minus const: median +0.04 [-0.04, +0.16] cm, kinc_rr closer in 49/106 episodes, layouts 7/19 (worse 12, sign p 0.359)
    seen by the model's data: 94 episodes, kinc_rr minus const median +0.08 [-0.00, +0.20], layouts closer 6/16 (sign p 0.454)
    never seen: 12 episodes, kinc_rr minus const median -0.14 [-0.53, -0.02], layouts closer 1/3 (sign p 1)
  kin_rr minus J: median -0.00 [-0.13, +0.04] cm, kin_rr closer in 54/106 episodes, layouts 9/19 (worse 10, sign p 1)
    seen by the model's data: 94 episodes, kin_rr minus J median -0.03 [-0.14, +0.02], layouts closer 9/16 (sign p 0.804)
    never seen: 12 episodes, kin_rr minus J median +0.22 [-0.01, +0.66], layouts closer 0/3 (sign p 0.25)
  modelf minus const: median +0.12 [-0.13, +0.34] cm, modelf closer in 49/106 episodes, layouts 10/19 (worse 9, sign p 1)
    seen by the model's data: 94 episodes, modelf minus const median +0.18 [-0.10, +0.39], layouts closer 7/16 (sign p 0.804)
    never seen: 12 episodes, modelf minus const median -0.43 [-0.66, -0.11], layouts closer 3/3 (sign p 0.25)
  modelf minus kinc: median +0.00 [-0.21, +0.22] cm, modelf closer in 52/106 episodes, layouts 11/19 (worse 8, sign p 0.648)
    seen by the model's data: 94 episodes, modelf minus kinc median +0.04 [-0.13, +0.30], layouts closer 8/16 (sign p 1)
    never seen: 12 episodes, modelf minus kinc median -0.38 [-0.91, -0.22], layouts closer 3/3 (sign p 0.25)

routes: successes within 45 s / official limit
  J 68 / 56  replan 65 / 53  const 68 / 55  model 65 / 49  sham 62 / 47  oracle 66 / 46  kin 69 / 48  kinc 70 / 45  modelf 63 / 42  kin_rr 69 / 55  kinc_rr 65 / 52
```

- 测试局退回 J 最多 4 %，状态都相同，没有失败的进程。
- 独立核对同 S-full：er、c_rr、挪的量和规划输入都对上。

## 意外

预测对照（测试局）：

| | 预测 | 结果 | |
|---|---|---|---|
| ScaleBFM kin_rr | 1.3–2.6 | 1.84 | 中 |
| ScaleBFM kinc_rr | 0.7–1.5 | 1.26 | 中 |
| ScaleBFM modelf − kinc_rr | −0.3 到 +0.5，猜 +0.1；modelf 赢约一成，kinc_rr 赢约两成半 | −0.04 [−0.22, +0.40]，10 / 18，p 0.815，打平 | 中 |
| ScaleBFM kinc_rr − const | −0.9 到 −0.2 | −0.23 [−0.70, −0.06]，15 / 18，p 0.0075 | 中（靠边） |
| 开发局 ScaleBFM modelf − kinc_rr | −0.5 到 +0.5 | −0.04 | 中 |
| SONIC kin_rr / kinc_rr | 3–6 / 3–6 | 4.04 / 3.96 | 中 |
| SONIC modelf − kinc_rr | −1.5 到 +1.0，猜 −0.4，modelf 赢约一成半 | −0.37 [−1.16, +0.36]，14 / 19，p 0.064，打平 | 中 |
| 退回 J | 每组 ≤ 15 % | 最多 4 % | 中 |
| 有路线的局 45 秒内，各组和 J 差 | ≤ 10 % | ScaleBFM kin_rr 110、kinc_rr 108（J 111）；SONIC 69、65（J 68） | 中 |

没想到的：

1. **kin_rr（只按电机参考根瞄，不加常数）的成功数最好**，ScaleBFM 有路线的局官方时限内：

   | J | modelf | kinc_rr | kin_rr |
   |---|---|---|---|
   | 93 | 91 | 92 | 98 |

   - 所有局：kin_rr 118 / 106，J 119 / 101。
   - SONIC 官方时限内：kin_rr 55，J 56，modelf 42。
   - 挪过目标的各组里，只有 kin_rr 在 SONIC 官方时限内没掉。
   - 它挪得最少，entry 停偏最接近 J。
2. **SONIC 上 modelf 对 kinc_rr 差一点显著**（14 / 19，p 0.064）。这是模型在 SIMPLE 上离"赢"最近的一次，但没过线。
3. **kinc_rr 稳稳赢了固定常数**（ScaleBFM 15 / 18）：按电机参考根瞄，本身就是一个比标定更好的不用模型的做法。

## 结论与下一步

1. **两个控制器都落在登记的"打平"一支。**
   - ScaleBFM：modelf − kinc_rr −0.04 [−0.22, +0.40]，10 / 18，p 0.815。
   - SONIC：−0.37 [−1.16, +0.36]，14 / 19，p 0.064。
   - 开发局也是打平：−0.04、+0.00。
2. **读法照判据：SIMPLE 上比固定标定更准的那部分，来自"按电机实际跟的参考去瞄，再加一个常数"，不来自学出来的自身模型。SIMPLE 上自身模型不算贡献。**
   - kinc_rr 自己就赢了 const：−0.23 [−0.70, −0.06]，15 / 18，p 0.0075。
   - 停偏中位数：J 3.97、kin_rr 1.84、const 1.74、kinc_rr 1.26、modelf 1.28、oracle 0.61 cm。
3. **成功数上，不用模型的 kin_rr 最好**（意外 1）。TabletopGrasp 仍然几乎不奖励预抓取那一下的精度：所有挪过的组 entry 都比 J 差，差得最少的是 kin_rr。

**证据强度**：
- 一次运行，4 次抖动重复；
- 测试局 112 / 106 个局次，18 / 19 个布局；
- 开发局 44 / 55 个局次，结论相同。

**其他可能的解释**：
1. **模型的框架不对**：它是在"参考搬到真实骨盆上"的框架里训的。换到电机参考根的框架里重训（目标改成"实际的手减参考根放出来的手"），也许能预测剩下的跟踪误差。上限还有 0.65 cm 的空间（kinc_rr 1.26 对 oracle 0.61）。
2. **抖动几乎没传到手臂上**（precision 复核）。伸手几何真的变大时，kinc_rr 的常数可能失效，模型可能有用武之地；也可能模型自己的误差同样变大。

**四类陷阱**：
- **benchmark 选择**：同一批局，登记以前就定了。
- **数据泄漏**：c_rr 来自开发局 JR，运行以前冻结；开发局改成按位置名留一。
- **指标误用**：度量照登记，没换。
- **事后挑选**：kinc_rr 是复核在同一批测试局上离线找出来的。但它是对手，对模型不利；它的执行结果和离线外推一致（离线 +0.09，执行 −0.04）。

**下一步（要用户定方向）**：
1. **论文**：SIMPLE 上"一次命令的精度"的证据，不能记在自身模型上。三条路：
   - (a) 在电机参考根的框架里重训自身模型，再和 kinc_rr 比，先离线；
   - (b) 把精度的证据放到伸手几何真的会变、常数会失效的局上，另登记；
   - (c) 论文的主线改成别的。
2. **主表**：建议加"按电机参考根瞄"（kin_rr）这一行，或者直接把它并进 J。

## 独立复核

2026-11-25，一个独立的 agent 同时复核了 [selfmodel_precision3_20261125](selfmodel_precision3_20261125.md) 和 [selfmodel_sfull_20261125](selfmodel_sfull_20261125.md)。
- **它拿到的**：两份条目全文、precision2 的条目和复核、4090 上的逐局记录、代码路径。
- **不让它读的**：TODO、DECISIONS、handoff、beliefs。

原文（英文，两份条目贴的是同一份）：

> ## Findings
>
> 1. **Every number in both conclusions reproduces, and the files on disk are unchanged.**
>    - **Tables:** I re-ran `precision.py table3` on the 4 precision trees and `table_sfull` on the 2 S-full trees. All 6 outputs are byte-identical to the stored `table3.txt` and `table_sfull.txt`. The blocks pasted in the entries are identical to those files. The older `table` and `table2` still regenerate byte-identically in all 4 trees.
>    - **My own re-implementation** loads rows by exact file name, clusters layouts with its own code, uses a different bootstrap seed and an exact binomial sign test. It matches every median, paired median, episode and layout count, sign p, seen/unseen split, scale and fallback count, and success count. Only bootstrap bounds move, by Monte Carlo noise (e.g. dev ScaleBFM modelf − kinc_rr is [−0.57, +0.12] against the table's [−0.51, +0.12]).
>
>    | | precision3 ScaleBFM test | precision3 SONIC test | S-full ScaleBFM | S-full SONIC |
>    |---|---|---|---|---|
>    | modelf / SF − kinc_rr | −0.04 [−0.22, +0.42], 10/18, p 0.815 | −0.37 [−1.13, +0.37], 14/19, p 0.064 | +0.10 [−0.23, +0.30], 6/18, p 0.238 | +1.29 [−0.28, +6.15], 9/25, p 0.23 |
>    | kinc_rr − const | −0.23 [−0.70, −0.06], 15/18, p 0.0075 | −0.40 [−0.65, 0.00], 11/19, p 0.648 | −0.35 [−0.81, +0.04], 12/18 (not in the table) | **−1.28 [−2.06, −0.55], 19/25, p 0.015** (not in the table) |
>    | SF − const | – | – | −0.32 [−0.74, +0.19], 12/18 | −0.17 [−1.39, +1.71], 15/25 |
>    | Fallbacks (kin_rr / kinc_rr / modelf or SF) | 1 / 1 / 1 | 4 / 4 / 4 | 0 / 0 / 0 | 1 / 1 / 1 |
>    | Routed successes, 45 s / official (J; modelf or SF; kin_rr; kinc_rr) | 111/93; 105/91; 110/98; 108/92 | 68/56; 63/42; 69/55; 65/52 | 28/24; 27/21; 28/25; 28/21 | 14/11; 13/8; 16/14; 15/13 |
>
>    - **Dev:** ScaleBFM −0.04 (5/9), SONIC +0.00 (6/15).
>    - **Main metric, all episodes:** precision3 ScaleBFM J 119/101, kin_rr 118/106. S-full TabletopGrasp: ScaleBFM J 30/26, SF 29/23, kin_rr 30/27; SONIC J 15/12, SF 14/9, kin_rr 17/15.
>    - **Medians:** all match, e.g. precision3 ScaleBFM J 3.97, kin_rr 1.84, const 1.74, kinc_rr 1.26, modelf 1.28, oracle 0.61 cm.
>
> 2. **er is the motor's own reference end, with no carry onto the true pelvis.**
>    - **Recomputation:** I rebuilt er from `ref0`, the plan's last row Q[−1] and the look's goal, without calling `ref_root_end`. It matches `shifts3.jsonl` and `shifts_sf.jsonl` within 0.0005 cm on all 317 + 57 episode-repeats.
>    - **Shifts and planner inputs:**
>      - The shifts equal −er and −(er + c_rr) within 0.0005 cm.
>      - Every prep file, at every scale, equals J's prep plus k·shift rotated by the root's yaw (within 3e-10 m). No other field changed.
>    - **J's films:** the reference root is held exactly through the reach (difference 0), and FK of the final reference frame equals er's jaw (difference 0).
>    - **kin_rr's films:** the reference end lands 0.10–0.36 cm from the look's goal (median per repeat, full-scale episodes), against J's |er| of 2.3–3.8 cm. So kin_rr really aims the motor's reference at the goal.
>    - **qpos0 is not in er's value.** `SC.frames` (which reads qpos0) is called only inside an assertion (`precision.py` 118–120).
>    - **Where sim truth still enters:**
>      - **(i) The motor's root is itself anchored on sim truth.** `Motor._stand(pose=None)` uses `Motor.pelvis()`, which returns `sim.d.qpos` (`motor.py` 234–235, 326, 510). The look's goal is anchored to the same true pelvis at the stance (`simple_bench.py` 338, 380). I measured the two anchors: they are equal within 0.0007 cm and 3e-5°. So in the root's frame, er is (reference end relative to its root) minus (goal relative to the believed pelvis). The true pelvis cancels except the root's height, and er would be computable on a real robot. This does not matter.
>      - **(ii) Shared by every arm:** cuRobo plans in the true pelvis frame taken from A's film (`prep2.py` 54–78, `plan2.py` 120–122). This mismatch with the motor's root is where J's own aim error er comes from.
>      - **(iii) Only the model-side arms (g, kin, kinc, model, modelf, SF):** they carry the reference onto qpos0 and feed W, J and the carried reference to the model (`s_check.py` 117–130, 157–166, 177–192). At the command the root is 1.9–2.2 cm from the true pelvis on ScaleBFM, and 2.9–5.9 cm (max 18.9 cm) on SONIC.
>      - **(iv) const** is rotated by W's yaw (`precision.py` 152, 361, 394). Yaw only, minor.
>    - **Does it matter?** The asymmetry gives modelf/SF privileged input, so the tie is conservative against the model. It does matter for deployment: kinc_rr runs on a real robot as is; SF does not.
>
> 3. **c_rr came only from the dev JR rows and was frozen before every run.**
>    - **Recomputation:** from `simple_dev_20261124/bench/<ctrl>/rows/JR_TabletopGraspMP.jsonl`: 11 / 15 rows, levels 3–4, no jitter key, the same names as `const_*.json`. Matches the file within 0.0005 cm per episode.
>    - **Timing:**
>
>      | Event | Time (UTC) |
>      |---|---|
>      | Registration commit `dcb83c3b` | 21:23:30 |
>      | c_rr files written | 21:34:55 / 21:34:57 |
>      | First use (trial) | 21:36:05 |
>      | S-full test start | 21:41:21 |
>      | precision3 dev start | 22:07:03 |
>      | precision3 test start | 22:19:47 |
>
>    - The copies in `asrun3/`, `asrun_sf/` and both trials all have md5 05a1… / 8058…. Both entries changed only by appending below their registration.
>    - **Leave-one-out:** test uses the full mean (crr_n 11 / 15). Dev leaves out by position name: n = 10 on ScaleBFM, 14 or 15 on SONIC; recomputed per episode within 0.0005 cm.
>    - **Caveat:** leaving out by name is not leaving out by layout. ScaleBFM dev L3_e0, L3_e7 and L3_e9 form one layout. This affects dev only.
>
> 4. **New leakage: dev and test positions coincide physically.**
>    - 5 of 11 ScaleBFM dev episodes (and 4 of 15 SONIC) have their look goal 0.21–0.93 cm from a test position, e.g. dev L3_e5 ↔ test L2_e4 at 0.21 cm, L4_e1 ↔ L2_e5 at 0.34 cm. Random placement would give about 0.15 such pairs.
>    - This affects const, c_t and c_rr alike.
>    - **Effect of dropping them:** c_rr moves by [−0.44, −0.15, +0.16] (ScaleBFM) and [−0.53, −0.37, +0.45] cm (SONIC). Offline kinc_rr goes 1.10 → 1.29 and 4.53 → 4.72.
>    - **Anchored on the executed rows**, modelf − kinc_rr becomes −0.09 (13/18, p 0.096) on ScaleBFM and −0.25 (13/19) on SONIC.
>    - It does not flip anything, but it favours the constant arms, and neither entry's leakage check mentions it.
>
> 5. **The file-glob fix is complete.**
>    - `rows_of` (`precision.py` 129–137) now requires an uppercase letter after `<arm>_`. No other arm-name pair collides (J_/JR_, model_/modelf_, S_/SF_).
>    - **Provenance check:** every planned row of every arm has its `joint_src` in `P/<that arm>/step2*`, and every fallback row carries J's plan. There are 0 exceptions (10 arms × 317 episode-repeats, 5 arms × 57).
>    - `bench_via.sh` deletes only `rows/<arm>_<Task>.jsonl` (line 55). The kin and kinc rows still have their 20:43Z mtimes.
>    - The only residue is cosmetic: the row counts in `pipeline.log` (line 61), e.g. "arm kinc: 56 rows" in S-full ScaleBFM.
>
> 6. **(a) Power: the ScaleBFM tie is real; the SONIC tie is underpowered.**
>    - **Method:** resample layouts from the median-centred observed differences, shift by δ, apply the registered rule (bootstrap interval below 0 and layout sign p < 0.05). 300 simulations × 1000 bootstrap draws.
>    - **The sign test needs:** 14/18, 15/19 or 18/25 layouts.
>
>    | | power at δ = 0.3 / 0.5 / 1.0 cm | smallest δ detectable at 80 % |
>    |---|---|---|
>    | precision3 ScaleBFM | 0.30 / 0.71 / 0.96 (0.95 at 0.6) | about 0.55 cm |
>    | precision3 SONIC | 0.09 / 0.30 / 0.61 (0.69 at 1.5) | above 1.5 cm |
>    | S-full ScaleBFM | 0.84 / 0.95 / 1.00 | about 0.3 cm |
>    | S-full SONIC | 0.03 / 0.03 / 0.06 | none: uninformative |
>
>    - **ScaleBFM:** the interval's lower bound (−0.22) rules out a modelf advantage above about 0.2–0.25 cm. It cannot rule out 0.1–0.2 cm, which is below the replanning floor (the oracle's 0.61 cm).
>    - **SONIC:** the lower bound is −1.13 cm.
>      - The near-win depends on layout *means*. With layout medians it is 11/19 (p 0.65).
>      - Per repeat the difference is −0.58, +0.36, −1.10, −0.41.
>
> 7. **(b) A predictor in the reference-root frame has a real offline ceiling, but the current model captures none of it.**
>    - **Setup:** target t = jaw − er (root frame), on precision2's test J rows, leave-one-layout, ridge with λ chosen by inner leave-one-layout.
>    - **ScaleBFM:** sd of t is [0.91, 0.73, 0.48] cm, and 64–85 % of its variance lies between layouts.
>
>    | Predictor (ScaleBFM) | Offline median miss (cm) | − kinc_rr | Layouts better |
>    |---|---|---|---|
>    | kin_rr (t itself) | 1.88 | +0.99 | 4/18 |
>    | kinc_rr (dev c_rr) | 1.10 | 0 | – |
>    | Leave-one-layout mean of about 100 test episodes | 1.00 | +0.03 | 10/18 |
>    | The self model's implied t (p − u) | 1.09 | +0.10 | 8/18 |
>    | Ridge on er + goal | 0.76 | −0.23 | 16/18 |
>    | Ridge on er + goal + joints, joint velocities, plan end, reference joints, duration | **0.58** | −0.42 | 15/18 (R² 0.71 / 0.49 / 0.61) |
>    | Same plus the true pelvis against the root (sim truth) | 0.59 | −0.40 | 15/18 |
>    | Ridge on er + goal, trained on the 44 dev episode-repeats only | 0.82 | −0.18 | 17/18 |
>    | Full ridge, trained on dev only | 0.91 | −0.12 | 11/18 |
>    | Oracle | 0 (0.61 executed) | | |
>
>    - **From offline to executed:** kinc_rr's 1.10 offline becomes 1.26 executed, which is exactly √(1.10² + 0.61²) with the oracle's 0.61 as the floor.
>      - On that mapping the ceiling ridge would execute at about 0.84 cm: −0.4 cm against kinc_rr, which this design detects only about 45 % of the time.
>      - Trained on dev only, it would execute at about 1.02 cm (−0.24, power about 0.25).
>    - **What this shows:**
>      - About half of kinc_rr's remaining error is predictable at the command from proprioception, given in-distribution episodes. That contradicts S-full's "复核的离线数说这部分可预测的不多".
>      - The current model predicts none of it, even though 92 of the 112 test episodes are at layouts in its training data.
>      - A 6-feature affine fit on dev alone is a stronger model-free opponent than kinc_rr (17/18 layouts offline).
>    - **SONIC:** sd of t is [9.6, 5.3, 1.6] cm, and only 14–26 % lies between layouts.
>      - Ridge reaches R² ≤ 0.27, and trained on dev it is worse than c_rr.
>      - The self model's implied t is the best offline predictor: 3.47 against 4.53 (−0.52, 10/19).
>      - SONIC extrapolation is unreliable, but offline and executed both lean toward the model.
>
> 8. **(c) The success counts are noise.**
>    - **precision3 ScaleBFM, routed, official limit:** kin_rr 98 against J 93 (+7 / −2, exact McNemar p 0.18). But the no-op arm `replan` (same goal, new IK seed) also scores 98 (+5 / −0). kin_rr against replan is +3 / −3. Per repeat, kin_rr − J is −1, +3, +3, 0 and replan − J is 0, +3, +2, 0: J was simply an unlucky draw.
>    - **SONIC:** J and replan disagree on official success in 25 of 106 routed episode-repeats (24 %). That is the noise floor. S-full SONIC kin_rr against J is +4 / −1 at the official limit (p 0.375) and +3 / −1 at 45 s (p 0.625). At a 24 % disagreement rate, about 7 of 29 episodes would differ by chance alone.
>    - **The one significant success effect runs the other way:** at 45 s on ScaleBFM precision3, the larger shifts cost successes (const −7, p 0.02; modelf −6, p 0.03). kin_rr loses only 1. kin_rr is the least harmful shift, not a success improver.
>
> 9. **The numbers are right, but some claims in the text are not.**
>    - **precision3 conclusion 3 and 意外 1, "kin_rr 成功数最好":** on ScaleBFM it ties replan (98). On SONIC at the official limit J is best (56) and kin_rr ties const (55). On SONIC at 45 s kinc has 70, above kin_rr's 69. "只有 kin_rr 在 SONIC 官方时限内没掉" is wrong too: const is also at 55, and kin_rr did drop by 1.
>    - **"所有挪过的组 entry 都比 J 差，差得最少的是 kin_rr":**
>      - On ScaleBFM, kin (+0.04) and sham (+0.04) are closer to J than kin_rr (+0.08).
>      - On SONIC, const is better than J (−0.26 [−0.43, −0.13], 14/19), and kin_rr (−0.00), kinc_rr (−0.15) and modelf (−0.09) are not worse.
>    - **S-full 意外 1:** on ScaleBFM, kin_rr's 95/82 and TabletopGrasp 30/27 are shared with const.
>    - **The gain over the constant exists only on ScaleBFM precision3.**
>      - On SONIC precision3, neither modelf (12/19, p 0.36) nor kinc_rr (11/19, p 0.65) beats const.
>      - c_rr adds nothing to kin_rr there (kin_rr − kinc_rr +0.00, 10/19).
>      - Unreported: on S-full SONIC, kinc_rr beats const (−1.28, 19/25, p 0.015; exploratory, not a registered comparison).
>
> 10. **One modelf outlier is a planner artifact.**
>     - ScaleBFM L1_e1 has modelf at 29 cm in all 4 repeats.
>     - Cause: the full shift failed IK; the 2/3 retry returned a plan whose end joints differ from J's by 83–87°, and the arm hit the table.
>     - In S-full, SF at the same position also used 2/3 (75°) and missed by 0.89 cm.
>     - It costs modelf one layout in the sign test and does not change any verdict.
>
> 11. **The four pitfalls.**
>     - **Benchmark selection:** precision is tested only on TabletopGrasp, with a ±6 cm jitter mostly absorbed by the table-edge clamp (`simple_bench.py` 365). Registered in advance and scoped to SIMPLE, but this is a setting where a constant wins by construction.
>     - **Data leakage:**
>       - c_rr is clean in time.
>       - The dev/test position overlap (Finding 4) is new and favours the constants.
>       - The model's training data covers 92/112 test layouts, which favours the model.
>       - The unseen splits (20 / 12 episodes, 3 layouts) have no power.
>     - **Metric misuse:** the metric is as registered. The layout-mean sign test is sensitive to outliers (Finding 10, and SONIC's means against medians). The success counts are noise (Finding 8).
>     - **Post-hoc selection:** kinc_rr was found offline on these same test episodes, and the predictions were written after seeing the offline +0.09. So precision3 re-executes a known result. S-full is a semi-independent replication (same positions, no jitter, never used for the er analysis): +0.10. kin_rr's success "win" is a post-hoc highlight and it is noise.
>
> ## Verdicts
>
> | Conclusion | Verdict | Reason |
> |---|---|---|
> | precision3 1: both controllers land in the registered tie branch | **Supports** | Every number reproduces. It holds per repeat, with both arms at full scale, and without falls. Informative on ScaleBFM (rules out a modelf edge above about 0.22 cm). Underpowered on SONIC (80 % power only above 1.5 cm). |
> | precision3 2: the gain over the fixed calibration comes from the motor's reference plus a constant; the self model earns no credit on SIMPLE | **ScaleBFM: supports. SONIC: insufficient evidence** | ScaleBFM: kinc_rr − const 15/18, modelf − kinc_rr −0.04. SONIC: no significant gain over const to explain; c_rr adds nothing to kin_rr; the tie is underpowered and leans toward the model (−0.37 executed, −0.52 offline). It reads correctly as "no credit shown", not as "the model cannot help" (Finding 7). |
> | precision3 2, sub-claims (kinc_rr beats const; the medians) | Supports | Reproduced exactly. ScaleBFM only. |
> | precision3 3: kin_rr best on successes; every shifted arm worse than J at entry, kin_rr least | **Does not support** | kin_rr's 98 equals the no-op replan arm's 98. SONIC J 56 > kin_rr 55 = const 55. kin and sham are closer to J at entry on ScaleBFM; const is better than J at entry on SONIC. "TabletopGrasp barely rewards pregrasp precision" is supported. |
> | precision3 evidence strength | Supports | Counts are correct. Add the power caveat for SONIC. |
> | precision3 alternative 1: retrain in the reference-root frame, 0.65 cm of room | Supports as an open alternative | Offline ceiling 0.58 against 1.10 (R² 0.5–0.7). The current model gets none of it. A dev-only affine fit gets −0.18. |
> | precision3 alternative 2: reach geometry barely varies | Supports (untested) | The jitter is clamped at the table edge. |
> | precision3 pitfalls (leakage, post-hoc) | Partly supports | True in time, but misses the dev/test position overlap (Finding 4) and the model's layout exposure. |
> | precision3 next step 2: add a kin_rr row or merge it into J | Supports on precision grounds only | kin_rr − J −2.13, 18/18; the reference end lands within 0.2 cm of the goal. Not on success grounds. |
> | S-full 1: SF beats neither const and kinc_rr together; no model credit on the official episodes | **Supports** (as the rule's outcome) | ScaleBFM is informative (rules out an SF edge above 0.23 cm). SONIC has essentially no power (≤ 0.11 at 1.5 cm). SF does not even beat const there. |
> | S-full 1, sub: SF beats only the mis-framed kinc | Supports | −0.55, 14/18, p 0.031 |
> | S-full 2: main table S → SF (94/78, 33/17); official limit still −3 / −3 | Supports | Numbers reproduce. The −3 is not significant (McNemar p 0.25 / 0.45). |
> | S-full 3: recommend a model-free row; kin_rr's successes are better | Recommendation supported; "kin_rr better" **does not support** | p 0.375 on SONIC; ties const on ScaleBFM; at the no-op noise level (Finding 8). |
> | S-full alternative 1: "复核的离线数说可预测的不多" | Does not support | About half of the residual is predictable offline (Finding 7), just not by this model. |
> | S-full alternative 2: few episodes, low power | Supports | Finding 6 |
> | S-full 意外 3: cuRobo is nondeterministic | Supports | J and the no-op replan disagree on 4 % (ScaleBFM) and 24 % (SONIC) of routed official outcomes. |
> | S-full leakage check ("这批局之前没做过 er 的分析") | Partly supports | True, but the positions are shared with the jittered test set and overlap the dev calibration set. |
>
> ## How I verified it
>
> - **Tables:** re-ran `precision.py table`, `table2`, `table3` and `table_sfull` (with `--s-choice`, `--check`, `--range`) into scratch and compared them with `cmp`. Matched the pasted markdown blocks against the files.
> - **Statistics:** my own script loads rows by exact `<arm>_<Task>.jsonl` names and asserts no duplicates. It has its own layout clustering (connected components, 1 cm), its own bootstrap seed, an exact binomial sign test, a check of the routed sets across `shifts`, `shifts2` and `shifts3`, a provenance check of `joint_src` against pick for each arm, and counts of successes, fallbacks, scales and falls. I also checked that the J and JR rows are identical (0 differences).
> - **Integrity (CPU only, `CUDA_VISIBLE_DEVICES=""`):**
>   - Recomputed er with `ArmFK` from `ref0` and Q[−1], and c_rr from the dev bench rows and the `ctrack_state` dirs.
>   - Checked dev leave-one-out by name, every prep file at every scale, J's film reference (root and end), the kin_rr and kinc_rr film ends against the goal, the root against the row's belief anchor and against qpos0, file times, and md5s on Mac and the 4090 (all match the entries).
>   - Measured the dev/test goal distances.
> - **Probes:**
>   - Layout-resampling power simulation.
>   - Leave-one-layout ridge (nested λ) and a dev-trained ridge, on t = jaw − er.
>   - The self model's implied t from `shifts2` (p − (er − g)).
>   - An anchored re-estimate with a non-overlapping c_rr.
>   - Exact McNemar tests against J and against the replan arm.
>   - Robustness: per repeat, both arms at full scale, without falls, and layout medians.
> - **Housekeeping:** read-only throughout. Scratch `/tmp/rv_p3_b7a87220` (remote) and `/tmp/rv_p3_mac_8kfW` (Mac) are deleted. `git status` is unchanged. No GPU was used and no process of anyone else's was touched. I did not read TODO.md, DECISIONS.md, the other handoff docs or beliefs.md.
>
> ## Experiments that could overturn the conclusion
>
> 1. **Self model retrained in the motor's root frame (primary).**
>    - Target: actual palm minus the reference placed by `ref0`. Inputs: no qpos0 or W, only root-frame proprioception and the reference.
>    - Offline gate on precision2's test J rows, leave-one-layout: it must beat c_rr (1.10) and a dev-fitted affine on (er, goal) (0.82). The ceiling is about 0.58.
>    - If it passes, register an execution against kinc_rr and the dev affine on fresh positions. Beating both under the registered rule would overturn "no credit".
> 2. **A powered SONIC replication.**
>    - SONIC is where the point estimates favour the model.
>    - It needs roughly 3–4× the episode-repeats (e.g. 60 positions, or 12 repeats) to reach 80 % power at about 0.5–1 cm.
>    - Use a fresh jitter seed and positions never used for diagnosis, and pre-register the layout-median sign test so outliers cannot decide it.
> 3. **Geometry that breaks the constant.** Apply the stance offset after the table-edge clamp (`simple_bench.py` 365), and vary target and pelvis height by about ±10 cm. Calibrate c_rr and the affine fit at the nominal geometry. If kinc_rr degrades and the model does not, the model earns credit.
> 4. **A clean calibration set.** Rebuild const, c_t and c_rr from dev positions at least 5 cm from every test position, then rerun kinc_rr. The anchored estimate moves ScaleBFM to −0.09 (13/18). This alone is unlikely to flip the result, but it removes a bias in the constants' favour.
> 5. **Data efficiency.** Plot the error of a model-free affine fit against N = 3…100 real episodes. If the self model (zero task episodes) matches the fit at a large N, report the credit as "N episodes saved", not as precision.
> 6. **Any success claim** needs at least 3 planner seeds per arm, with replan as the reference arm.

### 复核后的更正（2026-11-25）

数全部重算对上，流程、er、c_rr、串组的修法都核对过。

1. **"打平"成立，但两个控制器的分量不同。**
   - **ScaleBFM 的打平是真的**：区间下沿 −0.22 cm，排除了模型有超过约 0.2 cm 的优势；这次的设计对 0.55 cm 的差有 80 % 的检验力。
   - **SONIC 的打平检验力不够**：只有差超过 1.5 cm 才有 80 % 的把握测出来，而且点估计偏向模型（执行 −0.37，离线 −0.52）。
2. **结论 2 改为：ScaleBFM 成立；SONIC 证据不足。**
   - SONIC 上 kinc_rr 本身没赢 const（11 / 19），c_rr 对 kin_rr 也没加上什么（10 / 19），所以没有"比标定更准的那部分"需要解释。
   - SONIC 的读法是"没显示出模型的贡献"，不是"模型帮不上"。
3. **撤回"kin_rr 成功数最好"**（意外 1、结论 3）。
   - ScaleBFM 上它的 98 和什么都不挪、只换 IK 种子的 replan 组一样（98）。J 那 93 是运气不好的一次：逐次重复 kin_rr − J 是 −1、+3、+3、0，replan − J 是 0、+3、+2、0。
   - SONIC 官方时限内 J（56）最高，kin_rr 和 const 都是 55。"只有 kin_rr 没掉"也不对。
   - 成功数上唯一显著的是反方向：ScaleBFM 45 秒内，挪得多的组掉分（const −7，p 0.02；modelf −6，p 0.03），kin_rr 只少 1。kin_rr 是"害处最小的挪法"，不是"让成功变多的挪法"。
4. **撤回"所有挪过的组 entry 都比 J 差、kin_rr 差得最少"**：ScaleBFM 上 kin、sham 比 kin_rr 更接近 J；SONIC 上 const 的 entry 反而比 J 好（−0.26，14 / 19）。
5. **其他解释 1（在电机参考根的框架里重训）有实在的空间，现在的模型一点都没拿到。**
   - 用发令时刻就有的本体信息（关节、关节速度、规划终点、参考关节、时长，加 er 和目标）做 ridge 回归，按留一布局预测 t = 手 − er：ScaleBFM 离线中位数 0.58 cm，对 c_rr 的 1.10 是 −0.42，15 / 18。
   - 现在的模型隐含的 t 是 1.09，等于没有。
   - **只用开发局 44 个局次拟合的 (er, 目标) 仿射**，离线就到 0.82（17 / 18 个布局比 c_rr 好），是一个比 kinc_rr 更强的不用模型的对手，以后必须带上。
   - SONIC 上 ridge 几乎学不到（R² ≤ 0.27），反而现在的模型隐含的 t 离线最好（3.47 对 4.53）。
6. **新的泄漏：开发局和测试局有几个位置物理上重合。**
   - ScaleBFM 开发局 11 局里有 5 局、SONIC 15 局里有 4 局，目标离某个测试位置 0.21–0.93 cm；随机摆放只会有约 0.15 对。
   - 这让 const、c_t、c_rr 都占了便宜。去掉它们以后，ScaleBFM modelf − kinc_rr 变成 −0.09（13 / 18），不翻结论。
7. **仿真真值的分布**：
   - **er 的值不靠真实骨盆**：电机的根和开局那一眼的目标锚在同一个站位上，两者差 0.0007 cm，所以 er 在真机上也能算。
   - **模型那几组用了真实骨盆**（g、kin、kinc、model、modelf、SF），等于给了模型额外的信息。所以这次打平对模型是偏宽的；部署时 kinc_rr 能直接用，SF 不能。
8. **一个异常局是规划器造成的**：ScaleBFM L1_e1 的 modelf 4 次都停在 29 cm 外。全量挪 IK 失败，2/3 的那条路线终点关节和 J 差 83–87°，手撞到桌子。不改任何结论。
9. **benchmark 选择**：抖动在桌沿被限位吸收了（`simple_bench.py` 第 365 行），这是常数天然占优的设定。登记时就这样，但要写明。
10. **下一步（取代原来的"下一步"）**：
    - (a) **在电机参考根的框架里重训自身模型**：目标是"实际的手掌 − ref0 放出来的手掌"，输入不用 qpos0 和 W。
      - 先离线过关：在 precision2 测试局的 J 行上按留一布局，要同时赢 c_rr（1.10）和开发局拟合的 (er, 目标) 仿射（0.82）；上限约 0.58。
      - 过关了再登记执行，用新的位置和新的抖动种子，对手是 kinc_rr 和开发局仿射。
    - (b) **让伸手几何真的变**：站位扰动加在桌沿限位之后，目标和骨盆高度变 ±10 cm，看常数和仿射会不会失效。
    - (c) **SONIC 要有检验力**：要 3–4 倍的局次，并预先登记按布局中位数的符号检验。
    - **成功数**：以后要报，至少 3 个规划器种子，以 replan 为参照组。
    - **常数**：从离任何测试位置至少 5 cm 的开发局位置重估。
