# selfmodel_precision2_20261125：补上"参考离目标的差"以后，模型的完整预测能不能赢过常数和不用模型的逐局基线

- 日期：2026-11-25（设计、预测、判据；跑之前写下，单独提交）
- 机器 / GPU：4090，1–3 号卡（ScaleBFM 1 号，SONIC 2、3 号），和 [selfmodel_precision_20261124](selfmodel_precision_20261124.md) 同样的分配。
- 代码版本：`selfwm` 加未提交的 `scripts/experiments/simple_eval_20261112/precision.py`（新加三种挪法和对应的表），其余和上一次一样（`precision_run.sh` 只加三种挪法的名字），跑时抄进输出目录的 `asrun/`，md5 记在结果里。
- 配置：同上一次。SIMPLE 的 TabletopGrasp，感知栏，单目 RGB-D，G1 加 Dex3，ScaleBFM 和 SONIC，控制器不改参数，不用伺服和查表。
- 关联：
  - [selfmodel_precision_20261124](selfmodel_precision_20261124.md) 的独立复核：model 那一组只挪了"预测的手减参考的手"，没管"参考在 J 路线终点离目标的差"。后者发命令时就能用 J 的路线和发令时刻的状态算出来：ScaleBFM 长度中位数 1.96 cm，SONIC 9.45 cm。常数把它吸收了。
  - 复核离线外推：补上这段差以后，ScaleBFM 上 model 减 const −0.58 cm（−0.81 到 −0.35），18 / 18 个布局更近；对"参考离目标的差加跟踪常数" −0.48 cm，18 / 18。SONIC 打平。
  - 用户 2026-11-24 定的主论断：一次命令的精度。

## 假设

一次命令下去手停在哪 = 参考在终点离目标的差（运动学，发命令时已知）+ 控制器的跟踪误差（自身模型预测的部分）。

上一次只用模型修了第二项。两项都修以后：

**模型的完整预测，能不能比固定常数、比"运动学的差加一个跟踪常数"都更准？** 后者是不用模型时最强的逐局做法。

## 做法

### 局：沿用上一次，不重跑 J

- 开发局：`outputs/simple_dev_20261124/precision_20261124/<控制器>/r0–r3/`（第二次开发局）。
- 测试局：`outputs/simple_eval_20261112/precision_20261124/<控制器>/r0–r3/`。
- 每次重复里同样的站位抖动、同样的 J 路线、同样的发令时刻状态。J、const、oracle 三组的行直接用上一次的，不重跑。新加的三组从同样的状态出发，和它们逐局配对。

### 三种新挪法（只挪预抓取的目标）

记号都在发令时刻的朝向坐标里（cm）：

- g：参考在 J 路线终点的夹爪中心减目标。用 `s_check.reference` 在发令时刻的状态上重建 J 的参考，取终点。不用测，不用模型。
- p：模型预测的"实际的手减参考的手"。手掌三个球的平均，就是上一次 model 那一组用的 `pred`。
- c_t：跟踪常数，开发局（3、4 级，不抖动，`outputs/simple_dev_20261124/bench/<控制器>/rows/JR_*`，和 const 同一批局）上"实际的夹爪中心减参考终点的夹爪中心"的平均。开跑前算好、存文件、记 md5，之后不改。开发局用留一布局的版本，和 const 一样。

三组：
- **kin**：挪 −g。只去掉运动学的差。
- **kinc**：挪 −(g + c_t)。不用模型时最强的逐局基线。
- **modelf**：挪 −(g + p)。模型的完整预测。

规划、缩放重试（全量、2/3、1/3，都不行退回 J）、执行都和上一次完全一样。

### 顺序

1. 用开发局跑三组新挪法，查流程判据。
2. 流程判据过了，接着跑测试局，不等人。

## 度量

和上一次一样：
- 预抓取停偏，每组减 J、按局配对；
- 按布局（预抓取点 1 cm 以内算一个布局）重抽 4000 次的 95 % 区间和符号检验；
- 见过和没见过的布局分开报；
- entry 停偏；
- 主指标：TabletopGrasp 的成功局数，45 秒内和官方时限内；
- 每组的缩放和退回次数。

核心比较是 **modelf 减 const** 和 **modelf 减 kinc**。

## 预测（跑之前写；看过复核的离线外推，没看过任何新挪法的执行结果）

ScaleBFM（测试局）：
- kin 停偏中位数 3.5 到 6 cm，可能比 J（3.97）还差。参考离目标的差在侧向（−1.65 cm）和跟踪误差的侧向部分方向相反，只去掉前者会把后者露出来。
- kinc 1.0 到 2.0 cm。
- modelf 0.8 到 1.8 cm（离线外推约 1.2）。
- modelf 减 const −1.0 到 −0.1，中间猜 −0.5；modelf 减 kinc −0.9 到 0，中间猜 −0.4。
- 核心判据两条都过的可能，我估约六成。执行和外推在 ScaleBFM 上中位数差 0.5 cm 以内，但缩放重试会削掉一部分挪动。
- 开发局（留一常数）：modelf 减 const −1.5 到 0。

SONIC（测试局）：
- kin 6 到 12 cm，比 J（5.92）差。参考离目标的差（9.45 cm）和跟踪误差大致抵消，只去掉一个就露出另一个。
- kinc 3 到 7 cm，modelf 3 到 7 cm。
- modelf 减 const −1.5 到 +1.0，多半打平；两条都过的可能约一成半。

两个控制器：
- 每组退回 J 不超过 15 %。
- 主指标（有路线的局，45 秒内）：各组和 J 差不超过 10 %。

## 判据（什么结果会让我改做法，下一步做什么）

**核心判据，每个控制器分开，以测试局为准**。"赢"的定义和上一次一样：按布局重抽的 95 % 区间整个在 0 以下，并且符号检验 p < 0.05。

- **modelf 同时赢 const 和 kinc**：
  - 这个基准上可以说"模型的逐局预测比标定、比运动学加常数都准"，SIMPLE 上模型算贡献。
  - S 改成完整预测（运动学的差加模型），另登记，在 SIMPLE 官方局（不抖动）上重跑 S。
  - 主表的 S 用这个版本。
- **modelf 赢 const、不赢 kinc**：增益来自运动学的那段差，不来自模型。SIMPLE 上模型不算贡献，主表写"运动学修正加常数"。
- **modelf 两个都不赢**：同上一次，常数就够。

**流程判据（开发局，决定接着跑测试局）**：
- 出现以下任一情况，就停下先查，不跑测试局：
  - 有进程失败；
  - 某组退回 J 超过 30 %；
  - 新三组任一组的站位抖动、发令时刻状态和同局 J 不同。
- 测试局的预测和判据就是本节这些，开发局的结果不改它们。

## 预测修订

2026-11-25，跑完以后补记实现上和登记写法不同的地方（预测和判据没改）：

- **代码副本的目录名**：登记写"抄进 `asrun/`"，实际抄进各控制器目录的 `asrun2/`，旧那次的 `asrun/` 没动。
- **新建 `precision2_launch.sh`**：原来的 `precision_launch.sh` 没改，这样旧那次的 started / done 标记和旧启动脚本的 md5 都保持不变；新的标记叫 `started2` / `done2`。
- **`precision_run.sh` 比"只加三种挪法的名字"改得多**：加了一个 `NEW_ARMS=1` 模式，跳过 A、plan、JR、state，进度写 `run2.log`，表写 `table2.txt`。旧的那条路行为不变。
- **`precision.py` 的 `table` 和新的 `table2` 共用了一段改写过的表格代码**：已验证旧 `table` 在开发局和测试局、两个控制器上的输出都和原来的 `table.txt` 逐字节一致。
- **`bench_via.sh` 的 S 步把 4 个代码文件重新抄进了各重复的 `asrun_S/`**：开跑前核对过，这 4 个文件和旧副本 md5 相同，内容没有变化。
- **试跑**：在开发局 r0 第 30、48 两个位置的副本上，放在 `outputs/simple_dev_20261124/precision2_trial/`，r0 本身没动。
- **c_t 文件的写入时间**：两份都在 2026-11-24 19:57Z 写好。登记的提交 `2425acc1` 在 19:49:38Z，开发局在 20:00Z 开跑。

## 结果（贴原始输出，不转述）

### 时间、代码、c_t

```
2425acc1 2026-11-25T03:49:38+08:00            (登记提交 = 19:49:38Z)
ctrack_scalebfm.json 2026-11-24 19:57:12Z  md5 65c83f0016901c0d642a29c9828e0a43
ctrack_sonic.json    2026-11-24 19:57:17Z  md5 1e86e06299b4b055045fe6fb99953fea
dev  started2 2026-11-24T20:00:01Z  done2 2026-11-24T20:15:13Z
test started2 2026-11-24T20:16:12Z  done2 2026-11-24T20:54:44Z
```

`asrun2/` 里的代码（开发局、测试局、两个控制器相同，和 Mac 工作区一致）：

```
82cd6dc26e91ec4721d53ea7bff1b3cb  bench_via.sh
3161d616875f6c4a238a3c79fa29b115  plan2.py
f74315246ba75c5a4e5897f2326d05c5  precision.py
8845ae032d3213efb7804e90aca3af78  precision_run.sh
4106eba55de8b3ecb4b543a8cfca7911  s_check.py
805111408bf83bef8724b109a427adb1  simple_bench.py
d6cad388663085d857ca478bc665a3d3  precision2_launch.sh (scripts/ 下)
```

`outputs/simple_dev_20261124/precision_const/ctrack_<控制器>.json` 的汇总字段：

```
== ctrack_scalebfm.json
  arm "JR"
  n 11
  mean_ctrack_h_cm [-3.879, 1.638, 3.368]
  sd_ctrack_h_cm [1.78, 0.549, 1.776]
  mean_g_h_cm [0.711, -1.538, -0.007]
  mean_g_plus_ctrack_h_cm [-3.168, 0.1, 3.361]
  mean_miss_h_cm [-3.168, 0.1, 3.363]
  jaw_fk_vs_row_mm {"median": 0.04, "max": 0.07}
== ctrack_sonic.json
  arm "JR"
  n 15
  mean_ctrack_h_cm [-2.427, -6.917, 1.863]
  sd_ctrack_h_cm [4.564, 6.952, 4.346]
  mean_g_h_cm [-1.453, 8.215, 0.484]
  mean_g_plus_ctrack_h_cm [-3.88, 1.299, 2.347]
  mean_miss_h_cm [-3.878, 1.3, 2.347]
  jaw_fk_vs_row_mm {"median": 0.14, "max": 0.84}
```

- **g + c_t 等于 const 的平均停偏**：差不超过 0.002 cm。这是按定义相加必然成立的，只说明两边用的是同一批局和同一个坐标系。
- **`jaw_fk_vs_row_mm`**：规划这边算出的夹爪中心和 bench 实测的 `jaw_mm`，在 film 最后一帧预抓取上的差。

### 试跑（开发局 r0 第 30、48 两个位置的副本）

kin、kinc、modelf 的实际停偏，对"J 的停偏 + 缩放 × 挪动"：

```
== scalebfm
  r0 TabletopGraspMP_L3_e0: g [0.51, -1.309, 0.14] pred [-2.941, 2.381, 3.177] c_t [-4.216, 1.72, 3.605] (n 8) | J miss [-2.257, 0.301, 2.45] |3.34| | pred vs shifts.jsonl 0.0034 cm
    kin    scale 1        tried ['1:None']  J miss + scale*shift [-2.77, 1.61, 2.31] |3.95|  executed [-2.62, 1.61, 2.4] |3.90| (err_cm 3.9)  diff 0.17 cm
    kinc   scale 2/3      tried ['1:IK_FAIL', '2/3:None']  J miss + scale*shift [0.21, 0.03, -0.05] |0.22|  executed [0.45, -0.23, -0.31] |0.59| (err_cm 0.586)  diff 0.43 cm
    modelf scale 1        tried ['1:None']  J miss + scale*shift [0.17, -0.77, -0.87] |1.17|  executed [0.66, -1.15, -1.17] |1.77| (err_cm 1.768)  diff 0.69 cm
  r0 TabletopGraspMP_L4_e8: g [1.024, -1.545, -0.656] pred [-3.794, 1.817, 1.926] c_t [-4.02, 1.725, 3.532] (n 9) | J miss [-2.574, -0.868, 1.09] |2.93| | pred vs shifts.jsonl 0.0042 cm
    kin    scale 1        tried ['1:None']  J miss + scale*shift [-3.6, 0.68, 1.75] |4.06|  executed [-3.38, 0.74, 1.73] |3.87| (err_cm 3.871)  diff 0.23 cm
    kinc   scale 1/3      tried ['1:IK_FAIL', '2/3:IK_FAIL', '1/3:None']  J miss + scale*shift [-1.58, -0.93, 0.13] |1.83|  executed [-1.89, -0.98, 0.19] |2.14| (err_cm 2.139)  diff 0.33 cm
    modelf scale 2/3      tried ['1:IK_FAIL', '2/3:None']  J miss + scale*shift [-0.73, -1.05, 0.24] |1.30|  executed [-1.4, -1.11, 0.38] |1.83| (err_cm 1.835)  diff 0.69 cm
  |executed - (J miss + scale*shift)| median per arm (cm): kin 0.20 (n 2), kinc 0.38 (n 2), modelf 0.69 (n 2)
== sonic
  r0 TabletopGraspMP_L3_e0: g [0.814, 1.983, 0.867] pred [-3.047, -1.598, 5.0] c_t [-2.245, -6.692, 1.064] (n 13) | J miss [-5.834, 1.974, 4.47] |7.61| | pred vs shifts.jsonl 0.0032 cm
    kin    scale 1        tried ['1:None']  J miss + scale*shift [-6.65, -0.01, 3.6] |7.56|  executed [-6.28, 0.83, 4.4] |7.71| (err_cm 7.712)  diff 1.22 cm
    kinc   scale 1        tried ['1:None']  J miss + scale*shift [-4.4, 6.68, 2.54] |8.40|  executed [-4.35, 7.3, 3.45] |9.17| (err_cm 9.173)  diff 1.10 cm
    modelf scale 1        tried ['1:None']  J miss + scale*shift [-3.6, 1.59, -1.4] |4.18|  executed [-2.77, 0.91, 1.05] |3.10| (err_cm 3.097)  diff 2.67 cm
  r0 TabletopGraspMP_L4_e8: g [-8.959, 6.112, 3.321] pred [0.182, -3.147, 0.098] c_t [-2.199, -6.442, 1.323] (n 14) | J miss [-6.706, 3.487, 4.65] |8.87| | pred vs shifts.jsonl 0.0026 cm
    kin    scale fallback tried ['1:IK_FAIL', '2/3:IK_FAIL', '1/3:IK_FAIL']  J miss + scale*shift [-6.71, 3.49, 4.65] |8.87|  executed [-6.71, 3.49, 4.65] |8.87| (err_cm 8.877)  diff 0.00 cm
    kinc   scale fallback tried ['1:IK_FAIL', '2/3:IK_FAIL', '1/3:IK_FAIL']  J miss + scale*shift [-6.71, 3.49, 4.65] |8.87|  executed [-6.71, 3.49, 4.65] |8.87| (err_cm 8.877)  diff 0.00 cm
    modelf scale fallback tried ['1:IK_FAIL', '2/3:IK_FAIL', '1/3:IK_FAIL']  J miss + scale*shift [-6.71, 3.49, 4.65] |8.87|  executed [-6.71, 3.49, 4.65] |8.87| (err_cm 8.877)  diff 0.00 cm
  |executed - (J miss + scale*shift)| median per arm (cm): kin 0.61 (n 2), kinc 0.55 (n 2), modelf 1.34 (n 2)
```

### 开发局 `table2.txt`

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/precision_20261124/scalebfm  repeats 4, episodes 80, routed episode-repeats 44, layouts 9, falls in routed 0
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 17 / 17  J 40 / 30  replan 39 / 30  const 40 / 30  model 40 / 31  sham 40 / 29  oracle 40 / 31  kin 38 / 30  kinc 40 / 32  modelf 40 / 29

routes with every arm's row: 44; fallback to J's plan: replan 0 (0 %), const 2 (5 %), model 1 (2 %), sham 0 (0 %), oracle 1 (2 %), kin 1 (2 %), kinc 4 (9 %), modelf 1 (2 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 44 2/3 0 1/3 0 fallback 0; const 1 28 2/3 4 1/3 10 fallback 2; model 1 31 2/3 3 1/3 9 fallback 1; sham 1 36 2/3 3 1/3 5 fallback 0; oracle 1 36 2/3 4 1/3 3 fallback 1; kin 1 43 2/3 0 1/3 0 fallback 1; kinc 1 27 2/3 6 1/3 7 fallback 4; modelf 1 33 2/3 7 1/3 3 fallback 1
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0

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
  model minus const: median +0.33 [-0.36, +0.77] cm, model closer in 14/44 episodes, layouts 2/9 (worse 7, sign p 0.18)
    seen by the model's data: none
    never seen: 44 episodes, model minus const median +0.33 [-0.36, +0.77], layouts closer 2/9 (sign p 0.18)
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
  model minus const: median -0.04 [-0.12, +0.00] cm, model closer in 30/44 episodes, layouts 5/9 (worse 4, sign p 1)
    seen by the model's data: none
    never seen: 44 episodes, model minus const median -0.04 [-0.12, +0.00], layouts closer 5/9 (sign p 1)
  modelf minus const: median -0.02 [-0.06, +0.04] cm, modelf closer in 25/44 episodes, layouts 4/9 (worse 5, sign p 1)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus const median -0.02 [-0.06, +0.04], layouts closer 4/9 (sign p 1)
  modelf minus kinc: median -0.03 [-0.09, +0.04] cm, modelf closer in 24/44 episodes, layouts 6/9 (worse 3, sign p 0.508)
    seen by the model's data: none
    never seen: 44 episodes, modelf minus kinc median -0.03 [-0.09, +0.04], layouts closer 6/9 (sign p 0.508)

routes: successes within 45 s / official limit
  J 40 / 30  replan 39 / 30  const 40 / 30  model 40 / 31  sham 40 / 29  oracle 40 / 31  kin 38 / 30  kinc 40 / 32  modelf 40 / 29
```

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/precision_20261124/sonic  repeats 4, episodes 80, routed episode-repeats 55, layouts 15, falls in routed 9
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 13 / 13  J 36 / 26  replan 36 / 28  const 33 / 21  model 34 / 21  sham 34 / 25  oracle 36 / 26  kin 35 / 25  kinc 38 / 26  modelf 34 / 26

routes with every arm's row: 55; fallback to J's plan: replan 0 (0 %), const 7 (13 %), model 4 (7 %), sham 3 (5 %), oracle 8 (15 %), kin 8 (15 %), kinc 8 (15 %), modelf 6 (11 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 55 2/3 0 1/3 0 fallback 0; const 1 36 2/3 4 1/3 8 fallback 7; model 1 36 2/3 9 1/3 6 fallback 4; sham 1 44 2/3 6 1/3 2 fallback 3; oracle 1 39 2/3 3 1/3 5 fallback 8; kin 1 41 2/3 6 1/3 0 fallback 8; kinc 1 36 2/3 8 1/3 3 fallback 8; modelf 1 42 2/3 6 1/3 1 fallback 6
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0

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
  model minus const: median +1.05 [+0.00, +3.21] cm, model closer in 19/55 episodes, layouts 3/15 (worse 11, sign p 0.0574)
    seen by the model's data: none
    never seen: 55 episodes, model minus const median +1.05 [+0.00, +3.21], layouts closer 3/15 (sign p 0.0574)
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
  model minus const: median -0.07 [-0.56, +0.15] cm, model closer in 28/54 episodes, layouts 8/15 (worse 6, sign p 0.791)
    seen by the model's data: none
    never seen: 54 episodes, model minus const median -0.07 [-0.56, +0.15], layouts closer 8/15 (sign p 0.791)
  modelf minus const: median +0.00 [-0.21, +0.21] cm, modelf closer in 26/54 episodes, layouts 6/15 (worse 8, sign p 0.791)
    seen by the model's data: none
    never seen: 54 episodes, modelf minus const median +0.00 [-0.21, +0.21], layouts closer 6/15 (sign p 0.791)
  modelf minus kinc: median +0.00 [-0.01, +0.26] cm, modelf closer in 23/54 episodes, layouts 5/15 (worse 9, sign p 0.424)
    seen by the model's data: none
    never seen: 54 episodes, modelf minus kinc median +0.00 [-0.01, +0.26], layouts closer 5/15 (sign p 0.424)

routes: successes within 45 s / official limit
  J 35 / 25  replan 35 / 27  const 32 / 20  model 33 / 20  sham 33 / 24  oracle 35 / 25  kin 34 / 24  kinc 37 / 25  modelf 33 / 25
```

开发局的流程判据，三条都没触发：
- 新挪法的 24 次执行都是 0 failed processes；
- 退回 J 最多 15 %（SONIC 的 kin、kinc）；
- 站位抖动和发令时刻状态和 J 不同的局都是 0。

所以按登记接着跑了测试局。

### 测试局 `table2.txt`

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/precision_20261124/scalebfm  repeats 4, episodes 120, routed episode-repeats 112, layouts 18, falls in routed 0
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 64 / 64  J 119 / 101  replan 119 / 106  const 112 / 98  model 115 / 98  sham 116 / 91  oracle 116 / 94  kin 120 / 95  kinc 110 / 95  modelf 113 / 99

routes with every arm's row: 112; fallback to J's plan: replan 0 (0 %), const 1 (1 %), model 0 (0 %), sham 1 (1 %), oracle 1 (1 %), kin 3 (3 %), kinc 1 (1 %), modelf 1 (1 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 112 2/3 0 1/3 0 fallback 0; const 1 83 2/3 15 1/3 13 fallback 1; model 1 86 2/3 15 1/3 11 fallback 0; sham 1 100 2/3 8 1/3 3 fallback 1; oracle 1 99 2/3 7 1/3 5 fallback 1; kin 1 103 2/3 0 1/3 6 fallback 3; kinc 1 75 2/3 20 1/3 16 fallback 1; modelf 1 90 2/3 15 1/3 6 fallback 1
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0

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
  model minus const: median +0.69 [+0.23, +1.14] cm, model closer in 36/112 episodes, layouts 7/18 (worse 11, sign p 0.481)
    seen by the model's data: 92 episodes, model minus const median +0.59 [+0.10, +1.14], layouts closer 7/15 (sign p 1)
    never seen: 20 episodes, model minus const median +1.10 [+1.02, +1.14], layouts closer 0/3 (sign p 0.25)
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
  model minus const: median -0.08 [-0.12, -0.05] cm, model closer in 77/112 episodes, layouts 10/18 (worse 8, sign p 0.815)
    seen by the model's data: 92 episodes, model minus const median -0.09 [-0.19, +0.00], layouts closer 7/15 (sign p 1)
    never seen: 20 episodes, model minus const median -0.08 [-0.13, -0.07], layouts closer 3/3 (sign p 0.25)
  modelf minus const: median -0.00 [-0.04, +0.05] cm, modelf closer in 57/112 episodes, layouts 10/18 (worse 8, sign p 0.815)
    seen by the model's data: 92 episodes, modelf minus const median -0.00 [-0.05, +0.07], layouts closer 8/15 (sign p 1)
    never seen: 20 episodes, modelf minus const median -0.03 [-0.07, +0.01], layouts closer 2/3 (sign p 1)
  modelf minus kinc: median -0.04 [-0.12, +0.03] cm, modelf closer in 65/112 episodes, layouts 11/18 (worse 7, sign p 0.481)
    seen by the model's data: 92 episodes, modelf minus kinc median -0.04 [-0.14, +0.05], layouts closer 9/15 (sign p 0.607)
    never seen: 20 episodes, modelf minus kinc median -0.03 [-0.16, +0.04], layouts closer 2/3 (sign p 1)

routes: successes within 45 s / official limit
  J 111 / 93  replan 111 / 98  const 104 / 90  model 107 / 90  sham 108 / 83  oracle 108 / 86  kin 112 / 87  kinc 102 / 87  modelf 105 / 91
```

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/precision_20261124/sonic  repeats 4, episodes 120, routed episode-repeats 106, layouts 19, falls in routed 6
main metric, all episodes (the arms equal J where J has no plan): successes within 45 s / official limit
  A 39 / 39  J 72 / 60  replan 69 / 57  const 72 / 59  model 69 / 53  sham 66 / 51  oracle 70 / 50  kin 73 / 52  kinc 74 / 49  modelf 67 / 46

routes with every arm's row: 106; fallback to J's plan: replan 0 (0 %), const 3 (3 %), model 2 (2 %), sham 3 (3 %), oracle 8 (8 %), kin 0 (0 %), kinc 5 (5 %), modelf 4 (4 %)
shift scale used (1, 2/3, 1/3, fallback): replan 1 106 2/3 0 1/3 0 fallback 0; const 1 93 2/3 3 1/3 7 fallback 3; model 1 86 2/3 15 1/3 3 fallback 2; sham 1 79 2/3 12 1/3 12 fallback 3; oracle 1 73 2/3 12 1/3 13 fallback 8; kin 1 80 2/3 12 1/3 14 fallback 0; kinc 1 82 2/3 10 1/3 9 fallback 5; modelf 1 92 2/3 4 1/3 6 fallback 4
state at the command equal to J's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm), routed episode-repeats differing: replan 0, const 0, model 0, sham 0, oracle 0, kin 0, kinc 0, modelf 0

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
  model minus const: median +3.35 [+2.41, +5.01] cm, model closer in 22/106 episodes, layouts 1/19 (worse 18, sign p 7.63e-05)
    seen by the model's data: 94 episodes, model minus const median +3.39 [+2.38, +5.06], layouts closer 1/16 (sign p 0.000519)
    never seen: 12 episodes, model minus const median +2.81 [+2.04, +8.16], layouts closer 0/3 (sign p 0.25)
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
  model minus const: median +0.32 [+0.00, +0.46] cm, model closer in 42/106 episodes, layouts 8/19 (worse 11, sign p 0.648)
    seen by the model's data: 94 episodes, model minus const median +0.33 [-0.07, +0.48], layouts closer 7/16 (sign p 0.804)
    never seen: 12 episodes, model minus const median +0.24 [+0.01, +1.81], layouts closer 1/3 (sign p 1)
  modelf minus const: median +0.12 [-0.13, +0.34] cm, modelf closer in 49/106 episodes, layouts 10/19 (worse 9, sign p 1)
    seen by the model's data: 94 episodes, modelf minus const median +0.18 [-0.10, +0.39], layouts closer 7/16 (sign p 0.804)
    never seen: 12 episodes, modelf minus const median -0.43 [-0.66, -0.11], layouts closer 3/3 (sign p 0.25)
  modelf minus kinc: median +0.00 [-0.21, +0.22] cm, modelf closer in 52/106 episodes, layouts 11/19 (worse 8, sign p 0.648)
    seen by the model's data: 94 episodes, modelf minus kinc median +0.04 [-0.13, +0.30], layouts closer 8/16 (sign p 1)
    never seen: 12 episodes, modelf minus kinc median -0.38 [-0.91, -0.22], layouts closer 3/3 (sign p 0.25)

routes: successes within 45 s / official limit
  J 68 / 56  replan 65 / 53  const 68 / 55  model 65 / 49  sham 62 / 47  oracle 66 / 46  kin 69 / 48  kinc 70 / 45  modelf 63 / 42
```

旧数据没被改动：
- 重算旧的 `table`，开发局和测试局两个控制器都和原来的 `table.txt` 逐字节一致；
- 新三组用的 pred 和旧 `shifts.jsonl` 最大差 0.005 cm。

完整记录在 `outputs/simple_{dev_20261124,eval_20261112}/precision_20261124/<控制器>/`（`r0–r3/` 下的新行、`shifts2.jsonl`、`run2.log`、`table2.txt`）。

## 意外

预测对照（测试局）：

| | 预测 | 结果 | |
|---|---|---|---|
| ScaleBFM kin | 3.5–6，比 J 差 | 4.30（J 3.97） | 中 |
| ScaleBFM kinc | 1.0–2.0 | 1.94 | 中（靠上沿） |
| ScaleBFM modelf | 0.8–1.8 | 1.28 | 中 |
| ScaleBFM modelf − const | −1.0 到 −0.1，猜 −0.5 | −0.41 | 中 |
| ScaleBFM modelf − kinc | −0.9 到 0，猜 −0.4 | −0.56 | 中 |
| ScaleBFM 两条都过（估约六成） | — | 过 | — |
| 开发局 ScaleBFM modelf − const | −1.5 到 0 | −0.38 | 中 |
| SONIC kin | 6–12，比 J 差 | 9.71（J 5.92） | 中 |
| SONIC kinc | 3–7 | 6.82 | 中（靠上沿） |
| SONIC modelf | 3–7 | 3.87 | 中 |
| SONIC modelf − const | −1.5 到 +1.0，多半打平 | −0.40，按定义不算赢 | 中 |
| SONIC 两条都过（估约一成半） | — | 没过 | — |
| 退回 J | 每组不超过 15 % | 新三组最多 3 %（ScaleBFM）、5 %（SONIC） | 中 |
| 主指标（有路线的局，45 秒内），各组和 J 差 | 不超过 10 % | ScaleBFM kinc 102 / 111（−8 %）、modelf 105；SONIC modelf 63 / 68（−7 %） | 中 |

没想到的：

1. **SONIC 上 kinc 比 const 差很多（6.82 对 4.63 cm）。** g + c_t 的平均就等于 const，差别全在逐局那部分：
   - 按局减掉 g 以后，停偏反而变大。说明在 SONIC 上，g 和跟踪误差逐局反向相关：参考离目标越远，手就越往目标那边偏回来。
   - 发令时刻重建的参考，并不是 SONIC 实际在跟的东西。原因没查；一个可能是命令发出后 SONIC 的骨盆还在动。
   - ScaleBFM 上也有一点（kinc 1.94 对 const 1.74），但弱得多。
   - 因此"手停在哪 = g + 跟踪误差"这个拆法，对 SONIC 不太成立。
2. **SONIC 挪过目标的各组，官方时限内的成功都明显比 J 少**，按有路线的局：

   | J | const | model | oracle | kin | kinc | modelf |
   |---|---|---|---|---|---|---|
   | 56 | 55 | 49 | 46 | 48 | 45 | 42 |

   - 45 秒内差得少（J 68、modelf 63）。
   - oracle 也掉，说明挪预抓取点本身让 SONIC 后面变慢，和挪得准不准无关。原因没查。
   - 登记只预测了 45 秒内。
3. **执行出来的 modelf − const（−0.41）比复核离线外推（−0.58）小。**
   - 缩放重试削掉了一部分：modelf 全量 90 / 112，const 全量 83 / 112。
   - 试跑里 modelf 的"执行对外推"差也最大：ScaleBFM 0.69 cm，SONIC 1.34 cm 中位数。
4. **预抓取更准，entry 和成功都没跟着变好**（ScaleBFM，测试局，有路线的局）：
   - modelf 的 entry 停偏比 J 差 +0.28 cm；
   - 45 秒内成功 105，J 111、const 104。

   和上一次一样：TabletopGrasp 的 entry 从实测的手重新瞄，这个任务几乎不奖励预抓取那一下的精度。

## 结论与下一步

**1. ScaleBFM 落在登记的"modelf 同时赢 const 和 kinc"一侧。** 证据是测试局 ScaleBFM 表的这几行：
- `modelf minus const: median -0.41 [-0.71, -0.12] cm, modelf closer in 80/112 episodes, layouts 15/18 (worse 3, sign p 0.00754)`
- `modelf minus kinc: median -0.56 [-0.96, -0.24] cm, ... layouts 15/18 (worse 3, sign p 0.00754)`
- 停偏中位数：J 3.97、kin 4.30、kinc 1.94、const 1.74、modelf 1.28、oracle 0.61 cm。

开发局方向一致：对 kinc 赢（9 / 9）；对 const −0.38，7 / 9，不显著。

在这个基准上可以说：**对 ScaleBFM，按自身模型逐局预测去修正一次命令，比固定标定、比"运动学加常数"都准**。上一次的"常数就够"是 model 那一组漏了运动学那段差造成的。

**2. SONIC 落在登记的三条之外。**
- 对 kinc 赢：−2.33 [−2.86, −1.67]，16 / 19，p 0.0044。
- 对 const：区间 [−1.10, −0.08] 在 0 以下，但布局 12 / 19，符号检验 p 0.359，按"赢"的定义不算赢。

保守读法：**SONIC 上模型没有赢过固定标定，记为打平（略偏模型）**，不算贡献。另外，"手停在哪 = g + 跟踪误差"这个拆法在 SONIC 上本身就不太成立（意外 1），kinc 在 SONIC 上是个偏弱的对手。

**3. 精度没有变成成功**（意外 4），SONIC 官方时限内还掉了（意外 2）。论文里"一次命令的精度"要用停偏当指标来证明，TabletopGrasp 的成功数证明不了它。

**证据强度**：
- 一次运行，每组每局一次；
- 测试局 112（ScaleBFM）/ 106（SONIC）个有路线的局次，来自 28 / 27 个位置 × 4 次抖动，18 / 19 个布局；
- 开发局 44 / 55 个局次，9 / 15 个布局。

**其他可能的解释**：
1. **常数是用少量真实局估的。**
   - const 和 c_t 在 ScaleBFM 上只用了 11 个开发局。c_t 前后、上下的 sd 约 1.78 cm，均值的标准误约 0.54 cm，和 0.41 cm 的增益同一量级。
   - 用多得多的真实局估常数，差距可能缩小或消失。那样模型的长处就是"不需要真实局"（数据效率，和 BiGym 的复核结论一样），而不是逐局预测更准。
   - 可以离线查：在测试局自己身上按留一布局估 const 和 c_t（每次约 100 局），再比。
2. **抖动几乎没传到手臂上**（上一次复核：同一位置伸手向量只变 0.6 / 1.4 / 0.1 cm）。
   - 这次模型能用上的逐局变化很小，增益主要来自它把固定的那部分估得更准。
   - 伸手几何真的变大时，结果可能更有利于模型，也可能模型自己的误差变大。
3. **这是第二次尝试。** model 那一组是在看到它输了以后修的，有"改到赢为止"的风险。
   - 缓解：这次的改动是把一个发命令时就能算出的运动学项补上（不是调参数）；登记在执行之前；用的是同一批局；还加了最强的不用模型的逐局对手 kinc。

**四类陷阱**：
- **benchmark 选择**：只有 TabletopGrasp，因为只有它的伸手被障碍挡、需要规划（上一次就定了），不是这次挑的。用的是上一次登记的同一批抖动局。
- **数据泄漏**：
  - const、c_t 都来自开发局的 JR 行（3、4 级，不抖动），在 19:57Z 写好，早于任何一次执行。
  - 自身模型是空地上采的数据训的，没见过 SIMPLE 的局。
  - "见过 / 没见过"按登记的划分照报：ScaleBFM 没见过的 3 个布局 modelf − const −0.30 [−0.49, +0.36]，3 / 3，样本小。
- **指标误用**：
  - 停偏的定义、配对、按布局重抽、符号检验都和登记的一样。
  - entry、成功数、官方时限都照报了，其中对模型不利的（意外 2、4）也写了。
- **事后挑选**：
  - 一次运行，没有挑重复；开发局和测试局都报了；SONIC 不利的结果照报。
  - 风险是"第二次尝试"（解释 3）。预测是在看过复核的离线外推之后写的，这一点登记里写明了。

**下一步（按判据）**：
1. **S 改成完整预测**：挪 −(g + p)，用这次的缩放重试（全量、2/3、1/3，都不行退回 J）。另登记，在 SIMPLE 官方局（不抖动）上重跑，两个控制器都跑，主表的 S 用这个版本。
2. **请复核离线查"常数用多少真实局估"**（解释 1）。如果常数用更多真实局估就追平，结论改写成"数据效率"。
3. **SONIC 上 g 和跟踪误差反向相关**（意外 1）先记下，离线查骨盆在发令后是否还在动。优先级低。

## 独立复核

2026-11-25，独立的 agent。它拿到的是：
- 登记的各节、预测修订、结果、结论；"意外"一节没给，所以它说结论里引用的意外 1、2、4 找不到；
- 4090 上的逐局记录和代码路径；
- 上一次的条目和复核。

原文（英文）：

> ## Findings
>
> 1. **All the numbers reproduce. The registered win for ScaleBFM is real and robust.**
>    - Rerunning `precision.py table2` on the current rows gives `table2.txt` byte for byte in all four controller dirs. The tables pasted in the entry are identical to those files, and the old `table` is also byte-identical.
>    - My own reimplementation (loaders, layouts, bootstrap, exact binomial sign test) matches every pregrasp median, paired median, episode and layout count, sign p, seen/unseen split, scale and fallback count, success count (all episodes and routed only) and fall count.
>    - The one difference is the dev ScaleBFM bootstrap interval for modelf − const: mine is [−1.40, +0.00], the table's [−1.62, +0.00]. The layout partition is the same; only the label order fed to the random draw differs. It is Monte Carlo noise; the upper bound of 0.00 is stable and the test intervals match exactly.
>    - **ScaleBFM test, settled miss medians (cm):** J 3.97, kin 4.30, kinc 1.94, const 1.74, modelf 1.28, oracle 0.61.
>    - **ScaleBFM test, paired:**
>      - modelf − const: −0.41 [−0.71, −0.12], closer in 80/112 episodes, 15/18 layouts, p 0.0075.
>      - modelf − kinc: −0.56 [−0.96, −0.24], closer in 81/112, 15/18 layouts, p 0.0075.
>    - **SONIC test, paired:**
>      - modelf − const: −0.40 [−1.10, −0.08], 12/19 layouts, p 0.359.
>      - modelf − kinc: −2.33 [−2.86, −1.67], 16/19 layouts, p 0.0044.
>    - **Fallbacks to J's plan, test:** ScaleBFM kin 3, kinc 1, modelf 1; SONIC kin 0, kinc 5, modelf 4. Dev maximum is 8/55 (15 %), SONIC kin and kinc.
>    - **Routed successes, within 45 s / official limit:**
>      - ScaleBFM: J 111/93, const 104/90, kinc 102/87, modelf 105/91.
>      - SONIC: J 68/56, const 68/55, kinc 70/45, modelf 63/42.
>    - **Robustness of the ScaleBFM win over const:**
>      - Per repeat: −0.31, −0.47, −0.51, −0.16.
>      - Where both arms ran at full scale: −0.61 [−1.57, −0.37], 12/13 layouts (n 83).
>      - So it is not an artefact of the scale retries.
>
> 2. **Integrity checks pass on all 317 episode-repeats (test 112 + 106, dev 44 + 55).**
>    - **g and p:** recomputed on CPU with `s_check.reference`, `model_input` and `VP.predict`. They match `shifts2.jsonl` within 0.0005 cm. p matches the old model arm's `pred` in `shifts.jsonl` within 0.005 cm (that file is rounded to 0.01).
>    - **Shifts:** kin, kinc and modelf equal −g, −(g + c_t) and −(g + p) within 0.0005 cm.
>    - **Planner inputs:** every prep file, at every scale, equals J's prep plus `from_head(k·shift)` exactly, with no other field changed.
>    - **Plans sent:**
>      - Each arm row's `joint_src` is the scale directory that `pick2` chose.
>      - Fallbacks ran J's own plan, and each state's `plan_src` equals J's row; 0 mismatches.
>      - The film reference of each new-arm run matches the reference rebuilt from its plan within 0.11 cm (ScaleBFM) and 0.16 cm (SONIC).
>    - **Start state:** jitter, jitter_rep, walk, walk_err_cm, stand, bel, odo_cm and pregrasp S0 / S3 / W / J / t_cmd / goal_mm equal J's in every row; 0 differ.
>    - **Old data untouched:** old rows were last written at 19:31:51Z or earlier; new rows were written 20:05–20:54Z. Each tree and controller has 12/12 new-arm runs with 0 failed processes.
>    - **c_t provenance:**
>      - Recomputed exactly from the dev bench JR rows: 20 rows, levels 3–4, no jitter key, file times 14:03Z / 14:04Z. These are the same 11 / 15 episodes as `const_*.json`.
>      - The ctrack files were written at 19:57:12Z / 19:57:17Z. That is after the registration (19:49:38Z), before the trial (19:57:45–19:59:08Z) and before dev (20:00:01Z).
>      - Their md5 equals the `asrun2/` copies, and the test arms used the full dev c_t throughout.
>
> 3. **Bug: on SONIC dev, the "leave-one-layout" constants are not leave-one-layout.**
>    - `precision.py` lines 253–255 (c_t in `shifts2`) and lines 192–195 (const in `shifts`, the earlier run) leave out dev episodes whose *look goal* lies within 2 cm of this episode's look goal.
>    - ScaleBFM: the episode's own position was dropped in 44/44 (distance median 0.08 cm).
>    - SONIC: the own position is in the c_t set in 49/55, but was dropped in only 19. SONIC's look goal for the same position moves 2.0–13.7 cm (median 2.2) between the unjittered bench and the jittered repeats.
>    - So in 30/49 SONIC dev episode-repeats, const and c_t include the episode's own position. This affects dev and SONIC only and favours const and kinc; it does not touch the test result.
>    - Fix: leave out by position name or layout id, not by goal distance.
>
> 4. **Main problem: kinc is not "the strongest model-free per-episode baseline". A correctly framed one ties modelf on ScaleBFM (offline).**
>    - **How g is defined.** It carries the reference onto the *true pelvis* at the command (`s_check.py` line 165, `Ra @ jw + pa`). The motor's reference has its own root (`ref0[0:7]`), held through the reach. The reach end placed by that root, `er = jw − goal`, is also known at the command and needs no model.
>    - **Where the hand actually lands.** The measured jaw is much nearer the reference-root end than the carried end:
>
>      | | jaw to reference-root end | jaw to carried end |
>      |---|---|---|
>      | ScaleBFM test | 1.88 cm | 4.65 cm |
>      | SONIC test | 4.31 cm | 11.14 cm |
>
>    - **A model-free arm in that frame.** kinc_rr shifts by −(er + c_rr), where c_rr is the mean of (jaw − er) on the same 11 dev JR rows. For ScaleBFM, c_rr = [−1.62, −0.65, +0.80] with sd [0.97, 0.66, 0.55], against c_t's sd [1.78, 0.55, 1.78].
>    - **Offline results, ScaleBFM:**
>
>      | Comparison | Median | 95 % interval | Layouts closer | Sign p |
>      |---|---|---|---|---|
>      | test, modelf − kinc_rr (both extrapolated; 1.09 vs 1.10 cm) | +0.09 | [−0.06, +0.26] | 8/18 | 0.815 |
>      | test, executed modelf − extrapolated kinc_rr | +0.23 | [−0.06, +0.48] | – | – |
>      | dev, modelf − kinc_rr | +0.02 | [−0.51, +0.36] | 5/9 | – |
>
>      With c_rr taken from only 3 random dev episodes, modelf − kinc_rr is +0.02 (5–95 %: −0.40 to +0.16).
>    - **Offline result, SONIC:** modelf − kinc_rr −0.61 [−1.47, −0.11], 10/19 layouts, p 1. Not a win.
>    - **Why the model ties.** The per-episode part of p is mostly the frame term u = (reference-root end) − (carried end).
>      - On ScaleBFM, u has sd [1.09, 0.84, 1.01] cm and corr(p, u) is 0.79 / 0.85 / 0.89.
>      - After removing u, the model's correlation with the remaining tracking error drops to 0.29 / 0.36 / 0.76.
>      - The leftover spread is about the same either way: sd(t − u) = [0.90, 0.74, 0.48] vs sd(t − p) = [1.01, 0.72, 0.37].
>
> 5. **Alternative (a), small-sample constants: not the explanation by itself, but a model-free fit on more real episodes ties.**
>    - **Setup:** constants re-estimated leave-one-layout on the test episodes themselves (92–108 per estimate). The const barely moves ([−3.01, 0.22, 3.44] vs dev [−3.17, 0.10, 3.36]); c_t moves somewhat ([−3.43, 1.89, 2.94] vs dev [−3.88, 1.64, 3.37]).
>    - **Extrapolated medians:** const_LOL 1.90 (worse than the dev const's 1.82), kinc_LOL 1.50, modelf 1.09.
>      - modelf − const_LOL: −0.62 [−0.90, −0.40], 18/18 layouts.
>      - modelf − kinc_LOL: −0.43 [−0.72, −0.29], 17/18 layouts.
>    - **Anchored on the executed rows** (the executed const or kinc miss moved by k·(c_LOL − c_dev)):
>      - modelf − const_LOL: −0.44 [−0.75, −0.13], 15/18, p 0.0075.
>      - modelf − kinc_LOL: −0.41 [−0.92, −0.02], 14/18, p 0.031.
>      - So more real episodes for the registered constants do **not** close the gap.
>    - **But a model-free affine fit closes it:**
>      - miss = a + B·g, fit leave-one-layout on about 100 real episodes, gives 1.36; modelf − affine is −0.06 [−0.39, +0.02], 13/18, p 0.096. Not a win.
>      - The same fit on er gives 0.92; modelf − affine(er) is +0.19 [+0.04, +0.41].
>    - **How well extrapolation matched execution, ScaleBFM test:**
>      - |executed − (J miss + k·shift)| median is 0.28 (kin), 0.47 (modelf), 0.62 (kinc) and 0.64 cm (const); the 90th percentile is 0.65–1.99.
>      - Paired modelf − const: executed −0.41, extrapolated −0.29 at the scale used and −0.58 at full scale. Paired modelf − kinc: −0.56 executed against −0.42 / −0.74.
>    - **SONIC:** the same mismatch has a median of 2.1–3.1 cm and a 90th percentile of 8.5–14.8 cm, so SONIC offline numbers are not trustworthy beyond their sign.
>
> 6. **Alternative (b), SONIC g against tracking error: the anti-correlation is real, but the moving pelvis is not the cause.**
>    - **Confirmed:** on SONIC test, corr(g, t) per axis is −0.10 / −0.70 / −0.72 (within layouts −0.05 / −0.65 / −0.73), and the slope of t on g is −0.27 / −0.90 / −0.81.
>    - **ScaleBFM shows it too:** −0.32 / −0.52 / +0.05 (within layouts −0.51 / −0.77 / −0.23). That is why kinc is no better than const there (+0.04 [−0.16, +0.17], 7/18).
>    - **The pelvis does move a lot after the command on SONIC:**
>      - median 10.6 cm (mean 9.2 cm backward) and 5.2° of yaw, with the settle 8.2 s after the command (ScaleBFM: 1.9 cm, 1.9°);
>      - this drift drives t forward and back (corr 0.83);
>      - but it is nearly uncorrelated with g (0.23 / 0.18 / −0.04), and so is the part of t it causes (0.19 / −0.06 / −0.08).
>    - **The actual cause:** g = er − u and t = (jaw − er) + u share the frame term u with opposite signs.
>      - On SONIC, u has sd [3.71, 6.05, 3.63] cm (mean −9.72 cm sideways), corr(t, u) is 0.20 / 0.63 / 0.91, and corr(g, reference root − pelvis) is −0.85 / −0.85 / −0.75.
>      - The hand goes where the motor's reference root puts it, not where the true pelvis would carry it.
>
> 7. **Post-hoc selection.**
>    - **What is principled:**
>      - Adding g makes the correction target the same as the metric's (the goal). It is not a parameter search.
>      - It was registered before any run, on the same episodes, with the same retry rule.
>      - The trial ran after the registration, and dev started 53 s after the trial with nothing changed.
>    - **What is not clean:**
>      - The fix and its size were found on these same test episodes: the previous reviewer's offline −0.58 (I get −0.58 extrapolated). The predictions were written after seeing that.
>      - kinc came from the same offline analysis, where its loss (−0.48, 18/18) was already known, and it is mis-framed (Finding 4).
>      - So this is a re-execution of a known result, not an independent confirmation. The bar it cleared was chosen on the same data, and it was low.
>    - **Unseen layouts are underpowered:** 3 layouts, 20 episodes, modelf − const −0.30 [−0.49, +0.36], p 0.25.
>
> 8. **Metric.** The pregrasp metric is used as registered and correctly paired.
>    - Better precision does not become success:
>      - ScaleBFM routed successes within 45 s: J 111, modelf 105.
>      - ScaleBFM entry miss, modelf − J: +0.28 [+0.16, +0.45].
>      - SONIC within the official limit: J 56, modelf 42.
>    - The entry cites 意外 1, 2 and 4, but it has no 意外 section.
>
> ## Verdicts
>
> | Conclusion | Verdict | Reason |
> |---|---|---|
> | 1. ScaleBFM meets the registered "modelf beats both const and kinc" | Supports | Every number reproduces; holds per repeat, at full scale only, and with another bootstrap seed |
> | 1. "…more accurate than a fixed calibration" | Supports (this benchmark, these episodes) | Loses neither to the dev const nor to one estimated from about 100 test episodes (anchored −0.44, 15/18). Caveats: the test set was reused; unseen layouts are underpowered |
> | 1. "…more accurate than kinematics plus a constant" | Does not support | True only against the pelvis-carried kinc. Kinematics in the motor's own reference frame plus a constant from the same 11 dev episodes ties offline (+0.09 [−0.06, +0.26], 8/18), even with a constant from 3 episodes. Offline only, but enough to stop the claim |
> | 1. The earlier "a constant suffices" came from omitting the kinematic gap | Supports | Same episodes: model − const +0.69 becomes modelf − const −0.41 |
> | 1. Dev agrees in direction | Supports | Reproduced: kinc 9/9; const −0.38, 7/9 |
> | 2. SONIC falls outside the three outcomes; conservative tie, no model credit | Supports | 12/19 layouts, p 0.359; also ties the reference-frame kinematic baseline (10/19); SONIC extrapolation is unreliable |
> | 2. "hand = g + tracking" breaks on SONIC; kinc is a weak opponent | Supports, with a corrected mechanism | Anti-correlation confirmed. The cause is the frame term u shared by g and t, not pelvis drift, and it is present on ScaleBFM too |
> | 3. Precision does not become success; SONIC official-limit successes drop; precision must be shown with the miss | Supports | Counts above |
> | Evidence strength | Supports | Counts are correct |
> | Alternative 1 (constants from few episodes; would make it "data efficiency") | Does not support as stated | More real episodes for const and c_t do not close the gap; a correctly framed model-free baseline closes it with about 3 episodes, so data efficiency is not much of an edge either |
> | Alternative 3 / pitfall: post-hoc change was principled | Supports, with a caveat | Principled and registered, but derived on the same test set against a comparator chosen there |
> | Pitfall: leakage — c_t only from dev; dev uses leave-one-layout | Test: supports. SONIC dev: does not support | The bug in Finding 3 |
> | Pitfall: no metric misuse | Supports | – |
> | Next step 1 (switch S to the full prediction; main table uses it as the model's contribution) | Insufficient evidence | Run the reference-frame kinematic arm first, or the table credits kinematics to the model |
> | Next step 3 (is the pelvis still moving after the command?) | Answered: not the cause | Finding 6 |
>
> ## How I verified it
>
> - **Tables:**
>   - Reran `precision.py table2` and `table` into a scratch dir and diffed them against the stored tables. Also diffed the entry's pasted blocks against `table2.txt`.
>   - Wrote an independent reimplementation of the medians, layouts, bootstrap, sign test, splits, scales, fallbacks, successes and state equality.
> - **Integrity:**
>   - For every episode-repeat, recomputed g and p on CPU with `CUDA_VISIBLE_DEVICES=""` and the jrefwm model.
>   - Checked the prep files at all scales, `joint_src` against `pick2`, and each film's reference against the plan it should have sent.
>   - Checked state equality, file times, the md5 of the `asrun2/` copies against the Mac working tree, and the failed-process counts.
>   - Recomputed c_t from the dev bench rows and the `ctrack_state` dirs, and audited the leave-one-layout exclusions per episode.
> - **Alternatives:**
>   - Leave-one-layout constants on the test episodes, both extrapolated and anchored on the executed rows.
>   - Affine model-free fits.
>   - A reference-root kinematic baseline, with c_rr recomputed from the same dev rows.
>   - Correlations of g, t and u, and pelvis drift and yaw from the J films between the command and the settle.
> - **Housekeeping:** all read-only. The scratch dirs `/tmp/rv_prec2_k7q3` (remote) and `/tmp/rv_prec2_mac` (Mac) are deleted. No GPU was used, and no code, output or git state was touched.
>
> ## Experiments that could overturn the conclusion
>
> 1. **Run the reference-frame kinematic arm (primary).** On the same test episodes, add kinc_rr, shifted by −(er + c_rr).
>    - er: the jaw at J's last joint row under the motor's reference root at the command, i.e. FK of `st["ref0"]` with no carry onto the pelvis.
>    - c_rr: the mean of (jaw_mm − er) over the same 11 / 15 dev JR rows, frozen before the run.
>    - Also add kin_rr = −er. Keep the same retry and fallback rule.
>    - Register it: if modelf − kinc_rr's interval includes 0 or the sign test has p ≥ 0.05, "beats kinematics plus a constant" is overturned for ScaleBFM. The offline prediction is a tie (about +0.1 cm).
> 2. **Fresh episodes.** Use a new `--jitter-seed` and positions never used for diagnosis, with modelf, const, kinc and kinc_rr. The current test set was used to find the fix.
> 3. **Data-efficiency curve.** Fit a model-free affine model (miss = a + B·g, or a + B·er) on N = 1…100 real episodes. If N ≤ 5 already ties, the model's remaining edge is negligible.
> 4. **Make the reach geometry really vary.** Apply the stance offset after the table-edge clamp, or vary target position and pelvis height. Only if modelf beats kinc_rr there, and not just const, does the self model earn credit for its per-episode prediction.
> 5. **Model ablation.** Train or evaluate the model against the reference in its own root frame. If its residual (p − u) adds nothing over c_rr, what it learned is the frame offset, not tracking. Offline, the residual correlation is 0.29 / 0.36 / 0.76: small, and mostly in the vertical axis.
>
> The code in question is in `/Users/zed/code/loco/scripts/experiments/simple_eval_20261112/`:
> - `precision.py` lines 192–195 and 253–255: the goal-distance leave-one-out.
> - `precision.py` line 250: g.
> - `s_check.py` line 165: the carry onto the pelvis.

### 复核后的更正（2026-11-25）

数全部重算对上，流程核对全过。**但复核找到一个比"常数"更强的不用模型的对手，它让结论 1 的后半句不成立。**

1. **"比固定标定准"成立。** 换成在测试局自己身上用约 100 局按留一布局估的常数，以执行的行为基准，ScaleBFM 上 modelf 还是赢：−0.44 [−0.75, −0.13]，15 / 18。所以"其他解释 1"（常数局少）不成立。
2. **撤回"比运动学加常数准"。** kinc 的框架不对：
   - **g 的算法**：g 把参考按发令时刻的真实骨盆搬过去（`s_check.py` 第 165 行 `Ra @ jw + pa`）。
   - **电机实际跟的东西**：电机跟的是参考自己的根（`ref0[0:7]`）。手实际停的位置离"参考自己的根放出来的终点"（er）近得多：ScaleBFM 1.88 cm 对 4.65 cm，SONIC 4.31 对 11.14 cm。
   - **一个不用模型的对手**：按 er 再加一个开发局冻结的常数（kinc_rr），离线外推就和 modelf 打平：
     - ScaleBFM +0.09 [−0.06, +0.26]，8 / 18 个布局；常数只用 3 个开发局估也打平。
     - SONIC −0.61 [−1.47, −0.11]，10 / 19，不算赢。
   - **按逐局记录拟合的仿射对手也打平**：miss = a + B·g，按留一布局用约 100 个真实局拟合，ScaleBFM 上 −0.06，13 / 18，p 0.096。
   - 这些都是离线结果，要另登记、执行后才算。
3. **模型这次赢常数，靠的主要是一个运动学的框架项**：u = 参考根放出来的终点 − 搬到骨盆上的终点。
   - 模型的预测 p 和 u 的相关是 0.79 / 0.85 / 0.89。
   - 扣掉 u 以后，模型对剩下的跟踪误差的相关只有 0.29 / 0.36 / 0.76。
   - 也就是说，模型学到的主要是"把参考搬回电机自己的框架"，这件事不用模型就能算。
   - 这个框架差是我们的流程自己引进的：训练和检查都把参考搬到真实骨盆上（`jref_data.build`、`s_check.frames`）。
4. **另一个要查的问题（我补的，复核没提）**：
   - `s_check.frames` 用的是发令时刻的**真实骨盆**（`qpos0`，仿真里的值），g、kin、kinc、model、modelf 都经过它。
   - 实际机器人上只有自己的估计。这和"不用仿真真值"的规矩有冲突，从 S 那次起就是这样，之前没发现。
   - er 只用电机自己的参考根和开局那一眼的目标，不用真值。
5. **SONIC 上 g 和跟踪误差反向相关是真的，但原因不是骨盆在动。**
   - 发令后骨盆确实动了约 10.6 cm，但它和 g 几乎不相关。
   - 原因是 g 和跟踪误差共用框架项 u，符号相反；ScaleBFM 上也有。
   - 结论里"下一步 3"已有答案。
6. **SONIC 开发局的"留一布局"不是真的留一**：`precision.py` 第 192–195、253–255 行按目标距离 2 cm 排除，但 SONIC 抖动后同一位置的目标会移 2–14 cm。
   - 后果：SONIC 开发局的 49 个局次里，30 个的常数含了本局自己的位置。
   - 只影响开发局和 SONIC，偏向 const 和 kinc，不碰测试局。
   - 以后改成按位置名排除。
7. **事后挑选**：这次修正和它的大小，都是在同一批测试局上离线找到的，kinc 也是在那里选的。所以这是"重新执行一个已知结果"，不是独立确认，而且门槛定低了。
8. **对照判据的读法改为**：照登记，ScaleBFM 落在"两条都赢"一侧，这一条照报。但"SIMPLE 上模型算贡献"要等 kinc_rr 执行出来再说。
9. **下一步**：
   - 另登记 `selfmodel_precision3_20261125`：同一批抖动局上加 kin_rr（只去 er）和 kinc_rr（er 加开发局常数），核心是 modelf 对 kinc_rr。
   - `selfmodel_sfull_20261125` 在测试局开跑以前加同样两组，核心改成 SF 对 const、对 kinc_rr 都要赢（预测修订里写明）。
   - 官方局（不抖动）没被拿来做过 er 的分析，算半独立的一批，但位置和抖动局相同。
