# selfmodel_sfull_20261125：S 改成完整预测（运动学的差加模型），在 SIMPLE 官方测试局上和 A、J、常数逐局配对

- 日期：2026-11-25（设计、预测、判据；任何 S-full 运行以前写下，单独提交）
- 机器 / GPU：4090，ScaleBFM 在 1 号卡，SONIC 在 2、3 号卡，和前几次一样。跑之前查 nvidia-smi，0 号卡上有别人的进程，不碰。
- 代码版本：`selfwm` 加未提交的 `scripts/experiments/simple_eval_20261112/{precision.py, precision_run.sh}`，以及新的启动脚本。跑时抄进输出目录，md5 记在结果里。
- 配置：
  - SIMPLE 官方评测局，四个抓取任务各 30 局，感知栏，`sensor=rgbd`；
  - 站位不抖动；
  - ScaleBFM 和 SONIC（`REACH_SONIC_NETS=gpu`）都不改参数；
  - 自身模型是 selfmodel_jrefwm_20261124 的两份，不重训；
  - 不用仿真真值，不用伺服和查表。
- 关联：
  - [selfmodel_precision2_20261125](selfmodel_precision2_20261125.md)：抖动的测试局上，ScaleBFM 的完整预测（modelf）同时赢了常数（−0.41 cm，15 / 18 个布局）和"运动学的差加跟踪常数"（−0.56 cm，15 / 18）。SONIC 对常数打平，对后者赢。它的判据写明：ScaleBFM 两条都赢，就把 S 改成完整预测，另登记，在 SIMPLE 官方局（不抖动）上重跑，主表的 S 用这个版本。
  - [selfmodel_s_20261124](selfmodel_s_20261124.md)：原来的 S（只开终点检查，挪 −p，挪不动就放弃），A、J、S 的数都在那里。
  - [selfmodel_s_jr_20261124](selfmodel_s_jr_20261124.md)：测试局上 J 带录像重跑（JR），和 J 逐局相同，发令时刻的状态从它来。
  - 用户 2026-11-24：主论断是一次命令的精度；主表只跑 A、J、S。

## 假设

精度实验在抖动的局上测出来的"完整预测比常数准"，在 SIMPLE 官方局（不抖动）上也成立。并且把 S 改成完整预测、挪不动时退回 J（而不是放弃），能去掉原来 S 丢分的主要原因：原 S 在 TabletopGrasp 上比 J 少 7 局，几乎都是 IK 无解就放弃造成的。

## 做法

### S-full（下面记作 SF）

在 J 的基础上，发令以前只改预抓取的目标，记号同 [selfmodel_precision2_20261125](selfmodel_precision2_20261125.md)：
- **g**：在发令时刻的状态上用 `s_check.reference` 重建 J 的参考，参考终点的夹爪中心减目标。
- **p**：模型预测的"实际的手减参考的手"，手掌三个球的平均，即 precision2 里 modelf 用的那个。
- **挪法**：把预抓取目标挪 −(g + p)，不设门槛（每个有路线的局都挪），一轮，和 modelf 完全一样。
- **缩放重试**：cuRobo 规划用全量、2/3、1/3，都不行就退回 J 自己的路线，和精度实验一样。

### 同一批局上的对手（都不用模型）

- **A**：直线伸手，已有，不重跑。
- **J**：已有，从 JR 取发令状态，JR 和 J 逐局相同。
- **const**：挪"开发局平均停偏"的负值，用精度实验里冻结的 `const_<控制器>.json`，不重算。
- **kinc**：挪 −(g + c_t)，用 precision2 里冻结的 `ctrack_<控制器>.json`，不重算。

const 和 kinc 只做对照，不进主表。主表只有 A、J、S，S 换成 SF。

### 局和流程

- **局**：测试局，即 `outputs/simple_eval_20261112/bench_rgbd_20261124/<控制器>/` 的 J、JR（`via/` 路线、`films_JR/` 录像、`rows/JR_*`）。
- **只重跑 J 有路线的局**：ScaleBFM 28 局（全在 TabletopGrasp），SONIC 29 局（28 局 TabletopGrasp、1 局 XMovePick）。其余局 SF 和 J 一样退回直线伸手，计分时直接用 J 的行。
- **输出**：新目录 `outputs/simple_eval_20261112/sfull_20261125/<控制器>/`，原来的目录只读。
- **试跑**：开发局（`outputs/simple_dev_20261124/bench/<控制器>/` 的 JR）上取 2 个有路线的位置，确认流程能跑通、发令状态和 JR 相同。试跑的数不报、不用。

## 度量

1. **主指标**：SIMPLE 四个抓取任务的成功局数，45 秒内和官方时限内，重点是 TabletopGrasp。A、J、旧 S（selfmodel_s_20261124）、SF 并排。
2. **有路线的局上逐局配对**，SF、const、kinc 各减 J：
   - 45 秒内和官方时限内的成功数之差；
   - 只有一边成功的局数。
3. **预抓取停偏**：夹爪中心对开局那一眼的目标，停稳以后。
   - 各组的中位数；
   - SF − const、SF − kinc 的按局配对中位数；
   - 按布局（预抓取点 1 cm 以内算一个布局）重抽 4000 次的 95 % 区间和符号检验；
   - entry 停偏照报。
4. **缩放和退回次数**，每组。
5. **流程核对**：
   - 没有路线的局 SF 和 J 相同（直接用 J 的行，这条只是确认局号对齐）；
   - 每组的发令状态和 JR 相同的局数；
   - 失败的进程数。
6. **模型在这批局上的预测误差（只报）**：JR 录像上实际的"手减参考"对 p 的误差（中位数、90 % 分位）；发令时 29 个关节超出训练数据 0.5–99.5 % 范围的个数（`jref_data.py range`）。

## 预测（跑之前写；没看过任何 SF、const、kinc 在官方局上的执行结果）

能参考的已知数：
- selfmodel_s_20261124 复核里离线外推的"常数在这批局上留 1.26 cm、旧模型挪法留 2.04 cm"（ScaleBFM）；
- precision2 抖动测试局的执行结果。

没有在官方局上离线外推 SF。

**ScaleBFM（有路线 28 局）**：
- 预抓取停偏中位数：J 3.0–5.0，const 1.0–2.2，kinc 1.2–2.5，SF 0.8–1.8 cm。
- SF − const：−0.9 到 +0.1，中间猜 −0.4。按"赢"的定义（布局区间整个在 0 以下且符号检验 p < 0.05）赢的可能约四成：只有一次重复，28 局，比 precision2 的 112 局少得多。
- SF − kinc：−1.0 到 0，中间猜 −0.5，赢的可能约四成半。
- 有路线的局上的成功：
  - SF 45 秒内 24–28（J 28），官方时限内 19–25（J 24）；
  - const、kinc 同样的区间。
- 主指标：SF 四个任务 91–95 / 76–82（J 95 / 81，旧 S 88 / 77），TabletopGrasp 26–30 / 21–27（J 30 / 26，旧 S 23 / 22）。
- 退回 J：不超过 4 / 28。

**SONIC（有路线 29 局）**：
- 预抓取停偏中位数：J 4–9，const 3–7，kinc 4–10，SF 3–7 cm。
- SF − const：−1.5 到 +1.5，赢的可能约一成。
- SF − kinc：−3 到 0，赢的可能约三成半。precision2 上是 16 / 19 个布局，这次局少。
- 有路线的局上的成功：SF 45 秒内 9–16（J 14），官方时限内 5–11（J 11）。理由：precision2 上 SONIC 挪过的各组官方时限内都掉了。
- 主指标：SF 四个任务 29–36 / 14–20（J 34 / 20，旧 S 34 / 18），TabletopGrasp 10–17 / 6–12（J 15 / 12，旧 S 15 / 10）。
- 退回 J：不超过 4 / 29。

**两个控制器**：
- 没有路线的局和 J 逐局相同：ScaleBFM 92、SONIC 91，按定义成立。
- 发令状态和 JR 全部相同。
- 模型预测误差：ScaleBFM "手减参考"的误差中位数 0.5–1.5 cm；SONIC 2–6 cm。
- 关节超出范围：中位数 0 到 1，最大不超过 8。

## 判据（什么结果会让我改做法，下一步做什么）

**流程（先查，再读数）**：
- 出现以下任一情况，就停下先查代码，不报 SF 的数：
  - 有进程失败；
  - 某组发令状态和 JR 不同；
  - 某组退回 J 超过 30 %。

**主表**：照 precision2 的判据，主表的 S 换成 SF，旧 S 的数留在 selfmodel_s_20261124 里，作为早期版本。

**精度（每个控制器分开）**：
- **SF 对 const、对 kinc 都赢**：官方局和抖动局一致，这个控制器上"一次命令的精度"的论断成立。
- **打平**（区间跨 0，或符号检验不显著）：
  - 读法：官方局只有一次重复、28 / 29 局，检验力不够。精度的论断仍以 precision2（112 个局次）为准，并写明官方局上没有单独显著。
  - 不为了显著去加局、调参数。
- **SF 输给 const**（区间整个在 0 以上）：和 precision2 矛盾。先查两次的差别（抖动、局、代码），查清以前不往论文里写精度的论断。

**成功**：
- 有路线的局上，SF 45 秒内比 J 少 4 局以上（任一控制器）：挪目标本身让成功变少。主表照实报。
- 下一步另登记一个变体，只在预测的停偏超过一个半径时才挪，或挪完以后 entry 照 J 走，用开发局选半径。
- SF 少得不超过 3 局：主表照报，不再在 SIMPLE 上调 S。

## 预测修订

2026-11-25，测试局开跑以前补记实现（预测和判据没改）：

- **脚本**：新文件 `sfull_run.sh`（一个控制器的整条流程）和 `sfull_launch.sh`（`test` / `trial`）。
- **`precision.py`**（md5 `6a56abc4bd1159ad35af020bb975a3de`）：
  - `shifts2` 加了 `--arms/--const/--out`，默认行为不变。SF 和 modelf 用同一个算式；const 和原 `shifts()` 一样（开发局留一布局）。
  - `pick` 加了 `--row-fmt`：SONIC 的 XMovePick 那一局和 TabletopGrasp 局号重复，规划行文件改叫 `plan_row_<task>_<ep>.json`。
  - 新增 `table_sfull`。
  - 旧的 `table`、`table2` 在开发局和测试局、两个控制器上，和原来的文件逐字节一致（8 / 8）。
- **不抖动**：不传 `--stance-jitter`。抖动分支挡在 `elif a.stance_jitter > 0` 后面，默认 0，所以和 J 走同一条代码路径。
- **`--film-ref`**：各组都带，它只记录。JR 带它、J 不带，两者逐局相同（selfmodel_s_jr_20261124）。
- **源目录只读**：`step1`、`step2`、`try_act02`、`via`、`films_JR` 用符号链接，JR 的行复制一份，新的行和录像都写在新目录。
- **4090 上的 `jref_data.py`**：原来是没经 ruff 排版的旧版（`c21421fb…`），和 git HEAD 的语法树完全相同，只差换行。已同步成 HEAD 版本（`8e7e1130…`），两边一致。
- **试跑**：开发局第 30、48 两个位置，两个控制器，在 `outputs/simple_dev_20261124/sfull_trial/`，第一行 `# notebook: skip sfull trial`。结果：
  - 进程没有失败；
  - 三组的发令状态和 JR 都是 2 / 2 相同，没有行带站位抖动；
  - 用到了 1、2/3、1/3 各种缩放，没有退回 J；
  - 表各节都打出来了；
  - 开发局 bench 没被改动。
- **试跑的数不报、不用。**

**2026-11-25，测试局开跑以前，看了 precision2 的独立复核以后的修订（加两组对手、改核心判据）：**

- **为什么改**：复核找到一个更强的不用模型的对手。离线外推它在抖动的测试局上和 modelf（就是 SF 的挪法）打平：ScaleBFM +0.09 [−0.06, +0.26]，8 / 18 个布局。原来的 kinc 把参考按真实骨盆搬过去，框架不对，是个偏弱的对手。
- **新加两组**（都不用模型、不用仿真真值）：
  - **er**：J 的参考在终点的夹爪中心，用电机自己的参考根（`ref0[0:7]`）放，不搬到骨盆上，减开局那一眼的目标。坐标用参考根的朝向，所以不经过真实骨盆。
  - **kin_rr**：挪 −er。
  - **kinc_rr**：挪 −(er + c_rr)。c_rr 是开发局（和 const 同一批 3、4 级 JR 局）上"实测夹爪中心 − er"的平均。测试局开跑以前算好，存 `outputs/simple_dev_20261124/precision_const/ctrack_rr_<控制器>.json`，记 md5。
  - 规划、缩放重试、退回 J 都和别的组一样。
- **核心判据改为**：每个控制器，SF 只有**同时赢 const 和 kinc_rr**，才算模型在官方局上的贡献。"赢"的定义不变。kinc 照报，不再是核心。
- **主表**：S 仍然换成 SF（照 precision2 的判据）。如果 SF 没赢 kinc_rr，我向用户建议主表加一行"不用模型：参考运动学加常数"（kinc_rr），由用户定。
- **新两组的预测**（看过复核在抖动局上的离线外推；没看过官方局上的任何 er 数）：
  - ScaleBFM：kin_rr 停偏中位数 1.2–2.8，kinc_rr 0.7–1.8 cm；SF − kinc_rr −0.4 到 +0.5，中间猜 +0.1；SF 赢的可能约一成。
  - SONIC：kin_rr 3–7，kinc_rr 3–7 cm；SF − kinc_rr −1.5 到 +1.5；SF 赢的可能约一成。
  - 有路线的局上的成功：kin_rr、kinc_rr 和 SF 同样的区间。
  - 退回 J：每组不超过 4 局。
- **一个已知的问题，照记**：
  - SF 和旧 S、kinc 一样，经 `s_check.frames` 用发令时刻的真实骨盆（`qpos0`，仿真里的值）搬参考、算模型输入；er 只用电机自己的参考根。
  - 这和"不用仿真真值"有冲突，从 selfmodel_s_20261124 起就这样，这次不改。
  - 只有 SF 赢了 kinc_rr，才值得另做一个不用真值的 SF。
- **其余不变**：局、流程、度量、其余预测和判据都照原样。

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
## 结果（贴原始输出，不转述）

```
started 2026-11-24T21:41:21Z   done 2026-11-24T22:05:56Z
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

`outputs/simple_eval_20261112/sfull_20261125/scalebfm/table_sfull.txt`（各节按登记的度量编号，所以顺序是 1、2、4、5、3、6）：

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/sfull_20261125/scalebfm  episodes 120, routed 28 (J's rows with a plan 28, the same episodes: True), the others take J's rows: 92

1. main metric: successes within 45 s / official limit (old S: 7 given up, failures)
  task                                A          J          S         SF      const       kinc     kin_rr    kinc_rr
  XMovePickTeleop (30)         28 / 28    28 / 28    28 / 28    28 / 28    28 / 28    28 / 28    28 / 28    28 / 28 
  XMoveBendPickTeleop (30)     27 / 27    27 / 27    27 / 27    27 / 27    27 / 27    27 / 27    27 / 27    27 / 27 
  BendPickMP (30)              10 / 0     10 / 0     10 / 0     10 / 0     10 / 0     10 / 0     10 / 0     10 / 0  
  TabletopGraspMP (30)         18 / 18    30 / 26    23 / 22    29 / 23    30 / 27    30 / 25    30 / 27    30 / 23 
  all (120)                    83 / 73    95 / 81    88 / 77    94 / 78    95 / 82    95 / 80    95 / 82    95 / 78 

2. routed episodes (28): paired successes against J
  S      45 s    : J 28, S 21, S - J -7; only S 0, only J 7  [TabletopGraspMP_L0_e1, TabletopGraspMP_L1_e1, TabletopGraspMP_L1_e5, TabletopGraspMP_L2_e0, TabletopGraspMP_L2_e1, TabletopGraspMP_L2_e4, TabletopGraspMP_L2_e8]
  S      official: J 24, S 20, S - J -4; only S 0, only J 4  [TabletopGraspMP_L1_e5, TabletopGraspMP_L2_e0, TabletopGraspMP_L2_e4, TabletopGraspMP_L2_e8]
  SF     45 s    : J 28, SF 27, SF - J -1; only SF 0, only J 1  [TabletopGraspMP_L2_e1]
  SF     official: J 24, SF 21, SF - J -3; only SF 0, only J 3  [TabletopGraspMP_L0_e2, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e6]
  const  45 s    : J 28, const 28, const - J +0; only const 0, only J 0
  const  official: J 24, const 25, const - J +1; only const 1, only J 0  [TabletopGraspMP_L0_e6]
  kinc   45 s    : J 28, kinc 28, kinc - J +0; only kinc 0, only J 0
  kinc   official: J 24, kinc 23, kinc - J -1; only kinc 0, only J 1  [TabletopGraspMP_L2_e2]
  kin_rr 45 s    : J 28, kin_rr 28, kin_rr - J +0; only kin_rr 0, only J 0
  kin_rr official: J 24, kin_rr 25, kin_rr - J +1; only kin_rr 1, only J 0  [TabletopGraspMP_L1_e1]
  kinc_rr 45 s    : J 28, kinc_rr 28, kinc_rr - J +0; only kinc_rr 0, only J 0
  kinc_rr official: J 24, kinc_rr 21, kinc_rr - J -3; only kinc_rr 0, only J 3  [TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e6, TabletopGraspMP_L2_e2]

4. shift scale used (1, 2/3, 1/3, fallback) and fallback to J's plan:
  SF     1 22 2/3 5 1/3 1 fallback 0; fallback 0 / 28 (0 %)
  const  1 21 2/3 2 1/3 5 fallback 0; fallback 0 / 28 (0 %)
  kinc   1 20 2/3 3 1/3 5 fallback 0; fallback 0 / 28 (0 %)
  kin_rr 1 27 2/3 1 1/3 0 fallback 0; fallback 0 / 28 (0 %)
  kinc_rr 1 23 2/3 2 1/3 3 fallback 0; fallback 0 / 28 (0 %)
  full shift length median, cm: SF 4.24, const 4.62, kinc 5.25, kin_rr 2.53, kinc_rr 4.05; |g| 2.04, |p| 4.93, |er| 2.53
  reference root (kin_rr, kinc_rr): |yaw of the root - W's| median 0.33, max 1.83 deg; J's jaw to the root's end median 2.16 cm, to the carried end 4.66 cm; c_rr episodes 11-11

5. pipeline checks
  state at the command equal to JR's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm): SF 28/28, const 28/28, kinc 28/28, kin_rr 28/28, kinc_rr 28/28
  rows carrying a stance jitter: 0
  failed processes: SF 0, const 0, kinc 0, kin_rr 0, kinc_rr 0; rows on the routed episodes: SF 28/28, const 28/28, kinc 28/28, kin_rr 28/28, kinc_rr 28/28

3. pregrasp settled miss, cm (jaw centre against the look's goal), 28 routed episodes, 18 layouts; medians skip missing rows
  J      median   3.81  mean   4.80  missing 0
  SF     median   1.22  mean   1.20  missing 0 | minus J: median -2.78 [-4.32, -1.87], closer in 28/28; layouts closer 18/18 (sign p 7.63e-06)
  const  median   1.47  mean   1.80  missing 0 | minus J: median -2.60 [-4.28, -1.60], closer in 28/28; layouts closer 18/18 (sign p 7.63e-06)
  kinc   median   1.75  mean   1.87  missing 0 | minus J: median -2.40 [-3.20, -1.40], closer in 28/28; layouts closer 18/18 (sign p 7.63e-06)
  kin_rr median   1.93  mean   2.07  missing 0 | minus J: median -2.34 [-3.36, -1.05], closer in 28/28; layouts closer 18/18 (sign p 7.63e-06)
  kinc_rr median   1.18  mean   1.14  missing 0 | minus J: median -3.17 [-4.21, -2.33], closer in 28/28; layouts closer 18/18 (sign p 7.63e-06)
  SF minus const: median -0.32 [-0.75, +0.19] cm, SF closer in 17/28 episodes, layouts 12/18 (worse 6, sign p 0.238)
    seen by the model's data: 23 episodes, SF minus const median -0.31 [-0.83, +0.29], layouts closer 10/15 (sign p 0.302)
    never seen: 5 episodes, SF minus const median -0.60 [-2.02, +0.15], layouts closer 2/3 (sign p 1)
  SF minus kinc_rr: median +0.10 [-0.23, +0.30] cm, SF closer in 11/28 episodes, layouts 6/18 (worse 12, sign p 0.238)
    seen by the model's data: 23 episodes, SF minus kinc_rr median +0.01 [-0.27, +0.30], layouts closer 6/15 (sign p 0.607)
    never seen: 5 episodes, SF minus kinc_rr median +0.14 [+0.13, +0.30], layouts closer 0/3 (sign p 0.25)
  SF minus kinc: median -0.55 [-1.24, -0.04] cm, SF closer in 22/28 episodes, layouts 14/18 (worse 4, sign p 0.0309)
    seen by the model's data: 23 episodes, SF minus kinc median -0.41 [-1.37, -0.04], layouts closer 11/15 (sign p 0.118)
    never seen: 5 episodes, SF minus kinc median -0.57 [-1.13, -0.52], layouts closer 3/3 (sign p 0.25)

3. entry settled miss, cm (jaw centre against the look's goal), 28 routed episodes, 18 layouts; medians skip missing rows
  J      median   2.33  mean   2.40  missing 0
  SF     median   3.01  mean   2.92  missing 0 | minus J: median +0.40 [+0.27, +0.57], closer in 4/28; layouts closer 4/18 (sign p 0.0309)
  const  median   2.85  mean   2.91  missing 0 | minus J: median +0.46 [+0.29, +0.68], closer in 2/28; layouts closer 2/18 (sign p 0.00131)
  kinc   median   2.92  mean   2.99  missing 0 | minus J: median +0.56 [+0.28, +0.85], closer in 2/28; layouts closer 2/18 (sign p 0.00131)
  kin_rr median   2.53  mean   2.67  missing 0 | minus J: median +0.16 [+0.02, +0.35], closer in 5/28; layouts closer 5/18 (sign p 0.0963)
  kinc_rr median   3.07  mean   2.96  missing 0 | minus J: median +0.43 [+0.19, +0.54], closer in 2/28; layouts closer 2/18 (sign p 0.00131)
  SF minus const: median -0.03 [-0.11, +0.03] cm, SF closer in 18/28 episodes, layouts 12/18 (worse 6, sign p 0.238)
    seen by the model's data: 23 episodes, SF minus const median -0.04 [-0.11, +0.06], layouts closer 9/15 (sign p 0.607)
    never seen: 5 episodes, SF minus const median -0.02 [-0.19, -0.00], layouts closer 3/3 (sign p 0.25)
  SF minus kinc_rr: median -0.03 [-0.08, +0.02] cm, SF closer in 18/28 episodes, layouts 12/18 (worse 6, sign p 0.238)
    seen by the model's data: 23 episodes, SF minus kinc_rr median -0.02 [-0.08, +0.04], layouts closer 9/15 (sign p 0.607)
    never seen: 5 episodes, SF minus kinc_rr median -0.06 [-0.09, -0.03], layouts closer 3/3 (sign p 0.25)
  SF minus kinc: median -0.07 [-0.27, +0.01] cm, SF closer in 18/28 episodes, layouts 12/18 (worse 6, sign p 0.238)
    seen by the model's data: 23 episodes, SF minus kinc median -0.06 [-0.27, +0.01], layouts closer 9/15 (sign p 0.607)
    never seen: 5 episodes, SF minus kinc median -0.08 [-0.27, -0.07], layouts closer 3/3 (sign p 0.25)

6. the model's error on JR's films (actual palm minus reference at rest, against p = the SF arm's prediction)
  28 episodes (0 without a check row: fell or too short): |actual - p| median 1.24, 90 % 2.92 cm; |actual| (the reference alone) median 5.83, 90 % 9.83 cm
  jref_data.py range on JR's command states: joints at the command outside the training paths' 0.5-99.5 % band (28 reaches): median 0.0, mean 1.00, max 9, counts [18, 5, 2, 1, 0, 0, 0, 1, 0, 1]
    seen layouts: 19 reaches, median 0.0
    held-out layouts: 9 reaches, median 0.0
```

`outputs/simple_eval_20261112/sfull_20261125/sonic/table_sfull.txt`：

```
== /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/sfull_20261125/sonic  episodes 120, routed 29 (J's rows with a plan 29, the same episodes: True), the others take J's rows: 91

1. main metric: successes within 45 s / official limit (old S: 6 given up, failures)
  task                                A          J          S         SF      const       kinc     kin_rr    kinc_rr
  XMovePickTeleop (30)          5 / 5      5 / 5      5 / 5      5 / 5      5 / 5      5 / 5      5 / 5      5 / 5  
  XMoveBendPickTeleop (30)      3 / 3      3 / 3      3 / 3      3 / 3      3 / 3      3 / 3      3 / 3      3 / 3  
  BendPickMP (30)              11 / 0     11 / 0     11 / 0     11 / 0     11 / 0     11 / 0     11 / 0     11 / 0  
  TabletopGraspMP (30)         10 / 10    15 / 12    15 / 10    14 / 9     16 / 12    17 / 13    17 / 15    16 / 14 
  all (120)                    29 / 18    34 / 20    34 / 18    33 / 17    35 / 20    36 / 21    36 / 23    35 / 22 

2. routed episodes (29): paired successes against J
  S      45 s    : J 14, S 14, S - J +0; only S 4, only J 4  [TabletopGraspMP_L0_e6, TabletopGraspMP_L0_e7, TabletopGraspMP_L0_e8, TabletopGraspMP_L0_e9, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e7, TabletopGraspMP_L1_e9, TabletopGraspMP_L2_e2]
  S      official: J 11, S 9, S - J -2; only S 3, only J 5  [TabletopGraspMP_L0_e6, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e6, TabletopGraspMP_L1_e7, TabletopGraspMP_L2_e2, TabletopGraspMP_L2_e3, TabletopGraspMP_L2_e5, TabletopGraspMP_L2_e6]
  SF     45 s    : J 14, SF 13, SF - J -1; only SF 4, only J 5  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e4, TabletopGraspMP_L0_e6, TabletopGraspMP_L0_e7, TabletopGraspMP_L0_e9, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e7, TabletopGraspMP_L1_e9, TabletopGraspMP_L2_e4]
  SF     official: J 11, SF 8, SF - J -3; only SF 2, only J 5  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e4, TabletopGraspMP_L0_e5, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e7, TabletopGraspMP_L2_e4, TabletopGraspMP_L2_e5]
  const  45 s    : J 14, const 15, const - J +1; only const 5, only J 4  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e7, TabletopGraspMP_L0_e9, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e4, TabletopGraspMP_L1_e7, TabletopGraspMP_L1_e8, TabletopGraspMP_L1_e9, TabletopGraspMP_L2_e2]
  const  official: J 11, const 11, const - J +0; only const 4, only J 4  [TabletopGraspMP_L0_e0, TabletopGraspMP_L0_e7, TabletopGraspMP_L0_e8, TabletopGraspMP_L1_e4, TabletopGraspMP_L1_e7, TabletopGraspMP_L1_e8, TabletopGraspMP_L2_e2, TabletopGraspMP_L2_e5]
  kinc   45 s    : J 14, kinc 16, kinc - J +2; only kinc 5, only J 3  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e2, TabletopGraspMP_L0_e6, TabletopGraspMP_L0_e7, TabletopGraspMP_L0_e9, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e7, TabletopGraspMP_L2_e5]
  kinc   official: J 11, kinc 12, kinc - J +1; only kinc 4, only J 3  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e2, TabletopGraspMP_L0_e6, TabletopGraspMP_L1_e2, TabletopGraspMP_L1_e7, TabletopGraspMP_L2_e4, TabletopGraspMP_L2_e5]
  kin_rr 45 s    : J 14, kin_rr 16, kin_rr - J +2; only kin_rr 3, only J 1  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e6, TabletopGraspMP_L0_e9, TabletopGraspMP_L2_e8]
  kin_rr official: J 11, kin_rr 14, kin_rr - J +3; only kin_rr 4, only J 1  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e6, TabletopGraspMP_L1_e8, TabletopGraspMP_L1_e9, TabletopGraspMP_L2_e8]
  kinc_rr 45 s    : J 14, kinc_rr 15, kinc_rr - J +1; only kinc_rr 5, only J 4  [TabletopGraspMP_L0_e1, TabletopGraspMP_L0_e5, TabletopGraspMP_L0_e6, TabletopGraspMP_L0_e7, TabletopGraspMP_L0_e9, TabletopGraspMP_L1_e7, TabletopGraspMP_L1_e9, TabletopGraspMP_L2_e2, TabletopGraspMP_L2_e8]
  kinc_rr official: J 11, kinc_rr 13, kinc_rr - J +2; only kinc_rr 4, only J 2  [TabletopGraspMP_L0_e5, TabletopGraspMP_L0_e6, TabletopGraspMP_L0_e7, TabletopGraspMP_L0_e8, TabletopGraspMP_L1_e7, TabletopGraspMP_L2_e8]

4. shift scale used (1, 2/3, 1/3, fallback) and fallback to J's plan:
  SF     1 27 2/3 1 1/3 0 fallback 1; fallback 1 / 29 (3 %)
  const  1 25 2/3 2 1/3 2 fallback 0; fallback 0 / 29 (0 %)
  kinc   1 22 2/3 4 1/3 2 fallback 1; fallback 1 / 29 (3 %)
  kin_rr 1 27 2/3 0 1/3 1 fallback 1; fallback 1 / 29 (3 %)
  kinc_rr 1 25 2/3 2 1/3 1 fallback 1; fallback 1 / 29 (3 %)
  full shift length median, cm: SF 4.03, const 4.72, kinc 6.58, kin_rr 2.89, kinc_rr 5.83; |g| 8.77, |p| 9.53, |er| 2.89
  reference root (kin_rr, kinc_rr): |yaw of the root - W's| median 6.35, max 10.90 deg; J's jaw to the root's end median 3.75 cm, to the carried end 12.08 cm; c_rr episodes 15-15

5. pipeline checks
  state at the command equal to JR's (jitter, jitter_rep, walk, stand; pregrasp S0, S3, W, J, t_cmd, goal_mm): SF 29/29, const 29/29, kinc 29/29, kin_rr 29/29, kinc_rr 29/29
  rows carrying a stance jitter: 0
  failed processes: SF 0, const 0, kinc 0, kin_rr 0, kinc_rr 0; rows on the routed episodes: SF 29/29, const 29/29, kinc 29/29, kin_rr 29/29, kinc_rr 29/29

3. pregrasp settled miss, cm (jaw centre against the look's goal), 29 routed episodes, 25 layouts; medians skip missing rows
  J      median   6.33  mean   7.58  missing 0
  SF     median   5.93  mean  10.12  missing 0 | minus J: median -0.06 [-1.86, +2.22], closer in 15/29; layouts closer 13/25 (sign p 0.839)
  const  median   4.62  mean  14.37  missing 0 | minus J: median -0.44 [-1.89, +1.57], closer in 16/29; layouts closer 14/25 (sign p 0.69)
  kinc   median   8.08  mean   9.30  missing 0 | minus J: median +1.56 [-0.88, +3.64], closer in 10/29; layouts closer 8/25 (sign p 0.152)
  kin_rr median   4.38  mean   8.27  missing 0 | minus J: median -0.69 [-2.06, +0.95], closer in 16/29; layouts closer 14/25 (sign p 0.69)
  kinc_rr median   4.51  mean   5.68  missing 0 | minus J: median -1.12 [-2.65, +0.28], closer in 19/29; layouts closer 17/25 (sign p 0.108)
  SF minus const: median -0.17 [-1.30, +1.71] cm, SF closer in 16/29 episodes, layouts 15/25 (worse 10, sign p 0.424)
    seen by the model's data: 21 episodes, SF minus const median -0.03 [-1.15, +1.71], layouts closer 10/17 (sign p 0.629)
    never seen: 8 episodes, SF minus const median -1.19 [-4.70, +15.41], layouts closer 5/8 (sign p 0.727)
  SF minus kinc_rr: median +1.29 [-0.28, +6.15] cm, SF closer in 11/29 episodes, layouts 9/25 (worse 16, sign p 0.23)
    seen by the model's data: 21 episodes, SF minus kinc_rr median +1.29 [-0.51, +6.15], layouts closer 7/17 (sign p 0.629)
    never seen: 8 episodes, SF minus kinc_rr median +1.90 [-2.41, +19.86], layouts closer 2/8 (sign p 0.289)
  SF minus kinc: median -0.21 [-2.59, +2.54] cm, SF closer in 15/29 episodes, layouts 12/25 (worse 12, sign p 1)
    seen by the model's data: 21 episodes, SF minus kinc median -0.21 [-2.59, +2.54], layouts closer 8/17 (sign p 1)
    never seen: 8 episodes, SF minus kinc median +0.03 [-6.96, +15.62], layouts closer 4/8 (sign p 1)

3. entry settled miss, cm (jaw centre against the look's goal), 29 routed episodes, 25 layouts; medians skip missing rows
  J      median   3.64  mean   5.76  missing 0
  SF     median   4.22  mean   8.40  missing 0 | minus J: median +0.44 [-0.23, +2.24], closer in 11/29; layouts closer 9/25 (sign p 0.307)
  const  median   3.61  mean   7.20  missing 1 | minus J: median +0.21 [-0.32, +1.14], closer in 11/28; layouts closer 8/24 (sign p 0.152)
  kinc   median   4.00  mean   6.68  missing 0 | minus J: median +0.01 [-0.39, +1.29], closer in 13/29; layouts closer 11/25 (sign p 0.839)
  kin_rr median   3.69  mean   6.99  missing 0 | minus J: median -0.18 [-0.48, +0.26], closer in 16/29; layouts closer 14/25 (sign p 0.69)
  kinc_rr median   4.19  mean   6.91  missing 0 | minus J: median +0.09 [-0.13, +1.81], closer in 11/29; layouts closer 9/25 (sign p 0.23)
  SF minus const: median +0.51 [-0.38, +0.79] cm, SF closer in 12/28 episodes, layouts 11/24 (worse 13, sign p 0.839)
    seen by the model's data: 20 episodes, SF minus const median +0.07 [-0.47, +1.34], layouts closer 9/16 (sign p 0.804)
    never seen: 8 episodes, SF minus const median +0.57 [-0.10, +12.77], layouts closer 2/8 (sign p 0.289)
  SF minus kinc_rr: median +0.16 [-0.55, +1.15] cm, SF closer in 13/29 episodes, layouts 11/25 (worse 14, sign p 0.69)
    seen by the model's data: 21 episodes, SF minus kinc_rr median -0.27 [-0.83, +0.92], layouts closer 9/17 (sign p 1)
    never seen: 8 episodes, SF minus kinc_rr median +1.90 [-0.27, +15.71], layouts closer 2/8 (sign p 0.289)
  SF minus kinc: median +0.19 [-0.21, +1.37] cm, SF closer in 11/29 episodes, layouts 9/25 (worse 15, sign p 0.307)
    seen by the model's data: 21 episodes, SF minus kinc median +0.04 [-0.36, +1.32], layouts closer 7/17 (sign p 0.804)
    never seen: 8 episodes, SF minus kinc median +0.97 [-1.29, +19.00], layouts closer 2/8 (sign p 0.289)

6. the model's error on JR's films (actual palm minus reference at rest, against p = the SF arm's prediction)
  29 episodes (0 without a check row: fell or too short): |actual - p| median 4.78, 90 % 18.95 cm; |actual| (the reference alone) median 13.61, 90 % 22.34 cm
  jref_data.py range on JR's command states: joints at the command outside the training paths' 0.5-99.5 % band (29 reaches): median 0.0, mean 0.38, max 3, counts [21, 6, 1, 1]
    seen layouts: 20 reaches, median 0.0
    held-out layouts: 9 reaches, median 0.0
```

- **流程判据都没触发**：没有失败的进程；发令状态 28 / 28、29 / 29 和 JR 相同；退回 J 每组最多 1 局。
- **独立核对**：agent 另写了一个不经过 `shifts2` 的检查，重算 er、c_rr 和挪的量。和记录差不超过 0.0005 cm；每个缩放的规划输入都等于 J 的输入加挪的量（差 ≤ 1e-14）。
- **原目录没动**：`bench_rgbd_20261124` 里没有文件被改。

## 意外

预测对照：

| | 预测 | 结果 | |
|---|---|---|---|
| ScaleBFM J / const / kinc / SF 停偏 | 3.0–5.0 / 1.0–2.2 / 1.2–2.5 / 0.8–1.8 | 3.81 / 1.47 / 1.75 / 1.22 | 中 |
| ScaleBFM kin_rr / kinc_rr | 1.2–2.8 / 0.7–1.8 | 1.93 / 1.18 | 中 |
| ScaleBFM SF − const | −0.9 到 +0.1，赢约四成 | −0.32 [−0.75, +0.19]，12 / 18，不算赢 | 中 |
| ScaleBFM SF − kinc | −1.0 到 0，赢约四成半 | −0.55 [−1.24, −0.04]，14 / 18，p 0.031，赢 | 中 |
| ScaleBFM SF − kinc_rr | −0.4 到 +0.5，猜 +0.1，赢约一成 | +0.10 [−0.23, +0.30]，6 / 18，不算赢 | 中 |
| ScaleBFM 有路线的局 SF 45 秒 / 官方 | 24–28 / 19–25 | 27 / 21 | 中 |
| ScaleBFM 主指标 SF（TabletopGrasp） | 91–95 / 76–82（26–30 / 21–27） | 94 / 78（29 / 23） | 中 |
| SONIC J / const / kinc / SF | 4–9 / 3–7 / 4–10 / 3–7 | 6.33 / 4.62 / 8.08 / 5.93 | 中 |
| SONIC kin_rr / kinc_rr | 3–7 / 3–7 | 4.38 / 4.51 | 中 |
| SONIC SF − const / − kinc / − kinc_rr | −1.5 到 +1.5 / −3 到 0 / −1.5 到 +1.5 | −0.17 / −0.21 / +1.29 | 中 |
| SONIC 有路线的局 SF 45 秒 / 官方 | 9–16 / 5–11 | 13 / 8 | 中 |
| SONIC 主指标 SF（TabletopGrasp） | 29–36 / 14–20（10–17 / 6–12） | 33 / 17（14 / 9） | 中 |
| 退回 J | 每组不超过 4 局 | 最多 1 局 | 中 |
| 模型误差中位数 ScaleBFM / SONIC | 0.5–1.5 / 2–6 cm | 1.24 / 4.78 cm | 中 |
| 关节超出范围 | 中位数 0–1，最大 ≤ 8 | ScaleBFM 中位数 0、**最大 9**；SONIC 0、3 | ScaleBFM 最大值偏 |

没想到的：

1. **不用模型、也不加常数的 kin_rr，成功数最好。** 它只按电机自己的参考根去瞄。
   - ScaleBFM：主指标 95 / 82，TabletopGrasp 30 / 27（J 30 / 26）。
   - SONIC：36 / 23，TabletopGrasp 17 / 15（J 15 / 12），有路线的局官方时限内比 J 多 3（只有 kin_rr 多 4、J 多 1）。
   - 它挪得最少（中位数 2.5 / 2.9 cm），entry 停偏也最接近 J（ScaleBFM +0.16）。
   - 只有一次运行、每局一次，SONIC 的差在噪声范围内（见第 3 条）。
2. **J 自己的参考在电机框架下本来就没对准目标**：|er| 中位数 2.53（ScaleBFM）/ 2.89 cm（SONIC）。kin_rr 去掉的就是这一项，所以它更像是修了 J 的瞄准，不算一个新方法。
3. **cuRobo 同样的输入每次规划不一样**：两次试跑的规划输入和 IK 种子都相同，SONIC 的终点关节差到 20°，执行出来的停偏差到 0.55 cm，有一局官方时限内的成功翻了。各组只在同一次运行里配对比较。
4. **SONIC 上参考根和 W 的朝向差中位数 6.35°**（ScaleBFM 0.33°），比预想的大。
5. **ScaleBFM 有一局发令时 9 个关节超出训练范围**，高于预测的 8。

## 结论与下一步

1. **照修订后的核心判据，两个控制器上 SF 都没有同时赢 const 和 kinc_rr，官方局上模型不算贡献。**
   - ScaleBFM：SF − const −0.32 [−0.75, +0.19]，12 / 18，p 0.238；SF − kinc_rr +0.10 [−0.23, +0.30]，6 / 18。停偏中位数：kinc_rr 1.18、SF 1.22、const 1.47、kin_rr 1.93、J 3.81 cm。
   - SONIC：SF 5.93 cm，和 J（6.33）差不多（−0.06 [−1.86, +2.22]）；SF − kinc_rr +1.29 [−0.28, +6.15]，9 / 25。
   - SF 只赢了框架不对的 kinc（ScaleBFM −0.55，p 0.031），kinc 已不是核心对手。
2. **主表的 S 照登记换成 SF**：ScaleBFM 94 / 78（TabletopGrasp 29 / 23），SONIC 33 / 17（14 / 9）。
   - 旧 S 的"修不好就放弃"已去掉：ScaleBFM TabletopGrasp 45 秒内从 23 回到 29。
   - 官方时限内仍比 J 少：ScaleBFM −3，SONIC −3。
3. **照修订，SF 没赢 kinc_rr，我向用户建议主表加一行不用模型的对手。** kin_rr 的成功数还更好（意外 1、2）：
   - 它更像是修了 J 的瞄准，可以考虑直接并进 J；
   - 这要用户定，因为会改主指标的 J 行。

**证据强度**：一次运行，每组每局一次，有路线的 28 / 29 局，18 / 25 个布局；规划器本身有噪声（意外 3）。

**其他可能的解释**：
1. **模型是在"参考搬到真实骨盆上"的框架里训的**，它学的主要是那个框架项（precision2 复核）。换到电机参考根的框架里重训，它预测剩下的跟踪误差也许能赢 kinc_rr。
   - 上限还有空间：ScaleBFM 抖动测试局上 oracle 0.61、kinc_rr 1.26 cm。
   - 但复核的离线数说这部分可预测的不多。
2. **局少，检验力不够**：官方局只有一次重复。精度的结论以 precision3（112 / 106 个局次）为准。

**四类陷阱**：
- **benchmark 选择**：官方测试局，四个任务全报。
- **数据泄漏**：const、c_t、c_rr 都来自开发局，在运行以前冻结；这批局之前没做过 er 的分析。
- **指标误用**：度量和登记一样，对 SF 不利的（官方时限少 3）也报了。
- **事后挑选**：一次运行，没有挑；新对手 kinc_rr 是开跑以前加的，对模型不利，不是为了让模型赢。

**下一步**：
- 等用户定：主表要不要加 kin_rr / kinc_rr 这一行，以及要不要把 kin_rr 的瞄准并进 J。
- 精度的读法见 [selfmodel_precision3_20261125](selfmodel_precision3_20261125.md)。

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

1. **结论 1、2 成立。**
   - ScaleBFM 的打平有信息量：排除了 SF 有超过 0.23 cm 的优势。
   - SONIC 几乎没有检验力（差 1.5 cm 时检验力 ≤ 0.11），SF 在那里连 const 都没赢。
   - 官方时限内的 −3 / −3 不显著（McNemar p 0.25 / 0.45）。
2. **撤回"kin_rr 的成功数更好"**（意外 1、结论 3）。SONIC 上 p 0.375；ScaleBFM 上它和 const 并列。都在规划器噪声的范围内：J 和只换 IK 种子的 replan 在 SONIC 上有 24 % 的局官方时限内成败不同。建议主表加不用模型的那一行，理由只在精度上（kin_rr − J −2.13，18 / 18；参考终点落在目标 0.2 cm 内），不在成功数上。
3. **撤回其他解释 1 里"复核的离线数说这部分可预测的不多"。**
   - 用发令时刻的本体信息做 ridge，离线能预测 kinc_rr 剩下误差的一半左右（ScaleBFM 0.58 对 1.10）；现在的模型没拿到。
   - 详见 precision3 的更正第 5 条。
4. **一个没登记的发现**：S-full SONIC 上 kinc_rr 赢了 const（−1.28，19 / 25，p 0.015）。只作探索，不算结果。
5. **泄漏核对补充**："这批局之前没做过 er 的分析"属实，但位置和抖动局相同，也和开发局的标定位置有重合（precision3 更正第 6 条）。
6. **下一步**：同 precision3 的更正第 10 条。
