# simple_sonic_vrref_20261126：SIMPLE 里 SONIC 的手部目标改用参考骨盆坐标，先测主指标

- **日期**：2026-11-26，在任何运行之前写好，单独提交。
- **用户的决定**（11-26）："SONIC 用方案 1，先测一遍主指标"。方案 1 是把 SIMPLE 里 SONIC 的手部目标坐标从 `vr_frame="robot"` 改成 `"reference"`，先在主指标上测一遍再决定。
- **机器**：4090 的 2、3 号卡。和以前主指标的 SONIC 一样，四个任务的进程分在两张卡上。开跑前看 nvidia-smi。0 号卡不碰，1 号卡留给 HA 采集。
- **代码**：`selfwm`。主指标流程（`bench_via.sh`、`simple_bench.py`、`bench_jref_table.py`，`BENCH_SET=sensor=rgbd`）照 [simple_rgbd_20261124](simple_rgbd_20261124.md) 和 [selfmodel_sfull_20261125](selfmodel_sfull_20261125.md)。只加一处：一个选 SONIC `vr_frame` 的开关，默认仍是 `"robot"`，加了以后默认行为逐位不变。跑之前把用到的脚本抄进输出目录的 `asrun/`，记下 md5。
- **关联**：
  - [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md)：开发阶段发现，`"robot"` 下 SONIC 的手在有些姿态里一直摆，2–11 cm/s，骨盆 1–2 cm/s，像闭环振荡。换成 `"reference"` 后，站立 3 s 内就静止，伸手后也不再一直摆。
  - `src/locomani/backends/sonic_onnx/motor.py` 的模块说明：11-16 在 a800 上选默认值时，5 次冒烟伸手，`"robot"` 误差 2.0–7.3 cm，`"reference"` 1.0–45 cm。下蹲那一下在参考坐标里丢掉了骨盆自己的 7 cm。

## 两种坐标的区别

- **`"robot"`**：手部目标按机器人自己估计的骨盆来算，所以要的是参考里的世界手位。骨盆一动，目标跟着变，手再带动骨盆，形成一个闭环。
- **`"reference"`**：手部目标按参考骨盆来算，SONIC 把它当成相对自己骨盆的位置。所以没有闭环，但手的世界位置会跟着骨盆的跟踪误差走，包括下蹲高度差、站位差。

## 假设

1. 去掉闭环以后，SONIC 的手不再摆，站着伸手的任务（XMovePick、TabletopGrasp）手更稳，成功不少于 `"robot"`。
2. 下蹲的任务（XMoveBendPick、BendPick）会变差，因为下蹲时骨盆高度跟不上参考，手跟着偏。

## 做法

- **对象**：SONIC，SIMPLE 官方评测局，四个抓取任务各 30 局，感知栏，`sensor=rgbd`。
- **组别**：在 `"reference"` 下跑 A（直线伸手）和 J（cuRobo 右臂关节轨迹当参考）。和已有的 `"robot"` 下 A、J 按局配对。
  - 那一眼（`cloud_blind/`、`look_dump_blind/`）和以前完全相同。
  - J 的路线：流程允许就沿用 SONIC 已有的路线。如果流程必须从新一局的状态重新规划，就重新规划，并报出有多少局的路线变了。
- **不跑 S 和 kin_rr**：这两列用 SONIC 的自身模型，而那个模型是在 `"robot"` 下采数据训的，换坐标以后不匹配。主指标里这两列标"待新模型"。
- **复现核对**：先在 `"robot"` 下重跑 8 局 A（每个任务 2 局），必须和以前的结果逐局相同，再开正式运行。

## 度量

- **主指标**：A、J 的成功局数，45 秒内和官方时限内都报，按任务分开，重点是 TabletopGrasp。
- **配对**：每个任务、每种做法，只有 `"reference"` 成功的局数和只有 `"robot"` 成功的局数，McNemar 精确检验。
- **手稳不稳**：合手前 0.5 s 的手速中位数。
- **手准不准**：伸到预抓取点、进入抓取位置以后停下时，手离目标点的距离。
- **下蹲任务**：骨盆高度和参考之差。
- **摔倒局数**。

## 预测

**45 秒内成功局数**

| 做法 | `"robot"`（已有） | `"reference"` 预测 | TabletopGrasp：已有 → 预测 |
|---|---|---|---|
| A | 29 | 20–42 | 10 → 6–16 |
| J | 34 | 25–45 | 15 → 10–20 |

**分任务**（`"reference"` 减 `"robot"`）

| 任务 | 预测 | 依据 |
|---|---|---|
| 站着的两个（XMovePick、TabletopGrasp），A、J 各自合计 | −2 到 +8 局 | 手不摆了 |
| 下蹲的两个（XMoveBendPick、BendPick），A、J 各自合计 | −10 到 +2 局 | 骨盆高度差直接传到手上 |

**手稳和下蹲**
- 合手前手速中位数：`"reference"` < 1 cm/s；`"robot"` 2–6 cm/s。
- 下蹲任务的骨盆高度差：`"reference"` 下中位数 3–8 cm。

**复现核对**：8/8 局相同。

## 判据

| 结果（45 秒内） | 结论和下一步 |
|---|---|
| A、J 的总数都不少于 `"robot"` 减 3，TabletopGrasp 都不少于 `"robot"` 减 1 | SIMPLE 里 SONIC 的默认改成 `"reference"`。主指标的 SONIC 一行从此用新坐标（旧数留作历史），S、kin_rr 待新模型。接着另行登记 SONIC 在 SIMPLE 里的重采 |
| 上一条成立，并且 A 或 J 的总数多 ≥ 5 局 | 同上，并照实写："robot" 下的闭环振荡一直在拖累 SONIC 的 SIMPLE 成绩 |
| A 或 J 的总数少 ≥ 4 局，或者 TabletopGrasp 少 ≥ 2 局 | 不改默认。少掉的局如果都在下蹲任务上，而且骨盆高度差能解释，就另提一个修正方案（参考坐标下补偿骨盆高度），报给用户定；否则 SONIC 在 SIMPLE 里不做自身模型，即方案 3 |
| 复现核对不是 8/8 | 先停下查清，不开正式运行 |

## 预测修订

2026-11-26，正式运行之前补记实现、复现核对和冒烟（预测和判据没改）。

### 开关

- **bench 选项 `sonic_vr_frame`**，取值 `robot`（默认）或 `reference`，用 `--bench-set sonic_vr_frame=reference` 打开，`bench_via.sh` 经 `BENCH_SET` 传入，`bench_via.sh` 本身没改。
  - 签名：只有不是 `robot` 时才变，末尾成 `.../sonic-vr-reference`；默认的签名不变。
  - 只对 SONIC 有效：别的控制器给 `reference` 直接报错。
  - 每行的 `bench.options` 多一个键 `sonic_vr_frame`。没有这个键的旧记录读进来是 `robot`。
- **改的文件**（Mac 和 4090 都只加这一处）：
  - `src/locomani/bench/options.py`：字段、签名、一致性检查。
  - `src/locomani/bench/spec.py`：能力表加一轴。
  - 四个 bench 的能力表：SIMPLE 两个值都是 ready；LocoManiLab 只有 `robot`，planned；BiGym 2.0、HumanoidArena 没有 SONIC，空。
  - `scripts/experiments/simple_eval_20261112/simple_bench.py`：解析开关，交给 `env.Env`。
  - `scripts/experiments/selfwm_20261029/env.py`：`Env` 加可选参数 `sonic_vr_frame`，是 SONIC 时替换 `SonicMotorConfig.vr_frame`；默认 None，不动电机。
  - Mac 上另有 `tests/test_bench_options.py` 新加 5 条，`docs/reference/config_fields.md` 新加一行。
- **为什么放在 `env.py`**：
  - `simple_bench` 的电机是 `env.Env` 建的（经 `capability_probe.Probe`），`SonicMotorConfig` 也只在 `Env` 里设（`REACH_SONIC_IFACE` 改 `interface` 就在这里）。所以开关作为显式参数交给建电机的地方，第一拍之前就生效。
  - 也可以不碰 `env.py`：在 `simple_bench` 里建完 `Env` 以后再替换 `env.m.sonic`。这次没这样做。
  - 02:34Z 同步到 4090 运行树的 `env.py` 在 lmlcmp3 的冻结清单里，触发了那边的 md5 核对，正式运行被拦下。协调方已让 lmlcmp3 改用私有副本。同一次同步还改了 `src/locomani/bench/locomanilab/__init__.py`，也属 locomanilab 目录，只加了能力表的一行。
- **4090 同步**：
  - 4090 上 `options.py`、`simple/__init__.py`、`env.py` 比 Mac 旧，缺 Mac 上别人没提交的 heft 改动，4090 上也没有自己独有的改动。
  - 所以只把开关这一处逐处补上去（`tools/vrref_patch.py`），没带过去 heft 的改动。补完和 Mac 版逐行对过，差的只有原来就差的那些行。
  - 同步前的原文件备份在 `backup_pre_switch/`。
- **测试**：Mac 版本的 `tests/test_bench_options.py` 28 条全过。Mac 上没有测试环境，是在 4090 的临时目录里跑的。

### 复现核对（`"robot"`，开关不给）

- 8 局：每个任务第 3、24 局，A，进程在 2、3 号卡上，任务到卡的分配和 bench_rgbd_20261124 相同。
- 和 `bench_rgbd_20261124/sonic` 比，两次都是行 8 / 8 相同、录像 8 / 8 逐位相同：
  - 行相同指逐键相同，只去掉 `wall_s` 和新键 `sonic_vr_frame`；新键的值是 `robot`。
  - 第一次 02:35Z 在共享运行树上跑；第二次 02:42Z 在下面的冻结副本里跑。

```
rows the same 8/8, films the same 8/8
```

### 冒烟（`"reference"`）

- 2 局：BendPick 第 3 局、TabletopGrasp 第 3 局，A。这两局属于测试局，正式运行会原样再跑一遍。
- 跑通了，0 个进程失败，签名是 `simple/perception-rgbd/pose-odometry/root-odometry/sonic-vr-reference`。
- **信号确实变了**：
  - `tools/vr_check.py`：同一次重置以后站 3 秒，发给 SONIC 的 VR 三点目标，两种坐标从第 0 拍起就差 2.8–4.7 cm。
  - 合手前 0.5 s 的手速：BendPick 0.71 cm/s，TabletopGrasp 0.50 cm/s；同局在 `"robot"` 下是 3.67 和 2.29 cm/s。
  - TabletopGrasp 第 3 局抓起来了（12.36 s），`"robot"` 下没抓起来。
  - BendPick 第 3 局没抓起来，预抓取停偏 25.2 cm；`"robot"` 下是 5.4 cm，抓起来了。
- **记录参考骨盆**：
  - A、J 都带 `--film-ref`，只多记录电机的参考位姿，用来量"骨盆高度和参考之差"。
  - 冒烟的两局带和不带 `--film-ref` 各跑一次：去掉 `--film-ref` 多出的键（各段的 `S3`、`t_cmd`、`W`、`J`，录像的 `ref`）以后，行逐键相同、录像逐位相同（2 / 2）。
  - `"robot"` 一侧取 `films_JR` 里的参考：JR 就是 J 带 `--film-ref`，selfmodel_s_jr_20261124 已证明和 J 逐局相同；下蹲局都没有路线，A、J、JR 发的指令相同，取用前逐局核对录像逐位相同。
  - 不用行里的 `pz`：`pz` 是要求全身 IK 达到的骨盆高度，冒烟里它和参考骨盆差 2.4–3.9 cm。

### J 的路线：重新规划

- **原因**：规划的起点是本目录 A 录像里预抓取开始那一帧（`prep2.py`），障碍物按 A 录像的第一帧放。冒烟里这一帧两种坐标差得很远：
  - 骨盆差 8.4 cm（BendPick）、25.6 cm（TabletopGrasp）。
  - 是走到站位的结果不一样：TabletopGrasp 第 3 局横向停偏 `"robot"` −33.5 cm，`"reference"` −10.4 cm。

  沿用旧路线，起点就对不上，所以照流程从新的 A 录像重新规划：A，然后规划，然后 J，和 simple_rgbd_20261124 一样。
- **规划器噪声**：cuRobo 同样的输入每次规划也不一样（selfmodel_sfull_20261125 意外 3），J 两种坐标的差里会混进规划器的噪声。
- **会报出的路线变化**：
  - 两边各有多少局有路线，哪些局只有一边有；
  - 两边都有路线的局里，关节路径相同的局数；
  - 规划起点的骨盆差和关节差，路径终点的关节差。

### 正式运行

- **输出**：4090 `/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_sonic_vrref_20261126/reference/`。
- **设置**：
  - `BENCH_SET="sensor=rgbd sonic_vr_frame=reference"`，`BENCH_EXTRA=--film-ref`，`REACH_SONIC_NETS=gpu`；
  - `BENCH_GPU=2`，`BENCH_GPUS="2 3"`；
  - 依次跑 A、plan、J，同时最多 4 个进程。
- **冻结副本**：从 `asrun/repo` 跑，它是 4090 运行树 `src`、`configs`、`scripts` 的副本，照 selfmodel_simpledata_20261126 的做法；`third_party`、`assets` 和只读的 `outputs` 是链接。开跑前先用 `asrun/MD5.check` 核对。
- **评估**：`tools/vrref_eval.py`，只用 CPU，按"度量"逐条输出；另外用 `bench_jref_table.py` 出主表。和 `bench_rgbd_20261124/sonic` 的 A、J 逐局配对。

### 冻结

- `FROZEN_UTC` 2026-11-26T02:44:54Z。
- `asrun/MD5.check`：1669 个文件，它自己的 md5 是 `b6611b945cf1d42dc160b9c670fb6048`。
- 主要文件的 md5（4090 冻结副本）：

| 文件 | md5 |
|---|---|
| `simple_bench.py` | `9ec4fe2e3280b248a77edb7cb6ebcdbc`（和 Mac 相同） |
| `bench_via.sh`（没改） | `82cd6dc26e91ec4721d53ea7bff1b3cb` |
| `bench_jref_table.py`（没改） | `8c4cc0d2fab53005fbffb57b967ecb19` |
| `selfwm_20261029/env.py` | `be6a6f8a7ef22718ecba51b2311115a6`（Mac `d48f0ccf…`，多 heft 的改动） |
| `bench/options.py` | `0318691d60d27b7e1a1a675e00bbafd1`（Mac `a2a9d6d7…`，多 heft） |
| `bench/simple/__init__.py` | `10fa6f3645741e01d89f52934097723c`（Mac `4433b81f…`，多 heft 和注释） |
| `bench/spec.py` | `1b2ddb58cbaae93188e45b8be731c10e`（和 Mac 相同） |
| `sonic_onnx/motor.py`（没改） | `6d51d28f3d19c9c2c05d49b0d1293bfc` |
| `curobo_avoid_20261120/plan2.py`（没改） | `3161d616875f6c4a238a3c79fa29b115` |
| `run_formal.sh` | `9c159551844ec5b78bc8e380dd26ee70` |
| `tools/vrref_eval.py` | `3be733c2ad518d5d9835d2b64ac1354b` |

## 结果（2026-11-26，执行 agent 的报告；原文由我从 4090 读取）

**运行概况**
- 修订：39324205。FROZEN_UTC 为 2026-11-26T02:44:54Z。正式运行从冻结副本 `asrun/repo` 跑，时间 02:47–03:20Z，没有进程失败。03:21Z 写了 DONE。
- 复现核对：每个任务取第 3、24 局，共享运行树和冻结副本各跑一遍，`rows the same 8/8, films the same 8/8`。
- 冒烟：两种坐标下，发给 SONIC 的手部目标从第 0 拍起就差 2.8–4.7 cm，说明开关确实起了作用。
- 一处偏离：A、J 都加了 `--film-ref`，只多记录电机参考位姿。核对过，加和不加结果逐位相同（2/2）。

**评估原文**：`eval.txt` 全文，md5 95b26515；后面接登记以外的 `pelvis_explain_A*.txt` 全文。复核后改成全文照贴：初稿按行号截取时错位了，A 栏第 4 节 pregrasp 的 all 行用成了 entry 那一节的。

```
reference: /wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_sonic_vrref_20261126/reference
robot:     /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/sonic

################ arm A: rows reference 120, robot 120, paired 120; signatures ['simple/perception-rgbd/pose-odometry/root-odometry/sonic', 'simple/perception-rgbd/pose-odometry/root-odometry/sonic-vr-reference', '|']
1/2. successes reference vs robot (only reference / only robot, McNemar exact p), 45 s | official limit
  success_45
    XMovePick      ( 30)    6 vs   5  (4/3, p 1.000)  diff +1
    XMoveBendPick  ( 30)    3 vs   3  (3/3, p 1.000)  diff +0
    BendPick       ( 30)    0 vs  11  (0/11, p 0.001)  diff -11
    TabletopGrasp  ( 30)   22 vs  10  (13/1, p 0.002)  diff +12
    all            (120)   31 vs  29  (20/18, p 0.871)  diff +2
  success_official
    XMovePick      ( 30)    6 vs   5  (4/3, p 1.000)  diff +1
    XMoveBendPick  ( 30)    3 vs   3  (3/3, p 1.000)  diff +0
    BendPick       ( 30)    0 vs   0  (0/0, p 1.000)  diff +0
    TabletopGrasp  ( 30)   22 vs  10  (13/1, p 0.002)  diff +12
    all            (120)   31 vs  18  (20/7, p 0.019)  diff +13
3. jaw speed over the 0.5 s before the close, cm/s: per-episode median, then the median over episodes (n)
    XMovePick      reference   0.50 (30)  robot   1.58 (30)  | ref - robot median -1.12 [-1.74, -0.72] over 30 pairs, ref lower in 28/30
    XMoveBendPick  reference   0.71 (30)  robot   2.87 (30)  | ref - robot median -1.45 [-2.78, -0.52] over 30 pairs, ref lower in 26/30
    BendPick       reference   0.71 (30)  robot   2.00 (30)  | ref - robot median -1.58 [-2.50, -0.51] over 30 pairs, ref lower in 28/30
    TabletopGrasp  reference   0.79 (30)  robot   2.52 (30)  | ref - robot median -0.88 [-1.92, -0.15] over 30 pairs, ref lower in 20/30
    all            reference   0.60 (120)  robot   2.18 (120)  | ref - robot median -1.16 [-1.74, -0.88] over 120 pairs, ref lower in 102/120
4. settled jaw centre to the goal after the pregrasp, cm, median (n)
    XMovePick      reference   7.33 (30)  robot   5.20 (30)  | ref - robot median +3.18 [+1.14, +3.66] over 30 pairs, ref lower in 9/30
    XMoveBendPick  reference  17.10 (30)  robot   8.85 (30)  | ref - robot median +9.75 [+8.03, +14.00] over 30 pairs, ref lower in 4/30
    BendPick       reference  24.38 (30)  robot   4.41 (30)  | ref - robot median +18.51 [+13.11, +21.15] over 30 pairs, ref lower in 0/30
    TabletopGrasp  reference   2.62 (30)  robot   5.72 (30)  | ref - robot median -1.98 [-7.95, -1.06] over 30 pairs, ref lower in 24/30
    all            reference   9.11 (120)  robot   4.92 (120)  | ref - robot median +3.66 [+2.25, +7.50] over 120 pairs, ref lower in 37/120
4. settled jaw centre to the goal after the entry, cm, median (n)
    XMovePick      reference   7.28 (30)  robot   7.36 (30)  | ref - robot median +1.50 [-0.72, +2.83] over 30 pairs, ref lower in 12/30
    XMoveBendPick  reference  20.47 (30)  robot   9.05 (30)  | ref - robot median +7.91 [+5.36, +13.90] over 30 pairs, ref lower in 3/30
    BendPick       reference  28.02 (30)  robot   5.94 (30)  | ref - robot median +21.63 [+12.77, +24.04] over 30 pairs, ref lower in 0/30
    TabletopGrasp  reference   2.51 (30)  robot   5.37 (30)  | ref - robot median -2.53 [-4.78, -0.93] over 30 pairs, ref lower in 25/30
    all            reference  10.02 (120)  robot   6.41 (120)  | ref - robot median +3.35 [+1.33, +4.71] over 120 pairs, ref lower in 40/120
5. squat episodes (pz >= 0): pelvis height minus the reference pelvis, last 0.5 s of the entry, cm, median (|median|, n)
    XMoveBendPick  reference   5.31 (|.|   5.31, 15)  robot   4.96 (|.|   4.96, 15)  | ref - robot median +0.69 [-0.12, +1.67] over 15 pairs, ref lower in 4/15
    BendPick       reference   6.80 (|.|   6.80, 22)  robot   4.76 (|.|   4.76, 22)  | ref - robot median +1.71 [+1.00, +2.29] over 22 pairs, ref lower in 1/22
    all            reference   6.28 (|.|   6.28, 37)  robot   4.87 (|.|   4.87, 37)  | ref - robot median +1.29 [+0.70, +1.79] over 37 pairs, ref lower in 5/37
5. squat episodes (pz >= 0): pelvis height minus pz, last 0.5 s of the entry, cm, median (|median|, n)
    XMoveBendPick  reference   7.39 (|.|   7.39, 15)  robot   4.92 (|.|   4.92, 15)  | ref - robot median +2.54 [+1.39, +3.04] over 15 pairs, ref lower in 2/15
    BendPick       reference   9.93 (|.|   9.93, 22)  robot   8.98 (|.|   8.98, 22)  | ref - robot median +0.78 [-0.29, +2.12] over 22 pairs, ref lower in 8/22
    all            reference   8.82 (|.|   8.82, 37)  robot   7.71 (|.|   7.71, 37)  | ref - robot median +1.78 [+0.77, +2.54] over 37 pairs, ref lower in 10/37
5. squat episodes (pz >= 0): pz minus the reference pelvis, last 0.5 s of the entry, cm, median (|median|, n)
    XMoveBendPick  reference  -2.45 (|.|   2.53, 15)  robot  -0.10 (|.|   0.18, 15)  | ref - robot median -2.24 [-2.32, -0.81] over 15 pairs, ref lower in 12/15
    BendPick       reference  -3.07 (|.|   3.07, 22)  robot  -3.65 (|.|   3.65, 22)  | ref - robot median +1.11 [+0.38, +1.53] over 22 pairs, ref lower in 4/22
    all            reference  -2.63 (|.|   2.66, 37)  robot  -3.10 (|.|   3.10, 37)  | ref - robot median +0.26 [-1.64, +1.17] over 37 pairs, ref lower in 16/37
   reference taken from films_JR (same qpos): {'sonic:A': 74}
6. falls: reference 0, robot 0; reference's: []
   no grasp: reference 0, robot 0

################ arm J: rows reference 120, robot 120, paired 120; signatures ['simple/perception-rgbd/pose-odometry/root-odometry/sonic', 'simple/perception-rgbd/pose-odometry/root-odometry/sonic-vr-reference', '|']
1/2. successes reference vs robot (only reference / only robot, McNemar exact p), 45 s | official limit
  success_45
    XMovePick      ( 30)    3 vs   5  (3/5, p 0.727)  diff -2
    XMoveBendPick  ( 30)    3 vs   3  (3/3, p 1.000)  diff +0
    BendPick       ( 30)    0 vs  11  (0/11, p 0.001)  diff -11
    TabletopGrasp  ( 30)   18 vs  15  (8/5, p 0.581)  diff +3
    all            (120)   24 vs  34  (14/24, p 0.143)  diff -10
  success_official
    XMovePick      ( 30)    3 vs   5  (3/5, p 0.727)  diff -2
    XMoveBendPick  ( 30)    3 vs   3  (3/3, p 1.000)  diff +0
    BendPick       ( 30)    0 vs   0  (0/0, p 1.000)  diff +0
    TabletopGrasp  ( 30)   15 vs  12  (9/6, p 0.607)  diff +3
    all            (120)   21 vs  20  (15/14, p 1.000)  diff +1
3. jaw speed over the 0.5 s before the close, cm/s: per-episode median, then the median over episodes (n)
    XMovePick      reference   0.50 (30)  robot   1.58 (30)  | ref - robot median -1.30 [-2.29, -0.72] over 30 pairs, ref lower in 28/30
    XMoveBendPick  reference   0.71 (30)  robot   2.87 (30)  | ref - robot median -1.45 [-2.78, -0.52] over 30 pairs, ref lower in 26/30
    BendPick       reference   0.71 (30)  robot   2.00 (30)  | ref - robot median -1.58 [-2.50, -0.51] over 30 pairs, ref lower in 28/30
    TabletopGrasp  reference   0.50 (30)  robot   3.18 (30)  | ref - robot median -2.02 [-2.83, -1.60] over 30 pairs, ref lower in 28/30
    all            reference   0.50 (120)  robot   2.37 (120)  | ref - robot median -1.65 [-2.09, -1.19] over 120 pairs, ref lower in 110/120
4. settled jaw centre to the goal after the pregrasp, cm, median (n)
    XMovePick      reference   8.19 (30)  robot   5.20 (30)  | ref - robot median +3.29 [+1.20, +5.29] over 30 pairs, ref lower in 7/30
    XMoveBendPick  reference  17.10 (30)  robot   8.85 (30)  | ref - robot median +9.75 [+8.03, +14.00] over 30 pairs, ref lower in 4/30
    BendPick       reference  24.38 (30)  robot   4.41 (30)  | ref - robot median +18.51 [+13.11, +21.15] over 30 pairs, ref lower in 0/30
    TabletopGrasp  reference   2.40 (30)  robot   6.19 (30)  | ref - robot median -2.44 [-4.99, -1.27] over 30 pairs, ref lower in 23/30
    all            reference  10.39 (120)  robot   5.25 (120)  | ref - robot median +5.18 [+3.34, +8.01] over 120 pairs, ref lower in 34/120
4. settled jaw centre to the goal after the entry, cm, median (n)
    XMovePick      reference   8.81 (30)  robot   7.36 (30)  | ref - robot median +1.23 [-0.23, +3.74] over 30 pairs, ref lower in 11/30
    XMoveBendPick  reference  20.47 (30)  robot   9.05 (30)  | ref - robot median +7.91 [+5.36, +13.90] over 30 pairs, ref lower in 3/30
    BendPick       reference  28.02 (30)  robot   5.94 (30)  | ref - robot median +21.63 [+12.77, +24.04] over 30 pairs, ref lower in 0/30
    TabletopGrasp  reference   3.71 (30)  robot   3.37 (30)  | ref - robot median +0.15 [-0.84, +1.38] over 30 pairs, ref lower in 13/30
    all            reference  10.71 (120)  robot   5.97 (120)  | ref - robot median +4.12 [+3.25, +6.26] over 120 pairs, ref lower in 27/120
5. squat episodes (pz >= 0): pelvis height minus the reference pelvis, last 0.5 s of the entry, cm, median (|median|, n)
    XMoveBendPick  reference   5.31 (|.|   5.31, 15)  robot   4.96 (|.|   4.96, 15)  | ref - robot median +0.69 [-0.12, +1.67] over 15 pairs, ref lower in 4/15
    BendPick       reference   6.80 (|.|   6.80, 22)  robot   4.76 (|.|   4.76, 22)  | ref - robot median +1.71 [+1.00, +2.29] over 22 pairs, ref lower in 1/22
    all            reference   6.28 (|.|   6.28, 37)  robot   4.87 (|.|   4.87, 37)  | ref - robot median +1.29 [+0.70, +1.79] over 37 pairs, ref lower in 5/37
5. squat episodes (pz >= 0): pelvis height minus pz, last 0.5 s of the entry, cm, median (|median|, n)
    XMoveBendPick  reference   7.39 (|.|   7.39, 15)  robot   4.92 (|.|   4.92, 15)  | ref - robot median +2.54 [+1.39, +3.04] over 15 pairs, ref lower in 2/15
    BendPick       reference   9.93 (|.|   9.93, 22)  robot   8.98 (|.|   8.98, 22)  | ref - robot median +0.78 [-0.29, +2.12] over 22 pairs, ref lower in 8/22
    all            reference   8.82 (|.|   8.82, 37)  robot   7.71 (|.|   7.71, 37)  | ref - robot median +1.78 [+0.77, +2.54] over 37 pairs, ref lower in 10/37
5. squat episodes (pz >= 0): pz minus the reference pelvis, last 0.5 s of the entry, cm, median (|median|, n)
    XMoveBendPick  reference  -2.45 (|.|   2.53, 15)  robot  -0.10 (|.|   0.18, 15)  | ref - robot median -2.24 [-2.32, -0.81] over 15 pairs, ref lower in 12/15
    BendPick       reference  -3.07 (|.|   3.07, 22)  robot  -3.65 (|.|   3.65, 22)  | ref - robot median +1.11 [+0.38, +1.53] over 22 pairs, ref lower in 4/22
    all            reference  -2.63 (|.|   2.66, 37)  robot  -3.10 (|.|   3.10, 37)  | ref - robot median +0.26 [-1.64, +1.17] over 37 pairs, ref lower in 16/37
   reference taken from films_JR (same qpos): {'sonic:J': 74}
6. falls: reference 0, robot 0; reference's: []
   no grasp: reference 0, robot 0

7. J's routes: reference 41, robot 29, both 27, only reference ['BendPickMP_L2_e9', 'XMovePickTeleop_L0_e4', 'XMovePickTeleop_L0_e6', 'XMovePickTeleop_L0_e7', 'XMovePickTeleop_L0_e8', 'XMovePickTeleop_L0_e9', 'XMovePickTeleop_L1_e4', 'XMovePickTeleop_L1_e6', 'XMovePickTeleop_L1_e7', 'XMovePickTeleop_L1_e8', 'XMovePickTeleop_L2_e0', 'XMovePickTeleop_L2_e2', 'XMovePickTeleop_L2_e5', 'XMovePickTeleop_L2_e8'], only robot ['TabletopGraspMP_L0_e1', 'TabletopGraspMP_L1_e1']
   shared routes with the same joint path: 0/27; end joints apart (largest joint, deg) median 63.75, max 115.56; plan goals apart median 4.19 cm, max 15.82
   plan start joints apart (largest, deg): median 58.01, max 69.81
   plan start pelvis apart, cm: median 16.22, max 28.52

== pelvis_explain_A.txt
reference arm A: settled entry miss against the pelvis pose error, medians over episodes (cm)
  XMoveBendPickTeleop  n 15  |actual|  13.29  |implied|  14.49  |height|   5.31  |actual-implied|   4.04  |actual-height|  14.60  tilt deg  10.98
  BendPickMP           n 22  |actual|  28.49  |implied|  29.30  |height|   6.80  |actual-implied|   4.47  |actual-height|  21.86  tilt deg  27.58

== pelvis_explain_A_robot.txt
sonic arm A: settled entry miss against the pelvis pose error, medians over episodes (cm)
  XMoveBendPickTeleop  n 15  |actual|   4.35  |implied|  13.42  |height|   4.96  |actual-implied|  11.21  |actual-height|   5.07  tilt deg   8.93
  BendPickMP           n 22  |actual|   4.58  |implied|  32.63  |height|   4.76  |actual-implied|  28.63  |actual-height|   4.65  tilt deg  28.13
```

**主指标**（45 秒内 / 官方时限内，括号里是 TabletopGrasp）

| 做法 | `"robot"`（现主指标） | `"reference"` |
|---|---|---|
| A | 29/18（10/10） | 31/31（22/22） |
| J | 34/20（15/12） | 24/21（18/15） |

**分任务看 45 秒内的成功局数**（reference 对 robot，括号里依次是只有 reference 成功、只有 robot 成功、McNemar p）

| 任务 | A | J |
|---|---|---|
| XMovePick | 6 对 5（4/3） | 3 对 5（3/5） |
| XMoveBendPick | 3 对 3 | 3 对 3 |
| BendPick | 0 对 11（0/11，p 0.001） | 0 对 11（0/11，p 0.001） |
| TabletopGrasp | 22 对 10（13/1，p 0.002） | 18 对 15（8/5） |

**其他度量**
- **合手前的手速**（中位数）：A 栏 reference 0.60、robot 2.18 cm/s；J 栏 0.50、2.37 cm/s。A 栏 120 局里有 102 局是 reference 更慢，J 栏是 110 局。
- **预抓取停下后，手离目标点的距离**（A 栏，中位数）：

  | 任务 | reference | robot |
  |---|---|---|
  | TabletopGrasp | 2.62 cm | 5.72 cm |
  | XMovePick | 7.33 cm | 5.20 cm |
  | XMoveBendPick | 17.10 cm | 8.85 cm |
  | BendPick | 24.38 cm | 4.41 cm |

- **下蹲局的骨盆高度减参考**（37 局，中位数）：reference 6.28 cm，robot 4.87 cm，差 +1.29 cm，区间 [+0.70, +1.79]。
- **登记以外的检查**（`pelvis_explain.py`，冻结以后才写的，不算正式度量）：下蹲局手的偏差，如果按骨盆整个位姿的误差来解释，只剩 4.0–4.5 cm 解释不了；只按高度解释，剩 14.6–21.9 cm。骨盆倾角差的中位数：XMoveBendPick 11.0°，BendPick 27.6°。
- **摔倒**：两边都是 0。
- **J 的路线**：按新一局的状态重新规划，0/27 条和旧路线相同。
  - 有路线的局：reference 41，robot 29，两边都有 27。
  - 只有 reference 有路线的 14 局：XMovePick 13 局，BendPick 1 局。
  - 规划起点的骨盆位置差，中位数 16.2 cm，最大 28.5 cm。

**对照判据**
- **第 1 行**（总数都不少于 robot − 3，TabletopGrasp 都不少于 robot − 1）：不成立。J 的总数少了 10 局。
- **第 3 行**（某一栏的总数少 ≥ 4 局）：成立，所以不改默认。
- **第 3 行的子条件**："少掉的局都在下蹲任务上，而且骨盆高度差能解释"。按原文不成立：
  - J 在 XMovePick 上也少了 2 局；
  - 骨盆高度差只比 robot 多 1.3 cm，解释不了下蹲时多出来的 13–20 cm 手偏。
  
  所以按登记，结论落到"SONIC 在 SIMPLE 里不做自身模型"，即方案 3。

**对照预测**

| 项 | 预测 | 实际 | 落点 |
|---|---|---|---|
| A 总数 | 20–42 | 31 | 中 |
| A 的 TabletopGrasp | 6–16 | 22 | 偏高 |
| J 总数 | 25–45 | 24 | 低 1 局 |
| J 的 TabletopGrasp | 10–20 | 18 | 中 |
| 站着的两个任务合计（A / J） | −2 到 +8 | +13 / +1 | A 偏高，J 中 |
| 下蹲的两个任务合计（A / J） | −10 到 +2 | −11 / −11 | 都低 1 局 |
| 手速 | reference < 1，robot 2–6 cm/s | 0.60 / 0.50；2.18 / 2.37 | 中 |
| 骨盆高度差 | 3–8 cm | 6.28 cm | 中 |
| 复现核对 | 8/8 | 8/8 | 中 |

**流程问题**：`bench_jref_table.py` 打完主表以后报 `KeyError: 'B'`，因为它的第 2 节要求有 B 组。主表照常输出了。

## 意外

1. **站着的 TabletopGrasp 上，A 栏从 10 局涨到 22 局**（13 对 1，p 0.002），预抓取误差从 5.7 cm 降到 2.6 cm。这说明在 "robot" 坐标下，闭环振荡一直在拖累 SONIC 站着抓的成功。
2. **下蹲任务上，变差的原因是骨盆倾角，不是高度。** BendPick 上骨盆倾角差 28°，按整个骨盆位姿算，能解释掉大部分手偏。登记里预想的"只补偿骨盆高度"修不好。
3. **走过去再伸手的 XMovePick，预抓取误差也变大了**（7.3 cm 对 5.2 cm）。在 reference 坐标下，手跟着实际骨盆走，走完以后的站位差会直接传到手上。
4. **J 栏变差，有一部分来自路线**：路线全部重新规划过，规划起点的骨盆位置差中位数 16 cm。J 栏前后的差别里，混着路线不同的影响。

## 结论与下一步（待复核）

- **按登记：不改默认，主指标的 SONIC 一行保持 "robot"。** J 栏少 10 局；子条件不成立，结论落到方案 3。
- **但这一轮暴露出一个能分开处理的现象**：
  - 站着时，reference 明显更好：TabletopGrasp 从 10 局到 22 局，官方时限内的总数从 18 局到 31 局（p 0.019）；
  - 下蹲时，reference 坏在骨盆倾角上。
  
  两个问题出在不同的身体状态上。
- **可以再试的修法**，要另行登记，并由用户来定：
  - (a) **按状态切换**：站着用 reference，下蹲时切回 robot。
  - (b) **慢的骨盆补偿**：用 reference，再加上一个经过低通（比如 1 s）的骨盆位姿修正。这样能补上静态的倾角和站位偏差，又不会形成快速闭环。
  - 两种都要在主指标上重测 A 和 J，并且 J 要沿用 robot 的路线，或者只比两边都有路线的局。
- **证据强度**：每格只跑了一次。TabletopGrasp 30 局里 0 级和 1 级的布局相同，有效样本大约 10 个布局。J 的差别里还混着 cuRobo 规划器每次结果不同的噪声。
- **其他可能的解释**：TabletopGrasp 的提升，可能部分来自走到站位以后的停偏不同，而不是手不晃了。例如第 3 局，robot 下横向停偏 −33.5 cm，reference 下是 −10.4 cm。
- **四类陷阱**：
  - benchmark 选择：四个任务全报；
  - 数据泄漏：无；
  - 指标误用：主指标没换；
  - 事后挑选：骨盆位姿检查是事后加的，已标明。

## 独立复核

（2026-11-26，独立 agent，只读；以下是要点，原意照录）

**总体**：数字基本都对得上，有三处解释要改。
1. TabletopGrasp 的提升主要来自走路：两种坐标下停的站位不同，robot 下更偏。"手不晃了"不是主因。
2. 28° 的倾角差，是参考自己前倾约 40°、SONIC 不跟造成的。站着弯腰伸手也有这个问题，不只下蹲。
3. J 少的 10 局基本就是 BendPick 少的 11 局，和路线无关，是纯坐标效应。路线的作用是把 A 栏在站立任务上的优势抹掉了。

判据按字面读确实是"不改默认、方案 3"，但触发它的全是 BendPick 在 45 秒内、超出官方 8 秒时限的成功局。

**1. 数字核对**
- 独立重算：
  - A 栏 TabletopGrasp 22 对 10，单边 13/1，p = 30/16384 = 0.0018；
  - BendPick 0 对 11，p = 2/2048 = 0.00098，A、J 两栏都是。
- 其余数字都对上。asrun 的 MD5.check 1669 个文件全部通过。
- 发现的问题：
  - **"评估原文"不是原样截取。** A 栏 pregrasp 的 all 行用成了 entry 那一节的（应为 9.11 / 4.92 / +3.66）；J 栏的标题和 XMovePick 行被删了。已改成全文照贴。
  - **p 值把 30 局当成了独立样本。** 0 级和 1 级的物体位置完全相同，2 级只差 1–6 cm；reference 下 10 个局号里有 9 个三级的结果相同。按局号聚类重算：

    | 任务 | 聚类方式 | 只 reference 成功 / 只 robot 成功 | p |
    |---|---|---|---|
    | TabletopGrasp | 按 10 个局号 | 6 / 1 | ≈ 0.125 |
    | TabletopGrasp | 0、1 级合并，加 2 级，共 20 组 | 10 / 1 | ≈ 0.012 |
    | BendPick | 按局号 | 7 / 0 | ≈ 0.016 |

  - **robot 的 TabletopGrasp 有 1 局假成功。** L2_e6 在 4.34 s 时是手指把物体碰起来了，结束时物体并不在手里。这一局两边都算成功，不影响配对计数。
  - **手速有记录精度下限**，0.5 cm/s 是一格。"0.50 / 0.60"其实是"慢到量不出"。

**2. 四类陷阱**

*TabletopGrasp 的提升从哪来*
- 这个任务要向左横移约 13 cm。robot 下横向停偏 −25 cm，反而停到了起点右边约 12 cm；reference 下停偏 −11.5 cm。两边的真实站位相差中位数约 17 cm。
- 结果是，robot 下有 17/30 局目标落在骨盆左侧，右手要往身前的左边够；reference 下是 0/30。
- robot 下目标在左侧的 17 局里，8 局在预抓取时就碰到了物体；目标在右侧的局一局没碰，reference 的 30 局也一局没碰。
- 只有 reference 成功的 13 局里，10 局在 robot 下目标在骨盆左侧，其中 7 局在预抓取时碰倒、2 局在进入时碰倒。预抓取时碰倒的 7 局里，有 4 局 robot 的预抓取误差只有 0.9–3.8 cm，失败不是因为手不准。
- 手速：robot 只在 8/13 局更快。同一种坐标内部，手速分不出成败：

  | 坐标 | 成功局手速中位数 | 失败局手速中位数 |
  |---|---|---|
  | reference | 1.69 cm/s | 0.50 cm/s |
  | robot | 2.28 cm/s | 2.71 cm/s |

- 结论：站位起主导，"手不晃"不是主因。J 栏也印证这一点：robot 下 cuRobo 绕障路线把 TabletopGrasp 从 10 局提到 15 局，路线能修的正是"直线伸手撞到物体"。为什么换坐标会改变走路的停点，这次的数据回答不了。

*下蹲变差的机制*
- `pelvis_explain.py` 的算法是对的。按帧独立重算：BendPick 实测手偏 28.5 cm，按骨盆位姿推出 29.3 cm，剩 4.5 cm 解释不了。
- 手偏的构成：转动约 18 cm，骨盆位置约 8 cm，其中高度 6.4 cm。
- **倾角差来自参考。** BendPick 下蹲局里，参考骨盆（我们全身 IK 的姿态）前倾中位数 40°，很多正好是 40°，像是碰到了上限。SONIC 实际只前倾 8–16°，两种坐标下几乎一样。原因是 SONIC 的腿和骨盆由它自己的 planner 决定，IDEL_SQUAT 只接收一个高度，VR 三点里没有"骨盆该俯仰多少"这个输入。robot 坐标用的是真实骨盆姿态，这个差不会传到手上；reference 坐标把它原样转给了手。
- **站着弯腰也有同样的问题。** XMoveBendPick 有 15 局站着伸手（SONIC 处于 IDLE），其中 12 局参考骨盆前倾超过 15°（中位数 25°），SONIC 实际是 −2.6°。reference 下这些局手偏 22 cm，0/12 成功。

*J 栏拆开看*

| 分组 | 局数 | J：reference 对 robot | 同一批局的 A |
|---|---|---|---|
| 两边都有路线 | 27 | 16 对 13 | 20 对 8 |
| 只有 reference 有路线 | 14 | 0 对 5 | 3 对 5 |
| 只有 robot 有路线 | 2 | 0 对 1 | 0 对 1 |
| 两边都没路线 | 77 | 8 对 15 | 同 A |

- J 的 −10，约等于 BendPick 上纯坐标造成的 −11，加上其余的 +1。
- 路线把有路线那 43 局上 A 栏的 +9 变成了 −3。
- XMovePick 的 −2 全部来自"只有 reference 有路线"的 13 局。

*判据的读法*
- 第 1 行不成立、第 3 行成立：作者读得对。
- 子条件第一句"少掉的局都在下蹲任务上"：作者用 XMovePick 的 −2 来否定它，理由弱。净损失全在 BendPick，XMovePick 的 −2 来自路线。
- 子条件第二句"骨盆高度差能解释"：确实不成立。
- "落到方案 3"作为实质结论不稳：
  1. 触发条件全来自 BendPick 在 45 秒内的成功。robot 的这 11 局在 12.0–13.1 s 才成功，官方时限是 8 s。按官方时限算，A 是 31 对 18，J 是 21 对 20，判据第 1 行反而会成立。
  2. J 的比较里混着路线重新规划的影响。
  3. 原因已经找到，可以修。

  建议写成："按登记字面是方案 3，是否进入方案 3 请用户定。"

**3. 两种修法**
- **(a) 下蹲时切回 robot**：
  - 下蹲状态能可靠判断，`Motor.squatting` 是我们自己发出的命令。
  - 但有四个问题：
    1. 覆盖不对。问题跟着参考骨盆前倾走，站着弯腰的局管不到。
    2. 切换的瞬间手目标会跳 9–13 cm，需要 0.5–1 s 的过渡。
    3. 下蹲时振荡会回来。
    4. 走路时用哪种坐标也要定。
  - 建议改成"参考骨盆前倾超过某个角度就切换"。
- **(b) 低通骨盆位姿修正**：
  - 方向对。用这次的录像反推，下蹲局手偏会降到 4.0–4.5 cm，站着弯腰的局降到 3.8 cm。只修姿态不修位置还不够。
  - 风险：它仍是一个回路，可能把快振荡变成慢振荡，"1 s"这个数没有依据；走路时要不要开也要试；会带进里程计误差；修正有滞后。
- **能推翻这些解释的实验**：只跑 A；先在非官方的开发局上定参数，再在官方 120 局上登记跑；45 秒和官方时限都报。

  | 组 | 做法 | 我的解释成立时的预期 | 推翻条件 |
  |---|---|---|---|
  | W1 | 走路用 robot，站定后用 reference | TabletopGrasp ≤ 14 局 | W1 ≥ 18 且 W2 ≤ 14：说明"手不晃"是对的，站位解释错 |
  | W2 | 走路用 reference，站定后用 robot | TabletopGrasp ≥ 18 局 | 同上 |
  | a1 | 进入 IDEL_SQUAT 时切到 robot，0.5 s 过渡 | BendPick 手偏 ≤ 6 cm，成功 7–11 局；站着弯腰的局仍有 ≥ 9/12 手偏 > 15 cm | BendPick ≤ 4 局 |
  | a2 | 同 a1，但参考骨盆前倾 > 10° 时也切 | 站着弯腰的局手偏 ≤ 8 cm，成功 ≥ 3 局 | 手偏仍 > 15 cm |
  | b1 | 低通位姿修正，τ 取 0.5、1、2 s；走路时关，站定后清零 | 下蹲手偏 ≤ 6 cm，手速 ≤ 1 cm/s，TabletopGrasp ≥ 19 局，BendPick ≥ 7 局 | 手速 ≥ 2 cm/s，或下蹲手偏 ≥ 10 cm，或 TabletopGrasp ≤ 15 局 |
  | b2 | 同 b1，走路时也开 | — | — |

  GPU 时间合计不到 1 小时；最好的方案再补跑 J 约 27 分钟。

**4. 判定**

| 论断 | 判定 | 理由 |
|---|---|---|
| 意外 1：闭环振荡在拖累站着抓 | 不支持（数字支持） | 提升主要来自站位；同一坐标内部手速分不出成败；聚类后 p 0.012–0.125 |
| 意外 2：坏在倾角不在高度 | 支持（两点修正） | 倾角差来自参考前倾约 40° 而 SONIC 只有约 10°；站着弯腰也有；倾角约占 2/3 |
| 意外 3：XMovePick 是站位差传到了手上 | 不支持（数字支持） | 走完后参考会重新对齐实测骨盆；传到手上的是伸手途中骨盆自己偏了约 4 cm |
| 意外 4：J 变差有一部分来自路线 | 支持，但低估了，也说偏了 | J 的 −10 几乎全是 BendPick 的坐标效应；路线的作用是抹掉 A 栏的 +9 |
| 按登记不改默认 | 支持 | J 少 10 局 |
| 结论落到方案 3 | 字面上对，实质证据不足 | 触发条件全在官方时限之外；原因已经找到，可以修；应交用户定 |
| 两个问题在不同身体状态上，可以分开处理 | 不支持 | 失败跟着参考骨盆前倾走，不跟下蹲状态走；TabletopGrasp 的好处来自走路阶段 |

此外，作者的四类陷阱自查漏了三项：按布局聚类、走路停偏这个混杂因素、45 秒和官方时限之间的敏感性。

## 复核后的更正（2026-11-26）

1. **收回意外 1。** TabletopGrasp 从 10 局涨到 22 局，主要原因是两种坐标下走到的站位不同：robot 下停偏约 25 cm，目标跑到身体左侧，预抓取时就碰倒了物体。不是"手不晃了"。手速确实降了，但同一种坐标内部，手速分不出成败。按布局聚类以后，p 在 0.012–0.125 之间。
2. **更正意外 2。** 倾角差来自我们的全身 IK 参考：骨盆前倾约 40°，而 SONIC 的骨盆由它自己的 planner 决定，只前倾约 10°。reference 坐标把这个差原样转给了手。站着弯腰伸手也会这样，不只是下蹲。
3. **收回意外 3 的解释。** 传到手上的是伸手途中骨盆自己偏了约 4 cm，不是走完以后的站位差。
4. **更正意外 4。** J 的 −10 几乎全是 BendPick 上纯坐标造成的。路线重新规划的作用，是把有路线那批局上 A 栏的 +9 变成 −3。
5. **结论更正**：
   - 按登记，不改默认，主指标保持 robot。
   - "落到方案 3"只是字面读法。触发条件全来自 BendPick 在 45 秒内、官方 8 秒时限之外的成功；按官方时限，A 是 31 对 18，J 是 21 对 20。原因已经找到，是参考骨盆姿态和 SONIC 实际的不一致，可以修。要不要进入方案 3，交用户定。
   - 收回"两个问题出在不同身体状态上"和"站着用 reference、下蹲用 robot"这个修法。
6. **下一步候选**，等用户定，定了再另行登记：
   - 先做 W1、W2，把站位的机制分清楚；
   - 再试 b1（低通位姿修正，走路时关）和 a2（参考骨盆前倾超过阈值就切到 robot）；
   - 都只跑 A，先在开发局上定参数，再在官方 120 局上登记跑一次；
   - GPU 时间合计不到 1 小时，加上 J 约 27 分钟。
7. **另外一个值得单独查的线索**：robot 坐标下，SONIC 走到 TabletopGrasp 的站位时横向停偏约 25 cm，比 reference 下大一倍多。这可能也是 SONIC 在 SIMPLE 里成绩差的原因之一，而且和自身模型无关。
