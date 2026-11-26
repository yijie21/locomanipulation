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

## 结果

## 意外

## 结论与下一步

## 独立复核
