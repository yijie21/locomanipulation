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

## 结果

## 意外

## 结论与下一步

## 独立复核
