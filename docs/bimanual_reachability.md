# 固定站姿的双臂可达实验

> **已归档（2026-10-20）**：VLA 数据采集与训练这一系列暂不使用，这一页不再随代码更新，命令可能已经跑不通。现在的主线是[场景 RSI](rsi.md)。

本页说明如何计算条件 IK 候选域，以及联合采样五维目标并用原双臂 expert 实际 reach。
先读[双臂设计](bimanual_design.md)；箱体接触和搬运属于[箱沿执行](box_rim_execution.md)。
本实验的场景只有 G1 和地面，不包含纸箱、平台或负载。

## 输入和输出

双抓点用机器人站位坐标系中的 `(x,y,h,d,theta)` 表达：x 向前、y 向左，
h 是共同距地高度，d 是两点间距，theta 是两点连线相对正对箱体的偏角。
条件 atlas 给定 h/d/theta 后扫描中点 x/y；联合搜索同时改变全部五个变量。
抓点位置固定，IK 可尝试不同向身体倾斜的工具角度。
一次真实执行使用该点选定的固定倾角，不在运动中重新优化倾角。

调用方需要准备：

- 仅地面的 `scene_locomani.xml` 和 `scene_meta.json`，后者 `objects` 为空；
  物理模型应与采集栈一致，不能用锁死机器人底座代替实际站立。
- 原 expert 站稳后的 `standing_state.json`：29 个关节、pelvis 位置和 wxyz 四元数；
  状态解析与支持字段以[实现](../src/locomani/tasks/crate/reachability_atlas.py)为准。
  不要用假定零关节姿态替代实测状态。
- 可选的 schema_version=1 配置文件，顶层为 `atlas` 与 `slices`。
  配置字段和默认值由 `AtlasConfig` / `ConditionalSlice` 定义；当前代表点导出要求 7 个唯一切片。

本次实验的环境生成脚本、原生快照和视频位于运行目录，属于实验产物，不随 Git 分发。
场景与状态是 CLI 的显式输入；版本化物流间场景是搬箱环境，不能直接当成这里的空地环境。

## 计算与查询

先完成[运行环境安装](setup.md)，再提供自己的空地场景和实测站姿路径：

```bash
. scripts/paths.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src "$VENV_SIM_PY" \
  scripts/map_ghost_reachability.py \
  --scene-xml /path/to/floor/scene_locomani.xml \
  --standing-state /path/to/standing_state.json \
  --output outputs/reachability/atlas.json \
  --validation-bank-output outputs/reachability/validation_bank.json --workers 4
```

`--config` 指定条件和搜索预算。`--cache` 仅复用场景、站姿、解析配置和实现身份都匹配的结果。
粗网格与中心先全覆盖，再细分混合边界；预算不足的格子保留未知，不自动判为失败。

输出 `atlas.json` 保留每次 IK 尝试、残差、关节/支撑/接触检查、格子支持点和计算身份：

| 分类 | 含义 |
|---|---|
| `candidate_interpolated` | 格子四角和中心均找到合格 IK 解；内部仅为插值候选 |
| `unresolved` | 支持点类别混合，或预算/细分深度不足 |
| `solver_not_proven` | 已测支持点未找到合格解，不代表物理不可能 |

查询使用同一模块的 `require_exact_slice`、`query_slice_region` 和
`candidate_intervals_along_line`。最后一个接口可把箱沿上的候选中点线段与候选域求交，
得到若干连续区间，保留中间空隙；当前不对未计算的 h/d/theta 做跨切片推断。
所有区间在用于真实箱体前，都需要当次站姿、箱体碰撞和夹持几何复核。

## 用原 expert 验证代表点

地图同时导出 `validation_bank.json`，每片选择内部候选和前方未证实边界各一个。
它是有目的的边界验证集，不是均匀随机成功率样本。
运行请求的严格字段见 [ghost CLI](../src/locomani/tasks/crate/ghost_reach_execution_cli.py)：
顶层必须提供 schema_version、source_targets、request_identity、seed、runtime、experiment。
`runtime` 选择 backend、gpu、instance 和空地 scene_dir；`experiment` 的判据由
[GhostReachCriteria](../src/locomani/tasks/crate/ghost_reach_execution.py)定义。
相对文件路径相对于请求文件解析。

```bash
PYTHONPATH=src "$VENV_SIM_PY" scripts/probe_ghost_reach.py request.json \
  --output outputs/reachability/validation/result.json --dry-run
PYTHONPATH=src "$VENV_SIM_PY" scripts/probe_ghost_reach.py request.json \
  --output outputs/reachability/validation/result.json
```

先按 [AGENTS](../AGENTS.md) 检查 GPU/instance；launcher 负责占用和释放自己的栈。
`--case-index` 只选择一个源样本，不改写目标或判据。
每组复用原 expert 的 `prepare_stationary_ready`；quiet admission 后才开启任务录像，
初始化录像独立保存。任务开始复用已准备的 PlannerArm，不再重复接管。
脚部移动、双手位姿误差、保持时长、碰撞和超时共同约束结果；禁止将失败后重试覆盖原记录。

重新调用地图命令并提供 `--cache` 和 `--dynamic-result` 可以附加实测证据。
导入检查原始目标身份与倾角，保留每次独立站姿的差异；成功点不会把整个格子染成已验证。
当前只证明空载 reach，不能推出合爪、承载或完整搬运成功。

## 五维联合采样与实测搜索

[联合采样入口](../scripts/generate_joint_reach_bank.py)生成 executor 可直接消费的任意大小目标集，
索引必须连续为 `1..N`。配置由[采样模块](../src/locomani/tasks/crate/joint_reach_sampling.py)校验：
`schema_version=1`，`method` 为 `sobol`、`stratified`、`grid` 或 `boundary_refinement`，
`bounds` 必须包含 `x_m/y_m/height_m/separation_m/yaw_deg` 的上下界。
Sobol/分层采样提供 `count`、`seed`、`bodyward_tilt_deg`；网格使用 `axis_values`。
可叠加 `anchors` 五元组列表和 `anchor_grid` 各轴列表，稳定去重后保留来源。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src "$VENV_SIM_PY" scripts/generate_joint_reach_bank.py sampling.json --output outputs/reachability/joint_bank.json
PYTHONPATH=src "$VENV_SIM_PY" scripts/probe_ghost_reach.py request.json --output outputs/reachability/shard_0/result.json --shard-index 0 --shard-count 4
```

请求的 `source_targets` 指向该目标集。每个并行分片使用独立 GPU/instance 请求和输出目录；
零基分片按 `(index-1) % shard_count` 选择目标，不改变目标坐标。
其他互斥选择方式是 `--case-index N`、`--case-range START:STOP`（含两端）和
`--case-indices FILE`（JSON 整数列表或每行一个整数）。

每条终态立即写入 `result.cases/000001.json`，保留完整轨迹、初始化、目标及身份；
`result.progress.jsonl` 和 `result.checkpoint.json` 保存紧凑进度，正常结束再汇总 `result.json`。
同一命令加 `--resume` 可跳过所有已落盘终态，包括失败；请求、目标、场景、判据、控制器配置、
实现哈希和所选索引必须一致。中断的录像不会覆盖，恢复会使用新的录像后缀。
运行期间不要修改这些源文件；断点恢复不是更换控制器后复用旧结果的途径。

必须分别报告：实测成功、实际伸手后的失败、控制器伸手前拒绝，以及初始化/基础设施失败。
原始 `execution_started` 只表示进入 executor，不能证明下发了伸手；
使用 `classify_reach_attempt` 检查轨迹中的 `controller_phase="reach"`，得到 `reach_commanded`。
成功还需原生状态轨迹复核双手同时满足位置/角度与脚部约束的保持窗口。
实测离散点和有限网格不是整个连续五维空间的可达证明。

边界加密的 `observations` 指向 `schema_version=1`、含 `observations` 列表的 JSON；
每行保存 `coordinates` 五元组、`sample_id`、`status`、`reason` 与 `reach_commanded`。
只有实际伸手的成功/失败参与归一化五维近邻边界采样，初始化、时钟和看门狗错误不参与。
原生轨迹与在线判定不一致时保留原状态，另设 `refinement_eligible=false` 排除该条加密依据并安排复测；
省略该字段兼容旧观察格式。正式加密前应完成原生复核，并冻结观察文件以免后续分析覆盖它。
`neighbor_count` 和 `min_normalized_distance` 控制邻居数及采样间距。
每个新目标记录成功/失败邻居、距离和观察集哈希。
改变倾角的重试与独立复位复测应另存目标集，保留首次结果；单一倾角失败不能说明其他倾角也失败。

## 本轮证据与历史

[2026-09-20 五维联合实测搜索](handoff/g1-joint-reach-search-20260920.md)完成
1,277 次尝试、898 次真正伸手、232 次在线与原生共同确认成功；包括联合覆盖、边界加密、
倾角对照、独立复测与异常补跑。交互报告、纸箱参数网格和完整时长预览见该记录。
这是有限实测点集，不能把成功点外包范围当成连续可达域。

[2026-09-20 可达域记录](handoff/g1-reachability-atlas-20260920.md)包含 7 个切片、
1,120 个目标、14 次实测及视频。原生轨迹验证的成功只有 2 点，IK 候选域仍需闭环验证。

启动问题的过程保留在[初期 reach](handoff/g1-ghost-reach-execution-20260920.md)、
[跳变诊断](handoff/g1-startup-jump-diagnosis-20260920.md)、
[reset 重跑](handoff/g1-ghost-reach-fixed-reset-repeat-20260920.md)和
[原 expert 与录像边界对照](handoff/g1-original-expert-task-boundary-20260920.md)。
原始[ghost 范围预览](handoff/g1-reachability-preview-20260920.md)只表示采样范围。
