# 固定站姿的双臂可达实验

本页说明如何从一个实际站稳状态计算条件 IK 候选域，并用原双臂 expert 验证。
先读[双臂设计](bimanual_design.md)；箱体接触和搬运属于[箱沿执行](box_rim_execution.md)。
本实验的场景只有 G1 和地面，不包含纸箱、平台或负载。

## 输入和输出

双抓点用机器人站位坐标系中的 `(x,y,h,d,theta)` 表达：x 向前、y 向左，
h 是共同距地高度，d 是两点间距，theta 是两点连线相对正对箱体的偏角。
给定 h/d/theta 后，扫描中点 x/y；抓点位置固定，IK 可尝试不同向身体倾斜的工具角度。
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

## 本轮证据与历史

[2026-09-20 可达域记录](handoff/g1-reachability-atlas-20260920.md)包含 7 个切片、
1,120 个目标、14 次实测及视频。原生轨迹验证的成功只有 2 点，IK 候选域仍需闭环验证。

启动问题的过程保留在[初期 reach](handoff/g1-ghost-reach-execution-20260920.md)、
[跳变诊断](handoff/g1-startup-jump-diagnosis-20260920.md)、
[reset 重跑](handoff/g1-ghost-reach-fixed-reset-repeat-20260920.md)和
[原 expert 与录像边界对照](handoff/g1-original-expert-task-boundary-20260920.md)。
原始[ghost 范围预览](handoff/g1-reachability-preview-20260920.md)只表示采样范围。
