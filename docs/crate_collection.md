# 双臂搬箱采集与回放

本页说明当前可运行的入口、配置边界和产物。实现依据为
[crate 导航](reference/change-map-crate.md)中的代码；字段默认值查
[生成的配置文档](reference/config_fields.md)，实验成功率查文末的具体报告。
旧流程完整保留在[历史指南](archive/crate_collection_20260914.md)，不作为当前运行指令。

## 选择专家

| 目的 | 入口与配置 | 产物和限制 |
|---|---|---|
| 视觉指导的低桌抓篮、抬起并站直 | `--expert visual --config configs/tasks/crate_visual_low_reach.json` | 原始篮子外观，仿真辅助规划，相机可见性约束；视频/控制 pilot，拒绝 `--record` |
| 全身弯腰与双手机械可达性诊断 | `--expert carry --config configs/tasks/crate_low_reach.json` | whole-body 动作 pilot，拒绝 `--record`；不等同于 visual 的持物站直流程 |
| 传统 planner 搬箱诊断 | `--expert carry`，`carry.reach_control=planner` | 视频、状态轨迹，可使用 `--record` |
| 箱壁/箱沿接触诊断 | `--expert wall --config configs/tasks/crate_wall.json` | 视频、轨迹、可选 ghost 目标，可使用 `--record` |

视觉专家的输入来源、可见性近似、失败处理和动作连续性见
[视觉采集指南](crate_visual_collection.md)。仿真信息可以辅助规划；其来源要记录，
失败接触不能被改写为成功。上述录制能力不代表数据已通过 VLA 学习验证。

## 环境与一集 pilot

按 [安装指南](setup.md)准备 sim、SONIC v1.1 与场景；whole-body 路径还需要
[POSE ownership 补丁](../patches/sonic_pose_ownership.md)。运行前按
[资源规则](../AGENTS.md#shared-resources-and-artifacts)检查 GPU、实例锁和当前占用。
下面的实例 1 / GPU 0 是示例，确认空闲后使用；单卡机器不要照抄 runner 的 GPU 1 默认值。

```bash
export INSTANCE=1
. scripts/paths.sh
bash scripts/install_scene.sh
PYTHONPATH=src "$VENV_SIM_PY" -m locomani.tasks.crate.bimanual_scene outputs/scene/crate_bimanual
PYTHONPATH=src "$VENV_SIM_PY" -m locomani.pipeline.run_crate_pilot \
  --expert wall --config configs/tasks/crate_wall.json \
  --scene outputs/scene/crate_bimanual --out outputs/crate_wall_example \
  --instance 1 --gpu 0 --rounds 1 --seed 0
```

`crate_wall.json` 显式开启 `wall_walk`；不加载配置时该字段默认为 false。
需要十集时改 `--rounds 10`，使用新输出目录。配置顺序为默认值 → `--config` →
显式 `--set KEY=VALUE`，解析结果写入运行目录并交付专家。更改 scene/profile 时
核对 `task_config.json`、场景副本和 manifest，避免把不同分布混成一个实验。
中断后 `--resume` 只从首个未完成 batch 继续；原始尝试和失败记录保留。

### Waist and reach

`BimanualArms` 共享双手与腰部 IK，但调用者决定自由度：
carry 使用 `carry.waist_dofs`（默认 `all`）；wall 无 carry 配置调用，仍为 pitch-only。
低桌 `crate_low_reach.json` / `crate_visual_low_reach.json` 选择 whole-body，
腰部三轴自由度在物理限位内参与求解。不要把 wall 的限制套用到 carry/visual。

腰关节 pitch、骨盆姿态和躯干相对世界的倾角是不同量；配置里的角度上限
不能直接当成机器人最终弯腰角。whole-body 路径协调支撑、髋部和双手，
实际跟踪、接触、箱体倾斜和抬升仍决定结果。物体超出可达范围应失败退出。
修改这些行为从 [crate 导航](reference/change-map-crate.md)定位消费者和相关测试。

### Table-height validation

低桌 profile 不会改变几何，必须单独构建低桌场景。固定桌面 XY 尺寸的配方：

```bash
PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/crate_table_height_20260913.py \
  outputs/scene/crate_bimanual outputs/scene/crate_table30 --height 0.30
```

然后把 `--scene` 指向 `outputs/scene/crate_table30`。视觉辅助模式的完整运行命令和
箱体外观/桌面碰撞处理见[低桌视觉 pilot](crate_visual_collection.md#run)。
改变高度、深度或姿态控制需要按[行为验证要求](reference/standards.md#checks)检查实际运动；
固定骨盆 CPU IK 可达性无法证明真实控制会跟踪成功。55 cm、30 cm 下蹲、30 cm
髋部前俯是不同实验，见文末；旧 squat profile 仅用于历史比较。

## 产物、录制与停止

运行目录保留 `inputs.json`、`task_config.json`、场景副本、分 batch 日志/manifest、
`videos_*/` 和默认开启的 `trace/*_state_trace.npz`。标签由专家和 batch seed 决定，
回放前读取实际文件名；不要把 carry/visual 轨迹当作 `wall_s0`。
`*_motion.jsonl`、ghost 目标和仿真状态是诊断信息，不自动成为 VLA 观测。
manifest 的结果须结合完整走近、抓取、抬升视频和物理记录判断；失败视频也保留。

wall / planner carry 的 `--record` 会强制 instance 0，并在未显式设置时
令 `--batch-rounds=10`。先确认 instance 0 属于本次任务；exporter 只在该实例运行。
复位成功后开始录制，任务成功保存、失败丢弃；manifest 的 `produced` 与 session
需要对齐。数据转换和核验见[数据指南](dataset.md)，三相机双臂配置为
[`gr00t_sonic_3cam_bimanual.py`](../configs/gr00t_sonic_3cam_bimanual.py)。
visual 和 whole-body carry 的完整动作导出尚未验证，runner 明确拒绝录制。

停止前台 pilot 用其终端的 Ctrl-C；需要清理时，先核对并停止本次 runner/expert，
再用 `bash scripts/stack/stop_instance.sh 1` 清理本次拥有的实例（示例为 1）。
该命令按实例作用，不替你判断资源归属。保留输出；不要按专家名全局杀进程。

## 10. 渲染并排视频(可选)

回放需要 state trace、与采集一致的场景副本及 Isaac 资产；不需要重新启动 SONIC 栈。
先按[渲染环境与资产构建](rendering.md#资产构建)检查房间、桌子源文件，
再构建该次运行的副本。以下针对上面的单集 wall 示例：

```bash
V="$LOCOMANI_ROOT/outputs/crate_wall_example"
R="$LOCOMANI_ROOT/outputs/renders_isaac/crate_review_example"
export LOCOMANI_SIM_DIR="$V/crate_bimanual"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  bash scripts/build_isaac_assets.sh --snapshot "$LOCOMANI_SIM_DIR"
mkdir -p "$R"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  MUJOCO_RENDER_ARGS="--insets" ISAAC_RENDER_ARGS="--insets" \
  bash scripts/render_wall_pilot.sh "$V" "$R" 1 wall_s0 > "$R/render_all.log" 2>&1
```

`LOCOMANI_SIM_DIR` 必须对应录制的副本；低桌运行不能误用默认高桌。
两个 `*_RENDER_ARGS` 分别控制 MuJoCo / Isaac 的 ego、左腕、右腕画中画，默认都关闭。
Isaac 画中画用独立的第二遍重放，开销较高；缺失或陈旧相机帧会报错。
脚本每集 Isaac 超时为 1500 s；超时失败需调查耗时后调整，不能把残片交付成完整视频。
缺失 trace 会被脚本跳过，所以 `RENDER_ALL_DONE` 本身不证明每集都渲染成功。
核对预计集数、输出文件、完整解码与首尾画面，检查房间/桌子、机器人和相机视角。

输出为 `sbs_roundNN.mp4`（左 MuJoCo、右 Isaac）及 `rNN/` 的中间文件。
Ghost 仅表示规划目标，不能作为真实接触或 VLA 观测的证据。
导出或打包前执行[视频导出规则](rendering.md#video-export)：保持全流程、正常速度和比例，
每条预览最多 **500,000 字节**，保留原片并报告实际大小。

### 中途停止渲染 / 清理

```bash
bash scripts/stack/stop_render.sh crate_review_example
```

此参数筛选输出目录；必须唯一匹配当前用户的一个受管任务。启动器自动生成内部 job ID，
不支持用户指定 `--job-id`。停止只向核验属于该任务的进程发信号；目录仍有 worker 时
拒绝重用。未受管的独立渲染器不在此命令范围内；详情见
[渲染生命周期实现](../src/locomani/pipeline/render_job.py)和[修复验证](handoff/review-fixes-20260914.md)。

## 已测证据与历史

- [运动连续性](handoff/crate-motion-flow-20260913.md)：carry 走近、交接与限速的比较。
- [55 cm 桌高](handoff/crate-table55-20260913.md)、[30 cm 下蹲](handoff/crate-table30-20260913.md)、
  [30 cm 髋部前俯](handoff/crate-low-reach-20260913.md)：不同配置和控制路径，不能混用结论。
- [视觉辅助持物站直](handoff/crate-upright-handoff-20260913.md)：完整尝试与失败类型。
- [原始设计](archive/bimanual_grasp_design.md)、[旧指南全文](archive/crate_collection_20260914.md)、
  [crate 台账](ledger/crate.md)：只在追溯相关实验时阅读。

当前代码支持的接口不等于可靠性证明；历史结论和后续撤回都依
[证据规则](reference/standards.md#evidence)核验。未建立任意深度、高成功率或 VLA 可学性保证。
