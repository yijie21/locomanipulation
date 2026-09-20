# G1 原双臂 expert 对照与任务录像起点修正

用户指出前次 ghost 视频把冗长的准备过程作为主要交付，要求从已站稳状态开始任务，
并要求实际运行一条原双臂 expert 视频。分支 `main`，未提交工作区。

## 原 expert 实跑

产物：`outputs/g1_original_expert_comparison_20260920/original`。
源配置为 `configs/experiments/bimanual_20ep_20260919.yaml`，task、backend、scene
配置完全相同，seed 7000；只运行一轮，使用受管 Stack 的 GPU 1 / instance 2。

先尝试原 `scripts/collect_crate.py`，资源准入在启动前拒绝：现有 2359 pids，
完整数据导出器需再预留 1800，超过 3640 安全上限。保留该失败日志。
本次需求是任务视频，随后通过同一个 `prepare`、`spec_from_config`、`expert_argv`
直接运行原 `tasks/crate/expert.py` 一轮，不启动训练数据导出器。
expert、场景、动作参数和原任务录像起点不变；没有导出训练数据，manifest 中
`produced=false` 不表示物理任务失败。无物理任务重试。

结果：`ok=true`、`why=completed`、双手接触，篮子抬升 0.20792055 m，
最终倾角 7.04°。首个 native 状态根部高度 0.787243 m，线速度约 0.0246 m/s。
录像在原 `reset_ready`、静止准入和距离检查后开始，之后行走、抓取、抬升。
视频完整时长 16.32 s、204 帧；预览 428,468 bytes，正常速度、完整解码及精确首末帧检查通过。

视频：`outputs/g1_original_expert_comparison_20260920/previews/original_bimanual.mp4`。
配置与入口身份见 `original/provenance.json`；首帧及原始轨迹见
`original/task_start_audit.json`、`original/trace/visual_crate_s7000_00_state_trace.npz`。

## 对前次交付的纠正

前次 ghost 已复用原 expert 的完整准备，但把视频提前到准备之前开始录制，
而原 expert 在准备之后开始录制。把这些准备视频当成正常任务视频，是交付与
episode 边界设计错误，不能用“接管的单帧变化很小”来替代用户的首帧就位要求。

正常 ghost 任务应在静止准入后真正开始新的录像，再驱动 reach；准备录像另存为
诊断附件。这不是裁掉已录好的失败片段，也不宣称物理准备阶段本身完全无瞬态。

实现位于 `ghost_reach_execution_cli.py`：`initialization_artifacts` 与任务 `artifacts`
分开。关闭准备录像后取得新状态并检查 quiet，发送任务录像开始命令，确认新 mp4
已打开，再取得更晚的 native seq，才交给 reach executor。准备失败不生成任务录像。
新增 `--case-index` 仅选择诊断执行项，原 14 组源目标校验与默认行为不变。
未改变控制器、共享准备流程、抓取策略或 reach 判据。

20 项针对性测试及 `make check` 通过。本次仅新跑第 1 组，无重试，结果成功。
原生任务轨迹 197 行，首 seq 2185，reach 边界 seq 2186，即 0.02 s 后进入 reach。
首帧根部高度 0.787554 m、速度约 0.00425 m/s，满足原 quiet 条件；准备轨迹
止于 seq 2159，所有初始化 reset 命令均早于任务首帧，任务记录内没有初始化重置。
独立 native 50 Hz 审计确认双手目标+足部约束连续满足 0.54 s；reach 阶段最大
20 ms 关节变化 0.672°。此结果只证明本次单组运行，不代表完整目标范围全部可达。

当前任务视频：`outputs/g1_original_expert_comparison_20260920/previews/ghost_reach_task.mp4`，
3.92 s、49 帧、396,664 bytes；完整时长、正常速度、完整解码和精确首末帧检查通过。
对照页为同目录 `index.html`。原始视频、初始化诊断和 trace 均保留；没有通过裁剪
旧视频伪造修复。代码身份与实测证据见 `ghost/provenance.json`、`ghost/result.json`、
`ghost/task_recording_audit.json`，交付验证见顶层 `delivery_summary.json`。

GPU 1 / instance 2 的两次受管任务及离线 renderer 都已结束，未停止其他会话进程。
文档最终检查 `make check-docs` 通过。
