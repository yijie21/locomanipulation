# G1 ghost reach：复用原 expert 的完整准备流程（2026-09-20）

分支 `main`，未提交工作区。用户授权：修正初始化迁移，再录制同一批 reach。

## 本次修改

`tasks/crate/expert.py::prepare_stationary_ready` 成为共用准备入口；原
`reset_ready` 保留原采样与场景放置，ghost 通过放置回调只重置平地机器人。
原 whole-body expert 的命令顺序、等待、125 帧平滑过渡、1.2 s ready
保持、第二次站位重置及实测静止准入均保留。ghost 在会话开始调用原
`round_runner.prime`，不再使用缩短的 PD pin 初始化。

准备完成的 `PlannerArm` 直接交给 runtime；runtime 构造不再次调用
`start()` 或 `_push()`。ghost 不发送行走指令，双手保持 OPEN。
原来修复的完整 29 关节 reset 输入校验仍保留；本次平地放置与原 expert
一样不额外指定关节向量。

录像从准备之前开始，跨两次显式重置持续记录；session prime 单独记录。
结果保存 phase、reset 命令时刻、观测 seq 和每段仿真时间。
跨 reset 时长使用单调 native seq / 50 Hz，不直接相减会归零的仿真时间。
这些观测时刻不是 reset acknowledgement，不能用于精确归因某一控制帧。

## 实验身份与判据

产物目录：`outputs/g1_ghost_reach_20260920_shared_expert_repeat`。
输入、代码身份见目录内 `input_provenance.json`、`code_provenance.json`。
使用与上一轮完全相同的 14 组 V2 目标、seed 42、平地场景和执行判据。
目标位置固定；本轮只测试朝身体倾斜 30° 的腕部方向，不能视为完整姿态可达边界。

- 双手实测捕获中心误差 ≤20 mm、方向误差 ≤15°，持续 ≥0.5 s。
- reach 最多 12 s；进入 reach 后足部水平位移或高度变化超过 30 mm 失败。
- 显式初始化重置的位姿跳变与真实物理运动分开标记；不将 reach 成功解释成整个初始化无滑移。

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/probe_ghost_reach.py \
  outputs/g1_ghost_reach_20260920_shared_expert_repeat/request.json \
  --output outputs/g1_ghost_reach_20260920_shared_expert_repeat/result.json
```

## 验证与结果

共享顺序、两次平地放置、已准备 arm 对象复用和 prime 录像的回归检查已加入。
针对性 56 项测试及 `make check` 通过；后者包含完整 CPU suite、Ruff、配置与文档检查。
本轮一次运行完整保留 14 组，无重试：14 组均通过准备准入；第 1、10、12 组 reach
成功，其余 11 组超时。独立 native 50 Hz 审计与全部 14 项运行判定一致。
第 9、13、14 组曾满足目标位姿，但仅保持 0.12、0.16、0.32 s，未达到 0.5 s。

进入 reach 的相邻 20 ms 样本间，最大关节变化 0.302°、根部水平变化
0.581 mm、足部水平变化 0.342 mm。第一秒足部累计水平位移最大 4.71 mm。
第二次重置后的准入期间根部最低高度 ≥0.7865 m，足部累计水平位移最大
8.25 mm；这些统计包含最终连续静止窗口之前的运动。

**不能宣称完整准备过程无跳变。** 首次重置前仍有明显恢复运动，最大根部/足部
水平位移分别为 12.56/16.52 cm；这些状态随后由首次重置丢弃。
两次显式重置的位姿不连续由命令与 settle 标记间的最强变化推定，不能当作真实动态，
也不能当作精确的应用 ACK。session prime 的 native 50 Hz 状态轨迹首帧根部高 0.176 m，属于
未启动控制器的 stack boot 状态，随后执行原 prime 重置，不属于已准备好的任务初始状态。
12.5 FPS 视频采用记录器一致的每4帧采样，视频首帧已处于重置后；原始状态轨迹没有裁剪。
因此本次解决的是迁移入口与原 expert 准备流程不一致、以及 reach 前重复初始化控制
对象的问题；初始化全段绝对连续仍未证明。

实测来源：产物目录中的 `result.json`、`shared_startup_audit.json`、
`audit_shared_startup.py`、原始 `videos/trace/*.npz`。原始视频及源轨迹均保留。
预览 `previews/index.html` 包含 session prime 和 14 组，合计 277.04 s，正常速度、
完整时长；15 个 mp4 为 335,144–455,211 bytes，全部完整解码并提取精确首末帧。
第 1 组 446,807 bytes。ZIP 仅包含合规预览与网页/验证记录，约 6.47 MB。
受管 stack instance 2 及离线 render 均已结束；未停止其他会话进程。

下一步若要求连准备阶段也完全连续，应专门调查首次重置前退出 POSE/回到 IDLE
的恢复瞬态及异步 reset 确认，不能仅根据原 expert 裁去准备阶段的视频推断没有此问题。

前次问题的证据：[原 expert 与迁移入口对比](g1-startup-vs-original-expert-20260920.md)、
[只修正 reset 向量的上一轮](g1-ghost-reach-fixed-reset-repeat-20260920.md)。
