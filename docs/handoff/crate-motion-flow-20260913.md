# Carry 停顿与关节连续性 — 2026-09-13

## 请求与实现范围

用户接受先合并重复等待、再限制最终腰部和双臂指令、最后做分组 GPU 对照的顺序。
这次修改 `bimanual_carry` 这条已被指出锁腰问题的历史控制路径。它仍使用仿真物体
位置；用于控制诊断，不等于符合视觉信息约束的 VLA 示范。
[RGB 专家](../crate_visual_collection.md) 仍是视觉指导入口，共用 `JointReference`。

- `carry.settle_seconds=0.3`：参考姿态到位后，连续观测到骨盆平移速度低于
  0.06 m/s、角速度低于 0.2 rad/s 才允许伸手。重复帧不能累计稳定时间，
  大于 0.1 s 的观测间隔重启计时，6 s 内不满足则保留为失败。
  0.3 s 是连续稳定窗口，并不承诺机器人走完后总共只等 0.3 s。
- `carry.joint_speed=0.7` rad/s、`carry.joint_acceleration=2.0` rad/s²：
  一条 `JointReference` 覆盖走近时抬臂、刹停、伸手、闭合和提起。
  限制作用于 IK 和现有补偿之后的最终命令，SONIC 同时收到相符的参考速度。
  控制交接保留位置和速度，不重新采样姿态或附加固定 hold。
- 关节目标受模型与调用方限位共同约束。初始已发送姿态若位于更窄的任务限位外，
  逐步进入限位，避免第一帧直接截断造成突跳。腰部前俯上限仍为 0.4 rad；
  侧弯和扭转逐步回到中立。闭合阶段使用固定的保持目标，避免追逐自身滤波输出。
- 保留原有横向、桌沿、可达性、双手捕获、箱体位移、接触、抬升和倾斜判据。
  这组对照没有调整补偿增益。wall 未启用 carry 的轨迹配置；其他 PlannerArm
  调用方保留默认接管行为。
- 重置在桌沿外 1.2–1.5 m 采样，同时记录站稳后的实际距离。发现重置漂移后，
  最终入口增加实际距离准入：超出范围在录制/行走前拒绝，原因保留在 manifest。
  三组对照使用同一个原始采样协议；报告也列出其实际距离合格数。

配置归属 [carry_config.py](../../src/locomani/tasks/crate/carry_config.py)，从
`CrateTaskConfig` 经 CLI/profile/override 一次解析后传入回合。没有用于保留旧版的
实验开关；对照通过冻结的源码版本运行。

## 协议与证据

工作分支 `fix/crate-flow-20260913`，基础版本 `e3a1fba`。
产物根目录 `outputs/crate_motion_flow_20260913/`，保留全部失败、原始视频和轨迹。

| 组 | 冻结版本 | 产物目录 |
|---|---|---|
| 原流程 + 共同采样/记录 | `ac7d0fa` | `baseline/` |
| 只合并等待 | `8ba1428` | `continuous_handoff/` |
| 合并等待 + 连续轨迹 | `4529eb1` | `bounded_motion/` |

每组 30 次，每 10 次重启栈；seed 11、111、211，机器人横向采样 ±15 cm，
箱子横向采样 ±10 cm。固定场景 `scene_depth04/` 从原 crate 场景复制，仅将篮子
初始 x 向桌内移 4 cm 并同步元数据，不改质量、接触或桌面。GPU 0 / instance 1，
SONIC v1.1，`no_encoder=False`。GPU 1 为其他工作，三组顺序执行，非锁步仿真。

共同的启动形式如下，工作树必须与组的版本一致，输出使用新的目录：

```bash
LOCOMANI_ROOT=<group-worktree> PYTHONPATH=src \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
  -m locomani.pipeline.run_crate_pilot --expert carry \
  --rounds 30 --batch-rounds 10 --seed 11 --gpu 0 --instance 1 \
  --scene outputs/crate_motion_flow_20260913/scene_depth04 \
  --out outputs/crate_motion_flow_20260913/<group>
```

各组 `config_snapshot.json` 记录代码/场景/配置哈希；`deploy_receipts.jsonl`
记录实际模型身份，`commands.jsonl` 记录命令。`*_motion.jsonl` 是发送参考、
速度、状态时间戳和阶段记录；`trace/*_state_trace.npz` 是仿真 qpos/qvel 原始轨迹。
诊断记录禁止覆盖已有同名文件。这些字段没有被自动声明为 VLA 输入。

初始失败测试已保存：`red.log` 为站稳后仍固定等待 1.00 s，
`red_reversal.log` 为首步肩关节发送速度 25 rad/s。
回归命令：`make test TESTS="tests/test_crate_motion_flow.py tests/test_crate_joint_reference.py"`。
测试覆盖实际发布通道、反向目标、非均匀间隔、交接延续速度、初始限位外姿态、
不稳定/重复观测和配置到回合的交付。

## 结果与计算

GPU 对照仍在进行；本节在完成后填入结果，不据部分运行宣称成功率提高。

计算入口：

```bash
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/crate_motion_report_20260913.py \
  outputs/crate_motion_flow_20260913
```

生成 `motion_report.json`。成功分母包含全部尝试，同时报告起始距离合格数。
停顿指标为已开始伸手的回合中 `brake_seconds + handoff_seconds`，不是主观视频打分。
指令加速度按相邻平均速度的时间中点计算，排除 5–100 ms 以外的间隔，并报告排除数。
身体指标将伸手最初 3 s 内的观测 seq 对齐到仿真轨迹，取骨盆角速度模长的 P95，
再比较各回合统计；该窗口可能包含接触，不等于纯自由空间上身抖动。

同 seed 保证采样命令一致，不能保证异步仿真的实际起姿、停步位置或接触完全一致。
不同组进入伸手的回合数也可能不同。报告用于检验控制趋势，不把差异直接解释为
单一因果机制，不把 30 次试验当作可靠的高成功率认证。

## 数据与后续边界

4 cm 是本次固定诊断距离；2–6 cm 来自先前的 CPU 可达性筛选，尚未被证明是
高成功率分布。如果 4 cm 的闭环表现不可靠，不继续扩大深度后声称“舒适区”。
夹持几何、实际站位和接触控制仍需独立设计。箱子不会在录制中被传送到更易抓的位置。

减少停顿、限制指令突变不能单独证明 VLA 不会停住。模型实际接收到的视觉、关节与
IMU 历史、动作/状态同步和闭环训练表现都需要验证。当前 RGB pilot 到正式 LeRobot
数据、token/history 消费者、双手推理及学生策略评估仍按原文档标为未完成。
不得用 carry 的 privileged 诊断示范替代这项信息可观测性验收。
