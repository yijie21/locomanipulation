# Carry 停顿与关节连续性 — 2026-09-13

## 请求与实现范围

用户接受先合并重复等待、再限制最终腰部和双臂指令、最后做分组 GPU 对照的顺序。
这次修改 `bimanual_carry` 这条已被指出锁腰问题的历史控制路径。它仍使用仿真物体
位置；用于控制诊断，不等于符合视觉信息约束的 VLA 示范。
[RGB 专家](../crate_visual_collection.md) 仍是视觉指导入口，共用 `JointReference`。

- `carry.settle_seconds=0.3`：参考姿态到位后，连续观测到骨盆平面速度低于
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

三组各完成 30 次，30 个采样标签的起点/朝向/箱子横向命令完全一致。
`protocol_identity.json` 核对了三组已编译模型的初始状态、质量/惯量、几何、摩擦、
限位、驱动和约束数组，以及全部九次实际 deploy 的模型哈希和 `NO_ENCODER=0`。
这些身份相同，不代表异步执行轨迹相同。

| 指标 | 原流程 | 只合并等待 | 合并等待 + 连续轨迹 |
|---|---:|---:|---:|
| 成功 / 全部尝试 | 0/30 | 0/30 | 11/30 |
| 实际起始距离在 1.2–1.5 m 内 | 29/30 | 30/30 | 30/30 |
| 进入伸手阶段 | 11 | 13 | 12 |
| 走近结束至伸手交接中位数 | 1.874 s | 1.280 s | 1.171 s |
| 重复接管耗时中位数 | 0.620 s | 0.021 s | 0.027 s |
| 全部伸手回合的最大发送速度 | 149.026 rad/s | 93.329 rad/s | 0.700 rad/s |
| 发送加速度 P95 的回合中位数 | 661.061 rad/s² | 685.854 rad/s² | 2.000 rad/s² |
| 早期骨盆角速度 P95 的回合中位数 | 3.187 rad/s | 3.549 rad/s | 0.287 rad/s |

最后一组 12 次进入伸手，11 次成功；其余 18 次在站位检查拒绝，其中横向偏差
14 次、距离桌沿不足 13 cm 四次。唯一伸手后失败是箱子在闭合前被推移超过 5 cm。
成功集停稳后的桌沿距离为 13.43–26.20 cm，测得前俯为 0.08–21.78°；不是每次都
固定弯腰至上限。所有 30 次实际起始距离为 1.208–1.467 m。

仅缩短等待改善了交接耗时，未改善这批抓取和晃动表现。连续轨迹组同时观察到
参考速度/加速度受限、身体运动指标下降以及抓取完成数增加。整条任务成功率
仍只有 36.7%，不能称为高成功率采集；后续主要瓶颈是走近后的站位。
没有为提高数字而放宽站位、接触或成功门槛，也没有继续叠加补偿增益调整。

早期运动窗口最多 3 s；第二组 2 次、第三组 1 次在窗口结束前已失败。
仅保留覆盖至少 2.9 s 的回合时，三组各有 11 次，骨盆角速度指标分别为
3.187 / 3.583 / 0.291 rad/s，见 `window_coverage.json`。该附加检查不改变
成功率分母或失败分类。

计算入口：

```bash
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/crate_motion_report_20260913.py \
  outputs/crate_motion_flow_20260913
```

生成 `motion_report.json`。成功分母包含全部尝试，同时报告起始距离合格数。
交接耗时指标为已开始伸手的回合中 `brake_seconds + handoff_seconds`，不是主观视频打分。
指令加速度按相邻平均速度的时间中点计算，排除 5–100 ms 以外的间隔，并报告排除数。
身体指标将伸手最初 3 s 内的观测 seq 对齐到仿真轨迹，取骨盆角速度模长的 P95，
再比较各回合统计；该窗口可能包含接触，不等于纯自由空间上身抖动。

同 seed 保证采样命令一致，不能保证异步仿真的实际起姿、停步位置或接触完全一致。
不同组进入伸手的回合数也可能不同。报告用于检验控制趋势，不把差异直接解释为
单一因果机制，不把 30 次试验当作可靠的高成功率认证。

## 数据与后续边界

4 cm 是本次固定诊断距离，可继续作为抓取调试起点；2–6 cm 来自先前的 CPU
可达性筛选，尚未被证明是高成功率分布。本次没有继续扩大深度后声称“舒适区”。
夹持几何、实际站位和接触控制仍需独立设计。箱子不会在录制中被传送到更易抓的位置。

减少停顿、限制指令突变不能单独证明 VLA 不会停住。模型实际接收到的视觉、关节与
IMU 历史、动作/状态同步和闭环训练表现都需要验证。当前 RGB pilot 到正式 LeRobot
数据、token/history 消费者、双手推理及学生策略评估仍按原文档标为未完成。
不得用 carry 的 privileged 诊断示范替代这项信息可观测性验收。


## 最终代码检查与预览

最终源码 `ec3ba42` 在三组对照之后增加实际起距准入，并让伺服末态转储使用完整
video label、排他创建文件，避免批次 round 编号重用时覆盖证据。控制轨迹与
`4529eb1` 相同；补充 GPU smoke 单列，不能并入 30 次对照的成功率。
补充运行仍使用 GPU 0 / instance 1、4 cm 场景；各目录的命令、配置快照和 manifest 保留：

| 最终 smoke | 成功 | 失败 / 范围 |
|---|---:|---|
| `final_carry_smoke`，seed 11 | 1/2 | 一次横向站位 +9.2 cm；实际起距 1.335 / 1.460 m |
| `final_visual_smoke`，seed 101 | 1/2 | 一次 `approach_timeout`；两次 `intervention=false` |
| `final_wall_smoke`，seed 11 | 1/1 | 默认不走路的 wall 协议，不能代表全流程成功率 |

RGB 成功集从桌沿外 1.379 m 开始，失败集从 1.294 m 开始。它们仅检查共享代码的
执行，没有进行学生 VLA 训练。结束后已核对 GPU 0 释放、instance 1 锁与端口空闲；
GPU 1 的其他工作未停止。

`make check` 及提交检查通过，含格式、配置字段文档、依赖/导航约束和 CPU 测试。
新增消费端回归覆盖姿态/速度交接、非均匀控制间隔、配置验证与 CLI 交付、
实际起距拒绝、状态轨迹对齐及既有证据防覆盖。历史瓶子/墙抓取的物理等价性
结论不由本次小规模 smoke 重新认证。

在本次原流程 30 次 manifest 上重新 `scripts/gate.py record crate`，再对连续轨迹组
执行 `compare crate`，结果 PASS 4/4：成功率、朝向残差、主要走近出口及新增失败
类别四项通过，未改 gate 容差。日志为 `gate_comparison.log/json`。该 gate 未覆盖
carry 缺少的历史 wall 阶段字段，不能替代上面的轨迹指标或证明广泛物理等价。

预览仅展示同一采样标签 `bimanual_s11_02` 的原流程失败与连续轨迹成功，
全部 90 次结果仍以上表和 manifest 为准。完整保留走近至结束，正常速度：

| 预览文件（产物根目录下） | 字节数 | 原片 / 预览时长 |
|---|---:|---:|
| `previews/baseline_s11_02_small.mp4` | 402,532 | 14.32 / 14.30 s |
| `previews/bounded_s11_02_small.mp4` | 333,482 | 17.84 / 17.80 s |

两片均为 640×480、10 fps，已核对字节数、完整解码、起止画面和一帧以内的时长差；
记录在 `previews/verification.json`。成功示例从桌沿外 1.326 m 开始，停稳时
距桌沿 18.52 cm，抓取时测得前俯 6.37°，抬升 11.33 cm。原始视频保留。
