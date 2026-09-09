# 桌前站位动态验证

本批用 `tools/diag_stance_grid.py` 扩大诊断路径的站位覆盖：脚本短距离走近、下蹲、规划手掌运输、受约束位姿反馈、原抓取伺服、抓起保持。默认采集器尚未切换到新运输/反馈路径。完整结果、实际站位图和原始记录见 [实验报告](../outputs/diag/stance_grid_20260906/REPORT.md)。

## 怎样解释站位

请求网格描述的是**下蹲前的停靠目标**，不是实际下蹲后的抓取站位。记录同时保存停靠后、下蹲后、专家交接后、运输开始时的目标相对骨盆坐标。下蹲前后的位置变化包含 `PlannerArm.start()` 的 0.6 s 接管保持及 `squat()`，本批不能将两段的影响分别归因。

实验在固定桌高、瓶子位置和朝向下，每个名义站位仅运行一次；部署进程每次重启，但没有完整控制器/仿真状态快照。不能据此画出连续可抓区域，也不能将名义网格单元成功当作该位置可靠。停靠 XY 偏差保留为测量字段；走近超时、朝向偏差超过 10°、跌倒、瓶子倾倒等拒绝均保留在分母中。`nominal_stance_met` 仅表示停靠 XY 偏差不超过 6 cm。

## 修复的诊断交接问题

原 `diag_crouched_handover.in_place_expert` 是从 VLA token 通道接入 Planner 的工具。新站位实验已经在 Planner 中，却复用了其朝向重设和 `exit_streamed`，导致额外转向。新增显式 `planner_active=True` 后保留已校准朝向，跳过这两项 token 退出操作；默认参数仍保留原 VLA 接管行为。两个回归测试分别约束两种入口。

另一个实验启动问题是 clean deploy 自动启动早于机器人 episode reset，朝向基准可能继承 reset 前状态。本工具先在空地保持并校准，再走向桌子。这是本实验的初始化处理；尚不能据此认定默认采集器中存在完全相同的错误。

## 后续验收顺序

1. 分开记录 `arm.start` 前后、下蹲开始/结束、稳定保持的实测 XY、yaw、高度、速度与时间。先区分接管瞬态和下蹲本身的漂移，再定义可交付的停靠/下蹲条件。
2. 在实际到达的身体状态上选择可跟踪的预抓取姿态。记录反馈是否触及参考修正上限；不能把静态 IK 可解当作 SONIC 可稳定交付，也不能靠放宽到位条件接受失败。
3. 将预抓取运输与接触抓取分别验收。到位之后仍可能在 `in` 阶段推倒瓶子，需要定位首次接触、手指/掌姿态、物体位移及实际发送周期。
4. 扩大上述有效样本、跨场景复测，再决定默认控制流迁移和数据重采。之后才做动作时间对齐、训练和独立 VLA 评估。3 m 起步时的目标可见性仍需单独处理。

## 复现边界

使用报告中保存的实验场景（不是任意当前 `outputs/scene/sim`）、encoder 开启的 clean deploy、instance 0。该脚本会 reset 该实例，必须使用专用实验实例。瓶子固定在桌 A 左侧世界 XY `[-0.45, 0]`；站位数组见报告目录 `cases.json`。每个输出目录拒绝覆盖，逐次 fresh deploy。

```bash
source scripts/paths.sh
# 同场景 stack 已启动；每个 case 前执行 restart_deploy。
CUDA_VISIBLE_DEVICES=0 INSTANCE=0 CLEAN_DEPLOY=1 NO_ENCODER=0 \
  bash scripts/restart_deploy.sh
CUDA_VISIBLE_DEVICES=0 INSTANCE=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$VENV_SIM_PY" src/teleop_free/tools/diag_stance_grid.py \
  --cases outputs/diag/stance_grid_20260906/cases.json --case 0 \
  --out outputs/diag/my_stance_run/planner_case_0
"$VENV_SIM_PY" src/teleop_free/tools/analyze_stance_grid.py outputs/diag/my_stance_run
```

分析器只汇总已写出 `result.json` 的 `planner_case_*`；未完成的批次应检查 `completed`，不能直接使用中途结果作为完整分母。捕获包含反馈参考、特权/部署状态和阶段事件（不是全部 Planner 命令的发送时间记录），不是已经可用于 VLA 训练的图像动作数据集。

## 分段复测（2026-09-06）

随后完成 6 次独立初始化：5 次姿态分段完整，1 次因接近朝向不合格拒绝。详见 [分段报告](../outputs/diag/posture_stages_20260906/REPORT.md)。有效原流程条在构造/接管期间移动 8.9 cm、下蹲期间移动 10.2 cm，接管前仍有 0.20 m/s 水平速度。

加入实测停稳门槛的两次复测，接管位移分别约 0.55/0.48 cm，但下蹲仍分别移动 11.1/10.4 cm。关闭下蹲手掌跟踪的一次试验仍移动 10.9 cm，不能把该水平变化仅归因于 IK。各组实际起始状态不同，这些是阶段定位证据，不是已量化的因果成功率改善。

`PlannerArm.start(upper_reference=...)` 现在允许显式保留调用方持有的 17 维冻结参考；默认入口保留实测关节种子。诊断的 `--settle-before-arm` 要求线速度与 yaw 差分速度连续达标，并保留等待中的身体位移；它尚未迁移至默认采集。`--preserve-reference` 只影响本诊断的首次接管，不代表后续 helper 接管已全部消除参考替换。

当前应建立实测停稳、实测蹲稳和蹲后几何的分开验收，而不是要求下蹲只改变 Z。3 cm 是本次几何保持诊断参考，不是已验证的自然下蹲运动界限。需要进一步测定可重复交付的蹲后站位并接回完整抓取。

## 连续抓取复测（2026-09-06）

已接通 `--continuous-grasp`：实测站稳、保留参考的一次接管、下蹲与实测蹲稳、同一控制器进入实测几何规划及原运输/反馈/抓取。正式 6 条均通过站稳和蹲稳，3 条抓起并保持；剩余为最终腕部前向下限拒绝、反馈无进展、动态净空拒绝各一条。详见 [连续抓取报告](../outputs/diag/continuous_grasp_20260906/REPORT.md)。

`in_place_expert(active_arm=...)` 借用控制器，调用方保留唯一清理责任；运输执行器也需显式保留已有参考，不能在内部重新把实测关节当作命令种子。实测抓取函数入口 XY 变化约 0.01–0.27 mm，不能将该短入口区间解释为全程身体漂移。

本批用绝对路径 `LOCOMANI_SIM_DIR` 为仿真、IK、诊断和 exporter 对象列表选择独立场景，没有写共享场景。此覆盖尚未接入所有旧采集/渲染工具。51 项测试通过；默认采集未启用新路径。运输参考循环实测仍约 41–45 ms，而名义 DT 是 20 ms，下一步应处理重复观测及参考速度一致性，再复测预抓取失败。

## 时序修复后的收尾结论

已修复运输过程内部重复阻塞观测。固定六次回归的运输调用中位周期约 21.0 ms（原约 41–45 ms），完整抓取仍为 3/6。时序修复已验证，任务可靠性尚未达标；没有继续放宽参数或扩大本轮实验。详细结果见 [报告](../outputs/diag/transport_timing_20260906/REPORT.md)，项目后续以 [验收收尾单](collection_acceptance.md) 为准，不再无限追加控制流重构批次。
