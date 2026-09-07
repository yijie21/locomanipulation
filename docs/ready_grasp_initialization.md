# 双手准备姿势的数据与部署初始化

当前双手端起的采集配置，需要部署端使用匹配的初始姿态。旧的 `initial_pose.py` 默认 token 来自手臂放下的数据；它继续保留给旧 checkpoint。初始化兼容不等于抓取策略或 VLA 已经通过验收。

## 显式使用

在现有的 token-only、异步评估命令中追加：

```sh
--reach-config configs/collection/ready_grasp_measured_prediction.json \
--initial-token-file configs/deployment/ready_grasp_initial_token.json
```

`run_ready_pilot.py` 也接受 `--initial-token-file`，会在开始采集前验证文件，保存路径与 SHA-256，并传给评估。它不修改训练输入、动作维度或模型参数。配对比较的两个模型应使用同一初始化设置。当前 Goal 的准备脚本尚未实际训练：专家独立验收没有通过。

这个选项针对当前 SONIC release、G1/Dex1、65 维 token+gripper 路径；本轮只验证了异步客户端。没有验证其他机器人、SONIC checkpoint、18 维控制或 lockstep 初始化。文件缺省时，旧初始化行为保持原样。

## 执行顺序

专家采集继续通过自己的编码器准备双手。VLA 评估不再调用这个编码器专用动作：显式配置准备 token 后，先发送固定 token、张开双手并保持 3 秒，检查手臂是否到达原有 0.30 rad 容差内，再执行原有目标可见性检查、启动记录并进入 VLA 的正常初始化与推理。

固定 token 不依赖物体位置，不执行导航、下蹲、抓取或恢复。准备阶段发生在任务记录和第一次模型请求之前。可见性门槛、任务成功条件和碰撞规则没有放宽。

## 本轮证据

- 从候选07八条示范开头、第一次运动模式之前的 IDLE/张手记录中取得102个真实 token，以 L1 距离选中实际样本 medoid；没有合成 latent 均值，也没有使用独立确认集。
- 三次15秒保持：净位移4.4、6.7、4.5 mm，手臂最大偏差约0.115–0.117 rad，3秒后目标始终满足原有0.8/0.9视野门槛，没有跌倒。
- 原`spawn_off=0.94 m`名义起点（相对计划站位，并非桌沿距离）的两次实际部署入口测试均进入窗口执行；第一次请求时最大手臂偏差0.083–0.092 rad，目标在视野内。
- 这两次只回放已记录的专家 token，最终都没有完成接近/抓取。它们证明初始化和窗口路径可执行，不能作为 VLA 学习效果。
- 名义复位参数相同不等于完整初始状态相同；底盘位置和控制器历史仍可能不同。这个选项没有通过中途重置位置来掩盖差异。

完整记录：`outputs/experiments/policy_goal_20260906/ready_initialization_probe/` 与 `oracle_execution_probe_ready_start/`。当前源代码的10项文件/姿态/入口测试覆盖文件校验、旧默认值、先填充再切换、姿态未到达和可见性拒绝。
