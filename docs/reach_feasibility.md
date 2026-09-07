# 桌面接近与运输可行性

## 当前接口与配置

`grasp_gym.py --reach-config configs/reach_tabletop_v1.json` 加载显式配置；不传时使用 `reach_config.DEFAULT_REACH`。配置 schema_version=1，每次执行的最终配置写入 `execution.metrics.reach_config`。

配置集中声明下蹲前深度范围、下蹲后深度范围、预计后移、深度余量、最大方位角、接近超时、下蹲上限，以及运输检查的间隙/采样步长。默认深度与下蹲参数沿用上一批，不扩大随机停靠范围。当前 WBC 标定下不允许将 max_drop 提高到 0.13 m 以上。非法数值、空深度交集和不支持的配置版本在创建控制对象前拒绝。

下蹲前范围按后移先验反推，与配置的 pre_min/pre_max 求交；下蹲后的实际测量与重规划仍是必要检查。参数文件不提供跨桌高、跨物体的成功保证。

## 运输检查做什么

正常专家采集为当前 TabletopTarget 创建 `TransportGuard`，使用目标和桌面 body 名称以及 IK 的场景模型。在 up/down 运输阶段，每条参考发送前，从实际测得的关节姿态到候选参考做关节空间线性分段检查。下蹲后仍使用实际状态；腿部在筛查中保持测得位置，腰部和主动臂随候选参考变化。最后一个 down→in 交接步也检查。

默认采样间隔不超过任一关节 0.04 rad，间隙 0.004 m。查询覆盖主动臂的肩、肘、腕和手指碰撞几何，对目标以及配置中的桌面生成接触。它持有独立 MjModel/MjData，既不修改仿真，也不修改 IK 的工作数据。动态目标位姿和手指位置每次用测量值更新。

首次出现间隙不足时返回 `transport clearance in ...`，该候选参考不发送，执行走现有失败清理路径。manifest 记录次数、最大查询时间，以及拒绝对应的部位、障碍物、间隙、插值位置和采样总数。

**这是一项局部参考运动筛查，不是已规划好的无碰撞全身轨迹。** 它不预测平衡控制器的实际响应、不验证身体移动、非主动臂、所有其他场景物体和自碰撞，也不覆盖导航/下蹲以及进入物体后的 in/close/lift。后面几个阶段需要允许合法抓取接触，不能直接套用“禁止所有接触”。测量中的关节与世界位姿存在时间误差，采样也不是连续碰撞证明；4 mm 是本次筛查余量，不是经过鲁棒性证明的界限。失败清理仍可能引起身体运动。

发生拒绝时目前退出该次尝试，不自动抬高手、后退或绕行。旧接口/直接 `_servo_continuous` 诊断调用只有明确传入 guard 才启用；正常采集默认启用，VLA token 执行没有插入专家真值保护。判断是否实际启用应查看 `transport_guard_enabled`，不能只看参数默认值。

## 为什么不用距离零值直接判碰撞

本轮在本地发现某些 mesh 对的 `mj_geomDistance` 零值与同状态的碰撞接触生成不一致。已撤出该原型，没有把这些零值当作真实碰撞证据。最终使用 `mj_collision` 生成带明确 margin 的接触，通过解析几何测试验证间隙和路径中间碰撞。距离接口的算法背景见 [MuJoCo 官方说明](https://mujoco.readthedocs.io/en/latest/computation.html#geom-distance)。

## 验证与可复现命令

```bash
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -m unittest discover -s tests

third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  src/teleop_free/tools/sweep_reach_feasibility.py \
  --state tests/fixtures/crouched_pregrasp.json \
  --meta outputs/diag/transition_fix_20260905/experiment_scene/scene_meta.json \
  --xml outputs/diag/transition_fix_20260905/experiment_scene/scene_locomani.xml \
  --out outputs/diag/my_reach_grid.json
```

扫描改变桌高、深度、横偏和 yaw，可用 `--shoulder-pitch-offset` / `--elbow-offset` 检查不同初始手姿。输出是固定身体姿态下的规划、IK 和关节路径筛查；其中运输采用静态最终朝向的 waypoint，**不是实际 servo 的朝向插值和闭环 rollout**。`accepted` 仅表示这些静态检查通过，不能直接喂给随机停靠采样器并宣称可抓。

本轮 24 项测试通过；正常采集第一批 3 次尝试中 2 次成功、1 次出生点转向超时，最后版本另 1 次成功。三次成功的运输查询次数为 144 / 132 / 122，单次最大 7.92 / 2.38 / 7.81 ms。此时没有生产路径拒绝样本；拒绝行为由 MuJoCo 中途碰撞测试和真实 servo 发送前拒绝测试覆盖。没有完成“同一完整仿真状态、有无 guard”的动态因果对照，不能声称已证明避免了碰倒。

未录制新数据、未训练 VLA。原始证据与限制见 `outputs/diag/reach_feasibility_20260906/REPORT.md`。

## 第三批：路径规划与实际交付诊断

新增 `JointTransportPlanner`，在静态测量姿态中先检查直线，受阻时做有预算的双向关节搜索。返回的每条边均经过 TransportGuard。该模块目前仅由 `diag_crouched_handover.py --transport-strategy planned` 使用；`baseline` 保留原运输，`guard` 仅增加筛查，方便分别记录结果。`--transport-feedback` 是独立的有界关节补偿实验，不能默认开启。

执行器每次发送前按实际状态重新筛查，限制参考速度及身体/目标漂移，最终检查实际手掌位置和朝向。中间 waypoint 用关节误差确认进度；最终 waypoint 使用任务空间到位判据，避免冗余 IK 中不同关节解导致“末端已到但一直等待”。到位后，原 servo 再次验证实际预抓取姿态，才能从 in 阶段继续。

本批动态诊断发现：原运输 3 次介入中 2 次抓起保持；guard 组 2 次介入中 1 次成功，另有 1 条在 VLA 前缀超时。静态路径初版 3 条均获得直接路径，但全部被跟踪判据阻断：其中两条末端误差未达标，第三条末端已达标却被关节条件过度约束。补偿初版也没有成功，不能用静态可行路径替代控制器交付验证。当前绕行搜索只有解析 MuJoCo 障碍场景测试，尚无真实机器人动态绕行成功样本。

这些组没有相同完整仿真/控制器快照，不能把比例差异归因于单一开关。详细结果、终点判据修正后的复测及命令见 `outputs/diag/transport_dynamic_20260906/REPORT.md`。尚未扩大随机停靠范围，尚未用本批实验重新训练 VLA。
