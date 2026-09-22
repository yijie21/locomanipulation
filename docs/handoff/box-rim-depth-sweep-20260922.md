# 箱沿线站位深度扫描（2026-09-22）

`outputs/box_rim_sweep_depth_20260922`：`scripts/box_rim_sweep.py`，模板 `outputs/box_rim_execution_20260920/near_stance_supported.request.json`
（`approach_mode` supported_pose，调整预算 0.30 m），机器人起点沿朝向 `robot_dxy[1]` 七档 {0.075 … 0.375} m（0.225 = 09-20 的 near stance；
箱心距 0.74 → 0.44 m），每档 10 局，每局横向 ±2 cm、偏航 ±3° 的种子抖动（seed 20260922），七档轮转，单栈 GPU1，20 s/局，70 局 25 min。
判据是这条线自己的试提；同时记录退出阶段。

| dy (m) | 箱心距 (m) | ok | 退出阶段 | 原因 |
|---|---|---|---|---|
| 0.075 | 0.74 | 0/10 | planning 9, reposition_settle 1 | no_coarse_ik_candidate 9, 预算超限 1 |
| 0.125 | 0.69 | 0/10 | planning 8, reposition_settle 2 | no_coarse_ik_candidate 8, 预算超限 2 |
| 0.175 | 0.64 | 0/10 | planning 8, reposition_settle 2 | no_coarse_ik_candidate 8, 预算超限 2 |
| 0.225 | 0.59 | 0/10 | planning 8, reposition_settle 2 | no_coarse_ik 6, no_static_supported 1, 预算超限 2, stale_observation 1 |
| 0.275 | 0.54 | 0/10 | planning 10 | no_coarse_ik 7, no_static_supported 2, unsupported_transit_waypoint 1 |
| 0.325 | 0.49 | 0/10 | planning 10 | no_coarse_ik 9, no_static_supported 1 |
| 0.375 | 0.44 | 0/10 | planning 9, ready_escape 1 | no_coarse_ik 6, no_static_supported 2, initial_escape 1, stale_observation 1 |

## 读法

- **深度不是判别变量。** 61/70 局在规划阶段就没有候选（`no_coarse_ik_candidate` 52、`no_static_supported_collision_free_candidate` 9），
  七档一样；抖动也不解释（规划成功的 8 局 |yaw| 中位 2.0°，失败的 1.5°，横向散在整个 ±2 cm）。规划器从**停稳后的实测姿态**出发
  （`planner.samples_per_rim` 5、`top_k` 4、`coarse_iterations` 500）几乎找不到抓点；09-20 那次能选到候选属于这 8/70 的少数。
- **规划成功的 8 局全部倒在调整走动上。** 目标站位都在箱心 0.515 m 处（y −0.905），需要走 ≤0.22 m，实际路径 0.30–0.31 m：
  步态以 0.45 m/s 刹车的过冲，不是距离。`maximum_total_reposition_m` 0.30 对这条走法就是上限。
- 因此 A 阶段对这条线的结论是：**在做站位/偏航随机化之前，先要有一个从实测姿态能出候选的规划器**，其次是把调整走动的刹车纳入预算
  （或把预算按 0.30 + 过冲放宽、并测过冲分布）。横向/偏航扫描（问题 3 的第三步）没有意义，未跑。
- 视频：`reel_first_pass.mp4`（七档各一局，27 s）；每局 `videos/<name>.mp4`；每局事件在 `rounds/<name>/result.json`。

## 更正（同日晚）：规划时机器人没朝着箱子

状态轨迹（`outputs/box_rim_diag_cam_20260922/state_trace`，从复位后第一帧起）显示复位后偏航是 110–130°，不是请求的 90°±3°；
篮子线偏航正确（0–5°），所以这个偏差只在起始朝向 90° 的场景出现。规划阶段机器人朝左偏 20–40° 看箱子，这才是 61/70 局
"无粗 IK 候选"的原因；进入调整走动后步态把朝向拉回 86–96°，所以走过的 8 局反而能规划。偏航在仿真里是直接写 qpos 的
（`privileged_pub.py:534`），偏差发生在 `box_rim_execution_cli.py:219–233` 的 reanchor + 4.5 s IDLE hold 期间，
与 memory `sonic-planner-heading-anchor`（重锚定时的偏航旋转 facing）一致，未逐行定位。

验证局（`dy0.225_r00`，`scene_reverse_camera`）：走 0.18 m 在预算内、规划成功、进到 `transit_raise` 才因右踝/地面
`observed_collision_during_motion` 停下——比扫描里任何一局都远。

另：扫描视频主画面是骨盆跟踪相机（`video_camera` 缺省），在这个房间里落在机器人背后的墙内；模板要指向
`outputs/box_rim_execution_20260920/scene_reverse_camera`（XML 相同，只多 `video_camera: reverse_room`），
`scene_xml` 也要同步改。执行器报的碰撞没有一次是平台：都是右手指 vs 右髋的自碰撞，或右踝 vs 地面。

## 未做

- 规划器为什么从实测姿态出不了粗 IK 候选：把 `rounds/dy0.325_r00` 的 `planning_snapshot` 喂给离线规划器复现，是下一步。
- 刹车过冲的分布（8 局 0.30–0.31）没有单独量。
