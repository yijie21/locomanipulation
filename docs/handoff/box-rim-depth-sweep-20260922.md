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

## 修复（同日）：重锚定之后再放一次、朝请求的偏航

探针（`outputs/box_rim_diag_cam_20260922/probe_init`）逐步采样：复位后 92–96° 稳定 3 s；`reanchor()` 的流式切换本身把机器人
转到 126°（记录 `heading_offset` 126°）；非 coordinated 路径随后按**实测**偏航保持，把 126° 锁死。按请求偏航保持能拉回 101°，
再复位一次并按请求偏航保持得到 93–97°（偏移量在传送后仍有效）。修法（`box_rim_execution_cli.py` `_initialise` pose=None 分支）：
reanchor 后再 `reset` 一次、`hold_yaw` 与 facing 用请求值——coordinated 路径靠第二次 reset 早就是这样。验证局
`outputs/box_rim_diag_fix_20260922` 进入规划时偏航 99°（此前 111–130°；残差 ~9° 是 IDLE 的收敛，探针里也是 96–101°）。

修复后重跑 70 局（`outputs/box_rim_sweep_depth_v2_20260922`，`scene_reverse_camera`）：前 14 局里"规划无候选"从 12/14 降到 1/14，
失败挪到后面的闸门——`stale_observation` 3、`unsupported_transit_waypoint` 4、`initial_escape_*` 3（右手指/右髋）、
`transit_tracking_error` 2、预算超限 1。完整表见下。

## 单集抓取尝试（同日，`outputs/box_rim_grasp_*_20260922`、`box_rim_coordinated_20260922`）

修好朝向后在 dy 0.225（箱心距 0.59 m，规划器给的候选不用走）按用户要求逐闸门放开、每次 1–3 集：

| 放开的闸门 | 集数 | 走到 | 拦下的 |
|---|---|---|---|
| 无 | 2 | `transit_raise`（手到箱前 1.2/1.8 cm） | 脚锚点 5.7 mm > 5 mm；重心 1.6 cm > 1.5 cm |
| 脚锚点 2 cm、重心 3 cm | 1 | `transit_above`（手张开悬在沿口上方） | 路点位置误差 4.5 cm > 3.5 cm |
| + 路点容差 6 cm | 4 | 3 集 `ready_escape`（右手指/腕 vs 右髋 0.02–0.1 mm 的"新接触对"），1 集 above_rim 5 个路点全过后左脚/地面 −7.2 mm 被判碰撞 | 逃逸簿记只认起始几何对；脚压地基线上限 6 mm |
| + 脚压地 12 mm、逃逸允许包络内的擦碰（代码） | 3 | 2 集 transit（手 3.4–6.1 cm 外） | **脚锚点 2.6–6.1 cm**：手伸到 0.70 m 高的沿口时站立控制器在挪脚；1 集擦碰加深 0.16→0.31 mm |
| `coordinated` 模式（VisualApproach + CoordinatedReach，09-20 的迁移路径） | 3 | 走入到 0.41–0.44 m、准入通过 | 第一帧 reach **参考**位姿手指与箱壁碰撞（`non_pad_target_collision`，实测无接触）——与 09-20 相同 |

结论：朝向修好后，`supported_pose` 这条线剩下的限制不是闸门而是物理——静态规划假设双脚不动，SONIC 在双臂前伸到 0.7 m 时会挪脚 2–6 cm，
后续每个路点都在这个位移上累计误差；`coordinated` 模式的抓点几何（手指贴外壁进入）对这个纸箱在参考层就碰壁。
两条都不是再放一个阈值能过的。逃逸擦碰的代码放宽（同包络内的新接触对按起始接触追踪，须清零）保留，测试覆盖。
视频：`outputs/box_rim_grasp_relaxed_20260922/videos/dy0.225_r00.mp4`（手悬在沿口上方那集）。

## `coordinated` 模式的接触规则（同日晚，`outputs/box_rim_coordinated{2..6}_20260922`）

用户要求"预计算说可达就驱动去抓"。五维包络的执行器本身拒绝带物体的场景（`ghost reach requires scene_meta.objects={}`），
能用同一控制器（CoordinatedReach）抓真箱子的只有 `coordinated` 模式。逐条放开它在参考层的碰撞规则，每次 3 集：

| 改动 | 参考层被拒的接触 |
|---|---|
| 指垫只许贴目标壁 → 手指几何可擦任何目标壁（`finger_wall_policy=any_wall`，1.5 mm 内） | 指尖链节入壁 3.2 mm |
| 参考层允许入壁到壁厚一半（5 mm） | 指节入壁 5.5–6.5 mm |
| 站位 0.39 → 0.55 m（手起始时在箱体投影外） | **腕部**（`wrist_yaw_link`）入侧壁 |
| 参考层允许入壁到壁厚（10 mm；实测层仍 1.5 mm） | 仍是腕部入侧壁 |

每次都倒在 reach 的第 2 步、实测无接触。轨迹显示抓点在侧壁前端 3.7 cm 处、离沿顶 6 mm（`x:-0.123`、深 0.006），
手从正前方沿壁线进入，规划出的手腕就骑在壁顶上——这是抓点定义/进入方向的几何问题，不是再放一档能过的。
另外 6 集里 3 集在走入阶段就被 0.30 m 行走预算拦下：站位差 4 cm 也会走出 0.32 m（刹车过冲），和 supported 模式一样。
代码保留 `any_wall` 策略与参考层壁厚容差（仅 coordinated 模式生效，supported 模式规则未变）。

## 未做

- 规划器为什么从实测姿态出不了粗 IK 候选：把 `rounds/dy0.325_r00` 的 `planning_snapshot` 喂给离线规划器复现，是下一步。
- 刹车过冲的分布（8 局 0.30–0.31）没有单独量。
