# 双臂搬箱数据采集:复现教程

> 从零到"10 集完整 episode + MuJoCo/Isaac 并排视频"的全部步骤,以及每一步的验收数字。
> 设计思路和踩过的坑见 `docs/bimanual_grasp_design.md`;渲染管线的原理见 `docs/rendering.md`。
> 本文只讲怎么跑、看什么、什么算对。所有路径相对仓库根目录。

## 0. 这条流水线产出什么

一集 episode = 机器人从 1.0–1.3 m 外走到桌前,停稳,两只手抓住篮子(NVIDIA 600×400 周转箱,
按 2/3 缩放)的两条长边上沿,抬起 10 cm 以上并保持。每集产出:

| 产物 | 位置 | 说明 |
|---|---|---|
| 第三人称录像 | `$SIM_VIDEO_DIR/wall_s0_NN.mp4` | sim 自带的骨盆跟踪相机 |
| 状态轨迹 | `$SIM_STATE_TRACE_DIR/wall_s0_NN_state_trace.npz` | 50 Hz `qpos`/`qvel`,nq = 47,渲染和事后分析都靠它 |
| 一行 manifest | `$SIM_VIDEO_DIR/manifest.jsonl` | 成功/失败、站位、闭合触发、重走位次数、各阶段耗时 |
| expert 日志 | `$SIM_VIDEO_DIR/pilot_s0.log` | 每 0.5 s 一行的阶段状态 |
| (可选)并排视频 | `outputs/renders_isaac/<run>/sbs_roundNN.mp4` | 左 MuJoCo、右 Isaac,同一条相机轨迹 |

expert 是 `src/teleop_free/bimanual_wall.py`(`WALL_FEATURE=rail WALL_WALK=1`),跑在 GR00T
WholeBodyControl 的 SONIC G1 部署上,MuJoCo 做物理。**采集数据时 VLA 相关的相机、state、
action 布局不变**,见 `docs/bimanual_grasp_design.md` 第 0 节。

## 1. 环境前提

1. 仓库按 `docs/setup.md` 装好:`third_party/GR00T-WholeBodyControl/.venv_sim`(sim + expert)、
   SONIC deploy 已编译、`scripts/paths.sh` 能 source。
2. Isaac 渲染(可选)需要 `scripts/setup_render_runtime.sh` 装好的 SIMPLE venv,
   `paths.sh` 里的 `RENDER_PY` 指向它。
3. **实例编号**:`INSTANCE` 决定端口(`5555 + 10·INSTANCE` 等)和 DDS domain。
   `INSTANCE=0` 是采集 / 评测栈,pilot 永远用非 0 的实例(本文用 2)。
   `stop_stack.sh` 的 `pkill` 不分实例,**不要用它**;`scripts/run_wall_pilot.sh` 退出时只杀自己实例的进程。
4. GPU:pilot 用 `CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1`(runner 默认),Isaac 渲染也在 GPU 1。
5. 容器 pid 上限 3840,磁盘要留 ≥ 20 GB(每集轨迹约 3 MB,一轮 Isaac 渲染约 100 MB)。

## 2. 资产与场景

篮子资产在 `assets/crates/blue_crate/`。**四根把手槽立柱已经去掉**(碰撞体和
`visual/blue_crate_noposts.obj` 都去掉了),两条长边上沿是 30 mm 厚的"墙",抓取点沿墙的
深度是自由参数。这是成功率从 1/30 到 20/20 的第一步,不要换回带立柱的网格。

```bash
. scripts/paths.sh
$SONIC_ROOT/.venv_sim/bin/python src/teleop_free/bimanual_scene.py outputs/scene/crate_bimanual
python -c "import json;m=json.load(open('outputs/scene/crate_bimanual/scene_meta.json'));\
print(m['surfaces']['A']['top'], m['objects']['blue_crate']['init_pos'])"
# -> 0.62 [-0.368, 0.0, 0.707]   桌面 0.62 m,篮子近壁贴桌沿,篮沿顶 0.792 m(≈ 骨盆高度)
```

两个环境变量可以改几何,默认值就是验证过的那套:`SCENE_TABLE_TOP`(桌面高度)、
`CRATE_SETBACK`(篮子近壁离桌沿的距离)。**不要为了让手臂避开桌沿去降桌子**:
桌面 0.45 时站立可达差 5–15 cm,要蹲 17 cm,膝盖又会顶桌沿(2026-09-07 实测)。

## 3. 三个离线预检(改任何几何或手臂姿势之前先跑)

都是纯运动学(`mj_forward`),不跑 policy,几十秒出结果。用 `RENDER_PY`(有 mujoco + EGL)跑。

```bash
# 3a. 墙顶抓取可达扫描:站位 × 深度 × 垫深 × 掌俯仰,IK 残差 < 1 cm 且无穿透的组合
$SONIC_ROOT/.venv_sim/bin/python src/teleop_free/tools/reach_wall_sweep.py | tail -30
# 期望:所有 stand-off + depth ≤ 0.40 的组合 OK(掌俯仰 60);expert 用 REACH_SUM = 0.38

# 3b. 走路姿势预检:指尖相对桌沿/篮壁平面的余量、ego 相机里篮子被手遮挡的比例
POSE_LEFT_DEG=10,10,0,60,0,30,0 MUJOCO_GL=egl $RENDER_PY src/teleop_free/tools/walk_pose_preflight.py
# 期望:四个站位 contacts 都是 '-';指尖在平面后 ≥ 0.12 m;遮挡 0%

# 3c. reach 曲线预检 + 预览视频
PW=0.02 OUT=0.15 UP=0.25 OVER=0.10 WARM=prev SAVE=outputs/reach/reach_preview/traj \
  MUJOCO_GL=egl $RENDER_PY src/teleop_free/tools/reach_preflight.py
MUJOCO_GL=egl $RENDER_PY src/teleop_free/tools/reach_preview.py   # -> outputs/reach/reach_preview/reach_preview.mp4
# 期望:四个站位 contacts '-',越沿时指尖离篮沿顶 +2.5 cm 以上,最终关节解等于 hover 解
```

预览视频上排是跟踪相机、下排是 ego 相机,四列是 0.13 / 0.18 / 0.24 / 0.31 m 四个站位。
**动作看起来像不像人,在这一步判断,不要用 10 集 pilot 去发现。**

## 4. 跑 pilot

```bash
V=$LOCOMANI_ROOT/outputs/videos/wall_pilot; mkdir -p $V/trace
INSTANCE=2 WALL_FEATURE=rail WALL_WALK=1 SIM_STATE_TRACE_DIR=$V/trace \
  nohup bash scripts/run_wall_pilot.sh 10 0 > $V/runner.log 2>&1 &
tail -f $V/pilot_s0.log | grep -E "^\[wall\] round|attempt|closing"
```

参数:集数、随机种子。runner 会起 instance 2 的整套栈(sim + deploy + agent server),
跑完自动只停自己的进程。跑完后:

```bash
python src/teleop_free/tools/summarize_wall_pilot.py $V
```

每集一行:`ok`、走位次数 `att`、闭合触发 `trig`、篮子首次移动时刻 `move_s`、集长、
静止帧比例 `static`、站位 `so`、`yaw`、失败原因。**验收标准**:

| 指标 | 合格 | 出处 |
|---|---|---|
| 成功率 | ≥ 8/10 | pilot 21 20/20,pilot 22 9/10,pilot 23 8/10 |
| `move_s` | 等于 lift 开始的时刻(集长 − 4 s 左右) | 篮子在 lift 前不能被碰。pilot 18 的 20/20 集都在走路时被指尖扫转 7–14°,这就是当时"站位 yaw 门"失败的真正原因 |
| `trig` | 绝大多数 `contact` | `floor` 是兜底;`seated` 是毫米判据 |
| `static` 中位数 | ≤ 0.10 | 图像不变且动作为零的帧比例;吸收态的预算 |
| 站位 `|yaw|` 中位数 | ≤ 3° | 篮子没被碰之后,步态自己的散布 |

想中途停:**只能 SIGKILL**,而且要从单独的命令里发(`pkill -f` 的模式会匹配到自己所在的
shell 命令行):

```bash
for p in $(pgrep -f "[b]imanual_wall.py"); do kill -9 $p; done   # [b] 让模式不匹配这行本身
```

之后 sim/deploy 还在,用 runner 里 `stop_mine` 的逻辑(按 `--instance 2` / 端口)杀掉。

## 5. 渲染并排视频

```bash
bash scripts/build_isaac_assets.sh --snapshot outputs/scene/crate_bimanual   # 场景变了才需要;rm 掉 .stamp 可强制
bash scripts/render_wall_pilot.sh $V outputs/renders_isaac/my_run 10 > outputs/renders_isaac/my_run/render_all.log 2>&1
```

每集约 3 分钟。第 0 集出来后先看三样:只有一个机器人、有头、脚在地面上。
每集目录 `rNN/` 里有 `tracking_mujoco.mp4`(MuJoCo 重渲)、`cam_traj.npz`(逐帧相机位姿)、
`ego_isaac.mp4`(Isaac 在同一相机轨迹上的回放)、`render.log`。

## 6. expert 的阶段与判据(改参数时对照)

| 阶段 | 触发 / 判据 | 关键常量(`bimanual_wall.py`) |
|---|---|---|
| 出生 | 站位后方 1.0–1.3 m,横向 ±0.15,朝向 ±10° | `WALK_SPAWN_BACK`,`WALK_SPAWN_LAT` |
| 走入 | 经 waypoint(站位后 0.35 m)直线走到站位;**手垂在体侧** | `WALK_SIDE_LEFT_DEG`(右臂镜像),`STAND_OFF_WALK = 0.24` |
| 停站 | 骨盆速度 < 0.06 m/s 持续 0.25 s | |
| 站位门 | yaw ≤ 10°,stand-off 0.13–0.32,横向 ≤ 8 cm;**出带则后退到 waypoint 再走,最多 3 次,整段录下** | `WALK_ATTEMPTS`,`WALK_STAND_OFF_BAND`,`WALK_LAT_TOL` |
| reach | 一条 Bezier 曲线:体侧 → 上提并外张 → 越过篮沿 → hover 点,2.5 s,IK 带姿态先验,**无停顿** | `REACH_S`,`REACH_UP/OUT/OVER`,`REACH_POSTURE_W` |
| 下探 | 参考点以 0.04 m/s 下降到篮沿顶下方 5 cm | `DESCEND_SPEED`,`GRIP_BELOW` |
| 闭合 | **contact**:参考还在下降、实测高度 0.3 s 内不变、横向在容差内、手垫已到篮沿顶;兜底 seated / floor | `CONTACT_STALL_S/MM`,`ACROSS_TOL`,`FLOOR_DWELL` |
| 抬起 | 开环 IK 目标上升 12 cm,找平 + 俯仰跟随(篮子被硬夹在平口里会随手俯仰) | `LIFT_HEIGHT`,`PITCH_FOLLOW_*` |
| 判定 | 上升 ≥ 6 cm、倾斜 ≤ 20°、双手都在接触 | `LIFT_MIN`,`TILT_OK_DEG` |

三条原则,每条都对应一次失败:

- **阶段切换只用画面或本体感知里可见的量**。contact 闭合看的是"手停了",不是毫米误差;
  hover 没有固定等待。
- **不要在导航阶段预备抓取姿势**。手前伸(pilot 18)会扫到篮沿,手端在胸前(pilot 21)
  会把篮子从 ego 画面里挡掉。G1 的肘关节垂手时正好在篮沿高度,小臂 0.30 m,
  所以任何从垂手出发的关节空间抬手都会扫过篮沿平面——reach 必须是笛卡尔曲线,
  肘要向外张开。
- **停偏了不丢弃,重新走位并录下**。数据里要有"停错 → 退回 → 再来"。

## 7. 已知坑

- **deploy 的 LowState 保护**:任何一次 80–100 ms 的控制周期卡顿都会让 SONIC deploy
  自行退出,之后每集都"did not stabilise"。同机另一个会话重启 deploy、起栈都能触发。
  expert 连续两集不稳定会自动 `scripts/restart_deploy.sh`;看 `$LOGS/deploy.log`。
- **不要编辑正在运行的 bash 脚本**(bash 按偏移量读);runner 改完要重新启动。
- **相对路径**:所有脚本用绝对路径,`mv` 错目录会让 `&&` 链静默中断。
- **Isaac 三个渲染坑已修**:场景里的 `/World/G1` 占位机器人(会被地面弹飞)、ego 相机
  才需要隐藏的头部网格、重建地面的残差把脚埋进去;以及 RTX 首帧后网格还要约 10 帧才加载完。
  出生点若在房间后墙里,调 `isaac_scene.py` 的 `BG_SHIFT_XY`。
- **IK 分支**:`arm_ik.solve` 没有姿态先验时,胸前高度的目标会解到手甩到身后的分支;
  跟踪曲线时手腕会漂到扭转 90–110° 的分支。`posture` / `posture_weight` 就是为此加的。

## 8. 复现清单

1. `bimanual_scene.py` 重建场景,验证桌面 0.62、篮子 x = −0.368。
2. 三个预检全部无接触。
3. 10 集 pilot,`summarize_wall_pilot.py`:成功 ≥ 8,`move_s` 都在 lift,`static` ≤ 0.10。
4. 渲染第 0 集,确认一个机器人、有头、脚着地。
5. 视频交付前看一遍 reach 段:一个连续动作,没有分阶段的停顿。
