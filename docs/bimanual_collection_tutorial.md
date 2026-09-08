# 双臂搬箱:跑 pilot、验收、渲染,以及离训练数据还差什么

> 怎么跑、产出在哪、什么算对。设计原因见 `docs/bimanual_grasp_design.md`,渲染原理见 `docs/rendering.md`。
> 所有路径相对仓库根目录;命令前先 `. scripts/paths.sh`。

## 0. 先说清楚这条流水线现在能给什么

| 产物 | 有没有 | 怎么来 |
|---|---|---|
| 每集第三人称视频、50 Hz 状态轨迹、manifest、expert 日志 | **有** | §4 pilot |
| MuJoCo / Isaac 并排视频 | **有** | §7 |
| **可训练的 LeRobot 数据集**(三相机 + 66 维 action) | **还没有** | expert `bimanual_wall.py` 没有接 exporter 的录制开关,见 §8 |

所以本文能复现的是"专家 10 集 ≥ 8 成功 + 视频";要出训练集还要做 §8 那一步。

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 环境 | 按 `docs/setup.md` 装好;`. scripts/paths.sh` 后有 `$VENV_SIM_PY`(sim + expert)、`$VENV_DATA_PY`(exporter/转换)、`$RENDER_PY`(SIMPLE venv,mujoco + EGL,可选,渲染和预检用) |
| SONIC deploy | 已编译(`$SONIC_DEPLOY_ROOT`) |
| **INSTANCE** | 决定端口(`5555 + 10·INSTANCE` 一组)和 DDS domain。**0 是采集/评测栈,pilot 永远用非 0**(本文用 2)。`INSTANCE` 必须在 `paths.sh` 之前设置——`paths.sh` 会把它默认成 0 |
| GPU | pilot:`CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1`(runner 默认);Isaac 渲染也在 GPU 1 |
| 资源 | 容器 pid 上限 3840(一个栈约 700);磁盘 ≥ 20 GB(轨迹 3 MB/集,Isaac 渲染 100 MB/集) |
| 别人的栈 | **不要用 `stop_stack.sh`**,它的 pkill 不分实例。停自己的用 §6 |

## 2. 场景

```bash
. scripts/paths.sh
$VENV_SIM_PY src/teleop_free/bimanual_scene.py outputs/scene/crate_bimanual
python3 -c "import json;m=json.load(open('outputs/scene/crate_bimanual/scene_meta.json'));\
print(m['surfaces']['A']['top'], m['objects']['blue_crate']['init_pos'])"
# 期望: 0.62 [-0.368, 0.0, 0.707]
```

含义:桌面 0.62 m;篮子近壁贴桌沿,篮沿顶 0.792 m ≈ 骨盆高度。篮子资产 `assets/crates/blue_crate/`
(NVIDIA 600×400 周转箱,按 2/3 缩放,**四根把手立柱已去掉**——这是成功率 1/30 → 20/20 的第一步,别换回带立柱的网格)。
可调:`SCENE_TABLE_TOP`、`CRATE_SETBACK`。**不要降桌子来躲桌沿**:0.45 m 桌面站立够不着,要蹲 17 cm,膝盖顶桌沿。

## 3. 预检(改几何或手臂姿势之后必跑;纯运动学,几十秒)

```bash
# a) 抓取可达:站位 × 深度 × 垫深 × 掌俯仰,IK 残差 < 1 cm 且无穿透
$VENV_SIM_PY src/teleop_free/tools/reach_wall_sweep.py | tail -30
#    期望:stand-off + depth ≤ 0.40 的组合全部 OK(掌俯仰 60);expert 用 REACH_SUM = 0.38

# b) 走路姿势:指尖到桌沿/篮壁的余量、ego 画面里篮子被手遮挡的比例
POSE_LEFT_DEG=10,10,0,60,0,30,0 MUJOCO_GL=egl $RENDER_PY src/teleop_free/tools/walk_pose_preflight.py
#    期望:四个站位 contacts 都是 '-';指尖在平面后 ≥ 0.12 m;遮挡 0%

# c) reach 曲线 + 预览视频
PW=0.02 OUT=0.15 UP=0.25 OVER=0.10 WARM=prev SAVE=outputs/reach/reach_preview/traj \
  MUJOCO_GL=egl $RENDER_PY src/teleop_free/tools/reach_preflight.py
MUJOCO_GL=egl $RENDER_PY src/teleop_free/tools/reach_preview.py
#    期望:contacts '-';越沿时指尖离篮沿顶 ≥ +2.5 cm;终点关节解 = hover 解
#    看 outputs/reach/reach_preview/reach_preview.mp4:上排跟踪相机、下排 ego,四列 = 0.13/0.18/0.24/0.31 m 站位
```

**动作像不像人,在 c) 的视频里判断,不要用 10 集 pilot 去发现。**

## 4. 跑 pilot

```bash
V=$LOCOMANI_ROOT/outputs/videos/wall_pilot; mkdir -p $V/trace
INSTANCE=2 WALL_FEATURE=rail WALL_WALK=1 SIM_STATE_TRACE_DIR=$V/trace SIM_VIDEO_DIR=$V \
  nohup bash scripts/run_wall_pilot.sh 10 0 > $V/runner.log 2>&1 &     # 参数:集数 种子
tail -f $V/pilot_s0.log | grep -E "^\[wall\] round|attempt|closing"
```

runner 做的事:起 instance 2 的整套栈(sim + deploy + agent server,**不起 exporter**),
跑 `bimanual_wall.py --rounds 10 --seed 0 --label wall_s0`,结束时只杀自己实例的进程。每集约 40–60 s。

| 产物 | 路径 |
|---|---|
| 第三人称视频 | `$V/wall_s0_NN.mp4` |
| 状态轨迹(50 Hz `qpos`/`qvel`,nq = 47) | `$V/trace/wall_s0_NN_state_trace.npz` |
| manifest(每集一行) | `$V/manifest.jsonl` |
| expert 日志(每 0.5 s 一行) | `$V/pilot_s0.log` |
| 栈日志(sim / deploy / agent) | `$(locomani_session_root 2)/{sim,deploy,agent}.log` |

## 5. 验收

```bash
python3 src/teleop_free/tools/summarize_wall_pilot.py $V
```

每集一行:`ok`、走位次数 `att`、闭合触发 `trig`、篮子首次移动时刻 `move_s`、集长、静止帧比例 `static`、站位 `so`、`yaw`、失败原因。

| 指标 | 合格 | 为什么 |
|---|---|---|
| 成功 | ≥ 8/10 | 历史:pilot 21 20/20、22 9/10、23 8/10 |
| `move_s` | = lift 开始时刻(约集长 − 4 s) | lift 前篮子不能被碰。pilot 18 的 20 集都在走路时被指尖扫转 7–14°,当时的"站位 yaw 门"失败其实是这个 |
| `trig` | 绝大多数 `contact` | `floor` 是兜底,`seated` 是毫米判据 |
| `static` 中位 | ≤ 0.10 | 画面不变且动作为零的帧比例,吸收态预算 |
| 站位 `|yaw|` 中位 | ≤ 3° | 篮子没被碰时步态自身的散布 |

## 6. 中途停止 / 清理

```bash
bash src/teleop_free/stop_instance.sh 2        # 只杀 instance 2 的 sim / deploy(按 --instance 与端口匹配)
for p in $(pgrep -f "[b]imanual_wall.py"); do kill -9 $p; done   # expert 本身;[b] 让模式不匹配这行自己
```

`pkill -f <模式>` 要从**单独的一条命令**里发——模式若出现在同一条 shell 命令行里会杀掉自己。
不要编辑正在运行的 bash 脚本(bash 按偏移量读),runner 改完要重启。

## 7. 渲染并排视频(可选)

```bash
bash scripts/build_isaac_assets.sh --snapshot outputs/scene/crate_bimanual        # 场景变了才需要;rm .stamp 可强制
mkdir -p outputs/renders_isaac/my_run
bash scripts/render_wall_pilot.sh $V outputs/renders_isaac/my_run 10 wall_s0 > outputs/renders_isaac/my_run/render_all.log 2>&1
```

每集约 3 min。产出 `outputs/renders_isaac/my_run/sbs_roundNN.mp4`(左 MuJoCo、右 Isaac,同一条相机轨迹);
每集目录 `rNN/` 有 `tracking_mujoco.mp4`、`cam_traj.npz`、`ego_isaac.mp4`、`render.log`。
第 0 集出来先看三样:只有一个机器人、有头、脚在地面上。

## 8. 到训练数据集还差的一步

现状:`bimanual_wall.py` 只写视频/轨迹/manifest,**不控制 exporter**,而 exporter 只在 `INSTANCE=0` 的栈里启动。
要出 LeRobot 数据集,需要:

1. 在 `bimanual_wall.py` 的每集里接 exporter 开关,和单手 `grasp_gym.py` 一样:任务开始(settle 之后)`p.cmd.key("c")` 开始录,
   成功再 `key("c")` 保存,失败 `key("x")` 丢弃;manifest 每行带 `ok` / `produced`(转换器按行和 session 里的 episode 一一对应)。
2. 在 **INSTANCE=0** 的栈上跑(`start_stack.sh` 默认 `--record-wrist-cameras`,三路相机都录);录之前确认 instance 0 没有别人的采集或评测。
3. 转换:`$VENV_DATA_PY src/teleop_free/tools/convert_to_hf_format.py --sessions <session> --manifest <manifest.jsonl> --out <ds> --holdout N --holdout-out <ds>_val --jobs 4`
   (视频流从 session 自动发现;`left_gripper` 键已在转换器里),然后 `check_dataset.py <ds>`。
4. 训练用 `configs/gr00t_sonic_3cam_bimanual.py`(ego + 两腕相机,state 60,action 64 + 1 + 1)。

## 9. expert 的阶段与判据(改参数时对照)

| 阶段 | 判据 | 常量(`bimanual_wall.py`) |
|---|---|---|
| 出生 | 站位后 1.0–1.3 m,横向 ±0.15,朝向 ±10° | `WALK_SPAWN_BACK`,`WALK_SPAWN_LAT` |
| 走入 | 经 waypoint(站位后 0.35 m)直线到站位;**手垂体侧** | `WALK_SIDE_LEFT_DEG`,`STAND_OFF_WALK = 0.24` |
| 停站 | 骨盆速度 < 0.06 m/s 持续 0.25 s | |
| 站位门 | yaw ≤ 10°,stand-off 0.13–0.32,横向 ≤ 8 cm;出带则退回 waypoint 再走,≤ 3 次,**整段录下** | `WALK_ATTEMPTS`,`WALK_STAND_OFF_BAND`,`WALK_LAT_TOL` |
| reach | 一条 Bezier:体侧 → 上提外张 → 越沿 → hover,2.5 s,IK 带姿态先验,**无停顿** | `REACH_S`,`REACH_UP/OUT/OVER`,`REACH_POSTURE_W` |
| 下探 | 参考点 0.04 m/s 下降到篮沿顶下 5 cm | `DESCEND_SPEED`,`GRIP_BELOW` |
| 闭合 | **contact**:参考仍在下降、实测高度 0.3 s 不变、横向在容差内、手垫到篮沿顶;兜底 seated / floor | `CONTACT_STALL_S/MM`,`ACROSS_TOL`,`FLOOR_DWELL` |
| 抬起 | IK 目标上升 12 cm,找平 + 俯仰跟随 | `LIFT_HEIGHT`,`PITCH_FOLLOW_*` |
| 判定 | 上升 ≥ 6 cm、倾斜 ≤ 20°、双手接触 | `LIFT_MIN`,`TILT_OK_DEG` |

三条原则(各对应一次失败):阶段切换只用画面/本体感知里可见的量;导航阶段不预备抓取姿势(前伸扫篮沿、端在胸前挡 ego);停偏了不丢弃,退回重走并录下。

## 10. 已知坑

- **deploy 的 LowState 保护**:一次 80–100 ms 的周期卡顿就会让 deploy 自行退出,之后每集 "did not stabilise"。同机别的会话起栈/重启 deploy 会触发。expert 连续两集不稳定会自动 `scripts/restart_deploy.sh`;看 `$(locomani_session_root 2)/deploy.log`。
- **Isaac**:`/World/G1` 占位机器人、ego 相机要隐藏的头部网格、地面残差埋脚——三个都已修;RTX 首帧后网格还要 ~10 帧才加载完;出生点在后墙里就调 `isaac_scene.py` 的 `BG_SHIFT_XY`。
- **IK 分支**:无姿态先验时胸前高度的目标会解到手甩身后的分支,跟踪曲线时手腕漂到 90–110° 扭转分支;`posture` / `posture_weight` 就是为此。

## 复现清单

1. §2 场景:桌面 0.62、篮子 x = −0.368。
2. §3 三个预检无接触,预览视频动作连续。
3. §4–5 10 集 pilot:成功 ≥ 8,`move_s` 都在 lift,`static` ≤ 0.10。
4. §7 渲染第 0 集:一个机器人、有头、脚着地。
5. 要训练集:先做 §8。
