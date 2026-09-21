# 双臂前伸包络：机器人站着不动能把两只手送到哪

固定站姿、双脚不动、场景里没有箱子：给两只手一对幽灵目标（中点 x/y、高度 h、间距 d、连线偏角 yaw），
让原 expert 的全身 IK + SONIC 真去伸，量**两手位置误差随 x 上升到哪里穿过 2 cm**。
**这一页从头到尾能复现 2026-09-21 的那张图。** 为什么这么测、每一步量到了什么、哪些结论是推断，
在 [记录](handoff/g1-capability-envelope-20260921.md)；早先在 expert 默认配置下的五维搜索在
[五维联合实测](handoff/g1-joint-reach-search-20260920.md)，它的可达面比这里小一圈——原因见记录第 1 节。

先记住四个数：正对目标（y=0、yaw=0）边界 **0.58–0.62 m**（h 0.55–0.85）；**h<0.50 要下蹲**，蹲下后
h=0.40 到 0.58、h=0.30 到 0.48–0.53、h=0.20 到 0.38–0.43、**h=0.10 够不到**；yaw=+30° 比 −30° 差 5–10 cm。
这张图筛的是**位置**，姿态误差只记录不筛。

## 1. 准备一次

先做完 [安装指南](setup.md) 第 0–5 步。空地场景是采集用的双臂场景去掉桌子和箱子，物理完全一致：

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
"$VENV_SIM_PY" -m locomani.tasks.crate.reach_envelope.floor_scene outputs/scene/g1_floor
```

产出 `scene_locomani.xml`（只剩地面、灯、机器人、两个观察相机）和 `scene_meta.json`（`objects` 为空，
`robot_start.xy` 归零）。**不要用锁死底座的模型代替**：机器人要真的站着。

## 2. 一次搜索 = 一个目录

```bash
bash scripts/reach_envelope.sh outputs/reach_envelope/<name>
```

第一次运行会把默认 `search.json` 写进目录——**开跑前改它，两轮之间不要改**：

| 块 | 默认 | 含义 |
|---|---|---|
| `grid` | y {−0.12, 0, 0.12} × h {0.55, 0.70, 0.85} × d {0.32, 0.44, 0.52} × yaw {−30…30, 5 档} | 135 条线；x 不在网格里，是被搜的量 |
| `envelope` | `base_anchor=measured`, `com_shift_max` 0.11, `com_free_radius` 0.10, `pelvis_pitch_max` 0.85, `pelvis_height_min` 0.55, `knee_max` 1.5, `pelvis_forward_max` 0.35, `tracking_cap` 0.15, `tracking_gain` 1.0 | 全身 IK 支撑上限与跟踪反馈；字段含义见 [config_fields](reference/config_fields.md) 的 `CrateCarryConfig` |
| `envelope_overrides` | `[{h_below: 0.50, envelope: {base_anchor: measured_xy, pelvis_height_target: 0.65}}]` | h 低于阈值的线叠加这些字段：把 IK 的骨盆拉到 0.65 m（实测交付 0.57 m），高度不锚定实测值。每个覆盖组一份 bank，组间串行、组内分片并行 |
| `criteria` | 手位 2 cm、姿态 15°、保持 0.5 s、12 s 超时、脚漂移 3 cm | ghost 执行器的在线判据；`boundary_search` 只用**位置**做过/不过 |
| `scene_dir` | `outputs/scene/g1_floor` | 第 1 节的场景 |
| `shards` | `[{gpu 1, instance 0}, {gpu 0, instance 1}]` | 每个分片一个 SONIC 栈，并行 |

**`envelope` 是本页与默认 expert 唯一的区别。** 留空 `{}` 就是原 expert 的包络（0.45 m 处止步）。
它经 `probe_ghost_reach.py` 请求文件的顶层 `envelope` 块送达，dry-run 输出的
`controller.resolved_original_reach_control_config.carry` 里能核对每个值是否到位：

```bash
"$VENV_SIM_PY" scripts/probe_ghost_reach.py outputs/reach_envelope/<name>/round_0/shard_0/request.json \
  --output /tmp/x.json --shard-index 0 --shard-count 2 --dry-run | python3 -c \
  "import sys,json; print(json.load(sys.stdin)['controller']['resolved_original_reach_control_config']['carry'])"
```

## 3. 它怎么搜

每轮：`boundary_search plan n` 从前几轮的结果给每条线**一个** x，生成 `round_n/bank.json` 和每个分片一份请求；
分片并行跑完后进入下一轮，直到没有线需要采样。

- 第 0 轮从**离线 IK 上界往里 10 cm** 起步（`ik_upper_bound.json`，缓存）。离线 IK 用最终包络求准入，
  只会高估——2026-09-21 高估 10–15 cm——所以只拿它定起点，不拿它说可达。
- 过（最佳手位 ≤2 cm）→ x + 5 cm；不过或 IK 拒绝 → x − 5 cm；初始化失败 → 同点重试。
- 一条线在 [x_pass, x_fail] 夹住即停；到 0.20 m 仍不过的线记为无解，不再重试。
- 2026-09-21：h 0.55–0.85 135 条线、8 轮、386 次、1.5 h（两栈）；h 0.30–0.50 135 条线、9 轮、586 次。
- **下蹲**（`pelvis_height_target`）：不蹲时 h=0.30 骨盆只到 0.73 m，肩撞躯干，45 条线里 40 条到 0.20 m 仍失败；
  拉到 0.65 后骨盆实测 0.57 m，h=0.30 正对到 0.48–0.53、h=0.40 到 0.58，过点姿态中位从 20° 降到 13°。
  拉到 0.58 与 0.65 交付相同（腿已弯到底），取 0.65。h≥0.55 未测下蹲，所以默认只对 h<0.50 生效。
- **最低可达高度**（y=0，下蹲）：h=0.20 正对 0.38–0.43 m（d 0.44/0.32），h=0.10 六条线在 0.20 m 处仍差 4–12 cm——
  骨盆到 0.55–0.61 已是下蹲极限，再低的目标要跪或走进去。

进度随时看：

```bash
"$VENV_SIM_PY" -m locomani.tasks.crate.reach_envelope.boundary_search status outputs/reach_envelope/<name>
```

每行是一条线：`ik≤0.65  0.50:p1.3  0.55:f4.8` 读作离线上界 0.65，0.50 过（1.3 cm），0.55 不过（4.8 cm）。
`-> 0.60` 是下一轮要试的 x。`STOP` 文件让它在轮与轮之间停下；`--from N` 从第 N 轮续跑。

## 4. 读结果

```bash
S=locomani.tasks.crate.reach_envelope; R=outputs/reach_envelope/<name>
"$VENV_SIM_PY" -m $S.boundary_search table $R          # boundary_table.json：每条线的夹持区间 + 每次采样的连续量
MPLBACKEND=Agg "$VENV_SIM_PY" -m $S.plot_boundary $R    # boundary_slices.png：每个 h×d 一个面板，格 = 该 (y, yaw) 的边界 x
MUJOCO_GL=egl "$VENV_SIM_PY" -m $S.render_reel $R 0.70 reel_h070   # videos/reel_h070_source.mp4：15 格同屏回放
ffmpeg -i $R/videos/reel_h070_source.mp4 -vf "setpts=0.5*PTS,scale=1024:-2" -r 12.5 -an \
  -c:v libx264 -preset slow -b:v 150k -maxrate 180k -bufsize 360k -pix_fmt yuv420p $R/videos/reel_h070.mp4
```

`boundary_table.json` 每条线记 `x_pass`、`x_fail`、`ik_upper`，以及每次采样的 `pos`（最佳手位误差）、
`orient`（同帧姿态误差）、`drift`（脚漂移）、`reason`、`case`（原始 case 文件，含 50 Hz 轨迹）。
**阈值不烧死在数据里**：换 3 cm 或加姿态门槛，改 `POS_TOL`/加一个过滤重跑 `table` 即可，不用重跑仿真。

视频从 qpos 轨迹离线复现，幽灵手是真实 Dex1 手的网格放在目标位姿上，橙线是当前手到目标；
交付文件按仓库规则压到 500,000 bytes 以内（2 倍速 + 150 kbps 约 330 kB / 40 s）。

## 5. 均匀覆盖与复测

网格只覆盖任务会出现的 (y, h, d, yaw)。要出五维投影图或验证插值，另起一个 `boundary_refinement` /
`sobol` bank 用同一 `envelope` 跑（`scripts/generate_joint_reach_bank.py`，格式见
[五维联合实测](handoff/g1-joint-reach-search-20260920.md)）。2026-09-21 的 Sobol 200 + 复测 100 表明：
网格线性插值对随机点过/不过的准确率 79%；同一点复测 87/100 判决一致，翻转全在夹持两端，
**噪声地板约一格（5 cm）**——单次结果的边界要按 ±5 cm 读。

## 6. 资源

- 一次试验约 26 s 墙钟；两栈并行时每卡各一个实例（端口块与 DDS 域随 instance 分开）。
- 开跑前：`cat /sys/fs/cgroup/pids.current`（上限 3840，一个不录制的栈约 180）、`nvidia-smi`、
  `find /workspace/tmp -maxdepth 3 -name stack.json`（有别人的栈就别用那个实例）。
- 分片进程是 `probe_ghost_reach.py` 起的栈，每次退出自己收；驱动被杀时用 `scripts/stop_stack.sh --instance N`。
- 请求里的 `envelope` 块由 `ghost_reach_execution_cli.py` 解析；这段接线目前仍在工作树里未提交
  （与另一会话的分片/resume 改动交错），提交后本行删掉。

## 7. 这张图不说明什么

- **不是抓取成功率。** 位置过的点里姿态 ≤15° 的只有约四分之一（中位 20°）；夹爪能容忍多大姿态误差没标定。
- 站姿是 expert 初始化站姿，不是走完停下的站姿（脚错开、骨盆偏航没注入）。
- 没有箱体/平台碰撞；右侧（+yaw、−y）的劣势是控制器跟踪侧的不对称，根因未定位。
- 离线 IK（`offline_ik_sweep` 类的对照）对包络的预测是错的：每一条包络结论都要经控制器实测。
