# ScaleBFM 采集侧倾诊断（2026-09-08）

结论：当前 expert 的走路→伸手切换，平滑了 pad 的几何位置，却直接切换腕部旋转，并立即加入前馈偏置。这是伸手初期侧倾/扭腰的已验证诱因。mode 2 没有胸部或手肘约束，使控制器可以通过腰部补偿；采集成功判据又没有姿态质量门槛，导致这种动作被保留。

本次仅做诊断，未修改生产采集代码。所有新实验在 CPU 上运行，不使用 GPU、渲染器、录制栈或网络端口。没有改动现有数据集。

## 证据与复现

诊断目录：`outputs/diagnostics/scalebfm_sidelean_20260908/`。`probe.py` 调用实际 ScaleBFM checkpoint、MuJoCo 和 `run_bottle.run_round`，仅在内存副本中替换单个因素；`Probe` 只记录，不写 LeRobot 视频。每次重置仿真和策略历史。使用当前默认 edge 场景（自动选择 y+）、STAND_D 0.45–0.55、GRIP_KP=40、GRIP_KD=1，固定种子 1、2、3，每个种子取一轮。CPU 单线程，每轮约 3–5 秒。

运行对照：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PROBE_SEEDS=1,2,3 \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  outputs/diagnostics/scalebfm_sidelean_20260908/probe.py baseline smooth_orientation
```

快速侧倾检查（CPU FK，约 0.4 秒）：

```bash
CUDA_VISIBLE_DEVICES='' third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  outputs/diagnostics/scalebfm_sidelean_20260908/check_trace.py \
  outputs/diagnostics/scalebfm_sidelean_20260908/baseline/trace/wall_s0_01_state_trace.npz
```

实际输出：`FAIL: torso roll exceeds diagnostic threshold 15.0 degrees`，峰值 18.40°，发生在 3.48 s。同一命令改为 `smooth_orientation/trace/...` 输出 PASS，整轮胸部 roll 峰值 14.67°。15° 是本次诊断阈值，并非已确定的产品验收标准。

## 1. 腕部旋转突变是直接诱因

`src/teleop_free/scalebfm/run_bottle.py`：

- 140 行：预先构造最终 `R_palm`，包含站立 yaw 和固定 −13° pitch。
- 235–236 行：伸手几何路径从当前实测 pad 出发，用 min-jerk 插值位置。
- 158–159 行：一进入伸手就用最终 `R_palm`，同时加入 BIAS 和比例反馈；旋转没有插值。
- `run_spike.wrist_from_pad` 用 `pad - R_palm @ OFF` 计算腕位置，OFF 长 15.15 cm。因此旋转突变也会改变腕的位置目标。

当前 edge 场景三个种子在切换处，腕目标单 tick 旋转变化分别为 59.40°、60.58°、59.74°，位置变化为 11.81、11.54、12.13 cm。

单变量对照：仅把右腕旋转从伸手开始的实测旋转，沿相同 min-jerk 时间参数做 Slerp 到最终旋转；保持 mode、pad 几何路径、偏置、站位及其它逻辑不变。

| 设置 | 伸手胸部 roll 峰值，种子 1/2/3 | 伸手腰 yaw 峰值，种子 1/2/3 | 抓取成功 |
|---|---|---|---|
| 当前 expert | 18.40 / 17.20 / 19.93° | 25.95 / 21.25 / 24.87° | 1/3 |
| 仅平滑腕旋转 | 8.90 / 8.80 / 9.70° | 15.52 / 17.00 / 15.53° | 1/3 |

这证实腕部姿态切换显著贡献于伸手初期的侧倾和扭腰；它不能解释或消除所有阶段的歪斜。仅平滑旋转时，后续合爪/抬起仍有约 13–15° 胸部 roll 和约 35° 前倾。三轮也不足以评估稳定成功率。

![腕旋转单变量对照](../outputs/diagnostics/scalebfm_sidelean_20260908/wrist_rotation_ab.png)

## 2. 控制模式允许腰部补偿，但不能仅据此判定“mode 2 天生必然侧倾”

直接读取 `assets/scalebfm/mode_table.pt` 和 selected_body_names：

- mode 2 有效目标只有 pelvis、left_wrist_yaw_link、right_wrist_yaw_link。
- torso_link、肩、肘均被 mask。
- `Refs.pelvis` 的世界系目标只有 yaw，roll/pitch 为零；保持骨盆水平并不等于限制腰部侧弯。

旧 v4 第 0 轮保存的真实 qpos 在 3.82 s：胸部 roll 20.0°、骨盆 roll 2.2°、腰关节 roll 18.5°，主要是腰部侧弯。当前 edge 场景同样在伸手初期出现胸部侧倾。

特别注意：mode 6 启用 pelvis + 双肩/肘/腕，但 torso_link 仍然被 mask。因此直接换 mode 6 并不能显式锁定躯干；需要合理肩肘参考并验证效果。现有表中 mode 7 才包含 torso，但还会启用其它所有 body，不能只换编号而保留其它目标为零。

## 3. 站位朝向确实不符合右肩对齐意图，但不是已证实的侧倾主因

`STAND_LEFT=0.15` 的注释说把瓶子放到右肩前方。然而 136 行令 yaw 始终朝向瓶子，所以在目标骨盆 yaw 坐标中，瓶子横向分量严格为零。左移站位后又转向瓶子，抵消了预期的右侧对齐。

单独把 yaw 改为桌沿内法线，使瓶子保持在身体右侧 15 cm：三个种子的整轮胸部 roll 峰值变为 21.11 / 19.97 / 19.25°，成功 3/3。它可能改善抓取，但没有解决侧倾，不能把几何不一致当作已验证主因。结合平滑旋转后也没有得到可靠成功（0/3）。

其它排除探针：

- 去掉 BIAS：伸手胸部 roll 17.94 / 17.61 / 18.90°，仍侧倾，抓取 0/3。
- 等走完再伸手：伸手胸部 roll 19.77 / 16.69 / 20.08°，没有改善；腰 yaw 降低到约 11–12°。该组有一次 MuJoCo DOF 39 QACC 数值异常警告，因此不把其后续成功数作为可靠质量证据。
- 仅种子 1 的旋转+偏置渐入探索，伸手 roll 9.44°，抓取失败；没有进一步验证，不作为推荐修改。

## 4. 数据接受条件缺少身体姿态质量检查

审计 `outputs/experiments/bfm_pilot_v4_20260908/train_dataset/manifest.jsonl` 中所有 produced 轮次的 review qpos（合计 200 train + 10 val；旧 corner 几何）：

- 210 集接受数据中，199 集（94.8%）胸部绝对 roll 峰值超过 15°。
- 39 集超过 20°。
- 胸部 roll 峰值中位数 / P90 / 最大值：18.81 / 20.48 / 22.46°。
- 腰关节 roll 峰值中位数 / P90 / 最大值：22.22 / 24.01 / 25.10°。

逐轮数值存于诊断目录 `dataset_roll.json`，列为 round、胸部 max_abs_roll、腰关节 max_abs_roll，单位度。这里是旧数据审计，不与当前 edge 的三种子实验混作同一数据集。

`run_bottle.py` 308–312 行主要以抬高瓶子、瓶子倾角及前序失败标志决定 ok；`record_bottle.py` 301 行只据 ok 保留。统计里只有 torso_pitch_max_deg，没有胸部 roll、腰 yaw/roll 或动作突变的验收门槛。因此“成功抓起来”会把明显歪身的轨迹一并保留。

## 5. 没有证据表明是记录器把水平骨盆命令写歪

用旧 v4 第 0 集 parquet 中的 observation.root_orientation 将 action.bfm 骨盆目标旋转还原到世界系，最大 roll/pitch 只有约 2.4e−7 / 6.6e−7 度。与此同时真实 qpos 已经明显侧弯。当前 CPU 对照完全不经过 parquet 或 VLA，也能复现。

因此本症状首先应在 expert→ScaleBFM 的参考生成处处理，而不是用 VLA 解码器的 level-pelvis 去修采集数据。此前 eval 的后仰/重新锚定是另一个问题。

## 下一步建议

优先让走路→伸手的完整腕部命令连续：旋转轨迹、腕位置、偏置/反馈接入一起设计，特别要区分上一帧的命令和当前实测 pad，避免以实测值重新起步导致命令跳变。先用固定种子验证侧倾、腕部速度/加速度、抓取成功及夹爪稳定性，再考虑改变控制模式。

增加身体姿态质量指标和可配置的数据验收阈值，避免仅凭抓瓶成功扩大采集。目前的 Slerp 探针只证明原因，尚不是经过完整验证的修复；不建议直接用其扩大采集。
