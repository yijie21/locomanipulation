# walk 场景：空地上只有一台 G1

给 SONIC 下的 G1 测运动性能用：MuJoCo 原生蓝地板，除了机器人什么都没有。场景名 `walk`（名字表 `configs/scenes.yaml`）。

## 构建

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
"$VENV_SIM_PY" -m locomani.tasks.walk.scene outputs/scene/walk
```

构建过程是 [触达包络的空地场景](reach_envelope.md#1-准备一次) 再加一台相机，所以机器人（双手 dex1、手指垫、夹爪齿比）
与 crate / warehouse 采集线的物理完全一致。`scene_meta.json` 里 `objects` 为空、`robot_start.xy` 为原点。

## 相机

| 相机 | 用途 |
|---|---|
| `walk_follow` | 视频默认相机（`video_camera`）。`trackcom` 模式，始终在机器人质心右侧 3 m、高 1 m 处横看，走多远都在画面中间；不随机器人转身 |
| `reach_overview` / `reach_side` | 固定在原点附近，机器人走出一米多就出画 |
| `head_camera` / 两个腕相机 | 机器人自带 |

跟拍检查图：`outputs/scene_previews/walk/follow_check.png`（原点、x=4 m、(4, 3) 转 90° 三个位置）。

## 已知

- XML 里 pelvis 在 x=-1.5（沿用 crate 场景的起点），reset 才把机器人放到 `robot_start`。跟拍相机的位置按 XML 里的 pelvis 算，
  改起点位置不用动相机。

## 探针（2026-09-23，结果目录均在 `outputs/diagnostics/`）

| 问题 | 入口 | 结论 |
|---|---|---|
| 小角度、小距离能不能做 | `scripts/experiments/walk_micro_20260923.sh` → `backends/sonic/diag_micro.py` | 3–20° 转向能做准；平移是一格一格的，要么不动要么冲出一截；`walk_to` 在短距离上判断的"已到达"不可信 |
| 指令速度 → 实测速度 | `scripts/experiments/speed_ramp_20260923.sh [--mode N]` → `diag_speed_ramp.py`，渲染 `render_speed_ramp_20260923.py` | SLOW_WALK ≤0.15 不走，0.20–0.50 不稳，≥0.55 线性；mode 18 更差，mode 20 ≥0.45 干净但有迟滞 |
| 走到线前刹车、绝不越线 | `scripts/experiments/brake_marker_20260923.sh --guard` → `diag_brake.py`，渲染 `render_brake_marker_20260923.py` | 按落脚点预测刹车，全程 0.25 时 50/50 不越线，身体最前点停在线前 15.5 ± 2.6 cm；两脚前后错开约 13 cm，切模式或延迟刹车都收不齐 |
| 刹车后双手能越线多远 | `scripts/experiments/brake_then_reach_20260923.sh` → `brake_then_reach_20260923.py`，渲染 `render_brake_reach_20260923.py` | 手最前点比骨盆远 65–69 cm；越线距离 = 这个值 − 刹车后骨盆到线的距离（28–39 cm）。依赖协作者尚未提交的 `ghost_reach_execution*.py` |
| 刹车后伸到纸箱抓取点（虚拟箱子碰撞检查） | `brake_then_reach_20260923.sh --box --reach-path staged --below-rim 0.010 [--orientation-gain 1.0]`，碰撞检查 `box_contact_20260923.py` | 能伸到（误差 0.5–1.4 cm）；直接伸手 5/5 次穿过侧壁，`reach_path=staged` 在伸过去途中不碰；`orientation_gain 1.0` 使成功次数 3/5→4/5，朝向误差 19.6°→12.4°；手掌碰箱沿是因为下落时下冲 1–3 cm |
| planner 路点 | [路点补丁](../patches/sonic_waypoint_target.md)，`diag_brake.py --waypoint` | 停得很齐，但本体只走到目标的 35–55%，不能用作位置控制 |

`diag_*` 都要求栈已经起在 walk 场景上：各 `.sh` 会自己起栈、收栈，只用 instance 0。
