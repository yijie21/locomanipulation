# walk 场景：空地上只有一台 G1

> **已归档（2026-10-20）**：VLA 数据采集与训练这一系列暂不使用，这一页不再随代码更新，命令可能已经跑不通。现在的主线是[场景 RSI](rsi.md)。

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
| 刹车后双手能越线多远 | `scripts/experiments/brake_then_reach_20260923.sh` → `brake_then_reach_20260923.py`，渲染 `render_brake_reach_20260923.py` | 手最前点比骨盆远 65–69 cm；越线距离 = 这个值 − 刹车后骨盆到线的距离（28–39 cm）。依赖 `src/locomani/tasks/crate/ghost_reach_execution*.py`（已合入） |
| 刹车后伸到纸箱抓取点（虚拟箱子碰撞检查） | `brake_then_reach_20260923.sh --box --reach-path staged --below-rim 0.010 [--orientation-gain 1.0]`，碰撞检查 `box_contact_20260923.py` | 能伸到（误差 0.5–1.4 cm）；直接伸手 5/5 次穿过侧壁，`reach_path=staged` 在伸过去途中不碰；`orientation_gain 1.0` 使成功次数 3/5→4/5，朝向误差 19.6°→12.4°；手掌碰箱沿是因为下落时下冲 1–3 cm |
| 下落下冲、z 反馈、守线收紧（均未采纳） | `--descend-speed`、`--z-guard`（`visual.descend_speed` / `descend_z_guard`）；`diag_brake --stride-mode recent2 --extra 0.185` | 放慢下落 4/5、z 保护 3/5，都没能稳定避免手掌压箱沿；收紧守线 0/50 越线，但最近离线只剩 1.7 cm、停位离散翻倍。**现用方案**：守线刹车（余量 20 cm、安全 3 cm、最小步长 14 cm）+ `reach_path=staged` + 抓取深度 1 cm + `orientation_gain 1.0`，walk 场景 4/5 |
| 守线 `extra` 0.20 → 0.17（2026-09-24） | `brake_marker_20260923.sh --guard --speeds 0.25 --marker 0.9 --extra 0.17`，`outputs/diagnostics/brake_guard025_extra017_20260924` | 30/30 不越线；骨盆到线 −34.3 ± 3.5 → −31.0 ± 4.2 cm，最前点最近 −7.4 → −5.7 cm。均值只赚 3 cm，散布不变——守线参数榨不出更多 |
| waypoint 结尾、过量下发（2026-09-24） | `--waypoint slow --marker 0.9/1.2/1.5 --waypoint-rounds 0`，deploy `outputs/deploy_waypoint_20260923`，`outputs/diagnostics/waypoint_end_*_20260924` | 本体走目标的 0.64/0.64/0.71，停位散布 3.7/5.0/6.4 cm 随目标放大，~4.5 s 就停；折到 0.9 m 线：0.9 目标骨盆 −32 cm（与守线同），1.2 目标 −13 cm 但最前点 10/10 越线 +5 ± 2.7。比例不稳、没有守线，不能替代守线；脚错开 10 cm，09-23 的 3.9 cm 不复现 |
| planner 路点 | [路点补丁](../patches/sonic_waypoint_target.md)，`diag_brake.py --waypoint` | 停得很齐，但本体只走到目标的 35–55%，不能用作位置控制 |

| 守线步长模型 `alternate`（2026-10-01，TODO 2.4.10.23.3） | `diag_brake --guard --speeds 0.25 --marker 0.9 --extra 0.20 --stride-mode alternate`（`visual.guard_stride_mode`），`outputs/diagnostics/brake_guard025_alternate_20261001` | 30/30 不越线、0 摔，但骨盆停在线前 37.9 ± 4.8 cm，比 floor 的 34.3 ± 3.4 更远：并步后预测 0.23 比 floor 的 0.14 保守，抵掉了大步后多走的那步 |
| 按步态相位刹车（2026-10-01，TODO 3.14.1） | 标定 `diag_brake --speeds 0.55 --marker 1.40`（40 局，标记处刹车、相位随机），拟合 `scripts/experiments/fit_phase_band_20261001.py`；验证 `--phase-brake c0,a,b --margin M`；专家里是 `visual.guard_brake phase` + `guard_phase_band` | 0.55 稳态：带距（刹车时骨盆到最终最前点）38.6 ± 3.7 cm，相位一次谐波留一法残差 sd 2.4、最差 6.4 cm；余量取 3 + 6.4 cm 后 30/30 不越线、0 摔，最前点停在线前 10.3 ± 3.3 cm（最近 5.3），守线 0.25 是 15.5 ± 2.6。`outputs/diagnostics/brake_055_{phasecal,phasebrake}_20261001` |
| 相位刹车在短助跑下（2026-10-01） | 同上，`--marker 0.60` 标定、`--marker 1.03` 验证（= warehouse 这条边起点 1.03 m，空地只有 1.285 m） | 助跑 0.6 m 不到稳态：带距残差 sd 5.6、最差 8.3 cm；20 局最前点 −12.4 ± 8.8 cm、1/20 越线 3.3 cm，均值和守线 0.25 一样、散布大三倍；仓库 10 局 1/10。相位刹车只在助跑 ≥ 1 m 时比守线好。`outputs/diagnostics/brake_055_{phasecal_run060,phasebrake_run060}_20261001` |
| 非零起始朝向（2026-10-01，TODO 2.4.10.9） | `diag_brake --reanchor auto\|on\|off`（默认 auto：复位后偏航离 0 超过 5° 先重锚 planner 朝向） | warehouse 起始朝向 90°：off 2/2 局 40 s 超时不前进；auto 3/3 走到 0.3 m 标记，横向 ≤ 3 cm，`outputs/diagnostics/reanchor_{off,auto}_20261001` |
| 终点线刹车换控制器：SONIC 对 ScaleBFM（2026-10-13，TODO 24.7） | `scripts/experiments/wbc_compare_20261012/brake_batch.sh` → `bfm_brake.py --program marker\|plan`（ScaleBFM，同一空地场景）+ `brake_marker_20260923.sh`（SONIC 原样）→ `brake_summary.py`，`outputs/wbc_compare_20261012/brake/` | 越线刹车惯性 ScaleBFM 4.5 ± 0.6 / 2.5 ± 0.6 / 0.5 ± 0.4 cm（0.55 / 0.35 / 0.25）对 SONIC 24.7 ± 3.3 / 10.4 ± 4.9 / 4.0 ± 1.4，横向 sd ≤ 1.6 对 4–5 cm；不越线程序两边 0/10 越线，ScaleBFM 脚尖停线前 4–8 cm（sd ≤ 2）对 SONIC 守线 13–22 cm（sd 3.5–12）；见 [wbc-compare handoff](handoff/wbc-compare-20261012.md) §9 |

`diag_*` 都要求栈已经起在 walk 场景上：各 `.sh` 会自己起栈、收栈，只用 instance 0。
