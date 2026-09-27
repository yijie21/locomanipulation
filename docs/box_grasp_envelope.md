# 双臂抓箱：默认配置、成功率随布局参数的变化、取值建议

这一页回答三个问题：现在双臂抓纸箱的**默认版本**是哪份配置；**每个布局参数**怎么影响"抓起并保持"的成功率（2026-09-26/27 共约 1,700 局实测）；
布场景和随机化时**该怎么取值**。方法和运行方式见[场景能力扫参](scene_sweep.md)，逐次数字见[台账](ledger/crate.md) 2026-09-26/27 各节。

## 1. 默认版本

`configs/experiments/box_platform_default_20260927.yaml`，场景名 `box_platform`（`configs/scenes.yaml`）：

| 项 | 值 | 来源 |
|---|---|---|
| 场景 | 蓝地板 + 10 cm 实心台 + 纸箱 0.45（宽）× 0.32（长）× 0.60（高）m，1 cm 壁、1.2 kg、摩擦 1.4，贴台边、偏角 0，无障碍物 | 扫参中心格；箱子物理与 warehouse 纸箱相同 |
| 专家 | warehouse 的 visual 专家（`box_room_grasp_200_cap120_20260925` 的全部旋钮）+ `ready_after_brake: quick` | 刹车后准备手臂 13.3 → 7.9 s，成功率不变（26/60 vs 28/60） |
| 每局随机 | 起点离台边 0.6–2.0 m、朝向 ±20°、横移 −8～0 cm | 第 2 步各维单独测出的范围 |
| 判据 | 抬起 + 躯干直立 + 保持 1 s（`retreat_distance 0`） | 用户 2026-09-26 定 |
| 实测 | 全维度同时随机 **24/60 = 0.40**（区间 0.29–0.53），四种箱子几何 6/15、5/15、3/15、10/15 | `box_range_quick_*_20260927` |

```bash
. scripts/paths.sh && export TMPDIR=/workspace/tmp SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
# 场景（已生成在 outputs/scene/box_platform；重建命令在 configs/scenes.yaml）
bash scripts/start_stack.sh collect       configs/experiments/box_platform_default_20260927.yaml --gpu 0
bash scripts/bimanual/run_agent_server.sh configs/experiments/box_platform_default_20260927.yaml --gpu 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/box_platform_default_20260927
```

成功局样片：`outputs/videos/box_range_quick_20260927.zip`（24 段，quick，全维度随机）、`outputs/videos/box_safe_range_20260927.zip`（11 段，扫参里按维度挑的）。

出训练数据、训练和评测：命令与核对表同 [warehouse §5–§6](warehouse.md#6-训练与评测)（把 `my_box` 换成本运行）。图像两种都能选（`images.source`）：`mujoco` 补渲用 `scripts/render_dataset_cameras.sh`、评测时 sim 自己出图；`isaac` 另要本场景的 Isaac stage（`configs/scenes.yaml` 里 `box_platform` 的 `isaac.build`，从 MJCF 的盒子生成，另加一块 MuJoCo 没画的灰地面）、补渲用 `_isaac.sh`、评测开镜像，`backend.render` 保持 `offline`。`eval.initial_token` 用 `configs/deployment/warehouse_initial_token.json`（本配置 60 局的开局手臂与它差 ≤ 0.03 rad，与 crate 那份差 0.6 rad）。判据 `retreat_distance 0`，闭环没有"后退"一列。冒烟：`box_platform_isaac_smoke_20260927`（采集 → Isaac 补渲 → convert → Isaac 录像 → 镜像 50/50）。

## 2. 参数定义与符号

| 参数 | 配置项 | 符号 |
|---|---|---|
| 箱高 / 长 / 宽 | 场景（`locomani.tasks.crate.box_platform_scene --height/--length/--width`） | 长 = 沿走入方向，宽 = 两手所抓两面壁的间距；抓点高度 = 台高 + 箱高 − 1 cm |
| 箱子离台边（setback） | `task.visual.object_setback_range` | 前壁在台边之后多少米 |
| 箱子偏角 | `task.visual.object_yaw_range_deg` | 俯视逆时针为正 |
| 起点离台边 | `task.visual.start_edge_min / start_edge_max` | 骨盆到台边 |
| 起点朝向 | `task.visual.start_yaw_range_deg` | 相对走入轴，左转为正；走近时专家仍命令走入轴 |
| 起点横移 | `task.visual.start_lateral_range` | 相对箱子中线，机器人在**左侧**为正；guard 走法不修横向 |

每局的实抽值写在 manifest 的 `placement` 里（`start_yaw_offset_deg`、`start_lateral_m`、`object_setback_m`、`object_yaw_offset_deg`）。

## 3. 成功率随参数的变化

每格 10–30 局，95% 区间约 ±15 个百分点；"合适"指不明显低于中心格的 43–47%。

### 3.1 箱子几何（第 1 步网格，宽 0.45，起点 1.2 m 正对）

| 箱高（抓点高度） | 长 0.20 | 长 0.32 | 长 0.44 |
|---|---|---|---|
| 0.30（0.39 m） | 3/15 | 0/10 | 0/10 |
| 0.45（0.54 m） | **15/30** | **12/30** | 1/10 |
| 0.60（0.69 m） | **13/30** | **13/30** | 2/15 |
| 0.75（0.84 m） | 0/10 | 3/15 | 0/10 |

箱宽（中心几何）：0.30 → 7/30，0.38 → 7/30，**0.45 → 13/30**，0.55 → 3/25。

### 3.2 箱子相对台边（中心几何，每局抽样后按实抽值合并）

| setback | 0 | 0–2.5 cm | 2.5–5 cm | ≥ 5 cm |
|---|---|---|---|---|
| 成功率 | **0.43** | 0.31 | 0.24 | 0/10 |

| 箱子偏角 | −7～−3.5° | −3.5～0° | **0°** | 0～3.5° | 3.5～7° | > 7° |
|---|---|---|---|---|---|---|
| 成功率 | 0.08 | 0.18 | **0.43** | 0.31 | 0.18 | 0.07 |

### 3.3 机器人起点（中心几何）

| 起点离台边 | 0.6–0.8 | 0.8–1.0 | 1.0–1.2 | 1.2 | 1.2–1.5 | 1.5–1.75 | 1.75–2.0 |
|---|---|---|---|---|---|---|---|
| 成功率 | 0.53 | 0.33 | 0.53 | 0.43 | 0.17 | 0.30 | 0.60 |

无单调趋势，合计 59/160 = 0.37：走近阶段把起点距离抹平了。

| 起点朝向 | −30～−20° | −20～−10° | −10～−3° | −3～3° | 3～10° | 10～20° | 20～30° |
|---|---|---|---|---|---|---|---|
| 成功率 | 0.26 | 0.48 | 0.42 | 0.40 | 0.29 | 0.33 | 0.27 |

| 起点横移 | −16～−8 cm | −8～−4 | −4～0 | 0 | 0～+4 | +4～+8 | +8～+16 |
|---|---|---|---|---|---|---|---|
| 成功率 | 0.08 | **0.46** | **0.44** | 0.42 | 0.15 | 0.24 | 0.08 |

横移明显不对称：机器人在箱子中线右侧 8 cm 内和正中一样，左侧偏一点就掉（右腕坐标系不是镜像的，见 memory `dex1-right-palm-frame-not-mirrored`）。

### 3.4 真正起作用的中间量：停位深度

几何参数都是通过"刹车后骨盆到箱子的距离"起作用的（H ≥ 0.45 的局合并）：

| 停位深度 | 0.30–0.36 | 0.36–0.39 | 0.39–0.42 | 0.42–0.45 | 0.45–0.48 | 0.48–0.52 | ≥ 0.52 |
|---|---|---|---|---|---|---|---|
| 成功率 | 0.25 | **0.46** | **0.49** | 0.42 | 0.25 | 0.08 | 0/10 |

箱长每加 2x、setback 每加 x，手就要多伸 x；箱长 0.44 让停位落到 0.51 m，所以全部失败。太近（< 0.36）则预备姿势碰到前壁。

## 4. 取值建议

| 参数 | 建议 | 备注 |
|---|---|---|
| 抓点高度 | **0.54–0.69 m** | 10 cm 台上箱高 0.45–0.60；warehouse 0.37 m 托盘上箱高 0.18–0.33（现纸箱 0.33 在上沿） |
| 箱长 | **≤ 0.32 m** | 0.32–0.44 之间没测，别超 0.32 |
| 箱宽 | **0.45 m**（最多 ±0.02） | 最窄的一维，两边都掉 |
| setback | **0** | 每 1 cm 都直接加在停位深度上 |
| 箱子偏角 | **0°**；必须随机时 ≤ ±2°，接受降到 25–30% | 转正身体也救不回来（§5 ⑤），原因未明 |
| 起点离台边 | **0.6–2.0 m 自由随机** | 数据多样性主要从这里来 |
| 起点朝向 | **±20°** | 右转略好 |
| 起点横移 | **−8～0 cm**（机器人在右侧） | 左侧不要给 |

按这套取值的产出率约 0.40–0.45；要 200 集成功局约 450–500 局，5 个实例并行约 1 小时（不录制 RTF 0.97–0.98）。

**搬回 warehouse 房间前先核一次**：同样的箱子几何在房间里（0.37 m 托盘）基线 26%，这里 43%，差在哪没查（台高、房间、或别的）；
房间里的默认参数仍是 `box_room_grasp_baseline_20260924`（[warehouse](warehouse.md) §2）。

## 5. 专家层面试过什么（2026-09-27，中心格，基线 28/60 = 0.47）

上限不是布局定的。逐帧复盘 835 局：卡住 = 下探前 1.4 s 内整对手横向甩 3–5 cm（与下探快慢、修正是否冻结无关）；脱手 = 1 cm 捏夹在壁上以 0.2–0.4 cm/s 竖直蠕滑。

| 改动 | 旋钮（默认关 / 原值） | 结果 | 判定 |
|---|---|---|---|
| ① 下探期保留共模横向修正 | `lateral_hold_common_mode` | 26/60；把箱子推走 9 cm | 否 |
| ② 局内重试 | `descend_retries` / `reseat_retries` | 12/30、①+② 25/60；第二次再卡（箱子已被推动） | 否，需先有下探期位移闸门 |
| ③ 竖直附着反馈 | `attachment_gain` | 15/30；**站直期脱手 27 → 13%** | 可选开启，总成功率未变 |
| ④ 起抬前坐好检查 | `seat_depth` / `seat_skew` | 7/30；控制器交付的坐姿本来就错开 1–2 cm | 否 |
| ⑤ 刹车后转向对准 | `face_after_brake` | ≤ 6° 转不动；转到位的格子仍 1/30 | 否 |
| 抓深 1 → 2 cm | `box_grasp_below_rim` | 10/30 | 否 |
| 下探放慢 | `descend_speed` 0.05/0.08/0.11 | 14/30、11/30、8/30 | 否 |
| 壁厚 20/30 mm | 场景 | 14/30、27/60；脱手 27 → 7–10% 但卡住 12 → 22–30% | 说明抓持侧可消、落点侧是上限 |
| 刹车后快速准备手臂 | `ready_after_brake quick` | 26/60，每局省 5.4 s | **默认** |
| 再压到 5.8 s | `ready_blend/hold/reanchor_*_seconds` | 23/60，单调下降 | 否 |

多次对照里"卡住"和"脱手"的比例互换而总数不变，说明有一批边缘抓取局在两者之间摇摆，看成功率时应合并计。

## 6. 复现与扩展

- 扫参文件：`configs/sweeps/box_envelope_s1_20260926.yaml`（几何网格）、`box_envelope_s2_20260926.yaml`（逐维链）；跑法见 [scene_sweep](scene_sweep.md) §3；
  报告 `outputs/sweeps/box_envelope_s{1,2}_20260926/report.md`，逐局 `outputs/runs/box_envelope_*`。
- 专家 A/B：`configs/experiments/box_centre_*_20260927.yaml`、`box_{lat,yaw}*_20260927.yaml`，对比脚本 `outputs/ab_expert_20260927/compare.py`。
- 新几何：`box_platform_scene --width/--length/--height [--wall]`，再按 `configs/experiments/box_platform_default_20260927.yaml` 改 `scene.dir`。
- 未做：横移方案 B（斜线走近）、下探期位移闸门（②b）、箱子偏角敏感的根因、③ 是否设默认。编号见 `TODO.md` 第 6 节。
