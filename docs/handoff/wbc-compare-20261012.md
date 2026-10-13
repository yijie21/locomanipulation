# SONIC 对 ScaleBFM：65 物体抓取线上的控制器对比（wbc_compare，2026-10-12）

状态（2026-10-12）：全部做完。手指律统一 + 65 场景适配器（§2–3），27 格开环探针与偏置常数（§4），手程序移植到 ScaleBFM（§5），SONIC 基线 + 对照批次（§6），**结果（§7）：受控臂 129 局配对 SONIC 48 对 ScaleBFM 77，p < 0.001；放开臂 55/130；并排视频与 reel 在 `outputs/wbc_compare_20261012/videos/`**。分支 `main`，未提交。任务 [TODO 24](../../TODO.md)，决策 [DECISIONS](../../DECISIONS.md) 2026-10-12 末节。代码在 `scripts/experiments/wbc_compare_20261012/`，产物在 `outputs/wbc_compare_20261012/`。 **续（§8，用户定的 24.2.2 + 24.6）：腰限修复后 SONIC `_wbc2` 48/130 对 `_wbc` 48/129（p = 1，交付精度换可达性）；ScaleBFM 姿态包络 31/156 个姿态可交付（全在虎口朝左、偏航 −60..+90°）；受控臂按 `_wbc2` 配对 48 对 68（p 0.005），按包络过滤的放开臂 48 对 104（p < 0.001）。** **续（§9，用户 2026-10-13 问的刹车）：空地终点线刹车，越线惯性 ScaleBFM 4.5 / 2.5 / 0.5 cm 对 SONIC 24.7 / 10.4 / 4.0（0.55 / 0.35 / 0.25），不越线程序两边 0 越线、脚尖停线前 ScaleBFM 4–8 cm 对 SONIC 13–22 cm；`outputs/wbc_compare_20261012/brake/`。**

## 1. 问题与五个决定

用户 2026-10-12：花在 SONIC 动作交付上的功夫太多，想把 SONIC 换成 ScaleBFM 重跑 65 物体抓取线（TODO 23.2），出并排视频，看成功率和抓取稳定性。先查台账：65 物体线 193 局里 103 局执行失败，分布是进入时撞倒 36、合爪后无双侧接触 30、抬起脱垫 18、退出撞倒 12（`outputs/graspnet_bottle_20261008/cache/summary.md`），合爪帧掌心误差只有 0.7–0.8 cm（grasp-cache handoff §13）——失败在手，不在站位；ScaleBFM 量过的优势是站位（停位 1–2 cm / yaw 0.3° 对 SONIC 7–11 cm / ±7°）和包络（84 格里掌垫误差 ≤10 cm 的 49 对 6–8），但它开环手误差 hover 3–6 cm、右手内偏 4–8 cm，也要 servo（ledger/scalebfm.md §1–2）。SONIC 交付不了倾斜 / 俯视手姿态（命令与实际差 5–20 cm），缓存排序因此把水平侧抓排前面。

| # | 决定 | 内容 |
|---|---|---|
| 1 | 受控臂 + 放开臂 | 受控臂：每物体同一缓存抓取（round 0 朝向）、同一站距、同一程序，只换控制器，记 ghost hands 和每阶段掌心残差。放开臂：ScaleBFM 用整个缓存按验证分排序（放开姿态层），站距不动 |
| 2 | servo | 每个控制器先做约 9 格开环探针得到一个前馈常数偏置；之上两边都用 SONIC 线的 entry gate + 3 cm/s 横向修正（上限 4 cm）。避免 v1.1 那种"程序在另一个控制器上标定"的接口错配（9/20 → 0/20） |
| 3 | 指标 | 成功沿用 `lifted`；物理稳定 = 进入时物体位移、合爪帧掌心残差、抬起 + 2 s 保持的掌系滑移；重复稳定 = 每物体每控制器 2 轮，按物体配对的三格表 |
| 4 | 放开臂只放姿态层 | 不放站距 |
| 5 | 用途 | 诊断和论文资产（Probe Before You Patch 的第二个平台，探测记录从按 checkpoint 键变成按控制器键）。不承诺换采集线：ScaleBFM 线当初停在 VLA 侧（动作行没速度 0/29、骨盆高度被 q99 截断、25/100 对 50/100），不是控制器侧 |

推论：SONIC 基线要重跑（程序多了偏置常数和 2 s 保持；轮次顺序按已知 5° 航向偏置平衡）；规模与 10-10 entry gate 实验相同（每控制器 130 局），能分出十来个点的差距，更小的差距读成"测不出"。

## 2. 第 1 步：手指律统一 + 场景适配器

两条线的 Dex1 手指律不一样，不统一则无双侧接触 / 滑移两个数不可比：

| | ScaleBFM 线（yplus200） | SONIC 线（deploy 手命令 → `base_sim.compute_hand_torques`） |
|---|---|---|
| PD | kp 1.0 / kd 0.025，`sim.grip_clip` 每局压到 0.6 | kp 1.5 / kd 0.1，ctrl 钳 20（`grasp_cache.py` KP / KD / FORCE_MAX） |
| 合爪目标 | −0.023 | −0.02（`agent_server._hand_poses`） |
| 指垫 | `bottle_visual.scene.make_sim` 加载时加硬（solref 0.002、solimp 0.995/0.9999） | XML 原样（friction 1.2、solref 0.004、solimp 0.99/0.999、condim 4） |

对比两边都用 SONIC 线那一列。改动：

- `src/locomani/backends/scalebfm/config.py`：新字段 `grip_clip`（默认 4.0 = 原来 `Sim` 里写死的值）；`runtime.Sim` 从配置读它，瓶子线每局 `sim.grip_clip = 0.6` 照旧。`configs/backend/scalebfm.yaml`、`docs/reference/config_fields.md`（重新生成）、`docs/scalebfm.md` 同步。
- `tests/test_scalebfm_grip_law.py`：`Sim.step` 对四个手指执行器施加的就是 `clip(kp (target − q) − kd dq, ±grip_clip)`；全开到合爪一步在 kp 1.5 下是 −0.06675 ctrl = −16.69 N（gear 250，关节 ±20 N 钳位在 `qfrc_actuator` 上；`d.actuator_force` 是 gear 前的标量）；配置里的钳位就是生效的钳位；`R.Sim` 单独加载时指垫保持 XML 值。
- `scripts/experiments/wbc_compare_20261012/bfm_object_scene.py`：`finger_config(scene)` 给出 SONIC 律的 `ScaleBfmBackendConfig`；`make_object_sim(meta, id)` 直接把 `outputs/graspnet_bottle_20261008/scenes/obj_XXX` 装进 `R.Sim`（两族场景的 `<actuator>`、`<option>` 和机器人 body 块 2026-10-12 diff 为空，物体静置姿态已在 XML 里重建），`verify_hand` 核对关节范围、±20 N、gear 250、四块指垫参数，不是原装 Dex1 就拒绝；`assert_matches_grasp_cache` 从 `grasp_cache.py` 文本重读 KP / KD / FORCE_MAX / JAW / CLOSE_S，两份常数漂了就报错（不直接 import 它：会拉进 grasp_planned 和 SONIC 栈）；`stance(sim)` = grasp_planned 的 STAND_FORWARD 0.38 / STAND_RIGHT 0.12 沿物体所贴桌边的内法线（瓶子线 face-on 朝向）。
- `check` 子命令：每个场景把机器人传送到站位，mode 2 下双手垂边 settle 1.5 s，再在空中按 SONIC 的 1.0 s 斜坡合右爪，记录骨盆 / yaw 交付、摔倒、物体位移和倾斜、掌垫相对垂边目标的误差、合爪到位时间、手指关节力峰值。

## 3. 验收（65 场景，CPU，60 s）

```
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \
  scripts/experiments/wbc_compare_20261012/bfm_object_scene.py check --objects all \
  --out outputs/wbc_compare_20261012/scene_check        # → scene_check.md / .json / check.log
```

| 量 | 结果 |
|---|---|
| 场景 / 摔倒 / 合不上爪 | 65 / 0 / 0 |
| settle 后骨盆 xy 误差 | 1.8 cm（65 场相同——所有物体都在基础场景的同一标称点，站位相同） |
| yaw 误差 | 0.0° |
| 右掌垫相对垂边目标 | 4.2 cm（左 3.0 cm）：ScaleBFM 开环手偏置，台账 §2 的"右手固定内偏 4–8 cm"在静止姿态下的值 |
| 物体位移 / 倾斜 | 中位 0.3 cm / 0.1°，最大 0.5 cm / 3.7°（022、034、043、052 倾斜 1.5–3.7°：settle 时物体自己再坐实） |
| 合爪到位（<1 mm） | 斜坡起 1.12 s（斜坡 1.0 s），终值 [−0.0201, −0.0194]，指 2 差 0.6 mm 没到 −0.02（PD 无积分；SONIC 同律同样） |
| 手指关节力峰值 | 1.0 N（跟着斜坡走，误差小；阶跃命令才到 16.7 N，单测里验） |

单测 `make test TESTS="tests/test_scalebfm_grip_law.py tests/test_visual_scene.py tests/test_configs.py"` 12 通过。

## 4. 第 2 步：开环手探针（24.2，决定 2）

**问的是**：手程序把右手工具（夹爪中心，腕前 0.1515 m）命令到站位周围的一格世界位姿、静止不动时，每个控制器把手交付到哪里——交付减命令就是 24.3 要从命令里减掉的前馈常数；格与格之间的散布说明一个常数够不够。

**格子**（`probe_cells.py`，两边共用）：以站位点为原点沿航向 前 0.30 / 0.38 / 0.46 m × 左 −0.20 / −0.12 / −0.04 m（物体在 −0.12）× 世界 z 0.67 / 0.74 / 0.81 m，27 格，左 −0.12 的 9 格是决定里的"约 9 格"。范围取自 10-10 两批 gate 的 114 个选中抓取（工具在骨盆前 0.35–0.38、右 0.12–0.16、z 0.65–0.77）。姿态每格相同：接近方向沿航向**下俯 30°**、钳轴水平——选中抓取接近仰角的中位数（q25–q90 全是 29–30°，三分之一水平）。格间先升到桌面上 0.22 m，回到中间列上方的"家"再过去、落下（grasp_planned 的分段路径，从同一个起点出发）。物体整场钉在站位后 1.5 m 的地上，不然每格都撞它。每格保持 2 s，取最后 0.5 s 的平均；误差写在**命令站位航向系**（前、左、上），两边同一个系。

**两边怎么交付**：ScaleBFM（`bfm_probe.py`）= 第 1 步的适配器 + mode 2 腕参考沿这条路以 0.15 m/s 走、六帧未来目标；SONIC（`sonic_probe.py` + `sonic_probe.sh`）= 一个 `hand_probe` 计划步走 `plan_episode.py`，接管、`align_robot_xy`、`_ik_setup`、每厘米解 IK、`retime` 流、臂 pin（DECOUPLE 默认）全是 grasp_planned 自己的那套——程序怎么交付抓取，探针就怎么交付格子；SONIC 的误差再拆成 IK 自身残差（解出关节的 FK 对目标）、臂 / 腰跟踪（实测工具对实测骨盆下那组关节的 FK）、规划后骨盆位移三项，和就是总误差。

```
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=osmesa PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \
  scripts/experiments/wbc_compare_20261012/bfm_probe.py --object 40 --device cuda --video \
  --out outputs/wbc_compare_20261012/probe/scalebfm                       # 125 sim s，GPU 上 9.5 min（渲染是大头）
bash scripts/experiments/wbc_compare_20261012/sonic_probe.sh 40 1 0        # 实例 1 GPU 0，约 6 min；视频 outputs/runs/wbc_probe_040/videos/
python3 scripts/experiments/wbc_compare_20261012/probe_summary.py outputs/wbc_compare_20261012/probe   # → probe.md, bias.json
```

**结果**（物体 040 的场景，`probe/probe.md`；预览 `probe/{scalebfm,sonic}/probe_preview.mp4`，SONIC 的是 4 倍速）：

| | ScaleBFM | SONIC |
|---|---|---|
| 有路径 / IK 终点在容差内（1.5 cm / 10°） | 27 / 27 | 27 / 26（格 2 远低角 IK 差 1.9 cm，偏置不算它） |
| 偏置 前 / 左 / 上 | −0.4 / +0.3 / +1.8 cm | −1.2 / −4.1 / +1.0 cm |
| 格间散布 sd | 1.8 / 0.9 / 0.5 cm | 1.2 / 2.6 / 1.9 cm |
| 原始误差模 中位 / p90 / 最大 | 2.7 / 3.5 / 4.5 cm | 5.0 / 8.9 / 9.5 cm |
| 减常数后残差 中位 / p90 / 最大 | 1.9 / 3.0 / 4.4 cm | 3.0 / 5.2 / 5.5 cm |
| 中间 9 格：偏置 / 残差 p90 | −0.4 / +0.3 / +1.8；2.5 cm | −1.1 / −3.5 / +1.0；4.9 cm |
| 掌姿态误差 中位 / 最大 | 2.8° / 6.1° | 6.9° / 13.9° |
| 骨盆漂移最大 | 2.8 cm / 2.3° | 3.5 cm / 3.5° |

- **ScaleBFM** 的误差结构：整体高 1.8 cm，前 0.46 那列短 2 cm，低行短 2 cm；骨盆站着不动（z 0.78）。一个常数减掉后 p90 3.0 cm，在 servo 的 4 cm 横向上限内。
- **SONIC** 的误差几乎全在"臂 / 腰跟踪"一项：IK 残差 ≤ 0.6 cm，骨盆位移 ≤ 2 cm，跟踪项低行 (−1.9, −5.7, +1.6) cm、高行 (−0.5, −2.4, 0)。臂关节被 pin 住（实测对命令 0.003 rad），差的是**腰**：低行 IK 要腰 yaw 0.24 / roll 0.15 / pitch 0.20，SONIC 交付 0.11 / 0.08 / 0.22——pitch 给了，yaw 和 roll 只给一半，手就落在目标右边 2–9 cm、短 1–5 cm，随伸得远、够得低线性变大。减一个常数后 p90 5.2 cm，超过 servo 的 4 cm 上限：决定 2 照原样做的话，SONIC 受控臂会把这一项带进结果里。
- 两边各自的站位航向：ScaleBFM 传送到 −90°；SONIC 由 spawn 坐定到 −71.7°（own_heading 规则，和 grasp_planned 一样），格子都在各自航向系里。

**第一轮（水平接近姿态）为什么作废**（存档 `probe/*/probe_run1_level.json`）：SONIC 在程序的 IK 限制下只有 5/27 格有路，低行 IK 自己就差 7–12 cm（够不到 0.67 的水平接近），而且一格 IK 跳支后面 22 格全跟着跳（每格从上一格出发）；ScaleBFM 水平姿态下偏置 (−1.8, −1.1, 0.0)、低行掌心自己低头 13°（涌现的俯仰）、残差 p90 4.3 cm。离线核过：水平姿态 15/27 可解，下俯 30° 25/27，40° 27/27。

**顺带发现（待用户定）**：`grasp_planned._ik_setup` 先套 `WAIST_LIMITS`（yaw ±0.25、roll ±0.05），紧接着 `StreamedArm.__init__` 把 12–14 的上下限**覆盖回** ±0.35 / ±0.15 / −0.05..0.20——10-08 "再收窄"的腰限从未到过 IK 规划器（第一轮 SONIC 解出的腰正好停在 0.35 / 0.15 / 0.20）；探针按程序原样复现，没有改。

用户 2026-10-12 定：SONIC 不补腰，按决定 2 原样只减常数（24.2.1 关闭）。

## 5. 第 3 步：手程序移植到 ScaleBFM（24.3）

`scripts/experiments/wbc_compare_20261012/bfm_grasp.py`：grasp_planned 的程序原样搬到 ScaleBFM runtime 上，只换"工具目标怎么变成动作"这一层。逐段对应：

| 段 | SONIC（grasp_planned） | ScaleBFM（bfm_grasp） |
|---|---|---|
| 站位 / 物体 | spawn 坐定后 `align_robot_xy`；物体按 run_grasp 的每物体 yaw（rng 1000 + id）纯 yaw 放下 | 传送到同一站位规则；物体同一 yaw 纯 yaw 放下，与机器人一起 settle 1.5 s |
| 候选 | `cached_candidates`（缓存按实测物体位姿变换，tier / 分数排序）→ IK 预排序选第一个能解的 | 同一个 `cached_candidates`；受控臂 `--select paired` 取 SONIC 同轮 record 的 `grasp.row`，放开臂 `--select score` 全缓存按验证分取最高 |
| 接近（开环） | IK 分段路径（升到桌面上 0.22、平移、落到预抓），0.6 rad/s 重定时，臂 pin | 腕参考走同一条分段工具路径，0.15 m/s，姿态在升 / 平移段转到位；两边预抓 / 抓取目标都减 `bias.json` 常数 |
| settle 起的闭环 | `SupportedMotion`：每拍全身 IK + 跟踪反馈（参考减实测的积分，死区 2 mm、0.04 m/s、0.2 m/s²、接管时保留初始偏差为原点，±8 cm 钳） | 同一个反馈律直接作用在腕参考上（`Tracking`）；六帧未来目标 = 修正后目标 + 命令速度外推 |
| entry gate / 横向修正 / 重试 / 合爪 / 接触超时 / 抬 / 保持 | `_grasp_like_the_expert` | 逐行同一代码路径，阈值从 bottle_simple_200.yaml 的 `reach.simple` 抄来（那个模块 import 会拉 SONIC 栈）；接触 = 右手两指 link 都与物体 body 有 MuJoCo 接触 |
| 没搬的 | 躯干前倾目标（`lean_angle`）、姿态反馈 | mode 2 没有躯干目标；姿态误差原样报 |

**偏置放在哪**：只减在开环的规划路径（预抓、抓取目标）上，闭环不再减。`SupportedMotion` 接管的第一拍把"规划路径终点 − 实测"记成反馈原点，之后围绕它积分，所以常数在闭环里会被数两次；移植版把 `Tracking` 的第一拍参考也设成规划路径终点（`pregrasp − bias`），第一版设成实测位置，原点变成 0、反馈被 ±8 cm 钳住，预抓 16 cm 的差追不回来。

**两次冒烟（物体 040）**：
- `--select cache`（缓存自己的顺序，不经 IK 预排序）：取到的是 tier 0 但接近方向偏航向 56°、掌心朝下的抓取；ScaleBFM 把这个姿态的腕交付在目标上方 19 cm，settle 超时后 dash 把瓶子推倒。姿态是 ScaleBFM 交付包络的强因素（探针里下俯 30° 掌心朝上的姿态 p90 3.5 cm），放开臂按分数选会撞上它——这是放开臂要量的东西，不改选法。
- `--select paired --pair-runs _gateA`（SONIC 10-10 选的 row 250，水平侧抓）：**抬起**，预抓开环误差 3.3 cm，settle 0.3 s 到 1.8 cm，entry 0.9 cm（gate 等了 0.42 s），合爪双侧接触，抬 9.5 cm，2 s 保持滑移 0.3 cm；11.5 sim s，GPU 上 58 s（渲染是大头）。

**SONIC 侧改动**：`grasp_planned.BIAS`（默认关；`bias_world` 把航向系常数转世界系）在 `_plan_ik` / `_plan_curobo_batch` 的预抓、抓取目标上减掉；`HOLD_S` 可改；record 多 `bias`、`hold_s`。`run_grasp.py --bias-file probe/bias.json [--bias-key sonic.all] --hold 2.0`。

## 6. 第 4 步：批次（24.4，2026-10-12 启动）

ScaleBFM 的仿真是确定性的（同 seed 两轮逐拍相同），"每物体 2 轮"在它那边只能是两个物体朝向：两边都 `--rounds 2 --yaws 2`，第 r 轮用第 r 个抽的 yaw（同一个 rng），按 (物体, 轮) 配对。

```
ROUNDS=2 bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 4 1 0 _wbc --rounds 2 --yaws 2 \
  --bias-file outputs/wbc_compare_20261012/probe/bias.json --hold 2.0          # SONIC 基线，实例 1-4 GPU 0
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=osmesa PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \
  scripts/experiments/wbc_compare_20261012/bfm_grasp.py --objects all --rounds 2 --yaws 2 --select score --video \
  --bias outputs/wbc_compare_20261012/probe/bias.json --out outputs/wbc_compare_20261012/bfm_grasp/free     # 放开臂，GPU 1
# SONIC 批次跑完后，受控臂：同上 --select paired --pair-runs _wbc --out .../bfm_grasp/paired
ROUNDS=2 bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 2 1 0 _wbc_nobias --rounds 2 --yaws 2 --hold 2.0   # 对照：只加保持、不减偏置
```

**SONIC `_wbc` 基线（129 局有 record，1 局 spawn 落点差 0.11 m 没起）**：抬起 48/129（第 0 轮 23/65，第 1 轮 25/64）。偏置起了它该起的作用——预抓开环误差中位 3.4–3.9 → 2.7–2.9 cm、p90 5.8 → 4.5 cm（对 10-10 的 gate 两批）；settle 0.32 s、entry 误差 1.2 cm、合爪帧掌轴偏差 0.5 cm 都和 gate 批一样（闭环本来就追得上）。但成功率比 gate 批（32、33/65）低 9 个点：第 0 轮"进入时撞倒" 11（gate 3、6），"抬起脱垫" 10（8）——保持从 1.5 s 加到 2 s 给了更多脱手时间，撞倒增多说不清是偏置还是批间散布（同配置两批就差过 3 对 9）。为分开这两件事，又起了一批对照 `_wbc_nobias`（2 s 保持、不减偏置）。

**对照结果（130 局）**：`_wbc_nobias` 抬起 53/130（26 / 27），`_wbc` 48/129；按 (物体, 轮) 配对，只偏置成 19 : 只无偏置成 24，exact McNemar p = 0.54——**偏置常数对 SONIC 的成功率没有可测的作用**，虽然它把预抓开环误差从 3.4 压到 2.9 cm（闭环本来就把它追回来了，合爪帧误差两批都 1.2–1.3 cm）；两批选中同一个缓存 row 的只有 81/129（IK 预排序对路径敏感），这也是配对散布的一部分。和 10-10 gate 批（32、33/65）的差距主要是保持时间：到抬起阶段的局数 gate 40–41 对 nobias 38–39，差不多；抬起后保持里脱手的 gate 8 对 nobias 12（每 65 局），2 s 比 1.5 s 多出的半秒又漏掉 4 个左右。结论：两边都按决定 2 减常数、保持 2 s（受控臂已按 `_wbc` 的 row 配对），`_wbc_nobias` 作对照一并报。

## 7. 结果（24.5；`compare_summary.py` → `outputs/wbc_compare_20261012/compare/{paired,free}.md`）

**受控臂**（同一缓存抓取、同一物体 yaw、同一程序，只换控制器；129 局配对，`compare/paired.md`）：

| | ScaleBFM 抬起 | ScaleBFM 失败 |
|---|---|---|
| SONIC 抬起 | 38 | 10 |
| SONIC 失败 | 39 | 42 |

SONIC 48/129，ScaleBFM 77/129；不一致格 10 : 39，exact McNemar **p < 0.001**；两轮分别 23 对 37、25 对 40；按物体看，ScaleBFM 两轮都成 30 个物体（SONIC 13），两轮都败 18 个（SONIC 30）。ScaleBFM 的 42 个失败里 15 个是"没得配"（SONIC 那轮 IK 无路径 12、无候选 2、另 1 个 SONIC 成了但 record 没 row）；排除后 114 局 SONIC 47 对 ScaleBFM 77。

| 失败类 | SONIC | ScaleBFM |
|---|---|---|
| 进入时撞倒 | 19 | 5 |
| 合爪后无双侧接触 | 19 | 11 |
| 抬起 / 保持中脱垫 | 20 | 7 |
| IK 无路径 | 12 | 0（没有 IK 层） |
| entry 超时（手到不了抓取） | 0 | 6 |
| settle / retreat / approach 撞倒 | 5 | 3 |
| 无候选 / 没得配 | 6 | 4 + 15 |

| 物理稳定量（到达该阶段的局；中位 / p90） | SONIC | ScaleBFM |
|---|---|---|
| 预抓开环误差 cm | 2.9 / 4.5 (n 110) | 3.1 / 9.7 (n 110) |
| settle 秒 | 0.32 / 0.56 | 0.32 / 1.50 |
| entry gate 等待 s | 0.82 / 1.92 | 0.32 / 2.00 |
| 合爪前物体位移 cm | 0.00 / 1.28 (n 87) | 0.00 / 0.07 (n 96) |
| 合爪帧工具对抓取误差 cm | 1.21 / 1.86 | 0.98 / 2.24 |
| 抬起 + 2 s 保持的滑移 cm | 0.97 / 4.07 (n 68) | 0.25 / 1.24 (n 85) |

读法：SONIC 的三大执行失败（进入撞倒、无接触、脱垫）在 ScaleBFM 上各减到 1/3–1/2，滑移 p90 从 4.1 降到 1.2 cm——同一个手程序，交付更准、抓得更稳；ScaleBFM 自己的失败模式是**姿态**：预抓开环误差 p90 9.7 cm 和 6 个 entry 超时全是 tier 1–2（下俯、偏航大）的抓取，它把这些姿态的腕交付在目标上方 10 cm（探针 §4 第一轮已见到）。SONIC 10 局独成的局，ScaleBFM 多半败在这一类。

**放开臂**（ScaleBFM 全缓存按验证分取最高、只过路径清障，姿态层放开；`compare/free.md`，不与 SONIC 按抓取配对，只按 (物体, 轮) 并列）：ScaleBFM **55/130**，对 SONIC 的 48/129 不一致格 27 : 34，p = 0.44——放开姿态层反而把受控臂的 77 丢回 55。它选到的抓取 tier 2（下俯 > 25° 或偏航大）67 局，抬起 22（33%）；tier 0 36 局抬 19（53%）；tier 1 23 局抬 15（65%）。预抓开环误差中位 / p90 6.5 / 15.0 cm（受控臂 3.1 / 9.7），entry 超时 12、settle / approach 撞倒 16：验证分偏爱的俯抓姿态正是 ScaleBFM 交付不了的。放开臂的两条结论：ScaleBFM 的姿态包络比 SONIC 的 IK 窄得多，受控臂的 77 里 SONIC 的 IK 预排序替它挑掉了交付不了的姿态；想真放开姿态层，选抓取时要带 ScaleBFM 自己的交付包络（探针按姿态再测一遍），不能只按验证分。

**视频**：`outputs/wbc_compare_20261012/videos/paired/obj_XXX_round_RR.mp4`（左 SONIC 带腕 / ego 小窗，右 ScaleBFM，品红 ghost 在命令抓取上，标题写物体 / 结果 / 原因；`*_preview.mp4` ≤ 500 KB），`videos/paired/reel.mp4` 四象限（两成 / SONIC 独成 / ScaleBFM 独成 / 两败，2 倍速）。`compare_videos.py --arm free` 出放开臂的片，标题带 `(free)`。

## 8. 第 6 步：腰限修复（24.2.2）与 ScaleBFM 的姿态交付包络（24.6；用户 2026-10-12 定）

**24.2.2 修法**：`grasp_planned.apply_waist_limits(ik)`（幂等）在 `StreamedArm` 接管之后再套一次，`_ik_setup` 里那次保留；`sonic_probe.py` 同样补一次。`StreamedArm.__init__` 的 ±0.35 / ±0.15 不动（主线 bottle 专家的校准值）。修完 SONIC 探针重跑、常数重算、基线 `_wbc2` 全新跑、受控臂按 `_wbc2` 重配对。

**SONIC 探针重跑（窄腰，`probe/sonic/probe.json`；旧的存 `probe_run2_widewaist.json`，旧常数 `bias_run2_widewaist.json`）**：23/27 可达（格 1、2、5、8 IK 超容差——低行的远格，窄腰够不到），偏置 (−0.5, −2.8, +0.7) cm（宽腰 (−1.2, −4.1, +1.0)），残差 p90 4.0 cm（5.2）。分解不变：IK 残差 ≈ 0、骨盆 ≈ 0、跟踪 (−0.7, −2.7, +0.5)；命令的腰 yaw 均 0.145 rad（上限 0.25）交付 0.078——**yaw 仍只交付一半**，roll 命令压到 0.04 后不再是误差源，所以左偏从 4.1 降到 2.8 cm。

**SONIC `_wbc2` 基线（窄腰 + 新常数 + 2 s 保持，130 局）**：48/130，对 `_wbc` 48/129 按 (物体, 轮) 配对 both 27 / 只 `_wbc` 21 / 只 `_wbc2` 21 / 都没 61，p = 1（`compare/sonic_wbc_vs_wbc2.md`，`sonic_pair_summary.py`）。**成功率不动，失败换了类**：进入撞倒 19 → 4、预抓开环误差中位 2.86 → 2.29 cm（交付确实更准），同时 IK 无路径 12 → 18、选中 tier 2 抓取 19 → 32、合爪无双侧接触 19 → 35——可达范围缩了，IK 预排序挑到更差的姿态。腰限是"交付精度换可达性"的旋钮，在这条线上净效为零。

**姿态包络探针（`bfm_attitude_probe.py` → `probe/scalebfm_attitude/attitude_{0,1}.json`；`attitude_summary.py` → `envelope.json` / `envelope.md`）**：姿态坐标就是 `_tier` 读抓取的三个量——接近仰角（下俯 0..90°，7 档）× 接近相对航向的偏航（−150..180°，12 档）× 绕接近轴的滚转（0 = 虎口朝左掌心朝上即 24.2 的姿态，180 = 朝右朝下；90° 仰角只有 0）= 156 个姿态，在 24.2 中列的两个高度（z 0.67 / 0.74）各放一次 = 312 格；每格从 HOME 原地转到姿态（≤ 90°/s）、过去、下、保持 2 s、回 HOME，摔倒就重置（0 次）。不录像的 ScaleBFM 跑得比实时快（744 sim s / 293 s 一片），录像才是 4 倍的成本。**可交付**（位置误差均值 ≤ 5 cm 且姿态误差 ≤ 15°）**31 / 156**：全在滚转 0，偏航 −60..+90°、仰角 0–90°（边缘有缺口，60°/75° 的偏航 0 刚过 5 cm）；**滚转 180 一个都没有**（5–29 cm、姿态差 15–147°）；偏航 ≤ −90 或 ≥ 120 也没有。灵敏度 3 / 4 / 5 / 6 / 8 cm → 11 / 18 / 31 / 36 / 40 个。放开臂原来挑的 tier 2 失败正是虎口朝右（滚转 180）和偏航 < −30° 的抓取。录像子集 `probe/scalebfm_attitude/video/attitude_0_preview.mp4`（26 格，214 KB）。

**受控臂 `bfm_grasp/paired2`（按 `_wbc2` 的选择配对）**：129 局配对 SONIC 48 对 ScaleBFM 68，both 35 / 只 SONIC 13 / 只 ScaleBFM 33 / 都没 48，exact McNemar p = 0.005（`compare/paired2.md`）。ScaleBFM 比对 `_wbc` 配对时的 77 少了 9：`_wbc2` 的 IK 预排序多挑了 13 个 tier 2 抓取（32 对 19），ScaleBFM 在那些姿态上 entry 超时 9、进入撞倒 8；失败类 SONIC → ScaleBFM：合爪无双侧接触 35 → 8、抬起脱手 18 → 11、进入撞倒 4 → 8、IK 无路径 18 → 0（ScaleBFM 不用 IK；19 局没得配）。稳定量中位 / p90：合爪前物体位移 0.00 / 1.34 → 0.00 / 0.05 cm，合爪帧误差 1.25 / 1.90 → 0.99 / 2.50 cm，保持滑移 0.77 / 3.72 → 0.29 / 1.94 cm，预抓开环误差 2.29 / 3.87 → 3.67 / 11.0 cm（ScaleBFM 的 p90 就是姿态问题）。**包络事后能预测 ScaleBFM 在配对抓取上的成败**：把 SONIC 选中的抓取按 `Envelope.reject` 离线分类，`paired2` 包络内 52 局抬起 48（92%）、包络外 54 局抬起 20（37%）；`paired` 同样 55 局 95% 对 55 局 45%。SONIC 选的 tier 2 抓取只有 7/33 在包络内。

**放开臂 `bfm_grasp/free2`（验证分排序 → 包络过滤 → 路径清障）**：130 局选中 120、抬起 105（129 局配对 SONIC 48 对 ScaleBFM 104，both 45 / 只 SONIC 3 / 只 ScaleBFM 59 / 都没 22，p < 0.001；`compare/free2.md`）。对没过滤的 `free` 55：选中的 tier 2 从 67 降到 40 但抬起 22 → 35（包络内的 tier 2 也能抓），tier 0 19/36 → 29/32、tier 1 15/23 → 41/48；预抓开环误差中位 / p90 6.5 / 14.9 → 3.3 / 5.3 cm；失败类：合爪无双侧接触 0（SONIC 35）、抬起脱手 8（17）、进入撞倒 1（9）、接近撞倒 0（7）、entry 超时 1（12），新增 6 局『包络内没有能清障的候选』、4 局无候选。每局按姿态拒掉的候选中位 2 个（最多 49）。**结论：放开姿态层本身没错，错的是用验证分当姿态的代理；按控制器自己的交付包络选抓取，ScaleBFM 从 55 到 104，比受控臂的 68 / 77 高，因为它不再被 SONIC 的 IK 预排序绑着——SONIC 能解的姿态和 ScaleBFM 能交付的姿态不是一个集合（配对抓取只有一半在包络内）。**

```
bash scripts/experiments/wbc_compare_20261012/sonic_probe.sh 40 1 0                         # SONIC 探针（窄腰），约 6 min
python3 scripts/experiments/wbc_compare_20261012/probe_summary.py outputs/wbc_compare_20261012/probe
ROUNDS=2 bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 4 1 0 _wbc2 --rounds 2 --yaws 2 \
  --bias-file outputs/wbc_compare_20261012/probe/bias.json --hold 2.0                       # SONIC 基线，36 min
python3 scripts/experiments/wbc_compare_20261012/sonic_pair_summary.py --a _wbc --b _wbc2
for k in 0 1; do CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=osmesa PYTHONPATH=src OMP_NUM_THREADS=1 \
  third_party/Isaac-GR00T/.venv/bin/python scripts/experiments/wbc_compare_20261012/bfm_attitude_probe.py \
  --object 40 --device cuda --shard $k --shards 2 --out outputs/wbc_compare_20261012/probe/scalebfm_attitude & done; wait   # 5 min
python3 scripts/experiments/wbc_compare_20261012/attitude_summary.py outputs/wbc_compare_20261012/probe/scalebfm_attitude
B=outputs/wbc_compare_20261012
bash scripts/experiments/wbc_compare_20261012/bfm_batch.sh 4 1 $B/bfm_grasp/paired2 --select paired --pair-runs _wbc2 --video \
  --rounds 2 --yaws 2 --bias $B/probe/bias.json                                              # 受控臂，4 进程
bash scripts/experiments/wbc_compare_20261012/bfm_batch.sh 4 1 $B/bfm_grasp/free2 --select score --video \
  --envelope $B/probe/scalebfm_attitude/envelope.json --rounds 2 --yaws 2 --bias $B/probe/bias.json   # 放开臂
python3 scripts/experiments/wbc_compare_20261012/compare_summary.py --sonic _wbc2 --arm paired2
python3 scripts/experiments/wbc_compare_20261012/compare_summary.py --sonic _wbc2 --arm free2
python3 scripts/experiments/wbc_compare_20261012/compare_videos.py --sonic _wbc2 --arm paired2 --reel --out $B/videos/paired2
python3 scripts/experiments/wbc_compare_20261012/compare_videos.py --sonic _wbc2 --arm free2 --out $B/videos/free2
```

## 9. 第 7 步：终点线刹车，SONIC 对 ScaleBFM（24.7；用户 2026-10-13 提出）

用户问 ScaleBFM 的刹车距离是不是也更近、能不能站得离桌子更近再抓。先答清楚：这条比较线里两边都没走路，站位是 `align_robot_xy` / `stance()` 直接放的（决策 4），§7–8 的差异与刹车无关；走路停位只有纸箱线的旧数字（ScaleBFM 停位 1–2 cm / yaw 0.3° 对 SONIC 7–11 cm / ±7°，ledger §1–2）。用户随后要的是 09-23 那条空地终点线实验（`backends/sonic/diag_brake.py`）换控制器再跑一遍：走到线、停下，最好刚好停在线上且不越线，看谁停得准。

**协议**（同 09-23）：空地 `outputs/scene/walk`，原点朝 +x，线在 1.5 m；指令速度 0.55 / 0.35 / 0.25，每档 10 局；停稳 3 s 读骨盆到线、身体最前的低位点（底面低于 0.37 m 的 geom，下称脚尖）到线、横向、yaw。两个程序：

- `marker`：骨盆越线那一拍发刹车命令。SONIC = IDLE（diag_brake 原样）；ScaleBFM = 参考冻结在当时的骨盆位置（`bfm_brake.py --program marker`）。量的是刹车惯性。
- `plan`（不越线）：各自最好的程序。SONIC = 落脚点守线刹车（`--guard --extra 0.20 --margin 0.03`，09-23 验证 50/50 的设置）加一组两段式（0.55，最后 0.3 m 降 0.20）；ScaleBFM = 骨盆参考走梯形（加减速 0.6 m/s²），终点 = 线 − 站立时脚尖伸出量（实测 13.8 cm）− 3 cm（`--program plan`），即把脚尖停到线前 3 cm。

ScaleBFM 仿真确定性，每局把线抖动 ±10 cm 换步态相位；SONIC 靠出生和步态自带散布。ScaleBFM 不录像一局 3 s 墙钟，SONIC 栈实时、单独跑（`brake_batch.sh`：ScaleBFM 先、SONIC 后）。渲染复用 `render_brake_marker_20260923.py`（改成按行读线位置、字幕标签从 results.json 取）；`runtime.Sim` 改成允许场景里没有被跟踪物体。SONIC 今天重跑与 09-23 一致（越线惯性 24.7 / 10.4 / 4.0 对 20.0 / 9.2 / 4.3 cm）。

**结果**（`outputs/wbc_compare_20261012/brake/brake.md`；单位 cm，正 = 越线；视频 `brake/videos/<程序>_<速度>.mp4`，左 SONIC 右 ScaleBFM，`reel_preview.mp4` ≤ 500 KB）

marker，越线惯性（骨盆停稳到线 mean ± sd [范围]；脚尖停稳到线）：

| 指令速度 | SONIC 实测速度 | SONIC 骨盆 | SONIC 脚尖 | 横向 sd / \|yaw\| | ScaleBFM 实测速度 | ScaleBFM 骨盆 | ScaleBFM 脚尖 | 横向 sd / \|yaw\| | 静止用时 |
|---|---|---|---|---|---|---|---|---|---|
| 0.55 | 0.53 | +24.7 ± 3.3 [17.8, 30.0] | +39.5 ± 2.7 | 4.4 / 2.4° | 0.53 | +4.5 ± 0.6 [3.8, 5.8] | +25.6 ± 2.5 | 1.6 / 0.2° | 0.59 s |
| 0.35 | 0.25 | +10.4 ± 4.9 [2.5, 19.4] | +26.7 ± 4.3 | 5.0 / 2.9° | 0.35 | +2.5 ± 0.6 [1.7, 3.3] | +20.9 ± 1.8 | 0.9 / 0.2° | 0.57 s |
| 0.25 | 0.12 | +4.0 ± 1.4 [1.7, 6.3] | +19.8 ± 3.2 | 3.7 / 2.8° | 0.25 | +0.5 ± 0.4 [−0.0, 1.1] | +16.2 ± 0.7 | 0.4 / 0.3° | 0.50 s |

plan，不越线（两边 30 局 0 越线、0 摔；脚尖停稳到线 mean ± sd，最差 = 全程最远点的最差局）：

| 速度 | 程序 | 骨盆 | 脚尖 | 最差 | 横向 sd / \|yaw\| | 目标误差（停稳骨盆 − 参考终点） |
|---|---|---|---|---|---|---|
| 0.55 | SONIC 守线 | −36.1 ± 12.2 [−52.9, −19.1] | −22.3 ± 12.4 | −2.3 | 4.6 / 0.5° | – |
| 0.55→0.20 | SONIC 守线两段式 | −36.8 ± 9.8 | −22.4 ± 10.1 | −10.1 | 6.5 / 1.8° | – |
| 0.55 | ScaleBFM 梯形 | −18.4 ± 0.8 [−19.7, −17.4] | −3.9 ± 0.6 | −2.9 | 0.8 / 0.3° | −1.6 ± 0.8 |
| 0.35 | SONIC 守线 | −27.3 ± 8.2 [−43.6, −20.9] | −12.8 ± 7.0 | −6.8 | 2.6 / 0.8° | – |
| 0.35 | ScaleBFM 梯形 | −20.4 ± 0.6 [−21.7, −19.9] | −7.0 ± 0.5 | −5.5 | 0.7 / 0.3° | −3.6 ± 0.6 |
| 0.25 | SONIC 守线 | −34.2 ± 4.0 [−38.2, −28.2] | −16.3 ± 3.5 | −10.3 | 1.9 / 1.7° | – |
| 0.25 | ScaleBFM 梯形 | −21.3 ± 2.0 [−25.9, −19.9] | −8.2 ± 2.2 | −6.6 | 0.2 / 0.2° | −4.5 ± 2.0 |

**解读**

1. ScaleBFM 的刹车惯性是 SONIC 的 1/5 到 1/8，散布是 1/5 到 1/8：0.55 时 4.5 ± 0.6 对 24.7 ± 3.3，0.25 时 0.5 ± 0.4 对 4.0 ± 1.4；0.5 s 站稳，横向 ≤ 1.6 cm、yaw 0.2°（SONIC 4–5 cm、2–3°）。它还真的按指令速度走（0.35 → 0.35；SONIC 0.35 → 0.25、0.25 → 0.12，1.5 m 走 12 s）。
2. 不越线程序两边都 0 越线，差在停得离线多远和散布：ScaleBFM 把脚尖停在线前 4–8 cm、sd ≤ 2 cm、骨盆 18–21 cm；SONIC 守线停在线前 13–22 cm、sd 3.5–12 cm、骨盆 27–37 cm，0.55 恒速最差一局离线只剩 2.3 cm。SONIC 的散布来自步长量化（IDLE 后必多迈一步，09-23），ScaleBFM 的来自参考终点的跟踪偏差：停稳骨盆比参考终点短 1.6（0.55）、3.6（0.35）、4.5 ± 2.0（0.25）cm，0.25 有两局短 8–9 cm（提前一步站定）。这个偏差是常数级，可以前馈补掉（24.7.1）。
3. 对抓取线的含义：ScaleBFM 走进去的站位能规划到 ±2 cm，SONIC 要留 15–35 cm 余量加 ±5–12 cm 散布，这就是"站位可以规划得更近"的数字。但 §7–8 的成功率差异不是站位造成的（两边都是放上去的）；要在 65 物体线上兑现它得让 ScaleBFM 真走进去，那是决策 4 之外的另一条线。

**待用户**（TODO 24.7.1）：给 ScaleBFM 的 plan 加 3 cm 前馈再验一轮（脚尖应停到线前 1–3 cm）；要不要让 ScaleBFM 在 65 物体线上真走进去。

```bash
bash scripts/experiments/wbc_compare_20261012/brake_batch.sh          # 全部：ScaleBFM 两程序 → SONIC 三组（栈单独）→ 渲染 → 汇总，约 30 分钟
B=outputs/wbc_compare_20261012/brake
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \
  scripts/experiments/wbc_compare_20261012/bfm_brake.py --program plan --out $B/bfm_plan          # 单独跑 ScaleBFM 一个程序
MUJOCO_GL=egl third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/render_brake_marker_20260923.py $B/bfm_plan
BRAKE_OUT=$PWD/$B/sonic_guard bash scripts/experiments/brake_marker_20260923.sh --guard --extra 0.20 --margin 0.03 \
  --speeds 0.55,0.35,0.25 --repeats 10                                                              # SONIC 一组（起栈、跑、落栈、渲染）
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/wbc_compare_20261012/brake_summary.py --out $B
```
