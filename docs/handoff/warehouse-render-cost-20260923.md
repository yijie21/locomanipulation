# warehouse 纸箱线：过冲撞箱的根因是渲染负担；离线渲染跑通，0/10 停得太远（2026-09-23）

接手入口。当前进度与下一步以根目录 [STATE](../../STATE.md) 为准，任务编号见 [TODO](../../TODO.md) 2.4.10，
决策理由见 [DECISIONS](../../DECISIONS.md)「2026-09-23 warehouse 撞箱的根因……」。本页是证据和复现方式。

> **2026-09-24 更正：两轮离线 pilot 里伸手段从未开始。** 占位帧的时间戳在专家循环取 `now` 之后打，慢于 33 ms 的 tick 每次读都重打，
> 估计被判"来自未来"，伸手参考时钟 440/440 tick 停在 0（`low_reach.reference_elapsed`）。下面两节里"手够不到 / 卡在 pregrasp 前"的读法
> 都错了——手根本没被命令离开 ready 姿态；站位数据仍有效。已修（专家循环在 `camera.read()` 之后再取 `now`），待重跑。见末尾"离线伸手时钟"一节。

## 结论

1. **warehouse 场景里的仿真只有 0.30–0.36× 实时**，原因是纸箱 `nor_cal_box` 的显示网格：46.3 万面，外加 2 盏投影灯
   （shadowsize 4096），3 路相机每 tick 渲染 39–44 ms。walk 场景同样 3 路相机只要 1.3 ms。
2. 慢仿真下 deploy 按墙钟推进参考，物理跟不上，步长偏大（0.17–0.34 m，正常 0.14 m），实际速度约是指令的 3 倍。
   **视频里的"过冲撞箱"就是这个现象**，不是专家或守线的移植 bug。
3. 新增 `backend.render: offline`（默认 `live`）：sim 不渲染，专家读占位帧，视频事后从 state trace 渲出。
   打开后 sim 回到 0.95–0.97×，走路段 0/10 碰箱、横偏 ≤ 7 cm。
4. 离线 10 局仍是 **0/10**，站位太远；伸手段的表现无效（见 09-24 更正）。

## 根因的证据

| 测量 | walk | warehouse | 来源 |
|---|---|---|---|
| sim 实时比 | 0.90–0.95 | 0.30–0.36（合成桌子 `box_pilot_*` 也是 0.36） | 每局 trace 行数 × 0.02 s ÷ manifest 的 `seconds` |
| 物理步进 | ~0.15 ms | ~0.15 ms | `outputs/diagnostics/warehouse_speed_20260923` |
| 3 路相机渲染 | 1.3 ms | 39–44 ms；隐藏纸箱显示网格 2.0 ms；关阴影 11 ms | 同上 |
| 不读相机 / 持续读 3 路 | — | 0.42 / 0.43，读不读没有差别 | 同上 |

两个被排除的假设：

- **资源抢占**：原样重跑 `box_room_grasp_hanging2_rerun_20260923`，CPU 只用 4.5–6.5 核，GPU0 上只有这一个栈，结果仍是 0.35×。
- **reanchor 朝向错**：横漂的方向每局随机，原样重跑降到 4–15 cm，不是固定的角度偏差。

sim 实时比的算法：state trace 每 0.02 s 仿真时间一行，行数 × 0.02 ÷ 该局墙钟 `seconds`。
渲染拆分复现（不起栈）：`bash scripts/experiments/warehouse_speed_probe_20260923.sh --repeats 5`。

## 离线 10 局（`outputs/runs/box_room_grasp_offline_20260923`，seed 7000–7009）

配置 `configs/experiments/box_room_grasp_offline_20260923.yaml`，相对 `box_room_grasp_hanging2_20260923` 只改了
`backend.render: offline`。流程是手臂下垂走入 → 守线刹车（`approach_stop: guard`、`guard_margin 0.15`）
→ 原地抬手 → 分段伸手（`reach_path: staged`）→ 抓取 → 后退 0.3 m。

| 局 | 结果 | 站位 depth（m） | 走路时身体离线的最小余量（m） |
|---|---|---|---|
| 00 | timeout | 0.60 | 0.26 |
| 01 | vision_lost | 0.67 | 0.31 |
| 02 | timeout | 0.67 | 0.26 |
| 03 | timeout | 0.53 | 0.21 |
| 04 | timeout | 0.61 | 0.25 |
| 05 | unsupported_assisted_reach | 0.72 | 0.37 |
| 06 | timeout | 0.57 | 0.22 |
| 07 | unsupported_assisted_reach | 0.72 | 0.38 |
| 08 | timeout | 0.65 | 0.29 |
| 09 | timeout | 0.53 | 0.20 |

- depth 取自 `manifest_00.jsonl` 的 `stand_attempts[].depth_m`；余量是 `guard.walk_body_past_line_max` 取负。
- 7 局 timeout 都是手在伸手段够不到：手指离抓点 27–44 cm（前后方向 23–42 cm，高度差约 12 cm）。
  这组数是用会话临时脚本重放 trace 得到的，脚本没有入库。
- 2 局站位 0.72 m，被伸手准入直接拒绝。
- 01 局 `vision_lost` **没有查过原因**。占位帧时间戳的 bug（每次读取打新时间戳，导致估计被判过期）修复前是 6/6 局 `vision_lost`。
- 预览视频（单个 ≤ 500 KB）在该 run 的 `previews/`，全分辨率在 `videos/`。

## 守线余量 0.03（`outputs/runs/box_room_grasp_offline_m003_20260923`，TODO 2.4.10.6）

配置 `configs/experiments/box_room_grasp_offline_m003_20260923.yaml`，相对离线那轮只改 `guard_margin` 0.15 → 0.03，同 seed。
sim 0.96–0.97×。结果 **0/10**：9 局 timeout，1 局 vision_lost。

| 局 | 结果 | 站位 depth（m） | 身体离线余量（m） | 手指离抓点最近（cm） | 同 seed 余量 0.15 时 |
|---|---|---|---|---|---|
| 00 | timeout | 0.47 | 0.13 | 23.6 | 0.60 / 34.6 |
| 01 | vision_lost | 0.41 | 0.08 | 25.8，双手压到箱子近壁，箱子抬 4.6 cm、倾 17° | 0.67 / 43.3 |
| 02 | timeout | 0.52 | 0.18 | 25.0 | 0.67 / 35.8 |
| 03 | timeout | 0.42 | 0.08 | 25.7，双手压到近壁，箱子抬 6.6 cm、倾 24° | 0.53 / 26.6 |
| 04 | timeout | 0.49 | 0.15 | 25.1 | 0.61 / 32.1 |
| 05 | timeout | 0.49 | 0.14 | 24.5 | 0.72（准入拒绝）/ 48.0 |
| 06 | timeout | 0.62 | 0.26 | 33.1 | 0.57 / 28.0 |
| 07 | timeout | 0.50 | 0.16 | 24.2 | 0.72（准入拒绝）/ 48.6 |
| 08 | timeout | 0.52 | 0.17 | 26.7 | 0.65 / 38.0 |
| 09 | timeout | 0.49 | 0.10 | 24.2 | 0.53 / 26.8 |

- 站位中位 0.63 → 0.49 m，平均近了 13 cm，和假设的 12 cm 一致；两局准入拒绝都消失了。走路段没有碰箱，身体离线最少还剩 8 cm。
- 手指差距中位 35 → 25 cm，但**卡在一个固定的位置**：除 06 外都是前后差 19–27 cm、高出 12–13 cm。
  staged 伸手的第一个目标 pregrasp 在抓点上方 8 cm、往身体方向 4 cm（`pregrasp_above 0.08`、`pregrasp_back 0.04`），
  手连 pregrasp 都没到：水平方向还差 15–23 cm。
- 站得最近的两局（01、03，depth 0.41–0.42）手下落到了箱子近壁的顶上，把箱子压斜。近壁在抓点前方 16 cm，
  所以停得再近也不是解法：手要从近壁上方越过去才能到端壁。
- 手指距离由 `scripts/experiments/reach_gap_20260923.py` 重放 trace 得到（每 5 帧取一帧，取两只手里较小的值）。
  上面“离线 10 局”一节的 27–44 cm 是上一会话用临时脚本算的，这里用新脚本重算的同一 run 是 26.6–48.6 cm。

结论（09-24 更正）：停位的问题解决了；"手卡在 pregrasp 前"是伸手时钟冻结的假象，不是包络或反馈问题。修复后同配置重跑，才是第一次真正的离线伸手测量。


## 修复后重跑（`outputs/runs/box_room_grasp_offline_m003b_20260924`，TODO 2.4.10.14）：第一次真正的离线伸手，0/10，三个几何限制

配置 `configs/experiments/box_room_grasp_offline_m003b_20260924.yaml` = m003 原样（余量 0.03、staged、抓深 1 cm）。sim 0.96×。
每局伸手参考时钟都推进了（2.2–7.0 s），10 局里 5 局走到合爪+试提，4 局伸手停滞，1 局抓前把箱子推走。视频在 `previews/`（≤500 KB）。

| 局 | 结果 | 站位 depth（m） | 首次机器人–箱子接触 | 合爪时指垫 z（箱沿 = 33.0 cm） |
|---|---|---|---|---|
| 00 | probe_not_attached | 0.51 | 13.4 s 手指（ready 姿态） | 32.4–32.9 |
| 01 | reach_stalled | 0.55 | 20.2 s 掌（下落段） | — |
| 02 | probe_not_attached | 0.52 | 21.2 s 手指（伸手段） | 见 00 |
| 03 | probe_not_attached | 0.46 | 13.4 s 手指（ready 姿态） | 31.8–32.3 |
| 04 | reach_stalled | 0.60 | 19.8 s 掌（下落段） | — |
| 05 | reach_stalled | 0.54 | 21.5 s 掌（下落段） | — |
| 06 | lost_attachment | 0.58 | 20.4 s 掌（下落段） | 32.2–33.1 |
| 07 | lost_attachment | 0.57 | 20.0 s 掌（下落段） | 见 06 |
| 08 | reach_stalled | 0.57 | 19.9 s 掌（下落段） | — |
| 09 | basket_displaced_before_capture | 0.40 | 13.9 s 手指（ready 姿态），箱子被推起 13.6 cm | — |

三个限制（都是几何的，按站位分层）：

1. **站位 ≤ 0.51 m：ready 姿态本身压在近壁上。** `prepare_stationary_ready` 把掌心放在向前 0.27 m，加上 0.19 m 的手指，指尖到 0.46 m；
   近壁在箱心前 0.16 m。抬手不是"原地抬升"，而是斜向前伸出去，第一次接触发生在抬手期间（TODO 2.4.10.12 的直接证据）。
2. **站位 ≥ 0.54 m：下落时掌先落在壁顶上。** 远站位 IK 以姿态换位置、掌俯仰丢失（09-23 walk 场景已量过），掌比指垫先到壁顶，
   下落被顶住 → `reach_stalled`；06/07 爪口还是卡到了沿边，合爪后一抬就掉。
3. **站位 0.46–0.52 m：伸手到位，但抓深 1 cm 夹不住。** 下落末端工具点误差 ≤ 1.5 cm，指垫几何中心在箱沿下 0.1–0.7 cm，
   对 1 cm 厚的壁只夹到顶上几毫米；试提 2.5 cm 时箱子只起 0–4 mm（`probe_not_attached`）。合爪时掌离箱沿还有约 10 cm，
   所以 09-23 "2.5 cm 抓深让掌压到壁顶" 的顾虑在这些站位上不成立——深抓的余地是有的。z 向跟踪修正在远站位已顶到上限
   （−0.14 / cap 0.15），近站位（03）只用了 −0.05…−0.11，深抓要在近站位试。

数据来源：`observations/<局>/observations.jsonl`（`low_reach.capture_error`、`tracking_correction`、`hand_openings`）；
指垫位置 = trace 重放 `mj_kinematics` 取 `*_dex1_finger_*_link` 的指尖 geom（group 0，沿指 10.6 cm）——注意 finger link 的
body 原点在指根，用 body 位置会高估 8 cm（本会话先犯了这个错）。首次接触 = trace 重放 `mj_forward` 查 `d.contact`。

下一步（各改一个量，10 局）：抓深 `box_grasp_below_rim` 0.01 → 0.03（限制 3）；ready 姿态改原地抬升（2.4.10.12，限制 1）；
远站位的散布（0.46–0.60，由守线刹车决定）要么收紧站位，要么给下落段补掌俯仰（限制 2）。


## 抓深 0.03（`outputs/runs/box_room_grasp_offline_deep03_20260924`，TODO 2.4.10.15）：0/10，比 1 cm 更糟，暴露了横向误差和修正积分

配置 = m003b 只改 `box_grasp_below_rim` 0.01 → 0.03。结果 0/10：8 局 `reach_stalled`（全在下落段）、2 局抓前推走箱子（0.42、0.45 m，ready 姿态）。
**没有一局合爪**（m003b 有 5 局）；6/10 局箱子被推斜 > 5°（m003b 1 局），5 局被顶起 11–15 cm。

| 局 | depth | 结果 | 下落末端 | 
|---|---|---|---|
| 00 | 0.47 | stalled | 右手两指垫都在壁外（x 34/24，壁 22），左手都在壁内（−12/−20）；预抓点就偏了 |
| 01 | 0.47 | stalled，箱子起 11 cm 倾 29° | 左腕勾在近壁顶上，右指垫压外壁面 |
| 02 | 0.56 | stalled，倾 10° | 左指垫压内壁面 |
| 04 | 0.45 | stalled，起 15 cm 倾 45° | 两手各一指垫压壁面，把箱子推倒 |
| 06 | 0.53 | stalled | 左手指垫压内壁面，横向误差 −4 cm 从下落开始就在 |
| 07 | 0.49 | stalled，起 11 cm 倾 28° | 左腕压近壁顶，右指垫压外壁面 |
| 08 / 09 | 0.57 / 0.52 | stalled | 远站位，末端无接触 |
| 03 / 05 | 0.42 / 0.45 | displaced | ready 姿态推走箱子（同 m003b 09） |

机制（逐 tick，`observations.jsonl` + trace 接触重放）：

1. **跨壁方向的横向误差在下落开始时就有 2–6 cm**（指垫中心 vs 壁面；m003b 同样有，只是 0.5 cm 抓深让指垫只擦到壁顶边缘）。
   爪口开 9.5 cm、壁 1 cm，每侧本有 4 cm 余量；误差 3–6 cm 时一侧指垫在下落 1–2 cm 后就顶到壁面。
2. **跟踪修正没有接触意识**：指垫顶住壁面后，横向修正从 −2 cm 一路积分到上限 ±15 cm（`tracking_cap`），把壁往侧面推；
   修正饱和后箱子开始翻（04 局：修正到 −15 cm 后倾角 8° → 41°）。这是把箱子推倒的力，不是下落本身。
3. 深抓本身可行：指垫到了箱沿下 3–7 cm，掌离沿仍有余量；限制不在几何，在横向对准和修正律。

判定：**抓深 0.03 在现有对准精度下不能用；先解决横向对准，再回来加深。** 1 cm 抓深也夹不住（m003b），所以这条线需要：
(a) 下落前用视觉/几何把跨壁误差压到 < 2 cm，或下落时按壁面接触做居中；(b) 修正律在接触后停止积分（横向 anti-windup），
否则任何深抓都会把箱子推倒。两者都是控制改动，不是旋钮。

数据：横向误差 = 两指尖 geom 中心 x（箱系）减壁面 x；接触 = `mj_forward` 后 `d.contact`；修正 = `low_reach.tracking_correction`。


## ready 原地抬升（`outputs/runs/box_room_grasp_offline_lifted_20260924`，TODO 2.4.10.12）：0/10，但抬手撞箱这一层没有了

配置 = m003b + `ready_pose: lifted` + `elbow_min_deg: -60`（提交 `ed5819d`：工具点解到向前 0.16、左右 ±0.16、上 0.05 m，掌俯仰 30°，
锁腰；两个求解器的 5° 肘下限改成旋钮，lifted 需要小臂向上折）。

| 局 | depth | 结果 | 首次接触 | 抬手时指尖离近壁 | 下落最佳误差（箱系 cm，L / R） |
|---|---|---|---|---|---|
| 00 | 0.44 | reach_stalled | 24.2 s 伸手段 | 8.7 cm | x −3.9 / +2.9，横向修正 +15 饱和 |
| 01 | 0.51 | reach_stalled | 23.7 s | 18.4 | x −1.3 / +3.6，修正 +14.6 / +11.9 |
| 02 | 0.43 | probe_not_attached | 22.3 s | 8.9 | ≤ 1.1 |
| 03 | 0.48 | reach_stalled | 21.3 s | 12.9 | z +1.6 / +2.6，修正未饱和 |
| 04 | 0.41 | reach_stalled | 20.9 s | 4.9 | x +4.9（右），修正 −15 饱和 |
| 05 | 0.58 | reach_stalled | 22.0 s | 22.2 | y −2.8，z +1.7 |
| 06 | 0.49 | probe_not_attached | 23.1 s | 13.4 | ≤ 1.1 |
| 07 | 0.47 | probe_not_attached | 21.4 s | 12.1 | ≤ 1.6 |
| 08 | 0.53 | reach_stalled | 23.3 s | 18.4 | x −5.0 / +2.6，修正 +15 饱和 |
| 09 | 0.50 | probe_not_attached | 23.1 s | 13.9 | ≤ 1.7 |

- **限制 1 解决**：10/10 局第一次接触都在伸手段（此前三轮里 0.40–0.51 m 的站位有 5 局在抬手期间接触，2 局把箱子推走）；
  箱子位移 0/10（deep03 2、m003b 1）。0.41–0.44 m 的站位现在能走到下落段。
- SONIC 跟住了小臂上折的姿态：ready 后伸手参考正常推进（3.7–7.6 s），没有新的失败形态。
- 剩下的失败全在下落段：4 局（00、01、04、08）是 deep03 里看到的**横向修正顶壁饱和**（±15 cm 上限，实测误差仍 4–5 cm），
  在 1 cm 抓深下也出现了；2 局（03、05）停在目标上方 1.6–2.7 cm，z 修正没饱和（−10…−14），被 3 s 无进展规则判停。
  4 局合爪+试提都是 `probe_not_attached`（1 cm 抓深夹不住，同 m003b）。
- 下一步就是 2.4.10.17（横向对准 + 修正 anti-windup），然后回抓深。

## 哪些旧结论作废

以下结论都是在 0.35× 慢仿真里测的，需要在满速下重测，不能直接引用：

- 守线余量 0.15 m：当初是为防撞箱加的；离线时身体离线仍空 20–38 cm，应回到 0.03 m（TODO 2.4.10.6）。
- `box_room_grasp_20260923` 的守线 6/10 越线。
- 扩包络的 carry 块把骨盆带后 0.3–1.3 m 导致的摔倒（TODO 2.4.7）。
- `warehouse_speed_20260923` 的横偏数据：`diag_brake` 不做 reanchor，起始朝向 90° 的场景一起步就转约 90°。速度数据仍然有效。

## 怎么跑

```bash
cd /workspace/code/locomanipulation
. scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"
CFG=configs/experiments/box_room_grasp_offline_20260923.yaml   # 复制一份，改 name 和要试的一个旋钮
bash scripts/start_stack.sh collect $CFG --gpu 0
bash scripts/bimanual/run_agent_server.sh $CFG --gpu 0
bash scripts/stop_stack.sh --instance 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/render_trace_videos.py outputs/runs/<run>
```

离线模式的用法和限制见 [仿真与渲染](../rendering.md#离线渲染backendrender-offline)。

## 未做 / 下一步

- ~~2.4.10.6~~ 已做，见上一节：停近 13 cm，0/10，手卡在 pregrasp 前。
- **2.4.10.7**（已判定 09-24）：不是伸手段的问题，是离线伸手时钟冻结；修复后重跑见 m003b 一节。
- **2.4.10.15**（已做）：抓深 0.03 更糟（0 合爪，6 局推斜），见上一节；先做横向对准 + 修正 anti-windup（2.4.10.17）。
- **2.4.10.12**（已做，`ed5819d`）：ready 原地抬升，抬手接触 0/10、箱子位移 0/10；见上一节。
- **2.4.10.16** 远站位掌俯仰（这轮 ≥0.53 的 2 局仍停滞，但主因已是横向修正）。
- **2.4.10.8 纸箱网格减面或关阴影**：VLA 闭环评测必须实时渲染，所以这一步最终绕不开。离线测过，隐藏网格后渲染从 39 ms 降到 2 ms。
  `assets/crates/nor_cal_box/visual/` 是协作者还没入库的目录，要先跟协作者商量。
- **2.4.10.9**：给 `diag_brake` 补 reanchor。

## 代码（`b826594`；`backend.render` 以外的开关都默认关闭）

- `backend.render`：字段在 `src/locomani/backends/sonic/config.py`，sim 端在 `src/locomani/sim/run_sim_locomani.py`，
  起栈时不等图像、不设视频目录在 `src/locomani/stack/manager.py`；专家的占位帧 `PlaceholderCamera` 在 `src/locomani/tasks/crate/expert.py`。
- 守线刹车的落脚预测在 `src/locomani/primitives/foot_watch.py`；手臂下垂走入（`HangingWalk`）、站起后重放置、按世界轴行走、
  `walk_body_past_line_max` 都在 `src/locomani/tasks/crate/expert.py`；后退在 `src/locomani/tasks/crate/retreat.py`。
  字段说明见 [config_fields](../reference/config_fields.md)。
- 不录制的栈按 5 核预留（`NO_EXPORTER_CORES`），见 `src/locomani/pipeline/run_context.py`。

## 离线伸手时钟冻结（2026-09-24，TODO 2.4.10.7 的判定）

- 证据：两轮离线 run 每个 reach tick 的 `low_reach.reference_elapsed` 都是 0.00（计划 3.0 s）；逐 tick `reason` 全部 `waiting_for_vision`；
  记录里 `image_timestamps − wall_time` 恒为 +0.1…+0.6 ms。实时的 `box_room_grasp_20260923`、`box_room_20260922` 同字段正常推进到 3–4.7 s。
- 机制：`expert.py` 循环先取 `now = time.time()` 再 `camera.read()`；`PlaceholderCamera` 在读取时打 `time.time()`，慢于 1/30 s 的 tick 每次都重打，
  于是 `CoordinatedReach.step` 的 `0 <= wall_time − estimate.timestamp` 不成立。只有没重打的 tick 被接受（所以没触发 `vision_lost`、跟踪修正在慢慢累积），
  但接受 tick 紧跟拒绝 tick 时 `dt` 被清零，参考时钟永远是 0。实时相机的戳来自 sim、早于 `now`，所以实时不受影响。
- 修复：读完相机再取 `now`（`expert.py`，一处），`tests/test_crate_guard_retreat.py` 加了占位帧戳 = 读取时刻的契约测试。
- 影响：离线模式此前从未真正执行过伸手；"离线 10 局"和"余量 0.03"两节里关于手指差距的行只反映 ready 姿态到抓点的几何距离。
  站位、守线、走路段碰箱的结论不受影响。
