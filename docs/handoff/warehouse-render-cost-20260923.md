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
- **2.4.10.7**（已判定 09-24）：不是伸手段的问题，是离线伸手时钟冻结；修复后重跑 m003 配置。
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
