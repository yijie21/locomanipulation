# pick_place 台账

走到桌前,右手抓起罐头,放进左边的篮子。设计见 `docs/handoff/pick_place_plan.md`,实现在
`src/locomani/tasks/bottle/place_phase.py`(bottle 专家的 1–3 步原样复用,lift 之后由
`BottleTaskConfig.goal` 切到 place 阶段),Task 注册为 `pick_place`(`src/locomani/tasks/pick_place/`),
配置 `configs/tasks/pick_place.json`,场景 `SCENE_PRESET=pick_place`(`outputs/scene/pick_place`,
`build_sim_scene.py --preset pick_place --harden-pads`)。

## §1 场景(2026-09-10)

- 篮子:SIMPLE `objects:0`(藤编篮),`assets/simple/basket`,0.30 × 0.20 × 0.15 m,中心 (-0.37, 0),yaw 90°,
  近壁离桌近边 3 cm。内矩形半宽 (0.12, 0.07)(篮子自身系),沿口 z 0.772。
- 罐头:SIMPLE graspnet `tomato soup can`(id 2)的 mesh,`assets/simple/soup_can`,x/y 缩 0.848、z 拉 1.321
  → 6.0 × 14 cm 细罐;(-0.40, -0.28),机器人右手边。grip 在罐心上方 3 cm(`reach.grip_below_center -0.03`),
  grip_z 0.722 = water bottle 的 0.723。
- 桌面 0.62 m(和 bottle 线一致)。dex1 指面 4 个碰撞 geom 硬化(priority 1, condim 4, solref 0.004,
  solimp 0.99/0.999;agentic 分支 §8 的 L1 参数),只在 `--harden-pads` 构建时生效,bottle/crate 的 gate 场景不变。
  2026-09-11 起这组常数是 `build_sim_scene.PAD_LEVELS["object"]`,与双臂线的 `"crate"` 档共用同一个实现
  (`harden_finger_pads(spec, level)`);`bimanual_scene.py` 原本有第二份实现和第二套常数。

## §2 为什么是这个罐头(四次 smoke,每次 2–4 局,seed 32000)

| 版本 | 结果 | 原因 |
|---|---|---|
| 9 cm 罐,0.62 桌 | 第一局抛异常 | grip_z 0.647:ready-grasp baseline 的 FK 模板只覆盖 grip_z ≥ 0.70(舒适下蹲 0.03–0.13),"target height has no comfortable baseline grasp" |
| 9 cm 罐,0.70 桌 | 0/2,[grasp] | 左手 ready 姿态碰桌(guard "left finger vs table"),另一局 "stand lacks a clear squat and short grasp path" |
| 20 cm 罐,0.62 桌,grip 在罐心 | 0/2,[grasp] | coupled squat 的 safe-hand sweep 报 palm(wrist_yaw_link)撞罐:俯仰 23° 的手掌压在 grip 上方 10 cm 的罐身上。bottle 没事是因为 grip 上方只有 5.7 cm 细瓶颈 |
| **14 cm 罐,grip 罐心上 3 cm** | 1/4 OK(1.1 cm),1 [turn],1 [stand],1 [carry] | 抓取通了;[turn] 是放置带太窄(见 §3),[carry] 是持物滑脱(见 §3) |

结论:grip 高度 0.70–0.73,grip 上方物体 ≤ 4–5 cm,直径 ≤ 6.5 cm,是这条抓取线不改标定就能接受的物体形状。
`grip_below_center` 的校验范围放宽到 [-0.05, 0.03],`TabletopTarget` 允许 grip 在罐心之上但仍在物体内。

## §3 shakedown pilot(seed0 32100,15 局,10 成功;`outputs/experiments/pick_place_pilot_20260910`)

batch 00 6/10,batch 01 4/5。失败:[grasp] 2(local alignment disturbed;stand infeasible——bottle 线常见的
discard 类),[carry] 1,[turn] 1,[retreat] 1。三个 place 侧问题与修法,都已在 batch 01 或之后生效:

1. **[carry] 持物滑脱 3.4–3.8 cm(place_carry 段)。** 不是接触蠕变:10 Hz 的 `place_trace` 显示修后各段漂移
   ≤ 0.7 cm。原因是 place 伺服每拍用 `upper_ref.MAX_STEP`(2°/拍 = 1.7 rad/s)限幅,冗余 IK(腰 + 7 臂关节)
   逐拍在姿态间跳,手抖把罐头甩松。改为 `JOINT_STEP` 0.01 rad/拍(0.5 rad/s;0.15 m/s 参考在 0.35 m 力臂上只要
   0.45 rad/s)。指面硬化同时加上,但单独效果没分离。
2. **[turn] 篮子进不了放置带(x 0.39–0.42)。** 骨盆停在桌边外 ~0.25 m 就不能再近(脚/膝伸出 0.22 m),篮心
   离边 0.34 m 时在骨盆系 x 0.38–0.43,原带 [0.22, 0.30] 靠走路根本到不了。离线 IK(策略同样的舒适限位 + 腰):
   z_pf +0.10–0.13 处 x 0.40 误差 0.3 cm,0.44 误差 5 cm。改:带 [0.26, 0.44],hover 从 +5 cm 降到 +3 cm,place
   期间腰 pitch 上限 0.20 → 0.35 rad,篮子近壁移到离边 3 cm(x -0.37)。
3. **放置后罐头倒在篮里(final offset 6.1–6.5 cm 而释放时 xy 误差 1.3–1.8 cm)。** 从沿口上 3 cm 松手是 17 cm
   自由落体,罐头每次都倒;-6 cm(8 cm 落差)一半倒。改为 -10 cm(离篮底 4 cm),收手先竖直抬 14 cm 再后退。
   "倒在篮里"按判据仍算成功(SIMPLE 也不要求竖直),这里是为了数据一致。同一现象曾被 [retreat] "moved 7 cm
   after the release" 错误归因,sweep 检查的参考点改到手抬离之后取。

放置精度(释放时罐心到篮心 xy):1.3–2.6 cm,全部按时释放;篮子在 ego 帧内 100%。单局 37–42 s。
视频:`outputs/videos/pick_place_pilot_20260910/reel_10ok.mp4`(10 局成功,4.4 min;clips.txt 是索引)。

## §4 验收 pilot(seed0 32300,最终配置;`outputs/experiments/pick_place_pilot2_20260910`)

19 局 10 成功(batch 00 5/10,batch 01 5/9),单局中位 42 s。**9 个失败全部在抓取侧**,place 阶段在抓起的 10 局里
10/10:[grasp] 局部对准撞动罐头 2、stand infeasible 2、局部对准超出舒适 IK 2、lift 时罐头倒 1、safe-hand IK
没跟上 1、coupled squat 没交付 1。这些都是 bottle 线 visible expert 自己的 discard 类(memory
`visible-expert-collection-facts`:yield ~0.62,19% infeasible-stand);这里 yield 0.53(shakedown 0.67),
样本太小分不出是否比 bottle 差。可疑点:6 × 14 cm 罐比 4.3 cm 瓶更容易被指尖推倒(grip 在罐心上 3 cm,力臂
10 cm),"disturbed target" 3 局 + "toppled in lift" 1 局。

place 侧数字(10 局):释放时罐心到篮心 xy 1.4–2.6 cm,全部按时释放(未触发 4 s 超时);篮子在 ego 帧内 100%;
持物期间掌系漂移最大 0.3–1.0 cm(§3 的 JOINT_STEP 修法有效);最终偏移 0.9–2.8 cm 7 局,4.8–7.3 cm 3 局
(罐头落地后倒下,仍算成功)。spawn gate `goal_gap` 0.21 从 batch 01 起生效,batch 00 里 y ∈ [-0.16, -0.12]
的两局都是抓取失败。

视频:`outputs/videos/pick_place_pilot2_20260910/reel_10ok.mp4`(10 局,4.6 min)。

## §5 未决

- 抓取侧 yield:先用同 seed 跑 water bottle 对照(same-seed paired),看 0.53 是罐头形状还是噪声。
- 罐头落地后倒:release -0.10 已是离篮底 4 cm;剩下的是松手时罐头在指间的倾斜,可看 `place_trace` 前的
  最后姿态或让 descend 结束时保持 0.3 s 再张开(与"不悬停"原则冲突,要权衡)。
- 篮子和罐头位置尚未随机化(plan §4);Isaac render 未验证篮子贴图。

## §6 bottle 行为闸门(2026-09-10 15:09–15:14 UTC,`outputs/experiments/gate_bottle_pickplace3_20260910`)

pick_place 改了 bottle 线共用的主代码(`round_loop`、`config`、`round_phases_approach`、`reach_config` /
`tabletop_grasp` 校验、`build_sim_scene`、`run_learn_pilot`),按规范跑 `scripts/gate.py compare bottle`:
bottle_visible.json、seed 29000、10 局、release policy、instance 0 / GPU1,与 baseline 同配方。7/10,PASS 16/16。
(`run_learn_pilot` 因 7 < 8 successes 退出 1,与闸门无关;前两次尝试分别因端口竞争和 exporter 名字检查作废。)

```
  PASS  yield                         0.6 -> 0.7
  PASS  end_depth_m                   0.3011 -> 0.3065
  PASS  squat_drop_m                  0.13 -> 0.13
  PASS  exit_majority                 visible brake depth -> visible brake depth
  PASS  stages                        new [] missing []
  PASS  failure_reasons               new []
  PASS  stage_s.initialize            11.5371 -> 11.7168
  PASS  stage_s.navigate              4.178 -> 4.5942
  PASS  stage_s.approach              0.0509 -> 0.0423
  PASS  stage_s.recovery_brake        1.2982 -> 1.155
  PASS  stage_s.grasp.coupled_squat   1.481 -> 1.4241
  PASS  stage_s.grasp.local_align     2.0922 -> 1.7823
  PASS  stage_s.grasp.close           2.0285 -> 2.0293
  PASS  stage_s.grasp.lift            2.7011 -> 2.7007
  PASS  stage_s.carry_and_verify      2.8803 -> 2.3916
  PASS  stage_s.verify_terminal_hold  2.0324 -> 2.0337
[gate] bottle: PASS (16/16)
```


## §7 skills 目录三步 plan（approach → grasp_single → place）冒烟：1/3（2026-09-28）

入口 `scripts/plan_episode.py`，实验 `configs/experiments/skills_pick_place_20260928.yaml`（= `pick_place.yaml`，场景改为
`outputs/scene/pick_place`，不录制），profile `bottle_simple`（**`simple` 抓取，不是本台账此前的 `coupled` / `ready_grasp_baseline`**），
GPU1 / instance 1，seed 200–202。`place` 跑在自建的 PlannerArm 上（`place_phase._phase_place` 原样）。

| 局 | 结果 | approach 停位 | 各 primitive |
|---|---|---|---|
| 0 | FAIL `[grasp] simple no bilateral pad contact after the close` | 0.37 / −0.18 m | approach 9 s → grasp 10 s |
| 1 | FAIL `[place] not over the container at the descent timeout (xy 17.5 cm)` | 0.32 / −0.19 m | approach 9 s → grasp 13 s → place 9 s；turn 后容器在骨盆系 x 0.43 y −0.04 |
| 2 | OK，罐头离篮心 7.3 cm，篮子 ego 可见 100% | 0.41 / −0.24 m | approach 8 s → grasp 15 s → place 19 s；下降超时 xy 4.5 cm 仍释放；撤手抬升差 30 cm 超时 |

- 结论是**管路通了、标定没做**：`simple` 抓罐头和 `simple` 之后接 place 都是第一次跑，三局三种结局。
  §4 的 10/10 是 `coupled` 策略 + 当时的 place 参数；这里换了抓取策略、站位规划（`stand_plan` 无抖动）和握持偏移来源。
- 局 1 的 17.5 cm 是 carry-over 伺服没把罐头送到篮心（turn 后容器 x 0.43 m，比 §4 的 0.35–0.38 m 远），局 2 的 4.5 cm 超时释放也偏远；
  下一步先量 place 前的站位深度，再决定是 `place` 前加一步 `approach(basket)` 还是调 `band_x`。
- 无录像（运行用的入口还没接 `SIM_VIDEO_DIR`）。逐 primitive 记录在 `outputs/runs/skills_smoke_pickplace_20260928/skills/`。

## §8 持物 `approach(basket)` 接 `place`：站位从抓取站改成放置带中心（2026-09-28）

TODO 7.7。入口 `scripts/plan_episode.py`，plan `configs/plans/pick_place_via_basket.json`（approach → grasp_single → **approach(basket)** → place），
`skills_pick_place_20260928.yaml`，GPU1 / instance 1，seed 300–309。持物 approach = `carry_to` 到站位（上身冻结、掌系漂移守卫）+ 原地 slow_walk 转向桌法线，
写 `Stance`（容器在骨盆系的深度 / 横向）。

| 运行 | 站位来源 | 到 place 的局 | place OK | 容器停位（深度 / 横向，m） | 失败 |
|---|---|---|---|---|---|
| `skills_pickplace_viabasket_20260928` | `stand_plan` 的抓取站（右手抓取几何） | 6/10（4 局 `[grasp] simple no bilateral pad contact`） | 2/6 | 0.26–0.42 / **−0.23…−0.32** | `[turn] container not in the placing band` ×3（y −0.15…−0.20 剩余）、`[place] descent timeout xy 8.5 cm` ×1 |
| `skills_pickplace_viabasket2_20260928` | 放置带中心（`band_x`/`band_y` 均值 = 0.35 / −0.11），`carry_to` 停距 0.06 | 7/10（3 局 grasp 失败） | 1/7 | 0.32–0.43 / **−0.19…−0.38**（瞄的是 −0.11：原地 slow_walk 转向漂移 0.1–0.27 m） | `[turn]` ×2、`[place] descent timeout` 7.8 / 34.2 cm ×2、`[approach] load slipped 3.4 cm`（转向中）×1 |
| `skills_pickplace_viabasket3_20260928` | 放置带中心 + 转向后残差修正（`carry_to(relative, turn_first=False)` ≤ 2 次，`CORRECT_TOL` 0.05） | 6/10（4 局 grasp 失败） | **3/6** | **0.33–0.39 / −0.08…−0.16**（6/6 进带；修正前残差 6–29 cm，修正 1–2 次） | `[place] descent timeout` 6.0 / 9.0 cm ×2、`[carry] raise timed out 4.7 cm short` ×1 |
| `skills_pickplace_direct10_20260928`（对照，§7 的三步 plan） | 无：抓完原地 place | 5/10（5 局 grasp 失败） | 0/5 | 转向段自己能把容器收到 x 0.33–0.35 / y −0.02…−0.09（带内） | `[place] descent timeout` 7.3 / 11.8 / 11.7 cm ×3、`[retreat] object moved 6 cm after release` ×1、`[carry] raise timed out 6 cm short` ×1 |

- 抓取站把容器放在右手抓取的横向（−0.2 m 左右），而 place 的转向段要它落在 `band_y` [−0.16, −0.06] 里：第一轮 3/6 在转向段 8 s 超时。
  第二轮把站位瞄到放置带中心，但**原地 slow_walk 转向本身漂移**，容器仍落在 −0.19…−0.38；`_loaded` 现在在转向后量残差、
  用 `carry_to(relative, turn_first=False)` 最多补两次（`CORRECT_TOL` 0.05）：第三轮 6/6 把容器送进放置带（横向 −0.08…−0.16），转向段不再超时，place 3/6。
- **place 前站位深度不是主因**：直接 place 的对照里转向段自己就把容器收进带内（x 0.33–0.35 / y −0.02…−0.09），
  之后仍是 `descent timeout` 7–12 cm（carry-over / 下降伺服没把罐头送到篮心），与 §7 局 1 同族。§4 的 10/10 是 `coupled` 抓取 + 当时的握持偏移；
  `simple` 抓罐头后的握持几何（`palm_frame_offset`）和 place 的伺服基准还没对过——这是 7.10 流程里"按失败族调一个旋钮"的第一个对象，本会话没调。
- 抓取本身 6/10：`simple` 抓罐头的 "合爪无双侧接触" 是这条线的主失败族（§7 也见），与走近 / 放置无关，属于 7.10 的标定项。

## §9 `claude -p` 规划器接真栈冒烟：`--agent`，3 局（2026-09-28）

TODO 7.6。`skills_agent_smoke_20260928`：`skills_pick_place_20260928.yaml`，`--agent --agent-model sonnet --max-steps 10`，seed 300–302。
每局一个 `claude -p` 会话（`--json-schema` 结构化输出、`--tools ""`、`--session-id` / `--resume`），请求 = 任务 + 菜单 + 目录手册 + 特权状态文本 + session 快照 + 上一条 trace；
全部落在 `skills/agent_io_NN.jsonl`。

| 局 | 调用数 / 用时 | agent 的序列 | 结果 |
|---|---|---|---|
| 0 | 10 / 306 s | approach(profile **pick_place** → 拒) → approach → grasp_single（合爪无接触）→ approach ×3（`[gym] object toppled during the settle`：罐头已被上一次抓倒）→ grasp_single（拒：无准备好的策略）→ navigate_to(stand_off 1.0，成功退开）→ approach（`[stand] out of reach`）→ turn_to | FAIL（罐头倒了，物理上不可恢复） |
| 1 | 10 / 217 s | approach(profile **simple** → 拒) → approach → grasp_single OK → place（`[carry] object slipped 3.2 cm`，罐头掉了）→ approach ×2（toppled）→ grasp_single（拒）→ approach → navigate_to → approach | FAIL |
| 2 | 10 / 196 s | approach(profile **soup_can** → 拒) → approach → grasp_single OK → place（`[carry] slipped 3.0 cm`）→ approach ×3（toppled / `[stand] neutral waist not delivered`）→ grasp_single（拒）→ approach ×2 | FAIL |

- **管路通了**：JSON 进出、被拒调用带 `check` 回传、agent 读到后改参数重来（两局的第一次都猜错 profile 名，第二次就用了目录里的名字）。
  之后已把可用 profile 名列进请求（`profiles` 字段）。
- 每次调用 8–65 s（sonnet，含思考），一局 10 次约 4–5 min；采集时用 4 s/次的估计（2026-09-03 Q8）已过时。
- 3 局里 2 局 grasp 成功后 place 的 carry-over 段掉罐（掌系滑 3.0–3.2 cm，与 §8 直接 place 的失败同源）；失败都是物理失败（抓倒、掉落），agent 的重试没有可恢复的路径——这正是 Q11"注入失败 paired 比较"要设计的地方，不在本次范围。

## §10 `navigate_to` 持罐冒烟（2026-09-28）

TODO 7.9。plan `configs/plans/pick_place_navigate.json`（approach → grasp_single → **navigate_to(xy [1.2, −1.6])**，桌边另一侧 1 m 多外），
`skills_navigate_smoke2_20260928`，seed 310–313。第一版（`skills_navigate_smoke_20260928`，目标 = 篮子周围 0.6 m 环）站位本来就在环里，0 段即"到"，不算样本。

| 局 | 结果 |
|---|---|
| 0、1、3 | grasp 失败（合爪无接触 ×2、抬起时失接触 ×1），没到 navigate |
| 2 | **navigate_to OK**：7 段路点、走 2.57 m（绕桌角）、其中 4 段先 slow_walk 转向再走、掌系最大漂移 0.8 cm、距离场剩 0.10 m |

- 持物分段走法（每段转向 + 守线走）在一个样本上成立；判据按 plan 末尾 primitive 选（持物即可），否则会被"没放进篮子"判 FAIL（第一版就是）。

## §11 `simple` 抓罐头 + place 标定：按 docs/skills.md §5 流程一次一个旋钮（2026-09-28 下午）

TODO 7.7.1。基线 = §8 第三轮 `skills_pickplace_viabasket3_20260928`（`bottle_simple`，seed 300–309）：grasp 6/10，place 3/6。每个变体是 `bottle_simple_200.yaml` 的
复制件改一个旋钮（`configs/experiments/skills_can_*_20260928.yaml`，profile 名登记在 `PROFILE_SOURCES`），同 plan、同 seed，`--profile <名>`。
`grasp_single` 的 trace 从这轮起带 simple 策略的 `stats`（entries / failure 证据）。

| profile | 旋钮 | grasp | place（到 place 的局） | 失败 |
|---|---|---|---|---|
| `bottle_simple`（基线） | — | 6/10 | 3/6 | grasp：无双侧接触 ×2、抓倒 ×2；place：descent timeout 6–9 cm ×2、raise 4.7 cm short ×1 |
| `can_entry15` | `entry_lateral_tolerance_m` 0.025 → 0.015 | 7/10（1 局站位 out of reach 未到抓） | 3/6 | grasp：无双侧接触 ×2、抓倒 ×1；place：**raise timed out 13.5 / 16.7 cm short** ×2、descent timeout 6.2 cm ×1 |
| `can_descend05` | `v_descend` 0.1 → 0.05、`descend_timeout` 4 → 8 | 6/10 | 2/6 | grasp：无双侧接触 ×3、抓倒 ×1；place：descent timeout 6.0 / 9.9 cm ×2，**82 / 94 cm** ×2（下降中罐头掉出、量到地上）——减速没帮助，负结果 |
| `can_overshoot0` | `entry_overshoot_m` 0.02 → 0.0 | 4/9（1 局站位 out of reach） | 0/4 | grasp：无双侧接触 ×3、抓倒 ×2；place：descent timeout 5.7–7.7 cm ×3、掌系滑 3.8 cm ×1。stats 里罐心仍在工具点前 0.7–1.1 cm——过冲不是深度的来源，负结果 |

- **横向容差不是抓取失败的原因**：`can_entry15` 的 stats 里所有 "无双侧接触" 局在合爪前都 `aligned=True`、横向偏差 ≤ 1.5 cm；共同点是罐心在工具点**前方** 0.8–1.0 cm
  （`offset_in_palm[0]`），即罐头比瓶子更深地进到指根——罐头直径 6.1 cm、瓶子 4.5 cm，同样的 2 cm 进入过冲对罐头太多。下一个旋钮因此是 `entry_overshoot_m`。
- place 的 "raise timed out N cm short" 是第二个失败族：抬到悬停高度的物体伺服（`v_travel` 0.15、`carry_timeout` 6 s）根本没把罐头抬起来（差 13–17 cm）；与 descent timeout 同为 PlannerArm 上物体伺服的跟踪问题。

- **三个旋钮都没动这两个失败族**（grasp 6/10 → 7/10、6/10、4/9；place 3/6 → 3/6、2/6、0/4，都在 10 局噪声内或更差）。证据指向旋钮够不着的地方：
  合爪前对准、罐头还是没有双侧垫接触，是 6.1 cm 罐身与 Dex1 指垫的闭合几何（`close_target` / 接触判定），不是进入；place 的 xy 漂移是 PlannerArm 上物体伺服的跟踪。
  §4 的 10/10 用的是 `coupled` 策略。试着把 `skills_pick_place_20260928.yaml` 的 task 块登记成 profile（`pick_place_coupled`）跑同一 plan：
  第一局就在 `_phase_approach` 里抛 `CoupledGraspPolicy has no attribute 'walking'`——skills 的 approach 只包了 `simple` 专家的阶段，coupled 的走近读的是另一套策略字段。
  现在 `approach` 的前置直接拒绝 `reach.strategy != simple` 的 profile（`check: profile`），登记已撤。
- **7.7.1 到此的状态**：三个 simple 旋钮各 10 局都没动两个失败族；能救罐头的 `coupled` 策略需要单独包装（design §9 第二步"把抓取核心搬进 catalogue"的范围），本会话不做。

## §12 `coupled` 策略经 skills 管路（7.7.1.1 走 A）：包装、10 局、掌系滑动容差（2026-09-28 下午）

包装 = 三处修补：`round_phases_approach` 读 `ctx.coordinated.walking` 改为 `getattr`（只有 whole_body / simple 有），
`RecoverableGraspPolicy.walk_tick` / `ReadyGraspPolicy.walk_tick` 接受走近阶段传的关键字（`edge_distance` 等，忽略），
`approach` 前置放行 `reach.strategy ∈ {simple, coupled}`。profile `pick_place_coupled` = `skills_pick_place_20260928.yaml` 的 task 块（§4 的配置）。
plan `pick_place_via_basket.json`，`--profile pick_place_coupled`，seed 300–309（与 §8/§11 同）。

| 运行 | 旋钮 | grasp | place（到 place 的局） | 失败 |
|---|---|---|---|---|
| `skills_can_coupled_pilot10b_20260928` | — | **5/10** | 2/5 | grasp：`visible expert: stand infeasible, no reposition` ×3（停位深 0.24–0.26 / 横 −0.28…−0.30）、局部对齐扰动 ×1、抬起时翻倒 ×1；持罐走向篮子时**掌系滑 3.2–3.7 cm** ×2（守卫 3 cm）、place carry-over 滑 3.6 cm ×1 |
| `skills_can_coupled_drift05_pilot10_20260928` | 持物 approach 与 place 的掌系容差 0.03 → 0.05 | **2/10** | 1/2 | grasp：`local alignment disturbed target or base` ×5、抬起 0 m ×1；持罐 approach(basket) **转向中掉罐** ×3（容差放宽后掉在原地 slow_walk 转向里，不是走动里） |
| `skills_can_coupled_direct10_20260928`（7.7.1.2 走 A：coupled 用自己的放置几何——**不持罐走近，原地转向放置**，plan `pick_place_direct.json`） | — | 4/10 | **3/4** | grasp：局部对齐扰动 ×4、stand infeasible ×1、抬起 −0.7 cm ×1；place：carry-over 掌系滑 3.1 cm ×1 |

- 管路通了：coupled 抓 → 持罐 approach(basket) → place 两局全程成功（罐头离篮心 1.6 cm）。
- 两个失败族与 simple 不同：(1) coupled 的 visible 版在停位上更挑（3/10 "stand infeasible"），走近用的是 simple 的守线刹车站位，不是 §4 当时的 creep 站位；
  (2) coupled 握持的罐头在走动中掌系滑 3.2–3.7 cm，正好压在守卫 3 cm 上（bottle 的 simple 握持 < 1 cm）——容差从此在 `approach(drift_max)` / `place(palm_drift_max)` 里可调，第二行是 0.05 的对照。
- 容差 0.05 没帮助：同 seed 下 grasp 从 5/10 掉到 2/10（`local alignment disturbed` 取代了 `stand infeasible`——coupled 的抓取对同一站位的逐局差异敏感），
  到达 place 的两局里 1 局成功；持罐转向时掉罐 3 局说明 coupled 握持在原地 slow_walk 转向下就会滑，放宽阈值只是把失败挪后。
- **7.7.1.1 到此**：coupled 已包进 skills 管路并能全程跑（2/10、1/10），但它带来的是另外两个失败族（停位挑剔 / 握持在转向中滑），不是罐头的解。
  两条线各自的失败证据都在 trace 的 `stats` 与 `[tag]` 里；下一步要么给 coupled 单独的走近站位（§4 当时的 creep 站位），要么回到闭合几何旋钮——待用户定。
- **7.7.1.2（走 A）**：§4 的 10/19 抓取率本来就是 53%，skills 里 coupled 抓取 4–5/10 与之一致；差在放置——§4 从不持罐走路。
  coupled 原地放置 3/4（持罐走近后 2/5、容差放宽后 1/2），所以 coupled 的默认 plan 不再插 `approach(goal)`（`default_plan` 按 profile 的 `reach.strategy` 选）。
  抓取侧的 50% 是 coupled 自己的失败族（局部对齐扰动罐头、stand infeasible），与 §4 相同，不属于本项。

## §13 Q11 失败注入配对比较：`move_target`（2026-09-29）

注入层见 [skills §7](../skills.md#7-失败注入与配对比较q11-验收)。plan `pick_place_via_basket.json`（profile `bottle_simple`），seed 300 起，GPU1 / instance 1。

**注入有效性**（plan 臂，3 局）：

| 注入 | 结果 | 说明 |
|---|---|---|
| `move_target` 横向 −0.17 m（`inj_move_plan3_20260929`） | 1/3，失败是 `[grasp] no bilateral pad contact`、`[carry] raise timed out` | **不是有效注入**：`grasp_single` 开始时按物体当前位姿重新规划（trace `plan.grip` 已是挪后的位置），0.17 m 在臂包络内，局 0 抓起并放进篮子 |
| `move_target` 沿桌沿 +0.60 m（罐头到 y = +0.32，`inj_move60_plan3_20260929`） | 见下 | 出臂包络；再走近仍从同一桌沿（`plan_for_target` 的边由 `init_pos` 定，站位按当前位置算） |

往桌子深处挪不可恢复（站位会落到桌面内），所以只沿桌沿挪。

**配对 10 局（seed 300–309，`move_target` +0.60 m，报告 `outputs/analysis/injection_20260929/report.md`）**：

| 臂 | 任务 OK | 注入后重新抓起 | 每局调用 | 失败去向 |
|---|---|---|---|---|
| plan `inj_move60_plan10_20260929` | **0/10** | 0/10（不会恢复） | 2 | 全在 `[grasp]`（entry 对不上 ×8、support reference ×2） |
| agent `inj_move60_agent10_20260929`（sonnet，`--max-steps 16`） | **0/10** | **5/10** | 6–16（中位 12） | 重抓后 5/5 在持罐 `approach(basket)` 转向时掉罐；3 局在重新站位的原地转向把罐头扫下桌；2 局用完 16 次调用 |

符号检验 p = 1.0（没有不一致对）。结论分两层：

- **决策层通了**：agent 每局都从 `[grasp]` 失败 + 特权状态里罐头的新位置推出"罐头不在原处"，走 `approach!（横向不修）→ navigate_to（到罐头旁）→ turn_to（转回朝桌）→ approach → grasp_single`，5/10 重新抓起；一次 `release` 清掉旧 `held`（round 0、9）也是对的。它还发现了两个 primitive 事实：
  `approach` 只修深度不修横向（0.6 m 横向偏移下 2.5 s 报 `[stand] arrival left the object out of reach`）；`nudge` 是量化脉冲，横向 0.1–0.3 m 修不出（与 7.8.1 一致）。
- **primitive 层挡住了任务**：(1) 重抓后的站位在罐头新位置（y = +0.32），篮子在 y = 0，持罐 `approach(basket)` 要先 slow_walk 转 ~90°，simple 握持在转向中掉罐 5/5（§12 已知的"握持转向滑"，现在成了恢复路径的必经点）；
  (2) `navigate_to` 把机器人停在罐头正旁、离桌沿 0.1–0.15 m，接着 `turn_to` 原地转向时手臂扫过桌面把罐头扫到地上（3/10）；(3) 每次 `nudge` 都清 `stance`，agent 再抓就被拒，来回消耗 2–4 次调用，两局用完预算。

也就是说 Q11 的"agent OK 率显著高于脚本"在这个注入上**没有出现**，但不是 agent 选错，而是恢复路径需要的两个动作（持物横移不转向、贴桌沿转向不扫桌面）目前没有 primitive 能做。要让配对比较有区分度，先补 primitive：持物侧移（`carry_to` 加 `strafe`，不转向）或 place 前的转向改为倒退-转-前进；`navigate_to` 目标点离桌沿留 ≥ 0.3 m；`turn_to` 在桌沿附近先收臂。

**配对 10 局 `drop_held`（seed 300–309，抓起后罐头回到 `init_pos`，报告 `outputs/analysis/injection_drop_20260929/report.md`）**：

| 臂 | 触发注入的局 | 任务 OK | 注入后重新抓起 | 注入后第一步 |
|---|---|---|---|---|
| plan `inj_drop_plan10_20260929` | 5/10（其余 5 局抓取本身失败，注入没触发） | 0/10 | — | 持罐 `approach(basket)` 5/5 报 `[approach] load lost while turning` |
| agent `inj_drop_agent10_20260929` | 5/10 | 0/10 | 2/5（seed 305、306） | **5/5 直接 `place`**——相信 session 的 `held` 而不是场景里罐头已在桌面（z 0.69、离手 0.23–0.32 m）；`place` 的持物守卫报 `[carry] object slipped 3.x cm` 后才重新走近 |

两臂都触发的只有 3 对（300、301、305），p = 1.0。重抓的两局之后 `place` 又报 `[carry] object slipped 3.3–3.4 cm`（与 move_target 一样卡在持罐转向 / 搬运）。

- 注入只在抓取成功后触发，而基线抓取 ~50%，所以 10 局只得 5 个注入、3 个配对；要 10 个配对得按"触发数"而不是局数跑（TODO 8.4.1）。
- agent 第一反应 5/5 是 `place`：请求里 `session.held` 说拿着，`scene.objects.soup_can` 说在桌上，它信了前者。系统提示没说"session 可能过期，以场景为准"；要不要加这句是 prompt 设计问题（agentic-plan Q4 的特权状态就是给它对账用的），加了就是给提示，不加就是在测它的对账能力——待用户定（TODO 8.4.2）。
- `drop_held` 的 `moved_m` 是 xy，看 `moved_z_m`（−0.23…−0.32 m）才是掉落量。

**第二轮（2026-09-29 晚，8.3.1 / 8.3.2 / 8.4.1 / 8.4.2 之后；`--until-injected 10`）**

| 注入 | plan 臂 | agent 臂 | 配对 | agent 重抓 | 报告 |
|---|---|---|---|---|---|
| `move_target` +0.60 m | `inj2_move_plan_20260929b` 0/11（10 触发） | `inj2_move_agent_20260929b` 0/10（10 触发） | 9 | 3/9 | `outputs/analysis/injection2_move_20260929/` |
| `drop_held` | `inj2_drop_plan_20260929b` 0/20（10 触发；注入后 8/10 `[approach] load lost while turning`、2/10 slipped） | 未跑（蠕变机制确认后停掉队列，省 agent 调用） | — | — | — |
| `move_target`（8.3.4 后 agent 重跑，跑 3 局即停） | 同上 plan 臂 | `inj3_move_agent_20260929c` 0/3 | 2 | 2/3 | `outputs/analysis/injection3_move_20260929/`；两次重抓都是 `place` 被 `placing_band` 拒 → 想 8 s → 罐头从冻结握持掉落 → `release` |

- **恢复链缩短了**：`approach` 自带横向预走（`prewalk` lateral −0.57～−0.63 m，残差 −0.09 m）后，agent 两次调用就重新站好并抓起（第一轮要 5–6 次）；提示规则让它不再先抓空一次（round 0 直接从场景看出罐头挪了）。
- **但重抓后 100% 掉罐，机制不是转向而是等待**：4/4 重抓后 agent 直接 `place`（basket 在骨盆系 (0.42, −0.43～−0.50)，带外），`place` 起步 1.2 s 内报 `[carry] object slipped 3.0–3.5 cm`；
  这 3 cm 是 agent 想下一步的 5–8 s 里滑掉的（无 primitive 在跑）——0.4–0.5 cm/s，正是 MuJoCo 抓握蠕变（memory `mujoco-grip-creep`）。第三轮 round 0 更直接：`place` 被 `placing_band` 拒绝、agent 想 8 s，罐头就从冻结握持里掉到桌上（z 0.95 → 0.65），全程没动过。
  固定 plan 抓完 < 1 s 就搬，所以 6/6 进带（§8）；agent 每步 6–9 s，握持撑不到下一步。第一轮"7/7 在持罐转向时掉"多半也是同一机制（转向前已滑 3 cm，LoadWatch 在转完第一次检查时报）。
- 由此 8.3.4 的两处（`STRAFE_YAW`、`placing_band`）方向对但不解决问题：在等待里滑掉的罐头，横移也救不回。
- **结论**：Q11 配对比较在"抓着东西跨过一次决策"的任务上无法区分 agent 与脚本，直到三选一（8.3.5，待用户定）：(a) sim 侧把指垫接触调硬（memory 说只有这条治蠕变；会改物理，采集数据一致性要重看）；(b) 藏掉决策延迟（Q5 的"提前规划"：primitive 执行中就问下一步，或用 haiku 把 6–9 s 压到 3–4 s，仍 > 1 s）；(c) executor 在两次调用之间跑"保持"步（更深的合爪设定点、或周期性重抬），效果未测。
- **更正（2026-09-29 晚，§14）**：上面的蠕变机制是错的。掉罐是 deploy 的 1 s 规划器超时：agent 想的时候 runner 什么都不发，1 s 后 deploy 释放冻结上身与手指命令，手臂甩回默认姿态，罐头落地（第二轮 round 0 下一次请求里罐头 z 0.03）。`place` 的 "slipped 3.0 cm" 以它自己的起点为参考，是在已掉的罐头旁动手臂越过阈值。真实蠕变约 0.1 cm/s。

## §14 持物跨过一次决策：deploy 规划器超时，不是指垫蠕变（TODO 8.3.5，2026-09-29 晚）

用户按 §13 的结论选 (a) 把指垫调硬。先量蠕变：`plan_episode.py --think-s 8` 让固定 plan 在每次成功调用后停 8 s
（agent 的决策延迟），每拍记罐头的掌系漂移、接触，存 qpos 流（`think_NN_K.npz`）。同 seed 400–409、`pick_place_via_basket.json`、
`bottle_simple` profile、GPU1 / instance 1。

| 运行 | 停顿 | 指垫 | 持物停顿 | 掉落 | 蠕变 cm/s 中位（范围） | 任务 OK | 抓起 | 失败族 |
|---|---|---|---|---|---|---|---|---|
| `think8_base_20260929`（6 局后停） | 静默（`--think-silent` 的行为） | `object` | 4 | **4** | — | 0/6 | 4/6 | `[grasp]` 2、`[approach] load slipped`（罐已在地上）4 |
| `think8_keep_20260929` | 保活（IDLE 帧 50 Hz，朝向锁停顿开始时） | `object` | 14 | 0 | 0.10（0.08–0.17） | 2/10 | 7/10 | `[grasp]` 3、`[place]` 2、`[retreat]` 1、`[carry]` 1、`placing_band` 拒 1 |
| `think8_padcrate_20260929` | 保活 | `crate`（`outputs/scene/pick_place_padcrate`） | 12 | 0 | 0.06（0.03–0.09） | 1/10 | 6/10 | `[grasp]` 4、`[place]` 4、`[carry]` 1 |

- **机制**：静默组 4/4 在停顿开始后 0.96–1.08 s 掉罐，之前掌系漂移 < 1 mm（一局 7 mm）。qpos 流里右臂在 0.1 s 内从持物姿态甩到
  （肩 roll −0.25、肩 yaw 0.58、肘 1.05），罐头飞出落地（dz −0.90～−0.95 m）。deploy `include/input_interface/zmq_manager.hpp`
  的 `PLANNER_TIMEOUT` 1000 ms：没有新的规划器消息就置 `has_upper_body_control_ = false`、`has_hand_joints_ = false`、IDLE；
  `deploy.log` 打 `[ZMQManager] Planner timeout (100x ms)`（静默组 395 条）。我们的 commander 只在调用方 `frame/hold` 时发帧，没有保活线程。
- **§13 的 agent 轮次是同一件事**：`inj2_move_agent_20260929b` round 0 抓完后第一次请求里罐头在手（z 0.937，touching right），
  agent 想 5.7 s，`place` 1.2 s 报 slipped 3.0 cm，下一次请求里罐头在地上（z 0.03，离手 0.56 m）。`place` 的 `_Watch` 以它自己的起点为参考。
  所以 §13 第一轮"持罐转向掉 5/5"、第二轮"place 起步滑 4/4"、第三轮"想 8 s 掉桌"都是超时，不是转向、不是蠕变；
  8.3.1 的持物横移和 8.3.4 的 `STRAFE_YAW` / `placing_band` 是按错的机制加的（保活组 round 8 就是被 `placing_band` 拒掉的固定 plan）。
- **(a) 的结果**：`crate` 档（solimp 0.995/0.9999）把蠕变从 0.10 降到 0.06 cm/s；两组都 0 掉落，任务与抓取率在 10 局噪声内（2/10 vs 1/10、7/10 vs 6/10），
  失败都在抓取和放置本身。按 0.1 cm/s，持罐跨三次决策（~25 s）漂 ~2.5 cm，每个 primitive 的守卫以自己的起点为参考，不会因此触发。
  **结论：指垫不是瓶颈，不改默认场景**；`pick_place_padcrate` 场景与 `configs/experiments/skills_pick_place_padcrate_20260929.yaml` 留作证据。
- **solref 在跑的 sim 里不起作用**：sim 200 Hz（`SIMULATE_DT` 0.005），MuJoCo refsafe 把时间常数抬到 ≥ 2·dt = 0.01 s，`object` 的 0.004 与
  `crate` 的 0.002 都是 0.01（玩具模型：0.01 / 0.004 / 0.002 蠕变相同；蠕变 ∝ 1 − solimp d0）。两档只差 solimp。
- **离线回放台**（`scripts/experiments/pad_creep_replay_20260929.py`，机器人按 qpos 流运动学驱动、手指按 deploy 的 PD、罐头自由）：
  基线参数 0.24–0.49 cm/s（现场 0.11，偏高 2–4 倍）；四组更硬的 solimp 在台上全部 1–2 s 掉罐，一组数值发散——而 `crate` 档在现场稳，
  所以台对"更硬"一侧不可信，只作排序用；结果在 `outputs/analysis/pad_creep_20260929/`。
- **未验证**：primitive 内部有没有 > 1 s 不发帧的空档（静默组 395 条超时远多于 4 次停顿；保活组 14 次持物停顿 0 掉落，说明至少持物段没有）；
  注入层 `pin_object` 的 0.5 s 加上前后开销是否跨过 1 s（注入时手臂在行走策略手里，影响未知）。

## §15 Q11 第四轮配对：runner 保活之后（TODO 8.3.5.3，2026-09-29 晚）

§14 之后 `Executor.idle` 在 `claude -p` 回答期间、注入 settle 期间按 50 Hz 重发当前命令（`agent_round` 把提问放进工作线程，
ZMQ 留在主线程）。两臂都在同一份代码上重跑（含 8.3.1 / 8.3.4 的 primitive），seed 300 起，`--until-injected 10`（上限 24 局），
sonnet、`--max-steps 16`，GPU1 / instance 1，顺序跑（队列 `outputs/runs/q11_round4_queue_20260929.sh`）。

| 注入 | plan 臂 | agent 臂 | 配对（两臂都触发） | 不一致对 agent : plan | p | agent 注入后重抓 |
|---|---|---|---|---|---|---|
| `move_target` +0.60 m | `inj4_move_plan_20260929d` 0/10（10 触发，全 `[grasp]`） | `inj4_move_agent_20260929d` **3/10**（10 触发） | 10 | 3 : 0 | 0.25 | 5/10 |
| `drop_held` | `inj4_drop_plan_20260929d` 0/19（10 触发，全 `[approach] load dropped`） | `inj4_drop_agent_20260929d` 1/24（9 触发，注入局 1/9） | 5 | 1 : 0 | 1.0 | 3/9 |

两种注入合计不一致对 4 : 0，双侧符号检验 p = 0.125。报告 `outputs/analysis/injection4_move_20260929/`、`injection4_drop_20260929/`；
预览 `outputs/videos/q11_round4_20260929/`（agent seed 301 恢复并放进篮子 367 KB；plan seed 301 抓空 123 KB）。

- **保活生效**：move agent 臂的 deploy 段 1 次规划器超时，drop agent 臂 3 次、都在最后一局（第 23 局，手里没东西，时间由 `Loop timing` 行数换算），
  前三轮 agent 臂是 110 / 84 / 22 次。所有重抓起来的罐头在每一次思考（6–16 s，最多连续 6 次提问、约 45 s）后都还在手里，只有一例例外：
  drop 第 14 局抓在 z 0.84（其他局 0.88–1.07），一次 7.4 s 思考里掉了，那段时间没有超时——浅抓的真实滑落。
- **agent 的恢复链**：move 第 1 局 `approach → [注入] → approach（自带横移预走）→ grasp_single → place?（placing_band）→ approach(basket) → place` 7 次调用完成；
  drop 注入后第一步 9/9 是 `release`（8.4.2 的提示规则，第一轮是 5/5 `place`）。
- **4 个 agent 成功里 2 个不干净**：move 第 5 局 `place` 报 "not over the container" 后 `release`、drop 第 2 局 `place` 报 slipped 3.3 cm，罐头都自己落进了篮子，判据算成功。
- **重抓之后丢罐的地方**：`place` 内部（slipped 3.3 cm、raise 差 18.8 cm）、持物横移（move 第 3 局 3 s 内 1 → 3.4 cm 的行走滑移）。不在思考里。
- **大多数 agent 失败根本没走到持物**：重新走近后的抓取失败（no bilateral pad contact，然后 "object toppled"）、`approach` 拒绝或失败
  （neutral waist not delivered、clearance、arrival out of reach）、`nudge` 6 次脉冲后仍偏 8–15 cm、调用用完。
- **结论**：保活之后配对比较测到的是决策——方向一致（4 : 0，agent 只赢不输），但每种注入 10 对以内、primitive 自身成功率（抓 ~50%、持物走近后放 ~50%）
  压住上限，统计不显著。要缩小 p，得加配对数或提高 `grasp_single` / `place` 的成功率，不是改提示。
- **仍开放**：drop agent 第 23 局的 3 次超时说明 agent 才用的 primitive（`navigate_to` / `walk_to` / `turn_to` / `nudge` 之一）内部有 > 1 s 不发帧的空档；
  这局手里没东西，没影响结果，但持物时会甩掉（TODO 8.3.5.5 改回开放）。

### §15a 规划器超时的来源：空手 `approach` 内部两段不发帧（TODO 8.3.5.5，2026-09-29 晚）

探针 `scripts/experiments/command_gap_probe_20260929.py`（= `plan_episode.py` + 每次发送打时间戳 + 静默 > 0.3 s 时每 0.1 s 抓一次主线程栈，
fixed plan 失败也继续跑）+ `configs/plans/command_gap_probe_20260929.json`（仿第 23 局：approach / nudge / walk_to / turn_to / navigate_to /
grasp_single / release 各几次），GPU0（空闲）/ instance 1，4 局，`outputs/runs/gap_probe_20260929b/skills/command_gaps.jsonl`。
第一次启动 `gap_probe_wrongports_20260929` 作废：探针在 `plan_episode.main` 钉端口之前 import 了 `agent_server`，commander 连到实例 0，每局起不来。

| 静默位置（栈） | 次数 | 长度 | 那时手里 |
|---|---|---|---|
| `approach` 刹车后：`_phase_approach` → `SimpleGraspPolicy.settle_for_admission`（`simple_policy.py:442`，只 `obs()` + `edge_guard.band_gap`，最长等 `stand_settle_timeout_s` 4 s）及其后的收尾 | 13 | 0.49–0.87 s（另 2 次 1.21 / 1.31 s 是局末最后一个 approach 连着局末记账） | 空；guard walk 下手臂归行走策略，手张开 |
| `approach` 起步前：`_prepare_grasp_policy` → `SimpleGraspPolicy.__init__`（arm IK、transport_guard、assisted_target）+ 首次 `edge_guard` 几何 | 12 | 0.45–0.73 s | 空 |
| `grasp_single` 内 `_lift_ready` → `bimanual_ik.solve`（POSE 流，规划器不计时） | 1 | 0.32 s | — |
| 局间（spawn、`end_video`） | 若干 | 0.3–3.3 s | 空（不在 primitive 内） |

- 这个 stack 的 deploy 记 2 次超时，对应上面 2 次 ≥ 1 s。primitive 内部在空闲 GPU 上最长 0.87 s，没过 1 s；`nudge` / `walk_to` / `turn_to` / `navigate_to` 每拍都发帧，
  `Navigator` 的调用是毫秒级（离线计时 set_goal / replan ≤ 2.5 ms）。
- 对上 agent 臂的 4 次超时：drop agent 第 23 局 3 次前面都是 SLOW_WALK 0.25 → IDLE，0.25 就是 guard walk 的 `walk_speed`（`simple_config.py:73`），即刹车后的 settle 段；
  move agent 1 次前面是 `walk_to` 到达的 IDLE，即下一个 `approach` 的起步段。第 23 局那几分钟 sim 出图从 ~30 Hz 掉到 ~24 Hz（GPU1 上别的项目启动了 UnrealEditor），
  0.5–0.9 s 的段被拉过 1 s——这是推断（没有那几分钟的栈），但位置由探针确定。
- **影响**：两段都只在空手 `approach` 里（持物的 `approach` 走 `_loaded` 路径，不经过这两段）；那时手臂归行走策略、手张开，deploy 超时后的 IDLE + 放开上身 / 手指控制没有可丢的东西。
  持物路径（持物 approach、place、carry、nudge）在 think8_keep、inj4 两个 plan 臂和两个 agent 臂里没有记到超时。
- **决定（用户，2026-09-30）：不修。** 不在这两段补发帧，也不在 commander 加后台保活（两者都改 bottle 采集线的共用代码）。

## §16 Q11 收尾（用户 2026-10-02 定，不再跑）

- **结论**：runner 保活之后，配对比较测到的是失败后的决策。两种注入方向一致：`move_target` plan 0/10 vs agent 3/10，
  `drop_held` 配对 0/5 vs 1/5；不一致对合计 4 : 0（全部只有 agent 过），双侧符号检验 p = 0.125，**不显著**。
- **为什么到此为止**：上限卡在 primitive 自身——`grasp_single` 抓罐头 ~50 %、持物走近后 `place` ~50 %；agent 的失败多数发生在
  拿到东西之前（重抓失败、`approach` 拒绝、`nudge` 量化）。加配对数或改提示词都推不动这个上限，要先提高这两个 primitive 的成功率。
- **以后若重开**：先把 `grasp_single`（罐头）和 `place` 各提到 ≥ 8/10，再每种注入跑 ≥ 20 对；推机器人 / 对负载施力需要 sim 侧施力命令
  （TODO 8.5，搁置）。三轮作废的教训（deploy 1 s 规划器超时）见 §14，保活已在 `Executor.idle`。
- 工作转向 scene_discovery（场景 → 自动发现任务 → 技能组装采集 → VLA），见 `docs/handoff/scene-discovery-plan-20261002.md`（分支 `scene_discovery`）。


## §17 `place_phase` 改动后放进容器的回归（scene_discovery TODO 9.12.10，2026-10-05）

`put_down_single`（`4da2bba`）改了 `place_phase`（台面几何分支、`_servo` 的 `until` / `level` / 日志），放进容器只有单元测试。按 §8 第三轮原样重跑：
`skills_pick_place_20260928.yaml` + `configs/plans/pick_place_via_basket.json`，seed 300–309，GPU1 / instance 1，运行 `sd_place_regress_20261005`。

| | 抓取 | place | 端到端 | place 失败 |
|---|---|---|---|---|
| §8 `skills_pickplace_viabasket3_20260928` | 6/10 | 3/6 | 3/10 | `descent timeout` 6.0 / 9.0 cm、`raise timed out` ×1 |
| `sd_place_regress_20261005` | 6/10 | **3/6** | 3/10 | `not over the container at the descent timeout` xy 6.5 / 7.4 / 9.0 cm |

- 结论：放进容器没被改坏；失败是 §8 的同一族（下降超时时罐头离篮心 6–9 cm），只是新日志措辞更具体。抓取失败 4 局：碰倒 ×2、合爪无双侧接触 ×1、抬起丢接触 ×1，也是 §8 的族。
- 10 局区分不了 ±1 局的差别（memory：10 局试跑测不出这条线）；这里只用来排除"改坏了"这种大退化。

## §18 合入 main 前后的技能回归，与 mars 上的 p4（scene_discovery TODO 19.18.4，2026-10-08）

同机、同日、同 seed，合并前的代码（tag `scene_discovery-pre-sync` = `a73cf0c`，worktree `../locomanipulation-scene_discovery`）
与合并后（`scene_discovery-sync` 的 `10b3895`/`0fcceaf`，worktree `../locomanipulation-scene_discovery-sync`）各跑一遍；
`plan_episode.py`、instance 1、GPU0、deploy 根 `outputs/bottle_continuous_20260914/deploy`。bottle 用 `bottle_simple_200.yaml`
（seed 100–109，§40 的那组）；pick_place 用 `skills_pick_place_20260928.yaml` + `configs/plans/pick_place_via_basket.json`
（seed 300–309，§17）；p4 是 `outputs/scene_discovery/adt_apartment/experiments/p4_canister_on_round_table.yaml`（罐子放到圆桌，
seed 700–709），mars 那行是同一文件只换场景 `adt_apartment_mars`。

| 线 | 合并后 | 合并前 | 之前的记录 |
|---|---|---|---|
| bottle（approach → grasp_single） | 7/10（grasp 2、approach 1） | 6/10（stand 2、grasp 2） | §40 9/10（09-28） |
| pick_place（抓 → 走近篮子 → place） | 2/10（grasp 6、place 2） | 4/10（grasp 3、turn 1、place 2） | §17 3/10（10-05） |
| p4 罐子 → 圆桌（stock） | 3/10（grasp 3、stand 2、place 2） | 3/10（stand 3、grasp 3、turn 1；`presync_regress_p4_20261008b`） | 第六版验收 80 局端到端 13（`put-down-single` 交接文档） |
| p4 罐子 → 圆桌（**mars**，合并后） | **1/10**（grasp 6，其中 5 局 `clearance: right_wrist_yaw_link` 碰到罐子；stand 2、approach 1） | — | — |

- 结论：合并前后没有看得出的差别（p4 都是 3/10）；10 局一组判不动小差别。同一 seed 两次跑结果也不同（pick_place 第 0 局：10-05 成、今天合并前败、合并后成），
  同 seed 三次的抓取成功 4–7/10。
- 单臂技能路径上合并前后的代码只差注释、一句报错和栈管理的记账；仿真电机力矩表改由场景 XML 读出，stock 场景逐项等于旧的写死列表。
- **作废**：04:06–07:02 容器内存打满（Unreal 编辑器 24 GB + 两个 ADT 栈各约 16 GB + 提交钩子的测试，上限 91 GB），
  这段时间的 `presync_regress_p4_20261008`、`sd_unreal_smoke_p4_20261008` 各有一局跑了约 10,500 s、两边 deploy 都死了。
  p4 合并前那行是 07:20 起单独重跑的 `presync_regress_p4_20261008b`。
- 背景负载：合并后 p4 与合并前 pick_place 跑时，Blender 烘焙（2 核）和 Unreal 导入房间（4 核）也在跑；各局用时正常（≤ 70 s）。
- **mars 上新出现的失败**：第 5–9 局抓取前的间隙检查判右腕 `right_wrist_yaw_link` 碰到罐子（stock 上 0 次）。mars 的腕上多了
  D405 相机支架且参与碰撞（make_dex1_model），抓取几何（`bottle_simple` 的接近方向与站位）是在 stock 上标定的。这是 `capability.yaml`
  需要在 mars 上重测的直接证据（TODO 19.18.5、19.18.8），不是合并带来的。
