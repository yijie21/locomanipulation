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
