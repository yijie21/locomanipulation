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

