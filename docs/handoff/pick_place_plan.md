# pick_place 任务设计:走到桌前,右手抓罐头,放进左边的篮子

状态:已实现并通过 10 局验收(2026-09-10,台账 `docs/ledger/pick_place.md`);实现与设计的差异见 §1 的实施注记与台账 §2–§3(放置带 [0.26, 0.44]、hover +3 cm、释放在篮内 10 cm、指面硬化、JOINT_STEP 0.01)。场景 `SCENE_PRESET=pick_place` 已建
(`src/locomani/sim/scene_lib.py`,资产 `assets/simple/{basket,soup_can}`,渲染与物理检查在
`outputs/scene_checks/pick_place_20260910/`)。本文只回答"运动逻辑怎么设计",实现另开 PR。

## 0. 设计原则(与 bottle 线一致)

专家用 privileged 位姿,但录下来的行为必须像是看着画面做的。具体约束,每条都有出处:

1. **决策时刻目标在画面里。** 单帧 VLA 没有记忆:走路时罐头和篮子都要在 ego 帧里;放置时篮子
   在帧里。门限沿用 bottle:`|u_x| <= 0.80, |u_y| <= 0.90`(`round_phases.py:423`),伺服期间
   可见率 >= 70%。memory `vla-target-must-be-in-frame` 定的篮子-目标间距 0.22–0.40 m,本场景
   0.28 m。
2. **没有隐藏的停止点和无差别帧。** 同一画面不能对应相反标签(memory `creep-hidden-stop-target`)。
   放置不悬停等待:到位即松手,或下降超时即松手(agentic 分支 §11 的 4 s 规则)。
3. **速度剖面连续。** 策略学的是专家的减速曲线而不是停点(memory `policy-stop-point-is-set-by-the-crawl`),
   所以每个阶段的移动都是有目标的匀速/减速段,没有 IDLE 发呆。
4. **闭环纠偏要被示范。** bottle 专家从不在桌前修正朝向,策略于是在 yaw 上失败(memory
   `stand-yaw-is-the-failure-variable`)。放置段天然要求转身 ~30° 对准篮子,把这段做成边看边转的闭环,
   是本任务对数据的最大增量。
5. **持物时间短。** dex1 软接触下罐头以 0.16–0.5 cm/s 滑出(memory `mujoco-grip-creep`,agentic §8/§12),
   从闭合到松手控制在 8 s 内;超过要先加指面硬化(agentic §8 的 pad 参数,主线 `build_sim_scene.py`
   目前没有 PAD_HARDEN,需要重新加)。

## 1. 几何(离线核过,机器人躯干直立)

**2026-09-10 实施时改动:罐头改为 6 × 14 cm 的细罐(330 ml slim can),grip 在罐心上方 3 cm。** ready-grasp baseline 的 FK 模板只支持
grip_z >= 0.70(舒适下蹲 0.03–0.13 m):9 cm 罐头 grip_z 0.647 第一局就抛 "target height has no comfortable
baseline grasp";把桌子抬到 0.70 又让左手 ready 姿态碰桌(guard "left finger vs table")并让蹲深不够
("stand lacks a clear squat and short grasp path")。所以桌面保持 0.62,把 soup can 的 mesh 在 z 上拉长到 14 cm
(`assets/simple/soup_can/model.xml` 的 mesh scale 0.848/0.848/1.321),`reach.grip_below_center` 设 -0.03(校验范围放宽到
[-0.05, 0.03]),grip_z 0.722 与 water bottle 的 0.723 一致(4 个模板 + 9 个 lower samples)。先试过 20 cm 罐 + grip 在罐心:
coupled squat 的 safe-hand sweep 报 palm(wrist_yaw_link)撞罐——俯仰 23° 的手掌压在 grip 上方 10 cm 的罐身上,
bottle 线之所以没事是因为 grip 上方只有 5.7 cm 的细瓶颈;所以 grip 上方的罐身要 <= 4-5 cm。

桌面 z 0.62,近边 x = -0.50,机器人从 -x 走来。篮子中心 (-0.34, 0),开口 0.20(x)× 0.30(y),
沿口高 z 0.77。罐头 (-0.40, -0.28),直径 5.9 cm 高 9.0 cm,在机器人右手边,篮子在它左边 0.28 m。

ego 相机归一化坐标(head_camera pitch 52°, fovy 70,4:3):

| 时刻 | 骨盆位姿 | 罐头 (u_x, u_y) | 篮子 (u_x, u_y) |
|---|---|---|---|
| 出生 1.2 m 外,正对桌子 | (-1.7, 0, 0°) | (0.24, 0.67) | (0.00, 0.74) |
| 出生,横向偏 -0.3 | (-1.7, -0.3, 0°) | (-0.02, 0.67) | (-0.25, 0.74) |
| 抓取站位,离边 0.32,罐头方位 -28° | (-0.82, -0.28, +28°) | (0.31, -0.29) | (-0.03, 0.05) |
| 放置站位(见 §2 第 5 步) | (-0.80, -0.28, +30°) | 手里 | 约 (0, 0) |

结论:出生到放置的每个时刻,两个物体都在 0.80/0.90 门限内,且**篮子在抓取站位就已经在画面中央**,
放置不需要重新走路,只需在原地转身 + 手臂过顶。但篮子中心在抓取站位的骨盆系是 (0.555, 0.02),
超过 0.50 m 的掌心包络(memory `sonic-reach-envelope`),所以必须转身 + 小步把篮子带进放置带。

## 2. 阶段(前 4 步原样复用 bottle 专家,第 5 步起是新增)

| # | 阶段 | 进入 | 退出 | 指令 | 出处 / 备注 |
|---|---|---|---|---|---|
| 1 | spawn + settle | 局开始 | 可见性门通过 | 出生 1.1–1.3 m,±20° yaw,横向 ±0.20 | `round_phases.py:135-457`,新增:篮子也过 0.80/0.90 门 |
| 2 | approach | settle 后 | 进 grip 带 [0.22, 0.36] 后蹲 | planner 走路 0.30/0.215/0.20 带 + brake + creep-to-band + crouch | `round_phases_approach.py:21-285`,站位只按罐头算(`plan.py:82-116`),篮子在左 0.28 m 不碍事 |
| 3 | reach / close / lift | 站位后 | lift >= 0.06 m 且可见率 >= 70% | 50 Hz reach servo,V_TRAVEL 0.15 / V_DESCEND 0.10 | `round_phases_approach.py:287-351`,不改 |
| 4 | rise | lift 通过 | 骨盆回到站高 | planner 离开 IDEL_SQUAT,同时 `stand_up + carry_pose` | 现有 `:353-360`。**注意**蹲起会让骨盆后移 4–16 cm(memory `grasp-servo-tracking-lag-and-crouch-shift`),第 5 步必须重新量篮子的骨盆系坐标,不能沿用抓取前的 |
| 5 | turn-to-basket | rise 完成 | 篮子进放置带:骨盆系 x ∈ [0.25, 0.35],y ∈ [-0.15, -0.05](右手正前偏右) | planner 原地转 + shuffle(与 creep-to-band 同一机制),朝向锁向篮子中心,速度 0.15–0.20 | 新增。转角约 30° 左转 + 前进/侧移 5–15 cm。整段篮子在画面中央,是"闭环纠偏"的示范段。上限 8 s(agentic §12 站位伺服限时) |
| 6 | carry-over | 进放置带 | 掌心到篮子中心正上方,xy 误差 < 3 cm | 掌心走一条抬高弧线:先抬到 z_world ≈ 0.86(罐底离沿口 +5 cm),再水平移到篮心;50 Hz 伺服在实测掌心上积分,lead 补偿 0.2 s × v;V 0.15 | 新增。不做 planner 下蹲(agentic §12:站高下 drop 点在可达域内,后蹲的 9 cm 冲击不值)。倾斜 > 25° 回抬(agentic §13) |
| 7 | descend + release | 到篮心上方 | 松手 | 以 0.10 m/s 下降到罐底离沿口 +3 cm;满足 xy < 3 cm 立即张开,或下降超 4 s 张开 | 新增。**不悬停**(原则 2)。罐头自由落体 3–8 cm,离线 15 cm 落体测试已验证落在篮内 |
| 8 | retreat | 松手 | 掌心离篮心 xy > 0.25 m | 先竖直上抬 8 cm(0.10 m/s),再后退 0.35 m(现有 back-off 走路),**收臂到 carry_pose 只在后退之后** | agentic §11:关节空间收臂曾从上方扫倒瓶子;bottle 现有 back-off `:361-370` 复用 |
| 9 | verify | 后退完成 | 2 s | 终端 hold 2 s | 判据见 §3 |

第 4–8 步的持物时间预算:rise 2 s + turn 3–5 s + carry 2 s + descend 1–4 s ≈ 8–13 s,已经贴着
原则 5 的上限。第一轮 pilot 先量 `carry_trace` 的掌-罐偏移,超过 2 cm 就上指面硬化,不要靠调速度硬撑。

## 3. 成功判据与失败分类

成功(仿 SIMPLE `check_object_in_container` + bottle 的 hold 协议):罐头几何中心在篮子内矩形
(篮心 ±0.07 x,±0.12 y,留 3 cm 壁厚)且 z 高于篮底 0.63、低于沿口 0.77,持续 2 s,期间两只手与
罐头无接触;**不要求竖直**(SIMPLE 也不要求;倒在篮子里算成功)。篮子位移 > 5 cm 或倾斜 > 15° 判失败
("撞篮")。

失败叶子在 `failure_taxonomy.py` 的 16 条之外新增:

| 叶子 | 阶段 | 判定 |
|---|---|---|
| F1 turn timeout | 5 | 8 s 内篮子未进放置带 |
| F2 basket out of view | 5–7 | 篮子 `|u_x|` > 0.8 的帧超过 30% |
| F3 slipped while carrying | 4–7 | 松手前掌-罐偏移 > 2.5 cm 或罐头落地 |
| F4 missed | 7 | 松手后罐头静止在篮外 |
| F5 on the rim | 7 | 罐头压在沿口上(z 在 0.74–0.80 且 xy 在壁带) |
| F6 knocked basket | 6–8 | 篮子位移 > 5 cm 或倾斜 > 15° |
| F7 retreat sweep | 8 | 松手后罐头 xy 又动了 > 5 cm |

## 4. 随机化(在 SIMPLE 的布局上收紧到右手可达)

- 篮子:中心 x ∈ [-0.36, -0.30],y ∈ [-0.05, 0.05],yaw 90° ± 15°。
- 罐头:始终在篮子**右侧**(一只手做决策,不学左右手选择):y ∈ [-0.35, -0.20],x 离近边 0.06–0.12 m
  (与 bottle 的 4–8 cm inboard 一致),yaw 随机。篮-罐间距落在 0.22–0.40 m 带内,否则重采。
- 出生:沿用 `spawn.edge_range (1.1, 1.3)`、`yaw_err 20°`、`lateral 0.20`。
- 出生门:罐头和篮子都过 0.80/0.90;不过的局丢弃并计 A5。
- 指令:`"pick up the red can and put it in the basket"`。
- 后续可选:篮子换到左侧更远(0.40 m)测泛化;罐头换 graspnet 其它 < 6.5 cm 的物体做多目标。

## 5. 与现有代码的接缝

- `tasks/pick_place/` 作为第四个 Task(`tasks/base.py` 契约),`round_loop` 复用 bottle 的 1–3 步,
  第 4 步之后替换 bottle 的 back-off + hold 段为 5–9。
- 放置伺服复用 `stream_reach` 的 50 Hz reach servo(它已经有 lead 与倾斜遥测),目标点换成篮心。
- 篮子的 privileged 位姿由 `privileged_pub` 发布(与 `objects[key]["cam"]` 一样带归一化相机坐标),
  `goal_object` 已经在 `scene_meta.json` 里。
- 配置:`PickPlaceTaskConfig` 继承 bottle 的 spawn/reach 字段,新增 `place_band`(x/y 带)、
  `release_clearance` 0.03、`descend_timeout` 4 s、`turn_timeout` 8 s、`hover_clearance` 0.05。

## 6. 验收(照 bottle 的做法)

1. 10 局 pilot(memory `pilots-are-10-episodes`),GPU 空闲时跑;记 `carry_trace.jsonl`
   (掌-罐偏移 10 Hz)和 `place_trace.jsonl`(掌心 xy 误差、倾斜)。
2. 通过线:抓起的局里 carry+place >= 6/10,放置误差中位数 < 4 cm,F3 <= 1。
3. 每个失败都要落进 §3 的叶子;归不进的先扩分类再调参数。
4. 视频:每局 ego 帧 + 第三视角 side-by-side,篮子和罐头在 ego 帧里的轨迹画出来(可见性证据)。

## 7. 未决问题

- 篮子 yaw:现在长轴沿 y(横向容差 ±12 cm,深度 ±7 cm)。放置误差主要来自深度方向的 reach 残差
  还是 yaw 带来的横向偏差,pilot 前不知道;若深度误差大,转 90° 把长轴对准前方。
- 蹲起后骨盆后移 4–16 cm 的方差会直接进入第 5 步的转身量;若 turn timeout 频发,考虑在 lift 后
  不完全站起(planner height 只回一半)。
- 罐头 6 cm 直径的抓取本身没有先例数据(bottle 是 4.3 cm),第一轮先看 B/C 类失败率,
  再看放置。
