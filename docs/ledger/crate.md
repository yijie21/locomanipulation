> **台账(append-only)**:本文件只在末尾追加新的日期化章节,不改写、不删除已有结论。
> 要更正一条结论,在新章节里写清楚并链接回旧的那一条,不要就地覆盖。
> 分类与格式规则见 `docs/reference/standards.md` §3。

# 台账:crate(双臂搬箱)线

## 2026-09-07 wall pilot 验收判据(来自 `docs/crate_collection.md` §5,表格原样保留在教程里)

`run_wall_pilot.sh` 10 集 pilot 的验收表,`WALL_FEATURE=rail`:

| 指标 | 合格 | 为什么 |
|---|---|---|
| 成功 | ≥ 8/10 | 历史:ready-stand pilot 16 20/20;走入 pilot 21 20/20,22 9/10,23 8/10 |
| `move_s` | = lift 开始时刻(约集长 − 4 s) | lift 前篮子不能被碰。pilot 18 的 20 集都在走路时被指尖扫转 7–14°,当时以为的"站位 yaw 门"失败其实是这个 |
| `trig` | 绝大多数 `contact` | `floor` 是兜底,`seated` 是毫米判据,都不该是主流 |
| `static` 中位 | ≤ 0.10 | 吸收态预算(单手线 cover400 的 creep 就是这么来的) |
| 站位 `so` | 0.13–0.32 内 | reach 预算 0.38 减去深度 [0.06, 0.25] |
| 站位 `\|yaw\|` 中位 | ≤ 3° | 篮子没被碰时步态自身的散布 |

## 2026-09-07 设计文档里嵌入的实测结果(来自 `docs/archive/bimanual_grasp_design.md`,原文未加日期,按文中"今天"与相邻 pilot 编号推断为 09-07 前后)

这些是设计文档正文里夹带的实测数字,不是设计目标;裁定摘录于此,详细上下文见归档原文。

- **R1(单手线)91/95 局失败是相位错位**(脚本走路时抬手、数据里手已在桌边抬好);改成 coupled(蹲 + 伸同时)后 **8/8**,去停顿使闭环翻倍(`docs/archive/bimanual_grasp_design.md` §B)。
- **heading 死区实测**:recoverable 策略 89 轮里仍有 3 轮在 spawn 处因朝向死区(7–8°)超时("could not turn"),`diag_turn.py` 量出 15–120° 各档都停在 6.5–8.1° 残差(`docs/archive/bimanual_grasp_design.md` §A)。
- **夹爪硬化前的蠕滑**:0.30 kg 瓶子默认接触下 0.5 cm/s 蠕滑掉落;codex 07 独立集 24 集里 **3 集在 hold 阶段滑出约 1 m**,是 pad 硬化(`PAD_HARDEN`)成为前置条件的直接证据(`docs/archive/bimanual_grasp_design.md` §D)。

## 2026-09-10 随机化 10 集 pilot + MuJoCo|Isaac 并排视频(branch session,GPU0 / instance 2)

用户要求:再录 10 集 SONIC 双臂搬箱,场景在合理范围内随机化,成功失败都要,第三视角 + ego/左腕/右腕画中画,左 MuJoCo 右 Isaac。
配置:`configs/tasks/crate_wall.json` 默认(`rail`,`WALL_WALK=1`),seed 910;随机轴 = 现有三条(箱子横向 ±8 cm、出生点
1.0–1.3 m / ±0.15 m / ±10°、站位由步态散布),箱子 yaw / 桌高 / 外观仍未打开(教程 §8)。GPU1 被外部任务占 96%,
所以 pilot 和渲染都在 GPU0(教程默认 GPU1)。

| 指标 | 结果 | 合格线 |
|---|---|---|
| 成功 | **8/10** | ≥ 8/10 |
| 失败 | r2 `[hold] lost contact: ['right']`;r5 `[lift] rose 0.0 cm`(站位横向 +5.9 cm、箱子 dy −6.7 cm,双手闭在箱沿旁边) | — |
| `trig` | 10/10 `contact` | contact 为主 |
| `so` | 0.18–0.26 | 0.13–0.32 |
| `\|yaw\|` 中位 | 2.3° | ≤ 3° |
| `static` 中位 | **0.15**(max 0.30) | ≤ 0.10 —— **不合格**,吸收态预算超了 |
| 每集 | 25–47 s(r0–3 约 26 s,r4–9 约 44 s;`deploy_restarts` 全 0) | 40–60 s |
| move_s | 5.6–8.0 s,均在 lift 段(len 10–12 s) | = lift 起点 |

`static` 超线是本次唯一的红项,r1/r2 达 0.26/0.30;没查原因(本次是采集性能测试,不是调参)。
产物:pilot `outputs/videos/crate_random10_20260910/`(`wall_s910_NN.mp4`、`trace/`、`manifest.jsonl`、`pilot_s910.log`);
并排视频 `outputs/renders_isaac/crate_random10_20260910/sbs_roundNN.mp4`(10 个,1280×480,12.5 fps,10–12 s,共 70 MB;
左 MuJoCo 跟踪相机 + 三路画中画,右 Isaac 同一相机轨迹)。渲染约 3 min/集。
`summarize_wall_pilot.py` 之前把 trace 名写死为 `wall_s0`,非 0 seed 的 move_s/static 全是 nan,已改为按 manifest 的 video 名取 label(a0cd7d4)。

## 2026-09-10 更正:上一节的 10 集是 ready-stand 变体,走入版重跑 4/10(主机 load 130–176)

上一节(同日,seed 910)按教程 §4 的 `WALL_WALK=1 WALL_FEATURE=rail` 起的 pilot **没有走路段**:重构 step 3 之后这两个
环境变量不再被读,`wall_walk` 默认 false,10 集全是出生在站位后 0.45 m 瞬移的 ready-stand 变体(每集 25–47 s、视频
10–12 s、日志无 `[walk]` 行)。那 10 个并排视频作为"抓取段"仍有效,但不是完整 episode;教程、runner、静默失败表已改
(b07caef)。同 seed 用 `--config configs/tasks/crate_wall.json` 重跑:

| 指标 | 走入版(本节) | ready-stand(上一节) | 合格线 |
|---|---|---|---|
| 成功 | **4/10** | 8/10 | ≥ 8/10 |
| 失败 | r1 `[stand]` 横向 −9.6 cm(3 次走位后);r2 停太近 0.11 m;r3 `[descend]` never seated;r5 停太远 0.38 m;r6 站位处摔倒;r7 停太远 0.45 m | r2 hold 失接触;r5 lift 0 cm | — |
| 站位 `so`(成功集) | 0.14–0.23 | 0.18–0.26 | 0.13–0.32 |
| `static` 中位 | 0.01 | 0.15 | ≤ 0.10 |
| 每集 | 29–87 s;视频 9–33 s | 25–47 s | 40–60 s |
| `deploy_restarts` | 0 | 0 | — |

**裁定:4/10 不能当作 expert 的成功率**。5/6 失败在走入停站,走入 pilot 历史是 20/20、9/10、8/10(09-07);本次主机
load average 131 → 176(32 核,`/workspace/code/video_grasp` 的 teacher-benchmark,5 个 python 进程,其中 3 个在 pilot 中途
启动),GPU1 另有 97% 的外部任务。走路阶段对 20 ms 周期最敏感,ready-stand 变体绕过了它所以看不出。要得到可信的走入
成功率,机器安静时重跑同 seed;若仍 ≤ 6/10 再查 `STAND_OFF_WALK` / 走位逻辑。`static` 0.01 说明上一节的 0.15 来自
ready-stand 的抬臂 + 瞬移前的等待,不是走入流程的问题。
产物:pilot `outputs/videos/crate_walk10_20260910/`;并排视频 `outputs/renders_isaac/crate_walk10_20260910/sbs_roundNN.mp4`
(10 个,含走路段,9–33 s,共 106 MB;左 MuJoCo + 三路画中画,右 Isaac)。

## 2026-09-10 难度场景 10 集诊断:箱子内缩 {1, 3} cm × yaw {−15, −7.5, 0, +7.5, +15}°(branch session)

目的是看清双臂采集的难题在哪个阶段,不设合格线。代码 166cd37:`CrateTaskConfig.crate_setback_levels_m` /
`crate_yaw_levels_deg`(默认空 = 行为不变),reset 时按 5×2 设计固定打乱顺序取值;走入站位改为沿箱子实测法线
(之前假定箱子正对 −x)。其余三轴(横向 ±8 cm、出生点)照旧随机。seed 921,GPU0 / instance 2,CPU 余量 12 核,
`SIM_VIS=none`。1 集预检(1 cm,+15°)站位 0.227 m、法线 yaw 残差 −2.7°,几何正确。

| rd | setback | yaw | so 实测 | lat 实测 | 结果 | 远端手误差 [沿杆, 横向, 高] mm |
|---|---|---|---|---|---|---|
| 0 | 1 cm | +15 | 0.21 | −0.039 | OK | — |
| 1 | 3 cm | +15 | 0.26 | −0.059 | descend never seated | L [−16, 19, 24] |
| 2 | 3 cm | 0 | 0.30 | −0.012 | descend never seated(2 次走位) | L [−9, 19, 32] R [−4, −14, 20] |
| 3 | 1 cm | −15 | 0.21 | −0.014 | OK | — |
| 4 | 1 cm | −7.5 | 0.16 | −0.069 | OK | — |
| 5 | 1 cm | +7.5 | 0.26 | −0.060 | descend never seated | L [−22, 18, 30] |
| 6 | 1 cm | 0 | 0.21 | −0.033 | OK | — |
| 7 | 3 cm | +7.5 | 0.26 | −0.046 | descend never seated | L [−22, 38, 38] |
| 8 | 3 cm | −15 | 0.15 | +0.008 | OK(rise 0.155,tilt 6.8°) | — |
| 9 | 3 cm | −7.5 | 0.27 | −0.066 | descend never seated | R [−4, −24, 21] |

成功 5/10;闭合触发 5/5 `contact`;成功集 static 0.05–0.08、失败集 0.17–0.34(下探时手在空中反复找)。

**裁定**:难题不在走入、也不在 setback 或 yaw 本身,而在**站位散布 × 箱子转角的组合**。
- 分界是实测站位:`so` ≥ 0.26 m 的 5 集全败,≤ 0.21 m 的 5 集全成;走入目标 `STAND_OFF_WALK` 0.24,步态散布让它落在
  0.15–0.30。
- 失败集里差得远的那只手总是箱子转离机器人的那一端(+yaw → 左手,−yaw → 右手);半长 0.20 m 的箱子转 15° 让那端把手
  多出约 5 cm,reach 预算 0.38 减去 0.26–0.30 的站位只剩 0.08–0.12,远端手的 IK 先牺牲高度,判据 `never seated`
  就是这么触发的。yaw 为 0 但 so 0.30 的 r2 两手都差,是同一件事的纯深度版。
- setback 1 vs 3 cm 没有独立效应(1 cm 也败、3 cm 也成),它只是把远站位变得更常见。
- 这是 09-07 那批 20/20 看不到的:当时箱子正对,两只手对称,所以 so 到 0.32 都能过。

**下一步(未做)**:把站位目标向远端偏移半个转角位移(约 0.10·sin θ)并压低 `STAND_OFF_WALK` 或在 `[stand]` 门里按远端手
的 reach 判定(so + 0.20·|sin θ| ≤ 0.38 − DEPTH_MIN),让走位重来去修,而不是到下探才发现。
产物:`outputs/videos/crate_hard10_20260910/`(视频、trace、manifest 含 `crate_setback` / `crate_yaw_deg`);并排视频
`outputs/renders_isaac/crate_hard10_20260910/sbs_roundNN.mp4`。

**同日重跑(带 ghost hands,seed 921 同设计)**:5/10,分界重复——站位 ≥ 0.27 m 的三集全败(r2 0.31、r5 0.27、r7 0.30),
0.24–0.25 的两集成功;新增一个近站位反例 r8(3 cm,−15°,so 0.15):两手沿杆方向差 +34 / +66 mm,转 15° 后杆是斜的,
近站位把垫落到了杆的端头之外。r1 这次在站位阶段失败(横向 −13 cm),说明同 seed 的步态散布本身不可复现,单集
结果不能跨轮比较,分界要看 so。视频:`outputs/renders_isaac/crate_hard10g_20260910/sbs_roundNN.mp4`(左右都画了
ghost hands,洋红 = IK 想去的手腕位姿;r1 没有 ghost,它没进抓取阶段)。原始:`outputs/videos/crate_hard10g_20260910/`。

**第三轮(ghost = 纯目标 `ref`,3e18990;seed 921 同设计)**:7/10,失败 r3(1 cm,−15°,so 0.29)、r5(1 cm,+7.5°,so 0.26)、
r7(3 cm,+7.5°,so 0.29),全是 `never seated`,全在 so ≥ 0.26;成功集 so 0.16–0.24。三轮同 seed 分别 5、5、7/10:10 集的
成功数本身有 ±2 的噪声,分界(站位 ≥ 0.26 m 必败,≤ 0.25 m 成功)三轮一致。视频:`outputs/renders_isaac/crate_hard10r_20260910/sbs_roundNN.mp4`
——**这一批是最终版**(ghost 画的是纯目标点,hover → 杆上 → 下探,不再随伺服 overdrive 抖动);前两批留作对照。
