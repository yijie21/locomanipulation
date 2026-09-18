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

## 2026-09-11 双臂线的入口重构:driver 与 expert 分家,四处不对称一并修掉

**起因**(用户):双臂和 pick-and-place 的脚本调用逻辑和单臂线不一样,先把 `start_stack.sh` 和 `bimanual_wall.py`
分开,再查还有哪些设计不对称。

**改前**:`scripts/run_wall_pilot.sh`(83 行)和 `run_bimanual_pilot.sh`(72 行)是同一份文本的两份拷贝
(155 行里 33 行不同,10 行是注释),每份同时做四件事:钉 instance/GPU、起栈、跑 expert、用自己手写的 `pgrep -f`
收尾。而单臂线早就分开了:`run_learn_pilot.py` 管栈,`grasp_gym.py` 假设栈已起。两份拷贝还**已经漂移**——
`WALL_WALK`/`WALL_FEATURE` 的拒绝守卫只有 wall 那份有。

**改后**:`src/locomani/pipeline/run_crate_pilot.py`(driver)+ `scripts/run_crate_pilot.sh`(唯一的壳)+
两个 3 行同名 shim。同时修掉的:

1. `pipeline/stack.py` 的 `start_stack()`/`stop_instance()` **原本是死代码**(自称 decision 8 的唯一 seam,却只有
   `export_env` 有调用者);现在 crate 线真的走它,顺带补上 `MUJOCO_EGL_DEVICE_ID`(`CUDA_VISIBLE_DEVICES` 不会
   重映射 EGL 设备号,只设后者等于渲染在 GPU0、算在 GPU1)。
2. `SonicBackendConfig.no_encoder` 默认是 **True**(eval 的协议),而 `start_stack.sh` 默认 0。crate 走 planner 通道,
   `NO_ENCODER=1` 会静默丢掉手臂和下蹲参考。driver 显式写 False,并有测试盯着。
3. **manifest 是 append 的**,而两个 runner 都指向同一个 `outputs/videos/wall_pilot/manifest.jsonl`,
   `gate.py record crate` 读的正是它 → 两次 pilot 混成一个总体。现在一个 run 一个目录,重复跑要 `--resume`。
4. expert 用 `os.environ.setdefault` 取端口,**继承来的 `TF_*` 会压过 `locomani.sim.ports`**;driver 起 expert 前
   把整块端口从 env 里删掉,只留 `INSTANCE`。
5. ghost trace 过去取决于一个 runner 自己不设的环境变量(所以默认是**关**的,`render_wall_pilot.sh` 注释写反了);
   现在 driver 默认写 `<out>/trace`。

**同批修掉的另外四条不对称**(用户点名):

| | 改动 |
|---|---|
| C pad 硬化 | 两套实现两套常数(`build_sim_scene` opt-in solref 0.004 / `bimanual_scene` 无条件 solref 0.002,后者再改一遍写出的 XML)→ 一份实现 + 命名标定 `PAD_LEVELS{object,crate}`。**证明物理中性**:新旧 XML 只差两个 `contype=0 conaffinity=0` 的 ghost geom(旧代码按 `dex1_col*` 前缀匹配,把装饰也硬化了),编译后 **47 个可碰撞 geom 的 priority/condim/solref/solimp 零差异**,baseline 不用重录 |
| D 配置 | `CrateTaskConfig` 补 `prompt`(`stack.py` 读的字段,之前 crate 栈退回 "walk forward")和 `deploy_self_heal_rounds` |
| E deploy 自愈 | 三处各写各的 → `backends/sonic/deploy.py` 的 `restart()` + `SelfHeal`。原来:grasp_gym 有超时和成功判定;`bimanual_wall` 内联、**没有成功判定**(只打印退出码);`bimanual_carry` **完全没有**,一次 LowState trip 就让剩下每一集都是 "did not stabilise" |
| F exporter | `--record` 接好了,协议与单手线逐字相同(`c` 沉降后开录 / `c` 存 / `x` 弃),manifest 带 `produced`;`--record` 强制 instance 0 并设 `--batch-rounds 10`,expert 侧另有 `ports.require_exporter()` 拒绝空录。双臂线第一次**能**出 LeRobot 数据集(§11 从"还差的一步"改写) |

两个 `main()` 的重复(端口块、场景检查、priming、计时、manifest、打印)收进 `tasks/crate/round_runner.py`。

**验收**:`gate.py compare crate` 两次,`SONIC_POLICY=policy/release` 固定(baseline 是 2026-09-09 录的,
今天默认已翻到 v1.1,不钉就会把 checkpoint 的差异算到重构头上)。

**门闸结果,以及一个 gate 自身的缺陷**(三轮,全部 `SONIC_POLICY=policy/release`、seed 0、
`configs/tasks/crate_wall.json`、instance 1 / GPU1):

| run | 代码 | ok | `stand_off_m` | verdict | 失败项 |
|---|---|---|---|---|---|
| `crate_runner_refactor_20260911_release` | 只有 runner 重构 | 10/10 | 0.2225 | **PASS 8/8** | — |
| `crate_tasks_defg_20260911` | + C/D/E/F | 8/10 | 0.1889 | FAIL 6/8 | `stand_off_m`、`failure_reasons` |
| `crate_variance_run3_20260911` | + C/D/E/F(**与上一行完全相同的代码**) | 8/10 | 0.2065 | FAIL 7/8 | `failure_reasons` |

baseline(2026-09-09)ok 8/10、`stand_off_m` 0.2149、容差 ±0.010。

**结论:两处失败都是 gate 的分辨率问题,不是回归。** 证据:

1. **同一份代码跑两轮,`stand_off_m` 差 0.0176 m = 容差的 1.8 倍**(run2 0.1889 vs run3 0.2065)。
   `stand_off_m` 是走入之后**实测**的站位距离,却用着 `TOL["depth_m"]=0.01` —— 那个容差是为 `depth_m` 定的,
   而 `depth` 是 `0.38 − stand-off` 再夹到 [0.06, 0.11] 的量,本来就稳定。逐集看,同 seed 的实测站位在两轮之间
   最大差 **11 cm**(run1 vs run2),与本文件 2026-09-10 那条"同 seed 的步态散布本身不可复现"一致。
2. **`failure_reasons` 在 run2 和 run3 报的是不同的新字符串**(run2 只有 `[descend] never seated`,
   run3 多一个 `[stand] lateral ... off the centre line`)——这正是 `docs/STATE.md` 自己列的未决项
   "gate baselines are 10 rounds (crate): rare failure strings show as 'new'"。
3. **`yield` 三轮 0.8 / 1.0 / 0.8,对 baseline 的 0.8 全部 PASS**;`depth_m`、`exit_majority`、
   `stage_s.*`、`yaw_err_deg` 三轮全 PASS。
4. run1 与 run2 用的场景 XML **md5 相同**,逐集**计划**站位(`stand_off`)完全一致 —— 分叉发生在计划之后,
   即走入与下探的时序,不在被改动的代码里。

**没有**为了让改动通过而放宽容差。要真正修这个 gate,两条路(留给用户裁):
(a) 机器空的时候按 STATE 的既有计划把 crate baseline 重录到 30 集;
(b) 给 `stand_off_m` 一个自己实测的容差(三轮中位数 0.189–0.223,±0.02 是下限),不要再借 `depth_m` 的。

**文档巡查补记(同日,用户追问"相应的文档更新了吗")**:按改动涉及的每个标识符 grep 了一遍 `docs/`,
查出一个**真 bug**:C 把 `--harden-pads` 从 `store_true` 改成了 `nargs="?"` 带 choices,而
`docs/reference/runbook.md:75` 与 `docs/pick_place_collection.md:26` 里记录的命令是
`--harden-pads outputs/scene outputs/scene/pick_place` —— 可选值会**吞掉第一个位置参数**,
实测报 `invalid choice: 'outputs/scene'` 后 exit 2,即 pick_place 的建场景命令被改坏。
已改成 `--harden-pads`(仍是 store_true)+ 独立的 `--pad-level {object,crate}`(默认 object):
文档里的写法逐字复原,两套标定仍然都能从 CLI 选到。crate 场景 XML 不受影响(重建后与修复前逐字相同)。
这条值得记:**改一个 flag 的 arity 之前,先 grep 文档里它的既有写法**。

**根因与裁决(同日)**:把 gate 的失败追到底,发现它不是 gate 自己的毛病,而是走入的**站位交付**问题。
三轮 30 局里 29 局有实测站位,成功 26 局落在 0.148–0.265 m,3 局 `[descend] never seated` 落在
0.282 / 0.285 / 0.301 —— **零重叠**,与 2026-09-10 难度场景那条界(≥0.26 败)一致,但这次是**普通场景**,
说明界是这条线本身的,不是 yaw/内缩设计带来的。交付误差(实测 − 计划)均值 ≈ 0、**sd 5.7 cm、跨度 −9.8…+12.0 cm**;
计划站位跨度只有 8.7 cm 且三轮逐字相同(由 seed 决定),实测跨度 15.3 cm ——
**步态噪声比整个计划分布还大,是散布而不是计划在挑哪局成功。** 横向同理:有一局
`[stand] lateral -14.1 cm off the centre line after 3 walk-ins`,重试了三次没收敛。

对**采集**的影响比对 gate 严重:每一集被留下都是因为散布碰巧落在近侧,数据集会只有近站位,
没有远站位的正确行为 —— 和 memory `policy-stop-point-is-set-by-the-crawl` 同形。

**用户裁决 2026-09-11:"如果这个是数据采集的策略调整问题的话先放一放,之后再调整。"**
因此本条搁置,gate 的 `stand_off_m` 容差一并搁置(同一个根因),**没有**为让改动通过而放宽任何容差。
之后接手时最便宜的第一步:查走入的重试/收敛判据覆不覆盖深度轴(单臂线有
`tasks/bottle/arrival_validation.py`,双臂线没有对应层)。


## 2026-09-12 腰锁的依据过期了:`diag_lean.py` 恢复重测,release 0.09 / v1.1 0.87

**起因**(用户,解释双臂采集流程时追问):"腰部采集的上半身弯不下腰是因为腰部关节锁死了是吗" → 锁确实在
(`BimanualArms.__init__` 的 `ik.lo[12:15] = ik.hi[12:15] = 0`,expert 每 tick 主动把 waist=0 推进
`upper_body_position[17]` 的 0–2 槽),但锁的**依据**是 `diag_lean.py` 当年测的"命令的腰前倾只交付 2–10%"。
那个数字是 `policy/release` 的,而 `SONIC_POLICY` 自 2026-09-11 起默认 `policy/sonic_v1_1`。用户:"在 v1.1 上恢复并重跑一次 diag_lean.py"。

**恢复**:`src/locomani/backends/sonic/diag_lean.py`(原件 `78536ae:src/teleop_free/diag_lean.py`,被 refactor step 2 删除)。
与 2026-09-11 恢复 `diag_wrist.py` 同一模式、同一目录。协议原样保留:一条冻结的前伸手臂姿势,腰 pitch 扫
0 / 0.10 / 0.20 / 0.30 / 0.40 / 0.52 rad,每档保持 2.5 s,两条通道(planner = 双臂/单臂 expert 用的 17 维;
streamed = 29 维 pose 流)。只测量,**没有改任何生产常量**。

**结果**(instance 1 / GPU1,scene `outputs/scene/sim`,两个栈只差 `SONIC_POLICY`,同一小时):

| cmd rad | release 交付 | ratio | v1.1 交付 | ratio |
|---|---|---|---|---|
| 0.10 | 1.3° | 0.23 | 4.1° | **0.72** |
| 0.20 | 1.4° | 0.12 | 9.9° | **0.86** |
| 0.30 | 1.6° | 0.09 | 17.8° | **1.03** |
| 0.40 | 1.8° | 0.08 | 23.2° | **1.01** |
| 0.52 | 2.0° | 0.07 | 25.9° | **0.87** |

planner 通道中位:release **0.09**(0.07–0.23)、v1.1 **0.87**(0.72–1.03);streamed 0.10 / 0.94。
**对照组做对了它该做的事**:release 读到 7–23%,与历史记录的 2–10% 同量级 —— 恢复的脚本复现了当年的测量,
差异全部来自 checkpoint(9.7×)。v1.1 满命令时腰到 25.9°,**顶到 MJCF 的 29.79° 行程**,即此时的约束是硬件
不是 policy。第二轮(`arm/`,planner-only)独立复现:release 0.06–0.23,v1.1 0.72–1.02。

**"腰买到什么"要拆开看,不能直接读手掌位移。** 两个 checkpoint 的手掌都下沉了 11–12 cm,但 release 的腰只动了
2° —— 所以位移不能算到腰头上。用实测关节做 FK(冻结实测手臂,分别用交付的腰角和把腰扳直)拆分:

| policy | cmd 0.52 时 | 腰自身贡献 | 手臂自身贡献(命令恒定) |
|---|---|---|---|
| release | 腰 1.9° | 前 −0.2 cm / 高 −0.7 cm | 高 **−7.1 cm** |
| v1.1 | 腰 25.7° | 前 −2.2 cm / 高 **−10.4 cm** | 高 +1.0 cm |

- **v1.1**:下沉几乎全部是腰给的,手臂纹丝不动(+1 cm)。腰是**降高度**的工具,不是**加前伸**的工具 ——
  它还倒扣 2.2 cm 前伸(手掌挂在腰关节**下方**,前倾时向后下方划弧)。骨盆高度全程不变(0.788),**不是偷偷下蹲**。
- **release**:命令腰前倾不但一点腰都不给,还**把手臂跟踪带坏**了(右肩 pitch 误差 8.2→13.5°、肘 −6.0→+12.3°,
  手臂自身就掉了 7.1 cm)。这解释了当年"命令 34° 实测 2.8°"的 pilot 4 为什么还伴随手的位置漂移。
- 代价:v1.1 满前倾时骨盆漂移 2.1 cm(0 前倾时 0.1 cm)。

**没有改生产常量。** `WAIST_PITCH_MAX = 0.0` 和三条 IK 边界保持原样,只把 `bimanual_carry.py` 里过期的注释、
`docs/crate_collection.md` §6 和 `docs/reference/silent_failures.md` 的诊断表改成 checkpoint-conditional 的说法
(后者顺带修掉一个更老的错:表头写的 `src/teleop_free/diag_*.py` 路径早已不存在,表里大部分脚本只在 git 历史里)。

**留给用户裁决**:解不解腰。解的话最小代价路径是 ready-stand pilot(`wall_walk=false`,变量最少)先跑 10 集
A/B —— 但注意两件事:(1) 一旦上身参考带腰,**所有既有 crate baseline 的数据含义都变了**(它们的 waist 列恒为 0),
`gate.py` 的 baseline 要重录;(2) 这条线真正卡着的是走入的站位交付散布(本文件 09-11 条,用户已搁置),
腰能降 10 cm 高度但换不来前伸,对"站得太远够不到"这个失败模式**没有直接帮助**。

产物:`outputs/diag/lean_20260912/`(`summary.md`、`lean_{release,sonic_v1_1}.json` 两通道、
`arm/lean_*.json` 带实测手臂关节、每次运行的 stack/diag 日志)。

## 附:双臂闭环首个正式结果(原 `docs/evaluation.md`,2026-09-18 并入)

### 2026-09-15 结果：`crate_vla200_record_20260914` checkpoint-10000

验证批 `outputs/crate_eval10_20260915c` 9/10；正式 100 局 `outputs/crate_eval100_20260915`
（seed 6000，instance 2 / GPU 1，端口 5700）：**71/100**，Wilson 95% 0.61–0.79。失败 20 局
"闭合了但没抬起"（其中 18 局结束时双手都没接触箱子、抬升 0），5 局到桌边从未闭合（最近 0.34 m），
4 局 reset 后没站稳。成功局抬升中位 15.2 cm、倾角 7.4°、18 s。同一批前 10 个 seed 在 100 局跑里
8/10、验证跑 9/10，单局不可跨栈重启复现。原片 `videos_NN/`，预览 `previews/`
（100 片，最大 499,012 字节，共 38.2 MB，`verification.json`）。与单臂同一 checkpoint 家族的
对照：单臂 17/100（见上文 2026-09-15 条目）。
