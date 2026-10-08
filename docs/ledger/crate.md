> **台账(append-only)**:本文件只在末尾追加新的日期化章节,不改写、不删除已有结论。
> 要更正一条结论,在新章节里写清楚并链接回旧的那一条,不要就地覆盖。
> 分类与格式规则见 `docs/reference/standards.md` §3。

# 台账:crate(双臂搬箱)线

## 2026-09-07 wall pilot 验收判据(来自 `docs/bimanual_design.md` §5,表格原样保留在教程里)

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
`docs/reference/runbook.md:75` 与 `docs/single_arm_design.md:26` 里记录的命令是
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
`docs/bimanual_design.md` §6 和 `docs/reference/silent_failures.md` 的诊断表改成 checkpoint-conditional 的说法
(后者顺带修掉一个更老的错:表头写的 `src/teleop_free/diag_*.py` 路径早已不存在,表里大部分脚本只在 git 历史里)。

**留给用户裁决**:解不解腰。解的话最小代价路径是 ready-stand pilot(`wall_walk=false`,变量最少)先跑 10 集
A/B —— 但注意两件事:(1) 一旦上身参考带腰,**所有既有 crate baseline 的数据含义都变了**(它们的 waist 列恒为 0),
`gate.py` 的 baseline 要重录;(2) 这条线真正卡着的是走入的站位交付散布(本文件 09-11 条,用户已搁置),
腰能降 10 cm 高度但换不来前伸,对"站得太远够不到"这个失败模式**没有直接帮助**。

产物:`outputs/diag/lean_20260912/`(`summary.md`、`lean_{release,sonic_v1_1}.json` 两通道、
`arm/lean_*.json` 带实测手臂关节、每次运行的 stack/diag 日志)。

## 附:双臂闭环首个正式结果(原 `docs/single_arm_design.md`,2026-09-18 并入)

### 2026-09-15 结果：`crate_vla200_record_20260914` checkpoint-10000

验证批 `outputs/crate_eval10_20260915c` 9/10；正式 100 局 `outputs/crate_eval100_20260915`
（seed 6000，instance 2 / GPU 1，端口 5700）：**71/100**，Wilson 95% 0.61–0.79。失败 20 局
"闭合了但没抬起"（其中 18 局结束时双手都没接触箱子、抬升 0），5 局到桌边从未闭合（最近 0.34 m），
4 局 reset 后没站稳。成功局抬升中位 15.2 cm、倾角 7.4°、18 s。同一批前 10 个 seed 在 100 局跑里
8/10、验证跑 9/10，单局不可跨栈重启复现。原片 `videos_NN/`，预览 `previews/`
（100 片，最大 499,012 字节，共 38.2 MB，`verification.json`）。与单臂同一 checkpoint 家族的
对照：单臂 17/100（见上文 2026-09-15 条目）。

## 2026-09-19 文档走查同批:双臂 20 局采集 17/20,一局在出生阶段即被判掉

**为什么跑**:按 README/runbook 的指令并行采集单双臂各 20 局(GPU 0 / 实例 0 与
GPU 1 / 实例 1),不论成败都保留第三视角视频。同批的单臂结果见
[单臂台账 §37](bottle.md)。

**结果**:`bimanual_20ep_20260919`,seed0 7000,batch_rounds 10 × max_batches 2,
**17/20**。失败 3 局:`basket_tilt` 2、`invalid_initial_distance` 1。

**一个只有双臂线才有的记账缺口**:`invalid_initial_distance` 那一局
(`visual_crate_s7100_01`)**没有视频**。`tasks/crate/expert.py:166` 在回合开始前就
返回,此时没有任何画面存在,所以 20 行 manifest 只对应 19 个 mp4。这不是丢帧也不是
写坏——跑起来之后才失败的(`basket_tilt`、`runtime_error`)都有完整视频。要"每局都有
视频"的场合,双臂线要按轮次而不是按文件数记账。

**并行可行性**(实测):两个录制栈 pid 峰值 2325 / 3840,CPU 15.36 核,两卡各一个 sim
(1490 MiB)。与 `docs/bimanual.md:131` 记的"约 1800"同量级,**第三个栈仍然不可以**。

## 2026-09-22 SONIC v1.1 有没有"小步微调"通道:没有;而且 09-11 的速度台阶图在今天的部署上不复现

**问题。** 纸箱走篮子线的失败多在停位(`docs/handoff/box-via-basket-line-20260922.md`:30 局里 8 局横向偏出、2 局停远)。
用户问 v1.1 能否像速度跟踪型 G1 控制器那样原地踏步、几厘米几度地微调。规划器接口本身没有角速度通道
(`PlannerMessage` 只有 mode / movement 单位向量 / facing 单位向量 / speed 标量 / height),所以只能量"它实际能给什么"。
工具 `src/locomani/backends/sonic/diag_shuffle.py`,本次新增 `--march`(SLOW_WALK 下 movement 向量置零);
runner `scripts/experiments/gait_micro_20260922.sh`;栈 `locomani.stack.cli up --no-record`,v1.1,默认部署根,GPU 0,
产物 `outputs/diagnostics/gait_micro_v11_20260922{,_knee}/legs.json`。每腿 5 s,1.5 s 起算,每档 3 次,60 腿 0 摔倒。

**1. 零 movement 的 SLOW_WALK 不踏步。** 0.30 / 0.35 / 0.45 各 6 腿:净速 0.000–0.010 m/s,步频 0–0.3 步/s
(与 IDLE 对照 0.001–0.005 同量级),偶发一次 0.4 步/s 的抖动。v1.1 没有"原地踏步待命"态,movement 为零就是站着。

**2. 今天的 speed → 净速图是连续的,不是 09-11 量到的台阶。** 同一 checkpoint、同一工具、同一协议(`bottle.md` §21):

| 命令 | 09-11 v1.1 净速 | 今天 v1.1 净速(中位[min,max]) | 今天 cm/步 | 今天 carry cm |
|---|---|---|---|---|
| 0.20 | 0.009 | **0.050** [0.042, 0.085] | 5.7–10.0 | 1.9–7.7 |
| 0.24 | — | 0.130 [0.126, 0.157] | 11–14 | 4.1–4.8 |
| 0.26 | 0.009 | 0.164 [0.156, 0.166] | 11–15 | 4.4–7.9 |
| 0.28 | 0.007 | 0.171 [0.152, 0.179] | 11–13 | 8.4–9.4 |
| 0.30 | 0.044 | **0.188** [0.179, 0.201] | 13 | 8.6–10.4 |
| 0.32 | 0.067 | 0.254 [0.205, 0.281] | 14–20 | 6.5–8.7 |
| 0.35 | 0.083 | 0.247 [0.245, 0.295] | 17–22 | 8.9–11.8 |

今天的曲线和 09-11 的 **release** 曲线更像(release 0.30 → 0.203,0.35 → 0.245)。部署 receipt 与 deploy.log 都确认加载的是
`policy/sonic_v1_1`,所以不是 checkpoint 贴错。09-11 之后变过的东西:落座高度修复(同日)、部署二进制、栈的起法(旧 `start_stack.sh`
"walk forward" → `Stack`)。哪一个改变了映射没有分离,**但可操作的事实是:仓库里按 09-11 台阶图挑的常量(`walk_speed_min` 0.35
"部署在 0.3 以下不迈步"、`visible_shuffle_speed` 0.32 "台阶之上的爬行")今天对应的是 0.25 m/s + 9–12 cm 的 carry,不再是爬行。**
这直接对上纸箱线走入实测过冲 10–20 cm。carry 也回到了 release 的量级(09-11 v1.1 只有 2–4 cm)。

**3. 最小可控步长约 5 cm/步。** 0.20 是今天量到的最低档:0.6–0.9 步/s、6–10 cm/步、净 0.05 m/s,三次散度 2×。
0.20 以下今天没量(09-11 的 0.185 → 0.008 属于已失效的那张图)。没有"一步 1–2 cm"的档。

**4. 横向 / 斜向:通道在 0.32 附近打开,打开后就是 15–25 cm/s,没有细档。**
movement 与 facing 夹角 crab,`lat` 为起始朝向系的横向分速,`turned` 为 5 s 内偏航变化:

| 命令 | crab 30° lat | crab 45° lat | crab 90° lat | 90° carry cm |
|---|---|---|---|---|
| 0.30 | 0.06–0.11 | 0.06–0.10 | **0.03 / 0.03 / 0.18**(双稳) | 2.4–11 |
| 0.32 | 0.09–0.11 | 0.11–0.13(1 腿 stall 0.02) | 0.12–0.20 | 9.8–13.5 |
| 0.35 | 0.11–0.14 | 0.11–0.16 | 0.18–0.25 | **15–16** |

90° 时 `fwd` 为 −0.02 … −0.06(横移时略后退),`travel_err` +8 … +27°。

**5. 走着的时候朝向守不住。** 60 腿里每一条走动腿的 `turned` 都是 +3 … +16°(中位 +8°),方向一致(左偏),IDLE 与 march 腿 +0 … +6°。
配合 `primitives_walk.turn_to` 记录的 7–8° 朝向死区,v1.1 在 5 s 的走动里会积累 10° 量级的偏航,而停下来后又转不进 7° 以内。

**结论。** v1.1 不提供"原地踏步 + 三轴连续微调":零 movement 是站立;前向最小量子 ~5 cm/步、0.05 m/s;横向 ≥0.32 才开、开了就 15–25 cm/s
且停下滑 15 cm;走动中偏航漂移 3–16°,停下的朝向精度 7–8°。能做微调的是 ScaleBFM 那一类 keypoint 驱动控制器(`scalebfm.md`,站位散布 1–2 cm)。
对纸箱线更直接的事:先按今天的图重挑 `walk_speed_min` / 减速带(0.20–0.24 交付 0.05–0.13 m/s、carry 2–5 cm,比 0.35 的 0.25 m/s + 10 cm 更适合最后 30 cm),
而不是继续在 09-11 的台阶图上修 `stand_off`。

## 2026-09-22 慢走入对照(`walk_speed_min` 0.35 → 0.20):停位平均近了 7 cm,散布没有变小,0/20

同一天速度图重测(上一节)之后的第一组对照。`configs/experiments/box_pilot_slow_20260922.yaml` 与 `box_pilot_env_20260922`
唯一差别是 `visual.walk_speed_min` 0.35 → 0.20(`walk_speed` 仍 0.35;速度律 `0.7·|误差|` 夹在两者之间,所以最后 ~0.3 m 命令落在 0.20–0.24)。
GPU 0,instance 0,seed 7000 起 20 局,`outputs/runs/box_pilot_slow_20260922`。

| 组 | 有站位的局 | 接受站位深度 中位 / sd / 范围 (m) | 横向 中位 / sd (m) | 结果 | why 前三 |
|---|---|---|---|---|---|
| env(0.35) | 16/20 | 0.320 / 0.071 / 0.26–0.50 | +0.095 / 0.156 | 0/20 | basket_displaced 8, reach_stalled 6, fall 2 |
| slow(0.20) | 19/20 | **0.389** / 0.081 / 0.26–0.55 | +0.118 / 0.118 | 0/20 | reach_stalled 7, basket_displaced 6, fall 4 |

`depth_m` 是接受站位时视觉抓点在机器人系的 x;计划停点 `stand_off` 0.50。慢走入把过冲平均收回 7 cm(0.32 → 0.39,与 carry 10 cm → 2–5 cm 一致),
**但 sd 没有变(0.07 → 0.08)**,横向 sd 也只是 0.16 → 0.12。所以刹车速度不是散布的来源。

散布来源的线索,三组 pilot 一致:**深度过冲与横向误差负相关**(corr(depth, |lat|) = −0.50 / −0.24 / −0.51),过冲局的 |lat| 中位 0.10–0.15 m,
未过冲局 0.03–0.11 m。`visual_approach.py` 的 walk → settle 切换要求 `|point[1]| < stand_lateral_tolerance`(0.14)**同时**满足,
横向没进容差时机器人继续往前走着修横向,深度就一路过冲。横向本身有系统偏置:慢走入组 19 个站位里 15 个 lateral > 0(中位 +0.12);
上一节量到每条走动腿偏航左漂 +3…+16°/5 s,在 0.5 m 距离上 8° 就是 7 cm。

**下一步**(未做):把横向容差从 settle 闸门里拿出来(先按深度停、停下再用 IDLE facing 修朝向),或者在走入时把 facing 按实测偏航反向补偿;
两者都不需要新的步态档位。`pregrasp_above` 和 `reach_stalled` 对齐两项仍在交接清单里。

## 2026-09-22 作废:合成桌子上的三组纸箱 pilot 与上面两节的停位结论

用户看了 `box_pilot_slow_20260922/reel_failures.mp4` 指出场景不对:纸箱线本该在扫描的 Blender 仓库房间里、箱子放在中间的平台上
(箱沿线 9-20 起一直用的 `outputs/box_rim_execution_20260920/scene_reverse_camera`),而 `0abb131` 把纸箱搬到篮子线时新建了一个
合成桌子场景 `outputs/scene/box_table37`(`bimanual_scene.py --preset box_bimanual --table-top 0.371`),没有决策记录。
**作废**:`box_pilot_20260922`、`box_pilot_env_20260922`、`box_pilot_slow_20260922`(以及只跑了 10 局就停掉的 `box_pilot_turn_20260922`)
四组数据,和上面"慢走入对照"一节里关于停位散布、横向偏置、走→settle 闸门的结论——它们是合成桌子上的现象,房间里要重测。
保留的:同日速度图重测(`diag_shuffle`,默认 `sim` 场景,与纸箱无关);`stand_lateral_gate` 代码与测试(机制本身与场景无关,但未在房间里验证)。
用户指定的起点:箱沿线验证局 `dy0.225_r00` 的初始化位置(房间 `robot_start` (0, −1.205, 90°) 再前移 0.225 m),后续 pilot 都从那里起。

## 2026-09-22 纸箱线回到物流房间:篮子线 expert 从 `dy0.225_r00` 的起点跑 20 局,0/20

`configs/experiments/box_room_20260922.yaml`:场景 `outputs/box_rim_execution_20260920/scene_reverse_camera`(扫描房间、箱子在托盘垛上),
`visual.start_pose: scene` + `start_advance: 0.225`(房间 `robot_start` (0, −1.205, 90°) 前移到 (0, −0.98),离平台沿 0.425 m,箱心 0.59 m),
其余旋钮同作废的 `box_pilot_env_20260922`。代码改动:expert 放置按场景 `robot_start` 朝向工作(走入轴、`hold_yaw`、facing、物体偏航取场景自身),
`initial_robot_xy`/`edge_distance`/`scene_object_yaw` 在 `visual_approach.py`。录像确认场景正确(`videos/`,`reel_failures.mp4` 前 6 局)。

| 项 | 值 |
|---|---|
| 结果 | 0/20;why:reach_stalled 8,external_fall_abort 6,unsupported_assisted_reach 4,basket_displaced 1,approach_unsupported_reach 1 |
| 初始平台沿距离 | 0.385–0.394(实测,起点放置正确) |
| 接受站位深度 | 中位 0.472,sd 0.067,范围 0.27–0.66(计划 0.50) |
| 接受站位横向 | 中位 +0.013,sd 0.077(合成桌子上是 +0.10–0.12 / sd 0.12–0.16 —— 那个偏置是合成场景的) |
| 最大抬起 | 0.16 m(一局),触碰记录 19 局空 |
| 每局用时中位 | 17 s |

新现象:摔倒 6/20(合成桌子上 2–4),起点离平台只有 0.4 m、几乎不走;站位分布两头都有(0.27 与 0.66)。未分析。

## 2026-09-23 warehouse 纸箱线十组 pilot 全 0/10;根因是纸箱网格拖慢渲染,前九组都在 0.35× 慢仿真里测的

同 seed 7000,每组 10 局,每次改一个旋钮(TODO 2.4.1–2.4.10)。前九组 sim 只有 0.30–0.36× 实时,**上一节的 `box_room_20260922` 也在内**(0.32–0.36,按 trace 行数 × 0.02 s ÷ 墙钟算),
所以这些组的停位、越线、摔倒结论都要在满速下重测。证据和复现见 [warehouse 渲染负担交接](../handoff/warehouse-render-cost-20260923.md)。

| run | 改动 | 结果 |
|---|---|---|
| `box_room_fb07_20260923` | 伸手反馈 0.7/0.04 | 0/10 |
| `box_room_carry_20260923` | 整个 carry 块回退 | 0/10,摔倒 3→0 |
| `box_room_so40_20260923` | `stand_off` 0.40 | 0/10,失败移到碰动箱子 5/10 |
| `box_room_approach_20260923` / `box_room_steer_20260923` | 起点 advance 0 + 刹车余量 0.05 / `walk_steer: depth` | 0/10 / 0/10,停位离散 1.2 cm |
| `box_room_grasp_20260923` | 守线刹车 + staged 伸手 + 1 cm 抓深 + 后退 0.3 m | 0/10,守线 6/10 越线 |
| `box_room_grasp_hanging_20260923` | 手臂下垂走、刹后抬手(抬着手走时 9/10 在 1.6 s 内撞箱) | 0/10,走路段接触 4/10 |
| `box_room_grasp_hanging2_20260923` (+`_rerun`) | 站起后重放置 + 世界轴行走,`guard_margin` 0.03→0.15 | 0/10;原样重跑仍 0.35×,排除资源抢占 |
| `box_room_grasp_offline_20260923` | `backend.render: offline` | sim 0.95×,走路 0/10 碰箱;0/10,停在 depth 0.53–0.72 m,手指离抓点 27–44 cm |
| `box_room_grasp_offline_m003_20260923` | 离线 + `guard_margin` 0.15→0.03 | 0/10;停位中位 0.63→0.49 m;**伸手段无效**(09-24:离线伸手时钟冻结,手从未离开 ready 姿态);2 局 ready 姿态的手压在近壁上 |
| `box_room_grasp_offline_m003b_20260924` | m003 原样,伸手时钟修复后 | 0/10,但 10/10 伸手执行:5 局合爪+试提(抓深 1 cm 只夹到壁顶几毫米,`probe_not_attached` 3 / `lost_attachment` 2),4 局远站位(≥0.54)掌落壁顶停滞,1 局(0.40)ready 姿态推走箱子 |
| `box_room_grasp_offline_deep03_20260924` | m003b + 抓深 0.01→0.03 | 0/10,0 合爪(m003b 5):8 局下落段停滞,6 局箱子被推斜>5°;跨壁横向误差下落前已有 2–6 cm,指垫顶壁面后横向修正积分到 ±15 cm 上限把箱子推倒 |
| `box_room_grasp_offline_lifted_20260924` | m003b + `ready_pose: lifted`（`elbow_min_deg -60`） | 0/10;抬手期间接触 0/10、箱子位移 0/10(此前 5 局/2 局);4 局合爪试提(1 cm 抓深夹不住),6 局下落停滞——4 局横向修正顶壁饱和 ±15 cm,2 局停在目标上方 2 cm |
| `box_room_grasp_offline_hold_20260924` | lifted + `lateral_hold_below_rim` | 0/10,试提 3;沿下冻结生效但唯一一次横向饱和发生在沿上方;4 局远站位 |
| `box_room_grasp_offline_align_20260924` | hold + `descend_align_tolerance 0.015` | 0/10,**试提 6**(1 cm 抓深全夹不住);align 进入 ≤2.8 cm 出口 ≤1.6 cm;停滞 2 远站位 + 2 局下落中横向再漂 3–5 cm |
| `box_room_grasp_offline_waist035_20260924` | align + `carry.waist_pitch_max` 0.52→0.35 | 0/10,试提 1(align 6):腰命令 20° 实测 12–16°——送达按比例约 2/3,不是上限;送达差 L 8.3 cm(align 6.7),残差中位 4.1(2.8)。负结果,基线仍是 align |
| `box_room_grasp_offline_ff_20260924` | align + 前馈 前 +2.5 / 上 −4 cm | 0/10,试提 0(align 6):前向送达差 −2.8→−4.3,z 修正饱和局 3→5,残差中位 4.7(2.8);1 局伸手段就合爪。送达差是伸展饱和,不是常量偏移;负结果 |

根因:`nor_cal_box` 显示网格 46.3 万面 + 2 盏投影灯,3 路相机渲染 39–44 ms(walk 1.3 ms);慢仿真下 deploy 按墙钟推进,
步长 0.17–0.34 m(正常 0.14)、实际速度约 3 倍,这就是视频里的"过冲撞箱"。离线这组是第一组可信的 warehouse 数据:
失败从"撞箱"变成"停得太远",守线余量回到 0.03 后停位问题解决。**09-24 更正**:两轮离线 run 的伸手段都没有执行(占位帧戳晚于循环的 `now`,参考时钟 0/440 tick 推进),
"手够不到"的读法作废;修复后重跑 m003 配置才是第一次离线伸手测量(`docs/handoff/warehouse-render-cost-20260923.md` 末节)。

## 2026-09-24 渲染负担的根因更正：单块网格的 2²⁰ 顶点门槛；Isaac 三路按需渲染 89 ms

三路相机（640×480）渲染一组，在 GPU0 上单独测，用真实 trace 里的状态（`scripts/experiments/bench_camera_render_20260924.py`）：

| 场景 | 串行 p50 / p90 | 每路一个进程 |
|---|---|---|
| `warehouse_textured`（现行基线） | 3.2 / 6.9 ms | 1.8 ms/帧 |
| `warehouse`（`scene_reverse_camera`，复现 agent 实时跑出 RTF 0.35 的场景） | 36.7 / 46.9 ms | 28.6 ms/帧 |
| crate / bottle | 1.7 / 1.6 ms | 1.0 / 0.9 ms/帧 |

两个 warehouse 场景的纸箱面数相同（462,730），区别是旧场景放在一块网格里，textured 按材质拆成 9 块（最大 231,168）。
同一批状态交叉对照（`ab_carton.py`）：旧场景藏掉纸箱 36→2–4 ms。合成网格扫描（`mesh_size_sweep.py`、`mesh_2pow20_check.py`）：
单块 349,525 面 0.31 ms，349,526 面 3.91 ms，正好是 3×面数 = 2²⁰；90 万面一块 10.2 ms，拆 9 块 0.6 ms。
所以上面"纸箱网格拖慢渲染"一节说的是对的现象，根因是单块网格超门槛，不是面数，解法是拆块不是减面。

Isaac（GPU1，`IsaacView` + warehouse USDC 场景，关自动曝光 / filmIso 30，三路相机，每帧 2 次 settle）：
连续 p50 72 / p90 107 ms；间隔 0.04 s 79 / 97 ms；**间隔 0.4 s（按需镜像的节奏）89 / 108 ms，p99 130，最大 135**。300 帧无空帧。
加上推理（bottle 评测 p50 72 ms），一轮约 160 ms，小于 0.4 s 的重规划周期。五个状态 × 三路相机与 MuJoCo 并排，几何对齐：
`isaac_view_check/view_check_grid.png`。证据全在 `outputs/diagnostics/render_speed_20260924/`。

镜像对真实 eval 栈冒烟（同日，TODO 4.7.9.10，`scripts/experiments/isaac_mirror_smoke_20260924.py`）：基线实验只改
`eval.image_source: isaac`，run `outputs/runs/isaac_mirror_smoke_20260924`。GPU0 被别的项目占着（72%），栈和镜像**都在 GPU1**，
所以耗时偏保守。50 次请求（每 0.4 s 一次）全部应答，seq 严格递增（中位步长 19 tick），状态年龄 p50 11 / p90 20 ms，
渲染 p50 98 / p90 120 ms，请求往返 p50 100 / p90 122 ms。按回报的 seq 取同一 qpos 渲 MuJoCo 并排 4 组，几何一致
（`eval/mirror_smoke/mirror_vs_mujoco.png`；没有驱动，机器人倒在箱边，画面是静止姿态）。SIGTERM 后镜像正常退出，
`eval/isaac_mirror.jsonl` 50 行。没测的：真实策略闭环、运动中的一致性闸门（4.7.9.8）。

## 2026-09-26 场景能力扫参：纸箱放在 10 cm 台上，专家 + SONIC 的"抓起并保持"随布局怎么变

方法见 [场景能力扫参](../scene_sweep.md)。专家 = `box_room_grasp_200_cap120_20260925` 的 visual 专家，`retreat_distance 0`；
每格 seed 50000 起，每批 5 局，Wilson 95% 判定；实例 0–4 同在 GPU0，不录制不渲染，每局 RTF 0.93–0.98。
报告与逐局 manifest：`outputs/sweeps/box_envelope_s{1,2}_20260926/report.md`、`outputs/runs/box_envelope_s{1,2}_20260926_*`。

**第 1 步（215 局，阈值 0.5）箱高 × 箱长**，宽 0.45，起点离台边 1.2 m 正对：

| 箱高（抓点） | 长 0.20 | 长 0.32 | 长 0.44 |
|---|---|---|---|
| 0.30（0.39） | 3/15 | 0/10 | 0/10 |
| 0.45（0.54） | 15/30 | 12/30 | 1/10 |
| 0.60（0.69） | 13/30 | 13/30 | 2/15 |
| 0.75（0.84） | 0/10 | 3/15 | 0/10 |

没有一格区间整体 ≥ 0.5。H ≥ 0.45 合并：停位深度 0.36–0.45 m 成 0.42–0.49，≥ 0.48 m ≤ 0.08。

**第 2 步（620 局，4 局不计，阈值 0.3）以 H0.60 × L0.32 × W0.45 为中心逐维**；下表按每局实抽值合并（含中心格 13/30）：

| 维度 | 结果 |
|---|---|
| 箱宽 | 0.30：7/30，0.38：7/30，0.45：13/30，0.55：3/25（below） |
| 起点离台边 0.6–2.0 m | 各段 0.17–0.60，无单调趋势；合并 59/160 = 0.37 |
| 起点朝向 ±30° | 各段 0.26–0.48，±20–30° 两端 0.26/0.27；合并 72/206 = 0.35 |
| 起点横移（+ = 机器人在箱子中线左侧） | −8..0 cm：13/29 ≈ 中心；0..+8 cm：6/30 = 0.20；\|横移\| > 8 cm 两侧都 2/25（below） |
| 箱子 setback | 0–2.5 cm 4/13，2.5–5 cm 4/17，≥ 5 cm 0/10（below） |
| 箱子偏角 | 中心 0°：13/30；0 < \|偏角\| ≤ 3.5°：7/30；3.5–7°：4/34；> 7°：1/15（below） |

读法：起点的距离和朝向几乎不影响（guard 走法把朝向拉回走入轴、停在线上）；决定成败的是停下后手到抓点的几何——
箱子偏角（几度就掉一半）、setback（每多 1 cm 手多伸 1 cm）、横移（右手坐标系不镜像，左偏先坏）、箱宽只在 0.45 附近。
中心格 13/30 在各链第一档里偏高（第一档合并约 0.3），真实中心率大概 0.3–0.4。提前停下的 above 格（8/15、6/10）估计偏高，不单独当结论。

## 2026-09-27 扫参 835 局的逐帧复盘：停在哪一步、手在哪里（脚本 `outputs/sweeps/box_envelope_s2_20260926/analysis/`）

- `reach_stalled`（278 局，占 33%）全部发生在 descend。停住时按箱体坐标分解两手的误差：主导轴 across 53%、along 34%、up 13%；
  两手**同向**的 across 偏移中位 2.9 cm（p75 4.8），两手张开度误差只有 1.2 cm；2/3 偏向机器人左侧。
  这个同向偏移在 descend **开始时只有 0.5 cm**（align 阶段只有 3% 超时），是在 8 cm 的下探过程中长出来的（67% 的局增长 > 1 cm，缩小的 2%）。
  成功局闭合时同向偏移中位 1.2 cm。也就是说：卡住不是因为伸不到，而是下探时整对手横向漂走，而 `lateral_hold_below_rim` 在贴近箱沿后冻结了横向修正。
- `lost_attachment`（191 局，23%）62% 在 stand、21% 在 hold：箱子已经抬起（峰值 0.147 m），97% 结束时夹爪完全闭合（夹着空气），
  失去附着发生在站直进度中位 0.85–0.96、躯干已直（6°）、`stand_forward_m` 顶在 0.35 的下限——是站直末段手被往前推时，1 cm 深的捏夹从箱沿上滑脱。
- `basket_tilt`（59 局，7%）发生在起抬（站直进度中位 0.32，附着误差在 0.02 就超 3 cm，躯干还前倾 29°）：起抬时两手不对称，箱子绕箱沿铰链翻。
- 所有失败都是终局：专家没有任何一处重试（stall → 超时结束；probe 失败 → 结束）。

## 2026-09-27 专家改进 ①② 的 A/B（中心格 H0.60 × L0.32 × W0.45，seed 50000+，每组 30–60 局）

| 组 | 成功 | 失败构成 |
|---|---|---|
| 基线（`box_envelope_s1…height0.6_length0.32` 13/30 + `box_centre_base_20260927` 15/30） | 28/60 = 0.47 | 脱手 27%，下探卡住 12% |
| ① `lateral_hold_common_mode`（`box_centre_cm_{a,b}_20260927`） | 26/60 = 0.43 | 脱手 12%，**卡住 37%** |
| ② 重试各 1 次（`box_centre_retry_a_20260927`） | 12/30 = 0.40 | 脱手 23%，卡住 20%，站不直 13% |
| ① + ②（`box_centre_cmretry_{a,b}_20260927`） | 25/60 = 0.42 | 卡住 33%，脱手 18% |

- ① 没有提高成功率，只把失败从"站直时脱手"换成"下探卡住"；闭合的局对得更正（1.0 → 0.8 cm），但更多局没闭合。
  卡住时两手同向偏移中位 4.1 → 2.9 cm，14/22 两手都没碰到箱子、只有 5% 到了沿下——修正在发，手没送到。
- ② 的重试会触发（30 局里下探重试 10 次、重坐 3 次），但第二次下探几乎都再次卡住（重试局 1/10、1/23 成功）。
  原因在箱子：到重试那一刻箱子已被推开 2.0 cm（只开 ②）/ **9.2 cm**（① + ②），重试后继续推到 10–30 cm（p75 70 cm）。
  共模横向修正把整对手一起挪，一侧的指垫顶着壁面就把箱子推走；重试只是再推一次。
- 结论：①② 都不设为默认（旋钮保留，默认关）。重试要有用，先要"箱子一动就停手"：下探期加位移闸门（reach 期已有 `basket_displaced_before_capture`，descend 期没有），退回后再判断能不能重来。
- 分析脚本 `outputs/ab_expert_20260927/compare.py`。

## 2026-09-27 专家改进 ③ 的 A/B（同一中心格、同一套 seed，每组 30 局；基线 28/60）

| 组 | 成功 | 失败构成 |
|---|---|---|
| 抓深 1 → 2 cm（`box_centre_depth2_20260927`） | 10/30 = 0.33 | 脱手 20%，卡住 20%，翻箱 10%，夹空 7% |
| 竖直附着反馈 `attachment_gain 1.0 / cap 0.03`（`box_centre_att_20260927`） | 15/30 = 0.50 | 卡住 27%，脱手 **13%**（基线 27%） |
| 两者叠加（`box_centre_attdepth2_20260927`） | 8/30 = 0.27 | 卡住 40%，脱手 23% |

- 脱手的机理（逐帧）：不是手被往前推，而是手相对箱子**竖直向上**滑（左 3.5 cm、右 1.4 cm 中位），随受载时间近似线性增长
  （失败局 0.6 → 3.4 cm / 8 s，成功局 0.2 → 1.8 cm），到失败时 97% 夹爪闭合夹空；指垫几何在本场景已是硬化参数。
  ③ 因此改成闭环：站直/保持阶段把每只手相对箱子的竖直漂移（低通 0.3 s、上限 3 cm）从该手目标里扣掉，探提阶段不作用。
- 抓深 2 cm 是负结果（与 09-24 的 3 cm 一致：下探更深把指垫顶偏），不采用。
- 附着反馈把站直期脱手减半（27 → 13%），成功率 0.50 与基线 0.47 分不开（n=30）；卡住 27% 在基线各次重跑的 7–37% 散布内。

## 2026-09-27 专家改进 ④ 的 A/B：起抬前坐好检查（`seat_depth 0.01`、`seat_skew 0.01`，不合格花一次重坐）

`box_centre_seat_20260927`：**7/30 = 0.23**（基线 28/60 = 0.47）。21 局走到闭合，第一次检查只有 3 局合格（闭合时指垫在沿下的深度中位 1.3 cm、p25 0.8 cm），
18 局花掉重坐；重坐后再检查 1/10 合格，7 局在第二次下探卡住。基线在这种坐法下本来能成 47%，检查把能成的局也退回去重坐，而重坐走的是 ② 的路（箱子已被推动）。
结论：不采用；就算要查坐姿，阈值也得按控制器实际交付的 0.8–1.7 cm 来定，且退回重坐前先要有 ②b 的位移闸门。

## 2026-09-27 专家改进 ⑤ 的 A/B：刹车后原地转向（`face_after_brake`），在第 2 步的四个格子上各 30 局

| 格子（基线） | 模式 | 请求角中位 | 交付/请求 | 结果 |
|---|---|---|---|---|
| 横移 +0..8 cm（6/30） | centre | 4.0° | 0.22 | **2/30** |
| 横移 +8..16 cm（2/25） | centre | 13.5° | 0.59 | 3/30 |
| 箱偏角 −7..0°（4/30） | box_yaw | 5.1° | 0.39 | 8/30 |
| 箱偏角 +7..14°（1/15） | box_yaw | 9.0° | 0.86 | 1/30 |

- 转向本身：绝对朝向指令保持 6 s，≤ 6° 基本不动（交付 2–4 成），9–12° 1–3 s 转到位，12–23° 交付约 85%。`Primitives.turn_to` 的逐拍重瞄在这个死区内更差（7.5° 转 2.8°）。
- 结论：⑤ 不采用。小偏移转不动；大偏移即使转到位（+7..14° 交付 86%）成功率也没变（1/30），说明箱子偏角的敏感不是"身体没对正"造成的，原因仍未知（候选：沿壁闭合窗口、`RimRegion.nearest` 的沿沿滑移目标、俯仰跟随轴）。
- 中途修的两处：绝对朝向替代 turn_to；转向记录里的 numpy 布尔写 manifest 崩局（已修，四组重跑）。

**①–⑤ 总结**：五项都没有把中心格从 0.47 抬高；③ 的竖直附着反馈把站直期脱手 27 → 13% 是唯一一个机理上确认有效的改动。
基线的失败里能被专家层面改动触及的两块——下探卡住和脱手——分别撞在"控制器把整对手送偏 3–5 cm、修正就推箱"和"1 cm 捏夹的蠕滑"上；
重试类改动在没有下探期位移闸门（②b）之前只会把箱子推得更远。

## 2026-09-27 横向前馈的离线否定，与下探速度对照的由来

- 计划试"下探时固定向右预偏 2–3 cm"（卡住局两手同向左漂）。先查基线类 5 个运行 135 个走到下探的局：下探结束时两手同向横向误差中位 **−0.5 cm**、均值 −0.6，
  左偏只占 61%；只有 24 个卡住局在 −2.8 cm。固定预偏 +2 cm 把落在 ±3.5 cm 闭合窗内的比例从 87% 降到 84%。**系统性偏置只有半厘米，固定前馈没有用，不跑。**
- 漂移发生的时段：卡住局在下探**前 1.4 s、还在箱沿以上、横向修正未冻结**时从 0.5 漂到 3.9 cm，沿下 4.7 s 里反而回了 0.7 cm；
  成功局同一段只漂 0.9 cm。所以既不是冻结、也不是偏置，而是快速下落时的一次性横向甩动，修正跟不上。
- 由此改跑 `visual.descend_speed` 0.05 / 0.08 / 0.11（基线 0 = 0.16）各 30 局：`box_centre_slow{05,08,11}_20260927`。

## 2026-09-27 下探速度对照（`descend_speed` 0.05 / 0.08 / 0.11，各 30 局；基线 0 = 0.16，28/60）

| 速度 | 成功 | 到箱沿用时中位 | 沿上横向漂移中位 / >2.5 cm 的局 |
|---|---|---|---|
| 0.16（基线） | 28/60 = 0.47 | 1.3 s | 1.1 cm / 12% |
| 0.11 | 8/30 = 0.27 | 1.5 s | 0.6 cm / 20% |
| 0.08 | 11/30 = 0.37 | 1.7 s | 1.2 cm / 17% |
| 0.05 | 14/30 = 0.47 | 2.0 s | 0.8 cm / 13% |

放慢没有减少沿上的横向甩动（各档都是中位约 1 cm、13–20% 的局超过 2.5 cm），成功率也没升。甩动不是修正跟不上速度，而是某些局里一次性的横向事件，与下探快慢无关。负结果，不采用。

## 2026-09-27 壁厚对照（用户提议）：同一箱子壁厚 10 → 20 / 30 mm，抓点移到壁中（`box_platform_scene --wall`，`grasp_points_for` 读 `box.wall`）

| 壁厚 | 成功 | 脱手 | 下探卡住 | 其他 |
|---|---|---|---|---|
| 10 mm（基线） | 28/60 = 0.47 | 27% | 12% | 站不直 5%、保持不稳 5% |
| 20 mm（`box_centre_wall20_20260927`） | 14/30 = 0.47 | **7%** | 30% | 保持不稳 7% |
| 30 mm（`box_centre_wall30_{a,b}_20260927`） | 27/60 = 0.45 | **10%** | 22% | 翻箱 8%、试提没抓住 7% |

- 厚壁确实把"抓住之后"的失败去掉了大半（脱手 27 → 7–10%，成功局保持结束时的附着误差 1.8 → 1.2–1.4 cm），和 ③ 附着反馈的效果同一个量级、同一类失败。
- 但下探卡住从 12% 升到 22–30%：壁厚 3 cm 时 9.5 cm 的夹爪留给落点的横向余量从 ±4.2 降到 ±3.2 cm，抓点也更靠里。总成功率因此不变。
- 结论：**抓持侧的失败（脱手、翻箱）是可以消掉的，用厚特征或 ③ 都行；消掉之后剩下的是下探落点这一层，它不随特征变，是 0.4–0.45 m 伸手距离上控制器把整对手送偏 3–5 cm 的问题。** 中心格的上限 ≈ 0.47 由这一层决定。

## 2026-09-27 刹车后准备手臂的"快速路径"（`visual.ready_after_brake quick`）：每局省 5.4 s，成功率不变

刹车后专家调用的是回合开始那套站立初始化（手臂归位、planner 重进、1.5 + 0.4 + 1.5 s 保持、朝向重锚、3 s 保持、READY 抬臂 2.5 s、保持 1.2 s），
从"arm preparation"到"reach"中位 13.3 s，占一局 29%。`quick` 只保留 POSE 流需要的朝向重锚 + 1 s、抬臂、保持和静止判定。

| 组 | 成功 | prep→reach | 一局时长 |
|---|---|---|---|
| full（基线 28/60） | 0.47 | 13.3 s | 35.9 s |
| quick（`box_centre_quickready_{a,b}_20260927`） | 26/60 = 0.43 | **7.9 s** | 29.2–30.6 s |

区间重叠（0.32–0.56 vs 0.35–0.59），失败构成又是卡住 ↑ 脱手 ↓ 的那种互换（30%/13% vs 12%/27%）——这个互换在 ①、③、壁厚、quick 四次对照里都出现，
说明有一批边缘抓取局在"卡住"和"滑脱"之间随微小时序差摇摆，两者应合起来看。建议 warehouse/扫参配置改用 quick；dataclass 默认仍是 full。

## 2026-09-27 再压准备手臂（`ready_blend_seconds 1.5`、`ready_hold_seconds 0.6`、`reanchor_hold_seconds 0.5`，在 quick 之上）

| 组 | 成功 | prep→reach | 一局 | 卡住 / 脱手 |
|---|---|---|---|---|
| full（基线） | 28/60 = 0.47 | 13.3 s | 35.9 s | 12% / 27% |
| quick | 26/60 = 0.43 | 7.9 s | 30 s | 30% / 13% |
| quick + 压缩（`box_centre_quickfast_{a,b}_20260927`） | 23/60 = 0.38 | **5.8 s** | 27.7 s | 35% / 20% |

区间仍重叠（0.27–0.51 vs 0.35–0.59），但三档单调下降、卡住比例单调上升；准备阶段没有一局不稳。9 个百分点的差要每组约 200 局才能定。
判定：quick（7.9 s）可用；再压到 5.8 s 大概要付几个点，不建议默认，三个时长旋钮保留（默认 2.5 / 1.2 / 1.0）。

## 2026-09-27 安全范围内联合随机 + quick（第 3 步验证）：`box_range_quick_*_20260927`，4 种几何 × 15 局，seed 60000+

起点 0.6–2.0 m、朝向 ±20°、横移 −8～0 cm 每局同时随机，箱子贴台边、偏角 0，`ready_after_brake quick`。

| 几何（H_L，W0.45） | 成功 | 失败构成 |
|---|---|---|
| H045_L020 | 6/15 | lost_attachment 4, basket_tilt 3, reach_stalled 2 |
| H045_L032 | 5/15 | lost_attachment 5, basket_tilt 3, empty_gripper 1 |
| H060_L020 | 3/15 | reach_stalled 5, lost_attachment 3, unstable_hold 2 |
| H060_L032 | 10/15 | reach_stalled 3, lost_attachment 2 |

合计 **24/60 = 0.40**（95% 区间 0.29–0.53），与逐维扫参时各维单独的水平一致，维度叠加没有再掉。
H0.60 × L0.20 这一格 3/15 偏低（第 1 步是 13/30），15 局的散布之内。成功局视频：`outputs/videos/box_range_quick_20260927.zip`。

## 2026-09-27 warehouse 200 集 checkpoint-5000：开环 + 闭环 10 局（Isaac 镜像），闭环评测器先修四处

运行 `box_room_grasp_200_cap120_20260925`，训练停在 5000 步（`summary.json` 的 train 条目手补）。开环（官方脚本，val 10 集，Isaac 渲的图）
平均 MSE 0.00123 / MAE 0.0217；预测均值基线 0.0350 / 0.140，块首真值保持 16 帧 0.00179 / 0.0206；两手夹爪都在对的时刻合到 −0.06。

闭环前 `vla_rounds.py` 在物流房间跑不对，四处都修了：物体名写死 `blue_crate`（warehouse 是 `nor_cal_box`，reset 静默跳过、第一局即 KeyError）；
摆放是合成桌子的副本（机器人偏航 0、物体 π/2、离台边 U(1.2, 1.5)），改为直接调 `expert.start_placement`（训练 205 局全在 (0, −1.455)、朝 +y、离台边 0.90 m）；
`eval.initial_token` 用的是 crate 的（肩 yaw 差 0.56/0.59 rad，token L2 1.54，数据首帧内离散 0.077），新文件
`configs/deployment/warehouse_initial_token.json`（194 集首帧的 snapped 均值，恰等于 medoid）；夹爪下限 −0.02（单臂量程）改取 `visual.close_target` −0.06。

| 组 | 抬起判据成功 | 失败 | 成功后再持 10 s 不掉 | 结尾倾角 > 15° |
|---|---|---|---|---|
| checkpoint-5000，seed 6000–6009 | **9/10**（Wilson 0.60–0.98） | 1 `close`：合爪后箱子被扯歪滑走 25 cm | 9/9 | 3/9（16.7°、18.0°、23.2°） |
| 专家采集（同一 run，950 局） | 走到后退阶段 242/950 = 0.25 | reach_stalled 0.39、lost_attachment 0.17、basket_tilt 0.11 | — | — |

首次合爪 27–47 s、抬起 30–51 s（训练集平均一集 32.7 s）；观测 → chunk 延迟 p50 166 ms（Isaac 渲染 p50 95 ms），跳 4 帧补偿；图像对应状态年龄 p50 11 ms。
"后退 0.3 m"一列 9/9 都是 timeout：训练数据按 `--trim-verification-tail` 在 hold 开始处裁掉，后退段不在数据里，这一列对本 checkpoint 不可测。
两者口径不同（策略只判抬起 + 持 2 s；专家还要直立持 5 s + 后退），10 局也不足以下结论（同一份代码 10 局之间曾差出 2/10 到 7/10），100 局待跑。
录像 `outputs/videos/warehouse_ckpt5000_closed_loop10_isaac_20260927.zip`（Isaac 第三视角 + Isaac ego/双腕小窗，与镜像同一 stage 和曝光，`render_isaac_videos.sh --traces eval/state_trace --insets`（当时叫 render_isaac_warehouse.sh）从每局 qpos 轨迹事后渲，一局 2–3.5 min）；早先的 MuJoCo 版 `..._closed_loop10_20260927.zip` 留作对照。

## 2026-09-27 评测时间模型：墙钟 vs lockstep（checkpoint-5000，Isaac 图，seed 6000–6099，各 100 局）

同一 checkpoint（`box_room_grasp_200_cap120_20260925` 的 5000 步），四个只评测的运行目录，只改 `eval.client` / `gap` / 延迟。
计划、实现和验收在 [lockstep + Isaac 评测计划](../handoff/lockstep-isaac-eval-plan-20260927.md) 第 10 节；证据 `outputs/lockstep_eval_20260927/`。

| 臂 | 时间模型 | 成功（Wilson 95%） | 失败 | 抬起后掉落 |
|---|---|---|---|---|
| `wc_hold100_20260927` | 墙钟，等图 + 推理 ~166 ms 期间参考定住，新块跳 4 帧接入（之前的循环） | **81/100**（0.72–0.87） | close 11、reach 8 | 12 |
| `ls_w0_replica100_20260927` | lockstep 复刻上一行（按墙钟跳帧分布定住、跳帧） | **83/100**（0.74–0.89） | close 11、reach 5、walk 1 | 13 |
| `ls_w3_target100_20260927` | lockstep 目标模型：拍照渲染不占世界时间，推理 2 帧里续走旧块 | **88/100**（0.80–0.93） | close 7、reach 5 | 15 |
| `wc_continue100_20260927` | 墙钟，间隙里续发旧块（`gap: continue`） | **83/100**（0.74–0.89） | close 11、reach 6 | 15 |

- **lockstep 校准通过**：墙钟 81 vs 复刻 83，同 seed 不一致 14 : 16，McNemar p = 0.86，失败构成逐项差 ≤ 3；到箱最近距离中位 0.344 vs 0.337 m。
  2026-09-02 单臂的未过（46 vs 32）与"停得远"签名在这里都没出现。
- **目标模型 vs 现状**：88 vs 81，不一致 11 : 18，p = 0.27，100 局判不动；少掉的是 close 和 reach。
- 同一 seed 的 lockstep 局不逐位复现：MuJoCo 默认 4× 多重采样让同一状态两次渲染差 1–2 灰度级（Isaac 图本来也不确定），策略输出因此差一个量化步后分叉。
- 墙钟下的补帧（`gap: continue`）83 vs 81，不一致 12 : 14，p = 0.85：续发线程抢 GIL，观测 + 推理从 70 ms 升到 95 ms，新块一半从第 5 帧接入，抵掉了参考连续的好处。


## 2026-09-28 四边线 pilot：warehouse 房间里从平台四条边走入，按 envelope ≥ 30% 的范围随机

**做了什么**:`warehouse_box_45x*` 场景(房间 + 按尺寸重建的纸箱,每条边测了空地),专家 = `box_platform_default_20260927`
(quick,判据抬起 + 直立 + 保持 1 s、不后退)。每局抽一条边(`visual.start_edges`),箱子搬到该边、沿边随机、setback 0–2.5 cm、
偏角 0～+3.5°;起点 0.6 m 到"该边空地 − 0.25 m"、朝向 ±20°、横移 −8～0 cm。7 份 pilot 各 10 局,不录制,GPU0 两条流水线并行
(旁边有 JeVLA 训练约 4.7 核),每局 RTF 中位 0.99–1.00、最低 0.989。实验文件 `configs/experiments/warehouse_edges_pilot_*_20260928.yaml`,
逐局 `outputs/runs/warehouse_edges_pilot_*_20260928/`,汇总脚本与表 `outputs/experiments/warehouse_edges_20260928/pilot_report.{py,md,json}`,
录像预览 `outputs/videos/warehouse_edges_pilots_20260928.zip`(70 段,均 ≤ 499,156 B)。

| pilot | 成功 | 失败原因 | 停位深度中位(m) |
|---|---|---|---|
| 边 xm(45x32x33) | 2/10 | lost_attachment 4、basket_tilt 2、stand_not_delivered 1、reach_stalled 1 | 0.481 |
| 边 xp(45x32x33) | 4/10 | lost_attachment 4、reach_stalled 2 | 0.431 |
| 边 ym(45x32x33) | 6/10 | reach_stalled 3、lost_attachment 1 | 0.475 |
| 边 yp(45x32x33) | 4/10 | basket_tilt 2、reach_stalled 2、lost_attachment 2 | 0.481 |
| 45x20x33(四边) | 3/10 | reach_stalled 3、lost_attachment 3、basket_tilt 1 | 0.404 |
| 45x32x18(四边) | 4/10 | reach_stalled 2、basket_tilt 2、lost_attachment 1、empty_gripper 1 | 0.453 |
| 45x20x18(四边) | 6/10 | lost_attachment 3、basket_tilt 1 | 0.445 |

合计 **29/70 = 0.41**;按边合并:xm 6/15、xp 5/13、ym 11/19、yp 7/23。每格 10 局的 95% 区间约 ±30 个百分点,
没有哪条边或哪个尺寸是 0–1/10;彼此高低 10 局分不出来。
停位深度按 envelope §3.4 分档:0.39–0.42 为 5/7、0.42–0.45 为 8/15、0.45–0.48 为 6/13、0.48–0.52 为 7/22(平台上 0.08)、≥ 0.52 为 0/3。
房间里的停位整体偏深(22/70 落在 0.48–0.52),但那一档没有像平台上那样掉光。
`45x32x18` 第一次起栈被资源闸门拒绝(`cpu: 13.55 cores in use + 5 needed`),等另一条流水线跑完后单独重跑。

## 2026-09-28 刹车后准备手臂再压缩：`ready_after_brake compact`，以及去掉刹车后的朝向重锚

**起因**：用户觉得"站定到抓取"和"手抬起到开始伸手"之间停得太久。28 条四边线成功局：站定→合爪中位 15.6 s，其中准备手臂 8.7 s（8.5–9.1），
轨迹上手先在走路高度 0.72 m 停 3.9 s（朝向重锚的 POSE 切换 + 1 s 等待 + `PlannerArm.start` 的 0.6 s 静止），抬到 0.90 m 后又停 2.4 s
（`ready_hold` 1.2 s、空操作后的 `sleep(0.3)`、静止判定、交接 POSE 流）。

**改动**：`visual.ready_after_brake: compact` = quick 去掉两个对刹车后无用的等待（planner arm 直接以测得的走路手臂为参考接管，
不再静止 0.6 s；空操作 placement 后不再 sleep 0.3 s），抬臂 1.5 s、保持 0、重锚后等待 0.3 s；`visual.reanchor_after_brake: false`
再去掉刹车后的重锚（回合开始时已重锚过，guard 走法全程在 planner 里）。

**A/B**：45x32x33、四边随机、40 局、三组同种子（同一局同一摆放，可配对），不录制，GPU0 两条并行，RTF ≈ 1。
`configs/experiments/warehouse_edges_prep_{quick,compact,compact_noreanchor}_20260928.yaml`，汇总 `outputs/experiments/warehouse_edges_20260928/prep_ab_report.{py,md}`。

| 组 | 成功 | 准备手臂（prep→reach 中位） | 站定→合爪（成功局中位） | 一局（成功局） | 卡住 / 脱手 |
|---|---|---|---|---|---|
| quick（现行） | 19/40 | 8.68 s | 15.5 s | 28.8 s | 7 / 7 |
| compact | 17/40 | **4.85 s** | 12.4 s | 24.9 s | 9 / 9 |
| compact + 不重锚 | 14/40 | **2.66 s** | 9.3 s | 23.7 s | 13 / 10 |

配对（两组都成功 / 仅 quick / 仅新组 / 都失败）：compact 7 / 12 / 10 / 11，McNemar p = 0.83；不重锚 5 / 14 / 9 / 12，p = 0.40。
同一摆放两次跑的结果本身就一半不一致，40 局分不出 5 个百分点。不重锚没有让身体漂或转：站定→合爪骨盆位移中位 11.9 cm、偏航 1.0°，与 quick 的 12.1 cm、2.3° 同量级；
但"下探卡住"从 7 升到 13，和 09-27 "压得越狠卡住越多"的方向一致。
**判定**：compact 省 3.8 s、每局省 3.9 s，成功率看不出代价，可以用于采集；不重锚再省 2.2 s，趋势变差，不采用，要定需每组约 200 局。
"边停边抬手"（刹车最后一步就开始抬）没做：要改走近循环，且 09-23 走路时抬手 6/10 局手指扎进箱子。
配对录像 `outputs/videos/warehouse_edges_prep_ab_20260928.zip`（7 个摆放 × 3 组，21 段，每段 ≤ 497,068 B）。

## skills 目录的 approach → ready → grasp_bimanual 经 plan 入口在真栈上跑：1/3（2026-09-28）

入口 `scripts/plan_episode.py`，实验 `box_centre_base_20260927.yaml`（场景 `W0.45_L0.32_H0.6`，render offline → 占位相机），
profile `warehouse_box`（guard 刹车 / lifted 抬手 / staged 路径 / contact 触发），GPU1 / instance 1，seed 300–302，不录制。
每局：`start_placement` + `stand_up` → approach（`attempt.walk_step` 循环 + 守线刹车）→ ready（`prepare_stationary_ready`
after_brake）→ grasp_bimanual（`expert.handoff` + `attempt.reach_step` 循环 + `evaluate`）→ 成功后 `hand_back`。

| 局 | 结果 | approach | ready | grasp_bimanual |
|---|---|---|---|---|
| 0 | OK（hold 通过，hand_back 0 s） | 11 s | 7 s | 16 s |
| 1 | FAIL `[reach] basket_tilt` | 8 s | 8 s | 13 s |
| 2 | FAIL `[reach] reach_stalled` | 8 s | 8 s | 10 s |

- 两个失败族都是这条线已有的（2026-09-24 基线 6/20 里 stall 8/20）；3 局分不出与采集入口 0.30 的差别，只证明四段拼接在真栈上走得通，
  每段的通道交接（SWING → FROZEN → POSE 流 → FROZEN）与 executor 的不变量检查一致。
- 逐 primitive 记录、ghost 轨迹在 `outputs/runs/skills_smoke_warehouse_20260928/skills/round_NN/`。

## skills 目录第二批：warehouse plan 10 局对照、`put_down_bimanual` 首个实现、持箱 `carry_to`（2026-09-28 晚）

TODO 7.8 / 7.11。入口 `scripts/plan_episode.py`，实验 `box_centre_base_20260927.yaml`（场景 `W0.45_L0.32_H0.6`），profile `warehouse_box`，
GPU1 / instance 1，seed 7000–7009（put_down 第二轮 7010–7019）。

| 运行 | plan | 到最后一步的局 | OK | 失败 |
|---|---|---|---|---|
| `skills_warehouse_pilot10_20260928` | approach → ready → grasp_bimanual | 10 | **4/10** | `lost_attachment` ×4、`reach_stalled`、`stand_not_delivered` |
| `skills_warehouse_putdown10_20260928` | … → **put_down_bimanual** | 6（4 局 reach 失败） | put_down 自身 **5/6**；plan 判据 0/10（判据错，见下） | `[put_down] load dropped while lowering` ×1 |
| `skills_warehouse_putdown10b_20260928` | 同上，判据修后 | 1（seed 7010–7019 这批 reach 9/10 失败：`reach_stalled` ×4、`lost_attachment` ×4、`unstable_hold`） | put_down 自身 1/1（z 误差 0.4 cm、倾角 1.6°、双手离开）；plan 判据 0/1 | `[put_on] centre xy [0.03, −0.61] is 0.05 m outside the surface footprint`：站直阶段把箱子拉向自己约 0.2 m，原地放下时箱心已越过台面近沿 5 cm（箱子半宽 0.16，近半悬空），只是搁在沿上 |
| `skills_warehouse_putdown_fwd20_20260928`（seed 7040–7059，**前送版**：下放前工具点前移到箱子近面越过台面近沿 5 cm，自动算 0.12–0.18 m） | … → put_down_bimanual | 3（这批 reach 17/20 失败） | put_down **1/3**，plan 判据 1/3 | 前送完成后一开始下放就 `load dropped`（箱子倾 8°）×2：伸臂前送把捏持松了 |
| `skills_warehouse_putdown_legs20_20260928`（seed 7000–7019，**腿前送版**：hand_back → `carry_to(relative [+0.15, 0], turn_first=false)` → take_over → put_down(forward_m 0)） | 见左 | 11（reach 9/20 失败） | carry **3/11**（走对 0.12–0.14 m 的 3 局），put_down 见下 | `walk_to` 走不了 0.15 m 的小步：8 局冲出 0.67–1.53 m，箱子撞台面 `load dropped`（v1.1 小位移量化） |
| `skills_warehouse_putdown_nudge20_20260928`（seed 7000–7019，**脉冲挪步版**：hand_back → `nudge(relative [+0.15, 0], pulse 0.5 s, tol 0.04, ≤ 3 脉冲)` → take_over → put_down） | 见左 | 9（reach 11/20 失败） | nudge **4/9 进 4 cm**（其余 4.4–8.5 cm 差），0 冲出、0 掉箱、朝向漂移 ≤ 0.07 rad；随后 put_down 4/4 把箱子放稳在台面上（z 误差 ≤ 0.3 cm、倾 ≤ 0.8°），但 3/4 被"手仍触箱"判 FAIL | 0.5 s 脉冲量化：单脉冲 0–12 cm（第一脉冲多半不动） |
| `skills_warehouse_putdown_nudge07_20_20260928`（seed 7020–7039，`nudge(pulse 0.7 s, tol 0.06, ≤ 4 脉冲)`） | 同上 | 11（reach 9/20 失败） | nudge **6/11 进 6 cm**；随后 put_down **6/6 放稳**（z 误差 ≤ 0.1 cm、倾 ≤ 0.1°；4/6 松手后手指仍贴壁，已不判失败） | 0.7 s 脉冲会触发整步：3 局单脉冲走 21 / 54 / 77 cm（掉箱 ×2、摔倒 ×1）、1 局脉冲量化停在 7 cm |
| `skills_warehouse_putdown_nudge05x6_20_20260928`（seed 7040–7059，`nudge(pulse 0.5 s, tol 0.05, ≤ 6 脉冲)`，现默认） | 同上 | 9（reach 11/20 失败） | nudge **3/9 进 5 cm**（5 局停在 5–10 cm：残差 < 12 cm 后的 0.3 s 短脉冲不动）；put_down 2/3 放稳（1 局下放时掉） | 0 冲出、0 摔；1 局到位时接触已丢 |
| `skills_box_fwd50_putdown20_20260928`（seed 7000–7019，**7.8.1.1 走 B**：`stand_forward_min` 0.35 → 0.50，箱子站直时留在台面上方；plan approach → ready → grasp_bimanual → put_down(forward_m 0)，不挪步） | 见左 | 8（reach 12/20 失败，抓取率与基线 4/10 相同） | put_down **8/8** 放稳（z 误差 0、倾 0°、松手后无接触），plan 判据 **8/8**（箱子在台面上） | — |
| `skills_warehouse_carry10_20260928` | … → hand_back → **carry_to(relative [−1, 0])**（先 slow_walk 原地转 180° 再走 1 m） | 6（4 局 reach 失败） | carry **3/6** | `load lost while turning` ×1、`load dropped`（走 0.55 / 0.87 m 后箱子滑到 0.10 / 0.20 m）×2 |
| `skills_warehouse_carryback_noturn10_20260928` | 同上但 `turn_first=false`（朝向不变倒退 1 m，即 retreat 的走法） | 2（这批 reach 8/10 失败） | carry **2/2**（倒退 0.95 / 1.04 m，箱高 0.25 → 0.19 / 0.27 → 0.23 m，守卫底线 0.16） | — |
| `skills_warehouse_carryback_noturn20_20260928`（seed 7020–7039） | 同上 | 10（10 局 reach 失败） | carry **6/10** | 箱子在 1 m 倒退中匀速下滑 3–9.5 cm（≥ 7.6 cm 的 4 局触到守卫底线：`not held at arrival` ×2、`load dropped` ×2） |

- **对照采集入口**（同 yaml 的采集基线 6/20 = 0.30；扫参台账里 quick 版 26/60）：plan 入口 4/10 在同一噪声带里，通道交接不改成功率（10 局 pilot 分不出 ±2 局，见 2026-09-07 规则）。
- **put_down 本身可用**：5/6 把箱子放回台面（终态高度误差 ≤ 0.7 cm，倾角 ≤ 2.5°，双手离开），每次 8–10 s；下放过程箱子最大倾角 5–7.5°。
  失败的 1 局在开始下放前双手接触就丢了（reach 交付的 hold 不稳）。第一轮 plan 判据全 FAIL 是判据的错：`put <obj> on <surface>` 用物体原点 z 与表面顶比，
  而纸箱原点在底面（`box.centre` z 0.3、半高 0.3），已改成看箱底（`scene_model.py`，容差 2 cm），第二轮见上表。
- **put_down 要先把箱子送回台面上方再放**：hold 结束时箱子被站直阶段拉近约 0.2 m，原地下放会搁在近沿（第二轮唯一样本）。第三轮加了前送（`forward_m`，默认按台面近沿 + 5 cm 自动算，0.05 m/s，解算残差 > 3 cm 即停）：1/3 放到台面上并过判据，2/3 在前送完成、开始下放的瞬间掉箱（倾 8°）——伸臂前送让捏持松开。腿前送版（第四轮）走对 0.12–0.14 m 的 3 局里 put_down 1/3（1 局放稳过判据；1 局放稳、倾 0.6°，但一只手仍贴着箱子；1 局下放时掉）；其余 8 局 `walk_to` 把 0.15 m 走成 0.7–1.5 m、撞台面掉箱。**结论**：放下本身可用，差的是"把箱子送回台面上方"这一步——臂送会松捏持、腿送缺 ≤ 0.2 m 的小步走法（planner 小位移量化）。第五轮按用户定的 A 路做了 `nudge`（定时 SLOW_WALK 脉冲 + 刹停 + 量残差）：0.5 s 脉冲从不冲出、不掉箱，但单脉冲交付量化（0–12 cm），3 脉冲内 4/9 进 4 cm；进了的局 put_down **4/4 放稳**。
  "手仍触箱"不再判失败（0.6 m 高的箱子松手抬臂后手指仍贴壁，箱子已静止在台面），改记 `hand_contact_after_release`，`back_m` 默认 0.04 → 0.08。0.7 s 脉冲、容差 0.06 的对照：进的局 put_down 6/6 放稳，但长脉冲会触发整步（21–77 cm，掉箱 2、摔 1）。取中（0.5 s ×6、容差 0.05，现为 `nudge` 默认）：3/9 到位、0 冲出、put_down 2/3。三个变体合计：**nudge 到位 13/29，到位后 put_down 12/13 放稳**。
  **7.8.1 结论**：放下这一段已经通了（箱子在台面上方时 12/13 放稳）；差的"送回台面上方"卡在 v1.1 planner 的平移量化——短脉冲不动、长脉冲整步，脉冲时长调不出厘米级；要么由 planner 侧给小步（waypoint 输入那条线），要么抓取后不做站直回拉。走法保留为 `nudge`（安全、可测），不再调。
  **7.8.1.1 走 B 解决了它**：`stand_forward_min` 0.50 让站直阶段的手目标留在骨盆前 0.50 m，箱子不再被拉回台沿；之后不挪步直接放，8/8 放稳并过判据，抓取率不变。
  skills 的 `warehouse_box` profile 从此读 `skills_box_fwd50_20260928.yaml`（采集线 `box_centre_base` 自己的 0.35 不动）。
- **持箱 carry 比持瓶差**：180° 转 + 1 m 走 3/6（瓶子 4/4）；掉的两局箱子在走动中沿双手向下滑（高度 0.27 → 0.10 / 0.20 m，
  守卫底线 = 抓取高度 + 6 cm）。不转倒退合计 **8/12**，转 180° 再走 **3/6**，但两种走法里箱子都在走动中匀速下滑（倒退 1 m 掉 3–9.5 cm，约 2 cm/s；转的版本掉到 0.10–0.20 m）：
  瓶颈是冻结上身下的捏持蠕变（[[mujoco-grip-creep]] 的走动版），不是转向。**7.8.2 决定：默认走法不改**（`turn_first=True`，持瓶实测），持箱 1 m 内可用、更远要么中途重新抬举，要么走 POSE 流带反馈的搬运（未做）。

## 2026-10-01 单边 warehouse 基线复跑与 TODO 2 的单旋钮对照（`s2_*_20261001`）

用户 2026-10-01 定：在 `loco_agent` 分支、单边走入（09-24 基线 `box_room_grasp_baseline_20260924`，场景 `warehouse_textured`，离线渲染、不录制）上做 TODO 2 剩下的项；
每组复制基线只改一个旋钮（`s2_*_20261001.yaml` 文件头写了改的哪一行），20 局，seed 7000 起；多组并行时各占一个实例，实测 RTF 0.96–0.98。
逐局汇总：`scripts/experiments/warehouse_s2_report_20261001.py <run>...`（结果、站位深度、下落段跨壁送达、各阶段相机可见、手臂晃动）。

| 组 | 改了什么 | 成功 | 与对照的不一致（基线独成 : 本组独成） | 失败构成 |
|---|---|---|---|---|
| `s2_base`（对照） | — | **6/20** | — | 停滞 6、脱手 3、倾斜 3、后退掉 2 |
| `s2_base_r2` | seed 7100 | 4/20 | — | 停滞 11、脱手 4、倾斜 1 |
| `s2_adv015` | `start_advance` −0.25 → −0.15 | 15/20 | 1 : 10（p 0.012） | 停滞 3、脱手 1、后退掉 1 |
| `s2_adv015_r2` | 同上，seed 7100 | 6/20 | 4 : 6（p 0.75） | 脱手 7、倾斜 3、停滞 2 |
| `s2_adv035` | `start_advance` −0.35 | 7/20 | 2 : 3 | 停滞 6、脱手 5 |
| `s2_retreat025` | `retreat_speed` 0.35 → 0.25 | 2/20 | 5 : 1（p 0.22） | 停滞 8、脱手 6、后退掉 2 + 超时 1 |
| `s2_rewalk050` | `guard_rewalk_depth` 0 → 0.50（站深了就退 0.4 m 再走） | 7/20 | 4 : 5（p 1） | 脱手 7、停滞 4、倾斜 1、后退掉 1 |
| `s2_continuity005` | `reach_continuity_weight` 0 → 0.05 | 2/20 | 5 : 1（p 0.22） | 停滞 12、倾斜 6 |
| `s2_armposture02` | `reach_arm_posture_weight` 0 → 0.2 | 3/20 | 6 : 3（p 0.51） | 停滞 8、倾斜 4（`evaluation_tilt`）、脱手 3 |
| `s2_altstride` | `guard_stride_mode` floor → alternate | 2/20 | 6 : 2（p 0.29） | 停滞 10、脱手 5、倾斜 3 |
| `s2_transit02` | `reach_rotation_weight_transit` −1 → 0.2 | **11/20** | 2 : 7（p 0.18） | 脱手 4、停滞 4、倾斜 1 |
| `s2_transit02_r2` | 同上，seed 7100（对 `s2_base_r2`） | **8/20** | 2 : 6 | 停滞 7、脱手 2、倾斜 2、后退掉 1 |
| `s2_phasebrake`（10 局） | 0.55 走入 + 相位刹车 + 起点 1.03 m（新走法，非单旋钮） | 1/10 | — | 分散 |

- **基线可复现**：09-24 6/20，今天 6/20（另一组 seed 4/20）；两次都 0 摔（TODO 2.5 的 6/20 摔倒是 09-22 旧 carry 专家的）。
- **起点平移会移动站位的双峰**（2.4.10.23.4）：起点离平台 0.884 m 时，早刹（t < 4.8 s）的局站得深（中位 0.565–0.575 m）、晚刹的浅（0.465 m）；
  0.785 m（−0.15）时早刹占多数（12–13/20）而且变成浅的那一峰（0.453–0.487 m）；0.984 m（−0.35）全部晚刹（0.481 m）。
  是哪一次落脚触发刹车决定了站位，起点平移就换了一次落脚。−0.15 两组合并 21/40 vs 10/40（p 0.027），但复制组单看 6 : 4 不显著；
  效应来自固定起点和步长的相位锁定，四边随机起点下不存在，**不改默认**。
- **后退放慢是负结果**（2.4.10.29.2）：进入后退 5 局里掉 2、超时 1，基线 8 局里掉 2。
- **下落段跨壁漂移的来源**（2.4.10.19）：40 局下落段用专家自己的 FK 在骨盆系里比命令与实测，实测的手相对骨盆的跨壁移动跟着命令走
  （corr 0.80、斜率 0.48）——**SONIC 只送达约一半的跨壁手臂命令**；骨盆自己同时横漂 10–15 mm（中位），命令里 25–41 mm 的跨壁移动多半是在补它，
  送达一半后世界系里还剩 15–21 mm。停滞局下落 5.2 s、合爪局 1.7 s，漂得越久末端误差越大（21.8 vs 13.5 mm）。
- **站位能收紧但不涨成功**（2.4.10.23.2）：重走让 20/20 局 ≤ 0.50 m（9 局重走一次、2 局两次），失败从停滞（6 → 4）换成脱手（3 → 7）。
- **手臂晃动的两种压法都伤成功率**（2.4.10.20）：晃动是 align + descend 段肘和腕的缓慢此消彼长（肘参考 sd 15°、相关 −0.91），
  逐 tick 步长只有 0.85°。阻尼 0.05 只压逐 tick 部分（−20%），sd 不变；手臂姿态先验 0.2 把 sd 压到 8.2°，但都多了停滞。
- **放开去预抓点途中的掌姿态是唯一复制过的正结果**（2.4.10.13）：两组合并 19/40 vs 10/40，不一致 13 : 4，p 0.049；腕相机在 reach 段仍全程看得到箱子。
  用户定设为专家默认（dataclass 0.2）；今天的对照臂文件钉回 −1。
- **相位刹车**（3.14.1）只在助跑 ≥ 1 m 的稳态下比守线好；这条边前空地 1.285 m，见 `docs/walk.md`。
- **相机可见性**（2.4.10.13 的前置统计）：两只腕相机在 reach / align / descend / lift / stand / hold 全程 100% 看得到箱子（≥ 4 个角点），
  ego 在 descend 只有 64%（手和前臂挡住）；walk 段腕相机 0%（手臂下垂）。
- **v1.1 速度图的台阶是 agent 侧造成的**（TODO 2.6）：同一个栈上，旧二进制 4add301、撤回座高修复、桌面场景都还是连续图；
  只有 09-11 的 agent 代码（`aae3d72`）复现台阶（0.26 → 0.008–0.024 m/s，原地踏步）。按代码对比是 `dd36fd8` 之前每帧都带
  `vr_3pt_*` 字段、把 deploy 切进 VR-3PT 编码器模式。`--no-encoder` 的栈 SLOW_WALK 完全不动。`outputs/diagnostics/speedmap_v11_*_20261001`。

## 2026-10-07 warehouse mars 50 条采集（用户）：照 runbook 第 1–4 步，Unreal，50 fps

运行 `warehouse_mars_50`（实验文件 `configs/experiments/warehouse_mars_50.yaml`，即 `warehouse_mars_smoke` 只改运行名、`collect` 保留 50 条每批 10 局最多 25 批、`train` 和 `eval.rounds`），只走 ym 边，D455 第一视角，`images.camera_delay` 打开。

| 步 | 结果 | 用时 |
|---|---|---|
| 1 建场景 | 场景已存在，两条命令拒绝覆盖（runbook 已补这一情况）；另建一份比对，`scene_locomani.xml` 完全相同。Unreal 编辑器 GPU 1，关卡 35 s，`robot` 是 mars | 约 1 分钟 |
| 2 采集 | 1 条通道（进程数只够一条）。120 局保留 50 条（42%）；失败 reach_stalled 23、lost_attachment 23、basket_tilt 17、stand_not_delivered 3，其余 4 种各 1 | 75 分钟，每局约 37 s |
| 3 Unreal 补渲 | 第一次在第一条就报 `KeyError: 'dataset_fps'` 退出（`view_render.plan` 把运行的帧率加进 look 却没交给 `fill`，新 session 必崩，Isaac 入口同样；已修，加了测试）。重跑 50 条全部渲完，每条约 670–830 张图对 1150–1420 行，每张 p50 126 ms | 80 分钟，每条约 95 s |
| 4 转换 | 训练集 45 条 53,800 行、验证集 5 条 5,575 行，50 fps；两边审计各 32 项全过 | 9 分钟 |

- 采集的保留率 42% 高于 2026-10-03 mars 冒烟的 9/30（四边线），也只有这一次。
- 审计只查了 stock 和 2cam 两个档：`check_dataset.bimanual` 按任务文本里有没有 `crate` 判断双臂，warehouse 的任务是 `pick up the box`，所以真正训练用的 3cam_bimanual 档没被检查。手工核对过：modality 里 state 8 组 46 维、action 三组 66 维、三路相机都在。
- 运行目录 3.1 GB。各步日志在 `outputs/collect_logs_warehouse_mars_50/`。

