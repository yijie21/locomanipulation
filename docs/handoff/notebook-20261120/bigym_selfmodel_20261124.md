# bigym_selfmodel_20261124：BiGym 2.0 自带控制器的自身模型（照 jrefwm 的办法采数据、训练、当天在真实局上检验）

- 日期：2026-11-24（设计、预测、判据；正式采数据之前写下，单独提交）
- 机器 / GPU：4090。规划服务放 1–3 号卡（SIMPLE 的运行已跑完，这三张卡空着），BiGym 的仿真在 CPU 上，渲染在 0 号卡。
- 代码版本：`selfwm` @ `179341e4` 加未提交的 `scripts/experiments/bigym_20261124/`，正式跑时抄进输出目录的 `asrun/`，md5 记在结果里：
  - `bigym_collect.py`（2d36ba7b）：采样；
  - `bigym_data.py`（569bca95）：`build` 转成 `via_point.py` 的格式，`range`、`stats`、`loo` 四种预测的对比；
  - `spheres24.py`（901c3cc1）：24 个球的挑法；
  - `collect.sh`（bb3f4e81）：启动脚本；
  - `bigym_bench.py`（5a6b98a0）：录像多存骨盆位姿、速度和关节；
  - `curobo/plan_server.py`（272938da）：不改。
  - 训练和评测用 `scripts/experiments/selfmodel_via_20261120/via_point.py`、`via_model.py`，一行不改（挑出的每个球带着它对应的 SIMPLE 标签，`via_point.py` 原样读）。
- 配置：
  - BiGym 2.0（commit 35ec409）自带的控制器：GR00T-WBC 管腿和腰，手臂收绝对关节目标直接进 PD，不改参数。
  - 两个任务：`reach_target_multi_modal`、`drawer_top_open`。站位用跑分脚本的 `--stance`：看一眼以后走到目标前方、侧移到右肩前，抽屉再蹲到 0.62 m。
  - 不用仿真真值做决定，不用伺服和查表。
- 关联：
  - 用户 2026-11-24："就用他们自带的控制器，如果没有训练过自身模型的话，就参照 scalebfm 和 sonic 的方式把它的自身模型训练出来再继续"；
  - [selfmodel_jrefwm_20261124](selfmodel_jrefwm_20261124.md)：采法、转换、训练、度量都照它；
  - [selfmodel_s_20261124](selfmodel_s_20261124.md) 的复核：SIMPLE 上不用模型的固定偏差修正比模型还准，所以这里"常数"这个对手和模型一样重要；
  - TODO 38.11。

## 假设

BiGym 的手臂是 PD 直接跟关节目标，没有重力补偿，开环会下垂、过冲。11-24 的试跑里，站好以后伸手停下离目标 2–4.5 cm，而用关节读数做积分反馈只差 0.3–0.5 cm。下垂随手臂姿态变（重力力矩随姿态变），所以：

**用 BiGym 自带控制器执行 cuRobo 关节轨迹采出来的数据训练自身模型，它预测实际手臂的误差，能不能明显小于"手臂就在参考上"，并且明显小于"加一个固定偏移"的常数？**

背景（跑分脚本的开发种子试跑，720000–720009，各 10 局）：
- 底座不动时手臂够不到大多数目标：伸手 A 0 / 10、J 1 / 10，抽屉都是 0 / 10。所以站位用 `--stance`：看一眼以后走到目标前、侧移到右肩前，抽屉再蹲下。这和 SIMPLE 里先走到站位再伸手是同一个流程。
- 用 `--stance` 以后：伸手 A、J、关节反馈都是 10 / 10（5 cm 的容差，A 停下差 4.43 cm，到顶了）；抽屉 A 3 / 10、J 3 / 10、关节反馈 7 / 10。开环下垂 2–4.5 cm，关节反馈 0.3–0.5 cm。
- 风险：`--stance` 把机器人放到目标的固定相对位置，真实局里伸手的相对几何只剩高度（±10 cm）、手腕姿态和走路的误差在变。SIMPLE 的复核正是因为这个，常数比模型还准。所以采数据时站位加扰动，检验时常数用真实局上的留一平均，比模型更难赢。

## 做法

### 数据（约一万条）

1. **站位**：BiGym 两个任务的重置，种子从 800000 起（官方评测种子 620000–629999、开发种子 720000–720099 都不用）。每个站位照 `--stance` 走过去，只是走到的点在朝向坐标里前后、左右各加 ±6 cm 内的均匀扰动（和 jrefwm 一样，按站位编号定种子），抽屉的下蹲高度加 ±3 cm。站好以后手张开，停 0.3 s，记下发命令时的状态。采数据时走到哪儿可以用仿真里目标的真实位置（这是训练数据，不是评测），两个任务各占一半站位。
2. **发命令那一刻起，场景不再和机器人碰撞**（柜子、桌子、目标球的碰撞关掉，地面不关）。自身模型只描述身体。
3. **抽目标和虚拟箱子**，在发命令时的朝向坐标里，照 jrefwm：
   - 一半在本局任务目标附近（伸手：目标球心；抽屉：把手的预抓取点；每个坐标标准差 5 cm）；
   - 另一半在右肩前方一块：朝向坐标里前 0.15–0.45 m、左右 −0.40 到 0.00 m、比骨盆高 0.00–0.35 m（右肩在前 0.01、右 0.10、高 0.29 m）。这块是用逆运动学在每个任务 6 个站位、各 400 个点上探出来的：这块里能到的 70 %（伸手）和 57 %（抽屉），整个大范围里只有 12–13 %；左右只到骨盆中线，因为规划不看躯干；
   - 手腕姿态在本局姿态上加偏航 ±35°、俯仰和横滚各 ±25°；
   - 0、1、2 个虚拟小箱子（概率 30 / 45 / 25 %），只给 cuRobo 用，让路线有各种绕法，仿真里没有。
4. **规划**：cuRobo 规划右臂 7 个关节，用 BiGym 自己 MJCF 生成的 G1 加 Dex1 配置（试跑里 FK 和 BiGym 差 0.0004 mm 以内）。每个样本最多抽 20 次，规划不出来就跳过。
5. **执行**：和跑分脚本里的 J 完全一样：右臂照规划的 min-jerk 关节参考（`Motor.reach_joints` 的定时，0.15 m/s），其余关节和底座保持；走完等手停稳，再停 1 s。
6. **每个站位采两条**：第二条先沿第一条倒着走回去、停 1 s，再抽、再规划、再执行。
7. **数量**：每个任务 2 500 个站位（伸手是站位 0–2499，抽屉是 2500–4999），共约一万条。规划服务每张空闲的卡 5 个，每个任务 15 个仿真进程。
8. **采样时对环境的三处改动**（只改实例，控制器不变）：不渲染（`camera_keys=()`）；关掉任务的成功保持；关掉时限（设在 `env.bigym._time_limit` 上）。
9. 规划不看躯干和左臂（和 SIMPLE 的 `plan2.py` 一样）。试跑 80 条里 20 条右臂擦到躯干，最深 2.4 mm，多半是肩部偏航连杆；照实录下，不剔除。

### 转换和划分

- 照 `jref_data.py build`：参考就是手臂真正收到的关节目标序列；参考的球和实际的球用 BiGym 的模型算；24 个时刻加一个停稳点；在发命令时真实骨盆的朝向坐标里。
- 球：从 cuRobo 配置的右臂和夹爪的球里挑出和 SIMPLE 那 24 个对应的子集（肩、肘、腕各段，夹爪的掌心和两指），挑法写进结果。
- 划分：按站位种子轮流发到 10 个桶，0–7 训练，8 对半分（早停、校准半径），9 测试。自助法按站位重抽。

### 训练

`via_point.py train`，全用默认设置（5 个成员，512×4，dropout 0.1，权重衰减 1e-4，30000 步），然后 `dev`、`eval`。

### 当天在真实局上检验

1. 在 BiGym 的开发种子 720000–720019 上，两个任务各跑 J 带录像（20 局 × 2 个任务），转成回放数据。
2. `via_point.py eval`：模型、参考、常数三种在这些真实局上的误差；`jref_data.py range` 的同类检查：发命令时 7 个右臂关节超出训练数据 0.5–99.5 % 范围的个数。
3. 官方评测种子一个不跑。

- 停稳点取伸手结束后 0.5 s 的平均（`TAIL`）。抽屉的真实局伸手结束后 0.5 s 就往里推，刚好够；`build` 要求伸手时长加 0.5 s 以内没有下一步动作。
- `via_point.py dev` 里按"站着、手掌走超过 1 m""没走路就伸手"分的停止线子集在 BiGym 上是空的（站着的手掌路程 0.09–0.56 m，每个站位都走过），不用它们，只报总体。

## 度量

- **扫过区域误差**（`via_model` 的定义，cm），四种预测：
  - 参考：偏移 0；
  - 常数（训练）：训练集的平均偏移，`via_point.py` 本来就报；
  - 常数（真实局留一）：只在真实局上算，每局用同一任务其余 19 局的平均偏移（`loo_same_task`），不用模型。这是 SIMPLE 复核里赢了模型的那种对手，主判据用它；另报用两个任务其余 39 局的平均（`loo_all`）；
  - 模型。
- **降幅** = 1 − 模型中位数 ÷ 参考中位数；**对常数的降幅** = 1 − 模型中位数 ÷ 常数中位数；95 % 自助区间（测试桶按站位重抽，真实局按局重抽）。
- 手掌三个球的停稳点误差，单独报（这就是伸手停下差多远）。
- 采样统计：规划成功率、摔倒数、两条各自的误差。

## 预测（启动前写，首次启动后锁定）

- **采样**：规划成功率（每个样本）不低于 70 %；摔倒 0；一万条在 2 小时内采完。
- **测试桶（9 号）上**：
  - 参考的扫过误差中位数 2 到 5 cm（试跑里停下差 2–4.5 cm，过程中还有过冲和滞后）；
  - 模型对参考的降幅 0.5 到 0.85（PD 加重力的下垂是确定、随姿态平滑变化的，比 SIMPLE 的两个控制器好学；jrefwm 是 0.64、0.51）；
  - 常数（训练）对参考的降幅 0.1 到 0.5；模型对它的降幅 0.3 到 0.7（测试桶里一半目标在右肩前一大块，姿态变化大，常数不够）。
- **手掌停稳点**（测试桶）：参考 2 到 4.5 cm，模型 0.3 到 1.5 cm，常数（训练）1 到 3 cm。
- **真实局（开发种子 40 局，每个任务 20 局）上**：
  - 模型的扫过误差中位数不超过测试桶的 1.5 倍；超出训练范围的关节中位数 0；
  - 常数（真实局留一）对参考的降幅 0.3 到 0.7（站位把几何固定住了，SIMPLE 上就是这样）；
  - 模型对常数（真实局留一）的降幅 0 到 0.4，中间猜 0.15。我不确定模型能赢：真实局里只剩高度和手腕姿态在变。

## 判据（什么结果会让我改做法，下一步做什么）

- **模型对常数（真实局留一）的降幅不到 0.1，或区间包含 0**：模型在 BiGym 的真实局上也比不过常数。主表照样带 S，但同时带"固定常数修正"这一行，S 的贡献只能写成两者的差；并且报告用户：在站位固定相对几何的基准上，逐局预测没有用武之地，要证明它得换成伸手几何会变的局（复核的实验 (a)：站位抖动）。
- **模型在测试桶上赢常数（训练）0.3 以上，在真实局上却赢不了常数（真实局留一）**：模型本身学会了，是基准的局太像；记下来，按上一条做。
- **模型对参考的降幅不到 0.3**：模型没学会，先查数据（参考对不对得上实际收到的目标、停稳点取得对不对），再考虑加数据。
- **真实局的误差超过测试桶的 1.5 倍**：分布不一致（采样的站位、目标不像真实局），先比两边的起始关节和目标分布，再补采。
- **其余情况**：模型可用，写 S 在 BiGym 上的登记（开发种子上选检查方式，官方种子上测 A、J、固定常数修正、S）。

## 试跑（`# notebook: skip`，不进结果）

- 预测写在试跑之前，看到试跑之后没改。
- 每个任务 20 个站位、共 80 条：
  - 规划成功率每个样本 80 / 80，每次尝试 0.77；
  - 摔倒 0，提前结束 0，1.5 s 内没停稳 0；
  - 规划器起始手腕和 BiGym 差 0.000 mm。
- 参考对实际（中位数，括号里 90 % 分位）：扫过误差 4.63 cm（11.5），手掌停稳点 2.78 cm（9.65）；伸手任务 5.80、抽屉 3.52 cm。
- 真实局 4 局（每个任务 2 个开发种子）：扫过误差 4.38 cm，手掌停稳点 2.37 cm；7 个右臂关节超出训练范围 0 个。
- `via_point.py train` 少于 1000 步会崩（`via_model.fit` 每 500 步验证一次），所以试跑用了 1000 步、2 个成员，checkpoint 已删。
- 用时：规划是瓶颈，每次尝试约 2 s。一万条在 15 个规划服务下约 30 分钟，10 个约 45 分钟，5 个约 90 分钟。录像约 2.5 GB。

## 预测修订

**2026-11-25，采样跑完以后、看结果之前**（采集代码为 BiGym agent 所写，以下按它的报告记）：

- **`bigym_data.py` 改过一次**（登记时 569bca95，现在 3d936850，只多 5 行）：伸手任务的真实局 720003、720009 在走到站位的途中就满足了成功条件（`t_success` 5.28 s、4.56 s），伸手之前局就结束了，登记的 `build` 因此 IndexError。改法是跳过这种局、记为 `no_reach`。
  - 真实局因此是 38 局：伸手 18、抽屉 20。
  - `loo_same_task` 在伸手任务上是其余 17 局的平均，抽屉是其余 19 局；`loo_all` 是其余 37 局。
- **新写 `bigym_test.py`（12522e03）**：`via_point.py eval` 不报测试桶的手掌停稳点，也不报对常数的降幅，所以另写了这个脚本。误差、偏移、自助法都调用 `via_point` 的代码，"全部"一行和 `eval` 逐位相同（3.81 / 3.50 / 0.43，降幅 0.89，区间 0.88–0.89）。
- **有一条命令第一行没写 `# notebook:`**：跑 `bigym_test.py` 的那一条，这一行写在了命令中间的 `echo` 里。
- **规划服务只用了 3 号卡的 5 个**（1、2 号卡当时在跑 SIMPLE 的精度实验），采样用了 72 分钟。

## 结果（贴原始输出，不转述）

输出都在 4090 的 `outputs/bigym_20261124/`：采样和训练数据在 `selfmodel/`，模型在 `selfmodel/model/`，真实局在 `sm_real/`；代码 md5 在各目录的 `asrun/`。

### 采样（2026-11-24 15:57:52–17:10:30Z，3 号卡 5 个规划服务，30 个仿真进程）

`selfmodel/collect.log` 末行：`[17:10:30] DONE: 5000 stances, 0 failed processes`

`selfmodel/stats.txt`：

```
5000 stances, 10000 samples, planned 9999 (1.00 a sample), attempts 12838 (0.78 planned an attempt); falls 0, episodes ended early 0
  attempt statuses {'None': 9997, 'MotionGenStatus.IK_FAIL': 2819, 'MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS': 20, 'MotionGenStatus.GRAPH_FAIL': 2}; near goals planned 0.74, far 0.82; plan wall median 6.97 s (ok 7.26, failed 5.96)
  wall 114290 s over 9999 reaches: 11.43 s a reach in one process; stance wall median 20.3 s; plan attempts a planned sample 1.28; reach duration median 1.20 s; planner's start FK vs BiGym max 0.000 mm
  drawer_top_open: 2500 stances, planned 5000/5000, falls 0, walk 3.8 cm ahead 11.7 cm aside (medians)
    reach 1: 2500 planned, attempts a sample 1.23, duration median 1.25 s, settle median 0.20 s (unsettled 0); the arm against its own body: 503 reaches touch, 0 over 5 mm, worst 2.5 mm
      touching pairs (reaches): {'right_shoulder_yaw_link / torso_link': 459, 'right_elbow_link / torso_link': 103, 'right_wrist_pitch_link / torso_link': 42, 'right_wrist_roll_link / torso_link': 38, 'right_shoulder_roll_link / torso_link': 34, 'right_wrist_pitch_link / right_hip_roll_link': 5, 'right_dex1_finger_link_2 / left_dex1_finger_link_1': 4, 'right_wrist_yaw_link / torso_link': 4, 'right
    reach 2: 2500 planned, attempts a sample 1.25, duration median 1.32 s, settle median 0.18 s (unsettled 0); the arm against its own body: 475 reaches touch, 1 over 5 mm, worst 5.1 mm
      touching pairs (reaches): {'right_shoulder_yaw_link / torso_link': 421, 'right_elbow_link / torso_link': 78, 'right_wrist_pitch_link / torso_link': 41, 'right_wrist_roll_link / torso_link': 40, 'right_shoulder_roll_link / torso_link': 38, 'right_wrist_yaw_link / torso_link': 9, 'right_dex1_finger_link_2 / left_dex1_finger_link_1': 7, 'right_wrist_pitch_link / right_hip_roll_link': 6, 'right_
  reach_target_multi_modal: 2500 stances, planned 4999/5000, falls 0, walk 13.3 cm ahead 11.4 cm aside (medians)
    reach 1: 2500 planned, attempts a sample 1.35, duration median 1.08 s, settle median 0.18 s (unsettled 3); the arm against its own body: 890 reaches touch, 1 over 5 mm, worst 9.5 mm
      touching pairs (reaches): {'right_shoulder_yaw_link / torso_link': 847, 'right_elbow_link / torso_link': 166, 'right_wrist_roll_link / torso_link': 106, 'right_wrist_pitch_link / torso_link': 62, 'right_shoulder_roll_link / torso_link': 60, 'right_dex1_base_link / torso_link': 7, 'right_wrist_yaw_link / torso_link': 7, 'right_dex1_base_link / left_shoulder_pitch_link': 4, 'right_dex1_finger_
    reach 2: 2499 planned, attempts a sample 1.30, duration median 1.08 s, settle median 0.18 s (unsettled 3); the arm against its own body: 918 reaches touch, 1 over 5 mm, worst 12.3 mm
      touching pairs (reaches): {'right_shoulder_yaw_link / torso_link': 878, 'right_elbow_link / torso_link': 163, 'right_wrist_roll_link / torso_link': 92, 'right_wrist_pitch_link / torso_link': 60, 'right_shoulder_roll_link / torso_link': 59, 'right_wrist_yaw_link / torso_link': 5, 'right_dex1_base_link / torso_link': 4, 'right_dex1_finger_link_2 / left_dex1_finger_link_1': 3, 'right_dex1_finge
```

### 转换和划分

`selfmodel/build.txt`：

```
outputs/bigym_20261124/selfmodel/data: 9999 paths ({'film': 0, 'short': 0} skipped), tasks [4999, 5000], buckets [1000, 1000, 999, 1000, 1000, 1000, 1000, 1000, 1000, 1000]; reference-vs-actual: swept-region error median 3.81 cm (p90 9.77), palm 2.86 cm, palm settled point 2.77 cm (p90 8.89); our FK against the simulator's right_end_effector: max 0.000 mm
  reach: 4999 paths, swept median 3.88 cm (p90 11.74), palm settled median 2.88 cm, duration median 1.08 s
  reach, reach 1: 2500 paths, swept median 3.90 cm (p90 11.37), palm settled median 2.86 cm, duration median 1.08 s
  reach, reach 2: 2499 paths, swept median 3.87 cm (p90 12.10), palm settled median 2.91 cm, duration median 1.08 s
  drawer: 5000 paths, swept median 3.76 cm (p90 7.76), palm settled median 2.64 cm, duration median 1.29 s
  drawer, reach 1: 2500 paths, swept median 3.81 cm (p90 7.88), palm settled median 2.61 cm, duration median 1.25 s
  drawer, reach 2: 2500 paths, swept median 3.69 cm (p90 7.62), palm settled median 2.65 cm, duration median 1.32 s
```

`selfmodel/splits.txt`：

```
train (buckets 0-7): 7999 paths, 4000 stances, reach/drawer [3999, 4000]
bucket 8, early stopping (cal 0): 498 paths, 249 stances, reach/drawer [256, 242]
bucket 8, calibration (cal 1): 502 paths, 251 stances, reach/drawer [244, 258]
test (bucket 9): 1000 paths, 500 stances, reach/drawer [500, 500]
input (9999, 499) ref (9999, 24, 24, 3) off (9999, 25, 24, 3)
```

### 真实局（开发种子 720000–720019，J，`--stance`）

```
```

`sm_real/build.txt`：

```
outputs/bigym_20261124/sm_real/replay: 38 paths ({'film': 0, 'short': 0, 'no_reach': 2} skipped), tasks [18, 20], buckets [0, 0, 0, 0, 0, 0, 0, 0, 0, 38]; reference-vs-actual: swept-region error median 2.59 cm (p90 7.08), palm 1.95 cm, palm settled point 2.26 cm (p90 2.94); our FK against the simulator's right_end_effector: max 0.000 mm
  reach: 18 paths, swept median 4.82 cm (p90 7.96), palm settled median 2.58 cm, duration median 0.60 s
  reach, reach 1: 18 paths, swept median 4.82 cm (p90 7.96), palm settled median 2.58 cm, duration median 0.60 s
  drawer: 20 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.14 cm, duration median 0.70 s
  drawer, reach 1: 20 paths, swept median 2.57 cm (p90 2.58), palm settled median 2.14 cm, duration median 0.70 s
```

`sm_real/range.txt`：

```
joints at the command outside the training paths' 0.5-99.5 % band (38 reaches, 7999 training paths): right arm (7) median 0.0, mean 0.05, max 2, counts [37, 0, 1]; all 29 median 0.0, max 5
```

### 训练（默认设置，0 号卡）

`selfmodel/model/train.out`：

```
datasets ['outputs/bigym_20261124/selfmodel/data']; settings {'dropout': 0.1, 'wd': 0.0001, 'seed': 0, 'width': 512, 'depth': 4, 'members': 5, 'iters': 30000, 'dev': 'cuda'}
paths 9999: train 7999, validation for early stopping 498, for calibration 502, test 1000, sector hold-out 0
rows: train 199975, early-stopping validation 12450; input 336; validation squared offset 4.06 cm2 a coordinate with no model
member 0: validation squared offset 0.161 cm2 at iteration 5200 (first check 0.416, last 0.166; 42s)
member 1: validation squared offset 0.169 cm2 at iteration 12200 (first check 0.415, last 0.182; 82s)
member 2: validation squared offset 0.146 cm2 at iteration 21800 (first check 0.482, last 0.151; 123s)
member 3: validation squared offset 0.144 cm2 at iteration 23600 (first check 0.434, last 0.145; 168s)
member 4: validation squared offset 0.164 cm2 at iteration 24400 (first check 0.464, last 0.166; 217s)
ensemble's members: mean validation squared offset 0.157 cm2
direction: 7190 of 7999 training paths have an out-of-bag prediction
radius[reference] member 0: best validation loss 0.0873 at iteration 1000 (14s)
radius[reference] member 1: best validation loss 0.0792 at iteration 1000 (28s)
radius[reference] member 2: best validation loss 0.0788 at iteration 1000 (42s)
radius[reference]: scale 1.294; calibration paths' error median 3.69 cm, 95% level 13.65 cm
radius[constant] member 0: best validation loss 0.0880 at iteration 1000 (14s)
radius[constant] member 1: best validation loss 0.1013 at iteration 1000 (29s)
radius[constant] member 2: best validation loss 0.0915 at iteration 1500 (45s)
radius[constant]: scale 1.363; calibration paths' error median 3.45 cm, 95% level 12.32 cm
radius[direction] member 0: best validation loss 0.2132 at iteration 1000 (15s)
radius[direction] member 1: best validation loss 0.2054 at iteration 500 (30s)
radius[direction] member 2: best validation loss 0.2127 at iteration 500 (44s)
radius[direction]: scale 1.828; calibration paths' error median 0.45 cm, 95% level 4.03 cm
saved outputs/bigym_20261124/selfmodel/model/via_point.pt
```

### 开发集、测试桶、真实局

`selfmodel/model/dev.log`：

```
validation, all: 1000 paths
  reference  swept-region error: arm median  3.67 p90  9.42 cm | hand median  2.73 cm | margin (median radius)  4.67 cm, inside 0.95 | fixed radius 13.65 cm, inside 0.95
  constant   swept-region error: arm median  3.39 p90  8.38 cm | hand median  2.72 cm | margin (median radius)  4.61 cm, inside 0.96 | fixed radius 12.32 cm, inside 0.95
  direction  swept-region error: arm median  0.43 p90  1.84 cm | hand median  0.33 cm | margin (median radius)  0.81 cm, inside 0.94 | fixed radius  4.03 cm, inside 0.96
  drop of the arm's median against the reference: 0.88 (95 % interval 0.88 to 0.89, resampling routes); hand 0.88; constant: arm 0.07
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing: 500 paths
  reference  swept-region error: arm median  3.54 p90 11.18 cm | hand median  2.63 cm | margin (median radius)  4.50 cm, inside 0.95 | fixed radius 13.65 cm, inside 0.93
  constant   swept-region error: arm median  3.64 p90 10.28 cm | hand median  2.94 cm | margin (median radius)  5.04 cm, inside 0.94 | fixed radius 12.32 cm, inside 0.94
  direction  swept-region error: arm median  0.46 p90  2.18 cm | hand median  0.36 cm | margin (median radius)  0.88 cm, inside 0.93 | fixed radius  4.03 cm, inside 0.95
  drop of the arm's median against the reference: 0.87 (95 % interval 0.86 to 0.88, resampling routes); hand 0.86; constant: arm -0.03
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, palm path over 1.0 m (THE STOP LINE: drop of the arm's median >= 0.25): 0 paths, too few
validation, standing, palm path under 0.6 m: 495 paths
  reference  swept-region error: arm median  3.53 p90 11.12 cm | hand median  2.61 cm | margin (median radius)  4.50 cm, inside 0.95 | fixed radius 13.65 cm, inside 0.94
  constant   swept-region error: arm median  3.61 p90 10.17 cm | hand median  2.89 cm | margin (median radius)  5.01 cm, inside 0.94 | fixed radius 12.32 cm, inside 0.95
  direction  swept-region error: arm median  0.45 p90  2.17 cm | hand median  0.36 cm | margin (median radius)  0.86 cm, inside 0.93 | fixed radius  4.03 cm, inside 0.95
  drop of the arm's median against the reference: 0.87 (95 % interval 0.86 to 0.88, resampling routes); hand 0.86; constant: arm -0.02
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, no walk before the reach: 0 paths, too few
validation, crouching: 500 paths
  reference  swept-region error: arm median  3.78 p90  7.68 cm | hand median  2.87 cm | margin (median radius)  4.78 cm, inside 0.96 | fixed radius 13.65 cm, inside 0.97
  constant   swept-region error: arm median  3.10 p90  6.57 cm | hand median  2.52 cm | margin (median radius)  4.26 cm, inside 0.97 | fixed radius 12.32 cm, inside 0.96
  direction  swept-region error: arm median  0.41 p90  1.31 cm | hand median  0.30 cm | margin (median radius)  0.78 cm, inside 0.96 | fixed radius  4.03 cm, inside 0.98
  drop of the arm's median against the reference: 0.89 (95 % interval 0.88 to 0.90, resampling routes); hand 0.90; constant: arm 0.18
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
```

`selfmodel/model/eval.log`：

```
test, ordinary hold-out: 1000 paths
  reference  swept-region error: arm median  3.81 p90  9.37 cm | hand median  2.85 cm | margin (median radius)  4.90 cm, inside 0.96 | fixed radius 13.65 cm, inside 0.96
  constant   swept-region error: arm median  3.50 p90  8.49 cm | hand median  2.83 cm | margin (median radius)  4.78 cm, inside 0.96 | fixed radius 12.32 cm, inside 0.96
  direction  swept-region error: arm median  0.43 p90  1.69 cm | hand median  0.34 cm | margin (median radius)  0.84 cm, inside 0.95 | fixed radius  4.03 cm, inside 0.96
  drop of the arm's median against the reference: 0.89 (95 % interval 0.88 to 0.89, resampling routes); hand 0.88; constant: arm 0.08
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
test, sector hold-out: 0 paths, too few
test, ordinary hold-out, standing: 500 paths
  reference  swept-region error: arm median  3.89 p90 10.47 cm | hand median  2.92 cm | margin (median radius)  5.15 cm, inside 0.96 | fixed radius 13.65 cm, inside 0.93
  constant   swept-region error: arm median  3.92 p90  9.33 cm | hand median  3.15 cm | margin (median radius)  5.37 cm, inside 0.95 | fixed radius 12.32 cm, inside 0.94
  direction  swept-region error: arm median  0.46 p90  2.51 cm | hand median  0.39 cm | margin (median radius)  0.95 cm, inside 0.95 | fixed radius  4.03 cm, inside 0.94
  drop of the arm's median against the reference: 0.88 (95 % interval 0.87 to 0.89, resampling routes); hand 0.87; constant: arm -0.01
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 0.99 of the paths
test, ordinary hold-out, standing, palm path over 1.0 m: 1 paths, too few
real paths replayed, replay: 38 paths
  reference  swept-region error: arm median  2.59 p90  7.08 cm | hand median  1.95 cm | margin (median radius)  3.62 cm, inside 0.95 | fixed radius 13.65 cm, inside 1.00
  constant   swept-region error: arm median  2.23 p90  4.63 cm | hand median  2.01 cm | margin (median radius)  3.04 cm, inside 0.92 | fixed radius 12.32 cm, inside 1.00
  direction  swept-region error: arm median  0.26 p90  0.77 cm | hand median  0.22 cm | margin (median radius)  0.43 cm, inside 0.89 | fixed radius  4.03 cm, inside 1.00
  drop of the arm's median against the reference: 0.90 (95 % interval 0.85 to 0.91, resampling paths); hand 0.89; constant: arm 0.14
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
real paths replayed, replay, walked first: 38 paths
  reference  swept-region error: arm median  2.59 p90  7.08 cm | hand median  1.95 cm | margin (median radius)  3.62 cm, inside 0.95 | fixed radius 13.65 cm, inside 1.00
  constant   swept-region error: arm median  2.23 p90  4.63 cm | hand median  2.01 cm | margin (median radius)  3.04 cm, inside 0.92 | fixed radius 12.32 cm, inside 1.00
  direction  swept-region error: arm median  0.26 p90  0.77 cm | hand median  0.22 cm | margin (median radius)  0.43 cm, inside 0.89 | fixed radius  4.03 cm, inside 1.00
  drop of the arm's median against the reference: 0.90 (95 % interval 0.85 to 0.91, resampling paths); hand 0.89; constant: arm 0.14
  fixed radius: direction / reference 0.29 (set on the calibration paths); direction wins on 1.00 of the paths
```

`selfmodel/model/test_bucket9.txt`：

```
bucket 9, all: 1000 paths, 500 stances
  reference       swept-region error median  3.81 p90  9.37 cm | hand  2.85 cm | palm settled point median  2.76 p90  8.85 cm
  constant_train  swept-region error median  3.50 p90  8.49 cm | hand  2.83 cm | palm settled point median  2.67 p90  7.27 cm
  model           swept-region error median  0.43 p90  1.69 cm | hand  0.34 cm | palm settled point median  0.16 p90  1.06 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.89 (0.88 to 0.89); constant_train vs reference 0.08 (0.03 to 0.12); model vs constant_train 0.88 (0.87 to 0.88)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.94 (0.94 to 0.95); constant_train vs reference 0.03 (-0.02 to 0.08); model vs constant_train 0.94 (0.94 to 0.95)
bucket 9, reach: 500 paths, 250 stances
  reference       swept-region error median  3.89 p90 10.47 cm | hand  2.92 cm | palm settled point median  2.87 p90 10.62 cm
  constant_train  swept-region error median  3.92 p90  9.33 cm | hand  3.15 cm | palm settled point median  3.11 p90  9.08 cm
  model           swept-region error median  0.46 p90  2.51 cm | hand  0.39 cm | palm settled point median  0.19 p90  1.69 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.88 (0.87 to 0.89); constant_train vs reference -0.01 (-0.09 to 0.08); model vs constant_train 0.88 (0.87 to 0.89)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.93 (0.92 to 0.94); constant_train vs reference -0.08 (-0.15 to -0.02); model vs constant_train 0.94 (0.92 to 0.94)
bucket 9, drawer: 500 paths, 250 stances
  reference       swept-region error median  3.78 p90  7.89 cm | hand  2.80 cm | palm settled point median  2.65 p90  6.72 cm
  constant_train  swept-region error median  3.16 p90  6.71 cm | hand  2.57 cm | palm settled point median  2.20 p90  5.51 cm
  model           swept-region error median  0.41 p90  1.31 cm | hand  0.31 cm | palm settled point median  0.13 p90  0.66 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.89 (0.89 to 0.90); constant_train vs reference 0.16 (0.08 to 0.21); model vs constant_train 0.87 (0.86 to 0.88)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.95 (0.94 to 0.96); constant_train vs reference 0.17 (0.10 to 0.21); model vs constant_train 0.94 (0.93 to 0.95)
bucket 9, reach 1 of the stance: 500 paths, 500 stances
  reference       swept-region error median  3.90 p90  9.71 cm | hand  2.85 cm | palm settled point median  2.78 p90  9.44 cm
  constant_train  swept-region error median  3.63 p90  8.87 cm | hand  2.89 cm | palm settled point median  2.74 p90  8.05 cm
  model           swept-region error median  0.44 p90  1.99 cm | hand  0.35 cm | palm settled point median  0.15 p90  1.28 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.89 (0.88 to 0.89); constant_train vs reference 0.07 (-0.00 to 0.13); model vs constant_train 0.88 (0.87 to 0.89)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.95 (0.94 to 0.95); constant_train vs reference 0.01 (-0.05 to 0.08); model vs constant_train 0.95 (0.94 to 0.95)
bucket 9, reach 2 of the stance: 500 paths, 500 stances
  reference       swept-region error median  3.73 p90  8.98 cm | hand  2.87 cm | palm settled point median  2.76 p90  8.62 cm
  constant_train  swept-region error median  3.44 p90  7.96 cm | hand  2.80 cm | palm settled point median  2.61 p90  6.69 cm
  model           swept-region error median  0.43 p90  1.54 cm | hand  0.33 cm | palm settled point median  0.17 p90  0.93 cm
  drops of the swept-region error median (95 % interval, resampling stances): model vs reference 0.88 (0.88 to 0.89); constant_train vs reference 0.08 (0.03 to 0.15); model vs constant_train 0.87 (0.86 to 0.88)
  drops of the palm settled point median (95 % interval, resampling stances): model vs reference 0.94 (0.93 to 0.95); constant_train vs reference 0.06 (-0.02 to 0.10); model vs constant_train 0.93 (0.93 to 0.94)
```

`sm_real/loo.txt`：

```
replay, all: 38 episodes
  reference       swept-region error median  2.59 p90  7.08 cm | hand  1.95 cm | palm settled point median  2.26 cm
  loo_same_task   swept-region error median  0.92 p90  3.22 cm | hand  0.68 cm | palm settled point median  0.38 cm
  loo_all         swept-region error median  1.70 p90  3.87 cm | hand  1.28 cm | palm settled point median  0.48 cm
  constant_train  swept-region error median  2.23 p90  4.63 cm | hand  2.01 cm | palm settled point median  1.31 cm
  model           swept-region error median  0.26 p90  0.77 cm | hand  0.22 cm | palm settled point median  0.53 cm
  drops of the median (95 % interval, resampling episodes): loo_same_task vs reference 0.65 (0.54 to 0.72); model vs reference 0.90 (0.85 to 0.91); model vs loo_same_task 0.72 (0.59 to 0.77); model vs loo_all 0.85 (0.77 to 0.87); model vs constant_train 0.88 (0.82 to 0.90)
replay, reach: 18 episodes
  reference       swept-region error median  4.82 p90  7.96 cm | hand  3.87 cm | palm settled point median  2.58 cm
  loo_same_task   swept-region error median  1.97 p90  3.70 cm | hand  1.35 cm | palm settled point median  0.71 cm
  loo_all         swept-region error median  2.57 p90  4.47 cm | hand  1.67 cm | palm settled point median  0.60 cm
  constant_train  swept-region error median  3.30 p90  5.82 cm | hand  2.37 cm | palm settled point median  1.59 cm
  model           swept-region error median  0.52 p90  0.84 cm | hand  0.50 cm | palm settled point median  0.25 cm
  drops of the median (95 % interval, resampling episodes): loo_same_task vs reference 0.59 (0.27 to 0.76); model vs reference 0.89 (0.81 to 0.92); model vs loo_same_task 0.74 (0.55 to 0.82); model vs loo_all 0.80 (0.69 to 0.83); model vs constant_train 0.84 (0.73 to 0.87)
replay, drawer: 20 episodes
  reference       swept-region error median  2.57 p90  2.58 cm | hand  1.94 cm | palm settled point median  2.14 cm
  loo_same_task   swept-region error median  0.78 p90  1.04 cm | hand  0.54 cm | palm settled point median  0.28 cm
  loo_all         swept-region error median  1.50 p90  1.78 cm | hand  1.09 cm | palm settled point median  0.42 cm
  constant_train  swept-region error median  2.09 p90  2.33 cm | hand  1.92 cm | palm settled point median  1.14 cm
  model           swept-region error median  0.22 p90  0.24 cm | hand  0.17 cm | palm settled point median  0.59 cm
  drops of the median (95 % interval, resampling episodes): loo_same_task vs reference 0.70 (0.67 to 0.77); model vs reference 0.92 (0.91 to 0.92); model vs loo_same_task 0.72 (0.63 to 0.76); model vs loo_all 0.86 (0.83 to 0.87); model vs constant_train 0.90 (0.89 to 0.91)
```

### 附：抽屉真实局的"停稳点"其实没停稳（2026-11-25 加查，不是登记的度量）

`tmp/settle_chk.py`、`settle_chk2.py`，读录像里的 `ee_p`：

```
real drawer: seed, reach phase, next phase, gap s, median ee speed cm/s in window, window drift cm
  ('720000', 'reach', 'settle', 0.02, 1.96, 1.06)
  ...（20 局，速度 1.22–2.09 cm/s，窗口内漂移 0.69–1.15 cm）
training drawer reaches 120: 0.5 s window right after the reach vs the last 0.5 s before the next move: median 0.30 cm, p90 0.68; ee speed in the early window median 0.91 cm/s
```

真实的抽屉局伸手结束后 0.5 s 就往里推，取停稳点的 0.5 s 窗口里手还在以约 2 cm/s 移动、漂移约 1 cm。训练数据是停稳后再停 1 s 取的。

## 意外

- **模型比预测好得多。**
  - 测试桶：模型对参考的降幅 0.89（预测 0.5–0.85），对常数（训练）0.88（预测 0.3–0.7），手掌停稳点 0.16 cm（预测 0.3–1.5）。
  - 真实局：对常数（真实局留一）的降幅 0.72，区间 0.59–0.77。预测是 0–0.4、中间 0.15，偏得最多。
  - 我低估了 BiGym 的手臂：它是 PD 直接跟关节目标，没有重力补偿，误差是确定的、随姿态平滑变化的，比 SIMPLE 里两个学出来的全身控制器好学得多。
- **常数（训练）几乎没用**（对参考降幅 0.08，预测 0.1–0.5）：训练数据一半在右肩前一大块里，各处偏差方向不同，平均起来互相抵消。真实局留一的常数就强得多（0.65），因为真实局彼此很像。
- **终点上抽屉是常数赢**：手掌停稳点，抽屉模型 0.59 cm、留一常数 0.28 cm；伸手任务模型 0.25、常数 0.71；合起来模型 0.53、常数 0.38。
  - 加查发现，抽屉真实局取停稳点的窗口里手还在动（约 2 cm/s，漂移约 1 cm），训练数据的同一窗口只差 0.30 cm。
  - 模型预测的是真正停稳的点，常数是从同样没停稳的真实窗口里平均出来的，所以这一项对常数有利。
  - 这不是登记的主度量（主度量是扫过误差），但"一次命令的精度"最关心的就是终点，要先弄清楚。
- **两局伸手在走路途中就成功了**（目标落进 5 cm 内），没有伸手，只能跳过。
- **真实局的参考误差比训练小**（2.59 对 3.81 cm）：真实局路线短，伸手时长中位数 0.60–0.70 s，训练是 1.08–1.29 s。
- **右臂常擦到躯干**：伸手任务约 36 %、抽屉约 20 % 的伸手碰到 `torso_link`，最深 12.3 mm（1 条超过 5 mm 以外都在 5 mm 内）。原因是规划不看躯干，照实录下，没有剔除。
- **抽屉 J 在开发种子 720000–720019 上 6 / 20**（试跑 720000–720009 是 3 / 10）。

## 结论与下一步

- **对照判据**：哪一条都没触发。
  - 模型对真实局留一常数的降幅是 0.72（0.59–0.77），不是"不到 0.1"；
  - 对参考 0.90，不是"不到 0.3"；
  - 真实局误差 0.26 cm，比测试桶 0.43 cm 还小，不是"超过 1.5 倍"。

  所以落在"其余情况：模型可用，写 S 在 BiGym 上的登记"。
- **和 SIMPLE 的对比**：SIMPLE 上常数比模型准，BiGym 上模型在扫过误差上远好于最强的常数。说明"模型的逐局预测有没有用"取决于控制器的误差好不好学：BiGym 的 PD 手臂好学，SIMPLE 的两个学出来的全身控制器难学。
- **证据强度**：
  - 一次采样、一个模型（5 个成员）；
  - 测试桶 1 000 条（500 个站位）；
  - 真实局 38 局，都是开发种子；
  - 每个控制器、每个任务只有一种设置。
- **其他可能的解释**：
  - 扫过误差量的是实际手臂到预测折线的距离，对"形状预测对、沿路线有错位"比较宽容，可能放大了模型的优势。终点上抽屉是常数赢，见"意外"。
  - 真实局只有 17–19 局给常数取平均，常数本身有噪声。但留一是公平的做法，而且真实局彼此很像，对常数有利，不是不利。
- **四类陷阱**：
  - **benchmark 选择**：两个任务是 11-24 按规则读完 40 个任务后选的，没有按结果挑。伸手任务成败已到顶，所以 S 的成败要看抽屉。
  - **数据泄漏**：
    - 训练用的站位种子从 800000 起，真实局是 720000–720019，没有重叠；
    - 但两边是同样两个任务的同一类重置，真实局的站位和目标落在训练分布里（超出范围的关节中位数 0）。这是有意的：检验的是"同一个基准上用不用模型"，不是跨任务泛化。
  - **指标误用**：主度量照登记报；手掌停稳点这一项不利于模型，也照报，并查了原因。
  - **事后挑选**：只训练了一个模型，没有挑 checkpoint 或设置。
- **下一步**：
  1. **先弄清终点。** 抽屉真实局在伸手之后加 1 s 停稳再往里推（所有做法都一样），用开发种子重跑 20 局，比终点上模型和留一常数谁准。这一步只为度量对得上，不改控制器。
  2. **写 S 在 BiGym 上的登记。**
     - 做法：A、J、固定常数修正、S（按模型预测的终点偏差挪目标）、上限（按本局实测偏差挪）。
     - 开发种子上定细节，官方种子 620000+ 上测。
     - 成败看抽屉；伸手任务成败到顶，只报终点偏差。
  3. 独立复核（这条结论会决定往 BiGym 的 S 上投入）。

## 独立复核

2026-11-25，独立的 agent。它拿到的是登记的各节、预测修订、原始结果和结论（"意外"一节和推理过程没给），以及 4090 上的数据、模型、代码路径。原文（英文）：

> **Verdict:** the model's lead over the best model-free constant on the swept-region metric holds up, and holds against stronger constants than the author tried. The end-point claim is a split. The model wins on the reach task and loses on the drawer under the registered measurement. My offline tests support the author's "unsettled window" explanation for the drawer loss, but only a registered rerun can confirm it.
>
> ## Findings
>
> 1. **The numbers reproduce exactly and nothing leaked.** `loo.txt`, `test_bucket9.txt` and `eval.log` match to the printed digits; only bootstrap intervals differ by about 0.01 because of the RNG.
>    - My own dense-sampled swept-error code agrees with `VM.sweep_err` to within 0.001 mm.
>    - The leave-one-out constant is correct. It leaves out the held-out episode, takes the mean over the other episodes of the same task, and uses the same offset table (heading frame at the command) and the same error function as the model.
>    - The model file lists only `selfmodel/data` as training data. Its seeds (800000–804999) do not overlap the real seeds (720000–720019).
>    - Bucket 9 is in no ensemble member's bootstrap draw, buckets are separated by stance, normalisation statistics come from training rows only, and `const_right` equals the training mean.
>
> 2. **(a) The model beats every model-free constant I could build** on the swept metric (all 38 real episodes):
>
>    | Opponent | Its swept median (cm) | Model's drop against it |
>    |---|---|---|
>    | Leave-one-out, same task | 0.92 | 0.72 [0.59, 0.77] |
>    | In-sample constant (includes the held-out episode) | 0.88 | 0.70 [0.58, 0.76] |
>    | Leave-one-out median | 0.92 | 0.72 |
>    | Per-task training constant | 1.97 | 0.87 |
>    | Joint-space leave-one-out + Cartesian base residual | 0.87 | 0.70 [0.57, 0.75] |
>
>    - The model wins 36–38 of the 38 episodes in every comparison.
>    - Adding the measured offset at the command to the constant changes nothing. That offset is identical in every episode (sd 0.00 cm), so reading the start state is not where the model's advantage comes from.
>    - The constant's noise is not the issue either: even the constant that has seen the held-out episode is only 0.04 cm better than leave-one-out.
>
> 3. **The swept metric flatters the model on the drawer only.**
>    - On the reach task every metric agrees (model's drop against the constant 0.58–0.85).
>    - On the drawer, the swept drop is 0.72. The palm settled point goes the other way: −1.11 [−2.26, −0.58], with the model better on only 2 of 20 episodes. Over all spheres and moments, time-aligned, the drop is only 0.25 [−0.07, 0.34].
>    - The mechanism: the real tail ticks lie on the model's last polyline segment, from the reference end to the predicted settled point, within 0.03–0.12 cm. The swept metric therefore charges nothing for the predicted end point being too far along that segment.
>    - At the reference's end moment, where training and real data use the same definition, the model is far better: 0.05 vs 0.28 cm on the drawer.
>
> 4. **(b) The unsettled-window explanation is supported**, by exploratory checks done after seeing the results:
>    - On real drawer episodes the palm moves along the model's predicted drift direction. It covers 0.42–0.58 of that drift inside the 0.5 s window and is still moving at 1.37 cm/s at the last tick.
>    - On unseen bucket-9 drawer paths, scoring the model's settled prediction against the real episodes' early-window definition raises its error from 0.13 to 0.38 cm. The definitional mismatch alone costs about 0.3 cm.
>    - I took the share of the drift completed in that window from training data only (buckets 0–7, median 0.54) and applied it to the model's prediction. The drawer end point then becomes model 0.11 vs constant 0.28 cm (drop 0.62 [0.41, 0.75], model better on 18 of 20).
>    - The catch: that same fix raises the swept error from 0.22 to 0.53 cm, so the two registered measures disagree whenever the window is unsettled.
>    - Design implication: under the current drawer protocol the "settled point" is the wrong target for S. The relevant target is the hand position when the next move starts.
>
> 5. **Several registered predictions missed, all in the model's favour, and the conclusions don't say so.**
>    - Test bucket, model vs reference: 0.89 (predicted 0.5–0.85).
>    - Test bucket, model vs training constant: 0.88 (predicted 0.3–0.7).
>    - Test bucket, model's palm settled error: 0.16 cm (predicted 0.3–1.5).
>    - Real episodes, model vs leave-one-out constant: 0.72 (predicted 0–0.4).
>    - Training constant vs reference: 0.08 (predicted 0.1–0.5).
>    - These are plausible for a deterministic PD arm, but they should be stated as misses.
>
> 6. **The two code changes after registration are legitimate.**
>    - `bigym_data.py`: the only diff is the 4-line `no_reach` skip. The films for 720003 and 720009 end at `side_settle` with 0 reach ticks, and the patch cannot touch collection rows.
>    - `bigym_test.py`: it only reports quantities the registered predictions need, bootstraps by stance as registered, and its numbers reproduce exactly.
>
> 7. **The practical gain is small.** A model-free constant already removes 65–87 % of the error. The model adds about 0.4–1.5 cm on the swept metric and ±0.3–0.5 cm on the end point. The reach task's success rate is saturated, so S against a fixed-constant correction will be hard to separate on success rate.
>
> 8. **Context.** The real episodes are very narrow and nearly deterministic: the aim point varies by about 1 cm in x and y, and in height by 5 cm (reach) or 0.2 cm (drawer). Every result is for the author's own opt-in `--stance` protocol, not BiGym's default fixed-base protocol.
>
> ## Recomputed key numbers (cm, medians)
>
> | | reach swept | reach palm settled | drawer swept | drawer palm settled | all swept | all palm settled |
> |---|---|---|---|---|---|---|
> | Reference | 4.82 | 2.58 | 2.57 | 2.14 | 2.59 | 2.26 |
> | Leave-one-out, same task | 1.97 | 0.71 | 0.78 | 0.28 | 0.92 | 0.38 |
> | Model | 0.52 | 0.25 | 0.22 | 0.59 | 0.26 | 0.53 |
>
> - **Model vs leave-one-out (swept):** reach 0.74 [0.55, 0.82], drawer 0.72 [0.64, 0.76].
> - **Model vs leave-one-out (palm settled):** reach 0.64 [0.46, 0.71], all −0.39 [−0.92, 0.34].
> - **Model vs reference (swept):** 0.90 [0.85, 0.91].
> - **Test bucket:** reference / training constant / model swept 3.81 / 3.50 / 0.43; drops 0.89, 0.08 [0.03, 0.13] and 0.88. Palm settled 2.76 / 2.67 / 0.16.
> - **Real error below the test bucket:** this partly reflects easier paths. Bucket-9 reaches shorter than 0.9 s score 0.46 (reach) and 0.40 (drawer), against 0.52 and 0.22 on the real episodes. I see no sign of a distribution shift.
>
> ## Experiments that could overturn the conclusions
>
> - **(a) Frozen constant, new episodes.** Freeze per-task constants (Cartesian and joint-space) fitted on the 38 existing episodes. Run unused dev seeds 720020–720099 (80 per task). Register in advance that the claim fails if the model's drop against the constant is below 0.1 or its interval includes 0, on the swept metric or on the all-sphere time-aligned metric. A second variant jitters the stance at test time, the registration's own experiment (a). A k-nearest-neighbour lookup over the 10k training paths would check whether "learned self model" reduces to "data lookup".
> - **(b) Two tests of the drawer end point.**
>   1. Change the label, not the benchmark: compute from the existing training films the offset at the moment the next move starts (reach end + 0.5 s). Register, retrain, and compare with leave-one-out on the existing 20 drawer episodes. This needs no new simulation and keeps the benchmark protocol unchanged.
>   2. The author's rerun with a 1.5 s settle. Register comparisons at both the true settled point and at +0.5 s. The explanation fails if leave-one-out still matches or beats the model at the true settled point.
> - **(c) Official seeds, run once.** BiGym's evaluation block is reserved only by convention: `protocol.py` uses the same reset function for every seed, and the official run is 100 episodes from 620000. Run the frozen model and the frozen dev-fitted constant once on 620000–620099 for both tasks, with criteria registered first. Rehearse on the extra dev seeds from (a) beforehand. Since only 18–20 dev episodes per task exist now, the main risk is reset modes those episodes never sampled.
>
> ## Verdicts
>
> | Conclusion | Verdict |
> |---|---|
> | No criterion triggered, so the model is usable; write the S registration | Supports |
> | Model vs leave-one-out constant 0.72 (0.59–0.77), vs reference 0.90 | Supports (holds against stronger constants) |
> | Real error 0.26 < test 0.43, so no distribution shift | Supports, though the stated reason is weak |
> | On BiGym the model beats the strongest constant on swept error | Supports |
> | "Per-episode prediction helps only when the controller is learnable" (BiGym vs SIMPLE) | Insufficient evidence (one confounded comparison) |
> | The drawer end-point loss is caused by the unsettled window | Supports (exploratory; needs the registered test) |
> | The swept metric may flatter the model | Supports, for the drawer tail only |
> | Pitfalls: no leakage, no post-hoc selection | Supports |
> | Pitfalls: metric use is fine | Partly: the "usable" decision rests on the swept metric, while the paper's end-point claim fails on the drawer as measured |
> | Pitfalls: benchmark selection is fine | Not checked (would need files I was told not to read); results apply only to `--stance` |
> | The constant's noise is not the issue | Supports (in-sample 0.88 vs leave-one-out 0.92) |
>
> All the analysis ran on CPU on the 4090 with read-only access to the outputs. Temp files went to `/tmp` and were deleted on both machines.

### 复核后的更正（2026-11-25）

数全部重算对上，没有泄漏。读法照下面改：

1. **"模型可用"成立，但要说清它靠的是扫过误差。**
   - 复核另造了四种更强的常数（含见过本局的、关节空间的、按任务的），模型都赢，38 局里赢 36–38 局。
   - 终点（手掌停稳点）上：伸手任务模型赢（降幅 0.64）；抽屉照登记的量法是常数赢，降幅 −1.11，模型只在 2 / 20 局更好。
   - 论文要说的是终点精度，所以 BiGym 抽屉上的终点结论现在不能写。
2. **扫过误差只在抽屉的末段上偏向模型**：真实局末段的点落在模型最后一段折线上（参考终点到预测停稳点），所以预测的终点走过头也不扣分。在参考结束那一刻（训练和真实局定义相同），模型 0.05 cm、常数 0.28 cm。
3. **抽屉终点输给常数，原因是"停稳点没停稳"**：复核的离线检查支持这个解释（探索性的，要登记后再测）。
   - 真实局 0.5 s 窗口里，手沿模型预测的漂移方向走了 0.42–0.58，最后一刻还在以 1.37 cm/s 移动。
   - 按训练数据（0–7 号桶）里这个窗口走完漂移的比例（中位数 0.54）去折算模型的预测，抽屉终点变成模型 0.11、常数 0.28 cm（降幅 0.62，18 / 20 局模型更好）。
   - 对 S 的含义：现在的抽屉流程里，S 该瞄准的是"下一步动作开始那一刻手在哪"，不是停稳点。
4. **实际增益不大**：不用模型的常数已经去掉 65–87 % 的误差。模型在扫过误差上再多 0.4–1.5 cm，终点上 ±0.3–0.5 cm。伸手任务成败到顶，S 对固定常数修正在成败上很难分开，要看抽屉。
5. **预测偏差**：都偏向模型，已列在"意外"里。
6. **撤回"模型有没有用取决于控制器好不好学"**（结论里 BiGym 对 SIMPLE 的对比），改成证据不足：只有一组对比，而且两个基准的局、量法都不同。
7. **真实局很窄**：瞄准点前后、左右只差约 1 cm，高度差 5 cm（伸手）或 0.2 cm（抽屉）。所有结论只对 `--stance` 这套流程成立，不是 BiGym 默认的底座不动。
8. **两处登记后的改代码合法**：`no_reach` 只跳过没伸手的两局；`bigym_test.py` 只报登记要的量，数逐位对上。
9. **下一步改成**：
   1. **复核的 (b)1：换标签，不改基准。** 从现有训练录像里取"伸手结束后 0.5 s，也就是下一步开始那一刻"的偏差当终点标签，重新训练，在现有 20 局抽屉真实局上和留一常数比终点。不用新仿真，流程不变；先登记。
   2. **复核的 (a)：冻结常数、新局。** 用现有 38 局拟出的每个任务的常数，在没用过的开发种子 720020–720099 上（每个任务 80 局）比。
   3. 1、2 都过了，再写 S 在 BiGym 上的登记。抽屉的 S 瞄准下一步开始那一刻；官方种子 620000–620099 只跑一次。
