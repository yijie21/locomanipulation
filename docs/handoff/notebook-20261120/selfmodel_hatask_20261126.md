# selfmodel_hatask_20261126：HA 任务局里，直接执行对比用自身模型修正（ScaleBFM、HEFT、SONIC）

- **日期**：2026-11-26。在开发任何任务局代码之前写好，单独提交。
- **这是哪一张表**：hadata 和 hadata_sonic 的判据表里写着"用这个臂登记 HA 的'直接执行 vs 用自身模型修正'对比，反解用 lmlcmp3 定下的做法"。本条目就是这次登记。用户 11-26 的方向是"现在不要再发散了"：只在三个 benchmark 上，用每个控制器自己的自身模型，和直接执行比任务指标。本条目是其中 HA 那一张。
- **用户的决定**（11-26，经协调方转达）：
  1. **时限**："官方成功判定，时限放宽到 3 倍"。搬箱子（P&PBox）4350 步，双桌（DoubleDesk）6000 步，两组、三个控制器都一样。官方时限内（1450 / 2000 步）的成功一起报，"抓起并举高"作次要指标。
  2. **闸门过不了**："照跑对比，把检验结果注在旁边"。三个控制器都跑 A 对 R；闸门 (a)(b)(c) 照算照报，但不拦着不跑，结果旁边注明哪一项没过。
     - **按用户 11-26 的决定，这张表三个控制器都跑。**
     - 这和 hadata、hadata_sonic 判据表里的两行不同："1.3–1.5 倍：先在 HA val 上校准，另行登记"和"> 1.5 倍：不用于对比"。这两行在这张表上不再拦着；倍数照算，落在哪一格就注哪一格。
  3. **顺序**（协调方定）：ScaleBFM 先跑；HEFT 的模型一好就跑；SONIC 最后。
  4. **搬东西时上身保持当前姿态**（协调方定的新规矩）：手合上以后的每一段走路（找货架、篮子时的原地转身，走到放置的站位），上身都保持合手那一刻的姿态，不回到默认姿态。两组相同，三个控制器相同。
- **机器**：4090（SAM 3 就在这台上，端口 18832）。从采集借 4 个槽位，具体槽位写进修订 1。4090b 继续采集，不跑这张表。
- **代码**：`selfwm`，`scripts/experiments/selfmodel_hacollect_20261126/` 下新加任务局脚本，复用 hr_real 的流程（修订 2a 和修订 2a 补记冻结的那一版，7dfaa816）。文件名和 md5 写进修订 1，正式运行之前冻结成单独的副本。
- **输出**：`loco_outputs/selfmodel_hatask_20261126/`。
- **关联**：
  - [selfmodel_hadata_20261126](selfmodel_hadata_20261126.md)：ScaleBFM、HEFT 的 HA 数据、训练（修订 2b）、评估 3，以及真实局的流程（修订 2a、2a 补记）；
  - [selfmodel_hadata_sonic_20261126](selfmodel_hadata_sonic_20261126.md)：SONIC 的同一套；
  - [selfmodel_lmlcmp3_20261126](selfmodel_lmlcmp3_20261126.md)：R 的定义（`reachable`）。LML 判定集上 ScaleBFM 的 R 对 A 是 26.0 % 对 13.6 %（+12.4 个百分点，p 4.07e-10），SONIC +3.3 个百分点（p 0.000729），HEFT +2.0 个百分点（p 0.0636）。

## 写这一节时看过什么

- **没有训练过任何 HA 模型**。2b 还没开始，评估 3 的倍数、区间、范围内比例都没人算过。
- **我们的控制器从来没有在 HA 里跑过抓、搬、放。** 现有的真实局（hadata 修订 2a）在最后一下伸手的标签处就结束了。
- **开发期的只伸手冒烟**（种子 296–299，不是正式局），一局到伸完手用掉的步数：
  - P&PBox：ScaleBFM 1232–1593 步，HEFT 1312–1759 步。官方时限是 1450 步。
  - DoubleDesk：ScaleBFM 1146–1445 步，HEFT 911–1151 步。官方时限是 2000 步。
- **正式真实局（200–239）**：只看过进度用的结束行和每一项的局数（hadata 修订 2a 补记第 12 条写过看了哪些），没有统计过结果。
- **之前的结果**：11-25 用 HA 自己的 SONIC 和它的速度规划器跑 ha_bench（开发种子 100/101），60 局成功 1 局。失败的原因是走路、找目标、时间和两张桌子之间 8.2 cm 高的横杆（8 局摔倒），不在伸手。

## 假设

1. 在 HA 的任务局里，把决定"手落在哪"的那几下命令（预抓取和进入）经控制器自己的 HA 自身模型反解、并缩回 IK 可达范围（R），手会比直接执行（A）落得更准。所以抓起更多，任务成功也更多。理由是 LML 上同样的做法让 ScaleBFM 的成功从 13.6 % 升到 26.0 %。
2. 这个收益在 HA 上会被抓起之后的环节（找货架或篮子、带着东西走、放下）冲淡。任务成功的差会小于"抓起并举高"的差。

## 做法

### 任务和局
- **任务**：P&PBox、DoubleDesk。三个控制器相同。OpenDoor 不跑，和 hadata 修订 2a 补记第 7 节一样。
- **种子**：
  - 正式：400–459，每个任务 60 个种子，repeat 0。
  - 冒烟：380–389。
  - 这些种子避开了已经用过或看过的：官方 0–2（`ha_sim.check_seed` 拒绝），ha_bench 100–105，正式真实局 200–239，开发冒烟 296–299。
- **数量**：每个控制器 60 个种子 × 2 个任务 × 2 组 = 240 局，三个控制器共 720 局。

### 两组
- **A（直接执行）**：每个命令的目标就是感知给出的目标。
- **R（自身模型修正）**：lmlcmp3 的 `reachable`，从 lc3_run 冻结的代码直接 import。
  - **做法**：hardgrasp 的不动点迭代，6 步：t ← clip_t(t + 目标 − D(t))。第一个候选就是目标本身；之后每个候选先检查它的规划 IK 误差，> 2 cm 就沿 t − 目标 往回缩到 IK 误差 ≤ 2 cm 的最大比例（二分 6 次；都不行就退回目标本身）。从候选里选 |D(t) − 目标|² + LAMBDA × |集成分歧|² 最小的那个，LAMBDA = 1.0。
  - **只改位置**：只改手点的位置目标，手腕朝向和 A 一样，和 LML 相同。
  - **经过模型的命令**：
    - P&PBox：左手预抓取、右手预抓取（各自单手），以及夹紧那一下（两只手同时往里 10 cm，到箱子侧面里 2 cm）；
    - DoubleDesk：右手预抓取，以及往下 8 cm 到抓取点那一下。
  - **两组都直接执行的命令**：先抬手的中间点（`reach_no` 0），合手后的举高，放置，篮子上方那一下。这和 LML 一样，中间点和举高在 LML 里也是两组都直接执行。
  - **夹紧那一下是 LML 没有的**（LML 的任务都是右手）：两只手同时动，用模型的双手段来预测（data3 的伸手 20 % 是双手同时）。
    - 迭代时两只手各按自己的预测误差更新；
    - 缩回时两只手用同一个比例，条件是两只手里较大的 IK 误差 ≤ 2 cm；
    - 选候选时，两只手的误差和分歧相加。
- **模型**：
  - 选哪个臂：hadata（SONIC 是 hadata_sonic）判据表定的那个，即评估 3 里在真实局上误差较低的那个臂（从头训或微调），数据 100 %，seed 0，2b 选出的最佳 checkpoint。LML 的 R 用的也是 seed 0。
  - 评估 3 对这个控制器是"样本不足"（干净伸手不到 10 下）时，改按 HA val 选误差较低的臂。
  - 选臂用的是 200–239 的真实局，和本条目的 400–459 不重叠。
  - 模型只在 CPU 上推理，和 lc3_run 一样。

### 配对
- **同一个种子，A 和 R 在同一个进程里先后跑**，所以机器和槽位总相同。
- **重放彩色图**：Isaac 的彩色图每次跑都有差别，物理和深度是确定的（hadata 修订 2a 第 12 节）。所以 R 在它第一个和 A 不同的命令（第一下预抓取）之前，每一眼的彩色图都用 A 存下的同名那一眼，深度、位姿和 SAM 3 照常现场算。这之后的眼（找货架、篮子）现场看。
- **一致性检查**：每一对都比较 R 和 A 从重置到第一下预抓取命令那一行的状态（关节、骨盆、命令前的参考帧），记录是否逐位相同。

### 一局的流程（两组相同，只有上面列出的那几下命令不同）
- **伸手阶段**：照 hadata 修订 2a 和 2a 补记，不改：
  - 看一眼，走到站位，最多 3 轮，到站位再看一眼，先抬手的中间点，预抓取；
  - HEFT 走 gait_heft，ScaleBFM、SONIC 走 hg_walk；
  - 站开距离各控制器不同：ScaleBFM 0.25 m，HEFT 0.22 m，SONIC 0.34 m；
  - 每个命令之前等 data3 的停稳规则，最多 6 s，等不到也照发。
- **伸手之后**：照 ha_bench 的固定顺序（它文件头的 TASK RULES），走路和伸手换成我们的（d3 的 plan_reach 全身 IK、hg_walk 或 gait_heft）。
  - **P&PBox**：
    1. 夹紧：两只手同时到侧面里 2 cm；
    2. 两只手合上，保持 25 步；
    3. 两只手一起举高 10 cm；
    4. 看一眼。转到本局起始朝向 +45° / +90° / +135°，直到 SAM 3 看到 "blue metal shelf"。目标层板是货架上低于箱底的最高一层；
    5. 用站位规则走到层板前面，层板中心至少在前方 0.34 m；
    6. 两只手一起把箱子移到层板中心上方，箱底离板 3 cm；
    7. 张开两只手。
  - **DoubleDesk**：
    1. 往下 8 cm 到抓取点；
    2. 右手合上，保持 30 步；
    3. 举高 15 cm；
    4. 看一眼。转到起始朝向 +45° / +90° / +135°，直到看到 "bin" 或 "basket"；
    5. 用站位规则走到篮子那张桌子前，篮沿中心至少在前方 0.40 m、右边 0.20 m；
    6. 抓取点移到篮沿中心上方 15 cm，朝向不变；
    7. 张开右手。
- **合手以后的走路**：上身（腰和两条胳膊）保持合手那一刻的参考姿态，手保持合着。合手之前的走路不变。
- **一局什么时候结束**：以下先到者。
  - 成功：HA 官方奖励到 1（`ha_sim` 的规则，raw_total ≥ 1）；
  - HA 的摔倒检测连续确认 5 步；
  - 3 倍时限；
  - 流程走完（张手）或中途放弃（目标没看到、规划失败），再保持 100 步。100 步是 ha_bench 的 `--hold` 默认值。
  - d3 的"跑丢"只记录，不结束一局。

### 仿真真值只用来打分
- 所有命令只用感知和机器人自己的状态（腿部里程计、IMU、关节），和修订 2a 一样。
- 仿真里的值只用来：
  - 判成功（HA 的奖励）和摔倒（HA 的检测）；
  - 判"抓起并举高"（物体的位姿）；
  - 算手的落点误差、接触和骨盆位移。

### 指标
- **主指标**：3 倍时限内的 HA 官方成功。
- **一起报**：官方时限内的成功，即成功那一步 ≤ 1450（P&PBox）或 ≤ 2000（DoubleDesk）。用的是同一批局。
- **次要指标**：
  - "抓起并举高"：箱子或锤子的高度比重置后高 ≥ 5 cm，连续 ≥ 50 步（1 s）；
  - 经过模型的那几下，手落点离目标的距离（末端标签处，仿真的手点），A 对 R 按下配对；
  - 每局走到哪一步结束、摔倒、接触、骨盆位移（hadata 修订 2a 补记第 10 条的口径）。

### 统计
- 每个控制器 120 对（60 个种子 × 2 个任务）。按对做 McNemar 精确检验（双侧），报成功数、R − A 的百分点差、不一致的对数、p。
- **主指标**：三个控制器一起做 Holm 校正，原始 p 和校正后的 p 都报。
- **其他指标**：只报原始 p。各任务的成功数分开列。
- **一致性检查**：所有对都进主分析（按种子配对，不管前缀是否相同）。只用前缀逐位相同的对，再算一遍作敏感性分析。

### 闸门（照算照报，不拦着不跑）
- **(a)** 干净伸手 ≥ 10 下（hadata 修订 2a 补记第 12 条）。
- **(b)** 选中的臂在真实局上的误差是 val 的 ≤ 1.3 倍，置信区间上限 ≤ 1.5。
- **(c)** 起始状态在训练范围内的比例 ≥ 70 %（hadata 和 hadata_sonic 的预测值）。
- 每个控制器的模型训完以后，在 200–239 的真实局上离线算，写在这个控制器的结果旁边：哪一项过了，哪一项没过，没过时倍数落在判据表的哪一格。

### 冒烟（正式运行之前）
- **ScaleBFM 的 A**：种子 380–389，两个任务，共 20 局，用准备冻结的那一版代码。
  - **停止规则**（协调方定）：A 的成功 ≤ 1/20，就停下来报告协调方，不进正式运行。
  - 不在冒烟种子上反复调参，调到过线为止。
- **R 的代码路径**：ScaleBFM 的 HA 模型还没有。先用 LML 的 D（train3b，ScaleBFM，seed 0）在 2 局冒烟上走通，只检查流程（能加载、能反解、能记录、一致性检查能算），结果不读。
- **每个控制器正式运行之前**：模型训好以后，种子 380 两个任务各跑一对 A 和 R，只检查代码，不读结果。

### 顺序和用时
- ScaleBFM 的模型训完、闸门算完、冒烟过了以后，在 4090 的 4 个槽位上跑 ScaleBFM。HEFT 的模型一好就跑 HEFT。SONIC 最后。
- **用时**：没成功的局大多跑到时限，P&PBox 4350 步、DoubleDesk 6000 步。按 HA 约 0.25 倍实时估，每局 6–8 分钟。每个控制器约 25–35 个槽位小时，4 个槽位约 7–9 小时。

### 记录
每局记下：
- 每一步走到了哪里（阶段）、成功那一步、摔倒那一步；
- 每 10 步物体的位姿（只用来打分）；
- 每个命令：目标、A 或 R 的实际命令、R 的迭代记录（每个候选、缩回比例、预测和分歧）、IK 误差、落点误差；
- 每一下伸手按 data3 的段格式存下来；
- 每一眼的彩色图、深度图和掩码；
- 一致性检查的结果。

## 预测

**冒烟（ScaleBFM A，20 局）**
- 3 倍时限内成功 0–4 局。我估计有一半以上的可能落在 ≤ 1/20，按规则停下。
- 依据：HA 自己的 SONIC 用它自己的走路，在官方时限下 60 局只成功 1 局；我们的控制器从没在 HA 里带着东西走过、放过东西。

**正式运行（如果跑），3 倍时限内的成功，每个控制器 120 对**

| 控制器 | A 的成功率 | R − A | Holm 校正后 p < 0.05 的可能 |
|---|---|---|---|
| ScaleBFM | 2–15 % | −2 到 +6 个百分点 | 约 20 % |
| HEFT | 0–5 % | −3 到 +3 个百分点 | < 10 % |
| SONIC | 0–10 % | −3 到 +5 个百分点 | < 10 % |

- 依据：
  - LML 上 ScaleBFM 的 R 对 A 是 +12.4 个百分点，但那里成功就是抓起；
  - HA 还要搬和放。我估计抓起之后能放成的比例是 20–50 %，所以差会被冲淡到原来的一半以下；
  - HEFT 在冒烟里 18 下伸手全都碰到东西，伸手时骨盆往前移 9–10 cm（hadata 修订 2a 补记）；
  - SONIC 在 HA 里伸完手一直抖（1–14 cm/s），LML 上它的 R 对 A 也只有 +3.3 个百分点。

**官方时限内的成功**
- P&PBox：三个控制器两组都是 0。只伸手就已经用掉 1232–1759 步，官方时限是 1450 步。
- DoubleDesk：每组 ≤ 5 %。

**抓起并举高**

| 控制器 | A | R − A |
|---|---|---|
| ScaleBFM | 10–35 % | +3 到 +15 个百分点 |
| HEFT | 0–15 % | −5 到 +5 个百分点 |
| SONIC | 5–25 % | 0 到 +10 个百分点 |

**机理和其他**
- **落点误差**（经过模型的那几下，中位数）：ScaleBFM 的 R 比 A 低 20–50 %；HEFT、SONIC 低 0–40 %。
- **摔倒**（HA 的检测）：ScaleBFM 5–20 % 的局，HEFT 5–25 %，SONIC 5–20 %。
- **一致性检查**：每个控制器 ≥ 95 % 的对，在第一下预抓取之前逐位相同。
- **闸门**：
  - ScaleBFM 过 (a)；
  - HEFT 很可能过不了 (a)（hadata 修订 2a 补记第 11 条预测 0–25 下干净伸手）；
  - (b)(c) 按 hadata 的预测，倍数 1.0–2.5，范围内比例 ≥ 70 %。所以 (b) 有不小的可能不过。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| 冒烟 ScaleBFM A 成功 ≤ 1/20 | 停下，把每局在哪一步结束报告协调方，不进正式运行，等决定。不在冒烟种子上调参 |
| 某个控制器 > 5 % 的对在第一下预抓取之前不逐位相同 | 先不读这个控制器的结果，查清原因 |
| 某个控制器 3 倍时限内成功 R − A ≥ +5 个百分点，且 Holm 校正后 p < 0.05 | "HA 任务局上，自身模型修正提高了任务成功（该控制器）"，进表，旁边注闸门 |
| 成功分不开，但抓起并举高 R − A ≥ +5 个百分点且 p < 0.05 | "抓起提高了，任务成功没测出差别"。报抓起之后失败在哪一步，不说提高了任务成功 |
| 成功和抓起都分不开 | "HA 上没测出差别"。再看落点：R 没有更准，说明模型在 HA 的任务状态上修不准（对照闸门 b、c）；R 更准但抓起没有更多，说明这里限制抓起的不是落点 |
| 成功或抓起 R − A ≤ −5 个百分点且 p < 0.05 | R 在 HA 上对这个控制器有害，HA 上不用 R。报缩回的比例和模型误差 |
| 某个控制器 A 和 R 的成功都 ≤ 2/120 | 到底了，成功这一项比不出东西。这个控制器只读抓起并举高和落点 |
| 闸门 (b) 或 (c) 没过 | 结果照报，旁边注"模型在 HA 真实局上没过 (b) / (c)"。即使 R 赢了，也不把它归到模型准 |

- **不做的事**：正式运行开始以后，不加局，不换种子，不改流程、阈值、站开距离，不换模型或 checkpoint。

## 预测修订

### 修订 1：冒烟过不了时换主指标（2026-11-26，冒烟开跑之前写定）

- **来历**：用户预先定的，经协调方转达，原话是选"改用'抓起并举高'作主指标"。写这一节时冒烟还没跑，自登记 3844746d 以来没跑过任何局，只读过代码。
- **协调方同意了登记里我自己定的三处**：
  1. P&PBox 的夹紧经过模型；
  2. 一局结束的四种情况，"跑丢"只记录；
  3. 模型的选法。
- **触发条件**：ScaleBFM 的 A 在 20 局冒烟（种子 380–389，两个任务）里，3 倍时限内成功 ≤ 1/20。只看这一项。
- **触发以后**：
  - 不停下。主指标立刻换成"抓起并举高"：箱子或锤子比重置后高 ≥ 5 cm，连续 ≥ 50 步（1 s），定义和登记里的次要指标相同。
  - 在同一批正式局（400–459）上照登记接着跑。局、种子、流程、顺序都不变。
  - HA 官方成功（3 倍时限和官方时限）照报，改作次要指标。
  - 三个控制器都用"抓起并举高"作主指标，保证这一张表的口径一致。Holm 校正改在"抓起并举高"上做，三个控制器一起。
- **没触发（≥ 2/20）**：照登记，主指标仍是 3 倍时限内的官方成功。
- **冒烟只跑 A**：380–389 只跑 ScaleBFM 的 A，不跑 R，也不看 R。
  - 登记"冒烟"一节里"用 LML 的 D 在 2 局冒烟上走通 R 的代码"，改到开发种子 296–299 上做，结果不读。
  - 每个控制器正式运行之前的代码检查（种子 380 两个任务各一对 A 和 R），在冒烟读完之后做，也只检查代码，不读结果。
- **判据表怎么读**（预测和判据原文不改，下面是触发以后的读法）：
  - 第 1 行"冒烟 ScaleBFM A 成功 ≤ 1/20：停下"，换成本修订的规则：把冒烟结果和每局在哪一步结束报告协调方，然后照常等模型、进正式运行。
  - 第 3 行的主指标换成"抓起并举高"：R − A ≥ +5 个百分点，且 Holm 校正后 p < 0.05，结论是"HA 任务局上，自身模型修正提高了抓起并举高（该控制器）"。任务成功作为次要指标附在旁边。
  - 第 4 行（"成功分不开，但抓起提高了"）不再单列，任务成功照报。
  - 第 5–8 行照原文读，其中"成功"读作"抓起并举高"。
  - 不在冒烟种子上反复调参调到过线，这一条不变。
- **预测不改**。登记里我估计有一半以上的可能触发。

### 修订 2：R 的代码从哪来、夹紧那一下怎么预测、闸门 (b) 和夹紧的落点（2026-11-26，冒烟开跑之前）

预测和判据一行不改。写这一节时没训练过任何 HA 模型，也没跑过本条目的任何一局。修订 2 的前三条是协调方 11-26 定的；第 1 条的等价核对结果和代码 md5，在冒烟之前补到本节末尾。

**1. R 的代码：照抄 lc3_run，不 import**
- **改正**："做法 · 两组"里写的"从 lc3_run 冻结的代码直接 import"做不到，原因有两个：
  - lc3_run 在模块一级 import lr_run，会把 LocoManiLab 的驱动一起带进来，HA 的运行环境里装不了；
  - lc3_run 在模块一级断言 `REACHWM_FDVEL` 关着，而 HA 的模型是 FDVEL 开着训的（hadata 修订 2b）。
- **改成**：把 R 用到的部分逐字抄进一个只含反解逻辑的文件 `ht_inv.py`：
  - lc3_run 的 `shrink` 和它的常数（SHRINK_IK 0.02、SHRINK_STEPS 6）；
  - lc3_run `episode` 里那段迭代（从 `t, evs = gh.copy(), []` 到 `best = evs[bi]`）；
  - lml_tasks 的 ITERS 6、LAMBDA 1.0、`clip_t` 和它用的 WIDE。
- **两项核对**，都在冒烟之前做：
  1. **文字核对**：抄过来的函数和那段迭代，和 lc3_run、lml_tasks 冻结版里的源码逐字相同。函数用 `inspect.getsource` 取，迭代段按上面两行的起止截取。只允许改两处，都列出来：调规划和调模型的那一行换成参数传进来的函数；手的那一侧从写死的 "right" 换成参数。
  2. **数值核对**（协调方要求）：
     - **在哪跑**：在 LML 的环境里，用 lc3_run 的原函数，和抄过来的版本喂同一组输入。
     - **输入**：至少 20 组真实的命令、状态和模型输出。来自第 3 条的开发局（种子 296–299，repeat 0–1，LML 的 D），取右手单手的 R：P&PBox 的右手预抓取，DoubleDesk 的右手预抓取和往下那一下，共约 24 组。
     - **怎么喂**：开发局记下每次规划的 IK 误差和每次模型的输出，按候选目标 t 的字节作键。两边都从记录里取，不重新规划、不重新推理。
     - **FDVEL**：两边设成一样。LML 的 D 是 FDVEL 关着训的，所以两边都关。
     - **要求**：缩回后的命令、迭代次数、选中的候选，三样逐位相同。
  - **核对覆盖不到的部分**：左手和夹紧在 lc3_run 里没有，所以不在数值核对里。
    - 左手：只是 `clip_t` 的那一侧换成 "left"，这是 lml_tasks 里本来就有的参数。
    - 夹紧的两手扩展是新写的代码，另外核对一项：两只手各自的更新，和把同样的预测输出代进单手公式算出来的逐位相同。

**2. 夹紧那一下：每只手按单手预测处理**（协调方 11-26 选的选项 1）
- **改正**："做法 · 两组"里"用模型的双手段来预测（data3 的伸手 20 % 是双手同时）"这句不对。
  - HA 的模型只用单手段训练：hadata 修订 2b 的训练输入是 `data/`，`data_both` 不进训练，这和 LML 一样。
  - 模型的输入是 [手的那一侧、目标、朝向、站姿、骨盆高度] 和起点状态，里面没有另一只手的信息。
- **改成**：每只手按单手预测处理，模型的输入里看不出另一只手在动。
  - 夹紧的每个候选（两只手各一个目标）先做一次两手的全身规划；
  - 再像 `data_both` 那样，每只手各切出一段；
  - 每段照单手段的格式读进模型，各得一个预测。
- 迭代、缩回、选候选照原文那三条，不改：两只手各按自己的误差更新；缩回用同一个比例，条件是较大的 IK 误差 ≤ 2 cm；选候选时两只手的损失相加。
- 这样做的误差有多大，由第 3 条闸门 (b) 的双手段那一行来量。

**3. 闸门 (b) 加一行：双手段单独报**（协调方 11-26 确认，只作标注，不拦着不跑）
- **为什么不用真实局**：200–239 的真实局里没有双手段。
  - hr_real 每下伸手都是单手命令：`command()` 里 mode 等于那一侧，`hands` 也只有那一侧。
  - P&PBox 的四下是左抬手、左预抓、右抬手、右预抓，两手夹箱子那一下不在真实局里。
- **用什么数据**：HA 自己采的 `data_both`，取 val 和 test_iid 两个划分，按修订 1 的规则剔除接触段。
  - 划分按重置号哈希，和训练不重叠。hadata 修订 2b 在开训前用 `hd_both_check.py` 实际核对一遍。
- **怎么算**：每只手用单手模型预测，也就是本条目选中的那个臂。和同一划分的单手段误差并排报。
- **写在哪**：闸门 (b) 原来那一项（真实局上的倍数）不变；这一行写在它旁边。

**4. 结果里多报一项：夹紧那一下每只手的落点误差，A 对 R**（协调方 11-26）
- **只有 P&PBox**：每局夹紧那一下，左手、右手分开。
- **落点误差**：末端标签处仿真的手点，到这一下感知给的目标（A 的命令）的距离，和次要指标同一个口径。
- **报什么**：
  - A、R 各自的中位数；
  - 按种子配对的差（R − A）的中位数，95 % 区间用 1000 次种子重抽样；
  - 两组都有夹紧那一下的局数。
- 只报，不做检验，不进 Holm。

**5. 等价核对的结果和代码 md5**（冒烟之前补在这里）

*开发局*
- **在哪跑、跑了什么**：4090 卡 2、卡 3，就是借来的槽位 2b、3a，13:19–13:58Z。用私有开发副本 `loco_outputs/selfmodel_hatask_20261126/dev_code`，从 asrun_real2 拷过来，再加 ht_*.py；拷之前先 `find -type l`，只有 repo/assets、repo/third_party 两个链接，不经链接写东西。
  - ScaleBFM，种子 296–299，repeat 0–1，两个任务，每个种子同一进程里先 A 后 R，共 32 局。
  - R 用 LML 的 D（train3b scalebfm models_D_s0，model.pt 10a21a3c），模板用 lmlreal 的记录，FDVEL 关，时限 1 倍。
  - 只看流程走到哪一步、有没有报错，成功率没读。
- **前缀一致性**：16 对都是 A == R，摘要逐位相同。
- **同一种子换进程，前缀会变**：DoubleDesk s297 r1 原跑的前缀在第 947 帧，换一个进程重跑是第 918 帧。HA 跨进程不能逐位复现，所以配对只在同一进程里成立。这正是设计里让 A、R 在同一进程先后跑、用摘要核对的原因。
- **所以"做法 · 配对"第一条是硬约束**（协调方 11-26）："同一个种子，A 和 R 在同一个进程里先后跑"，原文给的理由只是机器和槽位相同。现在这条是配对成立的前提：
  - 正式运行里，一对 A、R 不能拆到两个进程里跑；
  - 中断后续跑也一样：A 跑完了、R 没跑完的那一对，A 也要在新进程里和 R 一起重跑，旧的 A 不用；
  - 冻结的 ht_task 续跑时，看到 A 的记录已经在，就只跑 R。这会把一对拆开，不过前缀摘要会是 A ≠ R，能查出来。正式运行之前改成"A、R 只要有一个没跑完，就两个都在这个进程里跑"，重新冻结（asrun2），再补记 md5。冒烟只跑 A，不受影响。

*开发局里查出的错误*
- **现象**：DoubleDesk s297 r1 和 s299 r0 的 A 在举起那一下报 IndexError（帧号 1698 / 1837，超出参考长度 1508 / 1633），局记为"错误"结束。
- **原因**：hr_real 在参考用完时直接结束一局；任务局改成"保持最后一帧"，但没有把参考接长。夹紧之后保持加等待，帧号超出了参考的长度。
  - R 那边规划好了再执行，同样的情况会让新参考错位。不过 R 的反解里只要碰到越界都会报错，所以受影响的局都是以"错误"结束，不会悄悄出错。
- **改法**：tick 里帧号到末尾时，用最后一帧把参考接长 500 帧，行和帧一一对应。只改了 tick，不碰 R 的反解和特征。
- **重跑**：改好后，在同一个开发副本里用新代码重跑这两对（1 倍时限）。两对都正常结束，A == R（第 918、1058 帧）。

*开发期间对 ht_task 的其他改动*（都在冻结之前，A 和 R 一样）
- **双手命令的手序**：改成先右后左，和 d3_collect.sample_command 一致。
- **特征构造里的异常**：包成 RuntimeError，和 lc3_run 的 Models.features 一样，不会被当成"这个候选规划失败"。
- **P&PBox 找货架时的箱底高度**：用感知给的两个夹紧点的平均高度，加举高的 10 cm，减箱高的一半。
  - 这和 ha_bench 的口径一样：举高之后的抓取点高度减去箱高的一半。
  - 两组用同一组感知点，所以 A 和 R 选的是同一块层板。
- **合手以后走路时上身怎么保持**：做法里写的是"保持合手那一刻的参考姿态"，实际实现是每次走路开始那一行的参考。
  - 两者的差别是举高：合手之后、第一次走路之前要举高。照字面保持合手那一刻的姿态，等于在走路时把箱子或锤子放回原来的高度。
  - 所以这里把"合手那一刻"读作"合手以后、每次走路开始时"。

*核对结果*（`ht_checks.sh`：LML 的环境，lc3_formal.sh 的解释器和 PYTHONPATH，lmlcmp3 的冻结代码 `code/`；输出在 `dev/check2/all_equiv.{txt,json}`、`all_feat.txt`）
- **文字核对：通过**。
  - `shrink` 和 `clip_t` 与 lc3_run（eca94321）、lml_tasks（246b6d8a）的源码逐字相同。
  - 迭代那一段两边都是 17 行，只有一行不同：`t = LT.clip_t(t + (gh - pos), "right")` 换成了 `... side)`。
  - 原文预计要改两处，"调规划和调模型的那一行换成参数"其实不用改：抄过来的函数用同样的名字接收这些参数，那一行一字未动。
  - 常数（SHRINK_IK、SHRINK_STEPS、ITERS、LAMBDA）相同。HA 运行里打印的 WIDE `[[0.05, -0.6, 0.02], [1.0, 0.35, 1.5]]` 和 lml_tasks 的 L.WIDE 相同。
- **数值核对：通过**。
  - 右手单手的 R 一共 27 组，都逐位相同：P&PBox 的右手预抓取 7 组，DoubleDesk 的右手预抓取 10 组、往下那一下 10 组，其中 4 组来自重跑的两对。
  - 比的项：缩回后的每个候选（也就是每次缩回后的命令）、每个损失、每次缩回的记录（比例、是否缩回、规划次数、IK 误差）、迭代次数（都是 7）、选中的序号和选中的命令。
  - 两边 FDVEL 都关。各局记录里模型读取的标志也核过。
  - P&PBox s298 r1 在 1 倍时限里没走到右手预抓取，这一局没有右手组。
- **缩回的二分没被真实数据走到**：27 组里每个候选的 IK 误差都 ≤ 2 cm，一次都没缩回。所以另加一项合成核对，同样要求逐位相同：
  - 200 个带种子的例子，IK 误差 = max(0, |t − 中心| − 半径)；每 10 个里有 1 个让二分的第一个点规划失败。
  - lc3_run.shrink 对 ht_inv.shrink：200/200 相同，其中 156 个真缩回了；
  - 另一只手永远可达时，ht_inv.shrink2 对单手的 shrink：200/200 相同。
- **左手**（lml_tasks.clip_t 取 "left" 加 lc3_run.shrink）：8/8 逐位相同。
- **夹紧的两手扩展**：3/3 相同。每只手的更新、共同的缩回比例、损失等于两个单手损失之和，都逐位相同。
- **特征核对：通过**。R 选中的那个候选的模型输入（x0、u），和这一局自己的记录用 data.load 读回来的那一段比，32 段都逐字节相同。
  - 另外 7 下命令没有记录段：局在这一下伸手的等待里就到时限了。

*举起之后的环节*（冻结之前补跑，只看流程走到哪一步、有没有报错）
- 1 倍时限下，开发局走不到举起之后的环节。所以 14:01–14:17Z 在卡 2、卡 3 上，用开发种子又跑了 5 局 3 倍时限的 A（新代码，ht_task ca022a11）：
  - P&PBox s296 r0：举起后找货架的转身中 HA 判摔倒，正常结束。
  - P&PBox s297 r0、s299 r0：带着箱子转完 +45° / +90° / +135° 都没找到货架，保持 100 步后正常结束。
    - SAM 3 是有掩码的，是 ha_bench 自己的 find_shelf 没有通过：它要求至少 200 像素、层板直方图过阈值。
    - 箱底高度和 ha_bench 是同一个口径：0.85 + 0.10 − 0.104 = 0.846 m。
  - P&PBox s298 r0：找到货架，带着箱子走站位，几轮下来到了时限。
  - DoubleDesk s296 r0：找到篮子，带着锤子走站位时 HA 判摔倒。
  - 加上 1 倍时限的开发局（DoubleDesk 有 4 局走到了带着走）：举高、带着东西转身找目标、看到目标后带着走站位，这几步都走通了，没有报错。
- **还没走到的**：放上层板、移到篮子上方那一下命令，以及之后的张手。
  - 这一段代码和已经走通的举高是同一套：`shifted(exec_cmd(...))` 加 `arm.go` 直接执行，再加上 `head()` 和张手后的保持。我逐行读过一遍，没发现问题。
  - 冒烟里这一段如果报错，按第 6 条处理：不算进判定，单独报，先修代码。

*冻结*
- **冻结的是什么**：`loco_outputs/selfmodel_hatask_20261126/asrun/`，dev_code 原样拷过来，去掉 `__pycache__`。
  - 冻结时间 FROZEN_UTC 2026-11-26T14:18:10Z，只读。
  - 653 个文件各自的 md5 记在 `MD5`，MD5 文件本身的 md5 是 de311d7f。
  - 两个链接记在 `LINKS`：repo/assets、repo/third_party，指向共享树，只读用。
- **怎么核对**：从 asrun 跑的每一次，启动前都先跑 `ht_freeze.sh check`（`md5sum -c`）。
- 主要文件：

| 文件 | md5 |
|---|---|
| ht_task.py | ca022a11 |
| ht_inv.py | 6760ac73 |
| ht_equiv.py | 4294a235 |
| ht_feat_check.py | a0510cd0 |
| ht_run.sh（启动） | 9152e178 |
| ht_checks.sh | a2b1eae9 |
| ht_freeze.sh | be7f5df2 |
| hr_real.py（asrun_real2 的，未改） | 7dfaa816 |
| hc_ha.py / hc_ha.sh | 58dd6cdc / 9405b08d |
| 对照：lc3_run.py（lmlcmp3 冻结版） | eca94321 |
| 对照：lml_tasks.py | 246b6d8a |

- **冒烟**：14:18:24Z 从 asrun 开跑。ScaleBFM A，种子 380–389，repeat 0，3 倍时限。P&PBox 在卡 2（槽位 2b），DoubleDesk 在卡 3（槽位 3a）。

**6. 以"错误"结束的局不算进冒烟的判定**（协调方 11-26 定，冻结之前写定）
- **什么叫"错误"结束**：局记录里的 `end` 以 `error` 开头，就是任务局代码自己抛了异常，Kit 留着跑下一局。正常结束只有这几种：成功、HA 判摔倒、时限到，以及流程走完后保持到底（`held`）。
- **冒烟怎么判**：修订 1 的"A 的成功 ≤ 1/20"只在正常结束的局里数，分母也只算这些局。
  - 以"错误"结束的局不算进分子，也不算进分母。
  - 这些局单独报协调方：哪一局、在哪一步、什么异常。然后先修代码。
  - 有"错误"局的时候，不直接按修订 1 换主指标。等修好、重跑这些局之后再判。
- **为什么**：代码出错的局不能算成"A 失败"。不然一个代码错误就可能误触换主指标的规则。开发局里已经出过一次这种错误，见下面"开发局里查出的错误"。
- **正式运行同样处理**：以"错误"结束的局先修代码、再重跑，不当作失败进统计。

**7. 正式运行用 asrun2：只改续跑逻辑和 `--arms`**（协调方 11-26 的三个条件；冒烟照旧用 asrun）
- **为什么要改**：第 5 条说"一对 A、R 必须在同一个进程里跑"，但 asrun 的 ht_task 续跑时，看到 A 的记录在就只跑 R，会把一对拆到两个进程。
- **条件 1：和 asrun 的差别只有这两处**。`asrun2` 是 asrun 原样拷过来的，只换了 ht_task.py，其余 652 个文件的 md5 和两个链接都和 asrun 相同。
  - 冻结时间 FROZEN_UTC 2026-11-26T14:23:37Z，653 个文件，MD5 文件本身的 md5 是 2055f665。
  - ht_task.py 从 ca022a11 变成 0fab517e。
  - `diff -r` 只报这一个文件不同，另外报 asrun 里多出的几个 `__pycache__`：
    - 冒烟进程以 root 跑，只读权限挡不住 root，Python 在 asrun 里写了字节码缓存；
    - 这些缓存不在 MD5 里，`md5sum -c` 照常通过；
    - asrun2 拷过来时删掉了它们。
  - 冒烟已经写进 asrun 的这些 `__pycache__` 先不动（协调方 11-26）。中断测试和正式运行都加 `PYTHONDONTWRITEBYTECODE=1`，不往冻结目录里写任何东西；中断测试结束时再查一次 asrun2，要求 MD5 相同、没有 `__pycache__`。
  - asrun2 里的 ht_freeze.sh 没改，它的 `check` 只核对 asrun。所以从 asrun2 跑之前，直接在 asrun2 里 `md5sum -c --quiet MD5`。
  - diff 全文（`loco_outputs/selfmodel_hatask_20261126/asrun2_ht_task.diff`）：

```diff
@@ -762,7 +762,7 @@
     elif a.controller in HR.CLEAR_BY:
         B.CLEAR = HR.CLEAR_BY[a.controller]
     arms = [s for s in a.arms.split(",") if s]
-    assert set(arms) <= set(ARM_ID) and arms == sorted(arms), arms  # A before R
+    assert arms in (["A"], ["A", "R"]), arms  # A before R, in one process (R replays A's looks)
     seeds, repeats = HR.rng_list(a.seeds), HR.rng_list(a.repeats)
     for sd in seeds:
         ha_sim.check_seed(sd)
@@ -818,12 +818,22 @@
     for sd in seeds:
         for rp in repeats:
             name = f"{a.ha_task}_s{sd}_r{rp}"
-            for arm in arms:
-                if (a.out / arm / "log" / f"{name}.json").exists():
-                    continue
-                if arm == "R" and not (a.out / "A" / "log" / f"{name}.json").exists():
-                    print(f"{tag} {name}: R needs A's looks; A missing", flush=True)
+            done = {arm: (a.out / arm / "log" / f"{name}.json").exists() for arm in arms}
+            if all(done.values()):
+                continue
+            for arm in arms:  # revision 2 section 5: a pair runs in one process; whatever an earlier process left of
+                # this pair (a finished arm, or the partial looks / data of one cut short) is moved aside, both run here
+                d = a.out / arm
+                old = [p for p in [d / "log" / f"{name}.json", *sorted((d / "data").glob(f"{name}.*")),
+                                   *sorted((d / "data" / "arm").glob(f"{name}.*")), *sorted((d / "looks").glob(f"{name}_*"))] if p.exists()]  # fmt: skip
+                if not old:
                     continue
+                stale = a.out / arm / "stale" / f"{name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
+                stale.mkdir(parents=True)
+                for p in old:
+                    p.rename(stale / (p.parent.name + "__" + p.name))
+                print(f"{tag} {name}: {arm}: {len(old)} records from an earlier process moved to {stale}; {'/'.join(arms)} run here", flush=True)
+            for arm in arms:
                 lg = episode(a, ctx, arm, sd, rp)
                 if arm == "R":
                     la = json.loads((a.out / "A" / "log" / f"{name}.json").read_text())
```

- **新的规则**：
  - 一对的两个臂都有记录，就跳过；
  - 只要有一个臂没跑完，这一对在之前进程里留下的所有东西都挪到 `<臂>/stale/<局名>_<时间>/`，然后两个臂在这个进程里重跑。挪的东西包括跑完那个臂的记录，以及被打断那个臂已经存下的眼和数据。
  - `--arms` 只允许 A 或 A,R。
- **条件 2：人为中断测试**（结果补在这里，之后才开正式运行）：
  - 在开发种子上用 asrun2 跑一对，在 R 跑到一半时按 pid 杀掉进程，再续跑；
  - 要确认两个臂都在新进程里重跑，而且前缀 A == R；
  - 成功率不读。
  - 冒烟的槽位一空出来就做，大约 10 分钟，做完再把槽位还给采集。
  - **结果：通过**（`ht_resume_test.sh` 47f060d8，commit e6270ea8）
    - 卡 3（槽位 3a），开发种子 P&PBox s296 r0，A,R，1 倍时限，从 asrun2 跑，加了 `PYTHONDONTWRITEBYTECODE=1`。
    - 14:59:25Z 开跑前核对 asrun2 和它的 MD5 相同。15:02:20Z A 结束，用时 127 s。
    - 15:03:23Z，也就是 R 跑了 63 s（A 用时的一半）时，按 pid 杀掉 ht_task.py（pid 1936452，SIGKILL）。这时 R 已存下 4 个眼，还没有数据文件。
    - 同一条命令再跑一次。新进程把 A 的 7 个文件（跑完的记录）和 R 的 4 个文件（半截的眼）都挪进了各自的 `stale/ppbox_s296_r0_20261126T150412Z/`，然后在这个进程里重跑了 A 和 R。
    - 新进程打印"prefix A == R: True"，两个臂的前缀都是 737 帧。两个进程都没有 Traceback。
    - 测试结束后再查 asrun2：和 MD5 相同，没有 `__pycache__`。
    - 成功率没读。输出在 `dev/resume_ppbox`，过程在 `resume_ppbox.test` 和 `resume_ppbox.log`。
- **条件 3**：冒烟只跑 A，继续用 asrun。

### 修订 3：正式运行的结果里加两项描述性统计（2026-11-26 约 15:20Z，冒烟读完之后，协调方定）

- **写这一节时知道什么**：冒烟的 20 局 A 已经读过（见"结果"）。正式运行（400–459）还一局没跑，R 在任何种子上的结果都没读过（开发局和中断测试的成功率没读）。
- **加的两项**：都已经在登记的记录里（每局的 `max_lift_cm`、流程停在哪一步），只是在结果里单独列出来。不算新指标，不进任何判定，判据表一行不改。
  1. **DoubleDesk**：每局锤子的最大抬高（cm），A、R 分开列。
     - 冒烟里 DoubleDesk 10 局有 8 局是 0.0 cm，主指标"抓起并举高"在这个任务上很可能接近 0。
     - 如果正式运行里也是这样，结果里照实写"主指标在这个任务上到底"。这时"手落得准不准"靠登记好的落点误差（修订 2 第 4 条，夹紧那一下每只手的落点，A 对 R）来读。
  2. **P&PBox**：A、R 各有几局卡在 find_shelf 上（流程以"shelf not seen"结束），只陈述。
- **取数**：用 `ht_smoke_stats.py`（ab5ffe95）分别对 A 和 R 跑，它已经逐局打印 `max_lift_cm`，并按任务数出卡在找货架上的局数。
- **闸门**：ScaleBFM 训完、跑完评估后，先把闸门 (a)(b)(c) 的结果报协调方，再进正式运行。

### 修订 4：正式运行的槽位、启动脚本和代码检查（2026-11-26 约 15:30Z，正式运行和代码检查之前）

- **协调方 11-26**：闸门 (a)(b)(c) 按用户的决定只作标注。评估跑完把闸门结果发给协调方，不停下，直接做代码检查，检查过了就进正式运行。R 的模型照登记的规则选，不看闸门的数字临时改。检查没过，或者有局以"错误"结束，就停下报协调方。
- **槽位**：登记说"具体槽位写进修订 1"，修订 1 没写，补在这里：4090 的 2a、2b、3a、3b，也就是卡 2 和卡 3 各两个。
  - 15:23Z 设了 hold，这几个槽位里正在跑的 HEFT d3 分片跑完就不再开新的，没有停任何分片。
  - 卡 0（0a）和卡 1（1a、1b）继续采集。
- **启动脚本**：`ht_formal.sh`（248d2571），放在 `loco_outputs/selfmodel_hatask_20261126/`，不在冻结副本里。它只做四件事：
  1. 核对 asrun2 和它的 MD5；
  2. 设 `PYTHONDONTWRITEBYTECODE=1`、`HT_FDVEL=1`（HA 模型是按 FDVEL=1 训的）；
  3. 把 R 的模型设为 `<采集根目录>/train/<控制器>/models_<臂>_s0`，模板用这个控制器的 HA d3 文件；
  4. 调 asrun2 的 ht_run.sh：A,R，repeat 0，3 倍时限。
- **臂**：hadata 修订 2b 评估 3 选出的臂（真实局上误差较低的那个；样本不足时按 HA val），数据 100 %，seed 0。
- **代码检查**（登记："种子 380 两个任务各一对 A 和 R，只检查代码"）：
  - 用 3 倍时限，和正式运行相同。两个任务各占一个槽位，输出在 `check/<控制器>_<任务>`。
  - 只看这几项：
    - 没有以"错误"结束的局，没有 Traceback；
    - 前缀 A == R；
    - 局记录里 R 的模型 md5 和选中的模型相同，FDVEL 是 True；
    - `ht_checks.sh` 的两项核对：R 的代码和 lc3_run 等价，特征和 data.load 读出来的逐字节相同。
  - 成功、抓起并举高、落点误差都不读。
  - 脚本是 `ht_codecheck.sh <控制器> <臂>`（1c23b874），放在 `loco_outputs/selfmodel_hatask_20261126/`。每项打印 PASS 或 FAIL，最后一行是 CODE CHECK PASS 或 FAIL。它只读局记录里的 model 字段，不读成功和抬高。
  - **ht_equiv 的总判定在这里不能直接用**（运行之前写定）：
    - 它是为开发局写的，有两条开发期的条件：
      1. 局记录里模型读取器的 FDVEL 必须是关的。开发局用的是 LML 的 D，D 是关着 FDVEL 训的；HA 模型是开着 FDVEL 训的，正式运行也必须开着。
      2. 右手组至少 20 个。两对局凑不够。
    - 这两条在代码检查里一定报 FAIL，但都和 R 的算术无关：重放时用的是 HA 局里记下来的预测，不重新算特征。
    - 所以代码检查看它的各个部分：
      - 文本核对 PASS；
      - 每个右手、左手、双手组都逐位相同，右手组至少 1 个；
      - 合成缩回 200/200；
      - 除了这两类，没有别的 FAIL 行。
    - ht_equiv 自己的总判定和 FAIL 行数照实打印出来。
    - 这个读法先在开发核对的输出（`dev/check2/all_equiv.txt`）上试过，读出 27/27、8/8、3/3，和当时一致。那份输出里没有任务结果。
  - **ScaleBFM 的代码检查：通过**（`check/scalebfm_codecheck.txt`）
    - 16:54Z 在卡 3 上开跑：P&PBox 在槽位 3b，DoubleDesk 在槽位 3a。R 用 S_f1_s0（d585348f）。17:04:10Z 两个进程都结束了。
    - 17:04:39Z 跑 `ht_codecheck.sh scalebfm S`，结果：
      - 两个任务都打印了 DONE，0 个 Traceback，0 局以"错误"结束；
      - 两对的前缀 A == R 都是 True；
      - 4 份局记录里模型的 md5 都是 d585348f，FDVEL 都是 True；
      - ht_equiv 逐部分读：文本核对全是 True，右手 3/3、左手 1/1、双手 1/1 组逐位相同，合成缩回 200/200（缩回过的 156 个），除了预先说好的两类以外没有别的 FAIL 行；
      - 特征核对 FEAT CHECK PASS：6 段逐字节相同；
      - asrun2 和 MD5 相同，没有 `__pycache__`。
      - 最后一行：CODE CHECK PASS。
    - ht_equiv 自己的总判定是 EQUIV FAIL，3 行 FAIL，正是预先说好的那两类：两行"the run's model reader had FDVEL on"（P&PBox、DoubleDesk 的 R），一行"3 right-hand groups (the coordinator asked for >= 20)"。
    - 成功、抬高、落点都没读。
  - **ScaleBFM 正式运行开跑**：17:04:54Z – 17:05:54Z 起了 4 个进程（`formal/launch.txt`）。
    - 2a 跑 P&PBox 400–429，2b 跑 DoubleDesk 400–429，3a 跑 P&PBox 430–459，3b 跑 DoubleDesk 430–459。
    - 4 个进程都打印了"asrun2 = MD5"。
    - 2a、2b 的 HEFT 分片在 17:04 前已经结束，没有停任何分片。
- **正式结果的取数脚本**：`ht_formal_stats.py`（8ed77a72）。约 20:15Z 写定，那时正式局一局都没读过（只看过完成的对数和错误局数）。它按登记的指标和统计实现，主指标按修订 1 用"抓起并举高"：
  - **配对**：同一任务、同一种子，A、R 两个臂的记录都在才算一对。以"错误"结束的局单独列出，不进统计（修订 2 第 6 条）。
  - **一致性检查**：前缀不同的对超过 5 % 时，这个控制器的结果不打印（判据表第 2 行）。
  - **二值指标**：抓起并举高（主）、3 倍时限内成功、官方时限内成功。每个都报：
    - A、R 的数；
    - R − A 的百分点差；
    - 两类不一致对的数目；
    - McNemar 精确检验的双侧 p。
    - 另外按任务分开，再只用前缀相同的对算一遍作敏感性分析。
    - `--holm` 用来在三个控制器都跑完后做 Holm 校正。
  - **落点误差**：只算经过模型的那几下，按任务 × 命令 × 手分开。
    - 口径是末端标签处仿真的手点，到这一下感知给的目标。手点取 data.load 的 pos_f，目标取局记录里的 goal，单位 cm。
    - 报 A、R 的中位数，和按种子配对的差（R − A）的中位数，95 % 区间用 1000 次种子重抽样（seed 0）。P&PBox 的夹紧那一下按手分开报，就是修订 2 第 4 条。
    - 坐标系核对：每段记下的目标，要等于局记录里这只手实际执行的目标。
  - **其他**：
    - 每局怎么结束、流程走到哪一步、HA 判的摔倒；
    - 修订 3 的两项：DoubleDesk 每局的最大抬高，P&PBox 卡在 find_shelf 上的局数；
    - R 经过模型的命令里缩回过的比例，以及修正量和预测偏差的中位数；
    - 经过模型的那几下的接触（hd_touch 的规则）和骨盆位移，只作描述。
  - **在代码检查的两对上试过**：
    - 只看结构的模式下：2 对，前缀 2/2 相同；经过模型的手 12 个都有段，坐标系核对 12/12 相等；A 实际执行的目标和 goal 6/6 相同。
    - 完整模式也跑了一遍，输出直接丢掉，只看退出码是 0。
    - 一个指标值都没看。
- **正式运行怎么分**：每个槽位一个进程，跑一个任务的 30 个种子。
  - 卡 2：2a 跑 P&PBox 400–429，2b 跑 DoubleDesk 400–429；
  - 卡 3：3a 跑 P&PBox 430–459，3b 跑 DoubleDesk 430–459。
  - 输出放在 `formal/<控制器>_<任务>/`，同一个任务的两个进程共用，局名不重叠。
  - 某个进程中断了，就用同一条命令续跑，按 asrun2 的规则处理。

### 修订 5：HEFT、SONIC 照 ScaleBFM 的规则跑，另报摔倒的阶段（2026-11-26 约 23:05Z，ScaleBFM 的正式结果和复核之后，HEFT 的任务局一局都没跑之前）

- **写这一节时知道什么**：
  - ScaleBFM 的正式结果和独立复核都读过了（见"结果"）；
  - HEFT 的模型 23:00:31Z 才开训，评估还没跑；
  - HEFT、SONIC 的任务局一局都没跑，代码检查也没跑。
- **协调方 11-26 的决定**：
  1. **ScaleBFM 这一格**：按复核之后的说法定稿。复核建议的两个实验在 3×3 表的范围之外，不跑，由用户定。
  2. **HEFT、SONIC 照同一套规则跑**：采集完 → 训练 → 评估 → 闸门（只作标注）→ 代码检查 → 正式运行。
     - 先跑 HEFT，再跑 SONIC，最后在抓起并举高上做 Holm 校正。
     - 取数用同一个 `ht_formal_stats.py`（8ed77a72），不改。
  3. **摔倒**：HEFT 和 SONIC 的正式运行里，照修订 3 的描述项，报每组摔倒的局数和摔在哪个阶段。只作描述，不进判定，也不据此改任何东西。
- **摔倒的取数**：
  - `ht_formal_stats.py` 只打印摔倒的局数，不打印摔在哪个阶段，所以另写一个只读的小脚本 `ht_falls.py`（c1d56323）。它放在 `loco_outputs/selfmodel_hatask_20261126/` 下，不在冻结副本里。
  - 它只读局记录：`fell_step` 不为空就算摔倒，`phase_at_end` 就是摔倒时所在的阶段（摔倒时这一局就结束了）。以"错误"结束的局不算，和 ht_formal_stats 一样。
  - 每个任务、每组报：摔倒局数、按阶段的计数、每 10 个种子一组的计数；再按种子配对，报两组都摔、只有 A 摔、只有 R 摔、都没摔的对数。
  - 在 ScaleBFM 已经读过、报过的正式局上试过一次，输出和"结果"里事后数的一致：DoubleDesk A 42/60（carry 39、over_basket 2、lift 1），R 46/60（carry 43、over_basket 2、lift 1）；P&PBox A 6/60（search_shelf 5、carry 1），R 2/60（search_shelf 2）。
  - 新算出来的只有配对：DoubleDesk 两组都摔 31 对，只有 A 11 对，只有 R 15 对，都没摔 3 对；P&PBox 分别是 0、6、2、52 对。
- **HEFT 的安排**（和 ScaleBFM 相同，修订 4）：
  - **训练**：H20b 卡 0、1，hadata 冻结的 `hd_train.sh`，23:00:31Z 开始。HEFT 的 d3 和 d3b 都已采完，检查都通过了。
  - **评估**：等训练的 STATUS 变成 done，4090 上的等待脚本会自动起 `hd_eval.sh`。
  - **槽位**：23:01:00Z 在 4090 的 2a、2b、3a、3b 设了 hold。槽位上正在跑的 SONIC 分片照常跑完，一个都没停。
  - **R 的模型**：按 hadata 的规则选，即评估 3 真实局上误差中位数较低的那个臂；干净伸手不到 10 下（样本不足）时，改按 HA val 选。数据 100 %，seed 0。评估 3 自己会打印选臂那一行，照它的结果用，不看闸门改。
  - **代码检查和正式运行**：`ht_codecheck.sh heft <臂>`，过了就用 `ht_formal.sh` 在四个槽位上跑。槽位分法和修订 4 相同：2a P&PBox 400–429，2b DoubleDesk 400–429，3a P&PBox 430–459，3b DoubleDesk 430–459。
  - **什么时候停下来报告**：检查没过，或者有局以"错误"结束。

## 偏离

**偏离 1：开发局的成功和抓起并举高被看到了**（2026-11-26 约 14:35Z，协调方要求如实记下）
- **规则**："开发局只看流程走到哪一步、有没有报错，不读成功率"。
- **怎么发生的**：写冒烟报告脚本 `ht_smoke_stats.py` 时，拿加跑的 5 局 3 倍时限 A 的记录试跑了一遍。屏幕上除了"走到哪一步、怎么结束"，还打出了每局的这几列：`success_step`、`lifted_step`、找货架或篮子那一步的起始帧、`max_lift_cm`，以及按任务汇总的成功数和抓起并举高数。
- **涉及哪几局**：开发种子 P&PBox s296–s299 r0、DoubleDesk s296 r0，都是 ScaleBFM A，在 `dev/full_ppbox`、`full_ppbox2`、`full_doubledesk`。
  - 这几个种子不在冒烟（380–389）和正式运行的种子里。
- **影响**：
  - 代码在这之前（14:18:10Z）已经冻结，冒烟在这之前（14:18:24Z）已经开跑；
  - 没有据此改任何代码、参数或规则，之后也不会。

## 结果

### 冒烟：ScaleBFM A，种子 380–389，两个任务，3 倍时限（2026-11-26 14:18:24Z – 15:12Z）

- **代码**：从 asrun 跑（FROZEN 14:18:10Z），P&PBox 在卡 2（槽位 2b），DoubleDesk 在卡 3（槽位 3a）。两个进程都正常打印 DONE，没有 Traceback。
- **取数**：`ht_smoke_stats.py`（ab5ffe95，commit acf83ac0），只读局记录。命令：

```
python3 ht_smoke_stats.py smoke/scalebfm_ppbox,smoke/scalebfm_doubledesk A
```

- **原始输出**（全文也存在 `loco_outputs/selfmodel_hatask_20261126/smoke/smoke_report.txt`）：

```
task        seed  furthest step  ended                              steps success  lifted search_f  max_lift_cm
ppbox        380  search         held (shelf not seen)               2819    None    1642     1740  41.54
ppbox        381  search         held (shelf not seen)               3473    None    None     2092  0.38
ppbox        382  search         held (shelf not seen)               3418    None    None     2284  0.03
ppbox        383  carry          timeout                             4350    None    None     1925  4.06
ppbox        384  place          timeout                             4350    None    None     1777  1.67
ppbox        385  search         held (shelf not seen)               2889    None    None     1710  1.07
ppbox        386  carry          timeout                             4350    None    2176     2044  10.59
ppbox        387  search         held (shelf not seen)               2814    None    None     1946  0.28
ppbox        388  search         held (shelf not seen)               2816    None    2268     1975  7.7
ppbox        389  carry          timeout                             4350    None    None     3011  1.56
doubledesk   380  search         held (basket not seen)              2579    None    None     1654  0.0
doubledesk   381  carry          HA fall                             2387    None    None     1395  0.0
doubledesk   382  open           held (sequence done)                4152    None    None     1477  0.0
doubledesk   383  open           held (sequence done)                4550    None    None     1467  9.13
doubledesk   384  search         held (basket not seen)              2625    None    None     1589  0.0
doubledesk   385  open           held (sequence done)                4366    None    None     1388  0.0
doubledesk   386  search         held (basket not seen)              2732    None    None     1520  0.0
doubledesk   387  carry          HA fall                             2775    None    None     1432  0.0
doubledesk   388  carry          HA fall                             2703    None    None     1668  0.0
doubledesk   389  open           held (sequence done)                3734    None    None     1497  10.1

episodes 20; ended in error 0 (left out of the rule, revision 2 section 6): 
doubledesk: normal ends 10; success 0; grasp-and-lift 0 (before the search 0); stuck at the search (target never found) 3; ended inside the search otherwise 0; ends: HA fall 3, held (basket not seen) 3, held (sequence done) 4
ppbox: normal ends 10; success 0; grasp-and-lift 3 (before the search 1); stuck at the search (target never found) 6; ended inside the search otherwise 0; ends: held (shelf not seen) 6, timeout 4

A's success: 0/20 -> <= 1/20: revision 1, the main metric is grasp-and-lift
```

- **列的意思**：
  - "furthest step"是流程进到的最后一个环节：抓起（合手）→ 找货架或篮子 → 搬着走 → 放下（P&PBox）或到篮子上方（DoubleDesk）→ 张手。它只说明流程走到了哪里，不说明前面的环节都做成了。流程不用仿真真值判断抓没抓住，所以手里是空的也会接着往下走。例如 DoubleDesk s382、s385 一路走到了张手，但锤子从没离开过桌面（最大抬高 0.0 cm）。
  - "抓起并举高"另看 `lifted` 一列：有数字就是满足了（物体比重置后高 ≥ 5 cm，连续 ≥ 50 步），数字是满足的那一步。
  - "怎么结束"：
    - `timeout`：3 倍时限到（P&PBox 4350 步）；
    - `HA fall`：HA 判摔倒；
    - `held (shelf / basket not seen)`：找货架或篮子找不到，流程先结束，保持到底；
    - `held (sequence done)`：流程走完（张手）后保持到底，没判成功。
  - 20 局里没有成功，没有程序报错，也没有规划失败（命令的 IK 失败）。

- **找货架 / 找篮子**：
  - P&PBox 有 6 局卡在找货架上（ha_bench 的 find_shelf 一直没看到货架）：s380、381、382、385、387、388。
  - DoubleDesk 有 3 局卡在找篮子上：s380、384、386。
  - 在找货架开始之前就已经满足"抓起并举高"的：P&PBox 1 局（s380，第 1642 步满足，找货架从第 1740 步开始，它后来也卡在了找货架上）。DoubleDesk 0 局。
  - 另有 P&PBox 2 局是在找货架开始之后才满足的：s386（第 2176 步，找货架从第 2044 步开始）、s388（第 2268 步，从第 1975 步开始）。
  - 找货架和找篮子都在抓起之后，A 和 R 跑的是同一段代码（R 只改伸手的命令），所以两组会同样碰到。不过上面这 2 局说明"抓起并举高"也可能在找货架期间才满足，找不到货架而提前结束的局，留给它满足的时间就短一些。这一点对 A 和 R 也是一样的。

- **按修订 1 判**：20 局都是正常结束，没有"错误"局，所以修订 2 第 6 条不起作用。A 的 3 倍时限成功是 0/20，≤ 1/20，修订 1 触发：
  - 主指标换成"抓起并举高"，三个控制器都换，Holm 校正也改在它上面做；
  - HA 官方成功（3 倍时限和官方时限）照报，改作次要指标；
  - 不停下，照常等模型训好、做每个控制器正式运行之前的代码检查，然后进正式运行（400–459，从 asrun2 跑）。
  - 按协调方 11-26 的要求，不因为失败的原因是找货架就另作处理，find_shelf 也不调。
- **只陈述，不改规则**：换过去的主指标在这 20 局 A 里是 3/20，P&PBox 3/10，DoubleDesk 0/10。

- **槽位**：冒烟（15:12Z 前）和中断测试（15:08:27Z）都做完了，15:13:11Z 撤掉 hold_2b 和 hold_3a，两个槽位还给了采集。冒烟和中断测试的灵动岛登记都已归档。

### ScaleBFM：R 的模型和闸门（hadata 评估 3，正式运行之前）

- **R 的模型**：S_f1_s0（从头训，seed 0，数据 100 %，`train/scalebfm/models_S_s0`，model.pt 的 md5 是 d585348f）。照登记的规则选出：47 下干净伸手上，真实局误差中位数 S 是 4.36 cm，T 是 6.50 cm，选较低的那个。没有改。
- **闸门**（只作标注，用户 11-26）：
  - (a) 通过：干净伸手 47 下，来自 36 局。
  - (b) 倍数 0.689 ≤ 1.3，区间上限 1.672 > 1.5，第一行两个条件只满足一个。
    - 原数：`ratio to HA val S_f1_s0 0.689 [0.509, 1.672] (val resampled too [0.488, 1.758])`。
    - 双手段一行（val，kept）：S_f1_s0 双手段 9.07 cm [7.52, 11.88]，单手段 6.16 cm [5.70, 6.95]，双手是单手的 1.473 倍 [1.164, 1.994]。
  - (c) 通过：三项范围检查都在训练范围内的比例是 97.9 % [93.6, 100.0]（46 下在内，1 下在外）。
  - 原文在 hadata 条目的"结果"一节和 `train/scalebfm/eval/eval3.txt`。
- **观察**（协调方 11-26 要求写下，不据此做任何事）：S 和 T 在两个任务上差得很多。R 在 hatask 里修正的是预抓取（reach_no 1），抬手中间点（reach_no 0）两组都直接执行。预抓取上 T − S 的区间跨 0。原数：

```
  task doubledesk: n 22 (18 episodes); S_f1_s0 median 13.37 [10.58, 15.00] cm; T_f1_s0 median 6.82 [5.15, 8.52] cm; ratio to HA val S_f1_s0 2.114 [1.672, 2.372]; T_f1_s0 1.119 [0.846, 1.399]; T - S -6.55 [-7.77, -5.04] cm
  task ppbox: n 25 (18 episodes); S_f1_s0 median 2.93 [2.61, 3.64] cm; T_f1_s0 median 5.80 [4.84, 7.77] cm; ratio to HA val S_f1_s0 0.463 [0.413, 0.575]; T_f1_s0 0.953 [0.795, 1.276]; T - S 2.88 [1.35, 5.08] cm
  reach_no 1: n 11 (10 episodes); S_f1_s0 median 4.15 [2.72, 5.11] cm; T_f1_s0 median 3.50 [3.26, 4.24] cm; ratio to HA val S_f1_s0 0.655 [0.430, 0.807]; T_f1_s0 0.575 [0.536, 0.695]; T - S -0.65 [-1.04, 0.69] cm
  task x reach_no doubledesk 0: n 18 (18 episodes); S_f1_s0 median 14.08 [12.69, 15.59] cm; T_f1_s0 median 7.32 [5.78, 9.08] cm; T - S -6.76 [-7.67, -5.76] cm
  task x reach_no doubledesk 1: n 4 (4 episodes); S_f1_s0 median 3.67 [2.66, 4.36] cm; T_f1_s0 median 3.19 [2.10, 3.50] cm; T - S -0.47 [-1.28, 0.31] cm
  task x reach_no ppbox 0: n 18 (18 episodes); S_f1_s0 median 2.91 [2.60, 3.32] cm; T_f1_s0 median 7.32 [5.80, 8.77] cm; T - S 4.41 [2.75, 6.07] cm
  task x reach_no ppbox 1: n 7 (6 episodes); S_f1_s0 median 4.92 [2.67, 5.33] cm; T_f1_s0 median 3.89 [3.33, 4.26] cm; T - S -1.03 [-1.08, 1.08] cm
```

### ScaleBFM 正式运行：种子 400–459，两个任务，A 对 R（2026-11-26 17:04:54Z – 22:34:11Z）

**运行**
- 4 个进程（`formal/launch.txt`），从 asrun2 跑。R 用 S_f1_s0（d585348f），FDVEL 开，3 倍时限。
- 各进程最后一行的时间：
  - DoubleDesk：400–429 是 20:28:57Z，430–459 是 20:51:53Z；
  - P&PBox：400–429 是 22:25:48Z，430–459 是 22:34:11Z。
- 运行情况：
  - 4 个进程都打印了 DONE，0 个 Traceback，0 局以"错误"结束；
  - 没有续跑过，两个任务的 `stale/` 都是空的；
  - 120 对都打印了"prefix A == R: True"；
  - 跑完以后 asrun2 和 MD5 相同，没有 `__pycache__`。
- **槽位**：每个进程结束、核对完它那 30 对都齐了以后，撤掉对应的 hold，槽位还给采集（`selfmodel_hadata_20261126/logs/formal.log`）。撤的时间：2b 20:29:41Z，3b 20:53:00Z，2a 22:26:55Z，3a 22:35:01Z。

**取数**：`ht_formal_stats.py`（8ed77a72，commit 2160c0d7，正式局一局都没读过时写定）。命令如下，其中 RB 是 `selfmodel_simpledata_20261126/asrun/repo_b`，F 是 `selfmodel_hatask_20261126/formal`：

```
cd $RB && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=$RB PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 19 $PY $OT/ht_formal_stats.py --runs $F/scalebfm_ppbox,$F/scalebfm_doubledesk \
  --touch $L/selfmodel_hadata_20261126/asrun_train/selfmodel_hacollect_20261126/hd_touch.py --json $F/scalebfm_report.json 2>&1 | grep -v Warn > $F/scalebfm_report.txt
```

**原始输出**：全文在 `formal/scalebfm_report.txt`（md5 2234cceb），数值版在 `formal/scalebfm_report.json`。

```
controller ['scalebfm']; runs .../formal/scalebfm_ppbox,.../formal/scalebfm_doubledesk; reader flags {'FDVEL': True, 'GOAL_REF': True, 'APPROACH': True, 'QFIX': True}
pairs 120 (ppbox 60, doubledesk 60); ended in error (left out, revision 2 section 6) 0; one arm missing 0 []
consistency: prefixes identical 120/120; differ 0 (0.0 %): 
commands through the model, per hand: segments found 720, none recorded 0; frame check (segment target == the log's executed target) 720 equal, 0 differ []; A's executed == goal 360, differ 0

grasp-and-lift (main, revision 1): A 19/120 (15.8 %), R 13/120 (10.8 %), R - A -5.0 points; discordant R only 11, A only 17; McNemar exact p 0.3449
  ppbox: A 14/60 (23.3 %), R 8/60 (13.3 %), R - A -10.0 points; discordant R only 6, A only 12; McNemar exact p 0.2379
  doubledesk: A 5/60 (8.3 %), R 5/60 (8.3 %), R - A +0.0 points; discordant R only 5, A only 5; McNemar exact p 1
  identical prefixes only (120 pairs): A 19/120 (15.8 %), R 13/120 (10.8 %), R - A -5.0 points; discordant R only 11, A only 17; McNemar exact p 0.3449

success within 3x (secondary): A 0/120 (0.0 %), R 0/120 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1
  ppbox: A 0/60 (0.0 %), R 0/60 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1
  doubledesk: A 0/60 (0.0 %), R 0/60 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1
  identical prefixes only (120 pairs): A 0/120 (0.0 %), R 0/120 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1

success within the official budget (secondary): A 0/120 (0.0 %), R 0/120 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1
  ppbox: A 0/60 (0.0 %), R 0/60 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1
  doubledesk: A 0/60 (0.0 %), R 0/60 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1
  identical prefixes only (120 pairs): A 0/120 (0.0 %), R 0/120 (0.0 %), R - A +0.0 points; discordant R only 0, A only 0; McNemar exact p 1

landing error of the commands through the model (cm; hand point at the end label to the perceived target):
  ppbox pregrasp_left left: A median 12.51 (n 60), R median 7.35 (n 60); paired R - A median -2.76 [-5.72, -1.25] over 60 seeds
  ppbox pregrasp_right right: A median 31.13 (n 60), R median 21.05 (n 60); paired R - A median -4.59 [-8.92, -0.22] over 60 seeds
  ppbox squeeze left: A median 12.26 (n 60), R median 19.48 (n 60); paired R - A median +6.19 [+4.13, +11.00] over 60 seeds
  ppbox squeeze right: A median 34.87 (n 60), R median 20.67 (n 60); paired R - A median -2.68 [-9.98, +2.32] over 60 seeds
  doubledesk descend right: A median 8.43 (n 60), R median 9.41 (n 60); paired R - A median +0.71 [+0.05, +1.42] over 60 seeds
  doubledesk pregrasp_right right: A median 4.27 (n 60), R median 5.53 (n 60); paired R - A median +0.98 [+0.18, +1.89] over 60 seeds

how episodes ended, the furthest step, HA falls (per task and arm):
  ppbox A (n 60): ends {"timeout": 24, "held (shelf not seen)": 30, "HA fall": 6}; furthest {"carry": 19, "search": 35, "place": 5, "open": 1}; HA fall 6
  ppbox R (n 60): ends {"held (shelf not seen)": 25, "timeout": 32, "HA fall": 2, "held (sequence done)": 1}; furthest {"search": 27, "carry": 28, "place": 3, "open": 2}; HA fall 2
  doubledesk A (n 60): ends {"HA fall": 42, "held (basket not seen)": 12, "held (sequence done)": 5, "timeout": 1}; furthest {"carry": 39, "search": 12, "open": 6, "grasp": 1, "place": 2}; HA fall 42
  doubledesk R (n 60): ends {"HA fall": 46, "held (basket not seen)": 3, "held (sequence done)": 9, "timeout": 2}; furthest {"carry": 43, "search": 3, "open": 9, "place": 4, "grasp": 1}; HA fall 46

revision 3 (1) DoubleDesk max lift per episode, cm (seed: A, R):
  400: 0.0, 11.45; 401: 10.7, 0.0; 402: 2.36, 20.91; 403: 11.49, 0.0; 404: 0.0, 0.0; 405: 0.0, 0.0; 406: 0.0, 0.0; 407: 0.0, 0.0; 408: 0.0, 0.0; 409: 0.0, 0.0; 410: 0.0, 9.11; 411: 0.0, 0.0; 412: 26.88, 0.0; 413: 0.0, 8.31; 414: 0.0, 3.24; 415: 0.0, 0.0; 416: 0.0, 0.0; 417: 0.0, 0.0; 418: 0.0, 0.0; 419: 0.0, 15.05; 420: 0.0, 1.94; 421: 0.0, 0.0; 422: 11.48, 0.0; 423: 0.0, 0.0; 424: 0.0, 0.0; 425: 0.0, 22.05; 426: 0.0, 0.0; 427: 4.8, 19.92; 428: 0.0, 0.0; 429: 0.0, 0.0; 430: 2.98, 0.0; 431: 1.59, 0.0; 432: 3.01, 0.0; 433: 0.0, 1.11; 434: 0.0, 0.0; 435: 0.0, 0.0; 436: 0.0, 0.0; 437: 0.0, 0.11; 438: 0.0, 6.32; 439: 0.0, 0.0; 440: 0.0, 0.0; 441: 23.75, 0.0; 442: 0.0, 0.0; 443: 0.0, 0.0; 444: 0.0, 0.0; 445: 0.0, 0.0; 446: 0.0, 0.0; 447: 0.46, 0.0; 448: 0.0, 0.0; 449: 0.0, 0.0; 450: 0.0, 14.98; 451: 0.0, 0.0; 452: 0.0, 0.0; 453: 0.0, 5.25; 454: 0.0, 0.0; 455: 0.0, 0.0; 456: 0.0, 0.0; 457: 1.67, 0.0; 458: 12.92, 10.16; 459: 0.0, 0.0
  episodes with max lift 0.0: A 47/60, R 45/60
revision 3 (2) P&PBox stuck at find_shelf (sequence 'shelf not seen'): A 30/60, R 25/60

R's commands through the model: 300; with a shrink 4 (1.3 %); correction median 7.29 cm; predicted miss median 1.17 cm (two-hand: the larger hand)
contacts (hd_touch's rule) and pelvis displacement on the reaches through the model (descriptive):
  ppbox A: hands 240, touching 229; pelvis forward median -2.8 cm, lateral median -2.5 cm
  ppbox R: hands 240, touching 196; pelvis forward median -3.4 cm, lateral median -0.8 cm
  doubledesk A: hands 120, touching 109; pelvis forward median -2.8 cm, lateral median 1.8 cm
  doubledesk R: hands 120, touching 65; pelvis forward median -1.8 cm, lateral median 1.9 cm
```

- 第一行里两个运行目录的长路径换成了 `...`，其余照原样。

**主指标：抓起并举高（修订 1）**
- 合计：A 19/120（15.8 %），R 13/120（10.8 %），R − A −5.0 个百分点，McNemar 精确检验 p 0.3449。不一致的对里，只有 R 抓起的 11 对，只有 A 抓起的 17 对。
- 按任务：
  - P&PBox：A 14/60，R 8/60，R − A −10.0 个百分点，p 0.2379；
  - DoubleDesk：A 5/60，R 5/60，p 1。
- 120 对的前缀都相同，所以"只用前缀相同的对"那一遍和主分析完全一样。
- **Holm 校正**：要等 HEFT、SONIC 跑完后三个一起做，现在只有原始 p。Holm 校正后的 p 不会小于原始 p，所以 ScaleBFM 这一格"不显著"的结论，不会因为另外两个控制器的结果而改变。

**次要指标：HA 官方成功**
- 3 倍时限内和官方时限内，两组都是 0/120。

**落点误差**（经过模型的那几下，单位 cm，每格两组各 60 只手）

| 任务 | 命令 | 手 | A 中位 | R 中位 | R 比 A | 按种子配对 R − A 的中位 [95 %] |
|---|---|---|---|---|---|---|
| P&PBox | 预抓取 | 左 | 12.51 | 7.35 | −41 % | −2.76 [−5.72, −1.25] |
| P&PBox | 预抓取 | 右 | 31.13 | 21.05 | −32 % | −4.59 [−8.92, −0.22] |
| P&PBox | 夹紧 | 左 | 12.26 | 19.48 | +59 % | +6.19 [+4.13, +11.00] |
| P&PBox | 夹紧 | 右 | 34.87 | 20.67 | −41 % | −2.68 [−9.98, +2.32] |
| DoubleDesk | 预抓取 | 右 | 4.27 | 5.53 | +30 % | +0.98 [+0.18, +1.89] |
| DoubleDesk | 往下到抓取点 | 右 | 8.43 | 9.41 | +12 % | +0.71 [+0.05, +1.42] |

- "R 比 A"是 R 中位数除以 A 中位数再减 1。原始输出里没有这一列，是我按上面的中位数算的。
- 坐标系核对 720/720 相等；A 实际执行的目标和 goal 360/360 相同。
- 修订 2 第 4 条要报的夹紧那一下，就是表里第 3、4 行。两组每局都有夹紧那一下，60 局。

**一局怎么结束、摔倒**
- HA 判摔倒：
  - P&PBox：A 6/60，R 2/60；
  - DoubleDesk：A 42/60，R 46/60；
  - 合计：A 48/120（40 %），R 48/120（40 %）。
- 下面几项是事后从局记录里数的，原始输出里没有，不进判定：
  - 摔倒时停在哪个环节：DoubleDesk 大多是带着锤子走向篮子那张桌子的时候（A 42 次里 39 次，R 46 次里 43 次）；P&PBox 大多是找货架时的转身里（A 6 次里 5 次，R 2 次都是）。
  - DoubleDesk 的摔倒按种子每 10 个一组：A 是 8、6、5、8、9、6，R 是 8、8、7、6、8、9。两个进程（400–429、430–459）A 各 19、23 次，R 各 23、23 次。
  - 冒烟里 DoubleDesk A 是 3/10 摔倒。它和正式运行 A 的 42/60 比，Fisher 精确检验 p 0.0282。

**修订 3 的两项**
1. **DoubleDesk 每局锤子的最大抬高**：A 60 局里 47 局是 0.0 cm，R 是 45 局。冒烟是 10 局里 8 局，正式运行"也是这样"，所以按修订 3：**主指标在这个任务上到底**。
   - 两组都是 5/60，不一致的 10 对一边 5 对，没有一对两组都抓起来。
   - 修订 3 原文说这时看"修订 2 第 4 条，夹紧那一下"的落点误差。但夹紧只有 P&PBox 有，所以 DoubleDesk 看它自己经过模型的两下，也就是登记的次要指标：
     - 预抓取和往下到抓取点，R 都比 A 略差：配对差 +0.98 cm 和 +0.71 cm，区间都不含 0。
     - A 往下那一下的落点误差中位是 8.43 cm。
2. **P&PBox 卡在 find_shelf 上**：A 30/60，R 25/60。只陈述。find_shelf 没调。

**R 的命令**
- 经过模型的命令 300 下，其中缩回过的 4 下（1.3 %）。
- 修正量的中位是 7.29 cm，选中候选的预测偏差中位是 1.17 cm。
- 事后加看了规划的 IK 误差（局记录的 `ik_err_cm`，不进判定）：
  - A 经过模型的那几下，每个命令、每只手的中位 ≤ 0.002 cm，最大 1.607 cm，没有一只手 > 2 cm。所以 A 落点误差大，不是因为目标够不着；
  - R 的夹紧有左右手各 2 只 > 2 cm，最大 63.585 cm。

**接触和骨盆位移**（hd_touch 的规则，只作描述）
- 碰到东西的手：
  - P&PBox：A 240 只里 229 只，R 196 只；
  - DoubleDesk：A 120 只里 109 只，R 65 只。
- 骨盆位移的中位，两组都在 −3.4 到 +1.9 cm 之间。

**预测和结果**

| 项 | 预测（ScaleBFM） | 结果 | 落在区间里？ |
|---|---|---|---|
| 抓起并举高，A | 10–35 % | 15.8 % | 是 |
| 抓起并举高，R − A | +3 到 +15 个百分点 | −5.0 个百分点（p 0.3449） | 否，低于区间 |
| 3 倍时限内成功，A | 2–15 % | 0 % | 否，低于区间 |
| 3 倍时限内成功，R − A | −2 到 +6 个百分点 | 0 | 是 |
| 官方时限内成功 | P&PBox 两组 0；DoubleDesk 每组 ≤ 5 % | 0；0 | 是 |
| Holm 校正后 p < 0.05 的可能 | 约 20 % | 原始 p 0.3449，校正后不会更小 | 没发生 |
| 落点误差，R 比 A | 低 20–50 % | 6 行里 3 行低 32–41 %（其中夹紧右手的区间跨 0），3 行反而高 12–59 % | 一半是 |
| HA 摔倒 | 5–20 % 的局 | A 40 %，R 40 %（P&PBox 10 %、3.3 %，DoubleDesk 70 %、76.7 %） | 否，高于区间 |
| 一致性 | ≥ 95 % | 100 % | 是 |
| 闸门 | 过 (a)；(b) 倍数 1.0–2.5，有不小的可能不过；(c) ≥ 70 % | (a) 过；(b) 倍数 0.689，区间上限 1.672 > 1.5；(c) 97.9 % | (a)(c) 是；(b) 的倍数低于区间，区间上限没过 |

**判据表**（按修订 1 的读法，"成功"读作"抓起并举高"）
- **第 2 行**（> 5 % 的对前缀不同）：没触发，0 %。
- **第 3 行**（R − A ≥ +5 个百分点，且 Holm 校正后 p < 0.05）：没触发。
- **第 5 行**（成功和抓起都分不开）：**触发**，结论是"HA 上没测出差别"。原文接着要看落点：
  - **决定抓不抓得住的那一下，R 没有更准**：
    - P&PBox 夹紧，左手 R 差 6.19 cm（区间不含 0），右手分不开；
    - DoubleDesk 往下到抓取点，R 差 0.71 cm（区间不含 0）。
    - 照原文读：模型在 HA 的任务状态上修不准。对照闸门：(b) 的区间上限没过，(c) 过了。
  - **P&PBox 的预抓取，R 更准**：两只手的区间都不含 0。但 P&PBox 的抓起没有更多（−10.0 个百分点，p 0.2379）。照原文读：对预抓取这一下来说，限制抓起的不是它的落点。
- **第 6 行**（R − A ≤ −5 个百分点，且 p < 0.05）：没触发。R − A 正好是 −5.0，但 p 0.3449。
  - P&PBox 单独看是 −10.0、p 0.2379，也没触发；而且这一行是按控制器读的。
- **第 7 行**（A、R 都 ≤ 2/120）：
  - 抓起并举高 19、13，没触发；
  - 次要指标官方成功两组都是 0/120，在这一项上到底了。
- **第 8 行**（闸门 (b) 或 (c) 没过）：触发。注"**模型在 HA 真实局上没过 (b)**"：倍数 0.689 ≤ 1.3，区间上限 1.672 > 1.5，第一行两个条件只满足一个。R 没有赢，所以不存在把赢归到模型准不准的问题。
- **正式运行开始以后**：没加局，没换种子，没改流程、阈值、站开距离、find_shelf，没换模型或 checkpoint。

### HEFT：R 的模型和闸门（hadata 评估 3，正式运行之前）

- **评估 3 先停过一次**：第一次合并真实局时报 FAIL。原因是 7 局 P&PBox 真实局在伸手途中跑丢了，评估代码没预料到这种情况。修好以后，用 eval_code2 只重跑了评估 3，见 hadata 修订 2c。
- **R 的模型**：T_f1_s0（从 F 微调，seed 0，数据 100 %，`train/heft/models_T_s0`，model.pt 的 md5 是 0cced537）。
  - 干净伸手只有 5 下，样本不足，所以照规则按 HA val 选：T_f1_s0 5.703 cm，S_f1_s0 5.735 cm，选较低的 T。
  - 没有改。
- **闸门**（只作标注，用户 11-26）：
  - (a) 没过：干净伸手 5 下，不到 10 下。真实局合并后 229 段里有 223 段碰到东西。
  - (b) 样本不足，不算倍数。
    - 双手段一行（val，kept）：T_f1_s0 双手段 10.15 cm [5.83, 13.16]，单手段 5.53 cm [4.47, 6.48]，双手是单手的 1.834 倍 [1.085, 2.665]。
  - (c) 5 下干净伸手都在训练范围内（5/5，100 %），但只有 5 下。
  - 原文在 hadata 条目"结果 · HEFT"一节和 `train/heft/eval/eval3.txt`。
- 预测里写过"HEFT 很可能过不了 (a)"，结果就是这样。

## 意外

- 冒烟 A 的 3 倍时限成功是 0/20，落在预测"0–4 局，一半以上可能 ≤ 1/20"的范围里，修订 1 触发。
- 一半以上的局（20 局里 9 局）卡在找货架或找篮子上，比"带着东西走、放下"这些环节还多。
- DoubleDesk 的锤子 10 局里有 8 局最大抬高是 0.0 cm。流程照样走完了 4 局，到篮子上方张了手。

**ScaleBFM 正式运行**
- **R 抓起反而少了**：预测是 +3 到 +15 个百分点，结果 −5.0（p 0.3449）。少的都在 P&PBox（−10.0），DoubleDesk 持平。
- **摔倒比预测多很多**：两组都是 40 % 的局，预测 5–20 %。
  - 几乎都在 DoubleDesk（A 70 %，R 76.7 %），而且大多发生在"带着锤子走向篮子那张桌子"这一段。这一段不管锤子有没有抓起来都会走，多数局的锤子其实没离开桌面。
  - 冒烟里 DoubleDesk A 只有 3/10 摔倒，和正式运行的 42/60 比，Fisher p 0.0282。
  - 已经查过、可以排除的：
    - 摔倒在种子和两个进程之间分布均匀，没有越跑越多；
    - FDVEL 只进模型的输入特征（data.py 的 x0），不进执行；
    - asrun2 和 asrun 只差续跑逻辑。
  - 没查清的：冒烟种子 380–389 的场景本来就容易；或者正式运行里 A 在同一个进程里接在上一个种子的 R 后面跑（冒烟只跑 A）。这次没有加局去分辨，正式运行开始以后不加局。
- **A 的手离目标很远**：P&PBox 经过模型的那几下，A 的落点误差中位是 12–35 cm，可是规划的 IK 误差几乎是 0（最大 1.607 cm）。
  - 也就是说，目标够得着，手却没到。这几下 A 的手 240 只里有 229 只碰到了东西。
  - R 选中候选的预测偏差中位只有 1.17 cm，R 实际的落点误差中位是 7–21 cm。模型预测的偏差比实际小得多。
  - 闸门 (b) 是在干净伸手（不碰东西）上量的。任务局里经过模型的伸手，绝大多数不是这种。
- **同一个种子，A 和 R 抓不抓得起来几乎互不相关**：两组都抓起的只有 2 对。如果两组完全独立，按 19/120 和 13/120 算，期望也是 2.06 对。前缀到第一下预抓取都逐位相同，之后的结果却像重新掷了一次。
- **R 的手碰到东西的少**：P&PBox 240 只里 196 只（A 229 只），DoubleDesk 120 只里 65 只（A 109 只）。
- **P&PBox 夹紧，左手 R 比 A 差 6.19 cm**（区间 [+4.13, +11.00]），而两只手的预抓取都是 R 更准。
- **前后两半种子差得很多**（独立复核发现，我按局记录核过）。抓起并举高的局数：

  | 任务 | 组 | 400–429（卡 2） | 430–459（卡 3） |
  |---|---|---|---|
  | P&PBox | A | 10/30 | 4/30 |
  | P&PBox | R | 8/30 | 0/30 |
  | DoubleDesk | A | 4/30 | 1/30 |
  | DoubleDesk | R | 4/30 | 1/30 |

  - 合计，前一半 A 14、R 12，后一半 A 5、R 1。
  - P&PBox 的 −10 个百分点主要来自后一半。
  - 种子范围和显卡是绑在一起的，这次分不开是哪个造成的。两半的摔倒数差不多。

## 结论与下一步

**ScaleBFM（三个控制器里的第一个）**
- **结论**：HA 上没测出差别（判据表第 5 行，修订 1 的读法）。
  - 主指标抓起并举高：A 19/120，R 13/120，R − A −5.0 个百分点，McNemar p 0.3449（"结果 · ScaleBFM 正式运行"的 grasp-and-lift 一行）。
  - 也不能说 R 有害：第 6 行要求 p < 0.05，没满足。
  - 次要指标 HA 官方成功，两组 3 倍时限内和官方时限内都是 0/120，到底了。
- **落点**：决定抓不抓得住的那一下，R 没有更准。
  - P&PBox 夹紧，左手差 6.19 cm、右手分不开；DoubleDesk 往下那一下差 0.71 cm（落点误差表第 3、4、6 行）。
  - 照判据第 5 行，这说明模型在 HA 的任务状态上修不准。它和闸门 (b) 没过的那一条（区间上限 1.672 > 1.5）一致。
  - P&PBox 的预抓取 R 更准（第 1、2 行），抓起却没有更多。
- **注**：模型在 HA 真实局上没过 (b)。
- **证据强度**：
  - 一个控制器，一个模型（seed 0），每对各跑一次；
  - 60 个种子 × 2 个任务 = 120 对，共 240 局；
  - 主指标不一致的对只有 28 对，检验只分得出较大的差别。
- **其他可能的解释**：
  1. **接触**：任务局里经过模型的伸手几乎都碰到东西（A 在 P&PBox 是 229/240），模型的准确度却只在不碰东西的伸手上量过（闸门 (b)）。所以这次的结果只说明这个模型在这些状态下修不准，不能推出"用自身模型修正"这个做法本身在 HA 上没用。
  2. **噪声**：同一个种子 A、R 抓起几乎互不相关（两组都抓起 2 对，独立时期望 2.06 对）。抓起与否主要取决于前缀之后每次都不一样的细节，配对几乎没带来额外的灵敏度。
  3. **P&PBox 少了 6 个抓起的原因没检验**。可能是夹紧左手落得更差，也可能是 R 的手碰到箱子的少（夹紧本来就要压住箱子两侧）。两者都只是描述性的数，没检验过。
- **四类陷阱**：
  - **benchmark 选择**：任务（P&PBox、DoubleDesk）、种子（400–459）、局数都在登记里定死；没有丢掉任何局，没有"错误"局。
  - **数据泄漏**：
    - HA 模型的训练数据是 d3 采集，重置号从 91,000,000 和 92,000,000 起，HA 局的种子就用重置号，和 400–459 不重叠；
    - 选臂用的是 200–239 的真实局；
    - 归一化是模型状态的一部分，只来自训练数据。
  - **指标误用**：
    - 主指标换成抓起并举高，是修订 1 在冒烟之前写定的，由冒烟 A 的 0/20 触发，那时 R 一局都没读过；
    - 三个二值指标、6 行落点误差、修订 3 的两项都照登记全报了。
  - **事后挑选**：
    - 一次运行，模型按登记的规则选，没看闸门改；
    - 取数脚本在读任何正式局之前写定；
    - 事后加看的几项（IK 误差、摔倒的环节、"R 比 A"一列）都注明了，不进判定。
- **独立复核之后的更正**（复核原文见下一节；判据表落在哪一行不变）：
  - **"模型在 HA 的任务状态上修不准"改成"证据不足"**：
    - 这是判据第 5 行原文给的读法。但经过模型的那几下几乎都带接触；按复核的拆分，R 把竖直方向的下垂基本修掉了，水平方向没修好。
    - 现有数据分不开"模型不准"和"落点由接触主导"这两种解释。复核的实验一可以把它们分开，没跑。
    - 所以只能确定这个数字陈述：决定抓取的那一下，R 的落点没有更准。
  - **修正量的中位 7.29 cm 只含 240 个单手命令**：取数脚本只收数值型的 corr_cm，夹紧（按手存）没算进去。复核把夹紧按较大的那只手算进去，300 下的中位是 8.15 cm。
  - **DoubleDesk 预抓取"区间不含 0"不稳**：换一个种子重抽，区间下端是 +0.00；符号检验 p 0.093。往下那一下和 P&PBox 夹紧左手是稳的。
  - **"R 比 A"一列是两组中位数之比，不是配对差**：夹紧右手这一列是 −41 %，但配对差 −2.68 cm，区间跨 0。
  - **DoubleDesk"主指标到底"**：两组各 5/60，不是 0。更准确的说法是抓起太少，只有 10 对不一致，测不出中等大小的差。
  - **差别的区间**：复核按不一致对算的精确 95 % 区间是 [−13.3, +4.4] 个百分点。预测的 +3 到 +15，以及第 3 行的 +5 门槛，基本在区间外。所以数据对"ScaleBFM 在这套流程里有正收益"不利，不只是没测出差别。
  - **复核看到的 R 特有的失败**（事后看的，不进判定，我没逐项重算）：
    - R 有 4 局合手时箱子被弹高 17–69 cm，但没算作抓起；
    - R 的夹紧目标比感知点高，中位左手 +4.7 cm、右手 +6.1 cm；
    - P&PBox 里 d3 判"跑丢"的局，R 25 局、A 12 局。种子 421 和 459 的 R 跑丢以后，夹紧目标离机器人 1.3–1.6 m，这就是那几只手 21–64 cm IK 误差的来历。
- **下一步**：
  - ScaleBFM 这一格到此为止，照规则不加局。
  - 前后两半种子的差别（见"意外"）不影响同一进程里的配对，但 P&PBox 的 −10 个百分点不宜外推。
  - 复核建议的两个实验都要新开种子、新加局，在用户定的 3×3 表的范围之外。我不跑，留给协调方和用户决定。
  - 照顺序，HEFT 的模型一训好（d3b 采集完以后），就走评估 → 闸门 → 代码检查 → 正式运行；然后是 SONIC。
  - 三个都跑完以后，在抓起并举高上做 Holm 校正。

## 独立复核

### ScaleBFM 正式运行（2026-11-26）

- 复核 agent 只拿到了这些：
  - 登记，从开头到修订 4；
  - "结果 · ScaleBFM 正式运行"一节；
  - 代码和远程原始数据的路径。
- 下面是它的回复原文，只做了两处格式调整：标题各降一级；通知里转义的 `&amp;`、`&lt;` 还原成 `&`、`<`。

#### 独立复核（ScaleBFM 正式运行，种子 400–459）

我只读了登记、结果一节、代码和远程的原始数据。没有启动任何仿真或 GPU 进程，也没有改动已有文件。临时脚本和输出只写在远程 `/tmp/review_hatask_*`，本地副本在 `/Users/zed/.claude/jobs/f13ab968/tmp/review_hatask_*.py`。

#### 1. 数字核对

**自己重算的，全部对上**

我另写了脚本，只读 240 份局记录，不调用取数脚本。
- **主指标"抓起并举高"**（lifted_step 不为 None）：
  - 合计 A 19/120，R 13/120。不一致的对里只有 R 抓起的 11 对，只有 A 抓起的 17 对，McNemar 精确双侧 p 0.3449。
  - P&PBox：A 14，R 8，不一致 6 对 12，p 0.2379。
  - DoubleDesk：A 5，R 5，不一致 5 对 5，p 1。没有一对两组都抓起。
- **官方成功**：3 倍时限和官方时限下，两组都是 0/120。
- **摔倒**：
  - 合计 A 48，R 48。P&PBox A 6、R 2；DoubleDesk A 42、R 46。
  - 摔倒时所在的环节、每 10 个种子的分组数、两个进程的分组数，都和结果一节一致。
  - 冒烟 DoubleDesk A 是 3/10 摔倒，和正式运行 42/60 比，Fisher 精确检验 p 0.0282，也对。
- **DoubleDesk 最大抬高为 0 的局**：A 47/60，R 45/60。逐种子的数值和原始输出相同。
- **P&PBox 卡在找货架上**：A 30，R 25。
- **一局怎么结束、走到哪一步**：和原始输出一致。各有 1 或 2 局"流程走完后在保持时到了时限"，取数脚本把它们记成"时限到"，口径一致。
- **运行本身**：
  - 0 局以"错误"结束。
  - 120 对的前缀逐位相同。4 份进程日志里各有 30 行"prefix A == R: True"，没有 False。
  - R 的 120 份记录里模型 md5 都是 d585348f，FDVEL 都开着。
  - 没有续跑，`stale` 目录根本不存在，和"空的"等价。
- **IK 误差**：
  - A 经过模型的那几下，中位 ≤ 0.002 cm，最大 1.607 cm。
  - R 夹紧那一下超过 2 cm 的，左右手各 2 只，在种子 421 和 459，最大 63.585 cm。
  - 缩回过的 4 下（共 300 下），没有退回目标的情况。
- **冻结代码**：
  - 远程 asrun2 用 `md5sum -c` 核对通过，没有 `__pycache__`，MD5 文件本身的 md5 是 2055f665。
  - 本地 ht_task.py 是 0fab517e，ht_formal_stats.py 是 8ed77a72，远程 ht_formal.sh 是 248d2571。

**重跑取数脚本**
- 按材料里的命令重跑，输出写到 `/tmp/review_hatask_report.{txt,json}`。
- 和原文件逐字节相同（txt 的 md5 是 2234cceb）。结果一节贴的原始输出，换回长路径后也和它逐字相同。

**落点误差：从原始段文件独立算**
- 我不用 data.load，自己从段文件的 W、start、n_steps 算末端标签处的手点，和局记录里的 goal 比。
- 6 行的 A、R 中位数和配对差的中位数都和原始输出一致。
- 我用另一个随机种子重抽 5000 次，区间如下：
  - P&PBox 预抓取左 [−5.72, −1.09]，右 [−8.92, −0.22]；
  - 夹紧左 [+3.73, +10.96]，右 [−10.58, +2.19]；
  - DoubleDesk 往下那一下 [+0.06, +1.40]，预抓取 [+0.00, +1.91]。

**没对上，或写法不准的地方**（都不影响判定）
1. **修正量的中位 7.29 cm**：只含 240 个单手命令。取数脚本只收数值型的 corr_cm，60 个夹紧的记录是按手存的，被排除了。把夹紧按较大那只手算进去，300 下的中位是 8.15 cm。夹紧每只手的修正中位：左 10.1 cm，右 9.8 cm。
2. **"区间都不含 0"在 DoubleDesk 预抓取上不稳**：
   - 原区间 [+0.18, +1.89]；换一个种子重抽，下端是 +0.00；
   - 符号检验：60 对里 37 对 R 更差，p 0.093；
   - 往下那一下是稳的：40/60，p 0.014；
   - P&PBox 夹紧左手也稳：41/60，p 0.006。
3. **我没核的**：
   - 闸门 (a)(b)(c) 的数字（47 下干净伸手、0.689、1.672、97.9 %）来自 hadata，不在这次的材料里；
   - 接触和骨盆位移只靠重跑取数脚本核对（结果一致），没有独立重算接触规则。

#### 2. 四类陷阱

- **benchmark 选择：没发现问题。**
  - 任务、种子、局数都是登记里写定的。OpenDoor 不跑是更早的决定。
  - 两个任务的官方成功都是 0，不是挑了容易的。
  - 闸门 (b) 没过也照跑照报了。
- **数据泄漏：没发现迹象。**
  - 模型在 16:49Z 前训完，正式运行 17:04Z 开始，120 局 R 用的是同一个 md5。
  - 选臂用的是 200–239 的真实局。采集用的重置号从 89–92M 起编（结果局是 96M 起），和 400–459 不重叠。
  - 取数脚本只读局记录和段文件，不碰模型和归一化。
  - 另外，即使有泄漏，它也只会帮 R，而 R 没有赢。
- **指标误用：主指标换过，但换得合规；有几处写法需要标注。**
  - 换主指标是修订 1 在冒烟之前写定的条件规则，冒烟 0/20 触发。所有登记的指标都报了，不利的结果（R − A −5.0 个百分点，3 行落点 R 更差）也都报了。
  - (i) "决定抓不抓得住的那一下"（夹紧、往下）是事后的划分。登记第 5 行只说"再看落点"，没指定看哪一下。落点的完整答案是 6 行里 R 好 2 行（P&PBox 两下预抓取），差 3 行，分不开 1 行（夹紧右手）。
  - (ii) "R 比 A"那一列是两组中位数之比。夹紧右手这一列写 −41 %，但配对差是 −2.68，区间跨 0，单看这一列容易读成 R 明显更准。
  - (iii) DoubleDesk 的"主指标到底"：两组各 5/60，不是真的 0。它的问题是只有 10 对不一致，测不出中等大小的差，比"到底"更准确的说法是"太少"。
  - (iv) 夹紧的目标本来就设在箱子侧面里 2 cm，手被箱子挡住时落点误差有下限；这几下的落点误差大部分由接触决定（见补充发现 2）。
- **事后挑选：没发现。**
  - 只跑了一次，没有重跑、没有续跑、没有丢局，120 对全进了统计。
  - 取数脚本的提交时间是 20:10:56Z，早于第一个进程结束（20:28:57Z）。那时已有局记录写出来，我没法证实作者没看过；但脚本里的指标和阈值都照登记，没有可调的余地。
  - 结果一节里事后加的几项（摔倒的环节、分组、Fisher、IK 误差）都标明了"不进判定"。
- **判据表的读法：符合登记原文和修订 1。**
  - 第 2、3、6、7 行没触发；第 5、8 行触发。
  - 第 6 行：R − A 正好是 −5.0，满足第一个条件，但 p 0.3449 不满足，所以不触发，读得对。
  - 第 7 行：按修订 1 读作抓起并举高，19 和 13 都超过 2/120，不触发；官方成功另行注明到底，也对。
  - Holm 校正后的 p 不会小于原始 p，这一句也对。

#### 补充发现（事后看的，不进判定，建议写进记录）

1. **差别的区间**：抓起并举高 R − A 的 95 % 区间（按不一致对的精确区间）是 [−13.3, +4.4] 个百分点；P&PBox 单独是 [−22.0, +5.4]，DoubleDesk 是 [−10.4, +10.4]。登记预测的 +3 到 +15 和第 3 行的 +5 门槛，基本落在区间之外。所以数据不只是"没测出差别"，而是对 ScaleBFM 在这套流程里有正收益不利。
2. **落点误差拆成竖直和水平两部分**（中位，cm，手点减目标）：

   | 命令 | 竖直 A | 竖直 R | 水平 A | 水平 R |
   |---|---|---|---|---|
   | P&PBox 预抓取右手 | −22.5 | −4.9 | 19.8 | 19.5 |
   | P&PBox 夹紧右手 | −17.8 | −3.7 | 25.3 | 19.4 |
   | P&PBox 夹紧左手 | −2.7 | −0.2 | 11.4 | 19.2 |
   | DoubleDesk 往下 | +7.5 | +5.8 | 4.0 | 6.5 |

   R 把竖直方向的下垂基本修掉了，但水平偏差没修好，夹紧左手和 DoubleDesk 还变大了。而经过模型的那几下伸手几乎都带接触：A 在 P&PBox 是 229/240 只手，在 DoubleDesk 是 109/120 只手。闸门 (b) 是在干净伸手上算的，说明不了这些伸手；训练数据里有多少这种压在物体上的伸手，我没查。
3. **R 特有的失败**：
   - **合手时箱子被弹起**：R 有 4 局在合手时箱子被弹高 17–69 cm、但没算作抓起（种子 417、420、432、438）；A 没抓起的局里，最大抬高都不超过 7.5 cm。
   - **夹紧目标被抬高**：R 的夹紧命令比感知点高，中位左手 +4.7 cm、右手 +6.1 cm。
   - **机器人跑丢**：P&PBox 里 d3 判"跑丢"的局 R 25 局、A 12 局；在夹紧之前就跑丢的 R 3 局、A 0 局。其中种子 421 和 459 跑丢后，夹紧目标离机器人 1.3–1.6 m，这就是那 21–64 cm IK 误差的来历。
4. **两半种子不一样**：
   - P&PBox 的 R 在 400–429（卡 2）抓起 8/30，在 430–459（卡 3）是 0/30，Fisher p 0.0046。
   - 前一半 A 10、R 8，不一致 8 对 6，基本打平；P&PBox 的 −10 个百分点主要来自后一半（只有 A 抓起的 4 对，只有 R 的 0 对）。
   - 两个任务、两组在后一半都更少抓起（A 从 14 到 5，R 从 12 到 1），但两半的落点误差差不多。
   - 种子范围和显卡分不开。这不破坏同一进程里的配对，但 P&PBox 的 −10 不宜外推。

#### 3. 可能推翻这一读法的实验（只写设计，没有跑）

**实验一：把接触去掉，看决定抓取的那一下 R 是否更准**
- 针对的说法："R 没有更准，说明模型在 HA 的任务状态上修不准"。
- **做法**：
  - 新种子（例如 460–519），每个种子在同一个进程里跑 A、R、A_free、R_free。
  - A_free、R_free 分别和 A、R 完全一样，直到夹紧或往下那一下命令；在那一行把箱子或锤子移到远处（只动仿真里的物体，命令照常算），再执行同一条命令。
  - 在那一行记前缀摘要，要求 A_free 和 A、R_free 和 R 逐位相同。
  - 落点误差的口径照登记：末端标签处的手点到感知目标。DoubleDesk 桌面还在，手碰到桌面的对单独列出。
- **判据**（事先写定）：
  - R_free − A_free 的配对中位差 < 0，区间不含 0，且中位数降低 ≥ 20 %：读作"模型在没有接触时修得准，任务里的落点由接触主导"，推翻原读法。
  - 分不开或 R 更差：原读法成立。

**实验二：先量噪声底，再测只改预抓取的 R**
- 针对的说法："HA 上没测出差别"，以及"限制抓起的不是预抓取的落点"。
- **做法**：
  - 每个种子在同一进程里跑 A、R、A′。A′ 是第二次直接执行，所有的眼都重放 A 的。
  - 另外事先登记一组 R_pg：只有预抓取经过模型（这是 R 落点明确更准的那两下），夹紧和往下照 A 直接执行。
  - 用 240 对，两个任务各 120 对，种子另取，不和 400–459 重叠。
- **判据**：
  - A 对 A′ 在抓起上的不一致对接近 0，说明 R 对 A 的 28 对不一致是真实效应。
  - 在此基础上，R_pg − A ≥ +5 个百分点且 p < 0.05：推翻"没测出差别"（对这个变体）和"限制抓起的不是预抓取落点"。
  - A 对 A′ 的不一致和 R 对 A 差不多：现在的 28 对不一致主要是噪声，原读法更稳。
- **顺带**：把 400–429 和 430–459 换到另一张卡上，判断上面那个两半差异是跟着种子还是跟着卡。

#### 4. 判定

- **"ScaleBFM 在 HA 上没测出差别"：支持。** 主指标、官方成功、各任务的数、McNemar p、摔倒、修订 3 的两项我都独立重算过，全部对上；判据表按修订 1 的读法正确。区间还显示，登记预测的正收益基本被排除了。
- **"决定抓取的那一下 R 没有更准"：作为数字陈述基本支持**（夹紧左手和往下那一下 R 更差，夹紧右手分不开）。
- **由此推出"模型在 HA 的任务状态上修不准"：证据不足。** 这几下几乎都带接触，而 R 修好了竖直方向、没修好水平方向；现有数据分不开"模型不准"和"落点被接触主导"这两种解释。实验一可以把它们分开。
