# selfmodel_jrefwm_20261124：参考来自 cuRobo 关节轨迹的自身模型，在 SIMPLE 站位上采数据，ScaleBFM 和 SONIC 各一份

- 日期：2026-11-24（设计、预测、判据；正式采数据之前写下，单独提交）
- 机器 / GPU：4090，1–3 号卡（0 号卡常有别人的任务）
- 代码版本：`selfwm` @ `d69dadee` 加未提交的文件（正式跑时 `jref_collect.sh` 抄进输出目录的 `asrun/`，md5 记在结果里）：
  - `scripts/experiments/simple_eval_20261112/jref_plan_server.py`（新）：常驻的 cuRobo 规划服务
  - `scripts/experiments/simple_eval_20261112/jref_collect.py`（新）：一个站位上的采样
  - `scripts/experiments/simple_eval_20261112/simple_bench.py`：新加 `--collect`、`--film-ref`
  - `scripts/experiments/simple_eval_20261112/jref_data.py`（新）：录像转成 `via_point.py` 的数据格式；关节范围检查
  - `scripts/experiments/simple_eval_20261112/jref_collect.sh`（新）：4090 上的启动脚本
  - `scripts/experiments/simple_eval_20261112/bench_via.sh`：新加步骤 `JR`
  - 训练和评测用 `scripts/experiments/selfmodel_via_20261120/via_point.py`，一行不改
- 配置：SIMPLE 四个抓取任务的复刻场景，感知栏（`--column perception --cloud cloud_eyes_final`）；G1 加 Dex3；ScaleBFM 和 SONIC（`REACH_CONTROLLER=sonic REACH_SONIC_NETS=gpu`）都不改参数；不用伺服和查表。
- 关联：
  - [selfmodel_jref_20261124](selfmodel_jref_20261124.md)（38.10.2）：cuRobo 的关节轨迹直接当参考（J）之后，手臂加手掌离规划 ScaleBFM 5.0 cm、SONIC 10.3 cm，和路点 B 差不到 1 cm；ScaleBFM 读作两头都不满足，SONIC 读作有缺口。独立复核指出那次的量法有约 4.8 cm 的手形底数；
  - TODO 38.10.3 的原话（数据、训练、当天检验三条）；
  - [selfmodel_via_20261120](selfmodel_via_20261120.md)：`via_point.py` 的模型、误差定义和停止线。

## 假设

38.10.2 说明，把 cuRobo 的关节轨迹直接交给控制器，实际手臂还是偏几厘米。38.10.4 的方案是在下发之前，用自身模型预测"控制器跟这条参考时整条手臂实际在哪"，偏到障碍物上就重新规划。这一步先回答前提：

**用控制器自己执行 cuRobo 关节轨迹采出来的数据训练自身模型，它在 SIMPLE 的真实 J 局上，预测实际手臂的误差能不能明显小于"手臂就在参考上"这个假设本身的误差？**

两个控制器各训练一个模型，各自判。

## 做法

### 数据（每个控制器一份，约一万条）

在 SIMPLE 四个抓取任务的复刻场景里采，起始状态就是 SIMPLE 里发命令时的那种：

1. **留出布局**：120 个公开布局里，`crc32("任务:level:index") % 3 == 0` 的 38 个完全不采（XMovePick 10、XMoveBendPick 8、BendPick 11、TabletopGrasp 9），其余 82 个采。
2. **站位**：每个站位都照基准的流程走过去：感知栏的抓取、同样的站位规则和桌沿限位。只是站位点在发命令时的朝向坐标里前后、左右各加一个 ±6 cm 内的均匀扰动（按站位编号定种子）。到了之后手张到预抓取手形，停 0.3 s，桌子和物体照常有碰撞，右臂被桌沿顶着就顶着。这时记下发命令时的状态，和基准里 J 发命令时一样。
3. **发命令那一刻起，场景不再和机器人碰撞**（桌子、物体的碰撞都关掉，地面不关）。自身模型只描述身体，障碍物不是它的输入。
4. **抽目标和虚拟箱子**，都在发命令时的朝向坐标里：
   - 目标：一半在本局预抓取点附近（每个坐标标准差 5 cm）。另一半在右肩前方一块：前 0.20–0.55 m，左右 −0.40 到 0.10 m，高度在桌面以上 4 cm（至少 0.62 m）到 1.12 m 之间，这是站着、锁着腰的手臂够得到的范围。预抓取点低于这个范围时（Bend 两个任务的矮桌子），不在它附近抽。手腕姿态在本局抓取姿态上加偏航 ±35°、俯仰和横滚各 ±25°。
   - 60 % 的抽样放一块"桌面板"：厚 4 cm，上表面在真桌面高度，近边在手前方 −4 到 12 cm，往前推到离起始的球都有 3.5 cm 以上。这样的路线要从桌沿上方绕过去，和 SIMPLE 的规划一样。
   - 0、1、2 个小箱子（概率 30 / 45 / 25 %），放在手和目标之间，边长 4–20 cm，转 ±45°，离起始的球 3.5 cm 以上、离目标 8 cm 以上。
   - 箱子和板都是虚拟的，只用来让 cuRobo 规划出各种绕法，仿真里没有它们。
5. **规划**：cuRobo 规划右臂 7 个关节，机器人、末端连杆、碰撞球、规划参数都和 `plan2.py` 一样。唯一的区别：腰和右手在规划器里锁在 0，规划的坐标系是"把实际躯干放在腰为 0 时的骨盆"，所以实际的腰角不花代价，规划器只建一次。试跑里规划器算的起始手腕和仿真里差 0.1 mm。每个样本最多抽 20 次，都规划不出来就跳过。
6. **执行**：`Motor.reach_joints`，和 38.10.2 的 J 完全一样：右臂照规划走，其余关节和骨盆保持发命令时的参考不动，按手掌弧长重新定时到 0.15 m/s；走完等手停稳，再停 1 s。
7. **每个站位采两条**：第一条从刚走到的站位开始，和 SIMPLE 的起点一样。第二条先让手臂沿第一条的规划倒着走回去、停 1 s，再抽、再规划、再执行。
8. **数量**：每个布局 63 个站位 × 2 条，82 个布局约 10 300 条（规划不出来的跳过）。两个控制器用同一组种子。ScaleBFM 每个任务分 9 段，共 36 个进程；SONIC 每个任务分 6 段，共 24 个进程（显存）。规划服务每张卡 5 个，共 15 个。

### 转换成训练数据（`jref_data.py build`）

- **参考就是控制器真正收到的参考**：电机的参考姿态，即规划的右臂关节、起点差值的渐变，加上其余身体保持不动，录像里逐帧记下。参考的 24 个球用场景里的机器人算，右手放在**手指的指令**上。38.10.2 用的是规划时那只没张开的手，所以有 4.8 cm 的手形底数，这里没有。参考骨盆在发命令那一刻对齐到真实骨盆，发命令之前里程计的偏差不算控制器的。
- **实际手臂**：仿真里机器人的 24 个球，用真实手指。
- 时间、偏移、误差的定义全照 `via_model.build`：参考时长里等分 24 个时刻，加一个停稳点（最后 0.5 s 的平均）。所有量都在发命令时真实骨盆的朝向坐标里。
- 输入的格式就是 `via_model.build` 的：路点那些字段全填 0，速度 0.15，站立，右手；目标和姿态取参考的最后一个姿态。
- **划分**：训练用的 82 个布局按 crc32 顺序轮流发到 10 个桶，每桶 8 个左右。0–7 号训练；8 号对半分，一半早停、一半校准半径；9 号是测试。自助法按布局重抽。

### 训练

`via_point.py train <模型目录> <数据目录>`，全用默认设置：5 个成员，512×4，dropout 0.1，权重衰减 1e-4，30000 步。然后 `via_point.py dev`，最后 `via_point.py eval`。

### 当天在 SIMPLE 里检验

1. 用 `bench_via.sh` 的新步骤 `JR` 把 38.10.2 的 J 原样重跑一遍，只多加 `--film-ref`，让录像带上参考、行里带上模型需要的输入。每局成功与否、各段偏差都应该和 38.10.2 的 J 一模一样（仿真是确定的）；不一样就先查清楚，不往下做。
2. 有路线的 J 局（ScaleBFM 35 局，SONIC 41 局）的预抓取这一段转成回放数据。共三份：全部；见过的布局（ScaleBFM 27、SONIC 31）；留出的布局（ScaleBFM 8、SONIC 10）。
3. `via_point.py eval <模型> <三份回放>`；`jref_data.py range <训练数据> <回放>`，数发命令时 29 个关节超出训练数据 0.5–99.5 % 范围的个数。
4. SIMPLE 里这一段桌子是有碰撞的，训练数据里没有。这是检验本身的一部分，不另外处理。

## 度量

- **扫过区域误差**（`via_model` 的定义，单位 cm）：实际手臂每一帧的每个球，到它被预测的折线（参考加预测偏移）的最远距离。三种预测对比：
  - "参考"：偏移为 0，即假设手臂就在参考上，误差就是偏差本身；
  - "常数"：训练集的平均偏移，不用模型；
  - "模型"。
- **降幅** = 1 − 模型的中位数 ÷ 参考的中位数，附 95 % 自助区间。开发集和测试集按布局重抽，SIMPLE 回放按局重抽。
- 手掌三个球单独的误差，模型赢的比例，半径和覆盖率（`via_point.py` 本来就报）。
- 发命令时关节超出训练范围的个数：中位数、分布。
- 采样的统计：规划成功率（每次抽样、每个样本）、摔倒数、两条（第一条 / 第二条）各自的误差。
- **主指标照报**：JR 重跑的 SIMPLE 成功局数，45 秒内和官方时限内，重点 TabletopGrasp。按设计应该和 38.10.2 的 J 相同：ScaleBFM 98/63、TabletopGrasp 19/14；SONIC 25/19、TabletopGrasp 6/6。

## 预测

预测写在看过三次试跑之后，没有看过任何模型结果：

- 试跑 1：TabletopGrasp 2 个站位、4 条，发命令时偏移 2–4 cm，终点 5–15 cm；
- 试跑 2：四个任务各一个站位，其他三个任务多数规划不出来，所以改了目标范围；
- 试跑 3：每个任务 5 个站位，抽 12 次时能规划出来的样本 TabletopGrasp 10/10、BendPick 10/10、XMovePick 8/10、XMoveBendPick 6/10，于是改成最多抽 20 次。

| 量 | ScaleBFM | SONIC |
|---|---|---|
| 开发集（8 号桶，全部）的降幅 | 0.40–0.65 | 0.30–0.55 |
| 测试集（9 号桶）的降幅 | 0.40–0.65 | 0.30–0.55 |
| SIMPLE 回放（全部 J 局）的降幅 | 0.20–0.50 | 0.15–0.45 |
| SIMPLE 回放，留出布局的降幅 | 0.10–0.50 | 0.05–0.45 |
| SIMPLE 回放，参考的误差中位数（偏差本身） | 4–8 cm | 7–13 cm |
| 发命令时超出训练范围的关节数（中位数） | 0–1 | 0–2 |
| 每个样本能规划出来的比例（看过试跑 3，不算预测） | 85–98 % | 85–98 % |
| JR 和 J 的行 | 逐局相同 | 逐局相同 |

理由：

- J 的参考比路点那套简单：身体其余部分不动，速度固定。偏差主要来自控制器的接口：ScaleBFM 只读手腕的关键点，手肘由它自己定；SONIC 读 VR 三点，骨盆会漂。这类偏差相当系统，模型应该学得到不少，开发集上降幅要高于 `via_point` 在路点数据上的结果。
- SONIC 的骨盆漂移（38.10.2 中位数 15.5 cm）噪声大，降幅低一档。
- SIMPLE 回放比开发集难：桌子有碰撞，训练里没有；SIMPLE 的规划绕的是真实的桌沿和点云，数据里是虚拟的板和箱子。所以降一档。
- 起始状态就是在 SIMPLE 的站位上采的，超范围的关节应该很少。

## 判据

每个控制器分开判。

1. **停止线**：开发集（8 号桶，全部路径；这里路径都在 1 m 以内，不用原来"超过 1 m"那一档）降幅 ≥ 0.25。不过就停下来报告，不做 SIMPLE 回放之外的任何事。SIMPLE 回放照样跑一遍报数，但不算过关。
2. **SIMPLE 检验**（TODO 原话："预测误差要明显小于偏差本身；发命令时 29 个关节超出训练范围的个数，中位数不超过 2"）：
   - 全部 J 局的降幅 ≥ 0.25，且 95 % 区间下限 > 0；
   - 超范围关节数的中位数 ≤ 2。
   两条都满足算过关；有一条不满足就停下来报告，不往下做。
3. **留出布局**：只有 8–10 局，区间会很宽，不单独当判据。但如果全部局过关，而留出布局的降幅 < 0.10，就报告"只在见过的布局上成立"，由用户定要不要往下做。
4. 无论过不过，都停在这里向用户报告。38.10.4 等用户定。

## 局限（事先写明）

- 训练布局就是 SIMPLE 的公开布局（TODO 允许"直接在带桌子的场景里采"）。见过的布局上，J 局的起点和训练里的站位只差几厘米，目标也在同一个预抓取点附近，所以见过布局上的检验偏乐观。因此单独报留出布局。
- 训练时场景不碰撞，SIMPLE 里有碰撞；接触引起的偏差模型预测不了。
- 第二条样本的起点是"手臂倒着走回去之后"，和 SIMPLE 的起点不完全一样。开发集上分开报第一条和第二条。

## 预测修订

（首次启动后只在这里追加）

## 结果

全部在 4090 上跑，2026-11-24 08:04–11:10（4090 的时钟，UTC）。

代码版本更正：4090 的 checkout 是 `bc426d5` 加未提交的文件，不是头部写的 `d69dadee`。实际跑的代码以 `O/{scalebfm,sonic}/asrun/` 的副本为准，md5 见 Mac 上的 `outputs/jrefwm_20261124/asrun_md5.txt`。`jref_data.py` 的 `build` 部分在三个版本（872dad、87e273、c21421）里相同。

提交进 git 的版本又过了一遍 `ruff format`（`ruff check --fix` 只挪了 `jref_collect.py` 里一行 import 的次序），所以 md5 和 `asrun/` 不同。四个 Python 文件和格式化前比，语法树完全相同；`jref_collect.py` 除 import 次序外相同。输出目录 `/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/jrefwm_20261124/`（下面简写 `O/`）：
- 数据：`O/{scalebfm,sonic}/`，含 rows、films、logs、srv、asrun；
- 训练数据：`O/data_{scalebfm,sonic}/<任务>/`，每个任务一份 table.npz 和 arm.npz；
- 模型：`O/model_{scalebfm,sonic}/`，含 via_point.pt、train.log、dev.log、eval.log、range.txt、carry.txt、stats.txt；
- SIMPLE 回放：`O/check_{scalebfm,sonic}/simple_{all,seen,held}/`；
- JR 重跑：`outputs/simple_eval_20261112/bench_jref_20261124/{scalebfm,sonic}/rows/JR_*.jsonl` 和 `films_JR/`。

Mac 上的副本在 `outputs/jrefwm_20261124/`（日志、模型、ScaleBFM 的回放数据、脚本）。

### 跑的时候改了什么（都是程序错误或资源问题，没有改任何设置）

1. **第一次启动（08:04）规划服务起不来**：socket 路径超过 AF_UNIX 的 108 字符上限。改到 `/tmp/jrefwm_<控制器>/`，那次启动的空目录挪到 `O/scalebfm_sockfail_20261124T0806`。
2. **ScaleBFM 采集中 5 个进程因 CUDA 显存不够退出**。原因是我在 08:14–08:31 同时跑了 SONIC 的 JR 重跑，卡被占满。处理：
   - 新加 `simple_bench.py --skip-done`，用一次性的 `O/scalebfm/jref_resume.sh`（09:03–09:12，35 个进程，每个跑一个站位编号）补齐，已经写完的站位跳过；
   - 每个站位的结果由它的编号决定，所以补跑的站位和不中断时的一样；
   - 最后每个任务的站位数都等于布局数 × 63，没有重复：XMovePick 1260、XMoveBendPick 1386、BendPick 1197、TabletopGrasp 1323。
3. **SONIC 的 JR 重跑里 XMovePick 那个进程启动时显存不够**。之后单独重跑，分 3 个进程，再按局号顺序拼成 `rows/JR_XMovePickTeleop.jsonl`：`bench_jref_20261124/sonic/jr_xmp_sonic.sh`。
4. **SONIC 的规划服务每张卡 3 个**（登记的是 5 个），24 个采集进程不变。原因是显存：SONIC 每个进程约 1.7 GB，5 个服务时每张卡约 21 GB，卡是 22 GB。
5. **`jref_data.py` 的分桶改成在全部 82 个训练布局上发牌**，不再只用这份数据里有的布局。这样按任务分开建的四份数据，拼起来的划分和一次建成的一样。改在建任何训练数据之前，SIMPLE 回放不受影响（一律 9 号桶）。`range` 子命令改成可以收多份训练数据。

### 采集（`stats.txt`）

```
scalebfm BendPickMP: samples 2394, planned 2332 (0.974), draws 13435 (succeeded 0.173), near-pregrasp goals of the planned 0.28, fell 0
scalebfm TabletopGraspMP: samples 2646, planned 2646 (1.000), draws 4644 (succeeded 0.551), near-pregrasp goals of the planned 0.67, fell 0
scalebfm XMoveBendPickTeleop: samples 2772, planned 2609 (0.941), draws 18949 (succeeded 0.136), near-pregrasp goals of the planned 0.09, fell 0
scalebfm XMovePickTeleop: samples 2520, planned 2417 (0.959), draws 13560 (succeeded 0.177), near-pregrasp goals of the planned 0.30, fell 0
scalebfm all: samples 10332, planned 10004 (0.968), draw success 0.195, fell 0
scalebfm validation, sample 0 of a stance: 484 paths, arm median reference 5.54 cm, model 0.65 cm, drop 0.88
scalebfm validation, sample 1 of a stance: 486 paths, arm median reference 5.14 cm, model 0.62 cm, drop 0.88
sonic BendPickMP: samples 2392, planned 2354 (0.984), draws 11305 (succeeded 0.207), near-pregrasp goals of the planned 0.25, fell 1
sonic TabletopGraspMP: samples 2630, planned 2630 (1.000), draws 4695 (succeeded 0.552), near-pregrasp goals of the planned 0.61, fell 2
sonic XMoveBendPickTeleop: samples 2771, planned 2533 (0.914), draws 19776 (succeeded 0.128), near-pregrasp goals of the planned 0.12, fell 0
sonic XMovePickTeleop: samples 2520, planned 2495 (0.990), draws 9537 (succeeded 0.261), near-pregrasp goals of the planned 0.53, fell 0
sonic all: samples 10313, planned 10012 (0.971), draw success 0.219, fell 3
sonic validation, sample 0 of a stance: 486 paths, arm median reference 12.95 cm, model 4.73 cm, drop 0.63
sonic validation, sample 1 of a stance: 494 paths, arm median reference 11.43 cm, model 3.29 cm, drop 0.71
```

可用的路径（规划出来、没摔）：ScaleBFM 10004 条，SONIC 10009 条。

### JR 和 38.10.2 的 J 逐字段对比（`compare_jr.py`，除新加的 S3、W、J、t_cmd 外每个字段，wall_s 不比）

```
ScaleBFM
BendPickMP: JR 30 rows, success 45 s 30, official 0
TabletopGraspMP: JR 30 rows, success 45 s 19, official 14
XMoveBendPickTeleop: JR 30 rows, success 45 s 25, official 25
XMovePickTeleop: JR 30 rows, success 45 s 24, official 24
identical 120, differing 0
SONIC
BendPickMP: JR 30 rows, success 45 s 6, official 0
TabletopGraspMP: JR 30 rows, success 45 s 6, official 6
XMoveBendPickTeleop: JR 30 rows, success 45 s 4, official 4
XMovePickTeleop: JR 30 rows, success 45 s 9, official 9
identical 120, differing 0
```

### 训练（`train.log`）

- ScaleBFM：路径 10004 条，其中训练 8047、早停 592、校准 378、测试 987；不用模型时验证集的偏移平方 4.40 cm²/坐标，5 个成员早停时 0.157–0.175。
- SONIC：路径 10009 条，其中训练 8054、早停 602、校准 378、测试 975；不用模型时 42.79 cm²，成员 4.80–5.53。

### 停止线、测试集、SIMPLE 回放（`dev.log`、`eval.log` 原文）

```
=== scalebfm dev
validation, all: 970 paths
  reference  swept-region error: arm median  5.36 p90 10.58 cm | hand median  3.53 cm | margin (median radius)  6.59 cm, inside 0.95 | fixed radius 12.40 cm, inside 0.95
  constant   swept-region error: arm median  4.57 p90  9.19 cm | hand median  2.79 cm | margin (median radius)  6.09 cm, inside 0.96 | fixed radius 10.46 cm, inside 0.93
  direction  swept-region error: arm median  0.64 p90  1.70 cm | hand median  0.46 cm | margin (median radius)  1.10 cm, inside 0.94 | fixed radius  2.87 cm, inside 0.95
  drop of the arm's median against the reference: 0.88 (95 % interval 0.87 to 0.89, resampling routes); hand 0.87; constant: arm 0.15
  fixed radius: direction / reference 0.23 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing: 970 paths
validation, standing, palm path over 1.0 m (THE STOP LINE: drop of the arm's median >= 0.25): 38 paths
  reference  swept-region error: arm median  8.85 p90 20.38 cm | hand median  6.02 cm | margin (median radius) 10.36 cm, inside 0.92 | fixed radius 12.40 cm, inside 0.68
  constant   swept-region error: arm median  6.99 p90 20.69 cm | hand median  4.42 cm | margin (median radius)  8.59 cm, inside 0.95 | fixed radius 10.46 cm, inside 0.68
  direction  swept-region error: arm median  1.32 p90  5.64 cm | hand median  1.01 cm | margin (median radius)  1.97 cm, inside 0.87 | fixed radius  2.87 cm, inside 0.68
  drop of the arm's median against the reference: 0.85 (95 % interval 0.61 to 0.86, resampling routes); hand 0.83; constant: arm 0.21
  fixed radius: direction / reference 0.23 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, palm path under 0.6 m: 626 paths
=== scalebfm eval
test, ordinary hold-out: 987 paths
  reference  swept-region error: arm median  6.23 p90 10.99 cm | hand median  4.05 cm | margin (median radius)  7.83 cm, inside 0.94 | fixed radius 12.40 cm, inside 0.94
  constant   swept-region error: arm median  5.14 p90 10.01 cm | hand median  3.32 cm | margin (median radius)  6.65 cm, inside 0.94 | fixed radius 10.46 cm, inside 0.91
  direction  swept-region error: arm median  0.79 p90  1.93 cm | hand median  0.55 cm | margin (median radius)  1.36 cm, inside 0.94 | fixed radius  2.87 cm, inside 0.94
  drop of the arm's median against the reference: 0.87 (95 % interval 0.87 to 0.88, resampling routes); hand 0.86; constant: arm 0.18
  fixed radius: direction / reference 0.23 (set on the calibration paths); direction wins on 1.00 of the paths
test, sector hold-out: 0 paths, too few
real paths replayed, simple_all: 35 paths
  reference  swept-region error: arm median  7.90 p90 12.19 cm | hand median  5.87 cm | margin (median radius)  8.68 cm, inside 0.91 | fixed radius 12.40 cm, inside 0.94
  constant   swept-region error: arm median  5.44 p90 10.08 cm | hand median  4.18 cm | margin (median radius)  7.77 cm, inside 0.97 | fixed radius 10.46 cm, inside 0.97
  direction  swept-region error: arm median  2.82 p90  4.02 cm | hand median  2.58 cm | margin (median radius)  1.22 cm, inside 0.34 | fixed radius  2.87 cm, inside 0.54
  drop of the arm's median against the reference: 0.64 (95 % interval 0.58 to 0.83, resampling paths); hand 0.56; constant: arm 0.31
  fixed radius: direction / reference 0.23 (set on the calibration paths); direction wins on 1.00 of the paths
real paths replayed, simple_all, walked first: 0 paths, too few
real paths replayed, simple_seen: 27 paths
  reference  swept-region error: arm median  7.90 p90 11.90 cm | hand median  5.85 cm | margin (median radius)  8.63 cm, inside 0.89 | fixed radius 12.40 cm, inside 0.96
  constant   swept-region error: arm median  5.44 p90  9.92 cm | hand median  4.15 cm | margin (median radius)  7.75 cm, inside 0.96 | fixed radius 10.46 cm, inside 1.00
  direction  swept-region error: arm median  2.82 p90  3.76 cm | hand median  2.64 cm | margin (median radius)  1.20 cm, inside 0.33 | fixed radius  2.87 cm, inside 0.52
  drop of the arm's median against the reference: 0.64 (95 % interval 0.58 to 0.84, resampling paths); hand 0.55; constant: arm 0.31
  fixed radius: direction / reference 0.23 (set on the calibration paths); direction wins on 1.00 of the paths
real paths replayed, simple_seen, walked first: 0 paths, too few
real paths replayed, simple_held: 8 paths
  reference  swept-region error: arm median  7.68 p90 12.41 cm | hand median  6.08 cm | margin (median radius)  9.99 cm, inside 1.00 | fixed radius 12.40 cm, inside 0.88
  constant   swept-region error: arm median  5.50 p90 10.48 cm | hand median  4.20 cm | margin (median radius)  8.84 cm, inside 1.00 | fixed radius 10.46 cm, inside 0.88
  direction  swept-region error: arm median  2.69 p90  4.43 cm | hand median  2.24 cm | margin (median radius)  1.53 cm, inside 0.38 | fixed radius  2.87 cm, inside 0.62
  drop of the arm's median against the reference: 0.65 (95 % interval 0.42 to 0.87, resampling paths); hand 0.63; constant: arm 0.28
  fixed radius: direction / reference 0.23 (set on the calibration paths); direction wins on 1.00 of the paths
real paths replayed, simple_held, walked first: 0 paths, too few
=== sonic dev
validation, all: 980 paths
  reference  swept-region error: arm median 12.15 p90 20.60 cm | hand median 10.17 cm | margin (median radius) 16.55 cm, inside 0.94 | fixed radius 22.42 cm, inside 0.93
  constant   swept-region error: arm median 11.86 p90 19.53 cm | hand median  9.94 cm | margin (median radius) 16.04 cm, inside 0.94 | fixed radius 20.34 cm, inside 0.92
  direction  swept-region error: arm median  3.81 p90  8.64 cm | hand median  2.93 cm | margin (median radius)  6.48 cm, inside 0.95 | fixed radius 10.29 cm, inside 0.94
  drop of the arm's median against the reference: 0.69 (95 % interval 0.66 to 0.70, resampling routes); hand 0.71; constant: arm 0.02
  fixed radius: direction / reference 0.46 (set on the calibration paths); direction wins on 0.99 of the paths
validation, standing: 980 paths
validation, standing, palm path over 1.0 m (THE STOP LINE: drop of the arm's median >= 0.25): 26 paths
  reference  swept-region error: arm median 13.98 p90 22.97 cm | hand median 11.77 cm | margin (median radius) 18.91 cm, inside 0.92 | fixed radius 22.42 cm, inside 0.85
  constant   swept-region error: arm median 14.20 p90 19.99 cm | hand median 11.77 cm | margin (median radius) 19.07 cm, inside 0.92 | fixed radius 20.34 cm, inside 0.88
  direction  swept-region error: arm median  5.00 p90 13.48 cm | hand median  3.80 cm | margin (median radius)  9.33 cm, inside 0.85 | fixed radius 10.29 cm, inside 0.85
  drop of the arm's median against the reference: 0.64 (95 % interval 0.57 to 0.70, resampling routes); hand 0.68; constant: arm -0.02
  fixed radius: direction / reference 0.46 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, palm path under 0.6 m: 469 paths
=== sonic eval
test, ordinary hold-out: 975 paths
  reference  swept-region error: arm median 12.67 p90 21.34 cm | hand median 10.53 cm | margin (median radius) 17.38 cm, inside 0.93 | fixed radius 22.42 cm, inside 0.92
  constant   swept-region error: arm median 11.92 p90 19.98 cm | hand median 10.00 cm | margin (median radius) 16.61 cm, inside 0.93 | fixed radius 20.34 cm, inside 0.91
  direction  swept-region error: arm median  3.96 p90  8.93 cm | hand median  3.12 cm | margin (median radius)  6.90 cm, inside 0.93 | fixed radius 10.29 cm, inside 0.93
  drop of the arm's median against the reference: 0.69 (95 % interval 0.65 to 0.71, resampling routes); hand 0.70; constant: arm 0.06
  fixed radius: direction / reference 0.46 (set on the calibration paths); direction wins on 0.98 of the paths
test, sector hold-out: 0 paths, too few
real paths replayed, simple_all: 41 paths
  reference  swept-region error: arm median 14.13 p90 30.04 cm | hand median 11.69 cm | margin (median radius) 18.84 cm, inside 0.83 | fixed radius 22.42 cm, inside 0.76
  constant   swept-region error: arm median 12.73 p90 28.21 cm | hand median 10.60 cm | margin (median radius) 18.23 cm, inside 0.85 | fixed radius 20.34 cm, inside 0.76
  direction  swept-region error: arm median  6.96 p90 19.43 cm | hand median  4.54 cm | margin (median radius)  9.99 cm, inside 0.83 | fixed radius 10.29 cm, inside 0.66
  drop of the arm's median against the reference: 0.51 (95 % interval 0.40 to 0.65, resampling paths); hand 0.61; constant: arm 0.10
  fixed radius: direction / reference 0.46 (set on the calibration paths); direction wins on 0.95 of the paths
real paths replayed, simple_all, walked first: 0 paths, too few
real paths replayed, simple_seen: 31 paths
  reference  swept-region error: arm median 14.13 p90 28.67 cm | hand median 11.69 cm | margin (median radius) 18.84 cm, inside 0.81 | fixed radius 22.42 cm, inside 0.74
  constant   swept-region error: arm median 12.73 p90 27.86 cm | hand median 10.42 cm | margin (median radius) 18.23 cm, inside 0.84 | fixed radius 20.34 cm, inside 0.74
  direction  swept-region error: arm median  6.14 p90 19.43 cm | hand median  4.37 cm | margin (median radius) 10.49 cm, inside 0.81 | fixed radius 10.29 cm, inside 0.65
  drop of the arm's median against the reference: 0.57 (95 % interval 0.29 to 0.67, resampling paths); hand 0.63; constant: arm 0.10
  fixed radius: direction / reference 0.46 (set on the calibration paths); direction wins on 0.94 of the paths
real paths replayed, simple_seen, walked first: 0 paths, too few
real paths replayed, simple_held: 10 paths
  reference  swept-region error: arm median 15.28 p90 30.60 cm | hand median 13.42 cm | margin (median radius) 20.71 cm, inside 0.90 | fixed radius 22.42 cm, inside 0.80
  constant   swept-region error: arm median 13.50 p90 28.57 cm | hand median 11.53 cm | margin (median radius) 18.34 cm, inside 0.90 | fixed radius 20.34 cm, inside 0.80
  direction  swept-region error: arm median  7.56 p90 16.93 cm | hand median  5.42 cm | margin (median radius)  9.67 cm, inside 0.90 | fixed radius 10.29 cm, inside 0.70
  drop of the arm's median against the reference: 0.51 (95 % interval 0.35 to 0.70, resampling paths); hand 0.60; constant: arm 0.12
  fixed radius: direction / reference 0.46 (set on the calibration paths); direction wins on 1.00 of the paths
real paths replayed, simple_held, walked first: 0 paths, too few
```

### 发命令时 29 个关节超出训练范围的个数（`range.txt`，训练范围取四个任务全部训练路径的 0.5–99.5 %）

```
ScaleBFM
joints at the command outside the training paths' 0.5-99.5 % band (35 reaches): median 0.0, mean 0.51, max 7, counts [28, 3, 2, 0, 1, 0, 0, 1]
  seen layouts: 27 reaches, median 0.0
  held-out layouts: 8 reaches, median 0.0
joints at the command outside the training paths' 0.5-99.5 % band (8 reaches): median 0.0, mean 0.75, max 4, counts [6, 0, 1, 0, 1]
joints at the command outside the training paths' 0.5-99.5 % band (27 reaches): median 0.0, mean 0.44, max 7, counts [22, 3, 1, 0, 0, 0, 0, 1]
SONIC
joints at the command outside the training paths' 0.5-99.5 % band (41 reaches): median 0.0, mean 0.54, max 6, counts [30, 7, 2, 0, 0, 1, 1]
  seen layouts: 31 reaches, median 0.0
  held-out layouts: 10 reaches, median 0.0
joints at the command outside the training paths' 0.5-99.5 % band (10 reaches): median 0.0, mean 0.20, max 1, counts [8, 2]
joints at the command outside the training paths' 0.5-99.5 % band (31 reaches): median 0.0, mean 0.65, max 6, counts [22, 5, 2, 0, 0, 1, 1]
```

range.txt 第二、三行依次是留出布局和见过布局的回放。

### 诊断（没有预注册）：把发命令时的偏差原样带到底

实际手臂发命令时离参考的偏差，用机器人自己的关节读数就能算出来，不需要模型。这个预测器把这一偏差不变地加到整条参考上（`carry_baseline.py`）：

```
ScaleBFM
validation: 970 paths | arm median cm: reference 5.36 (drop 0.00), constant 4.57 (drop 0.15), direction 0.64 (drop 0.88), carry 5.61 (drop -0.05) | model / carry 0.11, model beats carry on 1.00
test: 987 paths | arm median cm: reference 6.23 (drop 0.00), constant 5.14 (drop 0.18), direction 0.79 (drop 0.87), carry 6.30 (drop -0.01) | model / carry 0.12, model beats carry on 1.00
simple_all: 35 paths | arm median cm: reference 7.90 (drop 0.00), constant 5.44 (drop 0.31), direction 2.82 (drop 0.64), carry 6.96 (drop 0.12) | model / carry 0.41, model beats carry on 1.00
simple_seen: 27 paths | arm median cm: reference 7.90 (drop 0.00), constant 5.44 (drop 0.31), direction 2.82 (drop 0.64), carry 7.29 (drop 0.08) | model / carry 0.39, model beats carry on 1.00
simple_held: 8 paths | arm median cm: reference 7.68 (drop 0.00), constant 5.50 (drop 0.28), direction 2.69 (drop 0.65), carry 6.89 (drop 0.10) | model / carry 0.39, model beats carry on 1.00
SONIC
validation: 980 paths | arm median cm: reference 12.15 (drop 0.00), constant 11.86 (drop 0.02), direction 3.81 (drop 0.69), carry 9.62 (drop 0.21) | model / carry 0.40, model beats carry on 0.98
test: 975 paths | arm median cm: reference 12.67 (drop 0.00), constant 11.92 (drop 0.06), direction 3.96 (drop 0.69), carry 9.92 (drop 0.22) | model / carry 0.40, model beats carry on 0.98
simple_all: 41 paths | arm median cm: reference 14.13 (drop 0.00), constant 12.73 (drop 0.10), direction 6.96 (drop 0.51), carry 10.53 (drop 0.25) | model / carry 0.66, model beats carry on 1.00
simple_seen: 31 paths | arm median cm: reference 14.13 (drop 0.00), constant 12.73 (drop 0.10), direction 6.14 (drop 0.57), carry 10.53 (drop 0.25) | model / carry 0.58, model beats carry on 1.00
simple_held: 10 paths | arm median cm: reference 15.28 (drop 0.00), constant 13.50 (drop 0.12), direction 7.56 (drop 0.51), carry 11.22 (drop 0.27) | model / carry 0.67, model beats carry on 1.00
```


## 意外

1. **四个降幅全部高于预测区间的上限。**

   | 降幅 | ScaleBFM 实测 | ScaleBFM 预测 | SONIC 实测 | SONIC 预测 |
   |---|---|---|---|---|
   | 开发集 | 0.88 | 0.40–0.65 | 0.69 | 0.30–0.55 |
   | SIMPLE 回放 | 0.64 | 0.20–0.50 | 0.51 | 0.15–0.45 |
   | SIMPLE 留出布局 | 0.65 | 0.10–0.50 | 0.51 | 0.05–0.45 |

   测试集和开发集一样。我对"控制器跟关节轨迹的偏差有多系统"一贯偏悲观，大约偏了 0.2。
2. **SONIC 在 SIMPLE 上参考的误差是 14.13 cm，高于预测的 7–13 cm。** ScaleBFM 是 7.90 cm，在预测的 4–8 cm 里，贴着上沿。超范围关节数（中位数都是 0）、JR 和 J 逐局相同，都和预测一致。
3. **登记的"参考"不是最强的不用模型的对手（复核发现）。**
   - 原因：参考在发命令时连倾斜一起对齐到真实骨盆，而 SIMPLE 里发命令时骨盆的倾斜（中位数 3.6° / 5.3°）比训练里大（1.0° / 3.0°）。
   - 换成不对齐的电机参考，再加开发集上拟合的常数偏移：模型的降幅 ScaleBFM 是 0.54，SONIC 只有 0.35 [0.19, 0.56]。
   - 对所有不用模型的对手，ScaleBFM 都在 0.48 以上。
4. **训练和 SIMPLE 的规划起点不一样（复核发现）。**
   - 训练里 cuRobo 总从真实手臂规划：规划第一行离真实手臂的中位数是 0°。
   - 38.10.2 留下的 J 规划是从 A 那一遍的起点规划的，离这一遍的真实手臂 8.6° / 10.3°（最大 22.9° / 32.8°）。
   - 所以"起始状态就是 SIMPLE 里那种"只对关节读数成立，对"参考相对实际手臂从哪出发"不成立。这让检验更难。
5. **真实局比数据难一截。**
   - 模型误差：开发集 0.64 cm → SIMPLE 2.82 cm（ScaleBFM，4.4 倍）；3.81 cm → 6.96 cm（SONIC，1.8 倍）。
   - 模型自带的误差半径在 SIMPLE 上只罩住 34 % / 83 % 的局，标定目标是 95 %。
6. **分任务差别大。** ScaleBFM 在 XMovePick 7 局上降 0.91，TabletopGrasp 28 局上降 0.63；SONIC 在 TabletopGrasp 上降 0.46，XMovePick 和 XMoveBendPick 上都是 0.70。
7. **把发命令时的偏差带到底，两边不一样。** 在 ScaleBFM 上几乎没用（开发集 −0.05，SIMPLE 0.12）；在 SONIC 上降 0.21–0.27，SONIC 发命令时的偏差有一部分会一直保持。
8. **其他。**
   - SONIC 每个站位的第一条比第二条难：降幅 0.63 对 0.71，参考的误差 12.95 对 11.43 cm。ScaleBFM 两条一样，都是 0.88。
   - 每次抽目标的规划成功率只有 13–55 %，抽 20 次后每个样本 91–100 %。
   - SONIC 采集 94 分钟，比我估计的 3 小时快。

## 结论与下一步

**结论**（每条后面是证据所在的位置）：

1. **ScaleBFM：通过 38.10.3 的全部判据，复核支持。**
   - 开发集降 0.88 [0.87, 0.89]（dev.log）。
   - SIMPLE 有路线的 J 局 35 局：手臂扫过区域误差的中位数从 7.90 cm 降到 2.82 cm，降 0.64 [0.58, 0.83]（eval.log）。
   - 超范围关节的中位数是 0（range.txt）。
   - 对复核造的所有不用模型的对手，降幅都在 0.48 以上（复核 `alt_ref_val.out`、`alt_const.out`）；从没训练过的布局 16 局，降 0.65（`seen_split.out`）。
2. **SONIC：通过全部判据，复核支持，但幅度要打折。**
   - 开发集降 0.69 [0.66, 0.70]。
   - SIMPLE 41 局：14.13 cm 降到 6.96 cm，降 0.51 [0.40, 0.65]。
   - 对最强的不用模型的对手（不对齐的电机参考加常数），降幅只有 0.35 [0.19, 0.56]，模型赢 78 % 的局。从没训练过的布局降 0.48。
   - 模型在 SIMPLE 上的 p90 还有 19.4 cm。
3. **适用范围：SIMPLE 有路线的局，以 TabletopGrasp 为主**（各 28 局），另有 XMovePick 7 / 8 局、SONIC 的 XMoveBendPick 5 局。两个 Bend 任务没有证据。

**证据强度**
- 每个控制器一次采集、一个模型（5 个成员的集成，种子 0）。
- 开发集和测试集各约 980 条路径，但只来自 8 个布局，区间偏窄。
- SIMPLE 检验 35 / 41 局，每局一次运行；仿真是确定的，重跑逐帧相同。

**其他可能的解释**
1. 降幅里有一部分是模型在纠正"参考"这个定义自带的倾斜误差，而不是在预测控制器的行为。复核的对手表把这部分剥掉之后，ScaleBFM 还剩 0.48 以上，SONIC 还剩 0.35。
2. 见过布局的检验局，起点和训练样本很近。但从没训练过的布局也一样（ScaleBFM 0.65），SONIC 低一档（0.48）。

**四类陷阱**
- **基准选择**：检验局是 J 里全部有路线的局，事先登记，一局没丢。覆盖窄：以 TabletopGrasp 为主，没有 Bend 任务。
- **数据泄漏**：没有发现。
  - 输入都在发命令那一刻或之前；
  - 按布局划分属实，SIMPLE 的检验局没有进训练；
  - 早停用的是开发集，但测试集结果一样。
- **指标误用**：没有占便宜。
  - 换成反向、对称、按时刻对齐的误差，降幅都没塌（0.53–0.67 / 0.50–0.55）；
  - 登记的参考不是最强对手，已经把更强的对手一并报出。
- **事后选择**：没有影响判定的改动。
  - 唯一在看结果之后改的代码是 `jref_data.py range`，改成可以收多份训练数据。换两种做法重算，中位数仍 ≤ 2。
  - "带到底"和复核的对手都标明是非预注册的。

**判据里的一处错误**：判据第 1 条写"这里路径都在 1 m 以内"，不对。开发集里超过 1 m 的有 38 / 26 条，那一档也过了（0.85 / 0.64）。判据已经锁定，不改原文，记在这里。

**对照判据**：两个控制器都落在"过关"一侧，两个留出布局的提示项也没有触发。按判据第 4 条，停在这里向用户报告，38.10.4 等用户定。

**如果做 38.10.4，要先处理这几件事：**
1. **误差半径先在 SIMPLE 那种数据上重新标定**，再拿来当避障余量。现在只罩住 34 % / 83 %。
2. **38.10.4 里规划会从发命令时的真实手臂出发**，训练和检验在规划起点上的差别自然就没了。复核提出的推翻实验 1 就是这个设置：每一局在发命令时重新规划，用真实桌子和点云，覆盖 120 个布局，补上 Bend 任务。可以当 38.10.4 的第一步。
3. **参考骨盆对齐时去掉倾斜、只对齐偏航**。这要先在开发集上定，不在测试局上定。
4. **ScaleBFM 的 2.8 cm 可以拿来对照点云检查。SONIC 的 7 cm（p90 19 cm）偏大**：要么只在 ScaleBFM 上做，要么 SONIC 先补数据（比如规划从和 SIMPLE 同样的偏离出发）。

## 独立复核

一个独立的 general-purpose agent，只拿到这份条目（预注册各节和结果）、Mac 上的原始输出和代码 diff，以及 4090 的只读访问。下面是它的回复原文。

**结论：两个控制器都按预注册判据"支持"。** SONIC 的领先幅度被参考的定义放大了。另外，SIMPLE 检验基本只覆盖 TabletopGrasp，这个范围要写进结论。复核没有发现数字错误，也没有发现泄漏或转换器的程序错误。

## 0. 做了什么、怎么做的

- 4090 上只读。我只在 `outputs/simple_eval_20261112/jrefwm_20261124/review/` 下写了脚本和输出。全程用 CPU（`CUDA_VISIBLE_DEVICES=` 为空），没碰任何 GPU，没动任何进程，最长一次跑了 77 s。
- Mac 上脚本只放在 `/tmp/review_jrefwm/`，仓库里没改任何东西。
- 我没有调用 `via_point.opened()`，因为它会以写方式打开模型目录里的日志。脚本里另写了只读的加载。

```
cd /wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=outputs/simple_eval_20261112/jrefwm_20261124; B=outputs/simple_eval_20261112/bench_jref_20261124
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE REACH_HAND_OFF=0.0415,0,0
$PY $O/review/recompute.py $O                                    # 结果里的数字重算
$PY -I $O/review/jr_check.py $B                                   # JR 和 J：行和录像逐帧对比
$PY $O/review/alt_ref.py $O $B outputs/scene/simple_g1_sonic val  # 从原始录像重建，加上不用模型的其他参考
$PY $O/review/metrics.py $O                                       # 换度量、泄漏、关节范围、站位
$PY $O/review/seen_split.py $O                                    # "见过的布局"按桶拆开
$PY $O/review/alt_const.py $O $B outputs/scene/simple_g1_sonic    # 其他参考加常数偏移
$PY $O/review/last.py $O $B                                       # 分任务、采集计数、缺的录像
```

## 1. 数字核对

**预注册的时间对得上。**
- 4e2630bd 提交于 16:03:40 +0800，即 08:03:40 UTC。4090 用的是 UTC，第一次启动在 08:04，`asrun/` 抄于 08:05:24。
- ScaleBFM 的 JR 重跑在 08:07–08:13，SONIC 的在 08:14–09:15，都在提交之后。
- 和 4e2630bd 相比，条目只有"结果"一节有改动，假设、做法、度量、预测、判据、局限一字未动。

**代码版本对得上。**
- `asrun/` 里的 md5 和 Mac 工作区的文件一致，`via_point.py`、`via_model.py` 没有改。
- 4090 的 git HEAD 是 bc426d5，不是条目写的 d69dadee。但实际跑的代码有 `asrun/` 的副本和 md5，不影响结论。
- `jref_data.py` 有三个版本：ScaleBFM asrun 872dad、SONIC asrun 87e273、仓库里的 c21421。三者只在"分桶发牌"和 `range` 子命令上不同，`build` 的参考、偏移、输入部分完全一样。
- 我从原始录像重建了 SIMPLE 回放表和开发集，和存的表比，参考、偏移、实际球最大差 6e-8 m。所以检验数据和训练数据确实是同一个转换器产出的。

**结果里的每个关键数都对得上**（`recompute.out`）：
- 停止线（8 号桶，全部）：ScaleBFM 0.88 [0.87, 0.89]，SONIC 0.69 [0.66, 0.70]。
- 测试集（9 号桶）：0.87 / 0.69。
- SIMPLE 回放：
  - 全部：0.64 [0.58, 0.83] / 0.51 [0.40, 0.65]；
  - 见过的布局：0.64 / 0.57；
  - 留出的布局：0.65 / 0.51。
- 参考的误差中位数：ScaleBFM 7.90 cm，SONIC 14.13 cm。
- 超范围关节数的中位数：都是 0，各档分布也一样。
- 第一条 / 第二条样本分开报：0.88 / 0.88 和 0.63 / 0.71。

**采集统计对得上。**
- 采样 10332 / 10313 条，规划出来 10004 / 10012 条，摔倒 0 / 3 次，可用 10004 / 10009 条。
- 每个任务的站位数 = 布局数 × 63，没有重复的站位，没有混进试跑的数据。
- 训练日志里的路径数（8047 / 592 / 378 / 987 等）、4.40 cm² 和 42.79 cm²、成员早停时的 0.157–0.175 和 4.80–5.53，都和日志一致。

**JR 和 J 一模一样。**
- 120 行逐字段相同（递归比较，只去掉新加的 S3、W、J、t_cmd 四个字段）。
- 119 对录像的 qpos 逐帧完全相同。缺的那一局（TabletopGrasp L1 e2）在 J 和 JR 里都没有任何阶段。
- 成功局数 98/63、25/19，和登记的一致。
- J 参考的实际速度在 0.1498–0.1503 m/s，和训练用的 0.15 一致。
- 有路线的局 35 / 41 全部进了回放，没有因为摔倒或时长过滤丢掉任何一局。

**一处文字错误，不影响结论。** 判据写"这里路径都在 1 m 以内"，但 dev.log 里超过 1 m 的有 38 / 26 条。停止线按的是全部路径；而超过 1 m 的那一档也过了（0.85 和 0.64）。

## 2. 四类陷阱

### 2.1 基准选择

- **检验局没有被挑过。** 检验局就是 J 里全部有路线的局，局数事先登记了，一局没丢。
- **但覆盖很窄，这是最大的问题。**

  | 控制器 | 回放局的构成 | 分任务降幅 |
  |---|---|---|
  | ScaleBFM（35 局） | TabletopGrasp 28、XMovePick 7，两个 Bend 任务 0 局 | 0.63 / 0.91 |
  | SONIC（41 局） | TabletopGrasp 28、XMovePick 8、XMoveBendPick 5，BendPick 0 局 | 0.46 / 0.70 / 0.70 |

  所以"SIMPLE 的 J 局"实际上主要是 TabletopGrasp。结论应写成"在 SIMPLE 有路线的 J 局上（以 TabletopGrasp 为主）"。在 Bend 任务上没有任何证据。
- **开发集和测试集的区间偏窄。** 按布局重抽只有 8 个单位，0.87–0.89 这种区间低估了不确定性。好在效应远大于判据。

### 2.2 数据泄漏：没有发现

**模型的输入都在发命令那一刻或之前。** `via_point.features` 用到的是：
- `x` 的前 213 列和 373–378 列：S3（发命令时、前 0.1 s、前 0.2 s 的 29 个关节、关节速度、重力方向、角速度、高度）、发命令时的手爪位置和姿态、目标和目标姿态、速度、手、参考时长；
- 参考本身算出来的量：5 个关键球的参考位置和前后差、手掌走过和剩下的路程。

逐项核过：
- Recorder 每 5 个 tick 采一次，PRE_S 0.3 s 正好是 15 个 tick，所以 S3 的最后一帧就是发命令那一刻。
- W 取自 `env.pelvis()`，即仿真的真实骨盆。
- 基准里记 t_cmd 到调用 `reach_joints` 之间没有 tick（method 是 direct）。
- 采集里规划期间仿真时间不走。
- `reach_joints` 执行时参考骨盆固定为 `base`，参考里没有发命令之后的信息。

**SIMPLE 的检验局没有进训练。**
- 训练表里 route ≥ 0 的行占 100 %（全部来自采集），留出布局的行 0 条。
- 回放表一律是 9 号桶、route = −1。
- 没有哪个布局同时落在两个桶里。按布局划分属实。

**要注意，"见过的布局"不等于"训练过的布局"。** 见过的布局里有一部分落在 8、9 号桶，权重从没在上面拟合过：

| 分组 | ScaleBFM | SONIC |
|---|---|---|
| 训练过的布局（0–7 号桶） | 19 局，降幅 0.65 | 23 局，降幅 0.63 |
| 从没训练过（8、9 号桶加留出布局） | 16 局，降幅 0.65 | 18 局，降幅 0.48 |

ScaleBFM 没有"见过布局更好"的迹象；SONIC 有一档差距。

**检验局离训练样本有多近**（用最近的训练路径衡量）：
- 按参考扫过的位置：SIMPLE 见过的局 1.53 cm，开发集（训练没见过的布局）1.60 cm。检验路径并不比开发集更像训练路径。
- 按发命令时的关节：SIMPLE 见过的局 0.030，开发集 0.092（标准化后的均方根）。起始状态确实很近，这是设计如此。

**一点小偏乐观。** 早停用的就是 8 号桶，也就是停止线那份数据。好在测试集（9 号桶）的结果和它一样，影响可以忽略。

### 2.3 度量误用

**扫过区域误差只看单向、不罚滞后，但这没有让模型占便宜。** 我另算了三种（`metrics.out`）：反向（预测折线到实际扫过区域）、对称、按时刻对齐（会罚滞后）。降幅都没塌：

| 降幅（SIMPLE 全部局） | 原定义 | 反向 | 对称 | 按时刻对齐 |
|---|---|---|---|---|
| ScaleBFM | 0.64 | 0.53 | 0.62 | 0.67 |
| SONIC | 0.51 | 0.55 | 0.50 | 0.50 |

**"参考"不是最强的不用模型的对手，这是主要发现。**

参考的定义：电机参考加指令手形，骨盆在发命令时连**整个旋转**一起对齐到真实骨盆。问题在于：
- SIMPLE 发命令时真实骨盆是倾斜的：中位数 3.6°（ScaleBFM）、5.3°（SONIC），最大 7.6° / 15.9°；
- 参考骨盆本身几乎不倾斜（≤ 0.02°）；
- 训练数据里这个倾斜要小得多：中位数 1.0° / 3.0°。

把参考跟着真实骨盆的倾斜一起转，会把 SIMPLE 上参考的误差抬高。我从原始录像造了几个不用模型的对手，对着同一份实际扫过区域打分：

| 不用模型的预测（SIMPLE 全部局） | ScaleBFM 中位数 | 模型对它的降幅 [95 %] | SONIC 中位数 | 模型对它的降幅 [95 %] |
|---|---|---|---|---|
| 预注册的参考 ref | 7.90 | 0.64 [0.58, 0.83] | 14.13 | 0.51 [0.40, 0.65] |
| ref_yaw：只对齐平移和偏航 | 5.79 | 0.51 [0.46, 0.76] | 13.87 | 0.50 [0.35, 0.65] |
| ref_raw：电机参考原样，不对齐 | 5.67 | 0.50 [0.42, 0.76] | 10.80 | **0.36 [0.21, 0.57]**，模型只在 76 % 的局上更好 |
| plan_real：规划的右臂，加上发命令时的真实身体 | 6.06 | 0.53 [0.46, 0.78] | 13.43 | 0.48 [0.34, 0.66] |
| ref + 开发集上拟合的常数偏移 | 5.38 | 0.48 [0.40, 0.75] | 12.78 | 0.46 [0.35, 0.63] |
| ref_raw + 常数偏移 | 6.07 | 0.54 [0.47, 0.74] | 10.76 | **0.35 [0.19, 0.56]**，胜率 0.78 |
| 模型 | 2.82 | — | 6.96 | — |

（单位 cm。）

- SONIC 的留出布局上，对 ref_raw 的降幅是 0.31 [0.01, 0.63]，下限只比 0 高一点点。
- 开发集上各种参考差别很小。ScaleBFM 是 5.04–5.60 cm，对应降幅 0.87–0.89；SONIC 的 ref_raw 是 9.45 cm，降幅 0.60（原来 0.69）。
- 作者的"带到底"诊断也指向同一个方向：SONIC 上模型 ÷ 带到底 = 0.66，相当于降幅只有 0.34。

结论：0.64 / 0.51 有一部分是模型在纠正"参考"这个定义本身的毛病。对 ScaleBFM，换哪个对手都还在 0.48 以上，结论很稳。对 SONIC，对最强对手大约只低三分之一。

**参考的定义在训练和回放里完全相同**（同一段代码，重建结果一致）。**但起始状态的分布并不一样**，和设计里"起始状态就是 SIMPLE 里发命令时的那种"有出入：

| | 训练（开发集） | SIMPLE |
|---|---|---|
| 规划第一行离发命令时真实手臂（ScaleBFM / SONIC） | 中位数 0.000°（最大 0.47° / 4.7°） | 中位数 8.6° / 10.3°（最大 22.9° / 32.8°） |
| 起点差值，即参考手臂减规划第一行（ScaleBFM / SONIC） | 6.1° / 17.2° | 1.1° / 1.9° |

也就是说，训练里 cuRobo 总是从真实手臂规划，而 38.10.2 留下的 J 规划不是。这让 SIMPLE 检验更难，不是更容易。它也部分解释了 SIMPLE 比开发集差的那一截：模型误差 2.82 对 0.64 cm，6.96 对 3.81 cm。

**转换器：没有发现程序错误。**
- **时间对齐**：参考停止变化的时刻在 dur 之后 −0.02 到 +0.03 s；取作发命令时刻的那一帧离 t_cmd 不超过 0.02 s。
- **朝向坐标系**：用的是发命令时真实骨盆的 W。
- **坐标系和关节顺序**：实际手臂用录像 qpos 加真实手指，参考用电机参考，按 MuJoCo 顺序配 "mj"，手指用指令值，没有 NaN。
- **第一刻参考和实际差多少**（24 个球里最大的那个，取中位数）：
  - ScaleBFM：SIMPLE 3.34 cm（最大 4.16），开发集 2.64 cm，算合理；
  - SONIC：SIMPLE 6.47 cm（最大 23.3），开发集 7.97 cm。参考起点离实际手臂就很远，但这部分模型能从关节读数里知道。

### 2.4 事后选择：没有影响判定的改动

- "跑的时候改了什么"那五条都是资源或程序问题，我逐条核过。补跑没有造成重复；没有任何改动依赖结果。
- 唯一发生在看过结果之后的代码改动：`jref_data.py range` 改成可以收多份训练数据（09:23:19），在 ScaleBFM 的 dev（09:22:11）和 eval（09:22:50）之后。我换两种做法重算，中位数仍 ≤ 2，所以不影响判定：
  - 只用本任务的训练数据：中位数 0 / 0；
  - 改用 1–99 % 范围：中位数 1 / 0。
- SONIC 用完全相同的设置训练，没有调参。
- "带到底"已经标明是非预注册的诊断。
- 应该写进"意外"的几项预测落空：
  - 开发集和 SIMPLE 回放的降幅，两个控制器都高于预测区间的上限；
  - SONIC 参考的误差 14.13 cm，高于预测的 7–13 cm。

## 3. 能推翻结论的实验

**已经跑了：换更强的、不用模型的对手**（上一节的表）。

- 推翻条件：模型对最强对手的降幅 < 0.25，或 95 % 区间下限 ≤ 0。
- ScaleBFM：最小 0.48 [0.40, 0.75]，**没有推翻**。
- SONIC：最小 0.35 [0.19, 0.56]（ref_raw 加常数），**没有推翻**，但"明显小于"的幅度从 51 % 缩到约 35 %。

**设计好、没跑的（超过 15 分钟）：**
1. 在 SIMPLE 每一局发命令那一刻（和 J 同一个状态），用 `jref_plan_server` 从**真实手臂**重新规划。障碍物用真实桌子和点云，覆盖 120 个公开布局，补上两个 Bend 任务，再跑一遍 JR。
   - 推翻条件：全部局上，或只看 Bend 任务时，模型对 ref_raw 加常数的降幅 < 0.25，或区间下限 ≤ 0。
2. 站位扰动超出训练的 ±6 cm（例如 ±12 cm）再做一遍。
   - 推翻条件：模型对最强对手的降幅掉到 0.25 以下。

## 4. 按预注册判据的判定

| 判据 | ScaleBFM | SONIC |
|---|---|---|
| 停止线：开发集降幅 ≥ 0.25 | 0.88 ✓ | 0.69 ✓ |
| SIMPLE 全部局：降幅 ≥ 0.25 且区间下限 > 0 | 0.64 [0.58, 0.83] ✓ | 0.51 [0.40, 0.65] ✓ |
| 超范围关节数中位数 ≤ 2 | 0 ✓ | 0 ✓ |
| 留出布局 ≥ 0.10（只是提示项） | 0.65 ✓ | 0.51 ✓ |
| **判定** | **支持** | **支持**（幅度被参考的定义放大） |

**ScaleBFM：支持。**
- 三条判据都满足。换对手、换度量、只看从没训练过的布局，降幅都在 0.48 以上。
- 限定：证据几乎全来自 TabletopGrasp 和 XMovePick。

**SONIC：支持。**
- 判据全部满足。即使把 ref_raw 当成参考，0.36 [0.21, 0.57] 也仍然过判据。
- 但结论里应写明两点：
  - 对不对齐骨盆的电机参考，误差只低约三分之一，模型只在 76 % 的局上更好；
  - 见过布局 0.63 对从没训练过布局 0.48，有差距。
- 模型在 SIMPLE 上的 p90 仍有 19.4 cm。

**给 38.10.4 的提醒（不在判据里）。** 模型自带的误差半径是按 95 % 覆盖标定的，但在 SIMPLE 上只覆盖了 ScaleBFM 34 %、SONIC 83 % 的误差（eval.log 里的 inside 0.34 / 0.83）。拿它当避障余量之前，必须先在 SIMPLE 上重新标定。

## 附：原始输出（4090 的 `review/*.out`）

```
==== recompute.out

######## scalebfm
table rows 10004 buckets [1107, 1111, 958, 980, 975, 982, 953, 981, 970, 987] route>=0 all: True held rows in training data: 0 k counts [5014, 4990]
layouts 82 layouts in >1 bucket: 0 | layouts per bucket [9, 9, 8, 8, 8, 8, 8, 8, 8, 8]
validation all (STOP LINE): 970 paths, 8 resampling units | arm median cm ref 5.36 (p90 10.58) const 4.57 model 0.64 (p90 1.70) | drop 0.881 [0.871, 0.889] | hand drop 0.870 | model wins 1.000
validation sample k=0: 484 paths, 8 resampling units | arm median cm ref 5.54 (p90 10.14) const 4.72 model 0.65 (p90 1.65) | drop 0.882 [0.871, 0.892] | hand drop 0.874 | model wins 1.000
validation sample k=1: 486 paths, 8 resampling units | arm median cm ref 5.14 (p90 10.93) const 4.39 model 0.62 (p90 1.78) | drop 0.880 [0.868, 0.889] | hand drop 0.861 | model wins 1.000
test bucket 9: 987 paths, 8 resampling units | arm median cm ref 6.23 (p90 10.99) const 5.14 model 0.79 (p90 1.93) | drop 0.874 [0.865, 0.884] | hand drop 0.864 | model wins 0.998
-- simple_all: rows 35, held 8, layouts 35, buckets [9], route [-1], check layouts also in training data: 27
SIMPLE simple_all: 35 paths, 35 resampling units | arm median cm ref 7.90 (p90 12.19) const 5.44 model 2.82 (p90 4.02) | drop 0.643 [0.584, 0.827] | hand drop 0.560 | model wins 1.000
   out-of-range joints: median 0.0 mean 0.51 max 7 counts [28, 3, 2, 0, 1, 0, 0, 1] (from x: same=True)
-- simple_seen: rows 27, held 0, layouts 27, buckets [9], route [-1], check layouts also in training data: 27
SIMPLE simple_seen: 27 paths, 27 resampling units | arm median cm ref 7.90 (p90 11.90) const 5.44 model 2.82 (p90 3.76) | drop 0.643 [0.583, 0.841] | hand drop 0.549 | model wins 1.000
   out-of-range joints: median 0.0 mean 0.44 max 7 counts [22, 3, 1, 0, 0, 0, 0, 1] (from x: same=True)
-- simple_held: rows 8, held 8, layouts 8, buckets [9], route [-1], check layouts also in training data: 0
SIMPLE simple_held: 8 paths, 8 resampling units | arm median cm ref 7.68 (p90 12.41) const 5.50 model 2.69 (p90 4.43) | drop 0.650 [0.424, 0.865] | hand drop 0.632 | model wins 1.000
   out-of-range joints: median 0.0 mean 0.75 max 4 counts [6, 0, 1, 0, 1] (from x: same=True)

######## sonic
table rows 10009 buckets [1093, 1111, 961, 991, 965, 989, 971, 973, 980, 975] route>=0 all: True held rows in training data: 0 k counts [5013, 4996]
layouts 82 layouts in >1 bucket: 0 | layouts per bucket [9, 9, 8, 8, 8, 8, 8, 8, 8, 8]
validation all (STOP LINE): 980 paths, 8 resampling units | arm median cm ref 12.15 (p90 20.60) const 11.86 model 3.81 (p90 8.64) | drop 0.687 [0.661, 0.704] | hand drop 0.712 | model wins 0.993
validation sample k=0: 486 paths, 8 resampling units | arm median cm ref 12.95 (p90 22.09) const 12.40 model 4.73 (p90 10.19) | drop 0.635 [0.607, 0.669] | hand drop 0.665 | model wins 0.986
validation sample k=1: 494 paths, 8 resampling units | arm median cm ref 11.43 (p90 19.39) const 11.20 model 3.29 (p90 5.95) | drop 0.712 [0.686, 0.729] | hand drop 0.752 | model wins 1.000
test bucket 9: 975 paths, 8 resampling units | arm median cm ref 12.67 (p90 21.34) const 11.92 model 3.96 (p90 8.93) | drop 0.688 [0.652, 0.706] | hand drop 0.704 | model wins 0.981
-- simple_all: rows 41, held 10, layouts 41, buckets [9], route [-1], check layouts also in training data: 31
SIMPLE simple_all: 41 paths, 41 resampling units | arm median cm ref 14.13 (p90 30.04) const 12.73 model 6.96 (p90 19.43) | drop 0.508 [0.401, 0.655] | hand drop 0.612 | model wins 0.951
   out-of-range joints: median 0.0 mean 0.54 max 6 counts [30, 7, 2, 0, 0, 1, 1] (from x: same=True)
-- simple_seen: rows 31, held 0, layouts 31, buckets [9], route [-1], check layouts also in training data: 31
SIMPLE simple_seen: 31 paths, 31 resampling units | arm median cm ref 14.13 (p90 28.67) const 12.73 model 6.14 (p90 19.43) | drop 0.566 [0.295, 0.674] | hand drop 0.626 | model wins 0.935
   out-of-range joints: median 0.0 mean 0.65 max 6 counts [22, 5, 2, 0, 0, 1, 1] (from x: same=True)
-- simple_held: rows 10, held 10, layouts 10, buckets [9], route [-1], check layouts also in training data: 0
SIMPLE simple_held: 10 paths, 10 resampling units | arm median cm ref 15.28 (p90 30.60) const 13.50 model 7.56 (p90 16.93) | drop 0.505 [0.355, 0.700] | hand drop 0.597 | model wins 1.000
   out-of-range joints: median 0.0 mean 0.20 max 1 counts [8, 2] (from x: same=True)

==== jr_check.out
scalebfm: rows identical 120, differing 0; success 45 s 98, official 63; films qpos identical 119, differing 0 (max |dq| 0), missing 1; route reaches (no fall) 35; J reference speed m/s min 0.1499 median 0.1500 max 0.1502; dur s min 1.66 max 6.09; reaches at the 0.6 s floor 0
sonic: rows identical 120, differing 0; success 45 s 25, official 19; films qpos identical 119, differing 0 (max |dq| 0), missing 1; route reaches (no fall) 41; J reference speed m/s min 0.1498 median 0.1500 max 0.1503; dur s min 1.75 max 6.96; reaches at the 0.6 s floor 0

==== alt_ref_val.out

######## scalebfm
reproduced the stored check table from the raw films: max |ref diff| 5.96e-08 m, |off diff| 4.77e-09 m, |X diff| 5.96e-08 m
converter diagnostics (SIMPLE J): reference stops moving at dur + [-0.020, 0.030] s; j0 - t_cmd in [-0.020, 0.000] s; true pelvis tilt at command median 3.61 max 7.62 deg, reference pelvis tilt max 0.02 deg; plan P0 vs real arm at command max 22.852 deg; start gap median 1.09 max 4.53 deg; P0 vs real median 8.62 deg; NaN fingers in any reference: False
first moment |real - reference| per path, max over the 24 spheres: median 3.34 cm, max 4.16 cm; palm spheres median 2.19 cm
SIMPLE all: 35 paths | arm median (p90) cm: ref 7.90 (12.19), ref_yaw 5.79 (8.57), ref_raw 5.67 (7.50), ref_nogap 7.90 (12.19), plan_real 6.06 (10.49), MODEL 2.82 (4.02) | drop of the model against each: ref 0.64, ref_yaw 0.51, ref_raw 0.50, ref_nogap 0.64, plan_real 0.53 | model beats: ref 1.00, ref_yaw 1.00, ref_raw 1.00, ref_nogap 1.00, plan_real 1.00 | 95 % interval of the drop (35 units): ref [0.58, 0.83], ref_yaw [0.46, 0.76], ref_raw [0.42, 0.76], ref_nogap [0.58, 0.83], plan_real [0.46, 0.78]
SIMPLE seen: 27 paths | arm median (p90) cm: ref 7.90 (11.90), ref_yaw 5.73 (8.20), ref_raw 5.45 (6.91), ref_nogap 7.90 (11.97), plan_real 5.92 (10.68), MODEL 2.82 (3.76) | drop of the model against each: ref 0.64, ref_yaw 0.51, ref_raw 0.48, ref_nogap 0.64, plan_real 0.52 | model beats: ref 1.00, ref_yaw 1.00, ref_raw 1.00, ref_nogap 1.00, plan_real 1.00 | 95 % interval of the drop (27 units): ref [0.58, 0.84], ref_yaw [0.44, 0.78], ref_raw [0.40, 0.78], ref_nogap [0.58, 0.84], plan_real [0.46, 0.81]
SIMPLE held: 8 paths | arm median (p90) cm: ref 7.68 (12.41), ref_yaw 6.34 (9.36), ref_raw 6.42 (8.37), ref_nogap 7.68 (12.49), plan_real 6.23 (9.72), MODEL 2.69 (4.43) | drop of the model against each: ref 0.65, ref_yaw 0.58, ref_raw 0.58, ref_nogap 0.65, plan_real 0.57 | model beats: ref 1.00, ref_yaw 1.00, ref_raw 1.00, ref_nogap 1.00, plan_real 1.00 | 95 % interval of the drop (8 units): ref [0.42, 0.87], ref_yaw [0.44, 0.84], ref_raw [0.41, 0.83], ref_nogap [0.42, 0.87], plan_real [0.29, 0.85]
validation rebuilt: 970 paths (table 970), max |ref diff| 5.96e-08 m
validation (bucket 8): 970 paths | arm median (p90) cm: ref 5.36 (10.58), ref_yaw 5.04 (9.66), ref_raw 5.38 (10.02), ref_nogap 5.36 (10.68), plan_real 5.60 (10.23), MODEL 0.64 (1.70) | drop of the model against each: ref 0.88, ref_yaw 0.87, ref_raw 0.88, ref_nogap 0.88, plan_real 0.89 | model beats: ref 1.00, ref_yaw 1.00, ref_raw 1.00, ref_nogap 1.00, plan_real 1.00 | 95 % interval of the drop (8 units): ref [0.87, 0.89], ref_yaw [0.86, 0.88], ref_raw [0.86, 0.89], ref_nogap [0.87, 0.89], plan_real [0.85, 0.90]
validation first moment |real - reference| max over spheres: median 2.64 cm; true pelvis tilt median 1.01 deg; start gap median 6.08 deg; ref stops at dur + [-0.020, 0.030] s; plan P0 vs real arm at command median 0.000 max 0.465 deg; true pelvis tilt max 9.02 deg

######## sonic
reproduced the stored check table from the raw films: max |ref diff| 5.96e-08 m, |off diff| 1.75e-08 m, |X diff| 5.96e-08 m
converter diagnostics (SIMPLE J): reference stops moving at dur + [-0.020, 0.030] s; j0 - t_cmd in [-0.020, 0.020] s; true pelvis tilt at command median 5.27 max 15.94 deg, reference pelvis tilt max 0.02 deg; plan P0 vs real arm at command max 32.824 deg; start gap median 1.89 max 6.93 deg; P0 vs real median 10.33 deg; NaN fingers in any reference: False
first moment |real - reference| per path, max over the 24 spheres: median 6.47 cm, max 23.27 cm; palm spheres median 5.06 cm
SIMPLE all: 41 paths | arm median (p90) cm: ref 14.13 (30.04), ref_yaw 13.87 (26.11), ref_raw 10.80 (22.98), ref_nogap 14.13 (30.04), plan_real 13.43 (30.08), MODEL 6.96 (19.43) | drop of the model against each: ref 0.51, ref_yaw 0.50, ref_raw 0.36, ref_nogap 0.51, plan_real 0.48 | model beats: ref 0.95, ref_yaw 0.93, ref_raw 0.76, ref_nogap 0.95, plan_real 0.95 | 95 % interval of the drop (41 units): ref [0.40, 0.65], ref_yaw [0.35, 0.65], ref_raw [0.21, 0.57], ref_nogap [0.40, 0.65], plan_real [0.34, 0.66]
SIMPLE seen: 31 paths | arm median (p90) cm: ref 14.13 (28.67), ref_yaw 12.72 (26.11), ref_raw 10.80 (22.98), ref_nogap 14.13 (28.67), plan_real 13.43 (22.65), MODEL 6.14 (19.43) | drop of the model against each: ref 0.57, ref_yaw 0.52, ref_raw 0.43, ref_nogap 0.57, plan_real 0.54 | model beats: ref 0.94, ref_yaw 0.94, ref_raw 0.77, ref_nogap 0.94, plan_real 0.97 | 95 % interval of the drop (31 units): ref [0.29, 0.67], ref_yaw [0.23, 0.66], ref_raw [0.07, 0.58], ref_nogap [0.29, 0.67], plan_real [0.21, 0.68]
SIMPLE held: 10 paths | arm median (p90) cm: ref 15.28 (30.60), ref_yaw 14.21 (25.42), ref_raw 11.03 (16.44), ref_nogap 15.35 (30.60), plan_real 13.11 (30.19), MODEL 7.56 (16.93) | drop of the model against each: ref 0.51, ref_yaw 0.47, ref_raw 0.31, ref_nogap 0.51, plan_real 0.42 | model beats: ref 1.00, ref_yaw 0.90, ref_raw 0.70, ref_nogap 1.00, plan_real 0.90 | 95 % interval of the drop (10 units): ref [0.35, 0.70], ref_yaw [0.27, 0.68], ref_raw [0.01, 0.63], ref_nogap [0.34, 0.70], plan_real [0.25, 0.68]
validation rebuilt: 980 paths (table 980), max |ref diff| 5.96e-08 m
validation (bucket 8): 980 paths | arm median (p90) cm: ref 12.15 (20.60), ref_yaw 11.72 (20.49), ref_raw 9.45 (19.43), ref_nogap 12.41 (20.83), plan_real 11.75 (21.37), MODEL 3.81 (8.64) | drop of the model against each: ref 0.69, ref_yaw 0.68, ref_raw 0.60, ref_nogap 0.69, plan_real 0.68 | model beats: ref 0.99, ref_yaw 0.99, ref_raw 0.98, ref_nogap 0.99, plan_real 0.99 | 95 % interval of the drop (8 units): ref [0.66, 0.70], ref_yaw [0.65, 0.69], ref_raw [0.56, 0.62], ref_nogap [0.66, 0.71], plan_real [0.64, 0.70]
validation first moment |real - reference| max over spheres: median 7.97 cm; true pelvis tilt median 2.96 deg; start gap median 17.19 deg; ref stops at dur + [-0.020, 0.030] s; plan P0 vs real arm at command median 0.000 max 4.684 deg; true pelvis tilt max 20.70 deg

==== metrics.out（fw = 原定义，rv = 反向，sym = 对称，ta = 按时刻对齐；(b) 里的 "same layout: nan" 是因为部分见过的布局不在 0–7 号桶，分组见 seen_split.out）

######## scalebfm
(a) validation: 970 paths | fw: ref 5.36 model 0.64 drop 0.88 (wins 1.00); rv: ref 5.14 model 0.57 drop 0.89 (wins 1.00); sym: ref 5.95 model 0.67 drop 0.89 (wins 1.00); ta: ref 6.62 model 0.68 drop 0.90 (wins 1.00);
(a) SIMPLE simple_all: 35 paths | fw: ref 7.90 model 2.82 drop 0.64 (wins 1.00); rv: ref 6.39 model 3.02 drop 0.53 (wins 1.00); sym: ref 7.90 model 3.02 drop 0.62 (wins 1.00); ta: ref 9.97 model 3.34 drop 0.67 (wins 0.89);
(a) SIMPLE simple_seen: 27 paths | fw: ref 7.90 model 2.82 drop 0.64 (wins 1.00); rv: ref 6.00 model 3.02 drop 0.50 (wins 1.00); sym: ref 7.90 model 3.02 drop 0.62 (wins 1.00); ta: ref 9.70 model 3.34 drop 0.66 (wins 0.89);
(a) SIMPLE simple_held: 8 paths | fw: ref 7.68 model 2.69 drop 0.65 (wins 1.00); rv: ref 6.52 model 3.09 drop 0.53 (wins 1.00); sym: ref 7.68 model 3.14 drop 0.59 (wins 1.00); ta: ref 10.69 model 3.35 drop 0.69 (wins 0.88);
(b) nearest training path, reference key-sphere sweep mean distance (cm) / joints at command (z-scored RMS): SIMPLE seen, same layout: nan / nan; SIMPLE seen, any layout: 1.53 / 0.030; SIMPLE held, any layout: 1.94 / 0.111; validation (unseen layouts), any: 1.60 / 0.092
(c) out-of-range joints, band (0.5, 99.5): own task's training only: median 0.0 mean 0.17 max 3 | all tasks' training: median 0.0 mean 0.51
(c) out-of-range joints, band (1.0, 99.0): own task's training only: median 0.0 mean 0.51 max 7 | all tasks' training: median 1.0 mean 1.69
(d) stances per task {'BendPickMP': 1197, 'TabletopGraspMP': 1323, 'XMoveBendPickTeleop': 1386, 'XMovePickTeleop': 1260}; duplicated (task, ep, rep): 0; samples tried 10332; distinct reps 63

######## sonic
(a) validation: 980 paths | fw: ref 12.15 model 3.81 drop 0.69 (wins 0.99); rv: ref 12.24 model 3.16 drop 0.74 (wins 0.99); sym: ref 13.15 model 3.85 drop 0.71 (wins 0.99); ta: ref 15.41 model 4.42 drop 0.71 (wins 0.99);
(a) SIMPLE simple_all: 41 paths | fw: ref 14.13 model 6.96 drop 0.51 (wins 0.95); rv: ref 13.38 model 6.03 drop 0.55 (wins 0.93); sym: ref 15.13 model 7.62 drop 0.50 (wins 0.95); ta: ref 18.31 model 9.14 drop 0.50 (wins 0.95);
(a) SIMPLE simple_seen: 31 paths | fw: ref 14.13 model 6.14 drop 0.57 (wins 0.94); rv: ref 12.65 model 5.82 drop 0.54 (wins 0.90); sym: ref 14.70 model 6.96 drop 0.53 (wins 0.94); ta: ref 17.74 model 9.14 drop 0.48 (wins 0.94);
(a) SIMPLE simple_held: 10 paths | fw: ref 15.28 model 7.56 drop 0.51 (wins 1.00); rv: ref 18.09 model 7.27 drop 0.60 (wins 1.00); sym: ref 18.99 model 7.79 drop 0.59 (wins 1.00); ta: ref 21.58 model 8.91 drop 0.59 (wins 1.00);
(b) nearest training path, reference key-sphere sweep mean distance (cm) / joints at command (z-scored RMS): SIMPLE seen, same layout: nan / nan; SIMPLE seen, any layout: 3.44 / 0.299; SIMPLE held, any layout: 3.82 / 0.389; validation (unseen layouts), any: 3.56 / 0.308
(c) out-of-range joints, band (0.5, 99.5): own task's training only: median 0.0 mean 0.22 max 4 | all tasks' training: median 0.0 mean 0.54
(c) out-of-range joints, band (1.0, 99.0): own task's training only: median 0.0 mean 0.41 max 5 | all tasks' training: median 0.0 mean 0.98
(d) stances per task {'TabletopGraspMP': 1323, 'XMoveBendPickTeleop': 1386, 'BendPickMP': 1197, 'XMovePickTeleop': 1260}; duplicated (task, ep, rep): 0; samples tried 10313; distinct reps 63

==== seen_split.out

######## scalebfm
trained(0-7): 19 episodes | ref median 7.48 cm, model 2.61 cm, drop 0.65, model wins 1.00 | nearest same-layout training path (key-sphere reference sweep): 1.75 cm
bucket8: 4 episodes | ref median 6.76 cm, model 1.57 cm, drop 0.77, model wins 1.00 | nearest same-layout training path (key-sphere reference sweep): none (layout not trained on)
bucket9: 4 episodes | ref median 8.29 cm, model 3.20 cm, drop 0.61, model wins 1.00 | nearest same-layout training path (key-sphere reference sweep): none (layout not trained on)
held: 8 episodes | ref median 7.68 cm, model 2.69 cm, drop 0.65, model wins 1.00 | nearest same-layout training path (key-sphere reference sweep): none (layout not trained on)
never trained on (bucket 8/9 + held): 16 episodes, drop 0.65; trained-on layouts: 19 episodes, drop 0.65

######## sonic
trained(0-7): 23 episodes | ref median 14.13 cm, model 5.25 cm, drop 0.63, model wins 0.96 | nearest same-layout training path (key-sphere reference sweep): 3.82 cm
bucket8: 4 episodes | ref median 15.25 cm, model 5.52 cm, drop 0.64, model wins 0.75 | nearest same-layout training path (key-sphere reference sweep): none (layout not trained on)
bucket9: 4 episodes | ref median 14.04 cm, model 8.85 cm, drop 0.37, model wins 1.00 | nearest same-layout training path (key-sphere reference sweep): none (layout not trained on)
held: 10 episodes | ref median 15.28 cm, model 7.56 cm, drop 0.51, model wins 1.00 | nearest same-layout training path (key-sphere reference sweep): none (layout not trained on)
never trained on (bucket 8/9 + held): 18 episodes, drop 0.48; trained-on layouts: 23 episodes, drop 0.63

==== alt_const.out

######## scalebfm
route reaches in JR rows 35, of them in episodes that fell 0; kept by the converter 35
ref: alone 7.90 cm; + validation-fitted constant 5.38 cm (p90 9.84); MODEL 2.82 cm; drop of the model against 'ref + constant' 0.48 [0.40, 0.75], model wins 1.00
ref_yaw: alone 5.79 cm; + validation-fitted constant 5.60 cm (p90 7.95); MODEL 2.82 cm; drop of the model against 'ref_yaw + constant' 0.50 [0.43, 0.76], model wins 1.00
ref_raw: alone 5.67 cm; + validation-fitted constant 6.07 cm (p90 8.08); MODEL 2.82 cm; drop of the model against 'ref_raw + constant' 0.54 [0.47, 0.74], model wins 1.00

######## sonic
route reaches in JR rows 41, of them in episodes that fell 0; kept by the converter 41
ref: alone 14.13 cm; + validation-fitted constant 12.78 cm (p90 28.64); MODEL 6.96 cm; drop of the model against 'ref + constant' 0.46 [0.35, 0.63], model wins 0.93
ref_yaw: alone 13.87 cm; + validation-fitted constant 12.15 cm (p90 25.03); MODEL 6.96 cm; drop of the model against 'ref_yaw + constant' 0.43 [0.32, 0.63], model wins 0.93
ref_raw: alone 10.80 cm; + validation-fitted constant 10.76 cm (p90 22.33); MODEL 6.96 cm; drop of the model against 'ref_raw + constant' 0.35 [0.19, 0.56], model wins 0.78

==== last.out

######## scalebfm
XMovePickTeleop: 7 episodes, ref 4.52 cm, model 0.41 cm, drop 0.91
TabletopGraspMP: 28 episodes, ref 8.04 cm, model 3.00 cm, drop 0.63
collection: {'samples': 10332, 'planned': 10004, 'fell': 0, 'usable': 10004}
film missing: TabletopGraspMP_L1_e2_perception_direct.npz (in films_J False, in films_JR False); route reach False; stages []; fell None

######## sonic
XMovePickTeleop: 8 episodes, ref 9.53 cm, model 2.85 cm, drop 0.70
XMoveBendPickTeleop: 5 episodes, ref 10.95 cm, model 3.26 cm, drop 0.70
TabletopGraspMP: 28 episodes, ref 18.18 cm, model 9.75 cm, drop 0.46
collection: {'samples': 10313, 'planned': 10012, 'fell': 3, 'usable': 10009}
film missing: TabletopGraspMP_L1_e2_perception_direct.npz (in films_J False, in films_JR False); route reach False; stages []; fell None
```

脚本和输出都在 4090 的 `/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/jrefwm_20261124/review/`，Mac 上的副本在 `/tmp/review_jrefwm/`。
