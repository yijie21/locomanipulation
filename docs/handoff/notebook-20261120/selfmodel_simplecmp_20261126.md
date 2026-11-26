# selfmodel_simplecmp_20261126：SIMPLE 上用 SIMPLE 训练的自身模型 D 修正命令——直线三组 A / R_D / S_D，路线 J0 / J_D / J_DR

- **日期**：2026-11-26。在任何正式运行之前写好，单独提交。
- **机器 / GPU**：4090（`ssh -p 27984 root@14.103.60.188`）。仿真每个进程约 1.0 GB 显存，cuRobo 规划每个约 1.4 GB。0 号卡不碰；每张卡总量 ≤ 20 GB；我们的仿真每张卡最多 2 个；CPU 合计 ≤ 8 个进程，nice 10。开跑前看 nvidia-smi 再定用哪几张卡（见"运行计划"）。
- **代码**：分支 `sonic-selfmodel`，HEAD b0c3dd5b，加 Mac 工作区的未提交改动（包括别人的，原样保留）。
  - 新目录 `scripts/experiments/selfmodel_simplecmp_20261126/`：
    - `scmp_correct.py`（42947896）：开关本体；
    - `scmp_route.py`（7d2b2ae3）：路线组的目标和规划；
    - `scmp_run.sh`（afa6f3b5）：正式运行；
    - `scmp_eval.py`（297ff474）：登记的评估；
    - `scmp_range.py`（15098f3d）：逐下的范围检查；
    - `scmp_offline.py`（0b1c52f8）：在线、离线对照；
    - `scmp_cmp.py`（cce19545）：逐局比较两组的行；
    - `scmp_check.sh`（b6a5a43b）：核对和冒烟；
    - `scmp_copy.sh`（269044cf）：私有代码副本；
    - `scmp_freeze.sh`（0b53fabe）：冻结。
  - `simple_eval_20261112/simple_bench.py`：在别人未提交的改动之上加了默认关闭的 `--correct`。改前 adb4af64，改后 8d873cbb。
  - 正式运行前用 `scmp_freeze.sh` 把 Mac 工作区抄到 `/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/asrun/code`（下称 `$O`）。同时记下：
    - 每个文件的 md5（CODE_MD5）和冻结时刻；
    - 输入的 md5：模型 D、模板、主指标 A 和 J 的行、经过点文件、规划。

  每一步开跑前都核对 CODE_MD5。共享树一个字节都不写：场景写到 `$O/evalout/scene/`，其余条目是指向共享树的只读链接。
- **配置**：ScaleBFM，感知栏（sensor=rgbd），腿部里程计（BFM_ROOT=odo），手点是 Dex3 手掌（REACH_HAND_OFF=0.0415,0,0）。
  - 模型 D：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/train/models_D_s0/E3_ensemble_f1`（frac 1.0，seed 0）；
  - 开关：REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1，X0_REF 关。
- **关联**：
  - [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md)：模型 D。评估 3 的 0.705 cm 和直接执行的 4.01 cm，评估 4 的范围检查；
  - [selfmodel_lmlcmp3_20261126](selfmodel_lmlcmp3_20261126.md)：R / S 规则、排除和"到顶"规则；更正第 8 条，SIMPLE 里保留 S 或 E；
  - 主指标 `bench_rgbd_20261124/scalebfm`：A、J 两栏；
  - selfmodel_sfull_20261125：S 栏的经过点修正。S 栏用的模型是 jrefwm（jrefwm_20261124：在 SIMPLE 上按 J 式关节参考训练的经过点模型），不是 LML 的模型；本条目的路线组把它换成 D，其余照 S-full。

## 假设

1. **直线伸手（主假设）。** 动作和控制器都不变，只把 A 的直线命令换成在 D 上反解得到的命令（R_D）。这样手落得更准，判定集上成功更多。
   - D 在真实 SIMPLE 局上预测落点的误差，比直接执行的偏差小得多。评估 3 的任务标签上：
     | 任务 | D 的误差 | 直接执行的偏差 |
     |---|---|---|
     | XMovePick | 0.36 cm | 3–4.5 cm |
     | XMoveBendPick | 0.37 cm | 3–4.5 cm |
     | BendPick | 1.2 cm | 3–4.5 cm |
   - 所以反解应当能把手掌的偏差压到 D 误差的量级。
   - 好处应当集中在 BendPick：A 在 BendPick 上失败的局，进入段手掌偏差中位数 6.6 cm，成功的局 2.3 cm。
2. **TabletopGrasp 被桌边挡住的布局。** D 只在自由空间里训练过，看不见桌子。在 A 的 12 个失败局上：
   - D 预测"手会到"，实际差了约 27 cm；
   - 反解只把命令挪 2–8 cm，大多数仍会被桌边挡住。

   这一批回答的问题是：修正后的命令能不能越过桌边。按 F1，这部分的成功差不归功于自身模型，见"判据"。
3. **R 和 S 在 SIMPLE 上基本一样。** SIMPLE 的目标都在可达范围里，冒烟时候选命令的 IK 误差都 ≤ 0.01 cm，R 的缩回一次都没触发。所以 R_D 和 S_D 发出的命令应当大多逐位相同。
4. **路线（J）已经到顶。** J 在 TabletopGrasp 上 45 s 成功 30/30，官方时限 26/30。J_D 只能看精度变化和有没有伤害，成功率上没有检验力。

## 做法

| 组 | 预抓取 | 进入段 | 局 |
|---|---|---|---|
| A 直接执行 | 目标 | 目标 | 复用主指标的行（见"复用"） |
| AL（A 加记录） | 目标；R 的反解照算、只记录 | 同左 | 120 |
| R_D | R：在 D 上反解，候选缩回到 IK ≤ 2 cm | R | 120 |
| S_D | S：在 D 上反解，不缩回 | S | 120 |
| J 路线 | cuRobo 规划的右臂关节轨迹（主指标的经过点文件） | 目标 | 复用主指标的行 |
| JL（J 加记录） | 同 J；记录 D 对这条关节轨迹的预测 | 目标；R 照算、只记录 | 28 个有路线的局 |
| J0 | 不挪目标，在本轮重新规划一次 | 目标；只记录 | 28 |
| J_D | 目标挪 −(D 预测的手掌 − 目标)，在本轮重新规划 | 目标；只记录 | 28 |
| J_DR | 同 J_D（同一个规划文件） | R：在 D 上反解 | 28 |

**反解**：照 lmlcmp3 的 R 和 S，搬到 SIMPLE 上。
- **迭代**：从目标出发，做 6 步不动点迭代 t ← clip_t(t + 目标 − D(t))，共 7 个候选。
  - clip_t 把候选限制在数据的 WIDE 框里；
  - 损失是 |D(t) − 目标|² + 1.0 × |D 的成员离散度|²；
  - 取第一个最小值。
- **缩回（只有 R）**：第一个以外的候选，先看它的规划 IK 误差。超过 2 cm，就在目标到候选的连线上二分 6 次，退到 IK ≤ 2 cm 的最大比例 a；一个都不行时 a = 0，也就是发目标本身。下一步从缩回后的候选出发。
- **对准哪一点**：对准手掌（`--correct-point palm`，协调者 11-26 定）。
  - 理由：bench 量的是手掌；D 预测手掌的误差 0.776 cm，13 cm 点 0.821 cm（真实局，任务标签）；直接执行时手掌偏 3.19 cm，13 cm 点偏 4.24 cm，差在姿态误差乘杠杆（A 的姿态差中位数 11°，杠杆 8.85 cm）。对准 13 cm 点会在手掌上留下 1.4–2.4 cm 的姿态杠杆误差。
  - D 预测的是 13 cm 点和姿态。由预测姿态（rot6 经 Gram–Schmidt 正交化）把它换算成手掌：手掌 = pos + R(0.0415 − 0.13, 0, 0)。
  - 和手掌目标比，也就是 simple_bench 量 err_cm 的那个目标。
  - 姿态始终是目标的姿态。候选移动时，手掌和 13 cm 点一起平移。
- **候选的规划**：每个候选用 simple_bench 这一下自己的参数，走一遍 Motor.reach：进入段的速度、起点、ik_from_ref 都照原样。
  - 在 Motor._blend 里截停，电机状态一点不动，做法同 plan_invert.plan_of；
  - 仿真在思考时暂停，思考时间不计入时限（见"注意"）。
- **D 的输入**：用 d3_collect.segments 在记录器的行上建一段，命令行之后的行是占位，只影响标签。之后按 data.load 补默认值，再 derive。
  - 这和录 A 时 sd_real 的路径完全一样；
  - 冒烟证实：对 A 的命令，在线算出的 x0、u 和 simpledata 的 A 记录逐位相同。
- **决策输入**：只有关节读数（FDVEL 的速度由读数差分得到）、IMU 的重力方向和航向变化、由关节算出的手、感知给的目标，以及模型 D。
  - 仿真骨盆只作为坐标原点和高度。place.Belief.to_sim 保证目标相对身体的位置就是机器人相信的位置。
  - 仿真真值只进入带标签的记录字段（落点、标签），不进任何决策。
- **路线组（J_D）**：照 S-full（precision.py shifts2 的 SF），只把模型换成 D。
  1. 从 JL 的记录取 D 对 J 那条关节轨迹终点手掌的预测。D 没见过这种轨迹的形状：轨迹形状不是 D 的输入，只有参考终点和时长是。
  2. 命令时刻朝向系里的平移量 = −(D 预测的手掌 − 目标手掌)。
  3. prep 的 goal_wrist 加上这个平移，用 plan2.py 重新规划（IK seed 1531）。规划失败就依次退到 2/3、1/3，都失败就用 J 自己的规划。
  4. 进入段照原样发，只记录，和 S-full 一样只修预抓取。

  J0 的平移是 0，其余同 J_D。cuRobo 同一输入的重规划每次不同（冒烟时 J0 的终点关节差 4–15°），所以 J_D 的配对对象是同一轮规划出来的 J0；和 J 的比较也报。

  J_DR（协调者 11-26 加）：预抓取用 J_D 的同一个规划文件，进入段走 R_D 的反解。所以 J_DR 和 J_D 在进入段命令之前必须逐位相同（进入段命令时刻的 S0 相同），这一条进一致性检查。
- **逐下记录**：R_D、S_D、AL、JL、J0、J_D、J_DR 的每一下都记 `s["correct"]`，内容是：
  - 候选：每个的 t、D 的 13 cm 点和手掌、离散度、两种 IK 误差；
  - 缩回过程；
  - 选中的和发出的命令；
  - D 对发出命令的预测：13 cm 点和手掌，命令行朝向系和世界系各一份；
  - 在不在伸手表里；
  - 这一下结束时（simple_bench 量 err_cm 的时刻）测得的落点，以及它和预测之差；
  - 执行的规划是不是被评估的那个（plan_same）。

  每局另存 data3 格式的段（任务标签），供离线算标签处的误差和范围检查。

**复用 A、J 的行**：在当前代码上核对了逐位相同，见"登记前的核对"第 1、2 项。正式运行里，AL 的 120 局要和 A 逐局相同，JL 的 28 局要和 J 逐局相同。这两项也是完整的副作用检查：不相同就停下，不做评估。

## 局、布局和判定集

- **局**：4 个任务 × 第 0–29 局，即主指标的同一批 120 局，按局配对。ep = 10 × level + index。
- **布局**：同一个 index 的三个 level 几乎一样，所以 120 局只是 40 个独立布局。区间按布局重抽；配对检验在布局上做。
- **TabletopGrasp 的挡边布局**：index 0、1、2、4，三个 level 都算，共 12 局。
  - 在看到任何新结果之前，按 A 的行固定下来：A 在这些局全失败，预抓取时手抬不过 0.78–0.84 m，卡在桌边，右手手指碰到桌子；其中 6 局骨盆被推开 12–15 cm。
  - 这 12 局单独报。其余 18 局记作"自由"。
- **分组之前的失败**：沿用 lmlcmp3 的排除规则。三类分开计：
  - 感知失败；
  - 看完那一眼之后的规划失败（直线组：发预抓取之前 IK 失败；路线组的规划失败退回 J，不算失败）；
  - 没有那一眼。

  超过一半的任务排除。主指标 A 的 120 局全都走到了预抓取和进入段，没有一局摔倒，所以三类都是 0。
- **到顶**：A ≥ 80 % 的任务标"到顶"，排除。
  - 45 s：XMovePick 28/30、XMoveBendPick 27/30 到顶；判定集 = BendPick（10/30）+ TabletopGrasp（18/30），60 局、20 个布局。
  - 官方时限：同样两个任务到顶。BendPick 是 0/30：t_success 约 12 s，超过官方时限，各组都不可能成功。所以官方时限的判定集实际只有 TabletopGrasp。
  - 路线组：J 在 45 s 是 30/30，官方 26/30（87 %），两个时限都到顶，没有判定集，只做描述。

## 度量

- **主比较 P1**：R_D 对 A，45 s，判定集（BendPick + TabletopGrasp，20 个布局）。
  - 每个布局算三个 level 的成功数，两组不同就记 + 或 −，做精确的双侧符号检验，p < 0.05。
  - 局上的 McNemar 精确检验一起报，但局之间不独立，不作判据。
  - 成功率差附 95 % 区间：按布局重抽 4000 次，seed 0。
  - 判定集照 lmlcmp3 的规则定，不因为预计 TabletopGrasp 的 F1 不可信就事先缩小（协调者 11-26 定）。
  - 读法：TabletopGrasp 上 F1 判"不可信"时，它的成功差照报，计入 P1，但不归功于自身模型。
- **次要结果（预先写好）**：BendPick 单独的 R_D 对 A，45 s，布局符号检验（10 个布局）、McNemar 和区间。这是唯一预计 F1 可信、又有提高余地的任务。
- **按任务读**：四个任务各自报 R_D 对 A、S_D 对 A、R_D 对 S_D，两个时限都报，同样的检验和区间。TabletopGrasp 再分自由 18 局和挡边 12 局。
- **路线**：J_D 对 J0（同一轮规划，主要的配对）、J_D 对 J、J0 对 J、J_DR 对 J0、J_DR 对 J_D，都在 28 个有路线的局上，两个时限。
- **机制**：
  - M1：每一下手掌偏差的中位数，A、R_D、S_D 各一个，按任务、按预抓取和进入段；
  - M2：D 对发出命令的预测，和这一下结束时实测落点之差（手掌），AL、R_D、S_D 各一个；
  - M3：挡边 12 局里，R_D 预抓取偏差 < 5 cm 的局数（越过了桌边）。
- **F1，模型在修正后的命令上还可信吗**：对 R_D 和 S_D 实际发出的命令，按任务、在判定集上、全部合起来分别算三项：
  - (a) D 在任务标签处的误差中位数（13 cm 点，没摔倒的段），除以两个分母：
    - 登记的 SIMPLE val 1.886 cm；
    - 最可比的留出格：data3b test_iid、预抓取加进入段、任务式走法，0.993 cm。

    区间按布局重抽。
  - (b) "在范围内"的比例：sd_range 的三项检查都满足，即 39 个量、最近邻、伸手表。阈值、训练集、表都和评估 4 相同，scmp_range.py 逐下算。
  - (c) 发出命令中 IK 误差 > 2 cm 的比例。

  任何一项成立，该任务就判"模型在修正后的命令上不可信"：
  - (a) 的倍数 > 1.3；
  - (b) 的比例 < 80 %；
  - (c) 的比例比 AL（A 的命令）高 ≥ 10 个百分点。

  这样判了的任务，成功差不归功于自身模型。AL 的三项作为参照一起报。
  - 用任务标签（协调者 11-26 定），因为各组只能记任务标签：主标签要等停稳，会改变整局的节奏。
  - 两个标签差别不小：A 的命令在 TabletopGrasp 上，主标签 2.26 cm = 1.20x，任务标签 2.82 cm = 1.50x（÷1.886）。两个分母（1.886、0.993）都报。
- **命令本身**：
  - S_D 预抓取命令中 IK > 3 cm 的比例；
  - R_D 缩回触发的次数和比例；
  - R_D 和 S_D 命令逐位相同的局数；
  - 修正量；思考时间。
- **一致性**：
  - AL = A、JL = J 逐局相同。比较行和每一下的全部字段，只去掉 wall_s、开关自己加的字段，以及当前代码多记的 `bench.options.sonic_vr_frame`。
  - 两组发出的命令逐位相同时，结果必须逐位相同。
  - plan_same 全为真。
  - 在线、离线预测之差 < 1e-4 m。

## 预测

**A 的现状**（主指标，成功局数，满分 30）

| 任务 | 45 s | 官方时限 |
|---|---|---|
| XMovePick | 28 | 28 |
| XMoveBendPick | 27 | 27 |
| BendPick | 10 | 0 |
| TabletopGrasp | 18 | 18 |

J 在 TabletopGrasp 上 45 s 30、官方 26，其余任务和 A 相同。

**按任务的成功局数（满分 30）**

| 任务 | R_D 45 s | R_D 官方 | R_D − A（45 s） | 理由 |
|---|---|---|---|---|
| XMovePick | 27–30 | 27–30 | −1 到 +2 | 到顶。A 失败的布局 0，进入段偏差 3.3 cm，成功局约 2.8 cm，偏差不是明显的原因 |
| XMoveBendPick | 26–30 | 26–30 | −1 到 +3 | 到顶。A 失败的 3 局 x 向偏 4 cm 左右，D 在这些局误差 0.3 cm，可能修好 |
| BendPick | 15–26 | 0–2 | +5 到 +16 | A 失败局进入段偏 6.6 cm，成功局 2.3 cm；D 在这里误差 1.2–1.5 cm，反解后预计 1–2.8 cm。官方时限仍不可能：整局约 12 s |
| TabletopGrasp | 15–22 | 15–22 | −3 到 +4 | 挡边 12 局里越过 0–4 局（开发局冒烟 6 局挡边里越过 1 局）；自由 18 局保住 15–18 |

- **TabletopGrasp 分开**：挡边 12 局，R_D 成功 0–4；自由 18 局，15–18。
- **判定集（BendPick + TabletopGrasp，60 局）**：A 28；R_D 31–46，差 +3 到 +18 局。
  - 布局上：BendPick 有 8 个布局还有提高余地（index 0、1、3、4、6、7、8、9）；TabletopGrasp 挡边的 4 个布局只能升，自由的 6 个只能降。
  - P1 成立（符号检验 p < 0.05）的可能约 45 %。要求高，例如 6 个 + 加 0 个 − 才够，7 个 + 加 1 个 − 还不够。
- **S_D**：和 R_D 的命令逐位相同的局 ≥ 115/120，成功数和 R_D 相差 ≤ 2 局。S_D 预抓取 IK > 3 cm 的比例 0–2 %。R_D 缩回触发 ≤ 3 % 的伸手。
- **官方时限的判定集（实际是 TabletopGrasp）**：R_D − A 在 −3 到 +4。

**机制**（每一下手掌偏差的中位数，cm）

| 任务、哪一下 | A | R_D |
|---|---|---|
| XMovePick 预抓取 / 进入 | 3.0 / 2.8 | 0.4–1.2 / 0.4–1.2 |
| XMoveBendPick 预抓取 / 进入 | 3.5 / 3.5 | 0.4–1.2 / 0.4–1.2 |
| BendPick 预抓取 / 进入 | 3.0 / 6.2 | 0.8–2.0 / 1.0–2.8 |
| TabletopGrasp 自由 预抓取 / 进入 | 约 2.5 / 约 2.0 | 1.0–2.5 / 0.8–2.0 |
| TabletopGrasp 挡边 预抓取 | 约 29 | 越过的局 < 5，其余 25–36 |

- **M2**：每个任务上，D 对 R_D 发出命令的预测误差，是它对 A 命令的误差的 0.7–1.5 倍。
- **M3**：挡边 12 局里越过 0–4 局。

**F1**（R_D 发出的命令；A 的命令作参照，下表 A 一栏是评估 3 / 评估 4 在任务标签上的值，冒烟用同一数据复现过）

| 任务 | A：D 误差 / 倍数 / 在范围内 | R_D 预测：D 误差 | 倍数（÷1.886） | 在范围内 | 判定 |
|---|---|---|---|---|---|
| XMovePick | 0.356 cm / 0.19x / 100 % | 0.25–0.6 cm | 0.13–0.32x | 95–100 % | 可信 |
| XMoveBendPick | 0.368 cm / 0.20x / 100 % | 0.25–0.6 cm | 0.13–0.32x | 95–100 % | 可信 |
| BendPick | 1.225 cm / 0.65x / 88.3 % | 0.9–1.8 cm | 0.48–0.95x | 78–95 % | 可信的可能约 75 %，(b) 可能擦线 |
| TabletopGrasp | 2.823 cm / 1.50x / 60.0 % | 2–4 cm | 1.1–2.1x | 45–70 % | 不可信（约 90 %）：A 的命令本身已经过了 (a)、(b) 两条线 |
| TabletopGrasp 自由 | 2.02 cm / 1.07x / – | 1.0–2.5 cm | 0.5–1.3x | – | 只报 |
| 判定集 | – | 1.2–2.5 cm | 0.64–1.33x | 70–85 % | 拿不准 |

- 第二个分母（0.993 cm）下的倍数是上表的 1.9 倍。例如 BendPick 0.9–1.8x，TabletopGrasp 2.0–4.0x。
- BendPick 的"在范围内"主要被进入段的关节拉低：A 的进入段只有 76.7 % 在范围内，出界的是腕部关节。
- (c) 不会触发：R 按构造 ≤ 2 cm；S 预计 0–2 %。

**路线（28 局）**
- 规划：J_D 全平移规划成功 ≥ 24 局，退回 J 自己规划的 ≤ 2 局。平移量中位数 2–6 cm。
- 预抓取手掌偏差中位数：
  - J 约 3–3.6 cm；
  - J0 和 J 相差 ≤ 0.5 cm；
  - J_D 1.2–2.8 cm。
- D 在 J 的关节轨迹上的预测误差（这一下结束时，手掌）：1.0–3.0 cm。
- 成功：
  - J_D：45 s 26–28/28，官方 21–27/28；
  - J0：45 s 27–28，官方 22–27。
  - J_D − J0 在官方时限上为 −4 到 +2。S-full 在同样的局上比 J 少了 3 局（官方），所以偏向"不增反减"。
- J_DR：
  - 进入段手掌偏差中位数 0.8–2.0 cm，J_D 预计 2.5–3.5 cm（照原样发的进入段；冒烟 2.92 / 3.17）；
  - 成功：45 s 26–28/28，官方 21–27/28；J_DR − J_D 在官方时限上为 −2 到 +3。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| **BendPick**：R_D − A ≥ +5 局，且 BendPick 自己的布局符号检验 p < 0.05，且 F1 在 BendPick 上可信 | 在 SIMPLE 上，自身模型的一次性修正提高了成功率。直线伸手默认改用 R_D，下一步在 SIMPLE 上跑 SONIC |
| BendPick：R_D − A 在 +1 到 +4，或 p ≥ 0.05，但进入段偏差降到 ≤ 2.8 cm | 精度上去了，成功没跟上。查 BendPick 的失败是不是感知造成的：BendPick 上目标本身的偏差 seen_cm 约 10 cm（各局 8.7–10.3），成功局和失败局一样。不说"模型帮了成功率" |
| BendPick：R_D − A ≤ 0，且进入段偏差降了 | BendPick 的失败不是执行偏差造成的，撤回"偏差是 BendPick 失败原因"的判断 |
| 任一任务：R_D 的偏差没有降到预测区间（例如 XMovePick 预抓取 > 1.5 cm） | 反解在 SIMPLE 上没有起作用。先查 M2：D 对修正命令的预测误差有没有变大 |
| **P1 成立**（判定集，45 s） | 判定集上 R_D 赢直接执行。但只在 F1 判"可信"的任务上把差别记到自身模型名下；要写明这个赢是哪个任务带来的 |
| P1 不成立 | 照实报各任务的差和区间，不合并成"赢" |
| **TabletopGrasp 挡边**：越过 ≥ 4/12 | 修正后的命令有时能越过桌边。但 F1 预计不可信，不归功于模型。下一步查是哪个方向的平移起了作用，考虑让模型看到桌子 |
| TabletopGrasp 挡边：越过 ≤ 1/12 | 自由空间训练的模型解决不了被挡住的问题，这类布局交给路线 J |
| TabletopGrasp 自由：R_D 比 A 少 ≥ 3 局 | 在 D 超出范围的任务上（A 的命令只有 60 % 在范围内），修正会伤害。TabletopGrasp 上不用 R_D |
| **F1** 在某任务上判"不可信" | 该任务的成功差不归功于自身模型，只报 |
| R_D 和 S_D 命令不同的局 > 5 | 缩回在 SIMPLE 上也有作用（意外）。报 S 的 IK > 3 cm 比例，回头看 lmlcmp3 的结论 |
| **路线**：J_D 的预抓取偏差 ≤ 2 cm，且 J0 ≥ 3 cm | D 虽没见过关节轨迹，也能修正它的落点 |
| 路线：J_D − J0 ≥ +3 局（官方） | 修正路线对官方时限有用 |
| 路线：J_D − J0 ≤ −3 局（官方） | 和 S-full 一样，修正路线伤害官方时限。路线组不再用平移修正 |
| 路线：J_DR − J_D ≥ +3 局（官方），且 J_DR 进入段偏差 ≤ 2 cm | 路线之后的进入段也值得修正，以后路线组的进入段用 R_D |
| **一致性**不满足：AL ≠ A 或 JL ≠ J 有任何一局，J_DR 和 J_D 在进入段命令之前不同，plan_same 有假，或在线、离线差 > 1e-4 m | 停下查清，不做评估 |

## 注意

- **思考时间不计时**：仿真在反解时暂停。
  - 冒烟时，每一下的思考时间：
    | | R | S |
    |---|---|---|
    | 预抓取 | 4–6 s | 2–3.5 s |
    | 进入段 | 1–3 s | 0.5–1.7 s |
  - 真机上这段时间要算进时限。所以官方时限的结果是"不算思考时间"的结果；思考时间逐下记录，一起报。
- 修正针对的是**感知给出的目标**，不是物体真实的位置。A 的 seen_cm 中位数：XMovePick 5.7、XMoveBendPick 5.4、BendPick 9.8、TabletopGrasp 3.6 cm，比执行偏差还大，而且成功局和失败局几乎一样。所以落得准不等于抓得着，判据第二行就是为这种情况写的。

## 登记前的核对（只用开发局或只发 A / J 原命令的测试局；原始输出见 `$O/checks/`）

所有命令第一行都写了 `# notebook: skip ...`，都在 2 号卡上。

**1. 开关关闭，逐位相同**：
- 改前的 simple_bench（adb4af64）和改后的（8d873cbb）：A 选项 8/8 局逐位相同，J 选项 2/2 局逐位相同。
- 改后的和主指标的行比：只差一个当前代码多记的选项字段 `bench.options.sonic_vr_frame`，每一下的 err_cm、ik_cm 都一样。

```
### 1. switch off: simple_bench_before.py (Ab) vs simple_bench.py (Ao), A options
  BendPickMP ep 0: identical; success_45 True / True; pregrasp: err 1.857 / 1.857, ik 0.0 / 0.0 | entry: err 2.273 / 2.273, ik 0.0 / 0.0
  BendPickMP ep 5: identical; success_45 True / True; pregrasp: err 2.106 / 2.106, ik 0.01 / 0.01 | entry: err 5.675 / 5.675, ik 0.01 / 0.01
  TabletopGraspMP ep 0: identical; success_45 False / False; pregrasp: err 30.09 / 30.09, ik 0.01 / 0.01 | entry: err 33.94 / 33.94, ik 0.01 / 0.01
  TabletopGraspMP ep 3: identical; success_45 True / True; pregrasp: err 2.56 / 2.56, ik 0.01 / 0.01 | entry: err 2.304 / 2.304, ik 0.0 / 0.0
  XMoveBendPickTeleop ep 0: identical; ...   XMoveBendPickTeleop ep 6: identical; ...
  XMovePickTeleop ep 0: identical; ...       XMovePickTeleop ep 5: identical; ...
identical 8 / 8
### 1b. J options: before (Jb) vs after (Jo)
  TabletopGraspMP ep 0: identical; success_45 True / True; pregrasp: err 3.6 / 3.6, ik None / None | entry: err 3.857 / 3.857, ik 0.0 / 0.0
  TabletopGraspMP ep 3: identical; ...
identical 2 / 2
### 2. main metric A rows vs Ao: 8 episodes, each "DIFFERS in bench.options.sonic_vr_frame (only in b)", every stage's err / ik equal
### 2b. main metric J rows vs Jo: 2 episodes, the same
### 3. dev tree A rows vs Ado (dev episodes 30, 36): the same
```

**2. 只记录的反解（dryR）没有副作用**：
- R 的全部候选都规划并记录，但发出 A 的命令：8/8 局和 A 逐位相同。
- 在 J 选项上，JL 2/2 局和 J 逐位相同。

```
### 4. dryR (AL: R planned and recorded, A command sent) vs switch off (Ao)
identical 8 / 8 (compared: every row and stage key but ['correct_reset', 'film', 'wall_s'] and stages' ['correct'])
### 4b. dryR on J options (JL) vs switch off (Jo)
identical 2 / 2
```

**3. 在线 = 离线，且和 simpledata 的 A 记录逐位相同**：

```
corr_AL: 16 segments; online = offline on 16 sent stages (plan_same): max |pred diff| 4.706e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (16 reaches): max |x0 diff| 0.000e+00, |u diff| 0.000e+00, |pred diff| 5.960e-08 m
corr_JL: 4 segments; online = offline on 4 sent stages (plan_same): max |pred diff| 3.935e-06 m
```

- 用小模板（real/task 的 240 段）重建 D，得到评估 3 的 0.821 cm；换 real/main 做模板，预测的最大差为 0。
- scmp_range.py 重算出评估 4 的最近邻阈值 21.152，任务标签 "all three" 87.1 %，和评估 4 的 ended 一节相同。

**4. 冒烟：R、S，开发局 TabletopGrasp 30–39**
- 机制跑通：plan_same 20/20；在线、离线最大差 4.9e-6 m。
- R 和 S 逐局相同：缩回 0 次，候选 IK 误差都 ≤ 0.002 cm。
- 结果（开发局，不计分）：成功 5/10，开发局 A 是 4/10。
  - 第 35 局是挡边局：A 预抓取偏 26.9 cm，R 修正 1.45 cm 后偏 2.1 cm，成功；
  - 30–34 五局挡边仍然失败，D 在这些局的误差 26–39 cm；
  - 自由的 36–39 四局，预抓取偏差从 1.8–5.1 cm 降到 1.2–2.5 cm，进入段从 1.5–3.5 cm 降到 0.85–1.13 cm。

**5. 路线冒烟：开发局 30、36**
- 流程跑通：JL → 平移 → plan2 → 选取 → J_D / J0。
- JL 和开发树的 JR 行除 bench 选项字段外相同。
- D 预测 J 的轨迹终点偏 4.0 和 6.6 cm，于是平移 4.0 和 6.6 cm，全平移都规划成功。
- 预抓取偏差：J_D 1.50 和 2.79 cm，J0 3.37 和 8.54，原 J 3.38 和 8.59。
- 进入段照原样发，J_D 2.92 / 3.17，J 2.69 / 1.65。
- D 在 J 轨迹上的预测误差 1.5–2.5 cm。
- J0 的规划和 J 不同，终点关节差 4.3–14.9°，印证了 cuRobo 每次重规划结果不同。
- 显存：规划时 2 号卡最高 19.26 GB，底数 17.8 GB，所以一个规划约 1.4 GB。

## 运行计划

| 步 | 内容 | 进程分钟 |
|---|---|---|
| JL | 28 局 × 约 20 s | 约 10 |
| 路线规划 | 56 个，约 67 s 一个，一次一个 | 约 63 |
| AL | 120 局 × 19 s | 约 38 |
| R_D | 120 × 19 s | 约 38 |
| S_D | 120 × 16.5 s | 约 33 |
| J_D + J0 + J_DR | 84 × 20 s | 约 28 |

- 分两路同时跑：
  - 一路是 JL → 规划 → J_D / J0 / J_DR，约 1.7 h；
  - 另一路是 AL → R_D → S_D，每路 2 个进程，约 55 min。
- 合计约 1.7 h。CPU 同时 ≤ 5 个进程。
- 按 08:37Z（北京时间 16:37）看到的显存（1 号卡 17.0、2 号卡 17.8、3 号卡 18.0 GB），建议：
  - 1 号卡 2 个仿真（到 19.0 GB）；
  - 2 号卡 1 个仿真，或 1 个规划（到 19.2 GB）；
  - 3 号卡 1 个规划或 1 个仿真（到 19.0–19.4 GB）。

  开跑前再看 nvidia-smi 定。
- 远程任务登记到灵动岛，`~/.claude/longjobs/scmp-*.json`。

## 预测修订

## 结果（贴原始输出，不转述）

## 意外

## 结论与下一步

## 独立复核
