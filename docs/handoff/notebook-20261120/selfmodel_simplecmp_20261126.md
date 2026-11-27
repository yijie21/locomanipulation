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

- 冻结 2026-11-26T08:55:02Z（`asrun/code`，CODE_MD5 每步核对通过）。正式运行 08:55:49Z 到 09:49:56Z，在 4090 上：车道 A（AL → RD → SD）在 3 号卡，车道 B（JL → 路线规划 → JD / J0 / JDR）在 2 号卡，0 号卡没碰，每步开跑前读 nvidia-smi，等到该卡已用显存加本步所需 ≤ 19800 MiB。登记的评估 `asrun/post.sh` 在 CPU 上跑。
- 原始输出目录：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/`（行在 `rows/`，修正记录在 `corr_*/`，评估在 `eval/`）。下面逐字贴出。逐局比较的文件（`cmp_*.txt`）每局一行，这里只贴首尾两行和逐行计数，全文见 `eval/`。
- 登记前另有一次 J_DR 冒烟（开发局 30）：J_DR 的预抓取和 J_D 相同，进入段偏差 2.92 → 1.23 cm（见"登记前的核对"）。

### launch.log（去掉 room 行）和 run.log（去掉 32 行各作业的 `exit 0`）

```
08:55:49 lane A: start SLOTS=3 3 STEP=AL
08:55:49 lane B: start SLOTS=2 STEP=JL
09:02:03 lane B: end SLOTS=2 STEP=JL (exit 0)
09:02:03 lane B: start SLOTS=2 PLAN_GPU=2 PLAN_PAR=1 STEP=route
09:11:58 lane A: end SLOTS=3 3 STEP=AL (exit 0)
09:11:58 lane A: start SLOTS=3 3 STEP=RD
09:29:20 lane B: end SLOTS=2 PLAN_GPU=2 PLAN_PAR=1 STEP=route (exit 0)
09:29:20 lane B: start SLOTS=2 STEP=JD
09:29:55 lane A: end SLOTS=3 3 STEP=RD (exit 0)
09:29:55 lane A: start SLOTS=3 3 STEP=SD
09:45:33 lane A: end SLOTS=3 3 STEP=SD (exit 0)
09:49:56 lane B: end SLOTS=2 STEP=JD (exit 0)
09:49:56 both lanes done

[08:55:49] JL start: routed TabletopGrasp episodes 0,1,2,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
[09:02:03] JL: 28 rows
[09:02:03] DONE JL
[09:02:04] route: 28 routed episodes with D's shift
[09:11:58] AL: 120 rows
[09:11:58] DONE AL
[09:29:19] route plans: JD planned 21 at the full shift, 2 of 7 at 2/3, 5 of 5 at 1/3; J0 28
[09:29:20] DONE route
[09:29:55] RD: 120 rows
[09:29:55] DONE RD
[09:45:33] SD: 120 rows
[09:45:33] DONE SD
[09:49:56] JD: 28 rows
[09:49:56] J0: 28 rows
[09:49:56] JDR: 28 rows
[09:49:56] DONE JD
```

`logs/` 下 16 个日志里没有 Traceback 或 error。

### 一致性：eval/cmp_A_AL.txt、cmp_J_JL.txt、cmp_SD_RD.txt、jdr_vs_jd.txt

```
$ head -1 cmp_A_AL.txt
a /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm/rows (120 rows), b /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/rows (120 rows): 120 episodes in both
$ tail -1 cmp_A_AL.txt
identical 0 / 120 (compared: every row and stage key but ['correct_reset', 'film', 'wall_s'] and stages' ['correct'])
# grep counts: episode lines 120; 'DIFFERS in bench.options.sonic_vr_frame (only in b); success_45' 120; ': identical;' 0; other 0

$ head -1 cmp_J_JL.txt
a /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm/rows (120 rows), b /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/rows (28 rows): 28 episodes in both
$ tail -1 cmp_J_JL.txt
identical 0 / 28 (compared: every row and stage key but ['correct_reset', 'film', 'wall_s'] and stages' ['correct'])
# grep counts: episode lines 28; 'DIFFERS in bench.options.sonic_vr_frame (only in b); success_45' 28; ': identical;' 0; other 0

$ head -1 cmp_SD_RD.txt
a /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/rows (120 rows), b /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/rows (120 rows): 120 episodes in both
$ tail -1 cmp_SD_RD.txt
identical 120 / 120 (compared: every row and stage key but ['correct_reset', 'film', 'wall_s'] and stages' ['correct'])
# grep counts: episode lines 120; 'DIFFERS in bench.options.sonic_vr_frame (only in b); success_45' 0; ': identical;' 120; other 0

$ cat jdr_vs_jd.txt
J_DR = J_D up to the entry's command: 28 / 28
```

### 在线、离线对照：eval/offline.txt

```
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_AL: 240 segments; online = offline on 240 sent stages (plan_same): max |pred diff| 5.049e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (240 reaches): max |x0 diff| 0.000e+00, |u diff| 0.000e+00, |pred diff| 1.192e-07 m
  XMovePickTeleop      n  60: D error at the task label median 0.356 cm = 0.189x SIMPLE val, 0.358x test_iid task-walk
  XMoveBendPickTeleop  n  60: D error at the task label median 0.368 cm = 0.195x SIMPLE val, 0.370x test_iid task-walk
  BendPickMP           n  60: D error at the task label median 1.225 cm = 0.649x SIMPLE val, 1.233x test_iid task-walk
  TabletopGraspMP      n  60: D error at the task label median 2.823 cm = 1.497x SIMPLE val, 2.843x test_iid task-walk
  all                  n 240: 0.821 cm = 0.435x / 0.827x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_RD: 240 segments; online = offline on 240 sent stages (plan_same): max |pred diff| 4.995e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (240 reaches): max |x0 diff| 2.839e+00, |u diff| 1.655e-01, |pred diff| 1.420e-01 m
  XMovePickTeleop      n  60: D error at the task label median 0.432 cm = 0.229x SIMPLE val, 0.435x test_iid task-walk
  XMoveBendPickTeleop  n  60: D error at the task label median 0.540 cm = 0.286x SIMPLE val, 0.544x test_iid task-walk
  BendPickMP           n  60: D error at the task label median 1.135 cm = 0.602x SIMPLE val, 1.143x test_iid task-walk
  TabletopGraspMP      n  60: D error at the task label median 2.106 cm = 1.117x SIMPLE val, 2.121x test_iid task-walk
  all                  n 240: 0.955 cm = 0.506x / 0.961x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_SD: 240 segments; online = offline on 240 sent stages (plan_same): max |pred diff| 4.995e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (240 reaches): max |x0 diff| 2.839e+00, |u diff| 1.655e-01, |pred diff| 1.420e-01 m
  XMovePickTeleop      n  60: D error at the task label median 0.432 cm = 0.229x SIMPLE val, 0.435x test_iid task-walk
  XMoveBendPickTeleop  n  60: D error at the task label median 0.540 cm = 0.286x SIMPLE val, 0.544x test_iid task-walk
  BendPickMP           n  60: D error at the task label median 1.135 cm = 0.602x SIMPLE val, 1.143x test_iid task-walk
  TabletopGraspMP      n  60: D error at the task label median 2.106 cm = 1.117x SIMPLE val, 2.121x test_iid task-walk
  all                  n 240: 0.955 cm = 0.506x / 0.961x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_JL: 56 segments; online = offline on 56 sent stages (plan_same): max |pred diff| 4.953e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (56 reaches): max |x0 diff| 2.969e+00, |u diff| 1.561e-01, |pred diff| 1.040e-01 m
  TabletopGraspMP      n  56: D error at the task label median 1.945 cm = 1.031x SIMPLE val, 1.958x test_iid task-walk
  all                  n  56: 1.945 cm = 1.031x / 1.958x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_JD: 56 segments; online = offline on 56 sent stages (plan_same): max |pred diff| 5.135e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (56 reaches): max |x0 diff| 3.220e+00, |u diff| 1.561e-01, |pred diff| 1.039e-01 m
  TabletopGraspMP      n  56: D error at the task label median 1.660 cm = 0.880x SIMPLE val, 1.672x test_iid task-walk
  all                  n  56: 1.660 cm = 0.880x / 1.672x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_J0: 56 segments; online = offline on 56 sent stages (plan_same): max |pred diff| 4.964e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (56 reaches): max |x0 diff| 2.973e+00, |u diff| 1.561e-01, |pred diff| 1.044e-01 m
  TabletopGraspMP      n  56: D error at the task label median 1.986 cm = 1.053x SIMPLE val, 2.000x test_iid task-walk
  all                  n  56: 1.986 cm = 1.053x / 2.000x
/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_JDR: 56 segments; online = offline on 56 sent stages (plan_same): max |pred diff| 5.135e-06 m (rounded to 1e-5 in the log); 0 stages without a pair
  against simpledata's A record (56 reaches): max |x0 diff| 3.220e+00, |u diff| 1.561e-01, |pred diff| 9.699e-02 m
  TabletopGraspMP      n  56: D error at the task label median 1.744 cm = 0.925x SIMPLE val, 1.756x test_iid task-walk
  all                  n  56: 1.744 cm = 0.925x / 1.756x
```

### 逐下的范围检查：eval/range.txt

```
training set 809804 segments; nearest-neighbour p99 21.152 (sd_range's eval4: 21.152)
AL: 240 reaches, in range 87.1 % (kept 240), D median 0.821 cm
RD: 240 reaches, in range 87.9 % (kept 240), D median 0.955 cm
SD: 240 reaches, in range 87.9 % (kept 240), D median 0.955 cm
JL: 56 reaches, in range 83.9 % (kept 56), D median 1.945 cm
JD: 56 reaches, in range 92.9 % (kept 56), D median 1.660 cm
J0: 56 reaches, in range 83.9 % (kept 56), D median 1.986 cm
JDR: 56 reaches, in range 92.9 % (kept 56), D median 1.744 cm
```

### 登记的评估：eval/eval.txt

```
== rows: A 120, J 120, AL 120, RD 120, SD 120, JL 28, JD 28, J0 28, JDR 28

== 1. consistency (every row and stage key but wall_s, --correct's fields and the option sonic_vr_frame)
  A = AL: 120 / 120 episodes identical
  J = JL: 28 / 28 episodes identical
  RD = SD: 120 / 120 episodes identical
  R_D episodes without a shrunk candidate (S_D's commands are then R_D's): 120 / 120

== 2. failures before the arms split (A's rows), per task and class
  XMovePick     : 0 (none)
  XMoveBendPick : 0 (none)
  BendPick      : 0 (none)
  TabletopGrasp : 0 (none)

== 3. success per task (TabletopGrasp also without / within the blocked layouts)
  -- 45 s
  XMovePick     : A 28, AL 28, RD 30, SD 30 (of 30)
  XMoveBendPick : A 27, AL 27, RD 30, SD 30 (of 30)
  BendPick      : A 10, AL 10, RD 30, SD 30 (of 30)
  TabletopGrasp : A 18, AL 18, RD 18, SD 18 (of 30); route J 30, JL 28, JD 27, J0 28, JDR 28
                  free: A 18, RD 18, SD 18, J 18, JD 16, J0 16 (of 18)
                  blocked: A 0, RD 0, SD 0, J 12, JD 11, J0 12 (of 12)
  -- official
  XMovePick     : A 28, AL 28, RD 30, SD 30 (of 30)
  XMoveBendPick : A 27, AL 27, RD 30, SD 30 (of 30)
  BendPick      : A 0, AL 0, RD 0, SD 0 (of 30)
  TabletopGrasp : A 18, AL 18, RD 18, SD 18 (of 30); route J 26, JL 24, JD 16, J0 23, JDR 16
                  free: A 18, RD 18, SD 18, J 17, JD 12, J0 15 (of 18)
                  blocked: A 0, RD 0, SD 0, J 9, JD 4, J0 8 (of 12)

== 4. decision set (lmlcmp3's rule: pre-split failures over half out; A >= 80 % 'ceiling' out)
  45 s     XMovePick     : A 28 / 30 -> out (ceiling)
  45 s     XMoveBendPick : A 27 / 30 -> out (ceiling)
  45 s     BendPick      : A 10 / 30 -> in
  45 s     TabletopGrasp : A 18 / 30 -> in
  official XMovePick     : A 28 / 30 -> out (ceiling)
  official XMoveBendPick : A 27 / 30 -> out (ceiling)
  official BendPick      : A 0 / 30 -> in
  official TabletopGrasp : A 18 / 30 -> in

== 5. paired tests
  P1 R_D vs A  45 s     decision set ['BendPick', 'TabletopGrasp']: 48 / 60 vs 28 / 60 (+33.3 pp [+15.0, +53.3] by layout); layouts + 8 / - 0 of 20, sign test p 0.007812; episodes + 20 / - 0, McNemar p 1.907e-06
  S_D vs A     45 s     decision set ['BendPick', 'TabletopGrasp']: 48 / 60 vs 28 / 60 (+33.3 pp [+15.0, +53.3] by layout); layouts + 8 / - 0 of 20, sign test p 0.007812; episodes + 20 / - 0, McNemar p 1.907e-06
  R_D vs S_D   45 s     decision set ['BendPick', 'TabletopGrasp']: 48 / 60 vs 48 / 60 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 20, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs A     45 s     XMovePick     : 30 / 30 vs 28 / 30 (+6.7 pp [+0.0, +20.0] by layout); layouts + 1 / - 0 of 10, sign test p 1; episodes + 2 / - 0, McNemar p 0.5
  R_D vs A     45 s     XMoveBendPick : 30 / 30 vs 27 / 30 (+10.0 pp [+0.0, +26.7] by layout); layouts + 2 / - 0 of 10, sign test p 0.5; episodes + 3 / - 0, McNemar p 0.25
  R_D vs A     45 s     BendPick      : 30 / 30 vs 10 / 30 (+66.7 pp [+40.0, +90.0] by layout); layouts + 8 / - 0 of 10, sign test p 0.007812; episodes + 20 / - 0, McNemar p 1.907e-06
  R_D vs A     45 s     TabletopGrasp : 18 / 30 vs 18 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs A     45 s     all four      : 108 / 120 vs 83 / 120 (+20.8 pp [+10.8, +32.5] by layout); layouts + 11 / - 0 of 40, sign test p 0.0009766; episodes + 25 / - 0, McNemar p 5.96e-08
  R_D vs A     45 s     TabletopGrasp free: 18 / 18 vs 18 / 18 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 6, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs A     45 s     TabletopGrasp blocked: 0 / 12 vs 0 / 12 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 4, sign test p 1; episodes + 0 / - 0, McNemar p 1
  route J_D vs J0 45 s     routed TabletopGrasp: 27 / 28 vs 28 / 28 (-3.6 pp [-10.7, +0.0] by layout); layouts + 0 / - 1 of 10, sign test p 1; episodes + 0 / - 1, McNemar p 1
  J_D vs J        45 s     routed TabletopGrasp: 27 / 28 vs 28 / 28 (-3.6 pp [-10.7, +0.0] by layout); layouts + 0 / - 1 of 10, sign test p 1; episodes + 0 / - 1, McNemar p 1
  J0 vs J         45 s     routed TabletopGrasp: 28 / 28 vs 28 / 28 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  J_DR vs J0      45 s     routed TabletopGrasp: 28 / 28 vs 28 / 28 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  J_DR vs J_D     45 s     routed TabletopGrasp: 28 / 28 vs 27 / 28 (+3.6 pp [+0.0, +10.7] by layout); layouts + 1 / - 0 of 10, sign test p 1; episodes + 1 / - 0, McNemar p 1
  P1 R_D vs A  official decision set ['BendPick', 'TabletopGrasp']: 18 / 60 vs 18 / 60 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 20, sign test p 1; episodes + 0 / - 0, McNemar p 1
  S_D vs A     official decision set ['BendPick', 'TabletopGrasp']: 18 / 60 vs 18 / 60 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 20, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs S_D   official decision set ['BendPick', 'TabletopGrasp']: 18 / 60 vs 18 / 60 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 20, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs A     official XMovePick     : 30 / 30 vs 28 / 30 (+6.7 pp [+0.0, +20.0] by layout); layouts + 1 / - 0 of 10, sign test p 1; episodes + 2 / - 0, McNemar p 0.5
  R_D vs A     official XMoveBendPick : 30 / 30 vs 27 / 30 (+10.0 pp [+0.0, +26.7] by layout); layouts + 2 / - 0 of 10, sign test p 0.5; episodes + 3 / - 0, McNemar p 0.25
  R_D vs A     official BendPick      : 0 / 30 vs 0 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs A     official TabletopGrasp : 18 / 30 vs 18 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs A     official all four      : 78 / 120 vs 73 / 120 (+4.2 pp [+0.0, +9.2] by layout); layouts + 3 / - 0 of 40, sign test p 0.25; episodes + 5 / - 0, McNemar p 0.0625
  R_D vs A     official TabletopGrasp free: 18 / 18 vs 18 / 18 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 6, sign test p 1; episodes + 0 / - 0, McNemar p 1
  R_D vs A     official TabletopGrasp blocked: 0 / 12 vs 0 / 12 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 4, sign test p 1; episodes + 0 / - 0, McNemar p 1
  route J_D vs J0 official routed TabletopGrasp: 16 / 28 vs 23 / 28 (-25.0 pp [-43.3, -7.1] by layout); layouts + 0 / - 4 of 10, sign test p 0.125; episodes + 0 / - 7, McNemar p 0.01562
  J_D vs J        official routed TabletopGrasp: 16 / 28 vs 24 / 28 (-28.6 pp [-50.0, -6.7] by layout); layouts + 0 / - 4 of 10, sign test p 0.125; episodes + 0 / - 8, McNemar p 0.007812
  J0 vs J         official routed TabletopGrasp: 23 / 28 vs 24 / 28 (-3.6 pp [-11.5, +0.0] by layout); layouts + 0 / - 1 of 10, sign test p 1; episodes + 0 / - 1, McNemar p 1
  J_DR vs J0      official routed TabletopGrasp: 16 / 28 vs 23 / 28 (-25.0 pp [-43.3, -7.1] by layout); layouts + 0 / - 4 of 10, sign test p 0.125; episodes + 0 / - 7, McNemar p 0.01562
  J_DR vs J_D     official routed TabletopGrasp: 16 / 28 vs 16 / 28 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1

== 6. mechanism: each stage's palm miss (cm, median; the episodes all three arms ran), D's prediction error of the sent command at the stage's end (palm, cm, median)
  XMovePick             pregrasp (n 30): miss A 3.02, RD 0.66, SD 0.66 | D's error AL 0.60, RD 0.65, SD 0.65
  XMovePick             entry    (n 30): miss A 2.79, RD 0.48, SD 0.48 | D's error AL 0.37, RD 0.48, SD 0.48
  XMoveBendPick         pregrasp (n 30): miss A 3.53, RD 0.47, SD 0.47 | D's error AL 0.39, RD 0.47, SD 0.47
  XMoveBendPick         entry    (n 30): miss A 3.49, RD 0.70, SD 0.70 | D's error AL 0.43, RD 0.68, SD 0.68
  BendPick              pregrasp (n 30): miss A 3.01, RD 0.96, SD 0.96 | D's error AL 0.99, RD 0.97, SD 0.97
  BendPick              entry    (n 30): miss A 6.19, RD 1.25, SD 1.25 | D's error AL 1.46, RD 1.25, SD 1.25
  TabletopGrasp         pregrasp (n 30): miss A 5.18, RD 2.49, SD 2.49 | D's error AL 2.76, RD 2.50, SD 2.50
  TabletopGrasp         entry    (n 30): miss A 4.76, RD 1.65, SD 1.65 | D's error AL 2.54, RD 1.66, SD 1.66
  TabletopGrasp free    pregrasp (n 18): miss A 2.54, RD 1.84, SD 1.84 | D's error AL 1.81, RD 1.85, SD 1.85
  TabletopGrasp free    entry    (n 18): miss A 2.03, RD 1.45, SD 1.45 | D's error AL 1.75, RD 1.45, SD 1.45
  TabletopGrasp blocked pregrasp (n 12): miss A 29.27, RD 28.81, SD 28.81 | D's error AL 27.22, RD 28.81, SD 28.81
  TabletopGrasp blocked entry    (n 12): miss A 32.18, RD 31.88, SD 31.88 | D's error AL 27.08, RD 31.87, SD 31.87
  TabletopGrasp blocked: R_D pregrasps within 5 cm (cleared the edge) 0 / 12 []; A 0 / 12

== 7. F1: is D still trustworthy on the commands R_D / S_D sent (AL: A's commands, the reference)
  AL  XMovePick         : n  60, D median 0.356 cm [0.295, 0.377] = 0.19x SIMPLE val [0.16, 0.20], 0.36x test_iid task-walk; in range 100.0 %; IK > 2 cm 0.0 %
  AL  XMoveBendPick     : n  60, D median 0.368 cm [0.306, 0.434] = 0.19x SIMPLE val [0.16, 0.23], 0.37x test_iid task-walk; in range 100.0 %; IK > 2 cm 0.0 %
  AL  BendPick          : n  60, D median 1.225 cm [1.143, 1.266] = 0.65x SIMPLE val [0.61, 0.67], 1.23x test_iid task-walk; in range 88.3 %; IK > 2 cm 0.0 %
  AL  TabletopGrasp     : n  60, D median 2.823 cm [2.023, 31.681] = 1.50x SIMPLE val [1.07, 16.80], 2.84x test_iid task-walk; in range 60.0 %; IK > 2 cm 0.0 %
  AL  TabletopGrasp free: n  36, D median 2.024 cm [1.773, 2.760] = 1.07x SIMPLE val [0.94, 1.46], 2.04x test_iid task-walk; in range 63.9 %; IK > 2 cm 0.0 %
  AL  decision set 45 s : n 120, D median 1.592 cm [1.258, 2.162] = 0.84x SIMPLE val [0.67, 1.15], 1.60x test_iid task-walk; in range 74.2 %; IK > 2 cm 0.0 %
  AL  all               : n 240, D median 0.821 cm [0.415, 1.219] = 0.44x SIMPLE val [0.22, 0.65], 0.83x test_iid task-walk; in range 87.1 %; IK > 2 cm 0.0 %
  RD  XMovePick         : n  60, D median 0.432 cm [0.416, 0.452] = 0.23x SIMPLE val [0.22, 0.24], 0.44x test_iid task-walk; in range 100.0 %; IK > 2 cm 0.0 %
  RD  XMoveBendPick     : n  60, D median 0.540 cm [0.359, 0.696] = 0.29x SIMPLE val [0.19, 0.37], 0.54x test_iid task-walk; in range 100.0 %; IK > 2 cm 0.0 %
  RD  BendPick          : n  60, D median 1.135 cm [1.060, 1.260] = 0.60x SIMPLE val [0.56, 0.67], 1.14x test_iid task-walk; in range 88.3 %; IK > 2 cm 0.0 %
  RD  TabletopGrasp     : n  60, D median 2.106 cm [1.827, 35.034] = 1.12x SIMPLE val [0.97, 18.58], 2.12x test_iid task-walk; in range 63.3 %; IK > 2 cm 0.0 %
  RD  TabletopGrasp free: n  36, D median 1.827 cm [1.549, 1.942] = 0.97x SIMPLE val [0.82, 1.03], 1.84x test_iid task-walk; in range 69.4 %; IK > 2 cm 0.0 %
  RD  decision set 45 s : n 120, D median 1.548 cm [1.219, 1.942] = 0.82x SIMPLE val [0.65, 1.03], 1.56x test_iid task-walk; in range 75.8 %; IK > 2 cm 0.0 %
  RD  all               : n 240, D median 0.955 cm [0.641, 1.126] = 0.51x SIMPLE val [0.34, 0.60], 0.96x test_iid task-walk; in range 87.9 %; IK > 2 cm 0.0 %
  SD  XMovePick         : n  60, D median 0.432 cm [0.417, 0.456] = 0.23x SIMPLE val [0.22, 0.24], 0.44x test_iid task-walk; in range 100.0 %; IK > 2 cm 0.0 %
  SD  XMoveBendPick     : n  60, D median 0.540 cm [0.355, 0.715] = 0.29x SIMPLE val [0.19, 0.38], 0.54x test_iid task-walk; in range 100.0 %; IK > 2 cm 0.0 %
  SD  BendPick          : n  60, D median 1.135 cm [1.060, 1.260] = 0.60x SIMPLE val [0.56, 0.67], 1.14x test_iid task-walk; in range 88.3 %; IK > 2 cm 0.0 %
  SD  TabletopGrasp     : n  60, D median 2.106 cm [1.816, 34.848] = 1.12x SIMPLE val [0.96, 18.48], 2.12x test_iid task-walk; in range 63.3 %; IK > 2 cm 0.0 %
  SD  TabletopGrasp free: n  36, D median 1.827 cm [1.551, 1.937] = 0.97x SIMPLE val [0.82, 1.03], 1.84x test_iid task-walk; in range 69.4 %; IK > 2 cm 0.0 %
  SD  decision set 45 s : n 120, D median 1.548 cm [1.219, 1.919] = 0.82x SIMPLE val [0.65, 1.02], 1.56x test_iid task-walk; in range 75.8 %; IK > 2 cm 0.0 %
  SD  all               : n 240, D median 0.955 cm [0.641, 1.135] = 0.51x SIMPLE val [0.34, 0.60], 0.96x test_iid task-walk; in range 87.9 %; IK > 2 cm 0.0 %
  JL  TabletopGrasp     : n  56, D median 1.945 cm [1.711, 2.052] = 1.03x SIMPLE val [0.91, 1.09], 1.96x test_iid task-walk; in range 83.9 %; IK > 2 cm 0.0 %
  JL  TabletopGrasp free: n  32, D median 1.949 cm [1.615, 2.120] = 1.03x SIMPLE val [0.86, 1.12], 1.96x test_iid task-walk; in range 87.5 %; IK > 2 cm 0.0 %
  JL  decision set 45 s : n  56, D median 1.945 cm [1.703, 2.052] = 1.03x SIMPLE val [0.90, 1.09], 1.96x test_iid task-walk; in range 83.9 %; IK > 2 cm 0.0 %
  JL  all               : n  56, D median 1.945 cm [1.703, 2.052] = 1.03x SIMPLE val [0.90, 1.09], 1.96x test_iid task-walk; in range 83.9 %; IK > 2 cm 0.0 %
  JD  TabletopGrasp     : n  56, D median 1.660 cm [1.398, 1.978] = 0.88x SIMPLE val [0.74, 1.05], 1.67x test_iid task-walk; in range 92.9 %; IK > 2 cm 0.0 %
  JD  TabletopGrasp free: n  32, D median 1.498 cm [1.326, 1.952] = 0.79x SIMPLE val [0.70, 1.03], 1.51x test_iid task-walk; in range 96.9 %; IK > 2 cm 0.0 %
  JD  decision set 45 s : n  56, D median 1.660 cm [1.393, 1.978] = 0.88x SIMPLE val [0.74, 1.05], 1.67x test_iid task-walk; in range 92.9 %; IK > 2 cm 0.0 %
  JD  all               : n  56, D median 1.660 cm [1.399, 1.972] = 0.88x SIMPLE val [0.74, 1.05], 1.67x test_iid task-walk; in range 92.9 %; IK > 2 cm 0.0 %
  JDR TabletopGrasp     : n  56, D median 1.744 cm [1.335, 1.932] = 0.92x SIMPLE val [0.71, 1.02], 1.76x test_iid task-walk; in range 92.9 %; IK > 2 cm 0.0 %
  JDR TabletopGrasp free: n  32, D median 1.492 cm [1.118, 1.850] = 0.79x SIMPLE val [0.59, 0.98], 1.50x test_iid task-walk; in range 96.9 %; IK > 2 cm 0.0 %
  JDR decision set 45 s : n  56, D median 1.744 cm [1.326, 1.921] = 0.92x SIMPLE val [0.70, 1.02], 1.76x test_iid task-walk; in range 92.9 %; IK > 2 cm 0.0 %
  JDR all               : n  56, D median 1.744 cm [1.326, 1.932] = 0.92x SIMPLE val [0.70, 1.02], 1.76x test_iid task-walk; in range 92.9 %; IK > 2 cm 0.0 %
  F1 RD XMovePick         : trustworthy
  F1 RD XMoveBendPick     : trustworthy
  F1 RD BendPick          : trustworthy
  F1 RD TabletopGrasp     : NOT trustworthy (in range < 80 %)
  F1 RD decision set 45 s : NOT trustworthy (in range < 80 %)
  F1 RD all               : trustworthy
  F1 SD XMovePick         : trustworthy
  F1 SD XMoveBendPick     : trustworthy
  F1 SD BendPick          : trustworthy
  F1 SD TabletopGrasp     : NOT trustworthy (in range < 80 %)
  F1 SD decision set 45 s : NOT trustworthy (in range < 80 %)
  F1 SD all               : trustworthy

== 8. the commands
  RD pregrasp: n 120, sent IK > 3 cm 0 (0.0 %), > 2 cm 0; correction median 3.07 cm, max 4.09; shrunk candidates 0 of 0; fallbacks 0; plan_same 120; think median 4.5 s, max 8.7
  RD entry   : n 120, sent IK > 3 cm 0 (0.0 %), > 2 cm 0; correction median 3.13 cm, max 7.32; shrunk candidates 0 of 0; fallbacks 0; plan_same 120; think median 1.3 s, max 3.9
  SD pregrasp: n 120, sent IK > 3 cm 0 (0.0 %), > 2 cm 0; correction median 3.07 cm, max 4.09; shrunk candidates 0 of 0; fallbacks 0; plan_same 120; think median 2.4 s, max 4.5
  SD entry   : n 120, sent IK > 3 cm 0 (0.0 %), > 2 cm 0; correction median 3.13 cm, max 7.32; shrunk candidates 0 of 0; fallbacks 0; plan_same 120; think median 0.8 s, max 2.0
  route JD: 28 episodes, plans by scale {'1': 21, '1/3': 5, '2/3': 2}, same as J's plan 0
  route J0: 28 episodes, plans by scale {'1': 28}, same as J's plan 0
  route shifts (D's -(predicted palm - goal) of J's path): median 3.98 cm, max 6.72
  J   pregrasp / entry miss median 3.81 / 2.33 cm (routed)
  JL  pregrasp / entry miss median 3.81 / 2.33 cm (routed); D's error at the pregrasp's end 2.18
  J0  pregrasp / entry miss median 3.86 / 2.34 cm (routed); D's error at the pregrasp's end 2.19
  JD  pregrasp / entry miss median 2.09 / 2.94 cm (routed); D's error at the pregrasp's end 1.90
  JDR pregrasp / entry miss median 2.09 / 1.41 cm (routed); D's error at the pregrasp's end 1.90
```


## 意外

以下逐条对照"预测"，只列落在预测区间外的，数字都来自上面的原始输出。

1. **BendPick 超出预测上沿**。45 s 时 R_D 30/30，预测 15–26（R_D − A 为 +20 局，预测 +5 到 +16）；A 失败的 20 局全部成功，没有一局反过来（布局 + 8 / − 0）。S_D 逐局相同。所以判定集 R_D 48/60，也高于预测的 31–46。"注意"里估计 BendPick 的感知偏差（seen_cm 约 10 cm）会限制成功，45 s 时没有出现这种限制。官方时限（BendPick 8.0 s）下各组仍都是 0/30，和预测（0–2）相符。
2. **路线组在官方时限下掉得比预测多**。
   - J_D 官方 16/28，预测 21–27；J_D − J0 为 −7 局，预测 −4 到 +2；J_DR 官方 16/28，预测 21–27。45 s 时 J_D 27、J0 28、J_DR 28，在预测内。
   - 事后诊断（**未登记**，只读 `rows/`）：TabletopGrasp 的官方时限是 16.0 s（`simple_bench.py` 的 `LIMIT_S`）。J0 的成功时刻中位数 15.77 s，J_D 15.94 s；J_D 预抓取的关节路径中位数 83.7 cm，J0 80.8 cm，时长 5.58 s 对 5.39 s。45 s 内成功、但成功时刻晚于 16.0 s 的局：J0 5 局，J_D 11 局，J_DR 12 局。J_D 比 J0 多掉的 7 局（0、7、10、12、16、17、22）成功时刻都在 16.06–16.22 s，即时限后 0.3 s 以内；这 7 局 J_D 的预抓取关节路径都比 J0 长 3.5–5.8 cm。诊断脚本是临时的，原样输出如下：

     ```
     ep  | J0 t_succ off/45 path dur | JD t_succ off/45 path dur | JDR t_succ off/45 path dur
       0 |   16.0 11  80.2 5.35 |  16.16 01  83.7 5.58 |   16.2 01  83.7 5.58
       1 |  16.14 01  86.9 5.79 |  16.16 01  87.7 5.84 |  16.16 01  87.7 5.84
       2 |  16.02 01  82.0 5.47 |  16.22 01  87.0 5.80 |  16.06 01  87.0 5.80
       4 | 15.760000000000002 11  79.0 5.27 | 15.940000000000001 11  82.5 5.50 |  15.82 11  82.5 5.50
       5 | 15.240000000000002 11  75.8 5.05 | 15.899999999999999 11  86.8 5.79 | 15.899999999999999 11  86.8 5.79
       6 | 16.080000000000002 01  86.6 5.77 |  16.26 01  91.3 6.09 |  16.22 01  91.3 6.09
       7 | 15.920000000000002 11  84.6 5.64 |  16.06 01  89.3 5.95 |  16.04 01  89.3 5.95
       8 | 15.620000000000001 11  78.7 5.25 | 15.760000000000002 11  81.7 5.44 |   15.8 11  81.7 5.44
       9 |  15.66 11  82.5 5.50 |  15.68 11  83.3 5.55 |  15.68 11  83.3 5.55
      10 |   16.0 11  80.0 5.33 |  16.18 01  83.5 5.57 |  16.16 01  83.5 5.57
      11 |  16.18 01  87.0 5.80 |  16.16 01  87.9 5.86 |  16.14 01  87.9 5.86
      12 |   16.0 11  81.8 5.45 |  16.22 01  87.6 5.84 |  16.12 01  87.6 5.84
      14 | 15.780000000000001 11  79.3 5.29 | 15.940000000000001 11  82.8 5.52 |  15.82 11  82.8 5.52
      15 | 15.260000000000002 11  75.1 5.01 | 15.620000000000001 11  82.4 5.49 | 15.580000000000002 11  82.4 5.49
      16 | 15.899999999999999 11  82.2 5.48 |  16.16 01  88.0 5.87 |  16.14 01  88.0 5.87
      17 |  15.86 11  85.3 5.69 |   16.1 01  90.2 6.01 | 16.080000000000002 01  90.2 6.01
      18 |  15.82 11  82.7 5.51 | 15.899999999999999 11  85.7 5.72 | 15.940000000000001 11  85.7 5.72
      19 |  15.16 11  74.9 5.00 |  15.68 11  83.1 5.54 |   15.7 11  83.1 5.54
      20 |  14.64 11  82.9 5.53 | 14.700000000000001 11  84.2 5.62 |  14.56 11  84.2 5.62
      21 |  16.26 01  80.4 5.36 |   None 00  82.4 5.50 |  16.48 01  82.4 5.50
      22 |  15.96 11  82.7 5.51 | 16.080000000000002 01  86.2 5.75 |  16.06 01  86.2 5.75
      23 | 14.040000000000001 11  76.0 5.07 |  14.08 11  78.6 5.24 |   14.1 11  78.6 5.24
      24 |  15.64 11  78.7 5.25 |   15.7 11  80.7 5.38 |  15.66 11  80.7 5.38
      25 | 15.440000000000001 11  76.9 5.13 |  15.46 11  77.5 5.17 |  15.48 11  77.5 5.17
      26 | 15.740000000000002 11  81.3 5.42 |  15.98 11  86.4 5.76 | 15.899999999999999 11  86.4 5.76
      27 |   15.5 11  77.5 5.16 |   14.3 11  79.9 5.33 |   14.3 11  79.9 5.33
      28 | 15.760000000000002 11  83.3 5.55 | 15.719999999999999 11  83.6 5.57 |  15.86 11  83.6 5.57
      29 |  15.68 11  79.5 5.30 | 15.780000000000001 11  81.3 5.42 | 15.740000000000002 11  81.3 5.42
     J0 t_success median 15.77 n 28 | pregrasp joint_dur median 5.39 | path median 80.8
     JD t_success median 15.94 n 27 | pregrasp joint_dur median 5.58 | path median 83.7
     JDR t_success median 15.9 n 28 | pregrasp joint_dur median 5.58 | path median 83.7
     {'signature': 'simple/perception-rgbd/pose-odometry/root-odometry/scalebfm', 'profile': 'configs/bench/simple.yaml'}
     J0 success_45 but t_success > 16.0 s (official limit): 5 [1, 2, 6, 11, 21]
     JD success_45 but t_success > 16.0 s (official limit): 11 [0, 1, 2, 6, 7, 10, 11, 12, 16, 17, 22]
     JDR success_45 but t_success > 16.0 s (official limit): 12 [0, 1, 2, 6, 7, 10, 11, 12, 16, 17, 21, 22]
     ```
3. **J_D 全平移规划成功 21/28**，预测 ≥ 24。其余 7 局 2 局在 2/3、5 局在 1/3 规划成功，没有一局退回 J 自己的规划（预测 ≤ 2，相符）。
4. **J_D 预抓取偏差 2.09 cm**，在预测区间 1.2–2.8 cm 内，但判据"J_D ≤ 2 cm 且 J0 ≥ 3 cm"差 0.09 cm 没够上（J0 3.86 cm）。J 的预抓取偏差 3.81 cm，略高于预测的 3–3.6 cm。
5. **M2 有两格出了 0.7–1.5 倍的预测**：XMoveBendPick 进入段，D 对 R_D 命令的误差 0.68 cm，是对 A 命令（0.43 cm）的 1.58 倍；TabletopGrasp 进入段 1.66 / 2.54 = 0.65 倍。其余各格在区间内。

### 事后计算（未登记）：各组的计算时间和 D 的前向计时

- 协调者 11-26 要求补的。只读正式运行的记录和主指标的行，不重跑，不改登记的结果。脚本和输出在 `/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/posthoc/`：
  - `compute_time.py`（719ff252）：各组加上自己的计算时间，重算两个时限的成功数；
  - `a_near.py`（be00897c）：A 的官方成功局离时限多远；
  - `plan_cost.py`（6179b568）：每个候选、每次 IK 检查的用时；
  - `timing.py`（354d60e8）：D 前向的离线计时，1 号卡，开跑前 1 号卡已用 17739 MiB。
- XMove 那 5 局的逐局数字来自一个一次性的私人脚本，读同样的 `rows/`（`t_success` 加 `think_s`），只摘了这 5 行。

```
$ compute_time.py run <main metric rows>
== success with each arm's logged compute time added to t_success (wall clock; see the docstring for what is counted)
   task                 arm  official +compute  45 s +compute  added per episode median / max (s); missing
   XMovePickTeleop      A      28/30    28/30  28/30   28/30     0.00 /   0.00; -
   XMovePickTeleop      RD     30/30    22/30  30/30   30/30     3.30 /   3.97; -
   XMovePickTeleop      SD     30/30    30/30  30/30   30/30     1.97 /   2.63; -
   XMovePickTeleop      J      28/30    28/30  28/30   28/30     0.00 /   0.00; -
   XMoveBendPickTeleop  A      27/30    27/30  27/30   27/30     0.00 /   0.00; -
   XMoveBendPickTeleop  RD     30/30     0/30  30/30   30/30     6.38 /  10.18; -
   XMoveBendPickTeleop  SD     30/30     7/30  30/30   30/30     3.32 /   5.41; -
   XMoveBendPickTeleop  J      27/30    27/30  27/30   27/30     0.00 /   0.00; -
   BendPickMP           A       0/30     0/30  10/30   10/30     0.00 /   0.00; -
   BendPickMP           RD      0/30     0/30  30/30   30/30     7.13 /   8.78; -
   BendPickMP           SD      0/30     0/30  30/30   30/30     3.83 /   5.44; -
   BendPickMP           J       0/30     0/30  10/30   10/30     0.00 /   0.00; -
   TabletopGraspMP      A      18/30    18/30  18/30   18/30     0.00 /   0.00; -
   TabletopGraspMP      RD     18/30     0/30  18/30   18/30     6.72 /   9.04; -
   TabletopGraspMP      SD     18/30     4/30  18/30   18/30     3.69 /   5.04; -
   TabletopGraspMP      J      26/30     2/30  30/30   30/30     2.54 /   3.49; -
   TabletopGraspMP      J0     23/28     0/28  28/28   28/28     4.19 /   5.16; -
   TabletopGraspMP      JD     16/28     0/28  27/28   27/28     3.81 /   6.67; -
   TabletopGraspMP      JDR    16/28     0/28  28/28   28/28     4.76 /   7.52; -
   four tasks (30-episode arms): A official 73 -> 73, 45 s 83 -> 83; RD official 78 -> 22, 45 s 108 -> 108; SD official 78 -> 41, 45 s 108 -> 108; J official 81 -> 57, 45 s 95 -> 95

== route plan times (t_plan_s, warm) on the 28 routed episodes, median / max; plus J_D's tries
   J         : median 2.54 s, max 3.49, n 28
   J0        : median 4.19 s, max 5.16, n 28
   JD scale 1: median 3.42 s, max 4.34, n 28
   (not counted) J0 ep 0: t_build_s 3.5, t_first_s 11.15

$ a_near.py <main metric rows>
  A XMovePickTeleop     : official 28/30, t_success within 0.4 s of the 16.0 s limit: 0 []; latest 12.44
  A XMoveBendPickTeleop : official 27/30, t_success within 0.4 s of the 16.0 s limit: 0 []; latest 13.64
  A BendPickMP          : official 0/30, t_success within 0.4 s of the 8.0 s limit: 0 []; latest None
  A TabletopGraspMP     : official 18/30, t_success within 0.4 s of the 16.0 s limit: 0 []; latest 13.18

$ (one-off) the 5 XMove episodes R_D won at the official limit, with R_D's think time added
    R_D official win ep 0: A t_success None, R_D t_success 12.56, think 3.58 s, sum 16.14 (limit 16.0)
    R_D official win ep 10: A t_success None, R_D t_success 12.540000000000001, think 3.12 s, sum 15.67 (limit 16.0)
    R_D official win ep 6: A t_success None, R_D t_success 13.24, think 6.66 s, sum 19.90 (limit 16.0)
    R_D official win ep 16: A t_success None, R_D t_success 13.26, think 5.92 s, sum 19.18 (limit 16.0)
    R_D official win ep 20: A t_success None, R_D t_success 13.24, think 6.66 s, sum 19.90 (limit 16.0)

$ plan_cost.py run
  pregrasp XMovePickTeleop      n  30: S_D think / 7 median 0.186 s; (R_D - S_D) / 6 median 0.147 s
  pregrasp XMoveBendPickTeleop  n  30: S_D think / 7 median 0.305 s; (R_D - S_D) / 6 median 0.320 s
  pregrasp BendPickMP           n  30: S_D think / 7 median 0.403 s; (R_D - S_D) / 6 median 0.395 s
  pregrasp TabletopGraspMP      n  30: S_D think / 7 median 0.384 s; (R_D - S_D) / 6 median 0.363 s
  pregrasp all four             n 120: S_D think / 7 median 0.348 s; (R_D - S_D) / 6 median 0.289 s
  entry    XMovePickTeleop      n  30: S_D think / 7 median 0.097 s; (R_D - S_D) / 6 median 0.077 s
  entry    XMoveBendPickTeleop  n  30: S_D think / 7 median 0.139 s; (R_D - S_D) / 6 median 0.153 s
  entry    BendPickMP           n  30: S_D think / 7 median 0.121 s; (R_D - S_D) / 6 median 0.121 s
  entry    TabletopGraspMP      n  30: S_D think / 7 median 0.104 s; (R_D - S_D) / 6 median 0.082 s
  entry    all four             n 120: S_D think / 7 median 0.110 s; (R_D - S_D) / 6 median 0.101 s

$ timing.py run/corr_RD models_D_s0 real/task/data/all.npz   (CUDA_VISIBLE_DEVICES=1)
segments 240 from /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126/run/corr_RD; torch 2.6.0+cu124; cuda True NVIDIA GeForce RTX 4090 D; load average (60.818359375, 63.185546875, 76.43896484375)
  cpu  threads 1 segments  1 (  5 member forwards) + to tensors : median     4.52 ms, p90     8.17 ms
  cpu  threads 1 segments  1 (  5 member forwards)              : median     2.96 ms, p90     3.35 ms
  cpu  threads 1 segments  7 ( 35 member forwards)              : median     5.11 ms, p90     5.63 ms
  cpu  threads 1 segments 35 (175 member forwards)              : median     9.17 ms, p90     9.49 ms
  cpu  threads 4 segments  7 ( 35 member forwards)              : median     2.45 ms, p90     2.77 ms
  cuda threads 1 segments  1 (  5 member forwards) + to tensors : median     8.83 ms, p90    12.91 ms
  cuda threads 1 segments  1 (  5 member forwards)              : median     4.10 ms, p90     6.91 ms
  cuda threads 1 segments  7 ( 35 member forwards)              : median     3.87 ms, p90     6.55 ms
  cuda threads 1 segments 35 (175 member forwards)              : median     4.96 ms, p90     7.62 ms
  GPU memory peak 44 MiB allocated
  per stage, D's part only: the switch's 7 sequential CPU calls ~ 32 ms; 7 sequential GPU calls ~ 62 ms; one GPU call on 7 candidates (35 forwards) ~ 3.9 ms
```

## 结论与下一步

> **定稿**（2026-11-26，独立复核回来以后；复核四条都判"支持"，带的限定已写进下面各条）。引用写成"结果 · 文件 · 节或行"，eval.txt 的节号就是它自己的 `== n.`；"复核"指下面"独立复核"一节。本节和结果、意外、复核一样，还没提交，等用户定。

### 结论

1. **45 s 下，自身模型 D 的一次性修正提高了 SIMPLE 直线伸手的成功率，提高全部来自 BendPick。**
   - P1 成立：判定集 R_D 48/60，A 28/60，+33.3 pp [+15.0, +53.3]；布局 + 8 / − 0（20 个布局），符号检验 p 0.0078（结果 · eval.txt §5 `P1 R_D vs A 45 s decision set`）。
   - 拆开看，BendPick 30/30 对 10/30，布局 + 8 / − 0，p 0.0078（§5 `R_D vs A 45 s BendPick`）；TabletopGrasp 18/30 对 18/30，逐局相同，+ 0 / − 0（§5 `R_D vs A 45 s TabletopGrasp`）。
   - F1 在 BendPick 上可信：0.60x，在范围内 88.3 %，IK > 2 cm 0.0 %（§7 `RD BendPick` 行和 `F1 RD BendPick : trustworthy`）。所以按判据，BendPick 的 +20 局记到自身模型名下。
   - TabletopGrasp 上 F1 判"不可信"（在范围内 63.3 %，§7 `F1 RD TabletopGrasp`），它的差是 0，照报，不归功。判定集合起来的 F1 也判"不可信"（75.8 %，§7 `F1 RD decision set 45 s`），原因就是 TabletopGrasp 这一半。
   - 到顶的两个任务也都到了 30/30：XMovePick 28 → 30，XMoveBendPick 27 → 30（§3 `-- 45 s`）。它们不在判定集里，只作描述。四个任务合计 83 → 108，布局 + 11 / − 0，p 0.00098（§5 `R_D vs A 45 s all four`）。
   - 机制和成功一致：BendPick 进入段手掌偏差中位数 6.19 → 1.25 cm，预抓取 3.01 → 0.96 cm（§6 `BendPick entry` / `pregrasp`）。这个下降落在预测区间内。
   - **限定一：BendPick 的结论只对这 10 个布局成立。** 翻过来的 8 个布局是同一种失败：姿态相同，侧向偏差方向相同（A 失败局的进入段偏差都在同一侧，y 向 −5.6 到 −6.9 cm），每个布局的 3 个 level 也几乎相同（复核"BendPick 的提高是不是真的"第 6 点）。所以这 +20 局实际上是一种失败被修好的 8 个实例，不是在多种伸手几何上都成立。换一种伸手几何还成不成立，没有检验过，由 selfmodel_simplecmp2_20261126 的 E1 来查。
   - **限定二：这批局严格说已经被看过。** "对准手掌"和各项预测，是登记前（dcheck.py、dstat.py，08:19–08:32Z；登记提交 08:53Z）在这 120 局的 A 记录上定的。登记前没有在这批局上跑过 R 或 S（复核"BendPick 的提高是不是真的"第 5 点）。风险低，但它们不是完全没看过的数据。
   - S_D 和 R_D 逐位相同，仿真也是确定的，所以"R_D、S_D 都 30/30"只算一个结果（复核"BendPick 的提高是不是真的"第 6 点）。
   - BendPick 的成功都落在 11.74–13.72 s，8 s 的官方时限本来就够不着（复核"BendPick 的提高是不是真的"第 3 点）。所以这条结论只在 45 s 口径下成立，不能当成 SIMPLE 的官方分数报。
2. **官方时限下，自身模型在 SIMPLE 上的贡献是四个任务合计 73 → 78（+5 局），不显著。** 主指标两种时限都报，所以两个数都写：
   - 45 s：83 → 108（§5 `R_D vs A 45 s all four`）；
   - 官方时限：73 → 78，+4.2 pp [+0.0, +9.2]，布局 + 3 / − 0（40 个布局），p 0.25；McNemar p 0.0625（§5 `R_D vs A official all four`）。
   - 这 5 局全来自 XMovePick（+2）和 XMoveBendPick（+3）。BendPick 的官方时限只有 8.0 s，各组都是 0/30（§3 `-- official`），所以 BendPick 的赢只在 45 s 下成立。官方时限的判定集是 18 对 18（§5 `P1 R_D vs A official`）。
   - **口径**：主指标的口径是仿真时间，计算时仿真暂停，所有栏一致。A 的 Motor.reach 规划、J 的 cuRobo 规划、S 栏和 kin_rr 的反算、本条目 R_D / S_D 的反解，都不计时（"注意"第 1 条）。上面两个数都是这个口径。
   - **另外按墙钟加上计算时间时**（事后，未登记；只读记录，不重跑；原始输出见"意外"末尾的"事后计算"）：把每局记录里的计算时间加到成功时刻上，按原时限重算。

     | 组 | 加上的计算时间（记录里的字段） | 每局中位（s） | 官方时限：原 → 加上后 | 45 s：原 → 加上后 |
     |---|---|---|---|---|
     | A | 记录里没有：Motor.reach 的规划没计时（缺） | — | 73 → 73 | 83 → 83 |
     | R_D | 两次反解的 think_s | XMovePick 3.30，XMoveBendPick 6.38，BendPick 7.13，TabletopGrasp 6.72 | 78 → 22 | 108 → 108 |
     | S_D | 同上 | 1.97，3.32，3.83，3.69 | 78 → 41 | 108 → 108 |
     | J（四个任务） | 有路线的 28 局：预抓取 cuRobo 规划的 t_plan_s；其余局同 A | TabletopGrasp 2.54 | 81 → 57（TabletopGrasp 26 → 2） | 95 → 95 |
     | J0（28 局） | 本轮规划的 t_plan_s | 4.19 | 23 → 0 | 28 → 28 |
     | J_D（28 局） | 各次尝试（1、2/3、1/3）的 t_plan_s 之和 | 3.81 | 16 → 0 | 27 → 27 |
     | J_DR（28 局） | J_D 的规划 + 进入段 R_D 的 think_s | 4.76 | 16 → 0 | 28 → 28 |

     - 分任务：R_D 是 XMovePick 22/30、XMoveBendPick 0/30、TabletopGrasp 0/30；S_D 是 30、7、4；A 是 28、27、18。XMove 那 5 局只剩 XMovePick 第 10 局（12.54 + 3.12 = 15.67 s）。
     - **45 s 下所有组都不受影响**。所以结论 1 不变。
     - **记录里缺的，都没算**：
       - 每一组直线伸手自己的 Motor.reach 规划（A 只有这一项）。按 S_D 每个候选的用时估计，每局约 0.46 s（预抓取 0.35 + 进入段 0.11）。A 的官方成功局最晚在 13.64 s，离 16.0 s 还有 2.36 s，所以这一项不改变 A 的数。
       - 看（感知）是离线预先算好的，各组相同。
       - 路线的障碍体素化和准备步骤没计时；J_D 用 D 算平移的时间没计时。
       - cuRobo 的 t_build_s（每个进程建一次规划器，约 3.5 s）和 t_first_s（首次编译，约 11 s）。
     - 运行计划里"一个规划约 67 s"是整个 plan2.py 进程的墙钟：启动、建规划器、首次编译、两次 plan_single、体素化和读写。热启动的单次规划只有 2.5–4.2 s，上表用的是这个。
     - 读法：官方时限很紧，按墙钟算，任何要在线计算的组都会掉，J 自己也从 81 掉到 57。所以这项计算比较的是"各组在现有实现下算得多慢"，不是方法本身。
   - **实现代价**（事后，未登记；离线用记录的段测，1 号卡，峰值 44 MiB，不改正式结果）：
     - R_D 的 4.5 s（预抓取中位）是未优化的实现：7 个候选逐个算，每个候选先用 Motor.reach 规划一次、再造特征、再调一次 D（5 个成员）。R 还为每个候选多做一次 IK 检查，又规划一次。都跑在高负载的 CPU 上：计时时 4090 的负载均值是 60–77，正式运行时还要加上我们自己的 5 个进程。
     - **D 自己的前向几乎不花时间**：CPU 单线程，一个候选（5 个成员）4.5 ms（含转张量）；7 个候选逐个算约 32 ms。把 7 个候选 × 5 个成员 = 35 次前向合成一批，放 GPU 上一次算完 3.9 ms，CPU 单线程合批 5.1 ms。D 只占预抓取思考时间的约 1 %。
     - **时间花在规划上**：S_D 每个候选约 0.35 s（预抓取）/ 0.11 s（进入段）；R 多出来的每次 IK 检查约 0.29 s / 0.10 s。也就是说，一个候选的代价约等于一次 Motor.reach 规划。R 的 IK 检查规划的是下一步评估马上又要规划的同一个候选，在现有代码里是重复的。
     - 所以这个代价主要是实现问题：串行规划、重复规划、负载高。方法本身的一部分是"每个候选要一个规划作为 D 的输入"，这一项省不掉，只能把规划做快或并行。另外，7 个候选是不动点迭代、前后相依，"合成一批"在现有方法下只能做到每一步 5 个成员一批；7 个候选一起算，是换成并行搜索。
3. **TabletopGrasp 的 18 对 18 只是计数，没有检验力，不能用来说明 D 有用还是没用。**
   - 自由的 18 局两组都全成，在天花板上；挡边的 12 局两组都全败，在地板上（§5 `TabletopGrasp free` / `blocked`；复核"TabletopGrasp 不变"一节）。这个比较区分不了"有作用"和"没作用"，加上 F1 不可信（在范围内 63.3 %），不能从它得出任何关于 D 的结论。
   - 能说的只有两件事实：挡边 12 局里 R_D 越过桌边 0 局，A 也是 0 局（§6 `TabletopGrasp blocked: R_D pregrasps within 5 cm`）；自由局的预抓取偏差 2.54 → 1.84 cm，但成功数已经没有上升空间（§6 `TabletopGrasp free`）。按判据，挡边布局交给路线 J。
4. **SIMPLE 上 R_D 和 S_D 没有区别，这次的结果不能用来选 R 还是 S。** 缩回一次都没触发，两组命令逐局相同，结果 120/120 相同（§1 `RD = SD`、`R_D episodes without a shrunk candidate`；§8 `shrunk candidates 0 of 0`）。所有发出命令的 IK 都 ≤ 2 cm（§8）。
5. **路线组：D 的平移让预抓取更准，但官方时限下少了 7 局。原因是每局慢了约 0.2 s，不是落点变差。**
   - 精度：J_D 预抓取手掌偏差中位数 2.09 cm，J0 3.86 cm（§8 `JD` / `J0 pregrasp / entry miss`）。判据"J_D ≤ 2 cm 且 J0 ≥ 3 cm"差 0.09 cm，没有达到。
   - 成功：45 s 下 J_D 27/28，J0 28/28（§5 `route J_D vs J0 45 s`）；官方时限下 16/28 对 23/28，差 −7 局，McNemar p 0.016，但布局 + 0 / − 4（10 个布局），符号检验 p 0.125（§5 `route J_D vs J0 official`）。
   - 区间有个小差异，照写：条目是 −25.0 pp [−43.3, −7.1]，复核算出 [−46.2, −7.1]。原因是每个布局的局数不等（index 3 只有 1 局），重抽时的加权不同。点估计和检验都相同（复核"数字核对"）。
   - 按判据，"J_D − J0 ≤ −3（官方）"这一行成立，下一步是路线组暂不用平移修正。
   - **原因：每局慢了约 0.2 s，不是落点变差。**
     - 少掉的 7 局在 45 s 上全部成功，只超时 0.06–0.22 s；J0 在这 7 局里是 15.86–16.00 s 成功（复核"路线组"一节；"意外"第 2 条的事后诊断也是这样）。
     - J_D 的预抓取时长 28/28 局都比 J0 长，中位数 +0.23 s；进入段时长相同。路径变长是因为平移把目标往前、往下挪了 3–6 cm（复核）。
     - 7 局都是完整平移规划成功的局，不是缩小平移的局（复核）。
     - "精度或抓取变差"不成立：45 s 上 27 对 28，唯一的失败是一局 2/3 平移、进入段偏 4.26 cm 的局，J_DR 修正进入段后这一局成功（复核）。
     - 背景：这个时限是一条刀刃。有路线的局都在 14.1–16.5 s 成功，J0 和 J 之间仅因重规划就翻了 1 局。
   - **保持原来时长的变体没有测过。** 平移修正本身能不能在不变慢的前提下保住成功，还不知道；由 selfmodel_simplecmp2_20261126 的 V1 来查。
   - J_DR 修好了进入段的偏差（1.41 cm，J_D 2.94 cm，§8），但官方成功和 J_D 一样是 16（§5 `J_DR vs J_D official`，+ 0 / − 0）。所以"J_DR − J_D ≥ +3"不成立。

### 证据强度

- **一次运行，一个模型**：D 是 seed 0、frac 1.0 的 E3_ensemble（5 个成员），没有换 seed。只有 ScaleBFM 一个控制器，只有感知栏（sensor=rgbd）一种设置。
- **局数**：直线三组各 120 局，即 4 个任务 × 10 个布局 × 3 个难度。判定集 20 个布局。BendPick 10 个布局里有 8 个还有提高余地，8 个全升；8 个布局时符号检验能达到的最小 p 就是 0.0078，所以它已经到了这个检验的上限。同一布局的 3 个难度不独立，McNemar 只作参考。
- **直线组没有重跑噪声**：AL 和主指标的 A 是两次独立运行，逐局相同（§1 `A = AL: 120 / 120`）。所以直线组的配对差是确定的，不确定性只来自布局的抽样。重跑同一个 R_D 只会得到同样的行，S_D 和 R_D 也逐位相同，都只算一个结果。
- **BendPick 的有效样本比 8 个布局还少**：翻过来的 8 个布局是同一种失败（同一姿态、同一侧向偏差）。符号检验把它们当 8 个独立布局，比实际的几何多样性乐观。
- **路线组有规划噪声**：cuRobo 的重规划每次不同。同为零平移的 J0 和 J，官方时限下 23 对 24，45 s 下 28 对 28（§5 `J0 vs J`）。J_D − J0 的 −7 比这个大，但布局层面不显著（p 0.125）。

### 其他解释

1. **状态相关预测和平均偏移的贡献没有拆开**：BendPick 的 +20 里，有多少来自 D 针对每一下状态的预测，有多少只是去掉了 A 命令的平均偏移，这次的结果分不开。
   - 有关的观察：每次修正的大小有起伏，预抓取中位 3.07 cm、最大 4.09 cm，进入段中位 3.13 cm、最大 7.32 cm（§8）。
2. **"思考时仿真暂停"本身改变了结果**：已排除。dryR（AL）也做了同样的反解和规划，只是照原样发命令，结果和 A 逐局相同（§1 `A = AL`）。
3. **BendPick 的失败其实来自感知，命令修得再准也没用**：被结果否定。登记时担心 BendPick 的目标偏差（seen_cm 约 10 cm）会限制成功，判据第 2 行就是为此写的。实际上 45 s 下 30/30。这说明在 BendPick 上，A 的失败主要来自执行偏差。至于 seen_cm 为什么没起作用（是偏差的方向抓取容忍得了，还是 seen_cm 没量到和抓取有关的那部分），这次的结果答不了。
4. **路线组变差来自规划噪声，不是平移本身**：可能性小。J0 和 J_D 在同一轮规划，零平移的 J0 对 J 只差 1 局（§5 `J0 vs J official`）；少掉的 7 局都是完整平移的局，J_D 的预抓取时长 28/28 局都更长（复核"路线组"一节）。但"时长"这个解释还只是相关，干预（V1）没做。

### 四类陷阱

- **benchmark 选择**：
  - 任务和局都是主指标原样的 4 个任务 × 0–29 局，登记时就定了；判定集照 lmlcmp3 的规则在登记里定好（§4）。
  - 到顶的两个任务被排除，但照样报了（§3、§5）。
  - BendPick 是登记时就指出的"唯一 F1 预计可信、又有提高余地"的任务，它的单独结果是预先写好的次要结果，不是事后挑出来的。
  - 局限：只有 SIMPLE 一个基准、ScaleBFM 一个控制器。我们的复刻流程在 BendPick 上够不着 8 s 的官方时限（成功最早 11.74 s），所以 BendPick 的结论只在 45 s 口径下成立，不能当 SIMPLE 官方分数报（复核"四类陷阱"）。
- **数据泄漏**：
  - D 的训练集是 selfmodel_simpledata 自己采的 data3、data3b 和中间点，都是机器人单独伸手的重置，没有用主指标的局。主指标 A 的局在 simpledata 里只用作评估 3。
  - 范围检查的阈值和训练集都和评估 4 相同（结果 · range.txt 第 1 行，p99 21.152）。
  - 开发和冒烟只用了开发局 30–39；对 0–29 局，登记前只跑过照原样发命令的 dryR 和开关关闭的核对。
  - 复核另查了：D 的输入里没有物体状态；骨盆高度和航向用的是仿真值，和 A 自己的规划器一样，影响很小（复核"BendPick 的提高是不是真的"第 4 点）。
- **指标误用**：
  - 两个时限都报了。P1 用 45 s 是登记时定的；官方时限的判定集结果（18 对 18）和四个任务合计（73 → 78）都写进了结论。
  - F1 用的是登记时定的任务标签，两个分母都报了（§7 每行的 `SIMPLE val` 和 `test_iid task-walk` 两个倍数）。
  - 手掌偏差用了仿真真值，但只作为度量，不进入任何决策。
- **事后挑选**：
  - 只有一次正式运行，没有重跑，没有换模型或 checkpoint，所有组全部报了。
  - **这批局严格说已经被看过**："对准手掌"和预测是登记前在这 120 局的 A 记录上定的（复核"BendPick 的提高是不是真的"第 5 点）。没有在这批局上先跑过 R 或 S，但它们不是完全没看过的数据。E1 用抖动后的新站位，部分补上这一点。
  - 事后分析有三项，都标了"未登记"：路线组的时刻诊断（复核独立核过，结论里写明"保持原时长的变体没测"）；所有组按墙钟加上计算时间的重算（只读记录）；D 前向的离线计时。后两项都在结论 2 里单列，不改登记的结果。

### 对照判据

| 判据行 | 结果 | 成立吗 |
|---|---|---|
| BendPick：R_D − A ≥ +5 局，p < 0.05，F1 可信 | +20，p 0.0078，可信（§5、§7） | **成立**（只在 45 s 下，只限这 10 个布局） |
| BendPick：+1 到 +4 或 p ≥ 0.05 / ≤ 0 | — | 不适用 |
| 任一任务偏差没降到预测区间 | 各任务都在区间内（§6） | 不成立 |
| P1 成立 | 48 对 28，p 0.0078（§5） | **成立**，赢来自 BendPick |
| 挡边越过 ≥ 4/12 | 0/12 | 不成立 |
| 挡边越过 ≤ 1/12 | 0/12（§6） | **成立**：交给路线 J |
| TabletopGrasp 自由局 R_D 比 A 少 ≥ 3 | 18 对 18 | 不成立 |
| F1 在某任务上不可信 | TabletopGrasp（§7） | **成立**：差为 0，不归功；18 对 18 本身也没有检验力 |
| R_D 和 S_D 命令不同的局 > 5 | 0（§1、§8） | 不成立 |
| 路线：J_D 预抓取 ≤ 2 cm 且 J0 ≥ 3 cm | 2.09 / 3.86（§8） | 不成立（差 0.09 cm） |
| 路线：J_D − J0 ≥ +3（官方） | −7 | 不成立 |
| 路线：J_D − J0 ≤ −3（官方） | −7（§5） | **成立**；原因是每局慢约 0.2 s，不是落点；保持原时长的变体未测 |
| 路线：J_DR − J_D ≥ +3 且进入段 ≤ 2 cm | 0；1.41 cm | 不成立 |
| 一致性不满足 | 全部满足（§1、offline.txt、jdr_vs_jd.txt） | 不成立 |

### 下一步

1. **先做 E1，再按判据第 1 行走。** 判据第 1 行的读法是：SIMPLE 上直线伸手默认用 R_D，下一步跑 SONIC。但复核指出 BendPick 的结论只限这 10 个布局、这批局已经被看过，所以先登记并跑 selfmodel_simplecmp2_20261126：
   - **E1（主项）**：BendPick 站位抖动 6 cm，A 对 R_D 配对，每组 60 次。判据照复核写的：
     - 推翻：R_D − A ≤ 0，或布局符号检验方向相反，或 R_D 进入段偏差中位数 > 2.8 cm；
     - 成立：R_D − A ≥ +5，布局 p < 0.05，抖动后 BendPick 的 F1 仍可信；
     - A 自己几乎不失败时，记为"证据不足"。
   - **E2**：只修进入段、只修预抓取，查"进入段偏 6.2 cm 是失败原因"这个解释。
   - **V1**：路线组只改时长（J0 放慢到 J_D 的时长、J_D 加快到 J0 的时长），查"慢 0.2 s"这个解释。
   - E3（没人看过的 BendPick 布局）先不做。
   - 写主指标时两个数都写：45 s 83 → 108，官方 73 → 78。
2. **计算时间**：已按墙钟给所有组重算（见结论 2，事后，未登记）。写进主指标时，官方时限的数字照主指标口径（仿真时间，计算时暂停，所有栏一致）。墙钟口径只作说明，各组一起报，不单列 R_D。要在墙钟口径下比，先把规划做快：去掉 R 的重复规划，合批或并行；D 的前向不是瓶颈。
3. **TabletopGrasp**：挡边布局交给路线 J。TabletopGrasp 上 R_D 不记功，18 对 18 也不当作"没用"的证据。
4. **路线组**：按判据暂不用平移修正，等 V1。
5. **beliefs.md**：等用户定了这次的改动再增量更新。

## 独立复核

> 2026-11-26，协调者安排的独立 agent 的回复，原样贴入（源文件 `/Users/zed/.claude/jobs/f13ab968/tmp/review_scmp/review_reply.md`，md5 d78009ca）。复核只读了假设、预测、判据、实现和结果，没读意外和结论。

### 判定

| 结论 | 判定 | 一句话 |
|---|---|---|
| (a) P1：判定集 45 s，R_D 胜过 A | 支持 | 48/60 对 28/60；按布局 +8/−0，符号检验 p = 0.0078。增量全部来自 BendPick，TabletopGrasp 贡献 0 |
| (b) BendPick 在 45 s 上从 10/30 升到 30/30 | 支持（只限这 10 个布局） | 数字和机制都对得上，没找到人为原因。8 个翻过来的布局失败方式相同，换一种伸手几何还成不成立，没有检验过 |
| (c) 路线组官方时限 J_D 16 对 J0 23 | 支持（数字和"变慢"的解释都成立） | 少掉的 7 局在 45 s 上全部成功，都只超时 0.06–0.22 s。这说明的是时间，不是精度 |
| (d) TabletopGrasp 18 对 18 | 支持（只作为计数） | 18 个自由局两组都全成，12 个挡边局两组都全败。这个比较没有检验力，不能读成"D 对 TabletopGrasp 没用"，也不能读成"有用" |

### 怎么核的

- **预注册**：把 `git show 31a65aa0` 的条目和现在的文件做了 diff。
  - 只有三处新增：第 363 行之后、"结果"段和"意外"之后，前面的假设、预测、判据、做法一字未改；
  - "预测修订"是空的。
- **冻结**：在 4090 上用 `md5sum -c` 重跑了 `asrun/code/CODE_MD5`、`KEY_MD5` 和 `INPUTS_MD5`，全部通过。
  - Mac 上 5 个 scmp 脚本的 md5 和 KEY_MD5 一致；
  - simple_bench.py 一致（8d873cbb）；
  - 主指标 A、J 行的 md5 和 INPUTS_MD5 一致。
- **自己写的脚本**：放在 `/Users/zed/.claude/jobs/f13ab968/tmp/review_scmp/scripts/`，没用 scmp_eval.py。
  - `recount.py`：成功数和配对检验；
  - `ident.py`：逐局逐字段比较；
  - `bendpick.py`、`bp2.py`：BendPick 逐局；
  - `route.py`：路线组逐局计时；
  - `f1.py`：F1 判定。
  - 原始行下载到同目录的 `data/` 里。远程只读。

### 数字核对（从原始行重算）

先核了成功标志：9 组共 592 行，`success_45` 和 `success_official` 都和 `t_success ≤ 45 / ≤ 官方时限` 逐行一致，0 处不符。

成功局数（45 s / 官方时限）：

| 任务 | A | AL | R_D | S_D | J | JL | J0 | J_D | J_DR |
|---|---|---|---|---|---|---|---|---|---|
| XMovePick | 28 / 28 | 28 / 28 | 30 / 30 | 30 / 30 | 28 / 28 | | | | |
| XMoveBendPick | 27 / 27 | 27 / 27 | 30 / 30 | 30 / 30 | 27 / 27 | | | | |
| BendPick | 10 / 0 | 10 / 0 | 30 / 0 | 30 / 0 | 10 / 0 | | | | |
| TabletopGrasp（30 局） | 18 / 18 | 18 / 18 | 18 / 18 | 18 / 18 | 30 / 26 | | | | |
| TabletopGrasp 有路线的 28 局 | | | | | 28 / 24 | 28 / 24 | 28 / 23 | 27 / 16 | 28 / 16 |
| 其中自由 16 局 | | | | | 16 / 15 | 16 / 15 | 16 / 15 | 16 / 12 | 16 / 12 |
| 其中挡边 12 局 | 0 / 0（A、R_D、S_D） | | | | 12 / 9 | 12 / 9 | 12 / 8 | 11 / 4 | 12 / 4 |

配对检验。布局 =（任务，index），区间按布局重抽 4000 次，seed 0：

| 比较 | 时限 | 成功数 | 布局 + / − | 符号检验 p | 局 + / − | McNemar p | 差及区间 |
|---|---|---|---|---|---|---|---|
| P1：R_D 对 A，判定集 | 45 s | 48/60 对 28/60 | 8 / 0（20 个） | 0.007812 | 20 / 0 | 1.907e-06 | +33.3 pp [+15.0, +53.3] |
| P1 | 官方 | 18 对 18 | | 1 | | | |
| BendPick | 45 s | 30 对 10 | 8 / 0（10 个） | 0.007812 | 20 / 0 | 1.907e-06 | +66.7 pp [+40.0, +90.0] |
| TabletopGrasp | 两个时限 | 18 对 18 | 0 / 0 | 1 | 0 / 0 | | |
| J_D 对 J0 | 官方 | 16/28 对 23/28 | 0 / 4 | 0.125 | 0 / 7 | 0.01562 | −25.0 pp |
| J_D 对 J0 | 45 s | 27 对 28 | | | | | |

- 和"结果"段比，所有计数、p 值、直线组的区间都一致。
- 唯一不一样的是路线组的区间：J_D 对 J0 官方时限我得到 [−46.2, −7.1]，条目是 [−43.3, −7.1]。原因是每个布局的局数不等（index 3 只有 1 局），重抽时怎么加权会有差别。点估计和检验都相同，不影响结论。
- 一致性检查我自己重做了：只去掉 wall_s、correct_reset、film、sonic_vr_frame 和每一下的 correct 字段，连 trace 一起比。
  - AL = A：120/120；
  - JL = J：28/28；
  - S_D = R_D：120/120；
  - R_D 的缩回 0 次，退回目标 0 次。

### BendPick 的提高是不是真的

1. **两组只差发出的命令。**
   - 发第一条命令之前的字段，R_D 和 A 逐局全同：seen_cm、walk、walk_err_cm、odo、bel、grasp_h、pz、pre_dir、obj_from_grasp、score，以及预抓取命令时刻的关节 S0，30/30。
   - 运行级设置全同：method、stand、side_off、column、cloud、controller 和 bench 选项，sonic_vr_frame 除外。
   - AL = A 说明，在候选上做规划这一步本身没有副作用；plan_same 120/120。
   - 之后的差别只来自命令。注意，进入段的起点 = 实测手 + 0.5 ×（上一条命令 − 实测手），所以修正后的预抓取命令也会挪动进入段的起点。这属于"命令"的一部分。
2. **成功判定方式相同。** 代码同一份（md5），标志和 t_success 逐行一致。R_D 的 30 次成功都是真举起来：
   - success_end 30/30，物体升高 15.9–22.4 cm；
   - 闭合时物体只挪动 0.5–4.7 cm；
   - 对比 A 的失败局：物体最后比初始低 0–7.78 cm，是被碰掉了。
3. **时间。** "提高只出现在 45 s"不是在挑时限，而是 8 s 的官方时限本来就够不着：
   - BendPick 的闭合要到第 10 s 左右才开始，各组闭合开始时刻的中位数：A 9.98 s，R_D 10.36 s；
   - 所有成功都落在 11.74–13.72 s（A 11.74–12.88，R_D 11.84–13.72）；
   - 整局最晚 17.1 s 就结束，13.72 s 以后什么也不会再发生；
   - 所以只要时限 ≥ 13.72 s，都是 30 对 10。
   - 把思考时间也算进去，R_D 每局中位数 7.1 s，最多 8.8 s，最晚 21.9 s 成功，在 45 s 上仍是 30/30。
   - 但要知道：算上思考时间，R_D 在其他任务上的官方时限结果会垮掉：XMovePick 22/30，XMoveBendPick 0/30，TabletopGrasp 0/30。思考更快的 S_D 是 30、7、4，而 A 是 28、27、18。不算思考时间是预注册写明的，不影响本条的 45 s 结论。
4. **真值有没有进 D 或决策：没有发现。**
   - D 用到的输入：关节读数（速度由差分得到）、由仿真骨盆姿态算出的重力方向和航向、由关节和手腕算出的手点、规划的参考，以及感知给的目标（经 place.Belief.to_sim）。
   - 走路特征在训练时和这里都是 0。`walk_real` 只是记录字段，pack 之后进不了 x0。
   - 不含任何物体状态。
   - 仿真骨盆的高度和航向用的是仿真值，而不是腿部运动学或 IMU 积分的估计。这一点和 A 自己的规划器一样，不是 D 独有的泄漏，影响很小。
5. **训练集和测试局重叠：没有。**
   - 训练文件是 data3、data3b、mid_data3、mid_data3b：
     - 重置编号 50,000,000–50,029,999 和 60,000,000–60,014,999；
     - 在空地场景 Env("open") 里采集，目标来自伸手表。
   - 90M–99M 段 0 个重置。测试局的记录在 91M（real/task）和本条的 95M–97M。
   - 标准化统计量是模型里的 buffer，load_state_dict 会覆盖由模板重建时算出的值。模板只提供形状。
   - 有一点要记下：登记之前（dcheck.py、dstat.py，08:19–08:32Z，提交是 08:53Z），用这 120 局里 A 的记录选定了"对准手掌"，也定了预测。这次复核没有发现登记前在测试局上跑过 R 或 S：checks/fake_run 是把 AL 的行软链成 RD、SD，R 的冒烟只用了开发局 30–39。风险低。但这批局严格说已经不是完全没看过的数据。
6. **机制。**
   - A 失败局的进入段偏差中位数 6.55 cm，成功局 2.34 cm。失败局的偏差方向几乎一样：都在同一侧，y 向 −5.6 到 −6.9 cm。
   - R_D 的进入段偏差 0.28–1.66 cm，中位数 1.25。
   - D 对 A 命令预测的进入段偏差，在 30 局上和实际偏差的相关是 0.98：预测成功局 2.96 cm、失败局 5.04 cm，实际是 2.34 和 6.55。
   - 所以 D 确实分得出哪些局会偏得多。但翻过来的 8 个布局是同一种失败的 8 个实例：同一个姿态、同一个侧向偏差；每个布局的 3 个 level 几乎相同。
   - S_D 和 R_D 逐位相同，仿真也是确定的，所以"R_D、S_D 都 30/30"只算一个结果。

### 四类陷阱

- **Benchmark 选择：没有发现问题。**
  - SIMPLE 是导师定的；四个抓取任务就是主指标那四个；判定集按预注册的到顶规则定。
  - 把到顶的两个任务也算进去，方向不变：四任务合计 108 对 83，局 +25/−0，布局 +11/−0，p = 0.00098。
  - 要写明一件事：我们的复刻流程在 BendPick 上够不着 8 s 的官方时限，所以 BendPick 的结论只在 45 s 口径下成立，不能当成 SIMPLE 官方分数报。
- **数据泄漏（包括标准化统计量）：没有发现**，见上面第 4、5 点。
- **度量误用：没有发现。**
  - P1 和 BendPick 的 45 s 口径、任务标签、两个分母，都是预注册的；
  - 官方时限的结果照实报了：P1 官方时限 18 对 18，p = 1；
  - 路线组的 −7 也报了；
  - 不算思考时间是预注册写明的，BendPick 在 45 s 上不受它影响。
- **事后挑选：没有发现。**
  - 预注册内容没改，预测修订是空的；
  - 模型（frac 1.0，seed 0）事先固定；
  - 各组各比较都报了，包括不利的；
  - 没找到被丢弃的正式运行。
  - 路线组"变慢"的解释是事后给的，但可以核，下面核了。
- **F1 是照预注册执行的。** 我从 range.json 的逐下记录和 rows 的 ik_cm 重算：

  | | D 误差中位数 | 倍数（÷1.886 / ÷0.993） | 在范围内 | IK &gt; 2 cm | 判定 |
  |---|---|---|---|---|---|
  | BendPick | 1.135 cm | 0.60x / 1.14x | 88.3 % | 0 % | 用哪个分母都可信 |
  | TabletopGrasp | 2.106 cm | | 63.3 % | | 不可信 |
  | 判定集 | 1.548 cm | 0.82x / 1.56x | 75.8 % | | 不可信 |

  - AL 的参照值也和条目一致。
  - 按读法，P1 的赢只记在 BendPick 名下，和条目一致。
  - BendPick 的进入段有 23.3 % 不在范围内（出界的是腕部关节），这些局也都成功了。

### 路线组：官方时限少 7 局

- **数字成立。** 少掉的 7 局是 (L0,e0)、(0,7)、(1,0)、(1,2)、(1,6)、(1,7)、(2,2)：
  - J0 在 15.86–16.00 s 成功，其中 3 局正好 16.00 s；
  - J_D 在 16.06–16.22 s 成功，7 局在 45 s 上全部成功；
  - 7 局都是全量平移规划成功的局。全量规划的 21 局里，官方时限 J_D 12 对 J0 19。
- **"路径变长导致变慢"有数据支持。**
  - J_D 的预抓取时长在 28/28 局都比 J0 长：中位数 +0.23 s，范围 0.02–0.74 s；7 个丢局里是 +0.23 到 +0.39 s；
  - 掌心路径长度中位数 83.7 对 80.8 cm，Motor 按路径长度和固定速度定时长；
  - 进入段时长相同：3.90 对 3.99 s；
  - t_success 之差中位数 +0.14 s，它和"预抓取结束时刻之差"相差的中位数只有 0.08 s。
- **路径为什么变长。** 平移主要是向前 2.9–5.8 cm、向下 0.6–4.1 cm（D 预测 J 的掌心会少到约 4 cm），目标更远了。cuRobo 规划出的终点关节和 J 差 15–89°，J0 只差 0–16°。
- **"精度或抓取变差"这个解释不成立。** 45 s 上 J_D 27 对 J0 28。唯一的 45 s 失败是 (2,1)：2/3 平移，进入段偏差 4.26 cm。J_DR 用同一条规划、修正了进入段，这一局在 16.48 s 成功。
- **背景：这个时限是一条刀刃。**
  - 有路线的 TabletopGrasp 局都在 14.1–16.5 s 成功；
  - J0 有 8 次成功落在 (15.8, 16.0] s；
  - J0 和 J 之间仅因为重规划就翻了 1 局：(0,2) 从 15.96 变成 16.02 s。
  - 所以"官方时限 −7"量的是每局约 0.2 s 的时间，不是落点精度。照判据读成"路线组不再用平移修正"没有违反预注册，但原因是时间。保持 J 原有时长的变体能不能消掉这个代价，没有测过。
- **另两条对不上预测。**
  - 预注册的"J_D 预抓取偏差 ≤ 2 cm 且 J0 ≥ 3 cm"差一点没达到：2.09 对 3.86 cm；
  - 全量平移规划成功 21/28，预测是 ≥ 24。

### TabletopGrasp 不变

- **计数相同。** 两个时限都是 18 对 18，而且是同样的 18 局（局 +0/−0）。
- **自由局在天花板上。** 两组都 18/18，成功都在 11.6–13.2 s，离 16 s 很远，所以官方时限和 45 s 结果一样。
  - R_D 把精度提高了：预抓取偏差 2.54 → 1.84 cm，进入段 2.03 → 1.45 cm；
  - 但成功数已经没有上升空间。
- **挡边局在地板上。** 两组都 0/12。R_D 的预抓取偏差 28.81 cm，A 是 29.27 cm，越过桌边 0/12。
- 所以"不变"只是一个计数，这个比较没有能力区分"有作用"和"没作用"。加上 F1 不可信（在范围内 63.3 %），不能从它得出任何关于 D 的结论。

### 可能推翻结论的实验

要推翻的结论是：在 D 上反解能提高 BendPick 的 45 s 成功率。

**E1（主推，4090 上约 20 分钟）：换伸手几何，A 和 R_D 配对。**
- **做法**：在 BendPick 的 30 局上加 `--stance-jitter 0.06`，分别用 `--jitter-rep 1` 和 `--jitter-rep 2`，`--jitter-seed 0`。
  - 每局的抖动按（任务，level，index，rep）抽，两组用同一次抽样；
  - 感知目标不变，站位前后左右最多挪 6 cm，伸手的距离和方向随之改变；
  - 每组 60 次新的伸手，2–3 个进程。
- **为什么做这个**：
  - 翻过来的 8 个布局是同一个姿态、同一侧向的偏差；
  - 同一 index 的 3 个 level 几乎相同；
  - "对准手掌"和预测都是在这批局的 A 记录上定的。
  - 这正是"要在多种高度、方向上证明模型的功劳"该检验的地方。
- **判据（开跑前写进新条目）**：
  - 推翻：R_D − A ≤ 0（60 次伸手中），或按 10 个布局的符号检验方向相反，或 R_D 进入段偏差中位数 &gt; 2.8 cm；
  - 成立：R_D − A ≥ +5，布局符号检验 p &lt; 0.05，并且抖动后的起点上 BendPick 的 F1 仍可信；
  - 如果抖动后 A 自己几乎不失败，就没有提高的余地，记为"证据不足"，不算支持。

**E2（查机制，每组约 10 分钟）：只修一段。**
- 在同样 30 局上分别跑 `--correct-stages entry`（只修进入段）和 `--correct-stages pregrasp`（只修预抓取）。
- 如果只修进入段收不回 +20 的大部分，那么"进入段偏 6.2 cm 是失败原因"这个解释就是错的，哪怕 R_D 整体的提高依然成立。

**E3（更强，但更贵）：没人看过的 BendPick 布局。**
- 如果 SIMPLE 给 BendPick 也留了没用过的 hard-old / hard 两级（`SIMPLE_LEVELS=3,4`），就为这 20 局各做一次观察和抓取生成，再跑 A 对 R_D 的配对。

**补充**：仿真是确定的，重跑同一个 R_D 只会得到同样的行。能换的只有几何（E1、E3）或模型种子：D 换 seed 1 重训，4090b 上约 9 分钟。

### 说明

- 按要求没有读"意外"和"结论与下一步"。
- 远程只读；没有提交，也没有改条目或任何输出。
