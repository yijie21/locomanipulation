# selfmodel_bench_20261123：多路点自身模型放回 SIMPLE 的抓取任务，感知栏，ScaleBFM 和 SONIC 两个控制器

- 日期：2026-11-23
- 机器 / GPU：vastgpu，只用 GPU 1。
- 代码版本：`selfwm` @ `12a675b6` 加未提交的改动（本条目提交时评测脚本的改动还没写完，跑之前的版本会抄进输出目录的 `asrun/`）。
- 配置：SIMPLE 官方评测局，四个抓取任务各 30 局（XMovePickTeleop、XMoveBendPickTeleop、BendPickMP、TabletopGraspMP）；只跑感知栏（`simple_bench.py --column perception --cloud cloud_eyes_final`）；机器人 G1 加 Dex3 手；控制器 ScaleBFM 和 SONIC（`REACH_CONTROLLER=sonic REACH_SONIC_NETS=allcpu`），都不改参数。
- 关联：
  - [多路点自身模型](selfmodel_via_20261120.md)：空地回放上的四批测试，以及 SONIC、AMO 的结果；
  - [curobo_avoid_20261120](curobo_avoid_20261120.md)：4090 上从机器人自己那一眼的点云规划绕行路线的整条流程，这里照用；
  - 用户 2026-11-23：“只跑感知栏，然后我们不是做了 scalebfm 和 sonic 两个模型吗，就把这两个作为底层控制器都跑了”。

## 要回答的问题

多路点自身模型到现在只在空地回放上验证过：障碍物不在仿真里，也没有抓取任务。放回基准之后：

1. 桌子真的在仿真里时，沿规划路线伸手，挪过命令是不是比原样发命令碰得少、到得准？
2. 任务成功的局数有没有变化？
3. 换成 SONIC 是不是同样的结论？

## 做法

每个控制器各自做一遍，互不共用路线，因为两个控制器走到站位后的姿态不一样。

- **第一遍，直线伸手**（基准原来的做法，记作 A）：`--method direct`，不带路线，带 `--film --trace`。这一遍同时给出每局伸手开始时机器人自己知道的状态（关节读数、里程计认为的骨盆位置），规划要用。
- **规划**（照 curobo_avoid 的流程）：障碍物是开局那一眼的点云去掉机器人自己和目标之后的 2 cm 体素（`obstacles_box.py`）；cuRobo 只放开右臂 7 个关节，腰和手锁住，激活距离 3 cm，失败的局用 2 cm 重试一次（`plan2.py`，和 4090 当时事先登记的设置相同）；路线压成路点，容差 1 cm（`via2.py`）。
  - 和 4090 那次有一处不同：体素按机器人里程计认为的骨盆位置搬到伸手开始时的身体坐标里，不用仿真里骨盆的真实位移。目标本来就是这样处理的。
- **第二遍，沿规划路线，原样发**（记作 B）：`--avoid`，`--method direct`。
- **第三遍，沿规划路线，挪过命令**（记作 C）：同一条路线，预抓取这一段伸手的路点和目标先用该控制器自己的多路点模型挪过再发。ScaleBFM 用 `via_point_20261121`，SONIC 用 `via_point_sonic_20261123`，两个都只见过空地上的随机路线。补偿设置和第四批测试相同：只挪位置，上限 8 cm。
- 三遍里最后 8 cm 的慢速进入都原样发。多路点模型的训练数据里没有这一段，它管不到。
- 没有规划出路线的局，B 和 C 都退回直线伸手，这样每个任务仍是 30 局。
- 没有用仿真真值：目标、抓取、障碍物都来自开局那一眼（Isaac 画面、双目深度、SAM 3、GraspGen-X），走过去之后只靠腿部里程计。成功判定是 SIMPLE 自己的。
- 没有开发局。公开的 30 局既是测试局，所以全部设置在跑之前定死，都取已有的默认值或 4090 当时登记的值，不看结果调。流程能不能跑通，只在每个任务的第 0 局上试；试跑之后除了修程序错误不改任何设置，改了什么都记在结果里。

## 度量

- **成功**：SIMPLE 的判定，目标被抬起。报两个口径：45 秒内，和官方时限内（BendPick 8 秒，其余 16 秒）。
- **预抓取偏差**：预抓取这一段停稳后，手掌离想要的预抓取点多远。
- **进入偏差**：慢速进入停稳后，手掌离抓取点多远。
- **伸手途中的真接触**：预抓取这一段里（从发命令到停稳），右臂从肩到手的任何一个连杆和桌子或者目标以外的物体有穿透。按局算比例。左手、躯干靠在桌沿上的接触不算在内，单独报。
- **目标被碰动**：合手之前目标移动超过 1 cm。
- 主要的比较是 C 对 B，只在两遍都用了路线的局上配对。A 作参照。区间按局自助抽样。

## 写预测之前已经知道的

- ScaleBFM，感知栏，Isaac 那一眼，直线伸手（4090，11-20）：45 秒内成功 XMovePick 23、XMoveBendPick 25、BendPick 30、TabletopGrasp 2；BendPick 在官方 8 秒内是 0。
- 4090 的 curobo_avoid：
  - TabletopGrasp 带路线后直线伸手的“手被桌沿挡住”从 18 局降到 1 局，45 秒成功 12 / 29（原样发），用单条命令的自身模型修最后的命令是 17 / 29；
  - 路线没被照着走，手离规划 4.1 cm，28 / 28 局手指越过桌沿时擦到桌面；
  - 路线约 80 cm，成功时刻推到 16 到 19 秒，多数超过官方的 16 秒；
  - 另外三个任务里，只放开右臂的规划器在 83 / 90 局够不到预抓取点，只有 7 局有路点。
- 今天在这台机器上各试了一局 XMoveBendPick 第 0 局，直线伸手：ScaleBFM 12.5 秒抓起，进入偏差 2.8 cm；SONIC 没抓起，进入偏差 4.8 cm。
- 空地回放（第四批）：ScaleBFM 挪过命令后手掌终点偏差 3.4 降到 0.9 cm；SONIC 5.2 降到 1.45 cm，但它的偏差背着障碍物，补偿没有少占间隙，蹭到的略多。
- 真实的手比规划用的球大 1 到 2 cm。

## 预测（启动前写）

**有路线的局数**：ScaleBFM 在 TabletopGrasp 上 26 到 29 局，另外三个任务合计 3 到 12 局。SONIC 的站姿不同，TabletopGrasp 20 到 29 局，另外三个任务合计 0 到 15 局。所以 C 对 B 的比较基本只在 TabletopGrasp 上。

**ScaleBFM，有路线的局：**

- 预抓取偏差中位数：B 2.5 到 4.5 cm，C 0.6 到 1.6 cm。
- 进入偏差中位数：B 和 C 都在 2 到 4 cm，C 比 B 小不超过 1 cm，因为最后一段没有补偿。
- 伸手途中右臂的真接触：B 80 到 100 %，C 45 到 90 %。
- TabletopGrasp 45 秒成功：A 1 到 4 局，B 9 到 15 局，C 10 到 18 局。C 比 B 多 0 到 5 局。
- 另外三个任务三遍的成功局数几乎一样，每个任务相差不超过 2 局。

**SONIC：**

- 直线伸手 A 的 45 秒成功：XMovePick 3 到 15，XMoveBendPick 2 到 15，BendPick 5 到 25，TabletopGrasp 0 到 3。范围宽是因为没有先例。
- 有路线的局，预抓取偏差中位数：B 4 到 7 cm，C 1.2 到 3 cm。
- 伸手途中右臂的真接触：B 30 到 90 %，C 不比 B 低，可能更高。
- TabletopGrasp 45 秒成功：B 0 到 6 局，C 0 到 8 局。
- 摔倒：每遍不超过 3 局。

**我预计会成立的**：两个控制器上 C 的预抓取偏差都明显小于 B。**我预计分不出的**：成功局数。**我没有把握的**：ScaleBFM 上真接触会不会减少。

## 判据（每个控制器各自判）

- **到得准**：有路线的局里，C 减 B 的预抓取偏差，逐局配对差的中位数为负，区间不含 0。
- **碰得少**：有路线的局里，伸手途中右臂真接触的比例 C 低于 B，差的区间不含 0。区间含 0 记为分不出；C 高于 B 且区间不含 0 记为更差。
- **任务成功**：四个任务 120 局的 45 秒成功局数，C 减 B，区间不含 0 才算有差别。只有 C 成功和只有 B 成功的局数都报。
- **有效性**，不满足的局不进配对比较并计数：
  - 第二、三遍到达伸手起点时的状态和第一遍一致，判法是规划的终点离这一遍自己的预抓取点不超过 1 cm；
  - C 的摔倒不比 B 多 3 局以上，否则整个比较不算。
- 哪一条不成立都照实写，不换设置重跑。

## 已知的局限

- 公开的 30 局既是开发又是测试，没有另一批可以调设置的局。
- 规划器锁着腰，够不到的预抓取点规划不出路线，所以四个任务里实际受检验的主要是 TabletopGrasp。
- 最后 8 cm 没有补偿，成功率能动的空间有限。两个控制器的数据里都录了这一段（ScaleBFM 10000 次，SONIC 11919 次），可以另训一个模型，这次不做。
- TabletopGrasp 找目标的规则是 11-20 看过失败之后换的，等于在测试局上调过，这一点沿用旧结论时已经说明。
- 每种做法一个种子，跑一次。

## 补充：两个对照（2026-11-24 写）

写这一节的时候，ScaleBFM 的 A、B、C 三遍已经跑完并且看过结果；SONIC 的三遍还在跑，结果没看；两个对照一局都没跑过。所以这是看过一部分结果之后加的对照，不是事先的设计，照这个身份报告。

### 为什么加

C 和 B 的差别，可能来自模型对每一局的预测，也可能只来自“路线整体挪了一个大致固定的量”。两种来源要分开，否则 C 的结果没法归到模型头上。

### 做法

两个对照都沿用同一批规划路线，每个控制器各 120 局，其余设置和 B、C 相同。

- **D，固定后撤**：路线的每个路点和预抓取目标都挪同一个量，在发命令时的朝向坐标里是向后 6 cm、向上 2 cm，左右不动。不用模型，不看这一局。这个量是照 ScaleBFM 的 C 在桌面抓取上挪目标的典型值取整得到的，两个控制器用同一个量，SONIC 的 C 是什么样没有参与。
- **E，常数修正**：`REACH_VIA_COMP_MODE=const`，每条路线都按模型文件里存的训练集平均偏差来挪，一轮。ScaleBFM 的这个量在终点约是向右 2.4 cm，SONIC 的约是向后 4.2 cm、向上 7.3 cm。它不看这一局的路线和状态。

### 预测

- **D 和 C 差不多**（我估计有六七成把握）：有路线的局里，D 的 45 秒成功局数和 C 相差不超过 3 局，伸手途中右臂真接触的比例和 C 相差不超过 0.15，预抓取偏差中位数和 C 相差不超过 2.5 cm。
- **E 在 ScaleBFM 上更像 B**：它挪的量小，而且方向是向右，不是向后。成功局数和 B 相差不超过 3 局。
- SONIC 上 E 挪的量大，结果没有把握，不写范围。

### 判据（每个控制器各自判）

- **模型的功劳**：有路线的局里 C 的 45 秒成功局数多于 D，逐局配对差的区间（按局自助抽样）不含 0，才能写“增益来自模型对每一局的预测”。
- C 和 D 分不出，或者 D 更好：写成“增益来自把预抓取点后撤，换成一个固定的量也一样”，不归到模型头上。
- 接触比例同样判一遍。
- D 和 E 只是对照，不当成方法，也不拿来挑一个更好的设置。

## 结果（贴原始输出，不转述）

### 跑的经过

- **试跑**（每个任务第 0 局，ScaleBFM，11-23 16:01 UTC，80 秒）：四局里只有 TabletopGrasp 那一局规划出路线。试跑之后只加了几个记录用的字段（模型对原命令的预期 `comp_pred0`、目标挪了多少 `comp_shift_goal`、想要的路线里手腕最高到哪 `comp_wrist_top`、发命令时的 29 个关节 `S0`），没有改任何设置。
- **ScaleBFM**：11-23 16:06–16:41 UTC，`bench_via.sh ... all`，输出在 `outputs/simple_eval_20261112/bench_via_20261123/scalebfm`，跑的脚本抄在它的 `asrun/`。
- **SONIC**：
  - 第一次 11-23 16:41 UTC 启动，机器过载后重启，只跑了 A 的 25 局。这些行留在 `sonic_aborted_20261123`，没有用。
  - 第二次 11-24 01:07–01:48 UTC，输出在 `.../sonic`。A 里 XMovePickTeleop 的进程在第 13 局之后无报错退出，单独重跑第 14 局正常。第 14–29 局补跑了 A（两个进程），补做了规划（8 局出了路线），这 8 局的 B 和 C 也重跑并替换。没带路线时的旧行留作 `*.straight_eps14-29.jsonl`。
- **对照**（提交 `97ad6aa6`）：ScaleBFM 的 D、E 11-24 02:03–02:10 UTC 跑完。SONIC 的 D 跑到 46 / 120 局时机器又重启，E 没开始，**SONIC 的对照没有结果**，没有重跑。
- **对照的预测是什么时候写的**：正文写于 11-24 01:09 UTC 前后，那时 SONIC 的结果还没出来。提交是 01:49:30，在那之前一分钟我已经读了 SONIC 的表。预测一个字没改，但提交时间证明不了这一点。
- **独立复核没做。** 说明写好了（`.../bench_via_20261123/review/BRIEF.md`），数据只在 vastgpu 上。

### ScaleBFM 的 A、B、C（`scalebfm/table.out`）

```
scalebfm: rows A 120, B 120, C 120
task                   arm   n  paths  official  45 s   pregrasp cm   entry cm   falls  knocked  no grasp
XMovePickTeleop         A   30      0        23    23          3.08       2.86       0       30         0
XMovePickTeleop         B   30      7        25    25          3.08       2.84       0       30         0
XMovePickTeleop         C   30      7        29    29          3.08       2.75       0       30         0
XMoveBendPickTeleop     A   30      0        25    25          3.32       3.30       0       30         0
XMoveBendPickTeleop     B   30      0        25    25          3.32       3.30       0       30         0
XMoveBendPickTeleop     C   30      0        25    25          3.32       3.30       0       30         0
BendPickMP              A   30      0         0    30          1.40       2.29       0       30         0
BendPickMP              B   30      0         0    30          1.40       2.29       0       30         0
BendPickMP              C   30      0         0    30          1.40       2.29       0       30         0
TabletopGraspMP         A   30      0         2     2         26.74      27.72       0       13         1
TabletopGraspMP         B   30     28         7    12          3.45       2.37       0       28         1
TabletopGraspMP         C   30     28         0    19          9.92       2.93       0       27         1

episodes with a planned path: 35; of them B and C both reached the planned start (goal within 1 cm): 35; by task: XMovePickTeleop 7, XMoveBendPickTeleop 0, BendPickMP 0, TabletopGraspMP 28
  pregrasp error, 35 pairs: B 3.63 cm, C 10.65 cm; paired difference C - B median +6.85 cm (+6.33 to +7.65); C worse in 34, better in 1
  entry error, 35 pairs: B 2.46 cm, C 2.66 cm; paired difference C - B median +0.50 cm (-0.15 to +0.84); C worse in 23, better in 12
  right arm in penetration with a table or another object during the pregrasp reach: B 0.771, C 0.314; difference -0.457 (-0.629 to -0.286); only B 16, only C 0
  deepest such penetration, median over the episodes that have one: B 0.09 cm, C 0.03 cm
  success_45 on these episodes: B 14, C 25; only B 3, only C 14
  success_official on these episodes: B 9, C 6; only B 7, only C 4
  the correction: goal moved median 8.00 cm, at the 8 cm limit in 0.54 of the episodes; largest via move median 8.00 cm, at the limit in 0.69; wanted wrist top median 1.018 m, over 1.10 m in 0.00
  the model's expected end offset of the original command against B's: error median 11.15 cm (expected size 6.91 cm, B's 3.63 cm)

45 s successes over 120 episodes: B 92, C 103; C - B +11 (+3 to +19); only B 3, only C 14; A 80
falls: A 0, B 0, C 0
joints (of 29) outside the model's training range at the pregrasp command, routed episodes: XMovePickTeleop median 2 (none outside in 2 of 7); TabletopGraspMP median 11 (none outside in 0 of 28)
the same over every episode of arm A: XMovePickTeleop median 0 (none outside in 24 of 30); XMoveBendPickTeleop median 0 (none outside in 22 of 30); BendPickMP median 5 (none outside in 0 of 30); TabletopGraspMP median 11 (none outside in 0 of 29)
```

规划（`plan_rows.jsonl`、`plan_rows_act02.jsonl`）：TabletopGrasp 3 cm 规划出 27 局，2 cm 重试多出 1 局，1 局没有抓取所以没有起点，1 局够不到；XMovePick 7 局；XMoveBendPick 和 BendPick 各 30 局都是够不到目标（腰锁着）。

### ScaleBFM 的对照 D、E（`scalebfm/controls.out`）

```
scalebfm: arms B C D E; episodes where every arm ran the planned path: 35 (XMovePickTeleop 7, XMoveBendPickTeleop 0, BendPickMP 0, TabletopGraspMP 28)
arm  45 s  official  pregrasp cm  entry cm  arm contact  goal moved cm (median)
 B     14         9         3.63      2.46        0.771  0.00
 C     25         6        10.65      2.66        0.314  8.00
 D     26         7         9.93      2.72        0.200  6.32
 E     11         8         3.48      2.56        0.800  1.42
C against D: 45 s successes 25 and 26, difference -1 (-6 to +4), only C 3, only D 4; arm contact 0.314 and 0.200, difference +0.114 (+0.000 to +0.257); pregrasp error paired difference median +0.44 cm (-0.49 to +0.97)
C against E: 45 s successes 25 and 11, difference +14 (+7 to +20), only C 15, only E 1; arm contact 0.314 and 0.800, difference -0.486 (-0.657 to -0.314); pregrasp error paired difference median +6.96 cm (+6.24 to +7.64)
D against B: 45 s successes 26 and 14, difference +12 (+5 to +19), only D 14, only B 2; arm contact 0.200 and 0.771, difference -0.571 (-0.743 to -0.400); pregrasp error paired difference median +6.50 cm (+6.29 to +6.92)
E against B: 45 s successes 11 and 14, difference -3 (-6 to +0), only E 0, only B 3; arm contact 0.800 and 0.771, difference +0.029 (+0.000 to +0.086); pregrasp error paired difference median +0.09 cm (-0.18 to +0.12)
C against B: 45 s successes 25 and 14, difference +11 (+4 to +18), only C 14, only B 3; arm contact 0.314 and 0.771, difference -0.457 (-0.629 to -0.286); pregrasp error paired difference median +6.85 cm (+6.33 to +7.65)
45 s successes over all 120 episodes: B 92, C 103, D 104, E 89; falls: B 0, C 0, D 0, E 0
by task, 45 s: XMovePickTeleop B 25 C 29 D 30 E 24; XMoveBendPickTeleop B 25 C 25 D 25 E 25; BendPickMP B 30 C 30 D 30 E 30; TabletopGraspMP B 12 C 19 D 19 E 10
```

### SONIC 的 A、B、C（`sonic/table.out`，补齐之后）

```
sonic: rows A 120, B 120, C 120
task                   arm   n  paths  official  45 s   pregrasp cm   entry cm   falls  knocked  no grasp
XMovePickTeleop         A   30      0         7     7          4.01       5.93       0       28         0
XMovePickTeleop         B   30      9         8     8          3.93       5.93       0       28         0
XMovePickTeleop         C   30      9         8     8          3.82       5.81       0       28         0
XMoveBendPickTeleop     A   30      0         3     3          6.59       7.23       0       20         0
XMoveBendPickTeleop     B   30      5         3     3          6.59       7.23       0       21         0
XMoveBendPickTeleop     C   30      5         2     2          7.29       7.23       0       21         0
BendPickMP              A   30      0         0     6          4.49       6.73       0       19         0
BendPickMP              B   30      0         0     6          4.49       6.73       0       19         0
BendPickMP              C   30      0         0     6          4.49       6.73       0       19         0
TabletopGraspMP         A   30      0         5     5         14.28       8.49       0       14         1
TabletopGraspMP         B   30     28         5     8          2.92       3.51       0       24         1
TabletopGraspMP         C   30     28         6     9          4.08       3.69       0       26         1

episodes with a planned path: 42; of them B and C both reached the planned start (goal within 1 cm): 42; by task: XMovePickTeleop 9, XMoveBendPickTeleop 5, BendPickMP 0, TabletopGraspMP 28
  pregrasp error, 42 pairs: B 2.48 cm, C 3.35 cm; paired difference C - B median +0.35 cm (-0.09 to +0.86); C worse in 25, better in 17
  entry error, 42 pairs: B 3.10 cm, C 3.41 cm; paired difference C - B median -0.02 cm (-0.31 to +0.32); C worse in 21, better in 21
  right arm in penetration with a table or another object during the pregrasp reach: B 0.429, C 0.452; difference +0.024 (-0.048 to +0.119); only B 1, only C 2
  deepest such penetration, median over the episodes that have one: B 0.10 cm, C 0.07 cm
  success_45 on these episodes: B 12, C 12; only B 4, only C 4
  success_official on these episodes: B 9, C 9; only B 3, only C 3
  the correction: goal moved median 2.82 cm, at the 8 cm limit in 0.00 of the episodes; largest via move median 3.50 cm, at the limit in 0.00; wanted wrist top median 1.015 m, over 1.10 m in 0.00
  the model's expected end offset of the original command against B's: error median 3.39 cm (expected size 2.79 cm, B's 2.48 cm)

45 s successes over 120 episodes: B 25, C 25; C - B +0 (-5 to +6); only B 4, only C 4; A 21
falls: A 0, B 0, C 0
joints (of 29) outside the model's training range at the pregrasp command, routed episodes: XMovePickTeleop median 4 (none outside in 0 of 9); XMoveBendPickTeleop median 4 (none outside in 1 of 5); TabletopGraspMP median 9 (none outside in 0 of 28)
the same over every episode of arm A: XMovePickTeleop median 3 (none outside in 0 of 30); XMoveBendPickTeleop median 3 (none outside in 1 of 30); BendPickMP median 2 (none outside in 1 of 30); TabletopGraspMP median 9 (none outside in 0 of 29)
```

### 对照判据

| 判据 | ScaleBFM | SONIC |
|---|---|---|
| 到得准：C 减 B 的预抓取偏差为负，区间不含 0 | +6.85 cm（+6.33 到 +7.65），**不成立，方向反了** | +0.35 cm（−0.09 到 +0.86），不成立 |
| 碰得少：伸手途中右臂真接触 C 低于 B | 77.1 % 和 31.4 %，差 −0.457（−0.629 到 −0.286），成立 | 42.9 % 和 45.2 %，差 +0.024（−0.048 到 +0.119），分不出 |
| 任务成功：120 局 45 秒成功 C 减 B | 92 和 103，+11（+3 到 +19），成立 | 25 和 25，0（−5 到 +6），分不出 |
| 有效性：到达规划的起点 | 35 / 35 | 42 / 42 |
| 有效性：C 的摔倒不比 B 多 3 局以上 | 0 和 0 | 0 和 0 |
| 对照。模型的功劳：C 的成功局数多于 D，区间不含 0 | 25 和 26，−1（−6 到 +4），**不成立** | 没跑完 |
| 对照。D 对 B | 26 和 14，+12（+5 到 +19） | 没跑完 |
| 对照。E 对 B | 11 和 14，−3（−6 到 0） | 没跑 |

### 对照预测

| 量 | 预测 | 实际 | 在范围里吗 |
|---|---|---|---|
| ScaleBFM 有路线的局，TabletopGrasp / 另外三个任务 | 26–29 / 3–12 | 28 / 7 | 在 |
| SONIC 有路线的局，TabletopGrasp / 另外三个任务 | 20–29 / 0–15 | 28 / 14 | 在 |
| ScaleBFM 预抓取偏差，B | 2.5–4.5 cm | 3.63 | 在 |
| ScaleBFM 预抓取偏差，C | 0.6–1.6 cm | 10.65 | **差得远** |
| ScaleBFM 进入偏差，B 和 C | 都在 2–4 cm，C 比 B 小不超过 1 cm | 2.46 和 2.66 | 范围在，方向不对 |
| ScaleBFM 右臂真接触，B | 80–100 % | 77 % | 略低 |
| ScaleBFM 右臂真接触，C | 45–90 % | 31 % | 低于范围 |
| ScaleBFM TabletopGrasp 45 秒成功，A / B / C | 1–4 / 9–15 / 10–18 | 2 / 12 / 19 | C 多 1 局 |
| ScaleBFM 另外三个任务三遍相差不超过 2 局 | | XMovePick 23 / 25 / 29 | 不在 |
| SONIC 直线伸手 45 秒成功，四个任务 | 3–15、2–15、5–25、0–3 | 7、3、6、5 | TabletopGrasp 高于范围 |
| SONIC 预抓取偏差，B / C | 4–7 / 1.2–3 cm | 2.48 / 3.35 | 都不在 |
| SONIC 右臂真接触，C 不比 B 低 | | 43 % 和 45 % | 对 |
| SONIC TabletopGrasp 45 秒成功，B / C | 0–6 / 0–8 | 8 / 9 | 都高于范围 |
| 摔倒每遍不超过 3 局 | | 全是 0 | 对 |
| 对照：D 和 C 的成功局数相差不超过 3 局 | | 26 和 25 | 对 |
| 对照：D 和 C 的接触比例相差不超过 0.15 | | 0.200 和 0.314 | 对 |
| 对照：D 和 C 的预抓取偏差相差不超过 2.5 cm | | 9.93 和 10.65 | 对 |
| 对照：E 的成功局数和 B 相差不超过 3 局 | | 11 和 14 | 对 |

我写的三句总预测：“C 的预抓取偏差明显小于 B”两个控制器上都错了；“成功局数分不出”在 ScaleBFM 上错了，在 SONIC 上对；“ScaleBFM 上真接触会不会减少没有把握”，实际减少了，但对照说明原因不是模型。

### 读法

- **多路点模型在基准里预测不准。** 它对原命令终点偏差的预期和 B 实际的偏差相比，ScaleBFM 上差 11.15 cm（预期 6.91 cm，实际只偏 3.63 cm），SONIC 上差 3.39 cm（预期 2.79，实际 2.48）。预期和实际的大小相当，方向对不上。
- **原因是起始状态超出了训练范围。** 发预抓取命令时，29 个关节里超出训练集 0.5–99.5 % 范围的个数：TabletopGrasp 上 ScaleBFM 中位数 11 个、SONIC 9 个，没有一局全在范围里。机器人侧跨一步后贴着桌子站，躯干后仰，右臂被桌沿顶着。模型只见过空地上的站姿。接法查过，和 `gen_reach.py` 里的调用一致。
- **ScaleBFM 上的增益来自后撤，不是来自模型。** C 把目标往回拉了 6 到 8 cm（54 % 的局顶到上限），手停得更靠后，慢速进入那一段变长。不用模型的固定后撤 D 抓起 26 局，C 是 25 局。XMovePick 有路线的 7 局里，B 有 5 局在伸手途中把目标碰走，C 只有 1 局。
- **代价是时间。** TabletopGrasp 成功时刻的中位数从 15.9 秒到 16.9 秒，官方 16 秒内抓起从 7 局降到 0 局。
- **接触大多很浅。** 有接触的局里最深穿透的中位数是 0.09 cm 和 0.03 cm，是手指擦过桌面这一类。
- **这个基准分不出规划方法。** 11-20 的归因实验已经得到过同一个结论（B46）：固定模板“后撤 8 cm、抬到预抓取高度、再伸过去”和 cuRobo 路线一样不卡。30 局是同一张桌子，障碍物永远在目标下方。
- **规划器锁着腰**，XMoveBendPick 和 BendPick 一条路线都没有，这两个任务上三遍是同一条命令。

### 结论

1. 在 SIMPLE 的抓取任务上，多路点自身模型没有被证明有用。空地上的结论（第四批测试）没有迁移过来。
2. 让机器人绕过桌沿的是 cuRobo 的路线；再加一条固定后撤，TabletopGrasp 的 45 秒成功是 2、12、19（直线、路线、路线加后撤）。
3. 用户 2026-11-24 定的下一步见 TODO 38.10：贡献集中在自身模型上，把控制接口里的 IK 换成 cuRobo 的关节轨迹，SIMPLE 的成功局数作主指标，交给 4090 执行。
