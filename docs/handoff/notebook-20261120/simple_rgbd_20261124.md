# simple_rgbd_20261124：新的默认组合（SIMPLE、ScaleBFM、单目 RGB-D）在 SIMPLE 四个抓取任务上的主指标，五种伸手做法，两个控制器

- 日期：2026-11-24（设计、预测、判据；任何正式运行之前写下，单独提交）
- 机器 / GPU：4090，1–3 号卡（0 号卡常有别人的任务）
- 代码版本：`selfwm` @ `93a67e86` 加未提交的一个文件（跑的时候 `bench_via.sh` 把用到的脚本抄进输出目录的 `asrun*/`）：
  - `scripts/experiments/simple_eval_20261112/look_frame.py`：新的，给 RGB-D 那一眼补整帧的点云（md5 `70ac03a6fc885cc5db2d4c10d4aa6a29`）
  - 不改：`bench_via.sh`（md5 `91b29e9524520c47124ea2eaab9fefe1`）、`simple_bench.py`（md5 `abc246fd48fdf16d6f734167b3cacb46`）、`bench_jref_table.py`
- 配置：SIMPLE 官方评测局，四个抓取任务各 30 局，只跑感知栏；开关 `--bench-set sensor=rgbd`，签名 `simple/perception-rgbd/pose-odometry/root-odometry/<控制器>`；ScaleBFM 和 SONIC（`REACH_SONIC_NETS=gpu`）都不改参数；不用仿真真值，不用伺服和查表。
- 关联：
  - [selfmodel_jref_20261124](selfmodel_jref_20261124.md)：同样五种做法在双目那一眼上的结果，输出在 4090 的 `outputs/simple_eval_20261112/bench_jref_20261124/`，这次逐局和它配对；
  - TODO 38.11：用户 2026-11-24 的要求，"现在这个默认的组合设置为 simple+scalebfm+单目 rgbd"。

## 要回答的问题

默认组合从双目换成单目 RGB-D 以后，主指标是多少；在双目上看到的那几条读法换了这一眼还在不在。

单目 RGB-D 这一眼是复刻场景里 MuJoCo 的头部相机：彩色图加渲染出的深度，深度是理想值，画面是白底、没有房间。目标的点和抓取来自 `cloud_blind/`（11-19 用 `blind.py look` 做的，桌面抓取用只看深度的旧规则找目标）。双目那一眼是 SIMPLE 的 Isaac 画面，双目匹配加 LingBot-Depth。

两眼不同的地方，都是看一眼的结果，不是控制器：

- 目标的点和抓取不同。11-20 的统计：BendPick 上 MuJoCo 画面多是从侧面低位抓（30 局里 8 局从上往下），Isaac 画面 28 局从上往下；另外三个任务的抓取移动中位数 2 到 3 cm。
- 规划器的障碍物不同。双目来自 `look_dump_eyes/`，RGB-D 来自这次补出的 `look_dump_blind/`。复刻场景只有桌子和物体，没有房间，深度也没有误差。
- 桌子前沿两眼相同，都用 `blind_f1_<task>.json`。

## 做法

1. **补 RGB-D 那一眼的整帧点云**（`look_frame.py`）。`blind.py look` 只在带 `--eyes` 时存整帧，所以 `cloud_blind/` 没有规划器要的那份。脚本照 `blind.py look` 的方式让机器人站 1 秒，取同一帧：彩色、每个像素在机器人坐标里的点、哪些像素有深度、骨盆的估计、相机位姿。这一步不用 SAM 3，目标的点仍用 `cloud_blind/`。每局打印 `cloud_blind/` 的点离这一帧的点多远，用来核对两者在同一个坐标里。试跑一局（TabletopGrasp 第 0 局）：中位数 0.405 cm，95 % 分位 0.473 cm。
2. **五种做法，两个控制器**，和 selfmodel_jref_20261124 的 38.10.2 相同：

   | 记号 | 做法 | 用不用模型 |
   |---|---|---|
   | A | 直线伸手 | 不用 |
   | M | 固定模板：后撤 8 cm、抬到预抓取高度、再伸过去 | 不用 |
   | B | cuRobo 的手路点经我们的全身 IK | 不用 |
   | D | B 加固定后撤：每个路点和目标向后 6 cm、向上 2 cm | 不用 |
   | J | cuRobo 的右臂关节轨迹直接当参考 | 不用 |

   每个控制器各自做 A 和规划，然后在同一批路线上跑 M、B、D、J。没有路线的局退回直线伸手。
3. **怎么跑**：`bench_via.sh`，环境变量 `BENCH_SET=sensor=rgbd`，其余和双目那次逐项相同。ScaleBFM：A、规划、B 在 1 号卡；D、M、J 同时跑，都在 1 号卡。SONIC：四个任务的进程分在 2、3 号卡，规划在 2 号卡。
4. **计分**：`bench_jref_table.py <输出目录> outputs/scene/simple_g1_sonic`，和双目那次相同。
5. 没有开发局：设置全部取已有的默认值，跑完不改。

## 度量

1. **主指标**：SIMPLE 的成功（目标被抬起），每个任务 30 局，45 秒内和官方时限内都报，重点是 TabletopGrasp。
2. 流程核对：规划那一步有多少局得到障碍物文件、多少局规划出路线；每一遍有没有进程失败；`look_frame.py` 打印的两眼坐标差。
3. 和双目那次逐局配对：同一个控制器、同一种做法，两眼的 45 秒成功数之差，以及只有一边成功的局数。

## 预测（启动前写，首次启动后锁定）

- **流程**：`look_frame.py` 120 局全部写出，坐标差的中位数每局都不超过 1 cm（试跑是 0.4 cm，像是固定的小偏移）。规划那一步 120 局里至少 115 局有障碍物文件；有路线的局数 25 到 60（双目是 ScaleBFM 35、SONIC 41；复刻场景没有房间也没有深度误差，假障碍少，路线可能更多，但 TabletopGrasp 的桌沿仍然挡路）。所有遍 0 个进程失败。
- **ScaleBFM 直线伸手 A，45 秒内**：共 55 到 95 局（双目 80）。
  - XMovePick 20 到 30、XMoveBendPick 20 到 30（双目 23、25，两眼抓取差得不多）；
  - BendPick 5 到 30（双目 30；MuJoCo 画面下多是侧面低位抓，11-19 旧代码在这一眼上只有 10）；
  - TabletopGrasp 0 到 10（双目 2；直线伸手撞桌沿，和看哪一眼关系不大）。
  - 官方时限内：BendPick 0（8 秒不够，两眼都是），总数 30 到 70。
- **SONIC 直线伸手 A，45 秒内**：共 15 到 40（双目 25）。
- **有路线的做法，ScaleBFM**：在所有带路线的做法都走了自己路线的局上，D 不少于 B，M 和 J 都不少于 A。理由：双目上这几条差距主要在 TabletopGrasp 的桌沿，桌沿两眼都看得见。
- **SONIC**：五种做法的 45 秒总数两两相差都不超过 10（双目 23 到 31）。

## 判据（什么结果会让我改读法，下一步做什么）

- **流程没通过**（某一遍有进程失败，规划那一步少于 115 局有障碍物文件，或者两眼坐标差的中位数有一局超过 1 cm）：不能说新的默认组合能跑规划路径的做法。文档写明只能跑直线伸手，先查原因，不报这一眼规划路径做法的数。
- **流程通过**：新的默认组合的主指标就是这次的数，以后每次汇报都带它，双目的数也照报。
- **ScaleBFM 在带路线的局上 B 比 D 多 5 局以上**：固定后撤的好处只在双目那一眼上成立，38.10.2 的读法要注明只对双目。
- **两眼的 A 总数相差 15 局以上**：看哪一眼比怎么伸手影响更大，报告里两眼的数要并排放，不能只报一个。
- 其余情况：双目上的读法在 RGB-D 上也成立，报数即可。

## 预测修订

## 结果（贴原始输出，不转述）

4090，2026-11-24 12:16–12:51 UTC，每格跑一次。输出在 4090 的 `outputs/simple_eval_20261112/bench_rgbd_20261124/{scalebfm,sonic}/`（逐局 `rows/`，日志 `logs/`，跑的脚本副本 `asrun*/`，计分 `table.txt`），配对在同目录的 `pair_looks.txt`。

**流程**：每局的整帧点云（`look_frame.py`，`outputs/simple_eval_20261112/look_logs/look_frame_*.log`，逐局一行 JSON，下面是汇总这些行的输出）：

```
DONE 120 files, 0 failed processes
look_frame_BendPickMP.log episodes 30 with cloud 30 median cm min/max 0.508 0.562 p95 max 0.697 over 1 cm 0
look_frame_TabletopGraspMP.log episodes 30 with cloud 30 median cm min/max 0.403 0.409 p95 max 0.517 over 1 cm 0
look_frame_XMoveBendPickTeleop.log episodes 30 with cloud 30 median cm min/max 0.477 0.488 p95 max 0.557 over 1 cm 0
look_frame_XMovePickTeleop.log episodes 30 with cloud 30 median cm min/max 0.48 0.491 p95 max 0.569 over 1 cm 0
```

运行记录（`phase.log`、两个 `pipeline.log`）：

```
start 2026-11-24T12:16:50Z bc426d5 + uncommitted 70ac03a6 abc246fd 91b29e95 8c4cc0d2 
scalebfm phase1 done 2026-11-24T12:39:27Z
sonic phase1 done 2026-11-24T12:43:13Z
scalebfm phase2 done 2026-11-24T12:45:13Z
sonic phase2 done 2026-11-24T12:51:13Z
all done 2026-11-24T12:51:13Z
== outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm/pipeline.log
[12:16:51] start: simple/perception-rgbd/pose-odometry/root-odometry/scalebfm, episodes 0-29, model none, commit bc426d5 + uncommitted (copies in asrun/)
[12:20:54] arm A (GPUs 1): 120 rows, 0 failed processes
[12:20:54] DONE A
[12:22:05] step 1: 120 rows, 120 obstacle files
[12:22:56] step 2a: 120 prep files
[12:31:37] step 2b: 120 rows, planned 28
[12:34:55] step 2b retry at 2 cm: 92 rows, planned 0
[12:34:55] step 2c: 28 via files
[12:34:55] DONE plan
[12:39:27] arm B (GPUs 1): 120 rows, 0 failed processes
[12:39:27] DONE B
[12:39:28] template: 28 via files
[12:44:54] arm J (GPUs 1): 120 rows, 0 failed processes
[12:44:54] DONE J
[12:45:07] arm M (GPUs 1): 120 rows, 0 failed processes
[12:45:07] DONE M
[12:45:13] arm D (GPUs 1): 120 rows, 0 failed processes
[12:45:13] DONE D
== outputs/simple_eval_20261112/bench_rgbd_20261124/sonic/pipeline.log
[12:16:51] start: simple/perception-rgbd/pose-odometry/root-odometry/sonic, episodes 0-29, model none, commit bc426d5 + uncommitted (copies in asrun/)
[12:22:37] arm A (GPUs 2 3): 120 rows, 0 failed processes
[12:22:38] DONE A
[12:23:47] step 1: 120 rows, 120 obstacle files
[12:24:37] step 2a: 120 prep files
[12:33:13] step 2b: 120 rows, planned 29
[12:36:30] step 2b retry at 2 cm: 91 rows, planned 0
[12:36:30] step 2c: 29 via files
[12:36:30] DONE plan
[12:43:13] arm B (GPUs 2 3): 120 rows, 0 failed processes
[12:43:13] DONE B
[12:43:14] template: 29 via files
[12:50:08] arm J (GPUs 2 3): 120 rows, 0 failed processes
[12:50:08] DONE J
[12:50:58] arm M (GPUs 3 2): 120 rows, 0 failed processes
[12:50:58] DONE M
[12:51:13] arm D (GPUs 2 3): 120 rows, 0 failed processes
[12:51:13] DONE D
```

**计分**：`SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE PYTHONPATH=src $PY -I scripts/experiments/simple_eval_20261112/bench_jref_table.py <输出目录>/<控制器> outputs/scene/simple_g1_sonic`：

```
######## scalebfm
scalebfm: arms A M B D J; rows A 120, M 120, B 120, D 120, J 120

1. successes, 30 episodes a task: within 45 s / within the official limit (falls)
  task                             A           M           B           D           J
  XMovePickTeleop            28 / 28     28 / 28     28 / 28     28 / 28     28 / 28
  XMoveBendPickTeleop        27 / 27     27 / 27     27 / 27     27 / 27     27 / 27
  BendPickMP                  10 / 0      10 / 0      10 / 0      10 / 0      10 / 0
  TabletopGraspMP            18 / 18     29 / 29     29 / 16      29 / 5     30 / 26
  all 120                    83 / 73     94 / 84     94 / 71     94 / 60     95 / 81

2. episodes where every path arm (M B D J) ran its path: 28 (XMovePickTeleop 0, XMoveBendPickTeleop 0, BendPickMP 0, TabletopGraspMP 28)
  arm  45 s  official  right arm: any  >1 mm  >2 mm   other links   off plan cm: median  p75   max   <=1.5 cm   pregrasp cm  entry cm  success time s
   A    16        16           1.000  1.000  0.500         0.964                25.25  26.78  31.49       0.00          5.48      4.90           12.92
   M    27        27           1.000  0.036  0.000         1.000                26.98  28.46  30.65       0.00          2.37      1.79           15.22
   B    27        14           1.000  0.821  0.071         0.929                10.97  12.44  15.20       0.00          2.36      1.90           15.94
   D    27         3           0.286  0.000  0.000         0.929                12.52  14.06  17.72       0.00          9.45      2.44           16.64
   J    28        24           0.893  0.179  0.000         0.929                 7.29  8.49  11.93       0.00          3.81      2.33           15.77

3. pairs on these episodes, X - Y (95 % interval)
  J - D: 45 s successes 28 and 27, +1 (+0 to +3), only J 1, only D 0; right arm any penetration +0.607 (+0.429 to +0.786), over 1 mm +0.179 (+0.036 to +0.321); off plan paired median -5.30 cm (-6.14 to -4.97)
  J - B: 45 s successes 28 and 27, +1 (+0 to +3), only J 1, only B 0; right arm any penetration -0.107 (-0.250 to +0.000), over 1 mm -0.643 (-0.821 to -0.464); off plan paired median -3.11 cm (-4.54 to -1.62)
  J - M: 45 s successes 28 and 27, +1 (+0 to +3), only J 1, only M 0; right arm any penetration -0.107 (-0.250 to +0.000), over 1 mm +0.143 (+0.000 to +0.321); off plan paired median -19.42 cm (-19.96 to -19.09)
  D - B: 45 s successes 27 and 27, +0 (-3 to +3), only D 1, only B 1; right arm any penetration -0.714 (-0.893 to -0.536), over 1 mm -0.821 (-0.929 to -0.679); off plan paired median +2.01 cm (+1.39 to +3.16)
  M - B: 45 s successes 27 and 27, +0 (+0 to +0), only M 0, only B 0; right arm any penetration +0.000 (+0.000 to +0.000), over 1 mm -0.786 (-0.929 to -0.607); off plan paired median +16.38 cm (+15.77 to +17.07)

4. J: the reference's arm against the plan's start, largest joint: median 1.72 deg, max 4.53; path median 80.0 cm, duration median 5.33 s
######## sonic
sonic: arms A M B D J; rows A 120, M 120, B 120, D 120, J 120

1. successes, 30 episodes a task: within 45 s / within the official limit (falls)
  task                             A           M           B           D           J
  XMovePickTeleop              5 / 5       5 / 5       5 / 5       5 / 5       5 / 5
  XMoveBendPickTeleop          3 / 3       3 / 3       3 / 3       3 / 3       3 / 3
  BendPickMP                  11 / 0      11 / 0      11 / 0      11 / 0      11 / 0
  TabletopGraspMP            10 / 10       5 / 5     17 / 10      20 / 9     15 / 12
  all 120                    29 / 18     24 / 13     36 / 18     39 / 17     34 / 20

2. episodes where every path arm (M B D J) ran its path: 29 (XMovePickTeleop 1, XMoveBendPickTeleop 0, BendPickMP 0, TabletopGraspMP 28)
  arm  45 s  official  right arm: any  >1 mm  >2 mm   other links   off plan cm: median  p75   max   <=1.5 cm   pregrasp cm  entry cm  success time s
   A     9         9           0.966  0.966  0.862         0.862                28.62  29.90  36.34       0.00          6.12      5.65           12.30
   M     4         4           0.862  0.793  0.379         0.931                29.91  35.30  54.94       0.00          9.27      7.43           14.90
   B    16         9           0.793  0.483  0.172         0.862                13.78  16.11  37.16       0.00          2.58      3.91           15.87
   D    19         8           0.828  0.517  0.276         0.828                16.45  17.47  24.25       0.00          8.42      4.60           16.34
   J    14        11           0.862  0.552  0.310         0.759                15.81  18.61  31.23       0.00          6.33      3.64           14.86

3. pairs on these episodes, X - Y (95 % interval)
  J - D: 45 s successes 14 and 19, -5 (-10 to +0), only J 1, only D 6; right arm any penetration +0.034 (-0.069 to +0.172), over 1 mm +0.034 (-0.172 to +0.241); off plan paired median -0.84 cm (-3.07 to +0.48)
  J - B: 45 s successes 14 and 16, -2 (-8 to +3), only J 3, only B 5; right arm any penetration +0.069 (+0.000 to +0.172), over 1 mm +0.069 (-0.069 to +0.207); off plan paired median -0.02 cm (-1.47 to +2.35)
  J - M: 45 s successes 14 and 4, +10 (+4 to +16), only J 11, only M 1; right arm any penetration +0.000 (-0.103 to +0.103), over 1 mm -0.241 (-0.448 to -0.034); off plan paired median -15.79 cm (-18.22 to -12.60)
  D - B: 45 s successes 19 and 16, +3 (-3 to +9), only D 6, only B 3; right arm any penetration +0.034 (+0.000 to +0.103), over 1 mm +0.034 (-0.138 to +0.207); off plan paired median +2.41 cm (+1.34 to +3.44)
  M - B: 45 s successes 4 and 16, -12 (-18 to -6), only M 1, only B 13; right arm any penetration +0.069 (-0.069 to +0.207), over 1 mm +0.310 (+0.103 to +0.517); off plan paired median +14.07 cm (+11.65 to +20.03)

4. J: the reference's arm against the plan's start, largest joint: median 2.40 deg, max 5.13; path median 86.5 cm, duration median 5.77 s

```

**和双目那次逐局配对**：`python3 -I pair_looks.py bench_rgbd_20261124 bench_jref_20261124`，每格是 RGB-D 对双目 45 秒内的成功局数，括号里是只有 RGB-D 成功 / 只有双目成功的局数：

```
scalebfm: 45 s successes, RGB-D vs stereo, same episodes (only RGB-D / only stereo)
  arm          XMovePickTeleo        XMoveBendPickT            BendPickMP        TabletopGraspM               all 120
   A          28 vs 23 (7/2)        27 vs 25 (5/3)       10 vs 30 (0/20)        18 vs 2 (16/0)      83 vs 80 (28/25)   paired eps 120
   M          28 vs 30 (0/2)        27 vs 25 (5/3)       10 vs 30 (0/20)       29 vs 12 (17/0)      94 vs 97 (22/25)   paired eps 120
   B          28 vs 25 (5/2)        27 vs 25 (5/3)       10 vs 30 (0/20)       29 vs 12 (17/0)      94 vs 92 (27/25)   paired eps 120
   D          28 vs 30 (0/2)        27 vs 25 (5/3)       10 vs 30 (0/20)       29 vs 19 (11/1)     94 vs 104 (16/26)   paired eps 120
   J          28 vs 24 (6/2)        27 vs 25 (5/3)       10 vs 30 (0/20)       30 vs 19 (11/0)      95 vs 98 (22/25)   paired eps 120

sonic: 45 s successes, RGB-D vs stereo, same episodes (only RGB-D / only stereo)
  arm          XMovePickTeleo        XMoveBendPickT            BendPickMP        TabletopGraspM               all 120
   A            5 vs 6 (5/6)          3 vs 6 (3/6)         11 vs 6 (9/4)         10 vs 7 (6/3)      29 vs 25 (23/19)   paired eps 120
   M            5 vs 9 (5/9)          3 vs 6 (3/6)         11 vs 6 (9/4)          5 vs 2 (5/2)      24 vs 23 (22/21)   paired eps 120
   B            5 vs 8 (5/8)          3 vs 6 (3/6)         11 vs 6 (9/4)       17 vs 11 (10/4)      36 vs 31 (27/22)   paired eps 120
   D            5 vs 8 (5/8)          3 vs 6 (3/6)         11 vs 6 (9/4)       20 vs 10 (13/3)      39 vs 30 (30/21)   paired eps 120
   J            5 vs 9 (5/9)          3 vs 4 (3/4)         11 vs 6 (9/4)        15 vs 6 (12/3)      34 vs 25 (29/20)   paired eps 120
```

## 意外

1. **ScaleBFM 直线伸手在 TabletopGrasp 上 18 局，预测 0 到 10。** 双目那次是 2 局。逐局配对是 16 局只有 RGB-D 成功，0 局只有双目成功。
   - 手臂照样碰桌沿：28 局有路线的局里，直线伸手的右臂穿进桌子超过 1 mm 的比例是 1.000，但 28 局里还是抓起来了 16 局。
   - 两眼在这个任务上的差别是目标的点和抓取：RGB-D 用只看深度的旧规则，双目用 SAM 3 找 "object"。抓取变了，预抓取点也就变了。这次没有拆开看是哪一样起的作用。
2. **官方时限内 ScaleBFM 直线伸手总数 73，预测 30 到 70。** 多出来的就是 TabletopGrasp 的 18 局，它们全都在 16 秒内。
3. **SONIC 五种做法的 45 秒总数从 24（M）到 39（D），相差 15，预测不超过 10。**
   - 差别全在 TabletopGrasp：M 5，D 20；另外三个任务五种做法相同。
   - 双目那次 SONIC 的 M 也最低（23），只是没这么明显。
4. **按任务看，两眼差得很大，只是总数抵消了。**
   - ScaleBFM 直线伸手：BendPick 10 对 30，20 局只有双目成功；TabletopGrasp 18 对 2。总数 83 对 80。
   - 判据写的是总数相差 15 局以上，这条没有触发。但按任务的差远超 15。
5. **SONIC 在两个走过去抓的任务上很低。** XMovePick 5、XMoveBendPick 3，双目是 6、6，和看哪一眼无关。
6. **固定后撤（D）在 ScaleBFM 上 45 秒内 94 局，官方时限内只有 60 局。** TabletopGrasp 29 局里只有 5 局在 16 秒内，后撤的路线更长，慢。
7. **过程上的一处：** 补整帧点云那一步（`look_frame.py`）在 12:15:59 跑完，登记的提交是 12:16:09，正式运行 12:16:50 开始。这一步是运行的输入，它的预测写的时候只看过试跑的那一局（0.405 cm）。
8. **TabletopGrasp 0 级和 1 级的 10 个布局，目标位置和机器人起点相同。** 但这次跑出来逐局的成败和成功时刻只在 1 到 8 局（共 10 局）上相同，各种做法不一。所以不是重复局，但相关，这个任务的 30 局不等于 30 个独立布局。

## 结论与下一步

- **新的默认组合（感知栏、ScaleBFM、单目 RGB-D）的主指标**：直线伸手 45 秒内 83 局、官方时限内 73 局，TabletopGrasp 18 / 18（结果第 1 张表，`A` 列）。
  - 不用模型的对手里，45 秒内最好的是 J：95 / 81，TabletopGrasp 30 / 26。
  - 官方时限内最好的是 M：94 / 84，TabletopGrasp 29 / 29。
  - SONIC：直线伸手 29 / 18；45 秒内最好的是 D，39 / 17；官方时限内最好的是 J，34 / 20。
- **双目上的读法在 RGB-D 上还成立的：**
  - ScaleBFM 在有路线的 28 局上，D 不少于 B（27 对 27），M、J 都不少于 A（27、28 对 16）；
  - J 的手离规划最近（中位数 7.29 cm，B 10.97，D 12.52）；
  - 固定后撤让穿桌超过 1 mm 的比例从 B 的 0.821 降到 0（第 2、3 张表）。
- **不成立或要改口的：**
  - "直线伸手在 TabletopGrasp 上几乎全失败"只对双目那一眼成立（第 1 张表，以及配对表 16 / 0）。
  - SONIC 上各种做法分得开：M 比 B 少 12 局，区间 -18 到 -6（第 3 张表）。
- **证据强度**：每格跑一次，每个任务 30 局。TabletopGrasp 的 30 局只有 20 个不同布局（意外 8）。有路线的局 ScaleBFM 28、SONIC 29，几乎全在 TabletopGrasp。
- **其他解释**：
  - 两眼的差别主要来自目标和抓取的来源（深度规则对 SAM 3），不是深度质量本身。
  - BendPick 上 RGB-D 的抓取多是侧面低位抓，直线伸手在 8 秒内到不了，在 45 秒内也只有 10 局，所以这个任务的数是抓取决定的，不是伸手的做法决定的。
- **四类陷阱**：
  - **benchmark 选择**：SIMPLE 的四个抓取任务全报，每个任务全部 30 局官方评测局，没有挑。
  - **数据泄漏**：这次五种做法都不用模型。但 `cloud_blind/` 的目标规则和 GraspGen-X 的设置是 11-12 到 11-19 在这些局上调出来的，这一眼本身见过测试局。
  - **指标误用**：45 秒内和官方时限内都报，和登记一致。
  - **事后挑选**：每格一次，五种做法、两个控制器全部报出。
- **对照判据**：
  - 流程通过：120 局整帧点云，坐标差中位数每局 0.40 到 0.56 cm；两个控制器都是 120 个障碍物文件，所有遍 0 个进程失败。所以新的默认组合的主指标就是这次的数，以后每次汇报都带它，双目的数照报。
  - B 比 D 多 5 局以上：没有触发，27 对 27。
  - 两眼 A 总数相差 15 局以上：没有触发（ScaleBFM 3、SONIC 4）。但按任务的差远超 15（意外 4），所以报告按任务把两眼并排放。这比判据要求的多报，不改读法。
- **下一步**：S 这一行（TODO 38.10.4），在这个组合上做：先在开发局上定半径和修法（[selfmodel_s_dev_20261124](selfmodel_s_dev_20261124.md)），再登记、跑测试局，和这次的 A、J 逐局配对。

## 独立复核

2026-11-24，独立的 agent。它拿到的只有做法、预测和判据、原始输出、结论，以及 4090 上的逐局记录和代码路径，没有拿到"意外"一节和推理过程。原文（英文）：

> The headline numbers reproduce exactly, but two of the three claims hold only in part. "D ≥ B" holds only at 45 s, and the stereo-only straight-reach failure is mostly a grasp failure, not a reach failure.
>
> **1. Recomputed numbers.** Every cell matches (ScaleBFM A 83/73, M 94/84, B 94/71, D 94/60, J 95/81; SONIC 29/18, 24/13, 36/18, 39/17, 34/20); success_45 / success_official agree with t_success and the limits in every row; every row's signature is perception-rgbd; no falls. Routed: ScaleBFM 28 (all TabletopGrasp), SONIC 29 (one XMovePick); routed counts match (A 16, M 27, B 27/14, D 27/3, J 28/24); penetration shares recomputed from hits match (B 0.821, D 0.000 over 1 mm); the paired RGB-D vs stereo table matches, aligned by level and index. Off-plan distances not recomputed (need numpy/mujoco). On unrouted episodes every path arm equals A (92/92, 91/91): the simulation is deterministic for identical commands.
>
> **2. Pitfalls.** Benchmark selection: all 30 episodes of the four pick tasks, hard levels kept for development. But every TabletopGrasp episode is the same chips can; levels 0 and 1 have identical layouts, and ScaleBFM A's success pattern over indices 0–9 (0001011111) repeats exactly in all three levels (18 = 6 × 3). The effective sample is about 10, so table 3's intervals are too narrow. Data leakage: the author's admission stands; in addition the stereo baseline's target rule was switched after counting hits on TabletopGrasp's 30 test episodes (blind.py help text: 12 vs 27). The RGB-D look is MuJoCo rendering the physics geometry itself with depth_noise none: close to ground-truth surfaces, should be labelled so. Could not confirm whether M's template and D's offset were set on these episodes. Metric misuse: both metrics in the tables, but claim (b) uses 45 s only; at the official limit the same 28 episodes give D 3 and B 14. Post-hoc selection: one run per cell, all reported; but "best arm" labels rest on gaps of one episode (J 95 vs 94; J − D interval 0 to +3).
>
> **3. Predictions and criteria.** Met: pipeline checks, route count, ScaleBFM A total 83, SONIC A 29, routed predictions at 45 s. Missed: ScaleBFM A TabletopGrasp 18 vs 0–10 (only via claim c); official total 73 vs 30–70 (not mentioned); SONIC spread 15 vs ≤10 (only as "the arms separate"). "B more than D by 5+" names no metric and the declared main metric reports both; at the official limit B − D = +11, so reading it only at 45 s favours the author. "A totals differ by 15+": correctly not triggered (3, 4), per-task reported anyway.
>
> **Claim (c) in the rows** (ScaleBFM A, TabletopGrasp): the hand's error at the grasp entry is either about 2 cm or about 30 cm. RGB-D: 18/30 reached the pregrasp, all 18 lifted. Stereo: 13/30 reached it, only 2 lifted. The straight reach is blocked in 12 vs 17 of 30, so the stereo near-total failure is mostly grasps failing after the hand arrived. BendPick: RGB-D side grasps in 22 episodes (2 lifted), top in 8 (8 lifted): decided by the grasp, as the author says.
>
> **SONIC noise.** On RGB-D TabletopGrasp, indices i and i+10 share the target pose and cached grasp (only distractors differ); SONIC disagrees on 3 (A), 2 (M), 5 (B), 4 (D), 7 (J) of these 10 pairs, ScaleBFM on 0–1. So SONIC's B, D, J (16, 19, 14) are within noise; M's deficit (4) is large and repeats on stereo, probably real.
>
> **4. Experiments that could overturn the conclusions.** E1 (depth vs target/grasp source): rebuild the RGB-D look with blind.py look --unnamed sam (SAM 3 target, MuJoCo depth) + grasps_graspgen.py; build a hybrid from the stereo dump's mask with its recorded model depth + GraspGen-X; run A on TabletopGrasp and BendPick, both controllers. If ScaleBFM TabletopGrasp drops toward 2/30 with the mask/grasp swapped while depth stays perfect, the difference is the target/grasp source, not depth. E2 (SONIC replicates): rerun SONIC A, B, D, J on TabletopGrasp three times jittering stand by ±5 mm; if each arm's replicate spread is at least its gap to the next arm, the ranking is noise.
>
> **Verdicts.** (a) Supports, with caveats: reproducible, the registered pipeline criterion passed; label it noise-free rendered depth, note effective n ≈ 10 on TabletopGrasp, say the settings were tuned on these episodes. (b) Partly supports: M ≥ A and J ≥ A on both metrics; D ≥ B only at 45 s as a ceiling tie (27 vs 27); stereo's D advantage (26 vs 14) does not replicate; at the official limit D falls well short (3 vs 14), which would trigger the criterion. (c) Supports the counts only for ScaleBFM (18 vs 2, paired 16/0, about 6 independent setups); SONIC fails about two-thirds on both looks (10 vs 7). "The straight reach is what fails" is not supported: stereo's failures were mostly after the reach; not enough evidence that depth is the cause.

### 复核后的更正（2026-11-24）

复核的数全部核对无误，下面照它改读法，原来的结论保留在上面不删：

1. **判据"B 比 D 多 5 局以上"触发了。** 登记里没写用哪个时限，主指标两个都报。官方时限内有路线的 28 局上 B 14、D 3。所以照判据：固定后撤的好处只在双目那一眼上成立，38.10.2 的读法要注明"只对双目"。45 秒内 27 对 27 是两边都到顶的并列。
2. **"直线伸手在 TabletopGrasp 上几乎全失败只对双目成立"改成：** ScaleBFM 上的计数是 18 对 2，但双目那次 13 局手到了预抓取点，只有 2 局抓起来。失败主要在手到了以后的抓取，不是伸手被挡（RGB-D 被挡 12 局，双目 17 局）。两眼的差来自目标和抓取的来源，没有证据说是深度。SONIC 两眼都失败约三分之二（10 对 7）。
3. **样本量。** TabletopGrasp 的 30 局都是同一个薯片罐；ScaleBFM 直线伸手的成败在三级里逐个相同（0001011111），有效样本约 10，第 3 张表的区间偏窄。SONIC 上 B、D、J（16、19、14）在噪声之内；M 的 4 局两眼都低，像是真的。
4. **这一眼的标注。** RGB-D 那一眼是复刻场景按物理几何渲染的深度，没有噪声，接近真实表面，报告里写成"理想深度"。目标规则和抓取设置在这些测试局上调过，双目那条找目标的规则也是数过这 30 局之后换的。
5. **没提的偏差补上：** 官方时限内 ScaleBFM 直线伸手总数 73，超出预测 30 到 70；SONIC 各做法的差 15，超出预测的 10。
6. **主指标不变**：83 / 73，TabletopGrasp 18 / 18，标"理想深度、TabletopGrasp 有效样本约 10"。
