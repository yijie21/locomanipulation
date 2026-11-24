# selfmodel_s_dev_20261124：S 这一行的开发局，标定误差半径、选修正方式

- 日期：2026-11-24（设计、预测、判据和选法，在开发局任何运行之前写下，单独提交）
- 机器 / GPU：4090，ScaleBFM 用 1 号卡；SONIC 用 2、3 号卡，0 号卡空闲时也用
- 代码版本：`selfwm` @ `6a5fd67d` 加未提交的文件（4090 的检出 `bc426d5` 加未提交的文件。跑到的文件和 Mac 逐个 md5 相同，只有 `simple_bench.py` 例外：4090 那份和 RGB-D 那次 J 跑的是同一份（md5 abc246fd），和 Mac 的 `99c0d3fd` 只差模块开头的帮助文字，代码相同），跑的时候抄进输出目录的 `asrun*/`：
  - 新的：
    - `scripts/experiments/simple_eval_20261112/s_check.py`：S 的检查、修正、挑选和计分；
    - `scripts/experiments/simple_eval_20261112/s_run.sh`：一轮轮检查和重新规划。
  - 改的：
    - `episodes.py`：`SIMPLE_LEVELS` 可以加等级，默认不变；
    - `bench_via.sh`：`SIMPLE_EVAL_OUT`、`BENCH_TASKS`，新的一步 `S`，默认不变。
- 配置：感知栏，`sensor=rgbd`；ScaleBFM 和 SONIC（`REACH_SONIC_NETS=gpu`）都不改参数；自身模型是 selfmodel_jrefwm_20261124 训好的两份（4090 的 `outputs/simple_eval_20261112/jrefwm_20261124/model_{scalebfm,sonic}`），不重训。
- 关联：
  - TODO 38.10.4；
  - [selfmodel_jrefwm_20261124](selfmodel_jrefwm_20261124.md)：模型，以及复核指出的半径问题，在 SIMPLE 的真实局上只罩住 34 % 和 83 %；
  - [simple_rgbd_20261124](simple_rgbd_20261124.md)：同一组合下 A、J 在测试局上的数；
  - 测试局的登记另写一份 `selfmodel_s_20261124`，在这里选完以后、测试局开跑以前提交。

## 要回答的问题

S 是"J 加自身模型"：
1. 发令以前，用该控制器的自身模型预测整条手臂实际会走到哪；
2. 对照开局那一眼的点云检查；
3. 不过关就让 cuRobo 重新规划；
4. 修不好就放弃这次伸手。

S 里有几样要定：误差半径、两条检查各开不开、阈值。这些都不能在测试局上定。这一份在开发局上把它们定下来，定法事先写好。

## 开发局

SIMPLE 下载包里 TabletopGrasp 还有两级评测局，测试从来没用过：`hard-old` 和 `hard`，各 10 局，共 20 个布局。和测试局的区别：
- 布局都是新的，目标位置不同；
- `hard` 每局 4 个干扰物（测试局 3 个），机器人起点往后 4 到 8 cm。

只用 TabletopGrasp，因为测试局里有路线的局几乎都在这个任务上：RGB-D 上 ScaleBFM 28 局、SONIC 29 局，其中 SONIC 有 1 局在 XMovePick。另外三个任务 cuRobo 规划不出路线，J 和 S 都退回直线伸手，两者相同。

这 20 局放在单独的目录树 `outputs/simple_dev_20261124/`（`SIMPLE_EVAL_OUT`）。它们记成第 3、4 级（`SIMPLE_LEVELS=0,1,2,3,4`），局号 30–49；测试局的输入一个不动。这一眼和测试局做法相同：
1. `blind.py look`：只看深度的规则找目标，不用 SAM 3；
2. GraspGen-X 出抓取；
3. `blind.py run` 量桌子前沿；
4. `look_frame.py` 补整帧点云。

差别：测试局的桌沿文件是 11-12 的代码做的，开发局用现在的代码。

## 做法

1. **每个控制器在开发局上跑 A、规划、JR。** JR 就是 J 带录像重跑，录下发令那一刻的状态和电机的参考。`bench_via.sh`，`BENCH_TASKS=TabletopGraspMP`，局号 30–49。
2. **重推参考的核对。** 从发令时刻的状态，照 `Motor.reach_joints` 的算法重推 J 路线的参考，和录像里存的参考比。已在双目那次的 JR 录像上做过，只查代码，不定参数：33 条路线最大差 0.08 cm，时长和路径长度全部相同。开发局上再报一次。
3. **标定误差半径 r。**
   - 用 `jref_data.py build` 把开发局的 JR 做成回放数据，`s_check.py calib` 算模型预测的扫掠误差；
   - r 取开发局路线扫掠误差的 90 % 分位，每个控制器各一个；
   - 同时报模型原来的固定半径在开发局上罩住多少。
4. **三种 S 在开发局上各跑一遍，两个控制器都跑。** 两条检查的定义：
   - **终点检查**：模型预测手掌停下时离规划终点的偏差超过 2 cm 就不过关；修法是把目标反向挪这个偏差，重新规划。
   - **间隙检查**：预测的每个球的路线离点云的最小间隙，减去球半径和 r，小于 0 就不过关；修法是把障碍物整体挪"最近那个球的预测偏差"的反方向，重新规划。
   - 检查用的点云：开局那一眼的体素，放在机器人以为的位置。和规划器一样，去掉起点手臂占着的和目标附近 2 cm 的。

   | 记号 | 终点检查 | 间隙检查 |
   |---|---|---|
   | S-end | 开 | 关 |
   | S-clear | 关 | 开 |
   | S-both | 开 | 开 |

   - 每种最多重新规划 2 轮，重新规划用 `plan2.py`，和 J 同一个规划器、同样的参数；
   - 每局取第一个过关的候选，2 轮后仍不过关就放弃这次伸手，计为失败；
   - J 的路线第一次就过关的局，S 发的就是 J 的路线，结果应当和 J 逐字段相同，这一条作核对。
5. **选法（写死）：**
   - 两个控制器在开发局上 45 秒内成功局数的总和最多的那种，定为测试局上的 S；
   - 并列时选重新规划局数少的；
   - 两个控制器用同一种，半径各用各的。

   如果三种都不比 J 多，测试局上仍跑总和最多的那种，并在测试局登记里写明"开发局上 S 没有赢 J"。

## 度量

1. 开发局上每个控制器：A、J、三种 S，在有路线的局上 45 秒内和官方时限（16 秒）内的成功数。S 放弃的局算失败。
2. 半径：开发局上模型预测扫掠误差的 50、80、90、95 % 分位；原固定半径罩住的比例。
3. 每种 S：第一轮过关、重新规划后过关、放弃各多少局；和 J 逐局配对，只有 S 成功、只有 J 成功各多少局。
4. 流程核对：重推参考和录像的最大差；J 的路线第一次过关的局，S 和 J 的结果相同的局数。

## 预测（启动前写，首次启动后锁定）

- **流程**：
  - 20 局里有路线的，每个控制器 12 到 20 局（测试局 30 局里 28、29 局有路线）；
  - 重推参考和录像的最大差不超过 0.2 cm；
  - J 的路线第一次过关的局，S 的结果和 J 全部相同。
- **半径**：
  - 原固定半径（ScaleBFM 2.87 cm）在开发局上罩住 0.3 到 0.8；
  - 90 % 分位：ScaleBFM 2.5 到 6 cm，SONIC 4 到 15 cm。
  - 理由：复核在 SIMPLE 测试局上看到预测误差中位数 2.8 cm，比留出的采集数据大，开发局应当类似。
- **J 在开发局有路线的局上，45 秒内**：ScaleBFM 不少于有路线局数的 70 %（测试局 28 / 28），SONIC 30 % 到 70 %（测试局 14 / 29）。
- **终点检查第一轮不过关的比例**：ScaleBFM 10 % 到 60 %，SONIC 50 % 到 100 %。理由：测试局上 SONIC 的手离预抓取点中位数 6.3 cm，ScaleBFM 3.8 cm。
- **间隙检查第一轮不过关的比例**：两个控制器都在 10 % 到 70 %，宽区间：r 还没定，桌面离手的终点很近。
- **三种 S 对 J**：
  - ScaleBFM 上没有一种比 J 多 2 局以上，J 已接近满分；
  - SONIC 上至少一种不少于 J。
  - 不预测选中哪一种。

## 判据（什么结果会让我改做法）

- **流程没过**：重推参考差超过 0.5 cm，或者 J 的路线第一次过关的局里有一局 S 和 J 结果不同。先查代码，不选，不跑测试局。
- **某一种 S 放弃了一半以上有路线的局**：这种检查对这一眼太严，选法照旧，但在测试局登记里写明。
- **三种都比 J 少**：照上面的选法定，测试局登记里写明，并预测 S 在测试局上不比 J 多。
- 其余情况：照选法定，写进测试局登记。

## 预测修订

- 2026-11-24 14:10Z，第一次启动后，任何 S 局跑之前：流程停在第一轮重新规划后的检查上，两处代码错误，改完从三种 S 的第 0 轮重跑；A、规划、JR 和半径不重跑（局和输入都没变）。预测、判据、选法都不改。
  1. 重新规划失败（cuRobo 报 IK_FAIL）时 `plan2.py` 也写一个计划文件，只是里面没有轨迹，检查读它就出错。改成：没有轨迹就算"这一轮没有候选"，不再重新规划，前面各轮也没过关就放弃（登记里"2 轮后仍不过关就放弃"）。
  2. 终点检查在修过的候选上，量的是模型预测的偏差本身，而不是预测的手掌离原目标多远。登记的修法是"把目标反向挪这个偏差"，让手落回原目标；照原来的算法，模型只要预测出稳定的偏差，修过的候选永远不过关，修法形同虚设。改成：预测的手掌静止位置对 J 自己那条计划的终点（第 0 轮两者相同，结果不变）。登记原文"离规划终点的偏差"在修过的候选上有两种读法，这里取和修法一致的那种。
  - 两处都在旧的输出上用一轮检查验证过（不跑新的局）：ScaleBFM 的终点检查，第 1 轮 7 个候选预测离原目标 0.07–1.02 cm，都过关；另 4 局重新规划没有轨迹。

## 结果（贴原始输出，不转述）

输出在 4090 的 `outputs/simple_dev_20261124/`。

`logs/dev.log`（第二次启动，第一次停在检查上，见"预测修订"）：

```
[13:51:17] grasps (exit 0): 20 files
[13:54:31] table edge (exit 0): 20 rows, 20 with an edge, 18 blind.py successes
[14:03:09] scalebfm: A, plan, JR done; [14:03:09] DONE JR
[14:03:14] scalebfm: radius (90 % of the development paths' errors) 4.06 cm; {"reference": {"n": 11, "q50": 7.61, "q80": 11.71, "q90": 12.49, "q95": 20.79, "max": 29.1, "palm_median": 5.65, "inside_fixed": 0.818, "fixed_cm": 12.4}, "direction": {"n": 11, "q50": 3.12, "q80": 3.66, "q90": 4.06, "q95": 12.78, "max": 21.49, "palm_median": 3.06, "inside_fixed": 0.364, "fixed_cm": 2.87}}
[14:04:34] sonic: A, plan, JR done; [14:04:34] DONE JR
[14:04:39] sonic: radius (90 % of the development paths' errors) 16.99 cm; {"reference": {"n": 15, "q50": 12.81, "q80": 20.57, "q90": 24.08, "q95": 26.97, "max": 29.92, "palm_median": 10.28, "inside_fixed": 0.867, "fixed_cm": 22.42}, "direction": {"n": 15, "q50": 6.39, "q80": 11.58, "q90": 16.99, "q95": 20.23, "max": 26.28, "palm_median": 4.9, "inside_fixed": 0.733, "fixed_cm": 10.29}}
[14:07:50] rerun after the two check fixes (see the entry 预测修订): resumes at the three checks
[14:08:58] scalebfm S_end: [14:08:58] chosen: 6 via files, given up 5
[14:09:19] sonic S_end: [14:09:19] chosen: 10 via files, given up 5
[14:11:24] scalebfm S_clear: [14:11:24] chosen: 0 via files, given up 11
[14:11:56] sonic S_clear: [14:11:56] chosen: 0 via files, given up 15
[14:13:10] scalebfm S_both: [14:13:10] chosen: 0 via files, given up 11
[14:14:12] sonic S_both: [14:14:12] chosen: 0 via files, given up 15
[14:18:13] all done
```

规划（`bench/<控制器>/pipeline.log`）：ScaleBFM `step 2b: 20 rows, planned 11`，SONIC `step 2b: 20 rows, planned 15`，2 cm 重试都是 `planned 0`。

各轮（`bench/<控制器>/S_<种>/s_pipeline.log`）：

```
scalebfm S_end   round 0: 11 checked, 0 passed | replan 1: 11 rows, planned 6 | round 1: 6 checked, 6 passed | chosen: 6 via files, given up 5
scalebfm S_clear round 0: 11 checked, 0 passed | replan 1: planned 9 | round 1: 9 checked, 0 passed | replan 2: planned 3 | round 2: 3 checked, 0 passed | given up 11
sonic    S_end   round 0: 15 checked, 0 passed | replan 1: 15 rows, planned 10 | round 1: 10 checked, 10 passed | chosen: 10 via files, given up 5
sonic    S_clear round 0: 15 checked, 0 passed | replan 1: planned 10 | round 1: 10 checked, 0 passed | replan 2: planned 5 | round 2: 5 checked, 0 passed | given up 15
S_both 两个控制器 round 0 都是 0 passed，最后全部放弃（11、15）
```

终点修法重新规划失败的原因（`plan_round1.jsonl` 里 `"ok": false` 的行）：10 局 `MotionGenStatus.IK_FAIL`，1 局 `FINETUNE_TRAJOPT_FAIL`。

重推参考对录像（`S_end/state.jsonl`）：ScaleBFM 11 局最大 0.055 cm，SONIC 15 局最大 0.081 cm；时长和路径长度全部相同。

表（`s_check.py table <S> <bench> --arm S_<种> --j-arm JR`，`S_<种>/table.txt`；第一次算表在配对那段报错，改了以后从同样的行重算）：

```
===== scalebfm S_end
  TabletopGraspMP (20)         4 / 4      10 / 7       6 / 3
episodes J ran a plan on: 11; S kept J's plan 0 (rows identical to J's: 0), replanned and passed 6, gave up 5
  45 s successes: J 10, S 6, S - J -4 (95 % interval -7 to -1), only S 0, only J 4
  official limit successes: J 7, S 3, S - J -4 (95 % interval -7 to -1), only S 0, only J 4
  TabletopGraspMP_L3_e0      chosen None J 1 S 0 | round 0: end 5.27 cm, clear -4.33 cm (gap -1.33, reference 1.16), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L3_e2      chosen 1    J 1 S 1 | round 0: end 8.83 cm, clear -4.59 cm (gap -1.59, reference -0.41), worst hand_middle_0_link at u 0.565
  TabletopGraspMP_L3_e3      chosen 1    J 1 S 1 | round 0: end 8.24 cm, clear -4.6 cm (gap -1.6, reference -0.87), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L3_e5      chosen 1    J 1 S 1 | round 0: end 4.51 cm, clear -4.48 cm (gap -1.48, reference 0.76), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L3_e6      chosen 1    J 1 S 1 | round 0: end 7.82 cm, clear -3.99 cm (gap -0.99, reference -0.63), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L3_e7      chosen 1    J 1 S 1 | round 0: end 3.83 cm, clear -4.38 cm (gap -1.38, reference 1.01), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L3_e8      chosen 1    J 1 S 1 | round 0: end 4.89 cm, clear -4.33 cm (gap -1.33, reference 0.93), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L3_e9      chosen None J 1 S 0 | round 0: end 4.66 cm, clear -4.6 cm (gap -1.6, reference 1.13), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L4_e1      chosen None J 1 S 0 | round 0: end 4.83 cm, clear -3.0 cm (gap -0.0, reference -0.9), worst hand_index_1_link at u 0.391
  TabletopGraspMP_L4_e8      chosen None J 1 S 0 | round 0: end 5.05 cm, clear -2.91 cm (gap 0.09, reference 0.54), worst hand_index_1_link at u 0.087
  TabletopGraspMP_L4_e9      chosen None J 0 S 0 | round 0: end 5.27 cm, clear -3.59 cm (gap -0.59, reference -0.19), worst hand_index_1_link at u 0.391
===== scalebfm S_clear / S_both（两者相同）
  TabletopGraspMP (20)         4 / 4      10 / 7       0 / 0
  45 s successes: J 10, S 0, S - J -10 (95 % interval -11 to -8), only S 0, only J 10
  official limit successes: J 7, S 0, S - J -7 (95 % interval -10 to -4), only S 0, only J 7
===== sonic S_end
  TabletopGraspMP (20)         5 / 5      10 / 6       8 / 4
episodes J ran a plan on: 15; S kept J's plan 0 (rows identical to J's: 0), replanned and passed 10, gave up 5
  45 s successes: J 10, S 8, S - J -2 (95 % interval -5 to +0), only S 0, only J 2
  official limit successes: J 6, S 4, S - J -2 (95 % interval -5 to +0), only S 0, only J 2
  TabletopGraspMP_L3_e0      chosen 1    J 1 S 1 | round 0: end 3.96 cm, clear -3.95 cm (gap -0.95, reference 0.86), worst hand_index_0_link at u 0.217
  TabletopGraspMP_L3_e2      chosen None J 0 S 0 | round 0: end 19.33 cm, clear -0.86 cm (gap 2.14, reference -1.73), worst hand_middle_1_link at u 0.522
  TabletopGraspMP_L3_e3      chosen None J 0 S 0 | round 0: end 11.01 cm, clear -6.69 cm (gap -3.69, reference -1.44), worst wrist_pitch_link at u 0.217
  TabletopGraspMP_L3_e4      chosen 1    J 1 S 1 | round 0: end 5.34 cm, clear -4.6 cm (gap -1.6, reference -0.82), worst hand_palm_link at u 0.217
  TabletopGraspMP_L3_e5      chosen 1    J 0 S 0 | round 0: end 5.83 cm, clear -3.83 cm (gap -0.83, reference -0.82), worst hand_palm_link at u 0.043
  TabletopGraspMP_L3_e6      chosen 1    J 1 S 1 | round 0: end 17.15 cm, clear -4.51 cm (gap -1.51, reference -1.02), worst hand_index_1_link at u 0.435
  TabletopGraspMP_L3_e7      chosen 1    J 1 S 1 | round 0: end 10.1 cm, clear -2.21 cm (gap 0.79, reference 0.33), worst wrist_pitch_link at u 0.0
  TabletopGraspMP_L3_e8      chosen None J 1 S 0 | round 0: end 8.67 cm, clear -6.1 cm (gap -3.1, reference -1.21), worst wrist_pitch_link at u 0.13
  TabletopGraspMP_L3_e9      chosen 1    J 1 S 1 | round 0: end 13.13 cm, clear -4.35 cm (gap -1.35, reference -0.68), worst hand_index_0_link at u 0.348
  TabletopGraspMP_L4_e0      chosen None J 1 S 0 | round 0: end 6.54 cm, clear -4.76 cm (gap -1.76, reference 0.28), worst hand_palm_link at u 0.174
  TabletopGraspMP_L4_e1      chosen 1    J 1 S 1 | round 0: end 3.26 cm, clear -5.44 cm (gap -2.44, reference -0.25), worst wrist_pitch_link at u 0.13
  TabletopGraspMP_L4_e3      chosen 1    J 1 S 1 | round 0: end 4.44 cm, clear -4.67 cm (gap -1.67, reference 0.99), worst hand_middle_1_link at u 0.217
  TabletopGraspMP_L4_e4      chosen 1    J 1 S 1 | round 0: end 2.42 cm, clear -4.45 cm (gap -1.45, reference 0.17), worst hand_palm_link at u 0.217
  TabletopGraspMP_L4_e5      chosen None J 0 S 0 | round 0: end 13.93 cm, clear -1.33 cm (gap 1.67, reference 5.03), worst hand_index_0_link at u 0.217
  TabletopGraspMP_L4_e8      chosen 1    J 0 S 0 | round 0: end 6.33 cm, clear -2.83 cm (gap 0.17, reference 2.43), worst hand_palm_link at u 0.261
===== sonic S_clear / S_both（两者相同）
  TabletopGraspMP (20)         5 / 5      10 / 6       0 / 0
  45 s successes: J 10, S 0, S - J -10 (95 % interval -13 to -6), only S 0, only J 10
  official limit successes: J 6, S 0, S - J -6 (95 % interval -10 to -2), only S 0, only J 6
```

（每张表的三列是 A、J（开发局上是 JR）、S，45 秒内 / 官方时限内。）

## 意外

逐条对预测：
1. 有路线的局：ScaleBFM 11，预测 12–20，少 1 局；SONIC 15，在区间内。
2. 重推参考差 0.055 / 0.081 cm，预测不超过 0.2 cm，符合。
3. "J 的路线第一次过关的局，S 和 J 全部相同"：一局都没有第一次过关，无从验证。
4. 原固定半径罩住的比例 0.364 / 0.733，预测 0.3–0.8，符合。
5. 90 % 分位：ScaleBFM 4.06 cm，在 2.5–6 内；SONIC 16.99 cm，超过预测上限 15。
6. J 在有路线的局上 45 秒内：ScaleBFM 10 / 11（91 %），预测不少于 70 %，符合；SONIC 10 / 15（67 %），在 30–70 % 内。
7. 终点检查第一轮不过关的比例：ScaleBFM 100 %，预测 10–60 %，超出；SONIC 100 %，在 50–100 % 内。模型对每条路线都预测手掌停下时差 2.4 到 19 cm，ScaleBFM 3.8 到 8.8 cm，没有一条在 2 cm 内。
8. 间隙检查第一轮不过关的比例：两个控制器都是 100 %，预测 10–70 %，超出。原因在 J 自己的参考就贴着点云：参考离保留下来的体素只有 −1.7 到 +1.2 cm（SONIC 一局 5 cm），而半径是 4.06 / 16.99 cm，所以不论预测多准都过不了。
9. ScaleBFM 没有一种比 J 多 2 局以上：符合（都更少）。SONIC 至少一种不少于 J：不符合，最好的 S_end 8 对 J 10。
10. 没想到的：
    - 终点修法挪了目标、重新规划出路线的局，S 的成败和 J 逐局相同（ScaleBFM 6 / 6，SONIC 10 / 10），目标挪 4 到 9 cm 没改变任何一局的结果。S 少的局全在放弃的那些里（ScaleBFM 放弃 5 局，其中 J 成功 4 局；SONIC 放弃 5 局，J 成功 2 局）。
    - 放弃的原因几乎都是挪过的目标 IK 无解（10 / 11 是 IK_FAIL）：反着模型预测的偏差挪，目标往下压进桌子。

## 结论与下一步

- **照选法定：S_end。** 两个控制器 45 秒内成功数之和：S_end 6 + 8 = 14，S_clear 0，S_both 0。测试局上的 S 用 S_end（`--tau-end 2 --delta 1e9`，最多 2 轮），间隙检查不开，所以半径在测试局上不起作用。
- **对照判据：**
  - 流程判据没有触发：重推参考差远小于 0.5 cm；"J 的路线第一次过关的局 S 和 J 不同"一局都没有，因为没有这种局。
  - "某一种 S 放弃了一半以上有路线的局"：S_clear 和 S_both 在两个控制器上都放弃了全部，触发；S_end 放弃 5 / 11（45 %）、5 / 15（33 %），没触发。照判据，选法照旧，测试局登记里写明。
  - "三种都比 J 少"：触发。照判据写进测试局登记："开发局上 S 没有赢 J"，并预测 S 在测试局上不比 J 多。
- **读法：** 在这批开发局上，自身模型的检查只会让 S 失分不会得分：修过的局结果不变，放弃的局丢分。证据强度：每个控制器一次运行，20 局（有路线的 11 和 15 局），同一个薯片罐，局之间相关。
- **其他可能的解释：**
  - 模型的终点预测偏大（复核已指出它在 SIMPLE 局上的误差大于留出数据）；如果偏差是假的，挪目标就是白挪，只是没造成伤害，因为后面的抓取阶段会纠正；
  - 或者偏差是真的，但 SIMPLE 的抓取阶段本来就吸收 4–9 cm 的伸手误差，所以修不修都一样。这两种解释在这批数据里分不开，要看手实际停下的位置（录像里有），下一步在测试局的 S 录像上比。
- **四类陷阱：**
  - benchmark 选择：开发局是 SIMPLE 下载包里所有没用过的 TabletopGrasp 局，没有挑；
  - 数据泄漏：开发局没进训练，测试局的输入一个没动（输出目录分开）；
  - 指标误用：选法事先写死用 45 秒内的和，两个时限都报了，官方时限内的排序相同（S_end 3 + 4 = 7，另两种 0）；
  - 事后挑选：三种都报了，一次运行。中途改了两处代码（"预测修订"），改在任何 S 局跑之前。
- **下一步：** 写测试局 S 的登记（selfmodel_s_20261124）并提交，跑 S_end；同时在测试局上算模型的预测误差和起始状态是否在训练范围内。

## 独立复核

和测试局一起复核（会写进论文的是测试局的数）。
