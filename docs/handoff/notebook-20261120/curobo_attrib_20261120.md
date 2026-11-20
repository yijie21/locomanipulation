# curobo_attrib_20261120：TabletopGrasp 不再卡住，是因为绕开了桌沿，还是因为多段、变慢、换了姿态时间表？

- 日期：2026-11-20
- 机器 / GPU：4090（`ssh -p 27984 root@14.103.60.188`），只用 CPU，只跑 direct。
- 代码版本：sonic-selfmodel @ 4344165c，加未提交的改动：
  - `motor.reach` 的 `via_rot`；
  - `simple_bench.py --avoid`，这次再加：路点文件可以带 `speed`。
- 配置 / EXP：
  - SIMPLE TabletopGraspMP 感知栏（Isaac 那一眼，`cloud_eyes_final`）；
  - 有 cuRobo 路线的 28 局，即 30 局去掉没有抓取的 ep 12 和没有路线的 ep 24。
- 关联：curobo_avoid_20261120（复核的 E1），beliefs B43、B45。

## 假设

- **H1'**：卡住消失是因为手的**位置路径**先后撤、再抬过桌沿，而不是因为下面这几样，所以只要位置路径绕开，姿态怎么转都不会卡：
  - 把伸手切成多段；
  - 参考时长变长（2.6 → 5.7 s）；
  - 手腕姿态按 cuRobo 的时间表转。
- **H2'**：在这个基准里，**不用感知**的固定"先撤后抬"模板，以及别的局的 cuRobo 路线，效果和本局按那一眼规划的路线差不多。理由是 30 局的桌子都摆在机器人正前方、桌沿离骨盆 13–17 cm，场景几乎一样。如果成立，就说明这个基准证明不了"按感知规划"的必要性，只证明了"要先撤后抬"。

## 预测（启动前写，首次启动后锁定）

**设定：**
- 7 组路线，每组 28 局 × 3 个站位（`--side-off` −0.20 / −0.19 / −0.21），共 84 次伸手。
- 主指标："卡住"，即预抓取停在离目标 20 cm 以外，统计 84 次里卡住了几次。
- 各组路线：
  - **A0 直线**：原样。
  - **A1 规划**：cuRobo 路线的位置加 cuRobo 的姿态，同 curobo_avoid 第 3 步。
  - **B 直线分段**：直线上均匀放和 A1 同样多的路点，速度调到总时长和 A1 一样；姿态按直线伸手的方式，沿路程均匀从起点转到目标。
  - **C 规划位置 + 直线姿态**：A1 的位置路点，姿态沿路程均匀转。
  - **D 直线位置 + 规划姿态**：B 的位置和速度，A1 的姿态路点。
  - **E 固定模板**：不用感知。先沿机器人朝向后撤 8 cm，再抬到预抓取高度，再去目标；姿态沿路程均匀转。
  - **F 错配路线**：用下一局（按序循环）的 cuRobo 路点相对它起点的位移，按两局的朝向差转过来，加在本局起点上；姿态沿路程均匀转。

**预测（卡住次数 / 84）：**

| 组 | 预测 | 理由 |
|---|---|---|
| A0 | 45–60 | 名义站位是 17/28；±1 cm 横移不改变"手从桌板下面直着往上" |
| A1 | 0–6 | 名义站位是 0/28 |
| B | 40–60 | H1'：参考轨迹照样穿过桌板，多段和变慢不帮忙 |
| C | 0–8 | H1'：位置路径是关键 |
| D | 40–60 | 同 B |
| E | 0–10 | H2'：场景几乎一样，先撤后抬就够。最不确定，因为 8 cm 后撤可能不够让手指离开桌板下沿；cuRobo 路线后撤 5–10 cm，而且同时在转腕 |
| F | 0–10 | H2' |

**次指标（报告，不作判据）：**
- 45 s 成功和 16 s 官方成功；
- 碰桌（手离开出发点 ≥ 3 cm 后，右侧部位还接触桌子）的局数和 tick 数；
- 手点路径长度。

## 判据（什么结果会让我放弃或修改假设）

- **B ≤ 12 / 84**（卡住率 ≤ A0 的约 1/4）：H1' 错，卡住消失主要来自多段和变慢。curobo_avoid 的"绕行"归因作废，下一步查是不是参考太快、控制器跟不上。
- **C ≥ 30 且 D ≤ 12**：起作用的是手腕姿态时间表，不是位置路径。H1' 错，下一步查姿态。
- **C 和 D 都 ≤ 12**：位置和姿态任一个都够。H1' 部分成立，记为"两条路都能避开"。
- **E 和 F 都 ≤ A1 + 6**：H2' 成立。这个基准证明不了按感知规划的必要性。
  - 对外只能说"要先撤后抬，cuRobo 加那一眼是一种通用实现"。
  - 要证明感知必要，下一步做随机障碍物测试（桌沿位置、高度、障碍物随机），对比固定模板和 cuRobo。
- **E 或 F ≥ A1 + 15**：按本局感知规划确实有用，H2' 错。
- **A1 ≥ 10 / 84**：curobo_avoid 的结果对 1 cm 站位扰动不稳，要先查这个，再看其余各组。

## 预测修订

## 结果（贴原始输出，不转述）

**跑法：**4090 上 tmux `curobo_attrib`（`curobo/run_attrib.sh`），07:55:46 开始，08:01:27 结束。588/588 行，0 个进程失败。

- 路点文件由 `curobo/vias_e1.py` 生成，逐局路程和时长见 `attrib/vias.txt`：B、D 的时长和 A1 逐局相同；E 的路程 65–72 cm、约 4.7 s；F 的路程 75–89 cm。
- 冒烟测试（ep 0，名义站位）：B 卡住（预抓取差 30.07 cm）；E 没卡（3.37 cm，15.3 s 成功）。
- 汇总：`python3 attrib_sum.py attrib/rows --per`，输出在 `attrib/summary.txt`。逐局一行，每组三个站位依次是 −0.19、−0.20、−0.21，X 表示卡住。

```
A0 {"n": 84, "stuck": 49, "stuck_by_stance": {"-0.19": 17, "-0.20": 17, "-0.21": 15}, "s45": 6, "s16": 6, "table": 84, "hits_median": 195.5, "path_cm_median": 25.8, "dur_s_median": 5.46, "vs_A1": "only A0 stuck 49, only A1 stuck 0, p 3.6e-15"}
A1 {"n": 84, "stuck": 0, "stuck_by_stance": {"-0.19": 0, "-0.20": 0, "-0.21": 0}, "s45": 33, "s16": 15, "table": 81, "hits_median": 46.5, "path_cm_median": 78.5, "dur_s_median": 8.5, "vs_A0": "only A1 stuck 0, only A0 stuck 49, p 3.6e-15"}
B {"n": 84, "stuck": 49, "stuck_by_stance": {"-0.19": 16, "-0.20": 17, "-0.21": 16}, "s45": 7, "s16": 4, "table": 84, "hits_median": 294.0, "path_cm_median": 27.4, "dur_s_median": 8.6, "vs_A0": "only B stuck 4, only A0 stuck 4, p 1", "vs_A1": "only B stuck 49, only A1 stuck 0, p 3.6e-15"}
C {"n": 84, "stuck": 1, "stuck_by_stance": {"-0.19": 1, "-0.20": 0, "-0.21": 0}, "s45": 29, "s16": 13, "table": 84, "hits_median": 169.0, "path_cm_median": 74.0, "dur_s_median": 8.4, "vs_A0": "only C stuck 1, only A0 stuck 49, p 9.1e-14", "vs_A1": "only C stuck 1, only A1 stuck 0, p 1"}
D {"n": 84, "stuck": 41, "stuck_by_stance": {"-0.19": 14, "-0.20": 13, "-0.21": 14}, "s45": 9, "s16": 4, "table": 84, "hits_median": 275.0, "path_cm_median": 56.0, "dur_s_median": 8.6, "vs_A0": "only D stuck 7, only A0 stuck 15, p 0.13", "vs_A1": "only D stuck 41, only A1 stuck 0, p 9.1e-13"}
E {"n": 84, "stuck": 0, "stuck_by_stance": {"-0.19": 0, "-0.20": 0, "-0.21": 0}, "s45": 45, "s16": 45, "table": 84, "hits_median": 17.0, "path_cm_median": 65.4, "dur_s_median": 7.54, "vs_A0": "only E stuck 0, only A0 stuck 49, p 3.6e-15", "vs_A1": "only E stuck 0, only A1 stuck 0, p 1"}
F {"n": 84, "stuck": 3, "stuck_by_stance": {"-0.19": 2, "-0.20": 0, "-0.21": 1}, "s45": 27, "s16": 10, "table": 84, "hits_median": 167.0, "path_cm_median": 74.0, "dur_s_median": 8.62, "vs_A0": "only F stuck 0, only A0 stuck 46, p 2.8e-14", "vs_A1": "only F stuck 3, only A1 stuck 0, p 0.25"}
0 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
1 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
2 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
3 A0:... A1:... B:... C:... D:... E:... F:...
4 A0:... A1:... B:... C:X.. D:... E:... F:...
5 A0:... A1:... B:XXX C:... D:XXX E:... F:...
6 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
7 A0:XXX A1:... B:.XX C:... D:... E:... F:...
8 A0:... A1:... B:... C:... D:... E:... F:...
9 A0:XXX A1:... B:XXX C:... D:... E:... F:...
10 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
11 A0:XXX A1:... B:XXX C:... D:... E:... F:X..
13 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
14 A0:XXX A1:... B:XX. C:... D:XXX E:... F:...
15 A0:... A1:... B:... C:... D:... E:... F:...
16 A0:XXX A1:... B:XXX C:... D:... E:... F:...
17 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
18 A0:... A1:... B:... C:... D:... E:... F:...
19 A0:... A1:... B:... C:... D:XXX E:... F:...
20 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
21 A0:XXX A1:... B:XXX C:... D:XXX E:... F:X.X
22 A0:XXX A1:... B:XXX C:... D:XXX E:... F:...
23 A0:... A1:... B:... C:... D:... E:... F:...
25 A0:... A1:... B:... C:... D:... E:... F:...
26 A0:XX. A1:... B:XXX C:... D:X.X E:... F:...
27 A0:XX. A1:... B:... C:... D:... E:... F:...
28 A0:... A1:... B:... C:... D:... E:... F:...
29 A0:... A1:... B:... C:... D:... E:... F:...
```

## 意外

- 各组卡住次数全部落在预测区间内：A0 49、A1 0、B 49、C 1、D 41、E 0、F 3。
- **不用感知的固定模板 E 比 cuRobo 路线 A1 还好**：
  - 45 s 成功 45 / 84 对 33 / 84，16 s 官方成功 45 / 84 对 15 / 84；
  - 碰桌 tick 中位数 17 对 46.5；
  - 路程 65 对 79 cm，时长 7.5 对 8.5 s。

  E 的成功全部在 16 s 以内（s45 = s16 = 45），A1 的 33 次成功里只有 15 次在 16 s 以内。
- 所有组都碰桌（84 / 84），只是持续时间不同：E 17 tick，A1 46.5，A0 195.5，B 294。"手离开出发点 3 cm 后右侧仍接触桌子"这个指标太宽，分不开"擦一下"和"一直顶着"。
- 另外，B（同段数、同时长的直线）比直线 A0 碰桌更久（294 对 195.5 tick），变慢只是让手顶在桌沿上的时间更长。
- D 比 A0 少卡 8 次（41 对 49；只有 D 卡 7 次，只有 A0 卡 15 次，p = 0.13），姿态时间表有一点作用，但不显著。

## 结论与下一步

（2026-11-20，独立复核前的草稿）

1. **H1' 成立：起作用的是手的位置路径（先撤后抬），不是多段、变慢或姿态时间表。**
   - B（同段数、同时长的直线，直线姿态）卡住 49 / 84，和直线 A0 的 49 一样；逐局配对 4 : 4，p = 1。
   - C（规划位置 + 直线姿态）卡住 1，和 A1 的 0 无差（1 : 0）。
   - D（直线位置 + 规划姿态）卡住 41，和 A0 差异不显著（7 : 15，p = 0.13）。
   - 判据上：B > 12；C ≤ 12 且 D ≥ 30 → H1' 不被推翻，位置是主因。
2. **H2' 成立：在这个基准上，按那一眼规划的路线没有优势。**
   - E（固定模板：后撤 8 cm、抬到预抓取高度、再去目标，不用感知）卡住 0 / 84；F（别局的路线）卡住 3 / 84。都 ≤ A1 + 6。
   - E 在成功率上还明显优于 A1：16 s 官方成功 45 对 15 / 84，45 s 成功 45 对 33。
   - 所以 curobo_avoid 里"cuRobo 加那一眼"的价值，在 TabletopGrasp 上体现不出来：30 局的桌子都在正前方、桌沿离骨盆 13–17 cm，一个固定动作就够。
3. **对外怎么说：**TabletopGrasp 失败的原因是"手从桌板下面直线往上"；"伸手前先后撤再抬"就能解决。cuRobo 加感知是一种通用实现，但这个基准证明不了它比固定模板好。要证明感知规划的必要性，得换成障碍物位置、高度随机的测试。
4. **证据强度：**direct 一种方法；28 局 × 3 个横向站位（±1 cm）= 84 次；一个种子。站位扰动下各组的卡住数很稳（A0 17 / 17 / 15，B 16 / 17 / 16）。model 没跑。

**其他可能的解释：**
- E 好过 A1，可能不是"路线形状更好"，而是 E 的路程更短（65 对 79 cm），而且第二段竖直上抬时手正好在桌沿前方。也可能和 A1 有段数差异有关：E 只有 2 个路点，A1 有 7–10 个，每段至少 0.6 s。
- E 的 8 cm 后撤和抬高量，是我根据 cuRobo 路线的样子（后撤 5–10 cm）定的，不是完全独立于 curobo_avoid 的结果。

**四类陷阱：**
- benchmark 选择：只在 TabletopGrasp 一个任务上，而且正是它的场景单一，才让模板够用。这一点就是本条结论本身，已写明。
- 数据泄漏：E 只用了目标（预抓取位姿）和机器人朝向，没用障碍物信息。F 用了别局的感知规划。
- 指标误用：主指标"卡住"按预注册的定义没变。碰桌指标太宽（所有组都是 84 / 84），已在意外里写明，没有拿它下结论。
- 事后挑选：7 组和 3 个站位全部报告；各组设计在启动前写进了条目。

**对照判据：**
- B > 12 → 不是"变慢"；
- C ≤ 12、D ≥ 30 → 是位置；
- E、F ≤ A1 + 6 → H2' 成立 → 下一步做随机障碍物测试。

**下一步：**
1. 如果要继续推"按感知避障"：做随机障碍物测试（桌沿位置、桌面高度、额外障碍物随机），比较模板 E 和 cuRobo。
2. 如果只要 SIMPLE 榜单上的 TabletopGrasp：用 E 这种"先撤后抬"，比 cuRobo 简单而且快，16 s 内成功 45 / 84。不过这正是用户担心的"像在刷榜"的那类手写规则，要先和用户讨论再用。

### 复核后修订（2026-11-20，以此为准）

1. **结论 1 保留：**分段、变慢都不是原因。
   - B = A0（49 对 49）。
   - 复核把 B 再放慢到 0.03 m/s，16 个卡住局里放出 3 个（3 : 0，p = 0.25）：极慢时速度最多有一点作用。
2. **结论 2 修正：**起作用的是"先把手抬到预抓取高度"，不是"后撤"。
   - E0（直接竖直抬到预抓取高度、不后撤）在 16 个卡住局里卡住 0，全部 26 局卡住 0。
   - E8nl（只后撤 8 cm、不抬）卡住 11 / 16。
   - 姿态也不是完全无关：D 放出了第 7、9、11、16 局，又新卡住了第 5、19 局。
3. **结论 3 保留，但只限这个基准。**
   - 原因：28 局是同一张桌子、障碍永远在目标下方，"先抬到目标高度"天然就能绕开。
   - 所以这个基准既区分不了按感知规划和模板，也区分不了 F 借来的路线。
   - E 每局没用场景信息；但 8 cm 和两段式是看过 cuRobo 在这些局上的路线后定的，算不上"完全不用场景知识"。
4. **结论"E 的成功率高于 A1"撤回。**
   - 16 s 成功 45 对 15 的差距，主要是 E 的路程短了约 1 s（中位差 0.95 s）。
   - E 有一次是假成功（`E_-0.19_9`，食指把物体顶起一下），真实是 44 / 84。
   - 45 s 成功逐局 8 : 1（p = 0.039），但同类模板 E0 没复现（名义站位 11 对 11）。
   - 这条本来就是预注册时写明"报告，不作判据"的次指标。
5. **证据强度修正：**84 次伸手实际约等于 28 个独立局。
   - ±1 cm 站位只让起点移动 3–4 mm；第 0 局和第 10 局的起始状态相同。
   - 逐局检验：D 对 A0 是 2 : 5，p = 0.45。
6. **对 curobo_avoid 的影响：**TabletopGrasp 的卡住原因是"手从桌板下面往前伸，还没抬过桌面高度"。
   - cuRobo 加那一眼能解决，"先竖直抬到预抓取高度"同样能解决。
   - 要证明感知规划有用，必须换成障碍物随机的场景。

## 独立复核

（general-purpose 独立 agent，只给了假设、预测、判据、结果四节，代码和数据路径，以及待核的四条论断；原文如下）

Verdicts first: claims 1–3 are supported. Claim 4's numbers reproduce, but it doesn't show the fixed template beats the planned route (证据不足). All your numbers match the raw rows exactly. I found no implementation bug. Three of my 60 reruns change how to read the result:

- **Lifting the hand straight up from the start, with no back-off (E0), removes the stuck by itself.** Backing off 8 cm and then going straight to the goal, with no lift (E8nl), leaves 11 of 16 stuck. What works is "rise first", not "back off".
- **The 16 s success gap (45 vs 15) comes mostly from E's path being about 1 s shorter** (median gap 0.95 s, range 0.53–2.89 s). It says little about path shape.
- **The 84 reaches behave like about 28 independent episodes.** Tested per episode, D vs A0 gives p=0.45, not 0.13.

## 1. Recount from the raw rows (588 rows, none missing)

| Group | Stuck (−0.19/−0.20/−0.21) | Success 45 s / 16 s | Success among non-stuck reaches |
|---|---|---|---|
| A0 | 49 (17/17/15) | 6 / 6 | 6/35 = 0.17 |
| A1 | 0 | 33 / 15 | 33/84 = 0.39 |
| B | 49 (16/17/16) | 7 / 4 | 6/35 |
| C | 1 | 29 / 13 | 29/83 |
| D | 41 (14/13/14) | 9 / 4 | 9/43 |
| E | 0 | 45 / 45 | 45/84 = 0.54 |
| F | 3 (2/0/1) | 27 / 10 | 27/81 |

- **Pairwise tests:** B vs A0 is 4:4 (p=1), D vs A0 is 7:15 (p=0.134), F vs A1 is 3:0 (p=0.25). All match yours.
- **Threshold doesn't matter:** "stuck" at 10, 15 or 20 cm gives the same counts (only B goes from 49 to 50 at 10 cm). The errors are clearly two-humped.
- **Mismatch 1, a false success:** `E_-0.19_9` counts as a success at 6.9 s, during the pregrasp. The index finger knocked the object up for a moment; at the end it is not lifted (rise 0.0). E's real count is 44/84 on both success measures.
- **Mismatch 2, the effective sample size is about 28:**
  - Moving the stance ±1 cm shifts the reach start by only 3.4–3.8 mm (median).
  - 15 of the 17 episodes where A0 gets stuck are stuck at all three stances.
  - Episodes 0 and 10 start from the same state, with goals 3 mm apart.
- **Per-episode sign tests:**
  - B vs A0: 2:3 (p=1); D vs A0: 2:5 (p=0.45).
  - E vs A1 on 45 s success: 8:1 (p=0.039); on 16 s success: 12:0 (p=0.0005).

## 2. Four pitfalls

- **Benchmark choice.** All 28 episodes share one table: edge x=−0.902 m, top 0.785 m, slab 10 cm thick. Robot heading is −1.3° to −5.8°. The hand starts 3.7–10.1 cm behind the edge, at a height inside the slab. Every pregrasp is a top-down pose about 8 cm above the object, 16–21 cm above the tabletop. So "lift to pregrasp height" clears the edge automatically, and this benchmark cannot tell perception-based planning apart from a template.
- **Data leakage.**
  - Per episode, E uses only the start hand position and the heading (both from the robot's own body sense) and the perceived pregrasp that every group uses. It uses no table geometry.
  - At design level, though, the 8 cm and the two-leg shape were chosen after looking at cuRobo's routes on these same episodes. "No perception" is not "no scene knowledge".
  - F borrows the plan from an almost identical scene (episode 9 borrows from 10, which is the same as 0).
  - E0 below shows that even the 8 cm isn't needed.
- **Metric misuse.**
  - "Stuck" is defined and applied the same way in every group.
  - The 16 s success mixes in path duration. A1's successes fall at 15.4–18.3 s (median 16.12 s); E's latest is 15.52 s.
  - With a 17 s limit it would be A1 30 vs E 45.
  - Projection: given E's duration, all 33 A1 successes land within 16 s. Given A1's duration, only 19 of E's 44 do.
- **Post-hoc selection.**
  - By your pre-registered criteria, H1' holds (B=49 and D=41, both above 12) and H2' holds (E=0 and F=3 are within A1+6).
  - Claim 4 rests on secondary measures you marked "报告，不作判据" (reported, not a criterion), so raising it to a conclusion is post hoc.
  - Excluding episode 24 (no cuRobo route) does not favour E.

## 3. Implementation checks (no bug found)

- **Via files were applied:** the `avoid` field is null for A0, gives 7–10 points for A1/B/C/D/F and 2 for E. The goal matches within 0.28 cm.
- **Group files are what they claim:**
  - C's points equal A1's, D's rotations equal A1's, and D's points equal B's.
  - My independent rebuild of E matches the author's to 2e-16.
- **B/D speed reached the controller:**
  - The file speeds are 0.051–0.075 m/s.
  - B's pregrasp time minus A1's is −0.06 to +0.40 s; D's equals B's exactly.
  - A0 takes 5.46 s and B 8.6 s.
- **Same starting states:** for every (stance, episode), the first-tick hand position is identical across all 7 groups (0 mm difference), as are the walk and the goal.
- **Deterministic:** my rerun of A0 and E on episode 0, nominal stance, gives identical stages and trace.
- **Small imprecision, conclusion unaffected:**
  - Via files are built from the nominal-stance start; the other two stances differ by up to 28 mm (episode 4) and 79 mm (episode 26, −0.21).
  - During the 0.3 s hold before the reach, the hand moves about 16 mm, so E's real back-off is about 8±2 cm.
- **`vias_e1.py` math is correct:** rotation interpolation, path fractions, durations (constants 0.15 m/s and 0.6 s match the controller), heading from the pelvis x-axis, and the direction of F's rotation. Start-to-goal wrist angles are 78–114°, nowhere near the unstable 180°.

## 4. Falsification runs (4090 CPU, nominal stance −0.20, 60 episodes)

I generated the via files with `scripts/vias_review.py $C/step2 $C/attrib/via_B $C/review_attrib`. Runs used `scripts/run_review.sh <group> -0.20 <episodes>`, the same environment variables and bench command you gave. "Stuck episodes" means the 16 where A0 is stuck at nominal stance (episode 10 dropped as a duplicate of 0).

| Run | Episodes | Stuck | 45 s success | For comparison |
|---|---|---|---|---|
| Reproduce A0 and E, episode 0 | 2 | identical to originals | | |
| E0: start → straight up to pregrasp height → goal | 16 stuck episodes | 0/16 | | A0 16/16 |
| E0, all except 10 and 23 | 26 | 0/26 | 11/26 | same 26: E 14, A1 11, C 8 |
| E8nl: back off 8 cm → straight to goal | 16 stuck episodes | 11/16 | | A0 16, B 15 |
| Bslow: B at 0.03 m/s (B is ~0.07) | 16 stuck episodes | 12/16 | | B 15; vs B 3:0, p=0.25 |

- **On the 45 s success:** E0 vs A1 is 2:2, E0 vs E is 1:4 (p=0.38), and E vs A1 at nominal stance is 4:1 (p=0.38).
- **Untested files:** `via_E4` and `via_E12` (4 cm and 12 cm back-off) are built but were not run, because E0 already showed back-off distance doesn't matter.

## 5. Verdict per claim

1. **支持.** At A1's timing (about 2× slower), slicing the line into segments and slowing down don't help. Slowing to 0.03 m/s does free 3 of 16, so speed has a small effect at extreme slowness. Strongest alternative: B's segments lie on one straight line, so B really only tests timing. The stuck is a geometric jam (moving forward before rising high enough), which slowing barely relieves.
2. **支持 the main point, with two corrections:**
   - Attitude is not irrelevant. D fully frees episodes 7, 9, 11 and 16 (plus two stances of 27), and newly sticks 5 and 19.
   - The useful part of the position path is "rise first", not "back off": E0 is 0/16 and E8nl is 11/16. The "先撤后抬" (back off, then lift) explanation should drop "back off".

   Strongest alternative: D puts rotations planned for A1's curved path onto a straight line, so it tests a mismatched attitude schedule.
3. **支持, for this benchmark only.** The even simpler E0 also gets 0 stuck. Strongest alternative: it isn't that planning is unnecessary. Here the obstacle is always below the goal, so "first rise to goal height" works as a perfect built-in prior. F works for the same reason: it borrows plans from near-identical scenes. A benchmark with randomized obstacles would be needed to say anything about perception.
4. **证据不足.**
   - The numbers hold (45 corrected to 44).
   - The 16 s gap is mostly E being about 1 s shorter.
   - On 45 s success, the per-episode result is 8:1 (p=0.039) and the sibling template E0 doesn't reproduce it (11 vs 11 at nominal stance).
   - Strongest alternative: path duration, number of legs, or per-episode noise, not a better path shape.

Everything is in `<Mac 会话临时目录>/review_attrib/`:
- recount and check scripts: `scripts/recount.py`, `scripts/timing.py`, `scripts/why.py`, `scripts/impl.py`, `scripts/revsum.py`
- run scripts: `scripts/vias_review.py`, `scripts/run_review.sh`
- the original rows, copied down: `data/`
- my 60 rows, copied down: `rev/rows/`

On the remote machine, my 60 rows, logs and via files are in `/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_bench_20261119/curobo/review_attrib/`. I changed nothing else on either machine.
