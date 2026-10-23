# 从原子技能开始的抓取进化：记录与复现（2026-10-20 → 10-23）

这一页给要接着做、或者重跑这条线的开发者。2026-10-20 起，场景 RSI 的内环从一个**空的技能目录**开始——只有动作层的 8 个动作（走到某处、转身、侧移、姿态、伸手、合爪、松爪、保持）——让 Claude 自己写出抓取技能，任务族叫 `grasp_atomic`。四次运行把封存集从 0 做到 38 / 50。机制见[详解](../rsi_design.md)，通用命令见 [runbook](../rsi.md)，本页只写这条线特有的东西。

## 1. 任务族

| 项 | 内容 |
|---|---|
| 任务 | `held(X)`，单手拿起，10 个物体：橙子（厨房岛台深处）、咖啡罐和马克杯（餐桌）、洗洁精（28 cm 高，灶台）、遥控器和蜡烛（矮茶几）、木勺（平放在 0.93 m 的台面上）、小花瓶（展示柜下层）、大盆栽（展示柜上层，1.08 m）、狼摆件（角柜） |
| 起点 | 机器人出生在房间里任何能走到的地方、离物体 0.5 m 以上；物体在台面上随机摆放 |
| 三个池 | 诊断 30（种子 410000）、验收 40（510000）、封存 50（610000）：每个物体 3 / 4 / 5 个起点 |
| 种子 | `scripts/rsi/seeds/atomic`：只有一个 `__init__.py`，`plan_for` 不覆盖任何目标 |
| 文件 | `scripts/rsi/families/grasp_atomic.json`、`scripts/rsi/families/grasp_atomic_tasks.jsonl`；角色读的方案 `scripts/rsi/plans/grasp_atomic.md` |
| 走路 | 0.25 m/s（第一次运行是 0.5 m/s，见下） |

## 2. 四次运行

同一批起点：三次续跑都用 `replay.py setup` 从上一次的某一轮接着，池一个起点都不换，所以四次的封存集数字可以直接排在一起（第一次除外，它是 0.5 m/s）。

| 运行 | 沙盒（`$LOCOMANI_WORKSPACE/code/rsi-scene/` 下） | 日期 | 流程 | 轮次 | 被接受 | 封存集 / 50 | 费用、时长 |
|---|---|---|---|---|---|---|---|
| A | `grasp-atomic-a` | 10-20 | 0.5 m/s；1 个 Engineer；规则第 7 版 | 0–5 | 第 1、5 轮 | 0 → 18 → 24 | 57 USD，5 h |
| B0 | `ga-confirm-025-20261021` | 10-21 | 0.25 m/s；3 个 Engineer + Composer + 经验库（TODO 32.12）；规则第 10 版 | 0–5 | 第 1 轮 | 0 → 31 | 93 USD，5 h |
| B | `ga-cont-b-20261022` | 10-22 | 从 B0 第 6 轮接着；放宽的验收门、候补池、物体专属流程（TODO 33.8、33.13、33.14）；规则第 14 版 | 6–9 | 第 6、9 轮 | 31 → 35 → 36 | 约 98 USD，5 h |
| C | `ga-cont-c-20261022` | 10-22 | 从 B 第 10 轮接着；失败类型、`stage()`、强制构建、可抓性（TODO 33.15–33.18）；停止线 90%；规则第 15 版 | 10–12 | 第 11 轮 | 36 → 38 | 约 73 USD，4 h |

几件要知道的事：
- B0 第 1 轮写出的那一版就到 31 / 50，之后四轮的候选都没过当时的验收门（多赢 5、少赢 1 的 p 是 0.11，门要 p ≤ 0.05）。这是 10-22 把门放宽成“多赢 − 少赢 ≥ 3 且少赢 ≤ 多赢 / 3”的直接原因（TODO 33.8）。B 第 6 轮接受的就是 B0 第 5 轮被旧门拒掉的那个候选，原样再送。
- C 停在“这一次运行的验收门调用满 12 次”。10-23 在 C 的沙盒上把诊断池重抽了一次（现任 30 / 30 全成，角色看不到任何失败；换进失败起点到 15 / 30，`loop/redraw_20261023.log`），没有续跑。
- 费用是账本里累计的角色调用费；B、C 的账本带着复制来的历史，这里减掉了。

## 3. 被接受的四版各改了什么

每版技能在运行沙盒的 `outputs/scene_rsi_20261016/grasp_atomic/incumbents/<哈希>/skills/`；最终版的快照在 `scripts/rsi/seeds/grasp_c11`（约 2,000 行），是 `grasp_prep` 任务族的种子。代码是 Claude 写的，下面是人的转述。

| 轮 | 哈希 | 改动 | 验收池 / 40 |
|---|---|---|---|
| 1（B0） | `971864dc7013` | 从空目录写出 `pick.py`：站位选在台面向外法线 30° 以内的射线上，右手水平侧抓；`plan_for` 把单物体的 `held(X)` 交给它。三份草稿的核心一样、分不出高下，Composer 选了第 2 份 | 0 → 21 |
| 6（B） | `49632436cd0b` | B0 第 3、5 轮重开搁置时写的两样：`rim.py`，开口容器（马克杯）隔着一面壁沿口捏；`doorway.py`，出生在别的房间、地图上没有可达站位时先走到门口。旧门拒过（多赢 5 少赢 1），新门接受 | 21 → 25 |
| 9（B） | `011262a9020f` | `clutter.py`：高台面（0.895 m 的岛台）上侧抓时前臂下方有固定杂物，先改手的俯仰；`recipes/tall_narrow.py`：洗洁精这类窄边只比爪口小不到 1 cm 的高物体，闭环入手代替 IK 入手。第 8 轮的候选（多赢 2，进候补池）加第 9 轮的草稿合成 | 25 → 29 |
| 11（C） | `5cba2fa49d31` | `pick.py`：截面放不进爪口时把抓点抬高（`raise_fit`）；分段抬高地走近；俯捏之后检查有没有抬起来，没有就松开重夹一次 | 29 → 33 |

## 4. 现任还败在哪（10-23 的分析）

封存 50 + 验收 40 共 90 局：现任 71 成、19 败（第 1 版是 52 成、38 败）。逐局归类在 `outputs/scene_rsi_20261016/failures_20261023/analysis.md`：

- 手进入时把物体碰倒或推走：7 局（盆栽在架子上被推下去，咖啡罐、花瓶先倒再夹、又掉）；
- 木勺一个物体 7 局（平放、1 cm 高、台面 0.93 m：俯捏的手送不到，或捏到了抬起时滑脱）；
- 走路 5 局（3 局出生点贴着 25 cm 高的脚凳又面朝它，起步 6–8 秒就倒，四个版本一局不差；2 局停点差 5–9 cm，技能不重走）；
- 物体比爪口宽 2 局，侧抓够不到 1 局。

19 局里有 12–13 局是“参考位姿算得出来、控制器没把手送到”：逆运动学残差中位数 0.69 cm，实测手差 7.5 cm。这是 10-23 做身体模型、站位抓法挑选器和准备动作层的原因（TODO 33.1.4、33.12、33.5，见第 6 节）。

## 5. 怎么复现

前提：[runbook](../rsi.md) 第 0–2 节做完（环境变量、复现材料、场景），Claude CLI 装好。`$PY`、`$S` 同 runbook。

### 5.1 从空种子跑一次（相当于 B0）

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/ga-repro
python3 scripts/rsi/sandbox.py $S grasp_atomic
cd $S && RSI_MAX_WALL_H=5 nohup setsid $PY scripts/rsi/loop.py run grasp_atomic > loop.log 2>&1 &
```

做什么：`sandbox.py` 按任务族文件的 `sandbox` 块建沙盒——技能目录换成 `scripts/rsi/seeds/atomic`，只保留块里列出的驱动文件，经验库去掉 grasp_atomic 自己的条目，抽三个池。`loop.py run` 默认就是 3 个 Engineer + Composer。墙钟上限 5 小时是这四次运行的设置（默认 6）。
得到什么：账本 `$S/outputs/scene_rsi_20261016/grasp_atomic/loop/ledger.jsonl`，`cd $S && $PY scripts/rsi/loop.py status grasp_atomic` 看进度。原来第 1 轮就有一版被接受，验收池 0 → 21 / 40；大模型写代码每次不同，复现的是量级。

### 5.2 接着上一次跑（相当于 B、C）

```bash
python3 scripts/rsi/replay.py setup $LOCOMANI_WORKSPACE/code/rsi-scene/ga-repro grasp_atomic 6 $LOCOMANI_WORKSPACE/code/rsi-scene/ga-repro-2
cd $LOCOMANI_WORKSPACE/code/rsi-scene/ga-repro-2 && RSI_MAX_WALL_H=5 nohup setsid $PY scripts/rsi/loop.py run grasp_atomic --stop-rate 0.9 > loop.log 2>&1 &
```

做什么：第一条用当前提交建一个新沙盒，放进上一次运行到第 6 轮开始时的全部状态（同一批池、账本、现任、搁置），现任的诊断局和验收局在当前代码下重跑；流程改动（新规则、新工具）从这一轮起生效，起点不变。`--stop-rate 0.9`：现任在验收池上已经过了 70%，不加它一开跑就停；续跑沿用记下的值。现任在诊断池上失败少于 10 局时，驱动程序先重抽诊断池（`loop.py redraw grasp_atomic` 可以单独做）。
得到什么：新沙盒里 `iter_NN/` 从第 6 轮接着编号，更早的轮次照搬过来。

### 5.3 封存集

```bash
cd $S && $PY scripts/rsi/sealed.py grasp_atomic --all; cd -
```

做什么：在封存集 50 局上跑种子和每一版被接受的现任（不加 `--all` 只跑种子和最终版）。
得到什么：`$S/outputs/scene_rsi_20261016_hidden/grasp_atomic/sealed/summary.json` 的 `versions` 列出每版的成功数。

### 5.4 看过程

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 $PY scripts/rsi/report/evolution_media.py --sandbox $S --family grasp_atomic --out outputs/ga_repro_media
```

做什么：每次被接受的改动取一个它从败变成的诊断起点，改前改后并排成一段视频，加一张“每个物体 × 每个版本”的总览图。
得到什么：`outputs/ga_repro_media/step_NN.mp4`、`overview.png`，网页用的预览每段不超过 500 KB。

### 5.5 不写回

从空种子开始的任务族写不回主仓库的技能目录（新文件和目录里的同名文件冲突，[详解](../rsi_design.md)第 10 节）。这条线的产物留在沙盒里；要拿它当下一条线的种子，像 `grasp_prep` 一样在任务族文件里指一个快照目录（`scripts/rsi/seeds/grasp_c11`）。

## 6. 已经做好、还没在运行里用过的

10-23 做完，都没在一次运行里一起用过；下一次运行是它们的第一次检验（细节在 TODO 33、37 和[详解](../rsi_design.md)第 5 节）：

| 项 | 是什么 | 怎么用 |
|---|---|---|
| 内环提速（TODO 37.1–37.4） | 角色的实验命令 `scripts/rsi/try.py`、Manager 读压缩的历史和机制表、Engineer 软截止、验收门第二枪 | 下一次 `loop.py run` 默认开 |
| 诊断池重抽（33.23） | 现任把诊断池全做成时换进失败起点 | 自动；C 的沙盒已重抽、可直接续跑 |
| 身体模型预测器（33.1.4） | 一个伸手目标控制器送不送得到、差多少 | 角色经 `scripts/rsi/body.py` 查；技能还没接 |
| 站位和抓法挑选器（33.12） | 站位、手、抓法、姿态一起打分，兜底顺序站着 → 降骨盆 → 准备动作 | 角色经 `scripts/rsi/choose.py` 查；技能还没接 |
| 准备动作层和 `grasp_prep`（33.5） | 转、挪邻居的目标挑选器；三个“原地抓不到”的任务，种子是 C 的最终版 | `python3 scripts/rsi/sandbox.py $S grasp_prep`，再 `loop.py run grasp_prep --stop-rate 0.9` |

两条可选的下一次运行：续跑 C 的沙盒（看新工具能不能啃动那 19 个顽固起点），或者开 `grasp_prep`（让循环写准备动作）。都是用户的决定，到 10-23 都没开。

## 7. 记录在哪

- 沙盒：第 2 节的四个。每个里面 `outputs/scene_rsi_20261016/grasp_atomic/loop/`（账本、每轮的提示和回答）、`diag/`（逐局记录和 50 Hz 轨迹）、`incumbents/`（每版技能）；隐藏池在 `outputs/scene_rsi_20261016_hidden/grasp_atomic/`。
- 当时的启动脚本：`outputs/rsi_speed_20261021/confirm/launch.sh`（B0）、`outputs/scene_rsi_20261016/grasp_atomic_b/launch.sh`、`outputs/scene_rsi_20261016/grasp_atomic_c/launch.sh`。
- 分析：`outputs/scene_rsi_20261016/failures_20261023/`（失败归类、TODO 总览）、`speedup_20261023/`（一轮 79 分钟的分解）、`body/`（身体模型的表和拟合报告）、`choose_20261023/`（挑选器的离线核对）。
- 网页报告（claude.ai 上私密，分享要在页面里给权限）：进化过程 https://claude.ai/artifact/Cm4CZTFfFp8tvid6GxyKKp （A 和 B0，10-21）；流程改了什么、成功率为什么升 https://claude.ai/artifact/RewrUG8Bas7pFtNTB12hSU （B 和 C，10-22，每个物体的流程图和视频）。媒体源在 `outputs/scene_rsi_20261016/grasp_atomic_media/`。
