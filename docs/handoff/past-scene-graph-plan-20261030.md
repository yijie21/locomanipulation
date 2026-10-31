# PAST：动作条件局部场景图记录器的方案（2026-10-30）

这一页记 2026-10-30 和用户逐条问答（Q1–Q26）定下的目标、决策和理由，任务清单在 [TODO 39](../../TODO.md)，决策摘要在 DECISIONS 2026-10-30。流程机制见[详解](../rsi_design.md)，命令见 [runbook](../rsi.md)。研究动机的原始文档是用户的 `~/docs/PAST_Action_Conditioned_Local_Scene_Graphs.md`（PAST：Privileged Action-conditioned Scene-graph Transitions），不在仓库里。开发在分支 `henry-dev` 上。

## 1. 目标

给场景 RSI 加一个**记录器**：每局运行时从仿真读特权信息（真值位姿、接触、支撑、持握），按动作组织成一条条局部场景图记录（一个步骤、一次重试一条），和现有的 `.json` / `.npz` 一起保存。Reviewer 读它来诊断失败、提假设；人也能打开看。

第一版只验证**单次运行内的诊断效率**（原文的 H1）：Reviewer 用更少的时间、费用和工具调用，提出命中率不更低的假设。跨任务检索（H2）、修复复用和条件修正（H3）放第二阶段（第 8 节）。

记录器是测量仪器：它只写不读，不影响任何动作决策；它在 Engineer 不能改的层；它不改变任何一局的结局（第 5 节的验证就是这一条）；它的侧文件不进验收门、哨兵和回归的任何判定。

## 2. 起点：已有的东西

记录能从现成的通道里拼出来，不用动控制器：

- **步骤日志**：`SkillContext.note(name, out, **extra)`（`src/locomani/tasks/scene/episode.py`）把每个步骤的 `step`、`ok`、`why`、`t0`、`t1` 和动作层返回的数值记进 `steps`；`stage()`（`stage.py`）给每次重试加 `check`（检查值）和 `attempt`。步骤名是技能自己起的字符串。
- **原语调用日志**：`Motor` 的每个原语（`walk_to`、`turn`、`sidestep`、`posture`、`reach`、`grip`、`release`、`hold`、`squeeze`）经 `_logged` 记进 `calls`：名字、起止时间、参数、结果（TODO 33.20）。嵌套调用算在外层里。
- **轨迹**：`Motor.tick()` 每拍（50 Hz）往 `trace` 记 `t`、`primitive`、`mode`、骨盆、关节、参考和实测手腕、夹爪指令、手指、每个追踪物体的位姿；`arrays()` 转成 NPZ。
- **接触**：动作层已经每拍遍历 `d.contact`（`between_fingers`、`gripping`、`touching`），但不记录，只用来判 `held`。
- **谓词**：`src/locomani/scene_discovery/predicates.py` 从状态算 `on`、`in`、`held`、`upright`、`lifted`、`near`、`in_room`；验收门和 Reviewer 用的是同一套。
- **场景文件** `scene_meta.json`：`surfaces`（`top`、`of_object`、`half_extent`、`yaw`、`approach_edges`）和 `objects`（`surface`、`box`、`container`、`free`、`semantic.movable`、`body`）。
- **写出**：`scripts/rsi/scene_round.py` 每局写 `<label>.json`，加 `--npz` 时写 `<label>.npz`；诊断池、验收池、封存集都经它。
- **Reviewer 的入口**：`scripts/rsi/digest.py` 生成摘要（`pick-repro` 第 1 轮 344 行），`prompts/reviewer.md` 第 4 条让它先读摘要、不够再开 `.json` / `.npz`。

## 3. 决策（Q1–Q26，2026-10-30）

| # | 问题 | 决定 |
|---|---|---|
| Q1 | 动作粒度 | `ctx.note()` 这一层的步骤，`stage()` 每次重试单独一条；原语调用是记录内的事件 |
| Q2 | 代码放哪层 | 冻结层：接触通道在 `motor.py` 轨迹，构造器在 `src/locomani/tasks/scene/past.py`；不是技能，也不是原子动作 |
| Q3 | 采集方式 | 运行时只多记接触通道；局结束后用纯函数从步骤日志、调用日志、轨迹和场景文件构造记录 |
| Q4 | 关系集合 | `predicates.py` 的谓词 + 原始接触 `touching` + 由接触推出的 `supported_by` |
| Q5 / Q10 | 范围 | 操作步骤的初始范围按第 4.4 节；走路、转身、姿态步骤也记录，范围是机器人、目标物体、站位足迹和路径上的障碍、碰到的刚体 |
| 补充（同日） | 实体、关系、属性按原文补全 | 实体按相关性选（邻居 0.30 m、站位足迹、走廊、上方净空 0.5 m、区域），不按接触选；关系五类（接触、支撑、持握、空间、相对位姿）加谓词，以角色为中心；属性加质心、速度、手臂关节、脚 |
| Q6 | 存放 | 每局一个侧文件 `<label>.past.jsonl`，三个池一律生成 |
| Q7 | 第一版范围 | 单次运行内的诊断：采集、构造、摘要新增、查看工具、提示词和规则 |
| Q8 / Q19 | 验证有用 | 不做 A/B 重放；开一次新运行和上一次运行对照（第 6 节） |
| Q9 / Q18 | 回填 | 构造器从第一天起接受没有接触通道的旧局（接触边 `unknown`）；回填脚本放第二阶段 |
| Q11 | 接触采样 | 每拍；只记一侧是追踪物体或机器人手臂 / 手部刚体、另一侧不是地面的对 |
| Q12 | 角色绑定 | 从步骤内的原语调用推断，不要求技能声明 |
| Q13 | 期待的转移 | 按原语的小表给 `expected`，和 `observed` 并列 |
| Q14 | 摘要新增 | 每个失败局的关系翻转时间线 + 每个任务第一个不期待转移的计数和成功局对照 |
| Q15 | 查看工具 | `scripts/rsi/past.py show` / `compare` |
| Q16 | 谁能看 | 文件所有角色读得到；第一版只在 `reviewer.md` 点名；规则第 23 版 |
| Q17 | 不变性验证 | `pick-repro` 现任诊断池 50 局重跑逐局全同 + 单元测试 |
| Q20 | 指标 | 主：探针"已验证"比例、净救回局数；次：Reviewer 用时、费用、工具调用数、打开 NPZ 的次数 |
| Q21 | 文档 | 本页、TODO 39、DECISIONS、`rsi_design.md` 第 5 节一段 |
| Q22 | 提交 | 五步，每步 `make check` 后按名字暂存提交到 `henry-dev` |
| Q23 | 隐藏池 | 侧文件不进任何判定 |
| Q24 | 提示词读法 | 摘要 → `.past.jsonl` → NPZ 最后手段 |
| Q25 | 给人看 | 第一版文本（`show`）；`html` 图形页第二阶段 |
| Q26 | 新运行 | `pick_single`，新沙盒 `pick-past-a`，从 `pick-repro` 的种子版起，和它的 7 轮对照 |

## 4. 设计

### 4.1 分层与归属

| 部件 | 位置 | 谁能改 | 职责 |
|---|---|---|---|
| 接触通道 | `src/locomani/backends/scalebfm/motor.py` | 人（冻结层的测量字段，不是开关） | 每拍记接触对进 `trace`，`arrays()` 导出 |
| 构造器 | `src/locomani/tasks/scene/past.py` | 人 | `build(rec, arrays, meta, task) -> list[dict]`，纯函数 |
| 写出 | `scripts/rsi/scene_round.py` | 人 | 每局写 `<label>.past.jsonl` |
| 摘要 | `scripts/rsi/digest.py` | 人 | 新增两节 |
| 查看 | `scripts/rsi/past.py` | 人 | `show`、`compare` |
| 提示词、规则 | `scripts/rsi/prompts/reviewer.md`、`scripts/rsi/rules.md` | 人（一次插手） | 读法和规则第 23 版 |

规则第 1 条说动作层"只能加默认关闭的开关"，针对的是会改变行为的东西；接触通道只往 `trace` 里多记几个整数，不碰仿真，和 `_logged` 的调用日志（TODO 33.20）是同一类。第 5 节的 50 局逐局全同是它不改行为的证据。

### 4.2 接触通道

`Motor.tick()` 里 `trace.append(...)` 的那一行多一个字段 `contacts`：这一拍里满足条件的刚体对 `(a, b)` 列表，`a < b`，刚体用 MuJoCo 的 body id。条件：一侧是追踪物体（`self.track` 的值）或机器人的手臂 / 手部刚体（两侧的 `*_shoulder_*`、`*_elbow_*`、`*_wrist_*`、`*_dex1_*` 链接，按名字前缀在 `__init__` 里解析一次成 id 集合），另一侧不是地面（`world` body 0 和名为 `floor` 的 geom 所属刚体）。遍历 `d.contact` 的代码 `between_fingers` 已经有，复用同样的循环。

`arrays()` 导出三列整数 `contact_k`（拍序号）、`contact_a`、`contact_b`（`int32`，拍内无接触时没有行），加一张刚体名表 `bodies`（字符串数组，下标是 body id）。50 Hz × 300 s 上限 × 每拍几对，最坏几万行，压缩后远小于现有的关节轨迹。`fingers`、`objects` 等现有字段不变，NPZ 的旧读者不受影响。

### 4.3 记录格式（`schema: "PAST-locomani-0.1"`）

一局一个 `.past.jsonl`，每行一条记录，按 `t0` 排序。字段沿原文的结构，名字按这个仓库的习惯（米、秒、世界系、四元数 `wxyz`——MuJoCo 的顺序，和 NPZ 里一致）：

```json
{
  "schema": "PAST-locomani-0.1",
  "record_id": "canister_pickup_s100000/03.1",
  "label": "canister_pickup_s100000", "task": "canister_pickup",
  "step_index": 3, "attempt": 1, "step": "pregrasp",
  "kind": "reach",
  "skills": "a031ceb0d3de", "set": {"PREGRASP_BACK_M": 0.14},
  "predicate_version": "predicates@<git blob hash>",
  "action": {
    "t0": 12.4, "t1": 15.1, "ok": true, "why": "",
    "calls": [{"name": "reach", "side": "right", "t0": 12.4, "t1": 15.1, "args": {...}, "ok": true}],
    "check": {"jaw_err_cm": 1.3}
  },
  "roles": {"target": "Canister_1", "effector": "right_hand", "support": "ApartmentEnv__0", "goal_surface": null, "container": null,
            "stance": "region:stance", "corridor": "region:corridor", "pregrasp": "region:pregrasp_target"},
  "scope": {"initial": ["Canister_1", "right_hand", "left_hand", "robot", "ApartmentEnv__0", "ApartmentEnv", "Mug_2", "Shelf_3",
                        "region:stance", "region:corridor", "region:pregrasp_target"],
            "initial_why": {"Mug_2": "neighbour 0.18 m", "Shelf_3": "above target 0.31 m", "ApartmentEnv": "furniture of support"},
            "added": [{"id": "BarStool_1", "t": 13.7, "why": "touched by robot"}]},
  "entities": {
    "Canister_1": {"type": "object", "body": "Canister_1", "mass_kg": 0.8, "half_extent_m": [0.05, 0.05, 0.11], "com_m": [0, 0, 0.01],
                   "movable": true, "container": false, "upright_axis": [0, 0, 1]},
    "Mug_2": {"type": "object", "...": "..."},
    "right_hand": {"type": "effector", "bodies": ["right_wrist_yaw_link", "right_dex1_finger_1_link", "right_dex1_finger_2_link"], "jaw_max_m": 0.095},
    "ApartmentEnv__0": {"type": "surface", "of": "ApartmentEnv", "top_m": 0.4182, "half_extent_m": [0.335, 0.18], "yaw": -1.5708,
                        "approach_edges": ["-x"], "fixed": true},
    "ApartmentEnv": {"type": "furniture", "box_m": {"half_extent": [...], "centre": [...]}, "fixed": true},
    "Shelf_3": {"type": "surface", "...": "..."},
    "robot": {"type": "robot"},
    "region:stance": {"type": "region", "shape": "disc", "centre": [..], "radius_m": 0.35},
    "region:corridor": {"type": "region", "shape": "segment", "a": [..], "b": [..], "half_width_m": 0.08},
    "region:pregrasp_target": {"type": "region", "shape": "point", "p": [..], "wrist_att": "level"}
  },
  "G_before": {"t": 12.4,
    "states": {
      "Canister_1": {"p": [..], "q": [..], "v_mps": 0.0, "w_radps": 0.0, "bottom_z_m": 0.418, "yaw": 0.3},
      "Mug_2": {"p": [..], "q": [..], "v_mps": 0.0},
      "right_hand": {"p": [..], "q": [..], "jaw_m": 0.09, "att": "level", "arm_q": [7 个关节角], "ref_err_cm": 0.0},
      "left_hand": {"p": [..], "q": [..], "jaw_m": 0.09},
      "robot": {"pelvis_p": [..], "pelvis_z_m": 0.74, "yaw": 1.2, "feet_p": [[..], [..]], "fell": false, "primitive": "reach", "mode": 2}
    },
    "edges": [
      {"s": "Canister_1", "r": "supported_by", "o": "ApartmentEnv__0", "v": true, "src": "derived"},
      {"s": "Canister_1", "r": "touching", "o": "right_hand", "v": false, "src": "observed"},
      {"s": "Canister_1", "r": "held", "o": "right_hand", "v": false, "src": "predicate"},
      {"s": "Canister_1", "r": "upright", "o": null, "v": true, "src": "predicate"},
      {"s": "Shelf_3", "r": "above", "o": "Canister_1", "v": true, "gap_m": 0.31, "src": "derived"},
      {"s": "Mug_2", "r": "beside", "o": "Canister_1", "v": true, "gap_m": 0.07, "side": "+y", "src": "derived"},
      {"s": "Canister_1", "r": "near_edge", "o": "ApartmentEnv__0", "v": true, "dist_m": 0.12, "edge": "-x", "src": "derived"},
      {"s": "BarStool_1", "r": "blocking", "o": "region:stance", "v": null, "src": "unknown"},
      {"s": "Mug_2", "r": "blocking", "o": "region:corridor", "v": false, "src": "derived"},
      {"s": "robot", "r": "dist", "o": "Canister_1", "v": 0.62, "bearing_deg": 8.0, "src": "derived"},
      {"s": "robot", "r": "facing", "o": "Canister_1", "v": 8.0, "src": "derived"},
      {"s": "right_hand", "r": "reach_vec", "o": "region:pregrasp_target", "v": [0.48, 0.21, 0.52], "src": "derived"},
      {"s": "right_hand", "r": "approach", "o": "Canister_1", "v": "-x", "wrist_att": "level", "src": "derived"},
      {"s": "Canister_1", "r": "rel_pose", "o": "right_hand", "v": null, "src": "derived"}
    ]},
  "events": [{"t": 13.7, "edge": ["BarStool_1", "touching", "robot"], "from": false, "to": true, "src": "observed"},
             {"t": 13.9, "edge": ["Canister_1", "moved", null], "from": false, "to": true, "cm": 3.2, "src": "derived"}],
  "G_after": {"t": 15.1, "states": {...}, "edges": [...]},
  "expected": [{"edge": ["right_hand", "at", "region:pregrasp_target"], "to": true, "tol_cm": 3.0},
               {"edge": ["Canister_1", "moved", null], "to": false},
               {"edge": ["*", "blocking", "region:corridor"], "to": false}],
  "discrepancies": [],
  "outcome": {"step_ok": true, "round_ok": false, "round_why": "...", "failure_class": "knocked"},
  "evidence": {"npz": "canister_pickup_s100000.npz", "k0": 620, "k1": 755, "json_step": 3}
}
```

约定：
- **三种来源分开**（原文"观测事实、派生谓词、因果假设分开"）：`src` 为 `observed`（接触、位姿）、`predicate`（`predicates.py` 算的）或 `derived`（支撑、空间、相对位姿、`moved`、`slipped`，都是从位姿和包围盒算的几何量）。记录里没有因果判断；`discrepancies` 只是 `expected` 和 `G_after` 的差。
- **未知不是假**：没有接触通道的旧局，所有 `touching` 边和依赖它的 `supported_by`、`blocking`（按接触判的部分）记 `"v": null, "src": "unknown"`；几何边照算。
- **边以角色为中心**：只记一端是 `target`、`effector`、`robot` 或某个 `region` 的对，不记全对全；邻居之间的关系不记。每条记录一两百条边、一局几十 KB。
- **实体属性**（原文的 geometry、mass / CoM、pose / velocity、joint states）：物体有质量、半尺寸、质心（`meta.objects[*].box.centre`）、竖直轴，状态里有位姿、线速度和角速度（相邻两拍差分）、底面高度、偏航；手有组成刚体和夹爪最大开口，状态里有位姿、开口、腕姿态档（水平 / 俯 / 竖直，和 `body.reach()` 同一分法）、肩 3 + 肘 1 + 腕 3 的关节角、相对参考的误差；机器人有骨盆位姿和高度、偏航、两脚位置、是否摔倒、当前原语和模式；面有顶高、半尺寸、偏航、可接近的边、所属家具；区域有形状和参数。
- **一步一条、重试各一条**：`record_id` 是 `<label>/<step_index>.<attempt>`；`stage()` 的多次尝试在步骤日志里是同名多行，按 `attempt` 分开。
- **`kind`** 是步骤内主导原语的种类（`reach`、`grip`、`release`、`walk`、`turn`、`sidestep`、`posture`、`hold`、`squeeze`、`mixed`、`none`），由第 4.5 节的规则定；步骤名 `step` 原样保留但不参与任何推断。
- **`expected`** 来自第 4.5 节的表；查不到的 `kind` 为空列表。
- **`evidence`** 指向 NPZ 和拍区间，Reviewer 需要原始数据时直接按区间读。

### 4.4 范围规则

范围按**相关性**选，不按接触选（用户 2026-10-30：按接触加入会漏掉占住站位的凳子、物体上方的隔板、挡住走廊的邻居——报告里最常见的几类失败原因，失败前都没被碰到）。初始范围只用步骤开始时能知道的，全部从 `scene_meta` 的包围盒和当前状态算；每个实体记 `initial_why`。

操作步骤（`kind` 不是 `walk` / `turn` / `sidestep` / `posture` / `hold`）：
- **角色实体**：任务物体（`task.objects`）、两只手、`robot`、每个任务物体当前的支撑面（`G_before` 时刻按第 4.6 节的 `supported_by` 推出，推不出就用 `meta.objects[k].surface`）和它所属的家具（`meta.surfaces[*].of_object`）、目标谓词点名的台面或容器（`on(X, S)` 的 S、`in(X, C)` 的 C）。
- **邻居物体**：和目标物体水平距离 ≤ NEIGHBOUR_M（0.30 m）的可动物体；包围盒和接近走廊相交的可动物体。
- **家具 / 障碍**：包围盒和站位足迹相交的家具或碰撞体；目标正上方 OVERHANG_M（0.5 m）内有东西的面或家具（隔板、上层台面）。
- **区域**：站位足迹（`walk_to` 的目标点，半径 STANCE_M 0.35 m 的圆盘）、接近走廊（预抓点到目标的线段向两侧各扩 CORRIDOR_HALF_M 0.08 m）、预抓目标点（`reach` 的目标和腕姿态）、放置区（目标台面的放置面或容器的内腔）。这一步没有对应原语时区域不出现。
- **执行中加入**：这一步里和机器人任何部位或任务物体发生接触、又不在初始范围里的刚体，带首次接触时间和原因；地面不加。

走路类步骤：`robot`、任务物体、站位足迹区域、和足迹或机器人到站位的路径（线段扩 0.35 m）相交的家具 / 碰撞体 / 可动物体，执行中碰到的刚体；`expected` 是停位误差在阈值内（`stop_err_cm`，动作层本来就返回）和足迹里没有 `blocking`。

四个常数（NEIGHBOUR_M、OVERHANG_M、STANCE_M、CORRIDOR_HALF_M）是按夹爪和手臂尺寸估的，写在 `past.py` 顶部，记录的 `scope.selector` 带它们的值，改了能分辨。

### 4.4.1 关系（边）

五类，和原文的 spatial、contact、support、holding、relative-pose 对应。只记一端是角色实体或区域的对（第 4.3 节"以角色为中心"）：

| 类 | 边 | 值 | 怎么算 |
|---|---|---|---|
| 接触 | `touching(a, b)` | 真 / 假 / 未知 | 接触通道 |
| 支撑 | `supported_by(X, S)`、`in(X, C)` | 真 / 假 | 接触 + X 底面在 S 顶 ±2 cm + 未被持；容器内腔包含 |
| 持握 | `held(X, hand)`、`between_fingers(X, hand)` | 真 / 假 | 两指都接触（`predicates._held` 的定义）；在两指之间 |
| 空间 | `above(S, X)` 带 `gap_m`（上方净空）；`beside(N, X)` 带 `gap_m`、`side`（邻居在目标的哪一侧、缝多宽）；`near_edge(X, S)` 带 `dist_m`、`edge`（离哪条台沿多远）；`blocking(B, region)` 带 `depth_m`（刚体和区域相交多深）；`dist(robot, X)` 带 `bearing_deg` | 真 / 假 + 数值 | 包围盒几何（物体用 `box` 的半尺寸和当前位姿，面用 `half_extent` 和 `yaw`，机器人用骨盆位置） |
| 相对位姿 | `rel_pose(X in hand)`（物体在夹爪坐标系的位置和姿态，抓住后才有值）；`reach_vec(hand → 目标点 in pelvis)`（前伸、外侧、高度——和 `body.reach()` 的表同一坐标）；`approach(hand → X)` 带 `wrist_att`（从物体的哪一面接近、腕姿态档）；`facing(robot → X)`（机器人朝向和目标方位的夹角） | 数值 | 位姿换算 |
| 谓词 | `upright(X)`、`lifted(X, …)`、`near(X, Y)` | 真 / 假 | `predicates.py` 原样 |

空间边里 `blocking` 和 `above` 的真值有一半靠接触通道（刚体是不是真顶住了），没有接触通道时按几何给值、`src` 标 `approx`。

### 4.5 角色推断与期待表

角色从步骤内的 `calls` 推：

| 步骤内的原语 | `kind` | 角色 | `expected` |
|---|---|---|---|
| 只有 `walk_to`（可含嵌套 `turn`） | `walk` | `robot` → 站位 `region` | 机器人到站位 ≤ 停位阈值（取 `walk_to` 返回的 `stop_err_cm` 和参数里的容差）；站位足迹里没有 `blocking` |
| `turn` / `sidestep` / `posture` / `hold` 单独 | 同名 | `robot` | 无 |
| `reach(side)`，夹爪开着 | `reach` | `effector` = 那只手，`target` = 离目标点最近的任务物体 | 手到目标 ≤ 3 cm（和 `pred_p_miss` 的 3 cm 一致）；`target` 不动（`moved` 保持假）；走廊里没有 `blocking`；邻居不动 |
| `grip(side)` | `grip` | `effector`、`target` = 两指之间的物体，没有则离夹爪最近的任务物体 | `held(target, effector)` 变真 |
| `reach(side)`，夹爪闭着且 `held` 为真 | `reach` | 同上，另有 `carried` = 被持物体 | `held` 保持真；若目标高于起点 2 cm 以上，`supported_by` 变假（抬起） |
| `release(side)` | `release` | `effector`、`target` = 松手前被持的物体 | `held` 变假；若目标谓词有 `on(X, S)` 或 `in(X, C)` 且这是 `X`，`supported_by S` / `in C` 变真 |
| `squeeze`、`grip("both")` | `squeeze` / `grip` | `effector` = `both_hands` | `held` 两手都真 |
| 多种原语混合 | `mixed` | 按出现的原语合并角色 | 合并各自的期待 |
| 没有原语（纯规划、`settle`） | `none` | 只有任务物体 | 无 |

这张表按原语、不按物体，也不读步骤名。技能里如果出现这张表没有的组合，记录照样生成，只是 `expected` 为空、`kind` 为 `mixed`。

### 4.6 构造器算法（`past.build`）

输入：这一局的 `rec`（`scene_round.py` 写 `.json` 之前的字典，含 `steps`、`calls`、`ok`、`why`、`failure_class`）、`arrays`（`Motor.arrays()` 的输出，含接触通道）、`meta`（场景文件）、`task`（目标和物体）。输出：记录列表。步骤：

1. 把 `steps` 按 `(t0, t1)` 切成区间；同名多行按 `attempt` 分开；`calls` 按时间归到区间（嵌套的只算外层）。
2. 每个区间按第 4.5 节定 `kind`、角色、`expected`。
3. 按第 4.4 节在 `t0` 那一拍的状态上选初始范围（邻居、障碍、上方净空、区域都用包围盒算，记 `initial_why`）；扫这一区间的接触通道，加入新碰到的刚体。
4. 在 `t0` 和 `t1` 那两拍算 `G_before` / `G_after`：实体状态来自 NPZ（物体位姿和差分速度、手腕、手臂关节、夹爪、骨盆、脚）；`touching` 来自接触通道；谓词调用 `predicates.compile_goal` 对范围内的实体对逐个算（`held` 直接复用 `gripping` 的定义：两指都接触）；`supported_by(X, S)` 为真当且仅当 X 和 S 的刚体接触、X 的底面在 S 的 `top` 上下 2 cm 内、且 X 不被任何手持；空间边和相对位姿边按第 4.4.1 节的几何算。
5. 区间内每隔 1 拍重算 `touching`、每 5 拍（0.1 s，和 `JUDGE_DT` 一样）重算谓词、支撑边和 `blocking`，边的真值一变就记一条 `events`（数值边只在 `G_before` / `G_after` 记，不记每次变化）；物体位移超过 2 cm 记 `moved`（邻居也算），被持物体和手腕距离变化超过 1 cm 记 `slipped`（和 `digest.py` 现有的 knocked / slip 口径一致）。
6. `discrepancies` = `expected` 里在 `G_after` 没成立的项。
7. 填 `outcome`、`evidence`，排序，返回。

没有接触通道时第 4 步和第 5 步的接触边与 `supported_by` 记 `unknown`，`held` 退化为从 `fingers` 和物体位姿近似（两指开口小于物体宽度且物体在两指之间 1 cm 内），`src` 标 `approx`。

构造器不 import 动作层和 MuJoCo，只读数组和字典，单元测试用手写的小数组就能覆盖。

### 4.7 写出

`scene_round.py` 在写 `<label>.json` 之后、同一个 `try` 里调用 `past.build` 写 `<label>.past.jsonl`（先写 `.tmp` 再改名，和 `.json` 一样）。构造器抛异常时写一行 `{"schema": ..., "error": "..."}`，不影响局的记录（和失败分类器的处理一样：仪器的 bug 不花一局）。不加 `--npz` 时轨迹仍在内存里，照样构造；诊断池、验收池、封存集都生成。

`diag_pool.py` 把退役起点改名为 `_retired_<label>.*` 时把 `.past.jsonl` 一起改。`batch.outcomes` 只读 `.json`，不受影响。

### 4.8 摘要新增（`digest.py`）

在现有"按失败类型"和"按任务"两节之间加一节，总共不超过 115 行（现有 344 行的三分之一）：

- **每个任务**：失败局里第一个不期待的转移（`discrepancies` 非空的第一条记录的 `kind` + 边）按出现次数排，每条后面写成功局在同一 `kind` 步骤上的对照（成功局里这条边多数怎样）。
- **每个失败局一行**：从第一个操作步骤起的关系翻转时间线，`t  step.kind  边 从→到 [期待 ✓/✗]`，只列不期待的和最后三条，超过 6 条折叠成"… 还有 N 条，`past.py show <label>`"。

没有 `.past.jsonl` 的目录（旧沙盒）这一节不出现。

### 4.9 查看工具（`scripts/rsi/past.py`）

- `past.py show <diag 目录>/<label>`：按步骤打印 `G_before` → 事件 → `G_after`，边一行一条：`Canister_1 —supported_by→ ApartmentEnv__0  true→false  (期待 false ✓)`；`--step N` 只看一步；`--json` 原样输出。
- `past.py compare <失败局> <成功局>`：按 `kind` 序列对齐两局（最长公共子序列），并排打印每对步骤的边差异和事件，不同处标出；两局任务不同时也能比（跨物体看同一种动作），这是第二阶段检索的雏形。
- 第二阶段：`html`（每步前后两张小图排一页）、`query`（跨运行检索）。

### 4.10 提示词与规则

- `reviewer.md` 第 4 条之后的"可读数据"加一项：`.past.jsonl` 是什么、`show` / `compare` 怎么用；读法顺序改成"摘要 → `.past.jsonl` → `.npz` 最后手段"，并说明记录里的 `expected` 是按原语的固定表、不是因果判断。
- `rules.md` 第 23 版加一条（**code**）：每局的 `.past.jsonl` 由运行器生成，技能不能改它，也不许读它来做决策（技能目录里出现对 `.past.jsonl` 或 `past.py` 的引用，草稿作废——和第 30 条"不许写物体名"同一机制）。
- 不改 `manager.md`、`engineer.md`、`composer.md`（Q16）。

## 5. 验证

- **单元测试**（`tests/test_scene_past.py`）：手写 3–4 步的小轨迹和步骤日志，覆盖 `kind` 推断、范围加入、`supported_by` 推导、事件生成、`unknown` 退化、异常写错误行；`tests/test_scalebfm_motor.py` 加接触通道的导出形状和过滤条件。
- **不变性**（Q17）：`pick-repro` 现任（`loop/` 里最新被接受的版本）在它的诊断池 50 局上开着接触通道重跑到一个新目录，和存档的 `.json` 逐局比 `ok`、`why`、`fell`、`dropped`、`t_end`，要求全同；同时也比 NPZ 里 `robot` 和 `obj_*` 的最大差，记进本页第 7 节。这是验收门哨兵（现任 5 局逐局复现）的同一标准放大到 50 局。
- **`make check`** 每步提交前过。

## 6. 新运行与指标

不做 A/B 重放（Q19）。记录器和提示词就位后，新沙盒 `pick-past-a` 跑 `pick_single`（`sandbox.py ~/workspace/code/rsi-scene/pick-past-a pick_single`，然后 `loop.py run`），现任从 `pick-repro` 的种子版起，规则第 23 版，别的开关和 `pick-repro` 一样（候选数、步速、停止线从它的 `meta.json` 抄）。

对照是 `pick-repro` 的 7 轮（同机、同场景、同诊断种子）。**这是非配对比较**：大模型写的代码和假设每次都不同，两次运行的差里有运行间噪声（TODO 32.11 的重放就是为了去掉它）。结论只能说方向，不能说显著；要下结论还得做重放 A/B，留在第二阶段。

指标（Q20），都从账本和 `iter_NN/` 里算，脚本放 `scripts/rsi/report/past_monitor.py`：
- 主：每轮假设里探针判"已验证"的比例；探针净救回局数（`probe.json`）。
- 次：Reviewer 每轮的用时、费用（账本 `review` 行）、工具调用数、其中打开 `.npz` 或自己写分析脚本的次数（从会话记录数）、打开 `.past.jsonl` / 调用 `past.py` 的次数。

采用条件：主指标不变差且次指标下降，或主指标上升。结果写进本页第 7 节和 STATE。

## 7. 实现步骤与提交（Q22）

每步在 `henry-dev` 上 `make check` 后按名字暂存、提交：

1. 接触通道：`motor.py` 的 `trace` 字段、`arrays()` 的三列和刚体名表；`tests/test_scalebfm_motor.py`。
2. 构造器：`src/locomani/tasks/scene/past.py`，含 `unknown` 退化；`tests/test_scene_past.py`。
3. 写出：`scene_round.py`、`diag_pool.py` 的改名；50 局不变性验证，结果记在这里。
4. 摘要、`past.py`、`reviewer.md`、`rules.md` 第 23 版、`rsi_design.md` 第 5 节一段、runbook 一条命令；TODO 39 勾选。
5. 新运行 `pick-past-a`，`past_monitor.py`，结果进本页和 STATE。

记录（每步做完后按日期填在这里；截至 2026-10-30 只有方案，代码未动）：
- 2026-10-30：方案、TODO 39、DECISIONS 条目。

## 8. 第二阶段（不在本次）

- 跨任务检索 `past.py query`：按 `kind`、角色关系和物理条件（物体尺寸、支撑面高度、手）匹配本次运行和经验库里已完成运行的记录，比较事件模式；需要先建跨运行的索引（和 `records.py build` 同一种做法，排除隐藏池的种子段）。
- 回填脚本：对旧沙盒的 `.json` / `.npz` 生成 `.past.jsonl`，接触边 `unknown`。
- `past.py html`：每步前后两张小图排一页，放 `outputs/` 下。
- Engineer / Manager 提示词点名；记录和验收门账本的链接（修复复用、条件修正）。
- 重放 A/B（`replay.py setup` + `loop.py run --max-iters 1 --stop-after probe`，有 / 无 PAST 各若干次）给 H1 一个配对的结论。

## 9. 被否决的方案（及原因）

- 放进 `skills/` 让循环进化记录器：被测者改仪器，记录跨版本不可比。
- 做成第 10 个原子动作：动作层没有步骤语义，而且记录不是动作。
- 运行时现场算谓词、现场写记录：不能回填、不能单测、验证"不改结局"更难。
- 只记原语级或只记技能级：前者看不到意图、一局几十条；后者看不出失败在哪一步发展。
- 要求技能传 `roles=`：Engineer 漏传或传错就污染记录。
- 把转移序列纳入哨兵比较：谓词版本一升哨兵就误报场景变了。
- 第一版就做 A/B 重放：用户定先开新运行看方向。

## 10. 局限

- 非配对比较（第 6 节）。
- `expected` 表只覆盖现有原语的常见用法；技能发明的新组合只有观测没有期待。
- `supported_by` 靠接触和高度推，物体悬在容器壁上、或被两件家具同时托着时可能推错；记录里标了 `derived`，Reviewer 能回到 `touching` 看原始接触。
- 空间边用包围盒算：圆的、斜放的物体缝宽偏小，`blocking` 的相交深度是包围盒的，不是网格的；这些是解释性的量，不是判定。
- 第 4.4 节的四个常数是估的；邻居半径取小了漏实体，取大了记录变长。改了常数记录的 `scope.selector` 不同，跨运行比较时要看。
- 碰倒旁边物体现在不进成败判定（TODO 27.13）；记录会记下这些 `touching` 和 `moved` 事件，但它们和成败的关系要 Reviewer 自己判断。
- 只有仿真里的特权信息；原文的约束"部署策略不能依赖真值"这里天然满足——技能不读记录（规则第 23 版）。
