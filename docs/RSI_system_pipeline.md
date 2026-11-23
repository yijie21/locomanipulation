# 场景 RSI 的代码结构与系统流水线（2026-11-01，11-16 补）

这一页是对 ADT 公寓场景上的 RSI（recursive self-improvement）流水线的一次静态分析：代码分几层、每个模块管什么、一轮外环和一轮内环各走哪些步、数据落在哪里，以及 2026-10-30 加进去的 PAST 场景图记录器接在哪。原理和取舍见[详解](rsi_design.md)，命令见 [runbook](rsi.md)，实测结果见[进展报告](handoff/scene-rsi-report-20261019.md)，记录器的方案见 [PAST 方案](handoff/past-scene-graph-plan-20261030.md)。行数是 2026-11-01 `henry-dev` 上的。

11-16 补了三处：ADT 场景怎么建（第 1.1 节）、11-11 接进内环的知识库钩子（第 2.2 节第 0 步），以及同一任务族里 10 个任务之间怎么共享知识（第 5 节，对着 `ga-past-b` 的结果分析）。

## 1. 代码结构

分三层：**被进化的**（技能目录，只有 Engineer 能改）、**冻结的运行层**（仿真、动作层、运行器、谓词、记录器，人改了要重测）、**驱动程序**（`scripts/rsi/`，由人写，组织 Claude 角色和统计判定）。

```
src/locomani/
├── backends/scalebfm/
│   ├── runtime.py          ScaleBFM 控制器（M 22200），跟随骨盆 / 手腕 / 脚踝目标 —— 冻结
│   └── motor.py  (993)     动作层：walk_to turn sidestep posture reach grip release hold squeeze
│                           + 每拍轨迹 trace / 原语调用日志 calls / 接触通道 contacts —— 只许加默认关闭的开关
├── scene_discovery/
│   ├── predicates.py       held on in upright lifted near in_room：成败判据 —— 冻结
│   └── scan_room.py …      场景转换、分桶用的工具
└── tasks/scene/
    ├── world.py   (233)    SceneWorld：加载 ADT MuJoCo 场景、放物体、seat、state() 给谓词
    ├── start.py   (268)    起点抽样（机器人出生位姿、物体摆放、placement 选项）
    ├── navigate.py         占据图 + 走路路径
    ├── episode.py (158)    run_episode：一局的骨架（settle → plan(ctx) → hold → 判成败 → RoundResult + 轨迹）
    ├── stage.py   (161)    stage()：抓取步骤的"动作 → 检查 → 有限重试"合同（规则 31）
    ├── body.py / choose.py / prep.py / graspability.py
    │                       给角色用的身体模型、站位抓法挑选器、准备动作选择、可抓性几何
    ├── past.py    (796)    PAST 记录构造器：局结束后从 steps / calls / trace / 场景文件造场景图记录（TODO 39）
    └── skills/             ★ Engineer 唯一能改的目录
        ├── __init__.py     plan_for(task) → 一串技能调用
        ├── pick.py (862)   拿起（侧抓、俯捏、换手、换边…）
        ├── place.py (831)  放到台面（搬运计划、余量检查）
        ├── into.py (443)   放进容器
        ├── near.py (289)   拿着走到某物旁
        ├── held.py / clearance.py / cover.py / grasp_program.py   辅助：持握、避让、遮挡、种子抓取程序
        ├── bimanual.py / pack.py   两手抱取和两手放入、放入再抱箱（11-05，物流间线用，ADT 族不走到）
        └── test_*.py       技能自己的单元测试（写回时跑）

scripts/rsi/                驱动程序（人写；report/ 和仓库的 rsi_kit/ 不进沙盒，角色读不到）
├── outer.py / outer_run.py 外环：发现任务 → 分桶 → 选缺口 → 建任务族 → 逐族跑内环 → 写回 → 回归 → 能力表
├── loop.py (3033)          内环驱动：六个角色、起点池、审计、账本、续跑、停止线
├── gate.py                 验收门：候选 vs 现任，配对比较（哨兵 → 第一阶段 30 局 → 全池 → 结论）
├── sealed.py / regress.py / writeback.py / capability_write.py   封存集、回归、三方合并写回、能力表
├── sandbox.py              沙盒：git archive HEAD + 符号链接 third_party 和场景 + 自带 git
├── scene_round.py / batch.py / family.py   一局怎么跑、12 进程队列、任务族文件和起点池
├── digest.py / failure_class.py / records.py / isolate.py / design.py / diag_pool.py / try.py
│                           给 Reviewer 的视图：摘要、失败类型、记录库、动作层归因、设计探针、诊断池重抽、角色自测
├── reflect.py + experience/   经验库：每个跑完的任务族一条反思
├── knowledge.py             两层知识库：建库（build，一个带 tag 的共享文件，11-18）、整库写成 KNOWLEDGE.md 并标注命中（eval）、重放对照表（report）（11-11）
├── replay.py               从跑完的运行的某一轮起重放，比较流程改动；--knowledge 装入指定的库
├── past.py                 show / compare 看场景图记录（TODO 39）
├── prompts/{reviewer,manager,engineer,composer,reflector,consolidator}.md   角色提示词
├── rules.md                规则第 25 版（39 条），角色读的唯一人类知识
├── families/*.json  plans/*.md  seeds/   任务族、方案、空种子
└── report/                 报告、视频、监控脚本
```

数据位置（`$LOCOMANI_WORKSPACE`，本机 `~/workspace`，集群 `/workspace`）：

| 内容 | 位置 |
|---|---|
| 沙盒 | `<WORKSPACE>/code/rsi-scene/<沙盒名>/` |
| 内环状态 | `<沙盒>/outputs/scene_rsi_20261016/<任务族>/{pool_diag/, diag/incumbent_<hash>/, incumbents/, loop/}` |
| 每轮迭代 | `…/loop/iter_NN/`（`diag_digest.md`、`reviewer_context.json`、`hypotheses.json`、`probe/`、`manager.json`、候选、验收结论） |
| 账本 | `…/loop/ledger.jsonl`、`meta.json`、`parked.json`、`candidates_pool.json` |
| 隐藏池（角色读不到） | `<沙盒>/outputs/scene_rsi_20261016_hidden/<任务族>/{pool_accept/, pool_sealed/, gate/}` |
| 角色临时目录 | `<WORKSPACE>/tmp/rsi_scene_<沙盒名>/` |
| 登记表、能力表、记录库 | 主仓库 `outputs/scene_rsi_20261016/outer/registry.json`、`configs/scene_discovery/capability_mars_scalebfm.yaml`、`outputs/scene_rsi_20261016/records/library.jsonl` |

### 1.1 ADT 场景从哪来（离线一次，人跑）

```
scripts/adt_download.py     只取序列的 main_groundtruth（约 20 MB）和用到物体的 DTC 模型，不下 2.3 TB 全量
→ adt_to_scene.py geometry  scene_discovery/adt.py：ADT 的 y 轴朝上转成仿真的 z 轴朝上（ADT_TO_SIM），物体取第一帧位姿；
                            floorplan.py 墙和房间，surfaces.py 固定家具朝上的面作台面
→ adt_to_scene.py estimate  Claude 估名字、质量、摩擦、是否容器，properties.py 的规则校正
→ adt_to_scene.py mujoco --robot mars   scene_locomani.xml：可动物体用凸包，家具凸包只和机器人碰
→ scene_add_panels.py / scene_add_colliders.py / scene_settle.py --write   补面板和碰撞体，写入沉降后的位姿
→ （可选）adt_to_usd.py     Isaac 或 Unreal 的渲染资产
```

产物在 `outputs/scene_discovery/adt_apartment_mars/`：序列 `Apartment_release_clean_seq131_M1292`，`scene_meta.json` 里有 354 个物体、69 个台面。这些是 ADT 的衍生物，按 DECISIONS 2026-10-20 打包进了 `rsi_kit/`（`scripts/rsi/kit.py pack / restore`）。用这个场景的任务族有 `pick_single`、`r1_*`、`r2_*`、`cal_r2`–`cal_r4`、`grasp_atomic`、`grasp_atomic_past`、`grasp_prep`；`gso_*` 和 `box_bimanual` 是物流间。

## 2. 系统流水线

### 2.1 外环（`outer_run.py`，一轮 = 一批任务族）

```
场景文件 + 能力表 + 技能目录
   │ 1 发现 (scene_discover_tasks.py, Claude)   → 50–60 个任务，每个是谓词合取
   │ 2 分桶 (程序查能力表)                       → 能做 / 缺技能 / 做不到
   │ 3 选缺口 (outer.py select)                  → 按"解锁任务数 ÷ 估价"排序，取 ≤ 6 组
   │ 4 建任务族 (outer.py family)                → families/r<轮>_<缺口>.json（≤ 10 任务 + 三个池的种子）
   └ 5 每个任务族顺序：
        建沙盒 → 检查任务 → 内环 (loop.py) → 封存集 (sealed.py) → 经验库条目 (reflect.py)
        → 写回 (writeback.py 三方合并进主仓库 skills/) → 单元测试
        → 回归 (regress.py 重跑所有登记任务族的封存集，p ≤ 0.05 变差就不推进登记表)
        → 能力表 (capability_write.py) → 下一轮发现读新表
```

### 2.2 内环（`loop.py run <任务族>`，一轮迭代约 75 分钟）

```
现任 incumbent（技能目录快照 + 参数覆盖）
   │
   ├─ 0 驱动程序准备：现任跑诊断池 (50 局，batch.py 12 进程) → diag/incumbent_<hash>/{.json, .npz, .past.jsonl}
   │     诊断池失败 < 10 局就重抽 (diag_pool.py)；撤销已被做成的搁置；
   │     知识库钩子 (knowledge_step，11-11)：每条条目的条件在这些诊断局上求值 → 沙盒的 scripts/rsi/knowledge/KNOWLEDGE.md（整库加标注，11-18）
   │     + iter_NN/knowledge_hits.json，账本记一行 knowledge；库为空时不生成 KNOWLEDGE.md，提示里也没有知识块；
   │     同类失败 ≥ 2 局时单独重跑那条命令，判动作层还是技能 (isolate.py)；
   │     生成 diag_digest.md (digest.py：失败类型计数、每任务几何事实、首失败步骤、区分数、场景图转移一节)
   │     + reviewer_context.json (现任、参数、round_features 每局一行的条件列表、mechanisms 机制表、压缩 history、focus、parked、attribution、designs_last)
   │
   ├─ 1 Reviewer (claude -p, high)：读规则 / 方案 / 上下文 / 摘要 → .past.jsonl → NPZ 最后手段
   │     工具：graspability / body.reach / choose / prep / records.query / try.py (≤ 40 局/次, ≤ 240 局/轮)
   │     有库时还整个读 KNOWLEDGE.md（规则 39）
   │     输出 hypotheses.json：≤ 4 条 {claim, instead, evidence, delta.set, predicted_fixed / broken, needs_code?, measure?, design?, knowledge_refs?}
   │     + infeasible 声明（硬约束 check 程序 + 出路 + 1–3 次尝试）
   │
   ├─ 2 Prober (程序)：每条假设的参数 delta 在 50 个诊断起点上真跑 → status 已验证 / 已否定 / 有赢有输 / 仍是假设
   │     measure 的查记录库 (records.py)；design 的跑设计探针 (design.py, ≤ 2 个/轮)；claim 的跑 claim_check + attempts → 搁置
   │
   ├─ 3 Manager (claude -p, xhigh)：读账本 / 机制表 / 探针结果 / repair_notes → engineer (≤ 3 条假设) | reopen | review | stop
   │     提过两轮没做的机制优先；没给名额的由驱动程序强制占最后一个 (forced)
   │
   ├─ 4 Engineer ×3 (claude -p, xhigh, 并行)：各在一份技能目录副本里改代码 / 参数 → 草稿
   │     每份草稿一交就跑诊断池 (early_diag)；两份交了再 10 分钟没交的停掉 (late)
   │     审计：改了 skills/ 以外的文件、越界读隐藏目录、技能里写物体名 (规则 30)、读记录器 (规则 38) → 草稿作废
   │
   ├─ 5 Composer (claude -p, xhigh)：看三份草稿逐起点的赢输 → 合成一个候选 / 选一份 / 候补池里的一个 / 都不送
   │
   ├─ 6 验收门 (gate.py, 程序, 隐藏验收池 60 局)：
   │     静态检查 (能加载、参数认得、未被现行规则拒过) → 哨兵 (现任 5 局逐局复现) →
   │     第一阶段 (按任务分层 30 局, 净赢 > 0) → 第二阶段 (其余) →
   │     结论：b − c ≥ 3 且 c ≤ b / 3，摔倒掉落 ≤ 现任 + 1，p 照报不要求
   │     接受 → 新现任；拒 → 多赢多于少赢的进候补池 (留 2)；合成候选被拒时同轮再送最好的草稿 (second shot)
   │
   └─ 停止：现任验收池 ≥ 70% (--stop-rate) | 验收门调用满 12 | 墙钟 6 h | Manager stop | 连续两轮审计作废 (异常)
        → sealed.py 在封存集 (80 局) 上跑种子和每一版被接受的现任 → 报告和能力表用的数字
```

### 2.3 一局（`scene_round.run_round` → `episode.run_episode`）

```
起点 JSON → 重置机器人、摆物体、seat 进台面的抬到板面 → settle 1 s
→ skills.plan_for(goal) → 逐个技能；抓取步骤经 stage()（动作 → 检查实测状态 → 重试预算内重走 / 重夹）
   每个动作层原语：_logged 记进 calls；每拍 tick 记 trace（位姿、关节、参考、夹爪、物体、接触对）
→ hold → 谓词判成败（judge_s 时窗口内每 0.1 s 都要成立）→ RoundResult
→ 写 <label>.json（steps、calls、failure_class + events、skills hash、set、motor 签名）
   + <label>.npz（50 Hz 轨迹）+ <label>.past.jsonl（每步一条场景图记录）
```

### 2.4 三个起点池与防过拟合

| 池 | 局数 | 谁看 | 用途 |
|---|---|---|---|
| 诊断池 | 50 | 所有角色 | 找原因、试参数、探针；只看方向 |
| 验收池 | 60 | 只有验收门 | 配对判候选 vs 现任 |
| 封存集 | 80 | 内环结束后跑一次 | 报告和能力表 |

隐藏池在 `…_hidden/`，角色调用期间目录权限 000；每次角色调用后扫记录里的工具调用路径，越界即作废（Reviewer / Manager / Composer 或唯一的 Engineer 越界作废一轮，三个 Engineer 之一越界只作废它的草稿；连续两轮作废内环异常停下）。

## 3. 关键设计选择

- **候选 = 技能代码而不是参数**：和 warehouse / bottle 那条线的区别；验收门不变，候选从参数集变成技能目录快照。
- **一次验收 = 一次配对检验**：三个 Engineer 的草稿由 Composer 合成一个候选再进门，避免多重比较。
- **人类知识只有一份**：`rules.md` + 任务族方案文本；每改一版算一次插手（现在第 23 版）。
- **仪器和被测者分离**：失败类型、摘要、记录库、身体模型、挑选器、场景图记录都在冻结层或驱动程序，Engineer 只能改 `skills/`。
- **经验库是循环自己写的**：`reflect.py` 读账本生成；人写的经验库 10-19 试过没好处，没采用。

## 4. PAST 记录器在流水线里的位置（TODO 39）

```
motor.tick  ──contacts──▶ trace ──▶ arrays() ──▶ NPZ (contact_k / contact_a / contact_b, bodies)
                                            │
scene_round.run_round ──rec(steps, calls)───┼──▶ past.build(rec, arrays, meta, task) ──▶ <label>.past.jsonl
                                            │
digest.past_lines ◀─────────────────────────┘        ◀── Reviewer：摘要 → past.py show / compare → NPZ
loop.named_objects：skills/ 里引用记录器 → 草稿作废（规则 38）
```

一条记录对应一个技能步骤（或一次重试）。它只写不读，不改任何一局的结局（50 局重跑待做），不进验收门、哨兵和回归。

### 4.1 怎么保存

- **时机**：一局跑完、`scene_round.py` 写 `<label>.json` 之前，`past.build()` 从四个来源构造，`past.write()` 落盘。运行中不构造图；动作层每拍只多记一个 `contacts` 字段。
- **来源**：步骤日志 `steps`（哪一步、起止、成败、检查值、第几次重试）、原语调用日志 `calls`（走、伸手、合爪各自的参数和起止）、轨迹 NPZ（50 Hz 的位姿、关节、夹爪、物体、接触对）、场景文件 `scene_meta.json`（包围盒、质量、台面顶高、所属家具）。
- **格式与位置**：每局一个 `<label>.past.jsonl`，和 `.json` / `.npz` 同目录，一行一条记录；三个池都生成。真实一局 5 步约 24 KB。先写 `.tmp` 再改名；构造器出错只写一行 `{"error": …}`，不影响局的成败。
- **旧局**：10-23 之前建的沙盒没有接触通道和调用日志，记录照样能造——接触类的边记 `unknown`，步骤种类改从轨迹的原语标签推（`kind_src: "primitive"`，`show` 里显示 `reach?`），这时推不出用哪只手，`expected` 为空。

### 4.2 一条记录里有什么（schema `PAST-locomani-0.1`）

以 `pick-repro` 真实一局 `jam_pickup_s100047` 的第 3 步为例：

| 块 | 字段 | 内容 | 这一步里 |
|---|---|---|---|
| 动作上下文 | `step` `step_index` `attempt` `kind` `kind_src` `skills` `set` `action{t0,t1,ok,why,calls,check,info}` | 步骤名和序号、第几次重试、原语种类（walk / turn / sidestep / posture / hold / reach / grip / release / squeeze / mixed / none）及其来源、技能版本 hash、参数覆盖、起止时间、成败、原语调用、`stage()` 的检查值 | `grasp [reach] 21.2–28.2 s OK` |
| 角色 | `roles{target, effector, carried, support, goal_surface, container, stance, corridor, pregrasp}` | 从原语调用推出的目标物体、用的哪只手、拿着什么、当前支撑面、目标谓词点名的台面 / 容器、三种区域 | target=Jam_Anon, support=KitchIsland__0 |
| 范围 | `scope{selector, initial, initial_why, added}` | 四个常数、初始实体列表、每个为什么进来、执行中碰到才加入的（带时间和原因） | 任务物体、机器人、支撑面、所属家具、邻居 WoodenSpoon（"neighbour 0.05 m"） |
| 实体静态属性 | `entities{id: {type, …}}` | 物体：刚体名、质量、半尺寸、质心、能否移动、是否容器、竖直轴；面：顶高、半尺寸、偏航、可接近边、所属家具；手：组成刚体、最大开口；机器人；区域：形状和参数（圆盘 / 线段 / 点 / 矩形） | Jam_Anon 0.5 kg，半尺寸 [0.03, 0.04, 0.09] |
| 前后状态 | `G_before.states` `G_after.states` | 物体：位置、四元数、线速度、角速度、底面高、偏航；手：位置、开口、相对参考的误差、腕姿态档、手臂 7 关节角；机器人：骨盆位置和高度、偏航、是否摔倒、当前原语和模式 | |
| 前后的边 | `G_before.edges` `G_after.edges`，每条 `{s, r, o, v, src, …数值}` | 接触 `touching`；持握 `gripped`（两指都碰到）`held`（谓词：夹住且离开所有台面）`rel_pose`（物体在手里的偏置）；支撑 `supported_by` `in`；空间 `above`（上方净空 `gap_m`）`beside`（邻居的缝 `gap_m`、在目标哪一侧 `side`）`near_edge`（离哪条台沿多远）`blocking`（和站位 / 走廊相交多深）`dist`（机器人到目标，带方位角）`facing`；相对位姿 `reach_vec`（目标点在骨盆系的前伸 / 外侧 / 高度）`approach`（从物体哪一面接近、腕姿态）`at`（手或机器人到没到目标点，带误差）；谓词 `upright`。只记一端是角色或区域的对。`src` ∈ observed / predicate / derived / approx / unknown | `supported_by KitchIsland__0 true→false`；`beside WoodenSpoon gap 0.052 +x`；`near_edge -x 0.127` |
| 事件 | `events[{t, edge, from, to, src, cm?}]` | 区间内布尔边的每次翻转（接触和夹持逐拍查，其余每 0.1 s）、`moved`（物体位移 > 2 cm）、`slipped`（被持物体和手腕距离变化 > 1 cm） | 25.2 s `moved 2.3 cm`；25.3 s 离开台面 |
| 期待与差异 | `expected[{edge, to, …}]` `discrepancies[{…, observed}]` | 按原语的固定表：walk → 到站位、足迹无阻挡；reach → 手到目标 3 cm 内、目标和邻居不动、走廊无阻挡；grip → `gripped`；拿着往上伸 → `held` 且离开支撑；release → 松开，目标谓词点名时落到台面 / 进容器。`discrepancies` 是期待里没成立的项，只是差异，不是因果判断 | |
| 结果与证据 | `outcome{step_ok, round_ok, round_why, failure_class}` `evidence{npz, k0, k1, json_step}` | 步骤和整局的成败、失败类型；NPZ 文件名和拍区间 | ticks 1058–1409 |

### 4.3 谁怎么读

- **Reviewer**（提示词规定的顺序）：先看摘要 `diag_digest.md` 里的 "Scene-graph transitions" 一节（≤ 115 行：每个任务失败局里第一个不期待的转移计数，对照成功局同类步骤的值；每个失败局从第一个操作步骤起的翻转时间线，不期待的标 `*`）；再 `scripts/rsi/past.py show <局>` 按步骤看，或 `compare <失败局> <成功局>` 按原语种类对齐两局、只列值不同的边；最后才按 `evidence` 的拍区间读 NPZ。Manager / Engineer 第一版不读它。
- **人**：`.past.jsonl` 本身是可读 JSON；`show` / `compare` 的文本输出和摘要那一节给人看也够用；`--json` 导出单条。节点—边的图形页（`past.py html`）是第二阶段。
- **技能**：不许读。技能代码里 import `locomani.tasks.scene.past`、或字符串含 `past.jsonl` / `rsi/past.py`，草稿作废（规则第 38 条，`loop.named_objects`）。

## 5. 同一任务族里 10 个任务之间怎么共享知识（2026-11-16）

以 `grasp_atomic_past` 为例：10 个物体（Orange、CoffeeCan、DishSoap、Mug、TvRemote、WoodenSpoon、Candle、VaseSmall、Plant4、DisplayWolf）的 `held(X)` 任务，从只有动作层的空种子开始，没有经验库（`library_exclude` 排除了每一条）。三个池按物体平分：诊断池 30 局（每物体 3 局）、验收池 40 局（4 局）、封存集 50 局（5 局）。运行结果见 [ga-past-b 结果](handoff/ga-past-b-results-20261102.md)。

结论：**现在没有一个显式的知识共享模块。** 10 个任务在一个内环里一起进化，共享靠四样东西间接实现：同一份技能代码、禁止写物体名、10 个任务一起做配对验收、角色看到的是跨任务的同一份证据。显式的两层知识库代码已写好（第 5.5 节），但还没建过库，没参与过运行。

### 5.1 载体：同一份技能代码

10 个任务的目标都是 `held(X)`。`plan_for` 把它们全都交给 `pick_held`，再交给 `pick.pick(ctx, key)`（`skills/__init__.py`），所以只有一条执行路径。Engineer 改的是 `pick.py` 里的共享逻辑，每处改动同时作用到 10 个物体上。

从某个物体学到的东西，以按几何特征分支的条件和常数留在代码里（`pick.py` 开头的常数区）：

- `TOP_FLAT_M`：扁平物体放在矮台面上时，改用俯捏（遥控器、钱包学出来的）；
- `COVERED_VIA_OUT_M`、`NARROW_SIDE_MAX_M`：物体上方有遮挡时，用另一条接近路线（狼摆件学出来的）；
- `LOW_SPAN_M`、`LOW_WIDTH_M`：物体宽度接近爪口时，降到更窄的高度去抓。

条件写的是特征，不是名字。以后来一个同样扁平或同样被遮挡的物体，会自动走进对应分支。现在实际起作用的"共享层"就是代码里这些带条件的分支。

### 5.2 强制泛化：技能里不许写物体名（规则 30）

技能可以为一类物体写专门的流程（`recipes/`，带 `SIGNATURE` 和 `match(obj)`），由分发器按特征选，通用 pick 兜底。技能代码里只要出现场景物体的 key、刚体名、ADT 实例名或名字词干，草稿就作废（`loop.named_objects`）。所以即使一个修法只为救一个物体，也必须写成"台面高于多少、上方有遮挡"这样的特征条件。这条规则把每个物体上的局部经验强行写成带条件、可复用的形式。

### 5.3 选择压力：10 个任务一起做配对验收

验收门每次都同时覆盖 10 个任务（`gate.py`）：

- 第一阶段 30 局按任务比例分层抽，净赢不大于 0 就拒；
- 结论要多赢减少赢至少 3，且少赢不超过多赢的三分之一；
- 哨兵每个任务取 1 局，必须复现现任的结局。

所以修好 A 物体、却弄坏 B 物体的改动会被拒；每个物体只有 4 局，只救一个物体的改动也很难攒够净赢 3。`ga-past-b` 的记录：

| 轮 | 候选 | 逐物体变化（验收池） | 结论 |
|---|---|---|---|
| 0 | `0fa018` | 空种子变成 8 个物体都会做 | 接受，26 / 40 |
| 1 | `d16e63`、`4006ff` | 只有马克杯多 1 局 | 拒（多赢 1、少赢 0） |
| 3 | `ea93e5` | 洗洁精多 2、马克杯多 1、橙子多 1，咖啡罐少 1 | 接受（多赢 4、少赢 1） |
| 6 | `33aebe` | 狼摆件、马克杯、咖啡罐各多 1，橙子少 1 | 拒（多赢 3、少赢 1，净 2） |

第 3 轮和第 6 轮说明，一次共享代码的改动同时在几个物体上有正有负，验收门只按总的净收益做决定。封存集上两个被接受的版本逐局配对：第 3 轮版本多赢 7、少赢 1（31 → 37 / 50）。

### 5.4 诊断视角：角色看到的是跨任务的同一份证据

- Reviewer 读的 `diag_digest.md` 同时列出 10 个任务的失败类型和首个失败步骤；场景图转移那一节按任务对照失败局和成功局。所以假设天然是跨物体的机制（比如"手腕压到台面前下沿"），而不是单个物体的补丁。
- Prober 在全部 30 个诊断起点上验证假设，结果写的是救回了哪些起点、弄坏了哪些起点，按物体就能看出影响面。
- 记录库（`records.py`）的 `measure` 按特征列跨所有任务的局做 Fisher 检验，比如"台面高于 1 m 的局失败更多"。
- 机制表（`loop.mechanism_table`）按 Manager 的 `repeats` 把多轮里的同一机制串起来，前几轮的结论靠账本和压缩 history 传到后面。

### 5.5 显式的两层知识库：代码已写，没建过库

`knowledge.py`（[设计](handoff/shared-knowledge-plan-20261111.md)，规则 39）把上面隐式的东西变成显式条目：

```
跑完的账本 → 聚出机制，以及每个机制救回 / 弄坏的局（按现任版本和局标签）
          → join 诊断局的 PAST 图列和 start.* 特征 → 证据表（列按区分度排序）
          → Consolidator（LLM）只写 condition / signature / repair，程序校验列名和物体名
          → 程序按 evidence.by_object 定 generality：多个物体净非负是 multi_object，只有一个物体是 single_object，
            有物体净负先用正例物体的特征区间收窄条件（narrowed_by），排除不掉就是 contested
          → 下一轮在现任诊断局上求值条件，整库加命中标注写进 KNOWLEDGE.md 给 Reviewer
```

这里判断"10 个任务之间是否共享"的依据就是 `by_object`：一条修法在几个物体上净正。截至 11-16，仓库里没有 `scripts/rsi/knowledge/` 目录，即还没对 `ga-past-b` 跑过 `build`，HITS.md 从未生成；设计稿里的 `hand_hooked_under_slab`（狼摆件 3 / 3、盆栽反例）是示意，不是实测产出。

### 5.6 对照与问题

| | 现在实际起作用的 | 设计中的显式知识库 |
|---|---|---|
| 知识的形式 | 技能代码里按特征写的分支和常数 | 条件、失败签名、修法组成的 JSON 条目 |
| 跨物体还是单物体怎么定 | 不显式定，由验收门的总净收益隐式筛 | 程序按净正的物体数定 generality |
| 单物体知识放哪 | `recipes/` 加 `match()`（规则允许，主库现在没有） | `single_object` / `contested` 条目加 `narrowed_by` |
| 新任务怎么复用 | 写回主仓库 `skills/` 后，新任务族直接继承代码 | 条件在新任务的诊断局上命中才给角色看 |

1. **局部知识容易被验收门淘汰**：每物体 4 个验收局，只救一个物体的改动很难净赢 3。橙子和木勺 8 轮都没做成，部分原因可能在这里；这和 SSLA "保留局部适配"的目标有张力。
2. **共享知识没有显式的适用范围**：代码里的分支条件是 Engineer 按当时看到的物体写的，不记录它从哪几个物体验证出来、哪些是反例；这正是 `knowledge.py` 要补的。
3. **下一步**：在 SRA0889 上跑 `knowledge.py build ga-past-b grasp_atomic_past --iters 0-3`（沙盒在那台机器的 `/workspace/code/rsi-scene/`），看抽出几条、`shared` / `local` 划得对不对，再按设计稿第 7.2 节做配对重放。

## 6. 薄弱处（2026-11-01 的观察，未动）

1. **诊断只看方向**：50 局同配置重复过 6 / 10 和 2 / 10，Reviewer 的假设命中率受噪声影响大；新运行对比旧运行也是非配对的。
2. **`predicates._held` 的 resting 判定**查同一足迹里所有台面，包括物体上方的隔板（`bottom − top` 为负也满足 `≤ 0.02`），隔板下的物体 `held` 永远为假。
3. **技能文件越来越大**（`pick.py` 862 行、`place.py` 831 行），规则只限制"不许写物体名"，没有限制结构；Composer 合成时三方合并冲突的概率会上升。
4. **外环和内环的衔接是串行的**：同一轮外环的任务族都从轮初提交出发，前一族写回的进步后一族看不到。
5. **本机的角色调用**需要 `--allowedTools`（组织策略下 `bypassPermissions` 不放行 Bash），`loop.py` 的 `claude()` 还没改。
