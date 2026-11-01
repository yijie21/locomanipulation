# 场景 RSI 的代码结构与系统流水线（2026-11-01）

这一页是对 ADT 公寓场景上的 RSI（recursive self-improvement）流水线的一次静态分析：代码分几层、每个模块管什么、一轮外环和一轮内环各走哪些步、数据落在哪里，以及 2026-10-30 加进去的 PAST 场景图记录器接在哪。原理和取舍见[详解](rsi_design.md)，命令见 [runbook](rsi.md)，实测结果见[进展报告](handoff/scene-rsi-report-20261019.md)，记录器的方案见 [PAST 方案](handoff/past-scene-graph-plan-20261030.md)。行数是 2026-11-01 `henry-dev` 上的。

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
        └── test_*.py       技能自己的单元测试（写回时跑）

scripts/rsi/                驱动程序（人写；report/ 和仓库的 rsi_kit/ 不进沙盒，角色读不到）
├── outer.py / outer_run.py 外环：发现任务 → 分桶 → 选缺口 → 建任务族 → 逐族跑内环 → 写回 → 回归 → 能力表
├── loop.py (≈2900)         内环驱动：六个角色、起点池、审计、账本、续跑、停止线
├── gate.py                 验收门：候选 vs 现任，配对比较（哨兵 → 第一阶段 30 局 → 全池 → 结论）
├── sealed.py / regress.py / writeback.py / capability_write.py   封存集、回归、三方合并写回、能力表
├── sandbox.py              沙盒：git archive HEAD + 符号链接 third_party 和场景 + 自带 git
├── scene_round.py / batch.py / family.py   一局怎么跑、12 进程队列、任务族文件和起点池
├── digest.py / failure_class.py / records.py / isolate.py / design.py / diag_pool.py / try.py
│                           给 Reviewer 的视图：摘要、失败类型、记录库、动作层归因、设计探针、诊断池重抽、角色自测
├── reflect.py + experience/   经验库：每个跑完的任务族一条反思
├── past.py                 show / compare 看场景图记录（TODO 39）
├── prompts/{reviewer,manager,engineer,composer,reflector}.md   角色提示词
├── rules.md                规则第 23 版（38 条），角色读的唯一人类知识
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
   │     同类失败 ≥ 2 局时单独重跑那条命令，判动作层还是技能 (isolate.py)；
   │     生成 diag_digest.md (digest.py：失败类型计数、每任务几何事实、首失败步骤、区分数、场景图转移一节)
   │     + reviewer_context.json (现任、参数、mechanisms 机制表、压缩 history、focus、parked、attribution、designs_last)
   │
   ├─ 1 Reviewer (claude -p, high)：读规则 / 方案 / 上下文 / 摘要 → .past.jsonl → NPZ 最后手段
   │     工具：graspability / body.reach / choose / prep / records.query / try.py (≤ 40 局/次, ≤ 240 局/轮)
   │     输出 hypotheses.json：≤ 4 条 {claim, instead, evidence, delta.set, predicted_fixed / broken, needs_code?, measure?, design?}
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

一条记录对应一个技能步骤（或一次重试），含动作上下文、角色绑定、按相关性选的实体（目标、两手、支撑面、目标台面 / 容器、0.30 m 内的邻居、站位足迹和走廊里的家具、上方 0.5 m 内的面、区域）、前后两张局部图（状态 + 五类边：接触、持握、支撑、空间、相对位姿，加谓词）、中间的翻转事件、按原语的期待转移和差异、结果和 NPZ 证据区间。它只写不读，不改任何一局的结局（50 局重跑待做），不进验收门、哨兵和回归。

## 5. 薄弱处（2026-11-01 的观察，未动）

1. **诊断只看方向**：50 局同配置重复过 6 / 10 和 2 / 10，Reviewer 的假设命中率受噪声影响大；新运行对比旧运行也是非配对的。
2. **`predicates._held` 的 resting 判定**查同一足迹里所有台面，包括物体上方的隔板（`bottom − top` 为负也满足 `≤ 0.02`），隔板下的物体 `held` 永远为假。
3. **技能文件越来越大**（`pick.py` 862 行、`place.py` 831 行），规则只限制"不许写物体名"，没有限制结构；Composer 合成时三方合并冲突的概率会上升。
4. **外环和内环的衔接是串行的**：同一轮外环的任务族都从轮初提交出发，前一族写回的进步后一族看不到。
5. **本机的角色调用**需要 `--allowedTools`（组织策略下 `bypassPermissions` 不放行 Bash），`loop.py` 的 `claude()` 还没改。
