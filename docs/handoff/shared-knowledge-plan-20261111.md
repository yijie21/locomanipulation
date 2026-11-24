# 场景 RSI 的两层知识库：详细设计（2026-11-11）

对 [SSLA 提案](../Robot_RSI_Shared_Knowledge_System_Design.md) 在现有场景 RSI 流水线（[详解](../rsi_design.md)、[代码结构](../RSI_system_pipeline.md)）上的落地设计。本页只写第一版：怎么从已有的账本和 PAST 场景图记录自动抽出带条件的知识条目、怎么分成共享层和局部层、怎么按命中检索给角色、怎么验证。状态：设计稿，未实现。

## 0. 已定的四个决定（2026-11-11 讨论）

| 问题 | 决定 | 理由 |
|---|---|---|
| 谁消费知识 | **角色读**（Reviewer、Engineer 的上下文），不进运行时技能，不进驱动程序的搜索 | 改动最小；技能是每轮重写的代码，没有稳定 API 可以挂知识；驱动程序读留到第二版 |
| 条目的单位 | **条件 → 失败签名 → 修法**：一条 = 可机检的图条件 + 不成立的期待 + 有效和无效的修法方向 + 逐物体证据 | 能直接从账本的翻转局 join PAST 图列得到；SSLA 的子过程 schema 需要跨动作的 PAST 记录，现在没有 |
| 第一版的数据 | **`ga-past-b` 离线建**（`grasp_atomic_past`，10 个 pick 任务，8 轮，2,660 条 PAST 记录），不跑新的 RSI | 唯一带 PAST 的运行；先回答"能不能抽出有用的条目" |
| 建库方式 | **程序 join 证据，LLM 只写条件和修法**，分层和检索由程序求值完成 | 条件可求值才能自动分层、收窄、按命中检索；人写的文本经验库 10-19 配对重放没有效果，纯文本不再试 |

两层不是两个库：同一个库里的条目按 `generality` 分 `multi_object`、`single_object` 和 `contested`，差别在条件的宽窄和证据来自几个物体（2026-11-23 以前这个字段叫 `scope`，值是 `shared` / `local`；因为和跨任务族的 shared knowledge 撞名改掉，下文已按新名字写）。提升（promotion）就是条件放宽后重新求值仍满足 `multi_object` 的判据。

## 1. 和现有部件的关系

| 现有部件 | 在本设计里的角色 |
|---|---|
| 账本 `loop/ledger.jsonl`（`probe` / `engineer` / `compose` / `gate` 行） | 翻转局和机制文字的来源 |
| `reviewer_context.json` 的 `mechanisms` 表（按 Manager 的 `repeats` 把跨轮的同一假设连起来） | 机制聚类直接复用，不另写 |
| `.past.jsonl` + `records.graph_columns()` / `start_features()` | 条件的词汇表：条件只能用这些列 |
| `records.judge()` 的求值方式（`OPS`、按任务过滤） | 条件求值复用它的算子和行选择 |
| `reflect.py` 的隔离调用（上下文目录在仓库外、`FORBIDDEN` 扫描、JSON schema 输出） | Consolidator 用同一套调用方式 |
| 经验库 `scripts/rsi/experience/` 和 `{LIBRARY}` 提示块 | 第一版并存；A/B 时两者同时关或同时开，只比知识库的增量 |
| `sandbox.py` 建沙盒、`loop.py` 每轮组 Reviewer 上下文 | 检索挂在这两处 |
| `replay.py` + `loop.py run --max-iters 1 --stop-after probe` | 验证手段 |

## 2. 知识条目

一个机制一条，JSON 文件 `scripts/rsi/knowledge/<来源任务族>/<机制 slug>.json`，进 git（和经验库一样）。

```json
{
 "id": "grasp_atomic_past/hand_hooked_under_slab",
 "version": 1,
 "status": "candidate",
 "generality": "contested",
 "generality_why": "net negative on Plant4_1; positive on DisplayWolf (3/3); Plant4_1 lost a start when the condition spanned every slab",
 "source": {
  "family": "grasp_atomic_past", "sandbox": "ga-past-b", "iters": [2, 3, 4, 5],
  "mechanism_ids": ["H3_stance_free_edges_and_deep_reach", "H9_hang_filter_own_support"],
  "accepted_in_iter": 5, "built_at": "2026-11-12T.."
 },
 "condition": {
  "all": [
   {"feature": "start.support_top_m", "op": ">=", "value": 0.71},
   {"feature": "start.support_top_m", "op": "<=", "value": 0.89},
   {"feature": "start.grasp_z_m", "op": ">", "value": 0.69},
   {"feature": "walk.blocking.what", "op": "==", "value": "own_support"}
  ],
  "narrowed_by": [{"feature": "start.obj_height_m", "op": "<=", "value": 0.20, "why": "positives' range; drops Plant4_1"}]
 },
 "signature": {
  "first_bad.kind": "reach", "first_bad.edge": "at robot_right_hand pregrasp",
  "failure_class": ["side_reach", "other"],
  "reads_as": "the pre-reach stalls 20+ cm off while the wrist presses the slab's front-underside edge"
 },
 "repair": {
  "works": "filter stances when choosing them: predict both hanging jaws at the walk's end and skip a stance that puts one inside the object's own support footprint",
  "does_not": [
   "raise the minimum stance distance globally (fixed the target, broke 4 other starts)",
   "a post-walk retract before the via (fixed 0 of 30)"
  ],
  "where": "stance choice, before the walk, not the reach",
  "parameters_touched": ["pick.D_MIN_M (do not)", "pick.hand_hooked (new)"]
 },
 "evidence": {
  "by_object": {
   "DisplayWolf": {"triggered": 3, "fixed": 3, "broken": 0},
   "Plant4_1":    {"triggered": 2, "fixed": 0, "broken": 1}
  },
  "diag_rounds": ["pick_DisplayWolf_s410029", "pick_DisplayWolf_s410009", "..."],
  "gate": {"b": 4, "c": 0, "accepted": true},
  "sealed": {"DisplayWolf": [2, 5], "others_changed": false},
  "counterexamples": ["pick_Plant4_1_s410018: the all-slab variant re-planned a success and lost it"]
 },
 "audit": {"names_checked": true, "hidden_checked": true}
}
```

字段说明：

| 字段 | 谁写 | 约束 |
|---|---|---|
| `condition` | LLM 写，程序校验 | 只能用证据表里出现的列名（第 3.2 节）和 `records.OPS` 的六个算子；`all` 必填、`any` 可选；求值时缺列算不触发，不把未知当真（SSLA 第 3 节） |
| `condition.narrowed_by` | 程序写 | 收窄时追加的特征区间（第 4.2 节），只用 affordance 列，不用名字 |
| `signature` | LLM 从证据表选 | `first_bad.*` 和 `failure_class` 取翻转局里占多数的值；`reads_as` 是给人和 Reviewer 看的一句话 |
| `repair` | LLM 从 `mechanism` / `compose.notes` / `gate` 提炼 | `works` 一句话；`does_not` 每条带账本里的数字；不许出现物体名 |
| `evidence` | 程序算 | 按物体的触发 / 救回 / 弄坏计数；只用诊断池的局（第 6 节） |
| `generality` / `generality_why` | 程序定，LLM 不碰 | 第 4.1 节的规则 |
| `status` | 程序 | `candidate` → `tested`（经过一次配对重放）→ `retired`（后来的证据净负） |

## 3. 建库：`scripts/rsi/knowledge.py build <沙盒> <任务族> [--iters 0-3] [--out scripts/rsi/knowledge]`

### 3.1 聚机制（程序）

读最后一轮（或 `--iters` 上限那一轮）的 `loop/iter_NN/reviewer_context.json` 的 `mechanisms` 表：每行一个机制，带它在各轮的假设 id、探针状态、建没建、草稿翻转、Composer 和验收门的结论。对每个机制收集它的**翻转局集合**：

- 探针行 `probe.results[i].fixed / broken`（按参数改动在诊断池上真跑的翻转）；
- 草稿的诊断翻转：`iter_NN/composer_context.json` 里每份草稿对现任的 fixed / broken 标签；
- 验收门只取 `gate` 行的 b / c 和按任务、按失败类型的分解，**不取逐起点**（验收池隐藏）。

机制只保留"至少有一个翻转局"或"被验收门接受"的；只在 Reviewer 嘴里出现、探针无翻转、从没建过的机制不进库（没有证据可 join）。

### 3.2 证据表（程序）

对每个机制，取它翻转局所属的任务在诊断池上的**全部**局（翻转的和没翻转的），每局取**改动之前那一版现任**的记录（条件描述的是修之前的局面），生成行：

| 列 | 来源 |
|---|---|
| `label`、`task`、`object`、`flip ∈ {fixed, broken, unchanged}` | 账本 |
| `start.*`：`spawn_dist_m`、`spawn_bearing_deg`、`obj_z`、`obj_yaw_deg` | `records.start_features()`（已有） |
| `start.support_top_m`、`start.grasp_z_m`、`start.obj_height_m`、`start.obj_thickness_m`、`start.obj_mass_kg`、`start.container`、`start.flat`、`start.side_fits_n`（`one_hand.side_fits_deg` 的段数）、`start.hug_span_m` | **新增** `records.affordance_columns()`：从 PAST 记录第一条的 `entities[target].affordance` 和场景文件取 |
| `<kind>.*`：`reach.at.err_cm`、`walk.blocking.depth_m / .what`、`reach.above.gap_m`、`reach.beside.gap_m / .side`、`reach.reach_vec.fwd / .out / .z`、`grip.moved_before_grip`、`t_touch_to_grip_s`、`slipped`、`approach.side` | `records.graph_columns()`（已有） |
| `first_bad.kind`、`first_bad.edge`、`n_unexpected`、`failure_class` | 已有 |

然后程序给每列算一个**分离度**：数值列用翻转集和非翻转集的中位数差除以合并四分位距，并给两组的 [10%, 90%] 区间；类别列给两组的计数表。按分离度排序，前 12 列连同全表一起交给 LLM。这一步不下结论，只是把"哪些列在翻转局上不一样"摆出来。

`walk.blocking.what` 现在是刚体名（如 `CornerDisplay__1`）。为了条件能跨场景，加一个派生列 `walk.blocking.role ∈ {own_support, other_furniture, free_object, none}`，由 PAST 的 `roles.support` 和实体类型算出。

### 3.3 Consolidator（LLM，一次 `claude -p`，每个机制一次）

- 调用方式复制 `reflect.py`：上下文目录放在 `$LOCOMANI_WORKSPACE/tmp/rsi_knowledge/<family>/<mechanism>/`，仓库外；`CLAUDE_CODE_DISABLE_CLAUDE_MDS`；输出受 JSON schema 约束（`condition`、`signature`、`repair` 三块）；会话记录按 `FORBIDDEN` 扫一遍，命中隐藏目录或 `rsi_kit` 的条目作废。
- 输入：证据表（第 3.2 节）、该机制各轮的 `claim` / `instead` / `evidence`、Engineer 的 `mechanism` 文字、Composer 的 `notes`、验收结论、经验库里这个任务族的 `edits` 表（如果已有）。
- 提示词 `scripts/rsi/prompts/consolidator.md` 的硬要求：
  1. 条件只用证据表的列名和六个算子，区间取翻转局的范围并说明为什么这样取；
  2. 条件要能把翻转局和同任务的非翻转局分开，分不开就写更宽的条件并在 `signature.reads_as` 里说明；
  3. `repair.does_not` 只写账本里真跑过、翻转为负或为零的改动；
  4. 任何地方不写场景物体的名字、刚体名或名字词干（规则 30 的词表由程序传入，`loop.named_objects` 同一套检查）；
  5. 不猜测机器人或控制器的数字；身体事实在 `rules.md`。
- 模型 `RSI_MODEL`（默认 `claude-opus-5-5`），推理档位 high（和 Reviewer 一致，不需要 xhigh）。费用估计：每个机制约 1 USD，`ga-past-b` 有翻转证据的机制约 10–15 个。

### 3.4 校验与求值（程序）

LLM 写完立即：

1. **语法**：列名在证据表里、算子在 `OPS` 里、数值类型匹配；不过就带错误重写一次，再不过丢弃并记 `build.log`。
2. **名字**：`loop.named_objects` 同一套词表扫 `repair` / `signature.reads_as` / `generality_why`；命中作废重写一次。
3. **求值**：在**全部可见记录**（该沙盒 `records.scan()` 的行，诊断池所有版本的局）上跑 `condition`，得到触发集；再和机制的翻转局集合交叉，得到 `evidence.by_object`。
4. **最低证据**：触发的翻转局 ≥ 2，且触发集里翻转局的占比高于非触发集（Fisher p ≤ 0.10，门槛松于 `records.judge` 的 0.05，因为这里是写库不是判假设）；不满足就记"条件不分离"重写一次，再不过丢弃。

### 3.5 产物

- `scripts/rsi/knowledge/<family>/<slug>.json`：条目；
- `scripts/rsi/knowledge/<family>/BUILD.md`：这次建库的日志（哪些机制进了、哪些丢了、为什么、费用）；
- `scripts/rsi/knowledge/INDEX.md`：所有条目一行一条（id、generality、条件的一句话、来源物体数、状态），给人看；角色不读 INDEX，只读命中（第 5 节）。

## 4. 分层

### 4.1 规则（程序，按 `evidence.by_object`）

| 情况 | `generality` |
|---|---|
| 触发 ≥ 2 个物体；每个物体 fixed − broken ≥ 0；至少一个 > 0 | `multi_object` |
| 只触发 1 个物体，净正 | `single_object` |
| 有物体净负 | 先尝试收窄（4.2）；收窄后满足第一行 → `multi_object`（条件带 `narrowed_by`）；否则 `contested`，负物体进 `counterexamples` |
| 触发的翻转局 < 2，或触发集和翻转集不分离 | 丢弃 |

"物体"是任务的目标物体（pick 族一个任务一个物体）。到 place / into 族时，键改成 (目标物体, 目标台面或容器) 对，规则不变。

### 4.2 收窄（程序）

当某个物体净负时，取**净正物体**的 affordance 列（`start.obj_height_m`、`start.obj_thickness_m`、`start.support_top_m`、`start.container`、`start.flat`、`start.side_fits_n`）的取值范围，外扩 10%，逐列尝试追加到 `condition.narrowed_by`，每加一列重新求值；一旦负物体全部掉出触发集就停，记下用了哪几列和为什么。所有列都加上还掉不出去，就不收窄，`scope` 定为 `local`，负物体留在 `counterexamples`。

收窄只加 affordance 列，不加 `<kind>.*` 的运行时列：前者是开局就知道的，后者要跑了才知道，检索时现任的局已经有了所以也能用，但第一版只用前者，保持条件可解释。

### 4.3 提升与退役（跨任务族建库时）

`knowledge.py build` 对第二个任务族（将来 `gso_all` 或 ADT 的 place 族）跑时，加 `--promote`：把库里所有 `single_object` 和 `contested` 条目在新任务族的可见记录上重新求值，若合并后满足 4.1 第一行，`generality` 升为 `multi_object`，`version` 加 1，旧版本留在 `history`。反过来，`multi_object` 条目在新任务族上有物体净负，降为 `contested` 并记反例。每次变动都在 `INDEX.md` 和 `BUILD.md` 留一行。

## 5. 检索

### 5.1 什么时候算

- **建沙盒时**（`sandbox.py`）：把 `scripts/rsi/knowledge/` 拷进沙盒，去掉本任务族自己的条目和任务族文件 `library_exclude` 点名的任务族的条目（和经验库同一个字段、同一套逻辑）；种子现任如果有诊断局就算一次命中。
- **每轮 Reviewer 之前**（`loop.py`）：现任的诊断局已经跑完（这是 Reviewer 的输入），对每条条目求值，写 `loop/iter_NN/knowledge_hits.json` 和沙盒的 `scripts/rsi/knowledge/HITS.md`。求值是纯 Python 读 JSON，毫秒级，不另起进程。

### 5.2 命中的判定

- `multi_object` 条目：现任诊断局里**至少 1 局**满足 `condition` 就算命中；
- `single_object` / `contested` 条目：满足 `condition`（含 `narrowed_by`）的局至少 1 局；因为 `narrowed_by` 已经把物体特征限住了，不另加判断；
- 命中的局里**失败局**才算有用：全是成功局的命中列为"条件命中但现任已做成"，放在 HITS 末尾一行，不展开。

### 5.3 角色看到什么

`HITS.md`，每条命中一节，按命中的失败局数排序：

```
## hand_hooked_under_slab  (multi_object, v2, from grasp_atomic_past + r1_held_single; tested)
Hits this incumbent's diagnostic rounds: pick_DisplayWolf_s410029 (failed), pick_DisplayWolf_s410009 (failed), pick_Plant4_1_s410028 (ok)
Condition: support top 0.71–0.89 m, grasp above 0.69 m, the stance footprint intrudes into the object's own support
Reads as: the pre-reach stalls 20+ cm off while the wrist presses the slab's front-underside edge (first_bad reach/at)
Worked elsewhere: filter stances when choosing them -- predict the hanging jaws at the walk's end, skip a stance that puts one inside the object's own support footprint (DisplayWolf 3/3; gate b 4 c 0)
Did not work: raise the minimum stance distance globally (broke 4); a post-walk retract (fixed 0 of 30)
Counterexamples: pick_Plant4_1_s410018 -- the all-slab variant re-planned a success and lost it
```

`{LIBRARY}` 提示块改成两句：命中文件在哪、怎么用——"这些是别的任务族在相似局面下验证过的方向；本任务族的局决定；引用时写条目 id，这样驱动程序能数引用"。Reviewer 的 `hypotheses.json` 加可选字段 `knowledge_refs: [id]`，`library_monitor.py` 的 `cites` 统计改读它（现在是正则匹配文字）。Engineer 的提示块给同一个 HITS，但只给 Manager 选中的假设引用的那几条。Manager 和 Composer 第一版不读。

### 5.4 没命中时

`HITS.md` 只有一行"no entry's condition holds on this incumbent's diagnostic rounds"。角色看到的和没有库一样。这是和现有经验库最大的行为差别：不命中就不给。

## 6. 隔离与审计

- **只用可见数据建库**：`knowledge.py build` 读沙盒 `outputs/scene_rsi_20261016/` 下的诊断局和账本；验收池、封存集的逐起点记录在 `_hidden/`，不读；验收门只取 b / c 聚合。`records.build` 排除隐藏种子段的规则照用。
- **不含名字**：条目的所有文字字段过 `loop.named_objects` 的词表（场景的物体 key、刚体名、ADT 实例名、名字词干）；`evidence.by_object` 和 `diag_rounds` 含物体名，是**给程序和人看的**，不进 `HITS.md` 的正文——HITS 里的命中局标签本来就是本任务族自己的，不是泄漏。
- **技能不许读**：`loop.named_objects` 对 `skills/` 的字符串检查加 `rsi/knowledge`、`knowledge_hits`、`HITS.md`；引用即草稿作废（规则第 38 条的同一机制）。
- **规则文件**：加第 39 条（知识库：条目怎么来、角色怎么用、技能不读），规则升到第 25 版，按惯例记为一次人工插手。
- **沙盒 keep 列表**：空种子任务族的 `sandbox.keep` 加 `scripts/rsi/knowledge.py` 和 `scripts/rsi/knowledge`。

## 7. 验证

### 7.1 第一步：人看条目（不花模型费之外的钱）

`knowledge.py build ga-past-b grasp_atomic_past --iters 0-7` 跑完，看 `BUILD.md` 和每条条目：

- 条件是否只用了开局可知的列加少数运行时列；区间是不是翻转局的真实范围；
- 触发集是不是落在对的物体上（挂手条目触发 DisplayWolf，不触发桌面物体）；
- `generality` 的判定和人的直觉是否一致（经验库 `grasp_atomic` 那条 lesson 里"只算物体自己的板、别的板会误伤盆栽"应该被自动产出为一次收窄）；
- 丢弃了多少机制、为什么。

预期产出 8–12 条条目；若少于 5 条，说明证据表或条件语法太窄，先修这个再往下。

### 7.2 第二步：同一运行内的时间切分配对重放

库不能在建它的数据上测。可做的干净对照：

1. `knowledge.py build ga-past-b grasp_atomic_past --iters 0-3 --out <临时库>`：只用前四轮的证据；
2. `replay.py setup ga-past-b grasp_atomic_past 4 <沙盒 A_k>`，k = 1..3，拷入临时库；同样建 `<沙盒 B_k>` 不拷；
3. 每个沙盒 `loop.py run grasp_atomic_past --max-iters 1 --stop-after probe`：只跑 Reviewer 和 Prober；
4. 比较（`replay.py report` 加列，或新的 `knowledge.py report`）：

| 指标 | 来源 |
|---|---|
| 假设的探针状态分布（verified / refuted / hypothesis / mixed） | `probe` 行 |
| 净救回局数（fixed − broken 之和） | `probe` 行 |
| 预测命中率 `predicted_fixed_hit / predicted_fixed_n` | `probe` 行 |
| 引用了哪些条目、引用的假设探针状态是否更好 | `knowledge_refs` |
| Reviewer 用时、费用、工具调用数 | `review` 行 |

原运行第 4 轮 Reviewer 的真实假设作为第三个参照（它当时没有库）。预期是一个待检验的假设：带库的 Reviewer 在挂手类机制上更快给出已验证的探针。费用：6 次重放，每次 Reviewer 约 1.6 USD 加探针算力（30 局 × 假设数），合计约 15 USD、本机单卡约 3 小时。

这一步回答的是"结构化、按命中检索的自身历史记忆有没有用"，**不是跨任务迁移**。

### 7.3 第三步（第二版）：跨任务族

`gso_all_s<seed>` 第一次内环（TODO 40.8）跑完后，`knowledge.py build <gso 沙盒> gso_all_s0 --promote`，看 ADT 的 `single_object` / `contested` 条目有没有在物流间的物体上触发并升级，和 `multi_object` 条目有没有被降级。再用 `gso_all_s1` 带 / 不带库各跑一次内环做真正的跨任务对照。这是另一个决定点，不在本页范围。

2026-11-23：跨任务族的另一条通道——代码同步——在第一次交替运行（gso-alt-a）后落了地：两个任务族共用的计算抽进 `skills/grasp_common.py`，同步冲突即失败、先跑测试再过门，见 [runbook 7.5](../rsi_logistic_room_1030_single_bimanual.md)。

## 8. 文件与改动清单

| 文件 | 改动 |
|---|---|
| `scripts/rsi/knowledge.py`（新，约 400 行） | `build`（3.1–3.5、4.1–4.3）、`eval <沙盒> <现任目录>`（5.1–5.2，写 `knowledge_hits.json` 和 `HITS.md`）、`index`、`report`（7.2 的表） |
| `scripts/rsi/prompts/consolidator.md`（新） | 3.3 的提示词 |
| `scripts/rsi/knowledge/`（新目录） | 条目、`INDEX.md`、各任务族的 `BUILD.md` |
| `scripts/rsi/records.py` | 新增 `affordance_columns()` 和 `walk.blocking.role` 派生列；`row()` 合入 |
| `scripts/rsi/sandbox.py` | 拷知识库、按 `library_exclude` 去条目、种子现任有诊断局时算一次命中 |
| `scripts/rsi/loop.py` | Reviewer 之前调 `knowledge.eval`；`{LIBRARY}` 块加命中文件的两句；`named_objects` 加三个字符串；`hypotheses.json` schema 加 `knowledge_refs` |
| `scripts/rsi/replay.py` | `setup --knowledge <目录>`：把指定库拷进重放沙盒 |
| `scripts/rsi/report/library_monitor.py` | `cites` 优先读 `knowledge_refs` |
| `scripts/rsi/rules.md` | 第 39 条，第 25 版 |
| `scripts/rsi/families/*.json` | 空种子任务族的 `sandbox.keep` 加两项 |
| `tests/test_rsi_knowledge.py`（新） | 合成账本和 PAST 记录：机制聚类、证据表列、条件语法校验、求值缺列不触发、分层四种情况、收窄掉出负物体、名字检查作废、HITS 只列命中、没命中时一行 |
| `docs/rsi_design.md` 第 5 节、`docs/rsi.md` 第 5 节、`docs/reference/change-map-scene.md` | 一段说明和两条命令 |

实现顺序：
1. `records.affordance_columns` + `knowledge.py build` 到证据表为止（不调 LLM），在 `ga-past-b` 上跑，看表；
2. Consolidator 调用 + 校验 + 求值 + 分层，跑 `ga-past-b` 全部 8 轮，人看条目（7.1）；
3. `eval` / `HITS.md` / `sandbox.py` / `loop.py` / 规则第 39 条 / 测试；
4. 7.2 的配对重放。

每步都可以停下来看结果再决定下一步。

## 9. 风险与未定项

- **样本小**：每个物体诊断池 3 局，一个机制的翻转局常常只有 2–4 局；条件区间会过窄或过宽。4.1 的门槛（≥ 2 个物体）和 3.4 的 p ≤ 0.10 都是初值，7.1 看完条目再调。
- **条件描述的是"修之前"的局面**，但有些机制的翻转证据来自建在别的改动之上的草稿（如第 3 轮的三合一候选）。`compose` 行的归因文字能分一部分，分不开的机制 `repair.works` 里要注明 confounded，和经验库的 verdict 一致。
- **`<kind>.*` 列依赖技能的步骤种类**：按原语种类键控已经让它跨技能版本稳定（TODO 39.6），但一个技能把走和伸手合成一步时列会错位；第一版接受。
- **角色读了会不会反而收窄搜索**：10-19 人写经验库和现有经验库都没看到效果，也没看到坏处；7.2 的重放同时看假设数和机制多样性（`library_monitor.py` 的 `mechanisms` 列）。
- **提升需要第二个任务族**：第一版所有 `multi_object` 都是"跨物体"意义上的，跨任务要等 7.3。
- **要不要让 Prober 用 `repair.does_not` 直接跳过已知无效的参数探针**：这是"驱动程序读"的第一步，留到第二版讨论。
