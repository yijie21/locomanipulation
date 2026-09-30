# 可拼接的技能（skills）：plan 入口、agent 规划器、新物体标定

`src/locomani/skills/` 把三条已调顺的专家线切成 agent 能拼的 primitive（设计与决策在
[primitive 契约设计](handoff/primitive-contracts-design-20260928.md)，本页只讲怎么跑）。一个 primitive 是一个阻塞执行、
自带时序的动作（走近、抓、搬、放），读 [session](handoff/primitive-contracts-design-20260928.md#3-session-对象) 与特权状态，
返回带 `[tag]` 的结果；executor 在每次调用前查前置、后校验不变量、每次调用写一行 trace。术语（栈、deploy、特权状态、trace）见
[词汇表](reference/glossary.md)。

## 1. 准备一次

先做完[安装指南](setup.md)第 0–4 步，再准备要用的场景（单臂 [§1](single_arm.md#1-准备一次)，纸箱 [§1](warehouse.md#1-准备一次)）。
**不要 `source scripts/paths.sh`**：它在 `INSTANCE` 未设时导出实例 0 的端口，入口会对错 deploy 说话（症状 `did not stabilise after reset`
+ `no body_q_target`）。入口按 `--instance` 自己钉端口。

```bash
cd <仓库根>
export SONIC_DEPLOY_ROOT="$PWD/outputs/bottle_continuous_20260914/deploy" PYTHONPATH="$PWD/src"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
```

起栈前查三样：`nvidia-smi`（目标 GPU 有空）、5 秒 CPU 采样（`/sys/fs/cgroup/cpu.stat`，配额 15.36 核，忙着 > 11 核就别起）、
`/workspace/tmp/p03_i<实例>/stack.json` 不存在（存在 = 那个实例被别的会话占着）。

## 2. 跑一个固定 plan

```bash
$PY scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml --gpu 1 --instance 1 --rounds 3 --dry-run
$PY -u scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml --gpu 1 --instance 1 --rounds 10 --seed 100 --name <run>
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 10 \
    --plan configs/plans/pick_place_via_basket.json --name <run>
```

- **做什么**：起栈（这个进程自己的，`--instance` 的端口块，无 exporter）→ 每局按规划站位 teleport → 按 plan 逐个调 primitive → 用任务判据
  （`task.goal` 有值时是模板判据 `put X in C`，否则"末了还持着目标"）判成败 → 写 `outputs/runs/<run>/skills/manifest.jsonl`、
  `agent_trace_NN.jsonl`、`videos/round_NN.mp4`。
- **参数**：实验文件给场景、后端和任务（`task.target` / `task.goal`）；`--plan` 是 JSON 列表 `[{"primitive": 名, "args": {...}}, ...]`
  （现成的在 `configs/plans/`）；不给 `--plan` 就按任务推：单手目标 `approach → grasp_single [→ approach(goal) → place]`，
  双手目标 `approach → ready → grasp_bimanual`；`--seed` 是第 0 局的物体朝向种子，逐局 +1；`--profile` 用别的 profile 名替换场景为目标选的那个（§5）。
- **就绪与停**：日志出现 `[plan] round 0:` 就在跑；`Ctrl-C` 或跑完都会收栈（`--keep-stack` 留着）。看结果：
  `python3 -c "import json;[print(json.loads(l)['round'],json.loads(l)['ok'],json.loads(l)['why']) for l in open('outputs/runs/<run>/skills/manifest.jsonl')]"`。

`--dry-run` 只打印解析出的 plan、场景和栈参数，不起任何进程；换实验文件或 plan 后先跑一次它。

`--think-s 8` 让固定 plan 在每次成功调用后停 8 s（agent 一次决策的时长），停顿走 `Executor.idle`（§4）；
持单手物体时 trace 多一行 `{"event": "think", "drift_m", "drift_rate_cm_s", "dz_m", "touching_end", ...}`，qpos 流存 `skills/think_<局>_<第几次>.npz`。
`--think-silent` 停顿期间什么都不发，复现 agent runner 的空档：deploy 在 1 s 收不到规划器消息后释放冻结上身和手指命令，
手里的东西 ~1 s 后被甩掉（[pick_place §14](ledger/pick_place.md)）。

## 3. 目录（agent 能调什么）

| primitive | 参数（省略取默认） | 从哪个通道进 → 出 | 做什么 / 状态 |
|---|---|---|---|
| `approach(target, profile)` | — | swing/frozen → swing（guard 走近）或 frozen（持物） | 单臂：朝桌法线、守线刹车、写站位；双臂：走入 + 刹车。**持物时**：`carry_to` 到目标的规划站位再原地转向桌法线（放置前走到容器） |
| `grasp_single(target, profile, side)` | `side=right` | swing → frozen | 走 profile 的 `reach.strategy`：`simple`（bottle 10 局 9/10）或 `coupled`（罐头，pick_place §12）；其余策略在 `approach` 前置被拒 |
| `ready(profile)` / `grasp_bimanual(target, profile)` | — | → frozen / frozen → pose_stream | 双臂原地抬手；协同伸抓到 hold |
| `place(container, profile, side)` | — | frozen → frozen | 抬、转到放置带、送、降、松、抬开、后退、验收 |
| `put_down_bimanual(profile, surface, forward_m, lower_speed, release_seconds, raise_m, back_m)` | `surface` 默认物体原表面；`forward_m` 默认按台面近沿自动算 | pose_stream → pose_stream | 双手同步下放到表面静止高度 → 松手 → 抬开；判据只看物体（高度、倾角），松手后手指贴壁只记指标。`warehouse_box` profile 的 `stand_forward_min` 0.50 让箱子留在台面上方，之后 `forward_m: 0` 直接放 8/8（crate 台账） |
| `carry_to(xy \| relative, ...)` | `turn_first=true, turn_mode=slow_walk, drift_max=0.03` | frozen → frozen | 上身冻结走直线，方位差 > 0.2 rad 先原地转；持物守卫每拍查（单臂掌系漂移 < 3 cm + 接触；双臂高度底线 + 双手接触） |
| `navigate_to(xy \| target, stand_off, step, ...)` | `stand_off=0.6, step=0.45` | swing/frozen → 不变 | 沿占用图的距离场分段走；空手每段边走边朝向路点，持物每段先转（slow_walk）再守线走 |
| `walk_to(xy)` / `turn_to(yaw, mode)` / `retreat(distance)` | — | — | 直走（持物时自动走 `carry_to` 的守卫）；原地转（< 9° 不准）；持物倒退 |
| `nudge(relative, tol, pulse_s, max_pulses)` | `tol=0.04, speed=0.35, pulse_s=0.5, max_pulses=4` | swing/frozen → 不变 | ≤ 0.4 m 的站位修正：定时 SLOW_WALK 脉冲 → 刹停 → 量残差，逐次缩短；v1.1 没有厘米级平移，`walk_to` 走 0.15 m 会冲 0.7–1.5 m。持物时带守卫 |
| `take_over(side)` / `hand_back()` / `release(sides)` | — | frozen ↔ pose_stream | 通道切换；松手 |

被拒的调用不执行，trace 里是 `rejected: [{check, measured, bound, note}]`，常见 `check`：`arm_owner`（通道不对）、`stance`（没先走近）、
`profile`（走近与抓取的 profile 混配）、`held`（持物状态不允许）、`map`（栈没有占用图）、`unknown_arg`。

## 4. 让 agent 自己选（`--agent`）

```bash
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 3 \
    --agent --agent-model sonnet --max-steps 12 --name <run>
```

- **做什么**：没有固定 plan。每局起一个 `claude -p` 会话（`--output-format json --json-schema ... --tools ""`，不开工具、不读文件；
  第一次 `--session-id`，之后 `--resume`），把任务、场景菜单（`scene.instances()`）、目录（每个 primitive 的参数与默认值）、
  特权状态文本（物体在骨盆系与世界系的位置、距离、接触、是否在 ego 视野）和 session 快照发给它；它回一个 `{"next", "args", "rationale"}`；
  executor 执行后把那条 trace 回给它，直到它答 `done` / `abort` 或到 `--max-steps`。成败仍由 sim 判据算，agent 说了不算。
  等它回答的 5–9 s 里 runner 用 `Executor.idle` 按 50 Hz 重发当前命令（规划器手臂：IDLE、朝向锁等待开始时的骨盆 yaw；流式手臂：重推当前参考）；不发的话 deploy 1 s 后释放冻结上身和手指命令，手里的东西被甩掉（[pick_place §14](ledger/pick_place.md)）。注入的 settle 等待走同一个 `idle`。
- **参数**：`--agent-model` 是 `claude` 的模型别名（`sonnet` 默认，`opus`、`haiku`）；`--max-steps` 每局最多几次调用。
- **产物**：除 §2 的文件外每局多一份 `skills/agent_io_NN.jsonl`（每次请求、回复、耗时、用量、错误），manifest 里 `agent: true`、`agent_calls`。
- **前提**：机器上装了 Claude Code 并已登录（`claude --version`）；一次调用约 5–15 s。`claude` 失败（超时、非零退出、不合 schema）
  记成 `abort` 加原因，不会让局崩掉。

## 5. 新物体怎么标定

profile 把走近参数和抓取参数绑在一起（设计 §6），场景按物体几何选 profile（`SceneModel.profile_for`：非双手 → `bottle_simple`；
双手 + 台面 ≥ 0.25 m → `crate_low_table`；否则 `warehouse_box`）。**没有"按几何自动调参"的机制**；新物体的路是：

1. 复制最近的实验文件（单臂 [§3](single_arm.md#3-新建一次实验)，纸箱 [§3](warehouse.md#3-新建一次实验)），改场景与目标；
   在 `src/locomani/skills/profiles.py` 的 `PROFILE_SOURCES` 登记一个新 profile 名指向它。
2. `--profile <新名> --rounds 10` 跑一次 pilot（10 局是这个仓库的 pilot 规模，见 [标准](reference/standards.md#evidence)）。
3. 看 `manifest.jsonl` 的 `why` 按 `[tag]` 分族；**每次只改一个旋钮**（走近：`stand_off` / `guard_*`；抓取：`entry_*` / `close_*`），再 10 局。
4. 与原线的 gate 对照（`scripts/gate.py compare <line>`）不退步后才把新 profile 写成默认。
   已走过一遍的例子（罐头，2026-09-28，[pick_place §11](ledger/pick_place.md)）：`can_entry15` / `can_descend05` / `can_overshoot0` /
   `pick_place_coupled` 四个 profile 各改一处，`--profile <名>` 同 seed 10 局；`grasp_single` 的 trace 里 `metrics.stats` 带 simple 策略的进入 / 失败证据，先看它再选旋钮。
5. 站位准入带：`STANCE_BANDS`（`profiles.py`）默认全开。要不要收，先跑回放：

```bash
$PY scripts/replay_preconditions.py --out outputs/analysis/precondition_replay_<日期>
```

它读已录的 plan 运行（`agent_trace_*.jsonl`）、bottle 采集 manifest（`park`）和纸箱 manifest（`stand_attempts`），
按 profile 报当前带误拒多少成功局、放行多少失败局，并扫一遍候选深度 / 横向上界。2026-09-28 的结论：三条线的成功与失败在站位深度上重叠
（bottle 中位 0.324 vs 0.325，warehouse 0.436 vs 0.454），带不分族，失败在抓取内部（`[grasp]`、`reach_stalled`、`lost_attachment`），所以带保持全开。

## 6. 证据与台账

真栈结果按线记在台账：单臂 [bottle §40–41](ledger/bottle.md)、pick_place [§7–8](ledger/pick_place.md)、纸箱 [crate 末节](ledger/crate.md)。
离线测试 `make test TESTS="tests/test_skills_*.py"`（CPU，假栈）。

## 7. 失败注入与配对比较（Q11 验收）

agent 在正常局里的选择一个枚举器也能做（[agentic 决策 §2](handoff/agentic-plan.md)）；它唯一不可替代的贡献是**失败之后的决策**，
而自然失败率凑不出统计量，所以失败要注入。`--inject <json>` 让 runner 在**某个 primitive 第一次成功之后**触发一次世界事件
（`src/locomani/skills/injection.py`，走 sim 的 `pin_object`：钉住 `settle_s` 秒再松开），固定 plan 与 agent 在同一 seed 上吃到同一事件，
谁都不会被告知——agent 只看到下一次请求里的特权状态（物体位置、接触）和 session 快照（`drop_held` 后 `held` 仍是旧值，本体感觉也会这么说）。

| kind | 触发后 | 做什么 | 固定 plan 的下场 | agent 要做的 |
|---|---|---|---|---|
| `move_target` | `approach(target)` | 物体沿世界 xy 挪 `offset_m`，高度姿态不变 | `grasp_single` 抓空 | 再 `approach` 再抓 |
| `drop_held` | `grasp_single` | 物体回到场景 `init_pos + offset_m`（直立） | 持物 `approach(goal)` 报 `[approach] load dropped` | `release` 清掉旧 `held`，再走近、再抓 |

推机器人（走偏）和对负载施力还没有 sim 侧命令，没做。注入文件在 `configs/injections/`。

```bash
# 两臂同 seed、同注入；plan 臂不会恢复，是对照
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 10 --seed 300 \
    --plan configs/plans/pick_place_via_basket.json --inject configs/injections/pick_place_move_target.json --name <plan run>
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 10 --seed 300 \
    --agent --max-steps 16 --inject configs/injections/pick_place_move_target.json --name <agent run>
# 抓起后才触发的注入（drop_held）：跑到 10 次触发为止，最多 24 局
$PY -u scripts/plan_episode.py ... --rounds 24 --until-injected 10 --inject configs/injections/pick_place_drop_held.json --name <run>
$PY scripts/injection_report.py --plan <plan run> --agent <agent run> --out outputs/analysis/injection_<日期>
```

- **做什么**：`injection_report.py` 按 seed 配对两份 `manifest.jsonl`，只数两臂都真的触发了注入的局；给出配对 OK 数、不一致对
  （只有 agent 过 / 只有 plan 过）与精确双侧符号检验 p 值，并把 agent 在事件之后的调用串（`名!` 失败、`名?` 被拒）按第一步归类。
- **产物**：trace 里多一行 `{"event": "injection", "before", "commanded", "after", "moved_m", ...}`；manifest 多 `seed`、`injection`、`injected`；
  报告 `report.md` + `pairs.json`。
- **先验一次注入有效**：先跑 plan 臂 3 局，确认注入后固定 plan 确实失败（不失败的注入不是注入），再花 agent 调用。
- 注意：agent 恢复一次至少要 4–6 次调用（`release → approach → grasp_single → approach → place`），`--max-steps` 按 16 起。
- 2026-09-29 第一轮结果（[pick_place §13](ledger/pick_place.md)）：`move_target` 横向 0.17 m 被 `grasp_single` 吸收（它开始时按物体当前位姿重规划），
  沿桌沿 0.60 m 才有效；配对 10 局 plan 0/10、agent 0/10 但 agent 5/10 重新抓起，之后全部在持罐转向时掉罐。
  agent 摸出来的 primitive 事实：`approach` 只修深度不修横向；`nudge` 是量化脉冲且会清 `stance`；`navigate_to` 贴桌沿停、随后的 `turn_to` 会扫桌。
- 2026-09-29 第二轮前补的 primitive（TODO 8.3.1 / 8.3.2）：持物 `approach(goal)` 已面向同一桌沿且站位在 1.2 m 内时**不转向直接横移**（`carry_to(turn_first=False)`）；
  `navigate_to` 的点目标推到离每个台面 ≥ 0.30 m（`edge_margin`）；`turn_to` 离台面 < 0.25 m 时拒绝（`edge_clearance`）；`nudge` 重新量 stance 而不是清掉；
  空手 `approach` 站位偏离走近线 > 0.12 m 时先横移到站位后 0.6 m 的引导点再走。系统提示加了一条：session 快照是本体感觉、场景是真值，冲突时先 `release` 再继续（8.4.2）。
- `--until-injected N`：跑到 N 次注入触发为止（`--rounds` 变成上限），配对数不再被基线抓取率砍半（8.4.1）。
- 第二轮 `move_target`（plan 0/9、agent 0/9、重抓 3/9）暴露的两处，已改：抓完身体带 0.35 rad 的 wrap，横移判据从 0.16 放到 0.6 rad（`STRAFE_YAW`）；
  `place` 自己会转身搬到容器，简单握持在那一转里滑，所以 `place` 现在要求容器在 profile 放置带内（骨盆系 `band_x × band_y` ± 0.10 m，`placing_band` 违规），
  带外先 `approach(container)`；coupled profile 原地放置，免检。
- **2026-09-29 晚的更正**（[pick_place §14](ledger/pick_place.md)）：上面三轮里"持罐转向掉""place 起步滑 3 cm"都不是握持问题，
  是 agent 想的 5–9 s 里 runner 不发帧，deploy 1 s 规划器超时后甩掉手臂；这三轮 agent 臂的结果作废。保活后持罐停 8 s 只漂 0.6–1.2 cm。
- 第四轮（保活后两臂重跑，[pick_place §15](ledger/pick_place.md)）：`move_target` plan 0/10 vs agent 3/10（重抓 5/10），`drop_held` 配对 0/5 vs 1/5；
  合计不一致对 4 : 0，p = 0.125。重抓的罐头跨思考不再掉；剩下的失败在重抓本身、走近、`nudge` 的量化和 `place` 内部。
