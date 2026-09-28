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

## 3. 目录（agent 能调什么）

| primitive | 参数（省略取默认） | 从哪个通道进 → 出 | 做什么 / 状态 |
|---|---|---|---|
| `approach(target, profile)` | — | swing/frozen → swing（guard 走近）或 frozen（持物） | 单臂：朝桌法线、守线刹车、写站位；双臂：走入 + 刹车。**持物时**：`carry_to` 到目标的规划站位再原地转向桌法线（放置前走到容器） |
| `grasp_single(target, profile, side)` | `side=right` | swing → frozen | 走 profile 的 `reach.strategy`：`simple`（bottle 10 局 9/10）或 `coupled`（罐头，pick_place §12）；其余策略在 `approach` 前置被拒 |
| `ready(profile)` / `grasp_bimanual(target, profile)` | — | → frozen / frozen → pose_stream | 双臂原地抬手；协同伸抓到 hold |
| `place(container, profile, side)` | — | frozen → frozen | 抬、转到放置带、送、降、松、抬开、后退、验收 |
| `put_down_bimanual(profile, surface, lower_speed, release_seconds, raise_m, back_m)` | `surface` 默认物体原表面 | pose_stream → pose_stream | 双手同步下放到表面静止高度 → 松手 → 抬开；判据只看物体（高度、倾角、无手接触）。**首个实现，2026-09-28 真栈样本见台账** |
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
