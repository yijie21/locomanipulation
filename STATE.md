# STATE — 2026-10-05（会话：检查器按实测设定整体匹配、Claude 提任务循环跑通、报告 doc；上一份 STATE 见 `def96be`）

工作目录 `/workspace/code/locomanipulation-scene_discovery`（worktree），分支 `scene_discovery`（基于 `loco_agent` 的 `023a31a`），未推送。
`outputs/`、`third_party/`、`datasets/` 是指向主目录的符号链接，**stage by name，永远不要 `git add -A`**。未跟踪的 `.serena/` 不是本线的，别提交。
本会话改动：`git log --oneline def96be..HEAD`（`06afba2` 代码 / 测试 / 配置 / 文档 + 本次 handoff）；`git diff --stat def96be 06afba2` 15 个文件。
设计与决策：`docs/handoff/scene-discovery-plan-20261002.md`；DECISIONS 2026-10-05 两节。报告 doc（私有，TODO 9.17）：
https://claude.ai/code/artifact/132adcd3-de5d-4ff1-99ba-1bc683f47549

## 进度

- 已确认：TODO 9 补了 9.13（采集吞吐）、9.14（补渲 → 转换 → 训练 → 三格评测），用户 2026-10-05 要求。
- 已确认：检查器单手抓取按 `capability.yaml grasp_settings` 整体匹配（瓶子线 + 咖啡罐推车 4/10）；失败试跑记成技能缺口
  （`deep_entry` inboard ≥ 0.14 且抓点离板 ≤ 0.08，`deep_reach` inboard ≥ 0.26）；头顶有板记 `*.covered`。
  模板空间 2830 个：桶 2 / 1144 / 1684，第 1 桶 = 拿起咖啡罐、罐子放到圆桌；可试跑 85。
- 已确认：只有一边放得进夹爪的物体，出生朝向限制在长边沿走近方向 ±20°（`plan_episode.spawn_object_yaw`）；单元测试过，**未上栈跑**。
- 已确认：放进容器回归 `sd_place_regress_20261005` 抓取 6/10、place 3/6，与 pick_place 台账 §8 相同（§17）。
- 已确认：Claude 提任务循环 `scripts/scene_discover_tasks.py`，Opus 一次运行 `discovery/opus_20261005b/`：49 个任务，
  按改后阈值重检（`recheck_20261005.json`）桶 2 / 38 / 9；47/49 落在模板空间内。3 次调用 5.32 USD、20.5 min。
  `opus_20261005_upright_bug` 作废（检查器 upright bug，已修）。
- 已确认：`make check` 全过；`06afba2` 提交时 pre-commit 全量通过。没有运行中的栈或 discovery 进程（instance 1 空）。

## 假设 / 待验证

- 待验证：±20° 出生朝向能把咖啡罐抓取率从约 0.24 提上去多少（9.12.11 的依据是 28 局分组统计，≤ 20° 也有败局）。
- 假设：`overhead.open_above` 0.60 m 是规则，没测过。
- 假设：采集一轮约 200 h（按整局 13/80、每局 48 s 外推，9.13）。
- 待查：26/64 个自由体的 `upright_axis` 与实际静止姿态差 75–125°（9.15）；书已查清（ADT 正立 = 平放），其余未查。
  后果：目标里写 `upright(X)` 的局在运行时会误判失败。
- 我的判断，未测：扁平物体（钱包、钥匙、木勺）要"从上往下捏"（9.16.1）。

## 下一步（TODO 编号）

1. 9.10.3：试跑 6 个只缺实测的提议任务（罐子 → 岛台 / 垃圾桶 / 餐桌 / 长凳 / 玻璃柜，大花瓶 → 马桶水箱），各 10 局约 1 h GPU——**待用户定**。
2. 9.16：第二个新技能（推荐台面高度抓取）——**待用户定**；9.15 先看一张盆栽 / 书的渲染再定修法。
3. 9.10.2 → 9.14.1：任务文件接上 `plan_episode`（谓词合取 + 说法），再拿"罐子放到圆桌"录约 10 局走通录制 → 补渲 → 转换 → 短训 → 闭环。

## 卡点与坑

- 停队列 / discovery：按 pid TERM 外层脚本 → runner（discovery 的 `claude -p` 子进程另杀）→ `stack.json` 里的 sim / deploy，查 `ss -ltnp | grep :5565`；`kill -INT` 无效。
- 提交前对新文件跑 `/venv/main/bin/ruff format <files>`；失败区域阈值取整到失败试跑本身落在区域内（浮点边界）。
- Isaac 脚本要 `ISAAC_ASSETS=/workspace/code/locomanipulation/assets/isaac`；起栈前查 `nvidia-smi`、`stack.json`、`bash scripts/cpu_headroom.sh`（别的项目常占 5–9 核）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation-scene_discovery
export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/outputs/bottle_continuous_20260914/deploy PYTHONPATH=$PWD/src TMPDIR=/workspace/tmp
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python    # 不要 source scripts/paths.sh
A=outputs/scene_discovery/adt_apartment                        # ADT 衍生，只在 outputs
# 检查器（模板空间）：写 $A/feasibility/{tasks.jsonl,summary.json}
$PY scripts/scene_feasibility.py --scene $A/scene_meta.json
# Claude 提任务（Opus，约 20 min / 5 USD）：写 <out>/{tasks.jsonl,iteration_NN.jsonl,claude_io.jsonl,report.json}
$PY -u scripts/scene_discover_tasks.py --scene $A/scene_meta.json --model opus --iterations 2 --per-room 30 --out $A/discovery/<name>
# 单任务试跑（实验文件在 $A/experiments/，例：罐子放到圆桌）
$PY -u scripts/plan_episode.py $A/experiments/p4_canister_on_round_table.yaml --gpu 1 --instance 1 --rounds 10 --seed 700 --name <run>
python3 scripts/experiments/put_down_acceptance_20261003.py outputs/runs/<run>*
make test TESTS="tests/test_scene_discovery_feasibility.py tests/test_scene_discovery_proposals.py tests/test_plan_episode_spawn.py"; make check
```
