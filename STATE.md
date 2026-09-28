# STATE — 2026-09-28（会话：primitive 契约设计 → `skills` 包 → 四个黑盒包装 → 真栈回归 → `carry_to`）

工作目录 `/workspace/code/locomanipulation-loco_agent`（worktree），分支 `loco_agent`，基于 `9955487`。
上一份 STATE（lockstep + Isaac 评测，`main` 分支那条线）在 `9955487` 里，那条线的未提交改动在主目录 `/workspace/code/locomanipulation`，本 worktree 不碰。
**未提交**：本会话全部代码与文档（`git status` 列出的 22 个路径；TODO 7.12）。
没有运行中的栈；GPU1 上约 3.7 GB 是别的项目的。`outputs/` 是指向主目录 outputs 的符号链接，**stage by name，永远不要 `git add -A`**。

## 当前进度（细节在设计文档 `docs/handoff/primitive-contracts-design-20260928.md` §9a、台账 bottle §40–41 / pick_place §7 / crate 末节）

- 已确认：设计文档定稿，D1–D7 按推荐值定（DECISIONS 2026-09-28 节 1–17）。粒度是"技能"不是"阶段"；profile 把走近参数和抓取参数绑在一起。
- 已确认：`src/locomani/skills/` 建好：session / contract / profiles / scene_model / executor / catalogue；78 个 CPU 测试 + 全仓 `make check` 通过。
- 已确认：四个黑盒都包了。单臂 approach / grasp_single / place 调 bottle 原阶段函数（构造 `RoundContext`，站位用 `plan_for_target + sample_stand`）；
  双臂把 `run_attempt` 每拍逻辑纯提取到 `tasks/crate/attempt.py`，`handoff()` 留在 `expert.py`，29 个双臂离线测试不改一行通过。
- 已确认（真栈，`scripts/plan_episode.py`，GPU1 / instance 1）：bottle plan 3/3 + 10 局 9/10；pick_place plan 1/3；warehouse plan 1/3。三条线的通道交接都走通。
- 已确认：`carry_to` 实现并测量（持瓶）：直走/倒退 1–2 m 漂移 < 1 cm（6/6）；90° 转再走 3/3；180° 转 IDLE 0/2、两次 90° 1/2、SLOW_WALK 4/4 → 默认 `turn_first + slow_walk`。
- 已确认：`grasp_single` 产出 FROZEN（`SimpleGraspPolicy.execute()` 自己交还流）；`place` 跑在 PlannerArm 上，不需要 `take_over`。

## 卡点 / 待验证

- 待验证：`simple` 抓罐头 + place 的标定（1/3：合爪无接触 ×1、carry-over 伺服差 17.5 cm ×1）；§4 的 10/10 是 `coupled` 策略。
- 待验证：warehouse plan 与采集入口 0.30 的对照（只有 3 局）。
- 假设：`carry_to` 的双臂守卫（高度底线 + 双手接触）可用——只在假栈上测过，持箱没量。
- 假设：转向段不需要守卫（`turn_to` 无回调，转完查一次）。
- 已知坑：shell 里 `source scripts/paths.sh` 会导出 instance 0 的端口，agent 对错 deploy 说话（症状 "did not stabilise after reset" + "no body_q_target"）；入口已按 `--instance` 钉端口，启动时**不要 source paths.sh**。
- 已知：`approach` / `walk_to` 拒绝持物状态，所以"抓 → 搬 → 到容器前"的 plan 在 `approach(bin)` 处断（TODO 7.7）。

## 下一步（TODO）

1. 7.7 跨位置搬运回归：放行持物的 approach，标定 place。
2. 7.11 warehouse plan 10 局对照。
3. 7.12 提交本会话代码（先 `make check`）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation-loco_agent
export SONIC_DEPLOY_ROOT="$PWD/outputs/bottle_continuous_20260914/deploy" PYTHONPATH="$PWD/src"   # 不要 source scripts/paths.sh
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
$PY scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml --gpu 1 --instance 1 --rounds 3 --dry-run
$PY -u scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml --gpu 1 --instance 1 --rounds 10 --seed 100 --name <run>
$PY -u scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml --gpu 1 --instance 1 --rounds 3 --plan configs/plans/bottle_carry_turn180slow.json --name <run>
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 3 --name <run>   # approach→grasp→place
$PY -u scripts/plan_episode.py configs/experiments/box_centre_base_20260927.yaml --gpu 1 --instance 1 --rounds 3 --name <run>     # 双臂
make test TESTS="tests/test_skills_*.py tests/test_crate_visual_execution.py"; make check
```

- 起栈前查：`nvidia-smi`、5 秒 CPU 采样（`/sys/fs/cgroup/cpu.stat`，配额 15.36 核，> 11 核在忙就别起）、`/workspace/tmp/p03_i1/stack.json` 是否存在。
- 每次运行产物：`outputs/runs/<run>/skills/{manifest.jsonl,agent_trace_NN.jsonl,round_NN/}`、`videos/`（入口自 carry 运行起录像）。
- 预览：`outputs/videos/skills_carry_20260928/`；plan 文件：`configs/plans/`。
