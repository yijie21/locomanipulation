# STATE — 2026-10-01（会话：TODO 第 1–3 节做完并提交；上一份 STATE 见 `24c7bce`）

工作目录 `/workspace/code/locomanipulation-loco_agent`（worktree），分支 `loco_agent`，未推送。`outputs/` 是指向主目录 outputs 的符号链接，**stage by name，永远不要 `git add -A`**。
本会话改动：`git log --oneline 24c7bce..HEAD`（`7a6f0dd` 代码/测试/配置/文档，`267d6bb` + 本次 handoff）；`git diff --stat 24c7bce 7a6f0dd` 63 个文件。
没有运行中的栈（已确认：`/workspace/tmp/p03_i*/stack.json` 一个都没有，本会话的 lane 脚本全部结束）。GPU0/GPU1 上现有进程都是别的项目的。

## 进度

- 已确认：TODO 第 1–3 节的原有项全部勾选或带原因放弃；`make check` 全量通过（提交时 pre-commit 也过）。逐项结论在 TODO 条目、DECISIONS 2026-10-01 节、
  crate 台账 2026-10-01 节、`docs/walk.md` 探针表。
- 已确认：新默认值两个——`box_grasp_below_rim` 0.025 → 0.01；`reach_rotation_weight_transit` −1 → 0.2（用户定；单边 warehouse 两组 19/40 vs 10/40，p 0.049）。
  今天的 `s2_*` 对照配置钉回 −1；09-24 单边基线文件不设该字段，此后按 0.2 跑（文件头已注明）。其余新旋钮默认关。
- 已确认：单边基线复跑 6/20（另一 seed 4/20），与 09-24 一致；0 摔。

## 假设 / 待验证

- 待验证：`reach_rotation_weight_transit` 0.2 在 crate 蓝篮子线和 `box_platform` 上不退化（TODO 2.4.10.13.1，同一个伸手控制器，没测过）。
- 待验证：单臂运行目录评测经共用的 `closed_loop.py` 实跑（TODO 1.18.1，缺带权重的单臂 checkpoint）；wall/carry 的 `run()` 只有单元测试（1.17.1）。
- 假设：v1.1 速度图台阶的来源是 `dd36fd8` 之前每帧带 `vr_3pt_*` 字段——已确认的是"09-11 的 agent 代码复现台阶、今天的不复现"，具体哪一行没单独隔离。
- 假设（Q11，沿用）：drop agent 第 23 局 3 次超时是那几分钟 GPU1 被占、sim 出图 30 → 24 Hz，把空手静默拉过 1 s。

## 下一步（TODO 编号）

1. 2.4.10.13.1：在 crate 篮子线 / `box_platform` 上各 20 局对照 0.2 vs −1。
2. Q11 已收尾（2026-10-02，pick_place 台账 §16）。**本分支 `scene_discovery`**：Demo 1 第 0 周，TODO 9.4–9.8（`--record`、场景文件格式、ADT 转换、能力说明、查重）；计划 `docs/handoff/scene-discovery-plan-20261002.md`。ADT 数据在 `/workspace/datasets/adt/`。

## 卡点与坑

- `kill -INT` 停不掉 `plan_episode.py`（很可能是脚本里 `&` 起的进程 SIGINT 被忽略，见 memory background-jobs-ignore-sigint）；按 pid TERM runner + `stack.json` 里的 sim / deploy。之后必查：`ss -ltnp | grep :5565`（sim 的 multiprocessing 子进程会孤儿化占着相机端口，
  下次起栈报 "camera port 5565 is already in use"），以及 `stack.json` 里的 pid 是否还活着（活着的 sim 会让下一次 `up()` 拒绝启动）。
- deploy.log 跨所有 stack 累积：数 `Planner timeout` 前按 `===== deploy start` 分段。
- 在 `plan_episode.main` 外 import `agent_server` 之前先按 `--instance` 钉端口（探针第一次就是连到实例 0，`gap_probe_wrongports_20260929` 作废）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation-loco_agent
export SONIC_DEPLOY_ROOT="$PWD/outputs/bottle_continuous_20260914/deploy" PYTHONPATH="$PWD/src"   # 不要 source scripts/paths.sh
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
# Q11 配对（第四轮队列原样：outputs/runs/q11_round4_queue_20260929.sh，顺序跑四臂 + 两份报告）
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 24 --until-injected 10 --seed 300 \
    --plan configs/plans/pick_place_via_basket.json --inject configs/injections/pick_place_move_target.json --name <plan run>
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 24 --until-injected 10 --seed 300 \
    --agent --max-steps 16 --inject configs/injections/pick_place_move_target.json --name <agent run>
$PY scripts/injection_report.py --plan outputs/runs/<plan run> --agent outputs/runs/<agent run> --out outputs/analysis/<name>
# 找 > 0.3 s 不发帧的位置（抓主线程栈 → skills/command_gaps.jsonl）
$PY -u scripts/experiments/command_gap_probe_20260929.py configs/experiments/skills_pick_place_20260928.yaml --gpu 0 --instance 1 \
    --rounds 4 --seed 500 --plan configs/plans/command_gap_probe_20260929.json --name <run>
make test TESTS="tests/test_skills_*.py"; make check
```

- 起栈前查：`nvidia-smi`、`/workspace/tmp/p03_i1/stack.json`（pid 是否活）、`ss -ltn | grep :5565`、5 秒 CPU 采样（配额 15.36 核）。
- 产物路径（第四轮运行 / 报告 / 预览视频、think8 与探针证据）列在 [pick_place 台账](docs/ledger/pick_place.md) §14 / §15 / §15a。
- 持物停顿测量（`--think-s 8`）的命令见 pick_place 台账 §14；本会话报告脚本 `scripts/experiments/{warehouse_s2_report,fit_phase_band}_20261001.py`。
- 单边 warehouse 单旋钮对照的做法：复制 `box_room_grasp_baseline_20260924.yaml` 改一行、`collect.instance` 各占一个实例，`bash scripts/start_stack.sh collect <yaml> --gpu N`
  再 `bash scripts/bimanual/run_agent_server.sh <yaml> --gpu N`；多组并行时先跑 `scripts/cpu_headroom.sh`，录制的运行要 1800 个 pid 余量（别的项目常占 ~1800）。
