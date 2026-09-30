# STATE — 2026-09-30（会话：TODO 8.3.5 掉罐机制 → 保活 → Q11 第四轮 → 空档定位；上一份 STATE 见 `7cab2b9`）

工作目录 `/workspace/code/locomanipulation-loco_agent`（worktree），分支 `loco_agent`。`outputs/` 是指向主目录 outputs 的符号链接，**stage by name，永远不要 `git add -A`**。
没有运行中的栈（已确认：instance 1 的 runner / sim / deploy 全部 TERM，5565 端口空）。GPU0/GPU1 上现有进程都是别的项目的（depth_server、UnrealEditor、SimpleRoboJev）。
代码与文档已提交 `df1d420`；细节在 pick_place 台账 §14 / §15 / §15a、DECISIONS 2026-09-30 节、TODO 8.3.5。

## 进度

- 已确认：§13 三轮"重抓后掉罐"是 deploy 的 1 s 规划器超时（`zmq_manager.hpp` `PLANNER_TIMEOUT`），不是抓握蠕变。`--think-s 8` 静默 4/4 在 ~1 s 掉、保活 0/14 掉；
  真实蠕变 0.10 cm/s（`object` 档）/ 0.06（`crate` 档）。硬指垫不落地（DECISIONS 2026-09-30 决策 2）。
- 已确认：`Executor.idle` 保活已实现并提交（agent 提问、注入 settle、`--think-s` 都走它）；第四轮 agent 臂的 deploy 段超时 1 / 3 次（前三轮 110 / 84 / 22）。
- 已确认（Q11 第四轮，seed 300 起，`--until-injected 10`，sonnet，16 步）：
  - `move_target` +0.60 m：plan 0/10 vs agent 3/10，10 对，不一致 3 : 0，p = 0.25；agent 重抓 5/10。
  - `drop_held`：plan 0/19（10 触发）vs agent 1/24（9 触发），5 对，1 : 0，p = 1.0；重抓 3/9；注入后第一步 9/9 `release`。
  - 合计不一致对 4 : 0，p = 0.125。重抓的罐头跨每次思考都还在手里（1 例浅抓 z 0.84 除外）。
  - 4 个 agent 成功里 2 个不干净（`place` 报失败后罐头落进篮子，判据算成功）。
- 已确认：剩余的规划器静默只在空手 `approach` 内，用户定不修（8.3.5.5.1）；8.3.1 / 8.3.4 的 primitive 保留（8.3.5.4）。

## 假设 / 待验证

- 假设：drop agent 第 23 局 3 次超时是那几分钟 GPU1 被 UnrealEditor 占用、sim 出图 30 → 24 Hz，把 0.5–0.9 s 的空手静默拉过 1 s（位置由探针确定，拉长原因没有栈证据）。
- 待验证：agent 相对 plan 的优势（4 : 0）在更多配对下是否显著；按第四轮成功率估，每种注入约需 20 对以上。
- 已知上限（第四轮观察，非对照实验）：agent 失败大多没走到持物——重抓失败并碰倒罐头、`approach` 拒绝或失败、`nudge` 6 脉冲后仍偏 8–15 cm；
  持物后在 `place` 内部或持物横移里丢罐。

## 下一步（TODO）

1. 8.6 待用户定：Q11 下一步（加配对数 / 先提 `grasp_single`、`place` 成功率 / 就此收尾写结论）。
2. 8.5 未做：推机器人 / 施力需要 sim 侧 xfrc 命令；放置点被占需要第三个物体。

## 卡点与坑（沿用 + 本会话新增）

- `kill -INT` 停不掉 `plan_episode.py`；按 pid TERM runner + `stack.json` 里的 sim / deploy。之后必查：`ss -ltnp | grep :5565`（sim 的 multiprocessing 子进程会孤儿化占着相机端口，
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
# 持物停顿测量（保活 / --think-silent 复现旧空档）
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 10 --seed 400 \
    --plan configs/plans/pick_place_via_basket.json --think-s 8 --name <run>
# 找 > 0.3 s 不发帧的位置（抓主线程栈 → skills/command_gaps.jsonl）
$PY -u scripts/experiments/command_gap_probe_20260929.py configs/experiments/skills_pick_place_20260928.yaml --gpu 0 --instance 1 \
    --rounds 4 --seed 500 --plan configs/plans/command_gap_probe_20260929.json --name <run>
make test TESTS="tests/test_skills_*.py"; make check
```

- 起栈前查：`nvidia-smi`、`/workspace/tmp/p03_i1/stack.json`（pid 是否活）、`ss -ltn | grep :5565`、5 秒 CPU 采样（配额 15.36 核）。
- 产物路径（第四轮运行 / 报告 / 预览视频、think8 与探针证据）列在 [pick_place 台账](docs/ledger/pick_place.md) §14 / §15 / §15a。
