# STATE — 2026-10-01（会话：TODO 第 1–3 节按顺序做完；上一份 STATE 见 `24c7bce`，Q11 的部分原样留在下面）

工作目录 `/workspace/code/locomanipulation-loco_agent`（worktree），分支 `loco_agent`。`outputs/` 是指向主目录 outputs 的符号链接，**stage by name，永远不要 `git add -A`**。
没有运行中的栈（已确认：`/workspace/tmp/p03_i*/stack.json` 一个都没有，本会话的 lane 脚本全部结束）。GPU0/GPU1 上现有进程都是别的项目的。
本会话改动已提交（用户 10-01 说提交）；`make check` 全量通过。细节：TODO 1.x / 2.x / 3.x 各条、DECISIONS 2026-10-01 节、
crate 台账 2026-10-01 节、`docs/walk.md` 探针表新三行。

## 进度

- 已确认（第 1 节）：同 venv spawn 全部改进程内（1.15、1.16）；wall/carry 拆成 `run(config, ...)`、去掉 `DUMP_DIR`/`GHOST_TRACE` 全局（1.17）；
  两个评测器共用 `pipeline/closed_loop.py`（1.18）；中断续跑实机演练通过（1.20.10，`drill_resume3_20261001`）。
  1.16 闸门：cap120 checkpoint-5000 同 seed 10 局，旧 4/10、新 5/10；1.18 双臂运行目录实跑 4 局 3/4。单臂运行目录没法实跑：磁盘上没有带权重的单臂 checkpoint。
- 已确认（第 2 节，单边 warehouse，20 局一组）：基线复跑 6/20（另一 seed 4/20）。九个单旋钮里只有 `reach_rotation_weight_transit 0.2`
  换 seed 复制后同向（合并 19/40 vs 10/40，p 0.049）；其余负或判不动（台账表）。跨壁漂移 = SONIC 只送达约一半跨壁命令 + 骨盆横漂。
  v1.1 速度图台阶的来源是 09-11 的 agent 代码（`dd36fd8` 前每帧带 `vr_3pt_*`），不是部署。
- 已确认（第 3 节）：相位刹车稳态 0.55 有效（30/30 不越线），warehouse 这条边助跑不够（仓库 1/10）；3.11 用闭环 81/100 关闭。
- 新默认值两个：`box_grasp_below_rim` 0.025 → 0.01；`reach_rotation_weight_transit` −1 → 0.2（用户定）。其余新旋钮默认关。

## 假设 / 待验证

- 待验证：`reach_rotation_weight_transit` 默认 0.2（用户 10-01 定）在 crate 篮子线、`box_platform` 上没测过。
- 假设（Q11，沿用）：drop agent 第 23 局 3 次超时是那几分钟 GPU1 被占、sim 出图 30 → 24 Hz，把空手静默拉过 1 s。
- 待验证（Q11，沿用）：agent 相对 plan 的优势（4 : 0）在更多配对下是否显著。

## 下一步（TODO）

1. 8.6 待用户定：Q11 下一步（加配对数 / 先提 `grasp_single`、`place` 成功率 / 就此收尾写结论）。
2. 8.5 未做：推机器人 / 施力需要 sim 侧 xfrc 命令；放置点被占需要第三个物体。

## 卡点与坑（沿用 + 本会话新增）

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
