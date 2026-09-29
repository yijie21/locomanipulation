# STATE — 2026-09-29（会话：Q11 失败注入 paired 比较，TODO 8；上一份 STATE 见 `1a2a588`）

工作目录 `/workspace/code/locomanipulation-loco_agent`（worktree），分支 `loco_agent`。`outputs/` 是指向主目录 outputs 的符号链接，**stage by name，永远不要 `git add -A`**。
没有运行中的栈（两条配对队列都跑完，`plan_episode.py` 自己收栈）。

## 本会话做了什么（细节：docs/skills.md §7、pick_place 台账 §13、TODO 8）

- 已确认：TODO 7 无剩余项；立 TODO 8（Q11）。注入层 `src/locomani/skills/injection.py`（`move_target` / `drop_held`，某 primitive 第一次成功后 `pin_object` 0.5 s 再松开，一局一次，agent 不被告知）、`plan_episode.py --inject`、`Executor.note`、manifest 记 `seed`/`injection`/`injected`、`scripts/injection_report.py`（按 seed 配对、精确符号检验、恢复段归类、`regrasped`）、6 个 CPU 测试、`configs/injections/`。
- 已确认（真栈）：`move_target` 横向 0.17 m **无效**（`grasp_single` 开始时按当前位姿重规划；1/3，失败是基线族）；沿桌沿 +0.60 m 有效（plan 0/3、0/10 全 `[grasp]`）。
- 已确认（配对 10 局，seed 300–309）：plan 0/10 vs agent 0/10（p = 1.0），agent **5/10 重新抓起**；重抓后 5/5 在持罐 `approach(basket)` 转向时掉罐（simple 握持转向滑），3/10 贴桌沿 `turn_to` 扫罐下桌，2/10 用完 16 次调用。Q11 的区分度卡在 primitive：持物横移、贴桌沿转向、`nudge` 清 stance、`approach` 不修横向（TODO 8.3.1 / 8.3.2，待用户定）。
- 已确认（配对 `drop_held`，台账 §13 末）：注入只在抓起后触发 → 各臂 5 个注入、3 个配对；plan 0/10、agent 0/10，agent 重抓 2/5，第一反应 5/5 直接 `place`（信 `held` 不信场景）。
- 已提交 `1ecdfdc`（注入层 + move_target 结果）；drop_held 结果与 8.4.x 在本次提交。

## 卡点 / 待验证

- 已知：配对数受基线抓取率限制（8.4.1）；agent 对 `held` 过期的对账靠 place 的守卫报错才发生（8.4.2 待用户定）。
- 已知：`nudge` 会清 `stance`，agent 每次 nudge 后再抓都被拒；是否让 grasp 容忍 nudge 后的小位移待用户定。
- 已知坑（沿用）：`pgrep -f` / `pkill -f` 在 shell 链里自匹配（本会话两次中招：等待循环永不退出、pkill 杀掉自己的 shell）；用脚本文件 + `grep -v pgrep`。

## 下一步（TODO）

1. 8.3.1 / 8.3.2 待用户定：补哪个 primitive 才能让配对比较有区分度（推荐先做持物横移不转向 + `navigate_to` 离桌沿 0.3 m）。
2. 8.5 未做：推机器人 / 施力需要 sim 侧 xfrc 命令；放置点被占需要第三个物体。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation-loco_agent
export SONIC_DEPLOY_ROOT="$PWD/outputs/bottle_continuous_20260914/deploy" PYTHONPATH="$PWD/src"   # 不要 source scripts/paths.sh
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 10 --seed 300 --plan configs/plans/pick_place_via_basket.json --inject configs/injections/pick_place_move_target.json --name <run>
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 10 --seed 300 --agent --max-steps 16 --inject configs/injections/pick_place_move_target.json --name <run>
$PY scripts/injection_report.py --plan <plan run> --agent <agent run> --out outputs/analysis/injection_<日期>
make test TESTS="tests/test_skills_*.py"; make check
```

- 起栈前查：`nvidia-smi`、`pgrep -af scripts/plan_episode.py | grep -v pgrep`、`/workspace/tmp/p03_i1/stack.json`。
- 每次运行产物：`outputs/runs/<run>/skills/{manifest.jsonl,agent_trace_NN.jsonl[,agent_io_NN.jsonl]}`、`videos/`；注入事件是 trace 里的 `{"event":"injection"}` 行。
