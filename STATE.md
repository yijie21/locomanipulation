# STATE — 2026-09-28 下午（会话：TODO 7 推到底 + 三个标定项 7.7.1 / 7.8.1 / 7.8.2）

工作目录 `/workspace/code/locomanipulation-loco_agent`（worktree），分支 `loco_agent`，基于 `079619f`。
上一份 STATE（`skills` 包搭起、四个黑盒包装、`carry_to` 测量）在 `079619f`。`outputs/` 是指向主目录 outputs 的符号链接，**stage by name，永远不要 `git add -A`**。
没有运行中的栈（最后一个队列运行结束后 `plan_episode.py` 自己收栈）；GPU1 上约 8 GB 是别的项目的。

## 本会话做了什么（细节：设计文档 §9b、台账 pick_place §8–9 / crate 末节、DECISIONS 2026-09-28（晚））

- 已确认：TODO 7.4 / 7.6 / 7.7 / 7.9 / 7.10 / 7.11 / 7.12 / 7.8.2 关闭；7.7.1 / 7.8.1 做到数据说"旋钮够不着"为止（各留一个待用户定的子项）。
- 已确认（真栈，GPU1 / instance 1，`scripts/plan_episode.py`）：
  - 持物 `approach(basket)` → `place`：三轮 10 局。抓取站位版 2/6、放置带中心版 1/7、**中心 + 残差修正版 3/6**（容器 6/6 进放置带）；对照直接 place 0/5。
  - warehouse plan `approach → ready → grasp_bimanual` **4/10**（采集入口 6/20，同噪声带）。
  - `put_down_bimanual` 7 个样本 6 次放稳（≤ 0.7 cm、≤ 2.5°）；判据改看箱底后唯一样本暴露：站直把箱子拉近 0.2 m，原地放下搁在台面近沿（TODO 7.8.1 前送）。
  - 持箱 `carry_to`（180° slow_walk 转 + 1 m）3/6；不转倒退 1 m 2/2（TODO 7.8.2 默认走法待定）。
  - `navigate_to` 持罐 1 局：7 段 2.57 m 绕桌角，漂移 0.8 cm（pick_place §10）；空手真栈未跑。
  - `--agent`（`claude -p`，sonnet）3 局：管路通，每局 10 次调用 200–300 s，失败都是物理失败（抓倒、掉罐）。
- 已确认：准入带回放 932 条站位——任何深度 / 横向带都不分成功与失败，`STANCE_BANDS` 全开。
- 已确认（下午，标定三项）：
  - 7.7.1 罐头：三个 simple 单旋钮 pilot（横向容差、下降减速、进入过冲）各 10 局都不优于基线（grasp 6/10、place 3/6）；stats 证明合爪前已对准；`coupled` 未被包装（approach 前置现在拒绝）。
  - 7.8.1 放下前送：臂前送 1/3（伸臂松捏持）、腿前送 carry 3/11（`walk_to` 走不了 0.15 m 小步，冲 0.7–1.5 m 撞台面）；放下本身合计 8/10 放稳。卡在小步走法。
  - 7.8.2 持箱 carry：倒退不转 8/12 vs 转 3/6，但两种都以约 2 cm/s 下滑（蠕变）；默认不改，持箱 ≤ 1 m 可用。
- 已确认：`make check` 通过（提交前再跑一次）；`tests/test_skills_*.py` 100+ 个 CPU 测试。

## 卡点 / 待验证

- 待用户定：7.7.1.1 包装 `coupled` 策略或给 simple 加闭合几何旋钮；7.8.1 的 ≤ 0.2 m 小步 primitive（IDLE 挪步）。
- 已知：持箱 carry 的下滑不是转向问题（memory `bimanual-carry-creep`）。
- 假设：`put_down_bimanual` 的下放高度只按场景表面顶算；搬到台面之外的地方放（地面）没有表面项，需要给 `surface` 一个"floor"。
- 已知：warehouse plan 运行无视频（`backend.render: offline`）；pick_place 运行有，预览在 `outputs/videos/skills_batch2_20260928/`。
- 已知坑（沿用）：不要 `source scripts/paths.sh`；Bash 后台任务 600 s 上限会杀队列，用 `nohup setsid` + 日志 Monitor。

## 下一步（TODO）

1. 7.7.1.1（包装 coupled 或加闭合旋钮）与 7.8.1 的小步 primitive，先由用户定方向。
2. Q11 失败注入的 paired 比较设计（agent 只在失败恢复上有不可替代的贡献；3 局冒烟里没有可恢复的失败）。
3. put_down 的地面放置（`surface` 没有 floor 项）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation-loco_agent
export SONIC_DEPLOY_ROOT="$PWD/outputs/bottle_continuous_20260914/deploy" PYTHONPATH="$PWD/src"   # 不要 source scripts/paths.sh
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 10 --seed 300 --plan configs/plans/pick_place_via_basket.json --name <run>
$PY -u scripts/plan_episode.py configs/experiments/skills_pick_place_20260928.yaml --gpu 1 --instance 1 --rounds 3 --agent --name <run>
$PY -u scripts/plan_episode.py configs/experiments/box_centre_base_20260927.yaml --gpu 1 --instance 1 --rounds 10 --seed 7000 --plan configs/plans/warehouse_put_down.json --name <run>
$PY scripts/replay_preconditions.py --out outputs/analysis/precondition_replay_<日期>
make test TESTS="tests/test_skills_*.py"; make check
```

- 起栈前查：`nvidia-smi`、5 秒 CPU 采样（配额 15.36 核）、`/workspace/tmp/p03_i1/stack.json` 是否存在。
- 每次运行产物：`outputs/runs/<run>/skills/{manifest.jsonl,agent_trace_NN.jsonl[,agent_io_NN.jsonl]}`、`videos/`。plan 文件：`configs/plans/`。
