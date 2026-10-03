# STATE — 2026-10-03（会话：scene_discovery Demo 1 第 0 周 + 第一轮四个范围试跑；loco_agent 那边的 STATE 停在 `023a31a`）

工作目录 `/workspace/code/locomanipulation-scene_discovery`（worktree），分支 `scene_discovery`（基于 `loco_agent` 的 `023a31a`），未推送。
`outputs/`、`third_party/`、`datasets/` 是指向主目录的符号链接，**stage by name，永远不要 `git add -A`**。
本会话改动：`git log --oneline 023a31a..HEAD`（9 个提交 + 本次 handoff）；loco_agent 上另有 `9d777c9`（技能去重 15 → 13）、`023a31a`（Q11 收尾）。
设计与决策：`docs/handoff/scene-discovery-plan-20261002.md`（19 项 + §4a 第一轮 C 方案）、`docs/handoff/scene-file-format-20261003.md`、DECISIONS 2026-10-02 / 10-03 三节。

## 进度

- 已确认：第 0 周 9.1–9.7 完成；9.4 `--record` 实跑通过（bottle 默认 plan 20 局 15 存 5 弃，与 manifest 逐行对上；与 bottle 采集线逐字段一致）。
- 已确认：ADT 公寓 → `outputs/scene_discovery/adt_apartment/`（v2 `scene_meta.json` + `scene_locomani.xml`）；2 个房间、71 个台面（35 个可走近）、
  353 个物体 + 外壳 `ApartmentEnv`（64 个自由体，47 个跌落测试停住）；Isaac 舞台在主仓库 `assets/isaac/scene/adt_apartment`（自由体 USD 已按新原点重导）。
- 已确认：模板任务 2830 个（严格口径）：第 1 桶 0、第 2 桶 1200（可试跑 72，挡在技能缺口 1169——"放到台面" 与 "抓取高度 0.66–0.89 m 之外" 为主）、第 3 桶 1630。
  数字来自 `outputs/scene_discovery/adt_apartment/feasibility/{summary.json,pilot_groups.json}`。
- 已确认：第一轮四个范围试跑（`outputs/runs/sd_pilot_*_20261003c`，各 10 局）：咖啡罐 4/10（推车顶 0.70 m、开放、离边 4 cm）；
  中号花瓶 0/10、小盆栽 1/10（有顶格子 0.73 m，手指入手时扫到柜体）；青苹果 0/10（离边 26.5 cm，从没停到够得着的位置）。写进 capability.yaml `pilots_adt_20261003`。
- 已确认：之前两轮试跑（`*_20261003`、`*_20261003b`）作废：一个是出生点检查用错图，一个是物体原点在底部（DECISIONS 2026-10-03 第 4、6 条）。

## 假设 / 待验证

- 假设：咖啡罐 4/10 与罐头在 0.62 m 桌上的 6/10 "同量级"——都是 10 局，区分不了（10 局 pilot 的噪声见 memory ten-round-pilots-cannot-measure-this-line）。
- 待验证："单手放到台面"成本最低（只看了现有 `place` 的阶段，没动手）；"有顶格子""深够"各挡多少任务（9.9.9）；plan 录的数据训出来不输采集线（9.4.1.1，只比了字段）。
- 假设：Claude 估的质量 / 摩擦大致可信——只做了跌落测试（停不停得住），没有实测值对照。

## 下一步（TODO 编号）

1. 9.9.7 待用户定第一个新技能（推荐"单手放到台面"，由 Claude 自己造）；9.9.8 待用户定咖啡罐是否现在就采。
2. 9.9.9 把两个新缺口加进检查器并重算排序；9.8 查重（scoop_check + 文献）还没开始。
3. 9.10 Claude 提任务 + 检查反馈循环（第 2 周）。

## 卡点与坑

- GPU0 上有别的会话的 instance 0 **录制**栈（`warehouse_mars_smoke`，2026-10-03 下午）；我们的试跑都在 GPU1 / instance 1。`--record` 必须用 instance 0：先确认它空着。
- 停队列：先 TERM 外层脚本，再 TERM runner，再查 `stack.json` 里的 pid 和孤儿相机子进程（`ss -ltnp | grep :5565`，环境里 `INSTANCE=1`）。
  `pkill -f <串>` 会连带杀掉含同一串的监视脚本；`kill -INT` 停不掉 plan_episode（SIGINT 被忽略）。
- worktree 里没有 `assets/isaac`：Isaac 脚本要 `ISAAC_ASSETS=/workspace/code/locomanipulation/assets/isaac`。`/workspace/tmp/p03_i1/stack.json.stale_20261003` 可删。
- ADT 数据与签名链接在 `/workspace/datasets/adt/`；ADT 衍生物（场景文件、任务清单、试跑实验文件）只放 `outputs/`，不进 git（许可）。根分区 ~2 GB：设 `TMPDIR`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation-scene_discovery
export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/outputs/bottle_continuous_20260914/deploy PYTHONPATH=$PWD/src TMPDIR=/workspace/tmp
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python    # 不要 source scripts/paths.sh
S=outputs/scene_discovery/adt_apartment/scene_meta.json
# 场景各阶段（geometry / estimate / mujoco / refresh）的用法在 scripts/adt_to_scene.py 文件头；refresh 原地重算规则字段、保住 Claude 字段
/venv/main/bin/python scripts/adt_to_scene.py refresh --scene $S && /venv/main/bin/python scripts/adt_to_scene.py mujoco --scene $S
$PY scripts/scene_drop_test.py --scene $S && $PY scripts/scene_feasibility.py --scene $S   # → feasibility/{tasks.jsonl,summary.json}
blender --background --python scripts/adt_to_usd.py -- $PWD/$S /workspace/code/locomanipulation/assets/isaac/scene/adt_apartment
# 试跑（实验文件在 outputs/scene_discovery/adt_apartment/experiments/，队列脚本 pilot_queue_20261003c.sh）
$PY -u scripts/plan_episode.py outputs/scene_discovery/adt_apartment/experiments/p3_coffee_canister.yaml --instance 1 --gpu 1 --rounds 10 --seed 0 --name <run>
python3 scripts/experiments/pilot_summary_20261003.py outputs/runs/sd_pilot_*_20261003c
# 录成训练数据 + 转换 + 与采集线逐字段比
$PY -u scripts/plan_episode.py configs/experiments/bottle_simple_200.yaml --instance 0 --gpu 1 --rounds 20 --seed 100 --record --name <run>
bash scripts/convert_data.sh outputs/runs/<run>
make test TESTS="tests/test_scene_discovery*.py tests/test_plan_episode_record.py"; make check
```
起栈前查：`nvidia-smi`、`ls /workspace/tmp/p03*/stack.json`、`ss -ltn | grep -E ':55(5|6)5 '`、5 秒 CPU 采样（配额 15.36 核，> 11 核别起）。
