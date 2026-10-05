# STATE — 2026-10-05（会话：第一个新技能 `put_down_single` 造完、验收通过、已提交 `4da2bba`）

工作目录 `/workspace/code/locomanipulation-scene_discovery`（worktree），分支 `scene_discovery`（基于 `loco_agent` 的 `023a31a`），未推送。
`outputs/`、`third_party/`、`datasets/` 是指向主目录的符号链接，**stage by name，永远不要 `git add -A`**。
本会话改动：`git log --oneline 695b531..HEAD`（`4da2bba` + 本次 handoff）；`git diff --stat 695b531 4da2bba` 23 个文件。
设计与决策：`docs/handoff/scene-discovery-plan-20261002.md`、DECISIONS 2026-10-02 / 10-03 四节；本技能的规格、门槛、六版逐版证据：
`docs/handoff/put-down-single-20261003.md`。

## 进度

- 已确认：用户定第一个新技能 = "单手放到台面"（Claude 造，9.9.7）；咖啡罐暂不采（9.9.8）。
- 已确认：`put_down_single` 验收 **12/20 通过**（门槛 ≥ 10/20，跑之前写定）：罐子从 0.70 m 推车搬到 0.52 m 圆桌，抓取成功的前 20 局里走近台面 + 放下成功；
  第六版 80 局（seed 600–679），门槛外 2 局另计共 13/22；六版共 171 局、约 2.3 h。成功局倾角 0.3–0.4°，离点 0.1–3.5 cm。
- 已确认：capability.yaml `place.on_surface_single` 写回（measured_one）；检查器重跑：可试跑 72 → 264，剩下唯一的技能缺口 `grasp_single.grip_height`（882），
  第 1 桶仍 0（9.12.8）。改前结果另存 `outputs/scene_discovery/adt_apartment/feasibility/*_20261003_before_put_down.*`。
- 已确认：`plan_episode` 出生点要离所有台面边沿 ≥ 0.45 m（改了所有 scene_discovery 场景的出生距离，罐子 1.2 → 0.8 m）。
- 已确认：13 局成功预览在 `outputs/scene_discovery/adt_apartment/put_down_v6_success_previews_20261003/`（+ `.zip`），已逐个核对大小 / 解码 / 时长。
- 已确认：`make check` 全过；提交时 pre-commit 全量通过。instance 1 已空出，没有运行中的栈。

## 假设 / 待验证

- 待验证：放进容器（`place`）在 `place_phase` 改动后没有实跑过，只有单元测试（9.12.10）。
- 假设：交接后底盘仍会移动、蹲时后移 7–13 cm 是规划器 IDLE / IDEL_SQUAT 的固有行为（只有 `poses` 与 `place_servo` 的相关性，没隔离）。
- 未定位："腰没回中"（`[stand] neutral waist not delivered`）在验收里频繁出现，前一局成功 / 失败 / 抓取败之后都有（9.12.7）。
- 假设：抓住与否主要由罐子随机朝向决定（28 局的分组统计，|a| ≤ 20° 6/9 也有失败，不是唯一因素；9.12.11）。

## 下一步（TODO 编号）

1. 9.12.10 pick_place 跑 10 局，确认放进容器没被 `place_phase` 改坏。
2. 9.12.4 / 9.12.5：蹲后按当前位姿再选点；`approach(台面)` 持物转身掉罐 3/20。
3. 9.9.9 两个新缺口进检查器；9.8 查重；9.10 Claude 提任务的反馈循环。

## 卡点与坑

- 停队列：先 TERM 外层脚本，再 TERM runner，再按 `/workspace/tmp/p03_i1/stack.json` 的 pid TERM sim / deploy，杀掉孤儿相机子进程
  （`ss -ltnp | grep :5565`），把 `stack.json` 挪开。`kill -INT` 停不掉 plan_episode。
- `pgrep -f <串>` 会匹配到含同一串的监视命令本身（本会话两个 Monitor 因此不退出）；用 pid 或日志判断结束。
- zsh 里 `rm -f dir/*.png` 在没有匹配时整条命令中止；批处理写成 bash 脚本文件再跑。
- 验收队列脚本 `outputs/scene_discovery/adt_apartment/put_down_accept_queue_20261003.sh <run 前缀> <seed 起点>`：每批 20 局，抓住 20 局即停。
- worktree 里没有 `assets/isaac`：Isaac 脚本要 `ISAAC_ASSETS=/workspace/code/locomanipulation/assets/isaac`。根分区小：设 `TMPDIR=/workspace/tmp`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation-scene_discovery
export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/outputs/bottle_continuous_20260914/deploy PYTHONPATH=$PWD/src TMPDIR=/workspace/tmp
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python    # 不要 source scripts/paths.sh
E=outputs/scene_discovery/adt_apartment/experiments/p4_canister_on_round_table.yaml   # on(罐子, 圆桌)，ADT 衍生只在 outputs
$PY scripts/plan_episode.py $E --gpu 1 --instance 1 --rounds 3 --dry-run
$PY -u scripts/plan_episode.py $E --gpu 1 --instance 1 --rounds 10 --seed 700 --name <run>
python3 scripts/experiments/put_down_acceptance_20261003.py outputs/runs/<run>*        # 门槛统计 + 失败分族
$PY scripts/scene_feasibility.py --scene outputs/scene_discovery/adt_apartment/scene_meta.json   # 重跑检查器
make test TESTS="tests/test_skills_put_down_single.py tests/test_place_phase.py tests/test_plan_episode_spawn.py"; make check
```
起栈前查：`nvidia-smi`、`ls /workspace/tmp/p03*/stack.json`、`ss -ltn | grep -E ':55(5|6)5 '`、5 秒 CPU 采样（配额 15.36 核，> 11 核别起）。
