# STATE — 2026-09-25（会话：bottle 走近改成 warehouse 方式并设为默认）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，已推送到 `4655254`；其上的 `e887520`（handoff）和 Isaac 数据集渲染的提交都未推送。
**运行中（2026-09-25 起，TODO 4.7.6.1）**：`box_room_grasp_200_cap120_20260925` 采集栈占实例 0 / GPU0，Isaac 补渲轮次占 GPU1，之后自动 convert → train → 100 局闭环（编排在会话 scratchpad，进度看 `<run>/pipeline.log`、`agent.out`）；`box_room_grasp_200_20260925` 是上限不够时停掉的半截（7/30），不用。GPU1 上 `depth_server.py` 与 SimpleRoboJev vision worker 是别的项目的。理由与被否决的方案见 DECISIONS 最后一节。

## 工作树

- 未跟踪、与本项目无关、不要提交：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`；`third_party/GR00T-WholeBodyControl` 挂着 worktree `outputs/deploy_{docbuild_20260924,pinned_20260925}/`（给 4.8.5 用，删时用 `git worktree remove`）。

## 当前进度

- **bottle 默认专家已换成 guard 走近（已确认，TODO 5.8，`e3d49d9`）**：`reach.simple.approach_stop: guard` 是代码默认。
  流程：手臂交给步态自然下垂 → `SLOW_WALK 0.25`，方向在第一帧定死、直指规划站位点 → 按桌边守线刹车（`tasks/bottle/edge_guard.py`，挡的是桌面那一层）
  → 接管 → `_lift_ready`（warehouse 的 `lifted_ready`，关节空间插值）→ 左手留在原位，右手滑到预抓取点、进入、闭合、抬起。
  主线 5 份实验文件写了 guard 全套参数；记录旧走法的 9 份钉 `visible`。
- **实测（已确认，台账 §39）**：同一组 seed 40000。`bottle_simple_guard_lifted_20260925` 10/10，0 局越线；骨盆离桌边 0.25–0.40 m（旧走法 0.12–0.31）；
  手在桌边后 0.08–0.22 m，不再后退；肩外展 18–23°；每局 39.5–44.3 s。旧走法基线 `bottle_simple_10ep_20260925` 10/11。
- **文档已跟上（已确认，`4655254`）**：single_arm 首段、single_arm_design（新节、图 ②、开头两张表）、change-map-bottle、台账 §39、配置字段表。README 不用改。
- 用户定：**bottle 专家不再微调。**

## 卡点 / 待验证

- 已确认：`lifted_ready` 插值中途手指先往前扫 6–8 cm（参考轨迹约 3 cm，跟踪滞后 4–5 cm），停得最近的两局抬手时离桌边只剩 6 mm / 1.2 cm；守线余量没算这一段（5.4.1.1，待用户定）。
- gate8 闸门及其基线 `tests/gate/baseline_simple.json` 仍是旧走法，拿 guard 去比必然 FAIL（5.8.1，待用户定）。
- 待验证：新默认采到的数据能否走通转换 → 训练 → 闭环评测。假设 `eval.initial_token` 不用改（开局仍是规划器的下垂姿势）（5.10）。
- 假设（数据与之一致，未做单变量验证）：逐帧修正方向导致步态停走，原因是 deploy 在方向变化时重新规划（记忆 sonic-planner-replans-on-any-direction-change）。
- 上个会话留下、仍未解决：协作者"一开始摔倒"的原因（4.8.2.2 等，用户已搁置）；crate 仍用 4add301（4.8.4.3）。

## 下一步（编号见 `TODO.md`，顺序等用户定）

- **5.10**：用新默认跑一遍整条线（`bottle_simple_200` 或缩小版），核对 `initial_token`。
- **5.8.1**：用 guard 在 8 局 bank 上重录 gate8 基线；或者 **5.8.2**：删掉旧走法的代码（两件都待用户定）。
- **4.8.2.2**：问协作者说的"摔倒"是 sim.log 起栈时那行，还是逐局失败。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export PYTHONPATH="$LOCOMANI_ROOT/src"
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 本机 bottle/warehouse 用 5cab2fb；crate 不设
bash scripts/cpu_headroom.sh; nvidia-smi; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑
# 停栈后等 ≥20 s 再起；pkill -f 会匹配自己的命令行，杀进程先 ps 找 pid 再 kill

# 采集（默认 guard）：起栈（起完即退）→ agent server（跑完自己收栈）；--name 另起 run，--dry-run 只打印
bash scripts/start_stack.sh collect configs/experiments/bottle_simple_guard_20260925.yaml --gpu 0 --name <run>
bash scripts/single_arm/run_agent_server.sh configs/experiments/bottle_simple_guard_20260925.yaml --gpu 0 --name <run>
#   这份试验文件是 successes 10 / batch_rounds 10 / max_batches 1：不足 10 集时 agent 退出码 1（设计如此）

# 本会话的分析脚本（原在 scratchpad，已复制到 outputs/bottle_guard_20260925/）
python3 outputs/bottle_guard_20260925/export_previews.py outputs/runs/<run> outputs/videos/<run>   # 按 manifest 对视频，压到 ≤500 KB 并逐段核对
python3 outputs/bottle_guard_20260925/compare_guard.py <基线 run 名> <试验 run 名>                  # 逐局对比停位 / 守线记录（在仓库根目录跑）
PYTHONPATH=src "$VENV_SIM_PY" outputs/bottle_guard_20260925/ready_compare.py <run 名>             # 抬手后肩外展、肘外、抬手过程离桌边最小距离
# 逐局记录：manifest 的 execution.metrics.simple.{guard,ready,plan,phases}

# deploy 源码指纹与能力标记
(cd <deploy_root>/src && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum)   # 期望 c1360628…
for m in POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1 SONIC_AUTO_START SONIC_DDS_DOMAIN; do grep -aq $m <binary> && echo "有 $m" || echo "缺 $m"; done

# 提交：make check（文档有行数预算：README 100、single_arm/bimanual 210）→ 按文件名 add → git commit --no-verify
make check
```
