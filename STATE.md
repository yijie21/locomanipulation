# STATE — 2026-09-25（会话：从零按文档复现三场景；outputs 清盘；P7/P8/P19 与文档 1–4 修复）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已推送到 `b3b645f`；`b8f6c82`、`9a08847` 与本次 handoff 提交**未推送**（TODO 4.8.8）。
没有运行中的栈、镜像或复现 agent。理由与被否决方案在 DECISIONS 最后一节（2026-09-25 从零复现）。

## 工作树

- **未提交（已完成，`make check` 全量通过，等用户说提交）**：P7/P8/P19 与文档 1–4 的改动——`src/locomani/pipeline/{run_context,run_convert,run_stack}.py`、
  `src/locomani/tasks/crate/collect.py`、`tests/{test_run_context,test_dry_run_delivery,test_interpreter_spawn_rule}.py`、`tests/fixtures/interpreter_spawns.json`、
  `README.md`、`docs/{setup,single_arm,bimanual,warehouse}.md`、`patches/sonic_{pose,hand}_ownership.md`。提交时按文件名 add。
- 未跟踪（与本会话无关，不要提交）：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`。
- `third_party/GR00T-WholeBodyControl` 挂着两个 worktree（4.8.5 用，勿删；删时用 `git worktree remove`）：`outputs/deploy_docbuild_20260924/`、`outputs/deploy_pinned_20260925/`。

## 当前进度

- **从零复现（已确认，4.6.1）**：无上下文 agent 从 clone 只按文档装 0–4 步（指纹 `c1360628…`、mujoco 3.11.0），bottle 7/10、crate 8/10、warehouse 3/10，
  **30 局零开局失败、零摔倒**。报告 `outputs/fresh_repro_20260925/report.md`（P1–P20），启动器 `launch.sh` 同目录；沙盒已删。
- **已修（已确认，未提交）**：P7 holdout 准入按 `batch_rounds × max_batches`（起栈与两条 collect 入口）；P8 convert 的 stats 在 `.venv_data_collection` 跑
  （三数据集与训练 venv 逐字节相同，沙盒 0–4 环境 convert 全通）；P19 文档写明 CUDA 13 运行时；文档 1–4（TODO 4.6.2.4–4.6.2.7，沙盒核对过 3、4）。
- **outputs 清盘（已确认）**：旧 expert 数据与**全部 checkpoint** 作废，删 163 G，`/workspace` 余 ~208 G；旧数据集不可恢复。清单 `outputs/cleanup_20260925/`。
- 之前的成果不变：Isaac 按需镜像（`docs/handoff/isaac-vla-mirror-20260925.md`，真实策略闭环未跑）；setup 钉 1983e888 + 三补丁；gate8 bank 入库。

## 卡点 / 待验证

- **协作者仍 `initial_stance_unstable`**：文档化的安装本身已排除（已确认，4.6.1）。原因**待验证**；候选（假设）：他机器的系统层（CUDA 运行时版本、
  mujoco 未钉 3.11.0、CPU 配额 / 同时多实例）或停栈后不足 20 s 重起。开局判定失败只记一个 `false`，看不出哪种退出（4.8.2.1 待用户定）。
- 待验证：P19 的 `cuda-cudart-dev-13-0` 安装行没在干净机器上跑过。
- 待验证：crate 换成带夹爪补丁的二进制是否等价（本机 crate 仍是 4add301，4.8.4.3）。
- 用户暂缓：TODO 4.6.2.8–4.6.2.10（crate 开局距离区间、warehouse 基线 holdout、场景包/Blender/`python` 等小项）。

## 下一步（编号见 `TODO.md`，等用户定顺序）

- 用户确认后提交上面"未提交"的改动，再做 **4.8.8** 推送。
- **4.8.2** 协作者开局失败：等用户定怎么拿到他那边的证据（**4.8.2.1** 开局诊断是否先加）。
- 旧 expert 已作废：新 expert 的采集/训练由用户安排（旧 checkpoint 已全部删除）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
bash scripts/cpu_headroom.sh; nvidia-smi; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑
# 本机 bottle/warehouse 用 5cab2fb：export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"；crate 不设
# 停栈后等 ≥20 s 再起；pkill -f 会匹配自己的命令行，杀进程写脚本文件

# 从零复现（隔离 agent）：见 outputs/fresh_repro_20260925/launch.sh（env -i + 沙盒 HOME，claude -p，hooks/记忆/MCP 关）
# P8 证据：outputs/fresh_repro_20260925/p8_stats_ab/compare.txt；文档 3/4 核对：outputs/fresh_repro_20260925/doc_fixes_1_4/

# deploy 源码指纹与能力标记
(cd <deploy_root>/src && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum)   # 期望 c1360628…
for m in POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1 SONIC_AUTO_START SONIC_DDS_DOMAIN; do grep -aq $m <binary> && echo "有 $m" || echo "缺 $m"; done

# 提交：make check → 按文件名 add → git commit --no-verify（钩子会被别人的 WIP 干扰）
make check
```
