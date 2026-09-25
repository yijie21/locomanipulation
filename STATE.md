# STATE — 2026-09-25（会话：严格按 README 跑 bottle 采集；strace 逐文件核实；拖慢实验；修复推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，已推送到 `2063a17`，本 handoff 提交在它之上。
没有运行中的栈；两个沙盒 `/workspace/readme_bottle{,_strace}_20260925/` 已删（用户定）。理由与被否决的方案见 DECISIONS 最后一节。

## 工作树

- 未跟踪（与本项目无关，不要提交）：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`。
- `third_party/GR00T-WholeBodyControl` 挂着两个 worktree（给 4.8.5 用，不要删；删时用 `git worktree remove`）：`outputs/deploy_docbuild_20260924/`、`outputs/deploy_pinned_20260925/`。

## 当前进度

- **严格按 README 的 bottle 采集不摔（已确认，4.6.3）**：origin d505880，`env -i` 沙盒（不给 CUDA_HOME、LD_LIBRARY_PATH、`.env`；sudo 换成拒绝桩），
  开局 20/20 站稳，保留 10/20。拖慢实验的基线组又是 10/10 站稳，合计 30 局零开局失败。报告 `outputs/readme_bottle_20260925/report.md`。
  这个结论只覆盖本机的系统层：CUDA 13、NVIDIA EGL、apt 源、CPU 配额都是本机提供的，测不到。
- **strace 逐文件核实（已确认，4.6.4）**：系统目录零写入；沙盒外只写了 `/tmp/cdds.LOG`（上游写死）和 git-lfs 的 `/tmp/sock-*`；读取的非系统文件只有 `~/.ssh`。
  逐条清单 `outputs/readme_bottle_strace_20260925/report_paths.md`、`paths.tsv`，原始 trace 是 `traces.tar.gz`。strace 下 19/20 开局失败，那是 strace 自身造成的，不算。
- **拖慢实验（已确认，4.8.2.4）**：每 100 ms 停 16 ms，三组同一批 10 个 case。基线 7/10；只拖 sim（RTF 0.69）1/10，失败是髋部碰桌 8 次；只拖 deploy 8/10。
  **三组开局失败都是 0**。"栈慢就开局摔"这个说法已被推翻。报告 `outputs/slowdown_20260925/report.md`。
- **已推送**：P7、P8、P19（nvcc 那句已按实测改写）、文档修复 1–4、单臂排错表的起栈摔倒行，以及两个让 make check 通过的小修（henry 的 prepare.py 格式、README 行数）。

## 卡点 / 待验证

- 协作者"一开始摔倒"的原因仍未知（待验证）。用户定：这条线到此为止，其余另行单独解决（4.8.2.2、4.8.2.3、4.8.2.5、4.8.2.6）。
- 待验证：没有 nvcc 的机器上 cmake 能否配置（4.6.2.3.1）；`cuda-cudart-dev-13-0` 那一行没在干净机器上跑过。
- 假设（未测）：sim 慢时的髋部碰桌，是因为抓取阶段按墙钟计时（4.8.2.6）。
- 待验证：crate 换成带夹爪补丁的二进制是否等价（本机 crate 仍用 4add301，4.8.4.3）。

## 下一步（编号见 `TODO.md`，顺序等用户定）

- **4.8.2.2**：问协作者说的摔倒，是 sim.log 起栈时那行，还是 `collect_NN.log` 里的逐局失败；同时要他们的 RTF（pose_sidecar）和 deploy 的 `Loop timing`。
- **4.8.2.1**：开局判定失败时，是否在 manifest 里记下卡在哪个条件（待用户定）。
- 旧 expert 已作废：新 expert 的采集和训练由用户安排。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
bash scripts/cpu_headroom.sh; nvidia-smi; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑
# 本机 bottle/warehouse 用 5cab2fb：export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"；crate 不设
# 停栈后等 ≥20 s 再起；pkill -f 会匹配自己的命令行（本会话又踩一次），杀进程先 ps 找 pid 再 kill

# 严格 README 沙盒的做法（沙盒已删，脚本还在）：
#   outputs/readme_bottle_20260925/sb.sh        env -i 跑一条命令，跨调用保留 export 和 cwd
#   outputs/readme_bottle_strace_20260925/sbt.sh 同上，外面套 strace；analyze_traces.py → paths.tsv → summarize_paths.py
#   outputs/slowdown_20260925/{duty_cycle.py,run_arm.sh,analyze.py}   单进程 SIGSTOP 占空比、每组驱动、汇总
# 从 sidecar 算 sim RTF：只累加相邻行 t 的正增量（每局重置时仿真时钟归零）

# deploy 源码指纹与能力标记
(cd <deploy_root>/src && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum)   # 期望 c1360628…
for m in POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1 SONIC_AUTO_START SONIC_DDS_DOMAIN; do grep -aq $m <binary> && echo "有 $m" || echo "缺 $m"; done

# 提交：make check（文档有行数预算：README 100、single_arm/bimanual 210）→ 按文件名 add → git commit --no-verify
make check
```
