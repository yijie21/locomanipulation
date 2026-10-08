# STATE — 2026-10-08（会话：state 改回官方 46 维、官方教程对照文档、warehouse_mars_50 采集、旧数据迁移文档）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
进行中（10-08）：`scene_discovery`（loco_agent 会话的 skills / Q11 / scene_discovery 工作）合入 main，做在分支 `scene_discovery-sync`（worktree `/workspace/code/locomanipulation-scene_discovery-sync`；合入前的分支打了 tag `scene_discovery-pre-sync`）：`10b3895` 合并、`0fcceaf` 改用 mars + Unreal、之后一个文档提交。**还没快进到 main、没推送，等用户定**。分支的 TODO 7/8/9 改编号为 17/18/19。
已确认（10-08）：到 `b8f5e9c` 为止都已推到 origin/main；之后只有本次 handoff 提交，未推送。
已确认（10-08 收尾时）：本机没有留下仿真栈、策略服务、镜像或 Unreal 编辑器（`stop_unreal.sh` 已跑）。GPU1 上常驻别的项目的 depth_server 约 3.7 GB。
别的会话：henry-zhao 直接推 origin/main 或 `henry-dev`，推送不跑钩子，可能自带检查失败（记忆 origin-main-can-carry-failing-checks）。

## 磁盘

- 已确认（10-08）：`/workspace` 剩 93 GB（91%），根分区 `/` 剩 1.1 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 本次新增：`outputs/runs/warehouse_mars_50` 3.1 GB，各步日志 `outputs/collect_logs_warehouse_mars_50/`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认（DECISIONS 2026-10-07 三节，`804f962`、`f13199f`）：warehouse mars 线时序对齐官方（50 fps、执行 20 拍、相机延迟、实测推理延迟分布）；SONIC 两条线 state 改回官方 46 维，数据集不用重转，旧 53/60 维 checkpoint 不能再服务或评测。
- 已确认（DECISIONS 2026-10-08）：Dex1 方向维持 0 闭、5.5 开（真机未验，16.6）。
- 已确认（`84f0fe7`）：对照文档 `docs/official_comparison.md`，README"通用"给入口；与教程不同的项和理由都在里面。颜色增强与教程相同。
- 已确认（`6a6bd10`，台账 crate.md 2026-10-07）：`warehouse_mars_50` 照 runbook 第 1–4 步做完：120 局保留 50 条（75 分钟）、Unreal 补渲 80 分钟、转换 9 分钟；训练集 45 条 53,800 行、验证集 5 条，50 fps，审计两边各 32 项全过。没训练。
- 已确认（`6a6bd10`）：修了 884a181 起 Unreal/Isaac 补渲对新 session 必崩的 `KeyError: 'dataset_fps'`。
- 已确认（`b8f5e9c`）：`docs/migrate_old_data.md` + README 顶部提示：旧数据不重新采集，派生 50 fps 新运行后实时录制的只重新转换、离线出图的重渲再转换；文档里每段命令都跑过。
- 已确认：试验时误改写过 `warehouse_edges_45x20x18_20260928` 的原始 session，已用副本逐字节恢复（DECISIONS 2026-10-08）。

## 待验证 / 卡点

- 已确认（pick_place 台账 §18）：合并前后同机同 seed 回归看不出差别（bottle 7 vs 6、pick_place 2 vs 4、p4 3 vs 3，各 10 局）；mars 上 p4 1/10，其中 5 局右腕 D405 支架碰罐子（19.18.8），`capability.yaml` 是 stock 的实测（19.18.5）。
- 卡点（19.18.6）：`plan_episode --record` 录的运行补渲不了（不设状态轨迹目录、不写 `collect/inputs.json`、不复制场景），所以 mars + Unreal 的录制 → 补渲 → 转换还没走通。
- 已确认（10-08 04:06–07:02）：容器内存打满（91 GB 上限）三小时，这段时间的局作废；跑 ADT 栈前先看 `memory.current`（记忆 container-memory-91gb-adt-stacks）。

- 待验证（16.16.1）：实时录制按 50 fps 转换时审计 "action vs state" +0 帧不过，只在 1 条的 `crate_smoke` 上见过。
- 待验证（16.16.4）：实时录制的图实测约 30.3 Hz，代码推算约 33 Hz，迁移文档写的是 33，要改。
- 待用户：16.14.1（批 32、20000 步）、16.15.4（`check_dataset` 漏审 warehouse 的双臂档）。
- 待用户（13.6.18）：warehouse mars 实验文件顶层 `backend.dataset_fps: 30.0` 与 `dataset.fps` 不一致（采集不读它）。
- 待协作者：真机三台相机帧率与延迟（13.6.16.1）；相机内参（13.3、13.6.15.2）；腰固件力矩上限（13.6.11.5）；真机 C++ 踝关节增益（16.13）；告诉 henry 16.10、16.12.1、16.16.3。
- 待验证（同前未结）：16.11 钩子偶发失败；13.6.15.1、14.4、14.5、9.11.1、12.9.13、12.20.4、12.9.6。

## 下一步（编号见 `TODO.md`）

- **16.15.3**：训练 `warehouse_mars_50`（单卡 1 万步）再闭环评测 100 局；这是第一个 46 维、50 fps、带相机延迟的 checkpoint，也用于 16.7.2、16.2.1。
- **16.2.1**：用这个 checkpoint 起 `run_gr00t_server.sh`，客户端在本机连上核对（不上真机）。
- **16.4**：`warehouse_mars_200` 的 50 fps 版本要在采集它的机器上做（本机没有这个运行，16.4.1）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; ss -ltn | grep -E ':(55[5-9][0-9]|5700) '
make check                                                       # 全量检查；提交钩子也跑它
bash scripts/train.sh outputs/runs/warehouse_mars_50 --gpu 0     # 16.15.3；长任务用 nohup setsid 脱离，Monitor 看日志
# 闭环评测（docs/warehouse.md 第 6 步）：start_stack.sh eval / run_vla_server.sh / start_unreal.sh --gpu 1 + run_unreal_mirror.sh / bimanual/run_vla_inference.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_server.sh --model-path outputs/runs/<run>/model/checkpoint-<step>
bash scripts/run_real_robot_inference.sh --host localhost --init-token-file configs/deployment/warehouse_initial_token.json [--hand-wire sim]
# 后台任务的监控管道：grep --line-buffered；mawk 要 -W interactive，cut 会缓冲
# 提交：按文件名 add（不要 -A，别带上 assets/hero.png、.serena、node_modules）→ commit → 用户说了才 push
```
