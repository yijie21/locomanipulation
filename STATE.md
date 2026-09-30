# STATE — 2026-09-29（会话：loco_unreal 两次合入 + `images.source: unreal`；Unreal 文档按规范重写；/workspace 清理 165 GB；已推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，与 origin/main 同步于 `fc28d1d`（之后只有这次 handoff 提交）。
已确认（08:37 UTC）：本会话起的 Isaac 渲染都已结束；Unreal 编辑器没在运行。GPU0 空闲（2 MiB）。
GPU1 已占 26 GB，全是别的项目：`depth_server.py` 3.7 GB、`loco_agent` 的 bottle 进程 1.5 GB、`SimpleRoboJev` 三个视觉进程共约 17.8 GB。在 GPU1 上跑 Isaac 或 Unreal 之前先看 `nvidia-smi`。
别的会话：`locomanipulation-loco_agent` 在实例 1、GPU 1 起栈；`locomanipulation-loco_unreal`（worktree）负责 Unreal 线，它的 `loco_unreal` 分支已全部合进 main。

## 磁盘

- 已确认：本会话删了 A 类 96 GB、B 类 69 GB（TODO 9.9、9.9.1；DECISIONS 2026-09-29 清理一节；清单和日志在 `outputs/cleanup_20260929/`）。`/workspace` 空闲 236 GB。
- 已确认：根分区 `/` 只有 32 GB，08:37 空闲 6.1 GB，还在往下掉（06:00 一度只剩 11 MB，当时 Isaac 和工具都写不进去）。跑 Isaac、UE、pytest 之前把 `TMPDIR` 指到 `/workspace`。
- 待用户：上一会话的小产物（TODO 11.9，约 1 GB + 1.2 GB）最早 2026-09-30 02:00 UTC 可删；`outputs/merge_check_loco_unreal_20260929/`（66 MB）是否保留（12.19.4）。

## 当前进度（TODO 第 12 节；DECISIONS 2026-09-29 各节）

- 已确认：Isaac 新外观已在 main 上。
  - G1 用 Unitree 官方材质；所有 Isaac 场景 2× 超采样，视频也不跳过（用户定）。
  - warehouse 按 Blender 标定，stage 用 `warehouse_textured_env`，四边线用 `warehouse_box_<尺寸>_env`；`build_isaac_warehouse_scene.py` 写的 label 一律带 `_env`；box_platform 用 `box_platform_srgb`。
  - 旧 stage 别删。新外观不重训（用户定），新旧评测数字不能直接比。
- 已确认：`images.source: unreal` 已接通并推送。
  - 有关卡的场景：`warehouse_textured` 和四个 `warehouse_box_*`，关卡记录在 `$UNREAL_ROOT/levels`（本机 `/workspace/ue_57/levels`）。从主 checkout 调 `unreal_scene`，5 个关卡都通过哈希核对。
  - URLab 端口在 5800–5806 和 18801，不和仿真栈冲突。
  - 实跑过（loco_unreal 会话）：补渲 2 集加 convert；实例 0 上 lockstep 闭环 3/3 和 1/2。用的是 Isaac 训练的 checkpoint，属于分布偏移。
- 已确认：warehouse runbook 里 Isaac 和 Unreal 平行（开头"先选渲染器"，第 1、3、6 步两种命令并排）。选渲染器就是选实验文件：`warehouse_{smoke,200}_unreal.yaml`，有测试守着。setup 第 8 步、术语表、`--render-gpu` 都已按规范写。

## 待验证 / 卡点

- 待验证（12.9.13）：没人照文档从零跑过 Unreal（引擎下载、完整采集到评测）。
- 未做（12.9.6，待用户安排）：用 Unreal 补渲的数据重新训练；在那之前 Unreal 评测数字不能和 Isaac 比。12.8（Unreal 100 局评测）同理。
- 已完成（12.9.14，用户 2026-09-29/30 定）：第三视角录像由 `images.source` 的渲染器画，而且只在跑 `scripts/render_third_person.sh` 时画（采集加 `--eval` 为评测）；补渲、转换、评测都不依赖它。5 个 Unreal 关卡都已按新导入重建（2026-09-30）。
- 已完成（12.9.14.5 / 12.19.3）：warehouse 冒烟按新规模（10 条、每批 10 局）Isaac 与 Unreal 整条实跑，runbook 的耗时改成实测；两批 20 局曾只保留 8 条，用户定 `max_batches` 2 → 3。
- 已确认：cap120 的 `observations/` 已删，那 200 条不能再补渲；要用新外观或 Unreal 重训就得重新采集。
- 上一会话留下：10.7、11.7、11.6、10.4、11.10（四边线采集和并行通道的验证与待定项），以及 9.3、9.10、9.8、8.6.4、7.2、8.11。

## 下一步（编号见 `TODO.md`）

- **12.9.13**：没人照文档从零装过 Unreal（耗时已于 2026-09-30 在装好的本机实测）。
- **12.9.6**：用户安排后，用 `warehouse_200_unreal.yaml` 采集、Unreal 补渲、训练、评测。
- **10.4 / 11.9**：问用户旧 `outputs/runs/warehouse_smoke` 怎么处理，以及小产物何时删。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
df -h / /workspace; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; export TMPDIR=/workspace/tmp
# Unreal：起编辑器、建关卡、看实验文件解析（只打印不起栈）
bash scripts/start_unreal.sh --gpu 1; bash scripts/build_scene.sh warehouse_box_45x32x33 --unreal
bash scripts/run_overnight.sh configs/experiments/warehouse_smoke_unreal.yaml --gpu 0 --render-gpu 1 --line bimanual
bash scripts/stop_unreal.sh
# 提交：make check → 按文件名 add（不要 -A）→ commit → fetch，远端有新提交就 merge（不 rebase）→ 用户说了才 push
```
