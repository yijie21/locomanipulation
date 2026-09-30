# STATE — 2026-09-30（会话：第三视角录像按渲染器统一并做成一条独立命令；5 个 Unreal 关卡重建；warehouse 冒烟按新规模实跑；已推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认：与 origin/main 同步于 `db71fc2`，之后只有这次 handoff 提交。
已确认（05:00 UTC 前后）：本会话起的仿真栈、Isaac、Unreal 编辑器都已关掉，GPU0 空闲。
GPU1 约 23 GB 是别的项目（`SimpleRoboJev-wt-batch2` 视觉进程三个、`depth_server`）。在 GPU1 上跑 Isaac 或 Unreal 前先看 `nvidia-smi`：Isaac 补渲实测约 7 GB，放得下。
别的会话：henry-zhao 直接推 origin/main，且推送不跑钩子。已确认：09-30 拉下来的提交自带两处检查失败，已在合并提交里修掉。

## 磁盘

- 已确认：`/workspace` 空闲 158 GB（09-30 05:00）。本会话新增约 100 GB，其中三份 10 步冒烟模型各 24 GB（TODO 12.20.3 待用户定）。
- 已确认：根分区 `/` 只有 32 GB，空闲约 5 GB。跑 Isaac、UE、pytest 前把 `TMPDIR` 指到 `/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3 的删除都还没定，删前要在这里写明 24 小时。

## 当前进度（TODO 12.9.14、12.20；DECISIONS 2026-09-29/30 第三视角一节）

- 已确认，已推送：第三视角录像由 `images.source` 的渲染器画，只在跑下面这条命令时画，别的步骤都不依赖它：
  `bash scripts/render_third_person.sh <run> [--gpu N] [--eval] [--all] [--dry-run] [局名]`
  - 不加参数：采集的成功局，写到 `videos_<渲染器>/`（MuJoCo 是 `videos/`），预览写到 `previews_<渲染器>/`。
  - `--eval`：闭环评测的局，写到 `eval/` 下。
  - 所有局用同一个机位：四边线场景是每局的边机位，带 ghost 夹爪，第一视角和双腕做小窗。
  - convert 有录像就链接进数据集，没有也照常转换。
  - 三种渲染器都实跑核对过：画面与 MuJoCo 逐帧对齐；Isaac 机器人完整；Unreal 与前一天渲的差 1.3–1.8/255。
- 已撤回：实验文件开关 `images.third_person`（`4144452`），已被上面这条命令取代；读到旧运行副本里的这个键时直接丢掉。
- 已确认：5 个 Unreal 关卡都按新导入重建（导入里带第三视角相机的自由刚体和两只 ghost）。代码早于 `7f8f4b5` 的 checkout（例如 `loco_unreal` worktree）读这些关卡会报不认识的关节。
- 已确认：warehouse 冒烟按新规模整条实跑，Isaac 和 Unreal 各一次。runbook 里的规模和耗时都已改成实测值；逐步计时在 `outputs/warehouse_smoke_timing_20260930/timing.tsv`。
  - 采集到评测：Isaac 约 50 分钟，Unreal 约 31 分钟。第三视角录像另外约 20 和 18 分钟。
  - 两次都是 2 批 20 局，保留 11 条和 8 条，因此用户把 `max_batches` 从 2 改为 3（`db71fc2`）。
- 已确认：本机每批 10 局的采集通道要约 1600 个进程，只开得出 1 条。

## 待验证 / 卡点

- 待验证（12.20.4）：还没按 `max_batches: 3` 跑过冒烟，第三批能不能补齐没实测。
- 待验证（12.9.13）：没人照文档从零装过 Unreal。
- 未做（12.9.14.3 剩下的部分）：`mujoco` 运行的仿真实时录像，四边线仍用场景的 `video_camera`，不是每局的边机位。
- 未做（12.9.6，待用户安排）：用 Unreal 补渲的数据重训。在那之前，Unreal 的评测数字不能和 Isaac 比。
- 上一会话留下：10.4（旧 `outputs/runs/warehouse_smoke` 怎么处理）、11.9、12.19.4，以及 10.7、11.7、11.6、11.10。

## 下一步（编号见 `TODO.md`）

- **12.20.3**：问用户这次的三份 10 步模型（各 24 GB）和测试副本删不删；要删的话先在这里写明 24 小时。
- **12.20.4**：可选，按 `max_batches: 3` 再跑一次冒烟采集，看第三批能否补齐。
- **12.9.6**：用户安排后，用 `warehouse_200_unreal.yaml` 做 Unreal 的整条训练和评测。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
df -h / /workspace; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; export TMPDIR=/workspace/tmp
# 第三视角录像（按 images.source 选渲染器；Unreal 要先 start_unreal.sh）
bash scripts/render_third_person.sh outputs/runs/warehouse_smoke_20260930 --dry-run
bash scripts/render_third_person.sh outputs/runs/warehouse_smoke_20260930 --gpu 1 --eval --all
# Unreal：起编辑器，场景改过就重建关卡，关编辑器
bash scripts/start_unreal.sh --gpu 1; bash scripts/build_scene.sh warehouse_box_45x32x33 --unreal; bash scripts/stop_unreal.sh
# 长时间的计时或队列：写成脚本文件，脱离终端跑，再用 Monitor 盯日志（后台 Bash 有 600 s 上限）
nohup setsid bash outputs/warehouse_smoke_timing_20260930/queue2.sh > outputs/warehouse_smoke_timing_20260930/queue2.log 2>&1 &
# 提交：make check → 按文件名 add（不要 -A）→ commit（钩子跑全量检查）→ fetch，远端有新提交就 merge（不 rebase），
#       merge 后先 make check（henry 的提交可能自带失败）→ 用户说了才 push
```
