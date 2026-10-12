# STATE — 2026-10-12（会话：任意物体抓取的抓取来源——默认离线缓存、graspnet 可选，接口 + 文档交付并推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-12）：origin/main = `a2cfb23`（本会话两组提交 + 合并 origin 4 个提交，已推送）；本 handoff 提交之后 main 比 origin 多 1 个，未推送（用户说了才推）。
已确认（10-12）：实例 1–4 空，graspnet 和 CuRobo 服务都已停；GPU 0 / 1 空。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败（10-12 合并时就修了一个：`warehouse_mars_200_isaac.yaml`）。

## 提交

- 已确认：`0035add` 主线 CuRobo 出路线开关（22.6）；`68ce01c` GraspNet 线 + 抓取缓存线 + 执行层改动 + 抓取来源接口与文档（21.3.4.5、23.2.8、23.7）；`a2cfb23` 合并 origin（合并里补了 origin 漏改的 isaac 孪生配置）。`git log --oneline -5` 为准。
- 没提交（不是这条线的、来路不明）：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1.md`–`step7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`。
- 提交钩子会跑整套 CPU 检查，并先把未暂存的改动藏起来；未跟踪的测试依赖未暂存的 src 改动时钩子会失败。STATE.md 上限 60 行（`lint_repository.py`）。

## 磁盘

- 已确认（10-12）：`/workspace` 剩 52 GB（95%）；跑仿真 / pytest / 安装前 `export TMPDIR=/workspace/tmp`。容器内存 91 GB。
- 已确认：本线还在的产物见 TODO 23.2.7 下 10-12 的"复核磁盘"一条（`_gate*`/`_high*` 260 个 2.2 GB、`presort_b/ghost_rounds/` 392 MB、`cache/videos_best_full/` 202 MB、`_demo*` 约 30 MB）；缓存 `scenes/obj_*/grasp_cache.npz` 留。删不删待用户。
- 待用户（更早）：21.3.4.6、22.7、11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7；要删先在这里写明 24 小时。

## 当前进度

- 已确认（23.7）：抓取来源 `cache`（默认）/ `graspnet`，`run_grasp.py --grasp-source`、`GraspPlanned(source=)`、计划步 `source`；动手臂前就绪检查（`grasp_cache` / `graspnet_server` / `grasp_source`）。`run_grasp.py` 默认 `--planners ik --rounds 1`。
- 已确认：`install_graspnet.sh` 在空目录从零装通（104 s，安装包已在 uv 缓存里；自检 `110 grasps`）；graspnet 服务 1.6 GB 显存、几秒起好。
- 已确认（冒烟，实例 1，物体 040，10-12）：缓存来源抬起 74 s；服务没开时动手臂前被拒；graspnet 来源在现在的默认配置下抬起 80 s。第 5 步出片命令跑通（ghost 37 s / 局，预览 472 KB）。
- 已确认（同执行逻辑，10-08 / 10-10）：缓存 25/65（同配置重跑 18）对 graspnet 16/65，配对 18 对 9（p 0.061）；没双指接触 9 对 15、抬起脱手 7 对 13。现在默认配置（预检 + arm pin + 闸门）缓存两批 32、33/65。
- 待验证：graspnet 在现在默认配置下的成功率（只有 1 局冒烟），对比表停在 10-10 之前的执行逻辑（23.7.4）。
- 文档：读者页 `docs/object_grasp.md`、详解 `docs/object_grasp_design.md`；过程在 `docs/handoff/grasp-cache-20261010.md`（第 1–13 节 + 10-12 附注）和 `graspnet-bottle-planner-comparison-20261008.md`。

## 下一步（编号见 `TODO.md`，都待用户定）

- **23.7.4**：graspnet 来源在现在的默认配置下两批 65 局（约 40 分钟），补一组同执行逻辑的当前对比。
- **21.3.5.1 / 23.7.6**：缓存按物体稳定姿态分键（物体被碰歪时现在无候选）。
- **23.2.7**：本线剩余产物删不删。

## 未解决 / 卡点

- 待用户：23.2.7、23.3–23.5、23.7.4–23.7.6、21.3.5、22.3.6–22.3.8、22.5、22.7、21.3.4.2–21.3.4.4、21.3.4.6、21.3.4.8.1、21.3.1、19.8.1、19.8.3、19.18.3.1、16.14.1、16.15.4、13.6.18。
- 已确认（环境）：镜像的 `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` 指向不可写的 `/.uv`，`CUDA_HOME` 是 13.0（torch cu128 扩展要 `/usr/local/cuda-12.8`）；`install_graspnet.sh` 已处理。
- 已确认（环境）：numpy 大页缺页会卡在内核整理——离线 numpy 重活设 `NUMPY_MADVISE_HUGEPAGE=0`。CuRobo 0.7.4 批量为 1 有两个崩溃 bug（服务已自愈）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp
make check
D=scripts/experiments/graspnet_bottle_20261008
bash $D/cache_all.sh 12                                   # 建全部缓存，约 5 分钟（场景重建会删缓存）
bash $D/run_one.sh 40 1 0 _demo                           # 一个物体一局，缓存来源，约 75 s
bash $D/install_graspnet.sh                               # graspnet 一次性安装
GPU=1 bash $D/servers.sh start graspnet                   # graspnet 服务 :5790；用完 servers.sh stop graspnet
bash $D/run_one.sh 40 1 0 _demo_gn --grasp-source graspnet
bash $D/run_batch.sh 4 1 0 _mine [--grasp-source graspnet]   # 65 个物体，实例 1–4，约 20 分钟；STOP 文件停
python3 $D/cache_summary.py _mine                          # → outputs/graspnet_bottle_20261008/cache/summary_mine.md
# 停服务按 pid 文件（servers.sh stop），不要 pkill -f；停栈：bash scripts/stop_stack.sh --instance N
```
