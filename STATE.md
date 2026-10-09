# STATE — 2026-10-09（会话：SIMPLE 借鉴项 21.3、GraspNet 物体上 IK 规划 vs SIMPLE 的 CuRobo 规划）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认：origin/main = `8c79be6`；本次 handoff 提交后 main 比 origin 多 1 个提交，未推送（用户说了才推）。
已确认（10-09 收尾）：本会话没有留下任何进程——graspnet / CuRobo 服务已停，实例 1–4 的锁都空，Unreal 编辑器已关；GPU1 上只有别的项目的 depth_server（3.7 GB）。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 未提交的改动（不在 handoff 提交里）

- `scripts/experiments/graspnet_bottle_20261008/`（21.3.4 全部实验代码，ruff 过）、`docs/handoff/graspnet-bottle-planner-comparison-20261008.md`、`docs/README.md` 一行索引（指向前者）。提交与否待用户（TODO 21.3.4.5）；要提交就三者一起按文件名 add。
- 提交钩子 lint 整个工作区（含未跟踪的实验脚本），未格式化会挡住任何提交：`/venv/main/bin/ruff format <文件>`。STATE.md 上限 60 行。

## 磁盘与内存

- 已确认（10-09）：`/workspace` 剩 75 GB（93%），根分区 `/` 约 1.6 GB；跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。容器内存上限 91 GB，现用约 25 GB。ADT 栈约 16 GB、Unreal 带公寓关卡约 24 GB，起栈前看 `memory.current`。
- 本会话新增：`outputs/graspnet_bottle_20261008/`（812 MB，含 `videos_full/` 778 MB）、`outputs/runs/gnb_obj_*`（2.0 GB，其中调试 smoke 682 MB、batch1 154 MB）、`outputs/light_shift_20261008/`、`third_party/graspnet-baseline`（8.3 GB，运行环境）。删不删待用户（21.3.4.6）。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

本会话（TODO 21.3；DECISIONS 2026-10-08 末三节、2026-10-09 一节）：
- 已确认：21.3.3（开局 IK 判可行）不做——实际站位上 IK 误差预测成败 AUC 0.56；21.3.2（跌落测稳定姿态）不需要。
- 已确认：21.3.1（补渲随机外观）用户定直接做，计划 `docs/handoff/appearance-randomization-plan-20261008.md`，之后再测；Unreal 光照开关已提交（`d6a8208`）。待用户定评测留出目标。
- 已确认：16.15.3 训练在约 500 步被用户叫停，没有 checkpoint；`warehouse_mars_50` 数据集仍完好（45 + 5 条，50 fps，46 维 state）。
- 已确认：21.3.4.1 完成——65 个 GraspNet 物体各跑我们的 IK 一局、SIMPLE 的 CuRobo 一局，抬起 16/65 对 15/65，规划中位 1.5 s 对 17 s；66 个左右对比视频（≤ 500 KB，逐个解码过）在 `outputs/graspnet_bottle_20261008/videos/`，结果表 `summary.md`。失败主要在执行层：合爪未同时接触 15/19、抬起脱手 13/9、进入碰倒 12/8；规划失败只有 5/9。
- 已确认：CuRobo 的 17 s 里约 10 s 是新批量形状的准备（我们没把候选补齐），求解本身约 3 s。**假设**：补齐并预热后约 6 s，未实测（21.3.4.4）。
- 已确认：协作者"SIMPLE 采集 agent 有泛化抓取 / 避障 / 并行 / 轨迹优化"——功能都有，但抓取是离线缓存（`detect_grasps` 为空函数）、真值位姿、固定基座手臂、开环执行（证据见 DECISIONS 2026-10-09）。
- **假设**：每个物体每边只有一局，16 对 15 看不出两个规划器的差别；"失败在执行层"是按失败原因计数得出，没有做干预对照。
- 待验证：ScaleBFM 是否比 SONIC POSE 流送手更准——现有 84 格对比用的是 SONIC planner 通道，不可比（21.3.4.3）。

更早（仍有效）：SIMPLE 已发表我们的基础设施层，19.8 查重 Level 3；warehouse 手臂规划线已叫停；scene_discovery 在 main（mars + Unreal），mars 上 p4 1/10，5 局被右腕 D405 支架拦下。

## 下一步（编号见 `TODO.md`）

- **21.3.4.2**（待用户）：要比较两个规划器就每个物体多跑几个朝向；或接着修执行层的三类失败。
- **21.3.4.5 / 21.3.4.6**（待用户）：提交实验代码与记录文档；调试运行删不删。
- **19.18.8**：mars 上重标单手抓取几何（或查 D405 支架碰撞体），再跑 p4 10 局。

## 未解决 / 卡点

- 卡点（19.18.6）：`plan_episode --record` 录的运行补渲不了（缺 `SIM_STATE_TRACE_DIR`、`collect/inputs.json`、场景副本）。
- 待用户：21.3.1 评测留出目标、21.3.4.2–21.3.4.6、19.8.1、19.8.3、19.18.3.1、16.14.1、16.15.4、13.6.18。待协作者：13.6.16.1、13.3、13.6.11.5、16.13。
- 待验证：19.18.5、19.18.7、16.16.1、16.11 钩子偶发失败。
- 已确认（工具）：MuJoCo 3.11 `mj_geomDistance` 会返回 0.0 和反号；mink 1.3.0 碰撞上界单位错；stock G1 的 URDF 与 MJCF 躯干、肩各差 1 cm、腕差 5 mm（CuRobo 用 `outputs/graspnet_bottle_20261008/curobo/g1_stock.urdf`）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && export TMPDIR=/workspace/tmp
awk '{print $1/1e9" GB"}' /sys/fs/cgroup/memory.current; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
make check                                                       # 全量检查；提交钩子也跑它
# 21.3.4 复现（记录文档第 6 节）：服务在 GPU1，4 条通道在 GPU0 实例 1–4；停批跑 touch outputs/graspnet_bottle_20261008/STOP
bash scripts/experiments/graspnet_bottle_20261008/servers.sh start
bash scripts/experiments/graspnet_bottle_20261008/run_one.sh <物体id> <实例> 0 [_tag]   # 单个物体，两局（IK、CuRobo）
bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 4 1 0                     # 全部 65 个
PYTHONPATH=src python3 scripts/experiments/graspnet_bottle_20261008/make_videos.py outputs/graspnet_bottle_20261008
bash scripts/experiments/graspnet_bottle_20261008/servers.sh stop
bash scripts/train.sh outputs/runs/warehouse_mars_50 --gpu 0     # 16.15.3；长任务 nohup setsid，Monitor 看日志
# 提交：按文件名 add（不要 -A，别带 assets/hero.png、.serena、node_modules、papers/、step*.md）→ commit → 用户说了才 push
```
