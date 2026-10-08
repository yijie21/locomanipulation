# STATE — 2026-10-08（会话：SIMPLE 论文调研、19.8 查重、warehouse 手臂规划 CuRobo → mink，已叫停）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-08）：到 `b8f5e9c` 为止都已推到 origin/main；main 现在比 origin 多 41 个提交（含本会话 `75ad611` 和 handoff），**未推送**，等用户说（TODO 19.18.10）。
已确认（10-08 收尾时）：本机没有本会话留下的仿真栈、Unreal 编辑器、排队脚本或 CuRobo / mink 进程；GPU1 上常驻别的项目的 depth_server 约 3.7 GB，SimpleRoboJev 的 6 GB worker 会来来去去（记忆 shared-cards-other-projects）。
已确认：loco_agent 那个 Claude 会话（pid 962611）的工作目录已删，用户在那个终端退出它（19.18.13）。henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 磁盘与内存

- 已确认（10-08 本会话末）：`/workspace` 剩 87 GB（92%），根分区 `/` 剩 1.6 GB；跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。容器内存上限 91 GB，现用约 36 GB。
- 已确认：ADT 栈约 16 GB、Unreal 编辑器带公寓关卡约 24 GB，**ADT 栈一次只跑一个，起栈前看 `memory.current`**（DECISIONS 2026-10-08 续）。
- 本会话新增产物：`outputs/{curobo,mink}_warehouse_20261008/`（34 MB / 247 MB）、`outputs/runs/warehouse_mars_guard_{off,mink}_20261008`（约 0.9 GB）、`outputs/simple_tabletop_grasp_mp_20261008/`（17 MB），未跟踪的 `papers/`（183 MB）和仓库根 `step1.md`–`step7.md`（19.8.3 待用户定去处）。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

本会话（2026-10-08 下午，TODO 21.3；DECISIONS 末节）：21.3.2 / 21.3.3 不做；21.3.1 直接做，计划 `docs/handoff/appearance-randomization-plan-20261008.md`，之后再测，待用户定评测留出目标。16.15.3 训练停在约 500 步；孤儿 pid 2122899 经用户同意已杀。Unreal 编辑器已关，本会话没有留下 GPU 进程。光照开关（`in_editor/lighting.py`、镜像 `--light-scale/--light-temperature`）已提交，`make check` 过。

上一会话（TODO 19.8、20、21；DECISIONS 2026-10-08 最后三节）：
- 已确认：SIMPLE 已发表我们的基础设施层，没做 LLM 发现任务 / 写 plan / 按缺口长技能；它的自动采集是 AMO + CuRobo 开环 + 手写分解，SONIC 线没有自动采集（21.1，记忆 simple-paper-covers-infra-layer）。
- 已确认：19.8 查重 Level 3；"按缺口长技能"已被 Agentic Skill Discovery 写过。决策 #1 改不改措辞**待用户**（19.8.1），计划文档没动。
- 已确认：warehouse 手臂规划线**用户叫停**。CuRobo 离线对照停在 61/238；mink 碰撞约束 `visual.collision_guard`（默认 off，`75ad611`）真栈 20 局配对 off 9/20、mink 7/20（p 0.75），实测的手两组都只有 2/20 压进箱壁（台账 crate.md 2026-10-08 最后一节，记忆 warehouse-collision-guard-null-result）。
- 已确认：同配置同 seed，`warehouse_mars_50` 3/20、这次 off 9/20，20 局只有 14 局结局相同——这条线 20 局配对量不出几局的差别。

scene_discovery（TODO 19.18，上一会话，仍有效）：
- 已确认：已快进进 main（tag `scene_discovery-pre-sync` = 合入前分支头）；`make check` 全过；场景 `adt_apartment_mars` 与 Unreal 关卡建成，三路并排对得上。
- 已确认：mars 上 p4 1/10，其中 5 局抓取前的间隙检查判右腕（带 D405 支架）碰到罐子；stock 上没有这一类。

warehouse mars 线（别的会话 10-07/08，仍有效）：`warehouse_mars_50` 数据集做完没训练（训练 45 条、验证 5 条，50 fps）；state 已改回官方 46 维。

## 待验证 / 卡点

- 卡点（19.18.6）：`plan_episode --record` 录的运行补渲不了（缺 `SIM_STATE_TRACE_DIR`、`collect/inputs.json`、场景副本，ADT 场景文件 `target_object` 为空）。冒烟实验文件：`outputs/scene_discovery/adt_apartment_mars/experiments/p4_canister_on_round_table_unreal.yaml`。
- 待验证：19.18.5（capability.yaml 包络只在 stock 上测过）、19.18.7（双臂技能不读 `ready_after_brake`）、16.16.1、16.11 钩子偶发失败。
- 假设：mars 上 p4 掉到 1/10 主要是腕部碰撞；另外 5 局与 stock 失败族相同，10 局分不出 mars 是否更差。
- 待用户：19.8.1、19.8.3、21.3.1 留出目标、19.18.3.1、16.14.1、16.15.4、13.6.18。待协作者：13.6.16.1、13.3、13.6.11.5、16.13。
- 已确认（工具）：MuJoCo 3.11 `mj_geomDistance` 会返回 0.0 和反号；mink 1.3.0 碰撞上界单位错（记忆 mujoco-geomdistance-and-mink-collision-bugs）。

## 下一步（编号见 `TODO.md`）

- **19.18.8**：mars 上重标单手抓取几何（或查 D405 支架碰撞体是否比实物大），再跑 p4 10 局。
- **19.18.6**：补通 `plan_episode --record` → 补渲 → 转换的三个缺口，用上面的冒烟实验文件走一遍。
- **16.15.3**：训练 `warehouse_mars_50`（单卡 1 万步）再闭环评测 100 局。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp
awk '{print $1/1e9" GB"}' /sys/fs/cgroup/memory.current; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
make check                                                       # 全量检查；提交钩子也跑它
export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/outputs/bottle_continuous_20260914/deploy   # scene_discovery 试跑用
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u scripts/plan_episode.py outputs/scene_discovery/adt_apartment_mars/experiments/p4_canister_on_round_table.yaml --gpu 0 --instance 1 --rounds 10 --seed 700 --name <run>
bash scripts/train.sh outputs/runs/warehouse_mars_50 --gpu 0     # 16.15.3；长任务 nohup setsid，Monitor 看日志
# collision guard 重开（默认 off）：PYTHONPATH 加 outputs/mink_warehouse_20261008/pylib，实验文件 warehouse_mars_guard_mink_20261008
# 多条实验并排：实验文件写 collect.lanes: 1，否则 start_stack collect 会自动多开通道、占别人的实例
# 停栈只停自己的实例：bash scripts/stop_stack.sh --instance <k>
# 提交：按文件名 add（不要 -A，别带 assets/hero.png、.serena、node_modules、papers/、step*.md）→ commit → 用户说了才 push
```
