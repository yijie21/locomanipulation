# STATE — 2026-10-08（会话：scene_discovery 合入 main、改用 mars + Unreal、合并前后回归、删旧 worktree）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-08）：到 `b8f5e9c` 为止都已推到 origin/main；之后 39 个提交（warehouse 线的 handoff、scene_discovery 合入、STATE）都**未推送**，推送等用户说（TODO 19.18.10）。
已确认（10-08 收尾时）：本机没有我留下的仿真栈、Unreal 编辑器或排队脚本（stack 实例 0 / 1 已按状态文件收掉，`stop_unreal.sh` 已跑）。GPU1 上常驻别的项目的 depth_server 约 3.7 GB。 TODO 20 的手臂规划线（CuRobo → mink 碰撞约束，`visual.collision_guard` 默认 off，20 局配对 off 9/20、mink 7/20）已由用户叫停，没有留下栈或进程（DECISIONS 2026-10-08 最后一节）。
已确认：loco_agent 那个 Claude 会话（pid 962611）还开着，它的工作目录 `../locomanipulation-loco_agent` 已删，里面再跑命令会失败；用户在那个终端退出它（19.18.13）。
别的会话：henry-zhao 直接推 origin/main 或 `henry-dev`，推送不跑钩子，可能自带检查失败（记忆 origin-main-can-carry-failing-checks）。

## 磁盘与内存

- 已确认（10-08）：`/workspace` 剩 87 GB（92%），根分区 `/` 约 1 GB，跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。本次新增：`outputs/scene_discovery/adt_apartment_mars` 1.6 GB（含房间烘焙 `unreal_room/room.glb` 1.1 GB）；Unreal 项目 Content 1.7 GB、`/workspace/ue_57/levels/adt_apartment_mars` 0.4 GB；回归运行 `outputs/runs/{sync,presync}_regress_*_20261008*`、`sync_mars_p4_20261008`。
- 已确认（10-08）：容器内存上限 91 GB。Unreal 编辑器带公寓关卡约 24 GB，一个 ADT 栈约 16 GB；04:06–07:02 两个 ADT 栈 + 编辑器 + 测试把内存打满三小时、deploy 全死（DECISIONS 2026-10-08 续）。**ADT 栈一次只跑一个，起栈前看 `memory.current`**。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

scene_discovery（TODO 19.18，DECISIONS 2026-10-08 两节）：
- 已确认：已快进进 main（`10b3895` 合并、`0fcceaf` mars + Unreal、`6f4d645` 台账 / TODO）；合入前的分支是 tag `scene_discovery-pre-sync`（`a73cf0c`）。分支 TODO 7/8/9 现在是 **17/18/19**。三个旧 worktree 已删，分支 `scene_discovery-sync` / `scene_discovery` / `loco_agent` 还在（19.18.12 待定）。
- 已确认：`make check` 全过（2393 passed、5 skipped），提交钩子每次都过。
- 已确认：场景 `adt_apartment_mars`（`configs/scenes.yaml`，目录只在 `outputs/`）`check_mars` 过；Unreal 关卡 `adt_apartment_mars`（房间关卡 `AdtApartment`）建成；同状态 Unreal 与 MuJoCo 三路并排对得上：`outputs/renders/adt_mars_check_20261008/unreal_vs_mujoco_{spawn,facing_canister}.png`。
- 已确认（pick_place 台账 §18）：同机同 seed 合并后 / 合并前 bottle 7/10 vs 6/10、pick_place 2/10 vs 4/10、p4 3/10 vs 3/10，看不出差别（10 局只排除大退化）。
- 已确认：mars 上 p4 1/10，其中 5 局抓取前的间隙检查判右腕 `right_wrist_yaw_link`（带 D405 支架）碰到罐子；stock 上没有这一类。

warehouse mars 线（别的会话 10-07/08 的结果，仍有效）：
- 已确认：时序对齐官方（50 fps、执行 20 拍、相机延迟）、state 改回官方 46 维、旧 53/60 维 checkpoint 不能再服务或评测（DECISIONS 2026-10-07）；Dex1 方向维持 0 闭、5.5 开（真机未验，16.6）。`warehouse_mars_50` 数据集做完没训练：训练集 45 条、验证集 5 条，50 fps，`outputs/runs/warehouse_mars_50/{train,val}_dataset`，Unreal 补渲的视频在各自 `videos/chunk-000/observation.images.*`。

## 待验证 / 卡点

- 卡点（19.18.6）：`plan_episode --record` 录的运行补渲不了（不设 `SIM_STATE_TRACE_DIR`、不写 `collect/inputs.json`、不复制场景且 ADT 场景文件 `target_object` 为空），所以 mars + Unreal 的录制 → 补渲 → 转换没走通。冒烟实验文件已备好：`outputs/scene_discovery/adt_apartment_mars/experiments/p4_canister_on_round_table_unreal.yaml`。
- 待验证（19.18.5）：`capability.yaml` 的包络全是 stock 机器人的实测；检查器对 mars 的分桶未验证。
- 待用户（19.18.3.1）：公寓 Unreal 曝光沿用 warehouse 的数值、未校准，灯下木地板偏亮；看图定要不要调。
- 待验证（19.18.7）：双臂技能不读 `visual.ready_after_brake`，没在 mars / main 新默认上跑过。
- 假设：mars 上 p4 掉到 1/10 主要是腕部碰撞；另外 5 局（stand 2、grasp 2、approach 1）与 stock 的失败族相同，10 局分不出 mars 是否也更差。
- 待验证（16.16.1）：实时录制按 50 fps 转换时审计 "action vs state" +0 帧不过，只在 1 条的 `crate_smoke` 上见过；（16.16.4）实时录制的图实测约 30.3 Hz，迁移文档写的 33 要改。
- 待用户：16.14.1（批 32、20000 步）、16.15.4（`check_dataset` 漏审 warehouse 双臂档）、13.6.18（warehouse mars 实验文件顶层 `backend.dataset_fps: 30.0` 与 `dataset.fps` 不一致）。
- 待协作者：真机相机帧率与延迟（13.6.16.1）、相机内参（13.3、13.6.15.2）、腰固件力矩上限（13.6.11.5）、真机 C++ 踝增益（16.13）；告诉 henry 16.10、16.12.1、16.16.3。
- 待验证（同前未结）：16.11 钩子偶发失败；13.6.15.1、14.4、14.5、9.11.1、12.9.13、12.20.4、12.9.6。

## 下一步（编号见 `TODO.md`）

- **19.18.8**：mars 上重标单手抓取几何（或查 D405 支架碰撞体是否比实物大），再跑 p4 10 局。
- **19.18.6**：补通 `plan_episode --record` → `render_dataset_cameras_unreal.sh` → `convert_data.sh` 的三个缺口（19.18.6.1–3），用上面的冒烟实验文件录几局走一遍。
- **16.15.3**（warehouse 线）：训练 `warehouse_mars_50`（单卡 1 万步）再闭环评测 100 局。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp
awk '{print $1/1e9" GB"}' /sys/fs/cgroup/memory.current; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
make check                                                       # 全量检查；提交钩子也跑它
# scene_discovery 单任务试跑（ADT 栈，一次只跑一个；deploy 根照分支的用法）
export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/outputs/bottle_continuous_20260914/deploy
A=outputs/scene_discovery/adt_apartment_mars
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u scripts/plan_episode.py $A/experiments/p4_canister_on_round_table.yaml --gpu 0 --instance 1 --rounds 10 --seed 700 --name <run>
# 重建公寓 Unreal 关卡（做法在 scenes.yaml adt_apartment_mars 的 build、unreal.room_build、unreal.build）
bash scripts/start_unreal.sh --gpu 1 --cpus 24-27 && "$BRIDGE_PY" -m locomani.sim.unreal.build_level adt_apartment_mars
bash scripts/train.sh outputs/runs/warehouse_mars_50 --gpu 0     # 16.15.3；长任务用 nohup setsid 脱离，Monitor 看日志
# 停自己的栈：按 /workspace/tmp/p03*/stack.json 的实例收（Stack.attach(i).down()），别用不分实例的 stop_stack
# 提交：按文件名 add（不要 -A，别带上 assets/hero.png、.serena、node_modules）→ commit → 用户说了才 push
```
