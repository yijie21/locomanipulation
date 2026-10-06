# STATE — 2026-10-03（会话：warehouse 线换 mars + Unreal 默认 + 只走 ym 边；已推送 `4db7deb`）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认：`4db7deb` 连同之前 12 个本地提交已推到 origin/main（handoff 提交之后本地会再领先 1 个）。
已确认（10-03 收尾时）：没有留下仿真栈、Unreal 编辑器或训练进程。两张卡上常有别的项目（SimpleRoboJev 5.9 GB 进程时有时无、depth_server 3.7 GB 在 GPU1）；`train.sh` 在别人占 > 4 GB 时拒绝启动。
别的会话：henry-zhao 直接推 origin/main，推送不跑钩子，他的提交可能自带检查失败。

## 磁盘

- 已确认：`/workspace` 空闲约 125 GB；根分区 `/` 约 1.8 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12（含 13.6.12.1 本次新增约 1 GB）、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认（DECISIONS 2026-10-03）：warehouse 线默认是 mars + Unreal + 只走 ym 边。runbook 用 `warehouse_mars_smoke`（Isaac 版 `_isaac`，完整规模 `warehouse_mars_200`）；stock 的 `warehouse_smoke` 等保留、仍四边。
  - mars 的 Isaac 机器人 USD（`assets/isaac/robot/mars/`）、4 个 Isaac stage、4 个 Unreal 关卡已建；stage、关卡、补渲标记都记 `robot`，对不上拒绝。
  - 照 runbook 跑到转换（当时四边）：30 局保留 9 条，Unreal 补渲 9 局（标记 `robot: mars`），转换 8+1 条（`outputs/runs/warehouse_mars_smoke_4edges_20261003`）。
  - `editor_guard.sh` 只按编辑器自己的 GPU/CPU 判卡死和打满；计数在空载下测过，没在编辑器真卡死时测过。
- 已确认（13.6.11.1–4）：腰抖动的机制和 50 N·m 的来源已查清（DECISIONS 腰部一节）；仿真设置没查出错。
- 已确认（13.6.2–13.6.4）：配对 10 局 bottle 9/10 = stock；warehouse 腰 50 时 6/10 = stock。mars 失败都是抬起后单手脱开（13.6.4.1）。
- 已确认（henry 10-03，`33e8cd8`..`5ff2d7c`，10-04 rebase 合入）：mars 的 MJCF 和网格缺失时由 `make_dex1_model.write_mars()` 生成，`build_isaac_assets.sh` 与 `prepare_logistics_scene.py` 先调它；Unreal 导出丢掉有可视体的 body 上的碰撞 geom（`export_mjcf.strip_to_dynamic`，腕部 D405 曾被碰撞盒以 UE 默认灰格子罩住）；D405 支架的碰撞圆盘和盒子放 group 3，仍碰撞、相机不画（参照新增的 `assets/unitree_G1_with_wrist_camera_new`）；URLab 补丁按 30° 棱角拆分顶点，Unreal 里 D405 背面不再呈内凹（`patches/urlab_fresh_capture.patch`）。合入时他的提交带着 3 处检查失败，10-04 已修（用户）：`f156df1` 把 `warehouse_mars_smoke` 的 `start_edges` 改回四边（已改回 `[ym]`），新 README 没过 ruff format，`docs/warehouse.md` 超 220 行（上限放宽到 300）。
  - 已确认（10-04，用户）：mars 小臂改银色；外壳按真机照片改为缎面银（metallic 0.75、roughness 0.6、底色 0.45，外观名 `g1_satin_20261004`，stock 共用）；mars 腰部电机和骨盆画黑（DECISIONS 2026-10-04 两节）。mars 模型、7 个 `*_mars` 场景、几何样例已重建，本机 9 个 Unreal 关卡 10-04 按新外观名重建；协作者 pull 后要自己重建（见提交说明）；Isaac 机器人 USD 下次用到时按 MJCF 哈希自动重建，Isaac 外观没对照片核过（13.6.14.1）。
  - 已确认（10-06，用户）：第一视角可选脸上的 D435（场景 `*_mars_d435`，实验 `warehouse_mars_smoke_d435_unreal`、`_d435_isaac`、`_d435_mujoco`，DECISIONS 2026-10-06）；本机已建场景、Isaac stage、Unreal 关卡。D435 上专家产出率未测（13.6.15.1）。

## 待验证 / 卡点

- 待验证（13.6.5.6）：只走 ym 的 mars 采集没实跑，只 `--dry-run` 核了摆放（全部 ym，起点 0.6–1.035 m）。
- 待验证（13.6.5.7）：runbook 第 5–7 步在 mars 上没跑（用户：暂不训练）；13.6.5.3 Isaac 版 mars 没实渲。
- 待用户（13.6.5.4）：mars 四边冒烟 9/30，stock 09-30 是 11/20、8/20；分不清是机器人还是波动。
- 风险（13.6.11）：真机腰固件上限未知；专家抱箱要腰约 44 N·m（p95），真机若是 35 会塌、手抖。待协作者 13.6.11.5。
- 待协作者（13.3）：D455 / D405 真机内参。未做：bottle、crate 仍是 stock（13.6.7.2）。
- 同前未结：14.4、14.5、9.11.1、12.9.13、12.20.4、12.9.6。

## 下一步（编号见 `TODO.md`）

- **13.6.5.6**：按 runbook 跑一次只走 ym 的 `warehouse_mars_smoke` 第 2–4 步（约 40 分钟），看保留率。
- **13.6.4.1**：腰 50 下的单变量对照（支架不碰撞、腕 5 N·m），查 mars 抬起后单手脱开。
- **13.6.11.5**：请协作者在真机上记腰 13/14 号电机的 `tau_est`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp MUJOCO_GL=egl
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader; cat /sys/fs/cgroup/pids.current
# warehouse 冒烟（默认 mars、Unreal、只走 ym）：照 docs/warehouse.md；编辑器开在 GPU 1
bash scripts/start_unreal.sh --gpu 1
bash scripts/bimanual/run_agent_server.sh configs/experiments/warehouse_mars_smoke.yaml --dry-run   # 先看每局摆放
# 本次的队列脚本（重试 pid/显存不足）：outputs/mars_assets_20261003/smoke{,_tail}.sh，日志 smoke/queue.log
bash scripts/build_scene.sh warehouse_box_45x32x33_mars --isaac    # mars 机器人 USD + stage
bash scripts/build_scene.sh warehouse_box_45x32x33_mars --unreal   # 首次导入可能 zmq.error.Again，重跑即可
PYTHONPATH=src $VENV_SIM_PY outputs/mars_pilots_20261002/bend80/osc.py   # 弯腰抖动分析（另有 analyse.py、waist_load/）
# 提交：按文件名 add（zsh 下变量不分词，用数组；不要 -A，别带上 assets/hero.png）→ commit（钩子跑全量检查）→ 用户说了才 push
```
