# STATE — 2026-10-03（会话：warehouse 线换 mars + Unreal 默认 + 只走 ym 边；mars 的 Isaac/Unreal 资产建好）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认（10-03）：用户说了提交并推送，本会话的提交连同之前 12 个本地提交一起推到 origin/main。
已确认（10-03 收尾时）：没有留下仿真栈、Unreal 编辑器或训练。两张卡上都有别的项目（SimpleRoboJev 的 5.9 GB 进程时有时无、depth_server 3.7 GB）；上卡前先看 `nvidia-smi`，`train.sh` 在别人占 > 4 GB 时拒绝启动。
别的会话：henry-zhao 直接推 origin/main，推送不跑钩子，他的提交可能自带检查失败（`cb3ae2f` 的 README 就过不了 ruff，已在 `62ed89f` 里格式化）。

## 磁盘

- 已确认：`/workspace` 空闲 127 GB；根分区 `/` 约 1.9 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认（DECISIONS 2026-10-03 两节）：warehouse 线默认是 mars + Unreal + 只走 ym 边。runbook 用 `warehouse_mars_smoke`（Isaac 版 `_isaac`），stock 的 `warehouse_smoke` 等保留、仍四边。
  - mars 的 Isaac 机器人 USD（`assets/isaac/robot/mars/`）、4 个 Isaac stage、4 个 Unreal 关卡已建；stage、关卡、补渲标记都记 `robot`，对不上拒绝。
  - 照 runbook 跑到转换（当时四边）：30 局保留 9 条，Unreal 补渲 9 局，转换 8+1 条（`outputs/runs/warehouse_mars_smoke_4edges_20261003`）。训练和评测按用户要求没跑。
- 已确认（TODO 13.6；DECISIONS 2026-10-02、2026-10-03）：mars 机器人与 7 个 `*_mars` 场景已建好，原场景名仍指向 stock。
  - `make_dex1_model.py` 写出 `g1_29dof_with_dex1_mars.xml`（自检：FK 误差 3e-7 mm，质量 35.02 kg）；`prepare_logistics_scene.py --robot mars` 改写 warehouse 场景包里的机器人。
  - 力矩：按 URDF（踝 35、髋 roll 139、腕 13.4），但腰 roll/pitch 用 50（`MARS_EFFORT_OVERRIDES`，用户定；URDF 和 Unitree 官方都是 35）。
  - 相机：ego 是 D455（俯仰 52°、fovy 65），腕部是 D405（fovy 58），都是规格值，没有标定。
  - 外观：手腕三节、Dex1 底座、D405 支架画黑；D405 本体银色（10-03 曾改黑又改回：Unreal 里看到的灰格子不是它，是罩着它的碰撞盒）；头顶 D455 银白；骨盆银色（`robot_look.SHELL_MESHES`）；碰撞体取所在部件的颜色。外观名仍是 `g1_gloss_20261001`。
  - 已确认：7 个场景和测试样例都按最新提交 `b46e0ce` 重建；全量测试通过；stock 模型重新生成后逐字节不变。10-03：Unreal 导出不再带碰撞体（`export_mjcf.strip_to_dynamic`，mars 的碰撞体 alpha 1，腕部 D405 碰撞盒以 UE 默认灰格子材质罩住相机）；mars 模型和 `outputs/scene/warehouse_textured_mars`、`warehouse_box_45x32x33_mars` 已重建，`assets/isaac/robot/mars` 已删；Unreal 关卡待用户重建后重渲。
- 已确认（13.6.2–13.6.4，`outputs/mars_pilots_20261002/`）：配对 10 局，bottle stock 9/10、mars 9/10；warehouse 腰 35 时 0/10，腰 50 时 6/10 = stock。mars 的 4 局失败都是抬起后单手脱开，原因待查（13.6.4.1）。

## 待验证 / 卡点

- 未验证（13.6.5.6）：只走 ym 的 mars 采集没实跑，只 `--dry-run` 核了摆放。
- 待用户（13.6.5.4）：mars 四边冒烟保留 9/30，stock 09-30 是 11/20、8/20；要不要配对重跑。
- 待用户：13.6.4.1（mars 抬起后单手脱开）的腰 50 单变量对照还没做。
- 风险（13.6.11）：专家抱箱时的前倾要腰俯仰约 39 N·m；真机的腰如果真是官方的 35，就撑不住这个姿势。
- 待协作者（13.3）：D455 / D405 的真机内参（仓库里没有）。
- 未做：bottle、crate 仍是 stock（13.6.7.2）；Isaac 版 mars 没实渲（13.6.5.3）；runbook 第 5–7 步在 mars 上没跑。同前未结：14.4、14.5、9.11.1、12.9.13、12.20.4、12.9.6。

## 下一步（编号见 `TODO.md`）

- **13.6.5.6**：按 runbook 跑一次只走 ym 的 `warehouse_mars_smoke`（第 2–4 步约 40 分钟），确认保留率。
- **13.6.4.1**：腰 50 下的单变量对照（支架不碰撞、腕 5 N·m），沿用 `outputs/mars_pilots_20261002/queue*.sh` 的写法。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp MUJOCO_GL=egl
df -h / /workspace; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; cat /sys/fs/cgroup/pids.current
# mars 模型（third_party 里的生成文件被 gitignore；改了 make_dex1_model.py 或 robot_look.py 后要重跑）
PYTHONPATH=src $VENV_SIM_PY src/locomani/sim/make_dex1_model.py
# 重建 7 个 mars 场景：先把旧的 outputs/scene/*_mars 挪走（生成器不会覆盖已有目录），再按 configs/scenes.yaml 的 build 建
bash scripts/build_scene.sh bottle_mars   # 会重新解包场景包；本会话只跑了其中 build_sim_scene.py 那一步
bash scripts/build_scene.sh warehouse_textured_mars
for s in box_platform_mars warehouse_box_45x32x33_mars warehouse_box_45x20x33_mars warehouse_box_45x32x18_mars warehouse_box_45x20x18_mars; do bash scripts/build_scene.sh $s; done
# 测试样例要跟着 mars 场景一起重建
$VENV_SIM_PY tests/fixtures/geometry/build_fixture.py outputs/scene/sim_lowcom_mars/scene_locomani.xml /workspace/tmp/fx && cp /workspace/tmp/fx/{scene_locomani.xml,scene_meta.json,source.json} tests/fixtures/geometry/
# 配对试跑（10 局、不录数据）：用 nohup setsid 在后台跑，日志在 outputs/mars_pilots_20261002/queue.log
bash scripts/start_stack.sh collect configs/experiments/warehouse_mars_pilot_20261002c.yaml --gpu 0
bash scripts/bimanual/run_agent_server.sh configs/experiments/warehouse_mars_pilot_20261002c.yaml --gpu 0
python3 outputs/mars_pilots_20261002/stage_table.py warehouse c     # stock 和 mars 的逐局对比表
python3 outputs/mars_pilots_20261002/make_previews.py warehouse c   # 左右并排的预览视频，每个 < 500 KB
PYTHONPATH=src $VENV_SIM_PY outputs/mars_look_compare_20261002/compare.py <tag>   # 外观核对图（上排 stock、下排 mars）
# warehouse 冒烟（默认 mars、Unreal、只走 ym）：照 docs/warehouse.md，编辑器 start_unreal.sh --gpu 1
# 提交：按文件名 add（不要 -A，别带上 assets/hero.png）→ commit（钩子跑全量检查）→ 用户说了才 push
```
