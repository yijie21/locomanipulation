# STATE — 2026-10-03（会话：mars 真机 URDF 设为 bottle / warehouse 默认机器人；第 1–2 步完成，腰改 50，外观对齐 stock）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认：本地比 origin/main 多 13 个提交（从 `577ce78` 合入 henry 的 `cb3ae2f` 起，到本次 handoff），都没推送（推送前先问用户）。
已确认（10-03 收尾时）：本会话的试跑队列全部结束，没有留下仿真栈。GPU0 约 2 GB、GPU1 约 10 GB，都是别的项目（JeVLA / SimpleRoboJev、depth_server）；上卡前先看 `nvidia-smi`。
别的会话：henry-zhao 直接推 origin/main，推送不跑钩子，他的提交可能自带检查失败（`cb3ae2f` 的 README 就过不了 ruff，已在 `62ed89f` 里格式化）。

## 磁盘

- 已确认：`/workspace` 空闲 127 GB；根分区 `/` 约 1.9 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认（TODO 13.6；DECISIONS 2026-10-02、2026-10-03）：mars 机器人与 7 个 `*_mars` 场景已建好，原场景名仍指向 stock。
  - `make_dex1_model.py` 写出 `g1_29dof_with_dex1_mars.xml`（自检：FK 误差 3e-7 mm，质量 35.02 kg）；`prepare_logistics_scene.py --robot mars` 改写 warehouse 场景包里的机器人。
  - 力矩：按 URDF（踝 35、髋 roll 139、腕 13.4），但腰 roll/pitch 用 50（`MARS_EFFORT_OVERRIDES`，用户定；URDF 和 Unitree 官方都是 35）。
  - 相机：ego 是 D455（俯仰 52°、fovy 65），腕部是 D405（fovy 58），都是规格值，没有标定。
  - 外观：手腕三节、Dex1 底座、D405 支架画黑；头顶 D455 银白；骨盆银色（`robot_look.SHELL_MESHES`）；碰撞体取所在部件的颜色。外观名仍是 `g1_gloss_20261001`。
  - 已确认：7 个场景和测试样例都按最新提交 `b46e0ce` 重建；全量测试通过；stock 模型重新生成后逐字节不变。
- 已确认（13.6.2–13.6.4）：配对 10 局试跑的结果在 `outputs/mars_pilots_20261002/`：`*_stages.md`、`warehouse_ablations.md`、`jitter/jitter.md`，预览视频在 `previews/`（每个 < 500 KB）。
  - bottle：stock 9/10，mars 9/10。
  - warehouse：腰 35 时 0/10，手抖 9.4 mm；腰 50 时 6/10（与 stock 的 6/10 相同），手抖 2.8 mm（与 stock 相同）。
  - 已确认：腰 50 下，mars 的 4 局失败都是箱子抬起后单手脱开，stock 没有这种失败；原因待查（13.6.4.1）。
  - 假设（未重跑验证）：这些试跑用的是外观修改前的场景；外观只改了颜色，不影响 `record: false` 试跑的物理结果。

## 待验证 / 卡点

- 待用户（13.6.10）：先做 13.6.4.1 的腰 50 单变量对照，还是先开始第 3 步（13.6.5，Isaac/Unreal）。
- 风险（13.6.11）：专家抱箱时的前倾要腰俯仰约 39 N·m；真机的腰如果真是官方的 35，就撑不住这个姿势。
- 待协作者（13.3）：D455 / D405 的真机内参（仓库里没有）。
- 未做（13.6.5–13.6.7）：mars 的 Isaac/Unreal 资产、`robot` 字段检查、runbook 改用 `*_mars`。同前未结：14.4、14.5、9.11.1、12.9.13、12.20.4、12.9.6。

## 下一步（编号见 `TODO.md`）

- **13.6.10**：问用户先查脱手（13.6.4.1），还是先做第 3 步（13.6.5）。
- **13.6.4.1**：腰 50 下的单变量对照（支架不碰撞、腕 5 N·m），沿用 `outputs/mars_pilots_20261002/queue*.sh` 的写法。
- **13.6.5**：mars 的 Isaac 机器人资产（按机器人名分开存）、warehouse `_mars` 场景的 Isaac 场景和 Unreal 关卡。

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
# 提交：按文件名 add（不要 -A，别带上 assets/hero.png）→ commit（钩子跑全量检查）→ 用户说了才 push
```
