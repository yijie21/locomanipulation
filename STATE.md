# STATE — 2026-10-02（会话：mars 真机 URDF 设为 bottle / warehouse 默认机器人，第 1–2 步完成，停在用户审阅）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认：本地比 origin/main 多 `577ce78`（合入 henry 的 `cb3ae2f`）、`62ed89f`、`81ef9cd` 和本次 handoff，均未推送（推送前问用户）。
已确认（10-02 第 2 步结束时）：试跑队列都已结束，没有留下仿真栈；GPU0 约 0 GB，GPU1 约 3.7 GB（别的项目）。
GPU0 白天有别的会话起过仿真栈（`loco-agent` 的 scratchpad 树），上卡前先看 `nvidia-smi`。
别的会话：henry-zhao 直接推 origin/main，推送不跑钩子，他的提交可能自带检查失败（这次 `assets/unitee_G1/README.md` 就是）。

## 磁盘

- 已确认：`/workspace` 空闲 147 GB；根分区 `/` 只剩约 1.8 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认（TODO 13.6；DECISIONS 2026-10-02）：mars 机器人与 7 个 `*_mars` 场景（`62ed89f`、`81ef9cd`）。原场景名仍是 stock。
  - `make_dex1_model.py` 写 `g1_29dof_with_dex1_mars.xml`；`prepare_logistics_scene.py --robot mars` 改写 warehouse 包里的机器人。
  - 已确认：全量测试过；stock 模型重生成逐字节不变；ego D455 俯仰 52°、fovy 65，腕部 D405 fovy 58（规格值，无标定）。
  - 已确认：外观名仍 `g1_gloss_20261001`（stock 数值没变，改名会废掉现有关卡）；mars 靠 `scene_meta.json` 的 `robot` 区分。
- 已确认（13.6.2–13.6.3）：配对试跑在 `outputs/mars_pilots_20261002/`（对比表 `*_stages.md`、`warehouse_ablations.md`；预览视频 `previews/`，每个 < 500 KB）。
  - bottle：stock 9/10，mars 9/10。
  - warehouse：stock 6/10，mars 0/10。原因是腰 roll/pitch 35 N·m：只把它恢复 50 就 5/10，其余 6 个单变量都 0/10。

## 待验证 / 卡点

- 待用户（13.6.4）：warehouse 在 35 N·m 腰下抱不起箱子（腰俯仰塌到 ~35°；stock 抓箱姿势要 ~39 N·m）。选项：仿真腰用 50、改专家少弯腰、或先确认真机腰的固件上限（用户 09-30 只确认过踝是 35）。
- 待用户：看完第 2 步再做第 3 步（13.6.5：mars 的 Isaac 资产、warehouse `_mars` 的 Isaac 场景和 Unreal 关卡）。
- 待验证（14.5）：新外观下的数据集补渲和评测镜像没单独实跑过。
- 待用户（14.4）：Isaac 深色件要不要单独压暗；（14.6）旧外观数据集/checkpoint 是否按新外观重做。
- 待协作者（13.3）：D455 / D405 的真机内参（仓库里没有）。
- 待验证（9.11.1）：没有新读者照 warehouse.md 从头跑过。（12.9.13）没人照文档从零装过 Unreal。（12.20.4）`max_batches: 3` 没实测。
- 未做（12.9.6，待用户安排）：用 Unreal 补渲的数据重训。

## 下一步（编号见 `TODO.md`）

- **13.6.4**：用户定 warehouse 的腰力矩处理，再按决定重跑 warehouse 配对试跑。
- **13.6.5–13.6.7**：用户同意后建 mars 的 Isaac/Unreal、`robot` 字段拒绝检查、runbook 改用 `*_mars`、推送。
- **14.4 / 14.5**、**12.20.3 / 14.7**：同前（Isaac 深色件；删哪些产物）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
df -h / /workspace; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; cat /sys/fs/cgroup/pids.current; export TMPDIR=/workspace/tmp
# 改了 src/locomani/sim/robot_look.py（数值和 NAME 一起改）之后：重建全部 Unreal 关卡
bash scripts/start_unreal.sh --gpu 1
for s in warehouse_textured warehouse_box_45x32x33 warehouse_box_45x20x33 warehouse_box_45x32x18 warehouse_box_45x20x18; do bash scripts/build_scene.sh $s --unreal; done
bash scripts/stop_unreal.sh
# 新外观实渲核对用的沙盒运行（只放了 config、manifest 和指向原运行 trace/scene 的软链）
bash scripts/render_third_person.sh outputs/g1_look_v4a_check_20261001/run_isaac visual_crate_s7000_00 --gpu 1
# mars 模型和场景（third_party 里的生成文件 gitignore，新 clone 先跑）
PYTHONPATH=src $VENV_SIM_PY src/locomani/sim/make_dex1_model.py
bash scripts/build_scene.sh bottle_mars; bash scripts/build_scene.sh warehouse_textured_mars; bash scripts/build_scene.sh warehouse_box_45x32x33_mars
# 配对试跑（10 局、不录数据）：configs/experiments/{bottle,warehouse}_{stock,mars}_pilot_20261002.yaml
# 提交：按文件名 add（不要 -A，别带 assets/hero.png）→ commit（钩子跑全量检查）→ fetch，远端有新提交就 merge → 用户说了才 push
```
