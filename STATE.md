# STATE — 2026-10-01（会话：mars URDF 渲染；G1 材质改成 v4a，Isaac 与 Unreal 共用 `robot_look.py`，已推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认：`1e6380e` 已推送，与 origin/main 同步；henry 09-30 的 6 个提交都在 main 里，没有待合并的内容。
已确认（10-01 收尾时）：本会话没有留下在跑的 Unreal 编辑器或 Isaac；GPU0 约 0 GB，GPU1 约 3.7 GB（别的项目）。
GPU0 白天有别的会话起过仿真栈（`loco-agent` 的 scratchpad 树），上卡前先看 `nvidia-smi`。
别的会话：henry-zhao 直接推 origin/main，推送不跑钩子，他的提交可能自带检查失败（这次 `assets/unitee_G1/README.md` 就是）。

## 磁盘

- 已确认：`/workspace` 空闲 147 GB；根分区 `/` 只剩约 1.8 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认，已推送（`f7eadec`；TODO 14.3；DECISIONS 2026-10-01 材质一节）：G1 材质只在 `src/locomani/sim/robot_look.py` 定义，外观名 `g1_gloss_20261001`。
  - Isaac 的 `stage.G1_LOOKS` 由它生成。Unreal 建关卡时写入材质，`source.json` 记 `robot_look`；对不上时 `run_context.unreal_scene` 拒绝并提示重建。
  - 两种数据集补渲标记都记 `robot_look`；没有这个字段的旧会话按另一种外观处理，不会续补。
  - 已确认：本机 5 个 Unreal 关卡已重建（新外观，第三视角 1280×960）；旧 `source.json` 备份在 `outputs/g1_look_v4a_check_20261001/levels_before/`。
  - 已确认：Isaac、Unreal 各实渲一局第三视角带三路小窗（对比图在同一目录）；提交钩子的全量检查通过。
  - 已确认：Isaac 的深色件仍比 Unreal 亮（TODO 14.4）。
  - 假设：别的 checkout 若还是旧代码，不检查外观名，会直接用上新外观的关卡（共用 `/workspace/ue_57/levels`）。
- 已确认（TODO 13.2.1）：henry 的 mars URDF 在 `assets/unitee_G1/`，多角度渲染在 `outputs/g1_urdf_render_20261001/`；用户还没说这份对不对。
  - 已确认：它没有腕部相机（13.2.3）。saturn patch 在 `1e6380e` 上 `git apply --check` 通过。

## 待验证 / 卡点

- 待验证（14.5）：新外观下的数据集补渲和评测镜像没单独实跑过。
- 待用户（14.4）：Isaac 深色件要不要单独压暗；（14.6）旧外观数据集/checkpoint 是否按新外观重做。
- 待协作者（13.2.2、13.2.3、13.3）：mars URDF 是否正确；D405 腕部相机的安装位姿；云台相机型号和内参。
- 待验证（9.11.1）：没有新读者照 warehouse.md 从头跑过。（12.9.13）没人照文档从零装过 Unreal。（12.20.4）`max_batches: 3` 没实测。
- 未做（12.9.6，待用户安排）：用 Unreal 补渲的数据重训。

## 下一步（编号见 `TODO.md`）

- **13.2.2**：用户确认 mars URDF 后，打 saturn patch，把生成器指向 `assets/unitee_G1`，跑 `make_dex1_model.py` 自检。
- **14.4 / 14.5**：问用户 Isaac 深色件是否单独压暗；任一 warehouse 运行上实跑一次新外观的数据集补渲。
- **12.20.3 / 14.7**：问用户哪些产物删（10 步冒烟模型三份各 24 GB 最大）。

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
# 真机 URDF：先检查 patch
git apply --check outputs/g1_urdf_compare_20260930/saturn_urdf_work.patch
# 提交：按文件名 add（不要 -A，别带 assets/hero.png）→ commit（钩子跑全量检查）→ fetch，远端有新提交就 merge → 用户说了才 push
```
