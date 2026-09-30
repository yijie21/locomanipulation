# STATE — 2026-09-30 下午（会话：真机 URDF 试接入后回退；三个 runbook 改写，Unreal 在 warehouse 每一步与 Isaac 平行；已推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认：`34fcd13` 已推送，与 origin/main 同步（上午的 handoff `c8c9675` 一起推出）；之后只有这次 handoff 提交。
已确认（09-30 下午）：本会话没有留下在跑的仿真栈、Isaac 或 Unreal 编辑器，GPU0 空闲。GPU1 约 23 GB 是别的项目（`SimpleRoboJev` 的批量评测、`depth_server`），会涨会落；上 GPU1 前先看 `nvidia-smi`。
已确认：那个项目的批量评测跑起来时容器进程数约 1900，采集的准入检查（exporter 要 1800）会拒绝起栈；不录制的试跑（`collect.record: false`）能起。
别的会话：henry-zhao 直接推 origin/main，推送不跑钩子，他的提交可能自带检查失败。

## 磁盘

- 已确认：`/workspace` 空闲 157 GB，根分区 `/` 空闲约 5 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5 的删除都没定；要删的话先在这里写明 24 小时。

## 当前进度

- 已确认，已推送（`34fcd13`，TODO 9.11、12.9.14.7；DECISIONS 2026-09-30 runbook 一节）：
  - 三个 runbook 只留命令、一句"做什么"、一句"得到什么"、每步总耗时；移出的内容在 `warehouse_design.md` 第 10 节；行数上限已降（215/125/115）。
  - warehouse：第 0 步 `RUN=warehouse_smoke`（Unreal 是 `warehouse_smoke_unreal`），共用命令写 `$RUN`；和渲染器有关的命令同一步里并排两条。
  - `render_third_person.sh --side-by-side`：写 `<局>_vs_mujoco.mp4`，左 MuJoCo 回放、右所选渲染器，逐帧对齐；缺 MuJoCo 半边就先渲。
  - 验证：两份冒烟上实跑（Isaac 11 局约 50 s，Unreal 8 局约 1 min）；warehouse 每条命令代入两个 `$RUN` 跑 `--dry-run`，30/36 过，6 条是设计好的拒绝。
- 已确认（TODO 13；DECISIONS 2026-09-30 回退一节）：真机 URDF（`assets/unitree_G1`）用户说给错了，saturn 接入全部回退，没提交。
  - 改动在 `outputs/g1_urdf_compare_20260930/saturn_urdf_work.patch`，在 `34fcd13` 上 `git apply --check` 通过。
  - third_party 里生成的 saturn 文件已删；stock 模型重新生成后与会话前逐字节相同。
  - 用户确认的真机事实：踝力矩固件上限 35 N·m（仿真 50，13.4）；VLA 第一视角是云台相机（型号 D435 还是 D455 未定，13.3）。

## 待验证 / 卡点

- 待协作者（13.2、13.3）：正确的真机 URDF，云台相机型号和内参。
- 待验证（9.11.1）：没有新读者照新 warehouse.md 从头跑过；只做了 `--dry-run` 和改动部分的实跑。bottle、crate 的命令没变，没重跑冒烟。
- 待验证（12.9.13）：没人照文档从零装过 Unreal。
- 待验证（12.20.4）：还没按 `max_batches: 3` 跑过冒烟，第三批能不能补齐没实测。
- 未做（12.9.14.3 剩下的部分）：`mujoco` 运行的仿真实时录像，四边线仍用场景的 `video_camera`，不是每局的边机位。
- 未做（12.9.6，待用户安排）：用 Unreal 补渲的数据重训；之前 Unreal 的评测数字不能和 Isaac 比。

## 下一步（编号见 `TODO.md`）

- **12.20.3 / 12.20.5 / 13.5**：问用户哪些产物删（10 步冒烟模型三份各 24 GB 最大）；要删先在这里写明 24 小时。
- **13.2–13.4**：等协作者的正确 URDF 和相机型号；用户定仿真踝力矩要不要对齐 35 N·m。
- **12.9.6**：用户安排后，用 `warehouse_200_unreal.yaml` 做 Unreal 的整条训练和评测。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
df -h / /workspace; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; cat /sys/fs/cgroup/pids.current; export TMPDIR=/workspace/tmp
# warehouse runbook 的写法：先选运行名，再照 docs/warehouse.md 跑
RUN=warehouse_smoke_20260930   # 已有的 Isaac 冒烟；Unreal 的是 warehouse_smoke_unreal_20260930
bash scripts/render_third_person.sh outputs/runs/$RUN --side-by-side --dry-run
# 真机 URDF 回来时：先检查 patch 还能不能打上
git apply --check outputs/g1_urdf_compare_20260930/saturn_urdf_work.patch
# 提交：make check → 按文件名 add（不要 -A）→ commit（钩子跑全量检查）→ fetch，远端有新提交就 merge（不 rebase）→ 用户说了才 push
```
