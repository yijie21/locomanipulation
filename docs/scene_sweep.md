# 场景能力扫参（双臂纸箱）

> **已归档（2026-10-20）**：VLA 数据采集与训练这一系列暂不使用，这一页不再随代码更新，命令可能已经跑不通。现在的主线是[场景 RSI](rsi.md)。

结果、默认配置和取值建议在[双臂抓箱：默认配置与参数范围](box_grasp_envelope.md)；本页只讲方法和怎么跑。

目的：测出 G1 + SONIC + 现行 warehouse 双臂专家在什么样的场景布置下，**抓起并保持**的成功率不低于 50%，
之后把 warehouse 采集的场景布置限制在这个范围里（用户 2026-09-26）。
扫的是"专家 + SONIC"整体的能力，不是 SONIC 单独的：专家有几项常量按 warehouse 纸箱调过（掌面俯仰 60°、抬高的
预备姿势），抓点高度离 0.68 m 越远，失败里越可能混进常量不适配。某一档看起来像这种情况时，只改那一个旋钮补测再下结论。

## 1. 扫哪些量

| 组 | 轴（扫参文件里的名字） | 实现 |
|---|---|---|
| 箱子（每种尺寸一个场景） | `width` 两手所抓两面壁的间距、`length` 沿走入方向、`height` | `locomani.tasks.crate.box_platform_scene` |
| 每局抽样 | `start_edge` 起点骨盆到台边、`start_yaw_deg` 起点朝向偏离走入轴、`start_lateral` 起点横移、`object_setback` 箱子离台边、`object_yaw_deg` 箱子偏角 | `task.visual.start_edge_min/max` 与四个 `*_range*` 字段（[字段表](reference/config_fields.md)） |

固定条件（用户定）：台面 0.10 m 实心（抓点高度 ≈ 0.10 + 箱高 − 0.01）；箱子沿用 warehouse 纸箱的物理（1.2 kg、摩擦 1.4、1 cm 壁）；
箱子必须在视野里（`assisted` 位姿按相机几何 + 射线判可见，看不见就 `approach_vision_lost`）；周围没有障碍物（场景去掉了整个房间）。
台边和机器人起点位置与 warehouse 相同，所以距离的含义一致。

两点行为要知道：guard 走法的朝向命令固定是走入轴，所以 `start_yaw_deg` 测的是 SONIC 边走边转回正；
横向误差在走入时被清零（`visual_approach.py` `_walk_command`），`start_lateral` 测的是现行专家能容忍多大偏移，
不是 SONIC 斜着走的能力（后者要先加"起步算一次斜线方向、全程不变"，TODO）。
刹车线是台边与箱前壁中更近的一条，所以 `length` 与 `object_setback` 直接加在伸手深度上，`stand_off` 吸收不了。

## 2. 一个格子怎么判

每个格子是一份普通实验文件，由 `run_overnight.sh --to collect`（起栈 + 专家，跑完收栈）跑，`collect.record: false`、`backend.render: offline`
（仿真不出图，也不起 exporter）。成功 = 专家的 `ok`（抬起、躯干直立、保持；`retreat_distance: 0`）。

`collect.stop_rule`（`batches.stop_verdict`）：每批 5 局，从第 10 局起算成功率的 Wilson 95% 区间，整个区间高于 0.5 记 `above`，
整个低于记 `below`，到 30 局仍跨 0.5 记 `undecided`。不计入的局：RTF < 0.9（仿真跟不上墙钟会把定时阶段判成控制失败），
以及量到的是栈而不是布局的失败（`collect.NOT_A_MEASUREMENT`：起步站不稳、初始距离不合法、运行时错误等）。规则在开跑前写进扫参文件，跑完不改。

链（`chains`）上的档位从中心往外排：内一档 `below`，外面的档不再跑（单调维度更难的一侧不会更好）。
正负两侧写成两条链（右掌坐标系不是镜像的，两侧难度不同）。

## 3. 跑

```bash
. scripts/paths.sh && export PYTHONPATH="$LOCOMANI_ROOT/src"
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 仅原作者机器（与 warehouse 采集相同）；新机器按安装第 3 步打补丁后不要这行
SW=configs/sweeps/box_envelope_s1_20260926.yaml
"$VENV_SIM_PY" -m locomani.tasks.crate.sweep cells $SW      # 列出格子和状态，不动任何东西
# 每个实例一个终端，同一个循环，只换 I（实例）和 G（显卡）
I=0 G=0; while CELL=$("$VENV_SIM_PY" -m locomani.tasks.crate.sweep next $SW --instance $I --gpu $G); do
    bash scripts/run_overnight.sh "$CELL" --gpu $G --line bimanual --to collect; sleep 20; done
touch outputs/sweeps/<sweep>/STOP                           # 每个循环跑完当前格子后退出
"$VENV_SIM_PY" -m locomani.tasks.crate.sweep report $SW     # 写 outputs/sweeps/<sweep>/report.{md,json}
```

`next` 领格子前先查余量（与起栈同一条规则，不够就每分钟再试；剩下的格子都在等别的实例的内一档时也等），先接着跑本实例没跑完的格子；
一个格子起栈或驱动失败超过 3 次记 `failed`。格子的场景第一次用到时生成在 `outputs/sweeps/<sweep>/scenes/`，
实验文件在 `cells/`，运行目录是普通的 `outputs/runs/<sweep>_<格子>/`（manifest、`visual_NN.log`、trace 都在那里）。

报告每行：计入的 k/n、成功率与区间、排除的局数、失败局停在哪个阶段（`visual_NN.log` 里最后到达的阶段）、
首次停位深度与横向偏移中位数、RTF 中位数与最小值。

## 4. 顺序

1. **高度 × 长度网格**（`box_envelope_s1_20260926.yaml`，12 格）：决定手相对骨盆要去多高、越过前壁多深。
2. 以第 1 步 warehouse 纸箱几何那一格（高 0.60 × 长 0.32，13/30）为中心逐维扫（`box_envelope_s2_20260926.yaml`）：宽度 3 档，另外五个每局量各分两侧成链。中心本身只有约 0.45，阈值 0.5 下各档都会 undecided，所以这一步阈值 0.3，问的是"哪一档明显坏掉"（用户 2026-09-26）。
3. 在推荐范围里所有维度一起随机跑 40 局，看叠加后是否仍达标。
