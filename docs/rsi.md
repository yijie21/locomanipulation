# 场景 RSI：让机器人自己学会新任务

机器人在一套公寓场景里，由 Claude 提任务、写技能代码，再由验收测试决定留不留下改动，能做成的任务一轮比一轮多。整套机制和为什么这样设计见 [场景 RSI 详解](rsi_design.md)，现在做到哪一步见[进展报告](handoff/scene-rsi-report-20261019.md)，名词见[术语表](reference/glossary.md)。

按规模分四级，每级都能单独跑，跑通上一级再跑下一级。耗时是 2026-10 在这台机器上（2 张卡、约 15 个 CPU 核）的实测；模型费按 Claude Opus 计。

| 级别 | 做什么 | 耗时 | 模型费 |
|---|---|---|---|
| L0 冒烟 | 一局；用假的角色走一遍内环 | 约 20 分钟 | 无 |
| 核对 | 用仓库里的技能目录重测已记录的起点，和记录逐局比 | 一个任务族约 10 分钟，全部约 2 小时 | 无 |
| L1 | 一个任务族从种子技能目录进化到停 | 6–10 小时 | 约 70 USD |
| L2 | 一轮外环：发现任务、选缺口、几个任务族的内环、写回、回归 | 1–2 天 | 约 300 USD |
| L3 | 按原来的顺序从头复现到现在 | 约 4 天 | 约 600 USD |

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
PY=$VENV_SIM_PY
```

做什么：设置路径变量，把 `src/` 加进 `PYTHONPATH`；后两行固定临时目录和线程数（`OMP_NUM_THREADS` 改成 1 会改变仿真结局），`$PY` 是仿真用的 Python。下面的命令都在仓库根目录执行。
前提：[安装](setup.md)里“场景 RSI”一列要求的步骤都已完成，包括 ADT 数据、ScaleBFM 权重和 Claude CLI。

## 1. 恢复复现材料（每台机器一次）

```bash
python3 scripts/rsi/kit.py restore
```

做什么：把仓库里 `rsi_kit/` 的材料放到 `outputs/` 下对应的位置：场景文件、三次发现的任务清单、三个标定样本的任务清单、可站边覆盖文件、各任务族的起点池和最近一次的逐局结局，并写好回归用的登记表。这些都是大模型生成或在当时的场景上抽的，重新生成会不一样，所以随仓库提供。
得到什么：打印每类材料恢复了多少个文件。已有的文件内容不同时不覆盖，打印 `kept`；登记表已存在时不写。几秒钟。

## 2. 建场景（每台机器一次）

```bash
bash scripts/build_scene.sh walk
SC=outputs/scene_discovery/adt_apartment_mars
OV=outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json
/venv/main/bin/python scripts/adt_to_scene.py mujoco --scene $SC/scene_meta.json --robot mars
/venv/main/bin/python scripts/scene_add_panels.py --scene $SC --approach-edges $OV
/venv/main/bin/python scripts/scene_add_colliders.py --scene $SC --approach-edges $OV
$PY scripts/scene_settle.py --scene $SC --write
$PY scripts/scene_lint.py --scene $SC
```

做什么：第一条建一个只有机器人的空场景，后面用它的机器人模型。第三条从恢复出来的场景文件生成 MuJoCo 场景：家具、台面板、能拿的物体，机器人换成 mars 版 G1；接着加挡物体的家具面板、墙和家具上部的碰撞盒，让物体落稳，最后检查场景。每一步为什么要做见[场景文件格式](handoff/scene-file-format-20261003.md)第 3 节。
得到什么：`$SC/` 下的 `scene_locomani.xml`、`env_only.xml`、`mesh/`，以及 `panels.json`（52 块面板）、`colliders.json`（墙约 6677 个盒子）、`settle.json`、`lint.json`。最后一条打印 `[lint] 0 errors`，警告约 20 条是书和摆件静置时的小漂移，不影响运行。碰撞盒约 10 分钟，其余每条不到 2 分钟。

## 3. L0 冒烟

### 3.1 跑一局

```bash
P=outputs/scene_rsi_20261016/kit/pick_single/pool_sealed
CUDA_VISIBLE_DEVICES=0 $PY scripts/rsi/scene_round.py pick_single --pool $P \
  --skills src/locomani/tasks/scene/skills --out outputs/rsi_smoke/one --labels tv_remote_pickup_s300005 --npz
```

做什么：用仓库里现在的技能目录，从拿起遥控器的一个起点跑一局：机器人走到茶几旁，拿起遥控器。
得到什么：`outputs/rsi_smoke/one/tv_remote_pickup_s300005.json`，`"ok": true`；同名 `.npz` 是 50 Hz 轨迹。约 2 分钟，大半是加载。

### 3.2 用假的角色走一遍内环

```bash
S=/workspace/code/rsi-scene/smoke
python3 scripts/rsi/sandbox.py $S pick_single
cd $S && $PY scripts/rsi/loop.py run pick_single --stub --max-iters 1; cd -
```

做什么：第一条建一个沙盒：把当前提交的代码导出到 `$S`，再在里面抽诊断池（50 个起点）和验收池（60 个）。第二条在沙盒里跑一次迭代，角色的回答是写死的，不调用 Claude：现任在诊断池上跑 50 局，跑探针，验收门只做静态检查。
得到什么：`$S/outputs/scene_rsi_20261016/pick_single/loop/iter_00/` 下有 `hypotheses.json`、`probe.json`、`gate.json`，账本 `loop/ledger.jsonl` 里这一条标 `void` 为 `stub`。约 15 分钟。看完删掉 `$S`。

## 4. 核对已记录的结果

```bash
$PY scripts/rsi/regress.py run check_$(date +%Y%m%d) --families pick_single
```

做什么：用仓库里的技能目录，在 pick_single 的封存集（80 个起点）上重跑，和恢复出来的记录逐个起点配对比较。
得到什么：一行结果，例如 `pick_single 49 -> 49 / 80 ... b 0 c 0`：箭头左边是记录，右边是这次；b 是这次成、记录败的起点数，c 反过来。同一台机器上应逐局相同；换了 GPU 或驱动，少数局会不同，b 和 c 各在 5 以内算正常。约 10 分钟。去掉 `--families` 就核对全部 8 个任务族和 3 个标定样本（1488 局，约 2 小时），记录是 2026-10-20 的：任务族合计 187 / 632。这些记录是在走路速度 0.25 m/s 下测的（现在的默认）；2026-10-21 登记表和复现材料暂时是 0.5 m/s 的记录，按 0.25 m/s 重测并更新之前（TODO 35.2），核对会和记录有差别。

```bash
$PY scripts/rsi/successes.py
```

做什么：按任务把测过的局合起来，数做成（至少一半成功）、偶尔成、没成的任务。
得到什么：`outputs/scene_rsi_20261016/outer/successes.json`，打印三类的个数和清单，应和[报告](handoff/scene-rsi-report-20261019.md)第 7 节一致。几秒钟。

## 5. L1：一个任务族的内环

以 pick_single（拿起 8 种物体）为例，从种子技能目录开始，这也是原来的第一个任务族。

### 5.1 换成种子技能目录

```bash
git switch -c rsi-repro
python3 scripts/rsi/kit.py seed
git commit -qam "scene RSI: seed skills catalog"
```

做什么：开一个分支，把技能目录换成第一次写回之前的那一版（动作层加两个移植来的抓取程序），提交。沙盒只导出已提交的代码，所以必须提交。
得到什么：`src/locomani/tasks/scene/skills/` 只剩 3 个文件（`__init__.py`、`grasp_program.py`、`pick.py`），打印 `the seed catalog`。

### 5.2 建沙盒、跑内环

```bash
S=/workspace/code/rsi-scene/pick-repro
python3 scripts/rsi/sandbox.py $S pick_single
cd $S && nohup setsid $PY scripts/rsi/loop.py run pick_single > loop.log 2>&1 &
```

做什么：建沙盒并抽池，然后在后台跑内环，直到满足停止条件（[详解](rsi_design.md)第 5 节）。每次迭代：Reviewer 读诊断池的记录提假设，Prober 试参数，Manager 选一条，Engineer 写代码，验收门判。
得到什么：账本在 `$S/outputs/scene_rsi_20261016/pick_single/loop/`。原来这一族跑了 6 次迭代、6 次验收（接受 4 次），约 10 小时，现任在验收池上从 0 到 26 / 60。

```bash
cd $S && $PY scripts/rsi/loop.py status pick_single; cd -
```

做什么：打印进度：第几次迭代、每次验收的结论、现任在验收池上的成绩、已花的时间和模型费。随时可以看。

### 5.3 封存集、写回、登记

```bash
cd $S && $PY scripts/rsi/sealed.py pick_single; cd -
SUM=$S/outputs/scene_rsi_20261016_hidden/pick_single/sealed/summary.json
python3 scripts/rsi/writeback.py $S pick_single
$PY scripts/rsi/regress.py add pick_single $S pick --last $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['final'])" $SUM)
```

做什么：第一条在封存集上跑种子和最终的现任；第二条把最终现任合并进主仓库的技能目录；第三条把这个任务族登记进回归，记录取最终现任在封存集上的那一批，以后每次写回或改场景都和它比。
得到什么：`$SUM` 里 `versions` 列出种子和最终版本各自的成功数，`final` 是最终版本的名字。原来种子 0 / 80，最终 41 / 80（当时的场景）。写回改动了 `src/locomani/tasks/scene/skills/`，跑 `make test` 通过后提交。

## 6. L2：一轮外环

```bash
R=outputs/scene_rsi_20261016/outer/repro_round1
$PY scripts/scene_discover_tasks.py --scene outputs/scene_discovery/adt_apartment_mars/scene_meta.json \
  --capability configs/scene_discovery/capability_mars_scalebfm.yaml --approach-edges $OV \
  --model claude-opus-5-5 --iterations 2 --per-room 30 --out $R/discovery
```

做什么：Claude 读场景文件、能力表和技能目录，按房间提任务；程序把每个任务分进三个桶。想用原来那一轮的任务，跳过这条，把恢复出来的 `outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round1` 链接到 `$R/discovery`。
得到什么：`$R/discovery/tasks.jsonl`，约 60 个任务。约 20 分钟、5 USD。

```bash
$PY scripts/rsi/outer.py select $R --loops /workspace/code/rsi-scene/pick-repro
$PY scripts/rsi/outer.py family $R on_single
```

做什么：第一条按“解锁的任务数 ÷ 估计花费”给缺口排序，最多取 6 个；第二条为一个缺口写任务族文件，每个选中的缺口各跑一次。
得到什么：`$R/selection.json`（打印选中的缺口），`scripts/rsi/families/r1_on_single.json` 等。提交这些文件，沙盒才看得到。

```bash
$PY scripts/rsi/outer_run.py $R r1_on_single r1_held_single r1_in_single --parallel 2 > $R/driver.log 2>&1 &
```

做什么：对每个任务族依次建沙盒、检查任务、跑内环、跑封存集、写回、跑单元测试、回归、更新能力表；同时最多跑两个任务族。
得到什么：每个任务族的状态在 `$R/run/<任务族>.json`，各步日志在 `$R/run/<任务族>/`；写回改动了技能目录（没提交），能力表 `configs/scene_discovery/capability_mars_scalebfm.yaml` 更新。一个任务族 2–8 小时。

## 7. L3：从头复现

按原来的顺序做下面这些。任务族文件都在 `scripts/rsi/families/` 里，发现阶段可以用恢复出来的任务清单，这样任务和原来一样。

| 顺序 | 做什么 | 原来的结果（封存集） |
|---|---|---|
| 1 | L1：pick_single | 0 → 41 / 80 |
| 2 | 第 1 轮外环，任务清单 `rsi_round1`；按顺序 r1_on_single、r1_held_single、r1_in_single、r1_held_near_single、r1_slide_then_grasp | 22 / 80、31 / 80、9 / 80、33 / 80、38 / 80 |
| 3 | 第 2 轮外环，任务清单 `rsi_round2`；只跑 r2_on_single、r2_in_single（原来做到这里暂停） | 25 / 72、15 / 80 |
| 4 | 标定样本：`calibrate.py` 的 cal_r2、cal_r3、cal_r4（任务清单已恢复，跳过 `sample`），每个 `family.py draw`、`batch.py`、`calibrate.py report` | 判进第 1 桶的任务真做成的：11 / 24、13 / 24、12 / 23 |
| 5 | 全部回归、`successes.py` | 做成的任务 30 个，偶尔成 15 个（2026-10-20 的场景） |

和原来的差别要心里有数：
- 原来边跑边改了场景（6 次）和规则（`rules.md` 从第 1 版到第 6 版），复现时一开始就是最终的场景和规则，所以各族的数字会不同，量级应相近。
- 原来人插手的地方（修驱动、改规则、写回时手工合并 6 处）列在[详解](rsi_design.md)第 8 节；复现时写回如果冲突，`writeback.py` 什么都不写并打印冲突的文件，手工合并后再提交。
- 能力表在仓库里，是第 1 阶段探针加各族封存集写回的结果；重测探针要 `scripts/rsi/capability_probe.py`，它用的空地场景 `outputs/scene/walk_mars` 还没有登记构建命令。

## 8. 看过程：图和视频

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 $PY scripts/rsi/visualize.py --sandbox $S --out outputs/rsi_repro/visual
```

做什么：从一个沙盒的账本和轨迹画出每次被接受的改动：线索图、改动前后同一起点的对比视频。
得到什么：`outputs/rsi_repro/visual/` 下的图和视频，网页用的预览版每段不超过 500 KB。约 10 分钟。报告里的 Unreal 版视频用 `report_unreal.py`，它读的是原来那几个沙盒，只能在原来的机器上跑。

## 9. 常见错误

- `scene_lint.py` 报 error：不要开跑循环。先看 `lint.json` 里是哪一类（物体掉下去、目标开局就成立、竖直轴不对），多半是场景没按第 2 节的顺序建，或者少了面板、碰撞盒。
- 内环一开始就停，`loop.log` 里说缺 `pool_diag`：沙盒没建完，重新跑 `sandbox.py`。
- 账本里某次迭代标 `void`：角色读了验收池或沙盒外的文件，或者 Engineer 改了技能目录以外的文件。这一次不算，驱动会接着跑下一次。
- 回归打印 `worse at p <= 0.05`：有任务族变差，登记表不推进。先查最近改了什么，再决定是否接受。
- 仿真很慢、结局和记录差很多：确认 `OMP_NUM_THREADS=2`，并且机器上没有别的仿真在跑（所有批次共用 12 个仿真槽）。
