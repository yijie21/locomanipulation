# 场景 RSI：让机器人自己学会新任务

机器人在一套公寓场景里，由 Claude 提任务、写技能代码，再由验收测试决定留不留下改动，能做成的任务一轮比一轮多。整套机制和为什么这样设计见 [场景 RSI 详解](rsi_design.md)，现在做到哪一步见[进展报告](handoff/scene-rsi-report-20261019.md)，名词见[术语表](reference/glossary.md)。10-20 起从空技能目录长出抓取的那条线（任务族 `grasp_atomic`，四次运行到封存集 38 / 50）单独记在[记录与复现](handoff/grasp-atomic-line-20261023.md)。

按规模分四级，每级都能单独跑，跑通上一级再跑下一级。耗时是 2026-10 在这台机器上（2 张卡、约 15 个 CPU 核）的实测；模型费按 Claude Opus 计。

| 级别 | 做什么 | 耗时 | 模型费 |
|---|---|---|---|
| L0 冒烟 | 一局；用假的角色走一遍内环 | 约 20 分钟 | 无 |
| 核对 | 用仓库里的技能目录重测已记录的起点，和记录逐局比 | 一个任务族约 10 分钟，全部约 2 小时 | 无 |
| L1 | 一个任务族从种子技能目录进化到停 | 约 5–10 小时（墙钟上限默认 6 小时） | 约 70–90 USD（三个 Engineer 时 32.12 是 5 小时 87.5 USD） |
| L2 | 一轮外环：发现任务、选缺口、几个任务族的内环、写回、回归 | 1–2 天 | 约 300 USD |
| L3 | 按原来的顺序从头复现到现在 | 约 4 天 | 约 600 USD |

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
PY=$VENV_SIM_PY
```

做什么：设置路径变量，把 `src/` 加进 `PYTHONPATH`；后两行固定临时目录和线程数（`OMP_NUM_THREADS` 改成 1 会改变仿真结局），`$PY` 是仿真用的 Python。下面的命令都在仓库根目录执行。沙盒、仿真槽、角色的临时目录和 ScaleBFM 权重都在 `$LOCOMANI_WORKSPACE` 下（默认 `/workspace`）；没有 `/workspace` 写权限的机器在第一条之前 `export LOCOMANI_WORKSPACE=<可写目录>`（[安装](setup.md)第 1 节），后面的命令照抄。只有一张卡的机器再 `export RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6`：内环、验收门、封存集和回归默认把 12 个仿真进程轮流放到 0、1 两张卡上，这三个变量改默认值，下面带 `CUDA_VISIBLE_DEVICES=1` 或 `MUJOCO_EGL_DEVICE_ID=1` 的命令也改成 0。
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

做什么：新克隆先做[安装](setup.md)第 4 步（每个克隆都要，`outputs/` 不在 git 里），否则第一条找不到 `outputs/scene/layout/layout.json`。第一条建一个只有机器人的空场景，后面用它的机器人模型。第三条从恢复出来的场景文件生成 MuJoCo 场景：家具、台面板、能拿的物体，机器人换成 mars 版 G1（它会重写 `third_party/` 里 mars 机器人模型的文件，内容不变）；接着加挡物体的家具面板、墙和家具上部的碰撞盒，让物体落稳，最后检查场景。每一步为什么要做见[场景文件格式](handoff/scene-file-format-20261003.md)第 3 节。
得到什么：`$SC/` 下的 `scene_locomani.xml`、`env_only.xml`、`mesh/`，以及 `panels.json`（52 块面板）、`colliders.json`（墙约 6677 个盒子）、`settle.json`、`lint.json`。最后一条打印 `[lint] 0 errors`，警告约 20 条是书和摆件静置时的小漂移，不影响运行。碰撞盒约 10 分钟，`adt_to_scene.py` 约 4 分钟，其余每条不到 2 分钟。

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
S=$LOCOMANI_WORKSPACE/code/rsi-scene/smoke
python3 scripts/rsi/sandbox.py $S pick_single
cd $S && $PY scripts/rsi/loop.py run pick_single --stub --max-iters 1; cd -
```

做什么：第一条建一个沙盒：把当前提交的代码导出到 `$S`，再在里面抽诊断池（50 个起点）和验收池（60 个）。第二条在沙盒里跑一次迭代，角色的回答是写死的，不调用 Claude：现任在诊断池上跑 50 局，跑探针，验收门只做静态检查。
得到什么：`$S/outputs/scene_rsi_20261016/pick_single/loop/iter_00/` 下有 `hypotheses.json`、`probe.json`、`gate.json`，账本 `loop/ledger.jsonl` 里这一条的 `"void": true`。约 15 分钟。看完删掉 `$S`。

## 4. 核对已记录的结果

```bash
$PY scripts/rsi/regress.py run check_$(date +%Y%m%d) --families pick_single
```

做什么：用仓库里的技能目录，在 pick_single 的封存集（80 个起点）上重跑，和恢复出来的记录逐个起点配对比较。
得到什么：一行结果，例如 `pick_single 49 -> 49 / 80 ... b 0 c 0`：箭头左边是记录，右边是这次；b 是这次成、记录败的起点数，c 反过来。同一台机器上总数应相近，但不保证逐局相同（实测见过 b 3 c 3、总数相同；原因没查，用户 10-22：研究初期不查）；换了 GPU 或驱动，少数局会不同，b 和 c 各在 5 以内算正常。没有任务族在 p ≤ 0.05 上变差时，登记表里每族的 `last` 前移到这次的标签，以后的核对和它比。注意：登记表和复现材料里的记录（标签 `sleep_20261021`）是走路 0.5 m/s + 休眠下测的，现在默认 0.25 m/s，所以会差得多些；`regress.py` 发现动作层设置不同时只作参照比较、不挡、登记表照样前移（TODO 36；用户 10-22：研究初期不按 0.25 m/s 重测）。约 10 分钟。去掉 `--families` 就核对全部 8 个任务族和 3 个标定样本（1488 局，约 2 小时）；记录里任务族合计 169 / 632（0.25 m/s 的上一批 `surfaces_20261020` 是 187 / 632）。

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

做什么：开一个分支，把技能目录换成第一次写回之前的那一版（动作层加两个移植来的抓取程序），提交。沙盒只导出已提交的代码，所以必须提交；干净环境里提交之前先 `git config user.name` 和 `user.email`。
得到什么：`src/locomani/tasks/scene/skills/` 只剩 3 个文件（`__init__.py`、`grasp_program.py`、`pick.py`），打印 `the seed catalog`。

### 5.2 建沙盒、跑内环

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/pick-repro
python3 scripts/rsi/sandbox.py $S pick_single
cd $S && nohup setsid $PY scripts/rsi/loop.py run pick_single > loop.log 2>&1 &
```

做什么：建沙盒并抽池，然后在后台跑内环，直到满足停止条件（[详解](rsi_design.md)第 5 节）。每次迭代：Reviewer 读诊断池的记录提假设，Prober 试参数并给假设标状态，Manager 选至多 3 条（提过两轮以上、没做过、没被否定的机制会被驱动程序强制占一个名额），3 个 Engineer 并行写代码，Composer 合成一个候选，验收门判（`--candidates 1` 只用一个 Engineer、没有 Composer）。沙盒里带经验库，不含这个任务族自己的那条。每局记录带失败类型和关键事件，验收结论按类型给多赢少赢（TODO 33.15–33.17）。现任在验收池上已经超过 70% 时接着跑，要加 `--stop-rate`（比如 `--stop-rate 0.85`），否则一开跑就停；续跑沿用上次记下的值。
角色在沙盒里能用的命令（规则第 18、32 条；都是人写的，循环只调用）：
- `scripts/rsi/try.py`：点名几个起点、用一份技能目录跑，走 12 进程队列并行，返回逐局结果和对现任的翻转；每次至多 40 局、每个角色每轮至多 240 局（TODO 37.1）。
- `PYTHONPATH=src $PY scripts/rsi/body.py reach --x 0.45 --y-out 0.05 --z 0.9 --att level --hand right`：身体模型——这个伸手目标（骨盆朝向系：前伸、朝手外侧、离地高度，加手腕姿态）控制器送不送得到、差多少。表在 `outputs/scene_rsi_20261016/body/`，建沙盒时复制进去（TODO 33.1.4）。
- `PYTHONPATH=src $PY scripts/rsi/choose.py outputs/scene_discovery/adt_apartment_mars Mug_2 --start <起点文件>`：站位和抓法一起选——每种抓法、左右手、站位、姿态的预计送达概率，最好的在前，并说明兜底顺序（站着 → 降骨盆 → 准备动作，TODO 33.12）。
- `PYTHONPATH=src $PY scripts/rsi/prep.py outputs/scene_discovery/adt_apartment_mars DishSoap_1 --start <起点文件> --skills <技能目录>`：挑选器没有站位时，转、拉、挪邻居哪种目标能换来站位，技能包里有没有实现（TODO 33.5；任务族 `grasp_prep` 用起点的 `placement` 选项专抽这类局面）。
驱动程序的开关（都有默认值）：三份草稿里两份交了再过 10 分钟没交的停掉（`--engineer-late-min`，0 不停，TODO 37.3）；合成候选被拒时同一轮再送最好的那份草稿（`--second-shot`，默认开，算一次验收门调用，TODO 37.4）；现任在诊断池上失败少于 10 局时，下一轮开头重抽诊断池——按诊断种子段抽新起点、现任跑一遍、失败的换进来到 15 局（`--diag-min-fails` / `--diag-target-fails` / `--diag-draw-max`，0 关；`loop.py redraw <族>` 在续跑前单独做一次，TODO 33.23）；一张卡的机器见第 0 节的 `RSI_GPUS` / `RSI_PROCS`。
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
$PY scripts/rsi/outer.py select $R --loops $LOCOMANI_WORKSPACE/code/rsi-scene/pick-repro
$PY scripts/rsi/outer.py family $R on_single
```

做什么：第一条按“解锁的任务数 ÷ 估计花费”给缺口排序，最多取 6 个；第二条为一个缺口写任务族文件，每个选中的缺口各跑一次。
得到什么：`$R/selection.json`（打印选中的缺口），`scripts/rsi/families/r1_on_single.json` 等。提交这些文件，沙盒才看得到。

```bash
$PY scripts/rsi/outer_run.py $R r1_on_single r1_held_single r1_in_single --parallel 2 > $R/driver.log 2>&1 &
```

做什么：对每个任务族依次建沙盒、检查任务、跑内环、跑封存集、给经验库写一条、写回、跑单元测试、回归、更新能力表；同时最多跑两个任务族。
得到什么：每个任务族的状态在 `$R/run/<任务族>.json`，各步日志在 `$R/run/<任务族>/`；写回改动了技能目录、经验库多了这一族的条目（都没提交），能力表 `configs/scene_discovery/capability_mars_scalebfm.yaml` 更新。一个任务族 2–8 小时。内环异常停下（角色调用失败、审计连续作废两轮、验收门出错）、写回后的测试或回归没过，这个任务族就停在那一步，原因写在状态文件的 `blocked`；写回之后被挡的，技能目录已恢复到写回之前（`restored`）。查完把 `step` 改回要重跑的那一步再起。

## 7. L3：从头复现

按原来的顺序做下面这些。任务族文件都在 `scripts/rsi/families/` 里，发现阶段可以用恢复出来的任务清单，这样任务和原来一样。

| 顺序 | 做什么 | 原来的结果（封存集） |
|---|---|---|
| 1 | L1：pick_single | 0 → 41 / 80 |
| 2 | 第 1 轮外环，任务清单 `rsi_round1`；按顺序 r1_on_single、r1_held_single、r1_in_single、r1_held_near_single、r1_slide_then_grasp | 22 / 80、31 / 80、9 / 80、33 / 80、38 / 80 |
| 3 | 第 2 轮外环，任务清单 `rsi_round2`；只跑 r2_on_single、r2_in_single（原来做到这里暂停） | 25 / 72、15 / 80 |
| 4 | 标定样本：`calibrate.py` 的 cal_r2、cal_r3、cal_r4（任务清单已恢复，跳过 `sample`），每个 `family.py draw`、`batch.py`、`calibrate.py report` | 判进第 1 桶的任务真做成的：11 / 24、13 / 24、12 / 23 |
| 5 | 全部回归、`successes.py` | 做成的任务 31 个，偶尔成 13 个（2026-10-20 的场景） |
| 6 | 抓取从原子技能开始：`grasp_atomic` 从空种子跑一次（[记录与复现](handoff/grasp-atomic-line-20261023.md)第 5 节） | 0 → 31 / 50（第 1 轮） |
| 7 | 接着跑：`replay.py setup` 从第 6 轮起（放宽的验收门和候补池），再从第 10 轮起（失败类型和 `stage()`，`--stop-rate 0.9`） | 36 / 50，再 38 / 50 |

和原来的差别要心里有数：
- 原来边跑边改了场景（6 次）和规则（`rules.md` 第 1 到 6 版；grasp_atomic 那条线上到第 19 版），复现时一开始就是最终的场景和规则，所以各族的数字会不同，量级应相近。
- 原来人插手的地方（修驱动、改规则、写回时手工合并 6 处）列在[详解](rsi_design.md)第 8 节；复现时写回如果冲突，`writeback.py` 什么都不写并打印冲突的文件，手工合并后再提交。
- 能力表在仓库里，是第 1 阶段探针加各族封存集写回的结果；重测探针要 `scripts/rsi/capability_probe.py`，它用的空地场景先 `bash scripts/build_scene.sh walk_mars` 建。

## 8. 看过程：图和视频

两个脚本都把一局的 50 Hz 轨迹回放到完整的 ADT 场景上渲染（循环跑的是只有物理的场景副本，状态相同），半透明紫球是指令给的手腕位置；视频都带 ≤ 500 KB 的预览版（`previews/`），给人看、放网页、打包只用预览版（[AGENTS](../AGENTS.md#shared-resources-and-artifacts)）。一张卡的机器把 `MUJOCO_EGL_DEVICE_ID` 改成 0。

**任何沙盒：`evolution_media.py`**

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 $PY scripts/rsi/report/evolution_media.py --sandbox $S --family pick_single --out outputs/rsi_repro/visual
```
做什么：读沙盒的账本，取种子和每一次被验收门接受的版本，再读它们在诊断池上的逐局记录和轨迹，出三样东西（`--only step,overview,task` 选其中几样，默认全出）。`step_NN.mp4`：第 NN 次被接受的改动，在它改前改后都跑过的起点里选一个被它从失败翻成成功、轨迹最短的，左改前右改后，3 倍速，字幕卡默认是 Manager 那一轮的中文总结（`--cards <JSON>` 按版本哈希换成自己写的，`"_names"` 给物体改名；种子没有技能时左边放起点第一帧）。`task_NN_<任务>.mp4`：任务族的每个任务一段，取最终版本做成的最短一局，没有成功就取最短的失败局（规划拒绝也是一局，机器人站着不动），左种子右最终版本，字幕卡列出每个版本在该任务诊断起点上的成功数；只要最终技能、不要种子对照时加 `--versions final`，同样的起点和字幕卡，单画面，文件名 `final_NN_<任务>.mp4`，不覆盖双画面的那套。`overview.png`：每个任务一行、每个版本一列，每格是那一局决定成败的那一刻（成功取抬起后一秒，失败取结尾）。
得到什么：`outputs/rsi_repro/visual/` 下的视频、同名海报 `.jpg`、`previews/`，索引 `versions.json`（版本、哈希、来自第几轮、验收门数字、Manager 总结）和 `tasks.json`（每段任务视频的起点和各版本成功数；`--versions final` 的写到 `tasks_final.json`）。一段视频约 1 分钟，单画面约 40 秒；被接受两次、8 个任务的运行约 10 分钟。

**只用于 pick-a 那次运行：`visualize.py`**

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 $PY scripts/rsi/report/visualize.py --sandbox /workspace/code/rsi-scene/pick-a --out outputs/scene_rsi_20261016/visual
```

做什么：2026-10-16 为 pick_single 的第一次运行（原机器的 `pick-a` 沙盒）手写的纪录片，讲"Reviewer 每一轮看到了什么"：那次的 5 个版本哈希、起点编号、每轮的线索和字幕都写在代码里（`VERSIONS`、`clue_0`–`clue_3`、`main` 的 `cards`）。`--only clue,step,infeasible,overview` 选：`clue_NN.png` 第 NN 轮的线索图，画在那一局 Reviewer 引用的那一刻上（罐子被手掌推开、导航图上没有地上的凳子、遥控器侧着抓进不去、右手站位全被吧台凳挡住）；`step_NN.mp4` 同一起点改前改后并排，2 倍速，前面一张线索卡；`infeasible_book.png`、`infeasible_wallet.png` Reviewer 论证现有夹爪抓不了的两个任务；`overview.png` 5 个指定起点 × 5 个版本。
得到什么：只对 `pick-a` 沙盒有效，换一个沙盒它在开头查到缺那几个版本目录就退出并提示改用上面那条；pick-a 的诊断局不在复现材料里，别的机器上跑不了。它同时是其他报告脚本共用的底层（`Scene` 回放和渲染、`Round` 读一局、`why_zh` 失败判词的中文对照）；要给自己的运行出线索图，照 `clue_N` 另写。Unreal 版视频用 `scripts/rsi/report/report_unreal.py`，它读原来那几个沙盒和写死的清单，只能在原来的机器上跑；给别的运行渲染要先按 `configs/scenes.yaml` 的 `adt_apartment_mars.unreal.room_build` 建公寓关卡，再改它的清单。

## 9. 常见错误

- `scene_lint.py` 报 error：不要开跑循环。先看 `lint.json` 里是哪一类（物体掉下去、目标开局就成立、竖直轴不对），多半是场景没按第 2 节的顺序建，或者少了面板、碰撞盒。
- 内环一开始就停，`loop.log` 里说缺 `pool_diag`：沙盒没建完，重新跑 `sandbox.py`。
- 账本里某次迭代或草稿标 `void`：角色读了验收池或沙盒外的文件，或者 Engineer 改了技能目录以外的文件。三个 Engineer 中一份草稿越界，只作废这一份，这一轮接着走；Reviewer、Manager、Composer 或唯一的 Engineer 越界，这一轮作废（账本记 `void_iter`），现任不变，进下一轮；连续两轮作废，内环停下，`loop.py` 退出码 3，查完用 `--reprobe N` 续跑（TODO 36.7）。验收门调用标 `void` 是有起点缺了局，这次不算数；草稿或候选标 `named_objects` 是技能代码里写了物体名字（规则第 30 条），作废。
- 验收门从规则第 14 版起按“多赢 − 少赢 ≥ 3 且少赢 ≤ 多赢 / 3”判，不再要求 p ≤ 0.05（TODO 33.8）；被拒但多赢多于少赢的最好 2 个候选在 `loop/candidates_pool.json`，账本 `pool` 行记进出。Manager 决定里的 `base` 是从候补接着改，Composer 的 `adopted_pool` 是把候补原样再送；旧规则拒过的候选可以再判。
- 回归打印 `worse at p <= 0.05`：有任务族变差，登记表不推进。先查最近改了什么，再决定是否接受。
- 仿真很慢、结局和记录差很多：确认 `OMP_NUM_THREADS=2`，并且机器上没有别的仿真在跑（所有批次共用 12 个仿真槽）。
