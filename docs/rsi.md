# 场景 RSI：让机器人自己学会新任务

机器人在一套公寓场景里，由 Claude 提任务、写技能代码，再由验收测试决定留不留下改动，能做成的任务一轮比一轮多。整套机制和为什么这样设计见 [场景 RSI 详解](rsi_design.md)，现在做到哪一步见[进展报告](handoff/scene-rsi-report-20261019.md)，名词见[术语表](reference/glossary.md)。10-20 起从空技能目录长出抓取的那条线（任务族 `grasp_atomic`，四次运行到封存集 38 / 50）单独记在[记录与复现](handoff/grasp-atomic-line-20261023.md)。同一套流程在物流间的应用（双手抱起纸箱，任务族 `box_bimanual`，TODO 38）见[物流间 runbook](rsi_logistics_room.md)。

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

一张卡、没有 `/workspace` 的机器（SRA0889 这类），第 0 节合起来是：

```bash
cd ~/workspace/locomanipulation
export LOCOMANI_WORKSPACE=$HOME/workspace
. scripts/paths.sh
export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
export RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6 MUJOCO_EGL_DEVICE_ID=0
PY=$VENV_SIM_PY
```

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
得到什么：`outputs/rsi_smoke/one/tv_remote_pickup_s300005.json`，`"ok": true`；同名 `.npz` 是 50 Hz 轨迹（含接触通道 `contact_k` / `contact_a` / `contact_b` 和刚体名表 `bodies`）；同名 `.past.jsonl` 是场景图记录，每个技能步骤一行（TODO 39），`PYTHONPATH=src $PY scripts/rsi/past.py show outputs/rsi_smoke/one/tv_remote_pickup_s300005` 按步骤打印实体、前后关系、翻转事件和期待与差异——里面若只有一行 `"error"`，是记录器出错，局本身不受影响，把那一行报给维护者。约 2 分钟，大半是加载。

### 3.2 用假的角色走一遍内环

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/smoke
python3 scripts/rsi/sandbox.py $S pick_single
cd $S && $PY scripts/rsi/loop.py run pick_single --stub --max-iters 1; cd -
```

做什么：第一条建一个沙盒：把当前提交的代码导出到 `$S`，再在里面抽诊断池（50 个起点）和验收池（60 个）。第二条在沙盒里跑一次迭代，角色的回答是写死的，不调用 Claude：现任在诊断池上跑 50 局，跑探针，验收门只做静态检查。
得到什么：`$S/outputs/scene_rsi_20261016/pick_single/loop/iter_00/` 下有 `diag_digest.md`（Reviewer 读的摘要，含 "Scene-graph transitions" 一节）、`hypotheses.json`、`probe.json`、`gate.json`，现任的 50 局在 `…/pick_single/diag/incumbent_<hash>/`（每局 `.json`、`.npz`、`.past.jsonl`），账本 `loop/ledger.jsonl` 里这一条的 `"void": true`。约 15 分钟。看完删掉 `$S`。沙盒是 `git archive HEAD`，没提交的改动进不去；本机只有一张卡时按第 0 节设 `RSI_GPUS` / `RSI_PROCS` / `RSI_SIM_SLOTS`。

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

### 5.1 选起点：空种子，还是种子技能目录

两种起点，由任务族文件决定，不用改代码：

**(a) 空种子——从动作层长出技能**（任务族文件带 `sandbox` 块：`grasp_atomic`、`grasp_atomic_past`、`grasp_prep`、`box_bimanual`）。技能目录只有一个 `__init__.py`，`plan_for()` 对所有目标返回 None，种子一局也做不成；抓取从零由循环写出。`sandbox.py` 建沙盒时把 `sandbox.seed`（`scripts/rsi/seeds/atomic`）复制覆盖到沙盒的 `skills/`，并只保留 `sandbox.keep` 列的文件和 `closure` 追出的 import 闭包，经验库去掉所有代码补丁（规则 27）——角色找不到旧技能和旧记录。主仓库的技能目录不动，不用切分支。最终现任**写不回**主仓库（新文件和同名文件冲突，[详解](rsi_design.md)第 10 节），结果以封存集为准。10-20 以后的运行（grasp_atomic 四次、grasp_prep、box_bimanual）都是这种。

**(b) 种子技能目录——接着已有技能改**（`pick_single` 和两轮外环的所有族）。技能目录是第一次写回之前的那一版：动作层加两个移植来的抓取程序，种子已经偶尔能抓。要先换掉主仓库的技能目录并提交到一个分支（沙盒 `git archive HEAD`，只导出已提交的代码；干净环境里提交前先 `git config user.name` 和 `user.email`）：

```bash
git switch -c rsi-repro
python3 scripts/rsi/kit.py seed          # skills/ 只剩 __init__.py、grasp_program.py、pick.py，打印 the seed catalog
git commit -qam "scene RSI: seed skills catalog"
```

最终现任能三方合并写回主仓库（5.3）。

下面以 (a) 的 `grasp_atomic_past` 为例：ADT 公寓里 10 个 `pick_*` 任务（橙子、咖啡罐、洗洁精、马克杯、遥控器、木勺、蜡烛、小花瓶、盆栽、狼摆件），和 `grasp_atomic` 同样的任务、空种子和起点种子段，区别是沙盒带 `scripts/rsi/past.py`、不带经验库（`library_exclude` 列全部条目，纯从头），用来验证场景图记录和它的记录库列（TODO 39.6）。(b) 的命令把任务族名换成 `pick_single`、沙盒名换成 `pick-repro` 即可；原来 pick_single 跑了 6 次迭代、6 次验收（接受 4 次），约 10 小时，现任在验收池上从 0 到 26 / 60。

### 5.2 建沙盒、跑内环

```bash
git status --short                        # 任务族文件和代码都要已提交：沙盒只导出 HEAD
S=$LOCOMANI_WORKSPACE/code/rsi-scene/ga-past-a
python3 scripts/rsi/sandbox.py $S grasp_atomic_past
cd $S && nohup setsid $PY scripts/rsi/loop.py run grasp_atomic_past > loop.log 2>&1 &
cd -
```

做什么：建沙盒并抽池（诊断池 30、验收池 40，空种子族的池比 `pick_single` 小），然后在后台跑内环，直到满足停止条件（[详解](rsi_design.md)第 5 节）。`grasp_atomic_past` 的任务族文件把停止线设成现任在验收池上 85%（`stop_rate: 0.85`；默认 70%，命令行 `--stop-rate` 优先），其余条件不变：这次运行的验收门调用满 12 次、墙钟 6 小时、Manager 停、连续两轮审计作废。一张卡的机器先按第 0 节设 `RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6`；本机没有 `/workspace` 时 `export LOCOMANI_WORKSPACE=$HOME/workspace` 要在 `. scripts/paths.sh` 之前。建沙盒打印 `sandbox spec: seed scripts/rsi/seeds/atomic, N src modules kept`，几十秒。每次迭代：Reviewer 读诊断池的记录提假设，Prober 试参数并给假设标状态，Manager 选至多 3 条（提过两轮以上、没做过、没被否定的机制会被驱动程序强制占一个名额），3 个 Engineer 并行写代码，Composer 合成一个候选，验收门判（`--candidates 1` 只用一个 Engineer、没有 Composer）。沙盒里带经验库，不含这个任务族自己的那条。每局记录带失败类型和关键事件，验收结论按类型给多赢少赢（TODO 33.15–33.17）。现任在验收池上已经超过 70% 时接着跑，要加 `--stop-rate`（比如 `--stop-rate 0.85`），否则一开跑就停；续跑沿用上次记下的值。
角色在沙盒里能用的命令（规则第 18、32 条；都是人写的，循环只调用）：
- `scripts/rsi/try.py`：点名几个起点、用一份技能目录跑，走 12 进程队列并行，返回逐局结果和对现任的翻转；每次至多 40 局、每个角色每轮至多 240 局（TODO 37.1）。
- `PYTHONPATH=src $PY scripts/rsi/body.py reach --x 0.45 --y-out 0.05 --z 0.9 --att level --hand right`：身体模型——这个伸手目标（骨盆朝向系：前伸、朝手外侧、离地高度，加手腕姿态）控制器送不送得到、差多少。表在 `outputs/scene_rsi_20261016/body/`，建沙盒时复制进去（TODO 33.1.4）。
- `PYTHONPATH=src $PY scripts/rsi/choose.py outputs/scene_discovery/adt_apartment_mars Mug_2 --start <起点文件>`：站位和抓法一起选——每种抓法、左右手、站位、姿态的预计送达概率，最好的在前，并说明兜底顺序（站着 → 降骨盆 → 准备动作，TODO 33.12）。
- `PYTHONPATH=src $PY scripts/rsi/records.py features --task pick_Mug_2` 和 `PYTHONPATH=src $PY scripts/rsi/records.py query events.reach_gap_cm ">=" 5 --task pick_Mug_2`：记录库——角色看得到的全部历史局，每局一行，起点、事件和每一步的数值展开成列；查一个量和成败的关系。假设里写了 `measure` 的，驱动程序在探针之后按它定状态（TODO 33.15.1）。原仓库的库文件由人汇总：`python3 scripts/rsi/records.py build <沙盒> ...`，写到 `outputs/scene_rsi_20261016/records/library.jsonl`，建沙盒时拷进去。场景图记录（每局 `.json` 旁的 `<label>.past.jsonl`，每个步骤、每次重试一条，TODO 39、规则第 38 条）用 `PYTHONPATH=src $PY scripts/rsi/past.py show <diag 目录>/<label>` 看一局、`past.py compare <失败局> <成功局>` 对照两局；摘要的“Scene-graph transitions”一节从它来，旧沙盒没有它就没有这一节（[详解第 5 节](rsi_design.md)）。
- 假设里写 `design`，驱动程序在探针之后跑设计探针：一个因素分几档，强制在同一批诊断起点上（`scripts/rsi/design.py`；单独试可以 `PYTHONPATH=src $PY scripts/rsi/design.py grasp_atomic --pool <起点目录> --skills <技能目录> --out <目录> --factor spawn_bearing --levels 0 180 --starts 2`，TODO 33.19）。
- `PYTHONPATH=src $PY scripts/rsi/prep.py outputs/scene_discovery/adt_apartment_mars DishSoap_1 --start <起点文件> --skills <技能目录>`：挑选器没有站位时，转、拉、挪邻居哪种目标能换来站位，技能包里有没有实现（TODO 33.5；任务族 `grasp_prep` 用起点的 `placement` 选项专抽这类局面）。
驱动程序的开关（都有默认值）：三份草稿里两份交了再过 10 分钟没交的停掉（`--engineer-late-min`，0 不停，TODO 37.3）；合成候选被拒时同一轮再送最好的那份草稿（`--second-shot`，默认开，算一次验收门调用，TODO 37.4）；同一类失败（走路摔倒）有 2 局以上时，驱动程序把每局失败那一刻的命令从同一起点单独再跑一遍，判是动作层还是技能的毛病，动作层的写进 `loop/escalations.jsonl`（`scripts/rsi/isolate.py`，TODO 33.20）；现任在诊断池上失败少于 10 局时，下一轮开头重抽诊断池——按诊断种子段抽新起点、现任跑一遍、失败的换进来到 15 局（`--diag-min-fails` / `--diag-target-fails` / `--diag-draw-max`，0 关；`loop.py redraw <族>` 在续跑前单独做一次，TODO 33.23）；一张卡的机器见第 0 节的 `RSI_GPUS` / `RSI_PROCS`；角色的推理档位 `--effort 角色=档位`（默认 Reviewer high，其余 xhigh，TODO 37.5）。
得到什么：账本在 `$S/outputs/scene_rsi_20261016/grasp_atomic_past/loop/`，每轮一个 `iter_NN/`；现任的诊断局在 `…/grasp_atomic_past/diag/incumbent_<hash>/`，每局 `.json`、`.npz`、`.past.jsonl`。`grasp_atomic` 从空种子的第一次运行是 2 次采纳、封存集 0 → 31 / 50（[记录](handoff/grasp-atomic-line-20261023.md)）。

```bash
cd $S && $PY scripts/rsi/loop.py status grasp_atomic_past; cd -
```

做什么：打印进度：第几次迭代、每次验收的结论、现任在验收池上的成绩、已花的时间和模型费。随时可以看。

第 0 轮的 Reviewer 结束后（账本里出现 `"step": "review"`，约 20–40 分钟），确认记录器和角色的权限都在工作：

```bash
cd $S
F=outputs/scene_rsi_20261016/grasp_atomic_past
ls $F/diag/incumbent_*/ | head                                                 # 每局有 .json .npz .past.jsonl
sed -n '/Scene-graph transitions/,/^## /p' $F/loop/iter_00/diag_digest.md     # 场景图一节 + 末尾的图列清单
grep '"step": "review"' $F/loop/ledger.jsonl | tail -1                         # "void": false，seconds，cost_usd
grep -c "requires approval" $F/loop/iter_00/reviewer_raw.json                  # 应为 0：Bash 放行了
cd -
```

得到什么：三个文件都在、摘要有那一节、`void` 为 false、计数为 0，就可以放着跑。空种子的现任一局都不动（`plan_for` 返回 None，记录 `why` 是 `no skill covers this goal`），这种局不加载世界、不跑仿真，`.past.jsonl` 只有一行 `note`（2026-11-01 之前建的沙盒里是 0 字节的空文件，意思相同）；第一个候选开始动以后，每局才有按步骤的记录，一开始图列也只有 `hold.dist_m` 这类；`reach.*` / `grip.*` 列要到第一个被接受的候选之后才出现。计数不为 0 说明这台机器的 `claude -p` 没放行 Bash（`loop.py` 的 `--allowedTools`，TODO 39.6），Reviewer 什么工具都跑不了，停掉查。

之后要看 Reviewer 看到的东西（都在沙盒里跑，只读）：

```bash
cd $S
D=$F/diag/$(ls $F/diag | grep incumbent | tail -1)
PYTHONPATH=src $PY scripts/rsi/past.py show $D/<label>                       # 一局：每步的实体、前后关系、翻转事件、期待与差异
PYTHONPATH=src $PY scripts/rsi/past.py compare $D/<失败局> $D/<成功局>        # 两局按步骤种类对齐，只列值不同的边
PYTHONPATH=src $PY scripts/rsi/records.py features --task pick_Mug_2         # 记录库的列（含场景图列 reach.beside.gap_m 等）和范围
PYTHONPATH=src $PY scripts/rsi/records.py query reach.beside.gap_m "<" 0.03 --task pick_Mug_2 --expect fails   # 一个 measure 的分组结果
python3 -c "import json;[print(h['id'],h.get('status'),h.get('measure')) for h in json.load(open('$F/loop/iter_00/hypotheses.json'))['hypotheses']]"
cd -
```

做什么：`show` / `compare` 看单局和对照；`features` / `query` 是 Reviewer 写 `measure` 前后用的同一套命令；`hypotheses.json` 里 `status` 由探针或 `measure` 定（已验证 / 已否定 / 仍是假设）。要量改动有没有用，就看账本 `review` 行的用时费用、用了 `measure` 的假设数随轮次的变化，和 Reviewer 调用里打开 `.npz`、自己写脚本的次数（会话记录在 `$LOCOMANI_TMP/rsi_scene_ga-past-a/`）。

### 5.3 封存集、写回、登记

```bash
cd $S && $PY scripts/rsi/sealed.py grasp_atomic_past; cd -
SUM=$S/outputs/scene_rsi_20261016_hidden/grasp_atomic_past/sealed/summary.json
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['final'], s['versions'])" $SUM
```

做什么：在封存集（50 局）上跑种子和每一版被接受的现任，`versions` 是各版本的成功数、`final` 是最终版本的名字。11-27 起内环正常停下时（不是异常停、pause、`--stub`、重放或 `--alternating`）`loop.py` 已自动跑过这一步，账本有 `sealed` 行（rc、final、ok/n、missing），输出在 `loop/sealed.log`；上面的手工命令只在 `missing` 非空时补跑（TODO 55.4）。空种子的族到这里为止：写不回主仓库，结果以 `summary.json` 为准，记进方案页和 STATE。

(b) 种子技能目录的族（`pick_single`）还要写回和登记：

```bash
python3 scripts/rsi/writeback.py $S pick_single
$PY scripts/rsi/regress.py add pick_single $S pick --last $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['final'])" $SUM)
```

做什么：第一条把最终现任三方合并进主仓库的技能目录；第二条把这个任务族登记进回归，记录取最终现任在封存集上的那一批，以后每次写回或改场景都和它比。
得到什么：`$SUM` 里 `versions` 列出种子和最终版本各自的成功数，`final` 是最终版本的名字；`prediction` 是每个版本上身体模型的预测对实送的比较：空手伸手的 AUC、预计误差对实送误差、失败局里出手前就被判"多半够不到"的有几局（TODO 33.1.2）。有经验库的运行再跑 `python3 scripts/rsi/report/library_monitor.py run $S --out $S/library.json`：假设里点名经验库的比例和点了哪几条、用词接近原文的比例、每轮假设数、Manager 认定的旧机制数（TODO 33.11，只读记录，几秒钟）。原来种子 0 / 80，最终 41 / 80（当时的场景）。写回改动了 `src/locomani/tasks/scene/skills/`，跑 `make test` 通过后提交。


### 5.4 看场景图记录（`.past.jsonl`）

每局跑完，运行器在 `.json` / `.npz` 旁边写一份 `<任务>_s<seed>.past.jsonl`（TODO 39）：每行一条记录，对应一个技能步骤或一次重试，一局几十 KB。给 Reviewer 和给人看的是同一份。位置（以 `ga-past-b` 为例）：

```
$S/outputs/scene_rsi_20261016/grasp_atomic_past/
├── diag/incumbent_<hash>/<任务>_s<seed>.past.jsonl   现任在诊断池上每一局的记录
├── diag/candidate_<hash>/<任务>_s<seed>.past.jsonl   每份草稿、Composer 候选在诊断池上的记录（被接受后改名成 incumbent_）
├── loop/iter_NN/diag_digest.md                      每轮给 Reviewer 的摘要，含 "Scene-graph transitions" 一节
└── loop/iter_NN/reviewer_context.json                Reviewer 的上下文；`round_features` 每局一行，列就是知识条目条件可用的那些（`start.*`、`<步骤类>.<关系>`、`first_bad.*`）
$S/outputs/scene_rsi_20261016_hidden/grasp_atomic_past/
└── gate/<NN_hash>/candidate/、sealed/<版本>/         验收池和封存集的局，人能看，角色不能
```

`_retired_` 开头的是诊断池重抽时换下来的起点，可以不看。在主仓库根目录读（`past.py` 不改任何文件）：

```bash
D=$S/outputs/scene_rsi_20261016/grasp_atomic_past/diag/incumbent_<hash>
for f in $D/pick_*.json; do python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['label'], r['ok'], r.get('failure_class'))" $f; done | column -t   # 哪些局成、哪些败
PYTHONPATH=src $PY scripts/rsi/past.py show $D/pick_Mug_2_s410023                 # 一局，按步骤展开
PYTHONPATH=src $PY scripts/rsi/past.py show $D/pick_Mug_2_s410023 --step 2 --json # 某一步的原始 JSON
PYTHONPATH=src $PY scripts/rsi/past.py compare $D/<失败局> $D/<成功局>             # 两局按步骤种类对齐，只列值不同的边
sed -n '/Scene-graph transitions/,/^## /p' $S/outputs/scene_rsi_20261016/grasp_atomic_past/loop/iter_03/diag_digest.md  # 一轮的汇总
```

`show` 每一步的输出：标题行 `序号.重试 步骤名 [原语种类] 起止时间 成败`（种类后带 `?` 是从轨迹的原语标签推的、没有调用日志）；`roles`（目标、哪只手、支撑面、站位等）；`check`（动作层的检查值）；每条边一行 `主体 —关系→ 客体  前→后 （数值） （期待 ✓/✗）`；`events` 中途的翻转和时刻；`joined` 执行中才碰到的实体；最后一行是这一步在 `.npz` 里的拍区间。值的写法：`true` / `false`；`unknown` 是算不出（旧局没有接触通道）；`-` 是那一刻没有这条边。空种子没动的局只有一行 `no records: no skill covers this goal`。

关系有五类，再加谓词：接触 `touching`；持握 `gripped`（两指都碰到）、`held`（谓词：夹住且离开所有台面）、`rel_pose`（物体在手里的偏置）；支撑 `supported_by`、`in`；空间 `above`（上方净空）、`beside`（邻居的缝和方位）、`near_edge`（离哪条台沿多远）、`blocking`（和站位或接近走廊相交多深）、`dist` / `facing`（机器人到目标）；相对位姿 `reach_vec`（目标点在骨盆系的前伸 / 外侧 / 高度）、`approach`（从物体哪一面接近）、`at`（手或机器人到没到目标点）；谓词 `upright`。`expected` 是按原语的固定表（伸手要到位、目标和邻居不动、走廊无阻挡；合爪要两指夹住；拿着往上伸要离开支撑……），`discrepancies` 是没成立的期待——只是差异，不是因果。完整字段见[代码结构与流水线](RSI_system_pipeline.md)第 4.2 节。

自己写分析时直接读 JSON：`head -1 $D/<label>.past.jsonl | python3 -m json.tool`；字段有 `kind`、`roles`、`scope`（范围和每个实体为什么进来）、`entities`、`G_before` / `G_after`（`states`、`edges`）、`events`、`expected`、`discrepancies`、`outcome`、`evidence`。记录库按步骤种类把它们展开成列（`reach.beside.gap_m`、`grip.moved_before_grip`、`first_bad.kind` 等），`scripts/rsi/records.py features` 列出来（5.2）。2026-11-02 之前的记录有两处误报，记录器已修（TODO 39.7）：走路步骤的 `blocking` 把整个公寓（`ApartmentEnv`）和目标所在的家具算成挡住站位，伸手的 `at` 拿手腕而不是夹爪中心去比、总差约 15 cm；读旧记录时忽略这两类 ✗（下面的页面自动去掉第一类）。

给人看的图形页（不进沙盒，TODO 39.7）：

```bash
PYTHONPATH=src $PY scripts/rsi/report/past_view.py $D/<label>                          # 一局一页
PYTHONPATH=src $PY scripts/rsi/report/past_view.py $D/<失败局> --compare $D/<成功局>     # 两局按步骤种类对齐
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src $PY scripts/rsi/report/past_view.py $D/<label> --frames   # 加每步前后两帧 MuJoCo 画面
PYTHONPATH=src $PY scripts/rsi/report/past_view_en.py $D/<label>                       # 英文版（参数相同），写到 <label>.en.html
```

一个目录全部生成见 5.5。

得到什么：`outputs/scene_rsi_20261016/past_view/<label>.html`（`--out` 改），一个自包含的 HTML，浏览器直接打开；不带帧约 50 KB，带帧约 0.5 MB。每一步左边是俯视图（世界 xy：台面、家具、站位圆盘、接近走廊按真实大小，物体是方块、手是点、机器人带朝向；步骤开始时淡色、结束时实色，位移画箭头；接触、持握、支撑、邻居、阻挡、接近画在两个实体之间，到位和伸手向量画成箭头，上方净空是标注），右边是按类分组的关系清单（前 → 后、数值、来源、期待 ✓ / ✗），图下是中途事件（同一条边 0.3 秒内的反复翻转合成一行"抖动 N 次"）。整局一个固定视野，"放大到这一步"按钮切换；"只看翻转 / 差异"只留这一步变了的和没成立的；悬停在关系名上看中文释义；URL 加 `#N` 直接打开第 N 步。对照页两局的每一对步骤占一行，值不同的边标黄。

### 5.5 一个目录的场景图页全部生成（`past_view_all.sh`）

把一个目录里每一局都生成一页，再写一个索引页。默认英文版、带 MuJoCo 帧。在主仓库根目录执行：

```bash
# 1. 环境（新终端；第 0 节已设过就跳过）
cd ~/workspace/locomanipulation
export LOCOMANI_WORKSPACE=$HOME/workspace
. scripts/paths.sh

# 2. 选目录：一个版本在诊断池上的全部局（diag/ 下有哪些版本先 ls 看）
S=$LOCOMANI_WORKSPACE/code/rsi-scene/ga-past-b
ls $S/outputs/scene_rsi_20261016/grasp_atomic_past/diag/
D=$S/outputs/scene_rsi_20261016/grasp_atomic_past/diag/incumbent_0fa018936751

# 3. 生成
bash scripts/rsi/report/past_view_all.sh $D                 # 默认：英文、带帧
bash scripts/rsi/report/past_view_all.sh $D --no-frames     # 只要图和关系清单，每局几秒
bash scripts/rsi/report/past_view_all.sh $D --lang zh       # 中文版

# 4. 看：浏览器打开索引页，逐局点进去
xdg-open outputs/scene_rsi_20261016/past_view/ga-past-b__incumbent_0fa018936751/index.html
```

参数：

| 参数 | 作用 | 默认 |
|---|---|---|
| `--lang en\|zh` | 页面语言 | `en` |
| `--no-frames` | 不渲染 MuJoCo 帧（不需要 GPU） | 带帧 |
| `--retired` | 连诊断池重抽时换下的 `_retired_*` 局一起生成 | 跳过 |
| `--gpu N` | 帧用的 EGL 设备（`MUJOCO_EGL_DEVICE_ID`）；SRA0889 的设备 1 是核显，必须用 0 | `0` |
| `--out DIR` | 输出目录 | `outputs/scene_rsi_20261016/past_view/<沙盒>__<目录名>/` |
| `--scene DIR` | 场景目录（`scene_meta.json`） | `outputs/scene_discovery/adt_apartment_mars` |

得到什么：输出目录里每局一页 `<label>.en.html`（中文版是 `<label>.html`），和一个 `index.html`——每局一行：局名、任务、结果、失败类型、步骤数、页面链接，上方是总成功数。终端逐局打印 `[i/N] <label>`，最后打印索引页路径和成功数。带帧时每局要重新加载一次 MuJoCo 场景，约 10 秒；`ga-past-b` 一个诊断目录 22 局约 3.5 分钟、8 MB，不带帧约 5 秒、2 MB。某一局生成失败时，终端打印 `FAILED` 和日志位置（输出目录下的 `.log_<label>`），索引里那一行标 `page failed`，其余局照做，最后退出码 1。也可以对验收池或封存集的目录用（隐藏目录 `…_hidden/<任务族>/gate/<NN_hash>/candidate/`、`sealed/<版本>/`），只是那里的局只给人看。

### 5.6 知识库

```bash
PYTHONPATH=src $PY scripts/rsi/knowledge.py build <沙盒> <任务族> --iters 0-3
```

做什么：从这个沙盒的账本和场景图记录抽知识条目，写进 `scripts/rsi/` 下的 `knowledge/knowledge.jsonl`（一个共享文件，每条带 `tag`：`single` 单手抓起、`bimanual` 双手抱起、`mixed` 其他；用户 11-18），替换这个任务族以前的条目，别的保留；建库日志在 `knowledge/build/<任务族>.md`（每个机制进了还是丢了、为什么），加 `INDEX.md`。证据没变的机制沿用上次的结果，不再调用模型。约每个机制 1 USD，10 到 15 分钟。建库同时把这个任务族的假设引用条目（`knowledge_refs`）的结局记进 `knowledge/usage.jsonl`，定出每条条目的状态（`tested`、`candidate`、`doubtful`）；只更新这一项用 `PYTHONPATH=src $PY scripts/rsi/knowledge.py usage <沙盒> <任务族>`。

```bash
PYTHONPATH=src $PY scripts/rsi/knowledge.py eval <沙盒> <沙盒>/outputs/scene_rsi_20261016/<任务族>/diag/incumbent_<hash> <任务族> --out /tmp/hits
```

做什么：把库里每条条目在这版现任的诊断局上求值，写 `/tmp/hits/knowledge_hits.json` 和沙盒里的 `<沙盒>/scripts/rsi/knowledge/KNOWLEDGE.md`（整个库，每条下面标出条件成立的局）；打印命中失败局的条目数。内环每轮 Reviewer 之前自动做这一步。`sandbox.py --fresh` 建的沙盒不带知识库和经验库；两个任务族交替跑、共享知识和代码见 `scripts/rsi/alternate.py` 和[单手 / 双手测试场景 runbook](rsi_logistic_room_1030_single_bimanual.md) 第 7 节。


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
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 $PY scripts/rsi/report/evolution_media.py --sandbox $S --family pick_single --out outputs/scene_rsi_20261016/pick_repro_media --only task --versions final
```
做什么：读沙盒的账本，取种子和每一次被验收门接受的版本，再读它们在诊断池上的逐局记录和轨迹，出三样东西（`--only step,overview,task` 选其中几样，默认全出）。`step_NN.mp4`：第 NN 次被接受的改动，在它改前改后都跑过的起点里选一个被它从失败翻成成功、轨迹最短的，左改前右改后，3 倍速（`--cards <JSON>` 按版本哈希用 `"label"` 指定起点，`"_names"` 给物体改名；种子没有技能时左边放起点第一帧）。`task_NN_<任务>.mp4`：任务族的每个任务一段，取最终版本做成的最短一局，没有成功就取最短的失败局（规划拒绝也是一局，机器人站着不动），左种子右最终版本，每个版本在该任务诊断起点上的成功数写进 `tasks.json`；只要最终技能、不要种子对照时加 `--versions final`，同样的起点，单画面，文件名 `final_NN_<任务>.mp4`，不覆盖双画面的那套。`overview.png`：每个任务一行、每个版本一列，每格是那一局决定成败的那一刻（成功取抬起后一秒，失败取结尾）。视频开头没有字幕卡（2026-11-25 起），第一帧就是场景，顶部字幕条标版本和成败。
得到什么：`outputs/rsi_repro/visual/` 下的视频、同名海报 `.jpg`、`previews/`，索引 `versions.json`（版本、哈希、来自第几轮、验收门数字、Manager 总结）和 `tasks.json`（每段任务视频的起点和各版本成功数；`--versions final` 的写到 `tasks_final.json`）。一段视频约 1 分钟，单画面约 40 秒；被接受两次、8 个任务的运行约 10 分钟。

**只用于 pick-a 那次运行：`visualize.py`**

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 $PY scripts/rsi/report/visualize.py --sandbox /workspace/code/rsi-scene/pick-a --out outputs/scene_rsi_20261016/visual
```

做什么：pick_single 第一次运行（原机器的 `pick-a` 沙盒）的手写纪录片，讲 Reviewer 每一轮看到了什么：那次的 5 个版本哈希、起点编号、每轮的线索和字幕都在代码里（`VERSIONS`、`clue_0`–`clue_3`、`main` 的 `cards`）。`--only clue,step,infeasible,overview` 选：`clue_NN.png` 第 NN 轮的线索图，画在那一局 Reviewer 引用的那一刻上；`step_NN.mp4` 同一起点改前改后并排，2 倍速，前面一张线索卡；`infeasible_book.png`、`infeasible_wallet.png` Reviewer 论证现有夹爪抓不了的两个任务；`overview.png` 5 个指定起点 × 5 个版本。
得到什么：只对 `pick-a` 沙盒有效，换一个沙盒它在开头查到缺那几个版本目录就退出并提示改用上面那条；pick-a 的诊断局不在复现材料里，别的机器上跑不了。它同时是其他报告脚本共用的底层（`Scene` 回放和渲染、`Round` 读一局、`why_zh` 失败判词的中文对照）；要给自己的运行出线索图，照 `clue_N` 另写。

## 9. Unreal 渲染

前提是[安装](setup.md)第 8 步（引擎、URLab 插件、`$BRIDGE_PY`）和 PATH 上的 Blender 5.x；`$SC` 是第 2 节的场景目录。编辑器一台机器只能开一个，它渲染的卡要和仿真的卡分开，一张卡的机器写 `--gpu 0` 并且不和内环同时跑；首次启动编译着色器要几分钟，之后约 15 秒。

### 9.1 生成 Unreal 资产：每台机器一次

```bash
bash scripts/start_unreal.sh --gpu 1
blender --background --python scripts/adt_to_usd.py -- --unreal $SC/scene_meta.json $SC/unreal_room
"$BRIDGE_PY" -m locomani.sim.unreal.build_level --room $SC/unreal_room --room-level AdtApartment
bash scripts/build_scene.sh adt_apartment_mars --unreal
```

做什么：第 1 条起编辑器并等它接受命令（关卡是在运行中的编辑器里建的）。第 2 条 Blender 把公寓里除可动物体外的所有东西连同吊顶灯烘成 `$SC/unreal_room/room.glb` 和 `room_scene.json`。第 3 条把它建成房间关卡 `/Game/Levels/AdtApartment`。第 4 条建场景关卡 `adt_apartment_mars`（`configs/scenes.yaml` 的 `unreal.build`）：复制房间关卡，导入机器人、可动物体、第三人称机位和 ghost 夹爪，把场景 XML 的哈希记在 `$UNREAL_ASSETS/adt_apartment_mars/source.json`。
得到什么：`$UNREAL_ASSETS/adt_apartment_mars/` 和宿主工程里的两个关卡，以后每次渲染直接用；第 3 条成功时 `$SC/unreal_room/unreal_level_report.json` 里 `"ok": true`。只有第 2 节的场景改过（`scene_locomani.xml` 变了）才重做第 4 条，否则渲染会因哈希不符拒绝；房间烘焙和房间关卡不用重做。烘焙约 3 分钟，房间关卡约 20 分钟（`room.glb` 1.1 GB、400 个网格；命令最多等一小时）。

### 9.2 渲染视频：每次

```bash
bash scripts/start_unreal.sh --gpu 1
PYTHONPATH=src MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$BRIDGE_PY" scripts/rsi/report/report_unreal.py all --media outputs/rsi_repro/visual --out outputs/rsi_repro/visual_unreal
bash scripts/stop_unreal.sh
```

做什么：第 1 条在编辑器没开着时才需要。第 2 条读第 8 节 `evolution_media.py` 输出目录里的索引（`steps.json`、`tasks.json`、`tasks_final.json`），把每段视频的那几局通过 URLab 回放进关卡，沿 MuJoCo 版同一条机位轨迹以 1280 × 960 拍摄，再用同样的起点、版本和顶部字幕条合成；`all` 或点名（`step_01`、`task_07_tv_remote_pickup`、`final_03_jam_pickup`）。不带 `--media` 时它渲染的是进展报告自己的清单（`ps_00`、`on_02` 等，`--help` 列全），那些局只在原机器上。第 3 条关编辑器。
得到什么：`--out` 下每段 `<名字>.mp4`、`previews/`、海报，`<名字>/<面板>/NNNNN.jpg` 逐帧，和 `compare_*.jpg`（同一时刻 Unreal 与 MuJoCo 上下对照）；一段几分钟。渲染期间不要跑别的重任务，编辑器的守护脚本在 CPU 饱和时会把它杀掉，日志在 `$UNREAL_ROOT/logs/`。

## 10. 常见错误

- `scene_lint.py` 报 error：不要开跑循环。先看 `lint.json` 里是哪一类（物体掉下去、目标开局就成立、竖直轴不对），多半是场景没按第 2 节的顺序建，或者少了面板、碰撞盒。
- 内环一开始就停，`loop.log` 里说缺 `pool_diag`：沙盒没建完，重新跑 `sandbox.py`。
- 账本里某次迭代或草稿标 `void`：角色读了验收池或沙盒外的文件，或者 Engineer 改了技能目录以外的文件。三个 Engineer 中一份草稿越界，只作废这一份，这一轮接着走；Reviewer、Manager、Composer 第一次越界先换新会话重跑一次（账本记 `rerun`，提示词里点名越界的类别和路径，第一次的文件移到隐藏树 `void/iter_NN/first/`），重跑再越界、或唯一的 Engineer 越界，这一轮作废（账本记 `void_iter`），现任不变，进下一轮（TODO 55.3）；连续两轮作废，内环停下，`loop.py` 退出码 3，查完用 `--reprobe N` 续跑（TODO 36.7）。验收门调用标 `void` 是有起点缺了局，这次不算数；草稿或候选标 `named_objects` 是技能代码里写了物体名字（规则第 30 条）或读了场景图记录器——import `locomani.tasks.scene.past`、字符串含 `past.jsonl` / `rsi/past.py`（规则第 38 条），作废。某局的 `.past.jsonl` 只有一行 `"error"`：记录器在这一局上出错，局的结局和 `.json` / `.npz` 不受影响，摘要里这一局就没有场景图时间线；把错误行报给维护者。
- 验收门从规则第 14 版起按“多赢 − 少赢 ≥ 3 且少赢 ≤ 多赢 / 3”判，不再要求 p ≤ 0.05（TODO 33.8）；被拒但多赢多于少赢的最好 2 个候选在 `loop/candidates_pool.json`，账本 `pool` 行记进出。Manager 决定里的 `base` 是从候补接着改，Composer 的 `adopted_pool` 是把候补原样再送；旧规则拒过的候选可以再判。
- 回归打印 `worse at p <= 0.05`：有任务族变差，登记表不推进。先查最近改了什么，再决定是否接受。
- 仿真很慢、结局和记录差很多：确认 `OMP_NUM_THREADS=2`，并且机器上没有别的仿真在跑（所有批次共用 12 个仿真槽）。
