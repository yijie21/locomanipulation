# 运行基础设施：一个 CLI、配置组、阶段与并行

2026-09-16 的重构把"采集 → 转换 → 训练 → 评测"从一堆 shell runner 收敛成
**一个命令、七个配置组、五个阶段**。这一页是重构后的现状说明：怎么组合一次运行、
产物落在哪、怎么切任务和场景、怎么在多张卡上并行跑多个配置。
决策过程和当时的取舍记录在 [基础设施计划](handoff/infra-plan-20260916.md)（历史文档，不再更新）。

任务本身怎么做（专家策略、验收标准、失败家族）在各采集指南里：
[单臂](collection.md)、[双臂](crate_collection.md)、[pick-and-place](pick_place_collection.md)。

## 1. 一次运行由什么组成

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m locomani.cli show +experiment=bottle_simple_lowcom     # 只打印组合结果，不运行
```

`python -m locomani.cli <collect|convert|train|eval|stats|show> [覆盖项]` 是唯一入口
（`make run ARGS=...` 同义）。一次运行 = 七个配置组 + 五个根字段：

| 组 | 决定什么 | 目录 |
|---|---|---|
| `task` | 专家策略、验收门、失败分类、spawn 随机范围 | `configs/task/` |
| `embodiment` | 单臂 / 双臂：动作与状态维度、相机集合、GR00T modality 与 tag | `configs/embodiment/` |
| `scene` | 冻结场景目录，以及可选的固定样本库（`cases`） | `configs/scene/` |
| `backend` | SONIC 控制器参数（编码器、手型、相机、DDS domain、lockstep） | `configs/backend/` |
| `collect` | 采集规模：成功条数、每批轮数、批数上限、种子、实例、是否录制 | `configs/collect/` |
| `train` | 微调步数、保存间隔、holdout、GPU 数、global batch、基座模型 | `configs/train/` |
| `eval` | 闭环轮数、种子、预算、重规划间隔、策略端口 | `configs/eval/` |

根字段：`name`（运行名，决定 `out=outputs/runs/<name>`）、`stage`、`gpu`、`seed`、`out`。

组合方式有两种，**优先用实验文件**，因为它是可版本化、可复现的那一份：

```bash
# 版本化配方：configs/experiment/<name>.yaml 里写 override /task、override /scene …
python -m locomani.cli collect +experiment=bottle_simple_lowcom gpu=1 name=my_run
# 临时改一两个字段：点号路径覆盖，任何组的任何字段都能改
python -m locomani.cli collect +experiment=bottle_simple_lowcom collect.successes=50 gpu=1 name=my_run
```

现有实验文件：`bottle_simple_lowcom`（冻结单臂专家 + 8 局固定库）、
`bottle_simple_learn200`（同一专家，随机 spawn 采 200 条 + 10k 步微调 + 100 局闭环）、
`pick_place_default`、`crate_table30_record`。新配方加一个 YAML，不改代码。

YAML 里没有的字段在组合时直接报错（每组都有 dataclass schema，`__post_init__` 的校验照常触发），
所以"配置只生效了一半"这种失败形态在入口就被挡住。任务与本体的兼容性也在入口检查：
`crate` 只能配 `g1_bimanual`，`bottle` / `bottle_visual` / `pick_place` 只能配 `g1_single_arm`。

## 2. 阶段、产物与断点

五个阶段各是一个独立进程（`src/locomani/stages/`），彼此**只通过运行目录里的 manifest 通信**：

| 阶段 | 做什么 | 关键产物 |
|---|---|---|
| `collect` | 按批起停仿真栈，跑专家，录 LeRobot session | `collect/manifest_NN.jsonl`、`collect/videos_NN/`、`collect/state_trace_NN/` |
| `convert` | session → LeRobot 数据集，切训练 / 验证 | `train_dataset/`、`val_dataset/` |
| `train` | GR00T 微调（`scripts/train_bottle.sh`） | `model/checkpoint-<steps>/`、`train.log` |
| `eval` | 起评测栈 + 策略服务，跑闭环 | `eval/closed_loop.json`、`eval/videos/`、`eval/traces/` |
| `stats` | 汇总各阶段 manifest | `stats/summary.json` |

每个阶段结束写 `<out>/<stage>/manifest.json`：`ok`、该阶段消费的配置组的 `config_hash`、
上游 manifest 的 hash、产物路径、指标、`seconds`、以及**起跑瞬间实测**的 CPU/pid/GPU
（`resources_at_start`，事后判断"是不是被别的进程挤了"就靠它）。
运行目录里还有 `run_config.yaml|json`（完整解析后的配置，provenance）、
`commands.jsonl`、`timings.jsonl`、`deploy_receipts.jsonl`。

重跑同一个 `name`：阶段 manifest 存在且 `ok`、配置 hash 不变、上游 hash 不变 → **跳过**；
否则从该阶段重来。`collect` 内部还有批级断点（`collect/inputs.json` 记录已完成的 batch 和保留条数），
中断后同目录重跑接着采。这不是任意时刻的 episode 级恢复：先核对最后一个 batch 的日志再继续。
换场景或改任务配置时用新目录，不要覆盖旧产物。

## 3. 仿真栈：三个进程，一个对象

仿真（MuJoCo）、控制器（C++ deploy）、导出器（LeRobot 录制）是三个进程，由
`locomani.stack.Stack` 作为**自己会话的子进程**拉起：各自独立日志、各自明确的就绪信号
（`Init Done`、相机端口、`Image publish frequency`）、按进程组停止并回收。
活着的栈把 pid / 端口 / 路径写进 `<session_root>/stack.json`，任何进程都能 `Stack.attach` 上去。

`collect` 和 `eval` 阶段自己起停栈，正常流程不需要手动干预。手工检查动作时：

```bash
python -m locomani.stack.cli up --instance 0 --gpu 1 --scene outputs/scene/sim_lowcom --prompt "pick up the bottle"
python -m locomani.stack.cli status --instance 0          # pid、端口、健康检查
python -m locomani.stack.cli restart-deploy --instance 0   # deploy 掉 LowState 后的自愈
python -m locomani.stack.cli down --instance 0             # 只停自己这个实例
```

实例即端口块：每个端口 = 基准 + `10 × instance`，并用 `$LOCOMANI_TMP/locomani_locks/instance_N.lock`
上 flock 排他。**只有 instance 0 会跑导出器**，所以并行要按 GPU 分，不要按实例分（见第 4 节）。
多实例同时跑还要求 deploy 打过 `sonic_dds_domain.patch`（[安装指南](setup.md)），否则第二个 deploy 会去接管第一个 sim。

## 4. 多卡并行：作业文件与 launcher

一个作业文件描述"哪几个配置、各跑哪些阶段"，launcher 把它们铺到指定的卡上：

```bash
python -m locomani.launch run configs/jobs/example.yaml --gpus 0,1   # 并行跑
python -m locomani.launch run configs/jobs/example.yaml --dry-run    # 只打印计划
python -m locomani.launch ls --name 'bottle*'                        # 读注册表
python -m locomani.launch compare outputs/runs/A outputs/runs/B --stage eval
```

作业条目的字段：`name`（同时是默认运行名，可用 `run_name` 分开）、`overrides`（就是 CLI 覆盖项）、
`stages`、`gpu`（`any` 或具体卡号）、`resources`（按阶段覆盖资源申报）。
一个作业的阶段串行、作业之间并行，**一张卡同时只有一个作业**。

起跑前每个阶段都要过一次实测准入（容器 CPU 配额、pid 上限、该卡的空闲显存），
不满足就等待并打印原因；申报值的默认量级：

| 阶段 | CPU 核 | pid | 需要的空闲显存 |
|---|---|---|---|
| `collect` / `eval` | 8 | 1800 | 6000 / 12000 MiB |
| `train` | 4 | 300 | 28000 MiB |
| `convert` / `stats` | 4 / 1 | 200 / 10 | 0 |

这些数字不是理论值：导出器的线程会把容器 pid 用掉一大截（上限 3840），
CPU 配额是 15.36 核而不是 `nproc`，配额打满时仿真 RTF 会掉到 0.18、所有按墙钟计时的专家动作都会失败。
每个阶段结束在 `outputs/registry.jsonl` 追加一行（运行名、阶段、ok、GPU、耗时、指标、配置 hash）。

## 5. 换任务、换场景、换本体

| 想跑什么 | 怎么写 |
|---|---|
| 单臂抓瓶（冻结 simple 专家） | `task=bottle_simple scene=lowcom embodiment=g1_single_arm` |
| 单臂 pick-and-place | `+experiment=pick_place_default` |
| 双臂抓篮子 | `+experiment=crate_table30_record`（`task=crate_table30 embodiment=g1_bimanual`） |
| 换桌面 / 物体布局 | 只换 `scene=<name>`；`scene.cases` 指固定样本库，`null` 表示每局随机采样 |
| 换一版采集规模 | `collect=learn_200` 或 `collect.successes=… collect.batch_rounds=…` |

场景目录是**冻结产物**（`outputs/scene/<name>/`，由 `scripts/install_scene.sh` 生成），
运行开始时会整份拷进 `<out>/scene/`，所以事后改场景不会影响已完成的运行。

## 6. 排错

| 症状 | 原因与处理 |
|---|---|
| `needs patches/sonic_hand_ownership.patch and a rebuild` | streamed-pose 专家（`reach.strategy` 为 `simple` / `whole_body`）要求打过夹爪交接补丁的 deploy。按 [安装第 3 步](setup.md) 重编，或 `export SONIC_DEPLOY_ROOT=<打过补丁的 deploy 根>` 后再起 launcher |
| launcher 一直打印 `waiting: cpu …` / `pids …` | 别的进程占了配额；等它退出，或在作业里下调该阶段的 `resources` 申报（只在确实更省时才改） |
| `partial model exists` | `train` 拒绝往已有 `model/` 里再训一次。检查旧 checkpoint，然后换运行名 |
| 某阶段"被跳过"但你想重跑 | 改了配置就会自动重跑；确实要强制重来就删掉该阶段的 `manifest.json`（及其下游产物） |
| 栈起不来、端口被占 | `python -m locomani.stack.cli status --instance N`；锁文件在 `$LOCOMANI_TMP/locomani_locks/` |

更细的失败形态（包括历史上那些"看起来像机器人的错，其实是配置只生效一半"的案例）见
[隐性失败](reference/silent_failures.md)。

## 7. 进一步阅读

- [配置字段表](reference/config_fields.md)：由 dataclass 生成，字段含义和默认值以它为准。
- [Runbook](reference/runbook.md)：已实现入口的命令速查，含不录制的手工检查流程。
- [代码导航](reference/change-map-runtime.md)：改运行时、栈或阶段代码前读。
- [基础设施计划](handoff/infra-plan-20260916.md)：这次重构的 8 条决策与验证记录（历史）。
