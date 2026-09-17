# 运行基础设施：一份实验文件，六个脚本

2026-09-17 的重构把"采集 → 转换 → 训练 → 评测"变成**每个阶段一个扁平脚本**：从上往下读完一个文件，
就知道这个阶段做了什么。没有配置组合、没有作业调度器、没有阶段注册表。
决策过程见 [重构计划](handoff/refactor-flat-scripts-20260917.md)（历史文档）。

任务本身怎么做（专家策略、验收标准、失败家族）在各采集指南里：
[单臂](collection.md)、[双臂](crate_collection.md)、[pick-and-place](pick_place_collection.md)。

## 1. 一次运行 = 一份实验文件

```bash
. scripts/paths.sh
export SONIC_DEPLOY_ROOT=<打过夹爪补丁的 deploy 根>      # 见第 6 节
python scripts/pipeline.py configs/experiments/bottle_simple_200.yaml --gpu 1
```

`configs/experiments/<name>.yaml` 是**完整**的：一次运行读的每个值都在里面，没有继承、没有组合。
八个块——`name` + `task` / `embodiment` / `scene` / `backend` / `collect` / `train` / `eval`。
新实验就是**复制一份改几行**；两份实验 `diff` 一下，就是两次运行的全部差别：

```bash
diff configs/experiments/bottle_simple_200.yaml configs/experiments/bottle_simple_smoke.yaml
# 只有 8 行不同：采集规模、训练步数、评测轮数
```

现有实验：`bottle_simple_200`（冻结单臂专家采 200 集 + 10k 步 + 100 局）、
`bottle_simple_gate8`（8 局固定库，闸门用）、`bottle_simple_smoke`（全链路冒烟）、
`pick_place`、`crate_table30_record`（双臂录制线）。

运行开始时这份文件会被**逐字节复制**到 `outputs/runs/<name>/config.yaml`。
**下游阶段读的永远是那份副本**，不是 `configs/experiments/` 里的原件——所以跑到一半改实验文件，
不会改变正在跑的运行，也不会改变已完成运行的含义。

另有 `configs/{task,backend,eval}/` 是**按名字引用的 profile 库**（`profiles.load("task", "bottle_simple")`），
专家、评测器和测试用它们，实验文件则把选中的那份**展开**进自己。

## 2. 六个脚本，一个调度

| 脚本 | 做什么 |
|---|---|
| `scripts/collect_bottle.py` | 单臂 / pick-and-place：按批起栈、跑 `grasp_gym.py`、停栈，直到收够 |
| `scripts/collect_crate.py` | 双臂：按 `task.expert` 选专家（visual / wall / carry），同样按批 |
| `scripts/convert.py` | session → LeRobot 数据集 → GR00T 统计 → `check_dataset --run-dir` |
| `scripts/train.py` | 把 `train`/`embodiment` 块变成 `train_bottle.sh` 的环境变量 |
| `scripts/eval_bottle.py` | 起评测栈 + 策略服务，整份 `eval` 块交给评测器 |
| `scripts/eval_crate.py` | 双臂评测器自带栈，本脚本只把运行交给它 |
| `scripts/pipeline.py` | 按 `task.plugin` 分派上面这些，按顺序跑 |

每个脚本都能单独跑：`python scripts/train.py outputs/runs/<name> --gpu 1`。
公共部分在 `scripts/run_common.py`（读实验、复制、资源检查、dry-run 打印、写摘要）。

**先看会发生什么，再动手**——每个脚本都有 `--dry-run`，打印将执行的每条命令和它专属的环境变量，
且**一个字节都不写**：

```bash
python scripts/collect_bottle.py configs/experiments/bottle_simple_gate8.yaml --gpu 1 --dry-run
```

## 3. 并行、跳过、续跑

**并行就是多开终端**，一张卡一个：

```bash
python scripts/pipeline.py configs/experiments/A.yaml --gpu 0     # 终端 1
python scripts/pipeline.py configs/experiments/B.yaml --gpu 1     # 终端 2
```

没有作业文件、没有调度器、没有 GPU 分配器。要比较结果就看各运行目录的摘要：

```bash
cat outputs/runs/*/summary.json
```

**产物存在就跳过**：`summary.json` 有 collect 条目 / `train_dataset/` / `model/` / `eval/closed_loop.json`。
要重跑某个阶段，**删掉它的产物**就行——整个续跑故事就这一条，`ls` 一眼看得见。
`pipeline.py` 另有 `--from` / `--to` 限定跑哪几个阶段。

采集内部还有批级续跑：`collect/inputs.json` 记录已完成的批次和保留条数，同名重跑接着采。
这不是任意时刻的 episode 级恢复——先看最后一批的日志和 manifest 再继续。

## 4. 运行目录长什么样

| 路径 | 内容 |
|---|---|
| `config.yaml` | 实验文件的逐字节副本，这次运行的 provenance |
| `summary.json` | 每个阶段一条记录：保留数、产出率、失败家族、数据集、checkpoint、成功率 |
| `stack_env.json` | 仿真栈告诉子进程的**全部**环境变量 |
| `commands.jsonl`、`timings.jsonl` | 每个子进程的命令行与耗时 |
| `collect/manifest_NN.jsonl`、`videos_NN/`、`state_trace_NN/` | 逐局成败与原因、第三人称录像、物理轨迹 |
| `train_dataset/`、`val_dataset/` | LeRobot 数据集（`train.holdout` 为 0 时没有验证集） |
| `model/checkpoint-<steps>/` | 微调产物 |
| `eval/closed_loop.json`、`eval/videos/`、`eval/eval_config.json` | 闭环结果、录像、**实际交给评测器的那份配置** |

## 5. 仿真栈：三个进程，一个对象

仿真（MuJoCo）、控制器（C++ deploy）、导出器（LeRobot 录制）由 `locomani.stack.Stack` 拉起：
各自独立日志、各自明确的就绪信号、按进程组停止并回收。
`StackSpec.settings()` 是**唯一**决定子进程环境的函数，其结果落盘为 `stack_env.json`。

阶段脚本自己起停栈，正常流程不用管。手工检查动作时：

```bash
python -m locomani.stack.cli up --instance 0 --gpu 1 --scene outputs/scene/sim_lowcom
python -m locomani.stack.cli status --instance 0
python -m locomani.stack.cli down --instance 0
```

实例即端口块（基准 + `10 × instance`），并用 `$LOCOMANI_TMP/locomani_locks/instance_N.lock` 排他。
**只有 instance 0 会跑导出器**，所以并行按 GPU 分而不是按实例分；多实例还要求 deploy 打过
`sonic_dds_domain.patch`（[安装指南](setup.md)）。

## 6. 排错

| 症状 | 原因与处理 |
|---|---|
| `needs patches/sonic_hand_ownership.patch` | streamed-pose 专家（`reach.strategy` 为 `simple`/`whole_body`）要求打过补丁的 deploy。按 [安装第 3 步](setup.md) 重编，或 `export SONIC_DEPLOY_ROOT=<打过补丁的根>`。这个检查在**起任何进程之前**执行 |
| `not enough headroom -- cpu/pids/gpu ...` | 资源不足时脚本**拒绝启动并说明原因**，不等待也不硬跑。配额打满时仿真会掉墙钟，专家的计时阶段会集体失败、看起来像机器人故障 |
| `<run>/config.yaml differs from ...` | 同名运行的实验文件被改过。换一个 `--name`，不要在运行中途改配置 |
| `model/ exists` / `closed_loop.json exists` | 阶段拒绝覆盖已有产物。检查后删掉它再重跑 |
| 某阶段"被跳过"但你想重跑 | 删掉它的产物（见第 3 节） |
| 栈起不来、端口被占 | `python -m locomani.stack.cli status --instance N`；锁文件在 `$LOCOMANI_TMP/locomani_locks/` |

产物少了什么这类问题有专门的网：

```bash
python scripts/artifact_contract.py check outputs/runs/<name> bottle_eval
```

契约记录的是"一个阶段**必须**留下什么"（`tests/contracts/*.json`）。它存在的理由很具体：
2026-09-17 评测录像悄悄消失了一整天，阶段照样报成功、测试照样全绿。

更细的失败形态见 [隐性失败](reference/silent_failures.md)。

## 7. 进一步阅读

- [配置字段表](reference/config_fields.md)：由 dataclass 生成，字段含义和默认值以它为准。
- [Runbook](reference/runbook.md)：已实现入口的命令速查，含不录制的手工检查流程。
- [代码导航](reference/change-map-runtime.md)：改运行时、栈或阶段脚本前读。
- [重构计划与实测记录](handoff/refactor-flat-scripts-20260917.md)：这次重构的决策、每步验收数字（历史）。
