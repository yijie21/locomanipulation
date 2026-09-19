# 单臂：从采集到评测

走近桌子、单手抓瓶。**这一页从头到尾能跑完整条线**，不需要跳到别的文档。
为什么这么设计、试过什么被否掉了，在 [单臂详解](single_arm_design.md)；
哪天跑出什么数，在 [单臂台账](ledger/bottle.md)。

## 1. 准备一次

先做完 [安装指南](setup.md) 第 0–5 步。以下命令都在仓库根目录的 **Bash** 终端里跑。

**每开一个新终端都要跑这三行**（环境变量不跨终端存活；没有第三行时任何 collect/eval
都会在起进程之前拒跑 `needs patches/sonic_hand_ownership.patch`）：

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 打过夹爪补丁的 deploy
```

**每台机器只建一次场景**：

```bash
SCENE_PRESET=bottle_only bash scripts/install_scene.sh          # 建 outputs/scene/sim
"$VENV_SIM_PY" "$LOCOMANI_SRC/sim/build_sim_scene.py" \
  "$LOCOMANI_ROOT/outputs/scene" "$LOCOMANI_ROOT/outputs/scene/sim_lowcom" \
  --preset bottle_only --harden-pads --object-com-drop 0.05     # 建 outputs/scene/sim_lowcom
```

两个场景**都要建**：`install_scene.sh` 只建前者，`bottle_simple_*.yaml` 用的是后者
（瓶子重心下移 5 cm）。装完核对：

```bash
"$VENV_SIM_PY" -c 'import json; print(json.load(open("outputs/scene/sim/scene_meta.json"))["target_pool"])'
```

目标池里要有 `water_bottle`，`install_scene.sh` 会回显 `finger pads HARDENED`。
用哪个场景由实验文件的 `scene.dir` 决定，两者在 `outputs/scene/` 下并存。

新机器还要跑一次 `bash scripts/link_gr00t_src.sh`——它改的是**训练 venv**
（`third_party/Isaac-GR00T/.venv`，不是 `$VENV_SIM_PY`），把 `gr00t` 指向 `src/gr00t/`；指错时训练拒跑。

## 2. 一次运行 = 一份实验文件

`configs/experiments/<name>.yaml` 是**完整**的：这次运行读的每一个值都在里面，没有继承也没有组合。
八个块——`name` + `task` / `embodiment` / `scene` / `backend` / `collect` / `train` / `eval`。
运行开始时它被**逐字节复制**到 `outputs/runs/<name>/config.yaml`，**下游阶段读的永远是那份副本**。

现有单臂实验：`bottle_simple_200`（200 集 + 10k 步 + 100 局）、`bottle_simple_gate8`
（8 局固定 case bank，闸门用）、`bottle_simple_smoke`（全链路冒烟）、`pick_place`。

## 3. 新建一次实验

复制一份改几行，就这样：

```bash
cp configs/experiments/bottle_simple_200.yaml configs/experiments/my_line.yaml
sed -i 's/^name: .*/name: my_line/' configs/experiments/my_line.yaml
$EDITOR configs/experiments/my_line.yaml     # 改 §4 表里那几个字段
diff configs/experiments/bottle_simple_200.yaml configs/experiments/my_line.yaml
```

顶层 `name` **不缩进**；缩进的那个 `name` 是 `embodiment.name`，改错了不报错，只会让运行
目录名不变而具身配置错掉。`diff` 出来的行就是两次运行的全部差别。

## 4. 每次都要改的字段

这里只说字段**控制什么**，不复述默认值。`task`/`backend`/`eval` 的默认值查
[配置字段表](reference/config_fields.md)；**`collect` 和 `train` 是普通 YAML、没有 dataclass，
那张表里查不到**，它们的唯一出处是实验文件自身的注释。

| 字段 | 控制什么 | 调大 / 调小意味着 |
|---|---|---|
| `name` | 运行目录名 | 同名重跑会接着采，换名字才是另起一份数据 |
| `collect.successes` | **保留**多少条成功示范 | 数据量。带固定 case bank 时它退化成事后验收门槛（见 §5） |
| `collect.batch_rounds` | 每批**尝试**几局 | 尝试预算。带 bank 时这才是真正决定跑几局的字段 |
| `collect.max_batches` | 最多几批 | 批数上限，采不够会明确报错而不是无限跑 |
| `collect.seed0` | 第一批的采样种子 | 换它就是换一组 case；批次号往后累加 |
| `collect.record` | 录不录 LeRobot 数据 | `false` 只采样打分不落数据，比录制快很多 |
| `scene.dir` | 用哪个场景 | `sim_lowcom` 是 `simple` 线的默认 |
| `scene.cases` | 固定 case bank 文件 | 非空就是固定对照，见 §5 |
| `train.steps` | 优化器步数 | 训练时长的主要来源 |
| `train.save_steps` | 存 checkpoint 的间隔 | 要冒烟就让它等于 `steps`，否则一个 checkpoint 都存不下 |
| `train.holdout` | 留几集做验证 | 必须**小于** `collect.successes`，否则训练集为空 |
| `eval.rounds` | 闭环评测跑几局 | 20 局只能看出"翻倍级"差异，100 局才判得动 10 个点 |
| `eval.seed` | 评测的采样种子 | 与采集的 seed 刻意错开 |
| `eval.policy_port` | policy server 的端口 | 默认 5590，**并行评测必须改**（见 §6） |
| `eval.initial_token` | 交接时 deploy 举着的姿势 | **必须和训练数据的开局姿势一致**，配错 100 局全卡在 `[stand]` |

## 5. 固定 case bank 的例外

实验文件给了 `scene.cases` 时，"成功够了就停"（`--until-ok`）**不会**传给 gym：整个
`batch_rounds` 预算必定跑满，`successes` 退化成事后的最低验收门槛。所以
`bottle_simple_gate8.yaml`（带 bank）把 `successes` 改成 1 **不会**只采一集——要改的是
`batch_rounds`。这是有意的：固定 bank 的对照要保留全部尝试，不能成功后重采样填满。

## 6. 跑

整条线一条命令，按 `task.plugin` 分派、按顺序跑：

```bash
python scripts/pipeline.py configs/experiments/my_line.yaml --gpu 1 --dry-run   # 先看会发生什么
python scripts/pipeline.py configs/experiments/my_line.yaml --gpu 1             # 真跑
```

也可以分阶段跑，方便在中间插手工检查：

```bash
python scripts/collect_bottle.py configs/experiments/my_line.yaml --gpu 0   # 采集
python scripts/convert.py        outputs/runs/my_line                       # 转换（不用卡）
python scripts/train.py          outputs/runs/my_line --gpu 0               # 训练
python scripts/eval_bottle.py    outputs/runs/my_line --gpu 1               # 闭环评测
```

**命令行只有 6 个参数**，其余一切都在实验文件里：

| 参数 | 含义 |
|---|---|
| `<experiment.yaml>` | `collect` 阶段的位置参数：从哪份实验文件开一次新运行 |
| `<run_dir>` | `convert`/`train`/`eval` 的位置参数：接着哪次运行往下做 |
| `--gpu N` | 用哪张卡。`convert` 不需要 |
| `--name X` | 运行目录改叫 X（副本里的 `name:` 仍是来源实验的） |
| `--from` / `--to` | 只有 `pipeline.py` 有：只跑 `collect`/`convert`/`train`/`eval` 中的一段 |
| `--dry-run` | 打印每个阶段将执行的命令和环境变量，**一个字节都不写** |

**产物已存在的阶段自动跳过**（`summary.json` 的 collect 条目 / `train_dataset/` / `model/` /
`eval/closed_loop.json`）。要重跑就**删掉它的产物**——整个续跑故事就这一条，`ls` 一眼看得见。

**并行是多开一个终端换一张卡**，没有作业文件也没有调度器——**但换卡不够，还要换两个端口**：

| 端口 | 谁的 | 怎么换 | 不换会怎样 |
|---|---|---|---|
| 29500 | torchrun（`train_bottle.sh`） | `export PORT=29540` | `EADDRINUSE`（端口 29500），train 阶段直接挂 |
| 5590 | policy server（eval） | 实验文件里改 `eval.policy_port` | `ZMQError: Address already in use`，eval 起不来 |

两个都**不是每卡一个**，是进程范围的 TCP 端口；采集阶段不受影响（它的端口按 instance 偏移）。

```bash
export PORT=29540                                                                # 终端 2 专用
python scripts/pipeline.py configs/experiments/bottle_simple_200.yaml  --gpu 0   # 终端 1
python scripts/pipeline.py configs/experiments/bottle_simple_gate8.yaml --gpu 1   # 终端 2
cat outputs/runs/*/summary.json                                  # 比较结果
```

采集内部还有批级续跑：`collect/inputs.json` 记录已完成的批次，同名重跑接着采。这**不是**
任意时刻的 episode 级恢复——先看最后一批的日志和 manifest 再继续。

## 7. 同一套脚本的另外两条线

`pick_place`（右手抓罐头放进篮子）和 `bottle_visual` 用**完全一样的 `collect_bottle.py` +
`eval_bottle.py`**，只差实验文件的 `task.plugin` / `task.goal` / `task.place` 和场景；命令原样适用。

## 8. 产物

路径相对于 `outputs/runs/<name>/`：

| 路径 | 内容 |
|---|---|
| `config.yaml` | 实验文件的逐字节副本，这次运行的 provenance |
| `summary.json` | 每阶段一条记录：保留数、产出率、失败家族、数据集、checkpoint、成功率 |
| `stack_env.json` | 仿真栈告诉子进程的**全部**环境变量 |
| `commands.jsonl`、`timings.jsonl` | 每个子进程的命令行与耗时 |
| `collect/manifest_NN.jsonl`、`collect/videos_NN/`、`collect/state_trace_NN/` | 逐局成败与原因、第三人称录像、物理轨迹 |
| `train_dataset/`、`val_dataset/` | LeRobot 数据集（`train.holdout` 为 0 时没有验证集） |
| `train_dataset/third_person/episode_NNNNNN.mp4` | 每集的第三人称录像，**按集号对齐**（硬链接，不占额外磁盘）。不是相机流，不进策略 |
| `model/checkpoint-<steps>/` | 权重 + 归一化 processor |
| `eval/closed_loop.json`、`eval/videos/`、`eval/eval_config.json` | 闭环结果、录像、**实际交给评测器的那份配置** |
| `eval/closed_loop.manifest.jsonl` | **逐局记录**——失败阶段与标签只在这里。契约共要求 9 个路径，用下面的命令查全 |

**读日志**：`collect_NN.log` 找 `EPISODE OK` 或带标签的失败原因；`train.log` 找 loss 和
`Copying experiment config directory`（存档时的实际字样）；`eval/serve.log` 里有 `listening`
才算 policy server 起来了。**光看退出码不算跑完**——要 `summary.json` 里有那一条、产物真的在那里：

```bash
python scripts/artifact_contract.py check outputs/runs/<name> bottle_eval
```

## 9. 排错

| 症状 | 原因与处理 |
|---|---|
| `needs patches/sonic_hand_ownership.patch` | 要打过补丁的 deploy。按 [安装第 3 步](setup.md) 重编，或改 `SONIC_DEPLOY_ROOT`。这个检查在**起任何进程之前**执行 |
| `not enough headroom -- cpu/pids/gpu ...` | 资源不足时脚本**拒绝启动并说明原因**。配额打满时仿真掉墙钟，专家的计时阶段会集体失败、看起来像机器人故障 |
| `<run>/config.yaml differs from ...` | 同名运行的实验文件被改过。换一个 `--name` |
| `model/ exists` / `closed_loop.json exists` | 阶段拒绝覆盖已有产物。检查后删掉再重跑 |
| 100 局全卡在 `[stand]` | `eval.initial_token` 和训练数据的开局姿势不一致（§4 最后一行） |
| `[streamed_arm] POSE switch unconfirmed` | deploy 回落到冻结的默认片段，**这一局机器人整局不动、数据作废**。重跑该局；实测 2026-09-18 十局里出现一次 |
| 栈起不来、端口被占 | 先看 §6 的两个端口；栈本身用 `python -m locomani.stack.cli status --instance N`，锁在 `$LOCOMANI_TMP/locomani_locks/` |
| 阶段"被跳过"但你想重跑 | 删掉它的产物（§6） |

手工起停栈用 `python -m locomani.stack.cli up|status|down --instance N`（正常流程不需要）。
**只有 instance 0 会跑导出器**，所以并行按 GPU 分而不是按实例分。
更细的失败形态见 [隐性失败](reference/silent_failures.md)。

## 10. 进一步阅读

- [单臂详解](single_arm_design.md)：专家设计、控制原则、被否掉的方案。
- [单臂台账](ledger/bottle.md)：逐轮实测数字与裁定。
- [配置字段表](reference/config_fields.md)：字段含义与默认值以它为准。
- [双臂](bimanual.md)：抓篮子那条线，结构与本页逐节对应。
- [MuJoCo 仿真与 Isaac 渲染](rendering.md)：离线重渲与视频导出。
