# 双臂：从采集到评测

视觉指导走近、俯身抓篮、抬起并站直。**这一页从头到尾能跑完整条线**，不需要跳到别的文档。
为什么这么设计、试过什么被否掉了，在 [双臂详解](bimanual_design.md)；
哪天跑出什么数，在 [双臂台账](ledger/crate.md)。

和单臂用**同一套扁平脚本**，只有采集和评测分线（`run_collect_bimanual.py` / `run_eval_bimanual.py`），
转换和训练完全共用。结构与 [单臂](single_arm.md) 逐节对应。

## 1. 准备一次

先做完 [安装指南](setup.md) 第 0–5 步，并应用
[POSE ownership 补丁](../patches/sonic_pose_ownership.md)。

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"

bash scripts/install_scene.sh
"$VENV_SIM_PY" -m locomani.tasks.crate.bimanual_scene \
  outputs/scene/crate_bimanual_depth04 --crate-setback 0.04
"$VENV_SIM_PY" scripts/experiments/crate_table_height_20260913.py \
  outputs/scene/crate_bimanual_depth04 outputs/scene/crate_table30_depth04 --height 0.30
```

**三步都要跑。** 降桌高和选控制 profile 是两件事，低桌配置文件**不改变桌面几何**。
`--crate-setback 0.04` 同时改 XML 初始位置和 reset 元数据，**任务 JSON 不能代替它**——
此前本页漏写这 4 cm，生成的场景与实验对不上（[复现核对](handoff/crate-reproduction-20260914.md)）。

构建结果包含 `scene_locomani.xml` 和 `scene_meta.json`。
新机器还要跑一次 `bash scripts/link_gr00t_src.sh`（venv 的 `gr00t` 指向 `src/gr00t/`）。

## 2. 一次运行 = 一份实验文件

`configs/experiments/<name>.yaml` 是**完整**的：这次运行读的每一个值都在里面，没有继承也
没有组合。八个块——`name` + `task` / `embodiment` / `scene` / `backend` / `collect` /
`train` / `eval`。运行开始时它被**逐字节复制**到 `outputs/runs/<name>/config.yaml`，
**下游阶段读的永远是那份副本**：跑到一半改实验文件不影响正在跑的运行。

现有双臂实验：`crate_table30_record`（录制线，20 条成功 / 每批 10 局 / 最多 2 批）。

## 3. 新建一次实验

```bash
cp configs/experiments/crate_table30_record.yaml configs/experiments/my_crate.yaml
sed -i 's/^name: .*/name: my_crate/' configs/experiments/my_crate.yaml
$EDITOR configs/experiments/my_crate.yaml     # 改 §4 表里那几个字段
diff configs/experiments/crate_table30_record.yaml configs/experiments/my_crate.yaml
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
| `collect.batch_rounds` | 每批**尝试**几局 | **这条线上真正决定跑几局的字段**，见 §5 |
| `collect.max_batches` | 最多几批 | 每批重启一次栈；默认整个 visual pilot 用一批 |
| `collect.successes` | 保留多少条成功 | 在这条线上**只是事后的验收门槛**，见 §5 |
| `collect.seed0` | 第一批的采样种子 | 换它就是换一组 case |
| `collect.record` | 录不录 LeRobot 数据 | `false` 只留视频/相机/动作记录，不进转换器 |
| `collect.instance` | 端口块与 exporter 目录 | 双臂不强制实例 0，可与单臂并行（见 §6） |
| `task.expert` | 用哪个专家 | `visual` 是本页这条线；`wall` / `carry` 是诊断专家 |
| `scene.dir` | 用哪个场景 | 就是 §1 建的 `outputs/scene/crate_table30_depth04` |
| `train.steps` / `save_steps` | 优化器步数 / 存档间隔 | 要冒烟就让 `save_steps` 等于 `steps`，否则存不下 checkpoint |
| `train.holdout` | 留几集做验证 | 必须**小于**实际保留的集数 |
| `eval.rounds` | 闭环评测跑几局 | 20 局只看得出"翻倍级"差异，100 局才判得动 10 个点 |
| `eval.seed` | 评测的采样种子 | 与采集的 seed 刻意错开 |
| `eval.policy_port` | policy server 的端口 | 默认 5590，**并行评测必须改**（见 §6） |

## 5. 双臂没有"成功够了就停"

分两层，容易只记住一层：

- **批内**：`batch_rounds` 局一定跑满。双臂专家只接 `--rounds`（`crate/prepare.py:expert_argv`），
  `--until-ok` 根本不传给它——**除非专家崩了**，异常会向上抛并中止整批。
- **批间**：`successes` 是**活判据**。`run_collect_bimanual.py:134` 的循环条件是
  `kept < successes and batches < max_batches`，够了就不再开新批（单臂同一行，这不是双臂特性）。

所以尝试数是 `batch_rounds ×（实际跑了几批）`，不是 `× max_batches`。**要保证跑满 N 批，
`successes` 必须大于 `batch_rounds × N` 能采到的数量**；反过来把 `successes` 调小只会少跑批。
预算用尽仍不够时采集阶段非零退出，但失败记录和视频**仍然保留**。

## 6. 跑

整条线一条命令跑完，或者一个阶段一个终端，方便中间插手工检查：

```bash
bash scripts/run_overnight.sh configs/experiments/my_crate.yaml --gpu 1 --line bimanual   # 整条线，无人值守
# 采集两个终端；起栈脚本两条线共用，按 task.plugin 冻结各自的场景
bash scripts/collect_data.sh               configs/experiments/my_crate.yaml --gpu 1   # 纸箱同线：handoff/box-via-basket-line-20260922.md
bash scripts/bimanual/run_agent_server.sh  configs/experiments/my_crate.yaml --gpu 1
bash scripts/convert_data.sh outputs/runs/my_crate              # 转换（不用卡）
bash scripts/train.sh        outputs/runs/my_crate --gpu 1      # 训练
# 闭环评测：三个终端，和单臂同构；这条线每批重起的是 deploy，不是整个栈
bash scripts/start_stack.sh                 outputs/runs/my_crate --gpu 1
bash scripts/run_vla_server.sh              outputs/runs/my_crate --gpu 1
bash scripts/bimanual/run_vla_inference.sh  outputs/runs/my_crate --gpu 1
```

> **`eval` 块整份交给评测器，但这条线只实现了其中一部分。** 它读 `rounds` / `seed` /
> `policy_port` / `execution_horizon` / `budget_s` / `initial_token`；`client` 只支持
> `blocking`，写别的会**在启动前报错**而不是静默跑成别的；其余字段落盘到
> `eval/ignored_eval_fields.json`，可以直接比对"声明了什么"和"用上了什么"。
也可以分阶段跑，方便在中间插手工检查：

**命令行只有 6 个参数**，其余一切都在实验文件里：

| 参数 | 含义 |
|---|---|
| `<experiment.yaml>` | `collect` 阶段的位置参数：从哪份实验文件开一次新运行 |
| `<run_dir>` | `convert`/`train`/`eval` 的位置参数：接着哪次运行往下做 |
| `--gpu N` | 用哪张卡。`convert` 不需要 |
| `--name X` | 运行目录改叫 X（副本里的 `name:` 仍是来源实验的） |
| `--from` / `--to` | 只有 `run_overnight.sh` 有：只跑 `collect`/`convert`/`train`/`eval` 中的一段 |
| `--dry-run` | 打印每个阶段将执行的命令和环境变量，**一个字节都不写** |

**产物已存在的阶段自动跳过**（`summary.json` 的 collect 条目 / `train_dataset/` / `model/` /
`eval/closed_loop.json`）。要重跑就**删掉它的产物**。同名重跑从首个未完成 batch 继续
（读 `collect/inputs.json`），**不是** episode 级续采——先检查未完成 batch 的记录。

**并行**：双臂不强制实例 0，所以可以 GPU 1 / 实例 1 录制，与单臂（GPU 0 / 实例 0）同时跑。
**同一块 GPU 上不能跑两个仿真。** 容器 pid 上限 3840，开销几乎全在 exporter：实测 sim + deploy 仅
**176** 个 pid（2269 → 2441），所以不录制的运行按 400 准入、录制的按 1800——**不要并行第三个录制栈**。
CPU 同理：起栈要 8 核余量，驱动器接到已起的栈上只再要 2 核（栈的 ~5 核已经在用量里；2026-09-22 之前它也被算 8 核，别的会话占 3 核就接不上）。

**换卡还不够，另外两个端口是进程范围的**（采集不受影响，它按 instance 偏移）：

| 端口 | 谁的 | 怎么换 | 不换会怎样 |
|---|---|---|---|
| 29500 | torchrun（`train.sh`） | `export PORT=29540` | `EADDRINUSE`，train 阶段直接挂 |
| 5590 | policy server（eval） | 实验文件里改 `eval.policy_port` | `ZMQError: Address already in use` |

`collect.record=true`（默认）时采集阶段在本实例上起 data exporter，每批一个 session，
写入 `$LOCOMANI_TMP/p03_i<实例>/outputs/<时间戳>/`，并记在
`<run>/collect/inputs.json` 的 `sessions`，与 `manifests` 同序一一对应。

## 7. 产物

路径相对于 `outputs/runs/<name>/`。**采集产物的落点与单臂不同**：

| 路径 | 内容 |
|---|---|
| `config.yaml` | 实验文件的逐字节副本 |
| `summary.json` | 每阶段一条记录：保留数、产出率、失败家族、数据集、checkpoint、成功率 |
|  `manifest_NN.jsonl`、`videos_NN/` | 逐局成败与原因、录像。**在顶层**，单臂在 `collect/` 之下 |
| `trace/` | 状态轨迹。**全批共用一个目录**，单臂按批分目录 |
| `collect/inputs.json` | 批次进度、exporter session 与 manifest 的对应 |
| `train_dataset/`、`val_dataset/` | LeRobot 数据集 |
| `train_dataset/third_person/episode_NNNNNN.mp4` | 每集的第三人称录像，**按集号对齐**（硬链接，不占额外磁盘）。不是相机流，不进策略 |
| `model/checkpoint-<steps>/` | 权重 + 归一化 processor |
| `eval/closed_loop.json`、`eval/videos_NN/` | 闭环结果与录像 |
| `eval/closed_loop.manifest.jsonl` | **逐局记录**——失败阶段与标签只在这里 |
| `eval/ignored_eval_fields.json` | 这次评测**没有实现**的 `eval` 字段及其值 |

manifest 行带 `task`、`produced`、`ok`，成功局还带
`execution.metrics.training_tail.end_seq`（进入直立持物验收的序号）——
转换器按它裁掉验收尾段。训练用 `configs/gr00t_sonic_3cam_bimanual.py`
（60 维状态、66 维动作、三路相机）。

**读日志**：`visual_NN.log`（日志按 `task.expert` 命名，不是 `collect_NN`）找逐局成败；`train.log` 找 loss 和 `Copying experiment config directory`；`eval/serve.log` 里有 `listening` 才算 policy server 起来了。**光看退出码不算跑完**。产物齐不齐有专门的网：
`eval/serve.log` 里有 `listening` 才算 policy server 起来了。**光看退出码不算跑完**。
产物齐不齐有专门的网：

```bash
python scripts/artifact_contract.py check outputs/runs/<name> crate_eval
```

## 8. 排错

| 症状 | 原因与处理 |
|---|---|
| 场景与别人的实验对不上 | §1 那 4 cm `--crate-setback`，或桌高那一步没跑 |
| `not enough headroom -- cpu/pids/gpu ...` | 资源不足时脚本**拒绝启动并说明原因**。配额打满时仿真掉墙钟，专家的计时阶段会集体失败、看起来像机器人故障 |
| 第三个录制栈起不来 / 段错误 | 容器 pid 上限。每个录制栈按 1800 预算，开销几乎全在 exporter（§6） |
| `<run>/config.yaml differs from ...` | 同名运行的实验文件被改过。换一个 `--name` |
| 采集非零退出但视频还在 | `successes` 门槛没达到（§5）。记录是完整的 |
| 评测开局动作突变 | 已修（2026-09-18）：回合之间不再交还 planner，settle 跑在 token 通道上。此前底盘在 settle 里漂中位 0.09 m、最大 0.83 m。要复现旧行为加 `--no-clean` |
| `[streamed_arm] POSE switch unconfirmed` | deploy 回落到冻结的默认片段，**这一局机器人整局不动、数据作废**。重跑该局 |
| 一整批只跑了几局就结束 | 专家抛异常会中止整批（§5）。在 `visual_NN.log` 里搜 `Traceback` |
| 栈起不来、端口被占 | `python -m locomani.stack.cli status --instance N`；锁在 `$LOCOMANI_TMP/locomani_locks/` |

`check_dataset` 一定要带 `--run-dir`：它让审计从本次运行的 `trace/` 按 exporter 的
`priv_seq` 取箱子的**世界**高度来判"闭合后抬升"。没有它，检查器只能用骨盆坐标系的位姿列，
而躯干直立时这一列会把 20 cm 的真实抬升读成几厘米（2026-09-14 验证批：世界抬升
7/7 通过，骨盆系只判出 2/7）。更细的失败形态见 [隐性失败](reference/silent_failures.md)。

## 9. 进一步阅读

- [双臂详解](bimanual_design.md)：视觉专家的信息来源、连续控制、观测/动作协议、诊断专家。
- [双臂台账](ledger/crate.md)：逐轮实测数字与裁定。
- [配置字段表](reference/config_fields.md)：字段含义与默认值以它为准。
- [单臂](single_arm.md)：抓瓶那条线，结构与本页逐节对应。
- [MuJoCo 仿真与 Isaac 渲染](rendering.md)：并排视频与离线重渲。
