# 运行机制

三条主线（[bottle](single_arm.md)、[crate](bimanual.md)、[warehouse](warehouse.md)）共用同一套运行规则，都写在这一页。
runbook 只列命令；想改参数、并行跑、重跑某个阶段，或者弄清命令背后怎么运转时，读这一页。名词见[术语表](reference/glossary.md)。

## 1. 实验文件与运行目录

一次运行由一份实验文件决定：`configs/experiments/<名字>.yaml`。这次运行要读的配置全部写在文件里，没有继承，也不引用别的配置文件。

采集开始时，脚本把实验文件原样复制到 `outputs/runs/<名字>/config.yaml`。之后的转换、训练、评测都只读这份副本，所以运行途中修改实验文件，不影响正在进行的运行。
同名再开一次运行时，如果实验文件和已有副本不一致，脚本拒绝启动（`<run>/config.yaml differs from ...`），这时用 `--name` 换一个运行名。

新建实验就是复制一份，再改几行：

```bash
cp configs/experiments/bottle_simple_smoke.yaml configs/experiments/my_run.yaml
sed -i 's/^name: .*/name: my_run/' configs/experiments/my_run.yaml
$EDITOR configs/experiments/my_run.yaml
diff configs/experiments/bottle_simple_smoke.yaml configs/experiments/my_run.yaml
```

- 运行名是顶格的 `name`。缩进的 `name` 属于别的块（例如 `embodiment.name`），改错了不会报错。
- `diff` 列出的行就是两次运行的全部差别。

## 2. 常改的字段

这里只说字段控制什么。`task`、`backend`、`eval` 的默认值和完整说明查[配置字段表](reference/config_fields.md)；`collect` 和 `train` 没有对应的配置类，说明只在实验文件的注释里。

| 字段 | 控制什么 | 注意 |
|---|---|---|
| `collect.successes` | 保留多少条成功的示范 | 什么时候停采，见下一小节 |
| `collect.batch_rounds` / `max_batches` | 每批尝试几局 / 最多几批 | 批数用完还不够，采集以非零状态退出，已采的记录照样保留 |
| `collect.seed0` | 第一批的随机种子 | 换种子就是换一组开局；对照实验两边要用同一个 |
| `collect.record` | 是否录制训练数据 | `false` 只打分、留录像，不产出数据集，比录制快 |
| `collect.instance` | 用哪个实例（一组端口和一个 exporter 目录） | 同时跑的两个运行要用不同实例 |
| `scene.name` | 用哪个场景 | 名字在 `configs/scenes.yaml` 里查；`scene.dir` 直接给目录，两者只能给一个 |
| `images.source` | 策略看到的相机图由谁渲：`mujoco`、`isaac` 或 `unreal` | 训练集和闭环评测用同一个；区别见[图像来源](rendering.md#图像来源imagessource) |
| `dataset.fps` | 数据集的帧率：`25` 或 `50`。转换器从 50 Hz 的原始录制里每 2 行或每 1 行取一行 | 没写这个块的文件按 25。补渲只渲取到的行，所以 25 渲过的 session 不能直接转 50：删掉 session 的 `meta/offline_render.json` 再补渲一遍；50 渲过的可以转 25。闭环评测目前只接受 25，50 的 checkpoint 走 [真机部署客户端](../src/real_robot/README.md) |
| `train.steps` / `save_steps` | 训练步数 / 每隔多少步存一个 checkpoint | `save_steps` 大于 `steps` 时一个 checkpoint 都存不下 |
| `train.holdout` | 留几条做验证集 | 必须小于保留条数，否则训练集为空；设 0 就没有验证集，也就不能做开环评测 |
| `eval.rounds` / `eval.seed` | 闭环评测跑几局 / 评测的随机种子 | 20 局只能分辨成倍的差别，100 局才能分辨 10 个百分点 |
| `eval.policy_port` | 策略服务的端口 | 两个评测同时跑时必须不同，且不能落在实例端口范围里（第 6 节） |
| `eval.initial_token` | 策略接管那一刻 deploy 保持的姿势 | 必须和训练数据的开局姿势一致；不一致时每局都卡在开局 |
| `eval.client` / `latency_frames` | 评测的时钟和模拟的推理延迟 | 见第 8 节 |

### 采集什么时候停

- bottle：保留数一到 `successes` 就停，不必跑完当前这批。
- bottle 用了固定开局库（`scene.cases`）时：每批一定跑满 `batch_rounds`，`successes` 只在最后检查是否达标。固定开局库用于对照实验，所以要保留全部尝试。
- crate 和 warehouse：每批一定跑满 `batch_rounds`；一批结束后，保留数够了就不再开新批。总尝试局数是 `batch_rounds` 乘以实际跑的批数。
- 要正好跑 N 局：设 `batch_rounds: N`、`max_batches: 1`。

## 3. 命令行参数

其余一切都在实验文件里，命令行只有下面这些参数：

| 参数 | 含义 |
|---|---|
| `<实验文件>` | 采集的参数（`start_stack.sh collect`、`run_agent_server.sh`）：用这份文件开一次新运行 |
| `<运行目录>` | 采集之后各步的参数：接着这次运行往下做 |
| `--gpu N` | 用哪张卡。转换不用卡 |
| `--name X` | 运行目录改叫 X；副本里的 `name` 仍是原来的 |
| `--dry-run` | 只打印将要执行的命令和环境变量，不写任何文件。建场景、起栈、采集、转换、训练、评测的脚本都支持；`run_overnight.sh`、`open_loop_eval.sh` 和出录像的脚本不支持 |
| `--until-kept N` | 只有 `bimanual/run_agent_server.sh` 有：保留到 N 条后，在当前这批结束时暂停；去掉它再跑同一条命令就接着采 |
| `--line bimanual` | 只有 `run_overnight.sh` 有：用双臂的专家和评测（crate、warehouse）；不给就是 bottle |
| `--backbone-path DIR` | 只有 `run_vla_server.sh` 和 `open_loop_eval.sh` 有：在别的机器（例如集群）上训练的 checkpoint，记录的视觉语言骨干路径在本机不存在，用它指向本机的副本，例如 `~/Checkpoints/Cosmos-Reason2-2B` |
| `--render-gpu N` | 只有 `run_overnight.sh` 有：`images.source` 是 `isaac` 或 `unreal` 时，补渲和镜像用哪张卡（Unreal 是它的编辑器那张卡）。旧名 `--isaac-gpu` 照收 |
| `--from` / `--to` | 只有 `run_overnight.sh` 有：只跑 `collect`、`convert`、`train`、`eval` 中的一段 |

## 4. 跳过、续跑与重跑

- `run_overnight.sh` 遇到产物已经存在的阶段就跳过：采集看 `summary.json` 里 collect 条目的保留数够不够，转换看 `train_dataset/`，训练看 `model/`，评测看 `eval/closed_loop.json`。
- 单独运行各步的脚本时，已有产物的阶段拒绝覆盖（`model/ exists`、`closed_loop.json exists`）。要重跑某个阶段，先删掉它的产物。
- 采集可以中断后接着跑：`collect/inputs.json` 只记录跑完的批，中断的那批整批重来，半截的 manifest 改名为 `interrupted_manifest_NN_*.jsonl`。
- 重新评测同一个运行：先把旧结果挪开，例如 `mv outputs/runs/my_run/eval outputs/runs/my_run/eval_old`。
- 评测读 `summary.json` 里 train 条目记下的 checkpoint。训练中途停了，或者想评别的 checkpoint，用下面这条命令手工写入（例子是停在 5000 步）：

```bash
"$VENV_SIM_PY" -c "from pathlib import Path; from locomani.pipeline.run_context import summarise; r = Path('outputs/runs/my_run').resolve(); summarise(r, 'train', checkpoint=str(r / 'model/checkpoint-5000'), steps=5000)"
```

## 5. 仿真栈

- 采集和闭环评测都要先起仿真栈：MuJoCo 仿真进程（sim）、SONIC 控制器（deploy），录制时还有数据录制进程（exporter）。
- `start_stack.sh` 起好栈就返回，栈留在后台。跑局的脚本（`run_agent_server.sh`、`run_vla_inference.sh`）结束时会关掉它。
- 跑局的脚本被强行中断、栈还留着时，手工关掉：`bash scripts/stop_stack.sh --instance 0`（只关这一个实例）。查状态用 `python -m locomani.stack.cli status --instance 0`，实例锁在 `$LOCOMANI_TMP/locomani_locks/`。
- 每次起栈，`sim.log` 开头都有一行 `Robot has fallen, height: 0.199 m`。这是 deploy 接管之前机器人没人控制、倒在地上，第一局重置时会站起来，录像里看不到。
- deploy 第一次启动会构建 TensorRT engine，大约 10 分钟，之后用缓存。
- 关栈后至少等 20 秒再起下一个。间隔太短时 deploy 收不到机器人状态，开局站不稳（`initial_stance_unstable`）。

## 6. 并行

**双臂线（crate、warehouse）的采集默认并行。** 一条“通道”（lane）是一个仿真栈加一个跑局进程，每条通道占一个实例。`start_stack.sh collect` 先量这台机器现在的空闲资源，据此决定开几条通道、各放在哪张卡和哪个实例上，然后逐个起栈，结果写进 `<运行>/collect/lanes.json`。起栈时和 `run_agent_server.sh` 开始第一局前，都会打印开了几条通道：

```
collect: 2 lanes in parallel -- one stack and one driver process each (instance 2 on GPU 0, instance 3 on GPU 1)
  what fits: cpu 2 (10.8 cores free / 4 per lane); pids 2 (1594 free / 760 per lane); gpu 8 (...); instances 8 (...)
```

通道数取下面几项里最小的一个（`src/locomani/pipeline/lanes.py`）：

| 资源 | 一条通道按多少算 | 依据 |
|---|---|---|
| CPU（容器配额 15.36 核，减去正在用的） | 录制且不渲染 4 核；实时渲染 10 核 | 2026-09-28 实测一条 warehouse 通道平均 +2.6 核，90 分位 +3.7 核 |
| 进程数（上限 3840，留 200） | 录制：200 + 140 × `collect.batch_rounds`；不录制：400 | exporter 每录一局增加约 130 个线程，每批重启后清零；每批 4 局约 760，每批 10 局约 1600 |
| 每张卡的空闲显存 | 录制且不渲染 2000 MiB；实时渲染 6000 MiB | 实测一条通道约 1.5 GB |
| 每张卡的仿真数 | 不渲染（`backend.render: offline`）最多 4 个；实时渲染 1 个 | 两个实时渲染的仿真放在同一张卡上，会让 deploy 丢状态 |
| 空闲实例 | 实例 0–9 中没有栈、没加锁、端口没被占的 | 别的程序也会占端口，比如 Unreal 编辑器占了实例 0 的整段端口 |
| 系统内存 | **不算**（规划器不量它） | 一条录制通道常驻约 5 GB：仿真进程 2.9 GB、exporter 1–1.4 GB、deploy 0.3 GB（2026-09-29 SRA0889 实测）。容器里 CPU 配额先把通道数压下来，所以够用；没有 cgroup 配额的桌面机上 CPU 一行不限制，只剩显存在算，32 GB 内存、swap 2 GB 的机器被开出 4 条，第一局刚开始驱动进程就被内核 OOM 杀掉，终端只打印一个 `Killed`。这样的机器写 `--lanes 1`，起栈前 `free -h` 的可用内存要有“通道数 × 5 GB” |

- 默认用机器上所有的卡；`--gpu 0` 或 `--gpu 0,1` 只用这些卡，通道轮流分到每张卡上。第一条通道优先放在 `collect.instance` 上。
- 只开一条通道：`start_stack.sh collect ... --lanes 1`，或在实验文件里写 `collect.lanes: 1`。写数字 N 表示“最多 N 条”。只开一条时和以前完全一样：栈只起在 `collect.instance` 上，这个实例被占就报错。
- 所有通道共用同一个运行目录。第 k 批的种子仍是 `seed0 + 100 × k`，不管哪条通道去跑。已保留的条数够了就不再开新批，但已经在跑的批会跑完，所以最后保留的条数可能多于 `collect.successes`。中断后重跑同一条命令，中断时所有没跑完的批会从头再跑。
- 单臂线（bottle）还是一条通道，`--gpu` 只取第一张卡。`run_overnight.sh` 的双臂线采集默认用所有卡（`--collect-gpu`、`--lanes` 可以改）。场景扫参的每个格子固定一条通道，因为扫参自己按实例开了多个循环。
- 手工关栈时，要把 `lanes.json` 里的每个实例都关掉：`bash scripts/stop_stack.sh --instance <实例>`。

**别的并行方式**：再开一个终端，换一张卡、换一个实例（`collect.instance`），跑另一个实验。下面两个端口是整台机器共享的，这时要改：

| 端口 | 谁用 | 怎么改 | 不改会怎样 |
|---|---|---|---|
| 29500 | 训练（`train.sh` 里的 torchrun） | 在第二个终端里 `export PORT=29540` | 训练报 `EADDRINUSE` 退出 |
| `eval.policy_port` | 策略服务 | 改实验文件的 `eval.policy_port` | 策略服务报 `ZMQError: Address already in use` |

另外，5555–5672 是实例 0–9 的仿真端口（每个实例占 10 个号），策略服务的端口不能落在这个范围里。三条主线的实验文件用 5700（crate、warehouse）和 5710（bottle），同时评测 crate 和 warehouse 时把其中一个改掉。

资源不够时脚本拒绝启动并说明原因（`not enough headroom`）。CPU 被占满时仿真变慢，专家的定时动作会成批失败，看起来像机器人出了问题。

## 7. 查看一次运行留下的记录

路径相对于 `outputs/runs/<名字>/`：

| 文件 | 内容 |
|---|---|
| `config.yaml` | 实验文件的原样副本 |
| `summary.json` | 每个阶段一条：保留数、成功率、失败类别、数据集和 checkpoint 的位置 |
| `stack_env.json` | 仿真栈传给各进程的全部环境变量 |
| `commands.jsonl`、`timings.jsonl` | 每个子进程的命令行和耗时 |
| `deploy_receipts.jsonl` | 每次启动或重启 deploy 时记下的工作目录、可执行文件和 checkpoint 的哈希 |
| `eval/eval_config.json` | 实际交给评测程序的配置 |

- 比较多次运行：`cat outputs/runs/*/summary.json`。
- 检查某个阶段的产物是否齐全：`python scripts/artifact_contract.py check outputs/runs/<名字> bottle_eval`。bottle 用 `bottle_collect`、`bottle_convert`、`bottle_train`、`bottle_eval`，crate 和 warehouse 用 `crate_collect`、`crate_eval`。warehouse 评测时不录像，它的录像事后渲到 `eval/videos_isaac/` 或 `eval/videos_unreal/`，所以 `crate_eval` 里 `eval/videos` 一项对 warehouse 总是报缺，其余各项照常检查。
- 光看退出码不能说明跑完了，要看 `summary.json` 里有没有这一阶段的条目、产物在不在。

## 8. 评测的时钟

- 默认是 lockstep（`eval.client: lockstep`）：拍照、渲染和推理的时候仿真暂停，算完再继续。推理延迟按 `latency_frames` 帧模拟（默认 2 帧），这期间机器人继续执行上一个动作块。这样渲染快慢不会影响成功率。
- 另外两种是按墙钟跑的 `blocking`（crate、warehouse）和 `async`（bottle），只在复现 2026-09-28 之前的结果时使用。
- warehouse 上已经校准过：同一个 checkpoint，墙钟 81/100，lockstep 复刻墙钟的延迟 83/100，默认设置 88/100。bottle 的 lockstep 还没有和墙钟对照过。
- 细节见[双臂详解的时间模型一节](bimanual_design.md#评测的时间模型墙钟与-lockstep)。

## 9. 开环评测

`scripts/open_loop_eval.sh` 在验证集上逐帧比较策略预测的动作和示范动作，不起仿真。它不属于流水线，参数都要写名字：

- `--model-path` 是 checkpoint 目录，`--dataset-path` 是 `val_dataset/`，`--out-dir` 是输出目录；两个路径给反了，会在加载模型之前报错。
- 输出 `metrics.json`（每条轨迹和平均的 MSE、MAE）、`plots/`（每条轨迹一张图）和 `run.log`。
- 在集群上训练的 checkpoint 要加 `--backbone-path ~/Checkpoints/Cosmos-Reason2-2B`，闭环评测的策略服务 `run_vla_server.sh` 也一样。

## 10. 其他入口

- 不起仿真，查看 bottle 专家实际用的选项：`python3 -m locomani.tasks.bottle.cli_args --task-config configs/task/bottle_visible.yaml`。
- 比较两组实验的行为：`python3 scripts/gate.py --help`，两边要用相同的评测设置。
- 不经过流水线、手工起栈跑一局 bottle 或 pick-and-place：见[单臂详解的手工一局](single_arm_design.md#手工起栈跑一局)。
- ScaleBFM 控制器的采集和评测是另一套入口，见 [ScaleBFM](scalebfm.md)。
