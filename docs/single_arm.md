# bottle：单臂抓瓶

> **已归档（2026-10-20）**：VLA 数据采集与训练这一系列暂不使用，这一页不再随代码更新，命令可能已经跑不通。现在的主线是[场景 RSI](rsi.md)。

机器人走到桌前，停稳后单手抓起桌上的水瓶。整条线只用 1 张 GPU。
下面按冒烟规模写（2 条数据、10 步训练、2 局评测），从采集到评测跑完大约 12 分钟。冒烟只检查每一步能不能跑通，训练 10 步的模型不会抓，闭环 0/2 是正常结果。耗时是 2026-09-28 照这一页实跑的结果。
改参数、并行、重跑某一步见[运行机制](running.md)；设计和更多细节见 [bottle 详解](single_arm_design.md)；名词见[术语表](reference/glossary.md)。

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
```

做什么：设置路径变量，把 `src/` 加进 `PYTHONPATH`，并读入本机的 `locomani.local.sh`。下面的命令都在仓库根目录执行。
前提：[安装](setup.md)第 0–5 步已经完成。

## 1. 建场景（每台机器一次）

```bash
bash scripts/build_scene.sh bottle
```

做什么：重建 `outputs/scene/sim`（桌上只有一个水瓶），再在它的基础上建 `outputs/scene/sim_lowcom`（瓶子重心下移 5 cm），bottle 的实验用后者。
得到什么：两个目录，各有 `scene_locomani.xml` 和 `scene_meta.json`；打印的内容里有 `finger pads HARDENED`。几秒钟。

## 2. 采集

```bash
bash scripts/start_stack.sh collect configs/experiments/bottle_simple_smoke.yaml --gpu 0
bash scripts/single_arm/run_agent_server.sh configs/experiments/bottle_simple_smoke.yaml --gpu 0
```

做什么：第一条起仿真栈（MuJoCo 仿真、SONIC 控制器、数据录制），起好就返回，栈留在后台。第二条运行专家：一局接一局让机器人走近、抓瓶，保留成功的局；保留够 2 条后结束，并关掉仿真栈。
得到什么：运行目录 `outputs/runs/bottle_simple_smoke/`，里面有实验文件的副本 `config.yaml`、每局成败 `collect/manifest_00.jsonl`、每局录像 `collect/videos/`。起栈约 45 秒（deploy 第一次启动还要多约 10 分钟构建 TensorRT engine）；采集约 1.5 分钟，日志 `collect_00.log` 里每局以 `EPISODE OK` 结尾；开局站起时骨盆离出生点超过 0.35 m（`stand drift` 行）的局记成 `[stand] walked … off the spawn`，不算专家失败。

## 3. 转换

```bash
bash scripts/convert_data.sh outputs/runs/bottle_simple_smoke
```

做什么：把录制的数据转成 LeRobot 数据集，按 `train.holdout` 留出验证集，并检查数据。不用 GPU。
得到什么：`train_dataset/`（1 条）和 `val_dataset/`（1 条）。约 30 秒。

## 4. 训练

```bash
bash scripts/train.sh outputs/runs/bottle_simple_smoke --gpu 0
```

做什么：用训练集微调 GR00T。
得到什么：`model/checkpoint-10/`，日志在 `train.log`。约 1 分钟；`model/` 占约 24 GB 磁盘。

## 5. 评测

开环评测（可选）：在验证集上逐帧比较模型预测的动作和示范动作，不起仿真。

```bash
bash scripts/open_loop_eval.sh --model-path outputs/runs/bottle_simple_smoke/model/checkpoint-10 \
  --dataset-path outputs/runs/bottle_simple_smoke/val_dataset --out-dir outputs/runs/bottle_simple_smoke/eval/open_loop
```

得到什么：`eval/open_loop/metrics.json`（每条轨迹的 MSE、MAE）和 `plots/`。约 30 秒；冒烟模型的 MSE 约 0.04。

闭环评测：让策略在仿真里控制机器人。需要两个终端，按顺序启动：

| 终端 | 命令 | 做什么 | 怎么算好了 |
|---|---|---|---|
| 1 | `bash scripts/start_stack.sh eval outputs/runs/bottle_simple_smoke --gpu 0` | 起评测用的仿真栈（lockstep 时钟，不录制） | 命令返回 |
| 1 | `bash scripts/run_vla_server.sh outputs/runs/bottle_simple_smoke --gpu 0` | 策略服务：加载 checkpoint，收观测、回动作 | 打印 `listening`；一直运行，评测结束后按 Ctrl-C |
| 2 | `bash scripts/single_arm/run_vla_inference.sh outputs/runs/bottle_simple_smoke --gpu 0` | 逐局重置场景、让策略控制机器人、记录成败 | 跑完 `eval.rounds` 局后关掉仿真栈并退出 |

得到什么：`eval/closed_loop.json`（冒烟为 0/2）、逐局记录 `eval/closed_loop.manifest.jsonl`、录像 `eval/videos/`。起栈和策略服务各约 25 秒；两局约 7 分钟。`eval.log` 末尾打印 `closed loop: 0/2`，每局的失败原因写在 `why` 里，例如 `[vla] walk: never closer than 1.07 m`。

熟悉以后可以一条命令跑完第 2–5 步（不含开环评测）：`bash scripts/run_overnight.sh configs/experiments/bottle_simple_smoke.yaml --gpu 0`。已经有产物的步骤会被跳过。

## 6. 看结果

路径相对于 `outputs/runs/<名字>/`：

| 看什么 | 在哪里 |
|---|---|
| 每一步的汇总（保留数、成功率、checkpoint） | `summary.json` |
| 采集每局的成败和原因 | `collect/manifest_NN.jsonl`；日志 `collect_NN.log` 里每局以 `EPISODE OK` 或失败原因结尾 |
| 采集录像 | `collect/videos/` |
| 闭环每局的成败、失败阶段 | `eval/closed_loop.manifest.jsonl`；录像在 `eval/videos/` |
| 产物是否齐全 | `python3 scripts/artifact_contract.py check outputs/runs/<名字> bottle_eval`（采集、转换阶段的契约名见 `tests/contracts/`） |

## 7. 常见报错

| 报错 | 处理 |
|---|---|
| `not enough headroom` | 机器上的 CPU、进程数或显存不够，脚本拒绝启动。等别的任务结束，或换一张卡（[运行机制](running.md#6-并行)） |
| `config.yaml differs from` | 同名运行的实验文件被改过。加 `--name` 换个运行名 |
| `model/ exists`、`closed_loop.json exists` | 这一步已经有产物，脚本不覆盖。确认不要了就删掉再跑 |
| `needs patches/sonic_hand_ownership.patch` | deploy 没打补丁，回到[安装第 3 步](setup.md#3-编译-deploy) |
| 闭环每局都卡在 `[stand]` | `eval.initial_token` 和训练数据的开局姿势不一致 |

更多报错见 [bottle 详解第九节](single_arm_design.md#九运行细节2026-09-28-从-bottle-runbook-移来)。

## 8. 接下来

- 完整规模：把上面命令里的 `bottle_simple_smoke` 换成 `bottle_simple_200`（200 条数据、1 万步训练、100 局评测）。采集约 300 局保留 200 条，一局约 45 秒；100 局评测约 6 小时。
- 改参数：复制一份实验文件再改，常改的字段见[运行机制第 2 节](running.md#2-常改的字段)。
- 同一套命令也能跑 pick-and-place（`pick_place`，右手抓罐头放进篮子），只换实验文件和场景，见 [bottle 详解第四节](single_arm_design.md#四pick-and-place-这条变体)。
- 为什么这样设计、试过什么：[bottle 详解](single_arm_design.md)；每一轮的实测数字：[bottle 台账](ledger/bottle.md)。
