# crate：双臂抓筐

机器人看着筐走到矮桌前，俯身用双手抓起筐，再抬起、站直。整条线只用 1 张 GPU。
下面按冒烟规模写（2 条数据、10 步训练、2 局评测），从采集到评测跑完大约 8 分钟。冒烟只检查每一步能不能跑通，训练 10 步的模型不会抓，闭环 0/2 是正常结果。耗时是 2026-09-28 照这一页实跑的结果。
改参数、并行、重跑某一步见[运行机制](running.md)；设计和更多细节见 [crate 详解](bimanual_design.md)；名词见[术语表](reference/glossary.md)。

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
```

做什么：设置路径变量，把 `src/` 加进 `PYTHONPATH`，并读入本机的 `locomani.local.sh`。下面的命令都在仓库根目录执行。
前提：[安装](setup.md)第 0–5 步已经完成。

## 1. 建场景（每台机器一次）

```bash
bash scripts/build_scene.sh crate
```

做什么：先建 `outputs/scene/crate_bimanual_depth04`（筐在桌上往里挪 4 cm），再把桌子降到 0.30 m，得到 crate 实验用的 `outputs/scene/crate_table30_depth04`。
得到什么：两个目录，各有 `scene_locomani.xml` 和 `scene_meta.json`。约 3 秒。

## 2. 采集

```bash
bash scripts/start_stack.sh collect configs/experiments/crate_smoke.yaml
bash scripts/bimanual/run_agent_server.sh configs/experiments/crate_smoke.yaml
```

做什么：第一条起仿真栈（MuJoCo 仿真、SONIC 控制器、数据录制），起好就返回，栈留在后台。它先按空闲资源决定并行开几条通道（每条一个栈、一个跑局进程）并打印出来，`--lanes 1` 只开一条（[运行机制](running.md#6-并行)）。第二条在每条通道上运行视觉专家：一局接一局让机器人走近、抓筐、抬起，保留成功的局。每批 3 局一定跑满，保留够 2 条就不再开新批；结束时关掉所有仿真栈。
得到什么：运行目录 `outputs/runs/crate_smoke/`，里面有 `config.yaml`、每局成败 `manifest_00.jsonl`、每局录像 `videos/`、每局状态轨迹 `trace/`（双臂的这些文件在运行目录顶层）。起栈约 45 秒；采集约 2 分钟。每局的结果打印在 `visual_00.log` 里。

## 3. 转换

```bash
bash scripts/convert_data.sh outputs/runs/crate_smoke
```

做什么：把录制的数据转成 LeRobot 数据集，裁掉每条末尾的持物验收段，按 `train.holdout` 留出验证集，并检查数据。不用 GPU。
得到什么：`train_dataset/` 和 `val_dataset/`。约 30 秒。

## 4. 训练

```bash
bash scripts/train.sh outputs/runs/crate_smoke --gpu 0
```

做什么：用训练集微调 GR00T（三路相机、60 维状态、66 维动作）。
得到什么：`model/checkpoint-10/`，日志在 `train.log`。约 1 分钟；`model/` 占约 24 GB 磁盘。

## 5. 评测

开环评测（可选）：在验证集上逐帧比较模型预测的动作和示范动作，不起仿真。

```bash
bash scripts/open_loop_eval.sh --model-path outputs/runs/crate_smoke/model/checkpoint-10 \
  --dataset-path outputs/runs/crate_smoke/val_dataset --out-dir outputs/runs/crate_smoke/eval/open_loop
```

得到什么：`eval/open_loop/metrics.json`（每条轨迹的 MSE、MAE）和 `plots/`。约 20 秒；冒烟模型的 MSE 约 0.03。

闭环评测：让策略在仿真里控制机器人。需要两个终端，按顺序启动：

| 终端 | 命令 | 做什么 | 怎么算好了 |
|---|---|---|---|
| 1 | `bash scripts/start_stack.sh eval outputs/runs/crate_smoke --gpu 0` | 起评测用的仿真栈（lockstep 时钟，不录制） | 命令返回 |
| 1 | `bash scripts/run_vla_server.sh outputs/runs/crate_smoke --gpu 0` | 策略服务：加载 checkpoint，收观测、回动作 | 打印 `listening`；一直运行，评测结束后按 Ctrl-C |
| 2 | `bash scripts/bimanual/run_vla_inference.sh outputs/runs/crate_smoke --gpu 0` | 逐局重置场景、让策略控制机器人、记录成败 | 跑完 `eval.rounds` 局后关掉仿真栈并退出 |

得到什么：`eval/closed_loop.json`（冒烟为 0/2）、逐局记录 `eval/closed_loop.manifest.jsonl`、每局状态轨迹 `eval/state_trace/`。起栈和策略服务各约 25 秒；两局约 2.5 分钟。`eval/rounds_00.log` 里每局一行，例如 `[crate-vla] round 0: FAIL [vla] walk: never closer than 1.41 m (71 s)`；评测进程最后打印 `eval: 0/2`。

评测时仿真不录像。要看录像（可选），从状态轨迹渲：

```bash
MUJOCO_GL=egl "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/crate_smoke \
  --traces outputs/runs/crate_smoke/eval/state_trace --out outputs/runs/crate_smoke/eval/videos
```

得到什么：`eval/videos/<局>.mp4`，每局一个。两局约 15 秒。

熟悉以后可以一条命令跑完第 2–5 步（不含开环评测）：`bash scripts/run_overnight.sh configs/experiments/crate_smoke.yaml --gpu 0 --line bimanual`。已经有产物的步骤会被跳过。

## 6. 看结果

路径相对于 `outputs/runs/<名字>/`：

| 看什么 | 在哪里 |
|---|---|
| 每一步的汇总（保留数、成功率、checkpoint） | `summary.json` |
| 采集每局的成败和原因 | `manifest_NN.jsonl`；日志是 `visual_NN.log` |
| 采集录像 | `videos/` |
| 闭环每局的成败、失败阶段 | `eval/closed_loop.manifest.jsonl` |
| 闭环录像 | `eval/videos/`（第 5 步最后那条命令） |
| 产物是否齐全 | `python scripts/artifact_contract.py check outputs/runs/<名字> crate_eval` |

## 7. 常见报错

| 报错 | 处理 |
|---|---|
| `not enough headroom` | 机器上的 CPU、进程数或显存不够，脚本拒绝启动。等别的任务结束，或换一张卡（[运行机制](running.md#6-并行)） |
| `config.yaml differs from` | 同名运行的实验文件被改过。加 `--name` 换个运行名 |
| `model/ exists`、`closed_loop.json exists` | 这一步已经有产物，脚本不覆盖。确认不要了就删掉再跑 |
| 一批里很多局是 `initial_stance_unstable` | 关栈后不到 20 秒就重起了。等 20 秒，确认没有残留的仿真进程再起 |
| `[streamed_arm] POSE switch unconfirmed` | 这一局机器人整局不动，数据作废，重跑这一局 |

更多报错见 [crate 详解第七节](bimanual_design.md#七运行细节2026-09-28-从-crate-runbook-移来)。

## 8. 接下来

- 完整规模：把 `crate_smoke` 换成 `crate_table30_record`（20 条数据、3000 步训练、100 局评测）。采 20 局约保留 15 条；不到 20 条时采集以非零状态结束，已采的照样能做后面各步。
- 改参数：复制一份实验文件再改，常改的字段见[运行机制第 2 节](running.md#2-常改的字段)。
- 物流房间里抓纸箱用同一套脚本，但场景和出图方式不同，单独一页：[warehouse](warehouse.md)。
- 为什么这样设计、试过什么：[crate 详解](bimanual_design.md)；每一轮的实测数字：[双臂台账](ledger/crate.md)。
