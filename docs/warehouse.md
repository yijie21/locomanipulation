# warehouse：物流房间双臂抓纸箱

机器人在物流房间里从托盘四条边中随机一条走近（也可以固定从一个地方开始，见 §2），双手扣住纸箱沿，抬起、站直。策略看到的相机图由 Isaac Sim 渲染（也可以改用 Unreal，见 §10），所以要 2 张 GPU：GPU 0 跑仿真、训练和策略，GPU 1 跑 Isaac。
下面按冒烟规模写（2 条数据、10 步训练、2 局评测），从采集到评测跑完大约 50 分钟（其中评测录像约 20 分钟，可以不渲；这两个数是按 2× 超采样估计的，还没整条实跑过）。冒烟只检查每一步能不能跑通，训练 10 步的模型不会抓，闭环 0/2 是正常结果。
和 crate 用同一套脚本，不同的是：采集时仿真不渲染相机，转换前要用 Isaac 补渲；评测时另开一个 Isaac 进程出图。原因和细节见 [warehouse 详解](warehouse_design.md)；改参数、并行、重跑见[运行机制](running.md)。

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
```

做什么：设置路径变量，把 `src/` 加进 `PYTHONPATH`，并读入本机的 `locomani.local.sh`。下面的命令都在仓库根目录执行。
前提：[安装](setup.md)第 0–5 步和第 7 步（Isaac）已经完成。

## 1. 建场景（每台机器一次）

```bash
bash scripts/build_scene.sh warehouse_textured
bash scripts/build_scene.sh warehouse_box_45x32x33
bash scripts/build_scene.sh warehouse_box_45x32x33 --isaac
```

做什么：第一条从仓库自带的资产包建物流房间 `outputs/scene/warehouse_textured`；第二条以它为底，量出托盘四条边外的空地，写进场景 `outputs/scene/warehouse_box_45x32x33`（纸箱尺寸与房间原来的相同：宽 0.45 × 长 0.32 × 高 0.33 m），下面的采集用这个场景；第三条建这个场景给 Isaac 用的版本，写到 `assets/isaac/scene/warehouse_box_45x32x33_env/`，需要一个带 `pxr` 的 Python：脚本自动在 `python3` 和 PATH 上 `blender` 自带的 Python 里找，都没有就 `pip install usd-core` 或用 `USD_PY` 指定。
得到什么：`outputs/scene/warehouse_box_45x32x33/scene_locomani.xml` 和 `assets/isaac/scene/warehouse_box_45x32x33_env/scene.usda`。每条几秒到一分钟。

## 2. 采集

```bash
bash scripts/start_stack.sh collect configs/experiments/warehouse_smoke.yaml
bash scripts/bimanual/run_agent_server.sh configs/experiments/warehouse_smoke.yaml
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/warehouse_smoke
```

做什么：第一条按这台机器的空闲 CPU、进程数、显存和实例，决定并行开几条通道（一个仿真栈加一个跑局进程），打印出来并逐个起栈，起好就返回；只要一条就加 `--lanes 1`，只用某张卡就加 `--gpu 0`（[运行机制](running.md#6-并行)）。第二条在每条通道上运行视觉专家，开始前再打印一次通道数：一局接一局让机器人走近、扣住箱沿、抬起，保留成功的局；每批 4 局一定跑满，保留够 2 条就不再开新批（已经在跑的批会跑完），结束时关掉所有仿真栈。第三条（可选）从每局的状态轨迹渲出第三视角录像，画面里半透明的品红夹爪是专家给的抓取目标；每局按走入的那条边选机位。
得到什么：运行目录 `outputs/runs/warehouse_smoke/`，里面有每批一个的逐局成败 `manifest_NN.jsonl`、状态轨迹 `trace/`、录像 `videos/`、通道安排 `collect/lanes.json`。每条通道起栈约 40 秒；采集约 3 分钟，一局约 35 秒；录像约 15 秒。

**每局从哪里开始（默认：四边随机）。** 每局从托盘四条边（`xm/xp/ym/yp`）中随机抽一条，纸箱搬到这条边上，机器人从这条边前方走过来。每个随机量的范围，都是单独改这一项时成功率仍不低于 30% 的区间（[双臂抓箱的参数范围](box_grasp_envelope.md) §3）：起点离台边 0.6 m 到“这条边外的空地 − 0.25 m”（xm 0.86、xp 0.80、ym 1.04、yp 1.38 m；房间的货架和墙没有碰撞体，起点不能放进去），起点朝向 ±20°，起点横移 −8～0 cm（机器人在箱子中线右侧），纸箱离台边 0–2.5 cm、偏角 0～+3.5°，沿边位置在整条边上任意取。
同样设置在房间里实测保留 17/40（[双臂台账](ledger/crate.md) 2026-09-28）。第二条命令开始第一局之前，会把这份设置打印出来：

```
collect: scene warehouse_box_45x32x33, 2 kept wanted in batches of 4 rounds (at most 3). Each round draws (task.visual):
  walk-in  toward one of the platform's edges xm, xp, ym, yp, drawn each round
  start    0.6 m up to xm 0.862 / xp 0.798 / ym 1.035 / yp 1.379 m before that edge (its free floor - 0.25 m)
  robot    heading -20..20 deg off the walk-in, lateral -0.08..0 m (+ = left of the box centre line)
  box      setback 0..0.025 m, yaw 0..3.5 deg, along the edge -1..1 of the room it leaves
  from one place only, as warehouse collected before: its *_fixed_start experiment (docs/warehouse.md §2)
```

在终端里运行时，打印后等 10 秒才开始第一局，这时按 Ctrl-C 就不会录下任何一局，仿真栈也会一起关掉。不看这份打印，也可以先加 `--dry-run` 运行第二条命令，它只打印、不启动。

**只从一个地方开始。** 要像 2026-09-28 之前那样，每局都从同一个位置出发（场景自带的起点，在 −y 边前方，往后退 0.25 m；纸箱在原位，只加 ±2 cm、±5° 的抖动），就把上面三条命令里的实验文件换成 `configs/experiments/warehouse_smoke_fixed_start.yaml`，运行名随之变成 `warehouse_smoke_fixed_start`。后面几步的 `outputs/runs/warehouse_smoke` 也要换成它。完整规模对应 `warehouse_200_fixed_start.yaml`。这两份文件和默认文件相比，只有摆放方式不同，专家参数完全一样。场景仍用 `warehouse_box_45x32x33`。

## 3. 补渲相机

```bash
bash scripts/render_dataset_cameras_isaac.sh outputs/runs/warehouse_smoke --gpu 1
```

做什么：采集时仿真没有出图，录下的相机帧是黑色占位图。这一步按状态轨迹，用 Isaac 把成功局的三路相机（第一视角和左右手腕）渲出来，替换占位图。
得到什么：录制数据里成功局的相机帧被替换。每局约 3 分钟（估计：三路相机按 2× 超采样渲，每次约 0.21 s；Isaac 启动另需约 20 秒），打印 `episode 2: 907 renders for 1838 rows` 这样的每局统计。

## 4. 转换

```bash
bash scripts/convert_data.sh outputs/runs/warehouse_smoke
```

做什么：把录制的数据转成 LeRobot 数据集，按 `train.holdout` 留出验证集，并检查数据。不用 GPU。
得到什么：`train_dataset/` 和 `val_dataset/`。约 40 秒。

## 5. 训练

```bash
bash scripts/train.sh outputs/runs/warehouse_smoke --gpu 0
```

得到什么：`model/checkpoint-10/`，日志在 `train.log`。约 1 分钟；`model/` 占约 24 GB 磁盘。

## 6. 评测

开环评测（可选）：在验证集上逐帧比较模型预测的动作和示范动作，不起仿真。

```bash
bash scripts/open_loop_eval.sh --model-path outputs/runs/warehouse_smoke/model/checkpoint-10 \
  --dataset-path outputs/runs/warehouse_smoke/val_dataset --out-dir outputs/runs/warehouse_smoke/eval/open_loop
```

得到什么：`eval/open_loop/metrics.json`（每条轨迹的 MSE、MAE）和 `plots/`。约 30 秒；冒烟模型的 MSE 约 0.07。

闭环评测：需要三个终端，按顺序启动：

| 终端 | 命令 | 做什么 | 怎么算好了 |
|---|---|---|---|
| 1 | `bash scripts/start_stack.sh eval outputs/runs/warehouse_smoke --gpu 0` | 起评测用的仿真栈（lockstep 时钟，不录制） | 命令返回 |
| 1 | `bash scripts/run_vla_server.sh outputs/runs/warehouse_smoke --gpu 0` | 策略服务：加载 checkpoint，收观测、回动作 | 打印 `listening`；一直运行，评测结束后按 Ctrl-C |
| 2 | `bash scripts/run_isaac_mirror.sh outputs/runs/warehouse_smoke --gpu 1` | Isaac 镜像：策略每次要观测时，按仿真此刻的状态渲出三路相机图 | 先刷几百行启动日志，打印 `isaac mirror listening on` 才算好；评测结束后按 Ctrl-C |
| 3 | `bash scripts/bimanual/run_vla_inference.sh outputs/runs/warehouse_smoke --gpu 0` | 逐局重置场景、让策略控制机器人、记录成败 | 跑完 `eval.rounds` 局后关掉仿真栈并退出 |

得到什么：`eval/closed_loop.json`（冒烟为 0/2）、逐局记录 `eval/closed_loop.manifest.jsonl`、每局状态轨迹 `eval/state_trace/`、镜像每次渲染的记录 `eval/isaac_mirror.jsonl`。起栈、策略服务、镜像各约 20 秒；两局约 2.5 分钟。`eval/rounds_00.log` 里每局一行，例如 `[crate-vla] round 0: FAIL [vla] walk: never closer than 1.06 m (70 s)`。

评测时仿真不录像。要看评测录像（第三视角，带第一视角和双腕小窗），事后渲：

```bash
bash scripts/render_isaac_videos.sh outputs/runs/warehouse_smoke --traces outputs/runs/warehouse_smoke/eval/state_trace --out outputs/runs/warehouse_smoke/eval --insets --gpu 1 --all
```

默认只渲成功局，`--all` 连失败局一起渲。冒烟的两局各约 10 分钟（估计：一局 99 秒的画面，1280×960，Isaac 侧按 2× 超采样渲）。原片在 `eval/isaac/<局>/side_by_side.mp4`；`eval/previews_isaac/` 里是压到每个文件 500 KB 以内的预览，长的局切成几段（冒烟每局 11 段）。

熟悉以后可以一条命令跑完第 2–6 步（不含开环评测和录像）：`bash scripts/run_overnight.sh configs/experiments/warehouse_smoke.yaml --gpu 0 --render-gpu 1 --line bimanual`。

## 7. 看结果

路径相对于 `outputs/runs/<名字>/`：

| 看什么 | 在哪里 |
|---|---|
| 每一步的汇总（保留数、成功率、checkpoint） | `summary.json` |
| 采集每局的成败和原因 | `manifest_NN.jsonl` 的 `ok` 和 `why` |
| 采集录像 | `videos/`（第 2 步第三条命令渲的） |
| 闭环每局的成败、失败阶段 | `eval/rounds_00.log` 每局一行；`eval/closed_loop.manifest.jsonl` |
| 闭环录像 | `eval/previews_isaac/`（第 6 步最后一条命令渲的） |

## 8. 常见报错

| 报错 | 处理 |
|---|---|
| `not enough headroom` | 机器上的 CPU、进程数或显存不够，脚本拒绝启动。等别的任务结束（[运行机制](running.md#6-并行)） |
| 转换报 `still hold black placeholder frames` | 漏了第 3 步补渲 |
| 评测每局报 `mirror: no reply from the mirror` 或 `camera/state streams never became ready` | 镜像没起来或还没就绪。看终端 2，等它打印 `isaac mirror listening on`（Unreal 是 `unreal mirror listening on`）再起终端 3 |
| `an Unreal editor is already running` 后面跟着 `not` 和卡号 | 编辑器开在另一张卡上。`bash scripts/stop_unreal.sh` 后用要的卡重开 |
| `cannot use images.source unreal` | 这个场景没有 Unreal 关卡（`configs/scenes.yaml` 的 `unreal` 条目写了原因），只有物流房间的场景能用 |
| `config.yaml differs from` | 同名运行的实验文件被改过。加 `--name` 换个运行名 |
| 失败大多是 `reach_stalled`、`lost_attachment` | 这是这条线目前主要的失败形态（手停在箱沿上方、合爪后脱开），不是环境问题 |

更多报错和评测前要核对的项见 [warehouse 详解](warehouse_design.md)。

## 9. 只有一张卡

把实验文件复制一份，`images.source` 改成 `mujoco`，策略就改看 MuJoCo 渲的图。这时第 1 步不跑第三条；第 3 步换成 `bash scripts/render_dataset_cameras.sh outputs/runs/<名字> --gpu 0`；闭环评测不起终端 2 的镜像；`run_overnight.sh` 去掉 `--render-gpu 1`。训练集和评测必须用同一种图。

## 10. 改用 Unreal 出图

前提：[安装](setup.md)第 8 步已做完。只换策略看到的图，物理、专家和其余命令都不变；训练集和评测必须用同一种图，所以要从采集开始另跑一次。

```bash
bash scripts/start_unreal.sh --gpu 1
bash scripts/build_scene.sh warehouse_box_45x32x33 --unreal
```

做什么：第一条在 GPU 1 上起 Unreal 编辑器。编辑器一直开着，下面几条命令都连它出图。用完执行 `bash scripts/stop_unreal.sh` 关掉。第二条在编辑器里建这个场景的 Unreal 关卡，每个场景建一次。场景改过以后要重建，否则运行时报 `rebuild it` 并拒绝出图。
得到什么：第一条打印 `unreal editor ready` 后返回，第一次启动要编译几分钟着色器，之后约 15 秒；第二条约 20 秒，最后打印这个关卡的 `source.json`。

然后把实验文件复制一份，`images.source` 改成 `unreal`，按第 2–6 步跑，只换两处：

```bash
bash scripts/render_dataset_cameras_unreal.sh outputs/runs/<名字>
bash scripts/run_unreal_mirror.sh outputs/runs/<名字>
```

做什么：第一条代替第 3 步，在编辑器里补渲成功局的三路相机；第二条代替第 6 步终端 2 的 Isaac 镜像。两条都不带 `--gpu`，用的是编辑器所在的卡。
得到什么：补渲每次约 130 ms，和第 3 步一样打印每局统计；镜像打印 `unreal mirror listening on` 才算好，之后再起终端 3。第 6 步的录像命令不变，画面仍是 Isaac 渲的。一条命令跑完是 `run_overnight.sh` 加 `--render-gpu 1`，编辑器没开会自动开。

## 11. 接下来

- 完整规模：把 `warehouse_smoke` 换成 `warehouse_200`（200 条数据、1 万步训练、100 局评测）。四边随机还没有按这个规模跑过，按 17/40 估计，保留 200 条约需 470 局。只从一个地方开始的 `box_room_grasp_200_cap120_20260925`（当时的专家，刹车后准备手臂较慢，成功要求后退 0.3 m）实测：950 局保留 204 条（产出率 0.22），一局约 45 秒；训练到 5000 步停下，开环 MSE 0.00123，Isaac 镜像下闭环 100 局墙钟 81、lockstep 88（[双臂台账](ledger/crate.md) 2026-09-27）。
- 调专家参数做对照实验：从 `box_room_grasp_baseline_20260924` 复制，做法见 [warehouse 详解第 7 节](warehouse_design.md#7-做对照实验)。
- 台上纸箱、四边线、Blender 场景这些支线：见 [warehouse 详解第 8 节](warehouse_design.md#8-相关的支线)。
- 每一轮的实测数字：[双臂台账](ledger/crate.md)。
