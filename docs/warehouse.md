# warehouse：物流房间双臂抓纸箱

机器人在物流房间里从托盘的 −y 边（`ym`）走近，起点、朝向和纸箱位置每局随机，双手扣住纸箱沿，抬起、站直。策略看到的相机图由 Unreal 或 Isaac Sim 渲染，所以要 2 张 GPU：GPU 0 跑仿真、训练和策略，GPU 1 跑渲染器。
机器人是实验室真机 G1 的模型 mars，场景名都带 `_mars`；它和原来 stock 机器人的区别、换回 stock 的运行名，见 [warehouse 详解](warehouse_design.md#11-mars-机器人)。
下面按冒烟规模写（10 条数据，其中 1 条留作验证；10 步训练；2 局评测）。冒烟只检查每一步能不能跑通，训练 10 步的模型不会抓，闭环 0/2 是正常结果。耗时是 2026-09-30 照这一页实跑的结果（当时是 stock 机器人）：从采集到评测，Unreal 约 31 分钟，Isaac 约 50 分钟。2026-10-03 在 mars 上用 Unreal 跑了第 1–4 步：采集 3 批约 19 分钟、保留 9 条，补渲约 19 分钟，转换约 2 分钟。
和 crate 用同一套脚本，不同在出图：采集时仿真不渲染相机，转换前用渲染器补渲；评测时另开一个渲染器进程出图。原因见 [warehouse 详解](warehouse_design.md)；改参数、并行、重跑见[运行机制](running.md)。

## 先选渲染器

Unreal 和 Isaac 只在出图上不同，物理、专家、转换、训练都一样。选哪个由实验文件的 `images.source` 决定，训练集和评测用同一个渲染器。

| | Unreal（默认） | Isaac |
|---|---|---|
| 运行名（第 0 步的 `RUN`） | `warehouse_mars_smoke` | `warehouse_mars_smoke_isaac` |
| 安装 | [安装](setup.md)第 8 步 | [安装](setup.md)第 7 步 |
| 第 1 步 | 起编辑器，建 Unreal 关卡 | 建 Isaac 的机器人和 stage |
| 第 3 步补渲 | `render_dataset_cameras_unreal.sh` | `render_dataset_cameras_isaac.sh` |
| 第 6 步的镜像 | `run_unreal_mirror.sh` | `run_isaac_mirror.sh` |

下面的命令都用运行名 `$RUN`。和渲染器有关的命令，同一步里 Unreal、Isaac 各写一条。

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
RUN=warehouse_mars_smoke   # 用 Isaac：RUN=warehouse_mars_smoke_isaac
```

做什么：第一条设置路径变量，把 `src/` 加进 `PYTHONPATH`，并读入本机的 `locomani.local.sh`。第二条选运行名：实验文件是 `configs/experiments/$RUN.yaml`，运行目录是 `outputs/runs/$RUN`。命令都在仓库根目录执行。
前提：[安装](setup.md)第 0–5 步，以及所选渲染器的那一步，已经完成。

## 1. 建场景（每台机器一次）

```bash
bash scripts/build_scene.sh warehouse_textured_mars
bash scripts/build_scene.sh warehouse_box_45x32x33_mars
```

做什么：第一条从仓库自带的资产包建物流房间，并把其中的机器人换成 mars。第二条以它为底，量出托盘四条边外的空地，建出采集用的场景（纸箱宽 0.45 × 长 0.32 × 高 0.33 m）。
得到什么：`outputs/scene/warehouse_box_45x32x33_mars/scene_locomani.xml`。每条几秒到一分钟。
这台机器已经建过的话，两条都会打印 `refusing to replace existing` 加场景目录并以非零状态退出，不会覆盖；直接用已有的场景，往下做即可。要重建，先删掉 `outputs/scene/` 下对应的目录。

再给所选渲染器建这个场景的版本。

Unreal：

```bash
bash scripts/start_unreal.sh --gpu 1
bash scripts/build_scene.sh warehouse_box_45x32x33_mars --unreal
```

做什么：第一条在 GPU 1 上起 Unreal 编辑器，第 3、6、7 步都连它出图，用完执行 `bash scripts/stop_unreal.sh` 关掉。第二条在编辑器里建这个场景的关卡，机器人和纸箱从场景里导入；场景改过要重建。
得到什么：第一条打印 `unreal editor ready` 后返回，第一次启动要编译几分钟着色器，之后约 15 秒。第二条约 1 分钟，最后打印关卡的 `source.json`，其中 `robot` 是 `mars`。编辑器第一次导入 mars 的网格时可能超时报 `zmq.error.Again`，重跑这条即可。

Isaac：

```bash
bash scripts/build_scene.sh warehouse_box_45x32x33_mars --isaac
```

做什么：先把 mars 机器人转成 Isaac 用的 USD（已建过就跳过），再写出这个场景的 Isaac stage。它需要一个带 `pxr` 的 Python，脚本会自己找；找不到就 `pip install usd-core`，或用 `USD_PY` 指定。
得到什么：`assets/isaac/robot/mars/g1_dex1.usda` 和 `assets/isaac/scene/warehouse_box_45x32x33_mars_env/scene.usda`，第一次约 20 秒。

## 2. 采集

```bash
bash scripts/start_stack.sh collect configs/experiments/$RUN.yaml
bash scripts/bimanual/run_agent_server.sh configs/experiments/$RUN.yaml
```

做什么：第一条按空闲资源决定开几条并行通道（一个仿真栈加一个跑局进程），逐个起栈，起好就返回。第二条在每条通道上运行视觉专家，一局接一局让机器人走近、扣住箱沿、抬起，保留成功的局；每批 10 局，保留够 10 条就不再开新批，最多 3 批，结束时关掉所有仿真栈。
得到什么：`outputs/runs/$RUN/` 下每批一个的逐局成败 `manifest_NN.jsonl`，以及状态轨迹 `trace/`。起栈约 40 秒；本机只开得出 1 条通道，20 局约 12.5 分钟。
可选参数：第一条加 `--lanes 1` 只开一条通道，加 `--gpu 0` 只用这张卡；内存小的机器要写 `--lanes 1`。第二条加 `--dry-run` 只打印每局怎么摆放，不开跑。
3 批跑完仍不够 10 条时，第二条以非零状态结束，已采的照样能做后面各步。每局的摆放范围、通道数和怎么补够见[详解第 10 节](warehouse_design.md#10-运行细节2026-09-30-从-runbook-移来)。
要每局都从同一个地方出发：`RUN=warehouse_mars_smoke_fixed_start`（只有 Unreal 版）。要像 stock 线那样四条边轮流走入，把实验文件里的 `task.visual.start_edges` 改成 `[xm, xp, ym, yp]`，换个运行名。
第一视角默认是头顶云台上的 D455。要换成脸上固定的 D435：`RUN=warehouse_mars_smoke_d435_unreal`（Unreal）、`warehouse_mars_smoke_d435_isaac`（Isaac）或 `warehouse_mars_smoke_d435_mujoco`（只有一张卡，见第 11 节）；第 1 步的场景名换成 `warehouse_textured_mars_d435` 和 `warehouse_box_45x32x33_mars_d435`，Unreal 关卡、Isaac stage 也建这个场景的。D435 视野比 D455 窄（竖直 42° 对 65°），专家是在 D455 上调的。

## 3. 补渲相机

采集时仿真不出图，录下的相机帧是黑色占位图。这一步按状态轨迹，用所选渲染器渲出成功局的三路相机（第一视角和左右手腕），替换占位图。

MuJoCo（`images.source` 是 `mujoco` 的运行，即[第 11 节](#11-只有一张卡)那份复制出来的实验文件）：

```bash
bash scripts/render_dataset_cameras.sh outputs/runs/$RUN --gpu 0
```

做什么：不起仿真栈，在 `--gpu` 那张卡上用 EGL 离屏渲染，按每行数据用到的仿真时刻从状态轨迹渲出三路相机，写回占位图。
得到什么：同 Unreal。三种渲染器同一次运行只能用一种：补渲脚本和 `images.source` 不一致会拒绝，convert 也只认这一种。

Unreal：

```bash
bash scripts/render_dataset_cameras_unreal.sh outputs/runs/$RUN
```

做什么：在第 1 步起的编辑器里逐局补渲；编辑器所在的卡由 `start_unreal.sh` 定，所以这里不带 `--gpu`。
得到什么：成功局的相机帧被替换，每局打印一行 `episode 2: 907 renders for 1838 rows` 这样的统计。冒烟 9 条约 19 分钟。

Isaac：

```bash
bash scripts/render_dataset_cameras_isaac.sh outputs/runs/$RUN --gpu 1
```

做什么：在 GPU 1 上起 Isaac，逐局补渲，渲完退出。
得到什么：同 Unreal。冒烟 11 条约 26 分钟。

## 4. 转换

```bash
bash scripts/convert_data.sh outputs/runs/$RUN
```

做什么：把录制的数据转成 LeRobot 数据集，按 `train.holdout` 留出验证集，并检查数据。不用 GPU。
得到什么：`train_dataset/` 和 `val_dataset/`，约 2–4 分钟。之前跑过第 7 步的第三视角录像的话，录像会链接进数据集的 `third_person/`。

## 5. 训练

```bash
bash scripts/train.sh outputs/runs/$RUN --gpu 0
```

做什么：用训练集微调 GR00T。
得到什么：`model/checkpoint-10/`，日志在 `train.log`。约 1.3 分钟；`model/` 占 24 GB 磁盘。

## 6. 评测

开环评测（可选）：在验证集上逐帧比较模型预测的动作和示范动作，不起仿真。

```bash
bash scripts/open_loop_eval.sh --model-path outputs/runs/$RUN/model/checkpoint-10 \
  --dataset-path outputs/runs/$RUN/val_dataset --out-dir outputs/runs/$RUN/eval/open_loop
```

得到什么：`eval/open_loop/metrics.json`（每条轨迹的 MSE、MAE）和 `plots/`。约 20 秒；冒烟模型的 MSE 约 0.08。

闭环评测：三个终端，按顺序启动：

| 终端 | 命令 | 做什么 | 怎么算好了 |
|---|---|---|---|
| 1 | `bash scripts/start_stack.sh eval outputs/runs/$RUN --gpu 0` | 起评测用的仿真栈（lockstep 时钟，不录制） | 命令返回 |
| 1 | `bash scripts/run_vla_server.sh outputs/runs/$RUN --gpu 0` | 策略服务：加载 checkpoint，收观测、回动作 | 打印 `listening`；评测结束后按 Ctrl-C |
| 2 | Unreal：`bash scripts/run_unreal_mirror.sh outputs/runs/$RUN` | 镜像：策略每次要观测时，按仿真此刻的状态，在第 1 步起的编辑器里渲出三路相机图 | 打印 `unreal mirror listening on`；评测结束后按 Ctrl-C |
| 2 | Isaac：`bash scripts/run_isaac_mirror.sh outputs/runs/$RUN --gpu 1` | 同上，在 GPU 1 上用 Isaac 渲 | 打印 `isaac mirror listening on`；评测结束后按 Ctrl-C |
| 3 | `bash scripts/bimanual/run_vla_inference.sh outputs/runs/$RUN --gpu 0` | 逐局重置场景、让策略控制机器人、记录成败 | 跑完 `eval.rounds` 局后关掉仿真栈并退出 |

得到什么：`eval/closed_loop.json`（冒烟为 0/2）、逐局记录 `eval/closed_loop.manifest.jsonl` 和 `eval/rounds_00.log`、每局状态轨迹 `eval/state_trace/`。三个进程各约 20 秒起好；两局约 3–4 分钟。评测不录像，要看录像用第 7 步的命令加 `--eval`。

熟悉以后可以一条命令跑完第 2–6 步（不含开环评测和录像）：`bash scripts/run_overnight.sh configs/experiments/$RUN.yaml --gpu 0 --render-gpu 1 --line bimanual`。用 Unreal 时，编辑器没开会自动在 `--render-gpu` 那张卡上开。

## 7. 第三视角回放视频

采集和评测时都不录第三视角，要看时从状态轨迹事后渲；别的步骤都不需要它。Unreal 和 Isaac 用同一条命令：

```bash
bash scripts/render_third_person.sh outputs/runs/$RUN --gpu 1
```

做什么：按 `images.source` 选渲染器，给每个成功局画一段第三视角录像：机位按这局走入的那条边选，半透明品红夹爪是专家的抓取目标，三个小窗是第一视角和双腕。Unreal 用第 1 步起的编辑器，Isaac 用 `--gpu` 那张卡。
得到什么：`videos_unreal/<局>.mp4`（Isaac 是 `videos_isaac/`），`previews_unreal/`（`previews_isaac/`）里是每个 500 KB 以内的预览。冒烟 Isaac 11 局约 7.5 分钟，Unreal 8 局约 3.6 分钟。要让录像随数据集导出，在第 4 步之前跑。
可选参数：
- `--side-by-side`：另写 `<局>_vs_mujoco.mp4`，左边是 MuJoCo 回放，右边是所选渲染器，同一机位、逐帧对齐；缺的 MuJoCo 回放会先渲。冒烟约 1 分钟。
- `--eval`：渲第 6 步闭环评测的局，写到 `eval/` 下。冒烟两局都失败，要同时加 `--all`；Isaac 约 13 分钟，Unreal 约 14 分钟。
- `--all` 连失败局一起渲；末尾写局名只渲那几局；`--dry-run` 只打印要渲哪些局。已经渲过的局会跳过。

只想快速看动作对不对，可以不用渲染器，直接用 MuJoCo 回放：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/$RUN
```

做什么：把每局的状态轨迹逐帧写回 MuJoCo，从同一机位出图，成功和失败的局都渲。
得到什么：`videos/<局>.mp4`，20 局约 45 秒。渲评测的局要加 `--traces outputs/runs/$RUN/eval/state_trace --out outputs/runs/$RUN/eval/videos`。

## 8. 收尾

```bash
bash scripts/stop_unreal.sh
bash scripts/stop_stack.sh --instance N # 中途被打断而还挂着
```

做什么：关掉第 1 步起的 Unreal 编辑器；用 Isaac 时没有编辑器，跳过。采集和闭环评测的仿真栈在各自命令结束时已经自己关掉；中途被打断而还挂着的，用 `bash scripts/stop_stack.sh --instance N`（N 是起栈时打印的实例号）。
得到什么：打印 `unreal editor stopped`，`nvidia-smi` 里不再有 UnrealEditor 进程。

## 9. 看结果

路径相对于 `outputs/runs/$RUN/`：

| 看什么 | 在哪里 |
|---|---|
| 每一步的汇总（保留数、成功率、checkpoint） | `summary.json` |
| 采集每局的成败和原因 | `manifest_NN.jsonl` 的 `ok` 和 `why` |
| 采集录像 | `videos_unreal/` 或 `videos_isaac/`；MuJoCo 回放在 `videos/`（第 7 步） |
| 闭环每局的成败、失败阶段 | `eval/rounds_00.log` 每局一行；`eval/closed_loop.manifest.jsonl` |
| 闭环录像 | `eval/previews_unreal/` 或 `eval/previews_isaac/`（第 7 步加 `--eval`） |

## 10. 常见报错

| 报错 | 处理 |
|---|---|
| `not enough headroom` | 机器上的 CPU、进程数或显存不够，脚本拒绝启动。等别的任务结束（[运行机制](running.md#6-并行)） |
| 第二条命令刚开第一局就退出，终端只有一个 `Killed` | 内存不够，通道开多了。先 `bash scripts/stop_stack.sh --instance N` 收掉还挂着的栈（N 是起栈时打印的实例号），再用 `--lanes 1` 重来 |
| 转换报 `still hold black placeholder frames` | 漏了第 3 步补渲 |
| 评测每局报 `mirror: no reply from the mirror` 或 `camera/state streams never became ready` | 镜像还没起好。等终端 2 打印 `unreal mirror listening on`（Isaac 是 `isaac mirror listening on`）再起终端 3 |
| `an Unreal editor is already running` 后面跟着 `not` 和卡号 | 编辑器开在另一张卡上。`bash scripts/stop_unreal.sh` 后用要的卡重开 |
| 第三视角命令报 `has no third-person camera and ghost grippers` | Unreal 关卡建得太早：`bash scripts/build_scene.sh <场景> --unreal` 重建 |
| `cannot use images.source unreal` | 这个场景没有 Unreal 关卡，只有物流房间的场景能用 |
| `the scene carries mars` | 关卡或 Isaac stage 是给另一个机器人建的。按报错末尾打印的命令重建 |
| `config.yaml differs from` | 同名运行的实验文件被改过。加 `--name` 换个运行名 |
| 失败大多是 `reach_stalled`、`lost_attachment`、`basket_tilt` | 这是这条线目前主要的失败形态（手停在箱沿上方、合爪后脱开、抬起时箱子倾斜过大），不是环境问题 |

更多报错和评测前要核对的项见 [warehouse 详解](warehouse_design.md)。

## 11. 只有一张卡

Unreal 和 Isaac 都要第二张卡。只有一张卡时，把实验文件复制一份，`images.source` 改成 `mujoco`，`RUN` 设成新文件的名字，策略就改看 MuJoCo 渲的图；mars 冒烟已有现成的 `warehouse_mars_smoke_mujoco` 和 D435 版 `warehouse_mars_smoke_d435_mujoco`。这时第 1 步不建渲染器的版本；第 3 步换成 `bash scripts/render_dataset_cameras.sh outputs/runs/$RUN --gpu 0`；闭环评测不起终端 2；`run_overnight.sh` 去掉 `--render-gpu 1`；第 7 步的 `--gpu` 写 0。训练集和评测必须用同一种图。

## 12. 接下来

- 完整规模：`RUN=warehouse_mars_200`（200 条数据、1 万步训练、100 局评测），Isaac 是 `warehouse_mars_200_isaac`。按 stock 在 −y 边 19 局保留 11 条的产出率估计，保留 200 条约需 350 局。
- 调专家参数做对照实验：见 [warehouse 详解第 7 节](warehouse_design.md#7-做对照实验)。
- 台上纸箱、四边线、Blender 场景这些支线：见 [warehouse 详解第 8 节](warehouse_design.md#8-相关的支线)。
- 每一轮的实测数字：[双臂台账](ledger/crate.md)。
