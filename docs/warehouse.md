# warehouse：物流房间双臂抓纸箱

机器人在物流房间里从托盘四条边中随机一条走近（也可以固定从一个地方开始，见 §2），双手扣住纸箱沿，抬起、站直。策略看到的相机图由 Isaac Sim 或 Unreal 渲染，二选一（见下面的"先选渲染器"），所以要 2 张 GPU：GPU 0 跑仿真、训练和策略，GPU 1 跑渲染器。
下面按冒烟规模写（10 条数据，其中 1 条留作验证；10 步训练；2 局评测）。2026-09-30 照这一页整条实跑：从采集到评测，Isaac 约 50 分钟（其中补渲约 26 分钟），Unreal 约 31 分钟（这次只保留了 8 条）；第 7 步的第三视角录像另算，采集和评测的都渲，Isaac 约 20 分钟、Unreal 约 18 分钟。每批 10 局、最多 3 批：实跑时两批 20 局分别保留了 11 条和 8 条，所以允许第三批（多约 6 分钟）；3 批仍不够时见第 2 步。冒烟只检查每一步能不能跑通，训练 10 步的模型不会抓，闭环 0/2 是正常结果。
和 crate 用同一套脚本，不同的是：采集时仿真不渲染相机，转换前要用渲染器补渲；评测时另开一个渲染器的进程出图。原因和细节见 [warehouse 详解](warehouse_design.md)；改参数、并行、重跑见[运行机制](running.md)。

## 先选渲染器

Isaac 和 Unreal 只在出图上不同，物理、专家、采集、转换、训练的命令都一样。选哪个由实验文件的 `images.source` 决定，训练集和评测用的是同一个渲染器。

| | Isaac（默认） | Unreal |
|---|---|---|
| 实验文件 | `warehouse_smoke.yaml`，运行名 `warehouse_smoke` | `warehouse_smoke_unreal.yaml`，运行名 `warehouse_smoke_unreal` |
| 安装 | [安装](setup.md)第 7 步 | [安装](setup.md)第 8 步 |
| 第 1 步建场景 | 建 Isaac stage | 起编辑器，建 Unreal 关卡 |
| 第 3 步补渲 | `render_dataset_cameras_isaac.sh` | `render_dataset_cameras_unreal.sh` |
| 第 6 步终端 2 的镜像 | `run_isaac_mirror.sh` | `run_unreal_mirror.sh` |

下面每一步里和渲染器有关的命令，两种都写了。其余命令按 Isaac 的运行名写；用 Unreal 时，把实验文件和 `outputs/runs/warehouse_smoke` 换成上表 Unreal 那一列的。第三视角录像是可选的：第 7 步的一条命令按所选渲染器画，Isaac 和 Unreal 同一条。

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
```

做什么：设置路径变量，把 `src/` 加进 `PYTHONPATH`，并读入本机的 `locomani.local.sh`。下面的命令都在仓库根目录执行。
前提：[安装](setup.md)第 0–5 步，以及所选渲染器的那一步（Isaac 第 7 步，Unreal 第 8 步）已经完成。

## 1. 建场景（每台机器一次）

```bash
bash scripts/build_scene.sh warehouse_textured
bash scripts/build_scene.sh warehouse_box_45x32x33
```

做什么：第一条从仓库自带的资产包建物流房间 `outputs/scene/warehouse_textured`；第二条以它为底，量出托盘四条边外的空地，写进场景 `outputs/scene/warehouse_box_45x32x33`（纸箱尺寸与房间原来的相同：宽 0.45 × 长 0.32 × 高 0.33 m），下面的采集用这个场景。
得到什么：`outputs/scene/warehouse_box_45x32x33/scene_locomani.xml`。每条几秒到一分钟。

再按所选渲染器，建这个场景给它用的版本。Isaac：

```bash
bash scripts/build_scene.sh warehouse_box_45x32x33 --isaac
```

做什么：写出这个场景的 Isaac stage。它需要一个带 `pxr` 的 Python：脚本自动在 `python3` 和 PATH 上 `blender` 自带的 Python 里找，都没有就 `pip install usd-core` 或用 `USD_PY` 指定。
得到什么：`assets/isaac/scene/warehouse_box_45x32x33_env/scene.usda`，几秒。

Unreal：

```bash
bash scripts/start_unreal.sh --gpu 1
bash scripts/build_scene.sh warehouse_box_45x32x33 --unreal
```

做什么：第一条在 GPU 1 上起 Unreal 编辑器。编辑器一直开着，第 3、6 步都连它出图，用完执行 `bash scripts/stop_unreal.sh` 关掉。第二条在编辑器里建这个场景的 Unreal 关卡，每个场景建一次。场景改过以后要重建，否则运行时报 `rebuild it` 并拒绝出图。
得到什么：第一条打印 `unreal editor ready` 后返回，第一次启动要编译几分钟着色器，之后约 15 秒；第二条约 20 秒，最后打印这个关卡的 `source.json`。

## 2. 采集

```bash
bash scripts/start_stack.sh collect configs/experiments/warehouse_smoke.yaml
bash scripts/bimanual/run_agent_server.sh configs/experiments/warehouse_smoke.yaml
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/warehouse_smoke
```

做什么：第一条按这台机器的空闲 CPU、进程数、显存和实例，决定并行开几条通道（一个仿真栈加一个跑局进程），打印出来并逐个起栈，起好就返回；只要一条就加 `--lanes 1`，只用某张卡就加 `--gpu 0`（[运行机制](running.md#6-并行)）。第二条在每条通道上运行视觉专家，开始前再打印一次通道数：一局接一局让机器人走近、扣住箱沿、抬起，保留成功的局；每批 10 局一定跑满，保留够 10 条就不再开新批（已经在跑的批会跑完），最多 3 批，结束时关掉所有仿真栈。3 批跑完还不够 10 条时，打印 `stopped at 8/10 kept after 3 batches` 这样的一行并以非零状态结束；已采的照样能接着做第 3 步起的各步，只是数据少几条。`run_overnight.sh` 遇到这种情况会停在采集，重跑也不会多采；要凑够，复制实验文件把 `collect.max_batches` 调大，换运行名重采。第三条（可选）从每局的状态轨迹渲出 MuJoCo 的第三视角回放，全部局（含失败局）都渲，只给人看；用所选渲染器画第三视角录像和 Isaac 版回放见第 7 步。
得到什么：运行目录 `outputs/runs/warehouse_smoke/`，里面有每批一个的逐局成败 `manifest_NN.jsonl`、状态轨迹 `trace/`、录像 `videos/`、通道安排 `collect/lanes.json`。起栈约 40 秒。每批 10 局的通道要约 1600 个进程，2026-09-30 本机只开出 1 条，20 局约 12.5 分钟（一局约 37 秒）；录像（20 局）约 45 秒。

**每局从哪里开始（默认：四边随机）。** 每局从托盘四条边（`xm/xp/ym/yp`）中随机抽一条，纸箱搬到这条边上，机器人从这条边前方走过来。每个随机量的范围，都是单独改这一项时成功率仍不低于 30% 的区间（[双臂抓箱的参数范围](box_grasp_envelope.md) §3）：起点离台边 0.6 m 到“这条边外的空地 − 0.25 m”（xm 0.86、xp 0.80、ym 1.04、yp 1.38 m；房间的货架和墙没有碰撞体，起点不能放进去），起点朝向 ±20°，起点横移 −8～0 cm（机器人在箱子中线右侧），纸箱离台边 0–2.5 cm、偏角 0～+3.5°，沿边位置在整条边上任意取。
同样设置在房间里实测保留 17/40（[双臂台账](ledger/crate.md) 2026-09-28）。第二条命令开始第一局之前，会把这份设置打印出来：

```
collect: scene warehouse_box_45x32x33, 10 kept wanted in batches of 10 rounds (at most 3). Each round draws (task.visual):
  walk-in  toward one of the platform's edges xm, xp, ym, yp, drawn each round
  start    0.6 m up to xm 0.862 / xp 0.798 / ym 1.035 / yp 1.379 m before that edge (its free floor - 0.25 m)
  robot    heading -20..20 deg off the walk-in, lateral -0.08..0 m (+ = left of the box centre line)
  box      setback 0..0.025 m, yaw 0..3.5 deg, along the edge -1..1 of the room it leaves
  from one place only, as warehouse collected before: its *_fixed_start experiment (docs/warehouse.md §2)
```

在终端里运行时，打印后等 10 秒才开始第一局，这时按 Ctrl-C 就不会录下任何一局，仿真栈也会一起关掉。不看这份打印，也可以先加 `--dry-run` 运行第二条命令，它只打印、不启动。

**通道数（`--lanes`）。** 一条通道是一套仿真栈加一个跑局进程，并行只改变速度、不改变数据；几条通道由第一条命令按空闲资源决定，但它不算系统内存，一条通道常驻约 5 GB。没有 cgroup CPU 配额的桌面机（如 32 GB 内存的 SRA0889）会被开出 4 条并在第一局被内核杀掉，这样的机器写 `--lanes 1` 或在实验文件里写 `collect.lanes: 1`（[运行机制](running.md#6-并行)）。

**只从一个地方开始。** 要像 2026-09-28 之前那样，每局都从同一个位置出发（场景自带的起点，在 −y 边前方，往后退 0.25 m；纸箱在原位，只加 ±2 cm、±5° 的抖动），就把上面三条命令里的实验文件换成 `configs/experiments/warehouse_smoke_fixed_start.yaml`，运行名随之变成 `warehouse_smoke_fixed_start`。后面几步的 `outputs/runs/warehouse_smoke` 也要换成它。完整规模对应 `warehouse_200_fixed_start.yaml`。这两份文件和默认文件相比，只有摆放方式不同，专家参数完全一样。场景仍用 `warehouse_box_45x32x33`。

## 3. 补渲相机

采集时仿真没有出图，录下的相机帧是黑色占位图。这一步按状态轨迹，用所选的渲染器把成功局的三路相机（第一视角和左右手腕）渲出来，替换占位图。这一步不画第三视角录像，要的话另跑第 7 步的命令。Isaac：

```bash
bash scripts/render_dataset_cameras_isaac.sh outputs/runs/warehouse_smoke --gpu 1
```

做什么：在 GPU 1 上起 Isaac，逐局补渲，渲完退出。
得到什么：录制数据里成功局的相机帧被替换。相机帧的 mp4 不在运行目录下，而在采集时的 session 目录里（路径列在 `collect/inputs.json` 的 `sessions`）：`<session>/videos/chunk-000/observation.images.<ego_view|left_wrist|right_wrist>/episode_NNNNNN.mp4`，渲过的几 MB，失败局仍是 100 多 KB 的黑帧；`<session>/meta/offline_render.json` 记着是谁渲的。一次运行的相机只能由 `images.source` 指定的那个渲染器填，换一个会被拒绝。补渲每局约 2.5 分钟，冒烟 11 条共约 26 分钟（2026-09-30 实测每局 135–163 s，每次 0.21 s；Isaac 启动另需约 20 秒），打印 `episode 2: 907 renders for 1838 rows` 这样的每局统计。

Unreal：

```bash
bash scripts/render_dataset_cameras_unreal.sh outputs/runs/warehouse_smoke_unreal
```

做什么：在第 1 步起的编辑器里逐局补渲，用的是编辑器所在的卡，所以不带 `--gpu`。
得到什么：和 Isaac 一样打印每局统计，每次约 130 ms，每局 74–93 秒，冒烟 8 条共约 11 分钟（2026-09-30 实测）。

## 4. 转换

```bash
bash scripts/convert_data.sh outputs/runs/warehouse_smoke
```

做什么：把录制的数据转成 LeRobot 数据集，按 `train.holdout` 留出验证集，并检查数据。不用 GPU。
得到什么：`train_dataset/` 和 `val_dataset/`。约 2–4 分钟，大部分在重编码视频（2026-09-30 实测：Isaac 11 条 4.2 分钟，Unreal 8 条 1.8 分钟）。转换前跑过第 7 步的命令的话，每条数据的第三视角录像会链接进数据集的 `third_person/episode_N.mp4`；没跑就没有这个目录，转换照常。

## 5. 训练

```bash
bash scripts/train.sh outputs/runs/warehouse_smoke --gpu 0
```

得到什么：`model/checkpoint-10/`，日志在 `train.log`。约 1.3 分钟（实测 74–80 秒）；`model/` 占 24 GB 磁盘。

## 6. 评测

开环评测（可选）：在验证集上逐帧比较模型预测的动作和示范动作，不起仿真。

```bash
bash scripts/open_loop_eval.sh --model-path outputs/runs/warehouse_smoke/model/checkpoint-10 \
  --dataset-path outputs/runs/warehouse_smoke/val_dataset --out-dir outputs/runs/warehouse_smoke/eval/open_loop
```

得到什么：`eval/open_loop/metrics.json`（每条轨迹的 MSE、MAE）和 `plots/`。约 20 秒；冒烟模型的 MSE 约 0.08（2026-09-30 实测 0.076）。

闭环评测：需要三个终端，按顺序启动：

| 终端 | 命令 | 做什么 | 怎么算好了 |
|---|---|---|---|
| 1 | `bash scripts/start_stack.sh eval outputs/runs/warehouse_smoke --gpu 0` | 起评测用的仿真栈（lockstep 时钟，不录制） | 命令返回 |
| 1 | `bash scripts/run_vla_server.sh outputs/runs/warehouse_smoke --gpu 0` | 策略服务：加载 checkpoint，收观测、回动作 | 打印 `listening`；一直运行，评测结束后按 Ctrl-C |
| 2 | Isaac：`bash scripts/run_isaac_mirror.sh outputs/runs/warehouse_smoke --gpu 1` | Isaac 镜像：策略每次要观测时，按仿真此刻的状态渲出三路相机图 | 先刷几百行启动日志，打印 `isaac mirror listening on` 才算好；评测结束后按 Ctrl-C |
| 2 | Unreal：`bash scripts/run_unreal_mirror.sh outputs/runs/warehouse_smoke_unreal` | Unreal 镜像，作用同上；在第 1 步起的编辑器里渲，所以不带 `--gpu` | 打印 `unreal mirror listening on` 才算好；评测结束后按 Ctrl-C |
| 3 | `bash scripts/bimanual/run_vla_inference.sh outputs/runs/warehouse_smoke --gpu 0` | 逐局重置场景、让策略控制机器人、记录成败 | 跑完 `eval.rounds` 局后关掉仿真栈并退出 |

得到什么：`eval/closed_loop.json`（冒烟为 0/2）、逐局记录 `eval/closed_loop.manifest.jsonl`、每局状态轨迹 `eval/state_trace/`、镜像每次渲染的记录 `eval/isaac_mirror.jsonl`（Unreal 是 `eval/unreal_mirror.jsonl`）。起栈、策略服务、镜像各约 20 秒；两局约 3–4 分钟（先跑一局热身，每局 75–100 秒；2026-09-30 实测 Isaac 3.8 分钟、Unreal 3 分钟）。`eval/rounds_00.log` 里每局一行，例如 `[crate-vla] round 0: FAIL [vla] walk: never closer than 1.06 m (70 s)`。

评测时仿真不录像。要看评测录像，用第 7 步的命令加 `--eval`（Isaac 和 Unreal 同一条）：

```bash
bash scripts/render_third_person.sh outputs/runs/warehouse_smoke --gpu 1 --eval --all
```

画面和采集的第三视角录像一样（同一机位、第一视角和双腕小窗，小窗是镜像给策略的那种图；评测没有 ghost）。默认只渲成功局，冒烟的两局都失败，所以加 `--all`。原片在 `eval/videos_isaac/<局>.mp4`（Unreal 是 `videos_unreal/`），`eval/previews_isaac/`（`previews_unreal/`）里是压到每个文件 500 KB 以内的预览，每局切成 2–5 段。先渲小窗再渲第三视角：冒烟两局，Isaac 约 13 分钟（每局小窗 225–243 s + 录像 123 s），Unreal 约 14 分钟（每局 223–285 s + 130–140 s），2026-09-30 实测。

熟悉以后可以一条命令跑完第 2–6 步（不含开环评测和录像）：`bash scripts/run_overnight.sh configs/experiments/warehouse_smoke.yaml --gpu 0 --render-gpu 1 --line bimanual`。用 Unreal 时换成 `warehouse_smoke_unreal.yaml`，编辑器没开会自动在 `--render-gpu` 那张卡上开。

## 7. 第三视角回放视频

采集和评测时仿真都不录第三视角。每局的状态轨迹（采集是 `trace/<局>_state_trace.npz`，闭环评测是 `eval/state_trace/`）记录了每次仿真发布时的全部关节位置，回放它就能事后从任意机位出图。这和第 3 步的相机是两回事：数据集相机是策略要看的图，一次运行只能由一个渲染器填；回放视频只给人看，同一次采集的轨迹可以分别用 MuJoCo 和 Isaac 渲，互不影响，也不用重采。

**用所选渲染器画（一条命令）。** 第三视角录像是可选的，别的步骤都不需要它；要看的时候跑：

```bash
bash scripts/render_third_person.sh outputs/runs/warehouse_smoke --gpu 1
```

做什么：按实验文件的 `images.source` 选渲染器（Isaac、Unreal 或 MuJoCo，Isaac 和 Unreal 用和补渲、镜像同一套画面），给每个成功局画一段第三视角录像：机位和下面 MuJoCo 回放相同（四边线场景每局按走入的那条边选），半透明品红夹爪是专家给的抓取目标，三个小窗是第一视角和双腕。补渲过的局，小窗直接取数据集里补渲好的画面；没补渲的局，先由同一个渲染器把小窗渲出来。`--gpu` 是 Isaac 或 MuJoCo 用的卡；Unreal 用第 1 步起的编辑器，所在的卡由 `start_unreal.sh` 定。
参数：`--eval` 渲闭环评测的轨迹（第 6 步）而不是采集的；`--all` 连失败局一起渲；命令末尾写局名只渲那几局；`--dry-run` 只打印要渲哪些局、写到哪里。已经渲过的局会跳过，重跑只补缺的。
得到什么：`videos_isaac/<局>.mp4`（Unreal 是 `videos_unreal/`，MuJoCo 是 `videos/`；加 `--eval` 时在 `eval/` 下面），`previews_isaac/` 里是压到每个文件 500 KB 以内的预览。要让录像随数据集导出，在第 4 步转换之前跑。冒烟实测（2026-09-30）：Isaac 11 局 7.5 分钟（每局 36–44 秒，每帧 0.11 s），Unreal 8 局 3.6 分钟（每局 22–31 秒）；没补渲的局另加小窗，Isaac 每局约 75 秒；Isaac 启动另需约 20 秒。

下面两条命令是另外两种回放。

MuJoCo，先看动作对不对：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/warehouse_smoke
```

做什么：不起仿真栈，加载运行目录里的场景副本，把每条轨迹的关节位置逐帧写回 MuJoCo，用场景的录像相机出图，每局按走入的那条边选机位；有 `<局>_ghost.npz` 的局，画面里半透明的品红夹爪是专家给的抓取目标。成功和失败的局都渲。
得到什么：`videos/<局>.mp4`，每条轨迹一个。冒烟的 20 局约 45 秒。渲评测的轨迹：加 `--traces outputs/runs/warehouse_smoke/eval/state_trace --out outputs/runs/warehouse_smoke/eval/videos`。

Isaac，和数据集同一套画面，左 MuJoCo 右 Isaac 并排：

```bash
bash scripts/render_isaac_videos.sh outputs/runs/warehouse_smoke --gpu 1 --insets
```

做什么：先由 MuJoCo 渲跟踪机位并记下相机轨迹，再让 Isaac 在这个场景的 stage 上沿同一条相机轨迹渲，最后并排拼起来；`--insets` 另加一遍第一视角和双腕小窗，时间翻倍。默认只渲成功局（按 manifest 判），`--all` 全渲，命令行末尾写局名只渲那几局；`--stride 4` 是每 4 行轨迹渲一帧。渲评测的轨迹就是第 6 步最后那条命令（`--traces .../eval/state_trace --out .../eval`）。
得到什么：`isaac/<局>/side_by_side.mp4` 原片，`isaac/<局>/ego_isaac.mp4` 只有 Isaac 的第一视角，`previews_isaac/` 里是压到每个文件 500 KB 以内、可直接发的预览，长的局切成几段。1280×960、2× 超采样、带 `--insets` 时，一局 24 秒的采集局约 4 分钟（2026-09-30 实测），耗时和局长成正比。

Unreal：用上面那条一条命令的录像，没有和 MuJoCo 并排的回放。`scripts/experiments/unreal_trace_video_20260928.py` 能在编辑器里按轨迹渲单局并和 MuJoCo 并排，但它针对 2026-09-28 的手工导入（要 `--trace-joints` 关节表，导入里要有对应相机，先关 Nanite），不是这里的一步；要用先读它的文件头。

一张卡的机器：仿真栈关掉以后，上面两条命令的 `--gpu`、`MUJOCO_EGL_DEVICE_ID`、`CUDA_VISIBLE_DEVICES` 都写 0 即可。

## 8. 看结果

路径相对于 `outputs/runs/<名字>/`：

| 看什么 | 在哪里 |
|---|---|
| 每一步的汇总（保留数、成功率、checkpoint） | `summary.json` |
| 采集每局的成败和原因 | `manifest_NN.jsonl` 的 `ok` 和 `why` |
| 采集录像 | 所选渲染器画的 `videos_isaac/` 或 `videos_unreal/`；回放：`videos/`（MuJoCo）、`isaac/<局>/side_by_side.mp4` 和 `previews_isaac/`（Isaac）；都是第 7 步 |
| 闭环每局的成败、失败阶段 | `eval/rounds_00.log` 每局一行；`eval/closed_loop.manifest.jsonl` |
| 闭环录像 | `eval/previews_isaac/` 或 `eval/previews_unreal/`（第 7 步的命令加 `--eval`） |

## 9. 常见报错

| 报错 | 处理 |
|---|---|
| `not enough headroom` | 机器上的 CPU、进程数或显存不够，脚本拒绝启动。等别的任务结束（[运行机制](running.md#6-并行)） |
| 第二条命令刚开第一局就退出，终端只有一个 `Killed` | 内存不够，被内核杀掉了：通道开多了（见第 2 步“通道数”）。先 `bash scripts/stop_stack.sh --instance N` 收掉还挂着的栈（N 是起栈时打印的实例号），再用 `--lanes 1` 重来；中断的批会从头再跑 |
| 转换报 `still hold black placeholder frames` | 漏了第 3 步补渲 |
| 评测每局报 `mirror: no reply from the mirror` 或 `camera/state streams never became ready` | 镜像没起来或还没就绪。看终端 2，等它打印 `isaac mirror listening on`（Unreal 是 `unreal mirror listening on`）再起终端 3 |
| `an Unreal editor is already running` 后面跟着 `not` 和卡号 | 编辑器开在另一张卡上。`bash scripts/stop_unreal.sh` 后用要的卡重开 |
| 第三视角命令报 `has no third-person camera and ghost grippers` | Unreal 关卡建于 2026-09-29 加入第三视角之前：`bash scripts/build_scene.sh <场景> --unreal` 重建 |
| `cannot use images.source unreal` | 这个场景没有 Unreal 关卡（`configs/scenes.yaml` 的 `unreal` 条目写了原因），只有物流房间的场景能用 |
| `config.yaml differs from` | 同名运行的实验文件被改过。加 `--name` 换个运行名 |
| 失败大多是 `reach_stalled`、`lost_attachment` | 这是这条线目前主要的失败形态（手停在箱沿上方、合爪后脱开），不是环境问题 |

更多报错和评测前要核对的项见 [warehouse 详解](warehouse_design.md)。

## 10. 只有一张卡

Isaac 和 Unreal 都要第二张卡。只有一张卡时，把实验文件复制一份，`images.source` 改成 `mujoco`，策略就改看 MuJoCo 渲的图。这时第 1 步不跑第三条；第 3 步换成 `bash scripts/render_dataset_cameras.sh outputs/runs/<名字> --gpu 0`；闭环评测不起终端 2 的镜像；`run_overnight.sh` 去掉 `--render-gpu 1`。训练集和评测必须用同一种图。

## 11. 接下来

- 完整规模：把 `warehouse_smoke` 换成 `warehouse_200`（200 条数据、1 万步训练、100 局评测），Unreal 是 `warehouse_200_unreal`。四边随机还没有按这个规模跑过，按 17/40 估计，保留 200 条约需 470 局。只从一个地方开始的 `box_room_grasp_200_cap120_20260925`（当时的专家，刹车后准备手臂较慢，成功要求后退 0.3 m）实测：950 局保留 204 条（产出率 0.22），一局约 45 秒；训练到 5000 步停下，开环 MSE 0.00123，Isaac 镜像下闭环 100 局墙钟 81、lockstep 88（[双臂台账](ledger/crate.md) 2026-09-27）。
- 调专家参数做对照实验：从 `box_room_grasp_baseline_20260924` 复制，做法见 [warehouse 详解第 7 节](warehouse_design.md#7-做对照实验)。
- 台上纸箱、四边线、Blender 场景这些支线：见 [warehouse 详解第 8 节](warehouse_design.md#8-相关的支线)。
- 每一轮的实测数字：[双臂台账](ledger/crate.md)。
