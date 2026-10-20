# 任意物体抓取：离线抓取缓存或 graspnet

> **已归档（2026-10-20）**：VLA 数据采集与训练这一系列暂不使用，这一页不再随代码更新，命令可能已经跑不通。现在的主线是[场景 RSI](rsi.md)。

机器人站在桌边不走路，用右手抓起桌上的一个物体，抬起 10 cm 并停住。物体是 SIMPLE 的 GraspNet 物体集里能放进 Dex1 夹爪的 65 个，每个物体一份场景，物体放在 bottle 场景水瓶的位置。
抓取位姿有两个来源，默认是离线抓取缓存。为什么默认用缓存、两个来源的实测对比，见[任意物体抓取详解](object_grasp_design.md)；名词见[术语表](reference/glossary.md)。整条线只用 1 张卡，选 graspnet 时它的服务再占一张卡上的约 1.6 GB 显存。

## 先选抓取来源

两个来源只在第 2 步的准备和第 3、4 步命令末尾的一个参数上不同，其余步骤一样。

| | 离线抓取缓存（默认） | graspnet |
|---|---|---|
| 抓取从哪来 | 事先在物体网格上采样抓取，在只有一只手的仿真里逐个合爪、抬起、晃动，只留抓住的；每局按测得的物体位姿取用 | 每局从头部相机渲一张深度图，交给 graspnet-baseline 网络现场预测 |
| 一次性准备 | 第 2.1 步建缓存，约 5 分钟 | 第 2.2 步装 graspnet-baseline，约 2 分钟加下载时间 |
| 每次跑之前 | 不用开服务 | 先开 graspnet 服务 |
| 第 3、4 步的命令 | 不加参数 | 末尾加 `--grasp-source graspnet` |
| 65 个物体各一局，抬起几个 | 默认配置两批各 32、33 个 | 16 个；那是 2026-10-08 的执行逻辑，换成现在的执行逻辑后没有重测 |

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
```

做什么：设置路径变量（下面用到 `$VENV_SIM_PY`），把 `src/` 加进 `PYTHONPATH`，并读入本机的 `locomani.local.sh`。下面的命令都在仓库根目录执行。
前提：[安装](setup.md)第 0–5 步和第 7 步（SIMPLE 仓库）已经完成，bottle 的场景已经建好（[bottle](single_arm.md)第 1 步）。

## 1. 建物体场景（每台机器一次）

```bash
bash third_party/SIMPLE/data/_download.sh --only assets
NUMPY_MADVISE_HUGEPAGE=0 "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/make_scenes.py outputs/graspnet_bottle_20261008
"$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/hand_cloud.py outputs/graspnet_bottle_20261008
```

做什么：第一条下载 SIMPLE 的 GraspNet 物体（约 0.5 GB，已下载过会跳过）。第二条为每个能放进夹爪的物体复制一份 bottle 场景，把水瓶换成这个物体（真实尺寸，按体积算质量），先让它在桌上自由落 2 s，按停下来的姿态重建场景。第三条存下夹爪全开时手的表面点，规划时用来查手会不会碰到桌子和物体。
得到什么：`outputs/graspnet_bottle_20261008/scenes/obj_000/` 到 `obj_074/` 共 65 个场景目录、物体清单 `objects.json`、`hand_open_points.npy`。第二条最后打印 `65 scenes; 10 too wide for the jaw or never at rest`，约 1.5 分钟。
注意：重跑第二条会删掉并重建每个场景目录，第 2.1 步建的缓存也一起删掉，要重建缓存。

## 2. 准备抓取来源

### 2.1 离线抓取缓存（默认来源）

```bash
bash scripts/experiments/graspnet_bottle_20261008/cache_all.sh 12
```

做什么：12 个进程并行，给每个物体场景建抓取缓存。每个物体在网格上采约 300 个能放进夹爪的抓取，在一个只有右手、桌子和物体的 MuJoCo 模型里逐个合爪、抬 10 cm、停 1.5 s、左右晃，记下抓没抓住和物体在手里滑了多少。不用 GPU，不起仿真栈。
得到什么：每个场景目录里多出 `grasp_cache.npz` 和 `grasp_cache_summary.json`，日志在 `outputs/graspnet_bottle_20261008/logs/cache_obj_XXX.log`。约 5 分钟。65 个物体里 64 个有抓住的抓取，中位每个物体 243 个；物体 009 一个都没有，它的局会报没有合适的抓取。
只建一个物体：`"$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/grasp_cache.py 40`，约 1 分钟。

### 2.2 graspnet（选 graspnet 来源时）

```bash
bash scripts/experiments/graspnet_bottle_20261008/install_graspnet.sh
```

做什么：只需一次。克隆 graspnet-baseline 和 graspnetAPI 到 `third_party/`，固定在验证过的提交上；给它的 CUDA 扩展打补丁让它能在 torch 2.7 上编译；建独立的 Python 3.10 环境，按本机显卡编译 pointnet2 和 knn；下载 RealSense 权重；最后加载网络，在 graspnet-baseline 自带的示例场景上预测一次。每一步做过就跳过，失败后重跑会从失败的那一步继续。需要 CUDA 12.8 工具链（`/usr/local/cuda-12.8`）。
得到什么：`third_party/graspnet-baseline/.venv`（约 8 GB）和 `logs/checkpoint-rs.tar`，最后打印 `[graspnet] selftest: 110 grasps` 和 `done`。安装包已在 uv 缓存里时从零装约 2 分钟，新机器还要下载 torch 和 CUDA 库。graspnet-baseline 和它的权重只许非商业研究使用，都不进 Git。

```bash
GPU=1 bash scripts/experiments/graspnet_bottle_20261008/servers.sh start graspnet
```

做什么：每次跑 graspnet 来源之前执行一次。在 1 号卡上起 graspnet 服务，监听 5790 端口，起好就返回，服务留在后台。
得到什么：最后打印 `[graspnet] listening on :5790`，几秒钟；日志在 `outputs/graspnet_bottle_20261008/logs/graspnet_server.log`。用完关掉：`bash scripts/experiments/graspnet_bottle_20261008/servers.sh stop graspnet`。

## 3. 跑一个物体

```bash
bash scripts/experiments/graspnet_bottle_20261008/run_one.sh 40 1 0 _demo
bash scripts/experiments/graspnet_bottle_20261008/run_one.sh 40 1 0 _demo_gn --grasp-source graspnet
```

做什么：两条任选一条，第一条用缓存，第二条用 graspnet。三个数字依次是物体编号、仿真实例、GPU。命令在实例 1、0 号卡上起自己的仿真栈，把机器人放在桌边，取抓取、用 IK 规划手臂路线、闭环进入、合爪、抬起，跑完一局后关掉仿真栈。
得到什么：运行目录 `outputs/runs/gnb_obj_040_demo/`（第二条是 `gnb_obj_040_demo_gn/`）；这一局的记录在 `grasp_planned/round_00_ik/record.json`（`ok`、失败原因 `why`、抓取来源 `source`、选中的抓取 `grasp`），录像在 `videos/round_00.mp4`。日志倒数第二行 `[plan] round 0:` 后面是 `OK` 表示抬起，是 `FAIL` 时后面跟失败原因；最后一行 `[run_grasp] object 040 done:` 后面 0 表示抬起，1 表示失败。约 75 秒。物体 040 在缓存来源的每一批里都抬起了，单局仍可能失败。

## 4. 跑全部 65 个物体

```bash
bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 4 1 0 _mine
bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 4 1 0 _mine_gn --grasp-source graspnet
```

做什么：两条任选一条。开 4 条并行通道（实例 1–4，都在 0 号卡），每条通道依次领下一个还没跑的物体，每个物体跑第 3 步那样的一局。重跑同一个名字时，已经跑完的物体会跳过。中途要停：`touch outputs/graspnet_bottle_20261008/STOP`，各通道跑完手上的物体后退出；之后要删掉这个文件才能再跑。
得到什么：65 个运行目录 `outputs/runs/gnb_obj_XXX_mine/`，每个物体的日志 `outputs/graspnet_bottle_20261008/logs/obj_XXX_mine.log`，每条通道的进度 `batch_mine_lane0.log` 等，全部跑完时打印 `batch finished`。约 20 分钟。缓存来源预计抬起 30 个上下（两批实测 32、33）。

## 5. 看结果

```bash
python3 scripts/experiments/graspnet_bottle_20261008/cache_summary.py _mine
```

做什么：读一批的每个物体的记录，把每局分成抬起、没有合适的抓取（`no_candidate`）、手臂够不到（`no_path`）、执行失败（`execution`，附失败原因），并和 2026-10-08 的 graspnet 结果并列。不用 GPU。
得到什么：`outputs/graspnet_bottle_20261008/cache/summary_mine.md`（每个物体一行）和 `summary_mine.json`，终端打印前几行总数。几秒钟。

```bash
MUJOCO_GL=egl "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/ghost_overlay.py outputs/graspnet_bottle_20261008 --tag _mine --jobs 8
"$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/best_videos.py outputs/graspnet_bottle_20261008 --tag _mine
```

做什么：第一条在每局录像里把选中的抓取画成品红色的半透明夹爪（ghost）。第二条给每个物体出一条带字幕的录像（物体尺寸、质量、抓取来源、成败和失败原因），再压成可以直接发出去的预览。
得到什么：带 ghost 的录像在 `outputs/graspnet_bottle_20261008/mine/ghost_rounds/`；全分辨率成片在 `outputs/graspnet_bottle_20261008/cache/videos_mine_full/`，每条 500 KB 以内的预览和 `index.csv` 在 `cache/videos_mine/`，文件名带 `success` 或 `fail`。第一条每局约 40 秒，8 个进程并行，65 局约 5 分钟；第二条约 1 分钟。

## 6. 常见报错

- `FAIL [plan]` 后面跟 `"check": "grasp_cache"`：这个物体的场景没有缓存。跑第 2.1 步，或者只建这一个物体。
- `FAIL [plan]` 后面跟 `"check": "graspnet_server"`：graspnet 服务没在 5790 端口应答。按第 2.2 步开服务，看它的日志里有没有 `listening`。
- `no cached grasp fits here`：缓存里没有一个抓取在这一局的物体姿态下放得进去且不碰桌子。物体 009 比夹爪还宽，每局都会这样；其他物体偶尔出现，是出生时物体被碰歪了。
- `ik planner found no path`：手臂够不到任何一个候选抓取，多见于物体 005、018、020、023 这类很矮或很靠里的物体。
- 实例被占：`run_one.sh` 和 `run_batch.sh` 用的实例如果已经有别的仿真栈在跑，会起栈失败。换一个空闲的实例号，或者先看 [STATE](../STATE.md) 里写的谁在用。
- `install_graspnet.sh` 停在编译那一步：它会打印编译日志的位置，在 `/workspace/tmp/graspnet_build_pointnet2.log` 或 `graspnet_build_knn.log` 里找第一个 `error`。最常见的原因是机器上没有 CUDA 12.8 工具链。

## 接下来

- 两个来源的实测对比、失败原因的拆解、执行层这几天加的三项改动：[任意物体抓取详解](object_grasp_design.md)。
- 每一批的原始数字和过程：[graspnet 与两个规划器的对比](handoff/graspnet-bottle-planner-comparison-20261008.md)、[离线抓取缓存](handoff/grasp-cache-20261010.md)。
