# 自身模型在公开基准上：按开关跑组合

一次运行由几个开关定下来：哪个基准，哪个全身控制器，目标和抓取从仿真真值还是从感知来，感知用单目 RGB-D 还是双目。开关的写法和 `bench-options` 分支相同，两个分支各做各的任务。这一页只有命令；为什么这样分、两种看法差在哪，见[详解](selfwm_design.md)第 12 节，代码入口见[基准的代码导航](reference/change-map-bench.md)。

现在能跑的是 SIMPLE 的四个抓取任务，控制器 ScaleBFM 和 SONIC。LocoManiLab、BiGym 2.0、HumanoidArena 在开关表里登记为计划中，选了会在开跑前被拒绝，并打印原因。

| 步骤 | 做什么 | 耗时（4090） |
|---|---|---|
| 准备材料 | 评测局、空地场景、缓存的看一眼 | 下载约 1 分钟，补整帧点云约 2 分钟 |
| 跑一组 | 四个任务各 30 局，五种伸手做法 | ScaleBFM 约 28 分钟（一张卡），SONIC 约 34 分钟（两张卡） |
| 看结果 | 每种做法、每个任务的成功局数 | 约 1 分钟 |

## 0. 每个新终端先执行

```bash
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
export SIMPLE_ROOT=$PWD/third_party/SIMPLE PYTHONPATH=src
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
```

做什么：第一行是仿真用的 Python 环境；`SIMPLE_ROOT` 指向 SIMPLE 的检出目录，不设时脚本会去找 vastgpu 上的路径；第三行是自身模型采数据时的手掌位置和参考设置，评测要和它一致。命令都在仓库根目录执行。

前提：[安装](setup.md)的第 1、2、7、11、13 步已完成。

## 1. 看有哪些组合

```bash
$PY -m locomani.bench table
$PY -m locomani.bench show --bench simple
$PY -m locomani.bench show --bench simple --bench-set controller=sonic --bench-set sensor=stereo
```

做什么：第一条打印每个基准每个开关能取哪些值，每个值是能跑还是计划中，后面跟原因，最后一行是开关之间的绑定。第二条打印默认组合解析后的样子：每个开关的值、签名、哪些是你设的、哪些是自动补的。第三条换成 SONIC 加双目。组合不成立时打印原因，返回码是 2。

得到什么：默认组合的签名是 `simple/perception-rgbd/pose-odometry/root-odometry/scalebfm`。

SIMPLE 上的开关，每个用 `--bench-set 名字=值` 改，可以写多次：

| 开关 | 可选的值 | 说明 |
|---|---|---|
| `controller` | `scalebfm`（默认）、`sonic`；`amo`、`gr00t_wbc` 计划中 | 全身控制器，参数都不改 |
| `target` | `perception`（默认）、`truth` | 感知栏或真值栏；真值栏在起点和站定后各读一次目标的真实位姿 |
| `sensor` | `rgbd`（默认）、`stereo` | 感知栏看哪一眼：复刻场景的 MuJoCo 头部相机，或 SIMPLE 的 Isaac 双目 |
| `depth_noise` | `none` | RealSense 那样的深度噪声还没移植到这个分支 |
| `self_pose` | 跟着 `target` 走 | 真值栏配仿真的骨盆，感知栏配腿部里程计；单独设一个矛盾的值会被拒绝 |
| `controller_root` | `odometry` | ScaleBFM 拿到的骨盆位置来自腿部里程计，自身模型的数据都是这样采的 |

默认值写在 `configs/bench/simple.yaml`。2026-11-20 到 11-24 记录的数都是双目那一眼，要重复它们就加 `--bench-set sensor=stereo`。

## 2. 准备材料（每台机器一次）

评测局和空地场景：

```bash
D=outputs/simple_eval_20261112
huggingface-cli download USC-PSI-Lab/psi-data --repo-type dataset --include "simple-eval/*" --local-dir $D/dl
for t in XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP; do
    mkdir -p $D/x/$t && unzip -q $D/dl/simple-eval/G1Wholebody$t-v0.zip -d $D/x/$t; done
$PY scripts/experiments/simple_eval_20261112/episodes.py --floor outputs/scene/simple_g1_sonic
```

做什么：下载 SIMPLE 官方的评测局，约 51 MB，解开四个抓取任务。最后一条在空地上放一台 SIMPLE 的 G1，计分时用它算手臂的碰撞球。

得到什么：`$D/x/<任务>/` 下各 30 局；`outputs/scene/simple_g1_sonic/scene_locomani.xml`。

缓存的看一眼。感知栏每局开局时看一眼，得到目标的点云、抓取和规划器要的整帧点云。这些事先做好存下来，评测时只读：

| 开关 | 目录 | 怎么来 |
|---|---|---|
| `sensor=rgbd` | `cloud_blind/`：目标点云和抓取 | `blind.py look`，要 SAM 3 和 GraspGen-X（[runbook](selfwm.md)第 7 节） |
| `sensor=rgbd` | `look_dump_blind/`：整帧点云，规划路线用 | 下面的 `look_frame.py`，只要仿真环境 |
| `sensor=stereo` | `cloud_eyes_final/`、`look_dump_eyes/` | `look_all.sh`，还要 Isaac、双目深度服务（[runbook](selfwm.md)第 7 节） |
| 两种都用 | `blind_f1_<任务>.json`：桌子前沿 | `blind.py run` |

这些都在 `outputs/simple_eval_20261112/` 下，不进 git。4090 上有现成的一份（路径见 [4090 的交接记录](handoff/selfmodel-4090-20261123.md)），整个拷过来最省事，五样合计约 1.2 GB，其中 `look_dump_eyes/` 占 812 MB，只跑 RGB-D 时可以不拷。`look_dump_blind/` 也可以在本机重做：

```bash
for t in XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP; do
    BFM_ROOT=odo $PY scripts/experiments/simple_eval_20261112/look_frame.py $t; done
```

做什么：照 `blind.py look` 的做法让机器人在每局起点站 1 秒，用复刻场景的头部相机取一帧，存下每个像素在机器人坐标里的点、哪些像素有深度、骨盆的估计和相机位姿。这一步不找目标，目标仍用 `cloud_blind/` 里的。

得到什么：`$D/look_dump_blind/` 下 120 个文件，每局打印一行。`cloud_cm_median` 是 `cloud_blind/` 的目标点离这一帧的点多远，2026-11-24 在 0.40 到 0.56 cm 之间；超过 1 cm 说明两者不在同一个坐标里。

## 3. 跑一个任务

```bash
$PY scripts/experiments/simple_eval_20261112/simple_bench.py TabletopGraspMP --method direct --out outputs/try/rows.jsonl
```

做什么：默认组合下跑桌面抓取的 30 局，直线伸手。每局是站 1 秒、看一眼（读缓存）、走到站位、伸手到预抓取点、慢速进入、合手、抬起；成功是 SIMPLE 的判定，目标升高 9 cm。

得到什么：开头一行 `switches simple/perception-rgbd/...`，之后每局一行。`rows.jsonl` 每局一条记录，`t_success` 是第一次成功的时刻，从开局站稳的那 1 秒结束时算起，`bench` 是这次的开关。4090 上每局约 9 秒。

换组合只加开关：

```bash
$PY scripts/experiments/simple_eval_20261112/simple_bench.py TabletopGraspMP --method direct --out outputs/try/sonic.jsonl \
    --bench-set controller=sonic --bench-set sensor=stereo
```

做什么：换成 SONIC 和双目那一眼。SONIC 的三个网络默认放 GPU（`REACH_SONIC_NETS=gpu`），要 GPU 版的 ONNX Runtime；没有时设 `REACH_SONIC_NETS=allcpu`，慢得多。`--bench-set target=truth` 换成真值栏。`--episodes 0-4` 或 `--episodes 3,7` 只跑其中几局。

## 4. 四个任务、五种伸手做法

```bash
B=scripts/experiments/simple_eval_20261112/bench_via.sh; O=$PWD/outputs/bench_try
for s in A plan B D M J; do BENCH_REPO=$PWD BENCH_GPU=0 bash $B $O/scalebfm scalebfm none $s; done
```

做什么：`bench_via.sh <输出目录> <控制器> <模型目录> <步骤>`，每一步四个任务同时跑，各 30 局：

| 步骤 | 做什么 |
|---|---|
| `A` | 直线伸手，同时录下每局伸手开始时的状态 |
| `plan` | 开局那一眼的整帧点云当障碍物，cuRobo 规划右臂绕过桌沿的路线 |
| `B` | 沿路线取几个手的路点，经我们的全身逆运动学发给控制器 |
| `D` | B 的每个路点和目标都往后 6 cm、往上 2 cm，不用模型 |
| `M` | 固定模板：后撤 8 cm，抬到预抓取高度，再伸过去 |
| `J` | cuRobo 规划的右臂关节轨迹直接当控制器的参考 |

没规划出路线的局，B、D、M、J 退回直线伸手，所以每一步都是 120 局。第三个参数只有 `C`、`E` 两步用，是多路点自身模型的目录（[runbook](selfwm.md)第 7 节）；其余步骤写 `none`。开关用环境变量 `BENCH_SET` 加，空格分开，例如 `BENCH_SET="sensor=stereo"`；控制器是第二个参数。SONIC 写 `sonic`，并设 `REACH_SONIC_NETS=gpu`：`bench_via.sh` 不设时把 SONIC 的网络放 CPU，因为 vastgpu 的 ONNX Runtime 没有 GPU 版。`BENCH_GPUS="2 3"` 把四个任务的进程轮流分到这几张卡上，SONIC 每个进程约 1.7 GB 显存。

得到什么：`$O/scalebfm/pipeline.log` 每步一行，应当看到 `arm A (GPUs 0): 120 rows, 0 failed processes`、`step 1: 120 rows, 120 obstacle files`，以及 `step 2c` 那一行有多少局有路线；`rows/<步骤>_<任务>.jsonl` 每局一行，`logs/` 是每个进程的输出，`asrun/` 是这次用的脚本副本。

## 5. 看结果

```bash
$PY -I scripts/experiments/simple_eval_20261112/bench_jref_table.py $O/scalebfm outputs/scene/simple_g1_sonic
```

做什么：第一张表是每种做法、每个任务 30 局里 45 秒内和官方时限内（BendPick 8 秒，其余 16 秒）抓起来的局数；第二张只看所有做法都走了自己路线的那些局，比较伸手途中右臂有没有穿进桌子、离规划多远；第三张是两两配对的差和区间。

得到什么：默认组合（ScaleBFM，RGB-D）2026-11-24 的数见第 6 节。

## 6. 现在的数

SIMPLE 四个抓取任务的感知栏，四个任务共 120 局，每格是 45 秒内 / 官方时限内抓起来的局数，括号里是 TabletopGrasp 的 30 局。4090，2026-11-24，每格跑一次，记录在 [simple_rgbd_20261124](handoff/notebook-20261120/simple_rgbd_20261124.md)。

| 控制器 | 看哪一眼 | A 直线伸手 | M 固定模板 | B 路点 | D 固定后撤 | J 关节轨迹 |
|---|---|---|---|---|---|---|
| ScaleBFM | RGB-D（默认） | 83 / 73（18 / 18） | 94 / 84（29 / 29） | 94 / 71（29 / 16） | 94 / 60（29 / 5） | 95 / 81（30 / 26） |
| SONIC | RGB-D | 29 / 18（10 / 10） | 24 / 13（5 / 5） | 36 / 18（17 / 10） | 39 / 17（20 / 9） | 34 / 20（15 / 12） |
| ScaleBFM | 双目 | 80 / 50（2 / 2） | 97 / 67（12 / 12） | 92 / 56（12 / 6） | 104 / 55（19 / 0） | 98 / 63（19 / 14） |
| SONIC | 双目 | 25 / 19（7 / 7） | 23 / 17（2 / 2） | 31 / 21（11 / 7） | 30 / 18（10 / 4） | 25 / 19（6 / 6） |

- 主指标是第一行的 A：直线伸手，ScaleBFM，单目 RGB-D。
  - 这一眼的深度是复刻场景渲染的理想深度，没有噪声。
  - TabletopGrasp 的 30 局都是同一个薯片罐，0、1 级布局相同，有效样本约 10 局。
- 双目两行是 [selfmodel_jref_20261124](handoff/notebook-20261120/selfmodel_jref_20261124.md) 的数。
- 两眼的总数接近，按任务差得很大：
  - BendPick：RGB-D 10 局，双目 30 局；
  - TabletopGrasp 直线伸手：RGB-D 18 局，双目 2 局。
- 有路线的局，ScaleBFM 28 局、SONIC 29 局，几乎全在 TabletopGrasp；其余的局 M、B、D、J 都退回直线伸手。
- 固定后撤（D）在 RGB-D 上只是 45 秒内和 B 打平：有路线的 28 局，45 秒内两者都是 27；官方时限内 D 只有 3 局，B 有 14 局。双目上 D 比 B 多的那几局，换到 RGB-D 没有重现。
- SONIC 的 B、D、J 之间的差在噪声之内：同一个目标位姿的两局，结果常常不同。
- S（J 加自身模型的终点检查，[selfmodel_s_20261124](handoff/notebook-20261120/selfmodel_s_20261124.md)，RGB-D）：ScaleBFM 88 / 77（23 / 22），SONIC 34 / 18（15 / 10）。都不比 J 多；ScaleBFM 少的 7 局全是"修不好就放弃"。复核发现不用模型的固定偏差修正把手放得比模型还准，所以这一行还不能当模型有用的证据。

## 7. 常见错误

- `FAIL plan: no look dump`：这一眼的整帧点云不在。RGB-D 跑第 2 节的 `look_frame.py`，双目从 4090 拷 `look_dump_eyes/`。
- `REACH_CONTROLLER=... but the switches say ...`：环境里还留着旧的变量。把 `REACH_CONTROLLER`、`BFM_ROOT` 去掉，用 `--bench-set controller=` 换控制器。
- `does not support` 或 `is planned`：这个基准不支持这个值，或者还没实现，后面括号里是原因。
- 第一步就报找不到 `/workspace/code/locomanipulation-selfwm`：`bench_via.sh` 默认找 vastgpu 的检出目录，要设 `BENCH_REPO=$PWD`。
- SONIC 的进程分配显存失败：一张 24 GB 的卡上放的 SONIC 进程太多，用 `BENCH_GPUS` 分开。
- 4090 上 0 号卡常有别人的任务：开跑前看 `nvidia-smi`，空着才用，否则用 1 到 3 号卡（`BENCH_GPU=1`）。
