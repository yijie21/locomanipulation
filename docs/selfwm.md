# 自我模型抓取：让冻结的全身控制器抓得准

Unitree G1 加 Dex1 夹爪，全身控制器 ScaleBFM 不改。ScaleBFM 单独执行抓取命令时，手会偏 3 到 5 cm、姿态偏 10 度左右，在没见过的摆放上只抓成 19 %。这条线在控制器外面加三样东西：学出来的走路模型和抓取选择，腿部里程计，以及只读关节的连续伺服，把成功率提到 80 %。机制和取舍见[详解](selfwm_design.md)，按天的记录和每个数字的来源见[开发记录](handoff/selfwm-20261029.md)。

这一页只有命令。全部只用 MuJoCo，不开 Unreal，6 个进程。耗时是 2026-11 在这台机器上的实测。

| 步骤 | 做什么 | 耗时 |
|---|---|---|
| 准备材料 | 物体、抓取缓存、房间摆放、数据、模型 | 物体和缓存约半天，数据和训练约 5 小时 |
| 跑一组评测 | 一套摆放 147 局 | 伺服 3 到 6 分钟，落点反解约 10 分钟 |
| 看结果 | 成功率表、配对比较 | 几秒 |

## 0. 每个新终端先执行

```bash
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
R=scripts/experiments/reachwm_20261030
export I=$PWD/outputs/reachwm_20261030
export BFM_ROOT=odo OMP_NUM_THREADS=1 GRASP_ROOM_OBJECTS=$I/simready GRASP_ROOM_MIN_HELD=10
export APPROACH_WM_POST=$I/invert/wrist_graspr_odo/wm_ref.pt WALK_DATA=approach_odo,approach0_odo WALK_SUFFIX=_odo
export GRASP_ROOM_OUT=$I/grasp_room_sr
```

做什么：`BFM_ROOT=odo` 让 ScaleBFM 的骨盆位置输入来自腿部里程计而不是仿真真值；后两行指定物体集、伸手模型和走路模型；最后一行选摆放集。下面的命令都在仓库根目录执行。

前提：[安装](setup.md)里仿真环境和 ScaleBFM 权重两步已完成。

## 1. 准备材料（每台机器一次）

材料都在 `outputs/reachwm_20261030/` 下，不进 git。四类材料各由一个脚本生成，完整的阶段和参数写在脚本开头的说明里。

| 材料 | 脚本 | 得到什么 |
|---|---|---|
| 物体 | `scripts/experiments/reachwm_20261030/simready_objects.py` | NVIDIA SimReady 资产转成 MuJoCo 物体，128 个各有 10 个以上验证过的抓取 |
| 抓取缓存 | `scripts/experiments/graspnet_bottle_20261008/grasp_cache.py` | 每个物体的候选抓取，用只有手的模型合爪抬起验证过 |
| 伸手和走路数据 | `scripts/experiments/reachwm_20261030/gen_reach.py` | 冻结的 ScaleBFM 在各种命令下手实际到了哪 |
| 模型 | `scripts/experiments/reachwm_20261030/train.py`、`scripts/experiments/reachwm_20261030/walk_model.py` | 伸手模型和走路模型 |

建房间摆放：

```bash
$PY $R/grasp_room.py build --layouts 7
GRASP_ROOM_SEED=9200000 GRASP_ROOM_OUT=$I/grasp_room_test $PY $R/grasp_room.py build --layouts 7
GRASP_ROOM_SEED=9300000 GRASP_ROOM_OUT=$I/grasp_room_test2 $PY $R/grasp_room.py build --layouts 7
```

做什么：第一条建开发集，后两条建两套验收集。每套 7 个房间，每个房间 6 件家具、21 个物体，共 147 个摆放。

得到什么：`$GRASP_ROOM_OUT/layout_K/` 下的场景文件和 `layout.json`。

## 2. 跑一组评测

现在最好的配置，走路模型加抓取选择加连续伺服，伺服只读关节和腿部里程计：

```bash
export ROOM_GRASP_CONT=1 ROOM_GRASP_LIFT_CONT=1 ROOM_GRASP_PICK=8 ROOM_GRASP_PICK_ATT=1.0
export ROOM_GRASP_SERVO=1 ROOM_GRASP_SERVO_SRC=odo
$PY $R/room_grasp.py h --procs 6 --layouts 0,1,2,3,4,5,6 --methods model --tag _best
```

做什么：在当前摆放集的 147 个摆放上各跑一局。一局是看一眼物体，走过去，伸手，合爪，抬起，保持 1.5 秒。

得到什么：`$GRASP_ROOM_OUT/grasp/h_w*_best.json`，每局一条记录。约 10 分钟。开发集预期 147 局里成功 110 上下，单次运行的波动约 3 局。

对照组换 `--methods` 和开关：

| 对照 | 命令里改什么 | 验收集 294 局带重抓 |
|---|---|---|
| ScaleBFM 单独 | `--methods direct`，不设 `ROOM_GRASP_SERVO` | 65 |
| 连续伺服，不用学习模型 | `--methods feedback`，`ROOM_GRASP_FB_SERVO=1` 代替 `ROOM_GRASP_SERVO=1` | 220 |
| 落点反解加抓取选择 | `--methods model`，不设 `ROOM_GRASP_SERVO`，加 `ROOM_GRASP_PRE_FIX=2 ROOM_GRASP_ENTRY_FIX=2 ROOM_GRASP_GRIP_CHECK=1` | 215 |
| 现在最好的配置 | 上面的命令 | 235 |

验收集要换摆放集并打开重抓：

```bash
export GRASP_ROOM_OUT=$I/grasp_room_test ROOM_GRASP_RETRY=2 ROOM_GRASP_RETRY_MIN_Z=0.5
```

做什么：换到第一套验收集；夹爪读数显示没夹到东西时，在 0.5 m 以上的台面退出来再抓一次。第二套验收集把目录换成 `grasp_room_test2`。

换起点：加 `ROOM_GRASP_SEED=6200001`，同样的摆放换一组出发位置。比较两种做法时两组起点都跑。

## 3. 看结果

```bash
$PY $R/room_grasp.py d --tag _best
```

做什么：读这个标记的全部记录，打印成功数、合爪时手的误差中位数、摔倒数，以及每类家具的成功数。

比较两种做法要按摆放配对：数只有一边成功的局数，再做符号检验。同一个摆放换 1 cm 站位就会翻转十几局，所以不配对的成功数差 5 局以内不算差别。

## 4. 看过程

```bash
$PY $R/room_grasp_trace.py
MUJOCO_GL=egl $PY $R/room_grasp_video.py
```

做什么：第一条重跑指定的局，每个控制拍记录手、物体和接触；第二条把一个摆放的两种做法并排录成视频。参数见各自脚本开头的说明。

## 5. 更难的房间

```bash
GRASP_ROOM_HARD=1 GRASP_ROOM_OUT=$I/grasp_room_hard $PY $R/grasp_room.py build --layouts 7
export GRASP_ROOM_OUT=$I/grasp_room_hard
```

做什么：物体放到台面里 0.10 到 0.50 m 深，矮柜和台面上方加一块板，多一个 1.10 m 的高架。7 个房间共 203 个摆放。第二行之后，第 2、3 节的命令都在这套摆放上跑。

得到什么：`$I/grasp_room_hard/`，其中 `stations.json` 记着家具，评测时不用再设 `GRASP_ROOM_HARD`。连续伺服预期 203 局里成功 103 上下。

## 6. 从视觉得到抓取

把特权的抓取缓存换成相机看到的。四步，前三步各在自己的环境里跑，最后一步是第 2 节的评测加一个开关。下面是用腕部相机在 1 m 处看一眼的配置。

前提：main 的 `third_party/` 下有 `GraspGenX`、`sam3`、`lingbot-depth` 三份代码，`sam3d` conda 环境可用。安装和踩过的坑见[开发记录](handoff/selfwm-20261029.md)的 11-11 一节。

```bash
M=/workspace/code/locomanipulation        # main 的检出，第三方代码和 Dex1 夹爪配置在那里
S3=/workspace/miniconda3/envs/sam3d/bin/python
export HF_HOME=/workspace/huggingface_cache PYTHONNOUSERSITE=1
```

**第一步，看。**

```bash
for L in 0,1 2,3 4,5 6; do
  ROOM_GRASP_LOOK=1.0 VISION_CALIB=1 MUJOCO_GL=egl $PY $R/vision_grasp.py see --pose wrist100 --layouts $L &
done; wait
```

做什么：每个摆放上，机器人站在离物体 1 m 处，抬起要抓的那只手，用腕部相机拍一张彩色图和深度图。深度带 RealSense 式的缺陷，`VISION_CALIB=1` 再加标定残差。`--pose` 写别的名字，比如 `look100`，就改用头部相机。

得到什么：`$GRASP_ROOM_OUT/vision/see_wrist100_L*_s*.npz`。约 4 分钟。

**第二步，补深度，找物体。**

```bash
(cd $M && PYTHONPATH=third_party/lingbot-depth CUDA_VISIBLE_DEVICES=0 $S3 -s $OLDPWD/$R/vision_depth_fix.py $GRASP_ROOM_OUT/vision --pose wrist100)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:$M/third_party/sam3 HF_HUB_OFFLINE=1 $S3 -s scripts/rsi/sam3_server.py --port 18832 &
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $S3 -s $R/vision_seg.py $GRASP_ROOM_OUT --objects $GRASP_ROOM_OBJECTS/objects.json \
    --how refer --pose wrist100 --depth lb --tag w100
kill %1
```

做什么：第一条用 LingBot-Depth 把深度的洞补上。第二条在后台起 SAM 3 服务。第三条对每张图，让 SAM 3 按类别出候选，让 Qwen3-VL 按“某某，左数第几个，在哪件家具上”给框，取重合最多的候选，再把它的深度像素变成点云。最后一条停掉 SAM 3 服务。

得到什么：`vision/see_w100_L*_s*.npz` 是物体点云，`vision/seg_w100.json` 记着每个摆放找对没有。预期 203 个里找对 185 个上下。约 5 分钟。

**第三步，出抓取。**

```bash
(cd $M && CUDA_VISIBLE_DEVICES=0 third_party/GraspGenX/.venv/bin/python $OLDPWD/$R/vision_grasp_infer.py $GRASP_ROOM_OUT/vision \
    --pose w100 --assets-dir outputs/scene_rsi_20261016/graspgenx_20261026/gripper_assets)
```

做什么：GraspGen-X 用 Dex1 的夹爪配置，从每个物体的点云出 716 个带分数的抓取。

得到什么：`vision/grasp_w100_L*_s*.npz`。约 1 分钟。不要在 `GraspGenX` 目录里跑 `uv sync`，它会装回不支持这块显卡的旧 torch。

**第四步，抓。**

```bash
export ROOM_GRASP_CONT=1 ROOM_GRASP_LIFT_CONT=1 ROOM_GRASP_FB_SERVO=1 ROOM_GRASP_SERVO_SRC=odo
export ROOM_GRASP_LOOK=1.0 ROOM_GRASP_HAND_LOOK=1 ROOM_GRASP_VISION=w100
$PY $R/room_grasp.py h --procs 6 --layouts 0,1,2,3,4,5,6 --methods feedback --tag _w100
$PY $R/room_grasp.py d --tag _w100
```

做什么：每局先抬手看、放下手，再走过去抓，抓取用第三步的结果。之后不再看。

得到什么：203 局里成功 89 上下。同样条件下特权抓取是 104，去掉 `ROOM_GRASP_VISION` 就是它。

**可选，走路的停点改用视觉估计。**

```bash
ROOM_GRASP_WALK_EST=vis MUJOCO_GL=egl $PY $R/room_grasp.py h --procs 6 --layouts 0,1,2,3,4,5,6 --methods feedback --tag _w100w
```

做什么：走路前后各用头部相机拍一张带缺陷的深度图，把两片家具点云配准，得到实际走了多远，代替腿部里程计的估计。`ROOM_GRASP_WALK_EST=log` 只记录不使用。

得到什么：特权抓取从 104 到 118，不抬手时 116 到 125。整条视觉链不变，83 对 89。

## 7. 常见错误

- 视觉那几步报找不到 `sam3`：`sam3d` 环境里的安装指向一个已经不存在的目录，要像上面那样把 `third_party/sam3` 加进 `PYTHONPATH`。
- SAM 3 服务用 18832 端口，18822 是视觉线的，不要占。

- 成功数比预期低一大截：检查 `BFM_ROOT=odo` 和 `OMP_NUM_THREADS=1` 是否设了。模型是在腿部里程计下训练的，线程数会改变仿真结局。
- 后台跑到 10 分钟被杀：用 `nohup setsid` 启动，不要用工具自带的后台模式。
- 第一局和后面的局结果不一致：ScaleBFM 启动后第一次调用和之后不同，脚本已经先空跑一次，自己写新脚本时要照做。
- 伺服版本摔倒偏多：已知问题，5 到 9 局对 2 局，原因没查清。
