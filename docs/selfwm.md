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

**可选，把视觉抓取往里推。**

```bash
ROOM_GRASP_VISION_DEEP=0.03 $PY $R/room_grasp.py h --procs 6 --layouts 0,1,2,3,4,5,6 --methods feedback --tag _w100d3
```

做什么：视觉抓取沿夹爪前向往里推 3 cm，GraspGen-X 给 Dex1 的抓取偏浅约 2 cm，这是执行时的深度余量。只对 `ROOM_GRASP_VISION` 的抓取起作用。

得到什么：腕部相机 89 到 103，验收集 82 到 100。

**可选，头部相机先看，看不到再抬手。**

```bash
ROOM_GRASP_VISION_DEEP=0.03 ROOM_GRASP_HEAD_FIRST=lk100 $PY $R/room_grasp.py h --procs 6 --layouts 0,1,2,3,4,5,6 --methods feedback --tag _eye
```

做什么：前三步把姿势名换成 look100、标签换成 lk100 再跑一遍，得到头部相机在 1 m 处的抓取。每局先查头部相机有没有给出够得着的抓取，有就不抬手，没有才抬手并用 `ROOM_GRASP_VISION` 的抓取。

得到什么：约四分之三的局不抬手。成功数开发集 118，验收集 92，和每局都抬手合起来是平手。

**对照，到站位再看一次。**

`ROOM_GRASP_RELOOK=st`：前三步用 `--pose stance`、`--tag st` 得到站位上头部相机的抓取。走到站位后如果它给出够得着的抓取，就替换之前的，停点误差按真值给。开发集 99 对 103，没有增益。

**对照，第二个腕部视角。**

第一步 `--pose` 以 `wrist2` 开头时，换另一只手从另一侧再看一次，存成 `<pose>v`。两次分别过第二步，再合并：

```bash
$PY $R/vision_merge.py $GRASP_ROOM_OUT/vision --a wb2 --b wb2v --tag wb2m
```

评测加 `ROOM_GRASP_HAND_LOOK=2`。开发集 105 对 103，没有增益。

**可选，上游 VLM 指点。**

```bash
CUDA_VISIBLE_DEVICES=0 $S3 -s $R/vision_point.py $GRASP_ROOM_OUT --objects $GRASP_ROOM_OBJECTS/objects.json \
    --wrist wristb100 --head look100 --tag wbp
```

做什么：认出“左数第几个”是上游 VLM 的事，它在能看到整排物体的头部画面里给一个像素。这里直接给这个像素：头部相机看得到目标时取目标上的一点，看不到时取腕部画面里的一点。头部像素加深度得到一个三维点，腕部画面里 SAM 3 的候选中离它 6 cm 以内最近的那个就是目标，没有就在那个位置按深度切一块。SAM 3 服务要开着，头部和腕部的前两步要先跑过。之后第三步和评测照旧，标签用 wbp。

得到什么：`vision/see_wbp_L*_s*.npz`、`vision/seg_wbp.json`。找对 202 个，原来是 185 个。成功数开发集 114 对 103，验收集 106 对 100，合起来 34 好 17 坏。

**对照，两只眼睛互相核对。**

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $S3 -s $R/vision_agree.py $GRASP_ROOM_OUT --objects $GRASP_ROOM_OBJECTS/objects.json \
    --a wb100 --b lk100 --see-b look100 --tag wbx
```

做什么：腕部和头部各自认目标。两片点云中心在 6 cm 以内就用腕部的；不一致时把两个候选在头部画面里框出来编号，让 Qwen3-VL 选。两眼一致时 98 % 是对的，可以当置信度用；不一致时 Qwen3-VL 选不准，整体没有增益。

**验收集。**

```bash
GRASP_ROOM_HARD=1 GRASP_ROOM_SEED=5300000 GRASP_ROOM_OUT=$I/grasp_room_hard_ho $PY $R/grasp_room.py build --layouts 7
export GRASP_ROOM_OUT=$I/grasp_room_hard_ho
```

做什么：换种子重新摆 7 个房间。之后所有命令不变。

## 7. SIMPLE 评测集

SIMPLE 是别人的基准，机器人用它官方的 G1 加 Dex3 三指手，不用我们的 mars 加 Dex1。HITCH 论文在它的六个任务上各跑 30 局。

```bash
D=outputs/simple_eval_20261112
huggingface-cli download USC-PSI-Lab/psi-data --repo-type dataset --include "simple-eval/*" --local-dir $D/dl
for t in XMovePickTeleop XMoveBendPickTeleop; do mkdir -p $D/x/$t && unzip -q $D/dl/simple-eval/G1Wholebody$t-v0.zip -d $D/x/$t; done
BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src $PY scripts/experiments/simple_eval_20261112/pick.py XMovePickTeleop --side right --stand 0.27
```

做什么：前三行下载官方评测局并解开两个拿取任务，全部约 51 MB。`episodes.py` 把每一局的桌子、目标和干扰物体重建成 ScaleBFM 能加载的场景，机器人、物体质量和摩擦、求解器设置照 SIMPLE 的代码，需要 main 的 `third_party/SIMPLE`。`pick.py` 走过去，用 SIMPLE 自带的那一个 Dex3 抓取伸手、合三指、抬 15 cm。成功判定是 SIMPLE 的：目标升高 9 cm。物体位姿和手的位置都读仿真真值，这是带特权的基准。

得到什么：每局一行，最后一行是成功数。XMovePick 站位参数 0.27 时 30 局全成，0.30 时 25 局；XMoveBendPick 在 0.30 时 30 局全成。每局 12 到 14 秒，约 8 分钟跑完一个任务。

**从头部相机的深度出抓取。**

```bash
E=scripts/experiments/simple_eval_20261112; G=$M/third_party/GraspGenX
BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src $PY $E/see.py XMovePickTeleop
CUDA_VISIBLE_DEVICES=0 $G/.venv/bin/python $E/grasps_graspgen.py outputs/simple_eval_20261112/cloud \
    --assets-dir $G/ext/gripper_descriptions/gripper_descriptions/assets
BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src $PY $E/pick.py XMovePickTeleop --grasp graspgen --squeeze 0
```

做什么：`$M` 是 main 的检出目录。第一条让机器人在每局起点站 1 秒，用 SIMPLE 的头部相机渲染一张 640×360 的深度图，取出目标的点云；目标的像素来自仿真的分割，这是特权。第二条用 GraspGen-X 自带的 Unitree G1 三指手配置出 716 个带分数的手掌位姿，手指的张开和握拢姿势也取自这份配置。第三条按分数从高到低取第一个右手做得到的抓取，也就是手指朝前、手背朝上，然后走到让手掌目标在身前 0.27 m、偏右 0.22 m 的位置去抓。

得到什么：`outputs/simple_eval_20261112/cloud/` 下每局一个点云和一个抓取文件。XMovePick 28 / 30，XMoveBendPick 30 / 30。三条合计约 10 分钟一个任务。

**不读骨盆和手的真值。**

`pick.py` 加 `--odo`：ScaleBFM 知道的骨盆位置、走完后机器人以为自己在哪、伺服读的手掌位置，都改成关节读数加腿部里程计。起点位姿读一次，只用来定坐标系。四个拿取任务各 30 局：XMovePick 28，XMoveBendPick、BendPickMP、TabletopGraspMP 都是 30。任务名写 BendPickMP 或 TabletopGraspMP 时，解压到 `x/BendPickMP` 和 `x/TabletopGraspMP`。

**拿起、带走、放到篮子上。**

```bash
BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src $PY $E/place.py LocomotionPickBetweenTablesTeleop --odo
BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src $PY $E/place.py HandoverTeleop --odo
```

做什么：前面看和出抓取两步照旧。第一个任务在桌 1 拿起饼干盒，转身走到约 2 米外的桌 2，把盒子立在篮子上松手；第二个任务拿起后向左横移，同样放到篮子上，没有做右手交左手，SIMPLE 对这个任务只检查盒子碰到篮子。篮子在碰撞里是封口的，东西进不去，所以放置就是立在口沿上。篮子的位置和口沿高度读仿真真值。

得到什么：不加 `--odo` 时 25 / 30 和 23 / 30，加了是 9 / 30 和 18 / 30。转身加走 2 米后里程计和真实骨盆位置差 20 cm 左右。每局约 40 秒和 31 秒。

**不读任何仿真真值：`blind.py`，以及用 SIMPLE 的 Isaac 画面看一眼。**

```bash
W=$PWD; E=$W/scripts/experiments/simple_eval_20261112; D=$W/outputs/simple_eval_20261112
TAG=full bash $E/look_all.sh XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP
(cd $M/third_party/SIMPLE && .venv/bin/python -I $E/look_stats.py $D $D/look_logs \
    && .venv/bin/python -I $E/grasp_types.py $D cloud_blind cloud_eyes)
```

做什么：`blind.py` 是不读真值的那条流程，目标、抓取和桌面都从头部相机来，骨盆和手掌靠里程计和关节读数，用法写在文件开头。`look_all.sh` 只做每局开局的那一眼：机器人在复刻场景里站 1 秒，`eyes_server.py` 把这一刻的状态画进 SIMPLE 自己的 Isaac 场景，读头部左右两张图，`depth_server.py` 用双目匹配加 LingBot-Depth 得深度，SAM 3 找目标，GraspGen-X 出抓取。XMoveBendPick 这个任务 SIMPLE 已经删掉，脚本第一次跑时会从 SIMPLE 的 git 历史里取回它的任务文件。要先有开在 127.0.0.1:18832 的 SAM 3 服务，和 `blind.py look` 不加 `--eyes` 得到的 `cloud_blind/` 作对照。脚本默认用 1 号卡，`GPU=0` 可以改，其余环境变量写在脚本开头。

得到什么：`cloud_eyes/` 下每局一个点云和一个抓取文件，`look_dump_eyes/` 下是每局这一眼的中间结果，日志在 `look_logs/`。最后两条打印每个任务的深度误差、抓取的变化和抓法。四个任务找对目标 30、30、30、27 局，目标上的深度误差中位数 0.3 到 0.7 cm，机器空闲时每个任务约 5 分钟。桌面抓取的指令不说物体名，带 `--eyes` 时默认让 SAM 3 找 “object” 再取够得着的最大一个；`LOOK_ARGS="--unnamed depth"` 换回只看深度的旧规则，旧规则 30 局只找对 12 局。

**把这一眼画成视频，和两种方法的对比视频。**

```bash
cd $M/third_party/SIMPLE
MUJOCO_GL=egl .venv/bin/python -I $E/percept_film.py $D/look_dump_eyes $D/cloud_eyes $D/percept \
    "XMoveBendPickTeleop_L0_e0=走过去弯腰抓　第 0 局" "TabletopGraspMP_L0_e0=桌面抓取　第 0 局" --unnamed sam
bash $E/render_all.sh $D/film_20261119/npz $D/film/clips
.venv/bin/python -I $E/grid.py $D/film_20261119/npz $D/film/clips $D/film/grid \
    --methods direct,model --cols 2 --cell 800,450 --font 24 --bar 54
```

做什么：`percept_film.py` 按看一眼、找目标、量距离、变成三维点、出抓取、定抓取点六步，把机器人当时用的数据画出来。`render_all.sh` 把评测录下的状态逐帧画进 SIMPLE 的 Isaac 场景，四个任务分两路跑；`film_20261119/npz` 是 4090 上评测时录下的六局，在数据包里。`grid.py` 把同一局的几种方法拼成并排的一段，每格标方法名和 45 秒内有没有抓起来。

得到什么：`percept/percept_<局名>.mp4`，每局 33 秒、1280×720，加一张同名的封面图；`film/clips/` 下每局每种方法一段世界视角和一段头部视角；`film/grid/clip_A.mp4` 到 `clip_F.mp4`。这些是报告用的原片，每段 2 到 9 MB，没有按 500 KB 的预览规则压缩。

这两天的数字、用户的决定和交给 4090 的清单见 [11-19 到 11-20 的记录](handoff/simple-isaac-eyes-20261120.md)。

**一条命令走多个路点的伸手数据（多路点自身模型，2026-11-20 起）。**

```bash
V=scripts/experiments/selfmodel_via_20261120; SC=$I/scene/simple_g1_sonic
C=/path/to/curobo                                   # 4090 上是 loco_outputs/simple_bench_20261119/curobo
ROUTES=$C/via_tabletop:$C/step2,$C/box/via:$C/box/step2
PYTHONPATH=src $PY scripts/experiments/simple_eval_20261112/episodes.py --floor $SC
X="REACH_HAND_OFF=0.0415,0,0 REACH_SCENE_DIR=$SC REACH_GRASP=1 REACH_GRASP_ROOM=1"
X="$X REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 REACH_GRASP_IKREF=1 REACH_WRIST=1"
env $X REACH_VIA=1 REACH_ID0=2000000 REACH_OUT=$I/data_via_trial \
    $PY $R/gen_reach.py run --procs 6 --resets 300 --seed 31000000 --gpus 1
env $X REACH_VIA_REPLAY=$ROUTES REACH_ID0=2100000 REACH_OUT=$I/data_via_replay \
    $PY $R/gen_reach.py run --procs 6 --resets 140 --seed 33000000 --gpus 1
for d in data_via_trial data_via_replay; do
    REACH_OUT=$I/$d $PY $R/gen_reach.py merge
    $PY -I $V/arm_spheres.py $I/$d $SC && python3 -I $V/via_dev.py $I/$d
done
python3 -I $V/via_cover.py $I/data_via_trial $ROUTES
python3 -I $V/via_profile.py $I/data_via_replay
```

做什么：第一条在空地上放 SIMPLE 的那台机器人，建一次就行。`REACH_VIA=1` 让每次开场的第一段伸手沿一条随机路线走到预抓取点：路线有 0 到 3 个弯，压成最多 16 个路点，和速度、沿途的手腕朝向一起作为一条命令发下去，然后照旧慢速进入。`REACH_VIA_REPLAY` 把 cuRobo 在评测里规划出的真实路线搬到空地上重放，140 次是 35 条各 4 遍：第 1 遍原地出发，后 3 遍先走一小段。`arm_spheres.py` 用记录的关节把执行手臂的 24 个碰撞球逐帧算出来，实际轨迹和参考规划各一份。`via_dev.py` 算手和整条手臂离参考路线最远多少、往哪个方向偏，回放还会算同一条路线各遍之间方向是否一致。`via_cover.py` 检查随机路线的形状有没有把真实路线包住。`via_profile.py` 把回放时手的偏差沿路线分成 20 段，比较同一条路线各遍的整条偏差曲线；`via_dev.py` 打印的方向稳定度只比偏差最大的那一刻，在这些路线上不可靠，以 `via_profile.py` 为准。

得到什么：每个数据目录下 `all.npz` 是原始记录，`arm.npz` 是碰撞球，`via_dev.npz` 是每条路线的偏差。`via_dev.py` 打印按站姿、弯数、速度分组的中位数；回放那份逐条打印方向稳定度，1 是每遍都往同一边偏，0 附近是没有规律。`via_cover.py` 逐条打印真实路线离最近的随机路线有多远，全部不超过 1 才算盖住。数字怎么判、什么结果该停，写在[方案](handoff/notebook-20261120/selfmodel_via_20261120.md)里，第一次正式启动后不再改。真实路线目前只在 4090 上，回放和覆盖检查要先把它们拷到跑的那台机器。

## 8. 常见错误

- 视觉那几步报找不到 `sam3`：`sam3d` 环境里的安装指向一个已经不存在的目录，要像上面那样把 `third_party/sam3` 加进 `PYTHONPATH`。
- SAM 3 服务用 18832 端口，18822 是视觉线的，不要占。

- 成功数比预期低一大截：检查 `BFM_ROOT=odo` 和 `OMP_NUM_THREADS=1` 是否设了。模型是在腿部里程计下训练的，线程数会改变仿真结局。
- 后台跑到 10 分钟被杀：用 `nohup setsid` 启动，不要用工具自带的后台模式。
- 第一局和后面的局结果不一致：ScaleBFM 启动后第一次调用和之后不同，脚本已经先空跑一次，自己写新脚本时要照做。
- SIMPLE 评测把磁盘写满：每一局是一个新场景，运行时会把编译好的模型存进 `/workspace/tmp/mjb_cache`，每个约 180 MB。`episodes.py` 已经在自己的进程里关掉这个缓存，自己写新脚本时要先导入它。
- 伺服版本摔倒偏多：已知问题，5 到 9 局对 2 局，原因没查清。
