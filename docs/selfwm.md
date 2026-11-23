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

得到什么：每个数据目录下 `all.npz` 是原始记录，`arm.npz` 是碰撞球，`via_dev.npz` 是每条路线的偏差。`via_dev.py` 打印按站姿、弯数、速度分组的中位数；回放那份逐条打印方向稳定度，1 是每遍都往同一边偏，0 附近是没有规律。`via_cover.py` 逐条打印真实路线比百分之多少的随机路线更孤立，不超过 95 % 就算盖住；试跑前定的中位数规则的结果也一起打印，只作记录。数字怎么判、什么结果该停，写在[方案](handoff/notebook-20261120/selfmodel_via_20261120.md)里，第一次正式启动后不再改。真实路线目前只在 4090 上，回放和覆盖检查要先把它们拷到跑的那台机器。

**预测整条手臂实际扫过哪里（多路点自身模型的训练和评测，2026-11-21）。**

```bash
V=scripts/experiments/selfmodel_via_20261120; F=$I/fresh_routes
env $X REACH_VIA=1 REACH_VIA_STAND=0.3 REACH_VIA_LONG=0.5 REACH_ID0=3100000 REACH_OUT=$I/data_via2 \
    $PY $R/gen_reach.py run --procs 4 --resets 10000 --seed 39000000 --gpus 1
third_party/SIMPLE/.venv/bin/python -I $V/fresh_routes.py $PWD/third_party/SIMPLE $F/stand_start.json $F 150 20261121
env $X REACH_VIA_REPLAY=$F/via:$F/prep REACH_ID0=3200000 REACH_OUT=$I/data_via_fresh \
    $PY $R/gen_reach.py run --procs 2 --resets 252 --seed 41000000 --gpus 1
for d in data_via data_via2 data_via_fresh; do
    REACH_OUT=$I/$d $PY $R/gen_reach.py merge && $PY -I $V/arm_spheres.py $I/$d $SC && $PY $V/via_model.py build $I/$d
done
$PY $V/via_point.py train $I/via_point $I/data_via $I/data_via2
$PY $V/via_point.py dev $I/via_point
$PY $V/via_point.py eval $I/via_point $I/data_via_fresh $I/data_via_replay
```

做什么：`data_via` 是上一段命令把开场数换成 12000 采的那份。第一条补采 10000 次，其中三成不走路、从站立直接出发，一半的路线是又长又多弯的那种。`fresh_routes.py` 在 SIMPLE 的环境里用 cuRobo 规划一批新的真实路线：越过桌沿、越过桌上的盒子、伸到板子下面三种场景，起点是 `stand_start.json` 里录下的站姿；规划不出来的场景照实记下，不重抽。第三条把这些路线在空地上各回放 4 遍。`via_model.py` 的 build 把每条路线整理成一行：参考规划扫过的路线、实际扫过的路线和两者之差。`via_point.py` 的 train 训练逐点预测的模型，一个网络管路线上的每一个点；`dev` 只读验证集，打印停止线那一类路线上的结果；`eval` 读测试集和真实路线，模型定下来之后只跑一次。

得到什么：`eval` 对每个测试集打印三种预测的扫过区域误差，分别是照参考走、所有路线减同一条平均偏差、逐点模型，以及降幅和它的区间、各自要留的半径。2026-11-21 的结果：随机路线留出 11.02 cm 降到 3.50 cm；新规划的 63 条真实路线里，21 条长绕行路线降 61 %。`via_model.py` 的 train 和 eval 是第一轮把整条路线一次交给网络的模型，在真实路线上没有好处，留着作对照。假设、判据、原始输出和复核原文都在[方案](handoff/notebook-20261120/selfmodel_via_20261120.md)里。

**把预测用回去：按预测反向挪路点（2026-11-22）。**

```bash
env $X REACH_VIA_REPLAY=$F/via:$F/prep REACH_VIA_COMP=$I/via_point REACH_ID0=3300000 REACH_OUT=$I/data_via_comp \
    $PY $R/gen_reach.py run --procs 2 --resets 252 --seed 41000000 --gpus 1
REACH_OUT=$I/data_via_comp $PY $R/gen_reach.py merge
$PY -I $V/arm_spheres.py $I/data_via_comp $SC && $PY $V/via_model.py build $I/data_via_comp
$PY $V/via_comp_eval.py $I/via_point $I/data_via_fresh $I/data_via_comp
```

做什么：`REACH_VIA_COMP` 指向训练好的逐点模型。每条回放的路线在下命令之前，先让控制器只规划不执行，用模型预测手掌实际会走到哪，再把每个路点和终点反向挪这么多，重复两次，每个点最多挪 8 cm（`via_comp.py`）。随机种子和上一段直接发命令的那次回放相同，所以两份数据里同一次回放的起始状态逐位相同。`via_comp_eval.py` 把两份数据配成对，比较实际扫过的路线离原命令想要的路线有多远。

得到什么：先打印三项核对，分别是运行时算出的想要的路线、运行时的预测、事先替补偿命令做的规划，和离线算的是否一致，三项都应当是 0.00 mm。然后按路线类型打印手掌和整条手臂两组的误差中位数、降幅和区间、手最后停的位置、每个点挪了多少。2026-11-22 的结果：21 条长绕行路线上手掌 4.83 cm 降到 2.99 cm，终点 3.01 cm 降到 0.77 cm；肩、肘、腕没有改善，因为命令里只有手的路点。

**每种做法能安全穿过多窄的缝（用空地回放来算，2026-11-22）。**

```bash
env $X REACH_VIA_REPLAY=$F/via:$F/prep REACH_VIA_COMP=$I/via_point REACH_VIA_COMP_MODE=const REACH_ID0=3400000 \
    REACH_OUT=$I/data_via_const $PY $R/gen_reach.py run --procs 2 --resets 252 --seed 41000000 --gpus 1
REACH_OUT=$I/data_via_const $PY $R/gen_reach.py merge
$PY -I $V/arm_spheres.py $I/data_via_const $SC && $PY $V/via_model.py build $I/data_via_const
$PY $V/via_gap.py $I/via_gap $I/via_point $I/data_via_fresh $I/data_via_const $I/data_via_comp
```

做什么：第一条是对手"常数修正"的回放，每条路线的路点都按训练集的平均偏差挪同样多，不看具体路线。`via_gap.py` 把缝定义成：规划器想让每个球走的路线两侧 c 以外就是障碍，手指、手掌、腕任何一个球偏出去超过 c 就算碰到。它比较五种做法：小固定余量、标定过的固定余量、学出来的标量半径、常数修正、按逐点模型挪路点并在预测会碰时放弃。所有余量和半径都只用随机路线定，真实路线不参与。

得到什么：每种做法在缝的半宽 2 到 15 cm 下的成功率和碰撞率，以及"能安全通过的最窄的缝"，即成功不少于 90 %、碰撞不超过 5 % 的最小半宽，带区间。2026-11-22 的结果：21 条长绕行路线上，小固定余量 12.5 cm，按模型挪路点 11.25 cm，区间重叠；后者在 8 到 11 cm 几乎不碰撞，前者碰 14 % 到 51 %；另外三种做法 15 cm 以内成功率到不了 90 %。

**手腕朝向也补偿，以及把补偿器自己的回放放进训练（2026-11-23）。**

```bash
SIMPLE=$PWD/third_party/SIMPLE
for s in "train_routes 1500 20261123" "dev_routes 300 20261122"; do set -- $s; mkdir -p $I/$1
    cp $F/stand_start.json $I/$1/ && CUDA_VISIBLE_DEVICES=1 third_party/SIMPLE/.venv/bin/python -I $V/fresh_routes.py $SIMPLE $I/$1/stand_start.json $I/$1 $2 $3; done
T=$I/train_routes/subset; D=$I/dev_routes/detour     # 自己挑出来的路线子集，各有 via/ 和 prep/
M0=$I/via_point_20261122; M1=$I/via_point_onpol1; M2=$I/via_point_onpol2
replay() {  # replay <输出目录> <路线目录> <开场数> <种子> <REACH_ID0> [补偿用的模型 位置还是朝向]
    env $X REACH_VIA_REPLAY=$2/via:$2/prep REACH_ID0=$5 REACH_OUT=$1 ${6:+REACH_VIA_COMP=$6 REACH_VIA_COMP_MODE=$7} \
        $PY $R/gen_reach.py run --procs 2 --resets $3 --seed $4 --gpus 1
    REACH_OUT=$1 $PY $R/gen_reach.py merge && $PY -I $V/arm_spheres.py $1 $SC && $PY $V/via_model.py build $1; }
replay $I/onpol_r1_direct $T 1360 49000000 3900000
replay $I/onpol_r1_point  $T 1360 49000000 4000000 $M0 point
replay $I/onpol_r1_pose   $T 1360 49000000 4100000 $M0 pose
$PY $V/via_point.py train $M1 $I/data_via $I/data_via2 $I/data_via3 $I/onpol_r1_direct $I/onpol_r1_point $I/onpol_r1_pose
replay $I/onpol_r2_pose   $T 1360 49000000 4200000 $M1 pose
$PY $V/via_point.py train $M2 $I/data_via $I/data_via2 $I/data_via3 $I/onpol_r1_direct $I/onpol_r1_point $I/onpol_r1_pose $I/onpol_r2_pose
replay $I/dev_direct $D 164 47000000 3800000; replay $I/dev_point $D 164 47000000 3800000 $M0 point
replay $I/dev_pose   $D 164 47000000 3800000 $M2 pose
$PY $V/via_comp_eval.py $M2 $I/dev_direct $I/dev_pose && $PY $V/via_comp_eval.py $M0 $I/dev_direct $I/dev_point
python3 -I $V/via_vs.py $I/dev_point $I/dev_pose; $PY $V/via_parts.py $M2 $I/dev_direct $I/dev_pose; $PY $V/via_turn.py $I/dev_direct $I/dev_pose
replay $I/data_via_pose $F 252 41000000 3500000 $M2 pose && $PY $V/via_comp_eval.py $M2 $I/data_via_fresh $I/data_via_pose
$PY $V/via_gap.py $I/via_gap_pose $M2 $I/data_via_fresh $I/data_via_const $I/data_via_pose \
    --base $I/via_point --cal $I/dev_pose --seen 0,1,2,3,12,13,14,15
```

做什么：`REACH_VIA_COMP_MODE=pose` 在挪路点之外，把每个路点和终点的手腕命令按预测的朝向偏差反向转回去，每轮只走剩余修正的一半，共 6 轮，最多转 40°。只见过随机路线的模型预测不了这样转过的命令，所以要另外规划一批训练用的路线，把补偿器在上面的回放（直接发命令、只挪位置、位置加朝向）加进训练，再用新模型重跑一轮。三组路线各有分工：训练路线训练模型，开发路线定停在第几轮、定窄缝计分里我们这一方的余量，测试路线只计分。`via_vs.py` 把两组补偿回放逐对比较，`via_parts.py` 按肩、肘、腕、掌、指看模型错在哪，`via_turn.py` 看手的朝向对转过的命令实际响应了多少。`via_gap.py` 的 `--base` 让四个对手的余量仍然取自只见过随机路线的模型，`--cal` 让我们这一方的余量在开发路线的补偿回放上标定，`--seen` 把开发时看过的测试路线去掉再算一遍。`--rule margin` 把我们这一方的出发规则换成"预测的误差加一个在开发路线上定的固定余量"，`--cal-direct` 再加一个在开发路线上标定的固定余量做对手；2026-11-23 在另一批新测试路线上用这两个选项重做，比值 0.96，仍然不成立，但同样宽的缝通过率高得多（半宽 8 cm：70 % 对 45 %，碰撞 1 % 对 55 %）。随机路线上想让目标朝向也随机转开，用 `REACH_VIA_GOAL_TURN=0.7:45`；采集时某次开场被悄悄跳过、想看报错，加 `REACH_RAISE=1`。

得到什么：2026-11-23 的结果，41 条开发路线上腕、掌、指离想要的路线 7.91 cm，只挪位置 5.56 cm，两轮之后 3.79 cm。21 条测试绕行路线上 8.06 → 3.40 cm，终点朝向 19.3° → 7.1°，0 摔倒；整条手臂 24 个球没有改善（8.10 → 7.33 cm，区间含 0）。窄缝计分里我们 12.75 cm，小固定余量 12.50 cm，没有优势：我们的碰撞率处处为 0，但出发规则太保守。补偿回放里的 `jaw_err_cm`、`att_err_deg`、`ok` 是相对实际发出的命令算的，不要当任务成功率读。

**手腕只靠手臂转，以及把真障碍物放回去检查（2026-11-23）。**

```bash
M3=$I/via_point_onpol3; M4=$I/via_point_onpol4; W="REACH_VIA_COMP_CAP=0.12 REACH_VIA_COMP_CAP_TURN=60"
X0=$X; X="$X0 $W"                                          # 两轮回放用最宽的上限采
replay $I/onpol_r3_arm $T 1360 49000000 4700000 $M2 arm
$PY $V/via_point.py train $M3 $I/data_via $I/data_via2 $I/data_via3 $I/onpol_r1_direct $I/onpol_r1_point $I/onpol_r1_pose $I/onpol_r2_pose $I/onpol_r3_arm
replay $I/onpol_r4_arm $T 1360 49000000 4800000 $M3 arm
$PY $V/via_point.py train $M4 $I/data_via $I/data_via2 $I/data_via3 $I/onpol_r1_direct $I/onpol_r1_point $I/onpol_r1_pose $I/onpol_r2_pose $I/onpol_r3_arm $I/onpol_r4_arm
replay $I/dev_arm_wide $D 164 47000000 3800000 $M4 arm; X=$X0   # 下面两遍回到默认的 8 cm 和 40°
replay $I/dev_arm $D 164 47000000 3800000 $M4 arm; replay $I/dev_pose4 $D 164 47000000 3800000 $M4 pose
for d in dev_arm_wide dev_arm dev_pose4; do $PY $V/via_comp_eval.py $M4 $I/dev_direct $I/$d; done
python3 -I $V/via_vs.py $I/dev_pose4 $I/dev_arm $I/dev_arm_wide; $PY $V/via_parts.py $M4 $I/dev_direct $I/dev_arm --detour
$PY $V/via_scene.py $M4 $I/dev_routes $I/dev_direct $I/dev_point $I/dev_arm $I/dev_pose4
```

做什么：`REACH_VIA_COMP_MODE=arm` 让身体按原来的手腕朝向规划，手腕的转角只由手臂的七个关节实现（`Motor.reach` 的 `body_rot` 选项）；`pose` 是让全身逆解一起实现转角，它会拿躯干去凑，把肩带偏。`REACH_VIA_COMP_CAP`（米）和 `REACH_VIA_COMP_CAP_TURN`（度）是每个路点修正的上限，默认 0.08 和 40。`via_vs.py` 的第一个目录是基线，后面每个目录和它逐对比较，只看绕行路线，最后一行是仍偏 9 cm 以上的伸手各有多少。`via_parts.py` 加 `--detour` 只看绕行路线。`via_scene.py` 把每条路线规划时的箱子（路线目录 `prep/` 里存着）放回去，数记录下来的 24 个球有没有进箱子；第一个回放目录是基线，后面的和它逐对比较。回放是在空地上跑的，所以这是几何上的重叠，不是接触。

得到什么：2026-11-23 的结果。开发路线上（41 条，164 次伸手）同样 8 cm 和 40° 下 `arm` 3.33 cm、`pose` 3.70 cm；上限放宽到 12 cm 和 60° 反而让仍偏 9 cm 以上的伸手从 10 次变成 20 次，所以不放宽。第三批测试路线上（种子 20261125，49 条绕行路线，196 次伸手，只跑一次）：`arm` 8.66 → 3.69 cm，和 `pose` 的 3.94 cm 分不出来；仍偏 9 cm 以上的各占 14 %；窄缝比值 0.96，第三次不成立。补不动的那部分是高度问题：想要的路线上手腕要到 1.10 m 以上时，直接发命令就偏 10 cm 以上，补偿拉不回来。把箱子放回去之后次序反过来：直接发命令 45.9 % 的伸手会碰到，只挪位置 10.2 %，`arm` 26.0 %，`pose` 27.6 %，常数修正 61.7 %；想要的路线本身 6.6 %。前两批测试路线上只挪位置是 15.5 % 和 17.2 %。所以对着真障碍物用只挪位置（`REACH_VIA_COMP_MODE=point`，模型只需要随机路线），不用加朝向。SONIC 上直接发命令本来就只有 15.5 % 碰到，只挪位置没有再减少。

**换一个控制器做同样的事（SONIC，2026-11-23）。**

```bash
XS="$X REACH_CONTROLLER=sonic REACH_SONIC_NETS=allcpu"; MS=$I/via_point_sonic
env $XS REACH_VIA=1 REACH_VIA_STAND=0.3 REACH_VIA_LONG=0.5 REACH_ID0=5000000 REACH_OUT=$I/data_via_sonic \
    $PY $R/gen_reach.py run --procs 4 --resets 12000 --seed 51000000 --gpus 1
REACH_OUT=$I/data_via_sonic $PY $R/gen_reach.py merge && $PY -I $V/arm_spheres.py $I/data_via_sonic $SC
$PY $V/via_model.py build $I/data_via_sonic && $PY $V/via_point.py train $MS $I/data_via_sonic && $PY $V/via_point.py dev $MS
env $XS REACH_VIA_REPLAY=$F/via:$F/prep REACH_ID0=5200000 REACH_OUT=$I/data_via_fresh_sonic \
    $PY $R/gen_reach.py run --procs 2 --resets 252 --seed 41000000 --gpus 1
env $XS REACH_VIA_REPLAY=$F/via:$F/prep REACH_VIA_COMP=$MS REACH_ID0=5300000 REACH_OUT=$I/data_via_comp_sonic \
    $PY $R/gen_reach.py run --procs 2 --resets 252 --seed 41000000 --gpus 1
# 两组回放各自 merge、arm_spheres.py、via_model.py 的 build，同上
$PY $V/via_point.py eval $MS $I/data_via_fresh_sonic && $PY $V/via_comp_eval.py $MS $I/data_via_fresh_sonic $I/data_via_comp_sonic
```

做什么：采集、训练、评测、挪路点补偿都和 ScaleBFM 的相同，只多了 `REACH_CONTROLLER=sonic`。vastgpu 上的 onnxruntime 没有 GPU 版，所以 `REACH_SONIC_NETS=allcpu`，每次开场约 16 秒，12000 次用 4 个进程约 14 小时。`via_point.py` 的 dev 先打印验证集上的停止线，过了才看测试路线。

得到什么：2026-11-23 的结果，停止线降 49 %（线是 25 %）；留出的随机路线 14.75 → 7.63 cm；21 条测试绕行路线 8.55 → 4.08 cm；补偿后手掌 6.40 → 3.09 cm，整条手臂 8.55 → 5.17 cm，0 摔倒。AMO 用 `REACH_CONTROLLER=amo`；它的高度命令默认封顶在标称值 0.75 m（`REACH_AMO_HEIGHT_MAX`），不封顶时站着伸手有一半会自己走开，经过见 TODO 38.8.2。

## 8. 常见错误

- 视觉那几步报找不到 `sam3`：`sam3d` 环境里的安装指向一个已经不存在的目录，要像上面那样把 `third_party/sam3` 加进 `PYTHONPATH`。
- SAM 3 服务用 18832 端口，18822 是视觉线的，不要占。

- 成功数比预期低一大截：检查 `BFM_ROOT=odo` 和 `OMP_NUM_THREADS=1` 是否设了。模型是在腿部里程计下训练的，线程数会改变仿真结局。
- 后台跑到 10 分钟被杀：用 `nohup setsid` 启动，不要用工具自带的后台模式。
- 第一局和后面的局结果不一致：ScaleBFM 启动后第一次调用和之后不同，脚本已经先空跑一次，自己写新脚本时要照做。
- SIMPLE 评测把磁盘写满：每一局是一个新场景，运行时会把编译好的模型存进 `/workspace/tmp/mjb_cache`，每个约 180 MB。`episodes.py` 已经在自己的进程里关掉这个缓存，自己写新脚本时要先导入它。
- 伺服版本摔倒偏多：已知问题，5 到 9 局对 2 局，原因没查清。
