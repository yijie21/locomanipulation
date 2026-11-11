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

## 5. 常见错误

- 成功数比预期低一大截：检查 `BFM_ROOT=odo` 和 `OMP_NUM_THREADS=1` 是否设了。模型是在腿部里程计下训练的，线程数会改变仿真结局。
- 后台跑到 10 分钟被杀：用 `nohup setsid` 启动，不要用工具自带的后台模式。
- 第一局和后面的局结果不一致：ScaleBFM 启动后第一次调用和之后不同，脚本已经先空跑一次，自己写新脚本时要照做。
- 伺服版本摔倒偏多：已知问题，5 到 9 局对 2 局，原因没查清。
