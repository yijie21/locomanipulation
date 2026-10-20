# 术语表

各 runbook 直接使用下面这些词，不再逐页解释。每个词一行：意思，以及在仓库里的落点。

## 场景 RSI

| 词 | 意思 |
|---|---|
| 技能目录（catalog） | `src/locomani/tasks/scene/skills/`：拿起、放下、放进容器等技能和把目标翻成计划的 `plan_for`。内环改的就是它；种子是第一次写回之前那一版 |
| 任务族（family） | 一组要一起改进的任务和它们的三个起点池，一个文件 `scripts/rsi/families/<名字>.json`；一个内环只跑一个任务族 |
| 起点 / 起点池 | 起点：机器人位姿加物体摆放的一个 JSON。诊断池 50 个（角色看得到）、验收池 60 个（只有验收门用）、封存集 80 个（内环结束后跑一次） |
| 现任 / 候选 | 现任（incumbent）：最近一次被接受的技能目录加参数；候选（candidate）：Engineer 交出的新版本 |
| 验收门（gate） | `gate.py`：候选和现任在验收池上逐个起点配对比较，按固定规则判接受或拒绝 |
| 内环 / 外环 | 内环：一个任务族里反复“看记录、提假设、写代码、验收”。外环：发现任务、分桶、选缺口、为每个缺口建任务族跑内环、写回 |
| 分桶 / 缺口 | 分桶：程序判一个任务现有技能能做（第 1 桶）、缺点什么（第 2 桶）还是做不了（第 3 桶）。缺口：第 2 桶里缺的那一项 |
| 沙盒（sandbox） | 一个任务族专用的代码副本（`/workspace/code/rsi-scene/<名字>`），从当前提交导出，内环在里面跑 |
| 搁置（park） | 声明“现有动作做不到”的起点，经程序检查和闭环尝试成立后不再当修理目标，但仍算失败 |
| 写回 / 回归 / 登记表 | 写回：把内环最终的现任合并进主仓库的技能目录。回归：用当前技能目录重跑登记表 `outputs/scene_rsi_20261016/outer/registry.json` 里每个任务族的封存集，和上次逐局比较 |
| 标定样本 | 从没练过的任务里按桶抽的样本（cal_r2–cal_r4），量检查器判“能做”的准确率 |
| 能力表 | `configs/scene_discovery/capability_mars_scalebfm.yaml`：动作层探针和各技能封存集的实测范围，分桶读它 |
| 复现材料（kit） | `rsi_kit/`：大模型生成或在当时的场景上抽出的输入（任务清单、起点池、场景文件、逐局结局），`scripts/rsi/kit.py restore` 放回 `outputs/` |

## 运行与产物

| 词 | 意思 |
|---|---|
| 线（line） | 一种任务从建场景到评测的整套流程。现在的主线是场景 RSI（[runbook](../rsi.md)）；2026-10-20 归档的 VLA 线有三条：bottle（单臂抓瓶）、crate（双臂抓筐）、warehouse（双臂抓纸箱），每条一页 runbook（[README](../../README.md#归档vla-数据采集与训练2026-10-20-起暂不使用)）。脚本只分两套：`scripts/single_arm/` 给 bottle，`scripts/bimanual/` 给 crate 和 warehouse（`run_overnight.sh --line bimanual`） |
| 实验文件 | `configs/experiments/<name>.yaml`，一次运行读的**全部**配置，没有继承；新实验 = 复制一份改几行 |
| `locomani.local.sh` | `$LOCOMANI_WORKSPACE/locomani.local.sh`：这台机器专用、不进 Git 的设置（例如 deploy 编在别处时的 `SONIC_DEPLOY_ROOT`），`scripts/paths.sh` 先读它 |
| 场景名 | 实验文件里的 `scene.name`，在 `configs/scenes.yaml` 里查到场景目录和构建命令；`bash scripts/build_scene.sh <场景名>` 按它建场景 |
| 运行目录 | `outputs/runs/<name>/`。开跑时把实验文件逐字节复制成 `config.yaml`，之后每个阶段只读这份副本 |
| 阶段（stage） | `collect`（采集）→ `convert`（转成数据集）→ `train`（微调）→ `eval`（评测）。产物已存在的阶段自动跳过，重跑就删掉它的产物 |
| 局 / 批 | 局（round、episode）：从重置到判成败的一次尝试。批：连续若干局（`collect.batch_rounds`），录制时对应一个 session |
| manifest | `manifest_NN.jsonl`，每批一个，每局一行：成败 `ok`、失败原因 `why`、起点 `placement` 等 |
| trace（state trace） | `<局>_state_trace.npz`：每个控制帧（50 Hz）的 `seq`、`qpos`、`qvel`。事后录像、补渲、Isaac 重渲都从它来 |
| ghost | 专家命令的抓取目标，录像里画成半透明品红夹爪；只在第三视角，不进训练图 |
| case bank | 一组固定的开局条件（`scene.cases`），对照实验用它保证两边跑的是同一批 case |
| holdout | `train.holdout`：转换时留作验证集（`val_dataset/`）的集数，不参与训练 |
| 冒烟（smoke） | 用最小的规模把每一步都跑一遍，只检查流程通不通、产物齐不齐，不衡量效果。三条主线的冒烟实验文件是 `bottle_simple_smoke`、`crate_smoke`、`warehouse_mars_smoke` |

## 仿真栈里的进程

| 词 | 意思 |
|---|---|
| 栈（stack） | 仿真时常驻的一组进程：sim + deploy，录制时再加 exporter。`start_stack.sh` 起（起完即退，栈留在后台），驱动脚本跑完替你收，异常时 `stop_stack.sh --instance N` 手动收 |
| 实例（instance） | 一个编号对应一组端口和一个 exporter 目录；`collect.instance` 是第一个栈用的实例。不渲染的仿真一张卡最多 4 个，实时渲染的 1 个 |
| 通道（lane） | 双臂线并行采集的单位：一个栈加一个跑局进程，各占一个实例。`start_stack.sh collect` 按空闲资源定开几条，写进 `collect/lanes.json`，起栈和开跑前都会打印（[运行机制](../running.md#6-并行)） |
| sim | MuJoCo 仿真进程（`src/locomani/sim/run_sim_locomani.py`）：跑物理，发布状态和（`live` 时）相机图 |
| SONIC | NVIDIA GR00T-WholeBodyControl 里的人形全身控制策略；默认 checkpoint `policy/sonic_v1_1`（[安装](../setup.md)第 2 步） |
| stock、mars | 仿真里的两种 G1 模型。stock 是 SONIC 自带的 G1 接上 Dex1 夹爪；mars 是实验室真机 G1，按协作者给的 URDF 生成，带头顶云台 D455（第一视角，也可换成脸上的 D435）和腕部 D405 相机。场景名带 `_mars` 的是 mars，warehouse 线 2026-10-03 起默认用它（[warehouse 详解](../warehouse_design.md#11-mars-机器人)） |
| deploy | SONIC 的 C++ 控制器（`g1_deploy_onnx_ref`，安装第 3 步编译并打补丁）：把运动指令变成关节目标发给 sim |
| planner / POSE / token | deploy 的三种输入。planner：方向、速度、模式（站、走、蹲）这类高层指令，由 deploy 内的规划器生成全身动作；POSE：直接流式给全身姿态参考，专家伸手、抓取时用；token：SONIC 的 64 维离散运动码，VLA 的动作就是 token + 夹爪 |
| initial_token | `eval.initial_token`：策略接管那一刻 deploy 保持的姿势（一个 token 文件），必须与训练数据的开局一致 |
| 特权状态 | sim 每个控制帧发布的真值（完整 qpos、物体位姿），`seq` 是它的帧号。专家可以用，VLA 不用（只用本体状态和相机图） |
| exporter | 上游的数据录制进程，录制时随栈启动，把相机、状态、动作写成 session |
| session | exporter 的一次录制目录（在 `$LOCOMANI_TMP` 下，对应关系记在 `<run>/collect/inputs.json`），每批一个，是 convert 的输入 |
| priv_seq | exporter 每行数据记下的特权状态 `seq`，据此把数据集的行对回 trace 里的同一时刻 |
| lockstep | 评测用的仿真时钟（`eval.client: lockstep`，2026-09-28 起三条主线的默认）：sim 和 deploy 只在评测进程要求时前进一步，拍照、渲染和推理时仿真暂停，推理延迟按帧模拟。对照是按墙钟跑的 `blocking` 和 `async`（[运行机制](../running.md#8-评测的时钟)、[时间模型](../bimanual_design.md#评测的时间模型墙钟与-lockstep)） |
| RTF | 实时比 = 仿真时间 ÷ 墙钟时间。deploy 按墙钟推进参考，RTF 低于约 0.9 时机器人会"走得比指令快"，定时阶段被误判失败 |

## 专家、策略与评测

| 词 | 意思 |
|---|---|
| 抓取来源 | [任意物体抓取](../object_grasp.md)这条线的抓取位姿从哪来：`cache`（离线抓取缓存，默认）或 `graspnet`，`run_grasp.py --grasp-source` 选 |
| 离线抓取缓存（grasp cache） | 每个物体场景目录里的 `grasp_cache.npz`：事先在物体网格上采样、在只有一只手的 MuJoCo 模型里合爪抬起晃动验证过的抓取，每局按测得的物体位姿取用；`grasp_cache.py` 建，`cache_all.sh` 建全部 |
| graspnet | graspnet-baseline：从深度点云预测平行夹爪抓取的网络，本仓库用它的 RealSense 权重，作为可选的抓取来源，跑在单独的环境里（`install_graspnet.sh` 安装，`servers.sh start graspnet` 开服务） |
| 专家（expert）/ agent server | 脚本化的"遥操作员"：读特权状态驱动机器人完成任务，产出示范数据（`<线>/run_agent_server.sh`，`task.expert` 选哪一个） |
| VLA / GR00T | 视觉-语言-动作策略；本仓库微调 NVIDIA 的 GR00T（训练代码在 `src/gr00t/`），输入相机图、本体状态和指令，输出动作块 |
| LeRobot 数据集 | Hugging Face LeRobot 格式（parquet + mp4），convert 的产物、GR00T 的训练输入；格式见 [单臂详解](../single_arm_design.md) |
| 动作块（action chunk） | 策略一次输出的连续多步动作，执行一段后再问下一块 |
| 策略服务（policy server） | `run_vla_server.sh`：加载 checkpoint，收观测、回动作块，端口 `eval.policy_port`。真机部署用的是另一个：`run_gr00t_server.sh`（照 Isaac-GR00T 官方，见 `src/real_robot/README.md`） |
| 跑局（rounds driver） | `<线>/run_vla_inference.sh`：逐局重置、让策略控制机器人、记成败，跑完收栈 |
| 开环 / 闭环评测 | 开环：在验证集上逐帧比较策略预测与示范动作（`open_loop_eval.sh`），不起仿真。闭环：策略真的在仿真里控制机器人跑完整局 |

## 渲染

| 词 | 意思 |
|---|---|
| `backend.render` | `live`：采集时 sim 实时渲相机；`offline`：不渲，只留物理和 trace，数据集相机先存黑色占位帧 |
| 补渲 | convert 之前按 trace 给离线录制的 session 渲出数据集相机，覆盖占位帧（`render_dataset_cameras.sh` / `_isaac.sh` / `_unreal.sh`）；只渲成功局，失败局 convert 不读 |
| `images.source` | 策略看到的相机图由 `mujoco`、`isaac` 还是 `unreal` 渲；训练和闭环评测同一个（[图像来源](../rendering.md#图像来源imagessource)） |
| Isaac / SIMPLE | Isaac Sim 4.5：NVIDIA 的 RTX 渲染器，本仓库只用它出图，物理始终在 MuJoCo。它的 Python 环境借 SIMPLE 仓库的 venv（`$RENDER_PY`，安装第 7 步） |
| Isaac stage | 场景给 Isaac 用的 USD 版本，在 `assets/isaac/scene/<名字>/`，由 `configs/scenes.yaml` 该场景 `isaac.build` 的命令生成 |
| `IsaacView` | Isaac 出三路相机图（ego + 左右腕）的那个类（`src/locomani/sim/isaac/view.py`），Isaac 补渲和镜像共用 |
| 镜像（mirror） | 闭环评测时在另一张卡上出图的进程，Isaac 的是 `run_isaac_mirror.sh`，Unreal 的是 `run_unreal_mirror.sh`：策略要观测时，把 sim 此刻的状态渲成相机图。两者用同一个协议、同一个端口，策略端不区分（[机制](../rendering.md#vla-闭环的-isaac-画面按需镜像)） |
| Unreal / URLab | Unreal Engine 5.7：`images.source: unreal` 时出图的第二种渲染器，物理照样在 MuJoCo。URLab 是把 MuJoCo 模型导进 Unreal 的插件，本仓库给它打了补丁（`patches/urlab_fresh_capture.patch`），只用来按给定的状态摆姿态、截图（安装第 8 步） |
| Unreal 编辑器 | 一台机器开一个、常驻的 Unreal 进程（`start_unreal.sh --gpu N` 起，`stop_unreal.sh` 关），Unreal 的补渲和镜像都连它出图；`--gpu` 选的就是 Unreal 渲图的卡 |
| Unreal 关卡 | 场景给 Unreal 用的版本：Blender 烘焙的房间关卡，再导入该场景的机器人、纸箱和三路相机；由 `configs/scenes.yaml` 该场景 `unreal.build` 的命令建在编辑器里。没有关卡的场景在条目里写明不支持 |
| `$BRIDGE_PY` | 和 Unreal 编辑器通信的 Python 环境（安装第 8 步建），Unreal 的补渲、镜像、建关卡都在它里面跑 |

## 文档

| 词 | 意思 |
|---|---|
| runbook / 运行机制 / 详解 / 台账 | 每条线跑哪些命令（`docs/<线>.md`）；三条线共用的运行规则（`docs/running.md`）；为什么这样设计（`*_design.md`）；哪天跑出什么数（`docs/ledger/`） |
| STATE / TODO / DECISIONS | 仓库根的当前进度与卡点 / 任务账本（编号如 4.7.7 指这里）/ 每次决策的理由与被否决的方案 |
