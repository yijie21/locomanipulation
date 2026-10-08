# warehouse 详解

怎么跑在 [warehouse runbook](warehouse.md)；逐次实测数字在 [双臂台账](ledger/crate.md)；每次改动的理由在 [DECISIONS](../DECISIONS.md) 2026-09-22 起的各节。
warehouse 和 crate 是同一条管线：同一个 `tasks/crate` 视觉专家、同一套脚本、同一个双臂评测器。专家本身的设计见[双臂详解](bimanual_design.md)，这一页只写 warehouse 不一样的地方。

## 1. 和 crate 的三处不同

1. 场景：扫描来的物流房间，托盘上放一个纸箱。默认实验用 `warehouse_box_45x32x33`：由 `warehouse_textured` 生成，另外记下托盘每条边的走入朝向、边外空地 `clear_m` 和第三视角机位，所以每局可以从任一条边走入（`task.visual.start_edges`，runbook §2）；2026-10-03 起默认只走 −y 边。旧名 `warehouse` 是贴图之前建的，纸箱没有贴图，只留给旧运行。
2. 采集时仿真不渲染相机（`backend.render: offline`），只记录物理状态和每帧的 `qpos`（trace）。
3. 转换之前多一步"补渲"：按 trace 把数据集的三路相机图渲出来。

## 2. 为什么采集时不渲染

旧 `warehouse` 场景的纸箱是一整块网格，超过了 MuJoCo 单块网格 2²⁰ 个面的门槛，实时渲染时仿真速度只剩实时的 0.35 倍。deploy 按墙钟推进参考动作，于是机器人实际走得快了大约 3 倍，2026-09-24 实时渲染的一批是 0/20。`warehouse_textured` 已经把纸箱拆成 9 块，但实时渲染下的仿真速度还没测过（TODO 4.7.7），所以仍然用 `offline`。

判断是不是渲染拖慢了仿真：用 trace 的行数乘 0.02 秒，再除以 manifest 里这一局的 `seconds`，低于 0.9 就是被拖慢了。

## 3. 补渲在做什么

- 离线录制时仿真不出图，exporter 在相机列里存黑色占位帧；状态、动作、物体位姿都是实际记录的。
- 补渲按每行记下的仿真帧号，从 `trace/` 取出那一刻的 `qpos`，用同一个场景、同一组相机（第一视角和左右手腕）渲出图，覆盖占位帧。
- 两个补渲脚本都只渲成功局。失败局保留占位帧，因为转换不读它们；硬要 `--include-discarded` 会被拒绝。两个脚本都可以重跑，已经渲过的局会跳过。
- Isaac 版只渲从 50 Hz 降到 25 Hz 后保留的行，用的是闭环镜像同一个渲染函数（`IsaacView`，同一个 Isaac 场景、同一个曝光），所以训练图和评测图出自同一个渲染器。它只处理已经跑完的批，可以在采集还在进行时先在另一张卡上跑，采集结束后再跑一次补齐。
- 转换会核对 session 是哪个渲染器补的，和 `images.source` 不一致就拒绝。
- mars 线的补渲图比状态晚 20–63 ms，模拟真机相机（`images.camera_delay`，[渲染说明](rendering.md#图像来源imagessource)）；lockstep 评测的图按同一模型变旧。
- mars 的 smoke 系列数据集 50 fps，评测按官方客户端的节拍（一行一拍，新块接入 0.4 s 后再推理；DECISIONS 2026-10-07）；`warehouse_mars_200` 是已做的 25 fps 运行。

## 4. MuJoCo 图还是 Isaac 图

`images.source` 决定策略看到的图由谁渲：`mujoco` 和物理用同一个 MuJoCo 模型；`unreal` 用 Unreal 编辑器（URLab）渲，`isaac` 用 Isaac Sim 的 RTX 渲，光照和材质都更接近真实。2026-10-03 起 runbook 默认用 `unreal`（用户定）。训练集和闭环评测必须用同一个，区别见[图像来源](rendering.md#图像来源imagessource)。

`isaac` 需要：`backend.render: offline`；装好 Isaac（[安装第 7 步](setup.md)）；场景的 Isaac 版本（`bash scripts/build_scene.sh warehouse_box_45x32x33_mars --isaac`）；第二张卡。`unreal` 的条件相同，只是换成装好 Unreal（[安装第 8 步](setup.md)）和场景的关卡（`--unreal`）。Isaac 渲一次三路相机约 0.1 秒，和仿真放在同一张卡上会拖慢仿真。冲突的组合在加载配置时就会被拒绝；旧文件里的 `eval.image_source` 按 `images.source` 读取。

## 5. 评测前要核对的几项

评测只读运行目录里的 `config.yaml` 副本，不读 `configs/experiments/` 下的原件。已经开跑的运行要改下面这些字段，就改 `<run>/config.yaml`；用 `run_overnight.sh` 时，原件里的 `images.source` 也要一致，因为它按原件决定要不要起镜像。

| 项 | 要求 | 不对会怎样 |
|---|---|---|
| `summary.json` 的 train 条目 | 训练结束时自动写入，指向最后一个 checkpoint；要评别的 checkpoint 用[运行机制](running.md#4-跳过续跑与重跑)里的命令手工写 | `summary.json has no checkpoint; run the train stage first` |
| `eval.initial_token` | `configs/deployment/warehouse_initial_token.json`。warehouse 和 `box_platform` 两条线的数据开局，手臂都和它相差不到 0.03 rad；从旧实验文件抄来的 `crate_initial_token.json` 差 0.6 rad | 不报错，但策略从训练里没见过的姿势起步 |
| `images.source` | `isaac` 时起栈前要有场景的 Isaac 版本和 Isaac 的 Python 环境，评测时另开镜像 | 起栈时报错，并给出构建命令 |
| `<run>/eval/` | 不能有上一次的 `closed_loop.manifest.jsonl` 或 `closed_loop.json` | 前者报 `exists; pick another --out`；后者让 `run_overnight.sh` 直接跳过评测 |
| `eval.client` | 默认 `lockstep`，需要 `latency_frames` 或 `latency_dist`；栈由 `start_stack.sh eval` 按它起 | 栈和 `client` 不符时，跑局前报错，要重起栈 |

每局的起点、夹爪合拢下限（`task.visual.close_target`）和"后退"一列都取自 `task.visual`，和采集一致（[双臂详解第五节](bimanual_design.md#五双臂闭环评测器)）。

## 6. 怎么读评测结果

- `eval/closed_loop.json` 的 `successes`：箱子抬起至少 0.6 倍 `lift_height`、倾角小于 `max_tilt_deg`、两只手都接触，并且保持 2 秒。失败按 `[vla] walk`、`[vla] reach`、`[vla] close` 分阶段。
- `retreat_distance` 大于 0 时，成功局接着跑 `retreat_timeout` 秒，`retreated` 和 `retreat_reasons` 记录是否带着箱子后退了这么远。转换时在持物验收开始处裁掉了训练数据的尾段，训练集里没有后退动作，所以这一列目前一定是 `timeout`，应理解为"成功后又拿住这么久没掉"；`dropped`、`fallen` 才表示掉了。
- 逐局记录：`eval/closed_loop.manifest.jsonl`（起点 `placement`、抬升 `rise`、倾角 `tilt_deg`、最近距离 `min_dist`）；`eval/traces/round_NNN.json` 每个动作块一行（夹爪指令、延迟、镜像图的帧号）。
- 镜像的输出：分终端跑时在镜像的终端里，用 `run_overnight.sh` 时写到 `eval/isaac_mirror.log`；每次渲染另记一行到 `eval/isaac_mirror.jsonl`（帧号、状态的时效、渲染耗时）。镜像只在策略要观测时渲当前状态（[机制](rendering.md#vla-闭环的-isaac-画面按需镜像)）。
- 参照：`box_room_grasp_200_cap120_20260925` 的 checkpoint-5000，开环 MSE 0.00123；闭环 100 局墙钟 81，lockstep 88（[台账](ledger/crate.md) 2026-09-27 末两节）。

## 7. 做对照实验

- 调参基线是 `configs/experiments/box_room_grasp_baseline_20260924.yaml`：不录制，20 局（seed 7000–7019）成功 6 局。成功指抬起、躯干直立、后退 0.3 m 箱子不掉。它每局从同一个地方开始，不是现在默认的四边随机；四边随机下的对照从 `warehouse_edges_prep_compact_20260928` 复制（不录制，40 局保留 17，seed 91000 起；台账 2026-09-28），它的 `task` 和默认实验的完全相同。
- 基线文件本身不改。新实验复制它，只改 `name` 和一个参数。
- 同一份代码跑 10 局，成功数能在 2 到 7 之间波动，所以对照一律跑 20 局，两边用同一组 seed。
- 专家参数（`task.visual.*`）的含义查[配置字段表](reference/config_fields.md)，历次改动的效果查 [DECISIONS](../DECISIONS.md)。
- 目前主要的失败是 `reach_stalled`（手停在箱沿上方）和 `lost_attachment`（合爪后脱开），不是环境问题；进展见 [TODO](../TODO.md) 2.4.10.29。

## 8. 相关的支线

- 台上纸箱 `box_platform`、四边线 `warehouse_box_*`：场景都在 `configs/scenes.yaml`；默认配置和成功率随布局参数的变化见[双臂抓箱的参数范围](box_grasp_envelope.md)，方法见[场景能力扫参](scene_sweep.md)。
- 房间和纸箱怎么从 Blender 导出到 MuJoCo 和 Isaac：[Blender 场景](blender_scenes.md)。
- 离线渲染、Isaac 镜像的实现和限制：[仿真与渲染](rendering.md)。

## 9. 报错与原因（runbook 之外的）

| 现象 | 原因与处理 |
|---|---|
| 机器人走得太快、冲过头撞到箱子、整批 0 分 | `backend.render` 不是 `offline`，按第 2 节算实时比 |
| 转换报 `rendered by another renderer` | session 补渲用的渲染器和 `images.source` 不一致。训练图必须和评测图同源，换 `name` 重采 |
| 起栈报 `the training set would be empty`，或转换报 `leaves none of` | `train.holdout` 不小于能保留的条数。调小 `train.holdout`（0 表示不留验证集）：起栈时被拒的，改完同名重起；转换时才被拒的，配置已经冻结在运行目录里，只能换 `name` 重采 |
| 起栈报缺 POSE 补丁 | deploy 没打补丁，或者 `SONIC_DEPLOY_ROOT` 指错了（[安装第 3 步](setup.md)） |

## 10. 运行细节（2026-09-30 从 runbook 移来）

### 每局从哪里开始

默认（`warehouse_mars_*`，2026-10-03 用户定）每局都从托盘的 −y 边（`ym`）走过来：`task.visual.start_edges: [ym]`。stock 的 `warehouse_smoke`、`warehouse_200` 仍是 2026-09-28 的四边线，每局从四条边（`xm/xp/ym/yp`）中随机抽一条，纸箱搬到这条边上。两者的其他随机量相同，范围都是单独改这一项时成功率仍不低于 30% 的区间（[双臂抓箱的参数范围](box_grasp_envelope.md) §3）：
- 起点离台边 0.6 m 到"这条边外的空地 − 0.25 m"（ym 1.04 m；四边线另有 xm 0.86、xp 0.80、yp 1.38 m）。房间的货架和墙没有碰撞体，起点不能放进去。
- 起点朝向 ±20°，起点横移 −8～0 cm（机器人在箱子中线右侧）。
- 纸箱离台边 0–2.5 cm、偏角 0～+3.5°，沿边位置在整条边上任意取。

四边线在房间里实测保留 17/40，其中 −y 边 11/19（stock，[双臂台账](ledger/crate.md) 2026-09-28）；mars 在四边线上的冒烟 30 局保留 9 条（2026-10-03），只走 −y 边的 mars 还没实跑。`run_agent_server.sh` 开始第一局之前把这份设置打印出来；在终端里运行时，打印后等 10 秒才开始，这时按 Ctrl-C 不会录下任何一局，仿真栈也一起关掉。加 `--dry-run` 只打印、不启动。

`*_fixed_start` 实验（mars 的 `warehouse_mars_smoke_fixed_start`，stock 的 `warehouse_smoke_fixed_start`、`warehouse_200_fixed_start`）按 2026-09-28 之前的方式摆放：每局从场景自带的起点出发（在 −y 边前方，往后退 0.25 m），纸箱在原位，只加 ±2 cm、±5° 的抖动。它们和默认文件只有摆放方式不同，专家参数完全一样，场景也相同。

### 通道数

一条通道是一套仿真栈加一个跑局进程，并行只改变速度、不改变数据。`start_stack.sh collect` 按空闲 CPU、进程数、显存和实例决定开几条，但不算系统内存，而一条通道常驻约 5 GB。没有 cgroup CPU 配额的桌面机（如 32 GB 内存的 SRA0889）会被开出 4 条，并在第一局被内核杀掉；这样的机器写 `--lanes 1`，或在实验文件里写 `collect.lanes: 1`。被杀后重来时，中断的批从头再跑。本机每批 10 局的通道要约 1600 个进程，2026-09-30 只开得出 1 条（[运行机制](running.md#6-并行)）。

### 3 批采不够

3 批跑完还不够 `collect.successes` 条时，采集打印 `stopped at 8/10 kept after 3 batches` 这样的一行并以非零状态结束。已采的照样能接着补渲、转换、训练，只是数据少几条。`run_overnight.sh` 遇到这种情况停在采集，重跑也不会多采；要凑够，就复制实验文件、把 `collect.max_batches` 调大、换运行名重采。2026-09-30 两次实跑，20 局分别保留 11 条和 8 条，所以冒烟允许第三批（多约 6 分钟）。

### 补渲写到哪里

相机帧的 mp4 不在运行目录下，而在采集时的 session 目录里（路径列在 `collect/inputs.json` 的 `sessions`）：`<session>/videos/chunk-000/observation.images.<ego_view|left_wrist|right_wrist>/episode_NNNNNN.mp4`。渲过的几 MB，失败局仍是 100 多 KB 的黑帧；`<session>/meta/offline_render.json` 记着是谁渲的。2026-09-30 实测：Isaac 每局 135–163 s（每次 0.21 s，启动另需约 20 秒），Unreal 每局 74–93 s（每次约 130 ms）。

### 录像的画面

- 状态轨迹（采集是 `trace/<局>_state_trace.npz`，闭环评测是 `eval/state_trace/`）记录了每次仿真发布时的全部关节位置，回放它就能事后从任意机位出图。这和第 3 节的数据集相机是两回事：数据集相机是策略要看的图，一次运行只能由一个渲染器填；回放录像只给人看，同一份轨迹可以分别用 MuJoCo 和各个渲染器渲，也不用重采。
- `render_third_person.sh` 的机位、ghost 和小窗见[仿真与渲染](rendering.md#图像来源imagessource)。评测局的小窗由渲染器按每一帧的状态现渲，就是镜像给策略的那种图；评测没有 ghost。长的局预览切成 2–5 段。2026-09-30 冒烟两局评测：Isaac 每局小窗 225–243 s 加录像 123 s，Unreal 每局 223–285 s 加 130–140 s。
- Isaac 另有跟拍机位、可开路径追踪的并排视频 `render_isaac_videos.sh`，用法见[仿真与渲染](rendering.md#warehouseblender-房间的-isaac-重渲)。

## 11. mars 机器人

2026-10-03 起这条线用 mars（用户定）：实验室真机 G1 的模型，由 `make_dex1_model.py` 按协作者给的 URDF（`assets/unitree_G1_with_wrist_camera`）生成，第一视角是头顶云台上的 D455，腕部相机是 D405。关节、执行器和相机名都和原来的 stock 机器人相同，所以专家、SONIC 和数据集格式不变。场景名带 `_mars`；不带的名字仍是 stock 机器人，旧的运行和 checkpoint 都是 stock 的。要在 stock 上跑 runbook：第 1 步的场景名去掉 `_mars`，运行名用 `warehouse_smoke`、`warehouse_200`（Isaac 版）或加 `_unreal` 的 Unreal 版；它们仍从四条边走入。

- 力矩按 URDF，只有腰 roll/pitch 用 50 N·m，不用 URDF 的 35。35 时专家抱箱前倾要的力矩超过上限，腰塌到限位上来回弹，10 局成功 0 局、手抖 9.4 mm；50 时 10 局成功 6 局，和 stock 同种子的 6/10 一样（`outputs/mars_pilots_20261002/`）。SONIC 训练时腰也是 50。真机腰的固件上限还没确认，专家的姿势在真机上可能撑不住（TODO 13.6.11）。
- 成功率相同，失败方式不同：mars 的 4 局失败都是箱子抬起后一只手脱开，stock 是抬起前脱开或手停在箱沿上方。原因未查（TODO 13.6.4.1）。
- 2026-10-03 照 runbook 跑 `warehouse_mars_smoke` 到转换（当时还是四边线，运行目录改名为 `warehouse_mars_smoke_4edges_20261003`）：30 局保留 9 条（三批各 4、1、4），失败是 reach_stalled 9、lost_attachment 8、basket_tilt 4；stock 在 2026-09-30 的两次是 20 局保留 11 条和 8 条。只有这一次，分不清是机器人的差别还是批间波动。
- 相机视场用规格书的值（D455 竖直 65°、D405 58°），真机标定到了要换。
- 第一视角可换成脸上固定的 D435（用户 2026-10-06）：建场景时加 `--ego-camera d435_head`，记进 `scene_meta.json` 的 `ego_camera`；场景名带 `_d435`，实验文件 `warehouse_mars_smoke_d435_unreal`、`_d435_isaac`、`_d435_mujoco`，各自与同渲染器的 D455 文件只差场景。位姿是 URDF 的 `d435_head_link`（下俯 47.6°），竖直视场 42°（规格书 69°×42°）。仿真出图、专家判断目标是否在画面里、三个渲染器都读同一个 `head_camera`，所以换的是场景，不是运行时的开关。
- Unreal 关卡从场景导入机器人；Isaac 把 `assets/isaac/robot/mars/g1_dex1.usda` 加到房间的 stage 上。两者都在 `source.json` 里记下机器人，和场景的 `scene_meta.json` 对不上就拒绝渲染；补渲的标记也记机器人，同一个 session 不会混着两个机器人的图。
