# warehouse 详解

怎么跑在 [warehouse runbook](warehouse.md)；逐次实测数字在 [双臂台账](ledger/crate.md)；每次改动的理由在 [DECISIONS](../DECISIONS.md) 2026-09-22 起的各节。
warehouse 和 crate 是同一条管线：同一个 `tasks/crate` 视觉专家、同一套脚本、同一个双臂评测器。专家本身的设计见[双臂详解](bimanual_design.md)，这一页只写 warehouse 不一样的地方。

## 1. 和 crate 的三处不同

1. 场景：扫描来的物流房间，托盘上放一个纸箱（`scene.name: warehouse_textured`）。旧名 `warehouse` 是贴图之前建的，纸箱没有贴图，只留给旧运行。
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

## 4. MuJoCo 图还是 Isaac 图

`images.source` 决定策略看到的图由谁渲：`mujoco` 和物理用同一个 MuJoCo 模型；`isaac` 用 Isaac Sim 的 RTX 渲染，光照和材质更接近真实。训练集和闭环评测必须用同一个，区别见[图像来源](rendering.md#图像来源imagessource)。

`isaac` 需要：`backend.render: offline`；装好 Isaac（[安装第 7 步](setup.md)）；场景的 Isaac 版本（`bash scripts/build_scene.sh warehouse_textured --isaac`）；第二张卡。Isaac 渲一次三路相机约 0.1 秒，和仿真放在同一张卡上会拖慢仿真。冲突的组合在加载配置时就会被拒绝；旧文件里的 `eval.image_source` 按 `images.source` 读取。

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

- 调参基线是 `configs/experiments/box_room_grasp_baseline_20260924.yaml`：不录制，20 局（seed 7000–7019）成功 6 局。成功指抬起、躯干直立、后退 0.3 m 箱子不掉。
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
