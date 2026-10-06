# 仿真数据训练 GR00T，零样本部署到真机 G1（GR00T + SONIC）：设计

状态（2026-10-06）：第 7 节的任务 1、任务 2 已实现并有单元测试；其余为后续步骤。决策记录在 [DECISIONS](../../DECISIONS.md) 2026-10-06 一节，任务编号见 [TODO](../../TODO.md) 第 16 节。

## 1. 目标

用本仓库仿真采集的 warehouse 双臂抓箱数据微调 GR00T N1.7，不采真机训练数据，按官方 GR00T-WholeBodyControl（SONIC）的推理客户端把策略部署到实验室真机 G1 上。第一阶段的验收是走到箱子前而不摔，不是抓取。

## 2. 事实依据

### 2.1 频率

| 环节 | 官方 SONIC 流程 | 本仓库（2026-10-06 之前） |
|---|---|---|
| 控制与 token 发布 | 50 Hz（C++ 控制线程 50 Hz，客户端 `--action-publish-rate 50`） | 评测客户端按 `dataset_fps` 发布，25 Hz |
| 数据集采样网格 | 50 Hz（exporter `--data-collection-frequency 50`，`info.json` `fps: 50`） | 原始 session 50 Hz；转换器 `DST_FPS = 25` 抽行，数据集 25 fps |
| 相机 | 30 fps（OAK / RealSense 驱动固定），每个 50 Hz tick 重复最新帧 | 仿真实时相机约 30 Hz；离线补渲只渲抽到的行，其余行重复前一行 |
| GR00T 推理 | 2.5 Hz，每次 40 个 token（0.8 s），延迟补偿后逐 tick 播放 | `execution_horizon 16`，lockstep 冻结世界 |
| GR00T 对 fps 的依赖 | 无。模型只读 `delta_indices`（帧索引偏移），不读时间 | 同 |

结论：相机帧率不进入模型（视频 `delta_indices=[0]` 只取当前帧）。真正进入模型的时间步长是相邻两行 state / action 的间隔。官方 `UNITREE_G1_SONIC` 预训练先验是 20 ms 一步；25 Hz 微调要让模型重学一倍慢的 token 节奏。50 Hz 下 40 步 chunk 是 0.8 s，与官方部署相同。

真机参考数据集 `~/workspace/tmp/Grasp_toy_bear_v4_gr00t_v1`（155 集）印证了官方流程：`fps: 50`，三路相机 50 fps 容器，4 s 内 200 帧里 `mpdecimate` 判出约 47% 重复帧。

### 2.2 数据格式（真机 vs 仿真）

两边都是 SONIC exporter 的 LeRobot v2.1 布局：`observation.state` 43 维（29 身体 + 左右手各 7）、`observation.eef_state` 14 维、`action.motion_token` 64 维、`teleop.*_hand_joints` 7 维、三路 640×480 视频，`modality.json` 的切片一致。差别全在夹爪：

| | 真机（Dex1） | 仿真（Dex1 模型） |
|---|---|---|
| state 的手 7 维 | 同一个电机位置广播到 7 维 | 第 4、5 维是两指的米制位置，其余 0 |
| teleop hand_joints 命令 | 第 0 维，5.5 = 开，0 = 闭，连续渐变 | 第 0、1 维，米制，+0.0245 = 开，−0.02 = 闭（命令可到 −0.06） |
| 跟随 | 约 12–15 帧到中点；夹住物体停在 0.48 | 物理仿真 |

开合方向相反、单位不同。官方 v1.1 代码没有任何 Dex1 分支：C++ 只有 `Dex3Hands`（7 电机），客户端要求 hand 字段恰好 7 维。

### 2.3 官方推理客户端（`gear_sonic/scripts/run_vla_inference.py`）

- 观测：`g1_debug` 的 `body_q`（29）、`left_hand_q` / `right_hand_q`（7）、`base_quat`；拼成 43 维 q 后按关节组切成 7 个 state 组，加 `projected_gravity`。不算 `eef_state`。左手有 Dex3 专用耦合 hack（`left_hand_q[5,6] = [3,4]`）。
- 相机：`ComposedCameraClientSensor` 一路订阅；`right_wrist` 发给模型时 key 写成 `wrist_view`。
- 动作：要求 `motion_token`、`left_hand_joints`、`right_hand_joints`；|token| > 1.25 整块丢弃；不做 FSQ snap。
- 时序：异步推理线程；新 chunk 到达时 `round(delay × publish_rate)` 跳过过期动作；每 tick 发一个 token，protocol v4（`token_state`、`frame_index`、两只手 7 维）。
- C++ 侧对输入频率没有硬性要求：token 采样保持，200 ms 没收到只警告。

## 3. 决策

| 编号 | 决策 | 理由 |
|---|---|---|
| D1 | 数据集改为 50 fps，用现有 50 Hz 原始 session 重转，不重采 | 对齐预训练先验和官方部署；源数据已是 50 Hz |
| D2 | 50 fps 数据的图像每行独立补渲（Unreal），不用重复帧 | 用户选择；真机 D455 本身可到 90 fps |
| D3 | 三路相机 ego + left_wrist + right_wrist | 和仓库训练、评测一致；官方客户端加两路订阅 |
| D4 | state 保持仓库的 60 维布局；手的 7 维槽位填 Dex1 的 1 维开合值广播 | 真机 exporter 自己就是这样记的（2.2）；modality 配置不改 |
| D5 | action = motion_token 64 + 左右夹爪各 1 维 = 66 维 | 仓库现有 `SONIC_CRATE` 布局 |
| D6 | 夹爪统一用真机 Dex1 原始单位：0 = 闭，5.5 = 开 | 模型输出即机器人命令；真机数据集可直接对照 |
| D7 | 发给真机的 7 维 hand 字段把 1 维值广播到全部 7 维 | 真机驱动读哪一维未知；广播对任一维都正确 |
| D8 | 部署用官方客户端的结构，允许修改；代码放 `src_real_robot/`，只 vendor 客户端，`gear_sonic` 依赖按安装版本导入；GR00T 策略服务器复用 `src/gr00t`（本仓库 `serve_policy.py`） | 6.5 GB 的上游树不能跟踪；两份 GR00T 副本会分叉 |
| D9 | 真机 C++ 由用户自行编译带 Dex1 驱动的 SONIC 1.1；我们不替换机器人侧 | 机器人侧代码不可见 |
| D10 | 真机相机复用机器人上已验证的三相机服务器（D455 头部 30 fps，双 D405 30 fps） | 录过 toy-bear 数据集，key 与分辨率已对上 |
| D11 | 真机 toy-bear 数据集只做格式和分布对照，不进训练，不写 loader | 任务不同，进训练会模糊归因 |
| D12 | 上真机前先用同一客户端对接本仓库 MuJoCo 栈做 sim2sim | 唯一能暴露 wall-clock 下延迟补偿行为的验证 |
| D13 | 第一门槛：10 局，起点 0.6–1.0 m，30 s 超时，骨盆到箱子中心 ≤ 0.6 m 且站稳 2 s 算过；单腿失稳或躯干俯仰超 30° 急停记失败 | 用户确认 |

同源数据另留一份 25 fps 版本训对照 checkpoint。

## 4. 数据路径

### 4.1 转换频率成为运行的字段：`dataset.fps`

- 实验文件新增可选块 `dataset: {fps: 25 | 50}`。没有这个块的文件（所有历史运行）读作 25，即它们当年转换用的频率。其他值被拒绝：目标频率必须整除源的 50 Hz，否则抽帧不均匀（2026-08-28 的 10 Hz 伪影）。
- 入口 `run_context.load_run_config` / `open_run` 迁移并校验（`migrate_dataset`），和 `images` 一样。
- 消费者显式拿到值：`run_convert` 把 `--fps` 传给 `convert_to_hf_format.py`；补渲阶段（`view_render`）按它决定渲哪些行；`RunTraces`（审计的世界高度查表）从运行的 `config.yaml` 读它。
- `world_lift.resample_selection(n, dst_fps)`：步长 `50 / dst_fps`，50 时为恒等。

### 4.2 补渲标记记频率，转换器核对

- Isaac / Unreal 补渲的标记 `meta/offline_render.json` 新增 `dataset_fps`；没有这一项的旧标记读作 25。
- 一个按 F Hz 渲过的 session 能服务目标频率 G，当且仅当 F 能被 G 整除（50 渲的服务 25 转换；25 渲的不服务 50）。`require_rendered`（转换前）和 `unrendered`（逐集）都核对；不服务时拒绝，并说明要删掉标记、按新频率全部重渲。原因：此前 50 fps 转换会静默用重复帧训练。
- MuJoCo 补渲每行都渲，标记记 `dataset_fps: 50`，服务任何频率。

### 4.3 对现有 warehouse_mars_200 的操作

数据集目录是运行目录的一部分，而运行的 `config.yaml` 一旦开始就不改。50 fps 版本是一个新运行，共用同一批 session：

1. 新建实验文件 `configs/experiments/warehouse_mars_200_50hz.yaml`：复制 `warehouse_mars_200.yaml`，`name` 改成新名，加 `dataset: {fps: 50}`。
2. 新建运行目录 `outputs/runs/warehouse_mars_200_50hz/`，放入新实验文件的拷贝 `config.yaml`，从旧运行复制 `summary.json`（只留 `collect` 条目）、`manifest_*.jsonl`、`collect/`、`trace/`、场景目录。
3. 删掉 74 个 session 的 `meta/offline_render.json`，运行 `scripts/render_dataset_cameras_unreal.sh outputs/runs/warehouse_mars_200_50hz`：每行都渲，每局约 165 s，全部约 25 小时（单卡，p50 133 ms 一帧）。渲完的 session 视频对 25 fps 运行同样有效（它取偶数行）。
4. `scripts/convert_data.sh outputs/runs/warehouse_mars_200_50hz`。

### 4.4 夹爪单位（后续，见第 7 节 T3）

转换器把仿真的米制映射到 Dex1 单位：线性 −0.02 m → 0、+0.0245 m → 5.5，命令低于下界截断到 0；写入 `teleop.*_hand_joints` 第 0 维（动作）并广播到 `observation.state` 的手 7 维（状态，取第 4、5 维均值后映射）。映射函数与客户端的逆映射同源（`src_real_robot/real_robot/gripper.py`）。

## 5. 训练

GR00T N1.7（`src/gr00t`，上游 376ba890），`unitree_g1_sonic` tag，`configs/gr00t_sonic_3cam_bimanual.py`（视频 3 路、state 60、action 66，horizon 40）。50 fps 下 chunk 0.8 s。训练命令不变（`scripts/train.sh <run>`）。

## 6. 部署：`src_real_robot/`

```
src_real_robot/
  README.md            怎么在真机上跑（runbook）
  UPSTREAM.md          vendor 来源：GR00T-WholeBodyControl v1.1 a0732b6；复用的 gear_sonic 模块清单
  run_inference.sh     用 SONIC 的 .venv_sim 启动客户端
  real_robot/
    inference.py       客户端主循环（官方结构：异步推理线程、延迟补偿、键盘命令、50 Hz 发布）
    observation.py     观测拼装：60 维 state（手 7 维广播 Dex1 值、eef_state 由 FK 算出）、三路图像、prompt
    chunk.py           chunk 播放：snap、延迟补偿索引、逐 tick 取 token 与夹爪
    gripper.py         夹爪单位：模型输出单位 → Dex1 → 线上 hand 字段（真机广播 7 维 / 仿真米制两指）
    policy_client.py   和本仓库 serve_policy.py 的扁平观测协议（msgpack + ZMQ）
```

数据流：

```
robot C++ (g1_debug :5557) ─┐
camera server (:5555) ──────┼─ observation.py ─ policy_client ─▶ serve_policy.py (GPU 机, :5590)
prompt ─────────────────────┘                                        │ 40 × (64 token + 2 gripper)
                                                                     ▼
robot C++ (pose v4 :5556) ◀─ chunk.py ◀─ gripper.py ◀────────────────┘
```

与官方客户端的差别，全部有理由：

| 改动 | 为什么 |
|---|---|
| 三路相机，`right_wrist` 不改名 `wrist_view` | 数据集的 video key 就是 `right_wrist` |
| 客户端算 `eef_state`（pinocchio FK，与 exporter 同一函数） | 60 维 state 含双腕位姿；官方只发 43 + 3 |
| 手的 state 取 `hand_q[0]` 广播到 schema 给的宽度 | D4；去掉 Dex3 耦合 hack |
| 夹爪 1 维 → Dex1 单位 → 7 维广播发出 | D6、D7；`--gripper-units sim_m` 兼容米制 checkpoint，`--hand-wire sim` 兼容仿真桥 |
| token 做 FSQ snap | 和仓库评测一致（`sonic_token.snap`）；官方只做 |x| > 1.25 丢弃 |
| 对接本仓库 `serve_policy.py` 的 `get_action_raw` 扁平协议 | 服务器核对 checkpoint 的 modality 与配置一致并报 `identity` |
| 初始 token 可选数据集 medoid 文件 | 官方常量是前倾姿势，不在我们数据分布里（`initial_pose.py`） |

不变：50 Hz 发布、40 horizon、2.5 Hz 推理、`round(delay × rate)` 延迟补偿、键盘命令（p 暂停、k 启停 C++、i 初始姿势、t 改 prompt）、protocol v4 打包（`pack_pose_message`）、`ComposedCameraClientSensor`、`ZMQStateSubscriber`。

端口沿用官方默认：状态 5557、相机 5555、动作 5556、键盘 5580、策略 5590。sim2sim 时这些正是本仓库仿真栈实例 0 的端口。

## 7. 任务拆分

| 编号 | 任务 | 状态 |
|---|---|---|
| T1 | `dataset.fps` 字段贯穿转换、补渲、审计；闭环评测在非 25 Hz 运行上明确拒绝 | 本次交付 |
| T2 | `src_real_robot/` 推理客户端与单元测试 | 本次交付 |
| T3 | 转换器的夹爪单位映射（4.4）与 state 手 7 维广播 | 后续 |
| T4 | 新运行 `warehouse_mars_200_50hz`：全行补渲、转换、训练；25 fps 对照 | 后续 |
| T5 | sim2sim：仿真桥接受 Dex1 单位（`--hand-wire sim` 的逆映射）；官方客户端结构对接 MuJoCo 栈 | 后续 |
| T6 | 真机：相机服务器与 `g1_debug` 字段核对；夹爪手动 5.5 / 0 验方向；按 D13 跑 10 局 | 后续，需真机 |
| T7 | 50 Hz 的 lockstep 评测（`vla_lockstep.py` 的 `RATE = 25` 写死） | 未排期 |

## 8. 已知残余差距（接受）

- 仿真 50 fps 每行独立画面，真机约一半重复帧。
- sim2sim 的 C++ 与真机那份（带 Dex1 驱动）不同源。
- 真机 Dex1 驱动读 7 维 hand 字段的哪一维未知，靠广播规避。
- 真机 D455 / D405 内参未提供；仿真用规格书视场。
