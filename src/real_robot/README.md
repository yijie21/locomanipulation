# 真机部署：GR00T 策略 + SONIC 控制器跑在 G1 上

把本仓库训练出的 GR00T checkpoint 部署到实验室的真机 G1（29 自由度、Dex1 夹爪、头部 D455 和双腕 D405）。推理客户端 [inference.py](inference.py) 按 NVIDIA 官方的 `gear_sonic/scripts/run_vla_inference.py` 逐个函数对照着写，参数名、键盘、打印信息都和官方一致；不同的只有观测和动作格式（本仓库数据集的）和 Dex1 夹爪。策略服务器 [run_gr00t_server.py](run_gr00t_server.py) 仿照 Isaac-GR00T 官方的 `gr00t/eval/run_gr00t_server.py`。两边的来源和改动见 [UPSTREAM.md](UPSTREAM.md)。设计和决策见[设计文档](../../docs/handoff/real-robot-sim2real-plan-20261006.md)。

需要三台角色，可以是同一台机器：机器人本体的控制电脑（跑 SONIC C++ deploy 和相机服务器）、一台 GPU 机器（跑策略服务器）、一台跑本客户端的机器（装了 SONIC 的 `.venv_sim`）。

## 0. 前提

- 机器人上已装好并验证过：带 Dex1 驱动的 SONIC C++ deploy（1.1）、三相机服务器（key 为 `ego_view`、`left_wrist`、`right_wrist`，640×480）。本仓库不替换这两样。
- 机器人侧的 ZMQ 端口沿用官方默认：状态 `g1_debug` 5557、相机 5555、动作 `pose` 5556、键盘 5580。换了端口就在下面的命令里传对应参数。
- 本机已完成[安装](../../docs/setup.md)第 0–4 步（SONIC 的 `.venv_sim`）。策略服务器那台机器要第 5 步（GR00T 训练环境）。

## 1. 起策略服务器（GPU workstation PC）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_server.sh --model-path outputs/runs/<run>/checkpoint-<step>
# example
cd ~/workspace/locomanipulation
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_server.sh \
    --model-path ~/Checkpoints/gr00t-locomani-logistic-unreal_20261010_033747/checkpoint-30000
```

服务器是 [run_gr00t_server.py](run_gr00t_server.py)，仿照 NVIDIA Isaac-GR00T 官方的 `gr00t/eval/run_gr00t_server.py`（本仓库原样 vendor 在 `src/gr00t/eval/run_gr00t_server.py`），参数名和官方一致，跑在 Isaac-GR00T 的 `.venv` 里、用本仓库的 `src/gr00t`。真机部署不用仿真评测的 `scripts/run_vla_server.sh`：那个脚本从运行目录读评测配置，带 RTC 和仿真用的种子，真机都用不上。

做什么：加载 checkpoint（modality 配置和字段顺序取 checkpoint 自带的），在 5590 端口提供官方的 `ping`、`get_action`、`reset`、`get_modality_config`，外加一个 `identity`（checkpoint 文件哈希、各 modality 的字段顺序、夹爪和手的训练统计范围）。客户端和官方一样调 `get_action`，发嵌套的 (B, T, …) 观测。
得到什么：终端打印 `Modalities: ...`、`Gripper ranges: ...` 和 `Server is ready and listening on tcp://0.0.0.0:5590`。客户端启动时会核对 `identity` 报告的字段顺序和 `--schema` 一致，并按夹爪统计范围定夹爪单位，不一致就拒绝启动。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--model-path` | 无 | checkpoint 目录（`checkpoint-<step>`，不是运行目录） |
| `--embodiment-tag` | unitree_g1_sonic | 训练用的 embodiment tag |
| `--backbone-path` | `$COSMOS_BACKBONE_PATH` | 本机 Cosmos-Reason2-2B 目录。checkpoint 在别的机器（集群）上训的，记录的 backbone 路径本机没有时用它。不给时 `run_gr00t_server.sh` 自动加上 `COSMOS_BACKBONE_PATH`（`scripts/paths.sh`，默认 `~/Checkpoints/Cosmos-Reason2-2B`；权重放在别处就在 `locomani.local.sh` 里改，或直接传 `--backbone-path`） |
| `--denoising-steps` | 4 | action head 的去噪步数 |
| `--host` / `--port` | 0.0.0.0 / 5590 | 监听地址 |
| `--device` | cuda | 选卡用 `CUDA_VISIBLE_DEVICES` |
| `--dataset-path` + `--execution-horizon` | 无 | 官方的 ReplayPolicy：不加载模型，按数据集回放动作，用来先查动作链路。crate 数据另加 `--modality-config-path configs/gr00t_sonic_3cam_bimanual.py` |

## 2. 起机器人侧（机器人控制电脑）

按机器人上现有的流程启动 C++ deploy 和相机服务器。本客户端只要求它们监听上面的端口。
```bash
# Install camera server
cd ~/ether-onboard/third_party/GR00T-WholeBodyControl
bash install_scripts/install_camera_server.sh
```

```bash
# Run Unitree G1 robot
# Calibrate pan-tilt head camera
sudo systemctl stop g1_comp_servo.service
cd ~/g1_comp_servo_service/build # then move the camera to the bottom right position
./test_calibration # then wait several seconds, and the camera will pan
sudo systemctl start g1_comp_servo.service # then wait several seconds, and the camera will tilt
# Check camera
sudo systemctl status composed_camera_server.service
# Set camera position 
sudo systemctl stop g1_comp_servo.service
bash ~/set_head_camera.sh
# Launch three camera
sudo systemctl restart composed_camera_server
journalctl -u composed_camera_server -n 20 --no-pager   # "All 3 cameras ready" and FPS 

# Check gripper
sudo systemctl status dex1_gripper.service
cd && cd dex1_1_service/bin && ./test_dex1_1_gripper_server -l -r

# Set gripper 
sudo systemctl restart dex1_gripper.service
```

## 3. SONIC C++ deploy

```bash
# Run local workstation
cd ~/Unitree_G1_Robot/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real --zmq-host 172.28.36.68 --hand-type dex1_1
# wait for "Init done"
```

## 4. 起客户端

```bash
bash scripts/run_real_robot_inference.sh --host <策略服务器地址> --camera-host <机器人地址> --state-zmq-host <机器人地址> \
    --init-token-file configs/deployment/warehouse_initial_token.json --prompt "pick up the box"
# example
bash scripts/run_real_robot_inference.sh \
    --host localhost \
    --camera-host 192.168.123.164 --camera-port 5555 \
    --state-zmq-host localhost \
    --init-token-file configs/deployment/warehouse_initial_token.json \
    --prompt "pick up the box"
```

做什么：连接策略服务器，核对 checkpoint 字段顺序；订阅机器人状态和三路相机；在 5556 端口发布 protocol v4 的 token 消息；启动时处于暂停状态。
得到什么：终端打印 `PolicyServer is reachable.` 和 `Starting the policy loop with language prompt: ...`，暂停期间不断打印 `Pausing...`。放开后每到一个 chunk 打印一行 `New action chunk (prompt: ..., latency: ...)`，每 50 帧打印一行 `ZMQ: Sent latent action`，带帧号、chunk 下标和两只夹爪的值。`--help` 列出全部参数及说明。

## 5. 键盘发布器（另开一个终端跑 SONIC 的键盘发布器）：

```bash
~/workspace/locomanipulation/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -c "
import zmq
ctx = zmq.Context(); pub = ctx.socket(zmq.PUB); pub.bind('tcp://localhost:5580')
print('k=start/stop  i=init pose  p=pause/resume  t <text>=prompt  [ ]=left/right gripper toggle')
while True:
    key = input()
    pub.send_string('prompt:' + key[2:] if key.startswith('t ') else key)
"
```

| 键 | 作用 |
|---|---|
| `k` | 启动 / 停止 C++ 控制环（planner 模式） |
| `i` | 1 秒内平滑过渡到初始姿势并切到 POSE 模式；清掉缓存的 chunk |
| `p` | 暂停 / 继续策略环 |
| `t` | 改 prompt（发布器发送 `prompt:<文本>`） |
| `[` / `]` | 切换初始姿势里左 / 右夹爪的开合（5.5 开、0 闭） |
| `c` / `s` / `f` | 只打印一行；数据导出器在跑时由它处理录制（开始 / 成功结束 / 失败结束） |

第一次上真机的顺序：`k` 起控制环，`i` 到初始姿势，用 `]` 和 `i` 看右夹爪是不是 5.5 开、0 闭（方向错了就是 Dex1 驱动读的单位不同，停下来查），再 `p` 放开策略。

## 4. 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--host` / `--port` | localhost / 5590 | 策略服务器 |
| `--action-publish-rate` | 50 | 每秒向 deploy 发几个 token。要和训练数据的 `dataset.fps` 一致，否则动作按错误速度回放 |
| `--action-horizon` | 40 | checkpoint 每次预测的步数 |
| `--rate` | 2.5 | 每秒请求几次推理；新 chunk 到达时跳过推理期间已过期的步 |
| `--schema` | crate | `crate`：ego + 双腕相机、token + 两只夹爪；`bottle`：ego + 右腕相机、单夹爪；state 都是官方 46 维 |
| `--gripper-units` | auto | checkpoint 预测和观测夹爪的单位。`auto`：从服务器 `identity` 里 checkpoint 的夹爪统计范围判定（米制在 ±0.1 内，Dex1 在 0 到 5.5）；`dex1`：0 闭、5.5 开（目标格式）；`sim_m`：仿真米制（转换器改写夹爪单位之前训出的 checkpoint）。显式给的值和统计矛盾时拒绝启动 |
| `--hand-wire` | dex1 | deploy 读 hand 字段的方式。`dex1`：值广播到 7 维（真机）；`sim`：米制写在两个指爪位（本仓库的仿真桥，做 sim2sim 时用） |
| `--init-token-file` | 无 | 运行的 `eval.initial_token` 文件。不给就用 NVIDIA 的站立 token，它是前倾姿势，不在我们的数据分布里 |
| `--prompt` | pick up the box | 语言指令；运行中用 `t` 改 |
| `--no-snap-tokens` | 关 | 不把 token 投影到 FSQ 码本 |
| `--no-identity-check` | 关 | 不核对服务器报告的字段顺序 |
| `--action-zmq-host` | * | 动作 PUB socket 绑定的地址。官方是 localhost（客户端和 deploy 同机）；`*` 让另一台机器上的 deploy 也能连 |
| `--verbose-timing` | 关 | 每帧打印 Telemetry 计时（默认只在一帧超时时打印） |

## 5. sim2sim

同一个客户端对接本仓库的 MuJoCo 仿真栈（实例 0 的端口正是上面的默认值）：`bash scripts/start_stack.sh eval` 起栈和相机流，策略服务器照第 1 步（`scripts/run_vla_server.sh <run>` 也提供 `get_action` 和 `identity`，也能用），客户端加 `--hand-wire sim`。仿真桥读米制的指爪位，`--hand-wire sim` 把 Dex1 值换算回去。用 `sim_m` 单位训练的 checkpoint 另加 `--gripper-units sim_m`。

## 6. 常见错误

- `PolicyServer not reachable at ...`：第 1 步没起，或 `--host` / `--port` 不对。官方客户端这时只警告；这里直接退出，因为字段顺序要靠服务器来核对。
- 服务器报 `does not exist here; pass backbone_path`：checkpoint 是别的机器训的，而 `COSMOS_BACKBONE_PATH`（默认 `~/Checkpoints/Cosmos-Reason2-2B`）下没有权重，脚本启动时也会先打印一行提示。把权重放到那里，或在 `locomani.local.sh` 里改 `COSMOS_BACKBONE_PATH`，或直接加 `--backbone-path`。
- 服务器报 `Embodiment tag ... is not supported by this checkpoint`：`--embodiment-tag` 和训练时不一致。
- `the served checkpoint's field order is not the crate schema's`：checkpoint 不是用 `configs/gr00t_sonic_3cam_bimanual.py` 训的，换 `--schema`，或确认运行目录。
- `gripper units: --gripper-units dex1 contradicts the checkpoint, whose gripper statistics are in sim_m`：checkpoint 的夹爪是仿真米制（转换器写 Dex1 单位之前训的），去掉 `--gripper-units` 让它自动判，或改成 `sim_m`。反过来同理。`cannot be inferred` 是服务器没报出夹爪统计（回放模式，或 checkpoint 缺 `statistics.json`），这时必须显式给。
- 一直打印 `[DEBUG] prepare_observation: waiting for camera msg..` / `waiting for state msg..` / `waiting for cameras [...]`：相机服务器没在发图、`g1_debug` 没有消息，或者缺了方括号里列出的相机 key；检查 `--camera-host`、`--state-zmq-host` 和机器人侧进程。
- `[Warning] action['motion_token'] max (...) > 1.25. Exceeds action bound, skipping.`：策略输出越界，和官方客户端一样整块丢弃；连续出现说明观测格式和训练不一致。
- `Error in inference: ... must be in observation` 之类的服务器报错：服务器的严格校验发现观测缺 key 或形状不对，多半是 `--schema` 和 checkpoint 不配。
