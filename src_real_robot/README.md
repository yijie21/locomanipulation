# 真机部署：GR00T 策略 + SONIC 控制器跑在 G1 上

把本仓库训练出的 GR00T checkpoint 部署到实验室的真机 G1（29 自由度、Dex1 夹爪、头部 D455 和双腕 D405）。推理客户端的结构是 NVIDIA 官方的 `run_vla_inference.py`（来源见 [UPSTREAM.md](UPSTREAM.md)），观测和动作格式是本仓库数据集的。设计和决策见[设计文档](../docs/handoff/real-robot-sim2real-plan-20261006.md)。

需要三台角色，可以是同一台机器：机器人本体的控制电脑（跑 SONIC C++ deploy 和相机服务器）、一台 GPU 机器（跑策略服务器）、一台跑本客户端的机器（装了 SONIC 的 `.venv_sim`）。

## 0. 前提

- 机器人上已装好并验证过：带 Dex1 驱动的 SONIC C++ deploy（1.1）、三相机服务器（key 为 `ego_view`、`left_wrist`、`right_wrist`，640×480）。本仓库不替换这两样。
- 机器人侧的 ZMQ 端口沿用官方默认：状态 `g1_debug` 5557、相机 5555、动作 `pose` 5556、键盘 5580。换了端口就在下面的命令里传对应参数。
- 本机已完成[安装](../docs/setup.md)第 0–4 步（SONIC 的 `.venv_sim`）。策略服务器那台机器要第 5 步（GR00T 训练环境）。

## 1. 起策略服务器（GPU 机器）

```bash
bash scripts/run_vla_server.sh outputs/runs/<run> --gpu 0
```

做什么：用训练这个运行时的 modality 配置和 embodiment tag 加载 checkpoint，在 5590 端口提供 `get_action_raw` 和 `identity`。
得到什么：日志里出现 `listening` 一行。客户端启动时会核对 `identity` 报告的字段顺序和 `--schema` 一致，不一致就拒绝启动。

## 2. 起机器人侧（机器人控制电脑）

按机器人上现有的流程启动 C++ deploy 和相机服务器。本客户端只要求它们监听上面的端口。

## 3. 起客户端

```bash
bash src_real_robot/run_inference.sh --host <策略服务器地址> --camera-host <机器人地址> --state-zmq-host <机器人地址> \
    --init-token-file configs/deployment/warehouse_initial_token.json --prompt "pick up the box"
```

做什么：连接策略服务器，核对 checkpoint 字段顺序；订阅机器人状态和三路相机；在 5556 端口发布 protocol v4 的 token 消息；启动时处于暂停状态。
得到什么：终端打印 `policy loop at 50 Hz`，之后每到一个 chunk 打印一行延迟，每秒打印一行发布的帧号和两只夹爪的值。

键盘（另开一个终端跑 SONIC 的键盘发布器）：

| 键 | 作用 |
|---|---|
| `k` | 启动 / 停止 C++ 控制环（planner 模式） |
| `i` | 1 秒内平滑过渡到初始姿势并切到 POSE 模式；清掉缓存的 chunk |
| `p` | 暂停 / 继续策略环 |
| `t` | 改 prompt（发布器发送 `prompt:<文本>`） |
| `[` / `]` | 切换初始姿势里左 / 右夹爪的开合（5.5 开、0 闭） |

第一次上真机的顺序：`k` 起控制环，`i` 到初始姿势，用 `]` 和 `i` 看右夹爪是不是 5.5 开、0 闭（方向错了就是 Dex1 驱动读的单位不同，停下来查），再 `p` 放开策略。

## 4. 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--host` / `--port` | localhost / 5590 | 策略服务器 |
| `--action-publish-rate` | 50 | 每秒向 deploy 发几个 token。要和训练数据的 `dataset.fps` 一致，否则动作按错误速度回放 |
| `--action-horizon` | 40 | checkpoint 每次预测的步数 |
| `--rate` | 2.5 | 每秒请求几次推理；新 chunk 到达时跳过推理期间已过期的步 |
| `--schema` | crate | `crate`：ego + 双腕、60 维 state、token + 两只夹爪；`bottle`：ego + 右腕、单夹爪 |
| `--gripper-units` | dex1 | checkpoint 预测和观测夹爪的单位。`dex1`：0 闭、5.5 开（目标格式）；`sim_m`：仿真米制（转换器改写夹爪单位之前训出的 checkpoint） |
| `--hand-wire` | dex1 | deploy 读 hand 字段的方式。`dex1`：值广播到 7 维（真机）；`sim`：米制写在两个指爪位（本仓库的仿真桥，做 sim2sim 时用） |
| `--init-token-file` | 无 | 运行的 `eval.initial_token` 文件。不给就用 NVIDIA 的站立 token，它是前倾姿势，不在我们的数据分布里 |
| `--no-snap` | 关 | 不把 token 投影到 FSQ 码本 |
| `--no-identity-check` | 关 | 不核对服务器报告的字段顺序 |

## 5. sim2sim

同一个客户端对接本仓库的 MuJoCo 仿真栈（实例 0 的端口正是上面的默认值）：`bash scripts/start_stack.sh eval` 起栈和相机流，策略服务器照第 1 步，客户端加 `--hand-wire sim`。仿真桥读米制的指爪位，`--hand-wire sim` 把 Dex1 值换算回去。用 `sim_m` 单位训练的 checkpoint 另加 `--gripper-units sim_m`。

## 6. 常见错误

- `no policy server at ...`：第 1 步没起，或 `--host` / `--port` 不对。
- `the served checkpoint's field order is not the crate schema's`：checkpoint 不是用 `configs/gr00t_sonic_3cam_bimanual.py` 训的，换 `--schema`，或确认运行目录。
- 一直打印 `waiting for the cameras and the robot state`：相机服务器没有发出全部三路 key，或 `g1_debug` 没有消息；检查 `--camera-host`、`--state-zmq-host` 和机器人侧进程。
- `[chunk] skipped: motion_token leaves [-1.25, 1.25]`：策略输出越界，和官方客户端一样整块丢弃；连续出现说明观测格式和训练不一致。
