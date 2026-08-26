# 参考:模块地图 / 参数 / 诊断 / FAQ

> 查东西来这里:每个文件是干什么的、每个参数什么意思、哪个诊断脚本测什么。

### 模块地图

`src/teleop_free/` 有 60 多个文件,其中 **50 个是 `diag_*.py` 一次性测量脚本**。
真正的流水线只有下面这些:

| 文件 | 角色 |
|---|---|
| **`grasp_gym.py`** | **唯一采集入口**。一局的全部流程:采样 → 复位 → 走 → 接近 → 抓 → 抱稳,以及所有成功判据 |
| `primitives.py` | 动作原语:`walk_to` / `turn_to` / `reach` / `stand_up` / `carry_pose`,以及世界系↔骨盆系换算 |
| `stream_reach.py` | 伸手与抓取的 50 Hz 伺服状态机(`PlannerArm`、`grasp_streamed`) |
| `agent_server.py` | ZMQ 发布端,占据 Pico 遥操作服务器的位置(`frame` / `set_upper_body` / `token_frame`) |
| `reach_map.py` | 可达域体素图:骨盆系里哪些位置**抓得到**,以及站位联合求解 |
| `adaptive_grasp.py` | 从手当前的位置反推抓取朝向与进刀轴 |
| `arm_ik.py` | 单臂阻尼最小二乘 IK(骨盆系) |
| `grasping.py` | 由物体尺寸与 6D 位姿算出手该放哪 |
| `privileged_pub.py` | sim 侧:发布特权真值(:5581),并接收 episode 控制(:5582) |
| `run_sim_locomani.py` | sim 入口:上游 `run_sim_loop.py` 换成我们的双桌面场景 + 特权发布 + 三路相机 |
| `scene_random.py` | 逐局重掷场景外观(颜色、光照、地面),直接改 `mjModel`,不重建场景 |
| `sim_video.py` | 第三人称录像(ego + 双腕画中画、6D 线框盒、坐标读数) |
| `sim_vis.py` | `SIM_VIS` 开关的解析(见[可视化开关](#可视化开关)) |
| `paths.py` / `scripts/paths.sh` | 全仓库唯一的绝对路径来源(见[路径约定](setup.md#0-路径约定)) |
| `scene_lib.py` / `build_sim_scene.py` | 场景构建:物体、桌高、腕上相机、目标池都在这里 |
| `tools/convert_to_hf_format.py` | exporter 原始 session → LeRobot v2.1 |
| `tools/check_dataset.py` | **转完先跑这个**:结构 / modality / 抓取因果 / 时序对齐,四段审计([怎么用](dataset.md#转完先查一遍强烈建议)) |

脚本:

| 脚本 | 干什么 |
|---|---|
| `src/teleop_free/start_stack.sh` / `stop_stack.sh` | 起 / 停三进程 stack |
| `scripts/collect.sh` | **大规模采集**:按批换场景尺寸,批内逐局换目标/出生/外观 |
| `scripts/collect_pilot.sh` | **先导批**:单目标、外观钉死,只变出生点和物体位置,采到 N 条成功为止 |
| `scripts/export_rerun.py` | 把 episode 导成 Rerun `.rrd`,本地 `pip install rerun-sdk && rerun <file>` 看视频+曲线+3D |
| `scripts/install_scene.sh` | 解包并重建场景 |
| `scripts/paths.sh` | 所有 shell 脚本的路径来源 |
| `scripts/env_deploy.sh` | 编译 / 运行 C++ deploy 的环境(替代上游的 `install_deps.sh`) |
| `scripts/push_videos.sh` | 把一批录像替换进 `videos/` 并推送 |
| `scripts/par_batch.sh` | 多实例并行跑(迭代用,不落盘) |

不在流水线上的:`navigator.py`(FMM 导航,当前 gym 不走这条)、`zmq_driver.py`
(最早的 walk-only 验证)、`isaac_scene.py` / `compose_scene.py`(渲染检查)。

`collect_episode.py` 处在中间状态:`grasp_gym.py` 仍然从它 import 四个共享常量与
辅助函数(`PALM_DRIFT_MAX` / `STAND_SETTLE` / `obj_top` / `palm_frame_offset`),
但它**自己那条跑 episode 的路径是旧的**,只有 `FULL_TASK=1` 的完整任务才走它。

---

## 参数速查

### `grasp_gym.py` 的命令行参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--rounds N` | `5` | 跑几局 |
| `--record` | 关 | 驱动 exporter 落盘,只留通过全部判据的 |
| `--manifest PATH` | 无 | 每次**尝试**追加一行 JSON(目标、指令、成败、外观 seed、桌高、物体尺寸)。**要逐局指令就必须给** |
| `--seed N` | 随机 | 固定这一串局的采样,复现用 |
| `--targets` | `pool` | `pool` = 场景 `target_pool` 全体;`default` = 单一目标(旧行为);也可 `can,bottle` 指定 |
| `--spawn-off A [B]` | `1.0 3.0` | 出生在站位后方多远。一个值=固定,两个值=逐局在区间内取 |
| `--spawn-edge M [M2]` | 无(退回 `--spawn-off`) | 距**桌子**多远出生,覆盖 `--spawn-off`。站位离桌沿 0.17–0.32 m 且随目标移动,所以同一个 `--spawn-off` 每局的桌距都不同;这个是人真正会量的量,逐局二分解出。**本仓库实验用 `0.7`** |
| `--face-object` | 关(朝接近线) | 醒来正对**物体**而非接近线,`--spawn-yaw-err` 变成绕它的抖动。有横偏时两者不重合。**本仓库实验开** |
| `--spawn-lateral M` | `0.35` | 最多偏离接近线多远。`0` 会让每局都正对着走,那种修正模型就永远学不到。**本仓库实验用 `0.20`** |
| `--spawn-yaw-err DEG` | `90` | 初始朝向最多偏多少度,**在出生点**转出来。**本仓库实验用 `20`** |
| `--edge-jitter M` | `0.10` | 目标沿桌沿滑动的范围。默认只在 100 cm 桌沿的 20 cm 一段里动,接近段可能被学成"走到 +y 边",目标少的批次建议放宽到 `0.20` |
| `--until-ok N` | 无 | 跑到 **N 局成功**为止,`--rounds` 变成尝试次数上限 |
| `--dump-sample PATH` | 无 | 把这一局的场景采样写成 JSON,给 `diag_token_replay.py` 复现同一初始条件 |

### 环境变量

| 变量 | 默认 | 谁读 | 作用 |
|---|---|---|---|
| `LOCOMANI_WORKSPACE` | `/workspace` | 全部 | 仓库外路径的根,见[路径约定](setup.md#0-路径约定) |
| `SIM_VIDEO_DIR` | 未设 | sim | 设了才录第三人称视频。**必须在 `start_stack.sh` 之前 export,且必须是绝对路径**——sim 进程的 cwd 是 `third_party/GR00T-WholeBodyControl/`,不是仓库根,相对路径会写到那边去 |
| `SIM_ONSCREEN` | `0` | sim | `1` 开一个实时 MuJoCo 窗口(跟拍 pelvis)。**必须在 `start_stack.sh` 之前 export**,需要真的 `$DISPLAY`。grasp_gym.py 只是连过去的客户端,不管渲染,所以这个开关在 sim 这边,不在 gym 那边 |
| `SIM_VIS` | `full` | sim + gym | 调试可视化开关,见[下一节](#可视化开关) |
| `SCENE_RANDOM` | `1` | sim | `0` 固定外观,见[外观](collection.md#4-外观每局重掷) |
| `WRIST_CAMS` | `1` | sim + exporter | `0` 只录 ego 一路,见[相机](collection.md#相机) |
| `SCENE_SEED` | 未设 | `build_sim_scene.py` | 重建时重掷桌高与物体尺寸 |
| `INSTANCE` | `0` | 全部 | 端口块 + DDS domain,多实例并行 |
| `SCENE` | `locomani` | sim | `default` 换成上游空场景 |
| `SONIC_POLICY` | `policy/release` | deploy | 换 checkpoint(`policy/sonic_v1_1` 可用,但步长与时序要重测) |
| `NO_ENCODER` | `0` | deploy | `1` = **不加载 encoder**,token-only / VLA 部署配置 |
| `HAND` | `dex1` | 全部 | `dex1` 平行夹爪 / `dex3` 三指手 |
| `GYM_ARM_PIN` | `0` | gym | `1` 恢复 sim 侧关节 pin(动作更准,但录下的 latent 不再是产生该动作的 latent) |
| `GYM_ARM_CHANNEL` | `planner` | gym | `streamed` 切回旧的流式手臂通道 |
| `GYM_RESTART_DEPLOY` | `1` | gym | 每局重启 deploy。关掉会让第 2 局起漂移、摔倒 |
| `FULL_TASK` | `0` | `collect_episode.py` | 抓起 → 搬运 → 放下的完整任务(尚未迁移到新控制逻辑) |

**采数据时真正会去动的只有四个**:`SIM_VIDEO_DIR`、`SIM_VIS`、`WRIST_CAMS`、
`SCENE_SEED`。其余是排查问题或换配置才碰。

### 端口

`INSTANCE=i` 时全部加 `10*i`,DDS domain 也用 `i`,多实例互不干扰;只有 instance 0
挂 exporter 落盘。

| 端口 | 方向 | 内容 |
|---|---|---|
| 5555 | sim → exporter | 三路相机 RGB |
| 5556 | **agent → deploy** | `command` / `planner` / `pose` 三个 topic,机器人的全部意图 |
| 5557 | deploy → exporter | `g1_debug`,**录下来的 64 维 token 从这里来** |
| 5580 | agent → exporter | 录制控制键(`c` 开始/保存,`x` 丢弃) |
| 5581 | sim → agent | 特权真值(物体位姿、掌位、接触、ego 可见性) |
| 5582 | agent → sim | episode 控制(teleport、pin、ghost、录像开关) |

### 可视化开关

四个调试可视化由一个变量 `SIM_VIS` 控制:

```bash
SIM_VIS=full            # 默认,全开
SIM_VIS=none            # 录像完全干净
SIM_VIS=bbox,readout    # 只留 6D 线框盒 + 坐标读数
# 可选项: reach | ghost | bbox | readout
```

| 项 | 是什么 | 关掉省什么 |
|---|---|---|
| `reach` | 可达域点云:淡绿 = 能抓到的包络,**深绿核心 = 舒适区** | **约 5800 个 box geom** 的模型编译与渲染开销 |
| `ghost` | 洋红色的"幽灵夹爪",停在规划出的抓取位姿上,含抓取中心小球 | 渲染开销 |
| `bbox` | 目标与干扰物的 6D 定向线框盒 | 少量 |
| `readout` | 烧在录像里的骨盆系坐标文字 | 无 |

**这些从来没有污染过训练数据。**`reach` 和 `ghost` 在 geom group 4,MuJoCo 默认可视化
选项就隐藏它;三路相机和 exporter 用的正是默认选项。`bbox` / `readout` 只画进第三人称
mp4。所以这些开关是为了**录像干净**和**省开销**,不是为了数据干净。

两个实现细节:`reach` 的开销只有"不注入"才省得下来,所以它在 sim 启动时读一次,
**跑起来之后改不了**;`ghost` 是建场景时烘进 XML 的,关不掉它的可视化,只能让 gym
**不去摆它**(留在地板下面的出生点)—— 因此 **`SIM_VIS` 要同时给 stack 和 gym 设**:

```bash
export SIM_VIS=none                                   # 对两个进程都生效
SIM_VIDEO_DIR="$PWD/outputs/sim_video" bash src/teleop_free/start_stack.sh
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/teleop_free/grasp_gym.py --rounds 3
```

---

## 常用诊断脚本

`src/teleop_free/diag_*.py`(都用 `.venv_sim` 的 python,在 stack 起来后运行)。
结论都写在各自的 docstring 里 —— **它们是这个仓库大部分"为什么是这个数"的出处**。

| 脚本 | 回答什么 |
|---|---|
| `diag_channel.py` + `diag_channel_report.py` | planner 通道 vs streamed 通道的手臂投递精度(216 样本) |
| `diag_squatmode.py [--sweep]` | `IDEL_SQUAT` 的蹲深,以及 `height` → 骨盆高度的标定曲线 |
| `diag_lean.py` | 命令的腰部前倾有多少被交付(答案:2–10%,基本不交付) |
| `diag_height.py` | `height` 字段在 IDLE / SLOW_WALK 下是否有效(答案:没有) |
| `diag_ego_geom.py` | 物体在 ego 相机里的可见距离区间,随躯干前倾的变化 |
| `diag_token_replay.py --phase capture/replay` | 录下的 token 能否复现动作 |
| `diag_shuffle.py` | 每个 speed 命令实际走多快、一步多长 |
| `diag_turn.py` | 原地转身:每个模式、每个角度要多久,以及控制器的朝向死区 |
| `diag_reach_probe.py` | reach map 的实测校验 |
| `diag_ik_grasp.py` | 纯运动学抓取 bench(不经过 policy) |

数据集本身的检查不在这里,在 [`tools/check_dataset.py`](dataset.md#转完先查一遍强烈建议) —— 它不需要 stack 起着,只读磁盘上已经转好的数据集。

---

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `start_stack.sh` 卡在 `[2/3]` 十分钟 | 首次运行在编译 TensorRT engine,正常。之后走缓存,20–40 s |
| 没打印 `stack up` | 去 `$LOCOMANI_TMP/p03/{sim,deploy,exporter}.log` 看是哪一个没起来 |
| deploy 起来了但机器人不动 | 检查 :5556 上是不是有**多个** commander 在发。多个进程往同一个 deploy 发冲突意图,表现就是发呆或抽搐 |
| 每局都 `teleport did not land` | 上一局的 deploy 状态没清干净。`GYM_RESTART_DEPLOY=1`(默认)每局重启 deploy 就是为这个 |
| 没有第三人称录像 | `SIM_VIDEO_DIR` 必须在 `start_stack.sh` **之前**给,跑起来之后设无效 |
| 录像里没有绿色点云 | `SIM_VIS` 里没有 `reach`。它在 sim 启动时读一次,跑起来之后改不了 |
| `SIM_VIS=none` 但幽灵夹爪还在 | `SIM_VIS` 只给了 stack,没给 gym。两个进程都要设 |
| 录下来的 `planner_height` 全是 -1 | `patches/vla_action_space.patch` 没打。整段下蹲会从 action 里消失 |
| action 是 267 维而不是 78 维 | 同上,补丁没打 |
| `[manifest] ...: N produced rounds but M episodes on disk` | manifest 和 session 不是同一批。一批写一个 manifest、填一个 session,别混用 |
| `[manifest] ... join is out of phase` | 更糟:数量对上了但成败对不上。**别转这个 session**,指令会标错 |
| `check_dataset.py` 报 `action: MISSING ['left_hand_joints', ...]` | modality 的 action key 被改过名。GR00T 只 warn 不报错,照训但两只手会被静默丢掉 —— 必须修完再训 |
| `check_dataset.py` 报 `lag distribution {3: ...}` 之类的非零 | 视频和表差了那么多帧。先查 exporter 的时间戳,再查转换时的 50→30 Hz 重采样 |
| `check_dataset.py` 报 `body trails the token by -N frames` | action 和 state 在时间上被搞反了,是最严重的一类,**不要拿去训** |
| `[gym] targets not in the scene` | 场景是旧的。`bash scripts/install_scene.sh` 会解包并重建 |
| 采完之后场景尺寸不对了 | `collect.sh` 就地重建场景。`bash scripts/install_scene.sh` 恢复 |
| 换了机器路径全错 | `export LOCOMANI_WORKSPACE=<你的根>`,然后 `python src/teleop_free/paths.py` 自查 |
| 场景 MJCF 找不到 mesh | 重跑 `bash scripts/install_scene.sh`;它会探测并改写 tarball 里烘死的路径 |
| `ValueError: ... Error opening file ... scene_31dof.xml` | 新 clone 的 `third_party` 里没有 dex1 生成文件。先跑 `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/teleop_free/make_dex1_model.py`,再 `bash scripts/install_scene.sh` |
| `ModuleNotFoundError: No module named 'skfmm'` | `.venv_sim` 里没装 `scikit-fmm`(上游安装脚本不带,这个仓库自己的依赖)。`uv pip install --python third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scikit-fmm` |
