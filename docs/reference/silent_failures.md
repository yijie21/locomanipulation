# 参考:模块地图 / 参数 / 诊断 / FAQ

> 历史排查线索：本文件的路径、默认值与实验解释需逐项核验。当前开发入口见 [change-map.md](change-map.md)，证据规则见 [standards.md](standards.md#evidence)。

### 模块地图

`src/locomani/` 有 60 多个文件,其中 **50 个是 `diag_*.py` 一次性测量脚本**。
真正的流水线只有下面这些:

| 文件 | 角色 |
|---|---|
| **`grasp_gym.py`** | **唯一采集入口**。一局的全部流程:采样 → 复位 → 走 → 接近 → 抓 → 抱稳,以及所有成功判据 |
| `primitives.py` | 动作原语:`walk_to` / `turn_to` / `reach` / `stand_up` / `carry_pose`,以及世界系↔骨盆系换算 |
| `stream_reach.py` | 伸手与抓取的 50 Hz 伺服状态机(`PlannerArm`、`grasp_streamed`) |
| `teleop_arm.py` | **VR 3-point 通道**的手臂接口(`TeleopArm`):腕/颈位姿目标,deploy 的 encoder 走 `teleop` 模式,手臂与腰由 policy 自己解;没有 IK。配套执行器 `tasks/bottle/teleop_grasp_policy.py`,开关 `ReachConfig.teleop_arm`(ledger bottle.md §35) |
| `agent_server.py` | ZMQ 发布端,占据 Pico 遥操作服务器的位置(`frame` / `set_upper_body` / `token_frame`) |
| `reach_map.py` | 可达域体素图:骨盆系里哪些位置**抓得到**,以及站位联合求解 |
| `adaptive_grasp.py` | 从手当前的位置反推抓取朝向与进刀轴 |
| `arm_ik.py` | 单臂阻尼最小二乘 IK(骨盆系) |
| `upper_ref.py` | 上身参考通道:`RefTap` 读 deploy 的 `body_q_target`(被 `upper_body_position` 替换掉的那个量),`UpperRamp` 限速下发并附带速度。`--selfcheck` 不需要 stack |
| `grasping.py` | 由物体尺寸与 6D 位姿算出手该放哪 |
| `privileged_pub.py` | sim 侧:发布特权真值(:5581),并接收 episode 控制(:5582) |
| `run_sim_locomani.py` | sim 入口:上游 `run_sim_loop.py` 换成我们的双桌面场景 + 特权发布 + 三路相机 |
| `scene_random.py` | 逐局重掷场景外观(颜色、光照、地面),直接改 `mjModel`,不重建场景 |
| `sim_video.py` | 第三人称录像(ego + 双腕画中画、6D 线框盒、坐标读数) |
| `sim_vis.py` | `SIM_VIS` 开关的解析(见[可视化开关](#可视化开关)) |
| `tasks/bottle/place_phase.py` | pick-and-place 的 place 阶段(raise / turn-to-band / carry-over / descend / release / retreat / verify),由 `BottleTaskConfig.goal` 接在 lift 之后;`tasks/pick_place/` 只是注册 |
| `paths.py` / `scripts/paths.sh` | 全仓库唯一的绝对路径来源(见[路径约定](../setup.md#0-路径约定)) |
| `scene_lib.py` / `build_sim_scene.py` | 场景构建:物体、桌高、腕上相机、目标池都在这里 |
| `tools/convert_to_hf_format.py` | exporter 原始 session → LeRobot v2.1 |
| `tools/check_dataset.py` | **转完先跑这个**:结构 / modality / 抓取因果 / 时序对齐,四段审计([怎么用](../dataset.md#转完先查一遍强烈建议)) |

脚本:

| 脚本 | 干什么 |
|---|---|
| `locomani.stack.cli` | 起 / 停三进程 stack（`up` / `down --instance <N>`）。**停只用 `locomani.stack.cli down`,不要自己手写 kill** —— 它多杀 `spawn_main`(sim 的图像发布子进程,孤儿化后占着 :5555,下一次启动会被 `locomani.stack.cli up` 拒绝)和 `serve_policy.py`,还会等端口真正释放并回显剩余进程 |
| `scripts/run_overnight.sh` | **跑一次完整实验**:12 局闸门 → 采集 → 转换 → 审计 → 训练 → 闭环。`RUN=` 命名,`SPAWN_EDGE=` 定出生距离 |
| `scripts/run_closed_loop.sh` | **只评一个 checkpoint**:起 stack + 策略服务器,20 轮再续到 100。`CKPT=` 默认取训练最久的那个 |
| `scripts/archive/` | round 2/3/4 和旧 pipeline 的编排脚本。**历史记录,不维护,已知跑不起来**(调用的 `eval_closed_loop` 参数已不存在),留着是因为阶梯上那几个数的确切配置只写在里面 |
| `scripts/collect.sh` | **大规模采集**:按批换场景尺寸,批内逐局换目标/出生/外观 |
| `scripts/collect_pilot.sh` | **先导批**:单目标、外观钉死,只变出生点和物体位置,采到 N 条成功为止 |
| `scripts/export_rerun.py` | 把 episode 导成 Rerun `.rrd`,本地 `pip install rerun-sdk && rerun <file>` 看视频+曲线+3D |
| `scripts/install_scene.sh` | 解包并重建场景。`SCENE_PRESET` 在**这一行**上给才有效;不给 = 默认多物体桌面。venv 不存在时不重建并以非零退出 |
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

### ScaleBFM 线(`backends/scalebfm` + `tasks/bottle_visual`,2026-09-10 起)

SONIC 的候选替代 controller,单进程、无 stack、无墙钟;教程 [scalebfm_collection.md](../scalebfm_collection.md),台账 [ledger/scalebfm.md](../ledger/scalebfm.md)。

| 文件 | 角色 |
|---|---|
| `backends/scalebfm/runtime.py` | `Sim`(MuJoCo + ScaleBFM PD 200 Hz/decimation 4 + dex1 夹爪 PD)、`Policy`(9 tensor 观测、history 3、mode 2)、`Refs`(未来 6 帧) |
| `backends/scalebfm/network.py` | vendored HumanoidTransformer + `load_policy()`;资产在 `assets/scalebfm/`,FK MJCF **必须**是 ScaleTrack 那份 |
| `backends/scalebfm/config.py` | `ScaleBfmBackendConfig`:checkpoint、scene_dir、object_name、grip_kp/kd(默认 1.0/0.025) |
| `backends/scalebfm/{action,executor,chunks,reference,state,camera}.py` | 29 维原生动作编解码、25→50 Hz 执行器、延迟 chunk、SE(3) 参考、91 维学生状态、双相机观测 |
| **`tasks/bottle_visual/round.py`** | **唯一 expert 入口**:`run_case` 与 `python -m locomani.tasks.bottle_visual.round --config configs/tasks/bottle_visual.json` |
| `tasks/bottle_visual/config.py` | `BottleVisualTaskConfig`(默认 = yplus200 采集 expert)+ `GraspConfig` |
| `tasks/bottle_visual/{planner,motion,observation,sensor,visibility}.py` | 决策(walk/reach/entry/close/lift/carry)、连续参考、带延迟/遮挡/量化的场景估计、可见性门 |
| `tasks/bottle_visual/{scene,sampling,geometry}.py` | 三个桌高/深度层、`yplus_center_v1` 重置分布、边缘几何常量 |
| `tasks/bottle_visual/{audit,replay,recovery,harness}.py` | 离线验收、精确动作重放契约、恢复扰动、共享 harness |
| `tasks/bottle_visual/pipeline/` | 这条线自己的 collect / export / train / eval / video_check 驱动(尚未并入 `locomani/pipeline`) |

静默失败:`config.json` 里的 `source_sha256` 覆盖整条线的源码(`pipeline/provenance.line_sources`),改任何一个文件后往旧目录续跑会直接拒绝——这是设计,用新的实验 ID。

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
| `LOCOMANI_WORKSPACE` | `/workspace` | 全部 | 仓库外路径的根,见[路径约定](../setup.md#0-路径约定) |
| `SIM_VIDEO_DIR` | 未设 | sim | 设了才录第三人称视频。**必须在 `locomani.stack.cli up` 之前 export,且必须是绝对路径**——sim 进程的 cwd 是 `third_party/GR00T-WholeBodyControl/`,不是仓库根,相对路径会写到那边去 |
| `SIM_ONSCREEN` | `0` | sim | `1` 开一个实时 MuJoCo 窗口(跟拍 pelvis)。**必须在 `locomani.stack.cli up` 之前 export**,需要真的 `$DISPLAY`。grasp_gym.py 只是连过去的客户端,不管渲染,所以这个开关在 sim 这边,不在 gym 那边 |
| `SIM_VIS` | `none`(2026-09-10 前是 `full`) | sim + gym | 调试可视化开关,见[下一节](#可视化开关) |
| `TF_DATASET_FPS` | `30` | vla_round | 闭环回放的发布帧率,**必须等于 checkpoint 训练数据的 fps**。转换器 2026-08-28 起写 25,所以用重转数据训出来的 checkpoint 要 `--set backend.dataset_fps=25`(2026-09 起环境变量 `TF_DATASET_FPS` 不再被读取,只有 `SonicBackendConfig.dataset_fps`);不匹配不会报错,只是动作按错速度回放 |
| `SCENE_RANDOM` | `0` | sim | **默认关**(外观固定)。`1` 每局重掷颜色/光照/地面,见[外观](../collection.md#6-调整任务) |
| `WRIST_CAMS` | `1` | sim + exporter | `0` 只录 ego 一路,见[相机](../collection.md#相机与动作) |
| `SCENE_SEED` | 未设 | `build_sim_scene.py` | 重建时重掷桌高与物体尺寸 |
| `INSTANCE` | `0` | 全部 | 端口块 + DDS domain,多实例并行。**单臂采集必须用 `0`**:stack manager 在 instance≠0 且未设 `RECORD_INSTANCE=1` 时跳过 exporter(`[3/3] data exporter — skipped (instance 1 records no episodes)`),`--record` 采不出任何 session;crate 的采集可设 `collect.instance=<N>` 来改变实例。**而且 gym 不跟随 `INSTANCE`**:stack 挪到 5565/5566/5591 之后 grasp_gym 仍然绑 :5556、等 :5581,10 秒后 `no privileged state on :5581`。要换实例得把 `TF_*` 端口一起 export,光给 `INSTANCE` 不够。想避开别人又要采集,正确做法是 **instance 0 + `CUDA_VISIBLE_DEVICES=<另一块卡>`**——实例号只管端口和 exporter,GPU 是独立变量(2026-09-10) |
| `SCENE` | `locomani` | sim | `default` 换成上游空场景 |
| `SONIC_POLICY` | **`policy/sonic_v1_1`**(2026-09-11 起;此前是 `policy/release`) | deploy + 离线 encoder/decoder | 跑哪个 SONIC checkpoint。默认值只写在 `scripts/paths.sh`(shell)与 `locomani.paths.SONIC_POLICY`(python),`locomani.stack.cli up` / `locomani.stack.cli restart-deploy` 里的是同值兜底。**在 `locomani.stack.cli up` 之前 export**。切回旧 checkpoint 用 `SONIC_POLICY=policy/release`。三件随之而变的事:① encoder 观测维度 1751(release 1762)且 anchor 用 heading 归一化(orientation_mode 1),离线工具已改为从 checkpoint 自己的 `observation_config.yaml` 读布局;② **token 是 checkpoint 私有 latent**,见下一节;③ 抓取线的时序仍是 release 标定的(`TRACK_ROT_BOOST=4.5`、approach/brake 带),未重测。实测:v1_1 腰跟踪远好于 release(腰 RMSE 3.9° vs 12.9°,台账 bottle §16),但同一 case bank 上抓取 0/20 vs release 9/20(§17) |
| `NO_ENCODER` | `0` | deploy | `1` = **不加载 encoder**,token-only / VLA 部署配置。**采集(planner 模式)禁用**:没有 encoder 时上肢/下蹲参考被静默丢弃,抬臂阶段停滞(2026-09-03) |
| `HAND` | `dex1` | 全部 | `dex1` 平行夹爪 / `dex3` 三指手 |
| `GYM_ARM_PIN` | `0` | gym | `1` 恢复 sim 侧关节 pin(动作更准,但录下的 latent 不再是产生该动作的 latent) |
| `GYM_ARM_CHANNEL` | `planner` | gym | `streamed` 切回旧的流式手臂通道 |
| `restart_deploy`(`BottleTaskConfig`) | `false`(2026-09-10 前是每局重启) | gym | 每局重启 deploy。现在复位后用 reanchor 清朝向锚,不重启;每局 34 s 而不是 64 s(台账 bottle §11)。评测路径仍固定重启 |
| `FULL_TASK` | `0` | `collect_episode.py` | 抓起 → 搬运 → 放下的完整任务(尚未迁移到新控制逻辑) |

**采数据时真正会去动的只有四个**:`SIM_VIDEO_DIR`、`SIM_VIS`、`WRIST_CAMS`、
`SCENE_SEED`。其余是排查问题或换配置才碰。

### 一条通则:找不到就要吵

同一类 bug 一天之内出现了六次,都是**操作失败但退出码是成功**,都在几十行之后以
完全无关的面貌爆出来。最后两条尤其值得看:一条是前一条的**修复过程本身**,另一条
发生在**检查器**里 —— 检查失败时也会静默地报告"没问题":

| 操作 | 失败时 | 后果 |
|---|---|---|
| `cd "$EMPTY"` | 返回 0,原地不动 | 几个 GB 装进上游 clone(gitignored,`git status` 不提醒) |
| `. scripts/paths.sh` 从错误目录 | 短路,变量保持未设置 | 同上 |
| `str.replace(pattern, ...)` 模式不匹配 | 原样返回 | ego 相机停在原厂朝向,全局 0% 成功率,而采集/录像/转换全部"正常" |
| `cam.quat = [...]` 而 `alt.type` 是 EULER | 赋值被忽略 | 上一行那个 bug 的**修复本身**同样静默失败:fovy 生效了、朝向没有 |
| zsh 里 `set -- $var` | 不做单词分割 | 拼出来的 pgrep 模式变成 `run_.sh`,匹配不到 —— **检查器报告的不是世界的状态**,误判流水线已死 |
| 官方 `open_loop_eval.py` 的 `--steps` 默认 200 | `min(steps, len)` 静默截断 | 只评了 546–832 帧 episode 的前 200 帧(走路段),抓取根本没评到,却给出一个看起来正常的 MSE |
| `ISAACLAB_TO_MUJOCO` / `MUJOCO_TO_ISAACLAB` 取反了 | 图照跑,照样返回 64/29 个数 | 两个数组的**名字与内容是反的**(`ISAACLAB_TO_MUJOCO[mujoco_index] → isaaclab index`),所以"把 MuJoCo 序转成 IsaacLab 序"要用**名字读起来相反的那个**。`sonic_decoder.py` 已经写了这条警告,`sonic_encoder.py` 第一版还是踩了 —— 唯一露馅的地方是:把参考里的肘抬 20°,解码端回答的是**膝盖**。改对之后 top-2 响应落回同一条手臂 |
| 不发 `upper_body_velocity` | 上一次发过的值**永远**留在 reference 里 | `input_interface.hpp:381-390` 读 buffer 没有过期判定,`g1_deploy_onnx_ref.cpp:2985` 又把有效位 `std::ignore` 掉:只要还在发 position,速度就一直用最后收到的那一组。从来没发过则是硬 0 —— 位置在扫、参考速度读 0,而 encoder 的 mode-0 活输入里 290/644 是速度。收尾必须**显式发零**,不能靠省略 |
| frozen 接管时拿**实测**关节当种子 | 看着连续,参考跳 123°(单帧) | 被替换的是 k-planner 自己的手臂参考(`cpp:781` 逐字写进 `current_motion_` 的 10 个 lookahead 帧),不是实测姿态;两者相差整个 decoder 的跟踪误差(平均 15.5°,wrist pitch 上 61–65°)。种子要取 `body_q_target`(g1_debug key 22)—— 它就是那个量,**不带 default-angle 偏移**(`output_interface.hpp:192`,和 `body_q` 不同),而且替换只写进 observation buffer、从不写回 motion,所以覆盖期间它照样报 k-planner 的手臂。见 [collection.md](../archive/collection_20260914.md#上身接管两个台阶和它们的修法2026-08-28) |
| 部署帧率和训练数据的 fps 不一致 | 不报错,动作按 1.2× / 0.83× 速度回放 | chunk 是一串**帧**,发布速率必须等于训练数据的 fps。转换器 2026-08-28 起写 25(`DST_FPS`),而 `SonicBackendConfig.dataset_fps` 默认仍是 30(现有 checkpoint 都是 30 训的)—— **第一个用重转数据训的 checkpoint 必须显式带 `--set backend.dataset_fps=25`**(旧文档写的 `TF_DATASET_FPS=25` 在 2026-09 重构后已无效,eval_closed_loop 只读 config)。两边都是常数,谁都不知道对方,所以只能靠这条记着 |
| 编排脚本自己手抄一份 stack 清理逻辑 | 三个组件都报 ok,但一张图都没发 | sim 的图像发布器是 multiprocessing **子进程**,`kill -9` 父进程会把它孤儿化、继续占着 :5555。抄漏 `pkill -f spawn_main` 的那一份就是这么坏的,而 `kill_stack()` 一度有**四份**拷贝。好在 `locomani.stack.cli up` 检测到"组件都起来了但没有图像"会拒绝交出 stack,否则就是一个安静地渲染不出画面的 stack。**停 stack 只用 `locomani.stack.cli down`**,它还多杀 `serve_policy.py`、等端口真正释放、并回显剩余进程 |
| 在进程 cmdline 里做**子串**匹配(pgrep -f 同类) | 匹配到把整条命令当参数的 shell / 祖先进程 | `pkill -f` 杀掉发起它的 shell(exit 144,两次);watchdog 把两个健康 sim 判成 gone —— 换成扫 /proc 也一样,针在祖先的 cmdline 里。免疫的是**匹配结构**:模式必须是独立的 argv 元素,shell 把命令塞进单个 `-c` 参数时过不了 |
| 两个 runner 同一秒抢同一 instance 的端口块 | 两边都打印 "stack up",rc 0 | 一边的 gym 报 `Address already in use :5556`,另一边每局 1 s 报 `teleport landed 1.85 m off`(2026-09-10,pick-place session 与 session 29)。端口检查是 TOCTOU;2026-09-10 起 `locomani.stack.cli up` 用 `/workspace/tmp/locomani_locks/instance_<i>.lock` 上 `flock -n`,锁的 fd 被 sim/deploy/exporter 继承,任一存活就拒绝第二个栈(退出码 3,打印持有者),全部死掉自动释放;`locomani.stack.cli down` 只杀本 instance 的进程(按 comm=python + 环境里的 `TF_CAMERA_PORT` 匹配)。起栈前仍要读 STATE 的 RUNNING 行 |
| `build_sim_scene.py` 不带 `--harden-pads` 重建(或 `HARDEN_PADS=0 install_scene.sh`) | 场景正常,指面回到软接触 | 持物几秒后蠕变滑脱(2026-09-14:同 20 seed 软垫 5 次滑脱、硬垫 0 次);`install_scene.sh` 回显 `finger pads HARDENED`,或看 XML 里 `dex1_col_1` 有没有 `priority="1"` |
| place 伺服按 `upper_ref.MAX_STEP`(2°/tick)限幅 | 手臂看着在动 | 冗余 IK(腰 + 臂)逐拍换姿态,持物滑脱 3–4 cm,像接触蠕变;`place_trace` 10 Hz 漂移曲线能分辨(2026-09-10) |
| **用 A checkpoint 的 token 喂 B checkpoint 的 decoder** | 数字全在合法范围,图照跑 | token 是 **checkpoint 私有的 FSQ latent**。两个 checkpoint 的码本完全相同(都是 1/16 网格、`[−1.0, 0.9375]`,2026-09-11 实测),所以任何"取值合法 / 在网格上 / 在训练分布内"的检查都过 —— 变的是**含义**。2026-09-11 默认切到 `policy/sonic_v1_1` 之后:此前所有数据集(`action.motion_token`)与 GR00T checkpoint 都是 release 空间的,拿去 v1.1 部署是安静的胡话;离线 `eval_open_loop` / `sonic_decoder` 的"度数"指标同理。要复现旧数字就 `SONIC_POLICY=policy/release` 整条链一起切。运行记录里已经写进 `config_snapshot.json` 的 `sonic_policy` 字段(2026-09-11 起) |
| 离线 encoder 的观测布局写死成某个 checkpoint 的 | 图照跑,返回 64 个数 | v1.1 的 encoder 输入是 **1751** 维、anchor 块叫 `motion_anchor_orientation_heading_10frame_step5`(偏移 584,**orientation_mode 1**:左乘的是机器人 heading 而不是整个 base 四元数);release 是 1762 / 偏移 601 / mode 0。维度对不上时 `SonicEncoder` 会直接抛错,但**偏移和 anchor 算法对不上不会** —— 所以 `sonic_encoder.py` 现在从 checkpoint 自己的 `observation_config.yaml` 读布局(`EncoderLayout.from_config`),并按 anchor 块名字选 heading 与否,不再写死 |
| deploy 树里没有 `$SONIC_POLICY` 那个目录 | 曾经:deploy 在 ONNX Runtime 里退出,栈表现为超时 | `$SONIC_POLICY` 是相对 deploy 的 cwd 解析的,而 `LOCKSTEP_DEPLOY_ROOT` / `SONIC_DEPLOY_BIN` 会把 cwd 换到另一棵树;SIMPLE 自己那棵 `/workspace/deps/gear_sonic_deploy_lockstep` 只有 `policy/release`(我们的 `..._ours_lockstep` / `..._nvdef` 两棵有 v1_1)。2026-09-11 起 `locomani.stack.cli up` 与 `locomani.stack.cli restart-deploy` 在启动前显式检查并打印这棵树里有哪些 policy |

所以:**任何"找到就替换 / 进入 / 读取"的操作,失败路径必须响亮。**改完检查改没改,
`cd` 之后确认到了哪,读完确认读到了东西。这个仓库里凡是这类操作后面都该有一道断言
—— `build_sim_scene.py` 的 `_check_ego_camera` 就是模板:它验的是**编译后的模型**,
不是发射出来的文本,因为凡是读文本的检查都会跟着文本格式一起失效。

### 端口

`INSTANCE=i` 时全部加 `10*i`,DDS domain 也用 `i`;instance 0 总挂 exporter 落盘,其他实例要 `RECORD_INSTANCE=1`(session 在 `p03_i<i>/outputs`)。

> **端口分得开,DDS domain 分不开 —— 除非 deploy 打过补丁。**
> `INSTANCE` 只对 Python 侧和端口有效。**库存的 `g1_deploy_onnx_ref` 把 DDS domain
> 硬编码成 0**,根本不读 `SONIC_DDS_DOMAIN`(要读得先打
> [`patches/sonic_dds_domain.md`](../../patches/sonic_dds_domain.md) 再重编)。照
> [setup.md](../setup.md) 从零装出来的 deploy **没打这个补丁**,所以它不能参与任何
> 多实例场景。用没打补丁的二进制起第二个 stack,三方状态是:
>
> | 进程 | domain | 为什么 |
> |---|---|---|
> | 第二个 stack 的 sim(`INSTANCE=1`) | 1 | `run_sim_locomani.py` 认 `SONIC_DDS_DOMAIN` |
> | 第二个 stack 的 deploy | **0** | 硬编码,环境变量不读 |
> | 第一个 stack(sim + deploy) | 0 | — |
>
> 后果不是"两个 stack 互相干扰",而是**第二个 stack 的 deploy 静默接管第一个 stack
> 的 sim** —— 它和第一个 deploy 一起往同一个 sim 发 lowcmd,而它自己的 sim 一条指令
> 都收不到。排查时第一反应会是"第二个 stack 起坏了",而不是 DDS domain。

> **Python 侧的端口以前也分不开,现在由 `paths.sh` 统一派生。**
> 六个端口(相机 5555 / 命令 5556 / debug 5557 / 键盘 5580 / 特权 5581 /
> episode 5582)在 `agent_server.py`、`vla_round.py`、`privileged_pub.py` 里是
> `os.environ.get("TF_...", <instance-0 默认值>)`。而 `collect_pilot.sh` /
> `collect.sh` 从来不提 `INSTANCE` —— 于是即使 stack 用 `INSTANCE=1` 起来了,
> 它们启动的 `grasp_gym` 还是去连 5556,**连到 instance 0 的那个 stack 上**。
> 和 DDS domain 那个坑同一个形状,只是在 Python 侧:全都连上了,没有任何报错,
> 只是连错了 stack。现在这六个端口和 `SONIC_DDS_DOMAIN` 都由 `scripts/paths.sh`
> 一处派生,凡是 source 它的脚本(包括两个采集脚本)都自动一致。

> **`INSTANCE=1` 的 5590 和闭环评测的 `serve_policy` 撞端口。**
> `TF_KEYBOARD_PORT = 5580 + 10*INSTANCE`,所以 `INSTANCE=1` 占住 5590,而
> `serve_policy.py` 的默认端口正是 5590(`scripts/archive/run_round3.sh`、
> [evaluation.md](../evaluation.md) 都用它)。**多实例采集和闭环评测不能同时跑**,
> 或者给 `serve_policy` 换个端口。

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
SIM_VIS=none            # 默认(2026-09-10 起),录像完全干净
SIM_VIS=full            # 全开,调试用(2026-09-10 前的默认)
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
SIM_VIDEO_DIR="$PWD/outputs/sim_video" python -m locomani.stack.cli up --instance 0 --gpu 0 --prompt "pick up the water bottle"
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/locomani/tasks/bottle/grasp_gym.py --rounds 3
```

---

## 常用诊断脚本

都用 `.venv_sim` 的 python,在 stack 起来后运行;结论都写在各自的 docstring 里 —— **它们是这个仓库
大部分"为什么是这个数"的出处**。

**路径注意(2026-09-12 核对)**:老路径 `src/teleop_free/diag_*.py` 已不存在,而且下表里**大部分脚本已被
refactor step 2 删除**,只在 git 历史里(`78536ae:src/teleop_free/`)。今天真正在树上的是
`src/locomani/backends/sonic/diag_{lean,shuffle,wrist,vr3pt}.py`、`src/locomani/tasks/bottle/diag/diag_*.py`、
`src/locomani/rtc/diag_*.py`。**数字过期时的标准做法是把脚本恢复回来重测,而不是引用旧结论**
(`diag_wrist.py` 2026-09-11、`diag_lean.py` 2026-09-12 都是这么做的;两次都发现旧数字只对 `policy/release` 成立)。

| 脚本 | 回答什么 |
|---|---|
| `diag_channel.py` + `diag_channel_report.py` | planner 通道 vs streamed 通道的手臂投递精度(216 样本) |
| `diag_squatmode.py [--sweep]` | `IDEL_SQUAT` 的蹲深,以及 `height` → 骨盆高度的标定曲线 |
| `diag_vr3pt.py` | VR 3-point 腕/颈目标在 **encoder `teleop` 模式**下的交付(位置比、伺服残差、旋转、蹲下时手掌世界系漂移、颈点前倾→腰)。2026-09-12 实测 `outputs/diag/vr3pt_20260912/`:v1.1 reach 1.16 / 伺服残差 2.6 cm / 旋转 0.88 / 蹲 0.10 m 手掌只漂 2.3 cm / 颈前倾→腰 1.05;release 1.38 / 6.1 cm / 0.67 / 2.4 cm / 0.02。它顺带打印 deploy.log 里的 `[Mode Filter] Switched to mode 'teleop'` 一行 —— 没这一行的测量都是在 `g1` 模式下测的,腕字段是死的(ledger bottle.md 34 就是这样测出 0.01 的) |
| `diag_lean.py` | 命令的腰部前倾有多少被交付(**答案随 checkpoint 变**:`policy/release` 0.09、`policy/sonic_v1_1` 0.87,2026-09-12 实测 `outputs/diag/lean_20260912/`;老文档里"2–10%,基本不交付"只对 release 成立) |
| `diag_height.py` | `height` 字段在 IDLE / SLOW_WALK 下是否有效(答案:没有) |
| `diag_ego_geom.py` | 物体在 ego 相机里的可见距离区间,随躯干前倾的变化 |
| `diag_token_replay.py --phase capture/replay` | 录下的 token 能否复现动作 |
| `diag_shuffle.py` | 每个 speed 命令实际走多快、一步多长 |
| `diag_turn.py` | 原地转身:每个模式、每个角度要多久,以及控制器的朝向死区 |
| `diag_reach_probe.py` | reach map 的实测校验 |
| `diag_wrist_cam.py --dataset DS [--sweep]` | 腕相机挂载位置:从数据集里的真实抓取帧离线重建场景并**渲染**,报告物体离近平面多近、被裁掉多少。不需要 stack(它自己建模型) |
| `diag_ik_grasp.py` | 纯运动学抓取 bench(不经过 policy) |

数据集本身的检查不在这里,在 [`tools/check_dataset.py`](../dataset.md#转完先查一遍强烈建议) —— 它不需要 stack 起着,只读磁盘上已经转好的数据集。

**不需要 stack 的离线工具**(`src/teleop_free/tools/`,同样用 `.venv_sim` 的
python;`sonic_encoder` / `sonic_decoder` 需要 `SONIC_DEPLOY_ROOT` 指到
`third_party/GR00T-WholeBodyControl/gear_sonic_deploy`):

| 工具 | 回答什么 |
|---|---|
| `motion_smoothness.py A B --n 60 --events` | 两个数据集的运动质量逐项对比(jerk / 高频占比 / SPARC / cadence / token 台阶),外加"剧烈段来自哪里"的归因。结论见 [collection.md](../archive/collection_20260914.md#动作质量和人类-teleop-逐项量过一遍2026-08-28) |
| `sonic_encoder.py --selfcheck --dataset DS` | 离线跑 SONIC 的 motion encoder:任意 29 维参考动作 → 64 维 token,不用起 stack。`--selfcheck` 校验 1762 维的分块偏移、FSQ 码本、灵敏度与解码端一致性 |
| `sonic_decoder.py` | 反过来:token 差异 → 关节度数,给 open-loop 指标一个单位 |

---

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 双臂 pilot 每集只有 25 s、视频 10–12 s、日志里没有 `[walk]` 行 | 跑的是 ready-stand 变体。`WALL_WALK` / `WALL_FEATURE` 环境变量 2026-09-09 起不再被读(旋钮在 `CrateTaskConfig`,`wall_walk` 默认 false);给 `run_wall_pilot.sh` 传 `--config configs/tasks/crate_wall.json` 或 `--set wall_walk=true`。runner 现在见到这两个变量会拒绝启动(2026-09-10) |
| `walk_pose_preflight.py` / `reach_preflight.py` 对 `POSE_LEFT_DEG=…`、`PW=… OUT=…` 没反应,跑出来的是默认姿势 | 这两个预检 2026-09-09 argparse 化(refactor step 3)后不再读环境变量,旋钮是 `--pose-left-deg` / `--pw --out-dist --up --over --warm --save`;2026-09-11 起脚本见到旧变量会拒绝启动。`walk_pose_preflight` 还依赖 pilot trace `outputs/videos/wall_pilot/trace/wall_s0_05_state_trace.npz`,没有的机器用 `--trace none` |
| `locomani.stack.cli up` 报 "REFUSED: instance N is held by a live stack",但 `pgrep` 找不到任何栈,前一行有 `flock: Bad file descriptor` | 不是被占用,是锁文件没建成(2026-09-11 前锁目录写死 `/workspace/tmp`)。现在锁目录跟 `LOCOMANI_TMP`,建不了直接报目录名 |
| instance ≠ 0 的栈 "camera port never opened",sim.log 正常、子进程活着 | 2026-09-11 前 sim 的图像子进程永远绑 5555(`SimLoopConfig.camera_port` 没接 `TF_CAMERA_PORT`)。现在 `run_sim_locomani.py` 用 `ports.camera()`,sim.log 打 `[sim] camera publisher -> :PORT` |
| deploy.log 一直 "LowState is not available, waiting for robot to be ready",sim 在跑 | deploy 二进制没打 `patches/sonic_dds_domain.patch`,在 domain 0 听;`strings target/release/g1_deploy_onnx_ref \| grep -c SONIC_DDS_DOMAIN` 为 0 就是。打补丁 + `cmake --build` 重编(`docs/setup.md` 第 3 步) |
| "STACK IS NOT USABLE: the sim has not published a single image",但 `ss -ltnp` 里相机端口在、sim.log 几分钟后才出现 "Image publish frequency" | 子进程 stdout 进文件是 8 KB 块缓冲,不是没发。stack manager 2026-09-11 起 `export PYTHONUNBUFFERED=1`;老版本用 `SKIP_IMAGE_CHECK=1` 绕过 |
| `build_isaac_assets.sh` 打 "build OK",但 `scene/<label>/*/rgb_0000.png` 三张预览都是纯灰;Isaac 视频里机器人浮在灰底上、没有房间和桌子 | `build.log` 里有 `MISSING surface A: dining_table`:这台机器没有布局生成流水线的产物 `outputs/scene/{layout,surface_b}/asset3d/*` 和 `layout/background/mesh_model.ply`,`isaac_scene.py` 跳过它们继续建。从采集机拷这几个目录。2026-09-11 起脚本把 MISSING 行打到终端并 WARN |
| `render_wall_pilot.sh` 每集 `isaac exit 1`、只有 `tracking_mujoco.mp4` 没有 `ego_isaac.mp4`;`rNN/isaac_stdout.log` 只有一行 "no head_camera in …xml" | 相机明明在 xml 里:`frames.head_camera_spec` 的正则写死了属性顺序,MuJoCo 写出的顺序不同。2026-09-11 起按属性名逐个取,不再依赖顺序 |
| 所有 stack 结果同时变差:走位失控、"ready posture not delivered"、复位后摔倒,日志里没有一句提到 CPU | 容器 cgroup CPU 配额只有 15.36 核(`/sys/fs/cgroup/cpu.max`),`nproc` 的 32 和 load average 都是宿主机的数。别的任务把配额占满时 sim 实时因子掉到 0.18、gym 循环 9 Hz(2026-09-10,`/workspace/code/video_grasp` 的 benchmark)。起 stack 前 `bash scripts/cpu_headroom.sh`,余量 < 6 核或有 throttle 就别开;pose sidecar 的 `t` 对 `wall` 跨度可以事后算实时因子 |
| 双臂并排视频里没有 ghost hands | `trace/` 里没有 `<label>_NN_ghost.npz`。2026-09-11 之前:runner 只在调用者自己 export 了 `SIM_STATE_TRACE_DIR` 时才传 `--ghost-trace`,而 runner 自己不设,所以默认其实是关的(`render_wall_pilot.sh` 的注释写成"默认传",是错的)。现在 driver 默认写 `<out>/trace`,`--no-state-trace` 才关。渲染脚本只在文件存在时才画 |
| 双臂 pilot 的 10 集验收数字对不上、或 `gate.py compare crate` 的样本数多于跑过的局数 | 2026-09-11 之前每次 pilot 都写同一个 `outputs/videos/wall_pilot/manifest.jsonl`,而 expert 是以 **append** 打开它的:两次 pilot 混成一份,`gate.py` 把它当一个总体读。现在一个 run 一个目录(`OUT=`,不给就按时间戳新建),同目录跑第二次会被拒(续跑用 `--resume`),manifest 是 `manifest_NN.jsonl` |
| 直接调 `bimanual_wall.py` / `bimanual_carry.py`,它却连到了另一个 instance 的栈 | expert 用 `os.environ.setdefault` 取端口,所以**继承来的** `TF_*` 会压过 `locomani.sim.ports` 的公式。任何 source 过 `scripts/paths.sh` 而当时 `INSTANCE` 未设的 shell,都导出了 instance 0 的整块端口。新的采集阶段起 expert 前会把这一块从 env 里删掉,只留 `INSTANCE`;手工起 expert 时要么先 `unset` 它们,要么在 export `INSTANCE` 之后再 source `paths.sh`(2026-09-11) |
| （**2026-09-10 已修**）手写 `grasp_gym.py --task-config configs/tasks/xxx.json` 跑出来的局和采集线对不上 | 修复前 `--task-config` **只读 `goal` 和 `place` 两段**(它自己的 help 就这么写)。`reach` 段不生效 → reach config 退回 dataclass 默认值,`ready_grasp_recovery=False`,每局在 grasp 阶段被 `ready_grasp_recovery=False is retired` 直接拒绝(走近那段其实是成功的,很容易误读成控制器坏了);`spawn` 段也不生效 → 吃 grasp_gym 的宽默认(±90° 朝向偏差),机器人侧对桌子醒来,整批卡在 `[view] object not in ego view at spawn`,而且 cam **y** 出界(采集线正常的偶发失败是 cam **x** 出界)。现在 `--task-config` 携带整份 profile(reach + spawn + target + stand_retries),命令行显式给的仍然优先;`grasp_gym` 另外会在启动时拒绝跑不出结果的 reach 标定(`ready_grasp_recovery=False`),不再等走到瓶子跟前才失败。`scripts/collect_pilot.sh` 从重构 step 2 起到这次修复之前一直采不出任何一集,同因(2026-09-10,连踩三次) |
| （**2026-09-10 已修**：`--record` 现在启动时检查 exporter，没有就拒绝）`--record` 跑完一批,`$LOCOMANI_TMP/p03*/outputs` 下没有新 session | 用了 `INSTANCE≠0`。stack manager 只在 instance 0 起 exporter,别的实例打印 `[3/3] data exporter — skipped (instance N records no episodes)` 就跳过了,gym 照跑但什么都不落盘。采集只能用 instance 0;要避开别人换 **GPU** 而不是换 instance(`CUDA_VISIBLE_DEVICES`)(2026-09-10) |
| （**2026-09-10 已修**：端口统一由 `locomani.sim.ports` 按 `INSTANCE` 推导）换了 `INSTANCE` 之后 gym 立刻 `no privileged state on :5581 after 10s` | 修复前 gym 不读 `INSTANCE`。stack 按 instance 挪到 5565/5566/5591,gym 仍用 instance-0 默认端口绑 :5556、等 :5581。`INSTANCE` 只在 stack manager 自己的 shell 里展开成 `TF_*`;要让 gym 跟过去,得把 `TF_SONIC_PORT` / `TF_PRIVILEGED_PORT` / `TF_KEYBOARD_PORT` 一起 export(2026-09-10) |
| 后台 runner 判断"机器可用"后起 stack,却和别人撞了端口 | GPU 空闲 ≠ instance 空闲:另一个 session 可能正用同一 instance 的 5555-5582 在**另一张卡**上工作。runner 的准入检查必须同时看 (1) 目标 GPU 显存 (2) `cpu_headroom.sh` (3) 有没有活着的 sim/exporter/deploy 进程。2026-09-10 的 `visible200_train_eval_20260910.sh` 只做了 (1)(2),当天靠人工约定时间窗兜住;该脚本**不补丁**——`scripts/experiments/<name>.sh` 要与实际执行的那份一致,修复放进下一个 runner |
| 进程存活检查 `pgrep -f <name>` 命中一堆无关 shell | `pgrep -f` 匹配整条命令行,任何命令行里出现过该字符串的 `zsh -c` / `bash -c` 都会被算作"正在运行",连两天前的僵尸 shell 也算。要求 `comm` 真的是 python:`ps -eo comm,args --no-headers \| awk '$1 ~ /^python/ && /run_sim_locomani[.]py/'`。同一类坑还有 `pkill -f` 打到自己(见台账)(2026-09-10) |
| 每批(每次 `start_stack`)的第 1 局固定在出生阶段失败:`could not turn N deg ... timed out` 或 `object not in ego view at spawn`,第 2 局起正常 | deploy 的朝向锚在 stack 起来后的名义标定处已被捕获(yaw 0),第 1 局却按新 deploy 处理、不 reanchor。2026-09-10 修(`grasp_gym.start_control_once`,台账 bottle §14);再出现说明标定处又绕过了它 |
| `locomani.stack.cli up` 卡在 `[2/3]` 十分钟 | 首次运行在编译 TensorRT engine,正常。之后走缓存,20–40 s |
| 没打印 `stack up` | 去 `$LOCOMANI_TMP/p03/{sim,deploy,exporter}.log` 看是哪一个没起来 |
| planner 消息里给了 `vr_position` 腕目标,手掌却只动命令量的 ~1%(两个 checkpoint 都一样) | **字段的"有无"是 encoder 模式开关,不是 `stream_mode`**:deploy 在 `has_vr_3point_control_` 的**上升沿**把 encoder 切到 `teleop`(mode 1,腕目标才是伺服输入),下降沿切回 `g1`(mode 0,腕槽位清零),`zmq_manager.hpp:679-693`;而 `Planner()` 初始化和每次 safety reset 都把模式重置回 0(`cpp:3770`、`zmq_manager.hpp:498`)。2026-09-12 之前 `AgentCommander.frame()` 从开机起**每一帧**都带该字段,上升沿在 planner 初始化之前就烧掉了,之后再没有沿 —— deploy 一直停在 `g1`。现在字段只在 `cmd.vr_targets=True` 时发(`TeleopArm.start()` 在 planner 起来、机器人站稳、第一帧目标=实测位姿之后置位,`stop()`/`reset_pose()` 清),与 PICO manager 的时序一致。判据:deploy.log 里 `VR 3-point control enabled` 必须出现在 `Planner initialized` **之后**,并紧跟 `[Mode Filter] Switched to mode 'teleop'` |
| deploy 起来了但机器人不动 | 检查 :5556 上是不是有**多个** commander 在发。多个进程往同一个 deploy 发冲突意图,表现就是发呆或抽搐 |
| 每局都 `teleport did not land` / 第 3 局起 `walked N m off the spawn while standing up` | 上一局的 deploy 朝向锚没清干净。2026-09-10 起复位阶段用 `reanchor(p)` 清(不重启);若还出现,`--set restart_deploy=true` 回到每局重启对照 |
| 没有第三人称录像 | `SIM_VIDEO_DIR` 必须在 `locomani.stack.cli up` **之前**给,跑起来之后设无效 |
| 录像里没有绿色点云 | 默认就没有(`SIM_VIS=none`);要它就给 `SIM_VIS=full` 或含 `reach` 的列表。它在 sim 启动时读一次,跑起来之后改不了 |
| `SIM_VIS=none` 但幽灵夹爪还在 | `SIM_VIS` 只给了 stack,没给 gym。两个进程都要设 |
| 录下来的 `planner_height` 全是 -1 | `patches/vla_action_space.patch` 没打。整段下蹲会从 action 里消失 |
| action 是 267 维而不是 78 维 | 同上,补丁没打 |
| `[manifest] ...: N produced rounds but M episodes on disk` | manifest 和 session 不是同一批。一批写一个 manifest、填一个 session,别混用 |
| `[manifest] ... join is out of phase` | 更糟:数量对上了但成败对不上。**别转这个 session**,指令会标错 |
| `check_dataset.py` 报 `action: MISSING ['left_hand_joints', ...]` | modality 的 action key 被改过名。GR00T 只 warn 不报错,照训但两只手会被静默丢掉 —— 必须修完再训 |
| `check_dataset.py` 报 `lag distribution {3: ...}` 之类的非零 | 视频和表差了那么多帧。先查 exporter 的时间戳,再查转换时的 50→30 Hz 重采样 |
| `check_dataset.py` 报 `body trails the token by -N frames` | action 和 state 在时间上被搞反了,是最严重的一类,**不要拿去训** |
| `EPISODE FAILED [view] object outside ego view at stand` 每局都这样 | ego 相机没被重新瞄准,停在原厂的下视 + fovy 45。重建一次场景:`build_sim_scene.py` 现在会在编译后自检瞄准、不对就拒绝写盘。历史成因是它曾用 `str.replace` 改 `to_xml()` 的输出,而 MuJoCo 3.12 改了属性发射顺序,replace 静默失配 —— 用 3.12 装出来的旧版本会全局 0% 成功率 |
| `[gym] targets not in the scene` | 场景不是你以为的那套。报错会列出场景里**实际**有什么 —— 按那份清单对照下面[「场景装成了哪一套」](#场景装成了哪一套)判断是哪种病 |
| 采完之后场景尺寸不对了 | `collect.sh` 就地重建场景。`SCENE_PRESET=<你那套> bash scripts/install_scene.sh` 恢复 —— 漏掉 `SCENE_PRESET` 会顺带把桌上的物体换成默认那套 |
| 换了机器路径全错 | `export LOCOMANI_WORKSPACE=<你的根>`,然后 `python src/locomani/paths.py` 自查 |
| 场景 MJCF 找不到 mesh | 重跑 `SCENE_PRESET=<你那套> bash scripts/install_scene.sh`;它会探测并改写 tarball 里烘死的路径 |
| `ValueError: ... Error opening file ... scene_31dof.xml` | 新 clone 的 `third_party` 里没有 dex1 生成文件。先跑 `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python src/locomani/sim/make_dex1_model.py`,再 `SCENE_PRESET=<你那套> bash scripts/install_scene.sh` |
| `ModuleNotFoundError: No module named 'skfmm'` | `.venv_sim` 里没装 `scikit-fmm`(上游安装脚本不带,这个仓库自己的依赖)。`uv pip install --python third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scikit-fmm` |

## 场景装成了哪一套

`[gym] targets not in the scene` 的报错尾巴会列出场景里实际有的物体。拿它对照:

| 报错列出的物体 | 场景是 | 怎么办 |
|---|---|---|
| 只有 `water_bottle` | `SCENE_PRESET=bottle_only`,正确 | 报错另有原因:检查 `--targets` 拼写 |
| 8 个:`bottle, can, cup, fork, jar, napkin, orange, plate` | 默认多物体桌面 | `SCENE_PRESET=bottle_only bash scripts/install_scene.sh` |
| 其它任何组合 | 解包出来的素材不完整或过时 | 重新解包:删掉 `outputs/scene`,再跑 `SCENE_PRESET=... bash scripts/install_scene.sh` |

**那 8 个物体分不出两种病。**tarball 里烘焙的现成场景(`scene/sim/`)和一次默认重建
产出的**完全一样** —— 都是 `target_object=can`、`target_pool=[can, bottle, cup, jar]`、
那 8 个物体。所以「重建了但没给 SCENE_PRESET」和「压根没重建」在物体清单上不可区分。

这也是为什么 `install_scene.sh` 跳过重建这件事至今没人踩到:今天两者恰好重合。
但 tarball 是很久以前打的,`scene_lib.py` 一改两者就分家 —— 那是个潜伏的坑,不是
不存在的坑。真正的判据是重建到底跑没跑,所以脚本现在两件事都做:重建成功时回显
preset / target / target_pool / 桌上物体,跳过重建时响亮报错并以非零退出。

## 共享机器上的 CPU 抢占(2026-08-28,与两个同容器 session 联合定位)

这台机器 32 核,uid 1002 下有 ~90 个进程,**其他 session 与我们是同一个用户**。
闭环评测对时序极其敏感:CPU 饥饿会静默改变结果(实测同一份代码 ~38% → ~17%),
比崩溃危险,因为它照样产出一个看起来正常的数字。两次事故都是别的 session 的
**playwright-mcp 拉起的 headless chrome**,分别吃掉 9.3 核和 8.7 核、持续数小时到一天。

**根因,反直觉的那一层**(由 hand-rl session 逐进程量出):headless chrome **没有 GPU**,
WebGL/Canvas 因此落到 **SwiftShader 软件光栅化**,而它是**多线程**的 —— 所以不是烧
1 个核,是能吃满 8 个。叠加页面里无节制的 `requestAnimationFrame` + 自动播放
(标签页开着就永远满速重绘),就是我们看到的量级。

推论:**"headless" 不等于 "轻量"**。对纯 DOM 页面成立,对 WebGL 页面恰恰相反 ——
没有 GPU 反而把渲染压力全丢给 CPU。而"越需要用浏览器验收的东西(3D/WebGL),
代价越高",因为纯 DOM 看快照就够、3D 必须真渲染。

**一眼分类的判据**(按 `--type` 看,不是看进程数):

```bash
for p in $(pgrep -f "/opt/google/chrome/chrome"); do
  tr '\0' ' ' < /proc/$p/cmdline | grep -o -- "--type=[a-z-]*"
done | sort | uniq -c
```

高占用集中在 `--type=gpu-process` → SwiftShader 软光栅,页面有持续重绘的
WebGL/Canvas;集中在 `--type=renderer` → JS/DOM 的问题。**两类修法不同。**

**这台机器上已确立的规矩**(两个 session 独立得出同一结论):

- **只处置烧核的 chrome,不碰常驻的 MCP server**。实测有 10 个
  `node playwright-mcp --headless` 常驻(最老 2 天 4 小时,合计 612 MB,CPU 全 0%)
  —— 从外面分不清"泄漏"与"待命",误杀会让别人的工具凭空失效,而它们本身不构成压力。
  危害在 chrome,不在 server。
- 处置顺序:**先 `renice +19`,只有降权后仍抢不过来才 kill**(同 uid 可行;
  `cap_sys_nice` 被剥夺,我们**无法**提高自身优先级,cgroup 只读)。降权最坏让对方
  慢几秒,杀掉可能让对方白跑一次。
- **动手前必须重读进程归属**。归属会在决策窗口内变化 —— 本次就差点拿几分钟前的
  快照去杀一棵已经易主的进程树,是对方 session 提醒才没误伤。这是本表的老主题
  (检查器报告的必须是世界**当前**的状态)在多 session 环境下的新形态。
