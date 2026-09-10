# 交接：SONIC 动态回放与部署一致性审计

更新时间：2026-09-10 14:2x UTC（session 29 续写；12:03 之前的内容由 Codex 写）。工作目录 `/workspace/code/locomanipulation`。
本文区分事实、决策、假设；数值对应已保存实验，不把准备工作算成完成实验。

## 1. 当前目标与授权边界

- 当前研究任务：用户要求“开始对比官方动态动作回放，以及模型、PD 增益和仿真配置的一致性”，解释本地 SONIC 弯腰跟踪弱、协作者真机 G1 可以弯腰的差异。
- 当前这轮指令仅要求保存交接；未新启动或重跑实验。
- GPU1 / instance1 验证已获授权，常规诊断修正不需再次询问。不要占用 instance0/GPU0 或干扰其他会话任务。
- 旧 ScaleBFM Goal 已被用户明确取消。不要恢复 Goal、ABCD、训练或早期任务；按当前明确任务推进。
- 长期背景：视觉+本体状态驱动、利于 VLA 学习的人类式抓取，VLA 不读取物体真值；专家先验缺口曾要求暂记不修。诊断用 privileged 真值只作测量，不能据此给专家增加新特权控制。
- 读 `docs/STATE.md` 资源规则及 `docs/ledger/bottle.md` §12–15；STATE 混有不同会话旧的 RUNNING 条目，应以时间戳、进程和产物核实，不能盲信。
- 禁止删除/覆盖 outputs。历史 outputs 曾因 symlink merge 丢失，详情见 STATE incident。不要重跑已经完成的 phase1/phase2 或200集采集。
- 多个 Claude 同时修改本工作区；`git status` 不是本会话独占改动清单。不要回退、清理、提交他人的改动；未要求 commit。不要主动创建子代理。

## 1b. 2026-09-10 下午：checkpoint A/B 与随后的代码修复

**结论：v1.1 不能直接换进采集线。** 20 局固定 case bank，release **9/20** vs sonic_v1_1 **0/20**，
9 个不一致 case 全部 release 赢，精确 McNemar p=0.0039。机制是到位几何不是抓取：
v1_1 停在 0.498 m（release 0.377，plan_r 0.32），横向偏 −0.324 vs −0.214（t=−6.38），
approach 有 10/20 次 `stalled`（release 2/20）。要换 checkpoint 必须连带重标 approach 时序。
产物 `outputs/experiments/v11_bottle_ab20_20260910/`，台账 `docs/ledger/bottle.md` §17。

注意 §16 的参考回放结论（v1_1 腰 RMSE 3.9° vs release 12.9°）与此不矛盾：
v1.1 确实更会弯腰，但它走不到位。两条结论要一起读。

**已落地的四条修复**（`make check` 通过，未提交）：`locomani.sim.ports` 统一端口公式；
`reach_config.require_runnable()` 与 `ports.require_exporter()` 两个启动期拒绝；
`--task-config` 携带整份 profile；修好 `scripts/collect_pilot.sh`（自重构 step 2 起一直采不出数据）。
细节与动机见 ledger §17.3。**待办：主代码改动按规范要附 `scripts/gate.py compare` 输出，尚未跑。**

## 2. 最新运行状态（2026-09-10 12:40 UTC 核对）

**事实：phase3 已经跑出有效动态回放，并完成了 release / sonic_v1_1 单因素对照。GPU1 已交还。**

- `retry_reset/`（Codex 那轮）仍是 CPU gate 超时的记录，run.exit=1，未启动仿真，原样保留。
- CPU 争用已解除：Playwright Chrome 属于 `toporetarget-reimp-25` 会话，我通过 SendMessage 说明后对方自行关闭，
  未擅自终止对方进程。空闲从 −0.05 核回到 7.6–10 核。
- 新产物：`outputs/diagnostics/sonic_native_phase3_20260910/retry_reset_02/`（policy/release，3/3）与
  `.../v1_1/`（policy/sonic_v1_1，3/3）。两者 run.exit=0。
- 12:33 UTC pick-place 会话认领 GPU1（instance 0）。我的可选录像对照在 CPU gate 内被中止，**从未创建 stack**；
  `video_release/` 只是残留目录，不是结果。GPU1 当前 2 MiB，无我方进程。
- 结论摘要与全部数字见 `docs/ledger/bottle.md` §16、`docs/STATE.md`。

## 3. 已确认的决策及理由

1. 先用 C++ 原生 CSV loader 回放官方本地动态参考，绕开 Python `pose_chunk` 路径。理由：phase2 静态测试的关节速度始终为0，不能代表动态运动能力。
2. 实验使用私有可执行文件和诊断脚本，不修改生产模型、PD 或专家控制。逐项改变变量，保持可归因性。
3. phase3 第一条记录属于无效初始化，不计动态回放失败。模型加载时仿真缺乏有效控制，随后 start 后还等4秒；play开始时已经倒地。
4. 重试修正为模型 `Init Done` 后 reset、读取新 wall 时间戳状态、检查初始骨盆高度，再立即发送 `]t`。这仍是待验证诊断协议，不宣称修好了 SONIC。
5. 资源门槛：连续3个5秒窗口空闲>=6核且无 cgroup throttle，最多等10分钟；未满足则不创建 stack。避免CPU争抢使实时控制结果失真。
6. 官方深蹲参考 root 最低约0.332m，因此本诊断不以任务标准root<0.4m判定失败；脚本<0.15m保护中止也不是完整成功定义。
7. 一致性只说证据覆盖到的层面：实际LowCmd匹配当前C++不等于匹配训练/真机；同名身体参数一致不等于整机一致。

## 4. 已完成实验与结论

### Phase1：孤立腰部参考，已完成

产物 `outputs/diagnostics/sonic_lean_20260910/live/phase1_summary.json`，详见 ledger §13。
10正式case（neutral/reach各5次），每case planner/streamed ×6角度，共120条件，无记录到的跌倒，batch.exit=0。
请求29.8°时均值：

|条件|实际腰pitch°|实际躯干pitch°|电机腰目标°|
|---|---:|---:|---:|
|neutral planner|0.04|1.95|0.95|
|neutral streamed|0.68|6.33|1.11|
|reach planner|1.82|3.46|-9.57|
|reach streamed|2.30|6.31|-11.11|

事实：本地输出不忠实跟随孤立腰参考；不能由此断言G1/SONIC不能弯腰。
其他审计：ready-grasp `use_waist=False`；通用IK pitch上限0.20rad，物理约0.52rad；旧direct诊断绕过IK上限，因此该上限不能解释全部弱跟踪。

### Phase2：协调全身静态参考与实际encoder输入，已完成

目录 `outputs/diagnostics/sonic_lean_phase2_20260910/`；关键文件 `summary.json`、`encoder_report.json`、`encoder_root_check.json`、`plan.json`。
15/15 case（5条件×3次），run.exit=0，无记录跌倒，instance1关闭。
参考 `squat_001__A359` 第85帧，ramp3s/hold4s/return3s；这不是动态回放。

|条件|请求躯干°|实际躯干°|实际腰°|电机腰目标°|
|---|---:|---:|---:|---:|
|legacy_waist_052|29.79|5.35|0.27|-4.06|
|waist_only|25.17|4.90|0.32|-9.03|
|legs_waist|25.17|7.38|1.43|-6.03|
|legs_waist_root|34.97|13.06|1.16|-11.12|
|full_pose|34.97|16.52|7.14|-46.92|

实际 encoder 输入捕获：23365完整记录，1762输入float32/64token，mode0。稳态关节参考最大误差5.89e-8rad；重建rootpitch与请求相差<=0.05°；dq块全0。
文件尾7072字节不完整记录忽略，15个hold区间都有覆盖。
**结论：信号确实到达encoder；协调参考将实际弯腰4.9°提高到16.5°，但仍低于约35°。尚未找到最终根因。**
局限：同一deploy在reset间保留历史，不是严格配对RNG；私有binary为当前源码重编译，非原生产binary逐字节相同，legacy弱跟踪对照已复现。

### Phase3：动态回放已完成，checkpoint 对照已完成

目录 `outputs/diagnostics/sonic_native_phase3_20260910/`。

让回放有效的启动协议：等 `Init Done` → reset → `]` 接管 → **1 秒沉降** → 才 `t`。沉降窗口内骨盆 0.890→0.786 m、
ncon 0→33（reset 是瞬移，采样在物理步进前，机器人还悬空）。旧的 `]t` 连发就是在悬空态开播，这才是那条 0.119 m
记录的成因。两次运行各 3 遍全部完整播完：**423/424 帧、正好 50.00 fps、8.46 s、无跌倒**。

3 次均值，取参考最深前倾帧（参考躯干 40.3°、腰 29.3°）：

|checkpoint|躯干 pitch|腰 pitch|腰 RMSE|躯干 RMSE|
|---|---:|---:|---:|---:|
|policy/release|14.5°|8.1°|12.9°|12.4°|
|policy/sonic_v1_1|25.3°|21.4°|**3.9°**|**5.7°**|

唯一变量是 checkpoint（含其自带 observation 配置）。重复性：release 腰 8.3/8.2/7.9，v1_1 腰 21.4/21.5/21.2。
**动态并不能救静态弱跟踪**——release 的动态结果与 phase2 静态同量级，handoff 原假设 1 判否。
encoder 的 10 个参考槽位实测为 `k, k+5, …, k+45`，确实包含未来 0.9 s。

已用数字排除的环节：参考→encoder 2.1e-7 rad；参考与官方 MJCF 自洽（FK 躯干 pitch 复现 body_quat 到 0.01°，
所以"腰 pitch 正 = 前倾"在参考/官方/本仓一致）；deploy 观测 vs privileged 真值 ≤1.2°/2.3°；
`last_action` vs sim `cmd_q` ≤0.05 rad；两份配置都通过 deploy 自身维度校验（1762/994 与 1751/994）。

仍未解释、且**两个 checkpoint 共有**：`waist_pitch` 是唯一与自身参考反相关的指令通道（6 次运行全为 −0.90 左右，
参考要 +29.3° 时电机目标约 −60°；腿/臂 +0.95…0.99）。单通道符号翻转排除映射错位。腰 kp=28.5（armature×ω²），
实测该关节重力保持力矩 12–23 Nm，所以实际弯腰不是这条指令驱动的。

`sim_parity.json` / `parity_report.json` 结论不变：29 身体关节参数一致；Dex1 35.864952kg vs 官方 Dex3 36.1652352kg；
实际 LowCmd 的腰 pitch kp≈28.5012 / kd≈1.81445（不是 YAML 里的 400/250）。

配置疑点：`policy/release/observation_config.yaml` 是第三份手改配置，既不等于 `observation_config_sonic_release.yaml`
也不等于 `_sonic_v1_1.yaml`（多 `motion_root_z_position*` 共 11 维，正好对上实测 1762 vs 官方两份写的 1751；
anchor 用非 heading 归一化版本）。`policy/sonic_v1_1/` 是自洽的模型+配置对。

### 已完成的ScaleBFM工作（背景，勿重复）

`outputs/experiments/scalebfm200_noretract_sonic_20260910/RESULTS.json`：200成功并audit通过、49387帧、400视频，LeRobot v2.1转换和stats完成；该结果记录training_started=false，本会话没有启动其训练。
数据路径：同目录 `dataset/`。34集OSMesa+166集EGL，来源历史保留；中断seed4035跳过。
用户取消向身体收回约7cm：carry_back=0。成功标准对齐 `sonic_lift6cm_hold2s_v1`（抬升>6cm持续>2s，任务fall root<.4）；24s预算和采集质量audit保留。
吞吐：旧OSMesa中位约176s/集；EGL12集pilot 12/12、24视频解码通过、中位5.497s，后续约5–6s/集。
固定规则已写 `docs/scalebfm_collection.md`：正式批采GPU EGL、明确CUDA/EGL设备、先pilot及视频/吞吐验收；软件渲染须显式 `--allow-software-rendering`。
历史 make check、17项成功判据测试通过；这些不是phase3的新增测试。
其他会话的 `visible200` 是另一份SONIC数据，不要与本200集混淆。

## 5. 文件及实现关键点

本阶段新增、尚未提交的脚本：
- `scripts/experiments/sonic_native_replay_20260910.py`：原生PTY控制，3次目标，每次12s；记录state/debug/躯干FK；拒绝覆盖同名case。
- `scripts/experiments/sonic_native_replay_20260910.sh`：GPU1/instance1资源gate、启动和清理。gate超时不会stop未由其启动的stack。
- `scripts/experiments/sonic_native_parity_20260910.py`：CPU只读MJCF质量、选项、模型哈希、LowCmd和常数对比。
- 本阶段更新 `docs/STATE.md`、`docs/ledger/bottle.md` §15；本次创建本HANDOFF。

之前本会话相关脚本（具体变更读文件，勿重建）：
`sonic_lean_audit_20260910.py`、`sonic_lean_live_20260910.py`、`sonic_lean_batch_20260910.sh`、
`sonic_lean_phase2_20260910.py/.sh`、`sonic_encoder_capture_build_20260910.py`、
`sonic_encoder_capture_report_20260910.py`、`scalebfm200_20260910.py`，均在scripts/experiments。
之前ScaleBFM相关生产修改涉及 `configs/tasks/bottle_visual.json`、`tasks/bottle_visual/{config.py,success.py,round.py,pipeline/collect.py,pipeline/eval.py}`、`tests/test_visual_success.py`及collection/scalebfm文档；与其他会话修改交叠，不能按本表整体回退。
其他git status中的runtime、scene、refactor、assets/simple等修改不能归为本会话所有。

私有诊断binary：
`outputs/diagnostics/sonic_lean_phase2_20260910/deploy/target/release/g1_deploy_onnx_ref`。
运行cwd为该目录的 `deploy/`；policy/reference等symlink到生产目录。
生产树 `third_party/GR00T-WholeBodyControl/gear_sonic_deploy`；私有encoder_capture.cpp在实际Encode后捕获同一输入buffer和token。
单TU编译，缓存StateLogger ABI不符时私有重编译state_logger.o并优先链接；build日志/manifest/命令均保留，不用全量重编译。

Encoder捕获格式：little-endian `<IdIIi` = magic0x53454e43、double wall、uint32 ni、uint32 nt、int32 mode，接ni+nt个float32。
输入4:294为10×29关节参考（ISAACLAB顺序），294:584为速度，601:661为10×6 root参考；6D转rotation后需结合base quat重建世界方向。
`last_action`已经是MuJoCo顺序、缩放加default后的电机q目标，不能再重排/缩放；`body_q_target`在encoder上半身覆盖前，不代表实际encoder输入。

原生参考：`third_party/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/squat_001__A359/`。
424帧50Hz；CSV有header；joint_pos/vel为ISAACLAB顺序（旧模块注释可能误写MJ），用g1_joints.isaac_to_mujoco。body_quat wxyz，body0 pelvis/body7 torso。
参考首帧root约[-.00254,.01266,.79256]，quat[.71218,-.03302,-.02698,-.70070]（yaw约-89°），腰pitch .12222rad；默认reset不等于该首帧。
Reset API支持robot dxy/yaw/dof(MJ)/dof_from，暂不支持完整root roll/pitch/z；seat_on_floor会调整root高度，不能未经检查视为精确首帧reset。
原生input-type=zmq，不发送Pythonpose包；`]`start、`t`play、`r`restart、`i`heading reinitialize，换行切stream/native（不要误切）。当前无planner。

## 6. 未解决问题（按当前证据重排）

1. **已判否**：动态速度/未来帧能改善静态弱跟踪。有效动态回放与静态同量级。
2. **已判否（作为主因）**：启动/初姿对齐。修正协议后 3/3 有效、无跌倒、播放速率精确 50 fps，弱跟踪照旧。
3. **已有强证据**：checkpoint 是主导因素（腰 RMSE 12.9° → 3.9°）。尚未确认协作者真机跑的是哪个 checkpoint —— 这是
   现在最该问用户/协作者的一件事。
4. **未解释**：`waist_pitch` 指令通道与参考反相关（两个 checkpoint 共有）。可能是训练时 robot 资产与参考数据集的
   腰 pitch 符号约定不一致，也可能是这条通道的动作本就被设计成负载补偿项。判别需要训练侧配置或 IsaacLab 资产，
   目前仓库里没有。**不要**据此改生产代码或给腰加特权控制。
5. **未知**：训练 PD、协作者真机 checkpoint 与参考接口。仍不能宣称与真机一致。
6. **限制**：私有 binary 与官方发布 binary 的构建差异；静态 legacy 弱跟踪已复现，但不是二进制相同性证明。

## 7. 下一步与关键命令

1. 先问清协作者真机用的是哪个 checkpoint 与哪份 observation 配置。若是 sonic_v1_1，本地"不能弯腰"基本就解释完了。
2. 若要继续本地：把 `policy/sonic_v1_1` 接到生产链路是一个独立决定（会改变所有既有 deployment verdict，见
   memory「Deployment verdicts are checkpoint-conditional」），需用户拍板，不要顺手换。
3. 想要可视化证据：录像对照（`SONIC_NATIVE_VIDEO=1`）尚未跑成，等 GPU1 空出再跑，换新目录名。
4. 想追 waist_pitch 反号：需要训练侧的 G1 资产/关节约定或 IsaacLab 配置；仓库内已无更多可查证据。

```bash
cd /workspace/code/locomanipulation
export PYTHONPATH=$PWD/src
bash scripts/cpu_headroom.sh 3
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
# 重跑（必须换目录，脚本拒绝覆盖已有 case）
SONIC_NATIVE_OUT=$PWD/outputs/diagnostics/sonic_native_phase3_20260910/<新目录> \
  SONIC_NATIVE_POLICY=policy/sonic_v1_1 bash scripts/experiments/sonic_native_replay_20260910.sh
# 离线分析，不占 GPU
SONIC_NATIVE_OUT=<目录> third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/sonic_native_report_20260910.py
SONIC_NATIVE_OUT=<目录> third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/sonic_native_channel_audit_20260910.py
```

GPU1隔离环境（launcher已设置）：INSTANCE=1、SONIC_DDS_DOMAIN=1、CUDA_VISIBLE_DEVICES=1、MUJOCO_EGL_DEVICE_ID=1；TF_CAMERA_PORT=5565、TF_SONIC_PORT=5566、TF_OUT_PORT=5567、TF_KEYBOARD_PORT=5590、TF_PRIVILEGED_PORT=5591、TF_EPISODE_PORT=5592；OMP/OPENBLAS/MKL线程均1。EGL设备编号不能想当然跟随CUDA屏蔽重排。
动态库环境 `source scripts/env_deploy.sh`；仿真Python为 `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python`。
`bash scripts/stack/stop_instance.sh 1` 仅在确认是自己的instance1时使用；它会强制结束实例。
后台启动请用Python subprocess.Popen(..., start_new_session=True, stdin=DEVNULL, stdout=log, stderr=STDOUT)并保存PID，简单nohup &在本工具shell中曾未存活。

## 8. 验证与建议技能

- phase3新增Python脚本编译检查、shell bash -n通过；parity脚本已实际运行生成JSON。重试控制协议未通过动作验证，因为资源gate超时。
- 建议沿用 `/home/yijie/.agents/skills/diagnosing-bugs/SKILL.md`：先获得对应症状的有效回放闭环，再单因素测假设；无效初始化不是模型失败证据。
- 本交接采用handoff技能组织；用户明确要求项目根HANDOFF.md，优先于技能“临时目录”默认位置。
