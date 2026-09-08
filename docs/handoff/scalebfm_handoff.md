# ScaleBFM 分支交接:控制器替换、v3 单手 expert、准备用它采集训练数据

> 2026-09-08,分支会话(scalebfm spike)写给主线会话。这里是全部事实、文件、数字和
> 待办;主线按此在自己那边实现采集 + 训练。分支这边不再改主线文件。

## 一句话

ScaleBFM(上海 AI Lab,arXiv 2607.15163,MIT)在 sim 里可以替代 SONIC 做运动控制:
mode 2 只给 **骨盆 + 双腕** 的 6D 目标,腿、腰、肘由模型 inpaint。已验证:走路停位
1–2 cm、yaw 0.3°;自发弯腰到 35° 去够远处/低处目标;双臂搬箱 9/10 与 10/10(0.45 矮桌);
单手抓瓶 expert(v3)8/10;单进程、无墙钟、逐帧可复现,10 集 90 秒墙钟。

## 资产与代码(全部已在工作树,未 commit)

| 路径 | 作用 |
|---|---|
| `/workspace/models/scalebfm/humanoid_transformer_m_22200.pt` | 原始 checkpoint(HF `WeishuaiZeng/ScaleBFM`,73 MB) |
| `assets/scalebfm/model_22200_tensorrt_metadata.json` | 关节顺序(IsaacLab)、默认姿态、action_scale、PD 增益、history 3、future_idx [0..5]、14 个 selected body |
| `assets/scalebfm/mode_table.pt` | 8 种 control mode 的 mask 表;mode 2 = pelvis + left/right_wrist_yaw_link |
| `assets/scalebfm/g1_29dof_fk.xml` | 模型内部 FK 用的 MJCF(必须是 ScaleTrack 那份;ScaleBridge 那份有无关节 body,解析会断言失败) |
| `src/teleop_free/scalebfm/bfm_policy.py` | vendor 的网络 + 部署 wrapper + `load_policy()`,eager torch,5090 上 6.5 ms/步,不需要 TensorRT |
| `src/teleop_free/scalebfm/run_spike.py` | `Sim`(我们的场景 + 它的 PD 200 Hz/decimation 4 + 我们的夹爪 PD)、`Policy`(9 个 tensor 的 obs 拼装、history 缓冲)、`Refs`(骨盆/手参考 → 未来 6 帧)、双臂 pilot-23 复现 |
| `src/teleop_free/scalebfm/run_bend.py` | 双臂远站位 + 涌现弯腰版(9/10、矮桌 10/10) |
| `src/teleop_free/scalebfm/bottle_probe.py` | 单手 reach 精度/可见性探针,量出前馈偏置 |
| **`src/teleop_free/scalebfm/run_bottle.py`** | **v3 单手抓瓶 expert,推荐采集用** |
| `src/teleop_free/scalebfm/run_bottle_v4.py` | 无时钟尝试,未完成,不要用(原因见末尾) |
| `src/teleop_free/scalebfm/sonic_envelope.py`、`render_*.py`、`scripts/run_sonic_envelope.sh` | SONIC 对照 envelope,与本交接无关 |
| `docs/handoff/sonic_lockstep_collection_plan.md` | SONIC lockstep 化方案,已搁置 |

结果目录:`outputs/videos/scalebfm_spike`(双臂 10/10)、`scalebfm_bend`、`scalebfm_bend_t045`、
`scalebfm_bottle_v3`(单手 8/10,每支视频左第三人称右 ego)、`outputs/reach/scalebfm_bottle`(探针)。

## 策略接口(每个 50 Hz 步)

输入 9 个 tensor(batch 1,H=3 帧历史,F=6 未来帧,N=14 body):
`root_quat[1,3,4] wxyz`、`base_ang_vel[1,3,3]`(体坐标)、`dof_pos[1,3,29]`、`dof_vel[1,3,29]`
(**IsaacLab 顺序**,metadata `joint_names`)、`last_action[1,3,29]`(它自己上一步输出)、
`target_pos[1,6,14,3]`、`target_quat[1,6,14,4]`(**骨盆系**下 14 个 body 的目标,mode 2 只有
pelvis/左腕/右腕有效,其余填 0 会被 mask)、`mode_index[1]=2`、`time_offsets[1,6,1]=0..5`。
输出:29 个 PD 目标(IsaacLab 顺序)+ 回填历史的 raw action。PD:`stiffness/damping` 来自
metadata,力矩限幅 `torque_limit`,MuJoCo 步长 0.005 × 4。夹爪它不管,我们自己 PD
(`Sim.step`,kp 40、kd 1.0,力限 `grip_clip`,瓶子用 0.6 = 150 N)。

世界系 → 骨盆系:`pos_b = R(root_quat)^T (p_w − root_pos)`,`quat_b = quat_inv(root_quat) ⊗ q_w`
(`bfm_policy.quat_apply_inverse / quat_mul_inverse_left`)。手的目标是 **wrist_yaw_link** 位姿;
dex1 pad 中心在腕系 x 方向 0.1515 m(`run_spike.OFF`),`wrist_from_pad(pad, R) = pad − R·OFF`。

**已测得的策略特性(设计 expert 时必须考虑)**
- 对静态偏差响应弱:骨盆/手离目标 ~10 cm 以内基本不动;参考必须带速度(未来 6 帧在前进)
  且领先身体。按时间推进的参考天然满足,纯状态驱动的"胡萝卜"会卡在不动点(v4 的教训)。
- 手开环误差:单手水平抓取时短 6–11 cm、低 3–4 cm、掌心低头 13°(`bottle_probe` open.json)。
  用一个全局前馈偏置 + 13° 抬头 + 比例过驱动(增益 1.0、上限 5 cm)后,0.45–0.55 m 站距、
  瓶子在右肩前时抓点误差 0.6–1.7 cm。右手在双臂场景固定内偏 4–8 cm。
- 双臂抬起时骨盆自己后退 5–10 cm、下沉 3–8 cm;单手也有,站点预留 6 cm。
- 头相机(pitch 52°)在前倾 35° 时看地面,站距 ≥ 0.60 抓取瞬间丢瓶子;腕相机全程 0.85–0.96。
- 瓶子场景 `outputs/scene/sim` 没有箱子场景那套 pad 接触硬化,运行时补(`run_bottle.py main`:
  priority 1、condim 4、solref 0.002、solimp 0.995/0.9999),否则瓶子在爪里转 8–18°。
- 夹爪 PD 的目标必须一直是 GRIP_OPEN 或明确的合爪指令,不能跟随测量值(会自己慢慢合上)。

## v3 单手 expert 的动作逻辑(`run_bottle.py`,采集就用它)

原则:每帧命令是"瓶子位姿 + 机器人可见状态"的函数;相变由可见线索触发;无积分器;无 hover 停预;
无 squat 命令;无骨盆 servo;无重新接近;身体前倾是涌现的。

| 段 | 参考 | 终止线索 |
|---|---|---|
| 出生/settle 1.5 s | 骨盆钉在当前位,手垂体侧 | 固定(策略填历史) |
| 走路 | 骨盆参考沿直线梯形速度 0.5 m/s、加速度 0.6;站点 = 瓶子 − (d − 0.06)·接近方向 + 0.15 m 向左,d ~ U(0.45, 0.55);朝向平滑转到面向瓶子 | 骨盆参考剩余 0.5 m 时伸手开始 |
| 伸手(与走路重叠) | 右手 pad 从当前位置 min-jerk 到预抓点(瓶心 −2 cm 高、后退 0.10),时长 = 剩余走路时间 + 1 s(≥ 2.5 s);掌向 rot_z(yaw) 再抬头 13°;命令 = 目标 + 偏置(10.0, −3.0, 2.8 cm) + 1.0×(目标 − 实测 pad),上限 5 cm | 曲线走完 |
| 进刀 | 直线 2 s 到抓点,抓点用瓶子**实时**位姿 | pad 在抓点 along 2 cm、across/高 1.5 cm 内 → 合爪(超时 4 s 记 timeout) |
| 合爪 | 0.6 s smoothstep 到 −0.023,保持 0.3 s | 固定 |
| 抬起 | +0.10 m,1.5 s | 固定 |
| carry | 沿接近方向反向 0.15 m,高度不变,2 s + 保持 1 s | 结束 |

成功判据:结束时瓶子比起始高 ≥ 8 cm 且倾斜 ≤ 30°;走路/伸手阶段瓶子被碰即失败。
每集记录:stand_d_meas、stand_err、yaw_err、close_trigger、entry_err、rise、tilt、torso 最大前倾、
分阶段 ego/wrist 可见比例、state trace(`trace/wall_s0_NN_state_trace.npz`,qpos 50 Hz,与
pilot 格式一致,`tools/summarize_wall_pilot.py` 可直接读)。

v3 结果(`scalebfm_bottle_v3`,seed 1):8/10;停位比目标远 4–7 cm;合爪 1.0–1.3 s 触发;抬起
9 cm;失败一集瓶子在爪里转 37°,一集 carry 时摔倒(未查)。静止帧比例 0.08。
**注意**:v3 之后改了夹爪 PD(kd 0.5 → 1.0)和夹爪保持逻辑,v3 没用新参数重跑过,采集前先重跑 10 集。

## 准备训练:action 表示与数据集

- **action(每帧)**:mode 2 的关键点目标,骨盆系:pelvis(pos 3 + rot6d 6)+ 右腕(3 + 6)
  = 18 维,加夹爪 1 维。双臂再加左腕 9 维。位置单位米,rot6d 取旋转矩阵前两列。
  **记录发给策略的那个命令**(含偏置和比例项),不是几何目标;因果关系才成立。
- **chunk 与未来帧**:ScaleBFM 本来要未来 6 帧(0.1 s),VLA 的 action chunk 直接切片喂进去;
  部署时每 tick 用 chunk 里当前偏移起的 6 帧,不够就用最后一帧重复。
- **观测**:ego_view + right_wrist(与现在 modality 一致),state 用现有 53 维(腕相机是抓取
  阶段的主线索,ego 在抓取瞬间允许短暂丢目标)。`observation.<obj>_pose` 继续记录不声明。
- **converter**:`tools/convert_to_hf_format.py` 的 exporter 输入不存在了(没有 SONIC stack、没有
  exporter);数据直接由 expert 进程写:每 tick 的图像(tick 边界同步渲染)、state、action、
  夹爪。建议先写成和现有 LeRobot v2.1 相同的 parquet/mp4 布局,`check_dataset.py` 四段审计
  要过;modality.json 的 action 组改成上述 19 维,`configs/gr00t_sonic_2cam.py` 对应改。
- **decoder/serve_policy**:`serve_policy` 输出的 action 不再进 SONIC 的 token 通道,而是转成
  `Refs` 的未来帧格式在进程内调 `Policy`;评测循环就是 `run_bottle.py` 的 tick 换成 VLA 输出。
  可复用 `vla_round.py` 的 K/A 帧延迟建模(lockstep 逻辑),但不需要任何 ZMQ。
- **第一个实验(用户定的)**:不加多样性、不加 place,只验证 overfitting:v3 固定条件采
  ~200 集成功集(只留通过判据的),训一个与 47.5% baseline 同条件的模型,看闭环。
  基线成功率 SONIC 时代约 50%,先看能不能超过。
- **旧数据可能不作废**:mode 2 的命令 = 未来几帧的骨盆/腕位姿,旧 episode 里都记录了,
  可以近似重标(丢 servo 修正项),作为对照实验的起点。

## 已知未决 / 不要踩的坑

1. `run_bottle_v4.py`(无时钟、纯状态驱动)六轮 0/3,原因:状态驱动参考有不动点 +
   策略对静态偏差响应弱;若以后要做 DAgger 需要"任意状态可标注"的 expert,做法是保留 v3
   曲线形状与带速度的未来帧,只把参数 u 换成手在路径上的实测投影,并显式加 lead(~8 cm)
   和速度下限(~0.05 m/s)。当前阶段不需要。
2. 单手时 `sim.grip_clip` 用 0.6;夹持力不是瓶子倾斜的原因,pad 接触参数才是。
3. 不要用 ScaleBridge 的 `g1_29dof.xml` 给 FK;用 `assets/scalebfm/g1_29dof_fk.xml`。
4. GPU:分支一直用 `CUDA_VISIBLE_DEVICES=1`,`MUJOCO_GL=egl`;venv 用 `$SONIC_ROOT/.venv_sim`
   (torch 2.13 cu130 + mujoco 3.11)。
5. 场景:单手用 `outputs/scene/sim`(瓶子在 (−0.16, 0.385, 0.703),桌高 0.62,近桌沿
   34 cm);`run_spike.SCENE / OBJ` 是模块级变量,import 后立刻覆盖再构造 `Sim`。
6. 对照数字:SONIC 同样输入的 envelope(`outputs/videos/envelope_scalebfm_vs_sonic*.mp4`)
   84 格里 ≤ 10 cm 的:ScaleBFM 49、SONIC 站立 6、SONIC 下蹲 8。
7. 论文对 SONIC 的对比:global mode 成功率 97.8 vs 59.4;local mode 持平。真机路线未验证。
