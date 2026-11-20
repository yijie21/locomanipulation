# selfmodel_via_20261120：空地上的连续多路点伸手数据，和预测整条手臂实际轨迹的自身模型

- 日期：2026-11-20（方案；尚未启动）。同日在 vastgpu 上做了查新，按用户的决定改了两处：窄缝测试加"学出来的标量半径"这个对手，判据分开看方向和大小。改动都在第一次启动之前。
- 机器 / GPU：4090 机器（180 核，4 × RTX 4090；用 GPU 1–3，GPU 0 常有他人任务）。试跑也可以在 vastgpu 上做（只用 GPU 1，不超过 6 个进程）：那里已经建好同一个空地场景，冒烟测试 8 次开场单进程用了 17 s。
- 代码版本：分支 sonic-selfmodel；启动前需先提交 motor.py 的 via_rot（每个路点带手腕朝向）和 simple_bench.py 的 speed，数据对应确定的 commit
- 配置 / EXP：gen_reach.py 新开关 REACH_VIA（随机多路点）、REACH_VIA_REPLAY（回放真实 cuRobo 路线）；其余同 simple_selfmodel_wrist_20261119 的 graspr_wrist_odo（REACH_GRASP=1 REACH_GRASP_ROOM=1 REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 REACH_GRASP_IKREF=1 REACH_WRIST=1 BFM_ROOT=odo REACH_HAND_OFF=0.0415,0,0 REACH_SCENE_DIR=outputs/scene/simple_g1_sonic）
- 关联：curobo_avoid_20261120（cuRobo 路线执行时手偏离规划：direct 全程中位数 4.1 cm、最大 11.7 cm，28 局；复核后只看规划段、到最后一个路点为止，direct 和 model 都是 3.82 cm）；curobo_attrib_20261120（B46）；beliefs 里自身模型只描述身体、空地采数据
- 查新（2026-11-20，记录在主仓库根目录的 `step1.md` 到 `step7.md`）："障碍物交给几何，身体的执行偏差用学出来的模型预测"这个分工已经有人做过。
  - 最接近的是 Dynamic Tube MPC（arXiv 2411.15350，ICRA 2025）：在并行仿真里学跟踪误差随规划轨迹怎么变，再让误差范围留在空地里。它的误差是一个标量，范围是球，机器人是单腿跳跃机。同一类还有 Deep Learning Tubes for Tube MPC（RSS 2020）、闭环 RRT、Kim 等 RSS 2022。
  - 人形上最近的是 SWAP（arXiv 2610.25711）：G1，冻结的底层控制器，随机命令滚出实际扫过的空间，事后在没扫到的地方摆障碍，训练看点云出动作的规划器。它做的是走路避障，障碍到动作的映射是学出来的。
  - 留给我们的差别：预测整条手臂偏差的方向和形状，再把路点挪过去。加厚只会让空地变小，挪路点才可能穿过比原始偏差还窄的缝。这是假设，下面的判据就是为检验它写的。
- 和 curobo_avoid 判据的关系（2026-11-20 交接时补）：curobo_avoid 按预注册判据（被挡住 ≤ 4 局）把"路径版自身模型"搁置了，它的复核 E3（激活距离 5 cm 能否消除碰桌）也没跑。这条重新打开路径版自身模型，理由不同：用户要的是通用绕障，不是 TabletopGrasp；窄缝里统一安全距离不够用（见"后续"）。那次的偏差是在碰桌的情况下测的（有路线的 28 局里 27 局越过桌沿时碰桌约 1 s）；接触处规划余量中位数约 5.6 cm、手点偏差约 3.4 cm，接触原因未找到（候选：没记录的手腕姿态误差，或碰的是手指、手臂而不是手点），所以模型要预测整条手臂，不只手点。

## 背景和思路

避障拆成两半：障碍物交给几何（cuRobo 直接用点云查碰撞，无需训练，对任何形状都一样）；身体交给自身模型（给控制器一串路点命令，整条手臂实际怎么走）。只有身体这一半需要数据覆盖，而身体的命令空间是有界、低维的。

现有数据核对（2026-11-20）：
- 每段伸手按 10 Hz 存全身状态（29 关节角和角速度、骨盆世界位置和朝向、重力方向、两手位置），从下命令前 0.3 s 到结束后 1 s。
- 用关节角 + 骨盆位姿做正运动学重建手的位置（simple_g1_sonic 场景，Dex3 手掌偏移 0.0415 m），和记录对比：graspr_wrist_odo 200 段 9222 帧误差中位数 0.003 cm、95% 0.107 cm；approach_wrist_odo 200 段 21089 帧 0.039 / 0.175 cm。整条手臂的轨迹可直接算出。脚本 `scripts/experiments/selfmodel_via_20261120/fk_check.py`（4090 上 `loco_outputs/fk_check.py`）。
- 缺的：连续经过多个路点的命令（现有每段一个目标、参考路线为直线；抓取数据的三段是走完停约 1.3 s 再走下一段）。cuRobo 避障执行正是连续多路点。

现有数据里手偏离参考路线的最大距离（`scripts/experiments/selfmodel_via_20261120/dev_map.py`，graspr_wrist_odo 28336 次伸手）：

| 动作 | 中位数 | 90% |
|---|---|---|
| 站着伸手 | 3.32 cm | 6.39 cm |
| 蹲低（骨盆 0.52–0.70 m） | 7.14 cm | 12.25 cm |
| 目标高 0.7–1.0 m | 3.15 cm | 6.64 cm |
| 目标低于 0.4 m | 7.36 cm | 12.41 cm |
| 蹲低、目标低于 0.4 m、伸手超过 40 cm | 9.66 cm | 13.01 cm |

真实 cuRobo 路线（35 条：桌面抓取 28 + XMovePick 7，`scripts/experiments/selfmodel_via_20261120/via_stats.py`，伸手起点的骨盆朝向坐标系）：中间路点 7（1–10）；直线距离 39 cm（24–45）；路线长 79 cm（25–91）；最远偏离直线 26 cm（2–31），桌面抓取为往右外侧 22 cm、往上 14 cm；手腕沿途累计转动 139°（42–181）；路线不高过目标点。

## 方案

### 每次开场（reset）
同 graspr_wrist_odo：随机走一小段到站位，抓取目标按 room_commands 抽（高度 0.10–1.12 m 等，手腕转动同 REACH_WRIST），伺服关，10 Hz 记录。
- 第 0 段（新）：伸向预抓取点，沿随机多路点路线，`m.reach(side, pre, rot, via=..., via_rot=..., speed=..., servo=False)`，与 simple_bench --avoid 的下发方式相同。
- 第 1 段：慢速进入，同现有数据。
- 新存字段：路点位置和朝向（补齐到 16 个）、路点数、速度、生成参数、参考 IK 位姿序列（motor 每 3 cm 一个的全身关节，用作"手臂照参考走"的基线）。

### 随机路线生成（起点 = 手的实际位置，终点 = 预抓取点）
| 参数 | 分布 |
|---|---|
| 弯折点数 | 0：15%（直线，对照老数据），1：35%，2：30%，3：20% |
| 弯折点位置 | 沿路线 15%–85% 随机 |
| 偏离方向 | 垂直于直线的整圈均匀随机 |
| 偏离大小 | 3–35 cm |
| 先后撤或先上抬 | 10%：一半开头先退 2–10 cm，一半先抬 2–15 cm |
| 手腕 | 从起始朝向平滑转到目标朝向，弯折点处再随机多转 0–90°（试跑时是 0–60°，覆盖检查后放宽：真实路线手腕累计转 89–209°） |
| 目标 | 沿用 room_commands；站立高度（0.60 m 以上）的目标最近到身前 25 cm（试跑时最近 32 cm，覆盖检查后放宽：真实目标在 27–38 cm） |
| 速度 | 一半 0.15 m/s（默认），一半 0.08–0.25 m/s |
| 左右手 | 各一半（与现有数据一致；可改为只采右手） |

平滑曲线（向心 Catmull-Rom）连起点、弯折点、终点，1 cm 采样后用 Douglas–Peucker 1 cm 压成路点（同 via2.py）。路点超过 16 个时把容差每次放大一半，直到放得下。

一条路线出现下面任何一种情况就重抽，最多 30 次，再不行就走直线，每次开场记下抽了几次：
- 离地不到 5 cm；
- 进了躯干：骨盆竖直轴周围 16 cm 以内、高于 0.25 m。路线最开始的 6 cm 不算，手本来就在那里；
- 蹲下时进了大腿和膝盖所在的区域；
- 越过身体中线超过 25 cm；
- 出了命令抽样用的大盒子；
- 比路线两端中离肩更远的那一端还远 10 cm 以上。

代码在 `scripts/experiments/selfmodel_via_20261120/via_gen.py`，采集开关是 `gen_reach.py` 的 `REACH_VIA` 和 `REACH_VIA_REPLAY`。偏离方向的角度从"朝外"量起，朝外指离开身体中线：0° 朝外，90° 朝上，180° 朝里，270° 朝下。

### 覆盖检查（启动前定好）
35 条真实路线的特征（偏离大小和方向、路点数、长度比、手腕转动、终点，标准化后）到最近生成路线的距离，须不大于生成路线之间最近邻距离的中位数。不满足：生成器漏了一类形状，先补再采。脚本 `via_cover.py`。

起点只打印，不计入距离（2026-11-20 改，启动前）：回放和随机路线一样，都从手当时所在的位置出发；随机路线的起点彼此相差不到 1 cm，真实路线的起点只要差几厘米，标准化后就会被算成很远。

### 测试集
1. 普通留出：按开场编号分桶（同 data.py）。
2. 整片形状留出：偏离方向落在一个 45° 扇区（左下）的路线不参与训练。
3. 真实路线回放：35 条 cuRobo 路线在空地上执行（REACH_VIA_REPLAY），回答上次 4.1 cm 偏离有多少来自身体、多少来自碰桌子。
   - 每条跑 4 遍：第 1 遍从原地站立出发，后 3 遍先走一小段再出发，站位各不相同。同一条路线 4 遍的偏差方向是否一致，是"方向能不能学"的第一个检验，不需要训练模型。

### 标签和模型
- 离线正运动学算右臂和手的碰撞球（SIMPLE 的 cuRobo g1_29dof_with_dex3.yml 的球）每帧位置。
- 模型：输入同现有自身模型（起始状态 + 命令）加路点序列，输出各球在归一化时刻上相对参考的偏移；5 个成员的集成（同 E3_ensemble），成员分歧作为不确定度。
- 主指标"扫过区域误差"：每次伸手，实际各球位置到预测扫过路线的最大距离（静止障碍只关心扫过哪里，不关心时间）。基线：手臂照参考 IK 位姿走。
- 碰撞球共 24 个：右臂 9 个、手掌 3 个、手指 12 个。左手的伸手用左臂的连杆配同一组球的镜像，所以两只手的标签排列相同。已核对左右连杆坐标系互为镜像（零位下差 0.01 mm）。脚本 `arm_spheres.py`。

### 方向和大小分开看（2026-11-20 查新后加，启动前）

每次伸手取偏差最大的那一刻，把偏差写成一个向量：实际位置减去参考路线上最近的点。它的长度叫大小，它的指向叫方向。脚本 `via_dev.py`。

- **对手"标量半径"**：只预测大小，也就是 Dynamic Tube MPC 的做法。用同一份数据、同样的输入训练。它不能减小偏差，只能给出一个要留出的半径。
- **我们的"方向模型"**：预测整个偏差向量，执行前把路点反向挪过去。
- **比较的量**：两个模型各自需要的余量。把预测整体乘一个系数，在校准集上调到 95 % 的伸手不超出，再在测试集上取这个余量的中位数。
  - R_标量：标量半径模型给的半径；
  - R_方向：减去方向模型预测的偏差向量之后，剩下的偏差要留的半径。
- **方向稳定度**（不用训练，试跑就能算）：同一条真实路线的 4 遍回放，两两之间偏差方向的余弦，取平均。方向每次都一样是 1，方向是噪声时平均为 0。手和肘各算一个。
  - 试跑后改了算法（见"预测修订"）：不再只比偏差最大的那一刻，改成比整条偏差曲线。脚本 `via_profile.py`。
- **对手"常数修正"**（试跑后加，用户 2026-11-20 同意）：所有路线都减同一条平均偏差曲线，这条曲线在训练集上算，不看具体路线。试跑里手在终点总是停在命令之前约 3 cm，这类偏差一个常数就能修掉。
  - R_常数：减去这条平均曲线之后，剩下的偏差要留的半径，算法和上面两个相同。

### 规模和时间（4090 全力，144 个采集进程）
依据：graspr_wrist_odo 12000 次开场的 85%（10200 次）在 4090 上用了 14.4 分钟（08:18:43–08:33:05，2026-11-19）；自身模型训练两个种子 8.2 分钟（08:45:56–08:54:06）。

| 步骤 | 机器时间 |
|---|---|
| 试跑 300 次开场 + 35 条回放 | 约 3 分钟 |
| 正式采集 12000 次开场 | 约 17–25 分钟 |
| 正运动学标签 | 约 5 分钟 |
| 训练（集成，两个种子并行） | 约 10–20 分钟 |
| 评测（留出、扇区、回放、覆盖检查） | 约 5 分钟 |

学习曲线到 100% 仍明显下降则加采，每 12000 次约 20 分钟。

## 假设
H1：控制器执行连续多路点命令时，手臂偏离参考的相当一部分是系统性的、可由起始状态和命令预测；在空地数据上训练的全臂轨迹自身模型，能把扫过区域误差显著降到"手臂照参考走"的基线以下，并且在没见过的路线形状（扇区留出）和真实 cuRobo 路线（回放）上也成立。

## 预测（启动前写，首次启动后锁定）
- 试跑：摔倒 < 1%（现有 0.2%）；全身 IK 失败（ik_err_cm > 2）< 5%。
- 基线（手偏离参考路线的最大距离）：随机路线中位数 3–8 cm；真实路线回放中位数 2–6 cm。理由：现有直线伸手中位数 5.0 cm（站着 3.3 cm），真实路线是站着、目标高约 0.9–1.0 m，但路线长一倍。
- 方向稳定度（2026-11-20 加）：35 条真实路线上，手的稳定度中位数 0.6–0.9。理由：现有中途偏摆模型横向 R² 0.53，说明偏差有相当一部分是系统性的；同一条路线从相近的站位出发，应当比不同命令之间更一致。
- 余量之比 R_方向 / R_标量（2026-11-20 加）：普通留出 0.5–0.8。理由：下一条预测模型把误差中位数降 40–65 %，95 % 分位上通常降得少一些。
- 余量之比 R_方向 / R_常数（试跑后、训练任何模型之前加）：普通留出 0.5–0.8，真实回放 0.7–0.9。理由：试跑的真实回放里，用同一条路线其余几遍的平均来预测剩 1.56 cm，只用别的路线的平均剩 2.39 cm，比值 0.65；模型不知道其余几遍，在这些路线上很难比 0.65 更好。随机路线的姿态和形状更多样，一条平均曲线会更不够用。
- 模型（手的扫过区域误差中位数）：普通留出 1.5–3 cm（比基线降 40–65%）；扇区留出 2–4 cm；真实回放 1.5–3.5 cm。理由：现有终点模型在抓取场景把误差从 2.75 降到 1.91 cm（约 30%）；中途偏摆模型横向 R² 0.53、上下 0.25，可预测部分有限；全程模型输入更完整，预期好一些但不确定，区间放宽。

## 判据（什么结果会让我放弃或修改假设）
- 真实回放的基线偏离中位数 < 1.5 cm：身体在空地上很听话，上次 4.1 cm 主要来自碰桌子，自身模型在避障里作用小 → 停止这条线，向用户报告。
- 试跑时真实回放的方向稳定度（手，35 条路线的中位数；2026-11-20 加）：
  - ≥ 0.6：方向稳定，继续正式采集；
  - ≤ 0.3：方向主要是噪声，能学的只有大小，这条线就退化成在人形上重做 Dynamic Tube MPC → 停下，向用户报告；
  - 两者之间：把手和肘的稳定度都报给用户，由用户决定是否采集。
- 模型相对基线降幅 < 20%（普通留出）：偏差主要不可预测 → 停止这条线，向用户报告。原来写的退路"按集成分歧给安全距离"就是学出来的标量半径，查新后它是对手，不是我们的方法。
- 余量之比 R_方向 / R_标量（普通留出和真实回放都要满足；2026-11-20 加）：
  - ≤ 0.6：预测方向比只预测大小少留 40 % 以上的余量，进入窄缝测试；
  - > 0.8：方向带来的好处很小 → 不再主张"预测方向"，向用户报告；
  - 两者之间：报告给用户决定。
- 余量之比 R_方向 / R_常数（普通留出和真实回放都要满足；试跑后、训练任何模型之前加）：
  - ≤ 0.8：知道具体路线确实有用，成立；
  - > 0.9：方向模型学到的基本只是一个固定偏移 → 不再主张"按路线预测方向"，向用户报告；
  - 两者之间：报告给用户决定。
  - 这两个阈值是看过试跑的回放数字之后定的，约束力弱于启动前定的那些。
- 降幅 ≥ 40% 且扇区留出误差 ≤ 普通留出的 1.5 倍：进入窄缝测试和演示。
- 普通留出好、扇区留出 > 普通留出的 2 倍：覆盖不够，"通用"不成立 → 扩生成器再采。

## 后续：窄缝测试和演示（本条目之后，另开条目）

窄缝测试：缝宽由宽到窄扫一遍，比较 cuRobo 小安全距离（3 cm）、cuRobo 大安全距离（如 12 cm）、cuRobo 加学出来的标量半径（每次伸手按预测的半径加厚障碍，2026-11-20 查新后加，这是最强的对手）、cuRobo 加常数修正（所有路线按同一条平均偏差反向挪路点，试跑后加）、cuRobo 关节轨迹直接作控制器参考（不从手路点重解 IK）、cuRobo + 自身模型；看成功率随缝宽的曲线，转折点由空地数据事先预测。

演示 case：蹲下，从塞满瓶子的矮柜底层，穿过前排瓶子间的窄缝，取后排的一罐饮料。
- 选在蹲低、低目标、远伸手：上表里偏离是站着够桌面高度的 2–3 倍，cuRobo 设想与实际差得最远；也是人形特有（固定底座机械臂不蹲、不因平衡晃），回应"为何不用 MπNets 类机械臂方法"。
- 瓶子：又高又细、一碰就倒且连锁，失败一眼可见；成功时纹丝不动。真机用空塑料瓶。
- 缝宽：手宽加几厘米，介于 cuRobo 小安全距离和实际偏离之间。
- 画面：三联并排（小安全距离扫倒一排 / 大安全距离停住"找不到路线" / 本方法穿过拿到）；半透明手臂叠加（蓝 = cuRobo 以为的手臂，绿 = 自身模型预测，实体 = 实际）；收尾缝越推越窄，显示碰倒瓶子数，窄缝处慢镜头。
- 备选：纸杯金字塔旁取物（可与瓶子结合）；从鸟笼栏杆间取物（碰到只会卡住，不够醒目）。
- 前提：偏差里有相当部分可预测（见判据）；缝宽和位置事先按空地数据定，所有试次都展示（如 10 次里 9 次对 0 次），配缝宽曲线。先仿真，真机等仿真结果。

## 预测修订

- 2026-11-20，试跑之后：**事先定的"方向稳定度"算错了对象，改用整条偏差曲线。**
  - 事先的算法取每遍偏差最大的那一刻的向量来比。这些路线上手有两处差不多大的偏差：一处在路线前段，一处在终点（手停在命令之前约 3 cm），方向几乎相反。哪一处更大，每遍不一样，所以两遍几乎相同的回放也可能得到接近 −1 的余弦。
  - 证据：同一条路线各遍之间，最大偏差落在路线上的位置相差中位数 0.53 条路线长；独立复核用自己的代码得到同样的解释。
  - 改成：把手的偏差按在参考路线上的位置分成 20 段，得到一条 20 × 3 的曲线；同一条路线各遍曲线之间的余弦，以及用其余几遍的平均去预测这一遍后剩下多少。脚本 `via_profile.py`。原来的统计量仍由 `via_dev.py` 照原样打印，不能读成"方向是噪声"。
  - 这是看到结果之后改的，证据强度低于事先定好的判据。要不要按新算法继续，由用户决定。
- 2026-11-20，用户看过试跑报告后的三个决定（原话："1. 继续 2. 可以 3. 可以"）：按改后的算法继续；先补生成器再正式采集；判据里加"常数修正"这个对手。
- 2026-11-20，补生成器（第一次）：站立高度的目标最近放到身前 25 cm；弯折处手腕多转的上限从 60° 提到 90°。理由都来自覆盖检查的输出：真实目标有一部分在随机目标的范围之外，真实路线的手腕转动有一部分高于随机路线的 95 % 分位。
  - 用改后的代码试了 300 次开场（提交前的探索，输出在 `data_via_trial/cover_after_widening_explore.txt`）：随机目标在该高度的 5 % 分位从 32.7 cm 降到 29.0 cm，手腕转动的 95 % 分位从 186° 升到 259°；但覆盖检查通过的反而从 14 条变成 9 条，没通过的都在 1.01–1.31 倍之间。
- 2026-11-20，**覆盖检查这条规则本身有问题，等用户决定怎么改。**
  - 规则要求每条真实路线到最近随机路线的距离，不超过随机路线彼此之间最近距离的中位数。按这个定义，随机路线自己也只有一半能通过。它要求的其实是"真实路线落在随机路线较密的那一半里"。
  - 这个比值和采多少数据无关。离线按同样的抽样画路线来估（不跑仿真，脚本 `data_via_trial/cover_sim.py`，起点取自试跑数据）：300 条时通过 5 / 35，3000 条 6 / 35，12000 条 7 / 35。
  - 真实路线的每个特征都在随机路线的取值范围之内（35 / 35）。它们同时占了几个特征的尾部：路点多（7–10 个）、绕得远（长度是直线的 2 倍以上）、手腕转得多、目标又近又高。随机路线里同时满足"绕得远、目标近、目标高"的约占 5 %。
  - 想让 35 条都通过中位数规则，就得让生成器的大部分路线长得像这 35 条（其中 28 条来自同一张桌子），这和"生成器不针对测试路线"的初衷相反。离线试了两种加大绕行的设置（偏离量 3–40 cm 或 10–40 cm，弯数向 1–3 个倾斜），12000 条时仍是 7 / 35。
  - 一个可行的替代：要求真实路线不比 95 % 的随机路线更孤立，也就是它不是随机路线分布的离群点。离线估计 12000 条时，现在的生成器 32 / 35，偏离量 10–40 cm 的设置 35 / 35。

（第一次启动之后的修订写在这里。下面这条是启动前的说明。）

- 2026-11-20，启动前：写方向稳定度和余量之比的预测与判据时，已经看过一次冒烟测试的输出（8 条随机路线、3 条用随机路线伪造的回放各 2 遍），目的是确认脚本能跑。冒烟数据没有真实路线，不用来下任何结论，也不计入试跑。

## 结果（贴原始输出，不转述）

试跑，2026-11-20 11:53–12:01 UTC，vastgpu，提交 `e5ba2b3e`，3 个进程（nice 10，GPU 1；机器上同时有别的 session 的任务）。数据在 `outputs/reachwm_20261030/data_via_trial`（300 次开场，种子 31000000）和 `data_via_replay`（35 条真实路线各 4 遍，种子 33000000），每个目录有 `run_info.txt`。真实路线来自 4090，经 Mac 拷到 `/workspace/tmp/curobo_routes_20261120`，md5 记在 `data_via_replay/routes_md5.txt`。没有跳过的开场。

随机路线（`gen_reach.py merge`、`arm_spheres.py`、`via_dev.py` 的输出）：

```text
600 segments, 37145 steps, fell 0, walked 600, ok 44
600 segments, 37145 ticks, 9730 reference poses, 24 spheres -> /workspace/code/locomanipulation-selfwm/outputs/reachwm_20261030/data_via_trial/arm.npz
rebuilt jaw centre against the recorded one: median 0.025 cm, 95 % 0.136, max 0.686
last reference pose's jaw against the command: median 0.648 cm, 95 % 3.344 (the plan's own IK miss: median 0.010, 95 % 3.202)
paths 300, fell 0, analysed 300; plan's IK miss > 2 cm: 34; skipped resets 0
largest offset from the reference path (hand: jaw centre; arm: any of the 24 spheres):
  all                            n   300   hand median  6.96 p90 11.88   arm median 11.23 p90 17.44 cm
  standing                       n   139   hand median  3.95 p90  8.33   arm median  6.53 p90 13.74 cm
  crouching (pelvis asked)       n   161   hand median  9.14 p90 13.83   arm median 13.08 p90 19.92 cm
  0 bends                        n    56   hand median  6.14 p90 10.93   arm median 10.38 p90 16.46 cm
  1 bends                        n   122   hand median  7.28 p90 11.49   arm median 11.11 p90 16.20 cm
  2 bends                        n    78   hand median  6.56 p90 12.45   arm median 11.47 p90 20.12 cm
  3 bends                        n    44   hand median  7.68 p90 12.25   arm median 12.55 p90 19.41 cm
  speed 0.15 m/s                 n   142   hand median  8.01 p90 12.10   arm median 11.53 p90 16.15 cm
  speed below 0.15               n    60   hand median  6.23 p90 11.12   arm median 10.96 p90 18.11 cm
  speed above 0.15               n    98   hand median  6.39 p90 11.59   arm median 10.55 p90 17.79 cm
the arm's worst sphere is on: {'shoulder_pitch_link': 21, 'shoulder_roll_link': 4, 'shoulder_yaw_link': 72, 'elbow_link': 2, 'wrist_pitch_link': 24, 'hand_palm_link': 1, 'hand_thumb_0_link': 1, 'hand_thumb_1_link': 0, 'hand_thumb_
```

真实路线回放（同上三个脚本）：

```text
140 segments, 11404 steps, fell 0, walked 105, ok 25
140 segments, 11404 ticks, 4162 reference poses, 24 spheres -> /workspace/code/locomanipulation-selfwm/outputs/reachwm_20261030/data_via_replay/arm.npz
rebuilt jaw centre against the recorded one: median 0.056 cm, 95 % 0.091, max 0.160
last reference pose's jaw against the command: median 0.281 cm, 95 % 0.463 (the plan's own IK miss: median 0.010, 95 % 0.010)
paths 140, fell 0, analysed 140; plan's IK miss > 2 cm: 0; skipped resets 0
largest offset from the reference path (hand: jaw centre; arm: any of the 24 spheres):
  all                            n   140   hand median  3.79 p90  4.54   arm median  6.46 p90  9.85 cm
  standing                       n   140   hand median  3.79 p90  4.54   arm median  6.46 p90  9.85 cm
the arm's worst sphere is on: {'shoulder_pitch_link': 0, 'shoulder_roll_link': 1, 'shoulder_yaw_link': 10, 'elbow_link': 0, 'wrist_pitch_link': 0, 'hand_palm_link': 0, 'hand_thumb_0_link': 0, 'hand_thumb_1_link': 0, 'hand_thumb_2_
replayed paths, per path over its repeats (steadiness: mean cosine between repeats, 1 = the same direction every time):
  TabletopGraspMP_L0_e0              repeats 4  hand  3.42 cm  arm  6.13 cm  hand steadiness  0.14  elbow steadiness  0.89
  TabletopGraspMP_L0_e1              repeats 4  hand  3.32 cm  arm  6.97 cm  hand steadiness  0.16  elbow steadiness  0.95
  TabletopGraspMP_L0_e2              repeats 4  hand  3.75 cm  arm 10.25 cm  hand steadiness -0.16  elbow steadiness  0.97
  TabletopGraspMP_L0_e3              repeats 4  hand  4.11 cm  arm  7.22 cm  hand steadiness  0.11  elbow steadiness  0.92
  TabletopGraspMP_L0_e4              repeats 4  hand  3.47 cm  arm  5.35 cm  hand steadiness  0.18  elbow steadiness  0.78
  TabletopGraspMP_L0_e5              repeats 4  hand  4.20 cm  arm  9.32 cm  hand steadiness  0.02  elbow steadiness  0.48
  TabletopGraspMP_L0_e6              repeats 4  hand  3.82 cm  arm  9.72 cm  hand steadiness  0.20  elbow steadiness  0.96
  TabletopGraspMP_L0_e7              repeats 4  hand  4.25 cm  arm  9.73 cm  hand steadiness  0.18  elbow steadiness  0.95
  TabletopGraspMP_L0_e8              repeats 4  hand  4.43 cm  arm  6.68 cm  hand steadiness  0.28  elbow steadiness  0.81
  TabletopGraspMP_L0_e9              repeats 4  hand  2.99 cm  arm  6.34 cm  hand steadiness  0.18  elbow steadiness  0.74
  TabletopGraspMP_L1_e0              repeats 4  hand  3.54 cm  arm  5.73 cm  hand steadiness  0.08  elbow steadiness  0.91
  TabletopGraspMP_L1_e1              repeats 4  hand  3.93 cm  arm  5.83 cm  hand steadiness  0.13  elbow steadiness  0.85
  TabletopGraspMP_L1_e3              repeats 4  hand  3.91 cm  arm  6.18 cm  hand steadiness -0.08  elbow steadiness  0.73
  TabletopGraspMP_L1_e4              repeats 4  hand  3.65 cm  arm  6.33 cm  hand steadiness  0.11  elbow steadiness  0.95
  TabletopGraspMP_L1_e5              repeats 4  hand  4.16 cm  arm  7.22 cm  hand steadiness  0.12  elbow steadiness  0.84
  TabletopGraspMP_L1_e6              repeats 4  hand  4.65 cm  arm  9.10 cm  hand steadiness  0.99  elbow steadiness  0.88
  TabletopGraspMP_L1_e7              repeats 4  hand  4.20 cm  arm 10.30 cm  hand steadiness -0.08  elbow steadiness  0.83
  TabletopGraspMP_L1_e8              repeats 4  hand  4.05 cm  arm  6.92 cm  hand steadiness  0.11  elbow steadiness  0.87
  TabletopGraspMP_L1_e9              repeats 4  hand  4.36 cm  arm  8.01 cm  hand steadiness  0.06  elbow steadiness  0.90
  TabletopGraspMP_L2_e0              repeats 4  hand  2.25 cm  arm  4.23 cm  hand steadiness  0.17  elbow steadiness  0.64
  TabletopGraspMP_L2_e1              repeats 4  hand  2.83 cm  arm  4.48 cm  hand steadiness  0.22  elbow steadiness  0.88
  TabletopGraspMP_L2_e2              repeats 4  hand  3.92 cm  arm  8.72 cm  hand steadiness  0.24  elbow steadiness  0.95
  TabletopGraspMP_L2_e3              repeats 4  hand  4.09 cm  arm  6.99 cm  hand steadiness -0.11  elbow steadiness  0.85
  TabletopGraspMP_L2_e5              repeats 4  hand  4.10 cm  arm  6.45 cm  hand steadiness  0.08  elbow steadiness  0.95
  TabletopGraspMP_L2_e6              repeats 4  hand  4.47 cm  arm 10.18 cm  hand steadiness  0.99  elbow steadiness  0.93
  TabletopGraspMP_L2_e7              repeats 4  hand  3.45 cm  arm  6.28 cm  hand steadiness  0.11  elbow steadiness  0.84
  TabletopGraspMP_L2_e8              repeats 4  hand  3.41 cm  arm  5.95 cm  hand steadiness -0.17  elbow steadiness  0.80
  TabletopGraspMP_L2_e9              repeats 4  hand  4.00 cm  arm  7.76 cm  hand steadiness  0.09  elbow steadiness  0.94
  XMovePickTeleop_L1_e5              repeats 4  hand  3.24 cm  arm  4.02 cm  hand steadiness  0.98  elbow steadiness  0.83
  XMovePickTeleop_L1_e8              repeats 4  hand  2.55 cm  arm  3.57 cm  hand steadiness  0.45  elbow steadiness  0.57
  XMovePickTeleop_L1_e9              repeats 4  hand  3.29 cm  arm  4.91 cm  hand steadiness  0.33  elbow steadiness  0.89
  XMovePickTeleop_L2_e4              repeats 4  hand  3.32 cm  arm  4.99 cm  hand steadiness  0.51  elbow steadiness -0.12
  XMovePickTeleop_L2_e6              repeats 4  hand  3.67 cm  arm  4.89 cm  hand steadiness  0.99  elbow steadiness  0.83
  XMovePickTeleop_L2_e7              repeats 4  hand  3.47 cm  arm  4.93 cm  hand steadiness  0.43  elbow steadiness  0.02
  XMovePickTeleop_L2_e9              repeats 4  hand  2.81 cm  arm  4.01 cm  hand steadiness  0.43  elbow steadiness  0.55
  hand steadiness over 35 paths: median 0.16, lowest -0.17
  elbow steadiness over 35 paths: median 0.85, lowest -0.12
```

覆盖检查（`via_cover.py`）：

```text
random paths 300, real paths 35; a random path's nearest random path: median distance 1.394
feature            n_via  straight     ratio     off_x   off_out     off_z      turn        ax     a_out        az        bx     b_out        bz
random p05         0.000     0.296     1.000    -0.152    -0.238    -0.145    34.889     0.100     0.215     0.714     0.327     0.061     0.224
random p95         9.050     0.635     2.763     0.124     0.272     0.237   186.416     0.124     0.236     0.748     0.624     0.317     1.076
real min           1.000     0.243     1.007    -0.026    -0.062    -0.024    88.886     0.065     0.216     0.692     0.272     0.071     0.911
real max          10.000     0.449     2.313     0.020     0.248     0.206   209.087     0.120     0.329     0.735     0.376     0.195     1.000
TabletopGraspMP_L0_e0              nearest random path at  1.09 x the median  FAIL  largest gap in b_out
TabletopGraspMP_L0_e1              nearest random path at  1.27 x the median  FAIL  largest gap in bx
TabletopGraspMP_L0_e2              nearest random path at  1.22 x the median  FAIL  largest gap in bx
TabletopGraspMP_L0_e3              nearest random path at  1.11 x the median  FAIL  largest gap in bx
TabletopGraspMP_L0_e4              nearest random path at  0.87 x the median  pass  largest gap in n_via
TabletopGraspMP_L0_e5              nearest random path at  1.42 x the median  FAIL  largest gap in bx
TabletopGraspMP_L0_e6              nearest random path at  1.22 x the median  FAIL  largest gap in bx
TabletopGraspMP_L0_e7              nearest random path at  1.24 x the median  FAIL  largest gap in bx
TabletopGraspMP_L0_e8              nearest random path at  1.13 x the median  FAIL  largest gap in bx
TabletopGraspMP_L0_e9              nearest random path at  1.08 x the median  FAIL  largest gap in turn
TabletopGraspMP_L1_e0              nearest random path at  1.08 x the median  FAIL  largest gap in turn
TabletopGraspMP_L1_e1              nearest random path at  1.13 x the median  FAIL  largest gap in b_out
TabletopGraspMP_L1_e3              nearest random path at  0.87 x the median  pass  largest gap in bx
TabletopGraspMP_L1_e4              nearest random path at  0.89 x the median  pass  largest gap in bx
TabletopGraspMP_L1_e5              nearest random path at  1.05 x the median  FAIL  largest gap in bx
TabletopGraspMP_L1_e6              nearest random path at  1.33 x the median  FAIL  largest gap in bx
TabletopGraspMP_L1_e7              nearest random path at  1.42 x the median  FAIL  largest gap in bx
TabletopGraspMP_L1_e8              nearest random path at  1.20 x the median  FAIL  largest gap in bx
TabletopGraspMP_L1_e9              nearest random path at  1.53 x the median  FAIL  largest gap in bx
TabletopGraspMP_L2_e0              nearest random path at  0.71 x the median  pass  largest gap in off_out
TabletopGraspMP_L2_e1              nearest random path at  0.82 x the median  pass  largest gap in n_via
TabletopGraspMP_L2_e2              nearest random path at  1.43 x the median  FAIL  largest gap in turn
TabletopGraspMP_L2_e3              nearest random path at  0.97 x the median  pass  largest gap in bx
TabletopGraspMP_L2_e5              nearest random path at  1.07 x the median  FAIL  largest gap in bx
TabletopGraspMP_L2_e6              nearest random path at  1.68 x the median  FAIL  largest gap in bx
TabletopGraspMP_L2_e7              nearest random path at  1.02 x the median  FAIL  largest gap in turn
TabletopGraspMP_L2_e8              nearest random path at  0.90 x the median  pass  largest gap in turn
TabletopGraspMP_L2_e9              nearest random path at  1.28 x the median  FAIL  largest gap in bx
XMovePickTeleop_L1_e5              nearest random path at  0.98 x the median  pass  largest gap in off_x
XMovePickTeleop_L1_e8              nearest random path at  0.73 x the median  pass  largest gap in b_out
XMovePickTeleop_L1_e9              nearest random path at  0.84 x the median  pass  largest gap in straight
XMovePickTeleop_L2_e4              nearest random path at  0.79 x the median  pass  largest gap in b_out
XMovePickTeleop_L2_e6              nearest random path at  0.79 x the median  pass  largest gap in off_x
XMovePickTeleop_L2_e7              nearest random path at  0.80 x the median  pass  largest gap in b_out
XMovePickTeleop_L2_e9              nearest random path at  0.80 x the median  pass  largest gap in bx
covered 14 of 35 real paths; the rule asks for all of them
```

启动后加的分析（`via_profile.py`，见"预测修订"）：

```text
paths 35, repeats 4-4 each
where on the path the largest offset falls differs between a path's repeats by 0.53 of the path (median; 90 % 0.75)
agreement of the whole profile between repeats: median 0.84, 10 % 0.65, lowest 0.39
largest bin of the profile: 3.63 cm; left over after the path's other repeats' mean: 1.56 cm (0.24 of the squared size); after the other paths' mean: 2.39 cm (0.46)
  TabletopGraspMP      n 28  agreement 0.86  left over 1.53 cm of 3.90; knowing only the other paths 2.30 cm
  XMovePickTeleop      n  7  agreement 0.64  left over 1.90 cm of 2.94; knowing only the other paths 2.91 cm
offset in the last bin (ahead, left, up cm): first pass [-2.5   0.14  0.84], later passes [-2.74  0.15  0.91]
```

补生成器之后的探索（300 次开场，种子 35000000，代码是提交前的工作区；`via_cover.py` 输出的前 6 行和汇总行，逐条结果在 `data_via_trial/cover_after_widening_explore.txt`）：

```text
random paths 300, real paths 35; a random path's nearest random path: median distance 1.392
feature            n_via  straight     ratio     off_x   off_out     off_z      turn        ax     a_out        az        bx     b_out        bz
random p05         0.000     0.232     1.000    -0.162    -0.198    -0.155    30.657     0.101     0.213     0.713     0.290     0.051     0.246
random p95         8.000     0.626     2.850     0.140     0.238     0.234   258.510     0.125     0.238     0.749     0.625     0.306     1.100
real min           1.000     0.243     1.007    -0.026    -0.062    -0.024    88.886     0.065     0.216     0.692     0.272     0.071     0.911
real max          10.000     0.449     2.313     0.020     0.248     0.206   209.087     0.120     0.329     0.735     0.376     0.195     1.000
covered 9 of 35 real paths; the rule asks for all of them
```

## 意外

- 手的方向稳定度 0.16，远低于预测的 0.6–0.9，而肘是 0.85。原因见"预测修订"：统计量取的时刻在两处偏差之间跳。
- 规划的 IK 误差超过 2 cm 的占 11.3 %，预测是低于 5 %。站着 1 %，蹲下 20 %：误差来自蹲低的伸手。
- 覆盖检查只通过 14 / 35，没通过的全是 TabletopGrasp。真实目标在身前 27–38 cm、高 0.91–1.00 m；同样高度的随机目标在 32–57 cm，只有 18 % 在 38 cm 以内。生成器沿用了抓取房间的目标分布，没有 SIMPLE 桌面抓取这种贴着桌子站的近身目标。
- 回放时手的偏差有相当一部分对所有路线都一样：终点处手停在命令之前（后几遍平均往后 3.4 cm，独立复核的数）。只知道"别的路线平均怎么偏"，就能把最大偏差从 3.63 cm 降到 2.39 cm；知道这条路线自己其余几遍，降到 1.56 cm。
- 第 1 遍（原地站立出发）和后 3 遍（走一小段后出发）的偏差不同，复核给的终点偏差是往后 1.6 cm 对 3.4 cm。第 1 遍只有一种起始状态。

## 结论与下一步

对照事先的预测和判据：

| 项目 | 预测或判据 | 实测 | 结论 |
|---|---|---|---|
| 摔倒 | < 1 % | 0 / 300，0 / 140 | 符合 |
| 规划的 IK 误差 > 2 cm | < 5 % | 11.3 % | 没猜中 |
| 随机路线，手偏离参考路线（中位数） | 3–8 cm | 6.96 cm | 符合 |
| 真实回放，手偏离参考路线（中位数） | 2–6 cm；< 1.5 cm 就停 | 3.79 cm | 符合，不触发停止 |
| 真实回放，手的方向稳定度 | 0.6–0.9；≤ 0.3 就停 | 0.16 | 按字面触发停止；统计量无效，见"预测修订" |
| 覆盖检查 | 35 条全部盖住 | 14 条 | 没通过，生成器要先补 |

- **空地上身体确实偏。**回放的 3.79 cm 和在桌边测到的 3.8–4.1 cm 几乎一样，那次的偏差来自身体，不是碰桌子。
- **偏差是可重复的。**整条偏差曲线在各遍之间的余弦中位数 0.84（复核按时间对齐算是 0.87）；用其余几遍的平均预测，剩下 1.56 cm，原来是 3.63 cm。仿真是确定的，各遍只差起始状态，所以这说的是"对起始状态不敏感"，不是"噪声小"。
- **其中一部分是和路线无关的常数。**所以后面的比较除了"标量半径"，还要加一个"所有路线减同一条平均偏差曲线"的对手，方向模型要比它好才算数。这一条等用户同意后写进判据。
- **按事先的判据，这里应当停下报告用户。**已报告（2026-11-20）。建议继续，但要先做两件事：把近身目标补进生成器并重做覆盖检查；把常数修正这个对手写进判据。

## 独立复核

2026-11-20，新开的 agent，只给了数据路径、脚本、事先的统计量和判据、以及手 0.16 / 肘 0.85 这个结果，没有给我的解释。它的代码和输出在 `/workspace/tmp/review_via_20261120/`。原文：

**1. Recomputation (verified).** My own code reproduces the registered numbers.
- Hand steadiness: median 0.160, lowest -0.170.
- Hand largest offset: median 3.79 cm, p90 4.54 cm.
- Elbow steadiness: median 0.853, from via_dev.npz and my cosine code. I did not recompute the elbow offsets themselves.

**2. Why hand and elbow disagree (verified, except where marked inference).**
- The hand's largest offset is an argmax over a trace that has two competing peaks of nearly equal size. An early peak sits near tick 27, at about (+2.5, +2, -2) cm. A late peak sits at the end of the segment, at about (-3.5, 0, +1) cm, where the hand stops short of the commanded jaw.
- The two peaks point in nearly opposite directions. The argmax flips between them from repeat to repeat. The late peak wins in 40% of walked passes and in 3% of pass-1 passes. The offset magnitude is almost the same either way (3.4 to 4.5 cm), which explains why the median and p90 look tight.
- Within a route, the peak vector flips between two clusters, so the cosine is near 0.
- Pass 1 (standing reset, 35 segments) differs from passes 2-4 (walked, 105 segments), and this effect is real.
  - Pass 1 is always the same pristine stand. Walked starts are all distinct, differing by a few cm and a few degrees of yaw.
  - The mean end offset is -1.6 cm in x for pass 1 and -3.4 cm for walked.
  - The pass-1 peak against the mean of the walked peaks has median cosine -0.42.
- The registered statistic is therefore not a valid measure of "strays the same way each time". It is an unstable argmax that mixes two regimes, so it understates repeatability.
- Replacements I computed:

| Measure | Result |
|---|---|
| Peak vector, 3 walked repeats | steadiness 1.00 [0.99, 1.00] |
| Peak vector, 4 repeats at the route-mean-peak tick | 0.97 [0.96, 0.98] |
| Whole offset trace (hand minus time-matched reference, heading frame), 4 repeats | cosine 0.87 [0.85, 0.88] |
| Whole offset trace, 3 walked repeats | cosine 0.98 [0.97, 0.98] |
| Leave-one-out prediction from the other repeats' mean | residual SS fraction 0.19 (4 repeats), 0.05 (walked only) |

- Intervals are 95% bootstrap over routes (resampling the 35 per-route values), not over repeats.
- By task: 28 Tabletop routes give 0.86 and 7 XMove routes give 0.88 on the 4-repeat whole-trace cosine. The task difference is small.
- The elbow is more stable because its peak is not a competition between two opposite-direction peaks. The elbow vectors also share a common direction (mean (-2.3, 0, -2.4) cm against a median magnitude of 4.2 cm). Different routes still give an elbow cosine of 0.61, against 0.79 for the same route.
- The hand is less shared across routes: same-route cosine 0.24 against different-route 0.17 with the registered peak. Restricted to walked passes it is 0.73 against 0.49.

**3. Answer.**
- Repeats of the same route give the same hand deviation trace. About 81% of the trace's energy is predictable from the other repeats, and 95% among walked passes.
- Only part of that is route-specific. The terminal stop-short is mostly a global constant: mean (-3.4, +0.2, +1.0) cm for walked passes, with within-route SD (0.34, 0.26, 0.08) cm.
- Predicting the end offset from the same route's other repeats is worth much more than predicting it from other routes. The leave-one-out sum of squares is 30 against 132, though the 132 includes the pass-1/walked difference, so it is not a clean comparison.
- Run-to-run noise is small (sub-cm to about 1 cm), and the offset is systematic and learnable. By the registered rule (at least 0.6 means go on) the correct answer is "go on". The 0.16 is an artefact of the statistic.

**4. Problems and what the data cannot show.**
- Replays are deterministic simulation. Passes 2-4 differ only through their different start poses, so this is sensitivity to the start state, not true stochastic noise. 35 routes with 4 repeats cannot separate simulator noise from start-state effects.
- Pass 1 is confounded with "no walk". Its single stand state is not randomised, and the deviation is compared at only one standing configuration.
- The registered statistic should not be reported as "noise". With the argmax, even two near-identical traces can score cosine -1 when their peaks swap.
- Cosine ignores magnitude, and the offsets are only about 3 to 4 cm.
- The "reference" is the commanded jaw per tick, carried through the current pelvis pose. The hand-minus-reference offset therefore includes the pelvis and controller lag and stop-short, not only the arm path.
- The elbow comparison is not on the same footing as the hand. It uses a nearest-point polyline distance for each of the 24 spheres, so lag along the path is not counted, and the hand statistic is not like for like with it.
- Whether a model could predict the offset better than the constant bias plus a per-route mean is untested. I tested only prediction from the route's own other repeats.
- Not checked: whether the reference-pose spheres in arm.npz match what the controller actually tracked.
