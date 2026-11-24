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

**规则在试跑后改了**（2026-11-21，用户选的，原话："A 不加那一点"；原因和数字见"预测修订"）：一条真实路线只要不比 95 % 的随机路线更孤立，就算盖住。孤立程度指到最近一条随机路线的距离。生成器保持通用，不向这 35 条靠；没盖住的路线照实列出。上面那条中位数规则的结果仍然一起打印。

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

### 模型和评测的具体定义（2026-11-21，正式数据出来之前写下；脚本 `via_model.py`）

- **一行数据**：正式采集里每次开场的第一段伸手，摔倒的去掉。所有量都在下命令那一刻的朝向坐标系里，不做左右镜像，哪只手是输入之一。
- **参考扫过的路线**：参考规划时长等分成 24 个时刻，每个时刻 24 个碰撞球的位置。
- **实际扫过的路线**：下命令后同样 24 个时刻的实际位置，再加一个"稳定后"的点，取录像最后 0.5 秒的平均。
- **偏差**：实际减参考，共 25 个点 × 24 个球 × 3。预测的路线就是参考加预测的偏差，每个球一条 25 个点的折线。
- **扫过区域误差**（主指标，沿用上面的定义）：从下命令到录像结束，任何一帧任何一个球到它那条预测折线的最大距离。沿路线的滞后不算，偏到路线旁边才算。"手"是只看手掌 3 个球的同一个量。
- **三个预测**：
  - 参考：偏差为 0，就是"手臂照参考走"；
  - 常数修正：训练集上每只手一条平均偏差曲线；
  - 方向模型：5 个 MLP 的集成（每个 3 层、宽 512，各自在训练集的自助抽样上训练），输出是偏差在训练集主成分上的系数（保留 99.5 % 方差）。
- **输入**：下命令时和之前两帧的身体状态、手的位置和朝向、目标和朝向、最多 16 个路点和朝向、速度、要求的骨盆高度、哪只手、参考规划的时长、参考路线里 5 个球（肩、上臂、肘、腕、手掌）的位置。
- **半径**：每个预测各配一个小集成，从同样的输入预测自己这次的误差有多大。"标量半径"就是配给参考的那个。训练它用的是预测没拟合过的误差：训练集里一条路线的方向预测，只取自助抽样时没抽到它的那几个成员。半径整体乘一个系数，在校准集上调到 95 % 的误差落在里面；报告测试集上半径的中位数和实际落在里面的比例。
- **划分**：按开场编号的哈希分 10 桶，0–7 训练，8 验证兼校准，9 测试。扇区留出：最大的那个弯朝里朝下（角度 202.5–247.5°）的路线不进训练和校准，单独作为一个测试集。真实路线回放也只作测试。
- **判据对应脚本打印的哪一行**：
  - "模型相对基线降幅"看 `drop of the median against the reference` 里的 arm，也就是 24 个球的主指标；hand 一并报告。上面"预测"里写的 1.5–3 cm 说的是手。
  - 两个余量之比看 `margins:` 那一行。
- 这套流程在试跑的 300 条数据上跑过一遍，只为确认脚本能跑；训练集只有 233 条，那次的数字不作任何结论。那次的输出里也包含了真实路线回放的一栏，我看到过。

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

## 第二轮（2026-11-21，用户选 A；本节在任何第二轮的运行之前写下）

第一轮的结果：随机路线上降 38 %，35 条真实路线上没有好处。复核查出的原因是三条：训练数据里没有"不走路直接出发"的起始状态；路线越长好处越小；模型的好处集中在路线开头。

### 假设

H2：偏差是可重复的，学不到是因为模型把整条路线一次交给网络。改成沿路线逐点预测，每个点的偏差由起始状态和那一段参考运动决定，模型在短路线上学到的就能用到长路线上。再补上两类缺的数据，它在长的站立路线和新的真实路线上都应当比"照参考走"好。

### 改什么

- **模型**：逐点预测。一行数据是一条路线上的一个点（24 个时刻加稳定后的一点），输入是整条路线共用的量（下命令时的状态、手的起点、目标、速度、骨盆高度、哪只手、规划时长）加这个点附近的参考运动（进度、已走和剩余的路程、这个点前后几个时刻 5 个关键球的参考位置）；输出是这个点上 24 个球的偏差。所有点共用一个网络，5 个成员按路线做自助抽样。损失直接用偏差的平方误差，不再给每个分量同样的权重；每 200 步在验证集上看一次偏差的平方误差来选最好的状态。
- **数据**：补采一份。30 % 的开场不走路，从站立直接出发；50 % 的路线用"长路线"模式，1 到 3 个弯、偏离量 15–40 cm。其余不变。这两类是按复核指出的缺口补的通用类别，不按那 35 条路线的形状去凑。
- **半径**：校准用的数据和选最好状态用的数据分开，把验证桶按开场编号再对半分。余量之比在两边实际覆盖率不同时，同时报告固定半径之比。
- **最终测试集**：在这台机器上用 cuRobo 另外规划一批新的真实路线，场景和那 35 条无关，规划完不看结果直接回放。原来那 35 条已经被看过多次，只作参考，不作最终判据。

### 纪律

- 模型结构、正则、选哪个状态，只看训练集和验证集。测试集、扇区留出、新旧真实路线，在模型定下来之前不看。
- 定下来的含义：验证集上偏差的平方误差最低的那个设置，并且换随机种子后排名不变。

### 预测

- 验证集里手走过 1 米以上的站立路线，扫过区域误差中位数相对"照参考走"的降幅：25–40 %。第一轮的模型在这一类上是 13 %。
- 普通留出：降 40–50 %。
- 新的真实路线：降 20–35 %。
- 新的真实路线上，方向模型和标量半径的固定半径之比：不超过 0.8。

### 判据

- **停止线（用户 2026-11-21 同意）**：验证集里 1 米以上的站立路线，降幅到不了 25 % → 停，不再看测试集，向用户报告。
- 过了停止线才看最终测试。在新的真实路线上：
  - 降幅 ≥ 20 %，按路线自助抽样的 95 % 区间不含 0，而且固定半径之比 ≤ 0.8：成立，进入窄缝测试；
  - 降幅的区间含 0：这条线停，向用户报告；
  - 其余情况：报告给用户决定。
- 新的真实路线如果规划不出来或少于 20 条，这一轮的最终结论只能下到"普通留出和长路线"，要向用户说明。

### 第二轮的过程（只用训练集和验证集的部分）

逐点模型在第一批 12000 条上的训练（`via_point.py train`，模型在 `outputs/reachwm_20261030/via_point_dev_a`，设置：dropout 0.1、权重衰减 1e-4、宽 512、4 层、30000 步）：

```text
paths 12000: train 9055, validation for early stopping 570, for calibration 594, test 1117, sector hold-out 664
rows: train 226375, early-stopping validation 14250; input 336; validation squared offset 12.37 cm2 a coordinate with no model
member 0: validation squared offset 2.006 cm2 at iteration 17600 (first check 3.725, last 2.047; 68s)
member 1: validation squared offset 1.881 cm2 at iteration 23400 (first check 3.525, last 1.889; 136s)
member 2: validation squared offset 1.830 cm2 at iteration 15600 (first check 3.534, last 1.874; 204s)
member 3: validation squared offset 2.001 cm2 at iteration 22200 (first check 3.651, last 2.011; 273s)
member 4: validation squared offset 1.927 cm2 at iteration 20400 (first check 3.569, last 1.968; 342s)
ensemble's members: mean validation squared offset 1.929 cm2
direction: 8120 of 9055 training paths have an out-of-bag prediction
radius[reference]: scale 1.480; calibration paths' error median 10.75 cm, 95% level 19.06 cm
radius[constant]: scale 1.510; calibration paths' error median 9.30 cm, 95% level 18.61 cm
radius[direction]: scale 1.929; calibration paths' error median 3.38 cm, 95% level 13.12 cm
```

验证集（`via_point.py dev`；这些路线参与了"选停在哪一步"和半径校准，数字略偏乐观）：

```text
validation, all: 1164 paths
  reference  swept-region error: arm median 10.95 p90 16.97 cm | hand median  7.79 cm | margin (median radius) 16.49 cm, inside 0.95 | fixed radius 19.06 cm, inside 0.95
  constant   swept-region error: arm median  9.50 p90 16.12 cm | hand median  7.10 cm | margin (median radius) 14.70 cm, inside 0.95 | fixed radius 18.61 cm, inside 0.95
  direction  swept-region error: arm median  3.35 p90 10.36 cm | hand median  2.60 cm | margin (median radius)  7.06 cm, inside 0.95 | fixed radius 13.12 cm, inside 0.96
  drop of the arm's median against the reference: 0.69 (95 % interval 0.68 to 0.71, resampling paths); hand 0.67; constant: arm 0.13
  fixed radius: direction / reference 0.69 (set on the calibration paths); direction wins on 0.99 of the paths
validation, standing: 472 paths
  reference  swept-region error: arm median  6.65 p90 12.98 cm | hand median  4.34 cm | margin (median radius) 10.07 cm, inside 0.93 | fixed radius 19.06 cm, inside 0.98
  constant   swept-region error: arm median  6.76 p90 12.14 cm | hand median  4.42 cm | margin (median radius) 10.16 cm, inside 0.94 | fixed radius 18.61 cm, inside 0.98
  direction  swept-region error: arm median  1.76 p90  5.00 cm | hand median  1.28 cm | margin (median radius)  3.88 cm, inside 0.93 | fixed radius 13.12 cm, inside 1.00
  drop of the arm's median against the reference: 0.74 (95 % interval 0.71 to 0.75, resampling paths); hand 0.71; constant: arm -0.02
  fixed radius: direction / reference 0.69 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, palm path over 1.0 m (THE STOP LINE: drop of the arm's median >= 0.25): 34 paths
  reference  swept-region error: arm median  9.07 p90 15.75 cm | hand median  6.31 cm | margin (median radius) 14.13 cm, inside 0.94 | fixed radius 19.06 cm, inside 0.94
  constant   swept-region error: arm median  9.86 p90 15.37 cm | hand median  7.18 cm | margin (median radius) 14.19 cm, inside 0.94 | fixed radius 18.61 cm, inside 0.94
  direction  swept-region error: arm median  3.70 p90  8.96 cm | hand median  2.69 cm | margin (median radius)  8.49 cm, inside 0.97 | fixed radius 13.12 cm, inside 1.00
  drop of the arm's median against the reference: 0.59 (95 % interval 0.50 to 0.72, resampling paths); hand 0.57; constant: arm -0.09
  fixed radius: direction / reference 0.69 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, palm path under 0.6 m: 263 paths
  reference  swept-region error: arm median  5.69 p90 10.55 cm | hand median  3.71 cm | margin (median radius)  8.49 cm, inside 0.95 | fixed radius 19.06 cm, inside 1.00
  constant   swept-region error: arm median  6.14 p90  9.85 cm | hand median  4.07 cm | margin (median radius)  9.35 cm, inside 0.97 | fixed radius 18.61 cm, inside 1.00
  direction  swept-region error: arm median  1.22 p90  3.35 cm | hand median  0.93 cm | margin (median radius)  2.92 cm, inside 0.94 | fixed radius 13.12 cm, inside 1.00
  drop of the arm's median against the reference: 0.79 (95 % interval 0.76 to 0.80, resampling paths); hand 0.75; constant: arm -0.08
  fixed radius: direction / reference 0.69 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, no walk before the reach: 0 paths, too few
validation, crouching: 692 paths
  reference  swept-region error: arm median 12.84 p90 18.05 cm | hand median  9.91 cm | margin (median radius) 19.13 cm, inside 0.97 | fixed radius 19.06 cm, inside 0.93
  constant   swept-region error: arm median 11.15 p90 17.00 cm | hand median  8.61 cm | margin (median radius) 17.13 cm, inside 0.95 | fixed radius 18.61 cm, inside 0.94
  direction  swept-region error: arm median  5.02 p90 11.78 cm | hand median  4.03 cm | margin (median radius) 10.35 cm, inside 0.96 | fixed radius 13.12 cm, inside 0.94
  drop of the arm's median against the reference: 0.61 (95 % interval 0.59 to 0.63, resampling paths); hand 0.59; constant: arm 0.13
  fixed radius: direction / reference 0.69 (set on the calibration paths); direction wins on 0.99 of the paths
```

停止线：验证集里站立、手走过 1 米以上的 34 条，降幅 0.59，区间 0.50–0.72，过线。事先的预测是 25–40 %。

设置比较，只看验证集上的偏差平方误差（越小越好；第一行的 a 是上面那个模型，1.929）：

```text
via_point_dev_b --dropout 0.0 -> ensemble's members: mean validation squared offset 2.204 cm2
via_point_dev_c --dropout 0.2 --wd 0.01 -> ensemble's members: mean validation squared offset 1.952 cm2
via_point_dev_d --dropout 0.1 --width 1024 -> ensemble's members: mean validation squared offset 2.000 cm2
via_point_dev_e --dropout 0.1 --seed 1 -> ensemble's members: mean validation squared offset 2.008 cm2
sweep done
```

同一设置换种子是 1.93 和 2.01，所以 dropout 0.2、宽 1024 和默认设置分不出高下，不用 dropout 明显更差。定为默认设置，不再调。

新的真实路线（`fresh_routes.py`，种子 20261121，输出在 `outputs/reachwm_20261030/fresh_routes`，逐个场景的结果在 `scenes.json`）：150 个场景规划出 63 条。越过桌沿 9 / 50，越过桌上的盒子 12 / 50，这 21 条都是 7–8 个路点、长度约为直线的 2.1 倍；伸到板子下面 42 / 50，1–3 个路点、几乎是直线。没规划出来的主要是逆解失败：锁腰只动右臂时够不着。场景参数在正式种子之前用别的种子试过两次：目标原来放在身前 40–60 cm，8 个里只规划出 1 个，改到肩膀 40 cm 以内；立在身前的柱子和矮墙正好在垂着的手的位置，规划器拒绝从碰撞出发，去掉了。63 条每条回放 4 遍，数据在 `data_via_fresh`。

### 第二轮的最终模型和评测（2026-11-21）

补采：2026-11-21 14:58–17:28 UTC，vastgpu，4 个进程，10000 次开场（种子 39000000，ID0 3100000，`REACH_VIA_STAND=0.3 REACH_VIA_LONG=0.5`），数据在 `data_via2`，没有跳过的开场。当时的采集代码没提交，原样存在 `data_via2/asrun/`。偏差统计：

```text
20000 segments, 1312037 steps, fell 1, walked 14096, ok 2456
20000 segments, 1312037 ticks, 359882 reference poses, 24 spheres -> /workspace/code/locomanipulation-selfwm/outputs/reachwm_20261030/data_via2/arm.npz
rebuilt jaw centre against the recorded one: median 0.030 cm, 95 % 0.129, max 1.019
last reference pose's jaw against the command: median 0.618 cm, 95 % 2.899 (the plan's own IK miss: median 0.010, 95 % 3.090)
paths 10000, fell 0, analysed 10000; plan's IK miss > 2 cm: 1295; skipped resets 0
largest offset from the reference path (hand: jaw centre; arm: any of the 24 spheres):
  all                            n 10000   hand median  6.74 p90 12.17   arm median 10.97 p90 17.75 cm
  standing                       n  4606   hand median  4.25 p90  8.71   arm median  7.62 p90 13.60 cm
  crouching (pelvis asked)       n  5394   hand median  8.75 p90 13.56   arm median 13.07 p90 19.62 cm
  0 bends                        n  1205   hand median  5.20 p90 10.27   arm median  9.48 p90 15.04 cm
  1 bends                        n  4047   hand median  6.23 p90 11.71   arm median 10.41 p90 17.15 cm
  2 bends                        n  3104   hand median  7.36 p90 12.56   arm median 11.45 p90 18.41 cm
  3 bends                        n  1644   hand median  8.02 p90 13.30   arm median 12.04 p90 19.02 cm
  speed 0.15 m/s                 n  5073   hand median  6.86 p90 12.19   arm median 11.04 p90 17.49 cm
  speed below 0.15               n  1982   hand median  6.53 p90 11.64   arm median 10.89 p90 17.89 cm
  speed above 0.15               n  2945   hand median  6.69 p90 12.38   arm median 10.87 p90 18.08 cm
```

最终模型：两批数据合起来 22000 条，默认设置，模型在 `outputs/reachwm_20261030/via_point_20261121`（code: 554a28f8ca6200146ce1f6d485eb2cdc58bca4aa + uncommitted scripts (copies in asrun/): 1e4bd272167cb383  via_point.py 2177c65ea5c2a389  via_model.py  | 2026-11-21T17:31:25Z | eval started 2026-11-21T17:38:40Z: the first and only read of the test sets and the fresh routes with this model）。训练：

```text
paths 22000: train 16591, validation for early stopping 1028, for calibration 1077, test 2080, sector hold-out 1224
rows: train 414775, early-stopping validation 25700; input 336; validation squared offset 12.87 cm2 a coordinate with no model
member 0: validation squared offset 1.779 cm2 at iteration 27200 (first check 3.548, last 1.781; 33s)
member 1: validation squared offset 1.731 cm2 at iteration 17400 (first check 3.589, last 1.771; 65s)
member 2: validation squared offset 1.719 cm2 at iteration 16200 (first check 3.560, last 1.763; 99s)
member 3: validation squared offset 1.772 cm2 at iteration 17800 (first check 3.652, last 1.795; 131s)
member 4: validation squared offset 1.726 cm2 at iteration 27600 (first check 3.700, last 1.729; 164s)
ensemble's members: mean validation squared offset 1.745 cm2
direction: 14964 of 16591 training paths have an out-of-bag prediction
radius[reference]: scale 1.428; calibration paths' error median 10.87 cm, 95% level 19.85 cm
radius[constant]: scale 1.440; calibration paths' error median 9.59 cm, 95% level 19.28 cm
radius[direction]: scale 1.883; calibration paths' error median 3.31 cm, 95% level 12.46 cm
```

验证集（停止线在第三段）：

```text
validation, all: 2105 paths
  reference  swept-region error: arm median 11.05 p90 17.52 cm | hand median  7.71 cm | margin (median radius) 16.13 cm, inside 0.95 | fixed radius 19.85 cm, inside 0.95
  constant   swept-region error: arm median  9.70 p90 16.84 cm | hand median  7.20 cm | margin (median radius) 14.57 cm, inside 0.95 | fixed radius 19.28 cm, inside 0.95
  direction  swept-region error: arm median  3.35 p90 10.25 cm | hand median  2.60 cm | margin (median radius)  6.82 cm, inside 0.95 | fixed radius 12.46 cm, inside 0.95
  drop of the arm's median against the reference: 0.70 (95 % interval 0.68 to 0.71, resampling paths); hand 0.66; constant: arm 0.12
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 0.99 of the paths
validation, standing: 885 paths
  reference  swept-region error: arm median  7.02 p90 13.54 cm | hand median  4.39 cm | margin (median radius) 10.30 cm, inside 0.93 | fixed radius 19.85 cm, inside 0.98
  constant   swept-region error: arm median  6.92 p90 12.98 cm | hand median  4.57 cm | margin (median radius) 10.03 cm, inside 0.94 | fixed radius 19.28 cm, inside 0.98
  direction  swept-region error: arm median  1.76 p90  4.83 cm | hand median  1.27 cm | margin (median radius)  3.69 cm, inside 0.94 | fixed radius 12.46 cm, inside 0.99
  drop of the arm's median against the reference: 0.75 (95 % interval 0.73 to 0.76, resampling paths); hand 0.71; constant: arm 0.01
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, palm path over 1.0 m (THE STOP LINE: drop of the arm's median >= 0.25): 103 paths
  reference  swept-region error: arm median  9.47 p90 15.99 cm | hand median  6.56 cm | margin (median radius) 14.15 cm, inside 0.92 | fixed radius 19.85 cm, inside 0.97
  constant   swept-region error: arm median  9.93 p90 15.80 cm | hand median  6.91 cm | margin (median radius) 13.49 cm, inside 0.92 | fixed radius 19.28 cm, inside 0.97
  direction  swept-region error: arm median  3.20 p90  7.89 cm | hand median  2.50 cm | margin (median radius)  7.02 cm, inside 0.92 | fixed radius 12.46 cm, inside 0.99
  drop of the arm's median against the reference: 0.66 (95 % interval 0.62 to 0.69, resampling paths); hand 0.62; constant: arm -0.05
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, palm path under 0.6 m: 419 paths
  reference  swept-region error: arm median  5.69 p90 10.59 cm | hand median  3.59 cm | margin (median radius)  8.29 cm, inside 0.95 | fixed radius 19.85 cm, inside 1.00
  constant   swept-region error: arm median  5.94 p90  9.97 cm | hand median  3.93 cm | margin (median radius)  8.81 cm, inside 0.97 | fixed radius 19.28 cm, inside 1.00
  direction  swept-region error: arm median  1.21 p90  3.14 cm | hand median  0.89 cm | margin (median radius)  2.58 cm, inside 0.95 | fixed radius 12.46 cm, inside 1.00
  drop of the arm's median against the reference: 0.79 (95 % interval 0.77 to 0.80, resampling paths); hand 0.75; constant: arm -0.04
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
validation, standing, no walk before the reach: 116 paths
  reference  swept-region error: arm median  7.14 p90 15.73 cm | hand median  4.60 cm | margin (median radius) 11.76 cm, inside 0.91 | fixed radius 19.85 cm, inside 0.97
  constant   swept-region error: arm median  7.74 p90 15.42 cm | hand median  5.23 cm | margin (median radius) 11.36 cm, inside 0.96 | fixed radius 19.28 cm, inside 0.97
  direction  swept-region error: arm median  1.60 p90  4.99 cm | hand median  1.11 cm | margin (median radius)  3.94 cm, inside 0.97 | fixed radius 12.46 cm, inside 0.99
  drop of the arm's median against the reference: 0.78 (95 % interval 0.75 to 0.81, resampling paths); hand 0.76; constant: arm -0.08
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
validation, crouching: 1220 paths
  reference  swept-region error: arm median 12.94 p90 18.83 cm | hand median  9.99 cm | margin (median radius) 18.82 cm, inside 0.96 | fixed radius 19.85 cm, inside 0.93
  constant   swept-region error: arm median 11.56 p90 17.96 cm | hand median  8.91 cm | margin (median radius) 17.19 cm, inside 0.95 | fixed radius 19.28 cm, inside 0.93
  direction  swept-region error: arm median  5.12 p90 11.97 cm | hand median  4.11 cm | margin (median radius) 10.18 cm, inside 0.96 | fixed radius 12.46 cm, inside 0.92
  drop of the arm's median against the reference: 0.60 (95 % interval 0.58 to 0.62, resampling paths); hand 0.59; constant: arm 0.11
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 0.98 of the paths
```

停止线：站立、手走过 1 米以上的 103 条，降幅 0.66，区间 0.62–0.69，过线。于是看了最终测试，只看这一次（`via_point.py eval`）：

```text
test, ordinary hold-out: 2080 paths
  reference  swept-region error: arm median 11.02 p90 17.40 cm | hand median  7.80 cm | margin (median radius) 15.91 cm, inside 0.92 | fixed radius 19.85 cm, inside 0.94
  constant   swept-region error: arm median  9.88 p90 16.85 cm | hand median  7.43 cm | margin (median radius) 14.45 cm, inside 0.93 | fixed radius 19.28 cm, inside 0.94
  direction  swept-region error: arm median  3.50 p90 10.32 cm | hand median  2.70 cm | margin (median radius)  6.79 cm, inside 0.94 | fixed radius 12.46 cm, inside 0.95
  drop of the arm's median against the reference: 0.68 (95 % interval 0.67 to 0.70, resampling paths); hand 0.65; constant: arm 0.10
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 0.99 of the paths
test, sector hold-out: 1224 paths
  reference  swept-region error: arm median  7.50 p90 14.88 cm | hand median  4.81 cm | margin (median radius) 10.57 cm, inside 0.91 | fixed radius 19.85 cm, inside 0.97
  constant   swept-region error: arm median  7.11 p90 14.23 cm | hand median  4.91 cm | margin (median radius) 10.22 cm, inside 0.93 | fixed radius 19.28 cm, inside 0.97
  direction  swept-region error: arm median  2.05 p90  6.29 cm | hand median  1.50 cm | margin (median radius)  4.25 cm, inside 0.95 | fixed radius 12.46 cm, inside 0.98
  drop of the arm's median against the reference: 0.73 (95 % interval 0.71 to 0.74, resampling paths); hand 0.69; constant: arm 0.05
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
test, ordinary hold-out, standing: 911 paths
  reference  swept-region error: arm median  7.39 p90 13.09 cm | hand median  4.62 cm | margin (median radius) 10.11 cm, inside 0.91 | fixed radius 19.85 cm, inside 0.99
  constant   swept-region error: arm median  7.02 p90 12.68 cm | hand median  4.75 cm | margin (median radius) 10.08 cm, inside 0.93 | fixed radius 19.28 cm, inside 0.99
  direction  swept-region error: arm median  1.82 p90  4.92 cm | hand median  1.32 cm | margin (median radius)  3.61 cm, inside 0.94 | fixed radius 12.46 cm, inside 0.99
  drop of the arm's median against the reference: 0.75 (95 % interval 0.74 to 0.76, resampling paths); hand 0.71; constant: arm 0.05
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
test, ordinary hold-out, standing, palm path over 1.0 m: 120 paths
  reference  swept-region error: arm median  9.73 p90 13.83 cm | hand median  6.87 cm | margin (median radius) 13.41 cm, inside 0.91 | fixed radius 19.85 cm, inside 0.97
  constant   swept-region error: arm median  9.19 p90 13.61 cm | hand median  6.76 cm | margin (median radius) 12.89 cm, inside 0.91 | fixed radius 19.28 cm, inside 0.97
  direction  swept-region error: arm median  3.78 p90  6.98 cm | hand median  2.78 cm | margin (median radius)  6.92 cm, inside 0.92 | fixed radius 12.46 cm, inside 0.98
  drop of the arm's median against the reference: 0.61 (95 % interval 0.56 to 0.67, resampling paths); hand 0.59; constant: arm 0.06
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 0.99 of the paths
real paths replayed, data_via_fresh: 252 paths
  reference  swept-region error: arm median  4.55 p90  9.54 cm | hand median  2.81 cm | margin (median radius)  6.81 cm, inside 0.98 | fixed radius 19.85 cm, inside 1.00
  constant   swept-region error: arm median  4.30 p90  9.37 cm | hand median  2.89 cm | margin (median radius)  7.04 cm, inside 1.00 | fixed radius 19.28 cm, inside 1.00
  direction  swept-region error: arm median  0.90 p90  4.86 cm | hand median  0.69 cm | margin (median radius)  2.06 cm, inside 0.93 | fixed radius 12.46 cm, inside 1.00
  drop of the arm's median against the reference: 0.80 (95 % interval 0.77 to 0.82, resampling routes); hand 0.75; constant: arm 0.05
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 0.98 of the paths
real paths replayed, data_via_fresh, walked first: 189 paths
  reference  swept-region error: arm median  4.53 p90  9.10 cm | hand median  2.87 cm | margin (median radius)  7.08 cm, inside 0.98 | fixed radius 19.85 cm, inside 0.99
  constant   swept-region error: arm median  3.95 p90  9.35 cm | hand median  2.78 cm | margin (median radius)  6.88 cm, inside 0.99 | fixed radius 19.28 cm, inside 0.99
  direction  swept-region error: arm median  0.98 p90  5.88 cm | hand median  0.74 cm | margin (median radius)  2.02 cm, inside 0.91 | fixed radius 12.46 cm, inside 0.99
  drop of the arm's median against the reference: 0.78 (95 % interval 0.76 to 0.81, resampling routes); hand 0.74; constant: arm 0.13
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 0.98 of the paths
real paths replayed, data_via_replay: 140 paths
  reference  swept-region error: arm median  6.46 p90  9.82 cm | hand median  4.77 cm | margin (median radius)  9.73 cm, inside 0.95 | fixed radius 19.85 cm, inside 1.00
  constant   swept-region error: arm median  7.15 p90  9.64 cm | hand median  4.40 cm | margin (median radius) 10.82 cm, inside 1.00 | fixed radius 19.28 cm, inside 1.00
  direction  swept-region error: arm median  2.27 p90  2.90 cm | hand median  1.79 cm | margin (median radius)  5.85 cm, inside 1.00 | fixed radius 12.46 cm, inside 1.00
  drop of the arm's median against the reference: 0.65 (95 % interval 0.64 to 0.68, resampling routes); hand 0.63; constant: arm -0.11
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
real paths replayed, data_via_replay, walked first: 105 paths
  reference  swept-region error: arm median  6.33 p90  9.70 cm | hand median  4.62 cm | margin (median radius)  9.54 cm, inside 0.94 | fixed radius 19.85 cm, inside 1.00
  constant   swept-region error: arm median  7.06 p90  9.51 cm | hand median  4.29 cm | margin (median radius) 10.62 cm, inside 1.00 | fixed radius 19.28 cm, inside 1.00
  direction  swept-region error: arm median  2.27 p90  2.85 cm | hand median  1.77 cm | margin (median radius)  5.66 cm, inside 1.00 | fixed radius 12.46 cm, inside 1.00
  drop of the arm's median against the reference: 0.64 (95 % interval 0.64 to 0.67, resampling routes); hand 0.62; constant: arm -0.11
  fixed radius: direction / reference 0.63 (set on the calibration paths); direction wins on 1.00 of the paths
```

按路线类型和第几遍拆开（`via_point_20261121/by_kind.py`，用的是同样的函数）：

```text
== data_via_fresh
all                                routes 63 replays 252 | reference  4.55 constant  4.30 direction  0.90 cm | drop 0.80 (0.77 to 0.82) | direction better on 0.98 | palm path 0.39 m
FreshBox                           routes 12 replays  48 | reference  8.23 constant  8.47 direction  3.79 cm | drop 0.54 (0.38 to 0.69) | direction better on 0.92 | palm path 0.83 m
FreshEdge                          routes  9 replays  36 | reference  7.63 constant  7.18 direction  2.57 cm | drop 0.66 (0.55 to 0.80) | direction better on 1.00 | palm path 0.83 m
FreshUnder                         routes 42 replays 168 | reference  3.40 constant  3.82 direction  0.75 cm | drop 0.78 (0.76 to 0.81) | direction better on 1.00 | palm path 0.34 m
first pass (no walk)               routes 63 replays  63 | reference  4.60 constant  4.62 direction  0.76 cm | drop 0.84 (0.80 to 0.86) | direction better on 1.00 | palm path 0.40 m
later passes (walked)              routes 63 replays 189 | reference  4.53 constant  3.95 direction  0.98 cm | drop 0.78 (0.76 to 0.81) | direction better on 0.98 | palm path 0.39 m
== data_via_replay
all                                routes 35 replays 140 | reference  6.46 constant  7.15 direction  2.27 cm | drop 0.65 (0.64 to 0.68) | direction better on 1.00 | palm path 1.00 m
TabletopGraspMP                    routes 28 replays 112 | reference  6.88 constant  7.38 direction  2.34 cm | drop 0.66 (0.64 to 0.69) | direction better on 1.00 | palm path 1.01 m
XMovePickTeleop                    routes  7 replays  28 | reference  4.58 constant  4.79 direction  0.95 cm | drop 0.79 (0.77 to 0.81) | direction better on 1.00 | palm path 0.31 m
first pass (no walk)               routes 35 replays  35 | reference  7.11 constant  7.24 direction  2.32 cm | drop 0.67 (0.63 to 0.72) | direction better on 1.00 | palm path 0.99 m
later passes (walked)              routes 35 replays 105 | reference  6.33 constant  7.06 direction  2.27 cm | drop 0.64 (0.64 to 0.67) | direction better on 1.00 | palm path 1.00 m
```

## 第三步：把预测用回去（2026-11-22，用户："提交推送，然后把预测加回去"；本节在这一步的任何运行之前写下）

第二轮证明了模型预测得准。还没证明的是：按预测把路点反向挪过去之后，手臂是不是真的走到想要的地方。偏差会随命令变，所以挪完要重新预测，可能要迭代。

### 做法

- **想要的路线**：原命令的参考规划扫过的路线，也就是规划器以为手臂会走的地方。
- **补偿**：下命令之前，对候选命令先让控制器只规划不执行，得到它的参考路线，再用逐点模型预测实际会扫过哪里。在每个路点经过的时刻和终点，比较预测的手掌位置和想要的手掌位置，把这个路点（或终点）反向挪这么多。重复 2 次，每个点最多挪 8 cm。手腕朝向不动。
- **只能挪手**：命令里只有手的路点，肘和肩由控制器的全身逆解决定，所以直接受控的只有手掌那几个球。
- **对比**：新规划的 63 条真实路线，每条 4 遍，随机种子和第二轮直接发命令那次相同，所以每一次回放的起始状态逐位相同，是成对比较。主要看其中 21 条长绕行路线，共 84 对。
- **指标**：实际扫过的路线到想要的路线的扫过区域误差，手掌 3 个球和整条手臂 24 个球各算一个。

### 核对

补偿那一组里，在线算出的"原命令的参考路线"要和第二轮离线表里同一次回放的参考路线一致；在线对原命令的预测要和离线的预测一致。这两条不满足就说明在线的特征和训练时不一样，结果不算数。

### 基线（已知，来自第二轮直接发命令的数据）

21 条长绕行路线上，直接发命令时：手掌 4.83 cm，整条手臂 8.10 cm。模型对手掌路线的预测残差中位数 0.68 cm。

### 预测

- 手掌：降 30–55 %。理由：模型把手掌偏差从 2.28 cm 预测到只剩 0.68 cm，但挪路点会改变运动本身，迭代两次未必收敛到底。
- 整条手臂：降 10–35 %。理由：肘和肩不受路点直接控制。
- 模型对补偿后命令的预测残差不超过直接发命令时的 2 倍：挪动最多 8 cm，命令仍在训练分布之内。

### 判据

- 手掌降幅 ≥ 20 %，按路线自助抽样的 95 % 区间不含 0：挪路点这个用法成立。
- 在此基础上整条手臂的降幅区间也不含 0：进入窄缝测试的设计；整条手臂的区间含 0：只能主张"手的路线可以修正"，窄缝测试只对手成立，向用户报告。
- 手掌降幅 < 20 % 或区间含 0：照预测挪路点不管用，向用户报告，不进窄缝测试。
- 补偿那一组摔倒或规划逆解失败明显多于直接组（多出 5 次以上）：按失败处理并报告。

### 第三步的结果（2026-11-22）

补偿回放：2026-11-22 02:50–02:53 UTC，vastgpu，2 个进程，63 条新路线各 4 遍，种子 41000000（和直接发命令的 `data_via_fresh` 相同），数据在 `outputs/reachwm_20261030/data_via_comp`，没有跳过的开场，两组都没有摔倒。当时的代码没提交，原样存在 `data_via_comp/asrun/`（commit d39eedfd88edfc778415dbedbb0f4d662c855077 + uncommitte…）。正式运行之前在其中 8 次回放上试过一次流程，那 8 对的结果我看到过。

成对比较（`via_comp_eval.py`）：

```text
pairs 252 (direct 252, compensated 252 paths; falls: direct 0, compensated 0 of 252)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 21 routes, 84 pairs
  palm, error against the wanted sweep: direct 4.83 cm, compensated 2.99 cm; drop 0.38 (0.30 to 0.48); improved in 0.95 of the pairs
  arm,  error against the wanted sweep: direct 8.10 cm, compensated 6.08 cm; drop 0.25 (0.16 to 0.33); improved in 0.88 of the pairs
  palm's settled end against the wanted end: direct 3.01 cm, compensated 0.77 cm; drop 0.74 (0.69 to 0.78)
  points moved: median 2.34 cm, largest 8.00 cm; the goal 2.53 cm
  the model's own error (arm) on the run it predicted: direct 3.14 cm, compensated 3.20 cm
FreshBox: 12 routes, 48 pairs
  palm, error against the wanted sweep: direct 4.85 cm, compensated 2.78 cm; drop 0.43 (0.32 to 0.51); improved in 0.92 of the pairs
  arm,  error against the wanted sweep: direct 8.23 cm, compensated 5.88 cm; drop 0.29 (0.20 to 0.37); improved in 0.88 of the pairs
  palm's settled end against the wanted end: direct 3.36 cm, compensated 0.82 cm; drop 0.76 (0.67 to 0.80)
  points moved: median 2.48 cm, largest 7.89 cm; the goal 2.89 cm
  the model's own error (arm) on the run it predicted: direct 3.79 cm, compensated 3.45 cm
FreshEdge: 9 routes, 36 pairs
  palm, error against the wanted sweep: direct 4.71 cm, compensated 3.25 cm; drop 0.31 (0.23 to 0.52); improved in 1.00 of the pairs
  arm,  error against the wanted sweep: direct 7.63 cm, compensated 6.72 cm; drop 0.12 (-0.07 to 0.48); improved in 0.89 of the pairs
  palm's settled end against the wanted end: direct 2.67 cm, compensated 0.74 cm; drop 0.72 (0.65 to 0.78)
  points moved: median 2.15 cm, largest 8.00 cm; the goal 2.39 cm
  the model's own error (arm) on the run it predicted: direct 2.57 cm, compensated 2.81 cm
FreshUnder: 42 routes, 168 pairs
  palm, error against the wanted sweep: direct 2.26 cm, compensated 1.91 cm; drop 0.16 (0.10 to 0.25); improved in 0.73 of the pairs
  arm,  error against the wanted sweep: direct 3.40 cm, compensated 3.34 cm; drop 0.02 (-0.06 to 0.09); improved in 0.45 of the pairs
  palm's settled end against the wanted end: direct 1.71 cm, compensated 0.37 cm; drop 0.78 (0.75 to 0.81)
  points moved: median 1.27 cm, largest 8.00 cm; the goal 1.48 cm
  the model's own error (arm) on the run it predicted: direct 0.75 cm, compensated 0.74 cm
all: 63 routes, 252 pairs
  palm, error against the wanted sweep: direct 2.81 cm, compensated 2.14 cm; drop 0.24 (0.15 to 0.30); improved in 0.81 of the pairs
  arm,  error against the wanted sweep: direct 4.55 cm, compensated 4.05 cm; drop 0.11 (0.02 to 0.21); improved in 0.59 of the pairs
  palm's settled end against the wanted end: direct 2.15 cm, compensated 0.45 cm; drop 0.79 (0.77 to 0.81)
  points moved: median 1.69 cm, largest 8.00 cm; the goal 1.74 cm
  the model's own error (arm) on the run it predicted: direct 0.90 cm, compensated 0.88 cm
first pass (from the stand): 63 routes, 63 pairs
  palm, error against the wanted sweep: direct 1.75 cm, compensated 1.56 cm; drop 0.11 (0.04 to 0.45); improved in 0.87 of the pairs
  arm,  error against the wanted sweep: direct 4.60 cm, compensated 4.19 cm; drop 0.09 (-0.11 to 0.19); improved in 0.40 of the pairs
  palm's settled end against the wanted end: direct 1.89 cm, compensated 0.40 cm; drop 0.79 (0.72 to 0.83)
  points moved: median 1.67 cm, largest 8.00 cm; the goal 1.56 cm
  the model's own error (arm) on the run it predicted: direct 0.76 cm, compensated 0.76 cm
```

## 第四步：窄缝测试（2026-11-22 写的设计，等用户确认；还没有跑任何东西）

前三步的结论：模型能预测整条手臂实际扫过哪里；按预测挪路点能修正手，修正不了肩、肘、腕。窄缝测试要回答的是：这些能力合起来，能让机器人安全穿过多窄的缝，比别的做法窄多少。

分两步做。第一步用已有的空地回放数据，预测每种做法能过多窄的缝；第二步在有真障碍物的仿真场景里验证这个预测。原方案里就写了"转折点由空地数据事先预测"，第一步就是这个预测。

### 第一步：用空地数据预测能过多窄的缝

**缝的定义。**缝的半宽记作 c。规划器想让每个球走的路线两侧 c 以外就是障碍。一次伸手里，只要"要穿过缝的那部分"有任何一个球离开想要的路线超过 c，就算碰到。

- 要穿过缝的部分：手指、手掌、腕，共 17 个球。腕算进去是因为手穿过去的时候前臂末端也在缝里，而第三步表明腕修正不了，这样定义对我们更不利，但更诚实。
- 整条手臂 24 个球的版本一并报告。

**五种做法。**每种都在下命令之前决定走还是不走，走的话发什么命令：

| 做法 | 走的条件 | 发的命令 |
|---|---|---|
| 小固定余量 | c ≥ 3 cm | 原命令 |
| 标定过的固定余量 | c ≥ 一个固定值，取随机路线校准集上直接发命令时误差的 95 % 分位 | 原命令 |
| 学出来的标量半径 | c ≥ 这次伸手预测的半径（同样输入的小模型，校准到 95 %） | 原命令 |
| 常数修正 | c ≥ 它自己的固定余量，同样取校准集上的 95 % 分位 | 路点按每只手一条平均偏差反向挪 |
| 我们的 | 模型预测的扫过路线离想要的路线不超过 c 减去这次伸手的剩余半径 | 路点按逐点模型的预测反向挪 |

**计分。**c 从 2 cm 到 15 cm 每 1 cm 扫一遍。每次伸手三种结局：走了没碰到是成功，走了碰到是碰撞，没走是放弃。

- 主指标"能安全通过的最窄的缝"：成功不少于 90 %、碰撞不超过 5 % 的最小 c。
- 同时画成功率和碰撞率随 c 变化的曲线。

**数据。**21 条长绕行路线各 4 遍，84 次伸手一组。

- 已有：直接发命令（`data_via_fresh`）和按逐点模型挪路点（`data_via_comp`）两组。
- 要补跑：常数修正那一组回放，约 4 分钟。
- 要补训：只看这 17 个球的半径模型，用现有的训练表，不需要新数据。
- 所有半径和余量都只在随机路线的校准集上定，真实路线不参与。

**预测。**写这些数的时候我已经知道：长绕行路线上手掌的误差中位数，直接发是 4.83 cm，挪路点后是 2.99 cm；整条手臂是 8.10 和 6.08 cm。腕的数还没看过。

- 能安全通过的最窄的缝：小固定余量 8–11 cm（它从不放弃，要等到 95 % 的伸手本来就不碰才算安全）；标定过的固定余量 11–15 cm；标量半径 8–12 cm；常数修正和固定余量差不多；我们的 5–8 cm。
- 我们的做法比最好的对手窄 15–40 %。区间放得宽，是因为腕修正不了，它可能把我们的数拖到和对手接近。
- 在 c 小于 5 cm 时，小固定余量的碰撞率超过 50 %，我们的不超过 10 %，因为预测到会碰就放弃了。

**判据。**

- 我们能安全通过的最窄的缝不超过最好对手的 0.8 倍：成立，做第二步。
- 超过 0.95 倍：在"腕也要过缝"的定义下没有优势。这时单看手指和手掌 15 个球的版本，如果那里不超过 0.8 倍，结论限定为"手过缝有优势、前臂没有"，向用户报告；否则这条线的窄缝主张不成立。
- 两者之间：报告给用户决定。
- 任何一种做法在 c ≥ 12 cm 时成功率仍到不了 90 %，说明计分或校准有问题，先查再下结论。

### 第二步：有真障碍物的场景（草案，判据在跑之前补全）

（2026-11-23：完整的设计在后面"第四步第二部分"一节，取代这份草案；这份草案保留原样。）

- **场景**：桌沿上方加一道横杆，手要从桌面和横杆之间穿过去够后面的目标，缝宽就是两者之间的高度，从宽到窄扫。沿用已经跑通的规划和回放流程，障碍物作为固定的几何体加进仿真，记录手臂和它们的接触。
- **每种做法各自重新规划**：固定余量和标量半径把障碍加厚后再规划，缝太窄时规划器会找不到路线；我们的做法用小余量规划，再检查和挪路点。
- **要先核对的**：锁腰只动右臂时这类场景规划得出来的比例。第二轮里越过桌沿只有 9 / 50。比例太低就换场景。
- **第一步的预测在这里检验**：每种做法成功率掉下来的缝宽，应当和第一步给出的数一致。

### 这个设计没有覆盖的

- 肩和肘。它们不进缝，但在真实场景里可能碰到别的东西。第一步里整条手臂的版本会反映这一点，第二步的场景里它们在空处。
- 蹲下的伸手。真实路线全是站立的，蹲下时偏差大约是站立的两倍，是演示想用的情况，但现在没有蹲下的真实路线。
- 物体被碰到之后的后果。这里只算碰没碰到。

### 第四步第一部分的结果（2026-11-22）

常数修正那一组回放：2026-11-22 05:01–05:05 UTC，63 条新路线各 4 遍，种子 41000000，数据在 `outputs/reachwm_20261030/data_via_const`，没有跳过的开场，没有摔倒。它和直接发命令那一组的成对比较（`via_comp_eval.py`，前 10 行）：

```text
pairs 252 (direct 252, compensated 252 paths; falls: direct 0, compensated 0 of 252)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 21 routes, 84 pairs
  palm, error against the wanted sweep: direct 4.83 cm, compensated 5.57 cm; drop -0.15 (-0.26 to -0.06); improved in 0.10 of the pairs
  arm,  error against the wanted sweep: direct 8.10 cm, compensated 8.88 cm; drop -0.10 (-0.16 to -0.00); improved in 0.08 of the pairs
  palm's settled end against the wanted end: direct 3.01 cm, compensated 4.08 cm; drop -0.35 (-0.47 to -0.31)
  points moved: median 0.97 cm, largest 2.56 cm; the goal 1.42 cm
  the model's own error (arm) on the run it predicted: direct 3.14 cm, compensated 3.01 cm
```

计分（`via_gap.py`，输出在 `outputs/reachwm_20261030/via_gap_20261122`；训练半径模型的逐行日志略去，在 `gap.log` 里）：

```text
spheres: through 17, hand 15, arm 24
random paths: fit on 14964, early stopping on 1028, 95 % levels on 1077
through: fixed margin 19.39 cm (direct), 19.01 cm (after the mean offset), 12.44 cm (after the point model); radius scales 1.49 / 1.82
hand: fixed margin 19.39 cm (direct), 19.01 cm (after the mean offset), 12.44 cm (after the point model); radius scales 1.52 / 1.86
arm: fixed margin 19.85 cm (direct), 19.28 cm (after the mean offset), 12.46 cm (after the point model); radius scales 1.43 / 1.88
replays: 252 triples of 252; wanted sweep differs between the runs by at most 0.00 mm

detour routes, spheres 'through': 21 routes, 84 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 8.06 / 11.51 / 12.35; const 8.88 / 11.92 / 13.11; ours 5.73 / 8.70 / 9.26
  half-width cm          2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small   success    0.00  0.00  0.00  0.10  0.15  0.26  0.49  0.62  0.77  0.86  0.93  0.96  0.98  0.99
          collide    0.00  0.00  1.00  0.90  0.85  0.74  0.51  0.38  0.23  0.14  0.07  0.04  0.02  0.01
  fixed   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius  success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.06  0.14  0.18  0.27  0.39  0.50
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  ours    success    0.00  0.00  0.00  0.00  0.06  0.20  0.43  0.69  0.79  0.88  0.93  0.94  0.94  0.95
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small    12.50  (11.00 to 14.50)
    fixed      nan  (nan to nan)
    radius     nan  (nan to nan)
    const      nan  (nan to nan)
    ours     11.25  (9.50 to 15.00)
  ours / best other way (small): 0.90 (0.79 to 1.18)
  at a half-width of 5 cm, success / collision / abstention: small 0.10 / 0.90 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.00 / 0.00 / 1.00; const 0.00 / 0.00 / 1.00; ours 0.00 / 0.00 / 1.00

detour routes, spheres 'hand': 21 routes, 84 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 8.06 / 11.51 / 12.35; const 8.88 / 11.92 / 13.11; ours 5.61 / 8.37 / 9.26
  half-width cm          2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small   success    0.00  0.00  0.02  0.10  0.15  0.26  0.49  0.62  0.77  0.86  0.93  0.96  0.98  0.99
          collide    0.00  0.00  0.98  0.90  0.85  0.74  0.51  0.38  0.23  0.14  0.07  0.04  0.02  0.01
  fixed   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius  success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.04  0.07  0.13  0.18  0.24  0.36
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  ours    success    0.00  0.00  0.00  0.00  0.06  0.24  0.49  0.74  0.85  0.89  0.93  0.94  0.94  0.95
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small    12.50  (11.00 to 14.50)
    fixed      nan  (nan to nan)
    radius     nan  (nan to nan)
    const      nan  (nan to nan)
    ours     11.25  (9.25 to 14.75)
  ours / best other way (small): 0.90 (0.78 to 1.16)
  at a half-width of 5 cm, success / collision / abstention: small 0.10 / 0.90 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.00 / 0.00 / 1.00; const 0.00 / 0.00 / 1.00; ours 0.00 / 0.00 / 1.00

detour routes, spheres 'arm': 21 routes, 84 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 8.10 / 11.60 / 12.64; const 8.88 / 12.53 / 13.59; ours 6.08 / 9.35 / 11.54
  half-width cm          2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small   success    0.00  0.00  0.00  0.05  0.12  0.25  0.48  0.62  0.77  0.85  0.92  0.95  0.96  0.98
          collide    0.00  0.00  1.00  0.95  0.88  0.75  0.52  0.38  0.23  0.15  0.08  0.05  0.04  0.02
  fixed   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius  success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.06  0.13  0.15  0.32  0.48  0.67  0.75
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  ours    success    0.00  0.00  0.00  0.00  0.04  0.13  0.23  0.38  0.55  0.65  0.77  0.83  0.85  0.88
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small    12.75  (11.00 to 14.50)
    fixed      nan  (nan to nan)
    radius     nan  (14.50 to 15.00)
    const      nan  (nan to nan)
    ours       nan  (11.75 to 15.00)
  ours / best other way (small): nan (0.98 to 1.31)
  at a half-width of 5 cm, success / collision / abstention: small 0.05 / 0.95 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.00 / 0.00 / 1.00; const 0.00 / 0.00 / 1.00; ours 0.00 / 0.00 / 1.00

near-straight routes (under a board), spheres 'through': 42 routes, 168 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 2.79 / 4.34 / 5.17; const 3.53 / 5.59 / 6.40; ours 2.51 / 3.23 / 3.58
  half-width cm          2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small   success    0.00  0.00  0.85  0.93  0.98  0.99  0.99  0.99  0.99  0.99  0.99  0.99  0.99  0.99
          collide    0.00  0.00  0.15  0.07  0.02  0.01  0.01  0.01  0.01  0.01  0.01  0.01  0.01  0.01
  fixed   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius  success    0.00  0.00  0.15  0.44  0.74  0.91  0.97  0.99  0.99  0.99  0.99  0.99  0.99  0.99
          collide    0.00  0.00  0.00  0.01  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  ours    success    0.00  0.08  0.70  0.95  0.99  0.99  0.99  0.99  0.99  0.99  0.99  0.99  0.99  0.99
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small     5.25  (4.25 to 5.75)
    fixed      nan  (nan to nan)
    radius    7.00  (6.50 to 7.75)
    const      nan  (nan to nan)
    ours      4.75  (4.50 to 5.25)
  ours / best other way (small): 0.90 (0.86 to 1.06)
  at a half-width of 5 cm, success / collision / abstention: small 0.93 / 0.07 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.44 / 0.01 / 0.55; const 0.00 / 0.00 / 1.00; ours 0.95 / 0.00 / 0.05

all routes, spheres 'through': 63 routes, 252 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 3.54 / 9.54 / 10.94; const 4.31 / 10.09 / 11.45; ours 2.87 / 6.33 / 8.28
  half-width cm          2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small   success    0.00  0.00  0.57  0.65  0.71  0.75  0.83  0.87  0.92  0.95  0.97  0.98  0.99  0.99
          collide    0.00  0.00  0.43  0.35  0.29  0.25  0.17  0.13  0.08  0.05  0.03  0.02  0.01  0.01
  fixed   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius  success    0.00  0.00  0.10  0.29  0.49  0.61  0.65  0.67  0.68  0.71  0.72  0.75  0.79  0.83
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  ours    success    0.00  0.06  0.46  0.63  0.68  0.73  0.81  0.89  0.92  0.96  0.97  0.98  0.98  0.98
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small    11.25  (9.50 to 12.75)
    fixed      nan  (nan to nan)
    radius     nan  (14.00 to 14.75)
    const      nan  (nan to nan)
    ours      9.25  (8.25 to 10.75)
  ours / best other way (small): 0.82 (0.77 to 0.93)
  at a half-width of 5 cm, success / collision / abstention: small 0.65 / 0.35 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.29 / 0.00 / 0.70; const 0.00 / 0.00 / 1.00; ours 0.63 / 0.00 / 0.37
```

## 第四步补充：手腕朝向也补偿（2026-11-22，用户："做补偿 提交并推送"；本节在这一步的任何运行之前写下）

第一部分里"能穿过明显更窄的缝"不成立，剩下的误差主要是手的朝向：补偿只挪了路点的位置，手的朝向误差 16° 没有动，指尖因此比手掌多偏约 3 cm。

### 做法

- 手掌和手指在一次伸手里是一个刚体，共 15 个球。在每个路点经过的时刻和终点，用这 15 个球求出"想要的手"到"预测的手"之间差的那个转动，把这个路点（或终点）的手腕命令反向转这么多；位置照旧按手掌的预测偏差反向挪。
- 迭代 4 次，每个点的朝向最多修正 60°，位置最多 8 cm。迭代次数比只挪位置时多，因为以前量到控制器只跟上手腕转动的一半左右，一次修不到位。
- 开关 `REACH_VIA_COMP_MODE=pose`。只挪位置的模式不变，第三步的结果可以原样重现。
- 对比方式、路线、随机种子和第三步相同：63 条新路线各 4 遍，和直接发命令的那一组成对比较；然后用同一个计分脚本重算窄缝测试，"我们的"换成这一组。

### 写预测之前已经知道的

- 模型对朝向偏差预测得准：21 条长绕行路线上，手的朝向偏离参考在终点是 19.3°、途中 11.7°（用 15 个球算；复核用另一种算法是 16.5°），减去模型的预测后剩 2.9° 和 2.1°。这是在直接发命令的数据上量的。
- 只挪位置时：手指、手掌、腕的误差中位数 8.06 → 5.73 cm；能安全通过的最窄的缝 11.25 cm，小固定余量 12.5 cm。

### 预测

- 手的朝向误差（15 个球，中位数）：终点 19.3° → 5–10°，途中 11.7° → 4–8°。
- 手指、手掌、腕的误差中位数：3.5–5.0 cm。
- 能安全通过的最窄的缝：7.5–10 cm；和小固定余量的比值 0.6–0.8。
- 模型对这一组命令的预测残差不超过直接发命令时的 2 倍。这一条比位置补偿时更没把握：手腕命令要多转最多 60°，可能走到模型见得少的地方。

### 判据

沿用第一部分的，不放宽。这是看过第一次结果之后针对同一条判据的第二次尝试。

- 和最好对手的比值 ≤ 0.8：成立，做第二部分（真障碍物的场景）。
- > 0.95：没有优势。这条线能主张的就是"预测加放行让机器人几乎不碰撞"，不主张"穿过更窄的缝"。
- 两者之间：报告给用户决定。
- 摔倒或规划逆解失败比直接组多出 5 次以上，或者模型的预测残差超过 2 倍：按失败处理并报告。
- 三项在线核对仍须是 0.00 mm。

## 第五步：同一套做法换控制器（2026-11-22，用户："我现在需要在多个 baseline 上跑……sonic，amo 等 wbc 控制器都用来跑"；范围："1. 先 sonic 再 amo 2. 可以 3. 都在这台机器上采 4. 可以并行"）

方法只要求控制器是冻结的，不依赖是哪一个。这一步检验：换一个控制器，偏差是不是照样存在，同一个模型结构是不是照样学得会，挪路点是不是照样管用。

### 做法

每个控制器走和 ScaleBFM 相同的几步，脚本和设置不改，只换 `REACH_CONTROLLER`：

1. 采集多路点伸手数据，12000 次开场，30 % 不走路直接出发，50 % 长路线。
2. 训练逐点模型，默认设置，不调。
3. 评测：随机路线留出集；同样那 63 条新规划的真实路线各回放 4 遍。
4. 挪路点补偿的成对比较（只挪位置的版本，和 ScaleBFM 的第三步对应）。

- **SONIC**：采集脚本里现成，用它的默认设置（遥操作的三点接口，腿由它自己的速度规划器走）。这台机器的 onnxruntime 没有 GPU 版，网络放在 CPU 上，每次开场约 14 s。
- **AMO**：有接口相同的动作层，要先接进采集脚本并核对多路点伸手。
- SONIC 的三点接口只把手腕和头的目标发给控制器，肘和肩由 SONIC 自己定。所以"参考规划扫过的路线"里肘和肩那部分，对 SONIC 来说本来就不是它被要求跟踪的东西，"照参考走"这个基线在肘和肩上会比 ScaleBFM 时更差。手的部分两者可比。

### 写预测之前已经知道的

SONIC 的 4 次冒烟伸手：0 摔倒，手的终点误差 6.8–15.0 cm，比 ScaleBFM 的典型值大。

### 预测（SONIC）

- 直接发命令时手偏离参考路线比 ScaleBFM 大：站立路线的中位数 5–10 cm（ScaleBFM 3.7 cm）。
- 逐点模型在随机路线留出集上的降幅（整条手臂的扫过区域误差）：40–70 %。
- 新规划的 21 条长绕行真实路线上的降幅：30–60 %。
- 挪路点补偿后手掌到想要路线的误差：降 20–50 %。

### 判据（每个控制器各自判）

- 和 ScaleBFM 第二轮相同的停止线：验证集里站立、手走过 1 米以上的路线，降幅到不了 25 % → 这个控制器上不成立，向用户报告，不看测试集。
- 过了停止线：新的真实路线上降幅 ≥ 20 % 且按路线自助抽样的区间不含 0 → 模型在这个控制器上成立。
- 挪路点补偿：手掌降幅 ≥ 20 % 且区间不含 0 → 补偿在这个控制器上成立。
- 三个控制器的结果并排报告。任何一个不成立都照实报告，不换设置重试；要重试先问用户。

## 第四步补充之二：补偿器自己的回放进训练（2026-11-23，用户："1. 要"）

### 为什么

补偿是在模型上做优化：它找的是"模型说手会落在想要的地方"的命令，这种命令会跑到训练数据没盖住的地方，模型在那里是错的。随机路线盖不住（通用补采、随机路线上的补偿回放都试过，见"预测修订"）。把补偿器实际发出的命令和它们的真实结果加进训练，是这类问题的标准做法。

### 做法

1. **训练用的规划器路线**：换种子（20261123）让 cuRobo 另外规划，场景生成器和测试集相同，和开发路线（种子 20261122）、测试路线（种子 20261121）没有共同场景。用全部长绕行路线，加 150 条伸到板下的路线。
2. **第 1 轮**：在训练路线上各回放 4 遍：直接发命令、只挪位置、位置加朝向（后两种用三份随机数据训练的模型 `via_point_20261122`）。这三组回放加进训练，得到第 1 轮模型。
3. **第 2、3 轮**：用上一轮的模型在训练路线上再跑位置加朝向，加进训练。因为模型一变，补偿器发的命令也跟着变。
4. **每轮之后在 41 条开发路线上检查**（用这一轮的模型重跑位置加朝向）。腕 + 掌 + 指的中位数比上一轮改善不到 5 % 就停，最多 3 轮。
5. **补偿的设置固定**：每轮走剩余修正的一半，6 轮，手腕命令最多转 40°，转目标的朝向，不带信任规则。开发集上通过检查就不再调。
6. **测试集只跑一次**：63 条 × 4 遍，位置加朝向，用最后一轮的模型；然后重算窄缝。

### 预测

- 开发路线上，最后一轮模型对自己那一遍补偿回放的预测误差：3–4 cm（现在 6.87 cm，直接发命令 2.8 cm）。
- 开发路线上腕 + 掌 + 指的中位数：3.5–4.5 cm（直接 7.91，只挪位置 5.56）；手掌不比只挪位置差（≤ 2.8 cm）；终点朝向 6–10°（直接 20.0°），路上 4–6°。
- 测试集（21 条绕行 84 对）：腕 + 掌 + 指的中位数 3.5–5.0 cm，能过的最窄缝 7.5–10 cm，和小固定余量的比值 0.6–0.8。这三项和"第四步补充"里事先写的相同。

### 判据

- 开发路线上的检查（不过就不上测试集，向用户报告）：模型对补偿回放的预测误差不超过直接发命令那一遍的 2 倍；腕 + 掌 + 指的中位数比只挪位置低，且按路线自助抽样的区间不含 0。
- 测试集上的判据不变：比值 ≤ 0.8 成立，> 0.95 没有优势，中间向用户报告；摔倒和 IK 失败不比直接发命令多 5 次以上；三项一致性检查 0.00 mm。
- 被试跑看过的 8 条测试路线单独再算一遍（剩下 13 条）。
- 报告里写明：补偿用的模型见过同三类场景的规划器路线，只做预测的那个结果用的模型没见过。

## 第四步补充之三：换出发规则，在新测试路线上重做窄缝计分（2026-11-23，用户："1 做"）

### 为什么

第四步补充之二的测试里，腕、掌、指的误差降了 58 %，窄缝比值却是 1.02。原因在我们这一方的出发规则：它把"预测的误差"和"残差的 95 % 上界"相加，实际误差从来没有超过这个和，碰撞率处处为 0，成功率到 12.75 cm 才够 90 %。事后试的另一条规则在同一批路线上是 9.00 对 12.50 cm，但规则是看到结果之后选的，不算证据。这一步把它事先写下来，在没用过的路线上只跑一次。

### 做法

- **新的测试路线**：换种子（20261124）让 cuRobo 规划 450 个场景，场景生成器和以前相同。和训练路线（20261123）、开发路线（20261122）、原测试路线（20261121）都不共用场景。规划出来的路线全部用，各回放 4 遍。
- **三组回放，同样的路线同样的起点**：直接发命令；常数修正（对手，用 `via_point_20261121`）；位置加朝向（我们，用第 2 轮模型 `via_point_onpol2_20261123`，设置不变）。
- **我们这一方新的出发规则**：预测的误差 + m ≤ 缝的半宽就出发，否则放弃。预测的误差是"模型对实际发出的命令预测的扫过区域"离想要的路线多远。m 是一个固定数，取 41 条开发路线那一遍补偿回放（`onpol_dev_r2_pose`）上"实际误差 − 预测误差"的 95 % 分位，对腕 + 掌 + 指是 3.28 cm。`via_gap.py` 加选项 `--rule margin`，由脚本自己从开发路线那一遍算出 m，不手填。
- **对手**：和以前相同的四个（小固定余量、标定过的固定余量、标量半径、常数修正，余量取自只见过随机路线的模型）。另外加一个更公平的对手：**在开发路线上标定的固定余量**，直接发命令，缝的半宽不小于开发路线上直接发命令误差的 95 % 分位才出发。它和我们一样事先判断、用同一批开发路线，不像小固定余量那样事后从测试集读数。
- **计分不变**：成功不少于 90 %、碰撞不超过 5 % 的最窄半宽；比值 = 我们 ÷ 对手里最好的一个；主要看长绕行路线、腕 + 掌 + 指这一组球；区间按路线自助抽样。
- 原来那条规则（两项相加）在新路线上的结果顺带打印，只作对照。
- 模型、补偿设置、m、对手的余量都已经定死，新测试路线不用来定任何数。只跑一次。

### 写预测之前已经知道的

原测试集上的事后探索：新规则 9.00 cm，小固定余量 12.50 cm，比值 0.72（区间 0.64–1.20）；没被看过的 13 条路线上 8.25 对 11.25 cm，比值 0.73（0.68–0.80）。8 cm 的缝成功 83 %、碰撞 1 %。

### 预测

- 新路线上长绕行路线的误差和以前接近：直接发命令腕 + 掌 + 指中位数 7.5–8.5 cm，补偿后 3.2–4.0 cm。
- 我们能过的最窄半宽 8–10 cm；小固定余量 11–13 cm；在开发路线上标定的固定余量 11.5–13.5 cm。
- 比值 0.65–0.85。
- 我们在自己最窄的那个半宽上碰撞率不超过 5 %（规则的余量是按 95 % 定的，这一条不满足就说明余量在新路线上不够）。

### 判据

- 比值 ≤ 0.8 且区间上端 < 1：成立，"能安全通过更窄的缝"。
- 比值 > 0.95，或区间含 1 且比值 > 0.8：不成立。
- 比值在 0.8 和 0.95 之间且区间上端 < 1：有优势但不大，照实报告数值。
- 另外三条和以前相同：三项一致性检查 0.00 mm；摔倒不比直接发命令多 5 次以上；模型对补偿回放的预测误差不超过直接发命令那一遍的 2 倍。任何一条不满足，结果不算。
- 整条手臂 24 个球那一组照样打印。那一组预计不成立（补偿朝向把肩带偏），不作为这一步的判据。

### 第五步，AMO：先把它调到能用，再写预测（2026-11-23）

**AMO 原来不能用的原因是我们发给它的命令，不是它够不着。**

- 现象：在这个场景上，带路点的伸手有 47 % 会自己走开（200 次里 93 次，骨盆离开 20 cm 以上），这时发给它的速度命令是 0。
- 走了的弯路：我先说是目标太远，用户据此选了"把目标拉近"。实测相反，越近越高的目标越容易走开，这个说法不成立。之后逐项排除了质量分布、躯干后仰命令、每拍手臂逆解、手点位置、髋力矩上限、求解器设置、脚的接触形状，每项 40 次开场，都没有用。
- 线索：在 AMO 原来的机器人上各跑满 40 次，站着伸手走开 44–63 %，蹲着 4 %，和手点无关。
- **原因一：站得比标称高。** 动作层把参考规划里的骨盆高度（站立时 0.78–0.79 m）当作 AMO 的高度命令，AMO 自己的标称命令是 0.75 m。封顶在 0.75 m 之后，同样 200 次开场走开的降到 40 次（20 %）。
- **原因二：复位后的站姿。** 复位把关节全置零，AMO 从这个姿势站住就开始伸手，走开 24 / 60；走过一段再伸手是 17 / 130。复位后先原地踏 3 秒再取站姿，同样 200 次开场走开的降到 25 次（12 %）。
- 两处修复都是动作层的可选设置（`AmoConfig.height_max`、`step_in_s`），默认关，场景任务不受影响；自身模型的采集里默认开（`REACH_AMO_HEIGHT_MAX=0.75`、`REACH_AMO_STEP_IN=3`）。探测数据：`amo_envelope_20261123`（修复前）、`amo_envelope_fixed_20261123`（只封顶）、`amo_envelope_fixed2_20261123`（两处都修）。
- **和 SONIC 对照**：按"伸手结束时骨盆离规划位置超过 20 cm"来量，ScaleBFM 是 1.2 %，SONIC 12 %，AMO 修复后 16 %；手的终点误差中位数 SONIC 7.9 cm，AMO 8.0 cm。SONIC 的数据是原样用的，AMO 也原样用，只去掉摔倒的。

**做法**：和 SONIC 相同的四步，脚本和设置不改，`REACH_CONTROLLER=amo`。采集 12000 次开场，种子 55000000。

**AMO 和另外两个控制器不同的地方**：AMO 只管腿和腰。手臂由我们的关节 PD 跟踪，目标来自每拍一次的手臂逆解，逆解用的是量到的骨盆和腰。所以身体怎么动，手臂会自己追着补，手的误差主要来自腿部里程计的漂移、PD 的滞后和够不着。

**预测**：

- 站立路线手偏离参考路线的中位数 4–8 cm。
- 验证集停止线（站立、手走过 1 米以上）降幅 25–55 %。
- 留出的随机路线降幅 25–55 %；21 条测试绕行路线降幅 20–50 %。
- 挪路点补偿后手掌误差降 15–45 %。
- 走开的那一成多（骨盆移动几十厘米）模型大概预测不了，它们会把误差的高分位拉得很大；中位数的降幅受影响小。

**判据**：和 SONIC 相同（停止线 25 %；测试路线上降幅 ≥ 20 % 且区间不含 0；补偿后手掌降幅 ≥ 20 % 且区间不含 0）。不成立就照实报告，不换设置重试。

## 第五步补充：SONIC 上的手腕朝向补偿（2026-11-23，4090；用户："照 ScaleBFM 的'第四步补充之二'原样做一遍"；本节在这一步的任何运行之前写下）

在 4090 上做，进度记在 `docs/handoff/selfmodel-4090-20261123.md`。

### 做法

步骤和"第四步补充之二"相同，只换控制器（`REACH_CONTROLLER=sonic`）和机器。

1. **路线和种子不变：**
   - 训练路线 `train_routes/subset`：340 条 × 4 遍，种子 49000000；
   - 开发路线 `dev_routes/detour`：41 条 × 4 遍，种子 47000000；
   - 测试路线 `fresh_routes`：63 条 × 4 遍，种子 41000000。
2. **第 0 轮模型**是 `via_point_sonic_20261123`，只见过 SONIC 的 12000 次随机路线（`data_via_sonic`）。
3. **第 1 轮：**在训练路线上回放三组，加进训练，得到第 1 轮模型。
   - 三组是直接发命令、只挪位置、位置加朝向；后两组用第 0 轮模型做补偿。
   - 训练数据是 `data_via_sonic` 加这三组，网络和训练设置照默认。
4. **第 2、3 轮：**用上一轮的模型在训练路线上再跑位置加朝向，加进训练。
5. **每轮之后在开发路线上检查：**用这一轮的模型跑位置加朝向。
   - 腕 + 掌 + 指的中位数比上一轮改善不到 5 % 就停，最多 3 轮。
   - 开发路线上另跑直接发命令和只挪位置（第 0 轮模型）各一遍，作为配对的对照。
6. **补偿设置固定**为 `via_comp.py` 的默认值，不调：每轮走剩余修正的一半，6 轮，手腕命令最多转 40°，转目标的朝向，不带信任规则。
7. **测试集只跑一次：**63 × 4，位置加朝向，用最后一轮的模型。然后重算窄缝：
   - 窄缝计分要一组常数修正的回放。SONIC 还没有这一组，在测试路线上用第 0 轮模型跑一次（`REACH_VIA_COMP_MODE=const`）。
   - `--base` 用第 0 轮模型：四个对手的余量取自它的训练侧数据。
   - `--cal` 用最后一轮模型在开发路线上的那一遍：我们这一方的余量缩放在这里定。
   - `--seen 0,1,2,3,12,13,14,15`，和 ScaleBFM 相同。

**和 ScaleBFM 那次不同的地方：**

- **机器和网络：**全部回放都在 4090 上跑，SONIC 的网络放 GPU。
  - 成对比较的两遍都是 4090 上跑的。测试集直接发命令那一遍用 `data_via_fresh_sonic_4090`，它也用来和 vastgpu 对数。
  - vastgpu 上网络在 CPU，那边的 `data_via_fresh_sonic`、`data_via_comp_sonic` 不参与配对。
- **第 0 轮模型的训练数据：**只有一份随机数据。ScaleBFM 有三份，其中第三份专门把目标朝向转开。
- **窄缝计分的格子只到 15 cm：**
  - SONIC 直接发命令时，腕 + 掌 + 指误差的 90 % 分位是 12.44 cm（vastgpu 只挪位置那次的输出），小固定余量可能在 15 cm 以内到不了 90 % 的成功率。
  - 事先定：哪种做法在 15 cm 以内都没过，就把格子延长到 25 cm 再算一次。所有做法一起延长，两次的结果都报告。
- **规则 B 作为次要结果，在 SONIC 的测试之前写下：**
  - 规则：预测的误差加一个固定余量就出发，余量取最后一轮在开发路线上"实际 − 预测"的 95 % 分位。脚本是 `via_gap_20261123/gap_rules_exploratory.py`。
  - 证据强度：这条规则是看过 ScaleBFM 在同一批测试路线上的结果之后想出来的。对 SONIC，余量只由 SONIC 的开发路线定；但规则的形式受这批路线影响，所以只算弱证据。
  - 主判据仍是事先写的规则 A。

### 写预测之前已经知道的

- **21 条测试绕行路线上的 SONIC**（vastgpu，网络在 CPU）：

  | | 直接发命令 | 只挪位置 |
  |---|---|---|
  | 腕 + 掌 + 指 | 8.39 cm | 4.81 cm |
  | 手掌 | 6.40 cm | 3.09 cm |
  | 整条手臂 | 8.55 cm | 5.17 cm |
  | 手掌终点 | 4.54 cm | 1.58 cm |
  | 朝向，终点 / 路上 | 13.9° / 8.2° | 11.3° / 8.6° |

  模型对这两遍的预测误差分别是 4.08 cm 和 3.77 cm。ScaleBFM 直接发命令时是 1.32 cm。
- **SONIC 的朝向误差比 ScaleBFM 小**（终点 13.9° 对 19.3°），能补回来的空间也小。
- **SONIC 只挪位置时整条手臂也跟着改善**（降 40 %），ScaleBFM 没有。SONIC 的接口只收手腕和头的目标，手腕挪了，手臂跟着走。
- **ScaleBFM 上这一步的经过：**
  - 只见过随机路线的模型预测不了转过的命令（开发路线上 2.45 倍）；
  - 两轮补偿器回放进训练之后是 1.80 倍；
  - 腕 + 掌 + 指比只挪位置低 32 %，整条手臂比只挪位置差；
  - 窄缝比值 1.02，按判据没有优势。
- 写这一节时，4090 上的 SONIC 回放一次都还没跑成（第一次缺 SONIC 的权重文件，进程启动就退出了）。

### 预测（SONIC，4090）

降幅都相对 4090 上配对的那一遍直接发命令。

- **训练路线第 1 轮：**模型对位置加朝向那一遍的预测误差，是对直接发命令那一遍的 1.3–2.5 倍。
- **开发路线，最后一轮：**
  - 模型对补偿回放的预测误差，是对直接发命令那一遍的 1.0–1.8 倍；
  - 腕 + 掌 + 指的中位数比直接发命令降 45–60 %，比只挪位置低 5–25 %；
  - 终点朝向比直接发命令降 35–60 %；
  - 手掌比只挪位置差不超过 0.3 cm；
  - 整条手臂比只挪位置差 0–20 %；
  - 停在第 2 轮。
- **测试集（21 条绕行，84 对）：**
  - 腕 + 掌 + 指降 45–60 %，中位数 3.5–4.5 cm；
  - 终点朝向 6–9°；
  - 整条手臂 5.0–6.5 cm。
- **窄缝：**
  - 规则 A 的比值 0.95–1.3，没有优势。理由和 ScaleBFM 相同：残差半径叠在预测误差上太保守。SONIC 的残差更大，会更保守。
  - 规则 B 的比值 0.7–0.95。
- **这一节里把握最小的：**
  - 开发路线上"比只挪位置低、区间不含 0"这一条。SONIC 的朝向误差本来就小，有一半左右的可能过不了。
  - SONIC 模型本身的误差比 ScaleBFM 大三倍，这会让规则 A 更保守。

### 判据

和"第四步补充之二"相同，不放宽。

- **开发路线上的检查。**不过就不上测试集，向用户报告：
  - 模型对补偿回放的预测误差不超过直接发命令那一遍的 2 倍；
  - 腕 + 掌 + 指的中位数比只挪位置低，且按路线自助抽样的区间不含 0。
  - 3 轮之后仍不过，就停下报告，不改设置重试。
- **测试集：**
  - 窄缝比值 ≤ 0.8 成立，> 0.95 没有优势，中间向用户报告；
  - 摔倒和 IK 失败不比直接发命令多 5 次以上；
  - 三项一致性检查是 0.00 mm。
- **另外照第五步的判据判补偿本身：**腕 + 掌 + 指比直接发命令降 ≥ 20 %，且区间不含 0。
- **报告里要有的：**
  - 被试跑看过的 8 条测试路线去掉，单独再算一遍；
  - 补偿用的模型见过同三类场景的规划器路线；
  - 这 63 条测试路线在 SONIC 下已经被看过两次（vastgpu 的直接发命令和只挪位置）。

### 第五步补充：开工核对和第 1 轮（2026-11-23，4090）

**和 vastgpu 对数：对上。**在 4090 上用 SONIC（网络放 GPU）把 63 条测试路线直接回放一遍（`data_via_fresh_sonic_4090`，ID0 8100000），用 `replay_match.py` 和第 0 轮模型计算：

```
detour (box + edge)   4090:    reference  8.17 constant  8.95 direction  3.82 cm | drop 0.53 (0.48 to 0.58)
detour (box + edge)   vastgpu: reference  8.55 constant  8.99 direction  4.08 cm | drop 0.52 (0.45 to 0.59)
all                   4090:    reference  6.34 constant  8.49 direction  3.27 cm | drop 0.48 (0.46 to 0.54)
all                   vastgpu: reference  6.73 constant  8.60 direction  3.34 cm | drop 0.50 (0.47 to 0.53)
252 segments, 15691 steps, fell 0, walked 189, ok 102
```

四个中位数相差 2–6 %，在 10 % 以内，0 摔倒。两台机器第一遍的起点处处差 0.25 cm，走过之后差得更多，所以两台机器之间只比中位数。

**4090 上同一台机器能逐位重复。**开发路线直接发命令用同一种子跑两遍（`dev_direct_sonic_4090` 和 `_dup`）：

```
== A against B, reset by reset: 164 resets in both (A 164, B 164)
  all                    palm's start apart: median 0.000 cm, 95 % 0.000, max 0.000; identical (< 0.001 cm) on 1.00
    reference error A - B: median +0.000 cm, |A - B| median 0.000, 95 % 0.000, max 0.000
```

进程数取 3 的倍数、卡固定 1,2,3，同一次开场在每遍里都落在同一张卡上。

**第 1 轮**（`sonic_pose_4090/r1.sh`，06:33–07:31 UTC；训练路线三组各 1360 段、0 摔倒、0 跳过）。

训练路线上，第 0 轮模型位置加朝向（`onpol_sonic_r1_pose/comp_eval.out`）：

```
pairs 1356 (direct 1360, compensated 1360 paths without a fall; 4 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
detour paths (over an edge, over a box): 190 routes, 757 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 9.16 cm, compensated 5.24 cm; drop 0.43 (0.40 to 0.46); 90 % level 16.22 and 14.88 cm
  hand's attitude against the wanted one: settled end direct 17.6 deg, compensated 11.1 deg; along the path 8.6 and 6.7 deg
  the model's own error (arm) on the run it predicted: direct 3.86 cm, compensated 4.60 cm
all: 340 routes, 1356 pairs
  the model's own error (arm) on the run it predicted: direct 3.39 cm, compensated 3.58 cm
```

同一批路线只挪位置（`onpol_sonic_r1_point`）：腕 + 掌 + 指 5.06 cm，90 % 分位 9.24 cm。

第 1 轮模型 `via_point_sonic_onpol1_20261123`（`data_via_sonic` + 三组回放，16002 条路线）。开发路线位置加朝向（`dev_pose_sonic_r1`，ID0 8800000），对照直接发命令 `dev_direct_sonic_4090` 和只挪位置 `dev_point_sonic_4090`（第 0 轮模型，ID0 8700000）：

```
== dev_pose_sonic_r1/comp_eval.out
pairs 163 (direct 164, compensated 164 paths without a fall; 1 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.04 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.01 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
all: 41 routes, 163 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 9.26 cm, compensated 5.08 cm; drop 0.45 (0.35 to 0.53); 90 % level 15.49 and 14.03 cm
  palm, error against the wanted sweep: direct 7.08 cm, compensated 3.32 cm; drop 0.53 (0.44 to 0.57); improved in 0.96 of the pairs
  arm,  error against the wanted sweep: direct 9.46 cm, compensated 6.22 cm; drop 0.34 (0.26 to 0.43); improved in 0.88 of the pairs
  palm's settled end against the wanted end: direct 5.12 cm, compensated 1.52 cm; drop 0.70 (0.63 to 0.76)
  hand's attitude against the wanted one: settled end direct 15.2 deg, compensated 8.0 deg; along the path 8.5 and 5.3 deg
  points moved: median 3.00 cm, largest 8.00 cm; the goal 4.59 cm; wrist commands turned: median 7.4 deg, largest 40.0 deg
  the model's own error (arm) on the run it predicted: direct 3.28 cm, compensated 3.54 cm
== dev_pose_sonic_r1/vs_point.out
  wrist + palm + fingers position-only  5.21 cm, this run  5.08 cm; drop of the median 0.02 (-0.18 to 0.18); of the 90 % level  9.27 -> 14.03: -0.51 (-0.69 to -0.16); better on 0.56 of the pairs
  palm                   position-only  3.46 cm, this run  3.32 cm; drop of the median 0.04 (-0.12 to 0.10); of the 90 % level  6.19 ->  7.41: -0.20 (-0.45 to -0.01); better on 0.53 of the pairs
  arm (24 spheres)       position-only  5.70 cm, this run  6.22 cm; drop of the median -0.09 (-0.24 to 0.03); of the 90 % level 10.57 -> 14.24: -0.35 (-0.56 to -0.09); better on 0.46 of the pairs
  palm's settled end     position-only  1.50 cm, this run  1.52 cm; drop of the median -0.01 (-0.33 to 0.20); of the 90 % level  3.09 ->  4.84: -0.57 (-1.03 to -0.24); better on 0.50 of the pairs
== dev_pose_sonic_r1/turn.out
  all                   error  15.2  commanded  19.2  actual   9.3  predicted   8.0 deg | actual / commanded 0.64, predicted / commanded 0.59 | cos actual +0.64, predicted +0.96 | error left   8.0 deg
  commanded  5-15 ( 50) error  10.7  commanded   7.1  actual   7.9  predicted   7.1 deg | actual / commanded 1.04, predicted / commanded 0.90 | cos actual +0.90, predicted +0.99 | error left   5.3 deg
  commanded 15-25 ( 20) error  14.9  commanded  19.9  actual  11.0  predicted   9.7 deg | actual / commanded 0.57, predicted / commanded 0.52 | cos actual +0.89, predicted +0.95 | error left   6.0 deg
  commanded 25-90 ( 70) error  28.9  commanded  40.0  actual  13.8  predicted  10.8 deg | actual / commanded 0.38, predicted / commanded 0.30 | cos actual -0.55, predicted -0.69 | error left  36.3 deg
```

**对照判据（开发路线检查）：没过，接着跑第 2 轮。**
- 模型对补偿回放的预测误差 3.54 cm，对直接发命令 3.28 cm，1.08 倍，不超过 2 倍：过。
- 腕 + 掌 + 指比只挪位置低 2 %，区间 −18 % 到 18 % 含 0：没过。90 % 分位从 9.27 cm 变差到 14.03 cm。
- 按规则最多 3 轮。第 1 轮不套"比上一轮改善不到 5 % 就停"，没有上一轮的位置加朝向可比，ScaleBFM 那次也是从第 2 轮起才比。第 2 轮 07:34 UTC 启动（`rk.sh 2 8900000 9000000`）。

**和预测对不上的：**
- 训练路线第 1 轮，第 0 轮模型对位置加朝向那一遍的预测误差只是对直接发命令的 1.19 倍（绕行），全部路线 1.06 倍。预测是 1.3–2.5 倍。ScaleBFM 是 2.45 倍：SONIC 的第 0 轮模型对转过的命令预测得比 ScaleBFM 好得多。
- 命令转了 25° 以上的 70 对里，实际转动方向和命令方向的余弦是 −0.55，模型预测的是 −0.69：要求转得大时，SONIC 的手腕往反方向转，模型也这么预测，补偿器于是把命令推到 40° 的上限，剩下的误差 36.3°。腕 + 掌 + 指 90 % 分位变差应该来自这一组。这是看完数据后的解释，没有检验。

### 第五步补充：第 2 轮，到此停（2026-11-23，4090）

第 2 轮（`sonic_pose_4090/rk.sh 2 8900000 9000000`，07:34–08:00 UTC）。训练路线上用第 1 轮模型跑位置加朝向（`onpol_sonic_r2_pose`），1360 段，0 摔倒：

```
detour paths (over an edge, over a box): 190 routes, 757 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 9.16 cm, compensated 4.63 cm; drop 0.49 (0.45 to 0.53); 90 % level 16.22 and 15.59 cm
  hand's attitude against the wanted one: settled end direct 17.6 deg, compensated 9.8 deg; along the path 8.6 and 4.8 deg
  the model's own error (arm) on the run it predicted: direct 2.78 cm, compensated 3.15 cm
```

第 2 轮模型 `via_point_sonic_onpol2_20261123`（第 1 轮的四份数据 + 这一组）。开发路线（`dev_pose_sonic_r2`，ID0 9000000）：

```
pairs 163 (direct 164, compensated 164 paths without a fall; 1 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.04 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.01 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
all: 41 routes, 163 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 9.26 cm, compensated 5.14 cm; drop 0.44 (0.36 to 0.53); 90 % level 15.49 and 13.71 cm
  palm, error against the wanted sweep: direct 7.08 cm, compensated 3.45 cm; drop 0.51 (0.45 to 0.56); improved in 0.96 of the pairs
  arm,  error against the wanted sweep: direct 9.46 cm, compensated 6.25 cm; drop 0.34 (0.26 to 0.42); improved in 0.89 of the pairs
  palm's settled end against the wanted end: direct 5.12 cm, compensated 1.61 cm; drop 0.68 (0.63 to 0.73)
  hand's attitude against the wanted one: settled end direct 15.2 deg, compensated 8.8 deg; along the path 8.5 and 5.3 deg
  points moved: median 2.97 cm, largest 8.00 cm; the goal 4.44 cm; wrist commands turned: median 7.4 deg, largest 40.0 deg
  the model's own error (arm) on the run it predicted: direct 3.27 cm, compensated 3.48 cm
== vs_point.out (against position-only, dev_point_sonic_4090)
  wrist + palm + fingers position-only  5.21 cm, this run  5.14 cm; drop of the median 0.01 (-0.14 to 0.18); of the 90 % level  9.27 -> 13.71: -0.48 (-0.66 to -0.12); better on 0.55 of the pairs
  palm                   position-only  3.46 cm, this run  3.45 cm; drop of the median 0.00 (-0.11 to 0.08); of the 90 % level  6.19 ->  7.54: -0.22 (-0.44 to -0.03); better on 0.53 of the pairs
  arm (24 spheres)       position-only  5.70 cm, this run  6.25 cm; drop of the median -0.10 (-0.23 to 0.01); of the 90 % level 10.57 -> 14.49: -0.37 (-0.55 to -0.08); better on 0.42 of the pairs
== vs_prev.out (against round 1, dev_pose_sonic_r1)
  wrist + palm + fingers position-only  5.08 cm, this run  5.14 cm; drop of the median -0.01 (-0.08 to 0.09); of the 90 % level 14.03 -> 13.71: 0.02 (-0.07 to 0.05); better on 0.47 of the pairs
== turn.out
  all                   error  15.2  commanded  19.4  actual   9.2  predicted   7.8 deg | actual / commanded 0.65, predicted / commanded 0.58 | cos actual +0.69, predicted +0.96 | error left   8.8 deg
  commanded 25-90 ( 73) error  28.3  commanded  40.0  actual  13.1  predicted  10.5 deg | actual / commanded 0.39, predicted / commanded 0.30 | cos actual -0.52, predicted -0.72 | error left  33.4 deg
```

（`vs_prev.out` 的 "position-only" 一栏是上一轮，即第 1 轮。这是 `via_vs.py` 的固定标签。）

**对照判据：**
- **停在第 2 轮：**腕 + 掌 + 指 5.08 → 5.14 cm，比上一轮改善 −1 %（区间 −8 % 到 9 %），不到 5 %。
- **停下的这一轮，开发路线检查没过：**
  - 模型误差 1.06 倍，不超过 2 倍：过；
  - 腕 + 掌 + 指比只挪位置低 1 %，区间 −14 % 到 18 % 含 0：没过。
- **按判据不上测试集，向用户报告。**测试路线上的位置加朝向和常数修正都没有跑。
- **第五步的补偿判据本身**（比直接发命令降 ≥ 20 %、区间不含 0）在开发路线上满足：降 44 %，区间 36 % 到 53 %。但只挪位置也降 44 %，加朝向没有带来额外的好处。

**和预测对照**（开发路线，最后一轮）：

| 预测 | 结果 | 落在区间里吗 |
|---|---|---|
| 模型误差 1.0–1.8 倍 | 1.06 | 在 |
| 腕 + 掌 + 指比直接降 45–60 % | 44 % | 刚好在外 |
| 比只挪位置低 5–25 % | 低 1 % | 不在 |
| 终点朝向降 35–60 % | 15.2° → 8.8°，降 42 % | 在 |
| 手掌比只挪位置差不超过 0.3 cm | 好 0.01 cm | 在 |
| 整条手臂比只挪位置差 0–20 % | 差 10 % | 在 |
| 停在第 2 轮 | 停在第 2 轮 | 在 |
| 训练路线第 1 轮模型误差 1.3–2.5 倍 | 1.19 | 不在 |

**意外：**
- 两轮补偿器回放进训练，几乎没有改进：第 1 轮 5.08 cm，第 2 轮 5.14 cm。ScaleBFM 上第 1 轮就把模型误差从 2.45 倍降到 1.80 倍；SONIC 第 0 轮已经是 1.19 倍，没有多少可学的。
- 尾部变差来自"命令转了 25° 以上"的那一组（163 对里有 73 对）：
  - 实际转动方向和命令方向的余弦是 −0.52：要求转得大时，SONIC 的手腕往反方向转；
  - 模型也预测反向（−0.72），补偿器于是把命令推到 40° 上限，剩下 33.4° 的误差。
  - 这是看完数据后的解释，没有检验。要检验，可以只在命令转角 ≤ 25° 时补偿朝向，但这改了补偿设置，是新实验，留给用户决定。

### 第五步补充：独立复核（2026-11-23）

独立 agent（general-purpose，新上下文），只给它这一步的预注册、两节结果、原始输出和代码的路径，不给推理过程。它的回复原样如下：

> 我的结论是**支持**。所有数字都和原始输出对得上，规则是照预注册执行的，几处偏差都偏向朝向补偿这一方，对"没赢"这个结论只会更保守。但结论的范围要写清楚，下面第 4 部分说明。
>
> ## 1. 数字核对
>
> 两个结果节和结论里引用的数字，我逐项对过以下原始文件，没有发现不一致：
>
> - 开发路线：`comp_eval.out`、`vs_point.out`、`vs_prev.out`、`parts.out`、`turn.out`。
> - 训练路线：`onpol_sonic_r1_*`、`onpol_sonic_r2_pose`。
> - 训练和过程记录：`train.out`（路线数 16002、17362）、`match.txt`、`determinism.txt`、`r1.log` 和 `r2.log` 里的时间。
> - 比值：1.08、1.06、1.19、1.06，以及终点朝向降 42 %，复算都对。
>
> 我又用 `comp_pairs.npz` 独立重算了一遍：
>
> - 腕 + 掌 + 指的中位数：直接发命令 9.26 cm，只挪位置 5.21 cm，第 1 轮 5.08 cm，第 2 轮 5.14 cm，和记录一致。
> - 三组都去掉了同一个开场（第 42 号）。
>
> 只有两处小的省略，不影响结论：
>
> - 第 2 轮 `vs_point` 里"手掌终点"那一行没有贴，它对加朝向一方不利（−7 %）。
> - `vs_prev` 只贴了腕 + 掌 + 指一行。
>
> ## 2. 四类陷阱
>
> - **测试集的选取：没有问题。** 41 条开发路线是 ScaleBFM 时就定好的（种子 20261122），不是看了结果再挑的。但这 41 条全是绕行路线，没有伸到板下的路线，见第 4 部分。
> - **数据泄漏：没有直接泄漏，但训练路线里有近乎重复的路线。**
>   - 第 1、2 轮的训练数据只有 `data_via_sonic` 和训练路线上的回放，开发路线没有进训练；补偿也不用标定集。
>   - 开发路线和训练路线的目标没有一个完全相同。但 41 条里有 14 条，在训练路线中能找到一条 7 个路点都相差不到 2 cm 的，最近的只差 0.73 cm。
>   - 这只对加朝向一方有利：它用的第 1、2 轮模型见过这些训练路线，只挪位置一方用的第 0 轮模型没见过。
> - **指标使用：没有换指标。**
>   - 主指标是事先写好的腕 + 掌 + 指中位数。对它不利的结果也报了：90 % 分位明显变差（−48 %，区间 −66 % 到 −12 %），整条手臂也更差。
>   - 对它有利的也都在原始输出里：终点朝向比只挪位置好，13.9° 降到 8.8°。
> - **事后挑选：没有。**
>   - 第 1 轮其实略好（5.08 cm 对 5.14 cm），但按规则停在了第 2 轮，没有挑更好的那一轮。
>   - "要求转 25° 以上时手腕往反方向转"这个解释，记录里已经标明是事后的。
>
> **规则执行：**
>
> - **第 1 轮不套"改善不到 5 % 就停"：说得通。** 预注册只安排了第 0 轮模型在开发路线上跑直接发命令和只挪位置，没有安排位置加朝向，所以第 1 轮没有"上一轮"可比。ScaleBFM 那次第 0 轮模型（`via_point_20261122`）也没在开发路线上跑过位置加朝向，同样没有比。如果改成和只挪位置比，改善只有 2.5 %，会停在第 1 轮，结论还是没过、不上测试集。
> - **停在第 2 轮、不跑测试集：符合条文。** 第 2 轮比上一轮改善 −1 %，不到 5 %，按规则停；停下时开发路线检查没过，按判据不上测试集。条文里"3 轮之后仍不过"有一点歧义，也可以读成检查没过就该跑满 3 轮。ScaleBFM 那次检查通过了，没有遇到这种情况，所以先例不覆盖。不过第 2 轮对第 1 轮是 −1 %（区间 −8 % 到 9 %），跑第 3 轮翻盘的可能很小。
> - **配对比较：成立。**
>   - 同一台 4090、同一组卡（1、2、3 号），同一种子 47000000，路线列表和开场编号都有断言保证对齐。
>   - 每个开场在各遍里落在同一张卡上：直接发命令那遍用 6 个进程，补偿那几遍用 18 个，都是 3 的倍数。
>   - 起点和直接发命令那遍最多差 0.04 mm。逐位重复只在 6 进程对 6 进程上验证过，但起点核对已经覆盖了不同进程数的情况。
> - **过程上的不足：** 预注册这一节还没有提交，所有运行用的提交 ff62f33e 里没有这一节，所以没法用提交时间证明它写在运行之前。不过有好几条预测没中（比如 1.3–2.5 倍实际是 1.19 倍），不像是看了结果再写的。
>
> ## 3. 可能推翻结论的实验
>
> - **实验 1：只在转角小时补朝向。** 我在现有数据上事后按补偿器要求的终点转角分组：
>   - 转角小于 25° 的 90 对里，加朝向比只挪位置好 15 %，区间 7 % 到 22 %。
>   - 25° 以上的 73 对里差 46 %。
>
>   建议换一批新的开发路线（新的规划种子，大约 120 条绕行 × 4 遍，不能再用这 41 条），事先写好"只在转角不超过 25° 时补朝向"的规则。两边都用第 2 轮模型，和只挪位置配对比较。如果中位数更低、区间不含 0、90 % 分位也不变差，就推翻"SONIC 上补朝向没用"的一般说法。原结论只针对固定的默认设置，那部分不受影响。这一项改了补偿设置，是新实验，按记录要由用户决定。
> - **实验 2：两边用同一个模型，再加一组种子。** 在 41 条开发路线上用第 2 轮模型跑只挪位置，再换一个种子把两边各跑一遍，每遍大约 5 分钟。如果同模型下加朝向好 10 % 以上、区间不含 0，结论就要改。
>
> ## 4. 判定：支持
>
> 理由：
>
> - 数字全部对上，规则按条文执行，配对是有效的。
> - 两处偏差都偏向加朝向一方：它的模型更新过（在直接发命令那遍上的模型误差 3.27 cm，第 0 轮是 4.06 cm），而且见过近乎重复的路线。即便如此它也没有赢。
>
> 结论的措辞建议这样收窄：
>
> - **"没赢"是"没能证明更好"，不是"两者相当"。** 中位数的区间 −14 % 到 18 % 不排除有一个中等大小的好处，但 90 % 分位确实变差了，而且区间不含 0。
> - **限定在绕行路线上。** 我在训练路线上做了同模型对照：两边都用第 0 轮模型，这些路线它没见过。
>   - 190 条绕行路线：加朝向差 3.5 %，区间 −9.5 % 到 2.2 %，和开发路线的结论一致。
>   - 150 条伸到板下的路线：加朝向好 5.8 %，区间 3.4 % 到 8.6 %。
> - **这一比较同时换了两样东西：** 一是加了朝向补偿，二是换了训练过补偿器回放的模型。两者的作用分不开。
>
> 我没有写任何文件，也没有在远程机器上起 GPU 任务。

**回填后的结论（按复核收窄）：**
- 在 41 条开发绕行路线上，用默认补偿设置，SONIC 加朝向补偿**没能证明**比只挪位置更好：中位数低 1 %，区间 −14 % 到 18 %；90 % 分位变差，区间不含 0。
- 按事先的规则停在第 2 轮，不上测试集。
- 这一比较同时换了两样东西（加了朝向补偿、换了训练过补偿器回放的模型），两者的作用分不开。
- 复核事后在训练路线上做了同模型对照：伸到板下的路线上，加朝向好 5.8 %（区间 3.4 % 到 8.6 %）；绕行路线上差 3.5 %。
- 复核事后按转角分组：要求转角 < 25° 的 90 对里好 15 %（区间 7 % 到 22 %），≥ 25° 的 73 对里差 46 %。这两条都是看过数据后才分的组，只算线索。要检验得换新的开发路线另做实验（复核的实验 1），由用户决定。
- 复核指出：预注册这一节还没有提交，没法用提交时间证明它写在运行之前。提交时一并补上。

## 第五步补充之二：只在转角不大时补朝向（SONIC，新开发路线；2026-11-23，4090；用户："把这三件事都做了"；本节在任何运行之前写下）

### 为什么

第五步补充的复核事后按"补偿器要求的终点转角"分组：
- 转角 < 25° 的 90 对里，加朝向比只挪位置好 15 %（区间 7 % 到 22 %）；
- 25° 以上的 73 对里差 46 %。

这是看过数据后分的组，只算线索。这里把它写成事先定好的规则，换一批没见过的开发路线来检验。同时两边都用同一个模型，去掉"同时换了模型"的混杂。

### 做法

- **新开发路线：**`fresh_routes.py`，种子 20261127，规划 900 个场景，只用长绕行（越过桌沿、越过箱子）。
  - 和训练路线（20261123）、旧开发路线（20261122）、测试路线（20261121）、vastgpu 的第二批测试路线（20261124）都不共用场景。
  - 规划失败的场景照实记，不重抽。
- **回放：**SONIC，4090 的 1–3 号卡，网络在 GPU；每条路线 4 遍，种子 67000000；进程数都是 3 的倍数，每次开场在各组里落在同一张卡上。四组：
  - 直接发命令（ID0 10100000）；
  - 只挪位置（10200000）；
  - 位置加朝向，不设门槛（10300000）；
  - 位置加朝向，设门槛（10400000，`REACH_VIA_COMP_TURN_GATE=25`）。
  - 补偿都用第 2 轮模型 `via_point_sonic_onpol2_20261123`。
- **门槛规则：**先照常算位置加朝向的补偿。补偿后的命令在终点处要手转的角度超过 25° 的，这条路线改成只挪位置，同一个模型、同一个状态。
  - 角度的算法：原命令和补偿后命令的规划在最后一点上，手的 15 个球的最小二乘转角，和 `via_turn.py` 的 "commanded" 相同。
  - 门槛 25° 来自旧开发路线上的事后分组，这里不再调。
- **自检：**
  - 设门槛那一组的每条路线，发出的命令必须等于"不设门槛"组的命令（转角 ≤ 25°）或"只挪位置"组的命令（> 25°），差 0.00 mm；
  - 三组补偿的一致性检查（`via_comp_eval.py` 开头三行）都是 0.00 mm。
- **比较：**
  - 主比较：`via_vs.py` 只挪位置 → 设门槛；
  - 次要比较：只挪位置 → 不设门槛（同一模型下重做旧结论）。
  - 另外 `via_comp_eval.py` 各组对直接发命令，`via_turn.py` 看两组加朝向的。

### 预测

- **门槛触发的比例：**35–55 %（旧开发路线 73 / 163 = 45 %）。
- **主比较（腕 + 掌 + 指中位数）：**设门槛比只挪位置低 0–10 %。
  - 事后分组的 15 % 只在一半路线上，而且是挑出来的，会往 0 回落。
  - 我估计区间不含 0 的可能性约三分之一。
- **90 % 分位：**设门槛和只挪位置相差 ±10 % 以内。触发门槛的路线和只挪位置一样，不触发的那些误差本来就小。
- **次要比较：**不设门槛的加朝向和只挪位置的中位数相差 ±5 % 以内，90 % 分位差 20–50 %（旧开发路线差 48 %）。
- **终点朝向：**设门槛比只挪位置好 20–40 %。
- **模型误差：**模型对设门槛那一遍的误差不超过对直接发命令那一遍的 1.3 倍。

### 判据

- **在开发路线上成立：**
  - 设门槛的腕 + 掌 + 指中位数比只挪位置低，按路线自助抽样的 95 % 区间不含 0；
  - 并且 90 % 分位不变差：它的降幅区间不全在 0 以下。
- **成立：**在测试路线（`fresh_routes`，种子 20261121，63 条 × 4 遍，种子 41000000）上只跑一次。
  - 两组：只挪位置（ID0 10500000）、设门槛（10600000），都用第 2 轮模型；
  - 测试路线上用同一判据，做独立复核，报告。
- **不成立：**停下，记"在 SONIC 上，设了门槛的朝向补偿也没能证明比只挪位置更好"，不上测试集，不再换门槛。

### 第五步补充之二：运行记录

- **新开发路线**（`dev_routes2`，09:32–09:49 UTC）：

```
planned 380 of 900: edge 61/300, box 51/300, under 268/300
routes 112 (edge 61, box 51), resets 448
```

- **第一次运行没跑全，结果作废**（09:49–10:09 UTC，留在 `sonic_gate_4090/failed_oom_1/`）。
  - 四组和任务 B 的三组 ScaleBFM 同时在 1–3 号卡上跑，部分 SONIC 进程建网络时申请不到显存，退出：

```
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Exception during initialization: ... bfc_arena.cc:359 ... Failed to allocate memory for requested buffer of size 12582912
direct exit codes [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]   (260 of 448 segments)
point  exit codes [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]   (112)
pose   exit codes [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]   (336)
gated  exit codes [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]   (298)
```

  - 看到了什么：
    - 各组对直接发命令的 `via_comp_eval.py` 输出，但各组的子集不同，彼此不能比；
    - 门槛自检在三组共有的 74 次开场上：设门槛组的命令和应有的那一组差 0.000 cm，36 次触发门槛。
  - 主比较（`via_vs.py`）因为开场对不上报错，没有算出来。
- **重跑**（10:11 UTC 起，`sonic_gate_4090/run.sh`）：设置不变，种子、编号、进程数都相同，每次开场落在和第一次相同的卡上。
  - 两组一批，避免显存不够；
  - 每组必须所有进程退出码为 0、段数等于开场数，否则停下。
  - 预测和判据不改。

### 第五步补充之二：结果（重跑，10:11–10:33 UTC）

四组都跑全：各 448 段、0 摔倒，所有进程退出码 0（`sonic_gate_4090/*/run.log`）。

门槛自检（`gate_check.out`）：

```
448 resets in all three runs (point 448, pose 448, gated 448); gate 25.0 deg tripped on 244 = 0.545
  wanted sweep (comp_ref0) gated against pose: largest 0.000 cm
  within the gate (204): gated against pose largest 0.000 cm; its comp_turn zero on 0
  above the gate (244): gated against point largest 0.000 cm; its comp_turn zero on 244
  end turn of the ungated pose command, deg: median 29.4, quantiles 25/75/90 % 8.9 40.0 40.0
```

三组补偿对直接发命令（`*/comp_eval.out`；一致性检查三组相同：`median 0.00 mm, largest 0.46 / 0.24 / 0.00 mm`）：

```
point  all: 112 routes, 448 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 9.31 cm, compensated 4.71 cm; drop 0.49 (0.46 to 0.52); 90 % level 15.66 and 8.00 cm
  hand's attitude against the wanted one: settled end direct 18.0 deg, compensated 14.9 deg; along the path 8.4 and 8.9 deg
  the model's own error (arm) on the run it predicted: direct 2.98 cm, compensated 2.87 cm
pose   wrist + palm + fingers ... direct 9.31 cm, compensated 5.05 cm; drop 0.46 (0.41 to 0.50); 90 % level 15.66 and 13.37 cm
  hand's attitude ... settled end direct 18.0 deg, compensated 12.4 deg; along the path 8.4 and 4.8 deg
  the model's own error (arm) on the run it predicted: direct 2.98 cm, compensated 3.47 cm
gated  wrist + palm + fingers ... direct 9.31 cm, compensated 4.62 cm; drop 0.50 (0.47 to 0.53); 90 % level 15.66 and 8.11 cm
  hand's attitude ... settled end direct 18.0 deg, compensated 12.6 deg; along the path 8.4 and 6.9 deg
  points moved: median 2.94 cm, largest 8.00 cm; the goal 4.17 cm; wrist commands turned: median 6.0 deg, largest 24.9 deg
  the model's own error (arm) on the run it predicted: direct 2.98 cm, compensated 2.97 cm
```

主比较和次要比较（`gated/vs_point.out`、`pose/vs_point.out`）：

```
gated | 448 pairs, 112 routes
  wrist + palm + fingers position-only  4.71 cm, this run  4.62 cm; drop of the median 0.02 (-0.01 to 0.06); of the 90 % level  8.00 ->  8.11: -0.01 (-0.04 to 0.01); better on 0.33 of the pairs
  palm                   position-only  2.76 cm, this run  2.76 cm; drop of the median -0.00 (-0.03 to 0.02); of the 90 % level  4.91 ->  4.83: 0.02 (-0.04 to 0.07); better on 0.25 of the pairs
  palm's settled end     position-only  1.11 cm, this run  1.17 cm; drop of the median -0.05 (-0.11 to 0.02); of the 90 % level  2.25 ->  2.35: -0.04 (-0.14 to 0.04); better on 0.22 of the pairs
pose | 448 pairs, 112 routes
  wrist + palm + fingers position-only  4.71 cm, this run  5.05 cm; drop of the median -0.07 (-0.15 to 0.00); of the 90 % level  8.00 -> 13.37: -0.67 (-0.92 to -0.50); better on 0.43 of the pairs
  palm's settled end     position-only  1.11 cm, this run  1.66 cm; drop of the median -0.49 (-0.71 to -0.33); of the 90 % level  2.25 ->  5.52: -1.45 (-2.00 to -0.84); better on 0.31 of the pairs
```

**对照判据：不成立。**
- 设门槛的腕 + 掌 + 指中位数比只挪位置低 2 %，区间 −1 % 到 6 % 含 0。
- 90 % 分位变差 1.5 %，不显著（区间 −4 % 到 1 %），没有违反判据。（复核更正：原写"没有变差"。）
- 按事先写的：停下，记"在 SONIC 上，设了门槛的朝向补偿也没能证明比只挪位置更好"，不上测试集，不再换门槛。

**和预测对照：**

| 预测 | 结果 | 在区间里吗 |
|---|---|---|
| 门槛触发 35–55 % | 54.5 % | 在 |
| 主比较低 0–10 % | 低 2 % | 在 |
| 90 % 分位相差 ±10 % 以内 | 差 1 % | 在 |
| 不设门槛：中位数相差 ±5 % 以内 | 差 7 % | 不在（更差） |
| 不设门槛：90 % 分位差 20–50 % | 差 67 % | 不在（更差） |
| 终点朝向好 20–40 % | 14.9° → 12.6°，好 15 % | 不在 |
| 模型误差不超过 1.3 倍 | 1.00 倍 | 在 |

**事后分组（不是预注册的判据，只读，`gate_subset.out`）：**

```
448 pairs; ungated end turn <= 25 deg on 204
  end turn <= 25 (204 pairs): wrist + palm + fingers, position-only 3.69 cm, pose 3.25, gated 3.25
    pose against position-only: drop of the median +0.12 (+0.07 to +0.17), 66 routes; of the 90 % level +0.07 (-0.03 to +0.14), 66 routes
  end turn > 25 (244 pairs): wrist + palm + fingers, position-only 5.70 cm, pose 7.40, gated 5.70
    pose against position-only: drop of the median -0.30 (-0.45 to -0.18), 74 routes; of the 90 % level -0.88 (-1.04 to -0.63), 74 routes
```

- 分组用的是补偿器在发命令之前就算得出的量。复核在旧路线上事后看到"转角 < 25° 时好 15 %"，在新路线上重现：好 12 %（区间 7 % 到 17 %）。
- 但这一半路线的误差本来就小（3.69 cm 对 5.70 cm），整体中位数主要由另一半决定，所以整体只低 2 %。事先把判据定在整体中位数上，测不出这种只落在一部分路线上的好处。这是判据设计的局限，事后发现的，不改判定。

**意外：**
- 不设门槛的朝向补偿，同一模型下比只挪位置差得比旧结论还多：90 % 分位差 67 %，手掌终点差 49 %。旧结论里两边模型不同，这次排除了"换模型"的解释。
- 要求转 25° 以上时，实际转动方向和命令方向的余弦 −0.52，模型预测的是 −0.81，和旧开发路线的 −0.52 / −0.72 一样。

### 第五步补充之二：结论与下一步（独立复核前）

- **结论：** 在新开发路线上，只在转角 ≤ 25° 时补朝向，整体没能证明比只挪位置更好。
  - 主比较中位数低 2 %，区间含 0（`gated/vs_point.out` 第 1 行）；
  - 90 % 分位不变（同一行）。
  - 按判据停下，不上测试路线，不再换门槛值。
- **证据强度：** 一组路线（112 条，448 对），一个回放种子；开场按路线重抽样。
- **另外的解释：**
  - 好处只落在转角小的那一半路线（事后分组好 12 %），整体中位数被另一半决定。所以这个结果是"判据测不出局部好处"，不等于"门槛没用"。
  - 门槛 25° 是在旧路线上事后看到的；新路线上门槛附近的分布不同（触发 54.5 %），换门槛可能不同。但按事先写的不再调。
- **四类陷阱：**
  - benchmark 选择：路线是新种子、只取长绕行，和旧结论同一口径，没有挑；
  - 数据泄漏：模型是第 2 轮模型，训练用的是旧开发路线和空地数据，新路线没进训练；
  - 指标误用：主指标和判据事先写定，没换；事后分组标明不是判据；
  - 事后挑选：只有一次完整运行；第一次没跑全的已作废并保留，没有拿它做判定。
- **对照判据：** 落在"不成立"一侧，停下并记录。

### 第五步补充之二：独立复核（2026-11-23，原文）

- 怎么做的：独立 agent，只给它计划、结果和输出路径，只读、只用 CPU，不碰测试路线。
- 它写的文件：`/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/tmp/review_gate/`。
- 判定：
  - (a) 主结论：支持；
  - (b) 事后分组：窄义支持。

主要内容如下。

1. **数字核对**：全部和原始输出一致。
   - 自己写 kabsch，自己做按路线抽样（种子 7，1 万次）：
     - 主比较 `drop med +0.018 (-0.007,+0.058); p90 8.00 -> 8.11 drop -0.015 (-0.040,+0.011); better 0.33 tie 0.55`；
     - 门槛以内 `pose vs point <=25: 3.69 -> 3.25, +0.121 (+0.074,+0.175)`；
     - 模型误差比 0.998。
   - 更正：90 % 分位是变差 1.5 %，不显著。
   - 55 % 的配对两边一样（门槛以上发的是同一条命令）；只看不一样的那些，设门槛的好的占 73 %。
2. **四类陷阱**：
   - **路线**：112 条正好是全部规划成功的越桌沿和越箱子场景，没挑。
   - **泄漏**：没有。
     - 模型 07:57:30 UTC 存盘，新路线 09:31–09:48 UTC 才生成。
     - 场景终点没有完全重合的，最近距离中位 9.1 mm，最小 0.45 mm。
     - 隐患：不同种子生成的场景同名（900 个都同名），以后按名字拼接数据时要小心。
   - **指标**：评测脚本和 478aa17f 的版本一样；`gate_check.py`（09:47）早于运行（09:48:55）。
     - 但 git 证明不了计划先于运行：478aa17f（09:21）里没有这段计划，文件最后一次改动在出结果之后。
   - **挑选**：作废的第一次和重跑，在重叠的开场上逐位一样（`through_comp max |first-rerun| cm 0.00000`），没有挑的余地。
   - **门槛**：发出的命令和实际结果都和应有的那一组逐位一样。门槛用的量只来自规划和模型预测，发命令前就能算。
3. **局限，以及能推翻结论的实验**：
   - 区间上限 +5.8 %；55 % 的配对必然打平，功效低是设计决定的。
   - 事后试了 10°–39.9° 七个门槛，整体都在 +1.0 % 到 +2.4 % 之间，区间都含 0，换门槛救不回这个判据。
   - 推翻实验：
     - 新开发种子，约 350 条路线；
     - 只跑只挪位置和不设门槛两组，设门槛那组可以由这两组拼出来；
     - 事先登记一个等效界（如 ±3 %），把门槛以内的比较作为主判据，并给手掌终点设一个不劣界。
4. **事后分组的问题**：
   - **只在"腕 + 掌 + 指"上好**。门槛以内：
     - 手掌 +0.2 %（−3.0 % 到 +4.6 %）；
     - 手臂 +0.3 %；
     - 手掌终点 −9.6 %（−30 % 到 +4.5 %）。
   - **和"路线本来就容易"分不开**：分组用的转角和直接发命令时误差的秩相关是 0.83。
   - **转角大的一半多数补偿到顶了**：244 对里 187 对卡在 40° 上限，模型预测手往命令的反方向转（余弦 −0.81）。所以门槛实际分的是"补偿收敛了"和"补偿冲到上限"。
   - **区间本身诚实**，但这不是登记过的判据。
5. **建议措辞**："整体中位数降 2 %，区间 −1 % 到 6 %，没能证明更好；如果有好处，也在 6 % 以内；这是位置误差上的结论，终点手的朝向从 14.9° 改善到 12.6°，好 15 %。"

### 第五步补充之二：最终结论（采纳复核）

- **主结论**：在 SONIC、新开发路线（112 条，448 对）上，只在转角 ≤ 25° 时补朝向：
  - 整体位置误差中位数降 2 %，区间 −1 % 到 6 %，没能证明比只挪位置更好；如果有好处，也在 6 % 以内；
  - 终点手的朝向从 14.9° 改善到 12.6°；
  - 按判据停下，不上测试路线。
- **门槛以内好 12 %**：只作为"腕 + 掌 + 指"这一个量上的线索，不写成"转角小时补朝向更好"。
  - 手掌和手臂没有变化，手掌终点有变差的倾向；
  - 和"路线本来就容易、补偿没冲到上限"分不开。
- **不设门槛的朝向补偿**：同一个模型下 90 % 分位差 67 %。旧结论"没能证明更好"在 SONIC 上可以收紧为"明显更差"，原因是转角大时补偿冲到 40° 上限、方向预测反了。
- **如果以后还要做**：按复核的建议，事先登记等效界和门槛以内的主判据，用新开发种子，约 350 条路线。

## 第四步第二部分：有真障碍物的窄缝测试（2026-11-23 在 4090 上写的设计）

用户 2026-11-23："先把做法、预测和判据写全，交给我确认，确认之前不要跑。"这一节取代前面"第二步：有真障碍物的场景"那份草案。

**用户 2026-11-23 确认（"都同意"）：**横杆场景；先只做 ScaleBFM；我们的做法按预测扫过路线留 μ；所有余量在开发场景上定；这一轮不做关节轨迹对手；加"补偿后用固定余量"的消融。下面的预测和判据从此锁定，之后的改动写进"预测修订"。

### 要回答的问题

1. 在放了真障碍物的场景里，每种做法各自带着余量去规划、在冻结的控制器上执行，能安全穿过多窄的缝？我们的做法是不是比别的窄？
2. 第一部分用空地回放算出的"碰不碰"，在真障碍物前面还成立吗？

### 写设计之前已经知道的，改变了草案

- **规划器贴着障碍物走。**在训练和开发两批场景里，越过桌沿、越过箱子的路线，碰撞球离障碍物的最小距离中位数 0.8 cm、95 % 分位约 2 cm，激活距离虽然设的是 3 cm。
  - 而直接发命令时腕 + 掌 + 指偏离规划，中位数 8 cm，95 % 分位 12 cm（第一部分，ScaleBFM，测试绕行路线）。
  - 所以放上真障碍物以后"不加余量直接走"几乎必碰。curobo_avoid 里 28 局有 27 局碰到桌子，也是这个原因。
  - 草案隐含"缝的半宽就是规划留的余量"，这在真场景里不成立。每种做法都必须自己告诉规划器留多少余量：把障碍物各边加厚 m 再规划。
- **起点离桌沿很近。**站姿起点时手掌在身前 12.7 cm、高 0.72 m，正是桌沿的高度。桌沿场景规划失败的一部分就是起点已经碰到桌子。
- **规划成功率低。**锁腰只动右臂时，越过桌沿的场景只有 21 % 规划得出来（训练路线 107 / 500）。

### 场景：桌沿上方加一道横杆

- **桌子：**同桌沿场景，桌面高 0.70–0.90 m，前沿在身前 0.20–0.26 m。
- **横杆：**和桌沿平行、横跨桌宽，截面 3 cm × 3 cm，正好架在桌沿上方，杆底比桌面高 g。
- **目标：**越过桌沿 8–14 cm，高度在缝的中间（桌面 + g/2，上下随机 2 cm 以内）。手大致水平朝前（俯仰 0–30°，偏航 ±20°）。
- **为什么这样放：**
  - 手要先从桌沿前面抬起来、越过桌沿，再从桌面和横杆之间穿过去，所以是一条绕行路线，第一部分的好处主要就在这类路线上；
  - 目标在缝的中间，到了终点手离桌面、离横杆的空隙也由 g 决定，所以整条路线上起约束作用的都是缝宽。
- **缝宽的刻度：**用"缝比手厚多出多少"，f = g − h。
  - h 是腕、掌、指这 17 个球在手水平时竖直方向的总厚度，开跑前用正运动学量一次，预计 8–11 cm。
  - f 取 3、5、7、9、11、14、17、20、24 cm，共 9 档。
- **同一组底场景在 9 档上共用：**桌子、目标、朝向都不变，只换横杆高度，每档重新规划。这样各档之间是配对的。
- **数量和种子：**开发 20 个底场景，测试 30 个底场景，每个场景 4 遍（第 1 遍原地出发，后 3 遍先走一小段，和以前相同）。种子都是新的，不用 20261121–20261123。

### 五种做法（都用同一个规划器、同样的障碍物、同样的起点）

余量 m 的刻度是 1、2、3、4、6、8、10、12 cm。规划时把障碍物各边加厚 m，激活距离仍是 3 cm。规划不出来就算放弃。

| 做法 | 怎么选余量、发什么命令 |
|---|---|
| 固定余量 m | 用 m 规划，直接发规划出来的命令。8 个 m 都跑，画出整条取舍曲线 |
| 标定过的固定余量 | 上面那一族里取 m = 开发场景上直接发命令时腕 + 掌 + 指误差的 95 % 分位（往上取到刻度上） |
| 学出来的标量半径 | m 从小到大试：规划，用标量半径模型预测这条命令的半径 r（在开发场景上缩放到盖住 95 %），规划路线离真障碍物的最小距离 ≥ r 就出发，直接发命令 |
| 常数修正 | m 从小到大试：规划，路点按平均偏差反向挪，规划路线离真障碍物 ≥ 它自己的余量（开发场景上平移后误差的 95 % 分位）就出发 |
| 我们的 | m 从小到大试：规划，按模型做位置加朝向的补偿，预测补偿后的命令扫过哪里；预测的扫过路线离真障碍物 ≥ μ 就出发，发补偿后的命令 |
| 补偿后用固定余量（消融，不算对手） | 用一个固定的 m 规划，m 取开发场景上补偿后误差的 95 % 分位；补偿后总是出发。和"我们的"比，分开"补偿"和"每次伸手先查再走"各自的贡献 |

- **μ 怎么定：**开发场景上补偿回放的"实际扫过路线离预测扫过路线的最大距离"（腕 + 掌 + 指）的 95 % 分位。
  - 和第一部分不同，这里不再把"预测的偏差"和"残差半径"相加：预测的扫过路线本身已经带着偏差，只需要在它外面留残差。
  - 第一部分的规则 A 在这里也算一遍，作次要结果：残差半径按每次伸手预测，在开发场景上缩放。
- **余量都在开发场景上定：**用空地上的回放，只看这一类场景。第一部分对手的余量取自随机路线（含蹲下），约 19 cm，对它们不公平；这里改掉。
- **补偿用的模型：**ScaleBFM 第 2 轮模型 `via_point_onpol2_20261123`。它见过训练路线里桌沿、箱子、板下三类场景，没见过横杆。
  - 开发场景上先检查：模型对补偿回放的预测误差不超过对直接发命令那一遍的 2 倍。
  - 不过，就在另一批横杆训练场景（新种子，40 个底场景）上加一轮补偿器回放再训练，再检查一次。还不过就停下报告。
- **没有用仿真真值：**
  - 障碍物的几何直接给规划器，相当于完美的感知，五种做法相同；深度噪声的对照留给 38.6.2。
  - 手部伺服是关的。
  - 模型的输入和以前一样，骨盆位置来自腿部里程计。

### 执行和判定

- **真障碍物：**障碍物挂在一个 mocap 物体上。走路时停在远处，伸手开始时按机器人当时的站姿放到规划时的相对位置，和路线换坐标系的方式相同。
- **碰到的定义：**伸手开始到停稳之间，机器人任何碰撞体和障碍物有穿透（MuJoCo 的接触距离 < 0）。
  - 场景里手臂和手的碰撞体都是打开的（已核对）。
  - 主判定只看腕、掌、指；整条手臂另外报告。
  - 记录第一次接触的时刻、部位、最大穿透和接触力。
- **三种结局：**
  - 成功：出发了，没碰到；
  - 碰撞：出发了，碰到了；
  - 放弃：规划不出来或检查没过。
  - 另外报告手掌停稳后离目标多远。
- **主指标：**能安全通过的最窄的缝 f*，即成功 ≥ 90 %、碰撞 ≤ 5 % 的最小 f，区间按底场景自助抽样。同时画每种做法的成功率、碰撞率、放弃率随 f 变化的曲线。
- **"最好的对手"取下面四个里 f* 最小的：**
  - 标定过的固定余量；
  - 事后最好的固定余量（8 个 m 里在测试场景上 f* 最小的那个，是事后从测试集读出来的，对对手有利）；
  - 标量半径；
  - 常数修正。
- **第一部分的预测在这里检验：**同一条命令在空地上的回放加上障碍物的几何，用碰撞球算碰不碰，和真场景里 MuJoCo 的接触比。测试场景上每条出发了的命令都在空地上跑一遍。
- **跑之前的自检**（不过就先修，不计分）：
  - 障碍物放在远处时，真场景的回放和空地回放逐位相同；
  - 故意让手撞上的一次，接触被记到；
  - 规划器在加厚后的障碍物里规划出的路线，离真障碍物的距离不小于 m − 0.5 cm。

### 跑之前的第一步：只规划不回放的可行性探查（开发种子）

- **先量：**h 和每档 f 下各个 m 规划得出来的比例。
- **事先定的取舍：**
  - 最宽一档（f = 24 cm）上，m ≤ 4 cm 要在至少 60 % 的底场景规划得出来；
  - 最窄一档上，m = 1 cm 至少 30 %。
  - 达不到就只在开发种子上调桌沿距离、目标离桌沿的距离和横杆截面（只看规划比例，不看任何回放），调好后锁定，再生成测试场景。
- **规划失败的场景照实记下，不重抽。**

### 预测（ScaleBFM，测试场景）

- **规划：**最宽一档上 m ≤ 4 cm 在 40–80 % 的底场景规划得出来。m 只有在 f ≥ 2m 左右时才规划得出来。
- **固定余量：**
  - m = 1 cm 的碰撞率 60–95 %；
  - 标定过的固定余量 m 是 9–13 cm，它的 f* 是 20–28 cm 或者到不了；
  - 事后最好的固定余量 f* 是 12–20 cm。
- **我们的：**
  - μ 是 4–7 cm，选中的 m 中位数 4–7 cm；
  - 碰撞率每档都 ≤ 5 %；
  - f* 是 9–15 cm。
- **比值：**我们的 f* 是最好对手的 0.6–0.85 倍。
- **标量半径和常数修正：**都不如事后最好的固定余量。理由：第一部分里它们留的余量都比误差的分布宽。
- **第一部分的预测：**
  - 碰撞球的判定和 MuJoCo 的接触一致率 90–98 %；
  - 规划余量为 d 时，真场景的碰撞率不高于第一部分在 c = d 时的碰撞率，因为障碍物只在几个方向上。
- **把握最小的：**
  - 规划成功率。如果横杆让规划成功率掉得太多，每档能比的场景就少，区间会很宽。
  - 模型在横杆场景上会不会又遇到"补偿把命令推到没见过的地方"。

### 判据

- **主判据：**
  - 我们的 f* 不超过最好对手的 0.8 倍：真障碍物下"能穿过更窄的缝"成立，进入演示的设计（38.7）。
  - 超过 0.95 倍：真障碍物下没有优势。这条线只主张"误差降了、碰撞少了"，不主张"缝更窄"。
  - 两者之间：向用户报告。
- **最好的对手 f* 到不了的时候**（9 档里没有一档成功 ≥ 90 %）：
  - 我们的在某档成功 ≥ 90 %、碰撞 ≤ 5 %，而所有对手在同一档都做不到，就算成立；
  - 我们的也到不了：说明场景太难，回到开发种子重新调场景，不拿测试集调。
- **计分之前的检查：**
  - 摔倒比固定余量 m = 1 cm 那一组多 5 次以上，按失败处理；
  - 碰撞球判定和 MuJoCo 接触的一致率低于 90 %：先查几何再下结论；
  - 开发场景上的模型检查不过：先做上面那一轮补偿器回放；还不过，停下报告。

### 规模和时间（4090，1–3 号卡）

- **规划：**(20 + 30) 个底场景 × 9 档 × 8 个 m = 3600 次，约 1–2 小时。
- **回放：**ScaleBFM 每次开场约 2.1 s，24 个进程约每秒 11 次。
  - 开发场景空地回放三组（直接、补偿、常数），各 8 个 m，约 17000 次，约 30 分钟；
  - 测试场景真场景回放约 13000 次，加同样数量的空地对照，约 40 分钟。
- **代码（确认后写，先冒烟测试）：**
  - 场景生成和多余量规划：在 `fresh_routes.py` 的基础上写；
  - `gen_reach.py` 加两个开关：把 prep 里的障碍物放进仿真并记接触；按规则在几条候选路线里选一条；
  - 计分脚本。
  - 估计半天写完。

### 实现细节（2026-11-23，写代码时定，在任何规划和回放之前写下；不改上面的预测和判据）

- **"从小到大试余量"改成先全跑、后离线选，结果等价：**
  - 每个底场景、每档缝宽，8 个余量各自规划出的命令都在仿真里跑一遍；直接、常数修正、补偿三种发法各一组。
  - 同一底场景、同一遍，所有缝宽和余量的开场都从同一个状态出发：走路只由底场景和第几遍决定（`REACH_VIA_SLOTS`，走路键 = 底场景）。
  - 所以"站在这个状态、从小到大试余量、第一个通过检查的就出发"，结果就等于在那个余量的回放里读结局。补偿和预测在回放时按当时的状态算好并记下，离线选时直接用。
  - 前提是同一台机器上逐位重复。ScaleBFM 在 4090 上的这一点和 SONIC 一样，用第一项自检确认。
- **检查用的"规划路线离真障碍物的距离"：**用回放里参考扫过的路线（17 个球，相邻采样点之间再插 10 个点），到真障碍物的最小距离。
  - 对手的余量量的是"实际偏离参考的距离"，所以检查也用参考。cuRobo 自己算的距离只用于第三项自检。
- **种子：**
  - 底场景：开发 20261124，测试 20261125；横杆训练场景若要用，20261126。
  - 回放：开发 53000000，测试 59000000，横杆训练 61000000。
  - 都没用过。
- **可行性探查的取舍规则怎么读：**
  - "最宽一档 m ≤ 4 cm 要在至少 60 % 的底场景规划得出来"：m = 1、2、3、4 四个余量，每个的规划成功率都 ≥ 60 %。
  - "最窄一档 m = 1 cm 至少 30 %"：f = 3 cm、m = 1 cm 的规划成功率 ≥ 30 %。
- **h 怎么量：**到一个手水平的目标（身前 0.30 m、右 0.20 m、高 0.85 m，朝向为单位阵）在无障碍下规划一次，取终点 17 个球的竖直跨度（含半径）。
- **代码：**
  - `gap_routes.py`：场景和多余量规划；
  - `gap_scene.py`：带两个 mocap 障碍物的场景；
  - `gap_slots.py`：槽位表；
  - `gap_replay.sh`：回放；
  - `gen_reach.py` 新增 `REACH_VIA_SLOTS`、`REACH_GAP_OBST`，`via_gen.load_routes` 带上路线的障碍物。都默认关，关着时和原来一样跑。

### 可行性探查的结果（2026-11-23 07:37–07:43 UTC，开发种子 20261124，只规划）

`outputs/reachwm_20261030/gap_routes_dev`，20 个底场景 × 9 档 × 8 个余量 = 1440 次规划，4 个进程在 0 号卡上，每次 0.54 s。h = 8.70 cm（预测 8–11 cm）。

```
1440 plans, mean sec 0.54 ok 86
Counter({'MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION': 729, 'MotionGenStatus.IK_FAIL': 544, 'MotionGenStatus.FINETUNE_TRAJOPT_FAIL': 81, 'None': 52, 'MotionGenStatus.GRAPH_FAIL': 34})
f\m      1      2      3      4      6      8     10     12
  3  0/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
  5  0/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
  7  0/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
  9  0/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
 11  0/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
 14  1/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
 17  6/20   3/20   3/20   1/20   0/20   0/20   0/20   0/20
 20 10/20   9/20   5/20   4/20   0/20   0/20   0/20   0/20
 24 12/20  10/20   8/20   8/20   5/20   1/20   0/20   0/20
planned: clearance to the real obstacles below m - 0.5 cm on 0 of 86; to the thickened ones min 0.02 cm
```

**对照事先定的取舍：两条都没达到。**
- 最宽一档 m = 1–4 的成功率 60 / 50 / 40 / 40 %，要求每个都 ≥ 60 %。
- 最窄一档 m = 1 是 0 %，要求 ≥ 30 %。
- 第三项自检的规划部分过了：86 条规划离真障碍物都不小于 m − 0.5 cm。

**失败的原因**（`planwhy.py`；再用一份诊断副本，在无障碍物时解目标位姿，看各球离真桌板、真横杆多远）：
- **起点碰撞（729 次）：**只和桌面高度、余量有关，和缝宽无关。
  - 站立起点的手在身前约 0.2 m、0.72 m 高。桌面 0.70–0.73 m 的 4 个底场景连 m = 1 都失败（s07、s09、s19 是 72/72）。
  - 桌面高的，余量大了才碰。
- **目标位姿解不出（544 次）：**集中在窄缝和桌面高的场景。
  - 目标手形朝下 0–30°，手指往下伸。f = 5 cm 时，不加余量指尖就已在桌面以下 1.3–2.9 cm（s0、s4、s14、s16）；f = 11 cm 时只剩 0–2 cm。
  - "g = 手平放时的厚度 + f"这把尺子没算进手倾斜后多出的高度，所以 f ≤ 11 cm 全部失败。
  - 少数目标本身够不着，比如 s1：身前 0.364 m、偏中线 7 cm、手平放，没有障碍物也解不出。
- **事先允许调的只有桌沿距离、目标离桌沿的距离和横杆截面。**桌沿挪远能缓解起点碰撞；但指尖插进桌面这一条，三个旋钮都解决不了。
- **按规矩停下，向用户报告，等决定。**为了给出带数字的选项，在开发种子上只规划不回放，另跑了三个变体：
  - k1：只调允许的旋钮，桌沿 0.28–0.34 m，目标过桌沿 4–10 cm；
  - k2：k1 再加"每个底场景的手厚按它自己的目标朝向量（`--room goal`），手在缝里居中，上下偏移限在 ±f/4"；
  - k3：原来的桌沿和目标距离，只加 `--room goal`。
  - 测试种子没有碰过。

**三个变体的结果**（07:46–07:50 UTC，`gap_routes_dev_k1/k2/k3`，同样 1440 次规划）：

```
== k1 --edge 0.28,0.34 --past 0.04,0.10
 11  2/20   0/20 ...;  14  4/20   1/20 ...;  17  9/20   5/20   3/20   2/20 ...
 20  9/20   9/20   6/20   5/20   1/20   0/20   0/20   0/20
 24 10/20  10/20  10/20   8/20   5/20   1/20   0/20   0/20     (f <= 9: all 0)
 start collision m 1-3: 0/180 each; IK_FAIL 960 of 1440
== k2 --edge 0.28,0.34 --past 0.04,0.10 --room goal
--room goal: 8 of 20 goals reachable without obstacles; hand thickness at the goal attitude 12.8 to 17.8 cm (median 15.7)
  3  2/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
  5  7/20   2/20   0/20   0/20   0/20   0/20   0/20   0/20
  7  6/20   6/20   2/20   0/20   0/20   0/20   0/20   0/20
  9  8/20   6/20   4/20   2/20   0/20   0/20   0/20   0/20
 11  8/20   7/20   6/20   5/20   0/20   0/20   0/20   0/20
 14  8/20   7/20   8/20   7/20   4/20   0/20   0/20   0/20
 17  9/20   9/20   9/20   8/20   7/20   2/20   0/20   0/20
 20  9/20   9/20   9/20   9/20   9/20   5/20   2/20   0/20
 24 10/20  10/20  10/20   9/20   8/20   8/20   5/20   1/20
== k3 --room goal
--room goal: 11 of 20 goals reachable without obstacles; hand thickness at the goal attitude 9.0 to 17.7 cm (median 14.7)
  3  2/20 ...;  5  2/20   2/20 ...;  24 14/20  12/20  12/20   9/20   9/20   7/20   3/20   1/20
  start collision as in the first probe (m 1: 27/180 ... m 12: 171/180)
all three: planned clearance to the real obstacles below m - 0.5 cm on 0 plans
```

- k1（只调允许的旋钮）：起点碰撞在 m ≤ 3 时消失，但 f ≤ 9 仍然全部失败。允许的旋钮不够。
- k2：窄缝规划得出来了。剩下的上限是"够不着"：20 个目标里只有 8 个在没有障碍物时解得出（锁腰只动右臂），这类场景和窄缝无关。
- 手在目标朝向下的竖直厚度 9–18 cm，中位数 15 cm，比手平放时的 8.7 cm 厚得多。原设计按 8.7 cm 定缝宽，是指尖插进桌面的原因。

**按用户选的改法重跑探查（08:00–08:07 UTC，`gap_routes_dev2`，参数 `--edge 0.28,0.34 --past 0.04,0.10 --room goal --keep reachable`）：两条标准都达到，场景设置从此锁定。**

```
--room goal, --keep reachable: 81 drawn, 20 kept, 20 reachable; hand thickness at the goal attitude 11.2 to 18.8 cm (median 15.7)
1440 plans, mean sec 1.17 ok 603
f\m      1      2      3      4      6      8     10     12
  3  6/20   0/20   0/20   0/20   0/20   0/20   0/20   0/20
  5 18/20   5/20   0/20   0/20   0/20   0/20   0/20   0/20
  7 18/20  13/20   5/20   0/20   0/20   0/20   0/20   0/20
  9 17/20  17/20  11/20   5/20   0/20   0/20   0/20   0/20
 11 18/20  16/20  17/20  12/20   0/20   0/20   0/20   0/20
 14 20/20  19/20  19/20  18/20   9/20   0/20   0/20   0/20
 17 20/20  19/20  20/20  18/20  17/20   4/20   0/20   0/20
 20 20/20  20/20  20/20  20/20  19/20  11/20   4/20   0/20
 24 20/20  20/20  20/20  20/20  20/20  15/20  12/20   1/20
planned: clearance to the real obstacles below m - 0.5 cm on 0 of 603; to the thickened ones min 0.00 cm
start collision: m 1-6 0/180 each; m 8 45/180, m 10 63/180, m 12 99/180
```

- **对照标准：**
  - 最宽一档 m = 1–4 都是 20/20，要求 ≥ 60 %：过；
  - 最窄一档 m = 1 是 6/20 = 30 %，要求 ≥ 30 %：正好达到。
- **和预测对照：**
  - "最宽一档 m ≤ 4 在 40–80 % 的底场景规划得出"：结果 100 %，在区间外；
  - "m 只有在 f ≥ 2m 左右时才规划得出"：成立。
- **筛选的副作用：**留下的 20 个底场景桌面都在 0.77–0.89 m，低桌子全被筛掉了。
- **测试场景用同样的参数、种子 20261125、30 个底场景生成。**

**跑之前的三项自检（2026-11-23 08:02–08:14 UTC，4090，开发场景 `gap_routes_dev2` 最前面 24 条规划出来的路线，全在第 0 号底场景；种子 53000000，ID0 9100000，直接发命令，不计分）：三项都过。**

- **自检之零（额外加的）：新代码开关全关时和原来逐位相同。**重跑核对 4（`gap_check/default_off`），和 `data_via_fresh_4090` 比：

```
252 segments, 14975 steps, fell 0, walked 189, ok 125
== A against B, reset by reset: 252 resets in both (A 252, B 252)
  all                    palm's start apart: median 0.000 cm, 95 % 0.000, max 0.000; identical (< 0.001 cm) on 1.00
    reference error A - B: median +0.000 cm, |A - B| median 0.000, 95 % 0.000, max 0.000
```

- **第一项：障碍物放远时和原场景逐位相同。**`A_orig_far`（原场景 `simple_g1_sonic`）对 `B_gap_far`（带两个 mocap 物体的场景，`REACH_GAP_OBST=1`，障碍物在 10 m 外）：

```
== A against B, reset by reset: 96 resets in both (A 96, B 96)
  all                    palm's start apart: median 0.000 cm, 95 % 0.000, max 0.000; identical (< 0.001 cm) on 1.00
    reference error A - B: median +0.000 cm, |A - B| median 0.000, 95 % 0.000, max 0.000
    direction error A - B: median +0.000 cm, |A - B| median 0.000, 95 % 0.000, max 0.000
```

- **第二项：故意撞上的被记到。**`D_gap_hit`，横杆中心降到目标高度：

```
D_gap_hit: 96 reaches, fell 0; shapes touch (96, 3) t (96, 3) pen (96, 3) force (96, 3) t0 (96,)
  touch per column (through, arm, robot): [96, 96, 96] of 96
  first through-gap contact, s after placement: min 2.615, median 3.148, max 3.825
  deepest penetration per column, cm: [0.24, 0.24, 0.24]; largest force per column, N: [2770.1, 2770.1, 2770.1]
  contact already at placement: 0
```

- **第三项：规划路线离真障碍物不小于 m − 0.5 cm。**规划部分在探查里已经过了（603 条里 0 条低于）。按规划放真障碍物的回放 `C_gap_real` 也跑了：

```
C_gap_real: 96 reaches, fell 0
  touch per column (through, arm, robot): [0, 0, 0] of 96
== 24 个手臂球（arm_spheres.py）离真障碍物的最近距离，gap_check/clear.py
actual min over boxes, cm: median 5.19, min 2.57; below 0 on 0 of 96
wanted min over boxes, cm: median 4.52, min 2.39
Gap_s000_f03_m01 p0 | actual board 3.36, bar 4.72 | wanted 2.48 2.39
```

**看到的问题（不计分，只有一个底场景）：**
- **按计分用的 24 个球量，缝比名义上宽。**
  - 最窄一档 f = 3 cm、m = 1 cm 的规划路线离真障碍物还有 2.4 cm，名义上每边只多出 1.5 cm，m = 1 时规划器自己量的又只有 ≥ 0.5 cm。
  - 原因：手厚 h_s 和规划时的加厚都用规划器（cuRobo）自己那套碰撞球，比 24 个计分球胖。
- **第 0 号底场景上，直接发命令时实际扫过的路线反而比规划的离障碍物更远**（中位数 5.19 对 4.52 cm）。m = 1 cm 的 16 次伸手一次没碰。
- **这和事先的预测"固定 m = 1 cm 碰撞率 60–95 %"方向相反。**如果开发场景上普遍如此，各做法都在规划可行性上封顶，测试分不出高下。
- **下一步：**按设计在开发场景上跑空地回放定余量（`gap_dev`）。看这一步里小余量的碰撞情况（用空地回放的球算），如果小余量也几乎不碰，就停下问用户，不动测试场景的回放。

**开发场景的空地回放和定余量（2026-11-23 08:15–08:51 UTC，`gap_dev`，种子 53000000；直接 9200000、常数修正 9300000、补偿 9400000，各 18 进程，1–3 号卡）：模型检查过了；但小余量在开发场景上几乎不碰，测试分不出高下。按事先说的停下问用户，测试场景的回放没跑。**

三组各 2412 次伸手（603 条规划出来的路线 × 4 遍），0 摔倒。`gap_real.py calib`（`gap_dev/calib.out`）：

```
through, direct: 2412 reaches, median 4.11 cm, 90 % 6.69, 95 % 7.46
through, const: 2412 reaches, median 4.56 cm, 90 % 7.13, 95 % 7.85
through, comp: 2412 reaches, median 1.99 cm, 90 % 3.61, 95 % 4.73
through, comp against the predicted sweep: 2412 reaches, median 1.02 cm, 90 % 2.41, 95 % 4.04
 "fixed_cal_cm": 8.0, "const_cm": 7.845600830829588, "mu_cm": 4.0410499463369955, "comp_fixed_cm": 6.0,
 "scale_radius": 0.8327264249465861, "scale_rule_a": 2.4929455998039836
```

模型检查（`via_comp_eval.py`，`gap_dev/comp_eval.out`）：

```
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
Gap: 603 routes, 2412 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 4.11 cm, compensated 1.99 cm; drop 0.52 (0.50 to 0.54); 90 % level 6.69 and 3.61 cm
  the model's own error (arm) on the run it predicted: direct 0.96 cm, compensated 1.05 cm
```

模型误差 1.05 / 0.96 = 1.09 倍，不超过 2 倍：过。

实际扫过的路线（计分球，腕 + 掌 + 指）离真障碍物的距离，按余量 m 分（`gap_dev/look.py`，`look.out`）。自检里 MuJoCo 第一次接触时计分球离障碍物还有 0.1–0.6 cm，所以"< 0.5 cm"更接近 MuJoCo 的碰到：

```
== direct_open: 2412 reaches, fell 0, in the table 2412
  through clearance, cm: actual median 5.03, 5 % 2.22; wanted median 4.88, 5 % 2.14; actual - wanted median +0.21
  by margin m: n | actual through < 0 | < 0.5 cm | arm (24) < 0 | actual through clearance median, cm
    m  1:  628 | 0.000 | 0.002 | 0.000 |  3.17
    m  2:  516 | 0.000 | 0.000 | 0.000 |  4.38
    m  3:  448 | 0.000 | 0.000 | 0.000 |  5.53
    m  4:  372 | 0.000 | 0.000 | 0.000 |  6.49
    m  6:  260 | 0.000 | 0.000 | 0.000 |  8.36
    m  8:  120 | 0.008 | 0.017 | 0.008 | 10.34
    m 10:   64 | 0.031 | 0.078 | 0.031 | 11.56
    m 12:    4 | 0.000 | 0.000 | 0.000 | 11.97
== const_open:  m 1: 628 | 0.000 | 0.000 | 0.000 | 3.12
== comp_open:   m 1: 628 | 0.000 | 0.000 | 0.000 | 2.80
```

实际比规划离障碍物近了多少（`gap_dev/look2.py`，`look2.out`）：

```
== direct_open: actual minus wanted through clearance (negative: the hand came closer to an obstacle than planned)
  all            n 2412: quantiles 1/5/10/50 % -1.77 -1.14 -0.89 +0.21 cm; < -1 cm 0.074, < -2 cm 0.010; wanted through clearance median 4.88, 5 % 2.14
  m 1            n  628: quantiles 1/5/10/50 % -1.38 -0.95 -0.75 +0.30 cm; < -1 cm 0.041, < -2 cm 0.000; wanted through clearance median 2.84, 5 % 1.91
== const_open:   m 1 n 628: quantiles 1/5/10/50 % -1.19 -0.92 -0.76 +0.29 cm; < -1 cm 0.037, < -2 cm 0.000
== comp_open:    m 1 n 628: quantiles 1/5/10/50 % -0.77 -0.50 -0.39 -0.06 cm; < -1 cm 0.000, < -2 cm 0.000
```

**读法：**
- **事先预测"固定 m = 1 cm 碰撞率 60–95 %"，开发场景上按球算是 0 / 628。**
  - 偏差本身不小：直接发命令腕 + 掌 + 指中位数 4.11 cm，95 % 分位 7.46 cm。
  - 但朝障碍物方向的分量很小：最差的 1 % 也只比规划近 1.38 cm。偏差大多沿着障碍物表面，或离开障碍物。
- **设计依据的前提"规划器贴着障碍物走，直接发命令几乎必碰"在这个场景里不成立。**
  - 那个前提来自第一部分越过桌沿、越过箱子的路线：误差中位数 8 cm，且没有区分方向。
  - 这里的路线短，起点就在桌沿前，误差只有一半。
  - 规划器的球又比计分球胖，规划路线自带约 2 cm 空隙。
- **对测试结果的含义（按开发场景推断，没有在测试场景上验证）：**
  - 事后最好的固定余量会是 m = 1 cm，它几乎不碰，f* 只受"规划得出来"限制。
  - "我们的"要求预测扫过路线离障碍物 ≥ μ = 4.04 cm，标定过的对手要 7.5–8 cm 余量，都比需要的保守得多。
  - "我们的"大概率输给事后最好的固定余量，赢标定过的对手，但赢的原因是对手的余量量的是不分方向的误差，不是模型准。
- **即使把规划器的球改小、让缝变窄：**直接发命令 m = 1 cm 时比规划近 1 cm 以上的只有 4.1 %，能拉开的差距也只有 1 cm 左右，小于缝宽刻度的一档（2–3 cm）。
- **测试场景（`gap_routes_test`，种子 20261125）只规划了，回放没跑，规划统计也没看。**

**用户的决定（2026-11-23，看完上面开发场景的结果后）："停在这里，记下发现"。**
- 测试场景不回放。`gap_routes_test` 只规划过，种子 20261125 的回放结果不存在，以后另做时可以照用这组场景，也可以换种子。
- 第二部分在这个场景上没有结论：分不出各做法的高下。
- 留下的发现（开发场景，ScaleBFM，计分球）：
  - 偏差大，但很少朝向障碍物；
  - 按不分方向的误差定余量（标定的固定余量 8 cm、常数修正 7.85 cm、μ 4.04 cm）比实际需要的 1 cm 保守得多。
- 要在第二部分上分出高下，场景得让偏差朝向障碍物，例如长路线、越过箱子后再进缝，或者绕到障碍物侧面。这是新设计，由用户决定做不做。

**独立复核（2026-11-23）：判定"不支持（按原表述）"。**独立 agent（general-purpose，新上下文），只给它设计、结果段落、原始输出和代码的路径，明确不许打开测试场景。它的回复原样如下：

> **结论：不支持（按原表述）**
>
> 发现里有一部分站得住：按不分方向的误差定的余量（8、7.85、4.04 cm）确实比需要的大得多。但"1 cm 余量下几乎碰不到""1 cm 就够"这两句不成立。原因是 24 个计分球比 MuJoCo 里手的真实碰撞体薄很多，碰撞被系统性漏掉了。
>
> ## 1. 数字核对
>
> 两段里引用的数字和原始文件全部对得上：
> - `calib.out` / `calib.json`：4.11 / 7.46、7.85、μ 4.04 等；
> - `comp_eval.out`：模型误差 1.05 / 0.96；
> - `look.out`、`look2.out`；
> - 自检的 `summary.txt` 和 `AB_match.txt`；
> - 我重跑了一遍 `clear.py`：中位数 5.19 / 4.52 cm，最小 2.57 / 2.39 cm。
> - 规划统计 603/1440、起点碰撞数、h_s 11.2–18.8 cm 也一致。
>
> 有一处措辞不准：−1.38 cm 是 1 % 分位数，不是"最多近 1.4 cm"。最极端的一次近了 1.57 cm，有 6 次超过 1.4 cm。
>
> ## 2. 独立重算
>
> 我没有用计分球，而是在 MuJoCo 里把记录下来的每个 10 Hz 状态（关节、骨盆、手指）和真障碍物摆回去，跑一次 `mj_fwdPosition`，用和 `GapObstacles.cb` 同样的规则（dist < 0）判碰撞。机器人和障碍物都在世界系里摆放，不经过航向坐标系的换算。
>
> **你点名要查的几类错误都排除了：**
> - 用的是 `boxes`，不是 `boxes_plan`；
> - tick 区间是指令时刻到记录结束，和 MuJoCo 记接触的区间相同；
> - 走路后那 3 遍的障碍物放置也正确。
> - 证据：这套复现和自检的真实记录逐条一致，C 0/96、D 96/96。C 的 S、W 数组和空地回放逐位相同。
>
> **问题出在计分球的覆盖：**
> - 把手的凸包顶点和 17 个球的并集比，最多露在外面：手掌 3.4 cm，拇指尖 1.7 cm，指尖 1.2 cm，腕 1.5 cm。
> - 在 D（MuJoCo 96 次都记到接触）里，计分球离障碍物的最近距离始终没低于 0，最小也有 0.10–0.76 cm。
>
> **开发场景空地回放，按 MuJoCo 真实几何算的碰撞率：**
>
> | 余量 m | 直接发命令 | 常数修正 | 补偿 |
> |---|---|---|---|
> | 1 cm | **20/628 = 3.2 %** | 2.9 % | 0.6 % |
> | 2–4 cm | 0 | 0 | 0（m = 3 有 1 次左手碰） |
> | 10 cm | 7.8 %（只算右手腕掌指）；15.6 %（算整个身体，主要是左手撞桌板） | — | — |
>
> m = 1 直接发命令的 20 次碰撞：
> - 全是右手拇指尖碰桌板，最深 0.89 cm；
> - 全部在第 0 遍（原地出发），占第 0 遍的 20/157 = 12.7 %；走路后的三遍是 0；
> - 集中在 5 个底场景；
> - 计分球一次都没判出来（0/20）。
> - 10 Hz 采样之间的短暂碰撞会漏掉，所以这些数是下限。
>
> **方向那部分站得住：**真实几何下"比规划近多少"的 1 % 分位是 −1.25 cm（计分球 −1.38 cm）。会碰的原因是规划的参考路线本身离障碍物就近：m = 1 时按真实几何最近只剩 0.34 cm（5 % 分位 0.55 cm），计分球却说还有约 1.9 cm。规划时加的余量 m 不等于真实的空隙。
>
> ## 3. 四类陷阱
>
> - **基准选择：**
>   - 探查后改场景（桌沿挪远、只留够得着的目标）有用户批准、在回放之前，但改完只剩高桌子（0.77–0.89 m），偏差朝向障碍物的场景被筛掉了。
>   - 底场景只有 20 个，碰撞又集中在其中 5 个，0/628 不能当 628 个独立样本看。
>   - 把走路后的三遍（平均离障碍物更远）和第 0 遍合在一起统计，冲淡了第 0 遍的问题：第 0 遍实际比规划近，中位数 −0.69 cm。
> - **泄漏：**`gap_dev`、`gap_check` 里没有引用 `gap_routes_test`，我也没打开它。没看到泄漏的迹象，但"没看过"本身无法完全证实。
> - **指标误用（最主要）：**
>   - 用计分球的空地距离代替 MuJoCo 接触。可"计分球判碰撞准不准"正是事先登记要检验的第一部分预测，等于拿待检验的东西当量尺。
>   - 登记的"一致率 ≥ 90 %"放在这里会通过（96.8 %），但真碰撞一次没抓到（0/20）。碰撞稀少时，一致率这个指标本身就不对。
>   - "需要 1 cm"其实只是余量刻度的最小一档，不是测出来的需要量。
> - **事后选择：**"小余量几乎不碰就停下"这条规则是看过第 0 号底场景的自检之后才写的，不在预注册里。不过停下是保守做法，用户明确说了"停在这里，记下发现"，测试集也没动过，所以停在测试之前和预注册、用户指示都一致。只是停下的理由有一部分是错的。
>
> ## 4. 其他解释和能推翻它的实验
>
> **其他解释：**
> 1. 计分球太薄（已证实）。
> 2. 起始状态决定偏差方向：第 0 遍靠近障碍物，走路后的几遍远离。
> 3. 障碍物是水平板加 3 cm 深的横杆，只有竖直方向的偏差要紧。沿路线方向的滞后和停在目标之前（手掌终点差 1.6 cm）都平行于表面，"安全"有一部分是因为手没伸到位。
> 4. "真障碍物在接触之前就改变控制器的动作"：已排除。控制器看不到障碍物，接触余量为 0，C 和空地回放逐位相同。接触之后动作才会变（D 里力达到 2770 N），这影响碰撞的后果，不影响"第一次碰没碰到"。
> 5. 规划器只建模右臂；余量大时左手会撞上桌板，计分球也只覆盖右臂，所以看不到。
>
> **能推翻发现的实验（需要 GPU，要你点头）：**
> - 只用开发场景，`REACH_GAP_OBST=1` 真放障碍物，跑直接发命令和补偿两组，m = 1–4，约 2000 次伸手，几分钟。
> - 按我的复现预测：直接发命令 m = 1 约 3 %（第 0 遍约 13 %，拇指尖），m ≥ 2 约 0；补偿 m = 1 约 0.6 %。
> - 如果成立，"1 cm 足够"就被推翻；而且直接发命令和补偿在小余量下其实分得开（3.2 % 对 0.6 %），"测试分不出高下"也要改写。
> - 按开发场景粗算，m = 1 在 f = 20 的碰撞率 5/80 = 6.25 %，超过 5 % 的线；"事后最好的固定余量"的 f* 从 5 变成 7。
> - 同时把计分几何换成 MuJoCo 的凸包（或者更胖的球），重新定所有余量。
>
> ## 5. 判定理由
>
> "偏差朝向障碍物的分量小"站得住（真实几何下 1 % 分位 −1.25 cm），"不分方向的余量过于保守"也站得住：真实几何下 m = 2 就没有碰撞，远小于 4–8 cm。但"m = 1 几乎不碰""实际只需 1 cm"在 MuJoCo 真实几何下不成立，"测试分不出高下"也被削弱。
>
> 我没有在输出目录、仓库或本地项目里写任何东西，也没跑仿真或 GPU 任务。远程的脚本和结果在 `/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/tmp/review_gap/`：`mjcoll_all.py`、`geomclear.py`、`cover2.py`，结果在 `coll/` 和 `res/`。

**我的抽查**（`gap_routes_test` 以外，只读；脚本 `review_gap/rv_check.py`）：

```
D: 96 reaches reconstructed; touch thr 96; recorded touch 96
  reconstructed - recorded: median +0.075, spread (max - min) 0.365 s
== direct: 2412 reaches from 10 parts
  m  1: through touch 20/628 = 0.032; pass 0 20/157; walked 0/471; base scenes [9, 12, 13, 14, 19]; deepest 0.89 cm
  m  2: through touch 0/516 = 0.000
  m  3: through touch 0/448 = 0.000
  m  4: through touch 0/372 = 0.000
  m  6: through touch 0/260 = 0.000
  m  8: through touch 1/120 = 0.008; pass 0 1/30; walked 0/90; base scenes [9]; deepest 2.31 cm
  m 10: through touch 5/64 = 0.078; pass 0 1/16; walked 4/48; base scenes [1, 9]; deepest 1.21 cm
== const: m 1: through touch 18/628 = 0.029; pass 0 18/157; walked 0/471; base scenes [9, 12, 13, 14]
== comp:  m 1: through touch 4/628 = 0.006; pass 0 3/157; walked 1/471; base scenes [13]; deepest 0.34 cm
          m 3: 2/448; m 6: 1/260; m 10: 5/64
```

- **还原是准的：**还原出的第一次接触时间比运行时 MuJoCo 实际记到的晚 0.075 s（中位数），在 10 Hz 记录间隔之内。
- **复核目录里的 `test_direct.npz`** 是它对 20 条开发路线的调试输出（s = 0，f = 3），和测试场景无关。

**更正后的发现**（取代上面"读法"里与此冲突的说法；开发场景，ScaleBFM，按 MuJoCo 真实碰撞体还原，是下限）：
- **"m = 1 cm 几乎不碰""1 cm 就够"不成立。**
  - 直接发命令 m = 1 cm 碰 3.2 %，全在原地出发那一遍（12.7 %），走路后三遍是 0；
  - m = 2–6 cm 一次没碰。
  - 补偿把 m = 1 cm 的碰撞降到 0.6 %。
- **24 个计分球没包住手：**手掌最多露出 3.4 cm，拇指尖 1.7 cm，真碰撞计分球一次也没判出来（0/20）。
  - 设计里"标量半径""常数修正""我们的"三条规则都用这套球算离障碍物的距离，计分脚本里"空地上的球和 MuJoCo 接触对不对得上"也用它。照原样跑测试，这些都有偏差。
  - 以后另做，要先把计分几何换成 MuJoCo 的凸包（复核的 `mjcoll_all.py`、`geomclear.py` 可以用），再重新定所有余量。
- **站得住的部分：**
  - 偏差朝向障碍物的分量小，真实几何下 1 % 分位 −1.25 cm；
  - 按不分方向的误差定的余量（8、7.85、4.04 cm）远大于需要，真实几何下 m = 2 cm 就没有碰撞。
- **停下仍然合适，但理由改了：**不是"分不出高下"，而是计分几何有偏差，测试照原样跑结果不可信。用户的决定（停在这里）是按更正前的说法做的，已向用户报告更正。

### 补充：计分用的手换成包住真实碰撞体的球，开发场景放真障碍物（2026-11-23，4090；用户："把这三件事都做了"；本节在任何运行之前写下）

**为什么：**复核发现 24 个计分球没包住手（手掌最多露 3.4 cm），按真实碰撞体还原后，直接发命令 m = 1 cm 碰 3.2 %。三条规则和"球判碰撞准不准"那项检查都用这套球，照原样跑测试结果不可信。

**做法：**
1. **包住真实碰撞体的球（`gap_cover.py`）：**
   - 每个碰撞体所在的连杆上都有计分球：腕俯仰、腕偏航（手掌的 3 个球在它上面）、7 节手指。
   - 在连杆自己的坐标系里，取 MuJoCo 碰撞体凸包的顶点，加上每个三角面上的采样点，各归到同一连杆上"表面最近"的那个球，把球的半径放大到盖住归给它的点。圆心不动。
   - 肩、肘的球按同样办法处理；碰撞体所在连杆上没有球的，照实列出，那部分"整条手臂"的数字不算数。
   - 结果写进 `cover.json`。
   - 半径只影响"离障碍物多远"的计算。模型、误差、μ 和各对手的余量都按球心算，不变。
2. **计分脚本：**`gap_real.py` 加 `--radius cover.json`，用新半径算所有离障碍物的距离。默认不变。
3. **开发场景放真障碍物回放：**直接、常数修正、补偿三组，`REACH_GAP_OBST=1`，开发场景 `gap_routes_dev2`。
   - 种子 53000000，进程数和卡和空地那三组相同，所以碰到之前应和空地那一遍逐位相同。
   - ID0：直接 9500000、常数修正 9600000、补偿 9700000。
4. **开发场景上计分（`gap_real.py score`，用新半径）：**
   - 各做法每档缝的成功率、碰撞率、放弃率和 f*；
   - "空地上的球判碰不碰"和 MuJoCo 接触的一致情况。
   - 只是演练，不代替测试。
5. **测试场景不碰。**这一步之后向用户报告，由用户定要不要用修好的计分在测试场景上跑。

**预测：**
- **MuJoCo 接触（腕 + 掌 + 指，放真障碍物那一遍）：**
  - 直接发命令 m = 1 cm 碰 3–6 %，复核按 10 Hz 帧还原的 3.2 % 是下限，绝大多数在原地出发那一遍；
  - 常数修正 2.5–6 %；补偿 0–2 %；
  - m = 2–6 cm 各做法都 ≤ 0.5 %。
- **新半径：**
  - 手掌的球放大 2–4 cm，指尖的放大 1–2 cm；
  - 空地上用新半径判"会碰"，能抓到 MuJoCo 接触的全部（召回 100 %），m = 1 cm 时误报 10–40 %。
- **开发场景的 f*：**
  - 事后最好的固定余量是 m = 2 cm（m = 1 在某些档碰撞超过 5 %），f* 7–9 cm；
  - "我们的"用新半径后 f* 不比它小，比值 ≥ 1；
  - 标定过的固定余量（8 cm）f* ≥ 20 cm。

**判据：**
- 直接发命令 m = 1 cm 的 MuJoCo 接触率 < 1.5 % 或 > 10 %：和复核的还原对不上，先查原因，不往下做。
- 新半径漏掉任何一次深于 0.2 cm 的 MuJoCo 接触：球还没包住，先修，不用来计分。
- 没有接触的伸手，和空地那一遍不逐位相同：执行有问题，先查。

### 补充：方法改动（"包住真实碰撞体的球"怎么算）

- 时间说明：本节写在新半径算出、并已在开发场景上演练计分之后（计分在真障碍物回放跑完后自动接着跑），测试场景之前。测试场景没有碰。
- 原写法"在连杆自己的坐标系里归点"试过，不能用（`gap_cover/cover_bodyframe_tried.out`）：
  - 腕偏航连杆上既有手掌，也有腕部外壳，外壳离手掌的球约 10.5 cm。只能归给同连杆的手掌球，手掌的 3 个球从 1 cm 放大到 11.5 cm。
  - 腕横滚连杆上的碰撞体所在连杆没有球，盖不住。
- 改成按姿态拟合（`gap_cover.py` 的说明里写全了）：
  - 从开发场景空地那三组回放里抽 2000 个手臂姿态：一半实际记录的，一半参考姿态，手指用各段的手指角度。
  - 腕和手的碰撞体（腕俯仰、腕偏航、手指连杆上的）归 17 个"腕 + 掌 + 指"球；肩、肘、腕横滚的碰撞体归 7 个肩肘球。
  - 每个姿态里，每个点归给同组里表面最近的球，球放大到盖住它；过两遍。
  - 另抽 1000 个姿态检查覆盖。圆心不动。

### 补充：结果（2026-11-23，09:32–10:44 UTC）

新半径（`gap_cover/cover.out`，CPU，9 分 45 秒）：

```
15 geoms (10 through), uncovered []; fit on 2000 poses, check on 1000 others, from 3 runs
  through largest distance of a point outside the union per pose, cm: cuRobo's radii {'q50': 3.42, 'q95': 3.425, 'q99': 3.428, 'q100': 3.445}; the cover's {'q50': -0.0, 'q95': -0.0, 'q99': -0.0, 'q100': 0.009}
  arm     largest distance of a point outside the union per pose, cm: cuRobo's radii {'q50': 5.041, 'q95': 5.055, 'q99': 5.055, 'q100': 5.055}; the cover's {'q50': 0.022, 'q95': 0.022, 'q99': 0.022, 'q100': 0.022}
   6    elbow_link            3.82 ->  8.87  (+5.05)
   7    wrist_pitch_link      3.11 ->  4.01  (+0.89)
   8    wrist_pitch_link      3.11 ->  6.60  (+3.48)
   9    hand_palm_link        1.00 ->  3.66  (+2.66)
  10    hand_palm_link        1.00 ->  4.42  (+3.42)
  11    hand_palm_link        1.00 ->  4.42  (+3.42)
  12    hand_thumb_0_link     1.00 ->  4.40  (+3.40)
  15    hand_thumb_2_link     1.00 ->  2.72  (+1.72)
  18    hand_middle_1_link    1.00 ->  2.05  (+1.05)
  22    hand_index_1_link     1.00 ->  2.05  (+1.05)
```

真障碍物回放：三组各 2412 次，18 个进程全部退出码 0，0 摔倒。MuJoCo 自己记的接触（`gap_dev/obst_touch.out`）：

```
== direct_obst: 2412 reaches, fell 0, touched through / arm / any [23, 23, 36]
  m  1: through  22/628 = 0.035 (pass 0 22/157, walked 0/471); arm 22, any 22; base scenes [3, 9, 12, 13, 14, 19]; deepest through 0.06 cm
  m  2: through   0/516 = 0.000 ...   m 3: 0/448   m 4: 0/372   m 6: 0/260   m 8: 0/120
  m 10: through   1/ 64 = 0.016 (pass 0 1/16, walked 0/48); arm 1, any 10; base scenes [9]; deepest through 0.00 cm
== const_obst: 2412 reaches, fell 0, touched through / arm / any [19, 19, 28]
  m  1: through  19/628 = 0.030 (pass 0 19/157, walked 0/471); arm 19, any 19; base scenes [9, 12, 13, 14]; deepest through 0.06 cm
  m  2 to 12: through 0
== comp_obst: 2412 reaches, fell 0, touched through / arm / any [10, 10, 25]
  m  1: through   8/628 = 0.013 (pass 0 6/157, walked 2/471); arm 8, any 8; base scenes [13, 14]; deepest through 0.03 cm
  m  3: through   2/448 = 0.004 (pass 0 2/112, walked 0/336); arm 2, any 2; base scenes [11]; deepest through 0.10 cm
  other m: through 0
```

没碰的伸手和空地那一遍逐位比较（`gap_dev/obst_twin.out`，比 S 和 W 两个数组）：

```
== direct: 2412 reaches, compared ['S', 'W']; untouched 2376: not bit-identical to the open twin 0 []
  touched 36: first differing step after the command, median 28, min 21; first contact (s) median 2.46, min 1.72
  touched by another robot body only: 13; of them at placement 0; first contact (s) [3.46, 4.32, 5.04, 4.61, 4.94, 4.96, 5.14, 4.82, 4.92, 4.45, 5.2, 5.09, 5.16]
== const: ... untouched 2384: not bit-identical to the open twin 0 []
  touched by another robot body only: 9; of them at placement 0; first contact (s) 3.52 to 5.28
== comp: ... untouched 2387: not bit-identical to the open twin 0 []
  touched by another robot body only: 15; of them at placement 0; first contact (s) 3.84 to 5.56
```

空地上的球判"会碰"和 MuJoCo 接触，按余量分开（`gap_dev/sphere_by_m.out`，只看腕 + 掌 + 指）：

```
== direct: m  1 (628, MuJoCo touch 22): cuRobo: inside 0, touch not inside 22, inside no touch 0; cover: inside 47, touch not inside 0, inside no touch 25 (false alarm 0.53)
           m  8 (120, MuJoCo touch 0): cuRobo: inside 1, ... inside no touch 1; cover: inside 2, inside no touch 2
           m 10 (64, MuJoCo touch 1): cuRobo: inside 2, touch not inside 0, inside no touch 1; cover: inside 7, touch not inside 0, inside no touch 6
== const:  m  1 (628, MuJoCo touch 19): cuRobo: inside 0, touch not inside 19; cover: inside 35, touch not inside 0, inside no touch 16 (false alarm 0.46)
== comp:   m  1 (628, MuJoCo touch 8): cuRobo: inside 0, touch not inside 8; cover: inside 44, touch not inside 0, inside no touch 36 (false alarm 0.82)
           m  2 (516, MuJoCo touch 0): cover: inside 3, inside no touch 3
           m  3 (448, MuJoCo touch 2): cuRobo: inside 2, touch not inside 0; cover: inside 2, touch not inside 0
           m 10 (64, MuJoCo touch 0): cuRobo: inside 4, inside no touch 4; cover: inside 6, inside no touch 6
```

开发场景演练计分：`gap_dev/score_cover.out` 用新半径，`gap_dev/score_curobo.out` 用原半径。
- 脚本里的标签写 "test scenes"，其实算的是开发场景 `gap_routes_dev2`。
- 两边的标定 json 逐字相同：标定按球心算，不受半径影响，所以"重新定余量"之后余量不变。
- 碰撞按真障碍物那一遍里 MuJoCo 的接触算，所以固定余量的各行两边完全相同；只有用预测距离做决定的四种做法（radius、const、ours、rule_a）不同。

```
                              3               5               7               9              11              14              17              20              24
  fixed_m1       0.30/0.00/0.70  0.89/0.01/0.10  0.90/0.00/0.10  0.82/0.03/0.15  0.86/0.04/0.10  0.94/0.06/0.00  0.96/0.04/0.00  0.94/0.06/0.00  0.96/0.04/0.00
  fixed_m2       0.00/0.00/1.00  0.25/0.00/0.75  0.65/0.00/0.35  0.85/0.00/0.15  0.80/0.00/0.20  0.95/0.00/0.05  0.95/0.00/0.05  1.00/0.00/0.00  1.00/0.00/0.00
 新半径:
  ours           0.00/0.00/1.00  0.00/0.00/1.00  0.00/0.00/1.00  0.00/0.00/1.00  0.20/0.00/0.80  0.85/0.00/0.15  0.95/0.00/0.05  1.00/0.00/0.00  1.00/0.00/0.00
  rule_a         0.00/0.00/1.00  0.03/0.00/0.97  0.28/0.00/0.72  0.65/0.00/0.35  0.79/0.00/0.21  0.86/0.00/0.14  1.00/0.00/0.00  1.00/0.00/0.00  1.00/0.00/0.00
 原半径:
  ours           0.00/0.00/1.00  0.00/0.00/1.00  0.19/0.00/0.81  0.49/0.00/0.51  0.85/0.00/0.15  0.95/0.00/0.05  1.00/0.00/0.00  1.00/0.00/0.00  1.00/0.00/0.00
  rule_a         0.12/0.00/0.88  0.54/0.00/0.46  0.88/0.00/0.12  0.84/0.01/0.15  0.86/0.00/0.14  0.99/0.01/0.00  0.97/0.03/0.00  1.00/0.00/0.00  1.00/0.00/0.00

f*                新半径                         原半径
  fixed_m1        7.0  (5.0 to 24.0)            7.0  (5.0 to 24.0)
  fixed_m2       14.0  (9.0 to 20.0)           14.0  (9.0 to 20.0)
  fixed_cal       nan  (24.0 to 24.0), not reached 0.92    same
  radius          nan  (24.0 to 24.0), not reached 0.83    24.0  (20.0 to 24.0)
  const           nan  (24.0 to 24.0), not reached 0.96    20.0  (20.0 to 24.0), not reached 0.11
  ours           17.0  (14.0 to 20.0)           14.0  (11.0 to 17.0)
  rule_a         17.0  (11.0 to 17.0)           14.0  (7.0 to 14.0)
  best opponent: fixed_m1; ours / best: 2.43 (1.00 to 4.00)  |  2.00 (1.00 to 3.40)
margin picked: ours went 0.44, margin 4.0 cm  |  went 0.61, margin 2.0 cm
open-floor spheres against MuJoCo, direct through: agree 0.986, touch but not inside 0.000, inside but no touch 0.014  |  agree 0.990, touch but not inside 0.009, inside but no touch 0.001
```

**和预测对照：**

| 预测 | 结果 | 在区间里吗 |
|---|---|---|
| 直接 m = 1 碰 3–6 %，多数在原地出发那一遍 | 3.5 %，22 次全在原地出发那一遍 | 在 |
| 常数修正 m = 1 碰 2.5–6 % | 3.0 % | 在 |
| 补偿 m = 1 碰 0–2 % | 1.3 % | 在 |
| m = 2–6 各做法 ≤ 0.5 % | 补偿 m = 3 是 0.45 %（2/448），其余 0 | 在（贴边） |
| 手掌的球放大 2–4 cm | +2.66 到 +3.42 cm | 在 |
| 指尖放大 1–2 cm | +1.05 到 +1.72 cm | 在 |
| 召回 100 % | 三组都没漏 | 在 |
| m = 1 误报 10–40 % | 46 % / 53 % / 82 % | 不在（更保守） |
| 事后最好的固定余量是 m = 2，f* 7–9 | 是 m = 1，f* 7；m = 2 的 f* 是 14 | 不在 |
| "我们的"比值 ≥ 1 | 2.43 | 在 |
| 标定过的固定余量 f* ≥ 20 | 92 % 的重抽样里到 24 cm 都没过 | 在 |

**对照判据：**三条都没触发。
- 直接 m = 1 碰 3.5 %，在 1.5–10 % 之间；
- 新半径一次真实接触都没漏；
- 没碰的伸手和空地那一遍全部逐位相同。

**意外：**
1. **原来的球在 1 cm 余量下一次都没抓到。**三组共 49 次真实接触，原半径判"会碰"的是 0 次。复核的发现在这里完全坐实。
2. **f* 的定义让固定 1 cm 余量当了"最好的对手"。**
   - 它在 7 cm 那档过了线（0.90 / 0.00），但 14 cm 和 20 cm 两档碰撞 6 %，超过 5 %。
   - f* 只看"最窄的一档过线"，不要求更宽的也过。
   - 预测里写了"m = 1 在某些档碰撞超过 5 %"，但没想到定义会让它照样胜出。
3. **新半径让"我们的"更保守，却没换来更安全。**
   - f* 从 14 cm 退到 17 cm，选的余量从 2 cm 变成 4 cm，碰撞两边都是 0。
   - 原因：这一部分的碰撞由 MuJoCo 直接判，半径只进"我们的"做决定那一步。新球比真实碰撞体大（误报一半以上），所以更早放弃。
4. **右臂以外的身体部位碰到障碍物**：三组 13 / 9 / 15 次。
   - 都在发命令后 3.4–5.6 s，伸手已经结束；不是一放障碍物就碰上的。（复核更正：伸手还没结束，是左手手指碰桌板，见下面复核第 2 条。）
   - 余量大的场景多（直接发命令 m = 10 有 10 次），具体是哪个部位没有记。
   - 现在的成功定义不算这种接触。
5. **原半径在 m = 8 / 10 也有判"会碰"、但 MuJoCo 没碰的**（直接 1+1、常数修正 1+2、补偿 4）。原半径的球比真实碰撞体小，照理不该出现，可能是有的球伸出了碰撞体外。没查。

**结论与下一步（独立复核前）：**
- **复核的数字确认了。**真实接触率直接 3.5 %、补偿 1.3 %，和复核按帧还原的 3.2 % / 0.6 % 同一量级，略高，因为现在是每个仿真步都记。1 cm 余量下原来的球全部漏判。
- **新半径能作为"不漏判"的球**，但误报一半以上。
- 在真障碍物测试里，计分本来就用 MuJoCo 的接触；球只影响"我们的"那几种做法怎么做决定。所以"换计分几何"对第二部分的计分没有影响，影响的是做法本身。用新球在开发场景上让"我们的"变差（f* 14 → 17）。
- **上测试场景之前要用户定三件事：**
  1. "我们的"做决定用哪套球：原半径会漏真实接触，但在开发场景上没造成碰撞；新半径不漏但保守。
  2. f* 是否改成"这一档和所有更宽的档都过线"。
     - 在开发表上读：固定 1 cm 余量要到 24 cm 才算过；最好的固定余量变成 2 cm / 3 cm，f* 14。
     - "我们的"用原半径是 14，比值 1.0；用新半径是 17。
  3. 右臂以外的部位碰到障碍物算不算失败。
- **证据强度：**一组开发场景（20 个基础场景 × 9 档 × 4 遍），一个种子。四类陷阱留给复核。

### 补充：独立复核（2026-11-23，原文要点）

- 怎么做的：独立 agent，只读、只用 CPU，测试场景没碰。
- 它写的文件：`/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/tmp/review_cover/`，本地副本在 scratchpad 的 `review_cover/`。
- 判定：
  - (a) 复核的接触率得到确认：支持，但那是下限；
  - (b) 新半径包住、无漏判：支持，证据只有 49 次很浅的接触；
  - (c) 新半径更保守、开发集上没换来更安全：支持。
- 我抽查了 `other_body.out`，和它说的一致。

1. **数字核对：**全部复算一致。
   - 接触率：`direct m1: 22/628=0.0350 (p0 22, walked 0)   const m1: 19/628=0.0303   comp m1: 8/628=0.0127, m3: 2/448`；最深 `0.056 / 0.063 / 0.103` cm。
   - 逐位比较：扩到所有逐帧数组（S/W/J/C）、参考姿态（RQ/RP/RT）和 38–46 个逐段字段，没碰的伸手不同的个数仍是 0；碰了的伸手，第一处不同从不在接触之前。
   - m = 1 新球：`direct inside 47, missed 0, false 25 | const 35/0/16 | comp 44/0/36`。
     - 被碰的伸手，新球最小间隙 −0.14 到 −1.2 cm；
     - 新球第一次进入障碍物的时刻，和 MuJoCo 第一次接触相差 ±0.07 s（一例早 0.5 s）。
   - 和前一次复核的 10 Hz 还原逐次对照：
     - m ≤ 3 时，还原判"碰"的 MuJoCo 都记到了；
     - MuJoCo 多出的都极浅（0.003–0.017 cm），是 10 Hz 帧漏掉的：直接多 2 次，补偿多 4 次。
     - 所以 3.2 % / 0.6 % 是下限。补偿的真实值是 1.3 %，是复核的 2 倍，不是"略高"。
2. **更正我写的两处：**
   - **意外 4："伸手已经结束"不对。**
     - 别的部位第一次碰在命令后 3.5–5.6 s；参考动作长 5.8–8.0 s，伸手实际 6.6–9.6 s，接触发生在伸手过程中。
     - 全是**左手**食指、中指碰桌板（`left_hand_index_1/0_link–gap_board`）。
     - 直接发命令 m = 10 那 1 次和补偿 m = 3 那 2 次"右手碰"，都发生在左手先碰之后。
   - **意外 5 的原因：**不是球伸出碰撞体，也不是障碍物放置错位。
     - 那 9 次都是左手先碰到桌板，0.5–1.4 s 后右手的球才进入障碍物。左手一碰，真障碍物那一遍就和空地那一遍分开了。
     - 把空地轨迹截断在左手接触的时刻，m ≥ 6 的误报全部消失。
   - **障碍物放置两边一致：**603 条开发路线 prep 偏航都是 0，`boxes()` 和 `place()` 公式等价，接触时刻差一帧以内。
3. **判据"漏掉深于 0.2 cm 的接触"永远触发不了：**所有手部接触都 ≤ 0.10 cm。好在召回对任何深度都是 100 %。
4. **新球本身：**
   - 分组和碰撞体对得上：右臂 15 个碰撞体，contype/conaffinity 全是 1，没有 exclude。
   - 拟合没见过的姿态也包得住（更细的表面采样）：
     - 真障碍物三组回放：伸出最多 0.026 cm；
     - 第一部分另一套路线：0.026 cm；
     - 关节限位内随机取：0.129 cm，全在手掌那个碰撞体。
   - **反方向问题大：新球比真手胖很多。**球面伸出真实碰撞体：`palm 9–11: 2.24–3.21   wrist_pitch #8: 3.8–4.1   thumb_0: 2.6–3.4   finger bases ~2.0–2.5`（cm）。
     - 三个圆心固定的球包不紧扁平的手掌，误报和 f* 变差都来自这里。
     - 原来的小球也不全在碰撞体里：腕部伸出 0.4–0.6 cm，指尖 0.37 cm。
5. **"计分不依赖球"成立，但要补两点：**
   - **缝的档位是按 cuRobo 球定的：**f = 缝高 − 17 个 cuRobo 球的竖向厚度；规划时障碍物按 m 加厚，也是对 cuRobo 球。所以 "f* = 14 cm" 是比 cuRobo 那只手宽 14 cm，不是比真手。
   - **成功只看手部接触：**右小臂、肘的接触也不算失败。
6. **四类陷阱：**
   - **场景：**dev2 是看过一次试跑后重做的，作开发集没问题；没有回放引用 `gap_routes_test`。
   - **泄漏：**
     - 新球拟合和计分用同一批开发姿态，但拟合外的检查结果一样，影响可以忽略；
     - 标定就在被计分的同一批开发场景上算，开发集上 f* 偏乐观；
     - 没有东西漏到测试。
   - **指标：**
     - f* 卡在边上：固定 1 cm 在 7 cm 那档正好 0.90（72/80），固定 4 cm 在 14 cm 也正好 0.90；
     - "事后最好的固定余量"是在被计分的场景上挑的，偏强。
   - **事后改设定：**在开发集上改 f* 定义、改用哪套球都可以，但要在碰测试之前定死、注明是看过开发结果后改的，并在测试上同时报原登记的定义。
7. **三个待定的决定（`rescore.py`，开发场景）：**

```
cuRobo radii: registered f* ours 14 / best fixed_m1 7 → 2.00;  monotone: best fixed_m2 14 → 1.00 (1.00–2.20)
cover  radii: registered 17/7 → 2.43;                          monotone: 17/14 → 1.21 (1.00–2.80)
other-body counted: only m≥6 / fixed_cal / comp_fixed / radius / const rows change (≤0.09 at f=24), no f* changes, ours unchanged
```

   - **用哪套球：**
     - 还有两个选项：用预测的球心反推连杆位置和朝向，直接拿 MuJoCo 的真实网格判；或者重新布更多、更小的球。
     - 换球也影响 radius 和 const 这两个对手，用新球时它们都到不了线。
   - **f* 改成单调定义：**原则上支持，但要写明比值会从 2.0 变成 1.0，对我们有利。
     - 任何设定下，开发集上"我们的"都没赢过最好的固定余量，最多打平；
     - 可以考虑一个不是事后挑的对手：在开发集上选定的固定余量（比如 2 cm）。
   - **其他部位碰到算不算失败：**开发集上不改变任何 f*；建议按物理含义算失败，右小臂和肘一起算。

### 补充：最终结论（采纳复核）

- 复核的接触率确认了，而且是下限：1 cm 余量下真实接触直接 3.5 %、补偿 1.3 %；2 cm 及以上，右手只在左手先碰之后才碰。
- 原来的球在 1 cm 余量下一次都没抓到。新球一次不漏，但比真手胖 2–4 cm，误报 46–82 %。
- 第二部分的碰撞由 MuJoCo 直接判，球只影响"我们的"和 radius、const、rule_a 怎么做决定。新球让"我们的"在开发集上更保守（f* 14 → 17），没换来更安全。
- 开发集上，"我们的"在任何设定下都没赢过最好的固定余量，最多打平。
- 测试场景之前要用户定：
  - 做决定用哪套球（原来的 / 新的 / 改成按真实网格判，需另写代码）；
  - f* 用哪个定义：原登记的，还是"这一档和所有更宽的档都过线"；两个都报，选一个作主判据；
  - 左手、右小臂、肘碰到障碍物算不算失败。
- 记录：缝宽 f 是相对 cuRobo 那只手量的。

### 这个设计没有覆盖的

- **"cuRobo 关节轨迹直接作参考"这个对手**（38.6 原列表里有）：要改控制器那一侧的参考生成，第一部分也没有它。确认后可以单独加，估计多半天。
- **蹲下：**规划器锁腰，只有站立。
- **SONIC：**等任务 A（38.8.1.5）的结果再定要不要在 SONIC 上也做一遍。
- **碰到之后的后果**（物体被碰倒）：这里只算碰没碰到，留给演示。

## 第四步补充之四：手腕只靠手臂转，放宽修正的上限（2026-11-23，用户："2. 接着做"、"4. 然后开始做第二件"）

### 要解决的两件事

1. **整条手臂**：补偿朝向之后肩离想要的位置 7.5 cm（直接发命令 3.8 cm），24 个球的误差没有改善。
2. **一成补不动的路线**：两批测试路线里约 11 % 的绕行伸手补偿后仍偏 9 cm 以上，修正都顶到了 8 cm 和 40° 的上限。

### 动手之前在开发路线上已经看到的（探索，不是判据）

- 肩偏出去有两个来源。其一，全身逆解拿躯干去实现手腕的转角：只挪位置时规划里的肩跟着手后移 2.1 cm，正好抵消控制器自己前冲的 1.9 cm；转手腕时规划里的肩反而前移 1.1 cm。其二，手一转，控制器自己就多前倾：肩比它自己的规划靠前 3.2 cm（直接发命令时 1.9 cm）。
- 针对其一，给 `Motor.reach` 加了一个选项 `body_rot`：身体按原来的手腕朝向规划，手腕的转角只由手臂的七个关节实现（补偿模块的模式 `arm`）。用第 2 轮模型在 41 条开发路线上跑了一遍（`dev_arm_m2`，模型没见过这种规划）：腕 + 掌 + 指 3.79 → 3.15 cm，24 个球 7.72 → 7.29 cm，肩 7.47 → 7.23 cm。手更准了，肩基本没变。这一遍有一处精度问题（交给动作层的朝向被存成了单精度，个别路线规划前后差了几厘米），已修，数字只作参考。
- 针对其二，试过让身体按另一组点规划、按肩的预测偏差去挪（模式 `body`，12 次回放）：身体的点挪了 6 cm，规划里的肩只动了 0.4–1.2 cm，手还变差了。全身逆解并不让肩跟着这组点走。这条路放弃，代码保留但不用。
- 放宽上限到 12 cm 和 60°（`REACH_VIA_COMP_CAP`、`REACH_VIA_COMP_CAP_TURN`），12 次回放没有出问题。

### 做法

- 补偿器的回放再进两轮训练，用最宽的设置采：模式 `arm`，上限 12 cm 和 60°，在 340 条训练路线上各 4 遍。第 3 轮用第 2 轮模型发命令，第 4 轮用第 3 轮模型。模型的训练数据 = 第 2 轮模型的全部数据 + 这两轮回放。
- 每轮之后在 41 条开发路线上用当轮模型跑 `arm`、上限 12 cm 和 60°。腕 + 掌 + 指的中位数比上一轮改善不到 5 % 就停，最多两轮。
- 最后一轮模型在开发路线上再跑两个对照：原来的做法（模式 `pose`，8 cm 和 40°）、`arm` 配 8 cm 和 40°。三组用同一个模型，差别就只来自做法。

### 在开发路线上怎么取舍（现在写定）

- **用 `arm` 还是 `pose`**：同样 8 cm 和 40° 下，`arm` 的腕 + 掌 + 指中位数比 `pose` 低，且按路线自助抽样的区间不含 0，就用 `arm`；否则用 `pose`。
- **上限用哪一组**：12 cm 和 60° 下"补偿后仍偏 9 cm 以上"的伸手比例比 8 cm 和 40° 低，且腕 + 掌 + 指的 90 % 分位不升高、模型对那一遍的预测误差不超过直接发命令的 2 倍，就用宽的；否则用窄的。
- 肩的问题这一轮预计解决不了（第二个来源是控制器自己的前倾，没有找到能挪它的办法）。24 个球那一组照样报告，不作取舍依据。

### 测试

用第三批新规划的测试路线（种子 20261125），只跑一次。测试的预测和判据等开发结果出来、选定做法之后补在这里，在跑测试之前单独提交。

### 开发路线上的结果和选定的做法（2026-11-23，测试之前写下）

两轮都跑了（各 1360 次回放，0 摔倒）。41 条开发路线 × 4 遍，164 次伸手，直接发命令腕 + 掌 + 指 7.91 cm。三项一致性检查都是 0.00 mm。原始输出在各运行目录的 `comp_eval.out`，结果一节等用户说提交时一起补。

| 运行 | 模型 | 做法和上限 | 腕 + 掌 + 指中位数 | 90 % 分位 | 仍偏 9 cm 以上 | 24 个球 | 模型对这一遍的预测误差（直接发命令） |
|---|---|---|---|---|---|---|---|
| `onpol_dev_r2_pose` | 第 2 轮 | `pose`，8 cm 和 40° | 3.79 cm | 8.74 cm | 15 / 164 | 7.72 cm | 2.23 cm（1.24） |
| `onpol_dev_r3_arm` | 第 3 轮 | `arm`，12 cm 和 60° | 3.27 cm | 10.41 cm | 22 / 164 | 7.36 cm | 2.09 cm（1.26） |
| `onpol_dev_r4_arm` | 第 4 轮 | `arm`，12 cm 和 60° | 3.41 cm | 11.35 cm | 20 / 164 | 7.44 cm | 2.06 cm（1.23） |
| `onpol_dev_r4_arm_narrow` | 第 4 轮 | `arm`，8 cm 和 40° | 3.33 cm | 7.49 cm | 10 / 164 | 7.41 cm | 2.04 cm（1.23） |
| `onpol_dev_r4_pose` | 第 4 轮 | `pose`，8 cm 和 40° | 3.70 cm | 8.94 cm | 17 / 164 | 7.72 cm | 1.87 cm（1.23） |

按事先写的规则：

- **轮数**：第 4 轮比第 3 轮没有改善（3.27 → 3.41 cm，成对比较的区间含 0），到此停。按写定的做法，对照和测试都用最后一轮的模型 `via_point_onpol4_20261123`。
- **用 `arm`**：同样 8 cm 和 40° 下，`arm` 3.33 cm，`pose` 3.70 cm，中位数低 10 %，按路线自助抽样的区间 1 %–22 %，不含 0。
- **上限用窄的，8 cm 和 40°**：宽上限下仍偏 9 cm 以上的伸手是 20 次，窄上限 10 次，没有更低；90 % 分位从 7.49 升到 11.35 cm。两个条件都不满足。

宽上限为什么更差（探索）：两种上限只在 20 次伸手上发出了不同的命令，都是窄上限下修正已经顶格的那些。放宽之后这 20 次的修正走到 10.4 cm 和 60°，实际误差从 7.4 cm 变成 16.5 cm，模型对它们的预测误差从 4.7 cm 变成 10.0 cm。其余 144 次两种上限的命令和结果完全一样。也就是说，修正走得越远，模型越不准，两轮回放也没有把这一块学会。"补不动"的原因不在上限太小。

窄上限下仍偏 9 cm 以上的 10 次伸手集中在 41 条路线里的 5 条，直接发命令时它们就偏 11–16 cm，模型对这 10 次的预测误差是 6–14 cm。模型自己预测的剩余误差也偏大（4–9 cm，其余伸手中位数 2 cm），所以"先预测再决定出不出发"能把它们挑出来。

肩没有变：`arm` 7.33 cm，`pose` 7.52 cm，直接发命令 3.83 cm。和预计的一样。

### 测试的设计（2026-11-23，测试之前写下）

- **路线**：种子 20261125，450 个场景，规划出 172 条（越过桌沿 21、越过箱子 28、伸到板下 123），在 `outputs/reachwm_20261030/test3_routes`。和训练、开发、前两批测试路线都不共用场景。规划只用到 cuRobo，没有跑过控制器。全部路线各回放 4 遍。
- **四组回放，同样的路线同样的起点**（种子 57000000）：
  1. 直接发命令；
  2. 常数修正（对手，模型 `via_point_20261121`）；
  3. `arm`，8 cm 和 40°，第 4 轮模型：我们选定的做法；
  4. `pose`，8 cm 和 40°，第 4 轮模型：用来在测试路线上核对"`arm` 比 `pose` 好"这个选择。
- **窄缝计分**：和第二批测试完全相同的规则（`via_gap.py --rule margin`）。我们这一方的余量由脚本从 `onpol_dev_r4_arm_narrow` 算出，对腕 + 掌 + 指是 3.81 cm；对手的余量仍取自 `via_point_20261121`；在开发路线上标定的固定余量取自 `dev_via_pose2_20261122/direct`。
- 模型、做法、上限、余量都已定死，测试路线不用来定任何数。只跑一次。主要看 49 条绕行路线。

### 写预测之前已经知道的

- 上面开发路线的全部数字。
- 把同一套窄缝计分用在开发路线上（余量也取自同一遍，所以偏乐观）：我们 11.50 cm（区间 9.25–12.50），小固定余量 12.75 cm，开发路线上标定的固定余量 12.75 cm，比值 0.90（0.75–1.00）。8 cm 的缝：我们成功 70 %、碰撞 0 %；小固定余量成功 52 %、碰撞 48 %。
- 前两批测试（都是 `pose`、第 2 轮模型）：腕 + 掌 + 指 8.06 → 3.40 cm 和 8.39 → 3.88 cm；第二批里 11 % 的伸手仍偏 9 cm 以上；比值 1.02 和 0.96，判据两次都没过。

### 预测（49 条绕行路线，196 次伸手）

- 直接发命令腕 + 掌 + 指中位数 7.5–8.7 cm；`arm` 3.0–3.9 cm，降幅 0.52–0.62。
- `arm` 比 `pose` 的中位数低 3–15 %。区间含不含 0 我估计一半一半（开发路线上区间的下端只有 1 %）。
- 仍偏 9 cm 以上的伸手：`arm` 5–11 %，`pose` 8–14 %。
- 终点朝向误差 7–10°（直接发命令 19–22°）。
- 24 个球 7.0–7.8 cm（直接发命令 8.0–8.8 cm）；肩 7 cm 左右，没有改善。
- 模型对 `arm` 那一遍的预测误差 1.8–2.6 cm，不超过直接发命令那一遍的 2 倍。
- 窄缝：我们 11–13.5 cm，小固定余量 12–14 cm，开发路线上标定的固定余量 12.75–14 cm，比值 0.85–1.0。**预计判据仍然不成立，或者只是"有优势但不大"。** 8 cm 的缝上我们成功 60–75 %、碰撞不超过 5 %；小固定余量成功 40–55 %、碰撞 45–60 %。

### 判据

- **这一轮的主要问题，`arm` 是否比 `pose` 好**：测试路线上 `arm` 的腕 + 掌 + 指中位数低于 `pose`，且按路线自助抽样的区间不含 0，这个选择成立。区间含 0，报告"测试路线上分不出来"。`arm` 反而高且区间不含 0，开发路线上的选择没有复现，照实写。
- **手的精度**：`arm` 相对直接发命令的降幅不低于 0.5，区间下端不低于 0.4。不满足就说明前两批的结果在这批路线上没有保持。
- **补不动的那部分**：`arm` 仍偏 9 cm 以上的比例低于 `pose`，且两者之差按路线自助抽样的区间不含 0，才说"减少了"。否则只报告两个比例。
- **窄缝**，和以前相同：比值 ≤ 0.8 且区间上端 < 1 成立；比值 > 0.95，或区间含 1 且比值 > 0.8，不成立；中间且区间上端 < 1 是"有优势但不大"。
- **有效性**，任何一条不满足结果不算：三项一致性检查 0.00 mm；摔倒不比直接发命令多 5 次以上；模型对 `arm` 那一遍的预测误差不超过直接发命令那一遍的 2 倍。
- 24 个球和肩照样报告，不作判据。
- 跑完交独立复核，只给数据、脚本和结果。

### 测试的补充分析：把真障碍物放回去（2026-11-23，测试回放还在跑、没有看过任何测试结果时写下）

**为什么加这一项。** 上面的度量都是"离想要的路线多远"，各个方向一样算。真正要紧的是会不会碰到障碍物。每条路线规划时的箱子都存着（`prep_<名字>.npz` 的 `boxes`，和记录的手臂小球在同一个坐标系），回放是在空地上跑的，可以离线把箱子放回去，数一数记录下来的手臂会不会碰到。不用再跑仿真。脚本 `via_scene.py`（还没提交，md5 `509ef2effacbdfe757c96d5418dc8a41`）。一次伸手算"碰到"：24 个球里任何一个在任何一拍进了箱子（到箱子的距离减去球的半径小于 0）。箱子不在仿真里，所以这是几何上的重叠，没有接触力。

**写预测之前已经知道的**（41 条开发路线，164 次伸手，输出 `outputs/reachwm_20261030/dev_scene_check_20261123`；坐标系核对过：我们参考规划的间隙和规划器自己报的间隙逐条路线相关 0.61，都在 0–2.5 cm）：

| | 碰到的伸手 | 和直接发命令之差（区间） | 肩 / 肘碰到 |
|---|---|---|---|
| 想要的路线本身（我们的全身参考规划） | 9.1 % | | 0 |
| 直接发命令 | 34.8 % | | 0 |
| 只挪位置（`via_point_20261121`，只见过随机路线） | 15.2 % | −0.195（−0.299 到 −0.110） | 0 |
| `arm`，第 4 轮模型 | 20.1 % | −0.146（−0.268 到 −0.018） | 4 次 / 3 次 |
| `pose`，第 4 轮模型 | 26.2 % | −0.085（−0.220 到 +0.049） | 5 次 / 5 次 |

- 肩离箱子的间隙中位数 11–13 cm，肘 9 cm。肩被带偏的那 7 cm 在这类场景里碰不到东西。碰到的几乎都是手指（间隙中位数只有 1.4 cm，规划器留的余量本来就薄）。
- 按"离想要的路线多远"排，`arm` 最好（3.33 cm），只挪位置差得多（5.56 cm）。按"碰不碰到箱子"排，只挪位置最好。`arm` 对只挪位置 +0.049（−0.018 到 +0.122），分不出来；`pose` 对只挪位置 +0.110（+0.024 到 +0.207），更差。
- 只挪位置时手掌的误差和 `arm` 差不多（2.78 对 2.55 cm），终点还更准（0.78 对 1.29 cm）。加朝向多修的主要是指尖，代价是肩和肘开始碰到、修坏的时候错得更深（最深 4.7 cm，只挪位置 2.8 cm）。

**多跑一组。** 测试路线上加一组只挪位置的回放（`test3_point`，模型 `via_point_20261121`，和开发路线上那一遍相同的设置），在前四组跑完之后跑，同样的路线和起点。

**预测（49 条绕行路线，196 次伸手）。** 碰到的比例：想要的路线本身 6–13 %；直接发命令 28–42 %；只挪位置 11–22 %；`arm` 15–27 %；`pose` 19–32 %；常数修正 20–35 %（开发路线上没跑过）。肩和肘：直接发命令和只挪位置 0–1 %，`arm` 和 `pose` 1–5 %。碰到的伸手九成以上是手指碰的。

**判据。**

- 某种做法"在真障碍物上减少了碰撞"：它碰到的比例低于直接发命令，且两者之差按路线自助抽样的区间不含 0。只挪位置、`arm`、`pose`、常数修正各判一次。
- 只挪位置对 `arm`：两者之差的区间不含 0，就说比例低的那个更好；含 0 就是分不出来，这时下一步（有真障碍物的场景）用只挪位置，因为它不需要规划器路线进训练。
- 这一项是补充，不替换上面登记的判据，两边的结果都报告。

## 第四步补充之五：只挪位置，对着真障碍物的单侧度量（2026-11-23，用户："1. 改 2. 停 3. 做"；本节在规划第四批路线之前写下）

### 为什么

第三批测试之后有两件事要落到事先登记的测试上。

1. 把箱子放回去之后，只挪位置碰到的最少（10 %，直接发命令 46 %），而它在"离想要的路线多远"上并不是最好的。用户定：对着真障碍物的主方法改成只挪位置，模型只见过随机路线（`via_point_20261121`）。
2. "碰到的比例"这个数本身不稳。独立复核指出：多数重叠不到 1 cm；把箱子整体往前挪 3 cm，直接发命令的比例从 46 % 掉到 13 %。需要一个不靠阈值的量。事后算出来的"规划的间隙被执行吃掉多少"在三批路线上一致（95 % 分位直接发命令约 4 cm，只挪位置约 2 cm），但它是看到结果之后想到的，这里登记后在新路线上测。

### 度量（脚本 `via_scene.py`，还没提交，md5 `a26a78f82d1b65df6ba3f0118d728fed`）

- **被吃掉的间隙**：每次伸手，"想要的路线到箱子的最小间隙"减去"实际记录的手臂到箱子的最小间隙"，24 个球一起取最小。正数表示执行比规划更靠近箱子。它的 95 % 分位就是规划器要让 95 % 的伸手不比规划更靠近箱子，得多留的余量。
- **比值** = 只挪位置的 95 % 分位 ÷ 直接发命令的 95 % 分位，区间按路线自助抽样。
- **对坐标系的敏感性**：把所有箱子沿 x、y、z 各正负挪 1 cm，共六种，各算一次比值。
- **到位且不碰**：手掌停稳后离想要的终点不超过 2 cm（`Motor.reach` 自己的容差），并且全程没有任何球进箱子。
- **碰到的比例**照旧算，作为第三批结果的复现。

### 做法

- **第四批测试路线**：种子 20261126，900 个场景（前三批是 450 或 150 个，这次加倍是为了让绕行路线多一倍，区间窄一些），场景生成器不变。只回放绕行路线（越过桌沿、越过箱子），预计约 95 条，各 4 遍。伸到板下的路线不进这一步的任何度量，不回放。
- **四组回放，同样的路线同样的起点**（种子 59000000）：直接发命令；常数修正（`via_point_20261121`，对手）；只挪位置（`via_point_20261121`，主方法）；位置加朝向 `arm`（第 4 轮模型，8 cm 和 40°，对照）。
- 模型、设置、度量都已定死，只跑一次。跑完交独立复核。

### 写预测之前已经知道的

同一个脚本在已有路线上算出来的（开发路线和前三批测试路线，都是事后算的）：

| | 被吃掉的间隙 95 % 分位：直接 / 只挪位置 | 比值（区间） | 箱子挪 ±1 cm 时比值的范围 | 到位且不碰：直接 / 只挪位置 | 碰到：直接 / 只挪位置 |
|---|---|---|---|---|---|
| 开发路线（164 次） | 4.13 / 1.85 cm | 0.45（0.37–0.77） | 0.43–0.63 | 8.5 % / 78.7 % | 34.8 % / 15.2 % |
| 第一批测试（84 次） | 3.35 / 2.01 cm | 0.60（0.39–1.18） | 0.50–0.60 | 没算 | 41.7 % / 15.5 % |
| 第二批测试（180 次） | 3.93 / 2.14 cm | 0.54（0.45–0.72） | 0.55–0.63 | 6.7 % / 75.0 % | 45.6 % / 17.2 % |
| 第三批测试（196 次） | 3.78 / 2.01 cm | 0.53（0.41–0.78） | 0.52–0.63 | 6.6 % / 77.0 % | 45.9 % / 10.2 % |

- 第三批上常数修正：95 % 分位 3.94 cm，比值 1.04（0.81–1.26），到位且不碰 1.5 %。`arm`：3.25 cm，比值 0.86（0.73–1.39），箱子挪 ±1 cm 时 0.86–1.24，到位且不碰 60.2 %。
- 开发路线上只挪位置换成第 4 轮模型（`onpol_dev_r4_point`）：碰到的比例一样（15.2 %），95 % 分位 2.17 cm，不比只见过随机路线的模型好。所以主方法用后者。
- 直接发命令"到位"的本来就少（手掌终点在 2 cm 内的只有 11–13 %），所以"到位且不碰"的差距主要来自到位，其次才是不碰。

### 预测（约 95 条绕行路线，约 380 次伸手）

- 被吃掉的间隙 95 % 分位：直接发命令 3.3–4.2 cm；只挪位置 1.7–2.4 cm；比值 0.45–0.65。
- 箱子挪 ±1 cm 的六种情况，比值都在 0.40–0.70。
- 常数修正的比值 0.95–1.15；`arm` 0.75–0.95。
- 到位且不碰：直接发命令 5–10 %；只挪位置 72–82 %；`arm` 55–68 %；常数修正 0–4 %。
- 碰到的比例：想要的路线本身 6–12 %；直接发命令 38–48 %；只挪位置 10–18 %；`arm` 19–28 %；常数修正 55–63 %。
- 腕 + 掌 + 指离想要的路线（不作判据）：直接发命令 8.2–8.8 cm，只挪位置 5.8–6.4 cm。

### 判据

- **主判据**：比值 ≤ 0.7 且区间上端 < 1，成立，可以说"只挪位置让规划器需要多留的余量减少三成以上"。比值 > 0.9，或区间含 1，不成立。中间且区间上端 < 1，是"有改善但不到三成"，照实报数。
- **对坐标系不敏感**：六种挪法下比值都 ≤ 0.8。有一种超过，就要写明"这个结论依赖箱子位置准到 1 cm 以内"。
- **到位且不碰**：只挪位置高于直接发命令，两者之差的区间不含 0。
- **碰到的比例**：只挪位置低于直接发命令，区间不含 0（复现第三批）。
- **有效性**，任何一条不满足结果不算：一致性检查 0.00 mm；摔倒不比直接发命令多 5 次以上；想要的路线本身碰到的比例不超过 15 %（超过说明箱子和手臂不在同一个坐标系里）。
- `arm` 和常数修正只作对照，照样报告。
- 仍然要写明的限制：箱子不在仿真里，这是记录下来的手臂和箱子的几何重叠；真障碍物进仿真是下一步。

## 预测修订

- 2026-11-23，第四步第二部分，可行性探查之后、任何回放之前：**用户选定场景的改法（"按场景量手厚+筛目标"）。**
  - 缝宽 g = 这个底场景的手在它自己目标朝向下的竖直厚度 + f，不再用手平放时的 8.7 cm。厚度是 17 个球的竖直跨度，目标在桌面上 10 cm、无障碍时解位姿得到。
  - 手在缝里居中：手的最低点在桌面以上 f/2 + dz，dz 限在 ±f/4。
  - 桌沿 0.28–0.34 m（原 0.20–0.26），目标过桌沿 4–10 cm（原 8–14）。这两项在事先允许调的旋钮之内。
  - 只保留够得着的底场景：9 档的目标位置上，没有障碍物时都解得出位姿。从同一个随机数发生器依次抽，够数为止，所有抽到的都记下。保留下来的场景规划失败照实记，不重抽。
  - 原因：探查里 f ≤ 11 cm 全部失败。目标手形朝下 0–30°，按 8.7 cm 定的缝宽下指尖已经插进桌面；另有一半目标锁腰时根本够不着，和缝无关。
  - 预测和判据都不改。可行性探查的两条标准也不改，在开发种子上重跑一次：最窄一档 m = 1 cm 仍不到 30 % 就再问用户，不自己改档。
  - 预测原来是按"比平放的手厚多出 f"写的，现在 f 是"比手在目标朝向下的实际厚度多出多少"。含义更接近原意，也就是"缝比手厚多出多少"，数值没有重写。

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
- 2026-11-21，用户定了覆盖检查的规则（原话："A 不加那一点"）：改成"不比 95 % 的随机路线更孤立"，生成器不再动，偏离量的下限仍是 3 cm。
  - 按新规则重算已有的两份 300 条数据：试跑那份 35 / 35，补生成器之后的探索那份 35 / 35。中位数规则分别是 14 和 9。
  - 正式采集的 12000 条出来后再算一遍，以那次为准。离线估计那时是 32 / 35。

（第一次启动之后的修订写在这里。下面这条是启动前的说明。）

- 2026-11-20，启动前：写方向稳定度和余量之比的预测与判据时，已经看过一次冒烟测试的输出（8 条随机路线、3 条用随机路线伪造的回放各 2 遍），目的是确认脚本能跑。冒烟数据没有真实路线，不用来下任何结论，也不计入试跑。
- 2026-11-22，手腕朝向补偿，正式跑之前：**事先写的做法在 8 条试跑路线上把结果弄坏了，改了做法；之后的设置只在随机路线上挑。**
  - 事先写的做法：每轮把预测的朝向偏差整个转回去，4 轮，手腕命令最多转 60°。正式跑之前先在 63 条测试路线里的 8 条上试了一遍（第 0–3、12–15 条，都是长绕行，各 1 遍）。手掌误差 7.01 → 6.18 cm，终点 5.15 → 4.22 cm；手在终点的朝向误差 23.0° → 22.7°，没有变；模型对补偿后那一遍的预测误差从 2.49 cm 升到 5.51 cm。
  - 原因一：命令被挪到了模型没怎么见过的地方。集成里各个成员对手的分歧，普通命令是 0.4 cm，这些命令是 1.0 cm。
  - 改成：每轮只走剩余修正的一半，6 轮，手腕命令最多转 40°；每个路点新的修正只有在"成员分歧不超过原命令的 1.5 倍，也不超过 1.2 cm"时才收下，否则保留上一次的修正并把这个路点的步子减半。同样 8 条：手掌 7.01 → 3.07 cm，终点 5.15 → 2.03 cm，路上的朝向误差 12.0° → 3.8°；终点的朝向误差 23.0° → 24.3°，还是没有变。
  - 原因二，终点朝向为什么不动：路上，手实际转过的角度是命令转角的 0.57，方向一致（余弦 0.97），模型预测的是 0.49，预测对了。终点，手实际转过 0.88，但方向偏了（余弦中位数 0.67），模型预测的是 0.53、方向一致，预测错了。关节都离限位很远（1 rad 以上），不是关节顶住了。训练数据里路点的朝向是随机转过的，目标的朝向只有抓取姿态那一族；补偿把目标朝向转离了这一族，模型没见过。路上是跟随滞后，终点是稳态，两者的规律不同，模型拿路上的规律去套终点。
  - **这 8 条属于测试集**，看了两个版本。为了不再泄漏，后面的设置都在 200 条新的随机路线上挑（种子 43000000，不在任何训练集里，输出 `outputs/reachwm_20261030/dev_via_pose_20261122`），定下来之后在 63 条 × 4 遍上只跑一次。报告时把这 8 条路线单独拿出来再算一遍（剩下 13 条绕行路线）。
  - 200 条随机路线上的结果（站立、手掌路线长于 0.7 m 的 53 对；成对比较里有 19 对两遍走路后的起点不一样，已剔除）：
    | | 腕 + 掌 + 指 | 手掌 | 终点朝向 | 路上朝向 |
    |---|---|---|---|---|
    | 直接发命令 | 8.51 cm | 6.95 cm | 11.6° | 11.6° |
    | 只挪位置 | 6.75 cm | 4.62 cm | 13.5° | 12.5° |
    | 位置加朝向（带上面的信任规则） | 7.56 cm | 6.21 cm | 4.9° | 6.3° |
  - 随机路线上朝向补得动，终点也补得动（手实际转过的角度和模型预测的一致，方向余弦 0.94）。**和 8 条绕行路线上看到的不一样**，所以随机路线当不了这一步的开发集。
  - 随机路线上手掌反而变差，原因是信任规则太严：91 % 的路线有修正被拒，位置修正跟着一起被拒。
  - 另一个发现：补偿只在路点处修正。只挪位置时，模型自己预测的手掌误差从 5.55 cm 只降到 3.16 cm，路点之间修不到。
  - **改用规划器路线做开发集**：换种子（20261122）重新让 cuRobo 规划，300 个场景规划出 132 条，其中长绕行 41 条（越过桌沿 20、越过箱子 21），和测试集没有一个共同的场景。在这 41 条 × 4 遍上挑设置，输出 `outputs/reachwm_20261030/dev_via_pose2_20261122`。原来两个带信任规则的设置（不转目标、上限 20°）中途停掉了，没有用。
- 2026-11-22，开发用的 41 条绕行路线 × 4 遍（164 对，输出 `outputs/reachwm_20261030/dev_via_pose2_20261122`）。直接发命令和只挪位置这两行和测试集上的数字很接近（测试集 8.06 → 5.73 cm，4.83 → 2.99 cm），说明这批路线能代替测试集做开发：
    | | 腕 + 掌 + 指 | 手掌 | 整条手臂 | 手掌终点 | 终点朝向 | 路上朝向 | 模型对这一遍的预测误差 |
    |---|---|---|---|---|---|---|---|
    | 直接发命令 | 7.91 cm | 4.79 cm | 8.36 cm | 3.39 cm | 20.0° | 12.3° | 2.80 cm |
    | 只挪位置 | 5.56 cm | 2.78 cm | 6.12 cm | 0.78 cm | 18.9° | 11.9° | 2.99 cm |
    | 位置加朝向，不带信任规则 | 4.97 cm | 3.73 cm | 8.31 cm | 2.08 cm | 16.5° | 4.5° | 6.87 cm |
    | 同上，但不转目标的朝向 | 6.55 cm | 3.97 cm | 7.81 cm | 1.62 cm | 19.3° | 4.7° | 6.28 cm |
  - **转过朝向的命令，模型预测不了。** 模型对补偿后那一遍的预测误差是 6.87 cm，是直接发命令时的 2.45 倍，事先写的"不超过 2 倍"这条检查在开发集上就不成立。按部位看，错得最多的是肩（2.31 → 6.32 cm），肘、腕、掌、指都跟着变差：手腕命令转过 20° 以上之后，躯干的动作变了（肩离想要的位置从 3.8 cm 变成 7.5 cm），模型没见过。
  - 路上的朝向确实补回来了（12.3° → 4.5°），腕 + 掌 + 指的中位数也比只挪位置略好（5.56 → 4.97 cm），但手掌、终点、整条手臂都变差，90 % 分位没有变（9.87 和 9.91 cm）。目标朝向被转到上限 40° 的 44 对里，加朝向的结果比只挪位置差（7.07 → 9.02 cm）。
  - 终点：命令转 25° 以上的 118 对，手实际转的方向和命令的方向余弦只有 0.61，模型预测的是 0.94。命令转 15–25° 的 46 对是 0.88，朝向误差 16.9° → 8.2°。
  - 随机路线上没有这个问题（模型预测误差 4.67 → 5.79 cm），因为随机路线需要转的角度小（中位数 12°），绕行路线要转 20–40°。
  - **下一步：第三次通用补采。** 假设是训练数据里手的朝向只有抓取姿态那一族，转开 20° 以上的近身命令没见过。补 8000 次开场的随机路线，其中 70 % 把目标朝向绕随机轴转 0–45°，路上的朝向跟着一起变（开关 `REACH_VIA_GOAL_TURN=0.7:45`，其余和第二次补采相同，种子 45000000，输出 `outputs/reachwm_20261030/data_via3`）。生成器仍然不看规划器路线。
    - 检验不用再跑仿真：用三份数据重训逐点模型，拿它去预测上表第三行那 164 遍已经跑完的回放。
    - 预测：重训后模型对那一遍的预测误差从 6.87 cm 降到 4 cm 以下；对直接发命令那一遍保持在 3 cm 左右。
    - 判据：降到直接发命令那一遍的 2 倍以内，就用新模型在开发路线上重跑补偿、定设置，再在 63 条测试路线上只跑一次。降不到，说明通用的随机路线盖不住规划器路线这一块，向用户报告，由用户在"训练里加规划器路线""直接改参考轨迹的手臂关节""只做位置补偿"之间选。
- 2026-11-22，第三次通用补采的结果：**判据没过，通用的随机路线盖不住这一块。**
  - 补采 8000 次，0 摔倒，0 跳过（`data_via3`）。三份数据重训的模型在 `outputs/reachwm_20261030/via_point_20261122`，验证集上和原来差不多（站立不走路 206 条降 78 %，蹲低 1656 条降 60 %）。
  - 离线预测开发路线上已经跑完的回放（`parts_dev_routes.txt`、`turn_dev_routes.txt`）：直接发命令那一遍 2.89 cm（原模型 2.80），位置加朝向那一遍 6.56 cm（原模型 6.87）。6.56 > 2 × 2.89，判据没过。终点处手实际转的方向和模型预测的方向还是对不上（余弦 0.70 对 0.96）。
  - 身体没有迈步：补偿后的 164 遍里骨盆最多移动 6.2 cm（直接发命令 4.7 cm），只是倾斜更大（骨盆最大倾角的 90 % 分位 9.1° → 13.2°），肩跟着多偏 3–4 cm。
- 2026-11-22，两个不跑仿真的对照，回答"缺的到底是哪一类数据"（脚本和输出在 `outputs/reachwm_20261030/dev_via_pose2_20261122/onpolicy_check`，网络和设置与正式训练相同，新加的路线重复 5 次）：
    | 训练里另外加了什么 | 测的是什么 | 模型的预测误差 |
    |---|---|---|
    | 什么都不加（三份随机路线数据） | 开发路线上位置加朝向的 164 遍 | 6.56 cm |
    | 随机路线上的补偿回放 400 条 | 同上 164 遍 | 6.57 cm |
    | 另外 30 条规划器路线上的补偿回放 360 条 | 没进训练的 11 条规划器路线的 44 遍 | 7.24 → 2.52 cm，98 % 的路线变好 |
  - 第三行里训练和测试按路线分开，同一条路线的 4 遍都在同一边。只挪位置的回放在没见过的路线上也从 1.90 降到 1.16 cm，普通验证集不变（3.17 和 3.19 cm）。
  - **结论：模型缺的是"补偿器自己在规划器路线上发出的命令"这一类数据。** 补偿是在模型上做优化，它会把命令推到模型没见过的地方；随机路线上补偿器只需要转 12° 左右，推不到那里，所以不管是通用补采还是随机路线上的补偿回放都不管用。几百条规划器路线上的补偿回放就够了。
  - 这一步要不要做由用户定：此前用户定过"生成器不针对规划器路线"（2026-11-20、11-21），把规划器路线上的回放放进训练改变了"模型只见过随机路线"这句话。朝向补偿在这里停下，等用户决定。
- 2026-11-23，测试集还没跑，先改窄缝计分里两处余量的来源（`via_gap.py` 的 `--base` 和 `--cal`）：
  - 原来所有余量都由被测模型自己的训练侧数据定。现在被测模型的训练数据里有规划器路线上的回放，照原样做会有两个问题。
  - 问题一：四个对手（小固定余量、标定过的固定余量、标量半径、常数修正）的余量会跟着变，和第一部分的数字不再可比。改成：对手的余量仍然用只见过随机路线的模型 `via_point_20261121` 的训练侧数据定，和第一部分完全相同。
  - 问题二：我们这一方的余量如果用训练侧的标定集定，标定集里有训练路线的其他几遍，会偏小，在新路线上碰撞会超出 5 %。改成：我们这一方残差半径的缩放系数，在 41 条开发路线的补偿回放上定（最后一轮模型跑的那一遍，164 次伸手，取 95 % 分位）。开发路线不在训练里，也不参与计分。
  - 不带这两个选项时，脚本的输出和已提交的第一部分逐项相同（已核对）。
  - 三组路线的分工因此是：训练路线（种子 20261123）训练模型；开发路线（种子 20261122）定停在第几轮、定余量；测试路线（种子 20261121）只用来计分，只跑一次。

- 2026-11-23，第四步第二部分（真障碍物窄缝）上测试场景之前，按用户的决定定死计分设置。本条在测试场景任何回放之前写下。用户的原话是对四个问题的选择："New spheres as main"、"New as main, report both"、"Count as failure"、"Run it once"。
  - **为什么改：**开发场景演练和独立复核发现三件事（见"第四步第二部分"·补充的结果和复核）：
    - 原来的球在 1 cm 余量下漏掉全部真实接触；
    - 原登记的 f* 定义让一个在更宽档碰撞 6 % 的固定 1 cm 余量当了最好的对手；
    - 左手手指在伸手过程中碰桌板，原定义不算失败。
  - **这些改动都是看过开发场景结果后定的。**测试上同时报原登记的设置。
  - **主分析：**
    - "我们的"和 radius、const、rule_a 做决定时用新球（`gap_cover/cover.json`）；
    - 碰撞 = 机器人任何部位碰到障碍物（`--fail-any`）；
    - f* 用单调定义：这一档和所有更宽的档都满足成功 ≥ 90 %、碰撞 ≤ 5 %（`--monotone`）。
  - **同时报：**
    - 原登记的设置：原来的球、只算右手腕 + 掌 + 指、原 f* 定义；
    - 两个中间组合：新球 + 只算手；原来的球 + 任何部位。
    - 每份输出里两种 f* 定义都打印。
  - **主判据不变，用在主分析上：**
    - 我们的 f* ≤ 最好对手的 0.8 倍：成立；
    - > 0.95 倍：没有优势；
    - 两者之间：报告。
    - 计分之前的检查也不变。
  - **测试回放：**`gap_routes_test`（种子 20261125），ScaleBFM，回放种子 59000000，三种发法各放真障碍物一遍、空地一遍。
    - ID0 改为 10700000–11200000，原草稿的 9500000–10000000 和开发场景、任务 A 撞了；
    - 进程数、卡都和开发场景相同：每组 18 个进程，1–3 号卡；
    - 模型 `via_point_onpol2_20261123`，标定用开发场景的 `gap_dev/calib`。
  - **新的预测（主分析，根据开发场景；原登记的预测不改，照样对照）：**
    - **MuJoCo 手部接触：**
      - 直接发命令 m = 1 cm：2–6 %；补偿 0.5–3 %；
      - m ≥ 2 的手部接触只在别的部位先碰之后出现；
      - 别的部位碰到的占 0.3–1.5 %。
    - **f*：**
      - 最好的对手是固定 2–3 cm 余量，f* 11–17 cm；
      - "我们的" f* 14–20 cm；
      - 比值 1.0–1.5，落在"没有优势"一侧的把握约八成。
    - **原登记的设置：**最好的对手是固定 1 cm，f* 5–11 cm，比值 1.5–3。
    - **空地球判定和 MuJoCo 接触的一致率：**新球 ≥ 0.95，原来的球 ≥ 0.98。
    - **原登记的预测：**"m = 1 cm 碰撞率 60–95 %""比值 0.6–0.85"这几条，按开发场景看大概率落空。
  - **计分脚本核对（开发场景，`gap_dev/verify/*.out`，四种组合）：**和复核的 `rescore.py` 逐项一致。
    - 新球 + 单调定义：最好的对手是固定 2 cm，f* 14；"我们的" 17，比值 1.21（1.00 到 2.80）。
    - 原来的球 + 单调定义：比值 1.00（1.00 到 2.20）。
    - 原登记设置：比值 2.00（1.00 到 3.40），和原来的 `score_curobo.out` 相同。
    - 任何部位碰到都算失败，不改变任何做法的 f*。
  - **运行记录：**
    - 11:37 UTC 起跑。
    - 11:55 UTC 用户说"跑完 Real-obstacle replays 就停下来，把剩下的 3 个任务写到 todo 中"。真障碍物三组照常跑；空地对照、计分、复核和报告留到 TODO 38.6.3.8.2–.4。
    - 12:46:42 UTC 三组跑完，退出码全 0，各 3512 段；启动脚本按设计在空地对照处停下，没有计分（2026-11-24 4090 恢复后核对）。
    - 测试的接触结果还没看。上面的设置和预测不变，接着跑时照用。

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

### 正式采集和第一次评测（2026-11-21）

正式采集：2026-11-21 02:31–05:19 UTC，vastgpu，提交 `554a28f8`，4 个进程，12000 次开场（种子 37000000，ID0 3000000），数据在 `outputs/reachwm_20261030/data_via`。四个进程都正常结束，没有跳过的开场。采集后的分析第一次启动时卡了 9 小时：`arm_spheres.py` 每处理一帧都重新解压整个数据文件，试跑的数据小看不出来。改成每个数组只读一次后重跑，在试跑数据上核对过输出逐位相同。

偏差统计（`arm_spheres.py`、`via_dev.py`）：

```text
24000 segments, 1462206 ticks, 374256 reference poses, 24 spheres -> /workspace/code/locomanipulation-selfwm/outputs/reachwm_20261030/data_via/arm.npz
rebuilt jaw centre against the recorded one: median 0.024 cm, 95 % 0.135, max 0.970
last reference pose's jaw against the command: median 0.631 cm, 95 % 2.883 (the plan's own IK miss: median 0.010, 95 % 2.950)
paths 12000, fell 0, analysed 12000; plan's IK miss > 2 cm: 1405; skipped resets 0
largest offset from the reference path (hand: jaw centre; arm: any of the 24 spheres):
  all                            n 12000   hand median  6.64 p90 11.92   arm median 10.74 p90 17.05 cm
  standing                       n  5399   hand median  3.72 p90  7.59   arm median  6.66 p90 12.77 cm
  crouching (pelvis asked)       n  6601   hand median  8.79 p90 13.32   arm median 13.01 p90 18.77 cm
  0 bends                        n  2809   hand median  5.79 p90 10.93   arm median  9.86 p90 15.32 cm
  1 bends                        n  4562   hand median  6.44 p90 11.76   arm median 10.55 p90 17.10 cm
  2 bends                        n  2988   hand median  7.06 p90 12.18   arm median 11.17 p90 17.49 cm
  3 bends                        n  1641   hand median  7.97 p90 13.34   arm median 12.02 p90 18.99 cm
  speed 0.15 m/s                 n  5982   hand median  6.92 p90 12.42   arm median 11.08 p90 17.31 cm
  speed below 0.15               n  2505   hand median  6.43 p90 11.53   arm median 10.63 p90 16.64 cm
  speed above 0.15               n  3513   hand median  6.39 p90 11.50   arm median 10.40 p90 16.87 cm
the arm's worst sphere is on: {'shoulder_pitch_link': 851, 'shoulder_roll_link': 216, 'shoulder_yaw_link': 3068, 'elbow_link': 79, 'wrist_pitch_link': 718, 'hand_palm_link': 43, 'hand_thumb_0_link': 13, 'hand_thumb_1_link': 26, 'h
```

覆盖检查（`via_cover.py`，前 5 行、没通过的两条和汇总行；逐条结果在 `data_via/cover.txt`）：

```text
random paths 12000, real paths 35; a random path's isolation: median 0.770, 95 % level 1.380
feature            n_via  straight     ratio     off_x   off_out     off_z      turn        ax     a_out        az        bx     b_out        bz
random p05         0.000     0.242     1.000    -0.140    -0.228    -0.160    31.123     0.101     0.215     0.715     0.282     0.051     0.232
random p95         8.000     0.625     2.847     0.153     0.254     0.225   263.890     0.123     0.237     0.748     0.619     0.311     1.097
real min           1.000     0.243     1.007    -0.026    -0.062    -0.024    88.886     0.065     0.216     0.692     0.272     0.071     0.911
TabletopGraspMP_L0_e1              more isolated than  98.6 % of the random paths  FAIL  nearest at 0.52 spreads a feature, largest gap in off_x
TabletopGraspMP_L2_e6              more isolated than  99.2 % of the random paths  FAIL  nearest at 0.57 spreads a feature, largest gap in n_via
covered 33 of 35 real paths (95 % rule); median rule 7; inside the random paths' range in every feature 35
```

训练（`via_model.py train`，模型在 `outputs/reachwm_20261030/via_model_20261121`；code: 554a28f8ca6200146ce1f6d485eb2cdc58bca4aa + uncommitted via_model.py sha256 df6071b80bbc9ddf, plan entry sha256 83ec1c2026fe8dd1 2026-11-21T14:24:38Z；当时用的脚本原样存在模型目录里，后缀 `.asrun`）：

```text
paths 12000: train 9055, validation 1164, test 1117, sector hold-out 664; input 499
offsets: 1800 numbers a path, 157 principal components keep 99.5 % of the training variance
direction member 0: best validation loss 0.9662 at iteration 500 (14s)
direction member 1: best validation loss 0.9615 at iteration 500 (26s)
direction member 2: best validation loss 0.9531 at iteration 500 (38s)
direction member 3: best validation loss 0.9611 at iteration 500 (50s)
direction member 4: best validation loss 0.9490 at iteration 500 (63s)
direction: 8120 of 9055 training paths have an out-of-bag prediction
radius[reference] member 0: best validation loss 0.2835 at iteration 500 (6s)
radius[reference] member 1: best validation loss 0.2779 at iteration 500 (12s)
radius[reference] member 2: best validation loss 0.2734 at iteration 500 (19s)
radius[reference]: scale 1.458 puts 95% of the validation errors inside; validation error median 10.95 cm, its 95% level 18.93 cm
radius[constant] member 0: best validation loss 0.3821 at iteration 500 (6s)
radius[constant] member 1: best validation loss 0.4312 at iteration 500 (12s)
radius[constant] member 2: best validation loss 0.4265 at iteration 500 (19s)
radius[constant]: scale 1.508 puts 95% of the validation errors inside; validation error median 9.50 cm, its 95% level 18.36 cm
radius[direction] member 0: best validation loss 0.3867 at iteration 500 (6s)
radius[direction] member 1: best validation loss 0.3740 at iteration 500 (12s)
radius[direction] member 2: best validation loss 0.3690 at iteration 500 (18s)
radius[direction]: scale 1.619 puts 95% of the validation errors inside; validation error median 6.53 cm, its 95% level 14.28 cm
saved /workspace/code/locomanipulation-selfwm/outputs/reachwm_20261030/via_model_20261121/via_model.pt
```

评测（`via_model.py eval`）：

```text
validation (calibrated on these): 1164 paths
  reference  swept-region error: arm median 10.95 p90 16.97 cm | hand median  7.79 p90 13.48 cm | margin (median radius) 16.24 cm, errors inside 0.95 | fixed radius 18.93 cm, inside 0.95
  constant   swept-region error: arm median  9.50 p90 16.12 cm | hand median  7.10 p90 12.54 cm | margin (median radius) 14.67 cm, errors inside 0.95 | fixed radius 18.36 cm, inside 0.95
  direction  swept-region error: arm median  6.53 p90 12.71 cm | hand median  4.95 p90 10.02 cm | margin (median radius) 11.17 cm, errors inside 0.95 | fixed radius 14.28 cm, inside 0.95
  drop of the median against the reference: arm 0.40, hand 0.36 (constant: arm 0.13, hand 0.09)
  margins: direction / scalar radius 0.69, direction / constant 0.76
test, ordinary hold-out: 1117 paths
  reference  swept-region error: arm median 10.78 p90 16.96 cm | hand median  7.58 p90 13.37 cm | margin (median radius) 15.65 cm, errors inside 0.93 | fixed radius 18.93 cm, inside 0.94
  constant   swept-region error: arm median  9.49 p90 16.06 cm | hand median  7.09 p90 12.41 cm | margin (median radius) 14.36 cm, errors inside 0.94 | fixed radius 18.36 cm, inside 0.94
  direction  swept-region error: arm median  6.71 p90 12.71 cm | hand median  4.99 p90 10.07 cm | margin (median radius) 10.98 cm, errors inside 0.92 | fixed radius 14.28 cm, inside 0.94
  drop of the median against the reference: arm 0.38, hand 0.34 (constant: arm 0.12, hand 0.06)
  margins: direction / scalar radius 0.70, direction / constant 0.76
test, sector hold-out (inwards-down bends, never trained on): 664 paths
  reference  swept-region error: arm median  7.47 p90 14.81 cm | hand median  4.74 p90 11.00 cm | margin (median radius) 11.02 cm, errors inside 0.92 | fixed radius 18.93 cm, inside 0.96
  constant   swept-region error: arm median  7.16 p90 14.08 cm | hand median  4.94 p90 10.65 cm | margin (median radius) 11.09 cm, errors inside 0.95 | fixed radius 18.36 cm, inside 0.96
  direction  swept-region error: arm median  4.95 p90 10.47 cm | hand median  3.65 p90  8.08 cm | margin (median radius)  8.45 cm, errors inside 0.93 | fixed radius 14.28 cm, inside 0.96
  drop of the median against the reference: arm 0.34, hand 0.23 (constant: arm 0.04, hand -0.04)
  margins: direction / scalar radius 0.77, direction / constant 0.76
test, ordinary hold-out, standing: 510 paths
  reference  swept-region error: arm median  7.00 p90 12.79 cm | hand median  4.32 p90  8.99 cm | margin (median radius)  9.83 cm, errors inside 0.93 | fixed radius 18.93 cm, inside 0.98
  constant   swept-region error: arm median  6.76 p90 11.82 cm | hand median  4.50 p90  9.10 cm | margin (median radius) 10.07 cm, errors inside 0.95 | fixed radius 18.36 cm, inside 0.98
  direction  swept-region error: arm median  4.29 p90  8.60 cm | hand median  2.94 p90  6.67 cm | margin (median radius)  7.05 cm, errors inside 0.93 | fixed radius 14.28 cm, inside 0.99
  drop of the median against the reference: arm 0.39, hand 0.32 (constant: arm 0.04, hand -0.04)
  margins: direction / scalar radius 0.72, direction / constant 0.70
test, ordinary hold-out, crouching: 607 paths
  reference  swept-region error: arm median 13.30 p90 18.89 cm | hand median  9.99 p90 14.63 cm | margin (median radius) 19.09 cm, errors inside 0.94 | fixed radius 18.93 cm, inside 0.90
  constant   swept-region error: arm median 11.53 p90 18.27 cm | hand median  8.96 p90 14.07 cm | margin (median radius) 17.52 cm, errors inside 0.93 | fixed radius 18.36 cm, inside 0.90
  direction  swept-region error: arm median  8.40 p90 14.42 cm | hand median  6.75 p90 11.46 cm | margin (median radius) 14.22 cm, errors inside 0.91 | fixed radius 14.28 cm, inside 0.90
  drop of the median against the reference: arm 0.37, hand 0.32 (constant: arm 0.13, hand 0.10)
  margins: direction / scalar radius 0.74, direction / constant 0.81
real paths replayed (never trained or calibrated on): 140 paths
  reference  swept-region error: arm median  6.46 p90  9.82 cm | hand median  4.77 p90  6.37 cm | margin (median radius) 11.66 cm, errors inside 0.99 | fixed radius 18.93 cm, inside 1.00
  constant   swept-region error: arm median  7.30 p90 10.04 cm | hand median  4.86 p90  7.11 cm | margin (median radius) 13.75 cm, errors inside 1.00 | fixed radius 18.36 cm, inside 1.00
  direction  swept-region error: arm median  6.44 p90  9.01 cm | hand median  5.24 p90  7.84 cm | margin (median radius) 12.31 cm, errors inside 0.92 | fixed radius 14.28 cm, inside 1.00
  drop of the median against the reference: arm 0.00, hand -0.10 (constant: arm -0.13, hand -0.02)
  margins: direction / scalar radius 1.06, direction / constant 0.89
```

### 第五步，SONIC：采集和随机路线上的评测（2026-11-23）

采集 12000 次开场，`outputs/reachwm_20261030/data_via_sonic`（2026-11-22 14:12 到 11-23 04:03 UTC，4 个进程，0 次跳过）。模型 `outputs/reachwm_20261030/via_point_sonic_20261123`，默认设置。

数据本身（`data_via_sonic/analysis.txt`）：

```
23919 segments, 1634136 steps, fell 88, walked 16774, ok 1683
23919 segments, 1634136 ticks, 455540 reference poses, 24 spheres -> /workspace/code/locomanipulation-selfwm/outputs/reachwm_20261030/data_via_sonic/arm.npz
rebuilt jaw centre against the recorded one: median 0.046 cm, 95 % 0.141, max 2.562
last reference pose's jaw against the command: median 0.683 cm, 95 % 2.651 (the plan's own IK miss: median 0.010, 95 % 3.270)
paths 12000, fell 78, analysed 11922; plan's IK miss > 2 cm: 1911; skipped resets 0
largest offset from the reference path (hand: jaw centre; arm: any of the 24 spheres):
  all                            n 11922   hand median  8.26 p90 18.06   arm median 14.20 p90 26.34 cm
  standing                       n  5523   hand median  5.57 p90 10.63   arm median 10.46 p90 19.32 cm
  crouching (pelvis asked)       n  6399   hand median 11.32 p90 22.37   arm median 16.89 p90 30.24 cm
```

验证集上的停止线（`dev.out`）：

```
validation, standing, palm path over 1.0 m (THE STOP LINE: drop of the arm's median >= 0.25): 89 paths
  reference  swept-region error: arm median 12.58 p90 24.48 cm | hand median  9.07 cm | margin (median radius) 27.36 cm, inside 1.00 | fixed radius 31.67 cm, inside 0.97
  constant   swept-region error: arm median 12.19 p90 20.85 cm | hand median  9.02 cm | margin (median radius) 24.98 cm, inside 0.99 | fixed radius 28.45 cm, inside 0.94
  direction  swept-region error: arm median  6.37 p90 12.10 cm | hand median  4.63 cm | margin (median radius) 14.30 cm, inside 0.99 | fixed radius 18.89 cm, inside 0.99
  drop of the arm's median against the reference: 0.49 (95 % interval 0.46 to 0.55, resampling paths); hand 0.49; constant: arm 0.03
  fixed radius: direction / reference 0.60 (set on the calibration paths); direction wins on 0.97 of the paths
```

留出的随机路线（`eval.out`）：

```
test, ordinary hold-out: 1143 paths
  reference  swept-region error: arm median 14.75 p90 27.43 cm | hand median 10.66 cm | margin (median radius) 27.31 cm, inside 0.96 | fixed radius 31.67 cm, inside 0.93
  constant   swept-region error: arm median 13.48 p90 25.16 cm | hand median  9.85 cm | margin (median radius) 24.03 cm, inside 0.95 | fixed radius 28.45 cm, inside 0.93
  direction  swept-region error: arm median  7.63 p90 17.49 cm | hand median  5.62 cm | margin (median radius) 17.33 cm, inside 0.95 | fixed radius 18.89 cm, inside 0.91
  drop of the arm's median against the reference: 0.48 (95 % interval 0.47 to 0.50, resampling paths); hand 0.47; constant: arm 0.09
  fixed radius: direction / reference 0.60 (set on the calibration paths); direction wins on 0.97 of the paths
test, ordinary hold-out, standing: 501 paths
  reference  swept-region error: arm median 10.45 p90 20.09 cm | hand median  7.16 cm | margin (median radius) 19.79 cm, inside 0.99 | fixed radius 31.67 cm, inside 0.99
  constant   swept-region error: arm median 10.52 p90 19.35 cm | hand median  8.59 cm | margin (median radius) 19.27 cm, inside 0.97 | fixed radius 28.45 cm, inside 0.98
  direction  swept-region error: arm median  4.57 p90  9.32 cm | hand median  3.26 cm | margin (median radius) 10.18 cm, inside 0.98 | fixed radius 18.89 cm, inside 0.99
  drop of the arm's median against the reference: 0.56 (95 % interval 0.54 to 0.59, resampling paths); hand 0.55; constant: arm -0.01
  fixed radius: direction / reference 0.60 (set on the calibration paths); direction wins on 0.99 of the paths
test, sector hold-out: 631 paths
  reference  swept-region error: arm median 11.82 p90 21.12 cm | hand median  8.37 cm | margin (median radius) 22.08 cm, inside 0.97 | fixed radius 31.67 cm, inside 0.99
  constant   swept-region error: arm median 11.17 p90 20.27 cm | hand median  8.74 cm | margin (median radius) 20.41 cm, inside 0.97 | fixed radius 28.45 cm, inside 0.98
  direction  swept-region error: arm median  5.85 p90 11.72 cm | hand median  4.28 cm | margin (median radius) 11.51 cm, inside 0.97 | fixed radius 18.89 cm, inside 0.98
  drop of the arm's median against the reference: 0.51 (95 % interval 0.48 to 0.53, resampling paths); hand 0.49; constant: arm 0.06
  fixed radius: direction / reference 0.60 (set on the calibration paths); direction wins on 0.99 of the paths
```

- 停止线过了：降 0.49（区间 0.46–0.55），线是 0.25。
- 预测对照：站立路线手偏离参考路线的中位数 5.57 cm（预测 5–10 cm，ScaleBFM 3.7 cm）；留出集降幅 0.48（预测 0.40–0.70）。
- 63 条测试路线在 SONIC 下的回放和挪路点补偿还没跑。同一批路线在 SONIC 的站姿下起点差 4.9 cm（ScaleBFM 是 0），没有重新规划。

### 第四步补充之二：补偿器自己的回放进训练，第 1 轮（2026-11-23）

训练用的规划器路线：`outputs/reachwm_20261030/train_routes`（种子 20261123，1500 个场景规划出 624 条：越过桌沿 107、越过箱子 83、伸到板下 434；和测试、开发路线的目标一个都不重合）。取全部 190 条绕行加 150 条板下，共 340 条，各回放 4 遍（1360 次开场，种子 49000000）。

第 1 轮的三组回放都是 1360 段、0 摔倒：`onpol_r1_direct`、`onpol_r1_point`、`onpol_r1_pose`（后两组用 `via_point_20261122`）。训练路线上位置加朝向那一组的成对比较，190 条绕行 760 对：

```
detour paths (over an edge, over a box): 190 routes, 760 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.38 cm, compensated 5.41 cm; drop 0.35 (0.30 to 0.43); 90 % level 12.56 and 11.46 cm
  palm, error against the wanted sweep: direct 5.15 cm, compensated 3.81 cm; drop 0.26 (0.20 to 0.31); improved in 0.76 of the pairs
  arm,  error against the wanted sweep: direct 8.68 cm, compensated 8.08 cm; drop 0.07 (0.03 to 0.12); improved in 0.64 of the pairs
  palm's settled end against the wanted end: direct 3.40 cm, compensated 2.36 cm; drop 0.30 (0.23 to 0.39)
  hand's attitude against the wanted one: settled end direct 21.9 deg, compensated 17.6 deg; along the path 11.9 and 4.5 deg
  points moved: median 3.21 cm, largest 8.00 cm; the goal 4.42 cm; wrist commands turned: median 20.7 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.89 cm, for the one sent 2.10 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 3.21 cm, compensated 6.49 cm
```

第 1 轮模型 `via_point_onpol1_20261123`（三份随机数据 + 这三组回放）。用它在 41 条开发路线上重跑位置加朝向（`onpol_dev_r1_pose`，对照 `dev_via_pose2_20261122/direct`）：

```
pairs 164 (direct 164, compensated 164 paths without a fall; 0 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 41 routes, 164 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 7.91 cm, compensated 3.91 cm; drop 0.51 (0.40 to 0.59); 90 % level 11.83 and 9.04 cm
  palm, error against the wanted sweep: direct 4.79 cm, compensated 2.76 cm; drop 0.42 (0.35 to 0.48); improved in 0.92 of the pairs
  arm,  error against the wanted sweep: direct 8.36 cm, compensated 7.74 cm; drop 0.07 (-0.02 to 0.15); improved in 0.61 of the pairs
  palm's settled end against the wanted end: direct 3.39 cm, compensated 1.59 cm; drop 0.53 (0.46 to 0.63)
  hand's attitude against the wanted one: settled end direct 20.0 deg, compensated 9.0 deg; along the path 12.3 and 3.8 deg
  points moved: median 2.94 cm, largest 8.00 cm; the goal 3.78 cm; wrist commands turned: median 19.9 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.35 cm, for the one sent 2.36 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.27 cm, compensated 2.45 cm
```

和只挪位置那一组逐对比较（`via_vs.py`）：

```
onpol_dev_r1_pose | 164 pairs, 41 routes
  wrist + palm + fingers position-only  5.56 cm, this run  3.91 cm; drop of the median 0.30 (0.15 to 0.41); of the 90 % level  9.87 ->  9.04: 0.08 (-0.02 to 0.28); better on 0.74 of the pairs
  palm                   position-only  2.78 cm, this run  2.76 cm; drop of the median 0.01 (-0.18 to 0.14); of the 90 % level  6.09 ->  5.70: 0.06 (-0.17 to 0.22); better on 0.45 of the pairs
  arm (24 spheres)       position-only  6.12 cm, this run  7.74 cm; drop of the median -0.26 (-0.49 to -0.05); of the 90 % level 11.20 -> 12.54: -0.12 (-0.31 to 0.10); better on 0.24 of the pairs
  palm's settled end     position-only  0.78 cm, this run  1.59 cm; drop of the median -1.04 (-1.46 to -0.50); of the 90 % level  1.70 ->  4.59: -1.69 (-2.54 to -0.95); better on 0.25 of the pairs
```

按部位（`via_parts.py`）和手的朝向对命令的响应（`via_turn.py`，前四行是终点）：

```
dev_via_pose2_20261122/direct: 164 paths
  shoulder  model  1.06  plan  0.00  actual  3.83
  elbow     model  0.71  plan  0.00  actual  2.49
  wrist     model  0.82  plan  0.00  actual  3.12
  palm      model  0.89  plan  0.00  actual  4.79
  fingers   model  1.11  plan  0.00  actual  7.91
  all 24    model  1.27 (90 % 4.54)
reachwm_20261030/onpol_dev_r1_pose: 164 paths
  shoulder  model  2.02  plan  4.76  actual  7.50
  elbow     model  1.46  plan  3.92  actual  3.97
  wrist     model  1.53  plan  4.09  actual  2.80
  palm      model  1.59  plan  7.44  actual  2.76
  fingers   model  2.08  plan 12.00  actual  3.86
  all 24    model  2.45 (90 % 6.73)
reachwm_20261030/onpol_dev_r1_pose, table point 24: 164 pairs
  all                   error  20.0  commanded  29.7  actual  18.0  predicted  16.2 deg | actual / commanded 0.64, predicted / commanded 0.57 | cos actual +0.76, predicted +0.93 | error left   9.0 deg
  commanded 15-25 ( 42) error  16.9  commanded  22.4  actual  16.8  predicted  15.1 deg | actual / commanded 0.73, predicted / commanded 0.70 | cos actual +0.87, predicted +0.97 | error left   4.9 deg
  commanded 25-90 (122) error  22.2  commanded  36.0  actual  18.7  predicted  16.5 deg | actual / commanded 0.60, predicted / commanded 0.50 | cos actual +0.71, predicted +0.92 | error left  12.8 deg
```

- 开发路线上的两条检查都过了：模型对补偿回放的预测误差 2.45 cm，是它对直接发命令那一遍（1.27 cm）的 1.93 倍；腕 + 掌 + 指的中位数比只挪位置低 30 %（区间 0.15–0.41）。
- 预测对照：模型预测误差 2.45 cm（预测 3–4）；腕 + 掌 + 指 3.91 cm（预测 3.5–4.5）；手掌 2.76 cm（预测不超过 2.8）；终点朝向 9.0°（预测 6–10°）；路上朝向 3.8°（预测 4–6°）。
- 没有变好的：腕 + 掌 + 指的 90 % 分位只从 9.87 降到 9.04 cm，区间含 0；手掌终点比只挪位置差（0.78 → 1.59 cm）；整条手臂 24 个球比只挪位置差（6.12 → 7.74 cm），因为补偿朝向时躯干多倾，肩离想要的位置 7.5 cm（直接发命令 3.8 cm）。
- 终点处手实际转的方向和命令的方向余弦 0.76，模型预测 0.93，比第 0 轮（0.70 对 0.96）接近了一些，命令转 25° 以上的 122 对仍然剩 12.8°。

### 第四步补充之二：第 2 轮，到此停（2026-11-23）

第 2 轮回放 `onpol_r2_pose`（第 1 轮模型在训练路线上跑位置加朝向，1360 段、0 摔倒）。训练路线上的成对比较，190 条绕行 760 对：

```
detour paths (over an edge, over a box): 190 routes, 760 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.38 cm, compensated 3.82 cm; drop 0.54 (0.51 to 0.57); 90 % level 12.56 and 9.50 cm
  palm, error against the wanted sweep: direct 5.15 cm, compensated 2.65 cm; drop 0.48 (0.43 to 0.52); improved in 0.91 of the pairs
  arm,  error against the wanted sweep: direct 8.68 cm, compensated 7.18 cm; drop 0.17 (0.13 to 0.21); improved in 0.71 of the pairs
  palm's settled end against the wanted end: direct 3.40 cm, compensated 1.36 cm; drop 0.60 (0.53 to 0.64)
  hand's attitude against the wanted one: settled end direct 21.9 deg, compensated 10.1 deg; along the path 11.9 and 3.8 deg
  points moved: median 3.21 cm, largest 8.00 cm; the goal 4.33 cm; wrist commands turned: median 19.3 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.89 cm, for the one sent 2.34 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.15 cm, compensated 2.33 cm
```

第 2 轮模型 `via_point_onpol2_20261123`（第 1 轮的数据 + 这组回放）。用它在 41 条开发路线上重跑（`onpol_dev_r2_pose`）：

```
pairs 164 (direct 164, compensated 164 paths without a fall; 0 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 41 routes, 164 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 7.91 cm, compensated 3.79 cm; drop 0.52 (0.38 to 0.61); 90 % level 11.83 and 8.74 cm
  palm, error against the wanted sweep: direct 4.79 cm, compensated 2.71 cm; drop 0.43 (0.36 to 0.49); improved in 0.91 of the pairs
  arm,  error against the wanted sweep: direct 8.36 cm, compensated 7.72 cm; drop 0.08 (-0.02 to 0.15); improved in 0.60 of the pairs
  palm's settled end against the wanted end: direct 3.39 cm, compensated 1.35 cm; drop 0.60 (0.49 to 0.68)
  hand's attitude against the wanted one: settled end direct 20.0 deg, compensated 8.8 deg; along the path 12.3 and 3.7 deg
  points moved: median 2.95 cm, largest 8.00 cm; the goal 3.78 cm; wrist commands turned: median 20.4 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.41 cm, for the one sent 2.32 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.24 cm, compensated 2.23 cm
```

和只挪位置那一组逐对比较，再和第 1 轮那一组逐对比较（第二段里标成 position-only 的那一列其实是第 1 轮）：

```
onpol_dev_r2_pose | 164 pairs, 41 routes
  wrist + palm + fingers position-only  5.56 cm, this run  3.79 cm; drop of the median 0.32 (0.13 to 0.44); of the 90 % level  9.87 ->  8.74: 0.11 (0.02 to 0.29); better on 0.79 of the pairs
  palm                   position-only  2.78 cm, this run  2.71 cm; drop of the median 0.03 (-0.15 to 0.16); of the 90 % level  6.09 ->  5.57: 0.09 (-0.13 to 0.22); better on 0.48 of the pairs
  arm (24 spheres)       position-only  6.12 cm, this run  7.72 cm; drop of the median -0.26 (-0.49 to -0.05); of the 90 % level 11.20 -> 13.04: -0.16 (-0.33 to 0.08); better on 0.28 of the pairs
  palm's settled end     position-only  0.78 cm, this run  1.35 cm; drop of the median -0.73 (-1.25 to -0.36); of the 90 % level  1.70 ->  4.59: -1.70 (-2.56 to -1.02); better on 0.27 of the pairs
onpol_dev_r2_pose | 164 pairs, 41 routes
  wrist + palm + fingers position-only  3.91 cm, this run  3.79 cm; drop of the median 0.03 (-0.04 to 0.11); of the 90 % level  9.04 ->  8.74: 0.03 (-0.06 to 0.08); better on 0.48 of the pairs
  palm                   position-only  2.76 cm, this run  2.71 cm; drop of the median 0.02 (-0.02 to 0.08); of the 90 % level  5.70 ->  5.57: 0.02 (-0.04 to 0.10); better on 0.35 of the pairs
  arm (24 spheres)       position-only  7.74 cm, this run  7.72 cm; drop of the median 0.00 (-0.02 to 0.02); of the 90 % level 12.54 -> 13.04: -0.04 (-0.07 to 0.04); better on 0.49 of the pairs
  palm's settled end     position-only  1.59 cm, this run  1.35 cm; drop of the median 0.15 (-0.00 to 0.20); of the 90 % level  4.59 ->  4.59: -0.00 (-0.15 to 0.12); better on 0.58 of the pairs
```

按部位和终点处手的朝向对命令的响应：

```
dev_via_pose2_20261122/direct: 164 paths
  shoulder  model  0.91  plan  0.00  actual  3.83
  elbow     model  0.62  plan  0.00  actual  2.49
  wrist     model  0.79  plan  0.00  actual  3.12
  palm      model  0.89  plan  0.00  actual  4.79
  fingers   model  1.12  plan  0.00  actual  7.91
  all 24    model  1.24 (90 % 4.41)
reachwm_20261030/onpol_dev_r2_pose: 164 paths
  shoulder  model  1.58  plan  4.72  actual  7.47
  elbow     model  1.31  plan  3.94  actual  3.83
  wrist     model  1.31  plan  4.04  actual  2.69
  palm      model  1.46  plan  7.75  actual  2.71
  fingers   model  2.02  plan 11.73  actual  3.71
  all 24    model  2.23 (90 % 6.24)
reachwm_20261030/onpol_dev_r2_pose, table point 24: 164 pairs
  all                   error  20.0  commanded  30.6  actual  17.8  predicted  15.8 deg | actual / commanded 0.63, predicted / commanded 0.56 | cos actual +0.77, predicted +0.92 | error left   8.8 deg
  commanded 15-25 ( 37) error  16.4  commanded  22.7  actual  16.7  predicted  15.2 deg | actual / commanded 0.75, predicted / commanded 0.68 | cos actual +0.84, predicted +0.97 | error left   4.3 deg
  commanded 25-90 (127) error  22.0  commanded  36.5  actual  18.6  predicted  16.1 deg | actual / commanded 0.58, predicted / commanded 0.50 | cos actual +0.72, predicted +0.89 | error left  12.3 deg
```

- **按事先写的规则在第 2 轮停**：腕 + 掌 + 指的中位数 3.91 → 3.79 cm，比上一轮改善 3 %（区间 −4 % 到 11 %），不到 5 %。测试集用第 2 轮的模型。
- 开发路线上的两条检查都过：模型对补偿回放的预测误差 2.23 cm，是它对直接发命令那一遍（1.24 cm）的 1.80 倍；腕 + 掌 + 指的中位数比只挪位置低 32 %（区间 0.13–0.44），90 % 分位低 11 %（区间 0.02–0.29）。
- 没有变好的和第 1 轮一样：手掌终点比只挪位置差（0.78 → 1.35 cm），整条手臂 24 个球比只挪位置差（6.12 → 7.72 cm，肩离想要的位置 7.5 cm）。

### 第四步补充之二：测试集，只跑了这一次（2026-11-23）

`outputs/reachwm_20261030/data_via_pose`：63 条测试路线 × 4 遍，位置加朝向，第 2 轮模型，设置和开发路线上相同。252 段，0 摔倒，0 跳过。对照是原来那一遍直接发命令的回放 `data_via_fresh`。

```
pairs 252 (direct 252, compensated 252 paths without a fall; 0 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 21 routes, 84 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.06 cm, compensated 3.40 cm; drop 0.58 (0.52 to 0.67); 90 % level 11.51 and 6.80 cm
  palm, error against the wanted sweep: direct 4.83 cm, compensated 2.40 cm; drop 0.50 (0.39 to 0.56); improved in 0.96 of the pairs
  arm,  error against the wanted sweep: direct 8.10 cm, compensated 7.33 cm; drop 0.10 (-0.06 to 0.25); improved in 0.65 of the pairs
  palm's settled end against the wanted end: direct 3.01 cm, compensated 1.10 cm; drop 0.64 (0.53 to 0.72)
  hand's attitude against the wanted one: settled end direct 19.3 deg, compensated 7.1 deg; along the path 11.7 and 3.3 deg
  points moved: median 2.90 cm, largest 8.00 cm; the goal 3.70 cm; wrist commands turned: median 19.5 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.58 cm, for the one sent 2.16 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.32 cm, compensated 1.81 cm
```

窄缝计分 `outputs/reachwm_20261030/via_gap_20261123`（对手的余量取自 `via_point_20261121`，我们的余量缩放取自 `onpol_dev_r2_pose`）：

```
through: ours' residual scale 1.95 on the training side -> 3.04 on 164 reaches of onpol_dev_r2_pose (their residual: median 2.02 cm, 95 % 6.71 cm)
hand: ours' residual scale 1.97 on the training side -> 2.99 on 164 reaches of onpol_dev_r2_pose (their residual: median 2.02 cm, 95 % 6.71 cm)
arm: ours' residual scale 1.90 on the training side -> 3.04 on 164 reaches of onpol_dev_r2_pose (their residual: median 2.23 cm, 95 % 7.53 cm)
through: fixed margin 19.39 cm (direct), 19.01 cm (after the mean offset), 11.91 cm (after the point model); radius scales 1.49 / 3.04
hand: fixed margin 19.39 cm (direct), 19.01 cm (after the mean offset), 11.91 cm (after the point model); radius scales 1.52 / 2.99
arm: fixed margin 19.85 cm (direct), 19.28 cm (after the mean offset), 11.94 cm (after the point model); radius scales 1.43 / 3.04

detour routes, spheres 'through': 21 routes, 84 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 8.06 / 11.51 / 12.35; const 8.88 / 11.92 / 13.11; ours 3.40 / 6.80 / 9.61
  half-width cm          2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small   success    0.00  0.00  0.00  0.10  0.15  0.26  0.49  0.62  0.77  0.86  0.93  0.96  0.98  0.99
          collide    0.00  0.00  1.00  0.90  0.85  0.74  0.51  0.38  0.23  0.14  0.07  0.04  0.02  0.01
  fixed   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius  success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.06  0.14  0.18  0.27  0.39  0.50
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  ours    success    0.00  0.00  0.01  0.12  0.30  0.49  0.62  0.65  0.71  0.82  0.87  0.90  0.90  0.93
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small    12.50  (11.00 to 14.50)
    fixed      nan  (nan to nan)
    radius     nan  (nan to nan)
    const      nan  (nan to nan)
    ours     12.75  (10.50 to 15.00)
  ours / best other way (small): 1.02 (0.88 to 1.23)
  at a half-width of 5 cm, success / collision / abstention: small 0.10 / 0.90 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.00 / 0.00 / 1.00; const 0.00 / 0.00 / 1.00; ours 0.12 / 0.00 / 0.88

detour routes without the 8 looked at during development, spheres 'through': 13 routes, 52 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 7.79 / 10.16 / 10.91; const 8.48 / 10.60 / 11.31; ours 3.61 / 6.45 / 6.97
  half-width cm          2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small   success    0.00  0.00  0.00  0.08  0.17  0.29  0.52  0.69  0.85  0.94  0.98  1.00  1.00  1.00
          collide    0.00  0.00  1.00  0.92  0.83  0.71  0.48  0.31  0.15  0.06  0.02  0.00  0.00  0.00
  fixed   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius  success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.04  0.15  0.21  0.35  0.46  0.60
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const   success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  ours    success    0.00  0.00  0.00  0.10  0.31  0.52  0.63  0.65  0.73  0.90  0.96  0.96  0.96  0.98
          collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small    11.25  (10.00 to 12.25)
    fixed      nan  (nan to nan)
    radius     nan  (14.51 to 14.75)
    const      nan  (nan to nan)
    ours     11.00  (10.00 to 11.75)
  ours / best other way (small): 0.98 (0.88 to 1.15)
  at a half-width of 5 cm, success / collision / abstention: small 0.08 / 0.92 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.00 / 0.00 / 1.00; const 0.00 / 0.00 / 1.00; ours 0.10 / 0.00 / 0.90

near-straight routes (under a board), spheres 'through': 42 routes, 168 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 2.79 / 4.34 / 5.17; const 3.53 / 5.59 / 6.40; ours 2.18 / 3.01 / 3.12
    small     5.25  (4.25 to 5.75)
    ours      5.00  (4.75 to 5.25)
  ours / best other way (small): 0.95 (0.83 to 1.12)
```

**对照事先写的判据：**

- 一致性检查三项都是 0.00 mm；摔倒 0 次，和直接发命令相同；模型对这一遍的预测误差 1.81 cm，是直接发命令那一遍（1.32 cm）的 1.37 倍，不超过 2 倍。
- 预测对照：腕 + 掌 + 指的中位数 3.40 cm（预测 3.5–5.0，略好于预测）；终点朝向 7.1°（预测 5–10°）；路上朝向 3.3°（预测 4–8°，略好）。
- **窄缝比值 1.02（区间 0.88–1.23），按判据属于"> 0.95，没有优势"。** 我们 12.75 cm，小固定余量 12.50 cm。预测是 0.6–0.8，预测错了。去掉被看过的 8 条路线后是 11.00 对 11.25 cm，比值 0.98（0.88–1.15），结论一样。
- "能穿过明显更窄的缝"按事先写的计分方式不成立。

**为什么误差降了一半多，缝却没有变窄：**

- 我们这一方在每个缝宽下的碰撞率都是 0（允许 5 %），成功率到 12.75 cm 才够 90 %。也就是它放弃得太多。
- 放弃的规则是"预测的误差 + 残差半径 ≤ 缝的半宽"。残差半径是模型预测和实际之间的差的 95 % 上界，在开发路线上标定后缩放系数是 3.04（训练侧是 1.95），开发路线上残差的中位数 2.02 cm、95 % 分位 6.71 cm。两项直接相加是两个上界叠在一起，实际误差通常远小于它们的和。
- 小固定余量这个对手不放弃，它的"最窄的缝"实际上是事后从测试集的误差分布里读出来的（95 % 分位）。两边不对等：我们要事先判断，它不用。这一点在第一部分的设计里就是这样，当时没有显出来。

**事后的探索，不算证据**（`via_gap_20261123/gap_rules_exploratory.py` 和 `.txt`；规则是看到上面的结果之后才想的，余量只用开发路线定，没有用测试集定任何数）：

```
development run: 164 reaches | predicted error median 2.78 cm, actual 3.79 cm, actual - predicted: median +0.34, 95 % +3.28 cm; actual 95 % 10.32 cm; corr(predicted, actual) 0.88
test run, detour: 84 reaches | predicted median 2.69, actual 3.40 (90 % 6.80, 95 % 9.61); corr(predicted, actual) 0.83
detour routes: 84 reaches; margins from the development run: B 3.28 cm on top of the prediction, C 10.32 cm
  small margin, always goes (direct) narrowest 12.50 cm (11.00 to 14.50) | success / collision at 6, 8, 10 cm: 0.15 / 0.85; 0.49 / 0.51; 0.77 / 0.23
  ours, rule B                       narrowest  9.00 cm (7.75 to 15.00) | success / collision at 6, 8, 10 cm: 0.51 / 0.01; 0.83 / 0.01; 0.92 / 0.00
  ours, rule C (one fixed margin)    narrowest 10.50 cm (10.50 to 10.75) | success / collision at 6, 8, 10 cm: 0.00 / 0.00; 0.00 / 0.00; 0.00 / 0.00
  ours, always goes                  narrowest 10.00 cm (6.25 to 10.75) | success / collision at 6, 8, 10 cm: 0.81 / 0.19; 0.92 / 0.08; 0.96 / 0.04
    ratio to the small margin, B: 0.72 (0.64 to 1.20)
    ratio to the small margin, C: 0.84 (0.72 to 0.95)
    ratio to the small margin, always: 0.80 (0.48 to 0.85)
detour routes not looked at before: 52 reaches; margins from the development run: B 3.28 cm on top of the prediction, C 10.32 cm
  small margin, always goes (direct) narrowest 11.25 cm (10.00 to 12.25) | success / collision at 6, 8, 10 cm: 0.17 / 0.83; 0.52 / 0.48; 0.85 / 0.15
  ours, rule B                       narrowest  8.25 cm (7.24 to 8.75) | success / collision at 6, 8, 10 cm: 0.48 / 0.02; 0.88 / 0.02; 1.00 / 0.00
  ours, rule C (one fixed margin)    narrowest 10.50 cm (10.50 to 10.50) | success / collision at 6, 8, 10 cm: 0.00 / 0.00; 0.00 / 0.00; 0.00 / 0.00
  ours, always goes                  narrowest  7.25 cm (5.50 to 8.75) | success / collision at 6, 8, 10 cm: 0.83 / 0.17; 0.96 / 0.04; 1.00 / 0.00
    ratio to the small margin, B: 0.73 (0.68 to 0.80)
    ratio to the small margin, C: 0.93 (0.86 to 1.05)
    ratio to the small margin, always: 0.64 (0.55 to 0.76)
```

- 规则 B：预测的误差加上一个固定余量就出发，余量取开发路线上"实际 − 预测"的 95 % 分位（3.28 cm）。测试集绕行路线上最窄 9.00 cm，对小固定余量的比值 0.72（区间 0.64–1.20）；没被看过的 13 条路线上 8.25 对 11.25 cm，比值 0.73（0.68–0.80）；8 cm 的缝成功 83 %、碰撞 1 %（小固定余量 49 % 和 51 %）。
- 这说明误差的下降在一个不那么保守的出发规则下能变成更窄的缝，但规则是事后选的。要让它算数，得先把规则 B 写成判据，再在一批新规划的测试路线上只跑一次。

### 第五步，SONIC：63 条测试路线上的评测和挪路点补偿（2026-11-23）

停止线过了之后才跑的测试路线。直接回放 `data_via_fresh_sonic`（252 段，0 摔倒），按路线类型（`via_point_sonic_20261123/by_kind.txt`，reference 是照参考走，direction 是模型）：

```
all                                routes 63 replays 252 | reference  6.73 constant  8.60 direction  3.34 cm | drop 0.50 (0.47 to 0.53) | direction better on 0.98 | palm path 0.47 m
FreshBox                           routes 12 replays  48 | reference  8.68 constant  8.94 direction  4.02 cm | drop 0.54 (0.44 to 0.61) | direction better on 0.96 | palm path 0.83 m
FreshEdge                          routes  9 replays  36 | reference  8.18 constant  9.05 direction  4.22 cm | drop 0.48 (0.38 to 0.59) | direction better on 0.97 | palm path 0.85 m
FreshUnder                         routes 42 replays 168 | reference  6.00 constant  8.31 direction  3.23 cm | drop 0.46 (0.43 to 0.50) | direction better on 0.99 | palm path 0.42 m
first pass (no walk)               routes 63 replays  63 | reference  6.04 constant  8.65 direction  2.49 cm | drop 0.59 (0.54 to 0.66) | direction better on 1.00 | palm path 0.45 m
later passes (walked)              routes 63 replays 189 | reference  6.87 constant  8.56 direction  3.67 cm | drop 0.47 (0.43 to 0.50) | direction better on 0.98 | palm path 0.48 m
detour (box + edge)                routes 21 replays  84 | reference  8.55 direction  4.08 cm | drop 0.52 (0.45 to 0.59) | direction better on 0.96
```

挪路点补偿（只挪位置）`data_via_comp_sonic`（252 段，0 摔倒），成对比较：

```
pairs 252 (direct 252, compensated 252 paths without a fall; 0 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 21 routes, 84 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.39 cm, compensated 4.81 cm; drop 0.43 (0.37 to 0.50); 90 % level 12.44 and 7.55 cm
  palm, error against the wanted sweep: direct 6.40 cm, compensated 3.09 cm; drop 0.52 (0.44 to 0.58); improved in 0.98 of the pairs
  arm,  error against the wanted sweep: direct 8.55 cm, compensated 5.17 cm; drop 0.40 (0.32 to 0.46); improved in 0.95 of the pairs
  palm's settled end against the wanted end: direct 4.54 cm, compensated 1.58 cm; drop 0.65 (0.55 to 0.74)
  hand's attitude against the wanted one: settled end direct 13.9 deg, compensated 11.3 deg; along the path 8.2 and 8.6 deg
  points moved: median 2.72 cm, largest 8.00 cm; the goal 3.74 cm; wrist commands turned: median 0.0 deg, largest 0.0 deg
  palm, the error the model predicted: for the original command 4.62 cm, for the one sent 1.08 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 4.08 cm, compensated 3.77 cm
```

- 判据都过了：模型在 21 条绕行路线上降 52 %（区间 45–59 %，线是 20 % 且区间不含 0）；补偿后手掌误差降 52 %（区间 44–58 %，线同上）。
- 预测对照：新路线上的降幅 52 %（预测 30–60 %）；补偿后手掌降 52 %（预测 20–50 %，略高于预测）。
- 和 ScaleBFM 在同一批路线上并排（只挪位置的补偿）：

  | | ScaleBFM | SONIC |
  |---|---|---|
  | 模型：整条手臂扫过区域误差 | 8.10 → 3.14 cm（降 61 %） | 8.55 → 4.08 cm（降 52 %） |
  | 补偿：手掌到想要路线 | 4.83 → 2.99 cm（降 38 %） | 6.40 → 3.09 cm（降 52 %） |
  | 补偿：整条手臂 24 个球 | 8.10 → 6.08 cm（降 25 %） | 8.55 → 5.17 cm（降 40 %） |
  | 补偿：手掌终点 | 3.01 → 0.77 cm | 4.54 → 1.58 cm |

- SONIC 下整条手臂也跟着改善，ScaleBFM 下肩和肘没有。SONIC 的接口只收手腕和头的目标，手腕挪了，手臂跟着走；ScaleBFM 跟的是整条手臂的关键点。
- 同一批路线在 SONIC 的站姿下起点差 4.9 cm，没有重新规划。

### 第四步补充之三：新测试路线，新的出发规则，只跑了这一次（2026-11-23）

新测试路线 `outputs/reachwm_20261030/test2_routes`（种子 20261124，450 个场景规划出 166 条：越过桌沿 24、越过箱子 21、伸到板下 121；和另外三组路线的目标一个都不重合）。三组回放各 664 段、0 摔倒、0 跳过：`test2_direct`、`test2_const`、`test2_pose`。成对比较：

```
pairs 664 (direct 664, compensated 664 paths without a fall; 0 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 45 routes, 180 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.39 cm, compensated 3.88 cm; drop 0.54 (0.48 to 0.61); 90 % level 12.27 and 9.28 cm
  palm, error against the wanted sweep: direct 5.17 cm, compensated 2.55 cm; drop 0.51 (0.41 to 0.55); improved in 0.91 of the pairs
  arm,  error against the wanted sweep: direct 8.54 cm, compensated 7.42 cm; drop 0.13 (0.07 to 0.20); improved in 0.64 of the pairs
  palm's settled end against the wanted end: direct 3.52 cm, compensated 1.14 cm; drop 0.68 (0.59 to 0.75)
  hand's attitude against the wanted one: settled end direct 21.3 deg, compensated 8.9 deg; along the path 12.4 and 3.9 deg
  points moved: median 3.20 cm, largest 8.00 cm; the goal 4.19 cm; wrist commands turned: median 21.6 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 5.00 cm, for the one sent 2.28 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.44 cm, compensated 1.88 cm
```

窄缝计分 `outputs/reachwm_20261030/via_gap_test2_20261123`（`--rule margin`；ours_sum 是原来那条规则，只作对照）：

```
through: actual minus predicted error on onpol_dev_r2_pose: median +0.34 cm, 95 % level +3.28 cm (the margin of --rule margin)
hand: actual minus predicted error on onpol_dev_r2_pose: median +0.28 cm, 95 % level +3.54 cm (the margin of --rule margin)
arm: actual minus predicted error on onpol_dev_r2_pose: median +0.52 cm, 95 % level +2.91 cm (the margin of --rule margin)
through: direct command's error on dev_via_pose2_20261122/direct: 95 % level 12.56 cm (fixed_dev)
hand: direct command's error on dev_via_pose2_20261122/direct: 95 % level 12.56 cm (fixed_dev)
arm: direct command's error on dev_via_pose2_20261122/direct: 95 % level 12.63 cm (fixed_dev)
replays: 664 triples of 664; wanted sweep differs between the runs by at most 0.00 mm

detour routes, spheres 'through': 45 routes, 180 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 8.39 / 12.27 / 13.60; const 8.96 / 12.99 / 13.79; ours 3.88 / 9.28 / 13.84
  half-width cm            2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small     success    0.00  0.00  0.02  0.08  0.14  0.24  0.45  0.63  0.76  0.84  0.88  0.93  0.95  0.97
            collide    0.00  0.00  0.98  0.92  0.86  0.76  0.55  0.37  0.24  0.16  0.12  0.07  0.05  0.03
  fixed     success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius    success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.03  0.06  0.09  0.11  0.17  0.26  0.38
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const     success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  fixed_dev success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.93  0.95  0.97
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.07  0.05  0.03
  ours      success    0.00  0.00  0.00  0.11  0.46  0.59  0.70  0.79  0.83  0.87  0.89  0.89  0.91  0.96
            collide    0.00  0.00  0.00  0.00  0.00  0.02  0.01  0.00  0.00  0.00  0.00  0.00  0.02  0.01
  ours_sum  success    0.00  0.01  0.02  0.14  0.29  0.41  0.51  0.58  0.62  0.65  0.71  0.77  0.79  0.83
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small      13.75  (12.00 to 14.75)
    fixed        nan  (nan to nan)
    radius       nan  (nan to nan)
    const        nan  (nan to nan)
    fixed_dev  13.75  (12.75 to 14.75)
    ours       13.25  (9.25 to 15.00)
    ours_sum     nan  (12.64 to 15.00)
  ours / best other way (small): 0.96 (0.73 to 1.10)
  at a half-width of 5 cm, success / collision / abstention: small 0.08 / 0.92 / 0.00; fixed 0.00 / 0.00 / 1.00; radius 0.00 / 0.00 / 1.00; const 0.00 / 0.00 / 1.00; fixed_dev 0.00 / 0.00 / 1.00; ours 0.11 / 0.00 / 0.89; ours_sum 0.14 / 0.00 / 0.86


near-straight routes (under a board), spheres 'through': 121 routes, 484 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 2.90 / 4.61 / 5.44; const 3.78 / 5.73 / 6.79; ours 2.23 / 2.89 / 3.01
    small       5.50  (4.75 to 6.25)
    fixed_dev  12.75  (12.75 to 12.75)
    ours        6.25  (6.25 to 6.25)
  ours / best other way (small): 1.14 (1.00 to 1.32)

detour routes, spheres 'arm': 45 routes, 180 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 8.54 / 12.65 / 14.21; const 9.13 / 13.11 / 13.79; ours 7.42 / 14.65 / 16.21
    small      14.25  (12.50 to 14.75)
    fixed_dev  14.25  (12.75 to 14.75)
    ours         nan  (11.75 to 15.00)
  ours / best other way (small): nan (0.89 to 1.16)
```

**对照事先写的判据：**

- 三项一致性检查 0.00 mm；摔倒 0 次；模型对补偿回放的预测误差 1.88 cm，是直接发命令那一遍（1.44 cm）的 1.31 倍。这三条都满足，结果算数。
- 误差的预测对了：直接发命令 8.39 cm（预测 7.5–8.5），补偿后 3.88 cm（预测 3.2–4.0）。
- **窄缝比值 0.96（区间 0.73–1.10），按判据不成立。** 我们 13.25 cm，小固定余量和在开发路线上标定的固定余量都是 13.75 cm。预测是 0.65–0.85，又错了。原测试集上事后得到的 0.72 没有重现。
- 我们在自己最窄的半宽上碰撞率 1–2 %，不超过 5 %，余量在新路线上是够的。

**为什么：有一成的路线补不动，而"90 % 成功"正好卡在这一成上。**

- 180 次绕行伸手里有 20 次（11 %，9 条路线）补偿之后误差仍超过 9 cm。这 20 次里 95 % 的位置修正顶到了 8 cm 上限，100 % 的手腕转角顶到了 40° 上限；它们直接发命令时的误差中位数是 13.5 cm（其余 8.0 cm）。也就是控制器偏得太多，超出了补偿允许挪的范围。
- 模型知道这些补不动：对它们预测的剩余手掌误差是 6.9 cm（其余 2.2 cm），所以规则让它们放弃，没有去撞。成功率因此在 11–13 cm 一直停在 87–89 %。
- 原测试集上这样的伸手只有 6 %（84 次里 5 次），所以那里成功率能过 90 %。两批路线的差别就在难路线占 6 % 还是 11 %。

**两批测试路线上都成立的（表里的逐行数字，不是事先写的判据）：**

- 同样宽的缝，我们通过的比例高得多，而且几乎不碰撞。新路线上半宽 8 cm：我们成功 70 %、碰撞 1 %、放弃 29 %；小固定余量成功 45 %、碰撞 55 %；在开发路线上标定的固定余量到 13 cm 才开始出发。半宽 6 cm：我们 46 % 对 14 %。
- 小固定余量的碰撞率要到 13–14 cm 才降到 5 % 以下；我们在每个半宽下都不超过 2 %。

**结论：**"能安全通过的最窄的缝更窄"这个说法两次都没有通过事先写的判据，不能用。能用的说法是"同样宽的缝，通过率高、几乎不碰撞，补不动的时候自己知道放弃"。要让最窄的缝也变窄，得把那一成补不动的路线解决掉：放宽 8 cm 和 40° 这两个上限，并让模型在更大的修正下也见过数据。这没有做。

### 第四步补充之四：两轮回放和开发路线上的对照（2026-11-23）

2026-11-23 07:45–09:43 UTC，vastgpu，2 个进程（同时 4 个进程在采 AMO），提交 `91f03b5a` 加未提交的代码改动（`gen_reach.py`、`via_comp.py`、`scalebfm/motor.py`，各运行目录的 `asrun/` 里有当时的副本）。链条脚本 `outputs/reachwm_20261030/onpol_chain_arm.sh`。训练路线上的两轮回放 `onpol_r3_arm`、`onpol_r4_arm`（种子 49000000），模型 `via_point_onpol3_20261123`、`via_point_onpol4_20261123`，开发路线上的四遍 `onpol_dev_r3_arm`、`onpol_dev_r4_arm`、`onpol_dev_r4_arm_narrow`、`onpol_dev_r4_pose`（种子 47000000）。

各遍和直接发命令成对比较（`via_comp_eval.py`，每个运行目录的 `comp_eval.out`）：

```
09:43:50
train done rc=0 08:34:54
dev r3 done 08:40:06
train done rc=0 09:26:11
dev r4 done 09:43:00
== onpol_r3_arm: finished 2026-11-23T08:22:33Z 1360 segments, 91469 steps, fell 0, walked 1020, ok 482
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 190 routes, 760 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.38 cm, compensated 3.56 cm; drop 0.57 (0.55 to 0.60); 90 % level 12.56 and 10.86 cm
  palm, error against the wanted sweep: direct 5.15 cm, compensated 2.55 cm; drop 0.50 (0.46 to 0.54); improved in 0.87 of the pairs
  arm,  error against the wanted sweep: direct 8.68 cm, compensated 7.09 cm; drop 0.18 (0.15 to 0.23); improved in 0.70 of the pairs
  palm's settled end against the wanted end: direct 3.40 cm, compensated 1.29 cm; drop 0.62 (0.56 to 0.67)
  hand's attitude against the wanted one: settled end direct 21.9 deg, compensated 8.9 deg; along the path 11.9 and 3.6 deg
  points moved: median 2.96 cm, largest 12.00 cm; the goal 3.54 cm; wrist commands turned: median 19.4 deg, largest 60.0 deg
  palm, the error the model predicted: for the original command 4.96 cm, for the one sent 2.22 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.12 cm, compensated 2.58 cm
== onpol_r4_arm: finished 2026-11-23T09:14:48Z 1360 segments, 91555 steps, fell 0, walked 1020, ok 478
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 190 routes, 760 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.38 cm, compensated 3.49 cm; drop 0.58 (0.55 to 0.61); 90 % level 12.56 and 11.49 cm
  palm, error against the wanted sweep: direct 5.15 cm, compensated 2.52 cm; drop 0.51 (0.48 to 0.54); improved in 0.86 of the pairs
  arm,  error against the wanted sweep: direct 8.68 cm, compensated 7.14 cm; drop 0.18 (0.14 to 0.22); improved in 0.67 of the pairs
  palm's settled end against the wanted end: direct 3.40 cm, compensated 1.22 cm; drop 0.64 (0.60 to 0.68)
  hand's attitude against the wanted one: settled end direct 21.9 deg, compensated 8.4 deg; along the path 11.9 and 3.6 deg
  points moved: median 2.93 cm, largest 12.00 cm; the goal 3.55 cm; wrist commands turned: median 19.4 deg, largest 60.0 deg
  palm, the error the model predicted: for the original command 4.97 cm, for the one sent 2.28 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.11 cm, compensated 1.98 cm
== onpol_dev_r3_arm: finished 2026-11-23T08:40:06Z 164 segments, 13542 steps, fell 0, walked 123, ok 20
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 41 routes, 164 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 7.91 cm, compensated 3.27 cm; drop 0.59 (0.47 to 0.64); 90 % level 11.83 and 10.41 cm
  palm, error against the wanted sweep: direct 4.79 cm, compensated 2.65 cm; drop 0.45 (0.36 to 0.51); improved in 0.83 of the pairs
  arm,  error against the wanted sweep: direct 8.36 cm, compensated 7.36 cm; drop 0.12 (0.04 to 0.19); improved in 0.63 of the pairs
  palm's settled end against the wanted end: direct 3.39 cm, compensated 1.41 cm; drop 0.59 (0.44 to 0.67)
  hand's attitude against the wanted one: settled end direct 20.0 deg, compensated 8.1 deg; along the path 12.3 and 3.6 deg
  points moved: median 2.69 cm, largest 12.00 cm; the goal 3.14 cm; wrist commands turned: median 19.6 deg, largest 60.0 deg
  palm, the error the model predicted: for the original command 4.41 cm, for the one sent 2.25 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.26 cm, compensated 2.09 cm
== onpol_dev_r4_arm: finished 2026-11-23T09:32:34Z 164 segments, 13555 steps, fell 0, walked 123, ok 22
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 41 routes, 164 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 7.91 cm, compensated 3.41 cm; drop 0.57 (0.46 to 0.63); 90 % level 11.83 and 11.35 cm
  palm, error against the wanted sweep: direct 4.79 cm, compensated 2.69 cm; drop 0.44 (0.33 to 0.50); improved in 0.82 of the pairs
  arm,  error against the wanted sweep: direct 8.36 cm, compensated 7.44 cm; drop 0.11 (0.03 to 0.18); improved in 0.62 of the pairs
  palm's settled end against the wanted end: direct 3.39 cm, compensated 1.50 cm; drop 0.56 (0.45 to 0.68)
  hand's attitude against the wanted one: settled end direct 20.0 deg, compensated 8.1 deg; along the path 12.3 and 3.6 deg
  points moved: median 2.68 cm, largest 12.00 cm; the goal 3.17 cm; wrist commands turned: median 20.0 deg, largest 60.0 deg
  palm, the error the model predicted: for the original command 4.48 cm, for the one sent 2.31 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.23 cm, compensated 2.06 cm
== onpol_dev_r4_arm_narrow: finished 2026-11-23T09:38:05Z 164 segments, 13518 steps, fell 0, walked 123, ok 22
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 41 routes, 164 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 7.91 cm, compensated 3.33 cm; drop 0.58 (0.50 to 0.64); 90 % level 11.83 and 7.49 cm
  palm, error against the wanted sweep: direct 4.79 cm, compensated 2.55 cm; drop 0.47 (0.39 to 0.52); improved in 0.90 of the pairs
  arm,  error against the wanted sweep: direct 8.36 cm, compensated 7.41 cm; drop 0.11 (0.03 to 0.18); improved in 0.65 of the pairs
  palm's settled end against the wanted end: direct 3.39 cm, compensated 1.29 cm; drop 0.62 (0.53 to 0.70)
  hand's attitude against the wanted one: settled end direct 20.0 deg, compensated 8.1 deg; along the path 12.3 and 3.6 deg
  points moved: median 2.69 cm, largest 8.00 cm; the goal 3.17 cm; wrist commands turned: median 20.0 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.48 cm, for the one sent 2.22 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.23 cm, compensated 2.04 cm
== onpol_dev_r4_pose: finished 2026-11-23T09:42:55Z 164 segments, 13491 steps, fell 0, walked 123, ok 5
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 41 routes, 164 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 7.91 cm, compensated 3.70 cm; drop 0.53 (0.38 to 0.61); 90 % level 11.83 and 8.94 cm
  palm, error against the wanted sweep: direct 4.79 cm, compensated 2.76 cm; drop 0.42 (0.35 to 0.49); improved in 0.91 of the pairs
  arm,  error against the wanted sweep: direct 8.36 cm, compensated 7.72 cm; drop 0.08 (-0.02 to 0.15); improved in 0.62 of the pairs
  palm's settled end against the wanted end: direct 3.39 cm, compensated 1.19 cm; drop 0.65 (0.53 to 0.72)
  hand's attitude against the wanted one: settled end direct 20.0 deg, compensated 8.3 deg; along the path 12.3 and 3.7 deg
  points moved: median 2.97 cm, largest 8.00 cm; the goal 3.88 cm; wrist commands turned: median 20.7 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.48 cm, for the one sent 2.36 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.23 cm, compensated 1.87 cm
```

第 4 轮模型的三遍互相比（`via_vs.py`，当时的版本把基线那一列标成 position-only）。四段的基线依次是：`onpol_dev_r4_pose`、`onpol_dev_r4_pose`、`onpol_dev_r4_arm_narrow`、`onpol_dev_r3_arm`：

```
onpol_dev_r4_arm_narrow | 164 pairs, 41 routes
  wrist + palm + fingers position-only  3.70 cm, this run  3.33 cm; drop of the median 0.10 (0.01 to 0.22); of the 90 % level  8.94 ->  7.49: 0.16 (-0.11 to 0.21); better on 0.54 of the pairs
  palm                   position-only  2.76 cm, this run  2.55 cm; drop of the median 0.08 (-0.00 to 0.14); of the 90 % level  5.24 ->  5.22: 0.00 (-0.10 to 0.23); better on 0.45 of the pairs
  arm (24 spheres)       position-only  7.72 cm, this run  7.41 cm; drop of the median 0.04 (0.01 to 0.10); of the 90 % level 13.19 -> 12.11: 0.08 (-0.06 to 0.12); better on 0.78 of the pairs
  palm's settled end     position-only  1.19 cm, this run  1.29 cm; drop of the median -0.09 (-0.23 to 0.09); of the 90 % level  4.13 ->  3.72: 0.10 (-0.13 to 0.21); better on 0.47 of the pairs
onpol_dev_r4_arm | 164 pairs, 41 routes
  wrist + palm + fingers position-only  3.70 cm, this run  3.41 cm; drop of the median 0.08 (-0.02 to 0.19); of the 90 % level  8.94 -> 11.35: -0.27 (-1.31 to 0.12); better on 0.44 of the pairs
  palm                   position-only  2.76 cm, this run  2.69 cm; drop of the median 0.03 (-0.10 to 0.10); of the 90 % level  5.24 ->  7.67: -0.46 (-1.30 to -0.05); better on 0.35 of the pairs
  arm (24 spheres)       position-only  7.72 cm, this run  7.44 cm; drop of the median 0.04 (0.01 to 0.10); of the 90 % level 13.19 -> 15.00: -0.14 (-0.49 to 0.05); better on 0.68 of the pairs
  palm's settled end     position-only  1.19 cm, this run  1.50 cm; drop of the median -0.26 (-0.43 to 0.00); of the 90 % level  4.13 ->  5.55: -0.34 (-1.01 to 0.04); better on 0.33 of the pairs
onpol_dev_r4_arm | 164 pairs, 41 routes
  wrist + palm + fingers position-only  3.33 cm, this run  3.41 cm; drop of the median -0.03 (-0.12 to 0.02); of the 90 % level  7.49 -> 11.35: -0.52 (-1.50 to -0.01); better on 0.13 of the pairs
  palm                   position-only  2.55 cm, this run  2.69 cm; drop of the median -0.05 (-0.15 to 0.00); of the 90 % level  5.22 ->  7.67: -0.47 (-1.34 to -0.11); better on 0.05 of the pairs
  arm (24 spheres)       position-only  7.41 cm, this run  7.44 cm; drop of the median -0.00 (-0.02 to 0.01); of the 90 % level 12.11 -> 15.00: -0.24 (-0.60 to -0.02); better on 0.15 of the pairs
  palm's settled end     position-only  1.29 cm, this run  1.50 cm; drop of the median -0.16 (-0.31 to -0.01); of the 90 % level  3.72 ->  5.55: -0.49 (-1.16 to -0.03); better on 0.12 of the pairs
onpol_dev_r4_arm | 164 pairs, 41 routes
  wrist + palm + fingers position-only  3.27 cm, this run  3.41 cm; drop of the median -0.04 (-0.09 to 0.03); of the 90 % level 10.41 -> 11.35: -0.09 (-0.39 to 0.13); better on 0.33 of the pairs
```

仍偏 9 cm 以上的伸手，和按部位看（`via_parts.py`）：

```
onpol_dev_r2_pose          over 9 cm:  15/164 (0.091) | through median 3.79 90 % 8.74 95 % 10.32 | 24 spheres median 7.72 | model 2.23 (direct 1.24)
onpol_dev_r3_arm           over 9 cm:  22/164 (0.134) | through median 3.27 90 % 10.41 95 % 16.49 | 24 spheres median 7.36 | model 2.09 (direct 1.26)
onpol_dev_r4_pose          over 9 cm:  17/164 (0.104) | through median 3.70 90 % 8.94 95 % 10.25 | 24 spheres median 7.72 | model 1.87 (direct 1.23)
onpol_dev_r4_arm_narrow    over 9 cm:  10/164 (0.061) | through median 3.33 90 % 7.49 95 % 9.55 | 24 spheres median 7.41 | model 2.04 (direct 1.23)
onpol_dev_r4_arm           over 9 cm:  20/164 (0.122) | through median 3.41 90 % 11.35 95 % 17.40 | 24 spheres median 7.44 | model 2.06 (direct 1.23)
dev_via_pose2_20261122/direct: 164 paths
  shoulder  model  0.91  plan  0.00  actual  3.83
  elbow     model  0.66  plan  0.00  actual  2.49
  wrist     model  0.78  plan  0.00  actual  3.12
  palm      model  0.86  plan  0.00  actual  4.79
  fingers   model  1.11  plan  0.00  actual  7.91
  all 24    model  1.23 (90 % 4.42)
reachwm_20261030/onpol_dev_r4_arm: 164 paths
  shoulder  model  1.50  plan  5.07  actual  7.37
  elbow     model  1.23  plan  4.07  actual  3.81
  wrist     model  1.34  plan  4.64  actual  2.61
  palm      model  1.37  plan  7.98  actual  2.69
  fingers   model  1.88  plan 12.59  actual  3.31
  all 24    model  2.06 (90 % 8.56)
reachwm_20261030/onpol_dev_r4_arm_narrow: 164 paths
  shoulder  model  1.39  plan  4.92  actual  7.33
  elbow     model  1.19  plan  4.01  actual  3.70
  wrist     model  1.27  plan  4.51  actual  2.55
  palm      model  1.34  plan  7.73  actual  2.55
  fingers   model  1.84  plan 12.09  actual  3.26
  all 24    model  2.04 (90 % 6.00)
reachwm_20261030/onpol_dev_r4_pose: 164 paths
  shoulder  model  1.47  plan  4.77  actual  7.52
  elbow     model  1.05  plan  3.91  actual  3.85
  wrist     model  1.09  plan  4.18  actual  2.73
  palm      model  1.25  plan  7.44  actual  2.76
  fingers   model  1.60  plan 12.00  actual  3.56
  all 24    model  1.87 (90 % 5.82)
```

宽窄两种上限逐对比较（探索；两遍的 `comp_pairs.npz`）：宽上限比窄上限差 2 cm 以上的有 20 次，好 2 cm 以上的 1 次。这 20 次在窄上限下修正的最大值中位数是 8.0 cm 和 40°，宽上限下是 10.4 cm 和 60°；腕 + 掌 + 指的误差从 7.4 cm 变成 16.5 cm，模型对那一遍的预测误差从 4.7 cm 变成 10.0 cm。其余 144 次两遍的修正、误差、模型误差都相同（5.7 cm、31.2°、3.1 cm、1.8 cm）。

把窄缝计分用在开发路线上（探索，余量取自同一遍；输出没有进仓库，脚本在只有绕行路线的目录上跑到"板下路线"那张表时因为没有数据而退出，前面的表已经打完）：

```
through: actual minus predicted error on onpol_dev_r4_arm_narrow: median +0.14 cm, 95 % level +3.81 cm (the margin of --rule margin)
through: direct command's error on dev_via_pose2_20261122/direct: 95 % level 12.56 cm (fixed_dev)
detour routes, spheres 'through': 41 routes, 164 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 7.91 / 11.83 / 12.56; const 5.56 / 9.87 / 11.59; ours 3.33 / 7.49 / 9.55
  half-width cm            2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small     success    0.00  0.00  0.00  0.04  0.13  0.28  0.52  0.68  0.79  0.86  0.92  0.96  0.98  0.99
            collide    0.00  0.00  1.00  0.96  0.87  0.72  0.48  0.32  0.21  0.14  0.08  0.04  0.02  0.01
  fixed     success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius    success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.04  0.08  0.13  0.20  0.27  0.38
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const     success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  fixed_dev success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.96  0.98  0.99
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.04  0.02  0.01
  ours      success    0.00  0.00  0.00  0.01  0.30  0.57  0.70  0.79  0.84  0.87  0.93  0.96  0.98  0.98
            collide    0.00  0.00  0.00  0.00  0.01  0.02  0.00  0.00  0.00  0.01  0.01  0.01  0.00  0.01
  ours_sum  success    0.00  0.00  0.03  0.19  0.37  0.48  0.57  0.62  0.68  0.72  0.75  0.77  0.79  0.81
            collide    0.00  0.00  0.00  0.00  0.01  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small      12.75  (11.74 to 13.75)
    fixed        nan  (nan to nan)
    radius       nan  (nan to nan)
    const        nan  (nan to nan)
    fixed_dev  12.75  (12.75 to 13.75)
    ours       11.50  (9.25 to 12.50)
    ours_sum     nan  (11.87 to 15.00)
  ours / best other way (small): 0.90 (0.75 to 1.00)
```

### 第四步补充之四：第三批测试路线，只跑了这一次（2026-11-23）

2026-11-23 10:04–11:27 UTC，vastgpu，2 个进程（同时 4 个进程在采 AMO）。测试路线 `outputs/reachwm_20261030/test3_routes`（种子 20261125，450 个场景规划出 172 条：越过桌沿 21、越过箱子 28、伸到板下 123），09:47–09:51 规划，只用到 cuRobo。五组回放 `test3_direct`、`test3_const`、`test3_arm`、`test3_pose`、`test3_point`，各 688 次伸手，0 摔倒，0 跳过，种子 57000000。链条脚本 `test3_chain.sh`、`test3_chain2.sh`，各步的时刻在 `test3_chain.steps`。

先后顺序：判据的提交 `ea13f62a` 在 09:56:36，第一组回放 10:04:04 开始。补充分析的提交 `9a8637f0` 在 10:33:09（钩子跑完 10:39），那时第三组回放在跑，没有读过任何测试路线上的评测；只挪位置那组 11:16:32 开始，放回箱子的检查 11:27 才第一次在测试路线上跑。

`arm` 那一遍和直接发命令成对比较（`test3_arm/comp_eval.out`）：

```
pairs 688 (direct 688, compensated 688 paths without a fall; 0 pairs dropped: the two runs' wanted sweeps differ by over 1 mm)
check, wanted sweep at run time against the direct run's reference: median 0.00 mm, largest 0.00 mm
check, prediction at run time against the offline one: median 0.00 mm, largest 0.00 mm
check, the compensated command's plan beforehand against the one it ran with: median 0.00 mm, largest 0.00 mm
detour paths (over an edge, over a box): 49 routes, 196 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.66 cm, compensated 3.69 cm; drop 0.57 (0.42 to 0.65); 90 % level 12.42 and 12.02 cm
  palm, error against the wanted sweep: direct 5.29 cm, compensated 2.50 cm; drop 0.53 (0.40 to 0.58); improved in 0.84 of the pairs
  arm,  error against the wanted sweep: direct 9.05 cm, compensated 7.32 cm; drop 0.19 (0.08 to 0.28); improved in 0.57 of the pairs
  palm's settled end against the wanted end: direct 3.52 cm, compensated 1.17 cm; drop 0.67 (0.60 to 0.74)
  hand's attitude against the wanted one: settled end direct 22.1 deg, compensated 10.2 deg; along the path 11.8 and 3.8 deg
  points moved: median 2.89 cm, largest 8.00 cm; the goal 3.49 cm; wrist commands turned: median 20.7 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.79 cm, for the one sent 2.21 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.44 cm, compensated 2.28 cm
```

`pose`、常数修正、只挪位置三遍的同一段（各自目录的 `comp_eval.out`，三项一致性检查同样是 0.00 mm）：

```
detour paths (over an edge, over a box): 49 routes, 196 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.66 cm, compensated 3.94 cm; drop 0.54 (0.38 to 0.64); 90 % level 12.42 and 10.50 cm
  palm, error against the wanted sweep: direct 5.29 cm, compensated 2.62 cm; drop 0.51 (0.30 to 0.57); improved in 0.86 of the pairs
  arm,  error against the wanted sweep: direct 9.05 cm, compensated 7.55 cm; drop 0.17 (0.05 to 0.25); improved in 0.60 of the pairs
  palm's settled end against the wanted end: direct 3.52 cm, compensated 1.04 cm; drop 0.70 (0.61 to 0.74)
  hand's attitude against the wanted one: settled end direct 22.1 deg, compensated 10.4 deg; along the path 11.8 and 3.8 deg
  points moved: median 3.07 cm, largest 8.00 cm; the goal 4.54 cm; wrist commands turned: median 21.3 deg, largest 40.0 deg
  palm, the error the model predicted: for the original command 4.79 cm, for the one sent 2.30 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 1.44 cm, compensated 2.03 cm
detour paths (over an edge, over a box): 49 routes, 196 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.66 cm, compensated 9.04 cm; drop -0.04 (-0.13 to -0.02); 90 % level 12.42 and 12.95 cm
  palm, error against the wanted sweep: direct 5.29 cm, compensated 5.99 cm; drop -0.13 (-0.21 to -0.05); improved in 0.15 of the pairs
  arm,  error against the wanted sweep: direct 9.05 cm, compensated 9.66 cm; drop -0.07 (-0.11 to -0.00); improved in 0.14 of the pairs
  palm's settled end against the wanted end: direct 3.52 cm, compensated 4.66 cm; drop -0.32 (-0.46 to -0.28)
  hand's attitude against the wanted one: settled end direct 22.1 deg, compensated 22.3 deg; along the path 11.8 and 12.5 deg
  points moved: median 0.97 cm, largest 2.56 cm; the goal 1.42 cm; wrist commands turned: median 0.0 deg, largest 0.0 deg
  palm, the error the model predicted: for the original command 4.78 cm, for the one sent 5.10 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 3.75 cm, compensated 3.73 cm
detour paths (over an edge, over a box): 49 routes, 196 pairs
  wrist + palm + fingers, error against the wanted sweep: direct 8.66 cm, compensated 6.13 cm; drop 0.29 (0.21 to 0.36); 90 % level 12.42 and 10.87 cm
  palm, error against the wanted sweep: direct 5.29 cm, compensated 3.07 cm; drop 0.42 (0.32 to 0.48); improved in 0.95 of the pairs
  arm,  error against the wanted sweep: direct 9.05 cm, compensated 7.31 cm; drop 0.19 (0.10 to 0.32); improved in 0.78 of the pairs
  palm's settled end against the wanted end: direct 3.52 cm, compensated 0.90 cm; drop 0.74 (0.69 to 0.77)
  hand's attitude against the wanted one: settled end direct 22.1 deg, compensated 21.1 deg; along the path 11.8 and 11.9 deg
  points moved: median 2.45 cm, largest 8.00 cm; the goal 2.84 cm; wrist commands turned: median 0.0 deg, largest 0.0 deg
  palm, the error the model predicted: for the original command 4.78 cm, for the one sent 1.88 cm (what the loop left)
  the model's own error (arm) on the run it predicted: direct 3.75 cm, compensated 3.80 cm
```

`arm` 对 `pose`（`via_vs.py`，基线是 `pose`），按部位看（`via_parts.py --detour`）：

```
test3_arm | 196 pairs, 49 routes
  wrist + palm + fingers baseline  3.94 cm, this run  3.69 cm; drop of the median 0.06 (-0.04 to 0.14); of the 90 % level 10.50 -> 12.02: -0.14 (-0.24 to 0.11); better on 0.54 of the pairs
  palm                   baseline  2.62 cm, this run  2.50 cm; drop of the median 0.05 (-0.03 to 0.19); of the 90 % level  7.09 ->  8.25: -0.16 (-0.24 to 0.11); better on 0.46 of the pairs
  arm (24 spheres)       baseline  7.55 cm, this run  7.32 cm; drop of the median 0.03 (0.01 to 0.08); of the 90 % level 13.77 -> 13.54: 0.02 (-0.12 to 0.13); better on 0.78 of the pairs
  palm's settled end     baseline  1.04 cm, this run  1.17 cm; drop of the median -0.13 (-0.24 to 0.06); of the 90 % level  4.08 ->  3.57: 0.12 (-0.13 to 0.43); better on 0.54 of the pairs
  wrist + palm + fingers still over 9 cm: baseline 29/196 (0.148), this run 28/196 (0.143); difference -0.005 (-0.046 to +0.031); routes with any: 10 and 10
reachwm_20261030/test3_direct: 196 paths
  shoulder  model  1.23  plan  0.00  actual  4.08
  elbow     model  0.76  plan  0.00  actual  2.71
  wrist     model  0.89  plan  0.00  actual  3.43
  palm      model  0.91  plan  0.00  actual  5.29
  fingers   model  1.17  plan  0.00  actual  8.66
  all 24    model  1.44 (90 % 5.52)
reachwm_20261030/test3_arm: 196 paths
  shoulder  model  1.80  plan  5.35  actual  7.08
  elbow     model  1.48  plan  4.38  actual  3.58
  wrist     model  1.43  plan  4.95  actual  2.82
  palm      model  1.52  plan  8.43  actual  2.50
  fingers   model  1.91  plan 13.42  actual  3.62
  all 24    model  2.28 (90 % 8.93)
reachwm_20261030/test3_pose: 196 paths
  shoulder  model  1.49  plan  5.31  actual  7.15
  elbow     model  1.08  plan  4.30  actual  3.74
  wrist     model  1.06  plan  4.82  actual  2.72
  palm      model  1.31  plan  8.34  actual  2.62
  fingers   model  1.75  plan 13.03  actual  3.93
  all 24    model  2.03 (90 % 7.04)
```

窄缝计分（`via_gap_test3_20261123/gap.out`）：

```
through: actual minus predicted error on onpol_dev_r4_arm_narrow: median +0.14 cm, 95 % level +3.81 cm (the margin of --rule margin)
through: direct command's error on dev_via_pose2_20261122/direct: 95 % level 12.56 cm (fixed_dev)
detour routes, spheres 'through': 49 routes, 196 reaches
  error against the wanted sweep, median / 90 % / 95 % cm: direct 8.66 / 12.42 / 13.61; const 9.04 / 12.95 / 13.92; ours 3.69 / 12.02 / 13.48
  half-width cm            2     3     4     5     6     7     8     9    10    11    12    13    14    15
  small     success    0.00  0.00  0.00  0.09  0.18  0.32  0.44  0.54  0.64  0.79  0.87  0.92  0.97  0.97
            collide    0.00  0.00  1.00  0.91  0.82  0.68  0.56  0.46  0.36  0.21  0.13  0.08  0.03  0.03
  fixed     success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  radius    success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.03  0.08  0.16  0.25  0.31  0.38
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  const     success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  fixed_dev success    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.92  0.97  0.97
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.08  0.03  0.03
  ours      success    0.00  0.00  0.00  0.01  0.26  0.51  0.61  0.68  0.76  0.82  0.85  0.89  0.93  0.95
            collide    0.00  0.00  0.00  0.00  0.00  0.01  0.01  0.01  0.01  0.01  0.02  0.02  0.01  0.00
  ours_sum  success    0.00  0.01  0.02  0.20  0.36  0.46  0.52  0.55  0.57  0.60  0.63  0.65  0.69  0.70
            collide    0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:
    small      13.75  (12.50 to 15.00)
    fixed        nan  (nan to nan)
    radius       nan  (nan to nan)
    const        nan  (nan to nan)
    fixed_dev  13.75  (12.75 to 15.00)
    ours       13.25  (10.75 to 14.50)
    ours_sum     nan  (nan to nan)
  ours / best other way (small): 0.96 (0.83 to 1.05)
```

把箱子放回去（`test3_scene_check_20261123/scene.out`，基线是直接发命令）：

```
reachwm_20261030/test3_direct: 196 reaches on 49 detour routes
  median clearance cm / share touching            wanted              plan             model            actual
  shoulder                                13.34 / 0.000    13.34 / 0.000    12.20 / 0.005    12.27 / 0.015
  elbow                                    9.84 / 0.000     9.84 / 0.000     8.48 / 0.015     8.58 / 0.020
  wrist                                    5.92 / 0.000     5.92 / 0.000     5.78 / 0.005     5.75 / 0.026
  palm                                     3.81 / 0.031     3.81 / 0.031     4.21 / 0.005     3.99 / 0.010
  fingers                                  1.52 / 0.056     1.52 / 0.056     0.45 / 0.383     0.18 / 0.459
  all 24                                   1.50 / 0.066     1.50 / 0.066     0.45 / 0.383     0.18 / 0.459
  model's sweep within 0 cm of a box as the warning: 75 warned, 22 of 90 touches not warned, 7 warned without a touch
  model's sweep within 1 cm of a box as the warning: 152 warned, 2 of 90 touches not warned, 64 warned without a touch
  model's sweep within 2 cm of a box as the warning: 183 warned, 0 of 90 touches not warned, 93 warned without a touch
  depth of the touches: median 0.78 cm, largest 4.11 cm; reaches touching with: shoulder 3, elbow 4, wrist 5, palm 2, fingers 90
reachwm_20261030/test3_const: 196 reaches on 49 detour routes
  median clearance cm / share touching            wanted              plan             model            actual
  shoulder                                13.34 / 0.000    13.44 / 0.000    12.22 / 0.005    12.26 / 0.020
  elbow                                    9.84 / 0.000     9.77 / 0.000     8.63 / 0.015     8.66 / 0.020
  wrist                                    5.92 / 0.000     5.33 / 0.000     5.64 / 0.005     5.54 / 0.020
  palm                                     3.81 / 0.031     3.07 / 0.031     3.59 / 0.010     3.30 / 0.026
  fingers                                  1.52 / 0.056     1.32 / 0.056    -0.06 / 0.541    -0.22 / 0.612
  all 24                                   1.50 / 0.066     1.32 / 0.066    -0.06 / 0.541    -0.23 / 0.617
  model's sweep within 0 cm of a box as the warning: 106 warned, 23 of 121 touches not warned, 8 warned without a touch
  model's sweep within 1 cm of a box as the warning: 171 warned, 3 of 121 touches not warned, 53 warned without a touch
  model's sweep within 2 cm of a box as the warning: 187 warned, 0 of 121 touches not warned, 66 warned without a touch
  depth of the touches: median 0.85 cm, largest 4.25 cm; reaches touching with: shoulder 4, elbow 4, wrist 4, palm 5, fingers 120
  against test3_direct, 196 pairs on 49 routes: touching 90 (0.459) there, 121 (0.617) this run; difference +0.158 (+0.071 to +0.245); touching in both 81, only there 9, only this run 40
reachwm_20261030/test3_point: 196 reaches on 49 detour routes
  median clearance cm / share touching            wanted              plan             model            actual
  shoulder                                13.34 / 0.000    14.17 / 0.000    13.72 / 0.005    13.78 / 0.015
  elbow                                    9.84 / 0.000    10.16 / 0.000     9.02 / 0.000     9.03 / 0.010
  wrist                                    5.92 / 0.000     5.85 / 0.020     4.94 / 0.015     4.59 / 0.005
  palm                                     3.81 / 0.031     2.98 / 0.158     4.41 / 0.000     4.20 / 0.000
  fingers                                  1.52 / 0.056     1.10 / 0.311     1.34 / 0.051     1.16 / 0.087
  all 24                                   1.50 / 0.066     0.92 / 0.332     1.29 / 0.066     1.07 / 0.102
  model's sweep within 0 cm of a box as the warning: 13 warned, 11 of 20 touches not warned, 4 warned without a touch
  model's sweep within 1 cm of a box as the warning: 66 warned, 4 of 20 touches not warned, 50 warned without a touch
  model's sweep within 2 cm of a box as the warning: 169 warned, 0 of 20 touches not warned, 149 warned without a touch
  depth of the touches: median 0.91 cm, largest 2.71 cm; reaches touching with: shoulder 3, elbow 2, wrist 1, palm 0, fingers 17
  against test3_direct, 196 pairs on 49 routes: touching 90 (0.459) there, 20 (0.102) this run; difference -0.357 (-0.474 to -0.250); touching in both 18, only there 72, only this run 2
reachwm_20261030/test3_arm: 196 reaches on 49 detour routes
  median clearance cm / share touching            wanted              plan             model            actual
  shoulder                                13.34 / 0.000    14.46 / 0.000    12.19 / 0.071    11.90 / 0.071
  elbow                                    9.84 / 0.000    10.15 / 0.000     8.89 / 0.061     8.95 / 0.056
  wrist                                    5.92 / 0.000     5.63 / 0.031     5.55 / 0.005     5.46 / 0.000
  palm                                     3.81 / 0.031     2.18 / 0.281     4.29 / 0.000     4.06 / 0.015
  fingers                                  1.52 / 0.056    -0.46 / 0.561     1.11 / 0.102     0.88 / 0.224
  all 24                                   1.50 / 0.066    -0.64 / 0.566     1.04 / 0.163     0.79 / 0.260
  model's sweep within 0 cm of a box as the warning: 32 warned, 23 of 51 touches not warned, 4 warned without a touch
  model's sweep within 1 cm of a box as the warning: 94 warned, 10 of 51 touches not warned, 53 warned without a touch
  model's sweep within 2 cm of a box as the warning: 180 warned, 3 of 51 touches not warned, 132 warned without a touch
  depth of the touches: median 0.96 cm, largest 4.69 cm; reaches touching with: shoulder 14, elbow 11, wrist 0, palm 3, fingers 44
  against test3_direct, 196 pairs on 49 routes: touching 90 (0.459) there, 51 (0.260) this run; difference -0.199 (-0.327 to -0.071); touching in both 35, only there 55, only this run 16
reachwm_20261030/test3_pose: 196 reaches on 49 detour routes
  median clearance cm / share touching            wanted              plan             model            actual
  shoulder                                13.34 / 0.000    13.80 / 0.000    11.58 / 0.087    11.54 / 0.082
  elbow                                    9.84 / 0.000     9.99 / 0.000     8.59 / 0.066     8.47 / 0.071
  wrist                                    5.92 / 0.000     5.61 / 0.031     5.83 / 0.010     5.47 / 0.000
  palm                                     3.81 / 0.031     2.16 / 0.260     4.23 / 0.000     4.04 / 0.000
  fingers                                  1.52 / 0.056    -0.29 / 0.556     1.03 / 0.112     0.72 / 0.240
  all 24                                   1.50 / 0.066    -0.45 / 0.561     0.96 / 0.184     0.67 / 0.276
  model's sweep within 0 cm of a box as the warning: 36 warned, 19 of 54 touches not warned, 1 warned without a touch
  model's sweep within 1 cm of a box as the warning: 104 warned, 5 of 54 touches not warned, 55 warned without a touch
  model's sweep within 2 cm of a box as the warning: 183 warned, 3 of 54 touches not warned, 132 warned without a touch
  depth of the touches: median 1.35 cm, largest 4.75 cm; reaches touching with: shoulder 16, elbow 14, wrist 0, palm 0, fingers 47
  against test3_direct, 196 pairs on 49 routes: touching 90 (0.459) there, 54 (0.276) this run; difference -0.184 (-0.301 to -0.066); touching in both 44, only there 46, only this run 10
```

`arm`、`pose` 对只挪位置（`scene_vs_point.out` 和 `vs_point.out`）：

```
  against test3_point, 196 pairs on 49 routes: touching 20 (0.102) there, 51 (0.260) this run; difference +0.158 (+0.077 to +0.240); touching in both 16, only there 4, only this run 35
  against test3_point, 196 pairs on 49 routes: touching 20 (0.102) there, 54 (0.276) this run; difference +0.173 (+0.087 to +0.265); touching in both 16, only there 4, only this run 38
test3_arm | 196 pairs, 49 routes
  wrist + palm + fingers baseline  6.13 cm, this run  3.69 cm; drop of the median 0.40 (0.20 to 0.49); of the 90 % level 10.87 -> 12.02: -0.11 (-0.20 to 0.22); better on 0.77 of the pairs
  palm                   baseline  3.07 cm, this run  2.50 cm; drop of the median 0.19 (0.01 to 0.28); of the 90 % level  6.54 ->  8.25: -0.26 (-0.39 to 0.08); better on 0.53 of the pairs
  arm (24 spheres)       baseline  7.31 cm, this run  7.32 cm; drop of the median -0.00 (-0.22 to 0.11); of the 90 % level 11.89 -> 13.54: -0.14 (-0.24 to -0.04); better on 0.35 of the pairs
  palm's settled end     baseline  0.90 cm, this run  1.17 cm; drop of the median -0.30 (-0.52 to 0.01); of the 90 % level  2.42 ->  3.57: -0.48 (-1.24 to -0.03); better on 0.42 of the pairs
  wrist + palm + fingers still over 9 cm: baseline 36/196 (0.184), this run 28/196 (0.143); difference -0.041 (-0.102 to +0.010); routes with any: 12 and 10
test3_pose | 196 pairs, 49 routes
  wrist + palm + fingers baseline  6.13 cm, this run  3.94 cm; drop of the median 0.36 (0.13 to 0.49); of the 90 % level 10.87 -> 10.50: 0.03 (-0.10 to 0.18); better on 0.77 of the pairs
  palm                   baseline  3.07 cm, this run  2.62 cm; drop of the median 0.15 (-0.11 to 0.26); of the 90 % level  6.54 ->  7.09: -0.08 (-0.31 to 0.08); better on 0.53 of the pairs
  arm (24 spheres)       baseline  7.31 cm, this run  7.55 cm; drop of the median -0.03 (-0.27 to 0.06); of the 90 % level 11.89 -> 13.77: -0.16 (-0.36 to 0.01); better on 0.33 of the pairs
  palm's settled end     baseline  0.90 cm, this run  1.04 cm; drop of the median -0.15 (-0.47 to 0.04); of the 90 % level  2.42 ->  4.08: -0.69 (-1.22 to -0.39); better on 0.39 of the pairs
  wrist + palm + fingers still over 9 cm: baseline 36/196 (0.184), this run 29/196 (0.148); difference -0.036 (-0.107 to +0.026); routes with any: 12 and 10
```

**事先登记的判据逐条（提交 `ea13f62a`）：**

| 判据 | 数字 | 结论 |
|---|---|---|
| `arm` 的腕 + 掌 + 指中位数低于 `pose`，区间不含 0 | 3.69 对 3.94 cm，低 6 %（−4 % 到 14 %） | 区间含 0：测试路线上分不出来。开发路线上的 10 % 没有复现 |
| `arm` 相对直接发命令的降幅 ≥ 0.5，区间下端 ≥ 0.4 | 8.66 → 3.69 cm，0.57（0.42–0.65） | 成立 |
| `arm` 仍偏 9 cm 以上的比例低于 `pose`，区间不含 0 | 28 对 29 / 196，差 −0.005（−0.046 到 +0.031） | 没有减少 |
| 窄缝比值 ≤ 0.8 且区间上端 < 1 | 我们 13.25 cm，小固定余量和开发路线上标定的固定余量都是 13.75 cm，比值 0.96（0.83–1.05） | 不成立，第三次 |
| 有效性三条 | 一致性检查 0.00 mm；摔倒 0 对 0；模型误差 2.28 对 1.44 cm（1.58 倍） | 都满足 |

预测落在区间外的：仍偏 9 cm 以上的比例（预测 `arm` 5–11 %、`pose` 8–14 %，实际 14.3 % 和 14.8 %）；终点朝向（预测 7–10°，实际 10.2°；直接发命令预测 19–22°，实际 22.1°）；直接发命令的 24 个球（预测 8.0–8.8 cm，实际 9.05 cm）。其余都在区间内，包括窄缝的四个数和"判据仍然不成立"这句话。

**补充分析的判据逐条（提交 `9a8637f0`）：**

| | 碰到箱子的伸手（196 次） | 和直接发命令之差（区间） | 肩 / 肘碰到的伸手 | 结论 |
|---|---|---|---|---|
| 想要的路线本身 | 6.6 % | | 0 / 0 | |
| 直接发命令 | 45.9 % | | 3 / 4 | |
| 常数修正 | 61.7 % | +0.158（+0.071 到 +0.245） | 4 / 4 | 反而更多 |
| 只挪位置（`via_point_20261121`） | 10.2 % | −0.357（−0.474 到 −0.250） | 3 / 2 | 减少了 |
| `arm`（第 4 轮模型） | 26.0 % | −0.199（−0.327 到 −0.071） | 14 / 11 | 减少了 |
| `pose`（第 4 轮模型） | 27.6 % | −0.184（−0.301 到 −0.066） | 16 / 14 | 减少了 |

- 只挪位置对 `arm`：+0.158（+0.077 到 +0.240），区间不含 0。**只挪位置在真障碍物上更好。按登记的规则，下一步（有真障碍物的场景）用只挪位置。**
- 预测落在区间外的：直接发命令（预测 28–42 %，实际 45.9 %）；只挪位置（预测 11–22 %，实际 10.2 %）；常数修正（预测 20–35 %，实际 61.7 %）；肩和肘（直接发命令和只挪位置预测 0–1 %，实际 1–2 %；`arm` 和 `pose` 预测 1–5 %，实际 6–8 %）；"九成以上是手指碰的"只对直接发命令成立（只挪位置 17 / 20，`arm` 44 / 51）。
- 两种度量排出来的次序相反。离想要的路线：`arm` 3.69 cm，只挪位置 6.13 cm。碰不碰到箱子：只挪位置 10 %，`arm` 26 %。加朝向把指尖转回规划的位置，也就是离箱子只有 1.5 cm 的地方，又把上臂往前带；只挪位置把手掌的路线和终点修回去（终点 0.90 cm，`arm` 1.17 cm），上臂反而往后（肩的间隙 13.8 cm，直接发命令 12.3 cm）。

**同一项检查用在前两批测试路线和 SONIC 上（探索，没有事先登记；这两批路线以前用别的度量看过。输出 `outputs/reachwm_20261030/old_tests_scene_check_20261123`，第二批为此补跑了一组只挪位置 `test2_point`）：**

```
reachwm_20261030/data_via_fresh: 84 reaches on 21 detour routes
  all 24                                   1.50 / 0.119     1.50 / 0.119     0.43 / 0.274     0.19 / 0.417
  depth of the touches: median 0.83 cm, largest 3.59 cm; reaches touching with: shoulder 3, elbow 2, wrist 2, palm 1, fingers 35
reachwm_20261030/data_via_const: 84 reaches on 21 detour routes
  all 24                                   1.50 / 0.119     1.25 / 0.119    -0.03 / 0.524    -0.41 / 0.571
  depth of the touches: median 0.75 cm, largest 3.22 cm; reaches touching with: shoulder 4, elbow 3, wrist 0, palm 0, fingers 47
  against data_via_fresh, 84 pairs on 21 routes: touching 35 (0.417) there, 48 (0.571) this run; difference +0.155 (+0.024 to +0.286); touching in both 32, only there 3, only this run 16
reachwm_20261030/data_via_comp: 84 reaches on 21 detour routes
  all 24                                   1.50 / 0.119     1.11 / 0.226     1.23 / 0.107     1.21 / 0.155
  depth of the touches: median 0.74 cm, largest 2.46 cm; reaches touching with: shoulder 3, elbow 3, wrist 0, palm 0, fingers 10
  against data_via_fresh, 84 pairs on 21 routes: touching 35 (0.417) there, 13 (0.155) this run; difference -0.262 (-0.393 to -0.143); touching in both 13, only there 22, only this run 0
reachwm_20261030/data_via_pose: 84 reaches on 21 detour routes
  all 24                                   1.50 / 0.119     0.23 / 0.476     1.25 / 0.167     0.89 / 0.190
  depth of the touches: median 0.80 cm, largest 4.88 cm; reaches touching with: shoulder 5, elbow 8, wrist 1, palm 2, fingers 12
  against data_via_fresh, 84 pairs on 21 routes: touching 35 (0.417) there, 16 (0.190) this run; difference -0.226 (-0.381 to -0.083); touching in both 15, only there 20, only this run 1
  against data_via_comp, 84 pairs on 21 routes: touching 13 (0.155) there, 16 (0.190) this run; difference +0.036 (-0.060 to +0.155); touching in both 9, only there 4, only this run 7
reachwm_20261030/test2_direct: 180 reaches on 45 detour routes
  all 24                                   1.57 / 0.083     1.57 / 0.083     0.43 / 0.361     0.09 / 0.456
  depth of the touches: median 0.88 cm, largest 2.99 cm; reaches touching with: shoulder 4, elbow 1, wrist 0, palm 1, fingers 80
reachwm_20261030/test2_const: 180 reaches on 45 detour routes
  all 24                                   1.57 / 0.083     1.37 / 0.083     0.05 / 0.478    -0.18 / 0.594
  depth of the touches: median 0.98 cm, largest 2.94 cm; reaches touching with: shoulder 4, elbow 1, wrist 0, palm 3, fingers 106
  against test2_direct, 180 pairs on 45 routes: touching 82 (0.456) there, 107 (0.594) this run; difference +0.139 (+0.072 to +0.211); touching in both 80, only there 2, only this run 27
reachwm_20261030/test2_point: 180 reaches on 45 detour routes
  all 24                                   1.57 / 0.083     1.42 / 0.200     1.30 / 0.100     1.16 / 0.172
  depth of the touches: median 0.28 cm, largest 2.71 cm; reaches touching with: shoulder 2, elbow 1, wrist 4, palm 0, fingers 25
  against test2_direct, 180 pairs on 45 routes: touching 82 (0.456) there, 31 (0.172) this run; difference -0.283 (-0.389 to -0.167); touching in both 27, only there 55, only this run 4
reachwm_20261030/test2_pose: 180 reaches on 45 detour routes
  all 24                                   1.57 / 0.083     0.02 / 0.494     1.05 / 0.183     0.81 / 0.228
  depth of the touches: median 1.30 cm, largest 4.86 cm; reaches touching with: shoulder 12, elbow 9, wrist 1, palm 3, fingers 33
  against test2_direct, 180 pairs on 45 routes: touching 82 (0.456) there, 41 (0.228) this run; difference -0.228 (-0.361 to -0.089); touching in both 29, only there 53, only this run 12
  against test2_point, 180 pairs on 45 routes: touching 31 (0.172) there, 41 (0.228) this run; difference +0.056 (-0.056 to +0.161); touching in both 17, only there 14, only this run 24
reachwm_20261030/data_via_fresh_sonic: 84 reaches on 21 detour routes
  all 24                                   1.95 / 0.024     1.95 / 0.024     2.29 / 0.060     2.00 / 0.155
  depth of the touches: median 0.74 cm, largest 1.89 cm; reaches touching with: shoulder 1, elbow 3, wrist 1, palm 2, fingers 11
reachwm_20261030/data_via_comp_sonic: 84 reaches on 21 detour routes
  all 24                                   1.95 / 0.024    -0.44 / 0.690     1.61 / 0.024     1.16 / 0.131
  depth of the touches: median 0.73 cm, largest 2.96 cm; reaches touching with: shoulder 0, elbow 2, wrist 2, palm 2, fingers 9
  against data_via_fresh_sonic, 84 pairs on 21 routes: touching 13 (0.155) there, 11 (0.131) this run; difference -0.024 (-0.095 to +0.048); touching in both 8, only there 5, only this run 3
```

| | 想要的路线本身 | 直接发命令 | 常数修正 | 只挪位置 | 位置加朝向 |
|---|---|---|---|---|---|
| 开发路线（164 次） | 9.1 % | 34.8 % | | 15.2 % | `arm` 20.1 %，`pose` 26.2 % |
| 第一批测试（84 次） | 11.9 % | 41.7 % | 57.1 % | 15.5 % | `pose`（第 2 轮模型）19.0 % |
| 第二批测试（180 次） | 8.3 % | 45.6 % | 59.4 % | 17.2 % | `pose`（第 2 轮模型）22.8 % |
| 第三批测试（196 次） | 6.6 % | 45.9 % | 61.7 % | 10.2 % | `arm` 26.0 %，`pose` 27.6 % |
| SONIC，第一批测试（84 次） | 2.4 % | 15.5 % | | 13.1 % | |

- ScaleBFM 上，只挪位置在四批路线上都把碰到的比例从 35–46 % 降到 10–17 %，接近想要的路线本身的 7–12 %。位置加朝向在前两批上和只挪位置分不出来（+0.036 和 +0.056，区间含 0），在开发路线和第三批上更差。
- SONIC 上直接发命令本来就很少碰到（15.5 %），只挪位置没有再减少（−0.024，−0.095 到 +0.048）。SONIC 的偏差不朝着障碍物：直接发命令时间隙的中位数是 2.00 cm，和规划的 1.95 cm 一样；补偿把手拉回规划的路线，间隙变成 1.16 cm，终点误差从 4.54 降到 1.58 cm。补偿在真障碍物上有没有好处，取决于控制器的偏差朝哪边。

**规划的间隙被执行吃掉多少（探索，事后想到的度量）。** 每次伸手取"想要的路线的最小间隙 − 实际的最小间隙"：

| | 直接发命令：中位数 / 95 % 分位 | 只挪位置：中位数 / 95 % 分位 |
|---|---|---|
| 开发路线 | 0.89 / 4.13 cm | 0.11 / 1.85 cm |
| 第一批测试 | 0.94 / 3.35 cm | 0.14 / 2.01 cm |
| 第三批测试 | 1.16 / 3.78 cm | 0.30 / 2.01 cm |

规划器要让 95 % 的伸手不碰，直接发命令得多留约 4 cm，只挪位置约 2 cm。这是单侧的、对着真障碍物的量；窄缝计分是四面都算的管子，把无害的偏差也算了进去，所以三次都没过。这个度量是看到结果之后想到的，要作为结论得在新路线上事先登记再测。

**补不动的那部分是高度问题（探索）。** 按想要的路线上手腕小球的最高点分组，看补偿后仍偏 9 cm 以上的比例和补偿后的中位误差（训练路线用 `onpol_r2_pose`，开发 `onpol_dev_r4_arm_narrow`，第二批 `test2_pose`，第三批 `test3_arm`）：

| 手腕最高点 | 训练路线（760 次） | 开发（164 次） | 第二批测试（180 次） | 第三批测试（196 次） |
|---|---|---|---|---|
| 低于 1.10 m | 7 %，3.3 cm | 1 %，3.0 cm | 5 %，3.3 cm | 7 %，2.8 cm |
| 1.10–1.15 m | 21 %，5.8 cm | 25 %，6.3 cm | 22 %，6.2 cm | 25 %，6.5 cm |
| 1.15–1.20 m | 63 %，10.9 cm | 没有这样的路线 | 100 %（4 次），14.7 cm | 62 %（8 次），12.0 cm |

手腕要到 1.10 m 以上的路线，直接发命令就偏 10–13 cm，补偿也拉不回来。这是控制器够不到的高度。各批路线"补不动"的比例从 6 % 到 14 % 不等，是因为高路线占的比例不同（开发路线 22 %，第三批 31 %）。放宽修正的上限没有用，原因也在这里。

## 意外

- 手的方向稳定度 0.16，远低于预测的 0.6–0.9，而肘是 0.85。原因见"预测修订"：统计量取的时刻在两处偏差之间跳。
- 规划的 IK 误差超过 2 cm 的占 11.3 %，预测是低于 5 %。站着 1 %，蹲下 20 %：误差来自蹲低的伸手。
- 覆盖检查只通过 14 / 35，没通过的全是 TabletopGrasp。真实目标在身前 27–38 cm、高 0.91–1.00 m；同样高度的随机目标在 32–57 cm，只有 18 % 在 38 cm 以内。生成器沿用了抓取房间的目标分布，没有 SIMPLE 桌面抓取这种贴着桌子站的近身目标。
- 回放时手的偏差有相当一部分对所有路线都一样：终点处手停在命令之前（后几遍平均往后 3.4 cm，独立复核的数）。只知道"别的路线平均怎么偏"，就能把最大偏差从 3.63 cm 降到 2.39 cm；知道这条路线自己其余几遍，降到 1.56 cm。
- 第 1 遍（原地站立出发）和后 3 遍（走一小段后出发）的偏差不同，复核给的终点偏差是往后 1.6 cm 对 3.4 cm。第 1 遍只有一种起始状态。
- 正式评测（2026-11-21）：方向模型在随机路线上把扫过区域误差降了 38 %，在 35 条真实路线回放上降幅是 0，余量还略大于标量半径（1.06）。
- 方向模型的 5 个成员都在第一个检查点（第 500 步）就被选为最好。原因：`fit()` 把 157 个主成分系数各自缩放成同样的权重，而前 10 个主成分占偏差方差的 69 %，只占训练目标的 6 %。这是训练设置的缺陷，从训练日志就能看出。
- 复核查出真实路线上没有好处的三个原因：第 1 遍从站立直接出发，这种起始状态训练数据里没有（12000 次全部先走一段）；真实路线又长路点又多，随机路线里好处也随路线变长从 41 % 掉到 13 %；模型的好处集中在路线开头，后半段反而略差。

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

### 正式评测之后（2026-11-21）

| 项目 | 预测或判据 | 实测 | 结论 |
|---|---|---|---|
| 覆盖检查（95 % 规则） | 全部盖住；离线估 32 / 35 | 33 / 35 | 两条没盖住，照实列出 |
| 模型相对基线降幅，普通留出 | < 20 % 停；≥ 40 % 且扇区不超过 1.5 倍则进窄缝测试 | 38 %（手 34 %）；扇区 0.74 倍 | 两条线之间 |
| 手的扫过区域误差，普通留出 / 扇区 / 回放 | 1.5–3 / 2–4 / 1.5–3.5 cm | 4.99 / 3.65 / 5.24 cm | 只有扇区猜中 |
| R_方向 / R_标量，普通留出和回放都要 ≤ 0.6 | 预测普通留出 0.5–0.8 | 0.70 和 1.06 | 回放超过 0.8：按字面不再主张，报告用户 |
| R_方向 / R_常数，两者都要 ≤ 0.8 | 预测 0.5–0.8 和 0.7–0.9 | 0.76 和 0.89 | 普通留出成立，回放在中间档 |

- **随机路线上的好处是真的。**复核用自己的代码重算一致，没有数据泄漏；把训练缺陷修掉后普通留出的结果不变（6.70 对 6.71 cm）。
- **真实避障路线上，现在的证据不支持"有好处"。**原样结果是 0；复核训的几个变体在回放上落在 4.9–6.6 cm 之间（参考是 6.46），按路线自助抽样的差值区间是 −0.49 到 +0.70 cm，跨过 0。
- **偏差本身是可重复的（试跑），但"把整条路线一次交给网络"学到的东西不随路线变长而迁移。**好处主要来自路线开头：下命令那一刻身体的实际姿态和参考就差约 2 cm，这部分从状态里直接读得出来。
- 按判据，这里向用户报告，由用户决定下一步。已报告（2026-11-21）。

### 第二轮之后（2026-11-21）

| 项目 | 预测或判据 | 实测 | 结论 |
|---|---|---|---|
| 停止线：验证集，站立，手走过 1 米以上 | 降幅 ≥ 25 %；预测 25–40 % | 66 %（103 条，区间 62–69 %） | 过线 |
| 普通留出的降幅 | 预测 40–50 % | 68 % | 比预测高 |
| 新的真实路线的降幅 | ≥ 20 % 且区间不含 0；预测 20–35 % | 80 %（区间 77–82 %）；其中 21 条长绕行路线 61 %（48–70 %） | 满足 |
| 新的真实路线上固定半径之比 | ≤ 0.8 | 0.63 | 满足 |
| 新的真实路线不少于 20 条 | — | 63 条 | 满足 |

- **第二轮的假设成立。**沿路线逐点预测的模型，在长路线和全新规划的真实路线上都比"照参考走"好得多。按事先的判据，这条线进入窄缝测试。
- **对外该说的数是 61 %，不是 80 %。**80 % 主要来自 42 条近乎直线的短路线；和避障相关的是越过桌沿、越过盒子的 21 条，降幅 61 %，区间 48–70 %，其中 2 条比照参考走还差。
- **好处不是来自起点那个不用预测的偏差。**复核试了"把下命令那一刻的偏差一直保持"，没有任何好处；只看路线后半段，降幅反而更高。把局部参考运动的输入打乱，降幅从 80 % 掉到 27 %；把状态打乱，掉到 66 %。模型主要在用路线的局部几何。
- **这说明偏差是什么。**它主要是一种由参考运动的局部形状决定的系统性跟踪偏差，规划器手里的信息就足以预测它。这是推断，复核也这样看。
- **没验证的**：把预测用回去，也就是按预测反向挪路点之后，手臂是不是真的走到想要的地方。偏差会随命令变，可能要迭代。这是窄缝测试之前该先做的一步。另外没验证换随机种子后新路线上的结果变不变，以及不给状态输入的模型。
- 第一轮被看过多次的那 35 条路线上，降幅从 0 变成 65 %。它们只作参考：改模型结构的决定用到了对它们的分析。

### 第三步之后（2026-11-22）

| 项目 | 预测或判据 | 实测（21 条长绕行路线，84 对） | 结论 |
|---|---|---|---|
| 手掌到想要路线的误差 | 降 30–55 %；判据 ≥ 20 % 且区间不含 0 | 4.83 → 2.99 cm，降 38 %（30–48 %） | 满足 |
| 整条手臂 | 降 10–35 %；判据区间不含 0 | 8.10 → 6.08 cm，降 25 %（16–33 %） | 字面满足，见下 |
| 模型对补偿后命令的预测残差 | 不超过直接发时的 2 倍 | 3.20 对 3.14 cm | 满足 |
| 摔倒或规划失败 | 不比直接组多 5 次以上 | 两组都是 0 | 满足 |

- **按预测挪路点，手确实走到了更接近想要的地方。**两组的起始状态逐位相同，只差命令；动作的时长、路程和录像长度都一样；好处不只在终点，规划时长的前 80 % 里手掌也降了 39 %。
- **"整条手臂降 25 %"全部来自手掌和手指那几个球。**肩、上臂、肘、腕的球没有任何改善（0 到 −12 %，区间跨 0）。能挪的只有手的路点，肘和肩由控制器的全身逆解决定，挪手并不会把它们带到想要的位置。所以判据字面满足，但不能说"整条手臂被修正了"。
- **这决定了窄缝测试该怎么用这个模型**：手的路线可以预测并修正；肘和肩只能预测。对肘和肩，模型的用处是下命令前检查预测的扫过区域会不会碰到障碍，碰到就换一条路线或者放弃，而不是去挪它们。
- 有 2 条路线补偿后变差（`FreshBox_L0_e115`、`FreshEdge_L0_e66`），最差的一对手臂从 11.2 cm 变成 14.2 cm，原因没查。8 cm 的上限只在 84 对里碰到 1 次。
- 没做的：只挪终点、不挪路点的对照，所以途中的改善里有多少是终点带动的，分不开。

### 第四步第一部分之后（2026-11-22）

| 项目 | 预测或判据 | 实测（21 条长绕行路线，手指、手掌、腕） | 结论 |
|---|---|---|---|
| 我们能安全通过的最窄的缝 | 预测 5–8 cm | 11.25 cm（9.5–15.0） | 没猜中 |
| 小固定余量 | 预测 8–11 cm | 12.5 cm（11.0–14.5） | 没猜中 |
| 标定过的固定余量、标量半径、常数修正 | 预测 8–15 cm | 15 cm 以内都达不到 | 没猜中 |
| 我们的 / 最好的对手 | ≤ 0.8 成立；> 0.95 没有优势 | 0.90（0.79–1.18）；只用站立路线校准时 0.96 | 中间档：报告用户决定 |
| c < 5 cm 时的碰撞率 | 小固定余量 > 50 %，我们的 ≤ 10 % | 90 % 对 0（我们全部放弃） | 符合 |

- **"能穿过明显更窄的缝"这个说法，现在的证据不支持。**比值 0.90，区间跨过 1；换一种同样合理的校准方式就是 0.96。成功率上我们和小固定余量几乎一样，差值的区间含 0。
- **证据支持的是另一件事：几乎不碰撞。**缝的半宽 8–11 cm 时，小固定余量有 14–51 % 的伸手会碰到，我们的不超过 1 %，成功率相当。复核试了"原命令加我们的放行规则"，仍有 5–15 % 碰撞，所以安全来自挪过的命令和放行规则两者合起来，不是只靠放弃。
- **谨慎的几种做法几乎总是放弃。**固定余量标出来是 19 cm（只用站立路线是 15 cm），标量半径在 85 % 以上的伸手里放弃。给它们开卷选最好的参数，也只能和小固定余量打平。
- **预测错在哪**：我按手掌 3 个球的误差估的，而要过缝的还有手指。手指的误差大是因为手的朝向：朝向误差中位数 16.5°，挪路点之后还是 16.3°，因为补偿只挪了位置。手掌到指尖 10.3 cm，转 16° 指尖就偏 2.9 cm，正好是补偿后指尖比手掌多出来的那一截。
- **由此得出下一步**：把手腕朝向也按预测反向转回去。训练数据里手腕命令本来就带随机的大角度转动，模型见过；以前也量到控制器只跟上手腕转动的一半左右，是系统性的。
- 整条手臂 24 个球的版本里我们比小固定余量差：肩和肘修正不了，放行规则又会因为它们而放弃。

### 手腕朝向补偿和换控制器之后（2026-11-23）

- **成立的**：
  - 把手腕朝向也补偿，腕、掌、指离想要的路线从 8.06 cm 降到 3.40 cm（21 条测试绕行路线，降 58 %，区间 52–67 %），90 % 分位 11.51 → 6.80 cm，终点朝向 19.3° → 7.1°，0 摔倒。只挪位置时是 5.73 cm。
  - 前提是模型见过补偿器自己发出的命令。只见过随机路线的模型不行；通用的随机补采也不行；把补偿器在另一批规划器路线上的回放加进训练，两轮就够。
  - 同一套做法在 SONIC 上成立：模型在测试绕行路线上降 52 %，补偿后手掌降 52 %，整条手臂降 40 %。
- **不成立的**：
  - "能穿过明显更窄的缝"按事先写的计分仍然不成立（比值 1.02，区间 0.88–1.23）。
  - 整条手臂 24 个球没有因为补偿朝向而改善（降 0.10，区间含 0），95 % 分位变差（15.15 对 12.64 cm）。补偿朝向时躯干多倾，肩偏出去。
- **从这一步得到的认识**：
  - 补偿是在模型上做优化，会把命令推到模型没见过的地方。模型要在"补偿器会发的命令"上训练，这是方法的一部分，不是数据量的问题。
  - 窄缝计分输在出发规则上，不在误差上：我们的碰撞率处处为 0，说明余量留得太多。把"预测的误差"和"残差的上界"直接相加，是两个上界叠在一起。
  - 小固定余量这个对手不需要事先判断，它的结果是事后从测试集读出来的。和它比，等于要求我们的事先判断不比事后的答案差。
- **没做的和下一步**：
  - 等用户定：出发规则改成"预测的误差 + 开发路线上定的固定余量"，在新规划的测试路线上只跑一次（TODO 38.6.0.1.11）。
  - 整条手臂的问题：补偿目标里可以把肩和肘也算进去，或者限制躯干跟着倾。没有做。
  - 第 2 轮没有被证明比第 1 轮好（改善 3 %，区间 −4 % 到 11 %）；停在第几轮和余量的缩放都只靠 41 条开发路线。
  - AMO 在这个场景上伸手时走开，原因没找到（TODO 38.8.2.2）。

## 独立复核

### 试跑的复核

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

### 正式评测的复核（2026-11-21）

新开的 agent，给了原样脚本、模型、日志、数据路径、事先的定义和判据，以及评测的数字；没有给我对训练缺陷的看法。它的代码和输出在 `/workspace/tmp/review_via_model_20261121/`。原文：

Scripts and outputs: /workspace/tmp/review_via_model_20261121/ (common.py, v2..v8.py, v4.log, errs.npz).

**1. Verification (computed with my own code: independent per-sphere polyline distance and independent forward pass of via_model.pt)**

| Set | Metric (cm) | Reference | Direction (as-run) | vs eval.log |
|---|---|---|---|---|
| Hold-out, 1117 paths | arm median | 10.78 | 6.71 | match |
| Hold-out | hand median | 7.58 | 4.99 | match |
| Replay, 140 paths | arm median | 6.46 | 6.44 | match |
| Replay | hand median | 4.77 | 5.24 | match |

**2. Pipeline audit (computed)**
- Splits: bucket hash reproduces. Reset overlap train/val/test/sector is 0. No sector path is in train or val. Replay shares no reset with the training table and is not used in training or calibration.
- Inputs are all available before execution (state at and before the command, command, plan duration, reference spheres).
- All three predictors go through the same error code.
- Calibration: the radius scale is set on the same validation split used for radius early stopping (mildly optimistic). Test coverage is 0.92-0.93 against a 0.95 target. The radius model transfers badly to replay: correlation with log error 0.85 on test, 0.27 on replay. The replay margin ratio of 1.06 compares radii at unequal coverage (reference 0.99, direction 0.92). Rescaled to 95% on replay itself (diagnostic only) the ratio is 1.61. The "margin" is a median radius, not a 95% level; the fixed-radius ratio is 14.28/18.93 = 0.75.

**3. Training defect (computed)**
- Every member picked its best state at iteration 500, the first checkpoint, with 20000 iterations scheduled. Validation loss was 0.95-0.97, where 1.0 is predicting the mean.
- fit() divides each PCA coefficient by its own standard deviation, so the loss weights 157 components equally and most of them are noise.
- With checkpoints every 100 iterations, validation loss bottoms near iteration 200 and ends at 2.24 after 20000. Offset MSE on validation goes 4.8 to 10.9 cm², against 10.3 for the mean. So the model overfits fast and the 500-iteration grid is too coarse.
- Correct procedure: one global target scale, so the loss equals offset-space MSE; fewer components or stronger regularisation; checkpoints every 50 iterations; selection on validation only. Also train the radius models on out-of-bag errors and calibrate coverage on a split separate from early stopping.

Post-hoc variants, one member each, selected on validation offset MSE (bucket 8) only. Test, sector and replay numbers are post-hoc and for information.

| Variant | Val offset MSE | Test arm | Sector arm | Replay arm | Replay hand |
|---|---|---|---|---|---|
| As-run style, best at it 200 | 4.81 | 7.21 | 5.55 | 6.23 | 4.87 |
| 157 PCs, global scale | 4.18 | 6.81 | 5.20 | 6.55 | 4.32 |
| 40 PCs, global scale | 4.17 | 6.83 | 5.15 | 5.85 | 4.02 |
| 40 PCs, dropout 0.3, weight decay 1e-2 | 3.66 | 6.70 | 4.99 | 4.93 | 3.17 |
| Smaller (width 128 or 20 PCs) | 4.7 | 7.7 | 5.9 | 5.3-5.8 | 3.1-3.2 |

- Test is unchanged by the fixes (6.70 against the as-run 6.71).
- Replay moves between 4.9 and 6.6 cm across variants that look similar on validation. The "drop 0.00" is therefore not stable; it is variant spread on 35 routes, not a measured effect.
- The 40 PC dropout variant is not an established fix: its replay gain was seen among several variants.
- To decide without test or replay: select on validation offset MSE and check the pick is stable across seeds and regularisation settings.

**4. Why no gain on replay**

Computed:
- Pass 1 (stand, no walk) is far out of distribution in state: 100% of pass-1 rows have a state dimension more than 10 training SD off (dims 9, 19, 74, ...; 20+ SD for the worst, which look like joint angles), against 0% for passes 2-4. Pass 1 arm error is reference 7.11 against direction 8.31 (worse). Passes 2-4 are 6.33 against 5.93 (6% better).
- Replay paths are long and have many vias. Median hand path is 1.0 m against 0.56 m on the standing hold-out. Replay has 7-8 vias, while training is dominated by 0-5. On the standing hold-out the drop falls from 0.41 on short paths to 0.13 on paths over 1 m, and to 0.25-0.29 for 7-8 vias.
- The reference is already at the model's residual level: replay reference 6.46 against model hold-out 6.71. Offset R² is 0.01 on passes 2-4, -0.48 on pass 1, 0.56 on standing hold-out.
- The gain is early only: palm start offset 2.0 cm for the reference, 0.4 for the model. In the second half the model is worse (4.5 against 4.0 cm).
- The paired difference is noisy: the model wins on 61% of paths, mean paired difference about -0.15 cm. A route-bootstrap 95% interval for median(direction) minus median(reference) is [-0.49, +0.70] cm.

Ruled out: leakage or a table bug.

Cannot tell: how much is path length, via count or planner-path controller behaviour. The hold-out has few long multi-via standing paths. Per-sphere breakdown on replay shows the largest errors on the finger and shoulder_yaw spheres for both predictors.

**5. Pre-registered criteria, literal**

| Criterion | Result |
|---|---|
| Replay reference median < 1.5 cm, stop | Not triggered (6.46 arm, 3.79 hand per analysis.txt). |
| Drop < 20% on hold-out, stop | Not triggered (0.38 arm, 0.34 hand). |
| R_dir/R_scalar <= 0.6 on hold-out and replay | Not met (0.70 and 1.06). |
| R_dir/R_scalar > 0.8 | Replay 1.06 is above 0.8, hold-out 0.70 is in the middle band. The criterion is joint, so the result is "report to user", not a pass. |
| R_dir/R_constant <= 0.8 on both | Hold-out 0.76 passes, replay 0.89 fails. Not > 0.9, so middle band (report to user). |
| Drop >= 40% and sector <= 1.5x hold-out | Drop is 0.38 arm / 0.34 hand, strictly not met. Sector ratio 4.95/6.71 passes. |
| Sector > 2x hold-out | Not triggered. |
| Hand model 1.5-3 cm on hold-out | Not met (4.99). |
| Hand model replay 1.5-3.5 cm | Not met (5.24). |

Summary: the hold-out gain is real and robust to retraining. The replay claim is undecided: the as-run replay number depends on a training defect that I showed moves it by up to about 1.6 cm, and a route-level bootstrap interval spans zero. "Predicts real planner paths" is not supported.

### 第二轮最终评测的复核（2026-11-21）

新开的 agent，给了模型、原样脚本、日志、数据路径、第二轮事先写的假设和判据、以及评测的数字。它的代码和输出在 `/workspace/tmp/review_via_point_20261121/`。原文：

**Verdict.** The claimed numbers reproduce exactly. I found no leakage and no flaw that inflates the result. The honest headline for long detour routes is a drop of about 0.61 (CI 0.48-0.70), not 0.80.

All scripts and pickles are in /workspace/tmp/review_via_point_20261121/ (common.py, r1-r4.py, dense.py). Everything marked "verified" was computed. I used my own sweep-error code and my own MLP forward pass, which agrees with the library's to 6e-8.

| Set (arm median, cm) | reference | direction | drop | other |
|---|---|---|---|---|
| Hold-out, 2080 paths | 11.02 | 3.50 | 0.68 | hold-q0-offset 11.12 (-0.01); dense reference 10.97 |
| Fresh, all 252 replays | 4.55 | 0.90 | 0.80 (CI 0.77-0.82) | hold-q0-offset 5.11 (-0.12); dense reference 4.55 |
| Fresh, box (12 routes) | 8.23 | 3.79 | 0.54 (0.39-0.69) | |
| Fresh, edge (9 routes) | 7.63 | 2.57 | 0.66 (0.53-0.80) | |
| Fresh, under (42 routes) | 3.40 | 0.75 | 0.78 (0.76-0.81) | |
| Fresh, box+edge only (21 routes) | 8.10 | 3.14 | 0.61 (0.48-0.70) | |
| Old 35 routes | 6.46 | 2.27 | 0.65 | hold-q0-offset 8.63 (-0.34); dense reference 6.46 |
| Fresh palm only | 2.81 | 0.69 | 0.75 | box 0.61, edge 0.64, under 0.75 |
| Hold-out palm only | 7.80 | 2.70 | 0.65 | |
| Fresh, second-half ticks only | 4.55 | 0.79 | 0.83 | box 0.61, edge 0.69, under 0.82 |

**Item 2, leakage and fairness (verified unless marked).**
- (a) The inputs are the state at the command and the two samples before it (pre-command hold), jaw pose, target, speed, pelvis flag, hand, plan duration, and the local reference (the plan). None carries motion recorded after the command. The via points are not inputs at all (G_COLS drops them).
- (b) Each path is one chain-0 row with a unique reset id, and buckets are hashed per reset. Early-stopping and calibration halves are md5-split inside bucket 8, disjoint (1028/1077). No sector path is in train. The test and train reset sets are disjoint. data_via and data_via2 share 0 resets (different ID0 and seed ranges). Fresh resets overlap training resets 0.
- (c) Fresh and old routes appear only in `eval`. `train` is called with data_via and data_via2 only, and dev.log and the dev_a..e sweep use validation only. The caveat is design influence: the round-2 data gaps (standing starts, long paths) came from a review of the old 35 routes. Fresh scene parameters were tuned on two trial seeds for planner success only.
- (d) The baseline is not handicapped. I rebuilt a dense reference (min-jerk joint blend plus forward kinematics, 50 Hz, checked against the poses to 0.1 mm). Its error equals the 24-point reference's to 0.01 cm on all three sets.
- (e) k0 is the last sample before `m.reach` and follows a 0.3 s pre-hold, so it cannot contain command-caused motion. I read the collector code for this and did not test it empirically.

**Item 3, trivial part.**
- Holding the q=0 offset is no gain (-0.01 on hold-out, -0.12 on fresh). The q=0 offset is 3.4 cm, but the arm moves away from it.
- The constant-offset baseline gives 0.05-0.10.
- The second-half-only drop is not smaller, it is higher (0.83).

**Item 4, shuffles on fresh (3 repeats, within kind and walked flag).** The state inputs alone are not the explanation. Shuffling them gives dir 1.5 cm (drop 0.66), so the state accounts for about 0.14 of the 0.80 drop. Shuffling the local reference together with the plan duration gives 3.3 cm (drop 0.27), which is close to a state-only constant. Shuffling the goal gives 0.91 (no change). Shuffling speed, pelvis, hand and duration gives 0.93 (no change). The model is driven mainly by the path's local geometry plus some start state, as a path model should be. Most of the gain is a systematic tracking bias given the reference geometry (inference).

**Item 5, fair test?**
- Fresh paths differ from training. All are standing, with a median palm path of 0.39 m against 0.68 m, and median duration 2.5 s against 4.9 s. They have 1, 2, 3, 7 or 8 via points, 68 of 252 of them with exactly 1. Targets are higher (z about 0.94 against 0.64).
- Routes are near-duplicates of each other. Median nearest-neighbour palm-path distance is 2.3 cm, against 9.2 cm among random training paths. The 42 "under" routes are one near-straight family (palm path about 0.34 m).
- Under routes are 67% of the paths and drive the 0.80. Detour routes give 0.61 (CI 0.48-0.70), only box+edge, 21 routes, and per-route spread is wide. 2 of 21 routes are worse than the reference, and box route e64 goes from 9.4 to 11.1.
- On random hold-out standing paths longer than 0.7 m, the drop is 0.68-0.69, so 0.61 is consistent.
- Fresh routes test route shape only. The model has no scene input, so obstacle contact is not tested (I did not verify whether obstacles were present in the replay).
- Fresh margin coverage for direction is 0.93, below the nominal 0.95.

**Item 6, criteria as written (all verified from eval.log).**
- Stop line: validation standing paths over 1 m dropped 0.59 (CI 0.50-0.72), at least 0.25. Met.
- Fresh drop at least 20%: 0.80. Met.
- Fresh CI excludes 0: 0.77-0.82. Met.
- Fixed-radius ratio at most 0.8: 0.63. Met. It is calibration-set based and the radius is a global constant, so it is partly set by random-path error rather than by the fresh routes. On fresh the p90 error ratio is 0.51.
- Fresh route count at least 20: 63, so the final conclusion is not limited to hold-out and long paths.
- Predictions: all were far beaten. Fresh predicted 20-35%, got 80%. Ordinary hold-out predicted 40-50%, got 68%. The 1 m standing set predicted 25-40%, got 61%. These are pre-registered predictions, not criteria. That large a miss in the favourable direction fits a conservative prior; I found no sign of an artifact behind it.

Not checked: a retrained model without state inputs, seed variance on the fresh routes, and whether closed-loop use of the predictions (the narrow-slot test) holds up.

### 第三步的复核（2026-11-22）

新开的 agent，给了两组数据、模型、原样脚本、第三步事先写的设计和判据、以及对比的数字。它的代码在 `/workspace/tmp/review_via_comp_20261122/`。原文：

Result: the headline numbers reproduce, the pairing is valid, and I found no flaw that inflates the result. The two places I rely on the scripts' own numbers are the 2x model-residual criterion and the plan-beforehand-vs-executed check. My code is /workspace/tmp/review_via_comp_20261122/rev.py and an.py (own polyline-distance function; same wanted sweep comp_ref0 for both arms).

TABLE (detour, 84 pairs, cm medians; drop with route-bootstrap 95% interval; share of pairs improved)
- palm vs wanted sweep: 4.83 -> 2.99, drop 0.38 [0.31,0.48], 0.95
- arm (24 spheres): 8.10 -> 6.08, drop 0.25 [0.16,0.33], 0.88
- palm settled end vs wanted end: 3.01 -> 0.77, drop 0.74 [0.69,0.78], 0.99
- palm, plan-duration ticks only: 4.73 -> 2.96, drop 0.38 [0.30,0.48]
- arm, plan-duration ticks only: 7.94 -> 5.95, drop 0.25 [0.16,0.34]
- palm, ticks before 80% of plan: 4.22 -> 2.55, drop 0.39 [0.26,0.46]
- arm, ticks before 80% of plan: 7.07 -> 4.80, drop 0.32 [0.20,0.42]
- palm, ticks after plan end: 3.89 -> 1.48, drop 0.62
- arm, ticks after plan end: 6.93 -> 5.58, drop 0.19 [0.07,0.35]
- Per link group drop: shoulder/upper arm 0.00 [-0.20,0.12]; elbow -0.02 [-0.20,0.17]; wrist -0.12 [-0.39,0.14]; palm 0.38; fingers 0.30 [0.25,0.40]
- FreshUnder (168 pairs): palm 2.26 -> 1.91 (drop 0.16), arm 3.40 -> 3.34 (drop 0.02), as reported.

1. Recomputation (verified by computation): medians match to 0.01 cm and intervals to within 0.01. Using the full-resolution reference polyline instead of the 24 samples gives the same result (palm 4.80 -> 2.91, arm 8.10 -> 6.01). Rms over ticks gives palm drop 0.57 and arm drop 0.30, so the result does not depend on the max-over-ticks definition.

2. Pairing (verified): all 252 pairs have identical reset ids and k0. S at ticks k0, k0-1 and k0-2 is bit-identical in all 252, as are the whole pre-command S history, W at k0 (including the jaw), J, start, pelvis_z, hand_q, rot6 and side. The only differing static fields are the commanded target and via_p (at most 8 cm, as intended). Wanted is the same object in both arms: comp_ref0 equals the direct run's reference (0.00 mm per comp_eval.out; I did not recompute that check).

3. Fairness (verified): detour median ticks are 76 direct vs 77 comp, plan duration 5.13 s vs 5.12 s, reach time 6.56 s vs 6.58 s, palm path length 0.865 m vs 0.854 m. Restricting both arms to the common tick window leaves the numbers unchanged. The metric is one-sided, so a path that stopped early could score well. I checked the reverse direction (wanted palm point to nearest recorded point): worst case is 4.3 cm direct vs 3.1 cm comp, so the compensated arm does not stop early.

4. Where the gain comes from: the end point is not the whole story. The settled end has the largest relative gain (0.74), but error before 80% of the plan still drops 0.39 for the palm and 0.32 for the arm, so the path itself moves. Inference only: some of that path gain may come from the goal shift pulling the last leg. I cannot separate it without a goal-only run.

5. Worst pairs and cap (verified, except the cap count taken from comp_shift): the 8 cm cap was hit in 1 of 84 detour pairs (2 of 252 overall), so no capped-vs-uncapped comparison is possible. 12 detour pairs moved a point by 6 cm or more; the median largest shift per pair is 4.3 cm. Gain by largest shift: 2-4 cm (n=34) palm 3.90 -> 2.11, arm 7.09 -> 4.67; 4 cm or more (n=50) palm 5.41 -> 3.46, arm 9.25 -> 6.40. Palm worse in 4 of 84 detour pairs: FreshBox_L0_e115 (3 passes; worst 5.55 -> 6.88, largest shift 4.7 cm) and FreshBox_L0_e64 (4.29 -> 4.72). Arm worse in 10 of 84: FreshBox_L0_e115 (11.2 -> 14.2, 12.0 -> 14.7), FreshEdge_L0_e66 (5.7 -> 7.8), FreshBox_L0_e64. Routes e115 and e66 are worse on the route median for the arm; only e115 is worse for the palm. Nothing is catastrophic, and none of these pairs hit the cap.

6. Arm result (verified): the arm gain is entirely palm and fingers, which are the commanded hand's spheres. Shoulder, elbow and wrist show no gain (point estimates 0 to -0.12, intervals span 0). The "arm drop 0.25" is a maximum over spheres carried by palm and fingers, not evidence that the elbow or shoulder follow the hand.

7. Criteria, literal:
- Palm drop at least 20% with interval excluding 0: met (0.38, lower bound 0.31).
- Arm interval excludes 0: met literally (0.16 to 0.33), so the narrow-gap test design proceeds; the link-group split above shows the gain is palm and fingers only.
- Falls or planning failures not more than 5 above direct: met (0 vs 0).
- Model residual at most 2x direct: met per comp_eval.out (3.20 vs 3.14); I did not recompute it.
- Predictions: palm 30-55% (0.38) and arm 10-35% (0.25) both fall inside the predicted ranges.

One discrepancy: for FreshUnder, comp_eval.out says the palm improved in 0.73 of pairs. My strict count gives 0.58 (0.96 non-strict), so many pairs tie. I did not find the cause, and the medians still match. It does not affect the detour result.

### 第四步第一部分的复核（2026-11-22）

新开的 agent，给了计分的输出、三组回放、模型和训练表、原样脚本、第四步事先写的设计和判据。它的代码在 `/workspace/tmp/review_via_gap_20261122/`。它说没法核对回放的路线是否不在随机训练表里：两者是分开采的数据集，开场编号不重叠。原文：

All numbers below are computed by me, in /workspace/tmp/review_via_gap_20261122/ (rev.py, rev2.py, rev.out). I re-implemented the sweep error and the go rules. I reused the repo's table loading and model prediction (VP/VM) and the saved radius.pt. Nothing under /workspace/code/ was edited.

**Table (detour routes, 84 reaches, "through" set)**
| | small | ours | small + ours' go rule | fixed | radius | const |
|---|---|---|---|---|---|---|
| error med/90/95 (cm) | 8.06/11.51/12.35 | 5.73/8.70/9.26 | 8.06/11.51/12.35 (original command, same as small) | n/a | n/a | 8.88/11.92/13.11 |
| narrowest safe gap | 12.5 | 11.25 | 12.5 | none ≤15 | none ≤15 | none ≤15 |
| success / collide at c=8 | .49/.51 | .43/.01 | .33/.11 | 0/0 | 0/0 | 0/0 |
| success / collide at c=10 | .77/.23 | .79/.00 | .68/.11 | 0/0 | .06/0 | 0/0 |
| success / collide at c=11 | .86/.14 | .88/.00 | .83/.05 | 0/0 | .14/0 | 0/0 |
| ours' abstention at c=6, 8, 10, 12 | none | .94, .56, .21, .07 | same | | | |

**1. Reproduction (verified).** All three error rows, every curve (small, fixed, radius, const, ours) and the narrowest gaps match gap.log to two decimals. Detour: 12.5 and 11.25. All routes: 11.25 and 9.25. I did not redo the bootstrap intervals.

**2. Opponents (verified).**
- Standing-only calibration gives these new values.
  - The direct fixed margin goes from 19.39 to 15.30 cm.
  - The const margin goes from 19.01 to 15.49 cm.
  - The radius scale goes from 1.49 to 1.56.
- Fixed and const still never go on the 2–15 cm grid, because 15.3 and 15.5 are above it. They would pass at about 15.5 cm.
- Radius is slightly worse: success 0.06, 0.10 and 0.15 at c=10, 11 and 12, with no collisions. It abstains on 85 % or more.
- Oracle fixed margin on the detour routes: 12.5 cm at best, identical to small. A fixed margin cannot beat a never-abstaining way.
- Oracle const margin: about 13.25 cm.
- Oracle radius scale (set on the test routes): 0.75 gives 12.25 cm, 0.5 gives 12.5 and 1.0 gives 13.5. It beats small by 0.25 cm only with an oracle scale.
- No opponent beats small or ours under fair calibration.

**3. Is "ours" favoured?**
- Verified in code: the go rule is predicted-sweep (from comp_ref and comp_off) + residual radius ≤ c. It uses no recorded outcome.
- The residual-radius nets and their scale come only from the random tables (data_via and data_via2).
- I could not check whether the replay routes were absent from those random training tables.
- Calibrating the residual scale on standing paths only makes "ours" worse, not better. The scale goes from 1.82 to 2.06. The detour narrowest gap becomes 12.0 cm, a ratio of 0.96 to small. All routes become 10.0 cm, a ratio of 0.89. So the all-crouch calibration is not what makes "ours" look good.
- "small + ours' go rule" stays at 12.5 cm with 5 to 15 % collisions at c=8 to 12. So the shifted command contributes the safety. The abstention alone does not.
- Success, ours minus small: +0.07 at c=9, +0.01 at c=10, +0.02 at c=11. The route-bootstrap 95 % intervals are −0.06 to +0.19, −0.12 to +0.16 and −0.04 to +0.10. These intervals include 0.

**4. Hand error (verified).**
- Median "through" error by link, original → shifted:
  - wrist: 2.9–3.1 → 2.8–3.6 (no gain, as in step 3).
  - palm: 4.3–4.8 → 2.4–3.0.
  - thumb: 3.3–3.5 → 2.3–2.9.
  - index_0: 5.0–5.9 → 2.9–3.5.
  - middle_0: 5.5–6.3 → 3.4–3.9.
  - the outermost link spheres (middle_1, index_1): 6.6–7.9 → 4.1–5.6.
- The largest sphere error is middle_1 in 68 of 84 original reaches and in 56 of 84 shifted ones. The wrist is largest in 3 and 12 respectively. This is a time-matched comparison at the plan's 24 moments plus the settled end, so it is stricter than the sweep error.
- Palm position error at the settled end: 3.01 → 0.77 cm median, 4.65 → 1.56 at the 90 % level.
- Hand attitude error at the settled end, from palm and fingertip vectors: 16.5° → 16.3° median, 24.6° → 21.9° at the 90 % level. Along the path: 9.3° → 9.5° median.
- Palm-to-fingertip length is 10.3 cm. A 16° swing moves the fingertip about 2.9 cm relative to the palm. That equals the fingertip-minus-palm gap that remains after the shift (5.5 vs 2.4–3.0).
- So the remaining error is mainly attitude, which the shift does not touch. This is an inference from geometry, supported by the numbers above.

**5. Is the headline metric the right summary?**
- It hides that success is nearly the same for small and ours. Collisions are what differ: small collides on 14 % at c=11 and 23 % at c=10, ours on 0 %.
- The 0.90 ratio comes from a 5 % collision threshold crossed on a grid of 84 reaches (21 routes). The ratio interval 0.79–1.18 includes 1.
- Ours' success plateaus at 0.94–0.95 above 13 cm because it abstains on 5 % of reaches. Small reaches 0.99 at 15 cm.
- Under an unweighted "success only" reading, ours is no better than small.
- The curves do not show real obstacles, crouching, consequences of contact, or whole-arm gaps. On the 24-sphere set, ours does not reach 90 % within 15 cm.
- The curves do show that, among the ways tested, only ours is nearly collision-free: 0.01 at c=8 and 0.00 above. That is the clear advantage.

**6. Plan criteria.**
- Literal reading: the ratio is 0.90 against the best opponent (small). That falls in the 0.8–0.95 band, "report to the user to decide". It is not at or below 0.8 and not above 0.95.
- The "never 90 % at c≥12" clause is triggered for fixed, const and radius.
- Cause: the 95 % calibration margins are 19.4 and 19.0 cm, and the radius scale of 1.49 inflates the predicted tube. The margins are dominated by crouching paths (cal 95 % level: crouch 20.7, standing 15.3).
- Standing-only calibration still leaves fixed and const above the grid. This is a calibration design effect of a 95 % level plus the 15 cm grid cap. It is not a scoring bug (my recomputation matches). The plan's predictions of 11–15 for fixed and 8–12 for radius did not hold.
- Both the "ours ≤ 0.8× best opponent" and the "wrist makes ours no better than the opponents" readings are not supported. The detour ratio is 0.90 (all routes 0.82, interval 0.77–0.93).

### 第四步补充之二和第五步（SONIC）的复核（2026-11-23）

复核人只拿到数据、脚本、结果文件和方案里的判据，没有拿到我的解读。它的报告全文和代码已拷到 `outputs/reachwm_20261030/review_via_onpol_20261123`。交回的摘要原文：

> Review of the on-policy / wrist-attitude compensation run. I recomputed every item with my own code except check 2 of item 3, and no number differs from the author's. Full detail is in /workspace/tmp/review_via_onpol_20261123/REPORT.md; scripts c1b, c2, c4, c4b, c5, c6, c7 and c8 sit beside it.
>
> **1 Route separation: confirmed; disclosure only partly checkable.**
> - Training, development and test routes share 0 scenes, 0 goals and 0 identical via files (some file names repeat, but their contents differ).
> - The `dirs` of via_point_onpol2 are data_via, data_via2, data_via3 and the on-policy replays r1_direct, r1_point, r1_pose and r2_pose. All of those replay `train_routes/subset`. No development or test replay is in there.
> - The test routes were used earlier: data_via_fresh, data_via_comp, data_via_const, via_gap_20261122, and 8 routes in a trial of an earlier pose version. The plan discloses the 8-route trial.
> - That trial's half-gain, 6-round and 40 deg settings are still in the final method. $I holds no output directory for the trial, so I could not check it.
> - On the 13 unseen detour routes the result holds: direct 7.79 cm to ours 3.61 cm.
>
> **2 Error recompute: confirmed.** On 21 detour routes, 84 pairs, wrist+palm+fingers goes 8.06 to 3.40 cm (90 % level 11.51 to 6.80). Palm goes 4.83 to 2.40 cm, and all 24 spheres go 8.10 to 7.33 cm. Settled-end attitude goes 19.3 to 7.1 deg. `sweep_err` matches a brute-force point-to-polyline distance to 1e-16 m.
>
> **3 Consistency checks: confirmed for the first and third.** The largest differences are 1.4e-7 m, and all 252 pairs match. I did not recompute the second check.
>
> **4 Narrow-gap scoring: confirmed.**
> - `--base` and `--cal` contain no test information.
> - I get small 12.50 cm and ours 12.75 cm, ratio 1.02, so there is no advantage.
> - Ours needs 12.75 cm because its go rule adds the predicted error (median 2.7 cm) to 3.04 times the radius (median 4.3 cm). That sum has p90 12.7 cm, while the real error has p90 6.8 cm and a maximum of 13.6 cm. Actual error never exceeded the bound, so coverage is 100 % against a nominal 95 %.
> - The other opponents (fixed, radius, const) never pass, because their margins are about 19 cm. "Best other way" is therefore only "small", and its threshold is read from the test error distribution after the fact.
> - Rule B, recomputed with the margin from the development run only (3.28 cm), gives 9.0 cm, or 8.25 cm on the 13 unseen routes.
> - Rule B was invented after the test result was known, so it is a hypothesis for a new run, not evidence. Its interval is wide (ratio 0.64 to 1.20).
>
> **5 Stopping rule: applied as written, but noisy.** Round 1 median is 3.907 cm and round 2 is 3.786 cm, an improvement of 3.1 % (below 5 %), so it stops. The route-bootstrap interval is -4 % to +11 %, so round 2 is not shown to be better than round 1.
>
> **6 `onpolicy_check`: confirmed.** The split is by route index % 4, no scene is shared between routes, and the network settings are identical to the old model's. The printed numbers follow from the code.
>
> **7 SONIC: confirmed.** Reference error is 8.55 cm and the model's error is 4.08 cm, drop 0.52, matching `by_kind.txt`. On the wrist+palm+fingers set the drop is 0.54, so it is not inflated by elbow and shoulder. The stop line was evaluated at 04:22 and the test replay started at 04:33. The start state differs from ScaleBFM by 4.9 cm.
>
> **Problems, most serious first:**
> 1. The compensation does not improve the whole arm. The all-24 drop is 0.10 with an interval that includes 0, and ours' 95th percentile is worse than direct (15.15 vs 12.64 cm). The positive claim holds only for wrist+palm+fingers.
> 2. The pre-registered gap criterion failed (1.02 against a predicted 0.6 to 0.8). The favourable analyses (rule B, always-go) are all after the fact.
> 3. The 8-route peek shaped the final settings, and its outputs are not preserved.
> 4. The `ok`, `jaw_err_cm` and `att_err_deg` fields in `all.npz` are measured against the command sent, not the original goal. On detours `ok` falls from 0.417 to 0.107 and attitude error rises from 20.6 to 37.6 deg. They must not be read as task success, and no task-level success is supported.
> 5. The error is one-sided and ignores lag. The settled-end checks cover stopping short.
> 6. The dev-set stop decision and the 3.04 radius scale rest on 41 routes, so they are fragile.

我对六个问题的处理：

1. 属实，写进结论：补偿朝向只对腕、掌、指成立；整条手臂 24 个球不成立（降 0.10，区间含 0），95 % 分位还比直接发命令差（15.15 对 12.64 cm），因为躯干多倾、肩偏出去。
2. 属实，结论里就是这么写的：事先写的窄缝判据没过；规则 B 和"总是出发"都是事后的。
3. 属实。8 条试跑的输出当时只在临时目录里，已拷到 `outputs/reachwm_20261030/smoke_test_routes_20261122`（两版各一份）。"每轮一半、6 轮、上限 40°"这三个数确实是看过这 8 条之后定的，之后没有再调。
4. 属实。补偿回放里 `jaw_err_cm`、`att_err_deg`、`ok` 是相对实际发出的命令算的，不是相对原来的目标，不能当任务成功率读。本条目里没有用这三个字段下结论；`gen_reach.py` 的注释里补了这一句。
5. 属实，是这个误差的定义：只算偏到路线旁边多远，不算滞后；停得不够远由终点误差单独看。
6. 属实。停在第 2 轮和 3.04 这个缩放都只靠 41 条开发路线。第 2 轮没有被证明比第 1 轮好，只是按规则用了最后一轮。

### 第四步补充之三（新测试路线）的复核（2026-11-23）

复核人只拿到数据、脚本、结果文件和方案里的判据。报告全文和代码已拷到 `outputs/reachwm_20261030/review_via_test2_20261123`。交回的摘要原文：

> Review of the second test set (test2): the pre-registered criterion is NOT met. I found no leak. Full report: /workspace/tmp/review_via_test2_20261123/REPORT.md.
>
> 1. **Separate set: confirmed.**
> - Scene dicts and via arrays share nothing with the train, dev or first test routes. Seeds differ (20261124 against 20261123, 20261122, 20261121).
> - The model's training datasets are data_via, data_via2, data_via3 and the onpol_* replays of train_routes/subset. No test2 or dev replay is among them.
> - Timeline: model 04:58, dev calibration run 05:03, via_gap.py 05:45, test2 scenes 05:48, test2_direct started 05:53. The asrun snapshots equal the current scripts.
> - Caveat: the notebook section is uncommitted and its mtime is 06:31, after the results. So I cannot prove from timestamps that the pre-registration came first. The margin rule was chosen after seeing the first test set.
>
> 2. **Error against the wanted sweep: confirmed.** On the 180 detour reaches (45 routes), median / 90 % / 95 % in cm are direct 8.39 / 12.27 / 13.60 and compensated 3.88 / 9.28 / 13.84. These match comp_eval.out and gap.log. The 95 % tail is not better than direct.
>
> 3. **Narrow-gap rows: confirmed.**
> - Margin m is 3.28 cm and fixed_dev is 12.56 cm.
> - Narrowest half-width with success >= 0.90 and collisions <= 0.05: small 13.75, fixed_dev 13.75, ours 13.25 cm.
> - Ratio is 0.96. My 95 % interval from resampling routes is 0.73 to 1.12; gap.log gives 0.73 to 1.10.
> - The criterion needs <= 0.8, so it is not met. The prediction of 8-10 cm for ours failed.
> - Ours passes with exactly 162/180 = 0.90 success and 0 collisions. With m = 3.0 cm the narrowest is 13.0 cm; with m = 3.6 cm it is 13.75 cm (ratio 1.0).
>
> 4. **Reaches far from the wanted sweep: confirmed, with a qualification.**
> - 20 of 180 compensated detour reaches (11 %) have error above 9 cm, on 9 of 45 routes.
> - 19 of those 20 have the largest shift at 8 cm, and all 20 have the largest turn at 40 deg. Among the other 160 reaches the shares are 34 % (shift) and 44 % (turn).
> - Saturation is necessary but not sufficient. Of the 61 reaches saturated in both, only 19 (31 %) are far.
> - `comp_pairs.npz` stores `reset` as the index (reset % 100000), so a join by row order gives wrong numbers. Align by `reset % 100000`.
>
> 5. **Success / collision rates at 6, 8 and 10 cm:**
>    - small: 0.14/0.86, 0.45/0.55, 0.76/0.24.
>    - fixed_dev: abstains throughout (0/0).
>    - ours: 0.46/0.00, 0.70/0.006, 0.83/0.00.
>
> 6. **Problems, most serious first:**
>    - The headline is a tie, so the claimed gain is not shown.
>    - The result is knife-edge: one reach and the margin (m = 3.0 to 3.6 cm) move it.
>    - The bootstrap ratio interval silently drops 18 % of resamples where the opponents pass nowhere up to 15 cm, and the grid is capped at 15 cm. The interval is conditional and truncated.
>    - Ours misses 90 % through abstentions, not collisions. The roughly 11 % far reaches are not rescued by compensation.
>    - The pre-registration is not verifiable by time.
>    - The margin held on test2: its 95 % level is 3.31 cm against 3.28 cm, and 6 % of reaches exceeded it. I found no leak.

我对它指出的问题的处理：

- 打平、没有显出优势：属实，结论里就是"不成立"。
- 结果卡在一次伸手上：属实。成功率刚好 162 / 180；余量取 3.0 cm 是 13.0 cm，取 3.6 cm 是 13.75 cm。这个指标在难路线占一成上下时本身就不稳。
- 自助抽样的区间丢掉了对手在 15 cm 以内完全过不去的那 18 % 重抽样，网格也只到 15 cm：属实，区间是有条件的、被截断的，引用时要说明。没有改脚本重算，因为结论不靠这个区间的上端。
- 顶到上限是"补不动"的必要条件，不是充分条件（两项都顶到上限的 61 次里只有 19 次补不动）：属实，上面"为什么"一段的说法应当按这个读。
- 事先写判据的时间从文件上证明不了：属实。这一节是在跑之前写的，但没有单独提交，后来又在同一个文件里追加了结果，文件时间就晚于结果了。以后事先写的判据要在跑之前单独提交一次。

### 第四步补充之四（第三批测试路线）的复核（2026-11-23）

复核人只拿到数据、脚本、结果文件和提交里的判据（`91f03b5a`、`ea13f62a`、`9a8637f0`），没有拿到我的结论，也不读工作区里的方案文件。交代的话在 `outputs/reachwm_20261030/review_via_test3_20261123/BRIEF.md`，复核人自己的代码和输出也在那个目录。报告全文：

> # Review of the third test set (test3, seed 20261125) - 2026-11-23
>
> Own code: rv.py (loader, own sweep error, own bootstrap), partA.py (hand error), partB.py / partC.py / partD.py (scene
> check, frame, location, sensitivity), partE.py (controller's own ok flag). Outputs: A.out, B.out, C.out, D.out.
> Everything read from all.npz / arm.npz / table.npz / prep_*.npz; the experimenter's analysis scripts were only read.
> Bootstraps resample the 49 routes (2000 draws, my own seeds).
>
> ## Findings
>
> 1. Order of events is clean for the registered criteria, with two soft spots.
>    - Routes planned 09:52 (test3_routes mtime), 91f03b5a 07:38 (plan for arm/wide limits), ea13f62a 09:56:36 (design, predictions,
>      criteria), first replay launched 10:03 (asrun copied 10:03:33). So the ea13f62a criteria precede every replay. Scripts as run
>      are byte-identical to the working copy (cmp: via_comp, via_comp_eval, via_point, arm_spheres).
>    - Run order: direct 10:12:48, const 10:24:00, arm 10:42:24, pose 10:57:48, gap scoring 11:16, point 11:27:24, scene check 11:27:33.
>    - 9a8637f0 (supplement: boxes put back, extra `point` run, predictions, criteria) is 10:33:09. It says "written before any
>      test result was looked at". By then direct and const were finished and test3_const/comp_eval.out (direct vs const errors)
>      existed since 10:24; arm was still running. Whether anyone looked cannot be shown from files. The scene check itself on the
>      test routes ran at 11:27, after the commit; via_scene.py (md5 509ef2ef, matches the registered md5) was written 10:08:35 and
>      already run on the development routes at 10:08:40. The test boxes (prep_*.npz) existed since 09:52, so nothing about the
>      test routes' boxes was hidden; the supplement was designed after seeing the dev scene check (34.8 % direct, 15.2 % point,
>      20.1 % arm, 26.2 % pose), which is the same ordering as the test result - the predictions were informed by that.
>    - The point run was NOT part of the four registered replays; it was added in 9a8637f0 and ran last. Its model
>      (via_point_20261121) differs from the arm/pose model (onpol4), see 6.
>
> 2. Headline hand numbers reproduce exactly (A.out). Wrist+palm+fingers (the 17 spheres wrist_pitch + all hand_*), swept-region
>    error, 49 routes / 196 reaches, median cm: direct 8.66, const 9.04, point 6.13, arm 3.69, pose 3.94. 24-sphere medians
>    9.05 / 9.66 / 7.31 / 7.32 / 7.55. Drop vs direct (route bootstrap): const -0.04 (-0.13 to -0.01), point 0.29 (0.22-0.36),
>    arm 0.57 (0.40-0.65), pose 0.55 (0.39-0.64). arm vs pose: drop of the median 0.064 (-0.036 to 0.142); mean paired difference
>    0.00 cm (-0.25 to +0.26); arm lower on 54 % of reaches, tied on 13 %. Still above 9 cm: direct 90/196 (45.9 %), const 98,
>    point 36 (18.4 %), arm 28 (14.3 %), pose 29 (14.8 %); arm minus pose -0.005 (-0.046 to +0.031). All match the experimenter's
>    files. (My single palm-sphere medians differ from their "palm" (3 spheres); not an error.)
>    - The arm-over-pose choice made on the development routes (1-22 %, excluding 0) did not reproduce: the interval includes 0.
>      The 90 % level is worse for arm (12.02) than pose (10.50) and than point (10.87): the median gain does not reach the tail.
>    - Const is worse than direct and is a weak opponent by construction (one offset for all routes); the real opponent is `point`.
>      arm vs point: median drop 0.40 (0.20-0.49) - but see 3: the other metric reverses.
>
> 3. Scene check reproduces exactly (B.out). Touch = any of 24 spheres inside a box at any 10 Hz tick (my own box distance).
>    direct 90/196 = 0.459, const 0.617, point 0.102, arm 0.260, pose 0.276; the wanted plan itself 0.066. Differences vs direct
>    (route bootstrap): const +0.158 (+0.071 to +0.245) worse, point -0.357 (-0.469 to -0.240), arm -0.199 (-0.332 to -0.077),
>    pose -0.184 (-0.296 to -0.071). arm minus point +0.158 (+0.077 to +0.255): 35 reaches touch only for arm, 4 only for point.
>    So on collisions the cheap position-only `point` beats the headline method clearly, while on hand shape arm wins. The text
>    of the registered rule makes the next step `point`-based.
>    - Frame: arm.npz and the boxes are in one frame to about 1 cm: palm sphere at the first tick minus prep jaw_p = (+2.5,-0.4,-1.9) cm,
>      sd (0.9, 0.5, 1.4) over 196 reaches (constant offset = palm-to-jaw, small spread). The wanted plan touches only 6.6 %, consistent
>      with the planner leaving ~1.5 cm. But the score is very sensitive to the frame: moving all boxes +3 cm in x gives direct 0.133,
>      point 0.082, arm 0.245 (arm worse than direct!); z +3 cm: 0.929/0.806/0.944; z -3 cm 0.393/0.224/0.311; yaw 5 deg: 0.362/0.102/0.255
>      (C.out). A 1 cm frame error is not excluded and the absolute rates (and in x even the ordering direct vs arm) depend on it. The
>      ordering point < arm < direct holds under y, z and yaw changes, not under +3 cm in x.
>    - Radius: sphere radii are 1.00 cm (fingers etc.) up to 4.01 cm (palm). Scaling all radii x0.5..x1.25: direct .301-.510, point .071-.173,
>      arm .199-.327, pose .245-.367; ordering point < arm <= pose < direct < const is stable. Tolerance: requiring penetration > 0.5 cm /
>      > 1 cm: direct .301/.153, point .087/.036, arm .209/.117, pose .255/.209. Near-miss within 0.5 / 1 cm counts as touching:
>      direct .602/.791, point .235/.454, arm .383/.577. Touch depth median 0.78 cm (direct): most "touches" are sub-centimetre
>      overlaps with 1 cm finger spheres whose own planner margin was ~1.5 cm. At > 1 cm penetration pose (.209) is worse than arm
>      (.117), a gap that is not visible at the registered threshold (arm-pose +0.015, -0.041 to +0.082).
>    - 10 Hz: interpolating ticks to 100 Hz raises rates slightly (direct .480, point .133, arm .265, pose .291, const .633): ordering unchanged.
>    - Which spheres: fingers dominate (hand_index_1, hand_thumb_2, hand_middle_1). For arm/pose shoulder/elbow touch in 7.1 % / 9.2 % of
>      reaches (arm 14 shoulder_yaw, 11 elbow) against 2 % for direct and point; 6 (arm) and 7 (pose) reaches touch with a non-hand sphere only
>      (D.out). That is the cost of the wrist-turn compensation (the shoulder is displaced ~7 cm, unfixed).
>    - Where: touches occur along the way, not at the goal. First touch at median 0.28-0.43 of the ticks; palm-to-goal distance at first
>      touch ~29-35 cm; only 0 (direct, point), 5 (arm), 7 (pose) reaches first touch in the last 20 % of ticks. Touch is by box: FreshBox
>      direct .554 / point .116 / arm .312 / pose .286; FreshEdge .333 / .083 / .190 / .262.
>    - Is point vs arm fair? Models differ (point: via_point_20261121 random paths only; arm/pose: onpol4 trained with replays), and the modes differ
>      (position only vs attitude). That is the question being asked, but `point` was run with the OLDER, less-tuned model and still wins on collisions.
>      Trivial explanations checked: point's settled palm end error is smaller (0.90 vs 1.17 cm), so it does not stop short; its hand
>      is slightly higher along the route (dz vs wanted +0.24 cm vs arm -0.27 cm; settled +0.52 vs -0.12) - but raising the whole arm
>      by 0.5-1 cm only moves arm 0.260 -> 0.204-0.219, point stays ~0.10 (D.out). So height does not explain it; the attitude turn
>      (arm swinging fingers into the box, plus shoulder/elbow) does.
>    - Caveat for the whole check: it is a geometric overlap against boxes that were not in the simulation, with the arm's own 24 spheres,
>      and the wanted plan already overlaps in 6.6 % (plan's spheres differ from the 24 spheres).
>
> 4. The controller's own success flag points the other way and is not mentioned in the registered metrics. On the 196 detour reaches
>    (all.npz `ok`, `jaw_err_cm`, `att_err_deg`; E): ok direct 72 (36.7 %), const 65, point 61 (31.1 %), arm 17 (8.7 %), pose 15 (7.7 %);
>    median jaw_err 2.30 / 2.40 / 2.32 / 4.13 / 4.88 cm; att_err 22 / 22 / 21 / 43 / 44 deg. Likely these are measured against the
>    (moved, turned) commanded goal and so not comparable - I could not confirm which goal they use - but a reader should not be
>    left thinking task success was checked. Direct "ok" is already only 37 %. The headline is path shape, not task success.
>
> 5. Narrow gap (gap.out): "ours" is arm, margin 3.81 cm from onpol_dev_r4_arm_narrow (the registered source). Narrowest safe half-width: small 13.75,
>    fixed_dev 13.75, ours 13.25 (10.75-14.50). ratio = 13.25/13.75 = 0.9636, interval (0.83-1.05). By the registered rule ratio > 0.95 -> does NOT hold.
>    Conditional / truncated: the ratio's bootstrap is `ours / nanmin(others)` and the interval is `nanpercentile`, so every resample in which
>    ours (or all others) never reaches 90 % success within the 15 cm grid ceiling is silently dropped (gap.err shows the All-NaN warnings
>    from exactly those lines). The grid stops at 15 cm and ours reaches only 0.93-0.95 success at 14-15 cm and 0.89 at 13 cm; small
>    passes at 13.75 - so a meaningful part of resamples can be NaN and the interval is conditional on being defined. I did not recompute
>    the bootstrap (needs the model sweeps), so I cannot say how many were dropped. Also: "narrowest" takes the FIRST grid point meeting
>    the rule, with no monotonicity check; ours at 13 cm is 0.89 vs the 0.90 line, i.e. the 13.25 is a one-step knife edge. Whether it
>    is better than best-other is not established (interval includes 1). At 8 cm: ours 0.61 success / 0.01 collision; small 0.44 / 0.56.
>
> 6. Other points that could mislead.
>    - arm and pose were chosen/tuned using the 41 development routes; their margin (3.81) comes from the same dev run: an optimistic margin
>      (the plan says so). On test the arm model's predicted-error check is 2.28 cm (direct 1.44), within the registered 2x limit.
>    - Predicted ranges were missed for several items (table below); some predictions written AFTER the dev scene check were still off by a lot (direct 46 % vs 28-42 %).
>    - The 9 cm "still above" measure counts wrist/fingers max over 17 spheres; arm's 90 % level got worse than direct-vs-point. Residual failures cluster:
>      10 routes have any, in arm, pose and point alike (vs_pose.out), so ~14 % of reaches are not fixed by any variant.
>    - const run is a deliberately weak opponent and was worse than doing nothing; do not read "arm beats const" as strong.
>
> ## Criterion by criterion (both commits)
>
> | Criterion | Result | Holds? |
> |---|---|---|
> | arm better than pose (median lower and route-bootstrap interval excludes 0) | 3.69 vs 3.94; drop 0.064 (-0.036 to 0.142) | NOT established ("cannot tell on the test routes") |
> | hand accuracy: arm drop vs direct >= 0.5 and lower end >= 0.4 | 0.574 (0.40 to 0.65) by me; their 0.57 (0.42-0.65) | Holds (lower end 0.42 theirs; my 0.40 is exactly at the line, resampling noise) |
> | fewer reaches still above 9 cm for arm than pose, interval excludes 0 | 28 vs 29; -0.005 (-0.046 to +0.031) | Fails (only report both shares) |
> | narrow gap ratio <= 0.8 and upper < 1 | 0.96 (0.83-1.05) | Fails; also > 0.95 -> "not established" |
> | validity: three consistency checks 0.00 mm | 0.00 in all comp_eval.out | Holds |
> | validity: falls not > direct + 5 | 0 falls in all five runs | Holds |
> | validity: model error on arm run <= 2x direct | 2.28 vs 1.44 (1.58x) | Holds |
> | scene: method reduces collisions vs direct (CI excludes 0) | point -.357, arm -.199, pose -.184: all hold; const +.158 (worse) | point, arm, pose hold; const does not |
> | scene: point vs arm, interval excludes 0 | arm - point +0.158 (+0.077 to +0.255) | Excludes 0: point better; next step on point |
> | 24 spheres / shoulder: reported only | 7.32 cm (arm), shoulder 7.08 | n/a |
>
> Predictions outside their stated ranges: >9 cm arm 14.3 % (5-11 %) high; >9 cm pose 14.8 % (8-14 %) slightly high; end attitude arm
> 10.2 deg (7-10) slightly high; direct end attitude 22.1 (19-22) slightly high; direct 24-sphere 9.05 (8.0-8.8) high; scene: direct touching
> 45.9 % (28-42) high; point 10.2 % (11-22) low; const 61.7 % (20-35) far high; shoulder/elbow for direct and point 2 % (0-1) and arm/pose
> 7-9 % (1-5) high; fingers-only share of arm touches 45/51 = 88 % (>90). Inside: direct hand 8.66 (7.5-8.7), arm 3.69 (3.0-3.9), drop 0.57
> (0.52-0.62), arm over pose 6 % (3-15), 24-sphere arm 7.32 (7.0-7.8), model error 2.28 (1.8-2.6), wanted plan 6.6 % (6-13), arm 26.0 % (15-27, top
> edge), pose 27.6 % (19-32), gap numbers ours 13.25 (11-13.5), small 13.75 (12-14), fixed_dev 13.75 (12.75-14), ratio 0.96 (0.85-1.0),
> 8 cm gap ours 61 % / 1 %, small 44 % / 56 %.
>
> ## Could not verify
> - Whether any test result was viewed before 10:33 (not recoverable from files).
> - The narrow-gap bootstrap: number of NaN-dropped resamples (needs the model sweeps and via_gap internals; not recomputed).
> - What goal `jaw_err_cm` / `att_err_deg` / `ok` refer to for compensated runs.
> - The true frame offset between arm.npz and the box frame beyond about 1 cm (only indirect evidence: palm-start vs jaw_p spread, wanted-plan touch 6.6 %).
> - The spheres' correspondence to the planner's collision spheres (the wanted plan overlapping 6.6 % suggests they differ).

对复核意见的处理：

- 所有数字用复核人自己的代码复现，判据逐条的结论和我的相同。
- **`9a8637f0` 的先后顺序有软肋**：属实。提交时直接发命令和常数修正两组已经跑完，常数修正的评测文件已经存在。我当时只看过各步的时刻和进度行，但这一点从文件上证明不了。放回箱子的检查本身是 11:27 才第一次在测试路线上跑的，这一点可以验证。第四批测试改成在规划路线之前就提交（`51d0bf33`）。
- **"碰到的比例"对箱子的位置很敏感**：属实，这是最要紧的一条。多数重叠不到 1 cm；箱子整体往前挪 3 cm，直接发命令从 46 % 掉到 13 %，`arm` 反而比直接发命令差。"只挪位置最少"在所有挪法下都成立，但 46 % 和 10 % 这两个绝对数不能单独引用。第四批测试因此改用不靠阈值的量（被吃掉的间隙的 95 % 分位），并把"箱子挪 ±1 cm 时比值是否还成立"写进判据。
- **控制器自己的成功标志指向相反的方向**：`ok`、`jaw_err_cm`、`att_err_deg` 在补偿回放里是相对实际发出的命令算的（`gen_reach.py` 里有注释），所以不能和直接发命令比。复核人说得对，报告里不能让人以为检查过任务成功。第四批测试加了相对想要的终点的"到位且不碰"。另外要写明：只挪位置不改手的朝向（终点朝向误差 21°，和直接发命令一样），加朝向是 10°。
- **只挪位置和 `arm` 用的模型不同**：属实，而且对只挪位置不利（它用的是没见过规划器路线的旧模型）。开发路线上只挪位置换成第 4 轮模型，碰到的比例相同（15.2 %）。
- **窄缝区间丢掉了没有定义的重抽样、13.25 cm 卡在一格上**：属实，和前两次相同。这个说法三次没过，不再用。
- **24 个球和规划器的碰撞球是否一致**：是同一份（`arm_spheres.py` 读 SIMPLE 的 `spheres/g1_29dof.yml`，规划器的机器人配置 `g1_29dof_with_dex3.yml` 指向的也是它；规划器另外给每个球加 0.5 cm 的缓冲，检查里没有加）。想要的路线本身有 6.6 % 碰到，是因为回放只沿用规划器的手部路点，手臂姿态由我们的全身逆解另外解出，和规划器的关节路径不同。
- 没法验证的几条照录在报告末尾，没有补做。
