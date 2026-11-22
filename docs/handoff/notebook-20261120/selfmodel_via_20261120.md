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
- 2026-11-21，用户定了覆盖检查的规则（原话："A 不加那一点"）：改成"不比 95 % 的随机路线更孤立"，生成器不再动，偏离量的下限仍是 3 cm。
  - 按新规则重算已有的两份 300 条数据：试跑那份 35 / 35，补生成器之后的探索那份 35 / 35。中位数规则分别是 14 和 9。
  - 正式采集的 12000 条出来后再算一遍，以那次为准。离线估计那时是 32 / 35。

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
