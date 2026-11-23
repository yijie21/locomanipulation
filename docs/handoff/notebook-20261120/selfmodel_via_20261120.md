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
