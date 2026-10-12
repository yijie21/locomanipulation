# GraspNet 物体 + graspnet-baseline 抓取 + bottle 专家：我们的 IK 规划 vs SIMPLE 的 CuRobo 规划

状态（2026-10-08；10-09 补了 ghost 和固定抓取对比，第 5b 节）：实验代码在 `scripts/experiments/graspnet_bottle_20261008/`，65 个物体各跑一局 IK、一局 CuRobo，左右对比视频和结果表在 `outputs/graspnet_bottle_20261008/`（第 5 节）。任务编号见 [TODO](../../TODO.md) 21.3.4，决策在 [DECISIONS](../../DECISIONS.md) 2026-10-08 末节。

## 1. 目标

用户 2026-10-08 定：把 SIMPLE 转好的 GraspNet 物体摆到 bottle 场景的桌上，从第一视角的深度点云用 graspnet-baseline 预测抓取，机器人直接站在桌边（不走路），用 bottle 线 SONIC 驱动的抓取逻辑去抓；同时跑 SIMPLE 的 CuRobo 规划器，和我们的 IK 规划器对比；每个能塞进夹爪的物体出一条左右对比视频。

## 2. 组成

| 部分 | 文件 | 做什么 |
|---|---|---|
| 场景 | `make_scenes.py` | 每个物体一份 bottle 场景副本，物体替换水瓶的位置：真实尺寸、可视网格 `manifold.obj`、CoACD 碰撞块、按 400 kg/m³ 算质量（0.03–0.40 kg）。选最高的、窄边 ≤ 9.5 cm 的稳定姿态；SIMPLE 的稳定姿态有 14 个在我们的仿真里会倒，所以每个场景先放 2 s，按停下来的姿态重建，直到 2 s 内移动 ≤ 1 cm、转动 ≤ 3°。75 个里 65 个可用，10 个任何姿态都比夹爪宽 |
| 抓取模块 | `graspnet_server.py` | graspnet-baseline（RealSense 权重），单独的 venv（torch 2.7 cu128，pointnet2/knn 为 sm_120 重编）。输入相机坐标系点云，输出 NMS + 碰撞过滤后的抓取 |
| 感知 | `grasp_planned.py` `DepthCamera` | 用这一刻的特权 qpos 在 MuJoCo 里渲第一视角深度（640×480），裁出物体周围 22 cm、桌面以上 40 cm 的点 |
| 抓取转换 | `grasp_planned.py` `candidates` | graspnet 的接近方向 = Dex1 掌心 x，夹紧方向 = 掌心 y；两种绕接近轴 180° 的翻转都试。Dex1 沿接近方向推到 graspnet 深度允许的最深处，要求：张开的手（`hand_cloud.py`，夹爪全开 9.5 cm）不碰桌面和物体点云；从预抓到抓取整段直线进入都不碰；两指垫之间至少有 20 个物体点、沿手指长度至少覆盖 1.5 cm |
| 排序 | 同上 | 第 0 层：接近方向最多向下 25°、与机器人朝向夹角 ≤ 75°、两指水平（夹紧轴与水平夹角 ≤ 30°），就是 bottle 专家标定过的姿态；第 1 层：向下 ≤ 45°、两指水平；第 2 层：其余。每层内按 graspnet 分数 |
| 我们的规划 | `GraspPlanned._plan_ik` | bottle 专家的分段手路线：在身体旁抬手，越过桌面到预抓（抓取点后退 10 cm）上方，下到预抓，直线进入；ArmIK（右臂 + 腰）每 1 cm 解一次，带一个很弱的中立姿态先验（0.01）。按顺序取第一个能解、手不碰桌和物体的候选 |
| SIMPLE 的规划 | `curobo_server.py` | SIMPLE `MotionPlannerAgent` 的做法：对一层候选的预抓位姿做一次 batched MotionGen（不做 finetune，12 个 trajopt 种子，10 次尝试），取最短的成功轨迹；桌面是长方体、物体是 CoACD 网格；机器人是 stock G1 + Dex1 的右臂 + 腰（`curobo_robot_right.py`，346 个内切球，FK 与 MuJoCo 逐关节一致）；关节限位与 IK 规划相同；retract 配置取开局姿态 |
| 执行（两边相同） | `GraspPlanned.run`、`_grasp_like_the_expert` | 规划器给的关节路线开环流到预抓（0.6 rad/s）；之后是 bottle 专家自己的闭环（SupportedMotion）：settle → 沿接近方向进入（物体位置跟踪、两个横向轴的修正、对齐闸门、失败退一次重进）→ 握住测量位置合爪 → 直线抬 10 cm（反馈保持开）→ 停 1.5 s。抬起 ≥ 5 cm 且两指垫都接触算成功 |
| 跑法 | `run_grasp.py`、`run_one.sh`、`run_batch.sh`、`servers.sh` | 每个物体一个进程（模块导入时会复制场景路径），`plan_episode.py` 起栈、出生，第 0 局 IK、第 1 局 CuRobo，物体朝向两局相同 |
| 固定抓取（2026-10-09） | `grasp_planned.py` `fixed=True`，`run_grasp.py --fixed-grasp` | 第 0 局只感知一次，对排名前 40 的候选，我们的 IK 逐个规划、CuRobo 一次批量规划，取排序里第一个两边都到得了的候选（没有就取第一个任一边到得了的，记 `common: false`），连同站位写进 `grasp_planned/target.json`。第 0 局用 IK 执行到它；第 1 局把机器人放到同一站位，不再感知，按物体位姿的变化搬过来同一个抓取，用 CuRobo 规划过去 |
| 视频 | `make_videos.py` | 两局的仿真录像左右拼接（短的一边停在最后一帧），标注规划器和结果，用 `locomani.pipeline.video_preview` 压到 500 KB 以内 |
| 品红 ghost（事后补画） | `ghost_overlay.py` | 跑的时候 `SIM_VIS=none`，`set_ghost` 放好的 ghost 没画进录像。不重跑：录像用的是跟踪骨盆子树质心的相机（距离 2.5 m、方位 135°、俯仰 -20°），每帧只差一个 lookat 点；拿去掉机器人的静态场景渲染和每帧做配准（金字塔 Gauss-Newton，Tukey 权重剔除机器人和影子）求出这个点，再在 `record.json` 的 grasp 位姿（腕 = tool − R·OFFSET）渲 ghost，以 0.6 不透明度叠上去，不盖 ego/腕部小窗；右上角加一个 2.5 倍放大框，中心是 ghost 的夹持中心。对齐站位那次瞬移取开头 40 帧里帧间变化最大的一帧（130 局都在第 4–5 帧，变化 ≥ 10.2 灰度，其余帧 ≤ 7.1；靠配准失败来找，有 5 局会滑过去），ghost 从瞬移后一帧开始显示（规划期间已经在）。没选出抓取的局不画。输出 `ghost_rounds/obj_XXX/round_NN.mp4`，`make_videos.py` 优先用它 |

## 3. 和 bottle 专家不同的地方

- **机器人站位**：出生在规划站位后 0.5 m，接管手臂（POSE 流）后用仿真的 `align_robot_xy` 把骨盆放到物体后方 0.38 m、左侧 0.12 m。规划器 IDLE 时机器人会漂 0.1–0.3 m。
- **进入速度**：0.06 m/s（专家 0.12）。专家是在 4.5 cm 宽的瓶子上调的，这里的物体更宽，用原速度手指会把物体撞倒。
- **抬起**：直线上抬、反馈保持开、不回正躯干。按专家原来的抬法（冻结反馈、按躯干坐标保持、躯干回正），手会比指令低 12–18 cm，已经抬起的瓶子被放回桌上。
- **对齐和重进**：两个横向轴都修正。专家只修夹紧轴，但俯视的抓取掌心 z 轴是水平的。

和 SIMPLE 原版的差别（2026-10-09 读 `third_party/SIMPLE/src/simple/agents/mp.py`、`mp/curobo.py`、`cli/datagen.py` 核对）：
- 选轨迹：SIMPLE 的 `batch_plan_for_approach` 把候选按 `plan_per_traj`（默认 4）分组、每组取最短的成功轨迹，`MotionPlannerAgent` 再取列表最后一个（"only takes the last successful one"），也就是排序最靠后、有成功的那一组里最短的。我们的 `curobo_server.py` 取的是全体最短，所以 10-08 那轮的"SIMPLE 选法"并不完全照原版；固定抓取那轮（第 5b 节）不受影响。
- 候选数：`simple datagen` 默认 `plan_batch_size=1`，只规划过滤后的第一个抓取，根本不做选择；例子脚本才用 40。
- 起点：SIMPLE 从机器人配置里的初始关节规划（`planning_init_joint_states`），整局在开始前一次规划完、开环执行；我们从当前关节规划，预抓之后换成我们的闭环专家。
- 抓取来源：SIMPLE 读按物体稳定姿态预先算好的 GSNet 抓取（网格上的真值），只留接近方向与竖直向下夹角 < 20° 的俯抓；我们用 graspnet-baseline 对第一视角深度在线预测。

## 4. 试过但没用的做法

- 规划完开环执行规划器自己的进入和抬起：手离抓取点 4–8 cm，合爪抓空。
- 自己写的积分伺服：抬起时手按约 4 Hz 振荡，把罐子甩下桌。
- 按 IDLE 等规划：CuRobo 规划 20 s 期间机器人漂了 7 cm，抓取偏 8 cm。
- 采用 graspnet 给的任意姿态（俯视、倾斜）：手腕扭到限位附近，合爪时手偏离指令 5–20 cm。
- 姿态先验 0.1 拉向开局姿态：所有候选在抬手段就差 4–5 cm，IK 一个都解不出。
- CuRobo 不设 retract 配置：腰停在三个限位上，SONIC 把手送偏 13 cm。

## 5. 结果（2026-10-08，65 个物体，每个规划器每个物体一局）

| | 我们的 IK 规划 | SIMPLE 的 CuRobo 规划 |
|---|---|---|
| 抬起 | 16/65 | 15/65 |
| 规划耗时中位数（最长） | 1.5 s（33 s） | 17 s（82 s） |
| 找不到路线 | 5 | 9 |
| graspnet 没有放得进 Dex1 的候选 | 4 | 3 |
| 合爪后两指垫没同时接触 | 15 | 19 |
| 抬起时脱手 | 13 | 9 |
| 进入或重进时把物体碰倒 | 12 | 8 |

- 两边都抬起 7 个，只有 IK 抬起 9 个，只有 CuRobo 抬起 8 个，至少一边抬起 24 个，两边都没抬起 41 个。每个物体每边只有一局，两个规划器之间看不出差别（16 对 15）。
- 两边的主要失败都发生在执行阶段（合爪没夹住、抬起时滑掉、进入时碰倒），不在规划阶段：规划失败只占 IK 的 5 次、CuRobo 的 9 次。
- 合爪没同时接触的那组物体偏轻偏矮（质量中位 0.04 kg、高 6.1 cm）；抬起时脱手的那组偏重（中位 0.14 kg、高 10.6 cm）。成功的局里第 0 层（水平侧抓）占一半。
- CuRobo 每次规划约 17 s，期间由 POSE 流原地保持机器人。
- 这次按"每个物体一局"出视频，成功率只是描述，不是可比较的统计量。要比较两个规划器，每个物体至少要多跑几个朝向。
- 这一轮 CuRobo 的 9 次"找不到路线"里有 2 次是程序错误（物体 053、060）：一层只有一个候选、而且 MotionGen 成功时，`curobo_server.py` 取轨迹出错（`JointState does not have horizon`），被记成没有路线。2026-10-09 修了；一层一个候选但规划失败的（011、043）不受影响。
- 两边抓的不是同一个抓取（2026-10-09 加 ghost 后看出来的），所以这也不是只换规划器的控制变量对比。原因有两个。一是每局重新感知：物体位置两局相同（中位差 0.00 cm），但对齐后的站位差 3.2 cm（最大 5.6 cm），深度点云不同，graspnet 的输出两局都不一样（65/65），候选列表只有 7/65 数量相同。二是选法不同：IK 取排序里第一个能解的（中位第 1 个），CuRobo 对一整层做批量规划、取最短轨迹（中位第 3 个），两边同层只有 32/52。结果两边都选出了抓取的 52 个物体里，没有一个选的是同一个抓取（位置差 < 1 cm 且角度差 < 10°），抓取点中位差 1.9 cm、姿态中位差 61°。

产物：
- 对比视频 `outputs/graspnet_bottle_20261008/videos/obj_XXX_ik_vs_curobo.mp4`，65 个物体 66 个文件（物体 019 拆成两段），每个 ≤ 500 KB，已逐个解码检查；全分辨率原片在 `videos_full/`（778 MB）。2026-10-09 起带品红 ghost（`ghost_overlay.py`），每局的配准记录在 `ghost_rounds/obj_XXX/round_NN.json`。
- 结果表 `summary.md` / `summary.json`；每局的点云、graspnet 抓取、规划路线和执行轨迹在 `outputs/runs/gnb_obj_XXX/grasp_planned/round_*`。
- 早期调试的运行留在 `outputs/runs/gnb_obj_*_smoke*` 和 `gnb_obj_*_batch1`（第一轮批跑，代码修正前），没有删。

## 5b. 固定抓取结果（2026-10-09，65 个物体，两边抓同一个抓取）

用户 2026-10-09 定：固定同一个抓取，两边都去抓它。做法见第 2 节"固定抓取"一行。产物在 `outputs/graspnet_bottle_20261008/fixed/`（`videos/` 65 个 ≤ 500 KB、`summary.md`、`analysis.json`），运行在 `outputs/runs/gnb_obj_XXX_fixed`。

| | 我们的 IK 规划 | SIMPLE 的 CuRobo 规划 |
|---|---|---|
| 抬起（65 个物体） | 11 | 11 |
| 抬起（两边都到得了共享抓取的 54 个） | 11 | 11 |
| 开环到预抓的误差，中位（p90），51 对 | 3.3 cm（5.4） | 4.1 cm（13.6） |
| 预抓误差 > 8 cm | 1 | 11 |
| 到预抓的时间，中位 | 12.0 s | 3.9 s |
| 选抓取时规划前 40 个候选，中位 | 10.7 s（逐个） | 14.4 s（一次批量） |

- 共享抓取：65 个里 58 个选出了，54 个两边都到得了，4 个只有一边到得了，7 个没有（4 个 graspnet 没有放得进 Dex1 的候选，3 个两边都到不了）。共享抓取多数排在前面（第 0 名 19 个、第 1 名 7 个、第 2 名 12 个）。每个物体能到的候选：IK 中位 8 个，CuRobo 12 个。
- 两局目标核对：两边执行的抓取最大差 1.6 mm、0.4°，确实是同一个。
- 抬起：两边都抬起 5 个，只有 IK 6 个，只有 CuRobo 6 个，都没抬起 48 个（精确 McNemar p = 1.00）。两个规划器在抬起上看不出差别。
- 唯一有显著差别的是开环到预抓的误差：51 对里 IK 有 33 对更小，Wilcoxon p = 0.002；CuRobo 有 11 个物体超过 8 cm（最大 22.6 cm，物体 041），IK 只有 1 个。CuRobo 的路线更短更快（3.9 s 对 12.0 s，两边按同一个 0.6 rad/s 重新定时）。误差大的原因没查：路线短、SONIC 跟踪滞后、或者腰的姿态，都只是猜测。之后的闭环会把大部分误差补回来，所以没有变成抬起的差别。
- 失败（两边都到得了的 54 个）：IK 合爪没同时接触 21、进入时碰倒 18、抬起时脱手 3、进入超时 1；CuRobo 碰倒 17、合爪没同时接触 14、脱手 8、第 1 局规划不到共享抓取 3、进入超时 1。
- **还剩一个混杂：朝向。** 第 1 局（CuRobo）出生后的朝向在 58 个物体里全部比第 0 局（IK）多顺时针转一点，中位 4.8°，最大 9°。规划器和局序绑定（IK 总在第 0 局），所以这个差是系统性的。站位按各自朝向算，物体在骨盆坐标系里的位置两局只差中位 8 mm（最大 45 mm，物体 019），但抓取姿态相对身体系统性差约 5°。CuRobo 第 1 局"规划不到共享抓取"的 3 个可能和它有关。要去掉它，可以让一半物体先跑 CuRobo（局序抵消）。
- 运行条件：两条通道，仿真时间/墙钟中位 0.80（最低 0.74），10-08 那轮是 0.83。另一个项目的 Isaac 同时在跑，第三条通道会把 CPU 配额压满（10 s 里被节流 5.5 s），所以只用了两条。
- 物体 000–005、007–013 最初是按"第 1 局站到第 0 局的同一个世界位置"跑的；朝向不同，物体在骨盆坐标系里会差到约 4 cm，于是改回按各自朝向站位并重跑了这 13 个。旧的留在 `outputs/runs/gnb_obj_XXX_fixed_oldstance`，没删。

## 6. 复现

```bash
bash scripts/experiments/graspnet_bottle_20261008/servers.sh start          # GPU 1：graspnet :5790、CuRobo :5791（GPU=0 放 0 号卡，10-09 那次在 0 号卡）
bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 4 1 0        # 4 条通道，实例 1–4，GPU 0
MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
    scripts/experiments/graspnet_bottle_20261008/ghost_overlay.py outputs/graspnet_bottle_20261008 --jobs 8   # 配准约 1 分钟/局；只改画法时加 --reuse 读已存的相机
PYTHONPATH=src python3 scripts/experiments/graspnet_bottle_20261008/make_videos.py outputs/graspnet_bottle_20261008
# 固定抓取对比（2026-10-09）：运行名 gnb_obj_XXX_fixed，产物在 outputs/graspnet_bottle_20261008/fixed/
EGL_GPU=0 bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 2 1 0 _fixed --fixed-grasp
MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
    scripts/experiments/graspnet_bottle_20261008/ghost_overlay.py outputs/graspnet_bottle_20261008 --jobs 8 --tag _fixed
PYTHONPATH=src python3 scripts/experiments/graspnet_bottle_20261008/make_videos.py outputs/graspnet_bottle_20261008 --tag _fixed
bash scripts/experiments/graspnet_bottle_20261008/servers.sh stop
```

2026-10-12 起 `run_grasp.py` 的默认改成离线抓取缓存、只用 IK、每个物体一局（[任意物体抓取](../object_grasp.md)）。照上面复现这一轮，`run_batch.sh` 要设 `ROUNDS=2`，并在末尾加 `--grasp-source graspnet --planners ik,curobo --rounds 2`；固定抓取那一批同样加这些参数。graspnet-baseline 的安装现在由 `install_graspnet.sh` 完成。

一次性准备（已做过）：`make_scenes.py`、`hand_cloud.py`、`export_geometry.py` + `curobo_robot_right.py`；graspnet-baseline 克隆在 `third_party/graspnet-baseline`，权重 `logs/checkpoint-rs.tar`（非商业研究许可，不进 git）。
