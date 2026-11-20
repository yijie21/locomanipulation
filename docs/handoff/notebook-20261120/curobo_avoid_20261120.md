# curobo_avoid_20261120：机器人自己那一眼的点云 → cuRobo 避障路线，解决 TabletopGrasp 的"手被桌沿挡住"

- 日期：2026-11-20
- 机器 / GPU：4090（`ssh -p 27984 root@14.103.60.188`）。感知分析在 `.venv_sim`；cuRobo 在 SIMPLE 的 venv（`tools/simple_isaac/SIMPLE/.venv`，cuRobo 已为 sm_89 编译）。
- 代码版本：sonic-selfmodel @ 4344165c，新脚本先放在 scratchpad 和 4090 的 `loco_outputs/simple_bench_20261119/curobo/` 下，定型后再进仓库。
- 配置 / EXP：
  - SIMPLE TabletopGraspMP 的 30 局，感知栏，Isaac 那一眼（`look_dump_eyes`、`cloud_eyes_final`）；
  - cuRobo 用 SIMPLE 自带的 `data/robots/g1/curobo/g1_29dof_with_dex3.yml`（基座 pelvis，末端 right_hand_palm_link）。
- 关联：
  - `simple_bench_eyes_20261120`（追查：失败是手被桌沿挂住，不是手腕姿态）；
  - 录像 `outputs/simple_eval_20261112/tabletop_videos_20261120/`；
  - beliefs B43。

## 假设

- **H1：**TabletopGrasp 里"预抓取点差 20 cm 以上"的失败，原因是命令直接让手走直线，穿过了桌子前沿，不是控制器的跟踪偏差。所以只要给一条由感知算出、绕开桌沿的路线，大部分这类失败会消失。
- **H2：**头部相机那一眼的点云，在用机器人自己的模型去掉身体之后，能把桌面和桌沿定位到几厘米以内。不能再用"骨盆前 20 cm 以内都算自己身体"（`blind.py` 的 `OWN = 0.2`）：真实桌沿只在骨盆前约 14 cm，这一刀会把桌沿切掉。这正是之前桌沿被估远 7.6 cm（中位数）的原因。

## 预测（启动前写，首次启动后锁定）

### 第 1 步：点云 → 障碍物

用机器人在看的那一刻的关节状态和自己的模型，从相机视角算出哪些像素是自己的身体并去掉，再去掉目标罐子，剩下的点裁到手臂工作空间、做体素化。拿仿真真值（桌子方块、真实深度）对照：

- **桌面高度误差**（感知的桌面点中位高度减真实桌面 0.785 m），30 局 |误差| 的中位数在 0.3–1.5 cm。
  - 理由：目标上的深度误差中位数 0.71 cm；第 0 局 `support_of` 估的桌面是 0.7935，差 0.85 cm。
- **桌沿看不看得到**：在目标前方 ±30 cm 的横向带里，离真实桌沿 3 cm 以内至少有 50 个桌面高度的点，算看到。预测 20–30 / 30 局看到。
  - 理由：按相机位置（骨盆前约 8 cm、高 1.26 m、向下约 51°、竖直视场 77.5°），桌沿在画面下沿附近，约 83° 的俯角，在视场里但很靠边；前臂和手可能挡住一部分。
- **桌沿位置误差**（看到的那些局，感知的近沿减真实桌沿）：|误差| 的中位数 0.5–3 cm，最大不超过 8 cm。

### 第 2 步：离线让 cuRobo 规划

- **设定：**用重放录像里伸手开始那一刻的真实状态作起点，规划到每局所选抓取的预抓取点。障碍物是第 1 步的体素，碰撞激活距离 3 cm，目标罐子不算障碍物。
- 29 局（有抓取的）里有 25–29 局能规划出路线。
- 规划出的路线中，手、手腕、前臂到真实桌子的最小距离在 ≥ 90 % 的局里至少 2 cm。
- 单次规划用时中位数不超过 1 s。

### 第 3 步：接进评测，重跑

- **方案：**cuRobo 路线取几个手腕路点，direct 和 model 用同样的中间路点；最后的预抓取和伸入命令照旧（direct 原样发，model 用自身模型修正）。
- **TabletopGrasp：**
  - 预抓取差 20 cm 以上的局，从 18（direct）/ 19（model）降到每种方法 ≤ 4 局；
  - 45 s 内成功从 2 / 3 升到每种方法 4–12 局。上限受"合手时推开罐子"的失败（这次是 9 / 6 局）和加了路点后多花的时间限制。
  - 伸手过程中机器人碰到桌子的局，每种方法 ≤ 6 / 29。
- **实际路线和规划的偏差**（手，整条路上的最大值）：中位数 2–6 cm。
- **三个盒子任务**（同样加 cuRobo 路点重跑）：model 比 direct 的伸入误差优势仍在，逐局配对差的中位数每个任务都 ≤ −1.0 cm。原来是 −2.02 / −2.36 / −1.58 cm。

## 判据（什么结果会让我放弃或修改假设）

- **第 1 步：**
  - 桌沿在 > 10 局看不到，或看到的局误差中位数 > 3 cm：一眼的感知不够。先改感知（比如换个时机多看一眼，或往后站一点再看），再接 cuRobo。
  - 桌面高度误差中位数 > 2 cm：查深度。
- **第 2 步：**cuRobo 失败 > 4 局：先查原因（起点就在碰撞里、身体没去干净、余量太大），不进评测。
- **第 3 步：**
  - 规划的路线是无碰撞的，但被挡住的局仍 > 8 局：说明控制器实际走的和规划差太多，H1 的后半句错了。这正是"考虑自身模型的规划"的证据，转去查新。
  - 被挡住的局 ≤ 4：cuRobo 加感知够用，路径版自身模型搁置。
  - 盒子任务上 model 的优势缩到 > −1.0 cm：路点干扰了自身模型的作用，要查。

## 预测修订

## 结果（贴原始输出，不转述）

### 第 1 步（2026-11-20）

**怎么跑的：**`curobo/obstacles.py`，每局一个进程，最后用 `summary` 汇总，结果在 4090 的 `loco_outputs/simple_bench_20261119/curobo/step1/`（`step1.json`、`row_*.json`、`obst_*.npz`、`fig_bad.png`）。

**中途改过的两处：**
- 分割渲染关掉多重采样（`offsamples 0`）。不关的话，混色的边缘像素会给出不存在的几何体编号（index 320 of 259）。
- 按桌面高度挑桌面点时，不用 `support_of` 的估计（第 1 局高了 2.76 cm，筛空了），改用障碍物点在 0.5 cm 分箱里最密的高度。

第 12 局没有抓取，也就没有录像。它的姿态借用第 10 局第一帧，按起点差平移旋转过去；相机位置因此差 3.45 cm，其余各局是 0.16–0.40 cm。

汇总原始输出（`step1.json` 的 summary）：

```
"episodes": 30, "seen": 26,
"top_px_cm": {"n": 30, "median_abs": 1.15, "median": 1.15, "min": -0.09, "max": 2.86},
"top_est_cm": {"n": 30, "median_abs": 1.95, "median": 1.95, "min": 0.55, "max": 4.88},
"level_cm": {"n": 30, "median_abs": 1.23, "median": 1.23, "min": -0.69, "max": 4.14},
"edge_cm_seen": {"n": 26, "median_abs": 0.19, "median": -0.15, "min": -0.71, "max": 2.18},
"edge_cm_all": {"n": 30, "median_abs": 0.2, "median": -0.13, "min": -0.71, "max": 18.53},
"edge_own_cm": {"n": 30, "median_abs": 7.61, "median": 7.61, "min": 3.74, "max": 23.97},
"cam_dp_cm": {"n": 30, "median_abs": 0.2, "median": 0.2, "min": 0.16, "max": 3.45},
"vox_prec": {"n": 30, "median_abs": 0.88, "median": 0.88, "min": 0.66, "max": 0.93},
"vox_recall_zone": {"n": 30, "median_abs": 0.98, "median": 0.98, "min": 0.8, "max": 1.0},
"edge_px_inside": median 61 (56..61), "edge_px_table": median 61 (51..61), "edge_px_robot": 0 (0..0)
```

没看到桌沿的 4 局（原始行摘录）：

```
1 ... top_px 0.97 top_est 2.76 level 1.07 edge 9.36 n 0 own 20.98 ... prec 0.821 recall 1.0
6 ... top_px 0.91 top_est 2.37 level 2.03 edge 8.17 n 9 own 11.18 ... prec 0.899 recall 0.979
17 ... top_px 0.52 top_est 3.75 level 2.47 edge 18.53 n 0 own 23.97 ... prec 0.744 recall 0.944
24 ... off [-2.17, -0.02, 2.89] top_px 2.86 top_est 4.88 level 4.14 edge 10.69 n 0 own 11.03 ... prec 0.66 recall 0.804
```

`fig_bad.png`（第 0、1、6、17、24 局）显示：
- 第 1、6、17 局是亮面反光的桌子，第 24 局像玻璃。
- 在这几局里，靠近桌沿的桌面深度斜着往下塌了几厘米，所以按桌面高度 ±1.2 cm 挑不出点。但桌沿处的点还在，桌沿附近体素的覆盖率仍有 0.80–1.0。
- 每局在桌沿正下方都有一条竖直的"飞点"，从桌面一直到 z ≈ 0.3 m，是深度不连续处被填补出来的。场景里的桌子是 10 cm 厚的板（`table_size` z 0.1，底面 0.685 m），下面没有几何体。
- 真实桌沿在这 30 局里都在画面内，也都没被机器人身体挡住（`edge_px_robot` 全是 0）。

**第 2 步时发现的问题（2026-11-20 补）：**`look_dump_eyes` 里的 `mask` 和 `cloud` 在 17 / 30 局（1、2、4–9、12、14–18、22、23、27）不是罐子，而是 60 × 60 cm 的一片桌面。评测实际选抓取用的 `cloud_eyes_final` 在这些局是对的；只有第 24 局两者都偏，`cloud_eyes_final` 落在 x −0.356，真实目标在 −0.67 附近。

所以第 1 步"去掉目标"在这 17 局去掉的是一片远处桌面，罐子留在了障碍物里。对桌沿和桌面的数字影响很小：
- 被去掉的那片桌面在桌沿后面 10 cm 以上，桌沿附近的点还在；
- `y_t`（横向带中心）用的是错的 cloud 中位数，但桌子是直的、轴对齐，桌沿 x 不随 y 变。

`vox_prec` 里把罐子算成了"不在真实桌子附近"的体素，所以偏低。第 2 步改用 `cloud_eyes_final` 在三维里去掉目标。

原始对比（每局 dump 的 cloud 和 final 的中位位置、范围）：

```
 1 dump n  4062 med [0.35  0.08  0.827] span [0.6  0.6  0.03] | final n  3057 med [-0.685  0.028  0.863] span [0.06 0.22 0.05]
 2 dump n 22343 med [-0.51   0.214  0.808] span [0.6  0.58 0.09] | final n  3356 med [-0.682  0.003  0.864] span [0.07 0.22 0.04]
 0 dump n  3268 med [-0.68   0.029  0.866] span [0.09 0.23 0.08] | final n  3140 med [-0.681  0.029  0.866] span [0.06 0.22 0.05]
24 dump n 14233 med [-0.671  0.058  0.849] span [0.3  0.57 0.07] | final n   544 med [-0.356  0.078  0.854] span [0.16 0.14 0.04]
```

（其余各局见对话记录；同样格式，dump 的 span 为 0.5–0.6 m 的就是错的那 17 局。）

### 第 2 步（2026-11-20）

**怎么跑的：**
- `curobo/prep2.py`（`.venv_sim`）：用 `--trace` 重放 direct 的 30 局，`rows_eyes` 的结果 30/30 一致。从录像取"预抓取"阶段的第一帧作起点，从 trace 取想要的预抓取位姿。
- `curobo/plan2.py`（SIMPLE venv，cuRobo）：每局一个进程，4 张卡、8 路并行。
- `curobo/check2.py`（`.venv_sim`）：在仿真模型自己的几何上复查规划出的路线。

结果都在 `step2/`：`prep_*.npz`、`plan_*.npz`、`plan_rows.jsonl`、`check_rows.jsonl`、`fig_side.png`。

**cuRobo 的设定：**
- 机器人用 SIMPLE 自带的 `g1_29dof_with_dex3.yml`，改了这些：
  - 末端改成 `right_wrist_yaw_link`，评测命令的就是它；
  - 只放开右臂 7 个关节，腰和右手手指锁在起点读数；
  - 只有右臂和右手参与碰撞。
- 激活距离 3 cm。障碍物是第 1 步每个 2 cm 体素当一个方块（PRIMITIVE），坐标系是起点的骨盆系。

**中途改动和原因：**
1. SIMPLE 改过的 cuRobo 只按 `link_names` 建树，自碰撞掩码也按 `link_names` 的长度分配。所以所有碰撞部件和被锁关节的父、子 link 都要先列进 `link_names`，否则分别报 KeyError 和 IndexError 14 of 14。
2. **起点就在碰撞里**（ep 0 报 `INVALID_START_STATE_WORLD_COLLISION`）：
   - 手指离真实桌子 0.25 cm，压着体素 0.54 cm；食指尖在 x −0.877、z 0.631，伸进了桌板下面。
   - 改法：去掉和起点姿态的球距离 < 1 cm 的体素（`--free 0.01`），每局去掉 1–18 个。
3. **目标不算障碍物**：用 `cloud_eyes_final` 在三维里去掉 2 cm 内的体素（`--tgt 0.02`），原因见第 1 步的补充。
   - 没加这一步时（v1，结果移到 `step2_v1/`）是 26/29。ep 1 的预抓取位姿上，拇指和罐子的体素重叠 0.32 cm（罐子顶在 0.875 m），报 IK_FAIL。
4. 用时取同一进程里第二次规划的时间，第一次含编译。

v1（没有在三维里去掉目标）的汇总：

```
planned 26 of 29
t_plan median 2.56 max 4.91
clear_after median 3.315 min 3.01 >=2cm 26 of 26
start_table median 0.11 min -0.17
失败：1 IK_FAIL（目标位姿压在罐子体素上），6 FINETUNE_TRAJOPT_FAIL，24 IK_FAIL
```

v2（定稿：`--free 0.01 --tgt 0.02 --cloud-dir cloud_eyes_final --act 0.03`）的原始输出：

```
0 ok None drop 8 tgt 0 start_tab 0.25 t 2.69 dur 1.2 after 3.03 clear_vox 1.57 end_mm 0.0 jaw_max_z 0.987
1 ok None drop 129 tgt 116 start_tab 0.0 t 3.08 dur 1.76 after 3.7 clear_vox 1.12 end_mm 0.0 jaw_max_z 0.971
2 ok None drop 12 tgt 0 start_tab 1.26 t 2.38 dur 1.44 after 3.19 clear_vox 1.29 end_mm 0.0 jaw_max_z 0.983
3 ok None drop 11 tgt 2 start_tab -0.01 t 3.1 dur 1.08 after 3.29 clear_vox 1.91 end_mm 0.0 jaw_max_z 0.981
4 ok None drop 132 tgt 114 start_tab 0.05 t 2.65 dur 1.14 after 3.03 clear_vox 1.66 end_mm 0.0 jaw_max_z 1.0
5 ok None drop 214 tgt 213 start_tab 0.47 t 3.29 dur 1.16 after 3.19 clear_vox 2.19 end_mm 0.0 jaw_max_z 0.962
6 FAIL MotionGenStatus.FINETUNE_TRAJOPT_FAIL drop 203 tgt 199 start_tab 1.13 t 4.81 empty None goal_vox 5.66 retry MotionGenStatus.FINETUNE_TRAJOPT_FAIL from_pelvis 35.1
7 ok None drop 134 tgt 127 start_tab 1.43 t 3.16 dur 1.66 after 3.46 clear_vox 1.38 end_mm 0.0 jaw_max_z 0.986
8 ok None drop 134 tgt 126 start_tab 0.48 t 2.48 dur 1.16 after 3.29 clear_vox 1.64 end_mm 0.0 jaw_max_z 0.972
9 ok None drop 116 tgt 108 start_tab -0.01 t 3.32 dur 1.16 after 3.86 clear_vox 1.23 end_mm 0.0 jaw_max_z 0.994
10 ok None drop 33 tgt 20 start_tab 0.25 t 2.88 dur 1.16 after 3.47 clear_vox 1.57 end_mm 0.0 jaw_max_z 0.984
11 ok None drop 24 tgt 8 start_tab -0.1 t 2.31 dur 1.32 after 3.13 clear_vox 1.4 end_mm 0.0 jaw_max_z 0.988
12 skip no prep (no grasp)
13 ok None drop 34 tgt 24 start_tab 0.05 t 2.43 dur 1.08 after 3.16 clear_vox 1.73 end_mm 0.0 jaw_max_z 0.982
14 ok None drop 120 tgt 115 start_tab 0.01 t 2.93 dur 1.14 after 3.07 clear_vox 0.72 end_mm 0.0 jaw_max_z 0.969
15 ok None drop 118 tgt 109 start_tab 0.41 t 2.33 dur 1.14 after 3.3 clear_vox 1.58 end_mm 0.0 jaw_max_z 0.991
16 ok None drop 121 tgt 116 start_tab 1.27 t 2.29 dur 1.26 after 3.09 clear_vox 1.3 end_mm 0.0 jaw_max_z 0.99
17 ok None drop 125 tgt 122 start_tab 1.51 t 2.26 dur 1.32 after 3.18 clear_vox 1.62 end_mm 0.0 jaw_max_z 0.948
18 ok None drop 139 tgt 122 start_tab -0.13 t 2.47 dur 1.16 after 3.7 clear_vox 1.44 end_mm 0.0 jaw_max_z 0.99
19 ok None drop 7 tgt 1 start_tab -0.05 t 2.33 dur 1.24 after 3.84 clear_vox 1.39 end_mm 0.0 jaw_max_z 0.969
20 ok MotionGenStatus.GRAPH_FAIL drop 7 tgt 3 start_tab 0.07 t 1.61 dur 1.36 after 3.37 clear_vox 1.62 end_mm 0.0 jaw_max_z 0.988
21 ok None drop 8 tgt 2 start_tab 0.21 t 2.93 dur 1.16 after 3.1 clear_vox 1.47 end_mm 0.0 jaw_max_z 0.971
22 ok None drop 118 tgt 110 start_tab 1.31 t 3.45 dur 1.3 after 3.11 clear_vox 1.69 end_mm 0.0 jaw_max_z 0.991
23 ok None drop 118 tgt 113 start_tab -0.15 t 2.56 dur 1.12 after 3.23 clear_vox 1.57 end_mm 0.0 jaw_max_z 0.994
24 FAIL MotionGenStatus.IK_FAIL drop 100 tgt 86 start_tab -0.15 t 0.63 empty MotionGenStatus.IK_FAIL goal_vox None retry MotionGenStatus.IK_FAIL from_pelvis 68.6
25 ok None drop 22 tgt 13 start_tab -0.17 t 2.27 dur 1.1 after 3.59 clear_vox 1.43 end_mm 0.0 jaw_max_z 0.974
26 ok None drop 24 tgt 14 start_tab 1.33 t 2.2 dur 1.34 after 3.36 clear_vox 1.04 end_mm 0.0 jaw_max_z 0.965
27 ok None drop 152 tgt 137 start_tab 0.15 t 2.17 dur 1.1 after 3.36 clear_vox 1.92 end_mm 0.0 jaw_max_z 0.989
28 ok None drop 30 tgt 14 start_tab 0.11 t 2.21 dur 1.12 after 3.09 clear_vox 1.85 end_mm 0.0 jaw_max_z 0.984
29 ok None drop 25 tgt 12 start_tab -0.11 t 2.19 dur 1.16 after 3.39 clear_vox 1.65 end_mm 0.0 jaw_max_z 0.993
planned 27 of 29 | retry ok 0
t_plan median 2.47 max 4.81
clear_after median 3.29 min 3.03 >=2cm 27 of 27
tgt dropped median 108 min 0 max 213
dur median 1.16 jaw_max_z median 0.984
```

- `start_tab` 和 `after`：cuRobo 的球到真实桌板的距离。`start_tab` 是起点；`after` 是离开起点（≥ 3 cm）之后的最小值。
- `end_mm`：终点手腕离目标的距离。
- `fk_mm` 和 `fk_deg` 每局都是 0.0：cuRobo 的运动学和仿真模型一致。

**两个失败：**
- **ep 24：**不放障碍物也 IK 失败。目标离骨盆 68.6 cm、在桌沿后 50 cm，direct 的预抓取误差 62 cm。目标认错了（见上），不是规划的问题。
- **ep 6：**不放障碍物能规划，目标位姿离体素 5.66 cm，但带障碍物时 12 次重试都是 FINETUNE_TRAJOPT_FAIL。
  - 激活距离改 2 cm 能规划（`try_act02/`，用时 2.6 s）。
  - 早先一次试跑 `--tgt` 时，因为 cloud 坐标用错，实际什么都没去掉，那次也成功了一次：说明 ep 6 处在 cuRobo 随机种子的边界上。

`fig_side.png`（ep 0、6、17、1 的侧视图）：
- 起点的手点在桌沿前 4–9 cm、桌面以下；
- direct 的手几乎竖直往上，停在桌面上方 6–10 cm（手指压在桌沿上）；
- cuRobo 的路线先往回撤 5–10 cm，再抬起越过桌沿。

**在仿真模型几何上的复查（`check2.py`）：**
- `mj_geomDistance`（mujoco 3.11）在这里会在 4–9 cm 的相邻帧之间给出假的 0（ep 0：`6.7, 0.0, 7.3`），改用几何表面点：网格顶点，基本体采样。
- ep 0 的原始输出：

```
{'ep': 0, 'T': 60, 'table_start_cm': 0.08, 'table_min_cm': 0.08, 'table_after_cm': 3.14, 'table_worst': [0.08, ['right_hand_index_0_link', 'table'], 0], 'self_start_cm': 4.73, 'self_min_cm': 3.33, 'self_worst': [3.33, ['right_hand_thumb_2_link', 'right_hip_roll_link'], 6], 'jaw_fk_mm_max': 0.0, 'n_mov': 12, 'n_rest': 38}
```

全部 27 条路线（加上 ep 6 的 2 cm 回退）在仿真几何上的复查（`step2/check_rows.jsonl`）：

```
0  table start 0.08 after 3.14 worst [0.08, ['right_hand_index_0_link', 'table'], 0] | self start 4.73 min 3.33 [3.33, ['right_hand_thumb_2_link', 'right_hip_roll_link'], 6] | jaw fk mm 0.0
1  table start 0.1 after 2.75 ... | self start 4.63 min 1.05 [1.05, ['right_hand_thumb_2_link', 'right_hip_roll_link'], 11] | jaw fk mm 0.0
6 act02 table start 0.01 after 2.2 ... | self start 6.87 min 6.87 ... | jaw fk mm 0.0
19  table start 0.26 after 3.06 ... | self start 2.75 min 1.06 [1.06, ['right_hand_thumb_2_link', 'right_hip_roll_link'], 6] | jaw fk mm 0.0
26  table start -0.01 after 3.19 ... | self start 8.22 min 6.86 [6.86, ['right_elbow_link', 'torso_link'], 63] | jaw fk mm 0.0
n 27 table_after median 3.19 min 2.63 >=2cm 27
self_min median 3.16 min 1.05 <1cm 0
table_start median 0.08 max 0.32
jaw fk max mm 0.0
```

**对照第 2 步的预测：**
- 能规划 25–29 / 29：✓（27；v1 是 26）。
- 离开出发点后到真实桌子 ≥ 2 cm 的局占 ≥ 90 %：✓。cuRobo 球 27/27，最小 3.03 cm；仿真几何 27/27，最小 2.63 cm。
- 规划用时中位数 ≤ 1 s：✗（2.47 s）。
- 规划路线离机器人自己身体的最小距离是 1.05 cm（拇指对右髋），没有接触。cuRobo 里只有右臂、右手参与碰撞，躯干和腿不在里面，所以这一点是复查出来的，不是规划保证的。

### 第 3 步：TabletopGrasp（2026-11-20）

**接法：**
- `motor.reach` 加了 `via_rot`，每个路点一个手腕姿态，按段转过去；不传时行为不变。
- `simple_bench.py` 加了 `--avoid DIR`：预抓取那一下读 `via_<NAME>.npz`，按路点走，最后一段仍去各方法自己的目标（direct 原样，model 用自身模型修正）。没有路线的局照旧走直线。
- `curobo/via2.py` 把每条 cuRobo 路线用 Douglas–Peucker（1 cm）压成 7–10 个路点，偏离规划 ≤ 0.99 cm。手点路线约 80 cm，直线约 35 cm。
- ep 6 用 2 cm 激活距离的那条路线（`try_act02`），这是预注册之外的回退。ep 24 没有路线，照旧走直线。

**direct 和 model 的出发状态（伸手开始时）：**30 局中 26 局在 float 精度内一致（≤ 1e-5）。ep 6 差 0.0013，ep 22 差 0.535（基座 3 cm，髋关节约 0.1 rad），ep 26 差 0.069，都是走路那一段分岔了：model 的走路不完全可复现。按预注册，model 仍用 direct 的路点。

**跑法：**4090，每局一个进程，`--trace`。
- direct 加 `--avoid`；model 加 `--avoid`；model 不加 `--avoid` 也重跑一遍拿 trace（作对照）。
- direct 不加 `--avoid` 的对照用的是第 2 步重放的 `trace_rows/`，和 `rows_eyes` 30/30 一致。
- 结果在 `step3_tabletop/rows/`，汇总 `step3_tabletop/summary.txt`（`curobo/step3.py`）。

```
direct_straight {"n": 30, "success_45": 2, "success_official": 2, "stuck": 18, "table": 29, "no_grasp": 1, "fell": 0, "why": {}, "knocked": {"None": 17, "close": 12}, "dev": {"n": 0}, "t_success": {"n": 2, "median": 13.17, "min": 13.14, "max": 13.2}, "pre_err": {"n": 29, "median": 27.32, "min": 2.89, "max": 61.97}, "entry_err": {"n": 29, "median": 27.73, "min": 1.72, "max": 53.09}, "n_via": {"n": 0}}
direct_avoid {"n": 30, "success_45": 12, "success_official": 5, "stuck": 1, "table": 28, "no_grasp": 1, "fell": 0, "why": {}, "knocked": {"close": 26, "None": 3}, "dev": {"n": 28, "median": 4.1, "min": 2.86, "max": 11.71}, "t_success": {"n": 12, "median": 16.17, "min": 15.48, "max": 18.24}, "pre_err": {"n": 29, "median": 3.37, "min": 1.55, "max": 61.97}, "entry_err": {"n": 29, "median": 2.33, "min": 1.23, "max": 53.09}, "n_via": {"n": 28, "median": 7.0, "min": 7, "max": 10}}
model_straight {"n": 30, "success_45": 3, "success_official": 3, "stuck": 19, "table": 29, "no_grasp": 1, "fell": 0, "why": {}, "knocked": {"None": 18, "close": 11}, "dev": {"n": 0}, "t_success": {"n": 3, "median": 13.34, "min": 13.3, "max": 13.54}, "pre_err": {"n": 29, "median": 27.62, "min": 2.7, "max": 59.96}, "entry_err": {"n": 29, "median": 27.55, "min": 0.36, "max": 55.32}, "n_via": {"n": 0}}
model_avoid {"n": 30, "success_45": 17, "success_official": 2, "stuck": 1, "table": 28, "no_grasp": 1, "fell": 0, "why": {}, "knocked": {"close": 28, "None": 1}, "dev": {"n": 28, "median": 6.77, "min": 2.52, "max": 17.05}, "t_success": {"n": 17, "median": 16.5, "min": 15.9, "max": 18.7}, "pre_err": {"n": 29, "median": 6.5, "min": 1.17, "max": 59.96}, "entry_err": {"n": 29, "median": 2.35, "min": 0.35, "max": 55.32}, "n_via": {"n": 28, "median": 7.0, "min": 7, "max": 10}}
paired model - direct entry cm (straight): n 29 median 0.37 mean 1.97
paired model - direct entry cm (avoid): n 29 median -0.34 mean -0.2
```

**各列的含义：**
- `stuck`：预抓取停在离目标 20 cm 以外。剩下的 1 局是 ep 24（目标认错了，没有路线）。
- `table`：手离开出发点 ≥ 3 cm 之后，右臂或右手还碰着桌子。
- `dev`：手点离规划路线（规划路线再接到想要的预抓取位姿）的最大距离。model 的最后一段本来就要去修正过的目标，所以它的 `dev` 含有意的偏移。
- `success_official`：SIMPLE 自己的 16 s 时限。

**每局逐条：**见 `summary.txt` 的 `--per` 部分。

**`table` 的接触在什么时候发生：**

```
== direct_avoid
0 0 ticks 424 hit ticks 41 frac first/last 0.31 0.42 first (133, 4.74, -0.081, 0.07, ['right_hand_index_1_link']) last (178, 5.64, -0.073, 0.09, ['right_hand_index_1_link'])
2 2 ticks 465 hit ticks 137 frac first/last 0.27 0.59 first (124, 4.52, -0.055, 0.031, ['right_hand_index_0_link']) last (276, 7.56, -0.051, 0.109, ['right_hand_index_1_link'])
0 3 ticks 413 hit ticks 71 frac first/last 0.25 0.44 first (103, 4.04, -0.077, 0.061, ['right_hand_index_1_link']) last (180, 5.58, -0.059, 0.109, ['right_hand_index_1_link'])
0 6 ticks 437 hit ticks 112 frac first/last 0.22 0.53 first (97, 3.78, -0.053, 0.035, ['right_wrist_yaw_link']) last (231, 6.46, -0.049, 0.098, ['right_hand_index_1_link'])
direct_avoid episodes with hits 28 with hits between 10% and 80% of the reach 28
== direct_straight
0 0 ticks 274 hit ticks 208 frac first/last 0.2 1.0 first (55, 3.18, -0.096, -0.034, ['right_hand_index_0_link']) last (273, 7.54, -0.075, 0.082, ['right_hand_index_0_link'])
2 2 ticks 290 hit ticks 202 frac first/last 0.26 1.0 first (74, 3.52, -0.049, -0.065, ['right_wrist_yaw_link']) last (289, 7.82, -0.058, 0.112, ['right_hand_index_1_link'])
0 3 ticks 275 hit ticks 87 frac first/last 0.21 0.57 first (58, 3.14, -0.081, -0.023, ['right_hand_index_0_link']) last (157, 5.12, -0.002, 0.159, ['right_hand_index_1_link'])
0 6 ticks 272 hit ticks 194 frac first/last 0.25 1.0 first (68, 3.2, -0.048, -0.061, ['right_wrist_yaw_link']) last (271, 7.26, -0.055, 0.102, ['right_hand_index_1_link'])
direct_straight episodes with hits 29 with hits between 10% and 80% of the reach 29
```

（每个元组：tick、时刻 s、手点 x 减桌沿、手点 z 减桌面、接触的部位。）

**对照预测：**
- 卡住：每种方法都 ≤ 4 ✓（1 / 1）。
- 45 s 成功在 4–12：direct 12 ✓（在上沿）；model 17 ✗（高于预测）。
- 伸手中碰桌 ≤ 6/29：✗（28 / 28）。
- 路线偏差中位数在 2–6 cm：direct 4.1 ✓；model 6.77 ✗（含有意的最后偏移）。

### 第 1–3 步：三个盒子任务（2026-11-20）

**跑法：**4090 上 tmux `curobo_box`，脚本 `curobo/run_box.sh`，日志 `box/logs/pipeline.log`。
- 先用 `--film --trace` 重放 direct 直线：90/90 和 `rows_eyes` 一致。
- `obstacles_box.py`：去机器人自身像素同上；目标用 `cloud_eyes_final` 在三维 2 cm 内去掉；只保留体素，不做桌沿量测。
- `prep2.py` 和 `plan2.py` 按任务参数化，设定同 TabletopGrasp。失败的局用 2 cm 激活距离重试。

```
[06:47:30] step 1: 90 rows, 90 obstacle files
[06:47:45] step 2a: 90 prep files
[06:51:57] step 2b: 90 rows, ok 7
[06:53:59] step 2b retry at 2 cm: 83 rows, ok 0
[06:54:00] step 2c: 7 via files
[06:55:42] DONE
XMovePickTeleop {'IK_FAIL | empty IK_FAIL': 23, 'ok': 7}
XMoveBendPickTeleop {'IK_FAIL | empty IK_FAIL': 30}
BendPickMP {'IK_FAIL | empty IK_FAIL': 30}
XMovePickTeleop_L0_e7 IK_FAIL start_vox 13.86 start_tab 15.78 goal_vox None goal_tab None above_top 17.58 from_pelvis 38.5 ...
BendPickMP_L1_e5 IK_FAIL start_vox 14.32 start_tab 23.69 goal_vox None goal_tab None above_top 32.84 from_pelvis 29.6 ...
```

**IK 失败的原因：**83 局失败全是 IK_FAIL，不放障碍物也一样（`empty IK_FAIL`）。预抓取位姿离骨盆 30–42 cm，在桌面上方 14–34 cm。腰锁成伸手开始时的直立姿态、只动右臂的话，够不到。评测里控制器的全身 IK 会动腰。所以这是"只放开右臂"的建模限制，不是障碍物的问题。这三个任务里直线伸手本来也碰不到桌子（direct 直线的 `table` 0/30）。

**第 3 步汇总（`step3.py`，每个任务一份）：**

```
== XMovePickTeleop
direct_straight s45 23 official 23 stuck 0 table 0 entry 2.86 n_via 0 knocked {'close': 23, 'pregrasp': 7}
direct_avoid s45 25 official 25 stuck 0 table 0 entry 2.84 n_via 7 knocked {'close': 25, 'pregrasp': 5}
model_straight s45 23 official 23 stuck 0 table 0 entry 0.78 n_via 0 knocked {'close': 23, 'pregrasp': 7}
model_avoid s45 25 official 25 stuck 0 table 0 entry 0.79 n_via 7 knocked {'close': 25, 'pregrasp': 5}
paired model - direct entry cm (straight): n 30 median -2.02 mean -2.04
paired model - direct entry cm (avoid): n 30 median -2.01 mean -2.02
== XMoveBendPickTeleop
direct_straight s45 25 official 25 stuck 0 table 0 entry 3.3 n_via 0 knocked {'close': 30}
direct_avoid s45 25 official 25 stuck 0 table 0 entry 3.3 n_via 0 knocked {'close': 30}
model_straight s45 30 official 30 stuck 0 table 0 entry 0.79 n_via 0 knocked {'close': 29, 'lift': 1}
model_avoid s45 30 official 30 stuck 0 table 0 entry 0.79 n_via 0 knocked {'close': 29, 'lift': 1}
paired model - direct entry cm (straight): n 30 median -2.36 mean -2.56
paired model - direct entry cm (avoid): n 30 median -2.36 mean -2.56
== BendPickMP
direct_straight s45 30 official 0 stuck 0 table 0 entry 2.29 n_via 0 knocked {'close': 30}
direct_avoid s45 30 official 0 stuck 0 table 0 entry 2.29 n_via 0 knocked {'close': 30}
model_straight s45 30 official 0 stuck 0 table 0 entry 0.72 n_via 0 knocked {'close': 30}
model_avoid s45 30 official 0 stuck 0 table 0 entry 0.72 n_via 0 knocked {'close': 30}
paired model - direct entry cm (straight): n 30 median -1.58 mean -1.59
paired model - direct entry cm (avoid): n 30 median -1.58 mean -1.59
```

**对照预测：**盒子任务上 model 的配对优势每个都 ≤ −1.0 cm，✓（−2.01 / −2.36 / −1.58）。但只有 XMovePick 的 7 局真的用了路点，其余 83 局因为没有路线，仍然走直线。所以这条判据实际只在 7 局上检验过，是弱证据。

XMovePick 那 7 局里，"预抓取途中把盒子碰动"从 7 局降到 5 局，两种方法的成功都 +2。

## 意外

- **第 1 步：**
  - 看到的局里桌沿误差中位数只有 0.19 cm，好于预测的 0.5–3 cm。
  - 老办法（骨盆前 20 cm 一刀切）的误差中位数 7.61 cm，正好复现了之前"桌沿估远 7.6 cm"，H2 说的原因成立。
  - 感知到的桌面整体高约 1.15 cm，同时 x 方向偏约 −0.5 cm，每局几乎一样。看起来是 Isaac 相机和仿真相机之间的固定错位（之前在图像上也量到约 3 像素），不是随机噪声。
  - 机器人自己的 `support_of` 桌面估计比点云本身还差（中位数 1.95 cm，最大 4.88 cm）。
  - 桌沿下方有一条深度填补出来的竖直"飞点"墙。
- **第 2 步：**
  - 伸手开始时，右手在**每一局**都已经贴着桌子：cuRobo 球到桌板中位数 0.11 cm，最小 −0.17 cm；在仿真几何上 ep 0 是 0.08 cm。食指尖伸在桌板下面，成功的局也一样。~~卡住和没卡住的差别不在出发姿态~~ → 复核更正：起点在桌沿前约 3.8 cm、桌面下约 9.2 cm 的 7 局，direct 直线全卡住；其余 22 局里卡住 11 局（Fisher p = 0.026）。出发越深越容易卡。
  - `look_dump_eyes` 的目标 mask 在 17/30 局是错的（一片桌面），第 1 步没发现。
  - 规划用时中位数 2.47 s，比预测的 ≤ 1 s 慢。约 2700 个方块、没开 CUDA graph、每卡两个进程。
  - ep 6 卡在 cuRobo 随机种子的边界上。
  - mujoco 3.11 的 `mj_geomDistance` 在网格对之间会给出假的 0。
- **第 3 步（Tabletop）：**
  - 卡住从 18 / 19 降到 1 / 1，比预测的还干净。
  - model 用了路点后 45 s 成功 17 局，比 direct 的 12 局多，也超过预测上限。配对伸入误差 model − direct 的中位数是 −0.34 cm（直线时是 +0.37，但那时大多卡住，伸入误差没意义）。
  - **手还是会碰到桌沿**（复核更正：28/29，有路线的 27/28，每次约 1 s）：原写 28/28 局，在伸手的 22–59 % 处，手点在桌沿前 5–8 cm、桌面上方 3–11 cm，主要是食指。实际路线和规划差 4.1 cm（中位数），大于规划留的约 3.3 cm 余量。不再被挡住，但还是碰到了。
  - 绕路使路线长了一倍多（复核更正：直线距离中位数 39 cm，不是 35 cm）：成功时刻从约 13.2 s 推到约 16.2–16.5 s，正好跨过 SIMPLE 的 16 s 官方时限。官方成功率 direct 2 → 5，model 3 → 2。

## 结论与下一步

（2026-11-20，独立复核前的草稿）

1. **H1 成立。**TabletopGrasp 的"卡住"是因为手走直线穿过桌沿，不是跟踪误差。
   - 证据：只换成从那一眼算出的绕行路线，最后的命令不变，卡住就从 direct 18 / model 19 降到 1 / 1；剩下的一局是目标认错（ep 24），没有路线。
   - 每种方法 29 局有效，一个种子，一次运行。
2. **H2 成立。**用机器人自己的模型去掉身体后，那一眼的桌沿误差在看到的 26/30 局里中位数是 0.19 cm；老的 20 cm 一刀切是 7.61 cm。另外 4 局是亮面或玻璃桌，桌沿附近的深度往下塌。
3. **有路线之后，model 比 direct 好。**
   - 45 s 成功 17 比 12；伸入误差逐局配对差的中位数 −0.34 cm，均值 −0.2。
   - 差距比盒子任务（−1.6 到 −2.4 cm）小很多，单次运行，强度：低。
4. **路线没有被控制器照着走。**
   - 实际手点和规划差 4.1 cm（中位数，direct），大于规划留的约 3.3 cm 余量。28/28 局手指在越过桌沿时擦到了桌面，只是不再被挡住。
   - 这是"考虑自身模型的规划"的直接证据：规划器假设身体照着走，实际有约 4 cm 的偏差。
5. **代价是时间。**路线长约 80 cm，比直线的约 35 cm 长一倍多。成功时刻从约 13 s 推到 16–18.7 s，跨过 SIMPLE 的 16 s 官方时限，官方成功率 direct 2 → 5、model 3 → 2。
6. **盒子任务的检验很弱。**只放开右臂、锁住腰的规划器在 83/90 局够不到预抓取位姿（不放障碍物也 IK 失败）。只有 7 局真的用了路点。model 的优势在这些任务上没变（−2.01 / −2.36 / −1.58），但大部分局等于没加路点。

**其他可能的解释：**
- (1) 卡住消失可能部分来自"多了几段腿、伸手变慢"（0.15 m/s 走 80 cm），而不是避开了桌沿。还没做对照：同样慢、同样多段，但不避障的路线。
- (2) model 多出的成功可能来自它最后一段的修正，正好补偿了绕行带来的偏差；和"自身模型更准"不是一回事。
- (3) ep 22、26 两局 model 的出发状态和 direct 不同（走路分岔了），却用了 direct 的路点。

**四类陷阱：**
- benchmark 选择：TabletopGrasp 是因为它最差才挑出来的；盒子任务按预注册一起跑了，但实际只检验了 7 局。
- 数据泄漏：cuRobo 只用那一眼的点云和机器人自己的状态，没用仿真真值。真值只用来评估。路点是用 direct 的伸手开始状态离线规划的，这个状态评测时机器人自己也有，相当于在线规划，只是没真的在线跑。
- 指标误用：同时报了 45 s 和 16 s 官方两个口径。官方口径下 model 变差了（3 → 2），已写明。"碰桌"预测按原定义判 ✗，没有改定义。
- 事后挑选：设定中途改了三处，都是为了能跑通，不是看结果挑的，原因都记了，v1 的结果也保留：
  - 去掉出发姿态处的体素；
  - 目标改在三维里去掉；
  - ep 6 用 2 cm 回退。

**对照判据：**
- 第 3 步"被挡住的局 ≤ 4"成立（1 / 1）：cuRobo 加感知足以解决卡住，路径版自身模型搁置。
- 但"碰桌 28/28"和"跟踪偏差 4.1 cm > 余量"说明，规划时不考虑控制器实际怎么走是有代价的（擦碰、余量要留大）。这指向"考虑自身模型的规划"，值得查新。
- 盒子判据形式上成立（≤ −1.0），实际证据弱。

**下一步（按优先级）：**
1. 规划器放开腰（右臂 + 腰 3 个），让盒子任务也能规划，补上第 6 条的检验。
2. 加对照：同样段数、同样速度但不绕行的路线，排除"只是变慢了"（其他可能的解释 1）。
3. 把余量和跟踪偏差对上：激活距离改成 5 cm，或者路线速度降低。看擦碰能不能消失、要多花多少时间。
4. 缩短路线：先往回撤再抬起是因为出发时手在桌板下面。可以试着先只往下、往回撤一点，看能不能赶回 16 s 以内。
5. 查新："按控制器的实际跟踪偏差来规划（self-model-aware planning）"。

### 复核后修订（2026-11-20，以此为准；上面的草稿保留不删）

我抽查了复核指出的问题：
- ep 20 没碰桌，ep 24 没有路线却算进了碰桌数。这和第 3 步逐局输出里 `('2', 0) direct_avoid table False`、`('2', 4) table True` 一致。
- 起手越深越容易卡这一条和我的说法相反。我接受它的统计。

1. **H1：主效应成立，"绕行本身起作用"还没证明。**
   - 卡住 18/19 → 1/1。三种卡住定义结果一样；方法内配对 10:0 和 15:1。
   - 直线的参考轨迹本身就整块穿过桌板，在运动学上就会撞，不需要用跟踪误差来解释。
   - 但这次干预同时改了四样东西，没有拆开：位置路径、手腕姿态时间表、段数（8 段）、参考时长（2.6 → 5.7 s）。
   - 要做的对照是复核的 E1：同样段数和时长的直线分段；位置和姿态分开换；先撤后抬的固定模板。
2. **H2 成立，附尾部。**
   - 30 局桌沿误差中位数 0.20 cm。没看到桌沿的 4 局误差 8–19 cm，而机器人自己不知道是哪 4 局。
   - "看到"是按真值判定的，不是机器人能用的信号。
3. **"有路线后 model 优于 direct"：证据不足。**
   - 45 s 口径 17 比 12，不一致的局 6:1，p = 0.125。
   - 官方 16 s 口径反过来，direct 5、model 2，不一致的局 0:3。
   - 配对伸入误差中位数 −0.34 cm，去掉 ep 24 后 −0.50，Wilcoxon p = 0.18–0.30，95 % 区间跨 0。
   - 这件事要等多扰动重复（复核 E2）再说。
4. **碰桌：28/29 局，有路线的局里 27/28，每次约 1 s（8–137 tick），不只是擦一下。**
   - 接触的位置和直线卡住的位置几乎一样。路线让手挂住后又脱开了，没有完全避开。
   - 原先"手点偏差 4.1 cm 大于余量 3.3 cm"的论证是错位比较，作废。接触处规划余量中位数约 5.6 cm，那些 tick 的手点偏差约 3.4 cm，原因还没找到，候选是没记录的手腕姿态误差。
   - 复核排除了"规划和评测的手指姿态不一致"：用评测时的手指姿态重算，余量中位数 3.28 cm，最小 1.74 cm。
   - **撤回"这是考虑自身模型的规划的直接证据"。**按预注册判据（被挡住 ≤ 4 局），路径版自身模型搁置。要不要碰这个方向，先看复核的 E3：激活距离 5 cm 能不能消除接触、多花多少时间。
5. **代价是时间。**
   - 路线 80 cm，直线距离中位数是 39 cm（不是 35 cm）；参考时长从 2.6 s 到 5.7 s。
   - 有效速度差不多（0.14 和 0.15 m/s），准确说法是"路更长"，不是"走得更慢"。
   - 官方 16 s 口径下 direct 2 → 5，model 3 → 2。以后 16 s 和 45 s 两个口径并列为主指标。
6. **盒子判据：未检验。**
   - 没有路线的 83 局，两次运行逐局完全相同。
   - 有路线的 7 局，避障后配对差 −2.15，直线 −2.22。
   - 原来标的 ✓ 作废。
7. **路线偏差：**model 的 ✗ 是指标定义造成的。只看规划那段（到最后一个路点），两种方法都是 3.82 cm。
8. **数据泄漏：**没有实质泄漏。小问题有两处，都没量化：
   - 路点用真实骨盆位姿换坐标，没走里程计，误差 ≤ 0.49 cm、0.8°；
   - 去自身像素用了复制场景的整场景分割，包含了真实桌子的遮挡关系。

**修订后的下一步：**
1. 复核的 E1 归因对照，direct、28 局、每组加 2–3 个小扰动。不做这个，"cuRobo 加那一眼"的结论就站不住。
2. 复核的 E3：激活距离 5 cm，逐 tick 记手腕位姿。
3. 规划器放开腰，让盒子任务真正检验到。
4. model 对 direct 的优势：等 E2，即多扰动重复，加上"direct 加常数偏置"的对照。

## 独立复核

（2026-11-20，general-purpose agent，只给了结果 / 结论 / 原始输出路径 / 代码 diff；回复原样贴入。它的统计脚本在 scratchpad/review/。）

# 复核报告：curobo_avoid_20261120（独立、偏怀疑）

全程只读：远程没有写入任何文件，没有碰 tmux 或进程。所有统计都是把本地脚本经 ssh 的 stdin 送到远程跑的，每次几秒到几分钟。脚本在 `<Mac 会话临时目录>/review/`（`a1_tabletop.py` 到 `a10_ref_line.py`）。

## 一、判定表

| # | 结论 | 判定 | 理由 |
|---|---|---|---|
| 1 | H1：卡住是因为直线穿过桌沿，不是跟踪误差 | **主效应支持；"是绕行本身起作用"证据不足** | 卡住从 18/19 降到 1/1，三种卡住定义结果一样；方法内配对 10:0 和 15:1（p=0.002 / 0.0005）。我按 `motor.reach` 的指令重建了直线参考，手在 29/29 局都整块穿过 10 cm 桌板，所以参考本身就撞，用不着跟踪误差来解释。但没卡住的 11 局也一样穿板，"穿桌沿"只是必要条件，不是充分条件。另外这次干预同时改了位置路径、手腕姿态时间表（`via_rot`）、段数（8 段）和参考时长（2.6 → 5.7 s），没有拆开。 |
| 2 | H2：桌沿误差 0.19 cm（26/30），老办法 7.61 cm | **支持，附条件** | 数字全部复现。"看到"是按真值判定的，机器人自己不知道哪 4 局没看到，这 4 局误差 8–19 cm。30 局整体的中位数是 0.20 cm，结论成立，但应同时报尾部。 |
| 3 | 有路线后 model 优于 direct | **证据不足** | 45 s 成功 17 比 12：不一致的局是 6:1，精确检验 p=0.125。官方 16 s 口径反过来，direct 5、model 2，不一致的局 0:3。配对伸入误差中位数 −0.34，Wilcoxon p=0.30，95% 区间 [−1.71, 0.80]，而且算进了没有路线的 ep 24。 |
| 4 | 控制器没照着路线走，是"考虑自身模型的规划"的直接证据 | **前半句支持；定量论证不成立；"直接证据"不支持** | "28/28"是错的，应为 27/28。"4.1 cm > 3.3 cm 余量"是错位比较。预注册判据写的是"被挡住 ≤ 4 就把路径版自身模型搁置"，结论反过来把它当成证据。 |
| 5 | 代价是时间 | **支持** | 数字对。但直线距离中位数是 39 cm，不是 35 cm；路线有效速度 0.14 m/s，直线是 0.15 m/s，所以准确说法是"路更长"，不是"走得更慢"。 |
| 6 | 盒子任务的检验很弱 | **支持，而且比写的更弱** | 没有路线的 83 局，avoid 和直线两次运行逐局完全一样，等于同一次运行跑了两遍。盒子判据应标"未检验"，不该标 ✓。 |

## 二、数字核对（独立重算）

| 项 | 条目里 | 我算的 | 是否一致 |
|---|---|---|---|
| 卡住，直线 / 避障 | 18/19 → 1/1 | 18/19 → 1/1。三种定义完全一致：预抓取误差 > 20 cm；预抓取结束时手还没过桌沿；最后一个 tick 右手还接触桌子 | 一致 |
| 45 s 成功 | 2/3 → 12/17 | 2/3 → 12/17。其中成功后到结束仍拿着（`success_end`）的是 10/12、16/17 | 一致 |
| 16 s 官方成功 | 2/3 → 5/2 | 5/2 | 一致 |
| 重放一致性 | 30/30 | `trace_rows` 和 `rows_eyes` 的 direct 30/30 相同；model 直线重跑和 `rows_eyes` 也是 30/30 相同 | 一致 |
| 配对伸入误差（避障） | 中位数 −0.34，均值 −0.2 | −0.34 / −0.20（n=29，16 负 13 正）；去掉 ep 24 后 −0.50，Wilcoxon p=0.18 | 数字一致，但不显著 |
| 碰桌 | 28/28 | **28/29，其中有路线的 27/28**：ep 20 没碰，第 28 局是没有路线、照旧卡住的 ep 24。接触时长中位数 47 tick（约 0.94 s），范围 8–137 tick，不只是"擦一下" | **不一致** |
| 接触位置 | 伸手的 22–59 %；桌沿前 5–8 cm、桌面上方 3–11 cm | 22–59 %；x 4.7–9.2 cm、z −0.3–10.1 cm（5–95 分位）。部位里还有 `right_wrist_yaw_link`（171 tick），不只食指 | 基本一致 |
| 路线偏差 | direct 4.1 / model 6.77 | 4.10 / 6.77 | 一致，但含义见下文 |
| 第 2 步规划成功 | 27/29（v1 26/29） | 27/29（v1 26/29），失败是 ep 6 FINETUNE、ep 24 IK | 一致 |
| 规划用时 | 中位数 2.47，最大 4.81 s | 2.47 / 4.81 | 一致 |
| clearance（cuRobo 球） | 中位数 3.29，最小 3.03 | 3.29 / 3.03 | 一致 |
| clearance（仿真几何） | 27/27，最小 2.63 | 一致；但评测实际用的 ep 6 回退路线是 2.20，所以实际执行的 28 条最小是 2.20，仍 ≥ 2 cm | 小遗漏 |
| 盒子任务配对差 | −2.01 / −2.36 / −1.58 | 完全一致；只看 7 局有路线的：避障 −2.15，直线 −2.22 | 一致 |
| 只有 7 局用了路点 | 7 | 7（XMovePick 的 ep 15、18、19、24、26、27、29）。另外 83 局在没有障碍物时也是 IK_FAIL；2 cm 重试 0/83 | 一致 |
| 路线长度和时长 | 约 80 cm 对约 35 cm | 指令路线 80.2 cm，直线距离中位数 **39.3 cm**；参考时长 5.7 s 对 2.6 s | 直线长度偏小 |
| 出发状态 | 26/30 一致 | 26/29 一致；ep 22 两种方法的手点差了 **7.7 cm**（条目只写了"基座 3 cm"） | 一致，但低估了 ep 22 |

## 三、四类陷阱

### benchmark 选择

- **任务只有一个，问题可能是自己造成的。** TabletopGrasp 是因为最差才挑出来的。卡住的前提是每局伸手开始时手指都已经伸在桌板下面（真实几何距离 0–0.4 cm）。这是评测走位和站位造成的姿态，换个起手姿态也许同样能解决，不一定需要规划器。
- **"不在出发姿态"不准确。** 意外一节说"卡住和没卡住的差别不在出发姿态"。实际上起点在桌沿前约 3.8 cm、桌面下约 9.2 cm 的那 7 局，direct 直线全部卡住；其余 22 局只有 11 局卡住（Fisher p=0.026）。
- **盒子任务没检验到通用性。** 规划器在 83/90 局够不到，这一组没有起到检验作用。
- **主口径和官方口径方向相反。** 主口径用的是 45 s，但 SIMPLE 自己的 16 s 口径给出的方向相反，至少应该并列为主指标。

### 数据泄漏：没有发现实质泄漏

逐项查过：
- 障碍物来自那一眼的点云；目标用 `cloud_eyes_final` 去掉；目标位姿是评测自己的 `pw`。
- 真实桌子只出现在检查和诊断字段里。`prep` 里存了 `path` 和 `pre_err_cm`，但 `plan2.py` 只把它们打进日志。

有三处小问题：
1. **坐标系用的是真值。** 路点和体素用仿真的真实骨盆位姿转换，没有经过里程计。评测把里程计误差注入了抓取目标，却没有注入路点。误差最大 0.49 cm、0.8°，影响可以忽略，但严格说偏乐观。
2. **自身像素用了真实桌子的遮挡。** 去掉自身像素用的是复制场景的整场景分割（`obstacles.py` 的 `robot_px = geom & is_robot_geom[gid]`），等于用了真实桌子对机器人的遮挡。合规的做法是只渲染机器人，再和测得的深度做比较。影响大概很小，没有量化。
3. **离线规划。** 离线规划用的是 direct 直线重放的起点。direct 的避障运行在 29/29 局起点完全相同，等价于在线规划。model 有 3 局起点不同（ep 22 差 7.7 cm），这对 model 是吃亏，不是泄漏。

另外，第 1 步改桌面挑点规则是在看到 ep 1 的真值误差之后做的。它只影响 H2 的桌沿指标，不影响规划。

### 指标误用

- **"卡住"定义稳健。** 见上表，三种定义一致。唯一分歧是 model 避障 ep 26：手离桌沿差 0.5 cm，但在桌面上方 17 cm、没有接触，是 model 自己的命令偏移了 22 cm 造成的。
- **"碰桌"计数和含义都要改。**
  - 计数错了：28/29 里算进了没有路线的 ep 24。
  - 含义也要改：接触位置和直线卡住的位置几乎一样（手点在桌沿前 6.4 cm，桌面上方 7.4 cm 对 8.0 cm），持续约 1 s。也就是说，路线没有让手避开卡住的那个地方，只是让它挂住后又脱开了。
- **路线偏差对 model 不公平，对结论 4 也是错位比较。**
  - 这个偏差混进了最后停稳时的误差：28 局里有 12 局，最大偏差和预抓取的最终误差相差不到 0.5 cm。
  - 只看规划那段（到最后一个路点为止），direct 和 model 都是 3.82 cm，完全一样。model 的 6.77 全部来自它有意的最后偏移。
  - 在接触发生的那些 tick，手点偏差是 3.36 cm；而规划在这些位置给出的余量是 3.0–7.2 cm，中位数 5.6 cm。"3.3 cm"是出发附近（规划第 6–9 个点）的最小值，不是接触处的余量。所以单靠手点偏差解释不了接触，还有姿态误差没有记录。
  - 我排除了一个怀疑：规划把手指锁在起点读数（约 0–5°），评测里手指会张到预抓取开度（拇指约 42°、食指 15–40°）。在复制场景里用评测时的手指姿态重算，余量中位数从 3.17 变成 3.28 cm，最小 1.74 cm，没有一局 ≤ 0。所以接触不是手指姿态不一致造成的。
- **配对伸入误差** 应该去掉 ep 24（两种方法都卡住、都没有路线），并报置信区间。

### 事后挑选

- **三处设定改动本身没问题。**
  - 都发生在评测之前，依据的是规划器状态，不是评测结果。
  - 试跑 `step3_try/`（ep 0、3）和定稿逐字段相同；v1 的结果也保留了。
  - 敏感性：同时撤掉 ep 6 回退和三维去目标（ep 1 改回走直线），卡住最多 3/3，45 s 成功 direct 10、model 15，结论不变。
  - `--free` 每局去掉 1–18 个体素；仿真几何的余量 ≥ 2.2 cm，说明没有在桌沿上开出危险的洞。
- **真正的事后问题在结论的写法上。**
  - 预注册判据是"被挡住 ≤ 4 局 → 路径版自身模型搁置"。结论 4 和下一步 5 把没中的碰桌预测改写成"直接证据"，这是挪了球门。
  - 结论 3 的标题只用 45 s，官方口径的反转放到了结论 5 里。
  - 盒子判据标了 ✓，实际只有 7 局有差别。

## 四、可能推翻结论的实验

### E1：卡住消失到底是因为绕开桌沿，还是因为多了几段腿、时间变长、姿态变了

在 28 局有路线的局上只跑 direct（CPU 上可复现），每组再加 2–3 个小扰动重复，比如站位 ±1 cm。

| 组 | 做法 | H1 的预测 | 如果结果相反 |
|---|---|---|---|
| a 直线分段 | 路点均匀放在直线上，段数和每段时长都和路线一致（共 5.7 s），用直线的姿态时间表 | 仍卡住约 18 局（运动学上参考照样穿板） | 卡住 ≤ 4，"绕行"的归因被推翻 |
| b 拆开位置和姿态 | 两组："路线位置 + `via_rot=None`"和"直线位置 + 规划的姿态" | 位置那组起作用 | 姿态那组起作用，说明主因是手腕姿态 |
| c 不用感知 | 两组：固定模板（先后撤 8 cm、抬到预抓取高度，再去目标）；错配路线（用别的局的路线位移） | 不如 cuRobo | 和 cuRobo 一样好，"cuRobo 加那一眼够用"要改成"任何先撤后抬都够"，感知的作用就没有被证明 |

每组上仿真前先做一次 `a10_ref_line.py` 式的运动学预检。只要出现"参考离桌子有余量、但仍然卡住"的局，就直接证伪 H1。

### E2：model 是否真的优于 direct

1. 5 组扰动 × 28 局 × 2 种方法。预注册配对检验（140 对，McNemar），16 s 和 45 s 并列为主指标。
2. 常数偏置对照：direct 的目标下移 0.6 cm，这是 model 相对 direct 在伸入时的 z 方向配对差（−0.64 cm）。成功与否和 z 方向偏差显著相关（p=0.011）。如果 direct 加偏置后追平 model，那么 model 在 TabletopGrasp 上的优势只是一个常数修正，不是依赖状态的自身模型。
3. 在最后一个路点重新算 model 的修正。现在的修正是按"从起点走直线"预测的，而实际是走完路点后的最后一段，超出了它的训练分布。这一项检验"其他可能的解释（2）"。
4. 按时间对齐比较成功率，比如"伸入开始后 X 秒内"，以去掉 model 约 0.3 s 的滞后。

已有数据里的疑点：
- model 多出的 6 局里，ep 4、8 在直线时也是 model 独有的成功，和路线无关。
- 剩下 4 局里有 3 局，model 的伸入误差反而比 direct 大。
- 直线、都没卡住的 9 局里，model 的伸入误差反而大 0.81 cm（9 局中 7 局更大）。

### E3：针对结论 4

激活距离改成 5 cm，同时逐 tick 记录手腕位姿，算越过桌沿那一段按时间对齐的位置和姿态偏差。如果 5 cm 余量就能消除接触，而且多花的时间 ≤ 1 s，那么这个任务不需要"考虑自身模型的规划"。

## 五、建议修改的句子

- 意外和结论 4："28/28" 改为 "28/29（有路线的 27/28，ep 20 没碰；另一局是没有路线的 ep 24）"，"擦到"改为"接触约 1 s"。
- 结论 4：删掉"4.1 cm 大于 3.3 cm 余量"的论证和"直接证据"的说法，改为"接触处规划余量约 5.6 cm，手点偏差约 3.4 cm，还有姿态误差没有记录；先试更大余量"。
- 结论 3：降为"45 s 口径下方向有利、不显著，16 s 口径反向"，并报去掉 ep 24 后的结果。
- 路线偏差：model 的 ✗ 注明是指标定义造成的（规划那段两种方法都是 3.82 cm）。
- 盒子判据：✓ 改为"未检验"（83 局两次运行逐局相同；7 局：−2.15 对 −2.22）。
- "直线约 35 cm"改为约 39 cm，"变慢"改为"路更长"。
