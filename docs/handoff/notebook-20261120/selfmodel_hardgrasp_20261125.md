# selfmodel_hardgrasp_20261125：在要转身、要蹲、伸进柜子的抓取任务上，直接执行和用自身模型执行逐局对比

- 日期：2026-11-25。设计、预测、判据都在跑之前写下，单独提交。实现细节在设计局上定好并冻结以后，写进"预测修订"再正式跑。
- 机器：4090。仿真用 CPU，最多约 150 个进程。感知和规划用 1–3 号卡。0 号卡上有别人的任务，不用、不碰。我们自己的 keepalive 照常跑。
- 代码版本：`selfwm`，新脚本放在 `scripts/experiments/hardgrasp_20261125/`。每次跑之前把代码抄进输出目录的 `asrun/`，并记下 md5。
- 关联：
  - 用户 2026-11-25 的决定："应该跑两个对比实验就够了，一个是得到抓取目标姿势之后直接发个 wbc 执行，以及另一个使用了自身模型执行的"；"按照建议来，然后尽量多开进程，最好能在 2 个小时内全部跑完"。
  - "按建议"指两件事：这一轮 LocoManiLab 只跑 ScaleBFM，SONIC 在 LocoManiLab 里没有模型，以后采数据补；看一眼都在开局位置看，和现在的规则一样。
  - 用到的模型：
    - [locomanilab_selfmodel2_20261117_a800](../../../../research-notebook/experiments/locomanilab_selfmodel2_20261117_a800.md)：ScaleBFM 在 LocoManiLab 自己的仿真里训的模型，验证误差 1.54 cm；
    - [bigym_endpoint_20261125](bigym_endpoint_20261125.md)：BiGym 自带控制器的 M2，终点误差 0.1–0.4 cm。
  - [bigym_grasptol_20261125](bigym_grasptol_20261125.md)：搬盘子、翻杯子的夹起这一步容差 ≥ 20 mm，这次的 BiGym 任务换成更窄、更低、更高的位置。

## 假设

已经训好的自身模型，能把"一次命令"的手送得更准。在要求更高的抓取任务上，手更准应该能变成更多的成功局：
- 要先走过去、转向再抓的；
- 目标很低或在柜子里的；
- 目标很高的。

SIMPLE 和 BiGym 的搬盘子、翻杯子太宽松，手偏 2–4 cm 也能抓起来，所以显不出差别。这次换成更难的任务。

## 做法

### 1. 任务

LocoManiLab（ScaleBFM，Dex3 手），12 个任务：

| 类型 | 任务 |
|---|---|
| 先走过去、转向再抓（开局离物体 1.5–2.9 m，物体偏开局朝向 0–48°，罐子那个最多 158°） | `approach_table_and_lift_{apple, bowl, salt_shaker, dumbbell, statue, watering_can}`、`approach_table_and_stand_canned_drink_upright` |
| 低处、柜子里（开局离物体 0.4–0.8 m） | `take_statue_from_cabinet`（台面下的矮柜，离地约 7 cm）、`open_low_microwave_and_lift_apple`、`open_drawer_take_apple`、`lift_soap_bottle_from_cabinet`、`lift_candle_from_floor` |

BiGym（自带控制器，Dex1），4 个任务：`TakeCupsG1`（吊柜，1.18 m）、`DishwasherUnloadCupsG1`、`DishwasherUnloadPlatesG1`、`DishwasherUnloadCutleryG1`。`PickBoxG1` 要两只手，不跑。

### 2. 每局怎么跑（两组共用，直到发伸手命令那一刻）

1. 开局站稳，头部相机拍一张彩色图加深度图，这是整局唯一的一眼。
   - SAM 3 按任务里的物体名找目标；
   - GraspGen-X 出抓取（LocoManiLab 用 Dex3，BiGym 用 Dex1）；
   - 按固定规则挑一个抓取：先去掉够不着的和会撞到这一眼里家具的。

   这一眼每局只算一次、存下来，两组读同一份。
2. 按抓取算出站在哪、朝哪、蹲多低，规则固定。用控制器自己的走路走过去、转向、蹲下。机器人对自己位置的估计：
   - LocoManiLab 用腿部里程计。接不上时改用仿真骨盆，并写明。
   - BiGym 照 `bigym_bench.py` 用骨盆读数。已知这比真机乐观。

   两组用同一条规则。
3. 伸手分两段：先到预抓取点（沿接近方向往外退一段），再直线进到抓取点；然后合手，抬起，保持。中途不再看。
4. 要先开门或开抽屉的（微波炉、抽屉），用固定的开门动作，两组相同。
5. 抓起后还要放的，有时间就用固定的放置动作，两组相同；没时间就只报夹起这一步。

### 3. 两组的唯一区别：伸手命令怎么定

| | 直接执行 | 用自身模型执行 |
|---|---|---|
| LocoManiLab ScaleBFM | 我们的 IK 参考直接以目标为终点（`lml_tasks.py` 的 `alone`） | 模型反解命令，让预测的落点等于目标（`lml_tasks.py` 的 `selfmodel`：不动点迭代 6 步，`models_scalebfm_s0`） |
| BiGym 自带控制器 | cuRobo 规划到目标的手臂关节轨迹（M2 就是在这种命令上训的） | 同一条路线，终点按 M2 预测的"实际手减参考手"反向挪，cuRobo 重新规划；规划失败就按 2/3、1/3 缩小挪动量，再不行退回直接执行的路线 |

预抓取和抓取两段都按各自的规则定命令。每段只发一次，不看结果再修。

### 4. 局数

- **LocoManiLab**：官方配置里机器人和物体都不抖动，每个场景只有一种局面。我们自己加抖动：机器人开局位置 ±5 cm、朝向 ±10°，物体位置 ±3 cm、绕竖直轴 ±30°（物体要留在原来的支撑面上）。每个任务 30 局正式局，场景少的每个场景多跑几局。设计局用另一组种子。官方不抖动的那一局单独报。
- **BiGym**：正式局用官方评测种子 620000–620029，每个任务 30 局；设计局用 732000–732009。
- 两组按局配对，同一局两组读同一份那一眼。

## 度量

- **主度量：夹起并抬起的成功。** 物体被手夹住，离开原来的支撑面，抬高 ≥ 10 cm 并保持 1 s；LocoManiLab 的 lift 类任务按它自己的判定，抬高 30 cm 保持 1 s。仿真真值只在局后用来打分。
- **官方整件任务成功**：做了放置的任务报，不进判定。
- **手离目标多远**：合手那一刻，手的夹持中心到抓取目标的距离，按局配对。
- **失败原因**：每局归一类，顺序优先：
  - 没认出或看不见；
  - 站位够不着，或 IK 解不出；
  - 走路摔倒；
  - 伸手途中撞到家具；
  - 手偏了抓空；
  - 抬起时掉落；
  - 其他。
- **按用户规则必须报**：
  - 每局起始状态在不在模型训练数据的范围里（29 个关节逐个和训练数据的 0.5–99.5 % 区间比）；
  - 模型在这些局上的预测误差（预测落点和实际落点的距离），和它的验证误差对比。
- **统计**：每个 benchmark 合并所有任务，按局配对，McNemar 精确检验；另报按任务的两组成功数。

## 预测（跑之前写，没看过这些任务的任何运行）

**LocoManiLab ScaleBFM**（约 360 对）：
- **合手时手离目标**（中位数）：直接执行 5–10 cm，自身模型 1.5–4 cm。
  - 依据：11-18 关类任务上停在空中的偏差，单独执行 8.2–8.7 cm、模型 1.6–1.7 cm。这次要走路、要蹲，模型多数时候在训练范围外，会更差。
- **夹起成功率**：
  - 直接执行 0–25 %，自身模型 10–50 %，差 +5 到 +35 个百分点；
  - 按判据"赢"的可能约六成；
  - `take_statue_from_cabinet` 两组都接近 0：G1 蹲不到离地 7 cm 的地方。
- **起始状态在训练范围外的局**：低处和柜子类 ≥ 50 %，走过去类 20–60 %。
- **看错或看不见**：占全部局的 5–25 %。

**BiGym**（120 对）：
- **合手时手离目标**：直接执行 1.5–3 cm，自身模型 0.3–1.5 cm；落在训练范围外时，模型 0.5–3 cm。
- **夹起成功率**：
  - 直接执行 20–70 %；
  - 4 个任务里至少 3 个，两组差在 ±10 个百分点以内；
  - 餐具最可能有差别，因为东西小；
  - 按判据"赢"的可能约两成半。
- **官方整件任务成功**：0–50 %。

## 判据（什么结果会让我改做法，下一步做什么）

**"赢"**：一个 benchmark 合并后，自身模型的夹起成功率比直接执行高 ≥ 10 个百分点，并且 McNemar p < 0.05。

| 结果 | 下一步 |
|---|---|
| LocoManiLab 赢 | 在 LocoManiLab 里给 SONIC 采数据、训练，按同样办法再跑一遍（单独登记），再扩到更多任务 |
| 不赢，但手明显更准（配对的手离目标中位数差 < −1 cm，p < 0.05） | 结论是"手更准没变成更多成功"。看失败原因表：失败主要在感知、站位、撞家具、走路，就照实报告瓶颈不在自身模型 |
| 手没有更准，或更差 | 看起始状态是否在训练范围内。≥ 50 % 的局在范围外时，结论只能是"现成的模型推不到没见过的姿态"，不能说自身模型没用；下一步按这些姿态采数据、重新训练，再登记 |

另外三条排除规则：
- 某个任务直接执行的夹起成功率已经 ≥ 80 %，标记为"到顶"，它显不出差别；
- 某个任务一半以上的局败在感知或站位上，这个任务不进判定，单独报；
- 官方整件任务成功率只报，不进判定，因为它包括放置，放置不归自身模型管。

## 预测修订

### 修订 1：LocoManiLab 部分，设计冻结以后、正式跑以前（2026-11-25）

预测和判据都不改。下面是实现时定下的规则和改动。设计局用的是种子 1000–1009，每个任务两组各 10 局，结果已经看过，所以列在最后，不算正式结果。

**冻结的代码**：06:54:30Z，在 `$W/outputs/hardgrasp_20261125/locomanilab/asrun/`，`MD5SUMS` 文件的 md5 是 87389b61…。

| 文件 | md5 |
|---|---|
| `hg_run.py` | ce8397e3… |
| `hg_look.py` | b4cfe38e… |
| `hg_walk.py` | fc3a5693… |
| `hg_ggx_server.py` | ec3944af… |
| `hg_batch.sh` | ad2382da… |
| `hg_formal.sh` | c1e271bd… |
| `wm_s0.pt` | a7191a0a… |

Mac 上的副本和 4090b 上的 md5 一致。

1. **机器**：改在 4090b 的两张卡上跑，CPU 进程不超过 100 个，没用 4090 的 1–3 号卡。
2. **模型更正**：用的是 `lml_hostdata_20261118/models_scalebfm_s0`，验证误差 1.88 cm。
   - 它在 LocoManiLab 的 benchmark 宿主里训，身体和这些任务一致，也是 `lml_tasks.py selfmodel` 实际加载的模型。
   - 范围检查用同一份训练数据 `data_scalebfm/all.npz`。
   - 条目"关联"里写的 selfmodel2 模型（1.54 cm）训练时用的是另一个身体，写错了。
3. **任务从 12 个减到 7 个**：蜡烛（地上的矮凳）、苹果、盐瓶、碗（这三个在厨房台面）、柜子里的雕像、哑铃、浇水壶。放弃的 5 个：
   - 开低处微波炉拿苹果、开抽屉拿苹果：冻结以前没来得及做开门动作；
   - 浴室柜里拿皂液瓶：场景 FloorPlan405 不在 LocoManiLab 的资产清单里；
   - 客厅桌上抬雕像：10/10 局走到约 1 m 时，参考轨迹和实际位置偏差超过 0.5 m，被判"跟丢"。我们的走法不会绕障碍；
   - 把罐子立起来：罐子在开局朝向 158° 的方向，开局那一眼 10/10 局看不见。
4. **局数**：每个任务只用一个场景。正式局 31 局：官方不抖动的一局，加上 0–29 号。环境自带的重置抖动关掉，只用我们加的抖动。物体抖动后要通过环境自己的穿透检查，最多试 32 次，不行就放回不抖动的位置，并记下 `jitter_ok`。
5. **一眼**：
   - 头部相机 `robot0_head_camera`，640×480。
   - 走过去的任务，拍照前先把上身后仰 0.30 rad。不后仰的话，相机下俯 44°，拍不到 2.2 m 外的台面。
   - SAM 3 的提示词用任务里的物体名，两处例外：蜡烛另加 "pillar candle"；柜子里的雕像改用 "statue"，原来的提示词取到的是柜门。
   - GraspGen-X 用 Dex3，离群点阈值从 1.4 cm 放宽到 5 cm。原来的阈值会把 2.8 m 外的稀疏点云整个删掉。
6. **挑抓取**，按下面顺序筛，最后取 GraspGen 分数最高的：
   1. 自身模型的命令能表达这个抓取：朝向偏离站位朝向不超过 ±60°，俯仰 −5° 到 90°，滚转不超过 40°（模型的命令表达不了滚转）；
   2. 抓取点和预抓取点都落在训练数据的盒子里；
   3. 抓取和预抓取位姿的手掌盒子里，这一眼的家具点各不超过 3 个。

   两组用同一个抓取。
7. **站位和走法**：
   - 走过去的任务：站在让物体落在身前 0.42 m、偏右 0.18 m 的位置。半径 0.32 m 内有家具点，就每次往后退 5 cm，最多退 40 cm。走法是转、直走、再转。
   - 自己的位置用腿部里程计。支撑脚靠仿真脚底的法向接触力来选（当作脚底力传感器），朝向用仿真 IMU。那一眼的坐标锚在拍照时刻的仿真骨盆上，相当于地图原点。
   - 到站后里程计显示离站位还差 ≥ 10 cm，就补走一次。
   - 近处的任务原地不动。
8. **蹲多低**：抓取点高 ≥ 0.70 m 就站着；更低时骨盆高度取 z + 0.36 m，限在 0.52–0.76 m。
9. **伸手**：
   - 预抓取点沿 GraspGen 的接近方向（手掌法向）往外退 8 cm，然后直线进到抓取点。
   - 抓取点高 ≥ 0.80 m 时，最前面先伸到一个中间点：比预抓取点靠近身体 22 cm、高 5 cm。不加的话，手从髋部直线伸向台面，会穿过台面下的柜门。
   - 合手 0.9 s，抬起 0.45 m，保持 2 s。
   - 中间点和抬起这两段，两组都用直接命令。两组的区别只在预抓取和抓取这两段。
10. **度量的口径**：
    - "手离目标"量的是手点（wrist_yaw_link 往前 13 cm）到所选抓取的手点，不是真实的夹持中心，两者相差约 4.5 cm。
    - 从第一条伸手命令到合手，手、手腕或手肘碰到任何非目标的家具，都记"撞家具"，判得很严。
    - "掉落"包括抬起来了但没到判定高度的局。
    - lift 类任务用它自己的判定（抬高 30 cm、保持 1 s）；柜子里的雕像用我们的判定（抬高 ≥ 10 cm、保持 1 s）。
11. **设计局上看到的**（种子 1000–1009，不算结果）：
    - 合并后，只有直接执行成功的 2 局，只有自身模型成功的 9 局，p = 0.065。
    - 蜡烛：直接执行 2/10、自身模型 7/10；直接执行有 5 局撞到凳子；合手离目标 5.5 对 1.0 cm。
    - 碗、盐瓶、柜子里的雕像，多数局败在没有可用的抓取、看不见或 IK 够不到。按判据，一半以上的局败在感知或站位上，这些任务不进判定。
    - "看不见或没认出"在盐瓶上是 70 %，罐子是 100 %，高于预测的 5–25 %。

### 修订 2：BiGym 部分，设计冻结以后、正式跑以前（2026-11-25）

预测和判据都不改。设计局（种子 732000–732009）的结果已经看过，列在最后。

**冻结版本**：06:54:55Z。冻结副本在 `$W/outputs/hardgrasp_20261125/bigym/asrun/`，md5 清单 `md5.txt` 共 106 行，Mac 和 4090 上 104 个文件的 md5 一致。主要文件：

| 文件 | md5 |
|---|---|
| `bigym_hg.py` | 593cc463eb1a1a0cde52de5dc6641b89 |
| `graspgen_dex1.py` | 48ee102c… |
| `m2.py` | 308a17cc… |
| `bigym_formal.sh` | 75950282… |
| M2 权重 `via_point.pt` | 761c864a… |

1. **4 个任务只剩洗碗机取杯子（DishwasherUnloadCupsG1），共 30 对。** 放弃的三个都是看不见或够不着，按判据属于感知或站位失败：
   - **吊柜里拿杯子**：开局时杯子不在头部相机的画面里（0 像素），躯干后仰也一样；SAM 3 10 个种子都没找到。
   - **取餐具**：叉子不在开局画面里；转 90° 后只剩约 60 像素，大多还被自己的夹爪挡住；SAM 3 10/10 没找到。
   - **取盘子**：盘子在下层碗架中间。腿离门和架子留够距离的最近站位上，它在骨盆前方 0.60–0.67 m。7/10 种子没有一个抓取能解出 IK，蹲到 0.62、前倾 0.3 也不行；另外 3 个种子 SAM 3 认错了区域。
2. **感知**：
   - SAM 3 提示词 "mug"，分数阈值 0.1（这些渲染图上的分数只有 0.1–0.65）。
   - 取至少有 30 个点、离骨盆水平距离最近的那个实例。
3. **挑抓取**：
   - 依次过滤：接近方向离竖直向下 ≤ 60°；张开的夹爪在抓取点和往回 8 cm 的接近路径上，碰到这一眼家具点 ≤ 3 个；沿合拢方向重新居中后开口 ≤ 80 mm；GraspGen-X 分数 ≥ 0.5。
   - 剩下的按 GraspGen-X 的分数排序；走到站位后，取第一个预抓取点和抓取点都能解出 IK 的（误差在 1 cm / 10° 以内，手臂不贴身体）。
4. **站位**：
   - 先原地左转 90°，再横走、前走两轮，让物体点云的中位点落在骨盆前方 0.33 m、右侧 0.15 m。站位按点云中位点算，不按抓取算。
   - 前走时，腿的高度范围内离最近的家具点留 0.22 m。
   - 不蹲，躯干不前倾。
   - 去掉了预备姿态：它会让两条手臂都落到 M2 的训练范围外，预测误差变成 1–2 m。
5. **伸手**：
   - 第 1 段到预抓取点：直接执行用 cuRobo 规划，避开这一眼的体素。自身模型把终点挪 −(g + p) 再规划，失败时依次缩到 2/3、1/3。
   - cuRobo 连直接执行都规划不出来时（设计局 4/9，都是体素造成的 IK 失败），直接执行改走直线，自身模型就等于直接执行。
   - 第 2 段沿直线进到抓取点。自身模型沿用第 1 段实际用的挪动量，把整条直线平移；IK 够不到就缩到 2/3、1/3。原因是发第 2 段命令时 M2 在训练范围外。第 2 段 M2 的预测只记录，不用。
   - 之后合手 0.6 s，1.5 s 内抬高 15 cm，保持 1.5 s。
6. **计分**：
   - 夹起成功：最后 1 s 里每一步都抬高 ≥ 10 cm、手里还夹着杯子、没摔倒。
   - 不做放置，所以不报官方的整件任务成功。
   - 关掉了任务时限，是否在官方时限内另外记录。
7. **手离目标**：量的是夹爪中心到"感知出的抓取目标"的距离，用骨盆读数换到世界坐标，不是到真值目标。
8. **失败分类**：
   - "没认出"包括这一眼的目标中心离杯子真实中心超过 10 cm（局后打分）。
   - "撞家具"指伸手途中右臂或夹爪碰到任何非目标物体 ≥ 2 mm。
9. **统计**：只有一个任务，合并就是它自己。手的距离另加一个配对符号检验。
10. **设计局上看到的**（不算结果）：
    - **夹起成功两组都是 0/9。** 9 局合手后都判定夹住了，可手往上抬 9–14 cm 时，杯子不跟着动。试过的改法都还是 0/9：换抓取排序、只抓开口 ≤ 30 mm 的位置、往里推、只抓上半部、延长合手时间、慢抬。
    - **手离目标**：5 对里自身模型真的挪了终点，每一对都是自身模型更近，配对中位差 −2.2 cm，p = 0.06。直接执行的中位数是 5.3 cm，比预测的 1.5–3 cm 大。合手前，直接执行把杯子推开 27–32 mm，自身模型 2–10 mm。另外 4 对因为规划失败，两组完全一样。
    - **起始状态**：第 1 段命令时，8/9 局有 3–4 个关节（左臂和腰）在 M2 训练范围外；第 2 段命令时，中位数 14 个关节在范围外，包括右臂全部 7 个。
    - **M2 第 1 段的预测误差**：中位 0.8 cm，p90 2.5 cm，是验证误差的 2–8 倍。
11. **决定照跑，并预先说明**：主度量很可能两组都是 0，落在地板上，失败在夹持这一步，不在手准不准。按判据，这一局面只能读"手是否明显更准"那一行。跑这一组约 10 分钟。

## 结果

### LocoManiLab：ScaleBFM，7 个任务，抖动的 30 局 + 官方 1 局，按局配对

**输出位置**：4090b 上的 `$O=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/hardgrasp_20261125/locomanilab/`

| 文件 | 内容 | md5 |
|---|---|---|
| `analysis_formal.txt` | 正式统计 | fb01a01d9ca71f655e3be4744c1f5bd2 |
| `stats_formal.txt` | 冻结的 `hg_run.py stats` 的输出 | 414cdc89f1663fc4e22f5e33f32c27e7 |
| `formal.log` | 运行记录 | de76829c4e475de2999e7058c719f0cf |
| `asrun/MD5SUMS` | 冻结代码的 md5 清单 | 87389b613257a1d4b43d5e685b23bf1d |

逐局结果在 `episodes/`，开局那一眼在 `looks/`，第一次启动的日志在 `logs/launch1/`。

**分析脚本**：`analysis_formal.txt` 是冻结以后另写的 `scripts/experiments/hardgrasp_20261125/hg_analyze.py`（md5 c6f9d8f9…）算的。它只读 `episodes/`，包括按排除规则筛任务、配对差的区间、官方局单列。

**和修订 1 不一致的一处**：正式局不是从 `asrun/` 副本跑的。
- 第一次启动时，`asrun/` 副本里的 selfwm `env.py` 按文件自己的位置推仓库根目录，推到了 `outputs/hardgrasp_20261125`，找不到 `assets/scalebfm`。84 个跑局进程开局就全部退出，一局也没写出。
- 第二次改从 `$W/scripts/experiments` 跑同一批文件，跑前跑后都和 `asrun/MD5SUMS` 核对过，一致。启动脚本另写了 `hg_formal_w.sh`（md5 d963a478…）。
- 第一次拍好的那一眼都留着沿用。跑局时的身体状态和拍照时缓存下来的比，最大偏差 0.000 mm。

`formal.log` 原文：

```
formal start 2026-11-25T06:56:46Z tasks 7
looks done 2026-11-25T06:58:16Z
runs done 2026-11-25T06:59:07Z
stats done 2026-11-25T06:59:07Z
launch 1 failed: every run worker died at start (asrun: selfwm env.py REPO -> outputs/hardgrasp_20261125, assets/scalebfm not found); its looks are kept; logs in logs/launch1 2026-11-25T07:01:51Z
formal launch 2 start 2026-11-25T07:01:51Z tasks 7, code /wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments (md5 = asrun)
looks done 2026-11-25T07:02:11Z
runs done 2026-11-25T07:09:12Z
md5 after the runs: same as asrun
stats done 2026-11-25T07:09:12Z
```

`analysis_formal.txt` 原文：

```
formal episodes 0..29; 420 records

== per task (A = alone, S = selfmodel; success = the task's own lift judgement, take-statue: lift >= 10 cm 1 s)
IThorBathroomLiftCandleFromFloor  pairs 30  kept (perception+stance 27%)
  success A 8/30  S 18/30  | A-only 0  S-only 10  McNemar p 0.00195
  close-dist median A 5.9 cm (n 18)  S 1.0 cm (n 18)  | paired S-A median -4.8 cm, 95% CI [-5.9, -3.8], n 18, sign p 7.63e-06
  A reasons ok 8 perception 8 stance_or_ik 0 fell_walking 0 hit_furniture 10 missed 0 dropped 0 other 4 | start in range: pre-grasp cmd 18/18, grasp cmd 17/18; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 0.7 cm, grasp 1.1 cm (n 18; val 1.88) | grasp-reach miss 5.9 cm
  S reasons ok 18 perception 8 stance_or_ik 0 fell_walking 0 hit_furniture 0 missed 0 dropped 0 other 4 | start in range: pre-grasp cmd 18/18, grasp cmd 17/18; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 0.7 cm, grasp 1.0 cm (n 18; val 1.88) | grasp-reach miss 1.0 cm | model's own predicted miss 0.00 cm
IThorKitchenApproachTableAndLiftApple  pairs 30  kept (perception+stance 10%)
  success A 8/30  S 10/30  | A-only 0  S-only 2  McNemar p 0.5
  close-dist median A 5.4 cm (n 28)  S 4.5 cm (n 28)  | paired S-A median -0.0 cm, 95% CI [-0.5, +0.6], n 28, sign p 0.851
  A reasons ok 8 perception 0 stance_or_ik 2 fell_walking 0 hit_furniture 20 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 13/28, grasp cmd 15/28; joints out at pre-grasp median 1 | pred err (pred vs actual landing) pre-grasp 5.8 cm, grasp 4.9 cm (n 28; val 1.88) | grasp-reach miss 5.4 cm
  S reasons ok 10 perception 0 stance_or_ik 4 fell_walking 0 hit_furniture 16 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 13/28, grasp cmd 18/28; joints out at pre-grasp median 1 | pred err (pred vs actual landing) pre-grasp 7.7 cm, grasp 4.7 cm (n 28; val 1.88) | grasp-reach miss 4.5 cm | model's own predicted miss 0.04 cm
IThorKitchenApproachTableAndLiftBowl  pairs 30  EXCLUDED (perception+stance 100%)
  success A 0/30  S 0/30  | A-only 0  S-only 0  McNemar p 1
  close-dist median A 45.4 cm (n 8)  S 39.1 cm (n 7)  | paired S-A median -3.8 cm, 95% CI [-9.7, +2.6], n 7, sign p 0.453
  A reasons ok 0 perception 0 stance_or_ik 30 fell_walking 0 hit_furniture 0 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 4/8, grasp cmd 5/8; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 31.2 cm, grasp 19.9 cm (n 8; val 1.88) | grasp-reach miss 45.4 cm
  S reasons ok 0 perception 0 stance_or_ik 30 fell_walking 0 hit_furniture 0 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 4/8, grasp cmd 1/8; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 33.7 cm, grasp 18.7 cm (n 8; val 1.88) | grasp-reach miss 41.6 cm | model's own predicted miss 18.46 cm
IThorKitchenApproachTableAndLiftSaltShaker  pairs 30  EXCLUDED (perception+stance 67%)
  success A 2/30  S 7/30  | A-only 0  S-only 5  McNemar p 0.0625
  close-dist median A 4.5 cm (n 10)  S 2.0 cm (n 10)  | paired S-A median -1.7 cm, 95% CI [-3.1, +1.0], n 10, sign p 0.344
  A reasons ok 2 perception 20 stance_or_ik 0 fell_walking 0 hit_furniture 8 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 4/10, grasp cmd 6/10; joints out at pre-grasp median 1 | pred err (pred vs actual landing) pre-grasp 2.1 cm, grasp 2.4 cm (n 10; val 1.88) | grasp-reach miss 4.5 cm
  S reasons ok 7 perception 20 stance_or_ik 0 fell_walking 0 hit_furniture 3 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 4/10, grasp cmd 6/10; joints out at pre-grasp median 1 | pred err (pred vs actual landing) pre-grasp 2.0 cm, grasp 1.9 cm (n 10; val 1.88) | grasp-reach miss 2.0 cm | model's own predicted miss 0.01 cm
IThorKitchenTakeStatueFromCabinet  pairs 30  EXCLUDED (perception+stance 93%)
  success A 0/30  S 0/30  | A-only 0  S-only 0  McNemar p 1
  close-dist median A 31.0 cm (n 19)  S 23.7 cm (n 19)  | paired S-A median -5.5 cm, 95% CI [-23.1, +1.5], n 18, sign p 0.0963
  A reasons ok 0 perception 8 stance_or_ik 18 fell_walking 0 hit_furniture 4 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 20/20, grasp cmd 6/20; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 24.6 cm, grasp 16.0 cm (n 20; val 1.88) | grasp-reach miss 36.2 cm
  S reasons ok 0 perception 8 stance_or_ik 22 fell_walking 0 hit_furniture 0 missed 0 dropped 0 other 0 | start in range: pre-grasp cmd 20/20, grasp cmd 7/20; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 28.9 cm, grasp 17.9 cm (n 20; val 1.88) | grasp-reach miss 23.9 cm | model's own predicted miss 0.11 cm
IThorLivingRoomApproachTableAndLiftDumbbell  pairs 30  kept (perception+stance 28%)
  success A 11/30  S 11/30  | A-only 5  S-only 5  McNemar p 1
  close-dist median A 10.9 cm (n 30)  S 3.5 cm (n 29)  | paired S-A median -2.3 cm, 95% CI [-7.0, -0.8], n 29, sign p 0.000104
  A reasons ok 11 perception 0 stance_or_ik 5 fell_walking 0 hit_furniture 13 missed 1 dropped 0 other 0 | start in range: pre-grasp cmd 30/30, grasp cmd 24/30; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 1.4 cm, grasp 4.3 cm (n 30; val 1.88) | grasp-reach miss 10.9 cm
  S reasons ok 11 perception 0 stance_or_ik 12 fell_walking 0 hit_furniture 2 missed 4 dropped 1 other 0 | start in range: pre-grasp cmd 30/30, grasp cmd 19/29; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 1.8 cm, grasp 2.4 cm (n 29; val 1.88) | grasp-reach miss 3.5 cm | model's own predicted miss 0.05 cm
IThorLivingRoomApproachTableAndLiftWateringCan  pairs 30  EXCLUDED (perception+stance 52%)
  success A 3/30  S 8/30  | A-only 0  S-only 5  McNemar p 0.0625
  close-dist median A 12.8 cm (n 21)  S 5.8 cm (n 21)  | paired S-A median -2.1 cm, 95% CI [-6.9, +0.7], n 21, sign p 0.0784
  A reasons ok 3 perception 9 stance_or_ik 2 fell_walking 0 hit_furniture 12 missed 4 dropped 0 other 0 | start in range: pre-grasp cmd 21/21, grasp cmd 21/21; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 1.6 cm, grasp 3.9 cm (n 21; val 1.88) | grasp-reach miss 12.8 cm
  S reasons ok 8 perception 9 stance_or_ik 11 fell_walking 0 hit_furniture 0 missed 2 dropped 0 other 0 | start in range: pre-grasp cmd 21/21, grasp cmd 14/21; joints out at pre-grasp median 0 | pred err (pred vs actual landing) pre-grasp 3.5 cm, grasp 7.3 cm (n 21; val 1.88) | grasp-reach miss 5.8 cm | model's own predicted miss 0.02 cm

== pooled, ALL kept tasks: pairs 210  success A 32 (15.2%)  S 54 (25.7%)  diff +10.5 pp  | A-only 5  S-only 27  McNemar exact p 0.000113
   close-dist paired S-A median -2.31 cm, 95% CI [-3.83, -1.11], n 131, sign p 1.13e-08, Wilcoxon p 2.54e-10

== pooled, after the exclusion rule: pairs 90  success A 27 (30.0%)  S 39 (43.3%)  diff +13.3 pp  | A-only 5  S-only 17  McNemar exact p 0.0169
   close-dist paired S-A median -1.90 cm, 95% CI [-3.83, -0.70], n 75, sign p 2.18e-06, Wilcoxon p 5.52e-08

== pooled over all tasks, per group (formal jittered episodes)
  alone: n 210  reasons ok 32 perception 45 stance_or_ik 57 fell_walking 0 hit_furniture 67 missed 5 dropped 0 other 4
    pre-grasp reach: n 135  start in range 110/135  joints out (of 29) median 0 max 2  | pred err median 2.14 cm [IQR 1.11-15.87] n 135 (val 1.88)  | in-range only: 1.72 cm  | reach miss median 10.64 cm
    grasp reach: n 135  start in range 94/135  joints out (of 29) median 0 max 5  | pred err median 4.31 cm [IQR 2.07-9.55] n 135 (val 1.88)  | in-range only: 3.66 cm  | reach miss median 8.96 cm
  selfmodel: n 210  reasons ok 54 perception 45 stance_or_ik 79 fell_walking 0 hit_furniture 21 missed 6 dropped 1 other 4
    pre-grasp reach: n 135  start in range 110/135  joints out (of 29) median 0 max 2  | pred err median 4.37 cm [IQR 1.15-22.67] n 135 (val 1.88)  | in-range only: 2.55 cm  | reach miss median 5.02 cm  | model's own predicted miss median 0.66 cm
    grasp reach: n 134  start in range 82/134  joints out (of 29) median 0 max 6  | pred err median 4.52 cm [IQR 1.34-13.02] n 134 (val 1.88)  | in-range only: 2.13 cm  | reach miss median 4.40 cm  | model's own predicted miss median 0.03 cm

== official unjittered start (ep -1)
  IThorBathroomLiftCandleFromFloor: alone: fail (hit_furniture, close 7.6 cm, rise 0.5 cm) | selfmodel: success (ok, close 1.0 cm, rise 40.8 cm)
  IThorKitchenApproachTableAndLiftApple: alone: fail (hit_furniture, close 6.2 cm, rise 0.2 cm) | selfmodel: success (ok, close 0.9 cm, rise 42.6 cm)
  IThorKitchenApproachTableAndLiftBowl: alone: fail (stance_or_ik, close nan cm, rise 0.0 cm) | selfmodel: fail (stance_or_ik, close nan cm, rise 0.0 cm)
  IThorKitchenApproachTableAndLiftSaltShaker: alone: fail (hit_furniture, close 9.1 cm, rise 0.1 cm) | selfmodel: fail (hit_furniture, close 3.1 cm, rise 0.8 cm)
  IThorKitchenTakeStatueFromCabinet: alone: fail (stance_or_ik, close 27.6 cm, rise 0.2 cm) | selfmodel: fail (stance_or_ik, close 17.7 cm, rise 0.2 cm)
  IThorLivingRoomApproachTableAndLiftDumbbell: alone: success (ok, close 5.1 cm, rise 42.0 cm) | selfmodel: fail (missed, close 3.5 cm, rise 0.0 cm)
  IThorLivingRoomApproachTableAndLiftWateringCan: alone: success (ok, close 3.4 cm, rise 34.3 cm) | selfmodel: fail (missed, close 3.1 cm, rise 0.0 cm)

worker errors 8 [('IThorBathroomLiftCandleFromFloor', 'selfmodel', 7), ('IThorBathroomLiftCandleFromFloor', 'selfmodel', 27), ('IThorBathroomLiftCandleFromFloor', 'selfmodel', 3), ('IThorBathroomLiftCandleFromFloor', 'selfmodel', 25), ('IThorBathroomLiftCandleFromFloor', 'alone', 7)]; jitter fell back to unjittered 0; look-state deviation max 0.000 mm
```

`stats_formal.txt` 原文：冻结脚本的统计，31 对，把官方局也算进去了。

```
episodes -1:30, 434 records
== IThorBathroomLiftCandleFromFloor  pairs 31
  alone     success  8/31  official  8  lift10  8  close-dist median 6.1 cm  reasons {'hit_furniture': 11, 'ok': 8, 'other': 4, 'perception': 8}  start in range 19/19  grasp-reach pred err median 1.2 cm
  selfmodel success 19/31  official 19  lift10 19  close-dist median 1.0 cm  reasons {'ok': 19, 'other': 4, 'perception': 8}  start in range 19/19  grasp-reach pred err median 1.0 cm
  discordant: alone-only 0, selfmodel-only 11, McNemar exact p 0.000977; paired close-dist diff (S - A) median -4.9 cm (n 19)
== IThorKitchenApproachTableAndLiftApple  pairs 31
  alone     success  8/31  official  8  lift10  8  close-dist median 6.0 cm  reasons {'hit_furniture': 21, 'stance_or_ik': 2, 'ok': 8}  start in range 13/29  grasp-reach pred err median 4.8 cm
  selfmodel success 11/31  official 11  lift10 11  close-dist median 4.5 cm  reasons {'ok': 11, 'hit_furniture': 16, 'stance_or_ik': 4}  start in range 13/29  grasp-reach pred err median 4.6 cm
  discordant: alone-only 0, selfmodel-only 3, McNemar exact p 0.25; paired close-dist diff (S - A) median -0.0 cm (n 29)
== IThorKitchenApproachTableAndLiftBowl  pairs 31
  alone     success  0/31  official  0  lift10  0  close-dist median 45.4 cm  reasons {'stance_or_ik': 31}  start in range 4/8  grasp-reach pred err median 19.9 cm
  selfmodel success  0/31  official  0  lift10  0  close-dist median 39.1 cm  reasons {'stance_or_ik': 31}  start in range 4/8  grasp-reach pred err median 18.7 cm
  discordant: alone-only 0, selfmodel-only 0, McNemar exact p 1; paired close-dist diff (S - A) median -3.8 cm (n 7)
== IThorKitchenApproachTableAndLiftSaltShaker  pairs 31
  alone     success  2/31  official  2  lift10  2  close-dist median 4.8 cm  reasons {'hit_furniture': 9, 'perception': 20, 'ok': 2}  start in range 4/11  grasp-reach pred err median 2.7 cm
  selfmodel success  7/31  official  7  lift10  7  close-dist median 2.0 cm  reasons {'hit_furniture': 4, 'ok': 7, 'perception': 20}  start in range 4/11  grasp-reach pred err median 2.1 cm
  discordant: alone-only 0, selfmodel-only 5, McNemar exact p 0.0625; paired close-dist diff (S - A) median -2.1 cm (n 11)
== IThorKitchenTakeStatueFromCabinet  pairs 31
  alone     success  0/31  official  0  lift10  0  close-dist median 30.6 cm  reasons {'stance_or_ik': 19, 'perception': 8, 'hit_furniture': 4}  start in range 21/21  grasp-reach pred err median 12.4 cm
  selfmodel success  0/31  official  0  lift10  0  close-dist median 23.4 cm  reasons {'stance_or_ik': 23, 'perception': 8}  start in range 21/21  grasp-reach pred err median 16.9 cm
  discordant: alone-only 0, selfmodel-only 0, McNemar exact p 1; paired close-dist diff (S - A) median -5.6 cm (n 19)
== IThorLivingRoomApproachTableAndLiftDumbbell  pairs 31
  alone     success 12/31  official 12  lift10 12  close-dist median 10.8 cm  reasons {'ok': 12, 'hit_furniture': 13, 'stance_or_ik': 5, 'missed': 1}  start in range 31/31  grasp-reach pred err median 4.3 cm
  selfmodel success 11/31  official 11  lift10 12  close-dist median 3.5 cm  reasons {'missed': 5, 'ok': 11, 'hit_furniture': 2, 'stance_or_ik': 12, 'dropped': 1}  start in range 31/31  grasp-reach pred err median 2.3 cm
  discordant: alone-only 6, selfmodel-only 5, McNemar exact p 1; paired close-dist diff (S - A) median -2.3 cm (n 30)
== IThorLivingRoomApproachTableAndLiftWateringCan  pairs 31
  alone     success  4/31  official  4  lift10  4  close-dist median 12.6 cm  reasons {'ok': 4, 'perception': 9, 'stance_or_ik': 2, 'hit_furniture': 12, 'missed': 4}  start in range 22/22  grasp-reach pred err median 3.8 cm
  selfmodel success  8/31  official  8  lift10  8  close-dist median 5.7 cm  reasons {'missed': 3, 'ok': 8, 'perception': 9, 'stance_or_ik': 11}  start in range 22/22  grasp-reach pred err median 7.3 cm
  discordant: alone-only 1, selfmodel-only 5, McNemar exact p 0.219; paired close-dist diff (S - A) median -1.5 cm (n 22)
== pooled: alone-only 7, selfmodel-only 29, McNemar exact p 0.000313; paired close-dist diff median -2.5 cm (n 137)
```

### BiGym：洗碗机取杯子（DishwasherUnloadCupsG1，种子 620000–620029，30 对）

- 跑的是冻结副本，开跑时 md5 校验 106/106 通过（`asrun_md5_check.txt`）。
- 时间（UTC）：06:59:32 开始，07:00:05 拍完那一眼，07:01:06 结束。
- 输出在 4090 的 `$W/outputs/hardgrasp_20261125/bigym/formal/`：
  - `report.txt`，md5 6f0506ddf1c167edc52bd3dd0744a6d9；
  - `analysis.txt`，md5 13067289ace1cec9a6727130bafad803；
  - `rows/*.jsonl`、`look/`、`logs/`。
- `report.txt` 是冻结以后另写的 `scripts/experiments/hardgrasp_20261125/bigym_report.py` 算的（md5 d1f7ad45…）。它只读 `rows/`，不参与运行。

`analysis.txt` 开头：

```
rows 60, error rows 0

=== dishwasher_unload_cups === 30 pairs (seeds 620000-620029)
  look hash ok: 30/30; prefix (reset..stance) identical D vs S: 30/30; within the official budget D 15 S 20
  lift success: D 2/30 (6.7 %), S 3/30 (10.0 %), S - D +3.3 pp; discordant D-only 1, S-only 2; McNemar exact p = 1.0000
  hand to the grasp target at the close (cm), 20 pairs with both closes: D median 5.28, S median 4.27; paired S - D median -0.61 (S nearer on 11, farther on 2); sign test p = 0.0225
```

`report.txt` 第 [1]–[5] 节原文：

```
rows 60 (error rows 0), complete pairs 30

[1] grasp-and-lift success
  D 2/30, S 3/30; discordant D-only 1, S-only 2; McNemar exact p = 1

[2] hand (Dex1 pinch centre) to the grasp target at the close, cm
  all pairs with both closes: 20 pairs
    D median 5.28 cm, S median 4.27 cm; paired S - D median -0.61 cm, 95 % interval [-2.00, +0.00] (exact coverage 95.9 %); S nearer 11, farther 2, equal 7; sign test p = 0.02246
  pairs where S moved its goal (reach 1 scale 1, 2/3 or 1/3): 13 pairs
    D median 5.18 cm, S median 2.88 cm; paired S - D median -1.79 cm, 95 % interval [-2.54, -0.52] (exact coverage 97.8 %); S nearer 11, farther 2, equal 0; sign test p = 0.02246
  pairs where S ran D's route (D had no plan, or no scale planned): 7 pairs
    D median 5.69 cm, S median 5.69 cm; paired S - D median +0.00 cm, 95 % interval [+0.00, +0.00] (exact coverage 98.4 %); S nearer 0, farther 0, equal 7; sign test p = 1
    S's reach-1 scale counts: {'1.0': 13, 'None': 7}

[3] failure classes (registration order)
  class        D    S
  success       2    3
  perception    7    7
  reach         3    3
  fell          0    0
  furniture     1    1
  missed        2    4
  dropped      15   12
  other         0    0
    D no candidate with IK at the stance: 3
    D no target instance: 7
    S no candidate with IK at the stance: 3
    S no target instance: 7

[4] start state against M2's training band (29 joints, 0.5-99.5 %), at reach 1's command, D episodes
  20 episodes: any joint out 18, out-of-band joints median 4.0, max 4; right arm (7) any out 2, median 0.0
  joints out (episodes): {'left_shoulder_pitch_joint': 16, 'left_elbow_joint': 16, 'left_wrist_pitch_joint': 13, 'waist_pitch_joint': 12, 'right_wrist_roll_joint': 2, 'left_shoulder_roll_joint': 1, 'left_shoulder_yaw_joint': 1}
  at reach 2's command: out-of-band joints median 16.0, right arm median 7.0 of 7 (20 episodes)

[5] M2's prediction error, reach 1 (|predicted - actual settled palm offset|, cm; validation 0.1-0.4)
  D: n 20, median 1.00, p90 2.58, max 2.63; the actual offset itself median 2.81
  S: n 20, median 1.14, p90 2.63, max 11.28; the actual offset itself median 3.10
  reach 2 (recorded, not used; out of band): D median 173.4

```

`report.txt` 第 [6] 节（逐对）和第 [7] 节（掉落局在抬起之后杯子的接触）原文：

```
[6] per pair
  620000  D: dropped hand 5.16 mv 25.3 held True/False rise 0.01 sc None line False  |  S: dropped hand 2.63 mv 3.0 held True/False rise -0.03 sc 1.0 line False
  620001  D: dropped hand 6.05 mv 104.1 held True/False rise -0.04 sc None line True  |  S: dropped hand 6.05 mv 104.1 held True/False rise -0.04 sc None line True
  620002  D: reach(no candidate with ) hand None mv None held None/None rise None sc None line False  |  S: reach(no candidate with ) hand None mv None held None/None rise None sc None line False
  620003  D: perception(no target instance) hand None mv None held None/None rise None sc None line False  |  S: perception(no target instance) hand None mv None held None/None rise None sc None line False
  620004  D: perception(no target instance) hand None mv None held None/None rise None sc None line False  |  S: perception(no target instance) hand None mv None held None/None rise None sc None line False
  620005  D: perception(no target instance) hand None mv None held None/None rise None sc None line False  |  S: perception(no target instance) hand None mv None held None/None rise None sc None line False
  620006  D: dropped hand 6.39 mv 74.2 held True/False rise -0.25 sc None line False  |  S: dropped hand 5.87 mv 30.4 held True/False rise -0.04 sc 1.0 line False
  620007  D: missed hand 10.11 mv 6.4 held False/False rise 0.0 sc None line False  |  S: missed hand 5.58 mv 6.0 held False/False rise -0.01 sc 1.0 line False
  620008  D: dropped hand 5.37 mv 31.4 held True/False rise 0.0 sc None line False  |  S: dropped hand 4.05 mv 13.4 held True/False rise -0.04 sc 1.0 line False
  620009  D: furniture hand 4.04 mv 10.6 held False/False rise -0.0 sc None line True  |  S: furniture hand 4.04 mv 10.6 held False/False rise -0.0 sc None line True
  620010  D: dropped hand 5.96 mv 98.5 held True/False rise -0.32 sc None line True  |  S: dropped hand 5.96 mv 98.5 held True/False rise -0.32 sc None line True
  620011  D: perception(no target instance) hand None mv None held None/None rise None sc None line False  |  S: perception(no target instance) hand None mv None held None/None rise None sc None line False
  620012  D: reach(no candidate with ) hand None mv None held None/None rise None sc None line False  |  S: reach(no candidate with ) hand None mv None held None/None rise None sc None line False
  620013  D: OK hand 3.55 mv 36.4 held True/True rise 13.26 sc None line False  |  S: missed hand 1.01 mv 61.6 held False/False rise 0.0 sc 1.0 line False
  620014  D: perception(no target instance) hand None mv None held None/None rise None sc None line False  |  S: perception(no target instance) hand None mv None held None/None rise None sc None line False
  620015  D: dropped hand 4.95 mv 24.7 held True/False rise 0.02 sc None line False  |  S: dropped hand 2.88 mv 3.0 held True/False rise -0.02 sc 1.0 line False
  620016  D: dropped hand 5.15 mv 86.8 held True/False rise -0.33 sc None line True  |  S: dropped hand 5.15 mv 86.8 held True/False rise -0.33 sc None line True
  620017  D: perception(no target instance) hand None mv None held None/None rise None sc None line False  |  S: perception(no target instance) hand None mv None held None/None rise None sc None line False
  620018  D: dropped hand 5.91 mv 103.2 held True/False rise -0.08 sc None line True  |  S: dropped hand 5.91 mv 103.2 held True/False rise -0.08 sc None line True
  620019  D: dropped hand 5.57 mv 30.8 held True/False rise 0.09 sc None line False  |  S: dropped hand 2.6 mv 0.1 held True/False rise -0.06 sc 1.0 line False
  620020  D: dropped hand 5.64 mv 53.8 held True/False rise 0.01 sc None line False  |  S: missed hand 18.12 mv 0.1 held False/False rise 0.0 sc 1.0 line False
  620021  D: dropped hand 5.18 mv 36.9 held True/False rise -0.07 sc None line False  |  S: dropped hand 3.39 mv 5.8 held True/False rise -0.02 sc 1.0 line False
  620022  D: missed hand 4.61 mv 18.8 held False/False rise 0.0 sc None line False  |  S: OK hand 2.61 mv 1.5 held True/True rise 13.59 sc 1.0 line False
  620023  D: reach(no candidate with ) hand None mv None held None/None rise None sc None line False  |  S: reach(no candidate with ) hand None mv None held None/None rise None sc None line False
  620024  D: dropped hand 4.49 mv 87.9 held True/True rise 6.59 sc None line True  |  S: dropped hand 4.49 mv 87.9 held True/True rise 6.59 sc None line True
  620025  D: dropped hand 5.69 mv 104.1 held True/True rise 6.76 sc None line True  |  S: dropped hand 5.69 mv 104.1 held True/True rise 6.76 sc None line True
  620026  D: OK hand 3.52 mv 10.9 held True/True rise 12.03 sc None line False  |  S: OK hand 2.82 mv 3.9 held True/True rise 13.22 sc 1.0 line False
  620027  D: dropped hand 4.22 mv 14.2 held True/False rise -0.23 sc None line False  |  S: OK hand 2.66 mv 5.1 held True/True rise 12.02 sc 1.0 line False
  620028  D: perception(no target instance) hand None mv None held None/None rise None sc None line False  |  S: perception(no target instance) hand None mv None held None/None rise None sc None line False
  620029  D: dropped hand 6.19 mv 57.2 held True/False rise 0.04 sc None line False  |  S: missed hand 18.76 mv 0.1 held False/False rise -0.0 sc 1.0 line False

[7] dropped episodes: the mug's contacts after the lift
  620000 D: hand rise 10.45 cm, mug rise 0.01 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620000 S: hand rise 12.51 cm, mug rise -0.03 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620001 D: hand rise 8.71 cm, mug rise -0.04 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620001 S: hand rise 8.71 cm, mug rise -0.04 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620006 D: hand rise 2.39 cm, mug rise -0.25 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620006 S: hand rise 10.03 cm, mug rise -0.04 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620008 D: hand rise 12.21 cm, mug rise 0.0 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620008 S: hand rise 13.92 cm, mug rise -0.04 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug_1/mug', 'dishwasher/tray_mid', 0.0], ['mug_1/mug', 'mug/mug', 0.0]]
  620010 D: hand rise 8.61 cm, mug rise -0.32 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620010 S: hand rise 8.61 cm, mug rise -0.32 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620015 D: hand rise 10.44 cm, mug rise 0.02 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620015 S: hand rise 12.37 cm, mug rise -0.02 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620016 D: hand rise 8.98 cm, mug rise -0.33 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/tray_mid', 0.0], ['mug/mug', 'mug_1/mug', 0.0]]
  620016 S: hand rise 8.98 cm, mug rise -0.33 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/tray_mid', 0.0], ['mug/mug', 'mug_1/mug', 0.0]]
  620018 D: hand rise 8.77 cm, mug rise -0.08 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620018 S: hand rise 8.77 cm, mug rise -0.08 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
  620019 D: hand rise 9.69 cm, mug rise 0.09 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620019 S: hand rise 13.75 cm, mug rise -0.06 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620020 D: hand rise 9.47 cm, mug rise 0.01 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620021 D: hand rise 12.07 cm, mug rise -0.07 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620021 S: hand rise 13.6 cm, mug rise -0.02 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620024 D: hand rise 8.26 cm, mug rise 6.59 cm, gripper contacts 2, contacts [['mug/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_1', 0.1], ['mug/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_2', 0.1]]
  620024 S: hand rise 8.26 cm, mug rise 6.59 cm, gripper contacts 2, contacts [['mug/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_1', 0.1], ['mug/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_2', 0.1]]
  620025 D: hand rise 7.84 cm, mug rise 6.76 cm, gripper contacts 2, contacts [['mug_1/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_1', 0.1], ['mug_1/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_2', 0.1]]
  620025 S: hand rise 7.84 cm, mug rise 6.76 cm, gripper contacts 2, contacts [['mug_1/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_1', 0.1], ['mug_1/mug', 'g1_29dof_with_dex1_1/right_dex1_finger_link_2', 0.1]]
  620027 D: hand rise 14.36 cm, mug rise -0.23 cm, gripper contacts 0, contacts [['mug/mug', 'dishwasher/tray_mid', 0.0]]
  620029 D: hand rise 9.27 cm, mug rise 0.04 cm, gripper contacts 0, contacts [['mug_1/mug', 'dishwasher/dish_washer_mid_holder_1', 0.0], ['mug_1/mug', 'dishwasher/tray_mid', 0.0]]
```

## 意外

**LocoManiLab，预测和结果的对照：**

| 预测 | 结果 | 落在区间内吗 |
|---|---|---|
| 夹起成功率：直接执行 0–25 % | 15.2 % | 是 |
| 夹起成功率：自身模型 10–50 % | 25.7 % | 是 |
| 两组之差 +5 到 +35 个百分点 | +10.5（按判据排除后 +13.3） | 是 |
| 柜子里的雕像，两组都接近 0 | 都是 0/30 | 是 |
| 看错或看不见占 5–25 % | 45/210 = 21 %，加上蜡烛 4 局拍照崩溃，共 23 % | 是 |
| 手离目标：直接执行 5–10 cm | 伸手结束时的偏差中位数 8.96 cm | 是 |
| 手离目标：自身模型 1.5–4 cm | 4.40 cm | 略高于区间 |
| 低处和柜子类，起始状态在训练范围外的局 ≥ 50 % | 预抓取命令时，蜡烛和雕像 38/38 局都在范围内 | 否 |
| 走过去类，范围外的局 20–60 % | 25/97 = 26 % | 是 |

合并的"合手时手离目标"中位数没有算，表里两行用的是伸手结束时的偏差。

**复核更正**：这个登记的度量可以直接算。7 个任务合起来，直接执行 8.80 cm（n 134），自身模型 4.09 cm（n 132）；判定集是 6.23 对 3.12 cm。对照预测的结论不变：直接执行落在区间内，自身模型略高于区间。

**LocoManiLab，没想到的现象：**

1. **赢面几乎都来自蜡烛。**
   - 按判据排除后剩 3 个任务，只有自身模型成功的 17 对里，蜡烛占 10 对。去掉蜡烛，剩下的苹果和哑铃是 5 对 7，p = 0.77。
   - 5 个"走过去再抓"的任务（含被排除的盐瓶、浇水壶、碗）合起来是 5 对 17，p = 0.017。
     - **复核更正**：成功率 24/150 对 36/150，只差 +8.0 个百分点，没到登记的 10 个百分点。只有自身模型成功的 17 对里，10 对来自被排除的盐瓶和浇水壶。加上官方局是 7 对 18，p = 0.043。判定集恰好也是 5 对 17，两者只是巧合。
   - 蜡烛不用走路，起始状态全部在训练范围内。模型预测误差 1.0 cm，接近验证误差 1.88 cm。直接执行的手偏 5.9 cm，有 10 局归为撞家具。
     - **复核更正**：直接执行实际碰到凳子的是 14 局，其中 4 局照样成功；自身模型是 0 局。
2. **哑铃上手准了很多，成功数却一样（11 对 11）。**
   - 合手时手离目标的中位数从 10.9 cm 降到 3.5 cm。
   - 但自身模型在够得着的边上会过度修正：ep 19 它把预抓取命令往前推了 21 cm，IK 跟不上。站位/IK 失败 12 对 5，没碰到 4 对 1。
3. **走路以后，模型的预测误差约是验证误差的 2.4 倍。**
   - 抓取这一段的预测误差中位数：直接执行 4.31 cm，自身模型 4.52 cm，验证误差 1.88 cm。只看起始状态在范围内的局，是 3.66 和 2.13 cm。
     - **复核更正**：这几个数是 7 个任务合起来算的，包括不走路的蜡烛和雕像。只看走过去的任务，直接执行 4.41 cm、自身模型 4.52 cm，倍数仍约 2.4；其中起始状态在范围内的局是 4.08 和 3.03 cm。
   - 模型认为反解后的命令只差 0.03 cm，实际差 4.40 cm，自己估计的误差太乐观。
4. **"撞家具"是直接执行失败最多的一类：67 对 21。** 这个差距有一部分来自分类的先后顺序：自身模型的 IK 误差 > 3 cm 排在撞家具前面，算作站位/IK。去掉这条规则，撞家具是 98 对 69。
   - **复核补充**：
     - 撞家具只是失败时打的标签，不会中止这一局，也不影响成功判定。两组各有 25 个成功局，途中都碰过家具。
     - 两组各有 46 局是在厨房任务的中间点那一段第一次碰到家具。这一段两组都用直接命令，46 局完全相同，不构成两组的差别。
     - 第一次碰到家具的位置，按伸手段数局数：

       | 组 | 中间点 | 预抓取 | 抓取 |
       |---|---|---|---|
       | 直接执行 | 46 | 53 | 24 |
       | 自身模型 | 46 | 41 | 7 |
5. **官方不抖动的那一局，两组各赢两个任务。** 在哑铃和浇水壶上，自身模型离"感知出的抓取点"更近（3.5、3.1 cm），却没碰到物体，因为感知给的物体中心本身就偏了。
   - **复核更正**：哑铃偏了 8.3 cm，浇水壶偏了 10.3 cm。
6. **浇水壶差一点没被排除。** 它被排除的原因是感知加站位失败 52 %，刚过线，而这个比例是被自身模型自己的 IK 失败（11 对 2）推上去的。只按直接执行组算是 36 %。把它留下，判定集是 5 对 22，p = 0.0015。
7. **蜡烛有 4 局开局那一眼每次都崩溃。** 崩在 MuJoCo 的分割渲染里（`IndexError: index 906`），两组都记为"其他"。

**BiGym，预测和结果的对照：**

| 预测 | 结果 | 落在区间内吗 |
|---|---|---|
| 手离目标：直接执行 1.5–3 cm | 5.28 cm | 否 |
| 手离目标：自身模型 0.3–1.5 cm，落在训练范围外时 0.5–3 cm | 4.27 cm；自身模型真挪了终点的 13 对是 2.88 cm | 否（只有那 13 对落在"范围外"区间内） |
| 夹起成功：直接执行 20–70 % | 6.7 % | 否 |
| 没有"赢" | 2/30 对 3/30，p = 1 | 是 |
| 修订 2 预先说"两组都是 0" | 2 对 3 | 否 |

**BiGym，没想到的现象：**

1. **两组掉落 15 和 12 局，抬起时杯子完全不跟着动。** 抬完以后夹爪和杯子没有接触，杯子只碰着碗架和架上的齿。原因没查清，可能是夹持力或摩擦不够，也可能是杯子被齿卡住了。
2. **自身模型有 2 局（620020、620029）手偏了 18 cm。** 这两局重新规划本身成功了，但手停在离预抓取点 15 cm 的地方，M2 的预测误差 11 cm，原因没查清。
3. **起始状态 18/20 局在 M2 训练范围外。** 超出的主要是左臂和腰（转 90° 再横走的站法）；右臂只有 2/20 局超出。
4. ~~在官方时限内完成的局：直接执行 15，自身模型 20。~~
   - **复核更正**：这是记录的 bug，不是两组真有差别。每个 worker 先跑直接执行，再跑自身模型。`bigym_hg.py` 第 510 行在直接执行那一局把 `env.bigym._time_limit` 改成了 1e9，到自身模型那一局，第 509 行读出来的就是 1e9。我核对过 `rows/`：直接执行 30 行的 `budget_steps` 都是 1000，自身模型 30 行都是 1000000000。按 ≤ 1000 步重算，两组都是 15/20。`analysis.txt` 第 4 行的 `within the official budget` 也受了这个 bug 影响。

## 结论与下一步

（独立复核之前写；复核以后在原处加了"复核更正"）

### LocoManiLab：按判据算"赢"

**结果**：
- **判定集**：排除规则用完后剩蜡烛、苹果、哑铃 3 个任务，共 90 对。直接执行夹起成功 27 局（30.0 %），自身模型 39 局（43.3 %），差 +13.3 个百分点；只有直接执行成功的 5 对，只有自身模型成功的 17 对，McNemar p = 0.0169（`analysis_formal.txt`，"pooled, after the exclusion rule" 一行）。
- **7 个任务全算**：210 对，32 局对 54 局，差 +10.5 个百分点；5 对 27，p = 0.000113。
- **手离目标**：两组都走到合手的局里，7 个任务全算，配对差中位数 −2.31 cm，95 % 区间 [−3.83, −1.11]，n = 131。判定集是 −1.90 cm，区间 [−3.83, −0.70]，n = 75。（复核要求标明口径。）

**证据强度**：
- 1 次正式运行，每个任务一个场景，每个任务 30 个抖动局，按局配对，另有 1 局官方局；
- 只有 ScaleBFM 一个控制器，模型只用 s0 这一个种子。

**怎么理解**：
1. **最站得住的是蜡烛**，这是一个低处、不用走路的任务。
   - 起始状态全在训练范围内，模型预测误差 1.0 cm。
   - 手离目标从 5.9 cm 降到 1.0 cm；实际碰到凳子从 14 局降到 0（其中 10 局是直接执行的失败局）；成功 8 → 18（0 对 10，p = 0.002）。
   - 结论：自身模型在训练范围内的准度，能直接变成成功数。
2. **"走过去再抓"的 5 个任务合起来是 5 对 17**，p = 0.017。
   - 数据来自 `analysis_formal.txt` 的抖动局。
   - **复核更正**：原先写的出处 `stats` 不对，按它相加是 7 对 18。成功率只差 +8.0 个百分点，没到 10 个百分点的线。
   - 其中盐瓶、浇水壶、碗按判据被排除了，所以这个数不进判定，只作参考。
   - 判定集里去掉蜡烛只剩苹果和哑铃，是 5 对 7，p = 0.77。
   - 所以"走路、转向以后自身模型还能提高成功数"这一点，现在还不能下结论。
   - **补注（用户 11-25 问起后加）**：这次两组都没用走路模型，"自身模型"指的只是伸手模型。LocoManiLab 里没有训过走路模型：第一代走路模型 `post_wkp` 是在 selfwm 的 ScaleBFM 仿真里训的，训练时走 0.85 m 以内，场地是空地。两组走路都靠腿部里程计，差 ≥ 10 cm 就再补走一次。到抓取命令那一刻，里程计还留着约 2.6–2.8 cm 的位置误差，两组一样，伸手模型修不了。所以这次的结果检验不了"走路模型能不能让走过去、转身以后照样抓稳"。
3. **限制**：
   - 走路以后，模型的误差约是验证误差的 2.4 倍（只看走过去的任务，4.4–4.5 cm 对 1.88 cm）。
   - 在够得着的边上会过度修正，站位/IK 失败 79 对 57。
   - 模型对自己剩下的误差估计太乐观：它以为差 0.03 cm，实际差 4.40 cm。

### BiGym：没有"赢"，手也没有达到"明显更准"

**结果**：
- 夹起成功 2/30 对 3/30，p = 1。
- 两组都走到合手的 20 对里，配对差中位数 −0.61 cm，p = 0.022。没到判据要的 < −1 cm。
- 只看自身模型真挪了终点的 13 对，是 −1.79 cm，区间 [−2.54, −0.52]。这个分组是跑完以后才定的，不能拿来替代判据。

**对照判据**：落在"手没有更准"这一行。
- 18/20 局在 M2 训练范围外，超过一半。所以只能说"现成的 M2 推不到转身后的站姿"，不能说自身模型没用。
- **复核更正**：按判据读到这一行没错，但"推不到转身后的站姿"这个原因没有被证明。
  - 超出训练范围的是左臂和腰，负责伸手的右臂只有 2/20 局超出。
  - 合并差只有 −0.61 cm，主要原因是有 7 对 cuRobo 规划失败，按规则自身模型组照搬直接执行，两组完全一样。
  - BiGym 的自身模型组会对挪过的终点重新规划，路线和时长也跟着变。终点修正和换路线这两件事混在一起分不开。620020 和 620029 手偏 18 cm，更像是新路线造成的。
- **复核补充**：主度量几乎到底（掉落 15 和 12 局），这次对"自身模型能不能多成功"没有检验力。
- 夹起的瓶颈在夹持：掉落 15/12 局，杯子和夹爪之间没有保持住的接触。其次是感知，7 局没认出。

### 其他可能的解释

1. **一个固定的修正量，可能就够了。**
   - 蜡烛每局站在同一个地方，不用走路，直接执行的 5.9 cm 偏差可能大部分是固定偏向。
   - SIMPLE 上不用模型的固定偏差修正就和模型一样准（`selfmodel_s_20261124` 的复核）。
   - 这次按用户要求只跑两组，没有"不用模型的对手"，所以说不清蜡烛上的收益要不要模型来给。
   - **复核做的离线预演**：用设计局里直接执行的偏差中位数当常数，在蜡烛上修正。前提是偏差不随命令平移而变，蜡烛上同一对两组的偏差只差 0.8–1.2 cm，基本成立。结果：

     | 伸手段 | 直接执行 | 设计局常数 | 留一常数 | 自身模型 |
     |---|---|---|---|---|
     | 预抓取 | 6.0 cm | 1.6 cm | 1.8 cm | 0.7 cm |
     | 抓取 | 5.9 cm | 2.6 cm | 2.5 cm | 1.0 cm |

     - 一个固定修正很可能拿回蜡烛上大部分的成功差。
     - 模型也不只是一个常数：它的修正和逐局偏差高度相关（x 方向 0.96、z 方向 0.97），还剩约 1.5 cm 的优势。
2. **成功数的差别，可能有一部分来自修正的方向，而不是准度本身。** 修正如果总把手往上、往外挪，自然就少撞凳子和台面。逐局的落点偏差方向还没查。
3. **"手离目标"量的是到感知出的抓取点。** 感知本身偏了的局，这个数会误导，比如官方局里的哑铃，感知偏了 8.3 cm。

### 四类陷阱

| 陷阱 | 检查结果 |
|---|---|
| benchmark 选择 | 任务在跑之前登记好。5 个任务是在设计阶段因为跑不通被去掉的，对两组一样。排除规则事先登记。设计局里已经看到蜡烛的差别，但规则两组共用。 |
| 数据泄漏 | LocoManiLab 的模型用 `lml_hostdata_20261118` 的空地伸手数据训练，没用过这些任务的局；正式种子 0–29 和设计种子 1000–1009 分开。BiGym 正式局第一次用官方种子 620000–620029，M2 的训练数据来自 BiGym 伸手和抽屉两个任务的站位（种子从 800000 起，发命令后关掉场景碰撞），没用过洗碗机任务。 |
| 指标误用 | 主度量按登记的算。判定集和 7 个任务全算两种都报。手的距离按子集拆开的那几个数，都标明是跑完以后才分的。 |
| 事后挑选 | 每边只有一次正式运行。LocoManiLab 第一次启动一局都没写出，没有可挑的；两次启动都记在 `formal.log` 里。 |

### 下一步

- **按判据，LocoManiLab 赢了**：下一步是在 LocoManiLab 里给 SONIC 采数据、训练，按同样的办法再跑一遍（单独登记），再扩到更多任务。
- **复核后的下一步（建议先于 SONIC）：加第三组"固定修正"。**
  - **命令**：感知出的目标减去常数 c。c 按伸手段、链序号、骨盆高度档分开取，不看机器人状态。
  - **常数从哪来**：首选自身模型自己的训练数据 `lml_hostdata_20261118/data_scalebfm/all.npz`，也就是 `lml_tasks.lookup` 加一条骨盆高度匹配。备选是只跑直接执行组的新种子 2000–2009。不碰正式局 0–29 和官方局 −1。
  - **怎么跑**：读缓存的那一眼，跑 5 个任务，共 155 局。先重跑 2 局直接执行，确认和存档逐位相同。约 5 分钟 CPU。
  - **判据**：自身模型比固定修正高 ≥ 10 个百分点且 p < 0.05，才算"要模型"；否则结论改成"固定修正就够了"。
  - **顺带的变体**：只把蜡烛的命令抬高 5 cm，用来检验"靠修正方向，不靠准度"。
  - **次要检查**：BiGym 加一组"重新规划但终点不挪"；蜡烛换模型种子 s1 再跑一遍。
  - 这一组要单独登记，要不要跑由用户定。
  - **用户决定（2026-11-25）：不跑这一组。** 理由是固定修正不通用。蜡烛上量出了往下偏几厘米，补上就行；可换一个高度或方向，又得先错抓几次才能量出该补多少。自身模型的用意是训练一次，在任何位置都能直接伸准。所以不用模型的对手就是直接执行，模型的价值要靠覆盖多个高度、方向、姿态的评测来体现。
- **BiGym**：用转身横走后的站姿（左臂、腰）补采 M2 的数据，再登记重跑。夹不住杯子的原因要单独查，它不归自身模型管。

## 独立复核

2026-11-25，一个独立的 general-purpose agent 做的。它读了条目、两台机器上的原始输出和代码，没读 beliefs、memory 和其他 agent 的记录；两台机器上都只读，没有启动任何进程。

**判定**：
- **LocoManiLab"按判据算赢"：支持。** 但只限"在一个低处、不用走路的场景上，自身模型的命令比不加修正的直接命令成功更多"。
  - "走路、转向以后也能多成功"：证据不足，判定集里是 5 对 7。
  - "收益要靠模型，而不是一个固定修正"：证据不足，离线预演显示固定修正很可能拿回大部分。
- **BiGym"没赢，手也没有明显更准"：支持。** 结果描述是对的，但主度量几乎到底，没有检验力；"M2 推不到转身后的站姿"这个归因证据不足。

**我抽查过的**：BiGym 时限 bug 属实。`rows/` 里直接执行 30 行的 `budget_steps` 都是 1000，自身模型 30 行都是 1000000000；冻结副本 `bigym_hg.py` 第 509–510 行先读出时限，再把它改成 1e9。

需要更正的 9 处都已在原处标成"复核更正"或"复核补充"。复核原文如下：

> # selfmodel_hardgrasp_20261125 独立复核
>
> **结论先说：** LocoManiLab 的数字我都从逐局数据重算出来了，和条目一致，登记的"赢"成立，判定"支持"。但赢面有 10/17 来自蜡烛这一个场景，而且蜡烛上直接执行的偏差几乎是一个固定的下垂。"要靠自身模型、而不是一个固定修正"这一点证据不足。BiGym 的数字也对，"没赢、手没有明显更准"成立。
>
> 只读复核。两台机器上没有启动任何进程，远端没写文件，本地临时目录已删。
>
> ## 一、需要更正的地方
>
> 1. **第 578 行"走过去再抓 5 个任务 5 对 17（p = 0.017，`stats` 里各任务的配对数相加）"：数对，出处写错了，还漏了效应量。**
>    - 5 对 17 来自 `analysis_formal.txt` 的抖动局（0–29）：苹果 0/2、碗 0/0、盐瓶 0/5、哑铃 5/5、浇水壶 0/5。
>    - `stats_formal.txt` 含官方局，按它相加是 7 对 18，p = 0.043。
>    - 成功率是 24/150 对 36/150，只差 +8.0 个百分点，没到登记的 10 个百分点。17 对里有 10 对来自被排除的盐瓶和浇水壶。
>    - 判定集恰好也是 5 对 17，是巧合，容易让人混淆。第 527 行同样要改。
> 2. **第 556 行"在官方时限内完成：直接执行 15，自身模型 20，原因没查"：这是记录 bug，不是真差别。**
>    - 每个 worker 只建一个环境，同一个种子先跑直接执行组，再跑自身模型组（`bigym_hg.py:797-812`）。
>    - 第 510 行在直接执行那一局把 `env.bigym._time_limit` 改成了 1e9。第 509 行在自身模型那一局读到的就是 1e9。
>    - rows 里自身模型组 30 行的 `budget_steps` 全是 1000000000，直接执行组全是 1000。
>    - 按步数 ≤ 1000 重算，两组都是 15/20。`analysis.txt` 第 4 行也受影响。
> 3. **第 533–535 行"走路以后预测误差约是验证误差的 2.4 倍……只看范围内的局是 3.66 和 2.13 cm"：这些数是 7 个任务合起来算的，包括不走路的蜡烛和雕像。**
>    - 只看走过去的任务：直接执行 4.41 cm，自身模型 4.52 cm。2.4 倍碰巧不变。
>    - 只看范围内的局：直接执行 4.08 cm，自身模型 3.03 cm。
> 4. **第 522 行"合并的合手时手离目标中位数没有算"：这个登记的度量可以直接算。**
>    - 7 个任务：直接执行 8.80 cm（n 134），自身模型 4.09 cm（n 132）。
>    - 判定集：6.23 对 3.12 cm。
>    - 对照预测表的结论不变：直接执行在区间内，自身模型略高于区间。
> 5. **第 567 行"手离目标 −2.31 cm，n = 131"：这是 7 个任务全算的数，要标明。** 判定集是 −1.90 cm，区间 [−3.83, −0.70]，n 75。
> 6. **第 576 行"撞凳子从 10 局降到 0"：10 是失败分类里的局数。** 直接执行碰到凳子的局实际是 14 局，其中 4 局照样成功。自身模型是 0。
> 7. **第 537 行"感知中心偏了 8.3 cm"：只有哑铃是 8.3 cm，浇水壶是 10.3 cm。**
> 8. **第 536 行"撞家具 67 对 21"：还要补一句。** 两组各有 46 次撞家具的首次接触发生在厨房任务的中间点段。这一段两组都用直接命令，46 局完全相同，不是两组的差别。
>    - 首次接触所在段：直接执行是中间点 46、预抓取 53、抓取 24；自身模型是 46、41、7。
> 9. **第 595 行"所以只能说现成的 M2 推不到转身后的站姿"：按判据读成"手没有更准"在形式上没错，但这个原因没有被证明。**
>    - 超出训练范围的是左臂和腰。负责伸手的右臂只有 2/20 局超出。
>    - 合并差只有 −0.61 cm，主要原因是有 7 对因为 cuRobo 规划不出来，按规则自身模型组照搬直接执行，两组完全一样。
>
> ## 二、四件事
>
> ### 1. 核数字（从逐局数据独立重算）
>
> **LocoManiLab**（`episodes/*/*/ep*.json`，0–29 共 420 条）
>
> | 项目 | 重算 | 条目 |
> |---|---|---|
> | 判定集（蜡烛、苹果、哑铃，90 对） | 27 对 39，+13.3 个百分点，5 对 17，p = 0.0169 | 一致 |
> | 7 个任务全算（210 对） | 32 对 54，+10.5 个百分点，5 对 27，p = 0.000113 | 一致 |
> | 手离目标配对中位差 | 全部 −2.31 cm [−3.83, −1.11]，n 131；判定集 −1.90 cm [−3.83, −0.70]，n 75 | 一致 |
> | 去掉蜡烛 | 5 对 7，p = 0.774 | 一致 |
> | 加上官方局（31 对） | 判定集 6 对 19，p = 0.015；7 个任务 7 对 29，p = 0.0003 | — |
>
> **排除规则：**
> - 实现方式是两组合起来算"感知 + 站位/IK"失败的比例（`hg_analyze.py:82`），其中包括自身模型自己 IK 误差 > 3 cm 的局。这是一种说得通的读法。
> - 换其他读法：只看直接执行组、只算规划阶段的失败、或者把撞家具排在 IK 误差前面，浇水壶都会留下（36 %、30 %、33 %），雕像也会留下。判定集变成 5 对 22，p = 0.0015。
> - 也就是说，换哪种读法结论都不变弱，规则没有被用来偏袒自身模型。
> - "到顶"那条没有任务触发。
>
> **其他核对：**
> - 4 对错误记录都是蜡烛开局那一眼崩溃（`index 906 is out of bounds … size 906`），两组都记为"其他"。
> - 抖动回退到不抖动的局：0。
> - 冻结代码的 md5：远端现在的 `scripts/experiments` 用 `md5sum -c` 核对全部通过。
>
> **BiGym**（`rows/*.jsonl`，60 行）
> - 夹起成功：2 对 3，1 对 2，p = 1。
> - 手离目标（20 对两组都合了手）：直接执行 5.28 cm，自身模型 4.27 cm，配对中位差 −0.61 cm。自身模型更近 11 对、更远 2 对、相同 7 对，符号检验 p = 0.0225。
> - 自身模型真挪了终点的 13 对：−1.79 cm。
> - 失败分类和 `report.txt` 一致。开局状态哈希 60/60 一致，开局到站位的状态两组 30/30 相同。
>
> ### 2. 四类陷阱
>
> **重点一：决策里的仿真真值**
>
> LocoManiLab 里用到的真值，以及我的判断：
>
> | 用到的真值 | 两组共享吗 | 判断 |
> |---|---|---|
> | 那一眼锚在拍照时的仿真骨盆上，深度图没有噪声 | 共享 | 只是地图原点 |
> | 腿部里程计的朝向用仿真四元数（`hg_walk.py:389`） | 共享 | 所有记录里里程计和真值的朝向差都是 0.0000° |
> | 里程计的位置误差（抓取命令时） | 共享 | 蜡烛中位 0.09 cm，走过去的任务 2.6–2.8 cm，雕像 6.4 cm |
> | IK 规划锚在真实骨盆上（`lml_selfmodel.py:559-568`），命令按里程计的位置算 | 共享 | 里程计误差两组原样都有，模型修不了 |
> | 自身模型的输入用无噪声的本体信息，包括仿真骨盆高度（`lml_selfmodel.py:667`） | 只有自身模型组用 | 换到真机会变成估计值；但量级不足以造成蜡烛上 5 cm 的差别 |
> | 用真值判"跟丢"并中止 | 共享 | 自身模型组 4 次，直接执行组 1 次，对自身模型不利 |
>
> 总的判断：
> - 我没找到让某一组占便宜、能制造出这个结果的路径。
> - 但真值让两组共有的误差变得很小，蜡烛的定位误差几乎是 0。这样控制器自己的系统偏差（正是自身模型修的那部分）占总误差的比例被放大了。这是外推到真机的风险，不是组间偏向。
> - BiGym 照登记用骨盆读数，两组共享。
>
> **重点二：两组是否只差伸手命令**
> - LocoManiLab：
>   - 中间点和抬起两段，两组都用直接命令。代码里自身模型只在 k ≥ 0 时介入（`hg_run.py:474`），抬起段在第 529–542 行。
>   - 每对选的抓取完全相同，0 对不同。第一条命令时两组的骨盆位置差为 0.0 m（141 对）。
>   - 成功判定和中止规则两组相同。抬起都以"感知的目标点 + 0.45 m"为终点。
> - BiGym：自身模型组是用 cuRobo 对挪过的终点重新规划，所以路线和时长也变了。
>   - 620020 和 620029 两局手偏了 18 cm。它们的挪动量和其他成功局几乎一样（约 1.3、2.2、2.5 cm），偏差更像来自重新规划出的不同路线。
>   - 所以 BiGym 里"终点修正"和"换了一条路线"是混在一起的。
>
> **重点三："撞家具"的判定和分类顺序**
> - 撞家具只是一个标签，不中止这一局，也不影响成功。两组各有 25 个成功局在途中碰过家具。
> - IK 误差 > 3 cm 排在撞家具前面（`hg_run.py:591`）。效果是把自身模型组撞家具的局归到"站位/IK"，让"67 对 21"看起来差距更大；换个顺序是 98 对 69。
> - 在排除规则里，这个顺序反而把对自身模型有利的浇水壶排掉了。
> - 结论：这个顺序没有让直接执行组少算成功，只是夸大了"直接执行撞家具多"的说法。
>
> **另外几条**
> - **benchmark 选择：** 去掉的任务在设计局里都是两组 0 对 0，或者根本没跑通，和两组差别无关。但赢面靠一个场景：蜡烛占 10/17，去掉后 5 对 7。McNemar 把蜡烛的 18 个小抖动局当成独立证据，按任务算其实只有 1/3 个任务显出差别。设计局已经看到蜡烛 2 对 7，但正式种子是新抽的。
> - **数据泄漏：** 没找到。模型用空地伸手数据训练，正式种子第一次用，分析脚本只读数据。
> - **指标误用：** "手离目标"量的正是自身模型自己优化的那个手点，到感知出的目标的距离，所以天然对自身模型有利。主度量用成功率是对的。
> - **事后挑选：** 每边只有一次正式运行，事后分组都标了。只有"5 对 17"带着 p 值写进了结论，而它的效应量不够。
>
> ### 3. 推翻实验
>
> **离线预演（零成本，已做）：** 用设计局 1000–1009 里直接执行组的偏差中位数当常数。对蜡烛：
>
> | 伸手段 | 直接执行 | 设计局常数修正后 | 留一常数 | 自身模型 |
> |---|---|---|---|---|
> | 预抓取 | 6.0 cm | 1.6 cm | 1.8 cm | 0.7 cm |
> | 抓取 | 5.9 cm | 2.6 cm | 2.5 cm | 1.0 cm |
>
> - 这个预演假设偏差不随命令平移而变。蜡烛上这个假设基本成立：同一对两组的偏差只差 0.8–1.2 cm（中位）。
> - 直接执行在蜡烛上每一局都下垂 3–7 cm。成功的局要么下垂较小，要么 x 方向偏正。
> - 我预计一个固定修正就能拿回蜡烛上大部分的成功差。
> - 但模型并不只是一个常数：它的修正和逐局偏差相关性很高（x 方向 0.96、z 方向 0.97），还剩约 1.5 cm 的优势。
>
> **主推实验：加第三组"固定修正"（约 5 分钟 CPU，不用 GPU）**
> - **命令：** 感知的目标减去常数 c，c 按（伸手段、链序号、骨盆高度档）分，不看状态。
> - **常数从哪来：** 首选自身模型自己的训练数据 `lml_hostdata_20261118/data_scalebfm/all.npz`。取同链序号、骨盆高度 ±4 cm、目标附近的落点偏差中位数，也就是 `lml_tasks.lookup` 加一条骨盆高度匹配。
>   - 备选是设计局的常数。但设计局只有 7 局可用，而且可能是冻结前的代码跑的；要用就另跑新种子 2000–2009，只跑直接执行组。
>   - 任何情况下都不碰正式局 0–29 和官方局 −1。常数和代码 md5 写进新的修订以后再跑。
> - **怎么跑：** 读缓存好的那一眼，跑蜡烛、苹果、哑铃、盐瓶、浇水壶，局 −1..29，和已有的两组直接配对，三组的开局已核实完全一致。先重跑 2 局直接执行，确认结果和存档逐位相同。共 155 局，正式跑 434 局用了 7 分钟。
> - **判据：** 自身模型比固定修正高 ≥ 10 个百分点且 p < 0.05，才算"要模型"。否则结论改成"固定修正就够了"，尤其是固定修正在蜡烛上 ≥ 15/18 时。
> - **顺带的变体：** 在蜡烛上只加 z +5 cm，用来检验"修正的方向（抬高、离开凳子）而不是准度本身"这个解释。
> - 次要的检查：BiGym 加一组"重新规划但终点不挪"，把路线差异分出来；再用模型 s1 种子跑一遍蜡烛（`wm_s1.pt` 已有）。
>
> ## 三、判定
>
> **LocoManiLab"按判据算赢"：支持。**
> - 数字可以复现，两组只差伸手命令，仿真真值两组共享，排除规则换哪种读法结论都不变弱。
> - 范围限定为："在一个低处、不用走路的场景上，自身模型的命令比不加修正的直接命令成功更多"。
> - 走路、转向以后也能多成功：证据不足（判定集里是 5 对 7）。
> - 收益要靠模型而不是一个固定修正：证据不足，而离线预演显示固定修正很可能拿回大部分。
> - 建议先跑固定修正组，再投入给 SONIC 采数据。
>
> **BiGym"没赢，手也没有明显更准"：支持（结果描述正确）。**
> - 但主度量落在地板上（掉落 15 和 12 局），这个结果对"自身模型能不能带来更多成功"没有检验力。
> - "M2 推不到转身后的站姿"这个归因证据不足。
> - 时限那一条要按第一节第 2 条删掉或改正。
