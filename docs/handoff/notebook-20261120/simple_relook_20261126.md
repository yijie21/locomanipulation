# simple_relook_20261126：走到站位后再看一眼，按家具重新定位，站偏了补一步，再规划抓取

- **日期**：2026-11-26。预注册，在任何正式运行之前写好并提交。开发和冒烟已经跑过，见"开发"一节。协调人看过草稿后定了几件事，见末尾"协调人的决定"。
- **用户的决定**（11-26，协调人转达）：在 SIMPLE 的抓取流程里，走到站位后再看一眼，用家具点云配准重新定位；站位偏得多就补一步，再看再配准，最多 1–2 步；然后按新位置重新规划抓取。cuRobo 路线（J）和两列自身模型修正（S、kin_rr）都按新定位重算。用户的规则："任何新做法做出来当天就在 SIMPLE 里跑"。
- **要回答的问题**：
  - 第二眼加补步，能让 SIMPLE 主指标的四列（A、J、S、kin_rr）各多成功多少局？
  - 多出来的局，哪部分来自重新定位，哪部分来自补步？
- **机器**：不用 vastgpu。每一步开始前看 nvidia-smi，每卡总显存不超过 20 GB，不够就等（`run.sh` 里的 `gpu_wait`）。
  - **ScaleBFM**：4090（`ssh -p 27984 root@14.103.60.188`）的 1、3 号卡，0 号卡不碰。一个仿真进程约 0.6 GB，一个 cuRobo 规划进程约 1 GB（开发中量的，按整卡用量的增量）；规划一次最多 2 个。
  - **SONIC**：4090b（`ssh -p 62882 root@14.103.60.188`）的 CPU，网络放 CPU（`REACH_SONIC_NETS=allcpu`），最多 12 个仿真进程，nice 10。渲染和 cuRobo 按组分到 4090b 的 0、1 号卡。两卡开跑前各已用约 16.4–16.5 GB（HA 采集），只剩约 3.5 GB，所以规划和 sfull 两步会按显存门槛排队。
  - SONIC 不放 GPU 的原因：网络放 GPU 时一个进程约 7.4 GB，4090 的 1–3 号卡已用 16.3–17.4 GB（HA 采集），放不下。
- **代码**：分支 `sonic-selfmodel`，未提交的改动（只提交本条；代码以冻结副本 `asrun/` 为准，见"冻结"）：
  - `scripts/experiments/simple_eval_20261112/relook.py`：新文件，第二眼的看、配准。
  - `scripts/experiments/simple_eval_20261112/simple_bench.py`：加开关 `--relook` 及其参数，默认关。
  - `scripts/experiments/simple_eval_20261112/sfull_run.sh`：加两个环境变量。`SF_ARMS` 只跑部分修正列；`SF_BENCH_EXTRA` 给各列的 `simple_bench.py` 多传参数。不设时行为不变。
  - `scripts/experiments/simple_eval_20261112/bench_via.sh`：加环境变量 `BENCH_PLAN_PAR`，规划一次跑几个进程，默认 4，和原来一样。为的是每卡不超过 20 GB。
  - 新目录 `scripts/experiments/simple_relook_20261126/`：
    - `dev.sh`：开发跑局；
    - `smoke.sh`：整条流程的冒烟；
    - `idcheck.sh`、`idcmp.py`：开关关着时的逐位核对；
    - `devsum.py`：开发结果汇总；
    - `replay.py`：离线重放配准；
    - `relook_eval.py`：评估；`j0_eval.py`：J0 对照的评估；
    - `run.sh`：正式运行，一个控制器一份；
    - `py4090b_rl.py`、`mkrepo_b.sh`：在 4090b 上跑，见下。
    - `resume_scalebfm.sh`：ScaleBFM 的手动续跑（运行中加的，见"预测修订"第 1 条）；
    - `posthoc.py`：看到结果以后写的事后分析，见"结果"。
  - 4090 上只在私有副本 `/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126/repo`、`repo2` 里改，共享运行树没动。正式运行只从冻结副本 `asrun/` 跑。
  - **开关只在命令行**：`simple_bench.py --relook` 及其参数，没升成 `locomani.bench` 的选项（协调人的决定 3）。
- **关联**：
  - [simple_rgbd_20261124](simple_rgbd_20261124.md)：主指标，感知栏 `sensor=rgbd`，A、J 两列。
  - [selfmodel_sfull_20261125](selfmodel_sfull_20261125.md)：S（SF）、kin_rr 两列。其中"意外 3"：cuRobo 同样的输入，每次重规划的结果也不同。
  - [simple_sonic_vrref_20261126](simple_sonic_vrref_20261126.md)：SONIC 手部目标换坐标的主指标。主指标仍用 `"robot"`，本条也按 `"robot"`。

## 先测的问题（只读，已有主指标记录，11-26 已报协调人）

数据来自已有的主指标记录：`bench_rgbd_20261124`（A、J）和 `sfull_20261125`（SF、kin_rr）。两个控制器 × 4 个任务 × 30 局。

- **主指标复算一致**（45 秒内 / 官方时限内）：

  | 控制器 | A | J | S | kin_rr |
  |---|---|---|---|---|
  | ScaleBFM | 83/73 | 95/81 | 94/78 | 95/82 |
  | SONIC | 29/18 | 34/20 | 33/17 | 36/23 |

- **一局内各列走的是同一条路**：240 组里，每一局的 A、J、JR、SF、kin_rr 记的骨盆估计（`bel`）都逐位相同。
- **里程计带来的目标误差**：机器人以为的骨盆和真实骨盆之差，换算到抓取目标上，含高度。
  - ScaleBFM：中位数 0.19 cm，最大 0.83 cm。
  - SONIC：中位数 0.91 cm，90 分位 5.98 cm，最大 17.1 cm。
  - SONIC 的高度差另算（以为的骨盆高度减真实高度），中位数按任务是 0.85–2.83 cm，最大 5.77 cm，TabletopGrasp 最大。原因是骨盆下沉了，里程计还按站立高度算。
- **站位偏差**：真实站位和计划站位之差。
  - ScaleBFM：中位数 1.9 cm。TabletopGrasp 普遍偏右约 7.5 cm，最大 10.1 cm。
  - SONIC：中位数 19.6 cm，120 局里 92 局超过 10 cm。
    - TabletopGrasp：中位数 26 cm，主要是横向 −25 cm，朝向偏 −5.7°。30 局里有 23 局目标落在骨盆左侧。
    - XMovePick：往前冲过头 15.6 cm，目标只剩 14 cm 远。
- **SONIC 失败局按两项分组**：A 的 91 局失败里，
  - 两项都小：16 局；
  - 只是站偏：47 局；
  - 只是里程计偏：1 局；
  - 两项都大：27 局。
- **ScaleBFM 的失败和定位无关**。
- **结论**：SONIC 的站偏大多是它自己的里程计也看得见的偏差（以为的站位就离计划站位很远），是走路控制的问题，不是定位的问题。所以重新定位本身只能修高度和少数几局的水平偏差，要靠补步才能修站位。

## 假设

1. **家具配准能把骨盆定准，不依赖里程计**。第二眼和第一眼都看得到桌子、墙这些不动的东西，配准误差在 1 cm、0.3° 以内，比 SONIC 的里程计准得多。
2. **ScaleBFM 上几乎不变**。它的里程计已经很准，站位偏差最大约 10 cm，不到补步阈值，所以只有小于 1.3 cm 的定位修正。四列的变化应该落在噪声里。这一组是"不帮倒忙"的检查。
3. **SONIC 的增益主要来自补步**。它的失败主要跟站偏走，补一步能把站偏从约 20 cm 降到 10 cm 以内。重新定位只能修目标高度（TabletopGrasp 中位数 2.8 cm）和少数里程计大偏的局，增益小。补步的限制在于 SONIC 小步走得乱，详见开发结果。

## 做法

### 1. 第二眼：看和配准（`relook.py`）

- **什么时候看**：主流程走到站位、里程计给出骨盆估计以后，伸手之前。
- **看**：头部相机在 MuJoCo 复制场景里渲一张深度图（`see.head_frame`）。相机和标定与第一眼相同。渲染时仿真不前进，和第一眼一样不计时间。
  - 像素点按关节读数（`see.HeadKin`）放到骨盆坐标系，再按重力读数（IMU）转成水平坐标系。
  - 水平位置和朝向用机器人以为的骨盆位姿。
  - **骨盆高度用双脚的正运动学**：关节读数加重力读数，算出最低的脚底点，取它到骨盆的高度，不用里程计的 z。这是开发中发现的问题，见"开发"。
- **只留家具**：两眼都去掉以下几类点：
  - 机器人自身：按关节读数做正运动学，每个几何体的包围球外扩 3 cm；
  - 地面：高度低于 4 cm；
  - 远点：离相机超过 3 m。

  另外，第一眼按 SAM 3 的掩码（`cloud_blind` 的 `mask`）去掉目标；第二眼去掉离第一眼目标点 4 cm 以内的点，目标点按当前估计放置。
- **配准**：把第二眼的家具配到第一眼上。
  - 方法：点到面 ICP，求 x、y、z 和绕竖轴的转角（横滚、俯仰由重力读数给出），Huber 权重，对应点半径逐步收紧（12 cm 收到 2 cm）。
  - 先验：里程计作为先验，标准差为 x、y、z 各 5 cm，转角 3°。这样，家具约束不住的方向（比如沿一条长桌边）会停在里程计附近，不会滑走。
  - 粗搜：ICP 之前，在 x、y ±24 cm（步长 3 cm）、转角 ±4°（步长 1°）上做一次粗搜，按截断距离打分。
  - 输出取整到 0.1 mm 和 1e-5 rad，保证同一局的 A、J、S 走同样的步。
- **之后**：机器人的位姿估计换成配准结果，绑定到当时的真实骨盆。这和现在走完以后绑定里程计的做法相同（`place.Belief`），只是"以为的位置"换成了配准结果。

### 2. 补步

- **判断**：按配准后的位姿算出目标在身体坐标系里的位置，和计划站位下目标该在的位置比较。两者差超过 **12 cm** 就补一步。
- **怎么走**：
  - 走回计划站位：相对当前估计的位移，单步最长 0.40 m，朝向转回计划朝向，最多转 ±20°。
  - 步法同主流程的 `gen_reach.open_walk`：保持朝向走、最后转身、停止修正关掉。
  - 结束时刻不读仿真真值：路线走完后再给 settle 0.5 s 加 3 s。SONIC 另有一条规则：它自己的规划器连续 0.5 s 给出 IDLE（也就是它自己的骨盆估计已在目标 3 cm 以内）就提前结束。
  - 步内用腿部里程计更新位姿估计，走完再看一眼、再配准。
- **最多 2 步**，即最多看 3 眼。
- **SONIC 补第一步之前先收手**（`--relook-tuck 0.8 --relook-tuck-hold 1.0`，只用在 SONIC 上）：
  - 站着不动，用 0.8 s 把两臂的参考关节平滑地转到"手在髋旁"的姿势，再站 1.0 s，然后才走；
  - 姿势：肩俯仰 +0.30 rad（向后）、肩横滚 ±0.30 rad（向外）、肘 0.60 rad，其余为 0；
  - 收手期间腿部里程计照常计入，走路时参考关节保持收手姿势，伸手从这个姿势开始；
  - 来源：协调人提出的开发局小试，结果见"开发"的"收手再走"一段。ScaleBFM 照草稿，不收手：它在 12 cm 阈值下几乎不补步，失败也不是手搭桌面造成的。
- **已知的风险**：开发局第 3 级上，SONIC 补步时两手搭在桌上，补步越补越偏（见"开发"）。收手以后第 3 级明显好转，但手、腕、髋和桌子的接触仍出现在 9–10/10 局里，第一步也变得不准，多数局要补第二步。正式局里 TabletopGrasp 的站偏和第 3 级差不多大，可能遇到同样的问题。这一条写进判据。
- **补步的世界坐标目标**按仿真骨盆来算，和 `open_walk` 一样。电机的路线从那里起，它的骨盆估计也在上一次站定时锚在那里，所以传到控制器的只有相对位移。

### 3. 不用仿真真值

- **决策只用这些输入**：深度图、关节读数、重力读数、腿部里程计、第一眼的目标掩码。
- **真值只进记录**：以 `_true` 结尾的字段，包括估计误差、站位误差、补步实际走了多少、场景里别的东西有没有被碰动。
- **一处已知的真值用法，不是本条引入的**：SF 列的状态重建（`s_check.frames`）用了真实的 `qpos0`，见 selfmodel_sfull_20261125。本条照旧，结论里注明。

### 4. 重新规划（四列都按新定位）

- **A**：按新定位直线伸手。
- **J**：
  - 从带第二眼的 A 录像重新规划：`bench_via.sh plan`。`prep2.py` 取 A 录像里预抓取那一帧做起点，取 A 那一行的 `bel` 做定位，这时的 `bel` 已是最后一次配准的结果。
  - 然后跑 JR（J 加 `--film-ref`），带同样的 `--relook` 参数，所以伸手前走的是同一条路。
  - JR 和 J 的结果相同：旧主指标两个控制器各 120 局，成功与否、手的误差、成功时刻全部一致。所以本条只跑 JR，当 J 用。
- **S、kin_rr**：`sfull_run.sh`，`SF_ARMS="SF kin_rr"`，`SF_BENCH_EXTRA` 给同样的 `--relook` 参数，只在新 JR 里 cuRobo 给了路线的局上跑；其余局取 J 的行，规则和主指标相同。
- **评估**：用 `relook_eval.py`，不用 `table_sfull`，因为后者写死了五列。

### 5. 组别（按局配对）

两个控制器 × 4 个任务 × 30 局，和主指标是同一批局、同样的种子。

**ScaleBFM**：带第二眼的各列新跑，不带的复用主指标的记录。

| 列 | 带第二眼（新跑） | 不带（复用已有记录） |
|---|---|---|
| A | `A`，`--relook` | `bench_rgbd_20261124/scalebfm/rows/A_*` |
| J | `JR`，`--relook`，按新录像重规划 | `bench_rgbd_20261124/scalebfm/rows/J_*` |
| S | `SF`，`--relook`，在新 JR 给了路线的局上跑，其余取 J | `sfull_20261125/scalebfm/rows/SF_*`，其余取 J |
| kin_rr | 同上 | `sfull_20261125/scalebfm/rows/kin_rr_*`，其余取 J |

**SONIC**：两组都在 4090b 的 CPU 上新跑，在同一次运行里配对，不复用旧的 GPU 记录（协调人的决定 1）。
- 网络放 CPU 和放 GPU，结果不逐位相同，所以 CPU 上"不带第二眼"那一组（`none`）就是这次的基线：A、规划、JR、SF、kin_rr，都不带 `--relook`。
- "带第二眼"那一组（`relook`）跑同样的五步，都带 SONIC 的参数（含收手）。
- 主指标表里 SONIC 那一行照旧用 GPU 的老结果。这次的对比自成一组，不和主指标的数直接相减。
- 另外记一份 CPU `none` 对 GPU 老记录的比较（`eval_cpu_vs_gpu.txt`），只作记录，不进判据。

外加两个对照：

- **只看不补（SONIC，只跑 A，CPU）**：SONIC 的参数加 `--relook-steps 0`，即只重新定位，不补步，也就不收手。和 CPU `none` 的 A 比，用来把重新定位的贡献和补步的贡献分开。ScaleBFM 在 12 cm 阈值下几乎不补步，这一组和带第二眼的 A 基本相同，所以不跑。
- **只重规划（J0，两个控制器）**：不带第二眼，从"不带"那组的 A 录像再规划一遍，再跑 JR。ScaleBFM 用主指标的 A 录像，SONIC 用 CPU `none` 的 A 录像。用来量 cuRobo 每次重规划的噪声。J、S、kin_rr 带不带第二眼的差里，都混着这份噪声；A 不经过规划器，没有这份噪声。

**复用旧记录的前提**（只有 ScaleBFM 复用）：开关关着时输出逐位不变。

- ScaleBFM 已核对，用冻结副本 `asrun/repo` 又核对了一遍。每个任务取第 3、24 局跑 A 和 J，TabletopGrasp 取第 4、24 局跑 SF 和 kin_rr。结果：A 8/8、J 8/8、SF 2/2、kin_rr 2/2，行（去掉 `wall_s`）和录像都逐位相同，见"开发"。
- SONIC 不复用旧记录，所以不需要和 GPU 记录核对。

### 6. 度量

- **主指标**：每列 45 秒内和官方时限内的成功局数，按任务分开。
  - 配对检验：只有带第二眼成功的局数和只有不带成功的局数，McNemar 精确检验。
  - 按布局再做一次符号检验：同一任务的 10 个序号算一个布局，因为第 0、1 级共用布局，第 2 级只挪 1–6 cm。
- **定位**（只记录，和仿真真值比）：
  - 里程计误差、每一眼的配准误差（cm、度）；
  - 补步前后的站位误差；
  - 补了几步、补步花了多少时间；
  - 配准的内点比例、残差；
  - 场景里有没有别的东西被碰动；
  - 看和补步期间机器人与场景的接触：行里 `hits` 的 relook 段，哪个机器人部件碰了哪个物体。
  - **补步时目标物体有没有被撞动**（协调人 11-26 加的要求，起因是 LML 诊断：SONIC 在哑铃、浇水壶上 84–85 % 的局里物体在手伸过去之前就被撞走，大部分发生在修正步里）：每一眼记目标从第一眼起的位移（`target_moved_true_cm`），相邻两眼之差就是那一步撞动的量。每局都报；超过 1 cm 算"撞动"，列出局号、每眼的位移，以及这一局带不带第二眼的 A 是否成功。另记其他物体的最大位移（`scene_moved_true_cm`）。`relook_eval.py` 第 6 节。物体真值只当度量，不进任何决策。
- **核对**：
  - 同一局的 A、JR、SF、kin_rr，第二眼的记录（墙钟时间除外）和最后的位姿估计必须相同；
  - 旧记录里不能有第二眼的字段；
  - 摔倒局数。
- **失败分组**：带第二眼那一列的失败局，按补步后的真实站位误差（10 cm）和最后一眼的定位误差（2 cm）分组。
- **路线变化**：新旧 J 各在哪些局给了路线，两边都有的，只有一边有的。

## 开发（TabletopGrasp 开发局 30–49，即开发树的第 3、4 级，和正式局不重合）

开发树是 `outputs/simple_dev_20261124`。开发阶段 SONIC 的网络放 CPU（`allcpu`），所以数字和正式运行不逐位可比。

### 开关关着：逐位核对（ScaleBFM，已过）

（`idcheck.sh scalebfm "1 3"`，私有副本，`idcmp.py` 对比 `bench_rgbd_20261124/scalebfm` 和 `sfull_20261125/scalebfm`）

```
A: rows the same 8/8, films the same 8/8
J: rows the same 8/8, films the same 8/8
SF: rows the same 2/2, films the same 2/2
kin_rr: rows the same 2/2, films the same 2/2
```

### 配准

- **ScaleBFM**（`ro_sb`，20 局 34 眼）：
  - 配准误差中位数 0.05 cm，90 分位 0.30 cm，最大 0.47 cm；转角最大 0.07°。
  - 作为先验的里程计误差：中位数 0.11 cm，最大 1.27 cm。
  - 约束标准差：x 约 0.03 cm，y 约 0.1 cm，转角约 0.05°。内点约 0.99，残差约 5 mm。
- **SONIC 第一眼**：
  - 改高度之前，30–39 这一组里有两局配错了 40–50 cm，内点只有 12–14 %。
  - 原因：SONIC 骨盆下沉了 5–6 cm，里程计还按站立高度算，地面被抬到了 4 cm 截线以上，几万个地面点淹没了家具。
  - 离线重放：旧的落盘数据没存双脚高度，只能用真实高度代替，作为诊断。高度对了以后，这两局的误差降到 0.2 cm 以内。
  - 在线：骨盆高度改用双脚正运动学（sF、sG，15 局第一眼）：
    - 双脚给的高度误差 0.08–0.35 cm，里程计 z 的误差是 0.6–5.7 cm；
    - 第一眼配准误差 0.10–1.24 cm，转角 ≤ 0.25°，全部正确。
- **SONIC 补步之后那几眼**（sG，在线，有高度修正）：
  - 第 4 级：12 眼里 1 眼误差超过 5 cm；
  - 第 3 级：20 眼里 7 眼超过 5 cm。

  失败分两种：
  - 补步后里程计偏了 20–45 cm，接近或超出粗搜范围 ±24 cm；
  - 里程计只偏 1–4 cm，配准却跑到 10–13 cm 外，而且内点 0.89–0.94。场景没被碰动（`scene_moved_true_cm` ≤ 0.17 cm）。第二种的原因没查清，可能是桌面附近有几何上近似对称的结构。
- **放宽配准参数的重放**（旧落盘数据，高度用真实值代替）：
  - 粗搜放宽到 ±36 cm：反而多出一个第一眼的错配，错到 40.7 cm，内点 0.93。说明 40 cm 外就有几乎一样好的对齐，±24 cm 已经是安全的上限。
  - 先验放宽到 10 cm、5°：不比原来好。
  - 去掉粗搜：错得更多。
  - 所以配准参数维持原样。

### 补步的步法

SONIC，20 局，只走到第二眼加补步为止，不伸手（`--relook-only`）。下表是站位误差，单位 cm，中位数 [90 分位]；没注明局数的是每组 10 局。

| 变体 | 30–39 补步前 → 后 | 40–49 补步前 → 后 |
|---|---|---|
| 固定时间，转身，结束前多给 1 s（`so_d`，3 局） | 22 → 43 | — |
| 固定时间，不转身，多给 1 s（sC） | 30.1 → 38.1 | 16.4 → 6.6 [29.5] |
| 先站稳 1 s，转身（sA） | 31.9 → 32.5（9 局） | 16.4 → 8.3 [34.5] |
| 先站稳 1 s，不转身（sB） | 30.1 → 18.6 [51.6] | 16.4 → 10.0 [18.6] |
| SONIC 空闲即结束，最多多给 3 s，不转身（sD） | — | 16.4 → 7.3 [15.7] |
| SONIC 空闲即结束，最多多给 3 s，转身（sE，尚无高度修正） | — | 16.4 → 6.1 [11.7] |
| sE 加高度修正（sG） | 30.1 → 33.5 [42.0] | 16.4 → 4.7 [13.5] |
| sG 改成先转向再走（`--relook-face travel`，sH） | 3 局：21.2→16.8、22.0→54.9、34.5→27.6，每局多花 17–19 s | 2 局：16.0→44.6、20.3→35.3 |
| sG 加先后退 15 cm 再走（`--relook-back 0.15`，sI） | 1 局：21.2→47.4，多花 22.5 s，仍碰桌 | 1 局：16.0→7.4，多花 21.9 s |

sH、sI 各跑了 1–3 局就停了：明显更差、更慢。

sG 的两组：

- **第 4 级（40–49）**：10 局里 9 局补了步。补步后，配准误差超过 5 cm 的只有 1/12 眼。第二眼加补步一共多花 4.4–11.2 s，中位数 5.25 s。
- **第 3 级（30–39）**：10 局全补了 2 步，站位反而变差。补步后，配准误差超过 5 cm 的有 7/20 眼。每局多花 10.1–11.8 s。

第 3 级的问题：

- 场景里没有东西被碰动（`scene_moved_true_cm` ≤ 0.17 cm）。
- 逐拍看，SONIC 起步后先往反方向窜，接着在原地停住 2 s 多，规划器一直给 0.40 m/s 的命令，人却不动，像被什么挡住了。
- **原因是两只手搭在桌上**：dbg3 记录了补步时机器人和场景的接触。第 30、32 局补步时，两手的手指、腕部，以及右肘，都和 `table` 接触。补步拖着手走，走不动，也走不直。
- 所以试了"先后退 15 cm 离开桌子再走"（sI）。

- **SONIC 的小步为什么乱**：逐拍记录（`dump_dbg`）显示，同一个命令下：
  - 起步的头 0.5–1 s，常常先往命令以外的方向窜 10–30 cm。例如命令左移 20 cm，它先后退 27 cm。
  - 之后才朝目标走。
  - 原来的固定结束时间（路线时间加 0.5–1.5 s），常常在它还在走的时候就截断了。

  改成"它自己的规划器空闲了才结束"以后，第 4 级明显变好。第 3 级仍差，主要原因是手搭在桌上（上面 sG 一段）；改高度之前，还有几局配准错了，补步被带去了错的地方。
- **ScaleBFM 的小步走不动**：8–10 cm 的横移几乎不动，偏差反而略增（例如第 31 局 8.7 → 10.2 → 10.4 cm），每步还要花约 3 s。所以阈值取 12 cm：开发局里它的最大偏差是 10.6 cm，主指标里是 10.1 cm，都不到阈值，它就不补步。

### 收手再走（SONIC，协调人提出的小试，11-26 07:52–08:10Z，限时 30 分钟）

- **做法**：补第一步之前，站着把两臂转到"手在髋旁"的姿势（`tuck_arms`，见"做法"第 2 节），再走。其余同 sG。
- **判断规则**（协调人定的）：只看开发局 30–49。第 3 级变好、第 4 级不变差，就写进做法；否则不加。
- **两个变体**：
  - sJ：收手 0.8 s，之后站 0.3 s 就走；
  - sK：收手 0.8 s，之后站 1.0 s 再走。sJ 看下来，第 4 级第一步常常走偏，所以试了多站一会儿。
- **结果**（`--relook-only`，每组 10 局，站位误差 cm，中位数 [90 分位]，最大；"≤10" 是补步后站位误差不超过 10 cm 的局数；"配准 >5" 是补步后那几眼里配准误差超过 5 cm 的眼数）：

  | 变体 | 30–39 补步后 | ≤10 | 配准 >5 | 多花的时间 | 40–49 补步后 | ≤10 | 配准 >5 | 多花的时间 |
  |---|---|---|---|---|---|---|---|---|
  | sG（不收手） | 33.5 [41.2]，49.2 | 1 | 7/20 | 11.30 s | 4.7 [13.1]，17.0 | 7 | 1/12 | 5.24 s |
  | sJ（收手，站 0.3 s） | 12.9 [45.0]，65.3 | 3 | 2/20 | 11.86 s | 7.0 [10.7]，23.3 | 8 | 0/18 | 11.29 s |
  | sK（收手，站 1.0 s） | 13.4 [33.2]，51.5 | 3 | 5/19 | 12.69 s | 3.7 [7.5]，8.0 | 10 | 0/17 | 11.27 s |

  逐局（补步后站位误差，sG/sJ/sK）：
  - 30–39：30: 38/45/5，31: 24/3/14，32: 35/7/33，33: 32/65/13，34: 23/6/9，35: 20/15/7，36: 49/12/51，37: 41/13/18，38: 8/18/26，39: 40/11/11。sK 比 sG 好 8 局、差 2 局。
  - 40–49：40: 6/6/8，41: 5/8/3，42: 10/5/7，43: 2/3/1，44: 3/6/5，45: 3/9/4，46: 17/11/6，47: 3/3/3，48: 4/23/4，49: 13/10/2。
- **代价**：收手以后第一步变得不准（第 4 级第一步后的站位误差，sG/sK：40: 6/18，44: 3/20，45: 3/17），多数局要补第二步。第 4 级补两步的局从 3/9 变成 8/9，多花的时间中位数从 5.24 s 变成 11.27 s。这会压低官方时限内的成功数，45 秒内的主指标受影响小。
- **接触没有消失**：sJ、sK 两组里，补步阶段机器人和 `table` 的接触仍出现在 9–10/10 局里，部件有手指、腕、髋、骨盆。所以"手离开桌面"并没有完全做到，站位变好的原因可能不只是手不再拖在桌上，也可能是手臂姿势改变了 SONIC 起步的方式。没有再细查。
- **补步时目标有没有被撞动**（协调人后来加的要求，起因是 LML 诊断；目标从第一眼起的位移，真值只当度量）：
  - sG 的记录里还没有这个字段，所以用同样的参数重跑了一遍（sG2，08:23–08:29Z）。sG2 每局每一眼的站位误差和 sG 全部相同（20/20），说明后加的记录代码不改变仿真。
  - 不收手（sG2）：20 局里 0 局撞动（> 1 cm），最大 0.10 cm。
  - sJ：0/20，最大 0.10 cm。
  - sK：1/20 撞动。第 32 局第二步把目标撞走 18.89 cm（三眼依次 0.01、0.06、18.89 cm）。另有第 36 局 0.52 cm。
  - 其他物体（`scene_moved_true_cm`）三组都 ≤ 0.22 cm。
  - 行里的 `hits` 只记机器人和桌子等其他物体的接触，不含目标，所以目标有没有被碰只能看位移。
  - 收手以后撞动目标多了 1 局（20 局里），局数太少，看不出是不是收手造成的。正式运行每局都报（`relook_eval.py` 第 6 节），判据里另加一行。
- **结论**：按协调人的规则，sK 写进做法。第 3 级明显变好（中位数 33.5 → 13.4 cm）；第 4 级站位不差（4.7 → 3.7 cm，10 局都在 8 cm 以内），只是多花约 6 s，另有 1 局撞走目标（第 3 级）。sJ 不用：第 4 级的尾巴更差（最大 23.3 cm）。
- 限时到了，没有再试更长的站立时间或别的收手姿势。

### 4090b 上跑 SONIC

- 4090b 没有 `/workspace` 软链，`.venv_sim` 的解释器在那边起不来。
- `py4090b_rl.py` 照 `py4090b.py`（selfmodel_simplecollect_20261126）按 vePFS 的真实路径重建 sys.path，另外支持 bench 脚本里用到的 `-I`、`-m`、`-c` 和从标准输入读代码这几种调用方式。
- `mkrepo_b.sh` 建一份给 4090b 用的副本：
  - `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python` 换成调用 `py4090b_rl.py` 的两行脚本，所以 `bench_via.sh`、`sfull_run.sh` 不用改；
  - 场景 `simple_g1_sonic`、`walk_mars` 链到 selfmodel_simpledata_20261126 那两份把 `/workspace/` 换成真实路径的副本。
- 另设 `LOCOMANI_WORKSPACE`、`SONIC_POLICY` 两个环境变量，照 simpledata。
- **核对**：同一份代码，SONIC 网络放 CPU，带第二眼跑开发局第 30、40 局的 A，4090（`repo2`）和 4090b（`repo_b`）两边的行（去掉 `wall_s`）逐位相同，2/2。
- **cuRobo**：SIMPLE 的 venv 在 vePFS 上，4090b 上能直接跑。冒烟的第 41 局在 4090b 上规划成功（`ok: true`）。一个规划进程约占 1 GB 显存（按整卡用量的增量量的）。

### 冒烟（整条流程）

ScaleBFM，开发局第 40、41 局，GPU 1。为了走到补步，阈值临时取 5 cm。命令：

```
smoke.sh sb scalebfm 1 40 41 "--relook --relook-step-cm 5 --relook-extra-s 3 --relook-idle-end"
```

整条流程跑通：A → 规划（`obstacles_box`、`prep2`、`plan2`、`via2`）→ JR → `sfull_run.sh`（SF、kin_rr）→ `relook_eval.py`，没有进程失败。`table_sfull` 按预期失败，因为它写死了五列。`relook_eval.py` 的核对部分原文：

```
3. checks
  JR: the relook record and final belief equal A's in 2/2 episodes
  SF: the relook record and final belief equal A's in 1/1 episodes
  kin_rr: the relook record and final belief equal A's in 1/1 episodes
  old rows carrying a relook record: 0; relook A rows without one: 0
  falls: relook []; old []
5. routes (J ran a plan)
  relook 1, old 0, both 0, only relook [('TabletopGraspMP', 41)], only old []
```

另外看到的：

- 两局都补了 2 步，每局多花约 10 s。ScaleBFM 的小步还是走不动：站位误差 7.6 → 9.6 cm。
- 这两局不带第二眼时 A 就失败（手离目标 32–38 cm，目标在可达范围外）。J 两边各成功 1 局。
- 结果只说明流程能跑通，不说明效果。

**SONIC 的冒烟**（4090b，CPU，从冻结副本 `asrun/repo_b` 跑，开发局第 40、41 局，参数是 `RL_sonic`，含收手，08:13–08:23Z）：

```
SMOKE_REPO=asrun/repo_b SONIC_NETS=allcpu smoke.sh sbK sonic 0 40 41 "$(cat asrun/RL_sonic)"
```

- 整条流程在 4090b 上跑通：A、规划（cuRobo，SIMPLE 的 venv）、JR、`sfull_run.sh`（状态重建、`shifts2` 用到 torch 模型、规划、SF 和 kin_rr）、`relook_eval.py`，没有进程失败。`table_sfull` 照旧按预期失败。
- 核对：JR 2/2、SF 1/1、kin_rr 1/1 的第二眼记录和最终估计都和 A 相同；falls 为空。
- 两局都补了 2 步，站位误差中位数 18.15 → 5.49 cm，多花 10.82 s（中位数）；配准内点 0.99，第一眼误差 0.65 cm，最后一眼 0.40 cm。
- 目标没有被撞动（第 6 节：0/2），补步时两局都和 `table` 有接触。
- 收手以后的伸手能跑完：A 1/2 成功（开发树里 GPU 版不带第二眼是 0/2），J、S、kin_rr 各 1/2（GPU 版 2/2）。两局，只说明能跑，不说明效果。
- 单进程时一局 A 约 80 s。

### 定下的参数（正式运行用）

冻结在 `asrun/RL_scalebfm`、`asrun/RL_sonic`：

```
ScaleBFM: --relook --relook-step-cm 12 --relook-steps 2 --relook-max-step 0.4 --relook-extra-s 3 --relook-idle-end
SONIC:    --relook --relook-step-cm 12 --relook-steps 2 --relook-max-step 0.4 --relook-extra-s 3 --relook-idle-end --relook-tuck 0.8 --relook-tuck-hold 1.0
```

- **其余选项取默认**：
  - 转身：转回计划朝向，最多 ±20°；
  - 保持朝向走：`--relook-face fixed`；
  - 不先后退：`--relook-back 0`；
  - 不先站稳：`--relook-settle-s 0`。
- **配准参数**：`relook.py` 里的常数，`FEET_Z=True`，粗搜 ±24 cm，先验 5 cm、3°。
- **为什么是这组**：即 sG，SONIC 再加收手（sK）。第 4 级上 sG 最好；第 3 级上不收手的变体都不行，原因是手搭在桌上；收手以后第 3 级好转，第 4 级不差（见"收手再走"）。
- **"只看不补"对照**：SONIC 的参数加 `--relook-steps 0`。

**开关关着的核对，用冻结副本重做**：ScaleBFM，`asrun/repo`，08:11–08:13Z，同样的局（`idcheck_final/scalebfm.txt`）：

```
A: rows the same 8/8, films the same 8/8
J: rows the same 8/8, films the same 8/8
SF: rows the same 2/2, films the same 2/2
kin_rr: rows the same 2/2, films the same 2/2
```

**最终代码相对开发中途的改动**：都在 `--relook` 分支里，或者是新加的、默认不生效的选项：
- 双脚高度；
- SONIC 空闲即结束；
- 补步的朝向选项；
- 先后退选项；
- 收手选项（`--relook-tuck`、`--relook-tuck-hold`，默认 0，不收手）；
- 补步时记录与场景的接触；
- 记录场景是否被碰动；
- 开发用的落盘；
- `bench_via.sh` 的 `BENCH_PLAN_PAR`（默认 4，同原来）。

### 冻结

`/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126/asrun/`，FROZEN_UTC 2026-11-26T08:23:52Z。08:11:06Z 第一次冻结后改过两次，都在任何正式运行之前，每次重做 `MD5.check`：
- 08:14:50Z：`run.sh` 把 SONIC 各组分到两张卡上；
- 08:23:52Z：`relook_eval.py` 加第 6 节（目标和场景在看和补步期间的位移），按协调人的要求。
- `repo/`：给 4090 用，开发副本 `repo2` 的整份拷贝。src、scripts、configs、pyproject 是实拷，assets、outputs 下的各目录、third_party 链到原处。src 是 4090 运行树的版本，不是 Mac 工作区的版本：Mac 上 `options.py`、`paths.py`、`bench/simple/__init__.py` 正被别的会话改，没进这次运行。
- `repo_b/`：给 4090b 用，`mkrepo_b.sh` 从 `repo/` 建的，代码逐个文件相同。
- `MD5.check` 一共 3405 个文件，md5 `27da80d7`。`run.sh` 开始和结束时各核对一遍。

| 文件 | md5 |
|---|---|
| simple_eval_20261112/simple_bench.py | adb4af64 |
| simple_eval_20261112/relook.py | dc3e48bd |
| simple_eval_20261112/sfull_run.sh | ff12ad13 |
| simple_eval_20261112/bench_via.sh | 448e604e |
| simple_relook_20261126/run.sh | 760a9b26 |
| simple_relook_20261126/relook_eval.py / j0_eval.py | d5833947 / a71baf97 |
| simple_relook_20261126/py4090b_rl.py / mkrepo_b.sh | 49c82ebc / f67dab84 |
| RL_scalebfm / RL_sonic | 6f376fc1 / 9f35913e |

### 正式运行的步骤

`run.sh <ctrl>`，一个控制器一份，从冻结副本跑。每一步开始前等到所用的卡"已用 + 这一步要用"不超过 20 GB。

- **ScaleBFM**（4090，1、3 号卡，`nohup nice -n 10 bash asrun/repo/.../run.sh scalebfm`）：
  1. `relook` 组：`bench_via.sh` 的 A、plan、JR（带 `RL_scalebfm`），再 `sfull_run.sh`（`SF_ARMS="SF kin_rr"`，`SF_BENCH_EXTRA` 同上，`ARM_PAR=2`，`PLAN_PAR=2`）。三个 json 和 sfull_20261125 用的相同（`outputs/simple_dev_20261124/precision_const/`）。
  2. `j0` 组：链接主指标的 `films_A`、`rows_ep`，跑 plan 和 JR，不带 `--relook`。
  3. 评估：`relook_eval.py scalebfm <relook bench> <relook sfull> bench_rgbd_20261124/scalebfm sfull_20261125/scalebfm J`；`j0_eval.py <j0 bench> bench_rgbd_20261124/scalebfm J`。
- **SONIC**（4090b，CPU，`nohup nice -n 10 bash asrun/repo_b/.../run.sh sonic`）：
  1. 同时起三组：`none`（0 号卡，五步都不带 `--relook`）、`relook`（1 号卡，五步都带 `RL_sonic`）、`look_only`（0 号卡，只跑 A，`RL_sonic --relook-steps 0`）。最多 12 个仿真进程。
  2. `look_only` 和 `none` 的 A 都跑完后，起 `j0`（1 号卡）：从 `none` 的 A 录像再规划一遍，跑 JR。
  3. 评估：`relook_eval.py sonic <relook> <none> JR`（主比较）；`look_only` 对 `none`（只看 A 列）；`j0_eval.py <j0> <none> JR`；`none` 对 GPU 老记录（只作记录）。
- 环境照主指标：`BENCH_SET=sensor=rgbd`，`SIMPLE_ROOT` 是 4090 上的 SIMPLE，正式局（`SIMPLE_EVAL_OUT`、`SIMPLE_LEVELS` 不设）。规划一次 2 个进程（`BENCH_PLAN_PAR=2`）。
- `relook_eval.py` 的第 6 节给出每局补步时目标有没有被撞动。

## 预测（启动前写，首次启动后锁定）

**ScaleBFM**（45 秒内；旧数 A 83、J 95、S 94、kin_rr 95）

- **补步**：120 局里 0–5 局补步。
- **配准误差**：中位数 ≤ 0.3 cm，90 分位 ≤ 1 cm，转角 ≤ 0.2°。
- **A**：79–87 局，带不带的差在 ±4 局以内，McNemar p > 0.05。
  - 理由：位姿估计只动不到 1.3 cm，变化来自混沌放大，个别局翻转。
- **J**：90–98；**S**：89–97；**kin_rr**：90–98。
  - 理由：同上，再加上重规划的噪声。
- **官方时限内**：各列和旧数相差 ±5 局以内。
- **J0**（只重规划）：90–98，和旧 J 相差 ±4 局以内。

**SONIC**（CPU，`relook` 对 `none`，同一次运行里配对；GPU 老数是 A 29、J 34、S 33、kin_rr 36，官方时限内 A 18）

- **`none` 组本身**（只作记录，不进判据）：网络放 CPU 和放 GPU 不逐位相同，每局会各自分叉，但总数应该和 GPU 老数相近。45 秒内 A 22–36、J 26–42、S 25–41、kin_rr 28–44；官方时限内 A 12–24。
- **补步**：120 局里 70–110 局至少补一步。
- **补步后的真实站位误差**：中位数 5–15 cm（`none` 的 A 预计和旧的 19.6 cm 相近），10 cm 以内的局数 40–95。
  - 区间很宽，因为开发局两级差别很大：收手以后第 4 级 16.4 → 3.7 cm，第 3 级 30.1 → 13.4 cm。正式局像哪一级，事先不知道。
- **第一眼配准**：误差中位数 ≤ 0.5 cm，超过 2 cm 的不超过 5 %。
- **补步后那几眼**：误差超过 5 cm 的占 0–25 %。
- **A**：`relook` 比 `none` 多 −3 到 +18 局。
  - 理由：旧主指标里站偏 ≤ 10 cm 的局成功 11/28（39 %），> 10 cm 的成功 18/92（20 %）。补步若把一半的局带进 10 cm 以内，按这两个比率约多 6 局，再加上高度修正；收手让第 3 级那类局也能补正，上限放宽。但配错的局、补步时仍碰桌的局会补到更偏的地方，可能比原来还差。另外伸手从收手姿势开始，和 `none` 的起手姿势不同，这一点开发局没有试过伸手。
  - 分任务（`relook` 减 `none`）：
    - TabletopGrasp：−5 到 +12，站偏最大，也最可能碰桌；
    - XMovePick：0 到 +8，冲过头，要往后退，不贴桌；
    - BendPick、XMoveBendPick：各 −3 到 +4。
- **J、S、kin_rr**：各比 `none` 多 −5 到 +18 局。
  - 理由：每列的增量和 A 相近，再加重规划噪声。
- **官方时限内**：A 比 `none` 少 12 局到多 3 局。
  - 理由：开发局上第二眼加收手加补步多花 7–13.5 s（只补一步的第 4 级局也要约 7 s），旧的成功时刻中位数约 12 s，补过步的局大多超过 16 s，BendPick 超过 8 s。所以官方时限内的增量远小于 45 秒内，很可能为负。
- **补步时撞动目标**（> 1 cm，第 6 节）：`relook` 组 120 局里 0–12 局，多数在 TabletopGrasp。
  - 理由：开发局上收手的两组 40 局里只有 1 局撞动（sK 第 32 局，18.89 cm），另 1 局 0.52 cm；SIMPLE 的目标在桌面中间，不像 LML 那样贴在手边。但 LML 上 SONIC 的修正步撞走物体很常见，正式局里 TabletopGrasp 站得近，可能更多。
- **ScaleBFM 补步时撞动目标**：0–2 局（它几乎不补步）。
- **只看不补的 A**：比 `none` 多 −3 到 +6 局。
  - 理由：只修高度（TabletopGrasp 中位数 2.8 cm）和少数里程计大偏的局，不收手也不补步。
- **J0**：和 `none` 的 J 相差 ±6 局以内。

## 判据（什么结果会让我放弃或修改假设）

| 结果 | 结论和下一步 |
|---|---|
| ScaleBFM 冻结副本的开关关着核对不是全部相同（已过），或 `relook_eval` 的核对失败（同一局各列的第二眼记录、最终估计不一致，或旧记录里有第二眼字段） | 先停下查清，失败那部分的结果作废 |
| ScaleBFM 的 A 带不带第二眼差 ≥ 6 局（任一方向） | 第二眼不只是"看"，扰动了机器人或流程。先查清再看 SONIC 的数 |
| SONIC 的 A，`relook` 比 `none` 多 ≥ 10 局，且补步后站位误差中位数 ≤ 10 cm | 补步有效，假设 3 成立。主指标给 SONIC 加一行"走后再看"（注明是 CPU 上的配对对比）。下一步：把同样的补步用到 HumanoidArena 的 SONIC 上；查官方时限内的损失，把补步做快 |
| SONIC 的 A，`relook` 比 `none` 多 3–9 局 | 部分有效。按任务和"只看不补"拆开，看增量来自定位还是补步，以及剩下的失败局站位是否已经正了 |
| SONIC 补步后站位误差中位数 ≤ 10 cm，但 A 多 < 3 局 | 站位不是瓶颈，假设 3 不成立。SONIC 的失败在合手本身（敲到目标、合手时手在晃）。停止在站位上投入，转去查 SONIC 的手（vrref 的手速结果） |
| SONIC 补步后站位误差中位数 > 12 cm | 步法在正式局上不灵，补步的结论作废。看补步时的接触记录（行里 `hits` 的 relook 段）：收手已经在做法里，如果多数局仍是手或身体贴着桌子，下一步试先退离桌子再补、或换收手姿势，回开发局定；只按"只看不补"下关于定位的结论 |
| SONIC 某个任务 `relook` 的 A 比 `none` 少 ≥ 4 局 | 补步在这个任务上帮倒忙。照实报，按接触记录、配准误差和伸手起始姿势（收手）查是哪一种。阈值或步法只能回开发局改，不在正式局上调 |
| SONIC `relook` 组补步时撞动目标（> 1 cm）的局 ≥ 10 | 补步的收益有一部分被撞走目标吃掉。照实报，撞动的局不剔除。按任务和接触记录看是哪个部件撞的；下一步回开发局试补步时离目标保持距离（例如先退后再横移，或补步路线避开目标的方位），和 LML 诊断的结论合并考虑 |
| 补步后那几眼里，配准误差超过 5 cm 的超过 20 % | 配准对补步后的场景不可靠。加可靠性门槛（内点、残差、和里程计是否相容），回开发局再定，正式结果只按第一眼报 |
| J、S、kin_rr 的增减都落在 J0 量出来的噪声以内，A 却显著变化 | 照实报：规划器噪声盖住了这三列的差别，只有 A 下结论 |

## 预测修订

预测和判据都没改。下面是运行中的改动和查证，只影响怎么跑，不影响结果。

**1. ScaleBFM 手动续跑，显存门槛按实测改**（协调人同意，2026-11-26 10:07Z）
- **起因**：`run.sh` 在 SF、kin_rr 这一步之前，要求 1、3 号卡各空出 2.9 GB。两卡当时已用 17.7、17.9 GB（HA 和 lmlroute 的进程），从 09:26:54Z 起一直等。这个门槛定得偏保守。
- **实测的需要**（都在本条的开发中量的）：
  - 一个 cuRobo 规划进程约 1.0 GB：4090b 0 号卡跑一次 `plan2.py`，整卡用量 8493 → 9527 MiB；
  - 一个 ScaleBFM 仿真进程 0.82–0.95 GB：带第二眼的 A 跑着时，nvidia-smi 列出的四个进程是 858、878、818、953 MiB。
- **按实测定的门槛**：
  - SF、kin_rr 这一步：规划所在的卡 2.3 GB（两个规划进程，外加状态重建和 `shifts2` 的 torch）；另一张卡 1.0 GB（`ARM_PAR=2`，SF 在第一张卡，kin_rr 在第二张卡，每张卡一次一个仿真进程）；
  - J0 的规划：一张卡 2.3 GB；
  - J0 的 JR：四个仿真进程，每个按 1.0 GB，按 1、3 号卡的余量分。
- **规则**：每一步开跑前重读 nvidia-smi，按"当前已用 + 这一步要用 ≤ 20480 MiB"判断；较重的部分优先放 3 号卡；不够就等，每 2 分钟看一次；0 号卡不碰，不动别人的进程。
- **怎么做的**：按 pid 停掉了在等待中的 `run.sh`（1115146 和它的子 shell 1115153），这时它没有任何子进程在跑。随后起 `resume_scalebfm.sh`（md5 `d87d546e`，在 `$X/` 和 `scripts/experiments/simple_relook_20261126/` 下）。它用的是 `run.sh` 的原命令、冻结副本 `asrun/repo`、`RL_scalebfm` 和同样的环境变量，只换了显存门槛，冻结的 `run.sh` 没改。
- **结果**：10:07:13Z 起跑，SF、kin_rr 一步的卡是 "3 1"，规划在 3 号卡。

**2. 共享场景文件被误改的查证**（协调人要求，只读）
- **事件**：另一个实验在 09:53:17–09:55:38Z 误改了共享树 `outputs/scene` 下的 5 个文件，把 `/workspace` 改成 vePFS 路径，之后已逐字节恢复。
- **ScaleBFM**：这段时间没有进程在跑，`run.sh` 在等显存。
- **SONIC**：
  - 当时在跑不带第二眼组和 J0 组的 JR，以及带第二眼组的规划；
  - 每局的场景由 SIMPLE 自带的机器人生成，写在 `outputs/simple_eval_20261112/scene/`；
  - `asrun/repo_b` 的 `simple_g1_sonic`、`walk_mars` 是 simpledata 冻结的副本，修改时间 01:39:46Z，没被动过；
  - `nv_g1_43dof`、`simple_g1_gap` 这条流程不用。
- **核对**：10:04Z 手动核对 `MD5.check`，全部一致；没有失败记录，日志里没有 Traceback。
- **结论**：两组都不受影响。

## 结果（贴原始输出，不转述）

**运行记录**
- **ScaleBFM**：4090 的 1、3 号卡，2026-11-26 08:29:21Z 起跑，11:10:22Z 跑完（`STATUS` 为 done）。10:07Z 起手动续跑，见"预测修订"第 1 条。
- **SONIC**：4090b 的 CPU，08:31:55Z 起跑，11:23:04Z 跑完（`STATUS` 为 done）。
- **核对**：两边开始和结束时 `MD5.check` 都是 ok；没有 `FAILED` 文件；评估输出里没有 Traceback。
- **sfull 日志里的 `FAIL table_sfull`**：`table_sfull` 只认五列齐全，本条只跑 SF、kin_rr 两列，所以这行是预料中的，`sfull_run.sh` 照常返回 0。
- **显存**：每一步开跑前都核对过。4090 上的最高值是 3 号卡 19544 MiB（10:12:56Z，sfull 规划中）和 1 号卡 19688 MiB（11:05Z，J0 的 JR 有 3 个仿真进程在 1 号卡上），都没超过 20480 MiB。这两个峰值是会话里手动读 nvidia-smi 得到的，不在 `run.log` 里；`run.log` 只记了每一步开跑时的读数。0 号卡没用。
- **原始输出**（`$X = /wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126`）：
  - `$X/scalebfm/eval_relook.txt`、`eval_j0.txt`、`run.log`；
  - `$X/sonic/eval_relook.txt`、`eval_look_only.txt`、`eval_j0.txt`、`eval_cpu_vs_gpu.txt`、`run.log`；
  - 事后分析 `$X/sonic/posthoc.txt`（不在预注册里，见下）。
- **每局的行**：
  - ScaleBFM：`$X/scalebfm/{relook,j0}/bench/rows/`、`$X/scalebfm/relook/sfull/rows/`；
  - SONIC：`$X/sonic/{none,relook,look_only,j0}/bench/rows/`、`$X/sonic/{none,relook}/sfull/rows/`。
- **评估命令**：照 `run.sh` 和 `resume_scalebfm.sh`，从 `asrun/repo` 用 `.venv_sim/bin/python -I` 跑：
  - `relook_eval.py scalebfm $X/scalebfm/relook/bench $X/scalebfm/relook/sfull $OLD/bench_rgbd_20261124/scalebfm $OLD/sfull_20261125/scalebfm J`
  - `j0_eval.py $X/scalebfm/j0/bench $OLD/bench_rgbd_20261124/scalebfm J`
  - `relook_eval.py sonic $X/sonic/relook/bench $X/sonic/relook/sfull $X/sonic/none/bench $X/sonic/none/sfull JR`
  - `relook_eval.py sonic $X/sonic/look_only/bench $X/sonic/none/sfull_absent $X/sonic/none/bench $X/sonic/none/sfull JR`
  - `j0_eval.py $X/sonic/j0/bench $X/sonic/none/bench JR`
  - `relook_eval.py sonic $X/sonic/none/bench $X/sonic/none/sfull $OLD/bench_rgbd_20261124/sonic $OLD/sfull_20261125/sonic J`
  - 其中 `$OLD = /wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112`。
- **怎么读输出**：
  - 第 1 节每格是"带第二眼 vs 对照"，括号里依次是"只有带第二眼成功 / 只有对照成功"的局数和 McNemar p，再是布局层面"升 / 降"的个数和符号检验 p。
  - 第 4 节 `stance<loc<` 指补步后真实站位误差 ≤ 10 cm 且最后一眼定位误差 ≤ 2 cm，`>` 是超过。
  - SONIC 的 S、kin_rr 只在 JR 给了路线的局上跑，其余局取 J；ScaleBFM 同样。

### ScaleBFM（带第二眼 vs 主指标的老记录）

`eval_relook.txt` 第 1 节：

```
1. successes, relook vs old (only relook / only old, McNemar p over episodes; layouts up/down, sign p)
  45 s
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop           28 vs  28 (0/0 p 1.000; 0/0 p 1.000)    28 vs  28 (0/0 p 1.000; 0/0 p 1.000)    28 vs  28 (0/0 p 1.000; 0/0 p 1.000)    28 vs  28 (0/0 p 1.000; 0/0 p 1.000)
    XMoveBendPickTeleop       25 vs  27 (0/2 p 0.500; 0/2 p 0.500)    25 vs  27 (0/2 p 0.500; 0/2 p 0.500)    25 vs  27 (0/2 p 0.500; 0/2 p 0.500)    25 vs  27 (0/2 p 0.500; 0/2 p 0.500)
    BendPickMP                10 vs  10 (0/0 p 1.000; 0/0 p 1.000)    10 vs  10 (0/0 p 1.000; 0/0 p 1.000)    10 vs  10 (0/0 p 1.000; 0/0 p 1.000)    10 vs  10 (0/0 p 1.000; 0/0 p 1.000)
    TabletopGraspMP           17 vs  18 (0/1 p 1.000; 0/1 p 1.000)    30 vs  30 (0/0 p 1.000; 0/0 p 1.000)    29 vs  29 (0/0 p 1.000; 0/0 p 1.000)    30 vs  30 (0/0 p 1.000; 0/0 p 1.000)
    all                       80 vs  83 (0/3 p 0.250; 0/3 p 0.250)    93 vs  95 (0/2 p 0.500; 0/2 p 0.500)    92 vs  94 (0/2 p 0.500; 0/2 p 0.500)    93 vs  95 (0/2 p 0.500; 0/2 p 0.500)
    episodes missing in the relook run: {'A': 0, 'J': 0, 'S': 0, 'kin_rr': 0}
  official limit
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop           28 vs  28 (0/0 p 1.000; 0/0 p 1.000)    28 vs  28 (0/0 p 1.000; 0/0 p 1.000)    28 vs  28 (0/0 p 1.000; 0/0 p 1.000)    28 vs  28 (0/0 p 1.000; 0/0 p 1.000)
    XMoveBendPickTeleop       25 vs  27 (0/2 p 0.500; 0/2 p 0.500)    25 vs  27 (0/2 p 0.500; 0/2 p 0.500)    25 vs  27 (0/2 p 0.500; 0/2 p 0.500)    25 vs  27 (0/2 p 0.500; 0/2 p 0.500)
    BendPickMP                 0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    TabletopGraspMP           17 vs  18 (0/1 p 1.000; 0/1 p 1.000)    26 vs  26 (0/0 p 1.000; 0/0 p 1.000)    23 vs  23 (0/0 p 1.000; 0/0 p 1.000)    26 vs  27 (0/1 p 1.000; 0/1 p 1.000)
    all                       70 vs  73 (0/3 p 0.250; 0/3 p 0.250)    79 vs  81 (0/2 p 0.500; 0/2 p 0.500)    76 vs  78 (0/2 p 0.500; 0/2 p 0.500)    79 vs  82 (0/3 p 0.250; 0/3 p 0.250)
    episodes missing in the relook run: {'A': 0, 'J': 0, 'S': 0, 'kin_rr': 0}
```

第 2 节（只贴 `all`）：

```
2. localization and stance (record only; cm, deg): median [90th percentile] max
  all  n 120, steps taken [120, 0, 0]
    odometry xy         0.10 [  0.57]   0.78 (n 120)   yaw   0.19 [  0.46]   0.65 (n 120)
    first look xy       0.05 [  0.11]   1.92 (n 120)   yaw   0.01 [  0.03]   0.10 (n 120)
    last look xy        0.05 [  0.11]   1.92 (n 120)   yaw   0.01 [  0.03]   0.10 (n 120)
    stance before       1.91 [  7.79]  10.10 (n 120)   yaw   0.25 [  1.31]   2.74 (n 120)
    stance after        1.91 [  7.79]  10.10 (n 120)   yaw   0.25 [  1.31]   2.74 (n 120)
    off (robot's own)   2.06 [  7.92]  11.65 (n 120) ->   2.06 [  7.92]  11.65 (n 120)
    time added, s       0.00 [  0.00]   0.00 (n 120)
    registration: inliers   0.99 [  1.00]   1.00 (n 240), rms mm   5.04 [  5.30]   5.54 (n 240), sd y cm   0.11 [  0.14]   0.16 (n 240)
```

第 3–6 节：

```
3. checks
  JR: the relook record and final belief equal A's in 120/120 episodes
  SF: the relook record and final belief equal A's in 28/28 episodes
  kin_rr: the relook record and final belief equal A's in 28/28 episodes
  old rows carrying a relook record: 0; relook A rows without one: 0
  falls: relook []; old []

4. failures of the relook arm (45 s), by the true stance error after the steps (> 10 cm) and the last look's error (> 2 cm): fail / episodes
  A: XMovePickTeleop: stance<loc< 2/30 | XMoveBendPickTeleop: stance<loc< 5/30 | BendPickMP: stance<loc< 20/30 | TabletopGraspMP: stance<loc< 11/27 stance>loc< 2/3 | all: stance<loc< 38/117 stance>loc< 2/3
  J: XMovePickTeleop: stance<loc< 2/30 | XMoveBendPickTeleop: stance<loc< 5/30 | BendPickMP: stance<loc< 20/30 | TabletopGraspMP: stance<loc< 0/27 stance>loc< 0/3 | all: stance<loc< 27/117 stance>loc< 0/3
  S: XMovePickTeleop: stance<loc< 2/30 | XMoveBendPickTeleop: stance<loc< 5/30 | BendPickMP: stance<loc< 20/30 | TabletopGraspMP: stance<loc< 1/27 stance>loc< 0/3 | all: stance<loc< 28/117 stance>loc< 0/3
  kin_rr: XMovePickTeleop: stance<loc< 2/30 | XMoveBendPickTeleop: stance<loc< 5/30 | BendPickMP: stance<loc< 20/30 | TabletopGraspMP: stance<loc< 0/27 stance>loc< 0/3 | all: stance<loc< 27/117 stance>loc< 0/3

5. routes (J ran a plan)
  relook 28, old 28, both 28, only relook [], only old []

6. the target and the scene during the looks and steps (truth, record only; cm since the first look)
  XMovePickTeleop  n 30, stepped 0; target moved > 1 cm in 0
    other bodies moved > 1 cm: 0 
    touched in the relook phase (body: episodes): 
  XMoveBendPickTeleop  n 30, stepped 0; target moved > 1 cm in 0
    other bodies moved > 1 cm: 0 
    touched in the relook phase (body: episodes): 
  BendPickMP  n 30, stepped 0; target moved > 1 cm in 0
    other bodies moved > 1 cm: 1 [(24, [3.96, 'blue_screwdriver'])]
    touched in the relook phase (body: episodes): 
  TabletopGraspMP  n 30, stepped 0; target moved > 1 cm in 0
    other bodies moved > 1 cm: 2 [(4, [1.04, 'tape']), (23, [1.04, 'tape'])]
    touched in the relook phase (body: episodes): 
  all  n 120, stepped 0; target moved > 1 cm in 0
    other bodies moved > 1 cm: 3 [(24, [3.96, 'blue_screwdriver']), (4, [1.04, 'tape']), (23, [1.04, 'tape'])]
    touched in the relook phase (body: episodes):
```

`eval_j0.txt`（只重规划，对主指标的 J）：

```
J0 rows 120, old J rows 120, common 120
  45 s
    XMovePickTeleop        n  30  J0  28  old  28  only J0  0  only old  0  McNemar p 1.000
    XMoveBendPickTeleop    n  30  J0  27  old  27  only J0  0  only old  0  McNemar p 1.000
    BendPickMP             n  30  J0  10  old  10  only J0  0  only old  0  McNemar p 1.000
    TabletopGraspMP        n  30  J0  30  old  30  only J0  0  only old  0  McNemar p 1.000
    all                    n 120  J0  95  old  95  only J0  0  only old  0  McNemar p 1.000
  official limit
    XMovePickTeleop        n  30  J0  28  old  28  only J0  0  only old  0  McNemar p 1.000
    XMoveBendPickTeleop    n  30  J0  27  old  27  only J0  0  only old  0  McNemar p 1.000
    BendPickMP             n  30  J0   0  old   0  only J0  0  only old  0  McNemar p 1.000
    TabletopGraspMP        n  30  J0  26  old  26  only J0  0  only old  0  McNemar p 1.000
    all                    n 120  J0  81  old  81  only J0  0  only old  0  McNemar p 1.000
  routes: J0 28, old 28, both 28, only J0 [], only old []
```

### SONIC（CPU，`relook` vs `none`，同一次运行里配对）

`eval_relook.txt` 第 1 节：

```
1. successes, relook vs old (only relook / only old, McNemar p over episodes; layouts up/down, sign p)
  45 s
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop          10 vs   7 (10/7 p 0.629; 6/3 p 0.508)   10 vs   7 (10/7 p 0.629; 6/3 p 0.508)   10 vs   7 (10/7 p 0.629; 6/3 p 0.508)   10 vs   7 (10/7 p 0.629; 6/3 p 0.508)
    XMoveBendPickTeleop        4 vs   4 (2/2 p 1.000; 2/2 p 1.000)     4 vs   4 (2/2 p 1.000; 2/2 p 1.000)     4 vs   4 (2/2 p 1.000; 2/2 p 1.000)     4 vs   4 (2/2 p 1.000; 2/2 p 1.000)
    BendPickMP                14 vs   6 (8/0 p 0.008; 6/0 p 0.031)    14 vs   6 (8/0 p 0.008; 6/0 p 0.031)    14 vs   6 (8/0 p 0.008; 6/0 p 0.031)    14 vs   6 (8/0 p 0.008; 6/0 p 0.031)
    TabletopGraspMP           3 vs  13 (3/13 p 0.021; 3/7 p 0.344)    13 vs  11 (9/7 p 0.804; 4/4 p 1.000)   11 vs  16 (7/12 p 0.359; 3/6 p 0.508)    12 vs  16 (4/8 p 0.388; 3/6 p 0.508)
    all                    31 vs  30 (23/22 p 1.000; 17/12 p 0.458) 41 vs  28 (29/16 p 0.072; 18/9 p 0.122) 39 vs  33 (27/21 p 0.471; 17/11 p 0.345) 40 vs  33 (24/17 p 0.349; 17/11 p 0.345)
    episodes missing in the relook run: {'A': 0, 'J': 0, 'S': 0, 'kin_rr': 0}
  official limit
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop            0 vs   7 (0/7 p 0.016; 0/4 p 0.125)     0 vs   7 (0/7 p 0.016; 0/4 p 0.125)     0 vs   7 (0/7 p 0.016; 0/4 p 0.125)     0 vs   7 (0/7 p 0.016; 0/4 p 0.125)
    XMoveBendPickTeleop        2 vs   4 (0/2 p 0.500; 0/2 p 0.500)     2 vs   4 (0/2 p 0.500; 0/2 p 0.500)     2 vs   4 (0/2 p 0.500; 0/2 p 0.500)     2 vs   4 (0/2 p 0.500; 0/2 p 0.500)
    BendPickMP                 0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    TabletopGraspMP           0 vs  13 (0/13 p 0.000; 0/7 p 0.016)     1 vs   9 (1/9 p 0.021; 0/7 p 0.016)    0 vs  11 (0/11 p 0.001; 0/8 p 0.008)    0 vs  13 (0/13 p 0.000; 0/9 p 0.004)
    all                      2 vs  24 (0/22 p 0.000; 0/13 p 0.000)   3 vs  20 (1/18 p 0.000; 0/13 p 0.000)   2 vs  22 (0/20 p 0.000; 0/14 p 0.000)   2 vs  24 (0/22 p 0.000; 0/15 p 0.000)
    episodes missing in the relook run: {'A': 0, 'J': 0, 'S': 0, 'kin_rr': 0}
```

第 2 节：

```
2. localization and stance (record only; cm, deg): median [90th percentile] max
  XMovePickTeleop  n 30, steps taken [0, 5, 25]
    odometry xy         0.60 [  1.22]   4.83 (n 30)   yaw   0.58 [  1.24]   2.15 (n 30)
    first look xy       0.20 [  0.36]   0.47 (n 30)   yaw   0.08 [  0.19]   0.53 (n 30)
    last look xy        0.24 [  0.61]   0.87 (n 30)   yaw   0.07 [  0.26]   0.32 (n 30)
    stance before      20.00 [ 25.02]  26.90 (n 30)   yaw   2.59 [  5.57]   9.10 (n 30)
    stance after       12.11 [ 42.66]  51.83 (n 30)   yaw   3.07 [  8.19]  11.62 (n 30)
    off (robot's own)  20.10 [ 24.92]  27.15 (n 30) ->  12.14 [ 41.11]  52.63 (n 30)
    time added, s      12.43 [ 13.21]  13.34 (n 30)
    registration: inliers   0.98 [  1.00]   1.00 (n 60), rms mm   5.07 [  5.39]   5.99 (n 60), sd y cm   0.13 [  0.18]   0.30 (n 60)
  XMoveBendPickTeleop  n 30, steps taken [10, 5, 15]
    odometry xy         2.53 [ 13.35]  21.13 (n 30)   yaw   0.56 [  1.66]   6.57 (n 30)
    first look xy       0.34 [  0.50]   0.66 (n 30)   yaw   0.08 [  0.18]   0.54 (n 30)
    last look xy        0.29 [  0.43]   0.54 (n 30)   yaw   0.05 [  0.12]   0.37 (n 30)
    stance before      15.33 [ 26.74]  43.95 (n 30)   yaw   2.97 [  5.79]   8.04 (n 30)
    stance after        9.99 [ 24.81]  39.77 (n 30)   yaw   3.36 [  7.02]   8.79 (n 30)
    off (robot's own)  14.84 [ 27.06]  47.33 (n 30) ->   9.30 [ 22.80]  40.34 (n 30)
    time added, s       8.82 [ 12.73]  14.04 (n 30)
    registration: inliers   0.98 [  0.99]   1.00 (n 60), rms mm   5.24 [  5.52]   6.35 (n 60), sd y cm   0.13 [  0.18]   0.43 (n 60)
  BendPickMP  n 30, steps taken [13, 8, 9]
    odometry xy         0.29 [  1.38]   2.02 (n 30)   yaw   0.40 [  0.86]   1.11 (n 30)
    first look xy       0.28 [  0.69]   0.98 (n 30)   yaw   0.07 [  0.23]   0.34 (n 30)
    last look xy        0.22 [  0.52]  30.74 (n 30)   yaw   0.04 [  0.21]   1.44 (n 30)
    stance before      12.90 [ 23.75]  26.05 (n 30)   yaw   3.57 [  9.29]  13.57 (n 30)
    stance after        5.96 [ 12.95]  59.85 (n 30)   yaw   3.26 [  7.95]  10.83 (n 30)
    off (robot's own)  15.76 [ 26.90]  31.28 (n 30) ->   6.08 [ 13.12]  23.65 (n 30)
    time added, s       6.59 [ 12.26]  12.38 (n 30)
    registration: inliers   0.99 [  1.00]   1.00 (n 60), rms mm   4.83 [  5.15]   6.05 (n 60), sd y cm   0.12 [  0.15]   1.07 (n 60)
  TabletopGraspMP  n 30, steps taken [1, 7, 22]
    odometry xy         2.51 [  9.79]  20.58 (n 30)   yaw   1.06 [  1.53]   2.95 (n 30)
    first look xy       0.12 [  1.67]  14.49 (n 30)   yaw   0.06 [  0.17]   5.62 (n 30)
    last look xy        0.17 [ 20.73]  60.60 (n 30)   yaw   0.08 [  1.78]  49.44 (n 30)
    stance before      27.46 [ 35.80]  38.57 (n 30)   yaw   6.19 [ 10.43]  12.01 (n 30)
    stance after       12.64 [ 32.28]  67.98 (n 30)   yaw   2.03 [  7.69]  13.56 (n 30)
    off (robot's own)  29.95 [ 40.00]  41.97 (n 30) ->  12.54 [ 28.70]  44.90 (n 30)
    time added, s      12.31 [ 13.48]  13.86 (n 30)
    registration: inliers   0.96 [  0.99]   1.00 (n 60), rms mm    nan [   nan]    nan (n 60), sd y cm   0.18 [  0.77]    inf (n 60)
  all  n 120, steps taken [24, 25, 71]
    odometry xy         0.97 [  7.52]  21.13 (n 120)   yaw   0.53 [  1.46]   6.57 (n 120)
    first look xy       0.23 [  0.55]  14.49 (n 120)   yaw   0.07 [  0.21]   5.62 (n 120)
    last look xy        0.22 [  0.63]  60.60 (n 120)   yaw   0.06 [  0.28]  49.44 (n 120)
    stance before      19.46 [ 29.00]  43.95 (n 120)   yaw   3.43 [  9.10]  13.57 (n 120)
    stance after        9.76 [ 29.17]  67.98 (n 120)   yaw   3.18 [  7.89]  13.56 (n 120)
    off (robot's own)  19.41 [ 33.62]  47.33 (n 120) ->   9.33 [ 26.06]  52.63 (n 120)
    time added, s      11.58 [ 13.16]  14.04 (n 120)
    registration: inliers   0.98 [  1.00]   1.00 (n 240), rms mm    nan [   nan]    nan (n 240), sd y cm   0.13 [  0.24]    inf (n 240)
```

第 3–5 节：

```
3. checks
  JR: the relook record and final belief equal A's in 120/120 episodes
  SF: the relook record and final belief equal A's in 18/18 episodes
  kin_rr: the relook record and final belief equal A's in 18/18 episodes
  old rows carrying a relook record: 0; relook A rows without one: 0
  falls: relook [('TabletopGraspMP', 17)]; old []

4. failures of the relook arm (45 s), by the true stance error after the steps (> 10 cm) and the last look's error (> 2 cm): fail / episodes
  A: XMovePickTeleop: stance<loc< 7/11 stance>loc< 13/19 | XMoveBendPickTeleop: stance<loc< 13/16 stance>loc< 13/14 | BendPickMP: stance<loc< 10/22 stance>loc< 5/7 stance>loc> 1/1 | TabletopGraspMP: stance<loc< 11/13 stance>loc< 9/10 stance>loc> 7/7 | all: stance<loc< 41/62 stance>loc< 40/50 stance>loc> 8/8
  J: XMovePickTeleop: stance<loc< 7/11 stance>loc< 13/19 | XMoveBendPickTeleop: stance<loc< 13/16 stance>loc< 13/14 | BendPickMP: stance<loc< 10/22 stance>loc< 5/7 stance>loc> 1/1 | TabletopGraspMP: stance<loc< 2/13 stance>loc< 9/10 stance>loc> 6/7 | all: stance<loc< 32/62 stance>loc< 40/50 stance>loc> 7/8
  S: XMovePickTeleop: stance<loc< 7/11 stance>loc< 13/19 | XMoveBendPickTeleop: stance<loc< 13/16 stance>loc< 13/14 | BendPickMP: stance<loc< 10/22 stance>loc< 5/7 stance>loc> 1/1 | TabletopGraspMP: stance<loc< 4/13 stance>loc< 9/10 stance>loc> 6/7 | all: stance<loc< 34/62 stance>loc< 40/50 stance>loc> 7/8
  kin_rr: XMovePickTeleop: stance<loc< 7/11 stance>loc< 13/19 | XMoveBendPickTeleop: stance<loc< 13/16 stance>loc< 13/14 | BendPickMP: stance<loc< 10/22 stance>loc< 5/7 stance>loc> 1/1 | TabletopGraspMP: stance<loc< 2/13 stance>loc< 9/10 stance>loc> 7/7 | all: stance<loc< 32/62 stance>loc< 40/50 stance>loc> 8/8

5. routes (J ran a plan)
  relook 18, old 29, both 16, only relook [('BendPickMP', 10), ('BendPickMP', 24)], only old [('TabletopGraspMP', 2), ('TabletopGraspMP', 5), ('TabletopGraspMP', 6), ('TabletopGraspMP', 9), ('TabletopGraspMP', 16), ('TabletopGraspMP', 17), ('TabletopGraspMP', 18), ('TabletopGraspMP', 19), ('TabletopGraspMP', 22), ('TabletopGraspMP', 25), ('TabletopGraspMP', 26), ('TabletopGraspMP', 28), ('XMovePickTeleop', 24)]
```

第 6 节（最后一行 `all` 截断）：

```
6. the target and the scene during the looks and steps (truth, record only; cm since the first look)
  XMovePickTeleop  n 30, stepped 30; target moved > 1 cm in 9 (ep, cm at each later look, relook A ok, old A ok): [(1, [0.0, 17.06], False, True), (6, [0.0, 10.88], False, True), (8, [0.0, 1.52], True, False), (9, [0.0, 17.58], False, True), (10, [0.0, 17.25], False, False), (12, [0.0, 17.54], False, False), (21, [0.0, 17.12], False, False), (25, [0.0, 16.16], False, False), (26, [67.99], False, False)]
    other bodies moved > 1 cm: 0 
    touched in the relook phase (body: episodes): table 13
  XMoveBendPickTeleop  n 30, stepped 20; target moved > 1 cm in 5 (ep, cm at each later look, relook A ok, old A ok): [(7, [17.15, 17.15], False, False), (10, [], False, False), (13, [0.11, 14.66], False, False), (17, [3.27, 15.18], False, False), (26, [0.0, 17.14], False, False)]
    other bodies moved > 1 cm: 0 
    touched in the relook phase (body: episodes): table 7
  BendPickMP  n 30, stepped 17; target moved > 1 cm in 5 (ep, cm at each later look, relook A ok, old A ok): [(2, [17.14, 17.13], False, False), (15, [18.49, 18.49], False, False), (20, [18.41, 18.41], False, False), (24, [0.0, 53.11], False, False), (25, [14.48, 14.49], False, False)]
    other bodies moved > 1 cm: 0 
    touched in the relook phase (body: episodes): table 3
  TabletopGraspMP  n 30, stepped 29; target moved > 1 cm in 1 (ep, cm at each later look, relook A ok, old A ok): [(26, [2.38, 2.61], False, True)]
    other bodies moved > 1 cm: 0 
    touched in the relook phase (body: episodes): table 29
  all  n 120, stepped 96; target moved > 1 cm in 20 (ep, cm at each later look, relook A ok, old A ok): [(2, [17.14, 17.13], False, False), (15, [18.49, 18.49], False, False), (20, [18.41, 18.41], False, False), (24, [0.0, 53.11], False, False), (25, [14.48, 14.49], False, False), (26, [2.38, 2.61], False, True), (7, [17.15, 17.15], False, False), (10, [], False, False), (13, [0.11, 14.66], False, False), (17, [3.27, 15.18], False, False), (26, [0.0, 17.14], False, False), (1, [0.0, 17.06], False, True), (6, [0.0, 10.88], False, True), (8, [0.0, 1.52], True, False), (9, [0.0, 17.58], False, True), (10, [0.0, 17.25], False, False), (12, [0.0, 17.54], False, False), (21, [0.0, 17.12], False, F …
    other bodies moved > 1 cm: 0 
    touched in the relook phase (body: episodes): table 52
```

**只看不补**（`eval_look_only.txt`，只有 A 列有数，J、S、kin_rr 这一组没跑）。第 1 节：

```
1. successes, relook vs old (only relook / only old, McNemar p over episodes; layouts up/down, sign p)
  45 s
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop           10 vs   7 (3/0 p 0.250; 3/0 p 0.250)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    XMoveBendPickTeleop        8 vs   4 (5/1 p 0.219; 3/1 p 0.625)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    BendPickMP                 9 vs   6 (4/1 p 0.375; 3/1 p 0.625)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    TabletopGraspMP            9 vs  13 (1/5 p 0.219; 0/4 p 0.125)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    all                      36 vs  30 (13/7 p 0.263; 9/6 p 0.607)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    episodes missing in the relook run: {'A': 0, 'J': 120, 'S': 120, 'kin_rr': 120}
  official limit
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop           10 vs   7 (3/0 p 0.250; 3/0 p 0.250)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    XMoveBendPickTeleop        8 vs   4 (5/1 p 0.219; 3/1 p 0.625)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    BendPickMP                 0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    TabletopGraspMP            9 vs  13 (1/5 p 0.219; 0/4 p 0.125)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    all                       27 vs  24 (9/6 p 0.607; 6/5 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    episodes missing in the relook run: {'A': 0, 'J': 120, 'S': 120, 'kin_rr': 120}
```

第 2 节的 `all`：

```
  all  n 120, steps taken [120, 0, 0]
    odometry xy         0.97 [  7.52]  21.13 (n 120)   yaw   0.53 [  1.46]   6.57 (n 120)
    first look xy       0.23 [  0.55]  14.49 (n 120)   yaw   0.07 [  0.21]   5.62 (n 120)
    last look xy        0.23 [  0.55]  14.49 (n 120)   yaw   0.07 [  0.21]   5.62 (n 120)
    stance before      19.46 [ 29.00]  43.95 (n 120)   yaw   3.43 [  9.10]  13.57 (n 120)
    stance after       19.46 [ 29.00]  43.95 (n 120)   yaw   3.43 [  9.10]  13.57 (n 120)
    off (robot's own)  19.41 [ 33.62]  47.33 (n 120) ->  19.41 [ 33.62]  47.33 (n 120)
    time added, s       0.00 [  0.00]   0.00 (n 120)
    registration: inliers   0.98 [  1.00]   1.00 (n 240), rms mm   5.04 [  5.45]   6.35 (n 240), sd y cm   0.13 [  0.20]   2.88 (n 240)
```

**只重规划**（`eval_j0.txt`，对 `none` 的 JR）：

```
J0 rows 120, old JR rows 120, common 120
  45 s
    XMovePickTeleop        n  30  J0   7  old   7  only J0  0  only old  0  McNemar p 1.000
    XMoveBendPickTeleop    n  30  J0   4  old   4  only J0  0  only old  0  McNemar p 1.000
    BendPickMP             n  30  J0   6  old   6  only J0  0  only old  0  McNemar p 1.000
    TabletopGraspMP        n  30  J0  10  old  11  only J0  2  only old  3  McNemar p 1.000
    all                    n 120  J0  27  old  28  only J0  2  only old  3  McNemar p 1.000
  official limit
    XMovePickTeleop        n  30  J0   7  old   7  only J0  0  only old  0  McNemar p 1.000
    XMoveBendPickTeleop    n  30  J0   4  old   4  only J0  0  only old  0  McNemar p 1.000
    BendPickMP             n  30  J0   0  old   0  only J0  0  only old  0  McNemar p 1.000
    TabletopGraspMP        n  30  J0  10  old   9  only J0  3  only old  2  McNemar p 1.000
    all                    n 120  J0  21  old  20  only J0  3  only old  2  McNemar p 1.000
  routes: J0 29, old 29, both 29, only J0 [], only old []
```

**CPU 的 `none` 对 GPU 的老记录**（`eval_cpu_vs_gpu.txt`，只作记录，不进判据；这里"relook"一栏是 CPU 的 `none`，"old"是 GPU 的老记录）。第 1 节：

```
1. successes, relook vs old (only relook / only old, McNemar p over episodes; layouts up/down, sign p)
  45 s
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop            7 vs   5 (7/5 p 0.774; 4/3 p 1.000)     7 vs   5 (7/5 p 0.774; 4/3 p 1.000)     7 vs   5 (7/5 p 0.774; 4/3 p 1.000)     7 vs   5 (7/5 p 0.774; 4/3 p 1.000)
    XMoveBendPickTeleop        4 vs   3 (4/3 p 1.000; 4/2 p 0.688)     4 vs   3 (4/3 p 1.000; 4/2 p 0.688)     4 vs   3 (4/3 p 1.000; 4/2 p 0.688)     4 vs   3 (4/3 p 1.000; 4/2 p 0.688)
    BendPickMP                 6 vs  11 (3/8 p 0.227; 2/5 p 0.453)     6 vs  11 (3/8 p 0.227; 2/5 p 0.453)     6 vs  11 (3/8 p 0.227; 2/5 p 0.453)     6 vs  11 (3/8 p 0.227; 2/5 p 0.453)
    TabletopGraspMP           13 vs  10 (6/3 p 0.508; 4/2 p 0.688)    11 vs  15 (4/8 p 0.388; 1/4 p 0.375)    16 vs  14 (8/6 p 0.791; 4/3 p 1.000)    16 vs  17 (6/7 p 1.000; 3/2 p 1.000)
    all                    30 vs  29 (20/19 p 1.000; 14/12 p 0.845) 28 vs  34 (18/24 p 0.441; 11/14 p 0.690) 33 vs  33 (22/22 p 1.000; 14/13 p 1.000) 33 vs  36 (20/23 p 0.761; 13/12 p 1.000)
    episodes missing in the relook run: {'A': 0, 'J': 0, 'S': 0, 'kin_rr': 0}
  official limit
    task                                                         A                                       J                                       S                                  kin_rr
    XMovePickTeleop            7 vs   5 (7/5 p 0.774; 4/3 p 1.000)     7 vs   5 (7/5 p 0.774; 4/3 p 1.000)     7 vs   5 (7/5 p 0.774; 4/3 p 1.000)     7 vs   5 (7/5 p 0.774; 4/3 p 1.000)
    XMoveBendPickTeleop        4 vs   3 (4/3 p 1.000; 4/2 p 0.688)     4 vs   3 (4/3 p 1.000; 4/2 p 0.688)     4 vs   3 (4/3 p 1.000; 4/2 p 0.688)     4 vs   3 (4/3 p 1.000; 4/2 p 0.688)
    BendPickMP                 0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)     0 vs   0 (0/0 p 1.000; 0/0 p 1.000)
    TabletopGraspMP           13 vs  10 (6/3 p 0.508; 4/2 p 0.688)     9 vs  12 (4/7 p 0.549; 2/4 p 0.688)    11 vs   9 (7/5 p 0.774; 4/3 p 1.000)    13 vs  15 (6/8 p 0.791; 2/2 p 1.000)
    all                    24 vs  18 (17/11 p 0.345; 12/7 p 0.359) 20 vs  20 (15/15 p 1.000; 10/9 p 1.000) 22 vs  17 (18/13 p 0.473; 12/8 p 0.503) 24 vs  23 (17/16 p 1.000; 10/7 p 0.629)
    episodes missing in the relook run: {'A': 0, 'J': 0, 'S': 0, 'kin_rr': 0}
```

第 3 节：

```
3. checks
  JR: the relook record and final belief equal A's in 120/120 episodes
  SF: the relook record and final belief equal A's in 29/29 episodes
  kin_rr: the relook record and final belief equal A's in 29/29 episodes
  old rows carrying a relook record: 0; relook A rows without one: 120
  falls: relook []; old []
```

### 事后分析（不在预注册里，只作记录）

- `posthoc.py`（`scripts/experiments/simple_relook_20261126/posthoc.py`，md5 `a0f2164a`，副本在 `$X/posthoc.py`）是看到结果以后写的，命令是 `posthoc.py $X/sonic/relook/bench $X/sonic/none/bench $X/sonic/look_only/bench`。
- 第 1 部分拆开"走路时撞动"和"补步时撞动"。起因：行里的 `target_moved_true_cm` 是从走路之前的起始一眼量起的，不是从走到后的第一眼量起。所以第 6 节列出的 20 局里，含一局是走路时被撞动的（XMoveBendPick 第 10 局，只看不补组里同样 2.47 cm）。第 6 节标题里"since the first look"的说法不对。
- 第 2 部分比较下令的补步和真实的站位变化。
- 第 3 部分把伸手的误差和成功局数按补步后的真实站位误差分组。

```
1. the target's displacement: the walk vs the correction steps (truth, record only)
  relook    XMovePickTeleop      moved by the walk 0 
                                 moved by the steps 9 (ep, cm per step, relook A ok, none A ok): [(1, [0.0, 17.06], False, True), (6, [0.0, 10.88], False, True), (8, [0.0, 1.52], True, False), (9, [0.0, 17.58], False, True), (10, [0.0, 17.25], False, False), (12, [0.0, 17.54], False, False), (21, [0.0, 17.12], False, False), (25, [0.0, 16.16], False, False), (26, [67.99], False, False)]
  relook    XMoveBendPickTeleop  moved by the walk 1 [(10, 2.47)]
                                 moved by the steps 4 (ep, cm per step, relook A ok, none A ok): [(7, [17.15, 0.0], False, False), (13, [0.11, 14.55], False, False), (17, [3.27, 11.91], False, False), (26, [0.0, 17.14], False, False)]
  relook    BendPickMP           moved by the walk 0 
                                 moved by the steps 5 (ep, cm per step, relook A ok, none A ok): [(2, [17.14, -0.01], False, False), (15, [18.49, 0.0], False, False), (20, [18.41, 0.0], False, False), (24, [0.0, 53.11], False, False), (25, [14.48, 0.01], False, False)]
  relook    TabletopGraspMP      moved by the walk 0 
                                 moved by the steps 1 (ep, cm per step, relook A ok, none A ok): [(26, [2.34, 0.23], False, True)]
  relook    all                  moved by the walk 1 
                                 moved by the steps 19
  look_only XMovePickTeleop      moved by the walk 0 
                                 moved by the steps 0
  look_only XMoveBendPickTeleop  moved by the walk 1 [(10, 2.47)]
                                 moved by the steps 0
  look_only BendPickMP           moved by the walk 0 
                                 moved by the steps 0
  look_only TabletopGraspMP      moved by the walk 0 
                                 moved by the steps 0
  look_only all                  moved by the walk 1 
                                 moved by the steps 0

2. correction steps as executed (relook A): commanded vs the true stance change
  XMovePickTeleop      steps  55  commanded  20.6 cm  along / commanded 1.30 [p10 0.17, p90 2.25]  across  11.2 cm  |true - commanded|  19.1 [p90  44.5] cm
  XMoveBendPickTeleop  steps  35  commanded  20.4 cm  along / commanded 1.21 [p10 0.55, p90 2.07]  across  10.8 cm  |true - commanded|  18.3 [p90  40.7] cm
  BendPickMP           steps  26  commanded  17.5 cm  along / commanded 1.14 [p10 0.42, p90 1.65]  across   4.8 cm  |true - commanded|  11.7 [p90  19.9] cm
  TabletopGraspMP      steps  51  commanded  26.9 cm  along / commanded 1.18 [p10 0.40, p90 2.18]  across   8.6 cm  |true - commanded|  16.9 [p90  42.4] cm
  all                  steps 167  commanded  21.6 cm  along / commanded 1.19 [p10 0.33, p90 2.17]  across   8.6 cm  |true - commanded|  16.4 [p90  42.6] cm

3. the reach after the steps: pregrasp hand error (cm), median [p90], and successes, by the true stance error after
   the steps (relook; none has no looks, so its stance error is the relook's first look, the same walk)
  XMovePickTeleop      none   10-20: n 15 ok  4 err   5.6 [  8.7] | >20: n 15 ok  3 err   3.5 [  7.1]
  XMovePickTeleop      relook <=10: n 11 ok  4 err   4.4 [  9.2] | 10-20: n 10 ok  5 err   3.8 [  5.2] | >20: n  9 ok  1 err  17.9 [ 31.3]
  XMoveBendPickTeleop  none   <=10: n  7 ok  2 err   7.2 [ 10.4] | 10-20: n 14 ok  1 err   8.4 [ 10.8] | >20: n  9 ok  1 err   8.0 [ 16.0]
  XMoveBendPickTeleop  relook <=10: n 16 ok  3 err   7.6 [ 12.2] | 10-20: n  9 ok  0 err   7.3 [ 13.0] | >20: n  5 ok  1 err   3.9 [ 15.7]
  BendPickMP           none   <=10: n 12 ok  3 err   4.1 [  6.4] | 10-20: n 10 ok  1 err   3.7 [  5.8] | >20: n  8 ok  2 err   6.9 [ 10.2]
  BendPickMP           relook <=10: n 22 ok 12 err   4.2 [  6.4] | 10-20: n  6 ok  2 err   4.2 [  5.5] | >20: n  2 ok  0 err   8.7 [ 12.0]
  TabletopGraspMP      none   10-20: n  7 ok  0 err  15.0 [ 24.1] | >20: n 23 ok 13 err   4.8 [ 22.9]
  TabletopGraspMP      relook <=10: n 13 ok  2 err   4.0 [  9.3] | 10-20: n  8 ok  1 err   5.8 [  9.6] | >20: n  9 ok  0 err  11.5 [ 17.5]
  all                  none   <=10: n 19 ok  5 err   6.1 [  8.3] | 10-20: n 46 ok  6 err   6.1 [ 14.4] | >20: n 55 ok 19 err   4.9 [ 13.7]
  all                  relook <=10: n 62 ok 21 err   5.1 [  9.3] | 10-20: n 33 ok  8 err   4.7 [ 11.3] | >20: n 25 ok  2 err  12.3 [ 25.1]
```

**另外几项只读查询**（在 4090 上对上面的行和规划输出直接算，都不在预注册里）：
- **补步后那几眼里配准误差超过 5 cm 的**：167 眼中有 11 眼（6.6 %）；其中 9 眼在 TabletopGrasp（51 眼里的 17.6 %）。第一眼超过 2 cm 的：120 局中 3 局（4.24、4.81、14.49 cm）。
- **配准退化**：只有 1 眼的内点为 0、残差为 nan，是 TabletopGrasp 第 17 局的第二眼。这一局在补步时摔倒（8.1 s），也是第 3 节记的那次摔倒。
- **TabletopGrasp 上的 cuRobo 规划**：
  - `relook`：30 局里 15 局规划成功、1 局 `GRAPH_FAIL` 也给了路线，12 局 `IK_FAIL`，1 局 `FINETUNE_TRAJOPT_FAIL`，1 局没有抓取；
  - `none`：28 局成功，2 局 `IK_FAIL`；
  - `relook` 里规划失败的 13 局，补步后的真实站位误差中位数 19.5 cm，目标离骨盆中位数 54.7 cm；规划成功的 16 局，站位误差中位数 6.8 cm。
- **TabletopGrasp 上 J 列的成功，按有没有路线分**：
  - `relook`：有路线的 16 局里 J 13 局、A 3 局；没路线的 14 局里 J 0 局、A 0 局；
  - `none`：有路线的 28 局里 J 10 局、A 12 局；没路线的 2 局里 J 1 局、A 1 局。
- **补步时机器人哪个部件碰了桌子**（行里 `hits` 的 relook 段，按局计）：
  - TabletopGrasp：右手食指 28 局、右腕 26 局、左腕 25 局，收手以后双手仍贴着桌面；
  - XMovePick：右髋 7 局、左手中指 6 局；
  - XMoveBendPick：右膝 6 局、右髋 4 局；
  - BendPick：右膝 3 局。
  - `hits` 不记机器人和目标的接触，所以撞动目标的是哪个部件，记录里看不出来。
- **伸手时碰桌**：TabletopGrasp 的 A 在 pregrasp 阶段碰桌，`none` 30/30 局，只看不补 30/30 局，`relook` 28/30 局。三组都差不多，所以这一项分不出三组的差别。

## 意外

**先对照预测**（✓ 落在预测区间内，✗ 落在区间外）

| 预测 | 结果 |
|---|---|
| ScaleBFM 补步 0–5 局 | 0 局 ✓ |
| ScaleBFM 配准误差中位数 ≤ 0.3 cm，90 分位 ≤ 1 cm，转角 ≤ 0.2° | 第一眼 0.05 cm，90 分位 0.11 cm，转角最大 0.10° ✓（位置最大 1.92 cm） |
| ScaleBFM A 79–87，差 ±4 以内，p > 0.05 | 80，−3（0/3），p 0.250 ✓ |
| ScaleBFM J 90–98、S 89–97、kin_rr 90–98 | 93、92、93 ✓ |
| ScaleBFM 官方时限内各列差 ±5 以内 | −3、−2、−2、−3 ✓ |
| ScaleBFM J0 90–98，和旧 J 差 ±4 以内 | 95，差 0（0/0）✓ |
| SONIC `none` A 22–36、J 26–42、S 25–41、kin_rr 28–44；官方 A 12–24 | 30、28、33、33；官方 24 ✓（官方在上沿） |
| SONIC 至少补一步 70–110 局 | 96 局 ✓ |
| SONIC 补步后真实站位误差中位数 5–15 cm，≤ 10 cm 的 40–95 局 | 9.76 cm，62 局 ✓；补步前 19.46 cm，和旧的 19.6 cm 相近 ✓ |
| SONIC 第一眼配准误差中位数 ≤ 0.5 cm，超过 2 cm 的 ≤ 5 % | 0.23 cm，3/120（2.5 %）✓ |
| SONIC 补步后那几眼超过 5 cm 的 0–25 % | 11/167（6.6 %）✓ |
| SONIC A：`relook` 比 `none` 多 −3 到 +18 | +1（31 vs 30）✓ |
| 分任务：TabletopGrasp −5 到 +12 | −10（3 vs 13）✗ |
| 分任务：XMovePick 0 到 +8 | +3 ✓ |
| 分任务：BendPick −3 到 +4 | +8（14 vs 6）✗ |
| 分任务：XMoveBendPick −3 到 +4 | 0 ✓ |
| SONIC J、S、kin_rr 各多 −5 到 +18 | +13、+6、+7 ✓ |
| SONIC 官方时限内 A 少 12 局到多 3 局 | −22（2 vs 24）✗ |
| SONIC 补步时撞动目标 0–12 局，多数在 TabletopGrasp | 19 局（第 6 节列 20 局，其中 1 局是走路时撞动的），多数在 XMovePick（9 局），TabletopGrasp 只有 1 局 ✗ |
| ScaleBFM 补步时撞动目标 0–2 局 | 0 局 ✓ |
| SONIC 只看不补的 A 多 −3 到 +6 | +6（36 vs 30，p 0.263）✓（在上沿） |
| SONIC J0 和 `none` 的 J 差 ±6 以内 | 27 vs 28，差 −1（2/3）✓ |

**意外和没想到的现象**

1. **TabletopGrasp 上补步帮倒忙，但只在 A 列**。
   - A 从 13 局掉到 3 局（3/13，p 0.021）；按布局是 3 升 7 降，p 0.344，不显著。
   - 同一批局的 J 列是 13 vs 11，没掉。
   - 分开看：
     - 只重新定位、不补步，已经是 9 vs 13（−4，p 0.219）；
     - cuRobo 给了路线的 16 局（补步后真实站位误差中位数 6.8 cm，其中 4 局超过 10 cm）：J 成功 13 局，直接伸手的 A 只有 3 局。按站位分：≤ 10 cm 的 12 局 J 成功 11 局，> 10 cm 的 4 局 J 成功 2 局；同样这 16 局，`none` 的 J 只成功 4 局（独立复核指出原写法"落得准的 16 局"不对，已改）；
     - 规划失败的 13 局（站位误差中位数 19.5 cm，目标离骨盆 54.7 cm）：12 局 `IK_FAIL`、1 局 `FINETUNE_TRAJOPT_FAIL`，没有路线；再加上补步时摔倒的第 17 局，这 14 局 A、J 都是 0。
     - 规划失败不只是因为偏得多：`none` 的站偏中位数 27.5 cm，比这 13 局还大，却有 28/30 局规划成功。差别在偏的方向（独立复核查的 `grasp_h`）：这 13 局补步横向冲过了计划站位，目标落到骨盆右侧 24–58 cm，水平距离中位数 54.9 cm；`none` 的目标在正前方附近，水平距离中位数 39.4 cm（34.5–49.7 cm）。
   - `none` 里，站偏超过 20 cm 的 23 局，A 反而成功 13 局。SONIC 走路时横向停偏约 25 cm（[B58] 也记过）。对直接伸手来说，这个偏出来的位置不比规划的站位差。
2. **BendPick 上补步帮得多**：14 vs 6（8/0，p 0.008），按布局 6 升 0 降，p 0.031，四列一样。预测是 −3 到 +4。只看不补是 9 vs 6（+3）。
3. **`none` 组里，站偏不能预测失败**。
   - 站偏 ≤ 10 cm 的成功 5/19，10–20 cm 的 6/46，> 20 cm 的 19/55。
   - 假设 3 的出发点（旧 GPU 主指标里 ≤ 10 cm 成功 39 %，> 10 cm 成功 20 %）在这次运行里不成立。
4. **补步撞动目标比预测多，而且不在预测的任务上**。
   - 19 局：XMovePick 9 局，BendPick 5 局，XMoveBendPick 4 局，TabletopGrasp 1 局。
   - 位移多在 14–19 cm（19 局里 13 局）。像是被碰倒，没有查。
   - 19 局里只有 XMovePick 第 8 局（1.52 cm）A 成功。
   - 开发局只有 TabletopGrasp，所以开发时没看到这一点。
   - 只看不补的组里，补步撞动 0 局。
5. **官方时限内的损失比预测大得多**。A 是 2 vs 24。补步加收手中位数多花 11.58 s。XMovePick 在官方时限内 0 vs 7。
6. **补步本身走不准**。
   - 下令的步长中位数 21.6 cm，真实位移和下令之差中位数 16.4 cm，90 分位 42.6 cm。
   - 沿下令方向走出的比例中位数 1.19，10–90 分位 0.33–2.17。
   - 例：XMovePick 第 1 局，下令向后 14 cm、向侧 15 cm，实际向后走了 55 cm，侧向几乎没动。
   - 正式局整体像开发局的第 3 级：开发局 sK 第 3 级之差中位数 14.8 cm，90 分位 40.6 cm。
7. **收手以后，补步时双手仍贴着桌子**。TabletopGrasp 30 局里，右手食指碰桌 28 局，左、右腕 25、26 局。
8. **`relook_eval.py` 第 6 节的位移起点标错了**。
   - 写的是"从第一眼起"，实际是从走路前的起始一眼起，所以列出的 20 局里有 1 局是走路时撞动的。
   - 冻结代码没改，事后分析里拆开了。
9. **ScaleBFM 的 3 局翻转方向一致**（0/3，p 0.25）。
   - 三局是 XMoveBendPick 第 17、28 局（布局相近）和 TabletopGrasp 第 16 局，老记录里都是勉强抓住。
   - 新旧两次的手位置只差约 1 mm，合手阶段的碰触也相同，新的一次没抬起来。
   - J0 一局都没翻。所以位姿估计变 1–2 mm，就足以翻转这种勉强抓住的局。
10. **补步加收手相对"只看不补"是净负**：`relook` 对只看不补，A 是 17/22（31 vs 36）。分任务：TabletopGrasp 3/9，XMoveBendPick 1/5，BendPick 5/0，XMovePick 8/8。A 列那 1 局的增量来自"看"，不来自补步（独立复核指出，原结论漏了这组预注册的拆分）。
11. **站偏只在中位数上减半**：90 分位 29.00 → 29.17 cm，没变；最大值 43.95 → 67.98 cm，变差。补过步的 96 局里有 19 局补完更偏。
12. **SONIC 每局本身变化很大**：CPU 和 GPU 之间，什么都不改，A 也有 20/19 局翻转，BendPick 是 6 vs 11。分任务差 ±5 局以内，可能就是这种翻转。

## 结论与下一步

（初稿经独立复核后改过：原论断 5"限制收益的是补步走不准，不是定位"是事后分析，和判据的读法冲突，已撤回，改成下面第 5 条；J 列增量的噪声尺度、"落准的局"的写法、预测落空的几项都按复核意见改了。）

**论断和证据**

1. **家具配准的第一眼定得准；在 ScaleBFM 上不帮倒忙**。假设 1 在第一眼成立，假设 2 成立。
   - ScaleBFM：第一眼误差中位数 0.05 cm，90 分位 0.11 cm（ScaleBFM eval_relook 第 2 节）。120 局都没补步。四列分别差 −3、−2、−2、−2 局，p 都 ≥ 0.25（第 1 节）。J0 没有翻转。
   - SONIC：第一眼误差中位数 0.23 cm，90 分位 0.55 cm，3/120 局超过 2 cm；同期里程计中位数 0.97 cm，90 分位 7.52 cm（SONIC eval_relook 第 2 节 `all`）。
   - 补步以后的那几眼没那么可靠：TabletopGrasp 9/51 眼超过 5 cm；最后一眼超过 2 cm 的 8 局，A 全部失败，其中 3 局 `none` 是成功的（复核核对）。
2. **SONIC 补步把站偏的中位数减半，但 A 列不涨**。
   - 站位误差中位数 19.46 → 9.76 cm。90 分位 29.00 → 29.17 cm 没变，最大值 43.95 → 67.98 cm 变差（第 2 节 `all`）。
   - A 是 31 vs 30（23/22，p 1.000；按布局 17/12，p 0.458）。
   - 拆开看：只看不补 36 vs 30（+6，p 0.263）；带第二眼相对只看不补是 17/22，即 −5。所以 A 列这点增量来自"看"，补步加收手相对只看是净负（eval_relook、eval_look_only 第 1 节）。
3. **J 列的 +13 多半不是规划带来的**。
   - J 41 vs 28（29/16，p 0.072）。没有路线的局里 J 行和 A 行逐局相同（复核核对：`relook` 102/102，`none` 91/91），所以 +13 里有 +11 是三个非 TabletopGrasp 任务上 A 的变化。
   - TabletopGrasp 上 J 只是 13 vs 11（9/7，p 0.804）。
   - 合适的噪声尺度是两次等价运行之间的逐局分叉：CPU 对 GPU 的老记录，A 列 20/19、J 列 18/24 局翻转。J0 只量了"同一段录像重新规划"的噪声（5 局翻转），偏小。
   - S 39 vs 33（p 0.471），kin_rr 40 vs 33（p 0.349），都不显著。
4. **官方时限内补步是净损失；补步还撞动目标**。
   - 官方时限内 A 是 2 vs 24（0/22，p 0.000），四列都一样。补步加收手中位数多花 11.58 s。
   - 45 秒内没有一局是因为超时失败（`relook` 最长 33.8 s），所以 45 秒内的比较不受多花时间的影响。
   - 撞动目标：按预注册第 6 节的度量是 20 局；去掉走路时撞的 1 局，补步时撞的是 19 局（事后分析第 1 部分）。这 19 局里 `relook` 的 A 只成功 1 局，而 `none` 原来成功 4 局，所以撞动最多解释 A 少 3 局。
5. **事后看到的现象（不作结论，原因没定）**：
   - 补步执行误差中位数 16.4 cm，90 分位 42.6 cm（事后分析第 2 部分）。
   - 按补步后的站偏分组，同一批局 A 成功是：
     - ≤ 10 cm 的 62 局，`relook` 21 局，`none` 11 局；
     - 10–20 cm 的 33 局，`relook` 8 局，`none` 12 局；
     - > 20 cm 的 25 局，`relook` 2 局，`none` 7 局。
   - 这些增益全在三个非 TabletopGrasp 任务上；TabletopGrasp 上落准的局是 2 vs 3。分组用的是处理之后才出现的变量，不能当因果证据。
   - TabletopGrasp 上，计划站位让目标落在骨盆右侧约 21 cm（`stand 0.3`、`side_off -0.2`）。SONIC 不补步时横向走不够，目标反而在正前方，A 成功 13/30。补准了也不涨，可能是计划站位本身不适合 SONIC 直接伸手。

**证据强度**：
- 每组一次运行，每组 120 局（4 个任务 × 30 局），按局配对，种子和主指标相同。
- 第 0、1 级共用布局，按局的 McNemar 会高估显著性，所以同时报了按布局的符号检验。
- SONIC 在 CPU 上跑，基线是同次的 CPU `none`。

**其他可能的解释**
1. **分任务的差别部分是混沌翻转**：补一步以后，整局都和对照分叉了。BendPick 是 8/0、按布局 6/0，不太像纯翻转；TabletopGrasp 按布局是 3/7（p 0.344），证据弱。
2. **TabletopGrasp 的损失可能来自计划站位的几何**，不是补步扰动了机器人（见论断 5）。
3. **收手后的起手姿势可能是独立因素**。正式局里分不开：TabletopGrasp 30 局里 29 局补过步，补步前都收了手。

**四类陷阱**
- **benchmark 选择**：
  - 四个任务各 30 局，和主指标相同，全部报了，没剔除任务或局，撞动目标的局也没剔除。
  - 收手、idle 结束、12 cm 阈值都是在开发局 30–49 上定的，而开发局只有 TabletopGrasp 的第 3、4 级；"从收手姿势伸手"在开发局里没试过。
  - CPU 的 `none` 和 GPU 的老记录只是总数接近（30 vs 29），逐局有 39 局翻转。假设 3 的前提来自 GPU（站偏 ≤ 10 cm 成功 39 %，> 10 cm 20 %），在 CPU 的 `none` 上不成立（26 % 对 25 %）。
- **数据泄漏**：没有训练。复核读了冻结代码：补步命令和停步判断只用配准估计和是否摔倒，真值只写进 `_true` 字段。开发局和正式局不重合；看到正式结果以后没改参数。
- **指标误用**：
  - 45 秒内和官方时限内都报了，四列、分任务都报了。
  - "站偏减半"只说中位数，尾部没变好，已补上。
  - 第 6 节的标签错误在事后分析里更正，冻结输出原样贴出。
  - J 列的噪声尺度原先用错了（只用了 J0），已改。
- **事后挑选**：
  - 每组只跑一次，没有重跑；ScaleBFM 的手动续跑只改了显存门槛。
  - 初稿的论断 5 建立在事后分析上，并改读了判据。复核指出后已撤回，降为"事后看到的现象"。
  - 预注册里对带第二眼组不利的那组拆分（相对只看不补少 5 局），已写进论断 2。

**对照判据**（结果落在三行）
- **"补步后站位误差中位数 ≤ 10 cm，但 A 多 < 3 局"**：**落在这一行**（9.76 cm，+1）。
  - 按判据：站位不是瓶颈，假设 3 不成立；停止在站位上投入，转去查 SONIC 的手（vrref 的手速结果）。
  - 9.76 cm 贴着 10 cm 的线；补准了会不会涨，现有数据定不了，见下一步第 3 条。
- **"某个任务 relook 的 A 比 none 少 ≥ 4 局"**：**落在这一行**，TabletopGrasp −10。判据要查的三样：
  - 接触：补步时双手、手腕贴桌 25–28/30 局；伸手时三组都碰桌，分不出差别。
  - 配准误差：补步后那几眼里 9/51 眼超过 5 cm；最后一眼超过 2 cm 的 7 局在这个任务上，全部失败。
  - 伸手起始姿势（收手）：**没法查**，正式局里收手和补步总是一起出现（29/30 局）。只看不补组不收手、不补步，也已经少 4 局。
  - 阈值和步法不在正式局上调。
- **"补步时撞动目标 ≥ 10 局"**：**落在这一行**。按预注册度量 20 局，补步时撞的 19 局。
  - 撞动的局没剔除。`hits` 不记和目标的接触，看不出是哪个部件撞的；同一阶段碰桌的，XMovePick 主要是右髋和左手，XMoveBendPick 主要是右膝和右髋。
  - 按判据，下一步回开发局试补步时离目标保持距离（先退后再横移，或补步路线避开目标的方位），和 LML 诊断的结论合并考虑。
- 其余各行都没落在：
  - 各项核对通过：同一局各列第二眼的记录相同；旧记录里没有第二眼字段；SONIC 带第二眼组摔倒 1 局（TabletopGrasp 第 17 局）。
  - ScaleBFM 只差 −3 局。
  - SONIC A 没多 10 局，也不在多 3–9 局的区间。
  - 站偏中位数没超过 12 cm。
  - 补步后配准误差超过 5 cm 的是 6.6 %，没到 20 %（TabletopGrasp 单独是 17.6 %）。
  - A 不显著，J0 那一行不适用。
- **落空的预测**：TabletopGrasp A（预测 −5 到 +12，实际 −10）、BendPick A（−3 到 +4，实际 +8）、官方时限内 A（−12 到 +3，实际 −22）、补步撞动目标（0–12 局，实际 19 局）。详见"意外"的对照表。

**下一步**（给协调人定）
1. **按判据，主指标不给 SONIC 加"走后再看"那一行**。现在的补步在 45 秒内 A 列不涨，官方时限内是净损失，还会撞动目标。
2. **按判据，SONIC 上转去查手**（vrref 的手速结果）。
3. **用户的补步思路如果还想留，先用一个诊断实验定"补准了会不会涨"**（复核提的 E1）：
   - **做法**：SONIC 正式 120 局，补步改用仿真真值下令、按真值判停，直到真实站偏 ≤ 3 cm（最多 4 步），收手和不收手各一列，按 A 伸手，和 `none`、只看不补配对。
   - 这一组用了真值，只作天花板，不进主表。
   - **怎么读**：
     - 站准了 A 仍多不到 3 局，或不超过只看不补的 36 局：补步这条路停掉。
     - 多 10 局以上，且 TabletopGrasp 不再少 4 局以上：值得回开发局做步法。
   - **配套**：TabletopGrasp 的站位几何对照（复核提的 E2：用真值放到计划站位，和左移 20 cm 让目标在正前方的站位比），看 TabletopGrasp 是不是计划站位的问题。
4. **撞动目标**：按判据，回开发局试补步时离目标保持距离。开发局要加上 XMovePick 和 BendPick，撞动目标多在这两个任务上。
5. **第二眼的配准本身**：第一眼准，ScaleBFM 上不帮倒忙，SONIC 只看不补 +6（p 0.263，不显著）。要不要单独作为定位手段进主指标，需要另测。

## 独立复核

2026-11-26 11:31–11:45Z，一个独立的 general-purpose agent 做的。它只拿到假设、组别、度量、预测、判据、"结果"一节和初稿结论里的论断清单，以及原始输出和代码的路径，没有拿到推理过程。下面是它的回复原文，只把它的二级标题降成了三级。它的意见已经用来改"意外"第 1、10、11 条和整个"结论与下一步"：初稿的论断 5 撤回，改成"事后看到的现象"。

我对照远程原始输出和每局的行做了独立核对，四部分结论如下。数字核对过的部分基本都对得上。问题主要出在后半句论断的解读上，还有判据表的读法。

### 一、数字核对

**1. 贴出来的原始输出是原文。** 材料里贴的 271 行输出，我逐行在远程原文里都找到了（`eval_*.txt`、`posthoc.txt`、`run.log`）。唯一的例外是第 6 节 `all` 那行，作者自己标了截断。

**2. 必做的四项，我从每局的行重算，全部一致。** 算法是用 venv 的 python 通过 ssh 标准输入跑，远程没有写任何文件。

| 项 | 我重算的结果 | 是否一致 |
|---|---|---|
| SONIC A 列，45 秒内（relook / none / 只看不补） | XMovePick 10/7/10，XMoveBendPick 4/4/8，BendPick 14/6/9，TabletopGrasp 3/13/9，合计 31/30/36；官方时限内 relook 2、none 24 | 一致 |
| TabletopGrasp 有路线 / 没路线 | relook：有路线 16 局，J 13、A 3；没路线 14 局，J 0、A 0。none：有路线 28 局，J 10、A 12；没路线 2 局，J 1、A 1 | 一致 |
| 撞动目标（> 1 cm） | 任一眼超过 1 cm 的有 20 局。其中走路时撞的 1 局（XMoveBendPick 第 10 局，2.47 cm），补步时撞的 19 局（XMovePick 9、BendPick 5、XMoveBendPick 4、TabletopGrasp 1）。这 19 局里 relook 的 A 只成功 1 局（XMovePick 第 8 局） | 一致 |
| 补步后那几眼，配准误差 > 5 cm | 167 眼里 11 眼（6.6 %）。TabletopGrasp 9/51（17.6 %），XMovePick 1/55，BendPick 1/26，XMoveBendPick 0/35。第一眼超过 2 cm 的 3 局：4.24、4.81、14.49 cm | 一致 |

另外核对了一个前提：none、relook、只看不补三组的 `walk_err_cm` 和 `odo_cm` 在 120/120 局里逐位相同。relook 第一眼的站位误差和 none 的 `walk_err_cm` 最大差 0.0。所以事后分析第 3 部分"none 的站偏取 relook 第一眼"这个做法成立。

**3. "另外几项只读查询"都对得上。** 包括：
- 配准退化：TabletopGrasp 第 17 局第二眼，内点 0，残差 nan，8.1 s 摔倒；
- cuRobo 状态：relook 是 15 局成功 + 1 局 GRAPH_FAIL 也给了路线 + 12 局 IK_FAIL + 1 局 FINETUNE + 1 局没有抓取；none 是 28 局成功 + 2 局 IK_FAIL；
- 规划失败的局站偏中位数 19.48 cm、目标离骨盆 54.7 cm；规划成功的局站偏中位数 6.80 cm；
- 补步时碰桌的部件统计；
- pregrasp 阶段碰桌：30/30、30/30、28/30。

运行时间、`MD5.check`、没有 FAILED 文件、评估输出里没有 Traceback，也都核对过。只有一处找不到出处：显存峰值 19544 / 19688 MiB，在 `run.log` 和 `logs/` 里都没有记录。

**4. 逐条看论断，数字都对，但有以下问题。**

- **论断 1**：引用的数字没错，但"定位准"只对第一眼成立。
  - 补步后，TabletopGrasp 有 17.6 % 的眼误差超过 5 cm。
  - 最后一眼误差超过 2 cm 的有 8 局（7 局在 TabletopGrasp），这 8 局 A 全部失败。
  - 其中 TabletopGrasp 第 17、23、26 局在 none 里是成功的。
- **论断 2，"J 的 +13 超过了 J0 的噪声"，拿错了噪声尺度。**
  - relook 没路线的 102 局里，J 行和 A 行的成败和结束时刻 `t` 完全相同（none 里 91/91 也一样）。
  - 所以 J 的 +13 里，有 +11 其实是 A 在三个非 TabletopGrasp 任务上的变化；TabletopGrasp 上 J 只是 13 vs 11（9/7，p 0.804）。
  - J0 量的只是"同一段录像重新规划"的噪声。合适的尺度是两次等价运行之间的逐局分叉：CPU 对 GPU 的老记录，A 是 20/19 局翻转，J 是 18/24 局翻转。
- **论断 2 漏了预注册里专门用来拆分贡献的那组对比。**
  - 只看不补比 none 多 6 局（13/7）。
  - relook 比只看不补少 5 局（17/22，p ≈ 0.52）。分任务：TabletopGrasp 3/9，XMoveBendPick 1/5，BendPick 5/0，XMovePick 8/8。
  - 也就是说，A 列那一点点增量来自"看"，补步加收手相对只看是净负。
- **论断 3**：2 vs 24 是对的，但预测区间是"少 12 局到多 3 局"，实际少 22 局，预测落空了，结论里没写。另外，45 秒内的失败局没有一局是超时（relook 最长 33.8 s），所以 45 秒内的对比不受补步多花的时间影响。
- **论断 4**：19 局没错，但按预注册的第 6 节度量是 20 局。19 是事后修正，修正本身合理，需要写明。撞动的直接代价也有上限：这 19 局里 none 只成功过 4 局，所以撞动最多解释 A 净少 3 局。预测是 0–12 局，也落空了。
- **论断 5**：
  - 6.6 % 和 16.4 cm 都对。我改用 `step_h_cm` / `step_done_true_cm` 在起始朝向的坐标系里重算是 15.8 cm，结论不受坐标系影响。
  - **"落准的局，有路线的 J 成功 13/16"写法不对**：有路线的 16 局里有 4 局站偏超过 10 cm（13.4、16.6、29.0、42.8 cm）。准确的说法是：站偏在 10 cm 内且有路线的 12 局，J 成功 11 局；站偏超过 10 cm 的 4 局，J 成功 2 局。同样这 16 局，none 的 J 只成功 4 局。
  - **"落偏就够不着"也说得不准**：none 在 TabletopGrasp 上站偏中位数 27.5 cm，比 relook 规划失败局的 19.5 cm 还大，却有 28/30 局规划成功。
    - 差别在偏的方向：失败局里，目标落到骨盆右侧 24–58 cm（`grasp_h`），水平距离中位数 54.9 cm；none 是 39.4 cm（范围 34.5–49.7 cm）。
    - 也就是补步横向冲过了计划站位，把目标甩出了够得着的范围。

### 二、四类陷阱

**1. benchmark 选择**
- CPU 自成一组是预注册时定好的，这一点没问题。但 CPU 的 none 和 GPU 老记录只是总数接近（30 vs 29），逐局有 39 局翻转。
- 假设 3 的前提来自 GPU：站偏 ≤ 10 cm 的成功 11/28（39 %），> 10 cm 的成功 18/92（20 %）。在 CPU 的 none 上这个关系不存在：5/19（26 %）对 25/101（25 %）。TabletopGrasp 甚至反过来：站偏 10–20 cm 的 0/7，> 20 cm 的 13/23。
- 开发局只用了 TabletopGrasp 的第 3、4 级。收手、idle 结束、12 cm 阈值这些参数都来自这一个任务，再用到四个任务上；"从收手姿势伸手"在开发局里没试过。
- 第 0、1 级共用布局，按局做的 McNemar 会高估显著性。TabletopGrasp A 少 10 局：按局 p 0.021，按布局只有 3/7，p 0.344。

**2. 数据泄漏：没发现。**
- 读了冻结副本的 `simple_bench.py`（md5 adb4af64）：补步命令由配准估计 `p_c` 和计划站位算出；停不停步只看 `off`（来自估计）和是否摔倒；真值只写进 `_true` 结尾的记录字段。
- `step_walk` 用仿真骨盆和 `m.yaw()` 拼出世界坐标目标，和主流程 `open_walk` 的约定一样，作者也写明了，影响很小。
- 开发局 30–49 和正式局 0–29 不重合。

**3. 指标误用**
- **"站偏减半"只看了中位数**（19.46 → 9.76 cm）。90 分位没变（29.00 → 29.17 cm），最大值变差（43.95 → 67.98 cm），补过步的 96 局里有 19 局反而更偏。
- **"站偏"被当成"难度"用，但在 TabletopGrasp 上这不成立。**
  - 计划站位的参数是 `stand 0.3`、`side_off -0.2`，站准时目标在骨盆右侧约 21 cm。
  - SONIC 不补步时横向走不够，目标反而落在正前方（`grasp_h` 的 y 在 −6 到 +14 cm），A 成功 13/30。
  - relook 落在 10 cm 内的 13 局，A 只成功 2 局；同样这些局，none 成功 3 局。
- J 的增量用 J0 当噪声尺度，问题见第一部分。

**4. 事后挑选**
- 论断 5 完全建立在事后分析（`posthoc.py` 和那几项查询）上，而且推翻了预注册判据的读法，见第四部分。
- 预注册的"只看不补"拆分结果对 relook 不利（相对只看不补少 5 局），结论里没提。
- 我自己补做了一个按局配对、按补步后站偏分组的对比（同样是事后分析）：

  | 补步后站偏 | 局数 | relook A | 同样这些局 none A |
  |---|---|---|---|
  | ≤ 10 cm | 62 | 21 | 11 |
  | 10–20 cm | 33 | 8 | 12 |
  | > 20 cm | 25 | 2 | 7 |

  这对"补准了就有用"有一定支持，但增益全在三个非 TabletopGrasp 任务上；TabletopGrasp 落准时是 2 vs 3。而且分组用的是处理之后才出现的变量：落准的 62 局里也有 7 局撞动目标、23 局碰桌，不能当因果证据。

### 三、可能推翻主要结论的实验

**E1：真值补步（主实验）**
- **做法**：SONIC，CPU，正式 120 局，同样的种子。
  - 新列 P：走到后和 relook 一样收手（0.8 s + 1.0 s），然后补步改用仿真真值：每步按真实骨盆下令、按真值判断，直到真实站偏 ≤ 3 cm 或走满 4 步。最终的位姿估计也用真值，然后按 A 伸手。
  - 再加一列 P-notuck：同上，但不收手。
  - P 是诊断用的天花板，用了真值，不进主表。
  - 和 none、只看不补、relook 按局配对，做 McNemar 加布局符号检验，同时记录撞动目标和碰桌。
- **算推翻**：P 的真实站偏中位数 ≤ 3 cm（确实补准了），但 A(P) − A(none) < +3，或者 A(P) ≤ 只看不补的 36 局。这说明站准了也不涨，"限制收益的是补步走不准"不成立，回到预注册"站位不是瓶颈"的读法。
- **算支持**：A(P) − A(none) ≥ +10，并且 TabletopGrasp 不再少 4 局以上。
- **附带**：P 和 P-notuck 差 5 局以上，说明收手后的起手姿势是一个独立因素。

**E2：TabletopGrasp 的站位几何**
- **做法**：只跑 TabletopGrasp 30 局，用真值把机器人放到两个站位，A 直接伸手，收手和不收手各跑一遍：
  - (a) 计划站位；
  - (b) 计划站位往左移 20 cm，让目标在正前方，模拟 none 的落点。
- **算推翻**：如果 (b) 比 (a) 多 6 局以上，且 (a) ≤ none 的 13 局，那 TabletopGrasp 上 A 少的那 10 局是计划站位不适合 SONIC 直接伸手造成的，补得再准也没用。下一步应该改 SONIC 的站位目标，而不是改步法。

**E3（较便宜）：只在 TabletopGrasp 上用真值替换最后一眼的估计**
- **算推翻**：A 比 relook 多 3 局以上，就推翻"不是定位"。

### 四、判定

- **前半句"补步把站偏减半，但 A 列不涨"：支持。** 中位数 19.46 → 9.76 cm；A 是 31 vs 30（23/22，p 1.000；布局 17/12，p 0.458）。但要注明尾部没有改善。
- **后半句"限制收益的是补步走不准，不是定位"：不支持作为结论，证据不足，而且和预注册读法冲突。**
  - 它全部来自事后分析，分组用的是处理之后的变量。
  - 在决定 A 列结果的 TabletopGrasp 上，数据和它相反：落准时 A 没有涨（2 vs 3）；定位失败直接让 none 成功过的 3 局失败；只看不补在这个任务上也少 4 局。
  - CPU 的 none 本身就看不出站偏和成功率有关系。
  - 补步加收手相对只看不补是净 −5。
- **整体判定：证据不足。**

**判据表：结果落在三行。**
1. **"补步后站位误差中位数 ≤ 10 cm，但 A 多不到 3 局"**：9.76 cm，多 1 局。预注册写的结论是"站位不是瓶颈，假设 3 不成立；停止在站位上投入，转去查 SONIC 的手"。
2. **"某个任务 relook 的 A 比 none 少 ≥ 4 局"**：TabletopGrasp 少 10 局。
3. **"补步撞动目标 ≥ 10 局"**：19 局（按预注册度量是 20 局）。

没落在的行：各项核对都通过；ScaleBFM 只差 −3 局；A 没有多 10 局，也不在多 3–9 局的区间；站偏中位数不超过 12 cm；补步后配准误差超过 5 cm 的是 6.6 %，没到 20 %（但 TabletopGrasp 单独是 17.6 %）；J0 那一行不适用。

**作者读错的地方：**
- **第 1 行**：结论没写"假设 3 不成立"，论断 5 反而把它改读成"补步走不准"，等于继续在站位上投入。这是事后改判据。
- **第 2 行**：要求查的三项里，"伸手起始姿势（收手）"没有查。而且现有数据也分不开：TabletopGrasp 30 局里 29 局补了步，都收过手。
- **第 3 行**：只报了局数，没写规定的下一步（回开发局试补步时离目标保持距离），也没说预测落空了。

**落空的预测**（结论里都没点出来）：

| 预测项 | 预测（relook 减 none） | 实际 |
|---|---|---|
| TabletopGrasp A | −5 到 +12 | −10 |
| BendPick A | −3 到 +4 | +8 |
| 官方时限内 A | −12 到 +3 | −22 |
| 补步撞动目标 | 0–12 局 | 19 局 |

**建议论断 5 改写为**："按预注册，结果落在'站位 ≤ 10 cm 但 A 多不到 3 局'这一行，假设 3 不成立。事后看到：补步执行误差中位数 16 cm；落在 10 cm 内的局 A 配对多 10 局，落偏的局少 9 局；但 TabletopGrasp 上落准也不涨。是否'补准就会涨'，留给真值补步实验（E1）来定。"

本地留有核对用的副本：`/tmp/relook_rev_raw/raw.txt`（远程原始输出）、`/tmp/relook_rev_raw/frozen/simple_bench.py`（冻结代码），分析脚本是同目录下的 `an1.py`–`an7.py`。

## 协调人的决定（11-26，看过草稿后）

1. **SONIC 改用 CPU 跑**：两组都重跑，在同一次运行里配对，不复用旧的 GPU 记录，也不占 HA 的卡。放 4090b 的 CPU（最多 12 个进程，nice 10，启动方式照 `py4090b.py`，先确认 cuRobo 能跑：已确认）。cuRobo 和渲染的显存放在每卡总量不超过 20 GB 的卡上。主指标的 SONIC 那一行照旧用 GPU 的老结果；这次的对比自成一组，CPU 和 GPU 不逐位相同，CPU 上"没有第二眼"那组就是新的基线。
2. **ScaleBFM 照草稿开跑**：先只提交本条作为预注册，再做冻结，再在 1、3 号卡上跑。每卡总显存不超过 20 GB，跑之前先看 nvidia-smi。
3. **开关先只留在命令行**，不升成 bench 选项。
4. **提交前的小试**：SONIC 补步前先收手，只用开发局 30–49 判断，第 3 级变好、第 4 级不变差就写进做法，最多 30 分钟。结果：写进做法，见"开发"的"收手再走"。
5. **补步时防撞走目标**（协调人后来补充，起因是 LML 诊断：SONIC 在哑铃、浇水壶上 84–85 % 的局里物体在手伸过去之前就被撞走，大部分在修正步里，离物体最近的多是左腕）：收手试验里加记物体在补步前后的位移（已补，见"收手再走"）；正式运行每局都报补步时物体有没有被撞动（`relook_eval.py` 第 6 节，判据里另加一行）。物体真值只当度量。
6. **顺序**：冻结（`asrun/`）在提交本条之前就做好了，和协调人说的顺序相反；提交以后冻结内容不再改，`run.sh` 开始和结束时都核对 `MD5.check`。
7. **看过结果以后**（11-26）：
   - **照判据办**：主指标不加 SONIC 的"走后再看"那一行，转去查 SONIC 的手。LML 的收手测试（另一个实验）方向相同：物体留在原处以后，SONIC 的成功仍然 ≤ 17/139。
   - **真值补步对照**（"下一步"第 3 条）和**补步撞动目标的开发局试验**（第 4 条）**先不做**。等协调人和用户一起定"查 SONIC 的手"怎么做时，再决定要不要合进去。
   - 新加的 `resume_scalebfm.sh`、`posthoc.py` 先不提交，等用户说；`rss_status` 由协调人更新。
