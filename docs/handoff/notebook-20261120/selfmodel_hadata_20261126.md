# selfmodel_hadata_20261126：在 HumanoidArena 里给 ScaleBFM 和 HEFT 重采数据，训练 HA 自己的自身模型（第一批）

- **日期**：2026-11-26。在任何正式采集之前写好，单独提交。
- **用户的决定**（11-26）："SIMPLE 用方案一，HA 也重采，检验先不做"。控制器的接法照原样，ha_g1ctrl 不改，控制器读到的速度也不改。
- **只做 ScaleBFM 和 HEFT。** SONIC 在 HA 里伸完手会一直抖，手速 1–14 cm/s，不会衰减，按 LML 的停稳规则（2 cm/s）大部分伸手会被跳过。要不要放宽它的阈值，要用户定，之后另行登记。
- **机器**：4090 的 1–3 号卡，每卡最多 2 个进程（每个约 7.8 GB）。2、3 号卡在跑 `simple_sonic_vrref_20261126`，跑完以前只用 1 号卡；跑完以后扩到 1–3 号卡。0 号卡不碰。4090b 跑不了 HA：Isaac 的 Vulkan 实例建不起来，要修就得改那台机器的环境，所以不用。
- **代码**：`selfwm`，`scripts/experiments/selfmodel_hacollect_20261126/`，md5 见开发报告：hc_ha.py d056258d、hc_ha.sh 7ca6cec0。
  - 计划和执行直接调 LML 的 d3_collect：data3 式用 reset_plan / run_reset，data3b 式用 reset_plan_b / run_reset_b / augment_b。
  - 训练和评估沿用 t3b_*，加上 fdvel 的改动，以及 simpledata 修订 2 的中间点样本（k0 = 3）。
  - 正式采集前，把停稳参数从开发开关改成固定值，代码抄进 `asrun/`，记下 md5。
- **关联**：
  - [selfmodel_xfer_20261125](selfmodel_xfer_20261125.md)：LML 的模型用在 HA 上，误差是 val 的 5–7 倍；
  - [selfmodel_fdvel_20261126](selfmodel_fdvel_20261126.md)：差分速度的 F 模型在 HA 上是 2.71 倍（ScaleBFM）、2.15 倍（HEFT），仍然要重采；
  - [selfmodel_train3b_20261125](selfmodel_train3b_20261125.md)：配方 D；复核发现 M 练过头，最佳 checkpoint 在 2000–6000 步。

## 开发阶段的发现（看过冒烟，没看过正式数据）

- **起始扰动**：重置时给 43 个关节加高斯噪声，标准差 0.02 rad，和 LML 一样。第一下伸手的起点不再只有一个。关节分散程度和 LML 同一量级：
  - ScaleBFM：中位数 0.0096 rad，LML 0.0058 rad；
  - HEFT：0.0034 rad，LML 0.0014 rad。
- **出生点**：房间里最空的点 (−1.71, −0.09)，离最近的家具 2.2 m。碰到东西的段会记下接触。
- **"站着不动"时，HA 里关节动得比 LML 多。** 速度输入超出 LML 训练分布 6 个标准差的段，ScaleBFM 8–30 %，HEFT 43–70 %。
- **停稳**：ScaleBFM 走完路要 3–4 s 才停稳。等 3 s 只有约 1/3 停稳，等 6 s 全部停稳。
- **ScaleBFM 摔倒和跑偏比 LML 多**：xfer 400 次里摔倒 13.5 %、跑偏 25.8 %，LML 是 1 % 和 3 %。
- **HEFT 伸完第一下常常跑偏**：原地 67–100 %，LML 是 59 %。
- **吞吐**：每卡 2 个进程时，每卡每小时的重置次数如下，样本少，误差约 ±20 %。

  | 控制器 | data3 式 | data3b 式 |
  |---|---|---|
  | ScaleBFM | 141 | 130 |
  | HEFT | 157 | 150 |

- **每次重置能用的单手段**：data3 式 1.1–1.4 段（LML 1.3–1.6），data3b 式 1.2–1.7 段（LML 2.1–3.1）。

## 假设

1. HA 和 LML 的接法、物理、速度读数都不同。在 HA 里自己采数据，从头训练的模型在 HA 上应该比迁移过来的 F 准。数据量只有 LML 的一成左右时也是这样：按 LML 的学习曲线，误差大致随数据量的 −0.3 次方变化。
2. 拿 LML 的 F 在同一份 HA 数据上微调，比从头训练更省数据。数据越少，这个优势越明显。

## 做法

### 采集
- **数量**：每个控制器 data3 式 3,000 次重置，data3b 式 1,500 次重置，约为 LML 重置数的 10 %。
- **编号**：
  - data3 式：重置号从 91,000,000 起，种子从 89,100,000 起；
  - data3b 式：重置号从 92,000,000 起，种子从 89,600,000 起。
  - 这些编号段和 LML、xfer、冒烟的都不重叠。
- **停稳规则**：判据和 LML 相同，骨盆和两只手点都慢于 2 cm/s，持续 0.5 s。只把最长等待从 3 s 改成 6 s，参考的保持时间相应从 4.5 s 改成 7.5 s。等不到就跳过下一下伸手，和 LML 一样。
- **检查**：d3 原有的检查加 XFER CHECK，全部通过才能训练。碰到家具的段剔除。

### 训练
两个臂，都用 HA 训练集重算归一化：
- **主臂（从头训）**：配方 D（data3 式 + data3b 式 + 中间点样本，k0 = 3），标志 `REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1`，E3_ensemble。
- **对照臂（微调）**：从 LML 的 F（seed 0）出发，在同一份数据上微调。

两个臂共同的设置：
- **步数**：最多 3 万步，每 1000 步在 val 上评估一次，选最佳 checkpoint。train3b 的网格是每 2000 步，复核发现最佳点很早就出现，所以这次加密。
- **数据量**：第一批数据的 30 %、60 %、100 %，按重置嵌套，得到 HA 自己的学习曲线。
- **seed**：100 % 的两个臂各 3 个 seed，30 % 和 60 % 只用 seed 0。

### 评估
1. **HA 自己的 val 和 test_iid**：沿用 train3、train3b 的切法，分第一下和接着上一下、原地和先走路、data3b 的各类伸手。报位置误差中位数和集成分歧。
2. **xfer 的 HA 记录作额外测试集**：每个控制器 400 次，用冻结的 sx_eval，主取法和补充取法都报。直接和 F 在同一批记录上的绝对误差比。
3. **真实 HA 局**：
   - 把"看一眼、走到站位、伸手"的流程移到 ha_g1ctrl 的宿主上，在采集期间做完。
   - 任务用 P&PBox、DoubleDesk、OpenDoor，开发种子，每个任务 10 局，不碰官方种子 0–2。
   - 每一下伸手按 data3 的格式记下来，模型只预测、不改命令。
   - 模型训完当天离线评估：预测误差是 HA val 的几倍；起始状态在不在训练范围里，在范围内的和不在的分开报。

## 预测

**采集**
- ScaleBFM：摔倒 5–15 %，跑偏 15–30 %。
- HEFT：伸完第一下跑偏 60–100 %。
- 两个控制器停稳以后可用的单手段：data3 式每次重置 1.0–1.5 段，data3b 式 1.1–1.8 段。
- 用时：只用 1 号卡时，每个控制器约 30–45 小时；扩到 1–3 号卡以后，两个控制器合计 20–30 小时。

**模型**（第一批的 100 %，seed 0–2 的平均）

| 指标 | ScaleBFM 预测 | HEFT 预测 |
|---|---|---|
| 从头训：HA val 误差是 LML 上 D 的 val 的几倍 | 1.7–2.6 倍 | 1.5–2.4 倍 |
| 微调：HA val 误差比从头训低 | 10–35 % | 10–35 % |
| xfer 记录上，从头训的绝对误差比 F 低 | 10–40 % | 0–30 % |
| xfer 记录上，微调的绝对误差比 F 低 | 25–55 % | 15–45 % |

- 学习曲线：60 % 到 100 %，从头训的误差再降 8–20 %。
- 真实 HA 局：预测误差是 HA val 的 1.0–2.5 倍；起始状态在范围内的比例 ≥ 70 %。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| 采集检查有任何一项不过 | 停下查清，不训练 |
| 两个臂中较好的那个，在真实 HA 局上是 val 的 ≤ 1.3 倍，置信区间上限 ≤ 1.5 | 用这个臂登记 HA 的"直接执行 vs 用自身模型修正"对比。反解用 lmlcmp3 定下的做法 |
| 1.3–1.5 倍 | 先在 HA val 上校准，另行登记 |
| > 1.5 倍 | 不用于对比。先看误差是不是集中在范围外的起点，再决定在哪一类起点上加采 |
| 从头训 60 % 到 100 % 仍降 ≥ 10 %，而且微调 100 % 仍是 LML 上 D 的 val 的 > 1.3 倍 | 采第二批，补到 LML 的 30 %。之前先决定要不要做 ScaleBFM 和 HEFT 的多环境采集器（估计开发 1.5–2 天） |
| 微调用 30 % 的数据就追平从头训的 100 % | 以后换仿真器，先用少量数据微调 |
| 从头训在 xfer 记录上不比 F 准 | 假设 1 不成立，查 HA 数据有什么问题，再考虑加采 |

## 预测修订

### 修订 1：采集的实现细节和固定参数，正式采集之前（2026-11-26，冻结于 2026-11-26T02:45:08Z）

预测和判据一行不改。这里只写登记里没写到的实现细节和固定下来的参数。训练、评估和真实 HA 局的实现细节另写修订 2：训练的部分在开训之前提交，真实局的部分在真实局冒烟之前提交。

**停稳参数**（`hc_ha.py` 里的常量，开发时的命令行开关已删掉）
- 最多等 6 s，即 300 步；参考保持 7.5 s；阈值 ScaleBFM、HEFT 都是 2 cm/s。SONIC 用 5 cm/s，写在 [selfmodel_hadata_sonic_20261126](selfmodel_hadata_sonic_20261126.md) 的修订 1 里。
- 实现方式：宿主起来以后，直接改 d3_collect 的模块常量 `SETTLE_MAX_F`、`HOLD_S`、`SETTLE_V`。d3_collect 和 sx_collect 的等待每次都现读这几个常量，所以规则本身仍是 d3_collect 的：
  - 骨盆和两只手点（腕偏航连杆前方 13 cm）各自 0.1 s 的位移折成速度；
  - 连续 25 步都低于阈值才算停稳；
  - 等不到就跳过下一下伸手。
- 保持时间为什么是 7.5 s：d3 的 4.5 s 是按 3 s 的等待定的。等待改成 6 s 以后，开发冒烟里出现过 "motion ended"，因为保持比"缓冲延迟 + 等待"还短。所以改成等待 + 1.5 s。

**每段多记三个字段**（三个控制器都记，只作记录，不进模型输入）
- `settled_2cm`：结束标签在 2 cm/s 下也算停稳。条件是 `settled_end` 为真、没有摔倒，并且 `v_end_cms` < 2。`v_end_cms` 是标签行往前 25 步（含标签行）里的最大速度，窗口和停稳判据的窗口相同。ScaleBFM 和 HEFT 的阈值本来就是 2 cm/s，所以这一项和 `settled_end` 相同。
- `jit_pp_cm`：这一段自己那只手的手点，在同一个 25 步窗口里各位置两两之间的最大距离，也就是峰峰值。
- `jit_rms_cm`：同一组位置到它们平均位置的均方根距离。
- 没有停稳标签的段，或者摔倒的段，后两项记 NaN。

**碰到家具的段**
- 每段记两个量：
  - `touch_n`：从命令到标签，除脚以外的最大接触力；
  - `touch_z`：受这个力的机器人身体部位的高度。
- 仿真里关了自碰撞，所以"除脚以外的接触"只可能是地面、桌子或箱子。
- 剔除规则照 xfer：`touch_n` > 1 N 并且 `touch_z` ≥ 0.2 m。训练和评估都剔除这些段，合并以后报剔除了多少段。

**编号和种子**
- 第 i 次重置（i 从 0 起）：
  - 重置号是 id0 + i，HA 的局种子也用它；
  - 伸手计划用 `default_rng(seed0 + i)`；
  - 起始扰动用 `default_rng([seed0 + i, 1])`。
- ScaleBFM 和 HEFT 用同一套编号和种子，所以同一个重置号的伸手命令相同，这和 LML 一样。切分按重置号的哈希，两个控制器的 train / val / test 分法也就相同。
- 有几处数字相同。它们都不在同一个仿真器里，数据不会合在一起，所以不算重叠：
  - simpledata 的 SIMPLE 真实局，重置号从 90,000,000（主对比）和 91,000,000（任务）起，见 `sd_real.py` 的 `ID0`。后者和这里 data3 式的 91,000,000–91,002,999 数字相同。
  - d3_sonic 的 `--seed` 缺省是 91,000,000，是 LML 里 SONIC 采集的种子。
  - lml_tasks 的 `FREE_SEED` 是 93,000,000，是 LML 自由空间局的命令种子。它和 SONIC data3 式重置号的起点相同。
  - 开发冒烟的重置号从 90,000,000 起，种子从 89,000,000 起，和 sd_real 的主对比局数字相同。冒烟数据不进正式数据。

**分片和调度**（`hd_formal.sh`，4090 上只跑一个实例）
- **分片**：每个控制器 data3 式 3,000 次重置分 24 片，data3b 式 1,500 次分 12 片，每片 125 次。
  - 第 k 片做 i = k, k + 24, k + 48, …（data3b 式是 k, k + 12, …）；
  - 每 50 次重置存一个块，块写完才算数；
  - 一片中断后重跑，会跳过已经写好的块。
- **顺序**：ScaleBFM data3 式、data3b 式，HEFT data3 式、data3b 式，然后是 SONIC 的两种。
- **卡**：
  - 开始只用 1 号卡，开 2 个进程；
  - `simple_sonic_vrref_20261126/DONE` 出现以后，加上 2、3 号卡，每卡 2 个进程；
  - 0 号卡不碰。
- **核对和失败**：
  - 每个进程启动前先跑 `md5sum -c`，核对冻结拷贝和几份固定安装文件（见下文"冻结"）；
  - 退出码不为 0 的片不自动重跑。
- **合并**：一个控制器的一种采集全部成功以后，自动在 CPU 上跑 sx_collect 的 merge、check、summary 和 hc_rates（`hd_finish.sh`），每一步的退出码记进 `finish_status`。
  - check 读数据用的是训练要用的读入：fdvel 冻结的 FDVEL 版 `data.py`（174071cb），加上训练的四个标志；
  - 这份 data.py 和它读的 `env.py`（c551fddc）也钉进了 `MD5.check`。
- **让出槽位**：每个槽位有一个 hold 文件，用来给真实局冒烟让卡。文件在时，这个槽位不再开新片，已在跑的片照常跑完。
- **其他设置**：
  - data3b 式用 data3b 的 `b_config.json`。三个控制器的这份文件相同，md5 fd6a191b。
  - SONIC 关掉相机渲染（`--no-camera`，xfer 修订 2 的做法），因为采集不读图像。ScaleBFM 和 HEFT 走 g1 的接法，仿真步进本来就不渲染。

**正式采集开头和 sd_real 共用 1 号卡**
- 从约 02:47Z 起，simpledata 的 `sd_real.py` 在 1、2、3 号卡上各有 2–3 个进程，1 号卡上有 3 个，每个约 0.6 GB。它是 SIMPLE 的 120 局真实局，只跑一次。
- 所以正式采集的开头和它共用 1 号卡。显存装得下，但 1 号卡利用率到了 80 %。同一次重置（ScaleBFM 第 0 片的第一次），第一次冻结核对时还没有 sd_real，用了 83 s；第二次冻结核对时和 sd_real 共卡，用了 109 s。
- 预测里的用时按开发测速定的，不改。sd_real 跑完以后，如果 1 号卡仍明显比开发测速慢，就告诉协调方。

**预测里"可用的单手段"的口径**：hc_ha 的 `hc_rates` 报的"每次重置做成的单手伸手"。开发阶段的数和预测都用这个口径。同时另报一个数：data/ 里停稳、没摔、没碰家具的单手段，每次重置有几段。

**冻结**：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/asrun/`，FROZEN_UTC 2026-11-26T02:45:08Z。冻结脚本是 `hd_freeze.sh`。
- **拷了什么**：
  - 4090 上的实验代码，见下表；
  - 仓库的 `src/`，`selfwm_20261029`、`reachwm_20261030`、`locomanilab_20261117` 三个实验目录，以及两个场景；
  - assets 和 third_party 链到原处；
  - 仓库在 bc426d5e，g1_backend 在 25d262a5。
- **`MD5.check`**：共 642 行，它自己的 md5 是 7a88f236。除了拷过来的文件，还包括几份固定安装文件：
  - HA 的 env.sh、SONIC 接法的几份 Python、pp_box 的任务配置；
  - SONIC 的编码器和解码器 ONNX；
  - ScaleBFM 和 HEFT 的 policy.onnx 和 policy.yaml；
  - check 用的 fdvel 版 data.py 和 env.py。

| 文件 | md5 |
|---|---|
| selfmodel_hacollect_20261126/hc_ha.py | 58dd6cdc |
| selfmodel_hacollect_20261126/hc_ha.sh | 9405b08d |
| selfmodel_hacollect_20261126/hd_job.sh / hd_formal.sh / hd_finish.sh | 82cde807 / 206f1d33 / 67e91c86 |
| selfmodel_hacollect_20261126/hc_rates.py / hc_start.py | d6ce1e2d / b78bffe7 |
| selfmodel_xfer_20261125/sx_ha.py / sx_collect.py / sx_eval.py | 5caa230b / cd290c2e / 81f61381 |
| selfmodel_data3_20261125/d3_collect.py / d3_reach_table.json | dc92703a / 9d4c716a |
| humanoidarena_20261124/ha_sim.py | 1592d3c2 |
| ha_ctrl/ha_g1ctrl.py | 72a7bc22 |
| locomanilab_20261117/lml_selfmodel.py / lml_host.py | f5168f31 / ec679e16 |
| hardgrasp_20261125/hg_walk.py | fc3a5693 |
| reachwm_20261030/data.py | e0cabb83 |
| selfwm_20261029/env.py | be6a6f8a |
| src/locomani/backends/scalebfm/motor.py | 23269ad1 |

- **和开发报告比**：
  - hc_ha.py 从 d056258d 变成 58dd6cdc，改了三处：停稳参数改成常量，删掉开发开关，加上三个额外字段；
  - hc_ha.sh 从 7ca6cec0 变成 9405b08d，只改了第 2 行的注释。把这行改回去，md5 就是 7ca6cec0。
- **`reachwm_20261030/data.py` 不是训练用的版本**。这份拷贝是 4090 仓库里的旧版，采集不用它。check 读的是 fdvel 冻结的 FDVEL 版 174071cb，训练也用这一版，写在修订 2 里。
- **第一次冻结作废**：02:33:24Z 冻过一次，后来作废了，原因是 hd_finish.sh 的 check 会读到 asrun 里旧版的 data.py。改了 hd_finish.sh 和 hd_freeze.sh 以后，在 02:45:08Z 重新冻结。旧的那份移到了 `selfmodel_hacollect_20261126/dev/void/`。
  - 两次冻结之间，4090 仓库里有人改了 `selfwm_20261029/env.py`（c551fddc → be6a6f8a）和 `src/locomani/bench/` 的六个文件。这些改动来自 vrref，加的是 `sonic_vr_frame` 开关，缺省时行为不变。
  - HA 的采集不 import env.py，sx_collect 只在 SIMPLE 宿主里用它。下面的核对是在第二次冻结上重跑的，所以这些改动即使被读到，也没有改变采集的结果。

**冻结拷贝的核对**（2026-11-26，1 号卡）
- 从冻结拷贝重跑开发冒烟的第 0 片：重置号 90,000,000 + 0、2、4、6、8，种子从 89,000,000 起。
- 和开发冒烟 `dev/smoke_s6_scalebfm`（6 s、7.5 s、2 cm/s）比：两边都有的数组共 144 个（含 arm/），逐位相同。SONIC 的冻结拷贝同样对 `dev/smoke_v5_sonic`（5 cm/s）核对过，也是 144 个逐位相同。
- 新版只多出三个额外字段。
- 用的是 `hd_job.sh`，即正式采集的同一条启动路径。check 也在这次重跑的输出上用 `hd_finish.sh` 跑过一遍（CHECK PASSED、XFER CHECK PASSED），只是确认流程走得通。

**训练里的中间点样本（k0 = 3）**：simpledata 的修订 2（6697ed0b）已经提交，这里照它的做法：用 train3b 冻结版 `t3b_mid.py` 的拷贝，每个起点多存第 k − 3 行，k0 = 3。在 HA 数据上的实现和核对写进本条目的修订 2，开训之前提交。

## 结果

## 意外

## 结论与下一步

## 独立复核
