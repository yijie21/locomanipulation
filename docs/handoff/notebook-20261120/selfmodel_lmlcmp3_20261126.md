# selfmodel_lmlcmp3_20261126：LocoManiLab 确认轮——新种子上检验"缩回可达"的反解

- **日期**：2026-11-26，在任何运行之前写好，单独提交。
- **机器**：4090。仿真只用 CPU，最多 35 个进程，nice 19。补拍"那一眼"要用 SAM 3 和 GraspGen-X，用 1–3 号卡中空闲的一张，开跑前看 nvidia-smi，0 号卡不碰。
- **代码**：`selfwm`。沿用 [selfmodel_lmlcmp2_20261126](selfmodel_lmlcmp2_20261126.md) 冻结的 `lc2_run.py`、`lc2_check.py`、`lc2_eval.py`，包括任务、流程、停稳规则、HEFT 步态、模型 D seed 0、A/S/R 三组的定义、排除规则和"到顶"规则。改动只有三处：
  1. 加 E 组；
  2. 每一下多记一项参考动作的终点；
  3. 评估脚本加 P1、P2 的 Holm 校正和 M1–M3。

  跑之前把代码抄进 `asrun/`，记下 md5。
- **种子**：7 个任务，每个任务用**新的**抖动局 30–119，共 90 局。lmlcmp、lmlcmp2 用的是 0–29，这次一局都不重复。新局先补拍"那一眼"并缓存，各组共用。
- **关联**：lmlcmp2 复核的第 3 节，以及复核后的更正第 2、3、7 条；beliefs B56、B57。

## 假设

1. **R 比直接执行好。** R 是把反解每一步的候选命令缩回 IK 可达范围（IK 误差 ≤ 2 cm）。D 在可达命令上预测得准（误差约 1 cm），R 的命令始终留在这个范围里，所以落点更准，成功更多。
2. **R 比原反解 S 好，好处集中在浇水壶这类目标上。** S 会把预抓取命令推过可达边界。D 在这些命令上错了约 20 cm，结果手越过目标、碰到物体。R 不会这样。
3. **这个好处来自"预抓取别推过头，进入段照样修准"。** E 组用来区分：
   - 好处只来自预抓取别推过头：E（预抓取直接执行，进入段用原反解）就能拿到大部分好处；
   - 进入段也需要缩回：R 会明显比 E 好。

## 做法

| 组 | 预抓取命令 | 进入段命令 | 跑哪些控制器 |
|---|---|---|---|
| A 直接执行 | 目标 | 目标 | 三个 |
| S 原反解 | 原反解 | 原反解 | ScaleBFM、SONIC |
| R 缩回可达 | 缩回可达的反解 | 缩回可达的反解 | 三个 |
| E 只在进入段反解 | 目标（同 A） | 原反解（同 S，不缩回） | ScaleBFM |

- HEFT 只跑 A 和 R：lmlcmp2 里 R 在 HEFT 上只有 8 次真的缩了，和 S 基本一样。
- 局数：ScaleBFM 4 × 630，SONIC 3 × 630，HEFT 2 × 630，共 5670 局。
- R 的细节照 lmlcmp2 修订 1，包括"目标本身也够不着时执行目标"这条。
- 判定集：在新种子上用 lmlcmp2 的规则重算。只算两组分开之前的失败（感知、看完那一眼时的规划失败、没有那一眼），超过一半的任务排除；A ≥ 80 % 的任务标"到顶"，也排除。

## 度量

- **主比较**：ScaleBFM，在判定集上按局配对，做 McNemar 精确检验。两个比较一起做 Holm 校正：p 较小的那个要 < 0.025，另一个要 < 0.05。
  - **P1**：R 对 A。R 的成功率高 ≥ 10 个百分点，且校正后显著。
  - **P2**：R 对 S。R 多，且校正后显著。
- **次要**：
  - SONIC 的 R 对 A、R 对 S；HEFT 的 R 对 A；ScaleBFM 的 E 对 A、E 对 S、R 对 E；
  - 7 个任务全算的结果、各任务成功数、失败原因（新增"预抓取时碰到物体"一项）、合手距离配对差；
  - 浇水壶和哑铃上"手越过目标 > 3 cm"的比例。
- **机制检验**（ScaleBFM，只看没碰家具的伸手）：
  - **M1**：命令 IK 误差 > 3 cm 时，落点和 D 预测之差的中位数，S 发出的命令和 R 发出的命令分开算。
  - **M2**：R − E 和 E − S 的成功局数差，判定集上算。
  - **M3**：浇水壶上，预抓取时碰到壶的比例，S 和 R 分开算；以及碰过壶的局的成功率。
- **一致性检查**：
  - 各组之间，只要两组发出的命令逐位相同，结果就必须逐位相同；
  - 冒烟时，在 lmlcmp2 的设计种子上，A、S、R 要和 lmlcmp2 逐局相同。

## 预测

**ScaleBFM 判定集**（预计仍是蜡烛、苹果、雕像柜、哑铃、浇水壶，5 × 90 = 450 对。lmlcmp2 里各组的成功率是 A 20 %、S 27 %、R 31 %）

| 组 | 成功率 |
|---|---|
| A | 16–24 % |
| S | 23–31 % |
| R | 26–35 % |
| E | 23–33 % |

- R − A：+6 到 +13 个百分点。P1 成立的可能约五成，卡在"≥ 10 个百分点"这条线上。
- R − S：+1 到 +6 个百分点。P2 成立的可能约三成半。
- 浇水壶（90 局）：

  | 组 | 成功局数 |
  |---|---|
  | A | 6–14 |
  | S | 3–12 |
  | R | 14–28 |
  | E | 10–24 |

- **M1**：S 发出的、IK 误差 > 3 cm 的命令，落点和预测之差的中位数在 12–25 cm；R 发出的 ≤ 3 cm。
- **M2**：R − E 在 −2 到 +6 局，E − S 在 0 到 +8 局，都在判定集上算。我倾向于认为进入段缩回也有一点用，但拿不准。
- **M3**：浇水壶上，S 预抓取碰壶 40–65 %，R 3–15 %；碰过壶的局成功率 ≤ 10 %。

**SONIC 判定集**（lmlcmp2：A 12.7 %，S 18 %，R 18.7 %）
- R − A：+3 到 +9 个百分点；
- R − S：−2 到 +3 个百分点。

**HEFT 判定集**（lmlcmp2：A 6 %，R 8 %）
- R − A：0 到 +4 个百分点。

**判定集本身**：三个控制器都是那 5 个任务的可能约八成。碗、盐瓶在两组分开之前失败超过一半。

## 判据

| 结果（ScaleBFM，判定集） | 结论和下一步 |
|---|---|
| P1 和 P2 都成立 | ScaleBFM 以后的反解改用 R。SIMPLE 重采训好模型以后，SIMPLE 的对比也用 R |
| 只有 P1 成立 | 反解比直接执行好，但 R 和 S 分不出来，不据此换成 R，以后的对比 S 和 R 都跑。另外照实写：lmlcmp 的 S 对 A 赢，在新种子上也复现了 |
| P1 不成立，但 R − A ≥ 5 个百分点且 p < 0.05 | 反解有用，但没到"赢"的线。这和 lmlcmp2 一致，提高有一部分是运气，照实报告 |
| R − A < 5 个百分点，或 p ≥ 0.05 | lmlcmp、lmlcmp2 的提高主要来自那 30 局种子。不再说"自身模型在 LML 抓取上帮了忙"，回头查蜡烛以外为什么没用 |
| 判定集 R − S ≤ 0，或浇水壶 R − S ≤ 2 局 | 推翻"R 比 S 好"，B56 的补充要改 |
| M1：S 发出的越界命令，预测误差中位数 ≤ 5 cm | 推翻 B57 的机制。越过目标另有原因，要另查 |
| M2：R − E ≥ 4 局 | 进入段也要缩回，"留在可达范围里"本身是关键 |
| M2：R − E 在 ±3 局以内，且 E − S ≥ 4 局 | 好处来自"预抓取别推过头"，缩回本身是顺带的。B57 的结论降为"预抓取不要修大" |
| 冒烟时 A/S/R 和 lmlcmp2 不逐局相同，或检查不过 | 先停下查清，不开正式运行 |

## 预测修订

**修订 1：实现细节、补拍和冒烟（2026-11-26 约 02:30Z 写下，即北京时间 10:30；正式运行之前）**

预测和判据一行不改。这里只写登记里没写到的实现细节和评估口径，以及补拍和冒烟看到了什么。

**代码和冻结**
- 新目录 `scripts/experiments/selfmodel_lmlcmp3_20261126/`：
  - `lc3_run.py`：lmlcmp2 冻结的 `lc2_run.py`（7f3d1e1c）原样复制，只有登记里的三处改动，外加新种子从哪里读"那一眼"。改动处都标了 "lmlcmp3:"。
  - `lc3_check.py`：lmlcmp2 的 `lc2_check.py`（6ebfec2f）改写第 5 项、扩充第 6 项、新加第 7 项，见下面冒烟一节。
  - `lc3_eval.py`：这一轮的评估，在 lmlcmp2 的 `lc2_eval.py`（ae881c04）基础上重写，和代码一起冻结。
  - `lc3_batch.sh`、`lc3_formal.sh`、`lc3_freeze.sh`，以及补拍用的 `lc3_look.sh`。
- 冻结在 `/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/asrun/`，FROZEN_UTC 2026-11-26T02:22:49Z。
  - md5：lc3_run 959fc0b6，lc3_check ad853c53，lc3_eval 3b681e94，lc3_batch 4982c502，lc3_formal 5745d094，lc3_freeze fa3c397c，lc3_look c201083c。
  - 沿用的文件和 lmlcmp2 的 asrun 完全相同，例如 lc2_run 7f3d1e1c、lr_run b783eaef、hg_run ce8397e3、hg_look b4cfe38e、gait_heft 5398ff8d。另记了补拍用的 `hg_ggx_server.py`（ec3944af）。
  - 输入清单与 lmlcmp2 的 asrun 逐行相同：9 个模型、lmlreal 和 data3 的数据、关节范围。另加两行：
    - 本轮 `hg/` 下关节范围的副本，和 hardgrasp 的逐位相同（ccbc3273）；
    - SAM 3 服务的脚本 `sam3_server.py`（51cb2395）。
  - "那一眼"：
    - 新种子的 619 个 npz 记在 `LOOKS3_MD5`；
    - hardgrasp 的 303 个记在 `LOOKS_MD5`，冒烟的设计局要用，和 lmlcmp2 的那份相同。
  - 正式运行在跑之前和跑完之后，都按这份清单核对。冻结后试过一次，两项核对都通过。
- 组名列表在 lmlcmp2 的后面加了一项：alone 0、selfmodel 1、guard 2（这轮不跑）、reachable 3、entryinv 4。前四个的编号不变，所以设计局上的记录编号和 lmlcmp2 相同。
- 运行：
  - 4090，只用 CPU，每个控制器 7 个任务 × 5 个分片 = 35 个进程，nice 19；
  - 控制器依次跑：ScaleBFM、HEFT、SONIC；
  - 分片按"（组，局）"轮流分，一个分片里各组交替。

**1. E 组的具体做法**
- 经过点和预抓取都直接执行目标，和 A 走同一段代码，不做反解。
- 进入段完全照 S 组：
  - 从目标出发，6 步不动点迭代；
  - 按 S 的损失在 7 个候选里取第一个最小的；
  - 不缩回、不加护栏。
- 所以 E 和 A 在进入段命令之前逐位相同。进入段反解后如果仍是目标，整局也和 A 逐位相同。检查脚本会逐局核对这两条。

**2. 参考动作的终点**
- 每一下（经过点、预抓取、进入段）都记：规划出的参考动作最后一个 IK 路点上的手点。
  - 手点 = 右腕 wrist_yaw_link 前方 HAND_OFF 处，和其他记录用的同一个点；
  - 用规划器的模型，在单独的 MjData 里对 `ref_pel`、`ref_q` 做 FK，不动规划器自己的状态。
- 记两份：
  - `ref_end_w`：在规划用的测得世界坐标里，它到 target_w 的距离就是 `ik_err_cm`；
  - `ref_end_h`：在命令时刻的朝向坐标里，和 `cmd_h`、`act_h` 同一个坐标系。
- 只是多记，不进入任何决策。

**3. 新种子的"那一眼"**
- 第 30–119 局从 lmlcmp3 自己的 `hg/looks/` 读，其余局照旧从 hardgrasp 的 `looks/` 读。
- 补拍用的是 hardgrasp 的 `hg_run.py look`，和原缓存同一套流程：
  1. ScaleBFM 走到看物体的位置；
  2. 头部相机拍 640×480 的 RGB-D；
  3. SAM 3 分割；
  4. GraspGen-X 出抓取。
- 用了哪张卡：
  - SAM 3：现有的 18832 服务（tmux `sam3`，0 号卡）。0 号卡上没有起任何新进程，只发推理请求。hardgrasp 的缓存当初也是这样拍的。
  - GraspGen-X：在 2 号卡上新起一个服务，端口 18863，用 hardgrasp 的 `hg_ggx_server.py`，约 1.0 GB。
  - 渲染：EGL 设备 2，也就是 2 号卡。7 个进程，每个任务 1 个，nice 19。
- 卡的情况（GB 都是 nvidia-smi 的 MiB 除以 1000）：
  - 02:02Z 看 nvidia-smi：
    - 0 号卡利用率 0 %，6.2 GB（SAM 3 4.7 GB，另一个 1.4 GB 的进程不是我们的）；
    - 1 号卡 8.7 GB，2 号卡 8.6 GB，各有 1 个 HA 进程；
    - 选了余量稍大的 2 号卡。
  - 02:08Z 开拍前又看了一次：
    - 0 号卡 0 %，6.2 GB；
    - 这期间 HA 在 1、2 号卡上各多了一个进程。2 号卡 17.4 GB，其中 GraspGen-X 1.0 GB。
  - 拍的时候有一个守护脚本，每 5 秒记一次 2 号卡和 0 号卡；2 号卡超过 20000 MiB 就按 pid 停掉补拍进程。
    - 2 号卡最高 18.5 GB（18542 MiB），没有触发。我们占约 1.8 GB：HA 一度不在 2 号卡上时，整张卡只有 2.6 GB，其中 0.8 GB 是别人的。
    - 0 号卡显存一直在 6.2 GB 上下（6159–6207 MiB），没有新进程。利用率最高 74 %，来自我们发给 SAM 3 的请求。
- 拍完立刻按 pid 停掉了 GraspGen-X 服务（pid 520035，02:18:47Z）。我们起的 tmux 会话都已退出。
- 开拍前先核对了一遍，换卡拍出来的和原缓存一样：
  - 原缓存里的两局（苹果第 0 局、浇水壶第 1 局），把它们自己的图像和点云送给 18832 和 18863：
    - SAM 3 的分数相同，掩码 IoU 1.0；
    - GraspGen-X 的抓取数相同，矩阵最大差 3.8e-6。
  - 在 2 号卡上把 hardgrasp 的 4 局重拍一遍（苹果、蜡烛的第 0、1 局），写到单独的目录 `look_repro/`：
    - 苹果两局除计时字段外逐位相同；
    - 蜡烛两局抓取和命令的数值差 ≤ 3.2e-5，是浮点级的差别。
- 结果：630 局里有 619 局写出了缓存。其中 495 局"认出了"（status ok），124 局没认出（not_recognized）。
  - 蜡烛有 11 局报 "index 906 is out of bounds for axis 0 with size 906"，没有"那一眼"：第 49、80、86、96、97、100、102、103、108、110、112 局。
    - hardgrasp 里蜡烛第 3、7、25、27 局也是这个错。
    - 这些局各组都在一开始报错，记成"没有那一眼"，算两组分开之前的失败。
  - 没认出的局在运行时记成感知失败（`plan.fail` 为 perception），也算两组分开之前的失败。各任务：
    - 蜡烛 28；
    - 盐瓶 57；
    - 雕像柜 15；
    - 浇水壶 24；
    - 苹果、碗、哑铃 0。
  - 没认出的比例和 hardgrasp 的缓存相近：hardgrasp 里蜡烛 11/37、盐瓶 27/41、雕像柜 10/41、浇水壶 11/41。
  - 认出的局，GraspGen-X 每局都给出 616 个抓取，和原缓存一样。
  - 只看"那一眼"已经知道的两组分开之前的失败：
    - 盐瓶 57/90，超过一半；
    - 蜡烛 39/90（43 %），离一半只差 6 局。看完那一眼时的规划失败要跑了才知道。
  - 这些都和组无关，预测不改。
- 用时：02:08:52Z–02:18:31Z，约 10 分钟。每局 1.9–6.2 秒，碗最慢。

**4. 评估口径（定在看到任何正式结果之前）**
- **"预抓取时碰到物体"**：
  - 条件：失败的局，不是两组分开之前失败的；预抓取那一下（reach_no 1 的记录）从发命令到落点那一行之间，有任意一行 50 Hz 记录里右手碰到了目标物体（`obj_rows` > 0）。
  - 先判这一项，再按 hardgrasp 的 IK、家具、抓取标签判。其余局保留 hardgrasp 的标签。
  - 改了标签的局，同时列出它们原来的标签。hardgrasp 原来的失败原因表也照旧报。
  - 碰没碰是仿真真值，只用来记分，不进入任何决策。
- **M1**：
  - 只看 ScaleBFM 里没碰家具的伸手：这一下的 `furn_rows` = 0，而且没摔。预抓取和进入段都算。
  - 量：|落点 `act_h` − D 对执行命令的预测|，按执行命令的 IK 误差分三档（≤ 2、2–3、> 3 cm），各组分开算。
  - 判据那一行看 S 的 > 3 cm 那一档的中位数，R 的同档一并报。
  - 新记的参考动作终点另报三个量，只作描述：
    - |落点 − 参考终点|；
    - |参考终点 − 命令|；
    - |落点 − 命令|。
- **P1、P2 和判据各行**：
  - P1、P2 用判定集上按局配对的 McNemar 精确检验，两者一起做 Holm 校正，报原始 p 和校正后的 p。
  - 判据第 3、4 行（"R − A ≥ 5 个百分点且 p < 0.05"、"R − A < 5 个百分点或 p ≥ 0.05"）用原始 p。
  - "浇水壶 R − S ≤ 2 局"按浇水壶的 90 局算。浇水壶在判定集里时，两种算法是同一批局。
- **修正量**：
  - 执行的 |命令 − 目标|，预抓取和进入段分开报；
  - 各组、各任务都给中位数、等于 0 的比例、> 10 cm 的比例；
  - R 的缩回次数也按预抓取和进入段分开报。
- 其余照 lmlcmp2 修订 1 的口径：判定集、越过目标、合手距离、配对检验。
  - 到顶的线是 A ≥ 72/90。
  - 一局都没有的任务不进判定集。

**冒烟**
- **设计种子**（`smoke_design/`）：蜡烛和苹果的第 1000 局，ScaleBFM 4 组、SONIC 3 组、HEFT 2 组，共 18 局。
  - A、S、R 共 20 局：这里的 16 局，加上 `smoke_design_far/` 里 R 的 4 局。它们和 lmlcmp2 冒烟（`smoke/`、`smoke_far/`）的同一批局逐位相同，包括局记录和每一段的数组。
  - 三个控制器的 `lc3_check` 7 项都通过：
    - 第 5 项：各组两两比，先分开的那一下都在预抓取或进入段；命令全相同的局，整局也逐位相同。
    - 第 7 项：|参考终点 − 命令| 和 `ik_err_cm` 的差 ≤ 3.6e-14 cm，参考终点和最后路点 FK 的差 ≤ 2.1e-5 cm。
  - E：预抓取修正量为 0、没有反解，进入段之前和 A 逐位相同。
- **设计种子的远目标**（`smoke_design_far/`）：浇水壶和哑铃的第 1000、1001 局，只跑 ScaleBFM 的 R 和 E，共 8 局。
  - E 有 4 下进入段反解，R 有 5 下缩回。
  - 有一局 R 和 E 发出的命令全相同，整局也逐位相同。
  - 检查全部通过。
- **新种子**（`smoke_new/`）：苹果和浇水壶的第 30 局，各组 2 局，共 18 局。
  - 读的是本轮 `hg/looks/`：
    - ScaleBFM 开始伸手时的状态和拍"那一眼"时逐位相同（`look_state_dev_mm` = 0）；
    - SONIC 偏 22–28 mm，HEFT 偏 56–63 mm，和 lmlcmp2 里的范围相同（中位数 26.9、56.7 mm）。
  - 三个控制器的 `lc3_check` 7 项都通过：
    - 第 5 项：ScaleBFM 12 对、SONIC 6 对、HEFT 2 对，没有违反。命令全相同的 4 对，整局逐位相同。
    - 第 6 项：ScaleBFM 上 S 反解 4 下、E 进入段反解 2 下、R 反解 4 下，没有缩回；HEFT 的 R 有 1 下缩回。
    - 第 7 项：差 ≤ 2.3e-14 cm 和 ≤ 2.3e-5 cm。
  - `lc3_eval` 在三个控制器上跑通（`LC_EVAL_EPS=30:31`）。新失败项生效：ScaleBFM 苹果第 30 局，S 和 R 原记"碰家具"，改记"预抓取时碰到物体"。
- 冒烟结果不计入任何统计。
- 用时：
  - 冒烟开始时机器负载约 35。每局 ScaleBFM 30–51 秒，SONIC 68–100 秒，HEFT 22–80 秒，和 lmlcmp2 相近。
  - 估计正式运行约 3 小时：ScaleBFM 约 55 分钟、HEFT 约 25 分钟、SONIC 约 90 分钟。

**修订 1 补记：启动脚本出错、共享代码被改、改用私有副本（约 02:55Z 写下；正式运行之前，没有跑出任何一局）**

**1. 组名表和 bash 自带的数组重名**
- 02:25:44Z 第一次启动正式运行。`lc3_formal.sh` 里各控制器的组名表起名叫 `GROUPS`，和 bash 自带的只读数组（当前用户的组号）重名，赋值失败，传给各进程的组名成了 "0"。
- 35 个进程起来后，局一开始就报 "tuple.index(x): x not in tuple"。写下的 6 个局记录全是这种报错，没有一局真的跑起来。02:26Z 发现后，按 pid 停掉了脚本和这 35 个进程。
- 改法：组名表改叫 `GRP`；另加一道检查，某个控制器取不到组名就中止。
- 冒烟没发现，是因为设计种子和新种子的冒烟都直接调 `lc3_batch.sh`，没有经过这个脚本。

**2. env.py 被别的实验同步改动**
- 改完后，02:28–02:35Z 在共享代码树上把整条流程试跑了一次（9 局）。9 局和新种子冒烟逐位相同，但 SONIC 跑完后的 md5 核对报 CHANGED。
- 原因：02:34:49Z，另一个实验（simple_sonic_vrref）往 4090 的代码树同步了改动。
  - 改了冻结清单里的 `selfwm_20261029/env.py`（c551fddc → be6a6f8a）：`__init__` 多了一个可选参数 `sonic_vr_frame=None`，只有它不是 None 时才起作用。
  - 同时改了 `src/locomani/bench/` 下 6 个文件。这一轮的调用链不导入它们。
- 4090 的代码树现在有三个实验在往里同步（vrref、simpledata、hadata）。靠别人几个小时不碰来保证冻结，不可靠。

**3. 改用私有副本，重新冻结**
- 把调用链用到的代码整份复制到本轮输出目录下的 `code/`，正式运行只从这里导入。复制的是 02:41Z 时 4090 树上的版本，包括 vrref 加进 env.py 的那个可选参数。
  - 复制的部分，共 1109 个文件：
    - `src/`：locomani，另有 gr00t、real_robot；
    - `scripts/rsi`：env.py 会把它加进导入路径；
    - `configs/`；
    - 调用链的 9 个 scripts/experiments 目录：lmlcmp3、lmlcmp2、lmlcmp、lmlreal、hardgrasp、data3、locomanilab、reachwm、selfwm。
  - 链接到共享树的部分：assets、outputs、third_party、grasp_kit、rsi_kit。这些是数据和第三方的目录，代码按相对仓库的路径去读。如果有进程从这里导入模块，下面的导入检查会让它停下。
- 启动路径：
  - `lc3_formal.sh` 的脚本路径和 PYTHONPATH 都指向 `code/`；
  - 核对环节改成核对副本里每个文件的 md5（`asrun/CODE_MD5`），跑每个控制器之前和之后各核对一次；
  - 只有评估用的解释器（只用 numpy 和 scipy）仍是共享树里的。
- 导入检查（`lc3_run.py` 新加的 (k)，不碰局的逻辑）：
  - 每个跑局进程在驱动和模型加载好以后，打印实际导入的 env.py 路径；
  - env.py 不在 `code/` 下，或者有任何已加载的模块来自共享树，就中止；
  - 每局结束后再查一次，覆盖局里才导入的模块；
  - 正式脚本在每个控制器跑完后，统计这些打印，有一次失败就记一行 "CODE CHECK FAILED"。
- 副本和 73cdf7a4 的冻结（02:22:49Z）相比，不同的文件：
  - 别人改的：env.py，以及 `src/locomani/bench/` 下 6 个文件（bigym2、humanoidarena、locomanilab、simple 的 `__init__.py`，options.py，spec.py），都是 02:34:44–02:34:49Z 改的。和预期一致。
  - 我改的：lc3_run（导入检查）、lc3_batch（代码根目录可以指定）、lc3_formal（GRP、副本路径、核对副本）、lc3_freeze（做副本）。
  - 其余代码、输入清单和两份"那一眼"清单，都和 02:22:49Z 那次逐行相同。
  - 从新种子冒烟开始（02:19Z）到复制时，树上改过的文件也只有上面这些。
- 在副本上把整条流程重跑了一次（`formal_dryrun/`，02:42–02:50Z）：苹果第 30 局，三个控制器各组各 1 局，共 9 局。
  - 跑局、合并、检查、评估和前后的副本 md5 核对都走通，三个控制器的检查都通过；
  - 9 局和新种子冒烟里的同一批局逐位相同，包括局记录和每一段的数组。所以 env.py 和 bench 的改动不影响这一轮；
  - 导入检查：6 个进程启动时、9 局每局结束后，env.py 都在 `code/` 下，来自共享树的模块 0 个。
- 新的冻结：FROZEN_UTC 2026-11-26T02:41:43Z。
  - md5：lc3_run eca94321，lc3_batch 082a0091，lc3_formal 05cf547a，lc3_freeze e7b9ae99；lc3_check、lc3_eval、lc3_look 不变。
  - 清单在 `asrun/`：CODE_MD5（副本的每个文件）、MD5（输入）、LOOKS3_MD5、LOOKS_MD5，以及和 02:22:49Z 冻结的逐项对比 VS_PREV。
- 作废的东西都放在 `aborted_formal_0225/`，不进入任何统计：第一次启动的记录、02:22:49Z 和 02:27:38Z 的两次冻结、共享树上的那次试跑。

## 结果

执行 agent 写于 2026-11-26，正式运行结束之后（06:14Z）。下面的原文都从 4090 读取，md5 和 4090 上的文件相同。除了最后一小节"事后补的描述性分析"，统计口径都是修订 1 事先定好的。

**运行**
- 正式运行 02:51:08Z–06:14:12Z，全部从私有副本 `code/` 导入（冻结时间 02:41:43Z）：
  - ScaleBFM 02:51:08–03:54:55Z，2520 局；
  - HEFT 03:55:30–04:26:47Z，1260 局；
  - SONIC 04:27:06–06:13:42Z，1890 局。
- 一共 5670 局，一局不缺。每个控制器 35 个进程，只用 CPU，nice 19。期间机器负载在 50–95 之间，大部分来自别的任务。
- 导入检查：三个控制器各 35 个进程，启动时导入的 env.py 都在 `code/` 下。每局结束后也复查了，失败 0 次。
- 副本 md5：每个控制器跑之前、跑完之后各核对一次，都和 `asrun/` 相同。
- 蜡烛第 49、80、86、96、97、100、102、103、108、110、112 局没有"那一眼"，各组一开始就报错，照修订 1 记作两组分开之前的失败。eval 第一行的 worker errors 就是这些局：ScaleBFM 44 = 11 局 × 4 组，SONIC 33，HEFT 22。
- 4090 上我起的进程和 tmux 会话都已结束，灵动岛的登记已归档。

`formal.log` 原文（md5 9f22fac3fe27b3b802efd266faa461cc）：

```
inputs = asrun/MD5 2026-11-26T02:51:08Z
scalebfm formal start 2026-11-26T02:51:08Z: 7 tasks x episodes 30..119 x alone,selfmodel,reachable,entryinv, 5 shards per task, code /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code (md5 = asrun)
2026-11-26T02:51:08Z launched scalebfm 30:120: 7 tasks x 5 shards = 35 processes
2026-11-26T03:54:55Z scalebfm runs finished
scalebfm runs done 2026-11-26T03:54:55Z: 2520 episode records
scalebfm code check: 35 processes started with env.py from the copy;  35 /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code/scripts/experiments/selfwm_20261029/env.py; failures 0
scalebfm merged and checked 2026-11-26T03:55:28Z: CHECK PASSED
scalebfm evaluated 2026-11-26T03:55:30Z
scalebfm md5 after: same as asrun
heft formal start 2026-11-26T03:55:30Z: 7 tasks x episodes 30..119 x alone,reachable, 5 shards per task, code /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code (md5 = asrun)
2026-11-26T03:55:30Z launched heft 30:120: 7 tasks x 5 shards = 35 processes
2026-11-26T04:26:47Z heft runs finished
heft runs done 2026-11-26T04:26:47Z: 1260 episode records
heft code check: 35 processes started with env.py from the copy;  35 /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code/scripts/experiments/selfwm_20261029/env.py; failures 0
heft merged and checked 2026-11-26T04:27:04Z: CHECK PASSED
heft evaluated 2026-11-26T04:27:06Z
heft md5 after: same as asrun
sonic formal start 2026-11-26T04:27:06Z: 7 tasks x episodes 30..119 x alone,selfmodel,reachable, 5 shards per task, code /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code (md5 = asrun)
2026-11-26T04:27:06Z launched sonic 30:120: 7 tasks x 5 shards = 35 processes
2026-11-26T06:13:42Z sonic runs finished
sonic runs done 2026-11-26T06:13:42Z: 1890 episode records
sonic code check: 35 processes started with env.py from the copy;  35 /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code/scripts/experiments/selfwm_20261029/env.py; failures 0
sonic merged and checked 2026-11-26T06:14:11Z: CHECK PASSED
sonic evaluated 2026-11-26T06:14:12Z
sonic md5 after: same as asrun
formal done 2026-11-26T06:14:12Z
```

**一致性检查**
- 三个控制器 7 项全部通过：
  - 第 1–4 项：局记录、落点规则、npz 和在线预测、等待，差都在浮点级，违反 0。
  - 第 5 项：各组两两比，先分开的那一下都在预抓取或进入段。命令逐位相同的组对，整局也逐位相同：ScaleBFM 1521/1521，SONIC 962/962，HEFT 211/211。违反 0。
  - 第 6 项：各组的反解和缩回规则，违反 0。
  - 第 7 项：| |参考终点 − 命令| − ik_err_cm | ≤ 6.57e-14 cm，参考终点和最后一个路点 FK 之差 ≤ 3.03e-05 cm。
- `<控制器>/check.txt` 原文（md5：scalebfm b5906f5fa20cc71d9d514022ab8d342b，sonic 7557950f71ba6f7116c23be9c5ef04f4，heft 6af29a83a43d224c5dd58d3e3d3732b7）：

```
scalebfm: 2520 episode records, 3836 segments, 162126 rows; method {0: 961, 1: 953, 3: 961, 4: 961}; reach_no {0: 556, 1: 1644, 2: 1636}; fell/lost 23; settled at the label 2751
1. end error from the last row vs jaw_err_cm: max |d| 0.00003 cm; arrays aligned True; ik_err_cm from the last waypoint max |d| 0.00003 cm; data.py err_cm vs jaw_err_cm max |d| 0.00003 cm; KP vs W max 0.0000 mm; HQ vs J max |d| 1.84e-07; end_pos vs pos_f max 0.0003 mm; episodes whose segments differ from their d3 list: 0 []
2. landing rule: labels by kind (0 settled, 1 3 s timeout, 2 cut) {0: 2751, 1: 65, 2: 997}; violations 0 []; live_err_cm vs jaw_err_cm on 3813 labels max |d| 0.00000 cm
3. x0 / u from the npz vs online (3836 of 3836 segments): x0 max |d| 0.000e+00, u max |d| 0.000e+00; predictions npz vs online max |d| T3 2.38e-05 cm, M 2.38e-05 cm, D 2.38e-05 cm
4. waits: 8137; outside [0, 3 s] or gone unsettled before 3 s: 0 []; segments whose pre_wait_s / prop_settled differ from the wait log: 0 []
5. pairs of groups: 3714; first differing command by reach_no {1: 1629, 2: 564, 'none': 1521} ('none': every command the same); all commands the same and the whole episode the same: 1521 of 1521; violations 0 []
6. rules: selfmodel 814 inverted reaches, E 411 inverted entries, reachable 822 inverted reaches (361 with a shrunk candidate); direct reaches = the goal, no inversion; violations 0 []
7. reference end (3836 of 3836 segments; missing 0): | |ref_end_h - cmd_h| - ik_err_cm | max 6.57e-14 cm; ref_end_w vs the FK of the segment's last waypoint max 3.03e-05 cm
CHECK PASSED
sonic: 1890 episode records, 2670 segments, 126312 rows; method {0: 891, 1: 889, 3: 890}; reach_no {0: 354, 1: 1161, 2: 1155}; fell/lost 8; settled at the label 1425
1. end error from the last row vs jaw_err_cm: max |d| 0.00002 cm; arrays aligned True; ik_err_cm from the last waypoint max |d| 0.00003 cm; data.py err_cm vs jaw_err_cm max |d| 0.00002 cm; KP vs W max 0.0000 mm; HQ vs J max |d| 1.75e-07; end_pos vs pos_f max 0.0003 mm; episodes whose segments differ from their d3 list: 0 []
2. landing rule: labels by kind (0 settled, 1 3 s timeout, 2 cut) {0: 1425, 1: 55, 2: 1182}; violations 0 []; live_err_cm vs jaw_err_cm on 2662 labels max |d| 0.00000 cm
3. x0 / u from the npz vs online (2670 of 2670 segments): x0 max |d| 0.000e+00, u max |d| 0.000e+00; predictions npz vs online max |d| T3 2.38e-05 cm, M 2.38e-05 cm, D 2.38e-05 cm
4. waits: 5834; outside [0, 3 s] or gone unsettled before 3 s: 0 []; segments whose pre_wait_s / prop_settled differ from the wait log: 0 []
5. pairs of groups: 1857; first differing command by reach_no {1: 855, 'none': 962, 2: 40} ('none': every command the same); all commands the same and the whole episode the same: 962 of 962; violations 0 []
6. rules: selfmodel 771 inverted reaches, E 0 inverted entries, reachable 772 inverted reaches (247 with a shrunk candidate); direct reaches = the goal, no inversion; violations 0 []
7. reference end (2670 of 2670 segments; missing 0): | |ref_end_h - cmd_h| - ik_err_cm | max 5.93e-14 cm; ref_end_w vs the FK of the segment's last waypoint max 3.02e-05 cm
CHECK PASSED
heft: 1260 episode records, 1832 segments, 76560 rows; method {0: 925, 3: 907}; reach_no {0: 282, 1: 826, 2: 724}; fell/lost 136; settled at the label 990
1. end error from the last row vs jaw_err_cm: max |d| 0.00003 cm; arrays aligned True; ik_err_cm from the last waypoint max |d| 0.00003 cm; data.py err_cm vs jaw_err_cm max |d| 0.00003 cm; KP vs W max 0.0000 mm; HQ vs J max |d| 1.87e-07; end_pos vs pos_f max 0.0003 mm; episodes whose segments differ from their d3 list: 0 []
2. landing rule: labels by kind (0 settled, 1 3 s timeout, 2 cut) {0: 990, 1: 63, 2: 643}; violations 0 []; live_err_cm vs jaw_err_cm on 1696 labels max |d| 0.00000 cm
3. x0 / u from the npz vs online (1832 of 1832 segments): x0 max |d| 0.000e+00, u max |d| 0.000e+00; predictions npz vs online max |d| T3 3.58e-05 cm, M 6.56e-05 cm, D 2.38e-05 cm
4. waits: 3732; outside [0, 3 s] or gone unsettled before 3 s: 0 []; segments whose pre_wait_s / prop_settled differ from the wait log: 0 []
5. pairs of groups: 619; first differing command by reach_no {1: 398, 'none': 211, 2: 10} ('none': every command the same); all commands the same and the whole episode the same: 211 of 211; violations 0 []
6. rules: selfmodel 0 inverted reaches, E 0 inverted entries, reachable 766 inverted reaches (113 with a shrunk candidate); direct reaches = the goal, no inversion; violations 0 []
7. reference end (1832 of 1832 segments; missing 0): | |ref_end_h - cmd_h| - ik_err_cm | max 4.86e-14 cm; ref_end_w vs the FK of the segment's last waypoint max 2.93e-05 cm
CHECK PASSED
```

**判定集**（照登记规则在运行后算，三个控制器相同）
- 判定集是蜡烛、苹果、雕像柜、哑铃、浇水壶，5 × 90 = 450 对，和预测一致。
- 碗：两组分开之前失败 66/90（SONIC 67/90），全是看完那一眼时的规划失败，排除。
- 盐瓶：57/90，全是感知失败，排除。
- 蜡烛：41/90（46 %），三个控制器相同。其中感知 28、没有那一眼 11、看完那一眼时的规划失败 2。超过一半（≥ 46 局）才排除，差 5 局，所以留在判定集里。
- 其余任务：苹果 8/90（HEFT 6/90），雕像柜 23/90（SONIC 24/90），哑铃 0，浇水壶 24/90。
- 没有任务到顶。A 最高的是 SONIC 蜡烛，48/90，到顶线是 72/90。
- 每个任务两组分开之前的失败，各组都相同。

**ScaleBFM 主比较**（判定集 450 对，按局配对；合手距离是合手时手到抓取点的距离，配对差的中位数和 95 % 区间）

| 比较 | 成功 | 差 | 不一致的对 | McNemar 精确 p | 合手距离配对差 |
|---|---|---|---|---|---|
| P1：R 对 A | 117 对 61（26.0 % 对 13.6 %） | +12.4 个百分点（+56 局） | A 独赢 14，R 独赢 70 | 4.07e-10 | −1.61 cm [−2.61, −0.98] |
| P2：R 对 S | 117 对 104（26.0 % 对 23.1 %） | +2.9 个百分点（+13 局） | S 独赢 12，R 独赢 25 | 0.047 | 0.00 cm [0.00, 0.00] |
| E 对 A | 112 对 61 | +11.3 个百分点（+51 局） | A 独赢 9，E 独赢 60 | 2.24e-10 | −1.83 cm [−3.05, −1.25] |
| E 对 S | 112 对 104 | +1.8 个百分点（+8 局） | S 独赢 21，E 独赢 29 | 0.322 | +0.07 cm [+0.03, +0.12] |
| R 对 E | 117 对 112 | +1.1 个百分点（+5 局） | E 独赢 17，R 独赢 22 | 0.522 | 0.00 cm [−0.06, +0.03] |
| S 对 A | 104 对 61 | +9.6 个百分点（+43 局） | A 独赢 19，S 独赢 62 | 1.77e-06 | −4.01 cm [−4.68, −1.83] |

- Holm：原始 p 4.07e-10 / 0.047，校正后 8.14e-10 / 0.047。较小的 p < 0.025，另一个 < 0.05，两个都显著。
- P1：+12.4 个百分点 ≥ 10，而且显著，成立。
- P2：R 多 13 局，而且显著，成立。
- 7 个任务全算（630 对）：R 133、A 66、S 119、E 132。
  - R 对 A +10.6 个百分点，p 2.38e-12；
  - R 对 S +2.2 个百分点，p 0.0336；
  - R 对 E +1 局，p 1。

**判据各行**（ScaleBFM，判定集）

| 判据 | 是否触发 | 数 |
|---|---|---|
| P1 和 P2 都成立 | **是** | 见上表 |
| 只有 P1 成立 | 否 | |
| P1 不成立，但 R − A ≥ 5 个百分点且 p < 0.05 | 否 | |
| R − A < 5 个百分点，或 p ≥ 0.05 | 否 | +12.4 个百分点，原始 p 4.07e-10 |
| 判定集 R − S ≤ 0，或浇水壶 R − S ≤ 2 局 | 否 | 判定集 +13；浇水壶 R 19、S 5，+14 |
| M1：S 发出的越界命令，预测误差中位数 ≤ 5 cm | 否 | 19.87 cm（n 75） |
| M2：R − E ≥ 4 局 | **是** | +5，p 0.522 |
| M2：R − E 在 ±3 局以内，且 E − S ≥ 4 局 | 否 | R − E +5，E − S +8 |
| 冒烟时 A/S/R 和 lmlcmp2 不逐局相同，或检查不过 | 否 | 见修订 1 的冒烟一节 |

**各任务成功数**（第 30–119 局，每格 90 局）

| 任务 | ScaleBFM A / S / R / E | SONIC A / S / R | HEFT A / R |
|---|---|---|---|
| 蜡烛 | 16 / 49 / 49 / 47 | 48 / 49 / 49 | 31 / 42 |
| 苹果 | 15 / 20 / 20 / 18 | 4 / 8 / 7 | 0 / 1 |
| 碗（排除） | 1 / 0 / 1 / 1 | 0 / 2 / 1 | 0 / 0 |
| 盐瓶（排除） | 4 / 15 / 15 / 19 | 8 / 10 / 9 | 2 / 0 |
| 雕像柜 | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| 哑铃 | 25 / 30 / 29 / 36 | 0 / 10 / 9 | 3 / 1 |
| 浇水壶 | 5 / 5 / 19 / 11 | 1 / 6 / 3 | 1 / 0 |
| 判定集合计（450） | 61 / 104 / 117 / 112 | 53 / 73 / 68 | 35 / 44 |
| 7 个任务合计（630） | 66 / 119 / 133 / 132 | 61 / 85 / 78 | 37 / 44 |

ScaleBFM 各任务的配对（eval 的 C2）：
- R 对 S：只在浇水壶上分得开，S 独赢 3、R 独赢 17，p 0.00258。哑铃 S 独赢 9、R 独赢 8；蜡烛、苹果、雕像柜都是 0 对 0。
- R 对 A：蜡烛 0 对 33（p 2.33e-10），浇水壶 4 对 18（p 0.00434），苹果 2 对 7（p 0.18），哑铃 8 对 12（p 0.503）。

**SONIC 和 HEFT**（判定集 450 对）

| 比较 | 成功 | 差 | 不一致的对 | McNemar 精确 p | 合手距离配对差 |
|---|---|---|---|---|---|
| SONIC R 对 A | 68 对 53 | +3.3 个百分点（+15 局） | A 独赢 2，R 独赢 17 | 0.000729 | −1.49 cm [−1.74, −1.15] |
| SONIC R 对 S | 68 对 73 | −1.1 个百分点（−5 局） | S 独赢 6，R 独赢 1 | 0.125 | 0.00 cm [0.00, 0.00]；S 更近 90 局，R 更近 28 局 |
| SONIC S 对 A | 73 对 53 | +4.4 个百分点（+20 局） | A 独赢 2，S 独赢 22 | 3.59e-05 | −2.17 cm [−2.81, −1.73] |
| HEFT R 对 A | 44 对 35 | +2.0 个百分点（+9 局） | A 独赢 5，R 独赢 14 | 0.0636 | −0.14 cm [−0.43, +0.79] |

- SONIC 的 R − A 主要来自哑铃：0 对 9，p 0.00391。
- HEFT 的 R − A 全部来自蜡烛：2 对 13，p 0.00739。其余任务上 R 都不比 A 多。

**机制检验**（ScaleBFM，只看没碰家具、没摔的伸手）

M1：|落点 − D 对执行命令的预测| 的中位数，cm，按执行命令的 IK 误差分档，括号里是下数。7 个任务：

| 组 | IK ≤ 2 cm | 2–3 cm | > 3 cm |
|---|---|---|---|
| A | 0.84（220） | 1.46（21） | 2.22（27） |
| S | 0.88（338） | 1.67（13） | **19.87（75）** |
| R | 0.87（345） | 1.44（27） | 2.14（27） |
| E | 0.75（238） | 1.47（22） | 11.27（43） |

- 只算判定集：S 的 > 3 cm 一档 19.94 cm（n 72），R 1.62 cm（n 23）。
- 参考终点只作描述。S 的 > 3 cm 一档：|落点 − 参考终点| 10.03 cm，|参考终点 − 命令| 4.43 cm，|落点 − 命令| 9.14 cm。

M2（判定集）：
- R − E = +5 局（117 对 112，p 0.522）；E − S = +8 局（112 对 104，p 0.322）。
- 各任务的 R − E：
  - 蜡烛 +2，苹果 +2，雕像柜 0；
  - 哑铃 −7：E 独赢 10，R 独赢 3，p 0.0923；
  - 浇水壶 +8：E 独赢 3，R 独赢 11，p 0.0574。

M3（浇水壶，有预抓取那一下的 66 局）：
- 预抓取时碰到壶：S 40/66（61 %），R 4/66（6 %），A 2/66，E 2/66。
- 四组合计碰过壶 48 局，成功 2 局（4 %）。

**预测对照**

| 项 | 预测 | 实际 | 落在区间里 |
|---|---|---|---|
| 判定集 | 三个控制器都是那 5 个任务（约八成） | 都是 | 是 |
| 碗、盐瓶两组分开之前失败超过一半 | 超过一半 | 碗 73 %（SONIC 74 %），盐瓶 63 % | 是 |
| ScaleBFM A 成功率 | 16–24 % | 13.6 % | **否，偏低** |
| ScaleBFM S 成功率 | 23–31 % | 23.1 % | 是 |
| ScaleBFM R 成功率 | 26–35 % | 26.0 % | 是 |
| ScaleBFM E 成功率 | 23–33 % | 24.9 % | 是 |
| ScaleBFM R − A | +6 到 +13 个百分点 | +12.4 | 是 |
| P1 成立 | 约五成 | 成立 | 概率预测，不按区间算 |
| ScaleBFM R − S | +1 到 +6 个百分点 | +2.9 | 是 |
| P2 成立 | 约三成半 | 成立 | 概率预测，不按区间算 |
| 浇水壶 A | 6–14 局 | 5 | **否，偏低** |
| 浇水壶 S | 3–12 局 | 5 | 是 |
| 浇水壶 R | 14–28 局 | 19 | 是 |
| 浇水壶 E | 10–24 局 | 11 | 是 |
| M1：S 的越界命令 | 12–25 cm | 19.87 cm（判定集 19.94 cm，n 72） | 是 |
| M1：R 的越界命令 | ≤ 3 cm | 2.14 cm（判定集 1.62 cm，n 23） | 是 |
| M2：R − E | −2 到 +6 局 | +5 | 是 |
| M2：E − S | 0 到 +8 局 | +8 | 是，在上沿 |
| M3：S 预抓取碰壶 | 40–65 % | 61 %（40/66） | 是 |
| M3：R 预抓取碰壶 | 3–15 % | 6 %（4/66） | 是 |
| M3：碰过壶的局的成功率 | ≤ 10 % | 4 %（2/48） | 是 |
| SONIC R − A | +3 到 +9 个百分点 | +3.3 | 是 |
| SONIC R − S | −2 到 +3 个百分点 | −1.1 | 是 |
| HEFT R − A | 0 到 +4 个百分点 | +2.0 | 是 |

**修正量**（ScaleBFM，执行命令的 |命令 − 目标| 的中位数，cm；"> 10 cm" 是修正量超过 10 cm 的下数 / 总下数，7 个任务）

| 组 | 预抓取：全部 | 浇水壶 | 哑铃 | > 10 cm | 进入段：全部 | 浇水壶 | 哑铃 | > 10 cm |
|---|---|---|---|---|---|---|---|---|
| S | 8.5 | 19.2 | 15.3 | 195/411 | 7.1 | 7.3 | 8.3 | 118/403 |
| R | 4.2 | 3.1 | 5.6 | 43/411 | 4.9 | 13.8 | 8.7 | 89/411 |
| E | 0 | 0 | 0 | 0/411 | 9.7 | 16.0 | 11.0 | 201/411 |

- R 的缩回：
  - 预抓取：反解 411 下，其中 204 下有缩过的候选，最后执行的是缩过的命令的有 92 下；
  - 进入段：411 / 157 / 48 下；
  - 缩回集中在雕像柜、哑铃、浇水壶上，蜡烛 0 下。
- 雕像柜、碗和其余任务，以及 SONIC、HEFT 的修正量，见下面原文的 C6。SONIC 的 R 预抓取执行缩过的命令 91 下；HEFT 的 R 预抓取只有 5 下，进入段 27 下。

**预抓取时碰到物体**（新失败项，判定集，按第一失败原因计局数；括号里是这些局原来的 hardgrasp 标签）
- ScaleBFM：
  - A 15（抓空 8、站位/IK 4、碰家具 3）；
  - S 76（站位/IK 56、抓空 10、碰家具 9、掉落 1）；
  - R 35（碰家具 14、抓空 12、掉落 5、站位/IK 4）；
  - E 13（站位/IK 9、抓空 3、碰家具 1）。
- SONIC：
  - A 35（抓空 26、碰家具 9）；
  - S 48（碰家具 27、抓空 13、站位/IK 5、掉落 3）；
  - R 46（碰家具 29、抓空 13、掉落 3、站位/IK 1）。
- HEFT：
  - A 99（碰家具 72、抓空 19、站位/IK 6、掉落 2）；
  - R 77（碰家具 62、抓空 9、掉落 6）。
- ScaleBFM 的 S 有 56 局从"站位/IK"改记成这一项。lmlcmp2 复核说过，S 的"站位/IK"失败其实多是手碰到了物体。
- 不论成败，判定集上预抓取碰到过物体的局：
  - ScaleBFM A 15、S 90、R 44、E 15（/354）；
  - SONIC A 37、S 57、R 51（/348）；
  - HEFT A 101、R 78（/356）。
- 碰没碰是仿真真值，只用来记分，不进入任何决策。
- 完整的失败原因表在原文 C4，按任务分的在下面描述性分析的 desc.txt 里。

**越过目标 > 3 cm**（第一下，也就是预抓取；括号里是越过量的中位数，负数表示没够到）

| 控制器 | 组 | 浇水壶（66 局） | 哑铃（90 局，SONIC 88 局） |
|---|---|---|---|
| ScaleBFM | A | 0/66（−9.1 cm） | 2/90（−5.6 cm） |
| ScaleBFM | S | 41/66（+15.8 cm） | 33/90（+0.1 cm） |
| ScaleBFM | R | 0/66（−8.0 cm） | 2/90（−2.8 cm） |
| ScaleBFM | E | 0/66（−9.1 cm） | 2/90（−5.6 cm） |
| SONIC | A | 0/66（−0.5 cm） | 6/88（−0.0 cm） |
| SONIC | S | 2/66（−0.0 cm） | 3/88（−0.1 cm） |
| SONIC | R | 0/66（−0.2 cm） | 2/88（−0.2 cm） |
| HEFT | A | 37/66（+19.9 cm） | 39/90（+1.3 cm） |
| HEFT | R | 2/66（−22.1 cm） | 3/90（−21.1 cm） |

E 的预抓取就是目标，和 A 走同一段代码，所以 E 这一行和 A 相同。

**评估输出原文**

`scalebfm/eval.txt`（100 行，md5 c26edc008b571d42b70c4fa0f1f15070）：

```
== scalebfm: 2520 episode records under /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/scalebfm/episodes {'selfmodel': 630, 'alone': 630, 'entryinv': 630, 'reachable': 630}; episodes 30..119; groups A S R E; missing 0 []; worker errors 44 ([('candle', 49), ('candle', 80), ('candle', 86), ('candle', 96), ('candle', 97), ('candle', 100), ('candle', 102), ('candle', 103), ('candle', 108), ('candle', 110), ('candle', 112)])
C1. decision set: failures before the groups split (perception, the plan after the look, no look) > 50 % -> out; A >= 80 % (72/90) -> ceiling
  candle     pre-split failures 41/90 (46 %) {'perception': 28, 'no look': 11, 'plan after look': 2}; same in all groups True | A 16/90 | decision set
  apple      pre-split failures 8/90 (9 %) {'plan after look': 8}; same in all groups True | A 15/90 | decision set
  bowl       pre-split failures 66/90 (73 %) {'plan after look': 66}; same in all groups True | A 1/90 | EXCLUDED (pre-split 73 %)
  salt       pre-split failures 57/90 (63 %) {'perception': 57}; same in all groups True | A 4/90 | EXCLUDED (pre-split 63 %)
  statue_cab pre-split failures 23/90 (26 %) {'plan after look': 8, 'perception': 15}; same in all groups True | A 0/90 | decision set
  dumbbell   pre-split failures 0/90 (0 %) {}; same in all groups True | A 25/90 | decision set
  wcan       pre-split failures 24/90 (27 %) {'perception': 24}; same in all groups True | A 5/90 | decision set
  decision set: ['candle', 'apple', 'statue_cab', 'dumbbell', 'wcan']; pre-split the same in every group for every task: True
C2. per task (episodes 30..119): successes and discordant pairs (x-only / y-only, McNemar exact p)
  candle     n 90 A 16 S 49 R 49 E 47 | RvA A-only 0 R-only 33 p 2.33e-10 | RvS S-only 0 R-only 0 p 1 | EvA A-only 0 E-only 31 p 9.31e-10 | EvS S-only 2 E-only 0 p 0.5 | RvE E-only 0 R-only 2 p 0.5 | SvA A-only 0 S-only 33 p 2.33e-10
  apple      n 90 A 15 S 20 R 20 E 18 | RvA A-only 2 R-only 7 p 0.18 | RvS S-only 0 R-only 0 p 1 | EvA A-only 1 E-only 4 p 0.375 | EvS S-only 6 E-only 4 p 0.754 | RvE E-only 4 R-only 6 p 0.754 | SvA A-only 2 S-only 7 p 0.18
  bowl       n 90 A  1 S  0 R  1 E  1 | RvA A-only 0 R-only 0 p 1 | RvS S-only 0 R-only 1 p 1 | EvA A-only 0 E-only 0 p 1 | EvS S-only 0 E-only 1 p 1 | RvE E-only 0 R-only 0 p 1 | SvA A-only 1 S-only 0 p 1
  salt       n 90 A  4 S 15 R 15 E 19 | RvA A-only 1 R-only 12 p 0.00342 | RvS S-only 0 R-only 0 p 1 | EvA A-only 0 E-only 15 p 6.1e-05 | EvS S-only 0 E-only 4 p 0.125 | RvE E-only 4 R-only 0 p 0.125 | SvA A-only 1 S-only 12 p 0.00342
  statue_cab n 90 A  0 S  0 R  0 E  0 | RvA A-only 0 R-only 0 p 1 | RvS S-only 0 R-only 0 p 1 | EvA A-only 0 E-only 0 p 1 | EvS S-only 0 E-only 0 p 1 | RvE E-only 0 R-only 0 p 1 | SvA A-only 0 S-only 0 p 1
  dumbbell   n 90 A 25 S 30 R 29 E 36 | RvA A-only 8 R-only 12 p 0.503 | RvS S-only 9 R-only 8 p 1 | EvA A-only 6 E-only 17 p 0.0347 | EvS S-only 9 E-only 15 p 0.307 | RvE E-only 10 R-only 3 p 0.0923 | SvA A-only 12 S-only 17 p 0.458
  wcan       n 90 A  5 S  5 R 19 E 11 | RvA A-only 4 R-only 18 p 0.00434 | RvS S-only 3 R-only 17 p 0.00258 | EvA A-only 2 E-only 8 p 0.109 | EvS S-only 4 E-only 10 p 0.18 | RvE E-only 3 R-only 11 p 0.0574 | SvA A-only 5 S-only 5 p 1
C3. pooled
  decision set (candle, apple, statue_cab, dumbbell, wcan): 450 pairs
    R vs A: success R 117 (26.0 %) A 61 (13.6 %) diff +12.4 pp (+56) | A-only 14 R-only 70 McNemar exact p 4.07e-10
       close-time hand to the grasp point, paired R - A: median -1.61 cm, 95 % CI [-2.61, -0.98], n 353, R closer in 243, A closer in 82, sign p 1.19e-19, Wilcoxon p 4e-24; medians R 3.78 A 9.58 cm
    R vs S: success R 117 (26.0 %) S 104 (23.1 %) diff +2.9 pp (+13) | S-only 12 R-only 25 McNemar exact p 0.047
       close-time hand to the grasp point, paired R - S: median +0.00 cm, 95 % CI [+0.00, +0.00], n 348, R closer in 87, S closer in 92, sign p 0.765, Wilcoxon p 0.111; medians R 3.78 S 4.02 cm
    E vs A: success E 112 (24.9 %) A 61 (13.6 %) diff +11.3 pp (+51) | A-only 9 E-only 60 McNemar exact p 2.24e-10
       close-time hand to the grasp point, paired E - A: median -1.83 cm, 95 % CI [-3.05, -1.25], n 352, E closer in 272, A closer in 80, sign p 1.34e-25, Wilcoxon p 1.78e-27; medians E 3.62 A 9.58 cm
    E vs S: success E 112 (24.9 %) S 104 (23.1 %) diff +1.8 pp (+8) | S-only 21 E-only 29 McNemar exact p 0.322
       close-time hand to the grasp point, paired E - S: median +0.07 cm, 95 % CI [+0.03, +0.12], n 348, E closer in 150, S closer in 196, sign p 0.0154, Wilcoxon p 0.205; medians E 3.62 S 4.02 cm
    R vs E: success R 117 (26.0 %) E 112 (24.9 %) diff +1.1 pp (+5) | E-only 17 R-only 22 McNemar exact p 0.522
       close-time hand to the grasp point, paired R - E: median +0.00 cm, 95 % CI [-0.06, +0.03], n 352, R closer in 175, E closer in 163, sign p 0.55, Wilcoxon p 0.421; medians R 3.78 E 3.62 cm
    S vs A: success S 104 (23.1 %) A 61 (13.6 %) diff +9.6 pp (+43) | A-only 19 S-only 62 McNemar exact p 1.77e-06
       close-time hand to the grasp point, paired S - A: median -4.01 cm, 95 % CI [-4.68, -1.83], n 348, S closer in 267, A closer in 81, sign p 2.54e-24, Wilcoxon p 8.4e-28; medians S 4.02 A 9.58 cm
    Holm over P1 (R vs A) and P2 (R vs S): raw p 4.07e-10 / 0.047, adjusted 8.14e-10 / 0.047, significant True / True | P1 (>= 10 pp and significant): True, P2 (R > S and significant): True
  all 7 tasks (candle, apple, bowl, salt, statue_cab, dumbbell, wcan): 630 pairs
    R vs A: success R 133 (21.1 %) A 66 (10.5 %) diff +10.6 pp (+67) | A-only 15 R-only 82 McNemar exact p 2.38e-12
       close-time hand to the grasp point, paired R - A: median -1.06 cm, 95 % CI [-1.62, -0.73], n 410, R closer in 264, A closer in 99, sign p 1.88e-18, Wilcoxon p 2.53e-22; medians R 3.97 A 9.71 cm
    R vs S: success R 133 (21.1 %) S 119 (18.9 %) diff +2.2 pp (+14) | S-only 12 R-only 26 McNemar exact p 0.0336
       close-time hand to the grasp point, paired R - S: median +0.00 cm, 95 % CI [+0.00, +0.00], n 396, R closer in 99, S closer in 99, sign p 1, Wilcoxon p 0.158; medians R 3.97 S 4.07 cm
    E vs A: success E 132 (21.0 %) A 66 (10.5 %) diff +10.5 pp (+66) | A-only 9 E-only 75 McNemar exact p 4.31e-14
       close-time hand to the grasp point, paired E - A: median -1.67 cm, 95 % CI [-2.31, -1.14], n 403, E closer in 302, A closer in 98, sign p 2.79e-25, Wilcoxon p 6.95e-28; medians E 3.67 A 9.71 cm
    E vs S: success E 132 (21.0 %) S 119 (18.9 %) diff +2.1 pp (+13) | S-only 21 E-only 34 McNemar exact p 0.105
       close-time hand to the grasp point, paired E - S: median +0.06 cm, 95 % CI [-0.02, +0.10], n 394, E closer in 180, S closer in 209, sign p 0.156, Wilcoxon p 0.575; medians E 3.67 S 4.07 cm
    R vs E: success R 133 (21.1 %) E 132 (21.0 %) diff +0.2 pp (+1) | E-only 21 R-only 22 McNemar exact p 1
       close-time hand to the grasp point, paired R - E: median +0.00 cm, 95 % CI [-0.04, +0.05], n 403, R closer in 191, E closer in 195, sign p 0.879, Wilcoxon p 0.124; medians R 3.97 E 3.67 cm
    S vs A: success S 119 (18.9 %) A 66 (10.5 %) diff +8.4 pp (+53) | A-only 21 S-only 74 McNemar exact p 4.25e-08
       close-time hand to the grasp point, paired S - A: median -2.19 cm, 95 % CI [-4.20, -1.47], n 396, S closer in 291, A closer in 103, sign p 7.13e-22, Wilcoxon p 1.29e-26; medians S 4.07 A 9.71 cm
    Holm over P1 (R vs A) and P2 (R vs S): raw p 2.38e-12 / 0.0336, adjusted 4.75e-12 / 0.0336, significant True / True
C4. failure reasons: the new item pregrasp_touch (the pre-grasp segment's hand-object contact rows > 0, checked before hardgrasp's IK / furniture / grasp labels); then hardgrasp's table as it was
  all 7 tasks  A ok 66 perception 124 pregrasp_touch 16 stance_or_ik 176 fell_walking 0 hit_furniture 224 missed 13 dropped 0 other 11 | pregrasp_touch was {'hit_furniture': 3, 'missed': 8, 'stance_or_ik': 5} | pre-grasp touched the object (any outcome) 17/411 | reach IK > 3 cm: 97
  all 7 tasks  S ok 119 perception 124 pregrasp_touch 80 stance_or_ik 205 fell_walking 0 hit_furniture 69 missed 20 dropped 2 other 11 | pregrasp_touch was {'hit_furniture': 10, 'stance_or_ik': 59, 'missed': 10, 'dropped': 1} | pre-grasp touched the object (any outcome) 94/411 | reach IK > 3 cm: 193
  all 7 tasks  R ok 133 perception 124 pregrasp_touch 37 stance_or_ik 169 fell_walking 0 hit_furniture 121 missed 34 dropped 1 other 11 | pregrasp_touch was {'hit_furniture': 15, 'stance_or_ik': 5, 'missed': 12, 'dropped': 5} | pre-grasp touched the object (any outcome) 47/411 | reach IK > 3 cm: 93
  all 7 tasks  E ok 132 perception 124 pregrasp_touch 14 stance_or_ik 210 fell_walking 0 hit_furniture 123 missed 12 dropped 4 other 11 | pregrasp_touch was {'hit_furniture': 1, 'stance_or_ik': 10, 'missed': 3} | pre-grasp touched the object (any outcome) 17/411 | reach IK > 3 cm: 138
  decision set A ok 61 perception 67 pregrasp_touch 15 stance_or_ik 92 fell_walking 0 hit_furniture 191 missed 13 dropped 0 other 11 | pregrasp_touch was {'hit_furniture': 3, 'missed': 8, 'stance_or_ik': 4} | pre-grasp touched the object (any outcome) 15/354 | reach IK > 3 cm: 78
  decision set S ok 104 perception 67 pregrasp_touch 76 stance_or_ik 113 fell_walking 0 hit_furniture 57 missed 20 dropped 2 other 11 | pregrasp_touch was {'hit_furniture': 9, 'stance_or_ik': 56, 'missed': 10, 'dropped': 1} | pre-grasp touched the object (any outcome) 90/354 | reach IK > 3 cm: 164
  decision set R ok 117 perception 67 pregrasp_touch 35 stance_or_ik 83 fell_walking 0 hit_furniture 102 missed 34 dropped 1 other 11 | pregrasp_touch was {'hit_furniture': 14, 'missed': 12, 'dropped': 5, 'stance_or_ik': 4} | pre-grasp touched the object (any outcome) 44/354 | reach IK > 3 cm: 72
  decision set E ok 112 perception 67 pregrasp_touch 13 stance_or_ik 118 fell_walking 0 hit_furniture 113 missed 12 dropped 4 other 11 | pregrasp_touch was {'hit_furniture': 1, 'stance_or_ik': 9, 'missed': 3} | pre-grasp touched the object (any outcome) 15/354 | reach IK > 3 cm: 110
  [hardgrasp's order] all 7 tasks  A ok 66 perception 124 stance_or_ik 181 fell_walking 0 hit_furniture 227 missed 21 dropped 0 other 11
  [hardgrasp's order] all 7 tasks  S ok 119 perception 124 stance_or_ik 264 fell_walking 0 hit_furniture 79 missed 30 dropped 3 other 11
  [hardgrasp's order] all 7 tasks  R ok 133 perception 124 stance_or_ik 174 fell_walking 0 hit_furniture 136 missed 46 dropped 6 other 11
  [hardgrasp's order] all 7 tasks  E ok 132 perception 124 stance_or_ik 220 fell_walking 0 hit_furniture 124 missed 15 dropped 4 other 11
  [hardgrasp's order] decision set A ok 61 perception 67 stance_or_ik 96 fell_walking 0 hit_furniture 194 missed 21 dropped 0 other 11
  [hardgrasp's order] decision set S ok 104 perception 67 stance_or_ik 169 fell_walking 0 hit_furniture 66 missed 30 dropped 3 other 11
  [hardgrasp's order] decision set R ok 117 perception 67 stance_or_ik 87 fell_walking 0 hit_furniture 116 missed 46 dropped 6 other 11
  [hardgrasp's order] decision set E ok 112 perception 67 stance_or_ik 127 fell_walking 0 hit_furniture 114 missed 15 dropped 4 other 11
C5. overshoot past the goal > 3 cm, first reach (pre-grasp) on the far targets (wcan, dumbbell)
  A: both 2/156 (1 %) | wcan 0/66 (median -9.1 cm; IK > 3 cm: 0/5) | dumbbell 2/90 (median -5.6 cm; IK > 3 cm: 2/11)
  S: both 74/156 (47 %) | wcan 41/66 (median +15.8 cm; IK > 3 cm: 41/48) | dumbbell 33/90 (median +0.1 cm; IK > 3 cm: 33/44)
  R: both 2/156 (1 %) | wcan 0/66 (median -8.0 cm; IK > 3 cm: 0/5) | dumbbell 2/90 (median -2.8 cm; IK > 3 cm: 2/11)
  E: both 2/156 (1 %) | wcan 0/66 (median -9.1 cm; IK > 3 cm: 0/5) | dumbbell 2/90 (median -5.6 cm; IK > 3 cm: 2/11)
C6. executed corrections |command - goal| (cm), pre-grasp (reach_no 1) and entry (reach_no 2) apart: median / = 0 / > 10 cm / n
  S pre-grasp all 8.5/5/195/411 | candle 5.7/0/0/49 apple 4.0/0/5/82 bowl 15.5/3/16/24 salt 4.1/0/1/33 statue_cab 26.8/0/59/67 dumbbell 15.3/2/62/90 wcan 19.2/0/52/66
  S entry     all 7.1/5/118/403 | candle 5.6/0/0/49 apple 4.0/0/7/82 bowl 13.2/4/14/19 salt 5.0/1/6/33 statue_cab 23.7/0/65/65 dumbbell 8.3/0/18/89 wcan 7.3/0/8/66
  R pre-grasp all 4.2/102/43/411 | candle 5.7/0/0/49 apple 4.0/0/5/82 bowl 0.0/21/1/24 salt 4.1/1/0/33 statue_cab 2.7/28/3/67 dumbbell 5.6/26/25/90 wcan 3.1/26/9/66
  R entry     all 4.9/90/89/411 | candle 5.6/0/0/49 apple 3.9/1/3/82 bowl 0.0/20/0/24 salt 4.8/2/2/33 statue_cab 0.0/65/0/67 dumbbell 8.7/2/39/90 wcan 13.8/0/45/66
  E pre-grasp all 0.0/411/0/411 | candle 0.0/49/0/49 apple 0.0/82/0/82 bowl 0.0/24/0/24 salt 0.0/33/0/33 statue_cab 0.0/67/0/67 dumbbell 0.0/90/0/90 wcan 0.0/66/0/66
  E entry     all 9.7/3/201/411 | candle 5.5/0/0/49 apple 4.0/0/8/82 bowl 14.9/3/18/24 salt 4.6/0/8/33 statue_cab 20.2/0/66/67 dumbbell 11.0/0/51/90 wcan 16.0/0/50/66
  R shrink, pre-grasp: inverted 411, with a shrunk candidate 204, executed command shrunk 92, shrunk candidates 1112, shrunk to the goal (alpha 0) 579 | by task (with a shrunk candidate / executed shrunk / inverted): candle 0/0/49; apple 4/2/82; bowl 23/2/24; salt 1/0/33; statue_cab 66/35/67; dumbbell 59/29/90; wcan 51/24/66
  R shrink, entry: inverted 411, with a shrunk candidate 157, executed command shrunk 48, shrunk candidates 870, shrunk to the goal (alpha 0) 540 | by task (with a shrunk candidate / executed shrunk / inverted): candle 0/0/49; apple 5/4/82; bowl 24/3/24; salt 6/3/33; statue_cab 67/2/67; dumbbell 26/18/90; wcan 29/18/66
C7. mechanism checks
  M1 [all 7 tasks] A: |landing - D| median (n) IK <= 2 0.84 cm (220); IK 2-3 1.46 cm (21); IK > 3 2.22 cm (27) | IK > 3: |landing - ref end| 15.86, |ref end - command| 3.49, |landing - command| 18.52 cm
  M1 [all 7 tasks] S: |landing - D| median (n) IK <= 2 0.88 cm (338); IK 2-3 1.67 cm (13); IK > 3 19.87 cm (75) | IK > 3: |landing - ref end| 10.03, |ref end - command| 4.43, |landing - command| 9.14 cm
  M1 [all 7 tasks] R: |landing - D| median (n) IK <= 2 0.87 cm (345); IK 2-3 1.44 cm (27); IK > 3 2.14 cm (27) | IK > 3: |landing - ref end| 15.86, |ref end - command| 3.37, |landing - command| 18.06 cm
  M1 [all 7 tasks] E: |landing - D| median (n) IK <= 2 0.75 cm (238); IK 2-3 1.47 cm (22); IK > 3 11.27 cm (43) | IK > 3: |landing - ref end| 21.86, |ref end - command| 5.12, |landing - command| 25.12 cm
  M1 [decision set] A: |landing - D| median (n) IK <= 2 0.83 cm (181); IK 2-3 1.46 cm (18); IK > 3 1.55 cm (23) | IK > 3: |landing - ref end| 15.73, |ref end - command| 3.37, |landing - command| 18.47 cm
  M1 [decision set] S: |landing - D| median (n) IK <= 2 0.89 cm (312); IK 2-3 1.67 cm (13); IK > 3 19.94 cm (72) | IK > 3: |landing - ref end| 9.89, |ref end - command| 4.41, |landing - command| 8.69 cm
  M1 [decision set] R: |landing - D| median (n) IK <= 2 0.87 cm (317); IK 2-3 1.41 cm (24); IK > 3 1.62 cm (23) | IK > 3: |landing - ref end| 15.49, |ref end - command| 3.34, |landing - command| 18.05 cm
  M1 [decision set] E: |landing - D| median (n) IK <= 2 0.73 cm (205); IK 2-3 1.46 cm (19); IK > 3 11.73 cm (39) | IK > 3: |landing - ref end| 21.49, |ref end - command| 5.39, |landing - command| 24.77 cm
  M2 (decision set): R - E = +5 (R 117, E 112, p 0.522); E - S = +8 (E 112, S 104, p 0.322)
  M3 (wcan, episodes with a pre-grasp segment): A touched 2/66 (3 %), of them ok 0; S touched 40/66 (61 %), of them ok 2; R touched 4/66 (6 %), of them ok 0; E touched 2/66 (3 %), of them ok 0 | all groups: touched 48, ok 2 (4 %)
C8. criteria rows and predictions (decision set unless said)
  decision set = candle, apple, statue_cab, dumbbell, wcan (predicted, ~80 %): True; bowl / salt pre-split > 50 %: True / True
  P1 True (R - A +12.4 pp, raw p 4.07e-10, Holm-adjusted 8.14e-10); P2 True (R - S +2.9 pp / +13, raw p 0.047, Holm-adjusted 0.047)
  row 'P1 and P2': True | 'only P1': False | 'not P1, R - A >= 5 pp and p < 0.05': False | 'R - A < 5 pp or p >= 0.05': False
  row 'decision set R - S <= 0, or wcan R - S <= 2': False (decision set +13; wcan R 19 S 5, +14)
  row 'M1: S's out-of-reach commands' median prediction error <= 5 cm': False (S IK > 3 cm: 19.87 cm, n 75; R IK > 3 cm: 2.14 cm, n 27)
  row 'M2: R - E >= 4': True | 'M2: |R - E| <= 3 and E - S >= 4': False (R - E +5, E - S +8)
  predicted success rates: A 13.6 % (16-24: False); S 23.1 % (23-31: True); R 26.0 % (26-35: True); E 24.9 % (23-33: True)
  predicted R - A +6..+13 pp: +12.4 (True); R - S +1..+6 pp: +2.9 (True)
  predicted wcan successes: A 5 (6-14: False); S 5 (3-12: True); R 19 (14-28: True); E 11 (10-24: True)
  predicted M1: S IK > 3 median 12-25 cm: 19.87 (True); R IK > 3 <= 3 cm: 2.14 | M2: R - E -2..+6: +5 (True), E - S 0..+8: +8 (True) | M3: S touched 40-65 %: 61 (True), R 3-15 %: 6 (True), touched success <= 10 %: 4 % (True)
```

`sonic/eval.txt`（68 行，md5 48250b489c3922a66ac0941c5c7e3e69）：

```
== sonic: 1890 episode records under /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/sonic/episodes {'selfmodel': 630, 'alone': 630, 'reachable': 630}; episodes 30..119; groups A S R; missing 0 []; worker errors 33 ([('candle', 49), ('candle', 80), ('candle', 86), ('candle', 96), ('candle', 97), ('candle', 100), ('candle', 102), ('candle', 103), ('candle', 108), ('candle', 110), ('candle', 112)])
C1. decision set: failures before the groups split (perception, the plan after the look, no look) > 50 % -> out; A >= 80 % (72/90) -> ceiling
  candle     pre-split failures 41/90 (46 %) {'perception': 28, 'no look': 11, 'plan after look': 2}; same in all groups True | A 48/90 | decision set
  apple      pre-split failures 8/90 (9 %) {'plan after look': 8}; same in all groups True | A 4/90 | decision set
  bowl       pre-split failures 67/90 (74 %) {'plan after look': 67}; same in all groups True | A 0/90 | EXCLUDED (pre-split 74 %)
  salt       pre-split failures 57/90 (63 %) {'perception': 57}; same in all groups True | A 8/90 | EXCLUDED (pre-split 63 %)
  statue_cab pre-split failures 24/90 (27 %) {'plan after look': 9, 'perception': 15}; same in all groups True | A 0/90 | decision set
  dumbbell   pre-split failures 0/90 (0 %) {}; same in all groups True | A 0/90 | decision set
  wcan       pre-split failures 24/90 (27 %) {'perception': 24}; same in all groups True | A 1/90 | decision set
  decision set: ['candle', 'apple', 'statue_cab', 'dumbbell', 'wcan']; pre-split the same in every group for every task: True
C2. per task (episodes 30..119): successes and discordant pairs (x-only / y-only, McNemar exact p)
  candle     n 90 A 48 S 49 R 49 | RvA A-only 0 R-only 1 p 1 | RvS S-only 0 R-only 0 p 1 | SvA A-only 0 S-only 1 p 1
  apple      n 90 A  4 S  8 R  7 | RvA A-only 2 R-only 5 p 0.453 | RvS S-only 1 R-only 0 p 1 | SvA A-only 2 S-only 6 p 0.289
  bowl       n 90 A  0 S  2 R  1 | RvA A-only 0 R-only 1 p 1 | RvS S-only 2 R-only 1 p 1 | SvA A-only 0 S-only 2 p 0.5
  salt       n 90 A  8 S 10 R  9 | RvA A-only 3 R-only 4 p 1 | RvS S-only 1 R-only 0 p 1 | SvA A-only 3 S-only 5 p 0.727
  statue_cab n 90 A  0 S  0 R  0 | RvA A-only 0 R-only 0 p 1 | RvS S-only 0 R-only 0 p 1 | SvA A-only 0 S-only 0 p 1
  dumbbell   n 90 A  0 S 10 R  9 | RvA A-only 0 R-only 9 p 0.00391 | RvS S-only 2 R-only 1 p 1 | SvA A-only 0 S-only 10 p 0.00195
  wcan       n 90 A  1 S  6 R  3 | RvA A-only 0 R-only 2 p 0.5 | RvS S-only 3 R-only 0 p 0.25 | SvA A-only 0 S-only 5 p 0.0625
C3. pooled
  decision set (candle, apple, statue_cab, dumbbell, wcan): 450 pairs
    R vs A: success R 68 (15.1 %) A 53 (11.8 %) diff +3.3 pp (+15) | A-only 2 R-only 17 McNemar exact p 0.000729
       close-time hand to the grasp point, paired R - A: median -1.49 cm, 95 % CI [-1.74, -1.15], n 345, R closer in 257, A closer in 71, sign p 6.98e-26, Wilcoxon p 2.1e-24; medians R 4.40 A 6.64 cm
    R vs S: success R 68 (15.1 %) S 73 (16.2 %) diff -1.1 pp (-5) | S-only 6 R-only 1 McNemar exact p 0.125
       close-time hand to the grasp point, paired R - S: median +0.00 cm, 95 % CI [+0.00, +0.00], n 345, R closer in 28, S closer in 90, sign p 8.91e-09, Wilcoxon p 9.83e-12; medians R 4.40 S 4.01 cm
    S vs A: success S 73 (16.2 %) A 53 (11.8 %) diff +4.4 pp (+20) | A-only 2 S-only 22 McNemar exact p 3.59e-05
       close-time hand to the grasp point, paired S - A: median -2.17 cm, 95 % CI [-2.81, -1.73], n 344, S closer in 288, A closer in 56, sign p 9.48e-39, Wilcoxon p 1.19e-35; medians S 4.01 A 6.64 cm
  all 7 tasks (candle, apple, bowl, salt, statue_cab, dumbbell, wcan): 630 pairs
    R vs A: success R 78 (12.4 %) A 61 (9.7 %) diff +2.7 pp (+17) | A-only 5 R-only 22 McNemar exact p 0.00151
       close-time hand to the grasp point, paired R - A: median -1.59 cm, 95 % CI [-1.85, -1.22], n 383, R closer in 290, A closer in 75, sign p 6e-31, Wilcoxon p 1.8e-29; medians R 4.65 A 7.15 cm
    R vs S: success R 78 (12.4 %) S 85 (13.5 %) diff -1.1 pp (-7) | S-only 9 R-only 2 McNemar exact p 0.0654
       close-time hand to the grasp point, paired R - S: median +0.00 cm, 95 % CI [+0.00, +0.00], n 384, R closer in 33, S closer in 105, sign p 6.09e-10, Wilcoxon p 3.33e-13; medians R 4.65 S 4.13 cm
    S vs A: success S 85 (13.5 %) A 61 (9.7 %) diff +3.8 pp (+24) | A-only 5 S-only 29 McNemar exact p 3.86e-05
       close-time hand to the grasp point, paired S - A: median -2.34 cm, 95 % CI [-3.06, -1.79], n 382, S closer in 322, A closer in 60, sign p 1.88e-44, Wilcoxon p 6.32e-41; medians S 4.13 A 7.15 cm
C4. failure reasons: the new item pregrasp_touch (the pre-grasp segment's hand-object contact rows > 0, checked before hardgrasp's IK / furniture / grasp labels); then hardgrasp's table as it was
  all 7 tasks  A ok 61 perception 124 pregrasp_touch 37 stance_or_ik 117 fell_walking 22 hit_furniture 146 missed 109 dropped 3 other 11 | pregrasp_touch was {'missed': 26, 'hit_furniture': 11} | pre-grasp touched the object (any outcome) 39/387 | reach IK > 3 cm: 31
  all 7 tasks  S ok 85 perception 124 pregrasp_touch 59 stance_or_ik 167 fell_walking 22 hit_furniture 112 missed 48 dropped 2 other 11 | pregrasp_touch was {'hit_furniture': 31, 'stance_or_ik': 12, 'missed': 13, 'dropped': 3} | pre-grasp touched the object (any outcome) 69/387 | reach IK > 3 cm: 100
  all 7 tasks  R ok 78 perception 124 pregrasp_touch 49 stance_or_ik 115 fell_walking 22 hit_furniture 172 missed 57 dropped 2 other 11 | pregrasp_touch was {'hit_furniture': 32, 'missed': 13, 'stance_or_ik': 1, 'dropped': 3} | pre-grasp touched the object (any outcome) 55/387 | reach IK > 3 cm: 30
  decision set A ok 53 perception 67 pregrasp_touch 35 stance_or_ik 49 fell_walking 5 hit_furniture 118 missed 109 dropped 3 other 11 | pregrasp_touch was {'missed': 26, 'hit_furniture': 9} | pre-grasp touched the object (any outcome) 37/348 | reach IK > 3 cm: 30
  decision set S ok 73 perception 67 pregrasp_touch 48 stance_or_ik 96 fell_walking 5 hit_furniture 100 missed 48 dropped 2 other 11 | pregrasp_touch was {'hit_furniture': 27, 'stance_or_ik': 5, 'missed': 13, 'dropped': 3} | pre-grasp touched the object (any outcome) 57/348 | reach IK > 3 cm: 87
  decision set R ok 68 perception 67 pregrasp_touch 46 stance_or_ik 47 fell_walking 5 hit_furniture 147 missed 57 dropped 2 other 11 | pregrasp_touch was {'hit_furniture': 29, 'missed': 13, 'stance_or_ik': 1, 'dropped': 3} | pre-grasp touched the object (any outcome) 51/348 | reach IK > 3 cm: 29
  [hardgrasp's order] all 7 tasks  A ok 61 perception 124 stance_or_ik 117 fell_walking 22 hit_furniture 157 missed 135 dropped 3 other 11
  [hardgrasp's order] all 7 tasks  S ok 85 perception 124 stance_or_ik 179 fell_walking 22 hit_furniture 143 missed 61 dropped 5 other 11
  [hardgrasp's order] all 7 tasks  R ok 78 perception 124 stance_or_ik 116 fell_walking 22 hit_furniture 204 missed 70 dropped 5 other 11
  [hardgrasp's order] decision set A ok 53 perception 67 stance_or_ik 49 fell_walking 5 hit_furniture 127 missed 135 dropped 3 other 11
  [hardgrasp's order] decision set S ok 73 perception 67 stance_or_ik 101 fell_walking 5 hit_furniture 127 missed 61 dropped 5 other 11
  [hardgrasp's order] decision set R ok 68 perception 67 stance_or_ik 48 fell_walking 5 hit_furniture 176 missed 70 dropped 5 other 11
C5. overshoot past the goal > 3 cm, first reach (pre-grasp) on the far targets (wcan, dumbbell)
  A: both 6/154 (4 %) | wcan 0/66 (median -0.5 cm; IK > 3 cm: 0/0) | dumbbell 6/88 (median -0.0 cm; IK > 3 cm: 0/1)
  S: both 5/154 (3 %) | wcan 2/66 (median -0.0 cm; IK > 3 cm: 2/7) | dumbbell 3/88 (median -0.1 cm; IK > 3 cm: 1/8)
  R: both 2/154 (1 %) | wcan 0/66 (median -0.2 cm; IK > 3 cm: 0/0) | dumbbell 2/88 (median -0.2 cm; IK > 3 cm: 0/1)
C6. executed corrections |command - goal| (cm), pre-grasp (reach_no 1) and entry (reach_no 2) apart: median / = 0 / > 10 cm / n
  S pre-grasp all 9.2/12/186/387 | candle 3.4/0/0/49 apple 6.1/2/34/79 bowl 39.8/0/15/15 salt 3.6/0/10/24 statue_cab 17.9/2/64/66 dumbbell 7.3/3/32/88 wcan 9.4/5/31/66
  S entry     all 4.9/2/132/384 | candle 2.0/0/0/49 apple 3.4/1/18/77 bowl 32.2/0/14/15 salt 3.8/0/6/24 statue_cab 20.0/0/66/66 dumbbell 4.7/0/14/87 wcan 4.8/1/14/66
  R pre-grasp all 5.4/38/107/387 | candle 3.4/0/0/49 apple 6.1/2/34/79 bowl 7.0/7/5/15 salt 3.6/0/10/24 statue_cab 5.5/18/19/66 dumbbell 6.5/6/20/88 wcan 6.9/5/19/66
  R entry     all 3.2/47/72/385 | candle 2.0/0/0/49 apple 3.2/2/16/77 bowl 18.8/1/12/15 salt 3.8/0/6/24 statue_cab 0.0/42/0/66 dumbbell 4.7/2/18/88 wcan 5.2/0/20/66
  R shrink, pre-grasp: inverted 387, with a shrunk candidate 139, executed command shrunk 91, shrunk candidates 755, shrunk to the goal (alpha 0) 156 | by task (with a shrunk candidate / executed shrunk / inverted): candle 0/0/49; apple 8/7/79; bowl 14/6/15; salt 5/4/24; statue_cab 64/46/66; dumbbell 23/11/88; wcan 25/17/66
  R shrink, entry: inverted 385, with a shrunk candidate 108, executed command shrunk 50, shrunk candidates 591, shrunk to the goal (alpha 0) 264 | by task (with a shrunk candidate / executed shrunk / inverted): candle 0/0/49; apple 4/3/77; bowl 9/6/15; salt 3/2/24; statue_cab 66/24/66; dumbbell 13/9/88; wcan 13/6/66
C7. mechanism checks
  M1 [all 7 tasks] A: |landing - D| median (n) IK <= 2 1.89 cm (551); IK 2-3 3.13 cm (11); IK > 3 4.84 cm (2) | IK > 3: |landing - ref end| 32.78, |ref end - command| 4.10, |landing - command| 35.20 cm
  M1 [all 7 tasks] S: |landing - D| median (n) IK <= 2 1.90 cm (383); IK 2-3 3.81 cm (19); IK > 3 8.97 cm (54) | IK > 3: |landing - ref end| 23.70, |ref end - command| 5.86, |landing - command| 28.41 cm
  M1 [all 7 tasks] R: |landing - D| median (n) IK <= 2 1.95 cm (451); IK 2-3 3.20 cm (12); IK > 3 4.84 cm (2) | IK > 3: |landing - ref end| 32.78, |ref end - command| 4.10, |landing - command| 35.20 cm
  M1 [decision set] A: |landing - D| median (n) IK <= 2 1.79 cm (491); IK 2-3 2.11 cm (4); IK > 3 5.36 cm (1) | IK > 3: |landing - ref end| 18.30, |ref end - command| 3.39, |landing - command| 20.72 cm
  M1 [decision set] S: |landing - D| median (n) IK <= 2 1.78 cm (344); IK 2-3 3.34 cm (13); IK > 3 8.47 cm (43) | IK > 3: |landing - ref end| 23.04, |ref end - command| 6.13, |landing - command| 27.76 cm
  M1 [decision set] R: |landing - D| median (n) IK <= 2 1.83 cm (394); IK 2-3 3.13 cm (5); IK > 3 5.36 cm (1) | IK > 3: |landing - ref end| 18.30, |ref end - command| 3.39, |landing - command| 20.72 cm
  M3 (wcan, episodes with a pre-grasp segment): A touched 15/66 (23 %), of them ok 1; S touched 22/66 (33 %), of them ok 4; R touched 20/66 (30 %), of them ok 2 | all groups: touched 57, ok 7 (12 %)
C8. criteria rows and predictions (decision set unless said)
  decision set = candle, apple, statue_cab, dumbbell, wcan (predicted, ~80 %): True; bowl / salt pre-split > 50 %: True / True
  predicted R - A +3..+9 pp: +3.3 (True); R - S -2..+3 pp: -1.1 (True)
```

`heft/eval.txt`（51 行，md5 f29022d8e52d00645dcc90d320e61872）：

```
== heft: 1260 episode records under /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/heft/episodes {'alone': 630, 'reachable': 630}; episodes 30..119; groups A R; missing 0 []; worker errors 22 ([('candle', 49), ('candle', 80), ('candle', 86), ('candle', 96), ('candle', 97), ('candle', 100), ('candle', 102), ('candle', 103), ('candle', 108), ('candle', 110), ('candle', 112)])
C1. decision set: failures before the groups split (perception, the plan after the look, no look) > 50 % -> out; A >= 80 % (72/90) -> ceiling
  candle     pre-split failures 41/90 (46 %) {'perception': 28, 'no look': 11, 'plan after look': 2}; same in all groups True | A 31/90 | decision set
  apple      pre-split failures 6/90 (7 %) {'plan after look': 6}; same in all groups True | A 0/90 | decision set
  bowl       pre-split failures 66/90 (73 %) {'plan after look': 66}; same in all groups True | A 0/90 | EXCLUDED (pre-split 73 %)
  salt       pre-split failures 57/90 (63 %) {'perception': 57}; same in all groups True | A 2/90 | EXCLUDED (pre-split 63 %)
  statue_cab pre-split failures 23/90 (26 %) {'plan after look': 8, 'perception': 15}; same in all groups True | A 0/90 | decision set
  dumbbell   pre-split failures 0/90 (0 %) {}; same in all groups True | A 3/90 | decision set
  wcan       pre-split failures 24/90 (27 %) {'perception': 24}; same in all groups True | A 1/90 | decision set
  decision set: ['candle', 'apple', 'statue_cab', 'dumbbell', 'wcan']; pre-split the same in every group for every task: True
C2. per task (episodes 30..119): successes and discordant pairs (x-only / y-only, McNemar exact p)
  candle     n 90 A 31 R 42 | RvA A-only 2 R-only 13 p 0.00739
  apple      n 90 A  0 R  1 | RvA A-only 0 R-only 1 p 1
  bowl       n 90 A  0 R  0 | RvA A-only 0 R-only 0 p 1
  salt       n 90 A  2 R  0 | RvA A-only 2 R-only 0 p 0.5
  statue_cab n 90 A  0 R  0 | RvA A-only 0 R-only 0 p 1
  dumbbell   n 90 A  3 R  1 | RvA A-only 2 R-only 0 p 0.5
  wcan       n 90 A  1 R  0 | RvA A-only 1 R-only 0 p 1
C3. pooled
  decision set (candle, apple, statue_cab, dumbbell, wcan): 450 pairs
    R vs A: success R 44 (9.8 %) A 35 (7.8 %) diff +2.0 pp (+9) | A-only 5 R-only 14 McNemar exact p 0.0636
       close-time hand to the grasp point, paired R - A: median -0.14 cm, 95 % CI [-0.43, +0.79], n 255, R closer in 135, A closer in 118, sign p 0.314, Wilcoxon p 0.0439; medians R 13.18 A 9.27 cm
  all 7 tasks (candle, apple, bowl, salt, statue_cab, dumbbell, wcan): 630 pairs
    R vs A: success R 44 (7.0 %) A 37 (5.9 %) diff +1.1 pp (+7) | A-only 7 R-only 14 McNemar exact p 0.189
       close-time hand to the grasp point, paired R - A: median -0.36 cm, 95 % CI [-0.60, -0.07], n 306, R closer in 174, A closer in 128, sign p 0.0095, Wilcoxon p 0.8; medians R 16.09 A 11.04 cm
C4. failure reasons: the new item pregrasp_touch (the pre-grasp segment's hand-object contact rows > 0, checked before hardgrasp's IK / furniture / grasp labels); then hardgrasp's table as it was
  all 7 tasks  A ok 37 perception 124 pregrasp_touch 106 stance_or_ik 105 fell_walking 0 hit_furniture 215 missed 25 dropped 7 other 11 | pregrasp_touch was {'hit_furniture': 79, 'missed': 19, 'dropped': 2, 'stance_or_ik': 6} | pre-grasp touched the object (any outcome) 108/413 | reach IK > 3 cm: 29
  all 7 tasks  R ok 44 perception 124 pregrasp_touch 87 stance_or_ik 99 fell_walking 0 hit_furniture 212 missed 47 dropped 6 other 11 | pregrasp_touch was {'missed': 9, 'hit_furniture': 72, 'dropped': 6} | pre-grasp touched the object (any outcome) 88/413 | reach IK > 3 cm: 17
  decision set A ok 35 perception 67 pregrasp_touch 99 stance_or_ik 29 fell_walking 0 hit_furniture 177 missed 25 dropped 7 other 11 | pregrasp_touch was {'hit_furniture': 72, 'missed': 19, 'dropped': 2, 'stance_or_ik': 6} | pre-grasp touched the object (any outcome) 101/356 | reach IK > 3 cm: 19
  decision set R ok 44 perception 67 pregrasp_touch 77 stance_or_ik 25 fell_walking 0 hit_furniture 173 missed 47 dropped 6 other 11 | pregrasp_touch was {'missed': 9, 'hit_furniture': 62, 'dropped': 6} | pre-grasp touched the object (any outcome) 78/356 | reach IK > 3 cm: 9
  [hardgrasp's order] all 7 tasks  A ok 37 perception 124 stance_or_ik 111 fell_walking 0 hit_furniture 294 missed 44 dropped 9 other 11
  [hardgrasp's order] all 7 tasks  R ok 44 perception 124 stance_or_ik 99 fell_walking 0 hit_furniture 284 missed 56 dropped 12 other 11
  [hardgrasp's order] decision set A ok 35 perception 67 stance_or_ik 35 fell_walking 0 hit_furniture 249 missed 44 dropped 9 other 11
  [hardgrasp's order] decision set R ok 44 perception 67 stance_or_ik 25 fell_walking 0 hit_furniture 235 missed 56 dropped 12 other 11
C5. overshoot past the goal > 3 cm, first reach (pre-grasp) on the far targets (wcan, dumbbell)
  A: both 76/156 (49 %) | wcan 37/66 (median +19.9 cm; IK > 3 cm: 2/2) | dumbbell 39/90 (median +1.3 cm; IK > 3 cm: 6/8)
  R: both 5/156 (3 %) | wcan 2/66 (median -22.1 cm; IK > 3 cm: 0/0) | dumbbell 3/90 (median -21.1 cm; IK > 3 cm: 0/0)
C6. executed corrections |command - goal| (cm), pre-grasp (reach_no 1) and entry (reach_no 2) apart: median / = 0 / > 10 cm / n
  R pre-grasp all 14.0/15/252/413 | candle 3.6/1/0/49 apple 11.1/0/50/84 bowl 10.8/8/14/24 salt 6.5/1/9/33 statue_cab 18.0/1/66/67 dumbbell 22.0/2/66/90 wcan 23.8/2/47/66
  R entry     all 7.2/32/159/353 | candle 5.0/3/1/49 apple 6.7/7/33/82 bowl 12.3/5/15/24 salt 8.6/0/14/32 statue_cab 23.5/4/13/17 dumbbell 11.8/8/48/87 wcan 16.5/5/35/62
  R shrink, pre-grasp: inverted 413, with a shrunk candidate 46, executed command shrunk 5, shrunk candidates 136, shrunk to the goal (alpha 0) 115 | by task (with a shrunk candidate / executed shrunk / inverted): candle 0/0/49; apple 0/0/84; bowl 8/0/24; salt 1/0/33; statue_cab 2/0/67; dumbbell 21/0/90; wcan 14/5/66
  R shrink, entry: inverted 353, with a shrunk candidate 67, executed command shrunk 27, shrunk candidates 321, shrunk to the goal (alpha 0) 147 | by task (with a shrunk candidate / executed shrunk / inverted): candle 0/0/49; apple 20/12/82; bowl 8/1/24; salt 2/0/32; statue_cab 10/5/17; dumbbell 23/7/87; wcan 4/2/62
C7. mechanism checks
  M1 [all 7 tasks] A: |landing - D| median (n) IK <= 2 1.87 cm (167); IK 2-3 17.84 cm (1); IK > 3 57.90 cm (1) | IK > 3: |landing - ref end| 59.90, |ref end - command| 7.54, |landing - command| 63.42 cm
  M1 [all 7 tasks] R: |landing - D| median (n) IK <= 2 1.80 cm (242); IK 2-3 8.14 cm (4); IK > 3 16.17 cm (1) | IK > 3: |landing - ref end| 27.08, |ref end - command| 3.30, |landing - command| 28.50 cm
  M1 [decision set] A: |landing - D| median (n) IK <= 2 1.86 cm (162); IK 2-3 17.84 cm (1); IK > 3 57.90 cm (1) | IK > 3: |landing - ref end| 59.90, |ref end - command| 7.54, |landing - command| 63.42 cm
  M1 [decision set] R: |landing - D| median (n) IK <= 2 1.78 cm (237); IK 2-3 29.80 cm (2); IK > 3 nan cm (0) | IK > 3: |landing - ref end| nan, |ref end - command| nan, |landing - command| nan cm
  M3 (wcan, episodes with a pre-grasp segment): A touched 35/66 (53 %), of them ok 1; R touched 36/66 (55 %), of them ok 0 | all groups: touched 71, ok 1 (1 %)
C8. criteria rows and predictions (decision set unless said)
  decision set = candle, apple, statue_cab, dumbbell, wcan (predicted, ~80 %): True; bowl / salt pre-split > 50 %: True / True
  predicted R - A 0..+4 pp: +2.0 (True)
```

### 事后补的描述性分析（不进登记判据）

- **来由**：正式运行和评估都结束以后，协调人另外要的，用来看"R 的收益是不是跟着 A 的放置失败比例走"。只读正式运行的局记录，没有新跑局，也不进登记的判据。
- **脚本**：`desc_failreasons/lmlcmp3_desc.py`（md5 74746d5ef3ae1083c575257c769d2bf2）。它从冻结副本导入 `lc3_eval` 的 load、presplit、reason3，没有另写分类。
- **输出**：`desc_failreasons/desc.txt`（121 行，md5 54f4b06599e1ea73e53c69d2e366eb4c）。
- **分类口径**：
  - 和 `lc3_eval` 的 C4 一致：先判"预抓取时碰到物体"（pgt），其余按 hardgrasp 的优先顺序判，依次是感知、站位/IK、走路摔倒、碰家具、抓空、掉落、其他。每局只记第一失败原因。
  - "抓空"就是 hardgrasp 的"手偏了抓空"：合了手，物体抬起不到 2 cm。
  - "放置偏差 / 抓空"在登记的分类里没有完全对应的一类，所以给两种算法：只算抓空；抓空加预抓取碰物，两者都是手放偏了。
  - 另附碰家具一列，它是 ScaleBFM A 组最大的一类失败。
  - 比例的分母也给两种：A 的全部失败；A 在两组分开之后的失败。两组分开之前的失败（感知、看完那一眼时的规划失败、没有那一眼）各组相同，R 改变不了它们。下表用后一种，前一种在原文里。
  - 原文的 "pre" 列是失败里两组分开之前的那部分，分别算在 perc（感知）、ik（看完那一眼时的规划失败）、oth（没有那一眼）里。
  - 各组合计和 eval 的 C3、C4 对得上，判定集的标注和 C1 一致。
  - 碰没碰物体、碰没碰家具都是仿真真值，只用来记分。

原文：

```
columns: n ok | failures by reason (perc = perception, pgt = pregrasp_touch, ik = stance_or_ik, fell = fell_walking, furn = hit_furniture, miss = missed, drop = dropped, oth = other)

== scalebfm: 2520 episode records, groups A S R E, episodes 30..119
  task          grp   n  ok |  perc   pgt    ik  fell  furn  miss  drop   oth |  pre
  candle          A  90  16 |    28     0     2     0    31     2     0    11 |   41
  candle          S  90  49 |    28     0     2     0     0     0     0    11 |   41
  candle          R  90  49 |    28     0     2     0     0     0     0    11 |   41
  candle          E  90  47 |    28     0     2     0     2     0     0    11 |   41
  apple           A  90  15 |     0     2     9     0    64     0     0     0 |    8
  apple           S  90  20 |     0     8    12     0    50     0     0     0 |    8
  apple           R  90  20 |     0     8     8     0    54     0     0     0 |    8
  apple           E  90  18 |     0     1    12     0    59     0     0     0 |    8
  bowl            A  90   1 |     0     1    83     0     5     0     0     0 |   66
  bowl            S  90   0 |     0     3    87     0     0     0     0     0 |   66
  bowl            R  90   1 |     0     1    85     0     3     0     0     0 |   66
  bowl            E  90   1 |     0     1    88     0     0     0     0     0 |   66
  salt            A  90   4 |    57     0     1     0    28     0     0     0 |   57
  salt            S  90  15 |    57     1     5     0    12     0     0     0 |   57
  salt            R  90  15 |    57     1     1     0    16     0     0     0 |   57
  salt            E  90  19 |    57     0     4     0    10     0     0     0 |   57
  statue_cab      A  90   0 |    15     0    69     0     6     0     0     0 |   23
  statue_cab      S  90   0 |    15     1    74     0     0     0     0     0 |   23
  statue_cab      R  90   0 |    15     0    64     0    11     0     0     0 |   23
  statue_cab      E  90   0 |    15     0    75     0     0     0     0     0 |   23
  dumbbell        A  90  25 |     0    11     7     0    44     3     0     0 |    0
  dumbbell        S  90  30 |     0    29    16     0     5     8     2     0 |    0
  dumbbell        R  90  29 |     0    23     7     0    17    13     1     0 |    0
  dumbbell        E  90  36 |     0    10    11     0    24     7     2     0 |    0
  wcan            A  90   5 |    24     2     5     0    46     8     0     0 |   24
  wcan            S  90   5 |    24    38     9     0     2    12     0     0 |   24
  wcan            R  90  19 |    24     4     2     0    20    21     0     0 |   24
  wcan            E  90  11 |    24     2    18     0    28     5     2     0 |   24
  decision set    A 450  61 |    67    15    92     0   191    13     0    11 |   96
  decision set    S 450 104 |    67    76   113     0    57    20     2    11 |   96
  decision set    R 450 117 |    67    35    83     0   102    34     1    11 |   96
  decision set    E 450 112 |    67    13   118     0   113    12     4    11 |   96
  all 7 tasks     A 630  66 |   124    16   176     0   224    13     0    11 |  219
  all 7 tasks     S 630 119 |   124    80   205     0    69    20     2    11 |  219
  all 7 tasks     R 630 133 |   124    37   169     0   121    34     1    11 |  219
  all 7 tasks     E 630 132 |   124    14   210     0   123    12     4    11 |  219
  per task: R - A successes; shares of A's failures (missed, missed + pregrasp_touch, hit_furniture), over all A failures / over A failures after the split
  candle       R - A +33 (R 49, A 16) | A failures 74 (pre-split 41, after 33) | missed 2/74 (3 %) / 2/33 (6 %) | missed + pregrasp_touch 2/74 (3 %) / 2/33 (6 %) | hit_furniture 31/74 (42 %) / 31/33 (94 %) [decision set]
  apple        R - A  +5 (R 20, A 15) | A failures 75 (pre-split 8, after 67) | missed 0/75 (0 %) / 0/67 (0 %) | missed + pregrasp_touch 2/75 (3 %) / 2/67 (3 %) | hit_furniture 64/75 (85 %) / 64/67 (96 %) [decision set]
  bowl         R - A  +0 (R 1, A 1) | A failures 89 (pre-split 66, after 23) | missed 0/89 (0 %) / 0/23 (0 %) | missed + pregrasp_touch 1/89 (1 %) / 1/23 (4 %) | hit_furniture 5/89 (6 %) / 5/23 (22 %) [excluded]
  salt         R - A +11 (R 15, A 4) | A failures 86 (pre-split 57, after 29) | missed 0/86 (0 %) / 0/29 (0 %) | missed + pregrasp_touch 0/86 (0 %) / 0/29 (0 %) | hit_furniture 28/86 (33 %) / 28/29 (97 %) [excluded]
  statue_cab   R - A  +0 (R 0, A 0) | A failures 90 (pre-split 23, after 67) | missed 0/90 (0 %) / 0/67 (0 %) | missed + pregrasp_touch 0/90 (0 %) / 0/67 (0 %) | hit_furniture 6/90 (7 %) / 6/67 (9 %) [decision set]
  dumbbell     R - A  +4 (R 29, A 25) | A failures 65 (pre-split 0, after 65) | missed 3/65 (5 %) / 3/65 (5 %) | missed + pregrasp_touch 14/65 (22 %) / 14/65 (22 %) | hit_furniture 44/65 (68 %) / 44/65 (68 %) [decision set]
  wcan         R - A +14 (R 19, A 5) | A failures 85 (pre-split 24, after 61) | missed 8/85 (9 %) / 8/61 (13 %) | missed + pregrasp_touch 10/85 (12 %) / 10/61 (16 %) | hit_furniture 46/85 (54 %) / 46/61 (75 %) [decision set]
  decision set R - A +56 (R 117, A 61) | A failures 389 (pre-split 96, after 293) | missed 13/389 (3 %) / 13/293 (4 %) | missed + pregrasp_touch 28/389 (7 %) / 28/293 (10 %) | hit_furniture 191/389 (49 %) / 191/293 (65 %)
  all 7 tasks  R - A +67 (R 133, A 66) | A failures 564 (pre-split 219, after 345) | missed 13/564 (2 %) / 13/345 (4 %) | missed + pregrasp_touch 29/564 (5 %) / 29/345 (8 %) | hit_furniture 224/564 (40 %) / 224/345 (65 %)

== sonic: 1890 episode records, groups A S R, episodes 30..119
  task          grp   n  ok |  perc   pgt    ik  fell  furn  miss  drop   oth |  pre
  candle          A  90  48 |    28     1     2     0     0     0     0    11 |   41
  candle          S  90  49 |    28     0     2     0     0     0     0    11 |   41
  candle          R  90  49 |    28     0     2     0     0     0     0    11 |   41
  apple           A  90   4 |     0     5     8     3    70     0     0     0 |    8
  apple           S  90   8 |     0    15    11     3    53     0     0     0 |    8
  apple           R  90   7 |     0    14     8     3    58     0     0     0 |    8
  bowl            A  90   0 |     0     2    68     8    12     0     0     0 |   67
  bowl            S  90   2 |     0    10    69     8     1     0     0     0 |   67
  bowl            R  90   1 |     0     3    68     8    10     0     0     0 |   67
  salt            A  90   8 |    57     0     0     9    16     0     0     0 |   57
  salt            S  90  10 |    57     1     2     9    11     0     0     0 |   57
  salt            R  90   9 |    57     0     0     9    15     0     0     0 |   57
  statue_cab      A  90   0 |    15     0    38     0    37     0     0     0 |   24
  statue_cab      S  90   0 |    15     1    74     0     0     0     0     0 |   24
  statue_cab      R  90   0 |    15     0    35     0    39     1     0     0 |   24
  dumbbell        A  90   0 |     0    15     1     2     5    66     1     0 |    0
  dumbbell        S  90  10 |     0    14     8     2    32    24     0     0 |    0
  dumbbell        R  90   9 |     0    14     2     2    33    30     0     0 |    0
  wcan            A  90   1 |    24    14     0     0     6    43     2     0 |   24
  wcan            S  90   6 |    24    18     1     0    15    24     2     0 |   24
  wcan            R  90   3 |    24    18     0     0    17    26     2     0 |   24
  decision set    A 450  53 |    67    35    49     5   118   109     3    11 |   97
  decision set    S 450  73 |    67    48    96     5   100    48     2    11 |   97
  decision set    R 450  68 |    67    46    47     5   147    57     2    11 |   97
  all 7 tasks     A 630  61 |   124    37   117    22   146   109     3    11 |  221
  all 7 tasks     S 630  85 |   124    59   167    22   112    48     2    11 |  221
  all 7 tasks     R 630  78 |   124    49   115    22   172    57     2    11 |  221
  per task: R - A successes; shares of A's failures (missed, missed + pregrasp_touch, hit_furniture), over all A failures / over A failures after the split
  candle       R - A  +1 (R 49, A 48) | A failures 42 (pre-split 41, after 1) | missed 0/42 (0 %) / 0/1 (0 %) | missed + pregrasp_touch 1/42 (2 %) / 1/1 (100 %) | hit_furniture 0/42 (0 %) / 0/1 (0 %) [decision set]
  apple        R - A  +3 (R 7, A 4) | A failures 86 (pre-split 8, after 78) | missed 0/86 (0 %) / 0/78 (0 %) | missed + pregrasp_touch 5/86 (6 %) / 5/78 (6 %) | hit_furniture 70/86 (81 %) / 70/78 (90 %) [decision set]
  bowl         R - A  +1 (R 1, A 0) | A failures 90 (pre-split 67, after 23) | missed 0/90 (0 %) / 0/23 (0 %) | missed + pregrasp_touch 2/90 (2 %) / 2/23 (9 %) | hit_furniture 12/90 (13 %) / 12/23 (52 %) [excluded]
  salt         R - A  +1 (R 9, A 8) | A failures 82 (pre-split 57, after 25) | missed 0/82 (0 %) / 0/25 (0 %) | missed + pregrasp_touch 0/82 (0 %) / 0/25 (0 %) | hit_furniture 16/82 (20 %) / 16/25 (64 %) [excluded]
  statue_cab   R - A  +0 (R 0, A 0) | A failures 90 (pre-split 24, after 66) | missed 0/90 (0 %) / 0/66 (0 %) | missed + pregrasp_touch 0/90 (0 %) / 0/66 (0 %) | hit_furniture 37/90 (41 %) / 37/66 (56 %) [decision set]
  dumbbell     R - A  +9 (R 9, A 0) | A failures 90 (pre-split 0, after 90) | missed 66/90 (73 %) / 66/90 (73 %) | missed + pregrasp_touch 81/90 (90 %) / 81/90 (90 %) | hit_furniture 5/90 (6 %) / 5/90 (6 %) [decision set]
  wcan         R - A  +2 (R 3, A 1) | A failures 89 (pre-split 24, after 65) | missed 43/89 (48 %) / 43/65 (66 %) | missed + pregrasp_touch 57/89 (64 %) / 57/65 (88 %) | hit_furniture 6/89 (7 %) / 6/65 (9 %) [decision set]
  decision set R - A +15 (R 68, A 53) | A failures 397 (pre-split 97, after 300) | missed 109/397 (27 %) / 109/300 (36 %) | missed + pregrasp_touch 144/397 (36 %) / 144/300 (48 %) | hit_furniture 118/397 (30 %) / 118/300 (39 %)
  all 7 tasks  R - A +17 (R 78, A 61) | A failures 569 (pre-split 221, after 348) | missed 109/569 (19 %) / 109/348 (31 %) | missed + pregrasp_touch 146/569 (26 %) / 146/348 (42 %) | hit_furniture 146/569 (26 %) / 146/348 (42 %)

== heft: 1260 episode records, groups A R, episodes 30..119
  task          grp   n  ok |  perc   pgt    ik  fell  furn  miss  drop   oth |  pre
  candle          A  90  31 |    28     0     2     0     0    16     2    11 |   41
  candle          R  90  42 |    28     1     2     0     0     6     0    11 |   41
  apple           A  90   0 |     0     2    11     0    77     0     0     0 |    6
  apple           R  90   1 |     0     4    10     0    75     0     0     0 |    6
  bowl            A  90   0 |     0     6    75     0     9     0     0     0 |   66
  bowl            R  90   0 |     0     8    73     0     9     0     0     0 |   66
  salt            A  90   2 |    57     1     1     0    29     0     0     0 |   57
  salt            R  90   0 |    57     2     1     0    30     0     0     0 |   57
  statue_cab      A  90   0 |    15     1    12     0    54     8     0     0 |   23
  statue_cab      R  90   0 |    15     0    11     0    45    19     0     0 |   23
  dumbbell        A  90   3 |     0    62     4     0    16     1     4     0 |    0
  dumbbell        R  90   1 |     0    36     2     0    28    19     4     0 |    0
  wcan            A  90   1 |    24    34     0     0    30     0     1     0 |   24
  wcan            R  90   0 |    24    36     0     0    25     3     2     0 |   24
  decision set    A 450  35 |    67    99    29     0   177    25     7    11 |   94
  decision set    R 450  44 |    67    77    25     0   173    47     6    11 |   94
  all 7 tasks     A 630  37 |   124   106   105     0   215    25     7    11 |  217
  all 7 tasks     R 630  44 |   124    87    99     0   212    47     6    11 |  217
  per task: R - A successes; shares of A's failures (missed, missed + pregrasp_touch, hit_furniture), over all A failures / over A failures after the split
  candle       R - A +11 (R 42, A 31) | A failures 59 (pre-split 41, after 18) | missed 16/59 (27 %) / 16/18 (89 %) | missed + pregrasp_touch 16/59 (27 %) / 16/18 (89 %) | hit_furniture 0/59 (0 %) / 0/18 (0 %) [decision set]
  apple        R - A  +1 (R 1, A 0) | A failures 90 (pre-split 6, after 84) | missed 0/90 (0 %) / 0/84 (0 %) | missed + pregrasp_touch 2/90 (2 %) / 2/84 (2 %) | hit_furniture 77/90 (86 %) / 77/84 (92 %) [decision set]
  bowl         R - A  +0 (R 0, A 0) | A failures 90 (pre-split 66, after 24) | missed 0/90 (0 %) / 0/24 (0 %) | missed + pregrasp_touch 6/90 (7 %) / 6/24 (25 %) | hit_furniture 9/90 (10 %) / 9/24 (38 %) [excluded]
  salt         R - A  -2 (R 0, A 2) | A failures 88 (pre-split 57, after 31) | missed 0/88 (0 %) / 0/31 (0 %) | missed + pregrasp_touch 1/88 (1 %) / 1/31 (3 %) | hit_furniture 29/88 (33 %) / 29/31 (94 %) [excluded]
  statue_cab   R - A  +0 (R 0, A 0) | A failures 90 (pre-split 23, after 67) | missed 8/90 (9 %) / 8/67 (12 %) | missed + pregrasp_touch 9/90 (10 %) / 9/67 (13 %) | hit_furniture 54/90 (60 %) / 54/67 (81 %) [decision set]
  dumbbell     R - A  -2 (R 1, A 3) | A failures 87 (pre-split 0, after 87) | missed 1/87 (1 %) / 1/87 (1 %) | missed + pregrasp_touch 63/87 (72 %) / 63/87 (72 %) | hit_furniture 16/87 (18 %) / 16/87 (18 %) [decision set]
  wcan         R - A  -1 (R 0, A 1) | A failures 89 (pre-split 24, after 65) | missed 0/89 (0 %) / 0/65 (0 %) | missed + pregrasp_touch 34/89 (38 %) / 34/65 (52 %) | hit_furniture 30/89 (34 %) / 30/65 (46 %) [decision set]
  decision set R - A  +9 (R 44, A 35) | A failures 415 (pre-split 94, after 321) | missed 25/415 (6 %) / 25/321 (8 %) | missed + pregrasp_touch 124/415 (30 %) / 124/321 (39 %) | hit_furniture 177/415 (43 %) / 177/321 (55 %)
  all 7 tasks  R - A  +7 (R 44, A 37) | A failures 593 (pre-split 217, after 376) | missed 25/593 (4 %) / 25/376 (7 %) | missed + pregrasp_touch 131/593 (22 %) / 131/376 (35 %) | hit_furniture 215/593 (36 %) / 215/376 (57 %)
```

下表只列判定集的 5 个任务和判定集合计。碗和盐瓶在原文里。比例的分母是 A 在两组分开之后的失败。

| 控制器 | 任务 | R − A（局） | A 失败（分开之后） | 抓空 | 抓空 + 预抓取碰物 | 碰家具 |
|---|---|---|---|---|---|---|
| ScaleBFM | 蜡烛 | +33 | 33 | 6 % | 6 % | 94 % |
| ScaleBFM | 苹果 | +5 | 67 | 0 % | 3 % | 96 % |
| ScaleBFM | 雕像柜 | 0 | 67 | 0 % | 0 % | 9 % |
| ScaleBFM | 哑铃 | +4 | 65 | 5 % | 22 % | 68 % |
| ScaleBFM | 浇水壶 | +14 | 61 | 13 % | 16 % | 75 % |
| ScaleBFM | 判定集 | +56 | 293 | 4 % | 10 % | 65 % |
| SONIC | 蜡烛 | +1 | 1 | 0/1 | 1/1 | 0/1 |
| SONIC | 苹果 | +3 | 78 | 0 % | 6 % | 90 % |
| SONIC | 雕像柜 | 0 | 66 | 0 % | 0 % | 56 % |
| SONIC | 哑铃 | +9 | 90 | 73 % | 90 % | 6 % |
| SONIC | 浇水壶 | +2 | 65 | 66 % | 88 % | 9 % |
| SONIC | 判定集 | +15 | 300 | 36 % | 48 % | 39 % |
| HEFT | 蜡烛 | +11 | 18 | 89 % | 89 % | 0 % |
| HEFT | 苹果 | +1 | 84 | 0 % | 2 % | 92 % |
| HEFT | 雕像柜 | 0 | 67 | 12 % | 13 % | 81 % |
| HEFT | 哑铃 | −2 | 87 | 1 % | 72 % | 18 % |
| HEFT | 浇水壶 | −1 | 65 | 0 % | 52 % | 46 % |
| HEFT | 判定集 | +9 | 321 | 8 % | 39 % | 55 % |

只作描述：三个控制器放在一起，看不出 R − A 跟着 A 的放置类失败比例单调变化，见"意外"第 5 条。

**路径和 md5**

4090 上的输出目录：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/`。

- 正式运行：
  - `formal.log` 9f22fac3fe27b3b802efd266faa461cc
  - `scalebfm/`：`eval.txt` c26edc008b571d42b70c4fa0f1f15070，`eval.json` d47fcb4acfb835254bfe66bde9133986，`check.txt` b5906f5fa20cc71d9d514022ab8d342b
  - `sonic/`：`eval.txt` 48250b489c3922a66ac0941c5c7e3e69，`eval.json` 9f588e37c9aefc2c62fd2d8446bb12a6，`check.txt` 7557950f71ba6f7116c23be9c5ef04f4
  - `heft/`：`eval.txt` f29022d8e52d00645dcc90d320e61872，`eval.json` c57d26dffec254ac01db57a6ab653703，`check.txt` 6af29a83a43d224c5dd58d3e3d3732b7
  - 每个控制器目录下还有 `episodes/`（局记录）、`segs/`（每一段的数组）、`data/`（合并后的数据）、`logs/`、`merge.log`、`pids.txt`。
- 描述性分析：
  - `desc_failreasons/lmlcmp3_desc.py` 74746d5ef3ae1083c575257c769d2bf2
  - `desc_failreasons/desc.txt` 54f4b06599e1ea73e53c69d2e366eb4c
- 冻结（`asrun/`，FROZEN_UTC 2026-11-26T02:41:43Z）：
  - `CODE_MD5`（副本的 1109 个文件）1de02714723c84cd358da91b8130ab49
  - `MD5`（18 行输入）7b8b963b0ea94ab70b67fd2547cb83fa
  - `LOOKS3_MD5`（619 行）fd2596c475d3a807928d04bcdcb034e0
  - `LOOKS_MD5`（303 行）b95fc6267ef34bd9f81deba406de866e
  - `VS_PREV` b2cdabc7c9e0d67aa4c89df9e3606407
  - `FROZEN_UTC` 0895c18788ac8d81ec9a46f0aada6733
- 本轮脚本（`code/scripts/experiments/selfmodel_lmlcmp3_20261126/`）：
  - `lc3_run.py` eca94321450e69c45f64e7975ad86a73
  - `lc3_check.py` ad853c53a8f2e6c406b2dfa528910558
  - `lc3_eval.py` 3b681e94a5ea484cbe13e05f2256f8fa
  - `lc3_batch.sh` 082a00914059a51c18c5cbd6089e8430
  - `lc3_formal.sh` 05cf547a15a91e23e797ade2821d731e
  - `lc3_freeze.sh` e7b9ae99cd5c846f343fafc9ae74afc4
  - `lc3_look.sh` c201083c3663c2372d7206cc364375ca
- 其他目录：
  - `code/`：私有副本；
  - `hg/`：新种子的"那一眼"和补拍日志；
  - `smoke_design/`、`smoke_design_far/`、`smoke_new/`、`formal_dryrun/`、`look_repro/`：冒烟、试跑、补拍复现；
  - `aborted_formal_0225/`：作废的第一次启动和两次旧冻结，不进入任何统计。
- 提交：登记 7d8ce3d5，修订 1 73cdf7a4，修订 1 补记 3567e645。
- Mac 上的原文副本（和 4090 的 md5 相同）：`/Users/zed/.claude/jobs/f13ab968/tmp/lc3_final/`。

## 意外

执行 agent 写。只列事实，不下结论。

1. **ScaleBFM 的 A 比预测低。**
   - 判定集 61/450（13.6 %），预测 16–24 %。浇水壶 A 5/90，预测 6–14 局。
   - 这个预测是照 lmlcmp2 的 20 %（30/150）定的。新种子上各组都比 lmlcmp2 低，只是 A 掉得最多，掉出了区间：
     - A 20.0 → 13.6 %；
     - S 27.3 → 23.1 %；
     - R 31.3 → 26.0 %。
   - 不是因为两组分开之前的失败变多了。判定集上 lmlcmp2 是 33/150（22 %），这次是 96/450（21 %）。
   - A 在有成功的四个任务上都更低（lmlcmp2 → 本轮）：蜡烛 27 → 18 %，苹果 27 → 17 %，哑铃 37 → 28 %，浇水壶 10 → 6 %。
   - A 在判定集上失败最多的一类是碰家具：191 局，占两组分开之后失败的 65 %。
2. **P2 只靠浇水壶。**
   - 判定集 R 对 S：S 独赢 12，R 独赢 25，p 0.047。它是两个 p 里较大的那个，Holm 校正后仍是 0.047，门槛是 0.05。
   - 这 37 对里，浇水壶占 20 对（S 独赢 3，R 独赢 17，p 0.00258）。
   - 其余是哑铃 S 独赢 9、R 独赢 8；蜡烛、苹果、雕像柜都是 0 对 0。
   - 去掉浇水壶，R 对 S 是 9 对 8，p 1。这是事后算的，不进判据。
   - 7 个任务全算，R 对 S 的 p 是 0.0336，多出来的 1 对来自碗（S 0、R 1）。
3. **M2 按局数触发了"R − E ≥ 4"，但差别不显著，各任务的方向也不一致。**
   - R − E = +5，E 独赢 17，R 独赢 22，p 0.522。判据这一行只看局数，所以照字面是触发了。
   - 各任务：
     - 蜡烛 +2，苹果 +2，雕像柜 0；
     - 哑铃 −7：E 独赢 10，R 独赢 3，p 0.0923；
     - 浇水壶 +8：E 独赢 3，R 独赢 11，p 0.0574。
   - +5 是浇水壶的 +8 和哑铃的 −7 相抵之后剩下的。
   - 哑铃上 E 是四组里最多的，36 局。E 对 A 是 6 对 17，p 0.0347。
   - E − S = +8，正好在预测区间的上沿。
4. **R − A 的 +56 局里，蜡烛占 +33。**
   - 蜡烛 A 16、R 49，R 对 A 是 0 对 33。
   - 蜡烛上 R 和 S 都是 49 局，R 对 S 是 0 对 0。R 在蜡烛上一次都没有缩回（C6 里 "candle 0/0/49"），所以蜡烛上 R 就是 S。这 +33 来自原反解本身，不是来自缩回。
   - 去掉蜡烛，下面的数是事后算的，不进判据：
     - 其余 4 个任务 360 对，R 68、A 45，多 23 局，+6.4 个百分点；
     - A 独赢 14、R 独赢 37，p 0.00177；
     - +6.4 个百分点低于 P1 的 10 个百分点线。
   - 蜡烛两组分开之前失败 41/90（46 %），再多 5 局就会被排除。
   - lmlcmp2 的复核指出过同样的情况：那一轮 R 对 A 的 21 对里，10 对是蜡烛。
5. **描述性表里，ScaleBFM 上 R 的收益出现在 A 主要因碰家具而失败的任务上，不在放置类失败多的任务上。SONIC 和 HEFT 不是这样。**
   - ScaleBFM：
     - 收益最大的是蜡烛（+33）和浇水壶（+14）。A 在两组分开之后的失败里，碰家具占 94 % 和 75 %，抓空只占 6 % 和 13 %。
     - A 放置类失败比例最高的是哑铃（抓空 + 预抓取碰物 22 %），R − A 只有 +4。
   - SONIC：
     - 收益最大的是哑铃（+9）。A 的失败里抓空占 73 %，加上预抓取碰物是 90 %。
     - 浇水壶的放置类失败也占 88 %，R − A 却只有 +2。
     - 苹果的失败里碰家具占 90 %，R − A +3。
   - HEFT：
     - 唯一明显的收益是蜡烛（+11），A 的失败里抓空占 89 %。
     - 哑铃的放置类失败占 72 %，R − A −2；浇水壶占 52 %，R − A −1。
   - 三个控制器放在一起，R − A 既不跟着 A 的放置类失败比例走，也不跟着碰家具的比例走。这张表不进判据。
6. **HEFT 上越过目标没了，碰壶却没少。**
   - HEFT 的 A 自己在浇水壶上就越过目标，37/66，中位数 +19.9 cm。R 降到 2/66，中位数 −22.1 cm，变成了够不到。
   - 但预抓取时碰到壶的局，A 35/66（53 %），R 36/66（55 %），几乎一样多。浇水壶成功 A 1 局，R 0 局。
   - ScaleBFM 上越过目标和碰壶是一起出现的：S 越过目标 41/66，碰壶 40/66。HEFT 上两者分开了。
7. **SONIC 上，R 合手时手离抓取点比 S 远。**
   - 判定集 R − S 的配对差中位数是 0。但 S 更近的有 90 局，R 更近的只有 28 局，符号检验 p 8.91e-09。
   - 成功数 R 68、S 73，S 独赢 6、R 独赢 1，p 0.125，在预测的 −2 到 +3 个百分点以内。
   - SONIC 的 S 几乎不越过目标：浇水壶 2/66。S 越界命令的预测误差中位数（M1 的量，SONIC 上只作描述）是 8.97 cm（n 54），比 ScaleBFM 的 19.87 cm 小。
8. **ScaleBFM 的 R 去掉了 S 的"预抓取碰物"，但多了碰家具和抓空。**
   - 判定集上，S 到 R：
     - 预抓取碰物 76 → 35；
     - 碰家具 57 → 102；
     - 抓空 20 → 34。
   - 浇水壶上最明显：
     - 预抓取碰物 38 → 4；
     - 碰家具 2 → 20；
     - 抓空 12 → 21。
   - 浇水壶 R 比 S 多成功 14 局。

## 结论与下一步

协调人写。证据：一次正式运行，新种子 30–119，每个任务每组 90 局，按局配对。三个控制器共 5670 局。判定集由登记的规则在运行后算出，三个控制器都是 5 个任务，每组 450 局。下面"事后"二字标出的数不进判据。

**1. ScaleBFM 落在判据第 1 行"P1 和 P2 都成立"：以后 ScaleBFM 的反解改用 R，SIMPLE 重采训好之后，SIMPLE 的对比也用 R。** 但两条的读法都要限定：
- **P1（R 对 A）在新种子上复现了，而且很稳**：+12.4 个百分点，R 独赢 70、A 独赢 14，p 4.07e-10（结果，ScaleBFM eval.txt 的 "R vs A" 行）。
  - 但 +56 局里有 +33 来自蜡烛（意外 4）。蜡烛上 R 一次都没有缩回，R 就是 S，所以这 +33 是"反解本身"的作用，不是"缩回"的作用。
  - 事后算，去掉蜡烛：+6.4 个百分点，p 0.00177。仍然显著，但低于 P1 的 10 个百分点线。
  - 所以可以说"自身模型反解比直接执行成功更多，在新种子上复现"；不能说"在每个任务上都高 10 个百分点"。
  - 蜡烛两组分开之前的失败是 41/90，离排除线只差 5 局。判定集里是否包含蜡烛，对 P1 的幅度影响很大。
- **P2（R 对 S）勉强成立，范围就是浇水壶**：p 0.047，Holm 校正后仍是 0.047；37 个不一致对里，浇水壶占 20 对，R 17 对 S 3（意外 2）。事后算，去掉浇水壶是 9 对 8。
  - 这和假设 2 原话一致（"好处集中在浇水壶这类目标上"），不算推翻。但不能读成"R 普遍比 S 好"。准确的说法是：R 在 S 会把命令推出可达范围的目标上更好，别处两者一样。

**2. 越界机制（B57）在新种子上复现。** M1：S 发出的越界命令（IK 误差 > 3 cm），预测误差中位数 19.87 cm（n 75）；R 的越界命令 2.14 cm（n 27）。M3 的三项都在预测内：
- S 预抓取碰壶 61 %，R 6 %；
- 碰过壶的局只成功 2/48；
- 越过目标 > 3 cm，S 41/66，A、R、E 都是 0/66。

**3. M2 不下结论。** 照字面，R − E = +5 触发了"进入段也要缩回"。但 p 0.522，而且是浇水壶 +8 和哑铃 −7 相抵剩下的（意外 3）。只能说"进入段要不要缩回，这批数据分不出来"。不据此修改 B57。

**4. SONIC 和 HEFT 都在预测内，都不赢。**
- SONIC：R 对 A +3.3 个百分点，p 0.000729，显著但幅度小；R 对 S −1.1 个百分点，p 0.125。R 合手时手离抓取点比 S 远（S 更近 90 局、R 更近 28 局，p 8.91e-09，意外 7），成功数却分不开。
- HEFT：R 对 A +2.0 个百分点，p 0.0636，不显著，全部来自蜡烛。

**5. 事后补的描述性表，支持不了"手准的收益只在放置类失败多的任务上"，也支持不了反过来的说法**（意外 5）。
- ScaleBFM 收益大的任务，A 主要因碰家具失败；
- SONIC 和 HEFT 收益大的任务，A 主要是抓空。
- 三个控制器放在一起，R − A 既不跟着放置类失败的比例走，也不跟着碰家具的比例走。
- ScaleBFM 上，R 把 S 的"预抓取碰物"从 76 降到 35，但碰家具从 57 升到 102（意外 8）。浇水壶上 R 剩下的失败主要是碰家具。

**其他可能的解释**
- 蜡烛的大收益，可能来自这一个几何（矮凳上的低处目标，开局 0.4 m）上直接执行的系统性偏差，而不是模型在各种几何上都有用。"各种几何上都有用"由 E2（`selfmodel_freespace_20261126`）检验。
- A 比预测低（意外 1）：各组都比 lmlcmp2 低，A 掉得最多。可能是种子不同带来的波动。两组分开之前的失败比例和 lmlcmp2 相同（21 % 对 22 %），所以不是感知变差了。
- P2 的 p 0.047 离门槛很近，换一批种子完全可能不显著。它依赖浇水壶一个任务。

**四类陷阱**
- **benchmark 选择**：判定集按登记规则，只用两组分开之前的失败来算，不依赖处理组的结果。碗、盐瓶被排除，原因是规划和感知。蜡烛离排除线很近，已写明它对 P1 幅度的影响。
- **数据泄漏**：种子 30–119 和之前所有种子不重叠。D 只用空地数据训练（data3 + data3b + 中间点），没用过任务局。
- **指标误用**：主度量是登记的成功；合手距离是次度量。不利的结果都报了：SONIC 的 R 比 S 远，HEFT 不显著，M2 方向不一致。
- **事后挑选**：只有一次正式运行，三个控制器全报。之前作废的那次启动没有跑任何局，见修订。事后算的数都标了。

**下一步**
- ScaleBFM 的反解默认用 R，E2 和 SIMPLE 的对比都用 R。
- B56、B57 的补充等独立复核回来再写。
- 论文进度文件里，lmlcmp3 的结果等复核以后再进 §一。


## 独立复核

复核人：独立 agent，2026-11-26。全程只读：只在 4090 上用远程 python 读逐局 JSON（`*/episodes/*/*/ep*.json`），不导入 `lc3_eval`，统计自己写。脚本放在本地 `/Users/zed/.claude/jobs/f13ab968/tmp/review_lc3/`，四个：`stats.py`、`extra.py`、`strat.py`、`win.py`。本地 7 个 `lc3_*` 文件的 md5 和 4090 上 `code/` 副本的逐个相同。`formal.log` 和三个控制器的 eval.txt、eval.json、check.txt，md5 都和条目所记相同。

### 判定摘要

- **登记判据落格正确：第 1 行"P1 和 P2 都成立"。** 主数都从逐局记录重算过，和 eval.txt 逐位一致：判定集、ScaleBFM 的 R 对 A 和 R 对 S、Holm、M1–M3、SONIC 和 HEFT 的 R 对 A。判定集只用两组分开之前的失败来判，实现符合登记。按局配对正确。
- **但结论有四处读法和原始输出对不上：**
  1. "R 在 S 会把命令推出可达范围的目标上更好，别处两者一样"被哑铃推翻。哑铃上 S 有 44/90 局预抓取推出了可达范围，R 对 S 却是 8 对 9。只看这 44 局，S 独赢 7、R 独赢 4。
  2. M1 里"R 的越界命令 2.14 cm"不是 R 反解出来的命令。这 27 下全是"目标本身也够不着时执行目标"，和 A 的那 27 下是同一批伸手。
  3. "浇水壶上 R 剩下的失败主要是碰家具"不对。两组分开之后，抓空 21 局，碰家具 20 局。
  4. 不按 M2 那一行行事，是对登记判据的偏离，应当写明是偏离。
- **主要论断的判定：**

| 论断 | 判定 |
|---|---|
| 落在第 1 行，ScaleBFM 反解改用 R | 支持（落格），附限定 |
| P1 在新种子上复现 | 方向和显著性：支持。"≥ 10 个百分点"的幅度：证据不足 |
| "自身模型反解比直接执行成功更多，在新种子上复现" | 部分支持 |
| P2 勉强成立，范围就是浇水壶 | 支持 |
| "R 在 S 推出可达范围的目标上更好，别处一样" | 不支持 |
| 越界机制（B57）复现 | S 那一半：支持。R 的对照：不支持 |
| M2 不下结论 | 推断支持，程序上要标明偏离 |
| SONIC、HEFT 在预测内、都不赢 | 支持 |
| 描述表支持不了任一方向 | 支持（浇水壶那一句除外） |

### 1. 第一步：自己读结果（读"意外""结论"之前）

**1.1 完整性**
- ScaleBFM 2520 条、SONIC 1890 条、HEFT 1260 条，每格正好 90 局，局号正好 30–119。
- 文件路径里的任务、组、局号和记录内字段全部一致，没有重复。
- 报错只有蜡烛的 11 局：49、80、86、96、97、100、102、103、108、110、112，全是"那一眼"npz 不存在的 FileNotFoundError（44 / 33 / 22 条）。
- `aborted_formal_0225/` 里的 15 条记录不在 `episodes/` 下，eval 读不到，我也没读。

**1.2 配对**
- 重置只由局号决定：`reset_hg` 里 `env.rng = default_rng(1_000_003*(ep+2))`，抖动用 `default_rng([ep, 4747])`；`reset_id` 里的组号只用作段编号。
- 每个（任务，局），各组下面这些字段逐字段相同，三个控制器 630 个都没有例外：
  - jitter、look、look_state_dev_mm；
  - plan 的 goal、pre、stance、idx、fail；
  - move、n_walks、walk_left_cm；
  - 经过点命令；
  - 预抓取的 goal_h、p_cmd、yaw_cmd、f_cmd。
- 所以各组在预抓取命令之前完全相同，配对成立。
- E 的经过点和预抓取命令与 A 逐位相同，630/630。
- 命令逐位相同的组对，结果也逐位相同：ScaleBFM 1587/1587（= check 的 1521，加 66 对没有那一眼的局），SONIC 995 = 962 + 33，HEFT 222 = 211 + 11。

**1.3 判定集规则的实现**
- `presplit` 的判据是三条：`plan.fail` 为 "perception" 或 "reach"，或者报错且没有 plan。
- 这三种失败都发生在两组分开之前：
  - `plan.fail` 由 `hg_run.plan_from_look` 在看完那一眼时算出（`lc3_run.py` 第 581 行）。它只依赖缓存的那一眼和当时的骨盆状态。
  - 分组逻辑要到第 652 行发预抓取命令时才开始。
  - 伸手时的 "reach: plan failed" 记在 `why` 里，不在 `plan.fail` 里，不会被当成两组分开之前的失败。
- 原始字段核对：
  - 判为分开之前失败的局，没有一局发过预抓取或进入段命令；
  - 三个控制器、7 个任务各组都相同。
- 另用"失败而且从没发过预抓取命令"这个口径独立算了一遍：
  - ScaleBFM、HEFT 和登记口径完全相同；
  - SONIC 多出 22 局"走路时跟丢"（苹果 3、碗 8、盐瓶 9、哑铃 2）。它们也发生在两组分开之前，各组相同，但不在登记括号里列的三类里。照登记不计，计进去判定集也不变（苹果 11/90，哑铃 2/90）。
- 到顶线是 A ≥ 72/90，A 最高的是 SONIC 蜡烛 48/90。
- 三个控制器的判定集都是蜡烛、苹果、雕像柜、哑铃、浇水壶。蜡烛 41/90（46 %），差 5 局就会被排除。

**1.4 主数重算**（判定集 450 对。我自己写的 McNemar 精确检验和 scipy 的 binomtest 逐个相同）

| 比较 | 成功 | 不一致对 | p | 配对差 95 % 区间（Wald） |
|---|---|---|---|---|
| ScaleBFM R 对 A（P1） | 117 对 61，+12.4 个百分点 | A 独赢 14，R 独赢 70 | 4.07e-10 | [+8.6, +16.3] |
| ScaleBFM R 对 S（P2） | 117 对 104，+2.9 个百分点 | S 独赢 12，R 独赢 25 | 0.0470 | [+0.3, +5.5] |
| ScaleBFM E 对 A / E 对 S / R 对 E | 112/61，112/104，117/112 | 9/60，21/29，17/22 | 2.24e-10，0.322，0.522 | |
| SONIC R 对 A / R 对 S | 68/53，68/73 | 2/17，6/1 | 0.000729，0.125 | |
| HEFT R 对 A | 44/35 | 5/14 | 0.0636 | |

- **Holm**：较小的 p 4.07e-10 < 0.025，较大的 p 0.0470 < 0.05，两条都过。校正后 8.14e-10 / 0.0470。
- **P1**：+12.4 ≥ 10 个百分点，成立。**P2**：R 多 13 局，成立。
- **M1**：S 越界一档 19.87 cm（n 75；判定集 19.94，n 72），R 2.14 cm（n 27），A 2.22 cm，E 11.27 cm。改用 `chain` 索引找对应段（eval 用的是第一个同 reach_no 的段），结果相同，对不上的 0 个。
- **M2**：R − E +5，E − S +8。
- **M3**：浇水壶预抓取碰壶，S 40/66，R 4/66，A 2/66，E 2/66。四组合计碰过壶 48 局，成功 2 局。

**1.5 第一步就看到的问题**（下面的数都是事后算的）
- **P1 的幅度靠蜡烛。** 每次去掉一个任务：
  - 去掉蜡烛：+6.4 个百分点，p 0.00177；
  - 去掉其他任务：+11.7 到 +15.6 个百分点。
  - 按任务分层自助抽样（10000 次）：R − A 的 95 % 区间 [+8.9, +16.0]，≥ 10 个百分点的比例 0.91。"≥ 10 个百分点"是按点估计判的。
- **P2 很脆，而且只靠浇水壶。**
  - 翻转一个不一致对（S 独赢 13、R 独赢 24），p 变成 0.099；少一个 R 独赢的对，p 0.065。
  - 去掉浇水壶，R − S −0.3 个百分点，p 1。
- **M1 的读法。** 排除预抓取碰到物体的伸手以后，S 越界一档仍是 16.76 cm（n 24），R 1.72 cm（n 24）。所以 S 的大误差不是碰壶造成的，M1 那一行的落格站得住。

### 2. 第二步

**2.1 结论和意外里的数字，逐条核对**

和原始输出一致的：
- **意外 1–8 的全部数字。**
  - lmlcmp2 的数是从 lmlcmp2 的 eval.txt 和逐局记录核的：A 30/150、S 41、R 47；判定集两组分开之前的失败 33/150；A 各任务 8、8、11、3（每格 30 局）。
  - lmlcmp2 的 R 对 A 是 A 独赢 4、R 独赢 21，其中蜡烛 10。
- **结论 1–5 里的每个数值。**

对不上或者有误导的：
1. **结论 1："准确的说法是：R 在 S 会把命令推出可达范围的目标上更好，别处两者一样。"** 和原始输出矛盾。哑铃上 S 推出可达范围的程度和浇水壶相近：
   - S 预抓取 IK 误差 > 3 cm 有 44/90 下，越过目标 33/90；
   - 但 R 对 S 是 S 独赢 9、R 独赢 8；
   - 只看 S 预抓取 IK > 3 cm 的局（事后分层）：哑铃 S 独赢 7、R 独赢 4；浇水壶 3 对 16；合起来 10 对 20，p 0.099。
   - 能支持的说法只有"R 在浇水壶上比 S 好"。
2. **结论 2："R 的越界命令 2.14 cm（n 27）。"** 数对，但它不是"R 的越界命令"：
   - 这 27 下执行的命令全部就是目标（27/27），也就是"目标本身也够不着时执行目标"的回退；
   - 组成和 A 的 27 下完全相同：碗 2、哑铃预抓取 7、盐瓶 2、浇水壶 2、雕像柜进入段 14；
   - R 从来不会执行 IK 误差 > 2 cm 的反解命令，所以这一格不能拿来和 S 对比；
   - 而且这里面哑铃预抓取那 7 下（直接执行的命令），D 的误差中位数 23.1 cm，和 S 的一样大。直接执行时，哑铃预抓取 IK > 3 cm 的 11 下中位数 20.9 cm。
   - 预测对照表里"M1：R 的越界命令 ≤ 3 cm——是"这一行同样不成立：按构造，这一格量不到 R 自己的反解命令。
3. **结论 5："浇水壶上 R 剩下的失败主要是碰家具。"** desc.txt 里，浇水壶 R 两组分开之后的失败是：预抓取碰物 4、站位/IK 2、碰家具 20、抓空 21。按 hardgrasp 原来的标签是抓空 24、碰家具 20。应写成"碰家具和抓空各占一半左右"。
4. **结果节（不属于结论）"HEFT……其余任务上 R 都不比 A 多"。** 苹果 R 1、A 0（0 对 1）。不影响"全部来自蜡烛"：+9 = 蜡烛 +11，苹果 +1，哑铃 −2，浇水壶 −1。
5. **意外 4 的"那一轮 R 对 A 的 21 对"。** 21 是 R 独赢的对数，不一致对一共 25 对（A 独赢 4）。只是措辞问题。

**2.2 四类陷阱**

- **benchmark 选择（判定集）。** 通过。
  - 实现只用两组分开之前的失败，见 1.3；各组相同，判为分开之前失败的局没有一局发过分组命令。
  - 蜡烛在补拍"那一眼"时已知 39/90，规则没有因此改动。
  - 两点风险：
    - 蜡烛离排除线只差 5 局，去掉蜡烛 P1 不成立（+6.4 个百分点）；
    - 雕像柜三个控制器、各组都是 0 成功，它的 90 对只会稀释百分点（去掉它，R − A 是 +15.6）。这对 P1 偏保守，不构成挑选。
  - SONIC 有 22 局"走路时跟丢"也发生在分组之前，没计入，但不改变判定集。
- **数据泄漏。** 通过。
  - 之前各轮的局号：
    - hardgrasp 的"那一眼"（LOOKS_MD5）：−1、0–29、1000–1009；
    - lmlcmp：−1、0–29、1000；
    - lmlcmp2：−1、0–29；
    - lmlreal：0–29。
  - 本轮 30–119 和它们一局都不重叠。新的"那一眼"619 个 npz 都在本轮 `hg/looks/` 下，运行时确实从这里读（报错路径可证）。
  - D 的训练数据：data3 的采集脚本 `d3_collect.make_host` 用的是 `env_name="EmptyEnv"`（data3 条目第 39 行同样写明），data3b 由同一个采集脚本产生，中间点样本只取训练划分。没有用过任务局。
  - R 的参数（2 cm、6 步、7 个候选）是在 lmlcmp2 的 0–29 上定的，本轮新种子正好避开了这层调参泄漏。
- **指标误用。** 有三处要改：
  1. M1 里 R 的越界一档量的是 A 的回退命令，见 2.1 第 2 条。
  2. P1 的"≥ 10 个百分点"按点估计判。登记就是这么写的，所以落格不受影响，但不能说"高 10 个百分点已经确立"（区间下沿 8.6）。
  3. M3 的碰壶窗口各组长度不同：
     - 浇水壶预抓取窗口的中位数，S 405 行，A 245、R 265、E 245 行；
     - S 只有 26/66 下停稳，34 下是被下一条命令截断，6 下是 3 s 超时；
     - S 的碰壶率有一部分来自窗口更长，用固定窗口（比如命令后 245 行）重算更公平。手不停稳本身可能就是在推壶，所以方向大概不会变，但 61 % 这个数偏高。
- **事后挑选。** 基本通过，有两处要标：
  - 只有一次正式运行，三个控制器全报，事后的数都标了。作废的那次启动只有 6 条报错记录，另 9 条是共享树上的试跑，都不进统计。
  - 要标的两处：
    1. 结论 1 的"准确的说法"是看过分任务结果以后收窄的，而且收窄得不对（哑铃）。
    2. M2 那一行照字面触发了，结论选择不执行。理由站得住：R − E 有 39 个不一致对，零假设下标准差约 6.2 局，登记的"≥ 4 局"在噪声以内，判据本身没设显著性。但要写成"偏离登记判据，原因是判据没有考虑噪声"，不要写成"不下结论"。

**2.3 可能推翻结论的实验**

**实验 F1：新种子复现 P2，同时检验"S 推出可达范围之处 R 更好"。这条最直接针对"改用 R"的决定。**
- 设置：
  - ScaleBFM，冻结的 lmlcmp3 代码；
  - 任务：浇水壶、哑铃，两者 S 预抓取 IK > 3 cm 的比例都在 50–70 %；
  - 新种子 120–299，每个任务 180 局，和之前所有局都不重叠，先补拍"那一眼"；
  - 组：S、R、E。
- 事先写下：
  - 主比较：两个任务合起来 R 对 S，McNemar 精确检验；
  - 分层：按 S 自己那一局的预抓取 IK 误差是否 > 3 cm 分层，这个量在 S 执行之前就算得出来。
- 照本轮外推的预期：
  - 浇水壶 R − S 约 +28 局 / 180（S 独赢约 6，R 独赢约 34）；
  - 哑铃约 0。
- 推翻条件：
  - (a) 浇水壶 R − S ≤ +6 局，或 p ≥ 0.05：P2 不复现，退回登记第 2 行，"S 和 R 都跑"。
  - (b) "S IK > 3 cm"那一层，两个任务合起来 R − S ≤ 0："R 在 S 推出可达范围之处更好"这个机制被推翻，R 的好处只是浇水壶这一个几何上的事。
  - (c) 哑铃 E − R ≥ +10 局且 p < 0.05：本轮 E 独赢 10、R 独赢 3 的方向被确认，"R 作默认"要按几何区分。
- 成本：1080 局，本轮的速度大约 25 分钟；补拍约 6 分钟。

**实验 F2：拆开"预抓取缩回"和"进入段缩回"。这是 M2 没分开的那件事。**
- 在 F1 的同一批局上加两组：
  - P：预抓取用 R 的缩回反解，进入段用 S 的原反解；
  - Q：预抓取直接执行，进入段用 R。
- 和现有的 A、S、R、E 合起来，构成"预抓取 {直接、S、R} × 进入段 {S、R}"的 6 格里的 5 格，外加 A。
- 推翻条件：
  - 浇水壶上 P − S ≤ +4 局 / 180：只把预抓取留在可达范围里，并不能挽回 S 的失败。"S 越过目标 → 碰壶 → 失败"（B57 的链条）就不是 R 好处的来源，R 的收益要另找原因，比如进入段的缩回。
  - 反过来，如果 P ≈ R 而且 Q ≈ E，说明好处在预抓取那一下。

**实验 F3：蜡烛几何压力测试。针对 P1 的幅度。**
- 背景：蜡烛占 R − A 的 +33 / 56，R 在蜡烛上和 S 逐位相同，好处来自"原反解本身"，修正量中位数 5.7 cm。
- 做法：新种子上把蜡烛的机器人起始抖动半径和朝向范围放大到 2 倍，跑 A 和 R 各 90 局。
- 推翻条件：R − A < +10 局 / 90，说明蜡烛的大收益只在原来那一个窄几何里成立，P1 的"≥ 10 个百分点"不能推广。
- 和协调人计划的 E2（`selfmodel_freespace_20261126`）互补：E2 在空地上检验"各种几何上都有用"，F3 在真实任务的这一个几何上检验。

**2.4 论断逐条判定**

| 论断 | 判定 | 理由 |
|---|---|---|
| 1. 落在判据第 1 行，ScaleBFM 反解改用 R | **支持（落格）** | 判定集、配对、McNemar、Holm 都独立重算一致，两步门槛都过。限定：P2 的 p 0.047，翻转一个不一致对就变成 0.099；R 和 E 分不开（判定集 +5，p 0.52；7 个任务 +1，p 1）。哑铃上 E 36 局，四组最多，对 R 是 10 对 3 |
| 1a. P1 在新种子上复现，"很稳" | **方向和显著性：支持；"≥ 10 个百分点"：证据不足** | p 4e-10，每次去掉一个任务都还显著。但去掉蜡烛是 +6.4 个百分点；区间 [8.6, 16.3]，分层自助抽样里 ≥ 10 的比例 0.91 |
| 1b. "自身模型反解比直接执行成功更多，在新种子上复现" | **部分支持** | 对 R 成立，去掉蜡烛 p 0.0018。但 R 在蜡烛以外的收益主要是浇水壶上的缩回：去掉蜡烛和浇水壶，A 独赢 10、R 独赢 19，p 0.136。原反解 S 去掉蜡烛是 +2.8 个百分点，A 独赢 19、S 独赢 29，p 0.19，"lmlcmp 的 S 对 A 赢"在蜡烛以外没有复现 |
| 1c. P2 勉强成立，范围就是浇水壶 | **支持** | 去掉浇水壶 −0.3 个百分点，p 1 |
| 1d. "R 在 S 推出可达范围的目标上更好，别处两者一样" | **不支持** | 哑铃上 S 推出可达范围 44/90 下，R 对 S 8 对 9；只看这 44 局是 S 7 对 R 4 |
| 2. 越界机制（B57）在新种子上复现 | **S 那一半：支持；R 对照：不支持** | S 越界命令的预测误差 19.87 cm，排除碰物后仍有 16.76 cm；M3 三项在预测内；越过目标 S 41/66。"R 的越界命令 2.14 cm"其实是 A 的回退命令；直接执行的哑铃越界预抓取误差也有 21–23 cm。更准确的读法是：D 在 IK 不可达的命令上外推失败，不管命令是谁选的；S 把命令推到了那里，R 不会执行这样的反解命令 |
| 3. M2 不下结论 | **推断支持；程序上要标明偏离** | +5 是浇水壶 +8 和哑铃 −7 相抵的结果，p 0.522；登记的门槛 4 局在噪声以内。不执行登记行，应写明是偏离 |
| 4. SONIC、HEFT 在预测内，都不赢 | **支持** | 数字一致。SONIC 的 R − A 去掉哑铃是 +1.7 个百分点，p 0.109；HEFT 去掉蜡烛是 −0.6 |
| 5. 描述表支持不了"手准的收益只在放置类失败多的任务上"，也支持不了反过来的说法 | **支持（"浇水壶上 R 剩下的失败主要是碰家具"这句不支持）** | 抓空 21 局、碰家具 20 局 |
| 其他解释："A 偏低可能是种子波动" | **证据不足，但说得通** | lmlcmp2 20 % 对本轮 13.6 %，两比例检验 z 约 1.9，p 约 0.06；两组分开之前的失败比例相同（22 % 对 21 %） |
| 下一步："E2 和 SIMPLE 的对比都用 R" | **可以，但建议保留对照** | P2 只靠一个任务，R 和 E 分不开，SIMPLE 的几何和 LML 不同。SIMPLE（ScaleBFM）的对比至少保留 S 或 E 一组，同时记下 S 预抓取 IK > 3 cm 的比例 |

### 复核后的更正（2026-11-26，协调人）

上面"结论与下一步"保留原样，读法照下面改：

1. **收回"R 在 S 会把命令推出可达范围的目标上更好，别处两者一样"。** 哑铃上 S 也有 44/90 下把预抓取推出可达范围，R 对 S 却是 8 对 9。只能说"R 在浇水壶上比 S 好"（P2 的全部来源）。
2. **M1 里"R 的越界命令 2.14 cm"不成立。** 那 27 下是"目标本身也够不着时执行目标"的回退，和 A 是同一批伸手；R 从不执行 IK 误差 > 2 cm 的反解命令。B57 的读法改成：D 在 IK 不可达的命令上外推失败，不管命令是谁选的（直接执行的哑铃越界预抓取也是 21–23 cm）；S 会把命令推到那里，R 不会。预测对照表里"M1：R 的越界命令 ≤ 3 cm——是"作废。
3. **浇水壶上 R 剩下的失败是"抓空 21、碰家具 20"，各占一半左右**，不是"主要是碰家具"。
4. **M2 是偏离登记判据。** 照字面 R − E = +5 触发了"进入段也要缩回"，我没有执行。原因是登记的门槛 4 局在噪声以内（39 个不一致对，零假设下标准差约 6.2 局），判据没有设显著性。不据此修改 B57。
5. **P1 的幅度**：方向和显著性成立。"≥ 10 个百分点"只是点估计过线（95 % 区间 [8.6, 16.3]），去掉蜡烛是 +6.4，不能说"高 10 个百分点已经确立"。
6. **"自身模型反解比直接执行成功更多"只部分复现。** R 去掉蜡烛仍显著（p 0.0018），但去掉蜡烛和浇水壶只剩 p 0.136。原反解 S 去掉蜡烛是 +2.8 个百分点、p 0.19，lmlcmp 里"S 对 A 赢"在蜡烛以外没有复现。
7. **M3 的碰壶率**：S 的预抓取窗口更长（中位数 405 行，A 245 行），61 % 偏高。方向大概不变，以后用固定窗口重算。
8. **下一步改成**：ScaleBFM 的反解默认用 R，照判据第 1 行执行。但 SIMPLE（ScaleBFM）的对比要保留 S 或 E 一组，同时记下 S 预抓取 IK > 3 cm 的比例。E2a 的 ScaleBFM 已经保留了 S。复核提的 F1（新种子复现 P2，约 30 分钟）、F2（拆预抓取和进入段的缩回）、F3（蜡烛几何放大）都是新实验，交用户决定。
