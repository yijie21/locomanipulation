# selfmodel_simpleheft_20261126：SIMPLE 主指标加 HEFT（直接执行 A，闸门，再一次性修正 R_D）

- **状态**：跑完，独立复核已回（2026-11-26）。GATE PASS（XMovePick、XMoveBendPick），R_D 当天跑完。HEFT 的 A 进主指标（协调方 11-26 定，是对登记判据的事后放宽，见"意外"第 1、6 条）。R_D 多 4 局，"提高成功率"证据不足。
- **日期**：2026-11-26。
- **为什么做**：用户 11-26 收窄了范围："不要再发散了"。三个基准、三个控制器，各跑一次自己的自身模型，拿任务指标，和直接发命令比。
  - SIMPLE 上 ScaleBFM 已经有了：45 s 内 A 83/120，R_D 108/120（simplecmp）。
  - 本条补 HEFT 这一格。SONIC 那一格另开条目。
- **HEFT 在 SIMPLE 上还没有任何正式数**：`locomani.bench.simple` 对 `controller=heft` 的说明是"wired and smoke-tested, no evaluation yet"。所以 A 本身也是新数，要进主指标。
- **用哪个模型：D_heft。**
  - 路径：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/heft/models_D_s0/E3_ensemble_f1`。
  - 来源：train3b 的 HEFT 模型 D。LML 的 data3 + data3b + 两份 mid，训练 499 077 段。
  - 特征：GOAL_REF、APPROACH、QFIX，没有 FDVEL。
  - LML val 中位数 3.204 cm（train.json）。
  - 选它的理由：LML 主表里 HEFT 的 R 用的就是它（lmlroute 修订 1）。"每个控制器自己的自身模型"在 HEFT 这里就是它。
  - 参照模型 F_heft：`selfmodel_fdvel_20261126/heft/models_s0`，只用 data3，加了 FDVEL，val 4.598 cm。它只在闸门里**并排打印**，不参与决定。
- **已知的迁移情况**（xfer 组合 6，HEFT → SIMPLE，只有原地伸手）：误差是 LML val 的 1.06 倍 [0.86, 1.22]；模型 / 直接 0.44；超出训练范围 1.6 %。
  - 但那是原地伸手。SIMPLE 的任务局要先走到站位再伸手。nowalk 复核对 HEFT 的结论是"证据不足，留出集指向需要先走后伸"。
  - 所以要用真实任务局重新查，这就是闸门。
- **机器 / GPU**：4090（`ssh -p 27984 root@14.103.60.188`），1、2、3 号卡。
  - 不放 H20b：H20b 11-26 接入核对通过，但 HEFT 的 SIMPLE 任务局在那里没核对过。4090 上已经和 `ws/code/loco_heft` 对过逐位一致，这一格的量也小（6 个进程），所以 A、AL、R_D 都在 4090 上跑，同机按局配对。
  - 0 号卡不碰。每张卡总量 ≤ 20 GB（排队脚本的上限是 19 800 MiB）。
  - 我们每张卡最多 2 个进程，一共 6 个，nice 10，每进程 1 线程。
  - 12:20Z 的显存：1、2、3 号卡各用了约 16.4 GB（HA 采集，每卡两个）。一个 simple_bench 进程约 0.65 GB，两个加上去约 17.7 GB。
  - CPU：我们自己的进程合计 ≤ 80 核，现在约 26 核；本条再加约 6–8 核。
- **代码**：
  - 分支 `sonic-selfmodel`。私有副本从 Mac 的工作树拷过去（`sh_copy.sh`），包括别人还没提交的改动。不用 `ws/code/loco_heft` 本身，但和它对过逐位一致（见"登记前核对"）。
  - 修正开关用 simplecmp 的 `simple_bench.py --correct` 和 `scmp_correct.py`。本条在 `scmp_correct.py` 上加了两处，都只在给了 `--correct` 时才会走到，所以不影响默认行为：
    1. 控制器断言从"只许 scalebfm"放宽到 `("scalebfm", "heft")`。HEFT 的 Motor 是 ScaleBFM Motor 的子类，记录器、候选命令的规划和手点用的是同一份代码。它在这个身体里的参考延迟是 0 帧，和 ScaleBFM 一样（`sc_collect.DELAY_F`）。
    2. `REACHWM_FDVEL` 不再一律要求 1，改成按模型对：simpledata 的 D 要 1，train3b 的 D 要 0，`ModelD` 加载时检查。
  - 正式运行前用 `sh_freeze.sh` 冻结到 `asrun/code`，排除 `__pycache__`，md5 记在 `asrun/code/CODE_MD5`、`asrun/KEY_MD5`、`asrun/INPUTS_MD5`。
  - **局的目录树不写共享树**（协调方 11-26）。副本里的 `outputs/simple_eval_20261112`（测试局）和 `outputs/simple_dev_20261124`（开发局）是副本自己的真目录：
    - `scene/` 是副本自己的真目录。simple_bench 每局都重写 `scene/<局>/`，所以不用事先拷 scene 文件进去。
    - 其余条目（缓存好的那一眼、局的清单）链接到共享树，只读。
    - `sh_copy.sh` 把 `find -type l` 和每个链接的 realpath 记在副本的 `LINKS` 里。
    - `sh_run.sh` 开跑前检查：树和 `scene/` 都不是链接，`scene/` 的 realpath 在本条的输出目录下，否则不跑。
  - simplecmp 的条目和脚本还没提交，等用户定；`simple_bench.py` 的 `--correct` 也还在工作树里没提交。本条要用到它们，所以本条运行的代码以冻结副本为准。登记时的 md5：
    - `scmp_correct.py`（加了上面两处）d80b3ed1；
    - `simple_bench.py` 8e40e961；
    - `ss_f1pre.py` b420de40（和 simplesonic 的 F1-0 用的同一份，随本条提交）。
- **脚本**（`scripts/experiments/selfmodel_simpleheft_20261126/`）：
  - `sh_copy.sh`：建私有副本。
  - `sh_freeze.sh`：冻结。
  - `sh_run.sh A|R`：正式运行。
  - `sh_gate.sh`：闸门。它调用 simplesonic 的 `ss_f1pre.py`（F1-0，规则不变）和本条的 `sh_err.py`（真实局上的预测误差，F1(a)）。
  - `sh_f1.sh`：阶段 R 之后，R_D 命令上的 F1。
  - `sh_eval.py`：登记的评估，统计函数从 simplecmp 的 `scmp_eval.py` 导入。
- **输出**：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/`（`run/`、`gate/`、`checks/`）。
- **关联**：
  - [selfmodel_simplecmp_20261126](selfmodel_simplecmp_20261126.md)：ScaleBFM 的 A 对 R_D。R 规则、对准手掌、判定集、布局符号检验都照它。
  - [selfmodel_simplesonic_20261126](selfmodel_simplesonic_20261126.md)：F1-0 的规则和 `ss_f1pre.py`。
  - [selfmodel_train3b_20261125](selfmodel_train3b_20261125.md)：D_heft。
  - [selfmodel_xfer_20261125](selfmodel_xfer_20261125.md)：HEFT → SIMPLE 原地伸手的误差。
  - [selfmodel_nowalk_20261126](selfmodel_nowalk_20261126.md)：HEFT 能不能只采原地，证据不足。
  - [selfmodel_simpledata_20261126](selfmodel_simpledata_20261126.md)：过不了闸门时，采数据方案照它的完整配方。

## 假设

1. **主假设**：动作和控制器都不变，只把 A 的两下直线命令（预抓取和进入）换成在 D_heft 上反解出来的命令 R_D，HEFT 的手就落得更准、成功更多。这和 ScaleBFM 在 SIMPLE 上看到的一样。
2. **前提**：D_heft 在 HEFT 的 SIMPLE 任务局上可信。也就是起点在它的训练范围里，预测误差没有比它在 LML 上大很多。这条最可能不成立，所以先查（闸门）。
3. **增益上限**：一次性修正只改目标，修不了站位偏，也修不了手臂被桌沿挡住。开发局上的 HEFT A 见"登记前核对"。

## 做法

- **局**：主指标的测试局。四个抓取任务（XMovePick、XMoveBendPick、BendPick、TabletopGrasp），每个 30 局（level 0–2 × index 0–9），局号和种子都和主指标相同。
- **跑法**：
  - 感知栏，`--bench-set controller=heft --bench-set sensor=rgbd`。
  - "那一眼"是每局缓存好的 RGB-D 结果（`cloud_blind`、`look_dump_blind`），和其他控制器读的是同一份文件。
  - simple_bench 的 A 选项和 bench_via.sh 一样，加 `--trace`，不录像。
  - `REACH_HAND_OFF=0.0415,0,0`，`REACH_HEFT_DELAY=0`。
  - HEFT 在进程里从 ONNX 跑，走路用 gait_heft。
- **组**：
  - **A**：直接执行，不加 `--correct`。这就是 HEFT 的主指标 A 栏。
  - **AL**：A 加上 `--correct D_heft --correct-rule dryR`。发出去的还是 A 的命令，只是每一下都把 D_heft 对落点的预测记下来。所以 AL 必须和 A 逐局相同。AL 的记录用来算真实局上的预测误差。
  - **R_D**（过闸门才跑）：`--correct D_heft --correct-rule R`。R 规则照 simplecmp：
    - 6 步不动点迭代，对准手掌；
    - 目标收缩到 IK 误差 ≤ 2 cm；
    - 损失里加 λ = 1.0 乘集成的分散度。
  - 三组都在 4090 上跑，按局配对。
- **先后**：
  1. 阶段 A：A 和 AL 一起排队，每任务先 A 后 AL，每个作业 5 局。
  2. 闸门：`sh_gate.sh`，只读、只用 CPU、不跑仿真。
  3. 过了闸门，当天跑阶段 R：R_D，120 局。

## 闸门（阶段 A 跑完后算；规则写于 2026-11-26T12:35Z，在跑 A 之前）

按任务判，两项都满足才算"过"：

1. **F1-0：起点在训练范围里 ≥ 80 %。**
   - 规则和 simplesonic 的 F1-0 相同，`ss_f1pre.py` 原样用。
   - 起点：本条 A 栏 120 局，每局预抓取和进入两下的 `S0`，也就是下命令时的 29 个关节角。
   - 训练范围：D_heft 自己的训练划分（`data.splits` 的 train，499 077 段）。取每段起点行（`so + k0`）的 29 个关节，每个关节取 0.5 % 和 99.5 % 分位。29 个关节都在里面，才算"在范围内"。
   - 按任务、两下合起来算比例。预抓取和进入分开的数一起报。
2. **F1(a)：真实局上的预测误差 ≤ 1.3 倍。**
   - 数据：AL 记录的每一下伸手，在任务标签处（sd_real 的规则），不含摔倒的。
   - 算法：D_heft 对落点的预测（13 cm 点）和实际落点的距离，取中位数，除以 D_heft 自己的 LML val 中位数 3.204 cm。
   - 按任务、两下合起来算。分开的数、直接执行的偏差、模型 / 直接一起报。区间按布局（任务，index）重抽 1000 次，seed 0。

**结论**：至少一个任务"过"，记 GATE PASS，当天跑 R_D。四个任务都不过，记 GATE FAIL。

F_heft 的两项也打印，只作参照，不改变结论。F_heft 那一行的"在线 = 离线"是拿它和 D_heft 的在线记录比，不是核对。

**另外报，不进闸门**：D_heft 的 F1(a) 再按"这一下有没有碰到桌子"分开算一遍。碰桌子看行里的 `hits`，也就是这一段里手腕或手碰到 table 的记录。

- 理由：开发局上 HEFT 的手腕在预抓取时碰到桌子，手被挡在桌沿下面，模型没见过这种接触。
- 碰桌子的伸手误差大，说的是"桌子挡手"，不是"模型不认识 HEFT 在 SIMPLE 里的身体"。
- 两种情况下一步不同：后一种要在 SIMPLE 里采数据；前一种采数据也修不好，要绕开桌沿。
- 所以闸门不过时，采数据方案要按这个分开的数来写。

## 局和判定集

- **布局**：同一个 index 的三个 level 几乎一样，所以 120 局是 40 个布局。配对检验在布局上做。
- **排除**：照 simplecmp，三类分开计：
  - 感知失败；
  - 看完以后、发预抓取之前规划失败；
  - 没有那一眼。
  - 这几类在两组分开之前就失败了，不进判定集。
- **整任务排除**：照 lmlcmp3 和 simplecmp，两组分开之前就失败的局超过一半的任务，整个不进判定集。
- **到顶**：A ≥ 24/30（80 %）的任务不进那个时限的判定集，只作描述。
- **官方时限**：判定集的规则同上，和 simplecmp 一样。BendPick 整局约 12 s，超过 8 s 的官方时限，各组都不可能成功（ScaleBFM 的 A 和 R_D 都是 0/30）。它留在判定集里，但不贡献差别。

## 度量

- **主比较 P1**：R_D 对 A，45 s，判定集。
  - 每个布局比三个 level 的成功数，做精确的双侧符号检验，p < 0.05。
  - McNemar 一起报，不作判据。
  - 成功率差的 95 % 区间：按布局重抽 4000 次，seed 0。
- **官方时限**：同样的检验，判定集按上一节。
- **按任务**：四个任务各自报 A、R_D 的成功数，两个时限都报，加同样的检验和区间。
- **F1，模型在修正后的命令上还可信吗**：按任务算三项，任何一项不满足就判"不可信"：
  - (a) D_heft 在 R_D 记录上的误差中位数 / 3.204 cm > 1.3；
  - (b) R_D 起点的 F1-0 比例 < 80 %。这里用 F1-0 代替 simplecmp 的 39 项检查，因为 sd_range 一 import 就断言 FDVEL = 1，D_heft 没有 FDVEL；
  - (c) 发出命令里 IK 误差 > 2 cm 的比例，比 AL 高 ≥ 10 个百分点。
  - AL 的三项一起报，作参照。
- **机制**：
  - 每一下手掌偏差的中位数，A 和 R_D，按任务、按段；
  - D_heft 对发出命令的预测误差；
  - 模型 / 直接（AL）；
  - 手臂碰桌子的伸手数（行里的 `hits`）；
  - 修正量、缩回次数、思考时间。
- **一致性**：
  - AL 和 A 逐局相同。比较时去掉 `wall_s`、`correct_reset`、`film`、各段的 `correct` 字段，以及 bench 记录里的 `sonic_vr_frame`，和 simplecmp 的 `clean()` 一样。
  - 在线、离线预测之差 < 1e-4 m（日志按 1e-5 取整）。
  - plan_same 全为真。
  - 那一眼相同：A 每局的 `cloud`、`score`、`seen_cm` 和主指标 ScaleBFM A 栏同一局的相同。
  - 共享树没被写：开跑前 `sh_run.sh` 检查 `scene/`；跑完时副本自己的 `scene/` 下应有 120 个局目录（每局写一次）。

## 预测

- **A**：
  - 45 s 内 0–30/120。理由：
    - HEFT 在 SIMPLE 上从没正式跑过；
    - 开发局 TabletopGrasp 30、36 两局都失败，手腕在预抓取时碰到桌子，手掌差 31–34 cm（见"登记前核对"）；
    - LML 上 HEFT 的 A 在判定集里约 10 %。
  - 官方时限内 ≤ 45 s 的数，BendPick 是 0。
- **F1-0（D_heft，两下合起来）**：
  - 每个任务 40–85 %，预抓取高于进入。SONIC 上同样的规则是 85.0 % 对 32.5 %。
  - 有任务 ≥ 80 % 的把握约四成。
- **F1(a)**：
  - TabletopGrasp ≥ 3 倍。开发局两局是 10 倍，手腕都碰到了桌子，落点离目标很远，而模型没见过这种接触。
  - 其他三个任务 1.2–3 倍。哪个任务碰桌子多，哪个倍数就大。
  - 不碰桌子的伸手：1.0–1.6 倍。xfer 原地伸手是 1.06 倍，走到站位以后起点会远一些。
- **闸门**：GATE FAIL 的可能大一些（约 65 %）。
- **R_D（如果跑）**：
  - 45 s 内 R_D − A 在 −5 到 +10 局，布局符号检验不显著。
  - 没碰桌子的伸手，手掌偏差降 20–50 %。

## 判据

| 结果 | 结论和下一步 |
|---|---|
| AL 和 A 不逐局相同，或在线、离线之差 ≥ 1e-4 m，或 plan_same 有假 | 停下查清，不算闸门 |
| 那一眼和主指标 ScaleBFM A 栏不同 | 停下查清；A 不进主指标 |
| GATE FAIL（四个任务都不过） | 不跑 R_D。写 HEFT 在 SIMPLE 采数据的方案：照 simpledata 的完整配方，保留先走后伸（协调方 11-26，按 nowalk 复核），写清局数、每局耗时、核数、小时数。报协调方，不跑 |
| GATE PASS（≥ 1 个任务过） | 当天跑 R_D，四个任务 120 局，同机按局配对。只有闸门和 F1 都判"可信"的任务，成功差才记到模型名下 |
| P1 成立（R_D 多），而且带来提高的任务闸门和 F1 都过 | SIMPLE 上 HEFT 的一次性修正提高了成功率，写明是哪个任务带来的 |
| P1 成立，但带来提高的任务闸门或 F1 不过 | 照报，不归功于自身模型 |
| P1 不成立 | 照实报各任务的差和区间，不合并成"赢" |
| 某任务 R_D − A ≤ −3 局 | 修正在这里有害，照报，并写明这个任务的闸门和 F1 |
| 手掌偏差没降，而闸门和 F1 判"可信" | 反解没起作用，先查模型对修正命令的预测误差有没有变大 |

## 运行计划

- 阶段 A：48 个作业（A、AL 各 24），6 个进程。
- 阶段 R：24 个作业。
- 开发局上一局约 10–16 s，加上每个作业的启动。估计阶段 A 约 20–30 分钟，阶段 R 约 15 分钟。
- 每阶段开跑前跑 `nvidia-smi`，并登记到灵动岛（`~/.claude/longjobs/simpleheft.json`）。

## 登记前核对

都在 4090 上、`code_dev` 副本里跑，记录在 `checks/`。开发局只有 TabletopGrasp 有缓存好的那一眼（开发树里只有 `blind_f1_TabletopGraspMP.json`），其他三个任务的开发局作业都在读这个文件时退出（exit 1），没有跑起来。所以冒烟用了开发局 TabletopGrasp 30、36，另外用测试局 BendPick 第 0 局查下蹲那条路径的管线。测试局这一局只看退出码、AL = A 和 plan_same，没读成败和偏差。

1. **和 `ws/code/loco_heft` 逐位一致**：开发局 30、36，两份代码各跑一次 A，行完全相同。只有 bench 记录里多一个 `sonic_vr_frame`，这是新代码才记的字段，它的缺省值。
2. **AL = A**：开发局 30、36 和测试局 BendPick 0，三局去掉 `clean()` 去掉的字段后都相同。
3. **plan_same 全为真**：AL 和 R_D 的每一下都是。在线、离线预测之差最大 5.0e-6 m，日志按 1e-5 取整。
4. **R_D 跑得起来**：
   - 开发局 30、36：修正量 7.5–11.6 cm，IK 误差 ≤ 0.001 cm，没有退回。
   - 测试局 BendPick 0：两下都 plan_same，退出码 0。
   - 思考时间：预抓取约 5.3 s、进入约 1.6 s（TabletopGrasp）；BendPick 1.7 s 和 1.1 s。
5. **开发局上 HEFT 的 A**（这两局用来写预测）：
   - 两局都失败。预抓取的手掌偏差 35.0 和 37.3 cm，进入 31.2 和 33.6 cm。
   - 两局的右手腕都在预抓取时碰到桌子，手停在目标后方约 25 cm、下方约 23 cm（偏差向量 [−25, −8, −23] cm）。走路没问题：站位偏 3 cm，里程计偏 0.45 cm。
   - D_heft 对这两局的预测误差是 28–29 cm（预抓取）和 26–27 cm（进入），F1(a) 约 10 倍。四个起点的 F1-0 都在范围里。
   - R_D 的偏差几乎没变：预抓取 32.1 和 34.8 cm，进入 29.6 和 31.8 cm。手还是被桌子挡住。
6. **闸门脚本**：把开发局的 A 行冒充四个任务的行，跑通了 `sh_gate.sh`。输出格式和判定都对：F1-0 100 %，TabletopGrasp 的 F1(a) 10.3 倍，GATE FAIL。
7. **局的目录树**：前面几步用的是本条输出目录下的 `devout/`、`evalout/`。它们的 `scene/` 是真目录，其余链接到共享树，只读。协调方 11-26 要求树放进副本里，改成副本自己的 `outputs/simple_*` 真目录以后，开发局 30、36 的 A 重跑一次，见第 8 条。共享树里这几局的 scene 目录，修改时间都不在我们运行的时段里（开发树 08:22–08:23，测试树 BendPick_L0_e0 11:04）。
8. **换树以后的 A**：副本重拷一次（`sh_copy.sh code_dev`，树在副本里），开发局 30、36 的 A 重跑，行和第 1 条的逐位相同（只去掉 wall_s）。scene 写在副本自己的 `outputs/simple_dev_20261124/scene/` 下。

## 修订 1（2026-11-26T12:45Z，阶段 A 在跑，闸门还没算，R_D 开跑之前提交）

协调方 11-26 先定好了闸门的两种结果，免得等。下面两条替换"判据"表里 GATE FAIL 和 GATE PASS 两行的"下一步"。其他判据不变，预测不变。

1. **至少一个任务过闸门**：
   - R_D 四个任务都跑（120 局），每个任务旁边注明闸门过没过。这和用户对 HA 的规矩相同："照跑对比，把检验结果注在旁边"。
   - **主比较**：R_D 对 A，四个任务合计，45 s 和官方时限都报。
   - 旁边同时报：只算过闸门的任务，合计。
   - 登记的 P1（判定集上的布局符号检验）照报，不删。
   - 成功差只在闸门和 F1 都判"可信"的任务上记到模型名下，这条不变。
2. **四个任务都不过**：
   - 不用再等协调方，直接在 H20b 上采 HEFT 的 SIMPLE 数据。
   - 照 simpledata 的完整配方，保留先走后伸。原因是 nowalk 复核对 HEFT 是"证据不足，留出集指向需要先走后伸"。
   - 和 SONIC 共用 140 核，我自己分配。
   - 另开条目，登记后开跑，开跑时把局数和预计时间报协调方。
   - 新模型训好后，当天在真实局上检验（本条的闸门规则），然后跑 R_D。
3. **评估脚本跟着改**：`sh_eval.py` 加了第 5 节的主比较（四个任务合计）和"过闸门任务合计"，每个任务后面注明闸门结果。闸门结果按 `sh_gate.sh` 的规则从 `--gate` 目录读。新版 md5 7f342120，另存在 `asrun/rev1/sh_eval.py`。冻结副本里的旧版不动，评估用新版。
4. **HA 有没有被拖慢**（协调方 11-26 要求开跑前后看一眼）：
   - HA 采集的 `controller.step()` 频率：开跑前 30 分钟中位数 13.46 Hz（3 407 个记录），开跑后到 12:41Z 中位数 13.37 Hz（693 个），慢 0.7 %。每次重置的秒数，开跑前中位数 62 s，开跑后 53 s；样本少，又随重置的种类变，只作参考。
   - 显存：12:39Z 起每 5 s 采一次（`checks/mem.log`），1、2、3 号卡的峰值是 16 376、16 466、16 373 MiB，和开跑前一样。本条的 simple_bench 进程不录像，那一眼读的是缓存，在卡上看不出占用。
   - 读法：没有 10 % 以上的变慢，也远离 20 GB，所以不挪 H20b，也不降到每卡 1 个。阶段 R 开跑前后再看一次。

## 预测修订

没有。闸门算完以后、R_D 开跑以前都没改预测。修订 1 只改了闸门两种结果的"下一步"和评估的主比较。

## 结果（贴原始输出，不转述）

- **冻结和运行**：
  - 冻结在 2026-11-26T12:36:05Z（`asrun/FROZEN_UTC`，冻结时的提交 4d022c97）。CODE_MD5 在评估之后又核了一次，通过。
  - 4090 的 1、2、3 号卡，0 号卡没碰。每阶段开跑前的 nvidia-smi 在 `run/nvidia_smi_A.txt`、`run/nvidia_smi_R.txt`。
  - 阶段 A：12:36:26Z 到 13:13:11Z，48 个作业（A、AL 各 24），比预计的 20–30 分钟长。
  - 阶段 R：13:15:43Z 到 13:34:25Z，24 个作业。
  - 72 个作业退出码都是 0（`run/run.log`），`run/logs/` 的 12 个日志里没有 Traceback。
- **共享树没被写**：
  - 副本自己的 `asrun/code/outputs/simple_eval_20261112/scene/` 是真目录，有 120 个局目录，每个任务 30 个。
  - 共享树 `ws/code/loco/outputs/simple_eval_20261112` 里，`scene/` 下一层和其余两层都没有 12:36Z 以后改过的文件。
- **原始输出**都在 `/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/`：
  - `gate/gate.txt`（md5 3249faa5）：闸门；
  - `run/eval_A.txt`：阶段 R 之前的一致性；
  - `f1/f1.txt`（9791fc3c）：R_D 命令上的 F1；
  - `eval.txt`（4ac18fe7）、`eval.json`（f2eff547）：登记的评估；
  - 行在 `run/rows/`，修正记录在 `run/corr_AL/`、`run/corr_RD/`。

### 闸门：`sh_gate.sh` → `gate/gate.txt`（13:13:27Z，阶段 A 跑完后、R_D 开跑前）

```
gate 2026-11-26T13:13:27Z: rows /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/rows, AL record /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/corr_AL
== D_heft: train /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3b_20261125/heft/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/heft/data/mid_data3.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/heft/data/mid_data3b.npz; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '0', 'REACHWM_TRAIN_HELD': '', 'REACHWM_SUBSET': ''}
  train split 499077 segments (train.json n_train 499077: equal); val 4321, val starts in range 87.3 %
  rows A_XMovePickTeleop.jsonl: 30 episodes, md5 aff9c182538c6863aabe4bf89476b888
  rows A_XMoveBendPickTeleop.jsonl: 30 episodes, md5 8d1830e413711056aea44aab5bbac845
  rows A_BendPickMP.jsonl: 30 episodes, md5 6b971f539133cf8bc4f6c704ad2be330
  rows A_TabletopGraspMP.jsonl: 30 episodes, md5 40eddeb56a7b50fe0ed963d079d2266e
  in range (all 29 joints inside the train split's 0.5 / 99.5 %): share [95 %, layouts resampled]; joints out median; most often out (% of starts)
    XMovePickTeleop      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    BendPickMP           pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    BendPickMP           entry    n  30   83.3 % [ 66.7,  96.7]  out median 0  [('right_hip_roll_joint', 13.3), ('left_knee_joint', 3.3)]
    BendPickMP           both     n  60   91.7 % [ 83.3, 100.0]  out median 0  [('right_hip_roll_joint', 6.7), ('left_knee_joint', 1.7)]
    TabletopGraspMP      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    TabletopGraspMP      entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    TabletopGraspMP      both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    all                  pregrasp n 120  100.0 % [100.0, 100.0]  out median 0  []
    all                  entry    n 120   95.8 % [ 90.8, 100.0]  out median 0  [('right_hip_roll_joint', 3.3), ('left_knee_joint', 0.8)]
    all                  both     n 240   97.9 % [ 95.0, 100.0]  out median 0  [('right_hip_roll_joint', 1.7), ('left_knee_joint', 0.4)]
== F_heft_reference: train /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/data/all.npz; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '1', 'REACHWM_TRAIN_HELD': '', 'REACHWM_SUBSET': ''}
  train split 20518 segments (train.json n_train 20518: equal); val 2539, val starts in range 83.0 %
  rows A_XMovePickTeleop.jsonl: 30 episodes, md5 aff9c182538c6863aabe4bf89476b888
  rows A_XMoveBendPickTeleop.jsonl: 30 episodes, md5 8d1830e413711056aea44aab5bbac845
  rows A_BendPickMP.jsonl: 30 episodes, md5 6b971f539133cf8bc4f6c704ad2be330
  rows A_TabletopGraspMP.jsonl: 30 episodes, md5 40eddeb56a7b50fe0ed963d079d2266e
  in range (all 29 joints inside the train split's 0.5 / 99.5 %): share [95 %, layouts resampled]; joints out median; most often out (% of starts)
    XMovePickTeleop      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  entry    n  30   93.3 % [ 83.3, 100.0]  out median 0  [('right_hip_roll_joint', 6.7)]
    XMoveBendPickTeleop  both     n  60   96.7 % [ 91.7, 100.0]  out median 0  [('right_hip_roll_joint', 3.3)]
    BendPickMP           pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    BendPickMP           entry    n  30   30.0 % [ 10.0,  50.0]  out median 1  [('right_hip_roll_joint', 66.7), ('left_knee_joint', 3.3)]
    BendPickMP           both     n  60   65.0 % [ 56.7,  76.7]  out median 0  [('right_hip_roll_joint', 33.3), ('left_knee_joint', 1.7)]
    TabletopGraspMP      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    TabletopGraspMP      entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    TabletopGraspMP      both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    all                  pregrasp n 120  100.0 % [100.0, 100.0]  out median 0  []
    all                  entry    n 120   80.8 % [ 70.0,  90.0]  out median 0  [('right_hip_roll_joint', 18.3), ('left_knee_joint', 0.8)]
    all                  both     n 240   90.4 % [ 85.4,  95.4]  out median 0  [('right_hip_roll_joint', 9.2), ('left_knee_joint', 0.4)]
== D_heft: model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/heft/models_D_s0/E3_ensemble_f1, val 3.204 cm; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '0'}
  AL: /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/corr_AL: 240 segments (240 kept, not fallen); online = offline on 240 sent stages (plan_same): max |pred diff| 4.998e-06 m; 0 stages without a pair
    model error at the task label, median cm [95 %, layouts resampled] / val = F1(a) ratio [95 %]; direct miss median; model / direct [95 %]
    reaches with a table contact (rows' hits): 121 of 240 kept
    XMovePickTeleop      pregrasp n  30:  0.619 cm [ 0.401,  0.740] = 0.193x [0.125, 0.231]; direct  4.110 cm; model / direct 0.151 [0.105, 0.174]
    XMovePickTeleop      entry    n  30:  0.433 cm [ 0.328,  0.460] = 0.135x [0.102, 0.143]; direct  3.171 cm; model / direct 0.136 [0.104, 0.145]
    XMovePickTeleop      both     n  60:  0.451 cm [ 0.417,  0.484] = 0.141x [0.130, 0.151]; direct  3.413 cm; model / direct 0.132 [0.116, 0.142]
    XMoveBendPickTeleop  pregrasp n  30:  1.756 cm [ 1.045,  3.024] = 0.548x [0.326, 0.944]; direct  3.854 cm; model / direct 0.456 [0.428, 0.537]
    XMoveBendPickTeleop  entry    n  30:  1.065 cm [ 0.558,  1.742] = 0.332x [0.174, 0.544]; direct  3.494 cm; model / direct 0.305 [0.242, 0.360]
    XMoveBendPickTeleop  both     n  60:  1.402 cm [ 1.007,  2.018] = 0.438x [0.314, 0.630]; direct  3.635 cm; model / direct 0.386 [0.313, 0.513]
    BendPickMP           pregrasp n  30:  3.948 cm [ 1.684,  5.557] = 1.232x [0.526, 1.734]; direct  6.293 cm; model / direct 0.627 [0.409, 0.840]
    BendPickMP           entry    n  30:  4.433 cm [ 2.901,  5.361] = 1.384x [0.906, 1.673]  > 1.3; direct  9.782 cm; model / direct 0.453 [0.374, 0.576]
    BendPickMP           both     n  60:  4.367 cm [ 2.344,  5.346] = 1.363x [0.732, 1.668]  > 1.3; direct  6.872 cm; model / direct 0.635 [0.421, 0.694]
    TabletopGraspMP      pregrasp n  30: 34.457 cm [32.937, 35.106] = 10.754x [10.280, 10.957]  > 1.3; direct 43.630 cm; model / direct 0.790 [0.761, 0.804]
    TabletopGraspMP      entry    n  30: 29.056 cm [28.402, 29.785] = 9.069x [8.865, 9.296]  > 1.3; direct 37.622 cm; model / direct 0.772 [0.753, 0.787]
    TabletopGraspMP      both     n  60: 31.355 cm [30.189, 32.842] = 9.786x [9.422, 10.250]  > 1.3; direct 40.658 cm; model / direct 0.771 [0.743, 0.789]
    all                  pregrasp n 120:  2.981 cm [ 1.018,  4.748] = 0.931x [0.318, 1.482]; direct  5.509 cm; model / direct 0.541 [0.244, 0.728]
    all                  entry    n 120:  1.821 cm [ 0.717,  5.115] = 0.568x [0.224, 1.597]; direct  5.716 cm; model / direct 0.319 [0.177, 0.530]
    all                  both     n 240:  2.494 cm [ 0.956,  4.765] = 0.778x [0.298, 1.487]; direct  5.579 cm; model / direct 0.447 [0.230, 0.630]
    XMovePickTeleop free pregrasp n  29:  0.572 cm [ 0.401,  0.726] = 0.178x [0.125, 0.227]; direct  4.100 cm; model / direct 0.139 [0.105, 0.173]
    XMovePickTeleop free entry    n  29:  0.431 cm [ 0.327,  0.457] = 0.135x [0.102, 0.143]; direct  3.170 cm; model / direct 0.136 [0.103, 0.144]
    XMovePickTeleop free both     n  58:  0.446 cm [ 0.414,  0.483] = 0.139x [0.129, 0.151]; direct  3.407 cm; model / direct 0.131 [0.119, 0.141]
    XMoveBendPickTeleop free pregrasp n  28:  1.270 cm [ 1.045,  3.003] = 0.396x [0.326, 0.937]; direct  2.555 cm; model / direct 0.497 [0.420, 0.535]
    XMoveBendPickTeleop free entry    n  17:  0.557 cm [ 0.496,  0.597] = 0.174x [0.155, 0.186]; direct  2.158 cm; model / direct 0.258 [0.235, 0.284]
    XMoveBendPickTeleop free both     n  45:  1.045 cm [ 0.942,  2.645] = 0.326x [0.294, 0.825]; direct  2.378 cm; model / direct 0.439 [0.377, 0.516]
    BendPickMP free      pregrasp n   8:  0.857 cm [ 0.846,  0.983] = 0.268x [0.264, 0.307]; direct  3.112 cm; model / direct 0.275 [0.270, 0.330]
    BendPickMP free      entry    n   8:  0.865 cm [ 0.854,  0.924] = 0.270x [0.267, 0.288]; direct  3.102 cm; model / direct 0.279 [0.275, 0.311]
    BendPickMP free      both     n  16:  0.862 cm [ 0.850,  0.949] = 0.269x [0.265, 0.296]; direct  3.103 cm; model / direct 0.278 [0.274, 0.310]
    all free             pregrasp n  65:  0.850 cm [ 0.725,  1.045] = 0.265x [0.226, 0.326]; direct  3.824 cm; model / direct 0.222 [0.179, 0.311]
    all free             entry    n  54:  0.484 cm [ 0.434,  0.559] = 0.151x [0.135, 0.175]; direct  3.128 cm; model / direct 0.155 [0.140, 0.190]
    all free             both     n 119:  0.666 cm [ 0.503,  0.852] = 0.208x [0.157, 0.266]; direct  3.172 cm; model / direct 0.210 [0.156, 0.271]
== F_heft_reference: model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_fdvel_20261126/heft/models_s0/E3_ensemble_f1, val 4.598 cm; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '1'}
  AL: /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/corr_AL: 240 segments (240 kept, not fallen); online = offline on 240 sent stages (plan_same): max |pred diff| 3.781e-01 m; 0 stages without a pair
    model error at the task label, median cm [95 %, layouts resampled] / val = F1(a) ratio [95 %]; direct miss median; model / direct [95 %]
    XMovePickTeleop      pregrasp n  30: 12.889 cm [12.734, 12.991] = 2.803x [2.770, 2.825]  > 1.3; direct  4.110 cm; model / direct 3.136 [3.034, 3.355]
    XMovePickTeleop      entry    n  30:  1.786 cm [ 1.643,  2.008] = 0.388x [0.357, 0.437]; direct  3.171 cm; model / direct 0.563 [0.524, 0.618]
    XMovePickTeleop      both     n  60: 12.476 cm [ 7.301, 12.756] = 2.713x [1.588, 2.774]  > 1.3; direct  3.413 cm; model / direct 3.656 [2.040, 3.811]
    XMoveBendPickTeleop  pregrasp n  30: 12.625 cm [11.763, 15.046] = 2.746x [2.558, 3.272]  > 1.3; direct  3.854 cm; model / direct 3.276 [2.452, 5.050]
    XMoveBendPickTeleop  entry    n  30:  2.195 cm [ 0.784,  3.465] = 0.477x [0.170, 0.754]; direct  3.494 cm; model / direct 0.628 [0.362, 0.869]
    XMoveBendPickTeleop  both     n  60: 11.057 cm [ 7.295, 11.440] = 2.405x [1.587, 2.488]  > 1.3; direct  3.635 cm; model / direct 3.042 [1.398, 4.707]
    BendPickMP           pregrasp n  30: 12.319 cm [11.896, 13.262] = 2.679x [2.587, 2.884]  > 1.3; direct  6.293 cm; model / direct 1.957 [1.802, 3.224]
    BendPickMP           entry    n  30:  9.560 cm [ 6.155, 10.631] = 2.079x [1.339, 2.312]  > 1.3; direct  9.782 cm; model / direct 0.977 [0.733, 1.061]
    BendPickMP           both     n  60: 11.269 cm [10.627, 11.906] = 2.451x [2.311, 2.589]  > 1.3; direct  6.872 cm; model / direct 1.640 [1.169, 2.238]
    TabletopGraspMP      pregrasp n  30: 43.501 cm [42.763, 45.792] = 9.461x [9.300, 9.959]  > 1.3; direct 43.630 cm; model / direct 0.997 [0.982, 1.042]
    TabletopGraspMP      entry    n  30: 28.643 cm [27.939, 29.398] = 6.230x [6.076, 6.394]  > 1.3; direct 37.622 cm; model / direct 0.761 [0.748, 0.775]
    TabletopGraspMP      both     n  60: 36.454 cm [36.199, 37.576] = 7.928x [7.873, 8.172]  > 1.3; direct 40.658 cm; model / direct 0.897 [0.870, 0.904]
    all                  pregrasp n 120: 13.130 cm [12.846, 14.878] = 2.856x [2.794, 3.236]  > 1.3; direct  5.509 cm; model / direct 2.384 [2.003, 3.073]
    all                  entry    n 120:  3.499 cm [ 2.090, 10.381] = 0.761x [0.455, 2.258]; direct  5.716 cm; model / direct 0.612 [0.505, 1.023]
    all                  both     n 240: 12.693 cm [11.887, 13.285] = 2.761x [2.585, 2.889]  > 1.3; direct  5.579 cm; model / direct 2.275 [1.605, 3.032]
verdict (D_heft; F1-0 share >= 80 % and F1(a) ratio <= 1.3, both stages):
  XMovePickTeleop      F1-0 100.0 %  F1(a) 0.141x  -> PASS
  XMoveBendPickTeleop  F1-0 100.0 %  F1(a) 0.438x  -> PASS
  BendPickMP           F1-0  91.7 %  F1(a) 1.363x  -> FAIL
  TabletopGraspMP      F1-0 100.0 %  F1(a) 9.786x  -> FAIL
GATE PASS (XMovePickTeleop, XMoveBendPickTeleop)
```

### 阶段 R 之前的一致性：`run/eval_A.txt` 第 1 节

这是阶段 A 之后、用冻结副本里修订 1 之前那版 sh_eval 跑的。其余各节里 A、AL 的数，和下面 `eval.txt` 里的相同。

```
== rows: A 120, AL 120, RD 0, S 120

== 1. consistency
  A = AL: 120 / 120 episodes identical (every row and stage key but wall_s, --correct's fields and the option sonic_vr_frame)
  the look (cloud, score, seen_cm) = the main metric's ScaleBFM A: 91 / 120
  AL: 120 episodes logged, 240 stages, plan_same 240
```

"那一眼"一项 91/120，不同的 29 局只差在 `seen_cm`，见"意外"第 1 条。

### R_D 命令上的 F1：`sh_f1.sh` → `f1/f1.txt`

AL 的两段（F1-0 和误差）和 `gate.txt` 里 D_heft 的两段逐行相同（用 `diff` 比过）。只有行文件的 md5 不同：闸门读的是 `A_` 的行，这里链接的是 `AL_` 的行。下面只贴第一行和 RD 的两段。

```
F1 2026-11-26T13:37:09Z: rows /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/rows, records /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/corr_AL, /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/corr_RD
...
== D_heft_RD: train /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3b_20261125/heft/data/all.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/heft/data/mid_data3.npz,/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/heft/data/mid_data3b.npz; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '0', 'REACHWM_TRAIN_HELD': '', 'REACHWM_SUBSET': ''}
  train split 499077 segments (train.json n_train 499077: equal); val 4321, val starts in range 87.3 %
  rows A_XMovePickTeleop.jsonl: 30 episodes, md5 893e345cbbef847823d835d88d423249
  rows A_XMoveBendPickTeleop.jsonl: 30 episodes, md5 5939efddd19485f44cc6814e66be7f30
  rows A_BendPickMP.jsonl: 30 episodes, md5 343a738ad3ea7c012becd26305e567ff
  rows A_TabletopGraspMP.jsonl: 30 episodes, md5 11e17474b655993d743d1e171381f026
  in range (all 29 joints inside the train split's 0.5 / 99.5 %): share [95 %, layouts resampled]; joints out median; most often out (% of starts)
    XMovePickTeleop      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMovePickTeleop      both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    XMoveBendPickTeleop  both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    BendPickMP           pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    BendPickMP           entry    n  30   83.3 % [ 66.7,  96.7]  out median 0  [('right_hip_roll_joint', 16.7)]
    BendPickMP           both     n  60   91.7 % [ 85.0,  98.3]  out median 0  [('right_hip_roll_joint', 8.3)]
    TabletopGraspMP      pregrasp n  30  100.0 % [100.0, 100.0]  out median 0  []
    TabletopGraspMP      entry    n  30  100.0 % [100.0, 100.0]  out median 0  []
    TabletopGraspMP      both     n  60  100.0 % [100.0, 100.0]  out median 0  []
    all                  pregrasp n 120  100.0 % [100.0, 100.0]  out median 0  []
    all                  entry    n 120   95.8 % [ 90.8,  99.2]  out median 0  [('right_hip_roll_joint', 4.2)]
    all                  both     n 240   97.9 % [ 95.8,  99.6]  out median 0  [('right_hip_roll_joint', 2.1)]
== D_heft_RD: model /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_train3b_20261125/heft/models_D_s0/E3_ensemble_f1, val 3.204 cm; switches {'REACHWM_GOAL_REF': '1', 'REACHWM_APPROACH': '1', 'REACHWM_QFIX': '1', 'REACHWM_FDVEL': '0'}
  RD: /wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126/run/corr_RD: 240 segments (240 kept, not fallen); online = offline on 240 sent stages (plan_same): max |pred diff| 5.059e-06 m; 0 stages without a pair
    model error at the task label, median cm [95 %, layouts resampled] / val = F1(a) ratio [95 %]; direct miss median; model / direct [95 %]
    reaches with a table contact (rows' hits): 89 of 240 kept
    XMovePickTeleop      pregrasp n  30:  1.113 cm [ 0.856,  1.274] = 0.348x [0.267, 0.398]; direct  5.024 cm; model / direct 0.222 [0.192, 0.243]
    XMovePickTeleop      entry    n  30:  0.254 cm [ 0.230,  0.351] = 0.079x [0.072, 0.110]; direct  3.522 cm; model / direct 0.072 [0.066, 0.090]
    XMovePickTeleop      both     n  60:  0.508 cm [ 0.454,  0.651] = 0.159x [0.142, 0.203]; direct  4.113 cm; model / direct 0.123 [0.114, 0.152]
    XMoveBendPickTeleop  pregrasp n  30:  1.465 cm [ 1.035,  2.873] = 0.457x [0.323, 0.897]; direct  3.412 cm; model / direct 0.429 [0.382, 0.565]
    XMoveBendPickTeleop  entry    n  30:  1.058 cm [ 0.607,  1.714] = 0.330x [0.189, 0.535]; direct  3.209 cm; model / direct 0.330 [0.265, 0.393]
    XMoveBendPickTeleop  both     n  60:  1.372 cm [ 0.929,  2.122] = 0.428x [0.290, 0.662]; direct  3.407 cm; model / direct 0.403 [0.354, 0.508]
    BendPickMP           pregrasp n  30:  6.382 cm [ 3.666,  7.188] = 1.992x [1.144, 2.243]  > 1.3; direct  5.396 cm; model / direct 1.183 [0.782, 1.305]
    BendPickMP           entry    n  30:  5.173 cm [ 2.058,  7.955] = 1.615x [0.642, 2.483]  > 1.3; direct 11.500 cm; model / direct 0.450 [0.276, 0.553]
    BendPickMP           both     n  60:  5.992 cm [ 3.025,  7.544] = 1.870x [0.944, 2.355]  > 1.3; direct  6.279 cm; model / direct 0.954 [0.719, 1.168]
    TabletopGraspMP      pregrasp n  30: 40.013 cm [38.790, 40.929] = 12.489x [12.107, 12.774]  > 1.3; direct 47.449 cm; model / direct 0.843 [0.784, 0.877]
    TabletopGraspMP      entry    n  30: 34.648 cm [33.213, 34.986] = 10.814x [10.366, 10.919]  > 1.3; direct 40.885 cm; model / direct 0.847 [0.808, 0.860]
    TabletopGraspMP      both     n  60: 37.760 cm [37.193, 38.707] = 11.785x [11.608, 12.081]  > 1.3; direct 44.984 cm; model / direct 0.839 [0.813, 0.877]
    all                  pregrasp n 120:  2.878 cm [ 1.284,  6.669] = 0.898x [0.401, 2.081]; direct  5.403 cm; model / direct 0.533 [0.246, 1.116]
    all                  entry    n 120:  1.714 cm [ 0.661,  6.084] = 0.535x [0.206, 1.899]; direct  4.500 cm; model / direct 0.381 [0.164, 0.476]
    all                  both     n 240:  2.126 cm [ 1.161,  6.398] = 0.663x [0.362, 1.997]; direct  5.325 cm; model / direct 0.399 [0.250, 0.959]
    XMovePickTeleop free pregrasp n  30:  1.113 cm [ 0.856,  1.274] = 0.348x [0.267, 0.398]; direct  5.024 cm; model / direct 0.222 [0.191, 0.248]
    XMovePickTeleop free entry    n  30:  0.254 cm [ 0.230,  0.315] = 0.079x [0.072, 0.098]; direct  3.522 cm; model / direct 0.072 [0.065, 0.094]
    XMovePickTeleop free both     n  60:  0.508 cm [ 0.454,  0.651] = 0.159x [0.142, 0.203]; direct  4.113 cm; model / direct 0.123 [0.114, 0.152]
    XMoveBendPickTeleop free pregrasp n  30:  1.465 cm [ 1.035,  2.873] = 0.457x [0.323, 0.897]; direct  3.412 cm; model / direct 0.429 [0.385, 0.609]
    XMoveBendPickTeleop free entry    n  28:  0.661 cm [ 0.603,  1.676] = 0.206x [0.188, 0.523]; direct  2.333 cm; model / direct 0.283 [0.264, 0.391]
    XMoveBendPickTeleop free both     n  58:  1.314 cm [ 0.926,  1.812] = 0.410x [0.289, 0.565]; direct  2.664 cm; model / direct 0.493 [0.349, 0.509]
    BendPickMP free      pregrasp n  20:  6.147 cm [ 0.828,  8.083] = 1.919x [0.258, 2.523]  > 1.3; direct  5.107 cm; model / direct 1.204 [0.241, 1.384]
    BendPickMP free      entry    n  13:  1.034 cm [ 1.002,  4.874] = 0.323x [0.313, 1.521]; direct  3.455 cm; model / direct 0.299 [0.290, 0.479]
    BendPickMP free      both     n  33:  4.825 cm [ 1.002,  6.668] = 1.506x [0.313, 2.081]  > 1.3; direct  4.817 cm; model / direct 1.002 [0.290, 1.147]
    all free             pregrasp n  80:  1.259 cm [ 1.045,  1.378] = 0.393x [0.326, 0.430]; direct  4.572 cm; model / direct 0.275 [0.232, 0.299]
    all free             entry    n  71:  0.551 cm [ 0.349,  0.999] = 0.172x [0.109, 0.312]; direct  3.476 cm; model / direct 0.158 [0.101, 0.277]
    all free             both     n 151:  0.999 cm [ 0.864,  1.284] = 0.312x [0.270, 0.401]; direct  4.075 cm; model / direct 0.245 [0.220, 0.300]
```

### 登记的评估：`asrun/rev1/sh_eval.py`（md5 7f342120）→ `eval.txt`

修订 1 的 sh_eval 另存在 `asrun/rev1/`，旁边没有 simplecmp 的目录。所以要用 `PYTHONPATH` 指到冻结副本里 simplecmp 的目录，才能导入 `scmp_eval`。导入的就是冻结副本里那一份，统计函数没变。stderr 是空的。

```
$ PYTHONPATH=$O/asrun/code/scripts/experiments/selfmodel_simplecmp_20261126 $PY $O/asrun/rev1/sh_eval.py --run $O/run \
    --main /workspace/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm --gate $O/gate --f1 $O/f1 \
    --json $O/eval.json > $O/eval.txt
== rows: A 120, AL 120, RD 120, S 120

== 1. consistency
  A = AL: 120 / 120 episodes identical (every row and stage key but wall_s, --correct's fields and the option sonic_vr_frame)
  the look (cloud, score, seen_cm) = the main metric's ScaleBFM A: 91 / 120
  AL: 120 episodes logged, 240 stages, plan_same 240
  RD: 120 episodes logged, 240 stages, plan_same 240

== 2. failures before the arms split (A's rows), per task and class
  XMovePick     : 0 (none)
  XMoveBendPick : 0 (none)
  BendPick      : 0 (none)
  TabletopGrasp : 0 (none)

== 3. success per task
  -- 45 s
  XMovePick     : A 29, AL 29, RD 30 (of 30)
  XMoveBendPick : A 26, AL 26, RD 29 (of 30)
  BendPick      : A 3, AL 3, RD 3 (of 30)
  TabletopGrasp : A 0, AL 0, RD 0 (of 30)
  all           : A 58, AL 58, RD 62 (of 120)
  -- official
  XMovePick     : A 29, AL 29, RD 30 (of 30)
  XMoveBendPick : A 26, AL 26, RD 29 (of 30)
  BendPick      : A 0, AL 0, RD 0 (of 30)
  TabletopGrasp : A 0, AL 0, RD 0 (of 30)
  all           : A 55, AL 55, RD 59 (of 120)

== 4. decision set (pre-split failures over half out; A >= 24 / 30 'ceiling' out)
  45 s     XMovePick     : A 29 / 30 -> out (ceiling)
  45 s     XMoveBendPick : A 26 / 30 -> out (ceiling)
  45 s     BendPick      : A 3 / 30 -> in
  45 s     TabletopGrasp : A 0 / 30 -> in
  official XMovePick     : A 29 / 30 -> out (ceiling)
  official XMoveBendPick : A 26 / 30 -> out (ceiling)
  official BendPick      : A 0 / 30 -> in
  official TabletopGrasp : A 0 / 30 -> in

== gate (sh_gate.sh's rule): XMovePick PASS, XMoveBendPick PASS, BendPick FAIL, TabletopGrasp FAIL

== 5. paired tests (R_D vs A); revision 1: the four tasks together is the main comparison, the gate-passed tasks together beside it, every task with its gate
  main R_D vs A 45 s     all four tasks: 62 / 120 vs 58 / 120 (+3.3 pp [-0.8, +8.3] by layout); layouts + 6 / - 2 of 40, sign test p 0.2891; episodes + 6 / - 2, McNemar p 0.2891
       R_D vs A 45 s     gate-passed ['XMovePick', 'XMoveBendPick']: 59 / 60 vs 55 / 60 (+6.7 pp [+1.7, +13.3] by layout); layouts + 4 / - 0 of 20, sign test p 0.125; episodes + 4 / - 0, McNemar p 0.125
  main R_D vs A official all four tasks: 59 / 120 vs 55 / 120 (+3.3 pp [+0.8, +6.7] by layout); layouts + 4 / - 0 of 40, sign test p 0.125; episodes + 4 / - 0, McNemar p 0.125
       R_D vs A official gate-passed ['XMovePick', 'XMoveBendPick']: 59 / 60 vs 55 / 60 (+6.7 pp [+1.7, +13.3] by layout); layouts + 4 / - 0 of 20, sign test p 0.125; episodes + 4 / - 0, McNemar p 0.125
  P1 R_D vs A 45 s     decision set ['BendPick', 'TabletopGrasp']: 3 / 60 vs 3 / 60 (+0.0 pp [-6.7, +6.7] by layout); layouts + 2 / - 2 of 20, sign test p 1; episodes + 2 / - 2, McNemar p 1
     R_D vs A 45 s     XMovePick      [gate PASS]: 30 / 30 vs 29 / 30 (+3.3 pp [+0.0, +10.0] by layout); layouts + 1 / - 0 of 10, sign test p 1; episodes + 1 / - 0, McNemar p 1
     R_D vs A 45 s     XMoveBendPick  [gate PASS]: 29 / 30 vs 26 / 30 (+10.0 pp [+0.0, +20.0] by layout); layouts + 3 / - 0 of 10, sign test p 0.25; episodes + 3 / - 0, McNemar p 0.25
     R_D vs A 45 s     BendPick       [gate FAIL]: 3 / 30 vs 3 / 30 (+0.0 pp [-13.3, +13.3] by layout); layouts + 2 / - 2 of 10, sign test p 1; episodes + 2 / - 2, McNemar p 1
     R_D vs A 45 s     TabletopGrasp  [gate FAIL]: 0 / 30 vs 0 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
     R_D vs A 45 s     all four      : 62 / 120 vs 58 / 120 (+3.3 pp [-0.8, +7.5] by layout); layouts + 6 / - 2 of 40, sign test p 0.2891; episodes + 6 / - 2, McNemar p 0.2891
     R_D vs A official decision set ['BendPick', 'TabletopGrasp']: 0 / 60 vs 0 / 60 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 20, sign test p 1; episodes + 0 / - 0, McNemar p 1
     R_D vs A official XMovePick      [gate PASS]: 30 / 30 vs 29 / 30 (+3.3 pp [+0.0, +10.0] by layout); layouts + 1 / - 0 of 10, sign test p 1; episodes + 1 / - 0, McNemar p 1
     R_D vs A official XMoveBendPick  [gate PASS]: 29 / 30 vs 26 / 30 (+10.0 pp [+0.0, +20.0] by layout); layouts + 3 / - 0 of 10, sign test p 0.25; episodes + 3 / - 0, McNemar p 0.25
     R_D vs A official BendPick       [gate FAIL]: 0 / 30 vs 0 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
     R_D vs A official TabletopGrasp  [gate FAIL]: 0 / 30 vs 0 / 30 (+0.0 pp [+0.0, +0.0] by layout); layouts + 0 / - 0 of 10, sign test p 1; episodes + 0 / - 0, McNemar p 1
     R_D vs A official all four      : 59 / 120 vs 55 / 120 (+3.3 pp [+0.8, +6.7] by layout); layouts + 4 / - 0 of 40, sign test p 0.125; episodes + 4 / - 0, McNemar p 0.125

== 6. mechanism: each stage's palm miss (cm, median), D_heft's error of the sent command at the stage's end (palm, cm, median), reaches with a table contact (hits)
  XMovePick      pregrasp (n 30): miss A 2.69, RD 0.82 | D_heft's error AL 0.48, RD 0.84 | table contact A 1, RD 0
  XMovePick      entry    (n 30): miss A 1.97, RD 0.28 | D_heft's error AL 0.43, RD 0.28 | table contact A 1, RD 0
  XMoveBendPick  pregrasp (n 30): miss A 1.98, RD 1.51 | D_heft's error AL 1.75, RD 1.44 | table contact A 2, RD 0
  XMoveBendPick  entry    (n 30): miss A 2.33, RD 1.05 | D_heft's error AL 0.96, RD 1.02 | table contact A 13, RD 2
  BendPick       pregrasp (n 30): miss A 3.65, RD 6.53 | D_heft's error AL 3.98, RD 6.46 | table contact A 22, RD 10
  BendPick       entry    (n 30): miss A 7.06, RD 4.79 | D_heft's error AL 3.69, RD 4.79 | table contact A 22, RD 17
  TabletopGrasp  pregrasp (n 30): miss A 34.73, RD 31.90 | D_heft's error AL 28.10, RD 31.87 | table contact A 30, RD 30
  TabletopGrasp  entry    (n 30): miss A 30.67, RD 28.26 | D_heft's error AL 24.12, RD 28.23 | table contact A 30, RD 30
  AL XMovePick      model / direct (13 cm point, both stages): 0.132 [0.116, 0.142]; model 0.45 cm, direct 3.41 cm
  AL XMoveBendPick  model / direct (13 cm point, both stages): 0.386 [0.313, 0.513]; model 1.40 cm, direct 3.63 cm
  AL BendPick       model / direct (13 cm point, both stages): 0.635 [0.421, 0.694]; model 4.37 cm, direct 6.87 cm
  AL TabletopGrasp  model / direct (13 cm point, both stages): 0.771 [0.743, 0.789]; model 31.36 cm, direct 40.66 cm
  AL all            model / direct (13 cm point, both stages): 0.447 [0.230, 0.630]; model 2.49 cm, direct 5.58 cm

== 7. F1 on R_D's commands (AL's for reference): (a) D_heft's error / 3.204 cm > 1.3, (b) F1-0 share < 80 %, (c) the sent commands' IK > 2 cm share >= AL's + 10 pp -> not trustworthy
  AL XMovePick     : (a) 0.141x, (b) 100.0 %, (c) IK > 2 cm   0.0 % -> trustworthy
  RD XMovePick     : (a) 0.159x, (b) 100.0 %, (c) IK > 2 cm   0.0 % -> trustworthy
  AL XMoveBendPick : (a) 0.438x, (b) 100.0 %, (c) IK > 2 cm   0.0 % -> trustworthy
  RD XMoveBendPick : (a) 0.428x, (b) 100.0 %, (c) IK > 2 cm   0.0 % -> trustworthy
  AL BendPick      : (a) 1.363x, (b)  91.7 %, (c) IK > 2 cm   0.0 % -> NOT trustworthy ((a) ratio > 1.3)
  RD BendPick      : (a) 1.870x, (b)  91.7 %, (c) IK > 2 cm   0.0 % -> NOT trustworthy ((a) ratio > 1.3)
  AL TabletopGrasp : (a) 9.786x, (b) 100.0 %, (c) IK > 2 cm   0.0 % -> NOT trustworthy ((a) ratio > 1.3)
  RD TabletopGrasp : (a) 11.785x, (b) 100.0 %, (c) IK > 2 cm   0.0 % -> NOT trustworthy ((a) ratio > 1.3)

== 8. the commands
  RD pregrasp: n 120, sent IK > 2 cm 0; correction median 5.00 cm, max 16.16; shrunk candidates 0 of 0; fallbacks 0; plan_same 120; think median 3.4 s, max 6.9
  RD entry   : n 120, sent IK > 2 cm 0; correction median 2.83 cm, max 19.46; shrunk candidates 0 of 0; fallbacks 0; plan_same 120; think median 1.1 s, max 2.5
```

### HA 有没有被拖慢（阶段 R 跑完后再看一次）

`checks/ha_speed.py` 只读 HA 采集的日志。第一行是阶段 A 开跑前 30 分钟对开跑以后，第二行是阶段 A 期间对阶段 R 以后：

```
$ ha_speed.py 2026-11-26T12:36:26 30
window 30 min before 2026-11-26T12:36:26 vs after (to 13:39:20Z)
  controller Hz  before n 3429 mean  13.49 median  13.46 | after n 7025 mean  13.55 median  13.49
  s per reset    before n 259 mean  66.80 median  62.00 | after n 533 mean  66.23 median  63.00
$ ha_speed.py 2026-11-26T13:15:43 39
window 39 min before 2026-11-26T13:15:43 vs after (to 13:39:20Z)
  controller Hz  before n 4080 mean  13.40 median  13.38 | after n 2368 mean  13.94 median  13.83
  s per reset    before n 307 mean  66.80 median  63.00 | after n 182 mean  65.30 median  63.50
```

- 显存：`checks/mem.log` 每 5 s 记一次，12:38:50Z 到 13:38:47Z。阶段 R 期间 1、2、3 号卡的峰值是 16 402、16 466、16 411 MiB。0 号卡不是我们的，14 614 MiB。
- 读法：控制器频率和每次重置的秒数都没变慢，显存离 20 GB 远。

## 意外

1. **"那一眼相同"里有一项不是那一眼。**
   - 登记把 `cloud`、`score`、`seen_cm` 三项都当成"那一眼"，要求和主指标 ScaleBFM A 栏逐局相同，不同就停下查清。
   - 阶段 A 的结果：`cloud` 和 `score` 120/120 相同，`seen_cm` 有 29 局不同。其中 TabletopGrasp 25 局、BendPick 2 局、XMovePick 和 XMoveBendPick 各 1 局。差的中位数 0.04 cm，最大 0.31 cm（XMoveBendPick 第 11 局）。
   - 查清的结果：`seen_cm` 是缓存观测里物体点云的中位数到开始时物体真实位置的距离（simple_bench.py 第 530 行）。点云是缓存的，两边相同；物体的真实位置是控制器复位、仿真走过的那几步之后的位置，HEFT 和 ScaleBFM 复位不同，物体就停得略有不同。所以它是一项记录，不是机器人看到的东西。机器人的输入（点云、抓取和分数）两边相同。
   - **协调方 11-26 定**：HEFT 的 A 进主指标，作为新的一行。判断依据同上。这是对登记判据的事后放宽，见第 6 条。
2. **HEFT 的 A 比预测好得多。** 预测 45 s 内 0–30/120，实际 58/120。
   - 预测是照开发局两局写的。开发树里只有 TabletopGrasp 有缓存好的那一眼，所以这两局只能是 TabletopGrasp。
   - 测试局的 TabletopGrasp 确实和开发局一样：30 局全失败，60 下伸手都碰到桌子，手掌差约 30 cm。
   - 另外三个任务在开发局上跑不起来，我把 TabletopGrasp 的情况推到了它们身上。实际 XMovePick 29/30、XMoveBendPick 26/30，两个都到顶，不进判定集。
   - 结果是判定集只剩 BendPick 和 TabletopGrasp，P1 几乎没有可比的成功（A 3/60）。
3. **走到站位以后，D_heft 在不碰桌子的伸手上很准。**
   - 预测：F1-0 每个任务 40–85 %，不碰桌子的伸手误差是 val 的 1.0–1.6 倍。
   - 实际：F1-0 91.7–100 %；不碰桌子的伸手（119 下）是 0.21 倍。
   - 一个可能的原因：这两个任务是站着伸手，动作小，而 LML 的 val 里有更难的伸手（看过结果以后的猜测，没验证）。
   - BendPick 不过闸门，主要差在进入那一下（1.38 倍）。它进入时的起点有 13.3 %（A）和 16.7 %（R_D）右髋 roll 超出范围。下蹲伸手在 D_heft 的边缘上。
4. **BendPick 的预抓取，R_D 让手掌偏差变大了**：3.65 → 6.53 cm。进入那一下变小了：7.06 → 4.79 cm。
   - D_heft 在 R_D 的预抓取命令上误差 6.38 cm（1.99 倍），在 A 的命令上是 3.95 cm。修正把命令推到了模型更不准的地方。
   - 成功数没变：3 对 3，布局 +2/−2。
5. **XMoveBendPick 进入时碰桌子的伸手，从 13 下降到 2 下。** 这没预测到。它可能是这个任务多成功 3 局的一部分原因，但没有单独分开。
6. **偏离："A 进主指标"是对登记判据的事后放宽**（复核员指出）。
   - 登记的判据要求 `cloud`、`score`、`seen_cm` 三项都和主指标 ScaleBFM A 栏相同，否则"停下查清；A 不进主指标"。
   - 实际 `seen_cm` 有 29 局不同。放行是协调方看到 eval_A（13:14Z）之后定的，所以是对登记判据的事后放宽，记为偏离。
   - 它不影响 A 对 R_D 的比较：`seen_cm` 在 HEFT 自己的 A、AL、RD 之间 120/120 相同。
7. **偏离：修订 1 的标题时间写错了**（复核员指出）。
   - 标题写的是 12:45Z，实际提交时间是 12:42:10Z（0ef9ae62）。标题没改。
   - 提交时只有 XMovePick 的 A 跑完（12:40:55Z），闸门（13:13:27Z）和 R_D（13:15:43Z 开跑）都还没有。

## 四类陷阱检查

1. **benchmark 选择**：
   - 四个任务、120 局就是主指标的测试局。任务、局号和种子在登记时定，和 ScaleBFM 那一格相同，没有挑。
   - 过闸门的两个任务恰好是 A 已经到顶的两个，能涨的余地本来就小（最多 +1 和 +4）。
   - 复核后补充："+4"这个头条数全部来自登记时判为"到顶、只作描述"的两个任务，结论里写明了。判定集只剩 BendPick 和 TabletopGrasp，P1 在这里几乎测不出东西。
2. **数据泄漏**：
   - D_heft 只用 LML 的数据训练（train3b），没见过 SIMPLE 的任何局。
   - 闸门用的训练范围也是它自己的 LML 训练划分。测试局没有进过训练或 norm stats。
3. **指标误用**：
   - 主比较按修订 1 报：四个任务合计，旁边是过闸门任务的合计。修订 1 在算闸门、跑 R_D 之前就提交了（0ef9ae62）。
   - 登记的 P1（判定集上的布局符号检验）照报，结果 3/60 对 3/60，不成立，没删。
   - 两个时限都报了。45 s 合计的区间含 0，官方时限的不含 0；两个的符号检验都不显著。
   - 复核后补充：官方时限的区间只有 4 个布局不为零，百分位重抽在这里不可靠，以符号检验 p 0.125 为准。
   - 复核后补充：`f1.txt` 里 RD 行的 direct 列量的是"修正后命令的 13 cm 目标到落点"，不是手掌对目标的偏差，不能用来读机制（见结论第 4 条）。
4. **事后挑选**：
   - A、AL、R_D 每组只跑了一次。局、模型、R 规则都在登记时定，没换 checkpoint，没重跑。
   - "意外"第 3–5 条的原因分析是看过结果以后写的，都标了没验证。
   - 复核后补充：开跑后登记部分没改。唯一的事后改动是"那一眼"判据的放宽（"意外"第 6 条）。

## 结论与下一步

**报给协调方的三样数**（更新主指标）：

| | 45 s，四个任务合计 | 官方时限，四个任务合计 | TabletopGrasp（45 s / 官方） |
|---|---|---|---|
| HEFT A | 58/120 | 55/120 | 0/30 / 0/30 |
| HEFT R_D | 62/120 | 59/120 | 0/30 / 0/30 |

按任务，45 s（官方时限）：

| | XMovePick | XMoveBendPick | BendPick | TabletopGrasp |
|---|---|---|---|---|
| A | 29 (29) | 26 (26) | 3 (0) | 0 (0) |
| R_D | 30 (30) | 29 (29) | 3 (0) | 0 (0) |
| 闸门 | 过 | 过 | 不过 | 不过 |
| R_D 上的 F1 | 可信 | 可信 | 不可信 | 不可信 |

**结论**：

1. **HEFT 的 A**：45 s 内 58/120，官方时限内 55/120（`eval.txt` 第 3 节）。
   - 没有在两组分开之前失败的局（第 2 节）。
   - A = AL 120/120。协调方 11-26 定 A 进主指标，见"意外"第 1 条。这是对登记判据的事后放宽，记为偏离（"意外"第 6 条）；它不影响 A 对 R_D 的比较。
2. **闸门 GATE PASS**（`gate.txt` 最后几行）：
   - XMovePick、XMoveBendPick 过。
   - BendPick（F1(a) 1.36 倍）和 TabletopGrasp（9.79 倍）不过。
   - TabletopGrasp 60 下伸手全碰到桌子。D_heft 在不碰桌子的伸手上误差 0.21 倍。
3. **主比较（修订 1）**：R_D 对 A，四个任务合计，R_D 多 4 局，两个时限都不显著（`eval.txt` 第 5 节）。
   - **"R_D 提高了 HEFT 在 SIMPLE 上的成功率"：证据不足，方向一致但没有确立**（复核判定）。
   - 45 s：62 对 58，+3.3 个百分点 [−0.8, +8.3]，布局 +6/−2（共 40 个），符号检验 p 0.29。
     - 复核后补充：同一个比较在第 5 节后面"all four"那行是 [−0.8, +7.5]。上界随重抽的随机流变，下界稳定在 0 以下。
   - 官方时限：59 对 55，+3.3 个百分点 [+0.8, +6.7]，布局 +4/−0，p 0.125。
     - 复核后补充：40 个布局里只有 4 个不为零，百分位重抽在这里不可靠，以符号检验 p 0.125 为准。
   - **+4 全部来自登记时判为"到顶、只作描述"的两个任务**：XMovePick（A 29/30）和 XMoveBendPick（A 26/30）。判定集里的两个任务净 0。
   - 旁边，过闸门两个任务合计：59/60 对 55/60，+6.7 个百分点 [+1.7, +13.3]，布局 +4/−0（共 20 个），p 0.125。两个时限一样。
   - 登记的 P1（判定集 BendPick + TabletopGrasp，45 s）：3/60 对 3/60，p 1，不成立。
   - 对照判据表，落在"P1 不成立"一行：照实报各任务的差和区间，不合并成"赢"。没有任务 R_D − A ≤ −3。
4. **多出来的 4 局在哪儿**：都在闸门和 F1 都判"可信"的两个任务上，XMovePick +1，XMoveBendPick +3，没有一局变差（第 5、7 节）。
   - **"多出的 4 局来自自身模型的修正"：部分支持**（复核判定）。
   - 支持的一面：
     - 两组在分叉前逐位相同，命令改动是唯一的差别。
     - 4 局进入那一下的手掌偏差都降了。
     - 这两个任务的手掌偏差系统性下降（第 6 节，中位数）：
       - XMovePick：预抓取 2.69 → 0.82 cm，进入 1.97 → 0.28 cm；
       - XMoveBendPick：预抓取 1.98 → 1.51 cm，进入 2.33 → 1.05 cm。
   - 保留的一面：
     - **XMovePick 26 是避开了桌子，不是预测准。** D_heft 对 A 这一局的预测是偏 2.69 / 2.29 cm，实际偏 13.0 / 14.8 cm。A 这一局两下都碰了桌子，R_D 把命令挪了约 3.6 cm 以后没碰。
     - **没有对照。** 没有安慰剂改动的组，排除不了"命令的任何改动都可能把边缘局翻过去"。
   - 按判据，过闸门两个任务的成功差可以记到 D_heft 名下，因为闸门和 F1 都判"可信"。但只有 4 个布局不同，双侧符号检验最小也只能到 p 0.125，这个样本撑不起显著。
   - **`f1.txt` 里 RD 行的 direct 列不能用来读机制**（复核员指出）。它是"修正后命令的 13 cm 目标到落点"的距离，不是手掌对目标的偏差。例如 RD XMovePick 的 direct 5.024 cm 大于 AL 的 4.110 cm，但手掌偏差其实是 2.69 → 0.82 cm。读机制要用 `eval.txt` 第 6 节的手掌偏差。
5. **BendPick 和 TabletopGrasp**：模型在这两个任务上不可信，闸门和 F1 都不过，成功数也没变。
   - TabletopGrasp 是手腕碰到桌沿，R_D 以后偏差还是约 30 cm。一次性修正改不了。按闸门一节的理由，采数据也改不了。
   - "修正在 BendPick 有害"：证据不足（复核判定）。净 0：丢的 19、20 两局都是预抓取变差，模型对这两下的误差也变大了；另外多了 27、28 两局。没到判据里 −3 局的线。

**证据强度**：每组 120 局各跑一次，共 40 个布局。A、AL、R_D 同机、同卡、按局配对。过闸门两个任务的 +4 只来自 4 个不一致的布局。

**其他可能的解释**：
- 有的增益来自避开桌子，不是落点对得更准。复核后更正：这一局是 XMovePick 26（见结论第 4 条），XMoveBendPick 的 1、7、20 三局符合"修正降偏差"的读法。
- 这两个任务 A 已经接近全成功，失败的 5 局可能本来就在成败的边缘。任何把手挪近一点的改动，都可能翻过来其中几局。

**下一步**：
- 三样数已报协调方。独立复核已回，原样贴在下一节。
- 复核员提的两个实验，**用户收窄范围，不做**：
  - E1：安慰剂改动加混沌底；
  - E2：检验力足够的复制（约 90 个新布局）。
- 这一格不再往下做：
  - TabletopGrasp 的问题在桌沿，不在模型。
  - BendPick 下蹲伸手的模型误差偏大，要改就得采下蹲的数据，超出"每个控制器跑一次"的范围，等协调方定。

## 独立复核（2026-11-26，独立复核员）

**范围**
- 读了：登记部分（假设到修订 1、预测修订），以及"结果"里的原始输出块。
- 没读："意外"、"结论与下一步"，以及结果里的解读句。
- 怎么算：数字都是自己写脚本，从 4090 上的逐局行、`corr_AL/`、`corr_RD/` 和 D_heft 的四个训练文件重算的。没调用 sh_eval、sh_gate、sh_err、ss_f1pre。
- 怎么跑：只用 CPU，nice 19，≤ 2 线程，没起仿真。远端只读，唯一的例外是比对 scene 时在远端 /tmp 写过两个临时 md5 清单，已删。
- 没复核：HA 有没有被拖慢。

### 1. 登记有没有被改
- **4d022c97（12:35:30Z）→ 0ef9ae62（12:42:10Z）**：
  - 条目只加了"修订 1"一节。
  - sh_eval.py 从 8a53be5c 变成 7f342120，只加了第 5 节的四任务主比较、过闸门合计和闸门标注。
  - sh_gate、sh_err、sh_f1、sh_run、ss_f1pre 三个提交里 md5 都不变。
- **0ef9ae62 → 66df3a9b**：条目增 404 行、删 1 行，删的是"状态"那一行。假设、做法、闸门、判定集、度量、预测、判据、修订 1 都没动。
- **远端文件**：
  - `asrun/rev1/sh_eval.py` 的 md5 7f342120，和 0ef9ae62 相同。
  - 冻结副本里的 sh_* 和 ss_f1pre 和 4d022c97 相同。
  - scmp_correct d80b3ed1、simple_bench 8e40e961 和登记的值相同。simplecmp 目录没进 git，只能对条目里写的 md5。
- **贴的原始输出**：
  - 条目里的 gate.txt、eval.txt 和远端文件逐字相同（diff 过）。
  - md5 都对：gate.txt 3249faa5、f1.txt 9791fc3c、eval.txt 4ac18fe7、eval.json f2eff547。
- **时间线**（run.log）：
  - 修订 1 提交时，只有 XMovePick 的 A 跑完（12:40:55Z）。
  - XMoveBendPick 的 A 12:45:27Z 才开始，闸门 13:13:27Z 才算，R_D 13:15:43Z 才开跑。
  - 所以修订 1 看不到 R_D，也看不到闸门。
- **小处**：修订 1 标题写 12:45Z，实际提交是 12:42:10Z。

### 2. 数字核对（自己重算）
- **行数和标记**：
  - A、AL、RD 各 120 行。局号 = 10 × level + index，没有重复，也没有缺。
  - success_45、success_official 和 t_success 逐局一致。官方时限是 16 / 16 / 8 / 16 s，另一个是 45 s。
- **成功数（45 s / 官方）**：

  | 任务 | A | RD |
  |---|---|---|
  | XMovePick | 29 / 29 | 30 / 30 |
  | XMoveBendPick | 26 / 26 | 29 / 29 |
  | BendPick | 3 / 0 | 3 / 0 |
  | TabletopGrasp | 0 / 0 | 0 / 0 |
  | 合计 | 58 / 55 | 62 / 59 |

  AL 每格都和 A 相同。
- **主比较（四个任务合计）**：
  - 45 s：+3.3 pp，布局 +6/−2 of 40，符号检验 p 0.2891，McNemar +6/−2 p 0.2891。
  - 官方：+3.3 pp，+4/−0，p 0.125。
- **区间**：
  - 我另用一个随机源重抽 4000 次：45 s [−0.8, +7.5]，官方 [+0.8, +6.7]。
  - 作者主比较行的 45 s 是 [−0.8, +8.3]。同一个比较在后面"all four"那行是 [−0.8, +7.5]。上界随随机流的顺序变，下界稳定在 0 以下。
  - 官方区间不含 0。但 40 个布局里只有 4 个非零，百分位重抽在这里不可靠，以符号检验 p 0.125 为准。
- **过闸门两任务合计**：59/60 对 55/60，+4/−0 of 20，p 0.125，两个时限都一样。
- **登记的 P1**（判定集 BendPick + TabletopGrasp，45 s）：
  - 3/60 对 3/60，+2/−2，p 1。
  - 判定集规则按任务重算一致，分叉前失败 0 局。
- **闸门（F1-0）**：
  - 训练划分是我自己重做的（held 带 + crc32 桶 < 8），得 499 077 段，和 train.json 一致。
  - 规则：A 每下的起点 S0 落在训练起点 0.5 / 99.5 % 分位以内。
  - 两下合计，按任务：100.0 %、100.0 %、91.7 %（BendPick 进入 83.3 %）、100.0 %。
- **闸门（F1(a)）**：
  - 算法：AL 的 pred_h13 对我从 npz 读出的落点，用 13 cm 点。没有摔倒的局。
  - 按任务：0.141x、0.438x、1.363x、9.786x。BendPick 预抓取 1.232x，进入 1.384x。
- **闸门判定**：PASS、PASS、FAIL、FAIL，和 gate.txt 逐位一致。BendPick 是边缘，只坏在进入那一下。
- **R_D 命令上的 F1**：
  - (a) 0.158x（作者 0.159x，舍入差）、0.428x、1.870x、11.785x。
  - (b) F1-0 100 %、100 %、91.7 %、100 %。
  - (c) 发出命令里 IK > 2 cm 0/240，AL 也是 0/240。
  - 过闸门的两个任务，三项都判"可信"。
- **A = AL**：120/120。去掉的只有 wall_s、correct_reset、各段的 correct、sonic_vr_frame，不去掉时也只有这几项不同。
- **plan_same**：AL 240/240，RD 240/240。model_fallback 0。
- **那一眼对主指标 ScaleBFM A**：
  - cloud（都是 cloud_blind）和 score：120/120 相同。
  - 加上 seen_cm：91/120。
  - 不同的 29 局：TabletopGrasp 25 局，BendPick 2 局，XMovePick 1 局，XMoveBendPick 1 局。差值中位数 0.04 cm，最大 0.31 cm。
  - seen_cm 在 HEFT 自己的 A、AL、RD 之间 120/120 相同。ScaleBFM 的 simplecmp AL、RD 和主指标 A 也 120/120 相同。
  - 所以差别跟着控制器走。它记的是开局后"cloud 中位数到物体真位置"的距离，不是那一眼的输入变了。
- **seen_cm 和登记判据**：
  - 登记的判据要求 cloud、score、seen_cm 三项都相同，否则"停下查清；A 不进主指标"。
  - 放行是协调方在看到 eval_A（13:14Z）之后定的，属于对登记判据的事后放宽，应在条目里写明是偏离。
  - 它不影响 A 对 R_D 的比较，两组的 seen_cm 相同。
- **scene 文件**：
  - 副本的 120 个 scene xml 和共享树里现在的版本比，31 个逐字相同。另外 89 个只差在部分 mesh 多了 `content_type="model/obj"`。
  - 抽查的 scene_meta.json 只差在 dir 路径，物体位姿相同。
  - 共享树现在的版本是 simple_relook 写的，不是 11-24 主指标那次。所以这只说明场景内容一致，不能直接证明和那次逐字相同。

### 3. 四类陷阱
- **benchmark 选择**：
  - 四个任务、每任务 30 局、局号和种子都和主指标相同，登记时就定了。判定集规则也是登记时定的。
  - 结果判定集只剩两个任务：BendPick 在官方 8 s 内各组都不可能成功；TabletopGrasp 桌沿挡手，模型误差 9.8x。P1 在这里几乎测不出东西。
  - 修订 1 把主比较换成四任务合计，提交在闸门和 R_D 之前，当时只有 XMovePick 的 A（29/30）已经出来。不算按 R_D 结果挑。
  - 但"+4"这个头条数，来自登记时判为"到顶、只作描述"的两个任务。主比较旁边应写明这一点。
- **数据泄漏**：
  - D_heft 训练数据的 reset 号只有两段：20 000 000–20 029 999（data3、mid_data3）和 30 000 000–30 014 999（data3b、mid_data3b）。
  - SIMPLE 记录用的号段是 ≥ 95 000 000（AL 126M、RD 127M），训练数据里一个都没有。
  - d3_collect 的说明写的是 LocoManiLab 的 host（EmptyEnv）。没发现 SIMPLE 的局进了训练。
  - 闸门用的是测试局本身的仿真真值落点，这是登记的设计。它只决定 R_D 跑不跑和"可信"标签，不改 R_D 的命令。修订 1 之后四个任务照跑，它也不决定哪些局进主比较。
  - "过闸门任务合计"这个子集是用测试局真值划的，只能作旁注。
  - R_D 命令的输入：按 scmp_correct.py 的说明，是关节编码器、IMU、里程计信念和那一眼，仿真骨盆只当坐标原点和高度。我只读了说明和取坐标系的几行，没有逐行审。
- **度量误用**：
  - F1(a) 的分子分母同口径：都是 13 cm 点，分母是 D_heft 自己的 LML val 3.204 cm。
  - f1.txt 里 RD 行的 "direct" 是"修正后命令的 13 cm 目标到落点"的距离，不是手掌对目标的偏差。
  - 所以 RD XMovePick 的 direct 5.024 cm 大于 AL 的 4.110 cm，不代表变差。它的手掌偏差是 2.69 → 0.82 cm。读机制不能用这一列。
  - 手掌偏差是对"那一眼给出的目标"量的，是命令精度，和登记的度量一致。
- **事后挑选**：
  - 开跑后登记部分没改。唯一的事后改动是"那一眼"判据的放宽。
  - 过闸门的两个任务里，A 只在 4 个布局有失败（XMovePick 6；XMoveBendPick 0、1、7）。所以 +4/−0、p 0.125 已经是这个子集能达到的最小 p。
  - 这不是挑选，是检验力不够，只能看方向。

### 4. "多出的 4 局来自修正"站不站得住
- **分叉前是否逐位相同**：是。
  - A 和 RD 的 cloud、score、seen_cm、walk、walk_err_cm、odo、bel、grasp_h、pz、pre_dir、obj_from_grasp、bench 都 120/120 相同。
  - 预抓取的 S0 120/120 相同，两下的 goal_mm 120/120 相同。
  - 预抓取下命令的帧 f_cmd，AL 和 RD 120/120 相同。
  - trace 前 15–18 条相同，第一处不同是预抓取阶段的手位置。
  - 仿真是确定的（A = AL 120/120）。所以多出的局一定来自"命令变了"。
- **多出的 4 局的手掌偏差**（45 s 和官方是同样 4 局；预抓取 / 进入，A → RD，cm）：

  | 局 | 预抓取 | 进入 |
  |---|---|---|
  | XMovePick 26 | 13.017 → 0.629 | 14.845 → 0.542 |
  | XMoveBendPick 1 | 3.84 → 4.199 | 5.042 → 2.246 |
  | XMoveBendPick 7 | 2.876 → 6.147 | 4.083 → 2.74 |
  | XMoveBendPick 20 | 2.322 → 1.813 | 3.209 → 1.603 |

  进入那一下 4 局都降了；预抓取 2 局降、2 局升。
- **XMovePick 26 不是"模型预见了失败"**：
  - D_heft 对 A 这一局命令的预测是偏 2.69 / 2.29 cm，实际偏 13.0 / 14.8 cm，预测误差 11.5 / 12.9 cm。
  - A 这一局两下都碰了桌子（right_hand_middle_1_link），RD 没碰。
  - 增益来自命令挪了约 3.6 cm 以后没碰桌子，不是对这一局的准确预测。
- **XMoveBendPick 1、7、20 符合"修正降偏差"的读法**：模型对 A 进入那一下的预测偏差是 3.67、4.27、2.04 cm，实际是 5.04、4.08、3.21 cm。
- **这几局是边缘局**：
  - A 的成败不是手掌偏差的干净阈值。XMoveBendPick 里 A 成功的局进入偏差最大 4.37 cm，失败的局最小 3.21 cm。
  - 命令任何改动都可能把边缘局翻过去。没有对照组，排除不了这一点。
- **过闸门任务里，偏差确实系统性下降**：
  - XMovePick 30 局两下都降 > 1 cm，没有一局升 > 1 cm。中位数预抓取 2.69 → 0.82，进入 1.97 → 0.28。
  - XMoveBendPick 进入 22 局降 > 1 cm、0 局升，中位数 2.33 → 1.05。预抓取 1 局降、1 局升，中位数 1.98 → 1.51。
  - XMoveBendPick 进入碰桌子的伸手从 13 降到 2。
- **变差的局**：
  - 过闸门的两个任务里，没有一局从成功变成失败。
  - 45 s 下 BendPick（闸门没过）丢了 2 局（19、20）。都是预抓取变差：4.212 → 6.619，4.135 → 7.401。模型对 RD 这两下预抓取的误差是 6.69、7.45 cm，对 A 的是 2.20、1.84 cm。
  - 同一任务又多了 2 局（27、28），进入 8.423 → 2.951，7.989 → 0.793。净 0。
  - BendPick 预抓取 16/30 局升 > 1 cm，中位数 3.65 → 6.53；进入 21/30 局降 > 1 cm。
  - 没到判据里"≤ −3 局"的线。
- **读法**：
  - 过闸门的两个任务上，"多出的局是命令改动带来的"成立。
  - "是因为模型预测准"，4 局里 3 局成立，XMovePick 26 不成立。
  - "不是混沌翻转"，没有对照，证据不足。

### 5. 可能推翻结论的实验
- **E1 安慰剂改动加混沌底**：
  - 范围：XMovePick、XMoveBendPick 共 60 局。
  - P 组：每一下把 A 的命令挪一个向量。长度等于 R_D 这一下的修正量（corr_cm），方向在水平面随机，种子固定、和模型无关。规划和收缩到 IK ≤ 2 cm 都照 R_D。
  - A_ε 组：A 的命令随机挪 1–3 mm，跑 3 个种子，量混沌翻转的底。
  - 推翻：P 救回 A 的 5 局失败里 ≥ 3 局，或 P 净翻转 ≥ +3，或 A_ε 平均每 60 局翻 ≥ 2 局。这时 +4 和"任何改动"分不开。
  - 维持：P 净 ≤ +1 且手掌偏差不降，同时 A_ε 每 60 局翻转 ≤ 1 局。
- **E2 检验力足够的复制**：
  - 在这 40 个布局以外找新布局，先做那一眼，XMovePick、XMoveBendPick 各跑 A 和 R_D。
  - A 的失败率约 8 %。要有 ≥ 6 个不一致的布局才可能 p < 0.05，大约要 90 个新布局。
  - 预注册判据：布局符号检验 p < 0.05 且净 ≥ +6 布局，支持；净 ≤ 0，推翻"R_D 提高 HEFT 的成功率"。

### 6. 判定
- **A 45 s 58/120、官方 55/120（含分任务）**：支持。逐局重算一致。
- **A 进主指标**：支持，附条件。cloud、score 120/120 相同；seen_cm 有 29 局不同，最大 0.31 cm，跟着控制器走。但这是对登记判据的事后放宽，要写明是偏离。
- **GATE PASS（XMovePick、XMoveBendPick）**：支持。两项按任务的值逐位复现；BendPick 的 FAIL 是边缘（1.363x，只坏在进入）。
- **R_D 62/120、59/120（含分任务）**：支持。
- **R_D 提高了 HEFT 在 SIMPLE 上的成功率**：证据不足。
  - 主比较 45 s p 0.2891，官方 p 0.125。
  - 登记的 P1 是 3/60 对 3/60。
  - 过闸门子集 +4/−0、p 0.125，已是这个子集能到的最小 p。
  - 方向一致，但没有确立。
- **多出的 4 局全在过闸门的两个任务（59/60 对 55/60）**：支持（计数层面）。45 s 全体是 +6/−2，其中 BendPick 的 +2/−2 互相抵消。
- **多出的 4 局来自自身模型的修正**：部分支持。
  - 支持的一面：分叉前逐位相同，命令改动是唯一差别；4 局进入偏差都降了；过闸门任务的偏差系统性下降。
  - 保留的一面：XMovePick 26 是避开了桌子，不是预测准；没有安慰剂对照，排除不了边缘局翻转。
- **过闸门任务的成功差可以记到模型名下**：按判据规则支持，两个任务的闸门和 F1 都判"可信"。数量上仍受上一条限制。
- **修正在 BendPick 有害**：证据不足。净 0；丢的 2 局和预抓取变差、模型误差变大对得上，但没到 −3 的判据线。
