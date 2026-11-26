# selfmodel_simpleheft_20261126：SIMPLE 主指标加 HEFT（直接执行 A，闸门，再一次性修正 R_D）

- **状态**：预注册（2026-11-26）。先跑 A 和 AL，算闸门；过了闸门，当天跑 R_D。
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

## 结果（贴原始输出，不转述）

## 意外

## 结论与下一步

## 独立复核
