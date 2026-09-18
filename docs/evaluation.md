# 评测:open-loop 与 closed-loop

> 两套评测回答两个不同的问题,谁也替代不了谁。训练在
> [training.md](training.md)。

## 1. 任务与能力

| 项目 | 当前支持 |
|---|---|
| open-loop | `eval_open_loop.py`(自研 nMSE 套件)+ `gr00t/eval/open_loop_eval.py`(官方朴素 MSE),都只读 checkpoint + val 数据集,不起 sim,用来逐 checkpoint 选型 |
| closed-loop(单臂) | `scripts/eval_bottle.py` 自动起 token-only 栈 + policy server,跑够 `eval.rounds` 局,是唯一的验收标准 |
| closed-loop(双臂) | `eval_crate_closed_loop.py`,由 `scripts/eval_crate.py` 对 `task.plugin=crate` 自动路由到它(见 §6) |
| 统计工具 | Wilson 95% 区间(单臂样本量)、McNemar 配对检验(两配置比较)——`src/locomani/pipeline/seq_mcnemar.py` |

本仓库已有的硬证据:把**真值** token 开环回放,关节轨迹复现到 0.2–0.9°,机器人
却抓不到瓶子(浮动基座 1 分钟积分出 0.77 m 漂移)。所以:

- **open-loop** 回答"模型学没学到东西"——快、可逐 checkpoint 跑、用来选 checkpoint;
- **closed-loop** 回答"这个策略行不行"——唯一的验收标准。

bottle200 第一轮把这条原则演成了实例:开环关节角差 1.05°(贴测量噪声底),
闭环 6%。两个数字都对,测的不是一件事。

## 2. 环境与场景

open-loop 评测脚本(`src/locomani/pipeline/eval_open_loop.py`、`scripts/gr00t_open_loop.sh`
包的 `src/gr00t/eval/open_loop_eval.py`)只需要 checkpoint 和 val 数据集,跑在
**GR00T venv** 下(`common.TRAIN_PY = third_party/Isaac-GR00T/.venv/bin/python`),不起
sim、不占训练/评测之外的资源,可以和别的评测同时跑在别的卡上;它们不经过
`scripts/`,不是流水线的一个阶段,挑哪个 checkpoint 是手动决策。

closed-loop 评测(`eval_closed_loop.py`、`eval_crate_closed_loop.py`)要起完整的
token-only 栈(sim + deploy)、一个独立的 policy server 进程,还要用 gear_sonic
的相机客户端和 pinocchio FK 构造观测,所以跑在 **sim venv** 下
(`common.SIM_PY = $VENV_SIM_PY`)。经 `scripts/eval_bottle.py` 跑时,
这一切由 `scripts/eval_bottle.py` 自动管理(起栈、起 policy server、
`try/finally` 保证跑完拆栈);手动跑(见 §3)则要自己按顺序起停。

评测用到的配置组是 `configs/eval/*.yaml`(`EvalConfig`,字段 `rounds`/`seed`/
`client`/`budget_s`/`replan_interval`/`execution_horizon`/`ensemble_n`/
`ensemble_m`/`policy_port`/`spawn_edge`/`task`/`target`,外加一个 `backend:` 子块),
随运行目录的 `config.yaml` 一起落盘;实际交给评测器的那份写在 `eval/eval_config.json`。bottle 系
任务默认 policy 端口 5590,crate 默认 5700(避开单臂评测占用的实例端口块,见 §6)。

## 3. 跑一局(+读日志)

### open-loop:选 checkpoint 用

```bash
# 自研 nMSE 套件
third_party/Isaac-GR00T/.venv/bin/python src/locomani/pipeline/eval_open_loop.py \
    --model-path outputs/runs/<run>/model/checkpoint-N \
    --dataset outputs/runs/<run>/val_dataset \
    --out outputs/eval/<run>/open_loop_checkpoint-N.json \
    --wandb-run <run>          # val 曲线写回训练 run,同一根 x 轴

# 官方对照(排序用,不是数值对照,见 §5)
bash scripts/gr00t_open_loop.sh \
    --model-path outputs/runs/<run>/model/checkpoint-N \
    --dataset-path outputs/runs/<run>/val_dataset \
    --out-dir outputs/eval/<run>/checkpoint-N \
    --backbone-path ~/Checkpoints/Cosmos-Reason2-2B   # 仅 checkpoint 在别的机器上训练时需要,见下
```

**`--backbone-path`(别处训练的 checkpoint)**:checkpoint 的 `config.json` 把训练时用的
VLM backbone(Cosmos-Reason2-2B)位置原样记在 `model_name` 里——在线训练记的是 Hub id
`nvidia/Cosmos-Reason2-2B`,从本地副本训练记的是那台机器上的目录(集群上是
`/group-volume/.../Cosmos-Reason2-2B`)。拿到本机评测时这个目录不存在,加载会在解析完
checkpoint 配置后失败(报 `records its VLM backbone at ..., which does not exist here`);
`--backbone-path <本地 Cosmos-Reason2-2B 目录>` 把模型和 processor 的这条记录一起替换掉。
有这个参数的入口:`serve_policy.py`、`gr00t/eval/open_loop_eval.py`、
`scripts/gr00t_open_loop.sh`(也接受环境变量 `BACKBONE_PATH=`)和
`eval_crate_closed_loop.py`(透传给它起的 policy server);自研 `eval_open_loop.py` 和
`scripts/eval_bottle.py`(起 policy server 时不透传它)没有,单臂线要用它就绕开脚本走下面的
手动三步。本机训练的 checkpoint 不需要它:`model_name` 只要含 `Cosmos-Reason2`/`Qwen3-VL`
(本地目录路径也行,不再要求 `nvidia/` 前缀)就按原记录加载。传进来的目录必须存在且含
`config.json`,否则在加载模型前就报错。

跨 `execution_horizon ∈ {1,8,16,40}` 各报一遍——误差随 horizon 的增长曲线就是
"一个 chunk 敢信多少步"的直接证据(选 checkpoint 只跑 h16 即可,全 horizon 留给
选出的最优者)。

### closed-loop:跑一局(即一次完整的 N 轮评测)

```bash
export PYTHONPATH=src
python scripts/eval_bottle.py outputs/runs/<run_name> --gpu 1
```

前提是同一个运行目录下已有 `train/manifest.json`(先跑过 collect/convert/train,
见 [训练](training.md))。`eval` 阶段自己起栈、起 policy server、调用
`eval_closed_loop.py`(bottle 系任务)或 `eval_crate_closed_loop.py`(crate,见 §6),
跑完回收所有子进程,把 `successes`/`success_rate` 记进
`outputs/runs/<run_name>/eval/manifest.json`。**起跑前确认 §5 的初始 token 文件与
这次训练数据的开局姿势一致**,配错 100 局会全部卡在 `[stand]` 步骤而不是模型的错。

需要绕开 stage、手动控制每个环节时(调试单个进程、临时传一个 stage 不透传的参数),
仍然是原来那三步,分别在各自的 venv 下起:

```bash
# 1. stack,必须 token-only(带 encoder 时外部 token 流压不住,cpp:3096)
python -m locomani.stack.cli up --instance 0 --gpu 0 --prompt "pick up the water bottle" --no-encoder

# 2. policy server,第二张卡(别处训练的 checkpoint 加 --backbone-path,见上)
CUDA_VISIBLE_DEVICES=1 third_party/Isaac-GR00T/.venv/bin/python \
    src/locomani/pipeline/serve_policy.py --model-path <ckpt> --port 5590 \
    [--backbone-path ~/Checkpoints/Cosmos-Reason2-2B]

# 3. 评测本体(sim venv —— 它要 gear_sonic 的相机客户端和 pinocchio FK)
"$VENV_SIM_PY" src/locomani/pipeline/eval_closed_loop.py \
    --set rounds=100 --set seed=6000 --set policy_port=5590 --set backend.dataset_fps=25 \
    --reach-config outputs/runs/<run>/reach_config.json \
    --initial-token-file configs/deployment/ready_grasp_initial_token.json \
    --out outputs/eval/<run>/closed_loop.json --trace-dir outputs/eval/<run>/traces
```

参数是 `EvalConfig` 的字段(`--config <json>` 整份替换,`--set key=value` 逐个改),
和 `eval` stage 内部拼的命令一一对应——这条正是 `eval_stage._serve`/`run()` 自动
拼出来的那条,唯一的区别是这里手动指定了 policy 端口和初始 token 文件。

**读日志**:`<out>/eval.log` 是评测本体(`eval_closed_loop.py`/
`eval_crate_closed_loop.py`)的标准输出;`<out>/eval/serve.log` 是 policy server
的(启动成功看有没有 "listening" 字样——`eval_stage._serve` 轮询的就是这一行,
600 s 超时视为失败;`eval_crate_closed_loop.py` 起 server 时设了 `PYTHONUNBUFFERED=1`,
这一行即时落盘,否则会留在子进程缓冲区里、健康的 server 被误判为没起来);栈的启停命令记在 `<out>/commands.jsonl`。每一局的详细记录
在 `<out>/eval/traces/`(位姿、失败阶段、`stall_abort_s`/`full_budget` 等字段,
见 §5、§6),视频在 `<out>/eval/videos/`。`eval/manifest.json` 的 `ok=true` 加
`metrics.success_rate` 才是一次跑完的评测;进程被中断(见 §7)不会写 manifest,
下次跑等于从头再来。

## 4. 批量(+验收)

```bash
python scripts/pipeline.py configs/experiments/A.yaml --gpu 0    # 终端 1
python scripts/pipeline.py configs/experiments/B.yaml --gpu 1    # 终端 2
```

两份实验文件各跑一条线,产物已存在的阶段自动跳过(要重跑就删掉那个产物)。
跑完 `cat outputs/runs/*/summary.json` 并排看结果。

`pipeline.py` 把闭环评测接在训练后面自动跑;补采或重训之后,删掉哪个阶段的产物就只重跑哪一段
(`--from eval` 只跑评测)。要比较两次运行的成功率和失败构成,读各自的
`summary.json` 与 `eval/closed_loop.json`。

### 加速评测(A+C+B 筛选 harness)

一场 100 轮全预算闭环 ≈ 3.1 小时(失败轮 ~138 s = 90 s 预算烧满 + ~48 s 出生/
站立/走位开销),两臂 A/B ≈ 6 小时。三个正交的杠杆把筛选压到 ~1 小时:

- **A 并行实例**:两臂各占一个 stack(`INSTANCE=0/1`,DDS domain + 端口块隔离),
  policy server 各占一张 GPU,端口 **6590/6591**——不是 5590,因为 INSTANCE=1 的
  keyboard 端口恰好是 5590。
- **B 筛选预算**:`eval_closed_loop.py --budget 60`(梯子是 90)。偏差两臂共享
  且**永远朝下**(round 4 实测 7/48 的成功落在 60 s 之后)。结果 JSON 记
  `budget_s`,`full_budget` 只在 90 s 且无 stall-abort 时为 true。
- **C 序贯停止**:按轮配对(同 seed → 同初始条件;`appearance` 不一致直接拒测),
  每 20 配对轮一次 look:interim p<0.01 判 DIFFERENT,final(n=100)p<0.05,
  **追不上显著性就 FUTILE 提前收**。Pocock 风格的筛选阈值,不是精确的 STEP。

**筛选出的赢家必须再过一次 90 s 全预算确认才算数**(round 4 的教训:50 轮 A/B
报出的 +12 点在全预算配对下是 p=0.755 的噪声)。FUTILE 则是真实的"不值得再烧
6 小时"。

编排这套流程(两个并行 stack + 按轮做序贯 look)的脚本 `scripts/seq_ab.sh` 已随
本轮重构下线,配对/序贯判定逻辑本身还在 `src/locomani/pipeline/seq_mcnemar.py`,
但目前没有替代的一键驱动脚本。等价效果目前要拼:复制两份实验文件(一份加 `eval.ensemble_n=3`/`eval.ensemble_m=0.3`,
一份不加,两份都把 `eval.budget_s` 调小),各在一张卡上 `python scripts/eval_bottle.py <run> --gpu N`,
跑出两份 `closed_loop.json` 与 `traces/`,再手动喂给 `seq_mcnemar.py` 做配对判定。下面
几节里具体的历史命令(`seq_ab.sh` 的选项)因此不再能直接照抄执行,
保留仅供理解方法论与复核历史数字。

已用 2 轮 smoke 端到端验证过这套机制:双 stack 共存、按 appearance 配对、序贯
look、判定后自动清场。

逐日实验记录(2026-08-27 起的并行罚项、配对规则、观测流审计、stand-start 漂移定位、
干净全任务基线)已移入 [单臂台账](ledger/bottle.md)。

### 2026-08-28:闭环默认改为 20 局

一场 100 局全预算闭环是 4.5 小时,探索阶段付不起。**`--fresh` 的默认值改为 20**
(`eval_closed_loop.py`、`seq_mcnemar.py` 的 `N_MAX` 同时改),但必须知道 20 局
买到的是什么、买不到什么。

**买不到:一个能引用的成功率。** 20 局的 Wilson 95% 区间有 30-40 个点宽:

| 观测 | 率 | Wilson 95% | 宽度 |
|---|---|---|---|
| 2/20 | 10% | 3-30% | 27 点 |
| 4/20 | 20% | 8-42% | 34 点 |
| 6/20 | 30% | 15-52% | 37 点 |
| 10/20 | 50% | 30-70% | 40 点 |
| (对照)12/100 | 12% | 7-20% | 13 点 |

也就是说 **20 局测出 20% 和 40%,区间是重叠的**——单独一个 20 局百分数不构成
结论,写进任何文档之前都要用 100 局重测。

**买得到:大效应的配对判定。** 与同 seed 的另一场逐局配对时,20 对能判定的是:

    一边赢 6 对、另一边 0 对   p=0.031  显著
    一边赢 8 对、另一边 1 对   p=0.039  显著
    一边赢 7 对、另一边 1 对   p=0.070  不显著
    一边赢 9 对、另一边 2 对   p=0.065  不显著

大约是"翻倍级"的效应才判得出来。round3→round4 那种量级(+29/−9)在 20 局里也会
显著;而"不吸附"那种(+22/−19)在 100 局都判不出来,20 局更不可能——后者恰恰
是最需要警惕的那类,所以**20 局出的"没差别"不是"没效果",只是"这个尺度看不见"**。

**用法约定**:

- 探索、筛选、排除明显错误 → 20 局(默认)
- 任何要进梯子、进文档、进论文的数字 → `--fresh 100`,90 s 全预算
- 20 局说"有差别" → 值得花 100 局确认;20 局说"没差别" → 只能得出"不是大效应"

## 5. 产物与验收

### open-loop:自研 nMSE(`eval_open_loop.py`)

全部指标都配 persistence 基线(块首真值动作保持整块——数据 30.6% 的帧 token
与前帧相同,不摆这条线任何 MSE 都读不懂):

| 指标 | 定义 | 体现的性能 |
|---|---|---|
| `nmse_token` | 对 snap(量化吸附)后的 64 维 token 预测算 nMSE = MSE/var(gt),逐维归一化后再平均 | token(身体动作)预测的整体准确度,1.0 = 不如预测均值 |
| `nmse_token_unsnapped` | 同上但用 snap 前的原始连续预测 | 衡量量化吸附本身修正了多少误差 |
| `nmse_gripper` | 单独对 gripper 1 维算 nMSE | 抓取相关的关键维度单独评估,避免被 64 维 token 稀释 |
| `hit_rate_per_dim` | 预测(snap 后)token 索引与真值 token 索引相同的逐维命中率均值 | FSQ 空间每维实际只用 2–5 个格点,离散空间里命中率可读、MSE 不可读 |
| `hit_rate_all64` | 要求同一帧 64 维**全部**命中的比例 | 更严格的整体正确率 |
| `gripper_mae` | 预测与真值 gripper 值的平均绝对误差 | gripper 连续值(开合程度)的平均绝对误差 |
| `close_frame_gt` / `close_frame_pred` | 真值/预测 gripper 信号第一次穿过(真值的)5%–95% 分位数中点的帧号 | 夹爪开始闭合的时间点,两侧都用真值量程判定 |
| `close_frame_err`(及 `close_frame_err_abs`) | 预测闭合帧号 − 真值闭合帧号 | 抓取时机的提前/滞后帧数——任何 MSE 都看不见它 |
| **`deg_mean` / `deg_p95` / `deg_max`** | 真值/预测 token 分别喂 SONIC decoder(观测固定),取关节指令差的均值/95 分位/最大值,单位度 | **头条数字**,有物理单位 |
| `degrees_noise_floor` | decoder 对真值 token 的重建输出与实际记录关节指令的误差均值 | 误差最低可信线(~1.1°),低于它的差异不作数 |
| persistence 基线(每个指标各一份) | 块首真值动作原样冻结、复制到区间所有帧 | 判断模型是否只学到了数据的强自相关而非真实动力学 |

### open-loop:官方(`gr00t/eval/open_loop_eval.py`)

跑的是 NVIDIA 自己的脚本,不是上面这套自研 nMSE。只有两个原始(未归一化)标量
指标,64 维 token + 1 维 gripper 一起求平均,不拆分、不归一化:

| 指标 | 定义 |
|---|---|
| MSE / MAE(单轨迹) | 对整条轨迹所有时间步、所有动作维度一起求均值/均绝对值 |
| 平均 MSE / MAE(跨轨迹) | 每条轨迹的值再取算术平均 |
| 轨迹曲线图 | 每维一张 subplot(gt/pred/state joints 三条线 + 重规划红点),不产出可比较的数字 |

每张图对应一条轨迹(65 个纵向子图:64 token + 1 gripper);官方脚本对每条轨迹的
图用同一个文件路径,多条轨迹会互相覆盖,所以脚本跑完统一把图挪到
`<out-dir>/plots/`。产出:每条轨迹一张对比图、`<out-dir>/metrics.json`、
`<out-dir>/run.log`。`--model-path`/`--dataset-path` 传反会在加载模型前就报错;
checkpoint 记录的 backbone 目录在本机不存在时也一样(`records its VLM backbone at ...,
which does not exist here`),按 §3 加 `--backbone-path`。

**两套结果互相印证**(`scripts/crosscheck_open_loop.sh`,10 个 checkpoint,
Spearman rho +0.988,选出的最优 checkpoint 一致)——说明自研这一层(轨迹迭代、
execution-horizon 索引、指标算术)与官方实现一致。踩过一次坑:**官方 eval 的
`--steps` 默认 200,会静默截断**,val episode 长 546–832 帧、夹爪闭合在第 327 帧,
默认参数只评了走路段。改成按最长 episode 推导(832)后 rho 才从 +0.952 升到
+0.988。这个默认值对短 demo 合适,对长 episode 是静默截断。

<!-- ### bottle200 基线(checkpoint-10000,20 局 val)

| horizon | nMSE token(persistence) | 关节角差(persist.) | 闭合误差(persist.) |
|---|---|---|---|
| 1 | 0.120 | 1.05°(—) | 1.4 帧 |
| 16 | 0.142(0.323) | 0.93° | ~3 帧 |
| 40 | 0.176(0.895) | 1.30°(4.21°) | 4.9 帧(23.4) |

误差随 horizon 增长很缓,整块 1.33 s 可信;关节角差已到噪声底(1.16°)。 -->

### closed-loop:初始 token 文件

**初始 token 文件必须与训练数据的开局姿势一致。** 采集阶段(`scripts/collect_bottle.py`)
的初始 token 文件默认是 `configs/deployment/ready_grasp_initial_token.json`
(READY 抬臂姿势的裸 64 维 token,prelude 之后按 `ready_q()` 的手臂检查)。用
`bottle_whole_body` 采的数据从 SONIC 自己的垂臂参考开始(2026-09-14 起),对应
文件是 `configs/deployment/whole_body_initial_token.json`:190 集训练数据首帧
token 的量化均值,外加 `arms` 字段(首帧实测 14 个手臂关节,MuJoCo 顺序),
prelude 改按它检查。拿错文件的症状是 100 局全部 `[stand] token-only ready
posture not delivered`(2026-09-14 实测 10/10)。带 `arms` 的 dict 形式和裸列表
形式都能被 `initial_pose.load_token_file` 读取。选哪一份由 **`eval.initial_token`** 决定
(2026-09-17 起;此前 `eval` 阶段硬编码 `ready_grasp_initial_token.json`,不按任务切换)。
瓶子线的 `configs/eval/closed_loop_100.yaml` 已指向 `whole_body_initial_token.json`:
`bottle_simple` 的 200 集数据首帧手臂姿态与该文件逐关节相差 ≤0.3°。

### closed-loop:输出与验收

`outputs/runs/<name>/eval/` 下:`closed_loop.json`(`rounds`/`successes`/
`success_rate`/`wilson95`/`stand_retries_used`/失败阶段分布)、`traces/`(逐局位姿与失败签名)、
`videos/`、`serve.log`、`eval.log`。**验收只看 `success_rate` 配它的 Wilson
区间**——20 局能筛选方向,不能当最终数字(见 §4);任何要写进文档/论文的数字都
要 100 局全预算跑出。

**`[stand]` 失败就是这一局失败,不重试**(`stand_retries` 自 2026-09-17 起默认 0)。
采集侧一直就是这么算的(`task.stand_retries` 处处为 0),反倒是要出数字的评测侧此前宽松。
重试也分不清"重置后没站稳"和"走近站位本来就不对"——两者都以 `[stand]` 到达——
所以它会把策略自己造成的失败悄悄记在栈头上。只有诊断栈不稳时才临时设成 1,不要用它出结果。

设成 1 时,`closed_loop.manifest.jsonl` 仍是**每次尝试一行**而 `closed_loop.json` 是
**每回合一条**:重试跑的是**同一个 case**(相同 case_id、出生点、物体位姿和外观种子),
是同一个样本不是两局,所以回合的结果取它结束时的那次尝试,重试次数单独记在
`stand_retries_used`。2026-09-17 之前的读法直接用 manifest 行数当分母,于是分母普遍翻倍
(`eval_baseline43_noencoder` 的 49 行其实是 25 个回合),而且一局在重试后成功会被
报成 1/2 = 50%。当时的历史评测成功数都是 0,`success_rate` 0.0 未受影响,受影响的是
分母和 Wilson 区间。

> 双臂评测器 `eval_crate_closed_loop.py` **不读** `stand_retries`,也不写 `attempt`;
> 它 manifest 里的 `round` 是批内序号(0-9 循环,唯一标识是 `seed`),所以那边
> 一行就是一局,`len(rows)` 是对的。不要把单臂的按 `round` 归并搬过去——
> `crate_eval100` 的 100 局会被压成 10 局。

## 6. 调整任务(控制逻辑+判据表)

本节只留控制逻辑与设计结论。各版本的逐轮成功率、部署参数普查、DAgger 裁定、
失败细分表都在 [单臂台账](ledger/bottle.md) 里。

### 部署参数(闭环执行时的控制逻辑)

动作下发相关的几个参数互相有隐式生效前提(完整 help 文本见
`eval_closed_loop.py:86-124`)。**这张表的默认值就是代码里的默认值**——两者
分家过一次(表写 `--execution-horizon` 8 而代码是 16;表写 `--ensemble` 1 而
代码被改成了 3),所以改任何一个默认值时,这张表要一起改:

| 参数 | 默认值 | 生效条件 | 作用 |
|---|---|---|---|
| `--client` | `async` | 始终生效,决定下面三个参数哪些真正起作用 | `async` = 官方部署同款流式(worker 线程持续推理,主循环不间断发布,延迟补偿索引接入);`blocking` = execute-N-then-replan,round-1/2 基线,仅作 A/B 对照保留 |
| `--execution-horizon` | `16` | 仅 `--client blocking` | 一个 chunk 执行几帧再重新请求推理;`async` 下解析但不使用——传了不影响行为,只是原样记进结果 JSON,容易误读 |
| `--replan-interval` | `0.4`(秒) | 仅 `--client async` | worker 多久请求一次新 chunk,不是发布频率(发布固定 `DATASET_FPS`)。`0.4` 是梯子(5.9/13.7/28.0/47.5)的设置;`1.33`(=40/30)等于一整个 chunk 播完才重规划,chunk 间重叠归零 |
| `--ensemble` | `1`(关) | 仅 `--client async`;实际重叠的 chunk 数还受 `--replan-interval` 制约 | 混合最近 N 个仍覆盖"现在"的 chunk 预测(ACT 式时序集成)。**结论是 checkpoint 条件化的,换 checkpoint 必须复筛**(见下文历次判定) |
| `--ensemble-m` | `0.3` | 仅 `--ensemble>1` 且 `--client async` | 集成权重按 chunk 年龄衰减:`m>0` 偏新(默认)、`m=0` 均匀、`m<0` 偏旧 |

设计要点,每条都有伤疤背书:

- **复用 grasp_gym 自己的采样器**(`VLA_POLICY` 钩子,fork 点在 settle 之后)——
  参照线是脚本专家在同分布上的 **200/210 = 95.2%**,自己另写采样器 = 悄悄换分布,
  对照作废。
- **观测用 exporter 同款代码路径构建**(同一个 FK、同一个 gravity、图像原样透传),
  与训练数据构造上一致。
- VLA 局**出生即朝向 approach heading、跳过脚本转向**:token-only 模式下没收到
  token 时 `token_state` 置零,站立撑得住、主动转向是坏的;而训练 episode 本来
  就从转向完成后开始录。
- eval 会 `export NO_ENCODER=1`:gym 每局重启 deploy(`locomani.stack.cli
  restart-deploy`),不传下去的话第 2 局就被换回带 encoder 的 deploy。
- 结果 = 成功率 + Wilson 95% 区间 + **分阶段失败分布**(walk / reach / close /
  fell / timeout)。分阶段才可行动:失败在 close = 忠实继承了数据的缺陷;失败在
  walk = 没学会。
- 每局 trace 落盘(含停滞位姿),它就是下一轮靶向补数据的靶单。

### async 客户端(默认,官方部署同款)

`eval_closed_loop.py --client async`(默认)。结构照抄 NVIDIA 自己的
`gear_sonic/scripts/run_vla_inference.py`:

> **async 客户端和 `backend.clean_deploy` 是同一个决定,不是两个。** async 会装上
> `CleanClock` 的 commander,它丢弃**所有** planner 命令——包括启动 deploy 控制回路的
> `start_control`;顶替它engage的是 `SONIC_AUTO_START`,而栈只在 `clean_deploy` 为真时导出它。
> 两者拆开时 deploy 会接受模式切换、收到站立 token,却**从不启动控制回路**,机器人就倒在
> reset 把它放下的地方,回合报 `[stand] robot did not stabilise after reset`。
> 2026-09-17 实测:deploy 日志里 245 个已收 token、**零条 `Loop timing`**。
> 重构前这对绑在一行里(`wall) export CLEAN_DEPLOY=1; CL="--client async"`),配置组把它们
> 拆到两个 dataclass 后没人再守着。现在 `EvalConfig.__post_init__` 强制 `clean_deploy =
> (client == "async")`,实验文件里写什么都覆盖不掉;`scripts/eval_bottle.py` 也改成把**校验后**的
> backend 交给栈,而不是文件里的原始 dict(否则不变量在半路被绕过)。

```
worker 线程:每 0.4 s 取最新观测 → policy server 出 40 步 chunk
主循环:   30 Hz 不间断发 token;新 chunk 以延迟补偿索引接入
           (idx = 推理耗时 × 30,机器人执行的动作永远对应"现在")
```

线程分工承重:worker 持有相机/状态 socket 与 policy link,主线程持有
commander 与特权判据(ZMQ socket 不可跨线程)。与官方的两处有意差异:
30 Hz 发布(数据是 30 fps;官方 50/50,各自自洽)、FSQ 吸附保留(官方裸发浮点,
仅 `|x|>1.25` 整块丢弃)。`--client blocking` 保留作 A/B 对照。冻结 DiT 消融
(单卡,`--no-tune-diffusion-model`):val 曲线稳定落后全量 ~12%,约等于慢
1000–2000 步——DiT 贡献真实但不巨大;闭环等价性未测。

### 处方:靶向 DAgger

让**专家出生在学生卡住的位置**(离桌沿 0.35–0.55 m;臂垂就是自然站姿,无需在线
接管——也绕开了"专家标签要 encoder、学生驱动要 NO_ENCODER"的硬冲突),录它
抬臂→伸手→抓的恢复轨迹,再与原数据合并转换重训。

### 初始姿态:换掉 NVIDIA 的托管 token(2026-08-29)

VLA 回合开场先用一个站立 token 托管机器人,等第一个 chunk 算出来,再 1 s blend
过去(NVIDIA 的 initial-pose 机制)。托管用的常量原本是官方的
`LATENT_INITIAL_MOTION_TOKEN`,而官方 docstring 自己就写了它是 checkpoint-specific
的。

问题不是它无效,是**它的姿态不在训练分布里**:官方那个托管的是"双臂前伸屈肘、
躯干前俯",200 局里每一帧站立都是"手臂自然下垂、直立"。于是交接前机器人先摆
一个数据里不存在的姿势,blend 时再跳回数据里的姿势——第三人称录像里那个"先往前倾、
又突然挺起来"就是这个。

量的方法要注意:`sonic_decoder.joint_targets` 的**绝对值未经验证**(docstring
明写),只有 `joint_delta_deg` 可信。用它量(12 个抽样站立 token,30 帧平均):

| token | 到数据站立 token 的平均关节差 |
|---|---|
| **我们的 medoid** | **3.38°** |
| NVIDIA `LATENT_INITIAL_MOTION_TOKEN` | 23.02°(远 6.8 倍) |

**为什么是 medoid 不是合成。** 先试过三种合成,解码出来**全都比官方更远**:
逐维中位数(latent 非线性,token 的中位数不是姿态的中位数,解出 161° 的肩)、
`sonic_encoder` 的 `_stand_window()` 编 `DEFAULT_ANGLES`、以及同一个窗口编数据的
实测平均站姿。encoder→decoder **不是恒等**,所以合成在这里不可信。medoid 不需要
往返:它是自己的 stack 在静止站立时**真实发出过**的 token,取到其余 193 个 L1
距离和最小的那个。

落在 `src/locomani/backends/sonic/initial_pose.py`,默认生效;`TF_INIT_TOKEN=official`
可退回官方。实测录像里托管段已经变成手臂下垂直立,前倾和"挺一下"消失;3 局冒烟
2/3 成功。

## 7. 排错与停止

| 现象 | 原因 / 处理 |
|---|---|
| 100 局全部 `[stand] token-only ready posture not delivered` | 初始 token 文件与训练数据的开局姿势不一致(见 §5);用 `eval.initial_token` 指向与这份数据开局一致的文件 |
| 100 局全部死在 `[stand]`,策略一次动作都没被问过 | 先核对**评测实际用的**出生分布和阶段参数:运行目录的 `eval/eval_config.json` 就是交给评测器的那一份(2026-09-17 起整份 `--config` 传递)。此前阶段只手抄 4 个字段,`spawn_edge` 因此从配置的 1.1–1.3 m 变成评测器默认的 0.7 m,100 局全部作废 |
| 单个 20 局跑出一个"很好/很差"的数字就下结论 | 20 局的 Wilson 区间宽 30-40 点(见 §4);先看是不是与另一场配对显著,不显著就只能说"看不出大效应",不能说"没有差异" |
| 部署参数(`--ensemble`/`--replan-interval` 等)的旧结论直接套到新 checkpoint | 见 §6 历次翻案(dart200 上 ensemble +50,flow200 上为零;replan 1.6 在 walk200 上更差、在 dart200 上更好)——**换 checkpoint 必须复筛**,引用旧数字要引本文档的裁定小节,不要引中途重挖的原始 json |
| 闭环评测跑在错误的场景高度上 | 采集脚本按种子重建 `outputs/scene/sim`(桌高会变),评测栈读磁盘上当时的场景;不同批次场景不一致会让结果不可比(2026-09-04 曾把 smooth200 三臂跑废,归档为 `*_scene647`)。现在评测脚本每次启动前固定重建场景,混用旧数据前先确认场景哈希一致 |
| 双卡并行跑评测,GPU 参数没生效 | 曾出现评测脚本的 GPU 参数覆盖外部 `CUDA_VISIBLE_DEVICES`,并行两臂的 policy server 实际都落在 GPU0(已修);怀疑时看 `serve.log` 里的显存占用卡号 |
| `policy server did not come up`,`serve.log` 末尾是 `checkpoint ... records its VLM backbone at /group-volume/..., which does not exist here` | checkpoint 在别的机器上训练,`config.json` 的 `model_name` 指向那台机器的 Cosmos-Reason2-2B 目录。手动调用 `serve_policy.py`/`eval_crate_closed_loop.py`/`gr00t_open_loop.sh` 时加 `--backbone-path <本地目录>`(见 §3);`scripts/eval_bottle.py` 不透传这个参数。旧代码对本地目录路径还会报 `Unsupported model name`(要求 `nvidia/` 前缀),现已改为只匹配模型家族名。`serve.log` 里的 `listening` 行若始终不出现而进程仍在、端口已监听,是 stdout 缓冲(见 §3 读日志) |

**Ctrl-C 停止**:`eval_stage.run()` 用 `try/finally` 保证中断时 policy server 和
stack 都会被正常回收(不会留下孤儿 deploy 或占用 sim 端口),但 `manifest.json`
只在整局跑完后才写,中断后重跑等于从头开始——`eval_bottle.py` 本身不支持
断点续跑。手动调用 `eval_closed_loop.py` 时可以用 `--stall-abort SEC`(窗口内
净位移 < 3 cm 且尚未 commanded_close 就提前收局,实测取舍见下表;**默认关,
发表的数字必须保持关**,代价是单向地把成功记成失败)和 `--resume`(保留已有
manifest,从上次中断处补到 `--fresh` 轮——它不逐位复现原轮次,但成功率是蒙特
卡洛估计,补跑用 `seed + 已完成轮数` 重新抽样统计上仍然有效;**换了种子再
resume 会被拒绝**,要混用得 `--force-resume`)两个 opt-in 参数缩短探索时间:

| 窗口 | 触发 | 把成功判成失败 | 省下墙钟 |
|---|---|---|---|
| 15 s | 76/100 | 6 | ~44% |
| 20 s | 73/100 | 4 | ~40% |
| 25 s | 71/100 | 2 | ~37% |
| **30 s** | 70/100 | **1** | ~33% |

## 8. 进一步阅读

- [运行基础设施](infra.md)：配置组怎么组合、阶段产物与断点、多卡并行的作业文件与资源准入。
- [数据采集](collection.md) / [双臂采集](crate_collection.md):任务与场景怎么来。
- [数据集转换](dataset.md):train/val 数据集的字段与审计。
- [训练](training.md):从数据集到 checkpoint,含 checkpoint 事后选流程。
- `docs/lockstep.md`:经验延迟 lockstep 客户端的校准记录。

## 双臂 Closed-loop：`src/locomani/pipeline/eval_crate_closed_loop.py`

单臂闭环的 crate 版本（2026-09-15）。经 `scripts/eval_crate.py`（`task.plugin=crate`
时自动路由到这里）或直接手动调用都行；一条命令负责三件事：在指定实例 / GPU 上起
token-only 栈（`NO_ENCODER=1`，每批重启，与 crate pilot 一致）、用
`configs/gr00t_sonic_3cam_bimanual.py` 在自己的端口（默认 5700，避开实例端口块；
单臂评测的策略端口 5590 正是实例 1 的键盘端口，所以单臂评测运行期间双臂评测要用
`--instance 2`）起策略服务、在栈环境里运行 `locomani.tasks.crate.vla_rounds`，
最后把 manifest 聚合成 `closed_loop.json`（字段与单臂相同：`successes`、
`success_rate`、`wilson95`、`failure_stages`）。

每批**先跑一局不录像的热身回合**（`vla_rounds.warm_up`，约 10 s）。deploy 进程的第一局
必然经过它参考动作的第 0 帧半蹲（膝 0.67、骨盆 0.755 m），并在切到 streamed 的瞬间
一步跳到 token 姿态——0.3 s 内膝伸 0.5 rad、身高 +3.3 cm、俯仰摆 7.5°，就是从前每段
评测录像开头那下前倾；第二局起同样的序列完全无瞬态。实测矩阵、deploy 侧机制和五条
试过无效的方案见 [开局跳变](handoff/crate-eval-prelude-twitch-20260917.md)。

```bash
# 经 CLI(推荐):task=crate_table30 一类实验会自动路由到这里
python scripts/eval_crate.py outputs/runs/<crate_run> --gpu 1

# 手动调用(脱离 pipeline 单独调试用)
"$VENV_SIM_PY" -m locomani.pipeline.eval_crate_closed_loop \
  --checkpoint outputs/<run>/model/checkpoint-10000 \
  --scene outputs/scene/crate_table30_depth04 --out outputs/crate_eval_100 \
  --instance 1 --gpu 1 --serve-gpu 1 --rounds 100 --seed 6000 \
  [--backbone-path ~/Checkpoints/Cosmos-Reason2-2B]   # checkpoint 在别的机器上训练时,见 §3
```

`--backbone-path` 原样透传给它起的 `serve_policy.py`;目录不存在或没有 `config.json` 时,
参数解析阶段就报错,不会起栈。经 CLI 路由时 `eval_stage` 不传这个参数,别处训练的
checkpoint 只能走这条手动命令。

每局：`reset_ready` 的采样（抖动、yaw、桌沿距离）→ `prepare_token_pose` 用
`configs/deployment/crate_initial_token.json`（训练数据首帧 token 的量化均值 + 首帧手臂关节）
把机器人带到数据的开局姿势 → 阻塞式 chunk 循环：三路相机 + 60 维状态 → token [40,64] 与左右夹爪
[40] → 每次发 `--execution-horizon`（默认 16）帧，两只手各自展开 → 判定用 crate 专家自己的
`evaluation.evaluate`（抬升 ≥ 0.6×`lift_height`、倾角 < `max_tilt_deg`、双手同时接触），
连续 2 s 成立才算成功；否则沿用单臂的 `[vla] walk / reach / close` 失败串。
观测构造与策略连接在 `vla_obs.py` 里按 `TaskSchema` 切换（`ObsBuilder(schema=SONIC_CRATE)`、
`PolicyLink.get_chunk_bimanual`），策略服务的 modality 与 `SONIC_CRATE` 不一致会直接拒绝。


实测结果(2026-09-15,`crate_vla200_record_20260914` checkpoint-10000,71/100)在
[双臂台账](ledger/crate.md)。
