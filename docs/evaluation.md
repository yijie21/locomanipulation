# 评测:open-loop 与 closed-loop

> 两套评测回答两个不同的问题,谁也替代不了谁。训练在
> [training.md](training.md)。

## 为什么必须两套

本仓库已有的硬证据:把**真值** token 开环回放,关节轨迹复现到 0.2–0.9°,机器人
却抓不到瓶子(浮动基座 1 分钟积分出 0.77 m 漂移)。所以:

- **open-loop** 回答"模型学没学到东西"——快、可逐 checkpoint 跑、用来选 checkpoint;
- **closed-loop** 回答"这个策略行不行"——唯一的验收标准。

bottle200 第一轮把这条原则演成了实例:开环关节角差 1.05°(贴测量噪声底),
闭环 6%。两个数字都对,测的不是一件事。

## Open-loop:`src/teleop_free/tools/eval_open_loop.py`

```bash
third_party/Isaac-GR00T/.venv/bin/python src/teleop_free/tools/eval_open_loop.py \
    --model-path outputs/train/<run>/checkpoint-N \
    --dataset datasets/..._val \
    --out outputs/eval/<run>/open_loop_checkpoint-N.json \
    --wandb-run <run>          # val 曲线写回训练 run,同一根 x 轴
```

全部指标都配 persistence 基线(块首真值动作保持整块 —— 数据 30.6% 的帧
token 与前帧相同,不摆这条线任何 MSE 都读不懂):

| 指标 | 定义 | 体现的性能 |
|---|---|---|
| `nmse_token` | 对 snap(量化吸附)后的 64 维 token 预测算 nMSE = MSE/var(gt),逐维归一化后再平均 | token(身体动作)预测的整体准确度,1.0 = 不如预测均值;归一化消除了量纲问题 |
| `nmse_token_unsnapped` | 同上但用 snap 前的原始连续预测 | 衡量量化吸附本身修正了多少误差 —— 对比 `nmse_token` 可看出模型输出偏离最近码字多远 |
| `nmse_gripper` | 单独对 gripper 1 维算 nMSE | 抓取相关的关键维度单独评估,避免被 64 维 token 稀释(不分开的话决定成败的这 1 维会被 64 维淹没) |
| `hit_rate_per_dim` | 预测(snap 后)token 索引与真值 token 索引相同的逐维命中率均值 | FSQ 空间每维实际只用 2–5 个格点,离散空间里命中率可读、MSE 不可读 |
| `hit_rate_all64` | 要求同一帧 64 维**全部**命中的比例 | 更严格的整体正确率 —— 一帧动作是否被完整无误地复现 |
| `gripper_mae` | 预测与真值 gripper 值的平均绝对误差 | gripper 连续值(开合程度)的平均绝对误差 |
| `close_frame_gt` / `close_frame_pred` | 真值/预测 gripper 信号第一次穿过(真值的)5%–95% 分位数中点的帧号 | 夹爪开始闭合的时间点,两侧都用真值量程判定,避免预测用自己的中点"讨巧" |
| `close_frame_err`(及 `close_frame_err_abs`) | 预测闭合帧号 − 真值闭合帧号;聚合时再取绝对值均值 | 抓取时机的提前/滞后帧数 —— 抓取成败系于闭爪时机,任何 MSE 都看不见它 |
| **`deg_mean` / `deg_p95` / `deg_max`** | 把真值 token 和 snap 后的预测 token 分别喂给 SONIC decoder(观测固定),取关节指令差的均值/95 分位/最大值,单位为度 | **头条数字**。经 SONIC decoder 换算,有物理单位;可与"token 直接重放已有 0.2–0.9° 误差"对比 |
| `degrees_noise_floor` | decoder 对真值 token 的重建输出,与实际记录的关节指令之间的误差均值(数据 30 Hz vs decoder 50 Hz 历史) | 误差的最低可信线:脚本自测噪声底(~1.1°),低于它的 `deg_mean` 差异不作数,只是解码重建噪声 |
| persistence 基线(对以上每个指标都各算一份) | 块首真值动作原样冻结、复制到区间所有帧,作为"什么都没学"的参照 | 判断模型是否只学到了数据的强自相关(30.6% 帧 token 与前帧相同)而非真实动力学;不优于此基线即视为未学到东西 |

跨 `execution_horizon ∈ {1,8,16,40}` 各报一遍 —— 误差随 horizon 的增长曲线
就是"一个 chunk 敢信多少步"的直接证据(选 checkpoint 只跑 h16 即可,全 horizon
留给选出的最优者,省 3 小时)。

## GR00T Open-loop:`src/gr00t/eval/open_loop_eval.py`

```bash
bash scripts/gr00t_open_loop.sh \
    --model-path outputs/train/<run>/checkpoint-N \
    --dataset-path datasets/..._val \
    --out-dir outputs/eval/<run>/checkpoint-N
```

跑的是 NVIDIA 自己的 `gr00t/eval/open_loop_eval.py`,不是上面这套自研 nMSE。
只有两个原始(未归一化)标量指标,64 维 token + 1 维 gripper 一起求平均,不拆分、
不归一化:

| 指标 | 定义 | 体现的性能 |
|---|---|---|
| MSE(单轨迹) | `mean((gt_action - pred_action)²)`,对整条轨迹所有时间步、所有动作维度一起求均值 | 预测动作与真值的均方误差,对大误差更敏感,但未归一化——不同动作维度的量纲/方差差异被直接混在一起 |
| MAE(单轨迹) | `mean(\|gt_action - pred_action\|)`,同样对所有维度求均值 | 预测动作与真值的平均绝对误差,比 MSE 更贴近"平均差多少"的直觉,对离群点不那么敏感 |
| 平均 MSE(跨轨迹) | 每条轨迹的 MSE 再取算术平均 | 模型在整个评测集上的整体预测精度(MSE 口径) |
| 平均 MAE(跨轨迹) | 每条轨迹的 MAE 再取算术平均 | 模型在整个评测集上的整体预测精度(MAE 口径) |
| 轨迹曲线图(非标量) | 每个动作维度画 state joints / gt action / pred action 三条曲线,每次重规划处标红点 | 定性排查预测是否跟踪真值、是否在重规划点附近漂移,不产出可比较的数字 |

每张图对应**一条轨迹**,标题写明 traj_id、state/action modality key
(`plot_trajectory_results`,`open_loop_eval.py:49-128`);图里纵向堆叠
`action_dim` 个 subplot,每个 subplot 对应动作向量拉平后的**一个标量维度**
(标题只有 `Action {idx}` 这个数字索引,不带语义名字)。G1 SONIC 是 64 维
token + 1 维 gripper,一条轨迹的图就有 65 个纵向子图。

每个 subplot 的横纵坐标与曲线:

| 元素 | 定义 | 说明 |
|---|---|---|
| 横轴 | 帧序号,`0 ~ actual_steps-1` | 30 Hz 重采样,图上不标秒数 |
| 纵轴 | 该动作维度的原始数值 | 未归一化、未换算单位;token 维是量化空间的原始浮点值,gripper 维是原始开合数值 |
| `gt action` | 真值动作该维度随时间的取值 | 数据集里的真实值 |
| `pred action` | 预测动作该维度随时间的取值 | 由多段 `execution_horizon` 长度的 chunk 拼接而成 |
| `state joints` | 状态关节该维度随时间的取值 | 只在 state 与 action 维度形状**完全相等**时才画(动作直接是关节指令的情形);SONIC token 空间维度不对齐,这条曲线不出现 |
| 红点 `inference point` | 每隔 `execution_horizon` 步、真正重新调用一次策略的时刻 | `open_loop_eval.py:171` 的主循环只在这些点上取真实观测重新推理;两个红点之间的预测曲线来自同一次推理的 chunk,原样开环回放,不看任何新观测 |

看图时,红点间预测线相对真值线的漂移幅度,就是"一个 chunk 内误差怎么累积"的
直接证据——在红点处被拉回还是跨红点持续发散,决定这个 `execution_horizon`
选大了还是选合适。同维度内的曲线才能比较,不同 subplot 之间数值大小没有意义
(每维物理含义、量程都不同)。

官方脚本对每条轨迹的图用**同一个**文件路径,多条轨迹会互相覆盖 —— 所以这个脚本
不透传 `--save-plot-path`,让官方脚本走它自带的按 `traj_id` 命名的默认路径,
跑完后再统一挪到 `<out-dir>/plots/`。产出:每条轨迹一张对比图、
`<out-dir>/metrics.json`(逐轨迹 + 平均 MSE/MAE)、`<out-dir>/run.log`。

参数是 named flag(`--model-path` / `--dataset-path` / `--out-dir`,可选
`--horizon` / `--embodiment-tag`),顺序无关;`--model-path` 和 `--dataset-path`
若传反,脚本会在加载模型前就报错(用 `meta/info.json` 是否存在做判据),不用等
模型加载完才发现传错。

<!-- ### bottle200 基线(checkpoint-10000,20 局 val)

| horizon | nMSE token(persistence) | 关节角差(persist.) | 闭合误差(persist.) |
|---|---|---|---|
| 1 | 0.120 | 1.05°(—) | 1.4 帧 |
| 16 | 0.142(0.323) | 0.93° | ~3 帧 |
| 40 | 0.176(0.895) | 1.30°(4.21°) | 4.9 帧(23.4) |

误差随 horizon 增长很缓,整块 1.33 s 可信;关节角差已到噪声底(1.16°)。 -->

## Closed-loop:`src/teleop_free/tools/eval_closed_loop.py`

三个进程,两条硬前提:

```bash
# 1. stack 必须 token-only(带 encoder 时外部 token 流压不住,cpp:3096)
NO_ENCODER=1 SCENE_RANDOM=0 bash src/teleop_free/start_stack.sh "pick up the water bottle"

# 2. policy server,第二张卡
CUDA_VISIBLE_DEVICES=1 third_party/Isaac-GR00T/.venv/bin/python \
    src/teleop_free/tools/serve_policy.py --model-path <ckpt> --port 5590

# 3. 评测本体(.venv_sim —— 它要 gear_sonic 的相机客户端和 pinocchio FK)
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
    src/teleop_free/tools/eval_closed_loop.py --fresh 100 \
    --out outputs/eval/<run>/closed_loop.json
```

动作下发相关的几个参数互相有隐式生效前提(完整 help 文本见
`eval_closed_loop.py:86-124`)。**这张表的默认值就是代码里的默认值**——两者
分家过一次(表写 `--execution-horizon` 8 而代码是 16;表写 `--ensemble` 1 而
代码被改成了 3),所以改任何一个默认值时,这张表要一起改:

| 参数 | 默认值 | 生效条件 | 作用 |
|---|---|---|---|
| `--client` | `async` | 始终生效,决定下面三个参数哪些真正起作用 | `async` = 官方部署同款流式(worker 线程持续推理,主循环不间断发布,延迟补偿索引接入,见下文"async 客户端");`blocking` = execute-N-then-replan,round-1/2 基线,仅作 A/B 对照保留 |
| `--execution-horizon` | `16` | 仅 `--client blocking` | 一个 chunk 执行几帧再重新请求推理;`async` 下解析但不使用——传了不影响行为,只是原样记进结果 JSON,容易误读 |
| `--replan-interval` | `0.4`(秒) | 仅 `--client async` | worker 多久请求一次新 chunk,不是发布频率(发布固定 `DATASET_FPS`)。`0.4` 是梯子(5.9/13.7/28.0/47.5)的设置;`1.33`(=40/30)等于一整个 chunk 播完才重规划,chunk 间重叠归零 |
| `--ensemble` | `1`(关) | 仅 `--client async`;实际重叠的 chunk 数还受 `--replan-interval` 制约,间隔拉到 `1.33` 时重叠归零,`--ensemble` 设多大都没有意义 | 混合最近 N 个仍覆盖"现在"的 chunk 预测(ACT 式时序集成)。`N=1` 是梯子测量时的状态,打开后的数字**不进梯子**;实测(见下文"终表")梯子上 −13 点(p=0.060)、并行/时序退化下反而 +20 点(p=0.012)——已定性为部署开关,不转正 |
| `--ensemble-m` | `0.3` | 仅 `--ensemble>1` 且 `--client async` | 集成权重按 chunk 年龄衰减:`m>0` 偏新(默认,在延迟补偿之上再叠一层保守)、`m=0` 均匀、`m<0` 偏旧——ACT 论文自己用的是负值方向 |

设计要点,每条都有伤疤背书:

- **复用 grasp_gym 自己的采样器**(`VLA_POLICY` 钩子,fork 点在 settle 之后)——
  参照线是脚本专家在同分布上的 **200/210 = 95.2%**,自己另写采样器 = 悄悄换分布,
  对照作废。
- **观测用 exporter 同款代码路径构建**(同一个 FK、同一个 gravity、图像原样透传),
  与训练数据构造上一致。
- VLA 局**出生即朝向 approach heading、跳过脚本转向**:token-only 模式下没收到
  token 时 `token_state` 置零(cpp:3066),站立撑得住、主动转向是坏的;而训练
  episode 本来就从转向完成后开始录。
- eval 会 `export NO_ENCODER=1`:gym 每局重启 deploy(`restart_deploy.sh`),
  不传下去的话第 2 局就被换回带 encoder 的 deploy。
- 结果 = 成功率 + Wilson 95% 区间 + **分阶段失败分布**(walk / reach / close /
  fell / timeout)。分阶段才可行动:失败在 close = 忠实继承了数据的缺陷;失败在
  walk = 没学会,触发加历史的消融。
- 每局 trace 落盘(含停滞位姿),它就是下一轮靶向补数据的靶单。


## Experimental Results:
### bottle200 基线(101 局 fresh)

```
6/101 = 5.9%  (Wilson 3–12%)      脚本专家同分布 95.2%
失败:reach 91 · close 2 · walk 1 · pre-vla 1
```

91/95 的失败是同一个固定点:干净走到抓取站位 → 臂垂爪开站满 90 秒。训练数据里
专家**边走边抬臂**,"到桌边臂还垂着"零出现,模型对该状态的回答是"站着",而站着
还是该状态。6 局成功是随机逃逸(与出生距离无关)。教科书 covariate shift。

### Round 2(bottle250 = 200 采集 + 50 DAgger 恢复):处方起效

- 开环:同款质量(best=10000,nMSE@h16 0.158,关节角差 1.10–1.37° 贴噪声底,
  闭合误差 2.2 帧)。注意**开环数字跨轮不可直接比** —— val 集随数据集变了。
- 闭环终局:**14/102 = 13.7%(Wilson 8–22%)**,round 1 的 2.3 倍,置信区间
  几乎不重叠 —— 靶向 DAgger 方向确认。失败仍以 reach 停滞为主:50 局恢复数据
  补上了洞的一部分,没补完(配比 50:200)。
- 客户端 A/B 判定:async 在 3/38(7.9%)处存档,**与 blocking 的 13.7% 相比
  无改善迹象 —— 客户端不是瓶颈**。真正的病灶随后被数据测量找到:训练数据里
  到位→伸手中位 4.6 s 的准静态站立(官方人类遥操作参考集是 0.2 s),
  96% 超过一个 chunk 的覆盖能力 —— policy 在正确模仿"站着"。
  见 fast-reach 分支的 settle 削减与重采。

### Round 3(fast200 = 200 局,停顿修好,**零 DAgger**):28/100

**闭环 28/100 = 28.0%(Wilson 20.1–37.5%)**。round 1 的 4.7 倍、round 2 的 2.0 倍,
区间与 round 1 完全不重叠。脚本专家参考 200/210 = 95.2%。

最要紧的一点:**round 3 没有用任何 DAgger 数据。**round 2 靠额外采 50 局恢复轨迹
换来 2.3 倍;round 3 只是把脚本专家的停顿削掉(到位→伸手中位 4.6 s → 3.0 s,
episode 从约 35 s 压到 20–22 s),用**同样 200 局**又翻了一倍。这直接证实了
round 2 末尾那个诊断:病在**数据形状**,不在数据量 —— "只采 200 条"这个目标本身
没有错。

**失败结构整个变了**,比成功率更有信息量:

| | round 1 | round 3 |
|---|---|---|
| walk / approach 失败 | 主导(walk→reach 停滞) | **0 / 0** |
| reach(到桌边但不下闭合指令) | — | **67** |
| close(下了指令但没抬起来) | — | 5 |

67 次 reach 失败全部发生在**机器人已经站在桌边、距离 0.23–0.29 m、在可抓范围内**
的位置。走路那段的病治好了,瓶颈整体后移到"到位之后不扣扳机"。

而这里有个直接矛盾值得记下:**开环测出的夹爪闭合帧误差只有 1.75 帧**
(真值 326.9 / 预测 328.65),也就是说模型**离线完全知道**该什么时候闭合,闭环下
却不发。这是 covariate shift 的典型形状 —— 闭环观测落进了训练集没覆盖的状态。

**所以下一轮 DAgger 该补的不是恢复轨迹(round 2 那套),而是"站在桌边、手已伸出、
临近闭合"这个片段的多样性。**

开环(best = checkpoint-9000,20 局 val,noise floor 1.361°):

| horizon | nMSE token | persistence | deg_mean | persistence |
|---|---|---|---|---|
| 1 | 0.1272 | 0.0000 | 1.136 | 0.000 |
| 8 | 0.1434 | 0.1555 | 1.199 | 1.164 |
| 16 | 0.1592 | 0.3439 | 1.277 | 2.025 |
| 40 | 0.1794 | 0.8791 | 1.365 | 4.336 |

h=1 时 persistence 恒为 0(单步上"重复上一个动作"就是真值),从 h=8 起才有对照
意义。模型误差随 horizon 增长极慢(40 步只涨 41%),persistence 涨 5.6 倍 ——
模型在**预测轨迹**而不是复读。整条 chunk(h=40,1.33 s)的关节误差 1.365°
正好压在噪声底 1.361° 上,所以整个 chunk 可以信任。

**FSQ 吸附有反面证据**:同一 checkpoint 同一 horizon,`nmse_token` 0.1592(吸附后)
vs `nmse_token_unsnapped` 0.1369 —— **吸附让指标恶化约 16%**。官方
`run_vla_inference.py` 发的是原始浮点、不做吸附。开环已说明它有害,闭环对照待做
(ROADMAP 待办总表)。

### Round 4(fast250 = 200 + 50 靶向 DAgger):48/101,而且否掉了一个假信号

**闭环 48/101 = 47.5%(Wilson 38.1–57.2%)**,round 3 的 1.7 倍。但这一轮真正的
收获是**归因方法**:因为所有闭环跑都用同一个 seed、初始条件逐局相同,可以做
配对 McNemar,而不是只比总数。

| 比较 | 变的是什么 | 翻转 | McNemar p |
|---|---|---|---|
| round 3 → 确认跑 | **只换配置**(吸附 → 不吸附,同一个模型) | +22 / −19 | **0.755 不显著** |
| 确认跑 → round 4 | **只换模型**(加了 50 局 DAgger) | +31 / −14 | **0.016 显著** |
| round 3 → round 4 | 两者合计 | +29 / −9 | **0.002 显著** |

**FSQ 吸附那条结论被推翻了。**tier-1 的 50 轮 A/B 给出 19/50 vs 13/50
(+12 个百分点,看着很强),100 轮全预算配对检验说那是噪声。**50 轮闭环分辨不出
10 个百分点级别的差异** —— 这是本轮最该记住的一条。当时的判断是"方向清楚但
p=0.18 还不显著,需要全预算确认",确认跑正是为此设计的,它否掉了一个假信号。
图快跳过这一步,"不吸附有效"就会被写进文档。

(开环那边吸附确实恶化约 16%,那个测量没错 —— 只是**没有转化成闭环收益**。
开环指标与闭环成败的耦合比看上去松。)

真正起作用的是数据:靶向 DAgger 那 50 局,p=0.016。

**失败结构(52 个 reach 失败)**:

```
手臂没抬过        0
抬了但没够近     44   ← 仍是主病(85%)
手到位没扣扳机    3   ← DAgger 前是 12,这块基本补上了
```

DAgger 精准命中了"最后一厘米",但**中段 reach 覆盖仍是瓶颈**。下一轮应当在
同一个靶子上加量,而不是换靶子。

开环(best = checkpoint-10000,曲线到 10000 仍在降,没有 round 3 那种回升 ——
多了 25k 帧,同样 10k 步还没到过拟合拐点)。

### 交叉验证:官方 open-loop eval

`scripts/crosscheck_open_loop.sh` 用 NVIDIA 自己的
`gr00t/eval/open_loop_eval.py`(朴素 unnormalized MSE)对同一批 checkpoint 打分,
比的是**排序**不是数值:

```
best by NVIDIA's MSE : checkpoint-9000
best by our nMSE     : checkpoint-9000
Spearman rho         : +0.988  (10 个 checkpoint)
```

**AGREE。**逐点排序几乎完全对齐,两者在 4000/5000 和 6000/7000 出现同样的局部
回摆,同样在 9000 见底、10000 略回升。所以确实属于我们自己的那层 —— 轨迹迭代、
execution-horizon 索引、指标算术 —— 与官方实现一致。


### 对比官方: closed-loop eval
范围要说清楚:`serve_policy.py` 是官方 `Gr00tPolicy` + `PolicyServer` 的 71 行
封装,连 `parse_action_gr00t` 都从官方 eval import。**模型加载、归一化、
embodiment projection 若有错会同时骗过两边**,这次没有检验到那一层。

> **官方 eval 的 `--steps` 默认 200,会静默截断。**第一次跑给的是 DISAGREE
> (官方选 10000、我们选 9000,rho +0.952)。原因不是任何一方有 bug:
> `open_loop_eval.py:149` 是 `actual_steps = min(steps, traj_length)`,而我们的
> val episode 长 546–832 帧、**夹爪闭合在第 327 帧** —— 官方那一跑只评了前 200 帧
> 的走路段,根本没到抓取。改成按最长 episode 推导(832)后即 AGREE,rho 升到
> +0.988。这个默认值对 DROID 那种短 demo 合适,对 20 s @30Hz 的 episode 就是
> 静默截断,任何拿官方 eval 评长 episode 的人都会踩到。

### async 客户端(默认,官方部署同款)

`eval_closed_loop.py --client async`(默认)。结构照抄 NVIDIA 自己的
`gear_sonic/scripts/run_vla_inference.py`:

```
worker 线程:每 0.4 s 取最新观测 → policy server 出 40 步 chunk
主循环:   30 Hz 不间断发 token;新 chunk 以延迟补偿索引接入
           (idx = 推理耗时 × 30,机器人执行的动作永远对应"现在")
```

线程分工承重:worker 持有相机/状态 socket 与 policy link,主线程持有
commander 与特权判据(ZMQ socket 不可跨线程)。与官方的两处有意差异,
均注释在 `vla_round.make_vla_policy_async`:30 Hz 发布(我们的数据是
30 fps;官方 50/50,各自自洽)、FSQ 吸附保留(官方裸发浮点,仅
|x|>1.25 整块丢弃)。`--client blocking` 保留作 A/B 对照。
- 冻结 DiT 消融(单卡,`--no-tune-diffusion-model`):val 曲线稳定落后全量
  ~12%,约等于慢 1000–2000 步 —— DiT 贡献真实但不巨大;闭环等价性未测。

### 跑得更快:`--stall-abort` 与 `--resume`(都是 opt-in)

一场 100 轮约 3.1 小时,而时间去向是失衡的(round 3 的 100 条 trace 实测):

```
成功轮   中位 19.6 s   (做完就收)
失败轮   中位 89.8 s   (min 89.5 / max 90.0 —— 每一局都烧满 TIMEOUT_S=90)
```

失败轮里机器人**不是静止的**:最后 30 秒骨盆走了 0.25 m 的路程,净位移只有
0.013 m —— **18.8 倍**,是在原地踏步。所以判据必须是**窗口内的净位移**,不是
逐帧速度。(第一版检测器用逐帧速度,零触发。)

`--stall-abort SEC`:窗口内净位移 < 3 cm 且尚未 commanded_close 就提前收局。
把 `_stalled()` 逐帧重放在那 100 条 trace 上得到的实测取舍 —— 标定与实现是同一段
代码:

| 窗口 | 触发 | 把成功判成失败 | 省下墙钟 |
|---|---|---|---|
| 15 s | 76/100 | 6 | ~44% |
| 20 s | 73/100 | 4 | ~40% |
| 25 s | 71/100 | 2 | ~37% |
| **30 s** | 70/100 | **1** | ~33% |

**默认关,而且发表的数字必须保持关。**代价虽小但是**单向**的:它判错的每一局都是
把成功记成失败,偏差永远朝下。5.9 / 13.7 / 28.0 全部是在 90 秒全预算下测的,开了
这个开关测出来的数不在同一条梯子上。打开时命令行会打印一行警告,结果 JSON 里也
记 `stall_abort_s` 和 `full_budget`。用于诊断性扫描(checkpoint 扫、消融、
frozen-DiT),不用于 headline。

`--resume`:保留已有 manifest,补到 `--fresh` 轮为止。**它不会逐位复现原来的轮次**
—— `grasp_gym` 只开一个 RNG 流式消耗(`grasp_gym.py:302`),跳过轮次会平移之后所有
抽样。但成功率是蒙特卡洛估计,需要的是**同分布**而不是同一批样本,所以补跑用
`seed + 已完成轮数` 从同一分布再抽,统计上完全有效。(上次 ENOSPC 在第 69 轮把整场
丢掉,这条就是那次的保险。)

**换了种子再 resume 会被拒绝**,而不是警告:混出来的样本仍然合法、算术仍然正确,
正因如此下游没有任何地方会显示它是两个种子谱系的混合 —— 所以必须是硬错误。
确实要混就 `--force-resume`。

冒烟验收(`--fresh 5 --seed 6000`,与 round 3 的 manifest 前 5 行逐行比对):
appearance / obj_xy / robot_xy / stand_xy / yaw **五个字段五轮全部一致**,
默认路径行为未变。开关本身也实跑过:`--resume` 从 5 补到 7 行、`--stall-abort 30`
让两局在 42 s 和 77 s 收局(否则各 90 s),结果 JSON 记下
`stall_abort_s` / `full_budget` / `resumed_from`。

### 处方:靶向 DAgger(`scripts/collect_dagger.sh`)

让**专家出生在学生卡住的位置**(离桌沿 0.35–0.55 m;臂垂就是自然站姿,无需在线
接管 —— 也绕开了"专家标签要 encoder、学生驱动要 NO_ENCODER"的硬冲突),录它
抬臂→伸手→抓的恢复轨迹。`PROBE=1` 先 10 局探站立净空(实测 9/10 通过)。采完与
原数据合并转换(`--strip-pose-columns` 统一混代际 schema)重训。

### 初始姿态：换掉 NVIDIA 的托管 token（2026-08-29）

VLA 回合开场先用一个站立 token 托管机器人，等第一个 chunk 算出来，再 1 s blend 过去
（`vla_round.py`，NVIDIA 的 initial-pose 机制）。托管用的常量原本是官方的
`LATENT_INITIAL_MOTION_TOKEN`，而官方 docstring 自己就写了它是 checkpoint-specific 的。

问题不是它无效，是**它的姿态不在我们的训练分布里**：官方那个托管的是「双臂前伸屈肘、
躯干前俯」，我们 200 局里每一帧站立都是「手臂自然下垂、直立」。于是交接前机器人先摆
一个数据里不存在的姿势，blend 时再跳回数据里的姿势——第三人称录像里那个「先往前倾、
又突然挺起来」就是这个。

量的方法要注意：`sonic_decoder.joint_targets` 的**绝对值未经验证**（docstring 明写），
只有 `joint_delta_deg` 可信。用它量（12 个抽样站立 token，30 帧平均）：

| token | 到我们数据站立 token 的平均关节差 |
|---|---|
| **我们的 medoid** | **3.38°** |
| NVIDIA `LATENT_INITIAL_MOTION_TOKEN` | 23.02°（远 6.8 倍）|

两者互比：平均 23.75°、最大 133.8°，肩 pitch 和肘主导——和肉眼看到的「手往内扣」一致。

**为什么是 medoid 不是合成。** 先试过三种合成，解码出来**全都比官方更远**：逐维中位数
（latent 非线性，token 的中位数不是姿态的中位数，解出 161° 的肩）、`sonic_encoder` 的
`_stand_window()` 编 `DEFAULT_ANGLES`、以及同一个窗口编我们数据的实测平均站姿。
encoder→decoder **不是恒等**，所以合成在这里不可信。medoid 不需要往返：它是我们自己的
stack 在静止站立时**真实发出过**的 token，取到其余 193 个 L1 距离和最小的那个。

落在 `src/teleop_free/initial_pose.py`，默认生效；`TF_INIT_TOKEN=official` 可退回官方。
实测录像里托管段已经变成手臂下垂直立，前倾和「挺一下」消失；3 局冒烟 2/3 成功。


## 加速评测:`scripts/seq_ab.sh`(A+C+B 筛选 harness)

一场 100 轮全预算闭环 ≈ 3.1 小时(失败轮 ~138 s = 90 s 预算烧满 + ~48 s
出生/站立/走位开销),两臂 A/B ≈ 6 小时。三个正交的杠杆把筛选压到 ~1 小时:

- **A 并行实例**:两臂各占一个 stack(`INSTANCE=0/1`,DDS domain + 端口块隔离,
  与 `par_batch.sh` 采集用的同一套机制),policy server 各占一张 GPU,端口
  **6590/6591** —— 不是 5590,因为 INSTANCE=1 的 keyboard 端口恰好是 5590
  (`scripts/paths.sh` 里有警告的那个冲突)。
- **B 筛选预算**:`eval_closed_loop.py --budget 60`(梯子是 90)。偏差两臂共享
  且**永远朝下**(round 4 实测 7/48 的成功落在 60 s 之后)。结果 JSON 记
  `budget_s`,`full_budget` 只在 90 s 且无 stall-abort 时为 true。
- **C 序贯停止**:`tools/seq_mcnemar.py` 按轮配对(同 seed → 同初始条件;
  `appearance` 不一致直接拒测),每 20 配对轮一次 look:interim p<0.01 判
  DIFFERENT,final(n=100)p<0.05,**追不上显著性就 FUTILE 提前收**。
  Pocock 风格的筛选阈值,不是精确的 STEP —— 所以:

**筛选出的赢家必须再过一次 90 s 全预算确认才算数**(round 4 的教训:50 轮
A/B 报出的 +12 点在全预算配对下是 p=0.755 的噪声)。FUTILE 则是真实的
"不值得再烧 6 小时"。

```bash
# 对照(A 臂) vs 时序集成(B 臂),60 s 预算,最多 100 轮,提前停
ROUNDS=100 BUDGET=60 SEED=6000 bash scripts/seq_ab.sh \
    outputs/eval/seq_ens "" "--ensemble 3 --ensemble-m 0.3"
```

已用 2 轮 smoke 端到端验证:双 stack 共存、配对校验、序贯 look、判定后自动
清场(含 exporter —— 它不监听端口,端口扫杀漏掉它,par_batch 先踩过)。

### 2026-08-27 更新:并行罚项、配对规则、观测流审计

seq_ab 首次实战(时序集成 A/B)撞出四件事,全部已落地:

- **配对按 appearance,不按行号**(`seq_mcnemar.py`,756827d)。stand 阶段摔跤
  会以同一 appearance 重试,两次 attempt 都进 manifest,逐行配对在有重试碎片的
  场次直接错位(第一场 dye-swap 就死在这)。同 appearance 取**最后一次** attempt
  为该局结果;共享率 <90% 视为跨 seed,拒测。
- **并行罚项是真的,且不是帧率**:与 round4 完全同配置的对照臂在双实例并行下
  从 ~38% 塌到 ~17%(配对 p≈0.004),换实例(dye-swap)方向不变,剔除帧率
  degraded 的局(frame_rate_guard 标记)量级不变。而**时序集成臂在并行下完全
  不掉**(vs round4@60s 配对 +11/−12)。机制候选:CPU 调度抖动 → chunk 陈旧化。
  推论:**时序敏感的配置只能在单实例上做最终裁决**;run_meta 带 `parallel: true`
  (c7456a3),并行数字永不入梯。
- **观测流可审计了**:`tools/frame_rate_guard.py`(878dc0b/b1f89cd,逐局
  obs_hz 中位/degraded 标记,标记不作废)+ `tools/instance_watchdog.py`
  (30fa1fe,process/port/stream/progress 四轴,只报不动)+ seq_ab 结束时归档
  逐实例 sim.log(83975fb)。背景:一个泄漏的 chrome 曾吃掉 9 核 8 小时,把
  观测流饿到 13-15 Hz 而没有任何东西报警。
- **start_stack 的 image-publish 守卫**(9bd0148):孤儿 publisher 占住 :5555
  时,旧脚本会在零帧状态下报 "stack up"。

### 2026-08-27 发现:闭环大多数局没考走路(stand-start gap)

**~2/3 的闭环局在 VLA 接管时机器人已经站在抓取站位上**(首帧骨盆→物体距离中位
0.27-0.32 m ≈ 规划站位,而出生点是 0.78 m;R2 62% / R3 76% / R4 69% / ens90 84%)。
settle 漂移检查通过后到首个流式帧之间的 ~5 s 窗口(2 s IDLE hold + 客户端前奏)
里,机器人被向桌漂移 ~0.35 m,segment 未定位。旁证:`[gym] pre-walk` 日志与
trace 首帧互相印证到 8 mm(locomanipulation-04 复核)。

后果与边界:

- 闭环数字描述的是**混合任务**(~2/3 站位伸手 + ~1/3 完整走过去抓),与 95.2%
  脚本专家(全程走)的对比偏袒 VLA;"R2 后 walk 失败归零"是没考,不是学会了。
- **梯子的相对比较活着**:R3→R4 按走/没走分层,同卷对子(n=64)独立 +16/−5
  (p=0.027),三层方向一致。
- trace 现在逐局记录 `entry_pelvis` / `tstart_pelvis` / `prelude_s`,fork 处打
  `[gym] fork->vla` 日志(6e79896)——下一场直接分解缺口。修复(让 VLA 真正从
  出生点接管)排在 blocking90 之后;修复后梯子绝对值全部需要重新表述。

另:episode 保存后的三处无 facing 的 `hold(IDLE)` 曾把机器人转回世界 +x(视频里
"抓完后退再转身"的来源,5500759 修复,已录数据从未被污染——save 在转身之前)。

### 2026-08-27 终表:同一模型上的客户端/集成三方配对(单实例,90s,seed 6000)

|  | 成功率 | vs round4 配对 | p |
|---|---|---|---|
| round4 = async,无集成 | 48/101 = 47.5% | — | — |
| async + 时序集成 N3M0.3 (ens90) | 35/100 = 35% | +14/−27 | 0.060 |
| blocking,无集成 (blocking90) | 16/100 = 16% | **+7/−39** | **<1e-5** |

- **async 客户端在 round-4 模型上贡献 ~30 个点**(16%→47.5%,逐局配对判定)。
  这修订了 round2→3 的归因:当时 13.7%→28.0% 被全部归给数据形状,依据是
  "async 在 round-2 模型上无增益(3/38 vs 14/102)"——但那场 A/B 只有 38 局、
  不配对,而且客户端与数据是同时换的。现在的干净测量说明:**数据形状与
  客户端各占一部分,无法追溯拆分**(R2 模型的 async 场没跑满);R3→R4 的
  DAgger 增益不受影响(两轮同为 async)。
- **时序集成:梯子上 −13 点(p=0.060,边缘),并行/时序退化下 +20 点(p=0.012)**。
  处置:不转正,保留为部署开关。机制自洽:M=0.3 时每帧动作 56% 来自
  0.4-0.8 s 前的预测——时序干净时是拖累,时序抖动时是滤波。
- blocking 的失败分布(reach 80/close 4)就是 round-1 的"站立吸引子"画像,
  与 b7f220f 的机制推断(每 ~0.9 s 冻结 0.3-0.6 s 喂给站立吸引子)一致。

### stand-start 漂移定位(队列 #2,blocking90 百局标记)

prelude 两个候选**双双出局**:entry→t_start 位移中位 0.015 m,priming 执行段
0.014 m——**漂移不在 VLA 客户端里**。fork 时已在站位(73/99,y 中位 0.611)。
真正的窗口是 **settled(漂移检查通过,y≈0.94)→ fork 的 ~2-3 s**:位移中位
0.342 m,69/85 局 >0.25 m,双峰(~15% 的局纹丝不动)。窗口内只有 stream_mode
切换 / pin_arms / hold(2.0 IDLE) / start_video 四件事,细分标记已加,探针在跑。

### 2026-08-28:干净全任务基线(stand-start 修复后,队列 #4)

**round4 模型在完整任务(出生点走过去再抓)上的真实成绩:12/100 = 12%(Wilson 7-20%)。**
修复保持:100 局首帧 dist 中位 0.74,99/100 ≥0.55 —— 每一局都真的从出生点开始。
失败:reach 83(anatomy: arm-never-up 0 / **up-but-never-near 81** / staged-no-fire 2)
/ close 3 / walk 2。

改口规则,自此生效:

- **梯子(5.9 / 13.7 / 28.0 / 47.5)全部是"混合任务"数字**(~2/3 站位伸手 +
  ~1/3 完整任务),相对比较有效,绝对值不可再单独引用;与 95.2% 脚本专家
  (全程走)的对比只在本节的干净数字上成立。
- **12% 是第一个可发表的完整任务绝对值**,也是后续一切改进(20000 步训练、
  replan 0.2s、mid-reach DAgger)的配对基准(seed 6000,逐局可配)。
- 值得注意:walk 失败仅 2/100 —— VLA 的走路段本身几乎不挂;瓶颈与混合任务
  时代诊断一致,仍是 mid-reach(81/88),处方不变,基线换真。

### 2026-08-28:stand-start 修复升级为官方 initial-pose 机制

用户从官方教程(vla_inference: Customizing the Initial Pose)找到了正确原语:
**等待期不站在 planner 通道里,而是持续流式发送预置的站姿 latent token**
(`LATENT_INITIAL_MOTION_TOKEN`)。这正好命中我们量到的机制——drift 是
planner 通道的病,token 通道实测稳如磐石——所以 async 局现在的第一件事就是
灌站姿 token + enter_streamed,keepalive 线程在整个慢启动期间(连流+首推理)
保持它,最后 1 秒线性 blend 进首个 chunk(官方 client 同款)。

对比二次瞬移:spawn_off 中位 0.065→0.107(残余=1.5s settle,唯一还在 planner
里的窗口),但尾部 0.33→0.18 且有界,8/8 从出生点起步,blend 零摔倒。
**默认路径不再依赖任何 sim 特权——与真机部署同构**;二次瞬移降级为
LOCOMANI_REPLACE=1 应急开关。

## 2026-08-28:闭环默认改为 20 局

一场 100 局全预算闭环是 4.5 小时,探索阶段付不起。**`--fresh` 的默认值改为 20**
(`eval_closed_loop.py`、`seq_ab.sh` 的 `ROUNDS`、`seq_mcnemar.py` 的 `N_MAX` 三处
同时改),但必须知道 20 局买到的是什么、买不到什么。

**买不到:一个能引用的成功率。** 20 局的 Wilson 95% 区间有 30-40 个点宽:

| 观测 | 率 | Wilson 95% | 宽度 |
|---|---|---|---|
| 2/20 | 10% | 3-30% | 27 点 |
| 4/20 | 20% | 8-42% | 34 点 |
| 6/20 | 30% | 15-52% | 37 点 |
| 10/20 | 50% | 30-70% | 40 点 |
| (对照)12/100 | 12% | 7-20% | 13 点 |

也就是说 **20 局测出 20% 和 40%,区间是重叠的** —— 单独一个 20 局百分数不构成结论,
写进任何文档之前都要用 100 局重测。

**买得到:大效应的配对判定。** 与同 seed 的另一场逐局配对时,20 对能判定的是:

    一边赢 6 对、另一边 0 对   p=0.031  显著
    一边赢 8 对、另一边 1 对   p=0.039  显著
    一边赢 7 对、另一边 1 对   p=0.070  不显著
    一边赢 9 对、另一边 2 对   p=0.065  不显著

大约是"翻倍级"的效应才判得出来。round3→round4 那种量级(+29/−9)在 20 局里也会
显著;而 `--no-snap` 那种(+22/−19)在 100 局都判不出来,20 局更不可能 —— 后者恰恰
是我们最需要警惕的那类,所以**20 局出的"没差别"不是"没效果",只是"这个尺度看不见"**。

**用法约定**:

- 探索、筛选、排除明显错误 → 20 局(默认)
- 任何要进梯子、进文档、进论文的数字 → `--fresh 100`,90 s 全预算
- 20 局说"有差别" → 值得花 100 局确认;20 局说"没差别" → 只能得出"不是大效应"

## 2026-08-30:部署侧参数普查(Phase 0 + replan-interval 配对)

起因:协作者只改部署端参数报到 73%。本节是把部署侧因子系统扫一遍的记录,
**包括被杀掉的假设** —— 这一轮七个假设死了六个,记下来比记结论重要。

### Phase 0:离线扫描(`scripts/phase0_denoise_sweep.sh`)

walk200/checkpoint-10000,20 局 val,horizon 16。为这次调查给
`eval_open_loop.py` 加了 `std_ratio_token` / `std_ratio_gripper`(预测的逐维
std ÷ 真值的逐维 std)—— **nMSE 对残差的符号对称,看不见"幅度收缩"**,而
欠积分的 flow matching 干的正是这件事。

| steps | std_ratio_token | std_ratio_grip | 死区占比 | nmse(吸附) | nmse(裸) | deg_mean |
|---|---|---|---|---|---|---|
| 4 | 0.9911 | 1.0023 | 0.0658 | 0.1334 | 0.1101 | **1.3003** |
| 8 | 0.9964 | 0.9997 | 0.0915 | 0.1385 | 0.1149 | 1.3283 |
| 16 | 0.9945 | 0.9991 | 0.0977 | 0.1487 | 0.1238 | 1.3856 |
| 32 | 0.9983 | 1.0001 | 0.0871 | 0.1574 | 0.1317 | 1.4044 |

真值死区占比 0.0339;`deg_mean` 噪声底 **1.276**。

- **`--denoising-steps` 删除**:五个指标、四个点,全部单调恶化。4 是最优,机制上
  也讲得通 —— 模型按 4 步的 flow schedule 训出来,多走 Euler 步是沿一个为 4 步
  调好的向量场**过冲**。
- **snap 有害**:每一档裸预测都比吸附后低 ~21% nmse。闭环对照仍未做。
- **模型开环 1.3003 对噪声底 1.276** —— 只高 2%。**闭环丢掉的点,几乎全部在
  部署侧或分布侧,不在模型里。**

### 观测延迟 δ:33 ms,不是因素

`ComposedCameraClientSensor.read()` 每 10 帧自己打一次 `Image latency`,所以
这个数一直躺在跑过的 eval log 里,不用起 stack。walk200 两场共 2479 个样本:

    p10 18.1   p50 32.8   p90 47.1   p99 59.7   max 67.2 ms   (>80 ms 占 0.0%)

`latency_index` 未补偿的相机部分只有 ~33 ms,不到 25 Hz 的一帧。

### `--replan-interval 1.6` 配对:显著更差(seq_ab,seed 6000,20 局,60 s)

    pairs 20   A(0.4s) 9/20   B(1.6s) 1/20   discordant +A 8 / +B 0
    p = 0.0078   VERDICT DIFFERENT

|  | A 0.4 s | B 1.6 s |
|---|---|---|
| 成功 | 9/20 | 1/20 |
| 失败 min dist 中位 | 0.415 | **0.510** |
| 下蹲量中位 | 12.4 cm | 13.1 cm |
| grip 跌破 0.005 | 6/10 | 1/8 |

**更长的承诺损害的是走位**,而走位是这个 pipeline 里唯一本来没坏的东西:开环
走 1.6 s 收敛不到桌边,停在 0.510 m(专家合爪距离的 p95 是 0.505),后面全崩。
注意 arm B 有瑕疵:chunk 就是 40 帧 = 1.6 s,所以每周期播空后会卡在最后一帧
~0.2 s,同时引入了 blocking 的冻结病。但失败模式是**走位**不是伸手,结论不受
影响。

### 被杀掉的六个假设

| 假设 | 判据 | 结果 |
|---|---|---|
| 采样器幅度收缩 → 模式平均 | `std_ratio` | 0.99 / 1.00 → 死 |
| 观测延迟回拖造成不动点 | δ | 33 ms → 死 |
| 失败局手臂没抬起来 | `max palm_z` | **统计量循环**:成功局的 `max palm_z`(1.080)量的是**抓到之后举瓶子**的高度,抓握那一刻只有 0.718;失败局峰值 0.909 **反而更高** → 死 |
| 失败局身体没下蹲 | `pelvis_z` | 失败降 12.1/12.7 cm,成功降 11.4/11.9/12.7 cm,**一样** → 死 |
| 手掌贴得不够近 | `palm_obj` | 一个**失败**局达到 0.135 m 并停留 5 s,两个**成功**局只有 0.148 / 0.137 → 死 |
| 承诺更久(horizon 40) | 上表 | p=0.0078 反向 → 死 |

`max palm_z` 那条是本轮最值得记的教训:**一个在成功局里被结果本身污染的统计量,
会读成一个漂亮的机制。** 对照量必须取在"结果发生之前"的时刻(这里是首次合爪)。

### 唯一跨批复现的判别项:爪子

    成功: grip 打到 -0.0200 (clip 底)
    失败: grip 卡在 +0.0115 ~ +0.0195,跌破 0.010 的比例 0%

walk200 111 条 trace、seq_ab 两臂 38 局,**四批数据同一个带**。而训练集 gripper
是硬双峰:>0.020 占 53.3%,<0.005 占 43.5%,中间 [0.005,0.020] 只占 **3.3%**。
失败局整场停在训练分布的空洞里。

模型对"我到瓶子边上了"**有**表示:远时 24.5(全开),靠近掉到 ~18,手臂甩开还
会正确弹回 24.5。它只是从不越过那道坎。**但它区分不了"靠近 0.195 m"和
"到位 0.135 m"** —— 两者都输出 ~18.5,所以简单阈值二值化会对着空气合爪,必须
扫阈值而不能假设。

### 停住距离有一个窄甜区,损失的大头在这里

x 轴 = 机器人实际停住的距离(首次合爪前、t>5 s 的 `dist` 中位)。**分 run 看,
两个 run 峰值在同一格**:

| 停住距离 | walk200@0.7 (n=80) | walk200@1.2 (n=73) |
|---|---|---|
| [0.00,0.35) | 28% (n=29) | — |
| [0.35,0.40) | 14% (n=21) | 62% (n=8) |
| **[0.40,0.45)** | **40% (n=10)** | **93% (n=14)** |
| [0.45,0.50) | 7% (n=14) | 53% (n=19) |
| [0.50, …) | 0% (n=6) | 13% (n=31) |

**[0.40,0.45) 正是专家合爪距离的四分位区间**(p25 0.383 / 中位 0.422 / p75
0.452,171 局训练数据)。spawn 1.2 有 **31/73 局停在 0.50 m 以外**,那一格只有
13% —— 损失的大头不是"不会抓",是**停错地方**。

限制,必须写在结论旁边:

- **停住距离是结果不是处理,没有随机化。** 停得好的局可能本来其他方面也顺。
- 同一个停住距离在两个 run 里成功率不同([0.35,0.40) 是 14% vs 62%),所以还有
  run 级别的因素,停住距离不是全部。
- walk200@1.2 的 traced 子集是 73/100(视觉门拒的局没有 trace),45% 比真实的
  38% 偏高。

**能做因果结论的是出生距离**(它是随机化的)。两点线性外推:

    停住 = 0.383 + 0.194 x (spawn - 0.7)
    spawn 0.7 -> 停 0.383 (甜区左)   spawn 1.2 -> 停 0.480 (甜区右)
    spawn 0.92 -> 停 0.425 (甜区正中)

**我们两次测量一次偏左一次偏右,都没打中。** 预测:`--spawn-edge 0.9~1.0` 应
显著高于 1.2 和 0.7。这个预测很硬、很容易被证伪、且不需要改任何代码。

### trace 新增字段(`vla_round.py`)

`pelvis_z` / `torso_pitch` / `palm_quat` / `obj_in_palm`。前两个是为了把"抬手"
和"下蹲"分开(没有它们,`palm_z` 单独看会读成上面那个循环结论);后两个是因为
`palm_obj` 是标量距离,分不出"抓握位姿"和"同样远但朝向不对"。

### `seq_ab.sh` 补的两个静默错误

| | 原本 | 现在 |
|---|---|---|
| `TF_DATASET_FPS` | 没设 → `vla_round` 默认 30 | `DATASET_FPS`,默认 **25** |
| `--spawn-edge` | 没传 → `eval_closed_loop` 默认 0.7 | `SPAWN_EDGE`,默认 **1.2** |

外加 `boot_stack` 现在把 task prompt 传给 `start_stack.sh`(原先走它的默认值
`"walk forward"`),三者都写进 `run_meta.json`。

## 2026-08-31:普查收尾——grip-gain 配对、200 轮随机 spawn 扫描、悬崖修正

### `--grip-gain 3.0` 配对 A/B(spawn 1.1–1.3,各 20 轮)

A 4/20 vs B 3/20,McNemar p = 1.0000,**无差异**。决定性细节:

- A 自主合爪 4 次,palm-obj 中位 0.148 m(0.138–0.151),**4/4 成功**;
- B 被增益强迫在中位 0.211 m(0.173–0.279)合爪 17 次,只成 3 次;
- 失败类别整体从 reach 翻成 close,总成功一个没多。

**结论:gripper 是近乎完美的汇报者。** +0.0168 的"犹豫"是模型正确地说"还没
到位",不是不敢抓。修 close 是给失败改名,不是修失败。

### 随机 spawn 扫描(两实例同配置异 seed 7000/8000,--spawn-edge 0.7–1.4,200 轮)

总计 **79/200 = 39.5%**,gate 拒绝 29(几乎全在 [1.3,1.4):23/26)。

    spawn-edge   n   net 成功率 [Wilson95]
    [0.7,0.8)   29   34.5% [20,53]
    [0.8,0.9)   32   40.6% [26,58]
    [0.9,1.0)   21   47.6% [28,68]
    [1.0,1.1)   37   40.5% [26,57]
    [1.1,1.2)   26   50.0% [32,68]
    [1.2,1.3)   23   65.2% [45,81]
    [1.3,1.4)    3   3/3(gate 吃掉 23/26,无法解读)

**曲线平缓、CI 全部重叠**——之前从两个定点 run 外推的
`停住 = 0.383 + 0.194 × (spawn − 0.7)` 线性关系在单一随机化 run 内不成立
(corr(spawn, park) = +0.27),预言的 "0.9–1.0 甜区" 没有出现。net 有缓慢上升
趋势但 200 轮分辨不了;失败构成各 bin 一致,reach 主导。

### "窄甜区" → "悬崖" 的修正(本轮最重要的更正)

合并两 run 时 [0.40,0.45) bin 读出 93%(13/14)——是跨 run 汇总的假象。单一
随机化 run 内(n=103):

    park [0.00,0.35) 42%   [0.35,0.40) 68%   [0.40,0.45) 58%
         [0.45,0.50) 44%   [0.50,∞)  5% [1,24]
    park < 0.50 → 83 轮 55%      park ≥ 0.50 → 20 轮 5%

存活的不是"峰",是 **0.50 m 处的悬崖**。0.50 恰好是脚本专家站位生成器的物理
上限(build_rel_candidates x ≤ 0.275 → 标量 ≲0.485)外缘;但注意专家在
[0.50,0.62) 有**稀薄**覆盖(5.73% 帧、171/171 局),"零覆盖"的说法不成立,
且专家在 0.50–0.55 抓成过——悬崖的机制仍未证明,只有现象复现两次。

### 三条跨 run 方法论教训

1. **被结果污染的统计量**:max palm_z "成功 1.080 vs 失败 0.909" 只是在复述
   结局(成功局的 max 量的是瓶子被举到的高度;抓取本身发生在 0.718)。
2. **跨 run 绝对值不可比**:同 checkpoint 同 seed,定点 spawn 1.2 给 45%,
   spawn 1.1–1.3 给 8%;同一停位 bin 两 run 成功率 14% vs 62%。
3. **合并 run 的 bin 会制造假峰**:见上。结论只从单一随机化 run 内做。

### 对采集的落地(实现见 collection.md 2026-08-31)

失败质量钉在终止位置 + 专家无恢复演示 → `grasp_gym --dart-frac`(DART 状态
扰动变体,标签零污染),manifest 新增 dart / park_first / park 字段。

## 2026-08-31:prelude 偏移调查——"stand-start drift is back?" 警报的结案

`tstart_spawn_off` 验收线(0.1 m)在近期所有 eval 里每轮被突破(中位 0.19–0.26,
walk200 主评测 73/100 轮 WARNING),但**与成败无关**(成功局 0.189 vs 失败局
0.197)。逐段分解 + 两个历史 probe 对比 + 30 s hold-only 测量,因果链:

1. **re-teleport 不是失效,是退役**(`bbc207d`):被 NVIDIA initial-pose 机制
   (站立 token keepalive + 1 s blend)取代,当时验收 med 0.107。
   `LOCOMANI_REPLACE=1` 只是留的 sim-only 后门,默认关。
2. **数值翻倍的引入点是 `4967304`**(INIT_TOK 换成我们的 medoid):
   probe_initpose(NVIDIA)med 0.108 vs initpose_ours(ours)med 0.254——
   换 token 那次自己的 probe 里就写着,没人对验收线。
3. **但 medoid 无罪**。`diag_token_hold.py`(新增)纯 hold 30 s:
   ours median **0.010 m**,official 0.002 m——两个 token 都磐石稳定。
4. **主项是 planner→streamed 的模式切换台阶,不是 blend**(8 轮拆分 probe:
   hold_off median 0.084 / max 0.183,blend_off median 0.033)。enter_streamed
   时参考从 planner 姿态跳到 INIT_TOK 站姿,机器人物理地挪一小步——每轮都有、
   两种 token 都有、有界。diag 纯 hold 的 0.010 之所以量不到它,是因为 diag 在
   enter_streamed 之后才取基准点(盲点,已在 probe 中补量)。blend 期间
   chunk-0 的行走只有 ~0.03。0.084+0.033≈0.117,与 walk200 日志的
   fork→streamed 0.121 吻合。
5. 旧指标还把 planner settle(~0.13 m,采集时同样存在)算了进去:参照点是
   teleport 落点而非 settle 后的 handover 位置——对一个训练数据同样含有的
   位移报警。

**修法(全部落地,不开 re-teleport,保持 deployment parity)**:

- 参照点改为 handover 位置(grasp_gym 传 `s_fork` 的 pelvis);
- prelude 记账拆成 `hold_off`(含模式切换台阶,地板 ~0.18,WARNING 只盯它,
  阈值 0.25——健康时安静,真走飞了才响)+ `blend_off`(chunk-0 提前执行,
  策略自己的动作,落盘不报警)+ `tstart_spawn_off`(总量,保留连续性);
- 验收 probe(8 轮,seed 6000,spawn 1.2):hold_off med 0.084 max 0.183、
  blend_off med 0.033、total med 0.053——新阈值下零误报;
- **2026-08-31 前后的 tstart_spawn_off 不可比**(旧值含 settle 段,偏高
  ~0.13 m)。

方法论教训:验收指标混装了四段位移(settle / hold / blend / chunk-0 执行),
其中只有一段是它想守护的;警报每轮都响之后就没人看了。指标要一段量一件事。
