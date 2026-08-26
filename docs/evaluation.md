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

四个指标,每个都配 persistence 基线(块首真值动作保持整块 —— 数据 30.6% 的帧
token 与前帧相同,不摆这条线任何 MSE 都读不懂):

| 指标 | 为什么是它 |
|---|---|
| nMSE,token 与 gripper **分开** | nMSE = MSE/var = 1−R²,尺度无关;不分开的话决定成败的 1 维 gripper 被 64 维淹没 |
| 吸附后命中率(逐维 + 整帧) | FSQ 空间每维实际只用 2–5 个格点,离散空间里命中率可读、MSE 不可读 |
| **关节指令差(度)** | 头条数字。经 SONIC decoder 换算,有物理单位;脚本自测噪声底(数据 30 Hz vs decoder 50 Hz 历史,~1.1°),低于它的差异不作数 |
| 闭合时刻误差(帧) | 抓取成败系于闭爪时机,任何 MSE 都看不见它 |

跨 `execution_horizon ∈ {1,8,16,40}` 各报一遍 —— 误差随 horizon 的增长曲线
就是"一个 chunk 敢信多少步"的直接证据(选 checkpoint 只跑 h16 即可,全 horizon
留给选出的最优者,省 3 小时)。

### bottle200 基线(checkpoint-10000,20 局 val)

| horizon | nMSE token(persistence) | 关节角差(persist.) | 闭合误差(persist.) |
|---|---|---|---|
| 1 | 0.120 | 1.05°(—) | 1.4 帧 |
| 16 | 0.142(0.323) | 0.93° | ~3 帧 |
| 40 | 0.176(0.895) | 1.30°(4.21°) | 4.9 帧(23.4) |

误差随 horizon 增长很缓,整块 1.33 s 可信;关节角差已到噪声底(1.16°)。

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

### 交叉验证:官方 open-loop eval 说了什么

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
