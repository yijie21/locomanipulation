# Real-Time Action Streaming:async client 的三个机制

> `vla_round.make_vla_policy_async` 是 [evaluation.md](evaluation.md) "async 客户端"一节
> 提到的那个默认 client。这里把它拆成三个机制,各自讲清楚"不这么做会怎样"，再用实测数字
> 把流程走一遍。三者的位置都在 [vla_round.py](../src/teleop_free/vla_round.py)。

## 1. 实时动作分块(延迟补偿索引)

**问题**:推理不是瞬间完成的。GPU 去噪 + 网络往返实测要 0.3–0.6 秒(阻塞式 client 就是被
这段时间冻住的那个)。chunk 里的"第 0 帧"描述的是**请求发出那一刻**该做什么——如果拿到
chunk 后还从第 0 帧开始播,就是在执行一个已经过时的指令。

**解法**:给每个 chunk 记一个时间戳 `chunk_t0`(推理开始的那一刻),拿到 chunk 后不问"这是
第几个 chunk",只问"现在几点了、比 `chunk_t0` 过去了多久、按 30 帧/秒算应该播到第几帧"——
直接跳过已经"来不及"的那些帧。

![时间线图:t=0.40s chunk到达时,idx=round(0.40×30)=12,跳过前12帧(已过期),从第12帧开始播](realtime-chunk-timeline.svg)

**举例走一遍**(推理耗时取实测区间 0.3–0.6s 里的 0.4s 示意):

| 时刻 | 事件 |
|---|---|
| t = 0.00s | 发出推理请求,`chunk_t0 = 0.00` |
| t = 0.40s | chunk 到达(推理耗时 0.4s) |
| — | `idx = round(0.40 × 30) = 12` → **跳过前 12 帧**(占 40 帧 chunk 的 30%),从第 12 帧开始播 |

chunk 的 40 帧原本代表 t=0.00s 到 t=1.333s 这段未来。等它真正落地时,时钟已经走到 t=0.40s,
前 12 帧描述的"未来"已经变成了"现在之前",直接跳过,机器人执行的动作才始终对应"现在"而
不是"发请求那一刻"。

```python
with lock:
    if latest["t0"] is not None and latest["t0"] != chunk_t0:
        tok, grip = latest["chunk"]
        chunk_t0 = latest["t0"]
        idx = latency_index(time.time() - chunk_t0, RATE, len(tok))  # ← 这里算 12
j = min(idx, len(tok) - 1)
tok_j, grip_j = tok[j], float(grip[j])
idx = min(idx + 1, len(tok) - 1)  # 下一轮循环, idx 变 13
```
`vla_round.py:721-729`——`latency_index` 就是 NVIDIA 官方的 `calculate_latency_compensated_index`
(`vla_round.py:68-70`,verbatim import,不是照猫画虎重写的一份)。

**对照:阻塞式(blocking)client 怎么处理同一次等待**——等了 0.4s 拿到 chunk 后**永远从
第 0 帧开始播**,不管等了多久;等待期间机器人是**冻结**的,因为阻塞式在等
`link.get_chunk()` 返回时什么也不发([vla_round.py:426-435](../src/teleop_free/vla_round.py#L426-L435))。

## 2. 异步推理(worker / 主线程双时钟)

**问题**:推理慢(0.3–0.6s),但发指令给机器人必须快且不能停——真机的控制环等不了半秒。
如果用同一个循环"先等推理、再发指令",推理的那 0.3–0.6s 里机器人只能吃上一条指令的老本,
这正是阻塞式 client 每 ~0.9s 冻结 0.3–0.6s 的来源。

**解法**:把"算得慢的推理"和"发得快的发布"拆成两个各自独立运行的循环(线程),中间用一个
共享的"信箱"(`latest` 字典)传递结果。谁也不等谁:worker 线程按自己的节奏算完就把结果
放进信箱;主线程雷打不动地每 33ms(30Hz)从信箱里拿"当前最新的东西"发一帧,不管信箱里的
东西是不是刚放进去的。

![双线路示意图:worker线程车道疏(几段推理工作块),主线程车道密(连续不间断的30Hz发布),中间用latest字典当共享信箱](realtime-async-threads.svg)

- **worker 线程**:每次推理 0.3–0.6s,推完把结果写进 `latest`,按 `replan_interval`
  (默认 0.4s)歇一下再请求下一次——如果推理实际耗时已经逼近甚至超过 `replan_interval`,
  这个"歇一下"直接归零,worker 连轴转。
- **主线程**:固定 30Hz 连续发布,永远不因为 worker 是否在忙而停顿,只从 `latest` 里取
  "当前最新的"chunk 继续播,索引照第 1 节的规则往后走。
- 两者靠 `lock` 同步共享字典;靠职责分工避开 ZMQ socket 非线程安全的问题——相机流/状态流/
  policy socket 只活在 worker 里,commander 和特权状态查询只活在主线程里。

```python
def worker():
    while not stop.is_set():
        t0 = time.time()
        obs = obs_src.build()  # worker 专属:相机/状态 socket
        if obs is not None:
            tok, grip = link.get_chunk(obs)  # worker 专属:policy socket
            with lock:
                latest["chunk"] = (tok, grip)
                latest["t0"] = t0
        rest = inference_interval - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)  # 歇一下,没歇够就直接下一轮


th = threading.Thread(target=worker, daemon=True)
th.start()  # 主线程接着往下走,不等它
```
`vla_round.py:596-617`

## 3. 动作混合

### (a) 时序集成(ACT 式)

**问题**:新 chunk 每 `replan_interval`(默认 0.4s)落地一个,但每个 chunk 覆盖 1.33s 的
未来——也就是说在任意时刻,通常有好几个还没"过期"的 chunk 同时对"现在该做什么"有自己的
意见。只认最新那个、把旧的直接扔掉,每次切换都会在指令上留下一个小台阶(前后两次推理看到
的观测略有不同),这正是"手抬起来但一直不收敛"这类失败的典型成因。

**解法**(ACT / Diffusion Policy 的标准做法):不选边,投票——把所有仍然覆盖"现在"的
chunk 各自认为"现在该做的动作"取出来,按新旧程度加权平均。

![三个chunk A/B/C分别从t=0.0/0.4/0.8秒开始,在t=0.9秒这一刻同时投票:C权重44%,B权重32%,A权重24%,混合成一帧输出](realtime-ensemble-blend.svg)

**举例走一遍**(三个 chunk,t0 分别是 0.0s / 0.4s / 0.8s,在 t=0.9s 这一刻投票,
`ENSEMBLE_M=0.3`):

| chunk | t0 | 在 t=0.9s 对应的本地帧 idx | 权重(`exp(-0.3×age)` 归一化) |
|---|---|---|---|
| C(最新,age=0) | 0.8s | round((0.9−0.8)×30) = 3 | 44% |
| B(age=1) | 0.4s | round((0.9−0.4)×30) = 15 | 32% |
| A(最旧,age=2) | 0.0s | round((0.9−0.0)×30) = 27 | 24% |

三个 chunk 都还"活着"(各自的 1.33s 窗口都还没走完),都对 t=0.9s 这一刻有自己的预测——
C 最新、权重最高,但 A、B 仍然参与投票,边界处的跳变被这样的加权平均抹平。

```python
def _ensemble(buf, now, rate):
    picks, ws = [], []
    for age, (tok, grip, t0) in enumerate(reversed(buf)):  # age 0 = 最新
        i = int(round((now - t0) * rate))
        if i < 0 or i >= len(tok):
            continue  # 这个 chunk 已经放完了,不投票
        picks.append((tok[i], grip[i]))
        ws.append(np.exp(-ENSEMBLE_M * age))  # 越老权重越小
    w = np.asarray(ws) / sum(ws)
    tok = np.sum([p[0] * wi for p, wi in zip(picks, w)], axis=0)
    grip = float(np.sum([p[1] * wi for p, wi in zip(picks, w)]))
    return tok, grip
```
`vla_round.py:143-166`——默认关闭(`ENSEMBLE_N=1`),要 `eval_closed_loop.py --ensemble N>1`
才会真正投票。

### (b) 交接时的 1 秒淡入淡出

跟上面的多 chunk 投票是两码事——局刚开始时机器人被一个"站立不动"的托管 token 撑住,等
第一个真实 chunk 算出来后,不是硬切换,而是用 1 秒时间从托管 token 线性过渡到真实动作的
第一帧,像淡入淡出一样,避免交接瞬间的硬跳变。

```python
for i in range(n_bl):  # n_bl = 1.0s × 30Hz = 30 步
    a = (i + 1) / n_bl
    tk = (1.0 - a) * INIT_TOK + a * np.asarray(tok0[0])  # 线性插值
    p.cmd.token_frame(tk, ...)
```
`vla_round.py:648-656`——只要走 async 路径就一定执行,不受 `--ensemble` 控制。

## 三者怎么拼在一起

1. **异步推理**(第 2 节)是骨架——它创造出"worker 慢慢算、主线程从不停"这个结构,没有它,
   后面两个机制都无从谈起(阻塞式 client 单线程,压根没有"信箱"和"独立时钟"这回事)。
2. **实时分块索引**(第 1 节)和**时序集成**(第 3(a) 节)是**互斥的两选一,不是叠加**——
   代码里是 `if ENSEMBLE_N > 1: 用投票 else: 用延迟补偿单 chunk 索引`
   (`vla_round.py:704-729`)。默认(`ENSEMBLE_N=1`)走第 1 节那条路;显式 `--ensemble 3`
   才会切到第 3(a) 节的投票,这时候"该从第几帧开始播"这个问题直接被"混合所有活着的 chunk"
   取代了,不会同时用两套逻辑。
3. 第 3(b) 节的 1 秒淡入淡出交接独立于以上两者,只在每局**开场**触发一次,之后的整局都是
   第 1/3(a) 点的地盘。

## 第 1 节和第 3(a) 节:二选一还是能叠加?

代码上严格二选一(上面第 2 点),但数学上第 1 节其实是第 3(a) 节的**特例**:
`_ensemble()` 给缓冲区里每个 chunk 算"此刻该播第几帧"用的是同一条公式
`(now - t0) × rate`([vla_round.py:118](src/teleop_free/vla_round.py#L118)),只是在 N 个 chunk 上
各算一遍再加权平均;N=1 时混合一个东西等于什么都没混,跟第 1 节的单 chunk 索引是同一件事。

分岔成两条代码路径,是因为"chunk 播完了怎么办"这件事需要不同语义:官方 `latency_index`
越界时**夹到最后一帧**,适合"永远只有一个当前 chunk"的场景;`_ensemble()` 自己重写的取整
越界时**直接丢弃**,是因为多 chunk 投票下不能让一个早过期的旧 chunk 被强行拖回来稀释新鲜
预测的权重([vla_round.py:150-154](src/teleop_free/vla_round.py#L150-L154))。

默认走单 chunk 路径、时序集成要显式 `--ensemble N>1` 才开,是实验设计取舍,不是技术做不到
叠加——实测效果依场景反转(阶梯基准上 −13 点 p=0.060,并行/时序退化场景下反而 +20 点
p=0.012,见 [evaluation.md](evaluation.md) 终表一节),所以没有转正当默认。

## 生成后处理 vs 生成时引导:两条 RTC 实现路线

以上三个机制,连同 Physical Intelligence "Real-Time Chunking"(RTC,
[arXiv:2506.07339](https://arxiv.org/abs/2506.07339))论文提出的做法,本质上都是在解决
同一个问题——但走的是两条完全不同的路线。这个项目走的是**生成后处理**:先让模型各自独立
吐出完整 chunk,再在动作数值空间里做事后缝合。GR00T 官方模型(`gr00t_n1d7`)的去噪循环里
其实已经原生实现了**生成时引导**这条更接近 RTC 论文本意的路线,只是这个项目的调用链目前
没有接通它。两条路线对照如下。

### 实现原理对比

| 维度 | 生成后处理路线(这个项目实际用的) | 生成时引导路线(GR00T 官方模型自带,未接通) |
|---|---|---|
| **核心思路** | 模型各自独立生成完整 chunk,生成完之后在**动作数值空间**里做时间对齐 + 加权平均/插值缝合 | 把"必须跟上一个 chunk 衔接"当成**生成过程本身的约束**,冻结区不参与更新、自由区在**每一步去噪**里都能通过注意力"看见"冻结区 |
| **混合发生在哪个阶段** | 模型推理**结束之后**(post-hoc) | 模型推理**进行之中**(每个 denoising timestep 内部) |
| **对模型的要求** | 模型是黑盒,只要吐出 `(token, gripper)` chunk 即可 | 模型必须原生支持——去噪循环要接受"上一个 chunk 尾部"和几个超参作为同一次生成调用的输入 |
| **谁能实现** | 客户端(拿到 chunk 之后的任何一方)都能做 | 只有模型/推理服务端能做,客户端伸不进去 |
| **连续性怎么保证** | 加权平均抹平台阶,或线性插值过渡——数值层面缝合 | 冻结值锁定 + 指数斜坡权重,自由区从生成第一步起就知道前面是什么 |
| **越界/边界处理** | 官方 `latency_index` 越界夹到最后一帧;`_ensemble()` 自己重写的取整越界直接丢弃——两种语义因用途不同分裂成两条代码路径 | 冻结区恒定不变(`vel_strength=0`);冻结区与自由区之间用指数斜坡渐变,没有硬边界 |
| **分布内/外风险** | 多个独立结果做算术平均,预测差异大时平均值可能落在分布之外 | 结果全程由模型自己的去噪轨迹产出,理论上始终落在动作流形上 |
| **训练要不要配合** | 不需要,纯推理期后处理 | 理想情况需要——模型若训练时没见过"被冻结前缀锁住",推理时第一次遇到会打折扣 |
| **这个项目里的状态** | ✅ 实际在跑(`--client async` 路径) | ⚪️ 模型原生支持,调用链在 `Gr00tPolicy._get_action()` 这层被切断,结构上不可达 |

### 实现代码位置对比:生成后处理路线(`vla_round.py`,客户端)

| 机制 | 文件:行号 | 关键代码 |
|---|---|---|
| 延迟补偿索引 | [vla_round.py:721-729](../src/teleop_free/vla_round.py#L721-L729) | `idx = latency_index(time.time() - chunk_t0, RATE, len(tok))` |
| ACT 式时序集成 | [vla_round.py:143-166](../src/teleop_free/vla_round.py#L143-L166) | `_ensemble()`——`i = round((now - t0) * rate)`,多 chunk 加权平均 |
| 开局 1 秒线性插值 | [vla_round.py:648-656](../src/teleop_free/vla_round.py#L648-L656) | `tk = (1-a) * INIT_TOK + a * tok0[0]` |
| Worker/主线程异步骨架(承载以上三者) | [vla_round.py:596-617](../src/teleop_free/vla_round.py#L596-L617) | `def worker(): ... latest["chunk"] = (tok, grip)` |
| 单次 chunk 获取(黑盒调用点) | [vla_round.py:330-335](../src/teleop_free/vla_round.py#L330-L335) | `PolicyLink.get_chunk()` → ZMQ 请求,拿回一个完整 chunk |

### 实现代码位置对比:生成时引导路线(`gr00t_n1d7.py`,模型/服务端)

| 机制 | 文件:行号 | 关键代码 |
|---|---|---|
| 冻结前缀 + 指数斜坡 + 联合去噪 | [gr00t_n1d7.py:358-394](../src/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L358-L394) | `Gr00tN1d7ActionHead.get_action_with_features`——`vel_strength[:, :frozen_steps]=0`,指数 `ramp` |
| 模型层入口(RTC 开关) | [gr00t_n1d7.py:446-475](../src/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L446-L475) | `get_action(backbone_output, action_input, options)`——`options` 里的 `rtc_*` 四个 key |
| 训练侧配套(本地新加,默认关) | [gr00t_n1d7.py:239-261](../src/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L239-L261) | `RTC_TRAIN_P`/`RTC_TRAIN_KMAX`——随机冻结前缀、不计入 loss |
| **调用链断点**(RTC 因此走不到) | [gr00t_policy.py:380-417](../src/gr00t/policy/gr00t_policy.py#L380-L417) | `_get_action()` 文档写"options currently unused",实际调用 `self.model.get_action(**collated_inputs)` 时压根不传 `options` |

两条路线的分界线正好是**客户端(vla_round.py)vs 模型/服务端(gr00t_n1d7.py)**这个进程边界:
`serve_policy.py` 把模型包成一个只接受"当前观测"的黑盒端点,`vla_round.py` 因此天生够不到
生成过程内部,只能在拿到成品之后做数值层面的缝合;而 GR00T 官方模型的去噪循环里其实已经
预留好了接口,只是从 `Gr00tPolicy` 这一层开始就没人把线接过去。真要接通,`gr00t_policy.py`
(转发 `options` + 传入上一个 chunk)、`serve_policy.py`(新增接收上一个 chunk + rtc 超参的
能力)、`vla_round.py`(worker 已经在追踪 `chunk_t0`,可以据此换算 `rtc_frozen_steps`)三处
都要动。
