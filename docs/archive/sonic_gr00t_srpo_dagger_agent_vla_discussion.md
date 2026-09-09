# SONIC + GR00T / SRPO / DAgger / Agent-based VLA 讨论记录

> 导出范围：从“这篇paper：https://arxiv.org/pdf/2511.15605v1 的贡献是什么，为啥它能200 steps就训练好”开始，到关于 agent-based 与 VLA 路线的讨论为止。

---

## User

这篇paper：https://arxiv.org/pdf/2511.15605v1 的贡献是什么，为啥它能200 steps就训练好

## Assistant

这篇是 **SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models**，后来已经被 **CVPR 2026** 接收。它最核心的贡献其实很简单：

> **让 VLA 做 RL 时，失败的 rollout 也能提供有用的学习信号。**

而你注意到的“**200 steps 就训练好**”确实很夸张，但这里有一个非常重要的文字游戏：**这不是从零开始训练 200 steps，而且 1 个 RL step 也绝不是 1 次机器人 interaction。**

### 1. 它到底解决什么问题？

传统 VLA + RL，例如 GRPO，大致是：

```text
instruction + image
      ↓
     VLA
      ↓
 rollout trajectory
      ↓
成功 → reward = 1
失败 → reward = 0
```

问题在于，机器人任务失败有很多种：

```text
A: 完全没碰到杯子                 reward = 0
B: 抓到杯子但掉了                 reward = 0
C: 抓到杯子，移动到目标附近        reward = 0
D: 放进去但是最后碰倒了            reward = 0
E: 成功                           reward = 1
```

GRPO 看起来：

```text
A = B = C = D = 0
E = 1
```

所以 **A/B/C/D 这些失败 trajectory 里面的大量信息全浪费掉了**。论文认为这是 VLA RL 样本效率低的主要原因之一。

---

### 2. SRPO 的关键 idea：拿“自己成功的 trajectory”当老师

它并不需要人为规定：

```text
靠近杯子      +0.1
碰到杯子      +0.2
抓起来        +0.3
移动到目标    +0.4
放进去        +1
```

也不需要额外 expert trajectory 来计算 progress。

它做了一件很聪明的事情：

```text
同一个 task
        ↓
VLA rollout 8 次
        ↓
 ┌───────────────┐
 │ success traj 1│
 │ success traj 2│  ← 自己产生的 successful trajectories
 │ failure traj 1│
 │ failure traj 2│
 │ ...           │
 └───────────────┘
        ↓
      V-JEPA 2
        ↓
trajectory latent representation
```

成功轨迹变成 **self-reference**。

假设：

```text
成功 trajectory latent
        ● ●
       ●   ●

失败 A                         ×
失败 B                 ×
失败 C           ×
失败 D       ×
```

那么越接近成功 trajectory cluster：

```text
D > C > B > A
```

就给更高 reward。

具体来说，他们用 **V-JEPA 2** 这个已经在大量视频上预训练好的 latent world model，把整个 trajectory 编码成 latent representation，然后：

1. 对 successful trajectory embeddings 做 DBSCAN clustering；
2. 找 successful clusters 的 representative centers；
3. 对 failed trajectory 计算到最近 success center 的 L2 distance；
4. distance 越小 → progress reward 越高。

于是：

```text
完全失败           reward ≈ 0.05
稍微接近           reward ≈ 0.25
抓起来了           reward ≈ 0.55
差一点成功         reward ≈ 0.85
成功               reward = 1.0
```

再用这些 reward 算 GRPO-style advantage。

所以可以把 SRPO 简单理解成：

> **GRPO + 自己成功的 trajectory 当 reference + world-model latent 计算“你离成功还有多远”。**

---

### 3. 这里有个容易误解的地方：它并不是严格的 step-wise dense reward

论文叫它 **progress-wise reward**，容易让人以为：

```text
t=1 reward
t=2 reward
t=3 reward
...
```

其实它在主要 LIBERO simulation 实验里，更接近：

> **trajectory-level progress reward**

它把整个 trajectory \(o_{0:T}\) 输入 world-model encoder 得到一个 representation \(h_i\)，然后给这整个 rollout 一个 \(g_i\)。论文自己也明确说 SRPO 是 **trajectory-level advantage estimation**。

所以不是传统 robotics reward shaping 那种：

```text
distance(object, gripper)
distance(object, target)
contact reward
grasp reward
```

而是：

```text
这个完整行为轨迹
和成功行为轨迹有多像？
```

这一点我觉得是这篇最有意思的地方。

---

### 4. 那为什么它“200 steps 就训好了”？

这里就是重点。

#### 原因一：它根本不是从零开始

它的 RL training 是从：

**pretrained OpenVLA → one-shot SFT → SRPO**

开始的。

完整 pipeline 实际上是：

```text
Pretrained OpenVLA
      ↓
Action Chunking + Parallel Decoding
      ↓
One-shot SFT
      ↓
48.9% success
      ↓
SRPO online RL
      ↓
99.2%
```

而且这个 **one-shot SFT 本身训练量一点都不小**。

Appendix 明确写了：

- 8 × A100
- AdamW
- batch size = 8
- LoRA rank = 32
- **maximum SFT steps = 150,005**
- 然后才开始所谓的“200-step SRPO”。

所以：

> ❌ 200 steps 训练一个 VLA  
> ✅ 一个已经 pretrained + SFT 好、成功率已经 48.9% 的 VLA，再进行约 200 次 RL update

这两个概念差非常远。

---

### 5. 更关键：一个“RL step”里面有很多很多 trajectory

他们的 SRPO 参数是：

- Training batch size = **64**
- Samples per group = **8**
- action chunk = 8
- number of trials per task = 50

官方 siiRL 的 SRPO reproduction script 也明确写了：

```text
TRAIN_BATCH_SIZE = 64
ROLLOUT_N_SAMPLES = 8
```

也就是说一个 training iteration nominally 会生成：

\[
64\times8=512
\]

个 rollout samples。

所以“200 RL steps”粗略理解是：

\[
200\times512=102,400
\]

**trajectory rollouts**。

不是：

```text
机器人动 200 次
```

而更接近：

```text
rollout 几百条完整 trajectory
       ↓
算 reward / advantage
       ↓
update 一次
       ↓
这叫 1 step

重复 ~200 次
```

这就完全不神奇了。

官方 reproduction configuration 甚至设置：

```text
NUM_ENVS = 16
MAX_EPISODE_STEPS = 512
```

并通过大量 parallel environment 来产生这些 rollout。

所以它实际上是在用**极强的并行仿真吞吐量换 optimizer step 数量**。

---

### 6. 为什么 SRPO 仍然确实比 GRPO 快很多？

虽然“200 steps”有宣传成分，但它的方法本身确实提高了 sample efficiency。

最关键是：

#### GRPO

假设一次有 8 个 rollout：

```text
traj 1    success    1
traj 2    fail       0
traj 3    fail       0
traj 4    fail       0
traj 5    fail       0
traj 6    fail       0
traj 7    fail       0
traj 8    fail       0
```

实际上只有：

```text
成功 vs 失败
```

的信息。

#### SRPO

变成：

```text
traj 1    success          1.00
traj 2    almost success   0.91
traj 3    grasped          0.72
traj 4    approached       0.48
traj 5    wrong grasp      0.31
traj 6    bad approach     0.18
traj 7    hardly moved     0.07
traj 8    completely wrong 0.02
```

于是一次 rollout batch 的监督信息多很多。

论文也直接比较了 SRPO 与 GRPO，发现 SRPO learning curve 更陡，尤其 **LIBERO-Long**；作者把原因归因于 GRPO 基本浪费 failure episode，而 SRPO 能从 near-success trajectories 中提取信号。

---

### 7. 还有一个非常重要的原因：48.9% 的初始化特别适合 self-reference

这个其实论文没有特别强调，但从算法机制看非常关键。

它开始 RL 时已经有：

**48.9% success rate。**

而每个 group rollout：

**8 trajectories。**

所以一个 group 里通常很容易出现至少一条 success trajectory。

如果粗略假设每次 rollout 成功概率独立且都是 48.9%，那么 8 个里面一个成功都没有的概率只有：

\[
(1-0.489)^8\approx0.47\%
\]

换句话说大约：

**99.5% 的 group 至少有一个成功 sample。**

虽然实际任务间成功率并不相同，所以不能把这个数字当严格实验统计，但它很好地解释了为什么这个机制特别有效：

```text
初始 policy 已经“半会了”
          ↓
每次 rollout 很容易产生 success
          ↓
success 可以成为 reference
          ↓
给其他 failure 排序
          ↓
快速把“差一点成功”推成成功
```

所以 SRPO 本质上不是：

> “教一个完全不会做任务的机器人学会任务。”

而更像：

> **“一个已经有接近 50% 成功率的机器人，通过自己的成功经验，非常迅速地修正剩下的失败模式。”**

这也是为什么 200 updates 能产生这么大的提升。

---

### 8. 实际上甚至不是所有 task 都正好“200 steps”

论文 Section 5.2 给出的具体数字是：

| LIBERO suite | SRPO RL steps |
|---|---:|
| Spatial | **79** |
| Object | **59** |
| Goal | **103** |
| Long | **219** |

所以 abstract 里面“**just 200 RL steps**”其实是一个概括性的宣传说法。

最难的 **LIBERO-Long 实际是 219 steps**。

最终：

```text
One-shot SFT
48.9%

        ↓ SRPO

Spatial 98.8
Object  100.0
Goal    99.4
Long    98.6

Average 99.2%
```

---

### 9. 所以我觉得这篇真正值得关注的贡献不是“200 steps”

我会把贡献重要性排序成：

**第一：self-referential reward**

不需要人工 dense reward，也不需要额外 expert trajectory：

```text
自己 rollout
   ↓
自己的 success 当 positive reference
   ↓
自己的 failure 和 success 比较
```

这是很 scalable 的 idea。

**第二：用 world-model latent 当 reward space**

不是 RGB pixel similarity：

```text
pixel similarity ❌
CLIP/ImageBind semantic similarity ❌
V-JEPA2 world representation ✅
```

他们的实验也显示 ImageBind reward 很快 plateau 在约 85%，而 V-JEPA latent reward 可以继续提升。

**第三：把大量 failure rollout 重新变成训练数据**

这才是 sample efficiency 大幅提高的根本。

---

所以你看到这篇的时候，我建议你脑子里不要记成：

> **“SRPO 只需要 200 个 samples 就能训练 VLA。”**

而应该记成：

> **“对于一个已经 pretrained + 最高 150k-step one-shot SFT、已有约 49% 成功率的 VLA，SRPO 通过每次约 64×8 条并行 rollout 的 self-referential RL，在约 60–220 个 optimizer iterations 内把成功率推到接近 100%。”**

这样就合理多了。

而且这篇和你最近关注的 **simulation 里让 agent 自己产生数据/探索，而不是 scripted privileged controller 产生 demonstration** 其实非常相关：SRPO 提供的正是一个很有意思的机制——**不用 privileged dense reward，也不用 expert demonstrations，而是让 agent 已有的 successful rollouts 自动成为后续 exploration 的 reward reference**。这条思路其实比“200 steps”这个数字更值得借鉴。

---

## User

我最近在训练sonic+gr00t的人形机器人的locomanipulation策略，出现了采集200 episodes训练，open-loop的指标很高，但是closed-loop在simulation中evaluation的时候，老是走到桌子边就愣住了，不抓起来，另一个ai跟我说的是要用dagger解决这个问题，然后我在想是不是也可以用这个SRPO解决？

## Assistant

可以用 **SRPO**，但针对你现在这个非常具体的症状，我会把优先级排成：

**先 DAgger / targeted corrective data → 再考虑 SRPO 做 post-training。**

因为你这个：

> 200 episodes BC/SFT → open-loop 很高 → closed-loop 走到桌边 → 停住不抓

几乎就是 textbook 的 **covariate shift + recovery data 缺失**。

而 SRPO 解决的是另一个问题：**已经在做 online RL，但 binary success reward 太 sparse，失败 trajectory 没有学习信号**。它通过当前 policy 自己的 successful rollout 给 failed rollout 提供 progress reward。

所以二者不是互斥，反而很适合串起来。

---

### 先看你现在为什么“走到桌边就愣住”

你这个现象特别像下面这个过程：

```text
demonstration:

正常走近桌子
    ↓
停在一个非常标准的位置
    ↓
身体 / 手臂姿态正确
    ↓
reach
    ↓
grasp
```

你采的 200 episodes 很可能大量都是这种 **clean expert trajectory**。

GR00T 学到：

\[
p(a_t|o_t)
\]

在 demonstration distribution 上很好：

```text
expert state
     ↓
GR00T
     ↓
expert-like action
```

所以 open-loop：

```text
GT observation
GT observation
GT observation
GT observation
       ↓
action prediction

accuracy 很高
```

完全正常。

但 closed-loop：

```text
GR00T action
     ↓
SONIC execution
     ↓
一点 position error
     ↓
下一帧 observation 偏离 demo
     ↓
GR00T 再产生一点 error
     ↓
SONIC
     ↓
误差累积
```

最后机器人可能变成：

```text
        table
───────────────
        mug

 humanoid
    ↓
离 demonstration 中标准 pre-grasp pose
差 5 cm / 10 cm
身体 heading 差几度
arm pose 也不完全一样
```

而你的 dataset 里：

> **根本没有这个 state 对应“下一步应该怎么办”的 label。**

于是 GR00T 最安全的预测往往就是：

```text
small action
small action
small action
≈ no-op
```

看起来就像：

> **“走到桌子旁边突然愣住了。”**

这正是 DAgger 针对的问题。

---

### 为什么我会先用 DAgger

DAgger 的逻辑恰好就是：

```text
你现在的 GR00T policy
        ↓
closed-loop rollout
        ↓
自己走到那个尴尬的位置
        ↓
不要 reset
        ↓
让 expert/controller 接管
        ↓
记录：

「在这个 state 下，正确 action 是什么」
```

于是原始数据：

```text
D0 = 200 clean demonstrations
```

变成：

```text
D1 =
200 clean demos

+

policy 自己真正会访问的 states
           ↓
expert corrective actions
```

再训练：

```text
GR00T(D0 + D1)
```

这次它终于见过：

```text
离桌子太远一点怎么办
离桌子太近怎么办
身体偏左怎么办
手没有对准怎么办
脚的位置不理想怎么办
reach 开始晚了怎么办
```

这就是最直接地解决你的 failure mode。

---

### 那 SRPO 能不能解决？

**能，而且理论上很适合你，但它解决问题的方式完全不同。**

假设你的 rollout 现在是：

```text
A:
走到桌边
→ 停住
→ fail

B:
走到桌边
→ 手稍微伸出去
→ fail

C:
走到桌边
→ reach
→ 碰到物体
→ fail

D:
走到桌边
→ grasp
→ 掉了
→ fail

E:
走到桌边
→ grasp
→ lift
→ success
```

binary RL：

```text
A = 0
B = 0
C = 0
D = 0
E = 1
```

SRPO 想做的是：

```text
E = successful self-reference
          ↑
D  很接近
          ↑
C
          ↑
B
          ↑
A  很远
```

于是：

```text
A 0.1
B 0.3
C 0.55
D 0.85
E 1.0
```

然后 policy gradient 会倾向于：

```text
freeze at table
       ↓
reach
       ↓
contact
       ↓
grasp
       ↓
lift
```

所以从这个角度讲：

> **是的，SRPO 很有可能把你的“桌边 freeze”推过去。**

---

### 但你这里有一个 SRPO 比 DAgger 麻烦很多的问题

SRPO 有个隐藏前提：

> **你的 current policy 必须偶尔能成功。**

这非常重要。

假设你的当前 policy：

```text
100 rollout

95:
走到桌边 → freeze

5:
reach → 但全部失败

0 success
```

那 SRPO 就尴尬了：

```text
successful self-reference = ∅
```

没有 reference。

它没法告诉你：

> freeze → reach

到底哪个方向是正确的。

反过来，如果你的情况是：

```text
100 rollout

60 freeze
20 touch
10 grasp fail
10 success
```

那 SRPO 就特别有价值。

因为它已经有：

```text
10 successful trajectories
```

可以告诉其他 90 条 trajectory：

> **你们离成功有多远。**

因此有一个非常实用的判断：

#### 如果 closed-loop success ≈ 0–5%

我会选：

**DAgger。**

#### 如果 closed-loop success 已经有 20–50%

那我会认真考虑：

**SRPO / GRPO-style online RL。**

---

### 而且你是 SONIC + GR00T，不是普通 LIBERO

这里我觉得要特别谨慎。

你的 action hierarchy 更像：

```text
                GR00T
                  ↓
        high-level whole-body command
                  ↓
                SONIC
                  ↓
         joint-level execution
                  ↓
        humanoid physical state
```

所以你的 closed-loop distribution shift 是两层产生的：

\[
\pi_\text{GR00T}
\rightarrow
\pi_\text{SONIC}
\rightarrow
s_{t+1}
\]

GR00T 预测：

```text
desired hand/body motion
```

但 SONIC 真执行出来：

```text
desired + tracking error
```

然后下一次：

```text
GR00T observation
```

就已经不再是 demonstration distribution。

这比固定机械臂 LIBERO 严重得多。

---

### 我甚至有点担心直接照搬 SRPO 的 V-JEPA reward

这个很关键。

例如两个状态：

#### State A

```text
机器人站桌前
手距离杯子 20 cm
身体稳定
下一步可以 reach
```

#### State B

```text
机器人站桌前
手距离杯子 20 cm
但是：
重心不稳定
左脚位置不对
腰部 heading 不对
SONIC 接近 joint limit
```

RGB 看起来可能非常像。

V-JEPA latent 可能：

\[
d(z_A,z_B)
\]

也不大。

但对于 humanoid controller 来说：

```text
A → grasp 很容易

B → 必须先 reposition
```

这两个状态其实差很多。

因此在你的任务里，我甚至不会第一版就完全复制：

> V-JEPA2 trajectory distance

而会考虑一个 **robot-aware progress representation**。

例如：

\[
z_t =
[
z^\text{visual}_t,
q^\text{base}_t,
q^\text{hand}_t,
p^\text{EE}_t,
p^\text{object}_t,
c^\text{foot}_t
]
\]

也就是：

```text
visual latent
+
base pose / heading
+
EE pose
+
object pose
+
contact
+
可能还有 COM / foot states
```

然后再做：

```text
successful trajectory embedding
         ↑
progress distance
```

我觉得这甚至可能比原始 SRPO 更适合 humanoid loco-manipulation。

---

### 我会怎么做你这个实验

如果是我现在手上就是你这套 SONIC + GR00T，我不会马上重新采另外 200 条 clean demos。

我会直接拿现在的 checkpoint 做：

```text
200 BC episodes
        ↓
现有 GR00T
        ↓
simulation rollout
        ↓

专门保存 failure states
```

比如你发现：

```text
60%:
table edge freeze

20%:
reach miss

10%:
grasp miss

10%:
success
```

然后做一个非常简单的 **DAgger-lite**：

```text
policy rollout
      ↓
走到 freeze state
      ↓
pause
      ↓
从这个 exact simulator state
调用你的 expert
      ↓
继续完成：

reposition
reach
grasp
lift
```

你甚至不一定需要完整传统 iterative DAgger。

比如：

```text
200 original episodes

+

50 个 freeze recovery trajectory
+
30 个 reach-miss recovery
+
20 个 grasp-miss recovery
```

得到：

```text
300 episodes
```

再 finetune GR00T。

我猜这会比：

```text
重新采另外 200 个干净成功 demo
```

有效非常多。

---

### 然后我会再做 SRPO

我认为最有意思的 pipeline 其实是：

```text
                200 expert demos
                       ↓
                     SFT
                       ↓
                 GR00T + SONIC
                       ↓
              closed-loop rollout
                 ↙           ↘
              failure       success
                 ↓              ↓
          DAgger correction     │
                 ↓              │
             再训练             │
                 ↓              │
         policy success ↑       │
                        \       /
                         \     /
                          SRPO
                            ↓
                  self-referential RL
                            ↓
                    final policy
```

也就是：

> **DAgger 负责把 policy 拉进“会做”的 basin；SRPO 负责在这个 basin 里面继续优化。**

我觉得这比二选一更合理。

---

### 从研究角度，这甚至是个挺值得做的方向

你的场景刚好暴露 SRPO 的一个局限：

SRPO 假设：

```text
successful visual trajectory
≈
good policy reference
```

但是在 humanoid loco-manipulation 里面：

```text
视觉 progress
≠
physical feasibility progress
```

因为还有：

```text
balance
foot contact
base pose
joint configuration
SONIC tracking feasibility
```

所以很自然可以变成：

```text
SRPO
Self-reference in visual world latent
```

↓

```text
Humanoid SRPO

Self-reference in
visual + proprioception + contact
+ low-level controller feasibility latent
```

甚至 reward 可以是：

\[
R =
R_\text{task-progress}
+
\lambda_1R_\text{SONIC-feasibility}
+
\lambda_2R_\text{stability}.
\]

这会比单纯“把 SRPO 搬到 GR00T”更有研究味道。

---

**所以对于你现在这个具体 bug，我的判断是：**

> **先别急着上 SRPO。你 open-loop 很高、closed-loop 在一个固定阶段 freeze，是非常强的 DAgger 信号。先针对 freeze state 收 20–50 条 corrective/recovery trajectories，我非常想看 closed-loop success 会不会直接大幅上升。**

如果 DAgger 后你的 policy 已经能做到比如 **30–60% closed-loop success，但剩下的 failure 各种各样**，这时候再上 SRPO，我认为会非常合适。

而且因为你是在 simulation 里面，你有一个巨大的优势：**可以保存那个“走到桌边愣住”的 simulator state，然后反复从同一个 state fork 出 N 个 rollout。** 这样我们甚至可以设计一个比标准 DAgger 更省数据的 **failure-state branching + SRPO** 方法——我觉得这个方向比单纯复制 SRPO 更适合你现在的 SONIC+GR00T setting。

---

## User

因为我以为我收集了200episodes之后拿去训练，就一定能得到好的泛化效果，但是现在不是这样，但我需要简洁地跟别人解释一下，我应该怎么解释

## Assistant

你可以很简洁地解释成：

**“200 episodes 只能保证模型在 demonstration distribution 上拟合得不错，但不保证 closed-loop 泛化。实际 rollout 时，模型自己的小误差会不断累积，使机器人进入训练数据里没见过的状态；一旦发生 distribution shift，策略就不知道该怎么恢复，所以会出现走到桌边停住这种现象。”**

如果想再专业一点：

**“问题不在于 demonstration 数量本身，而在于 state coverage。我们的 200 条数据大多是成功、干净的 expert trajectories，所以 open-loop 指标很高；但 closed-loop 时存在 compounding error，policy 会访问 off-distribution states，而训练集中缺少这些状态下的 corrective/recovery actions，因此性能会明显下降。”**

一句话版本就是：

> **High open-loop accuracy does not imply good closed-loop performance, because behavior cloning suffers from covariate shift and compounding errors.**

所以别人问“为什么 200 episodes 还不够”，你可以直接说：

> **不是 200 条太少，而是这 200 条覆盖的状态不对；更多重复的成功轨迹，不一定比少量 failure/recovery trajectories 更有价值。**

这也是为什么 DAgger 对你现在的问题很对症。

---

## User

如何理解dagger这个词，为啥dagger就是专门用来处理这种问题的方法

## Assistant

DAgger 这个词本身就已经把它的方法说出来了：

**DAgger = Dataset Aggregation，数据集聚合。**

它不是“匕首 dagger”这个意思，而是一个缩写。

核心思想非常简单：**不要只用 expert 原本会访问到的状态训练，而是把“当前 policy 自己实际会走到的状态”也加入训练集。**

比如你现在的普通 BC 是：

```text
Expert demonstration:
s1 → s2 → s3 → s4 → grasp
```

你采 200 条数据，本质上都是 expert 走出来的状态：

\[
D=\{(s,a^*)\},\quad s\sim d_{\text{expert}}
\]

所以模型只学过：

> “如果我站在 expert 会站的位置，我下一步应该怎么做。”

但 closed-loop rollout 时，policy 自己可能变成：

```text
s1 → s2' → s3'' → 桌子旁边一个奇怪的位置
```

这里的 \(s_3''\) 根本不在原来的 demonstration dataset 里。

于是就出现你现在这个问题：

```text
training:
"我从没见过这个状态"

deployment:
"但我偏偏自己走到了这个状态"
```

这就是 **covariate shift**。

---

DAgger 做的事情就是：

```text
1. 先用 expert demo 训练一个 policy π

2. 让 π 自己 closed-loop rollout

3. π 会犯错，进入它自己造成的状态 s'

4. 在这些 s' 上问 expert：
   “如果是你，现在应该怎么做？”

5. 得到 (s', a*_expert)

6. 把这些新数据加入原 dataset

7. 重新训练
```

所以叫 **Dataset Aggregation**：

\[
D_0
\rightarrow
D_0 \cup D_1
\rightarrow
D_0 \cup D_1 \cup D_2
\rightarrow ...
\]

这里最关键的是，后面的 \(D_1,D_2\) 不是新的“漂亮 expert demonstration”，而是：

> **policy 自己会访问到的错误状态 + expert 在这些状态下的正确动作。**

---

这也是为什么 DAgger 是专门为你这种问题设计的。

普通 BC 的根本矛盾是：

\[
\text{training states}\sim d_{\text{expert}}
\]

但部署时：

\[
\text{test states}\sim d_{\pi}
\]

两个 distribution 不一样。

而 DAgger 的目标就是让训练数据逐渐变成：

\[
s\sim d_{\pi}
\]

也就是说：

> **你最终在哪些状态上运行，我就在哪些状态上教你。**

这句话基本就是 DAgger 的本质。

---

你可以拿你现在的例子理解。

原来的 200 episodes 可能都是：

```text
走到桌子前 30 cm
→ 身体朝向正
→ 手臂展开
→ reach
→ grasp
```

但你的 GR00T+SONIC 真正 rollout：

```text
走到桌子前 22 cm
→ 身体偏了 8°
→ 左手高度低了 4 cm
→ 然后 freeze
```

普通 BC 数据里可能完全没有这种状态。

DAgger 就会专门收：

```text
Observation:
机器人距离桌子 22 cm
heading 偏 8°
手臂当前位置 ...

Expert action:
先后退一点
身体旋转
调整 arm
然后 reach
```

训练以后，模型就会学到：

> “哦，原来我如果自己走歪了，还能这么救回来。”

所以 DAgger 学的其实不仅是 **task execution**，更重要的是 **error recovery**。

---

为什么普通 BC 特别容易出这个问题，还有一个经典原因叫 **compounding error**。

假设每一步 policy 只有 1% 概率犯错。

open-loop 看起来：

\[
99\% \text{ accuracy}
\]

很高。

但一个 episode 有 200 步，至少出现一次错误的概率是：

\[
1-0.99^{200}\approx86.6\%
\]

所以：

> **每一步都很准，不代表整条 closed-loop trajectory 会成功。**

而且第一次小错之后：

```text
进入 OOD state
→ 下一步更容易错
→ 更 OOD
→ 更容易错
```

误差不是独立的，而是会滚雪球。

这就是为什么你现在会看到：

> open-loop metric 很漂亮，但 closed-loop 突然卡死。

而 DAgger 就是在打断这个滚雪球过程。

---

最简洁地理解 DAgger，可以记这句话：

> **BC teaches the policy what to do when everything goes right; DAgger teaches it what to do after it has made a mistake.**

对应到你的项目里：

> **你那 200 条 demonstration 教会了 GR00T“正常情况下怎么抓”；DAgger 要补的是“SONIC 执行有偏差、自己走歪以后怎么继续抓”。**

所以它和你的 failure mode 是高度匹配的。

---

## User

我现在在想，现在的VLA路线逐渐进化到agent-based的机器人动作执行，agent-based通常可以在simulation中执行得非常好，因为simulation可以给到很多准确的privileged knowledge，然后agent可以调用一些原子技能或者函数来执行动作，但是在真实世界里好像还是还是得靠VLA，因为无法获得这些privileged knowledge之后，只能通过视觉获得信息，而这些信息又太模糊隐晦了，只能用VLA学到end-to-end的技能，我理解的对吗，或者需要纠正吗

## Assistant

你的理解**大方向是对的，但有一个关键点需要纠正**：

> **Agent-based 和 VLA 其实不是两条互斥路线。**
>
> Agent-based 描述的是“机器人怎么组织决策”；VLA 描述的是“机器人怎么从感知得到动作”。  
> 真正有希望的路线很可能是 **Agent + VLA + skill/controller 的 hierarchical system**，而不是 Agent 最后被 VLA 替代。

你观察到的 simulation / real-world 差距确实是真问题。

---

### 1. 为什么 simulation 里的 agent 看起来特别强？

simulation 可以直接告诉 agent：

```text
object_pose = [x,y,z]
object_id = "red cup"

gripper_pose = [...]
drawer_joint = 0.37
contact = True
distance_to_target = 0.12
```

然后 agent 可以直接：

```text
find_object("cup")
navigate_to(object_pose)
grasp(object_pose)
lift()
place(target_pose)
```

这里 agent 真正要解决的主要是：

> **What should I do next?**

而不是：

> **Where exactly is the cup?**
> **Did I actually grasp it?**
> **How far is my hand from it?**
> **Is the drawer actually open?**

所以很多 simulation agent paper 实际上把最难的 **perception / grounding** 问题绕过去了。

这也是为什么最近 REAL 这篇工作专门强调，他们要去掉 privileged simulator state，构建 **sim-to-real-consistent APIs without oracle perception**；最后才真正能在物理双臂移动机器人上测试 agent。

---

### 2. 但真机并不是“没有 privileged state → 只能 VLA”

这是我最想纠正你的地方。

真实世界虽然没有：

```text
get_object_pose()
get_contact_state()
```

这样的 simulator oracle，但我们仍然可以通过 perception 把这些量**估计出来**：

```text
RGB / RGB-D / tactile / proprioception
               ↓
        perception modules
               ↓
 object detection
 segmentation
 depth
 6D pose
 SLAM / mapping
 affordance
 contact estimation
               ↓
        estimated world state
               ↓
              Agent
```

所以完全可以存在一个真实世界 agent：

```text
camera
  ↓
GroundingDINO / segmentation / depth / pose estimation
  ↓
"cup at approximately (x,y,z)"
  ↓
LLM/VLM Agent
  ↓
navigate(cup)
  ↓
grasp(cup)
```

并不需要 VLA end-to-end。

事实上 **OK-Robot** 就是非常典型的例子：真实家庭环境中，用 VLM/open-vocabulary perception + navigation primitive + grasping primitive，而不是一个统一 end-to-end VLA，也能完成真实机器人 pick-and-place。

最新的 **AgenticLab** 也正是在研究真实世界 agent：perception → decomposition → execution → verification → replanning，而不是依赖 simulator privileged state。它反而发现真正的问题在于 occlusion、grounding consistency、spatial reasoning 等 perception-grounding failure。

---

### 3. 那为什么现在大家又这么喜欢 VLA？

因为上面这个 modular pipeline 有一个很严重的问题：

```text
RGB
 ↓
detector
 ↓
segmentation
 ↓
depth
 ↓
6D pose
 ↓
planner
 ↓
grasp planner
 ↓
controller
```

**每一层都会产生误差。**

比如：

```text
真实位置
    ↓
depth error 2cm
    ↓
6D pose error 3cm
    ↓
grasp point error
    ↓
IK error
    ↓
controller tracking error
    ↓
MISS
```

而机器人 manipulation 很多时候偏 **2~3 cm** 就已经失败了。

所以 end-to-end VLA 有一个非常大的优势：

```text
image
  ↓
VLA
  ↓
action
```

它不强迫系统显式回答：

> 杯子的 pose 是 `[0.532, -0.114, 0.782]`

而可以直接学习：

> **看到现在这个视觉状态，手应该往右下方移动一点。**

也就是说 VLA 可以把很多：

- perception uncertainty
- calibration error
- morphology
- contact dynamics
- visual appearance

隐式吸收到 policy 里面。

你说的：

> “视觉信息太模糊隐晦，所以让 VLA end-to-end 学”

这里有一部分确实是这个意思。

但更准确地说不是因为：

> 视觉信息没法显式恢复。

而是：

> **从视觉显式恢复完整、精确、稳定、对控制有用的 state 很难；有时候直接学习 observation → action 反而更鲁棒。**

这是一个重要区别。

---

### 4. 所以我认为未来很可能不是 Agent vs VLA，而是这个结构

我现在比较看好的是：

```text
                     Robot Agent
                         │
              reasoning / planning
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
      navigate          grasp           open
         │               │               │
         └───────────────┼───────────────┘
                         ↓
                skill-conditioned VLA
                         ↓
               continuous actions
                         ↓
                 low-level policy
                         ↓
                       robot
```

比如你的 humanoid 完全可以变成：

```text
                 VLM / Agent
                     ↓
          "先走到桌子前面"
                     ↓
            locomotion skill
                     ↓
                   SONIC

                 VLM / Agent
                     ↓
              "抓起杯子"
                     ↓
                  GR00T
                     ↓
              visual feedback
                     ↓
                  SONIC
```

甚至更细一点：

```text
Agent
 ↓
navigate_to(cup)
 ↓
SONIC

Agent
 ↓
pregrasp(cup)
 ↓
GR00T + SONIC

Agent
 ↓
grasp(cup)
 ↓
GR00T + SONIC

Agent
 ↓
检查：杯子抓起来了吗？
 ↓
VLM / vision verifier

失败
 ↓
replan
```

我觉得这比：

```text
language + image
      ↓
巨大 VLA
      ↓
从头走过去 → 抓 → 搬 → 放
```

要合理得多。

---

### 5. 而且 2025–2026 的趋势其实已经明显在往这里走

例如 **Being-0** 就是在真实 humanoid 上采用：

**Foundation Model high-level cognition + modular skill library + VLM connector**，

让 high-level reasoning 和 locomotion/manipulation skill 分开。

另外，VISTA 这类 2026 工作则采用：

```text
World Model
    ↓
high-level visual subgoal
    ↓
VLA
    ↓
low-level execution
```

它不是要求 VLA 自己同时解决长期 planning 和精确控制，而是让 world model 规划、VLA 执行。

所以实际上已经逐渐出现：

> **Agentic VLA**

而不是：

> Agent **or** VLA。

---

### 6. 真正的问题其实可以这样定义

simulation：

\[
s_t \rightarrow Agent \rightarrow skill
\]

其中 \(s_t\) 是**真实 world state**。

真实世界：

\[
o_t \rightarrow ? \rightarrow Agent \rightarrow skill
\]

这里：

\[
o_t = \text{RGB, depth, tactile, proprioception, ...}
\]

真正困难的是这个问号：

\[
o_t \rightarrow \hat{s}_t
\]

也就是：

> **如何从 noisy observation 得到 sufficiently actionable world representation。**

这里有三种路线：

| 路线 | 方法 |
|---|---|
| Modular | perception → explicit state → agent |
| End-to-end VLA | observation → action |
| **Hybrid** | observation → semantic/latent state → agent → VLA/skill |

我觉得第三条目前最值得关注。

---

### 7. 这和 simulation privileged learning 其实还能连起来

这里有一个我觉得非常有意思的方向。

你完全可以在 simulation 训练：

```text
Privileged Agent

GT object pose
GT contact
GT articulation
GT robot state
       ↓
perfect-ish reasoning
       ↓
skill execution
```

它可以大量产生成功 trajectory。

然后训练另一个 deployable policy：

```text
RGB + proprioception
        ↓
student / VLA
        ↓
imitate privileged agent
```

也就是：

\[
\pi_{\text{privileged}}
\rightarrow
\pi_{\text{vision}}
\]

这样 **privileged information 并不是 deployment requirement，而是 training-time teacher**。

这其实和 locomotion RL 里面非常成熟的 asymmetric actor-critic / privileged teacher → student 思想是一样的。

所以 simulation 的 privileged information **不是问题本身**。

真正的问题是：

> **你有没有设计一个 mechanism，把 privileged agent 的能力 distill 成一个只依赖真实传感器的 deployable policy。**

---

因此，我会把你原来的理解稍微改写成：

你原来的版本：

> Simulation agent 因为有 privileged knowledge 所以很强；real world 没有 privileged knowledge，只能靠视觉，因此最终还是得靠 end-to-end VLA。

我认为更准确的是：

> **Simulation 让 agentic robotics 看起来容易，是因为 privileged state 把 perception/grounding 问题隐藏掉了。真实世界的核心挑战是如何从 noisy sensory observations 获得足够可靠的 actionable state。但这并不意味着必须使用纯 end-to-end VLA；更可能的路线是 Agent 负责 long-horizon reasoning，VLM/world model 负责 perception 和 grounding，VLA 或专用 skill policy 负责难以显式建模的 closed-loop manipulation，low-level controller 负责稳定执行。**

如果压缩成一句话，我觉得可以记：

> **Agent solves “what to do”; VLA helps solve “how to do it from imperfect sensory observations.”**

而对你现在的 **SONIC + GR00T humanoid loco-manipulation** 来说，我其实觉得这个观点特别重要：**没必要期待 GR00T 自己成为整个 agent。更自然的是让 Agent 决定 task/subgoal，GR00T 解决视觉条件下的 manipulation，SONIC 解决物理可执行的 whole-body control。** 这三个层级刚好各自解决一个不同的问题。
