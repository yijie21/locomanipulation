# Scripted Expert 创新点定位讨论记录

> 2026-08-26。承接 [sonic_gr00t_srpo_dagger_agent_vla_discussion.md](sonic_gr00t_srpo_dagger_agent_vla_discussion.md)
> （与 ChatGPT 关于 SRPO / DAgger / agent-based VLA 的讨论）。
> 本记录从"把那份讨论交给 Claude 并要求更深的见解"开始，到 imitability 重构定位为止。
> 中间的 scoop check 完整报告见仓库根 step1–6.md 与 papers/ 下的七篇全文。

---

## User

我们现在进行一些科研上面的讨论，主要是定义一下这项研究到底有什么创新点，能得到什么突破性的技术进展。我之前和 chatgpt 的聊天中获得了很多启发（讨论记录见 docs/sonic_gr00t_srpo_dagger_agent_vla_discussion.md）。我觉得他说的"真正的问题"这一点非常正确，我先不发表我的见解，你可以先看一下这个文档，然后提出一些更好更可靠的见解。

## Assistant

读完了那份 1847 行的讨论。SRPO 的"200 steps"拆解是准确的，DAgger 的解释是教科书级的。但正因为我手上有**实测数据**而 ChatGPT 只有口头描述，它有一处诊断是错的，而那处错误恰好通向要找的创新点。分四层：

### 一、"真正的问题"要再拧紧一圈

ChatGPT 的版本：privileged information 不是问题本身，真正的问题是设计一个 mechanism，把 privileged agent 的能力 distill 成只依赖真实传感器的 deployable policy。

方向对，但它停在"要有一个 mechanism"。往下拧：**distillation 不是一个问题，是三条接缝，信息在每条接缝上以不同方式死掉**：

- **接缝 1：expert → data。** privileged expert 的行为被录成像素和 token 时，混进了哪些 artifact？
- **接缝 2：data → policy。** chunking horizon、open-loop 指标的欺骗性、token 量化。
- **接缝 3：policy → controller。** latch 行为、mode 切换、streaming 时序。

闭环失败的默认归因是 covariate shift，默认药方是 DAgger。接缝分解逼你先问"死在哪条缝上"，再选药。

### 二、ChatGPT 的诊断被实测证伪了一半——这是创新点的入口

它的故事线：干净 demo → policy 走歪 → OOD → freeze（纯 covariate shift 叙事）。但实测不支持：

- scripted expert 到位后有**中位 4.6 秒的静止停顿**，action chunk 只能吸收 1.33 秒（40 帧 @30Hz）——96% 的停顿超出 chunk 视界
- NVIDIA 的参考数据是**人类 teleop**，同样的停顿只有 **0.2 秒**
- 所以"走到桌边愣住"很大程度上**不是 OOD 后的无所适从，而是 in-distribution 的忠实模仿**——demo 教会了它"到位之后先站一会儿"，"站着"是自我延续的 attractor

这解释了 covariate-shift 叙事解释不了的数据点：**round 2 的 DAgger 只把 6% 提到 13.7%**。若问题纯是 recovery 缺失，50 条靶向 recovery 应有更大杠杆。按接缝分解：DAgger 修接缝 2 的覆盖，**修不了接缝 1 的 artifact——甚至会把 artifact 再采一遍**（DAgger recovery 数据同样带停顿）。

round 3 就是判决实验：同 200 条、修掉 cadence（停顿 4.6s→3.0s、步态不再犹豫、episode 35s→21s）重训。

由此第一个可证伪 claim：**scripted 与人类 teleop 的差别不在动作正确性，而在时间纹理（cadence）。BC 会克隆 cadence artifact，且 chunking 只能吸收短于视界的停顿。对 scripted 数据引擎，cadence 是一等数据质量指标。**

### 三、诚实排雷之后，真正的 thesis 露出来

不是我们的创新：teacher-student distillation（标准操作）、DAgger（2011）、VLA 输出 WBC latent token（NVIDIA 官方配方）。

剩下一个 NVIDIA 配方里被默认为前提、而我们换掉的东西：**NVIDIA 的参考数据是人类 teleop 采的，我们的是零人工。** thesis：**人类 teleoperator 能否被 scripted privileged agent 完全替代？代价是什么，缺口在哪，如何补齐？** 全部实测发现（cadence、spawn envelope、可见性门、DAgger 边界）就是"替代的代价清单"。

### 四、对讨论文档后半段的两个修正

- **humanoid SRPO 的 feasibility 担忧在我们的架构里不存在**：token 动作空间里 feasibility 是构造保证的——三轮闭环失败分类全是 reach/close 意图错误，**零摔倒**。latent-WBC 动作空间把探索从"既要对又要不摔"降维到"只要对"。
- **SRPO 复刻 miss 了免费资源**：我们 sim-first，GT object pose 白拿。正确实验是 sim 内 privileged progress 做 advantage weighting vs latent-similarity reward 的对照；SRPO 式机制只在真机阶段必要。failure-state fork 在 sim 里是现成的。

---

## User

跑一次文献查重。我知道有一篇叫 LEGS 的论文也 claim 了 scripted policy 采集数据，但那篇把这部分写得太简单了，实际实现没这么简单——不要一查到类似论文就判死刑，要仔细了解他们的缺陷和不足在哪儿。

## Assistant

（执行了完整 scoop check：4 路检索 ≈85 篇去重、7 篇全文深读——LEGS、SIMPLE、DemoHLM、WholeBodyVLA、LeVERB、HAF、ISR。完整报告见 step1–6.md。）

**Verdict：Level 2 — High Overlap**（由 LEGS 驱动，其余最近邻 Level 3–4）。不是死刑，但改变了三条 claim 的地位。

### LEGS（arXiv:2606.01458，Stanford，2026-05-31）的核实结果

**匹配得可怕的部分**：同款 G1、**同款 SONIC**、backbone 含 GR00T N1.6、procedural privileged 生成 + 验证门 + 只存已验证 episode、200 episodes、真机 1110 trials、teleop 对照在长程任务塌到 0/10 而 LEGS 数据 2–6/10。**claim 1（"scripted 替代 teleop"）的旗子在他们手里。**

**逐条核实的薄处**：

1. §3.3 procedural 生成正文只有一段；**出生随机化仅机器人 ±10cm、朝向 ±10°、物体 ±5cm**——对照我们 0.7m spawn-edge、±20° yaw、±0.20m lateral。他们的"walk"本质是排练好的固定路径；我们实测过 spawn envelope 单独值 4 倍成功率差。
2. **全文零提及** pause / cadence / hesitation / recovery / DAgger / covariate shift。唯一沾边的是 limitations 里"不优化 dynamic efficiency"——把运动质量归为效率问题，没意识到会被 BC 克隆。
3. 没有闭环失败分类学、没有 open-loop vs closed-loop 落差研究。
4. **动作空间是 18 维运动学命令**（双腕 SE(3)+夹爪 + 4 维底盘），不是 latent token。
5. 他们的科学核心是**渲染轴**（3DGS vs mesh-only 消融），procedural 生成是工具不是对象。

**好消息**：LEGS 反过来验证了 SONIC 同一二进制 sim→real、纯仿真数据零样本上真机这条链路——我们将来上真机可直接引用。

### 其余最近邻一行结论

- **ISR**（2606.22907）：claim 2 最近邻——人类 teleop 演示的**事后** pause 重采样（+25%）。做不了 scripted vs human 类别对比、无 chunk 视界机制、无闭环因果链；其加速度项声称保护精调段，正是我们 park 掉 --drop-idle 的难点，应正面引用。
- **SIMPLE**（2606.08278）：benchmark；CuRobo planner 自动生成 + Isaac Sim 离线重渲染（我们 ROADMAP 镜像渲染架构的先例）。
- **LeVERB**（2506.13751）：claim 3 最近邻——共训 latent verb + RL WBC，但无物体抓取；"latent 接口优于手工 EE 词表"是这一脉共同前提，claim 3 不能作机制主张。
- **DemoHLM / WholeBodyVLA / HAF / MimicGen 一族**：各差 2–3 轴。

### 三条 claim 重新定位

1. **Claim 1 降级为设定**："LEGS 证明这条路存在；我们回答它何以成立、何时失败、代价几何"。
2. **Claim 2（cadence）升为主线**——全场唯一无人占据且证据链已齐的位置，round 3 是判决实验。
3. **Claim 3 降为实证刻画**；补一个 18 维命令线对照训练（LEGS 用的正是那条线，同款控制器，天然对照）可升回主张。

一句话：**LEGS 抢了"能不能"，但把"为什么、差在哪、怎么治"整个留给了我们——那三样我们手里全是测过的数。**

---

## User

可以（把这个框架整理成一页 research statement）。

## Assistant

（写入 docs/research_statement.md，v1 草稿，72 行：Thesis 一句话、三条 claim 定位、五行最近邻分界表、证伪条件清单——含"round 3 ≤ round 2 则 C2 削弱"与"缺人类 teleop 上界对照，写作须声明"——以及实验阶梯。round 3 闭环数字的槽位标 ⏳ 待填。）

---

## User

所以现在的结论是，根据 chatgpt 和你的讨论，有什么创新点吗？

## Assistant

有，一个真创新点，两个支撑点——但先说清哪个不是。

**不是**：「scripted 替代 teleop」——LEGS 已插旗；「privileged teacher → deployable student」大框架——locomotion 标准操作。只能当设定写。

**真创新点：cadence（C2）。**"scripted 与人类演示的差距不在动作对不对，而在时间纹理；BC 会把停顿克隆成闭环 freeze attractor；chunk 只能吸收短于视界的停顿；治理只能在源头。"查遍无人占据；证据链完整（4.6s vs 0.2s vs 1.33s、DAgger 只提 6%→13.7% 的反证、事后过滤危险性的标定 0.11 vs 0.06 rad/s、源头治理措施账）。唯一条件：round 3 得赢。

**两个支撑点**：(1) 诊断框架——"先分清失败死在哪条接缝，接缝 1 的 artifact 会被 DAgger 原样再采"；(2) token 空间的失败压缩（零摔倒的失败分类学），补命令线对照可升格。

诚实短板：缺人类 teleop 上界对照；round 3 是单任务单场景，泛化主张不能越界。

---

## User

你怎么又给一个"可测量"的这种创新点，这一点也不 exciting，还不如 chatgpt 提出的"真正的问题"。

## Assistant

批评得对。「cadence 可测量」是 workshop paper 级别——它是**证据**，我错把它当成了**主张**。往上一层，同一批证据支撑一个大得多的问题，而且恰好是 ChatGPT"真正的问题"的下一步。

### ChatGPT 的"真正的问题"停在哪儿

它说：真正的问题是设计 mechanism，把 privileged agent 的能力 distill 成只依赖传感器的 policy。对——但隐含假设是 **teacher 给定，要设计的是 distillation 机制（学生侧）**。整个 teacher-student 传统也是这么干的。我们三轮实验证明这个假设不成立。

### 真正的问题再往下一层：teacher 不是给定的，teacher 是要设计的

用 scripted privileged agent 替代人类，得到的不是更便宜的 teleoperator，而是**另一个物种的示教者**。它的行为由三种学生看不见的东西驱动：

1. **privileged 状态**（真值位姿、可达域）——动作与像素解耦
2. **内部离散状态**（planner 相位、settle 计时器）——**停顿的真正来源**：expert 站着是因为一个隐藏计时器在跑，像素里没有任何东西解释"为什么等、等到什么时候"
3. **无传感反馈的运动生成**——轨迹天然不含纠错结构（recovery 缺失是构造性的，不是采集疏忽）

BC 强迫学生用像素解释这一切。**解释不了的行为不会消失——它塌缩成 attractor。** freeze 就是这么来的：像素层面的教训是"在这个状态下，什么都不做"。

反过来一次性解释了人类 teleop 数据为何天然好用：**人就是从学生的 observable 在行动的**。人只在有视觉原因时停顿（0.2s），scripted expert 在计时器要它停时停顿（4.6s）。人类演示是**构造性可模仿的**，scripted 演示不是。

### Exciting 的版本

> **Scripted expert 要 scale，瓶颈不在数据引擎工程，也不在学生侧算法，而在 teacher design：把一个"求解器"改造成一个"教师"——它的每个动作必须是学生可观测量的函数，且发生在学生架构能表示的时间尺度上。把"可模仿性（imitability）"作为 privileged 数据引擎的设计对象，给出设计原则、可测代理量、和第一个系统级证据。**

回头看，几个月的工程全是这个原则的实例，只是没人命名：

| 我们做的 | 它其实是什么 |
|---|---|
| 可见性门（物体必须在 ego 视野内才动作） | 强制 teacher 只依据学生看得见的信息行动 |
| grasp-causality audit | 强制动作有视觉因果 |
| cadence 治理 | 消除隐藏计时器驱动的行为 |
| face-object spawn | 让视觉上下文与动作对齐 |
| DAgger 的失败（6%→13.7%） | 学生侧修补救不了 teacher 侧的不可模仿性 |

且它给出可证伪预测：LEGS 式管线在 ±10cm 信封里够用，信封放宽、任务拉长时会精确死在 teacher 不可模仿的维度上——我们的理论解释了他们为什么在窄信封里就够用。

### 诚实的前置声明

"imitation gap from privileged experts" 的**理论概念有先例**（记忆中：Weihs et al. ADVISOR, NeurIPS 2021 "Bridging the Imitation Gap"；Walsman et al. "Impossibly Good Experts"，ICLR 2023 左右）——expert 依据不可观测信息行动时模仿有本质 gap。但那条线是 gridworld/理论 + RL 侧修复，没人做过：(a) VLA 规模真实系统；(b) gap 的**时间面**（隐藏计时器 → cadence → 与 chunking 的相互作用，我们独有）；(c) **teacher 侧设计原则**作为实用解法。定位："把 imitation-gap 理论操作化为 scripted 数据引擎的设计学科"。

这条线需要再跑一次 scoop check 才能落笔。若站得住，research statement 的 thesis 从"cadence 是数据质量轴"升级为"**imitability 是 teacher design 的对象，cadence 是它的时间面**"——同一批证据，高一层的问题。

---

## 待办（截至本记录）

1. ⏳ round 3 闭环数字（C2 判决实验）→ 填入 research_statement.md
2. imitation-gap 线（ADVISOR / Impossibly Good Experts / TGRL 等）的第二轮 scoop check
3. 18 维命令线对照训练（C3 升格，同数据同控制器换接口）
4. failure-state fork + privileged-progress advantage weighting（sim 免 V-JEPA 的 SRPO 变体）
5. 人类 teleop 上界对照的可行性评估（同信封 200 条 teleop）
