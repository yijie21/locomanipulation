> **台账(append-only)**:本文件只在末尾追加新的日期化章节,不改写、不删除已有结论。
> 要更正一条结论,在新章节里写清楚并链接回旧的那一条,不要就地覆盖。
> 分类与格式规则见 `docs/reference/standards.md` §3。

# 实验总表(2026-08-25 → 2026-09-03)

所有成功率均从 `outputs/eval/<arm>/closed_loop.manifest.jsonl` 重算(≥14 局的臂才收),配对检验是同 seed、同出生分布的臂逐轮 McNemar(x:y = 只有前者成功 : 只有后者成功)。失败细分来自 `tools/failure_taxonomy.py`(只对有 trace 的臂;部分臂 trace 少于 manifest 局数,细分按 trace 子集)。

**deploy 状态**这一列是理解全表的钥匙:`退化` = production 二进制 + planner settle 后 enter_streamed,行走 token 只按 1/3 幅度执行(2026-09-02 发现);`干净` = nvdef 二进制 + SONIC_AUTO_START,全程不碰 planner;`lockstep` = 仿真时钟步进。退化 deploy 上的数字彼此可比,但不可迁移到真机,也不能和干净协议的数字直接比。

## 0. 一页结论

1. **数据形状 > 数据量**:同样 200 局,去掉停顿的 fast200 把 bottle200 的 5.9% 翻到 28%;靶向 DAgger 再到 47.5%(R1–R4,退化 deploy,出生 0.7 m)。
2. **出生距离是最大的单一变量**:同一 walk200 模型,出生 0.7 m 20% vs 1.1–1.3 m 38%;fast250 在 0.7 m 47.5%。不同出生分布的数字不能横比。
3. **DART 远停恢复数据没有提高基线**:干净协议下 dart200 24/100 vs flow200 50/100(p=0.0002);它只是给 ensemble 制造了一个可以修的停顿(24→56)。ensemble 在 flow200 上为零(50 vs 46,p=0.67),在退化 deploy 上有害(32/50 → 12/50,p<0.001)。"ensemble 是通用增益"的旧结论(dart200 时代 31→73)是数据副作用 + 退化 deploy 的产物。
4. **RTC 三个变体全部显著负收益**(22% / 2% / 22% vs 50%),前提(已发出的 chunk 被如实执行)在 token 经 frozen WBC 执行的链路上不成立,关闭。
5. **lockstep 校准未过**(32 vs 46,p=0.049),原因未定位;wall-clock 干净协议仍是正式协议。
6. **失败的主体不是走不到,是伸手撞倒瓶子**:flow200 在干净协议上 76–78% 的失败是第一次伸手就把瓶子扫倒;dart200 的失败一半在导航段。这直接指向采集侧的慢速 reach(smooth200)。
7. **当前正式基线**:flow200 ckpt-10000 + 干净协议 + async 0.4 s 重规划 + 无 ensemble = **50/100**。

## 1. 数据形状 vs 数据量:R1–R4 阶梯(退化 deploy,出生 0.7 m,seed 6000)

**为什么跑**:第一版模型 5.9%,要分清是数据太少还是数据不对。四轮里只改一个变量:R2 加 50 局 DAgger 恢复;R3 不加数据、只把专家的停顿去掉;R4 在 R3 上加 50 局靶向 DAgger。

| 轮 | 臂 | 数据 | 成功率 | 与上一轮配对 | 失败细分 |
|---|---|---|---|---|---|
| R1 | bottle200 | bottle200(200 局脚本专家) | **6/101**(6%,CI 3–12) | — | 无 trace |
| R2 | bottle250 | + 50 局 DAgger 恢复 | **10/68**(15%,CI 8–25) | 不同 seed,不配对 | 无 trace |
| R3 | fast200 | fast200(同 200 局,专家停顿 4.6→3.0 s,零 DAgger) | **28/100**(28%,CI 20–37) | — | 无 trace |
| R4 | fast250 | + 50 局靶向 DAgger(0.35–0.55 m 处恢复) | **48/101**(48%,CI 38–57) | fast200 vs fast250 9:29,p=0.002 | 无 trace |

**结论**:R3 证明病在数据形状(同样 200 局翻倍);R4 证明靶向 DAgger 有效(+29:−9)。但 2026-08-28 发现这些局里约 2/3 在 VLA 接管前已被脚本走到桌边(stand-start gap),所以它们是「混合任务」数字;修复后同一 R4 模型的全任务干净基线是 12/100(该臂目录已不在,来源 docs/evaluation.md:504)。

## 2. 走路段进入任务:出生距离与 walk200(退化 deploy)

**为什么跑**:stand-start 修复后必须让策略自己走完全程,所以采集改为出生 1.1–1.3 m 的 walk200,并回答「出生距离对成功率影响多大」。

| 臂 | 模型 | 出生 | 成功率 | 配对 | 失败细分 |
|---|---|---|---|---|---|
| walk200 | walk200 | 1.1–1.3 m | **38/101**(38%,CI 29–47) | vs base100(flow200,退化→干净不可比,列出仅供参考)25:20,p=0.551 | B4 30;C3 5;A1 4(失败 40) |
| walk200_at07 | walk200 | 0.7 m | **20/100**(20%,CI 13–29) | vs fast250 同出生 30:8,p<0.001;vs fast200 14:13,p=1.000 | B4 56;C3 4;B5 2(失败 64) |
| walk200_ens | walk200 + ens N3 M0.3 | 1.1–1.3 m | **8/20**(40%,CI 22–61) | vs walk200 7:4,p=0.549 | B4 6;C3 2(失败 8) |
| ramp200 | ramp200(上肢 ramp 数据,20 局闸门) | 1.1–1.3 m | **6/20**(30%,CI 15–52) | vs fast200 — | B4 6;A1 3;B3 1(失败 14) |

**结论**:同一模型出生 0.7 m 反而只有 20%(walk200 在 0.7 m 没有对应数据,策略一出生就在伸手区);出生分布必须和训练一致。walk200_ens 20 局看不出 ensemble 收益。ramp200 闸门 6/20 失败,该数据线放弃。walk200 的失败 75% 是伸手撞倒瓶子,当时的粗分类把它记成'到桌边没合爪'。

## 3. DART 恢复数据与部署旋钮筛查(dart200,退化 deploy,出生 1.1–1.3 m)

**为什么跑**:walk200 的失败里 19% 停在 ≥0.5 m 的悬崖,成功率 5%;DART 在采集里注入'远停 → 走近再抓'的恢复演示。dart200 基线出现桌前停顿死循环,于是对部署旋钮(replan 间隔、ensemble N/M)做 20 局配对筛查,赢家续到 100 局。

| 臂 | 设置 | 成功率 | 配对 | 失败细分 |
|---|---|---|---|---|
| dart200 | 基线 async 0.4 s | **31/100**(31%,CI 23–41) | vs walk200 15:11,p=0.557 | B1 11;A1 7;B4 6(失败 29) |
| dart200_h40 | replan 1.6 s(整 chunk 执行) | **11/20**(55%,CI 34–74) | vs dart200 4:8,p=0.388 | B4 3;C3 1(失败 4) |
| dart200_ensN4 | ensemble N4 | **6/20**(30%,CI 15–52) | vs dart200_ens 10:1,p=0.012 | B6 6;B1 3;B3 1(失败 11) |
| dart200_ensM0 | ensemble N3 M0 | **8/20**(40%,CI 22–61) | vs dart200_ens 9:2,p=0.065 | B1 4;B6 3;B3 1(失败 10) |
| dart200_ensMn3 | ensemble N3 M−0.3 | **11/21**(52%,CI 32–72) | vs dart200_ens 7:3,p=0.344 | B4 2;B1 1;B? 1(失败 4) |
| dart200_h40ens | replan 1.6 s + ens | **12/20**(60%,CI 39–78) | vs dart200_h40 4:5,p=1.000 | C3 3;A1 1;B4 1(失败 5) |
| **dart200_ens** | ensemble N3 M+0.3(赢家续 100) | **73/100**(73%,CI 64–81) | vs dart200 3:27,p<0.001 | B6 9;B1 5;B? 2(失败 19) |

**结论(当时)**:数据效应为零、部署效应 +42 点,'ensemble 是解锁器'。dart200 基线的失败 24% 停远 + 38% 站太远(导航段),dart200_ens 的失败 47% 是桌前停滞。**后来被第 5、8 节推翻的部分**:这些都是退化 deploy 上的数,而且 ensemble 的增益是 DART 数据特有的。

## 4. 部署侧普查(退化 deploy,2026-08-30/31)

这一批是排除法,大多数臂只有 20 局或没有 manifest,按'只收可重算'的规则不列成功率,只记结论(来源 docs/evaluation.md:571–760):开环 denoising 步数 4 即够(模型开环只比噪声地板高 2%);观测延迟 33 ms 不是因素;replan 1.6 s 显著更差;grip-gain 3.0 无差异(爪子不是瓶颈);200 局随机出生扫描证伪'停位窄甜区',真实模式是 0.50 m 悬崖(≥0.50 m 成功 5%,<0.50 m 55%);初始姿态 token 换成数据 medoid(关节误差 23°→3.4°)。

## 术语:从第 5 节起需要的全部背景

**系统是什么**。机器人是 Unitree G1 人形机器人,在 MuJoCo 仿真里完成一件事:从离桌子 1 米多的地方走到桌边,弯腰,用右手抓起桌上的水瓶并抬起。控制分两层。上层是 VLA(视觉-语言-动作模型,GR00T N1.7 微调版),每 0.4 s 看一次相机图像和本体状态,输出一段未来动作(一个 chunk,16 帧)。下层是 SONIC 全身控制器(WBC),它不接收关节角,而是接收一个「运动 token」(动作意图的离散编码,如「向前走」「站着」「弯腰伸右手」),再自己算出 29 个关节的力矩去执行。VLA 输出的正是这种 token 序列加夹爪开合。SONIC 控制器以一个独立进程运行,项目里把这个进程叫 **deploy**(它就是 NVIDIA 发布的 `g1_deploy_onnx_ref` 部署程序)。

**数据从哪来**。没有人类遥操作。一个脚本专家(planner 模式下的 SONIC + 逆运动学)自己走过去抓瓶子,把过程录成演示,VLA 用这些演示做模仿学习。每一版数据集有一个名字,规则是 **<采集协议><成功局数>**:bottle200、fast200、walk200、dart200、flow200、smooth200 都是 200 局成功演示,名字前半段标记那一版采集协议的特征(fast = 去掉了停顿,walk = 出生更远要走路,dart = 加了远停恢复演示,flow = 一次成功不修复,smooth = 慢速伸手无停顿)。

**评测怎么做,臂怎么命名**。闭环评测(closed-loop)= 让 VLA 真的控制仿真机器人从出生点做完整任务,判定标准是瓶子被抬离桌面 6 cm 以上。每次评测固定 seed 6000,所以不同臂的第 k 局出生点完全相同,可以逐局配对比较(McNemar 检验:x:y 表示只有前者成功的局数 : 只有后者成功的局数)。每个评测运行叫一个「臂」,目录名就是臂名,规则大致是 **<模型或含义><局数>[_变体]**:

- `base100`:flow200 模型、干净 deploy、**无** ensemble,100 局。「base」= 基线部署配置。这是当前正式基线。
- `clean100`:flow200 模型、干净 deploy、**带** ensemble N3 M0.3,100 局。「clean」= 干净 deploy 协议下的第一个臂,当时默认带 ensemble。
- `planner50_base` / `planner50_ens`:flow200 模型,**退化** deploy(名字里的 planner 指退化的成因),50 局,无 / 有 ensemble。
- `dart200` / `dart200_ens`:dart200 模型,退化 deploy,无 / 有 ensemble;`dart200_cbase` / `dart200_cens`:同一模型在干净(c = clean)deploy 上的无 / 有 ensemble。
- `rtc100`、`rtc_ens100`、`rtctrain200_rtc`:RTC 的三个变体;`lock100`、`calib_*`、`lockstep14_*`:lockstep 时钟方案。

**deploy 的「干净」与「退化」**。这是 2026-09-02 才发现的一个陷阱。deploy 进程有两种进入工作状态的方式。老方式:客户端先发一条 planner 命令(planner 是 deploy 内置的简单运动规划器,给它速度指令它自己生成 token)让机器人站好,再切到「外部 token」模式接收 VLA 的 token。新方式:启动时用 `SONIC_AUTO_START=1` 让 deploy 自己进入工作状态,客户端从第一帧起就直接发 token,全程不碰 planner。实测发现,**只要进程走过一次 planner 模式,之后每一个「向前走」的 token 实际只走出原来 1/3 的距离**(同样 125 个 token 开环回放:干净 0.68–0.71 m,退化 0.02–0.22 m,数据集里对应 0.80 m),机制在 deploy 内部,没有定位到具体状态量,重启进程才消失。老方式的进程状态叫 **退化 deploy**,新方式叫 **干净 deploy**。2026-09-02 之前的所有评测都在退化 deploy 上做的;真机没有这种状态,所以那些数字彼此可比但不可迁移,正式数字从 base100 / clean100 起一律用干净协议。干净协议还统一了 deploy 的二进制(nvdef:NVIDIA 原始默认关节角,不带 lockstep 标志,只是普通 wall-clock 运行)。

**ensemble**。VLA 每 0.4 s 出一个 16 帧的 chunk,相邻 chunk 有重叠。ensemble N3 M0.3 = 对当前时刻取最近 3 个 chunk 各自的预测做加权平均,M 是新旧权重的偏置(+0.3 偏向旧的)。它是部署侧的旋钮,不改模型。

**async 客户端 / 重规划间隔**。「async」= 推理和执行并行:执行上一个 chunk 的同时算下一个,每 0.4 s(重规划间隔)换新 chunk 并按推理耗时跳过已经过去的帧。「blocking」= 算完再执行,期间机器人空转。replan 1.6 s = 整个 chunk 执行完再换。

**RTC(real-time chunking)**。换 chunk 时把还没执行完的旧 chunk 前缀「冻结」,让新 chunk 在前缀条件下补全(inpainting),避免拼接处的跳变。test-time RTC 只改推理;rtc-train 是训练时也按这个方式加噪的模型。

**lockstep**。仿真只在客户端命令时步进,推理期间按预先测得的延迟分布用旧 chunk 精确推进若干帧,再接新 chunk。目的是让评测确定性、不受 GPU 争抢影响。用它之前必须证明它和真实 wall-clock 同分布(校准)。

**DART**。采集时故意让专家在离桌 0.45–0.70 m 处先停一次,再走近抓,把「停远了怎么办」的恢复演示录进数据,针对策略常停在 ≥0.5 m 处这一失败。

**失败细分的叶子**(第 9 节):A 导航段没到桌边(A1 停远 ≥0.45 m、A2 走过头或走偏)、B 到了桌边但从未发合爪指令(B1 站在 0.36–0.45 m 手够不着、B2 手臂根本没动、B3 手到了瓶子旁却不合、B4 伸手时把瓶子撞倒、B5 反复接近又退回、B6 到位后整体静止 >10 s)、C 合了爪但没抬起(C1 合空、C2 合上后滑脱、C3 合爪时撞倒、C4 合爪高度不对)、D 抬起后又掉、E 摔倒。

## 5. deploy 状态与 ensemble(flow200 ckpt-10000,seed 6000,出生 1.1–1.3 m,2026-09-02)

**为什么跑**:第 7 节的 lockstep 校准一直对不上,追查过程中发现了退化 deploy 这件事(见术语)。这意味着之前所有闭环数字都是在一个「油门只有 1/3」的机器人上量的。于是定义了干净协议,并在同一个 flow200 模型上做一个 2×2:deploy 状态(干净 / 退化)× ensemble(有 / 无),看两件事:干净 deploy 上策略到底多少;早期「ensemble 明显提升」的结论在干净 deploy 上还成不成立。四个臂同一 seed、同一出生分布,逐局配对。

| 臂 | deploy | ensemble | 成功率 | 配对 | 失败细分 |
|---|---|---|---|---|---|
| **base100** | 干净 | 无 | **50/100**(50%,CI 40–60) | — | B4 38;C3 6;B5 4(失败 50) |
| clean100 | 干净 | N3 M0.3 | **46/100**(46%,CI 37–56) | vs base100 26:22,p=0.665 | B4 42;B1 5;B6 3(失败 54) |
| planner50_base | 退化(nvdef + planner settle) | 无 | **32/50**(64%,CI 50–76) | vs base100 9:17,p=0.169 | B4 10;B6 4;A1 1(失败 16) |
| planner50_ens | 退化 | N3 M0.3 | **12/50**(24%,CI 14–37) | vs planner50_base 24:4,p<0.001;vs clean100 16:4,p=0.012 | B4 22;C3 3;A1 2(失败 31) |
| flow200_ens | 退化(production),首个 flow200 臂 | N3 M0.3 | **8/14**(57%,CI 33–79) | 14 局,不配对 | B4 3;B1 1(失败 4) |

**怎么读这张表**:base100 和 clean100 只差一个 ensemble 开关,50 vs 46,不一致对 26:22,p=0.67,也就是 ensemble 在干净 deploy 上没有可测效果。planner50_base 和 planner50_ens 是同一模型在退化 deploy 上的对照,64% vs 24%,p<0.001:退化 deploy 上 ensemble 是明显有害的。退化 + 无 ensemble(64%)看起来高于干净基线(50%),但 p=0.17 不显著,而且退化状态是仿真部署的一个 bug 态,真机不存在,所以这个「更高」没有意义。flow200_ens 是 flow200 训完后第一个评测臂(14 局,退化 deploy,带 ensemble),只作历史记录。

**结论**:干净 deploy 上 ensemble 没有可测收益;退化 deploy 上 ensemble 有害。自此正式数字一律用干净协议,正式基线定为 base100 = 50/100。

## 6. RTC(real-time chunking)三变体(干净 deploy,flow200,seed 6000)

**为什么跑**:async 客户端每 0.4 s 换一个 chunk,两个 chunk 的拼接处动作会跳变,直觉上这会让伸手、合爪这类精细动作变差。RTC 是文献里针对这个问题的标准做法(见术语),所以在干净 deploy 上试了三种形式:只在推理时用(rtc100)、训练时也按 RTC 方式加噪再推理时用(rtctrain200_rtc)、推理时 RTC 再叠加 ensemble(rtc_ens100)。全部与 base100 同 seed 配对。

| 臂 | 变体 | 成功率 | 配对 vs base100 | 配对 vs clean100 | 失败细分 |
|---|---|---|---|---|---|
| rtc100 | test-time RTC | **14/64**(22%,CI 14–33) | 26:8,p=0.003 | 23:7,p=0.005 | B4 32;C3 12;B1 2(失败 50) |
| rtctrain200_rtc | rtc-train(P 0.5,KMAX 20,loss 0.0913)+ RTC | **1/50**(2%,CI 0–10) | 24:1,p<0.001 | — | B4 19;A1 18;B1 6(失败 49) |
| rtc_ens100 | test-time RTC + ens N3 M0.3 | **16/72**(22%,CI 14–33) | 30:10,p=0.002 | 29:10,p=0.003 | B4 39;B6 5;C3 5(失败 56) |

**怎么读这张表**:三行都比 base100(50%)低得多,配对 p 全部 ≤0.003,不是噪声。rtctrain200_rtc 几乎全失败(1/50),失败里 37% 是停在离桌 ≥0.45 m 的地方不动。rtc100 和 rtc_ens100 的失败里,伸手撞倒瓶子占 64–70%,合爪时撞倒(C3)占 24%,后者是基线(12%)的两倍,说明前缀冻结恰恰把合爪这种需要突变的动作压平了。

**结论**:RTC 的前提是「已经发出去的动作会被如实执行」,这样冻结前缀才有意义。但这里发出去的是 token,不是关节动作,token 由 frozen WBC 再解释一次执行,本身带 0.2 s 量级的滞后,冻结 token 前缀锁不住实际动作。三种形式都是负收益,关闭这条线;基线的 0.4 s async 重规划已经够用。

## 7. lockstep(仿真时钟 + 经验延迟分布)校准

**为什么跑**:正常评测是 wall-clock 的:仿真按真实时间跑,VLA 推理花多久机器人就空转多久,所以同一台机器上另一个进程抢 GPU 就会改变结果,两次评测不可能逐帧复现。lockstep(见术语)是为了得到确定性、可复现、可以扫描「延迟对成功率影响」的评测。但它引入了一个新的时间模型,必须先校准:同一模型、同一 seed 下,lockstep 和 wall-clock 的成功率应当统计上无差异。这一节就是校准的过程。

| 臂 | 设置 | 成功率 | 配对 | 失败细分 |
|---|---|---|---|---|
| lockstep14_ens / _jitter / _noseed | bring-up 期三个 14 局变体 | **3/14**(21%,CI 8–48) / **2/14**(14%,CI 4–40) / **2/14**(14%,CI 4–40) | — | 停远 + 站太远为主 |
| calib_wall | wall-clock 对照(production) | **9/19**(47%,CI 27–68) | — | B4 7;B1 1(失败 8) |
| calib_lock | lockstep 对照 | 0/12 | — | — |
| lock100 | 干净二进制 lockstep,经验延迟分布,ens N3 M0.3 | **32/100**(32%,CI 24–42) | vs clean100 29:15,p=0.049 | B4 35;A1 14;B1 6(失败 68) |

**怎么读这张表**:lockstep14_* 是 bring-up 期的三个 14 局小臂(不同的随机种子处理方式),成功率只有 2–3/14,远低于同期 wall-clock 的 9/17(calib_wall),差距大到不像统计噪声,于是开始诊断。诊断发现 lockstep 走的是干净 deploy(auto-start),而 wall-clock 走的是退化 deploy,再加上两边用的 deploy 二进制不一样(一个改过默认关节角),这才有了第 5 节的整套发现。把两边统一成干净二进制后重做 100 局配对(lock100 vs clean100):32 vs 46,不一致对 29:15,p=0.049,仍然差 14 点。lock100 的失败里 21% 停远、51% 伸手撞倒,系统性比 wall-clock 停得远(0.36 vs 0.27 m)。

**结论**:校准没有通过,lockstep 与 wall-clock 之间还有一个未定位的差异(候选:会话重置清空历史、图像新鲜度、观测同步)。在定位之前 lockstep 不能替代 wall-clock;正式协议仍是 wall-clock 干净协议。这条线最大的产出是意外的:它暴露了退化 deploy,并催生了干净协议。

## 8. DART 在干净协议下(2×2 补全,2026-09-03)

**为什么跑**:第 3 节里 dart200 + ensemble 曾经达到 73%,是全项目最高的数字,自然会想「远停再走近」这种采集机制是不是本身就更好。但那是退化 deploy 上的数。要回答这个问题,必须把 dart200 模型也放到干净 deploy 上,和 flow200 的两个臂(base100 / clean100)组成 2×2:数据(flow200 / dart200)× ensemble(无 / 有),四个臂同 seed 逐局配对。

| checkpoint | 无 ensemble | ensemble N3 M0.3 | base vs ens |
|---|---|---|---|
| flow200 | base100 **50/100**(50%,CI 40–60) | clean100 **46/100**(46%,CI 37–56) | 26:22,p=0.665 |
| dart200 | dart200_cbase **24/100**(24%,CI 17–33) | dart200_cens **56/100**(56%,CI 46–65) | 7:39,p<0.001 |
| flow200 vs dart200 | 37:11,p<0.001 | 20:30,p=0.203 | |
| 与退化 deploy 旧数配对 | dart200 vs cbase 16:12,p=0.572 | dart200_ens vs cens 23:13,p=0.133 | |

失败细分:dart200_cbase B4 33;A1 16;B1 14;B6 6(失败 76);dart200_cens A1 13;B1 13;B4 9;B6 6(失败 44)。

**怎么读这张表**:第一行是 flow200,第二行是 dart200,列是有无 ensemble。无 ensemble 时 dart200 比 flow200 低 26 点(24 vs 50,p=0.0002);有 ensemble 时 dart200 高 10 点(56 vs 46)但不显著(p=0.20)。dart200 自己的两个臂差 32 点(p<0.0001),而 flow200 自己的两个臂几乎没差。最后一行把干净 deploy 的新数和退化 deploy 的旧数(第 3 节的 31 / 73)配对,都不显著:dart200 模型对 deploy 状态不敏感,原因是它的失败在导航段(停远、站太远),那是 token 意图的问题,不是执行幅度的问题。

**结论**:「远停再走近」的采集机制没有提高基线,反而降了 26 点。它的作用是教会策略在桌前停顿,而 ensemble 恰好能把策略从这种停顿里拽出来,于是产生了 +32 点的「ensemble 增益」。这个增益是 dart200 数据的副作用被补偿的结果,不是通用的部署增益。当前正式基线仍是 flow200 + 干净协议 + 无 ensemble(50/100),下一步数据方向是新采集协议(第 11 节)重采一版做单变量对照。

## 9. 失败细分总表(tools/failure_taxonomy.py,2026-09-03)

叶子定义:A 导航(A1 停远 ≥0.45 m、A2 走过/走偏)、B 到位未合爪(B1 站太远 0.36–0.45、B2 手臂没动、B3 悬停、B4 伸手撞倒、B5 犹豫、B6 停滞)、C 合爪未抬(C1 合空、C2 滑脱、C3 合爪时撞倒、C4 高度不对)、D 抬起后丢、E 摔。旧 trace 无瓶子倾角,B4/C3 用瓶心位移 ≥5 cm 代替(base100 验证:50 局成功 0 局有,50 局失败 39 局有)。

| 臂 | trace 局数(OK) | A1 | A2 | B1 | B3 | B4 | B5 | B6 | B? | C1 | C3 | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walk200 | 73(33) | 4 |  |  | 1 | 30 |  |  |  |  | 5 |  |
| walk200_at07 | 80(16) | 1 |  |  |  | 56 | 2 | 1 |  |  | 4 |  |
| dart200 | 50(21) | 7 |  | 11 |  | 6 |  | 1 | 1 | 1 | 2 |  |
| dart200_ens | 76(57) |  | 1 | 5 |  |  |  | 9 | 2 |  | 1 | 1 |
| base100 | 100(50) | 1 |  | 1 |  | 38 | 4 |  |  |  | 6 |  |
| clean100 | 100(46) |  |  | 5 |  | 42 |  | 3 |  | 3 | 1 |  |
| planner50_base | 48(32) | 1 |  | 1 |  | 10 |  | 4 |  |  |  |  |
| planner50_ens | 43(12) | 2 |  |  |  | 22 | 1 |  |  | 2 | 3 | 1 |
| rtc100 | 64(14) | 1 |  | 2 |  | 32 | 1 |  | 1 | 1 | 12 |  |
| rtc_ens100 | 72(16) | 1 |  | 2 |  | 39 | 2 | 5 |  | 2 | 5 |  |
| rtctrain200_rtc | 50(1) | 18 |  | 6 |  | 19 | 1 |  |  | 3 | 2 |  |
| lock100 | 100(32) | 14 |  | 6 |  | 35 | 3 | 1 |  | 3 | 5 | 1 |
| dart200_cbase | 100(24) | 16 |  | 14 | 1 | 33 | 2 | 6 | 2 |  | 2 |  |
| dart200_cens | 100(56) | 13 |  | 13 |  | 9 |  | 6 | 1 |  | 2 |  |

**怎么读这张表**:每行一个臂,第二列是有 trace 的局数(括号内成功数),后面每列是落到该叶子的失败局数(空 = 0)。B4 这一列在 flow200 系(walk200、base100、clean100、rtc 三臂、lock100)全是最大的一格:这些模型走得到桌边、也伸手,但第一次伸手就把瓶子扫倒,然后再也不动。dart200 系(dart200、dart200_cbase / cens)的大格子在 A1 和 B1:停在 0.45 m 以外,或站在 0.36–0.45 m 手够不着;加了 ensemble 之后 A1/B1 减少、B6(到位后整体静止)增加,也就是 ensemble 把「停远」换成了「到位但停顿」。

**为什么这很重要**:评测脚本原来只有三桶,203 局「到桌边但没合爪」被当成一类。细分后发现其中绝大多数是撞倒。撞倒的机制在采集侧已经量到:SONIC 控制器的手臂落后于参考轨迹约 0.2 s × 速度,快速伸手时手比参考领先 5–6 cm,先撞上瓶子;采集侧把伸手速度减半后碰倒从 3/7 降到 0/7。smooth200 数据集就是按这个速度采的,所以它训出的模型有具体理由改善 B4。

## 10. 数据集谱系与训练

| 数据集 | 日期 | 采集协议(相对上一版的改动) | 采集局数 OK/总 | 训练 loss @10k | 主要用途 |
|---|---|---|---|---|---|
| bottle200 / bottle250 | 08-25 | 脚本专家,出生 0.7 m;250 = +50 DAgger | 无 manifest(目录已清) | ckpt 已删 | R1 / R2 |
| fast200(fastreach200) | 08-26 | 专家停顿 4.6→3.0 s,零 DAgger;fast250 = +50 靶向 DAgger | 200/224(89%) | ckpt 已删 | R3 / R4 |
| ramp200 | 08-28 | 上肢接管 ramp、DST_FPS 30→25、关外观随机、腕相机后移 | 193/207(93%) | — | 闸门 6/20,放弃 |
| walk200 | 08-29 | 出生 1.1–1.3 m,策略自己走 | 191/207(92%) | 0.0464 | walk200 系列、phase0 |
| dart200 | 08-31 | --dart-frac 0.7:远停(0.45–0.70 m)+ 走近再抓 | 208/249(84%) | 0.0537 | dart200 系列、2×2 |
| flow200 | 09-01 | 去 DART,one-shot 不修复,mid-band 0.20 | 200/262(76%) | 0.0483 | 当前正式基线、RTC、lockstep |
| rtctrain200 | 09-02 | flow200 数据 + RTC 训练目标(P 0.5,KMAX 20) | 同 flow200 | 0.0913 | rtctrain200_rtc(已删 ckpt) |
| smooth200(进行中) | 09-03 | 慢速 reach(V_TRAVEL 0.15/V_DESCEND 0.10/STAGE_UP 0.15)、lift 0.10、去停顿、闭环 creep 站位、one-shot 丢局、nvdef | 151/195(77%) | 未训 | 待训 + 干净协议配对 |

## 11. 采集协议迭代(2026-09-03,每批 ~12 局,INSTANCE 1 不录制除注明外)

**为什么跑**:看 flow200 的采集视频发现三类质量问题:右手在伸手时把瓶子碰倒;抓起后抬得太高(20 cm);走到桌边时站位不对(太近或太远),导致原来的脚本会「站起来、挪一步、再蹲下」修一次,这种修复段进了数据会让 VLA 学到犹豫。用户定的原则是:动作可以慢,但中间不能有明显停顿,需要修的局直接丢掉(one-shot)。下面每一批只改一项,用 ~12 局看采集成功率和丢局类型;OK/总 里的「总」包含被丢掉的局。gate0/1/2 是 smooth200 正式采集前的三次闸门批次(第一批 ≥12/20 才继续),都没过,分别对应三个 bug。

| 批 | 改动 | OK/总 | 主要失败 |
|---|---|---|---|
| planner_demo | planner 模式端到端确认 | 5/7(71%) | [grasp] water_bottle toppled in the in p |
| guard_demo | 左手 guard | 2/8(25%) | [grasp] grip target ended up at the body |
| dbg_demo | servo 遥测(量碰倒机制) | 2/8(25%) | [grasp] water_bottle toppled in the up p |
| slow_demo | 慢速 reach + staging 抬高 | 6/8(75%) | [stand] could not turn -N deg to the app |
| stand_demo | stand-wrong 诊断遥测 | 8/9(89%) | [grasp] grip target ended up at the body |
| turn_demo | 闭环转身 | 7/10(70%) | [grasp] grip target ended up at the body |
| crouch_demo | 下蹲后移遥测 | 8/10(80%) | [stand] could not turn +N deg to the app |
| band_demo | 蹲后带宽检查 + 修复循环 | 10/12(83%) | [stand] could not turn -N deg to the app |
| band2_demo | 同上复测 | 10/12(83%) | [view] object outside ego view at stand  |
| center_demo | one-shot 丢局 + 带中心 | 9/15(60%) | [grasp] [stand] band correction needed:  |
| still_demo | 停稳再量 | 10/13(77%) | [grasp] [stand] band correction needed:  |
| shift_demo | 蹲前带下移 | 10/12(83%) | [view] object outside ego view at stand  |
| nopause_demo | 去 settle + 刹车提前 4 cm | 10/12(83%) | [stand] could not turn -N deg to the app |
| smooth200_gate0_open_loop | 同上,INSTANCE 0 录制负载 | 10/27(37%) | [grasp] [stand] band correction needed:  |
| smooth200_gate1_nocreep | creep 挂错 flag 未生效 | 12/30(40%) | [grasp] [stand] band correction needed:  |
| smooth200_gate2_coast | creep 未等惯性 | 1/4(25%) | [grasp] [stand] band correction needed:  |
| smooth200 | creep + 惯性等待(正式采集) | 151/195(77%) | [grasp] [stand] band correction needed:  |

**怎么读这张表**:前七批是诊断和单项修复(碰倒 → 慢速伸手,3/7 降到 0/7;抬高 → lift 0.20 改 0.10)。band 到 nopause 五批都在修站位:先是发现刹车后机器人还会往前滑 5–18 cm、下蹲又把骨盆往后推 3–15 cm,两者叠加让站位散布超出手臂能抓的范围;试过等停稳再量(有效但引入 1.5 s 停顿,违反原则)、开环提前刹车(在不录制的实例上 10/11,一到录制负载下就 10/27,因为滑行量随机器负载变了)。最终方案是闭环 creep:刹车后不站定,等惯性卸掉,再以每步 1–5 cm 的碎步边走边量,进入目标带就立刻下蹲,全程没有站立帧。gate1 是 creep 挂在了一个被关掉的 flag 下根本没跑,gate2 是 creep 没等惯性就退出。

**结论**:碰倒问题由慢速伸手解决;站位问题的根因是开环刹车对机器负载敏感,只能闭环解决;正式采集(smooth200)在录制负载下稳定在 76–83%,与 flow200 的 76% 相同,而保留下来的每一局都没有停顿、没有修复段、没有碰倒。中间两次「越修越坏」的教训是:12 局的批次分不清 80% 和 60%,不要用小样本追上一批的主要失败。

## 12. Isaac 渲染开销与 hybrid clock 裁定(2026-09-05)

| 量 | 值 | 来源 |
|---|---|---|
| 稳态 obs→obs 预算 | 10 帧 @25fps(clean100 / base100 各 100 局) | `tools/latency_dist.py` |
| Isaac 单帧,连续 | p50 16.0 ms / p90 19.7 ms | `teleop_free.isaac.bench_render` |
| Isaac 单帧,间隔 0.2 s | p50 27.8 ms / p90 36.2 ms | 同上 |
| 折合额外陈旧帧 | p90 0.91 帧,约预算的 9% | 640×480,两卡空闲 |

**裁定:闭环评测用 Isaac 观测不需要 hybrid clock,纯 async 即可。** 详细条件、
四条限制和复现命令见 `docs/rendering.md`。

## 13. ready-grasp 采集协议的可学性实验(2026-09-07/08)

六臂配对(07 / 站近 / 横向 jitter / 站近+去 shuffle / seed 对照 / 站近+去 shuffle+放宽停靠带),同 seed 6000 逐局配对、60 局自动停。
**07d(站近 5 cm + `GYM_SHUFFLE_BAND=0`)30% → 48%,McNemar p = 0.052**;机制是策略停位由模仿 shuffle 的"蹭"决定,
不由专家停位决定;"没蹲"类别是训练 seed 方差。完整表、机制分析、方差底线与下一步见
[下一节「2026-09-07..09 ready-grasp arms」](#2026-09-0709-ready-grasp-arms);起点(creep 隐变量)也在那里。

## 2026-09-07..09 ready-grasp arms

# ready-grasp 采集协议的可学性实验(2026-09-07/08)

六个模型、同一评测协议、逐局配对。问题是:codex 重构后的 ready-grasp 专家(07)训出来的 VLA 为什么只有 30%,
采集侧改什么才能让**策略**(不是专家)变好。结论:**站近 5 cm + 去掉 shuffle 段**(07d)把闭环从 30% 提到 48%
(配对 McNemar p = 0.052),机制已验证;"没蹲"这一失败类别是训练 seed 方差,不是数据问题。

代码:`8c621fb` 为准,加上本轮之后未提交的四处改动(见 §7)。所有产物在
`/workspace/code/locomanipulation/outputs/experiments/ready07*_2026090[78]/`。

## 1. 起点:creep 的隐变量

cover400(16/100)对 flow200(50/100)的失败 77% 是"到桌边、90 s 内从不闭爪"。离线所有幅值指标正常,
唯一退化的是**时序**指标 `close_frame_err_abs`(10.25 vs 3.85 帧)。原因:STAND_COVER 每局随机抽一个站位格子,
creep 的停止判据是"当前位置 ≈ 抽中的格子"。格子不在相机里,于是同一物理位姿、同一画面,label 一半是"停下闭爪"
一半是"继续挪"。BC 回归到条件均值——半开夹爪慢慢蹭——闭环里没有任何观测能跨过那个不存在的阈值,成为吸收态。
creep 只占 2.8% 的帧,但坐在 phase 闸门上。

原则:**专家可以用特权信息计算动作,但动作必须是学生能观测到的量的函数。** 修法是把停止判据换成
深度带 [0.18, 0.27] m + 方位锥 ±36.5°(都从相对位姿算),验证后 ready07 的闭爪时序 1.83 帧、87 局里 1 局"立着不闭"。
详见 memory `creep-hidden-stop-target`。

## 2. 实验设计

| 项 | 设定 |
|---|---|
| 场景 | 冻结场景 `outputs/diag/grasp_mode_coverage_20260906/scene`,`SCENE_RANDOM=0` |
| 采集 | 每臂 60 集成功示范,spawn 用闭环评测分布(edge 1.1–1.3 m、yaw 20°、lateral 0.20、jitter 0.20),每 10 局重启 stack |
| 转换 | 25 Hz,`--trim-verification-tail`,holdout 6 → 54 train / 6 val,~22k 帧 |
| 训练 | GR00T-N1.7-3B slot 11,3000 步,batch 8×8,单卡,单 checkpoint |
| 评测 | `eval_closed_loop.py --fresh 100 --seed 6000 --client async --replan-interval 0.4 --budget 90`,ready token 初始化,**60 局自动停**,第 i 局各臂出生条件相同 → paired McNemar |
| 离线 | `eval_open_loop.py` on val,h=16 的 `close_frame_err_abs` |
| 一键 | `outputs/experiments/run_learn_pilot.py`(采→转→训→评);`summarize_arms.py` 出总表;`compare_arms.py` 配对 |

## 3. 六臂结果(配对,前 60 局)

| 臂 | 相对 07 的变量 | 专家 yield | 成功 | 不一致对 | p | 闭爪时序(帧)| 失败分类 |
|---|---|---|---|---|---|---|---|
| 07 | 基线(advance 0.03,有 shuffle)| 60/89 | 18/60 = 30% | — | — | 1.83 | far 16 · 扫倒 24 · 掉 1 |
| 07b | 站近 5 cm(advance 0.08)| 60/93 | 26/59 = 44% | 10 : 18 | 0.185 | 3.50 | 没蹲 15 · far 6 · 扫倒 8 · 掉 4 |
| 07c | 07 站位 + pregrasp 横向 jitter ±3 cm | 60/87 | 20/60 = 33% | 14 : 16 | 0.86 | 1.83 | far 15 · 扫倒 11 · 没蹲 8 · 掉 4 |
| **07d** | **07b 站位 + 去 shuffle**(`GYM_SHUFFLE_BAND=0`)| **60/78** | **29/60 = 48%** | **8 : 19** | **0.052** | **1.17** | 没蹲 13 · 扫倒 11 · 掉 4 · far 3 |
| s2 | 07 数据,训练 seed 42→7 | (同 07) | 18/60 = 30% | 13 : 13 | 1.00 | 3.33 | 扫倒 24 · far 11 · 没蹲 6 |
| 07e | 07d + 停靠带放宽(advance 0.04 ± 0.04)| 60/90 | 19/60 = 32% | 11 : 12 | 1.00 | 3.50 | 没蹲 18 · 扫倒 17 · far 3 |

07e vs 07d:9 : 19,p = 0.087。各臂训练 loss 0.068–0.078,数据量 21.7–22.6k 帧,推理延迟中位 70 ms(三臂核对)。

失败类别定义(`compare_arms.py`):far = 蹲下时骨盆离瓶 > 0.43 m 手够不着;扫倒 = 伸手时瓶子倒(按有无采到手指接触分两类);
没蹲 = 到桌边始终未下蹲;掉 = 闭爪后倒/掉。

## 4. 机制:策略的停位由"蹭"决定,不由专家停位决定

走近段基座速度中位(m/s)按骨盆→瓶距离分段:

| 距离段 | 07 专家 | 07 策略 | 07b 专家 | 07b 策略 |
|---|---|---|---|---|
| 0.60–0.70 | 0.26 | 0.27 | 0.26 | 0.22 |
| 0.45–0.50 | 0.30 | 0.07 | 0.04 | 0.003 |
| 0.40–0.45 | 0.41 | 0.01 | 0.08 | 0.02 |
| 0.35–0.40 | 0.04 | 0.04 | 0.22 | 0.007 |

专家在 0.40–0.50 还在 0.3–0.4 m/s 走、最后 5 cm 刹车;策略从 0.5 m 起就以 1–7 cm/s 蹭——这是 APPR_SHUFFLE
(0.20 → 1–2 cm/步)被平均后的产物,和 dart200 的"桌前发呆"同源。结果:专家 first_stop 中位挪 4 cm(0.364 → 0.324),
策略最近停靠只挪 7 mm(0.347 → 0.340)。去掉 shuffle(07d)后策略停位才真的挪进来(0.348 → 0.310),far 16 → 3。

各模型最近停靠中位 / P(蹲):07 0.348 / 1.00;07b 0.339 / 0.75;07c 0.334 / 0.87;07d 0.310 / 0.78;s2 0.352 / 0.90。
成功率随停靠距离单调下降(07:[0.25,0.30) 67%,[0.34,0.38) 18%,≥0.38 0%),专家自己也是([0.30,0.34) 89% → [0.37,0.43) 69%)。

## 5. 方差底线与"没蹲"

s2 只换训练 seed:总成功率纹丝不动(18/60 vs 18/60),但类别构成 far 16→11、没蹲 0→6、闭爪时序 1.83→3.33。
因此 ≤6 局的类别差和 ±1.5 帧的时序差是噪声;站得住的效应:站近把扫倒 24→8、jitter 把扫倒 24→11、去 shuffle 把 far 16→3、
07d 的 +18 点。

"没蹲"的三个数据侧解释全部被否定:窄带覆盖(07e 放宽后反而 18 局)、放宽覆盖(8 局里 4 局停在带内仍不蹲)、刹车静止隐门
(四臂训练数据里到桌→蹲的静止帧结构相同:最长静止段 0.60–0.66 s,静止帧 29–39%)。按 seed 对照,当作训练方差处理。

## 6. 结论与下一步

1. **最佳采集协议 = 07d**:`arrival_advance 0.08`、`recovery_timeout 15`、`GYM_SHUFFLE_BAND=0`、评测分布 spawn。
   两个成分都必要:只站近(07b)44%,站近 + 放宽(07e)32%,站近 + 去 shuffle(07d)48%。
2. 横向 jitter(07c)单独用没有总收益,但它把扫倒减半,可以叠在 07d 上作下一个单变量。
3. 下一杠杆是**数据量与 seed 数**:用 07d 协议采 200 集,每配置训两个 seed;不要再调站位带。
4. 几何上"从上方下压"不可行:dex1 指长 18.5 cm,近站位下 pregrasp 抬高 14 cm 会把指尖压在瓶顶(烟测 3/10),codex 的水平进爪是对的。

## 7. 代码改动与运行记录

已提交(`8c621fb`):`reach_config.py` 新字段(pregrasp_rise/gap/lateral_jitter/min_forward、local_reach_limit,
arrival_advance 上限 0.10);`coupled_grasp_policy.py`/`measured_grasp.py` 消费这些字段(默认 = 07 行为);
`grasp_gym.py` `GYM_SHUFFLE_BAND`;`tools/run_learn_pilot.py`;`convert_to_hf_format.py` `left_gripper`。
**未提交**(本文写作时):`reach_config.py` `arrival_advance_jitter`;`grasp_gym.py` 每局 advance 抽样;
`run_learn_pilot.py` `SQUAT_MAX` 默认 4096;`src/gr00t/experiment/launch_finetune.py` `GR00T_DATA_SEED`。

运行中踩过、已修的坑(细节在 memory):runner 用 `max(mtime)` 选 exporter session 目录在 teardown 变慢后错位一格
(改为等待新目录);批间 `stop_stack.sh` 全局 pkill 杀了别的 instance(改 `stop_instance.sh 0`);instance 1 的 keyboard
端口 5590 与 `serve_policy` 撞(烟测改 instance 3);`/workspace` 写满(单 checkpoint);其他 instance 的 MuJoCo EGL 上下文
落在 GPU0 触发 2048 MiB 显存守卫(`SQUAT_MAX=4096`);`pgrep -f` 路径片段在等待循环里自匹配死锁(改 `.done` 标记)。

保留的权重:`ready07_learn_20260907/model`(基线)与 `ready07d_decisive_20260907/model`(最佳),各 24 G;
07b/07c/07e/s2 的权重已删,其评测 trace、`paired_compare.txt`、`open_loop_val.json` 保留。

## 8. 07d 协议 × 200 集(2026-09-08 追加)

`outputs/experiments/ready07d_200_20260908/`:同一 reach_config、scene、评测分布 spawn、`GYM_SHUFFLE_BAND=0`,seed0 28000,
28 批 281 轮接受 200 集;其中 3 个 session(10:09-10:40 UTC)因同卡跑 ScaleBFM 仿真导致 deploy 自愈重启、exporter
episode 被切成 1-6 s 残段(convert 的顺序对齐看不出来,audit 的 grasp-causality 抓到),剔除后 **176 集训练、10 集 val**,
3000 steps,100 轮闭环(seed 6000,与 07d decisive 逐轮配对)。

| 臂 | 训练集 | 闭环 | 前 60 轮配对 | discordant(新赢 : 旧赢) | McNemar p |
|---|---|---|---|---|---|
| 07d decisive | 60/78 | 29/60 = 48% | 29/60 | | |
| **07d-200** | 176 | **57/100 = 57%** | **37/60 = 62%** | 22 : 14 | 0.24 |

结论:多 3 倍数据配对上涨 8 个点(前 60 轮),但 p = 0.24 仍不显著——60 轮配对分辨不了 10 个点以内的差距(memory:
`vla-baseline-round1-results`),要 2 个 seed 或 200 轮才能定。失败结构变了:07d-200 的 reach 失败停得更近(min dist
中位 0.24 m,decisive 0.31-0.42 m)但"到了桌边不合爪"仍是最大类(34/43),合爪没抬 9/43。第二个 seed 的训练是下一步。

### 8.1 ensemble N3 M0.3 筛查(2026-09-09):有害,不续

先把 §8 的 43 局失败按 trace 拆开(`outputs/experiments/ready07d_200_20260908/eval/failure_motion_20260909.txt`):
"到桌边不合爪" 34 局 = **静止型 15**(瓶子位移 0,palm_z 整局停在 0.86–0.91 m 休息位,从未降到成功局合爪的
0.71–0.74 m,手离瓶 0.23–0.29 m;6 局在 4–6 s 到桌边后彻底不动)+ **扫倒型 19**(伸手到 0.71–0.75 m,到桌后
5–10 s 内把瓶子推开 10–22 cm,之后手悬在 0.21–0.38 m 抖到超时,没有一局重新对准);另 9 局合爪时把瓶子推走
10–38 cm。静止型和 dart200 被 ensemble 拽出来的停顿同构,于是做 20 局配对筛查。

`eval_ens20/`:同 checkpoint-3000、scene、reach_config、token、seed 6000、async 0.4 s、90 s budget,只多
`--ensemble 3 --ensemble-m 0.3`,stack + policy server 全在 GPU1。

| 臂 | 前 20 局 | discordant(ens 赢 : base 赢) | McNemar p |
|---|---|---|---|
| base(§8 的前 20 局) | 9/20 | | |
| **ensemble N3 M0.3** | **3/20** | **3 : 9** | 0.146 |

逐局:base 赢的 9 局(0,1,2,4,8,9,10,13,14)全部变成"到桌边不合爪";ens 赢的 3 局(7,12,16)都是 base 的扫倒型。
静止型 3 局(11,17,18)一局没救回来。trace 里的机制:ensemble 把手臂下放推迟约 10 s 或干脆不放(round 1 手停在
休息位 0.37 m 外),合爪 grip 从未低于 0.021;round 2 反而走到 0.10 m 把瓶子扫走。三个 chunk 加权平均把伸手和
合爪这两个近似离散的转换抹平了,精度换来的 3 局不够抵消。

结论:07d-200 上 ensemble 是负向,不续到 100 局,`--ensemble 1` 仍是本线默认。和 flow200(50 vs 46)、dart200
(24→56)的历史一致——ensemble 只在"到位后整体静止"是主失败且策略本来会伸手的 checkpoint 上有用。

附:`src/teleop_free/tools/failure_anatomy.py` 的 arm-up 阈值在本线是反的(休息位 0.90 m 高于合爪高度 0.72 m),
它把 15 局从未伸手的静止型报成 "up but never near",arm-never-up 为 0 是假的。

## 9. 07d-200 第二个训练 seed(2026-09-09)

`outputs/experiments/ready07d_200_seed7_20260909/`:train_dataset / val_dataset 直接 symlink 到 §8 的 176 集
(166 训练 + 10 val),同 reach_config、scene、token、3000 steps,唯一变量 `GR00T_DATA_SEED=7`(§8 是默认 42);
训练 77 min(loss 0.171),100 局 seed 6000 逐局配对。训练 + policy server + eval stack 全在 GPU1。

| 臂 | 训练集 | seed | 100 局 | 前 60 局 | vs 07d-200 s42(100 局配对) | vs decisive(60 局配对) |
|---|---|---|---|---|---|---|
| 07d decisive | 60 | 42 | | 29/60 | | |
| 07d-200 s42(§8) | 176 | 42 | 57/100 | 37/60 | | 22:14,p 0.24 |
| **07d-200 s7** | 176 | 7 | **46/100** | 27/60 | 21:32,p 0.17 | 17:19,**p 0.87** |

同一份 176 集换个 seed 掉 11 个点,和 decisive 配对后完全持平(27 vs 29)。§8 的 "+8 点" 因此不能记为数据的贡献:
两个 seed 的跨度(46–57)已经盖住了 60 集和 176 集之间的差。3 倍数据在这条线上没有可分辨的收益,seed 方差
±5–6 点是本线的分辨率下限(memory `training-seed-variance-floor` 原先只看到 ≤6/60 的类别移动,这里是 11/100 的总数移动)。

失败结构也随 seed 大幅换形(按 §8.1 的 trace 拆法):

| 臂 | 成功 | 静止型(手不离休息位) | 扫倒型 | 伸手未合爪 | 合爪没抬 |
|---|---|---|---|---|---|
| s42 | 57 | 15 | 19 | 0 | 9 |
| s7 | 46 | **39** | 7 | 2 | 6 |

s7 几乎所有失败都是"站到桌边、手臂从不下放";s42 的模型更愿意伸手,于是把失败换成扫倒。这两种形态出自同一
份数据,只差初始化和 shuffle 顺序——所以对单个 checkpoint 的失败形态做归因(比如 §8 说"数据让 reach 失败停得更近")
是不可靠的,任何 class-level 的说法至少要两个 seed。

结论与去向:数据量这根杠杆在 176 集处已经平了;下一步不是第三个 seed,而是直接打两个 seed 共有的主失败——
到桌边后不启动伸手(s7 39/54,s42 15/43)。failure_anatomy 的 ARM-NEVER-UP 处方是对的:近桌 spawn 的 DAgger,
覆盖 "站定 → 抬手" 这个转换。

## 10. 采集耗时拆解与提速计划(2026-09-10,branch session)

**问题**:200 条 visible-expert 数据估 3.5 h,实测(2026-09-09,43 个 exporter session,350 局 / 260 OK)约 6 h。
**来源**:`tests/gate/results/bottle_*.json` 对应的 4 次 gate 跑,49 局 manifest 的 `execution.events` +
`/workspace/tmp/p03/outputs/2026-09-09-*` 的 episode 落盘时间 + `outputs/experiments/gate_bottle_main0/collect/pose_sidecar`。

每个成功 round 中位 64 s 的去向:

| 段 | 时长 | 说明 |
|---|---|---|
| `fresh_deploy`(每局 kill + 重启 deploy) | ~33 s | 重载 3 个 TensorRT engine(planner 712 MiB)+ `time.sleep(4.0)`;`BottleTaskConfig.restart_deploy` 默认 True |
| initialize | 7.8 s | 静默 IDLE 1 s + reset + 沉降 IDLE 4 s + 转身 + IDLE 2 s + ready 姿态 |
| navigate → carry_and_verify(录进数据集的部分) | 18.5 s | 885 帧 @50 Hz 中位;exporter parquet + 三路 mp4 在 complete 后 1 s 内落盘 |
| 'c' 之后 "let the write finish" 的 IDLE hold | 4 s | 实测写盘 1 s 就完 |

失败局也付重启:`[grasp]/[hold]` 失败 ~50 s,`[stand]` 失败 25 s。yield 0.62 → 每个成功 episode 1.6 局 ≈ 104 s,
加每批 43 s 起栈(9–10 局一批,≈ 4.5 s/局)≈ 110 s/条 → 200 条 ≈ 6 h。sim 本身 26.5 s 仿真时间用了 30.9 s 墙钟(0.86×)。

**杠杆(按收益)**,都未执行,裁定待实验:

1. **`restart_deploy=false`**(最多省一半)。重启的依据是 2026-08 的"第 2 局起漂移/摔倒/落点差几米";此后无重启路径已补了
   `_gym_started` 静默、`heading_offset` 补偿、IDLE 帧显式 facing 三样,但 docs/ledger/archive 里找不到修复后重测过
   `GYM_RESTART_DEPLOY=0` 的记录。实验:`--set restart_deploy=false` 30 局,`scripts/gate.py compare` 对 yield /
   end_depth / 落点漂移 / 失败字符串。成立则每局 64 → ~31 s,200 条 ≈ 3 h。不成立则查 deploy 残留状态(planner heading
   anchor 最可疑),用软复位替代进程重启。
2. 保留重启时可砍 5–7 s:'c' 后 hold 4 → 1 s;`fresh_deploy` 的 `sleep(4.0)` 与 `restart_deploy.sh` 2 s 一档的轮询改成等
   "transitioning to CONTROL"。
3. 固定停顿 3–4 s:复位后沉降 4 s(VLA 路径 1.5 s)、转身后 IDLE 2 s;碰 stand 分布,要过 gate。
4. 批次 10 → 20 局:每局省 2 s(exporter 线程增长与 pid 上限要先复测,`EXPORTER_THREADS=4` 之后没量过)。
5. yield 0.62 的 19% "stand infeasible" 是 visible 协议去掉 reposition 的代价(07d 0.71),只能在刹车后判定;协议决定,不为提速改。

**不做**:双 stack 并行(pid cgroup 3840,一个录像栈 ~2500 线程;deploy 对 80 ms 卡顿即退出,09-09 光外部 GPU1 任务就截断
8/40 session);lockstep 采集(`docs/handoff/sonic_lockstep_collection_plan.md`,已搁置,是长期解)。

## 11. 采集提速实验 1:`restart_deploy=false` + 复位 reanchor(2026-09-10,branch session)

§10 杠杆 1 的实验。代码 b2310c8:`round_phases._phase_spawn_and_settle` 在 deploy 没被本局重启时(`_gym_started`)
在 `start_control` 后加 `hold(1.5)` + `stream_reach.reanchor(p)`(双臂线每集用的那套),默认重启路径字节不变;
`prepare_ready` 失败时打印 ramp residual / joint err / push 数。

**第一轮(03:57–04:39)作废**:容器 cgroup CPU 配额 15.36 核(`cpu.max`,`nproc` 的 32 是宿主机)被
`/workspace/code/video_grasp` 的 benchmark 占满,sim 实时因子 0.18、gym 循环 9 Hz(3 s 内只 push 24–31 次,正常约 150),
A 臂(默认重启)也 1/10、0/10、0/10。它留下一条有效结论:**纯关重启(无 reanchor)0/10**,第 3 局起"站起来时走偏 0.5 m"、
复位后摔倒,2026-08 加重启的那个病复现;加 reanchor 后转身漂移 0.36 m → 0–2 cm。

**第二轮(04:54–05:43,CPU 余量 6–13 核、无 throttle)**:3 批 × 10 局,同一 case bank(`cases_0..2.json`,seed 29000–29002)
配对,交错 A0 B'0 A1 B'1 A2 B'2。A = 默认(每局重启),B' = 不重启 + reanchor。

| | A(重启) | B'(不重启 + reanchor) |
|---|---|---|
| 成功 | 16/30(7, 7, 2) | 18/30(5, 6, 7) |
| 不一致对 | A 独胜 6 | B' 独胜 8 |
| 每局中位(全部 / 成功) | 62.7 s / 63.9 s | **33.7 s / 34.5 s** |
| initialize → navigate | 7.9 s | 11.5 s(+1.5 s hold + toggle) |
| 刹车退出深度 `end_depth_m` 中位 | 0.301 | 0.301 |
| 停站 `park` / `first_stop` 中位 | 0.301 / 0.307 | 0.332 / 0.335 |
| 失败 | stand infeasible 3、path blocked 3、IK 不跟踪 2、转身超时 3(同一 case)、其他 3 | stand infeasible 4、**local alignment disturbed target or base 2**、转身超时 3、其他 3 |

`gate.py compare`(以本轮 A 为基线,`tests/gate/results/restart_ab_*_20260910.json`):13/15 PASS;两条 FAIL 都可解释——
`stage_s.initialize` 7.9 → 11.5 s 是设计代价,`failure_reasons` 新增 "local alignment disturbed target or base"(2/30)。

**裁定**:每局 64 → 34 s(−47%),30 局配对上成功率无差(16 vs 18,不一致 6:8,分不出方向),站位阶段零失败。两个要盯的
差异:不重启时底盘停得远 3 cm(`park` 0.332 vs 0.301,刹车退出深度不变,即刹车后的滑移少了或 IDLE 漂移方向变了),
以及 2/30 的 "local alignment 时底盘被扰动"——都指向 deploy 跨局残留的步态相位/IDLE 漂移(memory `planner-idle-drifts`),
reanchor 只修了朝向锚。下一步:用它采下一批 200 条(约 3.2 h 而不是 6 h),用 `check_dataset` 和闭环评测判断这 3 cm 是否
改变训练分布;若要保守,先在 quiet 机器上再跑 30 局把不一致对的方向定下来。**采集时 CPU 配额余量 ≥ 6 核是前置条件**
(`scripts/experiments/restart_ab_runner_20260910.sh` 里的 `cpu_free` 检查),否则任何结论都是噪声。
原始数据:`outputs/experiments/restart_ab_20260910/`(manifest2_*.jsonl、gym2_*.log、cpu2.log、cases_*.json)。

**2026-09-10 裁定(用户)**:采集默认改为不重启——`BottleTaskConfig.restart_deploy=False`、`configs/tasks/bottle_visible.json`
同步;`eval_closed_loop` 固定 `RESTART_DEPLOY=True`(评测路径没在无重启下量过)。gate 基线换成本节 B' 的 30 局
(`tests/gate/baseline_bottle.json`,旧基线留作 `tests/gate/baseline_bottle_restart_20260909.json`)。


## 12. SONIC 弯腰能力审计(2026-09-10,Codex)

用户问题:本地 SONIC 过去几乎不交付弯腰,协作者真机 G1 可以弯腰,是否本地有问题。
本次只读代码、检查运行参数、做 CPU 离线探针;没有改变专家行为、模型、仿真或正在执行的采集。
GPU0 为 cf 的 SONIC visible200, GPU1 为用户授权的 ScaleBFM 200-success;不可插入物理探针。

**确认的事实**:

- 当前 ready-grasp 的 `ready_q()` 腰 12:15 为零;主要抓取 IK、measured/recovery/coupled
  路径使用 `use_waist=False`。它规划的是下蹲配合手臂,未把主动腰前倾作为抓取自由度。
  入口 `src/locomani/tasks/bottle/ready_grasp_policy.py:35,78,308,336`、`measured_grasp.py:217`。
  `use_waist=False` 不代表物理关节被锁死,实际身体仍可能有反馈运动。
- 通用 `StreamedArm`/`PlannerArm` 的 IK 腰 pitch 范围被设为 [-0.05,0.20] rad(最大约11.5°),
  MJCF 物理范围是 [-0.52,0.52]。该 IK 限制不拦截直接 `move_to(q)` 的外部关节目标。
- `upper_body_position` 是 encoder 前的17维参考动作替换,不是电机目标直通。
  C++ `g1_deploy_onnx_ref.cpp` 约797行替换未来参考帧,约3320行由 decoder 输出计算电机目标。
  planner 下半身由 k-planner 生成;本地 streamed 路径的根姿态参考取 yaw-only;
  commander 的 VR 三点路径用名义 neck 点,也不等价于人的全身弯腰动作输入。
- 从 `78536ae:src/teleop_free/diag_lean.py` 恢复阅读旧测试:腰 pitch 0..0.52 rad,每档保持2.5s,
  默认手臂固定到伸手姿态,分别走 planner/streamed;主要测腰相对关节角,并记录骨盆姿态。
  没有髋/腿/根 pitch 协同参考,没有以躯干世界 pitch 为主指标。因此测试不能证明
  “SONIC/G1不支持弯腰”。直接 q14 赋值绕过上述0.20 IK限制,不能用该限制解释旧2–10%。
  原始 `outputs/diag/lean.json` 已随 outputs 事故丢失,本次不能复核那个比例。
- 活跃 deploy 使用 `policy/release/model_{encoder,decoder}.onnx` 和其 observation_config,
  `--input-type zmq_manager`、V2 planner,不是本地 `policy/v1_1` 或 low-latency 版本。
  官方区分 Planner/Pose 全身动作跟踪以及不同 checkpoint;协作者的入口/模型/增益尚未提供。
  参考: https://github.com/NVlabs/GR00T-WholeBodyControl 和
  https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html 。
- 仿真存在可选 waist pin 钩子,默认 None;未发现当前抓取调用主动 pin waist 的代码。
  这不是读取到了活跃进程内 pin 状态,不能据此证明运行时一切正常。

**离线实验及边界**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
scripts/experiments/sonic_lean_audit_20260910.py \
  --out outputs/diagnostics/sonic_lean_20260910/offline.json
```

实际 AgentCommander 序列化 → 解析二进制 payload → 读取 C++ 关节映射常量并重建参考 →
实际 release encoder/decoder(CPU单线程)。没有 socket、物理步进或 GPU。
0、0.1、0.2、0.3、0.4、0.52 rad 六档 wire/ref 全部等于请求;非零档 token 改变
8、18、28、33、37/64维。0.52档合成站立观测下单步腰目标相对0档改变0.1519 rad。
该数值**不是交付率**,合成本体历史不代表真实闭环轨迹。
结果 JSON 含 encoder/decoder/obs config/C++映射文件 SHA256。
负对照:在临时进程 monkeypatch `AgentCommander.set_upper_body` 把 pitch 清零,探针断言按预期失败;
正常探针及该文件 Ruff format/check 通过。它能检测丢信号,不能检测实际 C++ 活跃模式、
反馈稳定性、PD扭矩不足或仿真时序造成的交付差。

**结论与未验证解释**:
当前专家主动排除腰部参与是事实;历史“这条路是死的”属于过度外推,教程已收窄表述。
未找到离线链路关节映射/序列化丢失,不能据此认定完整在线链路无 bug。
孤立、不协调的参考被跟踪策略削弱,以及接口/模型/增益差异,都是待验证解释,
尚未复现和锁定旧2–10%的物理根因。不能直接放开 waist IK 就宣称修好。

**下一步受控验证(未执行,等采集结束并核对资源)**:

1. 取得协作者的控制入口、encoder/decoder/obs config 哈希、PD增益、动作参考和幅度。
2. 固定同一 scene/模型/初态/控制频率/参考时长/增益,比较孤立腰 pitch 与协作者的协调
   全身参考(含髋膝踝/根姿态);每种分别使用中立手臂、伸手姿态,至少5次配对复位。
   同轨迹可表达时再跨 Planner/Pose/VR 比较,一次只改变一个因素。
3. 分层记录 requested → wire → 实际deploy参考/模式 → token → decoder电机目标 →
   实际腰/髋/骨盆/躯干世界pitch,加跟踪RMSE、稳态偏差、响应时间、骨盆高度、跌倒和接触。
   参考即消失定位输入层;参考存在但电机目标小定位学习控制映射;电机目标足但实际角小
   再检查扭矩饱和/PD/时序/接触。不能用腰关节角代替躯干世界倾角。
4. 若本地能复现协作者动作,再研究将视觉抓取目标变成协调参考。专家修改后先做短 gate,
   数据训练及固定预算 VLA 闭环结果才决定是否替换当前策略。


## 13. GPU1 物理复现启动(2026-09-10,Codex)

用户明确授权开始验证、使用 GPU1。ScaleBFM 200集仍在该卡采集;资源核对后启动独立 SONIC
instance 1(DDS=1, ports 5565/5566/5567/5590/5591/5592),无 exporter。启动后 CPU约3.6/15.36核,
pids1500/3840,无throttle,GPU1约11GB/32GB;未停止两条采集,未修改专家或模型。

**已完成先导**:中立手臂、planner/streamed 两条路径、0..0.52rad六档,每档2.5s。
planner最高档实际腰pitch约0.008rad,streamed约0.010rad;复现了很低的孤立腰指令交付。
时序/FK躯干世界pitch见 `outputs/diagnostics/sonic_lean_20260910/live/neutral_0.json` 及两个trace文件;
`pilot_verdict.json` 使用最后1s均值和0档基线计算增益,50%阈值只用于诊断弱交付,不是任务成功标准。
这个先导没有录deploy debug,不能由它判断弱指令发生在哪一层。

**运行中,不是最终结果**:PID2482154,五次中立/伸手各自重复,每次都有planner/streamed且切换前reset。
新增debug记录token_state/last_action等原始字段;`body_q_target`不含upper覆盖,不能当覆盖后的encoder参考。
不改变两条路径的原始控制语义。参考文件/脚本hash、局限、参数保存在live/plan.json。
reset没有重置deploy内部RNG/历史,这还不是严格同随机状态的paired实验。
尚无协作者参考,本批只复现和分层诊断孤立腰指令;协调全身动作及模型/增益比较未执行。

启动方式(需先确认instance1归本实验所有且资源可用):

```bash
INSTANCE=1 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 bash scripts/stack/start_stack.sh 'SONIC isolated waist diagnosis'
bash scripts/experiments/sonic_lean_batch_20260910.sh
```

batch逐case检查CPU余量>=6核、无限流、pids<2800,每case180s超时,拒绝覆盖已存在结果。
完成或错误都会停止instance1并写live/batch.exit;batch.log给出当前case。当前命令为后台执行,
不要再次启动。下一步先读已有结果和进程状态;分辨实际电机目标与实测角,再选择下一个干预。


**进度复核(同日09:31 UTC)**:batch已退出1,不是仍在运行。neutral_1完成后、reach_1之前,
资源检查CPU余量4.8857核<6,无限流,pids2025,触发保护并关闭instance1。
已完成neutral_0先导+neutral_1正式重复,各12档结果;伸手和剩余重复未做。
ScaleBFM进程仍活跃,19/200 accepted、19次完成尝试。

在线debug进一步分层:output_interface/output_interface.hpp:339–346明确把last_action转换成
MuJoCo顺序并乘scale加default,因此读其[14]已经是腰电机目标,不能再次换序/缩放。
neutral_1最高0.52rad档末50样本,planner电机目标均值0.00694rad、实测0.00212rad;
streamed电机目标-0.00141rad、实测0.01120rad,躯干世界pitch分别2.26°/6.96°。
原始与汇总在live/neutral_1.json.*.trace.json、live/online_layer_summary.json。
这说明不能单独归因“电机收到大弯腰指令但PD/扭矩交付不了”:模型输出的腰目标本来就小。
仍未证明小输出由参考不协调还是在线模式/编码/模型行为造成;没有覆盖后的encoder观测。
观测age最大<3ms,但只有case前后CPU采样,不能证明case全程无资源竞争。
下一步资源恢复后补足剩余重复,并以协调参考干预;勿重复先导或覆盖neutral_1。


**用户要求恢复(同日)**:新5秒采样CPU余量12.34核、无限流、pids1168,GPU1显存9664/32607MiB。
启动独立instance1恢复supervisor PID2496602。batch脚本新增严格断点恢复:已有case须12档完整,
两条channel trace无跌倒且日志有完成标记,才跳过;部分/异常结果拒绝覆盖。neutral_1保留,
从reach_1继续到5次重复。原batch.exit/pid带时间戳归档,日志追加,未降低资源阈值。
恢复状态见live/resume.json、resume.log、stack.log、batch.log;启动及批次结束清理仅instance1。

恢复续记:首次重启后余量4.90核再次触发退出,未执行reach。将资源检查改为最多等10分钟,
要求连续3个5秒窗口均>=6空闲核、无限流、pids<2800,阈值未放宽。supervisor2497422
于09:38:32 UTC开始reach_1,启动前连续余量11.44/9.50/9.68核,neutral_1验证完整后跳过。


**本批完成**:batch.exit=0,instance1进程已关闭。neutral/reach各5次正式重复,每次两个通道、六个腰角,
共10个case/20次通道扫角/120个角度条件;均日志正常完成,无记录到的跌倒。先导neutral_0不计入汇总。
结果 `live/phase1_summary.json`(包括每次结果和范围)。最高请求0.52rad=29.8°时,
末50个样本均值再跨5次平均:

|姿态|通道|实际腰角°|躯干世界pitch°|debug电机腰目标°|
|---|---|---:|---:|---:|
|中立|planner|0.04|1.95|0.95|
|中立|streamed|0.68|6.33|1.11|
|伸手|planner|1.82|3.46|-9.57|
|伸手|streamed|2.30|6.31|-11.11|

新的重要区别:伸手时电机腰目标明显反向,并非所有姿态都只是输出接近零。
符合“模型为伸手姿态做平衡补偿”的解释,但未通过协调参考或编码输入干预验证,
不能宣称根因已锁定。这个结果进一步排除了单纯将upper_body当电机位置指令的解释。
当前整批仅验证孤立腰参考,不等价全身弯腰能力测试。下一步应捕获覆盖后的encoder输入并
对比协调全身参考,而不是继续重复同一扫角。原ScaleBFM采集32/200 accepted,32次完成尝试。


## 14. 协调参考与实际encoder输入验证启动(2026-09-10,Codex)

用户授权下一阶段。独立GPU1/instance1,supervisor2545023,输出
`outputs/diagnostics/sonic_lean_phase2_20260910/`。当前启动后等待连续CPU余量检查,尚无正式结果。
资源要求与前阶段相同,不干扰继续运行的ScaleBFM采集。

参考使用本地官方`squat_001__A359`第85帧,躯干世界前倾34.97°,原轨迹骨盆高0.557663m。
CSV关节顺序实测为IsaacLab(部分旧Python说明误写MuJoCo):以根quat+FK对比CSV torso quaternion,
帧0/79/140按IL解释误差<0.0001°,按MJ解释误差16.7/113.4/18.8°。此次按IL读入后转换正确。
这不是协作者真机轨迹复现,也不是完整动态蹲起回放。模式0消费joint+root orientation,
没有额外注入root position;原轨迹骨盆高度仅供筛选幅度参考。

|条件|变化|
|---|---|
|legacy_waist_052|与前阶段对应的0.52rad孤立腰参考,用于私有build回归检查|
|waist_only|同一参考第85帧腰pitch0.439575rad,其余保持初态|
|legs_waist|用参考的12个腿关节+3个腰关节;双臂与根姿态保持初态|
|legs_waist_root|在上一项上加入参考根roll/pitch,去除参考原始yaw|
|full_pose|再加入参考双臂,成为模式0支持的全身姿态参考|

每条件3重复,3s过渡/4s保持/3s返回;奇偶重复反转顺序,每条件重置。
各条件沿用现有pose消息的零joint_vel,因此只解释稳态最后1s,不声称动态参考一致性。
reset不重置deploy内部历史/RNG,不能当严格同状态配对。判断依据已预登记在plan.json。

**真实输入记录**:复制当前C++ translation unit到输出目录,在GatherTokenState实际Encode调用后
写出同一个float input buffer及token,保留walltime/mode/dim字段。原始C++/生产binary未改。
使用原CMake编译选项及库做单进程编译;发现旧state_logger archive与新头文件ABI不一致,
在私有目录重编state_logger对象解决链接,并保留build日志/manifest,故先跑legacy对照检验行为。
记录是实际覆盖后的buffer,不是`body_q_target`或离线重建。文件末尾可能有进程关闭时未完整写出
的record,解析器忽略不完整尾部并报告字节数;中间格式/维度错误直接失败。

命令:

```bash
python3 scripts/experiments/sonic_encoder_capture_build_20260910.py
bash scripts/experiments/sonic_lean_phase2_20260910.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/sonic_encoder_capture_report_20260910.py
```

当前已经启动,不要重复执行。launcher独立lease/拒绝已有encoder.bin,结束清理只instance1;
监看cases.log/resources.jsonl/progress.json/run.exit。新增Python Ruff检查与shell语法检查通过。
后续先验证legacy低交付及tensor映射,再解读协调条件;若资源等待超时,保留输出再决定续跑。


**启动结果复核**:本次phase2没有进入动作测试。resource_gate等10分钟后超时,run.exit=1,
instance1已清理,没有case文件或encoder.bin。末尾CPU余量8.1/8.3/8.1核仍streak=0;
按gate实现,free>=6却失败说明该窗口throttled_usec增加(当前JSON未保存其差值,不能报具体毫秒数)。
因此不能把此次当作协调参考失败或编码器实验结果。后续重试先检查限流与资源采样规则。

## 13. visible200_20260910:第一份不重启 deploy 的 200 集数据(2026-09-10,session cf)

配方 = `configs/tasks/bottle_visible.json` + `configs/pilot/learn_200.json`(seed0 30000,每批 10 局,holdout 10),
`through=convert`,instance 0 / GPU0,07:02–10:43 UTC。runner `scripts/experiments/visible200_20260910.sh`。

| 项 | 数字 |
|---|---|
| 批 / 局 / 留存 | 34 / 335 / **200**(yield 0.60;每批 2–8) |
| exporter session 核对 | 34/34:manifest 成败 = `discarded_episode_indices`,留存 episode 831–1433 行(中位 922) |
| 转换 | rc=0,1425 s;train 190 集 81,425 帧、val 10 集 4,743 帧,25 fps,910 MB + 52 MB |
| `check_dataset` | train 32/32 PASS,val 32/32 PASS(合爪时腕距 ≤30 cm 190/190、合爪后抬升 >3 cm 190/190、末端刚性 190/190、视频-表格 lag 0) |
| 每局(成功) | 32–39 s |
| 第 1 局失败 | **34/34**,全部在出生阶段(转身超时 22、物体不在 ego 视野 12)→ §12 |
| 第 2–10 局失败(101) | stand infeasible 72、合爪倒瓶 8、IK 不舒适 6、到位够不到 6、掌内滑脱 3、其他 6 |

**外部干扰**:07:45–08:30 另一 session 的 crate 采集 + Isaac 渲染上了 GPU0(它的端口探针在我这边的批间重启缝隙里看到 GPU0 空),
第 8–15 批 yield 43/80 = 0.54,其余 157/255 = 0.62;没有 deploy 自愈重启,session 核对与审计都干净,留存数据未见损伤,
但那 8 批(session 07-45-01 … 08-24-52)在训练后若出现分层差异先查它们。stand infeasible 72/101 明显高于 09-09 的 19%,
和"不重启 deploy 停站远 3 cm"(§11 `park` 0.332 vs 0.301)方向一致,待训练评测判断是否影响分布。

**裁定**:数据可用于训练;第 1 局的 10% 浪费由 §12 的修复消除(验证见 §12)。


**资源恢复后实际开始**:用户询问能否开始,新15秒采样总CPU0.27/0.24/0.25核,无限流,两GPU空闲。
恢复supervisor2576211,GPU1/instance1;旧超时日志归档,无旧动作需重跑。
legacy_waist_052第1次完成,encoder.bin实际生成并可解析:输入1762维、token64维、mode0。
最后1s50次编码中,腰参考0.51999998rad对应请求0.52,全部10帧29关节参考最大误差2.85e-8rad;
零dq与发送协议一致。躯干实测5.93°,腰0.315°,平均电机腰目标3.76°。
这确认至少本条件“参考在进入encoder前丢失/被裁剪”不成立,私有build仍复现低交付。
数据与分析:encoder_report.json;协调条件继续运行。此时ScaleBFM200已采集/转换/统计全部完成,
200episodes/49387frames/400videos,未训练。


**Phase2全部完成**:run.exit0,15/15case完整且未记录跌倒,instance1已清理。
实际encoder记录23365次,1762维input/64维token,报告覆盖15个稳定保持窗口;末尾7072字节不完整记录
被忽略(不影响已完成窗口)。summary.json、encoder_report.json保存逐次数据及组均值。

|条件(各3次)|目标躯干pitch°|实测躯干pitch°|实测腰pitch°|电机腰目标°|
|---|---:|---:|---:|---:|
|legacy0.52|29.79|5.35|0.27|-4.06|
|waist_only|25.17|4.90|0.32|-9.03|
|legs_waist|25.17|7.38|1.43|-6.03|
|legs_waist_root|34.97|13.06|1.16|-11.12|
|full_pose|34.97|16.52|7.14|-46.92|

全身组实测躯干3次16.56/16.56/16.46°,腿+root组14.07/12.60/12.51°。
所有条件实际encoder十帧关节参考与请求最大误差<=5.89e-8rad,mode均0,保持段dq为0。
额外验证anchor:取实际输入601:607的6D旋转,重建第三列,与最近debug的base_quat相乘得到
世界参考根姿态;与请求root_quat比较pitch。无root条件约0°,有root条件9.83°,最大误差0.05°,
结果encoder_root_check.json。该检查按墙钟近邻对齐,不声称完整6D旋转逐位一致。

结论:在这些条件中,腰/腿/根姿态指令没有在encoder前丢失或被裁剪。
加入协调参考让躯干前倾从4.90°逐步到16.52°,所以孤立腰测试低估了全身弯腰表现;
但完整静态参考仍明显低于35°目标,不能宣称跟踪问题解决。反向电机目标依然存在,
尤其完整姿态平均-46.92°,不能用简单“电机收到+35°但不动”描述整个链路。
仍待验证:动态官方参考(当前均零dq静态)、checkpoint/训练分布、模拟PD/负载/默认姿态一致性。
不把这个结果外推为真机G1不能弯腰或协作者做法已复现。没有改生产专家/控制器或模型。


## 15. 官方原生动态回放与模型/PD一致性 (2026-09-10, Codex)

用户授权 GPU1 验证。产物根目录 `outputs/diagnostics/sonic_native_phase3_20260910/`。

事实与已完成检查:
- 原生 C++ CSV loader 回放 `squat_001__A359`，424帧/50Hz，8.48s；包含关节速度和未来帧，绕开 Python 静态pose路径。
- 第一次 native_release_1.json 只有1帧，开始play时骨盆已低至0.119m。模型加载期间失控、start后等待4秒是初始化混杂因素；本次不计入动态跟踪成败。
- 修正诊断脚本在 Init Done 后reset，读取带新wall时间戳的状态，再立即start/play。原始失败产物保留；重试写入 retry_reset/，禁止覆盖。
- 重试runner PID2600051已经启动，但最近资源检查free约0核，尚未创建仿真；门槛连续3个5秒窗口free>=6且无限流，最多等待10分钟。不得将资源排队说成动作测试已开始。
- sim_parity.json: 29身体关节range/armature/damping/frictionloss一致，公共命名身体mass/inertia一致。
- parity_report.json: 当前Dex1身体总质量35.864952kg，官方本地Dex3为36.1652352kg；不能声称完整机器人动力学一致。
- 两份XML重力、solver、integrator、iterations、tolerance和XML dt=.002一致；configs.py默认sim_frequency=200覆盖成dt=.005，base_sim.py明确应用。该默认覆盖本身不是已确认bug。
- 初次失败样本LowCmd全部29关节kp/kd与当前C++常数匹配，和6位CSV的最大误差4.55e-5/4.27e-6；腰pitch实际kp28.5012/kd1.81445。不能把未生效的YAML高增益当实际PD。
- release和sonic_v1_1模型/配置SHA256保存在parity_report.json；私有诊断binary路径继承phase2。当前未更换生产模型/PD/专家逻辑。

假设与待判别问题:
1. 静态参考缺少动态速度/未来时序影响跟踪：有效原生回放若明显改善则支持，须先通过初始化检查。
2. 初态/朝向对齐导致回放失败：即使修正加载后reset，默认站姿仍不等同参考首帧，需要看preplay和首段误差后单独对照，不能直接归因模型能力。
3. 模型与观察配置搭配、手部质量/碰撞、仿真时序影响：有效基线后逐项改变；当前质量差仅事实，不是根因证据。
4. PD与训练设置不同：只确认了LowCmd和部署源码一致，尚无训练PD或协作者真机配置，不能宣称训练/真机一致。

恢复与命令:
```bash
cat outputs/diagnostics/sonic_native_phase3_20260910/retry_reset/runner.json
tail outputs/diagnostics/sonic_native_phase3_20260910/retry_reset/resources.jsonl
cat outputs/diagnostics/sonic_native_phase3_20260910/retry_reset/run.exit
# 确认旧进程已结束且没有已完成case后才能重启；成功case禁止覆盖
bash scripts/experiments/sonic_native_replay_20260910.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/sonic_native_parity_20260910.py
```

脚本Python编译及shell语法检查通过，离线parity脚本实跑通过。下一检查点读取有效动态trace、实际encoder dq/future blocks、reference帧进度和躯干误差；初始化无效先修诊断协议。任务root<.4判倒不用于该深蹲（参考最低root约.332）；仅<.15保护中止也不是完整动态成功判据。


Phase3状态更新（2026-09-10 12:03 UTC）：retry_reset资源gate等待10分钟后超时，run.exit=1，runner PID2600051已退出，未启动重试仿真。CPU占用溯源为Claude3265590的Playwright Chrome SwiftShader PID2598614，3秒实测14.76核/15.36核配额；未停止对方进程。完整恢复入口见项目根HANDOFF.md。

## 16. Phase3 结果：有效动态回放 + release / sonic_v1_1 单因素对照 (2026-09-10, session 29)

GPU1/instance1，用户已授权。产物 `outputs/diagnostics/sonic_native_phase3_20260910/{retry_reset_02,v1_1}/`。
未改动生产模型、PD 或专家控制；只改诊断脚本与启动协议。

### 16.1 让回放变成"有效"的启动协议

等 `Init Done` → reset → `]` 接管 → **1 秒沉降** → 才 `t`。沉降窗口里骨盆从 0.890 m 落到 0.786 m、ncon 从 0 到 33
（reset 是瞬移，采样在物理步进之前，机器人当时还悬空）。之前 `]t` 连发是在悬空态开始播放，这就是那条 0.119 m
无效记录的成因，不是模型问题。两次运行各 3 遍，全部 `complete`、run.exit=0、无跌倒、**423/424 帧、正好 50.00 fps、8.46 s**。

### 16.2 结果（3 次均值，取参考最深前倾帧；参考躯干 40.3°、腰 29.3°）

| checkpoint | 躯干 pitch | 腰 pitch | 腰 RMSE | 躯干 RMSE | 骨盆 z RMSE |
|---|---:|---:|---:|---:|---:|
| policy/release | 14.5° | 8.1° | 12.9° | 12.4° | 0.055 m |
| policy/sonic_v1_1 | 25.3° | 21.4° | **3.9°** | **5.7°** | 0.054 m |

同场景、同 PD、同协议、同参考，唯一变量是 checkpoint（及其自带 observation 配置）。腰 RMSE 差 3.3 倍。
重复性：release 腰 8.3/8.2/7.9，v1_1 腰 21.4/21.5/21.2。

**动态并不能救静态弱跟踪**：release 的动态回放（8.1°）与 phase2 静态协调参考（腰 7.14°/躯干 16.5°）同量级。
handoff 假设 1 判否。顺带确认 encoder 的 10 个参考槽位是 `k, k+5, …, k+45`，确实含未来 0.9 s。

### 16.3 已用数字排除的环节

- 参考 → encoder：捕获参考块与官方 CSV 最近邻匹配，最大误差 `2.1e-7 rad`。
- 参考自洽：用官方 `g1/g1_29dof.xml` 对 CSV `joint_pos` 做 FK，躯干 pitch 与 CSV `body_quat` 逐帧吻合到 0.01°
  （frame 0/45/79/150/330 = 2.17/38.47/40.28/9.03/16.13）。所以"腰 pitch 为正 = 前倾"在参考、官方 MJCF、本仓场景三者一致。
- 观测 → deploy：base/torso pitch 与 privileged 真值最大差 1.2°（release）/ 2.3°（v1_1），关节 ≤1.5°。
- 指令 → sim：`last_action` 与 sim 实收 `cmd_q` 全 29 维最大差 0.05 rad。
- 配置维度：deploy 自身校验两边都 `Dimension match`（release 1762/994，v1_1 1751/994）。

### 16.4 仍未解释、且**不是** checkpoint 差异的现象

`waist_pitch` 是唯一与自身参考**反相关**的指令通道：6 次运行全部 −0.90 左右，参考要 +29.3° 时电机目标约 −60°；
腿/臂各通道都是 +0.95…0.99（29×29 相关矩阵见各目录 `channel_audit.json`）。单通道符号翻转排除了映射错位
（错位是成对交换）。因此实际弯腰不是这条指令驱动出来的：腰 kp = 28.5（`policy_parameters.hpp` 里
`2.0*STIFFNESS_5020`，按 armature×ω² 算），而实测该关节的重力保持力矩就有 12–23 Nm。两个 checkpoint 同样反相，
所以这是共有的约定/训练问题，**不要写成 release 独有的 bug**。

`policy/release/observation_config.yaml` 是第三份手改配置，既不等于 `observation_config_sonic_release.yaml`
也不等于 `_sonic_v1_1.yaml`（多 `motion_root_z_position*` 共 11 维，正好对上实测 1762 vs 官方两份写的 1751；
anchor 用非 heading 归一化版本）。`policy/sonic_v1_1/` 是自洽的模型+配置对。

### 16.5 脚本与恢复

```bash
cd /workspace/code/locomanipulation
export PYTHONPATH=$PWD/src
# 重跑（换目录，禁止覆盖已有 case）；可选 SONIC_NATIVE_VIDEO=1 录像
SONIC_NATIVE_OUT=$PWD/outputs/diagnostics/sonic_native_phase3_20260910/<新目录> \
SONIC_NATIVE_POLICY=policy/sonic_v1_1 bash scripts/experiments/sonic_native_replay_20260910.sh
# 离线分析（不占 GPU）
SONIC_NATIVE_OUT=<目录> third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/sonic_native_report_20260910.py
SONIC_NATIVE_OUT=<目录> third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/sonic_native_channel_audit_20260910.py
```

`video_release/` 是 12:34 UTC 被中止的录像对照残留（pick-place session 认领 GPU1，我方在 CPU gate 内退出，
从未创建 stack），不是结果目录；重试请换名字。

### 16.6 三联对比视频（2026-09-10 12:55 UTC）

`outputs/diagnostics/sonic_native_phase3_20260910/compare/`
- `reference_vs_release_vs_v1_1.mp4`（1320×744，50 fps 实时，8.48 s）
- `reference_vs_release_vs_v1_1_half_speed.mp4`（同帧 0.5× 播放）
- `key_frames.png`（frame 45/79/330/400 拼版）、`frames/`（424 张 PNG，可重编码，132 MB）

生成方式：`scripts/experiments/sonic_native_render_20260910.py`，**离线运动学回放**已录轨迹，不新起仿真
（当时 GPU0 在训练、GPU1 在跑 pick_place pilot，CPU 只剩 5.9 核，起不了 stack）。参考格来自官方 CSV
（root z + quat + joint_pos 经 isaac_to_mujoco），两个交付格来自各自的 privileged trace，按 `play_elapsed`
重采样到参考的 50 Hz 栅格。三格共用生产场景 XML、同一相机，root 的 yaw 与 xy 被去掉（`deyaw()`），
所以格间差异只有姿态。顶栏数字是逐帧的 torso pitch / waist pitch / pelvis z。
渲染用 EGL on GPU1，全片 7.7 s 墙钟、约 2.5 核，对当时在跑的 pilot 影响可忽略。

### 16.7 渲染脚本 bug 修正与视频 v2（2026-09-10 13:0x UTC）

用户在 v1 视频里看到"下蹲时夹爪飞出去"。**是渲染脚本的 bug，不是仿真**：场景模型 free joint 之后的关节顺序是
`腿12 | 腰3 | 左臂7 | 左夹爪2(SLIDE) | 右臂7 | 右夹爪2`，我按 `qpos[7:7+29] = joints` 平铺，于是
`right_shoulder_pitch ≈ −2.0 rad` 被写进夹爪的**平移**关节 → 夹爪平移 2 米；右臂整体错位 2 格。
腿(0-11)和腰(12-14)在错位之前，所以 v1 视频里的躯干/腰角度和结论不受影响。
修法：按关节名取 `jnt_qposadr`（`body_qpos_index()`）。已记入 memory。

v2 增加：
- 两个交付格叠**橙色参考轮廓 ghost**（同相机、同去 yaw 根位姿，所以轮廓与交付之间的空隙就是没交付的部分）
- 底部两条**实时曲线**（waist pitch / torso pitch，白=参考、粉=release、绿=v1_1）＋播放头
- 关掉地面反射（`mat_reflectance=0`），否则镜像会让 ghost 描边翻倍

渲染后端：EGL 当时瞬时失败（另一 session 的 stack 正在启动；13:20 复测 EGL 正常），改用 OSMesa 软件渲染，
CPU 当时空闲 12 核。这是**显式的诊断例外**，不适用于正式采集。

### 16.8 弯腰的物理上限

`waist_pitch` 关节限位是 **±0.52 rad = ±29.79°**，仓库里每一份 G1 MJCF（官方 `g1_29dof.xml` /
`g1_29dof_old.xml` / `g1_29dof_with_hand.xml`、AMO 各版、`assets/scalebfm/g1_29dof_fk.xml`）都一致。
官方 13 段参考动作里，11 段的腰 pitch 峰值**正好等于 29.8°**——参考数据本身被限位削平了：
`squat_001__A359` 有 **118/424 帧（28%）**的腰 pitch 精确等于 0.520000 rad。
所以这台机器人"弯腰"最多 30°，参考里 40° 的躯干前倾 = 骨盆 pitch 11° + 腰 29.8°。
要更大的前倾只能靠髋和下蹲，不是腰。

## 17. release vs sonic_v1_1：干净的 20 局 A/B（2026-09-10, session 29）

`outputs/experiments/v11_bottle_ab20_20260910/`。instance 0 + `CUDA_VISIBLE_DEVICES=1`（GPU0 是别的 session 的训练）。
同一个 20 局固定 case bank（`cases20.json`，seed 7100，profile `configs/tasks/bottle_visible.json`），release 先跑。
前 10 个 case 与 `v11_bottle_pilot_20260910/cases10_v3.json` 逐字节相同，可做可复现性核对。

### 17.1 成功率

| | release | sonic_v1_1 |
|---|---:|---:|
| 成功 | **9/20** | **0/20** |

9 个不一致 case 全部是 release 赢，精确 McNemar **p=0.0039**。
可复现性：前 10 个 case，release 4/10（上一轮 5/10），v1_1 0/10（上一轮 1/10）。

### 17.2 行为差异（均值±sd，t = Welch t of v1_1 − release）

| 量 | release | sonic_v1_1 | t |
|---|---:|---:|---:|
| 到位横向偏移 (m) | −0.214±0.052 | −0.324±0.052 | −6.38 |
| park 距离 (m) | 0.377±0.044 | 0.498±0.092 | 5.05 |
| 到位前向距离 (m) | 0.310±0.030 | 0.376±0.084 | 3.12 |
| 复位沉降漂移 (m) | 0.100±0.053 | 0.179±0.123 | 2.58 |
| 每局墙钟 (s) | 28.4±6.2 | 21.7±5.0 | −3.7 |

approach 退出方式：release `visible brake depth` 17 / `stalled` 2；v1_1 `visible brake depth` 9 / **`stalled` 10**。
`plan_r`=0.32 m，release 超出 5.7 cm，v1_1 超出 **18 cm**。

失败构成：release = 8 stand-infeasible + 1 转向超时 + 1 IK + 1 碰倒；
v1_1 = 14 stand-infeasible + **5 arrival-out-of-reach** + 1 walked-off-spawn（后两种 release 一次都没有）。

**结论**：v1.1 在参考回放上腰跟踪远好于 release（§16），但在这条采集线上不可用——步长与刹车时序按 release 标定，
机器人根本走不到位。换 checkpoint 必须连带重标 approach 时序（`visible_brake_depth`、mid-band、shuffle band）。

**与仓库现有说法矛盾**：`start_stack.sh` 注释称 v1.1 几乎消除 IDLE 漂移（0.004 vs 0.03–0.12 m），
但当前采集路径下 v1.1 的沉降漂移更大（0.179 vs 0.100）。那条是 2026-08-21 在旧 reanchor 路径上测的，已过时。

脚本：`scripts/experiments/v11_bottle_ab20_20260910.sh`（跑）、`v11_ab20_report_20260910.py`（统计）、
`v11_pilot_video_pair_20260910.py`（同 case 并排视频，按 pose_sidecar 的 wall 时间戳配对，不按顺序猜）。

### 17.3 这次踩的坑与随后的修复

三次配置失败各烧掉一个 stack 周期，形状相同：**配置只生效一部分，但程序照跑**。
1. `INSTANCE=1` → exporter 被跳过（`instance N records no episodes`）且 gym 不跟随 `INSTANCE`，10 s 后 `no privileged state on :5581`。
2. 漏 `--reach-config` → `ready_grasp_recovery=False`，走到瓶子跟前才在 grasp 阶段被拒。
3. `--task-config` 不管 spawn → 吃 ±90° 宽默认，10/10 卡 `[view]`。

已落地的修复（`make check` 通过；size lint 无新增超限）：
- `locomani.sim.ports`：端口只有一份公式，按 `INSTANCE` 推导，导出的 `TF_*` 仍优先。收编了 `agent_server` /
  `privileged_pub` / `transition_capture` / `run_instance_exporter` / `run_sim_locomani` 的硬编码回退，以及
  crate 两个任务里自带的第三份拷贝。
- `reach_config.require_runnable()`：`ready_grasp_recovery=False` 在**启动时**拒绝并指出用哪个 profile。
- `ports.require_exporter()`：`--record` 没有活着的 exporter 就拒绝，并说明只有 instance 0 会起 exporter。
- `--task-config` 现在携带整份 profile（reach + spawn + target + stand_retries），命令行显式给的优先；
  验证：单独 `--task-config configs/tasks/bottle_visible.json` 生成的 case bank 与手工展开 9 个 flag 的结果逐字节相同。
- `scripts/collect_pilot.sh` 补上 `--task-config`。**它从重构 step 2（`0b801d8`）起一直是坏的**：不传 reach 标定，
  每局都会在 grasp 阶段被拒，照文档走的人会拿到 0 成功。

### 17.4 修复的回归（2026-09-10 14:4x，session 65 在实跑中发现）

`ports.require_exporter()` 第一版只匹配 cmdline 里的 `run_data_exporter`，但
`run_learn_pilot.py:176` / `run_ready_pilot.py:120` 把 `TF_EXPORTER_SCRIPT` 指向
`pipeline/run_instance_exporter.py`（`start_stack.sh` 的默认值才是 `run_data_exporter.py`）。
于是 exporter 明明活着，整条 pipeline 的 collect/eval 都会被我的检查拒绝——我自己的 runner 走的是
start_stack 默认值，所以没暴露。已改成显式的 `_EXPORTER_ENTRY_POINTS` 元组并补 `tests/test_instance_ports.py`。

教训：一个"拒绝运行"的检查必须自带测试。这条检查本来是为了消灭静默失败，结果自己制造了一个响亮的假阳性。

### 17.5 第二处同类回归（同日，自查发现）

`require_runnable` 会拦下**所有没显式给 `--reach-config` 的闭环 eval**。原因：VLA 轮次在 `_p_handover` 里
直接 `return ctx.vla_policy(...)`，根本走不到 `_p_approach` / `_p_grasp`，`execute_grasp` 永远不执行，
所以 reach 标定与它无关；而 `eval_closed_loop.py` 只在被要求时才传 `--reach-config`。
改为 `require_expert_calibration(VLA_POLICY, config, source)`，两个分支都有测试。

两处回归同一个形状：**我为了消灭静默失败加的拒绝，自己变成了假阳性**，而且都出在"哪条路径真的需要这个前提"
没有想全。第一处是别人在实跑中撞到的，第二处是我顺着同一形状自查出来的。
新加的每一条拒绝都必须回答：哪些合法调用路径会经过它？——并为"不该触发"的那一侧也写测试。

### 17.6 第三次（session cf 复审 diff 时提出）

cf 担心 `--task-config` 会把采集标定悄悄塞进闭环 eval，使其与台账里所有既有数字不可比。
**前提不成立**：今天没有任何 eval 路径传 `--task-config`（`run_learn_pilot` 的 eval 段传显式 `--reach-config`；
`eval_closed_loop.py` 没有这个参数；`run_eval_arm.sh` 两个都不传，仍走 `DEFAULT_REACH`）。

**但技术观察成立，且推翻了我的一句注释**：我写过"VLA 轮次与 reach 标定无关"，这只对 `execute_grasp` 成立。
`ready_grasp_policy.sample_stand` 读 `max_drop` / `backshift` / `ready_grasp_recovery`（第 90 行还有一条
`if not choices and config.ready_grasp_recovery` 的采样分支），所以标定会**移动采样出来的站位、进而改变 case 分布**；
`round_phases` 里还有一条 `ready_grasp_baseline` 分支专门在 `vla_policy is not None` 时执行。

处理：`_apply_task_config` 在 `VLA_POLICY` 非空时**不填 `reach`** 并打印 `skipped reach`，让"eval 保持自己的标定"
成为构造上的保证而不是运气；eval 若真要用采集标定，必须显式给 `--reach-config`（在 argv 里可见）并在台账登记。

三次假阳性里这次最危险：前两次是响亮的拒绝，这次会是**静默的分布改变**。

### 17.7 flow200 的 50/100 不能再当作外部对标(2026-09-10, session cf 提出并两次自我更正)

`ReachConfig` 与 `--reach-config` 由 `8c621fb`(2026-09-07)引入;`8c621fb^` 的 grasp_gym 里 `reach-config` 零命中。
flow200 的 eval 在 2026-09-02/03,**早于该参数存在**,所以它必然跑在当时的站位采样带上——
也就是 `8c621fb` 随后固化为 `DEFAULT_REACH` 的 post_min 0.22 / azimuth 36.5;
ready-grasp 线(七个 `ready_grasp_*.json` 加两个 task config)自 09-07 起统一为 0.30 / 42.0。
`sample_stand` 读这些字段,所以两者是不同的站位与 case 分布。

划界:
- **实验内部有效**:dart-vs-flow 的 2×2 四格同一实验同一协议,内部比较全部成立。
- **跨实验无效**:把 flow200 的 50/100 拎出来当作 09-07 之后 arm 的参考,是拿两个分布比。
  `docs/reference/runbook.md` 和 `docs/evaluation.md` 里"当前正式基线"的说法已加限定。

过程中被否掉的两个中间结论(都记下来,免得再走一遍):
1. "`run_learn_pilot` 的 eval 段传 `--task-config`,会被本次 diff 改掉分布"——不成立,那是 collect 段。
2. "flow200 的 argv 随 outputs 删除丢失,分布不可判定"——不成立,**查特性引入日期比考古 argv 更有力**。

一般性的坑仍然成立(只是不适用于 flow200):`run_eval_arm.sh` 的 `EXTRA=${7:-}` 会原样拼进 argv,
所以"哪个脚本产出"不是充分溯源,要看记录下来的 argv。

## 18. visible200 训练与闭环:离线优秀,闭环 2/100(2026-09-10, session cf)

第 13 节那份数据(34 批 335 局 → 200 集,check_dataset 32/32)训了 10k 步再跑 100 轮闭环。
**这是 visible_expert 协议第一次被训练**,不是 07d 的复现。

| 阶段 | 数字 |
|---|---|
| 训练 | 10000 步,4 h 08 m,loss 1.2866 → 0.0625,checkpoint-10000 12 GB |
| 闭环(seed 6000、async、replan 0.4、budget 90、无 ensemble、出生 1.1–1.3) | **2/100** |
| 失败构成 | `[vla]` 97、`[stand]` 1;97 局里 96 局是"到了桌边但从不下合爪指令" |
| 开环(自己的 val 10 集,horizon 40) | nmse_token 0.2593(persistence 0.9066)、nmse_gripper **0.0024**、close_frame_err_abs **2.3 帧**(persistence 18.9)、deg_mean 1.324° 对噪声底 1.209° |

**离线与闭环彻底背离。**模型在训练分布的观测下知道何时合爪(合爪时序误差 1.3–2.3 帧,夹爪 nmse 0.0024,
关节误差压在测量噪声底上),闭环里却几乎从不合爪。这不是没学会,是**决策时刻的观测落在训练分布之外**。

**机制,有数字支撑**:96 局"从不合爪"的最小掌距分布是 p10 0.18 / 中位 **0.27** / p90 0.35 m,
而专家的刹车带是 **0.30–0.31 m**(第 11 节;`visible_brake_depth=0.31`)。
只有 **20/96** 停进了 0.29–0.32,**绝大多数停得比专家近**。
visible_expert 协议按"隐变量原则"**删掉了 reposition / recovery 分支**(07d 用它救回 8% 的到位),
代价是 19%(本次 72/101)的到位被判 stand infeasible 丢弃 —— 也就是说,
**数据集里根本没有"我停在 0.22 m,现在该怎么办"的示范**。策略的停位由模仿接近速度曲线决定
(memory `policy-stop-point-is-set-by-the-crawl`:专家停位移 4 cm,策略停位只移 7 mm),
一旦停短就进入一个没有任何示范覆盖的状态。

**与 cover400 的区别要说清**:cover400 的病灶是标签里的隐变量,`close_frame_err_abs` 因此退化到 10.25 帧;
这里 close_frame_err_abs 是好的(1.3–2.3),所以**不是同一个病**。这次是纯粹的 covariate shift。

**裁定**:
1. 数字与 07d-200 的 57/100 同带可比(§17.7 的口径),所以 **visible_expert 协议在这个训练配方下不可用**。
2. 不能归因于 seed 方差:方差底线是 57 vs 46(memory `training-seed-variance-floor`),2/100 远在其外。
3. 也不能归因于 07:45–08:30 的 GPU0 争用:那 8 批的 yield 是 0.54 vs 0.62,而审计 32/32 全过,
   量级上解释不了 55 个百分点。
4. **下一步不是再训一个 seed**,是补回策略停短时的示范:要么恢复 recovery / reposition 分支
   (代价是重新引入被隐变量原则删掉的东西,需要用可观测量重写它的触发条件),
   要么用 DAgger 在策略自己的停位分布上补标注 —— 后者正是 ROADMAP 里排在最前面的杠杆。

产物:`outputs/experiments/visible200_20260910/`(model/checkpoint-10000、eval/closed_loop.manifest.jsonl、
open_loop/、eval_provenance/ 起止两次树快照)。评测本身在 16:10 第一次启动时因
`eval_closed_loop.py` 的 src 引导缺一层目录而立刻失败(修复 23b5436),16:14 重跑成功。

### 18.1 更正:§18 给的机制是错的(2026-09-10 深夜,用户质疑后逐局复查)

用户不信 2/100,要求复查。**数字是真的、条件可确立**(见下"已排除"),
但 §18 那段"策略停得比专家近、落在没有示范的状态"的机制**被逐局 trace 否掉了**。

逐局量 99 份 `eval/traces/round_*.json`,取最近接近时刻:

| | 失败 97 局 | 成功 2 局 |
|---|---|---|
| 物体在骨盆系的方位角 | p10 −45° / 中位 **−32°** / p90 −11° | **−39°** |
| 最近掌距 | p10 0.18 / 中位 **0.27** / p90 0.35 m | **0.25 m** |
| 整局最小夹爪指令 | **0.02**(张开 0.0245) | **−0.02**(闭合 −0.023) |

**方位角和距离完全不分成败** —— 成功局就落在失败局的分布里。
分开二者的只有一件事:**有没有下闭合指令**。而且是二元的,没有中间态。

再用 `latency.chunk_grip_min`(模型预测的 chunk 内夹爪最小值)拆"没预测"还是"没执行":

- **94/99 局模型根本没预测过闭合**(chunk 内最小值始终 ≥ 0);
- 2/99 预测了且执行了 → 两局都成功;
- 3/99 预测了闭合但执行流里没出现(round 027 −0.0175 → 执行 0.0048;052、065 同类)。

所以**不是执行链路把闭合吃掉了**,是模型在闭环里几乎从不决定闭合 ——
而同一个 checkpoint 在自己的 val 上开环预测夹爪 nmse 0.0024、合爪时序误差 2.3 帧。
**离线知道何时闭合,闭环不闭合,且与它站在哪、朝向哪无关。**这条比 §18 原来的说法更窄也更准:
决策时刻的观测里有某种东西与训练分布不同,而它不是位置也不是方位角。未定位。

**已排除(逐项查过,不是猜)**:checkpoint 路径;评测协议(async/replan 0.4/seed 6000/budget 90/无 ensemble);
`dataset_fps` 25 与数据一致;initial token 已传;评测用的就是采集那份冻结场景;出生分布与采集相同;
`SIM_VIS` 与训练图像无关(`reach`/`ghost` 在 geom group 4,ego 相机与 exporter 看不到);
评测期间工作树未变动(`eval_provenance/verdict`)。

**未能排除**:采集(`restart_deploy=false`,每局 `hold+reanchor`)与评测(钉死每局重启 deploy,
"锚 = 落地 yaw")建立朝向锚的机制不同,65 量过两者差约 8.5°。为此把 `restart_deploy` 做成
`EvalConfig` 字段(默认不变)并跑同 seed 前 20 局的配对对照,
**结果 0/17(跑到 17 局手动停掉,已无信息),全部 `[stand]`:前 14 局「复位后没站稳(pelvis z 0.094)」,之后 3 局变成「teleport 落点偏 0.81–0.92 m」—— deploy 跨局残留在累积,越跑越坏** —— 不重启时评测路径的机器人直接瘫倒,
这条臂测不了它想测的东西,只证明评测确实需要那次重启。锚不一致**仍是未排除的嫌疑**。
产物 `outputs/experiments/visible200_norestart_ab_20260910/`。

**下一步**(取代 §18 的建议):先定位"闭环里缺了什么观测",再谈补数据。
可做的最小实验:把某一局失败时刻的观测原样喂给模型离线推理(`obs_dump` 已有开关),
看它在那个观测下是否预测闭合;若预测闭合则问题在观测的构造/时序,若不预测则用该观测与训练集最近邻比对找差异。

### 18.2 §18.1 的 join 是错位的:round 号全部 +1,三条结论里有一条要改(2026-09-11, session cf)

§18.1 是把 `eval/traces/round_NNN.json` 和 `closed_loop.manifest.jsonl` **按 round 号对起来**算的。
这个 join 是错的:**trace 文件名来自 VLA client 自己的调用计数器,不是 manifest 的 round 号**。
manifest round 0 死在 `[stand]`,没进 VLA 阶段,计数器根本没起步,于是从它之后每个 trace 文件名都比它
描述的那一局小 1。

证据(`scripts/experiments/visible200_trace_join_20260911.py`):按 offset=+1 对齐,99/99 的 `why` 字符串
**逐字精确匹配**、99/99 的 `ok` 匹配;按 offset=0 只有 6/99。manifest 的成功局是 14、62,trace 文件是
013、061。

**修复**(2026-09-11):`round_id` 从 `round_loop.py`(写 manifest 的那个 `rnd`)一路传到三个 client
(`vla_client` / `vla_round` / `lockstep_client`),trace 按它命名,并把 `round` 也写进 json 里,
join 从此不必相信文件名。回归测试 `tests/test_trace_round_join.py`(在修复前的 HEAD 上 12 条断言全红)。

用正确 join 重算 §18.1 的四条结论:

| §18.1 的说法 | 重算后 |
|---|---|
| 94/99 局模型从没预测过闭合 | **成立**,94/99,且这 94 局 **0 局成功** |
| 2/99 预测了且执行了,两局都成功 | **成立**(round 014、062) |
| 3/99 预测了但没进执行流(round 027/052/065) | **要改,见下** |
| 方位角和距离不分成败 | **成立**(成功 −39.3°/0.246–0.290 m,落在失败的 p10–p90 内) |

**要改的那条**。局号是 028/053/066(不是 027/052/065),而且三局不是同一回事:

- **round 028**:chunk 最小 −0.0175,**执行流里到了 +0.0048**,刚好越过 0.005 的闭合判定 →
  它**确实执行了**一次很浅的闭合,manifest 写的是「commanded a grasp but no lift(物体横move 12 cm)」。
  所以这不是执行链路吃掉了闭合,是闭合太浅、把瓶子碰飞了。
- **round 053 / 066**:chunk 最小 −0.0024 / −0.0155,执行流最小 +0.0189 / +0.0142 —— 这两局才是
  **真的预测了但没进执行流**。066 是在第 28/227 次推理就预测了闭合,执行流整局没有低于 +0.0142。

所以「3 局预测了没执行」应为「**2 局预测了没执行,1 局执行了但太浅**」。这 2 局是 async client
把含闭合的那段 chunk 丢掉了,值得单独看一眼,但它解释不了 94 局。

**正确 join 之后新看见的事实(§18.1 没写)**:夹爪通道不是"糊",是**贴着轨道端点饱和**。

| | n | chunk 内夹爪最小值 |
|---|---|---|
| 从不闭合的局 | 94 | p10 +0.022 / 中位 +0.023 / max +0.023 |
| 闭合过的局 | 5 | −0.020 … −0.002 |

参照 `sonic_token.py:75` 的 `GRIPPER_MIN/MAX = −0.0200 / +0.0245`,闭合判定 `< 0.005`。
94 局里整整一局几百个 decision step 全部压在 **+0.0245 的张开端**,5 局闭合时压在 **−0.020 的闭合端**。
**这是一个二元决策从不翻转,不是一个连续量落偏。**这把"训练不足 / 精度不够"整类解释排掉了。

产物:`outputs/experiments/visible200_20260910/eval/trace_join_20260911.json`(逐局记录)。

### 18.3 定位了:模型的闭合决策是对的,手根本没够到瓶子(2026-09-11, session cf,诊断第 1 步)

**方法**:开 `OBS_DUMP` 重跑 seed 6000 的前 5 局(`scripts/experiments/visible200_obsdump_20260911.sh`,
协议逐字抄自 `run_learn_pilot.py` 的 eval stage),拿到 **912 份 npz**(每次 policy query 一份,含完整
observation + 当时的回答),再用 `src/locomani/pipeline/replay_obs.py` 离线重放和逐 key 替换。
复现忠实:r0 仍是 `[stand] pelvis z 0.094`,r1–r4 全是「at the table but never commanded a close」。

**先修了一个静默失败**:`--set backend.obs_dump=DIR` **过去完全无效**。`SonicBackendConfig.obs_dump` 的
唯一消费者是 `stack.py`,它把 `TF_OBS_DUMP` 导进 **stack 进程**的环境;而真正落盘的是
`vla_obs.PolicyLink.get_chunk` 里读的 `vla_round.OBS_DUMP` —— eval client **进程内的模块全局变量**,
从来没人赋值。配置被接受、被写进 config snapshot、一个文件都不写。同批发现 `grip_delay`、`replace`、
`lockstep.no_seed` 三个 knob 同病(它们的 config 默认值恰好等于模块默认值,所以**没有历史结果受影响**)。
四个都已在 `eval_closed_loop.py` 里接上;`tests/test_eval_config_reaches_the_client.py` 把规则钉死
(修复前 HEAD 上会点名这四个)。注意:「字段名有没有被引用」这种检查**抓不到**它——`obs_dump` 是被引用了,
只是在另一个进程里。

**三条独立证据指向同一个结论**:

1. **逐 key 替换**。以「机器人已在桌前」的 live dump 为底,把 dataset 的 key 一个一个换进去(donor 是
   val ep0 f273,一个模型 9/9 会闭合的帧):

   | 替换的 key | grip_min | 闭合? |
   |---|---|---|
   | 不替换(原始 live) | +0.0242 | no |
   | 10 个 `state.*` 全部,逐个 | +0.0237 … +0.0243 | **全部 no** |
   | `annotation.human.task_description` | +0.0238 | no |
   | `video.ego_view` | +0.0142 | no |
   | **`video.right_wrist`** | **−0.0129** | **YES** |

   四个不同 round 的底样本(obs_00130/00249/00468/00700)结论一致:只有换腕部相机会翻转,
   `−0.0069 … −0.0143`;换 ego 永远翻不过来。**闭合决策几乎完全由右腕相机驱动。**

2. **看图就明白了**(`outputs/experiments/visible200_obsdump_20260911/wrist_live_vs_dataset.png`):
   训练帧里瓶子**正正夹在两片夹爪中间**、竖直占满画面;live 帧里夹爪悬在**空桌面**上方,
   瓶子掉在左下角、在夹爪**外面且更低**。模型没闭合,是因为**爪子里没有东西**。

3. **掌心距离完美分离**(用 §18.2 的正确 join,重算原来那 99 局):

   | | n | 整局最小 palm-to-bottle |
   |---|---|---|
   | 成功 | 2 | p10 0.137 / 中位 **0.138** / max 0.138 m |
   | 失败 | 97 | min **0.156** / p10 0.201 / 中位 0.220 / p90 0.256 m |

   **97 局失败里,0 局曾经够到成功局的距离。**完全不重叠。而同一批数据的
   **base-to-bottle 距离完全重叠**(成功 0.246–0.290,失败 p10 0.183 / p90 0.350)。

**结论,并更正 §18/§18.1 的走向**:这**不是**观测编码 bug(没有 layout / 色彩 / 编解码问题),
**不是**模型拒绝闭合,**不是**执行链路吃掉闭合(912 次 live query 的回答最小值 +0.0215,
从没离开过 +0.0245 的张开端;离线重放同样不闭合,失败可离线复现)。
**模型的夹爪决策是正确的——它看着空爪子,正确地不闭合。**
真正的失败在上游:**策略的手臂/手掌停在离瓶子约 6–8 cm 远的地方,从来没伸到**。

§18.1 说「不是停位问题」是因为它量的是**底盘**停位。停位问题是有的,只是在**手掌**,不在底盘。
这也解释了为什么方位角和底盘距离都不分成败。

**顺带发现的真 bug(但不是本案主因)**:`state.right_hand` / `state.left_hand` 的 7 维向量,
**live 把两个 jaw 分量放在 index 0,1,而训练数据放在 index 4,5**(数值相同,槽位不同;
整段 run 的 per-index std:live 只有 0,1 非零,dataset 只有 4,5 非零)。
所以评测时模型在它学过的那两个槽位上读到的恒为 0。**但替换实验证明它不是主因**——
把 live 的 `state.right_hand` 换进一个会闭合的 dataset 观测,仍然 5/5 闭合。需要修,别归因。

**下一步**:问题变成「为什么手臂伸不到」。先看 §18.2 的 5 局视频和 palm 轨迹,
再对比专家的 palm 轨迹——失败局的 palm 是停在半路,还是伸向了错误的位置。

### 18.4 机制完整了:策略走到正确的预抓取站姿,然后**冻在那里 85 秒**(2026-09-11, session cf)

§18.3 说「手掌停在离瓶子 6–8 cm 外」。这个说法**会误导**:`palm_obj` 是三维距离,而这个差距
**几乎全是竖直方向的**。用 §18.2 的正确 join 把它拆开(原始 99 局,最近接近时刻):

| | n | palm→bottle | **竖直** palm_z−obj_z | 水平 |
|---|---|---|---|---|
| 成功 | 2 | 0.138 | **+0.029** | 0.134 |
| 失败 | 97 | 0.220 | **+0.191** | **0.108** |

**失败局的水平距离比成功局还近**(0.108 vs 0.134)。全部差距在高度:手掌停在瓶子上方 **19 cm**,
而不是 3 cm。

- `palm_z` 最小值:成功 **0.729 m**,失败 **0.875 m**,起始(rest)0.905 m。
- **94/97 局手掌从没低于 0.85 m** —— 需要下降 18 cm,实际只动了 3 cm。**手臂基本没离开 rest。**
- 时间:失败局的最近接近发生在 **5.6 s / 90 s**,之后 85 秒什么都没发生。成功局是 12.1 s / 17.6 s。

**策略的站姿是对的,只是不往下一步走。** 拿 live dump 的 `state.right_arm`(228 个 at-table 样本)
和专家的逐阶段中位数比:

| | sh_pitch | **wr_roll** | wrist x | wrist z |
|---|---|---|---|---|
| 专家 close−150/−120/−90(接近相) | 0.27–0.29 | **0.03–0.07** | 0.145 | 0.110 |
| 专家 close−60(**起势**) | 0.570 | **0.256** | 0.087 | 0.148 |
| 专家 close−20/−0(下探) | 0.24 | **0.23–0.26** | 0.150 | 0.086 |
| **LIVE @table** | **0.313** | **0.018** | **0.140** | **0.112** |

live 与专家的**接近相几乎完全重合**(sh_pitch 差 ~2°,wrist x 差 5 mm,z 差 2 mm),
但从没做过 close−60 的**起势**(肩抬 + 腕滚到 0.24)。
**腕滚 `wr_roll` 是最干净的判别量:专家 0.03 → 0.24 并保持到抓取结束,live 全程 0.018。**
而且是**冻住**的:85 秒里 sh_pitch std 0.012 rad(0.7°)、wrist x std 2.3 mm、wrist z std 2.1 mm。

**预测还是执行?是预测。**把 live 预测的 token 拿去和训练 token 库做最近邻(40 集 / 17440 帧,
`observation.eef_state[9]` = 腕在骨盆系的高度):

| 输入 | token[now] 最近邻腕高 | token[+1.6s] | 那一帧在闭合? |
|---|---|---|---|
| live(152 个样本) | 0.111(= live 实测 0.112) | **0.130(在往上)** | 0/152 |
| live,但腕部图换成专家 close−60 前的帧 | 0.111 | **0.104–0.110(在往下)** | 否(那时专家也还没闭合) |
| live,但腕部图换成专家抓取瞬间的帧 | 0.113–0.119 | 0.150–0.179(提起) | 是 |

**右腕相机就是下探动作的触发器。**给它一张「瓶子在爪子前方」的腕部图,它就命令手臂下去;
给它 live 那张(空桌面、瓶子在左下角且在爪外),它命令手臂**略微上抬**。机器人忠实执行了「保持不动」。

**触发器本身是学会了的,不是没学到。**把专家自己的观测喂回去:

| 输入(val ep0) | 模型的 token 最近邻 now→ahead | 判定 |
|---|---|---|
| close−150 / −120 / −90 | 0.109→0.110 / 0.107→0.120 / 0.112→0.114 | 保持(正确,专家还在接近) |
| **close−60** | **0.148→0.075** | **下探(正确)** |
| close−40 / −20 | grip +0.0077 / −0.0061 | 开始闭合(正确) |

逐 chunk 和 ground truth 比也吻合(close−90/−75/−60 三处,模型的 1.6 s 腕高曲线跟真值同向同幅)。
**所以模型是个好的开环预测器,触发器也学到了——它只是在闭环里永远到不了会触发它的那个状态。**

**结论(闭环吸收态)**:策略走到桌前,进入与专家**逐关节吻合**的接近姿态,然后保持。
离开这个姿态的起势在专家数据里是**脚本决定的**,不是观测触发的;而模型能触发下探的那个观测
(瓶子出现在腕部相机的爪前方)**只有做完起势才会出现**。要下去才能看见,要看见才会下去 ——
于是 85 秒原地不动,`[vla] reach: at the table but never commanded a close`,97/99 局。
与 memory `creep-hidden-stop-target` 记的是同一类:**相同的帧、相反的标签 → 闭环吸收态**。

这也解释了为什么开环指标看不出来:接近相是个长平台,「保持」在上面绝大多数帧都是对的,
起势只占几帧,在全帧平均里被抹平。**开环 nmse 优秀和闭环 2/100 并不矛盾。**

**下一步不是找 bug 了,是数据/表示问题。**三个方向,按代价排序:
(a) 给策略一个相位/进度信号(它现在必须从单帧图像里推断「我该起势了吗」,而那一帧和前一秒的帧几乎一样);
(b) 过采样起势那几帧(转换器的 quasi-static 裁剪保留「动作前的一拍」,但起势帧在全集里占比极低);
(c) DAgger:在「已到桌前、接近姿态、静止」这个状态上补专家标注 —— 这正是策略卡住的那个状态,
    而它在专家数据里是**瞬态**、从不停留。

### 18.5 更正 §18.4:「右腕相机是下探的触发器」这条**不成立**,是我读了噪声(2026-09-11, session cf,用户要求做对照后)

用户要求补两个对照:(a) 换一张「手臂在接近姿态、瓶子不在爪前」的专家腕部图会不会也触发;
(b) 把逐 key 扫描改成量 token(下探)而不是夹爪。两个都做了,**两个都是阴性**,并且暴露出
§18.4 的方法问题。

**方法缺陷**:§18.4 的下探判定是**单次采样**的最近邻腕高差。GR00T 的 action head 是 diffusion,
同一个 observation 重复问 10 次的离散度:

| observation | ahead−now 均值 | **sd** |
|---|---|---|
| live obs_00130 | +0.0207 | 0.0026 |
| live + 腕部图←f226 | +0.0019 | 0.0078 |
| live + ego←f226 | −0.0332 | 0.0167 |
| **专家自己在 close−60 的观测** | **−0.0718** | **0.0020** |

**§18.4 据以下结论的差值是 0.005–0.007,而噪声 sd 是 0.002–0.017 —— 完全在噪声里。**
换个 seed 重跑,同样的 donor 时而"下探"时而"上抬"。

**重做(8 次重复 + 3σ 判据,`scripts/experiments/visible200_trigger_controls_20260911.py`)**:

**(a) 没有任何一张腕部图能显著触发下探**,在两个 base 上都一样:

| donor 腕部图 | obs_00130 d(wrist_z) | obs_00468 |
|---|---|---|
| live(不替换) | +0.0222 ±0.0024 | +0.0123 ±0.0044 |
| f121/f151/f181 接近姿态,**画面里没有瓶子** | +0.0208…+0.0213 | +0.0084…+0.0128 |
| f211 起势,**瓶子在爪前** | +0.0150 ±0.0075 | +0.0088 ±0.0042 |
| f211 但**把瓶子抠掉**(其余逐像素不变) | +0.0197 ±0.0033 | +0.0120 ±0.0067 |
| f226 下探中,瓶子占满画面 | +0.0025 ±0.0081 | −0.0012 ±0.0039 |

全部「无显著位移」。f226 能把上抬**抵消掉**,但抵消不等于下探,而且过不了 3σ。
**抠掉瓶子的对照也因此无信息量**——原图本来就没触发,抠不抠都一样。

**(b) 逐 key 量下探:只有 `video.ego_view`,而且只在 2 个 base 里的 1 个上显著**
(obs_00130 −0.0342 ±0.0143 显著;obs_00468 −0.0077 ±0.0060 不显著)。
**`video.right_wrist` 在两个 base 上都不显著。**——注意这和 §18.4 的结论**方向相反**,
更说明那是噪声归因。

**所以要撤回的**:§18.4 里「右腕相机就是下探动作的触发器」「给它一张瓶子在爪前的图它就命令手臂下去」
这两句,以及由此推出的「要下去才能看见,要看见才会下去」这个**具体**的因果闭环。

**仍然成立的**(都是大余量的直接测量,不是噪声差):
- **夹爪归因于腕部相机**:余量极大(grip 的 sd 只有 0.0002–0.0005;用抓取瞬间的 donor,
  grip 从 +0.024 掉到 −0.012…−0.015,约 30–70σ,4/4 个 base 一致)。而用 f226(专家此刻还没合爪)
  当 donor 时模型**也不合爪**(+0.0216–0.0226)——**跟着腕部图走,而且走得对。**
- §18.4 的全部描述性测量:竖直差 19 cm、手臂冻住(85 s 内 0.7°/2 mm)、腕滚 0.018 vs 0.24、
  palm 距离完美分离、912 次 query 0 次闭合、live token 最近邻落在接近姿态。
- **模型在专家自己的完整观测上确实会下探**:close−60 时 −0.0718 ±0.0020,对比 live +0.0207 ±0.0026,
  约 25σ。

**修正后的、更强也更诚实的结论**:

> 模型在**专家的完整观测**上会下探(25σ),但把那个观测里的**任何单个 key** 移植进卡死的 live 观测,
> **都复现不出来**(最好的单 key 约 −0.034,且换个 base 就不显著)。
> 所以 live 的卡死状态与专家的转折状态之间的差异是**分布式的**,不是某一个输入。

这**不削弱**吸收态的结论(策略停在接近姿态、转折永不触发,§18.4 的描述性证据全部成立),
但它**删掉**了我给出的那个机制故事。「什么触发转折」目前**未定位**。

**下一步的方法教训**:这条线上任何基于 token 最近邻的差值结论,都必须重复采样并给 sd;
单次采样的 0.005 级差值没有信息量。夹爪不受此限(sd 小两个数量级)。

产物:`outputs/experiments/visible200_obsdump_20260911/{trigger_controls.json,
wrist_series_live_vs_expert.png,ablation_bottle_removed.png}`。

### 18.6 腕部相机在桌前确实看不见瓶子——但**专家也看不见**,所以这是结果不是原因(2026-09-11, 用户提问后实测)

用户问:96 局是不是因为停到桌前之后右腕相机里根本没有瓶子?**观测事实成立,因果方向相反。**

瓶子的绿盖+绿标是场景里唯一的饱和绿,可以直接数像素(在已知无瓶的专家帧 f121/f181 上是 0 px,
已知有瓶的 f211/f226 上是 1405/3612 px,检测干净)。

**live 卡死的 912 帧,右腕相机:**

| 阈值 | 帧数 |
|---|---|
| > 0 px | 138 / 912(15%) |
| > 50 px | **40 / 912(4.4%)** |
| > 800 px | 1 / 912 |

**基本等于看不见。**(同一批帧里 **ego 相机 833/912 = 91% 能看见**,中位 72 px ——
机器人并不是"看不到瓶子",是**腕部相机**看不到。)

**但专家在同一阶段也看不见**(12 集,按阶段数像素):

| 阶段 | 腕部相机瓶子像素(中位) | >50px 的集数 |
|---|---|---|
| close−150 | 4 | 4/12 |
| close−120 | 0 | 1/12 |
| **close−90** | **0** | **0/12** |
| **close−60(起势)** | **1632** | **12/12** |
| close−45 | 4053 | 12/12 |
| close−20 | 7814 | 12/12 |

**瓶子是在起势之后才进入腕部视野的 —— 因为手臂动了,所以看得见;不是因为看得见,所以手臂动。**
专家在 close−90 时 0/12 集能看见瓶子,照样开始起势。

所以:**"腕部相机看不见瓶子"是接近阶段的正常状态,是手臂还在上面的结果,不是它卡住的原因。**
这也正好解释了 §18.5 的对照 (a) 为什么全阴性 —— 在那个阶段,正确行为本来就不以腕部视野为条件。

**留下的一个未解矛盾(需要复核 §18.3/§18.4 的「是预测不是执行」)**:
token 最近邻这个弱代理说 live 的预测在"上抬"(d(wrist_z) +0.021),而实测手臂 85 秒纹丝不动
(0.7°/2 mm)。两者对不上。**夹爪那一侧不受影响**——912 次 query 的 grip 直接读数从没低于 +0.0215,
是模型自己的输出,无代理。但**手臂那一侧的「预测失败」结论是建立在弱代理上的,尚未独立证实**。
要坐实需要真正解码 token(SonicDecoder + 部署侧 994 维 obs)或比对当时实际执行的 token。

### 18.7 矛盾解决:**没有执行问题,手臂确实跟着 token 走**——是预测(2026-09-11, 用户要求查证)

§18.6 记的矛盾:token 最近邻代理说 live 预测在"上抬"(+0.021),而实测手臂 85 秒纹丝不动。
用三个**不依赖那个代理**的测量查掉了。

**1. 先给代理做标定(此前从没做过)。**用 40 集建库、另外 10 集留出,拿留出集的 ground-truth token
去查库:

| | |
|---|---|
| 代理推出的腕高 vs 真实腕高 | **r = 0.978**,误差 sd **0.0082 m** |
| §18.4 用的那个 (ahead−now) 差值 | **r = 0.966**,误差 sd **0.0107 m** |

**代理本身是准的。**所以 §18.5 的撤回原因要说清楚:**不是代理坏了,是我拿单次采样去读
0.005 级的差**——而采样噪声 sd 0.002–0.017、代理自身误差 sd 0.011,两头都盖过信号。撤回成立,理由更新。

**2. 指令的变化量与手臂的变化量是成比例的。**一局内 ~228 次 query、~85 s:

| | token sd | 右臂关节 sd (rad) | 腕高 sd (m) |
|---|---|---|---|
| live round 1–4 | 0.018–0.024 | 0.011–0.014 | 0.0016–0.0022 |
| **专家接近窗口(40 集)** | **0.0403** | **0.0236** | **0.0073** |

live 的指令变化约是专家的**一半**,手臂变化也约是专家的**一半** —— **同比例**。
而且 live 是在 **85 秒**的窗口里,专家只有 **2.4 秒**:时间长 35 倍,运动量反而只有一半。

**3. 手臂逐拍跟随 token。**相邻 query 的 |Δtoken| 与 |Δarm| 的相关:

| round | 零延迟 r | 滞后 ~2 s 的 r |
|---|---|---|
| 1 | +0.721 | +0.522 |
| 2 | +0.618 | +0.562 |
| 3 | +0.539 | +0.389 |
| 4 | +0.598 | +0.485 |

**指令一动,手臂就动。执行链路是通的。**
整局的手臂净漂移只有 0.11 rad(约 6°)—— 在原地小幅游走,没有任何净进展。

**结论**:`[vla]` 那 96 局**没有执行问题**。手臂忠实执行了它收到的指令,而模型发出的指令
**始终是"在接近姿态附近小幅摆动"**,从不是起势。§18.3/§18.4 的「是预测不是执行」**成立**,
现在有了不依赖 token 代理的独立证据(比例一致 + 逐拍跟随),而不再只靠最近邻差值。

那个 +0.021 的"上抬"读数:约为代理自身误差的 2 倍,且与"整局净漂移 6°、无净进展"并存,
应读作该阶段的代理偏置,而不是一个真的被持续指令的 2 cm 抬升。

### 18.8 闸门诊断**没验过**——但它把我引到了真正的原因:**策略是严格单帧、且完全没有速度输入**(2026-09-11)

按计划写了"同观测、异动作"的闸门诊断(`src/locomani/pipeline/dataset_ambiguity.py`),
要求它先在现有数据上抓到已知的病。**它没抓到**,如实记录:

| 阶段 | n | 视觉距离 | chunk 歧义度 | 腕高 dz 歧义度 |
|---|---|---|---|---|
| far approach | 2029 | 0.056 | 0.504 | 0.175 |
| late approach | 669 | 0.070 | 0.429 | 0.393 |
| **WIND-UP** | 331 | 0.130 | **0.270** | 0.404 |
| descent | 469 | 0.135 | 0.311 | 0.678 |
| close + lift | 1797 | 0.145 | 0.432 | 0.532 |

起势那一栏的 chunk 歧义度是**全场最低**的。原因事后很清楚:起势时手臂**已经动了**,
而手臂就在腕部画面里——所以那两个阶段的图**本来就不一样**,不存在"同图异动"。
**这条诊断对本案无效,不能用它给新设计打分。**

**但顺着它做的下一步给出了真答案。**把 live 卡死的观测和训练帧放进同一个缩略图空间比距离:

| | 到最近训练帧的视觉距离 |
|---|---|
| 训练帧 → 其他 episode 的最近帧 | p50 **0.0721** |
| **live 卡死帧 → 训练帧** | p50 **0.0538** |

**卡死状态不但在分布内,还比训练帧彼此之间更近**(只有 3% 超过训练集的 p90)。
而且它们匹配到的训练帧**100% 落在接近相**(相对合爪时刻 p10 −372 / 中位 **−110** / p90 −101),
**0% 落在起势或之后**。

也就是说:**策略稳稳地停在一个"离抓取还有约 4 秒"的、完全在分布内的状态上,然后永远停在那里。**
同时实测底盘在平台期的速度是 **0.79 cm/s 中位**(97 局),基本是停住的。

**为什么同一张图在训练里通向抓取、在闭环里通向 85 秒静止?因为区别不在图里。**
查 `configs/gr00t_sonic_2cam.py`(visible200 真正用的那份):

```
video: delta_indices=[0]     <- 单帧,没有任何历史
state: delta_indices=[0]     <- 53 维,全是位置:关节角 + projected_gravity
                                + right_wrist_pos/quat。没有速度,没有 previous_action
```

**这个策略在原理上无法区分"正在走近"和"已经停下"** —— 单张 RGB 不含速度,state 里也没有。
训练里接近相的帧总是"还在往前走",执行接近动作会让场景变化;闭环里底盘已经停了,
执行同样的动作是个空操作,画面不变,于是同一个动作被无限重复。**这才是吸收态的成因。**

**同一个仓库里的姊妹配置早就解决了这件事。**`configs/gr00t_bfm_visual_2cam.py`(ScaleBFM 线):

```
video: delta_indices=[-10, 0]   <- 两帧,-0.4 s 和现在
state: ... 'robot_velocity', 'previous_action'
```

它的 docstring 写着"Video history is -0.4 s and now at 25 Hz. The previous executed action is
part of proprioceptive history"——有人早就想清楚了,只是 SONIC bottle 这条线没跟上。

**由此改变的行动建议**:在重设计专家(约一天)之前,**先只改一个配置再训一次**。
`video: delta_indices=[-10, 0]` **不需要重采、不需要重转换**——视频本来就在那儿,只是多读一帧。
代价 ≈ 77 min 训练 + 一次 100 局闭环。它直接检验"缺的是速度信息"这个假设。
(`robot_velocity` / `previous_action` 现有 parquet 里**没有**这两列,要用得改转换器;
但 `action.motion_token` 在 parquet 里,previous_action 可由 delta −1 取得,无需重采。)

**留在台账里的方法教训**:我写的那条闸门诊断没验过就差点拿去当闸门。
**任何新诊断都必须先在已知病例上见红**,见不到红就是不合格——这次见不到红,反而比见红更有用。

### 18.9 更正 §18.8:"策略分不清走近和停下"**不成立**——它有这个信息(2026-09-11,用户质疑后)

用户问"这次训练是为了证明什么",一问就问出了问题。§18.8 提议改 `delta_indices=[-10,0]` 再训一次,
理由是"单帧不含速度,策略分不清正在走近和已经停下"。**这个前提是错的。**

策略的 state 里**有 12 维腿关节角**。在训练数据上做一个**线性**探针,只用**单帧**腿部姿态:

| | |
|---|---|
| 预测平滑后的腿部运动量,held-out **R²** | **0.665** |
| 分类「运动(前 30%)vs 静止(后 30%)」准确率 | **0.900**(随机 0.5) |

**走路和站定在单帧里是能分开的**——迈步中的腿型和并拢的腿型不一样。一个线性模型就有 90%,
3B 的 VLA 不会更差。**所以信息是有的,"缺速度"的解释不成立,§18.8 提的那次训练失去了理由。**

(§18.8 里其余的测量仍然成立:卡死状态在分布内且比训练帧彼此更近、100% 匹配接近相、
底盘平台期 0.79 cm/s、`gr00t_sonic_2cam.py` 确实是单帧无历史。**错的是从中推出的因果。**)

**那么现在还剩下什么?**逐条排除之后:

- 观测编码 ✗(§18.3) · 模型不会闭合 ✗(§18.3) · 执行链路 ✗(§18.7)
- 停位/方位角 ✗(§18.2) · 分布外 ✗(§18.8) · 缺速度信息 ✗(本节)
- 模型在**专家自己的观测**上每个阶段都预测正确(§18.4)

**剩下的解释是:学出来的策略在一个完全在分布内的状态上有一个不动点。**
它有足够的信息、在熟悉的状态里、对专家数据预测正确,但它输出的动作让它留在原地。
这是 behaviour cloning 的经典失效,**不需要再找"哪个输入错了"——因为没有错的输入。**

**这恰恰是该上 DAgger 的理由**:DAgger 治的就是"策略有一个专家没有的不动点",
而且**不需要先知道为什么**。前面这一串排除的价值,正是把"其实是个 bug / 少喂了一个输入"
这一整类更便宜的解释都干掉了,从而**证明**了这里确实需要 DAgger,而不是又一次调输入。

**方法教训(第二条)**:§18.8 我在"信息不在观测里"这个论断上,没有先做那个**只要几分钟的线性探针**
就去提议一次训练。**"某信息不可得"这类论断必须先用探针证伪一次再采信。**

## 19. 单臂专家重新设计建议:视觉误差驱动、桌边静止也能继续(2026-09-11, Codex)

状态:用户询问如何设计;以下是代码与既有实验支持的**待验证建议**,没有实施或新实验结果。
遵循 STATE 中用户的 NO DAgger 裁定;§18.9 的 DAgger 建议不再是本线下一步。

### 19.1 已知事实与推断边界

- §18.3–18.7:失败主要是手臂停在接近姿态;最近接近时掌心比瓶中心高约 19 cm,成功约 3 cm。
  闭爪没有发生时,瓶子没有进入夹取区域。优先修伸手的启动与持续推进,保留离散夹爪意图。
- `round_phases_approach._phase_approach` 调用 coordinated.walk_tick;当前 CoupledGraspPolicy 继承
  RecoverableGraspPolicy.walk_tick,在未回撤的初始状态持续 `_push(state, ready_q())`。
  结束走近后才调用 execute → finish → coupled_transport。visible_expert 确实移除了部分随机站位/
  重定位,但没有改变这个跨阶段结构。
- coupled_transport 还用进入阶段时的 start_z/start_palm 和相对下蹲完成量插值手臂目标,并有 0.8 s
  交接条件。这些是要审查的历史依赖,尚未证明每一项都造成了当前失败。
- §18.8 的图像缩略图最近邻结果支持“视觉上接近训练接近段”,不能证明整个观测与动作动力学联合
  分布完全一致。§18.9 的探针说明腿姿包含运动信息,不能证明 VLA 已有效利用。两者都不支持宣布
  已穷尽所有原因。新设计能否改善闭环,仍须实验。

### 19.2 将“以视觉为线索”写成可检查的接口

决策输入仅包含当前学生相机可提取的任务特征、学生已有的本体状态及固定任务配置。特征包含目标
位置/尺度、桌缘、目标可见性与定位不确定度,近场增加瓶身相对两爪的偏差。关节状态/FK用于确定
手的位置和可达性;不要求机器人只靠 RGB 估计自己的关节。固定相机标定与目标几何先验可以使用。

正式视觉伺服应从实际 RGB 提取特征。仿真真值投影可用于前期控制器调试与审计,必须标成 oracle,
不能据此宣布“已经使用 RGB”。把 world object pose 换成 camera frame pose 也没有自动消除特权:
被遮挡的目标、几十像素图像无法分辨的毫米深度,不能直接成为动作分支的依据。真值可留在独立的
碰撞否决与验收通道;若否决频繁改变示范动作,须审计学生能否观测这个原因,不能静默录成长保持。

沿用规范的 `decide(obs_snapshot, task_config) -> command`。阶段名只作日志,不要用 elapsed、隐藏
随机目标或“之前已走过 approach”作为开始伸手的前提。确需抓持记忆、速度或滤波历史时,必须显式
建模并保证学生也拿得到;当前单帧配置不应悄悄混入只在专家侧存在的历史。超时用于失败退出,
不用于生成抓取动作。低层限速器的参考连续性与高层任务决策分开检查。

### 19.3 控制规律:持续减小可见的任务误差

推荐保留 SONIC 执行与动作表示,先改专家的任务目标生成。每 tick 根据当前观测重算可达、安全的
局部目标与纠偏方向,在相机提供的证据变化时重算,不沿进入阶段时冻结的世界坐标点盲走。

| 当前可见条件 | 动作依据与意图 | 必须验证的边界 |
|---|---|---|
| 远处 ego 看见瓶子和桌缘,腕相机尚看不清 | 底盘修正朝向/距离;手臂随当前目标方向做安全预备,改善后续可见性 | 不在走路时提前伸满或让长手指扫过瓶顶 |
| 桌边、手仍高于目标,ego 有效而 wrist 无效 | 从 ego + FK 计算安全的下探/侧向预对准;需要时协调小幅下蹲 | 起动不能等待腕相机先看见瓶子;无需经历过走路阶段 |
| wrist 能稳定定位瓶身与两爪 | 按横向/高度/姿态/插入深度误差修正;ego 保留桌缘与粗几何约束 | 不能只把瓶子居中:图像居中不等于深度与夹取截面正确 |
| 瓶身进入夹取区且相对运动允许闭合 | 发离散 CLOSE 意图,由执行器限速闭合 | 保留当前夹爪原则,不用提前闭爪掩盖够不到 |
| 已抓持 | 根据实测手和瓶的关系提起、验证离桌 | 闭爪命令本身不是已抓稳的证据;保持段属于终态验证 |

跨相机可在共同任务坐标中融合有效测量,或按置信度切换并验证切换连续性;不要直接平均两组方向
可能相反的控制命令。腕相机无有效目标时由 ego 提供起动依据。两者都不可用时只能做由可见几何
支持的安全重新取景,或失败退出,不能靠透视真值继续抓。固定的像素阈值需从实际分辨率/遮挡误差
标定,不能用当前绿色瓶盖的像素计数直接替代一般视觉定位。

“手臂姿态是距离的连续函数”仍然不够:若 `q_target=f(distance)` 在桌边某处有 `q=q_target`,
而底盘也停了,仍是非终态固定点。近场控制需要反馈“目标与实际手/爪之间还剩什么误差”。概念上
是 `手部局部速度 = 根据当前夹取误差求得的有界纠偏`,由 IK/碰撞与姿态约束转换为参考,交给既有
SONIC 通道执行;不是改 VLA 输出格式。远处可以保持安全手姿,进入可安全操作的近场后,不允许
“误差仍大、存在安全纠偏,却只因处于旧 approach 阶段而保持”。超出手臂可达范围时须有明确的
底盘推进;不能用手臂抖动假装进展。局部纠偏也可能有约束死锁,需要下面的状态库检验,无全局收敛保证。

不要把“腕高从第一帧严格单调”作为要求。Dex1 长手指曾在上方下压时扫倒目标;安全绕行可以暂时
增大掌心距离。应在安全通路的局部误差上衡量进度,同时记录净位移/旋转和最终夹取几何。

### 19.4 原生采集覆盖:主动打破固定进程与动作的相关性

先建立独立于 VLA rollout 的初始状态库:远处走近、桌边静止且手高、桌边部分伸手、夹取区附近的小
偏差。所有状态先完成物理落地与低层初始化,再开始正式记录,不能把瞬移、稳定过程或不合法接触
当示范。用同一个专家从每类状态继续,不能让“从中间开始”另走一套剧本。这遵守 NO DAgger。

0.5–1.5 m 的出生距离扩展是候选,但 spawn-edge 距离不等于 pelvis-to-bottle 或 palm-to-bottle;
只改该范围不保证覆盖现有桌边静止状态,必须用实测几何确认。第一轮建议 20 集诊断 pilot 覆盖上述
四类各约 5 集;这是覆盖分配,不是成功率统计或最终训练比例。200 集的比例等 pilot 结果再定。
保留完整走近轨迹,避免全换成桌边短片后失去行走能力。

随机化目标/桌面可见几何、初始手姿与安全的初始运动条件,固定站位方位约 42°。同一误差在不同
episode 时刻、不同手姿下都要出现,相似手姿也要对应不同可见目标位置。不要第一轮就放宽站位带:
07e 已说明“更多覆盖”可以降低收益。速度 jitter 也不能只是每集抽一个学生看不到的控制增益:
固定名义视觉误差→速度规律,或使变化通过学生已有观测可辨;真实速度条件若必须用历史判断,
先显式对齐观测与标签,不能复活“随便加历史必能解决”的旧结论。

### 19.5 验收顺序与否证条件

1. **CPU 先验检查。**审计所有影响动作的输入和阶段条件;构造相同允许观测、不同内部计时/阶段/
   episode 随机数的输入,高层意图必须一致。对目标位置做物理一致的受控改变时,视觉定位与纠偏
   应按几何响应;不要通过拼接互相矛盾的 RGB/FK/真值来宣称因果。
2. **先校准旧数据检测器。**区分“接近时手臂保持”“任务无进展”“等待执行器追上目标”。动作目标
   不变但实测手正在收敛,不算卡死;手臂持续抖动而夹取误差不降,仍算无进展。按窗口联合记录参考
   与实测残差、实际位置/姿态范围、目标相对手的误差、底盘进展、可见性和安全否决原因。优先使用
   exporter/pose sidecar 的真实物理量,不把 motion_token 最近邻差异当确定的位移。旧数据应检出
   已知接近平台;已有失败 trace 应检出桌边静止。还须有“恒定目标但实测在动”“终态保持”“安全
   绕行”的负对照。阈值从噪声和低层跟踪滞后标定,不能为了旧数据见红随意调。
3. **专家 20 集 pilot。**特别检查桌边静止开局是否能直接发起安全纠偏,以及起动前后的 ego/wrist
   图像是否已提供充分线索。在原本已可达的近场状态,暂时不给底盘平移也应能推进手部对准;
   未可达状态应清楚发出底盘需求,不要求手穿越约束。计时只做诊断,不作为专家的换阶段触发。
4. **收集约 200 集后训练、短闭环筛查。**固定训练预算与正式评测 case bank,单独报告近场静止
   诊断集,不要把容易的近场开局混进原 100 局分母。主要观察手部推进、到达夹取区比例、冻结率/
   时长、扫倒率与最终成功率。早期短闭环可提前否决,最终结论需要至少两个训练 seed;历史同一份
   数据两 seed 已出现 46%/57% 的跨度。若声称“只因专家变好”,新旧需匹配训练步数与推理协议;
   visible200 为 10k steps,不能与新 3k 单 seed 作纯专家因果归因。

第一轮不同时切换 SONIC checkpoint、动作空间、相机历史、ensemble 或成功标准。已知 hand-state
槽位错位应作为独立 bug 修复并保证比较两臂使用相同路径,不把它记为本方案的机制收益。新旧都要
核对 round join、真实输入布局、25 Hz 与执行频率、config 在实际进程中的生效和 stage provenance。

否证:如果新专家从桌边状态可以继续,数据的视觉与进展检查也通过,但两个训练 seed 仍普遍冻结,
就应拒绝“仅修改这套专家结构足以解决”的主张,再检查学习目标、状态依赖和动作表示。无平台期
不是 BC 无固定点的充分条件;旧数据有平台期也不能证明所有平台期都有害。不承诺未测成功率。

### 19.6 外部依据的适用范围

- [Chaumette & Hutchinson, Visual Servo Control, Part I (2006)](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf):
  用当前图像特征相对目标特征的误差设计反馈,区分图像特征与从图像估计的三维位姿。支持本节的
  控制设计原则,不提供 G1/SONIC 或训练后 VLA 的收敛保证。
- [de Haan et al., Causal Confusion in Imitation Learning (2019)](https://arxiv.org/abs/1905.11979):
  行为克隆可能利用错误相关性,信息更多也不保证更好。支持设计受控变化去检查动作依据;
  不能据此认定本项目已证明某个具体视觉捷径。
- [Mandlekar et al., What Matters in Learning from Offline Human Demonstrations (CoRL 2021)](https://proceedings.mlr.press/v164/mandlekar22a.html):
  演示质量、算法选择和训练/评估目标差异影响离线模仿表现。最终必须用闭环评估本方案。

### 19.7 续 §19:Codex 实现的 attempt01-08 现状,以及我跑的 attempt09(2026-09-11, session cf 接手)

Codex 把 §19 实现出来了(`image_servo{,_config,_policy}.py` + `image_bottle.py` + `bottle_image_servo.json`,
11 个测试通过),跑到 attempt08 停下。**8 次尝试,每次 4 局,0 成功。**失败演化:
`RGB approach could not reach a visible grasp region`(02/07)→ `IK residual 0.035 m`(04)→
`baseline clearance: finger vs bottle`(06)→ `RGB servo sweep: finger vs bottle`(08)。
**手指扫到瓶子是收敛后的主要死因**,正是 §19.3 自己预警的 dex1 长指问题。

**发现一:`visual_prepare` 让手臂仍然是乘客,占了 89% 的 episode。**
attempt08 的 190 个样本:`visual_prepare` 170、`align` 16、`clear_entry` 4。
`visual_prepare` 期间**掌高的标准差只有 0.0073 m**(170 个样本),而掌心到目标的高度差全程维持在
0.18-0.26 m:

| 底盘到瓶深度 | n | 掌高(骨盆系) | 目标高 | 差 |
|---|---|---|---|---|
| >0.80 | 54 | 0.105 | −0.149 | 0.255 |
| 0.60-0.80 | 43 | 0.101 | −0.087 | 0.192 |
| 0.44-0.60 | 73 | 0.092 | −0.106 | 0.197 |
| <0.35 | 9 | 0.052 | −0.127 | 0.176 |

`prealign_depth=0.44` 以上走固定 view 位姿,以下才开伺服。**结构和 visible200 是同一个,只是压缩到了最后 11%。**

**算术**:实测下降速率 **0.06-0.10 m/s**,需要覆盖 **0.21 m** → 约 **2.3 s**;
而伺服窗口只有 **0.5-0.8 s**,死的时候还差 0.13-0.20 m。接近过程本身有 4-9 s。
按这个算术,"更早开始下降"应该足够。

**attempt09(我跑的,唯一改动 `reach.image_control.prealign_depth` 0.44 → 1.30,同 seed 31000)**:
乘客结构确实消失了(`align` 421 / `visual_prepare` 8,还两次到达 `enter`),**但仍然 0/4,且接近更差**
(2 局 44 s approach 超时,attempt08 是 0 局)。

**原因很干净:`reacquire_ticks` 从 4/3/19 暴涨到 686/144/709。**

| | r1 | r2 | r3 |
|---|---|---|---|
| attempt08 reacq | 4 | 3 | 19 |
| **attempt09 reacq** | **686** | **144** | **709** |
| attempt09 深度轨迹 | 1.18→0.90 | 0.56→0.34 | **0.66→1.21(退后)** |
| attempt09 高度差 | 0.173→0.160 | 0.243→0.154 | 0.172→0.140 |

`_tick` 在 `fit is None or uncertainty > approach_uncertainty_m(0.15)` 时回退到 `_view_q()`。
**下降所需的手臂位姿,恰好就是丢掉腕部视野的位姿**;丢了视野 → 回退 view 位姿 → 撤销刚才的下降 →
再伺服,来回几百次,底盘因此原地打转甚至后退(r3 深度从 0.66 退到 1.21),高度差几乎没降。

**所以真正的约束被测出来了,而且是一个二选一:**

> **要么保持 view 位姿(看得见,但手臂当乘客、来不及下降、手指扫倒瓶子),
> 要么提前下降(不当乘客,但丢失腕部视野、陷入 reacquire 循环)。**

Codex 的 `view_elbow` 位姿存在的理由正是前者;我的 attempt09 证明了后者的代价。
**两条都走不通,说明缺的是第三条:下降轨迹必须与"保持目标在腕部视野内"耦合**——
即下降的同时用腕部俯仰把目标保持在画面里(80° FOV 有余量),而不是先摆一个固定 view 位姿再下降。
这与用户最初提的"让 right_wrist_camera 一直正对瓶子,不管停下还是行走"是同一件事;
本轮把它从直觉变成了**被测量的硬约束**。

**另一条可能的出路**:让远场定位不依赖腕部相机。实测 ego 在卡死帧里 91% 能看见瓶子(§18.6),
但 1.2 m 处单目 cap 轮廓的 `uncertainty_m` 过不了 0.15 的门,所以现在远场其实是**靠腕部相机撑着**的。
把远场不确定度门或 ego 侧的定位做扎实,同样能解开这个二选一。

**配置已还原**(`image_control` override 已移除,回到 Codex 的 `prealign_depth=0.44`)。
attempt09 目录保留作证据。**没有改 Codex 的任何源码**;11 个测试仍然通过。

### 19.8 远场定位这条路:找到并修掉一个 0.545 m 的感知错误,以及它一直掩盖着的原地打转(2026-09-11, session cf)

§19.7 的判断"远场靠腕部相机撑着,应该把 ego 做扎实"**部分错了**。把 attempt08/09 的 622 次
**已接受**拟合按 `perception_error_m`(对真值的审计)分组后,真相相反:

| 使用的视角 | n | 深度 | 报告的 uncertainty | **真实误差** | 校准比 |
|---|---|---|---|---|---|
| ego + wrist 融合 | 551 | 1.10 | 0.0172 | **0.0176** | **1.0x 准确** |
| **仅 wrist** | **69** | 0.62 | **0.0757** | **0.545**(p90 0.623) | **低估 7x** |
| 仅 ego | 2 | 0.32 | 0.0944 | 0.049 | 1.9x |

**ego 从来不是弱项**:它在每个距离段都以 ~100% 命中瓶盖(0.9–1.4 m 时中位 20 px)。
**危险的是 wrist 单视角**:误差 0.545 m,其中 **98% 是纯深度**,而且一律把瓶子定位得**比真实位置近 0.53 m**
——单目"按轮廓宽度算深度"塌掉了。而它报告的 uncertainty 是 0.076,**顺利通过 0.15 的门**。

**根因**(`image_bottle.fit_bottle`):两视角不一致时回退到单视角,选择方式是
`min(observations, key=条件数)` —— 也就是挑**轮廓更大、条件数更好**的那个,桌边就是 wrist。
而那段注释自己写的是"Moving wrist proprioception can lag the RGB capture"。
**它把责任归给腕部相机,然后回退到腕部相机。**

**修法**:不一致时保留 **head 视角**。已加回归测试(`test_disagreeing_camera_pair_keeps_the_head_view`)。
(另外试过给单视角加"宽度敏感度"不确定度项,**无效,已撤**:错误的单目深度是自洽的——它认为瓶子很近,
于是它自己的深度不确定度也确实很小,该项仍放行 98% 的坏拟合。真正能区分的只有跨视角一致性。)

**修完感知,露出了一个它一直掩盖着的 bug:机器人原地打转。**
`_send` 的远场分支里 `yaw = pelvis_yaw + clip(atan2(target[1], target[0]), ±far_yaw_lead)` ——
`target` 是骨盆系里的瓶子,而站位**故意**横向偏置 `approach_lateral=0.195`,所以 `target[1]` 收敛到
**−0.195 而不是 0**。于是每个 tick 都命令"比当前朝向再右转约 10°",**永不收敛**。

attempt08 为什么没炸?因为 wrist 单视角把深度报小了,`depth <= walking_mid_depth(0.65)` 几乎总成立,
走的是不改朝向的那个分支。**感知一修好,这个分支开始运行,机器人就开始转圈。**

**修法**:朝向瞄准**接近目标**而不是瓶子(`goal = target - relative_goal`,正是 `movement` 用的同一个向量)。
瓶子方位角随距离发散(−9.5°→−31°),目标方位角收敛(−0.3°→−14°)。

**四轮对照(同 seed 31000,每轮 4 局)**:

| attempt | 改动 | 拟合数 | 审计中位 | **>0.2 m 错** | 仅 wrist | net/travel | 航向扫过 | 最近深度 |
|---|---|---|---|---|---|---|---|---|
| 08 | Codex 原版 | 190 | 0.011 | **20%** | 22% | 0.80 | 180° | 0.31 |
| 09 | prealign 0.44→1.30 | 432 | 0.023 | 4% | 6% | 0.38 | 1734° | 0.50 |
| 10 | + 不一致保留 head | 624 | 0.036 | **0%** | 0% | **0.07** | **2738°** | 0.88 |
| 11 | + 朝向瞄准目标 | 401 | 0.023 | **1%** | 1% | 0.54 | 729° | 0.43 |

**结果**:感知坏拟合 20% → 1%;打转 2738° → 729°(仍高于 08 的 180°,**残留未清**);
走近恢复(net/travel 0.07 → 0.54,最近深度 0.88 → 0.43,但仍不及 08 的 0.31)。

**四轮仍然 0/4**,而且 attempt11 的失败**回到了最初那个问题**:
`RGB servo sweep / baseline clearance: right_dex1_finger_*_link vs water_bottle_object`
—— **dex1 长手指扫倒瓶子**,正是 §19.3 自己预警的那条。

**现在的判断**:感知与远场朝向这两个**混淆因素已经removed**,剩下的是一个干净的近场几何问题:
长手指从上方进入时会先碰到瓶子。§19.3 已经写明"不要把腕高严格单调当要求,安全绕行可以暂时增大掌心距离"——
这正是现在要做的设计,而不是继续调远场。残留的 729° 打转也应一并查清(怀疑在 `depth<=0.65` 分支,
那里朝向仍是默认的 `_facing(state)`)。

产物:attempt09/10/11 目录;改动在 `image_bottle.py`(不一致时保留 head)与
`image_servo_policy.py`(朝向瞄准目标),各带证据注释;`tests/test_image_servo.py` 12 项通过。

### 19.9 特权教师 + 逐帧可学性审计:用真值控制,用 RGB 作证(2026-09-11, 用户提出)

用户的问题:专家全程用真值会不会让切换点又变得学不到?
**答案:把切换条件写成"两边都能算的量",真值只负责算得准;然后逐帧验证学生会不会做同样的决定。**

**先测感知预算**(1562 次融合拟合,对真值审计):

| 瓶子距离 | 中位误差 | **p90** | 腕部瓶盖像素 |
|---|---|---|---|
| 0.25-0.40 m | **0.0088** | **0.015** | 694 |
| 0.40-0.60 m | 0.0050 | 0.016 | 169 |
| 0.60-0.90 m | 0.0129 | 0.041 | 51 |
| 0.90-1.40 m | **0.0280** | **0.084** | 18 |

**近场感知很好,远场很差。**于是判据是:**一个切换只有在"容差 ≫ 该距离上的感知 p90"时才成立。**
按这张表,"什么时候合爪"(触发在 0.25-0.40 m)可以是切换;**"什么时候抬手"(0.6-1.2 m,p90 41-84 mm)
不能做成切换,必须是连续函数。**

**实现**(`image_servo_truth_control`,默认 False,不动 Codex 原路径):
控制端 `target` 换成真值;同一帧仍算 RGB 拟合,并用**同一个 `decide()`** 在 RGB 估计上**再求值一次**;
逐帧记录 `intent_rgb` / `agree` / `aim_delta_m` / `rgb_usable`。真值从不进入被记录的观测,
它只挑选动作;审计说明这个动作能否从图像复现。合爪不再要求 RGB 的 `precise`(改为记录 `precise_rgb`),
拟合不可用时也不再回退 view 位姿——控制完全不依赖感知。

**attempt12_truth 结果(143 帧,同 seed)**:

```
一致率          115/143 = 80.4%
RGB 拟合可用     118/143 = 82.5%
可用且一致       115/118 = 97.5%
```

| 瓶子距离 | n | **一致率** | 感知误差 |
|---|---|---|---|
| 0.25-0.40 m | 16 | **100%** | 0.016 |
| 0.40-0.60 m | 41 | 93% | 0.011 |
| 0.60-0.90 m | 39 | 97% | 0.014 |
| **0.90-1.50 m** | 47 | **49%** | **0.102** |

分歧的来源(28 帧):**25 帧是"RGB 根本不可用"**,只有 3 帧是真的选了不同意图
(都在 `prealign_depth=0.44` 这条边界上,正是"边界落在噪声里"的典型)。

**结论**:
1. **近场决策 100% 可复现** —— 抓取那几个切换是学得到的,设计在那里是成立的。
2. **不可学的部分全在远场**,而且主要是"学生根本看不清"(21% 的 `visual_prepare` 帧 RGB 不可用),
   不是"看清了但选错"。远场只需要走向瓶子,方位角即使深度很差也是准的——
   **远场应当改成只依赖方位角、不要求精确拟合**。
3. `prealign_depth=0.44` 这条硬边界应换成连续过渡——它同时也是 §19.8 里"手臂当乘客"的成因。

**专家可靠性**:不再有 reacquire 死循环,`align` 误差单调收敛(0.210→0.091)。
**但仍然 0/4**,三局全部死在 `right_dex1_finger_1_link vs water_bottle_object`——
**长手指几何问题现在是唯一的阻塞项**,感知与远场朝向两个混淆因素都已排除。

**注意**:近场只有 16 帧(episode 都在 18-21 s 被手指问题打断),100% 这个数**样本很小**,
等长手指问题解决、能跑完整回合后需要复测。

**另外修了两处 §19 的既有超标**(`make lint` 在 HEAD 上已经是红的,不是本次引入):
`_tick` 94 行、`reach_config.__post_init__` 89 行,均超 80 行上限;已分别抽出 `_log_sample` 与 `BOOL_FIELDS`。

### 19.10 长手指净空:三处改动实现并跑了 4 局,**我给的机制是错的**(2026-09-11)

按 §19.9 之后的计划实现三条,跑 attempt13_clearance(同 seed 31000)。

**先量几何(MJCF FK,不是估计)**:

```
手指沿进入轴最远伸到   0.1941 m (从 wrist 帧)
抓取点 OFFSET 在        0.1515 m
→ 手指超出抓取点        0.0426 m
瓶子半径                0.0226 m
+ 余量(感知 p90 0.015 + 底盘/跟踪 ~0.02)
────────────────────────────────
所需净空                0.1002 m      vs 原 entry_gap 0.075
```

**实现**:(1) `ServoLimits` 增 `finger_reach_past_grip` / `entry_margin` / `base_hold_band`,
净空改为**由几何 + 物体半径算出**(换物体自动适配),`entry_gap` 退化为下限;
(2) `decide(obs, limits, clearance)` 用算出的净空;
(3) `_advance_scale()` 把**底盘速度**按剩余净空连续压到 0(两个输入——本体的抓取点、视觉的物体——学生都看得到)。
逐帧记录 `grip_standoff_m` / `clearance_m` / `advance_min`,验收可直接算。

**结果:仍然 0/4**,但有一局从"手指扫倒"变成了"RGB grasp feedback timed out"——**约束在那一局确实起了作用**,
代价是死锁(底盘被压住 + 手臂在 `clear_entry` 里一直后退却始终不对准 → 超时)。另两局照旧扫倒。

**然后验收数据把我的机制否掉了。**逐帧的抓取点净空:

| | |
|---|---|
| 净空要求 | +0.100 m |
| 实测 median | **−0.024 m** |
| 实测 min | **−0.065 m** |
| 低于几何下限 0.0653 的帧 | **62%** |

**是负的** —— 抓取点沿进入轴已经**越过**瓶子。查进入轴方向(逐帧 `wrist_rotation[:,0]`,骨盆系):

```
全程 median:  fwd +0.253  lat +0.353  up −0.794
近场 410 帧:  fwd +0.180              up −0.860     ← 偏离骨盆前向 75°
```

**进入轴几乎是竖直向下的 —— 这个抓取是从上方罩下去的。**
所以净空是被**手臂自己的下降**吃掉的,不是被底盘的前进吃掉的。
**我改的那条"底盘速度耦合"在原理上就管不到它**,这也解释了为什么另两局照样扫倒。
(§19.8 那条"底盘比手臂快"的观察没错,但它解释的是"来不及下降",不是"手指扫瓶"。两件事。)

**`clear_entry` 触发了 335/522 帧**(64%)在试图抬起来,净空却始终是负的 —— 事后反应式的后退赢不过下压。

**下一步(明确)**:净空要作为**对命令 aim 的硬约束**来执行,而不是靠 `clear_entry` 事后补救——
即在未对准时,把 aim 沿进入轴**投影**到"抓取点不低于 clearance"的安全集里,
使它在几何上不可能进入瓶子的柱体;`decide()` 的 L 形结构保留,只是把反应式改成投影式。
底盘耦合这条可以保留(它没有害处,而且 §19.8 的"来不及下降"仍然真实),但它不是这个 bug 的解。

**另外**:这一局 `agree` 掉到 40%、`rgb usable` 掉到 42%(attempt12 是 80%/83%)——
手臂长时间停在罩住瓶子的怪姿势里,把腕部视野也挡住了。**这个姿势本身就不可学**,
又一条说明净空必须在几何上被保证、而不是被事后修正。

### 19.11 更正 §19.10:几何模型是**对的**,被否掉的只是"底盘速度"这个执行手段(2026-09-11,导视频时发现)

§19.10 写的"抓取点净空是负的(median −0.024)、几何模型被否掉"是**聚合口径造成的误读**。
按**每局最小值**看:

| round | 结局 | **最小抓取点净空** | 几何下限 | 判读 |
|---|---|---|---|---|
| r1 | finger_2 扫瓶 | **+0.050** | 0.0653 | 短于要求 → 手指尖距瓶心 0.050−0.0426 = **0.007 m**,落在半径 0.0226 内 → **必撞** |
| r2 | finger_1 扫瓶 | **+0.057** | 0.0653 | 同上,0.014 m,仍在半径内 → **必撞** |
| r3 | 超时 | **−0.065** | 0.0653 | 手臂在瓶子上方罩了 62 s 的怪姿势 |

**两局真正扫瓶的净空都是正的,而且正好卡在几何要求之下** —— 这**正是手指外伸模型预测的结果**。
§19.10 的 median −0.024 / 62% 越限,是 r3 那 62 秒的长时间停留贡献了绝大多数帧所致,
**是加权假象,不是机制。**

**所以要撤回的只有一句**:"抓取点越过了瓶子、几何模型被否"。**几何模型成立,而且预测准确。**

**仍然成立的**:进入轴近乎竖直(近场 fwd +0.180 / up −0.860,逐帧实测),
所以净空是被**手臂自身的下降**吃掉的,**底盘速度耦合管不到它** —— 这条否定依然有效,
两局照样扫瓶就是证据。

**对下一步的影响(变好了)**:既然模型能准确预测撞不撞,把净空改成**对 aim 的投影式硬约束**
就应当直接奏效——只要保证每帧 `grip_standoff_m ≥ clearance`,按模型手指就进不了瓶子的柱体。
验收也已经现成:逐帧的 `grip_standoff_m` 最小值不再低于 0.0653 即可,不必等成功率。

视频:`outputs/videos/image_servo_attempt13_clearance/{r0..r3,reel}.mp4`,
字幕带失败原因与该局最小净空(`scripts/experiments/image_servo_export_videos_20260911.py`)。
r3 的画面可以直接看到那个"罩在瓶子上方、把自己腕部视野也挡住"的姿势。

## 20. 默认 SONIC checkpoint 切到 sonic_v1_1(2026-09-11, session cf,用户裁定)

**这是一次配置决定,不是一次实验。** 用户指示:以后一律用 SONIC v1.1,把被牵连的代码与
文档全部改掉。本节记录改了什么、哪些既有结论因此失效、以及在读任何新数字之前必须先重测什么。

### 20.1 已知的反对证据(仍然成立,用户已裁定照切)

- §17:同一 case bank、配对 20 局,`policy/release` **9/20** vs `policy/sonic_v1_1` **0/20**,
  精确 McNemar **p=0.0039**。机制是**到位几何**不是抓取:v1.1 横向多走 11 cm(t=−6.38)、
  停得远 12 cm(0.498 vs 0.377,plan_r 0.32),approach 段 10/20 以 "stalled" 结束。
- 步长:同样 speed 0.30,v1.1 一步 4.3 cm,release 17 cm —— 现有 approach/brake 时序全是
  按 release 的步长标定的。
- 反方向的证据(也是切换的理由):§16 参考回放里 v1.1 的腰跟踪显著更好,
  waist RMSE **3.9°** vs release **12.9°**,torso 5.7 vs 12.4;且 v1.1 是自洽的
  model+config 对,而 `policy/release/observation_config.yaml` 是一份手改的第三变体。

### 20.2 改了什么(全部已验证)

| 层 | 改动 |
|---|---|
| 唯一默认值 | `scripts/paths.sh` 新增 `export SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"`;`locomani.paths.SONIC_POLICY` / `SONIC_POLICY_DIR` / `sonic_policy_file()` 是 python 那半边 |
| deploy 启动 | `start_stack.sh` / `restart_deploy.sh` 兜底默认同步改成 v1_1,并在启动前检查 `$SONIC_POLICY/model_decoder.onnx` 存在(外来 deploy 树只有 release 时会明确报错,不再表现为栈超时) |
| 离线 encoder | `sonic_encoder.py` 不再写死 release 的块表:`EncoderLayout.from_config()` 从 checkpoint 自己的 `observation_config.yaml` 推布局(跳过 `enabled: false`,cpp:1866 同款),并按 anchor 块名选 orientation_mode。v1.1 = 1751 维、anchor `..._heading_10frame_step5` @584、**heading 归一化**(左乘 `calc_heading_quat_d(base_quat)`,math_utils.hpp:300);release = 1762 维、@601、整个 base 四元数 |
| 离线 decoder | `sonic_decoder.py` / `eval_open_loop.py` / `rtc/{analyze_transition_capture,probe_prefix_models}.py` 默认解码器跟随 `SONIC_POLICY`。994 维 policy 观测两边**逐块相同**,这半边不需要分支 |
| provenance | `config_snapshot.json` 新增 `sonic_policy`(名字、目录、obs config 的 sha256、两个 ONNX 的字节数)—— 没有这个字段,事后无法判定一份数据集的 token 属于哪个 latent 空间 |
| 测试 | `tests/test_sonic_policy_default.py`(9 条):三个 shell 入口默认值一致、python 与 shell 一致、env 覆盖仍有效、两个启动脚本都有存在性检查、两套布局的维度/偏移/heading、`heading_quat` 丢俯仰留偏航、两种 anchor 在俯仰下确实不同、provenance 带 checkpoint |
| 文档 | `docs/setup.md`(下载 `--sonic-v1-1` + 验证)、`docs/collection.md`(链路图的模型路径)、`docs/dataset.md`(`motion_token` 行)、`docs/reference/silent_failures.md`(env 表 + 三条新静默失败行) |

验证:`sonic_encoder --selfcheck` 在 v1.1 与 release 上**都 PASS**(结构/码本/灵敏度);
单测全绿;`ruff` / `lint_env` / `lint_sizes` 无新增欠债。

### 20.3 码本相同,含义不同(2026-09-11 实测)

两个 encoder 在大量随机输入下的输出**都**落在 1/16 网格、饱和区间**都**是 `[−1.0, 0.9375]`
(32 级)。所以 `sonic_token.py` 的量化/吸附逻辑与 checkpoint 无关,
**而"合法 token"这类检查一个都拦不住跨 checkpoint 的误用** —— 变的是 latent 的含义。

**推论(重要)**:2026-09-11 之前采的每一份数据集(`action.motion_token`)与每一个 GR00T
checkpoint 都属于 **release 的 token 空间**。它们不能直接部署在 v1.1 的 deploy 上,
也不能用 v1.1 的 decoder 去算"度数"指标。要复现旧数字,整条链一起
`SONIC_POLICY=policy/release`。

### 20.4 切换后仍未重测的东西(读新数字之前必须先做)

1. **approach / brake 时序**:release 步长 17 cm/步 vs v1.1 4.3 cm/步。§17 的 0/20 主因就在这里。
2. `primitives_tuning.TRACK_ROT_BOOST = 4.5`:release 的逆增益。2026-08 的腕姿态扫描说在 v1.1 上
   "boost 无意义",但那是姿态扫描不是抓取线,所以**保留 4.5 并标注待测**,没有盲改成 1.0。
3. **站位/刹车带**(`reach_config` 的 0.30/42.0)与 `arrival_advance`:同样是 release 标定。
4. 期望的先后:先重测时序、再谈成功率;在 1 做完之前,任何 v1.1 的抓取成功率都不是 checkpoint 的属性。

## 21. Stage A:两个 checkpoint 的速度命令 → 步态交付图(2026-09-11, session cf)

§20.4 列的重标定第一步。**纯测量,没有改任何控制常量。**同一 session、同场景(`SCENE_RANDOM=0`)、
同协议(每腿 5 s、1.5 s warmup 后开始计、每腿前一次完整 reset + 朝空地转身)、每档 3 次。
工具 `src/locomani/backends/sonic/diag_shuffle.py` —— 它在 refactor step 2(`ec0eb1c`)被删,
本次从 `.claude/worktrees/old-78536ae/` 取回并移植 import;测量逻辑与被删版本逐行相同。
产物:`outputs/diagnostics/gait_speed_v11_20260911{,_knee}/{sonic_v1_1,release}/legs.json` + `report.json`,
报告脚本 `scripts/experiments/gait_speed_report_20260911.py`,runner `scripts/experiments/gait_speed_map_20260911.sh`。

### 21.1 实测表(net m/s 中位 [min,max],n=3;cm/步;carry = 命令 IDLE 后 3.5 s 的位移)

| 命令 | v1.1 net | v1.1 cm/步 | v1.1 carry | release net | release cm/步 | release carry |
|---|---|---|---|---|---|---|
| 0.185 | 0.008 [.008,.009] | 0.9 | 2.2 | 0.013 [.013,.014] | 1.2 | 2.7 |
| **0.20** `APPR_SHUFFLE` | 0.009 [.007,.016] | 1.0 | 2.9 | 0.018 [.014,.020] | 1.7 | 1.6 |
| **0.215** `APPR_MID` | 0.014 [.006,.014] | 1.3 | 1.8 | 0.016 [.007,**.039**] | 1.4 | 4.2 |
| 0.26 | 0.009 [.009,.010] | 0.6 | 1.9 | **0.123** [.118,.147] | 10.7 | 3.6 |
| 0.28 | 0.007 [.007,.007] | 0.5 | 2.9 | 0.179 [.164,.183] | 12.8 | 9.1 |
| **0.30** `APPR_COARSE` | **0.044** [.020,.065] | 3.1 | 2.9 | 0.203 [.151,.250] | 14.6 | 2.2 |
| 0.32 | 0.067 [.050,.073] | 3.4 | 3.5 | 0.239 [.221,.279] | 16.8 | 3.5 |
| **0.35** `NAV_SPEED_MIN` | 0.083 [.055,.166] | 9.7 | 4.4 | 0.245 [.240,.314] | 21.0 | 3.3 |
| 0.37 | 0.151 [.093,.193] | 13.2 | 2.8 | 0.260 [.231,.339] | 19.8 | 7.9 |
| 0.40 | 0.187 [.166,.195] | 10.9 | 3.7 | 0.348 [.293,.352] | 24.5 | 12.4 |
| **0.45** `NAV_SPEED` | 0.273 [.256,.307] | 15.9 | 6.8 | 0.400 [.392,.400] | 23.3 | 17.2 |
| IDLE 对照 | 0.0005,位移 0.1 cm(n=6) | — | — | 0.0027,0.1 cm(n=6) | — | — |

release 这条与 `primitives_tuning.py:306-317` 记录的 2026-08 表在 20% 内复现(0.26 0.123 vs 0.155,
0.30 0.203 vs 0.247,0.40 0.348 vs 0.29),**唯一不复现的是 0.215**:文档 0.040,实测中位 0.016,
三次 0.007/0.016/0.039 —— 5 倍散度。原因见 21.2:它正骑在 release 自己的台阶上。

### 21.2 两条曲线都是**台阶**,台阶位置相差 +0.065 的命令

- release:**0.215 → 0.26 之间跳 0.016 → 0.123 m/s(7.9×)**
- v1.1:**0.28 → 0.30 之间跳 0.007 → 0.044 m/s(6.6×)**

台阶以下两边同量级(release 0.013–0.018,v1.1 0.007–0.014,v1.1 慢 1.3–2×);
台阶以上 v1.1 一律更慢(0.32 处 0.067 vs 0.239 = 28%,0.40 处 54%,0.45 处 68%),步长也更短
(同样 ~1.4 Hz 步频下 v1.1 在 0.45 走 15.9 cm/步,release 23.3)。
**所以"v1.1 步长短"这个旧说法方向对,但可操作的事实是台阶位移** —— 按命令值写死的常量会落到台阶的另一侧。

### 21.3 这解释了 §17 的 0/20,而且不需要我先前那两个错的机制

先撤回两条:①"creep 8 s 预算超时"——`stream_reach._creep_to_band` 在**生产路径里没有调用者**
(定义在那里、被 `tests/test_ready_grasp_policy.py` 直接调用、一局里谁都不调它);
`round_phases_approach.py:77` 那句"最后几厘米现在闭环"是过期注释指向死代码。
②"开阔地净速证明 stall 不会触发"——净速和 approach 的进展量**不是同一个量**,approach 只计深度分量。

在路径内的算术(深度 1.2 → 刹车 0.31,coarse 0.58 m / mid 0.19 m / decel 0.12 m,`do_approach` 预算 45 s):

| 臂 | 命令 (coarse, mid, shuffle) | coarse | mid | decel | 合计 |
|---|---|---|---|---|---|
| release | (0.30, 0.215, 0.20) | 2.9 s | 12.2 s | 7.2 s | **22.2 s** |
| v1.1 现状 | (0.30, 0.215, 0.20) | **13.1 s** | 14.0 s | 10.8 s | **37.8 s** |
| v1.1 重挑 coarse | (0.41, 0.215, 0.20) | 2.9 s | 14.0 s | 10.8 s | **27.6 s** |

v1.1 现状下 37.8 s 是**纯闭合深度**的时间,45 s 预算只剩 7 s 余量,而这 7 s 还要装下横向修正
(§17 实测到位横向偏 11 cm)、抬臂、以及任何犹豫;同时 mid/decel 两段对 stall 规则
(5 s 内不足 1.5 cm = 0.003 m/s)的余量只有 4.5× 和 2.9×,横向一分速就压到阈值以下。
于是要么 45 s 超时、要么判 `stalled`,都停在距刹车深度约一个 decel band(0.12 m)的位置 ——
与 §17 实测 park 差 0.121 m 吻合,并直接产出它记录的 `[stand] arrival left the object out of reach`(×5)。
注意 visible profile 下 `ctx.coordinated` 非空,`round_phases_approach.py` 的 forward-only repair
**次数为 0**,所以停远一次就没有第二次机会。

### 21.4 重挑结论(待 Stage B 落地)

1. **`APPR_COARSE` 0.30 → 0.41**:这是唯一严重错的常量(0.044 vs release 0.203,4.6×),
   而 0.41 交付 0.187 m/s、repeat 散度 ±8%,把预算从 37.8 s 拉回 27.6 s。
2. **`APPR_MID` / `APPR_SHUFFLE` 匹配不了**:release 交付 0.016/0.018,而 v1.1 的台阶**跳过了这一段**
   (0.28 给 0.007,0.30 给 0.044)。没有命令能交付 0.016。可选:保持现状(0.0136/0.0087,慢 0.85×/0.48×),
   或两档都取 0.215(0.0136)。**不要**取 0.28–0.30 之间的插值:那是台阶内部,双稳。
3. 因此 v1.1 在 approach 里的动态范围只有 3×(0.067–0.203),release 是 11×(0.018–0.203):
   **v1.1 没有同等的慢爬行档**,decel 斜坡在它身上必然更平。这会改变最后一段的速度剖面,
   而记忆 `policy-stop-point-is-set-by-the-crawl` 说学到的停位正是由这段爬行决定的 ——
   属于**数据形状的改变**,不只是控制参数,Stage E 的配对验收必须把它当成一个真实变量。
4. `NAV_SPEED` 0.45 交付 0.273 vs release 0.400(68%),`NAV_SPEED_MIN` 0.35 交付 0.083 vs 0.245(34%)。
   长距离行走会慢,但都远在 stall 阈值之上(91× / 28×),不是阻塞项;>0.45 未测。
5. **carry 是好消息**:v1.1 在 approach 用得到的速度段只滑 1.9–3.5 cm(release 在 0.40/0.45 滑 12–17 cm),
   IDLE 对照 0.1 cm 说明这是真动量。所以 Stage C 的 `visible_brake_depth` 预期只需小修。

### 21.5 一处 provenance bug(已修,数据未受影响)

首次 runner 用 `SONIC_POLICY=... bash start_stack.sh` 的**前缀**写法,只传给了起栈命令,
sweep 进程仍读到默认值,于是 release 臂的 `legs.json` 把自己标成 `sonic_v1_1`。
**测量本身没问题** —— sweep 只通过 ZMQ 跟已起好的 deploy 对话,而 `stack.log` 的 `policy:` 行与
`deploy.log` 的 `Using observation config: policy/release/...` 都证明跑的是 release。
已改成 `export`,并且报告脚本改为**以 `stack.log` 为准**、与 `legs.json` 的自述不一致时显式打印警告
(而不是悄悄采信任一方)。另加两处工具改动:一条 IDLE 对照腿(carry 列没有它读不了),
以及"摔倒只损失当前这条腿、连续 3 次才中止"(被删版本第一次摔倒就 break 掉整个 grid)。

## 22. Stage B:`APPR_COARSE` 0.30 → 0.40,10 局配对 pilot(2026-09-11, session cf)

§21.4 的落地。**只动一个变量**:approach 的远场速度命令。MID/SHUFFLE 不动(§21.4 第 2 条:
v1.1 的台阶把 release 交付的 0.016–0.018 m/s 整段跳过去了,没有命令能匹配)。

### 22.1 代码:速度常量改为按 checkpoint 查实测表

新模块 `src/locomani/backends/sonic/gait_speeds.py`,带 §21.1 的实测表与出处:

| checkpoint | (coarse, mid, shuffle) | 交付 m/s |
|---|---|---|
| `release` | (0.30, 0.215, 0.20) —— **历史生产三元组,不动** | 0.203 / 0.016 / 0.018 |
| `sonic_v1_1` | (**0.40**, 0.215, 0.20) | 0.187 / 0.014 / 0.009 |

取 **0.40 而非反查的 0.41**:0.40 是实测点(0.187 m/s、repeat 散度 ±8%、release 的 92%),
0.41 落在 0.05 的间隔里靠插值,换来的 0.1 s 预算不值得一个没测过的命令。
未测过的 checkpoint 调 `approach_bands()` **直接抛错并给出测量命令**,不静默回退到别的 checkpoint。

改到的站点:`round_phases_approach.py`(生产)、`stream_reach._creep_to_band`(生产无调用者、有测试)、
`recoverable_grasp_policy.move_to_region`(profile 里 `recovery_attempts: 0`,休眠)。
后两处**正因为不跑才要改** —— 否则 release 时代的字面量会一直躺在那里,等某天有人打开 recovery 才炸。
`primitives_tuning.NAV_SPEED/NAV_SPEED_MIN` 不动:匹配 release 的 0.400 m/s 需要 >0.45 的命令(未测),
且两者对 stall 规则有 91×/28× 余量。`place_phase.turn_command(speed=0.20)` 属 pick_place 线,未测不动。
测试 `tests/test_sonic_policy_default.py::GaitBandTests` 5 条(release 三元组被钉住、v1.1 的 COARSE 必须
在自己台阶之上且达 release 的 80%、未测 checkpoint 抛错且错误信息含 `diag_shuffle`、approach 无字面量、
两处休眠站点都读表)。

### 22.2 结果:同 10 个 case(`v11_bottle_ab20_20260910/cases20.json` 前十),三臂逐局配对

| 臂 | ok | approach exit | brake x | stop x | carry | park |
|---|---|---|---|---|---|---|
| release(coarse 0.30 → 0.203 m/s) | **4/10** | brake 8,**stalled 1** | 0.309 | 0.306 | 0.003 | **0.349** [0.31,0.45] |
| v1.1(coarse 0.30 → 0.044 m/s) | **0/10** | brake 4,**stalled 5** | 0.31(5 局 nan) | 0.332 [0.31,**0.51**] | 0.001 | **0.467** [0.43,0.68] |
| v1.1(coarse **0.40** → 0.187 m/s) | **3/10** | brake 8,**stalled 0** | 0.307 | 0.300 | 0.007 | **0.393** [0.36,0.43] |

**验收(§21.4 定的、不看成功率的那几条)全部通过**:
- `stalled` **5/10 → 0/10**(release 是 1/10);
- `[stand] arrival left the object out of reach` **×3 → ×0**;
- park 中位 0.467 → **0.393**,补回 60% 的差距(release 0.349),离散也从 [0.43,0.68] 收到 [0.36,0.43];
- brake/stop 几何与 release 几乎相同(0.307/0.300 vs 0.309/0.306),**所以 Stage C 的 `visible_brake_depth` 不用改**;
- 墙钟:10 局 4 分 03 秒(24 s/局),远场段从每局约 13 s 回到约 3 s。

配对检验:vs v1.1-0.30 是 3:0(exact McNemar p=0.250),vs release 是 2:3(p=1.000)。
10 局配对分辨不了 3 vs 4,**所以"追平 release"这句只能由遥测支撑,不能由成功率支撑**;
但"修好了 arrival"这句是遥测直接给的。注意 release 的 carry 只有 0.003 m —— 旧注释里
"刹车后滑 5–18 cm"是被取代的 settle-loop 时代的数字,visible expert 不是那样停的。

### 22.3 失败模式搬家了:从 arrival 搬到 grasp

新臂的 7 局失败里:`[grasp] coupled safe-hand IK did not track` ×2、
`[grasp] visible expert: stand infeasible` ×2、`[grasp] local alignment timed out` ×1、
`[stand] could not turn -24 deg ...` ×1(**release 在同一 case 上报同一句**,属 case 属性不是 checkpoint)、
`[stand] walked 0.54 m off the spawn while standing up` ×1(**v1.1 特有**,站起过程中底盘走掉,
旧臂在另一 case 上是 0.64 m —— 这是起身/heading anchor 行为,不是步速,Stage C/D 要单独看)。

所以 approach 修好之后,**绑定约束变成了抓取相位**(IK 跟踪、local alignment)和起身漂移。
这与 §20.4 排的顺序一致:先时序、再刹车带、再蹲、最后腕姿态。

产物:`outputs/experiments/stageb_coarse_pilot_20260911/{manifest.jsonl,gym.log,stack.log,stageb_report.json,videos/}`;
报告脚本 `scripts/experiments/stageb_report_20260911.py`(三臂逐局配对表),runner
`scripts/experiments/stageb_coarse_pilot_20260911.sh`;带字幕的 reel
`outputs/videos/stageb_coarse_pilot_20260911/reel.mp4`(2 成 2 败,字幕写 case/结局/approach 退出原因/brake→stop/park),
导出脚本 `scripts/experiments/stageb_reel_20260911.py`(按 `video_start_wall` 对齐轮次与文件,不靠 mtime 猜)。

## 23. 抓取相位:`coupled safe-hand IK did not track` 的机制,以及一次不成立的验证(2026-09-11, session cf)

§22 之后绑定约束搬到抓取相位。本节定位了它的机制(结论牢固),并记录**一次无法下结论的验证**(样本量与上游方差)。

### 23.1 卡住的是旋转残差,不是伸不到

§22 的 pilot 里这个失败**只记了一个 tick 计数**,所以先补仪表:阻塞时刻记 IK 残差 `pe/re` + 容差 +
`aim`(骨盆系)+ phase + delivered + pelvis_z(上限 5 条),蹲下起点记 `pregrasp_root` /
`pregrasp_azimuth_deg` / `pelvis_at_squat` / `grip_root`。然后**只重跑失败的 3 个 case**。

case `7100:5` 连续三拍(产物 `outputs/experiments/stageb_ikblock_20260911/manifest.jsonl`):

| pos_err | tol | rot_err | tol | phase | delivered |
|---|---|---|---|---|---|
| 0.0088 m | 0.08 | **0.2013 rad** | **0.20** | 0.938 | 0.093 m |
| 0.0098 | 0.08 | **0.2208** | 0.20 | 0.965 | 0.098 |
| 0.0106 | 0.08 | **0.2323** | 0.20 | 0.986 | 0.102 |

**位置残差在容差内 9 倍 —— 手臂够得到。**超限的是腕姿态,只超 0.7%,且发生在 phase 0.94–0.99(下蹲末段)。

**机制**:transport 锁的是手掌的**世界系**姿态(`target_rotation` 取蹲下起点的世界掌姿,每拍按活的骨盆
旋转重算 `br.T @ target_rotation`),而骨盆在下蹲时前俯 —— 骨盆自己的俯仰就变成手臂的旋转需求。
从 pose sidecar 量最深蹲位的骨盆俯仰:release **+9.8 / +11.1 / +11.8 / +11.4°**(峰值到 +14.5),
v1.1 **+12.1 / +12.5°**(峰值 +14.4)。而 `TRANSPORT_ROTATION_TOL = 0.20 rad = 11.5°`:
**这个容差比下蹲自己产生的俯仰还小**,两个 checkpoint 都骑在阈值上,v1.1 多约 1° 就压过去。
两个 TOL 原本都是**没有注释、没有出处的裸字面量**。

**撤回我自己在诊断中途给的读法**:`pregrasp_ahead_of_pelvis_m`(pregrasp 点在骨盆前向的投影)
在 §22 的数据上把成败完全分开(OK ≤ −0.044,阻塞 ≥ −0.080),我据此说过"目标折到肩后、伸不到" ——
**位置残差证明伸得到**,`ahead` 是与腕姿态极端程度相关的**代理量,不是原因**。

### 23.2 改动:容差按实测推导,0.20 → 0.35 rad

0.35 rad = 20° 覆盖实测峰值 14.4° 加余量,仍能拒掉离谱的腕解。选它而不是"把锁姿态改到骨盆系"是因为:
容差只决定**要不要放弃**,IK 成功时的指令一个字都不变;改成骨盆系会真的改变手掌运动、从而改变录进
数据集的动作。护栏仍在:`screen_motion` 扫掠碰撞筛查与 `reference_check` 照跑,随后的 alignment
阶段会把姿态误差伺服到 ~0.03 rad(§22 的 r3 实测 0.50 → 0.035)。

### 23.3 验证不成立:上游方差压过了要测的效应

同 10 个 case 重跑(`outputs/experiments/stagec_rottol_20260911/`),**0/10**。但**不能读成容差改坏了**:

| 运行 | 轮次 | 进到下蹲 | IK-block 失败 | 见到的 blocked_ticks | stand infeasible | `[stand]` 阶段失败 |
|---|---|---|---|---|---|---|
| §22 pilot,tol 0.20 | 10 | **6** | **2** | [0,0,0,5,**11**,**11**] | 2 | 2 |
| 3 局重跑,tol 0.20 | 3 | 2 | 0 | [2,5] | 0 | 1 |
| tol 0.35 | 10 | **2** | **0** | [2,3] | **4** | **4** |

tol 0.35 那一轮的 8 局失败**全部发生在 coupled transport 之前**(4 局 `stand infeasible` + 4 局 `[stand]`
阶段:`teleport landed 0.10 m off`、`did not stabilise (pelvis z 0.085)`、`walked 0.55 m off the spawn`、
`could not turn -24 deg`),这些阶段**根本到不了那个容差**。所以:

- 容差的预期效应**方向正确**:tol 0.20 下 8 局进到下蹲、2 局 11-tick 中止;tol 0.35 下进到下蹲的 2 局
  blocked_ticks 只有 2 和 3、0 中止,且**没有出现**说明松容差放进坏解的新失败(无扫倒、无 `path blocked`)。
  但 **n=2,不成立**。
- 真正暴露出来的是**上游方差**:同一配置、同一批 case,"进到下蹲"的比例是 6/10、2/3、2/10。
  **所以 10 局 pilot 根本测不出这条线在 v1.1 上的成功率**,3/10 与 0/10 之间没有可比性 ——
  §22 里"3/10 vs 4/10 分辨不了"的说法要加强成:**连同一配置的两次 10 局之间都分辨不了**。
- 一个协议层面的疑点(未验证):这三次栈在 20 分钟内连开三个,而 `restart_deploy: false` 让 deploy
  跨轮复用;tol 0.35 那轮开局就 `teleport landed off` 与 `did not stabilise`,像是起栈状态更差。
  下次测这条线要么每轮重启 deploy,要么两次运行之间留冷却,并记 CPU gate。

### 23.4 下一步(按我现在的判断排序)

1. **先修上游**:`[stand]` 阶段的方差(起身走掉 0.5 m、teleport 偏、复位站不稳)已经是最大项,
   而且"起身走掉"是 v1.1 特有的(§22.3)。在它稳定之前,任何下游改动都测不出来。
2. 容差改动**保留**(机制牢固、方向正确、无新失败),但标注"未经足够样本验证"。
3. 要判定它,需要**汇总统计**而不是成功率:把"进到下蹲的轮次"的 `blocked_ticks` 分布累到 n≥10,
   约 30 局或只跑能进到下蹲的 case。

## 24. 复位把机器人抬高了 10.8 cm(全仓库 bug),以及残差其实是横向的(2026-09-11, session cf)

§23 定的顺序:先修上游。上游查出来的是一个**与 checkpoint 无关的复位 bug**。

### 24.1 漂移不是走路,是落地瞬态

pose sidecar 逐拍记着**指令**(mode/movement/facing/speed/height),所以能分清"走掉"和"被摔出去":
失败局的位移**全部发生在复位后第一秒**,而那一秒的指令是 `mode=IDLE` + 零 movement 向量;
之后都降到 1–4 cm/s。失败局第一秒走 0.48 m,成功局 0.07–0.145 m。
(第一版分析我算错了 —— sidecar 跨了仿真时钟复位,`t` 非单调,把 teleport 跳变算进了位移;
按时钟段切分后才是上面的数。)

### 24.2 根因:`_seat_on_floor` 用包围球半径当脚底

`privileged_pub._seat_on_floor()` 原来取 `min(geom_xpos[z] - geom_rbound)` 当最低点,
而 `geom_rbound` 是**包围球半径**。实测(`outputs/scene/sim/scene_locomani.xml`,控制器标称站姿
hip −0.1 / knee +0.3 / ankle −0.2):

| 量 | 值 |
|---|---|
| 标称站姿的骨盆 z(落座前) | 0.7930 |
| 代码估计的最低点 | **−0.0994** → 抬高 0.0994 → 落座 **0.8924** |
| **真实**最低表面点 | **+0.0084** → 应再降 8 mm → 落座 **0.7846** |
| 决定估计的那块脚底 mesh | `rbound = 0.1291`,真实向下延伸 **0.0213** |

**0.8924 与全部 62 局实测的 z0 = 0.892 完全一致;0.7846 与实测稳定高度 0.785 完全一致。**
所以**每一次复位、每一条任务线,都是一次 10.8 cm 自由落体**,落地瞬态就是 stand-drift 预算抓到的东西。
记忆条目 `sim-reset-teleports-into-the-air`("复位后骨盆比落点低 10 cm、ncon=0")到此有了原因。

### 24.3 修复与验收(验收统计不是成功率)

改为按 geom 类型算真实最低表面点(mesh 走顶点,sphere/capsule/box/cylinder 走解析式,
未知类型退回 rbound 并注明保守),落座留 `SEAT_CLEARANCE_M = 2 mm`。
测试 `tests/test_seat_on_floor.py` 4 条(五种 geom 的解析最低点、"包围球必须更悲观"、
落座间隙 = 2 mm、真实场景落座高度 <0.80 且 >0.77)。

验收用**复位后第一秒的位移**(下游任何改动都碰不到这个量,所以树里那个 0.35 rad 容差不构成混淆):

| | teleport z | 第 1 秒中位 | p90 | 最大 | 2 s 超 0.35 m | 摔倒 |
|---|---|---|---|---|---|---|
| 修复前 release(20) | 0.892 | 0.071 | 0.112 | 0.211 | 0/20 | 0 |
| 修复前 v1.1(42 局汇总) | 0.892 | 0.099 | 0.319 | **0.489** | **4/42** | **1** |
| **修复后 v1.1(10)** | **0.787** | **0.034** | **0.071** | **0.083** | **0/10** | **0** |

中位降 3 倍、最大降 6 倍,而且修复后的整条分布**比修复前 release 的还紧**(0.083 vs 0.211);
这一轮 10 局里 `[stand]` 阶段失败为 0。**这条改动改变了每一集的初始条件** ——
2026-09-11 之前的所有数据集都是从 10.8 cm 落体开始的,新旧数据在这一点上不同源。

### 24.4 噪声拿掉之后:残差是**横向**的,不是深度

修复后成功率 1/10(按 §23,10 局测不出率,不作为结论),但失败分布变了:
`stand infeasible` ×4、`local alignment outside comfortable IK` ×2、IK-block ×1、`disturbed` ×1。
把 park 拆成深度与横向(park² = depth² + lateral²,逐局中位):

| 臂 | park | depth | **lateral** |
|---|---|---|---|
| release | 0.349 | 0.306 | **0.167** |
| v1.1 coarse 0.30 | 0.467 | 0.332 | **0.350** |
| v1.1 coarse 0.40(§22) | 0.393 | 0.300 | **0.250** |
| v1.1 + 落座修复 | 0.400 | 0.297 | **0.272** |

**深度已经解决**(0.297 vs release 0.306),**残差 100% 在横向**:0.272 vs 0.167,**+10.5 cm**,
而且两边分布几乎不重叠(v1.1 最小 0.219 vs release 最大 0.250)。这正是 §17 记录的
"横向多走 11 cm",**速度修复没动它,落座修复也没动它** —— 落座修复只是把它从带噪声变成了确定性
(横向离散从 [0.19,0.309] 收到 [0.219,0.306])。

**因此要更正 §22/§23 里"Stage C 不需要"的说法**:brake **深度**确实不用改(depth 已对),
但 park 里还有 10.5 cm 的横向误差,它是现在 `stand infeasible` 与
`local alignment outside comfortable IK` 的直接来源。

### 24.5 下一步

approach 的最后 0.3 m 是**横走**:`mv = (aim_xy − pelvis_xy)` 朝规划站位,而 `facing` 固定在接近朝向,
所以 movement 向量相对朝向有横向分量,而 deploy 会把它的**模长归一化掉**,只留一个速度标量。
**§21 的速度表只量了直行**。所以下一件事是把同样的协议用**纯横向** movement 向量再量一次
(两 checkpoint,几个 band 速度),看 v1.1 的横向交付是不是显著低于前向 —— 那将直接解释这 10.5 cm。

## 25. 横向残差的真正来源:release 从来没执行过计划的站位方位角(2026-09-11, session cf)

§24.4 把残差定位到横向,并猜测"v1.1 横向交付弱"。**猜错了,而且错得有信息量。**

### 25.1 crab 测量:横向能力是**速度**的函数,不是 checkpoint 的

`diag_shuffle` 加了 `--crab`(movement 向量与**朝向**的夹角);协议上机器人始终沿 `free_heading`
选出的空旷方向行走、朝向反向转回,所以 crab 腿与直行腿有同样的避障净空。
第一轮只记了位移**模长**,读出"90° 比直行还快"这种不可能的结果 —— 模长分不清"横移了"和"转身直走了"。
补上 `fwd`/`lat`(起始朝向系分解)、`travel_err_deg`(实际行进方向 vs 命令方向)、`yaw_turned_deg` 后重测,
3 速度 × 3 角度 × 3 次,两 checkpoint(`outputs/diagnostics/gait_lateral2_20260911/`):

| 命令 | crab | release lat (m/s) | v1.1 lat (m/s) |
|---|---|---|---|
| 0.30 | 90° | **0.018 / 0.002 / 0.020** | 0.021 / 0.046 / 0.031 |
| 0.40 | 90° | **0.253 / 0.173 / 0.285** | **0.199 / 0.283 / 0.285** |

**两个 checkpoint 在同一命令上横向能力几乎相同**:0.40 都横移 0.17–0.29 m/s,0.30 都只有 0.002–0.046。
`travel_err` 在 90° 上两边都是 +9…+18°,`yaw_turned` 都只有 +11…+15°,**机器人确实在横移,不是转身直走**。
所以"v1.1 横向弱"是错的;真正发生的是 **Stage B 把 v1.1 的 COARSE 移到了 0.40,而横向通道正好在
0.30→0.40 之间打开**,于是 v1.1 的接近段第一次拥有了可用的横向通道。

### 25.2 于是 release 一直在少走计划的横向

case bank 自己记着计划站位(`rel`,骨盆系下的物体位置)。visible expert 的站位**只是抓取高度的函数**
(`sample_stand` 的 visible 分支不抽样),所以十个 case 的计划方位角**完全相同:37.4°**,plan_r 0.324。

| 臂 | 到位 − 计划 | 到位方位角中位 |
|---|---|---|
| release | **−8.9°** | 31.1°(逐局 13–45) |
| v1.1 coarse 0.30 | +5.2° | 40.9° |
| v1.1 coarse 0.40 | +2.5° | 40.0° |
| v1.1 + 落座修复 | **+5.4°** | 42.8°(逐局 36–46) |

**release 系统性地少走 9°,落在 ~28–31°;v1.1 忠实执行计划,落在 40–43°。**

### 25.3 而下游是按 release **实际到位**标定的,不是按计划

在 release 自己那 20 局上,方位角直接预测成败:

| 到位方位角 | release | v1.1 coarse 0.40 | v1.1 + 落座修复 |
|---|---|---|---|
| [0°, 30°) | **5/7** | — | — |
| [30°, 38°) | 3/6 | 2/2 | 0/2 |
| [38°, 50°) | **1/6** | 1/6 | 1/7 |

release 成功局方位角中位 **28.5°**、失败局 **35.5°**。
所以**计划的 37.4° 本身就压在失败带的边上**,这条线能work,靠的是 release 少走 9° 把它带回舒适区。
换句话说:**这个计划站位从来没有被真正执行过,下游(reach 包络、IK 舒适区、`stand infeasible`)
标定的是 release 交付的 ~29°,不是计划的 37.4°。**

### 25.4 一处死配置

`ReachConfig.azimuth_deg`(profile 里写 42.0)**没有任何消费者** —— 只在 `reach_config.py` 里声明并做
`0 < az < 90` 校验,`src/locomani` 里再没有第二处读它。站位方位角来自 `templates()` 的几何。
我在诊断中途据此说过"配置要求 42°",**那是错的,已撤回**。(顺带:这类"声明了、校验了、没人读"的字段
正是 `lint_env` 那类检查覆盖不到的洞。)

### 25.5 结论与下一步(是一次设计决定,未做)

修法不在执行侧,而在**计划侧**:把计划站位方位角移到这条线的真实工作点(~29°),
也就是把站位横向拉回约 5.5 cm(在 depth 0.257 处,29° 对应横向 0.142 而不是 0.197)。
今天没有这个旋钮 —— `sample_stand` 的 visible 分支只按模板中心减 `backshift`,
而 `backshift` 只沿深度方向移(加大它会把 plan_r 推过 `post_max = 0.40`)。
approach 里的 `appr_lat` 正是这个杠杆,但它在 refactor step 2 已被退役为 0.0。
**这会改变采集到的到位分布,所以是设计决定,不是标定** —— 证据齐了,等决定。

## 26. 把计划站位方位角改到 29°:到位角随之下降,失败模式对应消失(2026-09-11, session cf)

§25 的结论落地。**一次作废、一次成立**,两次都记在这里。

### 26.1 第一次作废:配置被路径绕过

新增 `ReachConfig.stand_azimuth_deg`(默认 **0 = 不改动**,所以仓库里 7 个 profile 全部不受影响),
在 `sample_stand` 的 visible 分支里把计划站位**保持 plan_r 旋转**到指定角。
没有复用死字段 `azimuth_deg`:7 个 profile 都把它写成 42.0,一旦赋予含义会把所有计划旋进失败带。

然后带 `--cases` 跑了 10 局,得 1/10、到位角 41.4° —— **作废**:`round_loop.py:99` 在固定 case bank 下
直接取 `can_xy / stand / rel`,**根本不调 `sample_stand`**,所以那个字段一次都没被读到。
配置写了、加载了、校验通过,而那条路径不读它 —— 又一次"配置半应用"。
(副产品:它成了"计划 37.5°"的第二个样本,与 §24 的落座修复轮 **1/10 与 1/10、42.8° 与 41.4°** 重复性良好。)

### 26.2 正确做法:只旋转 bank 里的计划站位

从同样 10 个 case 派生新 bank,**只把 `rel` 旋到 29° 并据此重算 `stand = can_xy − R(psi)·rel`**;
`can_xy / psi / spawn_off / spawn_lat / dpsi / appearance` 逐字段校验完全相同。计划站位平移 **4.8 cm**。

| 臂 | 计划 | **到位方位角** | depth | lateral | park | ok |
|---|---|---|---|---|---|---|
| release | 37.5 | 28.5 [13,39] | 0.306 | 0.167 | 0.349 | **4/10** |
| v1.1 + 落座修复 | 37.5 | 42.8 [36,46] | 0.297 | 0.272 | 0.400 | 1/10 |
| v1.1(作废轮) | 37.5 | 41.4 [34,50] | 0.298 | 0.257 | 0.388 | 1/10 |
| **v1.1,计划 29** | **29.0** | **32.0 [25,47]** | 0.299 | **0.189** | **0.350** | **5/10** |

- 到位角 **42.8 → 32.0°**,跟随 −8.5° 的计划改动(残留 +3° 偏差,与 §25 测到的 +2.5…+5.4° 一致);
- lateral **0.272 → 0.189**(release 0.167),park **0.400 → 0.350**(release 0.349),**depth 不变**(0.299);
- 失败模式按预测消失:`stand infeasible` **4 → 2**,`local alignment outside comfortable IK` **2 → 0**,
  `coupled safe-hand IK did not track` **1 → 0**;
- 新出现 1 局 `[hold] object slipped 90.4 cm in the palm frame` —— 那是**抓到之后**的阶段,属于往前推进。

配对检验:vs 落座修复轮 **4:0(exact p=0.125)**,vs 作废轮同样 4:0(p=0.125),vs release 2:1(p=1.000)。
按 §23.3,**10 局仍然测不出率**;成立的证据是遥测(方位角、lateral、park、失败模式),
它们全部按 §25 预测的方向移动,而 depth 作为对照没动。

### 26.3 这解释了 release 一直以来为什么能work

FK 模板的舒适抓取中心在 **30.3°**;`backshift = 0.08` **只沿深度**施加,把计划旋到 37.5°;
release 因为 COARSE 速度 0.30 上横向通道几乎是死的(§25.1),**从未执行那 7° 旋转**,实际落在 28.5°。
v1.1 在 COARSE 0.40 上横向通道是活的,忠实执行了 37.5°,于是落进失败带。
把计划改回 29° 之后,v1.1 落在 32°,与 release 的工作点重合,成功率也回到同一水平。

**所以这不是 v1.1 的缺陷,是一处一直存在、被 release 的执行缺陷掩盖的计划几何错误。**
更干净的长期修法是让 `backshift` **沿径向**施加(保持方位角):那样计划自动回到模板自己的 30.3°,
不需要额外旋钮 —— 但它会同时改变 plan_r(0.324 → 0.310),属于另一次单变量改动,未做。

## 27. 20 局配对确认:v1.1 从 0/20 回到 release 水平(2026-09-11, session cf)

§26 的 10 局证据只是遥测方向对;这一节是有功效的配对确认。同一批 20 个 case,
bank 只旋转 `rel`、重算 `stand`,其余字段逐字段相同。

| 臂 | ok | 到位方位角 | depth | lateral | park |
|---|---|---|---|---|---|
| release,计划 37.5 | **9/20** | 31.1 [13,45] | 0.307 | 0.204 | 0.369 |
| v1.1 旧(无任何修复) | **0/20** | 40.9 [30,49] | 0.332 | 0.305 | 0.467 |
| **v1.1 全部修复 + 计划 29** | **7/20** | 35.2 [25,40] | 0.298 | **0.207** | **0.363** |

- vs v1.1 旧:**7:0,exact McNemar p = 0.0156**;
- vs release:4:6,**p = 0.754** —— 统计上分不开。

lateral(0.207 vs 0.204)与 park(0.363 vs 0.369)与 release 重合,方位角离散从 [30,49] 收到 [25,40]。
"全部修复"= `APPR_COARSE` 0.40(§22)+ 落座修复(§24)+ 旋转容差 0.35(§23)+ 计划角 29°(§26)。

**所以 §20.1 那条"v1.1 不能直接换进采集线"到此解除**:它当时是对的(0/20),
原因不是 checkpoint 差,而是四处按 release 标定/掩盖的东西 —— 其中两处(落座、计划角)是
**与 checkpoint 无关的既有缺陷**,只是被 release 的执行缺陷掩盖着。

### 27.1 剩下的失败,以及一处新出现的

`stand infeasible` 6/20(release 8/20,更少)、`local alignment disturbed target or base` **5/20**、
turn 1、`[hold] object slipped` 1。**`disturbed` 是新的**:release 那 20 局里 0 次。
它发生在 alignment 阶段(抓取前的最后对齐),意思是对齐过程中物体或底盘被扰动。
这是下一条线索,但注意它出现在**能走到 alignment 的轮次变多之后** —— 属于阶段推进,不能直接当退化读。

### 27.2 撤回 §26.3 的"更干净的长期修法"

§26.3 提议把 `backshift` 改成沿径向施加。**错的,已撤回**:`project_squat`
(`coupled_grasp_policy.py:82-84`)把下蹲后退建模为 `backshift * forward`,是**沿骨盆前向的纯深度后退**,
所以用深度位移补偿它量纲正确;改成径向会凭空引入 `backshift·sin(az) ≈ 4 cm` 的横向误差。

正确的结构是:**计划方位角是"下蹲前"的量,而手臂舒适性是"下蹲后"的属性**。
按实测 0.089 m 后退换算到下蹲后:release 28.5→**22.9°**,v1.1 计划37.5 42.8→**35.2°**,
v1.1 计划29 32.0→**26.0°**。成功集中在下蹲后 23–26°,失败在 35°。
`stand_azimuth_deg` 因此是个"下蹲前"旋钮 —— 它有效,但更本质的标定量是下蹲后方位角,
把它显式化(按 `project_squat` 的预测把计划表达在下蹲后坐标里)是比径向 backshift 更该做的下一步。

## 28. Stage D 的腕交付率:两次测量互相矛盾,判定为工具问题,常量不动(2026-09-11, session cf)

`TRACK_ROT_BOOST = 4.5` 是 release 上"105° 命令、23° 交付"(22%)的倒数,作用在
`primitives_crouch.py:230`(把掌姿态误差放大后作为腕四元数发出)。§20.4 把它列为待重测项。
移植回 `diag_wrist.py`(refactor step 2 删除),并补上它原本没有的量:
每个命令的 **asked / delivered / ratio**(以 identity 档为参考系)与各轴 ratio 的极差。

**第一轮(无逐 case 复位)**:v1.1 读出 0.76 / 0.26 / 0.33 / 0.36(中位 0.34,极差 2.9 倍);
release 读出三个不同命令"交付"同一个 131.7°、掌心停在同一点且比 identity 退后 0.40 m ——
sim 日志证实 `Robot has fallen, height: 0.199 m`。**release 那臂无效**,而工具照样打印了数字。

**第二轮(补了逐 case 完整复位 + 持命令时的 fallen 检查)**:两臂**都**读出 ratio ≈ 0.01,
掌轴变化 0.013 / 0.004(远低于脚本自带的 0.05 "字段是死的"阈值),而且四个 case 的掌心位置**完全相同** ——
连位置都没生效。**在 release 上复现不出已知为真的 22%,所以是工具问题,不是机器人。**
差别只在逐 case 复位,说明复位清掉了这条通道的前置状态(stream mode / wrist target)。

**判定**:交付率**未测得**。`TRACK_ROT_BOOST` **保持 4.5 不动**,继续带 §22.1 的标注
(release 时代标定、v1.1 上未验证)。下一步是先确定 `set_wrists` 在哪个 stream mode 下被消费
(`agent_server.py:293` 打包 wrist pos/quat,默认 `PLANNER_VR_3PT`),让诊断在复位后显式进入该模式,
再重测。**在那之前不要改这个常量** —— 它的失败模式是积分器极限环(release 上实测过从 4.1 cm 发散到 27 cm)。

方法上值得记一笔:这是今天第三次"检查器报告的不是世界的状态"(前两次是 §25.1 的模长掩盖方向、
`ports.require_exporter` 式的过窄探针)。这次是**同一工具的两次运行互相矛盾**,而能判定谁错
只因为 release 上有一个**已知为真的先验值**可以对表 —— 没有那个先验,两轮都会被当成测量结果。
