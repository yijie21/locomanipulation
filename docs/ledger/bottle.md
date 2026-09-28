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

**结论**:R3 证明病在数据形状(同样 200 局翻倍);R4 证明靶向 DAgger 有效(+29:−9)。但 2026-08-28 发现这些局里约 2/3 在 VLA 接管前已被脚本走到桌边(stand-start gap),所以它们是「混合任务」数字;修复后同一 R4 模型的全任务干净基线是 12/100(该臂目录已不在,来源 docs/single_arm_design.md:504)。

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

这一批是排除法,大多数臂只有 20 局或没有 manifest,按'只收可重算'的规则不列成功率,只记结论(来源 docs/single_arm_design.md:571–760):开环 denoising 步数 4 即够(模型开环只比噪声地板高 2%);观测延迟 33 ms 不是因素;replan 1.6 s 显著更差;grip-gain 3.0 无差异(爪子不是瓶颈);200 局随机出生扫描证伪'停位窄甜区',真实模式是 0.50 m 悬崖(≥0.50 m 成功 5%,<0.50 m 55%);初始姿态 token 换成数据 medoid(关节误差 23°→3.4°)。

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
  `docs/reference/runbook.md` 和 `docs/single_arm_design.md` 里"当前正式基线"的说法已加限定。

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
| 文档 | `docs/setup.md`(下载 `--sonic-v1-1` + 验证)、`docs/single_arm_design.md`(链路图的模型路径)、`docs/single_arm_design.md`(`motion_token` 行)、`docs/reference/silent_failures.md`(env 表 + 三条新静默失败行) |

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

## 29. Stage E:在 v1.1 上重录 bottle gate 的 baseline(2026-09-11, session cf)

`tests/gate/baseline_bottle.json` 原本的 source 是 `restart_ab_20260910` —— release 时代的一次采集,
里面每个量(站位中位、各阶段时长、失败原因集合)都是 release 的。v1.1 成为默认之后它已不能当回归判据,
而 gate 的规则里"失败原因集合不得新增成员"会让每次候选运行都因为新串直接 FAIL。

**前提**:先把 §26/§27 的结论落进仓库 profile(`ready_grasp_visible.json` 与 `bottle_visible.json` 的
reach 块加 `stand_azimuth_deg: 29.0`;其余 6 个 ready_grasp profile 保持 0.0 —— 未测过的线不跟着动)。
**在这之前录 baseline 会把 0/20 的配置固化成基准。**

**录法**:刻意用**采集式批次**(采样 case、不用固定 bank、用仓库 profile),30 局 ——
与 release 那份同样大小。先录了一份 10 局的,随即作废重录:gate 的 yield 容差是 ±0.15 绝对值,
而 n=10 时 p≈0.3 的二项标准差就有 0.145,候选运行靠随机就能越界。
旧 baseline 另存为 `tests/gate/baseline_bottle_release_20260910.json`。

| | release baseline(30) | v1.1 baseline(30) |
|---|---|---|
| yield | **0.600** | **0.233** |
| end_depth_m | 0.3011 | 0.2998 |
| squat_drop_m | 0.13 | 0.13 |
| seconds | 33.7 | 19.95 |
| exit_majority | visible brake depth | 同 |

自比较 `gate.py compare` **PASS 15/15**(记录/比较路径本身正常)。

**两份 baseline 不可互比**:采样分布与 profile 都不同(记忆 `two-eval-calibrations-in-the-ledger`、
`站位采样带在 2026-09-07 变过` 是同一类陷阱)。0.600 vs 0.233 **不是** checkpoint 的对照;
可比的对照是 §27 的同 case bank 配对(7/20 vs release 9/20,p=0.754)。
可比的是两边都稳定的几何量:`end_depth_m` 0.3011 vs 0.2998、`squat_drop_m` 两边都是 0.13、
`exit_majority` 相同 —— 与 §22/§24 的测量吻合。

### 29.1 一条 OPEN:采样批次出现了固定 bank 批次没有的失败

v1.1 采样批次独有:`[grasp] baseline clearance: right_dexN_finger_N_link vs dining_table_*`(手指对**桌子**)、
`coupled safe-hand path blocked: ...finger...`、`coupled transport disturbed target or base`、
`coupled safe-hand IK did not track`、`local alignment timed out`。
而 §27 那 20 局固定 bank(同样 plan 29)**一次都没出现桌面净空类失败**。

两种解释未分辨:① 29° 的站位更"正对"物体,手臂越过桌沿更多,净空变紧;
② 采样分布(`--edge-jitter 0.20`、随机 spawn)比固定 bank 更宽,只是覆盖到了更难的配置。
**区分方法**:把这 30 局里触发净空失败的 case 固定成 bank,在 plan 37.5 与 plan 29 下各跑一次配对。
在分辨之前,不要把 0.233 归因于 plan 29。

## 30. 方位角改动被自己的对照否掉,并更正 §27 的归因(2026-09-11, session cf)

§29.1 留的那个 OPEN,用同 30 个 case 的配对跑完了。**结论与 §26/§27 相反,并且 §27 的归因是错的。**

case 来自 gate baseline 那 30 局的 manifest 反解重建(`rel` 由 `stand = can_xy − R(psi)·rel` 反解,
验证重现 29.00° / plan_r 0.3240 全 30 局),两个 bank 只差 `rel` 与据此重算的 `stand`(计划站位差 4.75 cm),
其余字段逐字段相同。**37.4 先跑、29 后跑** —— 如果机器状态在退化,吃亏的是 29,也就是被辩护的那一方。

| 同 30 case 配对 | plan 37.4 | plan 29.0 |
|---|---|---|
| ok | **7/30** | **3/30** |
| 到位方位角 | 38.3 [30,49] | 35.2 [18,46] |
| depth / lateral / park | 0.299 / 0.233 / 0.377 | 0.297 / 0.206 / 0.355 |
| **净空 / 路径阻塞** | **0/30** | **3/30** |
| **`[hold] object slipped`** | **0** | **4** |
| `stand infeasible` | 11 | 11 |

不一致格 **1:5**,exact McNemar **p = 0.2188**。方位角遥测照旧跟随计划(35.2 vs 38.3),
lateral / park 也跟随 —— **执行侧没问题,是这个计划角本身不更好。**
§29.1 的净空假设在方向上成立:29° 更正对物体,净空失败 0/30 → 3/30,另外多出 4 局"抓到后又丢"。

### 30.1 §27 的 7/20 vs 0/20 不是方位角的功劳

§27 比的是"**v1.1 + 全部四项修复 + plan 29**" vs "**v1.1 + 零修复 + plan 37.5**" ——
它把速度表(§22)、落座修复(§24)、旋转容差(§23)和方位角**捆在一起**。
那个 p=0.0156 证明的是"**这四项加起来把这条线从 0/20 救回 7/20**",**不是**方位角的效应。
**我把它写成了方位角的功劳,这是归因错误,在此更正。**

真正的单变量方位角测试只有两个(同代码、同其余配置,只改计划角):

| 实验 | case 来源 | 不一致格(29-only : 37.5-only) | p |
|---|---|---|---|
| §26,10 局 | cases20 派生 | 4 : 0 | 0.125 |
| §30,30 局 | 今天新采样重建 | 1 : 5 | 0.219 |
| **合并** | | **5 : 5** | **1.0** |

**方位角改动:无效应。** 两次方向相反、各自不显著,合并后完全无结论 ——
而 §26 那次之所以看起来强,是因为 10 局的功效本来就分不开(§23.3 已经写过这条,我又犯了一次)。

### 30.2 处置

`stand_azimuth_deg` 在 `ready_grasp_visible.json` 与 `bottle_visible.json` **回退为 0.0**
(= 保持模板几何,历史行为)。旋钮本身保留:它是有效的、被测试覆盖、默认不改动任何 profile,
而且这两次实验都靠它才能做成单变量。**不保留的是那个值。**

**今天仍然站得住的**(都是单变量 + 遥测,不依赖成功率):
`APPR_COARSE` 0.40(§22:`stalled` 5/10→0/10,park 0.467→0.393)、
落座修复(§24:复位后第一秒位移中位 0.099→0.034、最大 0.489→0.083、`[stand]` 阶段失败归零)、
旋转容差 0.35(§23:机制牢固,效应 n=2 未验证)。
**"v1.1 不能进采集线"仍然解除** —— §27 的 7/20 vs 0/20 依然有效,只是功劳属于这一整组修复。

**新的 OPEN**:为什么同一个到位方位角(35.2°)在 cases20 派生那批上给 7/20、在今天这批上给 3/30?
到位角单独不能跨 case 集预测成败,说明还有别的摆位变量(桌沿余量、物体沿边抖动)在起作用 ——
下一步该找的是那个变量,而不是继续调方位角。

## 31. 找那个变量:depth 不是,`spawn_lat` 只在 release 上是,v1.1 上是终端 IK 的**伸不到**(2026-09-12, session cf)

§30.2 留的问题:同一个到位方位角在两套 case 上给出 7/20 与 3/30,所以方位角不是那个变量。
这一节**没有靠猜特征**,而是先读主导失败的判据,再挖它自己记下的拒绝原因。

### 31.1 主导失败的内部原因:`coupled_terminal_ik`

`visible expert: stand infeasible` 的上游是 `assess()`;coupled 线的预测检查把每次拒绝的原因
记进 `execution.metrics.prediction_rejections`。挖出来每条臂都是同一个,**包括 release**:

| 臂 | stand-infeasible 轮数 | `coupled_terminal_ik` | `coupled_terminal_capture` |
|---|---|---|---|
| az374(plan 37.4) | 11 | **53** | 2 |
| az29(plan 29.0) | 11 | **46** | 9 |
| gate30(plan 29) | 11 | **51** | 4 |
| stagef(plan 29) | 6 | **24** | 6 |
| release | 8 | **35** | 5 |

再用它记的 `coupled_prediction_ik_errors`(pe / re / drop / 目标点)拆开
(容差 `IK_POSITION_TOL = 0.020`、`IK_ROTATION_TOL = 0.30`):

| 臂 | 位置误差中位 | 超位置容差 | 旋转误差中位 | 超旋转容差 | 目标 |xy| |
|---|---|---|---|---|---|
| az374 | **0.0404** | **100%** | 0.0929 | 19% | 0.357 |
| az29 | **0.0432** | **100%** | 0.1073 | 41% | 0.342 |
| release | **0.0564** | **100%** | 0.2366 | 74% | 0.361 |

**卡的是位置,是容差的 2–3 倍;旋转宽裕。**三条臂的失败目标点都在骨盆系 **|xy| ≈ 0.34–0.36 m**,
方位角各不相同(release x 0.234/y −0.237 vs az29 x 0.159/y −0.295)而**距离相同** ——
所以它是**伸不到**,不是站歪了。

### 31.2 候选变量的判别力(汇总 106 局 v1.1 / 19 局 release)

AUC = 随机取一成一败、失败那局的值更大的概率。

| 变量 | v1.1 成功中位 | 失败中位 | AUC | release AUC |
|---|---|---|---|---|
| `lateral` | 0.199 | 0.222 | 0.72 | 0.72 |
| `azimuth` | 33.7° | 36.7° | 0.72 | 0.66 |
| `park` | 0.357 | 0.374 | 0.70 | 0.74 |
| **`spawn_lat`** | 0.073 | 0.109 | 0.63 | **0.79** |
| `depth` | 0.300 | 0.297 | **0.42** | 0.66 |

**两个必须讲的结论**:

1. **depth 不是那个变量**(AUC 0.42)—— 刹车有效。而且 depth 被刹车钉住之后,
   `park` / `lateral` / `azimuth` **在代数上是同一个变量**(互为单调变换),
   所以 §25–30 我反复折腾的是一个**派生量**。
2. **那个 0.72 不能当因果读**:在配对的 30 局里方向是反的(az29 到位 lateral 更低 0.206、成绩更差 3/30)。
   跨臂汇总出现 Simpson 式反转,**这是汇总统计骗人的标准形状**。

### 31.3 `spawn_lat` 的链条只在 release 上成立

| |spawn_lat| | release ok / 到位 lateral | v1.1 ok / 到位 lateral |
|---|---|---|
| < 0.05 | **7/9**,0.167 | 6/24,0.228 |
| 0.05–0.12 | 1/6,0.213 | 5/27,0.222 |
| 0.12–0.25 | 1/4,0.274 | 6/36,0.214 |
| corr(|spawn_lat|, 到位 lateral) | **+0.49** | **−0.12** |

release 把出生横偏**原样带到到位**(COARSE 速度上横向通道是死的,§25.1),于是它成了支配变量,
居中出生 7/9;v1.1 **把它修掉了**(corr −0.12,三档到位 lateral 全在 0.21–0.23)——
这是横向通道在 0.40 上变活的正面效果,但 v1.1 的成功率在三档上**一律 ~20%**,
**所以限制 v1.1 的不是出生位置。**

### 31.4 最强的线索:终端 IK 排除了腰,而 v1.1 的腰是能用的

`assess_reach` 与 coupled 预测都用 `use_waist=False` 解终端 IK。把记录下来的目标点离线重解:

| | 离线 `use_waist=False` | 离线 `use_waist=True` |
|---|---|---|
| az374(n=59) | 位置中位 0.0110,25% 超限 | **0.0002,0% 超限** |
| az29(n=49) | 0.0095,35% | **0.0003,0%** |
| release(n=35) | 0.0139,31% | **0.0003,0%** |

**腰一参与,这些目标点的位置误差直接归零。** 而 §16 实测 v1.1 的腰跟踪是 release 的 3.3 倍
(waist RMSE 3.9° vs 12.9°)—— 也就是说 `use_waist=False` 在 release 上几乎不损失什么
(它本来也交付不了腰部前倾),在 v1.1 上**扔掉了真实可用的伸展量**。

**但这条还不能下结论,原因要写明**:我的离线重解**没能复现 live 的误差量级**(0.011 vs 0.040),
因为 live 的种子带着 `q[:12] = 实测腿部几何`,而 `use_waist=False` 把腰钉在种子值上 ——
这两样我无法从记录里重建。所以只能说"这些点在腰参与时绰绰有余",**不能说"腰能修好 live 的失败"**。
已补上缺的那一行:拒绝时同时记 `seed_q`(29 个关节)与 `use_waist`,一次 10 局就能精确对账。

### 31.5 下一步

1. 跑一次 10 局带新 `seed_q` 的采集,离线用**真种子**重解 `use_waist=False/True` —— 如果腰仍然把 100% 的
   拒绝救回容差内,那么改动就是让预测端(以及执行端)允许腰参与,并且这条改动**对 v1.1 的收益应当大于 release**。
2. 注意这不是免费的:腰参与会改变掌姿态与扫掠路径,`screen_motion` 与 §23 的旋转容差都要重新看。
3. release 那条线的独立结论可以直接用:**把 `--spawn-lateral` 收窄能显著提高 release 的产出**(7/9 vs 2/10),
   但它对 v1.1 无效 —— 两条线的瓶颈已经不同,采集参数不该再共用一套。

## 32. 下蹲的后撤量被低估 3.5 cm,腰在**活体**里救回 73% 的拒绝,而离线重解不可用(2026-09-12, session cf)

三批 10 局,同 10 个摆放(从 `az_ab30_20260911/az374` 的 bank 前十派生,即 29° 撤回后当前代码产生的 plan 37.4),
`policy/sonic_v1_1`,instance 0。30 局 0 成功 —— 按 §23 这**不是**可读的率,本条的结论全部来自逐局遥测。
脚本 `scripts/experiments/retreat_seed10_20260912.sh`(三批同一脚本),报告 `scripts/experiments/retreat_report_20260912.py`。

### 32.1 新增的那一行遥测:下蹲实际后撤了多少

`project_squat` 把下蹲预测成 `pelvis -= [0.8*(drop-0.015)*forward, drop-0.015]`,也就是**后撤 0.092 m**。
这个比例是 release 时代标的,而**从来没有任何一局记录过交付的后撤量** —— 循环只记 `pelvis_world` 采样,
竖直方向记 `delivered_drop_m`,水平一个字没有。现在 `coupled_transport[].squat_delivered_pose` 直接记
(前蹲坐标系下的 retreat / lateral / drop / pitch / yaw + 当次的预测值)。

| 量 | 实测(n=12 次下蹲) | 预测 |
|---|---|---|
| 后撤 | 中位 **0.1262 m** | 0.0920 m |
| 后撤/下沉 比 | 中位 **1.150** [0.91, 1.28] | 0.800 |
| 低估量 | 中位 **+0.0347 m**,**12/12 全为正** | — |
| 盆骨 pitch(底部) | 中位 **+16.5°** | **0**(`project_squat` 不改 quat) |

**方向很重要:预测是乐观的。**实际落点比批准的计划**更远离物体 3.5 cm**,而且姿态还多了 16.5° 的前倾。
所以 `assess` 批准的抓取,执行时并不是它批准的那一个 —— 这解释了为什么下游必须有 `local_align` 去伺服,
也解释了 `local alignment outside comfortable IK` / `timed out` 这一族失败。
**注意推论方向**:把比例改对会让预测的目标点**更远**,于是 `coupled_terminal_ik` 会拒绝得**更多**,不是更少。
今天的 0.8 是"让计划通得过"的那个乐观偏置。真正的修法在站位或下蹲深度,不是单改这个常量。

### 32.2 腰:这次是在活体里量的,73%

`assess` 与 `solve_transport_ik` 都用 `use_waist=False`。新增**影子解**:拒绝发生时,用**同一个目标、同一个
种子、同一个旋转目标**再解一次 `use_waist=True`,只记录、不使用(计划照旧拒绝)。

| | `use_waist=False` | `use_waist=True` |
|---|---|---|
| 位置误差中位 | 0.0529 m | **0.0019 m** |
| 超 `IK_POSITION_TOL=0.020` | **49/49** | **13/49** |
| 旋转误差中位 | 0.1101 rad | 0.0186 rad |
| 两个容差都进 | 0 | **36/49 = 73%** |

救不回来的 13 条落在 |xy| **0.402–0.468**(被救的中位 0.378)—— 干净地按距离分开,
即"腰能补一段,但再远就是真的伸不到"。结合记忆 `sonic-waist-lean-is-checkpoint-bound`(v1.1 跟踪 73% 的腰倾、
release 只有 28%),**这条改动在 v1.1 上比在 release 上更有本钱** —— 但那是预期,不是测量。
§31.5 的代价提醒仍然成立:腰参与会改掌姿态与扫掠路径,`screen_motion` 与 §23 的旋转容差要重看。

### 32.3 一条方法论结论:**离线重解 IK 不能当证据**

本条的两个结论都来自活体,原因是离线路径被证伪了:

- **活体是确定性的**:同一进程里用**完全相同的实参**再解一次,`|Δpe| = |Δre| = 0.00e+00`(n=16),
  且 `seed_mutated = 0`(解算器没有改种子)。
- **离线不复现**:用记录下来的 `target_root` / `seed_q` / `target_rot_local` 重解,得到 **0.068 / 0.010**,
  而活体报 **0.115 / 0.299** —— 离线在**两个目标上同时更好**,这不是收敛解能有的差。
- 已排除:目标点(从 world 反算逐位相符,Δ=0)、旋转目标(与模板旋转差 3e-15,确认就是模板)、
  种子(解前快照,未被改)、模块重复(`locomani.primitives.arm_ik` 解析到同一文件)、
  场景 XML(同一 `LOCOMANI_SIM_DIR`,mtime 未变)、迭代数/阻尼/rot_weight(扫过,无组合能同时命中)。
- **因此 §31.4 那条"离线重解后位置误差 0.0002 m、0% 超限"的线索作废**,由 32.2 的活体数字取代
  (0.0019 m、13/49 仍超)。差异原因**未找到**:在找到之前,这个仓库里任何离线 IK 重解都不能作为证据。

### 32.4 一处自我更正

我在本次诊断中途用"逐局样本对 dz 的过原点回归"重建后撤比,得到 release 1.15 / v1.1 1.43(AUC 0.945)。
**那个指标有偏**:后撤领先于下沉,过原点斜率系统性高于终点比值(直接测得 v1.1 是 1.150)。
两臂用的是同一个有偏指标,所以"v1.1 后撤多于 release"这个**相对**判断还没被推翻,但**绝对值作废**,
而且 release 没有直测 —— 要比 checkpoint,得在 release 上跑同样的遥测。

### 32.5 下一步

1. **release 臂跑同样 10 局**(同 bank),拿到直测的后撤比与影子解 —— 这才能说这两件事是不是 v1.1 特有的。
2. 腰的修法做成**设计决定**:预测端与执行端同时允许腰参与,然后重看 `screen_motion` 与旋转容差;
   先跑一次 30 局配对再决定是否进默认(§23 的教训:10 局测不出率)。
3. 后撤低估的修法**不是**改 0.8:先测"如果把预测改成实测 1.15,有多少计划会被拒"(离线算得,不需要跑),
   再决定是补偿(站位拉近/减小 drop)还是让 `local_align` 继续承担。
4. 离线/活体 IK 不一致的根因仍然悬着,优先级高于上面任何一条 —— 它决定哪些历史结论站得住。

### 32.6 与 crate 线同日测量的对账:一半是好消息,一半是对 32.2 的重要限定

crate session 在同一天(01:28-01:35 UTC)恢复了 `diag_lean.py` 并在两个 checkpoint 上量了腰倾交付率
(STATE / 台账 `crate.md` 2026-09-12):**planner 通道 release 0.09、v1.1 0.87**(streamed 0.10 / 0.94)。

**好消息**:这把 32.2 从"预期"提升为有独立测量支撑 —— 在 v1.1 上计划端让腰参与是**真能兑现的**
(87–94%),在 release 上几乎兑现不了(9–10%)。记忆里那条 "73% vs 28%" 被这次更干净的测量取代。

**限定一(必须先解决,才能动这条改动)**:`arm_ik.ArmIK.__init__` 把**索引 13/14(waist_roll / waist_pitch)
的关节上下限乘了 1.6**(注释:参考可以过冲,机器人自己会饱和)。而 crate 的测量说 **v1.1 在 0.52 命令下
到 25.9°,MJCF 的 `waist_pitch` 限位是 29.79°,绑定约束已经是硬件行程**。也就是说 32.2 的影子解**被允许
用到 1.6× 限位(约 47.7°)的腰角**,那部分解在机器人上根本做不出来。
**所以 36/49 这个数字是上界**,它里面有多少依赖超限的腰角,现在不知道 —— 影子解没记解出来的腰角。
**下一次必须记**:影子解的 `q[12:15]`,以及一条把腰限制在 MJCF 范围内的对照解。

**限定二**:crate 的 FK 分解说纯腰倾"给高度、**不给伸展**",在他们的姿态下甚至**损失 2.2 cm 前伸**
(手掌挂在腰轴下方)。这与 32.2 不直接矛盾 —— 他们量的是固定手臂姿态下的纯腰倾,IK 是腰与手臂**联合**
求解去够一个点,是两个问题 —— 但在动手之前必须把这两者调和:如果联合解的前伸增益主要来自腰角超限,
那 32.2 就是个假阳性。

**限定三**:`solve_transport_ik`(执行端)也是 `use_waist=False`。计划端单独放开会让计划与执行不一致,
这正是 §32.1 那类"批准的不是执行的"错误的来源。两端要一起改,或者都不改。

## 33. release 臂同 bank 10 局:0.8 这个常量两边都错,但 v1.1 错得多一倍,腰角不超限(2026-09-12, session cf)

`SONIC_POLICY=policy/release`,同一个 bank(`cases10_az374.json`),同脚本,instance 0,
`outputs/experiments/retreat_seed10_release_20260912`(stack.log 的 `policy:` 行已确认是 release,§21.5 的坑)。

| 量 | v1.1(30 局 / 12 次下蹲 / 49 条拒绝) | release(10 局 / 4 次下蹲 / 23 条拒绝) |
|---|---|---|
| 成功 | **0/30** | **4/10** |
| 后撤 | 0.1267 m | 0.1114 m |
| 后撤/下沉 比 | **1.150** | **1.003** |
| 对 0.8 预测的低估 | **+0.0347 m** | **+0.0194 m** |
| 底部盆骨 pitch | **+16.5°** | +10.8° |
| 终端目标 \|xy\| | **0.3777** | 0.3425 |
| 影子解救回 | 36/49 = **73%** | **23/23 = 100%** |

### 33.1 四个量指向同一件事:v1.1 在同一个计划站位上要求更多的伸展

比值 1.003 vs 1.150、低估 1.94 cm vs 3.47 cm、pitch 10.8° vs 16.5°、目标 \|xy\| 0.3425 vs 0.3777。
**所以 0.8 这个常量在两个 checkpoint 上都是错的(release 实际后撤≈下沉量,比值 1.0),
但其中约 1.5 cm 是 v1.1 特有的**,另外约 2 cm 是一个一直都错的陈旧常量。
pitch 的差与记忆 `sonic-waist-lean-is-checkpoint-bound`(v1.1 真的交付腰倾)一致。

### 33.2 逐局配对:4 : 0,而且这和 §27 不一致

| case | release | v1.1 ×3 |
|---|---|---|
| az30:1 / 3 / 6 / 7 | **ok** | x x x |
| 其余 6 个 | x | x x x |

不一致对:release-only **4**,v1.1-only **0**。单批精确 McNemar p=0.125(n=10 不够),
但 4 个 release 赢的 case 在 v1.1 的三批里 **3/3 全输**,方向没有歧义。
**这与 §27(同 case 配对 7/20 vs 9/20,p=0.754)相反** —— 说明 checkpoint 的差**依赖摆放**:
目标点落在伸展极限附近时 v1.1 输,落在舒适区时两边一样。
这正是 §25.3 的机制(下游按 release **少走**的到位角标定),现在有了配对数字;
但注意 §30 已经证伪"把计划角改到 29°"这个修法 —— release 的优势不只来自方位角。

### 33.3 腰角不超限(对 §32.6 限定一的部分回答)

新记的 `waist_q_rad`:release 的影子解用到的 **waist_pitch 中位 5.7°、最大 11.5°,
0/23 超过 MJCF 的 29.79° 限位**(roll 最大 8.6°)。也就是说腰的救援用的是**温和且机器人做得出**的角度,
不是靠 `ArmIK` 那 1.6× 放宽。**v1.1 那 49 条拒绝是在这个字段加进去之前跑的**,所以 v1.1 侧仍缺一次确认
(一次 10 局即可)。§32.6 的限定二(纯腰倾损失前伸)与限定三(执行端也关着腰)不受影响。

### 33.4 下一步

1. v1.1 再跑 10 局**只为拿 `waist_q_rad`**,确认 36/49 不依赖超限腰角。
2. 然后腰是一次设计决定:计划端与执行端一起放开,30 局配对(§23:10 局测不出率)。
3. 后撤常量:0.8 → 1.0(release)/1.15(v1.1)是**实测**,但直接改会让计划更难通过(§32.1),
   所以先离线算"改对之后有多少计划被拒",再决定补偿方式。
4. 一个新的事实可以直接用:**这条 bank 上 release 仍然明显强于 v1.1**。在腰的修法落地之前,
   要产量就用 release 采集;要推进 v1.1 就继续按 §32/§33 修伸展,不要再调方位角。

## 34. 修好 §28 的仪器之后:planner 模式下腕通道不是伺服目标,两个 checkpoint 都一样(2026-09-12, session cf)

`scripts/experiments/wrist_channel_20260912.sh`,两臂各一次,instance 0,
`outputs/diagnostics/wrist_channel_20260912/`。

### 34.1 §28 为什么测不出来:它没有"已知会动"的那一条腿

旧版每个 case 命令**同一个腕位置**,并且把"交付的旋转"定义为**相对第一个 case**。
通道若是死的,这个参照系就是静止的 rest 姿态本身,于是每个 case 都报 ~0 交付、ratio 0.01 ——
正是第二轮在两个 checkpoint 上打印出来的东西(包括那个**先验已知为 22%** 的 release)。
**仪器分不清"字段是死的"和"旋转被忽略",因为它没有任何一条腿是已知会动的。**

新版加了三段,每段是下一段的前提:

1. **位置(闸门)**:`rest` 与 `reach` 两条腿,**旋转都是 identity**。掌心若不动,通道就是死的,
   后面的旋转数字没有意义 —— 脚本直接打印判决并停止,不再输出 ratio。
2. **deploy 自己的参考**:`RefTap` 读 g1_debug 的 `body_q_target`,把"控制器从没让手臂动"
   与"让了但手臂没跟上"分开 —— §28 也分不开这两件事。
3. **旋转**,参照系是 `reach` 腿(同位置、identity 旋转),这才是旋转问题唯一正确的对照。

### 34.2 结果:命令 0.206 m,掌心动 0.002–0.003 m,两个 checkpoint 一样

| | 命令位移 | 掌心位移 | ratio | 到命令点残差 | deploy 参考变化 |
|---|---|---|---|---|---|
| sonic_v1_1 | 0.206 m | **0.003 m** | **0.017** | 0.28 m | **0.0000 rad** |
| release | 0.206 m | **0.002 m** | **0.011** | 0.28 m | **0.0000 rad** |

**而 deploy 确实收到了。**`deploy.log` 三处独立证据:
编码器观测表里 `10. vr_3point_local_target (9D) [offset: 901]`、`11. vr_3point_local_orn_target (12D) [offset: 910]`;
`[ZMQManager] VR 3-point control enabled`;以及 `[INFO] Initial VR 3-point compliance: [0.5, 0.5, 0]` 紧跟着
`Policy does not observe 'vr_3point_compliance'`(**顺带确证 compliance 是惰性的**,而且两个
checkpoint 的 `observation_config.yaml` 里 VR 块完全相同 —— 差异只在 heading 归一化)。
我们发的值也在分布内:head (0,0,0.40) 对 deploy 默认 (0.024,-0.008,0.403);腕比默认高 14 cm。

**结论:在 planner 模式下,VR 3-point 腕位姿是给编码器的一个*线索*,不是伺服目标。
这不是 v1.1 的退化,两个 checkpoint 一模一样。**手臂实际跟的是参考动作里的
`motion_joint_positions_wrists_10frame_step1` 等块(来自 `current_motion_`),不是我们这 21 个数。
deploy 源码也对得上:`zmq_manager.hpp:406` 只在 `active_mode_ == STREAMED_MOTION`
(或 planner 未就绪的切换窗口)才让 `pose_interface_` 提供 VR 3-point;planner 模式走基类,
也就是我们发的值 —— 它进观测,但不驱动手臂。

### 34.3 Stage D 不是"未测",是**不适用**:`TRACK_ROT_BOOST` 在死代码上

`TRACK_ROT_BOOST = 4.5` 的唯一消费者是 `primitives_crouch.track()`(第 230 行,
`set_wrists` + `frame()` 走的正是上面测到 ratio 0.01 的那条通道)。
而 **`Primitives.track()` 在 `src/locomani` 里没有任何调用者**(grep 全仓库:只有
`primitives_tuning.py:140` 的注释和 mixin 的装配)。生产抓取走的是 `streamed_arm._push` 的 29 关节流。

所以 Stage D **关闭为"不适用"**:这个常量无法影响生产,§28 两次测量互相矛盾的根因也在这里 ——
我们一直在量一条既不被生产使用、在 planner 模式下也不是伺服的通道。
(这是本线第三个"声明了、校验了/标定了、没人调用"的东西:`ReachConfig.azimuth_deg`、
`_creep_to_band`、现在是 `track()`。值得当成一类检查项。)

### 34.4 方向修正:v1.1 的 wrist-pose augmentation 受益的是**流式姿态**通道

v1.1 的 model card 说它为 "heading-stable 3-point teleoperation" 训练、带 wrist-pose augmentation;
但官方 PICO 的"弯腰抓东西"那条流程是 **POSE / STREAMED_MOTION**(文档原文:streaming the SMPL pose
to the C++ deployment side, "your motion will directly map to the robot"),而不是 planner 的 VR-3PT 子模式。
记忆 `sonic-streamed-motion-channel` 已经量过那条通道能给 ~1 cm 的掌心精度。
**所以下一步应该在 STREAMED_MOTION 上做,而不是继续推 planner 的腕通道** ——
`enter_streamed()` / `pose_chunk()` 已经实现。这也是为什么协作者用 Pico 一次就顺:
他用的是这条通道,而我们的 expert 在 planner 模式下自己做 IK。

## 35. §34 的结论要改:腕通道**是**伺服目标 —— 只是我们从没让 deploy 进过 `teleop` 编码器模式(2026-09-12, 本 session)

出发点是用户的问题:"协作者用 v1.1 走路、操作都没问题,为什么我们换过来这么难"。
按记忆 `question-the-layer-when-the-list-grows` 的做法,先读 vendor 源码和协作者用的那条流程,再动一个实验。
探针 `src/locomani/backends/sonic/diag_vr3pt.py`,runner `scripts/experiments/vr3pt_probe_20260912.sh`,
结果 `outputs/diag/vr3pt_20260912/`(两 checkpoint 各一份 json + deploy.log 副本,视频 `outputs/videos/vr3pt_20260912/`)。

### 35.1 官方 PICO 的"弯腰抓"走的是 `PLANNER_VR_3PT`,不是 STREAMED_MOTION —— §34.4 说反了

`pico_manager_thread_server.py:1843-1849, 2119`:VR_3PT 模式下 manager 发的是**planner 消息**
(`planner=True`),里面带 `vr_position`(9)/`vr_orientation`(12):左腕点、右腕点、颈点,**骨盆系**,
点 = `wrist_yaw_link + R_link @ [0.18, ∓0.025, 0]`、颈 = `torso_link + [0,0,0.35]`,姿态 = `inv(q_pelvis) * q_link`
(训练侧 `gear_sonic/envs/manager_env/mdp/observations.py:1348-1442` 同一公式,**用完整骨盆旋转**,不是 heading-only)。
**PICO 侧没有任何手臂 IK**(文档原文 "non-IK-based VR 3-point calibration"),下肢仍由 k-planner 生成;
手放哪是 **policy 自己解**的。POSE/STREAMED_MOTION 是全身 SMPL 动捕那条,和抓取无关。

### 35.2 §34 测到 0.01 的真正原因:字段的"有无"是 encoder 模式开关,而我们从开机起每帧都带着它

deploy 源码(三个已编译的树都有这段):
- `zmq_manager.hpp:1180-1222`:planner 消息带 `vr_position` → `has_vr_3point_control_ = true`;不带 → false。
- `zmq_manager.hpp:679-693`:该标志的**上升沿** → `current_motion->SetEncodeMode(1)`(`teleop`:输入是 k-planner 的
  **下肢**参考 + 三点目标 + heading 锚,`observation_config.yaml encoder_modes[1]`);下降沿 → 回 `g1`(mode 0,腕槽位清零)。
- `g1_deploy_onnx_ref.cpp:3770`:`Planner()` 初始化时 `planner_motion_->SetEncodeMode(initial_encoder_mode_)`(= 0);
  `zmq_manager.hpp:498`:每次 safety reset 若在 mode 1 也打回 0。

而 `AgentCommander.frame()` 在非 FROZEN 的每一帧都发 VR 字段(默认 `stream_mode` 就是 `PLANNER_VR_3PT`),
从**开机第一帧**起。于是上升沿在 planner 初始化之前就烧掉了,初始化把模式打回 0,之后标志一直为真、再没有沿 ——
deploy 永远停在 `g1`,`vr_3point_local_target` 被 "Not required for this mode - leave as zero"。
两份 09-12 的 deploy.log 顺序完全一致:`VR 3-point control enabled`(:261)**先于**
`Setting planner_motion encode mode to 0`(:265),然后一次 `teleop` 又回 `g1`(:276→:283,即 diag 的复位),此后再无切换。
**§34.2 的"deploy 确实收到了"是对的,"不是伺服目标"是错的:收到了,但编码器不在读它的那个模式。**
memory `sonic-encoder-sees-only-four-inputs` 早就写着 mode 0 只读四项 —— 缺的一步是问"谁把模式切到 1"。

### 35.3 修法与探针:按 PICO 的时序进通道

- `AgentCommander.vr_targets`(新,默认 False,`reset_pose()` 清):VR 字段**只在它为 True 时**进 planner 消息;
  FROZEN 仍只带 `upper_body_position`;其余帧**两者都不带**。`set_neck()` 让颈点可命令(原来钉在 `nominal_neck()`)。
- `backends/sonic/teleop_arm.py` `TeleopArm`:planner 起来、机器人站稳后,**第一帧目标 = 实测位姿**(PICO 每次进
  VR_3PT 也这样重标定,:2094),再置位 → 上升沿落在该落的地方;世界系目标每 tick 用**实时**骨盆位姿重表达,
  伺服侧加 `kp·clip(目标-实测)`;`stop()` 把上身钉回 FROZEN 并清字段(下降沿)。没有 IK,腰不锁。
- 探针六条腿,`policy/sonic_v1_1` 与 `policy/release` 各跑一遍(同脚本、同 bank 位置、instance 0):

| 腿 | 量 | sonic_v1_1 | release |
|---|---|---|---|
| 0 entry | 进 `teleop` 后 2 s 掌心 / 骨盆漂移 | 0.4 / 0.2 cm | 0.6 / 1.0 cm |
| 0 entry | deploy.log | `Planner initialized` → `VR 3-point control enabled` → **`Switched to mode 'teleop'`** | 同 |
| 1 reach(开环 kp=0) | 腕点到 [0.35,-0.22,-0.05]（骨盆系）,位移比 / 到命令点残差 | **1.16 / 4.8 cm** | 1.38 / 10.3 cm |
| 2 servo(kp 0.6) | 残差 | **2.6 cm** | 6.1 cm |
| 3 rotation | 掌心绕骨盆 y 转 60°,交付 / 残差 | **52.8° (0.88) / 2.2°** | 40.0° (0.67) / 5.4° |
| 4 squat 0.10 m | 世界系目标固定:交付蹲深 / 骨盆后撤 / 俯仰 / **掌心世界系漂移** / 残差 | 0.082 / 6.7 cm / 15.0° / **2.3 cm** / 5.3 cm | 0.101 / 6.8 cm / 9.8° / 2.4 cm / 3.4 cm |
| 5 lean | 颈点前倾 0.30 rad:腰 pitch 交付比 / 掌心世界系漂移 | **+0.316 rad (1.05) / 0.5 cm** | +0.006 (0.02) / 1.3 cm |

三件事一次量出来:(a) 腕通道在 `teleop` 模式下**就是伺服**,v1.1 比 release 好一倍(残差 2.6 vs 6.1 cm)——
model card 那句 "wrist-pose augmentation" 终于落在它所指的通道上;(b) **蹲下时 policy 自己补掉了骨盆的后撤和俯仰**
(掌心世界系只漂 2.3 cm,而 §32/§33 里 `project_squat` 预测不到的那部分是 3.5–6.6 cm)—— 也就是说 0.8 那个常数、
`squat_prediction_*`、`coupled_terminal_ik` 的整套预测-拒绝机制,在这条通道上**没有存在的理由**;
(c) **腰是颈点命令出来的**:v1.1 交付 1.05,而手在世界系里没动 —— crate 线 09-12 量到的"planner 关节通道 0.09/0.87"
是另一条通道的数字,这条通道上 release 才是那个不交付的(0.02)。

第一次探针用了相对位移 +0.15/-0.08(从垂手 rest 出发),把点放到骨盆下 35 cm、超出站立包络,ratio 0.71、残差 16 cm;
改成抓取带的绝对点后才是上表 —— 记着:**通道测试的目标要在包络内,否则量的是包络不是通道**
(`vr3pt_sonic_v1_1_run1_relative.json` 留作对照)。第一次跑时 lean 腿也无效(颈点当时还没接线)。

### 35.4 落到 expert:`TeleopGraspPolicy`(`tasks/bottle/teleop_grasp_policy.py`,`ReachConfig.teleop_arm`)

沿用 `CoupledGraspPolicy` 的 brake / stand / 蹲深脚手架,换掉执行:`assess()` 只做几何筛(蹲深范围、地面、初始清障)
并从模板挑**姿态**,**不再因 IK 解不出而拒绝站位**;`finish()` 在 `TeleopArm` 上做 蹲+运到 pregrasp(世界系目标,
蹲的后撤靠测量补)→ 对实测物体 local align(L 形:先横向对齐再进刀,`closure_geometry` + `closure_clear` + 
`CaptureStability` 判"就位")→ 合手 → 抬 10 cm。阶段名保持 gate 认识的那套。
profile `configs/tasks/bottle_visible_teleop.json` = `bottle_visible.json` + `teleop_arm: true`。
runner `scripts/experiments/teleop_pilot_20260912.sh`,bank 用 §33 那 10 个位置(`cases10_visible.json`,
coupled 策略上 release 4/10、v1.1 0/30)。**pilot 结果见 35.5(写入时补)。**

### 35.5 pilot:同一 bank 上 coupled v1.1 0/30 → teleop v1.1 **6/10**(runs 3–4),失败三种各有名字

runner `scripts/experiments/teleop_pilot_20260912.sh`,`outputs/experiments/teleop_pilot_20260912/`,每次 10 局、`--record`(6 局都落了 episode)。

| run | 变量 | 结果 | 失败 |
|---|---|---|---|
| run1 (`sonic_v1_1_run1_crash`) | 首版 | 第 1 局 align 12 s 不过 → 我的超时分支崩 | 位置 1.1 s 就进容差(depth +0.3 / lat −0.5 / h −1.1 cm),但 `closure_clear` 一直报 **`baseline shoulder/elbow comfort limit`** —— 那是给我们 IK 分枝定的关节舒适界,量的却是 policy 自己选的关节 |
| run2 (`_run2_comfortveto`) | 加遥测 | 同上,确认机制 | 同上 → `TeleopGraspPolicy.reference_check` 只留碰撞筛,不看关节界 |
| run3 (`_run3_badbank_grazeclose`) | 去掉关节界 | 0/10:6 局 `[view] object not in ego view at spawn`,2 局合手后 **抬起 0.000 m**,2 局 jaw_tilt 超时 | (a) bank 选错:`cases10_visible.json` 是 09-10 的 badspawn bank(出生 1.1–3.0 m),任何手臂都过不了 6 局;(b) 合手用了 StreamedArm 的"首次接触 +0.28"规则,接触在 alpha≈0.1 的一次擦碰上被锁住,停在 0.010 —— 4.5 cm 的瓶子上 6.6 cm 的爪距,`touching` 为真、夹持力为零 |
| **run4** (`_run4_6of10`) | §33 的 bank `cases10_az374.json`;合手改成 servo 的"持续 3 tick 双指接触 + 再走 0.6"规则;姿态也做反馈(`KP_ROT` 0.6) | **6/10**,抬起 7.4 / 7.9 / 8.7 / 7.9 / 11.1 / 9.9 cm,6 局都过了 carry+hold 尾巴 | 第 1 局 spawn 转身超时(release 在这个 bank 的第 1 局也是它);第 2、6 局 **蹲的时候把瓶子扫倒**(手停在抓取腕点上 8 cm 的 pregrasp,身体下 8 cm、后撤 8 cm、俯仰,滞后的指尖擦到瓶顶,sidecar 里 quat_w 0.30→0.67 在蹲的 1.5–2.1 s 内);第 8 局 jaw_tilt 0.34–0.37 超 0.30,而位置全在容差内 —— 模板姿态自带 0.21 的 jaw tilt 加 10° 姿态残差 |

对照:同一 bank、同一 checkpoint,coupled 策略 **0/30**(§33);release + coupled 4/10。
**换通道不是"调好了",是把三个静默机制拿掉了**:IK 拒绝(`coupled_terminal_ik`)、预测后撤(0.8)、关节舒适界 —— 每一个都是给 release 时代的自家 IK 定的,
在 policy 自己解手臂的通道上没有对应物。**run4 的 6 局成功里,`squat_delivered_pose.retreat_m` 6–10 cm、pitch 13–15°,没有任何预测,手掌世界系残差 2 cm 上下**。

run5(进行中,v1.1 与 release 配对):(a) **先悬停再蹲**——手先到抓取腕点上 0.15 m 的 hover,蹲的时候 hover 在世界系里被握住,蹲完再沿进刀轴下降;
(b) `level_jaw()` 把模板姿态绕进刀轴转到 jaw 水平(模板自带 0.20 rad tilt 是给自家 IK 的舒适选择,这条通道上不需要)。

### 35.6 run5 配对:**v1.1 7/10 对 release 3/10**,而且 release 的失败是一个机制,不是散的

同一 bank、**同一份代码**(两条腿之间没有任何改动)、只有 `SONIC_POLICY` 不同,
`outputs/experiments/teleop_pilot_20260912/{sonic_v1_1_run5,release_run5}/`,
报告 `scripts/experiments/teleop_pilot_report_20260912.py`。

|  | sonic_v1_1 | release |
|---|---|---|
| 成功 | **7/10**(过了 spawn 的 7/9) | 3/10 |
| 逐局配对 | **4 : 0** 判给 v1.1 | McNemar 精确 **p = 0.125** |
| 蹲后腕残差 | **1.2–3.7 cm** | 2.9–3.9 cm |
| local align 用时 | 1.5–4.4 s | 1.8–5.3 s |
| 抬起 | 8.4–11.6 cm | 12.2–13.8 cm |
| 失败的高度误差 | align 末段中位 **−0.6 cm** | **7 次失败全是 +2.2…+7.4 cm**(中位 2.8),残差 3.9–11.4 cm |

**~~这是这条线上 v1.1 第一次赢过 release~~ —— 这句话在 §35.10 被我自己推翻了**:run6 修好那两个门之后 release 从 3/10 变成 6/8,4:0 是**我的门**造成的,不是 checkpoint 的差别。(§33 在同一 bank 上是 release 4/10 对 v1.1 0/30。)
4:0 在 10 局上还不够判(记忆 `ten-round-pilots-cannot-measure-this-line`,p=0.125),
**但 release 的 7 次失败共用一个机制**:它在这条通道上**系统性地停高约 3 cm**,越过 `HEIGHT_TOL=0.020`——
和探针测到的 release 伺服残差 6.1 cm(v1.1 2.6 cm)是同一件事。失败不是抖出来的。

两个仍然属于**我自己**的失败,都是"私有常量比验收判据还紧",在自家驱动关节的通道上成立、在 policy 自己握着手臂的通道上不成立:
- **round 4**:`safe True` 从 t=2.1 s 一直到 7.4 s 还是超时。卡的是 `CaptureStability` 的速度门——
  它按**逐 tick(20 ms)**估相对速度,而 0.035 m/s 是当年对着我们自己限速的关节流标定的;
  这条通道上 policy 自己握着手臂,1 mm 的抖在 20 ms 上读成 0.05 m/s,在 0.2 s 上读成 0.005–0.01 m/s。
  改的是**估计的基线,不是阈值**(`SPEED_WINDOW_S = 0.2`),而且只改我自己的循环,`CaptureStability` 是和 coupled 线共用的,不动。
- **round 5**:lat 停在 −2.4…−2.9 cm,depth 冻在 +10.5 cm 整整 12 s。我的 L 形门在横向 >1.8 cm 时不许进刀,
  而这条通道的横向稳态误差就是 ~2 cm。门比 `LATERAL_TOL=0.026`(验收判据本身)还紧 → 死锁。改成 `LATERAL_ALIGN_M = LATERAL_TOL`。

run4 的三种失败里,**蹲时扫倒瓶子**(先悬停再蹲)和 **jaw_tilt 超界**(`level_jaw`)在 run5 里都不再出现:
run4 失败的 case 2/6/8 在 run5 全部成功。run6 用修好这两个门之后的同一份代码再跑一次配对。

### 35.7 对其它线的影响:控制路径没变,只有一列 provenance 变了(**gate 尚未重跑**)

`frame()` 现在只在 `cmd.vr_targets` 为真时带 VR 字段。对 coupled / crate / pick_place 这些线:
**deploy 两种情况下都停在 encoder 模式 `g1`**(改之前上升沿在 boot 烧掉、planner 初始化打回 0;改之后根本没有沿),
而 `g1` 模式压根不读 `vr_3point_local_target`(`cpp:2093` "leave as zero"),
`has_vr_3point_control_` 除了切模式之外不改变 deploy 的任何行为 —— 所以**控制上是同一件事**。
手臂本来也不是走这条通道:生产抓取是 `PLANNER_FROZEN_UPPER_BODY` 的 17 关节目标,那一支我没动。

唯一真实的差别是 **exporter 记的 `teleop.vr_3pt_position` / `vr_3pt_orientation` 会变成空**
(`run_data_exporter.py:407-412` 没有这两个 key 时写 None,不报错;`docs/single_arm_design.md` 里这两列标的就是"不进训练")。
改之前它们记的是**惰性的 rest 位姿常量**,不是任何人执行过的东西。

**这一条是推理,不是实测。** 规范要求改主代码的 PR 附 `scripts/gate.py compare` 的输出,
而 bottle gate 的 baseline 还是 29° profile 那次录的(STATE 里已有的 OPEN),所以**合并前要在 GPU 空闲时重跑一次 bottle gate**,
确认 coupled 线的 10 局与 baseline 一致。

### 35.8 顺带查出来的:`action.psi0_18` 在**已有的每一个数据集里都是全零**

担心的是"VR 字段改成按需发,会不会把别的线的数据列改坏"。查下来相反:
`convert_to_hf_format.compute_psi0_columns`(:319-327)**无条件**用 `teleop.vr_3pt_position/orientation`
算 `action.psi0_18`(18 维腕位姿动作,`modality_psi0.json` 的全部动作);
而 exporter 只在 `stream_mode == 5` 时写这两列,coupled 线录制窗口里一直是 3(`PLANNER_FROZEN_UPPER_BODY`)——
抽查 2026-09-11 的 session:864 帧,这两列**只有一种取值,全零**。

所以:(a) 我这次改动对 coupled / crate / pick_place 线**记录的内容零影响**(本来就是零);
(b) **`action.psi0_18` 在历史上的每一个 bottle 数据集里都是全零**,psi0 这套 modality 是死数据
(本线第四个"声明了、标定了、没人用"的东西,前三个见 §34.3);
(c) 在 teleop 通道上它第一次是真的:run5 的 episode 0 里 **621/1089 帧** `stream_mode == 5` 且带真实腕目标
(右腕 pelvis 系 [0.248, −0.376, 0.037],颈 [−0.003, −0.001, 0.404]),走路与抱稳段是零。
训练要用 psi0 这套的话,**零段与非零段的不连续要先处理**。

另外,exporter 在字段缺失时写的是**零,不是 None**,converter 不会崩 —— 这一条也验证过了。

### 35.9 run6:两个门修好之后——**成功率没动(7/10),但失败换了地方,收敛快了一倍**

同 bank 同 checkpoint,只改了 §35.6 说的那两个门(`SPEED_WINDOW_S=0.2`、`LATERAL_ALIGN_M=LATERAL_TOL`):

| | run5 | run6 |
|---|---|---|
| v1.1 成功 | 7/10 | **7/10** |
| 逐局配对 run5 vs run6 | — | **1 : 1**(没有方向) |
| local align 用时 | 1.5–4.4 s | **1.4–2.0 s** |
| 失败 | spawn 转身 / 速度门 / 横向死锁 | spawn 转身 / **hold 掉落** / 一次 depth+3.5 height−2.2 超时 |

**逐 case 看,两个门确实修好了**:run5 卡速度门的 `az30:3` 这次 1.9 s 就 OK;run5 横向死锁的 `az30:4` 这次一路走到抬起(8.7 cm)。
**但成功率没有变**,因为失败换到了别处 —— 10 局分辨不了这种差别(记忆 `ten-round-pilots-cannot-measure-this-line`),
所以这两个改动的依据是**机制 + 收敛时间**,不是成功率,别拿 7/10→7/10 当"没用"或"有用"。

新出现的 `[hold] object slipped 79.9 cm`:抬起 8.7 cm 之后在抱稳段掉了。原因是我的 `close_hand` 停在
"首次持续接触 + 0.6 个 ramp",而 servo 对 dex1 + `require_bilateral` 的规则是**合到 alpha 1.0**
(`servo_phases.py:301-302`)。部分夹持撑得过抬起,撑不过搬运。已改成默认 `full_close=True` 跟齐那条被验证过的规则。
(改的时候差点破坏配对:run6 的 release 腿还没起,新进程会 import 到改后的代码 —— 已先改回、等 run6 跑完再落。
记忆 `never-edit-a-running-bash-script` 的同类问题,对象是 .py 与"下一条腿"。)

### 35.10 **自我推翻:§35.6 那个 4:0 是我自己的门造出来的,不是 checkpoint 的差别**

run6 的 release 腿和 run5 的 release 腿之间,**只差 §35.6 那两个门的修法**,结果:

| release,同 bank 同 checkpoint | run5(门很紧) | run6(门改到验收判据) |
|---|---|---|
| 成功 | **3/10** | **7/10** |

而 v1.1 两次都是 7/10。也就是说:**那两个门紧到只有 v1.1 过得去**,
于是 run5 量出 4:0、我据此写了"v1.1 第一次赢过 release"——**错的**。

机制对得上:run5 里 release 的 7 次失败有 **5 次是纯 `height` 拒绝**,而它们的横向误差是 1.2–2.6 cm,
正好骑在旧的 `LATERAL_ALIGN_M=0.018` 上 —— 那条规则在横向没对齐时**扣掉进刀方向的分量**,
于是整段对齐都在一个被拉回来的瞄点上做,最终几何自然差一截;
再加上逐 tick 的速度估计对抖动更大的 release 更不利。两个门都**不成比例地惩罚了 release**。

**改完之后的结论:在这条通道上 v1.1 与 release 分不出来**(run6 两边都是 **7/10**)——
和 §27 在 coupled 通道上的结论一致。

**真正的结果不是"v1.1 赢了",而是:这条通道把 v1.1 从同一 bank 上的 0/30 变成 7/10。**
那才是"换成 v1.1 之后单臂任务该怎么做"的答案;v1.1 与 release 谁更强是另一个问题,答案仍然是"分不出来"。

方法论(第三次了,见 §28 的仪器、§30 的方位角):**在自己写的执行器里量 checkpoint 的差别,
先问那些常量是按谁标定的**。这次的形式是"私有门比验收判据还紧",两条都是从自家驱动关节的通道继承来的。

### 35.11 run7:合到底的夹持把 79.9 cm 的掉落变成 4.1 cm 的滑移;三次汇总 **20/30**

`full_close=True`(合到 alpha 1.0,跟 `servo_phases._phase_close` 对 dex1 + bilateral 的规则一致),
其余不变,`sonic_v1_1_run7_fullclose`:**6/10**。

| | run5 | run6 | run7 |
|---|---|---|---|
| v1.1 成功 | 7/10 | 7/10 | 6/10 |
| `[hold]` 滑移 | — | **79.9 cm(掉了)** | **4.1 cm**(阈值 `PALM_DRIFT_MAX=0.03`) |

**夹持确实变实了**(掉落 → 只滑 4.1 cm,刚过 3 cm 判据),**成功率没变**(6 对 7,10 局分辨不了)。
和 §35.9 一样:依据是机制,不是成功率。

**三次汇总:teleop 通道上 v1.1 20/30,同一 bank 上 coupled 的 v1.1 是 0/30**(Fisher 双侧 **p = 1.4e-08**)。
这才是可以引用的数字;"v1.1 对 release"的答案见 §35.10:**分不出来**。

run7 新出现两局 `[grasp] local alignment disturbed target or base`(前两局):
align 起点是抓取腕点上方 8 cm 的 pregrasp,而推进步长上限 `STEP_M` 是 0.2 m/s —— 9 cm 的下降只要 0.45 s,
横向还有 1-3 cm 残差时指尖就扫到瓶子了(t=0 height −8.3 cm → t=1.1 s +1.0 cm)。
**没有调它**:这是下一个该看的地方(下降速率按横向残差缩放,或者对齐到位再降),不是这次的结论。

## 36. 每批第 1 局在出生阶段必败:标定时的 start_control 抢占了朝向锚(2026-09-10,session cf)

> 这一节是分支 `fix/first-round-anchor` 上的内容,2026-09-12 合并时按 append-only 惯例移到文末并从 §14 改号为 §36。
> 正文未改。分支里引用它的地方(`single_arm_design.md`、`silent_failures.md`)写的是「台账 bottle §14」,即本节。

**现象**:`visible10_20260910`(2/2)与 `visible200_20260910`(17/17,截至第 17 批)每批的第 1 局都在出生阶段失败:
`[stand] could not turn N deg to the approach heading at the spawn (turn_to -88deg timed out)` 11 次、
`[view] object not in ego view at spawn` 6 次;第 2–10 局这两种失败一次没有。代价:每批白跑 1 局(~25 s + 10% 局数),
两条失败串混进 manifest 的 failure_reasons。

**根因**:`grasp_gym` 的场景装配在 stack 起来后做一次名义出生点标定——复位到名义位姿、`start_control()`、IDLE 2.5 s。
deploy 在它的**第一次** `start_control` 处捕获朝向锚(memory `sonic-planner-heading-anchor`),这时机器人朝向 yaw 0。
`round_phases._phase_spawn_and_settle` 用 `p._gym_started` 判断"这台 deploy 是否已经带锚":为 True 走 hold 1.5 s + `reanchor`,
为 False 走"锚 = 落地 yaw"。标志只在 round 阶段置 True,标定处没置,于是第 1 局 teleport 到 yaw ≈ −90° 后被当作新 deploy,
锚差 ~90°,IDLE facing 要么转不到(超时)要么站起来面朝别处(物体出视野)。第 2 局起标志为 True,正常。
2026-09-10 之前每局重启 deploy 时第 1 局同样跳过重启("标定已在这台 deploy 上跑过"),所以这个坑一直在,只是每批 1 局、被 yield 吞掉。

**修法**(commit 见下):`grasp_gym.start_control_once(p)` = `start_control()` + `p._gym_started = True`,标定处改用它;
`fresh_deploy` 重启后仍置 False。回归测试 `tests/test_review_2026_09_10.py::FirstRoundAnchorTests`。

**验证**(`outputs/experiments/firstround_fix_20260910`,修复分支的 src,同配方 3 批 × 4 局,seed0 32000,10:45–10:53 UTC):
每批第 1 局 **3/3 过出生阶段**(`turn_to done`、produced)——1 局 EPISODE OK(40 s),2 局在抓取阶段以普通原因失败
(stand infeasible、coupled IK did not track),与第 2–4 局同分布;主线同日 0/34。12 局共 4 OK,样本小、失败类型正常,
不作 yield 结论。**裁定**:合并;以后每批多出 1 局有效样本(采 200 条少跑约 34 局、约 15 分钟)。

## 附:评测视角的同批实验记录(原 `docs/single_arm_design.md` §4/§6,2026-09-18 并入)

以下 625 行原样来自评测指南,那里不再保留逐轮记录:**怎么跑**归 runbook、
**为什么这么设计**归详解、**某天跑出什么数**归本台账。

与上面 §1–§9 讲的是同一批实验的两个视角:上面按数据形状/数据量组织,
这里按评测协议与部署参数组织。两边数字不一致时以带日期、带 seed、带局数的那一条为准。

#### 2026-08-27 更新:并行罚项、配对规则、观测流审计

`seq_ab` 首次实战(时序集成 A/B)撞出四件事,全部已落地:

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
- **观测流可审计了**:`src/locomani/pipeline/frame_rate_guard.py`(878dc0b/b1f89cd,
  逐局 obs_hz 中位/degraded 标记,标记不作废)+
  `src/locomani/pipeline/instance_watchdog.py`(30fa1fe,process/port/stream/
  progress 四轴,只报不动)+ 评测结束时归档逐实例 sim.log(83975fb)。背景:一个
  泄漏的 chrome 曾吃掉 9 核 8 小时,把观测流饿到 13-15 Hz 而没有任何东西报警。
- **start_stack 的 image-publish 守卫**(9bd0148):孤儿 publisher 占住 :5555
  时,旧脚本会在零帧状态下报 "stack up"。

#### 2026-08-27 发现:闭环大多数局没考走路(stand-start gap)

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

#### 2026-08-27 终表:同一模型上的客户端/集成三方配对(单实例,90s,seed 6000)

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
  与机制推断(每 ~0.9 s 冻结 0.3-0.6 s 喂给站立吸引子)一致。

#### stand-start 漂移定位(队列 #2,blocking90 百局标记)

prelude 两个候选**双双出局**:entry→t_start 位移中位 0.015 m,priming 执行段
0.014 m——**漂移不在 VLA 客户端里**。fork 时已在站位(73/99,y 中位 0.611)。
真正的窗口是 **settled(漂移检查通过,y≈0.94)→ fork 的 ~2-3 s**:位移中位
0.342 m,69/85 局 >0.25 m,双峰(~15% 的局纹丝不动)。窗口内只有 stream_mode
切换 / pin_arms / hold(2.0 IDLE) / start_video 四件事,细分标记已加,探针在跑。

#### 2026-08-28:干净全任务基线(stand-start 修复后,队列 #4)

**round4 模型在完整任务(出生点走过去再抓)上的真实成绩:12/100 = 12%(Wilson 7-20%)。**
修复保持:100 局首帧 dist 中位 0.74,99/100 ≥0.55——每一局都真的从出生点开始。
失败:reach 83(anatomy: arm-never-up 0 / **up-but-never-near 81** / staged-no-fire 2)
/ close 3 / walk 2。

改口规则,自此生效:

- **梯子(5.9 / 13.7 / 28.0 / 47.5)全部是"混合任务"数字**(~2/3 站位伸手 +
  ~1/3 完整任务),相对比较有效,绝对值不可再单独引用;与 95.2% 脚本专家
  (全程走)的对比只在本节的干净数字上成立。
- **12% 是第一个可发表的完整任务绝对值**,也是后续一切改进的配对基准
  (seed 6000,逐局可配)。
- 值得注意:walk 失败仅 2/100——VLA 的走路段本身几乎不挂;瓶颈与混合任务
  时代诊断一致,仍是 mid-reach(81/88)。

#### 2026-08-28:stand-start 修复升级为官方 initial-pose 机制

官方教程(vla_inference: Customizing the Initial Pose)给出了正确原语:
**等待期不站在 planner 通道里,而是持续流式发送预置的站姿 latent token**
(`LATENT_INITIAL_MOTION_TOKEN`)。这正好命中量到的机制——drift 是 planner
通道的病,token 通道实测稳如磐石——所以 async 局现在的第一件事就是灌站姿
token + enter_streamed,keepalive 线程在整个慢启动期间(连流+首推理)保持它,
最后 1 秒线性 blend 进首个 chunk(官方 client 同款)。

对比二次瞬移:spawn_off 中位 0.065→0.107(残余=1.5s settle,唯一还在 planner
里的窗口),但尾部 0.33→0.18 且有界,8/8 从出生点起步,blend 零摔倒。
**默认路径不再依赖任何 sim 特权——与真机部署同构**;二次瞬移降级为
`LOCOMANI_REPLACE=1` 应急开关。

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
  闭合误差 2.2 帧)。注意**开环数字跨轮不可直接比**——val 集随数据集变了。
- 闭环终局:**14/102 = 13.7%(Wilson 8–22%)**,round 1 的 2.3 倍,置信区间
  几乎不重叠——靶向 DAgger 方向确认。失败仍以 reach 停滞为主:50 局恢复数据
  补上了洞的一部分,没补完(配比 50:200)。
- 客户端 A/B 判定:async 在 3/38(7.9%)处存档,**与 blocking 的 13.7% 相比
  无改善迹象——客户端不是瓶颈**。真正的病灶随后被数据测量找到:训练数据里
  到位→伸手中位 4.6 s 的准静态站立(官方人类遥操作参考集是 0.2 s),
  96% 超过一个 chunk 的覆盖能力——policy 在正确模仿"站着"。

### Round 3(fast200 = 200 局,停顿修好,**零 DAgger**):28/100

**闭环 28/100 = 28.0%(Wilson 20.1–37.5%)**。round 1 的 4.7 倍、round 2 的 2.0 倍,
区间与 round 1 完全不重叠。脚本专家参考 200/210 = 95.2%。

最要紧的一点:**round 3 没有用任何 DAgger 数据。**round 2 靠额外采 50 局恢复轨迹
换来 2.3 倍;round 3 只是把脚本专家的停顿削掉(到位→伸手中位 4.6 s → 3.0 s,
episode 从约 35 s 压到 20–22 s),用**同样 200 局**又翻了一倍。这直接证实了
round 2 末尾那个诊断:病在**数据形状**,不在数据量。

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
却不发。这是 covariate shift 的典型形状——闭环观测落进了训练集没覆盖的状态。

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
意义。模型误差随 horizon 增长极慢(40 步只涨 41%),persistence 涨 5.6 倍——
模型在**预测轨迹**而不是复读。整条 chunk(h=40,1.33 s)的关节误差 1.365°
正好压在噪声底 1.361° 上,所以整个 chunk 可以信任。

**FSQ 吸附有反面证据**:同一 checkpoint 同一 horizon,`nmse_token` 0.1592(吸附后)
vs `nmse_token_unsnapped` 0.1369——**吸附让指标恶化约 16%**。官方
`run_vla_inference.py` 发的是原始浮点、不做吸附。开环已说明它有害,闭环对照
见下方 round 4。

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
10 个百分点级别的差异**——这是本轮最该记住的一条。

(开环那边吸附确实恶化约 16%,那个测量没错——只是**没有转化成闭环收益**。
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

### 2026-09-15:垂臂开局 + 硬化手指垫(`bottle_vla200b_20260914`):17/100

采集 272 局 / 200 成功(74%,seed 30000)→ 190/10 episodes → 10,000 步(train loss
0.087)→ 100 局闭环(seed 6000,async,horizon 16,90 s,`whole_body_initial_token.json`
开局):**17/100**,Wilson 95% 0.11–0.26。失败 60 局是"到桌边但从未下闭合指令"
(最近距离中位 0.32 m),11 局是 spawn 时物体不在 ego 视野,7 局闭合了没抬起,
3 局走路没到,2 局 stand 阶段。与 `visible200` 的"爪子从不触发"是同一吸收态;
专家采集的 74% 成功率没有传递到策略里。离线逐键替换(`eval_dump10/`)表明闭合由
ego + 腕两路图像共同决定、状态无关,失败局停在专家的 park 距离(刹车 0.40 m)
而没有做专家随后 10 cm 的全身前移;刹车改 0.31 后同 20 seed 采集 17/20
(原 15/20)。原片 `outputs/bottle_vla200b_20260914/eval/videos/`,预览
`eval/previews/`(每片 ≤500,000 字节,`verification.json`)。

### 2026-08-30:部署侧参数普查(Phase 0 + replan-interval 配对)

起因:协作者只改部署端参数报到 73%。本节是把部署侧因子系统扫一遍的记录,
**包括被杀掉的假设**——这一轮七个假设死了六个,记下来比记结论重要。

#### Phase 0:离线扫描(`scripts/phase0_denoise_sweep.sh`)

walk200/checkpoint-10000,20 局 val,horizon 16。为这次调查给 `eval_open_loop.py`
加了 `std_ratio_token` / `std_ratio_gripper`(预测的逐维 std ÷ 真值的逐维 std)——
**nMSE 对残差的符号对称,看不见"幅度收缩"**,而欠积分的 flow matching 干的正是
这件事。

| steps | std_ratio_token | std_ratio_grip | 死区占比 | nmse(吸附) | nmse(裸) | deg_mean |
|---|---|---|---|---|---|---|
| 4 | 0.9911 | 1.0023 | 0.0658 | 0.1334 | 0.1101 | **1.3003** |
| 8 | 0.9964 | 0.9997 | 0.0915 | 0.1385 | 0.1149 | 1.3283 |
| 16 | 0.9945 | 0.9991 | 0.0977 | 0.1487 | 0.1238 | 1.3856 |
| 32 | 0.9983 | 1.0001 | 0.0871 | 0.1574 | 0.1317 | 1.4044 |

真值死区占比 0.0339;`deg_mean` 噪声底 **1.276**。

- **`--denoising-steps` 删除**:五个指标、四个点,全部单调恶化。4 是最优,机制上
  也讲得通——模型按 4 步的 flow schedule 训出来,多走 Euler 步是沿一个为 4 步
  调好的向量场**过冲**。
- **snap 有害**:每一档裸预测都比吸附后低 ~21% nmse。闭环对照仍未做。
- **模型开环 1.3003 对噪声底 1.276**——只高 2%。**闭环丢掉的点,几乎全部在
  部署侧或分布侧,不在模型里。**

#### 观测延迟 δ:33 ms,不是因素

`ComposedCameraClientSensor.read()` 每 10 帧自己打一次 `Image latency`。
walk200 两场共 2479 个样本:

    p10 18.1   p50 32.8   p90 47.1   p99 59.7   max 67.2 ms   (>80 ms 占 0.0%)

`latency_index` 未补偿的相机部分只有 ~33 ms,不到 25 Hz 的一帧。

#### `--replan-interval 1.6` 配对:显著更差(seed 6000,20 局,60 s)

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
失败模式是**走位**不是伸手,结论不受 arm B 的实现瑕疵影响。

#### 被杀掉的六个假设

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

#### 唯一跨批复现的判别项:爪子

    成功: grip 打到 -0.0200 (clip 底)
    失败: grip 卡在 +0.0115 ~ +0.0195,跌破 0.010 的比例 0%

walk200 111 条 trace、两臂 38 局,**四批数据同一个带**。而训练集 gripper 是硬
双峰:>0.020 占 53.3%,<0.005 占 43.5%,中间 [0.005,0.020] 只占 **3.3%**。
失败局整场停在训练分布的空洞里。

模型对"我到瓶子边上了"**有**表示:远时 24.5(全开),靠近掉到 ~18,手臂甩开还
会正确弹回 24.5。它只是从不越过那道坎。**但它区分不了"靠近 0.195 m"和
"到位 0.135 m"**——两者都输出 ~18.5,所以简单阈值二值化会对着空气合爪,必须
扫阈值而不能假设。

#### 停住距离有一个窄甜区,损失的大头在这里

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
13%——损失的大头不是"不会抓",是**停错地方**。

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

预测:`--spawn-edge 0.9~1.0` 应显著高于 1.2 和 0.7。这个预测很硬、很容易被
证伪、且不需要改任何代码(后文证伪了,见"悬崖修正")。

#### trace 新增字段

`pelvis_z` / `torso_pitch` / `palm_quat` / `obj_in_palm`。前两个是为了把"抬手"
和"下蹲"分开(没有它们,`palm_z` 单独看会读成上面那个循环结论);后两个是因为
`palm_obj` 是标量距离,分不出"抓握位姿"和"同样远但朝向不对"。

### 2026-08-31:普查收尾——grip-gain 配对、200 轮随机 spawn 扫描、悬崖修正

#### `--grip-gain 3.0` 配对 A/B(spawn 1.1–1.3,各 20 轮)

A 4/20 vs B 3/20,McNemar p = 1.0000,**无差异**。决定性细节:

- A 自主合爪 4 次,palm-obj 中位 0.148 m(0.138–0.151),**4/4 成功**;
- B 被增益强迫在中位 0.211 m(0.173–0.279)合爪 17 次,只成 3 次;
- 失败类别整体从 reach 翻成 close,总成功一个没多。

**结论:gripper 是近乎完美的汇报者。** +0.0168 的"犹豫"是模型正确地说"还没
到位",不是不敢抓。修 close 是给失败改名,不是修失败。

#### 随机 spawn 扫描(两实例同配置异 seed 7000/8000,--spawn-edge 0.7–1.4,200 轮)

总计 **79/200 = 39.5%**,gate 拒绝 29(几乎全在 [1.3,1.4):23/26)。

    spawn-edge   n   net 成功率 [Wilson95]
    [0.7,0.8)   29   34.5% [20,53]
    [0.8,0.9)   32   40.6% [26,58]
    [0.9,1.0)   21   47.6% [28,68]
    [1.0,1.1)   37   40.5% [26,57]
    [1.1,1.2)   26   50.0% [32,68]
    [1.2,1.3)   23   65.2% [45,81]
    [1.3,1.4)    3   3/3(gate 吃掉 23/26,无法解读)

**曲线平缓、CI 全部重叠**——之前从两个定点 run 外推的线性关系在单一随机化 run
内不成立(corr(spawn, park) = +0.27),预言的"0.9–1.0 甜区"没有出现。net 有缓慢
上升趋势但 200 轮分辨不了;失败构成各 bin 一致,reach 主导。

#### "窄甜区" → "悬崖" 的修正(本轮最重要的更正)

合并两 run 时 [0.40,0.45) bin 读出 93%(13/14)——是跨 run 汇总的假象。单一
随机化 run 内(n=103):

    park [0.00,0.35) 42%   [0.35,0.40) 68%   [0.40,0.45) 58%
         [0.45,0.50) 44%   [0.50,∞)  5% [1,24]
    park < 0.50 → 83 轮 55%      park ≥ 0.50 → 20 轮 5%

存活的不是"峰",是 **0.50 m 处的悬崖**。0.50 恰好是脚本专家站位生成器的物理
上限外缘;但注意专家在 [0.50,0.62) 有**稀薄**覆盖(5.73% 帧、171/171 局),
"零覆盖"的说法不成立,且专家在 0.50–0.55 抓成过——悬崖的机制仍未证明,只有
现象复现两次。

#### 三条跨 run 方法论教训

1. **被结果污染的统计量**:max palm_z "成功 1.080 vs 失败 0.909" 只是在复述
   结局(成功局的 max 量的是瓶子被举到的高度;抓取本身发生在 0.718)。
2. **跨 run 绝对值不可比**:同 checkpoint 同 seed,定点 spawn 1.2 给 45%,
   spawn 1.1–1.3 给 8%;同一停位 bin 两 run 成功率 14% vs 62%。
3. **合并 run 的 bin 会制造假峰**:见上。结论只从单一随机化 run 内做。

### 2026-08-31:prelude 偏移调查——"stand-start drift is back?" 警报的结案

`tstart_spawn_off` 验收线(0.1 m)在近期所有 eval 里每轮被突破(中位 0.19–0.26,
walk200 主评测 73/100 轮 WARNING),但**与成败无关**(成功局 0.189 vs 失败局
0.197)。逐段分解 + 两个历史 probe 对比 + 30 s hold-only 测量,因果链:

1. **re-teleport 不是失效,是退役**:被官方 initial-pose 机制(站立 token
   keepalive + 1 s blend)取代,当时验收 med 0.107。`LOCOMANI_REPLACE=1` 只是
   留的 sim-only 后门,默认关。
2. **数值翻倍的引入点是换成 medoid token 那次**:probe_initpose(NVIDIA)med
   0.108 vs initpose_ours(ours)med 0.254——换 token 那次自己的 probe 里就写着,
   没人对验收线。
3. **但 medoid 无罪**。`diag_token_hold.py` 纯 hold 30 s:ours median
   **0.010 m**,official 0.002 m——两个 token 都磐石稳定。
4. **主项是 planner→streamed 的模式切换台阶,不是 blend**(8 轮拆分 probe:
   hold_off median 0.084 / max 0.183,blend_off median 0.033)。enter_streamed
   时参考从 planner 姿态跳到 INIT_TOK 站姿,机器人物理地挪一小步——每轮都有、
   两种 token 都有、有界。
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

### 2026-08-31:dart200 部署筛查全集——停顿死循环、ensemble 翻案、blocking 三修复

dart200(208 局,146 条 DART 恢复演示)训练 10000 步(loss 0.047)后,基线闭环
出现新的主失败形态:**桌前停顿**(到桌后停滞 >30 s 占 38%,walk200 是 15%;
停滞中位 9.2 s vs 1.0 s)。机制:DART 数据里"远停 + 爬行走入"的帧量大,叠加
Markov 单帧观测,0.4 s 重规划每次都从静止画面重抽样"停顿的开头"——死循环,
靠采样随机性逃逸。

#### 20 轮筛查全集(全部 ckpt-10000、seed 6000、spawn 1.1–1.3,逐轮配对)

| 配置 | 成功 | 对基线不一致对 / p | 停滞>30s |
|---|---|---|---|
| 基线 async 0.4s | 6/20 | — | 38% |
| h40(--replan-interval 1.6,整 chunk 执行) | 11/20 | 8:3 / 0.23 | 20% |
| **ensemble N3 M+0.3** ⭐ | **16/20** | **12:2 / 0.013** | 10% |
| h40+ensemble | 12/20 | —(≈h40,重叠归零,参数表早有预言) | — |
| ensemble N4 | 6/20 | 11:1 / 0.006(负) | — |
| ensemble M0 | 8/20 | 10:2 / 0.039(负) | — |
| ensemble M−0.3 | 10/20 | 8:2 / 0.109(负) | — |

- **ensemble 的机制**:三个重叠 chunk 都对"当前瞬间"做过预测;停顿时最新 chunk
  说"站着",0.8/1.2 s 前的 chunk 说"此刻早该动了",加权平均把动作拽出不动点,
  同时保留 0.4 s 闭环纠正。
- **N=4 有毒**:第 4 个 chunk 只贡献自己 horizon 的第 30–40 帧,误差沿 horizon
  单调涨,尾帧毒化平均。**M 轴**:+0.3 > 0 > −0.3。
- **两项 walk200 时代结论在 dart200 上翻案**(replan-1.6:1/20→11/20;
  ensemble:−13→+50)。教训已写入参数表:**部署旋钮的结论是 checkpoint
  条件化的,换 checkpoint 必须复筛**;引用旧实验要引 docs 裁定,不要引中途重挖
  的原始 json。

#### blocking 客户端三修复(2026-08-31)

blocking 模式起步即前扑的根因:推理间隙(0.4–1 s)一帧不发,deploy 饥饿时把
参考**钳在最后一帧**——一个带前向速度的迈步中段姿态,50 Hz 控制器追着冻结
参考把机器人按倒。修复:

1. **gap feeder**:推理间隙由后台线程继续按 25 fps 喂当前 chunk 的剩余帧
   (第 17–40 帧),chunk 耗尽则显式保持末帧;
2. **站立 token prelude**(async 修复的移植):INIT_TOK prime → 首次推理期间
   钳在站立帧 → 1 s blend 交接 chunk-0。原实现用 chunk-0(走路意图)prime
   1.2 s,就是入场 drift;修后站立段位移 0.02 m;
3. **latency-compensated 起点**:新 chunk 从 `latency_index(now−t_obs)` 帧开始
   执行而非第 0 帧——原实现每周期倒带重演 ~0.5 s 的过去。

验证:10 局(H=40)**5 OK、0 摔倒**,水位 ≈ async-h40(blocking 税 ≈ 0)。
注意 blocking 下 ensemble 结构性不可用(单 chunk 无重叠)。

残余的入场前倾(enter_streamed 瞬间参考从 planner 跳到 medoid 站姿,SONIC
挪一小步接管,~0.08–0.18 m):结构性地板,与成败零相关,两种 init token 都有,
接受现状。

### 2026-08-31 终:100 轮三方链——数据×部署的交互作用

    walk200 基线 → dart200 基线 → dart200+ensemble(全部 seed 6000,spawn 1.1–1.3)
      34%(27/80)    31%(20/64)     73%(58/80)      [按轮 id 去重]
    名义值:38/101        31/100          73/100  Wilson95 [63.6, 80.7]

| 配对 | 不一致对 | McNemar p |
|---|---|---|
| 数据效应(基线部署下) | 11:10 | 1.00——**零** |
| 部署效应(dart200 上) | 30:4 | 0.00001 |
| 合计 | 40:9 | 0.00001 |

**DART 数据在默认部署下颗粒无收**:停滞 >30 s 占 48%(中位 19.3 s),恢复
技能被自己带来的停顿死循环吃光。**ensemble 是解锁器**(+42 点);而 fast250
时代证据显示 ensemble 单独用不涨(−13)。**交互作用:数据给技能,部署给兑现
通道,乘积才是 73%。**

机制判据(ensemble run 的 76 条 trace):**停位 ≥0.50 m 从 19% 降到 1/76**
——悬崖不是被爬上来,是没人再掉下去;停位内成功率 55%→76%。失败构成:
reach 21、view 4、close 2。

**协议脚注**:eval 的 `--resume` 会重启轮号,续跑段把前 20 个出生点重复评一次
——"101 轮"实际 ~80 独立出生点 + 20 重复。历史阶梯数字同协议,横向可比;报告
独立样本时用去重值。

阶梯更新:5.9 → 13.7 → 28.0 → 47.5(@0.7)→ **73.0(@1.1–1.3,数据+部署)**。

### 2026-09-02 裁定:deploy 状态与 ensemble(flow200 ckpt-10000,seed 6000,配对)

四臂,同一 checkpoint、同一 nvdef 二进制、同一种子:

| deploy 状态 | 无 ensemble | ensemble N3 M0.3 | 配对 McNemar |
|---|---|---|---|
| 干净(SONIC_AUTO_START,不碰 planner) | base100 **50/100** | clean100 **46/100** | 22:26,p=0.67 |
| 退化(planner settle 后 enter_streamed) | planner50_base **29/43** | planner50_ens **12/43** | 3:20,p<0.001 |

结论:

1. 干净 deploy 上 ensemble 没有可测收益;退化 deploy 上 ensemble **有害**
   (−39 点)。早期"ensemble 明显提升"(dart200 时代)在 flow200 上不成立;
   它是否本来就是退化 deploy 的产物,用现有数据无法判定(checkpoint 不同),
   但至少不能再当通用结论。
2. 退化 deploy + 无 ensemble(67%)高于干净 deploy + 无 ensemble(50%),配对
   8:16,p=0.15:planner settle 的起点摆放和退化 WBC 的惯性超调对这个策略是
   "帮忙"的。这不是可迁移的收益(真机没有 planner,也没有退化态)。
3. test-time RTC 在干净 deploy 上:rtc100 **14/64**(22%,提前停止),失败
   36 局到位不合爪、14 局抓了没抬——冻结前缀 + ramp 压制了合爪这类突变动作,
   负收益。rtc-train(RTC_TRAIN_P 0.5, KMAX 20,flow200 数据)+ 干净协议
   `--rtc`:rtctrain200_rtc **1/50**(2%),最小距离中位 0.40 m。test-time RTC
   + ensemble N3 M0.3:rtc_ens100 **16/72**(22%),配对 base100 30:10 p=0.002,
   clean100 29:10 p=0.003。裁定(2026-09-03):在"token 经 frozen WBC 执行"的
   链路上 RTC 的前提(已发出的 chunk 被如实执行)不成立,前缀 hard-mask 锁住
   的是 token 不是动作;RTC 两种形式都是负收益,不再投入。
4. lockstep(干净、经验延迟)lock100 **32/100**,与 clean100 配对 29:15,
   p=0.049,Δ14 点,校准未过;失败签名仍是停远(50/68 局 >0.35 m)。见
   docs/lockstep.md。

### 2026-09-03 裁定:dart200 在干净协议下(2×2 补全)

dart200 ckpt-10000,干净协议,seed 6000,spawn 1.1–1.3,与 flow200 两臂逐轮配对:

| checkpoint | 无 ensemble | ensemble N3 M0.3 | 配对(base vs ens) |
|---|---|---|---|
| flow200 | base100 **50/100** | clean100 **46/100** | 22:26,p=0.67 |
| dart200 | dart200_cbase **24/100** | dart200_cens **56/100** | 7:39,p<0.0001 |
| 配对(flow200 vs dart200) | 37:11,p=0.0002 | 20:30,p=0.20 | |

结论:

1. **"远停 + 走近再抓"(DART)采集机制没有带来更高的成功率**。不带 ensemble
   时 dart200 比 flow200 低 26 点(p=0.0002);带 ensemble 时高 10 点但不显著
   (p=0.20)。
2. **ensemble 的解锁效应是 dart200 这份数据特有的**(+32 点,p<0.0001),在
   flow200 上为零。机制与 08-31 的诊断一致:DART 数据教会策略在桌前停顿,
   ensemble 把它拽出不动点;flow200 没有停顿可拽。ensemble 不是通用部署增益,
   是对 DART 副作用的补偿。
3. 失败构成:dart200_cbase 的 76 局失败里 71 局是 reach 停滞,最近距离中位
   0.38 m、20 局 ≥0.45 m(flow200 base100:0.35 m、1 局);dart200_cens 剩下的
   44 局失败同样是远停(中位 0.41 m,16 局 ≥0.45)。
4. 当前正式基线仍是 flow200 + 干净协议 + 无 ensemble(50/100)。**该数字测于
   2026-09-02/03,早于 `ReachConfig` 引入,跑在旧的站位采样带上(post_min
   0.22 / azimuth 36.5),ready-grasp 线自 09-07 起是 0.30 / 42.0**——flow200
   的 2×2 内部比较不受影响,但它不能作为 09-07 之后任何 arm 的对标参考。
   下一步的数据改进方向是新采集协议(去停顿、慢速 reach)重采一版做单变量
   对照,而不是回到 DART。

### 2026-09-03:失败细分(`src/locomani/tasks/bottle/failure_taxonomy.py`)

离线跑在 traces 上,每局落到唯一叶子(A 导航 / B 到位未合爪 / C 合爪未抬起 /
D 抬起后丢 / E 摔 / F 其他,共 16 个叶子,定义见脚本头注)。旧 trace 没有瓶子
姿态和接触体,"撞倒"用瓶心 xy 位移 ≥ 5 cm 代替;用 base100 验证过这个代理:
50 局成功里 0 局在合爪前有 ≥5 cm 位移,50 局失败里 39 局有(中位 11 cm),
且视频帧确认是瓶子被伸过去的右手扫倒。新 trace 已加 `obj_tilt`(倾角)、
`touch`(碰到瓶子的手部 link)、`obj_z`,之后可以分左右手。

| 叶子 | base100 | clean100 | dart200_cbase | dart200_cens | rtc100 | rtc_ens100 |
|---|---|---|---|---|---|---|
| 失败局数 | 50 | 54 | 76 | 44 | 50 | 56 |
| A1 停远 | 1 | 0 | 16 | 13 | 1 | 1 |
| B1 站太远(0.36–0.45) | 1 | 5 | 14 | 13 | 2 | 2 |
| **B4 伸手撞倒瓶子** | **38** | **42** | 33 | 9 | 32 | 39 |
| B5 犹豫 | 4 | 0 | 2 | 0 | 1 | 2 |
| B6 桌前停滞 | 0 | 3 | 6 | 6 | 0 | 5 |
| C1 合空 | 0 | 3 | 0 | 0 | 1 | 2 |
| C3 合爪时撞倒 | 6 | 1 | 2 | 2 | 12 | 5 |
| 其他 | 0 | 0 | 3 | 1 | 1 | 0 |

结论:flow200 策略在干净 deploy 上的失败 **76–78% 是第一次伸手就把瓶子扫倒**,
再加合爪时撞倒,碰倒类合计占 88% / 80%;瓶子倒了之后策略从不恢复(桌前停滞到
超时)。dart200 的失败则一半在导航段(停远 + 站太远),碰倒只占 43% / 20%。
这和采集侧量到的机制一致(WBC 手臂落后参考 0.2 s × 速度,快速伸手时手先于
参考 5–6 cm 撞上瓶子):smooth200 的慢速 reach 正是针对这一类。

### 2026-09-04:smooth200(慢速 reach、去停顿数据)首轮裁定

smooth200 ckpt-10000(180 局,loss 0.0482,与 flow200 同配方),干净协议,
seed 6000,出生 1.1–1.3 m:

| 臂 | 结果 | 配对 |
|---|---|---|
| smooth200_base100(无 ensemble) | **17/100** | vs base100(50)42:9,p<0.0001;vs dart200_cbase(24)21:14,p=0.31 |
| smooth200_ens100(ens N3 M0.3,进行中) | 33/59 | vs smooth200_base100 同 60 局 6:29,p=0.0001;vs clean100 同 60 局 15:20,p=0.50 |

失败细分(base 臂,83 局失败):B4 伸手撞倒 68(82%)、B5 犹豫 8、C3 合爪时撞倒
3;71 局倾倒里 42 局倒前 1 s 有右手接触记录,29 局手部 geom 无接触。

结论(首轮):**把专家伸手速度减半、消掉采集碰倒,没有让策略少撞瓶子,反而把
基线从 50 打到 17**;撞倒仍是 82% 的失败。ensemble 在这份数据上又变成解锁器
(16% → 56%),回到 flow200+ensemble 的水平,与"流畅数据上 ensemble 无效"的
预测相反。数据形状的变化改变了 ensemble 的作用方向,部署旋钮的结论再一次被
证明是 checkpoint 条件化的。

#### 2026-09-04:专家 reach token 开环回放(诊断,20 局)

问题:策略伸手撞倒瓶子,是 token→WBC 执行不保真,还是策略定位不准?做法:把
机器人传送到录制的骨盆位姿、瓶子放到录制位姿,从"到瓶子附近后底盘静止的第一帧"
起把专家的 64 维 token + 右手关节以 25 fps 原样喂给 deploy。四个版本,结果一致:
**0/20 抓起**,11–17/20 撞倒(其中 3–6 局在传送落位瞬间就倒,是放置伪影)。在
录制的合爪帧比较回放与录制(骨盆系):

- 手掌位置误差 0.1–5 cm,中位 ~2 cm,方向一致地偏左(+y 1–5 cm):**手臂 token
  的执行是忠实的**。
- 瓶子相对骨盆的位置误差 5–50 cm 以上:**底盘的开环执行不可复现**(碎步、
  下蹲后移的落点随起步状态与负载变),手到了录制里的抓取位,瓶子不在那里。

裁定:撞倒不是手臂通道的问题;开环 token 执行在底盘上有 5–50 cm 的落点不确定性,
任何以 0.4 s chunk 开环执行的策略都要靠视觉在每次重规划时把这个误差补回来。


## 37. 把单臂出生对齐到双臂那条线:横向归零是确定的,13/20→16/20 不是(2026-09-19, 本 session)

**为什么跑**:用户看第三视角视频时发现,双臂从第 0 秒就正对篮子开始执行,单臂却先
没对准桌子、又调整很久,要求把单臂初始设置改成和双臂一致。

**两条线随机化的是相反的东西**,这是差异的根源,不是缺陷:

| | 单臂(bottle) | 双臂(crate) |
|---|---|---|
| 机器人朝向 | ±20°(`spawn.yaw_err_deg`) | **固定** `"yaw": 0.0`(`tasks/crate/expert.py:57`) |
| 机器人横向 | ±0.2 m(`spawn.lateral`) | 无,始终在接近轴上 |
| 机器人距离 | 1.1–1.3 m + 0.2 抖动 | 1.2–1.5 m |
| 物体 | 基本固定 | 篮子 ±5° yaw + 2 cm 位移 |

双臂随机化**目标物**,单臂随机化**机器人**。所以双臂永远不用转身。

**A/B**:同 seed0 40000,batch_rounds 10 × max_batches 2,只改 `task.spawn`
(`yaw_err_deg` 20→0、`lateral` 0.2→0、`edge_jitter` 0.2→0)。

| 臂 | 配置 | 成功 | 平均 \|横向\| | 失败细分 |
|---|---|---|---|---|
| 原 | `single_arm_20ep_20260919` | 13/20 | 0.085 m | 髋/桌 6(右 5 左 1);撞倒瓶子 1 |
| 对齐 | `single_arm_20ep_aligned_20260919` | **16/20** | **0.000 m** | 髋/桌 4(右 3 左 1) |

**裁定,分两半**:

- **确定的**:横向偏移精确归零,机器人从第 0 秒正对瓶子。逐帧看 ego 视图,原配置
  第 3–4 帧桌子完全离开视野(机器人在转身),对齐配置桌子第 1 帧居中且逐帧变大
  (笔直走向目标)。这是几何性质,不需要统计。
- **不确定的**:**13/20 → 16/20 不构成改进证据**。本线 20 局分辨不了成功率差异
  (见 [[ten-round-pilots-cannot-measure-this-line]] 同源判断),这个差距在噪声内。

**失败形态完全没变**:两组失败全是 `[grasp] simple clearance: hip_roll_link:
dining_table`。出生朝向不是这些失败的原因,**站位深度才是**;改 spawn 不解决它。

**顺带发现一个交付缺陷**(已修):`scripts/collect_bottle.py:84` 把
`--video-include-init` **无条件硬编码**传给 gym,而它左右每一个邻居都是按配置条件
传的,于是 `task.video_include_init: false` 从来没有送达过。后果是单臂第三视角视频
一直在拍 ready 姿势前奏,而双臂在 `reset_ready()` 里做完全部初始化才按录制键
(`tasks/crate/expert.py` 注释:"frame 0 of the episode is the task")——用户看到的
"调整很久"有一部分就是这个。eval 路径(`pipeline/eval_closed_loop.py:296`)写法是
对的,只有 collect 这条错了。又一例"配置声明了但没传下去"。

**没有采纳的那一半**:用户同时要求"初始姿势也跟双臂相同"。**不可行,且已试过**——
crate 式行走准备在这张桌子上是 2/20,见 `docs/single_arm_design.md:127` 与
`handoff/bottle-hanging-start-20260914.md#seventh-step-...`。原因是几何硬冲突:
dex1 手指自掌心下伸 19 cm,垂臂指尖 0.53 m,低于 0.62 m 桌面。双臂没这个问题是因为
它面对 30 cm 低桌、双手伸进篮子。单臂 expert 仍冻结在 `stage4_27_wrap`。

## 38. 按 setup.md 编出的 deploy 不摔,但抓取掉到 10/17(2026-09-24)

**问题**：协作者报采集时机器人摔倒，用户怀疑我们自己编过几次 SONIC deploy 却没同步到文档。静态核对（TODO 4.8.1）：
本机所有 deploy 都来自只在本机的 `lockstep-port` 分支（上游 1983e888 + 6 个未推送提交 + 工作树补丁）；setup.md
的做法（clone 上游 main 不钉版本 + 4 个补丁）本机从没编过。

**做法**：上游 b042411（当天 main）按 setup.md 第 2/3 步打补丁（vla_action_space 的 data 部分、exporter_priv_seq、
pose / hand ownership）并编译，sha 472ed2b，放在 `outputs/deploy_docbuild_20260924/`（上游 sparse worktree）；
权重复制自本机（与 HF 逐字节一致），不带 TRT 缓存。与现用二进制（5cab2fb，`bottle_continuous_20260914/deploy`）
同一份实验配置（`docrepro_20260924_bottle` 只改 name，`train.holdout` 10→0 以通过当天新加的准入检查，不影响采集），
同 seed，instance 0，GPU1；同 GPU 的对照臂在其后立刻重跑。比较脚本与结果 `outputs/deploy_docbuild_20260924/`。

| 臂 | 二进制 | 局数 | 成功 | 摔倒（骨盆 z < 0.4） | 骨盆最低 z | 髋碰桌 | 抬起时丢指垫接触 |
|---|---|---|---|---|---|---|---|
| 对照（复现 agent，GPU0，13:38） | 5cab2fb | 12 | 10 | 0 | 0.720 m | 1 | 0 |
| 对照（GPU1，文档臂之后） | 5cab2fb | 11 | 10 | 0 | 0.720 m | 0 | 0（另 1 局 lost pad contact） |
| 文档版 | 472ed2b | 17 | 10 | 0 | 0.718 m | 5 | 2 |

- **摔倒没有复现**：三臂 40 局开局全部站稳（stand drift ≤ 0.18 m），全程骨盆最低 0.718 m。
- **文档版抓取明显变差**：成功 10/17 对 20/23（Fisher p = 0.066），髋碰桌 5/17 对 1/23（p = 0.067）；
  同 seed 的前 6 局，两个对照臂 6/6 与 6/6，文档版 2/6。刹车停位几乎相同（骨盆离桌 x 均值 0.383 对 0.389 / 0.390 m，
  刹后滑行 0.053 对 0.046 m），差别不在接近段，而在伸手 / 下蹲段。
- 两个二进制在采集路径上的已知源码差别：本地提交（DDS 环境变量、planner 初始化超时 10 s、auto-start、wbc_token、
  lockstep 代码）和上游基点之后的提交（增益缩放默认不生效、观测别名、planner 参数校验）。**哪一处造成了抓取差别未定位**，
  样本也只到 p ≈ 0.07。

**结论**：文档与本机确实不一致，而且不一致有后果（抓取变差），但单实例 bottle 采集上它不产生摔倒；协作者的摔倒
需要他的 run 证据（TODO 4.8.2）。

## 39. 默认走近改成守线刹车 + 原地抬手（2026-09-25）

用户要求 bottle 按 warehouse 的方式走近：手臂交给步态自然下垂、全程 0.25 恒速、按桌边守线刹车停得越近越好、
停后双手原地抬起再抓。实现为 `reach.simple.approach_stop: guard`（`tasks/bottle/edge_guard.py`、
`SimpleGraspPolicy._lift_ready`），当天由用户定为默认。四个 run 同一份实验配置、同一组 seed 40000，instance 0，GPU0。

| run | 走近 / 抬手 | 局数 | 成功 | 停后骨盆离桌边 | 停后手离桌边 | 刹后滑行（守线预留 0.20） | 每局用时 |
|---|---|---|---|---|---|---|---|
| `bottle_simple_10ep_20260925` | 旧：张开下垂、三档速度、可见深度刹车、后退再竖直抬 | 11 | 10 | 0.12–0.31 m | −0.03…+0.10 m（10/11 局后退） | — | 36–40 s |
| `bottle_simple_guard_20260925` | 守线，方向逐帧修正 | 2 | 0 | 0.51–0.53 m | 0.35–0.37 m | 0.19 m | — |
| `bottle_simple_guard_straight_20260925` | 守线，方向起步定死；笛卡尔竖直抬手 | 10 | 9 | 0.24–0.40 m | 0.07–0.22 m | 0.13–0.17 m | 34–44 s |
| `bottle_simple_guard_lifted_20260925` | 同上；warehouse `lifted_ready` 关节插值抬手 | 10 | 10 | 0.25–0.40 m（中位 0.34） | 0.08–0.22 m | 0.14–0.17 m | 39.5–44.3 s |

- **方向逐帧修正让步态停走**：deploy 对移动方向做精确比较，方向一变就重新规划（10 Hz）；那 2 局步态是 15 cm →
  双脚站 1.3–1.9 s → 34 cm，守线只能按 35 cm 预留，两局都 `[stand] arrival left the object out of reach`。方向起步时
  定死、直指规划站位后，落脚间隔 0.24–1.18 s、无停顿（两轮不是同一 run，第一轮只 2 局）。
- **竖直抬手让肩往外张**：手留在下垂处（离中线 21 cm）竖直抬到 0.78 m，逐帧 IK 只能靠肩外展，抬完肩外展 36–74°、
  肘比肩外 9.4–17.5 cm（用户看视频指出）。换成 warehouse 的 `lifted_ready`（工具点骨盆前 16 cm、侧 16 cm、上 5 cm，
  掌心下倾 30°，肘下限 −60°，关节空间插值 2.5 s）后肩外展 18–23°、肘外 4.6–6.0 cm（走路下垂时 5 cm）。
- **桌面层最靠前的是下垂的 Dex1 手指**（10 局里 8–9 局），在骨盆前约 16 cm；所以骨盆比旧走法停得远（中位 0.34 对 0.18 m），
  但手在桌边后面、不再后退。停位散布来自步长（每步 0–19 cm），与 warehouse 的结论相同。
- **未解决**：`lifted_ready` 插值中途手指先前扫 6–8 cm（参考轨迹约 3 cm + 跟踪滞后 4–5 cm：抬到 40% 时右肩后摆少
  0.06–0.13 rad、肘少折至多 0.14 rad）；最近两局抬手时离桌边 6 mm / 1.2 cm，0 越线。守线余量没有算这一段，收紧守线前要先处理（TODO 5.4.1.1、5.6）。
- 预览：`outputs/videos/bottle_simple_guard{,_straight,_lifted}_20260925/`；逐局记录在 manifest 的 `simple.guard`、`simple.ready`。

## 40. skills 目录的 approach → grasp_single 经 plan 入口在真栈上跑：9/10（2026-09-28）

入口 `scripts/plan_episode.py`（TODO 7.6）：起栈（instance 1、GPU1、无 exporter）→ prime → 每局按规划站位 teleport（`stand_plan`，
无出生抖动，站位后 1.2 m）→ `skills.Executor` 跑 JSON plan → 判据 → `skills/manifest.jsonl` + `agent_trace_NN.jsonl`。
profile `bottle_simple`（= `bottle_simple_200.yaml` 的 `reach.simple` + guard），场景 `sim_lowcom`，deploy 根 `outputs/bottle_continuous_20260914/deploy`。

| 运行 | 局数 | OK | 失败 | approach 停位（对物体深度 / 横向） | 每局用时 |
|---|---|---|---|---|---|
| `skills_smoke_bottle_20260928b`（seed 0–2） | 3 | 3 | — | 0.42–0.46 / −0.18…−0.22 m | 31–34 s |
| `skills_bottle_pilot10_20260928`（seed 100–109） | 10 | 9 | `[grasp] simple bottle knocked before the close` ×1 | 0.34–0.49 / −0.05…−0.23 m | 29–33 s |

- 对照：同一条线的采集入口 `bottle_simple_200` 前 10 个 seed 10/10（§39）；10 局批次本身有 ±2 局波动，9/10 在噪声内。
  失败族"合爪前碰倒瓶子"是这条线已知的（§39 的 lifted 版本也出现过）。
- 每个 primitive 的时长稳定：approach 8–14 s，grasp_single 12–14 s；失败局 grasp 8 s 就退出。
- 第一次跑 `skills_smoke_bottle_20260928` 3/3 失败在 spawn（"did not stabilise after reset" + "no body_q_target"）：
  shell 里 source 了 `scripts/paths.sh`，它在 `INSTANCE` 未设时导出 instance 0 的 `TF_*` 端口，agent 进程对着 5556 发命令而
  deploy 在 5566。入口现在先按 `--instance` 把整块端口钉进环境。其他会话的 instance 0 采集在我起栈前 30 s 已结束，没有互相干扰。
- 这两次运行没有录像（`SIM_VIDEO_DIR` 补丁第一次没打上）；入口从下一次运行起每局 `start_video/end_video`。

## 41. `carry_to`：持瓶行走的漂移测量与实现（2026-09-28）

`carry_to` = 上身冻结（PLANNER_FROZEN_UPPER_BODY，手保持闭合）+ `Primitives.walk_to` + 每拍持物守卫（单臂：掌系偏移相对抓取时的漂移 < 3 cm
且该手接触物体；双臂：物体高度 ≥ 抓取高度 + 6 cm 且双手接触，接触闪断容忍 0.3 s）。plan：approach → grasp_single → carry_to，
`bottle_simple`，`sim_lowcom`，GPU1 / instance 1，seed 400。

| 运行 | 走法 | 局数 | OK | 走过 | 最大掌系漂移 | 漂移中位 |
|---|---|---|---|---|---|---|
| `skills_carry_back1_20260928` | 朝向不变，倒退 1 m | 3 | 3 | 0.94–0.98 m | 0.29–0.81 cm | 0.1–0.4 cm |
| `skills_carry_back2_20260928` | 朝向不变，倒退 2 m | 3 | 3 | 1.99–2.05 m | 0.29–0.71 cm | 0.1–0.4 cm |
| `skills_carry_turn1_20260928` | `walk_to(face_target=True)`：边走边把朝向翻 180° | 3 | 0 | 0.06 m | **3.0 / 3.2 cm**（另 1 局抓取失败） | — |
| `skills_carry_turnto1_20260928` | 先 `turn_to(180°)` 原地转（IDLE），再前行 1.5 m —— 站位朝向 −90°，**实际转 90°** | 3 | 3 | 1.44–1.46 m | 0.37–0.50 cm | — |
| `skills_carry_default15_20260928` | `carry_to(relative=[−1.5,0])` 内部先 IDLE 原地转 **180°** 再走 | 3 | 0 | 0.26 m / 转中丢 | 3.2 cm / 转中丢（另 1 局抓取失败） | — |
| `skills_carry_turn2x90_20260928` | 两次 `turn_to` 各 90°（IDLE），再前行 1.5 m | 3 | 1 | 1.43 m | 0.35 cm；另 1 局第二次转完已 3.9 cm、1 局抓取失败 | — |
| `skills_carry_turn180slow_20260928` | `turn_to(180°, mode=slow_walk)` 一次转，再前行 1.5 m | 3 | 1 | 1.46 m | 0.55 cm；另 2 局在 approach / grasp 失败，没到 carry | — |
| `skills_carry_turn180slow6_20260928`（seed 600–605） | 同上 | 6 | 3 | 1.44–1.50 m | 0.31–0.91 cm；到 carry 的 3 局全保住；另 3 局在 approach（站位远 ×2）/ grasp 失败 | — |

- 直走和倒退不掉：1–2 m 漂移 < 1 cm，和 2026-09-03 M0 的 0.4–1.3 cm 一致；`PALM_DRIFT_MAX` 3 cm 作为行走守卫余量够。
- 掉的是**转向**，且和转幅有关：90° 原地转 3/3 不掉；180° 转持瓶到 carry 的 5 次里 2 次保住（IDLE 一次转 0/2、两次 90° 1/2、SLOW_WALK 1/1），
  丢的都在转的过程中或转完的第一步（掌系漂移 3.2–3.9 cm）。`walk_to(face_target=True)` 边走边翻朝向最差，0.4 s 内 0.01 → 3.0 cm。
- 因此 `carry_to` 默认 `turn_first=True`：目标方位与当前朝向差 > 0.2 rad 时先 `turn_to`，再 `walk_to(face_target=False, face_yaw=方位)`；
  `turn_mode` 默认 `slow_walk`（180° 转到 carry 的 4/4 保住，漂移 0.3–0.9 cm；IDLE 一次转 0/2、两次 90° 1/2），`face_target=True` 保留为原始行为（已知会滑）。
- 转向段没有守卫（`turn_to` 无回调），转完立刻查一次持物；转中丢的会报 `[carry] load lost while turning`。
- 这一天的 approach 失败偏多（seed 600–605 里站位远 ×2），是这条线自己的 `[stand] arrival left the object out of reach`，与 carry 无关。
- 预览：`outputs/videos/skills_carry_20260928/back2_round00.mp4`（305 KB）、`turn1_round00.mp4`（265 KB）。原片在各运行目录 `videos/`。
