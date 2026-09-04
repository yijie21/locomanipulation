# Lockstep 评测:仿真时钟 + 模拟延迟

> 状态(2026-09-02 晚):机制跑通、逐帧可复现;100+100 校准在 wall 9/17、lock 0/12
> 时停止,定向诊断找到根因(见"校准诊断"节)。校准需在两臂都"干净 deploy"
> 的条件下重做;在此之前 **lockstep 数字不单独报告**。

## 它是什么

wall-clock 评测里,sim 按墙钟走、deploy 按自己的 50 Hz 时钟跑、推理那 0.4 s
里机器人继续执行旧 chunk——延迟是行为的一部分,GPU 争抢会直接改实验条件。
lockstep 把世界改成"只在被命令时前进":

- deploy 用 SIMPLE 的 `--sim-lockstep` 补丁(已 cherry-pick 到我们的源码,
  分支 `lockstep-port`,构建在 `/workspace/deps/gear_sonic_deploy_ours_lockstep`):
  每收到一条带 envelope 的 LowState 只跑一个控制 tick 并 ack;
- sim(`lockstep_sim.py`,`LOCKSTEP=1`)用 RPC 驱动:agent 发"这些 token 帧,
  每帧 2 tick",sim 逐 tick 提交边界、等 ack、跑 4 个物理子步,最后把边界处
  同步渲染的相机图像随 reply 返回;
- agent(`vla_round.make_vla_policy_lockstep`,`--client lockstep`)把所有墙钟量
  换成帧数:观测间隔 K 与接入延迟 A 从 wall-clock trace 的经验分布采样
  (`tools/latency_dist.py` → `--latency-dist`),或 `--latency-frames K` 固定;
  ensemble 按帧号对齐;RTC 的 elapsed/frozen = 精确整数。

**模拟延迟**是关键设计:推理期间 sim 不是暂停,而是用旧 chunk 精确推进 K 帧
再接入——wall-clock 里真实发生的事,只是把"墙钟流逝"换成帧记账。

## 换来什么

- 完全确定性:同 seed 逐帧复现(实测三次轨迹逐位相同);
- 对 GPU 争抢免疫:并行度只受显存限制;
- 延迟可扫描:`--latency-frames 8/11/14/20` 直接画 robustness 曲线。

## 校准(必须先过)

flow200 ckpt-10000、seed 6000、spawn 1.1–1.3、ensemble N3 M0.3,同一份 deploy
源码:wall-clock 100 局(INSTANCE=1,server GPU0 :6591)‖ lockstep 100 局
(INSTANCE=0,server GPU1 :6590,K ~ 经验分布:观测间隔 p50 10 帧,推理 ~3 帧)。

通过 = McNemar p>0.05 ∧ 成功率差 ≤8 点 ∧ 失败构成各项差 ≤10 点。

14 局先导(同种子逐轮配对):wall-clock 8/14,lockstep 3/14(seed 版)、
2/14(不 seed)、2/14(时序抖动模型);不一致对 6:1,p=0.125。lockstep 的失败
几乎全是"到桌后手臂保持 ready 不伸手"的不动点(骨盆 10–30 s 内位移
<0.12 m,chunk 里 220 次预测无一合爪);到达时瓶子方位角更常落在正前/左侧
(策略只对右侧伸手)。同一天 wall-clock 自己也跑出过同种子 0/2,方差很大,
所以交给 100+100 定量。

## 校准诊断(2026-09-02)

配对 12 局:wall 6/12,lock 0/12。成功当且仅当到桌后的最小骨盆-瓶距 ≤ ~0.27 m
(两种模式都如此);lockstep 系统性地停远 ~0.1 m(中位 0.36 vs 0.25),落在
伸手包络外。逐项排除后,用同一段数据集行走 token(episode 0 帧 10–135,真值
位移 0.80 m)做开环回放,逐帧记录:

| deploy 状态 | 位移 |
|---|---|
| lockstep(任何二进制、加抖动/加指令延迟) | 0.65–0.70 m |
| wall-clock,production 二进制(Aug 17),planner settle 后进 streamed | 0.02–0.22 m |
| wall-clock,lockstep-port 二进制,同上 | 0.59 / 0.02 / 0.04 |
| wall-clock,SONIC_AUTO_START 直接接管,**从不发 planner 命令** | **0.68 / 0.71** |
| 同上,但接管后先做一次 planner IDLE hold 再进 streamed | 0.22 |

结论:**deploy 一旦经过 planner 模式再切到 streamed,walking token 的执行幅度
只剩 ~1/3**(policy 输入逐 tick 一致、kp/kd 一致、接触数一致,输出 cmd_q 的踝
关节摆幅却 2 倍、前进量 1/6;机制在 deploy 内部,尚未定位)。迄今所有
wall-clock 闭环评测(blocking/async/RTC、2×2、flow200)都经过 planner settle,
都是在这个退化状态下量的;lockstep 从不进 planner,所以"走得太好",到桌
时的停止动力学不同,停得更远。另外校准的两臂用了不同二进制(W 臂 production,
L 臂 lockstep-port,后者带 SIMPLE 改的 `default_angles` 肩 roll 0.2→0,这个偏移
本身也把 0.68 压到 0.44),这是校准设置的错误。

因此下一步不是继续校准,而是先定"干净 deploy"的评测协议:wall-clock 客户端也
用 SONIC_AUTO_START 接管、全程 token 保持站立(与 lockstep shim 相同),二进制
统一为 NVIDIA 默认角度的 lockstep-port 构建
(`/workspace/deps/gear_sonic_deploy_ours_lockstep_nvdef`);然后 wall‖lock 重做
配对校准。策略在"干净 deploy"上的成功率是未知数(同二进制小样本:wall 2/7、
0/4;lock 0/12),因为数据的行走是 planner 驱动、评测的行走是 token 驱动。
诊断脚本:scratchpad `diag_stepresp2.py` / `diag_stepresp2_run.sh`(变体
wall/wall50/lockstep/lockd1/lockd2/wallB/wallN/lockJ/lockNJ/wallNA/wallA/wallNP),
policy-input dump `--policy-input-logfile`,`lockstep_sim` 的 `LOCKSTEP_CMD_DELAY`,
privileged state 新增 `ncon/xfrc/ctrl_abs/cmd_q/kp/kd/tau`。

## 干净协议下的配对校准(2026-09-02 下午)

两臂同一 nvdef 二进制、同一策略、同一种子:wall-clock clean100 **46/100**,
lockstep lock100 **32/100**;不一致对 29:15,McNemar p=0.049,Δ=14 点 → **未通过**。
失败构成:wall reach 49 / close 5;lock reach 58 / close 9 / walk 1。lockstep 失败局
里 50/68 的最小距离 >0.35 m(wall 只有 8/54):"停远"签名在干净 deploy 上依然存在,
所以它不是 deploy 造成的,而是帧计数延迟模型漏掉了 wall-clock 里的某个真实因素。
候选(按可能性):每局 reset 开新 session 清空 deploy 观测历史;图像新鲜度(边界即时
渲染 vs 旧 1–2 帧);观测与图像同步方式;A/K 抽样与真实 worker 周期的相关性。

## 忠实度证据(已验证)

- 同一 token(站立/行走/伸手/合爪)在两种模式下 deploy 的身体响应逐位一致
  (`diag_lockstep_stand2.py` 家族);
- 图像:RPC 返回的边界渲染与 5555 流形状、通道、均值一致;
- 力矩:sim 施加的 ctrl = kp·(q_target−q)+kd·(dq_target−dq),逐关节对账通过;
- 行走阶段到桌距离 0.29–0.46 m,与 wall-clock 的 0.31–0.40 m 一致。

## 六个不报错的坑(bring-up 全过程)

1. **planner 在 lockstep 下会死锁**(planner 线程不存在,`ZMQCommandManager`
   等 10 s 后 stop deploy)→ 非 VLA 时段全用站立 token;commander shim
   (`install_lockstep_shim`)丢弃所有 `planner=True` 命令,`hold()` 变成精确 N tick;
2. **zmq 的 start 只在 planner 模式生效** → `SONIC_AUTO_START=1`(SIMPLE 同款);
3. **commander 是 bind 的 PUB** → sim 持有唯一 PUB,起 PULL 中继
   (`TF_LOCKSTEP_XSUB_PORT` = 5584+10·INSTANCE),commander 在
   `TF_SONIC_PUB_CONNECT` 下改为 PUSH;
4. **INIT ramp(3 s 纯 PD)会让机器人倾倒**,策略介入前;reset 之后由
   60 tick 运动学 warm-up(`LOCKSTEP_WARMUP_TICKS`)让策略先填满历史再松开
   → 站立 z 0.781,与 wall-clock 一致。**任何诊断必须先让策略介入(≥200 tick)
   再 reset**,否则量到的是倒地的机器人;
5. `--default-motion neutral_kick_R_001__A543`(否则 `current_motion_` 是舞蹈片段);
6. TensorRT 引擎缓存按工具链版本:每个二进制在自己的树里运行
   (launcher 会 `cd` 到二进制所在树)。

Shell 侧:`stop_stack.sh` 会按名字杀 `serve_policy.py`,任何命令行里含该字面量
的链都会被自杀——多步链一律写成脚本文件运行;server 必须在 `start_stack.sh`
之后起。

## 相关文件

`src/teleop_free/lockstep_sim.py`、`lockstep_vendor/`(SIMPLE 协议,MIT)、
`vla_round.py`(`LockstepLink`/`LockstepClock`/`make_vla_policy_lockstep`)、
`tools/latency_dist.py`、`tools/eval_closed_loop.py --client lockstep`、
`start_stack.sh`/`scripts/restart_deploy.sh` 的 `LOCKSTEP=1` 分支、
`diag_lockstep_stand*.py`。
