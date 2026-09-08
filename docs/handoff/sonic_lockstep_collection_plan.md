# SONIC 采集的 lockstep 化:可加速、逐帧可复现,wall-clock 路径零改动

> 状态(2026-09-08):设计稿,**搁置**(用户决定先不投入,可能之后回来做)。未开工。前置调查见本文"现状"节,来源是
> `docs/lockstep.md` 与对 `lockstep_sim.py` / `vla_round.py` / deploy 补丁 /
> `run_data_exporter.py` / expert 各文件的逐行核对。

## 目标与约束

- `LOCKSTEP=1` 时,scripted expert 采集(`grasp_gym.py --record`、`bimanual_wall.py`)
  由仿真 tick 驱动:节奏由算力决定,同 seed 逐帧复现。
- `LOCKSTEP` 未设时,**每一行现有代码路径不变**:所有改动在 `sim_lockstep_`
  (C++)或 `lockstep_sim.enabled()`(Python)之后;wall-clock 采集的 manifest、
  parquet、训练脚本逐字节不受影响,用回归对照证明,不靠口头保证。
- 记录的数据格式与 wall-clock 完全一致(同一个 exporter 写),`check_dataset.py`
  四段审计通过,并与 wall-clock 同 seed 局逐列对账。

## 现状:现成的 lockstep 为什么不能直接用于采集

| 环节 | 现状 | 对采集的影响 |
|---|---|---|
| deploy `--sim-lockstep` | 不创建 Input/control/planner 线程,`LockstepControlLoop` 每 tick 跑一次 `Input(); ControlOnce()`;`PrepareLockstepMode()` 强制 STREAMED 模式 | expert 从走路到手臂全程 planner 模式(`planner_mode`、`upper_body_position`),lockstep 下 planner 不存在,`upper_body_position` 落空;发 `command(planner=True)` 会进 100 ms 睡眠循环,10 s 后 `operator_state.stop` |
| lockstep RPC(`lockstep_sim.py`,REP :5583+10·I) | 请求只认 `{"token":[64],"left":[7],"right":[7]}`;`_send_token` 以 v4 pose 消息带 envelope 重发;deploy 的 `DecodeLockstepBundle` 必须有 `token_state`/`frame_index` | planner 帧、v1 `pose_chunk` 都没有 envelope 表示,进不了 gate |
| expert 节奏 | `time.sleep(DT)`、`time.time()` 超时遍布 `primitives.py`/`stream_reach.py`/`upper_ref.py`/`grasp_gym.py`/`bimanual_wall.py`;`LockstepClock` 的后台泵只是让睡眠代码"看到世界在动",每 20 ms 一 tick,仍是实时 | 不改 expert 的计时就没有加速 |
| exporter(`run_data_exporter.py`) | 纯墙钟 50 Hz 采样最新消息;`teleop.planner_*` 有 ≤200 ms、SMPL ≤100 ms 的墙钟新鲜度门;图像来自 :5555 子进程按自己的墙钟重发共享内存 | 仿真快于实时就丢 tick,慢就重复行;planner 字段被新鲜度门清零;图像与状态不同步 |

## 设计(五段,每段可独立验收)

### 1. deploy:planner 在 lockstep 里按 tick 走(C++,只动 lockstep 树)

- `sim_lockstep_` 下不再 `PrepareLockstepMode()` 强制 STREAMED;保留 ZMQManager
  的正常模式机(command/planner/pose 三个订阅线程本来就还在)。
- 把 planner 线程体拆成 `PlannerOnce()`;`LockstepControlLoop` 每 tick 顺序为
  `Input(); PlannerOnce(); ControlOnce(); Commit`。`planner_dt_` 的推进改为
  每 tick 固定步长(与 wall-clock 的 50 Hz 期望值一致)。
- `handle_input` 里进 PLANNER 的 `sleep_for(100ms)` 等待和 `PLANNER_INIT_TIMEOUT`
  在 `sim_lockstep_` 下改成同步就绪检查(planner 状态在同一线程里,当 tick 就绪)。
- 盘点 planner 内部所有 `std::chrono` 用法(motion library 计时、mode 切换的
  cross-fade 8 帧、heading anchor),逐个换成 tick 计数;这是本段最大的未知量。
- gate:`DecodeLockstepBundle` 接受三种 bundle:token(现状)、planner 帧
  (`build_planner_message` 全字段 + envelope)、v1 pose 帧(`joint_pos` + envelope);
  释放规则不变(state envelope 带 NO_ACTION,或匹配的 bundle 已到)。

### 2. sim RPC:帧类型扩展,状态随 reply 返回

- `frames[i]` 增加互斥的 `planner: {...}` / `pose: {...}` 键;`_send_token` 泛化为
  `_send_frame`,按类型带 envelope 重发到 deploy。
- reply 增加 `priv`(privileged 字典,`update_reward` 里已经算好)和 `seq`,expert
  在 lockstep 下不再走 :5581 的 CONFLATE 订阅(去掉一处不确定性)。
- `LOCKSTEP_WARMUP_TICKS` 等现有旋钮不动。

### 3. expert:时钟抽象,墙钟量全部换成 tick

- `primitives.Clock`:`now()`/`sleep(dt)`/`tick()`。wall-clock 实现就是
  `time.time`/`time.sleep`;lockstep 实现把 `sleep(dt)` 变成 `round(dt/DT)` 次
  RPC step,`now()` 返回 `ticks*DT`。
- `AgentCommander.frame()`/`pose_chunk()` 在 lockstep 下不发 PUB,而是把帧放进
  下一次 step 的 `frames`(现有 shim 是直接丢弃,那是 VLA 评测的选择,采集不能丢)。
- 机械替换清单(行号见调查报告):`primitives.py` walk_to/turn_to/reach/grasp 的
  超时与 sleep;`stream_reach.py` 两个 50 Hz pacer、`hold()`、stale 检查 `observed['wall']`;
  `upper_ref.py` 的 0.5 s 新鲜度门与有限差分速度;`agent_server.hold()`;
  `grasp_gym.py` do_approach / 各 settle;`bimanual_wall.py` 走路超时、stall 计时、
  CONTACT_STALL 历史。wall-clock 实现下这些替换是恒等变换,用 pilot 回归证明。

### 4. 记录:exporter 成为第三个 barrier 参与者

- exporter 加 `--lockstep`:主循环不再 `sleep(loop_period)`,而是等 :5581 的
  `seq` 变化后采一行,并在 `TF_EXPORTER_ACK_PORT` 回 ack(seq);sim 的 `_tick`
  在 deploy ack 之后再等 exporter ack。每 tick 恰好一行,慢也不丢、快也不重。
- 图像:lockstep 下 sim 在 tick 边界把渲染结果**同步**发到 :5555(与子进程同一
  消息格式),不再依赖子进程的墙钟重发;exporter 的图像订阅端不改。
- 新鲜度门(100/200 ms)在 `--lockstep` 下改为"本 tick 内收到";其余字段逻辑不动。
- 上游文件走 `patches/`,与现有 exporter 补丁同一机制。

### 5. 验收(必须全过,顺序固定)

1. **wall-clock 回归**:`LOCKSTEP` 未设,跑 12 局闸门 + 10 集双臂 pilot,manifest
   指标与改动前基线逐项一致;转换一份数据,parquet 与基线 `cmp` 逐字节相同。
2. **确定性**:lockstep 同 seed 两次,state trace 逐位相同。
3. **忠实度**:10 个 seed 成对 wall ‖ lock,expert 成功率、停位 stand-off/lateral/yaw、
   合爪触发、抬起高度都在 wall-clock 自身的轮间方差内。
4. **吞吐**:报告 sim 秒 / 墙钟秒。预期 3–5×(每 tick 一次 sim↔deploy 往返 + 一次
   ONNX 推理 + exporter ack),不是 ScaleBFM 的 20×+。
5. **数据**:lockstep 录 10 集,`check_dataset.py` 通过;与 wall 同 seed 局逐列对账
   (state/action/teleop.*/priv_seq);拿这 10 集做一次训练 smoke。

## 工作量与风险

| 段 | 估计 | 主要风险 |
|---|---|---|
| 1 deploy | 2–3 天 | planner 内部隐藏的墙钟;lockstep 树与 production 树的 `default_angles` 差异(校准时踩过)必须统一 |
| 2 sim RPC | 1 天 | 无 |
| 3 expert 时钟 | 2 天 | 替换遗漏一处就退回实时,用"lockstep 下禁止调用 time.sleep"的断言兜底 |
| 4 exporter | 2 天 | 图像同步发布改动上游;三方 barrier 的死锁处理 |
| 5 验收 | 2 天 | 忠实度若不过,和 VLA 校准时一样需要定向诊断 |

合计约两周。ScaleBFM 路线零成本拿到同样的性质,这是两条路线在采集吞吐上的
真实差距。
