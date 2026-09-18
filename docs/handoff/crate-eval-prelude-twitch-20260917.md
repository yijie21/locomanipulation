# 双臂评测开局跳变：原因与修法（2026-09-17）

用户观察：双臂闭环评测每局开始时机器人"抽搐一下 / 往前倾倒的大跳变"，采集时却由专家
丝滑接管。以下全部是同一天在 crate 场景（`crate_table30_depth04`、`crate_initial_token.json`）
上的实测，探针脚本见 `scripts/experiments/crate_prelude_twitch_probe_20260917.py`，
原始遥测在 `outputs/twitch_body/<case>/arm_trace.json`，预览视频在 `outputs/twitch_enc/previews/`。

## 结论

**在 deploy 收到第一个外部 token 之前，它站的是一个没人定义过的姿态（下称 Z），
第一个 token 落地时一步跳到 token 姿态；从第二局起不再发生。**
Z 是膝 0.67 rad、骨盆 0.755 m、肘 −0.29；token 姿态是膝 0.17、骨盆 0.788、肘 +0.83。
切换瞬间 0.3 s 内膝伸 0.5 rad、身高 +3.3 cm、俯仰摆 7.5°、肘 9 rad/s。这就是视频里那一下。

**Z 不是任何被指定的姿态**（2026-09-17 逐个排除）：
参考动画 `neutral_kick_R_001__A543` 的第 0 帧是腿关节近似全零、骨盆 0.790 m；
reset 用的 `DEFAULT_DOF_ANGLES` 是膝 0.30。两个都不是 0.67。
真正的来源在 deploy 源码 `GatherTokenState`（`deploy.cpp:1662`）：

    if (!is_using_encoder_) {
        // No encoder configured; use token_state_data_ (can be set externally)

评测栈以 `NO_ENCODER=1` 启动，`is_using_encoder_` 为 false，策略的 token 输入直接取
`token_state_data_`——在第一个外部 token 到达前那是个**全零缓冲**。所以 Z 就是
**策略对零 token 的输出**，不是谁写下的姿态。对照实验：同一段 IDLE hold，encoder 开着
是膝 0.16 / 骨盆 0.787。第一个 token 把缓冲填上之后就一直用它
（`is_using_encoder_` 不会再变回 true），这正是第二局起没有瞬态的原因。

修法：**每个 batch 先跑一局完整但不录像的热身回合**（`vla_rounds.warm_up`，与正式回合
同一条 `prelude` 代码路径，独立 RNG 流）。之后所有录制的回合：prelude 期间骨盆高度 0.788
不变、俯仰 [−3.8°, −3.7°]、膝 0.18、手臂 0.00 rad/s；落地阶段俯仰 [−4.1°, −1.8°]。

## 实测矩阵（round 0 = 该 deploy 进程的第一局）

| 配置 | IDLE hold 时姿态 | 切换时躯干 | 切换时手臂 |
|---|---|---|---|
| `NO_ENCODER=1`（评测现状） | Z 姿态（膝 0.67，planner 参考 +1.02 与关节 −0.29 差 1.33 rad，3 s 不变） | 膝 0.67→0.17，z +3.3 cm，俯仰 +3.7→−3.8° | 单调阶跃 9.3 rad/s，1.14 rad |
| 同上 + sim arm pin 渐进 | 同上（pin 期间腿自行站起：膝 0.66→0.29） | 俯仰 −2.3→**−9.1**→−5.2°，膝 0.31→0.45→0.17 | 1.6 rad/s |
| encoder 开 | 正常站姿（膝 0.16，参考差 0.18） | 俯仰 **−13.6 … +4.7°**，膝 0.16→0.26→0.11 | 往返 1.35 rad，**26 rad/s** |
| encoder 开 + pin | 同上 | 俯仰 −10.3 … +0.7° | 1.6 rad/s |
| `start_control` 后立即切换（落地中） | — | 俯仰 **+10.7 … −6.8°**，膝 0.86→1.02→0.12 | 19.6 rad/s |
| 先 `start_control`+token、不 reset 的"prime" | 无效：round 0 仍在 Z | 同现状 | 同现状 |
| **热身回合后（本修法）** | token 姿态（膝 0.18） | **俯仰 [−3.8, −3.7]°，膝 0.18，z 不变** | **0.00 rad/s** |

每个配置的 round 1 都干净——这是找到修法的线索。

## 机制（deploy 源码，`outputs/bottle_continuous_20260914/deploy/include/input_interface/`）

- `zmq_manager.hpp` ~268：`if (new_mode != active_mode_) TriggerSafetyReset()`，**无绕过开关**。
- `zmq_endpoint_interface.hpp` ~275–315：safety reset 清空外部 token
  （`has_external_token_state_ = false; external_token_state_.SetData({})`）、
  `operator_state.play = false`、`current_motion = 当前动作的副本("temporary_motion")`、
  `use_zmq_stream = false`。切换后那几个 tick policy 手上没有 token，于是又落回零 token 的 Z 姿态——
  这是 encoder 开着时也会摆 18° 的原因。
- `NO_ENCODER=1` 下 planner 模式的手臂参考永远送不到 policy（仓库早有记录：
  "NO_ENCODER=1 is token/VLA only; in planner-mode collection it silently drops arm + crouch
  references"），所以 IDLE hold 期间整个身体停在 Z。
- `start_control()` 发 `planner=True`，`enter_streamed()` 发 `planner=False`：每局其实切换两次。
  第二局起两次切换都无扰动，说明 deploy 一旦收到过 token，零 token 缓冲就不再生效。
  具体是 `current_motion` 被替换还是别的状态位，没有继续追——热身回合不依赖这个细节。

## 采集为什么丝滑

采集栈带 encoder，专家全程不离开 planner 模式：手臂 planner 默认 → `UpperRamp` 限速到 ready，
中间没有模式切换，token 缓冲也从不为零。评测用不了这条路（NO_ENCODER 下它是死的，而模式切换躲不掉）。

## 走过的弯路（都已撤回，别再试）

1. "deploy 播默认片段"——机制说对了一半，姿态说错了（注释里的 elbow +1.0 实测是 −0.29），
   而且在 NO_ENCODER 下根本不是切换瞬间的事，是切换**前**就停在那里。
2. `enter_streamed(prime=)`：只消掉切换前 30 帧空转造成的 0.6 s 停滞，阶跃不变。**保留**，因为它确实
   让首包落在切换内部。
3. `UpperRamp` 斜坡：走 planner→encoder 通道，NO_ENCODER 下是死路；种子取自 planner 参考（+0.985），
   和实测（−0.285）差 1.27 rad，斜坡期间机器人一动不动。
4. sim arm pin 渐进 + 穿过切换：手臂确实平了（0.145 rad 内），但腿和腰照跳。
5. encoder 开：起点变成站姿，但 safety reset 的无 token 窗口让躯干摆 18°。
6. 落地时切换：两个瞬态叠加，更糟。

## 顺手发现

`crate_initial_token.json` 的 `arms`（肘 +0.826）和 token 实际 hold 住的姿态（+0.69–0.74）差 0.13 rad，
所有配置的 end err 都是它。它是 prelude 末尾 0.30 rad 检查的基准，没动。
