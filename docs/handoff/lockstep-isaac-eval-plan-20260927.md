# VLA 闭环评测：Isaac 渲图时冻结世界（lockstep + 模拟推理延迟）计划

状态（2026-09-27 晚）：**已实现**，W1、W2 验收通过；W0 / W3 / W4 的对照结果见第 10 节。原计划 W0 排第一，实施按 W1 → W2 → W0 → W3 → W4 → W5：W0 要在双臂 lockstep 客户端（W1）上跑，唯一的 checkpoint 用 Isaac 图训练，也要先有 W2。任务编号 [TODO](../../TODO.md) 4.7.11；相关的过渡修复是 4.7.6.1.2.11（blocking 客户端补帧），
一致性闸门是 4.7.9.8。Isaac 镜像本身的状态见 [镜像状态索引](isaac-vla-mirror-20260925.md)，lockstep 的原始设计与校准见
[archive/lockstep.md](../archive/lockstep.md)。

## 1. 问题（实测）

双臂闭环评测（`tasks/crate/vla_rounds.py`）是按墙钟跑的 blocking 循环。数据来自 `box_room_grasp_200_cap120_20260925`
checkpoint-5000 的 10 局（`eval/traces/round_*.json` 713 个 chunk、`eval/isaac_mirror.jsonl` 724 次渲染、`eval/state_trace/`）：

| 量 | 实测 |
|---|---|
| 一个周期 | p50 **810 ms**（p10 788，p90 820）；`eval.replan_interval` 这条线不读 |
| 执行段 | chunk 第 4–19 号动作共 16 个 × 40 ms = 640 ms；第 20–39 号从不使用 |
| 取图 + 推理 | p50 **166 ms**（p90 175，最大 257），其中 Isaac 三路 640×480 按需渲 p50 **95.5 ms**，推理约 70 ms |
| 跳帧 | `latency_index(166 ms)` → 新 chunk 从第 4 号起：713 个里 4 帧 634、3 帧 46、5 帧 32、6 帧 1 |
| 间隙里的指令 | 不发任何新帧；deploy 钳在旧 chunk 第 19 号 |
| 间隙里的身体 | 骨盆水平 2.9 cm/s、关节 0.066 rad/s（执行段 3.1 cm/s、0.101 rad/s）：物理和控制器照常在走 |

与现实相比有两处不符：

1. **多了约 95 ms 延迟。** 真实相机拍一张近乎瞬时（整条相机管线也是几十毫秒量级），Isaac 渲染的 95 ms 在现实里不存在，
   这段时间世界却照常演化。策略看到的图是旧的，动作到得更晚。
2. **间隙里参考冻结、之后跳 4 帧。** 跳帧假设"推理期间身体在执行那 4 个动作"，实际身体在追一帧定住的参考。
   训练数据里的参考是连续的。单臂 blocking 客户端 2026-08-31 修过同一问题（[台账](../ledger/bottle.md)「blocking 客户端三修复」），双臂没移植。

## 2. 目标：评测的时间模型与现实一致

| 环节 | 现实 | 现在（墙钟 + Isaac） | 目标 |
|---|---|---|---|
| 拍照 | 近乎瞬时 | 95 ms，世界在动 | **0**，世界冻结；可选加 C 帧相机延迟 |
| 推理 | 约 70 ms，世界在动，执行旧指令 | 70 ms，世界在动，参考冻结 | **L 帧**，世界按旧 chunk 连续推进 |
| 接入新 chunk | 从"已经过去的时刻"之后开始 | 从第 4 号开始（跳过的帧没人执行） | 从第 L（+C）号开始，跳过的正是间隙里执行掉的 |

原则：**只冻结渲染，不冻结推理。** 推理延迟在现实里存在，冻结它会让评测比现实乐观；渲染耗时是仿真独有的，保留它会让评测比现实悲观。

## 3. 为什么不能只暂停 MuJoCo

SONIC deploy 是独立进程，按自己的墙钟跑 50 Hz 控制循环，并有安全检查：LowState 超过约 50 ms 没更新就报
`Lost LowState data connection from robot!` 并退出。Isaac 一次渲染 95 ms，只停 sim 会先把 deploy 停掉。
要让世界整体静止，sim 和 deploy 必须一起按 tick 推进，也就是仓库已有的 **lockstep**。

## 4. 现有 lockstep：能直接用的和缺的

**已有（main 上的代码，2026-09-27 核对）**

- deploy `--sim-lockstep`：不起控制/planner 线程，每收到一条带 envelope 的 LowState 只跑一个控制 tick 并 ack。
  在 `patches/sonic_deploy_local_base.patch` 里，协作者按 [setup.md](../setup.md) 编的 deploy 也有。
- sim `LOCKSTEP=1`（`src/locomani/backends/sonic/lockstep_sim.py`）：RPC 驱动，每 tick 提交边界、等 ack、跑物理子步，
  并调用 `env.update_reward()`——特权状态照常发布、reset 照常轮询、state trace 照常记录。
  所以**冻结时特权流里最新一条就是边界状态，Isaac 镜像可以直接渲它**。
- 客户端 `src/locomani/backends/sonic/lockstep_client.py`（`LockstepLink`、`LockstepClock`、`make_vla_policy_lockstep`、
  `load_latency_dist`）；`EvalConfig.latency_frames` / `latency_dist` 模拟推理延迟：推理期间用旧 chunk 精确推进 K 帧再接入。
- 起栈：`backend.lockstep.enabled` → deploy `--sim-lockstep`、`SONIC_AUTO_START`、`--default-motion`（`stack/manager.py`）。

**缺的**

- 双臂评测器只实现 blocking：`eval_crate_closed_loop.PINNED = dict(client="blocking")`；lockstep 只有单臂 `eval_closed_loop --client lockstep` 在用。
- `EvalConfig` 在加载时拒绝 `image_source isaac` + `client lockstep`（理由是 lockstep 回复里带的是 MuJoCo 边界渲染图——实现没接上，原理不冲突）。
- lockstep sim 回复 `want_obs` 时会渲 MuJoCo 相机；Isaac 运行里 sim 是 `render: offline`，要改成只回状态。
- 双臂的 prelude（`place` / `prepare_token_pose` / `warm_up`）用 `time.sleep` 按墙钟推进，要改成按 tick。

**历史结论：lockstep 的校准没有通过**（2026-09-02，单臂 flow200）：同一二进制、同一策略、同一种子，墙钟 46/100、lockstep 32/100，
McNemar p = 0.049。lockstep 失败局里 50/68 停在离桌 > 0.35 m（墙钟 8/54），原因未查明。候选：每局 reset 开新 session 清空 deploy 的观测历史；
图像新鲜度（边界即时渲染 vs 墙钟里旧 1–2 帧）；观测与图像的同步方式；延迟抽样与真实推理周期的相关性。
在它被解释之前，**lockstep 下的成功率不能直接当作墙钟或现实的替身**，所以第一步是校准（W0）。

bring-up 时六个不报错的坑（archive/lockstep.md）在这里同样适用，尤其：lockstep 下没有 planner，任何 planner 命令会让 deploy 等 10 s 后停；
**必须先让策略接管 ≥ 200 tick 再 reset**，否则量到的是倒地的机器人；commander 是 bind 的 PUB，要走 sim 的中继端口。

## 5. 设计：一个周期怎么走

帧 = 一个 token（25 Hz，40 ms，2 个 50 Hz tick）。H = `execution_horizon`（16），L = 推理延迟帧数，C = 相机延迟帧数。

```
执行   send chunk[j .. j+H-1]          世界推进 H 帧（lockstep step）
观测   —— 世界冻结 ——                    读边界本体状态；请 Isaac 镜像渲当前状态（核对 seq = 边界 seq）
推理   —— 世界冻结 ——                    墙钟上花多久都行，新 chunk C' 到手
延迟   send chunk[j+H .. j+H+L+C-1]    世界按旧 chunk 继续推进 L+C 帧（补帧，参考连续）
接入   send C'[L+C .. L+C+H-1]          回到"执行"
```

- L 的取值：按实测推理 70 ms 取 `latency_frames: 2`；或 `latency_dist` 用墙钟 traces 里的推理耗时分布（要扣掉渲染的 95 ms，只留推理）。
- C 默认 0；部署硬件定了之后按它的相机管线延迟设。
- prelude 全部按 tick：站立 token 由 `LockstepClock` 推进，`hold(s)` 变成精确的 N tick；reset 之前策略先接管。
- 镜像接口加一个"期望 seq"：客户端带上边界 seq，镜像发现最新特权状态不是这一条就报错，保证图和状态是同一瞬间。

## 6. 工作项（按顺序，每一步单独验收）

| 编号 | 内容 | 位置 | 验收 |
|---|---|---|---|
| W0 | **校准 lockstep 本身**：用同一 checkpoint-5000、同一组 seed，让 lockstep 复刻墙钟现在的时间模型（冻结参考 ≈ 5 帧、跳 4 帧、图晚 95 ms），看能否复现墙钟的成功率与失败构成 | `vla_rounds` lockstep 分支的"墙钟复刻"模式 | 10 局先看失败构成是否一致，再定 100 局配对；不一致就先查第 4 节的候选原因 |
| W1 | 双臂 lockstep 客户端 | `tasks/crate/vla_rounds.py`（`LockstepLink` 推进、tick 版 prelude）、`pipeline/eval_crate_closed_loop.py`（放开 `client`）、`tasks/crate/prepare.eval_backend` | 站立 200 tick 后 reset、token 姿态到位；同 seed 两次（MuJoCo 图）轨迹逐位相同 |
| W2 | Isaac 接入 | `pipeline/eval_config.py`（去掉 lockstep + isaac 的拒绝）、`backends/sonic/lockstep_sim.py`（offline 时 `want_obs` 不渲 MuJoCo）、`backends/sonic/vla_obs.py` / `sim/isaac/mirror.py`（期望 seq） | 每个 chunk 记录的图 seq 等于边界 seq；镜像日志的状态年龄为 0 帧 |
| W3 | 延迟模型：L、C 与补帧语义 | `backends/sonic/lockstep_client.py` 的延迟采样、`vla_rounds` 的接入逻辑 | commands 日志里发出的帧号连续，无定住的间隙；跳过的帧数 = L + C |
| W4 | 对照 | 同 checkpoint、同 seed：墙钟（现状）vs lockstep（目标模型），各 10 局，再各 100 局 | 报告两边成功率、失败构成、周期计时；差异要能用时间模型解释 |
| W5 | 文档与配置 | `docs/warehouse.md` §6、`docs/rendering.md`、`docs/bimanual_design.md` §五、`configs/experiments` 模板 | 新读者照文档起 lockstep + Isaac 评测一次成功 |

过渡：4.7.6.1.2.11（墙钟 blocking 客户端补帧，间隙里续发旧 chunk 第 20 号起）可以先做，代价小，
能消掉"参考冻结再跳帧"，但消不掉多出的 95 ms。

## 7. 风险与未决

- **校准历史未过**（第 4 节）：W0 过不了，后面的数字都不能当现实的替身。
- **Isaac 渲染不是逐位确定**：同一状态两次渲染逐像素平均差 1.5–2.3/255（2026-09-27 录像回归），扩散头的采样也有随机性。
  lockstep 的"同 seed 逐位复现"只能在 MuJoCo 图下成立；Isaac 下只能报告分叉，不能声称逐帧复现。
- **按需渲与连续渲是否同图**（4.7.9.8）仍未验证；lockstep 下镜像仍是按需渲，这个问题不因 lockstep 消失。
- **协作者的 deploy**：lockstep 补丁在 setup.md 的补丁里，但没在双臂评测上跑过，要在 W1 用文档编出来的二进制实测一次。
- **相机延迟 C**：取决于部署硬件（相机型号、传输方式），未定。
- **速度**：lockstep 下每 16 帧（640 ms 仿真时间）花约 95 ms 渲染 + 70 ms 推理 + 仿真本身，大约实时或更快，不是瓶颈。
- 训练数据本身是按每行状态补渲的（相当于零延迟拍照），与目标时间模型一致，不需要重采。

## 8. 被否决的方案

| 方案 | 为什么不行 |
|---|---|
| 只暂停 MuJoCo sim | deploy 的 LowState 检查约 50 ms 就退出，渲染要 95 ms |
| 渲染和推理都冻结 | 现实里推理延迟存在，冻结它比现实乐观 |
| 只加补帧（4.7.6.1.2.11） | 参考连续了，但图仍晚 95 ms；作为过渡保留 |
| 降低渲染开销（减 settle 步、降分辨率） | 改变画质就与训练图不同源；也降不到接近 0 |
| 改 async 客户端 | 执行和推理重叠，但渲染的 95 ms 仍在世界时间里流逝，问题不变 |

## 9. 证据位置

- 周期、延迟、跳帧、间隙运动：`outputs/runs/box_room_grasp_200_cap120_20260925/eval/{traces,state_trace,isaac_mirror.jsonl}`；
  分析脚本 `eval/analysis/rounds_table.py`，台账 [crate.md](../ledger/crate.md) 2026-09-27 末节。
- Isaac 渲染耗时：同上镜像日志（按需 p50 95.5 ms）；数据集补渲 `isaac_dataset_render.log`（197 集中位 69.8 ms）。
- lockstep 设计、校准与坑：[archive/lockstep.md](../archive/lockstep.md)；单臂 blocking 三修复：[bottle 台账](../ledger/bottle.md) 2026-08-31。

## 10. 实施记录（2026-09-27）

### 改了什么

| 工作项 | 位置 | 内容 |
|---|---|---|
| 第 0 步（过渡） | `tasks/crate/vla_rounds.py` `GapFeeder` | 墙钟 blocking 循环在等图 + 推理时续发正在执行那块的后续帧（单臂 2026-08-31 的 gap feeder），`eval.gap: continue` 为默认，`hold` 保留旧行为 |
| 公共判分 | `vla_rounds.RoundScore` | 每块一次的判分、trace 行、后退、失败分类，墙钟与 lockstep 共用；墙钟的计时仍按墙钟秒 |
| W1 | `tasks/crate/vla_lockstep.py`；`pipeline/eval_crate_closed_loop.py`（`client` 接受 `blocking` / `lockstep`，下发 `gap` / `latency_*` / `camera_frames`）；`tasks/crate/prepare.eval_backend`（`lockstep.enabled` 随 `client`） | 按 tick 的局前准备（摆放 20 + 站立 150 + token 姿态 150 tick，第一次 reset 前先站 250 tick）；每次推理带 `policy.seed` |
| W2 | `pipeline/eval_config.py`（去掉 lockstep + isaac 的拒绝，单臂评测器仍拒绝）；`backends/sonic/lockstep_sim.py`（offline 时 step 回复不渲图、回复带 `priv_seq`、空闲时每 0.5 s 重发最新特权状态）；`sim/isaac/mirror.py`（`render` 可带 `seq`，只渲这一条）；`backends/sonic/vla_obs.py`（`build(images=, seq=)`、`settle_proprio`） | 图与本体状态取同一个冻结时刻 |
| W3 | `EvalConfig.gap` / `camera_frames`；`vla_lockstep.Latency`；`vla_lockstep latency-dist`（从墙钟 traces 生成分布，`--inference-only` 扣掉渲染） | 间隙续走旧块 L + C 帧，新块从第 L + C 号接入 |

bring-up 中查出并修掉的三处 lockstep 老化（2026-09-02 之后没人跑过 lockstep）：

1. `lockstep_vendor` 还在 `src/teleop_free/`，没人把它放进路径，lockstep sim 起来就 `No module named 'lockstep_vendor'`（唯一的测试自己加了路径所以一直过）→ 挪进 `locomani/backends/sonic/lockstep_vendor/`。
2. 起栈总给 deploy 传 `--planner-file`，`--sim-lockstep` 的 deploy 拒绝（`does not permit a planner model`）→ lockstep 下不传。
3. offline 的 lockstep sim 在第一个 tick 前不建特权发布器和 episode 服务器，而起栈等 `[episode] :`、镜像等第一条状态，第一个 tick 又要等它们 → 死锁；runner 开始服务前先建好并发一条，空闲时重发。

### W1、W2 验收

| 项 | 结果 |
|---|---|
| W1 站立 250 tick 后 reset，token 姿态到位 | 通过：热身局和每一局的 `[stand]` 检查都过，每次 reset 开新 session 并重新 prime |
| W1 同 seed 两次逐位相同（MuJoCo 图） | **不成立，原因已查明**：两次运行的局前 332 个 tick（含 reset、站立、token 姿态）qpos/qvel 逐位相同，第一次观测的本体状态相同、ego 图相同，只有左右腕图差 1 个灰度级，策略由此给出差一个量化步（1/16）的 token，之后分叉。策略服务端对同一观测 + 同一 seed 逐位相同（三次）。离线复现：MuJoCo 默认 4× 多重采样下同一状态在**同一进程**里渲三次，腕图就差 1–2 个灰度级；`vis.quality.offsamples = 0` 时进程内、跨进程都逐位相同。没有改渲染设置（会让评测图与训练图不同） |
| W1 每个周期 | 18 帧 = 36 tick，特权 seq 与 deploy 状态序号每周期各进 36（两次请求之间世界不动） |
| W2 图 seq = 冻结时刻 seq | **84/84** 个块相等；渲染 p50 86.8 ms（最大 109）不再计入世界时间；镜像在冻结的 sim 上拿到第一条状态（空闲重发） |

证据：`outputs/runs/ls_w1_mujoco_{a,b,c,d}_20260927`、`ls_w2_isaac_20260927`；比对脚本与观测转储在 `outputs/lockstep_eval_20260927/`（`same_rounds.py`、`diff_obs.py`、`render_determinism.py`、`seed_probe.py`、`obs_c|obs_d/`）。

### 10 局对照（W0 / W3 / 第 0 步，checkpoint-5000，seed 6000–6009，Isaac 图，`budget_s` 90）

| 臂 | 时钟与间隙 | 结果 | 失败 | 后退一列 | 每块跳帧 |
|---|---|---|---|---|---|
| `box_room_grasp_200_cap120_20260925`（原有） | 墙钟，`hold` | 9/10 | close 1 | timeout 9 | 4 帧 634、3 帧 46、5 帧 32 |
| `wc_continue10_20260927` | 墙钟，`continue`（第 0 步） | 8/10 | close 2 | timeout 6、dropped 2 | 4 帧 415、5 帧 314 |
| `ls_w0_replica10_20260927` | lockstep，`hold`，延迟按墙钟实际跳帧分布（W0 复刻） | 8/10 | reach 1、close 1 | timeout 6、dropped 2 | 4 帧 717、3 帧 34、5 帧 24 |
| `ls_w3_target10_20260927` | lockstep，`continue`，L = 2、C = 0（目标模型） | 9/10 | close 1 | timeout 9 | 2 帧 811 |

两两配对的不一致局都是 1–2 局，McNemar p = 1.0：10 局分不出差别，结论等 100 局。

**第 0 步的代价**：墙钟下续发线程和观测构建、推理请求在同一个 Python 进程里抢 GIL，观测 + 推理的中位从 70 ms 升到 94 ms
（延迟 p90 175 → 201 ms），新块从第 5 帧接入的比例从 5% 升到 43%。间隙里参考连续了，但间隙本身长了约一帧。
lockstep 下没有这个问题（间隙按帧数定）。

### 100 局对照（seed 6000–6099，其余同上；W0 校准与 W4）

| 臂 | 时钟与间隙 | 结果（Wilson 95%） | 失败 | 抬起后 | 每块跳帧 |
|---|---|---|---|---|---|
| `wc_hold100_20260927` | 墙钟，`hold`（2026-09-27 之前的循环） | **81/100**（0.72–0.87） | close 11、reach 8 | timeout 69、dropped 12 | 4 帧 7343、3 帧 547、5 帧 266 |
| `ls_w0_replica100_20260927` | lockstep 复刻墙钟（`hold`，墙钟跳帧分布） | **83/100**（0.74–0.89） | close 11、reach 5、walk 1 | timeout 69、dropped 13 | 4 帧 7079、3 帧 507、5 帧 336 |
| `ls_w3_target100_20260927` | lockstep 目标模型（`continue`，L = 2、C = 0） | **88/100**（0.80–0.93） | close 7、reach 5 | timeout 72、dropped 15 | 2 帧 8687 |
| `wc_continue100_20260927` | 墙钟，`continue`（第 0 步） | **83/100**（0.74–0.89） | close 11、reach 6 | timeout 65、dropped 15、budget 3 | 4 帧 3993、5 帧 3928 |

**W0 校准通过**（沿用 2026-09-02 的判据：McNemar p > 0.05、成功率差 ≤ 8 点、失败构成各项差 ≤ 10 点）：墙钟 81 对 lockstep 复刻 83，
同 seed 不一致 14 : 16，p = 0.86；失败构成 close 11 : 11、reach 8 : 5、walk 0 : 1；抬起后的掉落 12 : 13。2026-09-02 那次（单臂 flow200，46 vs 32，p = 0.049）
的"lockstep 停得远"签名这里没有：到箱最近距离中位 0.344 / 0.337 m，最近距离 > 0.55 m 的失败局 1/19 / 3/17（那次 50/68）。
那次的 lockstep 每局 reset 后 deploy 清空观测历史而墙钟不清，且两臂的 deploy 状态不同（见 archive/lockstep.md）；这次两臂都是干净 deploy、同一二进制、同一 Isaac 图，
lockstep 的冻结与模拟延迟在这个 checkpoint 上复现了墙钟的成绩。所以下面 lockstep 的数字可以当作"同一时间模型下墙钟会得到的数字"来读。

**W4（目标模型 vs 现状）**：88 vs 81，同 seed 不一致 11 : 18，p = 0.27——方向是目标模型好 7 点，100 局判不动。
失败构成里少掉的是 close（11 → 7）和 reach（8 → 5）。第 0 步（墙钟补帧）83 vs 81，不一致 12 : 14，p = 0.85：参考连续了，但续发线程让观测 + 推理从 70 ms 升到 95 ms、新块一半从第 5 帧接入，净效应为零；墙钟下它不是改进，默认值保留 `continue` 只因它不再让参考定住。延迟从"观测 + 渲染 + 推理"约 4 帧降到"推理"2 帧、间隙里参考连续，两者的效应在这里分不开（要分开得加一臂 lockstep `hold` + L = 2）。

