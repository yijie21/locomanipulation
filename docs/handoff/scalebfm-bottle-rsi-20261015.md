# ScaleBFM bottle 专家自主重标定：RSI 闭环的第二个实例（TODO 26.3，2026-10-15）

状态：方案定下，起点池在跑。分支 `main`。
上游：warehouse 那条 RSI 线第 4 阶段臂 A 的闭环，原样搬过来（它的方案和结果不进本实例的副本）；[SONIC → ScaleBFM 替换评测](controller-compare-20261013.md)（§10–14 bottle L1，§25 L2）。

## 1. 要回答的问题

- 用户 10-15：给一个控制器（ScaleBFM）和一个任务（bottle），RSI 闭环自己把任务调好，并导出关键节点的视频，让人看出进化过程。
- 第二个实例，验证 warehouse 那套闭环换一个任务还能用：换的只有适配层（一局的运行程序、起点池、冻结字段、成功规则、提示词），角色、探针、验收门、隔离副本不变。

## 2. 起点（已确认的事实）

| 事实 | 来源 |
|---|---|
| L2 标称 45 个起点：SONIC 40，ScaleBFM（mode 4，专家原样移植）39；掉物 2 对 6，掉物护栏不过；ego p95 0.736 对 0.492 | controller-compare §25 |
| ScaleBFM 的 6 局失败都是"抬起时没把瓶子抬上去"，计入掉物 | 同上 |
| 移植时在 L1 的 10 个起点上选了 mode 4（10/10；mode 6 10/10，mode 2 8/10） | controller-compare §20 |
| `bottle_round.py` 和 L2 的 `bfm_expert_bottle.py` 逐拍一致（3 个起点里 2 个逐位相同；第三个是 L2 进程的首局，差在 TorchScript 首次调用） | 10-15 实测 |

和 warehouse 不同：bottle 的起点已经接近 SONIC，提升空间小，进化可能主要在掉物上。停止线据此定（§9），在开跑前写死。

## 3. 冻结

- 起点、瓶子、桌子、场景不动；控制器 ScaleBFM M 22200 不动；位姿用仿真真值。
- 成功规则冻结：`simple.min_rise_m`（抬起至少 4 cm）、`simple.disturb_limit_m`（合爪前瓶子被推开 > 5 cm 即失败）、`recovery_timeout`（整局时限），由验收门拒绝；成功的其余判据在 `bfm_expert_bottle.py` 里（两侧指垫接触、后退 0.35 m 后保持 2 s、掌系内滑动 < 3 cm、瓶子高于桌面 3 cm），Engineer 改不到。
- 任何池都不删局。

## 4. 起点池

| 池 | 起点 | 用途 |
|---|---|---|
| 诊断池 | L1 10 个（种子 40000）+ L2 标称里 SONIC 实时率 ≥ 0.85 的局 | Reviewer 看逐帧、Prober 跑干预 |
| 验收池 | 新种子 45000 起，SONIC 实时率 ≥ 0.85 的前 60 局 | 验收门 |
| 封存集 | 新种子 46000 起，SONIC 实时率 ≥ 0.85 的前 80 局 | 最后和 SONIC 比、出对比视频；闭环角色碰不到 |

验收池和封存集由 SONIC 生产专家带录制跑出来（`sonic_pool.sh`，独占机器；录制会拉低实时率，低于 0.85 的局作废，所以多跑了 10 和 12 局）。验收池和封存集放在 `outputs/scalebfm_bottle_rsi_20261015_hidden/`。

## 5. 角色

同 warehouse 第 4 阶段的臂 A：Manager、Reviewer、Prober、Engineer、Gate。没有交接记录和包络（warehouse 的 `handoff.py`、`envelope.py` 没有 bottle 版）。冷启动：只给 `rules.md` 和本文 §1–§11。闭环在隔离副本里跑（`sandbox.py`），副本里的 STATE / TODO / DECISIONS 是一行说明。

## 6. 一轮迭代和验收门

- 一轮：Reviewer 出至多 4 条假设（带预测）→ Prober 在诊断池上从分叉点闭环跑每条 → Manager 选一条或打回 → Engineer 做候选（配置，可选代码）→ Gate。
- 分叉点：walk（整局）、ready（双手举起起）、grasp（伸手起）。
- 验收门同 warehouse §6.4：候选和 incumbent 在验收池同一批起点配对，前 30 局净胜 ≤ 0 就拒；接受要净胜 ≥ 3、单侧 exact McNemar p ≤ 0.05、摔倒和掉物都不超过 incumbent + 1、设的字段确实被读到、候选没被拒过。每次先重跑 incumbent 的 5 局做确定性哨兵。
- 掉物：抬起过 ≥ 3 cm、最后不在 ≥ 3 cm 处；摔倒：骨盆低于 0.40 m（l2_measure 的 bottle 规则）。

## 7. agent 能改什么

| 层 | 内容 | 条件 |
|---|---|---|
| 参数 | `task.reach` 里没冻结的字段（`simple.*`、`whole_body.*`）、ScaleBFM mode 0–7、参考变换 | 过验收门 |
| ScaleBFM 专属代码 | `extensions.py` 的钩子 | 验收门 |
| 共享专家代码 | `src/locomani/tasks/bottle/*`，只能加默认关闭的字段 | 验收门 + 相关单测 |
| 升级给人 | 场景、瓶子、成功规则、控制器、位姿来源、真机、冻结规则、预算 | 停下写报告 |

## 8. 人的知识

只有 `rules.md`（v1）：warehouse 的通用证据规则原样照搬，加 bottle 这条线已知的事实（§2 的表）。人插手记在 `loop/interventions.jsonl`。

## 9. 停止条件和预算

开跑前定死，先到先停：
- 验收池上 incumbent 比 SONIC 多成 ≥ 3 局（warehouse 是 5；bottle 起点离 SONIC 只差一两局、上限近，5 局很可能够不着）；
- 验收门调用满 8 次，或墙钟满 24 h，或模型花费满 200 USD；
- 连续 3 次拒绝且 Reviewer 没有新假设；Manager 叫停。

资源：ScaleBFM 用两张卡、12 个进程（只有这一个闭环在跑）；SONIC 栈在跑时不跑 ScaleBFM。

## 10. 视频

- 进化：`scripts/experiments/rsi_demo_20261015/evolution.py` 从台账导出每次接受的前后对比、各代同屏和每轮卡片（同 warehouse）。
- 对比：最终 incumbent 在封存集上对 SONIC，按 warehouse 的选片规则（`warehouse_compare.py` 的 bottle 版）。

## 11. 结果怎么读

| 结果 | 结论 |
|---|---|
| 封存集上 ≥ SONIC，掉物护栏过 | bottle 也是专家问题，25.9 的 bottle 一项按重标后的数字判 |
| 验收池涨了、封存集没涨 | 过拟合验收池；只认封存集 |
| 停在验收门上限，几乎没接受 | 起点已近上限，或 ScaleBFM 在 bottle 上的限制不在专家能调的范围内 |
