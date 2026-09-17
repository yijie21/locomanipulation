# F10 — exporter 静默丢帧（2026-09-17，根因未定位）

> 这是 2026-09-17 那次"只照文档跑一遍"的 walkthrough 里编号 **F10** 的发现。
> 同一轮还查出 F1–F9、F11 共十条，其余都已修（见文末"同一轮的其他发现"）。
> **本条只加了拦截，没有定位根因。**

## 一句话

exporter 会间歇性地只录下一局中的很小一部分，**而且大多数情况下不报错**：
一局 37 秒的成功回合可能只留下约 1.2 秒画面，然后作为 22 帧"示范"正常进入训练集。

## 怎么发现的

按 `docs/collection.md` 跑 1 局采集 → 转换，转换阶段崩在：

```
ValueError: training boundary outside recorded sequence range
  src/locomani/pipeline/verification_tail.py:15（--trim-verification-tail 触发）
```

崩溃只是症状。那一集的录制断在专家做完之前，于是训练边界落到了它记录区间之外。
**而同一个 session 里另外两集也一样残缺，只是它们的边界恰好落在区间内，
所以静默通过了全部检查。**

## 实测数据

密度 = 行数 / priv_seq 跨度。健康值约 1.1（exporter 每个特权序号单位记约 1.1 行）。

### session 对照（同一份配置、同一台机器、同一天）

| session | 时刻 | 集数 | 密度 | parquet 大小 |
|---|---|---|---|---|
| `2026-09-17-05-31-43`（gate8） | 05:31 机器空闲 | 8 | **1.10–1.15，全健康** | 371K–1008K |
| `2026-09-17-09-19-37`（v2） | 09:19 | 6 | 0.04 / 0.09 / 1.10 / 0.05 / 1.05 / 1.07 | 55K 54K 409K 62K 880K 397K |
| `2026-09-17-09-31-45`（v3） | 09:31 | **4**（produced=6） | 0.44 / 0.40 / 0.26 / 0.09 | 188K 346K 253K 61K |
| `2026-09-17-09-38-31`（v4） | 09:38 | 6 | 0.04 / 0.31 / 0.32 / 0.70 / 1.04 / 1.08 | 55K 275K 243K 595K 818K **0K** |

### 三种不同的坏法

1. **静默欠采样**（最危险）。v2 的 ep0：46 行覆盖 seq 3580–4800，实际分布是
   1 帧在 3580、**43 帧挤在 4063–4102（约 1.2 秒）**、2 帧孤立在 4504 和 4800。
   它的训练边界 4487 落在 [3580, 4800] 内 → 范围检查通过 → 转成 **22 帧**入库。
   当天五个残片里只有一个响了，纯属运气。
2. **整集丢失**。v3 的 manifest 有 6 条 produced，磁盘上只有 4 集。
   这一种被既有的硬校验抓住了，报错清楚。
3. **0 字节 parquet**。v4 的 ep5。原先由 pyarrow 抛出，只给一个路径，
   而且发生在其余各集已经转码完之后。

## 已经排除的

- **sim 侧正常**：图像发布频率全程 25.4–34.7 Hz（40 个采样，中位 32.5）。
- **exporter 主循环正常**：`total_loop_avg 11.58 ms`
  （`add_frame 1.50` / `check_recording_commands 8.43` / `poll_image 1.41`），
  远快于 ~32 Hz 的图像速率。
- **不是图像断流**：整个采集日志里只有 4 条
  `No new image message received for ~100ms. Reusing stale image.`
- **不是专家失败**：这些回合本身正常跑完，36–38 秒，v4 是 6/6 成功。
- **不是配置差异**：v2/v3/v4 与健康的 gate8 用的是同一份实验文件
  （只差 `successes`/`batch_rounds`/`steps`/`holdout`/`eval.rounds`）。

所以问题在 **exporter 收到图像之后、写进 parquet 之前**这一段。

## 唯一的线索（相关性，不是证据）

另一个项目 `/workspace/code/real2sim_pipeline` 的进程 **09:14:24** 启动
（pid 2216378，133–155% CPU；另一个 `utils.fuse_3d_boxes` 瞬时冲到 226%）。
**三次受损的 session 全部在这之后，健康的那次在这之前。**

容器 CPU 配额 15.36 核。观测到即使只用 3.5 核，3 秒内仍有 2/30 个周期被限流
（`/sys/fs/cgroup/cpu.stat`），说明瞬时尖峰会打满配额。

这只是时间相关性。**没有做过对照实验，因果未确立。**

## 现在挡在哪里

`src/locomani/pipeline/recording_density.py`，在转换的 **Pass 1**
（解码任何一帧之前）执行，与既有的 manifest 校验同一位置：

- 密度 < `MIN_RECORDING_DENSITY`（0.60）→ 拒绝，逐集打印实测数字
- 0 字节 parquet → 拒绝，并指名文件

阈值选 0.60 的依据就是上面那张表：健康最低 1.04，损坏最高 0.44，两边都有余量。
测试在 `tests/test_recording_density.py`，其中一例直接复刻了 v2 ep0 那个真实残片。

**这只把静默变成了响。遇到仍然采不到数据，要重采。**

## 下次接着查的建议

1. 先复现：确认机器空闲时能稳定拿到 1.10–1.15，再让另一个项目上负载，看是否复现。
   这是把相关性变成因果的最小实验。
2. 若复现，看 exporter 的落盘路径——`add_frame` 之后到 parquet 写入之间，
   是否有按时间丢弃、缓冲区有界丢弃，或写入线程被饿死。
3. `nr_throttled` 是容器级累计值，无法定位到某一段。要归因就得在采集前后各采一次
   快照，算区间差。

## 同一轮的其他发现

walkthrough 共十一条，F10 之外都已修并推送（`1bb063d`、`728ebe8`、`cf98c75`、`eb3b046`）：

| | 问题 |
|---|---|
| F1 | `install_scene.sh` 在 `set -euo pipefail` 下被 `grep \| head -1` 的 SIGPIPE 打断，tar 已解包但重定向和重建都没跑 |
| F2 | `collection.md` §3「跑一局」实际跑 8 局 |
| F3 | §2 建的是 `outputs/scene/sim`，但实验要的是 `sim_lowcom`，而且没人建它 |
| F4 | 生成的配置表不覆盖 `collect`/`train`（它们不是 dataclass） |
| F5 | `bottle_simple_gate8.yaml` 的 `holdout 10 > successes 8`，整条线必崩 |
| F6 | `--name` 让副本里的 `name:` 与目录不符（行为正确，缺文档） |
| F7 | 产物路径多一层 `<stage>/`（training.md、evaluation.md 共 6 处） |
| F8 | 固定 case bank 下 `successes` 不封顶，旋钮其实是 `batch_rounds` |
| F9 | `evaluation.md` 仍引用重构里删掉的 `locomani/stages` |
| F11 | 闭环评测把尝试次数当回合数，历史评测分母普遍翻倍 |
