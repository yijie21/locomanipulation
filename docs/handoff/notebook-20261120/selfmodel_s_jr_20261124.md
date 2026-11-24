# selfmodel_s_jr_20261124：测试局上 J 带录像重跑（JR），为 S 录下发令时刻的状态

- 日期：2026-11-24（启动前写）
- 机器 / GPU：4090。和 [simple_rgbd_20261124](simple_rgbd_20261124.md) 的 J 一样：ScaleBFM 在 1 号卡，SONIC 在 2、3 号卡，每个任务一个进程。和开发局的 S 运行同时跑，共用这几张卡。
- 代码版本：`selfwm` @ `99c0d3fd`（4090 的检出 `bc426d5` 加未提交的文件。跑到的文件和 Mac 逐个 md5 相同，只有 `simple_bench.py` 例外：4090 那份和 RGB-D 那次 J 跑的是同一份（md5 abc246fd），和 Mac 的 `99c0d3fd` 只差模块开头的帮助文字，代码相同），抄进输出目录的 `asrun_JR/`。
- 配置：
  - `bench_via.sh <dir> <控制器> none JR`，`BENCH_SET=sensor=rgbd`，`REACH_SONIC_NETS=gpu`；
  - 输出目录就是 RGB-D 那次的 `outputs/simple_eval_20261112/bench_rgbd_20261124/{scalebfm,sonic}/`，读同一份 `via/`（J 的路线）；
  - SIMPLE 官方评测局，四个抓取任务各 30 局，感知栏。
- 关联：TODO 38.10.4；[selfmodel_s_20261124](selfmodel_s_20261124.md)（S 在测试局上的登记，等开发局选完再写预测）。

## 假设

JR 和 J 发的是同一条路线、同一个参考，只多了录像（`--film-ref` 只记录，不改命令）。仿真对相同命令是确定的（复核核对过：没有路线的局，各做法和 A 逐局相同，92 / 92、91 / 91），所以 JR 应当和 J 逐局相同。

## 预测

- ScaleBFM、SONIC 各 120 局，JR 和 J 的 `success_45`、`success_official`、`t_success` 逐局全部相同。
- 理由：同一份 `via/`，同样的卡和进程数；`--film-ref` 只多存参考。

## 判据

- 全部相同：S 用 JR 录下的状态，S 和 J 的配对就是和 RGB-D 那次 J 的配对。
- 有任何一局不同：先查不确定性的来源（录像是否改了时序、卡上同时跑别的进程是否影响），查清之前不跑测试局上的 S。SONIC 的网络在 GPU 上，如果只有 SONIC 不同，记下来，S 改成和 JR 配对（JR 和 S 同样带录像）。

## 预测修订

## 结果（贴原始输出，不转述）

`outputs/simple_eval_20261112/bench_rgbd_20261124/phase.log` 和两个 `pipeline.log` 的末行：

```
JR start 2026-11-24T14:09:45Z bc426d5 + uncommitted abc246fd 967b61e0
JR all done 2026-11-24T14:17:28Z
[14:14:38] arm JR (GPUs 1): 120 rows, 0 failed processes
[14:17:28] arm JR (GPUs 2 3): 120 rows, 0 failed processes
```

逐局比较（`rows/J_*.jsonl` 对 `rows/JR_*.jsonl`，按 task、level、index 对齐，比 `success_45`、`success_official`、`t_success`）：

```
scalebfm J 120 JR 120 differ 0 []
scalebfm J 45/off 95 81 JR 45/off 95 81
sonic J 120 JR 120 differ 0 []
sonic J 45/off 34 20 JR 45/off 34 20
```

录像在 `{scalebfm,sonic}/films_JR/`，行在 `rows/JR_*.jsonl`。

## 意外

- 没有。比预计快：两个控制器 8 分钟跑完（RGB-D 那次 J 和 D、M 同时跑，用了约 24 分钟）。

## 结论与下一步

- JR 和 J 在两个控制器的 240 局上逐局相同，落在判据的"全部相同"一侧。S 用 JR 录下的发令状态，S 对 J 的配对就是对 RGB-D 那次 J 的配对。
- 证据强度：一次运行，每个控制器 120 局；只说明录像不改命令、仿真对相同命令确定，不涉及哪种做法好。
- 下一步：开发局选完，写测试局 S 的预测并提交，然后跑 S。

## 独立复核

不需要单独复核：这是流程核对，结论随 selfmodel_s_20261124 一起复核。
