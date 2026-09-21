# 重构：阶段入口改成扁平 bash（2026-09-21）

`/grill-me` 访谈的结果，八个决定；随后与协作者的方案对齐，改了其中三条。
上一轮同类重构是 [扁平 Python 脚本（2026-09-17）](refactor-flat-scripts-20260917.md)，本轮不推翻它的
结论（一份全展开 YAML、整份配置下传、产物契约、dry-run 送达测试），只改**入口的形状**。

## Problem Statement

2026-09-17 之后每个阶段是一个 Python 脚本。用户读代码时发现训练那条是 **python 调 bash 再调 python**
（`scripts/train.py` → `scripts/train_bottle.sh` 253 行 → GR00T 的 python），要求把入口统一成 bash，
并且"这几个 bash 直接调用功能性代码"。

核实后事实是：这种三明治**只有训练一条**成立；`convert.py` / `eval_bottle.py` / `collect_bottle.py`
里没有 bash，它们直接起 python 子进程。但用户要的不只是消灭那一层，还包括**把进程边界摊开**：
采集 = 起栈 + 跑 agent 两个终端，闭环评测 = 起栈 + 策略服务 + 跑局三个终端。

## 八个决定

1. **批次循环下沉到功能代码**，不写进 bash；两个 bash 互不调用。
2. **bash 里不出现任何实验 YAML 的字段**。整份 YAML 以路径下传，bash 只放机器级变量（venv、
   `CUDA_VISIBLE_DEVICES`、`SONIC_DEPLOY_ROOT`）。
3. 每个 bash **自带边界语义**（起完即退 / 前台守着 / 跑完自己收栈）；
   `scripts/pipeline.py` 改写成 `scripts/run_overnight.sh`，它是**唯一**允许调用其它脚本的脚本。
4. `scripts/` 下 8 个 Python 入口搬进 `src/locomani/pipeline/`，`scripts/` 顶层只剩 bash。
5. `run_agent_server.sh` 是采集的主驱动；**采集与评测的驱动按线分两份**，
   convert / train / open-loop / 起栈 / 策略服务各一份（共用的部分不复制，否则必然漂移）。
6. 验收沿用上一轮那把尺子：dry-run 送达 + 7 份产物契约 + `gate.py compare` + 10 步训练冒烟 +
   3 局闭环（核心断言是录像目录非空）。
7. 在主工作树开分支（`third_party/`、`outputs/` 不跟踪，worktree 里跑不了 GPU），不委托。
8. `train_bottle.sh` 的 253 行配方**移植进 Python**，用 `DRY=1` 做字节级 A/B。

与协作者方案对齐后改了三条：`pipeline.py` 由"删除"改为重写成 `run_overnight.sh`（决定 3）；
新增 `stop_stack.sh`；评测驱动改名 `run_vla_inference.sh`，与采集的 `run_agent_server.sh` 区分开。

## 最终形状

```
scripts/collect_data.sh      -> run_stack --mode collect    起录制栈（sim+deploy+exporter），起完即退
scripts/start_stack.sh       -> run_stack --mode eval       起评测栈（deploy 不带 encoder）
scripts/stop_stack.sh        -> locomani.stack.cli down     只停指定实例
scripts/run_vla_server.sh    -> run_vla_server              前台服务 checkpoint，Ctrl-C 即收
scripts/convert_data.sh      -> run_convert                 不用卡
scripts/train.sh             -> run_train                   配方在里面，直接 exec torchrun
scripts/open_loop_eval.sh    -> gr00t/eval/open_loop_eval.py（原 gr00t_open_loop.sh 改名）
scripts/run_overnight.sh     -> 上面这些                     唯一的组合脚本
scripts/single_arm/{run_agent_server,run_vla_inference}.sh
scripts/bimanual/{run_agent_server,run_vla_inference}.sh
```

每个 bash 十几行：参数检查、`. paths.sh`、导出 `PYTHONPATH`、一个 `exec`。

## 三个事实，是它们决定了栈的生命周期

1. **exporter 必须每批重起，与 pid 无关。** `convert` 按**位置**把 exporter session k 配给 manifest k，
   `audit_bottle_actions.py:141` 又按批号读 `frame_audit_{批:02d}.jsonl`。一个跨批的 session 会毁掉配对。
   所以：**sim + deploy 长命，exporter 每批换**（`run_collect_single_arm.open_exporter_session`）。
2. **实例锁原本随起栈进程退出而释放。** `Stack._acquire_lock` 是 `fcntl.flock`，持有者是调用 `up()` 的
   进程；`collect_data.sh` 起完即退，锁就没了，别的会话能在同一实例上再起一个栈。新增
   `Stack._require_instance_free()`：占用与否由 `stack.json` 里**子进程是否活着**判定。
3. **产物布局有两处声明过的变化。** 栈跨批存活后 `SIM_VIDEO_DIR` / `SIM_STATE_TRACE_DIR` 在起栈时定死，
   于是 `collect/videos_*/` → `collect/videos/`、`collect/state_trace_*/` → `collect/state_trace/`、
   `eval/videos_*/` → `eval/videos/`。文件名是 `gym_<时间戳>`（`round_phases.py:183`），跨批唯一，不覆盖。
   manifest、frame_audit、exporter session 仍按批。三份契约已按此更新并在 `note` 里写明。

## 已验证

- **转换**：新旧两条路的 dry-run 命令**字节完全一致**（6,468 字节，`bottle_simple_200_20260916`）。
- **训练**：`DRY=1` 下旧配方与新入口组装出的 `torchrun` 命令**逐 token 一致**，三种硬件分支各验一次
  （单卡 32 GB / 双卡 32 GB / 双卡 80 GB，各 39 个 token）。分支由一个假的 `nvidia-smi` 控制，两条路
  读到同样的硬件——`gpu_probe` 会覆盖注入的变量，所以注入 `VRAM_MIN` 是无效的。
- **测试**：`make check` 全绿（含 linters 与变更范围内的 CPU 测试）、`make check-docs` PASS。
  改了 4 个测试：`test_run_common.py` → `test_run_context.py`、`test_pose_ownership.py`（monkeypatch 要打在
  **读取值的模块**上，否则写进真实的 `outputs/runs/`——2026-09-21 就这么污染过一次）、
  `test_artifact_contract.py` 的 CRITICAL 表、以及三份契约本身。
- **dry-run**：两条线的八个入口全部跑通。

## 不吃 pid 的那部分已经跑过（2026-09-21）

用户批准先跑不占 pid 的验证。做了三件事：

- **开环评测跑通了真的 GPU**：`scripts/open_loop_eval.sh`（原 `gr00t_open_loop.sh`）在
  `bottle_simple_200_20260916` 的 `checkpoint-10000` × 10 集验证集上跑完 10/10 条轨迹，
  平均 MSE 0.001749 / MAE 0.027843，10 张图和 `metrics.json` 在
  `outputs/open_loop_refactor_20260921/`。台账里**没有**这条 checkpoint 的历史开环数字，所以这是
  首次测量、不是对照；扩散头是采样的，单次数字本身也不该当作基线。它证明的是这条改名后的
  路径端到端可用。
- **契约的改动正好等于声明的那三条**：`git diff HEAD~1 HEAD -- tests/contracts/` 只有
  `collect/state_trace_*/`、`collect/videos_*/`、`eval/videos_*/` 三行 pattern 和三处 note。
  旧布局的运行对**旧**契约 4/4 PASS（collect / convert / train / eval），对**新**契约只在那两条
  重命名的 pattern 上失败——差异是可枚举的，没有别的东西被顺手改掉。
- **补了 attach 缝的测试**（`tests/test_stage_attach.py`）：驱动在没有栈时必须报出**起栈的那条命令**，
  而且资源闸门要排在查栈之前——后者是顺序断言，防止以后有人把栈查询挪到准入检查前面。

## 不录制的冒烟跑通了（2026-09-21）

用户批准按 `record` 给 pid 门槛分档后跑的。**先实测再改阈值**：用 `locomani.stack.cli up --no-record`
（它本来就不经过 `require_headroom`，所以不需要绕过任何闸门）起一次 sim + deploy，带看门狗——
容器 pid 逼近上限会立刻收栈，因为 pid 耗尽会让**整台机器上所有进程**都 fork 失败，不只是自己这个。

| | pids |
|---|---|
| 起栈前 | 2269 |
| 就绪 | 2441 |
| 启动峰值 | 2445 |
| 停栈后 | 2266（无泄漏） |

**不录制的栈只要约 176 个 pid**，那 1800 几乎全是 exporter。于是 `shortfall` / `require_headroom`
多了一个 `record` 参数，非记录运行按 `NO_EXPORTER_PIDS = 400`（实测的两倍多，给 agent 进程和更重的
场景留余量）。这同时修好了一件一直不对的事：**评测从来不起 exporter，却一直按 1800 被拒**。

随后按文档的两终端流程跑了 `configs/experiments/bottle_smoke_norecord.yaml`：

- 终端 1 `collect_data.sh` 37 秒起栈并**退出**，sim + deploy 继续活着（pids 2302 → 2446）。
- 终端 2 `single_arm/run_agent_server.sh` attach 上去，跑批 00，1/1 保留，**自己收栈**；
  结束后 pids 回落、`stack.json` 归零。
- 产物落在新布局上：`collect/videos/gym_1789959859.mp4`、
  `collect/state_trace/gym_1789959859_state_trace.npz`——同一个时间戳，这就是新布局依赖的
  "跨批唯一文件名"在真实运行里的样子。`summary.json` 的 `sessions` 是 `[null]`（没有 exporter），
  `stack_env.json` 里 `SIM_VIDEO_DIR` / `SIM_STATE_TRACE_DIR` / `TF_FRAME_AUDIT_PATH` 逐项送达。

### 冒烟抓到的两件事

1. **provenance 回退，已修。** 第一次冒烟的 `commands.jsonl` 只有一行；旧路径有
   `stack_sim` / `stack_deploy` / `stack_exporter`。原因是 `run_stack` 建 `Stack` 时没把
   `record_command` 交过去——起栈换了个进程，记录器就被忘在原地了。修完重跑，三行齐全
   （`stack_exporter` 缺席是对的，这次不录制）。**这是 dry-run 和测试都看不见的一类缺陷。**
2. **CPU 准入把"它即将使用的那个栈"算成了负载。** 驱动第一次被拒（`8.41 cores in use + 8 needed
   > quota 15.36`），而那 8.41 里有一大半是终端 1 刚起的 sim + deploy。旧的单进程设计里，
   `require_headroom` 跑的时候栈还没起。`NEEDS["collect"]["cores"] = 8` 是按"栈 + 专家"定的，
   现在这两半由两个进程各付一次。**尚未修**：正确的做法和 pid 那次一样——起栈的付栈的钱，驱动只付
   agent 的钱——但需要先实测专家进程单独占多少核，不该拍脑袋。当前的应对是等一会儿重试。
   栈在驱动被拒后**仍然活着**，这正是 `stop_stack.sh` 存在的场景。

## 闭环评测的三终端流程也跑通了（2026-09-21）

pid 门槛按 `record` 分档后评测不再被拒（评测本来就不起 exporter），于是用
`configs/experiments/bottle_eval_smoke3.yaml`（3 局，`policy_port` 挪到 5700 以避开实例端口块）
在 GPU1 上跑了一次完整的三终端流程。checkpoint 是从 `bottle_simple_200_20260916` 借来的，
运行目录里的 `train` 条目是手写的——这是**路径冒烟，不是训练**。

- 终端 1 `start_stack.sh` 起评测栈（deploy 不带 encoder）并退出；
- 终端 2 `run_vla_server.sh` 约 20 秒后打出 `listening on tcp://0.0.0.0:5700`，输出同时进终端和
  `<run>/eval/serve.log`；
- 终端 3 `single_arm/run_vla_inference.sh` 等到端口应答才开第一局，跑完 3 局并**自己收栈**。

**`bottle_eval` 契约在一次完全由新路径产出的运行上 PASS**——七份契约里第一份被真运行验证的，
而且它的核心断言正是"录像目录非空"：`eval/videos/` 里 3 局 3 个 mp4。
`closed_loop.json`、`closed_loop.manifest.jsonl`、`config_snapshot.json`、`eval_config.json`、
`traces/`、`serve.log` 全部就位，`summary.json` 多了 `eval` 条目。

结果 **0/3**。三局是三种**不同**的失败家族（走近不足 1.25 m、合爪但无抬升、站位偏离 spawn 0.41 m），
这说明策略确实在被闭环驱动，而不是卡在一个吸收态。**3 局对成功率没有任何说明力**，这条线
100 局的历史数字在台账里，不要拿这个 0/3 去比。

## GPU 闸门（2026-09-21 晚，GPU1 让出后）

### 单臂行为闸门：两条路都对同一份冻结基线 FAIL，且 FAIL 在不同判据上

`bottle_simple_gate8.yaml`（固定库 `regression8_lowcom`，8 局）新旧两条路各跑一次：

| | yield | end_depth_m | failure_reasons | 判定 |
|---|---|---|---|---|
| 冻结基线（`stage4_wrap8p`） | 0.75 | 0.1469 | — | — |
| **新路径** | 0.75（6/8） | 0.1557 PASS | **FAIL**（多一个失败类别） | FAIL 4/5 |
| **旧路径**（未改动的代码） | 1.00（8/8） | **FAIL** 0.1695 | PASS | FAIL 4/5 |

**未改动的代码也过不了这道闸门**，只是挂在别的判据上。逐局配对（同库同种子）：8 局里 2 局不同，
都在 grasp 阶段。结论：**n=8 分辨不了这条线**，这和台账里"同版本 A/A 也 FAIL 于新失败类别"一致。
所以等价性按 2026-09-17 的先例用**送达**来定：专家 argv **29 个 token 逐字一致**；`stack_env.json`
24 个键里 5 处不同，全部可解释——`SIM_VIDEO_DIR` 是声明过的布局改动，另外四个（`RECORD_INSTANCE`、
`SIM_STATE_TRACE_DIR`、`TF_EXPORTER_SCRIPT`、`TF_FRAME_AUDIT_PATH`）**旧路径也送达了，只是从没被记进
`stack_env.json`**（它们当时在 `base_env` 里，而该文件只写 `spec.settings()`）。旧运行确实有
`state_trace_00/` 和 `frame_audit_00.jsonl` 为证。`RECORD_INSTANCE` 只被旧桥接 `pipeline/stack.py:114`
读取，在实例 0 上惰性。

### 抓到并修掉的两个真回退

1. **`world_lift` 找不到状态轨迹。** `world_lift.py:93` 把目录硬编码成按批的
   `collect/state_trace_{批}`，正是这次布局改动替换掉的。`check_dataset --run-dir` 因此在新路径的
   每一次采集上都会崩。改成先找按 run 的目录、找不到再回退到按批的（旧运行仍可审）。
   **dry-run、单元测试、产物契约都看不见它——是整条链跑出来的。**
2. **批 0 重起 exporter 会丢一帧相机对齐。** 驱动原本在每批之前都重起 exporter，包括批 0；
   那时 sim 的图像发布刚起来还没稳。实测：

   | 运行 | lag 0 占比 | 中位 | 判定 |
   |---|---|---|---|
   | 旧路径 | 12/14 = 86% | +0 | PASS |
   | 新路径，批 0 重起 | 4/8 = 50% | +0 | FAIL |
   | 新路径，批 0 重起（重复） | 3/8 = 38% | **+1** | FAIL |
   | 新路径，批 0 **不重起** | **8/8 = 100%** | +0 | PASS |
   | 新路径，两批（批 1 仍重起） | **10/10 = 100%** | +0 | PASS |

   最后一行很重要：**批间重起 exporter 没有代价**，有代价的只是在栈刚起、发布还没稳时重起。
   批 0 本来就有起栈脚本建好的 session，直接用它。

3. **`run_overnight.sh` 会漏下策略服务。** 它用 `kill`（SIGTERM）收尾，而 `run_vla_server.py`
   只处理 `KeyboardInterrupt`，于是包装进程死了、占 7 GB 显存的 `serve_policy.py` 活着。
   手动三终端不会暴露（Ctrl-C 送到整个前台组）。已改成同时处理 SIGTERM 并在 `finally` 里必收；
   验证方式是起服务、对包装进程发 SIGTERM、数子进程——从 1 变 0。

### 通过的

- **十步训练冒烟**：`run_overnight.sh` 一条命令跑完 采集 → 转换 → 训练（10 步，写出 `checkpoint-10`，
  配方自动选了单卡 8×8 accum + `adamw_bnb_8bit`）→ 3 局闭环，exit 0。
- **五份单臂产物契约全部 PASS**，且都是新路径端到端产出的运行：`bottle_collect`、`bottle_convert`、
  `bottle_convert_val`（`bottle_twobatch_probe`）、`bottle_train`、`bottle_eval`（`bottle_chain_smoke2`）。

## 尚未验证（阻塞在共享资源上）

**要起仿真栈的闸门一项都没跑。** 2026-09-21 两次测量：`pids.current` = 2205 → 2264 / 上限 3840，一个带 exporter 的采集栈
需要约 1800，`require_headroom("collect")` 会直接拒绝。占用来自外部作业（三个 `go2rl` python 共 662 线程、
GPU0/GPU1 合计 21 GB，后来又多了一个 `JeVLA/libero_pilot` 占 9.5 GB）加上 agent 工具链。
显存在当天晚些时候**更紧**而不是更松：GPU0 只剩 17.7 GB、GPU1 只剩 14.6 GB，训练需要 28 GB。
因此下列仍是**待办**，在它们通过之前旧路径不能删：

- 单臂 `bottle_simple_gate8.yaml` 8 局 + `gate.py compare simple`
- 双臂 20 局 + `gate.py compare crate`
- 10 步训练冒烟（显存不够：训练要 28 GB，当天两张卡最多只剩 17.7 GB）
- 其余六份产物契约（`bottle_eval` 已在新路径上 PASS）
- 顺带测一次"只重起 exporter 能回收多少 pid"——这个数字不改变代码形状，但决定文档里该怎么写批次代价
- 实测专家进程单独的 CPU 占用，把 `NEEDS[*]["cores"]` 也按"起栈的 / 驱动的"拆开（见上面第 2 点）

## 并存与删除

旧的 `scripts/{collect_bottle,collect_crate,convert,train,eval_bottle,eval_crate,pipeline}.py` 和
`scripts/train_bottle.sh` **仍在**，`scripts/run_common.py` 变成了指向 `run_context` 的 re-export，
保证并存期只有一份实现。闸门全部通过后一次性删除它们，并删掉这个 shim。

## 一处顺带补上的送达缺口

`train_bottle.sh` 把 `--embodiment-tag UNITREE_G1_SONIC` 硬编码在配方里，而实验文件的
`embodiment.gr00t_tag` 从来没被送到训练命令行。`run_train` 改成从配置送达；13 份实验文件的值都是
`unitree_g1_sonic`，所以这次改动是字节等价的，但字段从此真的生效了。
