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

## 尚未验证（阻塞在共享资源上）

**GPU 闸门一项都没跑。** 2026-09-21 容器 `pids.current` = 2205 / 上限 3840，一个带 exporter 的采集栈
需要约 1800，`require_headroom("collect")` 会直接拒绝。占用来自外部作业（三个 `go2rl` python 共 662 线程、
GPU0/GPU1 合计 21 GB）加上 agent 工具链。因此下列仍是**待办**，在它们通过之前旧路径不能删：

- 单臂 `bottle_simple_gate8.yaml` 8 局 + `gate.py compare simple`
- 双臂 20 局 + `gate.py compare crate`
- 10 步训练冒烟、3 局闭环（断言 `eval/videos/` 非空）、7 份产物契约逐项比对
- 顺带测一次"只重起 exporter 能回收多少 pid"——这个数字不改变代码形状，但决定文档里该怎么写批次代价

## 并存与删除

旧的 `scripts/{collect_bottle,collect_crate,convert,train,eval_bottle,eval_crate,pipeline}.py` 和
`scripts/train_bottle.sh` **仍在**，`scripts/run_common.py` 变成了指向 `run_context` 的 re-export，
保证并存期只有一份实现。闸门全部通过后一次性删除它们，并删掉这个 shim。

## 一处顺带补上的送达缺口

`train_bottle.sh` 把 `--embodiment-tag UNITREE_G1_SONIC` 硬编码在配方里，而实验文件的
`embodiment.gr00t_tag` 从来没被送到训练命令行。`run_train` 改成从配置送达；13 份实验文件的值都是
`unitree_g1_sonic`，所以这次改动是字节等价的，但字段从此真的生效了。
