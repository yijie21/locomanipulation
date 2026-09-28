# STATE — 2026-09-28（会话：文档按"只想跑通的人"重组；所有 VLA 评测默认 lockstep；三条线冒烟实跑）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，已推送到 origin/main（`4a3c0f3`，含 henry-zhao 的两个提交的 merge）。
没有运行中的栈、渲染或镜像。另一个检出目录 `locomanipulation-loco_agent`（`outputs/`、`third_party/` 是指向本仓库的软链接）的会话会在实例 1、GPU 1 上起栈，起本仓库的栈前先查实例锁和 `pids.current`。

## 这次做了什么（TODO 第 9 节；DECISIONS 2026-09-28 最后一节）

- 三批以前未提交的改动合成一个提交 `291cdeb`（TODO 4.7.11.8 / 7.4 / 8.12）。
- 代码：`paths.sh` 导出 `PYTHONPATH`、读 `$LOCOMANI_WORKSPACE/locomani.local.sh`（这台机器的 `SONIC_DEPLOY_ROOT`、`USD_PY` 在里面）；`scripts/build_scene.sh <场景名> [--isaac]`；`EvalConfig` 默认 `lockstep` + `latency_frames` 2。
- 实验文件：`bottle_simple_smoke` / `bottle_simple_200` / `crate_table30_record` 改 lockstep；新建 `crate_smoke`、`warehouse_smoke`、`warehouse_200`。
- 文档：README 是文档清单；三份 runbook 只写命令、做什么、得到什么（数字来自冒烟实跑）；新建 `docs/running.md`、`docs/warehouse_design.md`、`docs/agents/doc-ownership.md`；`setup.md` 重排，命令逐条不变；`ROADMAP.md` 归档，`reference/runbook.md` 并入 `running.md`。
- 冒烟实跑抓出的六个问题都已修（TODO 9.5.1）。

## 已确认

- 三条线按新 runbook 的命令从采集跑到闭环评测：bottle 12 分钟、crate 8 分钟、warehouse 40 分钟，闭环都是 lockstep 0/2（10 步的模型）。运行在 `outputs/runs/{bottle_simple_smoke,crate_smoke,warehouse_smoke}`，记录在 `outputs/docs_smoke_20260928/`。
- 建场景的命令输出到临时目录后，XML 和元数据去掉路径与现有 `outputs/scene/` 逐字相同（没有原地重建：`outputs/` 被另一个检出目录共用）。
- 文档内链接和锚点：这次改写没有新增断链，改写前就断的 8 个也修了。

## 待验证 / 卡点

- 未验证：bottle 的 lockstep 和墙钟对照（单臂实现 09-02 校准未过；TODO 9.3），要等下一个 bottle checkpoint。
- 待用户：看新文档；认可后按同样写法重写详解页、`rendering.md`、`scalebfm.md`（9.8）。
- 上一会话留下：四边线 compact 采集（TODO 8.6.4）仍待用户定两个问题；新 stage 下 checkpoint-5000 的 Isaac 评测是否偏离 81/83/88（TODO 7.2）。

## 关键命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
cat /sys/fs/cgroup/pids.current; timeout 30 bash scripts/cpu_headroom.sh 8; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
bash outputs/docs_smoke_20260928/smoke_line.sh crate_smoke bimanual        # 一条线的冒烟（第三个参数给 Isaac 的卡）；FROM=eval 只重跑评测
python3 scripts/artifact_contract.py check outputs/runs/<run> crate_eval   # bottle_eval / crate_collect / ...
# 提交：make check → 按文件名 add（不要 -A）→ git commit -- <文件> → fetch；远端有新提交就 merge（不 rebase，TODO 里记着提交哈希）→ push
```
