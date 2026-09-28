# STATE — 2026-09-28（会话：文档按"只想跑通的人"重组；所有 VLA 评测默认 lockstep；三条线冒烟实跑；已推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，与 origin/main 同步（`686d507`；之后只有这次 handoff 提交）。
没有运行中的栈、渲染或镜像（本会话的都已收掉）。另一个检出目录 `locomanipulation-loco_agent`（`outputs/`、`third_party/` 是指向本仓库的软链接）
的会话会在实例 1、GPU 1 上起栈；GPU 1 上还常驻一个与本项目无关的 `depth_server.py`（约 3.7 GB）。

## 磁盘（先处理）

- 已确认：`/workspace` 只剩 26 GB（98%）。本会话的冒烟产物约 97 GB：三个冒烟运行的 `model/` 各约 24 GB，
  `outputs/docs_smoke_20260928/try2/` 24 GB。都是 10 步的 checkpoint，没有用处。
- 计划删除（TODO 9.9）：上面这些 `model/` 目录和 `try2/`，**2026-09-29 11:45 之后**（AGENTS：删 outputs 要在 STATE 提前 24 小时写明）；
  用户同意可以提前删。运行目录里其余文件（manifest、日志、录像、`eval/`）保留，它们是 runbook 里数字的证据。
- 上一会话留下：2026-09-29 后可删 `outputs/experiments/warehouse_edges_20260928/trial_45x20x18`（107 MB，TODO 8.11）。

## 当前进度（TODO 第 9 节；DECISIONS 2026-09-28 最后两节）

- 已确认：文档重组、lockstep 默认、`paths.sh` / `build_scene.sh`、冒烟和完整规模实验文件都已提交并推送（9.1–9.7）。
  这台机器的 `SONIC_DEPLOY_ROOT`、`USD_PY` 在 `/workspace/locomani.local.sh`（不进 Git）。
- 已确认：三条线按新 runbook 的命令从采集跑到闭环评测。bottle 12 分钟、crate 8 分钟、warehouse 40 分钟（含 14 分钟评测录像），
  闭环都是 lockstep 0/2（10 步的模型，预期如此）。运行 `outputs/runs/{bottle_simple_smoke,crate_smoke,warehouse_smoke}`，
  驱动脚本和日志 `outputs/docs_smoke_20260928/`。
- 已确认：冒烟抓出的六个问题都修了并有测试（TODO 9.5.1；理由见 DECISIONS）。
- 已确认：建场景的命令（含 henry-zhao 改过的降桌高脚本）输出到临时目录，与现有 `outputs/scene/` 去掉路径后逐字相同。
- 已确认：文档链接和锚点全部能解析（改写前就断的 8 个也修了）；`make check` 通过。

## 待验证 / 卡点

- 待验证：bottle 的 lockstep 评测和墙钟对照（单臂实现 09-02 校准未过：墙钟 46 vs lockstep 32）。要等下一个 bottle checkpoint（9.3）。
- 待验证：crate 完整规模（`crate_table30_record`）在 lockstep 下的成功率，runbook 里只有 09-15 墙钟 71/100（9.10）。
- 待用户：看新文档；认可后按同样写法重写详解页、`rendering.md`、`scalebfm.md`（9.8）。
- 上一会话留下、未动：四边线 compact 采集（8.6.4）待用户定两个问题；新 stage 下 checkpoint-5000 的 Isaac 评测是否偏离 81/83/88（7.2）。

## 下一步（编号见 `TODO.md`）

- **9.9**：删冒烟的 `model/` 和 `try2/`（到时间或用户同意后）。
- **9.8**：用户认可后，按新写法重写详解页、`rendering.md`、`scalebfm.md`。
- **8.6.4**：四边线 compact 采集（先问用户）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh     # 路径、PYTHONPATH，并读 /workspace/locomani.local.sh
df -h /workspace; cat /sys/fs/cgroup/pids.current; timeout 30 bash scripts/cpu_headroom.sh 8; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
bash outputs/docs_smoke_20260928/smoke_line.sh crate_smoke bimanual            # 一条线的冒烟；第三个参数是 Isaac 的卡；FROM=eval 只重跑评测
python3 scripts/artifact_contract.py check outputs/runs/<run> crate_eval       # 或 bottle_eval / crate_collect ...
du -sh outputs/runs/*_smoke/model outputs/docs_smoke_20260928/try2             # 9.9 要删的东西
# 提交：make check → 按文件名 add（不要 -A）→ git commit -- <文件> → fetch；远端有新提交就 merge（不 rebase，TODO 里记着提交哈希）→ push
```
