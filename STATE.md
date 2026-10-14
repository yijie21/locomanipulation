# STATE — 2026-10-14（会话：TODO 25.4–25.8 连续做完，SONIC → ScaleBFM 替换评测；25.9 待用户）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-13）：origin/main = `a2cfb23`；本地 main 多 `b3008b4`（上次 handoff）、`154929b`（TODO 24 整条线）和 TODO 25 计划提交，都**未推送**（用户没说推）。
已确认（10-13，pgrep / nvidia-smi）：实例 0–4 空、GPU 0 / 1 空。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败（10-12 合并时修了 `warehouse_mars_200_isaac.yaml`）。

## 提交

- 已确认（10-13）：TODO 24 整条线作为一个提交 `154929b`（用户在 TODO 25 计划里定的），提交钩子通过。
- 没提交（不是这条线的、来路不明）：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1.md`–`step7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`。
- STATE.md 上限 60 行（`lint_repository.py`）。

## 磁盘

- 已确认（10-13 15:30 UTC）：`/workspace` 剩 40 GB（97%）；跑仿真 / pytest / 安装前 `export TMPDIR=/workspace/tmp`。容器内存 91 GB。
- **计划删除（用户 10-13 批准，10-13 15:34 UTC 写入；最早 2026-10-14 16:00 UTC 执行，TODO 9.12）**：约 120 GiB。三个 10 步冒烟 run 的 `model/`、`outputs/deploy_{docbuild_20260924,pinned_20260925}` 和 `.claude/worktrees/` 的 worktree、10-08 及以前 run 与 `outputs/fresh_repro_20260925` 的 ≥ 1 MB 文件、根目录 wandb / 缓存。清单与留下的东西见 `outputs/cleanup_20261013/summary.md`；执行 `bash outputs/cleanup_20261013/apply.sh`（到点前、SONIC 栈在跑时拒绝）。有异议在此之前改清单。
- TODO 24 产物 `outputs/wbc_compare_20261012/`；TODO 25 产物将放 `outputs/controller_compare_20261013/`。C1–C7 要下载 G1 重定向 LAFAN1，先看剩余空间。
- 待用户（更早）：21.3.4.6、22.7、11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7；要删先在这里写明 24 小时。

## 当前进度

- **TODO 25（25.1–25.8 做完，10-14）**：计划和结果在 `docs/handoff/controller-compare-20261013.md`（§9–25）。L1 关口通过（§21）。L2（§25）：bottle 标称 40 对 39 / 45，组合 2 对 35 / 46；warehouse 标称 20 对 6 / 40（p 0.003），组合 1 对 4 / 40。主判据不满足（warehouse 专家是和 SONIC 一起调出来的），组合档不否决；SONIC 怕延迟，ScaleBFM 真机缺世界位姿（腿里程计漂，§24）。
- 开跑前用户改判据（10-13）：a 按 50 N·m 测（= mars 标称，腰 50、踝 35），摔倒不太高就不否决——写成任一项 10 次摔倒 ≥ 3 次才否决（3/10 是这条线定的数）；h 否决、warehouse 生产档只含 ym 边按推荐。
- **TODO 24（做完，已提交）**：65 物体线受控臂 SONIC 48 对 ScaleBFM 77 / 129（p < 0.001），包络过滤放开臂 104；终点线刹车越线惯性 ScaleBFM 0.5–4.5 对 SONIC 4–25 cm。过程 `docs/handoff/wbc-compare-20261012.md` §1–9。

## 下一步（编号见 `TODO.md`）

- **下一步：25.9 最终判定交用户**（TODO 25.9.1：warehouse 专家要不要按 ScaleBFM 重新标定后再比；25.9.2：ScaleBFM 真机世界位姿来源）。用户没说之前不开 25.10。
- 10-13 晚用户指示已做完：24 前馈（handoff §10，按速度刹车前馈有效、按姿态无增益）；23 抓取来源定缓存；22.7 不删。
- 产物：`outputs/controller_compare_20261013/{clips,complex,switch,p25,l2}/`；L2 视频 `l2/videos/`。SONIC 运行目录 `outputs/runs/controller_compare_{l2,p25}_*`。
- 这批的坑：bottle L2 有一局挂住 12 分钟（看门狗 `queue_watchdog.sh`）；bottle 实时率 0.844–0.892，8 局作废后补跑 `_b` / `_c`。

## 未解决 / 卡点

- 待用户：25.9.1、25.9.2、25.11.1（机载电脑型号、外部定位；现按没有处理）、23.2.7、23.3–23.5、23.7.4–23.7.6、21.3.5、22.3.6–22.3.8、22.5、22.7、21.3.4.2–21.3.4.4、21.3.4.6、21.3.4.8.1、21.3.1、19.8.1、19.8.3、19.18.3.1、16.14.1、16.15.4、13.6.18。
- 已确认（规则）：SONIC 栈实时，旁边不跑 ScaleBFM 进程；ScaleBFM 仿真确定性，重复局要抖初始条件；批次跑着时不改它会再 import 的脚本。
- 已确认（环境）：镜像的 `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` 指向不可写的 `/.uv`，`CUDA_HOME` 是 13.0；numpy 大页缺页会卡内核整理——离线 numpy 重活设 `NUMPY_MADVISE_HUGEPAGE=0`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp
make check
D=scripts/experiments/wbc_compare_20261012; BFM_PY=third_party/Isaac-GR00T/.venv/bin/python; SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src $BFM_PY $D/bfm_brake.py --program plan --speeds 0.55 --repeats 3 --out /workspace/tmp/brake_smoke   # ScaleBFM 走路冒烟（TODO 25 的起点脚本）
# TODO 24 的全部命令见 docs/handoff/wbc-compare-20261012.md 各节末的命令块
```
