# STATE — 2026-10-13（会话：TODO 24 wbc_compare，SONIC 对 ScaleBFM——65 物体抓取 + 空地终点线刹车）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-13）：origin/main = `a2cfb23`；本地 main 只多 `b3008b4`（上次 handoff），未推送。本次 handoff **没提交**：钩子把未暂存的 src 改动藏起后在 HEAD 上跑未跟踪的 `tests/test_scalebfm_grip_law.py`，它断言 HEAD 还没有的 `grip_clip` 字段，只提交三份台账必败；台账随整线一起提交（用户说了才提）。
已确认（10-13，pgrep / nvidia-smi）：实例 0–4 空、GPU 0 / 1 空（刹车批次跑完后栈已降）。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败（10-12 合并时修了 `warehouse_mars_200_isaac.yaml`）。

## 提交

- 本线（TODO 24）全部未提交，用户没说提交：`git status --short` 为准。改动 `src/locomani/backends/scalebfm/{runtime,config}.py`、`configs/backend/scalebfm.yaml`、`scripts/experiments/graspnet_bottle_20261008/{grasp_planned,run_grasp}.py`、`scripts/experiments/render_brake_marker_20260923.py`、`docs/{README,scalebfm,walk,reference/config_fields}.md`；未跟踪 `scripts/experiments/wbc_compare_20261012/`、`docs/handoff/wbc-compare-20261012.md`、`tests/test_{scalebfm_grip_law,wbc_attitude,wbc_brake}.py`。
- 已确认（10-13）：`make check` 全树通过（`/workspace/tmp/wbc_make_check7.log`）。提交钩子会先藏起未暂存改动；三个未跟踪的 wbc 测试依赖未暂存的 src 改动，整线要一起提交，不要只提交一半。
- 没提交（不是这条线的、来路不明）：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1.md`–`step7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`。
- STATE.md 上限 60 行（`lint_repository.py`）。

## 磁盘

- 已确认（10-13）：`/workspace` 剩 45 GB（96%）；跑仿真 / pytest / 安装前 `export TMPDIR=/workspace/tmp`。容器内存 91 GB。
- 本线产物 `outputs/wbc_compare_20261012/`（compare / probe / brake，含视频和 50 Hz 轨迹）；上一条线的产物与删不删见 TODO 23.2.7。
- 待用户（更早）：21.3.4.6、22.7、11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7；要删先在这里写明 24 小时。

## 当前进度（TODO 24 全部做完；过程 `docs/handoff/wbc-compare-20261012.md` §1–9）

- 已确认 24.1–24.5（10-12）：受控臂 129 局配对 SONIC 48 对 ScaleBFM 77（p < 0.001）；偏置、腰限修复对 SONIC 都无效（48/130 对 48/129）。
- 已确认 24.6（10-12）：ScaleBFM 姿态包络 31/156 可交付（全在虎口朝左）；按包络过滤的放开臂 48 对 104（p < 0.001）；受控臂重配对 48 对 68。表 `outputs/wbc_compare_20261012/compare/`，视频 `videos/{paired,paired2,free2}/`。
- 已确认 24.7（10-13）：空地 1.5 m 终点线，0.55 / 0.35 / 0.25 × 10 局。越线刹车惯性 ScaleBFM 4.5 / 2.5 / 0.5 cm 对 SONIC 24.7 / 10.4 / 4.0（横向 sd ≤ 1.6 对 4–5 cm）；不越线程序两边 0/30 越线、0 摔，脚尖 ScaleBFM 线前 4–8 cm（sd ≤ 2）对 SONIC 守线 13–22 cm（sd 3.5–12）；SONIC 今日复现 09-23。表 `outputs/wbc_compare_20261012/brake/brake.md`，视频 `brake/videos/`（左 SONIC 右 ScaleBFM），预览 `reel_preview.mp4` 352 KB。
- 假设（未验）：ScaleBFM plan 的短停 1.6–4.5 cm 是常数级跟踪短停、3 cm 前馈能补掉（24.7.1）。推断（没做实验）：65 物体线的 104 对 48 与站位无关——两边都按决策 4 放到 0.38 m 站位，没有走路。
- 读者页只各加了一行：`docs/scalebfm.md`（Sim 无物体场景）、`docs/walk.md`（刹车换控制器）、`docs/README.md` handoff 行。
- 上一条线 23.7（抓取来源 cache / graspnet）的状态见 `git show b3008b4:STATE.md`；23.7.4 待验证未动。

## 下一步（编号见 `TODO.md`，都待用户定）

- **24.7.1**：ScaleBFM plan 终点加 3 cm 前馈再跑 30 局（GPU 1，分钟级），脚尖应停到线前 1–3 cm；另一问：要不要让 ScaleBFM 在 65 物体线真走进去。
- **24.6.1**：按包络给每个姿态单独一个前馈常数（`envelope.json` 已存 `bias_heading_cm`）。
- **23.7.4**：graspnet 来源在现在默认配置下两批 65 局（约 40 分钟）。

## 未解决 / 卡点

- 待用户：24.6.1、24.7.1、23.2.7、23.3–23.5、23.7.4–23.7.6、21.3.5、22.3.6–22.3.8、22.5、22.7、21.3.4.2–21.3.4.4、21.3.4.6、21.3.4.8.1、21.3.1、19.8.1、19.8.3、19.18.3.1、16.14.1、16.15.4、13.6.18。
- 已确认（规则）：SONIC 栈实时，旁边不跑 ScaleBFM 进程；ScaleBFM 仿真确定性，重复局要抖参数（刹车线 ±10 cm）；批次跑着时不改它会再 import 的脚本。
- 已确认（环境）：镜像的 `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` 指向不可写的 `/.uv`，`CUDA_HOME` 是 13.0；numpy 大页缺页会卡内核整理——离线 numpy 重活设 `NUMPY_MADVISE_HUGEPAGE=0`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp
make check
D=scripts/experiments/wbc_compare_20261012; BFM_PY=third_party/Isaac-GR00T/.venv/bin/python; SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
bash $D/brake_batch.sh [outdir]      # 24.7 全批：ScaleBFM marker+plan（GPU 1，分钟级）→ SONIC 栈 3 组（实例 0 独占）→ 汇总 + 视频；<run>/STOP 停；墙钟见 brake/batch.log
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src $BFM_PY $D/bfm_brake.py --program plan --speeds 0.55 --repeats 3 --out /workspace/tmp/brake_smoke   # 单组冒烟
MUJOCO_GL=egl $SIM_PY scripts/experiments/render_brake_marker_20260923.py <run dir>   # 出片 + summary.json
$SIM_PY $D/brake_summary.py --out outputs/wbc_compare_20261012/brake [--no-video]     # brake.md / brake.json / videos
# 65 物体线（bfm_grasp.py、bfm_attitude_probe.py、compare_summary.py）命令见 handoff §5、§8 末的命令块
```
