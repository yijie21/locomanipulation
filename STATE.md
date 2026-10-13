# STATE — 2026-10-13（会话：TODO 25 计划，SONIC → ScaleBFM 替换评测；TODO 24 已提交）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-13）：origin/main = `a2cfb23`；本地 main 多 `b3008b4`（上次 handoff）、`154929b`（TODO 24 整条线）和 TODO 25 计划提交，都**未推送**（用户没说推）。
已确认（10-13，pgrep / nvidia-smi）：实例 0–4 空、GPU 0 / 1 空。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败（10-12 合并时修了 `warehouse_mars_200_isaac.yaml`）。

## 提交

- 已确认（10-13）：TODO 24 整条线作为一个提交 `154929b`（用户在 TODO 25 计划里定的），提交钩子通过。
- 没提交（不是这条线的、来路不明）：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1.md`–`step7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`。
- STATE.md 上限 60 行（`lint_repository.py`）。

## 磁盘

- 已确认（10-13）：`/workspace` 剩 45 GB（96%）；跑仿真 / pytest / 安装前 `export TMPDIR=/workspace/tmp`。容器内存 91 GB。
- TODO 24 产物 `outputs/wbc_compare_20261012/`；TODO 25 产物将放 `outputs/controller_compare_20261013/`。C1–C7 要下载 G1 重定向 LAFAN1，先看剩余空间。
- 待用户（更早）：21.3.4.6、22.7、11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7；要删先在这里写明 24 小时。

## 当前进度

- **TODO 25（计划定稿，未开跑）**：用户要用 ScaleBFM 取代 SONIC，五轮问答定了 35 个决定，判据冻结在 `docs/handoff/controller-compare-20261013.md` §2–§4（DECISIONS 2026-10-13（续））。范围是仿真采集、仿真评测和真机；L1 原语层（B1–B8 stock、W0–W8 mars、C1–C7、S1、扰动 a–h、估计位姿臂）是关口，过了才移植专家跑 L2（每场景 ≥ 40 局配对，标称档 + 真机组合档）。
- 开跑前用户改判据（10-13）：a 按 50 N·m 测（= mars 标称，腰 50、踝 35），摔倒不太高就不否决——写成任一项 10 次摔倒 ≥ 3 次才否决（3/10 是这条线定的数）；h 否决、warehouse 生产档只含 ym 边按推荐。
- **TODO 24（做完，已提交）**：65 物体线受控臂 SONIC 48 对 ScaleBFM 77 / 129（p < 0.001），包络过滤放开臂 104；终点线刹车越线惯性 ScaleBFM 0.5–4.5 对 SONIC 4–25 cm。过程 `docs/handoff/wbc-compare-20261012.md` §1–9。

## 下一步（编号见 `TODO.md`）

- **25.1 否决关已过、25.2 bottle L1 做完（10-13）**：计划 §9–14，表在 `outputs/controller_compare_20261013/{w0,h,bottle}/`。bottle 上没有致命短板；ScaleBFM 停位和正手可达明显更好（B3 63 对 41），虎口朝右（4 对 11）和伸手时躯干姿态不如 SONIC；0.7 m/s 两边都越线。
- **25.3 warehouse L1 做完（10-13）**：计划 §15–17，表在 `outputs/controller_compare_20261013/warehouse/{warehouse_l1.md,extra/}`，视频 `warehouse/videos/`。无致命短板；双手扣箱要夹爪伺服（`--servo pad`），平抬 3–4/8 对 SONIC 6/8；停位散布小 2–17 倍；带载走路两边都差。新增 skills profile `warehouse_box_mars`。
- **下一步 25.4 复杂动作 C1–C7（mars）**：先下载 G1 重定向 LAFAN1（看磁盘），写 IK 关键帧合成；mode 扫描含 25.3.3（mode 6 对 7）；S1 切换瞬态。
- SONIC 实时率作废门槛已改 0.85（用户 10-13）。

## 未解决 / 卡点

- 待用户：25.11.1（机载电脑型号、外部定位；现按没有处理）、24.6.1、24.7.1、23.2.7、23.3–23.5、23.7.4–23.7.6、21.3.5、22.3.6–22.3.8、22.5、22.7、21.3.4.2–21.3.4.4、21.3.4.6、21.3.4.8.1、21.3.1、19.8.1、19.8.3、19.18.3.1、16.14.1、16.15.4、13.6.18。
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
