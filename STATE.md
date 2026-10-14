# STATE — 2026-10-14（TODO 25.4–25.8 做完，等用户做 25.9 最终判定）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-14）：origin/main = `a2cfb23`；本地 main 比它多 9 个提交，最新 `fc18526`（25.4–25.8），**都没推送**（用户没说推）。
已确认（10-14 02:50 UTC，pgrep / nvidia-smi）：没有 SONIC 栈，也没有 ScaleBFM 进程；GPU 0 / 1 空；看门狗和队列脚本都已退出。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 提交

- 不是这条线的、来路不明的文件，不提交：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1.md`–`step7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`。
- STATE.md 上限 60 行（`lint_repository.py`）。

## 磁盘

- 已确认（10-14 03:31 UTC）：`/workspace` 剩 150 GB（86%）。跑仿真 / pytest / 安装前先 `export TMPDIR=/workspace/tmp`。
- **9.12 清理已执行**（10-14 03:30 UTC，用户当场说不用等 24 小时）：6 个 worktree、三个冒烟 `model/`、8902 个 ≥ 1 MB 文件（47.9 GB），28 → 150 GB。
  - 记录 `outputs/cleanup_20261013/apply.log`，worktree 删前状态在 `worktree_state/`。`refactor-step1` 的 `outputs` / `third_party` 软链接先 unlink 再删。
  - `/workspace/tmp/p03/deploy.log` 2.3 GB 截到最后 20 MB（只有两个 9 天前的 `tail -F` 开着它）。22.7 的 CuRobo 产物没动。
- 更早待用户定的删除：21.3.4.6、11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7；要删先在这里写明，等 24 小时。

## 当前进度

- 已确认：TODO 25 的 25.1–25.8 做完，计划和结果在 `docs/handoff/controller-compare-20261013.md` §9–25。
  - L1 关口：通过（§21）。
  - L2（§25），SONIC 对 ScaleBFM：bottle 标称 40 对 39 / 45、组合 2 对 35 / 46；warehouse 标称 20 对 6 / 40（p 0.003）、组合 1 对 4 / 40。
  - 按 §2：主判据不满足；bottle 标称档掉物护栏不过；组合档否决不成立。
- 已确认：SONIC 怕动作延迟。40 ms 时 bottle 10 局摔 9 局，warehouse 一局都没站稳（§24）。
- 已确认：ScaleBFM 换成腿运动学 + IMU 估计位姿后，刹车 30/30 越线，两个移植专家都失败。原因是它走路时滑脚，估计跟着漂。
- 已确认（25.9.1，10-14）："warehouse 专家是配合 SONIC 调出来的"。ScaleBFM 照专家的骨盆参考深蹲到约 0.61 m，手下不到箱沿；骨盆限在 ≥ 0.74 m、加 descend 重试后，封存集 9 → 46/80（SONIC 30）。
- 10-13 晚用户的指示已经做完：24 加前馈（handoff §10）、23 抓取来源定为缓存、22.7 产物先不删。

## 下一步（编号见 `TODO.md`）

- 25.9.1：第 0–3 阶段做完（方案 `docs/handoff/scalebfm-warehouse-rsi-20261014.md` §12–16）。封存集 ScaleBFM 重标后 48/80 对 SONIC 30/80（p 0.013），护栏全过；增益几乎全来自第 1 阶段的干预归因（闭环自己只多 2 局）。第 4 阶段（§17）待用户定；25.9 整体判定也待用户（bottle 标称仍 39 对 40）。
- 25.9.1.7：伸手接了 CuRobo 路线（方案 §18，`visual.reach_route planned`，只在 ScaleBFM 局里）：封存集 48 → 68/80，途中不碰箱子。用前先起服务 `scripts/experiments/scalebfm_warehouse_rsi_20261014/route_service.sh`（GPU 0，端口 5795）。
- 25.9.2 待用户：ScaleBFM 上真机用什么做世界位姿。
- 用户没定 25.9 之前不开 25.10。待查 25.8.3（SONIC bottle 一局挂住 12 分钟）。

## 未解决 / 卡点

- 待用户：25.9.1.6（第 4 阶段）、25.9 整体判定、25.9.2、25.11.1（机载电脑型号、有没有外部定位）、23.2.7、23.3–23.5、23.7.5、23.7.6、21.3.5、22.3.6–22.3.8、22.5、22.7、21.3.1、19.8.1、19.8.3、16.14.1、16.15.4、13.6.18。
- 已确认（规则）：SONIC 栈按实时跑，旁边不能跑 ScaleBFM；长队列要配看门狗（`$D/queue_watchdog.sh`）；`pgrep -f` 写在等待循环里会匹配到循环自己。
- 已确认（环境）：py-spy 看别的进程需要 sudo，加了仍被拒；离线 numpy 重活设 `NUMPY_MADVISE_HUGEPAGE=0`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp
make check
D=scripts/experiments/controller_compare_20261013; SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
python3 $D/p25_summary.py                                                  # 25.5 总表 -> outputs/controller_compare_20261013/p25/p25.md
$SIM_PY $D/l2_measure.py --out outputs/controller_compare_20261013/l2     # L2 表和 §2 判定 -> l2/l2.md
$SIM_PY $D/clip_measure.py outputs/controller_compare_20261013/complex/{bfm,sonic} --out outputs/controller_compare_20261013/complex
# 整条 SONIC 队列 / ScaleBFM 批次：计划 §23 的命令块；单项命令见各节末尾
```
