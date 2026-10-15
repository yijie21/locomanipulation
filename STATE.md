# STATE — 2026-10-15 03:25 UTC（25.9.1.6 第 4 阶段 A、B 两臂在跑；25.9 待用户）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-15 03:25）：origin/main = `a2cfb23`；本地多 18 个提交，**都没推送**（用户没说推）。
已确认（10-15 03:21）：两臂闭环驱动在跑（`loop.py run --arm A --gpu 1`、`--arm B --gpu 0`，各最多 4 个 ScaleBFM 进程，约 4.7 GB 内存 / 进程）。没有 SONIC 栈和 CuRobo 服务。
已确认：机器 10-14 约 09:44 UTC 重启过一次（原因不明，内核版本变了）；/workspace 上的产物都在。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 提交

- 不是这条线的、来路不明的文件，不提交：`papers/`（含本次下载的 `2610_12424_roborsi.*`）、`node_modules/`、`package*.json`、`uv.lock`、`step1.md`–`step7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`。
- STATE.md 上限 60 行（`lint_repository.py`）。

## 磁盘

- 已确认（10-14 13:19）：`/workspace` 剩 135 GB（87%）。跑仿真 / pytest 前先 `export TMPDIR=/workspace/tmp`。
- 9.12 清理 10-14 03:30 已执行（用户免了 24 小时），记录 `outputs/cleanup_20261013/apply.log`。
- 更早待用户定的删除：21.3.4.6、11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7；要删先在这里写明，等 24 小时。

## 当前进度（方案 `docs/handoff/scalebfm-warehouse-rsi-20261014.md`）

- 已确认：TODO 25.1–25.8 做完（`docs/handoff/controller-compare-20261013.md` §9–25）。L2：bottle 标称 SONIC 40 对 ScaleBFM 39 / 45（掉物 2 对 6，护栏不过）；warehouse 是"专家原样"的 20 对 6 / 40。
- 已确认（25.9.1，§12–16）：warehouse 专家确实是配合 SONIC 调的。ScaleBFM 照骨盆参考深蹲到约 0.61 m，手下不到箱沿。
  - 修法是骨盆关键点不低于 0.74 m + descend 重试 2 次 + 重夹 2 次（`final.yaml`）。
  - 封存集 80 局：ScaleBFM 48 对 SONIC 30（p 0.013），护栏全过。
  - 增益几乎全来自第 1 阶段主会话的干预归因（9 → 46），闭环自己只多 2 局（p 0.5）。
- 已确认（25.9.1.7，§18）：伸手接了 CuRobo 路线（`visual.reach_route planned`，`final_route.yaml`），只接在 ScaleBFM 局里，SONIC 生产线没接。
  - 验收池 45 → 54（p 0.023），封存集 48 → 68（p 0.0001）；途中碰撞 18 → 0 局，一气呵成 11 → 100 / 140 局。
  - 默认 `straight` 逐拍不变（50/50）。
- 假设 / 待验证：带路线后"箱子倾斜"大减（11 → 1）的原因没隔离（25.9.1.7.2）；箱子最后举得更低（12 → 8 cm）没查（25.9.1.7.1）。
- 已确认：封存集已用四次（final、start、phase1_best、final_route），都是事先定的固定配置。
- 进行中（25.9.1.6，§17.1）：两臂各在隔离副本 `/workspace/code/rsi-phase4/arm-{a,b}` 里冷启动，10-15 03:20 UTC 开跑，预算各 12 次验收 / 36 h / 400 USD。
  - 副本基于 `e9f0530`，剔除第 1–3 阶段答案；两臂诊断池 50 局和主检出逐拍一致（含 GPU 0）。角色读主检出、`~/.claude`、`third_party/..` 就整臂停。

## 下一步（编号见 `TODO.md`）

- 25.9 待用户：整体判定（warehouse 已过；bottle 标称 39 对 40）。可选的 25.9.3：bottle 专家同法重调。
- 25.9.1.6：盯两臂到停（`loop.py status`，在副本里跑）；停后补跑 B 的预测（.3），两臂最终配置上封存集（.4），写 §19。
- 25.9.1.7.3：SONIC 生产线要不要也接路线（实时栈等 6 s 规划，要另外验证）。

## 未解决 / 卡点

- 待用户：25.9、25.9.2（真机位姿来源）、25.9.3、25.11.1、9.13、23.2.7、23.3–23.5、23.7.5、23.7.6、21.3.5、22.3.6–22.3.8、22.5、22.7、21.3.1、19.8.1、19.8.3、16.14.1、16.15.4、13.6.18。
- 已确认（规则）：SONIC 栈按实时跑，旁边不能跑 ScaleBFM；闭环角色不许读 `outputs/scalebfm_warehouse_rsi_20261014_hidden/`；`pkill -f` 会杀掉命令行里含同样字样的 shell，用 `pgrep -f '[c]urobo_service'` 拿 PID。
- 已确认：ScaleBFM 每个进程第一局要先 `warm_up`（TorchScript 首次调用差 6e-8），`bfm_round.py` 已带。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src OMP_NUM_THREADS=2
H=scripts/experiments/scalebfm_warehouse_rsi_20261014; O=outputs/scalebfm_warehouse_rsi_20261014; PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
bash $H/route_service.sh                                           # CuRobo 路线服务（GPU 0，端口 5795），reach_route planned 前必须起
CUDA_VISIBLE_DEVICES=1 $PY $H/bfm_round.py $O/pool_diag/states --spec $H/final_route.yaml --out /workspace/tmp/x --shard 0/4   # 一组局
$PY $H/route_eval.py                                               # 带 / 不带路线配对 -> $O/route/eval.md
$PY $H/sealed.py judge                                             # 封存集 final / start 对 SONIC -> $O/sealed/sealed.md
$PY $H/loop.py status                                              # 闭环台账（第 2 阶段）
(cd /workspace/code/rsi-phase4/arm-a && $PY $H/loop.py status)     # 第 4 阶段臂 A；arm-b 同
```
