# STATE — 2026-10-15 16:10 UTC（RSI 演示改为全 CuRobo，暂停中，等用户说开始；TODO 26.5）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-15 16:10）：origin/main = `a2cfb23`；本地多 31 个提交，**都没推送**（用户没说推）。
已确认（10-15 15:40）：没有任何实验在跑（SONIC 栈、ScaleBFM 局、CuRobo 服务、闭环驱动、流水线都已停），GPU 0 / 1 空。用户要求暂停，等他说开始。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 提交与磁盘

- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`。STATE.md 上限 60 行。
- 已确认（10-15）：`/workspace` 剩 122 GB（89%）。跑仿真 / pytest 前先 `export TMPDIR=/workspace/tmp`。副本 `/workspace/code/rsi-phase4/arm-{a,b,a2,b2}` 共约 10 GB，作证据保留。

## 当前进度

- 已确认（25.9.1.6，方案 `docs/handoff/scalebfm-warehouse-rsi-20261014.md` §17–19）：第 4 阶段每臂跑了两次。
  - 封存集 80 局：臂 A 68 / 68（两次是同一配置 `root_z_min 0.74` + `carry.base_anchor: reference`，逐拍相同）。
  - 扁平臂 B 57 / 40；第 2 阶段 final 48；SONIC 30。
- 已确认（26.1–26.2）：warehouse 对比视频 7 段、进化视频 4 段已导出到 `outputs/rsi_demo_20261015/warehouse/{previews,evolution/previews}/`，都 ≤ 300 KB。
  - 两边都是**分段式**伸手：先原地抬手，再直线伸。
- 已确认（26.3.1）：bottle 适配层在 `scripts/experiments/scalebfm_bottle_rsi_20261015/`（方案 `docs/handoff/scalebfm-bottle-rsi-20261015.md`）。
  - 诊断池 55 局；验收池 60 个起点。SONIC 在这些局上的分段式成败已作废，起点仍可用。
- 已确认（用户 10-15）：不要分段式抓取，只用 CuRobo 规划；两边都用；bottle 规划到举起位；先 warehouse（DECISIONS 10-15 末节）。
  - bottle 流水线、SONIC 封存集和刚起的 CuRobo 确定性测试都已停，测试没有结果。

## 下一步（编号见 `TODO.md`；用户说开始后才动）

- 26.5.1 CuRobo 路线局的确定性：同一批诊断局单进程 / 双进程交错各跑一遍，逐拍比 qpos。
- 26.5.2 路线接进 SONIC warehouse 生产专家（`src/locomani/tasks/crate/expert.py:532` 构造 CoordinatedReach），小批量试跑。
- 26.5.3 warehouse 从"原样移植 + CuRobo"冷启动 RSI，SONIC + CuRobo 跑验收池 / 封存集，导出对比和进化视频。

## 未解决 / 卡点

- 待验证：CuRobo 规划是否逐次一致。22.2 时 119 个请求重放有 117 个成败相同，这不等于轨迹逐位相同；不一致的话，验收门的哨兵和配对要先处理。
- 待验证：SONIC 实时栈接路线时要等约 6 s 规划，腰只交付一半，路线跟得好不好（25.9.1.7.3 = 26.5.2）。
- 已确认：这台机器上 bottle 栈的实时率不管录不录制都在 0.80–0.86；bottle 池的有效线是 0.80（bottle 方案 §4）。
- 已确认：`make check` 里 `tests/test_stack_start_failure.py::test_down_kills...` 在全量负载下偶发超时，单独跑 3/3 通过，和本次改动无关。
- 已确认：auto-memory 的 MEMORY.md 有 201 行，超出加载上限，最后一行会被截掉，要精简。
- 规则：SONIC 栈在跑时不跑 ScaleBFM；`pkill -f` 会杀掉命令行含同样字样的 shell，杀进程要写进脚本文件里用 `pgrep -f '[x]...'`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src OMP_NUM_THREADS=2
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python; W=scripts/experiments/scalebfm_warehouse_rsi_20261014
bash $W/route_service.sh /workspace/tmp/route_service.log          # CuRobo 路线服务（GPU 0，5795）；停：pgrep -f '[c]urobo_service'
bash /workspace/tmp/route_det/run.sh                                # 26.5.1 的确定性测试（规格 /workspace/tmp/route_det/a_route.yaml）
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 $PY scripts/experiments/rsi_demo_20261015/compare.py --task warehouse    # 对比视频
MUJOCO_GL=egl $PY scripts/experiments/rsi_demo_20261015/evolution.py /workspace/code/rsi-phase4/arm-a2 --task warehouse --out outputs/rsi_demo_20261015/warehouse/evolution
```
