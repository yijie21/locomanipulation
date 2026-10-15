# STATE — 2026-10-15 17:45 UTC（过夜计划 26.5.6：SONIC 接路线失败按约定停；ScaleBFM 冷启动进行中）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-15 16:10）：origin/main = `a2cfb23`；本地多 31 个提交，**都没推送**（用户没说推）。
已确认（10-15 17:00）：没有任何实验在跑；本会话起的路线服务、测量进程都已停，GPU 0 / 1 空。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 提交与磁盘

- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`。STATE.md 上限 60 行。
- 已确认（10-15）：`/workspace` 剩 122 GB（89%）。跑仿真 / pytest 前先 `export TMPDIR=/workspace/tmp`。副本 `/workspace/code/rsi-phase4/arm-{a,b,a2,b2}` 共约 10 GB，作证据保留。

## 当前进度

- 已确认（25.9.1.6，方案 `docs/handoff/scalebfm-warehouse-rsi-20261014.md` §17–19）：第 4 阶段每臂跑了两次。
  - 封存集 80 局：臂 A 68 / 68（两次是同一配置 `root_z_min 0.74` + `carry.base_anchor: reference`，逐拍相同）。
  - 扁平臂 B 57 / 40；第 2 阶段 final 48；SONIC 30。
- 已确认（26.1–26.2）：warehouse 对比视频 7 段、进化视频 4 段已导出到 `outputs/rsi_demo_20261015/warehouse/{previews,evolution/previews}/`，都 ≤ 300 KB。
- 已确认（26.3.1）：bottle 适配层在 `scripts/experiments/scalebfm_bottle_rsi_20261015/`（方案 `docs/handoff/scalebfm-bottle-rsi-20261015.md`）。
  - 诊断池 55 局；验收池 60 个起点。SONIC 在这些局上的分段式成败已作废，起点仍可用。
- 已确认（用户 10-15）：不要分段式抓取，只用 CuRobo 规划；两边都用；bottle 规划到举起位；先 warehouse（DECISIONS 10-15 末节）。
- 已确认（26.5.5，方案 §20，`outputs/rsi_speed_20261015/`，提交 `6b017b6`）：
  - 路线规划 5.5 → 0.34 s（`--no-self-collision-opt`，50 条路线逐位不变）；路线缓存 + 多服务实例，换分片重跑 12/12 逐拍相同（= 26.5.1）。
  - `carry.ik_jacobian: analytic`：单进程 3.4 倍；诊断池 37 → 33，与 OMP 浮点对照（33）分不开；会改结局，只在冷启动用。
  - `batch.py` + 12 进程 + `--gpu 0,1`：吞吐 3.15 → 12.1 仿真秒 / 秒；OMP 必须保持 2（OMP 1 改结局）。

## 过夜进度（26.5.6，用户 10-15 确认"按建议"）

- [x] 提交 26.5.5（`6b017b6`）。
- [~] SONIC 接路线：代码已改（`expert.py` 的 StreamedRoute / route_planner_for，`visual.reach_route_fallback`、`visual.reach_route_prior`），未提交。
  - 冒烟 10 局（种子 46000/46100，2 通道）：下垂先验 0/10，降速 0.5 rad/s 也 0/10。路线都规划成功（0.34–0.38 s），失败在路线之后。
  - 到达姿态腰更弯（俯仰中位 0.36 对分段式 0.21 rad），SONIC 腰只交付约 0.6。
  - 加试（超出约定，是我的判断）：`reach_route_prior: lifted` 1/10；同种子分段式基线 3/10。路线 3 种合计 1/30，腰仍弯（0.38 rad），原因未查清。
  - **按约定停下：不做 SONIC + CuRobo 对比视频。** ScaleBFM 冷启动改用现有起点池，停止条件对生产版 SONIC（验收池 27）。
- [ ] 冷启动副本 → RSI 闭环 → 封存集 → 进化视频 → 查看页。

## 下一步（编号见 `TODO.md`；用户说开始后才动）

- 26.5.5.6 冷启动配置：`carry.ik_jacobian: analytic` + `visual.reach_route: planned`，`--gpu 0,1`，先起 `route_service.sh` 再 export `TF_CUROBO_PORTS`。
- 26.5.2 路线接进 SONIC warehouse 生产专家（`src/locomani/tasks/crate/expert.py:532` 构造 CoordinatedReach），小批量试跑。
- 26.5.3 warehouse 从"原样移植 + CuRobo"冷启动 RSI，SONIC + CuRobo 跑验收池 / 封存集，导出对比和进化视频。

## 未解决 / 卡点

- 待验证：SONIC 实时栈接路线后规划约 0.34 s（原 6 s），腰只交付一半，路线跟得好不好（25.9.1.7.3 = 26.5.2）。
- 待观察：解析 IK 在诊断池掉物 3 → 6 局（对照 3），冷启动后看验收门的掉物护栏。
- 已确认：这台机器上 bottle 栈的实时率不管录不录制都在 0.80–0.86；bottle 池的有效线是 0.80（bottle 方案 §4）。
- 已确认：`make check` 里 `tests/test_stack_start_failure.py::test_down_kills...` 在全量负载下偶发超时，单独跑 3/3 通过，和本次改动无关。
- 已确认（10-15 17:00）：auto-memory 的 MEMORY.md 正好 200 行（上限），再加一条就要先合并或删一条。
- 规则：SONIC 栈在跑时不跑 ScaleBFM；`pkill -f` 会杀掉命令行含同样字样的 shell，杀进程要写进脚本文件里用 `pgrep -f '[x]...'`。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src OMP_NUM_THREADS=2
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python; W=scripts/experiments/scalebfm_warehouse_rsi_20261014
N=2 GPUS=0,1 bash $W/route_service.sh /workspace/tmp/route_service.log  # 路线服务 ×2，打印要 export 的 TF_CUROBO_PORTS；停：outputs/rsi_speed_20261015/stop_services.sh
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 $PY scripts/experiments/rsi_demo_20261015/compare.py --task warehouse    # 对比视频
MUJOCO_GL=egl $PY scripts/experiments/rsi_demo_20261015/evolution.py /workspace/code/rsi-phase4/arm-a2 --task warehouse --out outputs/rsi_demo_20261015/warehouse/evolution
```
