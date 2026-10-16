# STATE — 2026-10-16（场景 RSI 第 0 阶段完成；内环 5 轮，验收池 0 → 43.3%，封存集 51.2%；之后等用户说）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-15 16:10）：origin/main = `a2cfb23`；本地多 31 个提交，**都没推送**（用户没说推）。
已确认（10-16 09:31）：机器空闲。沙盒 `/workspace/code/rsi-scene/pick-a` 的内环跑完第 4 轮停了（`--max-iters 3`）；继续在沙盒里 `$PY scripts/experiments/scene_rsi_20261016/loop.py run pick_single --max-iters N`，查看 `... loop.py status pick_single`。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 提交与磁盘

- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`。STATE.md 上限 60 行。
- 已确认（10-15）：`/workspace` 剩 122 GB（89%）。跑仿真 / pytest 前先 `export TMPDIR=/workspace/tmp`。副本 `/workspace/code/rsi-phase4/arm-{a,b,a2,b2}` 共约 10 GB，作证据保留。

## 当前进度

- 已确认（25.9.1.6，方案 `docs/handoff/scalebfm-warehouse-rsi-20261014.md` §17–19）：第 4 阶段每臂跑了两次。
  - 封存集 80 局：臂 A 68 / 68（两次是同一配置 `root_z_min 0.74` + `carry.base_anchor: reference`，逐拍相同）。
  - 扁平臂 B 57 / 40；第 2 阶段 final 48；SONIC 30。
- 已确认（26.1–26.2）：warehouse 对比视频 7 段、进化视频 4 段已导出到 `outputs/rsi_demo_20261015/warehouse/{previews,evolution/previews}/`，都 ≤ 300 KB。
- 已确认（26.3.1）：bottle 适配层在 `scripts/experiments/scalebfm_bottle_rsi_20261015/`（诊断池 55、验收池 60 个起点仍可用）。
- 已确认（用户 10-15）：不要分段式抓取，只用 CuRobo 规划；两边都用；bottle 规划到举起位；先 warehouse（DECISIONS 10-15 末节）。
- 已确认（26.5.5，方案 §20，`outputs/rsi_speed_20261015/`，提交 `6b017b6`）：
  - 路线规划 5.5 → 0.34 s（`--no-self-collision-opt`，50 条路线逐位不变）；路线缓存 + 多服务实例，换分片重跑 12/12 逐拍相同（= 26.5.1）。
  - `carry.ik_jacobian: analytic`：单进程 3.4 倍；诊断池 37 → 33，与 OMP 浮点对照（33）分不开；会改结局，只在冷启动用。
  - `batch.py` + 12 进程 + `--gpu 0,1`：吞吐 3.15 → 12.1 仿真秒 / 秒；OMP 必须保持 2（OMP 1 改结局）。

## 过夜进度（26.5.6，用户 10-15 确认"按建议"；方案 §21）

- SONIC 接 CuRobo 路线：接上了，但同种子分段式 3/10，路线三种变体合计 1/30（腰更弯，原因未查清）。**按约定停，没做对比视频。**
- ScaleBFM 冷启动（`/workspace/code/rsi-curobo/arm-c`，17:54–20:40，38.3 USD，插手 1 次）：验收池 5 → 21 → 43 停。
  - 封存集：起点 5、终版 54、生产版 SONIC 30；比人工修法 + 路线（68）少 14（p 0.02）。
  - 进化视频 4 段（133–342 KB）：`outputs/rsi_demo_20261015/warehouse_curobo/evolution/previews/`。
- 查看页（私有）：https://claude.ai/artifact/Xu14Az6FoSwbbztBazzyyJ ，源文件 `outputs/rsi_demo_20261015/warehouse_curobo/index.html`。
- 路线服务 5795 / 5796 已停；机器空闲。

## 下一步（编号见 `TODO.md`；都待用户定）

- 27（方案 `docs/handoff/scene-rsi-plan-20261016.md` §7）：第 0 阶段已完成（27.2.1–27.2.7）。内环第 0–3 轮都过了验收门，第 4 轮被拒：验收池 0 → 7 → 13 → 21 → 26 / 60（§7.7–7.9）；第 1–3 轮采纳的是 Engineer 写的代码。书和钱包 14 局判做不到。封存集 80 局 0 → 11 → 24 → 35 → 41（51.2%，刚过 D8 第 1 桶线，每版都没少成一局，§7.10）。模型费 57.26 USD，门 5 / 12 次，插手 3 次（都是驱动或进程问题）。D13（做不到的任务，方案 §8）用户已定、全部按推荐，实现是 TODO 27.8（约半天）。下一步（27.8 / 第 1 阶段 / 碰撞检查层 / 换模型对照 / 合回主仓库 / 27.9 `held` 假成功）等用户说。
- RSI 进化 slides（私有）：https://claude.ai/artifact/N9rLqruxRDpjDzwNKyzaeS；pick_single 进化图解（线索图 + 前后对比视频，私有）：https://claude.ai/artifact/FPhdLToZ7ebhzVtPHM73Zv ，源 `outputs/scene_rsi_20261016/visual/`（`visualize.py`）
- 26.5.6.8：查不查 SONIC 走路线时腰更弯的原因；冷启动要不要提高停止线续跑（现在停在 SONIC + 5）。
- 26.5.6.9：bottle（26.5.4）和改方法（26.5.5.4）。

## 未解决 / 卡点

- 已确认（10-15）：解析 IK 下冷启动终版封存集掉物 4（SONIC 15）；第 0 轮候选因掉物 5 > 3 被验收门拒。
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
