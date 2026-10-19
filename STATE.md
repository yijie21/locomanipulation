# STATE — 2026-10-19（场景 RSI 第 2 轮外环暂停（用户 10-19）：内环 2/6 做完；效率与效果九条（TODO 27.10）做完，27.10.8 还没整轮跑过）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-19 03:20）：origin/main = `a2cfb23`；本地多 84 个提交（含本次），**都没推送**（用户没说推）。
已确认（10-19）：**第 2 轮暂停**（用户：“现在先暂停所有的尝试”）。`r2_on_single` 写回，`r2_in_single` 5 门全拒（种子 15/80，不写回）；后面 `r2_held_near_single`、`r2_held_single`、`r2_relay_regrasp` 没跑，`r2_in_room_none` 无效（“走到卧室”开局就成立）。Reviewer 回放（27.10.6）第一次有泄漏，结论撤回；经验库对比（27.10.6.1）10-19 跑完，不接（慢 13%、轮数多 10%，探针 31 对 32）；干净沙盒 `/workspace/code/rsi-scene/replay2-r2{in,on}{,-b,-c}`（共 8.4 GB）作证据保留，结果在各沙盒 `outputs/scene_rsi_20261016_hidden/<族>/replay/`。用户 10-19 取消模型费上限（27.11：内环 150 USD、D9 2000 USD），判断只看效果和效率。做成的任务（`successes.py`，方案 §13）：练过的 49 个目标 22 个 ≥ 50%、3 个有成功但不到一半；没练过的 107 个 40 个、9 个。进度报告（27.12）：`docs/handoff/scene-rsi-report-20261019.md`，网页版 https://claude.ai/artifact/1YY339VYjnSk8FTG1WCfBN（视频在 `outputs/scene_rsi_20261016/report/media/`）。
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

- 27（方案 `docs/handoff/scene-rsi-plan-20261016.md`）：第 0 阶段（§7）、第 1 阶段能力探针（§9，表 `configs/scene_discovery/capability_mars_scalebfm.yaml`，`capability_write.py` 生成；27.3.1 可站边覆盖文件 `--approach-edges`）都完成。**第 2 阶段完成（§10、里程碑 1 §10.1）**：第 1 轮发现 60 个任务（`outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round1/`），5 组内环按顺序跑并写回：`r1_on_single` → `r1_held_single` → `r1_in_single` → `r1_held_near_single` → `r1_slide_then_grasp`。第 1 桶 60 个里 4 → 16，2830 模板 4 → 41；模型费 309.46 USD（估 750），内环墙钟 24.8 h，门 19 次；插手：内环中 1、开跑前规则 1、写回时改检查器 / 能力表程序 4、用户 0。第 1 轮末目录下各族封存集：`pick_single` 58、`r1_on_single` 24、`r1_held_single` 39、`r1_in_single` 9、`r1_held_near_single` 33、`r1_slide_then_grasp` 38（各 /80，重测局在各沙盒 `sealed/catalog_after_r1_slide/`）。写回步骤：拷技能、参数写进常量 → 用新目录重测所有族封存集（`/workspace/tmp/wb6_regress.sh` 是最新一份）→ `capability_write.py --loop-root .../pick-a --pick-rounds ... --held-loop/--place-loop/--into-loop/--near-loop <沙盒>:<族>[:<重测目录>]` → `scene_feasibility.py --approach-edges` 重新分桶。**用户 10-18 定的都做完（27.4.6，§10.2）**：门第一阶段按任务分层、零起步族掉物上限、D13.4 最少起点（规则第 5 版）；蜡烛 `under_rim`；台面板 1 mm 余量（静置自转 33 → 5）；家具面板 + 沉降（`scene_add_panels.py`、`scene_settle.py`；自己掉下去的 5 → 0）；外环种子块。新物理基线各族封存集在各沙盒 `sealed/physics_20261018/`：59、25、34、9、31、40。场景备份 `outputs/scene_discovery/adt_apartment_mars.bak_20261018_*`。**第 3 阶段（27.5）开跑**：发现 `/workspace/tmp/discover_r2.sh` → `outer.py select .../outer/round2` → 族 `r2_*`（种子从 1,110,000 起）→ 内环同第 2 阶段。待用户定：碰撞检查层、换模型对照；还剩的慢漂移（鼠标在垫上约 1 mm/s）。
- 27.10（方案 §12，用户 10-19“先把这效率和效果相关的 4+5 条先完成”）：九条代码都已提交（`b84c407`、`183ee66`、`8ced519`）。检查器标定：`cal_r2` 判第 1 桶 11/24 ≥ 50%；加“搬运时长对夹持余量”规则后 `cal_r3` 13/24（54%）；再加放进容器两条、顶部夹距离、标定样本登记成实测（`registry.json` 里 `cal_r2..r4`，用户 10-19）后 `cal_r4` 12/23（52%），现在第 1 桶的 63 个目标都跑过（方案 §12.3）。竖直轴已应用（8 族封存集逐局相同，`sealed/upright_20261019/`）。外环驱动 `outer_run.py`（`writeback.py`、`regress.py` 登记表 `outputs/scene_rsi_20261016/outer/registry.json`）还没整轮跑过，恢复第 2 轮时用它。
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
