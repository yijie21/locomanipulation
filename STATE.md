# STATE — 2026-10-28（主线：场景 RSI，抓取线 grasp_atomic / grasp_prep）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，最新提交见 `git log --oneline -5`（没推送）。

## 提交与磁盘
- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`.playwright-mcp/`、
  `assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_*.md`、
  `run_engineer.out`、根目录的 `reports/`、`research_notes/`。`outputs/`、`third_party/`、`datasets/` 永不进仓库。只按路径 `git add`。
- 提交前确认 `git config --get core.bare` 是 false、`user.name` 是 yijie21。提交钩子对代码改动跑全套 CPU 检查（几分钟、多核）。
- **已确认：10-27“连着做 TODO”的改动除 37.5.1 / 33.19.1 外都没提交**（这两项 11-06 随 TODO 43.1 提交；另有新文件 `scripts/scene_mesh_budget.py`、
  `tests/test_scene_mesh_budget.py`），10-27 最后一次 `make check` 通过。用户说提交再提交，按 TODO 分组、按路径 add：
  42.2 `start.py` + `test_scene_task_runner.py`；4.7.8 `build_scene.sh`、
  `adt_to_scene.py`、`scene_mesh_budget.py`；10.7 `crate/collect.py`；其余是文档和 warehouse 循环的提示词。
- 已确认（10-27 08:20）：`/workspace` 可用 72 GB（93%）。Kimodo、CHOIS、GMR 已删（重装：`outputs/scene_rsi_20261016/traj_prior_20261026/install_kimodo.sh`）。

## 另一个会话（已确认）
- `locomanipulation-ea`：视觉 RSI 线，worktree `/workspace/code/locomanipulation-vision-rsi`，分支 `design/vision-rsi-20261023`。
  主线开仿真 / RSI / Unreal 前先发消息；Unreal 编辑器开着时整机 CPU 不能打满（`editor_guard.sh`，ffmpeg 加 `-threads 2`）。
- 它的封存池和 `grasp_atomic_refs/pool_sealed` 逐字节相同（42.1、42.2 的替换它都照做了）；新地图上它的 L0 是 37 / 50
  （没换的 37 个起点 30 → 30），它分支上的提交 e1325906、1c47a609。旧的视觉数字都是旧地图上的。

## 当前进度（已确认）
- Reviewer 改读失败卡片（TODO 43.1，规则 37，`rules.md` version 27）：两次真实回放跑通到 Prober，完整一轮只用桩跑过。下一步 43.2：两个并行的 Reviewer。
- grasp_atomic 现任仍是 run C 的 `5cba2fa49d31`。新地图（TODO 42）上 120 个位置赢 88（旧图 86），走路摔倒 7 → 0
  （`outputs/scene_rsi_20261016/walkmap_20261027/README.md`）。旧图上的封存 38 / 50 照旧算数，只是地图更准了（用户 10-27）。
- 场景文件已改（42.1）：sha256 主场景 `abe78c1f…`、可移动变体 `3395c67c…`；旧文件 `outputs/scene_discovery/<场景>.bak_20261027_before_walkmap/`。
  起点池里被换掉的原文件在 `<池>_replaced_20261027/`，映射记在各池 `_pool.json`。
- TODO 40（CuRobo 路线开关）三个变体都不净涨，默认关。41.1 / 41.2 离线第一步做完，等用户定落点。
- 25.9 控制器替换判定写在 `docs/handoff/controller-compare-20261013.md` §26：warehouse 重标后可以换，bottle 还不行，真机没法判。
- 进展报告：https://claude.ai/artifact/5KrdvZBjGQikFjnEpj1oYY 第 13 版（没加 10-27 的内容）。

## 下一步（编号见 `TODO.md`）
- 41.2：用户定落点（建议库原语：进入后 < 3 cm、下降后 < 5 cm 才合爪，否则退回重来一次），定了就实现并跑一次真运行。
- 41.1 / 33.2：用户定要不要采用“只算目标的账”（建议采用）。
- 其余待定的都在 DECISIONS 10-27“连着做 TODO”一节的“暂停”里（A800 三项、37.6、36.9、36.10、39.3、38.5–38.10、23.4.3）。

## 未解决 / 卡点
- 已确认：本机仿真确定（同代码同起点结局相同）；A800 和本机同设置 73 / 80 局相同，不配对；A800 只跑用户指定的实验。
- 已确认：`loop.session_usage` 只修了以后的账；10-27 以前账本里的 token 数会少记（花费 `total_cost_usd` 一直是对的），要准数读各轮 `*_raw.json` 的 `modelUsage`。
- 已确认：0.41 m 出生规则去掉的多是离家具近、走得短的“容易”起点（两个技能包在换掉的 5 个封存起点上都是原 4 / 5 → 新 1 / 5），成功率小跌是选起点的效应。
- 待验证：41.2 的“退回重来”能救回多少（离线只知道 18 局偏 ≥ 3 cm 合爪只赢 1；COAP 报重试救回 17%）。
- 待验证：创新点提议（DECISIONS 10-27 讨论一节）没查重。
- 规则：停进程用明确 PID 或进程组，不用 `pkill -f` / `pgrep -f`；CuRobo 服务按 `pgrep -f '[c]urobo_service'` 拿 PID 停；不改正在跑的 bash 脚本。

## 关键路径与命令
```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 && PY=$VENV_SIM_PY
# 走路地图：补静态落地物体（只改障碍物 / 出生区 / 站位边；--check 只打印）
/venv/main/bin/python scripts/adt_to_scene.py walkmap --scene outputs/scene_discovery/adt_apartment_mars/scene_meta.json [--check]
# 起点池：旧图 / 新图（或加出生净空规则）下逐个核对、替换
W=outputs/scene_rsi_20261016/walkmap_20261027; $PY $W/pools.py scan --old <旧 scene_meta> --new <新 scene_meta> [--clear 0 0.41 --tag _422]
# 现任在 120 个起点上（comp_ab 的三个池）重跑与对比
$PY $W/run_incumbent.py run --procs 12; $PY $W/run_incumbent.py report
# 设计探针单独跑一个原语（33.19.1）；建场景的网格闸门（4.7.8）
$PY scripts/rsi/design.py grasp_atomic --pool /tmp --skills /tmp --out <dir> --factor primitive --name turn --levels 90 180
python3 scripts/scene_mesh_budget.py outputs/scene/<名字>/scene_locomani.xml
# 新运行：照 outputs/scene_rsi_20261016/grasp_prep_a/launch.sh（建沙盒、核对、loop、封存集），nohup setsid 后台跑
```
- 场景 `outputs/scene_discovery/adt_apartment_mars/`；沙盒都在 `/workspace/code/rsi-scene/`；comp_ab 的池在 `outputs/scene_rsi_20261016/body/comp_ab_20261026/`。
