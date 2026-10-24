# STATE — 2026-10-24 02:10 UTC（主线：场景 RSI，抓取线 grasp_atomic / grasp_prep）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已确认：`origin/main` = `edeb44a0`（10-24 02:00 推送，先合并了 henry-zhao 10-23 的 5 个提交）；之后只有 handoff 提交，没推送。决策和被否决的方案见 `DECISIONS.md` 2026-10-23 – 10-24 一节。

## 提交与磁盘
- 不提交来路不明的文件：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1–7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`、`docs/robotics_rsi_locomanipulation_discussion_2026-10-12.md`、`.playwright-mcp/`、`run_engineer.out`。按名字暂存，不用 `git add -A`。
- henry-zhao 直接推 origin/main：推之前 `git fetch` 再合并；合并提交的钩子跑全套 CPU 测试（5–8 分钟）。只改根目录 `.md` 的提交只跑文档检查。
- 提交前确认：`git config --get core.bare` 是 false，`git config user.name` 是 yijie21（测试改坏过共享配置，29.8 已修）。
- 已确认（10-23 13:00）：`/workspace` 剩 138 GB；A/B 重放、replay2、rsi-phase4 四个臂的 `.git` 打成了副本根目录的 `_sandbox_git.bundle`，恢复步骤 `/workspace/code/rsi-scene/_SANDBOX_GIT_BUNDLES.md`。

## 另一个会话（已确认）
- `locomanipulation-ea`：视觉 RSI 新线，worktree `/workspace/code/locomanipulation-vision-rsi`，分支 `design/vision-rsi-20261023`，不碰 main。共用 12 个仿真槽；它在 GPU 1 上开 Unreal 和 SAM 3 服务。
- 约定：主线开仿真或 RSI 运行前先发消息给它（它会降进程数或关 Unreal）；RSI 循环在跑时它不开 Unreal；Unreal 开着时主线不做占满 CPU 的事。

## 当前进度（已确认）
- grasp_atomic 线：封存集 0 → 31 → 36 → 38 / 50（run B0、B、C，沙盒 `ga-confirm-025-20261021`、`ga-cont-b-20261022`、`ga-cont-c-20261022`）；C 续跑第 13、14 轮没接受，现任仍是 `5cba2fa49d31`（TODO 37.7）。交接文档 `docs/handoff/grasp-atomic-line-20261023.md`。
- grasp_prep 第一次运行（TODO 33.5.1，沙盒 `grasp-prep-a`）：封存集 17 → 27 → 31 / 40；循环写出准备动作 `clear.py`；每轮约 63 分钟；停在验收池 29 / 32。
- Reviewer 默认推理档位 high（37.5，`outputs/scene_rsi_20261016/reviewer_effort_20261023/analysis.md`）；经验库给法不改（33.11）。
- 10-23 做好、还没在任何运行里用过：33.1.2 预测进局记录、33.15.1 记录库、33.20 按层归因、33.19 设计探针、37.8 越界放行场景目录（规则第 20 版）。
- 记录库文件 `outputs/scene_rsi_20261016/records/library.jsonl`：110 个沙盒、15,911 局，没收 A800 的局。
- A800（`ssh -p 26804 root@14.103.60.188`）10-23 起连不上。

## 下一步（编号见 `TODO.md`）
- 37.9：开一次新运行，用 `184cb113` 之后建的沙盒，检验上面五项；跑哪个任务族（续 grasp_atomic 还是 grasp_prep）用户定，开跑前先通知 `locomanipulation-ea`。
- 33.21 / 33.22：A800 能连上后做脆弱起点和算力分工。
- 33.7（三个 Engineer 按范围分工）或 37.6（下一轮 Reviewer 提前跑），用户定。

## 未解决 / 卡点
- 已确认：登记表基准和复现材料是 0.5 m/s + 休眠的记录（`sleep_20261021`），默认已是 0.25 m/s；按用户不重测，回归结论只作参考。
- 已确认：A800 和本机同一设置只有 73 / 80 局结局相同，远端的局不和本机配对。
- 已确认：本机重测 pick_single 总数相近、不逐局相同（用户：不查）。
- 待验证：设计探针、记录库的判定、按层归因在真实运行里帮不帮得上（37.9）。
- 待查：部分会话账本里的 `usage` 偏少（37.5.1），分析一律用会话记录。
- 规则：停进程用明确的 PID 或进程组（`kill -TERM -<pgid>`），不用 `pkill -f` / `pgrep -f` 去匹配写在同一条命令里的文字；不改正在跑的 bash 脚本。

## 关键路径与命令
```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2 && PY=$VENV_SIM_PY
# 新运行：照 outputs/scene_rsi_20261016/grasp_prep_a/launch.sh 写一份（建沙盒、核对文件、loop、封存集），nohup setsid 后台跑
python3 scripts/rsi/sandbox.py $LOCOMANI_WORKSPACE/code/rsi-scene/<名字> grasp_prep
cd <沙盒> && RSI_MAX_WALL_H=8 $PY scripts/rsi/loop.py run grasp_prep --stop-rate 0.9; $PY scripts/rsi/sealed.py grasp_prep --all
# 续跑一个已停的沙盒（沙盒驱动要先按名字同步：loop.py 等单个文件，不拷 families/ experience/ seeds/）
cd <沙盒> && $PY scripts/rsi/loop.py run grasp_atomic --stop-rate 0.9 --max-iters 2
python3 scripts/rsi/records.py build <沙盒> ...        # 原仓库的记录库（写 outputs/scene_rsi_20261016/records/library.jsonl）
python3 scripts/rsi/report/library_monitor.py run <沙盒> --out <目录>/library.json   # 33.11 的两个数
python3 outputs/rsi_speed_20261021/ab/run_ab.py <plan.json> --jobs 8                  # 重放对照（37.5 用过）
nohup setsid $PY scripts/rsi/regress.py run <tag> --procs 12 > /workspace/tmp/regress_<tag>.log 2>&1 &
```
- 场景 `outputs/scene_discovery/adt_apartment_mars/`；沙盒都在 `/workspace/code/rsi-scene/`；登记表 `outputs/scene_rsi_20261016/outer/registry.json`（`last` = `sleep_20261021`）。
- 更早的线（warehouse / bottle 的 ScaleBFM RSI、SONIC、VLA）都停着，见 TODO 1–26。
