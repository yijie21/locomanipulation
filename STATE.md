# STATE — 2026-10-26 02:10 UTC（主线：场景 RSI，抓取线 grasp_atomic / grasp_prep / grasp_floor）

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
- run D（TODO 37.9，沙盒 `ga-cont-d-20261024`，10-24 02:14–09:18，约 123 USD）：从 C 第 15 轮接着跑 6 轮，12 次验收门全拒，现任不变、封存集 38 / 50；第 16 轮起 10 个候选全是 35 对 33（多赢 2、少赢 0），差一局净赢。37.9 的五项都在运行里起作用；暴露的两处毛病 10-24 修了：37.9.1（转角探针转完放不下就就近挪，并报丢掉的原因，规则第 21 版）、37.9.2（越界检查的 `..` 只拦碰到东西的路径，最后一节链接不跟）。
- grasp_prep 第一次运行（TODO 33.5.1，沙盒 `grasp-prep-a`）：封存集 17 → 27 → 31 / 40；循环写出准备动作 `clear.py`；每轮约 63 分钟；停在验收池 29 / 32。
- Reviewer 默认推理档位 high（37.5，`outputs/scene_rsi_20261016/reviewer_effort_20261023/analysis.md`）；经验库给法不改（33.11）。
- 33.1.2 预测进局记录、33.15.1 记录库、33.20 按层归因、33.19 设计探针、37.8 越界放行场景目录（规则第 20 版）：run D 里第一次用上（37.9）。
- 记录库文件 `outputs/scene_rsi_20261016/records/library.jsonl`：110 个沙盒、15,911 局，没收 A800 的局。
- A800（`ssh -p 26804 root@14.103.60.188`）10-23 起连不上。
- 10-24/25 “抓取能力”一类（TODO 33.1、33.9、33.13、33.24，提交 `59fff724`–`a0f83846`）：双手夹持查询、技能运行时查量过的曲线（快照、整次运行不变）、关键点约束抓取原语（规则第 23 版第 36 条，闭环 5 / 30 不写抓取代码）、流程迁移和推广检查（`pick_any` 任务族）、可推家具的场景变体 `adt_apartment_mars_movable`（四把椅子和踏脚凳）、推家具探针（36 局没摔，用身体推椅子 22 cm、踏脚凳 28 cm）、准备动作第四种 `floor` 和 `require_prep`（规则第 25 版第 34 条）、新任务族 `grasp_floor`。不做：33.1.6 越用越准（加任务记录 AUC 变差）。
- 33.9.5（沙盒 `ga-kp-20261024`）：空种子带原语，第 0 轮就 0 → 20 / 40，最后 23 / 40、封存 25 / 50（B0 31 / 50），2.9 h 就按停止线停了；续跑 `ga-kp-cont-20261025` 10-26 02:54 按墙钟停下，第 3–6 轮没接受新版本，封存仍 25 / 50（已确认）。

## 下一步（编号见 `TODO.md`）
- 正在跑（10-26 01:18 起，pgid 1905290，约 7.5 h，`outputs/scene_rsi_20261016/grasp_floor_a/queue.sh` / `queue.log`）：① 33.1.4 空地受控伸手探针（2,160 + 240 格）；② 33.9.5 续跑（02:54 结束，见上）；③ 33.24.4 `grasp_floor` 内环（沙盒 `grasp-floor-a`，02:54 开始，5 h，预计 08:30 前后连封存集结束）。跑完要告诉 `locomanipulation-ea`（它停着 Unreal 和批量任务等主线）。
- ① 的结果（10-26 01:25）：空地有能补的系统偏差（平均 2.74 cm、往前多伸 1.66 cm，闭环补完 3.07 → 1.57 cm）；补偿做成默认关的动作层开关 `RSI_REACH_COMPENSATION=1`（33.1.4 ②，已提交）。33.1.4.1 任务级对照已排在队列后面：等待进程 pgid 1939982（`outputs/scene_rsi_20261016/body/comp_ab_20261026/after_queue.sh`，队列一结束就跑 `run_ab.py run` 再 `report`，约 480 局 30 分钟；沙盒 `comp-ab-20261026` 从 `0b9a0705` 建，现任 `5cba2fa49d31`，120 个起点 = run D 重抽的诊断池 30 + 验收池 40 + 封存池 50，关 / 开 / 关 / 开交替；单局冒烟试过，开着时签名 `73adddd5a4`）。已告诉 `locomanipulation-ea`。之后看 ③ 循环写不写得出推椅子动作。
- TODO 38（10-26）：38.3 做完——GraspGen-X 抓取先验离线回放没过停止线（跨局 AUC 0.48，身体模型 0.77），不接主线；38.2 噪声底：补偿对照加了第三遍关着跑，B0 和 kp 的重跑沙盒已建好，`outputs/scene_rsi_20261016/noise_floor_20261026/launch.sh`（等待进程 pgid 2029117）在补偿对照之后两条并排跑，预计 15:00 前后连封存集结束。
- grasp_floor 的验收池原来只有 24 个起点（验收门要 ≥ 31），第 0 轮两次门调用没跑一局就被拒；04:00 在线补到 40、两次调用作废（`89489f7a`）。
- 33.21 / 33.22：A800 能连上后做脆弱起点和算力分工。33.7 / 37.6 用户定。

## 未解决 / 卡点
- 已确认：登记表基准和复现材料是 0.5 m/s + 休眠的记录（`sleep_20261021`），默认已是 0.25 m/s；按用户不重测，回归结论只作参考。
- 已确认：A800 和本机同一设置只有 73 / 80 局结局相同，远端的局不和本机配对。
- 已确认：本机重测 pick_single 总数相近、不逐局相同（用户：不查）。
- 已确认（run D）：设计探针、记录库判定、按层归因都被角色用上了；有没有帮上忙还看不出来，因为没有候选被接受。记录库判了 9 次，全是“已验证”，还没否定过一次。
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
