# 场景 RSI 在物流间 1030：人工挑选的单手 / 双手测试场景

[物流间 1030](rsi_logistic_room_1030.md)那条线的两个固定场景。物体不随机抽，而是由用户在网页上挑选；挑好之后固定不变，专门用来测 RSI。

| 场景 | 平台上的物体 | 任务族 | 目标 |
|---|---|---|---|
| `logistics_room_gso_single` | 20 件单手物体，没有纸箱 | `gso_single_grasp`，每件一个 `pick_<键>` | `held(X)` + `lifted(X, 0.20, 60)` |
| `logistics_room_gso_bimanual` | 7 件双手物体加纸箱（纸箱 0.3 kg） | `gso_bimanual_lift`，每件一个 `lift_<键>`，纸箱也有一个 | 同上，用双手 |

- **判定**：两个场景一样，都在每局最后 5 秒内每 0.1 s 检查一次，每次都要成立（`hold_s` 2 / `judge_s` 5）。单手也按这个判，比 `gso_grasp_s<seed>` 只在结束时看一次抬高 10 cm 更严。
- **起点技能**：主技能库（`sandbox.seed` null）。
- **每局的起点**：平台上所有物体都重新随机摆放（`shuffle_others`），目标物体中心离最近的台边不超过 0.35 m。机器人出生在能走到物体旁边的位置，离目标 0.5–4 m，朝向是正对目标再加一个 −80° 到 80° 之间均匀随机的偏角（任务族的 `"spawn": {"distance_m": [0.5, 4.0], "face_target_deg": [-80, 80]}`，用户 11-17）。同一个种子摆出来的起点总是一样，所以池里新旧技能比的是同一批起点。
- **方案文件**：`scripts/rsi/plans/gso_single_grasp.md` 和 `gso_bimanual_lift.md`，只写和 `gso_grasp.md` / `gso_bimanual.md` 不同的地方。

用户 2026-11-17 的决定：
- 旧的 `logistics_room_1030_s0..s9` 不删也不动。
- 两个新场景都建 MuJoCo 和 Unreal 版。
- 只做抓起和抱起，不做装箱；都抬高 20 cm、保持 5 秒（11-18 从 30 cm 降到 20 cm；第 9 节）。
- 只用训练划分里的物体，不要求放得进纸箱。
- 双手场景选 5–9 件，尽量多。
- 单手场景不放纸箱，所以能放 20 件。
- 我们做到冒烟为止；什么时候跑 RSI 内环由用户决定。
- 用户 11-18：内环有三种模式，单手单独跑、双手单独跑、两个任务族交替并共享知识和代码（第 7 节）。

决策记录在 TODO 40.22；代码在分支 `henry-dev`。

**现状（2026-11-17）**

两个场景都已建好：
- lint 都是 0 错误、0 警告，落地测试里所有物体都放稳了（20 / 20、8 / 8）。
- MuJoCo 和 Unreal 的预览图各两张，两个 Unreal 关卡已建。

起点检查（第 4 节）：
- 单手 20 个任务，每个都是 20 / 20。
- 双手 8 个任务，每个 6–13 / 20。
- 用户原本选了 8 件双手物体，结果 9 个任务里有 7 个一个起点都抽不到。按用户同意的“可以减少几个”，去掉了占地最大的布艺收纳方块（32 × 28 cm）。

可行性探针：单手场景有两件报 `beyond the hand's reach from the edge`，是塑料花盆 `gso_0233` 和木头字母积木车 `gso_0323`。这只是提示，不挡流程。

冒烟各一局（第 6 节），整条链都能跑通，目标都不成立：
- 单手：锤子 `pick_gso_0154` 在伸进去时被碰偏了 5 cm。马克杯 `pick_gso_0016` 抓住并抬离台面约 5 cm（当时目标是 30 cm，现在是 20 cm）。
- 双手：纸箱 `lift_nor_cal_box` 两手夹住后抬起约 14 cm（当时目标是 30 cm，现在是 20 cm，仍然不够）。

这是预期的：主库单手抓取只抬 0.10 m（`pick.LIFT_M`），双手抱起只抬 0.15 m（`bimanual.LIFT_DZ_M`），都不读目标里的高度；抬到 20 cm 要靠内环来学。

耗时是 2026-11-17 在 SRA0889（一张 RTX 5090）上的实测：

| 步骤 | 耗时 |
|---|---|
| 一个场景：建场景 + 沉降 + 落地测试 + lint + 任务族 + MuJoCo 预览图 | 9–10 s |
| 起点检查（20 个 seed × 全部任务） | 几秒 |
| 一个 Unreal 关卡加一张 Unreal 预览图 | 约 1 分钟（预览图 12–16 s） |
| 冒烟一局（仿真 28–60 s） | 4–7 s |

## 0. 每个新终端先执行

```bash
export LOCOMANI_WORKSPACE=$HOME/workspace   # 只在没有 /workspace 的机器上（如 SRA0889）
. scripts/paths.sh
export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
export RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
PY=$VENV_SIM_PY
```

和 [1030 runbook](rsi_logistic_room_1030.md) 第 0 节一样；前提也一样（物体网格已装好，`assets/scene/logistics_room_box_v1/` 已用 `git lfs pull` 取到）。

## 1. 先重建目录

```bash
$PY scripts/gso_catalog.py
```

做什么：按人工选择 `assets/mujoco_scanned_objects/hand_choices.json` 重建 `catalog.json`。人工选择在 1030 runbook 第 1 节的 `gso_gallery.py --serve` 页上点。挑选页和建场景都只读重建后的目录。

得到什么：`counts` 是 `single 283, bimanual 108`，`by_hand` 是排除 156、放回 1、改 shape 的单手 57 / 双手 49 / none 7。仓库里的 `catalog.json` 就是这一份。

## 2. 挑物体

```bash
$PY scripts/gso_gallery.py --pick --serve      # 打开 http://127.0.0.1:8765/pick.html
```

做什么：在图库网页旁边多写一个 `pick.html`，两个标签页：
- **单手**：正好 20 件，没有纸箱。
- **双手**：5–9 件，纸箱自动加上。

每个标签页只列目录里这一种类、训练划分的物体，卡片上有图、尺寸、质量、占地和“fits carton”标签，可以搜索、按类别筛、按占地或质量排序，点卡片就选上或取消。

每点一下：
- 立即存进 `assets/mujoco_scanned_objects/scene_picks.json`（进仓库），格式是 `{"scenes": {"single" | "bimanual": {"seed", "objects": [模型名]}}}`。
- 服务端按建场景的规则试摆一次，右侧画出平台俯视图：绿色是离台沿 0.35 m 的可达带，棕色是纸箱，物体画成包围盒并标序号。摆不下就写出是哪几件没位置。

页头的 seed 是摆放用的种子。图库页 `index.html` 照常可用。改了代码要重启服务（Ctrl-C 再起）。

试摆只看能不能放下；抱取要的两手通道和每局重摆其余物体，要等第 4 节的起点检查。

## 3. 建场景、出预览图

```bash
$PY scripts/gso_preview.py --picks single bimanual
```

做什么：对两个场景各走一遍 `prepare_logistics_scene.py <目录> --robot mars --rsi --picks single|bimanual`，然后沉降、落地测试、lint（必须 0 错误）、`gso_family.py` 写任务族，最后渲染一张 MuJoCo 预览图（左边俯视，右边房间视角，下方列出物体）。

摆放规则：
- 按文件里的 seed 摆放（`--place-seed N` 可以换），每件物体都在可达带上。
- 双手场景的纸箱在抱取带上：离台沿 0.18–0.31 m，长边沿台沿。单手场景把纸箱刚体删掉（场景文件 `gso.carton` 为 false）。
- 一种摆法放不下就依次试 `[seed, 1]`、`[seed, 2]` …，最多 20 次，用到第几次记在 `gso.place_attempt`。都不行就报出物体名退出，回第 2 节换掉物体。

已经存在的场景目录原样复用，要重建就先删掉它。`index.json` 会保留 1030 各实例的条目。

得到什么：
- 场景目录 `outputs/scene_discovery/logistics_room_gso_single/` 和 `logistics_room_gso_bimanual/`。
- 预览图 `outputs/gso_preview/logistics_room_gso_*.png`。
- 任务族文件 `scripts/rsi/families/gso_single_grasp.json` + `gso_single_grasp_tasks.jsonl`，`gso_bimanual_lift.json` + `gso_bimanual_lift_tasks.jsonl`。
- 两个场景都注册在 `configs/scenes.yaml`，所以 `bash scripts/build_scene.sh logistics_room_gso_single` 跑的也是同一条 build 命令。

## 4. 起点检查与可行性探针

```bash
PYTHONPATH=src $PY scripts/gso_start_check.py scripts/rsi/families/gso_bimanual_lift.json --out outputs/gso_preview/starts_gso_bimanual_lift.json
PYTHONPATH=src $PY scripts/gso_start_check.py scripts/rsi/families/gso_single_grasp.json --out outputs/gso_preview/starts_gso_single_grasp.json
$PY scripts/scene_feasibility.py --scene outputs/scene_discovery/logistics_room_gso_single/scene_meta.json --tasks scripts/rsi/families/gso_single_grasp_tasks.jsonl --out outputs/gso_preview/feasibility_single.json
```

做什么：
- **起点检查**：每个任务用 20 个 seed 按池的方式抽起点，条件是目标在可达带上、被抱的物体两侧和前面各留 0.18 m、其余物体重新随机摆放。每个任务打印抽到几个起点；有任务一个都抽不到，就退出码 1 并写出原因。只用 CPU。
- **可行性探针**：和 1030 runbook 第 4 节一样，只作参考，不挡流程。

一个起点都抽不到的任务，回第 2 节去掉占地大的物体，再删掉场景目录重建。11-17 的双手场景就是这样从 8 件减到 7 件的。

得到什么：终端最后一行 `{"family": ..., "stuck": []}`，以及 `--out` 的 JSON。

## 5. 检查代码

```bash
make test TESTS="tests/test_gso_scene_pick.py tests/test_gso_family.py tests/test_gso_catalog.py tests/test_gso_gallery.py tests/test_gso_preview.py tests/test_rsi_knowledge.py tests/test_rsi_alternate.py"; echo "exit $?"
```

做什么：
- 挑选文件的校验：种类、训练划分、数量是 20 / 5–9、不能重复。
- 试摆：固定 seed 结果一样；放不下时写出物体名；单手场景不放纸箱也能摆下 20 件。
- 挑选页的数据和保存。
- 任务族：默认抬 0.20 m、`--lift` 改高度、最后 5 s 判、主库起点、方案文件在沙盒的保留列表里。
- 知识库：共享文件的读写、按来源任务族替换、`tag`、`KNOWLEDGE.md` 的全文和标注、重建时沿用未变的机制。
- 交替驱动：三方合并和冲突、参数覆盖值写进常数、轮流和续跑、一方停下、异常阻塞、同步被拒；`gate.py` 的不变差规则；`loop.py sync` 的账本行。
- 目录、图库和预览的原有测试。

约 1 分钟。

得到什么：退出码 0。

## 6. 冒烟一局（不是 RSI）

```bash
$PY scripts/gso_pack_check.py --family gso_single_grasp --task pick_gso_0016
$PY scripts/gso_pack_check.py --family gso_bimanual_lift      # 第一个任务：纸箱
```

做什么：和 1030 runbook 第 5a 节一样：按任务族建世界，按池的方式抽一个起点（抽不到的 seed 跳过，最多 `--tries` 个），跑 `src/` 里主库 `plan_for(task)` 的计划，按最后 5 s 判定。不进沙盒、不建池、不过 gate。

得到什么：`outputs/gso_pack_check/<族>/<任务>_s<seed>.json` 和 `.npz`。11-17 的结果见本页开头：链能跑通，抬高不够。

## 7. 跑内环：三种模式

用户 11-18 定的三种模式：

| 模式 | 做什么 | 知识 | 代码 |
|---|---|---|---|
| 1 | 单手 `gso_single_grasp` 单独跑 | 开跑前不带任何知识；跑完建库 | 各自的沙盒 |
| 2 | 双手 `gso_bimanual_lift` 单独跑 | 同上 | 同上 |
| 3 | 两个任务族交替，一个迭代 = 单手一轮加双手一轮 | 一个共享文件，两边都写、都整个读 | 一个技能包，每半轮合并给对方 |

知识库（规则 39）的条目都在一个文件 `knowledge.jsonl` 里，每条带 `tag`：`single`（单手抓起）或 `bimanual`（双手抱起）。内环每一轮在 Reviewer 之前把整个库写成 `KNOWLEDGE.md`，每条下面标出它的条件在现任的哪些诊断局上成立（只标注，不过滤），Reviewer 和 Engineer 整个读。技能代码不能读知识库。

沙盒只导出已提交的代码，开跑前先确认要用的东西都已提交：

```bash
git status --short scripts/rsi/   # 应为空
```

**7.1 模式 1：单手单独跑**

```bash
S1=$LOCOMANI_WORKSPACE/code/rsi-scene/gso-single-a
python3 scripts/rsi/sandbox.py $S1 gso_single_grasp --fresh
cd $S1 && CUDA_VISIBLE_DEVICES=0 nohup setsid $PY scripts/rsi/loop.py run gso_single_grasp > loop.log 2>&1 &
cd -
cd $S1 && $PY scripts/rsi/loop.py status gso_single_grasp; cd -      # 看进度
```

做什么：和 1030 runbook 第 5 节一样：建沙盒、抽池，然后在后台跑内环，直到满足终止条件。`--fresh` 让沙盒既不带知识库也不带经验库，这次运行不继承任何以前的知识。
- **池**：按任务数分配（用户 11-18），默认每个任务诊断 3 局、验收 4 局、封存 5 局：单手 20 个任务是 60 / 80 / 100 局，双手 8 个任务是 24 / 32 / 40 局。验收池至少比验收门第一阶段（`gate.py` 的 `STAGE1`，30 局）多 1 局。局数记在任务族文件的 `pool_per_task` 和各池的 `n` 里，建沙盒、内环和封存集都从这里读；要改就用 `PYTHONPATH=src $PY scripts/rsi/gso_family.py outputs/scene_discovery/logistics_room_gso_single --per-task diag=3 accept=4 sealed=5` 重写任务族（可以只给其中几项），提交后再建沙盒。池种子：单手 9000000 / 9100000 / 9200000，双手 9500000 / 9600000 / 9700000。
- **终止条件**：验收池上赢到 80 %、跑满 5 小时、做满 5 轮或验收门调用满 12 次，先满足哪条就停。只想先试一轮，加 `--max-iters 1`。
- 场景目录必须在本机 `outputs/scene_discovery/` 下（第 3 节）。规则第 30 条照旧：技能代码里不能出现物体名字。

得到什么：账本 `$S1/outputs/scene_rsi_20261016/gso_single_grasp/loop/ledger.jsonl`，最后一行 `stop` 写着停下的原因。

跑完记录知识，再跑封存集：

```bash
PYTHONPATH=src $PY scripts/rsi/knowledge.py build $S1 gso_single_grasp
git -C scripts/rsi add knowledge && git commit -m "knowledge: gso_single_grasp entries"
cd $S1 && $PY scripts/rsi/sealed.py gso_single_grasp; cd -
```

做什么：
- `build` 从账本和场景图记录（`.past.jsonl`）里抽出机制，每个机制用一次模型调用写条件、失败特征和修法，再按每件物体上的证据分成 `shared` 或 `local`。条目写进主仓库 `scripts/rsi/` 下 `knowledge/knowledge.jsonl`，替换这个任务族以前的条目，别的任务族的条目保留；`tag` 按任务的物体自动判成 `single`。
- 同一个机制的证据没变时，重建直接沿用上次的结果，不再调用模型。约每个机制 1 USD，共 10–15 分钟。建库日志在 `knowledge/build/gso_single_grasp.md`。
- 第二条把条目提交，以后的沙盒才能导出它们。
- 第三条在封存集（单手 100 局）上跑种子和每一版被接受的现任（同 rsi.md 第 5.3 节），约 1–2 小时。

**7.2 模式 2：双手单独跑**

把 7.1 里的 `gso_single_grasp` 都换成 `gso_bimanual_lift`，沙盒目录换成 `gso-bimanual-a`。`tag` 自动判成 `bimanual`。

**7.3 模式 3：单手和双手交替**

```bash
R=$LOCOMANI_WORKSPACE/code/rsi-scene/gso-alt-a
python3 scripts/rsi/alternate.py init $R gso_single_grasp gso_bimanual_lift
CUDA_VISIBLE_DEVICES=0 nohup setsid $PY scripts/rsi/alternate.py run $R --iters 5 --max-hours 10 > $R/run.log 2>&1 &
$PY scripts/rsi/alternate.py status $R                                  # 看进度
```

做什么：
- `init` 建两个 `--fresh` 沙盒 `$R/a`（单手）和 `$R/b`（双手）、一个空的共享知识目录 `$R/knowledge`，以及合并用的共同祖先 `$R/sync/base`。两个沙盒的任务族文件里写上 `shares_skills_with`，Reviewer 和 Engineer 的提示会说明技能包和另一个任务族共用：`pick.py` 归单手，`bimanual.py` 归双手，共用的模块改了会影响对方。
- `run` 先单手、后双手，轮流各做半轮，两个半轮算一个迭代。每个半轮：
  1. 把共享知识复制进这一方的沙盒并提交，跑一轮内环（`loop.py run --max-iters 1`）。
  2. 用这一方到目前为止的整本账本重建它的条目。
  3. 把这一方的现任技能包（参数覆盖值写进常数）三方合并进另一方的现任。两边改了同一处时保留另一方的版本，记为冲突。合并结果在另一方的验收池上过“不变差”验收门（`loop.py sync`，`gate.py --non-inferior`）：少赢不超过多赢就成为另一方的新现任，否则作废。这次验收在另一方的账本里记一行 `gate`，计入它的验收门调用次数；结果也写进共享的 `sync.jsonl`，两边的 `KNOWLEDGE.md` 末尾都能看到。
- 终止条件（用户 11-18），满足任一条就停：
  1. 跑完 `--iters` 个迭代（5：单手 5 轮、双手 5 轮）。
  2. 整个运行跑满 `--max-hours` 小时（默认 10），算两边合计的时间：驱动实际在跑的时间累加，驱动停着的时间不算。每个阶段开始前检查，正在跑的阶段会跑完。
  3. 两边都在各自的验收池上赢到 `--stop-rate`（默认 0.8：单手 80 局赢 64 局，双手 32 局赢 26 局）。只有一方赢到时，这一方停下：跳过它的半轮，也不再往它合并代码，它的条目继续给对方用；另一方接着跑。
- 两边的内环都带 `--alternating` 运行：每个任务族只按停止线停下，不再因为它自己的 5 小时、5 轮、12 次验收门或 Manager 决定停而停；Manager 决定停时只结束这一轮。
- 一方的内环出错（角色调用失败、审计连续两轮作废、崩溃）或同步验收出错，不算跑完，整个运行停下，`run.log` 最后一行写着原因；查完再跑同一条 `run` 命令，从停下的那一步接着跑。建库得到 0 条或失败只记录，不停。
- 每个阶段做完都存一次状态（`$R/alternate.json`），中途停了再跑 `run` 会接着跑。同一个运行目录同时只能有一个驱动。

得到什么：`$R/knowledge/knowledge.jsonl`（两边的条目）、`sync.jsonl`（每次合并的文件、冲突、多赢 / 少赢、接受与否）、`$R/logs/`（每个半轮的内环和同步验收日志）。一个迭代约 2 小时：两轮内环各约 45 分钟，两次建库各 10–15 分钟，两次同步验收各 10–20 分钟。

跑完把知识发布到主仓库，再分别跑两边的封存集：

```bash
python3 scripts/rsi/alternate.py publish $R
git -C scripts/rsi add knowledge && git commit -m "knowledge: gso single / bimanual alternating run"
cd $R/a && $PY scripts/rsi/sealed.py gso_single_grasp; cd -
cd $R/b && $PY scripts/rsi/sealed.py gso_bimanual_lift; cd -
```

做什么：`publish` 把运行里的条目合进主仓库的 `knowledge.jsonl`，替换这两个任务族以前的条目，别的保留；提交由你来做。封存集里同步过来的版本也算一个现任版本。两边的最终代码要不要写回主技能库（`writeback.py`），由你决定。

## 8. Unreal 关卡与预览图

```bash
systemd-run --user --scope -p MemoryMax=24G bash scripts/start_unreal.sh --gpu 0   # 编辑器没开着时
PYTHONPATH=src "$BRIDGE_PY" scripts/gso_preview_unreal.py --picks single bimanual --build
bash scripts/stop_unreal.sh
```

做什么：
- `--build` 会先给缺关卡或关卡过期（场景 XML 的哈希对不上）的场景运行 `bash scripts/build_scene.sh <场景> --unreal`，关卡在 `/Game/Levels/logistics_room_gso_single` 和 `/Game/Levels/logistics_room_gso_bimanual`。
- 然后从和 MuJoCo 预览图同样的两个视角各拍一帧。
- 重建场景后再跑同一条命令，关卡会自动重建。

封存集和视频（MuJoCo 版和 Unreal 版）同 1030 runbook 第 6 节，命令里的族名换成上面两个。

得到什么：`outputs/gso_preview/logistics_room_gso_*_unreal.png`（2560 宽，高度随物体列表变），以及合并进去的 `index_unreal.json`。

## 9. 改物体、改判据

- **换物体**：回第 2 节重选，删掉场景目录，再跑第 3、4、8 节。只换摆法就用 `prepare_logistics_scene.py ... --place-seed N`，或者改页头的 seed。
- **抬升高度**：默认 0.20 m（`gso_family.py` 的 `PICKED_LIFT_M`；用户 11-18 从 0.30 m 降下来）。要换高度就运行 `PYTHONPATH=src $PY scripts/rsi/gso_family.py outputs/scene_discovery/logistics_room_gso_single --lift 0.25` 重写任务族（双手同理），场景不用改；再把两份方案里的高度改掉。改完要提交，并在方案和账本里记一次人工干预。`--lift` 只对这两个人工挑选的场景有效。
- **数量规则**：`src/locomani/sim/gso_objects.py` 的 `PICK_SCENES`（单手 20 件、无纸箱；双手 5–9 件加纸箱）。
- **两手留空**：`scripts/rsi/gso_family.py` 的 `HANDS_CLEAR_M`（0.18 m）。

## 10. 常见错误

- `prepare_logistics_scene.py` 报 `no placement of the N objects in 20 tries; no spot left for ...`：放不下，按报出的物体回第 2 节换掉，或者换 `--place-seed`。
- 报 `objects picked, want 20`（或 `want 5-9`）：挑选文件的件数不对，回第 2 节补齐。
- 报 `not single training objects of the catalog`：目录重建后，这件物体的种类变了或被排除了，回第 2 节换掉。
- `gso_start_check.py` 报 `no spot on A for <键> clear of the task objects (shuffle_others)`：太挤，见第 4 节。
- 挑选页数量上限不对，或保存时报 400：服务还是改代码前起的，重启 `--pick --serve`。
- 其余（GL、Unreal 编辑器、物理副本的路径）同 1030 runbook 第 10 节。
