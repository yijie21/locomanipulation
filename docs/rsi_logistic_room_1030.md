# 场景 RSI 在物流间 1030：扫描物体的单手 / 双手移动抓取

[物流间](rsi_logistics_room.md)那条线的场景，加上 Google 的 Scanned Objects 数据集（1030 件扫描的日用品，kevinzakka 的 MuJoCo 版）：每个场景实例 `logistics_room_1030_s<seed>` 从目录里抽 10 件（默认 6 件单手、4 件双手）和原来的 1.2 kg 纸箱一起随机放在中央 1.2 × 0.9 m 托盘平台的可达带上，机器人（mars 版 G1，Dex1 两指夹爪）走过去用一只手或两只手把目标拿起来。两个任务族由场景文件自动生成：`gso_grasp_s<seed>`（单手，每件单手物体一个任务，`held(X)` 加 `lifted(X, 0.10, 60)`）和 `gso_bimanual_s<seed>`（双手，每件双手物体和纸箱各一个任务，`held(X)` 加 `lifted(X, 0.20, 60)`，最后 5 秒一直成立）。11-05 加了装箱线（用户：把 10 件物体逐件放进纸箱再把纸箱拿起来）：`gso_pack_s<seed>`（单手物体，每件一个任务）和 `gso_pack_bimanual_s<seed>`（双手物体，每件一个任务），目标 `in(X, nor_cal_box)` + `held(nor_cal_box)` + `lifted(nor_cal_box, 0.20, 60)`，每局独立（纸箱是空的，只放一件，方案 `scripts/rsi/plans/gso_pack.md`）；为此实例改成 10 个（s0–s9），10 件都放得进纸箱（内空 0.43 × 0.30 m、沿高 0.33 m，留 2 cm，目录字段 `fits_carton`），纸箱固定放在离台沿 0.18–0.31 m、长边沿台沿的抱取带上。流程机制见[场景 RSI 详解](rsi_design.md)；2026-11-03 的决策（Q1–Q25）、11-04 的鞋类排除和 11-05 的装箱线在 [DECISIONS](../DECISIONS.md)，待办在 TODO 40。代码在分支 `henry-dev`（提交 `5ba83e16`，2026-11-04）。

现状（2026-11-05）：代码、测试和 10 个实例做完，装箱线单手 / 双手各冒烟一局跑通（第 5a 节），RSI 还没跑（用户：先只实现代码）。目录 1030 件按数据集自带的类别标签（`categories.json`）只用剩下的类别：Shoe、Toys、Stuffed Toys、Mouse、Media Cases、Legos、Keyboard、Car Seat、Action Figures、Hat 十类共 463 件按用户要求不进物流间（11-04 先排除鞋，11-05 扩到十类），用的是 Consumer Goods、Bottles and Cans and Cups、Bag、Board Games、Headphones、Camera 和 216 件没有标签的；可抽的单手 270、双手 90（其中放得进纸箱的 260 / 59，实例只抽这些），其余 207 件形状不合格，留出评测 96 件；实例 s0–s9 都过 `scene_lint.py`（0 错误；s3 一条静置漂移警告），十个 Unreal 关卡都建好，MuJoCo 和 Unreal 各有十张预览图（`outputs/gso_preview/`，第 2、7 节）。渲染器：MuJoCo 和 Unreal；Isaac 暂缓（`configs/scenes.yaml` 里这些实例没有 `isaac` 条目，TODO 40.6）。

耗时是 2026-11-04 / 05 在 SRA0889（一张 RTX 5090、16 核）上的实测。

| 步骤 | 耗时 |
|---|---|
| 装物体库（2 GB，一次） | 看网速 |
| 取类别标签（Gazebo Fuel，一次） | 几秒 |
| 建目录（1030 件，含每件 24 s 的静置测试） | 8 s |
| 物体图库网页（1030 张图） | 54 s |
| 一个实例：建场景 + 沉降 + 落地测试 + lint + 任务族 + 预览图 | 10–15 s |
| 一个实例的 Unreal 关卡 | 39–97 s |
| 一个实例的 Unreal 预览图（两个视角） | 15–31 s |
| 装箱线冒烟一局（`gso_pack_check.py`，载入 1 s，仿真 25–66 s） | 4–8 s |
| 内环一轮 / 封存集 | 同[物流间 runbook](rsi_logistics_room.md)的表 |

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
export RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
PY=$VENV_SIM_PY
```

做什么：和物流间 runbook 第 0 节一样，多一行 `MUJOCO_GL`：预览图和 pytest 里的渲染测试要离屏 GL。前提也一样（[安装](setup.md)“场景 RSI”一列、`git lfs pull` 取到场景包 `assets/scene/logistics_room_box_v1/`、`bash scripts/setup_sonic.sh`）。

## 1. 装物体库、建目录（每台机器一次）

```bash
bash scripts/install_gso.sh            # 或 bash scripts/install_gso.sh --from <已经下载的 mujoco_scanned_objects 目录>
python3 scripts/gso_categories.py      # 只在要刷新类别标签时；categories.json 已在仓库里
$PY scripts/gso_catalog.py
```

做什么：第一条把 1030 件物体的网格装到 `assets/mujoco_scanned_objects/models/`（git 忽略；仓库只跟踪 `catalog.json`、`categories.json`、LICENSE、README、VERSION），已经装好就什么都不做。第二条从数据集在 Gazebo Fuel 上的条目取每件物体的官方类别（Shoe 254、Consumer Goods 248、Toys 150、Bottles and Cans and Cups 53、Bag 28……216 件没有标签；kevinzakka 的 MuJoCo 版没带这些标签）写成 `categories.json`。第三条按每件物体的网格算尺寸、V-HACD 壳体体积、质量（体积 × 300 kg/m³，截到 0.05–1.5 kg）、预计抓取高度、24 s 落地静置测试，判成单手 / 双手 / 不能抓，并按名字哈希留出 10% 作评测集；只有 `--use-categories` 里的类别能被抽到，其余类别一律记 `none`，`why` 写 `category Shoe: not among the used categories`。默认是去掉 Shoe、Toys、Stuffed Toys、Mouse、Media Cases、Legos、Keyboard、Car Seat、Action Figures、Hat 之后剩下的类别加上没有标签的 `(none)`（用户 2026-11-05，`gso_objects.USED_CATEGORIES`）；`--use-categories Bag "Board Games"` 只用这几类，`--use-categories all` 全用，`--exclude-categories Shoe Toys` 用除这些以外的全部；类别名按 `categories.json` 的 `counts` 写，不分大小写，写错名字直接报错；阈值都是参数（`--jaw 0.095`、`--bimanual-width 0.20 0.45`、`--bimanual-min-height 0.10`、`--density 300`、`--held-out 0.10` 等，见 `--help`），写进文件。`catalog.json` 已在仓库里，和模型一样就不用重建；重建得到同一个文件。
得到什么：终端打印 `counts`（`single 270, bimanual 90, none 670, held_out 96, excluded 463`）和 `not_used_categories`、阈值和秒数；字段说明在 `assets/mujoco_scanned_objects/README.md`。

看全部物体长什么样（用户 2026-11-05）：

```bash
$PY scripts/gso_gallery.py               # --force 全部重画；--size 384；--workers 8
```

做什么：用 MuJoCo 离屏把目录里每件物体单独画一张（扫描网格和贴图、浅色地面、同一个斜上方视角、按包围盒取景），再写一个本地网页：卡片上是图、短语、键、质量、尺寸和单手 / 双手 / none、类别、是否排除的标签；可以按名字搜索，按种类、类别、划分、是否排除筛选，按质量或最长边排序；点卡片看大图和目录里的全部字段（含 `why`）。已画的图默认保留，网页每次重写。
得到什么：`outputs/gso_gallery/index.html`（浏览器直接打开）和 `img/<键>.jpg` 1030 张，共约 18 MB，约 1 分钟。

## 2. 建场景实例、出预览图

```bash
$PY scripts/gso_preview.py --seeds 10
```

做什么：对 seed 0–9 各建一个实例 `outputs/scene_discovery/logistics_room_1030_s<seed>/`：抽 10 件放得进纸箱的（目录的 `fits_carton`；一组在平台上放不下就按确定性规则重抽，`scene_meta.json` 的 `gso.draw_attempt` 记第几组；11-05 的目录下 seed 3 用的是第 1 组）、放到平台离台沿 0.35 m 的可达带上，纸箱放在离台沿 0.18–0.31 m、长边沿台沿（yaw 85–90°）的抱取带上（`prepare_logistics_scene.CARTON_PLACEMENT`，记在 `gso.carton_placement`）、壳体密度惯量、俯视相机 `platform_top`，再沉降（`scene_settle.py --write`）、落地测试（`scene_drop_test.py`，可行性探针读它）、lint（必须 0 错误，否则停）、写四个任务族（第 5 节），最后用 MuJoCo 离屏渲染俯视图和房间视角拼成一张图，图下列出每件物体的键、短语、单手 / 双手、质量、在不在可达带。已经存在的实例原样复用（要重抽就删掉它的目录）。
得到什么：`outputs/gso_preview/logistics_room_1030_s<seed>.png` 十张和 `index.json`；`scripts/rsi/families/` 下每个实例五个文件：`gso_grasp_s<seed>.json`、`gso_bimanual_s<seed>.json`、`gso_pack_s<seed>.json`、`gso_pack_bimanual_s<seed>.json`、`gso_s<seed>_tasks.jsonl`（s0–s9 的在仓库里）。十个实例约 1.5 分钟。

| 选项 | 作用 |
|---|---|
| `--seeds N` / `--first-seed K` | 实例 sK … sK+N-1 |
| `--objects 10 --bimanual 4` | 每个实例几件、其中几件双手（纸箱另算） |
| `--pool train` | `train`：训练划分；`eval_unseen`：留出的 96 件（评测实例，seed 从 100 起，别和训练实例重号）；`all` |
| `--scenes DIR` / `--out DIR` | 实例目录和图的目录（pytest 用临时目录） |
| `--families-dir DIR` / `--no-families` | 任务族写到哪、或不写 |

一个实例手动走一遍（预览脚本做的就是这几步）：

```bash
SC=outputs/scene_discovery/logistics_room_1030_s0
$PY scripts/prepare_logistics_scene.py $SC --robot mars --rsi --gso-seed 0
$PY scripts/scene_settle.py --scene $SC --write
$PY scripts/scene_drop_test.py --scene $SC/scene_meta.json
$PY scripts/scene_lint.py --scene $SC
PYTHONPATH=src $PY scripts/rsi/gso_family.py $SC
```

做什么：第二条是 `configs/scenes.yaml` 里 `logistics_room_1030_s0` 的 `build` 命令（`bash scripts/build_scene.sh logistics_room_1030_s0` 跑的就是它），参数 `--objects`、`--bimanual`、`--pool`、`--reach-depth`、`--catalog`、`--gso-models` 同上。最后一条从 `scene_meta.json` 写任务族：单手物体一行 `pick_<键>`、双手物体和纸箱一行 `lift_<键>`，每行 `placement.depth_m` 为 `[0, 0.35]`（目标起点限制在可达带，D13），提示语由物体短语生成；装箱族每件物体一行 `pack_<键>`（目标 `in(X, nor_cal_box)` + `held` + `lifted` 纸箱），`placement.objects.nor_cal_box` 是纸箱的抱取带，双手物体自己再限 yaw 让宽边沿台沿。
得到什么：`$SC/` 下 `scene_locomani.xml`、`env_only.xml`、`scene_meta.json`（每件物体的 `box`、`semantic`、`gso` 块和落地测试）、`gso/<键>/` 网格拷贝、`settle.json`、`drop_test.json`、`lint.json`；lint 打印 `[lint] 0 errors, 0 warnings`。

## 3. 检查代码

```bash
make test TESTS="tests/test_gso_catalog.py tests/test_gso_family.py tests/test_gso_gallery.py tests/test_unreal_export_film.py tests/test_run_context.py tests/test_image_source.py tests/test_scene_task_runner.py tests/test_scene_discovery.py src/locomani/tasks/scene/skills/test_pack.py src/locomani/tasks/scene/skills/test_plan_for.py src/locomani/tasks/scene/skills/test_into.py"; echo "exit $?"
make test TESTS="tests/test_gso_preview.py"; echo "exit $?"
```

做什么：第一条是纯函数和注册表的测试（短语、筛选规则、放得进纸箱的规则、类别选择与仓库里的目录只抽得到用的类别、留出划分、抽样、带约束的放置、四个任务族文件、Unreal 导出烤惯量、十个实例在注册表里、按物体的放置选项、`in` 用转过的盒中心、装箱链的分派和顺序、物理副本的绝对资源路径），约 30 s。第二条在临时目录里建一个实例并渲染一张图（第 2 节的全套），约 1 分钟；没有场景包、物体网格或离屏 GL 时跳过（跳过原因打印在 `-rs` 里）。
得到什么：退出码 0。

## 4. 可行性探针（提示，不挡）

```bash
$PY scripts/scene_feasibility.py --scene $SC/scene_meta.json --tasks scripts/rsi/families/gso_s0_tasks.jsonl --out outputs/gso_preview/feasibility_s0.json
```

做什么：按单手能力表给每个任务分桶。放置的几何规则（可达带）是硬约束，探针只作参考（Q20）：s0 的 11 个任务里 1 个报 `beyond the hand's reach from the edge`；双手任务的 `grasp_bimanual.*` 缺口要等第一次运行。报 `does not rest still in the sim` 说明没跑第 2 节的落地测试。

## 5. 跑内环

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/gso-grasp-a
python3 scripts/rsi/sandbox.py $S gso_grasp_s0
cd $S && CUDA_VISIBLE_DEVICES=0 nohup setsid $PY scripts/rsi/loop.py run gso_grasp_s0 --max-iters 1 > loop.log 2>&1 &
cd -
```

做什么：和物流间 runbook 第 4 节一样，任务族换成 `gso_grasp_s0`（单手，6 个任务，种子是空技能包 `scripts/rsi/seeds/atomic`，角色读的方案是 `scripts/rsi/plans/gso_grasp.md`）。双手族把三处 `gso_grasp_s0` 换成 `gso_bimanual_s0`（5 个任务：4 件双手物体加纸箱，种子为空即主技能库 —— 两手抱取 `skills/bimanual.py` 11-05 从 `seeds/box_strict_start` 移进了主库；方案 `scripts/rsi/plans/gso_bimanual.md`，`hold_s` 2 / `judge_s` 5）；装箱族换成 `gso_pack_s0`（6 件单手物体各一个任务）或 `gso_pack_bimanual_s0`（4 件双手物体），种子同样为空（主库 `skills/pack.py`：单手或双手放入，再双手抱起纸箱），方案 `scripts/rsi/plans/gso_pack.md`，`hold_s` 2 / `judge_s` 5；要跑哪个场景就换族名里的 seed（s0–s9）。沙盒目录每个族另起一个。沙盒导出的是**已提交**的代码：s0–s4 的任务族文件在提交 `5ba83e16` 里，场景实例目录要在本机建好（第 2 节）；改了任务族或场景就要再提交 / 重建。规则第 30 条仍然有效：技能代码里出现物体名字（`gso_0051`、短语或刚体名）的草稿作废，技能只能读 `box.half_extent`、`semantic`、`gso.grasp_height_m` 这些字段。
一个场景的全部任务一起进化（用户 11-06）：族名用 `gso_all_s0`（三处一起换）。它包含这个场景四个族的全部任务（s0–s9 都是 21 个：6 个单手抓起、5 个双手抱起、6 个单手装箱、4 个双手装箱），一轮迭代里所有技能一起诊断、修改，再在同一个 gate 上比较。每个任务的判据和它原来所在的族一样：单手抓起在计划结束、静止 2 秒后判一次，其余任务要在最后 5 秒内每次检查都成立（任务文件每行的 `hold_s` / `judge_s`）。起点技能是主库现有的；起点池按每个任务诊断 2 局、验收 2 局、封存 5 局，也就是 42 / 42 / 105 局，种子 480000 / 580000 / 680000；角色读的方案是 `scripts/rsi/plans/gso_all.md`。

什么时候停（用户 11-06）：这五种族的任务族文件写着 `stop_rate` 0.8、`max_wall_h` 5、`max_run_iters` 5，三条先满足哪条就停：现任技能在验收池上赢到 80 %、这次运行跑满 5 小时（停下到续跑之间的时间不算）、这次运行做满 5 轮迭代（续跑接着数）。每轮开始前检查一次，所以正在跑的那一轮会跑完，跑满 5 小时的时候最多再多出一轮的时间（约 1 小时）。`--max-iters` 只限制这一次命令，族文件里的 5 轮是整个运行的上限；命令行的 `--stop-rate` 和环境变量 `RSI_MAX_WALL_H` 仍然优先。停下的原因写在账本最后一行 `stop` 的 `reason` 里。
得到什么：账本 `$S/outputs/scene_rsi_20261016/gso_grasp_s0/loop/ledger.jsonl`；进度 `cd $S && $PY scripts/rsi/loop.py status gso_grasp_s0; cd -`。起点池种子：单手 440000 / 540000 / 640000，双手 450000 / 550000 / 650000，装箱 460000 / 560000 / 660000，双手装箱 470000 / 570000 / 670000，和其他族不重叠。

## 5a. 装箱线冒烟（一局，不是 RSI）

```bash
$PY scripts/gso_pack_check.py --family gso_pack_s0             # --task pack_gso_0558 --seed 0 --tries 10
$PY scripts/gso_pack_check.py --family gso_pack_bimanual_s0
```

做什么：不进沙盒、不建池、不过门：读任务族，按 RSI 一局的方式建世界（`scene_round.make_world`，ScaleBFM）、按族的 spawn 和任务行的 placement 采一个起点（纸箱带放不下的 seed 跳过，最多试 `--tries` 个），跑 `src/` 里主库 `plan_for(task)` 的计划（`episode.run_episode`，族的 `hold_s` / `judge_s`），逐步打印，写 `outputs/gso_pack_check/<族>/<任务>_s<seed>.json`（判定、步骤、判窗）和 `.npz`（轨迹）。`LOCOMANI_WORKSPACE` 要指向有 `models/scalebfm/humanoid_transformer_m_22200.pt` 的目录（第 0 节的 `paths.sh`；本机是 `$HOME/workspace`）。
得到什么（2026-11-05，s0，主库 `6674fb78`）：两条链都从头跑到尾、没有崩溃，都停在放入那一步的抓取上：
- `pack_gso_0558`（单手，墨盒 8.6 × 5 × 11.8 cm）：seed 0–2 的起点纸箱带放不下、跳过；seed 3 走到台沿（0.27 m 深），单手 pregrasp 偏 19 cm，重试偏 4.8 cm，`pick failed: pregrasp missed by 4.6 cm`；仿真 66 s、墙钟 7.7 s，判定 `in` 不成立。
- `pack_gso_0945`（双手，午餐袋 22 × 16 × 27 cm）：seed 0–1 跳过；seed 2 走 0.84 m 到抱取站位、下蹲，两手 pregrasp 偏 16 / 15 cm（IK 误差 < 1 cm，是手臂没跟到位），`hug pick failed`；仿真 25 s、墙钟 3.9 s。
这就是装箱线 RSI 的起点：计划分派、起点采样（含纸箱带）、单手 / 双手放入、抱纸箱、再判目标这条链都接通，抓取精度留给内环学（TODO 40.15）。冒烟修的一个真 bug：`world.physics_scene` 写物理副本时不改相对的 `file=`，扫描网格 `gso/<键>/…` 找不到（其他场景全是绝对路径）；现在副本里的相对资源路径都指回场景目录。

## 6. 封存集与视频

```bash
cd $S && $PY scripts/rsi/sealed.py gso_grasp_s0; cd -
cat $S/outputs/scene_rsi_20261016_hidden/gso_grasp_s0/sealed/summary.json
$PY scripts/rsi/report/evolution_media.py --sandbox $S --family gso_grasp_s0 --out outputs/scene_rsi_20261016/gso_grasp_s0_media --only task --versions final
```

做什么：同物流间 runbook 第 5、6 节。按任务的成功数就是“哪些物体学会了”；双手族的 90 件“untested”物体哪些抱得起，看它的封存集。

## 7. Unreal 关卡（每个实例一个）

```bash
bash scripts/start_unreal.sh --gpu 0      # 编辑器没开着时
bash scripts/build_scene.sh logistics_room_1030_s0 --unreal
PYTHONPATH=src "$BRIDGE_PY" scripts/gso_preview_unreal.py --seeds 10 --build
```

做什么：按 `configs/scenes.yaml` 里 `logistics_room_1030_s0` 的 `unreal.build` 建关卡：把场景的会动部分（机器人、纸箱、10 件物体和它们的扫描纹理）导出成 URLab 的 MJCF 导入进 `/Game/Levels/logistics_room_1030_s0`（房间是 warehouse 的关卡）。导出器剥掉碰撞壳之前先把每件物体的质量和惯量烤成显式 `<inertial>`，不然扫描物体没有质量、编译失败。s1–s9 换名字各建一次（每个实例一份关卡，10 件物体各不相同）；重建场景实例后关卡要重建（`source.json` 记着场景 XML 的哈希，不一致 `run_context.unreal_scene` 会拒绝）。
第三条给每个实例出一张 Unreal 图（用户 11-04）：把关卡里的第三视角机位（1280 × 960、fovy 45）依次放到平台正上方 2 m 和房间远角（和 MuJoCo 预览图同样的两个视角），各拍一帧静止场景（沉降后的物体、机器人在起点），图下列出物体；`--build` 让它先给缺关卡或关卡过期（场景哈希不符）的实例建关卡，不加就报错停下。
得到什么：`$UNREAL_ASSETS/logistics_room_1030_s<seed>/`（`mjcf/`、`source.json`），每个 39–97 s（s0–s9 的 11-05 都建了）；`outputs/gso_preview/logistics_room_1030_s<seed>_unreal.png` 十张（2560 × 1184：左俯视、右房间视角、下方物体列表）和 `index_unreal.json`，每张 15–31 s（主要是载关卡、起 PIE）。之后 Unreal 视频按物流间 runbook 第 7 节用 `scripts/rsi/report/report_unreal.py`，场景名从沙盒的 `media.json` 来。

## 8. 评测没见过的物体

```bash
SC=outputs/scene_discovery/logistics_room_1030_s100
$PY scripts/prepare_logistics_scene.py $SC --robot mars --rsi --gso-seed 100 --pool eval_unseen
```

做什么：从留出的 96 件（按名字哈希，训练实例永远抽不到）抽一个实例，后面的沉降、落地测试、lint、任务族同第 2 节；把学到的技能目录当种子（任务族文件的 `sandbox.seed` 指过去）建沙盒，直接跑 `sealed.py`。评测实例的 seed 从 100 起，`--pool all` 两个划分都抽。

## 9. 改阈值与判据

- 筛选阈值和排除的类别：`scripts/gso_catalog.py` 的参数（默认值在 `src/locomani/sim/gso_objects.py` 的 `Thresholds`，`--use-categories` 默认见上面第 1 节，类别名见 `categories.json` 的 `counts`）；改了重建目录，已建的实例记着它们用的 `catalog_sha256`，不受影响；要让已建实例换物体就删掉它的目录重建（关卡也要重建）。
- 放得进纸箱的规则：`gso_catalog.py --carton-inner 0.43 0.30 --carton-rim 0.33 --fit-margin 0.02`（`Thresholds.carton_inner_m` / `carton_rim_m` / `fit_margin_m`，目录版本 3，字段 `fits_carton` / `fit_why`）；纸箱的抱取带是 `prepare_logistics_scene.py` 的 `CARTON_PLACEMENT`（任务族从 `gso.carton_placement` 读，旧场景文件没有就用 `gso_family.CARTON_BAND`）。
- 任务判据：`scripts/rsi/gso_family.py` 的 `LIFT_M`（0.10 / 0.20 m）、`TILT_DEG`（60）、`POOL_SIZES`；改了重跑它，再改三份方案和 `scripts/rsi/rules.md`，算一次人工干预。
- 实例里物体多少、单双手比例、可达带深度：第 2 节的参数；`configs/scenes.yaml` 里记的是默认值。

## 10. 常见错误

- `prepare_logistics_scene.py` 报 `refusing to replace existing runtime scene`：实例目录已存在，删掉再建；预览脚本遇到已存在的实例直接复用。
- 报 `no catalog`：先做第 1 节。`gso_catalog.py` 报 `models have no entry in categories.json`：模型集和类别文件对不上，重跑 `scripts/gso_categories.py`（要联网）。`no spot on the surface` 出现在终端后接着 `drawing again` 是正常的重抽；20 组都放不下才退出，减小 `--objects`。
- 预览或渲染测试报 GL / EGL 错：第 0 节的 `MUJOCO_GL=egl` 要在 pytest 导入 mujoco 之前设好（放在命令行环境里，别在测试里设）。
- `scene_feasibility.py` 全部 `does not rest still in the sim`：少了落地测试那一步。
- Unreal 导出时 MuJoCo 编译报自由物体的质量和惯量为零：导出器版本太旧，剥碰撞壳之前没有烤惯量（`export_mjcf.bake_inertia`）。
- `gso_preview_unreal.py` 中途报 `zmq.error.Again`、之后 `no Unreal editor command server on :18801`：编辑器死了（11-05 建到 s5 的预览时一次），`bash scripts/start_unreal.sh --gpu 0` 再跑同一条命令，建好的关卡和图会跳过。
- `gso_preview_unreal.py` 报 `Unreal level missing` / `stale`：该实例没建关卡或场景重建后关卡没重建，加 `--build` 或手动 `build_scene.sh <实例> --unreal`；报 `PIE did not start` 或连不上编辑器：先 `bash scripts/start_unreal.sh --gpu 0`，且同一时间只能有一个进程用编辑器（建关卡时别渲染）。
- `gso_pack_check.py` 或一局报 `Error opening file 'gso/gso_0050/model_collision_0.obj'`：场景目录旁的 `<实例>_physics/` 是 11-05 之前写的旧副本（相对网格路径没改成绝对的，场景文件没更新它就不重写），删掉重跑；报 `FileNotFoundError: /workspace/models/scalebfm/…`：`LOCOMANI_WORKSPACE` 没设（第 0 节）。
- 起点采样打印 `seed N: skipped (no spot on A meets the placement option for nor_cal_box)`：纸箱的抱取带（离台沿 0.18–0.31 m、长边沿台沿）被别的物体占着，跳过这个 seed 是正常的，池也这样跳。
- 内环第一行就崩、`loop.log` 里找不到任务族或场景：沙盒看不到没提交的文件，先提交任务族文件；场景目录要在 `outputs/scene_discovery/` 下。
