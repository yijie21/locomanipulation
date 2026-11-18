# Google Scanned Objects 的筛选：程序规则加人工挑选

物流间场景用的扫描物体来自 Google 的 Scanned Objects 数据集，共 1030 件日用品，用的是 kevinzakka 转换的 MuJoCo 版。这些物体不能直接都放进场景：有鞋、玩具、扁平的东西、站不稳的东西，还有形状怪异、机器人很难抓的东西。

筛选分两步，结果都写进同一个目录文件 `assets/mujoco_scanned_objects/catalog.json`：
1. **程序筛选**（第 2 节）：按数据集的类别标签、尺寸、静置测试，把每件物体判成 `single`（单手抓）、`bimanual`（双手抱）或 `none`（不用）。
2. **人工筛选**（第 3 节）：在网页上逐件看图，把程序没筛掉的怪形状物体排除掉，或者改掉程序判错的种类，存进 `assets/mujoco_scanned_objects/hand_choices.json`；重建目录时人工选择压过程序规则。

建场景只读 `catalog.json`，不读 2 GB 的网格。各个场景怎么从目录里取物体见第 5 节。

| 阶段 | 单手 | 双手 | 不用 |
|---|---|---|---|
| 只按形状规则（不看类别） | 407 | 271 | 352 |
| 程序筛选（类别加形状） | 270 | 91 | 669 |
| 加上人工选择（2026-11-17，现在的目录） | 283 | 108 | 639 |

现在的目录里，单手 283 件中训练划分 261 件、留出 22 件，放得进纸箱的 276 件；双手 108 件中训练划分 103 件、留出 5 件，放得进纸箱的 79 件。

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
PY=$VENV_SIM_PY
```

和[物流间 1030 runbook](rsi_logistic_room_1030.md) 第 0 节一样；`MUJOCO_GL` 是给图库的离屏渲染用的。

## 1. 装网格、取类别标签（每台机器一次）

```bash
bash scripts/install_gso.sh
python3 scripts/gso_categories.py      # 只在要刷新类别标签时；categories.json 已在仓库里
```

做什么：
- 第一条把 1030 件物体的网格装到 `assets/mujoco_scanned_objects/models/`（约 2 GB，git 忽略）。已经装好就什么都不做；`--from <目录>` 用已经下载好的副本。
- 第二条从数据集在 Gazebo Fuel 上的条目取每件物体的官方类别，写成 `categories.json`（进仓库）。kevinzakka 的 MuJoCo 版没带这些标签。

得到什么：每件物体一个目录 `models/<名字>/`，里面有扫描网格 `model.obj`、贴图 `texture.png` 和最多 32 个 V-HACD 凸壳 `model_collision_*.obj`。类别分布：Shoe 254、Consumer Goods 248、Toys 150、Bottles and Cans and Cups 53、Bag 28、Media Cases 21、Action Figures 17、Board Games 17、Legos 10，其余几类各 1–4 件，216 件没有标签。

## 2. 程序筛选

```bash
$PY scripts/gso_catalog.py
```

做什么：对每件物体算尺寸、凸壳体积、质量，做一次静置测试，再按下面的规则从上到下判，第一条不满足的规则就是它被判成 `none` 的原因，写在记录的 `why` 里。规则和阈值在 `src/locomani/sim/gso_objects.py` 的 `graspable` 和 `Thresholds`，每个阈值都是 `gso_catalog.py` 的参数（`--help`），运行时的值写进目录的 `thresholds`。

| 顺序 | 规则 | 阈值 | 按这条判成 `none` 的件数 |
|---|---|---|---|
| 1 | 类别要在用的类别里 | 不用 Shoe、Toys、Stuffed Toys、Mouse、Media Cases、Legos、Keyboard、Car Seat、Action Figures、Hat（用户 11-04 / 05）；没有标签的照用 | 463 |
| 2 | 放在平面上站得住 | 掉到平面上看 24 s，倾斜不超过 3°、滑动不超过 5 mm | 11 |
| 3 | 不能太扁 | 高度至少 3 cm | 37 |
| 4 | 单手：窄边放得进张开的夹爪，抓点不能太低 | 窄边不超过 9.5 cm；抓点（台面 0.371 m 加半个高度）至少 0.40 m | 11 |
| 5 | 双手：宽边在两手能抱的范围，够高 | 宽边 0.20–0.45 m，高度至少 0.10 m | 147（两种都不是） |

满足第 4 条的是 `single`，满足第 5 条的是 `bimanual`。另外三项不影响种类，但场景会用：
- **质量**：凸壳体积 × 300 kg/m³，截到 0.05–0.3 kg（上限用户 11-07 定）。物体的质量按凸壳分布，所以重心和惯量跟着扫描形状走。
- **放得进纸箱**（`fits_carton`）：占地每边留 2 cm 后放得进纸箱的内空 0.43 × 0.30 m，高度不超过箱沿 0.33 m。
- **训练 / 留出划分**（`split`）：按名字的哈希留出 10 %，留给“没见过的物体”的评测；改阈值不会改变划分。

得到什么：`catalog.json`，终端打印 `counts`、不用的类别、阈值和秒数，约 10 秒。每件物体的字段说明在 `assets/mujoco_scanned_objects/README.md`。

程序规则只看包围盒和能不能站住，看不出形状：动物摆件、带把手的工具、开口的筐和花盆、半个人偶，只要尺寸合适都会被判成可抓。这就是要人工再筛一遍的原因。

## 3. 人工筛选

```bash
$PY scripts/gso_gallery.py --serve       # 打开 http://127.0.0.1:8765/index.html；远程机器经 VSCode 的端口转发打开
```

做什么：先用 MuJoCo 离屏把每件物体单独画一张图（同一个斜上方视角，按包围盒取景，第一次约 1 分钟，之后只画缺的），再在本机起一个小服务，网页列出全部 1030 件。每张卡片上有图、名字、类别、尺寸、质量、规则给的种类和原因，以及两项可以人工改的选择：
- **use**（排除 / 保留）：先按类别规则，点按钮翻转。把被排除类别的物体点成保留，就是越过类别规则放回来。
- **shape**（种类）：先按形状规则（规则给的那个按钮下面标着 `rule`），点 Single / Bimanual / None 改。

物体保留时按它的 shape 种类进场景，排除时就是 `none`。卡片上还有两个形状分，帮忙找怪形状：
- `fill`：凸包体积除以主轴包围盒体积，越低越不像盒子（动物、人偶、工具偏低）。
- `solid`：V-HACD 凸壳体积除以凸包体积，越低越空心（筐、花盆、收纳盒这类开口容器偏低）。

排序可以选“最不像盒子”“最空心”，从最可疑的看起。其他操作：
- 鼠标停在卡片上按键：1 / 2 / 3 改 shape，0 让 shape 回到规则，x 翻转排除。
- 原因取页头的下拉框：不规则形状、玩具或人偶、开口容器、难抓、扫描坏了、目测确认，或者自己写。
- 页头的批量按钮把当前筛出的卡片一起设，会先确认。
- 筛选：现在排除 / 保留的、能进场景的、人工设过的、放得进 / 放不进纸箱的、类别排除的。

每次点击立即写进 `assets/mujoco_scanned_objects/hand_choices.json`（进仓库），格式是 `{"objects": {模型名: {"excluded": true 或 false, "exclude_why": 原因, "shape": "single" 或 "bimanual" 或 "none", "shape_why": 原因}}}`，两项可以只有一项。人工的选择即使和规则一样也记下，这样以后规则改了也不会丢；只有点“回到规则”才删掉。不加 `--serve`、直接打开 `outputs/gso_gallery/index.html` 时，选择只存在浏览器里，用页头的下载按钮存出同一个文件放到那里。

得到什么：`hand_choices.json`。2026-11-17 用户点完的结果共 269 件：
- 排除 156 件，原因都是不规则形状；放回 1 件（被类别规则排除的 Thor 人偶）。
- 改 shape 113 件（有的和排除记在同一件上）：

| 规则给的 | 人工改成 | 件数 |
|---|---|---|
| none | single | 55 |
| none | bimanual | 24 |
| single | bimanual | 24 |
| bimanual | single | 2 |
| bimanual | none | 1 |
| 和规则一样，仍记下 | | 7 |

## 4. 重建目录，让人工选择生效

```bash
$PY scripts/gso_catalog.py
```

做什么：和第 2 节同一条命令。目录读 `hand_choices.json`（`--hand-choices` 换文件），判每件物体时：
- `excluded` 压过类别规则：排除的记 `none`，`why` 写原因；保留的不再看类别。
- `shape` 压过形状规则（静置测试、尺寸）：直接用人工给的种类，`why` 写原因。
- 每件记录多一个 `by_hand` 字段，`counts.by_hand` 统计排除、保留和各种类的件数。
- 名单里有不是模型名的、或者格式不对，建目录报错停下。

得到什么：`counts` 是 `single 283, bimanual 108, none 639`。从程序筛选到现在：程序判单手的 270 件里 226 件仍是单手、24 件改成双手、20 件被排除；程序判双手的 91 件里 59 件仍是双手、2 件改成单手、30 件被排除；程序判 `none` 的有 55 件改成单手、25 件改成双手。

重建只改目录。已经建好的场景记着它当时用的目录哈希（`catalog_sha256`），物体不会跟着变；要换物体就删掉场景目录重建。

## 5. 场景怎么用这份目录

- **随机抽取的实例** `logistics_room_1030_s<seed>`：每个实例按种子抽 10 件（默认 6 件单手、4 件双手），只抽训练划分、放得进纸箱的。见[物流间 1030 runbook](rsi_logistic_room_1030.md)。
- **人工挑选的两个测试场景** `logistics_room_gso_single`（20 件单手，无纸箱）和 `logistics_room_gso_bimanual`（7 件双手加纸箱）：在 `gso_gallery.py --pick --serve` 的挑选页上从目录里这一种类、训练划分的物体中挑，不要求放得进纸箱，存进 `assets/mujoco_scanned_objects/scene_picks.json`。见[单手 / 双手测试场景 runbook](rsi_logistic_room_1030_single_bimanual.md)。

两种场景都会把选中物体的网格复制到场景目录里，场景文件的 `gso` 块记着目录和挑选文件的哈希。

## 6. 改规则、重新筛

- **改阈值或类别**：改 `gso_catalog.py` 的参数（`--jaw`、`--bimanual-width`、`--min-height`、`--use-categories`、`--exclude-categories` 等），或者改 `Thresholds` 的默认值，再重建目录。人工选择不受影响，它压过新规则的结果。
- **改人工选择**：回第 3 节的网页改，再重建目录。想让某件回到程序规则，点“回到规则”。
- **看某件为什么是这个种类**：网页上点卡片右上角的 ⓘ，或者在 `catalog.json` 里看它的 `why` 和 `by_hand`。

## 7. 文件一览

| 文件 | 进仓库 | 谁写 | 是什么 |
|---|---|---|---|
| `assets/mujoco_scanned_objects/models/` | 否 | `scripts/install_gso.sh` | 1030 件物体的网格和贴图 |
| `assets/mujoco_scanned_objects/categories.json` | 是 | `scripts/gso_categories.py` | 每件物体的数据集类别 |
| `assets/mujoco_scanned_objects/hand_choices.json` | 是 | `scripts/gso_gallery.py --serve` | 人工排除 / 保留和人工定的种类 |
| `assets/mujoco_scanned_objects/catalog.json` | 是 | `scripts/gso_catalog.py` | 目录：每件物体的尺寸、质量、种类和原因、划分、放不放得进纸箱 |
| `assets/mujoco_scanned_objects/scene_picks.json` | 是 | `scripts/gso_gallery.py --pick --serve` | 两个测试场景挑的物体 |
| `outputs/gso_gallery/` | 否 | `scripts/gso_gallery.py` | 图库网页、每件物体的图、形状分 |

代码：规则和读写在 `src/locomani/sim/gso_objects.py`，测试在 `tests/test_gso_catalog.py`、`tests/test_gso_gallery.py`、`tests/test_gso_scene_pick.py`。
