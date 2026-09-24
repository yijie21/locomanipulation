# Blender 场景 → MuJoCo + Isaac

把一个 `.blend` 房间变成这条流水线能用的两套场景:**MuJoCo 管物理和策略看到的图像,Isaac 管离线重渲**。
两边不是同一份东西的两种格式,分工见 [§5](#5-为什么两边装的不一样)。

首次跑通的完整记录和实测数字在[台账](ledger/blender_scenes.md)。

## 已发布的新物流间纸箱场景

仓库包含[物流间场景资产包](../assets/scene/logistics_room_box_v1/README.md)：
`mujoco/scene_locomani.xml` 用于物理仿真，`isaac/logistics_room_box.usdc` 用于渲染。
房间/纸箱网格和贴图保存在资产包内；G1/Dex1 网格使用标准 SONIC 安装依赖的相对路径。
大文件用 Git LFS，XML/JSON 保持可读。
克隆后先执行 `git lfs pull`，再按资产包 README 的检查与运行准备步骤操作。
原始 `assets/blender/logistics_room.blend` 和 `NOR_CAL_Box_Refined_V2.blend` 同步保存；
场景包记录哈希、坐标、单位以及源文件资源重映射方法。
生成结果和任务运行仍写入 `outputs/`，不在资产目录录制或修改源文件。

### 纸箱与旧蓝篮的尺寸

以下顺序是物体自身坐标的长 × 宽 × 高，不是旋转后的世界坐标包围盒。

| 对象 | 当前场景使用的外尺寸 | 说明 |
|---|---|---|
| 新物流间 NOR-CAL 纸箱 | **45.0 × 32.0 × 33.0 cm** | MuJoCo 外碰撞范围与 Isaac 可见包围盒一致 |
| 旧双臂场景蓝色篮子 | **约 40.3 × 27.0 × 17.0 cm** | 当前资产在 x/y 缩小为原尺寸的约 2/3，高度不变 |
| 蓝篮原始未缩放资产 | 约 60.5 × 40.5 × 17.0 cm | 不是当前双臂 case 使用的尺寸 |

旧篮尺寸依据[模型定义](../assets/crates/blue_crate/model.xml)和
[场景配置](../src/locomani/sim/scene_lib.py)：实际视觉 OBJ 缩放后为
0.40342468 × 0.27009068 × 0.170294 m，简化碰撞体为 0.4026 × 0.2694 × 0.17 m。
新纸箱底面 z=0.371 m、顶沿 z=0.701 m；平台顶面高 0.371 m。
纸箱物理碰撞壁厚是 **0.010 m**：生成器中 `wall=0.005` 作为 MuJoCo 半尺寸使用，
历史记录里“5 mm 壁厚”的说法不能当作实际碰撞厚度。原历史保留，当前以编译模型为准。
尺寸/包围盒的检查证据见资产包 manifest；这不证明纸箱已能稳定抓起或搬运。

## 0. 一句话总览

```
.blend ──blender_to_usd.py──▶ <name>/<room>.usdc + textures/      （Isaac 离线重渲用）
.blend ──blender_to_mujoco.py──▶ <name>_mj/mesh/*.obj + room_parts.json   （贴图 PNG 从资产包拷入）
        ──build_mujoco_scene.py──▶ <name>_mj/scene_locomani.xml   （物理 + 策略图像用）
```

两条支路**互不依赖**，可以只做一条。仓库已经附带一套做好的（见上），新用户先用它验证环境，再换自己的 `.blend`。

## 1. 环境：需要什么、不需要什么

**不需要 conda，不需要仓库的任何 venv 装 `bpy`。** 两个转换脚本以 `blender --background --python` 方式跑在
**Blender 自带的 Python** 里；`bpy` 和 USD API 都是 Blender 自带的。三条支路各自的环境：

| 步骤 | 跑在哪个解释器里 | 怎么得到 | 验证 |
|---|---|---|---|
| `blender_to_usd.py` / `blender_to_mujoco.py` | Blender 5.x 自带 Python | 从 [blender.org](https://www.blender.org/download/lts/) 下 **5.x LTS** 官方 tar.xz，解开，把 `blender` 软链到 `PATH`（SRA0889 上是 `~/tools/blender-5.2.2-linux-x64`，软链 `~/tools/blender` 和 `~/.local/bin/blender`；`/opt` 下还残留 root 装的 4.2，`~/.local/bin` 在 PATH 里排前面所以不影响）。**不能**用 4.x：`NOR_CAL_Box_Refined_V2.blend` 是 Blender 5.2 存的，文件头是 5.0 起的新格式 `BLENDER17-`，4.x 直接报 `not a blend file`；USD 导出 API 也不同。换完软链后已开着的终端要 `hash -r`，否则 bash 仍用缓存的旧路径 | `blender --version` 打出 `Blender 5.x LTS` |
| `build_mujoco_scene.py`、`render_mujoco_frame.py`、场景校验 | 仓库的 `.venv_sim`（`$VENV_SIM_PY`） | [安装指南](setup.md) 第 0–5 步（`scripts/setup_sonic.sh`） | `"$VENV_SIM_PY" -c "import mujoco"` |
| 用 USD 真渲染一帧（可选） | 装了 `isaacsim` 的 render venv（`$RENDER_PY`） | [安装指南](setup.md) 第 7 步（SIMPLE，约 20 GB，**只有要 Isaac 像素才装**） | `"$RENDER_PY" -c "import isaacsim"` |
| 大文件 | Git LFS | `git lfs install && git lfs pull` | `assets/blender/*.blend` 不是 130 字节的指针文件 |

Blender 的 Python 要能 `import locomani`（脚本里用到仓库的工具函数）——靠环境变量，不靠安装：

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
blender --version
```

`uv`/`pip` 在这里一概用不上：**不要**试图把 `bpy` 装进 `.venv_sim` 或 `.venv` 去"统一环境"，
那是一个和本机 Blender 版本无关的 wheel，导出的 USD 会和 Isaac 这边对不上。

### 先用仓库自带的场景验证环境

```bash
git lfs pull
. scripts/paths.sh
"$VENV_SIM_PY" assets/scene/logistics_room_box_v1/check_assets.py --load-mujoco --load-usd
```

它校验两份资产树的哈希、拒绝绝对路径、量纸箱 OBJ、在 CPU 上编译 MJCF、用 Blender 自带的 USD API 打开 USDC。
通过说明 Blender、LFS、`.venv_sim` 三样都对了。

### 自己的 `.blend` 要满足的约定

- 放在 `assets/blender/`；贴图**打包进 .blend**（Blender：File → External Data → Pack Resources），
  这样文件是自包含的，LFS 只需要跟踪 `.blend`。仓库的房间 50 张贴图里 49 张是打包的，
  工作树里的 `assets/blender/textures/` 不是复现必需的。
- **公制、1 单位 = 1 米**、Z 朝上。`blender_to_usd.py` 会打印实际单位，不对就在 Blender 里改，不要在脚本里补缩放。
- 房间是布景：几千个物体共用几十个材质没关系（MuJoCo 支路按材质合并），但**工作面（桌/托盘）和目标物要是独立、命名清楚的物体**，
  后面 `build_mujoco_scene.py` 要按名字给它们加碰撞体。
- 相机和灯光可以放在 .blend 里，会一起进 USD；但灯光强度的换算见 §2。

## 2. 出 Isaac 场景

```bash
blender --background assets/blender/<room>.blend \
  --python scripts/blender_to_usd.py -- outputs/scene/<name>
```

产出 `outputs/scene/<name>/<room>.usdc` + `textures/`,**几何、材质、相机、灯光都带过来**。

它同时修掉 Blender USD 导出器的四处单位错误。每一处都是渲出来才发现的:

| 项 | 导出值 | 症状 |
|---|---|---|
| 相机 `focalLength`/`aperture` | 米(0.185 / 0.36) | 视场角算出来对,但 RTX 不认亚毫米焦距,**取景完全跑偏** |
| 灯光 `intensity` | 瓦特(20.69) | 画面几乎全黑,需要 ~1200 倍 |
| 曲线 `widths` | `2.0`,而舞台单位是米 | **一根地面走线渲染成 2 米宽的灰飘带** |
| `DomeLight` 环境贴图 | 一张纯色 EXR | RTX 报 `MirroredBall environment format not supported yet` 后**整个环境光失效**,背光的角落全黑 |

灯光强度**不写进 USD**,见 §4 的"已知未解决"。

## 3. 出 MuJoCo 场景

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
mj=outputs/scene/<name>_mj
blender --background assets/blender/<room>.blend \
  --python scripts/blender_to_mujoco.py -- "$mj"
cp -a assets/scene/logistics_room_box_v1/mujoco/textures "$mj"/          # MuJoCo 不读 JPG
cp assets/scene/logistics_room_box_v1/mujoco/texture_map.json "$mj"/     # JPG 名 → PNG 名
"$VENV_SIM_PY" scripts/build_mujoco_scene.py "$mj" \
  --base outputs/scene/crate_table30_depth04/scene_locomani.xml   # 机器人树从这里继承
```

第一步按**材质**(不是按物体)导出 OBJ —— MuJoCo 一个 geom 绑一个材质,而房间是几千个物体共用几十个材质。
纸箱同样按材质拆成 `mesh/nor_cal_00..08.obj`(它的 9 个材质里 6 个是实物照片),清单在 `carton_parts.json`。
它只写 OBJ 和这两份 JSON,**不产贴图**;PNG 贴图和 `texture_map.json` 是发布时转好、随资产包提供的,
第三步要读它们,所以先拷进来。换了新房间、贴图名对不上时,自己把 JPG 转成 PNG 并补 `texture_map.json`
(转法见资产包 README 的 "Regenerate from Blender")。

第三步在现成的 `scene_locomani.xml` 上做手术,生成新的场景。`--base` 必须是**已有的、带标定好机器人的场景**
(`crate_table30_depth04` 那份,在 `outputs/` 下、不进 git;manifest 记录的是发布时那份的哈希,本机的可能已经不同),
而不是正在生成的输出文件。脚本会先删掉自己上一次加的所有 asset、geom、相机和灯再重新添加,所以重复运行、
甚至误把输出当 `--base` 都得到同一份文件。它加的东西:

| 加了什么 | 为什么 |
|---|---|
| 每材质一个视觉 geom,`contype=0` | 房间只是布景,不参与碰撞 |
| 托盘/桌面一个 box 碰撞体 | 它就是工作面 |
| 目标物 5 块板(底 + 四壁) | 中空;凸包会变成实心块,手伸不进去 |
| 目标物 9 个视觉 geom,每材质一个 | 6 个面是实物照片贴图;MuJoCo 一个 geom 只绑一张贴图,整块导出就只能是纯色 |
| 地面 plane 保留碰撞、**关闭渲染** | plane 是平面碰撞的精确图元;它的棋盘贴图会和房间地板网格打架 |

**机器人树、关节默认值、执行器全部从 base 场景继承**,不重写 —— 重建机器人等于引入一次新标定。

生成的 XML 里相机有 `head_camera`、`left_wrist`、`right_wrist`(继承自 base)和 `reverse_room`(脚本加的,对应 Isaac 的 "02 Reverse room" 机位)。

## 4. 渲一帧并对比

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$VENV_SIM_PY" \
  scripts/render_mujoco_frame.py outputs/scene/<name>_mj/scene_locomani.xml \
  /tmp/mj.png --camera head_camera    # 也有 left_wrist / right_wrist / reverse_room

"$VENV_SIM_PY" scripts/compare_frames.py <isaac.png> /tmp/mj.png --out /tmp/pair.png
```

`MUJOCO_EGL_DEVICE_ID` 是 **EGL 设备表的下标,不是 CUDA 编号**:那张表里还有集显和 llvmpipe,
也不跟着 `CUDA_VISIBLE_DEVICES` 重排。单张 NVIDIA 卡的工作站上显卡是 0;填错会报
`Cannot initialize a EGL device display`(SRA0889 上 1、2 是没驱动的 Intel 集显,3 是 llvmpipe)。

`compare_frames.py` 报两个数,回答的是不同问题:

- **raw** —— 直接逐像素差,主要被曝光支配,而曝光只是个旋钮
- **structural** —— 对齐均值和方差之后的差,**旋钮修不掉的那部分**:阴影、全局光、噪点

判断标准是 structural。布置对不对看它,不看 raw。

### 已知未解决

**`blender_to_usd.py` 产出的 USD 单独渲染会偏暗且结构不符**(实测 structural 19.4%,而同一份内容在组合场景里是 4.2%)。
排除过:贴图文件、贴图路径、色彩空间、材质网络、相机、累积采样、资产流式加载、灯光强度写入时机、Hydra 结构性重同步 —— **九条都不是原因**。
所以灯光增益**不烘进 USD**,留给渲染时的调用方;仓库的数据集重渲走 `locomani.sim.isaac.isaac_render`,不受影响。
不要再从上面九条里找。

**运行时场景可能早于贴图提交**（2026-09-24 发现）。`configs/scenes.yaml` 的 `warehouse`（`outputs/box_rim_execution_20260920/scene_reverse_camera`）是 09-20 从 `outputs/scene/logistics_mujoco` 装的，
早于 `0bf8eed`（纸箱拆 9 块材质、6 张实物照片），所以那份场景里纸箱是一整块纯色网格。用 `scripts/prepare_logistics_scene.py`
从资产包重装即可（`warehouse_textured`）；场景名不改指向，重装的场景要起新名字。新实验一律用 `warehouse_textured`，
现行基线与跑法见 [warehouse](warehouse.md)。

**纸箱显示网格太重，拖慢 MuJoCo 仿真**（2026-09-23 实测）。`nor_cal_box` 的显示网格有 46.3 万面，加上场景里 2 盏投影灯
（shadowsize 4096），采集时 3 路相机每 tick 渲染 39–44 ms，warehouse 的 sim 只剩 0.30–0.36× 实时，机器人走得比指令快约 3 倍。
离线测过：隐藏纸箱显示网格后 2.0 ms，关阴影后 11 ms。目前的绕法是 [`backend.render: offline`](rendering.md#离线渲染backendrender-offline)，
但 VLA 闭环评测和数据集录制必须实时渲染，所以网格减面（目标约 5k 面）或降阴影还得做。
减面的源目录 `assets/crates/nor_cal_box/visual/` 尚未入库，改之前先和维护者对齐。
证据见 [warehouse 渲染负担交接](handoff/warehouse-render-cost-20260923.md)。

## 5. 为什么两边装的不一样

**MuJoCo 跑物理,并且生成策略训练用的图像;Isaac 只做采集完成后的离线重渲。**
所以房间在 Isaac 里是纯视觉背景,在 MuJoCo 里也是纯视觉 —— 但**碰撞只有地面、工作面和目标物**。

489k 三角面当碰撞体撑不住 50 Hz 的全身控制器,而且任务本来也碰不到墙和货架。
这不是偷懒,是仓库既有场景的一贯做法:现有的每个 `scene_locomani.xml` 都只有**一个 plane,一堵墙都没有**。

## 6. 接进采集之前还差什么

场景能渲不等于能采。专家读的是 `scene_meta.json`:

- `surfaces.A` —— 工作面的 `top`(高度)、`half_extent`、`xy`
- `objects.<名字>` —— `init_pos`、`height`、`mesh`、`surface`

没有这两块,场景搭好了专家也不知道在哪儿干活。字段含义见 [双臂详解](bimanual_design.md)。

在这个房间里采集（场景怎么建、基线配置、离线渲染、补渲数据集相机）见 [warehouse](warehouse.md)。

## 7. 常见错误对照

| 症状 | 原因 | 处理 |
|---|---|---|
| `blender: command not found` / 打出 4.x | 没装 5.x 或软链丢了（容器 recycle） | 重做 §1 的软链 |
| 换了软链，同一个终端里 `blender` 仍打出 4.2 | bash 缓存了上次解析到的路径 | `hash -r` 或开新终端 |
| 读房间 .blend 只警告 `newer Blender binary`，append 纸箱时报 `not a blend file` | 用的是 4.x：房间是 4.5 旧文件头还能读，纸箱是 5.2 新文件头读不了 | 装 5.x LTS（§1）；文件本身没坏 |
| `ModuleNotFoundError: locomani` | Blender 的 Python 看不到仓库 | `export PYTHONPATH="$LOCOMANI_ROOT/src"` 后再跑 |
| `.blend` 只有 130 字节 / 打开报损坏 | LFS 指针没拉 | `git lfs pull` |
| 贴图缺失（粉色/黑色面） | .blend 引用了作者机器上的路径 | 在 Blender 里 Pack Resources 后重存；仓库房间的修复见资产包 `rebuild_blender_scene.py` |
| Isaac 里画面全黑 / 取景跑偏 / 2 米宽灰带 | Blender USD 导出器的单位错误 | `blender_to_usd.py` 已修四处（§2 表）；灯光增益在渲染时给，不写进 USD |
| `build_mujoco_scene.py` 报 `FileNotFoundError: .../<name>_mj/scene_locomani.xml` | `--base` 指向了还没生成的输出文件 | `--base outputs/scene/crate_table30_depth04/scene_locomani.xml`（§3） |
| `build_mujoco_scene.py` 报 `texture_map.json` 不存在 | 导出脚本不产贴图 | 按 §3 从资产包拷入 `textures/` 和 `texture_map.json` |
| 加载 XML 报 `repeated name 'room_00' in mesh` | 旧版构建脚本在自己的输出上重复叠加 | 脚本已改为可重复运行；用正确的 `--base` 重新生成一次 |
| MuJoCo 报贴图读不了 | MuJoCo 不读 JPG | 只用资产包里的 PNG；手改场景时也只用 PNG |
| 渲染里纸箱是一块纯色，货架上的箱子却有纹理 | 2026-09-22 之前纸箱整块导出、材质写死牛皮纸色 | 用当前脚本重新导出并构建；`carton_parts.json` 应有 9 份、其中 6 份带 `Figure*.jpeg` |
| warehouse 里机器人走得太快、过冲撞箱，同样的配置在 walk 场景正常 | 纸箱 46.3 万面显示网格 + 投影灯让渲染拖慢 sim 到 0.35× 实时 | 算 sim 实时比：trace 行数 × 0.02 s ÷ manifest 的 `seconds`，低于 0.9 就是渲染拖慢；用 `backend.render: offline`（上面“已知未解决”） |
| `Cannot initialize a EGL device display` | `MUJOCO_EGL_DEVICE_ID` 指到了集显或不存在的 EGL 设备 | 单卡机器用 0（§4）；它是 EGL 下标，不是 CUDA 编号 |
| `render: no camera named head` | 相机名写错 | 用 `head_camera` / `left_wrist` / `right_wrist` / `reverse_room`，报错信息会列出全部 |
| 编译 MJCF 时 mesh 找不到 | `meshdir` 是绝对路径或指向别的机器 | 用 `prepare_logistics_scene.py` 生成运行副本；G1/Dex1 网格相对 `third_party/GR00T-WholeBodyControl` |
| 机器人在新场景里行为变了 | 重写了机器人树 | 永远 `--base` 继承现成的 `scene_locomani.xml`，只动布景和工作面 |

## 8. 进一步阅读

- [Blender 场景台账](ledger/blender_scenes.md):逐次实测数字、四个单位错误的发现过程、未解决项。
- [MuJoCo 仿真与 Isaac 渲染](rendering.md):既有的重渲链路与视频导出。
- [双臂](bimanual.md):这条线要对接的采集流程。
